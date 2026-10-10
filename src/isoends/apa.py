"""Tests of site usage: a gene's two most used sites (tandem 3'UTR sites, or two promoters) and classes of sites
against the reference sites.

Within a unit, a paired design is tested with the Cochran-Mantel-Haenszel test on the event x condition tables of
each replicate pair, and an unpaired design with the moderated two-sample t-test on each sample's usage. Across
units, the per-unit change is tested with the moderated one-sample t-test. q is over the genes tested.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .sites import INTRONIC, as_mask, distance_from_stop, same_exon, split_by_model_intron
from .stats import bh, cmh, mantel_haenszel_or, moderated_one_sample, moderated_two_sample

APA_CLASSES = {"composite exon": ("composite exon",), "skipped exon": ("skipped exon",),
               "internal exonic": ("internal exonic",)}
IPA = {"intronic (all)": INTRONIC}


def _site_counts(sites: pd.DataFrame, counts: pd.DataFrame, datasets) -> pd.DataFrame:
    """The clusters with one read-count column per dataset and their total (reads_unit); clusters without reads in
    these datasets are dropped."""
    w = counts[counts.dataset.isin(datasets)].pivot_table(index="cluster", columns="dataset", values="reads",
                                                          aggfunc="sum", fill_value=0)
    s = sites.set_index("cluster").join(w.reindex(columns=list(datasets), fill_value=0), how="inner")
    s["reads_unit"] = s[list(datasets)].sum(axis=1)
    return s[s.reads_unit > 0]


def _ratio(x, y):
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(y > 0, x / np.maximum(y, 1), np.nan)


def usage_test(event: pd.DataFrame, other: pd.DataFrame, design, params, stats_params, stem: str,
               up: str) -> pd.DataFrame:
    """Test the usage event / (event + other) between the conditions of one unit, one row per feature.

    event, other: read counts, features x datasets. Returns reads and usage per condition, delta_<stem>, the test's
    statistic and p, and the replicates behind it; per pair for a paired design (n_pairs_<up>, delta_<stem>_pair<k>,
    mh_odds_ratio, chi2), per sample for an unpaired one (t, n_samples_control, n_samples_treated). The unpaired
    test takes the samples with `min_reads_sample` reads, for features with event and other reads and two such
    samples per condition; when the robust variance prior cannot be estimated the non-robust one is used."""
    ctrl, trt = list(design.control), list(design.treated)
    a, b = event[ctrl].to_numpy(float), other[ctrl].to_numpy(float)
    c, d = event[trt].to_numpy(float), other[trt].to_numpy(float)
    rc, rt = (a + b).sum(axis=1), (c + d).sum(axis=1)
    out = pd.DataFrame({"event_reads_control": a.sum(axis=1), "other_reads_control": b.sum(axis=1),
                        "event_reads_treated": c.sum(axis=1), "other_reads_treated": d.sum(axis=1),
                        "reads_control": rc, "reads_treated": rt,
                        f"{stem}_control": _ratio(a.sum(axis=1), rc), f"{stem}_treated": _ratio(c.sum(axis=1), rt)},
                       index=event.index)
    out[f"delta_{stem}"] = out[f"{stem}_treated"] - out[f"{stem}_control"]
    out["design"] = "paired" if design.paired else "unpaired"
    if design.paired:
        per_pair = _ratio(c, c + d) - _ratio(a, a + b)
        out["n_pairs"] = len(design.pairs)
        out[f"n_pairs_{up}"] = (per_pair > 0).sum(axis=1)
        out["pairs"] = ";".join(label for label, _, _ in design.pairs)
        for k in range(len(design.pairs)):
            out[f"delta_{stem}_pair{k + 1}"] = per_pair[:, k]
        out["mh_odds_ratio"] = mantel_haenszel_or(a, b, c, d)
        out["chi2"], out["p"] = cmh(a, b, c, d)
        return out
    uc = pd.DataFrame(np.where(a + b >= params.min_reads_sample, _ratio(a, a + b), np.nan), index=event.index)
    ut = pd.DataFrame(np.where(c + d >= params.min_reads_sample, _ratio(c, c + d), np.nan), index=event.index)
    out["n_samples_control"], out["n_samples_treated"] = uc.notna().sum(axis=1), ut.notna().sum(axis=1)
    testable = ((out.n_samples_control >= 2) & (out.n_samples_treated >= 2)
                & (out.event_reads_control + out.event_reads_treated > 0)
                & (out.other_reads_control + out.other_reads_treated > 0))
    kw = dict(backend=stats_params.limma_backend, rscript=stats_params.rscript)
    try:
        res = moderated_two_sample(uc[testable], ut[testable], robust=stats_params.robust, **kw)
    except (ValueError, RuntimeError):
        res = moderated_two_sample(uc[testable], ut[testable], robust=False, **kw)
    out["t"], out["p"] = res.t.reindex(out.index), res.p.reindex(out.index)
    return out


def top_two_within_unit(unit: str, sites: pd.DataFrame, counts: pd.DataFrame, design, params, stats_params,
                        candidates: pd.Series, gene_names: pd.Series | None = None) -> pd.DataFrame:
    """Per gene, its two most used candidate sites in the unit (equally used sites: the lower coordinate first) and
    the test of distal usage between the conditions.

    sites: the run's classified clusters; counts: cluster, dataset, reads; candidates: boolean per cluster (indexed
    by cluster). PDUI = distal / (proximal + distal) reads. Genes with fewer than `min_reads` reads at the two sites
    in a condition, or without a test, are dropped."""
    s = _site_counts(sites, counts, design.datasets)
    s = s[candidates.reindex(s.index).fillna(False).to_numpy(bool)].copy()
    s["order"] = np.where(s.strand == "+", s.site, -s.site)
    n_sites = s.groupby("gene_id").size()
    s = s[s.gene_id.isin(n_sites.index[n_sites >= 2])].reset_index()
    if s.empty:
        return pd.DataFrame()
    two = (s.sort_values(["gene_id", "reads_unit", "site"], ascending=[True, False, True], kind="stable")
           .groupby("gene_id").head(2).sort_values(["gene_id", "order"], kind="stable"))
    prox = two.groupby("gene_id").head(1).set_index("gene_id")
    dist = two.groupby("gene_id").tail(1).set_index("gene_id").loc[prox.index]
    test = usage_test(dist[design.datasets], prox[design.datasets], design, params, stats_params, "pdui", "distal")
    test = test.rename(columns={"event_reads_control": "reads_distal_control",
                                "other_reads_control": "reads_proximal_control",
                                "event_reads_treated": "reads_distal_treated",
                                "other_reads_treated": "reads_proximal_treated",
                                "mh_odds_ratio": "mh_odds_ratio_distal"})
    out = pd.DataFrame({"unit": unit, "gene_id": prox.index, "chrom": prox.chrom.to_numpy(),
                        "strand": prox.strand.to_numpy(), "n_utr_sites": n_sites.reindex(prox.index).to_numpy(),
                        "proximal_cluster": prox.cluster.to_numpy(), "distal_cluster": dist.cluster.to_numpy(),
                        "proximal_site": prox.site.to_numpy(), "distal_site": dist.site.to_numpy(),
                        "proximal_hexamer": prox.hexamer.to_numpy(), "distal_hexamer": dist.hexamer.to_numpy()})
    out = pd.concat([out, test.reset_index(drop=True)], axis=1)
    out["utr_span_nt"] = (out.distal_site - out.proximal_site).abs()
    out["length_gained_nt"] = out.delta_pdui * out.utr_span_nt
    if gene_names is not None:
        out.insert(2, "gene_name", out.gene_id.map(gene_names).fillna(""))
    ok = (out.reads_control >= params.min_reads) & (out.reads_treated >= params.min_reads) & out.p.notna()
    return out[ok].reset_index(drop=True)


def tandem_candidates(sites: pd.DataFrame, params) -> pd.Series:
    """The clusters a tandem 3'UTR pair is chosen from, as a boolean indexed by cluster."""
    if params.tandem_sites == "terminal":
        chosen = sites.site_class == "terminal"
    elif params.tandem_sites == "three_prime_utr":
        chosen = sites.feature == "three_prime_utr"
    else:
        raise ValueError(f"tandem_sites must be 'terminal' or 'three_prime_utr', not {params.tandem_sites!r}")
    return pd.Series(chosen.to_numpy(), index=sites.cluster)


def annotate_tandem(t: pd.DataFrame, ref: dict, model_exons: pd.DataFrame, terminal: pd.DataFrame,
                    params) -> pd.DataFrame:
    """Add the same-exon check (same_exon, share_distal_models_split, exclusion_reason, is_tandem), q over the
    unit's tandem genes and each site's distance from the stop codon."""
    if t.empty:
        return t
    t = t.copy()
    q = t.rename(columns={"proximal_site": "proximal", "distal_site": "distal"})
    t["same_exon"] = same_exon(q, ref["exons"], params.downstream_nt).array
    t["share_distal_models_split"] = split_by_model_intron(q, model_exons, terminal, params.model_window_nt).to_numpy()
    reason = np.select([as_mask(t.same_exon.eq(False)), as_mask(t.share_distal_models_split > 0.5)],
                       ["no annotated exon holds both sites", "the distal-site models splice over the proximal site"],
                       default="")
    t["exclusion_reason"] = reason if params.same_exon_check else ""
    t["is_tandem"] = t.exclusion_reason == ""
    t["q"] = np.nan
    t.loc[t.is_tandem, "q"] = bh(t.loc[t.is_tandem, "p"])
    for side in ("proximal", "distal"):
        t[f"{side}_from_stop_nt"] = distance_from_stop(t.gene_id, t[f"{side}_site"], ref["canonical"], ref["stop"],
                                                       params.downstream_nt)
    return t


def site_models(rows: pd.DataFrame, site_cols, model_reads: pd.DataFrame, terminal: pd.DataFrame,
                window_nt: int = 100) -> pd.DataFrame:
    """For each row and site column: the best-supported model of the gene ending within `window_nt` of the site,
    with its reads per condition. model_reads: gene, transcript, reads_control, reads_treated."""
    out = pd.DataFrame(index=rows.index)
    mr = model_reads.merge(terminal[["transcript_id", "site"]].rename(columns={"transcript_id": "transcript",
                                                                                "site": "end3"}), on="transcript")
    mr["total"] = mr.reads_control + mr.reads_treated
    for col in site_cols:
        q = rows[["gene_id", col]].rename(columns={col: "pos"}).assign(row=rows.index)
        m = q.merge(mr, left_on="gene_id", right_on="gene")
        m = m[(m.end3 - m.pos).abs() <= window_nt]
        best = m.sort_values(["row", "total", "transcript"], ascending=[True, False, True]).drop_duplicates("row")
        best = best.set_index("row")
        stem = col.replace("_site", "")
        out[f"{stem}_model"] = best.transcript.reindex(rows.index)
        out[f"{stem}_model_reads_control"] = best.reads_control.reindex(rows.index)
        out[f"{stem}_model_reads_treated"] = best.reads_treated.reindex(rows.index)
    return out


def match_pairs(t: pd.DataFrame, tol: int = 50, valid: str | None = None) -> pd.DataFrame:
    """Across units: each gene's reference proximal / distal pair, the pair matched within `tol` nt at both ends by
    the most units (ties: more reads), and same_pair for the units whose pair matches it. With `valid`, the
    reference is chosen among the rows where that column is true."""
    t = t.copy()
    t["reference_proximal"] = np.nan
    t["reference_distal"] = np.nan
    t["same_pair"] = False
    reads = (t.reads_control + t.reads_treated).to_numpy()
    usable = t[valid].to_numpy(bool) if valid else np.ones(len(t), bool)
    position = pd.Series(np.arange(len(t)), index=t.index)
    for _, idx in t.groupby("gene_id").groups.items():
        rows = position[idx].to_numpy()
        p, d = t.proximal_site.to_numpy()[rows], t.distal_site.to_numpy()[rows]
        ok = usable[rows] if usable[rows].any() else np.ones(len(rows), bool)
        m = (np.abs(p[:, None] - p[None, :]) <= tol) & (np.abs(d[:, None] - d[None, :]) <= tol)
        score = np.where(ok, (m & ok[None, :]).sum(axis=1), -1)
        best = np.lexsort((-reads[rows], -score))[0]
        t.loc[idx, "reference_proximal"] = p[best]
        t.loc[idx, "reference_distal"] = d[best]
        t.loc[idx, "same_pair"] = m[best]
    return t


def across_units(table: pd.DataFrame, value: str, min_units: int, stats_params, up_label: str, down_label: str,
                 extra: dict | None = None) -> pd.DataFrame:
    """Moderated one-sample t-test of one value per gene and unit over the units, for genes measured in at least
    `min_units` units. table: gene_id, unit, `value` and optionally q_within. Returns per gene: mean_<value>, t, p,
    q, n_units, median_<value>, the units moving each way and n_units_q_within."""
    if table.empty:
        return pd.DataFrame()
    X = table.pivot_table(index="gene_id", columns="unit", values=value, aggfunc="first")
    X = X[X.notna().sum(axis=1) >= min_units]
    if X.empty:
        return pd.DataFrame()
    res = moderated_one_sample(X, robust=stats_params.robust, backend=stats_params.limma_backend,
                               rscript=stats_params.rscript)
    estimator = res.attrs.get("estimator", "")
    res = res.rename(columns={"mean": f"mean_{value}"})
    X = X.reindex(res.index)
    res[f"median_{value}"] = X.median(axis=1)
    res[f"n_units_{up_label}"] = (X > 0).sum(axis=1)
    res[f"n_units_{down_label}"] = (X < 0).sum(axis=1)
    if "q_within" in table:
        sig = table[table.q_within < stats_params.alpha].groupby("gene_id").size()
        res["n_units_q_within"] = sig.reindex(res.index).fillna(0).astype(int)
    for k, v in (extra or {}).items():
        res[k] = v.reindex(res.index)
    res = res.rename_axis("gene_id").reset_index()
    res.attrs["estimator"] = estimator
    return res


def class_within_unit(unit: str, sites: pd.DataFrame, counts: pd.DataFrame, design, label: str, in_class: pd.Series,
                      params, stats_params, gene_names: pd.Series | None = None, reference: str = "terminal",
                      class_column: str = "apa_class") -> pd.DataFrame:
    """Per gene: reads at the sites of one class against reads at the `reference` sites, tested between the
    conditions of the unit; q over the genes tested. in_class: boolean per cluster (indexed by cluster)."""
    s = _site_counts(sites, counts, design.datasets)
    chosen = in_class.reindex(s.index).fillna(False).to_numpy(bool)
    s["side"] = np.where(chosen, "class", np.where(s.site_class == reference, "reference", ""))
    s = s[s.side != ""]
    if s.empty:
        return pd.DataFrame()
    by = s.groupby(["gene_id", "side"])[design.datasets].sum()
    genes = by.index.get_level_values(0).unique()
    by = by.reindex(pd.MultiIndex.from_product([genes, ["class", "reference"]])).fillna(0)
    test = usage_test(by.xs("class", level=1), by.xs("reference", level=1), design, params, stats_params, "share",
                      "up")
    test = test.rename(columns={"event_reads_control": "class_reads_control",
                                "other_reads_control": "reference_reads_control",
                                "event_reads_treated": "class_reads_treated",
                                "other_reads_treated": "reference_reads_treated",
                                "mh_odds_ratio": "mh_odds_ratio_class"})
    cls = s[s.side == "class"].reset_index()
    main = (cls.sort_values(["gene_id", "reads_unit", "site"], ascending=[True, False, True])
            .drop_duplicates("gene_id").set_index("gene_id").reindex(genes))
    out = pd.DataFrame({"unit": unit, "gene_id": genes, class_column: label})
    out = pd.concat([out, test.reset_index(drop=True)], axis=1)
    out["class_reads"] = out.class_reads_control + out.class_reads_treated
    out["n_class_sites"] = cls.groupby("gene_id").size().reindex(genes).fillna(0).astype(int).to_numpy()
    out["main_site"] = main.site.to_numpy()
    out["main_site_class"] = main.site_class.to_numpy()
    out["main_site_reads"] = main.reads_unit.to_numpy()
    out["main_site_intron"] = main.intron.to_numpy() if "intron" in main else np.nan
    out["main_site_exon_rank"] = main.exon_rank.to_numpy() if "exon_rank" in main else np.nan
    if gene_names is not None:
        out.insert(2, "gene_name", out.gene_id.map(gene_names).fillna(""))
    ok = ((out.reads_control >= params.min_reads) & (out.reads_treated >= params.min_reads) & (out.class_reads > 0)
          & out.p.notna())
    out = out[ok].reset_index(drop=True)
    out["q"] = bh(out.p)
    return out

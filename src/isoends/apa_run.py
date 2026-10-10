"""The APA stage: cluster and classify each run's poly(A) sites, test every unit, then test across units. Also the
tails of reads by site, poly(A) signal usage and transcript-structure usage."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .apa import (APA_CLASSES, IPA, across_units, annotate_tandem, class_within_unit, match_pairs, site_models,
                  tandem_candidates, top_two_within_unit)
from .hist import grouped_quantiles
from .sites import classify_sites, cluster_sites, reference_models, site_region
from .stats import bh, wilcoxon_paired

PAS_MAIN = ("AATAAA", "ATTAAA")
FIVE_UTR = "5'UTR"
NON_TERMINAL = ("composite exon", "skipped exon", "intronic, no model", "internal exonic")
USAGE_TEST = {"paired": "Cochran-Mantel-Haenszel test over replicate pairs (continuity-corrected, 1 df)",
              "unpaired": "moderated two-sample t-test of per-sample usage over replicates"}


def usage_test_name(designs) -> str:
    """The within-unit usage test of the designs present."""
    present = sorted(set(designs))
    return "; ".join(USAGE_TEST[d] + (f" ({d} units)" if len(present) > 1 else "") for d in present)


def read_count(params) -> str:
    """The model_reads column counted: supported reads when the analysis requires them, else every read."""
    return "n_reads_supported" if params.require_supported else "n_reads"


def run_model_reads(model_reads: dict, column: str = "n_reads") -> pd.Series:
    """Reads per model over a run's samples."""
    parts = [m.groupby(m.transcript.astype(str), observed=True)[column].sum() for m in model_reads.values()]
    return pd.concat(parts, axis=1).sum(axis=1) if parts else pd.Series(dtype=float)


def add_gene_type(t: pd.DataFrame, gene_types: pd.Series, by: str | None = None) -> pd.DataFrame:
    """Add gene_type, coding (protein-coding gene) and q_noncoding: q over the tested non-coding genes of each
    family (`by`)."""
    if t is None or t.empty:
        return t
    t = t.copy()
    t["gene_type"] = t.gene_id.map(gene_types).fillna("novel")
    t["coding"] = t.gene_type.eq("protein_coding")
    t["q_noncoding"] = np.nan
    if "q" not in t:
        return t
    for _, idx in (t.groupby(by).groups.items() if by else [(None, t.index)]):
        sub = t.loc[idx]
        nc = sub.index[~sub.coding & sub.q.notna()]
        if len(nc):
            t.loc[nc, "q_noncoding"] = bh(sub.loc[nc, "p"])
    return t


@dataclass
class ApaResults:
    tables: dict = field(default_factory=dict)
    families: list = field(default_factory=list)


def analyse_apa(ctx) -> ApaResults:
    """All APA analyses. ctx: pipeline.Context."""
    cfg, sheet, ap, sp = ctx.cfg, ctx.sheet, ctx.cfg.apa, ctx.cfg.stats
    names = ctx.gene_names
    steps = ctx.steps
    out = ApaResults()
    units = sheet.units
    min_units = cfg.min_units(len(units))
    within_test = usage_test_name(sheet.design_names.values())
    site_tables, count_tables = [], []
    tandem, classes, ipa = [], [], []
    site_tails, pas_rows, usage_rows = [], [], []

    for run, cache in ctx.caches.items():
        per_sample = {d: cache.load(d, "sites") for d in cache.datasets}
        clusters, counts = cluster_sites(per_sample, ap.cluster_nt)
        steps.add(f"poly(A) clusters (sites within {ap.cluster_nt} nt chained)", len(clusters), unit=run)
        model_reads = {d: cache.load(d, "model_reads") for d in cache.datasets}
        canonical = reference_models(ctx.ref["canonical"], clusters.gene_id.unique(), ctx.model_exons[run],
                                     run_model_reads(model_reads, read_count(ap)), novel=ap.novel_gene_models)
        clusters = classify_sites(clusters, canonical, ctx.terminal[run], ap.downstream_nt, ap.model_window_nt,
                                  ap.intronic_models)
        clusters["region"] = site_region(clusters, ctx.ref["start"], ctx.ref["stop"])
        clusters.insert(0, "run", run)
        clusters.insert(3, "gene_name", clusters.gene_id.map(names).fillna(""))
        if ap.novel_gene_models:
            steps.add("novel genes placed on their best-supported IsoForge model",
                      int(clusters.drop_duplicates("gene_id").canonical_source.eq("IsoForge model").sum()), unit=run)
        for cls, n in clusters.groupby("site_class").reads.sum().items():
            steps.add(f"reads at {cls} sites", int(n), unit=run)
        site_tables.append(clusters)
        count_tables.append(counts.assign(run=run))
        by_cluster = clusters.set_index("cluster")
        members = {label: by_cluster.site_class.isin(classes_) for label, classes_ in {**APA_CLASSES, **IPA}.items()}
        members[FIVE_UTR] = by_cluster.region.eq("5'UTR") & by_cluster.site_class.isin(NON_TERMINAL)
        candidates = tandem_candidates(clusters, ap)

        for unit in sheet.units_of_run(run):
            design = sheet.design(unit)
            ut = sheet.unit_table(unit)
            t = top_two_within_unit(unit, clusters, counts, design, ap, sp, candidates, names)
            if len(t):
                t = annotate_tandem(t, ctx.ref, ctx.model_exons[run], ctx.terminal[run], ap)
                mr = _model_reads_by_condition(model_reads, ut, sheet, read_count(ap))
                t = t.join(site_models(t, ["proximal_site", "distal_site"], mr, ctx.terminal[run],
                                       ap.site_model_window_nt))
                t.insert(1, "run", run)
                tandem.append(t)
                steps.add(f"tandem 3'UTR genes tested (>= {ap.min_reads} reads per condition)", int(t.is_tandem.sum()),
                          unit=unit)
                if (~t.is_tandem).any():
                    steps.add("pairs excluded by the same-exon check", int((~t.is_tandem).sum()), unit=unit)
            for label, in_class in members.items():
                c = class_within_unit(unit, clusters, counts, design, label, in_class, ap, sp, names)
                if len(c):
                    c.insert(1, "run", run)
                    (ipa if label in IPA else classes).append(c)
                    steps.add(f"{label}: genes tested", len(c), unit=unit)
            st = _site_tails(cache, ut, sheet, clusters, t if len(t) else None, unit)
            if st is not None:
                site_tails.append(st)
            pas_rows.append(_pas_usage(clusters, counts, ut, sheet, unit))
            if cfg.usage_genes:
                usage_rows.append(_usage(model_reads, ut, sheet, unit, ctx.transcripts[run], cfg.usage_genes, names,
                                         read_count(ap)))

    tables = out.tables
    sites = pd.concat(site_tables, ignore_index=True)
    counts = pd.concat(count_tables, ignore_index=True)
    cond = sheet.table.set_index("dataset").condition
    counts["condition"] = counts.dataset.map(cond)
    wide = counts.pivot_table(index=["run", "cluster"], columns="condition", values="reads", aggfunc="sum",
                              fill_value=0)
    sites = sites.join(wide.rename(columns={sheet.control: "reads_control", sheet.treatment: "reads_treated"}),
                       on=["run", "cluster"])
    tables["sites"] = sites
    tables["sites_sample_counts"] = counts[["run", "cluster", "dataset", "reads", "reads_tail"]]

    tw = pd.concat(tandem, ignore_index=True) if tandem else pd.DataFrame()
    if len(tw):
        tw = match_pairs(tw, ap.match_tolerance_nt, valid="is_tandem")
    tw = add_gene_type(tw, ctx.gene_types, by="unit")
    tables["apa_tandem_within_unit"] = tw
    for u in units:
        out.families.append(dict(family=f"APA tandem 3'UTR within {u}", analysis="apa_tandem_within_unit",
                                 test=f"distal vs proximal reads: {within_test}",
                                 unit_of_replication="replicate", table="apa_tandem_within_unit",
                                 effect="delta_pdui", filter={"unit": u}))
    ta = pd.DataFrame()
    if len(tw) and len(units) >= 2:
        use = tw[tw.is_tandem & (tw.same_pair if ap.across_units_same_pair else True)]
        meas = tw[tw.is_tandem].groupby("gene_id").size()
        same = use.groupby("gene_id").size()
        extra = {"n_units_measured": meas, "median_utr_span_nt": use.groupby("gene_id").utr_span_nt.median(),
                 "reference_proximal": use.groupby("gene_id").reference_proximal.first(),
                 "reference_distal": use.groupby("gene_id").reference_distal.first(),
                 "median_proximal_from_stop_nt": use.groupby("gene_id").proximal_from_stop_nt.median(),
                 "median_distal_from_stop_nt": use.groupby("gene_id").distal_from_stop_nt.median(),
                 "n_units_same_pair": same}
        ta = across_units(use.assign(q_within=use.q), "delta_pdui", min_units, sp, "distal", "proximal", extra)
        est_tandem = ta.attrs.get("estimator", "") if len(ta) else ""
        if len(ta):
            ta.insert(1, "gene_name", ta.gene_id.map(names).fillna(""))
            ta["length_gained_nt"] = ta.mean_delta_pdui * ta.median_utr_span_nt
            steps.add(f"tandem 3'UTR genes tested across units (>= {min_units} units"
                      + (", same pair" if ap.across_units_same_pair else "") + ")", len(ta))
    ta = add_gene_type(ta, ctx.gene_types)
    tables["apa_tandem_across_units"] = ta
    out.families.append(dict(family="APA tandem 3'UTR across units, non-coding genes",
                             analysis="apa_tandem_across_units",
                             test="moderated one-sample t-test of per-unit delta PDUI over units; q over the "
                                  "non-coding genes",
                             unit_of_replication="unit", table="apa_tandem_across_units", effect="mean_delta_pdui",
                             q="q_noncoding", filter={"coding": False}))
    out.families.append(dict(family="APA tandem 3'UTR across units", analysis="apa_tandem_across_units",
                             test="moderated one-sample t-test of per-unit delta PDUI over units",
                             unit_of_replication="unit", table="apa_tandem_across_units", effect="mean_delta_pdui",
                             estimator=est_tandem if len(tw) and len(units) >= 2 else ""))

    for key, rows, labels in (("apa_class", classes, [*APA_CLASSES, FIVE_UTR]), ("apa_ipa", ipa, list(IPA))):
        w = add_gene_type(pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(), ctx.gene_types,
                          by=["unit", "apa_class"])
        tables[f"{key}_within_unit"] = w
        across = []
        for label in labels:
            est = ""
            for u in units:
                out.families.append(dict(family=f"APA {label} within {u}", analysis=f"{key}_within_unit",
                                         test=f"class vs terminal reads: {within_test}",
                                         unit_of_replication="replicate",
                                         table=f"{key}_within_unit", effect="delta_share",
                                         filter={"unit": u, "apa_class": label}))
            if len(w) and len(units) >= 2:
                sub = w[w.apa_class == label]
                a = across_units(sub.assign(q_within=sub.q), "delta_share", min_units, sp, "up", "down")
                est = a.attrs.get("estimator", "") if len(a) else ""
                if len(a):
                    a.insert(1, "gene_name", a.gene_id.map(names).fillna(""))
                    a.insert(2, "apa_class", label)
                    across.append(a)
                    steps.add(f"{label}: genes tested across units (>= {min_units} units)", len(a))
            out.families.append(dict(family=f"APA {label} across units", analysis=f"{key}_across_units",
                                     test="moderated one-sample t-test of per-unit delta share over units",
                                     unit_of_replication="unit", table=f"{key}_across_units",
                                     effect="mean_delta_share", filter={"apa_class": label},
                                     estimator=est if len(w) and len(units) >= 2 else ""))
        tables[f"{key}_across_units"] = add_gene_type(pd.concat(across, ignore_index=True) if across
                                                      else pd.DataFrame(), ctx.gene_types, by="apa_class")

    st = pd.concat(site_tails, ignore_index=True) if site_tails else pd.DataFrame()
    tables["tails_by_site"] = st
    tables["tails_by_site_tests"] = _site_tail_tests(st, sp) if len(st) else pd.DataFrame()
    tables["sites_pas_usage"] = pd.concat(pas_rows, ignore_index=True) if pas_rows else pd.DataFrame()
    us = pd.concat(usage_rows, ignore_index=True) if usage_rows else pd.DataFrame()
    tables["transcript_usage"] = us
    tables["transcript_usage_tests"] = (_usage_tests(us, sp, sheet.control, sheet.treatment) if len(us)
                                        else pd.DataFrame())
    return out


def _model_reads_by_condition(model_reads: dict, ut: pd.DataFrame, sheet, column: str = "n_reads") -> pd.DataFrame:
    frames = []
    for d, cond in zip(ut.dataset, ut.condition):
        m = model_reads[d][["gene", "transcript", column]].rename(columns={column: "n_reads"})
        m["gene"], m["transcript"] = m.gene.astype(str), m.transcript.astype(str)
        m["side"] = "reads_control" if cond == sheet.control else "reads_treated"
        frames.append(m)
    m = pd.concat(frames).pivot_table(index=["gene", "transcript"], columns="side", values="n_reads", aggfunc="sum",
                                      fill_value=0).reset_index()
    for c in ("reads_control", "reads_treated"):
        if c not in m:
            m[c] = 0
    return m


def _site_tails(cache, ut, sheet, clusters: pd.DataFrame, tandem: pd.DataFrame | None, unit: str):
    """Median tail of the reads at each tandem gene's proximal and distal sites, and at each class of site, per
    condition (the reads of a condition's samples pooled)."""
    frames = []
    for d, cond in zip(ut.dataset, ut.condition):
        h = cache.load(d, "site_tail_hist")
        if h.empty:
            continue
        h = h.assign(gene=h.gene.astype(str), condition="control" if cond == sheet.control else "treated")
        frames.append(h)
    if not frames:
        return None
    h = pd.concat(frames, ignore_index=True)
    cl = clusters[["cluster", "gene_id", "start", "end", "site_class"]].astype({"start": np.int64, "gene_id": object})
    cl = cl.rename(columns={"gene_id": "gene", "start": "site"}).sort_values("site")
    h = pd.merge_asof(h.astype({"site": np.int64, "gene": object}).sort_values("site"), cl, on="site", by="gene",
                      direction="backward")
    h = h[h.site <= h.end]
    rows = []
    if tandem is not None and len(tandem):
        t = tandem[tandem.is_tandem]
        for side in ("proximal", "distal"):
            sel = h[h.cluster.isin(set(t[f"{side}_cluster"]))]
            q = grouped_quantiles(sel[["cluster", "condition"]].astype(str), sel["tail"].to_numpy(),
                                  sel.n_reads.to_numpy(), qs=(0.5,)).reset_index()
            q["cluster"] = q.cluster.astype(int)
            q = q.merge(t[["gene_id", "gene_name", f"{side}_cluster"]].rename(columns={f"{side}_cluster": "cluster"}),
                        on="cluster")
            for _, r in q.iterrows():
                rows.append(dict(unit=unit, view="tandem", gene_id=r.gene_id, gene_name=r.gene_name, site=side,
                                 condition=r.condition, reads=int(r.n), median_tail_nt=r.q50))
    q = grouped_quantiles(h[["site_class", "condition"]].astype(str), h["tail"].to_numpy(), h.n_reads.to_numpy(),
                          qs=(0.5,)).reset_index()
    for _, r in q.iterrows():
        rows.append(dict(unit=unit, view="class", gene_id="", gene_name="", site=r.site_class, condition=r.condition,
                         reads=int(r.n), median_tail_nt=r.q50))
    return pd.DataFrame(rows)


def _site_tail_tests(st: pd.DataFrame, sp) -> pd.DataFrame:
    """Across units: distal minus proximal site tail (median over genes per unit and condition), Wilcoxon
    signed-rank test paired by unit."""
    t = st[st.view == "tandem"]
    if t.empty:
        return pd.DataFrame()
    w = t.pivot_table(index=["unit", "gene_id", "condition"], columns="site", values="median_tail_nt").dropna()
    if w.empty or "distal" not in w or "proximal" not in w:
        return pd.DataFrame()
    w["diff"] = w.distal - w.proximal
    per = w.groupby(["unit", "condition"])["diff"].median().unstack("condition")
    rows = []
    for cond in per.columns:
        r = wilcoxon_paired(per[cond], None, min_n=sp.wilcoxon_min_units)
        rows.append(dict(comparison=f"distal - proximal site tail ({cond})", n_units=r["n"],
                         median_difference_nt=r["median_difference"], n_units_greater=r["n_greater"],
                         n_units_less=r["n_less"], p=r["p"]))
    out = pd.DataFrame(rows)
    out["q"] = bh(out.p)
    return out


def _pas_usage(clusters, counts, ut, sheet, unit: str) -> pd.DataFrame:
    """Share of reads by poly(A) signal hexamer, per site class and condition."""
    c = counts[counts.dataset.isin(ut.dataset)].merge(clusters[["cluster", "site_class", "hexamer"]], on="cluster")
    c["condition"] = c.dataset.map(ut.set_index("dataset").condition)
    c["signal"] = np.where(c.hexamer.isin(PAS_MAIN), c.hexamer, np.where(c.hexamer.isin(["", "none"]), "none",
                                                                          "other hexamer"))
    g = c.groupby(["site_class", "condition", "signal"]).reads.sum().reset_index()
    g["share"] = g.reads / g.groupby(["site_class", "condition"]).reads.transform("sum")
    g.insert(0, "unit", unit)
    return g


def _usage(model_reads: dict, ut, sheet, unit, tx: pd.DataFrame, genes, names,
           column: str = "n_reads") -> pd.DataFrame:
    """CPM of each model structure of the chosen genes, per sample, from the reads counted in `column`; a structure
    without reads in a sample has CPM 0."""
    ids = set()
    for g in genes:
        ids |= set(names.index[names == g]) | ({g.split(".")[0]} if str(g).startswith("ENSG") else set())
    key = tx.set_index("transcript_id")
    rows = []
    for d, cond in zip(ut.dataset, ut.condition):
        m = model_reads[d].assign(n_reads=model_reads[d][column])
        total = m.n_reads.sum()
        sel = m[m.gene.astype(str).isin(ids)].copy()
        sel["gene"] = sel.gene.astype(str)
        sel["transcript"] = sel.transcript.astype(str)
        sel["structure_key"] = sel.transcript.map(key.structure_key).fillna("model not in GTF:" + sel.transcript)
        sel["n_exons"] = sel.transcript.map(key.n_exons)
        g = sel.groupby(["gene", "structure_key"], observed=True).agg(reads=("n_reads", "sum"), n_exons=("n_exons", "first"),
                                                       models=("transcript", lambda x: ",".join(sorted(set(x)))))
        for (gene, sk), r in g.iterrows():
            rows.append(dict(unit=unit, dataset=d, condition=cond, gene_id=gene, gene_name=names.get(gene, ""),
                             structure_key=sk, n_exons=r.n_exons, models=r.models, reads=int(r.reads),
                             cpm=r.reads / total * 1e6))
    us = pd.DataFrame(rows)
    if us.empty:
        return us
    info = us.drop_duplicates(["gene_id", "structure_key"]).set_index(["gene_id", "structure_key"])
    grid = pd.MultiIndex.from_product([ut.dataset, info.index.get_level_values(0).unique()], names=["dataset", "gene_id"])
    full = (us.set_index(["dataset", "gene_id", "structure_key"])[["reads", "cpm"]]
            .unstack(["gene_id", "structure_key"], fill_value=0).stack(["gene_id", "structure_key"], future_stack=True)
            .fillna(0).reset_index())
    full = full[full.set_index(["dataset", "gene_id"]).index.isin(grid)]
    full["unit"] = unit
    full["condition"] = full.dataset.map(ut.set_index("dataset").condition)
    full = full.join(info[["gene_name", "n_exons", "models"]], on=["gene_id", "structure_key"])
    full["reads"] = full.reads.astype(int)
    return full[["unit", "dataset", "condition", "gene_id", "gene_name", "structure_key", "n_exons", "models", "reads",
                 "cpm"]]


def _usage_tests(us: pd.DataFrame, sp, control: str, treatment: str) -> pd.DataFrame:
    """Per structure: mean CPM per unit and condition, and the Wilcoxon signed-rank test paired by unit."""
    per = us.groupby(["gene_name", "structure_key", "unit", "condition"]).cpm.mean().unstack("condition")
    if control not in per or treatment not in per:
        return pd.DataFrame()
    rows = []
    for (gene, sk), g in per.groupby(level=[0, 1]):
        r = wilcoxon_paired(g[treatment], g[control], min_n=sp.wilcoxon_min_units)
        rows.append(dict(gene_name=gene, structure_key=sk, n_units=r["n"], median_cpm_control=g[control].median(),
                         median_cpm_treated=g[treatment].median(), median_difference_cpm=r["median_difference"],
                         n_units_up=r["n_greater"], n_units_down=r["n_less"], p=r["p"]))
    out = pd.DataFrame(rows)
    out["q"] = bh(out.p)
    return out

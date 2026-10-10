"""Poly(A) tail length per sample, gene, transcript, category and gene set.

Per sample, the tail of a transcript or gene is the median over its reads, and the sample's overall tail is the
median of its per-transcript medians. Within a unit a feature's change is tested over the replicates: the moderated
one-sample t-test of the paired differences (paired design) or the moderated two-sample t-test of the sample
medians (unpaired design). Across units the per-unit change is tested with the moderated one-sample t-test.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .hist import grouped_quantiles
from .stats import (bh, bootstrap_ci, condition_t, moderated_one_sample, moderated_two_sample, wilcoxon_paired)
from .stats.readlevel import mann_whitney_counts

DECAY_KINDS = ("NMD", "retained intron")
TAIL_TEST = {"paired": "moderated one-sample t-test of the paired differences in median tail over replicate pairs",
             "unpaired": "moderated two-sample t-test of the sample median tails over replicates"}
SAMPLE_TEST = {"paired": "paired t-test over replicate pairs", "unpaired": "Welch t-test over replicates"}


def test_name(names: dict, designs) -> str:
    """The within-unit test of the designs present."""
    present = sorted(set(designs))
    return "; ".join(names[d] + (f" ({d} units)" if len(present) > 1 else "") for d in present)


def summarise_features(hist: pd.DataFrame, keys: list[str], count: str = "n_reads", mapq: bool = False) -> pd.DataFrame:
    """Per feature: n reads, mean, q25, q50 and q75 of the tail, exact from the counts. With mapq=True also n_mapq
    and q50_mapq over the reads counted in n_reads_mapq."""
    h = hist[hist[count] > 0]
    cols = keys + ["n", "mean", "q25", "q50", "q75"] + (["n_mapq", "q50_mapq"] if mapq else [])
    if h.empty:
        return pd.DataFrame(columns=cols)
    groups = h[keys[0]] if len(keys) == 1 else h[keys]
    extra = {"mapq": h["n_reads_mapq"].to_numpy()} if mapq else None
    res = grouped_quantiles(groups, h["tail"].to_numpy(), h[count].to_numpy(), extra=extra)
    res.index.names = keys
    return res.reset_index()


def gene_hist(hist: pd.DataFrame) -> pd.DataFrame:
    """Tail counts per gene, its transcripts pooled."""
    return hist.groupby(["gene", "tail"], observed=True)[["n_reads", "n_reads_mapq"]].sum().reset_index()


def transcript_summary(hist: pd.DataFrame) -> pd.DataFrame:
    """Per transcript: summarise_features over all of its reads, with the gene most of them were assigned to (a
    few models carry reads of two overlapping genes; equal counts: the first gene ID). Columns gene, transcript, n,
    mean, q25, q50, q75, n_mapq, q50_mapq."""
    h = hist[hist.transcript != ""]
    out = summarise_features(h, ["transcript"], mapq=True)
    reads = h.groupby(["transcript", "gene"], observed=True).n_reads.sum().reset_index()
    gene = (reads.sort_values(["transcript", "n_reads", "gene"], ascending=[True, False, True])
            .drop_duplicates("transcript").set_index("transcript").gene)
    out.insert(0, "gene", out.transcript.map(gene).to_numpy())
    return out


def _median_of(hist: pd.DataFrame) -> float:
    """The median tail of all reads in a count table."""
    if not int(hist.n_reads.sum()):
        return np.nan
    return float(grouped_quantiles(np.zeros(len(hist)), hist["tail"].to_numpy(), hist.n_reads.to_numpy(),
                                   qs=(0.5,)).q50.iat[0])


def replicate_values(per_sample: pd.DataFrame, key: str, design, min_reads: int, n_col: str = "n",
                     value_col: str = "q50"):
    """(control, treated, reads_control, reads_treated): features x replicates of the sample medians and their read
    counts, missing where the sample has fewer than `min_reads` reads. Columns are aligned by pair when paired."""
    ok = per_sample[per_sample[n_col] >= min_reads]
    value = ok.pivot_table(index=key, columns="dataset", values=value_col, aggfunc="first")
    reads = ok.pivot_table(index=key, columns="dataset", values=n_col, aggfunc="first")

    def wide(table, datasets):
        return table.reindex(columns=list(datasets)).set_axis(range(len(datasets)), axis=1)
    return (wide(value, design.control), wide(value, design.treated), wide(reads, design.control),
            wide(reads, design.treated))


def feature_change(per_sample: pd.DataFrame, key: str, design, min_reads: int, min_replicates: int,
                   n_col: str = "n", value_col: str = "q50"):
    """Per feature with enough replicates: its median tail per condition, the change and the reads behind it.

    Paired: the change is the mean of the paired differences, over the pairs where both samples have `min_reads`
    reads. Unpaired: the difference of the condition means, over the samples with `min_reads` reads. Returns
    (summary, control, treated), the last two holding the replicate values that enter the test."""
    c, t, nc, nt = replicate_values(per_sample, key, design, min_reads, n_col, value_col)
    if design.paired:
        both = c.notna() & t.notna()
        c, t, nc, nt = c.where(both), t.where(both), nc.where(both), nt.where(both)
        keep = both.sum(axis=1) >= min_replicates
        delta = (t - c).mean(axis=1)
    else:
        keep = (c.notna().sum(axis=1) >= min_replicates) & (t.notna().sum(axis=1) >= min_replicates)
        delta = t.mean(axis=1) - c.mean(axis=1)
    c, t = c[keep], t[keep]
    out = pd.DataFrame({"median_tail_control_nt": c.mean(axis=1), "median_tail_treated_nt": t.mean(axis=1),
                        "delta_tail_nt": delta[keep], "design": "paired" if design.paired else "unpaired",
                        "n_control": c.notna().sum(axis=1), "n_treated": t.notna().sum(axis=1),
                        "reads_control": nc[keep].sum(axis=1).astype(int), "reads_treated": nt[keep].sum(axis=1).astype(int)})
    if design.paired:
        diffs = t - c
        out["n_pairs"] = diffs.notna().sum(axis=1)
        out["n_pairs_shorter"] = (diffs < 0).sum(axis=1)
        for k in diffs.columns:
            out[f"delta_tail_pair{k + 1}_nt"] = diffs[k]
    return out.rename_axis(key), c, t


def feature_test(per_sample: pd.DataFrame, key: str, design, min_reads: int, min_replicates: int, stats_params,
                 n_col: str = "n", value_col: str = "q50") -> pd.DataFrame:
    """feature_change with the moderated t-test over the unit's replicates: t, p, q over the features tested, and
    the variance estimates. Empty with fewer than 3 features."""
    out, c, t = feature_change(per_sample, key, design, min_reads, min_replicates, n_col, value_col)
    if len(out) < 3:
        return pd.DataFrame()
    kw = dict(robust=stats_params.robust, backend=stats_params.limma_backend, rscript=stats_params.rscript)
    res = moderated_one_sample(t - c, **kw) if design.paired else moderated_two_sample(c, t, **kw)
    cols = ["t", "p", "q", "df_residual", "df_prior", "df_total", "s2_prior", "s2_post"]
    out = out.join(res[cols], how="inner")
    out.attrs["estimator"] = res.attrs.get("estimator", "")
    return out


def _across(D: pd.DataFrame, min_units: int, stats_params) -> pd.DataFrame:
    """Moderated one-sample t-test of per-unit changes (features x units) over the units."""
    D = D[D.notna().sum(axis=1) >= min_units]
    if len(D) < 3 or D.shape[1] < 2:
        return pd.DataFrame()
    res = moderated_one_sample(D, robust=stats_params.robust, backend=stats_params.limma_backend,
                               rscript=stats_params.rscript)
    res = res.rename(columns={"mean": "mean_delta_tail_nt"})
    D = D.reindex(res.index)
    res["median_delta_tail_nt"] = D.median(axis=1)
    res["n_units_shorter"] = (D < 0).sum(axis=1)
    res["n_units_longer"] = (D > 0).sum(axis=1)
    return res


def read_level_flags(gene_hists: dict, design, delta: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    """Read-level Mann-Whitney tests for the genes tested within a unit, from the per-tail counts: every control
    read against every treated read (median_tail_reads_*_nt, p_reads_pooled, q_reads_pooled) and, for a paired
    design, n_pairs_reads_consistent: the pairs whose own test has q < alpha in the direction of `delta`.
    Descriptive: reads are not independent replicates."""
    genes = set(delta.index)

    def pooled(datasets):
        d = pd.concat([gene_hists[x] for x in datasets if x in gene_hists])
        return d[d.gene.isin(genes)].groupby(["gene", "tail"], as_index=False).n_reads.sum()
    r = mann_whitney_counts(pooled(design.control), pooled(design.treated)).set_index("gene")
    out = pd.DataFrame({"median_tail_reads_control_nt": r.median_a, "median_tail_reads_treated_nt": r.median_b,
                        "p_reads_pooled": r.p, "q_reads_pooled": bh(r.p)}, index=r.index)
    if design.paired:
        consistent = pd.Series(0, index=out.index)
        for _, c, t in design.pairs:
            if c not in gene_hists or t not in gene_hists:
                continue
            rp = mann_whitney_counts(pooled([c]), pooled([t])).set_index("gene")
            q = pd.Series(bh(rp.p), index=rp.index)
            same = np.sign(rp.median_b - rp.median_a) == np.sign(delta.reindex(rp.index))
            consistent = consistent.add(((q < alpha) & same).astype(int), fill_value=0)
        out["n_pairs_reads_consistent"] = consistent.reindex(out.index).fillna(0).astype(int)
    return out.rename_axis("gene_id")


@dataclass
class TailResults:
    tables: dict = field(default_factory=dict)
    families: list = field(default_factory=list)
    estimators: dict = field(default_factory=dict)


def _sample_test(values: pd.Series, design) -> dict:
    """condition_t of one value per dataset between the conditions of a unit."""
    return condition_t([values.get(d, np.nan) for d in design.control], [values.get(d, np.nan) for d in design.treated],
                       design.paired)


def analyse_tails(ctx) -> TailResults:
    """All tail analyses. ctx: pipeline.Context."""
    cfg, sheet = ctx.cfg, ctx.sheet
    tp, sp = cfg.tails, cfg.stats
    out = TailResults()
    alpha = sp.alpha
    units = sheet.units
    min_units = cfg.min_units(len(units))
    names = ctx.gene_names
    classes = ctx.gene_class_members
    steps = ctx.steps
    designs = sheet.design_names
    sample_test = test_name(SAMPLE_TEST, designs.values())
    lo, hi = tp.window_nt
    edges = np.arange(0, tp.hist_max_nt + tp.hist_bin_nt, tp.hist_bin_nt)

    sample_rows, sample_tests = [], []
    gene_within, tx_within = [], []
    gene_delta, gene_delta_mapq, structure_delta, structure_meta = [], [], [], []
    gene_sample_frames, tx_sample_frames = [], []
    isoform_rows, cat_sample_rows = [], []
    thr_rows, hist_rows = [], []

    for unit in units:
        design = sheet.design(unit)
        ut = sheet.unit_table(unit).set_index("dataset")
        run = ut.run.iat[0]
        cache = ctx.caches[run]
        tx_info = ctx.transcripts.get(run)
        cat_of = tx_info.set_index("transcript_id").category if tx_info is not None else None
        key_of = None
        if tx_info is not None:
            key_of = tx_info.set_index("transcript_id").structure_key
            key_of = key_of.where(~key_of.astype(str).str.startswith("mono:"), "run:" + run + ":" + key_of.index)
        tx_sample, gene_sample, structure_sample, gene_hists = [], [], [], {}
        for d in ut.index:
            h = cache.load(d, "tail_hist")
            for c in ("gene", "transcript"):
                h[c] = h[c].astype(str)
            cond, pair = ut.condition[d], ut.pair[d]
            txs = transcript_summary(h).assign(dataset=d)
            gh = gene_hist(h)
            gene_sample.append(summarise_features(gh, ["gene"], mapq=True).assign(dataset=d))
            tx_sample.append(txs)
            if key_of is not None:
                hk = h[h.transcript != ""]
                hk = hk.assign(structure_key=hk.transcript.map(key_of).fillna("run:" + run + ":" + hk.transcript))
                structure_sample.append(summarise_features(hk, ["structure_key"]).assign(dataset=d))
            if tp.read_level:
                gene_hists[d] = gh[["gene", "tail", "n_reads"]]
            big = txs[txs.n >= tp.min_reads_transcript_sample]
            sample_rows.append(dict(unit=unit, dataset=d, condition=cond, pair=pair,
                                    reads_with_tail=int(h.n_reads.sum()), n_transcripts=int(len(big)),
                                    median_tail_nt=float(big.q50.median()) if len(big) else np.nan,
                                    median_tail_all_reads_nt=_median_of(gh)))
            for cls in tp.threshold_sets:
                g = gh[gh.gene.isin(classes.get(cls, set()))]
                n = int(g.n_reads.sum())
                thr_rows.append(dict(
                    unit=unit, dataset=d, condition=cond, pair=pair, gene_set=cls, n_genes=int(g.gene.nunique()),
                    reads=n, median_tail_nt=_median_of(g),
                    **{f"share_below_{tp.short_nt:g}nt": g.n_reads[g["tail"] < tp.short_nt].sum() / n if n else np.nan,
                       f"share_{lo:g}_{hi:g}nt": (g.n_reads[(g["tail"] >= lo) & (g["tail"] <= hi)].sum() / n
                                                  if n else np.nan)}))
                counts = np.histogram(np.clip(g["tail"].to_numpy(), 0, tp.hist_max_nt - 1), edges,
                                      weights=g.n_reads.to_numpy())[0]
                hist_rows.append(pd.DataFrame({"unit": unit, "dataset": d, "condition": cond, "gene_set": cls,
                                               "bin_start_nt": edges[:-1], "bin_end_nt": edges[1:],
                                               "reads": counts.astype(int)}))
            if cat_of is not None:
                by_cat = summarise_features(h.assign(category=h.transcript.map(cat_of).fillna("unclassified")
                                                     .to_numpy()), ["category"])
                txc = txs.assign(category=txs.transcript.map(cat_of).fillna("unclassified"))
                per_tx = (txc[txc.n >= tp.min_reads_category_transcript].groupby("category").q50
                          .agg(["median", "size"]))
                for r in by_cat.itertuples():
                    cat_sample_rows.append(dict(unit=unit, dataset=d, condition=cond, pair=pair, category=r.category,
                                                reads=int(r.n), read_median_tail_nt=r.q50,
                                                transcript_median_tail_nt=per_tx["median"].get(r.category, np.nan),
                                                n_transcripts=int(per_tx["size"].get(r.category, 0))))
        tx_sample = pd.concat(tx_sample, ignore_index=True)
        gene_sample = pd.concat(gene_sample, ignore_index=True)
        gene_sample_frames.append(gene_sample.assign(unit=unit))
        big = tx_sample[tx_sample.n >= tp.min_reads_transcript_sample].assign(unit=unit)[
            ["unit", "dataset", "gene", "transcript", "n", "q50"]]
        if cat_of is not None:
            big["category"] = big.transcript.map(cat_of).fillna("unclassified")
        tx_sample_frames.append(big)

        srows = pd.DataFrame([r for r in sample_rows if r["unit"] == unit]).set_index("dataset")
        res = _sample_test(srows.median_tail_nt, design)
        sample_tests.append(dict(
            unit=unit, design=designs[unit], n_control=res["n_control"], n_treated=res["n_treated"],
            median_tail_control_nt=res["mean_control"], median_tail_treated_nt=res["mean_treated"],
            delta_tail_nt=res["mean_change"], t=res["t"], p=res["p"],
            replicate_deltas_nt=";".join(f"{label}={srows.median_tail_nt[t] - srows.median_tail_nt[c]:.3f}"
                                         for label, c, t in design.pairs)))

        gs_all = gene_sample.rename(columns={"gene": "gene_id"})
        gw = feature_test(gs_all, "gene_id", design, tp.min_reads_gene, tp.min_replicates, sp)
        if len(gw):
            out.estimators[f"genes within {unit}"] = gw.attrs.get("estimator", "")
            if gs_all.n_mapq.fillna(0).sum():
                mq = feature_test(gs_all, "gene_id", design, tp.min_reads_gene, tp.min_replicates, sp,
                                  n_col="n_mapq", value_col="q50_mapq")
                if len(mq):
                    gw = gw.join(mq[["delta_tail_nt", "p", "q"]].add_suffix("_mapq"))
                    out.estimators[f"genes within {unit}, MAPQ"] = mq.attrs.get("estimator", "")
            if tp.read_level and gene_hists:
                gw = gw.join(read_level_flags(gene_hists, design, gw.delta_tail_nt, alpha))
            gw = gw.reset_index()
            gw.insert(0, "unit", unit)
            gw.insert(2, "gene_name", gw.gene_id.map(names).fillna(""))
            gene_within.append(gw)
            steps.add(f"genes tested within the unit (>= {tp.min_reads_gene} reads per sample, >= "
                      f"{tp.min_replicates} replicates)", len(gw), unit=unit)
            gene_delta.append(gw.set_index("gene_id").delta_tail_nt.rename(unit))
            if "delta_tail_nt_mapq" in gw:
                gene_delta_mapq.append(gw.set_index("gene_id").delta_tail_nt_mapq.dropna().rename(unit))

        tw = feature_test(tx_sample.rename(columns={"transcript": "transcript_id"}), "transcript_id", design,
                          tp.min_reads_transcript, tp.min_replicates, sp)
        if len(tw):
            out.estimators[f"transcripts within {unit}"] = tw.attrs.get("estimator", "")
            tw = tw.reset_index()
            gene_of = tx_sample.drop_duplicates("transcript").set_index("transcript").gene
            tw.insert(0, "unit", unit)
            tw.insert(2, "gene_id", tw.transcript_id.map(gene_of))
            tw.insert(3, "gene_name", tw.gene_id.map(names).fillna(""))
            fallback = "run:" + run + ":" + tw.transcript_id
            tw.insert(4, "structure_key", tw.transcript_id.map(key_of).fillna(fallback) if key_of is not None
                      else fallback)
            if cat_of is not None:
                tw.insert(5, "category", tw.transcript_id.map(cat_of).fillna("unclassified"))
            tx_within.append(tw)
            steps.add(f"transcripts tested within the unit (>= {tp.min_reads_transcript} reads per sample)",
                      len(tw), unit=unit)
        if structure_sample:
            change, _, _ = feature_change(pd.concat(structure_sample, ignore_index=True), "structure_key", design,
                                          tp.min_reads_transcript, tp.min_replicates)
            structure_delta.append(change.delta_tail_nt.rename(unit))
            if len(tw):
                structure_meta.append(tw.assign(reads=tw.reads_control + tw.reads_treated)[
                    ["structure_key", "transcript_id", "gene_id", "gene_name", "reads"]
                    + (["category"] if "category" in tw else [])])

        if cat_of is not None:
            iso = within_gene_isoforms(tx_sample, cat_of, ut.reset_index(), sheet, tp.min_reads_isoform)
            if len(iso):
                iso.insert(0, "unit", unit)
                iso.insert(2, "gene_name", iso.gene_id.map(names).fillna(""))
                isoform_rows.append(iso)

    tables = out.tables
    tables["tails_sample"] = pd.DataFrame(sample_rows)
    st = pd.DataFrame(sample_tests)
    st["q"] = bh(st.p)
    tables["tails_sample_within_unit"] = st
    est, ci_lo, ci_hi = bootstrap_ci(st.delta_tail_nt, n_boot=sp.bootstrap, seed=sp.seed)
    tables["tails_sample_across_units"] = pd.DataFrame([dict(
        n_units=int(st.delta_tail_nt.notna().sum()), n_units_shorter=int((st.delta_tail_nt < 0).sum()),
        n_units_longer=int((st.delta_tail_nt > 0).sum()), n_units_q_below_alpha=int((st.q < alpha).sum()),
        n_units_shorter_q_below_alpha=int(((st.q < alpha) & (st.delta_tail_nt < 0)).sum()),
        mean_delta_tail_nt=est, ci95_low_nt=ci_lo, ci95_high_nt=ci_hi,
        median_delta_tail_nt=st.delta_tail_nt.median(), bootstrap_resamples=sp.bootstrap, seed=sp.seed,
        units_not_significant=", ".join(st.unit[~(st.q < alpha)].astype(str)))])
    out.families.append(dict(family="tails: sample tail, within each unit", analysis="tails_sample_within_unit",
                             test=f"sample median tail (median of per-transcript medians): {sample_test}",
                             unit_of_replication="replicate; q over units", table="tails_sample_within_unit",
                             effect="delta_tail_nt"))

    gw = pd.concat(gene_within, ignore_index=True) if gene_within else pd.DataFrame()
    tables["tails_gene_within_unit"] = gw
    for u in units:
        out.families.append(dict(family=f"tails: genes within {u}", analysis="tails_gene_within_unit",
                                 test=TAIL_TEST[designs[u]], unit_of_replication="replicate",
                                 table="tails_gene_within_unit", effect="delta_tail_nt", filter={"unit": u},
                                 estimator=out.estimators.get(f"genes within {u}", "")))
    gs_out = pd.concat(gene_sample_frames, ignore_index=True).rename(columns={
        "gene": "gene_id", "n": "reads", "mean": "mean_tail_nt", "q25": "q25_tail_nt", "q50": "median_tail_nt",
        "q75": "q75_tail_nt", "n_mapq": "reads_mapq", "q50_mapq": "median_tail_mapq_nt"})
    gs_out.insert(2, "gene_name", gs_out.gene_id.map(names).fillna(""))
    tables["tails_gene_sample"] = gs_out[["unit", "dataset", "gene_id", "gene_name", "reads", "median_tail_nt",
                                          "q25_tail_nt", "q75_tail_nt", "mean_tail_nt", "reads_mapq",
                                          "median_tail_mapq_nt"]]
    tables["tails_transcript_sample"] = pd.concat(tx_sample_frames, ignore_index=True).rename(columns={
        "gene": "gene_id", "transcript": "transcript_id", "n": "reads", "q50": "median_tail_nt"})
    D = pd.concat(gene_delta, axis=1) if gene_delta else pd.DataFrame()
    by_unit = D.stack().rename("delta_tail_nt").reset_index() if len(D) else pd.DataFrame()
    if len(by_unit):
        by_unit.columns = ["gene_id", "unit", "delta_tail_nt"]
        by_unit.insert(1, "gene_name", by_unit.gene_id.map(names).fillna(""))
    tables["tails_gene_by_unit"] = by_unit
    ga = _across(D, min_units, sp) if len(units) >= 2 and len(D) else pd.DataFrame()
    if len(ga):
        out.estimators["genes across units"] = ga.attrs.get("estimator", "")
        gm = _across(pd.concat(gene_delta_mapq, axis=1), min_units, sp) if gene_delta_mapq else pd.DataFrame()
        if len(gm):
            out.estimators["genes across units, MAPQ"] = gm.attrs.get("estimator", "")
            ga = ga.join(gm[["mean_delta_tail_nt", "p", "q", "n_units"]].add_suffix("_mapq"))
        sig = gw[gw.q < alpha]
        for label, sub in (("shorter", sig[sig.delta_tail_nt < 0]), ("longer", sig[sig.delta_tail_nt > 0])):
            ga[f"n_units_q_within_{label}"] = sub.groupby("gene_id").size().reindex(ga.index).fillna(0).astype(int)
        ga = ga.rename_axis("gene_id").reset_index()
        ga.insert(1, "gene_name", ga.gene_id.map(names).fillna(""))
        ga.insert(2, "gene_type", ga.gene_id.map(ctx.gene_types).fillna(""))
        for cls, members in classes.items():
            ga[f"is_{cls}"] = ga.gene_id.isin(members)
        steps.add(f"genes tested across units (measured in >= {min_units} of {len(units)} units)", len(ga))
    tables["tails_gene_across_units"] = ga
    out.families.append(dict(family="tails: genes across units", analysis="tails_gene_across_units",
                             test="moderated one-sample t-test of the per-unit change over units",
                             unit_of_replication="unit", table="tails_gene_across_units",
                             effect="mean_delta_tail_nt", estimator=out.estimators.get("genes across units", "")))
    if len(ga) and "q_mapq" in ga:
        out.families.append(dict(family=f"tails: genes across units, MAPQ >= {tp.mapq_min}",
                                 analysis="tails_gene_across_units",
                                 test=f"the same test on reads with mapping quality >= {tp.mapq_min}",
                                 unit_of_replication="unit", table="tails_gene_across_units",
                                 effect="mean_delta_tail_nt_mapq", q="q_mapq",
                                 estimator=out.estimators.get("genes across units, MAPQ", "")))

    tw = pd.concat(tx_within, ignore_index=True) if tx_within else pd.DataFrame()
    tables["tails_transcript_within_unit"] = tw
    for u in units:
        out.families.append(dict(family=f"tails: transcripts within {u}", analysis="tails_transcript_within_unit",
                                 test=TAIL_TEST[designs[u]], unit_of_replication="replicate",
                                 table="tails_transcript_within_unit", effect="delta_tail_nt", filter={"unit": u},
                                 estimator=out.estimators.get(f"transcripts within {u}", "")))
    T = pd.concat(structure_delta, axis=1) if structure_delta else pd.DataFrame()
    ta = _across(T, min_units, sp) if len(units) >= 2 and len(T) else pd.DataFrame()
    if len(ta):
        out.estimators["transcripts across units"] = ta.attrs.get("estimator", "")
        ta = ta.rename_axis("structure_key").reset_index()
        meta = pd.concat(structure_meta, ignore_index=True) if structure_meta else pd.DataFrame(
            columns=["structure_key", "transcript_id", "gene_id", "gene_name", "reads"])
        best = meta.sort_values(["structure_key", "reads", "transcript_id"], ascending=[True, False, True]) \
            .drop_duplicates("structure_key").set_index("structure_key")
        ids = meta.groupby("structure_key").transcript_id.agg(lambda x: ",".join(sorted(set(map(str, x)))))
        ta.insert(1, "gene_id", ta.structure_key.map(best.gene_id))
        ta.insert(2, "gene_name", ta.structure_key.map(best.gene_name))
        ta.insert(3, "transcript_ids", ta.structure_key.map(ids))
        if "category" in best:
            ta.insert(4, "category", ta.structure_key.map(best.category))
        steps.add(f"transcript structures tested across units (>= {min_units} units)", len(ta))
    tables["tails_transcript_across_units"] = ta
    out.families.append(dict(family="tails: transcripts across units", analysis="tails_transcript_across_units",
                             test="moderated one-sample t-test of the per-unit change over units; the reads of "
                                  "models with one intron chain pooled", unit_of_replication="unit",
                             table="tails_transcript_across_units", effect="mean_delta_tail_nt",
                             estimator=out.estimators.get("transcripts across units", "")))

    cs = pd.DataFrame(cat_sample_rows)
    tables["tails_category_sample"] = cs
    cat_unit, cat_tests = category_tests(cs, sheet, cfg) if len(cs) else (pd.DataFrame(), pd.DataFrame())
    tables["tails_categories"] = cat_unit
    tables["tails_category_tests"] = cat_tests
    if len(cat_tests):
        out.families.append(dict(family="tails: transcript categories", analysis="tails_category_tests",
                                 test="Wilcoxon signed-rank test paired by unit", unit_of_replication="unit",
                                 table="tails_category_tests", effect="median_difference_nt"))

    iso = pd.concat(isoform_rows, ignore_index=True) if isoform_rows else pd.DataFrame()
    tables["tails_within_gene_isoforms"] = iso
    iso_sum, iso_tests = isoform_summary(iso, sp) if len(iso) else (pd.DataFrame(), pd.DataFrame())
    tables["tails_within_gene_by_unit"] = iso_sum
    tables["tails_within_gene_tests"] = iso_tests
    if len(iso_tests):
        out.families.append(dict(family="tails: within-gene isoforms", analysis="tails_within_gene_tests",
                                 test="Wilcoxon signed-rank test over units of the per-unit medians over genes",
                                 unit_of_replication="unit", table="tails_within_gene_tests",
                                 effect="median_difference_nt"))

    thr = pd.DataFrame(thr_rows)
    tables["tails_thresholds"] = thr
    tables["tails_histograms"] = pd.concat(hist_rows, ignore_index=True) if hist_rows else pd.DataFrame()
    thr_unit, thr_tests = threshold_tests(thr, sheet, cfg) if len(thr) else (pd.DataFrame(), pd.DataFrame())
    tables["tails_thresholds_by_unit"] = thr_unit
    tables["tails_threshold_tests"] = thr_tests
    if len(thr_tests):
        out.families.append(dict(family="tails: gene-set thresholds", analysis="tails_threshold_tests",
                                 test=f"within units: {sample_test}; across units: Wilcoxon signed-rank test paired "
                                      "by unit", unit_of_replication="replicate / unit",
                                 table="tails_threshold_tests", effect="median_difference"))

    gc_unit, gc_tests = gene_class_tests(by_unit, classes, sp) if len(by_unit) else (pd.DataFrame(), pd.DataFrame())
    tables["tails_gene_classes"] = gc_unit
    tables["tails_gene_class_tests"] = gc_tests
    if len(gc_tests):
        out.families.append(dict(family="tails: gene classes vs all genes", analysis="tails_gene_class_tests",
                                 test="Wilcoxon signed-rank test paired by unit of the class median change against "
                                      "the all-gene median change", unit_of_replication="unit",
                                 table="tails_gene_class_tests", effect="median_difference_nt"))
    return out


def category_tests(cs: pd.DataFrame, sheet, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per unit and category: the tail per condition (mean over its samples of the per-sample median, at read and
    transcript level), the change and its test over the unit's replicates. Across units: Wilcoxon signed-rank tests
    paired by unit, between conditions within a category and between categories."""
    ctrl, trt = sheet.control, sheet.treatment
    rows = []
    for (unit, cat), g in cs.groupby(["unit", "category"]):
        design = sheet.design(unit)
        by = g.set_index("dataset")
        r = dict(unit=unit, category=cat, reads=int(g.reads.sum()),
                 n_transcripts_control=float(g[g.condition == ctrl].n_transcripts.mean()))
        for lvl, col in (("read", "read_median_tail_nt"), ("transcript", "transcript_median_tail_nt")):
            c = g[g.condition == ctrl][col].mean()
            t = g[g.condition == trt][col].mean()
            r[f"{lvl}_tail_control_nt"], r[f"{lvl}_tail_treated_nt"] = c, t
            r[f"{lvl}_delta_tail_nt"] = t - c
            r[f"{lvl}_p_within_unit"] = _sample_test(by[col], design)["p"]
        rows.append(r)
    unit_tab = pd.DataFrame(rows)
    tests = []
    cats = ["protein coding", "retained intron", "lncRNA", "NMD"]
    mu = cfg.stats.wilcoxon_min_units
    for lvl in ("read", "transcript"):
        wide = unit_tab.pivot_table(index="unit", columns="category",
                                    values=[f"{lvl}_tail_control_nt", f"{lvl}_tail_treated_nt", f"{lvl}_delta_tail_nt"])
        for k in cats:
            if (f"{lvl}_tail_control_nt", k) not in wide:
                continue
            w = wilcoxon_paired(wide[(f"{lvl}_tail_treated_nt", k)], wide[(f"{lvl}_tail_control_nt", k)], min_n=mu)
            tests.append(dict(level=lvl, comparison=f"{k}: treated vs control", what="tail", **_rename_w(w)))
        for a, b in [("NMD", "protein coding"), ("retained intron", "protein coding"), ("lncRNA", "protein coding"),
                     ("NMD", "lncRNA"), ("NMD", "retained intron"), ("lncRNA", "retained intron")]:
            for what, col in (("tail with control", f"{lvl}_tail_control_nt"), ("change", f"{lvl}_delta_tail_nt")):
                if (col, a) in wide and (col, b) in wide:
                    w = wilcoxon_paired(wide[(col, a)], wide[(col, b)], min_n=mu)
                    tests.append(dict(level=lvl, comparison=f"{a} vs {b}", what=what, **_rename_w(w)))
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["q"] = bh(tests.p)
    return unit_tab, tests


def _rename_w(w: dict) -> dict:
    return dict(n_units=w["n"], median_difference_nt=w["median_difference"], n_units_greater=w["n_greater"],
                n_units_less=w["n_less"], p=w["p"])


def within_gene_isoforms(tx_sample: pd.DataFrame, cat_of: pd.Series, ut: pd.DataFrame, sheet,
                         min_reads: int) -> pd.DataFrame:
    """Genes with a protein-coding isoform and an NMD or retained-intron isoform, each with >= min_reads reads per
    condition: the read-weighted tail of each (sample medians weighted by the sample's reads, transcripts by their
    reads), the difference decay - coding with the control, and the change of each with the treatment."""
    t = tx_sample[tx_sample.transcript != ""].copy()
    t["category"] = t.transcript.map(cat_of).astype(object)
    t = t[t.category.isin(["protein coding", *DECAY_KINDS])]
    if t.empty:
        return pd.DataFrame()
    cond = ut.set_index("dataset").condition
    t["condition"] = np.where(t.dataset.map(cond) == sheet.control, "control", "treated")
    t["w"] = t.q50 * t.n
    g = t.groupby(["gene", "category", "transcript", "condition"]).agg(n=("n", "sum"), w=("w", "sum")).unstack(
        "condition")
    if ("n", "control") not in g or ("n", "treated") not in g:
        return pd.DataFrame()
    g = g[(g[("n", "control")] >= min_reads) & (g[("n", "treated")] >= min_reads)]
    per = g.groupby(level=["gene", "category"]).sum()
    ids = g.reset_index().groupby(["gene", "category"]).transcript.agg(lambda x: ",".join(sorted(map(str, x))))
    rows = []
    for (gene, cat), r in per.iterrows():
        if cat == "protein coding" or (gene, "protein coding") not in per.index:
            continue
        pc = per.loc[(gene, "protein coding")]
        rows.append(dict(gene_id=gene, kind=cat,
                         tail_control_nt=r[("w", "control")] / r[("n", "control")],
                         tail_treated_nt=r[("w", "treated")] / r[("n", "treated")],
                         coding_tail_control_nt=pc[("w", "control")] / pc[("n", "control")],
                         coding_tail_treated_nt=pc[("w", "treated")] / pc[("n", "treated")],
                         reads_control=int(r[("n", "control")]), reads_treated=int(r[("n", "treated")]),
                         coding_reads_control=int(pc[("n", "control")]), coding_reads_treated=int(pc[("n", "treated")]),
                         transcripts=ids.get((gene, cat)), coding_transcripts=ids.get((gene, "protein coding"))))
    d = pd.DataFrame(rows)
    if len(d):
        d["longer_by_nt"] = d.tail_control_nt - d.coding_tail_control_nt
        d["delta_tail_nt"] = d.tail_treated_nt - d.tail_control_nt
        d["coding_delta_tail_nt"] = d.coding_tail_treated_nt - d.coding_tail_control_nt
    return d


def isoform_summary(iso: pd.DataFrame, sp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per unit and kind: medians over genes. Over units: Wilcoxon signed-rank tests paired by unit."""
    per = (iso.groupby(["unit", "kind"]).agg(n_genes=("gene_id", "size"),
                                             share_genes_longer=("longer_by_nt", lambda x: float((x > 0).mean())),
                                             median_longer_by_nt=("longer_by_nt", "median"),
                                             median_delta_tail_nt=("delta_tail_nt", "median"),
                                             median_coding_delta_tail_nt=("coding_delta_tail_nt", "median"))
           .reset_index())
    tests = []
    for kind, g in per.groupby("kind"):
        w = wilcoxon_paired(g.median_longer_by_nt, None, min_n=sp.wilcoxon_min_units)
        tests.append(dict(kind=kind, comparison="decay isoform tail - coding isoform tail, with the control",
                          n_gene_unit_pairs=int(iso[iso.kind == kind].shape[0]),
                          share_pairs_longer=float((iso[iso.kind == kind].longer_by_nt > 0).mean()),
                          **_rename_w(w)))
        w = wilcoxon_paired(g.median_delta_tail_nt, g.median_coding_delta_tail_nt, min_n=sp.wilcoxon_min_units)
        tests.append(dict(kind=kind, comparison="change of the decay isoform vs change of the coding isoform",
                          n_gene_unit_pairs=int(iso[iso.kind == kind].shape[0]), **_rename_w(w)))
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["q"] = bh(tests.p)
    return per, tests


def threshold_tests(thr: pd.DataFrame, sheet, cfg) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per unit and gene set: each measure per condition (replicate samples averaged) and its test over the unit's
    replicates. Across units: Wilcoxon signed-rank test paired by unit."""
    tp = cfg.tails
    lo, hi = tp.window_nt
    measures = [f"share_below_{tp.short_nt:g}nt", f"share_{lo:g}_{hi:g}nt", "median_tail_nt"]
    rows = []
    for (unit, gset), g in thr.groupby(["unit", "gene_set"]):
        design = sheet.design(unit)
        by = g.set_index("dataset")
        for m in measures:
            c = g[g.condition == sheet.control][m].mean()
            t = g[g.condition == sheet.treatment][m].mean()
            res = _sample_test(by[m], design)
            rows.append(dict(unit=unit, gene_set=gset, measure=m, control=c, treated=t, delta=t - c,
                             n_control=res["n_control"], n_treated=res["n_treated"], t=res["t"],
                             p_within_unit=res["p"], reads=int(g.reads.sum())))
    per = pd.DataFrame(rows)
    per["q_within_unit"] = np.nan
    for (gset, m), idx in per.groupby(["gene_set", "measure"]).groups.items():
        per.loc[idx, "q_within_unit"] = bh(per.loc[idx, "p_within_unit"])
    tests = []
    for (gset, m), g in per.groupby(["gene_set", "measure"]):
        w = wilcoxon_paired(g.treated, g.control, min_n=cfg.stats.wilcoxon_min_units)
        tests.append(dict(gene_set=gset, measure=m, n_units=w["n"], median_control=g.control.median(),
                          median_treated=g.treated.median(), median_difference=w["median_difference"],
                          n_units_up=w["n_greater"], n_units_down=w["n_less"], p=w["p"]))
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["q"] = bh(tests.p)
    return per, tests


def gene_class_tests(by_unit: pd.DataFrame, classes: dict, sp) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per class and unit: the median change over the class's genes and over all genes. Over units: Wilcoxon
    signed-rank test of class against all genes, paired by unit."""
    rows = []
    for unit, g in by_unit.groupby("unit"):
        all_med = g.delta_tail_nt.median()
        for cls, members in classes.items():
            sub = g[g.gene_id.isin(members)]
            rows.append(dict(unit=unit, gene_class=cls, n_genes=int(len(sub)),
                             median_delta_tail_nt=sub.delta_tail_nt.median() if len(sub) else np.nan,
                             median_delta_tail_all_genes_nt=all_med, n_genes_all=int(len(g))))
    per = pd.DataFrame(rows)
    tests = []
    for cls, g in per.groupby("gene_class"):
        g = g[g.n_genes > 0]
        w = wilcoxon_paired(g.median_delta_tail_nt, g.median_delta_tail_all_genes_nt, min_n=sp.wilcoxon_min_units)
        tests.append(dict(gene_class=cls, median_class_nt=g.median_delta_tail_nt.median(),
                          median_all_genes_nt=g.median_delta_tail_all_genes_nt.median(), **_rename_w(w)))
    tests = pd.DataFrame(tests)
    if len(tests):
        tests["q"] = bh(tests.p)
    return per, tests

"""The TSS stage: alternative promoters and first exons, tested like the poly(A) sites.

The 5' end IsoForge gave each read (`tss_site`) is counted per sample, clustered per gene and placed on the gene's
reference transcript (sites.classify_tss). Two kinds of test:

  promoter switch   per gene, its two most used TSS among the canonical first exon, upstream TSS and alternative
                    first exons. DTUI = downstream / (upstream + downstream) reads.
  TSS classes       reads at upstream TSS, at alternative first exons and at internal exonic TSS, each as a share
                    against the canonical first exon.

A 5' end inside an internal exon can be a promoter or a truncated read.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .apa import across_units, class_within_unit, match_pairs, top_two_within_unit
from .apa_run import add_gene_type, read_count, run_model_reads, usage_test_name
from .gtf import first_exons
from .ingest import ensure_tss
from .sites import classify_tss, cluster_sites, reference_models
from .stats import bh

SWITCH_CANDIDATES = ("canonical first exon", "upstream TSS", "alternative first exon")
TSS_TESTS = ("upstream TSS", "alternative first exon", "internal exonic TSS")
RENAME = {"proximal": "upstream", "distal": "downstream", "pdui": "dtui", "n_utr_sites": "n_tss_candidates",
          "utr_span_nt": "tss_span_nt"}
POLYA_ONLY = ("hexamer", "mean_fraction_as", "share_high_a", "reads_tail", "proximal_hexamer", "distal_hexamer",
              "length_gained_nt")


def _rename(t: pd.DataFrame) -> pd.DataFrame:
    def new(c):
        for a, b in RENAME.items():
            c = c.replace(a, b)
        return c
    return t.rename(columns={c: new(c) for c in t.columns})


@dataclass
class TssResults:
    tables: dict = field(default_factory=dict)
    families: list = field(default_factory=list)


def analyse_tss(ctx) -> TssResults:
    """All TSS analyses. ctx: pipeline.Context."""
    cfg, sheet, tp, sp = ctx.cfg, ctx.sheet, ctx.cfg.tss, ctx.cfg.stats
    names, steps = ctx.gene_names, ctx.steps
    out = TssResults()
    units = sheet.units
    min_units = cfg.min_units(len(units))
    within_test = usage_test_name(sheet.design_names.values())
    site_tables, count_tables, switch, classes = [], [], [], []
    for run, cache in ctx.caches.items():
        ensure_tss(cache, cfg.runs[run], tp.require_supported)
        per_sample = {}
        for d in cache.datasets:
            t = cache.load_tss(d, tp.require_supported)
            per_sample[d] = t[~t.chrom.astype(str).isin(tp.exclude_chroms)]
        clusters, counts = cluster_sites(per_sample, tp.cluster_nt)
        steps.add(f"TSS clusters (5' ends within {tp.cluster_nt} nt chained)", len(clusters), unit=run)
        model_reads = {d: cache.load(d, "model_reads") for d in cache.datasets}
        canonical = reference_models(ctx.ref["canonical"], clusters.gene_id.unique(), ctx.model_exons[run],
                                     run_model_reads(model_reads, read_count(tp)), novel=tp.novel_gene_models)
        clusters = classify_tss(clusters, canonical, first_exons(ctx.model_exons[run]), tp.upstream_nt,
                                tp.max_upstream_nt, tp.model_window_nt)
        clusters.insert(0, "run", run)
        clusters.insert(3, "gene_name", clusters.gene_id.map(names).fillna(""))
        for cls, n in clusters.groupby("site_class").reads.sum().items():
            steps.add(f"reads at {cls}", int(n), unit=run)
        site_tables.append(clusters.drop(columns=[c for c in POLYA_ONLY if c in clusters]))
        count_tables.append(counts.assign(run=run))
        by_cluster = clusters.set_index("cluster")
        candidates = by_cluster.site_class.isin(SWITCH_CANDIDATES)
        for unit in sheet.units_of_run(run):
            design = sheet.design(unit)
            t = top_two_within_unit(unit, clusters, counts, design, tp, sp, candidates, names)
            if len(t):
                t.insert(1, "run", run)
                switch.append(t)
                steps.add(f"promoter switch: genes tested (>= {tp.min_reads} reads per condition)", len(t), unit=unit)
            for label in TSS_TESTS:
                c = class_within_unit(unit, clusters, counts, design, label, by_cluster.site_class.eq(label), tp, sp,
                                      names, reference="canonical first exon", class_column="tss_class")
                if len(c):
                    c.insert(1, "run", run)
                    classes.append(c)
                    steps.add(f"{label}: genes tested", len(c), unit=unit)

    tables = out.tables
    tables["tss_sites"] = pd.concat(site_tables, ignore_index=True) if site_tables else pd.DataFrame()
    tables["tss_sample_counts"] = (pd.concat(count_tables, ignore_index=True)[["run", "cluster", "dataset", "reads"]]
                                   if count_tables else pd.DataFrame())

    sw = pd.concat(switch, ignore_index=True) if switch else pd.DataFrame()
    across = pd.DataFrame()
    if len(sw):
        sw = match_pairs(sw, tp.match_tolerance_nt)
        sw["q"] = np.nan
        for _, idx in sw.groupby("unit").groups.items():
            sw.loc[idx, "q"] = bh(sw.loc[idx, "p"])
        sw = sw.drop(columns=[c for c in POLYA_ONLY if c in sw])
    if len(sw) and len(units) >= 2:
        use = sw[sw.same_pair] if tp.across_units_same_pair else sw
        extra = {"n_units_measured": sw.groupby("gene_id").size(),
                 "median_span_nt": use.groupby("gene_id").utr_span_nt.median(),
                 "reference_upstream": use.groupby("gene_id").reference_proximal.first(),
                 "reference_downstream": use.groupby("gene_id").reference_distal.first()}
        across = across_units(use.assign(q_within=use.q), "delta_pdui", min_units, sp, "distal", "proximal", extra)
        if len(across):
            across.insert(1, "gene_name", across.gene_id.map(names).fillna(""))
            steps.add(f"promoter switch: genes tested across units (>= {min_units} units)", len(across))
    tables["tss_switch_within_unit"] = add_gene_type(_rename(sw), ctx.gene_types, by="unit") if len(sw) else sw
    tables["tss_switch_across_units"] = add_gene_type(_rename(across), ctx.gene_types) if len(across) else across
    for u in units:
        out.families.append(dict(family=f"TSS promoter switch within {u}", analysis="tss_switch_within_unit",
                                 test=f"downstream vs upstream TSS reads: {within_test}",
                                 unit_of_replication="replicate", table="tss_switch_within_unit",
                                 effect="delta_dtui", filter={"unit": u}))
    out.families.append(dict(family="TSS promoter switch across units", analysis="tss_switch_across_units",
                             test="moderated one-sample t-test of per-unit delta DTUI over units",
                             unit_of_replication="unit", table="tss_switch_across_units", effect="mean_delta_dtui"))

    w = pd.concat(classes, ignore_index=True) if classes else pd.DataFrame()
    tables["tss_class_within_unit"] = add_gene_type(w, ctx.gene_types, by=["unit", "tss_class"]) if len(w) else w
    acc = []
    for label in TSS_TESTS:
        for u in units:
            out.families.append(dict(family=f"TSS {label} within {u}", analysis="tss_class_within_unit",
                                     test=f"class vs canonical first-exon reads: {within_test}",
                                     unit_of_replication="replicate", table="tss_class_within_unit",
                                     effect="delta_share", filter={"unit": u, "tss_class": label}))
        if len(w) and len(units) >= 2:
            sub = w[w.tss_class == label]
            a = across_units(sub.assign(q_within=sub.q), "delta_share", min_units, sp, "up", "down")
            if len(a):
                a.insert(1, "gene_name", a.gene_id.map(names).fillna(""))
                a.insert(2, "tss_class", label)
                acc.append(a)
        out.families.append(dict(family=f"TSS {label} across units", analysis="tss_class_across_units",
                                 test="moderated one-sample t-test of per-unit delta share over units",
                                 unit_of_replication="unit", table="tss_class_across_units",
                                 effect="mean_delta_share", filter={"tss_class": label}))
    tables["tss_class_across_units"] = (add_gene_type(pd.concat(acc, ignore_index=True), ctx.gene_types,
                                                      by="tss_class") if acc else pd.DataFrame())
    return out

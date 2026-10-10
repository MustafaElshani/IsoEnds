"""report.md, tests_summary and provenance.json, written from the tables and logs of an output directory."""
from __future__ import annotations

import datetime
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from . import __version__
from .util import LOG, atomic_write_json, sha256

TABLE_DOCS = {
    "validation": "every input check: scope, check, value, status, message",
    "qc_samples": "per sample: reads, tail calls, reads without a gene or model, reads at sites, A-rich 3' ends",
    "tails_sample": "per sample: the overall tail (median of per-transcript medians) and the reads behind it",
    "tails_sample_within_unit": "per unit: test of the sample tail over the replicates; q over units",
    "tails_sample_across_units": "mean change over units with a percentile bootstrap 95% CI",
    "tails_gene_sample": "per sample and gene: reads, median, quartiles and mean tail",
    "tails_transcript_sample": "per sample and transcript with enough reads: reads and median tail",
    "tails_gene_within_unit": "per unit and gene: change, moderated t over the replicates, q over the unit's genes, "
                              "and the descriptive read-level test",
    "tails_gene_by_unit": "per unit and gene: the change that enters the test across units",
    "tails_gene_across_units": "per gene: moderated one-sample t over units; q over genes",
    "tails_transcript_within_unit": "per unit and transcript: as for genes",
    "tails_transcript_across_units": "per transcript structure (intron chain): as for genes",
    "tails_category_sample": "per sample and transcript category: read-level and transcript-level median tail",
    "tails_categories": "per unit and category: tail per condition, change and its test over the replicates",
    "tails_category_tests": "categories and conditions compared over units",
    "tails_within_gene_isoforms": "per unit and gene: NMD or retained-intron isoform against the coding isoform",
    "tails_within_gene_by_unit": "per unit: medians over genes of the within-gene differences",
    "tails_within_gene_tests": "over units: tests of the per-unit medians",
    "tails_thresholds": "per sample and gene set: reads, median tail, share of short tails, share in the window",
    "tails_thresholds_by_unit": "per unit and gene set: each measure per condition and its test over the replicates",
    "tails_threshold_tests": "over units: treated against control",
    "tails_histograms": "per sample and gene set: tail histogram",
    "tails_gene_classes": "per unit and gene class: median change of the class and of all genes",
    "tails_gene_class_tests": "over units: class against all genes",
    "sites": "every poly(A) cluster of every run: position, class, region and reads per condition",
    "sites_sample_counts": "reads per poly(A) cluster and sample",
    "apa_tandem_within_unit": "per unit and gene: proximal and distal sites, PDUI per condition, delta PDUI, test, "
                              "same-exon check, distance from the stop codon, models ending at each site",
    "apa_tandem_across_units": "per gene: moderated one-sample t of the per-unit delta PDUI over units",
    "apa_class_within_unit": "per unit, gene and class (composite exon, skipped exon, internal exonic, 5'UTR): share "
                             "against terminal sites, test, main site",
    "apa_class_across_units": "per gene and class: moderated one-sample t of the per-unit delta share over units",
    "apa_ipa_within_unit": "all intronic sites against terminal sites, per unit and gene",
    "apa_ipa_across_units": "all intronic sites against terminal sites, across units",
    "tails_by_site": "median tail of the reads at proximal and distal tandem sites and at each class of site",
    "tails_by_site_tests": "over units: distal minus proximal site tail",
    "sites_pas_usage": "share of reads by poly(A) signal, per unit, site class and condition",
    "transcript_usage": "CPM of each model structure of the chosen genes, per sample",
    "transcript_usage_tests": "over units: CPM, treated against control",
    "tss_sites": "every TSS cluster of every run: position, class and reads",
    "tss_sample_counts": "reads per TSS cluster and sample",
    "tss_switch_within_unit": "per unit and gene: upstream and downstream TSS, DTUI per condition, delta DTUI, test",
    "tss_switch_across_units": "per gene: moderated one-sample t of the per-unit delta DTUI over units",
    "tss_class_within_unit": "per unit, gene and class (upstream TSS, alternative first exon, internal exonic TSS): "
                             "share against the canonical first exon, test",
    "tss_class_across_units": "per gene and TSS class: moderated one-sample t of the per-unit delta share over units",
    "tests_summary": "one row per family of tests: the test, its unit of replication, n tested, n at q < alpha",
}

REFERENCES = """\
- Benjamini Y, Hochberg Y (1995). Controlling the false discovery rate. *J R Stat Soc B* 57, 289-300.
- Cochran WG (1954). Some methods for strengthening the common chi-square tests. *Biometrics* 10, 417-451.
- Mantel N, Haenszel W (1959). Statistical aspects of the analysis of data from retrospective studies of disease.
  *J Natl Cancer Inst* 22, 719-748.
- Smyth GK (2004). Linear models and empirical Bayes methods for assessing differential expression in microarray
  experiments. *Stat Appl Genet Mol Biol* 3, Article 3.
- Ritchie ME, et al. (2015). limma powers differential expression analyses for RNA-sequencing and microarray studies.
  *Nucleic Acids Res* 43, e47.
- Phipson B, et al. (2016). Robust hyperparameter estimation protects against hypervariable genes and improves power
  to detect differential expression. *Ann Appl Stat* 10, 946-963.
- Xia Z, et al. (2014). Dynamic analyses of alternative polyadenylation from RNA-seq reveal a 3'-UTR landscape across
  seven tumour types. *Nat Commun* 5, 5274.
- Tian B, Pan Z, Lee JY (2007). Widespread mRNA polyadenylation events in introns indicate dynamic interplay between
  polyadenylation and splicing. *Genome Res* 17, 156-165.
- Pardo-Palacios FJ, et al. (2024). SQANTI3: curation of long-read transcriptomes for accurate identification of known
  and novel isoforms. *Nat Methods* 21, 793-797.
- Frankish A, et al. (2023). GENCODE: reference annotation for the human and mouse genomes in 2023. *Nucleic Acids
  Res* 51, D942-D949.
"""

STATISTICS = """\
**Units of replication.** Reads are never the unit of replication. Within a unit (one cell line, donor or
experiment) the replicates are its samples: replicate pairs when the sample sheet pairs each control sample with a
treated sample, otherwise the samples of each condition. Across units, each unit gives one value.

**Poly(A) tails.** Per sample, the tail of a gene or transcript is the median over its reads, and the sample's tail
is the median of its per-transcript medians. Within a unit, a gene's change is tested with limma's moderated t-test
(empirical Bayes, robust): one-sample on the paired differences, or two-sample on the sample medians. Across units,
the per-unit change is tested with the moderated one-sample t-test. Sample tails are compared with a paired or Welch
t-test within a unit, and summarised across units by the mean change with a percentile bootstrap 95% CI. Categories,
gene classes, within-gene isoform differences and short-tail shares are compared over units with Wilcoxon
signed-rank tests paired by unit.

**Site usage (APA and TSS).** A site's usage is the number of reads that end or start there. Within a unit, a paired
design is tested with the Cochran-Mantel-Haenszel test on the tables of each replicate pair, and an unpaired design
with the moderated two-sample t-test on each sample's usage. Across units, the per-unit change (delta PDUI, delta DTUI
or delta share) is tested with the moderated one-sample t-test.

**Families.** q is the Benjamini-Hochberg-adjusted P within the family named in tests_summary. Missing P values are
not counted.
"""


def _fmt(v, digits=3):
    if isinstance(v, (bool, np.bool_)):
        return str(bool(v))
    if isinstance(v, (int, np.integer)):
        return f"{int(v):,}"
    if isinstance(v, (float, np.floating)):
        if not np.isfinite(v):
            return ""
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        if abs(v) < 1e-3 and v != 0:
            return f"{v:.2e}"
        return f"{v:.{digits}g}"
    return "" if v is None or v is pd.NA else str(v)


def md_table(df: pd.DataFrame, max_rows: int = 60) -> str:
    if df is None or df.empty:
        return "_none_\n"
    d = df.head(max_rows)
    head = "| " + " | ".join(map(str, d.columns)) + " |\n| " + " | ".join("---" for _ in d.columns) + " |\n"
    body = "".join("| " + " | ".join(_fmt(v).replace("|", "/").replace("\n", " ") for v in r) + " |\n"
                   for r in d.itertuples(index=False))
    more = f"\n_{len(df) - max_rows:,} more rows in the table file._\n" if len(df) > max_rows else ""
    return head + body + more


def _versions() -> dict:
    out = {"python": sys.version.split()[0], "isoends": __version__, "platform": platform.platform()}
    for pkg in ("numpy", "pandas", "scipy", "pyarrow", "matplotlib", "pyyaml", "pysam", "isotracks"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def _counts(t: pd.DataFrame, effect: str, q: str = "q", alpha: float = 0.05) -> tuple[int, int]:
    """(n up, n down) at q < alpha."""
    sig = t[q] < alpha
    return int((sig & (t[effect] > 0)).sum()), int((sig & (t[effect] < 0)).sum())


def _per_unit(t: pd.DataFrame, effect: str) -> pd.DataFrame:
    """Per unit: n tested and the counts each way at q < 0.05."""
    return pd.DataFrame([dict(unit=u, n=len(d), up=_counts(d, effect)[0], down=_counts(d, effect)[1])
                         for u, d in t.groupby("unit", sort=False)]).set_index("unit")


def headlines(T, info) -> list[str]:
    """One line per result, from the tables."""
    out = []
    trt = info["treatment"]
    multi = len(info["units"]) > 1
    a, w = T("tails_sample_across_units"), T("tails_sample_within_unit")
    if len(a) and len(w) > 1:
        r = a.iloc[0]
        out.append(f"**Sample tails**: shorter with {trt} in {int(r.n_units_shorter)} of {int(r.n_units)} units "
                   f"({int(r.n_units_shorter_q_below_alpha)} at q < 0.05); mean change {r.mean_delta_tail_nt:+.1f} nt "
                   f"(bootstrap 95% CI {r.ci95_low_nt:+.1f} to {r.ci95_high_nt:+.1f}).")
    elif len(w):
        r = w.iloc[0]
        out.append(f"**Sample tail** ({r.unit}): {r.median_tail_control_nt:.1f} to {r.median_tail_treated_nt:.1f} nt, "
                   f"change {r.delta_tail_nt:+.1f} nt (P = {r.p:.2g}).")
    g = T("tails_gene_across_units")
    if len(g):
        up, down = _counts(g, "mean_delta_tail_nt")
        out.append(f"**Gene tails across units**: {len(g):,} genes tested; {down:,} shorter and {up:,} longer at "
                   f"q < 0.05 (median change {g.mean_delta_tail_nt.median():+.1f} nt).")
    gw = T("tails_gene_within_unit")
    if len(gw):
        per = _per_unit(gw, "delta_tail_nt")
        if len(per) == 1:
            r = per.iloc[0]
            out.append(f"**Gene tails within {per.index[0]}**: {int(r.n):,} genes; {int(r.down):,} shorter and "
                       f"{int(r.up):,} longer at q < 0.05.")
        else:
            out.append(f"**Gene tails within units**: median {int(per.n.median()):,} genes per unit; shorter at "
                       f"q < 0.05 in a median of {int(per.down.median()):,} per unit, longer in "
                       f"{int(per.up.median()):,}.")
    t = T("apa_tandem_across_units")
    if len(t):
        sig = t[t.q < 0.05]
        dist, prox = sig[sig.mean_delta_pdui > 0], sig[sig.mean_delta_pdui < 0]
        out.append(f"**Tandem 3'UTR across units**: {len(t):,} genes; {len(dist):,} distal and {len(prox):,} proximal "
                   f"at q < 0.05; median 3'UTR length gained {dist.length_gained_nt.median():+.0f} nt for distal and "
                   f"{prox.length_gained_nt.median():+.0f} nt for proximal shifts.")
    tw = T("apa_tandem_within_unit")
    if len(tw) and tw.is_tandem.any():
        per = _per_unit(tw[tw.is_tandem], "delta_pdui")
        r = per.median() if len(per) > 1 else per.iloc[0]
        where = "units (median per unit)" if len(per) > 1 else per.index[0]
        if info["params"]["apa"].get("same_exon_check"):
            check = f"{int((~tw.is_tandem).sum()):,} candidate pairs excluded by the same-exon check."
        else:
            split = tw.same_exon.eq(False).fillna(False) | (tw.share_distal_models_split > 0.5)
            check = (f"{int(split.sum()):,} pairs that an intron splits are flagged (same_exon, "
                     "share_distal_models_split) and kept: the same-exon check is off.")
        out.append(f"**Tandem 3'UTR within {where}**: {int(r.n):,} genes; {int(r.up):,} distal and {int(r.down):,} "
                   f"proximal at q < 0.05; {check}")
    for key, label, col in (("apa_class", "APA classes", "apa_class"), ("apa_ipa", "All intronic sites", "apa_class"),
                            ("tss_class", "TSS classes", "tss_class")):
        across, within = T(f"{key}_across_units"), T(f"{key}_within_unit")
        src, effect = (across, "mean_delta_share") if len(across) else (within, "delta_share")
        if len(src):
            parts = [f"{cls} {len(d):,} ({_counts(d, effect)[0]:,} up, {_counts(d, effect)[1]:,} down)"
                     for cls, d in src.groupby(col)]
            where = "across units" if len(across) else "within units, the units' tests pooled" if multi \
                else f"within {within.unit.iat[0]}"
            out.append(f"**{label} {where}**: " + "; ".join(parts) + " genes tested, q < 0.05.")
    s = T("tss_switch_across_units")
    if len(s):
        up, down = _counts(s, "mean_delta_dtui")
        out.append(f"**Promoter switch across units**: {len(s):,} genes; {up:,} towards the downstream and {down:,} "
                   "towards the upstream TSS at q < 0.05.")
    th = T("tails_thresholds_by_unit")
    if len(th):
        for gset, d in th[th.measure.str.startswith("share_below")].groupby("gene_set"):
            out.append(f"**{gset.replace('_', ' ')}, reads below {info['params']['tails']['short_nt']:g} nt**: "
                       f"{d.control.median() * 100:.1f}% to {d.treated.median() * 100:.1f}% (median over units), up "
                       f"in {int((d.treated > d.control).sum())} of {len(d)} units.")
    return out


def _design_line(info) -> str:
    units = info["units"]
    kinds = sorted({u["design"] for u in units.values()})
    n = pd.DataFrame(info["samples"])
    return (f"Design: {info['treatment']} against {info['control']} (control); {len(units)} unit(s), "
            f"{len(n)} samples, {' and '.join(kinds)} replicates. "
            + (f"Tests across units need a feature in at least {info.get('min_units')} units."
               if len(units) > 1 else "One unit: every test is over its replicates."))


def write_report(cfg, hash_inputs: bool = True) -> Path:
    """Write tests_summary, the figures, provenance.json and report.md for cfg.outdir. Returns the report path."""
    from .pipeline import read_table, write_table
    outdir = Path(cfg.outdir)
    info_path = outdir / "logs" / "run_info.json"
    if not info_path.exists():
        raise SystemExit(f"{info_path} not found: run `isoends tails`, `apa` or `tss` first")
    info = json.loads(info_path.read_text())
    cache: dict = {}

    def T(name):
        if name not in cache:
            cache[name] = read_table(outdir, name)
        return cache[name]

    ts = pd.concat([T(f"tests_summary_{stage}") for stage in ("tails", "apa", "tss")], ignore_index=True)
    write_table(ts, outdir, "tests_summary")
    figs = []
    if cfg.figures.enabled:
        from .figures import make_figures
        LOG.info("drawing figures")
        figs = make_figures(cfg, outdir)

    files = {"config": info.get("config"), "sample_sheet": info.get("sample_sheet"),
             "reference_gtf": info.get("reference_gtf")}
    for run, fs in info["runs"].items():
        files.update({f"{run}:{k}": v for k, v in fs.items() if k != "bams"})
    inputs = {}
    for k, v in files.items():
        if not v or v == "None" or not Path(v).exists():
            continue
        p = Path(v)
        st = p.stat()
        inputs[k] = {"path": str(p), "bytes": st.st_size,
                     "modified": datetime.datetime.fromtimestamp(st.st_mtime).isoformat(timespec="seconds")}
        if hash_inputs:
            LOG.info("hashing %s", p.name)
            inputs[k]["sha256"] = sha256(p, cfg.cache_dir)
    versions = _versions()
    estimators = sorted(set(ts.get("estimator", pd.Series(dtype=str)).dropna()) - {""})
    if any(e.startswith("R limma") for e in estimators):
        from .stats.limma import r_limma_available, r_limma_version
        exe = r_limma_available(cfg.stats.rscript)
        if exe:
            versions["R_limma"] = r_limma_version(exe)
    prov = dict(isoends_version=__version__, date=datetime.datetime.now().isoformat(timespec="seconds"),
                command=info.get("command"), config=info.get("config"), inputs=inputs, versions=versions,
                parameters=info["params"], seed=cfg.stats.seed, min_units=info.get("min_units"),
                limma_estimators=estimators, stages=info.get("stages"), host=platform.node(),
                figures=[f.name for f in figs])
    atomic_write_json(prov, outdir / "provenance.json")

    steps = []
    for f in sorted((outdir / "logs").glob("*_steps.json")):
        stage = f.name.removesuffix("_steps.json")
        steps += [dict(stage=stage, **r) for r in json.loads(f.read_text())["records"]]
    steps = pd.DataFrame(steps)
    samples = pd.DataFrame(info["samples"])
    val = T("validation")
    lines = [f"# IsoEnds report: {info['name']}", "",
             f"IsoEnds {__version__}, {prov['date']}, `{info.get('command', '')}`", "", _design_line(info), "",
             "## Results", ""]
    lines += [f"- {h}" for h in headlines(T, info)] or ["- (no results)"]
    lines += ["", "## Validation", ""]
    if len(val):
        bad = val[val.status != "ok"]
        lines += [f"{len(bad)} warning(s) or error(s) of {len(val)} checks:" if len(bad)
                  else f"All {len(val)} checks passed.", "", md_table(bad if len(bad) else val.head(20), 80)]
    rows = [dict(input=k, path=v["path"], size_MB=round(v["bytes"] / 1e6, 1), sha256=v.get("sha256", "")[:16])
            for k, v in inputs.items()]
    lines += ["## Inputs", "", md_table(pd.DataFrame(rows), 200), "Full SHA-256 digests are in provenance.json.", "",
              "### Sample sheet", "", md_table(samples, 400)]
    if info.get("excluded"):
        lines += [f"Datasets left out: {', '.join(info['excluded'])}", ""]
    sync = info.get("sync", {})
    if sync:
        srows = [dict(run=k, used=v.get("in_sync"), reads_on_present_models=v.get("share_reads_present"),
                      chain_agreement=v.get("share_chain_agreement"),
                      reads_on_characterised_models=v.get("share_reads_characterised")) for k, v in sync.items()]
        lines += ["### Characterised GTF", "", md_table(pd.DataFrame(srows), 100),
                  next((v.get("rule", "") for v in sync.values()), ""), ""]
    lines += ["## Parameters", "", "```yaml",
              yaml.safe_dump(json.loads(json.dumps(info["params"])), sort_keys=False).strip(), "```", "",
              "## Counts at each step", ""]
    if len(steps):
        lines += [md_table(steps[steps.step != "WARNING"][["stage", "unit", "step", "n"]], 400)]
        warn = steps[steps.step == "WARNING"]
        if len(warn):
            lines += ["Warnings:", ""] + [f"- {r.unit}: {r.message}" for r in warn.itertuples()] + [""]
    lines += ["## Tests", "", "One row per family of tests adjusted together.", "",
              md_table(ts.drop(columns=[c for c in ("units", "analysis") if c in ts]), 400), "## Figures", ""]
    if not figs:
        lines += ["_No figures were drawn._", ""]
    for f in figs:
        png = next((p for p in f.paths if str(p).endswith(".png")), f.paths[0] if f.paths else "")
        rel = Path(png).relative_to(outdir) if png else ""
        lines += [f"### {f.name}: {f.title}", "", f"![{f.title}]({rel})", "", f.caption, ""]
        if getattr(f, "audit", None):
            lines += ["_Style audit: " + "; ".join(f.audit) + "_", ""]
    lines += ["## Statistics", "", STATISTICS, "## Output tables", "",
              "In `tables/`, as Parquet and TSV, values unrounded:", ""]
    for name, doc in TABLE_DOCS.items():
        if (outdir / "tables" / f"{name}.parquet").exists():
            only = "" if (outdir / "tables" / f"{name}.tsv").exists() else ", Parquet only"
            lines.append(f"- `{name}` ({len(T(name)):,} rows{only}): {doc}")
    lines += ["", "## References", "", REFERENCES]
    path = outdir / "report.md"
    path.write_text("\n".join(lines) + "\n")
    LOG.info("wrote %s", path)
    return path

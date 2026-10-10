"""Orchestration: validate the inputs, summarise each run, run the analyses and write their tables.

The stages `tails`, `apa` and `tss` share one context (the cached per-sample summaries and the annotations) and
write their tables under <outdir>/tables. `report` draws the figures and writes report.md and provenance.json from
those tables.
"""
from __future__ import annotations

import json
import os
import re
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from . import __version__
from .annotation import run_transcripts
from .config import Config, SampleSheet, read_sample_sheet
from .gtf import reference_annotation, terminal_exons
from .ingest import RunCache, ingest_run, parquet_datasets
from .stats import family_row
from .util import LOG, StepLog, atomic_write_json, atomic_write_parquet, combined_fingerprint

TSV_MAX_ROWS = 2_000_000


@dataclass
class Context:
    cfg: Config
    sheet: SampleSheet
    caches: dict
    steps: StepLog
    validation: list = field(default_factory=list)
    transcripts: dict = field(default_factory=dict)        # run -> transcript table
    sync: dict = field(default_factory=dict)               # run -> check of the characterised GTF
    ref: dict | None = None
    gene_names: pd.Series = field(default_factory=pd.Series)
    gene_types: pd.Series = field(default_factory=pd.Series)
    gene_class_members: dict = field(default_factory=dict)
    model_exons: dict = field(default_factory=dict)        # run -> exons of its IsoForge models
    terminal: dict = field(default_factory=dict)           # run -> terminal exons of its models
    qc: pd.DataFrame = field(default_factory=pd.DataFrame)


def _check(rows: list, scope: str, check: str, value, status: str, message: str = "") -> None:
    rows.append(dict(scope=scope, check=check, value=value, status=status, message=message))
    if status in ("error", "warning"):
        LOG.log(40 if status == "error" else 30, "%s: %s: %s", scope, check, message)


def validate_inputs(cfg: Config) -> tuple[SampleSheet, list]:
    """Check the sample sheet and the input files before any read is read. Returns the sheet and the checks; exits
    with every error found."""
    rows: list = []
    errors = []
    sheet, notes = read_sample_sheet(cfg.samples, cfg.control, cfg.treatment, list(cfg.runs))
    for note in notes:
        _check(rows, "sample sheet", "design", None, "warning", note)
    for unit, design in sheet.design_names.items():
        d = sheet.design(unit)
        _check(rows, unit, "design", design, "ok", f"{len(d.control)} {cfg.control} and {len(d.treated)} "
               f"{cfg.treatment} samples" + (f" in {len(d.pairs)} pairs" if d.paired else ""))
    for name, run in cfg.runs.items():
        for key, path in run.input_files().items():
            if not Path(path).exists():
                errors.append(f"run {name}: {key} {path} not found")
        if not Path(run.read_annot).exists():
            continue
        in_parquet = set(parquet_datasets(run.read_annot))
        listed = set(sheet.run_datasets(name))
        missing = sorted(listed - in_parquet)
        unlisted = sorted(in_parquet - listed - set(sheet.excluded))
        if missing:
            errors.append(f"run {name}: sample sheet dataset(s) not in the read annotation: {', '.join(missing)}")
        _check(rows, name, "datasets in the read annotation", len(in_parquet), "error" if missing else "ok",
               f"missing {missing}" if missing else "")
        if unlisted:
            _check(rows, name, "datasets not in the sample sheet", len(unlisted), "warning",
                   f"left out: {', '.join(unlisted)}")
    if cfg.reference_gtf is None:
        _check(rows, "reference", "reference GTF", None, "warning",
               "none given: the APA and TSS analyses need it and are skipped")
    elif not Path(cfg.reference_gtf).exists():
        errors.append(f"reference GTF {cfg.reference_gtf} not found")
    if errors:
        for e in errors:
            _check(rows, "inputs", "error", None, "error", e)
        raise SystemExit("IsoEnds: input validation failed:\n  " + "\n  ".join(errors))
    return sheet, rows


def _ingest_one(args):
    """Summarise one run and cache its transcript annotation; returns (run, cache folder)."""
    cfg, sheet, run, threads = args
    from .util import setup_logging
    setup_logging()
    cache = ingest_run(cfg, sheet, run, threads=threads)
    _run_transcripts(cfg, run, cache)
    return run, cache.folder


def build_context(cfg: Config, jobs: int = 1, stage: str = "run") -> Context:
    """Validate the inputs, summarise every run (cached) and read the annotations."""
    sheet, rows = validate_inputs(cfg)
    steps = StepLog(cfg.outdir, stage)
    runs = [r for r in cfg.runs if r in set(sheet.table.run)]
    threads = max(2, min(8, (os.cpu_count() or 8) // max(1, jobs)))
    todo = [(cfg, sheet, r, threads) for r in runs]
    if jobs > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=min(jobs, len(todo))) as pool:
            folders = dict(pool.map(_ingest_one, todo))
    else:
        folders = dict(_ingest_one(args) for args in todo)
    caches = {run: RunCache(folders[run], sheet.run_datasets(run)) for run in runs}
    ctx = Context(cfg=cfg, sheet=sheet, caches=caches, steps=steps, validation=rows)

    qc_rows = []
    for run, cache in caches.items():
        index = cache.meta().get("tail_index")
        if index:
            _check(rows, run, "tail source rows without a tail call", index["n_no_call"], "ok",
                   f"{index['n_no_call']:,} of {index['n_rows']:,} rows; {index['n_duplicates']:,} duplicate rows "
                   "ignored")
        for d in cache.datasets:
            q = cache.qc(d)
            rate = q["n_tail_call"] / max(q["n_reads"], 1)
            _check(rows, d, "reads with a tail call", q["n_tail_call"], "ok" if rate > 0.5 else "warning",
                   f"{rate:.1%} of {q['n_reads']:,} reads; {q.get('n_tail_used', q['n_tail_call']):,} enter the tail "
                   f"analyses (source: {q['tail_source']})")
            n_supported = q.get("n_supported", q["n_reads"])
            _check(rows, d, "supported reads", n_supported, "ok" if n_supported else "warning",
                   f"{n_supported / max(q['n_reads'], 1):.1%} of {q['n_reads']:,} reads: cluster_supported, with "
                   "their transcript in the IsoForge GTF")
            _check(rows, d, "reads without a gene or model", q["n_no_model"],
                   "ok" if q["n_no_model"] == 0 else "warning",
                   f"{q['n_no_model']:,} reads without annot_transcript_id; {q['n_no_gene']:,} without a gene")
            _check(rows, d, "3' ends with A-rich downstream sequence", q["n_high_fraction_as"], "ok",
                   f"{q['n_high_fraction_as']:,} reads with fraction_As > {cfg.apa.internal_priming_fraction_as}")
            novelty = q.pop("novelty", {}) or {}
            qc_rows.append(dict(run=run, **q, **{f"novelty_{k}": v for k, v in novelty.items()}))
    ctx.qc = pd.DataFrame(qc_rows)
    steps.add("samples", len(sheet.table))
    steps.add("units", len(sheet.units))

    if cfg.reference_gtf is not None:
        LOG.info("reading the reference annotation")
        ctx.ref = reference_annotation(cfg.reference_gtf, cfg.cache_dir)
        genes = ctx.ref["genes"].set_index("gene_id")
        ctx.gene_names = genes.gene_name.astype(str)
        ctx.gene_types = genes.gene_type.astype(str)
    for run, cache in caches.items():
        tx, sync, exons = _run_transcripts(cfg, run, cache)
        ctx.transcripts[run] = tx
        ctx.sync[run] = sync
        ctx.model_exons[run] = exons
        ctx.terminal[run] = terminal_exons(exons)
        if sync.get("characterised_gtf"):
            ok = sync.get("in_sync")
            _check(rows, run, "characterised GTF describes this run", ok, "ok" if ok else "warning",
                   f"{sync.get('share_reads_present', np.nan):.1%} of reads on models present in it; "
                   f"{sync.get('share_chain_agreement', np.nan):.1%} of shared models with identical exon chains; "
                   f"{sync.get('share_reads_characterised', np.nan):.1%} of reads on characterised models"
                   + ("" if ok else "; novel models are left unclassified"))
        else:
            _check(rows, run, "characterised GTF", None, "warning",
                   "none: novel models are unclassified; annotated models keep their reference type")
    ctx.gene_names, ctx.gene_types = _isoforge_genes(ctx)
    ctx.gene_class_members = resolve_gene_classes(cfg.gene_classes, ctx.gene_names, base=cfg.source,
                                                  types=ctx.gene_types)
    for k, v in ctx.gene_class_members.items():
        steps.add(f"gene class {k}: genes", len(v))
    return ctx


def _isoforge_genes(ctx: Context) -> tuple[pd.Series, pd.Series]:
    """Gene names and types, with the genes the reference lacks added from the IsoForge models. Without a reference
    a gene takes the type of its annotated transcripts (protein_coding when any is); a gene outside the reference is
    "novel", "novel lncRNA" or "novel protein_coding"."""
    names, types = ctx.gene_names, ctx.gene_types
    if not ctx.transcripts:
        return names, types
    tx = pd.concat(ctx.transcripts.values(), ignore_index=True)
    own = tx[["gene_id", "gene_name"]].dropna().drop_duplicates("gene_id").set_index("gene_id").gene_name.astype(str)
    own = own[~own.index.isin(names.index)]
    names = pd.concat([names, own]) if len(names) else own
    t = tx[["gene_id", "transcript_type", "annotated"]].dropna(subset=["gene_id"])
    t = t.assign(transcript_type=t.transcript_type.fillna("").astype(str))
    t = t[~t.gene_id.isin(types.index)]
    if t.empty:
        return names, types
    coding = t[t.transcript_type == "protein_coding"].groupby("gene_id").size()
    first = t.drop_duplicates("gene_id").set_index("gene_id").transcript_type
    gene_type = first.where(~first.index.isin(coding.index), "protein_coding")
    if ctx.ref is not None:
        gene_type = gene_type.map(lambda v: f"novel {v}" if v in ("lncRNA", "protein_coding") else "novel")
    return names, pd.concat([types, gene_type]) if len(types) else gene_type


def _run_transcripts(cfg: Config, run: str, cache: RunCache):
    """The run's transcript annotation, its check against the characterised GTF and the model exons, cached."""
    r = cfg.runs[run]
    key = combined_fingerprint([r.gtf, r.characterised_gtf], extra=f"tx-v3|{cache.folder.name}")
    folder = Path(cfg.cache_dir) / "runs" / run / f"transcripts_{key}"
    if (folder / "sync.json").exists():
        return (pd.read_parquet(folder / "transcripts.parquet"), json.loads((folder / "sync.json").read_text()),
                pd.read_parquet(folder / "exons.parquet"))
    counts = []
    for d in cache.datasets:
        m = cache.load(d, "model_reads")
        counts.append(m.groupby(m.transcript.astype(str), observed=True).n_reads.sum())
    LOG.info("[%s] reading the IsoForge models%s", run, " and the characterised GTF" if r.characterised_gtf else "")
    tx, sync, exons = run_transcripts(r.gtf, r.characterised_gtf, pd.concat(counts, axis=1).sum(axis=1) if counts else None)
    atomic_write_parquet(tx, folder / "transcripts.parquet")
    atomic_write_parquet(exons, folder / "exons.parquet")
    atomic_write_json(sync, folder / "sync.json")
    return tx, sync, exons


def resolve_gene_classes(spec: dict, names: pd.Series, base: Path | None = None,
                         types: pd.Series | None = None) -> dict:
    """{class: set of gene IDs}. A class is {"regex": on gene names}, {"genes": [IDs or names]} or {"gmt": path,
    "set": name}, optionally restricted to one {"gene_type": ...}."""
    out = {}
    by_name = pd.Series(names.index, index=names.values)
    for cls, s in (spec or {}).items():
        ids = set()
        if not isinstance(s, dict):
            s = {"genes": list(s)}
        if s.get("regex"):
            rx = re.compile(s["regex"])
            ids |= set(names.index[names.astype(str).map(lambda x: bool(rx.search(x)))])
        genes = list(s.get("genes", []) or [])
        if s.get("gmt"):
            gmt = Path(s["gmt"]) if Path(s["gmt"]).is_absolute() or base is None else Path(base).parent / s["gmt"]
            for line in gmt.read_text().splitlines():
                f = line.rstrip("\n").split("\t")
                if len(f) > 2 and (s.get("set") in (None, f[0])):
                    genes += f[2:]
        for g in genes:
            g = str(g)
            if g.startswith("ENSG"):
                ids.add(g.split(".")[0])
            elif g in by_name.index:
                v = by_name.loc[g]
                ids |= set(v.tolist() if isinstance(v, pd.Series) else [v])
        if s.get("gene_type") and types is not None and len(types):
            ids = {g for g in ids if types.get(g) == s["gene_type"]}
        out[cls] = ids
    return out


def write_table(df: pd.DataFrame, outdir: Path, name: str) -> Path:
    """Write <outdir>/tables/<name>.parquet and .tsv, values unrounded. The TSV is skipped above TSV_MAX_ROWS."""
    folder = Path(outdir) / "tables"
    folder.mkdir(parents=True, exist_ok=True)
    df = (df if df is not None else pd.DataFrame()).copy()
    for c in df.columns:
        if isinstance(df[c].dtype, pd.CategoricalDtype):
            df[c] = df[c].astype(str)
        elif df[c].dtype == object:
            kinds = {type(v) for v in df[c].dropna()}
            if kinds and kinds <= {bool, np.bool_}:
                df[c] = df[c].astype("boolean")
            elif len(kinds) > 1 or (kinds and not kinds <= {str}):
                df[c] = df[c].map(lambda v: v if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
    atomic_write_parquet(df.reset_index(drop=True), folder / f"{name}.parquet")
    if len(df) <= TSV_MAX_ROWS:
        df.to_csv(folder / f"{name}.tsv", sep="\t", index=False, float_format="%.10g")
    else:
        LOG.info("%s: %s rows, written as Parquet only", name, f"{len(df):,}")
        (folder / f"{name}.tsv").unlink(missing_ok=True)
    return folder / f"{name}.parquet"


def read_table(outdir: Path, name: str) -> pd.DataFrame:
    p = Path(outdir) / "tables" / f"{name}.parquet"
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def families_table(families: list, tables: dict, alpha: float) -> pd.DataFrame:
    """One row per family of tests: the test, its unit of replication and the counts at q < alpha."""
    rows = []
    for f in families:
        t = tables.get(f["table"], pd.DataFrame())
        if len(t) and f.get("filter"):
            for k, v in f["filter"].items():
                t = t[t[k] == v]
        q = f.get("q", "q")
        if not len(t) or q not in t or f["effect"] not in t:
            rows.append(dict(family=f["family"], analysis=f["analysis"], test=f["test"],
                             unit_of_replication=f["unit_of_replication"], n_tested=0, n_q_below_alpha=0, n_up=0,
                             n_down=0, alpha=alpha, units="", note="not tested", estimator=""))
            continue
        units = ", ".join(map(str, sorted(t.unit.astype(str).unique()))) if "unit" in t else ""
        row = family_row(f["family"], f["analysis"], f["test"], f["unit_of_replication"], t, f["effect"], q,
                         alpha, units=units, note=f.get("note", ""))
        row["estimator"] = f.get("estimator", "")
        rows.append(row)
    return pd.DataFrame(rows)


def _run_info(cfg: Config) -> dict:
    path = Path(cfg.outdir) / "logs" / "run_info.json"
    return json.loads(path.read_text()) if path.exists() else {}


def _remove_tables(cfg: Config, names) -> None:
    for name in names:
        for ext in ("parquet", "tsv"):
            (Path(cfg.outdir) / "tables" / f"{name}.{ext}").unlink(missing_ok=True)


def _finish_stage(ctx: Context, stage: str, tables: dict, families: list) -> None:
    """Write a stage's tables and its tests summary, remove the tables its previous run wrote and this one did not,
    and record the run (design, inputs, parameters, checks) for the report."""
    from .apa_run import usage_test_name
    from .tails import SAMPLE_TEST, TAIL_TEST, test_name
    cfg, sheet = ctx.cfg, ctx.sheet
    info = _run_info(cfg)
    written = dict(tables, **{f"tests_summary_{stage}": families_table(families, tables, cfg.stats.alpha)})
    _remove_tables(cfg, set(info.get("tables", {}).get(stage, [])) - set(written))
    for name, table in written.items():
        write_table(table, cfg.outdir, name)
    ctx.steps.save()
    designs = {u: sheet.design(u) for u in sheet.units}
    kinds = sheet.design_names.values()
    info.update(
        tests=dict(tails=test_name(TAIL_TEST, kinds), sample=test_name(SAMPLE_TEST, kinds),
                   usage=usage_test_name(kinds)),
        isoends_version=__version__, name=cfg.name, config=str(cfg.source) if cfg.source else None,
        command=" ".join(sys.argv), control=cfg.control, treatment=cfg.treatment,
        samples=sheet.table.to_dict("records"), excluded=sheet.excluded,
        units={u: dict(design="paired" if d.paired else "unpaired", control=list(d.control), treated=list(d.treated),
                       pairs=[label for label, _, _ in d.pairs]) for u, d in designs.items()},
        runs={k: {kk: str(vv) for kk, vv in v.input_files().items()} | {"bams": v.bam_pattern}
              for k, v in cfg.runs.items()},
        reference_gtf=str(cfg.reference_gtf) if cfg.reference_gtf else None, sample_sheet=str(cfg.samples),
        params=cfg.params(), min_units=cfg.min_units(len(sheet.units)), sync=ctx.sync,
        gene_class_sizes={k: len(v) for k, v in ctx.gene_class_members.items()},
        cache={k: str(v.folder) for k, v in ctx.caches.items()})
    info.setdefault("tables", {})[stage] = sorted(written)
    info.setdefault("stages", {})[stage] = pd.Timestamp.now().isoformat(timespec="seconds")
    atomic_write_json(info, Path(cfg.outdir) / "logs" / "run_info.json")
    write_table(pd.DataFrame(ctx.validation), cfg.outdir, "validation")
    if len(ctx.qc):
        write_table(ctx.qc.merge(sheet.table[["dataset", "unit", "condition", "pair"]], on="dataset", how="left"),
                    cfg.outdir, "qc_samples")


def _run_stage(cfg: Config, stage: str, jobs: int, ctx: Context | None, needs_reference: bool) -> dict:
    from .apa_run import analyse_apa
    from .tails import analyse_tails
    from .tss_run import analyse_tss
    ctx = ctx or build_context(cfg, jobs, stage)
    ctx.steps = StepLog(cfg.outdir, stage)
    if needs_reference and ctx.ref is None:
        LOG.warning("no reference GTF: the %s analysis is skipped", stage)
        return {}
    res = {"tails": analyse_tails, "apa": analyse_apa, "tss": analyse_tss}[stage](ctx)
    _finish_stage(ctx, stage, res.tables, res.families)
    return res.tables


def run_tails(cfg: Config, jobs: int = 1, ctx: Context | None = None) -> dict:
    return _run_stage(cfg, "tails", jobs, ctx, needs_reference=False)


def run_apa(cfg: Config, jobs: int = 1, ctx: Context | None = None) -> dict:
    return _run_stage(cfg, "apa", jobs, ctx, needs_reference=True)


def run_tss(cfg: Config, jobs: int = 1, ctx: Context | None = None) -> dict:
    return _run_stage(cfg, "tss", jobs, ctx, needs_reference=True)


def run_report(cfg: Config, hash_inputs: bool = True) -> Path:
    from .report import write_report
    return write_report(cfg, hash_inputs=hash_inputs)


def run_all(cfg: Config, jobs: int = 1, hash_inputs: bool = True) -> Path:
    """Every stage, then the report. With the TSS analysis disabled, the TSS tables of an earlier run are removed."""
    ctx = build_context(cfg, jobs, "ingest")
    ctx.steps.save()
    run_tails(cfg, jobs, ctx)
    run_apa(cfg, jobs, ctx)
    if cfg.tss.enabled:
        run_tss(cfg, jobs, ctx)
    else:
        info = _run_info(cfg)
        _remove_tables(cfg, info.get("tables", {}).pop("tss", []))
        atomic_write_json(info, Path(cfg.outdir) / "logs" / "run_info.json")
    return run_report(cfg, hash_inputs=hash_inputs)

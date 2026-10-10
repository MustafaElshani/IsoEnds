"""Read an IsoForge run one sample at a time and keep only summaries.

For each sample (`dataset` in the read annotation) the reads are reduced to small tables, cached as Parquet:

  tail_hist       gene, transcript, tail (nt), n_reads, n_reads_mapq   reads with a tail call, counted per tail value
  model_reads     gene, gene_name, transcript, n_reads, n_reads_supported, n_reads_tail   reads per model
  sites           gene, chrom, strand, site, feature, signal, ...      reads per poly(A) site
  site_tail_hist  gene, site, tail, n_reads                            tails of the reads at each site
  tss_sites       gene, chrom, strand, site, feature, n_reads          reads per transcription start site

Medians, quartiles and shares of any grouping follow exactly from the per-tail counts, so no read is held after its
sample is summarised.

A supported read is one IsoForge keeps in its filtered read table: `cluster_supported` is true and its transcript is
reported in the IsoForge GTF.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from .config import Config, RunInputs, SampleSheet
from .gtf import reported_transcripts
from .tailsource import TailIndex, build_index_from_bams, build_index_from_table, uuid_pairs
from .util import LOG, atomic_write_json, atomic_write_parquet, combined_fingerprint

DICT_COLUMNS = ["annot_gene_id", "annot_gene_name", "annot_transcript_id", "chromosome", "strand", "polya_feature",
                "polya_signal", "transcript_novelty"]
KINDS = ("tail_hist", "model_reads", "sites", "site_tail_hist")
INGEST_VERSION = "3"

BASE_COLUMNS = ["annot_gene_id", "annot_gene_name", "annot_transcript_id", "chromosome", "strand", "read_start",
                "read_end", "polya_site", "polya_feature", "polya_signal", "cluster_supported", "fraction_As",
                "transcript_novelty"]


@dataclass
class IngestParams:
    tails_require_supported: bool
    min_tail_nt: int
    sites_require_supported: bool
    site_fallback_read_end: bool
    internal_priming_fraction_as: float
    mapq_min: int
    tail_cap_nt: int

    @staticmethod
    def from_config(cfg: Config) -> "IngestParams":
        return IngestParams(cfg.tails.require_supported, cfg.tails.min_tail_nt, cfg.apa.require_supported,
                            cfg.apa.site_fallback_read_end, cfg.apa.internal_priming_fraction_as,
                            cfg.tails.mapq_min, cfg.tails.tail_cap_nt)


def parquet_datasets(path: Path) -> list[str]:
    """The distinct `dataset` values of a read annotation, from the row-group statistics where they suffice."""
    pf = pq.ParquetFile(path)
    col = pf.schema_arrow.names.index("dataset")
    seen: set = set()
    for i in range(pf.metadata.num_row_groups):
        st = pf.metadata.row_group(i).column(col).statistics
        if st is not None and st.has_min_max and st.min == st.max:
            seen.add(str(st.min))
            continue
        values = pf.read_row_group(i, columns=["dataset"]).column("dataset")
        seen.update(str(v) for v in pc.unique(values).to_pylist() if v is not None)
    return sorted(seen)


def parquet_columns(path: Path) -> list[str]:
    return pq.ParquetFile(path).schema_arrow.names


def _recode_categories(cat: pd.Categorical | pd.Series, func) -> pd.Categorical:
    """Apply `func` to the categories of a categorical, merging categories that become equal."""
    c = pd.Categorical(cat)
    new = pd.Index(func(pd.Series(c.categories.astype(str))))
    uniq, inverse = np.unique(new.to_numpy(dtype=str), return_inverse=True)
    codes = np.where(c.codes >= 0, inverse[np.maximum(c.codes, 0)], -1)
    return pd.Categorical.from_codes(codes, categories=pd.Index(uniq))


def _with_blank(values) -> pd.Categorical:
    """A categorical with missing values as the empty string."""
    c = pd.Categorical(values)
    if "" not in c.categories:
        c = c.add_categories([""])
    return c.fillna("")


def _column(tbl: pa.Table, name: str, n: int):
    """A column as a pandas Categorical (dictionary columns) or NumPy array; None when absent."""
    if name not in tbl.column_names:
        return None
    col = tbl.column(name)
    if pa.types.is_dictionary(col.type):
        return pd.Categorical(col.to_pandas())
    return col.to_numpy(zero_copy_only=False) if col.null_count == 0 else col.to_pandas().to_numpy()


def _fill_category(c, value: str) -> pd.Categorical:
    c = pd.Categorical(c)
    if value not in c.categories:
        c = c.add_categories([value])
    return c.fillna(value)


def _supported(tbl: pa.Table, tx: pd.Categorical, n: int, reported) -> tuple[np.ndarray, np.ndarray]:
    """(supported, flagged): reads with `cluster_supported` whose transcript is among `reported`, and the reads with
    the flag alone. A missing column or reported=None leaves that condition out."""
    flag = _column(tbl, "cluster_supported", n)
    flagged = np.ones(n, bool) if flag is None else pd.Series(flag).fillna(False).to_numpy(dtype=bool)
    if reported is None:
        return flagged, flagged
    return flagged & pd.Index(tx.categories.astype(str)).isin(reported)[tx.codes], flagged


def ingest_sample(read_annot: Path, dataset: str, tail_index: TailIndex | None, params: IngestParams,
                  columns: list[str], reported=None) -> tuple[dict[str, pd.DataFrame], dict]:
    """One sample's summaries and QC counts. Tails come from `tail_index` when given, else from the read
    annotation's `polya_length`. reported: the transcript IDs of the IsoForge GTF, which a supported read needs."""
    has_len = tail_index is None and "polya_length" in columns
    want = [c for c in BASE_COLUMNS if c in columns] + (["polya_length"] if has_len else [])
    if tail_index is not None:
        want = ["read_name"] + want
    fmt = ds.ParquetFileFormat(read_options={"dictionary_columns": [c for c in DICT_COLUMNS if c in want]})
    tbl = ds.dataset(read_annot, format=fmt).to_table(columns=want, filter=ds.field("dataset") == dataset)
    tbl = tbl.unify_dictionaries()
    n = tbl.num_rows
    if has_len:
        tl = tbl.column("polya_length")
        tail = np.asarray(pc.cast(tl, pa.float64()).fill_null(-1).to_numpy(zero_copy_only=False))
        tail = np.where(np.isnan(tail), -1, np.round(tail)).astype(np.int64)
        mapq = np.full(n, -1, np.int16)
    elif tail_index is not None:
        hi, lo, valid = uuid_pairs(tbl.column("read_name"))
        t16, mapq = tail_index.lookup(hi, lo)
        tail = np.where(valid, t16, -1).astype(np.int64)
        del hi, lo, valid, t16
        tbl = tbl.drop_columns(["read_name"])
    else:
        tail = np.full(n, -1, np.int64)
        mapq = np.full(n, -1, np.int16)

    gene_raw = _column(tbl, "annot_gene_id", n)
    gene = _recode_categories(_with_blank(gene_raw if gene_raw is not None else [""] * n),
                              lambda s: s.str.replace(r"\.\d+$", "", regex=True))
    tx_raw = _column(tbl, "annot_transcript_id", n)
    tx = _with_blank(tx_raw if tx_raw is not None else [""] * n)
    blank_gene = gene.codes == gene.categories.get_loc("")
    blank_tx = tx.codes == tx.categories.get_loc("")
    supported, flagged = _supported(tbl, tx, n, reported)
    has_tail = tail >= params.min_tail_nt
    tail16 = np.minimum(np.maximum(tail, 0), params.tail_cap_nt).astype(np.int16)
    hq = mapq >= params.mapq_min

    in_tails = has_tail & ~blank_gene & (supported if params.tails_require_supported else True)
    th = pd.DataFrame({"gene": gene[in_tails], "transcript": tx[in_tails], "tail": tail16[in_tails],
                       "hq": hq[in_tails]})
    tail_hist = (th.groupby(["gene", "transcript", "tail"], observed=True)
                 .agg(n_reads=("hq", "size"), n_reads_mapq=("hq", "sum")).reset_index())
    del th
    model_reads = (pd.DataFrame({"gene": gene, "transcript": tx, "has_tail": has_tail, "supported": supported})
                   .groupby(["gene", "transcript"], observed=True)
                   .agg(n_reads=("has_tail", "size"), n_reads_supported=("supported", "sum"),
                        n_reads_tail=("has_tail", "sum")).reset_index())
    names = _column(tbl, "annot_gene_name", n)
    if names is not None:
        _, first = np.unique(gene.codes, return_index=True)
        name_of = pd.Series(np.asarray(names[first], dtype=object), index=gene.categories[gene.codes[first]])
        model_reads["gene_name"] = model_reads.gene.astype(str).map(name_of).fillna("").astype(str)
    else:
        model_reads["gene_name"] = ""

    site = _column(tbl, "polya_site", n)
    site = np.asarray(site, dtype=float) if site is not None else np.full(n, np.nan)
    strand = _column(tbl, "strand", n)
    if params.site_fallback_read_end:
        plus = np.asarray(strand == "+")
        end3 = np.where(plus, _column(tbl, "read_end", n), _column(tbl, "read_start", n)).astype(float)
        site = np.where(np.isnan(site) | (site <= 0), end3, site)
    usable = ~np.isnan(site) & (site > 0) & ~blank_gene
    if params.sites_require_supported:
        usable &= supported
    frac_a = _column(tbl, "fraction_As", n)
    frac_a = np.asarray(frac_a, dtype=float) if frac_a is not None else np.full(n, np.nan)
    feature = _column(tbl, "polya_feature", n)
    signal = _column(tbl, "polya_signal", n)
    chrom = _column(tbl, "chromosome", n)
    s = pd.DataFrame({"gene": gene[usable], "chrom": chrom[usable], "strand": strand[usable],
                      "site": site[usable].astype(np.int64),
                      "feature": _fill_category(feature[usable] if feature is not None else [None] * int(usable.sum()),
                                                "none"),
                      "signal": _fill_category(signal[usable] if signal is not None else [None] * int(usable.sum()),
                                               "none"),
                      "frac_a": frac_a[usable], "high_a": frac_a[usable] > params.internal_priming_fraction_as,
                      "has_tail": has_tail[usable]})
    sites = (s.groupby(["gene", "chrom", "strand", "site", "feature", "signal"], observed=True)
             .agg(n_reads=("has_tail", "size"), n_reads_tail=("has_tail", "sum"), sum_fraction_as=("frac_a", "sum"),
                  n_high_a=("high_a", "sum")).reset_index())
    keep = usable & has_tail
    site_tail_hist = (pd.DataFrame({"gene": gene[keep], "site": site[keep].astype(np.int64), "tail": tail16[keep]})
                      .groupby(["gene", "site", "tail"], observed=True).size().rename("n_reads").reset_index())
    del s

    novelty = _column(tbl, "transcript_novelty", n)
    qc = dict(dataset=dataset, n_reads=int(n), n_tail_call=int(has_tail.sum()), n_tail_used=int(in_tails.sum()),
              n_no_gene=int(blank_gene.sum()),
              n_no_model=int(blank_tx.sum()), n_site=int(usable.sum()), n_cluster_supported=int(flagged.sum()),
              n_supported=int(supported.sum()),
              n_high_fraction_as=int((frac_a > params.internal_priming_fraction_as).sum()),
              n_no_signal=int(pd.isna(signal).sum()) if signal is not None else None,
              n_mapq_known=int((mapq >= 0).sum()), n_mapq_hq=int(hq.sum()),
              tail_median_all_reads=float(np.median(tail[has_tail])) if has_tail.any() else None,
              tail_source=("tail table or BAMs" if tail_index is not None else "polya_length" if has_len
                           else "none"))
    if novelty is not None:
        qc["novelty"] = {str(k): int(v) for k, v in pd.Series(novelty).value_counts().items()}
    out = {"tail_hist": tail_hist, "model_reads": model_reads, "sites": sites, "site_tail_hist": site_tail_hist}
    for v in out.values():
        for c in v.columns:
            if isinstance(v[c].dtype, pd.CategoricalDtype):
                v[c] = v[c].cat.remove_unused_categories()
    return out, qc


TSS_COLUMNS = ["annot_gene_id", "annot_transcript_id", "chromosome", "strand", "read_start", "read_end", "tss_site",
               "tss_feature", "cluster_supported"]


def ingest_tss_sample(read_annot: Path, dataset: str, require_supported: bool, columns: list[str],
                      reported=None) -> pd.DataFrame:
    """Reads per transcription start site in one sample: gene, chrom, strand, site, feature, n_reads. With
    require_supported, supported reads at IsoForge's `tss_site`; without it, every read, at its own 5' end when it
    has no `tss_site`."""
    want = [c for c in TSS_COLUMNS if c in columns]
    fmt = ds.ParquetFileFormat(read_options={"dictionary_columns": [c for c in DICT_COLUMNS + ["tss_feature"]
                                                                     if c in want]})
    tbl = ds.dataset(read_annot, format=fmt).to_table(columns=want, filter=ds.field("dataset") == dataset)
    tbl = tbl.unify_dictionaries()
    n = tbl.num_rows
    gene_raw = _column(tbl, "annot_gene_id", n)
    gene = _recode_categories(_with_blank(gene_raw if gene_raw is not None else [""] * n),
                              lambda s: s.str.replace(r"\.\d+$", "", regex=True))
    strand = _column(tbl, "strand", n)
    site = _column(tbl, "tss_site", n)
    site = np.asarray(site, dtype=float) if site is not None else np.full(n, np.nan)
    if not require_supported:
        plus = np.asarray(strand == "+")
        end5 = np.where(plus, _column(tbl, "read_start", n), _column(tbl, "read_end", n)).astype(float)
        site = np.where(np.isnan(site) | (site <= 0), end5, site)
    usable = ~np.isnan(site) & (site > 0) & (gene.codes != gene.categories.get_loc(""))
    if require_supported:
        tx_raw = _column(tbl, "annot_transcript_id", n)
        usable &= _supported(tbl, _with_blank(tx_raw if tx_raw is not None else [""] * n), n, reported)[0]
    feature = _column(tbl, "tss_feature", n)
    d = pd.DataFrame({"gene": gene[usable], "chrom": _column(tbl, "chromosome", n)[usable], "strand": strand[usable],
                      "site": site[usable].astype(np.int64),
                      "feature": _fill_category(feature[usable] if feature is not None
                                                else [None] * int(usable.sum()), "none")})
    out = d.groupby(["gene", "chrom", "strand", "site", "feature"], observed=True).size().rename("n_reads").reset_index()
    for c in out.columns:
        if isinstance(out[c].dtype, pd.CategoricalDtype):
            out[c] = out[c].cat.remove_unused_categories()
    return out


def ensure_tss(cache: "RunCache", run: RunInputs, require_supported: bool) -> None:
    """Add the 5'-end counts to a run's cache where missing."""
    todo = [d for d in cache.datasets if not cache.tss_path(d, require_supported).exists()]
    if not todo:
        return
    columns = parquet_columns(run.read_annot)
    reported = reported_transcripts(run.gtf)
    for d in todo:
        atomic_write_parquet(ingest_tss_sample(run.read_annot, d, require_supported, columns, reported),
                             cache.tss_path(d, require_supported))


class RunCache:
    """The cached per-sample summaries of one run."""

    def __init__(self, folder: Path, datasets: list[str]):
        self.folder = Path(folder)
        self.datasets = list(datasets)

    def path(self, dataset: str, kind: str) -> Path:
        return self.folder / "samples" / dataset / f"{kind}.parquet"

    def tss_path(self, dataset: str, require_supported: bool = True) -> Path:
        return self.folder / "samples" / dataset / f"tss_sites_{'supported' if require_supported else 'all'}.parquet"

    def load_tss(self, dataset: str, require_supported: bool = True) -> pd.DataFrame:
        return pd.read_parquet(self.tss_path(dataset, require_supported))

    def load(self, dataset: str, kind: str) -> pd.DataFrame:
        return pd.read_parquet(self.path(dataset, kind))

    def qc(self, dataset: str) -> dict:
        return json.loads((self.folder / "samples" / dataset / "qc.json").read_text())

    def meta(self) -> dict:
        p = self.folder / "run.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def complete(self, dataset: str) -> bool:
        return all(self.path(dataset, k).exists() for k in KINDS) and (
            self.folder / "samples" / dataset / "qc.json").exists()


def run_cache_folder(cfg: Config, run: RunInputs) -> Path:
    params = IngestParams.from_config(cfg)
    tails_src = run.tails or (run.tail_bams[0] if run.tail_bams else None)
    extra = json.dumps({"v": INGEST_VERSION, "params": asdict(params), "tail_columns": run.tail_columns,
                        "bams": [str(b) for b in run.tail_bams]}, sort_keys=True)
    fp = combined_fingerprint([run.read_annot, tails_src, run.gtf], extra)
    return Path(cfg.cache_dir) / "runs" / run.name / fp


def ingest_run(cfg: Config, sheet: SampleSheet, run_name: str, threads: int | None = None) -> RunCache:
    """Summarise every sample of one run into the cache; cached samples are skipped. `threads` caps Arrow's
    thread pool."""
    if threads:
        pa.set_cpu_count(max(1, threads))
        pa.set_io_thread_count(max(1, threads))
    run: RunInputs = cfg.runs[run_name]
    params = IngestParams.from_config(cfg)
    folder = run_cache_folder(cfg, run)
    datasets = sheet.run_datasets(run_name)
    cache = RunCache(folder, datasets)
    todo = [d for d in datasets if not cache.complete(d)]
    if not todo:
        LOG.info("[%s] summaries cached in %s", run_name, folder)
        return cache
    folder.mkdir(parents=True, exist_ok=True)
    columns = parquet_columns(run.read_annot)
    reported = reported_transcripts(run.gtf)
    meta = cache.meta()

    index = None
    if run.tails is not None or run.tail_bams:
        idx_path = folder / "tail_index.parquet"
        if idx_path.exists():
            index = TailIndex.load(idx_path)
        else:
            LOG.info("[%s] reading per-read tails from %s", run_name, run.tails or "the BAMs")
            index = (build_index_from_table(run.tails, run.tail_columns) if run.tails is not None
                     else build_index_from_bams(run.tail_bams))
            index.save(idx_path)
        LOG.info("[%s] tail index: %s reads with a call (%s rows read, %s without a call, %s duplicate rows)",
                 run_name, f"{len(index):,}", f"{index.n_rows:,}", f"{index.n_no_call:,}", f"{index.n_duplicates:,}")
        meta["tail_index"] = dict(source=index.source, n_reads=len(index), n_rows=index.n_rows,
                                  n_no_call=index.n_no_call, n_duplicates=index.n_duplicates)
    elif "polya_length" in columns:
        LOG.info("[%s] tails from the read annotation's polya_length column", run_name)
    else:
        LOG.warning("[%s] no tails: the read annotation has no polya_length and no tail table or BAMs are "
                    "configured; the poly(A) tail analyses will be empty for this run", run_name)

    for d in todo:
        out, qc = ingest_sample(run.read_annot, d, index, params, columns, reported)
        for kind, table in out.items():
            atomic_write_parquet(table, cache.path(d, kind))
        atomic_write_json(qc, folder / "samples" / d / "qc.json")
        LOG.info("[%s] %s: %s reads, %.1f%% with a tail call, %s at poly(A) sites", run_name, d,
                 f"{qc['n_reads']:,}", 100 * qc["n_tail_call"] / max(qc["n_reads"], 1), f"{qc['n_site']:,}")
    meta.update(run=run_name, datasets=datasets, read_annot=str(run.read_annot), params=asdict(params),
                columns=columns)
    atomic_write_json(meta, folder / "run.json")
    return cache

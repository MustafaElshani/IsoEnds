"""Configuration: the inputs, the sample sheet and the analysis parameters.

A run is described by a YAML file (`isoends init` writes a template). Every parameter is recorded in the provenance
of each run.
"""
from __future__ import annotations

import copy
import math
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

import pandas as pd
import yaml

TOP_LEVEL_KEYS = ("name", "outdir", "cache_dir", "samples", "control", "treatment", "reference", "runs", "params",
                  "figures", "gene_classes", "usage_genes")
EXCLUDE = "exclude"


@dataclass
class TailParams:
    require_supported: bool = True        # only supported reads: cluster_supported, transcript in the IsoForge GTF
    min_tail_nt: int = 0                  # shortest length counted as a tail call; a read without a call has -1
    min_reads_gene: int = 20              # reads per sample for a gene to be tested
    min_reads_transcript: int = 20        # reads per sample for a transcript to be tested
    min_reads_transcript_sample: int = 21  # reads per sample for a transcript to enter the sample's tail (> 20)
    min_reads_isoform: int = 10           # reads per condition for each isoform of a within-gene comparison
    min_reads_category_transcript: int = 10  # reads per sample for a transcript to enter its category's tail
    min_replicates: int = 2               # replicate pairs, or samples per condition, a feature needs
    mapq_min: int = 1                     # mapping quality of the sensitivity analysis (tail tables with MAPQ)
    short_nt: float = 27                  # tails below this are counted as short
    window_nt: tuple = (30, 60)           # tail window reported per gene set
    hist_bin_nt: int = 3
    hist_max_nt: int = 300
    tail_cap_nt: int = 4095               # longer tails are counted at this length
    threshold_sets: tuple = ("ribosomal_proteins",)  # gene classes that get the short-tail measures
    read_level: bool = True               # add the descriptive read-level test columns


@dataclass
class ApaParams:
    cluster_nt: int = 25                  # sites closer than this (chained, per gene) form one cluster
    require_supported: bool = True        # only supported reads: cluster_supported, transcript in the IsoForge GTF
    site_fallback_read_end: bool = False  # use the read's 3' end when polya_site is missing
    min_reads: int = 20                   # reads per condition for a gene to be tested
    min_reads_sample: int = 10            # unpaired designs: reads per sample for its usage to enter the test
    downstream_nt: int = 5000             # nt past the canonical 3' end still counted as terminal
    model_window_nt: int = 25             # nt between a model's 3' end and a site for the model to end there
    tandem_sites: str = "three_prime_utr"  # "three_prime_utr" (IsoForge's label of the site) or "terminal"
    #                                       (the canonical last exon and up to `downstream_nt` past it)
    same_exon_check: bool = False         # exclude site pairs that no single annotated exon holds
    match_tolerance_nt: int = 50          # nt within which sites of different units are the same site
    across_units_same_pair: bool = False  # across units, use only the units whose site pair matches the reference
    intronic_models: str = "any"          # models that make an intronic site composite or skipped: "any" (every
    #                                       model ending there on the chromosome and strand) or "gene" (the gene's)
    novel_gene_models: bool = False       # place the sites of a gene outside the reference on its best-supported
    #                                       IsoForge model; without it they are not classified
    internal_priming_fraction_as: float = 0.5  # fraction_As above which a 3' end is counted as A-rich
    site_model_window_nt: int = 100       # nt for reporting the best-supported model ending at a site


@dataclass
class TssParams:
    enabled: bool = True
    require_supported: bool = True        # only supported reads, at IsoForge's tss_site
    cluster_nt: int = 50                  # 5' ends closer than this (chained, per gene) form one cluster
    upstream_nt: int = 1000               # nt upstream of the canonical first exon still counted as that exon
    max_upstream_nt: int = 50_000         # further upstream: not assigned to the gene
    model_window_nt: int = 50             # nt between a model's 5' end and a TSS for the model to start there
    min_reads: int = 20                   # reads per condition for a gene to be tested
    min_reads_sample: int = 10            # unpaired designs: reads per sample for its usage to enter the test
    match_tolerance_nt: int = 50
    across_units_same_pair: bool = False  # across units, use only the units whose TSS pair matches the reference
    novel_gene_models: bool = False       # as for the poly(A) sites
    exclude_chroms: tuple = ("chrM", "MT")


@dataclass
class StatsParams:
    alpha: float = 0.05
    min_units: int | None = None          # units a feature needs across units; None: max(2, 2/3 of the units)
    limma_backend: str = "auto"           # "auto" (R limma when available, else Python), "python" or "r"
    robust: bool = True                   # eBayes(robust = TRUE)
    rscript: str | None = None
    bootstrap: int = 10_000
    seed: int = 1
    wilcoxon_min_units: int = 5


@dataclass
class FigureParams:
    enabled: bool = True
    journal: str = "cell"                 # column widths: "cell" or "nature"
    formats: tuple = ("pdf", "svg", "png")
    dpi: int = 600
    genes: tuple = ()                     # genes named in volcanoes and drawn with IsoTracks
    isotracks: bool = True                # draw `genes` with IsoTracks when BAMs are configured
    isotracks_unit: str | None = None     # unit to draw them in (default: the first)
    n_label: int = 8                      # genes named per volcano


DEFAULT_GENE_CLASSES = {
    "ribosomal_proteins": {"regex": r"(?i)^(RP[LS]\d+[A-Z]?\d*|RPLP[0-2]|RPSA)$", "gene_type": "protein_coding",
                           "description": "cytoplasmic ribosomal-protein genes (protein coding)"},
    "snoRNA_host_genes": {"regex": r"(?i)^SNHG\d+$", "description": "snoRNA host genes (SNHG*)"},
    "histones": {"regex": r"(?i)^(H1-\d|H2AC\d+|H2BC\d+|H3C\d+|H4C\d+|H2AW|H2BU1|H3-3[AB]|H4-16)$",
                 "gene_type": "protein_coding", "description": "histone genes"},
    "mitochondrial": {"regex": r"(?i)^MT-", "description": "mitochondrially encoded genes"},
}


def _merge(base: dict, update: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (update or {}).items():
        out[k] = _merge(out.get(k, {}), v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def _build(cls, values: dict | None):
    values = dict(values or {})
    unknown = set(values) - {f.name for f in fields(cls)}
    if unknown:
        raise ValueError(f"unknown {cls.__name__} parameter(s): {', '.join(sorted(unknown))}")
    for f in fields(cls):
        if f.name in values and isinstance(f.default, tuple) and isinstance(values[f.name], list):
            values[f.name] = tuple(values[f.name])
    return cls(**values)


@dataclass
class RunInputs:
    """The files of one IsoForge run."""
    name: str
    read_annot: Path
    gtf: Path
    characterised_gtf: Path | None = None
    bed: Path | None = None
    tes_clusters: Path | None = None
    tss_clusters: Path | None = None
    tails: Path | None = None             # per-read tail table, used instead of polya_length
    tail_bams: tuple = ()                 # or BAMs carrying the pt:i tag
    bam_pattern: str | None = None        # "{dataset}" -> genome BAM, for IsoTracks
    tail_columns: dict = field(default_factory=lambda: dict(read_id="read_id", tail="ePolyA_tail_length",
                                                            mapq="mapping_quality"))

    def bam(self, dataset: str) -> Path | None:
        return Path(self.bam_pattern.format(dataset=dataset)) if self.bam_pattern else None

    def input_files(self) -> dict:
        out = {k: getattr(self, k) for k in ("read_annot", "gtf", "characterised_gtf", "bed", "tes_clusters",
                                               "tss_clusters", "tails")}
        out.update({f"tail_bam_{i}": p for i, p in enumerate(self.tail_bams)})
        return {k: v for k, v in out.items() if v is not None}


def _first(folders, *patterns: str) -> Path | None:
    for folder in folders:
        for pattern in patterns:
            hits = sorted(folder.glob(pattern))
            if hits:
                return hits[0]
    return None


def discover_run(name: str, spec: dict, base: Path) -> RunInputs:
    """Resolve a run's files from its IsoForge `results` folder and any explicit paths in `spec`. The characterised
    GTF and BED of isoforge-characterize are looked for in the results folder and in a sibling SQANTI3 folder."""
    spec = dict(spec)

    def path(v):
        if v is None:
            return None
        p = Path(str(v)).expanduser()
        return p if p.is_absolute() else (base / p)

    res = path(spec.pop("results", None))
    found = {}
    if res is not None:
        if not res.is_dir():
            raise FileNotFoundError(f"run {name}: results folder {res} not found")
        read_annot = _first([res], "*_isoforge_read_annot.parquet")
        if read_annot is None:
            raise FileNotFoundError(f"run {name}: no *_isoforge_read_annot.parquet in {res}")
        prefix = read_annot.name.removesuffix("_isoforge_read_annot.parquet")
        clusters = _first([res], f"{prefix}_TSS_TES_Clusters", "*_TSS_TES_Clusters")
        sqanti = [res] + [d for d in (res.parent / "SQANTI3", res.parent / "sqanti3") if d.is_dir()]
        stems = (f"{prefix}_isoforge_sqanti3_withCDS", f"{prefix}_isoforge_sqanti3", "*_isoforge_sqanti3_withCDS",
                 "*_isoforge_sqanti3")

        def cluster_file(file):
            return clusters / file if clusters and (clusters / file).exists() else None
        found = dict(read_annot=read_annot, gtf=_first([res], f"{prefix}_isoforge.gtf"),
                     characterised_gtf=_first(sqanti, *(f"{s}.gtf" for s in stems)),
                     bed=_first(sqanti, *(f"{s}.bed" for s in stems)),
                     tes_clusters=cluster_file("tes_clusters.bed"), tss_clusters=cluster_file("tss_clusters.bed"))
    for key in ("read_annot", "gtf", "characterised_gtf", "bed", "tes_clusters", "tss_clusters", "tails"):
        if key in spec:
            found[key] = path(spec.pop(key))
    tail_bams = tuple(path(b) for b in spec.pop("tail_bams", []) or [])
    bam_pattern = spec.pop("bams", None)
    if bam_pattern is not None:
        bam_pattern = str(path(bam_pattern))
    tail_columns = RunInputs(name, Path("."), Path(".")).tail_columns
    tail_columns.update(spec.pop("tail_columns", {}) or {})
    if spec:
        raise ValueError(f"run {name}: unknown key(s) {', '.join(sorted(spec))}")
    if found.get("read_annot") is None or found.get("gtf") is None:
        raise ValueError(f"run {name}: needs read_annot and gtf, or a results folder holding them")
    return RunInputs(name=name, tail_bams=tail_bams, bam_pattern=bam_pattern, tail_columns=tail_columns, **found)


@dataclass
class Config:
    name: str
    outdir: Path
    cache_dir: Path
    samples: Path
    control: str
    treatment: str
    reference_gtf: Path | None
    runs: dict
    tails: TailParams
    apa: ApaParams
    stats: StatsParams
    tss: TssParams
    figures: FigureParams
    gene_classes: dict
    usage_genes: tuple = ()
    source: Path | None = None

    def params(self) -> dict:
        """Every analysis parameter, for the provenance and the report."""
        return {"tails": asdict(self.tails), "apa": asdict(self.apa), "tss": asdict(self.tss),
                "stats": asdict(self.stats), "figures": asdict(self.figures), "gene_classes": self.gene_classes,
                "usage_genes": list(self.usage_genes), "control": self.control, "treatment": self.treatment}

    def min_units(self, n_units: int) -> int:
        if self.stats.min_units is not None:
            return int(self.stats.min_units)
        return max(2, math.ceil(2 * n_units / 3))


def load_config(path=None, overrides: dict | None = None, data: dict | None = None) -> Config:
    """Read a YAML config, or a dict of the same shape, into a Config. Relative paths are relative to the file."""
    if data is None:
        path = Path(path)
        data = yaml.safe_load(path.read_text()) or {}
        base = path.resolve().parent
    else:
        base = Path(path).resolve().parent if path else Path.cwd()
    data = _merge(data, overrides or {})
    unknown = set(data) - set(TOP_LEVEL_KEYS)
    if unknown:
        raise ValueError(f"unknown config key(s): {', '.join(sorted(unknown))}; known: {', '.join(TOP_LEVEL_KEYS)}")
    for key in ("samples", "control", "treatment", "runs"):
        if key not in data:
            raise ValueError(f"config needs `{key}`")

    def p(v):
        if v in (None, ""):
            return None
        q = Path(str(v)).expanduser()
        return q if q.is_absolute() else (base / q)

    runs_spec = data["runs"]
    if isinstance(runs_spec, list):
        runs_spec = {r.pop("name"): r for r in (dict(x) for x in runs_spec)}
    runs = {str(k): discover_run(str(k), v or {}, base) for k, v in runs_spec.items()}
    params = data.get("params", {}) or {}
    unknown = set(params) - {"tails", "apa", "tss", "stats"}
    if unknown:
        raise ValueError(f"unknown params section(s): {', '.join(sorted(unknown))}")
    outdir = p(data.get("outdir", "isoends_out"))
    classes = copy.deepcopy(DEFAULT_GENE_CLASSES)
    classes.update(data.get("gene_classes", {}) or {})
    return Config(
        name=str(data.get("name", "isoends")), outdir=outdir, cache_dir=p(data.get("cache_dir")) or outdir / "cache",
        samples=p(data["samples"]), control=str(data["control"]), treatment=str(data["treatment"]),
        reference_gtf=p((data.get("reference", {}) or {}).get("gtf")), runs=runs,
        tails=_build(TailParams, params.get("tails")), apa=_build(ApaParams, params.get("apa")),
        stats=_build(StatsParams, params.get("stats")), tss=_build(TssParams, params.get("tss")),
        figures=_build(FigureParams, data.get("figures")), gene_classes=classes,
        usage_genes=tuple(data.get("usage_genes", []) or []), source=Path(path).resolve() if path else None)


@dataclass(frozen=True)
class Design:
    """The samples of one unit that enter its tests. Paired: `pairs` holds (label, control dataset, treated
    dataset) and `control` / `treated` are aligned with it. Unpaired: every control and treated dataset."""
    unit: str
    paired: bool
    control: tuple
    treated: tuple
    pairs: tuple = ()

    @property
    def datasets(self) -> list[str]:
        return list(self.control) + list(self.treated)

    @property
    def n_replicates(self) -> int:
        return min(len(self.control), len(self.treated))


@dataclass
class SampleSheet:
    table: pd.DataFrame                   # dataset, unit, condition, pair, run
    control: str
    treatment: str
    excluded: list = field(default_factory=list)   # datasets left out: "exclude" or another condition

    @property
    def units(self) -> list[str]:
        return list(dict.fromkeys(self.table.unit))

    def unit_table(self, unit: str) -> pd.DataFrame:
        return self.table[self.table.unit == unit]

    def design(self, unit: str) -> Design:
        t = self.unit_table(unit)
        if (t.pair == "").all():
            return Design(unit, False, tuple(t.dataset[t.condition == self.control]),
                          tuple(t.dataset[t.condition == self.treatment]))
        pairs = []
        for pair, g in t.groupby("pair", sort=False):
            c = g.dataset[g.condition == self.control].tolist()
            tr = g.dataset[g.condition == self.treatment].tolist()
            if len(c) == 1 and len(tr) == 1:
                pairs.append((str(pair), c[0], tr[0]))
        return Design(unit, True, tuple(c for _, c, _ in pairs), tuple(t_ for _, _, t_ in pairs), tuple(pairs))

    @property
    def design_names(self) -> dict:
        """{unit: "paired" or "unpaired"}."""
        return {u: "paired" if self.design(u).paired else "unpaired" for u in self.units}

    def run_datasets(self, run: str) -> list[str]:
        return self.table[self.table.run == run].dataset.tolist()

    def units_of_run(self, run: str) -> list[str]:
        return list(dict.fromkeys(self.table[self.table.run == run].unit))


def read_sample_sheet(path: Path, control: str, treatment: str, runs: list[str]) -> tuple[SampleSheet, list[str]]:
    """Read and check the sample sheet (columns dataset, unit, condition and optionally pair and run). Returns the
    sheet and a list of warnings. Datasets with another condition, or "exclude", are left out. A unit whose samples
    carry a `pair` label is paired; a unit without labels is unpaired."""
    t = pd.read_csv(path, sep=None, engine="python", dtype=str).fillna("")
    t.columns = [c.strip() for c in t.columns]
    missing = {"dataset", "unit", "condition"} - set(t.columns)
    if missing:
        raise ValueError(f"sample sheet {path} lacks column(s): {', '.join(sorted(missing))}")
    if "pair" not in t.columns:
        t["pair"] = ""
    for c in t.columns:
        t[c] = t[c].str.strip()
    notes = []
    if t.dataset.duplicated().any():
        raise ValueError(f"sample sheet: duplicated dataset(s) {t.dataset[t.dataset.duplicated()].tolist()}")
    for label in (control, treatment):
        if label not in set(t.condition):
            raise ValueError(f"sample sheet: no dataset has condition {label!r}; conditions present: "
                             f"{sorted(set(t.condition))}")
    used = t.condition.isin([control, treatment])
    excluded = t.dataset[~used].tolist()
    if excluded:
        other = sorted(set(t.condition[~used]))
        notes.append(f"sample sheet: {len(excluded)} dataset(s) with condition {', '.join(other)} are left out: "
                     + ", ".join(excluded))
    t = t[used].copy()
    if "run" not in t.columns or (t.run == "").all():
        if len(runs) == 1:
            t["run"] = runs[0]
        elif set(t.unit) <= set(runs):
            t["run"] = t.unit
        else:
            raise ValueError("several runs are configured: the sample sheet needs a `run` column, or units named "
                             "after the runs")
    unknown_runs = set(t.run) - set(runs)
    if unknown_runs:
        raise ValueError(f"sample sheet names run(s) not in the config: {sorted(unknown_runs)}")
    sheet = SampleSheet(t.reset_index(drop=True), control, treatment, excluded)
    for unit, g in t.groupby("unit", sort=False):
        if g.run.nunique() > 1:
            raise ValueError(f"unit {unit}: its datasets come from several runs ({', '.join(sorted(set(g.run)))}); "
                             "a unit must come from one IsoForge run")
        labelled = g.pair != ""
        if labelled.any() and not labelled.all():
            raise ValueError(f"unit {unit}: some datasets have a `pair` label and some do not")
        design = sheet.design(unit)
        if not design.paired:
            if not design.control or not design.treated:
                raise ValueError(f"unit {unit}: needs at least one {control} and one {treatment} dataset")
            if design.n_replicates < 2:
                notes.append(f"unit {unit}: fewer than two replicates of a condition; no test within the unit")
            continue
        for pair, gp in g.groupby("pair", sort=False):
            n = gp.condition.value_counts()
            if n.get(control, 0) > 1 or n.get(treatment, 0) > 1:
                raise ValueError(f"unit {unit}, pair {pair}: more than one sample of a condition; a pair matches one "
                                 f"{control} with one {treatment} sample")
            if n.get(control, 0) != 1 or n.get(treatment, 0) != 1:
                notes.append(f"unit {unit}, pair {pair}: incomplete pair ({', '.join(gp.dataset)}); left out of the "
                             "paired tests")
        if not design.pairs:
            raise ValueError(f"unit {unit}: no complete {control} / {treatment} pair")
    return sheet, notes

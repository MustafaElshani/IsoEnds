"""IsoEnds from Python: run an analysis, or open a finished one.

    import isoends as ie

    exp = ie.Experiment.from_config("config.yaml")        # or ie.Experiment.from_isoforge(...)
    exp.validate()                                         # the input checks as a table
    res = exp.run(jobs=4)                                  # tails, APA, TSS and the report

    res = ie.load("isoends_out")                           # a finished run
    res.analyses()                                         # the analyses and how many genes each tested
    res.get("tandem")                                      # one analysis as a table
    res.volcano("tandem", label=["GENE1"])                 # a Matplotlib Axes
    res.volcano_grid("tail_genes")                         # one volcano per unit
    res.gene("GENE1")                                      # every table's rows for one gene
    res.plot_sites("GENE1", unit="A")                      # site usage along the gene, per condition
    exp.tails_of("GENE1", unit="A")                        # the gene's tail per sample
    exp.plot_tails("GENE1", unit="A")

Tables are pandas DataFrames; plots return an Axes, or a Figure for grids.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from . import pipeline
from .config import Config, load_config


@dataclass(frozen=True)
class Analysis:
    within: str
    across: str
    effect_within: str
    effect_across: str
    xlabel: str
    up: str
    down: str
    filter: tuple = ()                # ((column, value), ...)
    within_filter: tuple = ()
    q: str = "q"
    description: str = ""


ANALYSES = {
    "tail_genes": Analysis("tails_gene_within_unit", "tails_gene_across_units", "delta_tail_nt", "mean_delta_tail_nt",
                           "Change in poly(A) tail (nt)", "longer", "shorter",
                           description="gene poly(A) tails: median per sample, moderated t over replicates"),
    "tail_genes_reads": Analysis("tails_gene_within_unit", "", "delta_tail_nt", "delta_tail_nt",
                                 "Change in poly(A) tail (nt)", "longer", "shorter", q="q_reads_pooled",
                                 description="read-level Mann-Whitney test on pooled reads; descriptive only"),
    "tail_transcripts": Analysis("tails_transcript_within_unit", "tails_transcript_across_units", "delta_tail_nt",
                                 "mean_delta_tail_nt", "Change in poly(A) tail (nt)", "longer", "shorter",
                                 description="transcript poly(A) tails; across units, per intron chain"),
    "tandem": Analysis("apa_tandem_within_unit", "apa_tandem_across_units", "delta_pdui", "mean_delta_pdui",
                       "ΔPDUI (distal 3′UTR site usage)", "distal", "proximal",
                       within_filter=(("is_tandem", True),), description="tandem 3′UTR APA (PDUI)"),
    "noncoding_tandem": Analysis("apa_tandem_within_unit", "apa_tandem_across_units", "delta_pdui", "mean_delta_pdui",
                                 "ΔPDUI (distal site usage)", "distal", "proximal", filter=(("coding", False),),
                                 within_filter=(("is_tandem", True),), q="q_noncoding",
                                 description="tandem APA in lncRNAs and novel genes (q within non-coding genes)"),
    "composite": Analysis("apa_class_within_unit", "apa_class_across_units", "delta_share", "mean_delta_share",
                          "Δ share of composite-exon sites", "up", "down", filter=(("apa_class", "composite exon"),),
                          description="intronic polyadenylation by read-through (composite terminal exon)"),
    "skipped": Analysis("apa_class_within_unit", "apa_class_across_units", "delta_share", "mean_delta_share",
                        "Δ share of skipped-exon sites", "up", "down", filter=(("apa_class", "skipped exon"),),
                        description="intronic polyadenylation on an alternative last exon (skipped terminal exon)"),
    "internal": Analysis("apa_class_within_unit", "apa_class_across_units", "delta_share", "mean_delta_share",
                         "Δ share of internal exonic sites", "up", "down", filter=(("apa_class", "internal exonic"),),
                         description="cleavage inside an internal exon"),
    "five_utr": Analysis("apa_class_within_unit", "apa_class_across_units", "delta_share", "mean_delta_share",
                         "Δ share of 5′UTR sites", "up", "down", filter=(("apa_class", "5'UTR"),),
                         description="premature cleavage upstream of the start codon (5′UTR exons and introns)"),
    "intronic": Analysis("apa_ipa_within_unit", "apa_ipa_across_units", "delta_share", "mean_delta_share",
                         "Δ share of intronic sites", "up", "down", description="all intronic polyadenylation"),
    "tss_switch": Analysis("tss_switch_within_unit", "tss_switch_across_units", "delta_dtui", "mean_delta_dtui",
                           "ΔDTUI (downstream TSS usage)", "downstream", "upstream",
                           description="promoter switch between a gene's two most used TSS"),
    "tss_upstream": Analysis("tss_class_within_unit", "tss_class_across_units", "delta_share", "mean_delta_share",
                             "Δ share of upstream TSS", "up", "down", filter=(("tss_class", "upstream TSS"),),
                             description="alternative upstream promoters"),
    "tss_first_exon": Analysis("tss_class_within_unit", "tss_class_across_units", "delta_share", "mean_delta_share",
                               "Δ share of alternative first exons", "up", "down",
                               filter=(("tss_class", "alternative first exon"),),
                               description="alternative first exons in canonical introns"),
    "tss_internal": Analysis("tss_class_within_unit", "tss_class_across_units", "delta_share", "mean_delta_share",
                             "Δ share of internal exonic TSS", "up", "down",
                             filter=(("tss_class", "internal exonic TSS"),),
                             description="5′ ends inside internal exons (downstream promoters or truncated reads)"),
}


def load(outdir) -> "Results":
    """Open the tables of a finished run."""
    return Results(outdir)


class Results:
    """The tables of a run (<outdir>/tables), with accessors and plots."""

    def __init__(self, outdir):
        self.outdir = Path(outdir)
        info = self.outdir / "logs" / "run_info.json"
        if not info.exists():
            raise FileNotFoundError(f"{info} not found: is {self.outdir} an IsoEnds output directory?")
        self.info = json.loads(info.read_text())
        self._cache: dict = {}

    def __repr__(self):
        return (f"IsoEnds results {self.info.get('name')!r} in {self.outdir}: {len(self.units)} unit(s), "
                f"{len(self.info['samples'])} samples, {self.info['control']} vs {self.info['treatment']}")

    @property
    def units(self) -> list[str]:
        return list(self.info["units"])

    @property
    def tables(self) -> list[str]:
        return sorted(p.stem for p in (self.outdir / "tables").glob("*.parquet"))

    def table(self, name: str) -> pd.DataFrame:
        """A table by its file name (see `tables`), as written."""
        if name not in self._cache:
            self._cache[name] = pipeline.read_table(self.outdir, name)
        return self._cache[name].copy()

    def summary(self) -> pd.DataFrame:
        """tests_summary: one row per family of tests."""
        return self.table("tests_summary")

    def analyses(self) -> pd.DataFrame:
        """The analyses `get` and `volcano` accept, with the genes each tested and the number at q < 0.05."""
        rows = []
        for key, a in ANALYSES.items():
            w, x = self._filtered(a, "within"), self._filtered(a, "across")
            rows.append(dict(analysis=key, description=a.description, tested_within_units=len(w),
                             q_below_0_05_within=int((w[a.q] < 0.05).sum()) if len(w) and a.q in w else 0,
                             tested_across_units=len(x),
                             q_below_0_05_across=int((x[a.q] < 0.05).sum()) if len(x) and a.q in x else 0))
        return pd.DataFrame(rows)

    def _filtered(self, a: Analysis, level: str, unit: str | None = None) -> pd.DataFrame:
        name = a.within if level == "within" else a.across
        if not name:
            return pd.DataFrame()
        t = self.table(name)
        if t.empty:
            return t
        for col, val in a.filter + (a.within_filter if level == "within" else ()):
            if col in t:
                t = t[t[col] == val]
        if unit is not None and "unit" in t:
            t = t[t.unit == unit]
        return t

    def get(self, analysis: str, level: str = "auto", unit: str | None = None) -> pd.DataFrame:
        """The result table of one analysis (see `analyses()`), with `effect` and `q` columns added. level:
        "across" (tested over units), "within" (tested over the replicates of each unit; `unit` picks one) or "auto"
        (across when there are several units). Empty when the analysis was not run."""
        a = ANALYSES[analysis]
        if level == "auto":
            level = "across" if len(self.units) > 1 and len(self._filtered(a, "across")) else "within"
        t = self._filtered(a, level, unit)
        if t.empty or a.q not in t:
            return pd.DataFrame()
        t = t.assign(effect=t[a.effect_within if level == "within" else a.effect_across], q=t[a.q])
        t.attrs.update(analysis=analysis, level=level, xlabel=a.xlabel)
        return t.sort_values("q").reset_index(drop=True)

    def gene(self, gene: str) -> dict:
        """Every table's rows for one gene (by name or unversioned Ensembl ID)."""
        out = {}
        for name in self.tables:
            t = self.table(name)
            for col in ("gene_name", "gene_id"):
                if col in t:
                    hit = t[t[col].astype(str) == gene]
                    if len(hit):
                        out[name] = hit.reset_index(drop=True)
                        break
        return out

    def volcano(self, analysis: str, level: str = "auto", unit: str | None = None, label=(), n_label: int = 8,
                alpha: float = 0.05, ax=None, ycap: float | None = None):
        """A volcano of one analysis: effect against -log10 q, the counts each way, and the strongest hits and
        `label` named. `ycap` caps -log10 q. Returns the Axes."""
        from .figures import style
        from .figures import pubfig as pf
        t = self.get(analysis, level, unit)
        style.setup(self.info.get("params", {}).get("figures", {}).get("journal", "cell"))
        if ax is None:
            _, ax = pf.subplots(1, 1, width="single", aspect=style.square_aspect(1, 1, "single"))
        a = ANALYSES[analysis]
        if t.empty:
            ax.text(0.5, 0.5, "not tested", ha="center", va="center", transform=ax.transAxes)
            return ax
        lvl = t.attrs.get("level")
        names = t.gene_name.where(t.gene_name.astype(str) != "", t.gene_id) if "gene_name" in t else None
        style.volcano(ax, t.effect, t.q, names, alpha=alpha, label=label, n_label=n_label, xlabel=a.xlabel,
                      up_text=a.up, down_text=a.down,
                      ycap=ycap if ycap is not None else (50 if lvl == "within" and "p" in t and
                                                          analysis not in ("tail_genes", "tail_transcripts")
                                                          else None))
        where = "across units" if lvl == "across" else (unit or (t.unit.iat[0] if t.unit.nunique() == 1
                                                                  else "all units"))
        pf.stat_title(ax, f"{analysis.replace('_', ' ')}, {where}: {len(t):,} genes")
        pf.finalize(ax.figure)
        return ax

    def volcano_grid(self, analysis: str, cols: int = 5, label=()):
        """One within-unit volcano per unit, on one scale. Returns the Figure."""
        from .figures import pubfig as pf
        from .figures import style
        style.setup(self.info.get("params", {}).get("figures", {}).get("journal", "cell"))
        a = ANALYSES[analysis]
        w = self._filtered(a, "within")
        units = [u for u in self.units if u in set(w.get("unit", []))]
        cols = min(cols, max(len(units), 1))
        rows = int(np.ceil(len(units) / cols)) or 1
        fig, axes = pf.subplots(rows, cols, width="double", aspect=style.square_aspect(rows, cols, shared=True),
                                squeeze=False, sharex=True, sharey=True)
        lim = style.symmetric_limits(w[a.effect_within])[1] if len(w) else 1
        cap = None if analysis.startswith("tail") else 50
        ymax = min(float(np.nanmax(-np.log10(w[a.q].clip(lower=1e-300)))), cap or np.inf) * 1.08 if len(w) else 2
        for k, ax in enumerate(axes.flat):
            if k >= len(units):
                ax.set_axis_off()
                continue
            d = w[w.unit == units[k]]
            up, dn = style.volcano(ax, d[a.effect_within], d[a.q], None, xlim=(-lim, lim), ylim=(0, max(ymax, 1.5)),
                                   point_size=2, corner_counts=False, ticks=3, ycap=cap)
            ax.set_title(f"{units[k]}  ({up:,}↑ {dn:,}↓)", fontsize=pf.sizes()["tick"] - 1, pad=2)
            ax.set_xlabel(a.xlabel if k // cols == rows - 1 else "")
            ax.set_ylabel("$-$log$_{10}$ $q$" if k % cols == 0 else "")
        return fig

    def plot_sites(self, gene: str, unit: str | None = None, kind: str = "polyA", ax=None):
        """Site usage along one gene, 5' -> 3': each cluster's share of the gene's reads per condition, labelled
        with its class. kind: "polyA" or "tss"."""
        from .figures import pubfig as pf
        from .figures import style
        style.setup(self.info.get("params", {}).get("figures", {}).get("journal", "cell"))
        sites = self.table("sites" if kind == "polyA" else "tss_sites")
        counts = self.table("sites_sample_counts" if kind == "polyA" else "tss_sample_counts")
        s = sites[(sites.gene_name == gene) | (sites.gene_id == gene)]
        if s.empty:
            raise KeyError(f"{gene}: no {kind} sites")
        unit = unit or self.units[0]
        samples = pd.DataFrame(self.info["samples"])
        ut = samples[samples.unit == unit]
        run = ut.run.iat[0]
        s = s[s.run == run]
        c = counts[(counts.run == run) & counts.cluster.isin(s.cluster) & counts.dataset.isin(ut.dataset)]
        c = c.assign(condition=c.dataset.map(ut.set_index("dataset").condition))
        by = c.groupby(["cluster", "condition"]).reads.sum().unstack(fill_value=0)
        share = by / by.sum()
        s = s.set_index("cluster").join(share, how="inner")
        s["pos"] = np.where(s.strand == "+", s.site, -s.site)
        s = s.sort_values("pos")
        if ax is None:
            _, ax = pf.subplots(1, 1, width="onehalf", aspect=0.68)
        x = np.arange(len(s))
        ctrl, trt = self.info["control"], self.info["treatment"]
        ax.bar(x - 0.2, s.get(ctrl, 0) * 100, width=0.38, color=style.CONTROL, lw=0, label=ctrl)
        ax.bar(x + 0.2, s.get(trt, 0) * 100, width=0.38, color=style.TREATED, lw=0, label=trt)
        ax.set_xticks(x, [f"{int(v):,}\n{k}" for v, k in zip(s.site, s.site_class)], fontsize=pf.sizes()["tick"] - 1)
        for tick, k in zip(ax.get_xticklabels(), s.site_class):
            tick.set_color(pf.INK["text"] if k in style.CLASS_COLORS else style.class_color(k))
        ax.set_ylabel(f"Share of {gene} reads (%)")
        ax.set_xlabel(f"{'Poly(A)' if kind == 'polyA' else 'Transcription start'} site and class, 5′ → 3′ "
                      f"({s.chrom.iat[0]}, {s.strand.iat[0]} strand)")
        ax.legend(frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=2, borderaxespad=0.2)
        pf.stat_title(ax, f"{gene} in {unit}")
        return ax

    def plot_tail_change(self, genes, ax=None):
        """The per-unit tail change of some genes: one point per unit, bar at the median."""
        from .figures import pubfig as pf
        from .figures import style
        style.setup(self.info.get("params", {}).get("figures", {}).get("journal", "cell"))
        d = self.table("tails_gene_by_unit")
        d = d[d.gene_name.isin(list(genes)) | d.gene_id.isin(list(genes))]
        if ax is None:
            _, ax = pf.subplots(1, 1, width="single", aspect=0.8)
        order = [g for g in genes if g in set(d.gene_name) | set(d.gene_id)]
        rng = np.random.default_rng(0)
        for i, g in enumerate(order):
            v = d[(d.gene_name == g) | (d.gene_id == g)].delta_tail_nt.to_numpy()
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=10, color=style.TREATED_DARK, lw=0, zorder=2)
            ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=pf.INK["text"], lw=1.2, zorder=3)
        ax.axhline(0, color=pf.INK["axis"], lw=0.5)
        ax.set_xticks(range(len(order)), order)
        pf.categorical_axis(ax)
        ax.set_ylabel("Tail change per unit (nt)")
        return ax


class Experiment:
    """An IsoEnds analysis run from Python."""

    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._ctx = None

    def __repr__(self):
        return f"IsoEnds experiment {self.cfg.name!r}: {len(self.cfg.runs)} run(s) -> {self.cfg.outdir}"

    @property
    def outdir(self) -> Path:
        return Path(self.cfg.outdir)

    @classmethod
    def from_config(cls, path, **overrides) -> "Experiment":
        """From a YAML config; keyword arguments override its top-level keys."""
        return cls(load_config(path, overrides=overrides))

    @classmethod
    def from_isoforge(cls, runs, samples, control: str, treatment: str, reference_gtf, outdir="isoends_out",
                      params: dict | None = None, figures: dict | None = None, **extra) -> "Experiment":
        """Without a YAML file. runs: {run name: IsoForge results folder, or a dict as under `runs` in the
        config}. samples: a sample-sheet path or a DataFrame (dataset, unit, condition and optionally pair, run)."""
        outdir = Path(outdir).resolve()
        if isinstance(samples, pd.DataFrame):
            outdir.mkdir(parents=True, exist_ok=True)
            path = outdir / "samples.tsv"
            samples.to_csv(path, sep="\t", index=False)
            samples = path
        data = dict(name=extra.pop("name", "isoends"), outdir=str(outdir), samples=str(Path(samples).resolve()),
                    control=control, treatment=treatment, reference={"gtf": str(Path(reference_gtf).resolve())},
                    runs={k: ({"results": str(v)} if isinstance(v, (str, Path)) else v) for k, v in runs.items()},
                    params=params or {}, figures=figures or {}, **extra)
        return cls(load_config(data=data))

    def validate(self) -> pd.DataFrame:
        """The input checks as a table; exits on errors."""
        _, rows = pipeline.validate_inputs(self.cfg)
        return pd.DataFrame(rows)

    def context(self, jobs: int = 1):
        """The shared context of the stages: validated inputs, cached summaries and annotations."""
        if self._ctx is None:
            self._ctx = pipeline.build_context(self.cfg, jobs, "ingest")
        return self._ctx

    def tails(self, jobs: int = 1) -> Results:
        pipeline.run_tails(self.cfg, jobs, self.context(jobs))
        return self.results()

    def apa(self, jobs: int = 1) -> Results:
        pipeline.run_apa(self.cfg, jobs, self.context(jobs))
        return self.results()

    def tss(self, jobs: int = 1) -> Results:
        pipeline.run_tss(self.cfg, jobs, self.context(jobs))
        return self.results()

    def report(self, figures: bool | None = None, hash_inputs: bool = True) -> Path:
        """Write the report. figures: draw them or not; None follows the configuration."""
        enabled = self.cfg.figures.enabled
        if figures is not None:
            self.cfg.figures.enabled = figures
        try:
            return pipeline.run_report(self.cfg, hash_inputs=hash_inputs)
        finally:
            self.cfg.figures.enabled = enabled

    def run(self, jobs: int = 1, report: bool = True, figures: bool | None = None) -> Results:
        """Tails, APA, TSS (unless disabled in the configuration) and the report."""
        self.tails(jobs)
        self.apa(jobs)
        if self.cfg.tss.enabled:
            self.tss(jobs)
        if report:
            self.report(figures=figures)
        return self.results()

    def results(self) -> Results:
        return Results(self.outdir)

    def tails_of(self, gene: str, unit: str | None = None, by: str = "gene") -> pd.DataFrame:
        """A gene's tail per sample, or per transcript with by="transcript": reads, median, quartiles and mean,
        exact from the cached counts."""
        from .tails import summarise_features
        ctx = self.context()
        ids = self._gene_ids(gene)
        rows = []
        for d, row in ctx.sheet.table.set_index("dataset").iterrows():
            if unit is not None and row.unit != unit:
                continue
            h = ctx.caches[row.run].load(d, "tail_hist")
            h = h[h.gene.astype(str).isin(ids)].astype({"gene": str, "transcript": str})
            if h.empty:
                continue
            keys = ["gene"] if by == "gene" else ["gene", "transcript"]
            s = summarise_features(h, keys, mapq=False).rename(columns={"n": "reads", "q50": "median_tail_nt",
                                                                        "q25": "q25_tail_nt", "q75": "q75_tail_nt",
                                                                        "mean": "mean_tail_nt"})
            rows.append(s.assign(dataset=d, unit=row.unit, condition=row.condition, pair=row.pair))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def tail_counts(self, gene: str, unit: str | None = None) -> pd.DataFrame:
        """A gene's reads per tail length, per sample."""
        ctx = self.context()
        ids = self._gene_ids(gene)
        rows = []
        for d, row in ctx.sheet.table.set_index("dataset").iterrows():
            if unit is not None and row.unit != unit:
                continue
            h = ctx.caches[row.run].load(d, "tail_hist")
            h = h[h.gene.astype(str).isin(ids)]
            rows.append(h.groupby("tail", observed=True).n_reads.sum().rename("reads").reset_index().assign(
                dataset=d, unit=row.unit, condition=row.condition))
        return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()

    def plot_tails(self, gene: str, unit: str | None = None, ax=None, max_nt: int = 300):
        """The gene's tail distribution per condition, its samples pooled, with each sample's median as a tick."""
        from .figures import pubfig as pf
        from .figures import style
        style.setup(self.cfg.figures.journal)
        c = self.tail_counts(gene, unit)
        if c.empty:
            raise KeyError(f"{gene}: no reads with a tail call")
        if ax is None:
            _, ax = pf.subplots(1, 1, width="single", aspect=0.9)
        bins = np.arange(0, max_nt + 5, 5)
        for cond, col in ((self.cfg.control, style.CONTROL_DARK), (self.cfg.treatment, style.TREATED_DARK)):
            d = c[c.condition == cond]
            h = np.histogram(np.clip(d["tail"], 0, max_nt - 1), bins, weights=d.reads)[0]
            ax.step(bins[:-1] + 2.5, h / max(h.sum(), 1) * 100, where="mid", color=col, lw=1, label=cond)
            meds = self.tails_of(gene, unit)
            for m in meds[meds.condition == cond].median_tail_nt:
                ax.axvline(m, ymin=0, ymax=0.06, color=col, lw=1.2)
        ax.set_xlim(0, max_nt)
        ax.set_xlabel("Poly(A) tail (nt)")
        ax.set_ylabel("Reads per 5-nt bin (%)")
        ax.legend(frameon=False)
        pf.stat_title(ax, f"{gene}{' in ' + unit if unit else ''}; ticks: sample medians")
        return ax

    def isotracks(self, gene: str, unit: str | None = None) -> list:
        """Draw one gene's reads with IsoTracks. Needs `bams` in the run's configuration."""
        from .figures.isotracks import draw_genes
        return draw_genes(self.cfg, [gene], unit)

    def _gene_ids(self, gene: str) -> set:
        ctx = self.context()
        if str(gene).startswith(("ENSG", "ISOFORG")):
            return {str(gene).split(".")[0]}
        return set(ctx.gene_names.index[ctx.gene_names == gene])

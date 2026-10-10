"""Draw the README's example figures with the tables of the Python API.

    python docs/make_readme_figures.py --simulate FOLDER
    python docs/make_readme_figures.py OUTDIR [--config CONFIG] [--anonymise] [--figures tails apa tss gene reads]

Writes docs/img/example_*.png. --simulate makes a simulated experiment of 10 units in FOLDER, analyses it and draws
from it; this is how the figures of the README are made. With OUTDIR the figures are drawn from a finished run of
your own; --anonymise then numbers the units and names no gene.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import isoends as ie  # noqa: E402
from isoends.synthetic import make_experiment  # noqa: E402
from isoends.figures import pubfig as pf  # noqa: E402
from isoends.figures.apa import SCHEMATIC, _schematic  # noqa: E402
from isoends.figures.style import (INK, labels_fit, square_aspect, tail_limits, tail_violins,  # noqa: E402
                                   volcano)

IMG = Path(__file__).resolve().parent / "img"
APA_PANELS = [("tandem", "tandem 3'UTR", "ΔPDUI"), ("composite", "composite exon", "Δ share"),
              ("skipped", "skipped exon", "Δ share"), ("internal", "internal exonic", "Δ share"),
              ("five_utr", "5'UTR", "Δ share"), ("noncoding_tandem", "tandem 3'UTR", "ΔPDUI, non-coding genes")]
TSS_PANELS = [("tss_switch", "TSS switch", "ΔDTUI"), ("tss_upstream", "upstream TSS", "Δ share"),
              ("tss_first_exon", "alternative first exon", "Δ share"),
              ("tss_internal", "internal exonic TSS", "Δ share")]
TITLES = {"noncoding_tandem": "tandem 3'UTR, non-coding genes", "tss_switch": "promoter switch"}
UNITS_PER_ROW = 11


class Run:
    """A finished run, with its units numbered and its genes unnamed when anonymised."""

    def __init__(self, outdir, anonymise: bool):
        self.res = ie.load(outdir)
        self.anonymise = anonymise
        self.control, self.treatment = self.res.info["control"], self.res.info["treatment"]
        self.labels = {u: (str(i + 1) if anonymise else u) for i, u in enumerate(self.res.units)}
        self.conditions = {self.control: "Control", self.treatment: "Treated"} if anonymise else \
            {self.control: self.control, self.treatment: self.treatment}

    def names(self, t: pd.DataFrame):
        if self.anonymise or "gene_name" not in t:
            return None
        return t.gene_name.where(t.gene_name.astype(str) != "", t.gene_id)


def save(fig, name: str) -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    issues = [i for i in pf.audit(fig, verbose=False)
              if not any(a in i for a in getattr(fig, "_accepted_audit", ()))]
    print(f"{name}: " + ("audit OK" if not issues else "audit: " + "; ".join(issues)))
    fig.savefig(IMG / f"{name}.png", dpi=200, facecolor="white")
    plt.close(fig)


def volcano_panel(run: Run, ax, analysis: str, xlabel: str, n_label: int = 5, **kw) -> pd.DataFrame:
    a = ie.ANALYSES[analysis]
    t = run.res.get(analysis, **kw)
    if t.empty:
        ax.set_axis_off()
        return t
    within = t.attrs["level"] == "within"
    volcano(ax, t.effect, t.q, run.names(t), n_label=n_label, xlabel=xlabel, up_text=a.up, down_text=a.down,
            ycap=50 if within and not analysis.startswith("tail") else None)
    return t


def fig_tails(run: Run) -> None:
    res = run.res
    samples = pd.DataFrame(res.info["samples"]).set_index("dataset")
    ts = res.table("tails_transcript_sample")
    ts = ts.assign(condition=ts.dataset.map(samples.condition).map(run.conditions), unit=ts.unit.map(run.labels),
                   tail=ts.median_tail_nt)
    rng = np.random.default_rng(0)
    ts = pd.concat([g.iloc[rng.choice(len(g), min(len(g), 20_000), replace=False)]
                    for _, g in ts.groupby(["unit", "condition"])])
    order = list(run.labels.values())
    rows = [order[i:i + UNITS_PER_ROW] for i in range(0, len(order), UNITS_PER_ROW)]
    per_row = max(len(r) for r in rows)
    violin_height, volcano_height = 0.36, square_aspect(1, 1, width=174 * 0.42) * 0.42
    layout = [[f"a{k}"] * 3 for k in range(len(rows))] + [[".", "b", "."]]
    fig, axd = pf.mosaic(layout, width="double", aspect=violin_height * len(rows) + volcano_height,
                         gridspec_kw={"height_ratios": [violin_height] * len(rows) + [volcano_height],
                                      "width_ratios": [0.29, 0.42, 0.29]})
    limits = tail_limits(ts["tail"])
    for k, sub in enumerate(rows):
        ax = axd[f"a{k}"]
        tail_violins(ax, ts[ts.unit.isin(sub)], "unit", sub, "condition", list(run.conditions.values()), limits,
                     legend=k == 0, slanted=not run.anonymise and not labels_fit(order, 160 / per_row))
        ax.set_xlim(-0.6, per_row - 0.4)
    ax.set_xlabel("Unit")
    t = volcano_panel(run, axd["b"], "tail_genes", "Change in poly(A) tail (nt)")
    pf.stat_title(axd["b"], f"{len(t):,} genes, {t.attrs.get('level', '')} units")
    pf.panel_labels([axd["a0"], axd["b"]])
    save(fig, "example_tails")


def fig_volcano_panels(run: Run, panels, name: str) -> None:
    cols = 2 if len(panels) <= 4 else 3
    rows = int(np.ceil(len(panels) / cols))
    layout = [x for r in range(rows) for x in ([f"v{r * cols + c}" for c in range(cols)],
                                               [f"s{r * cols + c}" for c in range(cols)])]
    fig, axd = pf.mosaic(layout, width="double", aspect=square_aspect(rows, cols, under=SCHEMATIC),
                         gridspec_kw={"height_ratios": [1, SCHEMATIC] * rows})
    drawn = []
    for k, (analysis, kind, xlabel) in enumerate(panels):
        ax = axd[f"v{k}"]
        t = volcano_panel(run, ax, analysis, xlabel, n_label=4)
        if t.empty:
            axd[f"s{k}"].set_axis_off()
            continue
        ax.set_title(TITLES.get(analysis, kind), loc="left", fontsize=pf.sizes()["label"], pad=3, color=INK)
        _schematic(axd[f"s{k}"], kind)
        drawn.append(ax)
    pf.panel_labels(drawn)
    fig._accepted_audit = ["no x-axis title", "no y-axis title"]
    save(fig, name)


def pick_gene(run: Run) -> tuple[str, str]:
    """(gene, unit) of the strongest tandem shift among well-covered genes with at most four poly(A) clusters."""
    t = run.res.get("tandem", level="within")
    t = t[(t.q < 0.01) & (t.gene_name != "")]
    if t.empty:
        return "", ""
    sites = run.res.table("sites")
    n_clusters = sites.groupby(["run", "gene_id"]).size().rename("n_clusters")
    t = t.join(n_clusters, on=["run", "gene_id"])
    few = t[t.n_clusters <= 4]
    deep = few[few.reads_control >= min(1500, few.reads_control.quantile(0.9))] if len(few) else t
    best = deep.assign(score=deep.delta_pdui.abs()).sort_values("score", ascending=False).iloc[0]
    return best.gene_name, best.unit


def fig_gene(run: Run, exp) -> None:
    gene, unit = pick_gene(run)
    if not gene:
        return
    shown = "one gene" if run.anonymise else gene
    fig, axd = pf.mosaic([["a", "b"]], width="double", aspect=square_aspect(1, 2))
    ax = run.res.plot_sites(gene, unit=unit, ax=axd["a"])
    ax.set_box_aspect(1)
    ax.set_ylabel("Share of the gene's reads (%)" if run.anonymise else ax.get_ylabel())
    if run.anonymise:
        ax.set_xticks(ax.get_xticks(), [t.get_text().split("\n")[-1] for t in ax.get_xticklabels()])
        ax.set_xlabel("Poly(A) site class, 5′ → 3′")
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, list(run.conditions.values()), frameon=False, loc="lower left", bbox_to_anchor=(0, 1.0),
              ncol=2, borderaxespad=0.2)
    pf.stat_title(ax, f"{shown}, unit {run.labels[unit]}")
    if exp is not None:
        ax = exp.plot_tails(gene, unit=unit, ax=axd["b"])
        ax.set_box_aspect(1)
        handles, _ = ax.get_legend_handles_labels()
        ax.legend(handles, list(run.conditions.values()), frameon=False)
        pf.stat_title(ax, f"{shown}; ticks: sample medians")
    else:
        axd["b"].set_axis_off()
    pf.panel_labels([axd["a"], axd["b"]] if exp is not None else [axd["a"]])
    save(fig, "example_gene")


def fig_reads_vs_replicates(run: Run) -> None:
    """The genes of the unit with the smallest overall shift, tested with replicates and with reads as units."""
    within = run.res.get("tail_genes", level="within")
    unit = within.groupby("unit").effect.median().abs().idxmin() if len(within) else run.res.units[0]
    fig, axs = pf.subplots(1, 2, width="double", aspect=square_aspect(1, 2))
    for ax, analysis, title in ((axs[0], "tail_genes", "Replicates as units"),
                                (axs[1], "tail_genes_reads", "Reads as units")):
        t = volcano_panel(run, ax, analysis, "Change in poly(A) tail (nt)", n_label=0, level="within", unit=unit)
        ax.set_title(title, loc="left")
        if len(t):
            pf.stat_title(ax, f"q < 0.05: {(t.q < 0.05).mean():.0%} of {len(t):,} genes")
    pf.panel_labels(list(axs))
    save(fig, "example_reads_vs_replicates")


def simulated_run(folder) -> tuple[Path, Path]:
    """(output directory, config) of a simulated experiment of 10 units with changes in both directions; made and
    analysed when missing."""
    config, out = Path(folder) / "config.yaml", Path(folder) / "isoends_out"
    if not (out / "tables" / "tss_switch_across_units.parquet").exists():
        config = make_experiment(folder, n_units=10, n_background=1200, n_apa_background=1500, seed=3,
                                 effects="mixed")
        ie.Experiment.from_config(config).run(report=False)
    return out, config


FIGURES = {"tails": lambda run, exp: fig_tails(run),
           "apa": lambda run, exp: fig_volcano_panels(run, APA_PANELS, "example_apa"),
           "tss": lambda run, exp: fig_volcano_panels(run, TSS_PANELS, "example_tss"),
           "gene": fig_gene,
           "reads": lambda run, exp: fig_reads_vs_replicates(run)}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("outdir", nargs="?", help="an IsoEnds output directory")
    p.add_argument("--simulate", metavar="FOLDER", help="simulate an experiment in FOLDER and draw from it")
    p.add_argument("--config", help="the run's config, for the per-gene tail distribution")
    p.add_argument("--anonymise", action="store_true", help="number the units and name no gene")
    p.add_argument("--figures", nargs="+", choices=list(FIGURES), default=list(FIGURES))
    p.add_argument("--journal", default="cell")
    args = p.parse_args()
    if not args.outdir and not args.simulate:
        p.error("give an output directory or --simulate FOLDER")
    pf.apply_style(journal=args.journal)
    outdir, config = simulated_run(args.simulate) if args.simulate else (args.outdir, args.config)
    run = Run(outdir, args.anonymise or bool(args.simulate))
    exp = ie.Experiment.from_config(config) if config else None
    for name in args.figures:
        FIGURES[name](run, exp)


if __name__ == "__main__":
    main()

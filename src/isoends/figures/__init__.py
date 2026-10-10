"""The report figures, drawn from the output tables."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

from ..util import LOG
from . import apa as A
from . import tails as T_
from .style import Figure, setup


def make_figures(cfg, outdir: Path) -> list[Figure]:
    """Draw every figure the tables allow, off screen. A figure that fails is logged and skipped."""
    backend = matplotlib.get_backend()
    plt.switch_backend("Agg")
    try:
        return _make_figures(cfg, outdir)
    finally:
        plt.switch_backend(backend)


def _make_figures(cfg, outdir: Path) -> list[Figure]:
    from ..pipeline import read_table
    outdir = Path(outdir)
    info = json.loads((outdir / "logs" / "run_info.json").read_text())
    setup(cfg.figures.journal)
    fp = dict(formats=tuple(cfg.figures.formats), dpi=cfg.figures.dpi)
    genes = tuple(cfg.figures.genes)
    cache: dict = {}

    def T(name):
        if name not in cache:
            cache[name] = read_table(outdir, name)
        return cache[name]

    jobs = [
        ("tail violins", lambda: T_.fig_tail_violins(T, info, outdir, fp)),
        ("gene volcano", lambda: T_.fig_gene_volcano(T, info, outdir, fp, genes, cfg.figures.n_label)),
        ("classes and categories", lambda: T_.fig_classes_categories(T, info, outdir, fp)),
        ("within-gene isoforms", lambda: T_.fig_within_gene(T, info, outdir, fp)),
        ("thresholds", lambda: T_.fig_thresholds(T, info, outdir, fp)),
        ("volcano per unit", lambda: T_.fig_unit_volcanoes(T, info, outdir, fp, genes)),
        ("APA volcanoes", lambda: A.fig_apa_volcanoes(T, info, outdir, fp, genes)),
        ("PDUI and span", lambda: A.fig_pdui_span(T, info, outdir, fp, genes)),
        ("stop-codon axis", lambda: A.fig_stop_axis(T, info, outdir, fp)),
        ("APA per unit", lambda: A.fig_unit_grids(T, info, outdir, fp)),
        ("APA classes per unit", lambda: A.fig_class_unit_grids(T, info, outdir, fp)),
        ("TSS volcanoes", lambda: A.fig_tss_volcanoes(T, info, outdir, fp, genes)),
        ("5'UTR and non-coding APA", lambda: A.fig_other_apa(T, info, outdir, fp, genes)),
        ("QC", lambda: A.fig_qc(T, info, outdir, fp)),
    ]
    out: list[Figure] = []
    for label, job in jobs:
        try:
            res = job()
        except Exception as e:
            LOG.warning("figure %s failed: %s: %s", label, type(e).__name__, e)
            continue
        if res is None:
            continue
        out += res if isinstance(res, list) else [res]
    if cfg.figures.isotracks and genes:
        from .isotracks import draw_genes
        try:
            for f in draw_genes(cfg, list(genes), cfg.figures.isotracks_unit, outdir=outdir, info=info, tables=T):
                out.append(f)
        except Exception as e:
            LOG.warning("IsoTracks views failed: %s: %s", type(e).__name__, e)
    return out

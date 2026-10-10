"""The Python API: results, analyses, plots and per-gene tails."""
import numpy as np
import pandas as pd
import pytest

import isoends as ie


@pytest.fixture(scope="module")
def results(synthetic_run):
    return ie.load(synthetic_run["cfg"].outdir)


def test_every_analysis_is_reachable(results):
    a = results.analyses().set_index("analysis")
    assert set(a.index) == set(ie.ANALYSES)
    for key in ("tail_genes", "tandem", "composite", "skipped", "internal", "five_utr", "intronic",
                "noncoding_tandem", "tss_switch", "tss_upstream", "tss_first_exon"):
        assert a.loc[key, "tested_across_units"] > 0, key
    t = results.get("tandem")
    assert {"effect", "q", "gene_name"} <= set(t.columns) and t.attrs["level"] == "across"
    w = results.get("tandem", level="within", unit="LINE2")
    assert set(w.unit) == {"LINE2"} and w.is_tandem.all()


def test_plots_return_matplotlib_objects(results):
    import matplotlib
    ax = results.volcano("tandem", label=["TANDEM1"])
    assert isinstance(ax, matplotlib.axes.Axes)
    assert any(t.get_text() == "TANDEM1" for t in ax.texts)
    fig = results.volcano_grid("tail_genes")
    assert len([a for a in fig.axes if a.get_visible() and a.axison]) == 6
    ax = results.plot_sites("TANDEM1", unit="LINE1")
    assert len(ax.patches) == 4
    ax = results.plot_sites("TSS1", unit="LINE1", kind="tss")
    assert len(ax.patches) == 4
    matplotlib.pyplot.close("all")


def test_gene_view_collects_every_table(results):
    g = results.gene("TANDEM1")
    assert {"apa_tandem_within_unit", "apa_tandem_across_units", "sites", "tails_gene_across_units"} <= set(g)


def test_tails_of_a_gene_match_the_tables(synthetic_run):
    exp = ie.Experiment(synthetic_run["cfg"])
    exp._ctx = synthetic_run["ctx"]
    mine = exp.tails_of("RPL5", unit="LINE1").set_index("dataset")
    table = ie.load(synthetic_run["cfg"].outdir).table("tails_gene_sample")
    ref = table[(table.gene_name == "RPL5") & (table.unit == "LINE1")].set_index("dataset")
    assert np.allclose(mine.loc[ref.index, "median_tail_nt"], ref.median_tail_nt)
    assert (mine.loc[ref.index, "reads"] == ref.reads).all()
    counts = exp.tail_counts("RPL5", unit="LINE1")
    assert counts.groupby("dataset").reads.sum().loc[ref.index].tolist() == ref.reads.tolist()


def test_experiment_without_yaml(synthetic_run, tmp_path):
    cfg = synthetic_run["cfg"]
    root = cfg.source.parent
    samples = pd.read_csv(root / "samples.tsv", sep="\t")
    exp = ie.Experiment.from_isoforge(
        runs={r: str(root / r) for r in cfg.runs}, samples=samples, control=cfg.control, treatment=cfg.treatment,
        reference_gtf=root / "reference.gtf", outdir=tmp_path / "out", params={"stats": {"limma_backend": "python"}})
    v = exp.validate()
    assert (v.status != "error").all() and len(exp.cfg.runs) == 6


def test_report_follows_the_configured_figures(synthetic_run, monkeypatch):
    from isoends import pipeline
    cfg = synthetic_run["cfg"]
    seen = []
    monkeypatch.setattr(pipeline, "run_report", lambda c, hash_inputs=True: seen.append(c.figures.enabled))
    enabled = cfg.figures.enabled
    cfg.figures.enabled = False
    exp = ie.Experiment(cfg)
    exp.report()
    exp.report(figures=True)
    assert seen == [False, True] and cfg.figures.enabled is False
    cfg.figures.enabled = enabled


def test_plots_keep_the_matplotlib_backend(results):
    import matplotlib
    before = matplotlib.get_backend()
    results.volcano("tandem")
    assert matplotlib.get_backend() == before
    matplotlib.pyplot.close("all")


def test_read_level_is_descriptive_and_within_units_only(results):
    a = results.analyses().set_index("analysis")
    assert a.loc["tail_genes_reads", "tested_across_units"] == 0
    r = results.get("tail_genes_reads")
    g = results.get("tail_genes", level="within")
    assert r.attrs["level"] == "within" and len(r) == len(g)
    assert (r.q < 0.05).sum() >= (g.q < 0.05).sum()
    assert r.n_pairs_reads_consistent.between(0, r.n_pairs).all()

import os
import shutil

os.environ.setdefault("MPLBACKEND", "Agg")

import pytest  # noqa: E402

from isoends.config import load_config  # noqa: E402
from isoends.synthetic import make_experiment  # noqa: E402


def run_stages(cfg):
    from isoends import pipeline
    ctx = pipeline.build_context(cfg, jobs=1, stage="ingest")
    return dict(cfg=cfg, ctx=ctx, tails=pipeline.run_tails(cfg, ctx=ctx), apa=pipeline.run_apa(cfg, ctx=ctx),
                tss=pipeline.run_tss(cfg, ctx=ctx))


@pytest.fixture(scope="session")
def synthetic_run(tmp_path_factory):
    """A paired synthetic experiment (6 units x 3 replicate pairs) run through tails, apa and tss."""
    root = tmp_path_factory.mktemp("synthetic")
    return run_stages(load_config(make_experiment(root, n_units=6, n_replicates=3, seed=11)))


@pytest.fixture(scope="session")
def unpaired_run(tmp_path_factory):
    """The same experiment with a sample sheet without pairs."""
    root = tmp_path_factory.mktemp("unpaired")
    return run_stages(load_config(make_experiment(root, n_units=6, n_replicates=3, seed=11, paired=False)))


STRICT = {"apa": {"tandem_sites": "terminal", "same_exon_check": True, "across_units_same_pair": True,
                  "novel_gene_models": True, "intronic_models": "gene"},
          "tss": {"across_units_same_pair": True, "novel_gene_models": True}}


@pytest.fixture(scope="session")
def strict_run(synthetic_run):
    """The paired experiment again with the stricter site options: tandem pairs from the canonical last exon, the
    same-exon check, the same pair across units, novel genes on their best model, a gene's own models only."""
    from isoends import pipeline
    base = synthetic_run["cfg"]
    cfg = load_config(base.source, overrides={"outdir": str(base.outdir.parent / "strict"),
                                              "cache_dir": str(base.cache_dir), "params": STRICT})
    ctx = pipeline.build_context(cfg, jobs=1, stage="ingest")
    return dict(cfg=cfg, ctx=ctx, apa=pipeline.run_apa(cfg, ctx=ctx), tss=pipeline.run_tss(cfg, ctx=ctx))


def rscript_with_limma():
    from isoends.stats.limma import r_limma_available
    return r_limma_available()


needs_r = pytest.mark.skipif(not (shutil.which("Rscript") and rscript_with_limma()), reason="Rscript with limma")

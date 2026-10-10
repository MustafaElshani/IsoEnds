"""isoends: poly(A) tails, alternative polyadenylation and transcription start sites from IsoForge runs.

  isoends init      [--out isoends.yaml]      write a configuration template
  isoends validate  --config config.yaml      check the inputs and the sample sheet
  isoends all       --config config.yaml      tails, apa, tss and the report
  isoends tails     --config config.yaml      poly(A) tail analyses
  isoends apa       --config config.yaml      alternative polyadenylation
  isoends tss       --config config.yaml      alternative transcription start sites
  isoends report    --config config.yaml      figures, report.md and provenance.json from the tables
  isoends isotracks --config config.yaml --gene NAME [--unit UNIT]     read-level view of a gene
  isoends demo      [--out DIR]               make a synthetic experiment and analyse it
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .util import LOG, setup_logging

STAGES = ("tails", "apa", "tss", "all")


def _config(args):
    from .config import load_config
    overrides = {}
    if args.outdir:
        overrides["outdir"] = str(Path(args.outdir).resolve())
    for key in ("control", "treatment"):
        if getattr(args, key, None):
            overrides[key] = getattr(args, key)
    if getattr(args, "limma", None):
        overrides["params"] = {"stats": {"limma_backend": args.limma}}
    return load_config(args.config, overrides=overrides)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(prog="isoends", description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--version", action="version", version=f"isoends {__version__}")
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("validate", *STAGES, "report", "isotracks"):
        s = sub.add_parser(name)
        s.add_argument("--config", "-c", required=True)
        s.add_argument("--outdir", help="output directory, instead of the config's")
        s.add_argument("--control", help="control condition, instead of the config's")
        s.add_argument("--treatment", help="treatment condition, instead of the config's")
        s.add_argument("--verbose", "-v", action="store_true")
        if name in STAGES:
            s.add_argument("--jobs", "-j", type=int, default=1, help="runs summarised in parallel")
            s.add_argument("--limma", choices=["python", "r", "auto"], help="backend of the moderated t-test")
        if name in ("report", "all"):
            s.add_argument("--no-hash", action="store_true", help="record input sizes and dates without SHA-256")
            s.add_argument("--no-figures", action="store_true")
        if name == "isotracks":
            s.add_argument("--gene", required=True, action="append")
            s.add_argument("--unit")
    s = sub.add_parser("init")
    s.add_argument("--out", default="isoends.yaml")
    s = sub.add_parser("demo")
    s.add_argument("--out", default="isoends_demo")
    s.add_argument("--units", type=int, default=6)
    s.add_argument("--unpaired", action="store_true", help="replicates without pairs")
    s.add_argument("--no-figures", action="store_true")
    args = p.parse_args(argv)
    setup_logging(getattr(args, "verbose", False))
    import matplotlib
    matplotlib.use("Agg")

    if args.cmd == "init":
        from .templates import CONFIG_TEMPLATE
        out = Path(args.out)
        if out.exists():
            LOG.error("%s exists; not overwritten", out)
            return 1
        out.write_text(CONFIG_TEMPLATE)
        print(f"wrote {out}")
        return 0
    from . import pipeline
    if args.cmd == "demo":
        from .config import load_config
        from .synthetic import make_experiment
        cfg = load_config(make_experiment(args.out, n_units=args.units, n_apa_background=120,
                                          paired=not args.unpaired))
        cfg.figures.enabled = not args.no_figures
        print(f"demo data in {args.out}; report: {pipeline.run_all(cfg)}")
        return 0

    cfg = _config(args)
    if getattr(args, "no_figures", False):
        cfg.figures.enabled = False
    if args.cmd == "validate":
        sheet, rows = pipeline.validate_inputs(cfg)
        for r in rows:
            print(f"{r['status']:8s} {r['scope']}: {r['check']} {r['message']}")
        print(f"{len(sheet.table)} samples, {len(sheet.units)} units, runs: {', '.join(cfg.runs)}")
    elif args.cmd == "tails":
        pipeline.run_tails(cfg, args.jobs)
    elif args.cmd == "apa":
        pipeline.run_apa(cfg, args.jobs)
    elif args.cmd == "tss":
        pipeline.run_tss(cfg, args.jobs)
    elif args.cmd == "report":
        pipeline.run_report(cfg, hash_inputs=not args.no_hash)
    elif args.cmd == "all":
        pipeline.run_all(cfg, args.jobs, hash_inputs=not args.no_hash)
    elif args.cmd == "isotracks":
        from .figures.isotracks import draw_genes
        for figure in draw_genes(cfg, args.gene, args.unit):
            print(figure.paths[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())

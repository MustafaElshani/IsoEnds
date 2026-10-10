"""Read-level views of chosen genes, drawn with IsoTracks.

For a tandem 3'UTR gene the best-supported models ending at its proximal and distal sites are drawn with their reads
and tails, the two sites are marked, and the isoform that gains with the treatment is orange. For other genes
IsoTracks chooses the isoforms.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pyarrow.dataset as ds
import pyarrow.parquet as pq

from ..util import LOG
from .style import ORANGE, SLATE, Figure

GAIN, OTHER = ORANGE, SLATE


def _filtered_read_annot(read_annot: Path, transcripts, out: Path) -> Path:
    """A copy of the read annotation holding only the reads of these models."""
    if not out.exists():
        out.parent.mkdir(parents=True, exist_ok=True)
        t = ds.dataset(read_annot).to_table(filter=ds.field("annot_transcript_id").isin(list(transcripts)))
        pq.write_table(t, out)
    return out


def draw_genes(cfg, genes, unit=None, outdir=None, info=None, tables=None) -> list:
    """Draw each gene in `unit` (default: the first). Needs IsoTracks and `bams` in the run's configuration."""
    try:
        from isotracks import IsoTracks
        from isotracks.highlight import Highlight
    except ImportError:
        LOG.warning("IsoTracks is not installed: read-level views skipped")
        return []
    from ..pipeline import read_table
    outdir = Path(outdir or cfg.outdir)
    info = info or json.loads((outdir / "logs" / "run_info.json").read_text())
    T = tables or (lambda n: read_table(outdir, n))
    sheet = pd.DataFrame(info["samples"])
    unit = unit or sheet.unit.iat[0]
    ut = sheet[sheet.unit == unit]
    run = cfg.runs[ut.run.iat[0]]
    if not run.bam_pattern:
        LOG.warning("run %s has no `bams`: IsoTracks views skipped", run.name)
        return []
    groups = {"control": [str(run.bam(d)) for d in ut[ut.condition == cfg.control].dataset],
              "treated": [str(run.bam(d)) for d in ut[ut.condition == cfg.treatment].dataset]}
    missing = [b for v in groups.values() for b in v if not Path(b).exists()]
    if missing:
        LOG.warning("BAMs not found (%s ...): IsoTracks views skipped", missing[0])
        return []
    tandem = T("apa_tandem_within_unit")
    out = []
    for gene in genes:
        row = tandem[(tandem.unit == unit) & (tandem.gene_name == gene)] if len(tandem) else pd.DataFrame()
        transcripts, highlight, colors, note = None, None, None, ""
        if len(row) and pd.notna(row.proximal_model.iat[0]) and pd.notna(row.distal_model.iat[0]):
            r = row.iloc[0]
            transcripts = [r.proximal_model, r.distal_model]
            gains = r.distal_model if r.delta_pdui > 0 else r.proximal_model
            colors = {t: (GAIN if t == gains else OTHER) for t in transcripts}
            highlight = [Highlight(spans=[(int(r.proximal_site) - 15, int(r.proximal_site) + 15)], label="proximal",
                                   label_color=colors[r.proximal_model]),
                         Highlight(spans=[(int(r.distal_site) - 15, int(r.distal_site) + 15)], label="distal",
                                   label_color=colors[r.distal_model])]
            note = (f" Models: proximal-site {r.proximal_model} ({int(r.proximal_model_reads_control):,} → "
                    f"{int(r.proximal_model_reads_treated):,} reads), distal-site {r.distal_model} "
                    f"({int(r.distal_model_reads_control):,} → {int(r.distal_model_reads_treated):,} reads); sites "
                    f"{abs(r.distal_site - r.proximal_site):,.0f} nt apart; ΔPDUI {r.delta_pdui:+.3f} "
                    f"(CMH q {r.q:.2g}). Orange: the isoform that gains.")
        read_annot = run.read_annot
        if transcripts:
            key = "_".join(sorted(transcripts))[:120]
            read_annot = _filtered_read_annot(run.read_annot, transcripts,
                                              Path(cfg.cache_dir) / "isotracks" / f"{unit}_{gene}_{key}.parquet")
        stem = outdir / "figures" / f"fig11_isotracks_{unit}_{gene}"
        stem.parent.mkdir(parents=True, exist_ok=True)
        in_sync = bool(info.get("sync", {}).get(run.name, {}).get("in_sync"))
        view = IsoTracks(gene_name=gene, bam_files=groups, reference_gtf=str(cfg.reference_gtf),
                         annotation_gtf=str(run.characterised_gtf if in_sync and run.characterised_gtf else run.gtf),
                         read_annot=str(read_annot), bed_file=str(run.bed) if run.bed and in_sync else None,
                         polya_clusters=str(run.tes_clusters) if run.tes_clusters else None,
                         tss_clusters=str(run.tss_clusters) if run.tss_clusters else None,
                         min_reads_per_condition=5, max_isoforms=len(transcripts) if transcripts else 6,
                         condition_labels={"control": cfg.control, "treated": cfg.treatment}, verbose=False)
        if colors:
            drawn = view.transcripts()
            view.theme = view.theme.derive(isoform_colors=[colors.get(t, OTHER) for t in drawn])
            view.transcripts(refresh=True)
        kw = {"highlight": highlight} if highlight else {}
        if transcripts:
            lo_, hi_ = sorted((float(r.proximal_site), float(r.distal_site)))
            pad = max(150, int(0.25 * (hi_ - lo_)))
            kw.update(detail=(int(lo_) - pad, int(hi_) + pad), detail_share=0.4)
        view.plot(output=str(stem), formats=tuple(f for f in cfg.figures.formats if f in ("pdf", "svg", "png")),
                  width="double", dpi=cfg.figures.dpi, **kw)
        cap = (f"Read-level view of {gene} in {unit} (IsoTracks): the models drawn, every read assigned to them split by "
               f"condition ({cfg.control}, {cfg.treatment}; replicates pooled) with its poly(A) tail, poly(A) and TSS "
               f"clusters, and coverage." + note)
        (stem.parent / f"{stem.name}.caption.md").write_text(cap + "\n")
        out.append(Figure(name=stem.name, title=f"{gene} in {unit}", caption=cap,
                          paths=[str(stem.parent / f"{stem.name}.{f}") for f in cfg.figures.formats]))
    return out

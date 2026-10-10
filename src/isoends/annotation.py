"""Per-run transcript annotation: structure keys, and categories from the reference and the characterised GTF.

Categories: protein coding, retained intron, lncRNA, NMD, other, unclassified.
  Annotated models (ENST) take their reference transcript type.
  Novel models take their SQANTI3 class from the characterised GTF of isoforge-characterize, in this order:
  intron retention -> retained intron; predicted NMD -> NMD; coding -> protein coding; associated with a lncRNA ->
  lncRNA; else other. Without a characterised GTF that describes this run they stay unclassified.

The characterised GTF is used only when it describes this run: at least `min_present` of the reads lie on models
present in it and at least `min_chain_agreement` of the shared models have identical exon chains.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import gtf as G

CATEGORIES = ("protein coding", "retained intron", "lncRNA", "NMD")
_REFERENCE_TYPES = {"protein_coding": "protein coding", "retained_intron": "retained intron", "lncRNA": "lncRNA",
                    "nonsense_mediated_decay": "NMD"}
ISOFORGE_ATTRS = ["transcript_id", "gene_id", "gene_name", "transcript_type"]
SQANTI_ATTRS = ["structural_category", "subcategory", "coding", "predicted_NMD", "is_NMD", "is_IR", "is_lncRNA",
                "ref_transcript_type", "associated_gene", "associated_transcript", "ORF_length", "CDS_length",
                "CDS_start", "CDS_end"]


def _true(s: pd.Series) -> np.ndarray:
    return s.astype("string").str.upper().isin(["TRUE", "T", "1", "YES"]).fillna(False).to_numpy(bool)


def _is(s: pd.Series, value: str) -> np.ndarray:
    return s.astype("string").eq(value).fillna(False).to_numpy(bool)


def run_transcripts(isoforge_gtf, characterised_gtf=None, model_read_counts: pd.Series | None = None,
                    min_present: float = 0.5, min_chain_agreement: float = 0.95):
    """(transcripts, check, exons). transcripts: one row per IsoForge model with gene_id, gene_name, n_exons,
    structure_key, category, category_source and the SQANTI3 fields. check: how well the characterised GTF describes
    the run. exons: the models' exons. model_read_counts: reads per transcript_id, weighting the check."""
    exons = G.model_exons(isoforge_gtf)
    tx = G.model_attributes(isoforge_gtf, ISOFORGE_ATTRS).merge(G.structure_table(exons), on="transcript_id",
                                                                 how="left")
    tx["gene_id"] = G.strip_version(tx.gene_id)
    tx["annotated"] = tx.transcript_id.astype(str).str.startswith("ENST").to_numpy()
    check: dict = dict(characterised_gtf=str(characterised_gtf) if characterised_gtf else None)
    char = None
    if characterised_gtf is not None:
        char = G.model_attributes(characterised_gtf, ["transcript_id"] + SQANTI_ATTRS)
        chains = G.structure_table(G.model_exons(characterised_gtf))[["transcript_id", "exon_chain"]]
        both = tx[["transcript_id", "exon_chain"]].merge(chains.rename(columns={"exon_chain": "char_chain"}),
                                                         on="transcript_id")
        weights = (model_read_counts if model_read_counts is not None
                   else pd.Series(1.0, index=tx.transcript_id)).astype(float)
        with_reads = weights[(weights > 0) & (weights.index != "")]
        total = with_reads.sum()

        def share(ids) -> float:
            return float(with_reads[with_reads.index.isin(set(ids))].sum() / total) if len(with_reads) else np.nan
        agree = both.exon_chain == both.char_chain
        characterised = char.structural_category.notna()
        check.update(
            share_reads_on_isoforge_gtf_models=share(tx.transcript_id),
            n_models_with_reads=int(len(with_reads)),
            share_models_present=float(with_reads.index.isin(set(char.transcript_id)).mean())
            if len(with_reads) else np.nan,
            share_reads_present=share(char.transcript_id),
            n_models_compared=int(len(both)),
            share_chain_agreement=float(agree.mean()) if len(both) else np.nan,
            n_models_characterised=int(characterised.sum()),
            share_reads_characterised=share(char.transcript_id[characterised]))
        check["in_sync"] = bool(check["share_chain_agreement"] >= min_chain_agreement
                                and check["share_reads_present"] >= min_present)
        check["rule"] = (f"used when >= {min_present:.0%} of reads are on models present in it and >= "
                         f"{min_chain_agreement:.0%} of the shared models have identical exon chains")
        char = char[~char.transcript_id.isin(set(both.transcript_id[~agree]))] if check["in_sync"] else None
    else:
        check.update(in_sync=False, rule="no characterised GTF: novel models unclassified")
    if char is not None:
        tx = tx.merge(char, on="transcript_id", how="left")
    else:
        for c in SQANTI_ATTRS:
            tx[c] = pd.NA
    reference_type = tx.ref_transcript_type.fillna(tx.transcript_type)
    tx["reference_type"] = np.where(tx.annotated, reference_type.astype("string").fillna(""), "")
    characterised = ~tx.annotated & tx.structural_category.notna()
    ir = _true(tx.is_IR) | _is(tx.subcategory, "intron_retention")
    nmd = _true(tx.predicted_NMD) | _true(tx.is_NMD)
    coding = _is(tx.coding, "coding")
    lnc = _is(tx.transcript_type, "lncRNA") | _is(tx.ref_transcript_type, "lncRNA") | _true(tx.is_lncRNA)
    novel = np.select([ir, nmd, coding, lnc], ["retained intron", "NMD", "protein coding", "lncRNA"], default="other")
    known = pd.Series(tx.reference_type).map(_REFERENCE_TYPES).fillna("other").to_numpy()
    tx["category"] = np.where(tx.annotated, known, np.where(characterised, novel, "unclassified"))
    tx["category_source"] = np.where(tx.annotated, "reference", np.where(characterised, "SQANTI3", "none"))
    return tx, check, exons

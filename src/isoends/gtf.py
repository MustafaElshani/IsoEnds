"""GTF readers: the reference (Ensembl-canonical transcripts, exons, start and stop codons), IsoForge's models and
the characterised GTF of isoforge-characterize.

Genes are keyed on the unversioned Ensembl ID. Positions are 1-based and inclusive, as in the GTF.
"""
from __future__ import annotations

import gzip
import io
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

from .util import atomic_write_parquet, file_fingerprint

GTF_COLUMNS = ["chrom", "source", "feature", "start", "end", "score", "strand", "frame", "attributes"]


def strip_version(ids):
    """ENSG00000123456.7 -> ENSG00000123456. IDs without a version are unchanged."""
    s = pd.Series(ids, dtype="string") if not isinstance(ids, pd.Series) else ids.astype("string")
    return s.str.replace(r"\.\d+$", "", regex=True)


def _grep(path: Path, patterns: list[str]) -> str:
    """Lines of a GTF containing any of the fixed strings, with grep -F when available."""
    path = Path(path)
    if path.suffix != ".gz" and shutil.which("grep"):
        args = ["grep", "-F"] + [a for p in patterns for a in ("-e", p)] + [str(path)]
        run = subprocess.run(args, capture_output=True, text=True)
        if run.returncode not in (0, 1):
            raise RuntimeError(f"grep failed on {path}: {run.stderr}")
        return run.stdout
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as fh:
        return "".join(line for line in fh if any(p in line for p in patterns))


def _frame(text: str) -> pd.DataFrame:
    if not text:
        return pd.DataFrame(columns=GTF_COLUMNS)
    d = pd.read_csv(io.StringIO(text), sep="\t", header=None, names=GTF_COLUMNS, comment="#",
                    dtype={"chrom": str, "feature": str, "strand": str, "attributes": str}, quoting=3)
    return d


def read_gtf(path: Path, features=None, patterns=None) -> pd.DataFrame:
    """A GTF as a table, attributes left as text (see `attr`). `patterns` pre-filters lines by fixed strings."""
    path = Path(path)
    if patterns:
        d = _frame(_grep(path, list(patterns)))
    else:
        opener = gzip.open if path.suffix == ".gz" else open
        with opener(path, "rt") as fh:
            d = pd.read_csv(fh, sep="\t", header=None, names=GTF_COLUMNS, comment="#", quoting=3,
                            dtype={"chrom": str, "feature": str, "strand": str, "attributes": str})
    if features is not None:
        d = d[d.feature.isin(list(features))]
    return d.reset_index(drop=True)


def attr(d: pd.DataFrame, name: str) -> pd.Series:
    """One attribute of every row (missing -> <NA>)."""
    return d.attributes.str.extract(rf'(?:^|;\s*){name} "([^"]*)"', expand=False).astype("string")


def reference_annotation(gtf: Path, cache_dir: Path) -> dict[str, pd.DataFrame]:
    """The parts of the reference GTF IsoEnds uses, cached as Parquet:

    canonical: exons of each gene's Ensembl-canonical transcript, the first when a gene has several (gene_id,
               transcript_id, chrom, strand, start, end, rank from the 5' end, n_exons)
    stop:      the stop codon of each canonical transcript (gene_id, chrom, strand, start, end)
    start:     its start codon
    exons:     every exon of every transcript, with is_last for a transcript's 3'-most exon
    genes:     gene_id, gene_name, gene_type, chrom, strand
    Genes of the Y pseudoautosomal regions (_PAR_Y) are left out.
    """
    gtf = Path(gtf)
    key = file_fingerprint(gtf, extra="ref-v2")
    folder = Path(cache_dir) / "reference" / key
    names = ("canonical", "stop", "start", "exons", "genes")
    if all((folder / f"{n}.parquet").exists() for n in names):
        return {n: pd.read_parquet(folder / f"{n}.parquet") for n in names}
    folder.mkdir(parents=True, exist_ok=True)

    exons = read_gtf(gtf, features=["exon"], patterns=["\texon\t"])
    exons = pd.DataFrame({"gene_id": strip_version(attr(exons, "gene_id")),
                          "transcript_id": attr(exons, "transcript_id"),
                          "chrom": exons.chrom, "strand": exons.strand,
                          "start": exons.start.astype("int64"), "end": exons.end.astype("int64"),
                          "canonical": exons.attributes.str.contains('tag "Ensembl_canonical"', regex=False)})
    raw_gene = exons.gene_id.astype(str)
    exons = exons[~raw_gene.str.contains("_PAR_Y", regex=False)]
    key3 = np.where(exons.strand == "+", exons.end, -exons.start)
    exons = exons.assign(_k=key3)
    last = exons.groupby("transcript_id")._k.transform("max") == exons._k
    exons = exons.assign(is_last=last.to_numpy()).drop(columns="_k")

    can = exons[exons.canonical].drop(columns=["canonical", "is_last"]).copy()
    can["_k"] = np.where(can.strand == "+", can.start, -can.end)
    can = can.sort_values(["gene_id", "_k"]).drop(columns="_k")
    first_tx = can.groupby("gene_id").transcript_id.transform("first")
    can = can[can.transcript_id == first_tx]
    can["rank"] = can.groupby("gene_id").cumcount() + 1
    can["n_exons"] = can.groupby("gene_id").gene_id.transform("size")
    can = can.reset_index(drop=True)

    codons = read_gtf(gtf, features=["stop_codon", "start_codon"], patterns=['tag "Ensembl_canonical"'])
    codons = pd.DataFrame({"gene_id": strip_version(attr(codons, "gene_id")),
                           "transcript_id": attr(codons, "transcript_id"), "feature": codons.feature,
                           "chrom": codons.chrom, "strand": codons.strand, "start": codons.start.astype("int64"),
                           "end": codons.end.astype("int64")})
    codons = codons[codons.transcript_id.isin(set(can.transcript_id))]

    def codon(kind):
        return (codons[codons.feature == kind].groupby(["gene_id", "transcript_id", "chrom", "strand"], as_index=False)
                .agg(start=("start", "min"), end=("end", "max")))
    stop, start = codon("stop_codon"), codon("start_codon")

    genes = read_gtf(gtf, features=["gene"], patterns=["\tgene\t"])
    genes = pd.DataFrame({"gene_id": strip_version(attr(genes, "gene_id")), "gene_name": attr(genes, "gene_name"),
                          "gene_type": attr(genes, "gene_type"), "chrom": genes.chrom, "strand": genes.strand})
    genes = genes.drop_duplicates("gene_id")
    out = {"canonical": can, "stop": stop, "start": start,
           "exons": exons.drop(columns="canonical").reset_index(drop=True), "genes": genes.reset_index(drop=True)}
    for n, d in out.items():
        atomic_write_parquet(d, folder / f"{n}.parquet")
    return out


def model_exons(gtf: Path) -> pd.DataFrame:
    """Exons of every model in an IsoForge or characterised GTF: transcript_id, gene_id, chrom, strand, start,
    end."""
    d = read_gtf(gtf, features=["exon"], patterns=["\texon\t"])
    return pd.DataFrame({"transcript_id": attr(d, "transcript_id"), "gene_id": strip_version(attr(d, "gene_id")),
                         "chrom": d.chrom, "strand": d.strand, "start": d.start.astype("int64"),
                         "end": d.end.astype("int64")}).reset_index(drop=True)


def model_attributes(gtf: Path, names) -> pd.DataFrame:
    """The transcript lines of a GTF with the requested attributes, plus chrom, start, end, strand."""
    d = read_gtf(gtf, features=["transcript"], patterns=["\ttranscript\t"])
    out = pd.DataFrame({n: attr(d, n) for n in names})
    out["chrom"], out["start"], out["end"], out["strand"] = d.chrom, d.start, d.end, d.strand
    return out.drop_duplicates("transcript_id").reset_index(drop=True)


def reported_transcripts(gtf: Path) -> set:
    """The IDs of the transcripts an IsoForge GTF reports."""
    d = read_gtf(gtf, features=["transcript"], patterns=["\ttranscript\t"])
    return set(attr(d, "transcript_id").dropna().astype(str))


def terminal_exons(exons: pd.DataFrame) -> pd.DataFrame:
    """Each model's 3'-most exon: transcript_id, gene_id, chrom, strand, exon5 (the exon's 5' boundary) and site
    (the model's 3' end)."""
    plus = exons.strand == "+"
    last = pd.concat([exons[plus].sort_values("end").groupby("transcript_id").tail(1),
                      exons[~plus].sort_values("start").groupby("transcript_id").head(1)])
    last = last.assign(exon5=np.where(last.strand == "+", last.start, last.end),
                       site=np.where(last.strand == "+", last.end, last.start))
    return last[["transcript_id", "gene_id", "chrom", "strand", "exon5", "site"]].reset_index(drop=True)


def first_exons(exons: pd.DataFrame) -> pd.DataFrame:
    """Each model's 5'-most exon: transcript_id, gene_id, chrom, strand, site (the model's 5' end) and exon3 (the
    exon's 3' boundary)."""
    plus = exons.strand == "+"
    first = pd.concat([exons[plus].sort_values("start").groupby("transcript_id").head(1),
                       exons[~plus].sort_values("end").groupby("transcript_id").tail(1)])
    first = first.assign(site=np.where(first.strand == "+", first.start, first.end),
                         exon3=np.where(first.strand == "+", first.end, first.start))
    return first[["transcript_id", "gene_id", "chrom", "strand", "site", "exon3"]].reset_index(drop=True)


def structure_table(exons: pd.DataFrame) -> pd.DataFrame:
    """Per model: n_exons, exon_chain and structure_key. The key of a spliced model is its intron chain
    ("chr:strand:end1-start2,..."), the same in every run. The key of a mono-exonic model is its unversioned ID when
    annotated (ENST), else "mono:<ID>"."""
    e = exons.sort_values(["transcript_id", "start"])
    g = e.groupby("transcript_id", sort=False)
    starts = g.start.apply(lambda s: s.to_numpy())
    ends = g.end.apply(lambda s: s.to_numpy())
    first = g.first()
    rows = []
    for tid in starts.index:
        s, t = starts[tid], ends[tid]
        chain = ",".join(f"{a}-{b}" for a, b in zip(s, t))
        chrom, strand = first.at[tid, "chrom"], first.at[tid, "strand"]
        if len(s) > 1:
            key = f"{chrom}:{strand}:" + ",".join(f"{b}-{a}" for b, a in zip(t[:-1], s[1:]))
        elif str(tid).startswith("ENST"):
            key = str(tid).split(".")[0]
        else:
            key = f"mono:{tid}"
        rows.append((tid, len(s), chain, key))
    return pd.DataFrame(rows, columns=["transcript_id", "n_exons", "exon_chain", "structure_key"])

"""A synthetic IsoForge experiment with known effects, for the tests and `isoends demo`.

Every unit is one IsoForge run (read annotation with `polya_length`, IsoForge GTF, characterised GTF) with
`n_replicates` control and treated samples; the units share one reference GTF. Effects of the treatment:

  background genes     tails 15 nt shorter
  RPL5, RPS3, RPL11    tails 20 nt shorter, from 60 nt
  MT-CO1               tails 10 nt longer
  NMDG1                NMD isoform 20 nt longer than the coding isoform, and shortened more (25 against 15 nt)
  TANDEM1 (+)          distal 3'UTR site usage 0.3 -> 0.7
  TANDEM2 (-)          distal usage 0.7 -> 0.3
  CE1                  composite-exon site share 0.10 -> 0.40
  SE1                  skipped-exon site share 0.40 -> 0.10
  IE1                  internal exonic site share 0.05 -> 0.30
  NM1                  intronic site without a model, 0.10 -> 0.30
  SPLIT1               two terminal sites whose distal models splice over the proximal site: not tandem
  UTR5G                cleavage in the first intron, upstream of the start codon, 0.05 -> 0.25
  LNC1                 a lncRNA with two tandem sites, distal usage 0.3 -> 0.65
  NOVEL1               a gene outside the reference with two tandem sites, 0.35 -> 0.7
  TSS1                 an upstream promoter (first exon 5 kb upstream) 0.2 -> 0.6
  TSS2                 an alternative first exon in intron 1, 0.4 -> 0.1
  BG001...             optional genes (n_apa_background) with tandem, intronic and promoter events, some shifted

The sites of unit k are shifted by 3 k nt, so matching them across units needs the tolerance.

With effects="mixed" the experiment looks like a real one instead. The named site genes are left out and the
background genes change in both directions: about a third get shorter tails, a third longer ones (by about 12 nt),
with gamma-distributed tails around a gene-specific mean; in the genes with sites (n_apa_background) about four in
ten shift their tandem, intronic or start-site usage either way, some are lncRNAs and some start inside an internal
exon. Each unit has its own baseline and its own response to the treatment, which each gene follows to its own
degree, and each sample deviates a little per gene.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


@dataclass
class Model:
    tid: str
    exons: list
    ttype: str = "protein_coding"
    novelty: str = "Known"
    sq: dict = field(default_factory=dict)       # SQANTI3 attributes in the characterised GTF


@dataclass
class Component:
    """Reads ending at one site, assigned to one model, with a share of the gene's reads per condition."""
    site: int
    model: str
    share_control: float
    share_treated: float
    feature: str = "three_prime_utr"
    signal: str = "AATAAA"


@dataclass
class Gene:
    gid: str
    name: str
    chrom: str
    strand: str
    canonical: list                       # exons of the canonical transcript
    models: list                          # Model
    components: list                      # Component
    depth: int = 60                       # mean reads per sample
    tail_mean: float = 100.0
    tail_shift: float = -15.0
    tail_sd: float = 25.0
    tail_skewed: bool = False             # gamma-distributed tails instead of normal ones
    unit_response: float = 1.0            # how far the gene follows a unit's shift with the treatment
    unit_sd: float = 0.0                  # sd of the gene's own deviation per unit
    usage_sd: float = 0.0                 # sd per unit, on the log scale, of each site's usage with the treatment
    sample_sd: float = 4.0                # sd of a sample's own deviation in the gene's tail when effects are mixed
    tail_by_model: dict = field(default_factory=dict)   # model -> (mean, shift)
    stop: tuple | None = None             # (start, end) of the canonical stop codon
    start: tuple | None = None            # (start, end) of the start codon; default: 30 nt into exon 1
    gtype: str = "protein_coding"
    in_reference: bool = True
    extra_ref: list = field(default_factory=list)        # further reference transcripts: (tid, exons, type)
    canonical_id: str = ""


MIXED_NAMED = ("NMDG1", "RPL5", "RPS3", "RPL11", "MT-CO1")     # the named genes kept when effects are mixed


def _genes(n_background: int = 60, n_apa_background: int = 0, rng=None, mixed: bool = False) -> list[Gene]:
    g = []

    def enst(k):
        return f"ENST{k:011d}.1"

    g.append(Gene("ENSG00000900001.3", "TANDEM1", "chr1", "+", [(1000, 1300), (2000, 2200), (3000, 4500)],
                  [Model(enst(900001), [(1000, 1300), (2000, 2200), (3000, 4500)]),
                   Model("ISOFORT000000001", [(1000, 1300), (2000, 2200), (3000, 3600)], novelty="ISM",
                         sq={"structural_category": "incomplete-splice_match", "subcategory": "3prime_fragment",
                             "coding": "coding"})],
                  [Component(3600, "ISOFORT000000001", 0.7, 0.3), Component(4500, enst(900001), 0.3, 0.7)],
                  depth=150, stop=(3100, 3102), canonical_id=enst(900001)))
    g.append(Gene("ENSG00000900002.1", "TANDEM2", "chr1", "-", [(10000, 11500), (12000, 12300), (13000, 13300)],
                  [Model(enst(900002), [(10000, 11500), (12000, 12300), (13000, 13300)]),
                   Model("ISOFORT000000002", [(10900, 11500), (12000, 12300), (13000, 13300)], novelty="ISM",
                         sq={"structural_category": "incomplete-splice_match", "coding": "coding"})],
                  [Component(10900, "ISOFORT000000002", 0.3, 0.7), Component(10000, enst(900002), 0.7, 0.3)],
                  depth=150, stop=(11400, 11402), canonical_id=enst(900002)))
    g.append(Gene("ENSG00000900003.2", "CE1", "chr3", "+", [(1000, 1200), (2000, 2200), (4000, 5000)],
                  [Model(enst(900003), [(1000, 1200), (2000, 2200), (4000, 5000)]),
                   Model("ISOFORT000000003", [(1000, 1200), (2000, 2600)], novelty="NNC",
                         sq={"structural_category": "novel_not_in_catalog", "coding": "coding"})],
                  [Component(5000, enst(900003), 0.9, 0.6), Component(2600, "ISOFORT000000003", 0.1, 0.4, "intron")],
                  depth=120, stop=(4100, 4102), canonical_id=enst(900003)))
    g.append(Gene("ENSG00000900004.1", "SE1", "chr3", "+", [(10000, 10200), (13000, 13200), (15000, 16000)],
                  [Model(enst(900004), [(10000, 10200), (13000, 13200), (15000, 16000)]),
                   Model("ISOFORT000000004", [(10000, 10200), (12100, 12300)], novelty="NNC",
                         sq={"structural_category": "novel_not_in_catalog", "coding": "coding"})],
                  [Component(16000, enst(900004), 0.6, 0.9),
                   Component(12300, "ISOFORT000000004", 0.4, 0.1, "intron")],
                  depth=120, stop=(15100, 15102), canonical_id=enst(900004)))
    g.append(Gene("ENSG00000900005.1", "IE1", "chr3", "+", [(20000, 20200), (22000, 22600), (24000, 25000)],
                  [Model(enst(900005), [(20000, 20200), (22000, 22600), (24000, 25000)]),
                   Model("ISOFORT000000005", [(20000, 20200), (22000, 22400)], novelty="NIC",
                         sq={"structural_category": "novel_in_catalog", "coding": "coding"})],
                  [Component(25000, enst(900005), 0.95, 0.7),
                   Component(22400, "ISOFORT000000005", 0.05, 0.3, "exon")],
                  depth=120, stop=(24100, 24102), canonical_id=enst(900005)))
    g.append(Gene("ENSG00000900006.1", "NM1", "chr3", "+", [(31000, 31200), (32000, 32200), (34000, 35000)],
                  [Model(enst(900006), [(31000, 31200), (32000, 32200), (34000, 35000)])],
                  [Component(35000, enst(900006), 0.9, 0.7), Component(31600, enst(900006), 0.1, 0.3, "intron")],
                  depth=120, stop=(34100, 34102), canonical_id=enst(900006)))
    g.append(Gene("ENSG00000900007.1", "SPLIT1", "chr4", "+", [(1000, 1200), (2000, 3000)],
                  [Model(enst(900007), [(1000, 1200), (2000, 3000)]),
                   Model("ISOFORT000000006", [(1000, 1200), (2000, 2500)], novelty="ISM",
                         sq={"structural_category": "incomplete-splice_match", "coding": "coding"}),
                   Model(enst(900008), [(1000, 1200), (2000, 2300), (3800, 4000)])],
                  [Component(2500, "ISOFORT000000006", 0.5, 0.2), Component(4000, enst(900008), 0.5, 0.8)],
                  depth=120, stop=(2100, 2102), canonical_id=enst(900007),
                  extra_ref=[(enst(900008), [(1000, 1200), (2000, 2300), (3800, 4000)], "protein_coding")]))
    g.append(Gene("ENSG00000900009.1", "NMDG1", "chr4", "+", [(10000, 10300), (11000, 11200), (12000, 13000)],
                  [Model(enst(900009), [(10000, 10300), (11000, 11200), (12000, 13000)]),
                   Model(enst(900010), [(10000, 10300), (10800, 10900), (11000, 11200), (12000, 13000)],
                         ttype="nonsense_mediated_decay"),
                   Model("ISOFORT000000007", [(10000, 11200), (12000, 13000)], novelty="NIC",
                         sq={"structural_category": "novel_in_catalog", "subcategory": "intron_retention",
                             "coding": "coding"})],
                  [Component(13000, enst(900009), 0.6, 0.6), Component(13000, enst(900010), 0.25, 0.25),
                   Component(13000, "ISOFORT000000007", 0.15, 0.15)],
                  depth=200, stop=(12100, 12102), canonical_id=enst(900009),
                  tail_by_model={enst(900010): (120.0, -25.0), enst(900009): (100.0, -15.0),
                                 "ISOFORT000000007": (110.0, -20.0)},
                  extra_ref=[(enst(900010), [(10000, 10300), (10800, 10900), (11000, 11200), (12000, 13000)],
                              "nonsense_mediated_decay")]))
    for k, name in enumerate(["RPL5", "RPS3", "RPL11"]):
        s = 20000 + 5000 * k
        g.append(Gene(f"ENSG0000091{k:04d}.1", name, "chr5", "+", [(s, s + 800)], [Model(enst(910000 + k), [(s, s + 800)])],
                      [Component(s + 800, enst(910000 + k), 1.0, 1.0)], depth=150, tail_mean=60, tail_shift=-20,
                      tail_sd=25, canonical_id=enst(910000 + k), stop=(s + 500, s + 502)))
    g.append(Gene("ENSG00000198804.2", "MT-CO1", "chrM", "+", [(5904, 7445)], [Model(enst(361624), [(5904, 7445)])],
                  [Component(7445, enst(361624), 1.0, 1.0)], depth=150, tail_mean=50, tail_shift=10, tail_sd=15,
                  gtype="protein_coding", canonical_id=enst(361624), stop=(7443, 7445)))
    g.append(Gene("ENSG00000900011.1", "UTR5G", "chr7", "+", [(40000, 40300), (41000, 41200), (43000, 44000)],
                  [Model(enst(900011), [(40000, 40300), (41000, 41200), (43000, 44000)]),
                   Model("ISOFORT000000011", [(40000, 40600)], novelty="NNC",
                         sq={"structural_category": "novel_not_in_catalog", "coding": "non_coding"})],
                  [Component(44000, enst(900011), 0.95, 0.75), Component(40600, "ISOFORT000000011", 0.05, 0.25,
                                                                          "intron")],
                  depth=120, stop=(43100, 43102), start=(41050, 41052), canonical_id=enst(900011)))
    g.append(Gene("ENSG00000900012.1", "LNC1", "chr7", "+", [(50000, 50400), (51000, 53000)],
                  [Model(enst(900012), [(50000, 50400), (51000, 53000)], ttype="lncRNA"),
                   Model("ISOFORT000000012", [(50000, 50400), (51000, 52000)], ttype="lncRNA", novelty="ISM")],
                  [Component(52000, "ISOFORT000000012", 0.7, 0.35), Component(53000, enst(900012), 0.3, 0.65)],
                  depth=120, gtype="lncRNA", canonical_id=enst(900012)))
    g.append(Gene("ISOFORG000000001", "NOVEL1", "chr8", "-", [(20000, 21500), (22000, 22300)],
                  [Model("ISOFORT000000021", [(20000, 21500), (22000, 22300)], ttype="lncRNA", novelty="Intergenic"),
                   Model("ISOFORT000000022", [(20800, 21500), (22000, 22300)], ttype="lncRNA", novelty="Intergenic")],
                  [Component(20800, "ISOFORT000000022", 0.65, 0.3), Component(20000, "ISOFORT000000021", 0.35, 0.7)],
                  depth=120, gtype="lncRNA", in_reference=False, canonical_id="ISOFORT000000021"))
    g.append(Gene("ENSG00000900013.1", "TSS1", "chr9", "+", [(60000, 60200), (61000, 61200), (62000, 63000)],
                  [Model(enst(900013), [(60000, 60200), (61000, 61200), (62000, 63000)]),
                   Model("ISOFORT000000013", [(55000, 55150), (61000, 61200), (62000, 63000)], novelty="NNC")],
                  [Component(63000, enst(900013), 0.8, 0.4), Component(63000, "ISOFORT000000013", 0.2, 0.6)],
                  depth=120, stop=(62100, 62102), canonical_id=enst(900013)))
    g.append(Gene("ENSG00000900014.1", "TSS2", "chr9", "+", [(70000, 70200), (71000, 71200), (72000, 73000)],
                  [Model(enst(900014), [(70000, 70200), (71000, 71200), (72000, 73000)]),
                   Model("ISOFORT000000014", [(70500, 70600), (71000, 71200), (72000, 73000)], novelty="NNC")],
                  [Component(73000, enst(900014), 0.6, 0.9), Component(73000, "ISOFORT000000014", 0.4, 0.1)],
                  depth=120, stop=(72100, 72102), canonical_id=enst(900014)))
    if mixed:
        g = [x for x in g if x.name in MIXED_NAMED]
    for k in range(n_background):
        s = 1_000_000 + 20_000 * k
        tid = enst(920000 + k)
        tails = _mixed_tails(rng) if mixed else dict(tail_mean=80 + (k % 7) * 8, tail_shift=-15)
        g.append(Gene(f"ENSG0000092{k:04d}.1", f"GENE{k:03d}", "chr2", "+", [(s, s + 1500)], [Model(tid, [(s, s + 1500)])],
                      [Component(s + 1500, tid, 1.0, 1.0)], depth=60, canonical_id=tid, stop=(s + 900, s + 902),
                      gtype="lncRNA" if k % 10 == 9 else "protein_coding", **tails))
    g += _background_apa(n_apa_background, rng, mixed)
    return g


def _mixed_tails(rng) -> dict:
    """Tail parameters of a gene whose tail gets shorter, longer or stays, with its own response per unit."""
    mean = float(np.exp(rng.normal(np.log(95), 0.25)))
    shift = float(rng.choice([0.0, rng.normal(-12, 4), rng.normal(12, 4)], p=[0.3, 0.35, 0.35]))
    return dict(tail_mean=mean, tail_shift=shift, tail_sd=0.38 * mean, tail_skewed=True,
                unit_response=float(np.clip(np.exp(rng.normal(0, 0.6)), 0.2, 3.0)), unit_sd=float(rng.gamma(2.0, 2.5)),
                sample_sd=float(rng.gamma(2.0, 2.5)))


def _shifted(p: float, rng, share: float, size: float) -> float:
    """p moved on the logit scale, up or down by about `size`, for a `share` of the calls; otherwise p."""
    if rng.random() > share:
        return p
    logit = np.log(p / (1 - p)) + rng.choice([-1.0, 1.0]) * abs(rng.normal(size, 0.35 * size))
    return float(1 / (1 + np.exp(-logit)))


def _background_apa(n: int, rng, mixed: bool = False) -> list[Gene]:
    """Genes with a proximal and a distal 3'UTR site, one further poly(A) event (a composite-exon, skipped-exon,
    internal exonic or 5'UTR site) and a second start site (an upstream promoter, an alternative first exon or, when
    mixed, a start inside an internal exon). Not mixed: about two in seven shift each usage by a fixed amount.
    Mixed: about four in ten shift each usage by a gene-specific amount, usage varies per unit, and one gene in
    seven is a lncRNA."""
    out = []
    for k in range(n):
        b = 1_000_000 + 30_000 * k
        e1, e2, e3 = (b, b + 200), (b + 2000, b + 2200), (b + 4000, b + 6000)
        canon = f"ENST{960000 + k:011d}.1"
        short, ev, tss = (f"ISOFORT{700000 + 3 * k + j:09d}" for j in range(3))
        variant = k % 4
        event = {0: ([e1, (b + 2000, b + 2600)], b + 2600, "intron"),
                 1: ([e1, (b + 1200, b + 1400)], b + 1400, "intron"),
                 2: ([e1, (b + 2000, b + 2100)], b + 2100, "exon"),
                 3: ([(b, b + 600)], b + 600, "intron")}[variant]
        start_kind = (k // 4) % 3 if mixed else (1 if variant == 1 else 0)
        tss_exons = {0: [(b - 4000, b - 3850), e2, e3], 1: [(b + 3000, b + 3100), e3],
                     2: [(b + 2100, b + 2200), e3]}[start_kind]
        if mixed:
            pdui = float(np.clip(rng.beta(2.5, 2.5), 0.12, 0.88))
            pdui_t = _shifted(pdui, rng, 0.45, 0.7)
            x = rng.uniform(0.04, 0.15)
            x_t = _shifted(x, rng, 0.4, 0.9)
            y = rng.uniform(0.08, 0.3)
            y_t = _shifted(y, rng, 0.4, 0.8)
            tails = dict(_mixed_tails(rng), usage_sd=float(rng.gamma(2.0, 0.12)))
        else:
            pdui = rng.uniform(0.25, 0.75)
            pdui_t = pdui + rng.choice([0.0] * 5 + [0.2, -0.2]) + rng.normal(0, 0.02)
            x = rng.uniform(0.03, 0.1)
            x_t = x * rng.choice([1] * 5 + [3, 1 / 3])
            y = rng.uniform(0.05, 0.2)
            y_t = y * rng.choice([1] * 5 + [3, 1 / 3])
            tails = dict(tail_mean=float(rng.uniform(80, 120)),
                         tail_shift=float(rng.choice([0.0, -6.0, -12.0, -18.0]) + rng.normal(0, 3)))

        def shares(p, c, u):
            p = float(np.clip(p, 0.05, 0.95))
            t = 1 - c - u
            return [t * p, t * (1 - p), c, u]
        sc, st = shares(pdui, x, y), shares(pdui_t, x_t, y_t)
        noncoding = mixed and k % 7 == 6
        kind = "lncRNA" if noncoding else "protein_coding"
        out.append(Gene(f"ENSG0000096{k:04d}.1", f"BG{k + 1:03d}", "chr6", "+", [e1, e2, e3],
                        [Model(canon, [e1, e2, e3], ttype=kind),
                         Model(short, [e1, e2, (b + 4000, b + 4800)], ttype=kind, novelty="ISM"),
                         Model(ev, event[0], ttype=kind, novelty="NNC"),
                         Model(tss, tss_exons, ttype=kind, novelty="NNC")],
                        [Component(b + 6000, canon, sc[0], st[0]), Component(b + 4800, short, sc[1], st[1]),
                         Component(event[1], ev, sc[2], st[2], event[2]), Component(b + 6000, tss, sc[3], st[3])],
                        depth=int(rng.integers(60, 160)), gtype=kind,
                        stop=None if noncoding else (b + 4100, b + 4102),
                        start=(b + 2050, b + 2052) if variant == 3 and not noncoding else None,
                        canonical_id=canon, **tails))
    return out


def _attrs(d: dict) -> str:
    return " ".join(f'{k} "{v}";' for k, v in d.items())


def _gtf_lines(gene: Gene, transcripts, source: str, extra_attrs=None, canonical=None, stops=True) -> list[str]:
    """GTF lines of a gene and its transcripts [(tid, exons, type)]."""
    lo = min(s for _, ex, _ in transcripts for s, _ in ex)
    hi = max(e for _, ex, _ in transcripts for _, e in ex)
    base = {"gene_id": gene.gid}
    lines = ["\t".join([gene.chrom, source, "gene", str(lo), str(hi), ".", gene.strand, ".",
                        _attrs({**base, "gene_name": gene.name, "gene_type": gene.gtype})])]
    for tid, exons, ttype in transcripts:
        a = {**base, "transcript_id": tid, "gene_name": gene.name, "transcript_type": ttype}
        a.update((extra_attrs or {}).get(tid, {}))
        tag = ' tag "Ensembl_canonical";' if canonical is not None and tid == canonical else ""
        ex = sorted(exons)
        lines.append("\t".join([gene.chrom, source, "transcript", str(ex[0][0]), str(ex[-1][1]), ".", gene.strand, ".",
                                _attrs(a) + tag]))
        order = ex if gene.strand == "+" else ex[::-1]
        for i, (s, e) in enumerate(order, 1):
            lines.append("\t".join([gene.chrom, source, "exon", str(s), str(e), ".", gene.strand, ".",
                                    _attrs({**base, "transcript_id": tid, "gene_name": gene.name,
                                            "exon_number": i}) + tag]))
        if stops and tid == canonical and gene.stop is not None:
            lines.append("\t".join([gene.chrom, source, "stop_codon", str(gene.stop[0]), str(gene.stop[1]), ".",
                                    gene.strand, "0", _attrs({**base, "transcript_id": tid}) + tag]))
            st = gene.start
            if st is None:
                st = (ex[0][0] + 30, ex[0][0] + 32) if gene.strand == "+" else (ex[-1][1] - 32, ex[-1][1] - 30)
            lines.append("\t".join([gene.chrom, source, "start_codon", str(st[0]), str(st[1]), ".",
                                    gene.strand, "0", _attrs({**base, "transcript_id": tid}) + tag]))
    return lines


def _uuids(rng, n: int) -> list[str]:
    hi = rng.integers(0, 2 ** 63, n, dtype=np.int64)
    lo = rng.integers(0, 2 ** 63, n, dtype=np.int64)
    out = []
    for a, b in zip(hi, lo):
        h = f"{int(a):016x}{int(b):016x}"
        out.append(f"{h[:8]}-{h[8:12]}-4{h[13:16]}-{h[16:20]}-{h[20:]}")
    return out


def make_experiment(outdir, n_units: int = 6, n_replicates: int = 3, seed: int = 1, n_background: int = 60,
                    depth_scale: float = 1.0, n_apa_background: int = 0, control: str = "control",
                    treatment: str = "treated", paired: bool = True, tail_table: bool = False,
                    effects: str = "known") -> Path:
    """Write the experiment and its config to `outdir`; returns the config path. paired=False writes a sample sheet
    without pairs. tail_table=True leaves `polya_length` out of the read annotation and writes a per-read tail table
    with mapping qualities instead. effects: "known" (the named genes and uniform background effects) or "mixed"
    (gene-specific changes in both directions, see the module description)."""
    if effects not in ("known", "mixed"):
        raise ValueError(f"effects must be 'known' or 'mixed', not {effects!r}")
    mixed = effects == "mixed"
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    genes = _genes(n_background, n_apa_background, np.random.default_rng(seed + 1000), mixed)
    unit_rng = np.random.default_rng(seed + 2000)
    unit_shift = unit_rng.permutation(np.linspace(-16, 14, n_units)) if mixed else np.zeros(n_units)
    unit_base = unit_rng.normal(0, 8, n_units) if mixed else np.zeros(n_units)
    ref = ["##description: IsoEnds synthetic reference"]
    for g in genes:
        if g.in_reference:
            ref += _gtf_lines(g, [(g.canonical_id, g.canonical, g.gtype)] + list(g.extra_ref), "SYNTH",
                              canonical=g.canonical_id)
    (out / "reference.gtf").write_text("\n".join(ref) + "\n")
    model_of = {g.gid: {m.tid: m for m in g.models} for g in genes}
    sheet_rows, runs = [], {}
    for u in range(n_units):
        unit = f"LINE{u + 1}"
        udir = out / unit
        udir.mkdir(exist_ok=True)
        offset = 3 * u
        iso, char = [], ["##gtf-version 2.2"]
        for g in genes:
            tx = [(m.tid, list(m.exons), m.ttype) for m in g.models]
            iso += _gtf_lines(g, tx, "IsoForge", stops=False)
            sq = {}
            for m in g.models:
                a = {"structural_category": "full-splice_match" if m.novelty == "Known" else "novel_in_catalog",
                     "subcategory": "reference_match", "coding": "coding" if g.gtype == "protein_coding" else "non_coding",
                     "predicted_NMD": "True" if m.ttype == "nonsense_mediated_decay" else "False",
                     "ref_transcript_type": m.ttype if m.tid.startswith("ENST") else g.gtype}
                a.update(m.sq)
                sq[m.tid] = a
            char += _gtf_lines(g, tx, "IsoForge", extra_attrs=sq, stops=False)
        (udir / f"{unit}_isoforge.gtf").write_text("\n".join(iso) + "\n")
        (udir / f"{unit}_isoforge_sqanti3.gtf").write_text("\n".join(char) + "\n")
        read_frames, tail_frames = [], []
        response = ({g.gid: g.unit_response * unit_shift[u] + unit_rng.normal(0, g.unit_sd) for g in genes}
                    if mixed else {})
        usage = {g.gid: unit_rng.normal(0, g.usage_sd, len(g.components)) for g in genes if g.usage_sd}
        for k in range(n_replicates):
            replicate_noise = rng.normal(0, 3)
            for cond, treated in ((control, False), (treatment, True)):
                dataset = f"{unit}_{'Trt' if treated else 'Ctrl'}_Bio{k + 1}"
                sheet_rows.append(dict(dataset=dataset, unit=unit, condition=cond, pair=str(k + 1), run=unit))
                for g in genes:
                    n = int(rng.poisson(g.depth * depth_scale))
                    if n == 0:
                        continue
                    shares = np.array([c.share_treated if treated else c.share_control for c in g.components])
                    if g.usage_sd:
                        shares = shares * np.exp((usage[g.gid] if treated else 0) + rng.normal(0, 0.08, len(shares)))
                    comp = rng.choice(len(g.components), size=n, p=shares / shares.sum())
                    cs = [g.components[i] for i in comp]
                    ms = [model_of[g.gid][c.model] for c in cs]
                    mean = np.array([g.tail_by_model.get(c.model, (g.tail_mean, g.tail_shift))[0] for c in cs])
                    shift = np.array([g.tail_by_model.get(c.model, (g.tail_mean, g.tail_shift))[1] for c in cs])
                    centre = mean + unit_base[u] + (shift + response.get(g.gid, 0.0) + replicate_noise * 0.3
                                                    if treated else 0)
                    if mixed:
                        centre = np.maximum(centre + rng.normal(0, g.sample_sd), 5)
                    tail = (rng.gamma((centre / g.tail_sd) ** 2, g.tail_sd ** 2 / centre) if g.tail_skewed
                            else rng.normal(centre, g.tail_sd))
                    tail = np.maximum(1, np.round(tail)).astype(int)
                    tail = np.where(rng.random(n) < 0.02, -1, tail)
                    sgn = 1 if g.strand == "+" else -1
                    site = np.array([c.site for c in cs]) + sgn * offset + rng.integers(-4, 5, n)
                    five = np.array([min(s_ for s_, _ in m.exons) if g.strand == "+" else max(e for _, e in m.exons)
                                     for m in ms]) + rng.integers(-3, 4, n)
                    start = np.where(sgn > 0, five, site)
                    end = np.where(sgn > 0, site, five)
                    ids = _uuids(rng, n)
                    frame = pd.DataFrame({
                        "read_name": [f"{k}:{i}|{x}" for i, x in enumerate(ids)], "dataset": dataset,
                        "genome_build": "synth", "chromosome": g.chrom, "read_start": start, "read_end": end,
                        "strand": g.strand, "n_exons": [len(m.exons) for m in ms], "read_length": np.abs(end - start),
                        "annot_gene_id": g.gid, "annot_transcript_id": [c.model for c in cs], "annot_gene_name": g.name,
                        "transcript_novelty": [m.novelty for m in ms], "gene_novelty": "Known",
                        "fraction_As": np.round(rng.uniform(0, 0.4, n), 2), "tss_site": five.astype(float),
                        "tss_feature": "five_prime_utr", "polya_site": site.astype(float),
                        "polya_feature": [c.feature for c in cs],
                        "polya_signal": [c.signal if r > 0.05 else None for c, r in zip(cs, rng.random(n))],
                        "cluster_supported": True, "polya_length": tail})
                    if tail_table:
                        keep = rng.random(n) < 0.97
                        tail_frames.append(pd.DataFrame({
                            "read_id": np.asarray(ids)[keep], "ePolyA_tail_length": tail[keep],
                            "mapping_quality": rng.choice([0, 1, 60], size=n, p=[0.3, 0.1, 0.6])[keep]}))
                        frame = frame.drop(columns="polya_length")
                    read_frames.append(frame)
        ra = pd.concat(read_frames, ignore_index=True)
        for c in ("dataset", "genome_build", "chromosome", "strand", "transcript_novelty", "gene_novelty"):
            ra[c] = ra[c].astype("category")
        pq.write_table(pa.Table.from_pandas(ra, preserve_index=False), udir / f"{unit}_isoforge_read_annot.parquet",
                       row_group_size=50_000)
        runs[unit] = {"results": unit}
        if tail_table:
            pd.concat(tail_frames, ignore_index=True).to_csv(udir / f"{unit}_tails.csv", index=False)
            runs[unit]["tails"] = f"{unit}/{unit}_tails.csv"
    sheet = pd.DataFrame(sheet_rows)
    (sheet if paired else sheet.drop(columns="pair")).to_csv(out / "samples.tsv", sep="\t", index=False)
    config = {
        "name": "synthetic", "outdir": "isoends_out", "samples": "samples.tsv", "control": control,
        "treatment": treatment, "reference": {"gtf": "reference.gtf"}, "runs": runs, "usage_genes": ["NMDG1"],
        "figures": {"genes": ["TANDEM1", "CE1"], "formats": ["pdf", "png"], "dpi": 150},
        "params": {"stats": {"limma_backend": "python"}},
    }
    path = out / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return path

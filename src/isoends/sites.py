"""Read ends as sites: cluster them, then place every cluster on its gene's reference transcript.

Sites are the ends IsoForge gave each read (`polya_site`, `tss_site`). Within a gene, sites closer than `cluster_nt`
to the previous site join its cluster. A cluster's position is the median site of its reads; its feature and
poly(A) signal are the most common among its reads.

The reference transcript of a gene is its Ensembl-canonical transcript; with `novel_gene_models`, a gene outside the
reference is placed on its best-supported IsoForge model. Poly(A) site classes, positions measured 5' -> 3':

  terminal             in the last exon, or up to `downstream_nt` past it
  composite exon       in an intron, on models whose terminal exon begins at or before the upstream exon's end
  skipped exon         in an intron, on models whose terminal exon begins inside the intron
  intronic, no model   in an intron, with no model of the gene ending within `model_window_nt`
  internal exonic      inside an exon other than the last
  upstream, far downstream, no canonical transcript, opposite strand: not tested

Composite or skipped is decided by the majority of the models ending at the site (every model on the chromosome and
strand, or with `intronic_models: gene` only the gene's own); a tie is composite.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .hist import grouped_quantiles

INTRONIC = ("composite exon", "skipped exon", "intronic, no model")
CLASSES = ("terminal", "composite exon", "skipped exon", "intronic, no model", "internal exonic", "upstream",
           "far downstream", "no canonical transcript", "opposite strand")
TSS_CLASSES = ("canonical first exon", "upstream TSS", "alternative first exon", "intronic TSS, exon extension",
               "intronic TSS, no model", "internal exonic TSS", "far upstream", "downstream of the gene",
               "no canonical transcript", "opposite strand")


def _mode_by_reads(keys: pd.Series, values: pd.Series, weights: pd.Series, missing: str = "none") -> pd.Series:
    """Per key, the value with the most reads (ties: alphabetically first), ignoring `missing`."""
    d = pd.DataFrame({"k": keys.to_numpy(), "v": values.astype(str).to_numpy(), "w": weights.to_numpy()})
    d = d[d.v != missing]
    if d.empty:
        return pd.Series(dtype=str)
    s = d.groupby(["k", "v"], observed=True).w.sum().reset_index()
    s = s.sort_values(["k", "w", "v"], ascending=[True, False, True], kind="stable")
    return s.drop_duplicates("k").set_index("k").v


def cluster_sites(per_sample: dict[str, pd.DataFrame], cluster_nt: int = 25) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Cluster the sites of one run. per_sample: {dataset: site counts from ingestion}.

    Returns (clusters, counts). clusters: cluster, gene_id, chrom, strand, site (median over reads), start, end,
    peak, feature, hexamer, reads, reads_tail, mean_fraction_as, share_high_a. counts: cluster, dataset, reads,
    reads_tail."""
    frames = [t.assign(dataset=d) for d, t in per_sample.items() if len(t)]
    if not frames:
        return (pd.DataFrame(columns=["cluster", "gene_id", "chrom", "strand", "site", "reads"]),
                pd.DataFrame(columns=["cluster", "dataset", "reads", "reads_tail"]))
    s = pd.concat(frames, ignore_index=True)
    for c, default in (("signal", "none"), ("feature", "none"), ("n_reads_tail", 0), ("sum_fraction_as", 0.0),
                       ("n_high_a", 0)):
        if c not in s:
            s[c] = default
    for c in ("gene", "chrom", "strand", "feature", "signal", "dataset"):
        s[c] = s[c].astype(str)
    by_site = s.groupby(["gene", "chrom", "strand", "site"], sort=True).agg(reads=("n_reads", "sum")).reset_index()
    new = (by_site.gene.ne(by_site.gene.shift()) | by_site.chrom.ne(by_site.chrom.shift())
           | by_site.strand.ne(by_site.strand.shift()) | (by_site.site.diff() > cluster_nt))
    by_site["cluster"] = np.cumsum(new.to_numpy())
    s = s.merge(by_site[["gene", "chrom", "strand", "site", "cluster"]], on=["gene", "chrom", "strand", "site"])

    agg = s.groupby("cluster").agg(gene_id=("gene", "first"), chrom=("chrom", "first"), strand=("strand", "first"),
                                   start=("site", "min"), end=("site", "max"), reads=("n_reads", "sum"),
                                   reads_tail=("n_reads_tail", "sum"), sum_fa=("sum_fraction_as", "sum"),
                                   n_high_a=("n_high_a", "sum"))
    median = grouped_quantiles(s.cluster, s.site.to_numpy(), s.n_reads.to_numpy(), qs=(0.5,))
    agg["site"] = median.q50.reindex(agg.index).to_numpy()
    peak = by_site.sort_values(["cluster", "reads", "site"], ascending=[True, False, True]).drop_duplicates("cluster")
    agg["peak"] = peak.set_index("cluster").site.reindex(agg.index).to_numpy()
    agg["feature"] = _mode_by_reads(s.cluster, s.feature, s.n_reads).reindex(agg.index).fillna("").to_numpy()
    agg["hexamer"] = _mode_by_reads(s.cluster, s.signal, s.n_reads).reindex(agg.index).fillna("").to_numpy()
    agg["mean_fraction_as"] = agg.sum_fa / agg.reads
    agg["share_high_a"] = agg.n_high_a / agg.reads
    clusters = agg.drop(columns=["sum_fa", "n_high_a"]).reset_index()
    counts = (s.groupby(["cluster", "dataset"]).agg(reads=("n_reads", "sum"), reads_tail=("n_reads_tail", "sum"))
              .reset_index())
    cols = ["cluster", "gene_id", "chrom", "strand", "site", "start", "end", "peak", "feature", "hexamer", "reads",
            "reads_tail", "mean_fraction_as", "share_high_a"]
    return clusters[cols], counts


def as_mask(condition) -> np.ndarray:
    """A condition as a boolean array, with missing values false."""
    return pd.Series(condition).astype("boolean").fillna(False).to_numpy(dtype=bool)


def _sgn(strand) -> np.ndarray:
    return np.where(np.asarray(strand) == "+", 1, -1)


def reference_models(canonical: pd.DataFrame, genes, model_exons: pd.DataFrame,
                     model_reads: pd.Series | None = None, novel: bool = True) -> pd.DataFrame:
    """The Ensembl-canonical exons plus, with `novel`, the exons of the best-supported IsoForge model of each gene
    without a canonical transcript. Adds canonical_source ("Ensembl canonical" or "IsoForge model")."""
    can = canonical.assign(canonical_source="Ensembl canonical")
    if not novel:
        return can
    missing = set(map(str, genes)) - set(can.gene_id)
    ex = model_exons[model_exons.gene_id.isin(missing)]
    if ex.empty:
        return can
    reads = model_reads if model_reads is not None else pd.Series(dtype=float)
    ex = ex.assign(_reads=ex.transcript_id.map(reads).fillna(0), _len=ex.end - ex.start + 1)
    per_tx = ex.groupby(["gene_id", "transcript_id"], as_index=False).agg(reads=("_reads", "first"),
                                                                       length=("_len", "sum"))
    best = (per_tx.sort_values(["gene_id", "reads", "length", "transcript_id"], ascending=[True, False, False, True])
            .drop_duplicates("gene_id"))
    own = ex[ex.transcript_id.isin(set(best.transcript_id))].drop(columns=["_reads", "_len"]).copy()
    own["_k"] = np.where(own.strand == "+", own.start, -own.end)
    own = own.sort_values(["gene_id", "_k"]).drop(columns="_k")
    own["rank"] = own.groupby("gene_id").cumcount() + 1
    own["n_exons"] = own.groupby("gene_id").gene_id.transform("size")
    own["canonical_source"] = "IsoForge model"
    return pd.concat([can, own[can.columns]], ignore_index=True)


def site_region(clusters: pd.DataFrame, start: pd.DataFrame, stop: pd.DataFrame) -> np.ndarray:
    """Where each site lies on its gene's canonical transcript: 5'UTR (upstream of the start codon), CDS or 3'UTR
    (downstream of the stop codon); "non-coding" when the transcript has no CDS."""
    st = start.drop_duplicates("gene_id").set_index("gene_id")
    sp = stop.drop_duplicates("gene_id").set_index("gene_id")
    sg = _sgn(clusters.strand)
    start5 = np.where(sg > 0, clusters.gene_id.map(st.start), -clusters.gene_id.map(st.end))
    stop3 = np.where(sg > 0, clusters.gene_id.map(sp.end), -clusters.gene_id.map(sp.start))
    pos = clusters.site.to_numpy() * sg
    return np.select([as_mask(np.isnan(start5) | np.isnan(stop3)), as_mask(pos < start5), as_mask(pos <= stop3)],
                     ["non-coding", "5'UTR", "CDS"], default="3'UTR")


def canonical_frame(canonical: pd.DataFrame) -> pd.DataFrame:
    """Canonical exons with 5' -> 3' coordinates s5 <= s3 (minus-strand positions negated)."""
    plus = canonical.strand == "+"
    return canonical.assign(s5=np.where(plus, canonical.start, -canonical.end),
                            s3=np.where(plus, canonical.end, -canonical.start))


def _place(clusters: pd.DataFrame, canonical: pd.DataFrame):
    """Join each cluster to its gene's reference transcript. Returns (clusters with pos, first5, last3, exon_rank,
    intron and the reference columns; the reference exons in 5' -> 3' coordinates; has; same)."""
    d = clusters.copy()
    ex = canonical_frame(canonical)
    agg = dict(canonical_transcript=("transcript_id", "first"), cchrom=("chrom", "first"),
               cstrand=("strand", "first"), first5=("s5", "min"), last3=("s3", "max"),
               n_canonical_exons=("n_exons", "first"))
    if "canonical_source" in ex:
        agg["canonical_source"] = ("canonical_source", "first")
    d = d.join(ex.groupby("gene_id").agg(**agg), on="gene_id")
    d["pos"] = d.site * _sgn(d.strand)
    has = d.canonical_transcript.notna()
    same = has & (d.strand == d.cstrand) & (d.chrom == d.cchrom)
    m = d.loc[same, ["cluster", "gene_id", "pos"]].merge(ex[["gene_id", "s5", "s3", "rank"]], on="gene_id")
    d["exon_rank"] = d.cluster.map(m[(m.s5 <= m.pos) & (m.pos <= m.s3)].groupby("cluster")["rank"].min())
    d["intron"] = d.cluster.map(m[m.s3 < m.pos].groupby("cluster")["rank"].max())
    return d, ex, has, same


def _models_by_gene(models: pd.DataFrame, sort_by: str, columns, by="gene_id") -> dict:
    """{key: arrays of `columns`, sorted by `sort_by`} for looking up the models near a position; the key is the
    gene, or any other column(s) given as `by`."""
    return {g: tuple(t[c].to_numpy() for c in columns) for g, t in models.sort_values(sort_by).groupby(by)}


def classify_sites(clusters: pd.DataFrame, canonical: pd.DataFrame, terminal: pd.DataFrame,
                   downstream_nt: int = 5000, model_window_nt: int = 25, intronic_models: str = "any") -> pd.DataFrame:
    """Add site_class, canonical_transcript, exon_rank, intron (number of the canonical intron, 5' -> 3'), n_models
    and n_models_through to the poly(A) clusters. terminal: the run's model terminal exons (gtf.terminal_exons).
    intronic_models: "any" counts every model ending at an intronic site on its chromosome and strand, "gene" only
    the models of the site's gene."""
    if intronic_models not in ("any", "gene"):
        raise ValueError(f"intronic_models must be 'any' or 'gene', not {intronic_models!r}")
    d, ex, has, same = _place(clusters, canonical)
    conditions = [~has, ~same, d.pos < d.first5, d.pos > d.last3 + downstream_nt, d.pos > d.last3,
                  d.exon_rank == d.n_canonical_exons, d.exon_rank.notna()]
    d["site_class"] = np.select(
        [as_mask(c) for c in conditions],
        ["no canonical transcript", "opposite strand", "upstream", "far downstream", "terminal", "terminal",
         "internal exonic"], default="intronic")
    d.loc[d.site_class != "intronic", "intron"] = np.nan
    d.loc[~d.site_class.isin(["internal exonic", "terminal"]), "exon_rank"] = np.nan

    t = terminal.assign(p5=terminal.exon5 * _sgn(terminal.strand), psite=terminal.site * _sgn(terminal.strand))
    own = intronic_models == "gene"
    models = _models_by_gene(t, "psite", ("psite", "p5"), "gene_id" if own else ["chrom", "strand"])
    upstream_exon_end = ex.set_index(["gene_id", "rank"]).s3
    intronic = d.index[d.site_class == "intronic"]
    keys = d.gene_id[intronic] if own else zip(d.chrom[intronic], d.strand[intronic])
    sub, n_models, n_through = [], [], []
    for key, gene, pos, intron in zip(keys, d.gene_id[intronic], d.pos[intronic], d.intron[intronic]):
        psite, p5 = models.get(key, (np.empty(0), np.empty(0)))
        lo, hi = np.searchsorted(psite, [pos - model_window_nt, pos + model_window_nt + 1])
        through = int((p5[lo:hi] <= upstream_exon_end.loc[(gene, int(intron))]).sum())
        sub.append("intronic, no model" if hi <= lo else "composite exon" if 2 * through >= hi - lo
                   else "skipped exon")
        n_models.append(hi - lo)
        n_through.append(through)
    d["n_models"] = np.nan
    d["n_models_through"] = np.nan
    d.loc[intronic, "site_class"] = sub
    d.loc[intronic, "n_models"] = n_models
    d.loc[intronic, "n_models_through"] = n_through
    return d.drop(columns=["cchrom", "cstrand", "first5", "last3", "pos"])


def classify_tss(clusters: pd.DataFrame, canonical: pd.DataFrame, first: pd.DataFrame, upstream_nt: int = 1000,
                 max_upstream_nt: int = 50_000, model_window_nt: int = 50) -> pd.DataFrame:
    """Add site_class, canonical_transcript, exon_rank, intron, n_models and n_models_own_first_exon to the TSS
    clusters. first: the run's model first exons (gtf.first_exons). Classes, 5' -> 3':

      canonical first exon           in the first exon, or up to `upstream_nt` upstream of it
      upstream TSS                   further upstream, up to `max_upstream_nt`
      alternative first exon         in an intron, on models whose first exon ends inside that intron
      intronic TSS, exon extension   in an intron, on models whose first exon runs into the next exon
      intronic TSS, no model         in an intron, with no model of the gene starting within `model_window_nt`
      internal exonic TSS            inside an exon other than the first
    """
    d, ex, has, same = _place(clusters, canonical)
    d["first_exon3"] = d.gene_id.map(ex[ex["rank"] == 1].set_index("gene_id").s3)
    conditions = [~has, ~same, d.pos < d.first5 - max_upstream_nt, d.pos < d.first5 - upstream_nt,
                  d.pos <= d.first_exon3, d.pos > d.last3, d.exon_rank.notna()]
    d["site_class"] = np.select(
        [as_mask(c) for c in conditions],
        ["no canonical transcript", "opposite strand", "far upstream", "upstream TSS", "canonical first exon",
         "downstream of the gene", "internal exonic TSS"], default="intronic")
    d.loc[d.site_class != "intronic", "intron"] = np.nan

    f = first.assign(p5=first.site * _sgn(first.strand), p3=first.exon3 * _sgn(first.strand))
    models = _models_by_gene(f, "p5", ("p5", "p3"))
    next_exon_start = ex.set_index(["gene_id", "rank"]).s5
    intronic = d.index[d.site_class == "intronic"]
    sub, n_models, n_own = [], [], []
    for gene, pos, intron in zip(d.gene_id[intronic], d.pos[intronic], d.intron[intronic]):
        p5, p3 = models.get(gene, (np.empty(0), np.empty(0)))
        lo, hi = np.searchsorted(p5, [pos - model_window_nt, pos + model_window_nt + 1])
        own = int((p3[lo:hi] < next_exon_start.get((gene, int(intron) + 1), np.inf)).sum())
        sub.append("intronic TSS, no model" if hi <= lo else "alternative first exon" if 2 * own > hi - lo
                   else "intronic TSS, exon extension")
        n_models.append(hi - lo)
        n_own.append(own)
    d["n_models"] = np.nan
    d["n_models_own_first_exon"] = np.nan
    d.loc[intronic, "site_class"] = sub
    d.loc[intronic, "n_models"] = n_models
    d.loc[intronic, "n_models_own_first_exon"] = n_own
    return d.drop(columns=["cchrom", "cstrand", "first5", "last3", "first_exon3", "pos"])


def same_exon(pairs: pd.DataFrame, ref_exons: pd.DataFrame, downstream_nt: int = 5000) -> pd.Series:
    """For each row (gene_id, proximal, distal): does one annotated exon of the gene hold both sites? The last exon
    of each reference transcript is extended `downstream_nt` past its 3' end. Missing for genes not in the
    reference."""
    ex = ref_exons[ref_exons.gene_id.isin(set(pairs.gene_id))]
    plus = ex.strand == "+"
    start = np.where(~plus & ex.is_last, ex.start - downstream_nt, ex.start)
    end = np.where(plus & ex.is_last, ex.end + downstream_nt, ex.end)
    ex = pd.DataFrame({"gene_id": ex.gene_id.to_numpy(), "start": start, "end": end}).drop_duplicates()
    q = pairs[["gene_id"]].assign(lo=np.minimum(pairs.proximal, pairs.distal),
                                  hi=np.maximum(pairs.proximal, pairs.distal), row=np.arange(len(pairs)))
    m = q.merge(ex, on="gene_id")
    held = np.zeros(len(pairs), bool)
    held[m[(m.start <= m.lo) & (m.end >= m.hi)].row.unique()] = True
    known = pairs.gene_id.isin(set(ex.gene_id)).to_numpy()
    return pd.Series(held, index=pairs.index).astype("boolean").where(known)


def split_by_model_intron(pairs: pd.DataFrame, model_exons: pd.DataFrame, terminal: pd.DataFrame,
                          model_window_nt: int = 25) -> pd.Series:
    """For each row (gene_id, proximal, distal): the share of the gene's models ending at the distal site that have
    an intron over the proximal site. NaN when no model of the gene ends at the distal site."""
    out = pd.Series(np.nan, index=pairs.index)
    by_gene = {g: t for g, t in terminal[terminal.gene_id.isin(set(pairs.gene_id))].groupby("gene_id")}
    candidates = {}
    for i, gene, distal in zip(pairs.index, pairs.gene_id, pairs.distal):
        t = by_gene.get(gene)
        if t is None:
            continue
        ids = t.transcript_id[(t.site - distal).abs() <= model_window_nt].tolist()
        if ids:
            candidates[i] = ids
    needed = {tid for ids in candidates.values() for tid in ids}
    if not needed:
        return out
    ex = model_exons[model_exons.transcript_id.isin(needed)].sort_values(["transcript_id", "start"])
    chains = {tid: (e.start.to_numpy(), e.end.to_numpy()) for tid, e in ex.groupby("transcript_id")}
    for i, ids in candidates.items():
        p = pairs.at[i, "proximal"]
        split = sum(1 for tid in ids if np.any((chains[tid][1][:-1] < p) & (p < chains[tid][0][1:])))
        out[i] = split / len(ids)
    return out


def distance_from_stop(gene_ids, sites, canonical: pd.DataFrame, stop: pd.DataFrame,
                       downstream_nt: int = 5000) -> np.ndarray:
    """Distance (nt) along the spliced canonical transcript from the base after its stop codon to each site; the
    last exon is extended to sites up to `downstream_nt` past it. NaN for non-coding genes and for sites before the
    stop codon or in an intron."""
    wanted = set(gene_ids)
    exons = {g: e for g, e in canonical[canonical.gene_id.isin(wanted)].groupby("gene_id")}
    stops = stop[stop.gene_id.isin(wanted)].drop_duplicates("gene_id").set_index("gene_id")
    out = np.full(len(sites), np.nan)
    for k, (g, pos) in enumerate(zip(gene_ids, sites)):
        if g not in stops.index or g not in exons or pd.isna(pos):
            continue
        e = exons[g]
        blocks = sorted(zip(e.start.to_numpy(), e.end.to_numpy()))
        if e.strand.iat[0] == "+":
            first = int(stops.at[g, "end"]) + 1
            if pos < first or pos > blocks[-1][1] + downstream_nt:
                continue
            if pos > blocks[-1][1]:
                blocks[-1] = (blocks[-1][0], pos)
            if any(s <= pos <= t for s, t in blocks):
                out[k] = sum(max(0, min(t, pos) - max(s, first) + 1) for s, t in blocks) - 1
        else:
            last = int(stops.at[g, "start"]) - 1
            if pos > last or pos < blocks[0][0] - downstream_nt:
                continue
            if pos < blocks[0][0]:
                blocks[0] = (pos, blocks[0][1])
            if any(s <= pos <= t for s, t in blocks):
                out[k] = sum(max(0, min(t, last) - max(s, pos) + 1) for s, t in blocks) - 1
    return out

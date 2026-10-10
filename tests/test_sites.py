"""Site clustering, the class rules on hand-built models, the same-exon check and matching pairs across units."""
import numpy as np
import pandas as pd

from isoends.apa import match_pairs, top_two_within_unit
from isoends.config import ApaParams, Design, StatsParams
from isoends.gtf import terminal_exons
from isoends.sites import (classify_sites, cluster_sites, distance_from_stop, same_exon, split_by_model_intron)


def canonical(gene, strand, exons, chrom="chr1"):
    ex = sorted(exons, key=lambda e: e[0], reverse=(strand == "-"))
    return pd.DataFrame({"gene_id": gene, "transcript_id": f"{gene}-T", "chrom": chrom, "strand": strand,
                         "start": [s for s, _ in ex], "end": [e for _, e in ex], "rank": range(1, len(ex) + 1),
                         "n_exons": len(ex)})


def models(rows):
    """rows: (transcript_id, gene, strand, [exons]) -> model exons."""
    out = []
    for tid, gene, strand, exons in rows:
        for s, e in exons:
            out.append(dict(transcript_id=tid, gene_id=gene, chrom="chr1", strand=strand, start=s, end=e))
    return pd.DataFrame(out)


def clusters(rows):
    return pd.DataFrame([dict(cluster=i, gene_id=g, chrom="chr1", strand=st, site=float(s)) for i, (g, st, s)
                         in enumerate(rows)])


CAN = canonical("G", "+", [(1000, 1200), (2000, 2200), (4000, 5000)])


def test_classes_on_hand_built_models():
    mods = models([
        ("canon", "G", "+", [(1000, 1200), (2000, 2200), (4000, 5000)]),
        ("readthrough", "G", "+", [(1000, 1200), (2000, 2600)]),
        ("alt_last", "G", "+", [(1000, 1200), (3000, 3300)]),
        ("internal", "G", "+", [(1000, 1200), (2000, 2100)]),
    ])
    term = terminal_exons(mods)
    c = clusters([("G", "+", 5000), ("G", "+", 7000), ("G", "+", 2600), ("G", "+", 3300), ("G", "+", 2100),
                  ("G", "+", 1600), ("G", "+", 500), ("G", "+", 12000), ("G", "-", 4500), ("OTHER", "+", 100)])
    res = classify_sites(c, CAN, term, downstream_nt=5000, model_window_nt=25).set_index("cluster").site_class
    assert list(res) == ["terminal", "terminal", "composite exon", "skipped exon", "internal exonic",
                         "intronic, no model", "upstream", "far downstream", "opposite strand",
                         "no canonical transcript"]


def test_composite_vs_skipped_is_a_majority_ties_composite():
    mods = models([("rt", "G", "+", [(1000, 1200), (2000, 2600)]),
                   ("alt", "G", "+", [(1000, 1200), (2400, 2610)])])
    res = classify_sites(clusters([("G", "+", 2605)]), CAN, terminal_exons(mods))
    assert res.site_class.iat[0] == "composite exon"
    assert res.n_models.iat[0] == 2 and res.n_models_through.iat[0] == 1
    mods = models([("alt1", "G", "+", [(1000, 1200), (2400, 2610)]), ("alt2", "G", "+", [(1000, 1200), (2500, 2600)]),
                   ("rt", "G", "+", [(1000, 1200), (2000, 2600)])])
    assert classify_sites(clusters([("G", "+", 2605)]), CAN, terminal_exons(mods)).site_class.iat[0] == "skipped exon"


def test_which_models_decide_the_class():
    """Every model ending at the site on its chromosome and strand gives it a class; with intronic_models="gene"
    a model of another gene does not."""
    mods = models([("other", "H", "+", [(1000, 1200), (2400, 2610)])])
    res = classify_sites(clusters([("G", "+", 2605)]), CAN, terminal_exons(mods))
    assert res.site_class.iat[0] == "skipped exon" and res.n_models.iat[0] == 1
    res = classify_sites(clusters([("G", "+", 2605)]), CAN, terminal_exons(mods), intronic_models="gene")
    assert res.site_class.iat[0] == "intronic, no model" and res.n_models.iat[0] == 0
    minus = models([("other", "H", "-", [(2400, 2610), (3000, 3100)])])
    res = classify_sites(clusters([("G", "+", 2605)]), CAN, terminal_exons(minus))
    assert res.site_class.iat[0] == "intronic, no model"
    mods = models([("other", "H", "+", [(1000, 1200), (2000, 2300), (3800, 4000)])])
    pairs = pd.DataFrame({"gene_id": ["G"], "proximal": [2500], "distal": [4000]})
    assert np.isnan(split_by_model_intron(pairs, mods, terminal_exons(mods)).iat[0])


def test_equally_used_sites_take_the_lower_coordinate():
    sites = pd.DataFrame({"cluster": [1, 2, 3, 4, 5, 6], "gene_id": ["M"] * 3 + ["P"] * 3, "chrom": "chr1",
                          "strand": ["-"] * 3 + ["+"] * 3, "site": [100.0, 200.0, 300.0, 1100.0, 1200.0, 1300.0],
                          "hexamer": "AATAAA"})
    reads = {1: 60, 2: 20, 3: 20, 4: 20, 5: 20, 6: 60}
    counts = pd.DataFrame([dict(cluster=c, dataset=d, reads=n) for c, n in reads.items() for d in ("c1", "t1", "c2", "t2")])
    design = Design("U", True, ("c1", "c2"), ("t1", "t2"), (("1", "c1", "t1"), ("2", "c2", "t2")))
    t = top_two_within_unit("U", sites, counts, design, ApaParams(), StatsParams(),
                            pd.Series(True, index=sites.cluster)).set_index("gene_id")
    assert (t.loc["M", "proximal_site"], t.loc["M", "distal_site"]) == (200.0, 100.0)
    assert (t.loc["P", "proximal_site"], t.loc["P", "distal_site"]) == (1100.0, 1300.0)


def test_minus_strand_classes():
    can = canonical("M", "-", [(4000, 5000), (2000, 2200), (1000, 1200)])
    mods = models([("rt", "M", "-", [(1600, 2200), (4000, 5000)])])
    res = classify_sites(clusters([("M", "-", 1000), ("M", "-", 1600), ("M", "-", 3000), ("M", "-", 2100),
                                   ("M", "-", 600)]), can, terminal_exons(mods))
    assert list(res.site_class) == ["terminal", "composite exon", "intronic, no model", "internal exonic", "terminal"]
    assert res.intron.iat[1] == 2 and res.intron.iat[2] == 1


def test_clustering_chains_within_25_nt_per_gene():
    s = pd.DataFrame({"gene": ["G"] * 5 + ["H"], "chrom": "chr1", "strand": "+",
                      "site": [100, 120, 140, 170, 171, 125], "feature": "three_prime_utr", "signal": "AATAAA",
                      "n_reads": [5, 1, 1, 10, 2, 3], "n_reads_tail": 1, "sum_fraction_as": 0.1, "n_high_a": 0})
    cl, counts = cluster_sites({"s1": s}, cluster_nt=25)
    g = cl[cl.gene_id == "G"].sort_values("start")
    assert list(zip(g.start, g.end)) == [(100, 140), (170, 171)]
    assert list(g.reads) == [7, 12]
    assert g.site.iat[0] == 100.0
    assert cl[cl.gene_id == "H"].reads.iat[0] == 3
    assert counts.reads.sum() == 22


def test_same_exon_check_and_split_models():
    ref = pd.DataFrame({"gene_id": "G", "transcript_id": ["a", "a", "b", "b", "b"],
                        "start": [1000, 2000, 1000, 2000, 3800], "end": [1200, 3000, 1200, 2300, 4000],
                        "strand": "+", "is_last": [False, True, False, False, True]})
    pairs = pd.DataFrame({"gene_id": ["G", "G", "G", "X"], "chrom": "chr1", "strand": "+",
                          "proximal": [2500, 1100, 2500, 10], "distal": [2900, 2500, 7900, 20]})
    se = same_exon(pairs, ref, downstream_nt=5000)
    assert se.dtype == "boolean" and list(se.iloc[:3]) == [True, False, True] and pd.isna(se.iloc[3])
    mods = models([("short", "G", "+", [(1000, 1200), (2000, 2500)]),
                   ("long", "G", "+", [(1000, 1200), (2000, 2300), (3800, 4000)])])
    p = pd.DataFrame({"gene_id": ["G"], "proximal": [2500], "distal": [4000]})
    assert split_by_model_intron(p, mods, terminal_exons(mods)).iat[0] == 1.0


def test_matching_pairs_across_units():
    t = pd.DataFrame({"gene_id": "G", "unit": list("ABCD"), "proximal_site": [1000, 1012, 995, 3000],
                      "distal_site": [2000, 2030, 1990, 4000], "reads_control": 50, "reads_treated": 50})
    m = match_pairs(t, tol=50)
    assert list(m.same_pair) == [True, True, True, False]
    assert m.reference_proximal.iat[0] in (1000, 1012, 995)


def test_reference_pair_is_chosen_among_valid_pairs():
    t = pd.DataFrame({"gene_id": "G", "unit": list("ABCDE"), "proximal_site": [1000, 1005, 995, 3000, 3010],
                      "distal_site": [2000, 2010, 1990, 4000, 4005], "reads_control": 50, "reads_treated": 50,
                      "is_tandem": [False, False, False, True, True]})
    m = match_pairs(t, tol=50, valid="is_tandem")
    assert m.reference_proximal.iat[0] in (3000, 3010) and list(m.same_pair) == [False, False, False, True, True]


def test_distance_from_stop_spliced_and_extended():
    can = canonical("G", "+", [(1000, 1200), (2000, 2200), (3000, 4000)])
    stop = pd.DataFrame({"gene_id": ["G"], "transcript_id": ["G-T"], "chrom": "chr1", "strand": "+", "start": [2150],
                         "end": [2152]})
    d = distance_from_stop(["G"] * 4, [2153, 3000, 4500, 2500], can, stop, downstream_nt=5000)
    assert d[0] == 0 and d[1] == 48 and d[2] == 48 + 1500 and np.isnan(d[3])

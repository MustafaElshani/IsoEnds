"""End to end on synthetic experiments with known tail and site shifts, and the command line."""
import json

import numpy as np
import pandas as pd
import pytest

from isoends import pipeline
from isoends.cli import main
from isoends.config import load_config, read_sample_sheet
from isoends.hist import grouped_quantiles
from isoends.synthetic import make_experiment
from isoends.tailsource import TailIndex, uuid_pairs

TANDEM_GENES = {"TANDEM1", "TANDEM2", "LNC1", "NOVEL1"}
LABELLED_3UTR = TANDEM_GENES | {"SPLIT1"}       # every gene with two sites IsoForge labels three_prime_utr


def test_uuid_index_is_exact():
    import uuid
    rng = np.random.default_rng(0)
    ids = [str(uuid.UUID(int=int(rng.integers(0, 2 ** 63)) << 64 | int(rng.integers(0, 2 ** 63)))) for _ in range(5000)]
    hi, lo, ok = uuid_pairs(["12:34|" + i for i in ids])
    assert ok.all()
    u = uuid.UUID(ids[0]).int
    assert int(hi[0]) == u >> 64 and int(lo[0]) == u & ((1 << 64) - 1)
    tail = np.arange(5000, dtype=np.int16)
    idx = TailIndex.from_arrays(hi, lo, tail, np.zeros(5000, np.int16))
    q_hi, q_lo, _ = uuid_pairs(ids[::-1] + [str(uuid.uuid4())])
    t, _ = idx.lookup(q_hi, q_lo)
    assert (t[:-1] == tail[::-1]).all() and t[-1] == -1
    hi2 = np.array([7, 7, 7], np.uint64)
    lo2 = np.array([3, 1, 2], np.uint64)
    idx2 = TailIndex.from_arrays(hi2, lo2, np.array([30, 10, 20], np.int16), np.zeros(3, np.int16))
    assert list(idx2.lookup(hi2, lo2)[0]) == [30, 10, 20]


def test_quantiles_from_counts_equal_numpy():
    rng = np.random.default_rng(2)
    vals = {g: rng.integers(0, 300, rng.integers(1, 60)) for g in range(200)}
    rows = [(g, v, c) for g, x in vals.items() for v, c in zip(*np.unique(x, return_counts=True))]
    d = pd.DataFrame(rows, columns=["g", "v", "c"])
    q = grouped_quantiles(d.g, d.v, d.c)
    for g, x in vals.items():
        assert q.loc[g, "q50"] == np.median(x)
        assert q.loc[g, "q25"] == pytest.approx(np.quantile(x, 0.25))
        assert q.loc[g, "n"] == len(x)


def test_tails_come_from_the_read_annotation(synthetic_run):
    qc = synthetic_run["ctx"].qc
    assert set(qc.tail_source) == {"polya_length"}
    assert (qc.n_tail_call < qc.n_reads).all() and (qc.n_tail_used == qc.n_tail_call).all()


def test_known_tail_shifts_are_found(synthetic_run):
    t = synthetic_run["tails"]
    s = t["tails_sample_within_unit"]
    assert (s.delta_tail_nt < -10).all() and (s.q < 0.05).all() and set(s.design) == {"paired"}
    a = t["tails_sample_across_units"].iloc[0]
    assert a.ci95_low_nt < -15 + 3 and a.ci95_high_nt > -15 - 3 and a.n_units_shorter == 6
    g = t["tails_gene_across_units"].set_index("gene_name")
    bg = g[g.index.str.startswith("GENE")]
    assert (bg.q < 0.05).mean() > 0.9 and bg.mean_delta_tail_nt.median() == pytest.approx(-15, abs=2)
    assert g.loc["MT-CO1", "mean_delta_tail_nt"] > 5 and g.loc["MT-CO1", "q"] < 0.05
    thr = t["tails_thresholds_by_unit"]
    below = thr[thr.measure == "share_below_27nt"]
    assert (below.treated > below.control).all()
    w = t["tails_within_gene_isoforms"]
    nmd = w[w.kind == "NMD"]
    assert (nmd.longer_by_nt > 10).all() and (nmd.delta_tail_nt < nmd.coding_delta_tail_nt).mean() > 0.8
    assert {"protein coding", "NMD", "retained intron"} <= set(t["tails_categories"].category)


def test_replicates_are_the_units(synthetic_run):
    w = synthetic_run["tails"]["tails_gene_within_unit"]
    assert w.n_pairs.max() <= 3 and (w.df_residual <= 2).all()
    assert "" not in set(w.gene_id)


def test_transcripts_across_units_pool_one_intron_chain(synthetic_run):
    t = synthetic_run["tails"]
    a = t["tails_transcript_across_units"]
    assert a.structure_key.is_unique and len(a) > 60
    multi = a[a.transcript_ids.str.contains(",")]
    assert len(multi) == 0 or multi.gene_id.notna().all()


def test_known_site_shifts_are_found(synthetic_run):
    a = synthetic_run["apa"]
    t = a["apa_tandem_within_unit"]
    assert t.is_tandem.all() and set(t.gene_name) == LABELLED_3UTR
    tt = t[t.gene_name.isin(TANDEM_GENES)]
    assert (tt[tt.gene_name == "TANDEM1"].delta_pdui > 0.25).all()
    assert (tt[tt.gene_name == "TANDEM2"].delta_pdui < -0.25).all()
    assert (tt.q < 1e-5).all() and tt.same_pair.all() and set(tt.design) == {"paired"}
    split = t[t.gene_name == "SPLIT1"]
    assert len(split) and split.q.notna().all() and (split.share_distal_models_split > 0.5).all()
    assert (split.exclusion_reason == "").all() and t.same_exon.dtype == "boolean"
    across = a["apa_tandem_across_units"].set_index("gene_name")
    assert across.loc["TANDEM1", "n_units"] == 6 and across.loc["TANDEM1", "q"] < 0.01
    c = a["apa_class_across_units"]
    got = {(r.gene_name, r.apa_class, np.sign(r.mean_delta_share)) for r in c[c.q < 0.05].itertuples()}
    assert got == {("CE1", "composite exon", 1), ("SE1", "skipped exon", -1), ("IE1", "internal exonic", 1),
                   ("UTR5G", "composite exon", 1), ("UTR5G", "5'UTR", 1)}
    ipa = a["apa_ipa_across_units"]
    assert set(ipa[ipa.q < 0.05].gene_name) == {"CE1", "SE1", "NM1", "UTR5G"}
    sites = a["sites"]
    nm = sites[(sites.gene_name == "NM1") & (sites.site_class != "terminal")]
    assert set(nm.site_class) == {"intronic, no model"}


def test_five_prime_utr_and_noncoding_apa(synthetic_run):
    a = synthetic_run["apa"]
    sites = a["sites"]
    u5 = sites[(sites.gene_name == "UTR5G") & (sites.site_class == "composite exon")]
    assert set(u5.region) == {"5'UTR"}
    across = a["apa_tandem_across_units"].set_index("gene_name")
    assert not across.loc["LNC1", "coding"] and across.loc["LNC1", "gene_type"] == "lncRNA"
    assert across.loc["NOVEL1", "gene_type"] == "novel lncRNA" and not across.loc["NOVEL1", "coding"]
    assert across.loc["LNC1", "q_noncoding"] < 0.01 and across.loc["LNC1", "mean_delta_pdui"] > 0.2
    assert across.loc["NOVEL1", "q"] < 0.01 and across.loc["NOVEL1", "mean_delta_pdui"] > 0.2
    nov = sites[sites.gene_name == "NOVEL1"]
    assert set(nov.site_class) == {"no canonical transcript"} and set(nov.feature) == {"three_prime_utr"}


def test_stricter_site_options(strict_run, synthetic_run):
    a = strict_run["apa"]
    t = a["apa_tandem_within_unit"]
    tt = t[t.is_tandem]
    assert set(tt.gene_name) == TANDEM_GENES and tt.same_pair.all() and (tt.q < 1e-5).all()
    split = t[t.gene_name == "SPLIT1"]
    assert len(split) and (~split.is_tandem).all() and split.q.isna().all() and split.q_noncoding.isna().all()
    assert set(split.exclusion_reason) == {"the distal-site models splice over the proximal site"}
    nov = a["sites"][a["sites"].gene_name == "NOVEL1"]
    assert set(nov.canonical_source) == {"IsoForge model"} and set(nov.site_class) == {"terminal"}
    across = a["apa_tandem_across_units"].set_index("gene_name")
    assert "SPLIT1" not in across.index and across.loc["NOVEL1", "q"] < 0.01
    loose = synthetic_run["apa"]["apa_tandem_across_units"].set_index("gene_name")
    assert "SPLIT1" in loose.index
    assert across.loc["TANDEM1", "mean_delta_pdui"] == pytest.approx(loose.loc["TANDEM1", "mean_delta_pdui"])
    with pytest.raises(ValueError, match="tandem_sites"):
        pipeline.run_apa(load_config(strict_run["cfg"].source, overrides={
            "outdir": str(strict_run["cfg"].outdir), "cache_dir": str(strict_run["cfg"].cache_dir),
            "params": {"apa": {"tandem_sites": "other"}}}), ctx=None)


def test_alternative_tss(synthetic_run):
    t = synthetic_run["tss"]
    cls = t["tss_sites"].groupby("gene_name").site_class.agg(set)
    assert cls["TSS1"] == {"canonical first exon", "upstream TSS"}
    assert cls["TSS2"] == {"canonical first exon", "alternative first exon"}
    assert cls["GENE001"] == {"canonical first exon"}
    sw = t["tss_switch_across_units"].set_index("gene_name")
    assert sw.loc["TSS1", "mean_delta_dtui"] < -0.25 and sw.loc["TSS1", "q"] < 0.01
    assert abs(sw.loc["TSS2", "mean_delta_dtui"]) > 0.2
    c = t["tss_class_across_units"]
    got = {(r.gene_name, r.tss_class, np.sign(r.mean_delta_share)) for r in c[c.q < 0.05].itertuples()}
    assert got == {("TSS1", "upstream TSS", 1), ("TSS2", "alternative first exon", -1)}
    assert c.q_noncoding.isna().all() or "q_noncoding" in c


def test_unpaired_design_finds_the_same_effects(unpaired_run, synthetic_run):
    t, a, s = unpaired_run["tails"], unpaired_run["apa"], unpaired_run["tss"]
    w = t["tails_gene_within_unit"]
    assert set(w.design) == {"unpaired"} and "n_pairs" not in w
    assert (w.n_control == 3).all() and (w.df_residual == 4).all()
    paired = synthetic_run["tails"]["tails_gene_within_unit"]
    both = w.merge(paired, on=["unit", "gene_id"], suffixes=("", "_paired"))
    assert np.allclose(both.delta_tail_nt, both.delta_tail_nt_paired)
    g = t["tails_gene_across_units"].set_index("gene_name")
    assert g.loc["MT-CO1", "mean_delta_tail_nt"] > 5 and g.loc["MT-CO1", "q"] < 0.05
    assert (t["tails_sample_within_unit"].p < 0.05).all()
    tw = a["apa_tandem_within_unit"]
    tt = tw[tw.gene_name.isin(TANDEM_GENES)]
    assert set(tw.gene_name) == LABELLED_3UTR and tw.is_tandem.all() and set(tt.design) == {"unpaired"}
    assert (tt.n_samples_control == 3).all() and "chi2" not in tt and (tt.q < 0.05).mean() > 0.9
    across = a["apa_tandem_across_units"].set_index("gene_name")
    assert across.loc["TANDEM1", "q"] < 0.01 and across.loc["TANDEM2", "mean_delta_pdui"] < -0.25
    c = a["apa_class_across_units"]
    assert {("CE1", "composite exon"), ("SE1", "skipped exon")} <= {(r.gene_name, r.apa_class)
                                                                    for r in c[c.q < 0.05].itertuples()}
    sw = s["tss_switch_across_units"].set_index("gene_name")
    assert sw.loc["TSS1", "q"] < 0.01
    ts = pd.read_parquet(unpaired_run["cfg"].outdir / "tables" / "tests_summary_apa.parquet")
    assert ts.test.str.contains("two-sample").any() and not ts.test.str.contains("Haenszel").any()


def test_outputs_and_report(synthetic_run):
    cfg = synthetic_run["cfg"]
    cfg.figures.formats, cfg.figures.dpi, cfg.figures.isotracks = ("png",), 100, False
    report = pipeline.run_report(cfg, hash_inputs=True)
    out = cfg.outdir
    for name in ("tails_sample", "tails_gene_within_unit", "tails_gene_across_units", "tails_categories",
                 "tails_within_gene_isoforms", "tails_thresholds", "sites", "apa_tandem_within_unit",
                 "apa_tandem_across_units", "apa_class_within_unit", "apa_class_across_units",
                 "apa_ipa_within_unit", "apa_ipa_across_units", "tss_switch_across_units", "tests_summary",
                 "validation"):
        assert (out / "tables" / f"{name}.tsv").exists(), name
        assert (out / "tables" / f"{name}.parquet").exists(), name
    prov = json.loads((out / "provenance.json").read_text())
    assert all("sha256" in v for v in prov["inputs"].values())
    assert prov["seed"] == 1 and "numpy" in prov["versions"]
    text = report.read_text()
    assert "## Tests" in text and "## Statistics" in text and chr(8212) not in text
    figs = sorted(p.name for p in (out / "figures").glob("*.png"))
    for name in ("fig01_tail_violins.png", "fig02_gene_volcano.png", "fig07_apa_volcanoes.png",
                 "fig10a_tandem_volcano_by_unit.png", "fig12_qc.png", "fig13_tss_volcanoes.png"):
        assert name in figs, name
    assert (out / "figures" / "fig07_apa_volcanoes.caption.md").exists()
    ts = pd.read_parquet(out / "tables" / "tests_summary.parquet")
    assert {"family", "test", "unit_of_replication", "n_tested", "n_up", "n_down"} <= set(ts.columns)


def test_unpaired_report_draws(unpaired_run):
    cfg = unpaired_run["cfg"]
    cfg.figures.formats, cfg.figures.dpi, cfg.figures.isotracks = ("png",), 80, False
    text = pipeline.run_report(cfg, hash_inputs=False).read_text()
    assert "unpaired replicates" in text
    figs = {p.stem for p in (cfg.outdir / "figures").glob("*.png")}
    assert {"fig01_tail_violins", "fig02_gene_volcano", "fig07_apa_volcanoes", "fig13_tss_volcanoes"} <= figs


def test_cli_validate_and_init(synthetic_run, tmp_path, capsys):
    cfg = synthetic_run["cfg"]
    assert main(["validate", "--config", str(cfg.source)]) == 0
    assert "36 samples, 6 units" in capsys.readouterr().out
    assert main(["init", "--out", str(tmp_path / "c.yaml")]) == 0
    assert (tmp_path / "c.yaml").read_text().startswith("# IsoEnds configuration")


def test_single_unit_run(tmp_path):
    cfg = load_config(make_experiment(tmp_path, n_units=1, n_replicates=3, seed=5, n_background=40))
    ctx = pipeline.build_context(cfg)
    t = pipeline.run_tails(cfg, ctx=ctx)
    a = pipeline.run_apa(cfg, ctx=ctx)
    assert len(t["tails_gene_within_unit"]) > 40 and t["tails_gene_across_units"].empty
    assert set(a["apa_tandem_within_unit"].query("is_tandem").gene_name) == LABELLED_3UTR
    ts = pd.read_parquet(cfg.outdir / "tables" / "tests_summary_tails.parquet")
    assert (ts[ts.family == "tails: genes across units"].n_tested == 0).all()
    cfg.figures.formats, cfg.figures.dpi, cfg.figures.isotracks = ("png",), 80, False
    text = pipeline.run_report(cfg, hash_inputs=False).read_text()
    figs = {p.stem for p in (cfg.outdir / "figures").glob("*.png")}
    assert {"fig01_tail_violins", "fig02_gene_volcano", "fig07_apa_volcanoes", "fig12_qc"} <= figs
    assert not any(f.startswith("fig10") or f.startswith("fig06") for f in figs)
    assert "One unit: every test is over its replicates" in text


def test_tail_table_overrides_the_read_annotation(tmp_path):
    cfg = load_config(make_experiment(tmp_path, n_units=2, n_replicates=3, seed=4, n_background=30,
                                      tail_table=True))
    ctx = pipeline.build_context(cfg)
    assert set(ctx.qc.tail_source) == {"tail table or BAMs"} and (ctx.qc.n_mapq_known > 0).all()
    t = pipeline.run_tails(cfg, ctx=ctx)
    assert len(t["tails_gene_within_unit"]) > 30
    assert "delta_tail_nt_mapq" in t["tails_gene_within_unit"]
    assert "q_mapq" in t["tails_gene_across_units"]


def test_supported_reads_only(tmp_path):
    import pyarrow.parquet as pq
    path = make_experiment(tmp_path, n_units=1, n_replicates=2, seed=3, n_background=10)
    f = tmp_path / "LINE1" / "LINE1_isoforge_read_annot.parquet"
    d = pq.read_table(f).to_pandas()
    d["cluster_supported"] = np.arange(len(d)) % 4 != 0
    d.loc[d.index[:50], "polya_length"] = 0
    d["annot_transcript_id"] = d.annot_transcript_id.astype(str)
    d.loc[d.index[np.arange(len(d)) % 7 == 1], "annot_transcript_id"] = "ISOFORT999999999"
    d.to_parquet(f, index=False)
    ctx = pipeline.build_context(load_config(path))
    qc = ctx.qc
    supported = d.cluster_supported & (d.annot_transcript_id != "ISOFORT999999999")
    calls = d.polya_length >= 0                 # a length of 0 is a call; -1 is no call
    assert qc.n_cluster_supported.sum() == int(d.cluster_supported.sum()) and qc.n_supported.sum() == int(supported.sum())
    assert qc.n_tail_call.sum() == int(calls.sum()) and qc.n_tail_used.sum() == int((calls & supported).sum())
    assert qc.n_site.sum() == int(supported.sum())
    reads = pd.concat([ctx.caches["LINE1"].load(s, "model_reads") for s in qc.dataset])
    assert reads.n_reads.sum() == len(d) and reads.n_reads_supported.sum() == int(supported.sum())
    tss = pipeline.run_tss(load_config(path), ctx=ctx)["tss_sample_counts"]
    assert tss.reads.sum() == int((supported & (d.chromosome.astype(str) != "chrM")).sum())
    loose = load_config(path, overrides={"outdir": "loose", "params": {"tails": {"require_supported": False}}})
    qc = pipeline.build_context(loose).qc
    assert qc.n_tail_used.sum() == int((d.polya_length >= 0).sum())
    from_one = load_config(path, overrides={"outdir": "from_one", "params": {"tails": {"min_tail_nt": 1}}})
    qc = pipeline.build_context(from_one).qc
    assert qc.n_tail_call.sum() == int((d.polya_length > 0).sum())


def test_a_transcripts_tail_pools_its_reads_over_genes():
    from isoends.tails import transcript_summary
    h = pd.DataFrame({"gene": ["G1", "G1", "G2", "G2", "G1"], "transcript": ["T1", "T1", "T1", "", "T2"],
                      "tail": [10, 30, 50, 70, 90], "n_reads": [2, 1, 2, 5, 4], "n_reads_mapq": [2, 1, 2, 5, 4]})
    s = transcript_summary(h).set_index("transcript")
    assert list(s.index) == ["T1", "T2"]
    assert s.loc["T1", "n"] == 5 and s.loc["T1", "q50"] == 30 and s.loc["T1", "gene"] == "G1"
    assert s.loc["T2", "n"] == 4 and s.loc["T2", "q50"] == 90


def test_sample_sheet_designs(tmp_path):
    path = make_experiment(tmp_path, n_units=2, n_replicates=3, seed=2, n_background=5)
    sheet = pd.read_csv(tmp_path / "samples.tsv", sep="\t", dtype=str)
    runs = ["LINE1", "LINE2"]

    def read(table, control="control", treatment="treated"):
        table.to_csv(tmp_path / "s.tsv", sep="\t", index=False)
        return read_sample_sheet(tmp_path / "s.tsv", control, treatment, runs)

    s, notes = read(sheet)
    assert s.design_names == {"LINE1": "paired", "LINE2": "paired"} and not notes
    assert s.design("LINE1").pairs[0] == ("1", "LINE1_Ctrl_Bio1", "LINE1_Trt_Bio1")
    s, _ = read(sheet.drop(columns="pair"))
    assert s.design_names["LINE1"] == "unpaired" and len(s.design("LINE1").control) == 3
    other = sheet.assign(condition=sheet.condition.where(~sheet.dataset.str.endswith("Trt_Bio3"), "drug_B"))
    s, notes = read(other)
    assert len(s.table) == 10 and len(s.excluded) == 2 and "drug_B" in notes[0]
    assert any("incomplete pair" in n for n in notes)
    s, _ = read(other, treatment="drug_B")
    assert len(s.design("LINE1").pairs) == 1
    with pytest.raises(ValueError, match="no dataset has condition"):
        read(sheet, treatment="absent")
    with pytest.raises(ValueError, match="several runs"):
        read(sheet.assign(run=np.where(sheet.pair == "3", "LINE2", sheet.run)))
    with pytest.raises(ValueError, match="some datasets have a `pair` label"):
        read(sheet.assign(pair=sheet.pair.where(sheet.pair != "3", "")))
    with pytest.raises(ValueError, match="unknown config key"):
        load_config(path, overrides={"preset": "x"})


def test_without_reference_gene_types_come_from_isoforge(tmp_path):
    path = make_experiment(tmp_path, n_units=1, n_replicates=2, seed=6, n_background=10)
    cfg = load_config(path, overrides={"reference": {"gtf": None}})
    ctx = pipeline.build_context(cfg)
    assert ctx.ref is None and len(ctx.gene_class_members["ribosomal_proteins"]) == 3
    assert pipeline.run_apa(cfg, ctx=ctx) == {}
    assert len(pipeline.run_tails(cfg, ctx=ctx)["tails_gene_sample"])


def test_disabled_tss_removes_its_tables(tmp_path):
    path = make_experiment(tmp_path, n_units=2, n_replicates=2, seed=8, n_background=10)
    cfg = load_config(path)
    cfg.figures.enabled = False
    pipeline.run_all(cfg, hash_inputs=False)
    assert (cfg.outdir / "tables" / "tss_sites.parquet").exists()
    cfg.tss.enabled = False
    report = pipeline.run_all(cfg, hash_inputs=False).read_text()
    assert not (cfg.outdir / "tables" / "tss_sites.parquet").exists() and "tss_sites" not in report


def test_mixed_effects_go_both_ways(tmp_path):
    cfg = load_config(make_experiment(tmp_path, n_units=4, n_replicates=3, seed=9, n_background=150,
                                      n_apa_background=120, effects="mixed"))
    ctx = pipeline.build_context(cfg)
    g = pipeline.run_tails(cfg, ctx=ctx)["tails_gene_by_unit"]
    change = g[g.gene_name.str.startswith("GENE")].groupby("gene_name").delta_tail_nt.mean()
    assert (change < -8).sum() > 15 and (change > 8).sum() > 15
    apa = pipeline.run_apa(cfg, ctx=ctx)
    tandem = apa["apa_tandem_within_unit"].groupby("gene_name").delta_pdui.mean()
    assert (tandem < -0.08).sum() > 5 and (tandem > 0.08).sum() > 5
    assert {"composite exon", "skipped exon", "internal exonic", "5'UTR"} <= set(apa["apa_class_within_unit"].apa_class)
    assert (~apa["apa_tandem_within_unit"].coding).any()
    tss = pipeline.run_tss(cfg, ctx=ctx)["tss_class_within_unit"]
    assert {"upstream TSS", "alternative first exon", "internal exonic TSS"} <= set(tss.tss_class)
    with pytest.raises(ValueError, match="effects"):
        make_experiment(tmp_path / "x", effects="other")

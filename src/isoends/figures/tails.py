"""The poly(A) tail figures. Captions are written from the computed values."""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import pubfig as pf
from .style import (CATEGORY_ORDER, CONTROL_DARK, KIND_COLORS, MM_LEFT, TREATED_DARK, UPDOWN, fmt_q, labels_fit,
                    save, square_aspect, symmetric_limits, tail_limits, tail_violins, volcano, width_mm)

MAX_POINTS_VIOLIN = 60_000
UNITS_PER_ROW = 11


def _cond_label(info, which):
    return info["control"] if which == "control" else info["treatment"]


def fig_tail_violins(T, info, outdir, fp):
    """Split violins of per-transcript median tails per unit, or per replicate pair for one paired unit."""
    ts = T("tails_transcript_sample")
    tests = T("tails_sample_within_unit")
    across = T("tails_sample_across_units")
    if ts.empty or tests.empty:
        return None
    samples = pd.DataFrame(info["samples"]).set_index("dataset")
    ts = ts.assign(condition=ts.dataset.map(samples.condition), pair=ts.dataset.map(samples.pair),
                   tail=ts.median_tail_nt)
    units = list(dict.fromkeys(tests.unit))
    single = len(units) == 1
    by_pair = single and info["units"][units[0]]["design"] == "paired"
    if by_pair:
        ts["pos"] = "pair " + ts.pair.astype(str)
        order = [f"pair {p}" for p in info["units"][units[0]]["pairs"]]
    else:
        ts["pos"] = ts.unit
        order = units
    rng = np.random.default_rng(0)
    parts = []
    for (_, _), g in ts.groupby(["pos", "condition"]):
        parts.append(g.iloc[rng.choice(len(g), MAX_POINTS_VIOLIN, replace=False)] if len(g) > MAX_POINTS_VIOLIN
                     else g)
    ts = pd.concat(parts)
    rows = [order[i:i + UNITS_PER_ROW] for i in range(0, len(order), UNITS_PER_ROW)]
    per_row = max(len(r) for r in rows)
    wide = "double" if per_row > 4 else "onehalf"
    fig, axes = pf.subplots(len(rows), 1, width=wide, aspect=(0.4 if wide == "double" else 0.7) * len(rows),
                            squeeze=False)
    limits = tail_limits(ts["tail"])
    hue_order = [info["control"], info["treatment"]]
    slanted = not labels_fit(order, (width_mm(wide) - MM_LEFT) / per_row)
    for ax, sub in zip(axes[:, 0], rows):
        tail_violins(ax, ts[ts.pos.isin(sub)], "pos", sub, "condition", hue_order, limits, legend=ax is axes[0, 0],
                     slanted=slanted)
        ax.set_xlim(-0.6, per_row - 0.4)
        if not single:
            q = tests.set_index("unit").q
            for i, u in enumerate(sub):
                ax.text(i, 0.985, fmt_q(q.get(u, np.nan)), ha="center", va="top", color=pf.INK["muted"],
                        transform=ax.get_xaxis_transform(), fontsize=pf.sizes()["annot"] - 1)
    t = tests.iloc[0]
    if single:
        pf.fig_stat_title(fig, f"{units[0]}: change {t.delta_tail_nt:+.1f} nt, {pf.fmt_p(t.p)}", loc="right")
        cap = (f"Poly(A) tails of {units[0]}. Each half-violin is the distribution of per-transcript median tails "
               f"(transcripts with ≥ {info['params']['tails']['min_reads_transcript_sample']} reads in the sample; "
               f"axis limited to their 1st to 99th percentile) with {info['control']} (left) and "
               f"{info['treatment']} (right)"
               f"{', one replicate pair per position' if by_pair else ', the samples of a condition pooled'}; bars "
               f"mark the median and quartiles. The sample tail (median of the per-transcript medians) changed by "
               f"{t.delta_tail_nt:+.1f} nt over {int(t.n_control)} {info['control']} and {int(t.n_treated)} "
               f"{info['treatment']} samples ({info['tests']['sample']}, {pf.fmt_p(t.p)}).")
    else:
        a = across.iloc[0]
        cap = (f"Poly(A) tails per unit. Each half-violin pools the per-transcript median tails (transcripts with ≥ "
               f"{info['params']['tails']['min_reads_transcript_sample']} reads in a sample) of the "
               f"{info['control']} (left) or {info['treatment']} (right) samples of one unit; bars mark the median "
               f"and quartiles; the axis is limited to the 1st to 99th percentile. In each panel, per unit, q of the "
               f"test of the sample tail "
               f"(median of per-transcript medians; {info['tests']['sample']}), Benjamini-Hochberg over the "
               f"{int(a.n_units)} units. The tail was shorter with {info['treatment']} in {int(a.n_units_shorter)} of "
               f"{int(a.n_units)} units ({int(a.n_units_shorter_q_below_alpha)} at q < 0.05); mean change "
               f"{a.mean_delta_tail_nt:+.1f} nt (percentile bootstrap 95% CI {a.ci95_low_nt:+.1f} to "
               f"{a.ci95_high_nt:+.1f} nt over units, {int(a.bootstrap_resamples):,} resamples).")
    return save(fig, outdir, "fig01_tail_violins", "Poly(A) tail length per unit", cap, **fp,
                sources=[outdir / "tables" / "tails_transcript_sample.parquet"],
                accept=["no x-axis title", "x tick labels are crowded"])


def _gene_table(T):
    across = T("tails_gene_across_units")
    if len(across):
        return across.rename(columns={"mean_delta_tail_nt": "effect"}), "across"
    within = T("tails_gene_within_unit")
    if len(within):
        return within.rename(columns={"delta_tail_nt": "effect"}), "within"
    return pd.DataFrame(), ""


def fig_gene_volcano(T, info, outdir, fp, genes=(), n_label=8):
    """The gene-level volcano: tail change (nt) against -log10 q."""
    g, level = _gene_table(T)
    if g.empty:
        return None
    fig, ax = pf.subplots(1, 1, width="single", aspect=square_aspect(1, 1, "single"))
    up, dn = volcano(ax, g.effect, g.q, g.gene_name.where(g.gene_name != "", g.gene_id), label=genes,
                     n_label=n_label, xlabel="Change in poly(A) tail (nt)", up_text="longer", down_text="shorter")
    unit = g.unit.iat[0] if level == "within" else ""
    if level == "across":
        pf.stat_title(ax, f"{len(g):,} genes, ≥ {info['min_units']} units")
        cap = (f"Gene-level poly(A) tail change across units. For each gene measured in at least {info['min_units']} "
               f"of {len(info['units'])} units (≥ {info['params']['tails']['min_reads_gene']} reads per sample, ≥ "
               f"{info['params']['tails']['min_replicates']} replicates per unit), the change in a unit is the "
               f"change in its median tail ({info['treatment']} − {info['control']}) over the replicates; x is the "
               f"mean of the per-unit changes and y the Benjamini-Hochberg q of the moderated one-sample t-test "
               f"over units (dashed line, q = 0.05). {len(g):,} genes: {dn:,} shorter and {up:,} "
               f"longer at q < 0.05; median change {g.effect.median():+.1f} nt.")
    else:
        pf.stat_title(ax, f"{unit}: {len(g):,} genes")
        cap = (f"Gene-level poly(A) tail change in {unit}. x: the change in each gene's median tail "
               f"({info['treatment']} − {info['control']}; ≥ {info['params']['tails']['min_reads_gene']} reads per "
               f"sample, ≥ {info['params']['tails']['min_replicates']} replicates); y: Benjamini-Hochberg q of the "
               f"{info['tests']['tails']}. {len(g):,} genes: {dn:,} shorter and {up:,} longer at q < 0.05; median change "
               f"{g.effect.median():+.1f} nt.")
    return save(fig, outdir, "fig02_gene_volcano", "Gene-level poly(A) tail change", cap, **fp,
                sources=[outdir / "tables" / f"tails_gene_{level}_unit{'s' if level == 'across' else ''}.parquet"])


def fig_classes_categories(T, info, outdir, fp):
    """The change by gene class (one point per unit) and the tails by transcript category (split violins)."""
    gc = T("tails_gene_classes")
    ts = T("tails_transcript_sample")
    have_cat = "category" in ts and ts.category.isin(CATEGORY_ORDER).any()
    if gc.empty and not have_cat:
        return None
    fig, axd = pf.mosaic([["a", "b"]], width="double", aspect=0.5, gridspec_kw={"width_ratios": [1, 1.25]})
    caps = []
    ax = axd["a"]
    if len(gc):
        gc = gc[gc.n_genes > 0]
        classes = [c for c in gc.gene_class.unique()]
        allg = gc.drop_duplicates("unit")[["unit", "median_delta_tail_all_genes_nt"]].rename(
            columns={"median_delta_tail_all_genes_nt": "median_delta_tail_nt"}).assign(gene_class="all genes")
        d = pd.concat([allg, gc[["unit", "gene_class", "median_delta_tail_nt"]]])
        order = ["all genes"] + classes
        rng = np.random.default_rng(1)
        for i, c in enumerate(order):
            v = d[d.gene_class == c].median_delta_tail_nt.dropna().to_numpy()
            ax.scatter(i + rng.uniform(-0.15, 0.15, len(v)), v, s=9, color=TREATED_DARK if c != "all genes"
                       else pf.INK["muted"], lw=0, alpha=0.8, zorder=2)
            if len(v):
                ax.plot([i - 0.28, i + 0.28], [np.median(v)] * 2, color=pf.INK["text"], lw=1.2, zorder=3)
        ax.axhline(0, color=pf.INK["axis"], lw=0.5, zorder=0)
        pf.categorical_axis(ax, rotate=False)
        ax.set_xticks(range(len(order)), [o.replace("_", " ") for o in order], rotation=30, ha="right",
                      rotation_mode="anchor")
        ax.set_ylabel("Median tail change per unit (nt)")
        ax.set_box_aspect(1)
        tests = T("tails_gene_class_tests").set_index("gene_class") if len(T("tails_gene_class_tests")) else None
        parts = []
        for c in classes:
            n = gc[gc.gene_class == c].n_genes.median()
            if tests is not None and c in tests.index and np.isfinite(tests.at[c, "p"]):
                parts.append(f"{c.replace('_', ' ')} (median {int(n)} genes per unit; class − all genes "
                             f"{tests.at[c, 'median_difference_nt']:+.1f} nt, Wilcoxon signed-rank over "
                             f"{int(tests.at[c, 'n_units'])} units {pf.fmt_p(tests.at[c, 'p'])})")
            else:
                parts.append(f"{c.replace('_', ' ')} (median {int(n)} genes per unit)")
        caps.append("(A) Per unit, the median over a class's genes of the gene's change within the unit; each point "
                    "one unit, bar the median over units: "
                    + "; ".join(parts) + ".")
    else:
        ax.set_axis_off()
    ax = axd["b"]
    if have_cat:
        samples = pd.DataFrame(info["samples"]).set_index("dataset")
        d = ts[ts.category.isin(CATEGORY_ORDER)].assign(condition=lambda x: x.dataset.map(samples.condition))
        d["tail"] = d.median_tail_nt
        d["category"] = d.category.str.replace(" ", "\n")
        cats = [c.replace(" ", "\n") for c in CATEGORY_ORDER if c.replace(" ", "\n") in set(d.category)]
        rng = np.random.default_rng(2)
        d = pd.concat([g.iloc[rng.choice(len(g), min(len(g), MAX_POINTS_VIOLIN), replace=False)]
                       for _, g in d.groupby(["category", "condition"])])
        tail_violins(ax, d, "category", cats, "condition", [info["control"], info["treatment"]])
        ax.set_box_aspect(0.8)
        ct = T("tails_category_tests")
        extra = ""
        if len(ct):
            r = ct[(ct.level == "transcript") & (ct.what == "tail with control") &
                   (ct.comparison == "NMD vs protein coding")]
            if len(r) and np.isfinite(r.p.iat[0]):
                extra = (f" NMD against protein-coding transcripts with {info['control']}: median difference "
                         f"{r.median_difference_nt.iat[0]:+.1f} nt over units, Wilcoxon signed-rank "
                         f"{pf.fmt_p(r.p.iat[0])} (n = {int(r.n_units.iat[0])} units).")
        caps.append(f"(B) Per-transcript median tails by category (annotated models: reference type; novel models: "
                    f"SQANTI3 class where the characterised GTF describes the run), all units and samples pooled for "
                    f"display ({info['control']} left, {info['treatment']} right; ≥ "
                    f"{info['params']['tails']['min_reads_transcript_sample']} reads per transcript and sample)."
                    + extra + " The tests over units are in tails_category_tests.")
    else:
        ax.set_axis_off()
    pf.panel_labels([axd["a"], axd["b"]], journal=None)
    return save(fig, outdir, "fig03_classes_categories", "Tail change by gene class and transcript category",
                " ".join(caps), **fp, accept=["no x-axis title", "x tick labels are crowded"])


def fig_within_gene(T, info, outdir, fp):
    """NMD and retained-intron isoform tails against the coding isoform of the same gene."""
    iso = T("tails_within_gene_isoforms")
    if iso.empty:
        return None
    g = iso.groupby(["gene_id", "gene_name", "kind"]).agg(decay=("tail_control_nt", "median"),
                                                          coding=("coding_tail_control_nt", "median"),
                                                          n_units=("unit", "size")).reset_index()
    fig, ax = pf.subplots(1, 1, width="single", aspect=square_aspect(1, 1, "single"))
    ax.set_box_aspect(1)
    lim = float(np.nanpercentile(np.r_[g.decay, g.coding], 99.5)) * 1.05
    ax.plot([0, lim], [0, lim], color=pf.INK["ref"], lw=0.7, ls=(0, (3, 2)), zorder=0)
    parts = []
    for kind, c in KIND_COLORS.items():
        s = g[g.kind == kind]
        if s.empty:
            continue
        ax.scatter(s.coding, s.decay, s=5, color=c, lw=0, alpha=0.6, rasterized=True, label=kind, zorder=2)
        parts.append(f"{kind}: {len(s):,} genes, tail longer than the coding isoform's in "
                     f"{(s.decay > s.coding).mean():.0%} (median difference {(s.decay - s.coding).median():+.1f} nt)")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel(f"Coding-isoform tail, {info['control']} (nt)")
    ax.set_ylabel(f"Decay-isoform tail, {info['control']} (nt)")
    ax.legend(loc="lower right", frameon=False, markerscale=2)
    tests = T("tails_within_gene_tests")
    tx = ""
    if len(tests):
        r = tests[tests.comparison.str.startswith("decay isoform tail")]
        tx = " Over units: " + "; ".join(
            f"{k}, median over genes {m:+.1f} nt per unit, Wilcoxon signed-rank {pf.fmt_p(p)} (n = {int(n)} units)"
            for k, m, p, n in zip(r.kind, r.median_difference_nt, r.p, r.n_units) if np.isfinite(p)) + "."
    cap = (f"Within-gene isoform comparison. Genes expressing a protein-coding isoform and an NMD or intron-retaining "
           f"isoform (each ≥ {info['params']['tails']['min_reads_isoform']} reads per condition): the read-weighted "
           f"tail of each with {info['control']}, median over the units in which the gene was measured; dashed, "
           f"identity. " + "; ".join(parts) + "." + tx)
    return save(fig, outdir, "fig04_within_gene_isoforms", "NMD and intron-retaining isoforms against the coding "
                "isoform of the same gene", cap, **fp)


def fig_thresholds(T, info, outdir, fp):
    """Per gene set: tail histograms with the short-tail line and the window, and the share of short tails per
    unit."""
    h = T("tails_histograms")
    per = T("tails_thresholds_by_unit")
    out = []
    if h.empty or per.empty:
        return out
    tp = info["params"]["tails"]
    short, (lo, hi) = tp["short_nt"], tp["window_nt"]
    meas = f"share_below_{short:g}nt"
    for gset, hs in h.groupby("gene_set"):
        hs = hs.assign(frac=hs.reads / hs.groupby("dataset").reads.transform("sum"))
        prof = hs.groupby(["condition", "bin_start_nt"]).frac.mean().unstack(0)
        fig, axd = pf.mosaic([["a", "b"]], width="double", aspect=0.44, gridspec_kw={"width_ratios": [1.5, 1]})
        ax = axd["a"]
        ax.axvspan(lo, hi, color=pf.INK["grid"], lw=0, zorder=0)
        ax.axvline(short, color=pf.INK["ref"], lw=0.8, ls=(0, (3, 2)), zorder=1)
        for cond, c in ((info["control"], CONTROL_DARK), (info["treatment"], TREATED_DARK)):
            if cond in prof:
                ax.step(prof.index + tp["hist_bin_nt"] / 2, prof[cond] * 100, where="mid", color=c, lw=1.0,
                        label=cond)
        ax.set_xlim(0, tp["hist_max_nt"])
        ax.set_xlabel("Poly(A) tail (nt)")
        ax.set_ylabel(f"Reads per {tp['hist_bin_nt']}-nt bin (%)")
        ax.legend(frameon=False, loc="upper right")
        top = ax.get_ylim()[1]
        ax.text(short, top, f"{short:g} nt ", ha="right", va="top", fontsize=pf.sizes()["annot"],
                color=pf.INK["muted"])
        ax.text(hi, top, f" {lo:g}–{hi:g} nt", ha="left", va="top", fontsize=pf.sizes()["annot"],
                color=pf.INK["muted"])
        ax = axd["b"]
        p = per[(per.gene_set == gset) & (per.measure == meas)]
        lines = p
        by_pair = len(p) == 1 and info["units"][p.unit.iat[0]]["design"] == "paired"
        if by_pair:
            s_ = T("tails_thresholds")
            s_ = s_[s_.gene_set == gset]
            w_ = s_.pivot_table(index="pair", columns="condition", values=meas)
            lines = pd.DataFrame({"control": w_.get(info["control"]), "treated": w_.get(info["treatment"])}).dropna()
        for _, r in lines.iterrows():
            c = UPDOWN["up"] if r.treated > r.control else UPDOWN["down"]
            ax.plot([0, 1], [r.control * 100, r.treated * 100], color=c, lw=0.7, alpha=0.8, marker="o", ms=2.5)
        ax.set_xticks([0, 1], [info["control"], info["treatment"]])
        ax.set_xlim(-0.3, 1.3)
        pf.categorical_axis(ax)
        ax.set_ylabel(f"Reads with tail < {short:g} nt (%)")
        ax.set_box_aspect(1)
        axd["a"].set_box_aspect(1 / 1.5)
        pf.panel_labels([axd["a"], axd["b"]])
        tests = T("tails_threshold_tests")
        tt = tests[(tests.gene_set == gset) & (tests.measure == meas)] if len(tests) else pd.DataFrame()
        n_units = len(p)
        n_up = int((p.treated > p.control).sum())
        wtxt = (f"; Wilcoxon signed-rank over units {pf.fmt_p(tt.p.iat[0])}" if len(tt) and np.isfinite(tt.p.iat[0])
                else "")
        within = (f" Within units ({info['tests']['sample']}): q < 0.05 in {(p.q_within_unit < 0.05).sum()} of "
                  f"{n_units}.")
        cap = (f"Poly(A) tails of the {gset.replace('_', ' ')} ({int(hs.groupby('dataset').reads.sum().median()):,} "
               f"reads per sample, median). (A) Tail histograms ({tp['hist_bin_nt']}-nt bins, tails above "
               f"{tp['hist_max_nt']} nt in the last bin), each sample normalised to its reads and the samples of a "
               f"condition averaged over all units; dashed, {short:g} nt; grey band, {lo:g}–{hi:g} nt. "
               f"(B) Share of reads below {short:g} nt "
               f"per unit (replicate samples averaged): {p.control.median() * 100:.1f}% with {info['control']} and "
               f"{p.treated.median() * 100:.1f}% with {info['treatment']} (medians over units); up in {n_up} of "
               f"{n_units} units{wtxt}.{within if n_units else ''}"
               + (" Each line is one replicate pair." if by_pair else ""))
        out.append(save(fig, outdir, f"fig05_thresholds_{gset}", f"Short tails of the {gset.replace('_', ' ')}",
                        cap, **fp))
    return out


def fig_unit_volcanoes(T, info, outdir, fp, genes=()):
    """The within-unit gene volcano of every unit, on one scale."""
    w = T("tails_gene_within_unit")
    if w.empty or w.unit.nunique() < 2:
        return None
    units = list(dict.fromkeys(w.unit))
    cols = min(5, len(units))
    rows = int(np.ceil(len(units) / cols))
    fig, axes = pf.subplots(rows, cols, width="double", aspect=square_aspect(rows, cols, shared=True), squeeze=False,
                            sharex=True, sharey=True)
    lim = symmetric_limits(w.delta_tail_nt)[1]
    ymax = float(np.nanmax(-np.log10(w.q.clip(lower=1e-300)))) * 1.1
    counts = []
    for k, ax in enumerate(axes.flat):
        if k >= len(units):
            ax.set_axis_off()
            continue
        u = units[k]
        d = w[w.unit == u]
        up, dn = volcano(ax, d.delta_tail_nt, d.q, None, xlim=(-lim, lim), ylim=(0, max(ymax, 1.5)), point_size=2,
                         corner_counts=False, ticks=3)
        counts.append((u, up, dn, len(d)))
        ax.set_title(f"{u}  ({dn:,}↓ {up:,}↑)", fontsize=pf.sizes()["tick"], pad=2)
        ax.set_xlabel("Tail change (nt)" if k // cols == rows - 1 else "")
        ax.set_ylabel("$-$log$_{10}$ $q$" if k % cols == 0 else "")
    fig._accepted_audit = ["no x-axis title", "no y-axis title", "x tick labels are crowded"]
    med_short = int(np.median([c[2] for c in counts]))
    cap = (f"The gene-level test within each unit. Per unit, each gene's change in median tail (x) against the "
           f"Benjamini-Hochberg q (within the unit) of the {info['tests']['tails']}; same scales in every panel; in "
           f"brackets, genes shorter (↓) and longer (↑) at q < 0.05 (median over units {med_short:,} shorter).")
    return save(fig, outdir, "fig06_gene_volcano_by_unit", "Gene-level tail change within each unit", cap, **fp)

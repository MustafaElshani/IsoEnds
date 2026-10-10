"""The APA, TSS and QC figures."""
from __future__ import annotations

import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

from . import pubfig as pf
from .style import (CONTROL_DARK, INK, TREATED_DARK, UPDOWN, class_color, save, square_aspect, symmetric_limits,
                    volcano)

SCHEMATIC = 1.3 / 4     # height of the isoform diagram under a volcano, as a share of the volcano
UNTESTED = ("upstream", "far downstream", "no canonical transcript", "opposite strand")

CLASSES = ("composite exon", "skipped exon", "internal exonic")
YCAP = 50          # cap of -log10 q in within-unit volcanoes
N_TRACKED = 10     # genes most significant across units that are ringed in the per-unit grids


def _tandem(T):
    a = T("apa_tandem_across_units")
    if len(a):
        return a.rename(columns={"mean_delta_pdui": "effect"}), "across"
    w = T("apa_tandem_within_unit")
    if len(w):
        w = w[w.is_tandem]
        return w.rename(columns={"delta_pdui": "effect"}), "within"
    return pd.DataFrame(), ""


def _classes(T, label):
    a = T("apa_class_across_units")
    if len(a):
        return a[a.apa_class == label].rename(columns={"mean_delta_share": "effect"}), "across"
    w = T("apa_class_within_unit")
    if len(w):
        return w[w.apa_class == label].rename(columns={"delta_share": "effect"}), "within"
    return pd.DataFrame(), ""


EXON = "#B8BDC4"
ISOFORM_KEY = ("Under each panel, the two isoforms compared, coloured as the points: orange points are genes that "
               "move towards the orange isoform, teal points towards the teal one.")
_A1, _A2, _A3 = (3.4, 4.2), (4.9, 5.7), (7.3, 8.7)
_T1, _T2, _T3 = (1.8, 2.6), (4.0, 4.8), (5.6, 6.6)
_FULL = [(*_A1, False), (*_A2, False), (*_A3, True)]
_CANONICAL = [(*_T1, True), (*_T2, False), (*_T3, False)]
# kind -> (label and exons of the isoform that gains when the effect is negative, then of the one that gains when it
# is positive); an exon is (x0, x1, coloured) on an axis running 0 to 10 from 5' to 3'
ISOFORMS = {
    "tandem 3'UTR": ("proximal", [(*_A1, False), (*_A2, False), (7.3, 7.9, True)],
                     "distal", [(*_A1, False), (*_A2, False), (7.3, 8.9, True)]),
    "composite exon": ("full length", _FULL, "read-through", [(*_A1, False), (*_A2, False), (5.7, 6.4, True)]),
    "skipped exon": ("full length", _FULL, "alternative\nlast exon", [(*_A1, False), (*_A2, False), (6.2, 6.8, True)]),
    "internal exonic": ("full length", _FULL, "ends in an\ninternal exon", [(*_A1, False), (4.9, 5.35, True)]),
    "5'UTR": ("full length", _FULL, "ends before\nthe start codon", [(*_A1, False), (4.2, 4.6, True)]),
    "TSS switch": ("upstream TSS", _CANONICAL, "downstream\nTSS", [(3.0, 3.5, True), (*_T2, False), (*_T3, False)]),
    "upstream TSS": ("canonical", _CANONICAL, "upstream\npromoter", [(0.4, 1.0, True), (*_T2, False), (*_T3, False)]),
    "alternative first exon": ("canonical", _CANONICAL, "alternative\nfirst exon",
                               [(3.0, 3.5, True), (*_T2, False), (*_T3, False)]),
    "internal exonic TSS": ("canonical", _CANONICAL, "starts in an\ninternal exon", [(4.4, 4.8, True), (*_T3, False)]),
}


def _schematic(ax, kind: str):
    """The two isoforms a class compares, drawn 5' to 3' and coloured as the volcano above them: teal for the
    isoform that gains when the effect is negative, orange for the one that gains when it is positive. Shared exons
    are grey. Poly(A) classes end in a tail; start-site classes begin with a bent arrow."""
    ax.set_axis_off()
    ax.set_xlim(0, 10)
    ax.set_ylim(-1.2, 1.5)
    if kind not in ISOFORMS:
        return
    down_label, down_exons, up_label, up_exons = ISOFORMS[kind]
    starts = "TSS" in kind or "first exon" in kind
    size = pf.sizes()["annot"] - 1
    for y, color, label, exons in ((0.62, UPDOWN["down"], down_label, down_exons),
                                   (-0.62, UPDOWN["up"], up_label, up_exons)):
        ax.plot([exons[0][0], exons[-1][1]], [y, y], color=EXON, lw=0.8, zorder=1)
        for x0, x1, coloured in exons:
            ax.add_patch(Rectangle((x0, y - 0.26), x1 - x0, 0.52, color=color if coloured else EXON, lw=0, zorder=2))
        if starts:
            x0 = exons[0][0]
            ax.plot([x0, x0], [y + 0.26, y + 0.58], color=color, lw=0.8, solid_capstyle="butt")
            ax.annotate("", xy=(x0 + 0.6, y + 0.58), xytext=(x0, y + 0.58),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=0.8, shrinkA=0, shrinkB=0, mutation_scale=5))
            ax.text(6.9, y, label, ha="left", va="center", fontsize=size, color=color, linespacing=0.95)
        else:
            ax.text(exons[-1][1] + 0.12, y, "AAA", ha="left", va="center", fontsize=size, color=color)
            ax.text(3.15, y, label, ha="right", va="center", fontsize=size, color=color, linespacing=0.95)
    if kind == "5'UTR":
        ax.plot([5.15, 5.15], [0.29, 0.95], color=INK, lw=0.9, zorder=3)
        ax.text(5.15, 1.0, "ATG", ha="center", va="bottom", fontsize=size, color=pf.INK["muted"])


def fig_apa_volcanoes(T, info, outdir, fp, genes=(), n_label=6):
    """The tandem 3'UTR volcano and one volcano per class, each with the two isoforms it compares under it."""
    t, level = _tandem(T)
    panels = [("tandem 3'UTR", t, "ΔPDUI (distal usage)", "distal", "proximal")]
    for c in CLASSES:
        d, _ = _classes(T, c)
        panels.append((c, d, f"Δ share of {c} sites", "up", "down"))
    if all(p[1].empty for p in panels):
        return None
    layout = [["v0", "v1"], ["s0", "s1"], ["v2", "v3"], ["s2", "s3"]]
    fig, axd = pf.mosaic(layout, width="double", aspect=square_aspect(2, 2, under=SCHEMATIC),
                         gridspec_kw={"height_ratios": [1, SCHEMATIC, 1, SCHEMATIC]})
    caps = []
    for k, (kind, d, xl, up_t, dn_t) in enumerate(panels):
        ax = axd[f"v{k}"]
        if d.empty:
            ax.set_axis_off()
            axd[f"s{k}"].set_axis_off()
            continue
        names = d.gene_name.where(d.gene_name.astype(str) != "", d.gene_id)
        up, dn = volcano(ax, d.effect, d.q, names, label=genes, n_label=n_label, xlabel=xl, up_text=up_t,
                         down_text=dn_t, ycap=YCAP if level == "within" else None)
        ax.set_title(kind, loc="left", fontsize=pf.sizes()["label"], color=INK, pad=3)
        _schematic(axd[f"s{k}"], kind)
        caps.append((kind, len(d), up, dn))
    pf.panel_labels([axd[f"v{k}"] for k in range(4)])
    fig._accepted_audit = ["no x-axis title", "no y-axis title", "x tick labels are crowded"]
    tl = "across units" if level == "across" else f"in {t.unit.iat[0] if len(t) else ''}"
    capnote = (f" −log10 q above {YCAP:g} is drawn at {YCAP:g} (triangles)." if level == "within" else "")
    parts = [f"{k}: {n:,} genes, {u:,} up and {d:,} down" for k, n, u, d in caps]
    test = ("the moderated one-sample t-test of the per-unit change over units, Benjamini-Hochberg over genes; "
            f"genes measured in ≥ {info['min_units']} units" if level == "across" else
            f"the {info['tests']['usage']}, Benjamini-Hochberg over the unit's genes")
    same = info["params"]["apa"]
    gained = ""
    if len(t) and "length_gained_nt" in t:
        sig = t[t.q < 0.05]
        if len(sig):
            gained = (f" Median 3′UTR length gained over significant tandem genes (ΔPDUI × distance between the "
                      f"sites): {sig.length_gained_nt.median():+.0f} nt.")
    where = (f"in the canonical last exon (or ≤ {same['downstream_nt']:,} nt past it)"
             if same.get("tandem_sites") == "terminal" else "that IsoForge labels three_prime_utr")
    held = ", both in one annotated exon" if same.get("same_exon_check") else ""
    cap = (f"Alternative polyadenylation {tl}. (A) Tandem 3′UTR sites: per gene the two most used sites {where}"
           f"{held}; x, the "
           f"change in distal usage PDUI = distal / (proximal + distal) reads ({info['treatment']} − "
           f"{info['control']}). (B–D) Composite-exon (intron read-through), skipped-exon (alternative last exon) and "
           f"internal exonic sites: x, the change in their share of the gene's reads against terminal-site reads. "
           f"{ISOFORM_KEY} y: q of {test} (dashed, q = 0.05). "
           + "; ".join(parts) + "." + gained + capnote)
    return save(fig, outdir, "fig07_apa_volcanoes", "Tandem 3′UTR and intronic / internal polyadenylation", cap,
                **fp)


def fig_pdui_span(T, info, outdir, fp, genes=()):
    """ΔPDUI against the distance between the two sites, with curves of constant 100 and 500 nt gained."""
    t, level = _tandem(T)
    if t.empty:
        return None
    span = t.median_utr_span_nt if level == "across" else t.utr_span_nt
    fig, ax = pf.subplots(1, 1, width="single", aspect=square_aspect(1, 1, "single"))
    ax.set_box_aspect(1)
    sig = t.q < 0.05
    ax.scatter(span[~sig], t.effect[~sig], s=3, color=UPDOWN["ns"], lw=0, rasterized=True, zorder=1)
    for sel, c in ((sig & (t.effect > 0), UPDOWN["up"]), (sig & (t.effect < 0), UPDOWN["down"])):
        ax.scatter(span[sel], t.effect[sel], s=5, color=c, lw=0, rasterized=True, zorder=2)
    x = np.logspace(np.log10(max(span.min(), 10)), np.log10(span.max() * 1.2), 200)
    for L in (100, 500):
        for sgn in (1, -1):
            y = sgn * L / x
            m = np.abs(y) <= 1
            ax.plot(x[m], y[m], color=pf.INK["ref"], lw=0.7, ls=(0, (3, 2)), zorder=0)
        xm = x[np.argmin(np.abs(L / x - 0.9))]
        ax.text(xm, 0.92 if L == 100 else 0.62, f"{L} nt", fontsize=pf.sizes()["annot"] - 1,
                color=pf.INK["muted"], ha="left")
    ax.axhline(0, color=pf.INK["axis"], lw=0.5)
    ax.set_xscale("log")
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Distance between proximal and distal site (nt)")
    ax.set_ylabel("ΔPDUI")
    lab = t[t.gene_name.isin(genes)]
    if len(lab):
        pf.label_points(ax, (lab.median_utr_span_nt if level == "across" else lab.utr_span_nt).to_numpy(),
                        lab.effect.to_numpy(), lab.gene_name.to_numpy(), drop_crowded=False)
    far = t[span > 10_000]
    cap = (f"Tandem 3′UTR shifts against the distance between the two sites ({len(t):,} genes; coloured, q < 0.05, "
           f"orange distal, blue proximal). Dashed curves: constant 3′UTR length gained or lost, ΔPDUI × distance = "
           f"±100 and ±500 nt. {len(far):,} genes have sites more than 10 kb apart"
           + (" (checked: every tandem pair lies in one annotated exon)." if info["params"]["apa"]["same_exon_check"]
              else "; the same-exon check was off, so some may be split by an intron."))
    return save(fig, outdir, "fig08_pdui_span", "ΔPDUI against the distance between the sites", cap, **fp)


def _stop_axis(ax, t, x_prox, x_dist, sig_only=True, tracked=(), size=6):
    d = t[t.q < 0.05] if sig_only else t
    xp = np.clip(d[x_prox].to_numpy() / 1000, 0.01, None)
    xd = np.clip(d[x_dist].to_numpy() / 1000, 0.01, None)
    y = d.effect.to_numpy()
    ok = np.isfinite(xp) & np.isfinite(xd) & np.isfinite(y)
    for a, b, v in zip(xp[ok], xd[ok], y[ok]):
        ax.plot([a, b], [v, v], color=pf.INK["grid"], lw=0.5, zorder=0)
    ax.scatter(xp[ok], y[ok], s=size * 0.6, color="#9AA0A8", lw=0, zorder=1, rasterized=True)
    col = np.where(y[ok] > 0, UPDOWN["up"], UPDOWN["down"])
    ax.scatter(xd[ok], y[ok], s=size, c=col, lw=0, zorder=2, rasterized=True)
    if len(tracked):
        tr = d[ok & d.gene_id.isin(set(tracked)).to_numpy()]
        ax.scatter(np.clip(tr[x_dist] / 1000, 0.01, None), tr.effect, s=size * 3, facecolor="none",
                   edgecolor=pf.INK["text"], lw=0.5, zorder=3)
    ax.set_xscale("log")
    ax.axhline(0, color=pf.INK["axis"], lw=0.5)
    return int(ok.sum())


def fig_stop_axis(T, info, outdir, fp):
    """The proximal and distal site of each significant tandem gene by their distance from the stop codon."""
    t, level = _tandem(T)
    if t.empty:
        return None
    xp, xd = (("median_proximal_from_stop_nt", "median_distal_from_stop_nt") if level == "across"
              else ("proximal_from_stop_nt", "distal_from_stop_nt"))
    fig, ax = pf.subplots(1, 1, width="onehalf", aspect=0.72)
    n = _stop_axis(ax, t, xp, xd)
    ax.set_xlabel("Distance from the stop codon along the 3′UTR (kb)")
    ax.set_ylabel("ΔPDUI")
    ax.set_ylim(-1, 1)
    sig = t[t.q < 0.05]
    cap = (f"Where the tandem sites lie. For each tandem 3′UTR gene at q < 0.05 with a coding canonical transcript "
           f"({n:,} of {len(sig):,}), its proximal (grey) and distal (coloured: orange distal shift, blue proximal) "
           f"site, placed by their distance from the canonical stop codon along the spliced 3′UTR (kb, log scale; "
           f"the last exon runs on past its annotated end), and joined by a faint line at the gene's ΔPDUI. Median "
           f"distance from the stop codon: proximal {sig[xp].median() / 1000:.2f} kb, distal "
           f"{sig[xd].median() / 1000:.2f} kb.")
    return save(fig, outdir, "fig09_stop_axis", "Tandem sites along the 3′UTR", cap, **fp)


def fig_unit_grids(T, info, outdir, fp):
    """Per-unit tandem volcanoes and stop-codon panels on shared scales, the tracked genes ringed."""
    w = T("apa_tandem_within_unit")
    if w.empty or w.unit.nunique() < 2:
        return []
    w = w[w.is_tandem].rename(columns={"delta_pdui": "effect"})
    a = T("apa_tandem_across_units")
    genes = set(info["params"]["figures"].get("genes", []) or [])
    tracked = (set(a[a.q < 0.05].nsmallest(N_TRACKED, "q").gene_id) | set(a[a.gene_name.isin(genes)].gene_id)
               if len(a) else set())
    units = list(dict.fromkeys(w.unit))
    cols = min(5, len(units))
    rows = int(np.ceil(len(units) / cols))
    out = []
    ymax = min(float(np.nanmax(-np.log10(w.q.clip(lower=1e-300)))), YCAP) * 1.08
    xlim = symmetric_limits(w.effect, cap=1)
    for kind in ("volcano", "stop"):
        fig, axes = pf.subplots(rows, cols, width="double", aspect=square_aspect(rows, cols, shared=True),
                                squeeze=False, sharex=True, sharey=True)
        for k, ax in enumerate(axes.flat):
            if k >= len(units):
                ax.set_axis_off()
                continue
            u = units[k]
            d = w[w.unit == u]
            same = d.same_pair if "same_pair" in d else pd.Series(True, index=d.index)
            if kind == "volcano":
                up, dn = volcano(ax, d.effect, d.q, None, xlim=xlim, ylim=(0, max(ymax, 1.5)), point_size=2,
                                 corner_counts=False, ticks=3, ycap=YCAP)
                tr = d[d.gene_id.isin(tracked) & same]
                ax.scatter(tr.effect, np.minimum(-np.log10(tr.q.clip(lower=1e-300)), YCAP), s=9, facecolor="none",
                           edgecolor=pf.INK["text"], lw=0.5, zorder=3)
                ax.set_title(f"{u}  ({up:,}→ {dn:,}←)", fontsize=pf.sizes()["tick"] - 1, pad=2)
                ax.set_xlabel("ΔPDUI" if k // cols == rows - 1 else "")
                ax.set_ylabel("$-$log$_{10}$ $q$" if k % cols == 0 else "")
            else:
                _stop_axis(ax, d, "proximal_from_stop_nt", "distal_from_stop_nt", tracked=tracked & set(d.gene_id[same]),
                           size=4)
                ax.set_ylim(-1, 1)
                ax.set_box_aspect(1)
                ax.set_title(u, fontsize=pf.sizes()["tick"], pad=2)
                ax.set_xlabel("From stop codon (kb)" if k // cols == rows - 1 else "")
                ax.set_ylabel("ΔPDUI" if k % cols == 0 else "")
        fig._accepted_audit = ["no x-axis title", "no y-axis title", "x tick labels are crowded"]
        tol = info["params"]["apa"]["match_tolerance_nt"]
        if kind == "volcano":
            cap = (f"Tandem 3′UTR APA within each unit: ΔPDUI against the q of the {info['tests']['usage']} "
                   f"(q within the unit), same scales in every panel (−log10 q above "
                   f"{YCAP:g} drawn at {YCAP:g}, triangles); in brackets, genes shifting distal (→) and proximal (←) "
                   f"at q < 0.05. Ringed, where the unit's site pair matches the gene's usual pair within {tol} nt: "
                   f"the {len(tracked):,} genes tracked across units (the {N_TRACKED} most significant across units, "
                   f"and those named in the configuration).")
            name, title = "fig10a_tandem_volcano_by_unit", "Tandem 3′UTR APA within each unit"
        else:
            cap = (f"The sites of each unit's significant tandem genes (q < 0.05 within the unit) placed from the stop "
                   f"codon, as in figure 9: proximal grey, distal coloured by direction, joined at the gene's ΔPDUI; "
                   f"ringed, the tracked genes (as in figure 10a) with a matched site pair (tolerance {tol} nt).")
            name, title = "fig10b_stop_axis_by_unit", "Tandem sites along the 3′UTR in each unit"
        out.append(save(fig, outdir, name, title, cap, **fp))
    return out


def fig_class_unit_grids(T, info, outdir, fp):
    """The class volcanoes (composite exon, skipped exon, internal exonic) within each unit, on one scale."""
    w = T("apa_class_within_unit")
    if w.empty or w.unit.nunique() < 2:
        return []
    across = T("apa_class_across_units")
    units = list(dict.fromkeys(w.unit))
    cols = min(5, len(units))
    rows = int(np.ceil(len(units) / cols))
    out = []
    for k_cls, cls in enumerate(CLASSES):
        d_all = w[w.apa_class == cls]
        if d_all.empty:
            continue
        ac = across[across.apa_class == cls] if len(across) else across
        tracked = set(ac[ac.q < 0.05].nsmallest(N_TRACKED, "q").gene_id) if len(ac) else set()
        ymax = min(float(np.nanmax(-np.log10(d_all.q.clip(lower=1e-300)))), YCAP) * 1.08
        xlim = symmetric_limits(d_all.delta_share, cap=1)
        fig, axes = pf.subplots(rows, cols, width="double", aspect=square_aspect(rows, cols, shared=True),
                                squeeze=False, sharex=True, sharey=True)
        for k, ax in enumerate(axes.flat):
            if k >= len(units):
                ax.set_axis_off()
                continue
            d = d_all[d_all.unit == units[k]]
            up, dn = volcano(ax, d.delta_share, d.q, None, xlim=xlim, ylim=(0, max(ymax, 1.5)), point_size=2,
                             corner_counts=False, ticks=3, ycap=YCAP)
            tr = d[d.gene_id.isin(tracked)]
            ax.scatter(tr.delta_share, np.minimum(-np.log10(tr.q.clip(lower=1e-300)), YCAP), s=9, facecolor="none",
                       edgecolor=pf.INK["text"], lw=0.5, zorder=3)
            ax.set_title(f"{units[k]}  ({up:,}↑ {dn:,}↓)", fontsize=pf.sizes()["tick"] - 1, pad=2)
            ax.set_xlabel("Δ share" if k // cols == rows - 1 else "")
            ax.set_ylabel("$-$log$_{10}$ $q$" if k % cols == 0 else "")
        fig._accepted_audit = ["no x-axis title", "no y-axis title", "x tick labels are crowded"]
        cap = (f"{cls.capitalize()} sites within each unit: the change in their share of the gene's reads against "
               f"terminal-site reads (x) and the q of the {info['tests']['usage']} "
               f"(q within the unit); same scales in every panel (−log10 q above {YCAP:g} drawn at {YCAP:g}, "
               f"triangles); ringed, the {len(tracked):,} genes most significant for this class across units.")
        out.append(save(fig, outdir, f"fig10{'cde'[k_cls]}_{cls.replace(' ', '_')}_by_unit",
                        f"{cls.capitalize()} sites within each unit", cap, **fp))
    return out


def _volcano_panels(T, info, outdir, fp, panels, name, title, cap, genes=(), n_label=6):
    """Up to four volcanoes, two to a row, each with the two isoforms it compares under it. panels: [(kind, table,
    effect, xlabel, up, down)]."""
    panels = [p for p in panels if p[1] is not None and len(p[1])]
    if not panels:
        return None
    rows = (len(panels) + 1) // 2
    layout = [x for r in range(rows) for x in ([f"v{2 * r}", f"v{2 * r + 1}"], [f"s{2 * r}", f"s{2 * r + 1}"])]
    fig, axd = pf.mosaic(layout, width="double", aspect=square_aspect(rows, 2, under=SCHEMATIC),
                         gridspec_kw={"height_ratios": [1, SCHEMATIC] * rows})
    parts = []
    for k in range(2 * rows):
        if k >= len(panels):
            axd[f"v{k}"].set_axis_off()
            axd[f"s{k}"].set_axis_off()
            continue
        kind, d, eff, xl, up_t, dn_t = panels[k]
        ax = axd[f"v{k}"]
        names = d.gene_name.where(d.gene_name.astype(str) != "", d.gene_id)
        within = "unit" in d and "n_units" not in d
        up, dn = volcano(ax, d[eff], d.q if "q_family" not in d else d.q_family, names, label=genes, n_label=n_label,
                         xlabel=xl, up_text=up_t, down_text=dn_t, ycap=YCAP if within else None)
        ax.set_title(kind, loc="left", fontsize=pf.sizes()["label"], color=INK, pad=3)
        _schematic(axd[f"s{k}"], kind)
        parts.append(f"{kind}: {len(d):,} genes, {up:,} {up_t} and {dn:,} {dn_t}")
    pf.panel_labels([axd[f"v{k}"] for k in range(len(panels))])
    fig._accepted_audit = ["no x-axis title", "no y-axis title", "x tick labels are crowded"]
    return save(fig, outdir, name, title, cap + " " + "; ".join(parts) + ".", **fp)


def _level_table(T, within: str, across: str, filt: dict):
    a = T(across)
    t, level = (a, "across") if len(a) else (T(within), "within")
    for k, v in filt.items():
        if len(t) and k in t:
            t = t[t[k] == v]
    return t, level


def fig_tss_volcanoes(T, info, outdir, fp, genes=()):
    """The promoter switch and the TSS classes."""
    sw, level = _level_table(T, "tss_switch_within_unit", "tss_switch_across_units", {})
    eff = "mean_delta_dtui" if level == "across" else "delta_dtui"
    panels = [("TSS switch", sw, eff, "ΔDTUI (downstream TSS usage)", "downstream", "upstream")]
    for kind in ("upstream TSS", "alternative first exon", "internal exonic TSS"):
        d, lv = _level_table(T, "tss_class_within_unit", "tss_class_across_units", {"tss_class": kind})
        panels.append((kind, d, "mean_delta_share" if lv == "across" else "delta_share",
                       f"Δ share of {kind}", "up", "down"))
    test = ("the moderated one-sample t-test of the per-unit change over units, Benjamini-Hochberg over genes"
            if level == "across" else f"the {info['tests']['usage']}")
    cap = (f"Alternative transcription start sites. 5′ ends of reads with a supported TSS, clustered within "
           f"{info['params']['tss']['cluster_nt']} nt, placed on the canonical transcript. (A) Promoter switch: per "
           f"gene the two most used TSS (canonical first exon, upstream TSS, alternative first exons); DTUI = "
           f"downstream / (upstream + downstream) reads, ΔDTUI = {info['treatment']} − {info['control']}. (B–D) Reads "
           f"at upstream TSS (> {info['params']['tss']['upstream_nt']:,} nt upstream of the canonical first exon), at "
           f"alternative first exons in canonical introns and inside internal exons (downstream promoters, or reads "
           f"with truncated 5′ ends), each as a share against the canonical first exon. {ISOFORM_KEY} y: q of "
           f"{test}.")
    return _volcano_panels(T, info, outdir, fp, panels, "fig13_tss_volcanoes", "Alternative transcription start sites",
                           cap, genes)


def fig_other_apa(T, info, outdir, fp, genes=()):
    """5'UTR APA and tandem APA in non-coding genes."""
    u5, lv = _level_table(T, "apa_class_within_unit", "apa_class_across_units", {"apa_class": "5'UTR"})
    t, lt = _level_table(T, "apa_tandem_within_unit", "apa_tandem_across_units", {"coding": False})
    if lt == "within" and len(t):
        t = t[t.is_tandem]
    t = t.assign(q_family=t.q_noncoding) if len(t) else t
    panels = [("5'UTR", u5, "mean_delta_share" if lv == "across" else "delta_share", "Δ share of 5′UTR sites",
               "up", "down"),
              ("tandem 3'UTR", t, "mean_delta_pdui" if lt == "across" else "delta_pdui",
               "ΔPDUI, non-coding genes", "distal", "proximal")]
    cap = ("5′UTR and non-coding alternative polyadenylation. (A) Premature cleavage upstream of the canonical start codon "
           "(5′UTR exons and introns), as a share of the gene's reads against terminal sites. (B) Tandem sites in "
           "non-coding genes (lncRNAs and novel IsoForge genes); q "
           f"adjusted within the non-coding genes. {ISOFORM_KEY}")
    return _volcano_panels(T, info, outdir, fp, panels, "fig14_5utr_noncoding_apa", "5′UTR and non-coding APA", cap,
                           genes)


def fig_qc(T, info, outdir, fp):
    """Reads per sample, tail-call rate, reads without a model, A-rich ends, sites per gene and reads by class."""
    qc = T("qc_samples")
    sites = T("sites")
    if qc.empty:
        return None
    qc = qc.copy()
    qc["label"] = qc.dataset
    colors = np.where(qc.condition == info["control"], CONTROL_DARK, TREATED_DARK)
    fig, axd = pf.mosaic([["a", "b", "c"], ["d", "e", "f"]], width="double", aspect=0.72)
    x = np.arange(len(qc))
    bars = [("a", qc.n_reads / 1e6, "Reads (millions)"),
            ("b", 100 * qc.n_tail_call / qc.n_reads, "Reads with a tail call (%)"),
            ("c", 100 * qc.n_no_model / qc.n_reads, "Reads without a model (%)"),
            ("d", 100 * qc.n_high_fraction_as / qc.n_reads, "A-rich read ends (%)")]
    for key, v, yl in bars:
        ax = axd[key]
        ax.bar(x, v, color=colors, width=0.8, lw=0)
        if not np.nanmax(v.to_numpy()) > 0:
            ax.set_ylim(0, 1)
        ax.set_ylabel(yl)
        ax.set_xticks([])
        ax.set_xlabel(f"Samples ({len(qc)})")
    ax = axd["e"]
    if len(sites):
        per_gene = sites[sites.reads >= 10].groupby(["run", "gene_id"]).size()
        vals = per_gene.clip(upper=10).value_counts().sort_index()
        ax.bar(vals.index, vals.to_numpy() / vals.sum() * 100, color="#9AA0A8", width=0.8, lw=0)
        ax.set_xticks([1, 5, 10], ["1", "5", "≥10"])
        ax.set_xlabel("Poly(A) clusters per gene (≥ 10 reads)")
        ax.set_ylabel("Genes (%)")
    else:
        ax.set_axis_off()
    ax = axd["f"]
    if len(sites):
        cls = sites.groupby(["run", "site_class"]).reads.sum().unstack(fill_value=0)
        share = cls.div(cls.sum(axis=1), axis=0) * 100
        order = [c for c in ["terminal", "composite exon", "skipped exon", "intronic, no model", "internal exonic",
                             "upstream", "far downstream", "no canonical transcript", "opposite strand"]
                 if c in share]
        med = share[order].median()
        ax.barh(range(len(order)), med.to_numpy(), lw=0,
                color=["#C9CED6" if c in UNTESTED else class_color(c) for c in order])
        ax.set_yticks(range(len(order)), order)
        ax.set_xscale("log")
        ax.set_xlabel("Reads at poly(A) sites (%)")
        ax.set_ylabel("Site class")
        pf.categorical_axis(ax, axis="y")
        ax.invert_yaxis()
    else:
        ax.set_axis_off()
    pf.panel_labels([axd[k] for k in "abcdef"])
    fig._accepted_audit = ["no x-axis title"]
    tail_rate = (qc.n_tail_call / qc.n_reads)
    cap = (f"Quality control over {len(qc)} samples ({info['control']} teal, {info['treatment']} orange). "
           f"(A) Reads in the IsoForge read annotation (median {qc.n_reads.median() / 1e6:.1f} million). (B) Share with "
           f"a poly(A) tail call (median {tail_rate.median():.1%}). (C) Reads without an "
           f"assigned model. (D) Reads whose end is followed by A-rich genomic sequence (fraction of As > "
           f"{info['params']['apa']['internal_priming_fraction_as']}), possible internal priming. (E) Poly(A) clusters "
           f"(sites within {info['params']['apa']['cluster_nt']} nt chained) per gene with ≥ 10 reads. (F) Reads by "
           f"site class against the Ensembl-canonical transcript, median over runs (log scale).")
    return save(fig, outdir, "fig12_qc", "Quality control", cap, **fp)

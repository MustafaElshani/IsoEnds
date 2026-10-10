# Vendored from the scientific-visualization skill (scripts/pubfig.py), byte-identical below this header, so
# IsoEnds figures share its style, sizing, palettes, label placement, audit and save. Do not edit here; update the
# skill and re-vendor.
"""
pubfig: publication-figure helpers for the scientific-visualization skill.

Import from any plotting script:

    import sys; sys.path.insert(0, "<skill_dir>/scripts")
    import pubfig as pf

    pf.apply_style(journal="nature")                    # readability="comfortable" by default
    fig, axd = pf.mosaic([["a", "b"]], journal="nature", width="double", aspect=0.4)
    ...
    pf.panel_labels(axd.values())
    pf.audit(fig)                                       # sizes, overlaps, legends, spines ...
    pf.save(fig, "figures/fig1", sources=[csv], caption=text, cvd=True)
    # -> .pdf .svg .png + .provenance.json + .caption.md + _cvd.png

Plot helpers: bar_points, violin, volcano, density_scatter, venn, fit_4pl,
label_points. Stats: mean_ci, bh_fdr, overlap_p, fmt_p, fmt_n.
Checks: audit, check_palette, cvd_preview, fit_count.

Everything here is plain matplotlib, so figures stay fully editable.
"""
from __future__ import annotations

import itertools
import logging
import string
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_hex
from matplotlib.transforms import Bbox

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

MM = 1 / 25.4  # inches per millimetre

# --------------------------------------------------------------------------
# Journal presets. Widths in mm; font = journal body size; min_font = the
# smallest text the journal accepts at print size. Confirm against the
# current author guide before submission.
# --------------------------------------------------------------------------
JOURNALS = {
    "nature":  dict(single=89,  onehalf=120, double=183, font=7, min_font=5, label_size=8,
                    label_case="lower", family="sans"),
    "science": dict(single=57,  onehalf=121, double=184, font=7, min_font=5, label_size=8,
                    label_case="upper", family="sans"),
    "cell":    dict(single=85,  onehalf=114, double=174, font=7, min_font=6, label_size=8,
                    label_case="upper", family="sans"),
    "pnas":    dict(single=87,  onehalf=114, double=178, font=7, min_font=6, label_size=8,
                    label_case="upper", family="sans"),
    "ieee":    dict(single=88.9, onehalf=120, double=181, font=8, min_font=7, label_size=9,
                    label_case="paren", family="serif"),
    "neurips": dict(single=139.7, onehalf=139.7, double=139.7, font=8, min_font=6, label_size=9,
                    label_case="lower", family="sans"),
    "icml":    dict(single=82.5, onehalf=120, double=171.5, font=8, min_font=6, label_size=9,
                    label_case="lower", family="sans"),
    "generic": dict(single=90,  onehalf=140, double=190, font=7, min_font=5, label_size=8,
                    label_case="lower", family="sans"),
}

# --------------------------------------------------------------------------
# Readability levels: offsets (pt) added to the journal body size.
#   strict      – exactly the journal's stated sizes (tick labels 1 pt smaller)
#   comfortable – DEFAULT. +1 pt on axis titles, ticks = body size. What
#                 reviewers read easily; still inside most journals' range.
#   large       – for collages/figures that will be shrunk, slides, posters.
# The floor for tick labels is always the journal's min_font + 1.
# --------------------------------------------------------------------------
READABILITY = {
    "strict":      dict(label=0, tick=-1, legend=-1, title=0, panel=0, stroke=1.00),
    "comfortable": dict(label=1, tick=0,  legend=0,  title=1, panel=2, stroke=1.15),
    "large":       dict(label=3, tick=2,  legend=2,  title=3, panel=4, stroke=1.40),
}

SANS_STACK = ["Inter", "Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]
SERIF_STACK = ["Times New Roman", "Nimbus Roman", "STIXGeneral", "CMU Serif", "DejaVu Serif"]

# --------------------------------------------------------------------------
# Palettes
# --------------------------------------------------------------------------
PALETTES = {
    # Okabe & Ito (2008): colour-blind-safe default for categories.
    "okabe_ito": ["#0072B2", "#E69F00", "#009E73", "#D55E00",
                  "#56B4E9", "#CC79A7", "#F0E442", "#000000"],
    # Muted deep-navy + amber family.
    "navy_amber": ["#1B2A49", "#E3A33B", "#4F7CAC", "#B5543A",
                   "#6F9A6B", "#8A7BA8", "#9A9A9A"],
    # Two groups: grey control, navy treatment.
    "pair": ["#9AA5B1", "#1B2A49"],
    # Three groups: grey control + two treatments.
    "trio": ["#9AA5B1", "#1B2A49", "#E3A33B"],
    # Emphasis: one series coloured, the rest grey.
    "focus": ["#1B2A49", "#C9CED6", "#C9CED6", "#C9CED6", "#C9CED6"],
    # Paired pre/post or control/treated: pastel rose and blue fills …
    "rose_slate": ["#D4AFB9", "#9CADCE"],
    # … with matching dark accents for medians, quartiles and median labels.
    "rose_slate_dark": ["#8B4557", "#5B7A9D"],
    # Treated vs control, used consistently across panels.
    # blue vs amber: distinct from each other and from background grey, even pale
    "treat_control": {"control": "#1F5FA8", "treatment": "#D1782A"},
    # Volcano / up-down regulation.
    "updown": {"up": "#D1782A", "down": "#2F5D8A", "ns": "#C9CED6"},
    # Neutral inks for text, axes, reference lines.
    "ink": {"text": "#1A1A1A", "axis": "#2B2B2B", "grid": "#E6E6E6",
            "muted": "#7A808A", "ref": "#A9AEB5"},
}
INK = PALETTES["ink"]

CHARCOAL_NEON = LinearSegmentedColormap.from_list(
    "charcoal_neon", ["#141414", "#1E3446", "#146C7C", "#2BB39B", "#9BE564", "#E8FF8A"])
NAVY_AMBER_DIV = LinearSegmentedColormap.from_list(
    "navy_amber_div", ["#1B2A49", "#4F7CAC", "#F5F5F2", "#E3A33B", "#8A4B0F"])
# Light → dark navy: point density (dense core = darkest), monotonic lightness.
ICE_NAVY = LinearSegmentedColormap.from_list(
    "ice_navy", ["#D3DDE8", "#9DB3CB", "#5F86B0", "#335A86", "#1B2A49"])
for _cm in (CHARCOAL_NEON, NAVY_AMBER_DIV, ICE_NAVY):
    for _c in (_cm, _cm.reversed()):
        if _c.name not in mpl.colormaps:
            mpl.colormaps.register(_c)

BANNED_CMAPS = {"jet", "rainbow", "hsv", "gist_rainbow", "nipy_spectral", "gist_ncar"}

# Remembers the last apply_style() call so other helpers stay consistent.
_STATE: dict = {"journal": "nature", "readability": "comfortable", "sizes": {}}


def palette(name: str = "okabe_ito", n: int | None = None) -> list[str]:
    cols = PALETTES[name]
    if n is None:
        return list(cols)
    if n > len(cols):
        raise ValueError(f"Palette '{name}' has {len(cols)} colours, {n} requested. More than ~7 "
                         "categories is unreadable: facet, group, or highlight one series.")
    return list(cols[:n])


def tint(color, f: float = 0.55) -> str:
    """Lighter version of a colour (blend toward white by f)."""
    r, g, b = mpl.colors.to_rgb(color)
    return to_hex((r + (1 - r) * f, g + (1 - g) * f, b + (1 - b) * f))


def darken(color, f: float = 0.62) -> str:
    """Darker version of a colour, for outlines and median bars."""
    r, g, b = mpl.colors.to_rgb(color)
    return to_hex((r * f, g * f, b * f))


def treat_control_colors(order) -> dict:
    """{control: blue, treatment: amber} for [control, treatment] (reference first)."""
    tc = PALETTES["treat_control"]
    order = list(order)
    if len(order) == 2:
        return {order[0]: tc["control"], order[1]: tc["treatment"]}
    return group_colors(order)


def group_colors(groups, name: str | None = None) -> dict:
    """Map group names to colours, picking a sensible palette for the count."""
    groups = list(groups)
    if name is None:
        name = {2: "pair", 3: "trio"}.get(len(groups), "okabe_ito")
    return dict(zip(groups, palette(name, len(groups))))


# --------------------------------------------------------------------------
# Style
# --------------------------------------------------------------------------
def resolve_font(stack: list[str]) -> str:
    """First family in `stack` that is actually installed."""
    from matplotlib import font_manager as fm
    for name in stack:
        try:
            fm.findfont(fm.FontProperties(family=name), fallback_to_default=False)
            return name
        except ValueError:
            continue
    return stack[-1]


def sizes(journal: str | None = None, readability: str | None = None) -> dict:
    """Font sizes (pt) for a journal/readability combination."""
    journal = journal or _STATE["journal"]
    readability = readability or _STATE["readability"]
    spec, r = JOURNALS[journal], READABILITY[readability]
    base = spec["font"]
    tick = max(base + r["tick"], spec["min_font"] + 1)
    return dict(
        base=base, label=max(base + r["label"], tick), tick=tick,
        legend=max(base + r["legend"], tick), title=base + r["title"],
        panel=spec["label_size"] + r["panel"], min=spec["min_font"],
        annot=tick, stroke=r["stroke"],
    )


def apply_style(journal: str = "nature", readability: str = "comfortable",
                palette_name: str = "okabe_ito", transparent: bool = False) -> dict:
    """Set global rcParams for a clean, minimal, journal-sized, readable look."""
    spec = JOURNALS[journal]
    sz = sizes(journal, readability)
    k = sz["stroke"]
    _STATE.update(journal=journal, readability=readability, sizes=sz)
    family = "sans-serif" if spec["family"] == "sans" else "serif"
    face = resolve_font(SANS_STACK if spec["family"] == "sans" else SERIF_STACK)

    rc = {
        # Typography
        "font.family": family,
        "font.sans-serif": SANS_STACK,
        "font.serif": SERIF_STACK,
        "mathtext.fontset": "custom",       # math ($r$, $\mu$m) in the body face
        "mathtext.rm": face,
        "mathtext.it": f"{face}:italic",
        "mathtext.bf": f"{face}:bold",
        "font.size": sz["tick"],
        "axes.titlesize": sz["title"],
        "axes.titleweight": "regular",
        "axes.titlelocation": "left",
        "axes.titlepad": 4,
        "axes.labelsize": sz["label"],
        "axes.labelpad": 3,
        "xtick.labelsize": sz["tick"],
        "ytick.labelsize": sz["tick"],
        "legend.fontsize": sz["legend"],
        "legend.title_fontsize": sz["label"],
        "figure.titlesize": sz["title"],
        "text.color": INK["text"],
        "axes.labelcolor": INK["text"],
        # Axes: open L-frame (left + bottom always visible), no fill
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
        "axes.linewidth": 0.6 * k,
        "axes.edgecolor": INK["axis"],
        "axes.facecolor": "none" if transparent else "white",
        "axes.grid": False,
        "axes.axisbelow": True,
        "axes.prop_cycle": mpl.cycler(color=PALETTES[palette_name]),
        "axes.xmargin": 0.03,
        "axes.ymargin": 0.05,
        "grid.color": INK["grid"],
        "grid.linewidth": 0.4,
        # Ticks: outward, short, thin, a little breathing room
        "xtick.direction": "out",
        "ytick.direction": "out",
        "xtick.major.size": 2.5 * k,
        "ytick.major.size": 2.5 * k,
        "xtick.minor.size": 1.5 * k,
        "ytick.minor.size": 1.5 * k,
        "xtick.major.width": 0.6 * k,
        "ytick.major.width": 0.6 * k,
        "xtick.minor.width": 0.45 * k,
        "ytick.minor.width": 0.45 * k,
        "xtick.major.pad": 2.5,
        "ytick.major.pad": 2.5,
        "xtick.color": INK["axis"],
        "ytick.color": INK["axis"],
        # Marks
        "lines.linewidth": 1.0 * k,
        "lines.markersize": 3 * k,
        "lines.solid_capstyle": "round",
        "patch.linewidth": 0.5 * k,
        "errorbar.capsize": 2.2 * k,     # error bars always end in caps
        "scatter.edgecolors": "none",
        # Legend: no box
        "legend.frameon": False,
        "legend.handlelength": 1.2,
        "legend.handletextpad": 0.4,
        "legend.borderaxespad": 0.2,
        "legend.labelspacing": 0.3,
        "legend.columnspacing": 1.0,
        # Figure & export
        "figure.facecolor": "none" if transparent else "white",
        "figure.dpi": 150,
        "figure.constrained_layout.use": True,
        "figure.constrained_layout.h_pad": 3 / 72,
        "figure.constrained_layout.w_pad": 3 / 72,
        "figure.constrained_layout.hspace": 0.06,
        "figure.constrained_layout.wspace": 0.06,
        "savefig.dpi": 600,
        "savefig.bbox": None,          # keep the exact journal width
        "savefig.transparent": transparent,
        "image.cmap": "viridis",
        "svg.fonttype": "none",        # editable text in SVG
        "pdf.fonttype": 42,            # embedded TrueType, not Type 3
        "ps.fonttype": 42,
    }
    mpl.rcParams.update(rc)
    if face != (SANS_STACK if spec["family"] == "sans" else SERIF_STACK)[0] and \
            _STATE.get("warned_font") != face:
        _STATE["warned_font"] = face
        print(f"[pubfig] using font '{face}' (preferred face not installed)")
    return rc


def figsize(journal: str | None = None, width: str | float = "single",
            aspect: float = 0.75) -> tuple[float, float]:
    """(w, h) inches. width: 'single' | 'onehalf' | 'double' | mm. aspect = h / w."""
    spec = JOURNALS[journal or _STATE["journal"]]
    w_mm = spec[width] if isinstance(width, str) else float(width)
    return (w_mm * MM, w_mm * MM * aspect)


def tidy_ticks(ax, nbins: int = 5):
    """4–6 round-valued ticks per linear axis (matplotlib's default can give 9)."""
    if getattr(ax, "_pf_broken_upper", False):
        nbins = 2                                  # short upper segment of a broken axis
    for axis in (ax.xaxis, ax.yaxis):
        if type(axis.get_major_locator()) is mpl.ticker.AutoLocator:
            axis.set_major_locator(mpl.ticker.MaxNLocator(
                nbins=nbins, steps=[1, 2, 2.5, 5, 10], min_n_ticks=3))


def fit_count(ax, n_max: int, axis: str = "y", spacing: float = 1.5) -> int:
    """How many tick labels fit along an axis at the current font size.

    Draws the figure to measure the axes, so call it after the layout exists
    (e.g. right before filling a heatmap) and use the result to cap row labels.
    """
    fig = ax.figure
    fig.canvas.draw()
    bb = ax.get_window_extent()
    length_pt = (bb.height if axis == "y" else bb.width) * 72 / fig.dpi
    return int(max(1, min(n_max, length_pt // (sizes()["tick"] * spacing))))


def subplots(nrows: int = 1, ncols: int = 1, journal: str | None = None,
             width: str | float = "single", aspect: float | None = None, **kw):
    if aspect is None:
        aspect = 0.75 * nrows / ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize(journal, width, aspect), **kw)
    for ax in _axes_list(axes):
        tidy_ticks(ax)
    return fig, axes


def mosaic(layout, journal: str | None = None, width: str | float = "double",
           aspect: float = 0.5, **kw):
    """plt.subplot_mosaic sized to the journal. Returns (fig, dict_of_axes)."""
    fig = plt.figure(figsize=figsize(journal, width, aspect))
    axd = fig.subplot_mosaic(layout, **kw)
    for ax in axd.values():
        tidy_ticks(ax)
    return fig, axd


# --------------------------------------------------------------------------
# Finishing touches
# --------------------------------------------------------------------------
def _axes_list(axes):
    if isinstance(axes, dict):
        return list(axes.values())
    if hasattr(axes, "flat"):
        return list(axes.flat)
    if isinstance(axes, (list, tuple)) or hasattr(axes, "__iter__"):
        return list(axes)
    return [axes]


def panel_labels(axes, journal: str | None = None, start: int = 0,
                 x: float = -10, y: float = 5, align_rows: bool = True):
    """Bold panel letters just outside each axes' top-left corner (offsets in pt).

    align_rows: letters of panels that share a grid row sit at the same height,
    even when one panel carries marginal plots above it.
    """
    journal = journal or _STATE["journal"]
    case = JOURNALS[journal]["label_case"]
    size = sizes(journal)["panel"]
    axes = _axes_list(axes)
    for i, ax in enumerate(axes):
        letter = string.ascii_lowercase[i + start]
        text = {"lower": letter, "upper": letter.upper(), "paren": f"({letter})"}[case]
        ss = ax.get_subplotspec()
        key = (id(ss.get_gridspec()), ss.rowspan.start) if ss is not None else None
        top = getattr(ax, "_pf_top", 1.0)
        ann = ax.annotate(text, xy=(0, top), xycoords="axes fraction", xytext=(x, y),
                          textcoords="offset points", fontsize=size,
                          fontweight="normal" if case == "paren" else "bold",
                          ha="right", va="bottom", gid="panel_label")
        if align_rows and key is not None:
            ax.figure.__dict__.setdefault("_pf_rows", {}).setdefault(key, []).append((ax, ann))


def categorical_axis(ax, axis: str = "x", rotate="auto"):
    """Tidy a categorical axis: keep the axis line, drop the tick marks.

    rotate "auto": if the x labels would not fit side by side in their slots,
    turn them vertical, reading upward and hanging BELOW the axis
    (rotation=90, ha="right", va="center", rotation_mode="anchor": with
    ha="center" anchored text runs up through the bars instead).
    True forces it, False never rotates.
    """
    ax.tick_params(axis=axis, length=0, pad=4)
    ax.spines["bottom" if axis == "x" else "left"].set_visible(True)
    if axis != "x" or rotate is False:
        return
    labs = [t.get_text() for t in ax.get_xticklabels()]
    if not labs:
        return
    if rotate == "auto":
        ax._pf_autorotate = True        # re-checked after layout in finalize()
        fig = ax.figure
        ax_w_pt = ax.get_position().width * fig.get_figwidth() * 72
        lo, hi = ax.get_xlim()
        ticks = ax.get_xticks()
        slot_pt = (ax_w_pt * (np.min(np.diff(ticks)) / (hi - lo))) if len(ticks) > 1 else ax_w_pt
        widest = max(_text_width_pt(t, sizes()["tick"]) for t in labs)
        rotate = widest + 4.5 > slot_pt     # same margin the audit's crowding check needs
    if rotate:
        rotate_xlabels(ax)


def _labels_crowded(ax, renderer, min_gap_pt: float = 2.5) -> bool:
    labs = [t for t in ax.get_xticklabels() if t.get_visible() and t.get_text().strip()]
    bbs = sorted((_text_bbox(t, renderer) for t in labs), key=lambda b: b.x0)
    dpi = ax.figure.dpi
    return any((b2.x0 - b1.x1) * 72 / dpi < min_gap_pt for b1, b2 in zip(bbs, bbs[1:]))


def rotate_xlabels(ax, rotation: float = 90):
    """Vertical category labels hanging below the axis (anchor at the label's end)."""
    ax._pf_rotated = True
    for t in ax.get_xticklabels():
        t.set_rotation(rotation)
        t.set_rotation_mode("anchor")
        t.set_ha("right")
        t.set_va("center")


def image_axis(ax):
    """Heatmaps/images: no spines, no tick marks (the image is its own frame)."""
    for s in ax.spines.values():
        s.set_visible(False)
    ax.tick_params(length=0)
    ax.set_gid("image_axes")


def offset_spines(ax, amount: float = 3):
    for side in ("left", "bottom"):
        ax.spines[side].set_position(("outward", amount))


def trim_spines(ax):
    """Limit spines to the range of the tick marks."""
    x0, x1 = sorted(ax.get_xlim()); y0, y1 = sorted(ax.get_ylim())
    xt = [t for t in ax.get_xticks() if x0 <= t <= x1]
    yt = [t for t in ax.get_yticks() if y0 <= t <= y1]
    if xt:
        ax.spines["bottom"].set_bounds(min(xt), max(xt))
    if yt:
        ax.spines["left"].set_bounds(min(yt), max(yt))


def direct_label(ax, x, y, text, color, dx: float = 3, **kw):
    """Label a line at its end instead of using a legend."""
    kw.setdefault("fontsize", sizes()["legend"])
    ax.annotate(text, xy=(x, y), xytext=(dx, 0), textcoords="offset points",
                color=color, va="center", ha="left", **kw)


def stat_title(ax, text):
    """Muted, right-aligned statistics line above the axes (never collides with data)."""
    top = getattr(ax, "_pf_top", 1.0)
    if top > 1.0:   # marginal axes above: place the line above them instead
        ax.annotate(text, xy=(1, top), xycoords="axes fraction", xytext=(0, 5),
                    textcoords="offset points", ha="right", va="bottom",
                    color=INK["muted"], fontsize=sizes()["annot"])
    else:
        ax.set_title(text, loc="right", color=INK["muted"], fontsize=sizes()["annot"])


def fig_stat_title(fig, text, loc: str = "right"):
    """Muted one-line note above a whole figure (n, test, thresholds)."""
    xpos = {"right": 0.995, "left": 0.005, "center": 0.5}[loc]
    return fig.suptitle(text, x=xpos, ha=loc, color=INK["muted"],
                        fontsize=sizes()["annot"])


def refline(ax, y=None, x=None, **kw):
    style = dict(color=INK["ref"], lw=0.6, ls=(0, (3, 2)), zorder=0)
    style.update(kw)
    if y is not None:
        ax.axhline(y, **style)
    if x is not None:
        ax.axvline(x, **style)


def colorbar(mappable, ax, label: str = "", **kw):
    """Slim colorbar without an outline."""
    cb = plt.colorbar(mappable, ax=ax, fraction=0.05, pad=0.03, aspect=22, **kw)
    cb.outline.set_visible(False)
    cb.ax.tick_params(length=2, width=0.5)
    if label:
        cb.set_label(label)
    return cb


def label_points(ax, x, y, labels, color=None, max_radius: float = 36,
                 drop_crowded: bool = True):
    """Queue point labels; they are placed by finalize() once the layout is fixed.

    (Placing them earlier is wasted work: constrained_layout moves the axes
    again whenever titles or panel letters are added.)
    """
    pending = ax.figure.__dict__.setdefault("_pubfig_pending", [])
    pending.append((ax, np.asarray(x), np.asarray(y), np.asarray(labels), color, max_radius,
                    drop_crowded))


def finalize(fig):
    """Freeze the layout, then place queued point labels. Called by audit()/save()."""
    if not fig.__dict__.get("_pubfig_tidied"):
        for ax in fig.axes:
            if ax.name == "rectilinear":
                tidy_ticks(ax)
        fig._pubfig_tidied = True
    pending = fig.__dict__.get("_pubfig_pending", [])
    rows = fig.__dict__.get("_pf_rows", {})
    jobs = fig.__dict__.get("_pf_jobs", [])
    auto = [a for a in fig.axes if getattr(a, "_pf_autorotate", False)
            and not getattr(a, "_pf_rotated", False)]
    if auto:     # category labels judged on the final layout, not the planned one
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        for a in auto:
            if _labels_crowded(a, r):
                rotate_xlabels(a)
    if not pending and not rows and not jobs:
        return
    # constrained_layout sizes margins from the previous pass; decorations that
    # scale with their axes (ladders, strips) need a couple of passes to settle
    for _ in range(3):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        over = max((a.get_tightbbox(r).y1 - fig.bbox.y1 for a in fig.axes
                    if a.get_visible() and a.get_tightbbox(r) is not None), default=0)
        if over <= 0.5:
            break
    fig.set_layout_engine("none")      # positions are final from here on
    for job in jobs:                   # post-layout placement (e.g. spectrum ion labels)
        job()
    fig._pf_jobs = []
    # panel letters: same height across a grid row (axes heights can differ,
    # e.g. equal-aspect Venn panels or panels with marginal plots)
    for members in rows.values():
        tops = []
        for ax, ann in members:
            bb = ax.get_window_extent()
            tops.append(bb.y0 + getattr(ax, "_pf_top", 1.0) * bb.height)
        target = max(tops)
        for ax, ann in members:
            bb = ax.get_window_extent()
            ann.xy = (0, (target - bb.y0) / bb.height)
    fig._pf_rows = {}
    for args in pending:
        _place_labels(*args)
    fig._pubfig_pending = []


def _text_bbox(t, renderer):
    """Bounding box of the text itself (Annotation extents include the leader line)."""
    if isinstance(t, mpl.text.Annotation):
        t.update_positions(renderer)
    return mpl.text.Text.get_window_extent(t, renderer)


def _segments_cross(p1, p2, q1, q2) -> bool:
    def orient(a, b, c):
        return np.sign((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
    return (orient(p1, p2, q1) != orient(p1, p2, q2)) and (orient(q1, q2, p1) != orient(q1, q2, p2))


def _place_labels(ax, x, y, labels, color=None, max_radius: float = 36,
                  drop_crowded: bool = True):
    """Label points with text that avoids other labels, data points and the axes edge.

    Greedy search over candidate offsets (8 directions × several radii); the
    first label goes to its best spot, the next avoids it, and so on. Thin
    leader lines connect labels that ended up away from their point.
    """
    fig = ax.figure
    fs = sizes()["annot"]
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(r)
    pts = [c.get_offset_transform().transform(c.get_offsets())
           for c in ax.collections if len(c.get_offsets())]
    for ln in ax.lines:
        xy = ln.get_xydata()
        if ln.get_visible() and len(xy):
            pts.append(ln.get_transform().transform(xy))
    P = np.vstack(pts) if pts else np.empty((0, 2))
    dirs = [(1, 0.4), (-1, 0.4), (1, 1), (-1, 1), (0.4, 1), (-0.4, 1), (0, 1),
            (1, -1), (-1, -1), (0.4, -1), (-0.4, -1), (0, -1), (1, -0.4), (-1, -0.4)]
    radii = [x_ for x_ in (6, 10, 15, 21, 28, 36) if x_ <= max_radius]
    placed, anns, dropped, leaders = [], [], [], []
    lab_disp = ax.transData.transform(np.column_stack([np.asarray(x, float), np.asarray(y, float)]))
    # avoid text already in the axes (counts, annotations, other labels)
    for t in ax.texts:
        if t.get_visible() and t.get_text().strip():
            placed.append(_text_bbox(t, r))
    order = np.argsort(-np.asarray(y, float))
    x, y, labels = np.asarray(x), np.asarray(y), np.asarray(labels)
    for i in order:
        a = ax.annotate(labels[i], xy=(x[i], y[i]), xytext=(7, 3), textcoords="offset points",
                        fontsize=fs, color=color or INK["text"], zorder=5, in_layout=False)
        best, best_cost = None, np.inf
        for rad in radii:
            for dx, dy in dirs:
                off = (dx * rad, dy * rad * 0.8)
                a.xyann = off
                a.set_ha("left" if dx > 0.3 else "right" if dx < -0.3 else "center")
                a.set_va("bottom" if dy > 0.5 else "top" if dy < -0.5 else "center")
                bb = _text_bbox(a, r)
                out = (max(0, ax_bb.x0 - bb.x0) + max(0, bb.x1 - ax_bb.x1)
                       + max(0, ax_bb.y0 - bb.y0) + max(0, bb.y1 - ax_bb.y1))
                clash = sum(bb.overlaps(q) for q in placed)
                hits = int(((P[:, 0] > bb.x0) & (P[:, 0] < bb.x1) &
                            (P[:, 1] > bb.y0) & (P[:, 1] < bb.y1)).sum()) if len(P) else 0
                # leader line should not cut through the data
                p0 = np.array(ax.transData.transform((x[i], y[i])))
                p1 = np.array([np.clip(p0[0], bb.x0, bb.x1), np.clip(p0[1], bb.y0, bb.y1)])
                cross = 0
                if len(P):
                    seg = p0 + np.linspace(0.15, 0.85, 10)[:, None] * (p1 - p0)
                    d = np.sqrt(((P[None, :, :] - seg[:, None, :]) ** 2).sum(-1))
                    cross = int((d.min(0) < 3).sum())
                # leader must not pass through other labels or cross other leaders
                if True:
                    seg2 = p0 + np.linspace(0.1, 0.9, 12)[:, None] * (p1 - p0)
                    for q in placed:
                        if ((seg2[:, 0] > q.x0) & (seg2[:, 0] < q.x1) &
                                (seg2[:, 1] > q.y0) & (seg2[:, 1] < q.y1)).any():
                            clash += 1
                    clash += sum(_segments_cross(p0, p1, a0, a1) for a0, a1 in leaders)
                # ambiguity: label must be nearer its own point than any other labelled point
                own = np.linalg.norm(p1 - p0)
                others = np.array([np.linalg.norm(np.array([np.clip(o[0], bb.x0, bb.x1),
                                                            np.clip(o[1], bb.y0, bb.y1)]) - o)
                                   for j, o in enumerate(lab_disp) if j != i])
                ambiguous = int(len(others) and others.min() < own)
                cost = (100 * out + 61 * clash + 61 * hits + 12 * cross
                        + 30 * ambiguous + rad * 1.5)
                if cost < best_cost:
                    best, best_cost = (off, a.get_ha(), a.get_va(), bb), cost
            if best_cost < 10:   # clean spot found at this radius
                break
        if drop_crowded and best_cost > 60:
            a.remove()          # no clean spot: an unreadable label is worse than none
            dropped.append(str(labels[i]))
            continue
        a.xyann, (_, ha, va, bb) = best[0], best
        a.set_ha(ha); a.set_va(va)
        # leader: from just outside the point to the nearest edge of the text box
        tb = _text_bbox(a, r)
        p0 = np.array(ax.transData.transform((x[i], y[i])))
        q = np.array([np.clip(p0[0], tb.x0, tb.x1), np.clip(p0[1], tb.y0, tb.y1)])
        dist = np.linalg.norm(q - p0)
        px_per_pt = fig.dpi / 72
        if dist > 3.5 * px_per_pt:
            u = (q - p0) / dist
            start = p0 + u * 2.0 * px_per_pt
            end = q - u * 1.0 * px_per_pt
            (d0, d1) = ax.transData.inverted().transform(np.vstack([start, end]))
            ax.plot([d0[0], d1[0]], [d0[1], d1[1]], color=INK["muted"], lw=0.45,
                    zorder=4, solid_capstyle="butt", gid="leader")
        placed.append(_text_bbox(a, r))
        if dist > 3.5 * px_per_pt:
            leaders.append((p0, q))
        anns.append(a)
    if dropped:
        print(f"[pubfig] skipped crowded labels (no clear space): {', '.join(dropped)}")
    return anns


# --------------------------------------------------------------------------
# Statistics & plot helpers
# --------------------------------------------------------------------------
def errorbar_style(lw: float | None = None) -> dict:
    """Error-bar look used across the skill: thin line with end caps of the same
    weight. Pass to ax.errorbar(**pf.errorbar_style())."""
    k = sizes()["stroke"] if _STATE.get("sizes") else 1.0
    lw = 0.75 * k if lw is None else lw
    return dict(elinewidth=lw, capsize=2.2 * k, capthick=lw)


MEAN_LINE_LW = 0.8     # × stroke: the mean line on top of a bar: present, not heavy


def mean_ci(values, level: float = 0.95):
    """Mean and half-width of the t-based confidence interval."""
    v = np.asarray(values, float)
    v = v[~np.isnan(v)]
    n = len(v)
    if n < 2:
        return float(np.mean(v)) if n else np.nan, np.nan
    se = v.std(ddof=1) / np.sqrt(n)
    try:
        from scipy import stats
        q = stats.t.ppf(0.5 + level / 2, n - 1)
    except ImportError:
        q = 1.96
    return float(v.mean()), float(q * se)


def fmt_p(p: float, digits: int = 2) -> str:
    """Journal-style P value: 'P = 0.032', 'P = 3.1 × 10$^{-5}$', 'P < 10$^{-16}$'."""
    if p < 1e-16:
        return "$P$ < 10$^{-16}$"
    if p >= 0.001:
        return f"$P$ = {p:.{digits}g}" if p < 0.01 else f"$P$ = {p:.{digits}f}"
    m, e = f"{p:.{digits - 1}e}".split("e")
    return f"$P$ = {m} × 10$^{{{int(e)}}}$"


def fmt_n(counts, unit: str = "") -> str:
    """'n = 12' or 'n = 8–12' from an iterable of group sizes."""
    counts = list(counts)
    lo, hi = min(counts), max(counts)
    rng_ = f"{lo}" if lo == hi else f"{lo}–{hi}"
    return f"$n$ = {rng_}{(' ' + unit) if unit else ''}"


def bh_fdr(p):
    """Benjamini–Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / np.arange(1, n + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def _jitter(n, width, rng):
    return rng.uniform(-width, width, n)


def bar_points(ax, data, x, y, order=None, colors=None, error: str = "ci",
               width: float = 0.62, point_size: float = 9, seed: int = 0):
    """Bar of the mean + every observation + error bar (the reviewer-proof bar chart).

    error: 'ci' (95% t-CI) | 'sd' | 'sem'.
    """
    order = list(order or data[x].unique())
    colors = colors or group_colors(order)
    rng = np.random.default_rng(seed)
    for i, g in enumerate(order):
        v = data.loc[data[x] == g, y].dropna().to_numpy()
        m, ci = mean_ci(v)
        err = {"ci": ci, "sd": v.std(ddof=1), "sem": v.std(ddof=1) / np.sqrt(len(v))}[error]
        ax.bar(i, m, width=width, color=colors[g], alpha=0.28, lw=0, zorder=1)
        ax.plot([i - width / 2, i + width / 2], [m, m], color=colors[g],
                lw=MEAN_LINE_LW * sizes()["stroke"], zorder=2, solid_capstyle="butt")
        ax.scatter(i + _jitter(len(v), width * 0.28, rng), v, s=point_size,
                   color=colors[g], alpha=0.85, zorder=3, lw=0)
        ax.errorbar(i, m, yerr=err, color=INK["text"], zorder=4, **errorbar_style())
    ax.set_xticks(range(len(order)), order)
    ax.set_xlim(-0.6, len(order) - 0.4)
    categorical_axis(ax, "x")
    if ax.get_ylim()[0] > 0:
        ax.set_ylim(bottom=0)


def violin(ax, data, x, y, order=None, colors=None, points: bool = True,
           max_points: int = 400, seed: int = 0):
    """Half-transparent violin + slim box (median/IQR) + optional thinned points."""
    order = list(order or data[x].unique())
    colors = colors or group_colors(order)
    rng = np.random.default_rng(seed)
    vals = [data.loc[data[x] == g, y].dropna().to_numpy() for g in order]
    parts = ax.violinplot(vals, positions=range(len(order)), widths=0.8,
                          showextrema=False, showmedians=False)
    for body, g in zip(parts["bodies"], order):
        body.set_facecolor(colors[g]); body.set_alpha(0.35); body.set_edgecolor("none")
    for i, (g, v) in enumerate(zip(order, vals)):
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        iqr = q3 - q1
        lo, hi = v[v >= q1 - 1.5 * iqr].min(), v[v <= q3 + 1.5 * iqr].max()
        if points:
            sub = v if len(v) <= max_points else rng.choice(v, max_points, replace=False)
            ax.scatter(i + _jitter(len(sub), 0.07, rng) + 0.22, sub, s=2.5,
                       color=colors[g], alpha=0.5, lw=0, zorder=2, rasterized=True)
        ax.plot([i, i], [lo, hi], color=INK["text"], lw=0.7, zorder=3)
        ax.hlines([lo, hi], i - 0.035, i + 0.035, color=INK["text"], lw=0.7, zorder=3)  # caps
        ax.add_patch(mpl.patches.Rectangle((i - 0.05, q1), 0.10, iqr, facecolor="white",
                                           edgecolor=INK["text"], lw=0.7, zorder=4))
        ax.plot([i - 0.05, i + 0.05], [med, med], color=INK["text"], lw=1.3, zorder=5,
                solid_capstyle="butt")
    ax.set_xticks(range(len(order)), order)
    ax.set_xlim(-0.6, len(order) - 0.4)
    categorical_axis(ax, "x")


def _kde_curve(v, grid, bw=None):
    from scipy import stats as st
    if len(v) < 3 or np.allclose(v, v[0]):
        return None
    k = st.gaussian_kde(v, bw_method=bw)
    return k(grid)


def split_violin(ax, data, x, y, hue, order=None, hue_order=None, colors=None,
                 dark_colors=None, scale: str = "width", width: float = 0.8, gap: float = 0.008,
                 alpha: float = 0.8, edge: bool = False, cut: float = 0.0,
                 median_labels: bool = False, label_pos: str = "auto",
                 points: bool = False, max_points: int = 200, paired: bool = False,
                 link: bool = False, test: str | None = "auto", min_n: int = 3,
                 legend: bool = True, seed: int = 0):
    """Two conditions per category drawn as facing half-violins (control/treated,
    pre/post, WT/KO …), e.g. vehicle vs drug per cell line.

    Look: soft pastel fills with no outline; inside each half a short median
    bar (dark accent colour) with thinner quartile bars; optional median values
    in bold just outside the violin. One violin position per level of `x`.

    colors / dark_colors  fills and accents per hue level; default the
            pastel rose/blue pair with its dark accents (#8B4557 / #5B7A9D).
    alpha   fill opacity (0.8).  edge   draw an outline (off by default).
    gap     half-distance between the two halves (x units): 0.008 leaves a thin
            white divider down the middle; 0 makes them touch.
    scale   "width" (each half filled to the same width), "count" (half width
            ∝ n) or "area".
    label_pos  where the median value goes:
            "above"   (default) centred on top of its median bar, inside the half,
                      with a thin white halo: contained and tied to the bar;
            "inside"  beside the median bar, facing outward (may spill past a
                      narrow half);
            "outside" just beyond the violin edge (needs extra side room);
            "below"   a row of medians just above the x-axis, under each half.
    paired  repeated measures of the same unit → signed-rank test by default;
            link=True also joins the two medians with a thin line.
    test    "auto" (Wilcoxon if paired, else Mann–Whitney), "ks" (two-sample
            Kolmogorov–Smirnov: any difference in distribution shape),
            "wilcoxon", "mannwhitney", or None. Needs ≥ min_n per side.
            P values are Benjamini–Hochberg adjusted across categories.

    Returns {"groups": {category: {...}}, "hue_order": [...], "handles": [...]}.
    """
    from scipy import stats as st
    rng = np.random.default_rng(seed)
    order = list(order or data[x].dropna().unique())
    hue_order = list(hue_order or data[hue].dropna().unique())
    if len(hue_order) != 2:
        raise ValueError(f"split_violin needs exactly 2 levels of '{hue}', got {len(hue_order)}.")
    default_pal = colors is None
    if colors is None:
        colors = dict(zip(hue_order, palette("rose_slate", 2)))
    elif not isinstance(colors, dict):
        colors = dict(zip(hue_order, colors))
    if dark_colors is None:
        dark_colors = (dict(zip(hue_order, palette("rose_slate_dark", 2))) if default_pal
                       else {h: darken(c) for h, c in colors.items()})
    elif not isinstance(dark_colors, dict):
        dark_colors = dict(zip(hue_order, dark_colors))
    k = sizes()["stroke"]

    vals = {(g, h): data.loc[(data[x] == g) & (data[hue] == h), y].dropna().to_numpy()
            for g in order for h in hue_order}
    n_max = max(len(v) for v in vals.values()) or 1
    half = width / 2
    fs_lab = sizes()["annot"]
    if label_pos == "auto":
        label_pos = "above"

    out = {}
    for i, g in enumerate(order):
        rec = {"n": {}, "median": {}, "iqr": {}}
        for side, h in zip((-1, 1), hue_order):
            v = vals[(g, h)]
            rec["n"][h] = len(v)
            if len(v) == 0:
                continue
            q1, med, q3 = np.percentile(v, [25, 50, 75])
            rec["median"][h] = float(med)
            rec["iqr"][h] = (float(q1), float(q3))
            lo, hi = v.min() - cut * (v.max() - v.min()), v.max() + cut * (v.max() - v.min())
            grid = np.linspace(lo, hi, 256)
            dens = _kde_curve(v, grid)
            if dens is not None and dens.max() > 0:
                w = dens / dens.max() * half
                if scale == "count":
                    w *= len(v) / n_max
                elif scale == "area":
                    w *= dens.sum() / max(1e-12, dens.max() * len(dens))
                base = i + side * gap
                ax.fill_betweenx(grid, base, base + side * w, facecolor=colors[h],
                                 alpha=alpha, lw=0.6 * k if edge else 0,
                                 edgecolor=dark_colors[h] if edge else "none", zorder=1)
            if points:
                sub = v if len(v) <= max_points else rng.choice(v, max_points, replace=False)
                ax.scatter(i + side * (gap + 0.05) + side * rng.uniform(0, 0.08, len(sub)), sub,
                           s=2.5, color=dark_colors[h], alpha=0.4, lw=0, zorder=2, rasterized=True)
            # median + quartile bars inside the half, centred ~40% of the way out
            c = i + side * (gap + half * 0.375)
            ax.hlines([q1, q3], c - half * 0.2, c + half * 0.2, color=dark_colors[h],
                      lw=0.9 * k, alpha=0.75, zorder=3)
            ax.hlines(med, c - half * 0.3, c + half * 0.3, color=dark_colors[h],
                      lw=2.0 * k, zorder=4, capstyle="butt")
            if median_labels:
                from matplotlib import patheffects as pe
                halo = [pe.withStroke(linewidth=2.4, foreground="white")]
                lab = dict(color=dark_colors[h], fontsize=fs_lab, fontweight="bold", zorder=6)
                if label_pos == "outside":          # just beyond the violin's edge
                    ax.annotate(f"{med:.0f}", xy=(i + side * (gap + half * 0.95), med),
                                xytext=(side * 1.5, 0), textcoords="offset points",
                                va="center", ha="left" if side > 0 else "right", **lab)
                elif label_pos == "inside":         # inside the half, beside its median bar
                    ax.annotate(f"{med:.0f}", xy=(c + side * half * 0.3, med),
                                xytext=(side * 2.0, 0), textcoords="offset points",
                                va="center", ha="left" if side > 0 else "right",
                                path_effects=halo, **lab)
                elif label_pos == "below":          # a row of medians under the violins
                    ax.annotate(f"{med:.0f}", xy=(c, 0), xycoords=("data", "axes fraction"),
                                xytext=(0, 3), textcoords="offset points", va="bottom",
                                ha="center", **lab)
                else:                               # "above": centred on top of the bar
                    ax.annotate(f"{med:.0f}", xy=(c, med), xytext=(0, 1.8),
                                textcoords="offset points", va="bottom", ha="center",
                                path_effects=halo, **lab)
        both = np.concatenate([v for v in (vals[(g, h)] for h in hue_order) if len(v)]) \
            if any(len(vals[(g, h)]) for h in hue_order) else np.array([])
        rec["top"] = float(both.max()) if len(both) else np.nan
        a, b_ = (vals[(g, hue_order[0])], vals[(g, hue_order[1])])
        if link and len(a) and len(b_):
            ax.plot([i - half * 0.375, i + half * 0.375], [np.median(a), np.median(b_)],
                    color=INK["muted"], lw=0.6 * k, zorder=5)
        if test and len(a) >= min_n and len(b_) >= min_n:
            try:
                if test == "ks":
                    rec["test"] = "two-sample Kolmogorov–Smirnov"
                    rec["p"] = float(st.ks_2samp(a, b_).pvalue)
                elif (test == "wilcoxon") or (test == "auto" and paired):
                    m = min(len(a), len(b_))
                    rec["test"] = "Wilcoxon signed-rank"
                    rec["p"] = float(st.wilcoxon(a[:m], b_[:m]).pvalue)
                else:
                    rec["test"] = "Mann–Whitney U"
                    rec["p"] = float(st.mannwhitneyu(a, b_, alternative="two-sided").pvalue)
                rec["delta"] = float(np.median(b_) - np.median(a))
            except ValueError:
                pass
        out[g] = rec

    ps = [r["p"] for r in out.values() if "p" in r]
    if ps:
        adj = bh_fdr(ps)
        for r, q in zip([r for r in out.values() if "p" in r], adj):
            r["p_adj"] = float(q)

    ax.set_xticks(range(len(order)), order)
    room = 0.32 if (median_labels and label_pos == "outside") else 0.04
    ax.set_xlim(-half - gap - room, len(order) - 1 + half + gap + room)
    categorical_axis(ax, "x")
    handles = [mpl.patches.Patch(facecolor=colors[h], alpha=alpha, edgecolor="none", label=str(h))
               for h in hue_order]
    if legend:   # above the axes, so it never sits on the data
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 1.0),
                  ncols=2, borderaxespad=0)
    return {"groups": out, "hue_order": hue_order, "handles": handles}


def split_violin_grid(data, x, y, hue, order=None, journal: str | None = None,
                      width: str | float = "double", panel_mm: float = 22,
                      height_mm: float = 55, ylabel: str = "", **kw):
    """One narrow panel per category, all sharing a y-axis (subject-by-subject view).

    Use when there are many categories, or when each panel needs its own label
    under it (patient IDs, cell lines). Returns (fig, axes, results).
    Extra keyword arguments are passed to split_violin.
    """
    order = list(order or data[x].dropna().unique())
    n = len(order)
    total = (JOURNALS[journal or _STATE["journal"]][width] if isinstance(width, str)
             else float(width))
    if total / n < panel_mm * 0.8:
        print(f"[pubfig] split_violin_grid: {n} panels in {total:.0f} mm gives "
              f"{total / n:.0f} mm each: consider split_violin() on one axes instead.")
    fig, axes = plt.subplots(1, n, figsize=(total * MM, height_mm * MM), sharey=True,
                             gridspec_kw=dict(wspace=0.08))
    axes = np.atleast_1d(axes)
    results, handles = {}, []
    for ax, g in zip(axes, order):
        tidy_ticks(ax)
        kw.setdefault("width", 0.96)
        res = split_violin(ax, data[data[x] == g], x, y, hue, order=[g], legend=False, **kw)
        results[g] = res["groups"][g]
        handles = res["handles"]
        ax.set_xticks([])
        ax.set_xlabel(str(g), fontweight="bold")
        ax.spines["left"].set_visible(ax is axes[0])
        ax.tick_params(left=ax is axes[0], labelleft=ax is axes[0])
    axes[0].set_ylabel(ylabel or y)
    fig.legend(handles=handles, loc="outside upper right", ncols=len(handles), borderaxespad=0)
    return fig, axes, {"groups": results, "hue_order": res["hue_order"], "handles": handles}


def _balance_rows(groups: dict, rows: int) -> list[list[str]]:
    """Split ordered group names into `rows` runs of roughly equal category count."""
    names = list(groups)
    total = sum(len(groups[g]) for g in names)
    target = total / rows
    out, cur, cur_n = [], [], 0
    for g in names:
        n = len(groups[g])
        if cur and len(out) < rows - 1 and cur_n + n / 2 > target:
            out.append(cur); cur, cur_n = [], 0
        cur.append(g); cur_n += n
    out.append(cur)
    while len(out) < rows:
        out.append([])
    return out


def group_bracket(ax_strip, label: str, italic: bool = True, bold: bool = True):
    """Draw a bracket with a label inside a thin 'strip' axes spanning some panels."""
    c = INK["axis"]
    k = sizes()["stroke"]
    ax_strip.set_xlim(0, 1)
    ax_strip.set_ylim(0, 1)
    ax_strip.plot([0.02, 0.02, 0.98, 0.98], [1.0, 0.72, 0.72, 1.0], color=c, lw=0.6 * k,
                  clip_on=False, solid_capstyle="butt")
    ax_strip.text(0.5, 0.58, label, ha="center", va="top", fontsize=sizes()["legend"],
                  style="italic" if italic else "normal",
                  fontweight="bold" if bold else "normal", color=INK["text"])
    ax_strip.set_axis_off()
    ax_strip.set_gid("bracket")


def split_violin_panels(data, x, y, hue, groups=None, rows: int = 1, order=None,
                        journal: str | None = None, width: str | float = "double",
                        row_height_mm: float = 52, ylabel: str = "", ygrid: bool = False,
                        gap: float = 0.14, tests: bool = True, stars: bool = True,
                        panel_space: float = 0.02,
                        stat_note: str | None = None, hue_order=None, colors=None,
                        median_labels: bool = True, **kw):
    """One split-violin panel per category, categories bracketed by a parent group.

    The layout of large panel figures: cell lines bracketed by cancer type,
    patients by cohort, samples by batch: wrapped over `rows` rows, sharing
    one y-axis, with a single legend and a grey note above the figure.

    groups  {group name: [categories]} in display order, or the name of a
            column in `data` holding the group of each row. None = one group.
    Extra keyword arguments go to split_violin (scale, alpha, points, paired…).

    Returns (fig, axes, results): axes in display order, results as from
    split_violin with all categories pooled.
    """
    present = list(order or data[x].dropna().unique())
    if groups is None:
        groups = {"": present}
    elif isinstance(groups, str):
        gmap, seen = {}, {}
        for g, c in data[[groups, x]].dropna().drop_duplicates().itertuples(index=False):
            seen.setdefault(g, []).append(c)
        gmap = {g: [c for c in cs if c in present] for g, cs in seen.items()}
        groups = {g: cs for g, cs in gmap.items() if cs}
    else:
        groups = {g: [c for c in cs if c in present] for g, cs in groups.items()}
        groups = {g: cs for g, cs in groups.items() if cs}

    layout = _balance_rows(groups, rows)
    n_max = max(sum(len(groups[g]) for g in row) for row in layout)
    total_w = (JOURNALS[journal or _STATE["journal"]][width] if isinstance(width, str)
               else float(width))
    fig = plt.figure(figsize=(total_w * MM, rows * row_height_mm * MM))
    outer = fig.add_gridspec(rows, 1, hspace=0.32)

    axes, strips, results, handles = [], [], {}, []
    for r, row_groups in enumerate(layout):
        if not row_groups:
            continue
        row_first = True
        ratios, spans = [], []
        for i, g in enumerate(row_groups):
            start = len(ratios)
            ratios += [1.0] * len(groups[g])
            spans.append((g, start, len(ratios)))
            if i < len(row_groups) - 1:
                ratios.append(gap)
        pad = n_max - sum(1 for w in ratios if w == 1.0)
        if pad > 0:                      # keep panel width equal across rows
            ratios.append(pad + gap * (rows - 1))
        inner = outer[r].subgridspec(2, len(ratios), width_ratios=ratios,
                                     height_ratios=[1, 0.16], wspace=panel_space, hspace=0.02)
        for g, c0, c1 in spans:
            for j, cat in enumerate(groups[g]):
                ax = fig.add_subplot(inner[0, c0 + j], sharey=axes[0] if axes else None)
                tidy_ticks(ax)
                res = split_violin(ax, data[data[x] == cat], x, y, hue, order=[cat],
                                   hue_order=hue_order, colors=colors, legend=False,
                                   median_labels=median_labels, **{"width": 0.96, **kw})
                results[cat] = res["groups"][cat]
                hue_order, handles = res["hue_order"], res["handles"]
                ax.set_xticks([])
                ax.set_xlabel(str(cat), fontweight="bold")
                ax.spines["left"].set_visible(row_first)   # y-axis once per row
                ax.tick_params(left=row_first, labelleft=row_first)
                if ygrid:
                    ax.grid(axis="y", zorder=0)
                if row_first:
                    ax.set_ylabel(ylabel or y)
                    row_first = False
                axes.append(ax)
            strip = fig.add_subplot(inner[1, c0:c1])
            if g:
                group_bracket(strip, str(g))
            else:
                strip.set_axis_off()
            strips.append(strip)
        if pad > 0:                      # invisible filler so panels stay equal width
            fig.add_subplot(inner[0, -1]).set_axis_off()

    out = {"groups": results, "hue_order": hue_order, "handles": handles}
    if tests:
        for ax, cat in zip(axes, results):
            annotate_tests(ax, {cat: results[cat]}, stars=stars)
    fig.legend(handles=handles, loc="outside upper right", ncols=len(handles), borderaxespad=0)
    if stat_note:                     # left, so it never sits under the legend
        fig_stat_title(fig, stat_note, loc="left")
    per_panel = total_w / max(1, n_max)
    if per_panel < 18:
        print(f"[pubfig] split_violin_panels: {n_max} panels in {total_w:.0f} mm gives "
              f"{per_panel:.0f} mm each: use more rows or a wider canvas (width=…).")
    return fig, axes, out


def annotate_tests(ax, results, y=None, adjusted: bool = True, stars: bool = False,
                   only_significant: bool = False, alpha: float = 0.05, pad: float = 0.03):
    """Write the per-category P values from split_violin just above each violin.

    y: fixed height for every label (a common baseline). Default places each
    label over its own violin and grows the y-limit if needed.
    """
    groups = results["groups"] if "groups" in results else results
    fs = sizes()["annot"]
    lo, hi = ax.get_ylim()
    span = hi - lo
    need = hi
    for i, (g, rec) in enumerate(groups.items()):
        key = "p_adj" if adjusted and "p_adj" in rec else "p"
        if key not in rec:
            continue
        p = rec[key]
        if only_significant and p >= alpha:
            continue
        if stars:
            txt = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < alpha else "n.s."
        else:
            txt = fmt_p(p).replace("$P$ = ", "").replace("$P$ ", "")
        yy = y if y is not None else (rec.get("top", hi) + pad * span)
        need = max(need, yy + 0.06 * span)
        ax.text(i, yy, txt, ha="center", va="bottom", fontsize=fs, color=INK["muted"])
    if y is None and need > hi:
        ax.set_ylim(lo, need)


def volcano(ax, lfc, padj, names=None, lfc_thr: float = 1.0, p_thr: float = 0.05,
            top: int = 8, colors=None, point_size: float = 4):
    """Volcano plot from log2 fold-changes and adjusted p-values. Returns counts."""
    colors = colors or PALETTES["updown"]
    lfc = np.asarray(lfc, float)
    padj = np.clip(np.asarray(padj, float), 1e-300, 1)
    y = -np.log10(padj)
    up = (lfc >= lfc_thr) & (padj < p_thr)
    dn = (lfc <= -lfc_thr) & (padj < p_thr)
    ns = ~(up | dn)
    ax.scatter(lfc[ns], y[ns], s=point_size * 0.7, color=colors["ns"], lw=0, zorder=1, rasterized=True)
    ax.scatter(lfc[dn], y[dn], s=point_size, color=colors["down"], lw=0, zorder=2, rasterized=True)
    ax.scatter(lfc[up], y[up], s=point_size, color=colors["up"], lw=0, zorder=2, rasterized=True)
    refline(ax, y=-np.log10(p_thr)); refline(ax, x=lfc_thr); refline(ax, x=-lfc_thr)
    lim = np.nanmax(np.abs(lfc)) * 1.3      # headroom for labels
    ax.set_xlim(-lim, lim)
    ax.set_ylim(0, y.max() * 1.3)
    ax.xaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, integer=True, symmetric=True))
    ax.set_xlabel("log$_2$ fold change")
    ax.set_ylabel("$-$log$_{10}$ adjusted $P$")
    fs = sizes()["annot"]
    ax.text(0.02, 0.98, f"{dn.sum()} down", transform=ax.transAxes, ha="left", va="top",
            color=colors["down"], fontsize=fs)
    ax.text(0.98, 0.98, f"{up.sum()} up", transform=ax.transAxes, ha="right", va="top",
            color=colors["up"], fontsize=fs)
    if names is not None and top:
        names = np.asarray(names)
        sig = np.where(up | dn)[0]
        score = y[sig] * np.abs(lfc[sig])
        pick = sig[np.argsort(score)[::-1][:top]]
        label_points(ax, lfc[pick], y[pick], names[pick])
    return dict(up=int(up.sum()), down=int(dn.sum()), ns=int(ns.sum()))



def density_scatter(ax, x, y, fit: str | None = "ols", marginals: bool = True,
                    log: str | None = None, cmap: str = "ice_navy", s: float = 3,
                    bins: int = 80, fit_color: str = "#D1782A",
                    marginal_size: float = 0.16):
    """Thousands of points: colour = local density, OLS fit with 95% confidence
    band (filled) and 95% prediction band (dashed = where individual points
    fall), plus marginal histograms showing each variable's range.

    log: None | "x" | "y" | "both": for skewed data. Density, histograms and
    the fit are computed in log10 space and drawn on log axes.
    Returns a dict (n, r, p, rho, slope, intercept); r and the fit refer to the
    (log-)transformed values when log is set.
    """
    from scipy import stats as st
    x = np.asarray(x, float); y = np.asarray(y, float)
    lx = log in ("x", "both"); ly = log in ("y", "both")
    ok = np.isfinite(x) & np.isfinite(y)
    if lx:
        ok &= x > 0
    if ly:
        ok &= y > 0
    dropped = int((~ok).sum())
    x, y = x[ok], y[ok]
    u = np.log10(x) if lx else x          # working (possibly log) coordinates
    v = np.log10(y) if ly else y
    H, ue, ve = np.histogram2d(u, v, bins=bins)
    iu = np.clip(np.searchsorted(ue, u) - 1, 0, bins - 1)
    iv = np.clip(np.searchsorted(ve, v) - 1, 0, bins - 1)
    dens = H[iu, iv]
    order = np.argsort(dens)
    ax.scatter(x[order], y[order], c=dens[order], cmap=cmap, s=s, lw=0,
               rasterized=True, zorder=1, vmin=0, vmax=np.percentile(dens, 99))
    if lx:
        ax.set_xscale("log")
    if ly:
        ax.set_yscale("log")
    r, p = st.pearsonr(u, v)
    rho, _ = st.spearmanr(x, y)
    out = dict(n=len(x), r=r, p=p, rho=rho, dropped=dropped)
    if dropped:
        print(f"[pubfig] density_scatter: dropped {dropped} non-finite/non-positive points")
    if fit == "ols":
        n = len(u)
        b, a = np.polyfit(u, v, 1)
        uu = np.linspace(np.percentile(u, 0.5), np.percentile(u, 99.5), 200)
        vhat = a + b * uu
        s_err = np.sqrt(((v - (a + b * u)) ** 2).sum() / (n - 2))
        suu = ((u - u.mean()) ** 2).sum()
        t = st.t.ppf(0.975, n - 2)
        ci = t * s_err * np.sqrt(1 / n + (uu - u.mean()) ** 2 / suu)
        pi = t * s_err * np.sqrt(1 + 1 / n + (uu - u.mean()) ** 2 / suu)
        bx = (lambda q: 10 ** q) if lx else (lambda q: q)
        by = (lambda q: 10 ** q) if ly else (lambda q: q)
        ax.fill_between(bx(uu), by(vhat - ci), by(vhat + ci), color=fit_color, alpha=0.35,
                        lw=0, zorder=2)
        ax.plot(bx(uu), by(vhat), color=fit_color, lw=1.2, zorder=3)
        for sign in (-1, 1):
            ax.plot(bx(uu), by(vhat + sign * pi), color=fit_color, lw=0.7, ls=(0, (3, 2)), zorder=3)
        out.update(slope=b, intercept=a)
    if marginals:
        top = ax.inset_axes([0, 1.03, 1, marginal_size], sharex=ax)
        right = ax.inset_axes([1.03, 0, marginal_size, 1], sharey=ax)
        xb = np.logspace(u.min(), u.max(), 60) if lx else 60
        yb = np.logspace(v.min(), v.max(), 60) if ly else 60
        top.hist(x, bins=xb, color=INK["ref"], lw=0)
        right.hist(y, bins=yb, color=INK["ref"], lw=0, orientation="horizontal")
        for m in (top, right):
            m.set_gid("marginal")
            m.set_axis_off()
        ax._pf_top = 1.03 + marginal_size
    return out


def venn(ax, sets: dict, colors: dict | None = None, universe: int | None = None):
    """Two- or three-set Venn diagram with counts computed from the actual sets.

    Circles are equal-sized (not area-proportional) so every region is legible;
    say so in the caption. For 4+ sets use an UpSet plot instead.
    Returns {region_key: count} where region_key is e.g. 'A&B' or 'A only'.
    """
    names = list(sets)
    k = len(names)
    if k not in (2, 3):
        raise ValueError("venn() supports 2 or 3 sets; use an UpSet plot for more.")
    S = [set(sets[n]) for n in names]
    colors = colors or dict(zip(names, PALETTES["navy_amber"][:k]))
    r = 1.0
    centers = {2: [(-0.55, 0.0), (0.55, 0.0)],
               3: [(-0.58, 0.34), (0.58, 0.34), (0.0, -0.66)]}[k]
    for (cx, cy), n in zip(centers, names):
        ax.add_patch(mpl.patches.Circle((cx, cy), r, facecolor=colors[n], alpha=0.16, lw=0))
        ax.add_patch(mpl.patches.Circle((cx, cy), r, facecolor="none", edgecolor=colors[n],
                                        lw=1.2 * sizes()["stroke"]))
    # region centroids from a membership grid
    g = np.linspace(-2.2, 2.2, 441)
    GX, GY = np.meshgrid(g, g)
    inside = [((GX - cx) ** 2 + (GY - cy) ** 2) <= r ** 2 for cx, cy in centers]
    counts = {}
    fs = sizes()["label"]
    for combo in itertools.product([0, 1], repeat=k):
        if not any(combo):
            continue
        mask = np.ones_like(GX, bool)
        for m, c in zip(inside, combo):
            mask &= m if c else ~m
        members = set.intersection(*[S[i] for i in range(k) if combo[i]])
        for i in range(k):
            if not combo[i]:
                members -= S[i]
        key = "&".join(names[i] for i in range(k) if combo[i])
        key = key if sum(combo) > 1 else f"{key} only"
        counts[key] = len(members)
        cx, cy = GX[mask].mean(), GY[mask].mean()
        if sum(combo) == 1:          # push single-set labels outward a little
            cx, cy = cx * 1.08, cy * 1.08
        ax.text(cx, cy, f"{len(members):,}", ha="center", va="center", fontsize=fs,
                color=INK["text"], fontweight="bold" if sum(combo) == k else "normal")
    # set names outside the circles
    fsn = sizes()["legend"]
    for (cx, cy), n, s_ in zip(centers, names, S):
        d = np.array([cx, cy]); d = d / (np.linalg.norm(d) or 1)
        px, py = np.array([cx, cy]) + d * (r + 0.12)
        ha = "right" if d[0] < -0.3 else "left" if d[0] > 0.3 else "center"
        fx = {"left": 0.0, "right": 1.0, "center": 0.5}[ha]
        n_txt = f"$n$ = {len(s_):,}"
        kw_name = dict(fontsize=fsn, color=colors[n], fontweight="bold", ha=ha)
        kw_n = dict(fontsize=fsn, color=INK["muted"], ha=ha)
        # plain Text objects (not annotations) so constrained_layout reserves room
        up = mpl.transforms.offset_copy(ax.transData, fig=ax.figure, y=0.5, units="points")
        dn = mpl.transforms.offset_copy(ax.transData, fig=ax.figure, y=-0.5, units="points")
        if d[1] >= -0.3:   # beside/above the circle: name above, count below the anchor
            ax.text(px, py, n, va="bottom", transform=up, **kw_name)
            ax.text(px, py, n_txt, va="top", transform=dn, **kw_n)
        else:              # below the circle: name, then count underneath
            ax.text(px, py, n, va="top", **kw_name)
            ax.annotate(n_txt, xy=(fx, 0), xycoords=ax.texts[-1], xytext=(0, -1),
                        textcoords="offset points", va="top", annotation_clip=False, **kw_n)
    ax._pf_venn_r = r
    ax.set_aspect("equal")
    # limits hug the circles; set-name labels may sit outside the axes box
    # (they are still included in the layout, so nothing is clipped)
    ax.set_xlim(-1.7, 1.7) if k == 3 else ax.set_xlim(-1.65, 1.65)
    ax.set_ylim(-1.72, 1.4) if k == 3 else ax.set_ylim(-1.05, 1.05)
    ax.set_axis_off()
    ax.set_gid("image_axes")
    return counts


def overlap_p(a, b, universe: int) -> float:
    """Hypergeometric P that sets a and b overlap at least as much as observed."""
    from scipy import stats as st
    a, b = set(a), set(b)
    return float(st.hypergeom.sf(len(a & b) - 1, universe, len(a), len(b)))


def fit_4pl(x, y, n_draws: int = 500, seed: int = 0):
    """Four-parameter logistic (Hill) fit for dose–response data.

    Returns dict(bottom, top, ec50, hill, ec50_ci, predict) where
    predict(xx) -> (fit, lo, hi) gives the curve and a 95% confidence band
    from parameter-covariance sampling.
    """
    from scipy.optimize import curve_fit
    x = np.asarray(x, float); y = np.asarray(y, float)
    ok = (x > 0) & np.isfinite(y)
    lx, y = np.log10(x[ok]), y[ok]

    def f(lx_, bottom, top, logec50, hill):
        return bottom + (top - bottom) / (1 + 10 ** ((logec50 - lx_) * hill))

    p0 = [y.min(), y.max(), np.median(lx), 1.0]
    popt, pcov = curve_fit(f, lx, y, p0=p0, maxfev=20000)
    rng = np.random.default_rng(seed)
    draws = rng.multivariate_normal(popt, pcov, n_draws, check_valid="ignore")
    se = np.sqrt(np.diag(pcov))
    from scipy import stats as st
    t = st.t.ppf(0.975, max(len(y) - 4, 1))

    def predict(xx):
        lxx = np.log10(np.asarray(xx, float))
        curves = np.array([f(lxx, *d) for d in draws])
        return f(lxx, *popt), np.nanpercentile(curves, 2.5, 0), np.nanpercentile(curves, 97.5, 0)

    return dict(bottom=popt[0], top=popt[1], ec50=10 ** popt[2], hill=popt[3],
                ec50_ci=(10 ** (popt[2] - t * se[2]), 10 ** (popt[2] + t * se[2])),
                predict=predict)


# --------------------------------------------------------------------------
# Sequence logos (DNA, RNA, protein / peptide motifs)
# --------------------------------------------------------------------------
ALPHABETS = {
    "dna": "ACGT",
    "rna": "ACGU",
    "protein": "ACDEFGHIKLMNPQRSTVWY",
}

# Nucleotides: the colours readers expect (A green, C blue, G amber, T/U red),
# slightly muted. Protein: WebLogo-style chemistry classes.
LOGO_COLORS = {
    "dna": {"A": "#2E8B57", "C": "#2F5D8A", "G": "#E0A030", "T": "#C0392B"},
    "rna": {"A": "#2E8B57", "C": "#2F5D8A", "G": "#E0A030", "U": "#C0392B"},
    "protein": {
        **dict.fromkeys("GSTYC", "#2E8B57"),     # polar
        **dict.fromkeys("NQ", "#8A5AA8"),        # amide
        **dict.fromkeys("KRH", "#2F5D8A"),       # basic
        **dict.fromkeys("DE", "#C0392B"),        # acidic
        **dict.fromkeys("AVLIPWFM", "#2B2B2B"),  # hydrophobic
    },
}
LOGO_OTHER = "#C4CAD3"   # pooled letters too small to draw legibly

LOGO_CLASSES = {  # legend text for the protein scheme
    "Polar (G S T Y C)": "#2E8B57", "Amide (N Q)": "#8A5AA8", "Basic (K R H)": "#2F5D8A",
    "Acidic (D E)": "#C0392B", "Hydrophobic (A V L I P W F M)": "#2B2B2B",
}


def guess_alphabet(letters) -> str:
    s = set("".join(letters).upper()) - set("-.*X N")
    if s <= set("ACGT"):
        return "dna"
    if s <= set("ACGU"):
        return "rna"
    return "protein"


def read_sequences(path) -> list[str]:
    """Aligned sequences from FASTA or plain text (one sequence per line)."""
    text = Path(path).read_text().splitlines()
    if any(l.startswith(">") for l in text):
        seqs, cur = [], []
        for l in text:
            if l.startswith(">"):
                if cur:
                    seqs.append("".join(cur)); cur = []
            elif l.strip():
                cur.append(l.strip())
        if cur:
            seqs.append("".join(cur))
    else:
        seqs = [l.strip() for l in text if l.strip() and not l.startswith("#")]
    return [s.upper() for s in seqs]


def read_jaspar(path) -> dict:
    """JASPAR count matrices → {"MA0139.1 CTCF": DataFrame(position × letter)}."""
    import pandas as pd
    import re
    out, name, rows = {}, None, {}
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            if name and rows:
                out[name] = pd.DataFrame(rows)
            name, rows = line[1:].strip(), {}
            continue
        m = re.match(r"^([ACGTU])\s*\[?\s*([\d.\s]+?)\s*\]?$", line)
        if m:
            rows[m.group(1)] = [float(v) for v in m.group(2).split()]
    if name and rows:
        out[name] = pd.DataFrame(rows)
    for k, v in out.items():
        v.index = range(1, len(v) + 1)
    return out


def read_meme(path) -> dict:
    """MEME-format letter-probability matrices → {name: DataFrame}, with nsites."""
    import pandas as pd
    import re
    lines = Path(path).read_text().splitlines()
    alph = "ACGT"
    for l in lines:
        if l.startswith("ALPHABET="):
            alph = "".join(c for c in l.split("=", 1)[1].strip() if c.isalpha())
    out, i = {}, 0
    while i < len(lines):
        l = lines[i]
        if l.startswith("MOTIF"):
            name = " ".join(l.split()[1:])
            while i < len(lines) and "letter-probability matrix" not in lines[i]:
                i += 1
            if i >= len(lines):
                break
            m = re.search(r"nsites=\s*([\d.]+)", lines[i])
            w = int(re.search(r"w=\s*(\d+)", lines[i]).group(1))
            nsites = float(m.group(1)) if m else 20.0
            rows = [[float(v) for v in lines[i + 1 + k].split()] for k in range(w)]
            df = pd.DataFrame(rows, columns=list(alph)[:len(rows[0])], index=range(1, w + 1))
            out[name] = df * nsites        # back to (approximate) counts
            i += w
        i += 1
    return out


def reverse_complement(counts):
    """Reverse complement of a DNA/RNA position matrix."""
    comp = {"A": "T", "C": "G", "G": "C", "T": "A", "U": "A"}
    if "U" in counts.columns:
        comp = {"A": "U", "C": "G", "G": "C", "U": "A"}
    rc = counts.iloc[::-1].rename(columns=comp)[list(counts.columns)]
    rc.index = range(1, len(rc) + 1)
    return rc


def _hobohm_weights(seqs, threshold: float) -> np.ndarray:
    """Hobohm-1 clustering: sequences ≥ threshold identical share one unit of weight."""
    reps, members = [], []
    for i, s in enumerate(seqs):
        for r, rep in enumerate(reps):
            same = sum(a == b for a, b in zip(s, seqs[rep])) / max(1, len(s))
            if same >= threshold:
                members[r].append(i)
                break
        else:
            reps.append(i); members.append([i])
    w = np.zeros(len(seqs))
    for m in members:
        w[m] = 1.0 / len(m)
    return w


def sequence_counts(seqs, alphabet: str | None = None, weighting: str | None = None,
                    threshold: float | None = None):
    """Position × letter (weighted) counts from aligned, equal-length sequences.

    weighting: None, or "hobohm" (Seq2Logo's clustering; threshold defaults to
    0.63 for proteins, 0.8 for nucleotides) so redundant sequences don't
    dominate. Gaps and unknown letters are ignored at that position.
    """
    import pandas as pd
    seqs = [s.upper().replace("T", "U") if alphabet == "rna" else s.upper() for s in seqs]
    lengths = {len(s) for s in seqs}
    if len(lengths) != 1:
        raise ValueError(f"sequences must be aligned to one length; found lengths {sorted(lengths)}")
    alphabet = alphabet or guess_alphabet(seqs)
    letters = ALPHABETS[alphabet]
    if weighting == "hobohm":
        w = _hobohm_weights(seqs, threshold or (0.63 if alphabet == "protein" else 0.8))
    else:
        w = np.ones(len(seqs))
    L = lengths.pop()
    C = np.zeros((L, len(letters)))
    idx = {c: j for j, c in enumerate(letters)}
    for s, wt in zip(seqs, w):
        for i, c in enumerate(s):
            j = idx.get(c)
            if j is not None:
                C[i, j] += wt
    return pd.DataFrame(C, columns=list(letters), index=range(1, L + 1))


def logo_heights(counts, kind: str = "bits", background=None, pseudocount: float = 0.0,
                 small_sample: bool = True):
    """Letter heights for a logo.

    kind "bits"         Shannon information: height = p · (log2 A − H − eₙ),
                        eₙ the small-sample correction (Schneider et al. 1986).
         "probability"  height = p (each column sums to 1).
         "kl"           two-sided relative entropy vs the background:
                        height = p · log2(p / q); enriched letters above zero,
                        depleted below (Seq2Logo's Kullback–Leibler logo).
    background  {letter: q}; uniform if None. Use genomic / proteome
                frequencies when they are known.
    pseudocount weight β: p = (c + β·q) / (n + β), which stabilises columns
                with few observations (Seq2Logo uses a BLOSUM prior instead).
    Returns (heights DataFrame, info Series of column totals in bits).
    """
    import pandas as pd
    C = counts.astype(float)
    letters = list(C.columns)
    A = len(letters)
    q = (np.array([background.get(l, 0) for l in letters], float) if background
         else np.full(A, 1.0 / A))
    q = q / q.sum()
    n = C.sum(axis=1).to_numpy()[:, None]
    P = (C.to_numpy() + pseudocount * q) / np.maximum(n + pseudocount, 1e-12)
    with np.errstate(divide="ignore", invalid="ignore"):
        logp = np.where(P > 0, np.log2(P), 0.0)
    if kind == "probability":
        H = P
        info = pd.Series(np.ones(len(C)), index=C.index)
    elif kind == "bits":
        ent = -(P * logp).sum(axis=1)
        en = ((A - 1) / (2 * np.log(2) * np.maximum(n[:, 0], 1))) if small_sample else 0.0
        ic = np.clip(np.log2(A) - ent - en, 0, None)
        H = P * ic[:, None]
        info = pd.Series(ic, index=C.index)
    elif kind == "kl":
        with np.errstate(divide="ignore", invalid="ignore"):
            H = np.where(P > 0, P * (logp - np.log2(q)[None, :]), 0.0)
        info = pd.Series(H.sum(axis=1), index=C.index)
    else:
        raise ValueError("kind must be 'bits', 'probability' or 'kl'")
    return pd.DataFrame(H, columns=letters, index=C.index), info


_GLYPHS: dict = {}


def _glyph(letter: str):
    """Unit-box outline of a bold letter (cached)."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    key = (letter, mpl.rcParams["font.sans-serif"][0])
    if key not in _GLYPHS:
        fam = resolve_font(SANS_STACK)
        tp = TextPath((0, 0), letter, size=1, prop=FontProperties(family=fam, weight="bold"))
        v = tp.vertices.copy()
        # measure the ink only: CLOSEPOLY/STOP carry a dummy (0, 0) vertex that
        # would add the font's side bearing and shift the letter off-centre
        codes = tp.codes
        ink = v if codes is None else v[(codes != mpl.path.Path.CLOSEPOLY) &
                                        (codes != mpl.path.Path.STOP)]
        x0, y0 = ink.min(0); x1, y1 = ink.max(0)
        v[:, 0] = (v[:, 0] - x0) / (x1 - x0)
        v[:, 1] = (v[:, 1] - y0) / (y1 - y0)
        path = mpl.path.Path(v, tp.codes)
        path._pf_aspect = (x1 - x0) / (y1 - y0)       # natural width / height
        _GLYPHS[key] = path
    return _GLYPHS[key]


NARROW_ASPECT = 0.4      # letters thinner than this (e.g. I) keep a thin, centred stroke
NARROW_WIDTH = 0.32      # …drawn at this fraction of the column width


def draw_logo(ax, heights, colors=None, alphabet: str | None = None, width: float = 0.88,
              kind: str = "bits", highlight=None, ylabel: str | None = None,
              xticks_every: int = 1, start: int | None = None,
              small: str = "glyph", glyph_min="auto", gap: float = 0.006,
              min_height: float = 0.0, ymax="data", pad=None):
    """Stacked-letter logo from a heights table (position × letter).

    Letters are vector outlines scaled to their exact heights, stacked with the
    tallest on top (for negative heights, the most depleted at the bottom).

    small      how to draw letters shorter than `glyph_min` (too short for
               their outline to stay readable):
               "glyph" (default) keep the squashed letter: thin letters such
                       as I stay a thin centred stroke, and a gap separates
                       every stacked letter so neighbours never fuse;
               "bar"   a flat bar in the letter's colour, at its true height;
               "pool"  merge them into one grey "other" block at the base.
    glyph_min  "auto" (default): an outline is kept if it will print at least
               4 pt tall (and not flatter than 8% of its width): below that
               its shape is gone. Or a fraction of the visible y-range.
    gap        empty space between stacked letters, as a fraction of the
               y-range, so neighbours never touch: even two black residues.
    ymax       "data" (default): the axis ends just above the tallest column;
               "max": the theoretical maximum (2 bits DNA/RNA, 4.32 protein)
               for comparing logos across datasets; or a number.
    min_height kept for compatibility: > 0 pools letters below that fraction.
    highlight  positions to shade lightly (e.g. peptide anchor residues).
    """
    alphabet = alphabet or guess_alphabet(heights.columns)
    colors = colors or LOGO_COLORS.get(alphabet, {})
    pos = list(heights.index)
    if min_height and small != "pool":
        small, glyph_min = "pool", min_height

    # ---- axis range first: thresholds and gaps are relative to what is shown
    up_tot = heights.clip(lower=0).sum(axis=1)
    dn_tot = heights.clip(upper=0).sum(axis=1)
    tallest = float(up_tot.max()) if len(heights) else 0.0
    if kind == "bits":
        theo = np.log2(len(heights.columns))
        if ymax == "max":
            lo, hi = 0.0, theo
        elif ymax == "data":
            lo, hi = 0.0, (min(theo, tallest * 1.06) if tallest > 0 else theo)
        else:
            lo, hi = 0.0, float(ymax)
    elif kind == "probability":
        lo, hi = 0.0, 1.0
    else:
        lo0, hi0 = min(0.0, float(dn_tot.min())), max(0.0, float(up_tot.max()))
        span0 = (hi0 - lo0) or 1.0
        lo, hi = lo0 - 0.05 * span0, hi0 + 0.05 * span0
    span = hi - lo
    if glyph_min == "auto":
        # printed size of one data unit, from the axes' planned position
        fig = ax.figure
        bb = ax.get_position()
        ax_h_pt = bb.height * fig.get_figheight() * 72
        ax_w_pt = bb.width * fig.get_figwidth() * 72
        pt_per_y = ax_h_pt / span
        letter_w_pt = width * ax_w_pt / max(1, len(pos))
        hmin = max(4.0, 0.08 * letter_w_pt) / pt_per_y
    else:
        hmin = float(glyph_min) * span
    g = gap * span

    def put(letter, x, y0, h):
        color = colors.get(letter, INK["text"])
        inner = h - g if h > 2 * g else h * 0.6          # leave the gap, but never vanish
        yb = y0 + (h - inner) / 2
        if h >= hmin or small == "glyph":
            _place_glyph(ax, letter, x, yb, inner, width, color)
        else:   # a bar keeps the letter's footprint: an I bar is thin and central
            w = (width * NARROW_WIDTH if getattr(_glyph(letter), "_pf_aspect", 1) < NARROW_ASPECT
                 else width)
            ax.add_patch(mpl.patches.Rectangle((x - w / 2, yb), w, inner,
                                               facecolor=color, lw=0, zorder=2))

    ax._pf_pooled = False
    for x, (p, row) in enumerate(heights.iterrows(), start=1):
        for sign in (1, -1):
            items = sorted(((abs(h), l) for l, h in row.items() if sign * h > 1e-9))
            if sign < 0:
                items = sorted(items)          # smallest depletion nearest the axis
            base = 0.0
            if small == "pool":
                pooled = sum(h for h, _ in items if h < hmin)
                if pooled > 0:
                    ax._pf_pooled = True
                    y0 = 0.0 if sign > 0 else -pooled
                    ax.add_patch(mpl.patches.Rectangle((x - width / 2, y0), width, pooled,
                                                       facecolor=LOGO_OTHER, lw=0, zorder=1.5))
                    base = pooled
                items = [(h, l) for h, l in items if h >= hmin]
            for h, l in items:
                y0 = base if sign > 0 else -(base + h)
                put(l, x, y0, h)
                base += h

    if highlight:
        for hp in highlight:
            if hp in pos:
                xi = pos.index(hp) + 1
                ax.axvspan(xi - 0.5, xi + 0.5, color="#F1F3F6", lw=0, zorder=0)
    ax.set_xlim(0.5, len(pos) + 0.5)
    labels = [str(p) for p in pos] if start is None else [str(start + i) for i in range(len(pos))]
    ticks = [i + 1 for i in range(len(pos)) if i % xticks_every == 0]
    ax.set_xticks(ticks, [labels[t - 1] for t in ticks])
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_ylim(lo, hi)
    if kind == "bits":
        ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=4, steps=[1, 2, 2.5, 5, 10]))
        ax._pf_logo_top = hi
    elif kind == "probability":
        ax.set_yticks([0, 0.5, 1])
    else:
        refline(ax, y=0, ls="-", color=INK["axis"], lw=0.5)
    ax.set_ylabel(ylabel or {"bits": "Bits", "probability": "Probability",
                             "kl": "Enrichment (bits)"}[kind])
    ax.set_xlabel("Position")
    return ax


def _place_glyph(ax, letter, x, y0, h, width, color, pad=None):
    if h <= 0:
        return
    g = _glyph(letter)
    # stretching a thin letter (I) to the full column turns it into a slab that
    # reads as a box; keep it a thin centred stroke instead
    w = width * NARROW_WIDTH if getattr(g, "_pf_aspect", 1.0) < NARROW_ASPECT else width
    tr = mpl.transforms.Affine2D().scale(w, h).translate(x - w / 2, y0)
    ax.add_patch(mpl.patches.PathPatch(tr.transform_path(g), facecolor=color,
                                       edgecolor="none", lw=0, zorder=2))


def logo(ax, data, kind: str = "bits", alphabet: str | None = None, weighting=None,
         pseudocount: float = 0.0, background=None, highlight=None, colors=None,
         note: bool = True, **kw):
    """One-call logo from sequences (list/FASTA path) or a count matrix.

    Returns dict(counts, heights, info, n, consensus).
    """
    import pandas as pd
    if isinstance(data, (str, Path)) and Path(data).exists():
        data = read_sequences(data)
    if isinstance(data, pd.DataFrame):
        counts = data
        alphabet = alphabet or guess_alphabet(counts.columns)
    else:
        alphabet = alphabet or guess_alphabet(data)
        counts = sequence_counts(data, alphabet, weighting=weighting)
    Hm, info = logo_heights(counts, kind=kind, background=background, pseudocount=pseudocount)
    draw_logo(ax, Hm, colors=colors, alphabet=alphabet, kind=kind, highlight=highlight, **kw)
    n = float(counts.sum(axis=1).max())
    consensus = "".join(counts.idxmax(axis=1))
    if note:
        n_txt = f"{n:,.0f}" if abs(n - round(n)) < 1e-6 else f"{n:,.1f} (weighted)"
        stat_title(ax, f"$n$ = {n_txt} · {info.sum():.1f} bits")
    ymode = kw.get("ymax", "data")
    return dict(counts=counts, heights=Hm, info=info, n=n, consensus=consensus,
                alphabet=alphabet, ymax=ymode, max_bits=float(np.log2(len(counts.columns))))


LOGO_CLASSES_SHORT = {"Polar": "#2E8B57", "Amide": "#8A5AA8", "Basic": "#2F5D8A",
                      "Acidic": "#C0392B", "Hydrophobic": "#2B2B2B"}


def logo_legend(ax, alphabet: str = "protein", orientation: str = "vertical",
                position: str | None = None, compact: bool | None = None,
                fontsize: float | None = None, show_other: bool | None = None,
                ncols: int | None = None):
    """Colour key for logo letters: always outside the data, never on the motif.

    orientation  "vertical" (default; to the right of the axes) or
                 "horizontal" (one row across the figure, wrapping if needed).
    position     vertical: "right" (default). horizontal: "bottom" (default,
                 under the x-axis title) or "top".
    compact      class names only ("Polar", "Basic" …) instead of names with
                 their residues; default True for horizontal keys.
    fontsize     default one point below the tick labels (never below the
                 journal minimum), so the key stays secondary to the motif.
    Horizontal keys are figure-level and layout-aware, so they can't collide
    with axis titles, the stat line or panel letters.
    """
    fig = ax.figure
    horizontal = orientation.startswith("h")
    compact = horizontal if compact is None else compact
    sz = sizes()
    fs = fontsize or max(sz["min"] + 0.5, sz["tick"] - 1)
    if alphabet == "protein":
        items = dict(LOGO_CLASSES_SHORT if compact else LOGO_CLASSES)
    else:
        items = dict(LOGO_COLORS[alphabet])
    if show_other is None:   # only when some panel actually pooled small letters
        show_other = any(getattr(a, "_pf_pooled", False) for a in fig.axes)
    if show_other:
        items["Other" if compact else "Other (pooled small letters)"] = LOGO_OTHER
    handles = [mpl.patches.Patch(facecolor=c, edgecolor="none", label=l) for l, c in items.items()]
    common = dict(handles=handles, fontsize=fs, handlelength=0.9, handleheight=0.9,
                  handletextpad=0.35, borderaxespad=0.2, frameon=False)
    if not horizontal:
        return ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1.0), labelspacing=0.3,
                         ncols=ncols or 1, **common)
    if ncols is None:             # one row if it fits the figure width, else wrap evenly
        from matplotlib.textpath import TextPath
        from matplotlib.font_manager import FontProperties
        prop = FontProperties(family=resolve_font(SANS_STACK), size=fs)
        widths = [TextPath((0, 0), l, prop=prop).get_extents().width + fs * 2.3 for l in items]
        fig_pt = fig.get_figwidth() * 72 * 0.94
        if sum(widths) <= fig_pt:
            ncols = len(items)
        else:
            ncols = int(max(1, fig_pt // max(widths)))
    loc = "outside upper center" if (position or "bottom") == "top" else "outside lower center"
    return fig.legend(loc=loc, ncols=ncols, columnspacing=1.0, **common)


def motif_grid(motifs: dict, ncols: int = 2, kind: str = "bits", journal: str | None = None,
               width: str | float = "double", panel_height_mm: float = 26,
               sharey: bool = True, **kw):
    """Several motifs as a panel grid, each titled: the Signac MotifPlot layout.

    motifs: {name: count DataFrame or list of sequences}. Returns (fig, axes, results).
    """
    names = list(motifs)
    nrows = int(np.ceil(len(names) / ncols))
    total_w = (JOURNALS[journal or _STATE["journal"]][width] if isinstance(width, str)
               else float(width))
    fig, axes = plt.subplots(nrows, ncols, figsize=(total_w * MM, nrows * panel_height_mm * MM),
                             squeeze=False, sharey=sharey)
    results = {}
    for ax, name in zip(axes.flat, names):
        results[name] = logo(ax, motifs[name], kind=kind, note=False, **kw)
        ax.set_title(name, loc="left", fontsize=sizes()["label"])
    for ax in list(axes.flat)[len(names):]:
        ax.set_visible(False)
    if sharey and kind == "bits":        # one scale for all: fit the tallest motif
        tops = [getattr(a, "_pf_logo_top", None) for a in axes.flat if a.get_visible()]
        tops = [t for t in tops if t]
        if tops:
            axes.flat[0].set_ylim(0, max(tops))
    for i, ax in enumerate(axes.flat):
        if not ax.get_visible():
            continue
        r, c = divmod(i, ncols)
        if c > 0 and sharey:
            ax.set_ylabel("")
        if r < nrows - 1 and i + ncols < len(names):
            ax.set_xlabel("")
    return fig, axes, results


# --------------------------------------------------------------------------
# Annotated MS/MS spectra (peptide–spectrum matches)
# --------------------------------------------------------------------------
PROTON = 1.007276467
WATER = 18.010564684
AMMONIA = 17.026549101
CO = 27.994914620
AA_MONO = {  # monoisotopic residue masses (Da)
    "G": 57.021464, "A": 71.037114, "S": 87.032028, "P": 97.052764, "V": 99.068414,
    "T": 101.047679, "C": 103.009185, "L": 113.084064, "I": 113.084064, "N": 114.042927,
    "D": 115.026943, "Q": 128.058578, "K": 128.094963, "E": 129.042593, "M": 131.040485,
    "H": 137.058912, "F": 147.068414, "R": 156.101111, "Y": 163.063329, "W": 186.079313,
    "U": 150.953636, "O": 237.147727,
}
MODS = {  # Unimod monoisotopic deltas; key → (delta, label drawn above the residue)
    "oxidation": (15.994915, "ox"), "ox": (15.994915, "ox"),
    "dioxidation": (31.989829, "diox"),
    "phospho": (79.966331, "ph"), "phosphorylation": (79.966331, "ph"), "ph": (79.966331, "ph"),
    "carbamidomethyl": (57.021464, "cam"), "cam": (57.021464, "cam"),
    "acetyl": (42.010565, "ac"), "ac": (42.010565, "ac"),
    "deamidated": (0.984016, "de"), "deamidation": (0.984016, "de"),
    "methyl": (14.015650, "me"), "dimethyl": (28.031300, "me2"), "trimethyl": (42.046950, "me3"),
    "gg": (114.042927, "GG"), "glygly": (114.042927, "GG"),
    "tmt": (229.162932, "TMT"), "tmtpro": (304.207146, "TMT"),
    "nitro": (44.985078, "nitro"), "citrullination": (0.984016, "cit"),
    "pyro-glu": (-17.026549, "pyro"), "gln->pyro-glu": (-17.026549, "pyro"),
    "glu->pyro-glu": (-18.010565, "pyro"),
    # glycans (single residues and common compositions)
    "hexnac": (203.079373, "HexNAc"), "glcnac": (203.079373, "GlcNAc"),
    "o-glcnac": (203.079373, "GlcNAc"), "hex": (162.052824, "Hex"),
    "dhex": (146.057909, "Fuc"), "fuc": (146.057909, "Fuc"),
    "neuac": (291.095417, "NeuAc"), "neugc": (307.090331, "NeuGc"),
    "hexnac(1)hex(1)": (365.132196, "HexHexNAc"),
    "glyco": (203.079373, "glyco"),
}
UNIMOD = {35: "oxidation", 21: "phospho", 4: "carbamidomethyl", 1: "acetyl", 7: "deamidated",
          34: "methyl", 36: "dimethyl", 37: "trimethyl", 121: "gg", 737: "tmt", 2016: "tmtpro",
          43: "hexnac", 41: "hex", 425: "dioxidation", 354: "nitro", 28: "gln->pyro-glu",
          27: "glu->pyro-glu"}
ION_COLORS = {"b": "#2F5D8A", "y": "#C0392B", "a": "#2E8B57", "c": "#6F9A6B",
              "z": "#B5543A", "x": "#8A7BA8", "precursor": "#E0A030"}
UNMATCHED = "#BFC4CB"


def parse_peptide(peptide: str):
    """Parse a (modified) peptide in common search-engine notations.

    'GGGGNF[Oxidation]GPGPGSNFR'   name in brackets     'M(ox)'  short name
    'S[+79.966]'                   mass delta           'H[Phospho|pH]'  custom label
    '_VVSVLTVLH(Oxidation (H))QDWLNGK_'   MaxQuant modified sequence
    'PEPT(UniMod:21)IDE', 'M[UNIMOD:35]'  DIA-NN / ProForma UniMod ids
    'N[HexNAc]', 'T[Hex]', 'S[GlcNAc]'    glycan residues
    '[Acetyl]-PEPTIDE', '(Acetyl (Protein N-term))PEP…'  N-terminal

    Returns (residues, deltas, labels, nterm_delta).
    """
    pep = peptide.strip().strip("_").strip(".")
    residues, deltas, labels = [], [], []
    nterm = 0.0
    i = 0
    while i < len(pep):
        c = pep[i]
        if c.isalpha() and c.isupper():
            residues.append(c); deltas.append(0.0); labels.append("")
            i += 1
        elif c in "[(":
            j, depth = i, 0
            while j < len(pep):                    # matching bracket, nesting allowed
                if pep[j] in "[(":
                    depth += 1
                elif pep[j] in "])":
                    depth -= 1
                    if depth == 0:
                        break
                j += 1
            d, lab = _mod_delta(pep[i + 1:j])
            if residues:
                deltas[-1] += d
                labels[-1] = (labels[-1] + "·" + lab).strip("·")
            else:
                nterm += d
            i = j + 1
            if i < len(pep) and pep[i] == "-" and not residues:
                i += 1
        else:
            i += 1
    unknown = [r for r in residues if r not in AA_MONO]
    if unknown:
        raise ValueError(f"unknown residues {set(unknown)} in peptide")
    return residues, deltas, labels, nterm


def _mod_delta(token: str):
    """Mass delta and display label for one modification token."""
    import re
    t = token.strip()
    label = None
    if "|" in t:                                   # 'Phospho|pH' → custom label
        t, label = (x.strip() for x in t.split("|", 1))
    m = re.match(r"(?i)^unimod:(\d+)$", t)
    if m:
        uid = int(m.group(1))
        if uid not in UNIMOD:
            raise ValueError(f"UniMod:{uid} not in the built-in table; give the mass, e.g. [+79.966]")
        t = UNIMOD[uid]
    try:
        d = float(t)
        return d, label or (f"{d:+.0f}" if abs(d) >= 1 else f"{d:+.2f}")
    except ValueError:
        pass
    key = re.sub(r"\s*\([^()]*\)\s*$", "", t)     # 'Oxidation (M)' → 'Oxidation'
    key = key.lower().replace(" ", "").replace("_", "")
    if key not in MODS:
        raise ValueError(f"unknown modification '{token}'; use a mass like [+15.995], "
                         f"a UniMod id [UniMod:35], or one of {sorted(set(MODS))}")
    delta, default_label = MODS[key]
    return delta, label or default_label


def format_peptide(peptide: str, style: str = "short") -> str:
    """Display form of a peptide: 'VVSVLTVLH(ox)QDWLNGK' from any supported notation."""
    res, _, labs, nt = parse_peptide(peptide)
    body = "".join(r + (f"({l})" if l else "") for r, l in zip(res, labs))
    return (f"[{nt:+.0f}]-" if abs(nt) > 0.5 else "") + body


def precursor_mz(peptide: str, charge: int) -> float:
    res, d, _, nt = parse_peptide(peptide)
    M = sum(AA_MONO[r] for r in res) + sum(d) + nt + WATER
    return (M + charge * PROTON) / charge


def fragment_table(peptide: str, charges=(1, 2), ion_types=("b", "y"),
                   neutral_losses: bool = False):
    """Theoretical fragment m/z for a (modified) peptide.

    Columns: type, index (ion number), charge, loss, mz, label ('y5', 'b3²⁺', 'y7-H₂O').
    """
    import pandas as pd
    res, d, _, nt = parse_peptide(peptide)
    masses = [AA_MONO[r] + dd for r, dd in zip(res, d)]
    n = len(masses)
    rows = []
    losses = [("", 0.0)] + ([("-H₂O", WATER), ("-NH₃", AMMONIA)] if neutral_losses else [])
    for i in range(1, n):
        nsum = sum(masses[:i]) + nt
        csum = sum(masses[n - i:])
        neutral = {"b": nsum, "a": nsum - CO, "c": nsum + AMMONIA,
                   "y": csum + WATER, "x": csum + WATER + CO - 2 * 1.007825032,
                   "z": csum + WATER - AMMONIA + 1.007825032}          # z• (z+1)
        for t in ion_types:
            for z in charges:
                for lname, lmass in losses:
                    mz = (neutral[t] - lmass + z * PROTON) / z
                    sup = "" if z == 1 else f"$^{{{z}+}}$"
                    rows.append(dict(type=t, index=i, charge=z, loss=lname, mz=mz,
                                     label=f"{t}{i}{sup}{lname}"))
    return pd.DataFrame(rows)


def match_fragments(mz, intensity, frags, tol: float = 20.0, unit: str = "ppm"):
    """Assign each theoretical fragment the most intense peak within tolerance.

    Each observed peak is used once (priority: no loss before losses, low charge
    first, then larger ions). Returns frags with obs_mz, obs_int, error columns.
    """
    import pandas as pd
    mz = np.asarray(mz, float); inten = np.asarray(intensity, float)
    order = np.argsort(mz); mz, inten = mz[order], inten[order]
    f = frags.copy()
    f["_prio"] = (f["loss"] != "").astype(int) * 10 + f["charge"]
    f = f.sort_values(["_prio", "index"], ascending=[True, False])
    used = np.zeros(len(mz), bool)
    obs_mz, obs_int, err = [], [], []
    for t in f["mz"].to_numpy():
        w = t * tol * 1e-6 if unit == "ppm" else tol
        lo, hi = np.searchsorted(mz, [t - w, t + w])
        cand = [k for k in range(lo, hi) if not used[k]]
        if cand:
            k = max(cand, key=lambda q: inten[q])
            used[k] = True
            obs_mz.append(mz[k]); obs_int.append(inten[k])
            err.append((mz[k] - t) / t * 1e6 if unit == "ppm" else mz[k] - t)
        else:
            obs_mz.append(np.nan); obs_int.append(np.nan); err.append(np.nan)
    f["obs_mz"], f["obs_int"], f["error"] = obs_mz, obs_int, err
    return f.drop(columns="_prio").sort_values(["type", "index", "charge"]).reset_index(drop=True)


def read_peaks(path):
    """Two-column peak list (m/z, intensity) from CSV/TSV/whitespace text.
    A header row and comment lines (#) are skipped automatically."""
    import pandas as pd
    df = pd.read_csv(path, sep=None, engine="python", comment="#", header=None)
    df = df.iloc[:, :2].apply(pd.to_numeric, errors="coerce").dropna()
    if df.empty:
        raise ValueError(f"no numeric m/z, intensity pairs found in {path}")
    return df.iloc[:, 0].to_numpy(float), df.iloc[:, 1].to_numpy(float)


def read_mgf(path) -> list:
    """Spectra from an MGF file → [{title, pepmass, charge, rt, mz, intensity, …}]."""
    spectra, cur = [], None
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if line == "BEGIN IONS":
            cur = {"mz": [], "intensity": []}
        elif line == "END IONS" and cur is not None:
            cur["mz"] = np.array(cur["mz"]); cur["intensity"] = np.array(cur["intensity"])
            spectra.append(cur); cur = None
        elif cur is not None and "=" in line and not line[0].isdigit():
            k, v = line.split("=", 1)
            k = k.lower()
            if k == "pepmass":
                v = float(v.split()[0])
            elif k == "rtinseconds":
                k, v = "rt", float(v) / 60
            cur[k] = v
        elif cur is not None and line:
            parts = line.split()
            cur["mz"].append(float(parts[0])); cur["intensity"].append(float(parts[1]))
    return spectra


def _sequence_strip(ax_strip, residues, labels, matched_b, matched_y, fs=None, fs_small=None):
    """Peptide with fragment ladder: b ions below (blue), y ions above (red),
    modification tags in their own row on top (amber, matching the letter)."""
    n = len(residues)
    k = sizes()["stroke"]
    fs = fs or sizes()["label"] + 1
    fs_small = fs_small or max(sizes()["min"], sizes()["tick"] - 2)
    has_mods = any(labels)
    ax_strip.set_xlim(-0.8, n - 0.2)
    ax_strip.set_ylim(-1.25, 2.55 if has_mods else 1.25)
    ax_strip.set_axis_off()
    ax_strip.set_gid("sequence")
    mod_col = "#B7791F"
    fs_tag = max(fs_small + 1, min(sizes()["tick"], fs))
    for i, (r, lab) in enumerate(zip(residues, labels)):
        ax_strip.text(i, 0, r, ha="center", va="center", fontsize=fs,
                      color=mod_col if lab else INK["text"],
                      fontweight="bold" if lab else "normal")
        if lab:                              # own row above the ladder, in a pale box
            ax_strip.text(i, 1.62, lab, ha="center", va="bottom", fontsize=fs_tag,
                          color=darken(mod_col, 0.8), fontweight="bold",
                          bbox=dict(boxstyle="round,pad=0.18,rounding_size=0.25",
                                    facecolor=tint(mod_col, 0.82), edgecolor=mod_col,
                                    linewidth=0.5))
    cb, cy = ION_COLORS["b"], ION_COLORS["y"]
    arm = 0.38
    for i in sorted(matched_b):             # b_i: cleavage after residue i-1
        x = i - 0.5
        ax_strip.plot([x, x, x - arm], [0.12, -0.55, -0.55], color=cb, lw=0.8 * k,
                      solid_capstyle="butt", clip_on=False)
        ax_strip.text(x - arm / 2, -0.68, str(i), ha="center", va="top", fontsize=fs_small, color=cb)
    for j in sorted(matched_y):             # y_j: last j residues
        x = n - j - 0.5
        ax_strip.plot([x, x, x + arm], [-0.12, 0.55, 0.55], color=cy, lw=0.8 * k,
                      solid_capstyle="butt", clip_on=False)
        ax_strip.text(x + arm / 2, 0.68, str(j), ha="center", va="bottom", fontsize=fs_small, color=cy)


def _add_sequence_strip(ax, residues, labels, matched_b, matched_y):
    """Place the ladder above the axes, sized so every residue gets enough room."""
    fig = ax.figure
    n = len(residues)
    ax_w_pt = ax.get_position().width * fig.get_figwidth() * 72
    fs = sizes()["label"] + 1
    fs_small = max(sizes()["min"], sizes()["tick"] - 2)
    need = 2.1 * fs                         # comfortable pitch per residue (pt)
    w = min(0.98, n * need / ax_w_pt)
    pitch = w * ax_w_pt / n
    if pitch < need:                        # narrow panel: shrink letters to fit
        fs = max(sizes()["min"] + 1, pitch / 2.1)
        fs_small = max(sizes()["min"], min(fs_small, fs - 2))
    h = 0.31 if any(labels) else 0.20
    strip = ax.inset_axes([0.5 - w / 2, 1.03, w, h])
    _sequence_strip(strip, residues, labels, matched_b, matched_y, fs=fs, fs_small=fs_small)
    ax._pf_top = 1.03 + h + 0.02
    return strip


def _place_ion_labels(ax, peaks, fs):
    """Labels above their peaks, lifted with a dotted leader when they would collide."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    ax_bb = ax.get_window_extent(r)
    placed = []
    px = fig.dpi / 72
    for mz_, h, text, color in sorted(peaks, key=lambda t: -t[1]):
        ann = ax.annotate(text, xy=(mz_, h), xytext=(0, 2), textcoords="offset points",
                          ha="center", va="bottom", fontsize=fs, color=color, zorder=5,
                          annotation_clip=False)
        lift = 0.0
        for _ in range(40):
            bb = _text_bbox(ann, r)
            clash = [q for q in placed if bb.overlaps(q)]
            if not clash:
                break
            lift += (max(q.y1 for q in clash) - bb.y0) / px + 1.0
            ann.xyann = (0, 2 + lift)
        bb = _text_bbox(ann, r)
        if bb.y1 > ax_bb.y1 + 2:            # would leave the plot: skip rather than overlap
            ann.remove()
            continue
        placed.append(bb)
        if lift > 0:                          # dotted leader from peak top to the label
            y_top = ax.transData.inverted().transform((0, bb.y0 - 1 * px))[1]
            ax.plot([mz_, mz_], [h, y_top], color=color, lw=0.5, ls=(0, (1, 1.2)), zorder=4)


def _text_width_pt(text: str, size: float) -> float:
    """Printed width of a string in points (for layout decisions before drawing)."""
    from matplotlib.textpath import TextPath
    from matplotlib.font_manager import FontProperties
    clean = text.replace("$", "")
    prop = FontProperties(family=resolve_font(SANS_STACK), size=size)
    return TextPath((0, 0), clean, prop=prop).get_extents().width


def spectrum_plot(ax, mz, intensity, peptide: str | None = None, charges=(1, 2),
                  ion_types=("b", "y"), tol: float = 20.0, unit: str = "ppm",
                  neutral_losses: bool = False, header: str | None = None,
                  yaxis: str = "relative", show_sequence: bool = True, label_ions: bool = True,
                  min_label_rel: float = 0.0, xlim=None, note: bool = True,
                  dia: bool = False, isolation_width: float | None = None,
                  isolation_center: float | None = None):
    """Annotated MS/MS spectrum: sticks, matched fragments coloured and labelled,
    fragment-ladder sequence above.

    yaxis   "relative" (% of base peak, default), "absolute", or "both" (absolute
            left, % right: the classic layout).
    header  scan line drawn top-left, e.g. "Scan 10774 · m/z 689.32 · 2+ · RT 43.9 min".
    dia     the scan is data-independent (DIA/SWATH): it fragments every precursor
            in an isolation window, so co-isolated peptides contribute fragments.
            Requires `isolation_width` (m/z); `isolation_center` optional. The
            window is recorded in the result, shown in the summary line, and the
            returned caption states that the annotation is supporting evidence
            for the peptide, not a single-precursor identification.
    Returns dict(matches, coverage, n_matched, n_theoretical, base_peak, caption,
    dia, isolation_width, isolation_window).
    """
    if dia and not isolation_width:
        raise ValueError("dia=True needs isolation_width (the DIA window width in m/z), "
                         "so the figure and caption can state what was fragmented")
    mz = np.asarray(mz, float); inten = np.asarray(intensity, float)
    base = float(inten.max()) if len(inten) else 1.0
    y = inten / base * 100 if yaxis in ("relative", "both") else inten
    scale = 100 / base if yaxis in ("relative", "both") else 1.0
    k = sizes()["stroke"]
    ax.vlines(mz, 0, y, color=UNMATCHED, lw=0.5 * k, zorder=1)

    window = ((isolation_center - isolation_width / 2, isolation_center + isolation_width / 2)
              if dia and isolation_center is not None else None)
    out = dict(base_peak=base, dia=bool(dia), isolation_width=isolation_width if dia else None,
               isolation_window=window)
    matched_b, matched_y, label_jobs = set(), set(), []
    if peptide:
        frags = fragment_table(peptide, charges=charges, ion_types=ion_types,
                               neutral_losses=neutral_losses)
        m = match_fragments(mz, inten, frags, tol=tol, unit=unit)
        hit = m.dropna(subset=["obs_mz"])
        for _, row in hit.iterrows():
            col = ION_COLORS.get(row["type"], INK["text"])
            if row["loss"]:
                col = to_hex(mpl.colors.to_rgba(col, 0.55))
            ax.vlines(row["obs_mz"], 0, row["obs_int"] * scale, color=col, lw=0.9 * k, zorder=2)
            if label_ions and row["obs_int"] / base >= min_label_rel:
                label_jobs.append((row["obs_mz"], row["obs_int"] * scale, row["label"], col))
            if row["type"] == "b":
                matched_b.add(int(row["index"]))
            elif row["type"] == "y":
                matched_y.add(int(row["index"]))
        res, _, labs, _ = parse_peptide(peptide)
        out.update(matches=m, n_matched=int(len(hit)), n_theoretical=int(len(m)),
                   coverage=(len(matched_b | {len(res) - j for j in matched_y}) / max(1, len(res) - 1)))
        if show_sequence:
            _add_sequence_strip(ax, res, labs, matched_b, matched_y)
    ax.set_ylim(0, (y.max() if len(y) else 1) * 1.16)
    if xlim is not None:
        ax.set_xlim(*xlim)
    else:
        pad = (mz.max() - mz.min()) * 0.03 if len(mz) else 1
        ax.set_xlim(max(0, mz.min() - pad), mz.max() + pad)
    ax.set_xlabel("$m/z$")
    if yaxis == "relative":
        ax.set_ylabel("Relative intensity (%)")
        ax.set_yticks([0, 25, 50, 75, 100])
    elif yaxis == "absolute":
        ax.set_ylabel("Intensity")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
    else:
        ax.set_ylabel("Relative intensity (%)")
        ax.set_yticks([0, 25, 50, 75, 100])
        sec = ax.secondary_yaxis("right", functions=(lambda v: v * base / 100,
                                                     lambda v: v * 100 / base))
        sec.set_ylabel("Intensity")
        sec.ticklabel_format(style="sci", scilimits=(-3, 4), useMathText=True)
        ax.spines["right"].set_visible(False)
    top = getattr(ax, "_pf_top", 1.0)
    fs = sizes()["annot"]
    note_txt = None
    dia_txt = f" · DIA {isolation_width:g} $m/z$ window" if dia else ""
    if note and peptide:
        note_txt = (f"{out['n_matched']} ions · {out['coverage']:.0%} backbone coverage · "
                    f"±{tol:g} {unit}{dia_txt}")
    # header and note share one line when they fit; otherwise shorten, then stack
    ax_w = ax.get_position().width * ax.figure.get_figwidth() * 72
    width_of = lambda t: _text_width_pt(t, fs) if t else 0.0
    if note_txt and width_of(header) + width_of(note_txt) + 12 > ax_w:
        note_txt = (f"{out['n_matched']} ions · {out['coverage']:.0%} cov. · ±{tol:g} {unit}"
                    + (f" · DIA {isolation_width:g}" if dia else ""))
    stacked = bool(header and note_txt and width_of(header) + width_of(note_txt) + 12 > ax_w)
    if header:
        ax.annotate(header, xy=(0, top), xycoords="axes fraction",
                    xytext=(0, 4 + (fs * 1.45 if stacked else 0)),
                    textcoords="offset points", ha="left", va="bottom", fontsize=fs,
                    color=INK["text"])
    if note_txt:
        ax.annotate(note_txt, xy=(0 if stacked else 1, top), xycoords="axes fraction",
                    xytext=(0, 4), textcoords="offset points",
                    ha="left" if stacked else "right", va="bottom", fontsize=fs,
                    color=INK["muted"])
    if label_jobs:
        fig = ax.figure
        fig.__dict__.setdefault("_pf_jobs", []).append(
            lambda ax=ax, jobs=label_jobs: _place_ion_labels(ax, jobs, sizes()["annot"]))
    out["caption"] = _psm_caption(peptide, out, charges, ion_types, tol, unit)
    return out


def _psm_caption(peptide, out, charges, ion_types, tol, unit) -> str:
    """Caption sentence for an annotated spectrum; states DIA co-isolation if relevant."""
    if not peptide:
        txt = "MS/MS spectrum (unassigned)."
    else:
        txt = (f"MS/MS spectrum annotated with {format_peptide(peptide)}: "
               f"{out['n_matched']} of {out['n_theoretical']} theoretical "
               f"{'/'.join(ion_types)} fragments (charges "
               f"{', '.join(f'{c}+' for c in charges)}) matched within ±{tol:g} {unit}, "
               f"{out['coverage']:.0%} backbone coverage.")
    if out.get("dia"):
        w = out["isolation_width"]
        win = out.get("isolation_window")
        where = f" ({win[0]:.1f}–{win[1]:.1f} m/z)" if win else ""
        txt += (f" Data-independent acquisition: the scan fragments every precursor in a "
                f"{w:g} m/z isolation window{where}, so fragments of co-isolated peptides "
                "can contribute; the annotation is supporting evidence for this peptide, "
                "not a single-precursor identification.")
    return txt


def spectral_similarity(mz1, i1, mz2, i2, tol: float = 20.0, unit: str = "ppm"):
    """Normalised spectral contrast angle and cosine on peaks matched within tolerance."""
    a_mz, a_i = np.asarray(mz1, float), np.sqrt(np.asarray(i1, float))
    b_mz, b_i = np.asarray(mz2, float), np.sqrt(np.asarray(i2, float))
    va, vb, used = [], [], set()
    for m, v in sorted(zip(a_mz, a_i), key=lambda t: -t[1]):
        w = m * tol * 1e-6 if unit == "ppm" else tol
        cands = [k for k in np.where(np.abs(b_mz - m) <= w)[0] if k not in used]
        if cands:
            kk = max(cands, key=lambda q: b_i[q]); used.add(kk)
            va.append(v); vb.append(b_i[kk])
        else:
            va.append(v); vb.append(0.0)
    for kk in range(len(b_mz)):
        if kk not in used:
            va.append(0.0); vb.append(b_i[kk])
    va, vb = np.array(va), np.array(vb)
    cos = float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) or 1))
    sa = 1 - 2 * np.arccos(np.clip(cos, -1, 1)) / np.pi
    return dict(cosine=cos, spectral_angle=float(sa))


def _scale_pair(i1, i2, scale: str):
    """Scale two intensity vectors for side-by-side comparison.

    "each"   each to its own base peak (pattern only: hides abundance changes)
    "shared" both to the larger base peak (absolute differences preserved)
    "tic"    each to its total ion current, as % TIC (corrects loading)
    """
    i1, i2 = np.asarray(i1, float), np.asarray(i2, float)
    if scale == "each":
        return i1 / i1.max() * 100, i2 / i2.max() * 100, "Relative intensity (%)"
    if scale == "shared":
        m = max(i1.max(), i2.max())
        return i1 / m * 100, i2 / m * 100, "Intensity (% of max)"
    if scale == "tic":
        return i1 / i1.sum() * 100, i2 / i2.sum() * 100, "Intensity (% TIC)"
    raise ValueError("scale must be 'each', 'shared' or 'tic'")


def mirror_plot(ax, top, bottom, peptide: str | None = None, labels=("Experimental", "Predicted"),
                tol: float = 20.0, unit: str = "ppm", charges=(1, 2), ion_types=("b", "y"),
                show_sequence: bool = True, scale: str = "each"):
    """Experimental spectrum up, reference/predicted spectrum mirrored down.

    top/bottom: (mz, intensity) pairs. Fragments of `peptide` are coloured in
    both halves; the stat line gives the normalised spectral angle.
    """
    (m1, i1), (m2, i2) = top, bottom
    m1, i1, m2, i2 = (np.asarray(v, float) for v in (m1, i1, m2, i2))
    r1, r2, ylab = _scale_pair(i1, i2, scale)
    k = sizes()["stroke"]
    ax.vlines(m1, 0, r1, color=UNMATCHED, lw=0.5 * k, zorder=1)
    ax.vlines(m2, 0, -r2, color=UNMATCHED, lw=0.5 * k, zorder=1)
    jobs, mb, my = [], set(), set()
    if peptide:
        frags = fragment_table(peptide, charges=charges, ion_types=ion_types)
        for sign, (mm, ii, rr) in ((1, (m1, i1, r1)), (-1, (m2, i2, r2))):
            m = match_fragments(mm, ii, frags, tol=tol, unit=unit).dropna(subset=["obs_mz"])
            for _, row in m.iterrows():
                col = ION_COLORS.get(row["type"], INK["text"])
                h = float(rr[np.argmin(np.abs(mm - row["obs_mz"]))])
                ax.vlines(row["obs_mz"], 0, sign * h, color=col, lw=0.9 * k, zorder=2)
                if sign > 0:
                    jobs.append((row["obs_mz"], h, row["label"], col))
                    (mb if row["type"] == "b" else my).add(int(row["index"]))
        if show_sequence:
            res, _, labs, _ = parse_peptide(peptide)
            _add_sequence_strip(ax, res, labs, mb, my)
    refline(ax, y=0, ls="-", color=INK["axis"], lw=0.6)
    top_v = max(r1.max(), r2.max())
    ax.set_ylim(-top_v * 1.15, top_v * 1.18)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, symmetric=True))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{abs(v):g}"))
    ax.set_ylabel(ylab)
    ax.set_xlabel("$m/z$")
    allm = np.concatenate([m1, m2]); pad = (allm.max() - allm.min()) * 0.03
    ax.set_xlim(max(0, allm.min() - pad), allm.max() + pad)
    fs = sizes()["annot"]
    ax.text(0.01, 0.97, labels[0], transform=ax.transAxes, ha="left", va="top", fontsize=fs,
            color=INK["text"])
    ax.text(0.01, 0.03, labels[1], transform=ax.transAxes, ha="left", va="bottom", fontsize=fs,
            color=INK["text"])
    sim = spectral_similarity(m1, i1, m2, i2, tol=tol, unit=unit)
    top_ = getattr(ax, "_pf_top", 1.0)
    ax.annotate(f"spectral angle {sim['spectral_angle']:.2f} · cosine {sim['cosine']:.2f}",
                xy=(1, top_), xycoords="axes fraction", xytext=(0, 4),
                textcoords="offset points", ha="right", va="bottom", fontsize=fs,
                color=INK["muted"])
    if jobs:
        ax.figure.__dict__.setdefault("_pf_jobs", []).append(
            lambda ax=ax, jobs=jobs: _place_ion_labels(ax, jobs, sizes()["annot"]))
    return sim


def spectrum_panels(spectra: list, journal: str | None = None, width: str | float = "single",
                    panel_height_mm: float = 48, sharex: bool = True, **kw):
    """Stacked annotated spectra sharing the m/z axis (e.g. unmodified vs oxidised forms).

    spectra: list of dicts with keys mz, intensity, peptide, and optional header,
    charges. Extra keyword arguments go to spectrum_plot. Returns (fig, axes, results).
    """
    n = len(spectra)
    total_w = (JOURNALS[journal or _STATE["journal"]][width] if isinstance(width, str)
               else float(width))
    fig, axes = plt.subplots(n, 1, figsize=(total_w * MM, n * panel_height_mm * MM),
                             sharex=sharex, squeeze=False)
    axes = axes[:, 0]
    allm = np.concatenate([np.asarray(sp["mz"], float) for sp in spectra])
    pad = (allm.max() - allm.min()) * 0.03
    xl = (max(0, allm.min() - pad), allm.max() + pad)
    results = []
    for i, (ax, sp) in enumerate(zip(axes, spectra)):
        opts = dict(kw)
        opts.update({k: v for k, v in sp.items() if k in ("charges", "header", "ion_types")})
        results.append(spectrum_plot(ax, sp["mz"], sp["intensity"], sp.get("peptide"),
                                     xlim=xl, **opts))
        if i < n - 1:
            ax.set_xlabel("")
            ax.tick_params(labelbottom=False)
    return fig, axes, results


# --------------------------------------------------------------------------
# Treated vs control MS comparisons
# --------------------------------------------------------------------------
def _pair_peaks(m1, i1, m2, i2, tol, unit):
    """Pair peaks of two spectra within tolerance (most intense first).
    Returns rows (mz, i1, i2); unpaired peaks get 0 on the other side."""
    m1, i1, m2, i2 = (np.asarray(v, float) for v in (m1, i1, m2, i2))
    used2, rows = set(), []
    for k in np.argsort(-i1):
        w = m1[k] * tol * 1e-6 if unit == "ppm" else tol
        cand = [j for j in np.where(np.abs(m2 - m1[k]) <= w)[0] if j not in used2]
        if cand:     # nearest in m/z (then the more intense): a stronger neighbour
            j = min(cand, key=lambda q: (abs(m2[q] - m1[k]), -i2[q]))   # must not win
            used2.add(j)
            rows.append((m1[k], i1[k], i2[j]))
        else:
            rows.append((m1[k], i1[k], 0.0))
    for j in range(len(m2)):
        if j not in used2:
            rows.append((m2[j], 0.0, i2[j]))
    return np.array(rows) if rows else np.zeros((0, 3))


ISOTOPE_SPACING = 1.00335       # Da, 13C–12C; isotope peaks sit at this / charge apart


def isotope_envelope(mz: float, charge: int, n: int = 3) -> list:
    """Expected m/z of M, M+1, M+2 … for a precursor at `mz` with `charge`."""
    return [(("M" if i == 0 else f"M+{i}"), mz + i * ISOTOPE_SPACING / charge) for i in range(n)]


def _fmt_height(v: float) -> str:
    """Isotope heights with consistent decimals: 0.74, 2.0, 12."""
    if not np.isfinite(v):
        return "n.d."
    return f"{v:.2f}" if v < 1 else f"{v:.1f}" if v < 10 else f"{v:.0f}"


def _draw_isotope_envelope(ax, iso_df, up_c, dn_c, fs, xr):
    """Guide lines at the expected isotope m/z, and each isotope's measured height
    in both halves. Labels sit directly on the peaks when the envelope is wide
    enough; otherwise they form a callout row above (treated) and below (control)
    the envelope, each joined to its peak by a thin leader."""
    fig = ax.figure
    pos = ax.get_position()
    ax_w_pt = pos.width * fig.get_figwidth() * 72
    ax_h_pt = pos.height * fig.get_figheight() * 72
    x0, x1 = ax.get_xlim()
    lo, hi = ax.get_ylim()
    x_per_pt = (x1 - x0) / ax_w_pt
    fs_iso = fs
    for _, r in iso_df.iterrows():
        ax.axvline(r.expected_mz, color=INK["muted"], lw=0.5, ls=(0, (1.5, 1.5)), zorder=0.5)
    top_txt = [f"{r.isotope} {_fmt_height(r.treated)}" for _, r in iso_df.iterrows()]
    bot_txt = [f"{r.isotope} {_fmt_height(r.control)}" for _, r in iso_df.iterrows()]
    widths = [max(_text_width_pt(t, fs_iso), _text_width_pt(b, fs_iso)) + 4
              for t, b in zip(top_txt, bot_txt)]
    xs = [(r.measured_mz if r.found else r.expected_mz) for _, r in iso_df.iterrows()]
    gaps_pt = [abs(b - a) / x_per_pt for a, b in zip(xs, xs[1:])]
    direct = all(g >= (w1 + w2) / 2 for g, w1, w2 in zip(gaps_pt, widths, widths[1:]))
    t_h = [r.treated if r.found else 0.0 for _, r in iso_df.iterrows()]
    c_h = [r.control if r.found else 0.0 for _, r in iso_df.iterrows()]
    cols_t = [up_c if r.found else INK["muted"] for _, r in iso_df.iterrows()]
    cols_c = [dn_c if r.found else INK["muted"] for _, r in iso_df.iterrows()]
    from matplotlib import patheffects as pe
    kw = dict(fontsize=fs_iso, fontweight="bold", zorder=6,
              path_effects=[pe.withStroke(linewidth=2.4, foreground="white")])
    if direct:                                        # enough room: label on each peak
        for x, h, txt, col in zip(xs, t_h, top_txt, cols_t):
            ax.annotate(txt, xy=(x, h), xytext=(0, 2), textcoords="offset points",
                        ha="center", va="bottom", color=col, **kw)
        for x, h, txt, col in zip(xs, c_h, bot_txt, cols_c):
            ax.annotate(txt, xy=(x, -h), xytext=(0, -2), textcoords="offset points",
                        ha="center", va="top", color=col, **kw)
        return
    # callout rows: labels spread evenly around the envelope centre
    y_per_pt = (hi - lo) / ax_h_pt
    total = sum(widths) * x_per_pt
    cx = float(np.mean(xs))
    left = min(max(cx - total / 2, x0 + 2 * x_per_pt), x1 - total - 2 * x_per_pt)
    centres, run = [], left
    for w in widths:
        centres.append(run + w * x_per_pt / 2)
        run += w * x_per_pt
    lift = 14 * y_per_pt                               # row sits 14 pt beyond the tallest peak
    row_t = max(t_h) + lift
    row_c = max(c_h) + lift
    need_hi = row_t + fs_iso * 1.6 * y_per_pt
    need_lo = -(row_c + fs_iso * 1.6 * y_per_pt)
    if need_hi > hi or need_lo < lo:                   # make room, keep the axis symmetric
        m = max(need_hi, -need_lo, hi, -lo)
        ax.set_ylim(-m, m)
    for x, cxl, h, txt, col in zip(xs, centres, t_h, top_txt, cols_t):
        ax.plot([x, cxl], [h + 1.5 * y_per_pt, row_t - 1.5 * y_per_pt], color=col, lw=0.45,
                zorder=5, solid_capstyle="butt")
        ax.text(cxl, row_t, txt, ha="center", va="bottom", color=col, **kw)
    for x, cxl, h, txt, col in zip(xs, centres, c_h, bot_txt, cols_c):
        ax.plot([x, cxl], [-h - 1.5 * y_per_pt, -row_c + 1.5 * y_per_pt], color=col, lw=0.45,
                zorder=5, solid_capstyle="butt")
        ax.text(cxl, -row_c, txt, ha="center", va="top", color=col, **kw)


def compare_spectra(ax, treated, control, labels=("Treatment", "Control"), scale: str = "tic",
                    fold: float = 2.0, min_rel: float = 2.0, tol: float = 10.0,
                    unit: str = "ppm", ratio_strip: bool = True, label_top: int = 6,
                    cap: float = 5.0, colors: dict | None = None, precursor=None,
                    n_isotopes: int = 3, iso_tol: float | None = None):
    """Treated spectrum up, control mirrored down, on a common intensity scale,
    with peaks that change ≥ `fold` highlighted and a log₂(treated/control) strip.

    scale     "tic" (default, % of total ion current: corrects for loading),
              "shared" (one absolute axis) or "each" (own base peak: shows the
              pattern only and would hide abundance changes: avoid here).
    min_rel   only peaks ≥ this % (on the chosen scale, in either spectrum) are
              tested and coloured, so noise does not light up.
    cap       |log₂ ratio| shown for peaks present on one side only (drawn as ▲/▼).
    colors    {"up": …, "down": …}; default treatment amber, control blue (as in
              xic_plot and the occupancy bars; control blue). Each half is drawn in its
              condition's colour: pale and thin for unchanged peaks; at every
              m/z that changed ≥ fold, both halves at the same heavier weight in
              full colour.
    precursor (mz, charge) of the targeted peptide. Draws its expected isotope
              envelope (M, M+1, M+2 … at 1.00335/charge spacing) as guide lines,
              finds each isotope in both spectra (within `iso_tol`, default `tol`),
              draws them at full weight and labels them with their measured
              heights in both spectra. Isotope peaks take the label slots first;
              unrelated peaks that changed are labelled only with the slots left,
              only away from the envelope, and in muted grey: so a co-eluting
              neighbour is never presented as the peptide. Isotopes not found are
              reported ("n.d.") rather than silently skipped.
    A single spectrum per condition illustrates a difference; replicate XICs or
    quantification are the evidence (see xic_plot, grouped_bar_points).
    Ratio strip: peaks present in only one condition have an infinite ratio; they
    are drawn as ▲/▼ at ±`cap` and a key under the strip says so.
    df.attrs["caption"] gives a caption sentence for the strip convention.
    Returns a DataFrame of paired peaks with log2 ratio and 'change' class; with
    `precursor`, an 'isotope' column names the envelope peaks and
    df.attrs["precursor"] holds the per-isotope table (expected/measured m/z,
    heights in both spectra, ratio, found).
    """
    import pandas as pd
    (m1, i1), (m2, i2) = treated, control
    m1, i1, m2, i2 = (np.asarray(v, float) for v in (m1, i1, m2, i2))
    r1, r2, ylab = _scale_pair(i1, i2, scale)
    P = _pair_peaks(m1, r1, m2, r2, tol, unit)
    df = pd.DataFrame(P, columns=["mz", "treated", "control"])
    eps = 1e-9
    with np.errstate(divide="ignore"):
        lr = np.log2((df.treated + eps) / (df.control + eps))
    df["log2_ratio"] = np.clip(lr, -cap, cap)
    big = (df.treated >= min_rel) | (df.control >= min_rel)
    thr = np.log2(fold)
    df["change"] = np.where(big & (df.log2_ratio >= thr), "up",
                            np.where(big & (df.log2_ratio <= -thr), "down", "same"))
    df["isotope"] = ""
    iso_rows = []
    if precursor is not None:
        p_mz, p_z = float(precursor[0]), int(precursor[1])
        itol = tol if iso_tol is None else iso_tol
        for name, exp_mz in isotope_envelope(p_mz, p_z, n_isotopes):
            w = exp_mz * itol * 1e-6 if unit == "ppm" else itol
            near = df[(df.mz - exp_mz).abs() <= w]
            if len(near):     # the peak nearest the expected m/z, not the strongest
                j = (near.mz - exp_mz).abs().idxmin()
                df.loc[j, "isotope"] = name
                r = df.loc[j]
                ratio = (r.treated / r.control) if r.control > 0 else np.inf
                iso_rows.append(dict(isotope=name, expected_mz=exp_mz, measured_mz=r.mz,
                                     treated=r.treated, control=r.control, ratio=ratio,
                                     found=True))
            else:
                iso_rows.append(dict(isotope=name, expected_mz=exp_mz, measured_mz=np.nan,
                                     treated=np.nan, control=np.nan, ratio=np.nan, found=False))
        missing = [r["isotope"] for r in iso_rows if not r["found"]]
        if missing:
            print(f"[pubfig] compare_spectra: {', '.join(missing)} of the precursor "
                  f"{p_mz:.4f} ({p_z}+) not found within ±{itol:g} {unit}")
        df.attrs["precursor"] = pd.DataFrame(iso_rows)
    tc = PALETTES["treat_control"]
    up_c = (colors or {}).get("up", tc["treatment"])
    dn_c = (colors or {}).get("down", tc["control"])
    k = sizes()["stroke"]
    # every line carries its condition's colour. Unchanged peaks: pale and thin.
    # At every m/z that changed (either direction) BOTH halves are drawn at the
    # same heavier weight in full colour, so the pair reads as one comparison.
    changed = ((df.change != "same") | (df.isotope != "")).to_numpy()   # + precursor envelope
    for sign, col_name, base_c in ((1, "treated", up_c), (-1, "control", dn_c)):
        v = df[col_name].to_numpy()
        keep = v > 0
        pale, full = keep & ~changed, keep & changed
        ax.vlines(df.mz[pale], 0, sign * v[pale], colors=tint(base_c, 0.45),
                  linewidths=0.55 * k, zorder=2)
        ax.vlines(df.mz[full], 0, sign * v[full], colors=base_c, linewidths=1.3 * k,
                  zorder=3)
    refline(ax, y=0, ls="-", color=INK["axis"], lw=0.6)
    top_v = float(max(df.treated.max(), df.control.max()))
    ax.set_ylim(-top_v * 1.18, top_v * 1.22)
    ax.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=5, symmetric=True))
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{abs(v):g}"))
    ax.set_ylabel(ylab)
    pad = (df.mz.max() - df.mz.min()) * 0.03
    ax.set_xlim(max(0, df.mz.min() - pad), df.mz.max() + pad)
    fs = sizes()["annot"]
    ax.text(0.01, 0.97, labels[0], transform=ax.transAxes, ha="left", va="top", fontsize=fs,
            color=up_c, fontweight="bold")
    ax.text(0.01, 0.03, labels[1], transform=ax.transAxes, ha="left", va="bottom", fontsize=fs,
            color=dn_c, fontweight="bold")
    n_up = int(((df.change == "up") & (df.isotope == "")).sum())
    n_dn = int(((df.change == "down") & (df.isotope == "")).sum())
    iso_df = df.attrs.get("precursor")
    if iso_df is not None:
        m0 = iso_df.iloc[0]
        ratio_txt = ("M n.d." if not m0.found else
                     f"M {labels[0][0]}/{labels[1][0]} = "
                     + ("only in " + labels[0].lower() if not np.isfinite(m0.ratio) else f"{m0.ratio:.1f}×"))
        stat_title(ax, f"precursor {precursor[0]:.2f} ({int(precursor[1])}+) · {ratio_txt} · "
                       f"other peaks ≥{fold:g}×: {n_up} up, {n_dn} down")
    else:
        stat_title(ax, f"{n_up} higher in {labels[0].lower()} · {n_dn} higher in "
                       f"{labels[1].lower()} (≥{fold:g}×, peaks ≥{min_rel:g}%)")
    xr = df.mz.max() - df.mz.min()
    min_sep = max(1.6, xr * 0.04)                      # one label per isotope cluster
    n_iso_labelled = 0
    if iso_df is not None:
        _draw_isotope_envelope(ax, iso_df, up_c, dn_c, fs, xr)
        n_iso_labelled = int(iso_df.found.sum())
    # other changed peaks: only the slots left, away from the envelope, muted grey
    # isotope peaks take their slots first; unrelated peaks get what is left
    slots = max(0, label_top - n_iso_labelled) if iso_df is not None else label_top
    if slots:
        ch = df[(df.change != "same") & (df.isotope == "")].copy()
        if iso_df is not None:
            lo_env = iso_df.expected_mz.min() - min_sep
            hi_env = iso_df.expected_mz.max() + min_sep
            ch = ch[(ch.mz < lo_env) | (ch.mz > hi_env)]
        ch["size"] = np.maximum(ch.treated, ch.control)
        chosen = []
        for _, r in ch.sort_values("size", ascending=False).iterrows():
            side = r.treated >= r.control         # labels on opposite sides never collide
            if all(abs(r.mz - q.mz) > min_sep for q in chosen
                   if (q.treated >= q.control) == side):
                chosen.append(r)
            if len(chosen) >= slots:
                break
        for r in chosen:
            side = 1 if r.treated >= r.control else -1
            h = r.treated if side > 0 else r.control
            col = INK["muted"] if iso_df is not None else (up_c if r.change == "up" else dn_c)
            ax.annotate(f"{r.mz:.1f}", xy=(r.mz, side * h), xytext=(0, 2 * side),
                        textcoords="offset points", ha="center",
                        va="bottom" if side > 0 else "top", fontsize=fs, color=col)
    if ratio_strip:
        ss = ax.get_subplotspec()
        if ss is not None:       # split the panel's own grid cell → layout-aware strip
            sub = ss.subgridspec(2, 1, height_ratios=[1, 0.3], hspace=0.06)
            ax.set_subplotspec(sub[0])
            strip = ax.figure.add_subplot(sub[1], sharex=ax)
        else:
            strip = ax.inset_axes([0, -0.38, 1, 0.24], sharex=ax)
        tested = df[(df.treated >= min_rel) | (df.control >= min_rel)]
        c = tested.change.map({"up": up_c, "down": dn_c, "same": UNMATCHED})
        strip.vlines(tested.mz, 0, tested.log2_ratio, colors=c, linewidths=0.8 * k)
        only = tested[(tested.treated == 0) | (tested.control == 0)]   # one side only
        o_up, o_dn = only[only.log2_ratio > 0], only[only.log2_ratio < 0]
        strip.scatter(o_up.mz, o_up.log2_ratio, marker="^", s=10, color=up_c, lw=0, zorder=3)
        strip.scatter(o_dn.mz, o_dn.log2_ratio, marker="v", s=10, color=dn_c, lw=0, zorder=3)
        if len(o_up) or len(o_dn):   # key: the ratio is infinite, drawn at the cap
            from matplotlib.lines import Line2D
            hs, ls_ = [], []
            if len(o_up):
                hs.append(Line2D([], [], marker="^", ls="none", color=up_c, markersize=4))
                ls_.append(f"only in {labels[0]} (drawn at +{cap:g})")
            if len(o_dn):
                hs.append(Line2D([], [], marker="v", ls="none", color=dn_c, markersize=4))
                ls_.append(f"only in {labels[1]} (drawn at −{cap:g})")
            strip.legend(hs, ls_, loc="upper right", bbox_to_anchor=(1.0, -0.42), ncols=len(hs),
                         fontsize=max(sizes()["min"], sizes()["annot"] - 1), handletextpad=0.2,
                         columnspacing=0.8, borderaxespad=0, frameon=False)
        df.attrs["one_sided"] = dict(treated=int(len(o_up)), control=int(len(o_dn)), cap=cap)
        refline(strip, y=0, ls="-", color=INK["axis"], lw=0.5)
        refline(strip, y=thr); refline(strip, y=-thr)
        if iso_df is not None:
            for mzx in iso_df.expected_mz:
                strip.axvline(mzx, color=INK["muted"], lw=0.5, ls=(0, (1.5, 1.5)), zorder=0.5)
        strip.set_ylim(-cap * 1.15, cap * 1.15)
        strip.set_yticks([-cap, 0, cap] if cap >= 4 else [-thr, 0, thr])
        strip.set_ylabel("log$_2$ T/C")
        df.attrs["caption"] = (
            f"Strip, log₂({labels[0]}/{labels[1]}) for each paired peak, dashed at "
            f"±log₂ {fold:g}; peaks present in only one condition have no finite ratio and are "
            f"drawn as triangles at ±{cap:g} (▲ only in {labels[0]}, ▼ only in {labels[1]}).")
        strip.set_xlabel("$m/z$")
        ax.tick_params(labelbottom=False)
        ax.set_xlabel("")
        strip.set_gid("ratio_strip")
    else:
        ax.set_xlabel("$m/z$")
    return df


def extract_xic(scans, target_mz: float, tol: float = 10.0, unit: str = "ppm"):
    """Extracted ion chromatogram from MS1 scans: [{rt, mz, intensity}] → (rt, intensity).
    Sums intensity within tolerance of target_mz in each scan."""
    rt, it = [], []
    w = target_mz * tol * 1e-6 if unit == "ppm" else tol
    for sc in scans:
        mz = np.asarray(sc["mz"], float); inten = np.asarray(sc["intensity"], float)
        rt.append(float(sc["rt"])); it.append(float(inten[np.abs(mz - target_mz) <= w].sum()))
    o = np.argsort(rt)
    return np.array(rt)[o], np.array(it)[o]


def _auc(x, y):
    return float(np.trapezoid(y, x) if hasattr(np, "trapezoid") else np.trapz(y, x))


def _apex_rt(x, y) -> float:
    """Apex retention time, refined by a parabola through the top three points
    (with 2–3 scans above half-maximum the raw maximum is a whole MS1 cycle coarse)."""
    j = int(np.argmax(y))
    if 0 < j < len(y) - 1:
        y0, y1, y2 = y[j - 1], y[j], y[j + 1]
        den = y0 - 2 * y1 + y2
        if den < 0:
            return float(x[j] + 0.5 * (y0 - y2) / den * (x[j + 1] - x[j - 1]) / 2)
    return float(x[j])


def xic_plot(ax, data, rt: str = "rt", intensity: str = "intensity", condition: str = "condition",
             replicate: str = "replicate", order=None, colors=None, window=None,
             test: str = "welch", log_auc: bool = True, normalise: bool = False,
             fill: bool = True, show_areas: bool = False, area_unit: str = "a.u. × min",
             align=None, warn: bool = True):
    """Extracted-ion chromatograms, treated vs control: each replicate thin, the
    condition mean bold, and the peak area (AUC) compared across replicates.

    data    long table: rt (min), intensity, condition, replicate.
    order   conditions, reference first (e.g. ["Control", "Treated"]).
    window  (rt_min, rt_max) for integration; dashed. Default: whole trace.
            With `align`, the window is in aligned time, i.e. it follows each run.
    test    "welch" t-test on log10 AUC (log_auc=True) or raw AUC; None to skip.
    colors  default for two conditions: control blue, treatment amber.
    fill    shade the area under each mean trace (inside the window).
    show_areas  print on the panel the integrated area per condition (mean ± SD
            across replicates, in the condition's colour) and the fold change
            with its P value: the XIC is the quantitative evidence.

    align   retention-time alignment of replicate runs before drawing the mean.
            Chromatography drifts between injections; when the drift is
            comparable to the peak width and the MS1 cycle is slow (2–3 scans
            above half-maximum is normal for DIA), averaging unaligned traces
            gives a mean LOWER than every replicate: an artefact.
            • {replicate: anchor_rt} (or {(condition, replicate): anchor_rt}):
              shift each run so its anchor lands on the median anchor. Use the
              retention time the search engine reported for this precursor in
              that run (DIA-NN, Spectronaut and Skyline normalise RT across
              runs). This is the correct choice for MS data: the anchor is
              independent of the trace being drawn.
            • "apex": align on each trace's own apex. Circular fallback for
              when no external anchor exists: it forces the replicates to agree
              and so flatters the mean; say so in the caption.
            A time shift leaves each trace's integral unchanged, so alignment
            only affects the drawn shape of the mean, never the areas, fold
            change or test. With a `window`, the window is in aligned time and so
            moves with each run; the panel note then states how much that changes
            the areas relative to a fixed window instead of claiming "unchanged".
            The drift before (range of trace apexes) and the residual after are
            printed on the panel. Residual misalignment and sparse sampling (each
            run samples its apex at a different phase) still pull the mean a
            little below the replicates; the note's residual lets the reader
            judge that.
    warn    when align is None, warn if a condition's mean apex falls below its
            lowest replicate apex (always a misleading panel).

    Returns dict(auc table, areas {condition: (mean, sd, n)}, fold, p,
    alignment {drift_s, residual_s, cycle_s, shifts} or None, artefact flags).
    """
    import pandas as pd
    import warnings
    from scipy import stats as st
    order = list(order or data[condition].dropna().unique())
    colors = colors or treat_control_colors(order)
    k = sizes()["stroke"]

    # ---- traces per run, with their own apex and sampling interval
    runs = []
    for c in order:
        for r, g in data[data[condition] == c].groupby(replicate, sort=False):
            g = g.sort_values(rt)
            x, y = g[rt].to_numpy(float), g[intensity].to_numpy(float)
            runs.append(dict(condition=c, replicate=r, x=x, y=y, apex=_apex_rt(x, y)))
    cycle_min = float(np.median([np.median(np.diff(r["x"])) for r in runs if len(r["x"]) > 1]))

    # ---- alignment
    alignment = None
    if align is not None:
        if isinstance(align, str):
            if align != "apex":
                raise ValueError('align must be a {replicate: anchor_rt} mapping or "apex"')
            anchors = [r["apex"] for r in runs]
            how = "each trace's own apex: a fallback that forces the runs to agree"
        else:
            anchors = []
            for r in runs:
                key = (r["condition"], r["replicate"])
                v = align.get(key, align.get(r["replicate"], align.get(str(r["replicate"]))))
                if v is None:
                    raise KeyError(f"no anchor retention time for run {key}; give "
                                   "{replicate: rt} or {(condition, replicate): rt}")
                anchors.append(float(v))
            how = "the reported retention time"
        target = float(np.median(anchors))
        for r, a in zip(runs, anchors):
            r["shift"] = target - a
            r["x_raw"] = r["x"]
            r["x"] = r["x"] + r["shift"]
        drift_s = (max(r["apex"] for r in runs) - min(r["apex"] for r in runs)) * 60
        aligned_apex = [r["apex"] + r["shift"] for r in runs]
        residual_s = (max(aligned_apex) - min(aligned_apex)) * 60
        alignment = dict(drift_s=drift_s, residual_s=residual_s, cycle_s=cycle_min * 60,
                         anchor=how, shifts={(r["condition"], r["replicate"]): r["shift"] * 60
                                             for r in runs})

    x_all = np.concatenate([r["x"] for r in runs])
    grid = np.linspace(x_all.min(), x_all.max(), 400)
    lo, hi = window if window else (grid[0], grid[-1])
    aucs, apex, rep_apex_h, flags = [], {}, {}, {}
    for c in order:
        traces = []
        for r in (r for r in runs if r["condition"] == c):
            ax.plot(r["x"], r["y"], color=colors[c], lw=0.55 * k, alpha=0.4, zorder=2)
            traces.append(np.interp(grid, r["x"], r["y"], left=0, right=0))
            m = (r["x"] >= lo) & (r["x"] <= hi)
            aucs.append(dict(condition=c, replicate=r["replicate"], auc=_auc(r["x"][m], r["y"][m])))
        mean = np.mean(traces, axis=0)
        if fill:
            inside = (grid >= lo) & (grid <= hi)
            ax.fill_between(grid[inside], 0, mean[inside], color=colors[c], alpha=0.14,
                            lw=0, zorder=1)
        ax.plot(grid, mean, color=colors[c], lw=1.6 * k, zorder=3)
        j = int(np.argmax(mean)); apex[c] = (grid[j], mean[j])
        rep_apex_h[c] = [float(r["y"].max()) for r in runs if r["condition"] == c]
        flags[c] = bool(mean[j] < min(rep_apex_h[c]))

    # ---- warn when an unaligned mean is lower than every replicate
    if align is None and warn:
        for c in order:
            if flags[c]:
                ap = [r["apex"] for r in runs if r["condition"] == c]
                spread_s = (max(ap) - min(ap)) * 60
                warnings.warn(
                    f"xic_plot: the {c} mean trace peaks at {_sci_plain(apex[c][1])}, below every "
                    f"replicate (lowest {_sci_plain(min(rep_apex_h[c]))}, median "
                    f"{_sci_plain(float(np.median(rep_apex_h[c])))}); replicate apexes spread "
                    f"{spread_s:.0f} s against a {cycle_min * 60:.1f} s MS1 cycle. The mean is an "
                    "artefact of averaging unaligned peaks: pass align={replicate: reported_rt} "
                    "(or align='apex' as a fallback).", UserWarning, stacklevel=2)

    if window:     # window marked by thin dashed edges (the fill already shows the area)
        for edge in (lo, hi):
            ax.axvline(edge, color=INK["ref"], lw=0.5, ls=(0, (3, 2)), zorder=0)
    A = pd.DataFrame(aucs)
    if alignment is not None:
        if window:     # same window in raw time: how much does letting it follow each run matter?
            raw = np.array([_auc(r["x_raw"][(r["x_raw"] >= lo) & (r["x_raw"] <= hi)],
                                 r["y"][(r["x_raw"] >= lo) & (r["x_raw"] <= hi)]) for r in runs])
            alignment["window_area_change"] = float(np.max(np.abs(A["auc"].to_numpy() - raw)
                                                           / np.maximum(raw, 1e-12)))
        else:
            alignment["window_area_change"] = 0.0
    ref, trt = order[0], order[-1]
    a_ref = A.loc[A.condition == ref, "auc"].to_numpy(); a_trt = A.loc[A.condition == trt, "auc"].to_numpy()
    fold = float(np.mean(a_trt) / np.mean(a_ref)) if np.mean(a_ref) > 0 else np.inf
    p = np.nan
    if test and len(a_ref) > 1 and len(a_trt) > 1:
        f = (lambda v: np.log10(np.maximum(v, 1e-12))) if log_auc else (lambda v: v)
        p = float(st.ttest_ind(f(a_trt), f(a_ref), equal_var=False).pvalue)
    ax.set_xlabel("Retention time (min)" + (", aligned" if align is not None else ""))
    ax.set_ylabel("Intensity")
    ax.ticklabel_format(axis="y", style="sci", scilimits=(-3, 4), useMathText=True)
    y_data_max = float(max(r["y"].max() for r in runs))     # replicates, not the means
    ax.set_ylim(0, y_data_max * 1.05)
    if show_areas:     # fold and P are in the area box; the stat line says what was integrated
        stat_title(ax, f"$n$ = {len(a_ref)}/{len(a_trt)} replicates · integrated "
                       f"{lo:.1f}–{hi:.1f} min")
    else:
        stat_title(ax, f"AUC {fold:.1f}×" + (f" · {fmt_p(p)}" if np.isfinite(p) else "")
                   + f" · $n$ = {len(a_ref)}/{len(a_trt)}")
    areas = {c: (float(A.loc[A.condition == c, "auc"].mean()),
                 float(A.loc[A.condition == c, "auc"].std(ddof=1)) if (A.condition == c).sum() > 1
                 else float("nan"),
                 int((A.condition == c).sum())) for c in order}
    fig = ax.figure
    ax_h_pt = ax.get_position().height * fig.get_figheight() * 72
    ax_w_pt = ax.get_position().width * fig.get_figwidth() * 72
    fs = sizes()["annot"]
    fs_note = max(sizes()["min"], fs - 1)
    n_box = (len(order) + 2) if show_areas else 0
    note_lines = []
    if alignment:
        cyc = alignment["cycle_s"]
        if align == "apex":
            res_txt = "residual 0 s by construction"
        else:
            res_txt = (f"residual {alignment['residual_s']:.0f} s"
                       + (" ≈ one MS1 cycle" if alignment["residual_s"] <= 1.5 * cyc else
                          f" ≈ {alignment['residual_s'] / cyc:.1f} MS1 cycles"))
        dA = alignment["window_area_change"]
        area_txt = ("areas are unchanged by the shift" if dA < 1e-3 else
                    f"the window follows each run (areas within {dA:.1%} of a fixed window)")
        note = (f"Runs aligned on {alignment['anchor']}: drift {alignment['drift_s']:.0f} s "
                f"between runs, {res_txt}; {area_txt}.")
        note_lines = _wrap_to_width(note, fs_note, 0.94 * ax_w_pt)
    n_note = len(note_lines)
    need_pt = (n_box * fs + n_note * fs_note) * 1.45 + (8 if (n_box or n_note) else 0)
    if need_pt:        # headroom above the TALLEST plotted trace for the box and note
        frac = min(0.6, need_pt / max(ax_h_pt, 1))
        ax.set_ylim(0, y_data_max * 1.05 / (1 - frac))
    if show_areas:
        _xic_area_box(ax, order, areas, colors, fold, p, area_unit, apex)
    if note_lines:
        right = (np.mean([apex[c][0] for c in order]) - ax.get_xlim()[0]) / \
            np.diff(ax.get_xlim())[0] < 0.5
        xa, ha = (0.98, "right") if right else (0.02, "left")
        for i, t in enumerate(note_lines):
            ax.annotate(t, xy=(xa, 0.97), xycoords="axes fraction",
                        xytext=(0, -(n_box * fs + i * fs_note) * 1.45 - (4 if n_box else 0)),
                        textcoords="offset points", ha=ha, va="top",
                        fontsize=fs_note, color=INK["muted"])
    for c in order:        # condition names beside each mean apex, in their colour
        label_points(ax, [apex[c][0]], [apex[c][1]], [c], color=colors[c], drop_crowded=False)
    return dict(auc=A, areas=areas, fold=fold, p=p, alignment=alignment,
                mean_below_replicates=flags)


def _wrap_to_width(text: str, size: float, width_pt: float) -> list:
    """Greedy word wrap so each line prints no wider than width_pt."""
    words, lines, cur = text.split(), [], ""
    for w in words:
        trial = f"{cur} {w}".strip()
        if cur and _text_width_pt(trial, size) > width_pt:
            lines.append(cur); cur = w
        else:
            cur = trial
    if cur:
        lines.append(cur)
    return lines


def _sci_parts(v: float):
    e = int(np.floor(np.log10(abs(v)))) if v else 0
    return v / 10 ** e, e


def _sci_txt(v: float, digits: int = 2, exp: int | None = None) -> str:
    """Number as mathtext: 3.9 × 10$^{6}$ (or plain for 0.01–9999). `exp` forces
    the exponent, so a mean and its SD can share one."""
    if not np.isfinite(v):
        return "n.a."
    if v == 0:
        return "0"
    e = _sci_parts(v)[1] if exp is None else exp
    if exp is None and -2 <= e <= 3:
        dec = max(0, digits - 1 - e)
        return f"{v:.{dec}f}"
    return f"{v / 10 ** e:.{max(0, digits - 1)}f} × 10$^{{{e}}}$"


def _sci_mean_sd(m: float, sd: float, digits: int = 2) -> str:
    """'8.2 ± 0.84 × 10$^{4}$': mean and SD on one exponent, same formatter."""
    if not np.isfinite(m):
        return "n.a."
    e = _sci_parts(m)[1]
    if not np.isfinite(sd):
        return _sci_txt(m, digits)
    if -2 <= e <= 3:                                   # plain numbers, same decimals
        dec = max(0, digits - 1 - e)
        return f"{m:.{dec}f} ± {sd:.{dec}f}"
    mm, ss = m / 10 ** e, sd / 10 ** e
    dec = max(digits - 1, 0)
    sd_dec = max(dec, -int(np.floor(np.log10(ss))) + 1) if ss > 0 else dec
    return f"{mm:.{dec}f} ± {ss:.{sd_dec}f} × 10$^{{{e}}}$"


def _sci_plain(v: float) -> str:
    """Plain-text scientific for warnings/captions: 8.2e4 → '8.2 × 10^4'."""
    if not np.isfinite(v) or v == 0:
        return f"{v:g}"
    m, e = _sci_parts(v)
    return f"{v:.3g}" if -2 <= e <= 3 else f"{m:.2g} × 10^{e}"


def _xic_area_box(ax, order, areas, colors, fold, p, unit, apex):
    """Area per condition and fold change, stacked in the corner away from the peak."""
    fs = sizes()["annot"]
    x0, x1 = ax.get_xlim()
    apex_x = np.mean([apex[c][0] for c in order])
    right = (apex_x - x0) / (x1 - x0) < 0.5            # peak on the left → box on the right
    xa, ha = (0.98, "right") if right else (0.02, "left")
    lines = [(f"Peak area ({unit}, mean ± SD)", INK["muted"], "normal")]
    for c in order:
        m, sd, n = areas[c]
        lines.append((f"{c}: {_sci_mean_sd(m, sd)}", colors[c], "bold"))
    lines.append((f"Fold {fold:.1f}×" + (f", {fmt_p(p)}" if np.isfinite(p) else ""),
                  INK["text"], "bold"))
    for k, (txt, col, wt) in enumerate(lines):
        ax.annotate(txt, xy=(xa, 0.97), xycoords="axes fraction",
                    xytext=(0, -k * fs * 1.45), textcoords="offset points",
                    ha=ha, va="top", fontsize=fs, color=col, fontweight=wt)


def site_occupancy(data, site: str = "site", condition: str = "condition",
                   replicate: str = "replicate", modified: str = "modified",
                   unmodified: str = "unmodified"):
    """% modification per site and replicate: modified / (modified + unmodified) × 100."""
    d = data.copy()
    tot = d[modified] + d[unmodified]
    d["percent"] = np.where(tot > 0, d[modified] / tot * 100, np.nan)
    return d[[site, condition, replicate, "percent"]]


def _group_summaries(data, x, y, hue, order, hue_order, error):
    """Mean, error and data extent per (group, hue): computed before any drawing."""
    out = {}
    for g in order:
        for h in hue_order:
            v = data.loc[(data[x] == g) & (data[hue] == h), y].dropna().to_numpy()
            if not len(v):
                continue
            m, ci = mean_ci(v)
            err = {"ci": ci, "sd": v.std(ddof=1) if len(v) > 1 else 0,
                   "sem": v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0}[error]
            err = err if np.isfinite(err) else 0.0
            out[(g, h)] = dict(v=v, m=m, err=err, top=max(float(np.nanmax(v)), m + err),
                               bottom=min(float(np.nanmin(v)), m - err))
    return out


def needs_break(data, x: str, y: str, hue: str | None = None, break_ratio: float | None = 5.0):
    """True when the largest category mean exceeds `break_ratio` × the next one,
    i.e. one category would set the scale and flatten the rest. Returns
    (bool, ratio, dominant category)."""
    if not break_ratio:
        return False, float("nan"), None
    keys = [x] + ([hue] if hue else [])
    gm = data.groupby(keys)[y].mean().groupby(level=0).max().sort_values(ascending=False)
    if len(gm) < 2 or gm.iloc[1] <= 0:
        return False, float("nan"), None
    ratio = float(gm.iloc[0] / gm.iloc[1])
    return ratio > break_ratio, ratio, gm.index[0]


def broken_axes(fig, spec, heights=(1.0, 2.4), hspace: float = 0.0):
    """Two axes sharing x for a broken y-axis, built as part of the layout.

    spec: a SubplotSpec (e.g. gs[2, :] or a mosaic cell's spec): create this
    BEFORE plotting; splitting an existing axes afterwards makes
    constrained_layout collapse. hspace is the sub-grid's own spacing and is
    honoured by constrained_layout: keep it 0: the layout padding alone leaves
    a ~4 mm gap for the cut marks; the old 0.08 opened 10–20 mm and squashed
    the lower segment. Returns (upper, lower).
    """
    sub = spec.subgridspec(2, 1, height_ratios=list(heights), hspace=hspace)
    lower = fig.add_subplot(sub[1])
    upper = fig.add_subplot(sub[0], sharex=lower)
    upper._pf_broken_upper, lower._pf_broken_lower = True, True
    upper._pf_pair, lower._pf_pair = lower, upper
    for a in (upper, lower):
        tidy_ticks(a)
    return upper, lower


def _fixed_vgap(fig, gap_mm: float = 2.0):
    """Extra vertical spacing between panels as a physical distance rather than a
    fraction of the figure height (constrained_layout's hspace)."""
    eng = fig.get_layout_engine()
    if eng is not None and hasattr(eng, "set"):
        eng.set(hspace=gap_mm / (fig.get_figheight() * 25.4))


def broken_subplots(journal: str | None = None, width: str | float = "single",
                    aspect: float = 0.75, heights=(1.0, 2.4)):
    """A figure holding one broken-axis pair. Returns (fig, (upper, lower))."""
    fig = plt.figure(figsize=figsize(journal, width, aspect))
    _fixed_vgap(fig, 2.0)
    gs = fig.add_gridspec(1, 1)
    return fig, broken_axes(fig, gs[0], heights)


def _cut_marks(upper, lower):
    """Conventional // marks where the y-axis is cut (left spine, both segments)."""
    kw = dict(marker=[(-1, -0.6), (1, 0.6)], markersize=5, linestyle="none",
              color=INK["axis"], mec=INK["axis"], mew=0.8 * sizes()["stroke"], clip_on=False,
              zorder=10)
    upper.plot([0], [0], transform=upper.transAxes, **kw)
    lower.plot([0], [1], transform=lower.transAxes, **kw)


def _centre_pair_ylabel(upper, lower, text):
    """y-axis title centred across both segments (after layout)."""
    lower.set_ylabel(text)
    upper.set_ylabel("")

    def job(up=upper, lo=lower):
        fig = lo.figure
        r = fig.canvas.get_renderer()
        lb, ub = lo.get_window_extent(r), up.get_window_extent(r)
        lab = lo.yaxis.label
        ext = lab.get_window_extent(r)
        # a y-title is anchored at its right edge (va="bottom" before the 90° turn)
        x_ax = (ext.x1 - lb.x0) / lb.width
        y_ax = ((ub.y1 + lb.y0) / 2 - lb.y0) / lb.height
        lo.yaxis.set_label_coords(x_ax, y_ax)
    lower.figure.__dict__.setdefault("_pf_jobs", []).append(job)


def grouped_bar_points(ax, data, x: str, y: str, hue: str, order=None, hue_order=None,
                       colors=None, error: str = "ci", test: str | None = "welch",
                       stars: bool = False, width: float = 0.78, point_size: float = 8,
                       seed: int = 0, rotate_labels="auto", break_ratio: float | None = 5.0,
                       ylabel: str | None = None):
    """Grouped bars (mean) + every replicate + capped error bar, one group per x
    level, one bar per hue level: e.g. % oxidation per site, control vs treated.

    ax: an Axes, or an (upper, lower) pair from `pf.broken_axes` for a broken
    y-axis. When one category's mean exceeds `break_ratio` × the next (e.g.
    1,340 CPM against 114), a single linear axis flattens every other bar onto
    zero; draw into a broken pair instead: the upper segment covers the
    dominant bar(s), the lower is scaled to the rest, bars stay anchored at
    zero (unlike a log scale), cut marks sit on the spine, one legend, the
    y-title is centred across the pair (pass `ylabel`), and each P value goes
    on the segment that holds its group's top. With a single Axes the function
    only prints the advice (splitting afterwards breaks the layout).
    test: "welch", "mannwhitney", or None (nothing written) between the first
    two hue levels at each x, BH-adjusted across x.
    rotate_labels: "auto" turns long category labels vertical when they collide.
    Returns a stats DataFrame (attrs["break"] = ratio, dominant, broken).
    """
    if test not in (None, "welch", "mannwhitney"):
        raise ValueError(f"test must be 'welch', 'mannwhitney' or None, not {test!r}")
    import pandas as pd
    from scipy import stats as st
    pair = isinstance(ax, (tuple, list))
    axes = list(ax) if pair else [ax]
    order = list(order or data[x].dropna().unique())
    hue_order = list(hue_order or data[hue].dropna().unique())
    colors = colors or group_colors(hue_order)
    nh = len(hue_order)
    bw = width / nh
    summ = _group_summaries(data, x, y, hue, order, hue_order, error)
    tops = {g: max([summ[(g, h)]["top"] for h in hue_order if (g, h) in summ] or [0.0])
            for g in order}
    do_break, ratio, dominant = needs_break(data[data[x].isin(order)], x, y, hue, break_ratio)
    if do_break and not pair:
        print(f"[pubfig] grouped_bar_points: '{dominant}' is {ratio:.1f}× the next category, so "
              "the other bars sit near zero. Build the panel with "
              "pf.broken_axes(fig, spec) and pass (upper, lower) to break the axis.")
    # ---- draw the same bars into every segment (each clips to its own range)
    for axi in axes:
        rng = np.random.default_rng(seed)
        for gi, g in enumerate(order):
            for hi, h in enumerate(hue_order):
                if (g, h) not in summ:
                    continue
                d = summ[(g, h)]
                xc = gi - width / 2 + bw * (hi + 0.5)
                axi.bar(xc, d["m"], width=bw * 0.9, color=colors[h], alpha=0.28, lw=0, zorder=1)
                axi.plot([xc - bw * 0.45, xc + bw * 0.45], [d["m"], d["m"]], color=colors[h],
                         lw=MEAN_LINE_LW * sizes()["stroke"], solid_capstyle="butt", zorder=2)
                axi.scatter(xc + rng.uniform(-bw * 0.22, bw * 0.22, len(d["v"])), d["v"],
                            s=point_size, color=colors[h], lw=0, alpha=0.9, zorder=3)
                axi.errorbar(xc, d["m"], yerr=d["err"], color=INK["text"], zorder=4,
                             **errorbar_style())
    # ---- statistics
    rows = []
    if test and nh >= 2:
        for g in order:
            a_ = data.loc[(data[x] == g) & (data[hue] == hue_order[0]), y].dropna()
            b_ = data.loc[(data[x] == g) & (data[hue] == hue_order[1]), y].dropna()
            if len(a_) > 1 and len(b_) > 1:
                p = float(st.ttest_ind(b_, a_, equal_var=False).pvalue if test == "welch"
                          else st.mannwhitneyu(b_, a_, alternative="two-sided").pvalue)
            else:
                p = np.nan
            rows.append(dict(**{x: g}, mean_ref=a_.mean(), mean_trt=b_.mean(),
                             diff=b_.mean() - a_.mean(), p=p))
    S = pd.DataFrame(rows)
    if len(S) and S.p.notna().any():
        S["p_adj"] = np.nan
        ok = S.p.notna()
        S.loc[ok, "p_adj"] = bh_fdr(S.loc[ok, "p"].to_numpy())
    # ---- limits
    if pair:
        upper, lower = axes
        dom = {g for g in order if tops[g] > 0 and
               max(summ[(g, h)]["m"] for h in hue_order if (g, h) in summ) >= _dominant_floor(summ, order, hue_order)}
        rest_top = max([tops[g] for g in order if g not in dom] or [1.0])
        lo_hi = rest_top * 1.22
        dom_bottom = min(summ[(g, h)]["bottom"] for g in dom for h in hue_order if (g, h) in summ)
        up_lo = max(lo_hi * 1.05, dom_bottom * 0.9)
        up_hi = max(tops[g] for g in dom) * 1.2
        lower.set_ylim(0, lo_hi)
        upper.set_ylim(up_lo, up_hi)
        upper.spines["bottom"].set_visible(False)
        upper.tick_params(axis="x", which="both", bottom=False, labelbottom=False)
        upper.yaxis.set_major_locator(mpl.ticker.MaxNLocator(nbins=2, steps=[1, 2, 2.5, 5, 10],
                                                             min_n_ticks=2))
        _cut_marks(upper, lower)
        seg_of = {g: (upper if g in dom else lower) for g in order}
        span_of = {id(upper): up_hi - up_lo, id(lower): lo_hi}
    else:
        ymax = max(tops.values()) if tops else 1
        seg_of = {g: axes[0] for g in order}
        span_of = {id(axes[0]): ymax * 1.18}
        axes[0].set_ylim(0, ymax * 1.18)
    if len(S) and "p_adj" in S:
        for gi, g in enumerate(order):
            r = S[S[x] == g]
            if r.empty or not np.isfinite(r.p_adj.iloc[0]):
                continue
            q = r.p_adj.iloc[0]
            txt = ("***" if q < 1e-3 else "**" if q < 1e-2 else "*" if q < 0.05 else "n.s.") if stars \
                else fmt_p(q).replace("$P$ = ", "").replace("$P$ ", "")
            axi = seg_of[g]
            axi.text(gi, tops[g] + 0.035 * span_of[id(axi)], txt, ha="center", va="bottom",
                     fontsize=sizes()["annot"], color=INK["muted"])
    base = axes[-1]
    base.set_xticks(range(len(order)), order)
    base.set_xlim(-0.6, len(order) - 0.4)
    categorical_axis(base, "x", rotate=rotate_labels)
    top_ax = axes[0]
    top_ax.legend(handles=[mpl.patches.Patch(facecolor=colors[h], label=str(h)) for h in hue_order],
                  loc="lower right", bbox_to_anchor=(1, 1.0), ncols=nh, borderaxespad=0)
    if pair:
        _centre_pair_ylabel(axes[0], axes[1], ylabel or y)
    elif ylabel:
        axes[0].set_ylabel(ylabel)
    S.attrs["break"] = dict(ratio=ratio, dominant=dominant, broken=pair)
    return S


def _dominant_floor(summ, order, hue_order):
    """Smallest group mean still in the upper (dominant) segment: the sorted group
    means are split at their largest consecutive ratio."""
    gm = sorted((max(summ[(g, h)]["m"] for h in hue_order if (g, h) in summ) for g in order
                 if any((g, h) in summ for h in hue_order)), reverse=True)
    if len(gm) < 2:
        return gm[0] if gm else 0
    # split at the largest ratio between consecutive sorted means
    ratios = [gm[i] / max(gm[i + 1], 1e-12) for i in range(len(gm) - 1)]
    cut = int(np.argmax(ratios))
    return gm[cut]


# --------------------------------------------------------------------------
# Proteogenomics: peptide → sequence → transcript model
# --------------------------------------------------------------------------
TX_COLORS = {"reference": "#8A93A0", "novel": "#1B2A49", "peptide": "#D1782A",
             "retained": "#E3A33B", "exon": "#C9D3E0", "intron": "#F1F3F6"}
CODON = {}
_b = "TCAG"
_aa = "FFLLSSSSYY**CC*WLLLLPPPPHHQQRRRRIIIMTTTTNNKKSSRRVVVVAAAADDEEGGGG"
for _i, _c in enumerate((a + b + c) for a in _b for b in _b for c in _b):
    CODON[_c] = _aa[_i]


def revcomp(seq: str) -> str:
    return seq.upper().translate(str.maketrans("ACGTUN", "TGCAAN"))[::-1]


def translate(nt: str, stop: str = "*") -> str:
    nt = nt.upper().replace("U", "T")
    return "".join(CODON.get(nt[i:i + 3], "X") for i in range(0, len(nt) - 2, 3)).replace("*", stop)


def read_fasta(path) -> dict:
    """FASTA → {header: sequence}. A header like 'chr7:1000000-1012000' records
    the region offset (0-based start) in fasta_offset(header)."""
    out, name, buf = {}, None, []
    for line in Path(path).read_text().splitlines():
        if line.startswith(">"):
            if name:
                out[name] = "".join(buf).upper()
            name, buf = line[1:].split()[0], []
        elif line.strip():
            buf.append(line.strip())
    if name:
        out[name] = "".join(buf).upper()
    return out


def fasta_offset(header: str) -> tuple:
    """'chr7:1000001-1012000' → ('chr7', 1000000) (1-based inclusive header → 0-based)."""
    import re
    m = re.match(r"^([^:]+):(\d+)-(\d+)", header)
    if m:
        return m.group(1), int(m.group(2)) - 1
    return header, 0


def read_bed(path, kinds: dict | None = None, labels: dict | None = None) -> list:
    """BED12 (or BED6) transcript models → list of dicts.

    Each: name, chrom, strand, exons [(start, end)] 0-based half-open ascending,
    thick (start, end) or None, kind ('reference'/'novel', from `kinds` by name
    or guessed from the name), label (from `labels` {name: label}: e.g. map
    GENCODE display names like 'SNHG8-203' to accessions: else the name).
    """
    txs = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith(("#", "track", "browser")):
            continue
        f = line.split("\t") if "\t" in line else line.split()
        chrom, s0, e0, name = f[0], int(f[1]), int(f[2]), f[3] if len(f) > 3 else "tx"
        strand = f[5] if len(f) > 5 else "+"
        ts, te = (int(f[6]), int(f[7])) if len(f) > 7 else (s0, s0)
        if len(f) >= 12 and int(f[9]) > 0:
            sizes_ = [int(x) for x in f[10].strip(",").split(",")]
            starts = [int(x) for x in f[11].strip(",").split(",")]
            exons = [(s0 + st, s0 + st + sz) for st, sz in zip(starts, sizes_)]
        else:
            exons = [(s0, e0)]
        txs.append(_tx(name, chrom, strand, exons, (ts, te) if te > ts else None, kinds))
    if labels:
        for t in txs:
            t["label"] = labels.get(t["name"], t["label"])
    return txs


def read_gtf(path, transcript_ids=None, kinds: dict | None = None,
             labels: dict | None = None) -> list:
    """GTF/GFF2 transcript models (exon + CDS/start/stop_codon features) → list of dicts."""
    import re
    from collections import defaultdict
    ex, cds, meta = defaultdict(list), defaultdict(list), {}
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        f = line.split("\t")
        if len(f) < 9:
            continue
        m = re.search(r'transcript_id "([^"]+)"', f[8])
        if not m:
            continue
        tid = m.group(1)
        if transcript_ids and tid not in transcript_ids:
            continue
        s0, e0 = int(f[3]) - 1, int(f[4])
        meta[tid] = (f[0], f[6])
        if f[2] == "exon":
            ex[tid].append((s0, e0))
        elif f[2] in ("CDS", "start_codon", "stop_codon"):
            cds[tid].append((s0, e0))
    txs = []
    for tid, exons in ex.items():
        c = cds.get(tid)
        thick = (min(a for a, _ in c), max(b for _, b in c)) if c else None
        txs.append(_tx(tid, meta[tid][0], meta[tid][1], exons, thick, kinds))
    if labels:
        for t in txs:
            t["label"] = labels.get(t["name"], t["label"])
    return txs


def _tx(name, chrom, strand, exons, thick, kinds):
    kind = (kinds or {}).get(name)
    if kind is None:
        low = name.lower()
        kind = "novel" if any(k in low for k in ("novel", "pb.", "nic", "nnc", "isoseq", "flair",
                                                  "bambu", "stringtie", "talon")) else "reference"
    return dict(name=name, chrom=chrom, strand=strand, exons=sorted(exons), thick=thick,
                kind=kind, label=name)


def spliced(tx, genome: str, offset: int = 0) -> str:
    """Spliced transcript sequence 5'→3' (reverse-complemented on − strand)."""
    seq = "".join(genome[s - offset:e - offset] for s, e in tx["exons"])
    return revcomp(seq) if tx["strand"] == "-" else seq.upper()


def tx_genomic_positions(tx) -> list:
    """Genomic coordinate of each transcript nucleotide, 5'→3'."""
    pos = [p for s, e in tx["exons"] for p in range(s, e)]
    return pos[::-1] if tx["strand"] == "-" else pos


def locate_peptide(tx, genome: str, peptide: str, offset: int = 0):
    """Find the peptide in the transcript: annotated CDS frame first, then all 3 frames.

    Returns dict(tx_start, tx_end (nt, 0-based), frame, in_cds_frame,
    codons [[g1, g2, g3] …] genomic positions, segments [(gstart, gend)] merged,
    context_nt) or None.
    """
    seq = spliced(tx, genome, offset)
    gpos = tx_genomic_positions(tx)
    pep = peptide.upper()
    frames = [0, 1, 2]
    cds_frame = None
    if tx.get("thick"):
        ts, te = tx["thick"]
        start_g = ts if tx["strand"] == "+" else te - 1
        if start_g in gpos:
            cds_frame = gpos.index(start_g) % 3
            frames = [cds_frame] + [f for f in frames if f != cds_frame]
    for f in frames:
        aa = translate(seq[f:])
        k = aa.find(pep)
        if k >= 0:
            t0 = f + 3 * k
            t1 = t0 + 3 * len(pep)
            codons = [gpos[t0 + 3 * i: t0 + 3 * i + 3] for i in range(len(pep))]
            g = sorted(gpos[t0:t1])
            segs, a = [], g[0]
            for i in range(1, len(g)):
                if g[i] != g[i - 1] + 1:
                    segs.append((a, g[i - 1] + 1)); a = g[i]
            segs.append((a, g[-1] + 1))
            return dict(tx=tx["name"], tx_start=t0, tx_end=t1, frame=f,
                        in_cds_frame=(f == cds_frame), codons=codons, segments=segs,
                        peptide=pep)
    return None


def classify_origin(loc, reference, novel=None) -> str:
    """Origin of a located peptide relative to a reference transcript model, in
    the categories used for cryptic peptides: canonical, frameshift, 5′/3′ UTR
    (upstream/downstream), retained intron, novel exon, intergenic."""
    if loc is None:
        return "not found"
    gp = [p for c in loc["codons"] for p in c]
    ex = reference["exons"]
    in_exon = [any(s <= p < e for s, e in ex) for p in gp]
    span = (ex[0][0], ex[-1][1])
    if not any(span[0] <= p < span[1] for p in gp):
        return "intergenic"
    if not all(in_exon):
        in_intron = [not ie and span[0] <= p < span[1] for p, ie in zip(gp, in_exon)]
        if any(in_intron):
            if novel is not None and all(any(s <= p < e for s, e in novel["exons"])
                                         for p, ii in zip(gp, in_intron) if ii):
                covering = [(s, e) for s, e in novel["exons"]
                            if any(s <= p < e for p, ii in zip(gp, in_intron) if ii)]
                # retained intron: the novel exon spans an entire reference intron
                introns = list(zip([e for _, e in ex[:-1]], [s for s, _ in ex[1:]]))
                if any(cs <= i0 and ce >= i1 for cs, ce in covering for i0, i1 in introns):
                    return "retained intron"
                return "novel exon"
            return "intronic"
    th = reference.get("thick")
    if th:
        ts, te = th
        if all(ts <= p < te for p in gp):
            ref_loc_pos = tx_genomic_positions(reference)
            start_g = ts if reference["strand"] == "+" else te - 1
            f_ref = ref_loc_pos.index(start_g) % 3
            first = loc["codons"][0][0]
            idx = ref_loc_pos.index(first)
            return "canonical" if (idx - f_ref) % 3 == 0 else "frameshift"
        upstream = all(p < ts for p in gp) if reference["strand"] == "+" else all(p >= te for p in gp)
        return "5′ UTR (upstream)" if upstream else "3′ UTR (downstream)"
    return "non-coding (lncRNA)"


def _merged_blocks(exons) -> list:
    """This model's own exon blocks, sorted and merged where they overlap/touch."""
    out = []
    for s_, e_ in sorted(exons):
        if out and s_ <= out[-1][1]:
            out[-1] = (out[-1][0], max(out[-1][1], e_))
        else:
            out.append((s_, e_))
    return out


def transcript_track(ax, transcripts, region=None, highlight=None, colors=None,
                     label_rows: bool = True, chevrons: bool = True, exon_numbers: bool = False,
                     xlabel: str | None = None, peptide_line: bool = True, labels=None,
                     hollow_noncoding: bool = True, peptide: str | None = None,
                     key: bool = False):
    """Gene-model track: one row per transcript. Filled thick boxes = CDS;
    non-coding blocks (UTRs, and every block of a transcript with no CDS:
    BED thickStart == thickEnd) are hollow outlines, as genome browsers draw
    them, so at a lncRNA locus the ORF is the only filled feature. Introns are
    drawn as one line segment per gap between consecutive blocks: computed
    from THIS model's own merged exon list, never matched by endpoints (isoforms
    at a locus share first/last exons), so a single-exon model has no line
    and hollow boxes stay boxes. Chevrons give strand. Reference grey, novel navy.

    highlight: genomic segments [(start, end)] shaded across all rows (the
    peptide origin) and, in every model, coloured in the shape of whatever it
    lands on: filled at CDS height on a coding block, the outline's edges on a
    non-coding block, a coloured stretch of line on an intron, so the mark
    never reads as an exon of its own.
    peptide: the peptide sequence: labels the band ("LTAEKVRSV · 27 bp").
    peptide_line: thin vertical line through the span's centre across every
    model, beneath the model artists.
    hollow_noncoding: False fills every block (older look).
    exon_numbers: exon ordinal within each model (5′→3′) under each block.
    key: one-line key above the panel explaining the glyphs used.
    labels: {transcript name: display label}; the transcript dicts are unchanged.
    Returns the row y-positions.
    """
    colors = {**TX_COLORS, **(colors or {})}
    k = sizes()["stroke"]
    n = len(transcripts)
    allx = [p for t in transcripts for ex in t["exons"] for p in ex]
    x0, x1 = region or (min(allx), max(allx))
    pad = (x1 - x0) * 0.02
    ax.set_xlim(x0 - pad, x1 + pad)
    ys = []
    fs = sizes()["annot"]
    thin, thick = (0.26, 0.4) if hollow_noncoding else (0.16, 0.36)   # hollow boxes need height
    any_hollow = False
    for i, tx in enumerate(transcripts):
        y = -i
        ys.append(y)
        c = colors.get(tx["kind"], colors["reference"])
        blocks = _merged_blocks(tx["exons"])
        # intron line: one segment per gap between this model's own blocks
        for (s1, e1), (s2, e2) in zip(blocks[:-1], blocks[1:]):
            ax.plot([e1, s2], [y, y], color=c, lw=0.7 * k, zorder=1, solid_capstyle="butt")
            if chevrons:                               # strand arrows along the intron
                L = s2 - e1
                n_ch = int(L / ((x1 - x0) / 35))
                for j in range(1, n_ch + 1):
                    xc = e1 + L * j / (n_ch + 1)
                    d = (x1 - x0) * 0.004 * (1 if tx["strand"] == "+" else -1)
                    ax.plot([xc - d, xc, xc - d], [y + 0.07, y, y - 0.07], color=c,
                            lw=0.5 * k, zorder=1)
        th = tx.get("thick")
        if th and th[1] <= th[0]:
            th = None                                  # thickStart == thickEnd → non-coding
        for j, (s, e) in enumerate(blocks):
            segs = [(s, e, thin, False)]
            if th:
                ts, te = th
                segs = [(a_, b_, h, cds) for a_, b_, h, cds in (
                    (s, min(e, ts), thin, False), (max(s, ts), min(e, te), thick, True),
                    (max(s, te), e, thin, False)) if b_ > a_]
            for a_, b_, h, cds in segs:
                if cds or not hollow_noncoding:
                    ax.add_patch(mpl.patches.Rectangle((a_, y - h / 2), b_ - a_, h, facecolor=c,
                                                       edgecolor="none", zorder=2))
                else:
                    any_hollow = True
                    ax.add_patch(mpl.patches.Rectangle((a_, y - h / 2), b_ - a_, h,
                                                       facecolor="white", edgecolor=c,
                                                       lw=0.6 * k, zorder=2))
            if exon_numbers and x0 <= (s + e) / 2 <= x1:   # only exons inside the window
                num = j + 1 if tx["strand"] == "+" else len(blocks) - j
                ax.text((s + e) / 2, y - thick / 2 - 0.05, str(num), ha="center", va="top",
                        fontsize=max(sizes()["min"], fs - 1), color=INK["muted"], clip_on=True)
        if label_rows:
            lab = (labels or {}).get(tx["name"], tx["label"])
            ax.text(x0 - pad, y + thick / 2 + 0.06, lab, ha="left", va="bottom",
                    fontsize=fs, color=c if tx["kind"] == "novel" else INK["text"],
                    fontweight="bold" if tx["kind"] == "novel" else "normal")
        if highlight:
            _mark_peptide_on_model(ax, y, highlight, blocks, th, thin, thick, colors["peptide"],
                                   hollow_noncoding, min_w=(x1 - x0) * 0.004, k=k)
    span_lab = None
    if highlight:
        a_, b_ = min(h[0] for h in highlight), max(h[1] for h in highlight)
        w = max(b_ - a_, (x1 - x0) * 0.006)
        ax.axvspan(a_ - w * 0.5, b_ + w * 0.5, color=tint(colors["peptide"], 0.8), lw=0, zorder=0)
        if peptide_line:   # centre line through every model, under the model artists
            ax.axvline((a_ + b_) / 2, color=colors["peptide"], lw=0.7 * k, zorder=0.6,
                       solid_capstyle="butt", gid="peptide_line")
        nbp = sum(e - s for s, e in highlight)
        span_lab = f"{peptide} · {nbp} bp" if peptide else f"peptide · {nbp} bp"
        # label the band itself, just above the top row
        from matplotlib import patheffects as pe
        ax.annotate(span_lab, xy=((a_ + b_) / 2, 0.75), xycoords="data", xytext=(0, 1),
                    textcoords="offset points", ha="center", va="bottom", fontsize=fs,
                    color=darken(colors["peptide"], 0.8), fontweight="bold",
                    annotation_clip=False, zorder=6,
                    path_effects=[pe.withStroke(linewidth=2.6, foreground="white")])
    ax.set_ylim(-(n - 1) - (1.0 if exon_numbers else 0.6), 0.75 + (0.9 if highlight else 0))
    if key:
        _track_key(ax, colors, any_hollow, bool(highlight), exon_numbers, span_lab)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(lambda v, _: f"{v / 1e3:,.1f}"))
    chrom = transcripts[0]["chrom"]
    ax.set_xlabel(xlabel or f"{chrom} position (kb)")
    ax.set_gid("transcript_track")
    return ys


def _mark_peptide_on_model(ax, y, highlight, blocks, th, thin, thick, color, hollow, min_w, k):
    """Draw the peptide's span on one model in the SHAPE of what it lands on, so it
    never reads as an exon of its own: filled at CDS height on a coding block,
    the top and bottom edges of the outline on a non-coding block (no fill, so it
    stays an outline however narrow), and a coloured stretch of the intron line
    where it falls between blocks. Nothing outside the model."""
    lo_m, hi_m = blocks[0][0], blocks[-1][1]
    for a_, b_ in highlight:
        a_, b_ = max(a_, lo_m), min(b_, hi_m)
        if b_ <= a_:
            continue
        if b_ - a_ < min_w:                      # keep a 27 bp span visible at gene scale
            c_ = (a_ + b_) / 2
            a_, b_ = c_ - min_w / 2, c_ + min_w / 2
        # cut the span at block edges and at the CDS boundaries
        cuts = sorted({a_, b_} | {p for s_, e_ in blocks for p in (s_, e_) if a_ < p < b_}
                      | ({p for p in th if a_ < p < b_} if th else set()))
        for u, v in zip(cuts[:-1], cuts[1:]):
            mid = (u + v) / 2
            in_block = any(s_ <= mid < e_ for s_, e_ in blocks)
            if not in_block:                      # intron: colour the line only
                ax.plot([u, v], [y, y], color=color, lw=1.6 * k, zorder=3.2,
                        solid_capstyle="butt")
                continue
            coding = bool(th) and th[0] <= mid < th[1]
            h = thick if coding else thin
            if coding or not hollow:
                ax.add_patch(mpl.patches.Rectangle((u, y - h / 2), v - u, h, facecolor=color,
                                                   edgecolor="none", zorder=3))
            else:     # non-coding block: colour its top and bottom edges only, never a fill
                for yy in (y - h / 2, y + h / 2):
                    ax.plot([u, v], [yy, yy], color=color, lw=1.3 * k, zorder=3.1,
                            solid_capstyle="butt")


def _track_key(ax, colors, hollow, peptide, exon_numbers, span_lab):
    """One-line glyph key above a transcript track."""
    from matplotlib.patches import Rectangle
    from matplotlib.lines import Line2D
    c = colors["novel"]
    hs, ls_ = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="none")], ["CDS"]
    if hollow:
        hs.append(Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=c, lw=0.6))
        ls_.append("non-coding / UTR")
    if peptide:
        hs.append(Line2D([], [], color=colors["peptide"], lw=2.5))
        ls_.append("peptide-coding span, drawn in the shape of what it falls on "
                   "(not an exon, not the whole ORF)")
    if exon_numbers:
        hs.append(Line2D([], [], ls="none", marker="$3$", color=INK["muted"], markersize=4,
                         markeredgewidth=0))
        ls_.append("exon number within each model")
    ax.legend(hs, ls_, loc="lower left", bbox_to_anchor=(0, 1.0), ncols=len(hs),
              fontsize=max(sizes()["min"], sizes()["annot"] - 1), handlelength=1.4,
              handleheight=0.6, columnspacing=1.0, borderaxespad=0.1, frameon=False)


def sequence_zoom(ax, tx, genome: str, loc, reference=None, offset: int = 0,
                  flank_codons: int = 5, show_nt: bool = True, colors=None):
    """Codon-level zoom of where the peptide is encoded, in transcript orientation.

    Rows: region bar (reference exon / intron: e.g. 'exon 3' | 'retained intron'),
    nucleotides, codon ticks, amino acids (peptide boxed in amber). Returns the
    window as (tx_start, tx_end) and its genomic span.
    """
    colors = {**TX_COLORS, **(colors or {})}
    seq = spliced(tx, genome, offset)
    gpos = tx_genomic_positions(tx)
    f = loc["frame"]
    w0 = max(f, loc["tx_start"] - 3 * flank_codons)
    w0 -= (w0 - f) % 3
    w1 = min(len(seq) - (len(seq) - f) % 3, loc["tx_end"] + 3 * flank_codons)
    fs_nt = max(sizes()["min"], sizes()["tick"] - 1)
    fs_aa = sizes()["label"]
    k = sizes()["stroke"]
    ax.set_xlim(w0 - 0.5, w1 - 0.5)
    ax.set_ylim(-2.6, 1.9)
    ax.set_axis_off()
    # region bar: which part of the reference each nucleotide falls in
    if reference is not None:
        rex = reference["exons"]
        order = rex if reference["strand"] == "+" else rex[::-1]
        def region_of(g):
            for j, (s, e) in enumerate(order):
                if s <= g < e:
                    return f"exon {j + 1}"
            if rex[0][0] <= g < rex[-1][1]:
                return "retained intron" if any(s <= g < e for s, e in tx["exons"]) else "intron"
            return "outside reference"
        labs = [region_of(gpos[t]) for t in range(w0, w1)]
        start = w0
        for t in range(w0 + 1, w1 + 1):
            if t == w1 or labs[t - w0] != labs[t - 1 - w0]:
                lab = labs[start - w0]
                col = colors["retained"] if "intron" in lab else colors["exon"]
                ax.add_patch(mpl.patches.Rectangle((start - 0.5, 1.15), t - start, 0.38,
                                                   facecolor=col, edgecolor="white", lw=0.8))
                ax.text((start + t - 1) / 2, 1.34, lab, ha="center", va="center",
                        fontsize=fs_nt, color=INK["text"] if "intron" not in lab else darken(col, 0.55),
                        fontweight="bold")
                start = t
    # splice junctions inside the window (genomic jump between neighbours)
    for t in range(w0 + 1, w1):
        if abs(gpos[t] - gpos[t - 1]) != 1:
            ax.plot([t - 0.5, t - 0.5], [-2.3, 1.1], color=INK["muted"], lw=0.6,
                    ls=(0, (2, 2)))
            ax.text(t - 0.5, -2.45, "junction", ha="center", va="top", fontsize=fs_nt,
                    color=INK["muted"])
    # nucleotides
    if show_nt:
        for t in range(w0, w1):
            inpep = loc["tx_start"] <= t < loc["tx_end"]
            ax.text(t, 0.55, seq[t], ha="center", va="center", fontsize=fs_nt,
                    color=INK["text"] if inpep else INK["muted"], family="monospace")
    # codon ticks + amino acids
    for c0 in range(w0, w1 - 2, 3):
        ax.plot([c0 - 0.4, c0 + 2.4], [0.18, 0.18], color=INK["ref"], lw=0.5 * k)
        aa = CODON.get(seq[c0:c0 + 3], "X")
        inpep = loc["tx_start"] <= c0 < loc["tx_end"]
        if inpep:
            ax.add_patch(mpl.patches.FancyBboxPatch((c0 - 0.35, -0.95), 2.7, 0.9,
                                                    boxstyle="round,pad=0.02,rounding_size=0.25",
                                                    facecolor=tint(colors["peptide"], 0.8),
                                                    edgecolor=colors["peptide"], lw=0.6))
        ax.text(c0 + 1, -0.5, "stop" if aa == "*" else aa, ha="center", va="center",
                fontsize=fs_aa if aa != "*" else fs_nt,
                color=darken(colors["peptide"], 0.75) if inpep else INK["muted"],
                fontweight="bold" if inpep else "normal")
    # frame / orientation note
    ax.text(w0 - 0.5, -1.55, f"5′ … {tx['name']} (frame {loc['frame'] + 1}"
            f"{', annotated CDS frame' if loc['in_cds_frame'] else ', alternative frame'}) … 3′",
            ha="left", va="center", fontsize=fs_nt, color=INK["muted"])
    g = sorted(gpos[w0:w1])
    return (w0, w1), (g[0], g[-1] + 1)


def peptide_panel(ax, peptide: str, title: str = "", subtitle: str = "", colors=None):
    """The observed peptide as large boxed residues with positions: the start of the story."""
    colors = {**TX_COLORS, **(colors or {})}
    n = len(peptide)
    fig = ax.figure
    ax_w_pt = ax.get_position().width * fig.get_figwidth() * 72
    box_pt = 2.3 * (sizes()["label"] + 3)              # residue box ≈ square
    units = max(n, ax_w_pt / box_pt)                   # extra room left/right
    side = (units - n) / 2
    ax.set_xlim(-0.6 - side, n - 0.4 + side)
    ax.set_ylim(-0.9, 1.25)
    ax.set_axis_off()
    fs = sizes()["label"] + 3
    for i, a in enumerate(peptide):
        ax.add_patch(mpl.patches.FancyBboxPatch((i - 0.42, -0.42), 0.84, 0.84,
                                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                                facecolor=tint(colors["peptide"], 0.8),
                                                edgecolor=colors["peptide"], lw=0.7))
        ax.text(i, 0, a, ha="center", va="center", fontsize=fs, fontweight="bold",
                color=darken(colors["peptide"], 0.7))
        ax.text(i, -0.6, str(i + 1), ha="center", va="top", fontsize=sizes()["min"] + 0.5,
                color=INK["muted"])
    x_left, x_right = ax.get_xlim()
    if title:
        ax.text(x_left, 0.62, title, ha="left", va="bottom", fontsize=sizes()["label"],
                fontweight="bold", color=INK["text"])
    if subtitle:
        ax.text(x_right, 0.62, subtitle, ha="right", va="bottom", fontsize=sizes()["annot"],
                color=INK["muted"])
    ax.set_gid("peptide_panel")


def zoom_connect(ax_a, span_a, ax_b, span_b, color=None, fill: bool = True):
    """Zoom lines from a span on one axes (x data) to a span on another,
    bottom edge of the upper axes to top edge of the lower one."""
    from matplotlib.patches import ConnectionPatch
    color = color or TX_COLORS["peptide"]
    fig = ax_a.figure
    ya, yb = ax_a.get_position().y0, ax_b.get_position().y0
    upper, lower = (ax_a, ax_b) if ya >= yb else (ax_b, ax_a)
    su, sl = (span_a, span_b) if ya >= yb else (span_b, span_a)
    lines = []
    for xu, xl in zip(su, sl):
        cp = ConnectionPatch(xyA=(xu, 0), coordsA=upper.get_xaxis_transform(),
                             xyB=(xl, 1), coordsB=lower.get_xaxis_transform(),
                             color=color, lw=0.6, ls=(0, (3, 2)), zorder=0)
        cp.set_in_layout(False)
        fig.add_artist(cp); lines.append(cp)

    def _clear_labels(up=upper, lo=lower, cps=lines):
        """After layout: start below the upper panel's tick/axis labels and end
        above anything drawn on top of the lower panel."""
        r = fig.canvas.get_renderer()
        ub, lb = up.get_window_extent(r), lo.get_window_extent(r)
        ut, lt = up.get_tightbbox(r), lo.get_tightbbox(r)
        y_up = min(0.0, (ut.y0 - ub.y0) / ub.height - 0.02) if ut is not None else 0.0
        y_lo = max(1.0, (lt.y1 - lb.y0) / lb.height + 0.02) if lt is not None else 1.0
        for cp in cps:
            cp.xy1 = (cp.xy1[0], y_up)
            cp.xy2 = (cp.xy2[0], y_lo)
    fig.__dict__.setdefault("_pf_jobs", []).append(_clear_labels)
    return lines


def _event_region(ref, segments, pad_frac=0.08):
    """Genomic window around the peptide: the flanking reference exons and the
    intron(s) between them (the local 'event' view)."""
    a, b = min(s for s, _ in segments), max(e for _, e in segments)
    ex = ref["exons"]
    left = [s for s, e in ex if s <= a]
    right = [e for s, e in ex if e >= b]
    lo = max(left) if left else a
    hi = min(right) if right else b
    # include the whole exon on each side for context
    lo = min([s for s, e in ex if s <= lo <= e] or [lo])
    hi = max([e for s, e in ex if s <= hi <= e] or [hi])
    pad = (hi - lo) * pad_frac
    return lo - pad, hi + pad


def peptide_origin_figure(transcripts, genome: str, peptide: str, offset: int = 0,
                          reference: str | None = None, novel: str | None = None,
                          rna=None, rna_kw: dict | None = None, order: str = "peptide_first",
                          journal: str | None = None, width: str | float = "double",
                          flank_codons: int = 5, peptide_note: str = "", region=None,
                          local_zoom="auto", peptide_line: bool = True, labels: dict | None = None,
                          hollow_noncoding: bool = True, break_ratio: float | None = 5.0):
    """Composite: peptide → codon zoom → local event → whole transcript models (→ RNA).

    transcripts  list from read_bed / read_gtf (reference and novel models)
    genome       genomic sequence covering the models (offset = its 0-based start)
    reference / novel  names; default first 'reference'-kind / first isoform encoding the peptide
    rna          optional long table for the RNA panel (grouped_bar_points; see rna_kw)
    order        "peptide_first" (start at the peptide, zoom out) or "transcript_first"
    local_zoom   extra panel with just the flanking exons and intron(s). "auto" adds it
                 only when that window is ≥10× the codon panel's span (the codon panel
                 is already the close-up) and the gene is >2.5× wider than the window
    peptide_line thin vertical line through the peptide's span across every model in
                 the local view and the full track (in addition to the shaded band)
    labels       {transcript name: display label}, e.g. BED column-4 display names
                 ('SNHG8-203') → accessions ('ENST00000602520.8'); applied to the model
                 rows, the subtitle and matching RNA categories. The dicts from
                 read_bed/read_gtf are not modified.
    hollow_noncoding  non-coding blocks as outlines (the ORF is the only filled feature)
    break_ratio  break the RNA axis when one isoform's mean exceeds this × the next
                 (built into the layout; None keeps one linear axis)
    rna_kw       x, y, hue, hue_order, ylabel, and everything else grouped_bar_points
                 takes (test=None|"welch"|"mannwhitney", order, error, stars,
                 rotate_labels …): all forwarded. Feed abundances quantified against
                 the SAME assembly the models come from (see SKILL.md).
    Returns (fig, axes dict, info dict with origin category and coordinates).
    """
    byname = {t["name"]: t for t in transcripts}
    ref = byname[reference] if reference else next((t for t in transcripts if t["kind"] == "reference"),
                                                   transcripts[0])
    loc, src = None, None
    cands = [byname[novel]] if novel else [t for t in transcripts if t is not ref] + [ref]
    for t in cands:
        loc = locate_peptide(t, genome, peptide, offset)
        if loc:
            src = t
            break
    if loc is None:
        raise ValueError(f"peptide {peptide} is not encoded by any of the given transcript models "
                         "(checked the CDS frame and all three frames)")
    origin = classify_origin(loc, ref, src if src is not ref else None)
    labels = dict(labels or {})
    lab = lambda t: labels.get(t["name"], t.get("label", t["name"]))
    ev = _event_region(ref, loc["segments"])
    allx = [p for t in transcripts for e in t["exons"] for p in e]
    gene_span = region or (min(allx), max(allx))
    codon_span = 3 * (len(peptide) + 2 * flank_codons)          # nt shown in the codon panel
    local_ratio = (ev[1] - ev[0]) / max(codon_span, 1)
    if local_zoom == "auto":
        # the codon panel is already the close-up: add the local view only when it
        # is an order of magnitude wider than that AND the gene is much wider still
        local_zoom = local_ratio >= 10 and (gene_span[1] - gene_span[0]) > 2.5 * (ev[1] - ev[0])
    rows = ["pep", "zoom"] + (["local"] if local_zoom else []) + ["tx"]
    if order != "peptide_first":
        rows = rows[::-1]
    nt = len(transcripts)
    # row heights in mm (not shares of a total): fixed panels stay the same size
    # however many models the locus has, and tracks grow per model
    fs_mm = sizes()["annot"] * 25.4 / 72
    heights = {"pep": 15 + 2 * fs_mm, "zoom": 28 + 2 * fs_mm,
               "local": 3.6 * nt + 8 + 3 * fs_mm,           # no row labels: tighter
               "tx": (3.2 + 1.6 * fs_mm) * nt + 10 + 3 * fs_mm}   # a label above each row
    ratios = [heights[r] for r in rows]
    rna_break, rna_kw_ = False, dict(rna_kw or {})
    if rna is not None:
        bx, by, bh = (rna_kw_.get("x", "transcript"), rna_kw_.get("y", "value"),
                      rna_kw_.get("hue", "condition"))
        rna_break = needs_break(rna, bx, by, bh, rna_kw_.get("break_ratio", break_ratio))[0]
        ratios.append(70.0 if rna_break else 44.0)
    total_w = (JOURNALS[journal or _STATE["journal"]][width] if isinstance(width, str)
               else float(width))
    fig_h = sum(ratios) + 6
    fig = plt.figure(figsize=(total_w * MM, fig_h * MM))
    # keep the gap between panels a fixed physical distance whatever the figure height
    _fixed_vgap(fig, 2.0)
    gs = fig.add_gridspec(len(ratios), 1, height_ratios=ratios)
    axd = {r: fig.add_subplot(gs[i]) for i, r in enumerate(rows)}
    if rna is not None:                        # the split is part of the layout
        if rna_break:
            axd["rna_upper"], axd["rna"] = broken_axes(fig, gs[len(rows)], heights=(1.0, 1.9))
        else:
            axd["rna"] = fig.add_subplot(gs[len(rows)])
    for v in axd.values():
        tidy_ticks(v)
    peptide_panel(axd["pep"], peptide, title=f"{peptide_note + ' · ' if peptide_note else ''}"
                  f"{len(peptide)} aa", subtitle=f"origin: {origin} · {lab(src)}")
    (w0, w1), gspan = sequence_zoom(axd["zoom"], src, genome, loc, reference=ref, offset=offset,
                                    flank_codons=flank_codons)
    key_on = "local" if local_zoom else "tx"
    transcript_track(axd["tx"], transcripts, region=region, highlight=loc["segments"],
                     peptide_line=peptide_line, labels=labels, peptide=peptide,
                     hollow_noncoding=hollow_noncoding, key=(key_on == "tx"))
    if local_zoom:
        transcript_track(axd["local"], transcripts, region=ev, highlight=loc["segments"],
                         label_rows=False, exon_numbers=True, xlabel="",
                         peptide_line=peptide_line, labels=labels, peptide=peptide,
                         hollow_noncoding=hollow_noncoding, key=True)
        axd["local"].xaxis.set_major_locator(mpl.ticker.MaxNLocator(4))
        axd["local"].set_xlabel(f"{src['chrom']} (kb): local view", fontsize=sizes()["annot"],
                                color=INK["muted"])
    # zoom lines: peptide ↔ codons ↔ (local event ↔) whole gene
    zoom_connect(axd["pep"], (-0.42, len(peptide) - 0.58), axd["zoom"],
                 (loc["tx_start"] - 0.35, loc["tx_end"] - 0.65))
    if local_zoom:
        zoom_connect(axd["zoom"], (w0 - 0.5, w1 - 0.5), axd["local"], gspan)
        zoom_connect(axd["local"], axd["local"].get_xlim(), axd["tx"], ev)
    else:
        zoom_connect(axd["zoom"], (w0 - 0.5, w1 - 0.5), axd["tx"], gspan)
    info = dict(origin=origin, transcript=src["name"], transcript_label=lab(src),
                reference=ref["name"], reference_label=lab(ref),
                segments=loc["segments"], frame=loc["frame"], in_cds_frame=loc["in_cds_frame"],
                chrom=src["chrom"], strand=src["strand"], local_region=ev if local_zoom else None,
                local_ratio=local_ratio, rna_broken=rna_break,
                rows=rows + ((["rna_upper"] if rna_break else ["rna"]) if rna is not None else []))
    if rna is not None:
        kw = dict(x="transcript", y="value", hue="condition", break_ratio=break_ratio)
        kw.update(rna_kw_)
        x_col, y_col, hue_col = kw.pop("x"), kw.pop("y"), kw.pop("hue")
        ylabel = kw.pop("ylabel", "Abundance (TPM)")
        hue_order = kw.pop("hue_order", None) or list(dict.fromkeys(rna[hue_col]))
        kw.setdefault("colors", treat_control_colors(hue_order))
        rna_plot = rna
        if labels and rna[x_col].isin(labels.keys()).any():     # same names as the models
            rna_plot = rna.assign(**{x_col: rna[x_col].map(lambda v: labels.get(v, v))})
            if kw.get("order"):
                kw["order"] = [labels.get(v, v) for v in kw["order"]]
        target = (axd["rna_upper"], axd["rna"]) if rna_break else axd["rna"]
        info["rna_stats"] = grouped_bar_points(target, rna_plot, x_col, y_col, hue_col,
                                               hue_order=hue_order, ylabel=ylabel, **kw)
        axd["rna"].set_xlabel("")
    return fig, axd, info


# --------------------------------------------------------------------------
# Dose–response, IC50 and drug combinations
# --------------------------------------------------------------------------
def normalize_response(data, response: str = "response", by=None, vehicle=None,
                       positive=None, conc: str = "conc", out: str = "viability"):
    """Response as % of vehicle control (100) and, if given, a positive control (0).

    by: grouping columns (e.g. ["cell_line", "drug", "plate"]): controls are matched
    within each group. vehicle/positive: (column, value) selecting the control rows,
    or None to use the lowest concentration as vehicle.
    State the normalisation in the caption: an IC50 only means something with it.
    """
    import pandas as pd
    d = data.copy()
    by = list(by or [])
    def scale(g):
        v = (g.loc[g[vehicle[0]] == vehicle[1], response].mean() if vehicle
             else g.loc[g[conc] == g[conc].min(), response].mean())
        p = g.loc[g[positive[0]] == positive[1], response].mean() if positive else 0.0
        g[out] = (g[response] - p) / (v - p) * 100
        return g
    return (d.groupby(by, group_keys=False).apply(scale) if by else scale(d))


def fit_dose_response(conc, response, fix_bottom: float | None = None,
                      fix_top: float | None = None, n_boot: int = 1000, seed: int = 0,
                      absolute_level: float = 50.0):
    """Four-parameter logistic fit with both IC50 definitions and their CIs.

    Returns dict with
      ic50_rel   relative IC50 / EC50: the curve's inflection, halfway between the
                 fitted bottom and top. This is what most software prints.
      ic50_abs   absolute IC50: where the curve crosses `absolute_level`
                 (50% of control on a viability scale). NaN when the curve never
                 gets there (`reaches_level` False): then report "> max tested".
      emax       fitted response at the top concentration (the plateau reached),
      hill       Hill slope, span, r2, aic, and predict(xx) -> (fit, lo, hi).
    The two IC50s coincide only when the curve spans 0–100; they diverge exactly
    when Emax is partial: which is why Emax and Hill belong beside any IC50.
    fix_bottom / fix_top constrain the plateaus (e.g. fix_top=100 for viability
    normalised to vehicle); say so when you do.
    """
    from scipy.optimize import curve_fit
    x = np.asarray(conc, float); y = np.asarray(response, float)
    ok = (x > 0) & np.isfinite(y) & np.isfinite(x)
    lx, y = np.log10(x[ok]), y[ok]
    if len(y) < 4:
        raise ValueError("need at least 4 usable points (positive concentration, finite response)")
    desc = y[np.argsort(lx)][0] > y[np.argsort(lx)][-1]        # decreasing = inhibition

    def model(lx_, bottom, top, logic50, hill):
        return bottom + (top - bottom) / (1 + 10 ** ((logic50 - lx_) * hill))

    free = [fix_bottom is None, fix_top is None, True, True]
    p_full = [y.min() if fix_bottom is None else fix_bottom,
              y.max() if fix_top is None else fix_top, np.median(lx), -1.0 if desc else 1.0]

    def wrap(lx_, *free_p):
        p = list(p_full)
        it = iter(free_p)
        for i, is_free in enumerate(free):
            if is_free:
                p[i] = next(it)
        return model(lx_, *p)

    p0 = [p_full[i] for i, f in enumerate(free) if f]
    popt, pcov = curve_fit(wrap, lx, y, p0=p0, maxfev=40000)
    p = list(p_full)
    it = iter(popt)
    for i, is_free in enumerate(free):
        if is_free:
            p[i] = next(it)
    bottom, top, logic50, hill = p
    resid = y - model(lx, *p)
    ss_res, ss_tot = float(resid @ resid), float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    k = len(popt)
    aic = len(y) * np.log(max(ss_res, 1e-12) / len(y)) + 2 * k
    rng = np.random.default_rng(seed)
    draws_free = rng.multivariate_normal(popt, pcov, n_boot, check_valid="ignore")

    def full(d):
        q = list(p_full)
        jt = iter(d)
        for i, is_free in enumerate(free):
            if is_free:
                q[i] = next(jt)
        return q

    draws = [full(d) for d in draws_free]

    def predict(xx):
        lxx = np.log10(np.asarray(xx, float))
        curves = np.array([model(lxx, *d) for d in draws])
        return model(lxx, *p), np.nanpercentile(curves, 2.5, 0), np.nanpercentile(curves, 97.5, 0)

    def abs_ic50(params, level):
        b, t, l50, h = params
        if (level - b) * (t - level) <= 0:          # level outside the fitted range
            return np.nan
        return 10 ** (l50 - np.log10((t - level) / (level - b)) / h)

    x_top = 10 ** lx.max()
    emax = model(np.log10(x_top), *p)
    ic_abs = abs_ic50(p, absolute_level)
    boot_rel = np.array([10 ** d[2] for d in draws])
    boot_abs = np.array([abs_ic50(d, absolute_level) for d in draws])
    pct = lambda v: (float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))) \
        if np.isfinite(v).sum() > 10 else (np.nan, np.nan)
    return dict(bottom=bottom, top=top, hill=hill, span=top - bottom,
                ic50_rel=10 ** logic50, ic50_rel_ci=pct(boot_rel),
                ic50_abs=ic_abs, ic50_abs_ci=pct(boot_abs),
                reaches_level=bool(np.isfinite(ic_abs)), level=absolute_level,
                emax=float(emax), r2=float(r2), aic=float(aic), n=int(len(y)),
                max_conc=float(x_top), min_conc=float(10 ** lx.min()),
                decreasing=bool(desc), predict=predict, params=p)


def fmt_conc(v: float, unit: str = "nM") -> str:
    """Concentration for a label: 3 significant figures, thousands separated
    (1,420 nM, not 1.42e+03), scientific only below 0.01."""
    if not np.isfinite(v):
        return "n.d."
    if v >= 1000:
        return f"{float(f'{v:.3g}'):,.0f} {unit}"
    if v >= 0.01:
        return f"{v:.3g} {unit}"
    return f"{v:.2e} {unit}"


def dose_response(ax, data, conc: str = "conc", response: str = "response",
                  group: str | None = None, order=None, colors=None, unit: str = "nM",
                  fix_bottom=None, fix_top=None, band: bool = True, points: str = "mean",
                  error: str = "sem", ic50: str = "absolute", mark_ic50: bool = True,
                  annotate: bool = True, annotate_loc: str = "auto", level: float = 50.0,
                  legend: bool = True,
                  ylabel: str = "Viability (% of vehicle)", seed: int = 0):
    """Dose–response curves with 4PL fits, one curve per `group` (drug, cell line …).

    points  "mean" (mean ± `error` across replicates at each concentration) or "all".
    ic50    which IC50 to mark and annotate: "absolute" (crosses `level`, the
            honest default on a normalised viability scale), "relative"
            (curve inflection) or "both".
    mark_ic50  drop a line from the curve to the x-axis at that concentration.
    annotate   one line per group: IC50 (95% CI), Emax and Hill slope: an IC50
            without Emax and slope is not interpretable.
    annotate_loc  corner for that block: "auto" puts it where the curves leave
            room (top right for inhibition, bottom right for activation), or
            "upper left" / "lower left" / "upper right" / "lower right".
    Returns {group: fit dict} from fit_dose_response.
    """
    import pandas as pd
    groups = list(order or (data[group].dropna().unique() if group else ["all"]))
    colors = colors or (group_colors(groups) if group else
                        {"all": PALETTES["okabe_ito"][0]})
    k = sizes()["stroke"]
    fits, rows = {}, []
    xall = data[conc][data[conc] > 0]
    xx = np.logspace(np.log10(xall.min()), np.log10(xall.max()), 300)
    for g in groups:
        d = data if group is None else data[data[group] == g]
        d = d[d[conc] > 0]
        c = colors[g]
        if points == "all":
            ax.scatter(d[conc], d[response], s=6, color=c, alpha=0.6, lw=0, zorder=3)
        else:
            m = d.groupby(conc)[response].agg(["mean", "std", "count"])
            err = m["std"] / np.sqrt(m["count"]) if error == "sem" else m["std"]
            ax.errorbar(m.index, m["mean"], yerr=err.fillna(0), fmt="o", ms=3, color=c,
                        lw=0, zorder=3, **errorbar_style())
        try:
            fit = fit_dose_response(d[conc], d[response], fix_bottom=fix_bottom,
                                    fix_top=fix_top, seed=seed, absolute_level=level)
        except (RuntimeError, ValueError) as e:
            print(f"[pubfig] dose_response: no 4PL fit for {g} ({e}); points drawn without a curve")
            continue
        fits[g] = fit
        yy, lo, hi = fit["predict"](xx)
        if band:
            ax.fill_between(xx, lo, hi, color=c, alpha=0.15, lw=0, zorder=1)
        ax.plot(xx, yy, color=c, lw=1.3 * k, zorder=2)
        val = fit["ic50_abs"] if ic50 != "relative" else fit["ic50_rel"]
        if mark_ic50 and np.isfinite(val):
            y_at = fit["predict"]([val])[0][0]
            ax.plot([val, val], [ax.get_ylim()[0], y_at], color=c, lw=0.6 * k,
                    ls=(0, (2, 2)), zorder=1.5)
            ax.plot([val], [y_at], marker="o", ms=3.5, mfc="white", mec=c,
                    mew=0.9 * k, zorder=4)
        rows.append((g, fit))
    ax.set_xscale("log")
    ax.set_xlabel(f"Concentration ({unit})")
    ax.set_ylabel(ylabel)
    if level is not None and ic50 != "relative":
        refline(ax, y=level)
    if annotate and rows:
        fs = sizes()["annot"]
        # levels of detail; use the longest that fits the free half of the panel.
        # Emax is the last thing dropped: an IC50 without it is not interpretable
        variants = [[], [], [], []]
        for g, fit in rows:
            what = ("IC$_{50}$" if ic50 != "relative" else "EC$_{50}$")
            if ic50 == "relative" or fit["reaches_level"]:
                v = fit["ic50_rel"] if ic50 == "relative" else fit["ic50_abs"]
                ci = fit["ic50_rel_ci"] if ic50 == "relative" else fit["ic50_abs_ci"]
                base = f"{what} {fmt_conc(v, unit)}"
                with_ci = base + (f" [{fmt_conc(ci[0], unit).replace(' ' + unit, '')}–"
                                  f"{fmt_conc(ci[1], unit)}]" if np.isfinite(ci[0]) else "")
            else:     # never reaches the level: say so instead of printing a number
                base = with_ci = f"{what} > {fmt_conc(fit['max_conc'], unit)} (max tested)"
            tail = f" · E$_{{max}}$ {fit['emax']:.0f}% · Hill {abs(fit['hill']):.1f}"
            c = colors[g] if group else INK["text"]
            emax_only = f" · E$_{{max}}$ {fit['emax']:.0f}%"
            variants[0].append((with_ci + tail, c))
            variants[1].append((base + tail, c))
            variants[2].append((base + emax_only, c))
            variants[3].append((base, c))
        room = 0.55 * ax.get_position().width * ax.figure.get_figwidth() * 72
        lines = variants[-1]
        for cand in variants:
            if max(_text_width_pt(t, fs) for t, _ in cand) <= room:
                lines = cand
                break
        loc = annotate_loc
        if loc == "auto":     # inhibition curves empty the top right; activation the bottom right
            loc = "upper right" if np.mean([f["decreasing"] for _, f in rows]) >= 0.5 \
                else "lower right"
        vert, horiz = loc.split()
        x_a, ha = (0.98, "right") if horiz == "right" else (0.02, "left")
        y_a, va = (0.98, "top") if vert == "upper" else (0.02, "bottom")
        for i, (txt, c) in enumerate(lines):
            step = i if va == "top" else len(lines) - 1 - i
            ax.annotate(txt, xy=(x_a, y_a), xycoords="axes fraction",
                        xytext=(0, -step * fs * 1.45 if va == "top" else step * fs * 1.45),
                        textcoords="offset points", ha=ha, va=va, fontsize=fs, color=c)
    if legend and group is not None:
        handles = [mpl.lines.Line2D([], [], color=colors[g], lw=1.3 * k, label=str(g))
                   for g in groups if g in fits]
        ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1, 1.0),
                  ncols=min(len(handles), 3), borderaxespad=0)
    return fits


def dose_grid(data, row: str, col: str, conc: str = "conc", response: str = "response",
              group: str | None = None, row_order=None, col_order=None, journal=None,
              width: str | float = "double", panel_mm: float | None = None,
              panel_aspect: float = 1.0, max_height_mm: float = 230.0, share: str = "all",
              unit: str = "nM", ylabel: str = "Viability (% of vehicle)", **kw):
    """A grid of dose–response panels: rows × columns (e.g. cell lines × drugs),
    every panel on the same axes so curves are comparable at a glance.

    Panels are square by default: their height follows from the width each column
    actually gets, so the grid stays symmetric whatever the number of columns.
    panel_aspect  height / width per panel (1.0 square, <1 wider, >1 taller).
    panel_mm      fixed panel height in mm, overriding panel_aspect.
    max_height_mm cap on the whole figure (a journal page is ~230 mm of type):
                  square panels are kept while they fit, and flattened, with a
                  note, when the grid would not fit a page.
    group: a further split inside each panel (e.g. single agents vs combination).
    Returns (fig, axes dict keyed (row, col), table of fits: one row per panel
    and group with both IC50s, Emax, Hill and R²).
    """
    import pandas as pd
    rows_ = list(row_order or data[row].dropna().unique())
    cols_ = list(col_order or data[col].dropna().unique())
    total_w = (JOURNALS[journal or _STATE["journal"]][width] if isinstance(width, str)
               else float(width))
    nr, nc = len(rows_), len(cols_)
    # panel height from the width a column actually gets (axis labels and gaps removed)
    fs_mm = sizes()["tick"] * 25.4 / 72
    left, right, gap = 4.5 * fs_mm, 2.2 * fs_mm, 1.6 * fs_mm
    panel_w = max(12.0, (total_w - left - right - gap * (nc - 1)) / nc)
    panel_h = panel_mm if panel_mm else panel_w * panel_aspect
    extra = gap * (nr - 1) + 5.5 * fs_mm
    if max_height_mm and panel_h * nr + extra > max_height_mm and not panel_mm:
        panel_h = max((max_height_mm - extra) / nr, 22.0)
        print(f"[pubfig] dose_grid: {nr}×{nc} square panels would be "
              f"{panel_w:.0f} mm and the figure {panel_w * nr + extra:.0f} mm tall; "
              f"panels flattened to {panel_w:.0f}×{panel_h:.0f} mm to fit "
              f"{max_height_mm:.0f} mm. Use fewer rows per figure, a narrower "
              "`width`, or set max_height_mm=None to keep them square.")
    fig_h = panel_h * nr + extra
    fig, axes = plt.subplots(nr, nc, figsize=(total_w * MM, fig_h * MM),
                             sharex=(share in ("all", "x")), sharey=(share in ("all", "y")),
                             squeeze=False)
    _fixed_vgap(fig, 2.0)
    axd, recs = {}, []
    for i, rv in enumerate(rows_):
        for j, cv in enumerate(cols_):
            ax = axes[i, j]
            axd[(rv, cv)] = ax
            sub = data[(data[row] == rv) & (data[col] == cv)]
            if not len(sub):
                ax.set_axis_off()
                continue
            fits = dose_response(ax, sub, conc=conc, response=response, group=group,
                                 unit=unit, legend=(i == 0 and j == len(cols_) - 1),
                                 ylabel=ylabel if j == 0 else "", **kw)
            for g, f in fits.items():
                recs.append(dict(**{row: rv, col: cv}, group=g, ic50_abs=f["ic50_abs"],
                                 ic50_abs_lo=f["ic50_abs_ci"][0], ic50_abs_hi=f["ic50_abs_ci"][1],
                                 ic50_rel=f["ic50_rel"], emax=f["emax"], hill=f["hill"],
                                 r2=f["r2"], reaches_level=f["reaches_level"],
                                 max_conc=f["max_conc"]))
            if i == 0:
                ax.set_title(str(cv), loc="center", fontsize=sizes()["label"])
            if j == len(cols_) - 1:
                ax.annotate(str(rv), xy=(1.02, 0.5), xycoords="axes fraction", rotation=270,
                            ha="left", va="center", fontsize=sizes()["label"])
            if i < len(rows_) - 1 and share in ("all", "x"):
                ax.set_xlabel("")
            if j > 0 and share in ("all", "y"):
                ax.tick_params(labelleft=False)
    return fig, axd, pd.DataFrame(recs)


def ic50_forest(ax, table, label: str, value: str = "ic50_abs", lo: str = "ic50_abs_lo",
                hi: str = "ic50_abs_hi", group: str | None = None, colors=None,
                unit: str = "nM", sort: bool = True, reaches: str = "reaches_level",
                max_conc: str = "max_conc"):
    """IC50 per cell line (or per drug) with its 95% CI, on a log axis: the
    comparison plot that belongs beside a grid of curves. Curves that never reach
    the level are drawn as an arrow at the top concentration, not as a number."""
    t = table.copy()
    if sort:
        t = t.sort_values(value, na_position="last")
    groups = list(t[group].dropna().unique()) if group else ["all"]
    colors = colors or (group_colors(groups) if group else {"all": INK["text"]})
    k = sizes()["stroke"]
    ys = np.arange(len(t))[::-1]
    for y, (_, r) in zip(ys, t.iterrows()):
        c = colors[r[group]] if group else colors["all"]
        if reaches in r and not r[reaches]:
            ax.annotate("", xy=(r[max_conc] * 3, y), xytext=(r[max_conc], y),
                        arrowprops=dict(arrowstyle="->", color=c, lw=0.9 * k))
            ax.plot([r[max_conc]], [y], marker="|", color=c, ms=5)
            continue
        if np.isfinite(r.get(lo, np.nan)) and np.isfinite(r.get(hi, np.nan)):
            ax.plot([r[lo], r[hi]], [y, y], color=c, lw=0.9 * k, solid_capstyle="butt")
        ax.plot([r[value]], [y], marker="o", ms=4, color=c, zorder=3)
    ax.set_yticks(ys, [str(v) for v in t[label]])
    ax.set_ylim(-0.8, len(t) - 0.2)
    ax.set_xscale("log")
    ax.set_xlabel(f"IC$_{{50}}$ ({unit})")
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_gid("category_axis")        # named y categories: no y-axis line or title
    if group:
        ax.legend(handles=[mpl.lines.Line2D([], [], color=colors[g], marker="o", ls="none",
                                            label=str(g)) for g in groups],
                  loc="lower right", bbox_to_anchor=(1, 1.0), ncols=len(groups),
                  borderaxespad=0)
    return ax


def bliss_expected(ea, eb):
    """Bliss independence: expected fractional effect of A+B from the single agents
    (effects as fractions 0–1)."""
    ea, eb = np.asarray(ea, float), np.asarray(eb, float)
    return ea + eb - ea * eb


def combination_curves(ax, data, conc: str = "conc", response: str = "response",
                       arm: str = "arm", single=("Drug A", "Drug B"), combo: str = "A + B",
                       ratio: float | None = None, colors=None, unit: str = "nM",
                       expected: bool = True, level: float = 50.0, **kw):
    """Single agents and their fixed-ratio combination in one panel, with the Bliss
    expectation for the combination drawn as a dashed line.

    data: long table with an `arm` column naming the two single agents and the
    combination; `conc` is the TOTAL concentration for the combination arm.
    ratio: A:B mixing ratio (e.g. 1.0 for 1:1): used to split the combination's
    total concentration when computing the expectation. Returns (fits, expected_df).
    """
    import pandas as pd
    arms = [single[0], single[1], combo]
    colors = colors or dict(zip(arms, [PALETTES["okabe_ito"][i] for i in (0, 2, 5)]))
    fits = dose_response(ax, data[data[arm].isin(arms)], conc=conc, response=response,
                         group=arm, order=arms, colors=colors, unit=unit, level=level, **kw)
    exp_df = None
    if expected and all(a in fits for a in arms):
        r = ratio if ratio else 1.0
        fa_frac, fb_frac = r / (1 + r), 1 / (1 + r)
        x = np.logspace(np.log10(data[conc][data[conc] > 0].min()),
                        np.log10(data[conc][data[conc] > 0].max()), 200)
        top = max(fits[a]["top"] for a in single)
        ea = 1 - fits[single[0]]["predict"](x * fa_frac)[0] / top
        eb = 1 - fits[single[1]]["predict"](x * fb_frac)[0] / top
        exp = (1 - bliss_expected(np.clip(ea, 0, 1), np.clip(eb, 0, 1))) * top
        ax.plot(x, exp, color=INK["muted"], lw=1.0 * sizes()["stroke"], ls=(0, (4, 2)), zorder=2)
        # named in the legend, not on the curve: an inline label collides with the
        # IC50 block in a single-column panel
        leg = ax.get_legend()
        handles = (list(leg.legend_handles) if leg else [])
        labels_ = ([t.get_text() for t in leg.texts] if leg else [])
        handles.append(mpl.lines.Line2D([], [], color=INK["muted"],
                                        lw=1.0 * sizes()["stroke"], ls=(0, (4, 2))))
        labels_.append("Bliss expectation")
        ax.legend(handles, labels_, loc="lower right", bbox_to_anchor=(1, 1.0),
                  ncols=min(len(handles), 4), borderaxespad=0)
        exp_df = pd.DataFrame(dict(conc=x, expected=exp,
                                   observed=fits[combo]["predict"](x)[0]))
    return fits, exp_df


def synergy_matrix(ax, matrix, conc_a=None, conc_b=None, kind: str = "bliss",
                   effect: str = "inhibition", cmap=None, cbar_label=None, annot: bool = True,
                   unit: str = "nM", labels=("Drug A", "Drug B"), vmax=None):
    """Checkerboard of a two-drug matrix.

    matrix: 2-D array of the measured effect, rows = concentrations of drug B,
    columns = drug A; row 0 / column 0 are the single agents (concentration 0 of
    the other drug). `effect`: "inhibition" (0–100, higher = more killing) or
    "viability" (converted internally).
    kind: "effect" draws the measured surface; "bliss" or "hsa" draws the excess
    over that model (observed − expected, in percentage points, diverging around
    zero): positive = synergy, negative = antagonism.
    Returns the plotted matrix.
    """
    M = np.asarray(matrix, float)
    if effect == "viability":
        M = 100 - M
    if kind == "effect":
        shown, cmap = M, (cmap or "ice_navy")
        label = cbar_label or "Inhibition (%)"
        vmin, vmax_ = 0, (vmax or 100)
        center = None
    else:
        a0, b0 = M[0, :] / 100, M[:, 0] / 100          # single agents along the edges
        exp = (bliss_expected(np.repeat(b0[:, None], M.shape[1], 1),
                              np.repeat(a0[None, :], M.shape[0], 0)) if kind == "bliss"
               else np.maximum(np.repeat(b0[:, None], M.shape[1], 1),
                               np.repeat(a0[None, :], M.shape[0], 0)))
        shown = M - exp * 100
        v = vmax or np.nanmax(np.abs(shown))
        vmin, vmax_, center = -v, v, 0
        cmap = cmap or "navy_amber_div"
        label = cbar_label or f"{kind.upper() if kind == 'hsa' else 'Bliss'} excess (pp)"
    im = ax.imshow(shown, cmap=cmap, vmin=vmin, vmax=vmax_, origin="lower", aspect="auto")
    if conc_a is not None:
        ax.set_xticks(range(len(conc_a)), [f"{c:g}" for c in conc_a])
    if conc_b is not None:
        ax.set_yticks(range(len(conc_b)), [f"{c:g}" for c in conc_b])
    ax.set_xlabel(f"{labels[0]} ({unit})")
    ax.set_ylabel(f"{labels[1]} ({unit})")
    if annot:
        fs = max(sizes()["min"], sizes()["annot"] - 1)
        thr = (vmax_ - vmin) * 0.6
        for i in range(shown.shape[0]):
            for j in range(shown.shape[1]):
                val = shown[i, j]
                dark = abs(val - (center if center is not None else vmin)) > thr / 2
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=fs,
                        color="white" if dark else INK["text"])
    image_axis(ax)
    ax.set_gid("image_axes")
    cb = colorbar(im, ax, label)
    return shown


def combination_index(fits, single, combo, ratio: float = 1.0, fa=(0.5, 0.75, 0.9)):
    """Chou–Talalay combination index at given effect levels for fixed-ratio data.

    CI < 1 synergy, = 1 additivity, > 1 antagonism. Needs the single-agent and
    combination fits from dose_response (same units). Returns a DataFrame; values
    are NaN where an arm never reaches that effect: extrapolating past the tested
    range is where combination indices go wrong.
    """
    import pandas as ptd
    rows = []
    for level_fa in fa:
        out = {"fa": level_fa}
        doses = {}
        for name in (single[0], single[1], combo):
            f = fits[name]
            top = f["top"]
            target = top * (1 - level_fa)
            b, t, l50, h = f["params"]
            ok = (target - b) * (t - target) > 0
            doses[name] = (10 ** (l50 - np.log10((t - target) / (target - b)) / h)) if ok else np.nan
        d_combo = doses[combo]
        r = ratio / (1 + ratio)
        ci = (d_combo * r / doses[single[0]] + d_combo * (1 - r) / doses[single[1]]) \
            if np.isfinite(d_combo) else np.nan
        out.update({f"D_{single[0]}": doses[single[0]], f"D_{single[1]}": doses[single[1]],
                    f"D_{combo}": d_combo, "CI": ci})
        rows.append(out)
    return ptd.DataFrame(rows)


# --------------------------------------------------------------------------
# Accessibility preview
# --------------------------------------------------------------------------
_CVD = {  # Machado et al. (2009), severity 1.0, linear RGB
    "deuteranopia": np.array([[0.367322, 0.860646, -0.227968],
                              [0.280085, 0.672501, 0.047413],
                              [-0.011820, 0.042940, 0.968881]]),
    "protanopia": np.array([[0.152286, 1.052583, -0.204868],
                            [0.114503, 0.786281, 0.099216],
                            [-0.003882, -0.048116, 1.051998]]),
    "tritanopia": np.array([[1.255528, -0.076749, -0.178779],
                            [-0.078411, 0.930809, 0.147602],
                            [0.004733, 0.691367, 0.303900]]),
}


def _srgb_to_lin(c):
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def _lin_to_srgb(c):
    c = np.clip(c, 0, 1)
    return np.where(c <= 0.0031308, c * 12.92, 1.055 * c ** (1 / 2.4) - 0.055)


def simulate_cvd(rgb, kind: str):
    """rgb: (..., 3) floats 0–1. kind: deuteranopia | protanopia | tritanopia | grayscale."""
    lin = _srgb_to_lin(np.asarray(rgb, float))
    if kind == "grayscale":
        yv = lin @ np.array([0.2126, 0.7152, 0.0722])
        out = np.repeat(yv[..., None], 3, -1)
    else:
        out = lin @ _CVD[kind].T
    return _lin_to_srgb(out)


def _lab(rgb):
    lin = _srgb_to_lin(np.asarray(rgb, float))
    M = np.array([[0.4124, 0.3576, 0.1805], [0.2126, 0.7152, 0.0722], [0.0193, 0.1192, 0.9505]])
    xyz = lin @ M.T / np.array([0.95047, 1.0, 1.08883])
    f = np.where(xyz > 0.008856, np.cbrt(xyz), 7.787 * xyz + 16 / 116)
    return np.stack([116 * f[..., 1] - 16, 500 * (f[..., 0] - f[..., 1]),
                     200 * (f[..., 1] - f[..., 2])], -1)


def check_palette(colors, min_delta: float = 12, verbose: bool = True) -> dict:
    """Smallest pairwise colour difference (CIE ΔE76) under normal and CVD vision."""
    rgb = np.array([mpl.colors.to_rgb(c) for c in colors])
    res = {}
    for kind in ("normal", "deuteranopia", "protanopia", "tritanopia", "grayscale"):
        sim = rgb if kind == "normal" else simulate_cvd(rgb, kind)
        L = _lab(sim)
        d = [np.linalg.norm(L[i] - L[j]) for i, j in itertools.combinations(range(len(L)), 2)]
        res[kind] = float(min(d)) if d else np.inf
    if verbose:
        weak = [k for k, v in res.items() if k != "grayscale" and v < min_delta]
        msg = ", ".join(f"{k} {v:.0f}" for k, v in res.items())
        print(f"palette ΔE (min pairwise): {msg}" + (f"  ⚠ weak under {', '.join(weak)}" if weak else ""))
    return res


def cvd_preview(fig, path, width_px: int = 900):
    """Save a 2×2 sheet: original, deuteranopia, protanopia, greyscale."""
    from PIL import Image, ImageDraw
    finalize(fig)
    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[..., :3] / 255.0
    tiles = [("original", img)] + [(k, simulate_cvd(img, k))
                                  for k in ("deuteranopia", "protanopia", "grayscale")]
    h, w = img.shape[:2]
    scale = width_px / w
    tw, th = width_px, int(h * scale)
    sheet = Image.new("RGB", (tw * 2, (th + 28) * 2), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        from PIL import ImageFont
        font = ImageFont.truetype(str(Path(mpl.get_data_path()) / "fonts/ttf/DejaVuSans-Bold.ttf"), 16)
    except Exception:
        font = None
    for i, (name, t) in enumerate(tiles):
        im = Image.fromarray((t * 255).astype(np.uint8)).resize((tw, th))
        x0, y0 = (i % 2) * tw, (i // 2) * (th + 28)
        draw.text((x0 + 8, y0 + 5), name, fill=(27, 42, 73), font=font)
        sheet.paste(im, (x0, y0 + 28))
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
    return Path(path)


# --------------------------------------------------------------------------
# Annotated heatmaps (categories, sub-categories, clustering)
# --------------------------------------------------------------------------
def cluster_order(M, axis: int = 0, metric: str = "correlation", method: str = "average",
                  optimal: bool = True):
    """Hierarchical clustering order. Returns (order, linkage): linkage is None
    when there is too little data to cluster."""
    from scipy.cluster import hierarchy as hc
    from scipy.spatial.distance import pdist
    X = np.asarray(M, float)
    X = X if axis == 0 else X.T
    if len(X) < 3:
        return np.arange(len(X)), None
    d = pdist(X, metric=metric)
    d = np.nan_to_num(d, nan=float(np.nanmax(d)) if np.isfinite(d).any() else 1.0)
    Z = hc.linkage(d, method=method, optimal_ordering=optimal and len(X) < 600)
    return np.asarray(hc.leaves_list(Z)), Z


def _dendrogram(ax, Z, orient: str = "left", n: int = 0):
    """Thin grey dendrogram matching the clustering order used by the heatmap."""
    from scipy.cluster import hierarchy as hc
    with plt.rc_context({"lines.linewidth": 0.5 * sizes()["stroke"]}):
        hc.dendrogram(Z, ax=ax, orientation=orient, no_labels=True,
                      link_color_func=lambda _: INK["muted"])
    ax.set_axis_off()
    ax.set_gid("dendrogram")


def _ann_colors(values, name, given=None, cmap_for_many="tab20"):
    """Map category values to colours: given dict, else a skill palette, else tab20."""
    cats = list(dict.fromkeys(values))
    if given and name in given:
        return {c: given[name][c] for c in cats if c in given[name]} | \
               {c: INK["grid"] for c in cats if c not in given[name]}
    if len(cats) == 2:
        return dict(zip(cats, palette("pair", 2)))
    if len(cats) <= 8:
        return dict(zip(cats, palette("okabe_ito", len(cats))))
    cm = mpl.colormaps[cmap_for_many].resampled(len(cats))
    return {c: to_hex(cm(i)) for i, c in enumerate(cats)}


def _legend_panel(ax, blocks: dict):
    """Swatch legends stacked from the top of their own axes: {title: {label: colour}}."""
    ax.set_axis_off()
    fs = sizes()["legend"]
    line = fs * 1.8                        # points between entries

    def off(dy):
        return mpl.transforms.offset_copy(ax.transAxes, fig=ax.figure, y=-dy, units="points")

    y = 0.0
    for title, mapping in blocks.items():
        ax.text(0, 1, title, transform=off(y), fontsize=fs, fontweight="bold",
                va="top", ha="left", color=INK["text"], clip_on=False)
        y += line * 1.15
        for label, color in mapping.items():
            ax.plot([0.05], [1], marker="s", ms=fs * 0.75, color=color, lw=0,
                    transform=off(y + fs * 0.55), clip_on=False)
            ax.text(0.22, 1, str(label), transform=off(y), fontsize=fs, va="top",
                    ha="left", color=INK["text"], clip_on=False)
            y += line
        y += line * 0.6
    return y


def _fit_row_labels(ax, labels, min_gap_pt: float = 3.0, tries: int = 5):
    """Label as many rows as stay legible, re-measuring after each layout pass."""
    fig = ax.figure
    n = fit_count(ax, len(labels), spacing=2.2)
    for _ in range(tries):
        idx = (np.arange(len(labels)) if n >= len(labels)
               else np.linspace(0, len(labels) - 1, max(n, 2)).astype(int))
        ax.set_yticks(idx, [labels[j] for j in idx])
        ax.tick_params(left=len(idx) < len(labels), length=2, width=0.5, labelleft=True)
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        bbs = sorted((_text_bbox(t.label1, r) for t in ax.yaxis._update_ticks()
                      if t.label1.get_text()), key=lambda b: b.y0)
        gaps = [(b2.y0 - b1.y1) * 72 / fig.dpi for b1, b2 in zip(bbs, bbs[1:])]
        if not gaps or min(gaps) >= min_gap_pt or n <= 2:
            return len(idx)
        n = max(2, int(n * 0.8))
    return len(idx)


def annotated_heatmap(matrix, col_ann=None, row_ann=None, col_groups=None, row_groups=None,
                      cluster_rows: bool = True, cluster_cols: bool = False,
                      scale: str | None = "row", cmap: str | None = None, center=0.0,
                      robust: float = 99.0, vmin=None, vmax=None,
                      journal: str | None = None, width: str | float = "single",
                      height_mm: float = 120, value_label: str = "",
                      row_labels: bool | None = None, col_labels: bool = False,
                      ann_colors: dict | None = None, ann_track_mm: float = 2.6,
                      max_legend: int = 10, group_labels: bool = True,
                      dendrogram: bool = True, legend_mm: float = 26, stat_note: str = ""):
    """Heatmap with categorical annotation tracks, grouped columns and clustering.

    matrix      DataFrame (rows = features, columns = samples).
    col_ann     DataFrame indexed like matrix.columns, one column per annotation
                track (Treatment, CancerType, CellLine…), drawn top to bottom.
    col_groups  name of a col_ann column whose blocks are separated by a gap and
                labelled underneath (the sub-category structure), or None.
    row_ann / row_groups  the same for rows (tracks drawn left of the heatmap).
    scale       "row" (z-score per feature), "center" (subtract the row mean),
                or None. Say which in the caption.
    cmap        default: diverging navy↔amber for centred data, viridis otherwise.
    robust      colour limits from this percentile of |values| (symmetric when
                centred), so a few extreme cells don't flatten the map.
    row_labels  None = show them only if they fit at the current font size.

    Returns dict(fig, heatmaps, ax_ann, ax_legend, order_rows, order_cols, groups).
    """
    df = matrix.copy()
    if scale == "row":
        sd = df.std(axis=1, ddof=1).replace(0, np.nan)
        df = df.sub(df.mean(axis=1), axis=0).div(sd, axis=0).fillna(0)
        value_label = value_label or "Row z-score"
    elif scale == "center":
        df = df.sub(df.mean(axis=1), axis=0)
        value_label = value_label or "Centred value"
    else:
        value_label = value_label or "Value"

    # ---- ordering
    Zr = Zc = None
    if row_groups is not None and row_ann is not None:
        keys = row_ann.loc[df.index, row_groups]
        ordered = []
        for gname in dict.fromkeys(keys):
            rows_g = [r for r in df.index if keys[r] == gname]
            if cluster_rows and len(rows_g) > 2:
                o, _ = cluster_order(df.loc[rows_g].to_numpy(), axis=0)
                rows_g = [rows_g[i] for i in o]
            ordered += rows_g
        df = df.loc[ordered]
        dendrogram = False          # order is dictated by the groups, not a tree
    elif cluster_rows and len(df) > 2:
        order_r, Zr = cluster_order(df.to_numpy(), axis=0)
        df = df.iloc[order_r]
    if col_groups is not None and col_ann is not None:
        keys = col_ann.loc[df.columns, col_groups]
        blocks, seen = {}, []
        for c in df.columns:
            blocks.setdefault(keys[c], []).append(c)
        if cluster_cols:
            for g, cols in blocks.items():
                if len(cols) > 2:
                    o, _ = cluster_order(df[cols].to_numpy(), axis=1)
                    blocks[g] = [cols[i] for i in o]
        df = df[[c for cols in blocks.values() for c in cols]]
    else:
        if cluster_cols and df.shape[1] > 2:
            order_c, Zc = cluster_order(df.to_numpy(), axis=1)
            df = df.iloc[:, order_c]
        blocks = {"": list(df.columns)}

    V = df.to_numpy(float)
    if vmax is None:
        hi = np.nanpercentile(np.abs(V - (center or 0)), robust)
        vmax = (center or 0) + hi
        vmin = (center or 0) - hi if center is not None else np.nanpercentile(V, 100 - robust)
    if cmap is None:
        cmap = "navy_amber_div" if center is not None else "viridis"

    # ---- figure skeleton
    total_w = (JOURNALS[journal or _STATE["journal"]][width] if isinstance(width, str)
               else float(width))
    fig = plt.figure(figsize=(total_w * MM, height_mm * MM))
    tracks = list(col_ann.columns) if col_ann is not None else []
    legend_blocks = {}
    for t in tracks:
        cmap_t = _ann_colors(col_ann.loc[df.columns, t], t, ann_colors)
        if len(cmap_t) <= max_legend:
            legend_blocks[t] = cmap_t
    has_legend = bool(legend_blocks)
    show_dendro = bool(dendrogram and Zr is not None)
    leg_w = (legend_mm / total_w) if has_legend else 0.0
    ratios = [1 - leg_w] + ([leg_w] if has_legend else [])
    outer = fig.add_gridspec(1, len(ratios), width_ratios=ratios, wspace=0.04)
    i_main = 0
    # a track can never be thinner than its own label, or the names collide
    ann_track_mm = max(ann_track_mm, sizes()["tick"] * 1.55 / 72 * 25.4)
    # ~20% of the canvas goes to labels, brackets and the note, so scale up
    track_frac = (len(tracks) * ann_track_mm) / (height_mm * 0.78)
    row_tracks = list(row_ann.columns) if row_ann is not None else []
    for t in row_tracks:
        cmap_t = _ann_colors(row_ann.loc[df.index, t], t, ann_colors)
        if len(cmap_t) <= max_legend:
            legend_blocks[t] = cmap_t
    has_legend = bool(legend_blocks)
    n_cols_total = sum(len(c) for c in blocks.values())
    dend_w = max(0.10 * n_cols_total, 3.0) if show_dendro else 0.0
    n_cols_total = sum(len(c) for c in blocks.values())
    dend_w = max(0.10 * n_cols_total, 3.0) if show_dendro else 0.0
    has_strip = bool(group_labels and len(blocks) > 1)
    height_ratios = (([track_frac] if tracks else []) + [1 - track_frac - (0.05 if has_strip else 0)]
                     + ([0.05] if has_strip else []))
    r_track = 0 if tracks else None
    r_heat = 1 if tracks else 0
    r_strip = r_heat + 1
    grid = outer[i_main].subgridspec(
        len(height_ratios),
        len(blocks) * 2 - 1 + (1 if row_tracks else 0) + (1 if show_dendro else 0),
        height_ratios=height_ratios,
        width_ratios=(([dend_w] if show_dendro else [])
                      + ([1.8 * len(row_tracks)] if row_tracks else [])
                      + [w for i, cols in enumerate(blocks.values())
                         for w in ((len(cols), 0.03) if i < len(blocks) - 1 else (len(cols),))]),
        hspace=0.02, wspace=0.0)

    heatmaps, ann_axes, strip_axes = [], [], []
    if row_labels is None:
        row_labels = len(df) <= 60
    col_off = (1 if show_dendro else 0) + (1 if row_tracks else 0)
    for i, (g, cols) in enumerate(blocks.items()):
        col_i = i * 2 + col_off
        sub = df[cols]
        ax = fig.add_subplot(grid[r_heat, col_i])
        im = ax.imshow(sub.to_numpy(float), aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax,
                       interpolation="nearest")
        image_axis(ax)
        ax.set_xticks([])
        if col_labels:
            ax.set_xticks(range(len(cols)), cols, rotation=90)
        if i == 0 and row_labels:
            n_show = _fit_row_labels(ax, list(df.index))
            if n_show < len(df):
                print(f"[pubfig] annotated_heatmap: {len(df)} rows, {n_show} labels fit: "
                      "labelled a subset; give the figure more height to show all.")
        else:
            ax.set_yticks([])
        heatmaps.append(ax)

        if tracks:
            axa = fig.add_subplot(grid[r_track, col_i])
            rgb = np.zeros((len(tracks), len(cols), 3))
            for ti, t in enumerate(tracks):
                cmap_t = _ann_colors(col_ann.loc[df.columns, t], t, ann_colors)
                for ci, c in enumerate(cols):
                    rgb[ti, ci] = mpl.colors.to_rgb(cmap_t.get(col_ann.loc[c, t], INK["grid"]))
            axa.imshow(rgb, aspect="auto", interpolation="nearest")
            image_axis(axa)
            axa.set_xticks([])
            if i == 0:
                axa.set_yticks(range(len(tracks)), tracks)
                axa.tick_params(labelleft=True, left=False)
            else:
                axa.set_yticks([])
            ann_axes.append(axa)

        if has_strip:
            strip = fig.add_subplot(grid[r_strip, col_i])
            group_bracket(strip, str(g))
            strip_axes.append(strip)

    ax_row_ann = None
    if row_tracks:
        ax_row_ann = fig.add_subplot(grid[r_heat, 1 if show_dendro else 0])
        rgb = np.zeros((len(df), len(row_tracks), 3))
        for ti, t in enumerate(row_tracks):
            cmap_t = _ann_colors(row_ann.loc[df.index, t], t, ann_colors)
            for ri, r_ in enumerate(df.index):
                rgb[ri, ti] = mpl.colors.to_rgb(cmap_t.get(row_ann.loc[r_, t], INK["grid"]))
        ax_row_ann.imshow(rgb, aspect="auto", interpolation="nearest")
        image_axis(ax_row_ann)
        ax_row_ann.set_yticks([])
        ax_row_ann.set_xticks([])
        for ti, t in enumerate(row_tracks):     # names above the strips
            ax_row_ann.annotate(t, xy=(ti, 0), xytext=(0, 4), textcoords="offset points",
                                rotation=90, ha="center", va="bottom",
                                fontsize=sizes()["legend"], annotation_clip=False)

    axd = None
    if show_dendro:                    # inside the heatmap grid → same rows, same height
        axd = fig.add_subplot(grid[r_heat, 0])
        _dendrogram(axd, Zr, orient="left")
        axd.set_ylim(10 * len(df), 0)  # scipy leaves sit at 5, 15, 25 … → rows line up
        axd.set_xlim(axd.get_xlim()[0], 0)
    ax_leg = cb_host = None
    if has_legend:
        side = outer[1].subgridspec(2, 1, height_ratios=[1, 0.12], hspace=0.04)
        ax_leg = fig.add_subplot(side[0])
        _legend_panel(ax_leg, legend_blocks)
        cb_host = fig.add_subplot(side[1])
        cb_host.set_axis_off()

    if cb_host is not None:     # colour scale in its own slot under the legends
        cax = cb_host.inset_axes([0.0, 0.55, 0.72, 0.16])
        cb = fig.colorbar(im, cax=cax, orientation="horizontal")
        cb.outline.set_visible(False)
        cb.ax.tick_params(length=2, width=0.5, pad=1)
        cb.ax.set_title(value_label, fontsize=sizes()["legend"], loc="left", pad=3,
                        color=INK["text"])
        cb.locator = mpl.ticker.MaxNLocator(3)
        cb.update_ticks()
    else:
        cb = colorbar(im, ax=heatmaps[-1], label=value_label)
        cb.ax.set_box_aspect(8)
    if stat_note:
        fig_stat_title(fig, stat_note, loc="left")
    skipped = [t for t in tracks if t not in legend_blocks]
    if skipped:
        print(f"[pubfig] annotated_heatmap: {', '.join(skipped)} has more than {max_legend} "
              "categories: labelled by group brackets instead of a legend.")
    return dict(fig=fig, heatmaps=heatmaps, ax_ann=ann_axes, ax_row_ann=ax_row_ann,
                ax_dendrogram=axd, ax_legend=ax_leg,
                matrix=df, blocks=blocks, linkage_rows=Zr, colorbar=cb)


# --------------------------------------------------------------------------
# QA
# --------------------------------------------------------------------------
def _drawn_texts(fig):
    """(kind, Text) for every visible, non-empty text that is actually drawn."""
    out, tick_ids = [], set()
    for ax in fig.axes:
        for axis in (ax.xaxis, ax.yaxis):
            for t in axis.get_major_ticks() + axis.get_minor_ticks():
                tick_ids.update({id(t.label1), id(t.label2)})
        if not ax.axison or not ax.get_visible():
            continue
            try:
                drawn = axis._update_ticks()      # only ticks inside the view limits
            except Exception:                     # private API: fall back gracefully
                drawn = axis.get_major_ticks()
            for t in drawn:
                for lab in (t.label1, t.label2):
                    if lab.get_visible() and lab.get_text().strip():
                        out.append(("tick", lab))
    fig_texts = {id(t) for t in fig.texts}
    for t in fig.findobj(mpl.text.Text):
        if id(t) in tick_ids or not t.get_visible() or not t.get_text().strip():
            continue
        if t.axes is None and id(t) not in fig_texts:
            continue          # orphan (e.g. a tick label left over after relocating ticks)
        out.append(("text", t))
    return out


def _shared_label(ax, which: str = "y") -> bool:
    """True when this axes shares its scale with a sibling that carries the label."""
    group = (ax.get_shared_y_axes() if which == "y" else ax.get_shared_x_axes())
    sibs = [a for a in group.get_siblings(ax) if a is not ax]
    getter = "get_ylabel" if which == "y" else "get_xlabel"
    return any(getattr(a, getter)() for a in sibs)


def audit(fig, journal: str | None = None, final_width_mm: float | None = None,
          verbose: bool = True) -> list[str]:
    """Return a list of style/readability problems. Empty list == ready to save.

    final_width_mm: width the figure will actually be printed at, if it will be
    scaled (e.g. dropped into a collage or a LaTeX subfigure). Font checks then
    use the *effective* printed size.
    """
    journal = journal or _STATE["journal"]
    spec = JOURNALS[journal]
    sz = sizes(journal)
    issues: list[str] = []
    finalize(fig)
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    fig_bb = fig.bbox

    w_mm = fig.get_figwidth() / MM
    scale = (final_width_mm / w_mm) if final_width_mm else 1.0
    printed_w = w_mm * scale
    allowed = {spec["single"], spec["onehalf"], spec["double"]}
    if not final_width_mm and not any(abs(printed_w - a) < 1.0 for a in allowed):
        issues.append(f"Width {printed_w:.1f} mm is not a {journal} column width {sorted(allowed)}; "
                      "build the figure at its final size.")
    if scale < 0.999:
        issues.append(f"Figure will be scaled to {scale:.0%}; rebuild it at {final_width_mm:.0f} mm "
                      "instead of shrinking (fonts and lines shrink with it).")

    texts = _drawn_texts(fig)

    # 1. Size floor and readable target (effective printed size), summarised
    floor = spec["min_font"]
    too_small = [(t.get_fontsize() * scale, t.get_text()) for _, t in texts
                 if t.get_fontsize() * scale < floor - 1e-6]
    if too_small:
        sz_, txt = min(too_small)
        issues.append(f"{len(too_small)} text item(s) print below the {floor} pt minimum "
                      f"(smallest: '{txt[:20]}' at {sz_:.1f} pt).")
    if _STATE["readability"] != "strict":
        under = [(t.get_fontsize() * scale, t.get_text()) for k, t in texts
                 if k == "tick" and floor - 1e-6 <= t.get_fontsize() * scale < sz["tick"] * 0.999]
        if under:
            sz_, txt = min(under)
            issues.append(f"{len(under)} tick label(s) are smaller than the readable {sz['tick']} pt "
                          f"(e.g. '{txt[:15]}' at {sz_:.1f} pt).")

    # 2. Hierarchy: axis titles never smaller than their tick labels
    for ax in fig.axes:
        for axis, name in ((ax.xaxis, "x"), (ax.yaxis, "y")):
            lab = axis.label
            ticks = [t.label1 for t in axis.get_major_ticks() if t.label1.get_text()]
            if lab.get_text() and ticks and lab.get_fontsize() < ticks[0].get_fontsize():
                issues.append(f"{name}-axis title '{lab.get_text()[:20]}' is smaller than its tick labels.")

    # 3. Overlapping text anywhere in the figure
    boxes = [(t, _text_bbox(t, r)) for _, t in texts]
    for (t1, b1), (t2, b2) in itertools.combinations(boxes, 2):
        inter = Bbox.intersection(b1, b2)
        if inter is not None and inter.width > 1 and inter.height > 1:
            issues.append(f"Text overlap: '{t1.get_text()[:18]}' and '{t2.get_text()[:18]}'.")

    # 3a. Tick labels inside the plot area (e.g. rotated labels anchored at their
    #     centre run up through the bars: their boxes don't overlap each other)
    for ax in fig.axes:
        if not ax.axison or ax.get_gid() in ("marginal",):
            continue
        ab = ax.get_window_extent(r)
        tol = 1.5
        for axis, horiz in ((ax.xaxis, True), (ax.yaxis, False)):
            try:
                ticks = axis._update_ticks()
            except Exception:
                continue
            bad = []
            for t in ticks:
                for lab in (t.label1, t.label2):
                    if not (lab.get_visible() and lab.get_text().strip()):
                        continue
                    b = _text_bbox(lab, r)
                    if (b.x1 > ab.x0 + tol and b.x0 < ab.x1 - tol and
                            b.y1 > ab.y0 + tol and b.y0 < ab.y1 - tol):
                        bad.append(lab.get_text())
            if bad:
                issues.append(f"'{ax.get_label() or 'axes'}': {len(bad)} {'x' if horiz else 'y'} "
                              f"tick label(s) run into the plot area (e.g. '{bad[0][:18]}'): "
                              "for rotated labels use ha='right' with rotation_mode='anchor' "
                              "(pf.rotate_xlabels).")

    # 3b. Crowded tick labels (touching neighbours are unreadable at print size)
    min_gap = 2.5 * fig.dpi / 72     # 2.5 pt
    for ax in fig.axes:
        rv = getattr(ax, "_pf_venn_r", None)
        if rv is not None:
            p0 = ax.transData.transform((0, 0)); p1 = ax.transData.transform((rv, 0))
            r_pt = (p1[0] - p0[0]) * 72 / fig.dpi
            need = 3.0 * sizes()["label"]
            if r_pt < need:
                issues.append(f"'{ax.get_label() or 'venn'}': Venn circles are only {r_pt:.0f} pt in "
                              "radius: region counts will crowd; give the panel more room.")
        if not ax.axison:
            continue
        for axis, horiz in ((ax.xaxis, True), (ax.yaxis, False)):
            try:
                labs = [t.label1 for t in axis._update_ticks()
                        if t.label1.get_visible() and t.label1.get_text().strip()]
            except Exception:
                continue
            bbs = sorted((_text_bbox(t, r) for t in labs), key=lambda b: b.x0 if horiz else b.y0)
            for b1, b2 in zip(bbs, bbs[1:]):
                gap = (b2.x0 - b1.x1) if horiz else (b2.y0 - b1.y1)
                if gap < min_gap:
                    issues.append(f"'{ax.get_label() or 'axes'}': {'x' if horiz else 'y'} tick labels "
                                  "are crowded: fewer ticks, shorter names, or a wider panel.")
                    break

    # 4. Text clipped by the figure edge
    for t, b in boxes:
        if b.x0 < fig_bb.x0 - 1 or b.x1 > fig_bb.x1 + 1 or b.y0 < fig_bb.y0 - 1 or b.y1 > fig_bb.y1 + 1:
            issues.append(f"'{t.get_text()[:20]}' extends past the figure edge.")

    for ax in fig.axes + [c for a in fig.axes for c in a.child_axes]:
        tb = ax.get_tightbbox(r)
        if tb is not None and (tb.x0 < fig_bb.x0 - 1 or tb.x1 > fig_bb.x1 + 1
                               or tb.y0 < fig_bb.y0 - 1 or tb.y1 > fig_bb.y1 + 1):
            issues.append(f"'{ax.get_label() or 'axes'}' is cut off by the figure edge.")

    for ax in fig.axes:
        is_cbar = hasattr(ax, "_colorbar") or ax.get_label() == "<colorbar>"
        if is_cbar or ax.get_gid() == "marginal":
            continue
        name = ax.get_title() or ax.get_title(loc="left") or ax.get_label() or "axes"
        is_image = bool(ax.get_images()) or ax.get_gid() in ("image_axes", "transcript_track") or any(
            isinstance(c, mpl.collections.QuadMesh) for c in ax.collections)

        # 5. Frame: open L-frame on data axes (the upper segment of a broken-axis
        #    pair has no x-axis line and shares the pair's y-title by design)
        broken_upper = getattr(ax, "_pf_broken_upper", False)
        if ax.name == "rectilinear" and not is_image and ax.axison:
            if ax.spines["top"].get_visible() or ax.spines["right"].get_visible():
                issues.append(f"'{name}': top/right spines visible.")
            if not ax.spines["bottom"].get_visible() and not broken_upper:
                issues.append(f"'{name}': x-axis line hidden: reviewers expect it; use categorical_axis().")
            if not ax.spines["left"].get_visible() and not _shared_label(ax, "y") \
                    and ax.get_gid() != "category_axis":
                issues.append(f"'{name}': y-axis line hidden.")
            pair_label = broken_upper and getattr(ax, "_pf_pair", None) is not None \
                and ax._pf_pair.get_ylabel()
            if ax.has_data() and not ax.get_ylabel() and not _shared_label(ax, "y") \
                    and not pair_label and ax.get_gid() != "category_axis":
                issues.append(f"'{name}': no y-axis title (include units).")

        # 6. Colormaps
        for m in ax.get_images() + [c for c in ax.collections if hasattr(c, "get_cmap")]:
            cm = m.get_cmap()
            if cm is not None and cm.name.removesuffix("_r") in BANNED_CMAPS:
                issues.append(f"Colormap '{cm.name}' is not perceptually uniform: use viridis/cividis.")

        # 7. Legends: frameless, and not covering data
        leg = ax.get_legend()
        if leg is not None:
            if leg.get_frame_on():
                issues.append("Legend has a frame; set frameon=False.")
            lb = leg.get_window_extent(r)
            pts = []
            for ln in ax.lines:
                xy = ln.get_xydata()
                if len(xy):
                    pts.append(ln.get_transform().transform(xy))
            for c in ax.collections:
                offs = c.get_offsets()
                if len(offs):
                    pts.append(c.get_offset_transform().transform(offs))
            if pts:
                P = np.vstack(pts)
                inside = ((P[:, 0] > lb.x0) & (P[:, 0] < lb.x1) & (P[:, 1] > lb.y0) & (P[:, 1] < lb.y1))
                if inside.any():
                    issues.append(f"'{name}': legend covers {inside.sum()} data points: move it "
                                  "outside or use direct labels.")

        # 7b. In-axes text (annotations, stats) sitting on top of data points
        pts_all = []
        for c in ax.collections:
            offs = c.get_offsets()
            if len(offs) and not isinstance(c, mpl.collections.QuadMesh):
                pts_all.append(c.get_offset_transform().transform(offs))
        if pts_all:
            P = np.vstack(pts_all)
            skip = {id(ax.xaxis.label), id(ax.yaxis.label), id(ax.title),
                    id(ax._left_title), id(ax._right_title)}
            for t in ax.texts:
                if id(t) in skip or not t.get_visible() or not t.get_text().strip():
                    continue
                if t.get_gid() == "panel_label":
                    continue
                b = _text_bbox(t, r)
                hit = ((P[:, 0] > b.x0) & (P[:, 0] < b.x1) & (P[:, 1] > b.y0) & (P[:, 1] < b.y1)).sum()
                if hit:
                    issues.append(f"'{name}': text '{t.get_text()[:18]}' sits on {hit} data point(s).")

    # 8. Hairlines
    lw = [l.get_linewidth() * scale for ax in fig.axes for l in ax.lines
          if l.get_visible() and l.get_linestyle() not in ("None", "none", " ", "")
          and l.get_linewidth() > 0]
    if lw and min(lw) < 0.25:
        issues.append(f"A line prints at {min(lw):.2f} pt (< 0.25 pt) and may vanish.")

    issues = list(dict.fromkeys(issues))
    if verbose:
        print("audit: OK" if not issues else "audit:\n  - " + "\n  - ".join(issues))
    return issues


# --------------------------------------------------------------------------
# Export
# --------------------------------------------------------------------------
def _sha256(path, chunk=1 << 20):
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def save(fig, stem: str | Path, formats=("pdf", "svg", "png"), dpi: int = 600,
         metadata: dict | None = None, sources=None, caption: str | None = None,
         cvd: bool = False) -> list[Path]:
    """Save vector + raster versions side by side and a provenance record.

    sources: data files the figure was built from (hashed into
             <stem>.provenance.json so the figure can be traced to its data).
    caption: draft caption text, written to <stem>.caption.md.
    cvd:     also write <stem>_cvd.png (colour-vision / greyscale preview).
    """
    import datetime, json, platform
    finalize(fig)
    stem = Path(stem)
    stem.parent.mkdir(parents=True, exist_ok=True)
    out = []
    for ext in formats:
        p = stem.with_suffix(f".{ext}")
        kw = {"dpi": dpi} if ext in ("png", "tif", "tiff", "jpg") else {}
        if ext == "pdf":
            kw["metadata"] = {"Creator": "scientific-visualization skill", **(metadata or {})}
        fig.savefig(p, **kw)
        out.append(p)

    versions = {"python": platform.python_version(), "matplotlib": mpl.__version__,
                "numpy": np.__version__}
    for mod in ("pandas", "scipy", "seaborn"):
        try:
            versions[mod] = __import__(mod).__version__
        except ImportError:
            pass
    import sys as _sys
    prov = {
        "created": datetime.datetime.now().isoformat(timespec="seconds"),
        "script": str(Path(_sys.argv[0]).resolve()) if _sys.argv and _sys.argv[0] else None,
        "argv": _sys.argv[1:],
        "journal": _STATE["journal"], "readability": _STATE["readability"],
        "size_mm": [round(fig.get_figwidth() / MM, 1), round(fig.get_figheight() / MM, 1)],
        "sources": [{"path": str(Path(sp).resolve()), "sha256": _sha256(sp),
                     "bytes": Path(sp).stat().st_size} for sp in (sources or [])],
        "versions": versions,
        "outputs": [p.name for p in out],
    }
    pp = stem.parent / f"{stem.name}.provenance.json"
    pp.write_text(json.dumps(prov, indent=2))
    out.append(pp)
    if caption:
        cp = stem.parent / f"{stem.name}.caption.md"
        cp.write_text(caption.strip() + "\n")
        out.append(cp)
    if cvd:
        out.append(cvd_preview(fig, stem.parent / f"{stem.name}_cvd.png"))
    print("saved:", ", ".join(str(p) for p in out))
    return out

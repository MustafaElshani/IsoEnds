"""The look shared by every IsoEnds figure: the pubfig style, the colours of the IsoEnds logo, square panels, the
volcano and the tail violins, and a save step that audits the figure and writes it with its caption.

Teal marks the control, a decrease and 5' ends; orange marks the treatment, an increase and 3' ends."""
from __future__ import annotations

import contextlib
import io
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import pubfig as pf
from ..util import LOG

TEAL, ORANGE, SLATE, INK = "#0891B2", "#EA580C", "#64748B", "#1E293B"
CONTROL, TREATED = "#7CCBDD", "#F6A77A"
CONTROL_DARK, TREATED_DARK = "#0E7490", "#C2410C"
UPDOWN = {"up": ORANGE, "down": TEAL, "ns": "#C9CED6"}
END_COLOR, START_COLOR = ORANGE, TEAL
CLASS_COLORS = {"terminal": SLATE, "canonical first exon": SLATE}
KIND_COLORS = {"NMD": ORANGE, "retained intron": TEAL}
CATEGORY_ORDER = ["protein coding", "lncRNA", "retained intron", "NMD"]
MM_LEFT, MM_BOTTOM, MM_TOP = 13.0, 10.0, 6.0


def class_color(site_class: str) -> str:
    """The colour of a site class: slate for the reference sites, orange for other poly(A) classes and teal for
    other start-site classes."""
    if site_class in CLASS_COLORS:
        return CLASS_COLORS[site_class]
    return START_COLOR if "TSS" in site_class or "first exon" in site_class else END_COLOR


def square_aspect(rows: int, cols: int, width: str | float = "double", under: float = 0.0,
                  shared: bool = False) -> float:
    """The figure aspect (height / width) that leaves `rows` x `cols` square axes room for their labels. under:
    height of what sits under each panel, as a share of the panel. shared: only the outer panels carry labels."""
    w = width_mm(width)
    left, bottom = (4.0, 3.0) if shared else (MM_LEFT, MM_BOTTOM)
    side = (w - (MM_LEFT - left if shared else 0)) / cols - left
    return (rows * (side * (1 + under) + bottom + MM_TOP) + (MM_BOTTOM - bottom if shared else 0)) / w


def tail_limits(values, pad: float = 10.0) -> tuple[float, float]:
    """Axis limits (nt) that hold the 1st to 99th percentile of the tails, rounded to 10 nt."""
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if not len(v):
        return 0.0, 300.0
    lo, hi = np.percentile(v, [1, 99])
    return float(max(0.0, np.floor((lo - pad) / 10) * 10)), float(np.ceil((hi + pad) / 10) * 10)


def labels_fit(labels, slot_mm: float) -> bool:
    """Whether the longest of `labels` fits in a slot of this width at the tick font size."""
    longest = max((len(part) for label in labels for part in str(label).split("\n")), default=0)
    return longest * 0.7 * pf.sizes()["tick"] * 0.3528 + 1.5 <= slot_mm


def tail_violins(ax, data, x: str, order, hue: str, hue_order, limits=None, legend: bool = True,
                 ylabel: str = "Median poly(A) tail\nper transcript (nt)", slanted: bool = False,
                 medians: str = "above"):
    """Split violins of the column `tail`, control left and treated right, on an axis limited to `limits`
    (default: tail_limits of the data). The medians are printed for up to 12 violins: "above" their bars, or
    "below" the violins when the violins are narrow. slanted: draw the x labels at 30 degrees."""
    lo, hi = limits or tail_limits(data["tail"])
    pf.split_violin(ax, data, x, "tail", hue, order=list(order), hue_order=list(hue_order), colors=[CONTROL, TREATED],
                    dark_colors=[CONTROL_DARK, TREATED_DARK], test=None, median_labels=len(order) <= 12,
                    label_pos=medians, legend=legend)
    ax.set_ylim(lo, hi)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("")
    ax._pf_autorotate = False
    ax.set_xticks(range(len(order)), list(order), rotation=30 if slanted else 0, va="top",
                  ha="right" if slanted else "center", rotation_mode="anchor" if slanted else "default")
    return lo, hi


@dataclass
class Figure:
    name: str
    title: str
    caption: str
    paths: list = field(default_factory=list)
    audit: list = field(default_factory=list)


def setup(journal: str = "cell") -> None:
    pf.apply_style(journal=journal)


def width_mm(kind: str | float = "double") -> float:
    return pf.JOURNALS[pf._STATE["journal"]][kind] if isinstance(kind, str) else float(kind)


def save(fig, outdir: Path, name: str, title: str, caption: str, formats=("pdf", "svg", "png"), dpi: int = 600,
         sources=(), accept=()) -> Figure:
    """Audit the figure and save <outdir>/figures/<name>.<format> with its caption and provenance record. Audit
    notes containing a string of `accept` are not reported."""
    folder = Path(outdir) / "figures"
    folder.mkdir(parents=True, exist_ok=True)
    accepted = tuple(accept) + tuple(getattr(fig, "_accepted_audit", ()))
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        found = pf.audit(fig, verbose=False)
        paths = pf.save(fig, folder / name, formats=tuple(formats), dpi=dpi, sources=[str(s) for s in sources],
                        caption=f"**{title}.** {caption}")
    issues = [i for i in found if not any(a in i for a in accepted)]
    for i in issues:
        LOG.info("figure %s: audit: %s", name, i)
    for line in buf.getvalue().splitlines():
        if line.startswith("[pubfig] skipped"):
            LOG.info("figure %s: %s", name, line.strip())
    plt.close(fig)
    return Figure(name=name, title=title, caption=caption, paths=[str(p) for p in paths], audit=issues)


def symmetric_limits(effect, cap: float | None = None) -> tuple[float, float]:
    """Symmetric axis limits holding the 99.5th percentile of |effect|, at most `cap`."""
    v = np.abs(np.asarray(effect, float))
    v = v[np.isfinite(v)]
    lim = (float(np.percentile(v, 99.5)) * 1.15 if len(v) else 0.0) or 1.0
    return (-min(lim, cap), min(lim, cap)) if cap else (-lim, lim)


def fmt_q(q: float) -> str:
    return pf.fmt_p(q).replace("P", "q", 1) if np.isfinite(q) else "q n/a"


def fmt_n(x) -> str:
    return f"{int(x):,}"


def neglog(q, floor: float = 1e-300):
    return -np.log10(np.clip(np.asarray(q, float), floor, 1))


def volcano(ax, effect, q, names=None, alpha: float = 0.05, label=(), n_label: int = 8, xlabel: str = "",
            up_text: str = "up", down_text: str = "down", point_size: float = 4.0, xlim=None, ylim=None,
            corner_counts: bool = True, ycap: float | None = None, square: bool = True, ticks: int = 5):
    """A volcano: effect against -log10 q, non-significant points grey, the counts each way in the corners, and the
    `n_label` strongest hits and the `label` genes named. With `ycap`, -log10 q above it is drawn at the cap as a
    triangle. The axes are square unless square=False; `ticks` is the number of tick intervals aimed for. Returns
    (n_up, n_down)."""
    effect = np.asarray(effect, float)
    q = np.asarray(q, float)
    ok = np.isfinite(effect) & np.isfinite(q)
    effect, q = effect[ok], q[ok]
    names = np.asarray(names)[ok] if names is not None else None
    y = neglog(q)
    capped = np.zeros(len(y), bool)
    if ycap is not None:
        capped = y > ycap
        y = np.minimum(y, ycap)
    sig = q < alpha
    up, dn = sig & (effect > 0), sig & (effect < 0)
    for sel, c, s_ in ((~sig, UPDOWN["ns"], point_size * 0.7), (dn, UPDOWN["down"], point_size),
                       (up, UPDOWN["up"], point_size)):
        ax.scatter(effect[sel & ~capped], y[sel & ~capped], s=s_, color=c, lw=0, zorder=2 if c != UPDOWN["ns"] else 1,
                   rasterized=True)
        if (sel & capped).any():
            ax.scatter(effect[sel & capped], y[sel & capped], s=s_ * 1.4, color=c, lw=0, marker="^", zorder=2,
                       rasterized=True)
    ax.axvline(0, color=pf.INK["axis"], lw=0.5, zorder=0)
    if sig.any():
        pf.refline(ax, y=-np.log10(alpha))
    if xlim is None and len(effect):
        xlim = symmetric_limits(effect)
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is None and len(y):
        ylim = (0, max(float(np.nanmax(y)) * (1.08 if ycap is not None and capped.any() else 1.15), 1.5))
    if ylim is not None:
        ax.set_ylim(*ylim)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("$-$log$_{10}$ $q$")
    if corner_counts:
        fs = pf.sizes()["annot"]
        ax.text(0.02, 0.98, f"{int(dn.sum()):,} {down_text}", transform=ax.transAxes, ha="left", va="top",
                color=UPDOWN["down"], fontsize=fs)
        ax.text(0.98, 0.98, f"{int(up.sum()):,} {up_text}", transform=ax.transAxes, ha="right", va="top",
                color=UPDOWN["up"], fontsize=fs)
    if names is not None and len(effect):
        pick = set()
        if n_label:
            s = np.where(sig)[0]
            order = s[np.argsort(-(y[s] * np.abs(effect[s])))][:n_label]
            pick |= set(order.tolist())
        want = set(label)
        pick |= {i for i, n in enumerate(names) if n in want}
        pick = sorted(pick)
        if pick:
            idx = np.array(pick)
            ax.scatter(effect[idx], y[idx], s=point_size * 3, facecolor="none", edgecolor=pf.INK["text"], lw=0.5,
                       zorder=3)
            pf.label_points(ax, effect[idx], y[idx], names[idx], drop_crowded=True)
    pf.tidy_ticks(ax, nbins=ticks)
    if square:
        ax.set_box_aspect(1)
    return int(up.sum()), int(dn.sum())

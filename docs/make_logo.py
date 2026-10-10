"""Draw the IsoEnds logo.

    python docs/make_logo.py

Writes docs/img/isoends_logo.svg and isoends_logo_dark.svg (mark, name and tagline, one per page colour),
isoends_icon.svg (the square mark, for both page colours) and a PNG of each.

The mark is one gene's isoforms: the shared middle exons are grey and the two ends carry the colour. Each isoform
starts at a transcription start site (bent arrow) and ends at its own poly(A) site with a tail of its own length,
measured by the line under it.

Text is converted to outlines. Fonts: Georgia Bold (name), Liberation Mono Bold (the As), Liberation Sans (tagline).
The PNGs need cairosvg.
"""
from __future__ import annotations

from pathlib import Path

from matplotlib.font_manager import FontProperties, findfont
from matplotlib.path import Path as MPath
from matplotlib.textpath import TextPath, text_to_path

OUT = Path(__file__).resolve().parent / "img"

COLORS = {
    "light": dict(ink="#1E293B", start="#0891B2", end="#EA580C", body="#64748B", intron="#94A3B8", rule="#94A3B8",
                  tagline="#94A3B8"),
    "dark": dict(ink="#E6EDF3", start="#22B8D8", end="#F97316", body="#8391A7", intron="#64748B", rule="#8391A7",
                 tagline="#94A3B8"),
}
FONTS = {"name": dict(family="Georgia", weight="bold"), "tail": dict(family="Liberation Mono", weight="bold"),
         "tagline": dict(family="Liberation Sans", weight="normal")}

# icon colours: each has a contrast of at least 3:1 against white and against #0D1117
ICON_COLORS = dict(start="#0E9BBF", end="#EA580C", body="#6F7F96", intron="#7C8AA0", rule="#7C8AA0")

W = 340                      # width of the lockup
EXON_H, EXON_R = 12, 3
A_SIZE, A_STEP, A_GAP = 11.5, 8.2, 6.5
ROWS = (26, 62, 98)          # top of each isoform
# (exons as (x0, x1, part), number of As); part is "start", "body" or "end"
ISOFORMS = (
    (((16, 52, "start"), (116, 144, "body"), (170, 202, "body"), (226, 262, "end")), 7),
    (((68, 94, "start"), (116, 144, "body"), (170, 202, "body"), (226, 246, "end")), 4),
    (((16, 52, "start"), (116, 144, "body"), (170, 216, "end")), 2),
)
# the icon: the three isoforms without their middle exons, 1.35 times heavier, in a 128 x 128 square
ICON, ICON_K = 128, 1.35
ICON_ROWS = (20, 62, 104)
ICON_ISOFORMS = (
    (((6, 34, "start"), (52, 78, "end")), 3),
    (((22, 42, "start"), (52, 66, "end")), 4),
    (((6, 34, "start"), (52, 72, "end")), 2),
)
TAGLINE = "5′ START SITES · POLY(A) SITES · POLY(A) TAILS 3′"


def _prop(kind: str, size: float) -> FontProperties:
    prop = FontProperties(size=size, **FONTS[kind])
    findfont(prop, fallback_to_default=False)
    return prop


def _d(path: MPath, x: float, y: float) -> str:
    """An SVG path from a matplotlib text path, moved to (x, y) with y running down."""
    out = []
    for verts, code in path.iter_segments():
        pts = " ".join(f"{x + verts[i]:.2f} {y - verts[i + 1]:.2f}" for i in range(0, len(verts), 2))
        out.append({MPath.MOVETO: "M", MPath.LINETO: "L", MPath.CURVE3: "Q", MPath.CURVE4: "C"}.get(code, "Z")
                   + ("" if code == MPath.CLOSEPOLY else pts))
    return "".join(out)


def _advance(text: str, prop: FontProperties) -> float:
    return text_to_path.get_text_width_height_descent(text, prop, ismath=False)[0]


def glyphs(text: str, kind: str, size: float, x: float, baseline: float, fill: str, tracking: float = 0.0,
           anchor: str = "start") -> tuple[str, float]:
    """`text` as outlines. With tracking the letters are set one by one. Returns the SVG and the advance width."""
    prop = _prop(kind, size)
    if not tracking:
        width = _advance(text, prop)
        x0 = x - {"start": 0, "middle": width / 2, "end": width}[anchor]
        return f'<path d="{_d(TextPath((0, 0), text, prop=prop), x0, baseline)}" fill="{fill}"/>', width
    steps = [_advance(c, prop) + tracking for c in text]
    width = sum(steps) - tracking
    cursor = x - {"start": 0, "middle": width / 2, "end": width}[anchor]
    d = []
    for c, step in zip(text, steps):
        if c.strip():
            d.append(_d(TextPath((0, 0), c, prop=prop), cursor, baseline))
        cursor += step
    return f'<path d="{"".join(d)}" fill="{fill}"/>', width


def isoform(y: float, exons, n_as: int, c: dict, k: float = 1.0) -> list[str]:
    """One isoform: a bent arrow at its start site, exons joined by intron arcs (the first and last with a flat
    outer edge), and the poly(A) tail with its measuring line. k scales everything but the x positions."""
    h, r = EXON_H * k, EXON_R * k
    mid = y + h / 2
    out = []
    for (_, a1, _), (b0, _, _) in zip(exons, exons[1:]):
        out.append(f'<path d="M {a1} {mid:.1f} Q {(a1 + b0) / 2} {mid - min(20 * k, (b0 - a1) * 0.62 * k):.1f} {b0} '
                   f'{mid:.1f}" fill="none" stroke="{c["intron"]}" stroke-width="{1.3 * k:.2f}"/>')
    for i, (x0, x1, part) in enumerate(exons):
        if i == 0:
            out.append(f'<path d="M {x0} {y} H {x1 - r:.1f} q {r:.1f} 0 {r:.1f} {r:.1f} v {h - 2 * r:.1f} '
                       f'q 0 {r:.1f} {-r:.1f} {r:.1f} H {x0} Z" fill="{c[part]}"/>')
        elif i == len(exons) - 1:
            out.append(f'<path d="M {x0 + r:.1f} {y} H {x1} v {h:.1f} H {x0 + r:.1f} q {-r:.1f} 0 {-r:.1f} {-r:.1f} '
                       f'v {-(h - 2 * r):.1f} q 0 {-r:.1f} {r:.1f} {-r:.1f} Z" fill="{c[part]}"/>')
        else:
            out.append(f'<rect x="{x0}" y="{y}" width="{x1 - x0}" height="{h:.1f}" rx="{r:.1f}" fill="{c[part]}"/>')
    w = 1.9 * k
    x0 = exons[0][0] + w / 2
    top, tip = y - 8.5 * k, x0 + 12.5 * k
    out.append(f'<path d="M {x0:.2f} {y + 1} V {top:.2f} H {tip - 3 * k:.2f}" fill="none" stroke="{c["start"]}" '
               f'stroke-width="{w:.2f}" stroke-linecap="butt" stroke-linejoin="round"/>')
    out.append(f'<path d="M {tip + 2.2 * k:.2f} {top:.2f} l {-5.8 * k:.2f} {-3.5 * k:.2f} v {7 * k:.2f} Z" '
               f'fill="{c["start"]}"/>')
    x1 = exons[-1][1]
    prop = _prop("tail", A_SIZE * k)
    a_width = _advance("A", prop)
    first = x1 + A_GAP * k
    d = "".join(_d(TextPath((0, 0), "A", prop=prop), first + i * A_STEP * k, mid + A_SIZE * k * 0.3)
                for i in range(n_as))
    out.append(f'<path d="{d}" fill="{c["end"]}"/>')
    last = first + (n_as - 1) * A_STEP * k + a_width
    rule, tick = y + h + 5.5 * k, 2.6 * k
    out.append(f'<path d="M {first:.1f} {rule:.1f} H {last:.1f} M {first:.1f} {rule - tick:.1f} v {2 * tick:.1f} '
               f'M {last:.1f} {rule - tick:.1f} v {2 * tick:.1f}" fill="none" stroke="{c["rule"]}" '
               f'stroke-width="{k:.2f}" stroke-linecap="round"/>')
    return out


def lockup(mode: str) -> str:
    c = COLORS[mode]
    parts = []
    for y, (exons, n_as) in zip(ROWS, ISOFORMS):
        parts += ["<g>"] + isoform(y, exons, n_as, c) + ["</g>"]
    size, baseline = 44, 171
    iso_w = _advance("Iso", _prop("name", size))
    x0 = W / 2 - _advance("IsoEnds", _prop("name", size)) / 2
    parts.append(glyphs("Iso", "name", size, x0, baseline, c["ink"])[0])
    parts.append(glyphs("Ends", "name", size, x0 + iso_w, baseline, c["end"])[0])
    parts.append(glyphs(TAGLINE, "tagline", 7.6, W / 2, baseline + 21, c["tagline"], tracking=1.55,
                        anchor="middle")[0])
    height = baseline + 32
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {height}" width="{W}" height="{height}" '
            f'role="img" aria-label="IsoEnds">\n  ' + "\n  ".join(parts) + "\n</svg>\n")


def icon() -> str:
    parts = []
    for y, (exons, n_as) in zip(ICON_ROWS, ICON_ISOFORMS):
        parts += isoform(y, exons, n_as, ICON_COLORS, ICON_K)
    return (f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {ICON} {ICON}" width="{ICON}" height="{ICON}" '
            f'role="img" aria-label="IsoEnds">\n  ' + "\n  ".join(parts) + "\n</svg>\n")


def contrast(a: str, b: str) -> float:
    def lum(c):
        v = [int(c[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        v = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in v]
        return 0.2126 * v[0] + 0.7152 * v[1] + 0.0722 * v[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def main() -> None:
    import cairosvg
    OUT.mkdir(parents=True, exist_ok=True)
    for mode, stem in (("light", "isoends_logo"), ("dark", "isoends_logo_dark")):
        svg = lockup(mode)
        (OUT / f"{stem}.svg").write_text(svg)
        cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT / f"{stem}.png"), output_width=1360)
        print(f"wrote {stem}.svg/.png ({len(svg) / 1000:.1f} kB)")
    svg = icon()
    (OUT / "isoends_icon.svg").write_text(svg)
    cairosvg.svg2png(bytestring=svg.encode(), write_to=str(OUT / "isoends_icon.png"), output_width=512)
    print(f"wrote isoends_icon.svg/.png ({len(svg) / 1000:.1f} kB); contrast on white / dark: " + ", ".join(
        f"{k} {contrast(v, '#FFFFFF'):.1f}/{contrast(v, '#0D1117'):.1f}" for k, v in ICON_COLORS.items()))


if __name__ == "__main__":
    main()

"""Regenerate the README wordmark: ``assets/wordmark-light.svg`` and ``-dark.svg``.

    uv run python scripts/wordmark.py

Sets "Sidechain" in Geist Pixel over a Geist Mono kicker, in the saberhq.com type and
colour tokens, and outlines every glyph to a path -- so the SVG carries no font and renders
the same everywhere GitHub shows it. The README picks the variant with ``<picture>`` and
``prefers-color-scheme``.

Geist (SIL OFL) is fetched from Google Fonts into a temp dir on first run. Geist Pixel draws
each pixel as its own rounded contour on a 38-unit grid, so instead of tracing those curves
the wordmark is rebuilt from the grid: one cell -> one PX_PER_CELL-pixel square, adjacent
cells merged into runs. Same picture at this size, a fiftieth of the bytes, and crisp at 1x
and 2x. Needs fontTools, which the project env already carries.
"""
from __future__ import annotations

import math
import re
import tempfile
import urllib.request
from pathlib import Path

from fontTools.pens.recordingPen import DecomposingRecordingPen
from fontTools.pens.svgPathPen import SVGPathPen
from fontTools.pens.transformPen import TransformPen
from fontTools.ttLib import TTFont

OUT_DIR = Path(__file__).resolve().parent.parent / "assets"
CACHE = Path(tempfile.gettempdir()) / "sidechain-fonts"

# saberhq.com tokens (assets/tokens/colors.css there): ink, ink-3, accent.
PALETTES = {
    "light": {"ink": "#191613", "muted": "#6F6A62", "accent": "#D55E00"},
    "dark": {"ink": "#ECE7DF", "muted": "#999183", "accent": "#F08A3E"},
}

KICKER = "VIRTUAL CELL CHALLENGE · 2026"
WORD, MARK = "Sidechain", "."  # the full stop takes the accent

GRID = 38  # Geist Pixel's pixel grid, in font units (1000/em)
PX_PER_CELL = 2
KICKER_PX, KICKER_TRACK = 12.0, 0.14  # Geist Mono 500, tracked like the site's labels
GAP = 14  # kicker baseline -> wordmark top


def fetch_font(query: str) -> TTFont:
    """A static TTF from Google Fonts, e.g. ``Geist+Pixel`` or ``Geist+Mono:wght@500``."""
    CACHE.mkdir(parents=True, exist_ok=True)
    path = CACHE / (re.sub(r"[^A-Za-z0-9]+", "_", query) + ".ttf")
    if not path.exists():
        # No browser User-Agent -> the CSS API answers with plain TrueType URLs.
        css = urllib.request.urlopen(f"https://fonts.googleapis.com/css2?family={query}").read().decode()
        url = re.search(r"https://fonts\.gstatic\.com[^)]+", css).group(0)
        path.write_bytes(urllib.request.urlopen(url).read())
    return TTFont(path)


def _num(v: float) -> str:
    return f"{v:.2f}".rstrip("0").rstrip(".")


def outline_text(font: TTFont, text: str, px: float, x: float, baseline: float, track_em: float = 0.0):
    """Trace ``text`` at ``px`` pixels; returns (path_d per glyph, end_x)."""
    s = px / font["head"].unitsPerEm
    cmap, gs, hmtx = font.getBestCmap(), font.getGlyphSet(), font["hmtx"]
    paths = []
    for ch in text:
        name = cmap[ord(ch)]
        pen = SVGPathPen(gs, ntos=_num)
        gs[name].draw(TransformPen(pen, (s, 0, 0, -s, x, baseline)))
        if d := pen.getCommands():
            paths.append(d)
        x += hmtx[name][0] * s + track_em * px
    return paths, x - track_em * px


def glyph_cells(font: TTFont, name: str) -> set[tuple[int, int]]:
    """Grid cells a Geist Pixel glyph fills: one contour per pixel, read off its bounding box."""
    pen = DecomposingRecordingPen(font.getGlyphSet())
    font.getGlyphSet()[name].draw(pen)
    cells, pts = set(), []
    for op, args in pen.value + [("closePath", ())]:
        if op in ("moveTo", "closePath", "endPath") and pts:
            xs, ys = zip(*pts)
            w, h = max(xs) - min(xs), max(ys) - min(ys)
            if not (0.8 * GRID <= w <= GRID and 0.8 * GRID <= h <= GRID):
                raise ValueError(f"{name}: contour {w:.0f}x{h:.0f} is not one grid cell")
            cells.add((round(min(xs) / GRID), round(min(ys) / GRID)))
            pts = []
        pts += [p for p in args if isinstance(p, tuple)]
    return cells


def pixel_text(font: TTFont, text: str, x: int, baseline: int):
    """Rebuild ``text`` from grid cells; returns (path_d per glyph, end_x, top_y)."""
    cmap, hmtx = font.getBestCmap(), font["hmtx"]
    paths, top = [], baseline
    for ch in text:
        name = cmap[ord(ch)]
        rows: dict[int, list[int]] = {}
        for cx, cy in glyph_cells(font, name):
            rows.setdefault(cy, []).append(cx)
        runs = []
        for cy, xs in sorted(rows.items(), reverse=True):
            xs.sort()
            start = prev = xs[0]
            for cx in xs[1:] + [None]:
                if cx != prev + 1:
                    px, py = x + start * PX_PER_CELL, baseline - (cy + 1) * PX_PER_CELL
                    runs.append(f"M{px} {py}h{(prev - start + 1) * PX_PER_CELL}v{PX_PER_CELL}h-{(prev - start + 1) * PX_PER_CELL}z")
                    top = min(top, py)
                    if cx is not None:
                        start = cx
                prev = cx if cx is not None else prev
        if runs:
            paths.append("".join(runs))
        adv = hmtx[name][0]
        if adv % GRID:
            raise ValueError(f"{name}: advance {adv} is off the {GRID}-unit grid")
        x += adv // GRID * PX_PER_CELL
    return paths, x, top


def build() -> dict[str, str]:
    pixel = fetch_font("Geist+Pixel")
    mono = fetch_font("Geist+Mono:wght@500")

    kicker_base = round(mono["OS/2"].sCapHeight / mono["head"].unitsPerEm * KICKER_PX) + 1
    kicker, kicker_w = outline_text(mono, KICKER, KICKER_PX, 0, kicker_base, KICKER_TRACK)

    # Measure once to find the wordmark's height, then place its top GAP below the kicker.
    _, _, top0 = pixel_text(pixel, WORD + MARK, 0, 0)
    word_base = kicker_base + GAP - top0
    word, word_end, _ = pixel_text(pixel, WORD, 0, word_base)
    mark, mark_end, _ = pixel_text(pixel, MARK, word_end, word_base)

    width = math.ceil(max(kicker_w, mark_end)) + 1
    height = word_base + 2  # nothing in "Sidechain." descends below the baseline

    out = {}
    for name, c in PALETTES.items():
        body = [
            (
                f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
                f'viewBox="0 0 {width} {height}" role="img" aria-label="Sidechain - {KICKER}">'
            ),
            "<title>Sidechain</title>",
            f'<g fill="{c["muted"]}">',
            *(f'<path d="{d}"/>' for d in kicker),
            "</g>",
            f'<g fill="{c["ink"]}" shape-rendering="crispEdges">',
            *(f'<path d="{d}"/>' for d in word),
            "</g>",
            f'<g fill="{c["accent"]}" shape-rendering="crispEdges">',
            *(f'<path d="{d}"/>' for d in mark),
            "</g>",
            "</svg>",
        ]
        out[name] = "\n".join(body) + "\n"
    return out


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    for name, svg in build().items():
        path = OUT_DIR / f"wordmark-{name}.svg"
        path.write_text(svg)
        print(f"wrote {path.relative_to(OUT_DIR.parent)} ({len(svg):,} bytes)")


if __name__ == "__main__":
    main()

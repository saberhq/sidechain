"""Regenerate the wordmark assets: the README's SVGs and the site's favicon + social card.

    uv run python scripts/wordmark.py

Sets "Sidechain" in Geist Pixel over a Geist Mono kicker, in the saberhq.com type and
colour tokens, and outlines every glyph to a path -- so the SVG carries no font and renders
the same everywhere GitHub shows it. The README picks the variant with ``<picture>`` and
``prefers-color-scheme``. The site (``site/``) sets the same wordmark live, from the web
fonts; what it needs from here is the pixel grid in forms a browser can't set from CSS:
``site/static/favicon.svg`` + ``.png`` and ``apple-touch-icon.png`` ("S." in the accent),
and ``site/static/og.png``, the 1200x630 card social networks show when the page is shared.

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
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).resolve().parent.parent / "assets"
SITE_STATIC = Path(__file__).resolve().parent.parent / "site" / "static"
CACHE = Path(tempfile.gettempdir()) / "sidechain-fonts"

# saberhq.com tokens (assets/tokens/colors.css there): ink, ink-3, accent.
PALETTES = {
    "light": {"ink": "#191613", "muted": "#6F6A62", "accent": "#D55E00"},
    "dark": {"ink": "#ECE7DF", "muted": "#999183", "accent": "#F08A3E"},
}
PAPER, INK_2 = "#FAF9F7", "#45403A"  # --paper, --ink-2 (light); the social card is light-only

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


def cells_for(font: TTFont, text: str) -> tuple[list[tuple[int, int]], int]:
    """Grid cells ``text`` fills, laid out on the pixel grid (y up); and its advance in cells."""
    cmap, hmtx = font.getBestCmap(), font["hmtx"]
    cells, x = [], 0
    for ch in text:
        name = cmap[ord(ch)]
        cells += [(x + cx, cy) for cx, cy in glyph_cells(font, name)]
        adv = hmtx[name][0]
        if adv % GRID:
            raise ValueError(f"{name}: advance {adv} is off the {GRID}-unit grid")
        x += adv // GRID
    return cells, x


def _bbox(cells):
    xs, ys = zip(*cells)
    return min(xs), min(ys), max(xs) + 1, max(ys) + 1


def favicon_svg(font: TTFont, fill: str) -> str:
    """``S.`` as grid squares in a square viewBox, one colour so it reads on any tab bar."""
    cells, _ = cells_for(font, WORD[0] + MARK)
    x0, y0, x1, y1 = _bbox(cells)
    w, h = x1 - x0, y1 - y0
    side = max(w, h) + 2  # one cell of air all round
    ox, oy = (side - w) / 2 - x0, (side - h) / 2
    rects = "".join(
        f'<rect x="{_num(ox + cx)}" y="{_num(oy + (y1 - 1 - cy))}" width="1" height="1"/>'
        for cx, cy in sorted(cells)
    )
    return (
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {side} {side}" shape-rendering="crispEdges">'
        f'<g fill="{fill}">{rects}</g></svg>\n'
    )


def draw_cells(img: Image.Image, cells, px: int, left: int, top: int, fill: str) -> None:
    """Paint grid cells as ``px``-pixel squares; (left, top) is the cell bounding box's corner."""
    x0, _, _, y1 = _bbox(cells)
    d = ImageDraw.Draw(img)
    for cx, cy in cells:
        x, y = left + (cx - x0) * px, top + (y1 - 1 - cy) * px
        d.rectangle([x, y, x + px - 1, y + px - 1], fill=fill)


def favicon_png(font: TTFont, size: int, fill: str, background: str | None) -> Image.Image:
    cells, _ = cells_for(font, WORD[0] + MARK)
    x0, y0, x1, y1 = _bbox(cells)
    w, h = x1 - x0, y1 - y0
    px = max(1, int(size * (0.6 if background else 0.85)) // max(w, h))
    img = Image.new("RGBA", (size, size), background or (0, 0, 0, 0))
    draw_cells(img, cells, px, (size - w * px) // 2, (size - h * px) // 2, fill)
    return img


def tracked_text(d: ImageDraw.ImageDraw, xy, text: str, font, fill: str, track: float = 0.0) -> float:
    """PIL has no letter-spacing; place glyphs one by one. Returns the end x."""
    x, y = xy
    for ch in text:
        d.text((x, y), ch, font=font, fill=fill, anchor="ls")
        x += font.getlength(ch) + track
    return x


def social_card(pixel: TTFont, mono: TTFont, sans_path: Path, mono_path: Path) -> Image.Image:
    """1200x630: kicker, wordmark, tagline, footer strip -- the README header as a picture."""
    W, H, M = 1200, 630, 96
    c = PALETTES["light"]
    img = Image.new("RGB", (W, H), PAPER)
    d = ImageDraw.Draw(img)

    kicker_font = ImageFont.truetype(str(mono_path), 24)
    tracked_text(d, (M, 150), KICKER, kicker_font, c["muted"], track=24 * KICKER_TRACK)

    # Lay the word and the mark out together, then split the cells by glyph so the
    # full stop sits where the font puts it and takes the accent.
    cells, _ = cells_for(pixel, WORD + MARK)
    word_adv = cells_for(pixel, WORD)[1]
    word = [cell for cell in cells if cell[0] < word_adv]
    mark = [cell for cell in cells if cell[0] >= word_adv]
    x0, y0, x1, y1 = _bbox(cells)
    px = (W - 2 * M) // (x1 - x0)  # the whole wordmark spans the text column
    top = 150 + 36
    draw_cells(img, word, px, M, top + (y1 - _bbox(word)[3]) * px, c["ink"])
    draw_cells(img, mark, px, M + (_bbox(mark)[0] - x0) * px, top + (y1 - _bbox(mark)[3]) * px, c["accent"])
    word_bottom = top + (y1 - y0) * px

    tag_font = ImageFont.truetype(str(sans_path), 30)
    d.text((M, word_bottom + 66), "Predicting how a cell's transcriptome shifts when a gene is silenced.", font=tag_font, fill=INK_2, anchor="ls")
    d.text((M, word_bottom + 66 + 44), "A solo entry to the Virtual Cell Challenge 2026, built in the open.", font=tag_font, fill=INK_2, anchor="ls")

    d.rectangle([M, H - 84, W - M, H - 83], fill=c["ink"])  # the brand's section rule
    foot_font = ImageFont.truetype(str(mono_path), 17)
    tracked_text(d, (M, H - 50), "SABERHQ.COM  —  SIDECHAIN", foot_font, c["muted"], track=17 * 0.06)
    return img


def build_site_assets() -> list[tuple[Path, int]]:
    pixel = fetch_font("Geist+Pixel")
    mono = fetch_font("Geist+Mono:wght@500")
    mono_path = CACHE / "Geist_Mono_wght_500.ttf"
    fetch_font("Geist:wght@400")
    sans_path = CACHE / "Geist_wght_400.ttf"
    accent = PALETTES["light"]["accent"]

    SITE_STATIC.mkdir(parents=True, exist_ok=True)
    out = []
    p = SITE_STATIC / "favicon.svg"
    p.write_text(favicon_svg(pixel, accent))
    out.append((p, p.stat().st_size))
    for name, size, bg in (("favicon.png", 32, None), ("apple-touch-icon.png", 180, PAPER)):
        p = SITE_STATIC / name
        favicon_png(pixel, size, accent, bg).save(p, optimize=True)
        out.append((p, p.stat().st_size))
    p = SITE_STATIC / "og.png"
    social_card(pixel, mono, sans_path, mono_path).save(p, optimize=True)
    out.append((p, p.stat().st_size))
    return out


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
    root = OUT_DIR.parent
    for name, svg in build().items():
        path = OUT_DIR / f"wordmark-{name}.svg"
        path.write_text(svg)
        print(f"wrote {path.relative_to(root)} ({len(svg):,} bytes)")
    for path, size in build_site_assets():
        print(f"wrote {path.relative_to(root)} ({size:,} bytes)")


if __name__ == "__main__":
    main()

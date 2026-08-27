"""Cut one of Saber's photographs into a post banner.

    uv run python scripts/banner.py                          # list the pool, pick one
    uv run python scripts/banner.py photo.jpg --slug corpus  # banner for posts/corpus/
    uv run python scripts/banner.py photo.jpg --out DIR      # banner anywhere else

Every banner gets the same frame — that repetition, not a filter, is the visual
signature. The photo's pixels are untouched apart from the crop and resize:

- ``cover.jpg``  2000x800, 5:2 — the banner the post opens with (``image: cover.jpg``)
- ``social.jpg`` 1200x627, 1.91:1 — the link-preview card (``images: [social.jpg]``),
  so LinkedIn shows the same photo instead of the site-wide og.png

``--offset 0..100`` slides the crop window along whichever axis gets cut (50 = centre;
0 = left/top, 100 = right/bottom). EXIF is stripped on write — originals carry GPS —
but the capture date is read first and printed as the suggested ``caption:`` line
(the caption convention is the date the photo was taken, nothing more).

Photos come from the pool ``~/art/still/sidechain-banner/`` (run with no arguments to
list it) or from any path. JPEG/PNG/TIFF; HEIC needs an export first. Needs Pillow,
which the project env already carries for scripts/wordmark.py.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image, ImageOps

POOL = Path.home() / "art" / "still" / "sidechain-banner"
POSTS = Path(__file__).resolve().parent.parent / "site" / "content" / "posts"
CUTS = (("cover.jpg", 2000, 800), ("social.jpg", 1200, 627))
EXIF_DATE_TAGS = (36867, 36868, 306)  # DateTimeOriginal, DateTimeDigitized, DateTime


def list_pool() -> int:
    photos = sorted(p for p in POOL.glob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".tif", ".tiff"})
    if not photos:
        print(f"pool is empty: {POOL}\ndrop photos there, or pass any path directly")
        return 1
    for p in photos:
        with Image.open(p) as im:
            im = ImageOps.exif_transpose(im)
            date = taken(im) or "no date"
            print(f"{p.name:40s} {im.width}x{im.height}  {date}")
    return 0


def taken(im: Image.Image) -> str | None:
    """The capture date, as YYYY-MM-DD, from whichever EXIF date tag is present."""
    exif = im.getexif()
    for tag in EXIF_DATE_TAGS:
        raw = exif.get(tag) or exif.get_ifd(0x8769).get(tag)
        if raw:
            return str(raw)[:10].replace(":", "-")
    return None


def cut(im: Image.Image, width: int, height: int, offset: float) -> Image.Image:
    """Crop to width:height with the window slid by offset along the cut axis, then fit."""
    target = width / height
    if im.width / im.height >= target:  # source is wider — cut the sides
        crop_w = round(im.height * target)
        x = round((im.width - crop_w) * offset / 100)
        box = (x, 0, x + crop_w, im.height)
    else:  # source is taller — cut top/bottom
        crop_h = round(im.width / target)
        y = round((im.height - crop_h) * offset / 100)
        box = (0, y, 0 + im.width, y + crop_h)
    out = im.crop(box)
    if out.width > width:  # never upscale a small photo
        out = out.resize((width, height), Image.LANCZOS)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("photo", nargs="?", help="photo path; omit to list the pool")
    ap.add_argument("--slug", help="post bundle to write into: site/content/posts/<slug>/")
    ap.add_argument("--out", help="write into this directory instead of a post bundle")
    ap.add_argument("--offset", type=float, default=50, help="crop window position, 0..100 (default 50)")
    args = ap.parse_args()

    if not args.photo:
        return list_pool()
    src = Path(args.photo).expanduser()
    if not src.exists() and (POOL / args.photo).exists():
        src = POOL / args.photo  # bare pool filename works too
    if not src.exists():
        sys.exit(f"no such photo: {src}")
    if args.slug and args.out:
        sys.exit("--slug and --out are exclusive")
    if not args.slug and not args.out:
        sys.exit("say where it goes: --slug <post> or --out <dir>")
    dest = POSTS / args.slug if args.slug else Path(args.out).expanduser()
    dest.mkdir(parents=True, exist_ok=True)

    with Image.open(src) as im:
        im = ImageOps.exif_transpose(im)  # bake the rotation in before EXIF is dropped
        date = taken(im)
        im = im.convert("RGB")
        for name, w, h in CUTS:
            out = cut(im, w, h, args.offset)
            # save() writes no EXIF unless asked to, which is the point: originals carry GPS
            out.save(dest / name, quality=82, progressive=True, optimize=True)
            print(f"{dest / name}  {out.width}x{out.height}")

    print("\nfront matter:")
    print("image: cover.jpg")
    print(f'caption: "{date}"' if date else "caption: \"\"          # no EXIF date found — fill in by hand")
    print("images: [social.jpg]")
    return 0


if __name__ == "__main__":
    sys.exit(main())

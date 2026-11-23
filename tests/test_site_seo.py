"""The built site's machine-readable layer: titles, descriptions, canonical links, JSON-LD, llms.txt, Markdown copies.

    .venv/bin/python -m pytest -q tests/test_site_seo.py

Builds site/ with Hugo into a temporary directory; skips when Hugo is not installed. Also checks scripts/indexnow.py.
Notes: private/code-notes/tests/test_site_seo.py.md
"""
from __future__ import annotations

import importlib.util
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
BASE = "https://saberh.com/sidechain/"
PERSON = "https://saberh.com/#person"

_spec = importlib.util.spec_from_file_location("indexnow", REPO / "scripts" / "indexnow.py")
indexnow = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(indexnow)


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    if not shutil.which("hugo"):
        pytest.skip("hugo not installed")
    out = tmp_path_factory.mktemp("site")
    subprocess.run(["hugo", "--source", str(REPO / "site"), "--destination", str(out), "--quiet"], check=True)
    return out


def _html(built: Path, url: str) -> str:
    return (built / url.removeprefix(BASE) / "index.html").read_text()


def _graph(html: str) -> list[dict]:
    blocks = re.findall(r'<script type="?application/ld\+json"?>(.*?)</script>', html, re.S)
    assert len(blocks) == 1, f"{len(blocks)} JSON-LD blocks"
    return json.loads(blocks[0])["@graph"]


def _pages(built: Path) -> list[str]:
    return indexnow.locs((built / "sitemap.xml").read_text())


def test_every_page_has_unique_title_description_canonical(built):
    titles = {}
    for url in _pages(built):
        h = _html(built, url)
        title = re.search(r"<title>(.*?)</title>", h, re.S).group(1).strip()
        assert title, url
        assert url == BASE or title.split(" - ", 1)[-1].lower() != "sidechain", f"{url}: the title only names the site"
        assert title not in titles, f"{url} and {titles.get(title)} share the title {title!r}"
        titles[title] = url
        assert re.search(r'<meta name="?description"? content="[^"]{50,}"', h), f"{url}: no description"
        assert re.search(rf'<link rel="?canonical"? href="?{re.escape(url)}"?>', h), f"{url}: canonical"
        assert 'name="twitter:card"' in h or "name=twitter:card" in h, f"{url}: no X card"


def test_every_page_has_json_ld_with_the_author(built):
    for url in _pages(built):
        g = _graph(_html(built, url))
        types = {n["@type"] for n in g}
        assert {"Person", "WebSite", "SoftwareSourceCode"} <= types, f"{url}: {types}"
        assert any(n.get("@id") == PERSON for n in g), url
        page = [n for n in g if n.get("@id") == f"{url}#webpage"]
        assert page and page[0]["url"] == url, url


def test_posts_are_articles_and_the_controls_page_a_dataset(built):
    for url in _pages(built):
        g = {n["@type"]: n for n in _graph(_html(built, url))}
        if url.startswith(f"{BASE}posts/") and url != f"{BASE}posts/":
            a = g["BlogPosting"]
            assert a["headline"] and a["datePublished"] and a["author"] == {"@id": PERSON}, url
            assert all(i.startswith("https://") for i in a["image"]), url
            for k in ("datePublished", "dateModified"):
                assert re.fullmatch(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d[+-]\d\d:\d\d", a.get(k, "")), (url, k)
    d = {n["@type"]: n for n in _graph(_html(built, f"{BASE}vcc2026-control-cells/"))}["Dataset"]
    assert len(d["description"]) >= 50 and d["distribution"] and d["creator"] == {"@id": PERSON}
    assert len(d["isBasedOn"]["description"]) >= 50


def test_llms_txt_lists_every_page_as_markdown(built):
    text = (built / "llms.txt").read_text()
    assert text.startswith("# Sidechain\n\n> ")
    linked = re.findall(r"\]\((https://[^)]+)\)", text)
    for f in built.rglob("index.md"):
        url = BASE + str(f.relative_to(built).as_posix())
        assert url in linked, f"llms.txt does not list {url}"
    for url in linked:
        if url.startswith(BASE) and url.endswith(".md"):
            assert (built / url.removeprefix(BASE)).exists(), f"llms.txt links a missing {url}"


def test_markdown_copies_are_whole_and_absolute(built):
    copies = list(built.rglob("index.md"))
    assert len(copies) >= 4
    for f in copies:
        text = f.read_text()
        assert text.startswith("# "), f
        assert "{{<" not in text and "{{%" not in text, f"{f}: an unrendered shortcode"
        assert not re.search(r"\]\(/", text), f"{f}: a root-relative link"


def test_controls_copy_gives_its_figures_as_text(built):
    data = REPO / "site" / "static" / "data" / "controls"
    if not (data / "tables.json").exists():
        pytest.skip("the site's control-cells data is not built here")
    text = (built / "vcc2026-control-cells" / "index.md").read_text()
    assert "## The figures, in numbers" in text
    assert "&lt;" not in text and "{{" not in text
    for c, s in json.loads((data / "index.json").read_text())["contexts"].items():
        assert f"| {c} | {s['round']} |" in text, c
        assert f"| {c} | own |" in text and f"| {c} | regressed |" in text, c


def test_no_robots_file_below_the_root(built):
    assert not (built / "robots.txt").exists()


def test_indexnow_key_and_payload():
    k = indexnow.key()
    assert k and (REPO / "site" / "static" / f"{k}.txt").read_text().strip() == k
    body = indexnow.payload([f"{BASE}models/", "https://example.org/x"], k)
    assert body["urlList"] == [f"{BASE}models/"] and body["keyLocation"] == f"{BASE}{k}.txt"
    assert indexnow.locs("<urlset><url><loc> https://a/ </loc></url></urlset>") == ["https://a/"]

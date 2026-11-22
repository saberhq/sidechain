"""The site kit under site/: its copies, its lock, its file lists, and the header it draws.

    .venv/bin/python -m pytest -q tests/test_site_kit.py

Reads site/, saberhq.com's latest release over HTTPS (skips when unreachable), and a Hugo build of site/ in a
temporary directory (skips without Hugo). Writes nothing in the repo.
Notes: private/code-notes/tests/test_site_kit.py.md
"""
import importlib.util
import re
import shutil
import subprocess
import urllib.error
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
_path = str(REPO / "scripts" / "kit-pull")
_spec = importlib.util.spec_from_loader("kit_pull", SourceFileLoader("kit_pull", _path))
kit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kit)

PAGES = ("", "posts/", "posts/knobs-n-knockdowns/", "models/", "vcc2026-control-cells/")


def test_copies_match_lock():
    assert not kit.unlocked(), f"differ from site/kit.lock.json: {kit.unlocked()}; run scripts/kit-pull"


def test_no_stray_files_in_kit_folder():
    assert not kit.strays(), f"not kit files: {kit.strays()}; the kit folder holds copies only"


def test_lists_do_not_overlap():
    assert not set(kit.KIT) & set(kit.HELD)
    assert len(set(kit.KIT)) == len(kit.KIT)


def test_unlisted_shared_file_is_new_and_missing_kit_file_is_gone():
    extra = {"assets/blog.css", "data/vibes_more.json", "assets/vibes/wall.css",
             "assets/footer2.css", kit.PARTIALS + "socials.html", "data/footer.yaml"}
    paths = set(kit.KIT) | set(kit.HELD) | extra
    new, gone, held = kit.sort_shared(paths)
    assert new == ["assets/footer2.css", "data/footer.yaml", kit.PARTIALS + "socials.html"]
    assert gone == [] and held == sorted(kit.HELD)
    assert kit.sort_shared(paths - {"assets/post.css"})[1] == ["assets/post.css"]


def test_kit_matches_latest_release():
    try:
        src = kit.survey(timeout=10)
    except (urllib.error.URLError, TimeoutError, RuntimeError) as e:
        pytest.skip(f"saberhq.com unreachable: {e}")
    assert not src["new"], f"on saberhq.com {src['name']} and in no list: {src['new']}"
    assert not src["gone"], f"kit files not on saberhq.com {src['name']}: {src['gone']}"
    changed = kit.drift(src["texts"])
    assert not changed, f"differ from saberhq.com {src['name']}: {', '.join(changed)}; run scripts/kit-pull"


@pytest.fixture(scope="module")
def built(tmp_path_factory) -> Path:
    if not shutil.which("hugo"):
        pytest.skip("hugo not installed")
    out = tmp_path_factory.mktemp("site")
    subprocess.run(["hugo", "--source", str(REPO / "site"), "--destination", str(out), "--quiet"], check=True)
    return out


@pytest.mark.parametrize("page", PAGES)
def test_header_on_every_page(built, page):
    html = (built / page / "index.html").read_text()
    assert len(re.findall(r'<header class="?sq-siteheader"?>', html)) == 1
    assert re.search(r'class="?sq-tab sq-tab--sidechain"? href="?/sidechain/"? aria-current=', html), "Sidechain tab"
    assert bool(re.search(r'<nav class="?sq-subheader"?', html)) == (page != ""), "path strip"

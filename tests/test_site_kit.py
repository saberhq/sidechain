"""The site kit under site/ matches saberhq.com's main.

    .venv/bin/python -m pytest -q tests/test_site_kit.py

Reads site/ and saberhq.com's main over HTTPS; skips when it cannot be reached. Writes nothing.
Notes: private/code-notes/tests/test_site_kit.py.md
"""
import importlib.util
import urllib.error
from importlib.machinery import SourceFileLoader
from pathlib import Path

import pytest

_path = str(Path(__file__).resolve().parent.parent / "scripts" / "kit-pull")
_spec = importlib.util.spec_from_loader("kit_pull", SourceFileLoader("kit_pull", _path))
kit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(kit)


def test_kit_matches_main():
    try:
        changed = kit.drift(timeout=10)
    except (urllib.error.URLError, TimeoutError) as e:
        pytest.skip(f"saberhq.com's main unreachable: {e}")
    assert not changed, f"differs from saberhq.com's main: {', '.join(changed)}; run scripts/kit-pull"


def test_no_stray_files_in_kit_folder():
    assert not kit.strays(), f"not kit files: {kit.strays()}; the kit folder holds copies only"

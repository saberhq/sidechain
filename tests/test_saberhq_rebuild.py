"""saberhq_rebuild reads a feed's guids and finds the ones a page lacks.

    .venv/bin/python -m pytest -q tests/test_saberhq_rebuild.py

Reads nothing outside the test. Writes nothing.
Notes: private/code-notes/tests/test_saberhq_rebuild.py.md
"""
import importlib.util
import sys
from pathlib import Path

_SPEC = importlib.util.spec_from_file_location(
    "saberhq_rebuild", Path(__file__).resolve().parent.parent / "scripts" / "saberhq_rebuild.py"
)
sr = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(sr)

FEED = """<rss><channel>
<item><title>One</title><guid>https://example.org/posts/one/</guid></item>
<item><title>Two</title><guid isPermaLink="true"> https://example.org/posts/two/ </guid></item>
</channel></rss>"""


def test_guids_in_order():
    assert sr.guids(FEED) == ["https://example.org/posts/one/", "https://example.org/posts/two/"]


def test_missing_from_page():
    page = '<a class=sq-card href="https://example.org/posts/one/">One</a>'
    assert sr.missing(sr.guids(FEED), page) == ["https://example.org/posts/two/"]
    assert sr.missing(sr.guids(FEED), page + '<a href="https://example.org/posts/two/">') == []


def test_no_token_never_fails(monkeypatch, capsys):
    monkeypatch.delenv("GH_TOKEN", raising=False)
    monkeypatch.setattr(sr, "fetch", lambda url, timeout=30: FEED if url == sr.FEED else "<html></html>")
    monkeypatch.setattr(sys, "argv", ["saberhq_rebuild.py"])
    assert sr.main() == 0
    assert "not started" in capsys.readouterr().out


def test_nothing_new_starts_nothing(monkeypatch, capsys):
    monkeypatch.setattr(sr, "fetch", lambda url, timeout=30: FEED)
    monkeypatch.setattr(sys, "argv", ["saberhq_rebuild.py"])
    assert sr.main() == 0
    assert "no new post, nothing to start" in capsys.readouterr().out

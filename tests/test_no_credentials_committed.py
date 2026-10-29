"""No Lamin API key reaches a tracked file, and the key-shipping scripts never print or pass it.

    .venv/bin/python -m pytest -q tests/test_no_credentials_committed.py

Reads tracked files; writes nothing. Notes: private/code-notes/tests/test_no_credentials_committed.py.md
"""
from __future__ import annotations

import re
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
LAMIN_USER_ENV = Path.home() / ".lamin" / "current_user.env"
SHIPPERS = (REPO / "scripts" / "brev_lamin_key.sh", REPO / "scripts" / "brev_bootstrap.sh")


def _tracked_files(repo: Path) -> list[Path]:
    out = subprocess.run(["git", "-C", str(repo), "ls-files", "-z"],
                         capture_output=True, text=True, check=True).stdout
    return [repo / name for name in out.split("\0") if name]


def _local_api_key() -> str | None:
    if not LAMIN_USER_ENV.exists():
        return None
    for line in LAMIN_USER_ENV.read_text().splitlines():
        if line.startswith("lamin_user_api_key="):
            value = line.split("=", 1)[1].strip()
            return value if value and value != "null" else None
    return None


@pytest.mark.parametrize("repo", ["public", "private"])
def test_the_lamin_api_key_appears_in_no_tracked_file(repo):
    key = _local_api_key()
    if key is None:
        pytest.skip("no local Lamin API key to search for (CI, or a machine without one)")
    root = REPO if repo == "public" else REPO / "private"
    if not (root / ".git").exists():
        pytest.skip(f"{repo} repo not checked out here")

    hits = []
    for path in _tracked_files(root):
        try:
            if key in path.read_text(errors="ignore"):
                hits.append(str(path.relative_to(root)))
        except (OSError, UnicodeDecodeError):
            continue
    assert not hits, f"Lamin API key found in tracked file(s): {hits}"


def test_the_shipping_scripts_read_the_key_and_never_write_it_down():
    """No shipping script assigns a literal key to `LAMIN_API_KEY`."""
    literal = re.compile(r"""LAMIN_API_KEY\s*=\s*(?!["']?\$|["']?\s*$)["']?[A-Za-z0-9_\-]{8,}""")
    for script in SHIPPERS:
        text = script.read_text()
        assert not literal.search(text), f"{script.name} looks like it hardcodes a key"


def test_the_key_is_never_echoed_or_traced():
    """No shipping script enables `set -x` or prints the key anywhere but the `$TMP` file."""
    for script in SHIPPERS:
        for line in script.read_text().splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):  # the scripts *warn* about `set -x` in prose
                continue
            assert not re.match(r"set\s+-[a-z]*x", stripped), f"{script.name} enables tracing"
            if not re.match(r"(echo|printf)\b.*\$(KEY|LAMIN_API_KEY)\b", stripped):
                continue
            assert re.search(r'>\s*"?\$TMP"?\s*$', stripped), \
                f"{script.name} prints the key to a terminal: {stripped!r}"


def test_the_key_never_becomes_a_command_line_argument():
    """`brev exec` and `brev create` never carry the key."""
    text = (REPO / "scripts" / "brev_lamin_key.sh").read_text()
    for line in text.splitlines():
        if line.strip().startswith("#"):
            continue
        assert not re.search(r"brev\s+(exec|create).*\$KEY", line), \
            f"key passed through argv: {line.strip()!r}"


def test_the_box_key_file_is_gitignored_by_name():
    """`.lamin_env` is gitignored."""
    ignored = subprocess.run(["git", "-C", str(REPO), "check-ignore", "-q", ".lamin_env"],
                             check=False)
    assert ignored.returncode == 0, "add `.lamin_env` to .gitignore"

"""The data plane's one rule: an artifact's key IS its path under the data root.

ADR 0007 §1. `lamin_register.py` goes path → key, `lamin_pull.py` goes key → path, and
`log_run` keys its own outputs the same way. If those three ever disagree, a pull lands
a file where nothing looks for it, or two unrelated runs collide on one key — which is
not hypothetical: on 2026-08-28 five scored runs all wrote `runs/summary.json` and the
instance folded them into five versions of one artifact.

The tests point `SIDECHAIN_DATA_ROOT` at a tmp_path throughout. None of them touch the
hub or the real tree.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from sidechain.utils import lamin


@pytest.fixture()
def root(tmp_path, monkeypatch):
    d = tmp_path / "sidechain"
    d.mkdir()
    monkeypatch.setenv("SIDECHAIN_DATA_ROOT", str(d))
    return d.resolve()


# --- path <-> key ---------------------------------------------------------------


def test_the_key_is_the_path_under_the_data_root(root):
    p = root / "derived" / "xatlas-orion" / "hct116_full.npz"
    assert lamin.artifact_key(p) == "derived/xatlas-orion/hct116_full.npz"


def test_key_and_local_path_are_inverses(root):
    for key in ("cache/vcc2026/h1_pseudobulk.npz", "runs/mirror/loco_k562gwps/bundle"):
        assert lamin.artifact_key(lamin.local_path(key)) == key


def test_a_path_outside_the_tree_has_no_key(root, tmp_path):
    """Refusing beats inventing one: a `runs/<basename>` fallback is exactly how five
    runs' summary.json became five versions of one artifact."""
    with pytest.raises(ValueError, match="outside"):
        lamin.artifact_key(tmp_path / "elsewhere" / "summary.json")


def test_the_data_root_is_read_per_call_not_bound_at_import(root, tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.setenv("SIDECHAIN_DATA_ROOT", str(other))
    assert lamin.artifact_key(other / "a.npz") == "a.npz"


def test_a_key_from_the_hub_can_never_write_outside_the_tree(root):
    """Keys come back over the network, so `..` is hostile input, not a typo."""
    for hostile in ("../escape.npz", "../../etc/passwd", "derived/../../escape.npz"):
        with pytest.raises(ValueError, match="outside"):
            lamin.local_path(hostile)


def test_absolute_and_empty_keys_are_refused(root):
    for bad in ("/etc/passwd", "", "/"):
        with pytest.raises(ValueError):
            lamin.local_path(bad)


def test_a_dotted_key_that_stays_inside_is_fine(root):
    assert lamin.local_path("derived/../cache/x.npz") == root / "cache" / "x.npz"


# --- instance selection ---------------------------------------------------------


def test_instance_defaults_and_the_empty_string_means_do_not_connect(monkeypatch):
    monkeypatch.delenv("SIDECHAIN_LAMIN_INSTANCE", raising=False)
    assert lamin.instance() == "saberhq/sidechain"
    monkeypatch.setenv("SIDECHAIN_LAMIN_INSTANCE", "someone/else")
    assert lamin.instance() == "someone/else"
    monkeypatch.setenv("SIDECHAIN_LAMIN_INSTANCE", "")
    assert lamin.instance() == ""


# --- prepare_dest ----------------------------------------------------------------


def test_prepare_dest_makes_the_parent_and_returns_the_path(root):
    dest = root / "derived" / "x" / "cached.npz"
    assert lamin.prepare_dest(dest) == dest
    assert dest.parent.is_dir()


def test_prepare_dest_clears_a_stale_folder_so_a_re_pull_is_not_a_merge(root):
    """`download_to` is a recursive MERGE. A bundle re-pulled into a directory still
    holding a member the new version dropped is neither version — and `hash_dir` would
    then fail the check for a reason that looks like a corrupt download."""
    dest = root / "runs" / "mirror" / "loco" / "bundle"
    dest.mkdir(parents=True)
    (dest / "stale.json").write_text("stale")

    lamin.prepare_dest(dest)
    assert not dest.exists()


def test_prepare_dest_clears_a_stale_file(root):
    dest = root / "a.npz"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_bytes(b"old")

    lamin.prepare_dest(dest)
    assert not dest.exists()


# --- hashing --------------------------------------------------------------------


def test_hash_local_agrees_with_lamindbs_own_hasher(root):
    """Verifying a pull against a digest lamindb does not use would be a check that
    always passes."""
    from lamindb_setup.core.hashing import hash_dir, hash_file

    f = root / "a.npz"
    f.write_bytes(b"x" * 1000)
    assert lamin.hash_local(f) == hash_file(f)[1]

    d = root / "bundle"
    d.mkdir()
    (d / "one").write_text("1")
    (d / "two").write_text("2")
    assert lamin.hash_local(d) == hash_dir(d)[1]


def test_the_hash_changes_when_the_bytes_change(root):
    f = root / "a.npz"
    f.write_bytes(b"before")
    before = lamin.hash_local(f)
    f.write_bytes(b"after!")
    assert lamin.hash_local(f) != before


# --- the register/pull scripts agree with the module ----------------------------


def _load(name: str):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "scripts" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_script_{name}", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_register_derives_the_same_keys_the_module_does(root, capsys):
    """`--dry-run` exists so this can be asserted without an upload."""
    (root / "derived").mkdir()
    f = root / "derived" / "a.npz"
    f.write_bytes(b"x")

    assert _load("lamin_register").main([str(f), "--dry-run"]) == 0
    assert "would register derived/a.npz" in capsys.readouterr().out


def test_register_refuses_a_path_outside_the_tree_and_says_what_to_pass(root, tmp_path, capsys):
    outside = tmp_path / "stray.npz"
    outside.write_bytes(b"x")
    with pytest.raises(SystemExit, match="key-prefix"):
        _load("lamin_register").main([str(outside), "--dry-run"])


def test_pull_and_register_are_the_same_mapping(root):
    """The two directions share `utils.lamin`, so this is a wiring test: it fails if
    either script grows its own copy of the rule."""
    reg, pull = _load("lamin_register"), _load("lamin_pull")
    assert reg.artifact_key is lamin.artifact_key
    assert pull.local_path is lamin.local_path
    assert pull.hash_local is lamin.hash_local
    assert pull.prepare_dest is lamin.prepare_dest


def test_register_reads_a_path_list_from_a_file(root, tmp_path, capsys):
    """The backlog is dozens of files; `--from-file` is how they go in one Run."""
    (root / "cache").mkdir()
    listing = tmp_path / "paths.txt"
    names = []
    for i in range(3):
        f = root / "cache" / f"a{i}.npz"
        f.write_bytes(b"x")
        names.append(str(f))
    listing.write_text("# the backlog\n\n" + "\n".join(names) + "\n")

    assert _load("lamin_register").main(["--from-file", str(listing), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert all(f"would register cache/a{i}.npz" in out for i in range(3))


def test_register_refuses_an_argv_longer_than_lamindbs_cli_args_column(root, tmp_path):
    """Found on 2026-08-28: `ln.track()` writes " ".join(sys.argv[1:]) into
    Run.cli_args, a varchar(1024), with no truncation — thirteen absolute paths died
    inside Django with `DataError: value too long for type character varying(1024)`,
    naming neither the field nor the fix. Fail early, in a sentence."""
    reg = _load("lamin_register")
    (root / "cache").mkdir()
    f = root / "cache" / "a.npz"
    f.write_bytes(b"x")
    long_argv = [str(f)] * 40  # ~40 x len(path) chars, well past 1024

    with pytest.raises(SystemExit, match="from-file"):
        reg.main([*long_argv, "--dry-run"])
    assert reg.CLI_ARGS_MAX == 1024


def test_register_with_no_paths_at_all_is_a_usage_error(root):
    with pytest.raises(SystemExit):
        _load("lamin_register").main(["--dry-run"])


def test_register_refuses_to_overwrite_a_folder_artifacts_own_history(root, monkeypatch):
    """Folder artifacts share ONE storage key across versions
    (`uid[:16]`, lamindb/core/storage/paths.py:39-43), so re-registering a changed
    directory replaces the stored bytes of the previous version and makes it
    permanently unreadable. For the mirror bundles — the 32 KB this ADR exists to
    protect — that would destroy the backup while printing "registered"."""
    import sys
    import types

    reg = _load("lamin_register")
    bundle = root / "runs" / "mirror" / "loco" / "bundle"
    bundle.mkdir(parents=True)
    (bundle / "manifest.json").write_text("{}")

    class _Prior:
        hash = "a-different-hash"

    class _QS:
        def one_or_none(self):
            return _Prior()

    saved = []

    class _Artifact:
        def __init__(self, path, key=None, description=None):
            self.path, self.key, self.uid, self.size, self.hash = path, key, "u", 1, "h"

        @staticmethod
        def filter(**_kw):
            return _QS()

        def save(self):
            saved.append(self.key)
            return self

    mod = types.ModuleType("lamindb")
    mod.connect = lambda _s: None
    mod.track = lambda: None
    mod.finish = lambda: None
    mod.Artifact = _Artifact
    monkeypatch.setitem(sys.modules, "lamindb", mod)

    with pytest.raises(SystemExit, match="allow-folder-overwrite"):
        reg.main([str(bundle)])
    assert saved == []

    assert reg.main([str(bundle), "--allow-folder-overwrite"]) == 0
    assert saved == ["runs/mirror/loco/bundle"]


def test_register_lets_an_unchanged_folder_through(root, monkeypatch):
    """Re-running the backlog must be idempotent, not a wall — the guard fires on
    *different* contents, not on the key already existing."""
    import sys
    import types

    reg = _load("lamin_register")
    bundle = root / "bundle"
    bundle.mkdir()
    (bundle / "manifest.json").write_text("{}")
    same = lamin.hash_local(bundle)

    class _Prior:
        hash = same

    saved = []

    class _Artifact:
        def __init__(self, path, key=None, description=None):
            self.key, self.uid, self.size, self.hash = key, "u", 1, "h"

        @staticmethod
        def filter(**_kw):
            return type("QS", (), {"one_or_none": lambda self: _Prior()})()

        def save(self):
            saved.append(self.key)
            return self

    mod = types.ModuleType("lamindb")
    mod.connect = lambda _s: None
    mod.track = lambda: None
    mod.finish = lambda: None
    mod.Artifact = _Artifact
    monkeypatch.setitem(sys.modules, "lamindb", mod)

    assert reg.main([str(bundle)]) == 0
    assert saved == ["bundle"]

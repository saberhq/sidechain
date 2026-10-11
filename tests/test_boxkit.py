"""The box kit's three small tools: the profile report, the scoring wrapper's caps, the machine record."""

import importlib.util
import os
import stat
import subprocess
from pathlib import Path

KIT = Path(__file__).resolve().parents[1] / "scripts" / "boxkit"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, KIT / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


LOG = """=== 2026-10-11T05:00:00Z load 1.00 1.00 1.00 memavail_GiB 200
    PID    PPID   RSS %CPU ELAPSED COMMAND
  101       1 3145728 100 60 .venv/bin/python -m sidechain.eval.loco --real a.h5ad
  102     101 6291456 400 30 /x/.venv/bin/python /x/.venv/bin/cell-eval2 run -ap p.h5ad -ar r.h5ad
  103       1 1048576 100 60 python -m sidechain.eval.loco --delta-cache-build --delta-cache-jobs 4
  104       1 2097152 0.5 900 /usr/bin/something-else
card 0, 1000 MiB, 49140 MiB, 50 %, 10 %
card 1, 1 MiB, 49140 MiB, 0 %, 0 %
=== 2026-10-11T05:01:00Z load 3.00 2.00 1.00 memavail_GiB 150
    PID    PPID   RSS %CPU ELAPSED COMMAND
  101       1 3145728 100 120 .venv/bin/python -m sidechain.eval.loco --real a.h5ad
  102     101 8388608 400 90 /x/.venv/bin/python /x/.venv/bin/cell-eval2 run -ap p.h5ad -ar r.h5ad
card 0, 20000 MiB, 49140 MiB, 80 %, 10 %
card 1, 1 MiB, 49140 MiB, 0 %, 0 %
"""


def test_profile_report_kinds_and_gpu():
    rep = _load("box_profile_report")
    s = rep.summarise(rep.parse(LOG.splitlines()))
    assert (
        s["minutes"] == 2
        and s["load"]["max"] == 3.0
        and s["memavail_gib"]["min"] == 150
    )
    assert (
        s["kinds"]["score"]["n"] == 1
        and s["kinds"]["arm"]["n"] == 1
        and s["kinds"]["cache"]["n"] == 1
    )
    assert "something-else" not in str(s["kinds"])
    sc = s["kinds"]["score"]
    assert sc["wall_s_max"] == 90 and abs(sc["rss_gib_max"] - 8.0) < 1e-9
    assert abs(sc["cores_median"] - 4.0) < 1e-9  # 400 % over 90 s is 4 cores
    assert s["gpu"]["minutes_any_use"] == 2 and s["gpu"]["peak_card_mib"] == 20000
    assert "score: 1 processes" in rep.render(s)


def test_lscpu_parse_and_intel_flag():
    m = _load("box_machine")
    amd = "CPU(s): 124\nVendor ID: AuthenticAMD\nModel name: AMD EPYC 7763 64-Core Processor\nThread(s) per core: 1\nCore(s) per socket: 62\nSocket(s): 2\n"
    r = m.parse_lscpu(amd)
    assert (
        r["cores"] == 124
        and r["cpus"] == 124
        and not r["intel"]
        and r["threads_per_core"] == 1
    )
    intel = "CPU(s): 24\nVendor ID: GenuineIntel\nModel name: Intel Xeon\nThread(s) per core: 2\nCore(s) per socket: 6\nSocket(s): 2\n"
    r = m.parse_lscpu(intel)
    assert r["cores"] == 12 and r["intel"] and r["threads_per_core"] == 2


def test_scorer_caps_env_and_args(tmp_path):
    stub = tmp_path / "ce"
    stub.write_text(
        '#!/bin/bash\necho "ARGS $*"\necho "OMP=$OMP_NUM_THREADS POLARS=$POLARS_MAX_THREADS NUMBA=$NUMBA_NUM_THREADS RAYON=$RAYON_NUM_THREADS NUMEXPR=$NUMEXPR_NUM_THREADS MKL=$MKL_NUM_THREADS OPENBLAS=$OPENBLAS_NUM_THREADS"\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    wrap = KIT / "box_scorer_caps.sh"
    env = dict(os.environ, BOX_SCORER_THREADS="3")
    out = subprocess.run(
        ["bash", str(wrap), str(stub), "run", "-ap", "p"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    ).stdout
    assert "ARGS run -ap p --set gather_threads=3" in out
    for name in ("OMP", "POLARS", "NUMBA", "RAYON", "NUMEXPR", "MKL", "OPENBLAS"):
        assert f"{name}=3" in out, out
    out = subprocess.run(
        ["bash", str(wrap), str(stub), "score", "--x"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    ).stdout
    assert "ARGS score --x\n" in out and "gather_threads" not in out

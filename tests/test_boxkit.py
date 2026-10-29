"""The box kit: profile report, scorer caps and gate, queue order, done check, CPU env.

    .venv/bin/python -m pytest -q tests/test_boxkit.py

Writes temporary files only. Notes: private/code-notes/tests/test_boxkit.py.md
"""

import importlib.util
import os
import stat
import subprocess
import time
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


# --------------------------------------------------------------------------- the scoring gate


def _stub(tmp_path, seconds):
    """A fake cell-eval2: logs its start and end to a shared file, sleeps, exits 0."""
    stub = tmp_path / "ce"
    log = tmp_path / "events.log"
    stub.write_text(
        f'#!/bin/bash\necho "start $$ card=$CUDA_VISIBLE_DEVICES $1" >> {log}\nsleep {seconds}\necho "end $$" >> {log}\n'
    )
    stub.chmod(stub.stat().st_mode | stat.S_IEXEC)
    return stub, log


def _meminfo(path, gib):
    """A fake /proc/meminfo with `gib` GiB total and available, swapped in whole so the gate never reads half a file."""
    kb = gib * 1048576
    tmp = path.with_suffix(".tmp")
    tmp.write_text(f"MemTotal:       {kb} kB\nMemAvailable:   {kb} kB\n")
    os.replace(tmp, path)
    return path


def _gate(tmp_path, stub, args, **env):
    # a fake meminfo, so the gate never reads this machine's free memory
    mem = tmp_path / "meminfo"
    if not mem.exists():
        _meminfo(mem, 256)
    e = {
        **os.environ,
        "BOX_STATE": str(tmp_path / "state"),
        "BOX_LIVENESS": "none",
        "BOX_POLL_S": "1",
        "BOX_BUDGET_GIB": "100",
        "BOX_MEMINFO": str(mem),
        **env,
    }
    return subprocess.Popen(
        ["bash", str(KIT / "box_scorer_gate.sh"), str(stub), *args],
        env=e,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )


def _max_concurrent(log):
    cur = best = 0
    for ln in log.read_text().splitlines():
        cur += 1 if ln.startswith("start") else -1
        best = max(best, cur)
    return best


def test_gate_passes_other_subcommands_straight_through(tmp_path):
    stub, log = _stub(tmp_path, 0)
    p = _gate(tmp_path, stub, ["score", "--x"], BOX_SCORER_GIB="10")
    assert p.wait(timeout=20) == 0
    assert (
        "start" in log.read_text()
        and not (tmp_path / "state" / "scorer_gate.log").exists()
    )


def test_gate_admits_one_step_per_card(tmp_path):
    stub, log = _stub(tmp_path, 1)
    ps = [
        _gate(tmp_path, stub, ["run", "-ap", "p"], BOX_SCORER_GIB="10", BOX_CARDS="2")
        for _ in range(4)
    ]
    assert all(p.wait(timeout=60) == 0 for p in ps)
    assert log.read_text().count("start") == 4 and _max_concurrent(log) == 2
    assert not list((tmp_path / "state" / "gpu").glob("card*")) and not list(
        (tmp_path / "state" / "resv").glob("s.*")
    )


def test_gate_admits_what_the_memory_budget_allows(tmp_path):
    stub, log = _stub(tmp_path, 1)
    ps = [
        _gate(
            tmp_path,
            stub,
            ["run"],
            BOX_SCORER_GIB="40",
            BOX_CARDS="4",
            BOX_BUDGET_GIB="100",
        )
        for _ in range(4)
    ]
    assert all(p.wait(timeout=60) == 0 for p in ps)
    assert _max_concurrent(log) == 2  # 40 + 40 fits 100, a third does not


def test_gate_holds_a_step_until_the_box_really_has_the_memory(tmp_path):
    stub, log = _stub(tmp_path, 0)
    mem = _meminfo(tmp_path / "meminfo", 12)  # under 10 + the 6 GiB margin
    p = _gate(tmp_path, stub, ["run"], BOX_SCORER_GIB="10", BOX_CARDS="1")
    time.sleep(3)
    assert p.poll() is None and not log.exists()
    _meminfo(mem, 64)
    assert p.wait(timeout=20) == 0 and "start" in log.read_text()
    assert "memfree 12G" in (tmp_path / "state" / "scorer_gate.log").read_text()


def test_gate_caps_the_pools_and_picks_a_card(tmp_path):
    stub, log = _stub(tmp_path, 0)
    p = _gate(
        tmp_path,
        stub,
        ["run"],
        BOX_SCORER_GIB="10",
        BOX_CARDS="1",
        BOX_SCORER_THREADS="4",
    )
    assert p.wait(timeout=20) == 0
    assert "card=0" in log.read_text()


def test_gate_does_not_end_a_waiting_step_on_stop(tmp_path):
    stub, _log = _stub(tmp_path, 0)
    (tmp_path / "state").mkdir()
    (tmp_path / "state" / "STOP").write_text("")
    p = _gate(tmp_path, stub, ["run"], BOX_SCORER_GIB="10", BOX_CARDS="1")
    assert p.wait(timeout=20) == 0
    assert "STOP seen" in (tmp_path / "state" / "scorer_gate.log").read_text()


def test_gate_times_out_a_hung_step_and_gives_everything_back(tmp_path):
    stub, _log = _stub(tmp_path, 30)
    t0 = time.time()
    p = _gate(
        tmp_path,
        stub,
        ["run"],
        BOX_SCORER_GIB="10",
        BOX_CARDS="1",
        BOX_SCORER_TIMEOUT_S="2",
    )
    assert p.wait(timeout=30) == 124 and time.time() - t0 < 20
    assert "SCORER_TIMEOUT" in (tmp_path / "state" / "scorer_gate.log").read_text()
    assert not list((tmp_path / "state" / "gpu").glob("card*")) and not list(
        (tmp_path / "state" / "resv").glob("s.*")
    )


# --------------------------------------------------------------------------- longest-first ordering
def _makespan(durations, machines):
    free = [0.0] * machines
    for d in durations:
        i = free.index(min(free))
        free[i] += d
    return max(free)


def test_longest_first_keeps_layers_and_shortens_the_tail(tmp_path):
    order = _load("box_order")
    lines = [
        "short|a|x|list",
        "long|b|x|list",
        "mid|c|x|list",
        "mid|d|x|tmpl",
        "long|e|x|tmpl",
        "unknown|f|x|tmpl",
    ]
    est = {"short": 5.0, "mid": 10.0, "long": 20.0}
    out = order.longest_first(
        lines,
        lambda ln: est.get(ln.split("|")[0]),
        lambda ln: 0 if ln.endswith("tmpl") else 1,
    )
    assert out == [
        "long|e|x|tmpl",
        "mid|d|x|tmpl",
        "unknown|f|x|tmpl",
        "long|b|x|list",
        "mid|c|x|list",
        "short|a|x|list",
    ]
    durations = [2, 2, 2, 2, 2, 2, 12]  # one long arm among short ones: the tail
    assert (
        _makespan(sorted(durations, reverse=True), 2) == 12
        and _makespan(sorted(durations), 2) == 18
    )


def test_box_order_cli_reads_a_queue_file(tmp_path):
    q = tmp_path / "q.txt"
    q.write_text(
        "loco_k562gwps_union|n1|ref|design:0/1\nloco_hek293t|n2|ref|design:0/4\nloco_hek293t|n3|ref|tmpl\n"
    )
    r = subprocess.run(
        [
            "python3",
            str(KIT / "box_order.py"),
            str(q),
            "--est",
            "loco_hek293t=59,loco_k562gwps_union=21",
            "--layers",
            "tmpl,design",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert r.stdout.splitlines() == [
        "loco_hek293t|n3|ref|tmpl",
        "loco_hek293t|n2|ref|design:0/4",
        "loco_k562gwps_union|n1|ref|design:0/1",
    ]


# --------------------------------------------------------------------------- the done check
def test_done_verdict_only_ends_a_box_that_is_safe_and_registered():
    done = _load("box_done")
    ok, lines = done.verdict(
        {"safe_to_delete": True, "not_in_lamin": []}, "b1", fetched=True
    )
    assert (
        ok
        and lines[0] == "scripts/brev_key.sh delete b1"
        and "ledger.py box stop b1" in lines[1]
    )
    ok, lines = done.verdict({"safe_to_stop": True}, "b2", stoppable=True)
    assert (
        ok
        and any(ln.startswith("brev stop b2") for ln in lines)
        and "fetch" in lines[0]
    )
    ok, lines = done.verdict({"safe_to_delete": False}, "b3")
    assert not ok and "NOT READY" in lines[0]
    ok, lines = done.verdict({"safe_to_delete": True, "not_in_lamin": ["a", "b"]}, "b4")
    assert not ok and "2 file(s)" in lines[0]


def test_done_cli_reads_the_marker_over_ssh(tmp_path):
    fake = tmp_path / "ssh"
    fake.write_text(
        '#!/bin/bash\necho \'{"safe_to_stop": true, "not_in_lamin": []}\'\n'
    )
    fake.chmod(fake.stat().st_mode | stat.S_IEXEC)
    env = {**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"}
    r = subprocess.run(
        ["python3", str(KIT / "box_done.py"), "bx", "--marker", "m.json", "--fetched"],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )
    assert (
        r.returncode == 0
        and r.stdout.splitlines()[0] == "READY"
        and "brev_key.sh delete bx" in r.stdout
    )


def _cpu_env(tmp_path, cpuinfo):
    info = tmp_path / "cpuinfo"
    info.write_text(cpuinfo)
    script = f'. "{KIT / "box_cpu_env.sh"}"; echo "$BOX_CPU_ENV|${{OPENBLAS_CORETYPE:-}}|${{NPY_DISABLE_CPU_FEATURES:-}}"'
    env = {k: v for k, v in os.environ.items() if k not in ("OPENBLAS_CORETYPE", "NPY_DISABLE_CPU_FEATURES")}
    r = subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True,
                       env={**env, "BOX_CPUINFO": str(info)})
    return r.stdout.strip().split("|")


def test_cpu_env_pins_only_an_amd_box_with_avx512(tmp_path):
    zen4 = "vendor_id\t: AuthenticAMD\nmodel name\t: AMD EPYC 9554\nflags\t\t: fpu sse avx2 avx512f avx512bw\n"
    zen3 = "vendor_id\t: AuthenticAMD\nmodel name\t: AMD EPYC 7763\nflags\t\t: fpu sse avx2\n"
    intel = "vendor_id\t: GenuineIntel\nmodel name\t: Intel Xeon Gold 6338\nflags\t\t: fpu sse avx2 avx512f\n"
    assert _cpu_env(tmp_path, zen4) == ["pinned to AVX2 (AMD with AVX-512)", "ZEN", "X86_V4 AVX512_ICL AVX512_SPR"]
    assert _cpu_env(tmp_path, zen3) == ["unchanged (no AVX-512 AMD CPU)", "", ""]
    assert _cpu_env(tmp_path, intel) == ["unchanged (no AVX-512 AMD CPU)", "", ""]

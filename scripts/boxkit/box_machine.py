#!/usr/bin/env python3
"""Write what a box really is: CPU model, vendor, sockets, cores, threads, memory, disks, cards. Brev's catalogue has no CPU field and
its vCPU count is not always cores, so every box's first command is this one.

    python3 box_machine.py [out.json]        # prints the JSON and writes it when a path is given
"""

import contextlib
import json
import os
import platform
import re
import shutil
import subprocess
import sys


def parse_lscpu(text):
    kv = {}
    for ln in text.splitlines():
        if ":" in ln:
            k, v = ln.split(":", 1)
            kv[k.strip()] = v.strip()
    sockets = int(kv.get("Socket(s)", 1) or 1)
    cps = int(kv.get("Core(s) per socket", 0) or 0)
    tpc = int(kv.get("Thread(s) per core", 1) or 1)
    return {
        "model": kv.get("Model name", ""),
        "vendor": kv.get("Vendor ID", ""),
        "cpus": int(kv.get("CPU(s)", 0) or 0),
        "sockets": sockets,
        "cores": sockets * cps,
        "threads_per_core": tpc,
        "intel": kv.get("Vendor ID", "") == "GenuineIntel",
    }


def run(cmd):
    try:
        return subprocess.run(
            cmd, capture_output=True, text=True, timeout=20, check=False
        ).stdout
    except (OSError, subprocess.TimeoutExpired):
        return ""


def main(argv):
    mem_kb = 0
    with (
        contextlib.suppress(OSError, AttributeError, ValueError),
        open("/proc/meminfo") as fh,
    ):
        mem_kb = int(re.search(r"MemTotal:\s+(\d+)", fh.read()).group(1))
    gpus = []
    for ln in run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"]
    ).splitlines():
        if "," in ln:
            n, m = ln.rsplit(",", 1)
            gpus.append({"name": n.strip(), "mib": int(m)})
    disks = {}
    for p in ("/", "/ephemeral", os.path.expanduser("~")):
        if os.path.exists(p):
            u = shutil.disk_usage(p)
            disks[p] = {
                "total_gb": round(u.total / 1e9),
                "free_gb": round(u.free / 1e9),
            }
    info = {
        "hostname": platform.node(),
        "cpu": parse_lscpu(run(["lscpu"])),
        "mem_gib": round(mem_kb / 1048576, 1),
        "nproc_all": os.cpu_count(),
        "gpus": gpus,
        "disks": disks,
        "user": os.environ.get("USER", ""),
    }
    text = json.dumps(info, indent=1)
    print(text)
    if len(argv) > 1:
        with open(argv[1], "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))

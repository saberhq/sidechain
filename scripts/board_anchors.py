#!/usr/bin/env python
"""Recover the VCC 2026 leaderboard's own min-max anchors, and read board scores in RAW units.

Every board member is reported twice in a leaderboard snapshot: the raw statistic
(``pdsCosine``, ``exprMseUnbiasedCappedNorm``, …) and its scaled score (``scorePds``,
``scoreMse``, …). The scaling is affine per member -- ``scaled = (raw - b) / (r - b)`` with a
per-panel baseline anchor ``b`` and replicate anchor ``r`` -- so both anchors are recoverable
by regressing scaled on raw across entries.

Why bother: `RESULTS.md` established that a scaled score is model quality divided by that
fold's headroom, and is therefore unreadable without its ``(r - b)`` gap. That rule applies to
the leaderboard too. With the anchors in hand, a rival's scaled 0.705 becomes a raw
``pds_cosine`` of 0.8186, which is a number our own mirror folds can be compared against.

Two details that matter for the fit:

* **Clamped entries carry no information.** A member is clipped at 0.000, and on ``mse`` most
  of the board sits there. Those rows are dropped -- they constrain nothing and they drag the
  line.
* **The fit is trimmed.** A handful of entries were scored under a different panel or anchor
  revision; iteratively dropping residuals beyond 3x the median removes them. The reported
  max residual is the honesty check: on a good member it lands near 1e-3.

Usage::

    python scripts/board_anchors.py --snapshots ~/data/sidechain/vcc2026/leaderboards
    python scripts/board_anchors.py --snapshots <dir> --raw pds=0.7124 --raw mse=6.5236
    python scripts/board_anchors.py --snapshots <dir> --scaled pds=0.470

Re-run it after 2026-10-22: the D/E/F bundle revises the anchor set, and every number this
prints is conditional on ``vcc2026-valA-r4+vcc2026-valB-r4+vcc2026-valC-r4``.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

import numpy as np

# member -> (raw field, scaled field). Order is the order the board averages them in.
MEMBERS: dict[str, tuple[str, str]] = {
    "pds": ("pdsCosine", "scorePds"),
    "mse": ("exprMseUnbiasedCappedNorm", "scoreMse"),
    "nmae": ("deWilcoxonLfcNmae", "scoreNmae"),
    "fid": ("deWilcoxonDirectionFidelityYieldRaw", "scoreFid"),
    "reach": ("deWilcoxonDirectionReachRaw", "scoreReach"),
    "jac": ("deWilcoxonSigJaccard", "scoreJac"),
}


def load_entries(snapshot_dir: Path) -> dict[str, dict]:
    """Every distinct submission across every snapshot, keyed by its board id."""
    out: dict[str, dict] = {}
    for f in sorted(glob.glob(str(snapshot_dir / "lb_*.json"))):
        doc = json.load(open(f))
        for section in ("live", "final", "generalist"):
            for e in doc.get(section, {}).get("entries", []):
                out[e["id"]] = e
    return out


def fit_anchors(entries: dict[str, dict], raw_key: str, scaled_key: str, trim: int = 6):
    x, y = [], []
    for e in entries.values():
        rv, sv = e.get(raw_key), e.get(scaled_key)
        if rv is None or sv is None:
            continue
        if abs(sv) < 1e-9:  # clamped at the floor: carries no information about the slope
            continue
        x.append(rv)
        y.append(sv)
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 5:
        return None
    m = c = 0.0
    for _ in range(trim):
        A = np.vstack([x, np.ones_like(x)]).T
        m, c = np.linalg.lstsq(A, y, rcond=None)[0]
        res = np.abs(y - (m * x + c))
        keep = res <= max(3.0 * float(np.median(res)), 1e-9)
        if keep.all():
            break
        x, y = x[keep], y[keep]
    if m == 0:
        return None
    b = -c / m          # raw value that scales to 0
    r = b + 1.0 / m     # raw value that scales to 1
    return b, r, int(x.size), float(np.abs(y - (m * x + c)).max())


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshots", type=Path, required=True, help="directory of lb_*.json snapshots")
    ap.add_argument("--raw", action="append", default=[], metavar="MEMBER=VALUE",
                    help="convert a raw value to its scaled score (repeatable)")
    ap.add_argument("--scaled", action="append", default=[], metavar="MEMBER=VALUE",
                    help="convert a scaled score back to raw (repeatable)")
    ap.add_argument("--shared", action="store_true",
                    help="also list raw pds values posted byte-identically by more than one team")
    args = ap.parse_args()

    entries = load_entries(args.snapshots.expanduser())
    anchors: dict[str, tuple[float, float]] = {}

    print(f"{len(entries)} distinct submissions across {len(glob.glob(str(args.snapshots.expanduser() / 'lb_*.json')))} snapshots\n")
    print(f"{'member':7s} {'n':>5s} {'zero-anchor b':>14s} {'one-anchor r':>14s} {'direction':>10s} {'max resid':>10s}")
    for name, (rk, sk) in MEMBERS.items():
        fit = fit_anchors(entries, rk, sk)
        if fit is None:
            print(f"{name:7s} {'--':>5s}  (too few unclamped entries to fit)")
            continue
        b, r, n, resid = fit
        anchors[name] = (b, r)
        direction = "higher" if r > b else "lower"
        print(f"{name:7s} {n:5d} {b:14.5f} {r:14.5f} {direction:>10s} {resid:10.1e}")

    if "mse" in anchors:
        b, _ = anchors["mse"]
        print(f"\nmse clamps to 0.000 for any raw value above {b:.4f}.")

    for spec in args.raw:
        k, _, v = spec.partition("=")
        if k not in anchors:
            print(f"\n{k}: no anchor fitted")
            continue
        b, r = anchors[k]
        s = (float(v) - b) / (r - b)
        print(f"\nraw {k} {float(v):.4f} -> scaled {s:.4f}" + ("  (clamps to 0.000)" if s < 0 else ""))

    for spec in args.scaled:
        k, _, v = spec.partition("=")
        if k not in anchors:
            print(f"\n{k}: no anchor fitted")
            continue
        b, r = anchors[k]
        print(f"\nscaled {k} {float(v):.4f} -> raw {b + float(v) * (r - b):.4f}")

    if args.shared:
        by: dict[float, list[dict]] = {}
        for e in entries.values():
            if e.get("pdsCosine") is not None:
                by.setdefault(round(e["pdsCosine"], 10), []).append(e)
        print("\nraw pds values posted byte-identically by more than one team:")
        for k, v in sorted(by.items(), key=lambda kv: -kv[0]):
            teams = sorted({(x.get("teamName") or "?") for x in v})
            if len(teams) < 2:
                continue
            names = sorted({(x.get("modelName") or "?") for x in v})
            print(f"  {k:.10f}  {len(teams)} teams  names: {names}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python
"""T59 acceptance test: replay recorded mirror arms through the analytic pds path.

    .venv/bin/python scripts/analytic_pds_replay.py                 # every replayable arm
    .venv/bin/python scripts/analytic_pds_replay.py --fold loco_hct116
    .venv/bin/python scripts/analytic_pds_replay.py --out report.json

For each arm it rebuilds the pooled delta from the arm's OWN recorded sources and knobs
by calling `submit.build.pooled_delta` -- never a second implementation -- scores it with
`sidechain.eval.analytic_pds.score_delta`, and compares against the `pds_cosine` mean
recorded in that arm's `run/agg_results.csv`.

**Agreement is the easy half.** The boundary a caller crosses by accident is lambda > 0,
where the emitter draws a Poisson share and the group sum is exact only in expectation.
So this deliberately replays lambda > 0 arms too and reports the SIZE of the break rather
than a pass/fail -- a benchmark that only demonstrates agreement never finds its own edge.
(Asked for by session `271a46a8`, which owns the T59 spec.)

Two honest limits, both reported rather than hidden:
  * arms with `gamma != 1` or `similarity_beta != 0` are SKIPPED -- those knobs need
    inputs (`ctrl_tgt_cpm`, per-source control profiles) this replay does not thread;
  * two X-Atlas artifacts do not open on a 17 GB Mac, so their fold-subset equivalents
    are substituted. Each substitution is *validated by the identity check itself*: if the
    subset were not pooling-identical on that fold, the replay would fail, loudly.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from sidechain.eval.analytic_pds import LEGACY_MIN_LIBSIZE, prep_fold, score_delta
from sidechain.submit.build import apply_transfer_floors, pooled_delta, sources_from_specs

_POOL_MEMO: dict = {}

MIRRORS = Path("~/data/sidechain/runs/mirror").expanduser()
CACHE = Path("~/data/sidechain/cache/vcc2026").expanduser()
LOCAL = Path("~/data/sidechain").expanduser()

# recorded specs carry box paths; and the two full X-Atlas artifacts (5.8 / 6.9 GB) do not
# open here, so the fold-subset build is substituted -- see the module docstring.
SUBSTITUTE = {
    "derived/xatlas-orion/hek293t_full.npz": {
        "loco_hct116": "cache/vcc2026/foldsub/hek293t_full_hct116fold830.npz",
        "loco_hct116_pdex": "cache/vcc2026/foldsub/hek293t_full_hct116fold830.npz",
        "loco_hek293t_ch272": "cache/vcc2026/foldsub/hek293t_full_fold272.npz",
        "loco_k562gwps_union_ch272": "cache/vcc2026/foldsub/hek293t_full_fold272.npz",
    },
    "derived/xatlas-orion/hct116_full.npz": {
        "loco_hek293t_ch272": "cache/vcc2026/foldsub/hct116_full_fold272.npz",
        "loco_k562gwps_union_ch272": "cache/vcc2026/foldsub/hct116_full_fold272.npz",
    },
}
SUFFIXES = ("_pdex", "_rule", "_d3")     # variants score against the parent fold's cells


def real_h5ad(fold: str) -> Path | None:
    names = [fold] + [fold[: -len(x)] for x in SUFFIXES if fold.endswith(x)]
    for n in names:
        p = CACHE / f"{n}_real.h5ad"
        if p.exists():
            return p
    return None


PERT_COL = {"loco_k562gwps_union_ch272": "gene", "loco_k562gwps_union": "gene",
            "loco_k562gwps": "gene", "loco_k562gwps_pdex": "gene",
            "loco_k562_essential": "gene"}


def localise(spec: str, fold: str) -> tuple[str | None, str | None]:
    """(usable spec, substitution note) or (None, reason it cannot be replayed here)."""
    path, _, ctrl = spec.partition(":")
    rel = path.split("/data/sidechain/", 1)[-1]
    if rel in SUBSTITUTE:
        sub = SUBSTITUTE[rel].get(fold)
        if sub is None:
            return None, f"{Path(rel).name} does not open on this Mac and has no subset for {fold}"
        return f"{LOCAL / sub}:{ctrl}", f"{Path(rel).name} -> {Path(sub).name}"
    p = LOCAL / rel
    if not p.exists():
        return None, f"missing locally: {rel}"
    return f"{p}:{ctrl}", None


def _num(build: dict, key: str, default: float) -> float:
    """`build.get(k) or default` is WRONG for a knob whose "off" value is 0.0.

    `gamma: 0.0` is falsy, so `or 1.0` silently rewrites a gamma=0 arm as gamma=1 and
    the replay reports a confident wrong number. The T59 benchmark caught exactly that
    on `loco_k562gwps_pdex/ag_a100_g000` (7.5e-03 off), which is what a benchmark is for.
    """
    v = build.get(key)
    return default if v is None else float(v)


def recorded_pds(arm: Path) -> float | None:
    f = arm / "run" / "agg_results.csv"
    if not f.exists():
        return None
    for row in csv.DictReader(f.open()):
        if row.get("statistic") == "mean" and row.get("pds_cosine"):
            return float(row["pds_cosine"])
    return None


def replay_arm(arm: Path, fold_name: str, fold_cache) -> dict:
    s = json.loads((arm / "summary.json").read_text())
    b = s.get("build", {})
    out = {"fold": fold_name, "arm": arm.name, "recorded": recorded_pds(arm),
           "lam": _num(b, "emit_lambda", 0.0), "alpha": _num(b, "alpha", 1.0)}
    if out["recorded"] is None:
        return out | {"status": "skipped", "why": "no agg_results.csv"}
    if _num(b, "gamma", 1.0) != 1.0:
        return out | {"status": "skipped", "why": "gamma != 1 needs ctrl_tgt_cpm"}
    if _num(b, "similarity_beta", 0.0) != 0.0:
        return out | {"status": "skipped", "why": "similarity_beta != 0 needs control profiles"}
    if b.get("shrinkage") is None:
        # Same falsy-default class as the gamma bug: `bool(None)` is False while
        # `pooled_delta` defaults to shrinkage=True, so an absent field would silently
        # rebuild the arm with the knob OFF. The record does not say, so neither do we.
        # Checked HERE, with the other record-completeness tests, rather than later with
        # the environment ones: an arm we cannot reconstruct because the RECORD is
        # incomplete should say so whether or not its sources happen to be on this machine.
        return out | {"status": "skipped",
                      "why": "shrinkage not recorded (pooled_delta defaults to True)"}
    if _num(b, "min_libsize", LEGACY_MIN_LIBSIZE) != LEGACY_MIN_LIBSIZE:
        # `eval.loco` started recording its control-cell floor on 2026-09-20 and its
        # default moved to 1000 the same day. The fold is prepared once per fold at the
        # legacy 500, so an arm built at another floor would be scored against a control
        # profile that is not its own -- a small, plausible, wrong number.
        return out | {"status": "skipped",
                      "why": f"min_libsize {_num(b, 'min_libsize', LEGACY_MIN_LIBSIZE):g}, "
                             f"this replay prepares folds at {LEGACY_MIN_LIBSIZE:g}"}
    if any(x is not None for x in (b.get("shrink_overrides") or [])):
        # depth-aware shrinkage forces shrink ON for named sources only; the replay
        # threads one global flag, so reconstructing these would be a guess.
        return out | {"status": "skipped", "why": "shrink_overrides (knob d) not threaded"}
    specs, notes, renamed = [], [], {}
    for spec in (s.get("sources") or {}).get("pseudobulk") or []:
        got, note = localise(spec, fold_name)
        if got is None:
            return out | {"status": "skipped", "why": note}
        specs.append(got)
        if note:
            notes.append(note)
            # a substituted artifact has a different stem, and transfer floors are keyed
            # by stem -- `apply_transfer_floors` rightly REFUSES a name that matches no
            # source rather than attaching the floor to the wrong one, so remap here.
            before, _, after = note.partition(" -> ")
            renamed[Path(before).stem] = Path(after).stem
    if not specs:
        return out | {"status": "skipped", "why": "summary.json records no sources"}

    tf = b.get("transfer_floor") or {}
    tf = ({renamed.get(Path(k).stem, Path(k).stem): v for k, v in tf.items()}
          if isinstance(tf, dict) and any(tf.values()) else {})
    tiers = b.get("coverage_tiers") or None
    if tiers:
        tiers = tuple((float(a), float(c)) for a, c in tiers)
    shrink = bool(b["shrinkage"])
    vf = b.get("var_floor") or "none"

    # alpha is a pure scalar applied in score_delta, so arms that differ only in alpha
    # share one pooled delta. Memoised on everything that DOES change the pooling.
    key = (fold_name, tuple(specs), vf, shrink, tiers, tuple(sorted(tf.items())))
    deltas = _POOL_MEMO.get(key)
    if deltas is None:
        sources = sources_from_specs(specs, [])
        if tf:
            sources = apply_transfer_floors(sources, tf)
        targets_all = [str(p) for p in fold_cache.perts if str(p) != "non-targeting"]
        deltas = np.zeros((len(targets_all), len(fold_cache.genes)))
        cov = np.zeros(len(targets_all), dtype=bool)
        for i, t in enumerate(targets_all):
            d = pooled_delta(t, sources, fold_cache.genes, shrinkage=shrink,
                             var_floor=vf, coverage_tiers=tiers)
            if d is not None:
                deltas[i] = d
                cov[i] = True
        deltas = (deltas, cov)
        _POOL_MEMO.clear()          # one fold's deltas at a time; these are ~250 MB each
        _POOL_MEMO[key] = deltas
        out["pooled"] = True
    deltas = _POOL_MEMO[key]
    deltas, cov = deltas
    targets = [str(p) for p in fold_cache.perts if str(p) != "non-targeting"]
    out["covered"] = int(cov.sum())

    # Cross-check the reconstruction against the field the record already carries.
    # `covered_by_sources` sat in summary.json the whole time the 4.4e-06 residual went
    # unexplained -- 802 against 830 on afn_nosib -- and the replay simply never read it.
    # The lesson (session `271a46a8`, private `5faee4f`): check that your replay consumes
    # every field the record carries BEFORE suspecting the record. A mismatch here means
    # the rebuild has diverged from the real run, so the arm is reported, never scored.
    rec_cov = b.get("covered_by_sources")
    if rec_cov is not None and int(rec_cov) != int(cov.sum()):
        return out | {"status": "coverage_mismatch",
                      "why": f"rebuilt {int(cov.sum())} covered targets, record says {int(rec_cov)}"}
    got = score_delta(deltas, targets, fold_cache, alpha=out["alpha"], covered=cov)
    out |= {"status": "replayed", "analytic": got, "diff": got - out["recorded"],
            "substitutions": notes, "targets": len(targets)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fold", action="append", help="restrict to these folds (repeatable)")
    ap.add_argument("--out", type=Path, help="write the full result table as JSON")
    ap.add_argument("--tol", type=float, default=1e-9, help="agreement tolerance at lambda=0")
    args = ap.parse_args()

    rows = []
    for d in sorted(MIRRORS.iterdir()):
        if not (d / "bundle" / "manifest.json").exists():
            continue
        if args.fold and d.name not in args.fold:
            continue
        arms = sorted(a for a in d.iterdir() if (a / "summary.json").exists())
        if not arms:
            continue
        why = None
        real = real_h5ad(d.name)
        fold = None
        if real is None:
            why = "no real h5ad for this fold"
        else:
            try:
                # Pinned, not defaulted. Every arm on disk was scored before the
                # control-cell floor moved to 1000 on 2026-09-20 (T18 check 5), so the
                # arm this script exists to REPRODUCE was built at 500. An arm recorded
                # at any other floor is refused per-arm below rather than replayed
                # against the wrong control profile.
                fold = prep_fold(real, pert_col=PERT_COL.get(d.name, "perturbation"),
                                 min_libsize=LEGACY_MIN_LIBSIZE,
                                 cache=MIRRORS / d.name / "analytic_fold_cache.npz")
            except Exception as exc:                               # noqa: BLE001
                why = f"prep_fold failed: {exc}"
        if fold is None:
            for a in arms:
                rows.append({"fold": d.name, "arm": a.name, "status": "skipped", "why": why})
                print(f"--  {d.name:28s} {a.name:24s} skipped: {why}", flush=True)
            continue
        for a in arms:
            try:
                rows.append(replay_arm(a, d.name, fold))
            except BaseException as exc:                           # noqa: BLE001
                # SystemExit included: sidechain refuses bad configs by exiting, and one
                # refused arm must not take the whole benchmark down with it.
                rows.append({"fold": d.name, "arm": a.name, "status": "error", "why": repr(exc)})
            r = rows[-1]
            if r["status"] == "replayed":
                mark = "OK " if (r["lam"] == 0 and abs(r["diff"]) < args.tol) else \
                       "BRK" if r["lam"] > 0 else "!! "
                print(f"{mark} {r['fold']:28s} {r['arm']:24s} lam={r['lam']:<4g} "
                      f"rec={r['recorded']:.9f} got={r['analytic']:.9f} d={r['diff']:+.2e}",
                      flush=True)
            else:
                print(f"--  {r['fold']:28s} {r['arm']:24s} {r['status']}: {r.get('why')}",
                      flush=True)

    done = [r for r in rows if r["status"] == "replayed"]
    even = [r for r in done if r["lam"] == 0]
    lam = [r for r in done if r["lam"] > 0]
    agree = [r for r in even if abs(r["diff"]) < args.tol]
    print(f"\nreplayed {len(done)} of {len(rows)} arms; "
          f"lambda=0: {len(agree)}/{len(even)} agree within {args.tol:g}")
    if even:
        print(f"  worst |diff| at lambda=0: {max(abs(r['diff']) for r in even):.3e}")
    if lam:
        print("  lambda>0 (the identity SHOULD break):")
        for r in lam:
            print(f"    {r['fold']}/{r['arm']} lam={r['lam']:g} "
                  f"diff={r['diff']:+.4f}  ({abs(r['diff']) / max(r['recorded'], 1e-9):.1%} of the value)")
    if args.out:
        args.out.write_text(json.dumps(rows, indent=1) + "\n")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

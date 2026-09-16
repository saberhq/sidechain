#!/usr/bin/env python
"""Does protein-embedding geometry carry PERTURBATION-RESPONSE similarity?

The question that decides whether a gene-embedding prediction arm is worth training, answered
with no training, no GPU and no model -- from cached pseudobulks in a few minutes on a laptop.

Background, in one paragraph. A one-hot perturbation encoding gives every target gene its own
slot, learned independently, so a screen that measured gene Y teaches the model nothing about
gene X. Swapping the one-hot for a vector describing the gene's protein sequence is supposed to
buy CROSS-GENE TRANSFER: X and Y share representation, so measuring Y informs X. That purchase
is real only if genes near each other in embedding space actually respond similarly. This script
measures whether they do, by using the embedding as a nearest-neighbour index and nothing else --
deliberately the weakest possible learner, so the number it returns is a floor.

Two modes:

  within  For each target t in one corpus, predict t's response as the mean response of its k
          nearest embedding neighbours (t excluded). Upper bound: the neighbours are measured
          in the same line as the truth.

  cross   The deployment shape. Predict t's response in line B from line A only, three ways:
            SER arm   -- t's OWN response in A (what our backbone already does)
            ESM2 arm  -- t's NEIGHBOURS' responses in A, t never read (true gene generalisation)
            fusion    -- SER + w * ESM2, swept over w
          and report the per-target error correlation between the two arms, which is the number
          research/ideas/learned-arm-fusion.md pre-registered as the go/no-go for fusion.

Everything is measured on the RESIDUAL response -- each line's own mean response removed. That is
deliberate and it is the crux: the shared mean is already in the backbone, and `pds` ranks a
prediction against every other target's truth, so a component every target shares cannot help
tell them apart. Scoring against the uncentred response mostly measures how well you reproduced
the mean, which flatters every method equally.

Three controls, all reported, because a result without its control is a hypothesis:
  * SCRAMBLED embedding, same k -- is it the geometry, or just averaging k things?
  * RANDOM k others            -- same question, different angle.
  * the mean response          -- the real baseline (Rung 0b), not zero.

Raw cosine everywhere; no scaled scores. Genes missing from the embedding table are DROPPED, never
zero-filled: a zero vector is a fake nearest neighbour of every other zero vector, which would
manufacture the signal being tested.

Usage::

    python scripts/esm2_geometry_gate.py within k562_gwps_union_pseudobulk.npz
    python scripts/esm2_geometry_gate.py cross hepg2_all_pseudobulk.npz jurkat_all_pseudobulk.npz

``--embeddings <table.pt>`` scores any other symbol-keyed table on the same footing; ``--bootstrap
N`` (default 1000, 0 disables) appends a paired bootstrap over targets to the end of the output --
an interval on the margins that the point estimates alone never carried. In ``cross`` the fusion
interval is on the embedding maximum MINUS the scrambled maximum over the w grid, because the gain
at the best w is >= 0 whatever the embedding carries (w = 0 is in the sweep, and the best w is
chosen on the same targets). Every line that existed before the bootstrap was added still prints
byte for byte the same, for the same inputs and seed.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

# Retired HGNC symbols seen in our corpora, old -> current. The table lives in
# src/sidechain/data/gene_aliases.py -- ONE source of truth, shared with
# scripts/build_pert_features.py; its docstring carries the authority (HGNC), the verification
# date and the evidence file. Extend it there, never here.
from sidechain.data.gene_aliases import RETIRED_SYMBOLS as ALIAS

CACHE = Path("~/data/sidechain/cache/vcc2026").expanduser()
EMB = Path("~/data/sidechain/external/hf-arcinstitute-SE-600M/protein_embeddings.pt").expanduser()

CONTROL_LABELS = {"non-targeting", "Non-Targeting", "control", "unassigned"}

KS = (1, 3, 5, 10, 25, 50, 100, 200)
WS = (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0)


def load_delta(name: str):
    """Per-target log2 fold change against the corpus's own control row, target gene zeroed."""
    z = np.load(CACHE / name, allow_pickle=True)
    labels = np.array([str(x) for x in z["labels"]])
    genes = np.array([str(x) for x in z["genes"]])
    cpm = z["cpm_sum"] / z["n_cells"][:, None]
    ci = [i for i, l in enumerate(labels) if l in CONTROL_LABELS]
    if not ci:
        raise SystemExit(f"{name}: no control row among labels starting {labels[:5]}")
    ctrl = cpm[ci[0]]
    keep = np.array([i for i in range(len(labels)) if i not in set(ci)])
    labels, cpm = labels[keep], cpm[keep]
    d = np.log2((cpm + 1.0) / (ctrl + 1.0))
    gi = {g: i for i, g in enumerate(genes)}
    for r, l in enumerate(labels):          # pds excludes the target gene; so do we
        if l in gi:
            d[r, gi[l]] = 0.0
    return labels, genes, d


def embeddings(labels: np.ndarray, emb_path: Path):
    import torch

    table = torch.load(emb_path, weights_only=False, map_location="cpu")
    ok = np.array([(l in table) or (ALIAS.get(l) in table) for l in labels])
    e = np.stack([table[l if l in table else ALIAS[l]].numpy() for l in labels[ok]]).astype(float)
    return ok, e


def unit(m: np.ndarray) -> np.ndarray:
    return m / (np.linalg.norm(m, axis=1, keepdims=True) + 1e-12)


def knn_mean(sim: np.ndarray, source: np.ndarray, k: int) -> np.ndarray:
    s = sim.copy()
    np.fill_diagonal(s, -np.inf)            # a target is never its own neighbour
    idx = np.argsort(-s, axis=1)[:, :k]
    return np.stack([source[row].mean(0) for row in idx])


def boot_ci(values: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    """2.5/97.5 percentile of the mean of `values` under a PAIRED resample of target indices.

    Paired: one index draw is applied to the per-target differences themselves, so the embedding
    arm and its control are always resampled together and the arms' shared per-target difficulty
    cancels. The k-NN is deliberately NOT re-run per resample -- the neighbour sets are fixed by
    the embedding and the scramble permutation is fixed by the seed -- so the interval covers
    sampling noise in WHICH TARGETS were measured, and not noise in the neighbourhood structure.
    That makes it cheap (a mean over an array we already have) and slightly optimistic.
    """
    n = len(values)
    means = np.empty(n_boot)
    for i in range(n_boot):
        means[i] = values[rng.integers(0, n, n)].mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_header(n_boot: int, seed: int) -> None:
    print(f"\nbootstrap over targets: {n_boot} paired resamples, seed {seed}. The k-NN is NOT "
          f"re-run per\nresample -- neighbour sets are fixed by the embedding, the scramble "
          f"permutation by the seed --\nso the interval covers which-targets-were-measured noise "
          f"only.")


def boot_print(n_boot: int, label: str, point: float, lo: float, hi: float) -> None:
    verdict = "indistinguishable from zero" if lo <= 0.0 <= hi else "distinguishable from zero"
    print(f"bootstrap ({n_boot} paired resamples of targets, fixed neighbours and scramble): "
          f"{label} {point:+.4f} [ {lo:+.4f}, {hi:+.4f} ] -> {verdict}")


def boot_line(n_boot: int, label: str, values: np.ndarray, rng) -> None:
    lo, hi = boot_ci(values, n_boot, rng)
    boot_print(n_boot, label, float(values.mean()), lo, hi)


def boot_max_diff_ci(a: np.ndarray, b: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    """Interval on ``max_w mean(a_w) - max_w mean(b_w)`` under a paired resample of targets.

    ``a`` and ``b`` are (n_w, n_targets) per-target gains over the SAME weight grid -- the real
    fusion arm and the scrambled-embedding fusion arm. The best w is re-selected inside every
    resample, on both arms.

    Why the difference and not the gain itself: w = 0 is in the sweep, so the gain at the best w
    is >= 0 by construction and an interval on it can never straddle zero, whatever the embedding
    carries. Putting the scrambled arm through the SAME maximisation cancels that selection, so
    the difference is centred on zero when the geometry carries nothing. Same simplification as
    `boot_ci`: neighbour sets and the scramble permutation are fixed, not re-drawn per resample.
    """
    n = a.shape[1]
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        means[i] = a[:, idx].mean(1).max() - b[:, idx].mean(1).max()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_rng(seed: int):
    """A stream of its own, so every number printed before the bootstrap is bit-for-bit unchanged."""
    return np.random.default_rng([seed, 0xB007])


def run_within(corpus: str, seed: int, emb_path: Path, n_boot: int) -> None:
    rng = np.random.default_rng(seed)
    labels, _, d = load_delta(corpus)
    ok, e = embeddings(labels, emb_path)
    print(f"{corpus}: {ok.sum()}/{len(labels)} targets resolved by the embedding table "
          f"(unresolved are DROPPED, never zero-filled)")
    d = d[ok]
    n = len(d)
    r = d - d.mean(0)                       # residual: the shared mean response removed
    rn = unit(r)
    se = unit(e) @ unit(e).T
    perm = rng.permutation(n)
    se_scr = se[np.ix_(perm, perm)]
    shared = 1 - (np.linalg.norm(r) ** 2 / np.linalg.norm(d) ** 2)
    print(f"the shared mean response is {100 * shared:.1f}% of total squared length\n")

    def per_target(p):
        return (unit(p) * rn).sum(1)

    def score(p):
        return float(per_target(p).mean())

    margins: dict[int, np.ndarray] = {}      # per-target margin, kept for the bootstrap below
    print("mean cosine with the TRUE RESIDUAL (chance 0.000):")
    print(f'{"k":>5s} {"embedding":>11s} {"scrambled":>11s} {"random k":>10s} {"margin":>9s}')
    for k in KS:
        if k >= n:
            break
        ca = per_target(knn_mean(se, r, k))
        cb = per_target(knn_mean(se_scr, r, k))
        a, b = float(ca.mean()), float(cb.mean())
        if k in (10, 25):
            margins[k] = ca - cb
        ridx = np.stack([rng.choice(np.delete(np.arange(n), i), size=k, replace=False)
                         for i in range(n)])
        c = score(np.stack([r[row].mean(0) for row in ridx]))
        print(f"{k:5d} {a:+11.4f} {b:+11.4f} {c:+10.4f} {a - b:+9.4f}")

    if n_boot > 0 and margins:
        brng = boot_rng(seed)
        boot_header(n_boot, seed)
        for k in sorted(margins):
            boot_line(n_boot, f"k={k} margin", margins[k], brng)


def run_cross(a_name: str, b_name: str, k: int, seed: int, emb_path: Path, n_boot: int) -> None:
    from scipy.stats import pearsonr, spearmanr

    rng = np.random.default_rng(seed)
    la, ga, da = load_delta(a_name)
    lb, gb, db = load_delta(b_name)
    import torch

    table = torch.load(emb_path, weights_only=False, map_location="cpu")
    resolvable = {l for l in la if (l in table) or (ALIAS.get(l) in table)}
    targets = np.array(sorted(set(la) & set(lb) & resolvable))
    genes = np.array(sorted(set(ga) & set(gb)))
    print(f"A = {a_name}\nB = {b_name}")
    print(f"shared gene axis {len(genes):,} | shared resolved targets {len(targets):,}\n")

    gia, gib = {g: i for i, g in enumerate(ga)}, {g: i for i, g in enumerate(gb)}
    lia, lib = {l: i for i, l in enumerate(la)}, {l: i for i, l in enumerate(lb)}
    gcols_a = np.array([gia[g] for g in genes])
    gcols_b = np.array([gib[g] for g in genes])
    A = da[np.array([lia[t] for t in targets])][:, gcols_a]
    B = db[np.array([lib[t] for t in targets])][:, gcols_b]
    A, B = A - A.mean(0), B - B.mean(0)
    bn = unit(B)
    n = len(targets)

    e = np.stack([table[t if t in table else ALIAS[t]].numpy() for t in targets]).astype(float)
    se = unit(e) @ unit(e).T
    perm = rng.permutation(n)
    se_scr = se[np.ix_(perm, perm)]

    def score(p):
        return (unit(p) * bn).sum(1)

    ser, esm, scr = A, knn_mean(se, A, k), knn_mean(se_scr, A, k)
    print("mean cosine with B's TRUE residual (chance 0.000):")
    for name, p in [("SER arm: same gene in A", ser),
                    (f"embedding arm: k={k} neighbours in A", esm),
                    ("scrambled-embedding arm", scr),
                    ("A's mean residual", np.tile(A.mean(0), (n, 1)))]:
        c = score(p)
        print(f"  {name:38s} {c.mean():+.4f}   median {np.median(c):+.4f}")

    margin10 = None                         # per-target k=10 margin, kept for the bootstrap below
    print("\nk sweep for the embedding arm:")
    for kk in KS:
        if kk >= n:
            break
        ca = score(knn_mean(se, A, kk))
        cb = score(knn_mean(se_scr, A, kk))
        a, b = ca.mean(), cb.mean()
        if kk == 10:
            margin10 = ca - cb
        print(f"  k={kk:4d}  embedding {a:+.4f}  scramble {b:+.4f}  margin {a - b:+.4f}")

    cs, ce = score(ser), score(esm)
    print("\nfusion weight sweep -- SER + w * embedding (unit arms). Equal weight (w=1) gives a"
          "\nmuch weaker arm an equal vote and is NOT the right test:")
    print(f'{"w":>6s} {"fused":>9s} {"vs SER":>9s} {"scramble":>10s} {"geometry":>10s}')
    best = (0.0, float(cs.mean()))
    gain_emb, gain_scr = [], []      # per-target gain at each w, kept for the bootstrap below
    for w in WS:
        fv = score(unit(ser) + w * unit(esm))
        gv = score(unit(ser) + w * unit(scr))
        f, g = fv.mean(), gv.mean()
        gain_emb.append(fv - cs)
        gain_scr.append(gv - cs)
        print(f"{w:6.2f} {f:+9.4f} {f - cs.mean():+9.4f} {g:+10.4f} {f - g:+10.4f}")
        if f > best[1]:
            best = (w, float(f))
    print(f"\nbest w = {best[0]:.2f}, fused {best[1]:+.4f}, "
          f"gain over SER alone {best[1] - cs.mean():+.4f}")

    rho, pr_p = spearmanr(cs, ce)
    pear, pp = pearsonr(cs, ce)
    print("\nPER-TARGET ERROR CORRELATION between the two arms (the pre-registered go/no-go):")
    print(f"  Spearman {rho:+.4f} (p={pr_p:.1e})   Pearson {pear:+.4f} (p={pp:.1e})")
    print("  learned-arm-fusion kills the arm above ~0.7. Lower means the arms fail differently,"
          "\n  which is the entire case for fusing them.")

    if n_boot > 0:
        brng = boot_rng(seed)
        boot_header(n_boot, seed)
        boot_line(n_boot, f"k={k} margin", ce - score(scr), brng)
        ga, gb = np.stack(gain_emb), np.stack(gain_scr)
        lo, hi = boot_max_diff_ci(ga, gb, n_boot, brng)
        print("the raw gain at the best w is >= 0 by construction -- w = 0 is in the sweep and w "
              "was\nchosen on these same targets -- so the interval below is on the embedding "
              "maximum MINUS\nthe scrambled maximum, both re-selected over the same w grid inside "
              "every resample:")
        boot_print(n_boot, "fusion gain over the scrambled fusion (best w re-selected per "
                           "resample)",
                   float(ga.mean(1).max() - gb.mean(1).max()), lo, hi)
        if margin10 is not None:
            boot_line(n_boot, "k=10 margin (embedding arm - scrambled arm)", margin10, brng)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="mode", required=True)
    w = sub.add_parser("within", help="neighbours and truth in the same corpus (upper bound)")
    w.add_argument("corpus")
    c = sub.add_parser("cross", help="predict corpus B from corpus A (the deployment shape)")
    c.add_argument("corpus_a")
    c.add_argument("corpus_b")
    c.add_argument("-k", type=int, default=25)
    for p in (w, c):
        p.add_argument("--seed", type=int, default=20260903)
        p.add_argument("--embeddings", type=Path, default=None,
                       help="embedding table (.pt: symbol -> vector); default the ESM2 table "
                            "at the module-level EMB")
        p.add_argument("--bootstrap", type=int, default=1000, metavar="N",
                       help="paired resamples of targets for the 2.5/97.5%% interval; 0 disables")
    args = ap.parse_args()

    emb_path = Path(args.embeddings).expanduser() if args.embeddings else EMB
    if not emb_path.exists():
        raise SystemExit(f"embedding table not found at {emb_path}")
    if args.bootstrap < 0:
        raise SystemExit("--bootstrap takes a non-negative number of resamples (0 disables)")
    if args.mode == "within":
        run_within(args.corpus, args.seed, emb_path, args.bootstrap)
    else:
        run_cross(args.corpus_a, args.corpus_b, args.k, args.seed, emb_path, args.bootstrap)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

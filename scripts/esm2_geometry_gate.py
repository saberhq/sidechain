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
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

CACHE = Path("~/data/sidechain/cache/vcc2026").expanduser()
EMB = Path("~/data/sidechain/external/hf-arcinstitute-SE-600M/protein_embeddings.pt").expanduser()

CONTROL_LABELS = {"non-targeting", "Non-Targeting", "control", "unassigned"}

# Retired HGNC symbols seen in our corpora. Confirmed against genenames.org and cross-checked in
# NCBI Gene, Ensembl and UniProt on 2026-09-03. Kept in step with scripts/build_pert_features.py.
ALIAS = {
    "GARS": "GARS1", "LARS": "LARS1", "MARS": "MARS1", "NARS": "NARS1",
    "QARS": "QARS1", "YARS": "YARS1", "FGFR1OP": "CEP43",
    "HIST1H2BN": "H2BC15", "CCDC130": "YJU2B",
}

KS = (1, 3, 5, 10, 25, 50, 100, 200)


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


def embeddings(labels: np.ndarray):
    import torch

    table = torch.load(EMB, weights_only=False, map_location="cpu")
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


def run_within(corpus: str, seed: int) -> None:
    rng = np.random.default_rng(seed)
    labels, _, d = load_delta(corpus)
    ok, e = embeddings(labels)
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

    def score(p):
        return float((unit(p) * rn).sum(1).mean())

    print("mean cosine with the TRUE RESIDUAL (chance 0.000):")
    print(f'{"k":>5s} {"embedding":>11s} {"scrambled":>11s} {"random k":>10s} {"margin":>9s}')
    for k in KS:
        if k >= n:
            break
        a = score(knn_mean(se, r, k))
        b = score(knn_mean(se_scr, r, k))
        ridx = np.stack([rng.choice(np.delete(np.arange(n), i), size=k, replace=False)
                         for i in range(n)])
        c = score(np.stack([r[row].mean(0) for row in ridx]))
        print(f"{k:5d} {a:+11.4f} {b:+11.4f} {c:+10.4f} {a - b:+9.4f}")


def run_cross(a_name: str, b_name: str, k: int, seed: int) -> None:
    from scipy.stats import pearsonr, spearmanr

    rng = np.random.default_rng(seed)
    la, ga, da = load_delta(a_name)
    lb, gb, db = load_delta(b_name)
    import torch

    table = torch.load(EMB, weights_only=False, map_location="cpu")
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

    print("\nk sweep for the embedding arm:")
    for kk in KS:
        if kk >= n:
            break
        a = score(knn_mean(se, A, kk)).mean()
        b = score(knn_mean(se_scr, A, kk)).mean()
        print(f"  k={kk:4d}  embedding {a:+.4f}  scramble {b:+.4f}  margin {a - b:+.4f}")

    cs, ce = score(ser), score(esm)
    print("\nfusion weight sweep -- SER + w * embedding (unit arms). Equal weight (w=1) gives a"
          "\nmuch weaker arm an equal vote and is NOT the right test:")
    print(f'{"w":>6s} {"fused":>9s} {"vs SER":>9s} {"scramble":>10s} {"geometry":>10s}')
    best = (0.0, float(cs.mean()))
    for w in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.75, 1.0, 1.5, 2.0):
        f = score(unit(ser) + w * unit(esm)).mean()
        g = score(unit(ser) + w * unit(scr)).mean()
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
    args = ap.parse_args()

    if not EMB.exists():
        raise SystemExit(f"embedding table not found at {EMB}")
    if args.mode == "within":
        run_within(args.corpus, args.seed)
    else:
        run_cross(args.corpus_a, args.corpus_b, args.k, args.seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

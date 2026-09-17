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

Three modes:

  within  For each target t in one corpus, predict t's response as the mean response of its k
          nearest embedding neighbours (t excluded). Upper bound: the neighbours are measured
          in the same line as the truth.

  cross   The deployment shape. Predict t's response in line B from line A only, three ways:
            SER arm   -- t's OWN response in A (what our backbone already does)
            ESM2 arm  -- t's NEIGHBOURS' responses in A, t never read (true gene generalisation)
            fusion    -- SER + w * ESM2, swept over w
          and report the per-target error correlation between the two arms, which is the number
          research/ideas/learned-arm-fusion.md pre-registered as the go/no-go for fusion.

  compare Two rows against each other, PAIRED. Takes the per-target files two ``--dump`` runs
          wrote on the same footing (same corpora, same targets in the same order, same seed --
          anything else is refused) and bootstraps the row-minus-row difference under one shared
          resample of the targets and, where the permutation is resampled, one shared scramble
          draw. Two marginal intervals failing to overlap is a conservative test when the rows
          share every target; this is the test that decides "table X beats table Y", and the
          only place such a claim may come from.

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
    python scripts/esm2_geometry_gate.py compare row_a.npz row_b.npz

``--embeddings <table.pt>`` scores any other symbol-keyed table on the same footing; ``--bootstrap
N`` (default 1000, 0 disables) appends a paired bootstrap over targets to the end of the output --
an interval on the margins that the point estimates alone never carried. In ``cross`` the fusion
interval is on the embedding maximum MINUS the scrambled maximum over the w grid, because the gain
at the best w is >= 0 whatever the embedding carries (w = 0 is in the sweep, and the best w is
chosen on the same targets).

``--scrambles M`` (default 1) draws M scramble permutations instead of one. Draw 0 is the one every
pre-existing line uses; draws 1..M-1 come from a stream of their own, so nothing above the new
block moves. The block prints the scrambled arm's spread across draws -- what ONE permutation's
null carries, which on the 2026-09-15 batch was up to 0.005 and wider than the printed interval --
then the margin against the MEAN of the M draws, with a two-level bootstrap that redraws the
permutations with replacement as well as the targets. Ten draws cut the null's noise by sqrt(10).

The **extraction assay** (on by default, ``--no-assay`` to skip) prints, after everything else, the
mean cosine over forty human paralogue pairs against random pairs of the table's own genes, with a
z. A table that never left its initialisation reads z ~ 0 with a mean random cosine at zero and is
reported as EXTRACTION FAILED (three of the 2026-09-14 rows), so a DEAD verdict on it is never
mistaken for a verdict on the model; a table with z ~ 0 but a strongly non-zero random cosine is
reported as NO PARALOGUE STRUCTURE, which is a fact about what its neighbourhoods encode, not a
failed extraction (AIDO.Cell's line tables).

``--dump PATH`` writes the per-target cosines (embedding arm, every scramble draw, and in ``cross``
the SER arm and the per-w fusion gains) to an ``.npz`` for ``compare``. Every line that existed
before these options were added still prints byte for byte the same, for the same inputs and seed.
"""

from __future__ import annotations

import argparse
import json
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
BOOT_KS = (10, 25)              # the two fixed k every verdict is read at, never the sweep's best
SCRAMBLE_STREAM = 0x5C4A        # the rng stream the extra scramble permutations are drawn from

# The pre-gate extraction assay (T89, the T45 critics' third ask): a table that knows what a gene
# is puts paralogues near each other; a tensor that never left its initialisation does not. Forty
# human paralogue / close-family pairs, current HGNC symbols, chosen to span kinases, GTPases,
# cytoskeleton, chaperones, histones, transporters and transcription factors. The assay compares
# the mean cosine over the pairs the table resolves with that over random pairs of the same genes'
# universe; a z under EXTRACTION_Z_MIN means "the verdicts below are about the extraction, not the
# model" -- three T45 rows (scFoundation pos_emb, GREmLN, scLDM-20M) were exactly that.
PARALOG_PAIRS = (
    ("MAPK1", "MAPK3"), ("AKT1", "AKT2"), ("KRAS", "NRAS"), ("HRAS", "NRAS"), ("CDK4", "CDK6"),
    ("RAC1", "RAC2"), ("RHOA", "RHOB"), ("CDC42", "RAC1"), ("ACTB", "ACTG1"), ("TUBA1A", "TUBA1B"),
    ("TUBB", "TUBB4B"), ("HSPA1A", "HSPA1B"), ("HSPA8", "HSPA1A"), ("HSP90AA1", "HSP90AB1"),
    ("H2BC12", "H2BC15"), ("H4C1", "H4C2"), ("ATP1A1", "ATP1A2"), ("SLC2A1", "SLC2A3"),
    ("GAPDH", "GAPDHS"), ("ENO1", "ENO2"), ("PKM", "PKLR"), ("LDHA", "LDHB"), ("MYC", "MYCN"),
    ("JUN", "JUNB"), ("FOS", "FOSB"), ("STAT1", "STAT3"), ("SMAD2", "SMAD3"), ("CCND1", "CCND2"),
    ("CDKN1A", "CDKN1B"), ("BCL2", "BCL2L1"), ("CASP3", "CASP7"), ("EIF4A1", "EIF4A2"),
    ("RPL7", "RPL7A"), ("RPS27", "RPS27A"), ("PSMB5", "PSMB6"), ("PSMA1", "PSMA2"),
    ("POLR2A", "POLR2B"), ("SRSF1", "SRSF2"), ("HNRNPA1", "HNRNPA2B1"), ("YWHAB", "YWHAZ"),
)
EXTRACTION_Z_MIN = 2.0          # below this the table carries no paralogue structure
EXTRACTION_ISO_COS = 0.02       # |mean random cosine| under this AND z under the line: untrained
EXTRACTION_MIN_PAIRS = 10       # fewer resolved pairs than this and the assay is not run


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


def verdict_of(lo: float, hi: float) -> str:
    return "indistinguishable from zero" if lo <= 0.0 <= hi else "distinguishable from zero"


def boot_print(n_boot: int, label: str, point: float, lo: float, hi: float) -> None:
    print(f"bootstrap ({n_boot} paired resamples of targets, fixed neighbours and scramble): "
          f"{label} {point:+.4f} [ {lo:+.4f}, {hi:+.4f} ] -> {verdict_of(lo, hi)}")


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


# ------------------------------------------------------------------ the extraction assay

def extraction_assay(table: dict, seed: int, n_random: int = 4000) -> dict | None:
    """Paralogue-versus-random cosine on the raw table: is there any gene geometry to gate?

    Mean cosine over the `PARALOG_PAIRS` the table resolves (through the alias table too) against
    the mean over `n_random` random pairs of the table's own genes; z is the paralogue mean's
    distance from the random mean in units of the random pairs' standard error at the paralogue
    count. Returns None when fewer than EXTRACTION_MIN_PAIRS pairs resolve. Draws from a stream of
    its own, so nothing else printed moves.
    """
    def vec(g):
        if g in table:
            return table[g]
        a = ALIAS.get(g)
        return table[a] if a is not None and a in table else None

    pairs = [(a, b) for a, b in PARALOG_PAIRS if vec(a) is not None and vec(b) is not None]
    if len(pairs) < EXTRACTION_MIN_PAIRS:
        return None
    par = np.stack([np.concatenate([np.asarray(vec(a), dtype=float).ravel(),
                                    np.asarray(vec(b), dtype=float).ravel()]) for a, b in pairs])
    d = par.shape[1] // 2
    pa, pb = unit(par[:, :d]), unit(par[:, d:])
    par_cos = (pa * pb).sum(1)
    keys = list(table)
    rng = np.random.default_rng([seed, 0xE7A])
    i = rng.integers(0, len(keys), n_random)
    j = rng.integers(0, len(keys), n_random)
    ok = i != j
    ra = unit(np.stack([np.asarray(table[keys[x]], dtype=float).ravel() for x in i[ok]]))
    rb = unit(np.stack([np.asarray(table[keys[x]], dtype=float).ravel() for x in j[ok]]))
    rnd_cos = (ra * rb).sum(1)
    se = rnd_cos.std(ddof=1) / np.sqrt(len(pairs))
    z = float((par_cos.mean() - rnd_cos.mean()) / se) if se > 0 else float("inf")
    rc = float(rnd_cos.mean())
    # Two tiers, because a table built from expression context can carry real neighbourhoods
    # without putting paralogues together (AIDO.Cell's Jurkat table: z -1.5, mean random cosine
    # +0.57, yet it beats its scramble on every within arm), while an initialisation tensor has
    # neither: paralogues at random AND a mean random cosine at zero (GREmLN, scFoundation
    # pos_emb, AIDO positional: z 0.5 / -0.6 / 0.7 with random cosine 0.000).
    return {"n_pairs": len(pairs), "paralog_cos": float(par_cos.mean()), "random_cos": rc,
            "n_random": int(ok.sum()), "z": z, "ok": z >= EXTRACTION_Z_MIN,
            "untrained": z < EXTRACTION_Z_MIN and abs(rc) < EXTRACTION_ISO_COS}


def extraction_print(res: dict | None) -> None:
    print("\nextraction assay (paralogue pairs against random pairs of the table's own genes):")
    if res is None:
        print(f"  not run: fewer than {EXTRACTION_MIN_PAIRS} of the {len(PARALOG_PAIRS)} paralogue "
              f"pairs resolve in this table")
        return
    if res["ok"]:
        verdict = "paralogue structure present"
    elif res["untrained"]:
        verdict = (f"EXTRACTION FAILED: paralogues at random (z < {EXTRACTION_Z_MIN:.0f}) and a mean "
                   f"random cosine at zero -- this reads as an untrained tensor, so every verdict "
                   f"above is about the extraction, not the model")
    else:
        verdict = (f"NO PARALOGUE STRUCTURE (z < {EXTRACTION_Z_MIN:.0f}) in an anisotropic table "
                   f"(mean random cosine {res['random_cos']:+.2f}): not an initialisation tensor; "
                   f"its neighbourhoods, if any, are not gene-family ones -- read the within arms")
    print(f"  {res['n_pairs']} pairs: paralogue cos {res['paralog_cos']:+.4f}  random cos "
          f"{res['random_cos']:+.4f} ({res['n_random']:,} pairs)  z {res['z']:+.2f} -> {verdict}")


# ------------------------------------------------------------------- several scramble draws

def scramble_rng(seed: int, j: int):
    """Draw ``j >= 1`` of the scramble permutation.

    A stream of its own per draw, so draw 0 -- the legacy permutation, taken from the main rng
    before anything else -- and every number printed before the scramble block are unchanged, and
    so that draw j is the same permutation whether M is 5 or 10.
    """
    return np.random.default_rng([seed, SCRAMBLE_STREAM, j])


def scramble_perms(n: int, seed: int, n_scr: int, first: np.ndarray) -> list[np.ndarray]:
    """The M permutations: ``first`` (draw 0, from the main rng) then M-1 from their own streams."""
    return [first] + [scramble_rng(seed, j).permutation(n) for j in range(1, n_scr)]


def boot_rng_scr(seed: int):
    """The stream for the permutation-resampled bootstrap; separate from `boot_rng` so the
    fixed-scramble intervals do not move when ``--scrambles`` changes."""
    return np.random.default_rng([seed, 0xB007, SCRAMBLE_STREAM])


def boot_ci_scr(emb: np.ndarray, scr: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    """Interval on ``mean_t(emb - mean_j scr[j])`` with the targets AND the permutation draws
    resampled: a two-level bootstrap.

    ``scr`` is (n_draws, n_targets), one row per permutation. Every resample draws the M
    permutation indices WITH replacement and averages the scrambled arm over them per target,
    then draws the n target indices, so the interval brackets the M-draw-mean margin the gate
    prints beside it and carries which-targets-were-measured noise AND the Monte Carlo noise of
    that mean. What ONE permutation carries -- sd 0.0025 across draws at k=25 on the 2026-09-16
    sweep, up to 0.014 -- is the spread line printed above it; averaging M draws cuts it by
    sqrt(M). (The first cut of this function, 2026-09-16 morning, picked one draw per resample,
    which priced a single permutation's noise against a point that averaged ten: conservative by
    about sqrt(10) in that term, and a critic caught it.) Neighbour sets are still fixed by the
    embedding.
    """
    m, n = scr.shape
    means = np.empty(n_boot)
    for i in range(n_boot):
        js = rng.integers(0, m, m)              # the M draws, redrawn with replacement
        idx = rng.integers(0, n, n)
        means[i] = (emb[idx] - scr[js][:, idx].mean(0)).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_max_diff_ci_scr(a: np.ndarray, b: np.ndarray, n_boot: int, rng) -> tuple[float, float]:
    """`boot_max_diff_ci` with the scrambled fusion arm's permutation resampled too.

    ``a`` is (n_w, n_targets); ``b`` is (n_draws, n_w, n_targets), one scrambled fusion arm per
    permutation. Each resample draws the M permutation indices with replacement, averages the
    scrambled fusion arm over them, draws the targets, then re-selects the best w on both arms.
    The point it brackets is ``max_w mean(a_w) - max_w mean_t(mean_j b[j]_w)``.
    """
    m, n = b.shape[0], a.shape[1]
    means = np.empty(n_boot)
    for i in range(n_boot):
        js = rng.integers(0, m, m)
        idx = rng.integers(0, n, n)
        bs = b[js].mean(0)
        means[i] = a[:, idx].mean(1).max() - bs[:, idx].mean(1).max()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def scr_header(n_scr: int, seed: int) -> None:
    print(f"\nscramble draws: {n_scr} permutations. Draw 0 is the one every line above used; draws "
          f"1-{n_scr - 1} come from a\nstream of their own (seed {seed}), so nothing above moved. "
          f"The spread across draws is what ONE\npermutation's null carries; the margins below are "
          f"read against the mean of all {n_scr}, and their intervals\nresample the draws with "
          f"replacement as well as the targets:")


def scr_spread_line(label: str, draw_means: np.ndarray) -> None:
    print(f"  {label:<26s} over {len(draw_means)} draws: mean {draw_means.mean():+.4f}  "
          f"sd {draw_means.std(ddof=1):.4f}  min {draw_means.min():+.4f}  "
          f"max {draw_means.max():+.4f}")


def boot_print_scr(n_boot: int, n_scr: int, label: str, point: float, lo: float, hi: float) -> None:
    print(f"bootstrap over targets AND scramble draws ({n_boot} resamples; {n_scr} permutations "
          f"redrawn with replacement and averaged): "
          f"{label} {point:+.4f} [ {lo:+.4f}, {hi:+.4f} ] -> {verdict_of(lo, hi)}")


def write_dump(path: Path, **arrays) -> None:
    """The per-target cosines of one run, for ``compare``. Scalars become 0-d arrays; float
    arrays are stored as float32 (the cosines print to four decimals, and a 420-cell sweep at ten
    draws is ~0.5 GB at float32 -- disk is the scarce thing on the Mac)."""
    path = Path(path).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    slim = {key: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype.kind == "f" else v)
            for key, v in arrays.items()}
    np.savez_compressed(path, **slim)
    print(f"\nper-target cosines -> {path}")


def run_within(corpus: str, seed: int, emb_path: Path, n_boot: int, n_scr: int = 1,
               dump: Path | None = None, assay: bool = True) -> None:
    rng = np.random.default_rng(seed)
    labels, _, d = load_delta(corpus)
    ok, e = embeddings(labels, emb_path)
    print(f"{corpus}: {ok.sum()}/{len(labels)} targets resolved by the embedding table "
          f"(unresolved are DROPPED, never zero-filled)")
    d = d[ok]
    labels = labels[ok]
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

    emb_k: dict[int, np.ndarray] = {}        # per-target embedding-arm cosine at the fixed k
    scr_k: dict[int, np.ndarray] = {}        # ... and the scrambled arm's, draw 0
    print("mean cosine with the TRUE RESIDUAL (chance 0.000):")
    print(f'{"k":>5s} {"embedding":>11s} {"scrambled":>11s} {"random k":>10s} {"margin":>9s}')
    for k in KS:
        if k >= n:
            break
        ca = per_target(knn_mean(se, r, k))
        cb = per_target(knn_mean(se_scr, r, k))
        a, b = float(ca.mean()), float(cb.mean())
        if k in BOOT_KS:
            emb_k[k], scr_k[k] = ca, cb
        ridx = np.stack([rng.choice(np.delete(np.arange(n), i), size=k, replace=False)
                         for i in range(n)])
        c = score(np.stack([r[row].mean(0) for row in ridx]))
        print(f"{k:5d} {a:+11.4f} {b:+11.4f} {c:+10.4f} {a - b:+9.4f}")

    if n_boot > 0 and emb_k:
        brng = boot_rng(seed)
        boot_header(n_boot, seed)
        for k in sorted(emb_k):
            boot_line(n_boot, f"k={k} margin", emb_k[k] - scr_k[k], brng)

    # the extra scramble draws: (n_scr, n) per fixed k, row 0 the legacy permutation
    perms = scramble_perms(n, seed, n_scr, perm)
    scr_all = {k: np.stack([scr_k[k]] + [per_target(knn_mean(se[np.ix_(p, p)], r, k))
                                          for p in perms[1:]])
               for k in emb_k}
    if n_scr > 1 and emb_k:
        scr_header(n_scr, seed)
        for k in sorted(emb_k):
            scr_spread_line(f"k={k} scrambled arm", scr_all[k].mean(1))
        if n_boot > 0:
            srng = boot_rng_scr(seed)
            for k in sorted(emb_k):
                lo, hi = boot_ci_scr(emb_k[k], scr_all[k], n_boot, srng)
                boot_print_scr(n_boot, n_scr, f"k={k} margin",
                               float(emb_k[k].mean() - scr_all[k].mean()), lo, hi)

    if assay:
        import torch
        extraction_print(extraction_assay(torch.load(emb_path, weights_only=False,
                                                     map_location="cpu"), seed))

    if dump is not None:
        write_dump(dump, mode="within", corpus_a=corpus, corpus_b="", table=str(emb_path),
                   seed=seed, k_header=0, n_scrambles=n_scr, targets=labels,
                   ks=np.array(sorted(emb_k)),
                   **{f"emb_k{k}": emb_k[k] for k in emb_k},
                   **{f"scr_k{k}": scr_all[k] for k in emb_k})


def run_cross(a_name: str, b_name: str, k: int, seed: int, emb_path: Path, n_boot: int,
              n_scr: int = 1, dump: Path | None = None, assay: bool = True) -> None:
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
    a_mean = A.mean(0)              # ~1e-17 by construction; taken here so its printed sign of
    #                                 zero is the one the 2026-09-15 logs carry (summation order)
    # The column fancy-index above leaves A and B Fortran-ordered, and every k-NN pass gathers
    # ROWS of A: on 2,132 targets that gather cost 108 s at k=200 (250 of a run's 260 s) against
    # 1.6 s C-ordered. Same values, same arithmetic, only the memory layout changes.
    A, B = np.ascontiguousarray(A), np.ascontiguousarray(B)
    bn = unit(B)
    n = len(targets)
    if k >= n:
        raise SystemExit(f"-k {k} needs more than {n} resolved targets: at k >= n every target is "
                         f"its own neighbour and the embedding arm is identically zero (the k "
                         f"sweep skips such k; the header arm cannot)")

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
                    ("A's mean residual", np.tile(a_mean, (n, 1)))]:
        c = score(p)
        print(f"  {name:38s} {c.mean():+.4f}   median {np.median(c):+.4f}")

    emb_k: dict[int, np.ndarray] = {}        # per-target embedding-arm cosine at the fixed k
    scr_k: dict[int, np.ndarray] = {}        # ... and the scrambled arm's, draw 0
    print("\nk sweep for the embedding arm:")
    for kk in KS:
        if kk >= n:
            break
        ca = score(knn_mean(se, A, kk))
        cb = score(knn_mean(se_scr, A, kk))
        a, b = ca.mean(), cb.mean()
        if kk in BOOT_KS:
            emb_k[kk], scr_k[kk] = ca, cb
        print(f"  k={kk:4d}  embedding {a:+.4f}  scramble {b:+.4f}  margin {a - b:+.4f}")
    margin10 = emb_k[10] - scr_k[10] if 10 in emb_k else None

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

    ga_, gb_ = np.stack(gain_emb), np.stack(gain_scr)
    if n_boot > 0:
        brng = boot_rng(seed)
        boot_header(n_boot, seed)
        boot_line(n_boot, f"k={k} margin", ce - score(scr), brng)
        lo, hi = boot_max_diff_ci(ga_, gb_, n_boot, brng)
        print("the raw gain at the best w is >= 0 by construction -- w = 0 is in the sweep and w "
              "was\nchosen on these same targets -- so the interval below is on the embedding "
              "maximum MINUS\nthe scrambled maximum, both re-selected over the same w grid inside "
              "every resample:")
        boot_print(n_boot, "fusion gain over the scrambled fusion (best w re-selected per "
                           "resample)",
                   float(ga_.mean(1).max() - gb_.mean(1).max()), lo, hi)
        if margin10 is not None:
            boot_line(n_boot, "k=10 margin (embedding arm - scrambled arm)", margin10, brng)

    # the extra scramble draws: the scrambled arm at each fixed k, (n_scr, n), and the scrambled
    # fusion arm at the header k, (n_scr, n_w, n); row 0 is the legacy permutation everywhere
    perms = scramble_perms(n, seed, n_scr, perm)
    scr_all = {kk: [scr_k[kk]] for kk in emb_k}
    scr_h = [score(scr)]                                  # the header-k scrambled arm per draw
    gain_scr_all = [gb_]
    for p in perms[1:]:
        se_p = se[np.ix_(p, p)]
        scr_p = knn_mean(se_p, A, k)                     # the header k, once
        for kk in emb_k:
            scr_all[kk].append(score(scr_p) if kk == k else score(knn_mean(se_p, A, kk)))
        scr_h.append(score(scr_p))
        gain_scr_all.append(np.stack([score(unit(ser) + w * unit(scr_p)) - cs for w in WS]))
    scr_all = {kk: np.stack(v) for kk, v in scr_all.items()}
    scr_h = np.stack(scr_h)
    gain_scr_all = np.stack(gain_scr_all)

    if n_scr > 1:
        scr_header(n_scr, seed)
        scr_spread_line(f"k={k} scrambled arm", scr_h.mean(1))
        for kk in sorted(emb_k):
            if kk != k:
                scr_spread_line(f"k={kk} scrambled arm", scr_all[kk].mean(1))
        # the scrambled fusion's own best-w gain is ~0 by construction (w = 0 is in the grid), so
        # the draw-to-draw number worth seeing is its gain AT THE EMBEDDING ARM'S best w
        w_best = int(np.argmax(ga_.mean(1)))
        scr_spread_line(f"scrambled fusion at w={WS[w_best]:.2f}", gain_scr_all[:, w_best, :].mean(1))
        if n_boot > 0:
            srng = boot_rng_scr(seed)
            lo, hi = boot_ci_scr(ce, scr_h, n_boot, srng)
            boot_print_scr(n_boot, n_scr, f"k={k} margin", float(ce.mean() - scr_h.mean()), lo, hi)
            lo, hi = boot_max_diff_ci_scr(ga_, gain_scr_all, n_boot, srng)
            boot_print_scr(n_boot, n_scr, "fusion gain over the scrambled fusion (best w "
                                          "re-selected per resample)",
                           float(ga_.mean(1).max() - gain_scr_all.mean(0).mean(1).max()), lo, hi)
            if 10 in emb_k:
                lo, hi = boot_ci_scr(emb_k[10], scr_all[10], n_boot, srng)
                boot_print_scr(n_boot, n_scr, "k=10 margin (embedding arm - scrambled arm)",
                               float(emb_k[10].mean() - scr_all[10].mean()), lo, hi)

    if assay:
        extraction_print(extraction_assay(table, seed))

    if dump is not None:
        write_dump(dump, mode="cross", corpus_a=a_name, corpus_b=b_name, table=str(emb_path),
                   seed=seed, k_header=k, n_scrambles=n_scr, targets=targets,
                   ks=np.array(sorted(emb_k)), ws=np.array(WS),
                   ser=cs, emb_kh=ce, scr_kh=scr_h, gain_emb=ga_, gain_scr=gain_scr_all,
                   **{f"emb_k{kk}": emb_k[kk] for kk in emb_k},
                   **{f"scr_k{kk}": scr_all[kk] for kk in emb_k})


# ---------------------------------------------------------------------- compare, paired

def load_dump(path: Path) -> dict:
    z = np.load(Path(path).expanduser(), allow_pickle=False)
    out = {key: (z[key].astype(np.float64) if z[key].dtype.kind == "f" else z[key])
           for key in z.files}
    for key in ("mode", "corpus_a", "corpus_b", "table"):
        out[key] = str(out[key])
    for key in ("seed", "k_header", "n_scrambles"):
        out[key] = int(out[key])
    out["targets"] = np.array([str(t) for t in out["targets"]])
    out["ks"] = [int(x) for x in out["ks"]]
    return out


def boot_ci_pair_scr(ea, sa, eb, sb, n_boot: int, rng) -> tuple[float, float]:
    """Interval on the paired difference of two rows' margins, the permutation resampled too.

    Paired twice over. One target index draw serves both rows, and one redraw of the scramble
    draws (with replacement, averaged, as in `boot_ci_scr`) serves both rows: draw j is a pure
    function of (seed, n, j) -- draw 0 from the main rng, the rest
    from `scramble_rng` -- so two rows on one footing with one seed used the SAME permutation for
    every j, applied to two tables. Picking j independently per row would price a noise term the
    data do not carry (a row against itself would then read non-zero, and on the 2026-09-16 sweep
    it flipped 14 of 432 lead calls either way). Where the rows carry different numbers of draws,
    the common draws are used, and `compare` says so.
    """
    m, n = min(sa.shape[0], sb.shape[0]), len(ea)
    means = np.empty(n_boot)
    for i in range(n_boot):
        js = rng.integers(0, m, m)              # the same redrawn draws serve both rows
        idx = rng.integers(0, n, n)
        means[i] = ((ea[idx] - sa[js][:, idx].mean(0))
                    - (eb[idx] - sb[js][:, idx].mean(0))).mean()
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def boot_fusion_pair_scr(a: dict, b: dict, n_boot: int, rng) -> tuple[float, float]:
    """Paired difference of the two rows' fusion gains over their own scrambled fusions, best w
    re-selected per resample and per row; one target draw and one redraw of the scramble draws
    serve both rows, for the reason `boot_ci_pair_scr` gives."""
    m, n = min(a["gain_scr"].shape[0], b["gain_scr"].shape[0]), a["gain_emb"].shape[1]
    means = np.empty(n_boot)
    for i in range(n_boot):
        js = rng.integers(0, m, m)
        idx = rng.integers(0, n, n)
        sa, sb = a["gain_scr"][js].mean(0), b["gain_scr"][js].mean(0)
        ga = a["gain_emb"][:, idx].mean(1).max() - sa[:, idx].mean(1).max()
        gb = b["gain_emb"][:, idx].mean(1).max() - sb[:, idx].mean(1).max()
        means[i] = ga - gb
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def lead_of(lo: float, hi: float, name_a: str, name_b: str) -> str:
    if lo > 0.0:
        return f"{name_a} leads"
    if hi < 0.0:
        return f"{name_b} leads"
    return "indistinguishable"


def run_compare(a_path: Path, b_path: Path, n_boot: int, seed: int,
                json_out: Path | None = None) -> None:
    a, b = load_dump(a_path), load_dump(b_path)
    name_a, name_b = Path(a_path).stem, Path(b_path).stem
    if a["mode"] != b["mode"]:
        raise SystemExit(f"modes differ: A is {a['mode']}, B is {b['mode']}")
    if len(a["targets"]) != len(b["targets"]) or not np.array_equal(a["targets"], b["targets"]):
        common = len(set(a["targets"]) & set(b["targets"]))
        raise SystemExit(f"the two rows are not on the same footing: A has {len(a['targets'])} "
                         f"targets, B has {len(b['targets'])}, {common} in common -- a paired "
                         f"test needs the same targets in the same order (score both rows on one "
                         f"common footing first)")
    if a["mode"] == "cross" and a["k_header"] != b["k_header"]:
        raise SystemExit(f"header k differs: A used k={a['k_header']}, B k={b['k_header']}")
    if (a["corpus_a"], a["corpus_b"]) != (b["corpus_a"], b["corpus_b"]):
        raise SystemExit(f"the two rows were scored on different corpora -- A on "
                         f"{a['corpus_a']} / {a['corpus_b'] or '-'}, B on {b['corpus_a']} / "
                         f"{b['corpus_b'] or '-'} -- so their targets share names, not truths; "
                         f"a paired test compares two tables on ONE corpus pair")
    if a["seed"] != b["seed"]:
        raise SystemExit(f"the rows were scored with different seeds ({a['seed']} and "
                         f"{b['seed']}), so draw j of the scramble is a different permutation in "
                         f"each and nothing below would be paired; score both with one --seed")
    if n_boot <= 0:
        raise SystemExit("compare needs --bootstrap N > 0: the paired difference IS the interval")
    n = len(a["targets"])
    m = min(a["n_scrambles"], b["n_scrambles"])
    print(f"compare, PAIRED: A = {name_a}\n                 B = {name_b}")
    print(f"mode {a['mode']} | corpora {a['corpus_a']}" + (f" -> {a['corpus_b']}" if a["corpus_b"]
                                                            else "")
          + f" | {n:,} targets, identical and in the same order in both rows | seed {a['seed']}"
          f"\ntables:\n  A {a['table']}\n  B {b['table']}")
    print(f"scramble draws: A {a['n_scrambles']}, B {b['n_scrambles']}"
          + (f" -> paired on the {m} common draws" if a["n_scrambles"] != b["n_scrambles"] else ""))

    rng = boot_rng(seed)
    print(f"\npaired difference A - B: one resample of the targets serves both rows ({n_boot} "
          f"resamples, seed {seed}), and where\nthe permutation is resampled one redraw of the "
          f"draws (with replacement, averaged) serves both rows too --\ndraw j is the same "
          f"permutation in each. `leads` = the interval is clear of zero; the share is the "
          f"fraction of\ntargets on which A's per-target cosine is the higher.")
    rows = []

    def emit(label: str, point: float, lo: float, hi: float, per_target=None) -> None:
        lead = lead_of(lo, hi, "A", "B")
        share = None if per_target is None else float((per_target > 0).mean())
        tail = "" if share is None else f" | A ahead on {100 * share:.0f}% of targets"
        print(f"  {label:<68s} {point:+.4f} [ {lo:+.4f}, {hi:+.4f} ] -> {lead}{tail}")
        rows.append({"what": label, "diff": point, "lo": lo, "hi": hi, "lead": lead,
                     "share_a": share})

    ks = [k for k in a["ks"] if k in b["ks"]]
    for k in ks:
        ea, eb = a[f"emb_k{k}"], b[f"emb_k{k}"]
        sa, sb = a[f"scr_k{k}"], b[f"scr_k{k}"]
        d = ea - eb
        lo, hi = boot_ci(d, n_boot, rng)
        emit(f"k={k} embedding arm", float(d.mean()), lo, hi, d)
        d = (ea - sa[0]) - (eb - sb[0])
        lo, hi = boot_ci(d, n_boot, rng)
        emit(f"k={k} margin, scramble draw 0 fixed on both rows", float(d.mean()), lo, hi, d)
        if a["n_scrambles"] > 1 or b["n_scrambles"] > 1:
            lo, hi = boot_ci_pair_scr(ea, sa, eb, sb, n_boot, rng)
            d = (ea - sa[:m].mean(0)) - (eb - sb[:m].mean(0))
            emit(f"k={k} margin, permutation resampled ({m} draws, shared)", float(d.mean()),
                 lo, hi, d)

    if a["mode"] == "cross":
        ga, gb = a["gain_emb"], b["gain_emb"]
        means = np.empty(n_boot)                      # draw 0 fixed: `boot_max_diff_ci`, paired
        for i in range(n_boot):
            idx = rng.integers(0, n, n)
            means[i] = ((ga[:, idx].mean(1).max() - a["gain_scr"][0][:, idx].mean(1).max())
                        - (gb[:, idx].mean(1).max() - b["gain_scr"][0][:, idx].mean(1).max()))
        point = float((ga.mean(1).max() - a["gain_scr"][0].mean(1).max())
                      - (gb.mean(1).max() - b["gain_scr"][0].mean(1).max()))
        emit("fusion gain over the scrambled fusion, scramble draw 0 fixed (best w per resample, "
             "per row)", point, float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))
        if a["n_scrambles"] > 1 or b["n_scrambles"] > 1:
            lo, hi = boot_fusion_pair_scr(a, b, n_boot, rng)
            point = float((ga.mean(1).max() - a["gain_scr"][:m].mean(0).mean(1).max())
                          - (gb.mean(1).max() - b["gain_scr"][:m].mean(0).mean(1).max()))
            emit("fusion gain over the scrambled fusion, permutation resampled (best w per "
                 "resample, per row)", point, lo, hi)

    if json_out is not None:
        out = Path(json_out).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"a": name_a, "b": name_b, "mode": a["mode"], "n_targets": n,
                                   "corpus_a": a["corpus_a"], "corpus_b": a["corpus_b"],
                                   "k_header": a["k_header"], "n_boot": n_boot, "seed": seed,
                                   "seed_rows": a["seed"],
                                   "scrambles": [a["n_scrambles"], b["n_scrambles"]],
                                   "scrambles_paired": m,
                                   "table_a": a["table"], "table_b": b["table"],
                                   "rows": rows}, indent=1))
        print(f"\n-> {out}")


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
        p.add_argument("--scrambles", type=int, default=1, metavar="M",
                       help="scramble permutations to draw (default 1, the legacy single draw); "
                            "M > 1 appends the spread across draws and a bootstrap that "
                            "resamples the permutation too")
        p.add_argument("--dump", type=Path, default=None, metavar="PATH",
                       help="write the per-target cosines to this .npz, for `compare`")
        p.add_argument("--no-assay", action="store_true",
                       help="skip the paralogue-versus-random extraction assay printed after the "
                            "gate (on by default since T89; a failed assay means the table is a "
                            "random tensor and the verdicts are about the extraction)")
    cp = sub.add_parser("compare", help="two --dump files on one footing: the PAIRED row-minus-"
                                        "row difference, with its interval")
    cp.add_argument("row_a", type=Path)
    cp.add_argument("row_b", type=Path)
    cp.add_argument("--seed", type=int, default=20260903)
    cp.add_argument("--bootstrap", type=int, default=1000, metavar="N")
    cp.add_argument("--json", type=Path, default=None, metavar="PATH",
                    help="also write the differences and intervals here")
    args = ap.parse_args()

    if args.bootstrap < 0:
        raise SystemExit("--bootstrap takes a non-negative number of resamples (0 disables)")
    if args.mode == "compare":
        for p in (args.row_a, args.row_b):
            if not Path(p).expanduser().exists():
                raise SystemExit(f"no such dump: {p}")
        run_compare(args.row_a, args.row_b, args.bootstrap, args.seed, args.json)
        return 0

    emb_path = Path(args.embeddings).expanduser() if args.embeddings else EMB
    if not emb_path.exists():
        raise SystemExit(f"embedding table not found at {emb_path}")
    if args.scrambles < 1:
        raise SystemExit("--scrambles takes a positive number of permutations (1 = the legacy "
                         "single draw)")
    if args.mode == "within":
        run_within(args.corpus, args.seed, emb_path, args.bootstrap, args.scrambles, args.dump,
                   not args.no_assay)
    else:
        run_cross(args.corpus_a, args.corpus_b, args.k, args.seed, emb_path, args.bootstrap,
                  args.scrambles, args.dump, not args.no_assay)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Does the full-corpus X-Atlas rebuild reproduce the panel artifacts it supersedes?

The panel run accumulated 301 labels on the 18,106-symbol challenge axis. The full run
accumulated ~18,294 labels on the 38,584-symbol all-symbols axis. Same corpus, same
revision, same code path -- so for the 301 labels they share, some fields must agree to the
last bit and others must NOT, and knowing which is which is the whole value of the check.

MUST MATCH (same cells, same arithmetic):
  count_sum      raw counts per gene; a wider axis adds columns, it does not change the ones
                 already there. Compared on the 18,106 challenge symbols, aligned BY SYMBOL
                 because the two axes differ in order as well as membership.
  n_cells        only if no cell was dropped for having zero counts on the narrower axis --
                 which the panel sidecar records as 0. Checked, not assumed.
  total_counts_sum, pct_mt_*, batch_cells   corpus columns, independent of the emitted axis.

MUST DIFFER (and a match would mean the axis flag did nothing):
  libsize_sum    the sum over the EMITTED axis. Wider axis, larger libsize.
  cpm_sum, cpm_sq_sum   per-cell CPM is counts/libsize, so a wider libsize rescales every
                 cell. This is the field the model consumes, which is why a full-corpus
                 artifact and a panel artifact must never be pooled as two sources.

Plus two internal checks on the full run that only an all-genes axis can pass:
  libsize_sum == total_counts_sum   every gene token maps to an emitted column, so the sum
                 over our axis IS the corpus's own per-cell total. On the challenge axis it
                 is ~71% of it.
  count_sum.sum(1) == libsize_sum   self-consistency of the two accumulators.

Run it after any rebuild of a derived pseudobulk, from the repo root:

    uv run python scripts/verify_derived_pseudobulk.py            # both contexts
    uv run python scripts/verify_derived_pseudobulk.py hct116     # one

It reads `~/data/sidechain/derived/xatlas-orion/<ctx>_{panel,full}.npz` and their `.qc.npz`
sidecars, prints PASS/FAIL per check, and exits non-zero if anything failed. Nothing is
written. **It needs the RAM to hold a full artifact** -- 16.9 GB of arrays for the 18,294 x
38,584 case, so run it on the box that built them, not on the Mac.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from sidechain.data.stream_pseudobulk import PseudobulkSums

DERIVED = Path.home() / "data" / "sidechain" / "derived" / "xatlas-orion"
FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  {'PASS' if ok else 'FAIL'}  {name}" + (f"   {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def verify(context: str) -> None:
    panel = PseudobulkSums.load(DERIVED / f"{context}_panel.npz")
    full = PseudobulkSums.load(DERIVED / f"{context}_full.npz")
    pqc = np.load(DERIVED / f"{context}_panel.qc.npz", allow_pickle=True)
    fqc = np.load(DERIVED / f"{context}_full.qc.npz", allow_pickle=True)

    print(f"\n=== {context} ===")
    print(f"  panel {len(panel.labels):>6} labels x {len(panel.genes):>6} genes")
    print(f"  full  {len(full.labels):>6} labels x {len(full.genes):>6} genes")

    # The n_cells comparison is only exact if the narrower axis dropped no cell for having
    # zero counts on it. Establish that before relying on it.
    dropped = int(pqc["cells_dropped_zero"])
    check("panel dropped no cell for zero on-axis counts", dropped == 0,
          f"cells_dropped_zero={dropped}")

    shared = [lab for lab in panel.labels if lab in set(full.labels)]
    check("every panel label is present in the full run",
          len(shared) == len(panel.labels),
          f"{len(shared)}/{len(panel.labels)}"
          + (f"  missing: {sorted(set(panel.labels) - set(full.labels))[:5]}"
             if len(shared) != len(panel.labels) else ""))
    if not shared:
        return

    pi = {lab: i for i, lab in enumerate(panel.labels)}
    fi = {lab: i for i, lab in enumerate(full.labels)}
    prows = np.array([pi[lab] for lab in shared])
    frows = np.array([fi[lab] for lab in shared])

    # Align genes BY SYMBOL: the panel axis is in challenge order, the full axis is sorted.
    fcol = {g: i for i, g in enumerate(full.genes)}
    keep = [i for i, g in enumerate(panel.genes) if g in fcol]
    check("every panel gene exists on the full axis", len(keep) == len(panel.genes),
          f"{len(keep)}/{len(panel.genes)}")
    pcols = np.array(keep)
    fcols = np.array([fcol[panel.genes[i]] for i in keep])

    check("count_sum is bit-identical on the shared labels and genes",
          np.array_equal(panel.count_sum[np.ix_(prows, pcols)],
                         full.count_sum[np.ix_(frows, fcols)]))
    check("n_cells is identical",
          np.array_equal(panel.n_cells[prows], full.n_cells[frows]))

    # The corpus's own per-cell columns cannot depend on which genes we emit.
    # Indexed through dicts, not `list.index`: the full sidecar carries 18,294 labels and a
    # linear scan per lookup is 334M string comparisons.
    prow_of = {str(lab): i for i, lab in enumerate(pqc["labels"])}
    frow_of = {str(lab): i for i, lab in enumerate(fqc["labels"])}
    pr = np.array([prow_of[lab] for lab in shared])
    fr = np.array([frow_of[lab] for lab in shared])
    for field in ("total_counts_sum", "pct_mt_sum", "pct_mt_sq_sum"):
        check(f"{field} is identical (a corpus column, not an axis one)",
              np.allclose(pqc[field][pr], fqc[field][fr], rtol=1e-12, atol=1e-6))

    # And the fields that MUST move, because the axis moved.
    wider = full.libsize_sum[frows] > panel.libsize_sum[prows]
    check("libsize_sum is strictly larger on the wider axis", bool(wider.all()),
          f"{int(wider.sum())}/{len(wider)} rows larger")
    cpm_same = np.allclose(panel.cpm_sum[np.ix_(prows, pcols)],
                           full.cpm_sum[np.ix_(frows, fcols)])
    check("cpm_sum DIFFERS (per-cell CPM is renormalised by the wider libsize)",
          not cpm_same,
          "identical -- the axis flag did nothing" if cpm_same else "")

    # Internal checks the all-genes run alone can pass. Tolerances are RELATIVE and loose
    # enough to absorb summation order: these are sums of millions of float64 values
    # accumulated in different orders, so bit-equality is not the right bar here (it is, and
    # is used, for count_sum above, where both runs add the same values in the same order).
    emitted = np.array([frow_of[lab] for lab in full.labels])
    check("full libsize_sum == the corpus's own total_counts_sum (every token maps)",
          np.allclose(full.libsize_sum, fqc["total_counts_sum"][emitted], rtol=1e-8, atol=1.0))
    check("full count_sum row sums == full libsize_sum",
          np.allclose(full.count_sum.sum(axis=1), full.libsize_sum, rtol=1e-8, atol=1.0))

    # The sidecar keeps every requested label; the emitted object prunes the empty ones.
    check("labels_in_corpus equals the label count (nothing was filtered out)",
          int(fqc["labels_in_corpus"]) == len(fqc["labels"]),
          f"in_corpus={int(fqc['labels_in_corpus'])}  requested={len(fqc['labels'])}"
          f"  emitted={len(full.labels)}")

    # The construct-keyed aggregate, if it is there: its rows must sum to the gene rows.
    guide_path = DERIVED / f"{context}_full.guide.npz"
    if guide_path.exists():
        guide = PseudobulkSums.load(guide_path)
        check("the construct aggregate shares the full run's gene axis",
              list(guide.genes) == list(full.genes))
        ci = guide.labels.index("Non-Targeting")
        check("the control is pooled identically under both keyings",
              np.array_equal(guide.count_sum[ci],
                             full.count_sum[full.labels.index("Non-Targeting")]))
        print(f"  ..    construct aggregate: {len(guide.labels)} labels, "
              f"{int(guide.n_cells.sum()):,} cells")


def coverage(contexts: list[str]) -> None:
    """The question the full-corpus run exists to answer.

    The panel-scope artifact covers 0 of the 300 Jurkat targets, 0 of 300 K562-essential and
    0 of 40 HepG2-flowtest -- so those folds could only ever test COVERAGE, never transfer.
    The full artifact has to cover them for the with/without-X-Atlas arms to mean anything;
    if it does not, there is no point buying GPU hours for them.
    """
    cache = Path.home() / "data" / "sidechain" / "cache" / "vcc2026"
    panels = ["loco_jurkat_panel.csv", "loco_k562_essential_panel.csv",
              "hepg2_flowtest_perts.csv", "loco_k562_gwps_challenge_panel.csv"]
    print("\n=== LOCO panel coverage ===")
    for ctx in contexts:
        path = DERIVED / f"{ctx}_full.npz"
        if not path.exists():
            print(f"  {ctx}: {path.name} not present"); continue
        labels = set(PseudobulkSums.load(path).labels)
        print(f"  {ctx}  ({len(labels):,} labels)")
        for name in panels:
            csv = cache / name
            if not csv.exists():
                print(f"    {name:40} MISSING"); continue
            import pandas as pd
            df = pd.read_csv(csv)
            col = "target_gene" if "target_gene" in df.columns else df.columns[0]
            want = set(df[col].astype(str))
            hit = len(want & labels)
            print(f"    {name:40} {hit:>4}/{len(want):<4}"
                  + ("" if hit == len(want)
                     else f"   missing e.g. {sorted(want - labels)[:4]}"))


if __name__ == "__main__":
    contexts = sys.argv[1:] or ["hct116", "hek293t"]
    for ctx in contexts:
        verify(ctx)
    coverage(contexts)
    print(f"\n{len(FAILURES)} failure(s)" + (f": {FAILURES}" if FAILURES else ""))
    sys.exit(1 if FAILURES else 0)

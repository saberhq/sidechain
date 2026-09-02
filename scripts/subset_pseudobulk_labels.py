"""Row-subset a full-corpus PseudobulkSums to a label list, one array at a time.

    uv run python scripts/subset_pseudobulk_labels.py IN.npz OUT.npz --keep panel.csv \
        --control Non-Targeting

Why this exists. A full-corpus X-Atlas artifact is 18,294 labels x 38,584 genes, and
`PseudobulkSums` holds three float64 matrices of that shape -- 16.9 GB. Two of them will not
sit in a 16 GB Mac, which is why full-corpus submission builds have been box-only.

But **`pooled_delta` reads sources strictly row-wise**: `_log2fc_with_var` touches exactly two
rows (the target and the control), `control_cpm` one, `n_eff` two. Nothing in the pooling path
ever needs a row for a label it is not asked about. So a subset restricted to the labels a run
will actually request is **bit-identical for pooling** and three orders of magnitude smaller.
That is what the existing `foldsub/*_fold272.npz` artifacts are, and this script is the
missing shipped version of how they were made.

It never materialises the whole artifact: `np.load` on an `.npz` is lazy, so each array is
decompressed, sliced and released in turn. Peak is one array (5.6 GB) rather than three.

**This subsets LABELS (rows: which perturbation), never GENES (columns: which expression).**
Narrowing the gene axis changes the CPM denominator, because library size is summed over the
emitted axis -- the trap that made the panel-scope artifacts unpoolable with the full ones
(`derived/xatlas-orion/archive_panel_20260823/README.md`). The gene axis is copied verbatim.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROW_ARRAYS = ("count_sum", "cpm_sum", "cpm_sq_sum", "n_cells", "libsize_sum")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("src", type=Path)
    ap.add_argument("dest", type=Path)
    ap.add_argument("--keep", required=True,
                    help="CSV of labels to keep (column target_gene, or the first column)")
    ap.add_argument("--control", required=True, help="control label, always kept")
    args = ap.parse_args(argv)

    df = pd.read_csv(args.keep)
    col = "target_gene" if "target_gene" in df.columns else df.columns[0]
    want = list(dict.fromkeys(df[col].astype(str).tolist()))

    with np.load(args.src.expanduser(), allow_pickle=True) as z:
        labels = [str(x) for x in z["labels"]]
        pos = {lab: i for i, lab in enumerate(labels)}
        if args.control not in pos:
            raise SystemExit(
                f"control {args.control!r} is not a label in {args.src}. Controls are matched "
                f"EXACTLY, never by substring (ingest/checks.py); have e.g. {labels[:3]}"
            )
        keep_names = [args.control] + [w for w in want if w in pos and w != args.control]
        missing = [w for w in want if w not in pos]
        idx = np.array([pos[n] for n in keep_names], dtype=np.int64)

        out: dict[str, np.ndarray] = {
            "labels": np.array(keep_names, dtype=object),
            "genes": z["genes"],                       # verbatim: never subset the gene axis
        }
        if "sources" in z.files:
            out["sources"] = z["sources"]
        for name in ROW_ARRAYS:
            arr = z[name]                              # decompress one array...
            out[name] = arr[idx].copy()                # ...slice...
            del arr                                    # ...and release before the next
        extra = [f for f in z.files if f not in out and f not in ROW_ARRAYS]
        for name in extra:
            arr = z[name]
            out[name] = arr[idx].copy() if arr.ndim and arr.shape[0] == len(labels) else arr
            del arr

    args.dest.expanduser().parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.dest.expanduser(), **out)
    print(f"{args.src.name}: {len(labels)} labels -> {len(keep_names)} "
          f"({len(missing)} requested labels absent), genes {len(out['genes'])} unchanged")
    if missing:
        print(f"  absent: {missing[:8]}{'...' if len(missing) > 8 else ''}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

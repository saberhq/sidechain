#!/usr/bin/env python
"""Build a perturbation feature table for ``state tx train``.

``cell_load`` accepts a ``perturbation_features_file``: a ``torch.save``d dict mapping a
perturbation label to a feature vector. When it is set, that dict **replaces** the default
one-hot map entirely, which is what lets a model represent a gene it was never trained on.

The vectors come from ``arcinstitute/SE-600M/protein_embeddings.pt`` -- 19,790 HGNC symbols
x 5,120 dims, ESM2. Arc ships it beside the SE weights and its own transition configs leave
the slot ``null``.

Two hazards this script exists to remove, both of which fail silently otherwise:

1. **cell_load zero-fills.** A perturbation present in the training data but absent from the
   dict is set to a zero vector with a single ``INFO`` log line
   (``cell_load/data_modules/perturbation_dataloader.py``). Several hundred genes silently
   collapsing onto one vector is not a failure anyone notices from a loss curve. This script
   therefore refuses to write unless every requested label resolves, unless ``--allow-missing``
   is passed explicitly.
2. **The keys are HGNC symbols, not Ensembl.** That is a deliberate exception to the project's
   Ensembl rule -- ESM2 was computed against a symbol-keyed proteome, so the symbol is the join
   key upstream. It is contained by doing the bridge once, here, with the assert above and with
   the shared retired-alias table (``src/sidechain/data/gene_aliases.py``).

Usage::

    python scripts/build_pert_features.py \
        --embeddings ~/data/sidechain/external/hf-arcinstitute-SE-600M/protein_embeddings.pt \
        --labels     ~/data/sidechain/vcc2026/panels_union.csv \
        --labels     ~/data/sidechain/vcc2026/pert_counts.csv \
        --out        ~/data/sidechain/cache/vcc2026/pert_features_esm2.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Retired HGNC symbols seen in our corpora, old -> current. The table lives in
# ``src/sidechain/data/gene_aliases.py`` -- ONE source of truth, shared with
# ``scripts/esm2_geometry_gate.py``; its docstring carries the authority (HGNC), the verification
# date and the evidence file. Extend it there, never here.
from sidechain.data.gene_aliases import RETIRED_SYMBOLS as ALIAS

RETIRED_ALIASES: dict[str, str] = ALIAS      # the name this script has always used; same object

# Control labels are never perturbations. ``cell_load`` addresses the control arm through
# ``control_pert``, not through the feature dict, so a control label appearing here would be a
# bug rather than a gap. Matched case-insensitively.
CONTROL_LABELS ={"non-targeting", "nontargeting", "ntc", "control", "unassigned", "dmso_tf"}


def read_labels(paths: list[Path]) -> list[str]:
    """Read perturbation labels from CSV files (one column, or a ``target_gene``/``gene`` column)."""
    import pandas as pd

    out: list[str] = []
    for p in paths:
        df = pd.read_csv(p)
        for col in ("target_gene", "gene", "gene_target", "perturbation"):
            if col in df.columns:
                out.extend(df[col].astype(str).tolist())
                break
        else:
            if df.shape[1] != 1:
                raise SystemExit(
                    f"{p}: no target_gene/gene/gene_target/perturbation column and "
                    f"{df.shape[1]} columns, so the label column is ambiguous."
                )
            out.extend(df.iloc[:, 0].astype(str).tolist())
    return out


def read_labels_from_h5ad(paths: list[Path], columns: tuple[str, ...]) -> list[str]:
    """Read perturbation label *categories* from h5ad obs without touching the count matrix."""
    import h5py

    out: list[str] = []
    for p in paths:
        with h5py.File(p) as f:
            obs = f["obs"]
            for col in columns:
                if col in obs:
                    g = obs[col]
                    if hasattr(g, "keys") and "categories" in g:
                        out.extend(
                            x.decode() if isinstance(x, bytes) else str(x)
                            for x in g["categories"][:]
                        )
                    break
            else:
                raise SystemExit(f"{p}: none of {columns} present in obs.")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--embeddings", required=True, type=Path, help="protein_embeddings.pt from SE-600M")
    ap.add_argument("--labels", action="append", default=[], type=Path, help="CSV of perturbation labels (repeatable)")
    ap.add_argument("--labels-h5ad", action="append", default=[], type=Path, help="h5ad whose obs holds the labels (repeatable)")
    ap.add_argument(
        "--label-column",
        action="append",
        default=[],
        help="obs column(s) to try for --labels-h5ad, in order (default: gene_target, gene, perturbation)",
    )
    ap.add_argument("--out", required=True, type=Path, help="destination .pt")
    ap.add_argument(
        "--control-label",
        action="append",
        default=[],
        help="write an explicit ALL-ZERO vector for this label (repeatable). The control arm needs "
        "a perturbation vector like any other row, and zero is the right one for a residual model: "
        "no perturbation, no shift. Declaring it here matters because otherwise cell_load backfills "
        "it with the same silent zero-fill it uses for genuine coverage gaps, and the two become "
        "indistinguishable in the log.",
    )
    ap.add_argument(
        "--allow-missing",
        action="store_true",
        help="write anyway when some labels do not resolve. cell_load will ZERO-FILL them at train "
        "time, which is silent and usually wrong -- pass this only when you have read the miss list.",
    )
    args = ap.parse_args()

    import torch

    if not args.labels and not args.labels_h5ad:
        raise SystemExit("Give at least one --labels or --labels-h5ad.")

    table = torch.load(args.embeddings.expanduser(), weights_only=False, map_location="cpu")
    if not isinstance(table, dict):
        raise SystemExit(f"{args.embeddings}: expected a dict, got {type(table)}")
    dim = next(iter(table.values())).shape[-1]

    cols = tuple(args.label_column) or ("gene_target", "gene", "perturbation")
    raw = read_labels([p.expanduser() for p in args.labels])
    raw += read_labels_from_h5ad([p.expanduser() for p in args.labels_h5ad], cols)

    wanted = {s for s in raw if s and s.lower() not in CONTROL_LABELS}

    features: dict[str, "torch.Tensor"] = {}
    aliased: dict[str, str] = {}
    missing: list[str] = []
    for sym in sorted(wanted):
        if sym in table:
            features[sym] = table[sym]
            continue
        alias = RETIRED_ALIASES.get(sym)
        if alias is not None and alias in table:
            # Key by the label as it appears in OUR data -- cell_load looks the label up verbatim.
            features[sym] = table[alias]
            aliased[sym] = alias
            continue
        missing.append(sym)

    print(f"embeddings   : {args.embeddings} ({len(table):,} symbols x {dim})")
    print(f"labels wanted: {len(wanted):,} (controls excluded)")
    print(f"resolved     : {len(features):,}  direct {len(features) - len(aliased):,}, via alias {len(aliased):,}")
    if aliased:
        print("  aliases    : " + ", ".join(f"{k}->{v}" for k, v in sorted(aliased.items())))
    print(f"missing      : {len(missing):,}")
    if missing:
        print("  " + ", ".join(missing[:40]) + (" ..." if len(missing) > 40 else ""))

    if missing and not args.allow_missing:
        print(
            "\nREFUSING TO WRITE. cell_load fills an unresolved perturbation with a ZERO VECTOR and "
            "logs one INFO line, so these genes would train as if they were all the same gene.\n"
            "Fix the labels, extend src/sidechain/data/gene_aliases.py, or pass --allow-missing "
            "having read the list.",
            file=sys.stderr,
        )
        return 1

    for label in args.control_label:
        features[label] = torch.zeros(dim)
    if args.control_label:
        print(f"controls     : {len(args.control_label)} explicit zero vector(s) — "
              + ", ".join(repr(c) for c in args.control_label))

    out = args.out.expanduser()
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(features, out)

    sidecar = out.with_suffix(out.suffix + ".json")
    sidecar.write_text(
        json.dumps(
            {
                "embeddings_source": str(args.embeddings),
                "n_symbols_in_source": len(table),
                "feature_dim": int(dim),
                "n_features_written": len(features),
                "n_via_alias": len(aliased),
                "aliases": aliased,
                "n_controls_zeroed": len(args.control_label),
                "control_labels": list(args.control_label),
                "n_missing": len(missing),
                "missing": missing,
                "label_sources": [str(p) for p in args.labels] + [str(p) for p in args.labels_h5ad],
            },
            indent=2,
        )
        + "\n"
    )
    print(f"\nwrote {out} ({len(features):,} x {dim})")
    print(f"wrote {sidecar}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

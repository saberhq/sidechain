#!/usr/bin/env python
"""Build a perturbation feature table for ``state tx train``.

``cell_load`` accepts a ``perturbation_features_file``: a ``torch.save``d dict mapping a
perturbation label to a feature vector. When it is set, that dict **replaces** the default
one-hot map entirely, which is what lets a model represent a gene it was never trained on.

The vectors come, since 2026-09-16 (T90), from the ESM-2 3B gene table shipped inside
TranscriptFormer's ``tf_sapiens`` checkpoint -- 18,618 HGNC symbols x 2,560 dims, at
``~/data/sidechain/derived/transcriptformer-esm2-3b/esm2_3b_raw_table.pt`` (``LINEAGE.json``
beside it carries the bytes' origin and sha256). Before that they came from
``arcinstitute/SE-600M/protein_embeddings.pt`` -- 19,790 symbols x 5,120 dims, ESM-2 15B --
which Arc ships beside the SE weights and its own transition configs leave ``null``; pass it
with ``--embeddings`` to build against it again. Why the swap: on the T89 paired re-read of the
42-pair geometry sweep the 3B table ties Arc's on the deployment-shaped X-Atlas pairs, leads on
the four other lines and loses on two K562-ess pairs, at half the width, and no scored entry
rides on the featurizer (``private/research/ideas/pretrained-scfm-arms.md`` § Outcome T89;
Saber's call on A129). Both tables resolve every one of the 300 challenge targets and the
849-target panel union.

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
        --labels     ~/data/sidechain/vcc2026/panels_union.csv \
        --labels     ~/data/sidechain/vcc2026/pert_counts.csv \
        --out        ~/data/sidechain/cache/vcc2026/pert_features_esm2_3b.pt

    # the pre-T90 table, explicitly:
    python scripts/build_pert_features.py \
        --embeddings ~/data/sidechain/external/hf-arcinstitute-SE-600M/protein_embeddings.pt \
        --labels ... --out ~/data/sidechain/cache/vcc2026/pert_features_esm2.pt

The sidecar ``<out>.json`` records the table's path and sha256, so a feature file can always be
traced to the bytes it was built from.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

# The featurizer table: ESM-2 3B from TranscriptFormer's checkpoint (T90, 2026-09-16). One place,
# so a build without --embeddings and the tests agree on what "the default table" is.
DEFAULT_EMBEDDINGS = Path("~/data/sidechain/derived/transcriptformer-esm2-3b/esm2_3b_raw_table.pt")
DEFAULT_EMBEDDINGS_SHA256 = "429d2cfbed0cd817d8eae78efb8cba590137d0dcc2dce84c844fea85b7056328"

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
    ap.add_argument("--embeddings", type=Path, default=DEFAULT_EMBEDDINGS,
                    help="a {symbol: vector} .pt table; default the ESM-2 3B table at "
                         f"{DEFAULT_EMBEDDINGS} (T90). Arc's SE-600M protein_embeddings.pt is the "
                         "pre-T90 choice and still works here.")
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

    emb_path = args.embeddings.expanduser()
    if not emb_path.exists():
        raise SystemExit(f"embedding table not found at {emb_path}"
                         + (" -- pull it with scripts/lamin_pull.py (key derived/transcriptformer-"
                            "esm2-3b/esm2_3b_raw_table.pt) or pass --embeddings"
                            if emb_path == DEFAULT_EMBEDDINGS.expanduser() else ""))
    table = torch.load(emb_path, weights_only=False, map_location="cpu")
    if not isinstance(table, dict):
        raise SystemExit(f"{args.embeddings}: expected a dict, got {type(table)}")
    dim = next(iter(table.values())).shape[-1]
    emb_sha = hashlib.sha256(emb_path.read_bytes()).hexdigest()

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

    print(f"embeddings   : {emb_path} ({len(table):,} symbols x {dim}; sha256 {emb_sha[:12]}...)")
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
                "embeddings_source": str(emb_path),
                "embeddings_sha256": emb_sha,
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

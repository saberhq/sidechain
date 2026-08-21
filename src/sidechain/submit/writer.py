"""Out-of-core writer for a prediction h5ad, plus the `.vcc` packer.

Mirrors what `vcc prep` emits (read from its source, vcc-cli 0.1.0):
  X    csr_matrix group, float32 data, int32 indices, raw integer counts
  obs  index "0".."N-1" (strings); columns `target_gene` (symbols), `context`
  var  index = the 18,533 symbols in gene_names.csv order, no columns
and the contract it enforces: every listed perturbation in every context with
exactly `cells_per_pert` cells, no control rows, non-negative finite integral
values, <= max_counts_per_cell per row, <= max_stored_entries overall, no
explicitly stored zeros.

`vcc submit x.vcc` validates the CONTAINER only and the server reads the file
on a 128 GB machine, so a file written here never needs to fit in local RAM.
`verify_h5ad` re-reads only metadata (obs, var, indptr) and re-derives every
check that does not need the values -- run it before packing, every time.
"""
from __future__ import annotations

import os
import tarfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Self

import h5py
import numpy as np
import pandas as pd
import scipy.sparse as sp

try:  # anndata >= 0.11
    from anndata.io import read_elem, write_elem
except ImportError:  # pragma: no cover
    from anndata.experimental import read_elem, write_elem

PRED_MEMBER = "pred.h5ad.zst"          # same constant as vcc.vccfile
CELLS_PER_PERT = 400
MAX_COUNTS_PER_CELL = 1_000_000
MAX_CELLS = 400_000
MAX_STORED_ENTRIES = 4_750_000_000


class ContractError(ValueError):
    """The block or file violates the submission contract."""


@dataclass
class Contract:
    genes: list[str]
    perturbations: list[str]
    contexts: list[str]
    cells_per_pert: int = CELLS_PER_PERT
    pert_col: str = "target_gene"
    context_col: str = "context"
    control_label: str = "non-targeting"
    max_counts_per_cell: int = MAX_COUNTS_PER_CELL
    max_cells: int = MAX_CELLS
    max_stored_entries: int = MAX_STORED_ENTRIES

    def __post_init__(self) -> None:
        if len(set(self.genes)) != len(self.genes):
            raise ContractError("gene list has duplicates")
        if len(set(self.perturbations)) != len(self.perturbations):
            raise ContractError("perturbation list has duplicates")
        if self.control_label in self.perturbations:
            raise ContractError("control label must not be in the perturbation list")
        if len(self.contexts) * len(self.perturbations) * self.cells_per_pert > self.max_cells:
            raise ContractError("contract exceeds the total cell cap")


@dataclass
class WriteSummary:
    path: str
    n_cells: int
    n_genes: int
    nnz: int
    cells_per_block: dict = field(default_factory=dict)
    max_row_total: float = 0.0
    nnz_per_cell_median: float = 0.0


class SubmissionWriter:
    """Append (context, perturbation) blocks of integer counts to an h5ad on disk."""

    def __init__(self, path: str | Path, contract: Contract, *, compression: str | None = "gzip",
                 compression_opts: int | None = 4, chunk: int = 1 << 20):
        self.path = Path(path).expanduser()
        self.c = contract
        self.G = len(contract.genes)
        self._f = h5py.File(self.path, "w")
        self._f.attrs["encoding-type"] = "anndata"
        self._f.attrs["encoding-version"] = "0.1.0"
        X = self._f.create_group("X")
        X.attrs["encoding-type"] = "csr_matrix"
        X.attrs["encoding-version"] = "0.1.0"
        X.attrs["shape"] = np.array([0, self.G], dtype=np.int64)
        kw = {"chunks": (chunk,), "maxshape": (None,), "compression": compression, "compression_opts": compression_opts}
        self._data = X.create_dataset("data", shape=(0,), dtype=np.float32, **kw)
        self._indices = X.create_dataset("indices", shape=(0,), dtype=np.int32, **kw)
        self._indptr = X.create_dataset("indptr", shape=(1,), dtype=np.int64, maxshape=(None,),
                                        chunks=(1 << 16,), compression=compression, compression_opts=compression_opts)
        self._indptr[0] = 0
        self._nnz = 0
        self._n = 0
        self._perts: list[str] = []
        self._ctx: list[str] = []
        self._counts: Counter = Counter()
        self._max_row = 0.0
        self._nnz_rows: list[int] = []
        self._closed = False

    # -- blocks -------------------------------------------------------------
    def add_block(self, X, context: str, perturbation: str) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        if context not in self.c.contexts:
            raise ContractError(f"unknown context {context!r}")
        if perturbation not in self.c.perturbations:
            raise ContractError(f"{perturbation!r} is not in the perturbation list")
        M = sp.csr_matrix(X)
        if M.shape[1] != self.G:
            raise ContractError(f"block has {M.shape[1]} genes, contract has {self.G}")
        M.eliminate_zeros()
        M.sort_indices()
        d = M.data
        if d.size:
            if not np.all(np.isfinite(d)):
                raise ContractError("non-finite values")
            if d.min() < 0:
                raise ContractError("negative values")
            if not np.array_equal(d, np.round(d)):
                raise ContractError("fractional values -- submit raw integer counts")
        rows = np.asarray(M.sum(axis=1)).ravel()
        if rows.size and rows.max() > self.c.max_counts_per_cell:
            raise ContractError(f"a cell totals {rows.max():.0f} > {self.c.max_counts_per_cell}")
        n = M.shape[0]
        if self._n + n > self.c.max_cells:
            raise ContractError("total cell cap exceeded")
        if self._nnz + M.nnz > self.c.max_stored_entries:
            raise ContractError("density cap exceeded")
        # append
        self._data.resize((self._nnz + M.nnz,)); self._data[self._nnz:] = d.astype(np.float32)
        self._indices.resize((self._nnz + M.nnz,)); self._indices[self._nnz:] = M.indices.astype(np.int32)
        self._indptr.resize((self._n + n + 1,)); self._indptr[self._n + 1:] = M.indptr[1:].astype(np.int64) + self._nnz
        self._nnz += int(M.nnz)
        self._n += n
        self._perts.extend([perturbation] * n)
        self._ctx.extend([context] * n)
        self._counts[(context, perturbation)] += n
        self._max_row = max(self._max_row, float(rows.max()) if rows.size else 0.0)
        self._nnz_rows.extend(np.diff(M.indptr).tolist())

    # -- finish -------------------------------------------------------------
    def close(self, *, strict: bool = True) -> WriteSummary:
        if self._closed:
            raise RuntimeError("already closed")
        self._f["X"].attrs["shape"] = np.array([self._n, self.G], dtype=np.int64)
        obs = pd.DataFrame(
            {self.c.pert_col: np.asarray(self._perts, dtype=object),
             self.c.context_col: np.asarray(self._ctx, dtype=object)},
            index=np.arange(self._n).astype(str),
        )
        var = pd.DataFrame(index=pd.Index(self.c.genes, dtype=object))
        write_elem(self._f, "obs", obs)
        write_elem(self._f, "var", var)
        self._f.close()
        self._closed = True
        if strict:
            expected = {(c, p) for c in self.c.contexts for p in self.c.perturbations}
            missing = expected - set(self._counts)
            extra = set(self._counts) - expected
            wrong = {k: v for k, v in self._counts.items() if v != self.c.cells_per_pert}
            if missing or extra or wrong:
                raise ContractError(
                    f"incomplete submission: {len(missing)} (context, pert) blocks missing, "
                    f"{len(extra)} unexpected, {len(wrong)} with != {self.c.cells_per_pert} cells "
                    f"(e.g. {list(wrong.items())[:3]})"
                )
        return WriteSummary(
            path=str(self.path), n_cells=self._n, n_genes=self.G, nnz=self._nnz,
            cells_per_block=dict(self._counts), max_row_total=self._max_row,
            nnz_per_cell_median=float(np.median(self._nnz_rows)) if self._nnz_rows else 0.0,
        )

    def __enter__(self) -> Self:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        if not self._closed:
            if exc_type is None:
                self.close()
            else:
                self._f.close()
                self._closed = True


# -- verification from metadata only -------------------------------------------

def verify_h5ad(path: str | Path, contract: Contract) -> dict:
    """Re-derive every contract check that needs no values: obs, var, indptr, nnz.

    Values (non-negative, integral) are asserted at write time by add_block;
    this is the independent read-back that the file on disk says what the
    writer thinks it said.
    """
    path = Path(path).expanduser()
    with h5py.File(path, "r") as f:
        obs = read_elem(f["obs"])
        var = read_elem(f["var"])
        X = f["X"]
        shape = tuple(int(v) for v in X.attrs["shape"])
        indptr = X["indptr"][:]
        nnz = int(X["data"].shape[0])
        enc = X.attrs.get("encoding-type", "")
        enc = enc.decode() if isinstance(enc, bytes) else str(enc)
    problems = []
    if enc != "csr_matrix":
        problems.append(f"X encoding {enc!r}")
    if list(var.index.astype(str)) != list(contract.genes):
        problems.append("var_names differ from the gene list (set or order)")
    if shape != (len(obs), len(contract.genes)):
        problems.append(f"X shape {shape} vs obs {len(obs)} x genes {len(contract.genes)}")
    if int(indptr[-1]) != nnz or len(indptr) != shape[0] + 1:
        problems.append("indptr inconsistent with data length / row count")
    if contract.pert_col not in obs or contract.context_col not in obs:
        problems.append(f"obs must carry {contract.pert_col} and {contract.context_col}")
    else:
        perts = obs[contract.pert_col].astype(str)
        ctx = obs[contract.context_col].astype(str)
        if (perts == contract.control_label).any():
            problems.append("control rows present")
        counts = Counter(zip(ctx, perts))
        expected = {(c, p) for c in contract.contexts for p in contract.perturbations}
        if set(counts) != expected:
            problems.append(f"(context, pert) set differs: {len(set(counts) - expected)} extra, "
                            f"{len(expected - set(counts))} missing")
        bad = [k for k, v in counts.items() if v != contract.cells_per_pert]
        if bad:
            problems.append(f"{len(bad)} blocks without exactly {contract.cells_per_pert} cells")
    if shape[0] > contract.max_cells:
        problems.append("over the cell cap")
    if nnz > contract.max_stored_entries:
        problems.append("over the density cap")
    if problems:
        raise ContractError("; ".join(problems))
    return {"n_cells": shape[0], "n_genes": shape[1], "nnz": nnz,
            "nnz_per_cell_mean": nnz / max(shape[0], 1), "file_gb": round(path.stat().st_size / 1e9, 2)}


# -- packing -----------------------------------------------------------------------

def pack_vcc(h5ad_path: str | Path, vcc_path: str | Path, *, level: int = 3) -> Path:
    """tar( pred.h5ad.zst ) with normalized member metadata, exactly as `vcc prep` writes it."""
    import zstandard as zstd

    h5ad_path, vcc_path = Path(h5ad_path).expanduser(), Path(vcc_path).expanduser()
    vcc_path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=vcc_path.parent) as tmp:
        zst_path = os.path.join(tmp, PRED_MEMBER)
        cctx = zstd.ZstdCompressor(level=level, threads=-1)
        with open(h5ad_path, "rb") as src, open(zst_path, "wb") as dst:
            cctx.copy_stream(src, dst)

        def _normalize(ti: tarfile.TarInfo) -> tarfile.TarInfo:
            ti.uid = ti.gid = 0
            ti.uname = ti.gname = ""
            ti.mtime = 0
            return ti

        with tarfile.open(vcc_path, "w") as tar:
            tar.add(zst_path, arcname=PRED_MEMBER, filter=_normalize)
    return vcc_path

"""Write and read an `.h5ad` CSR matrix without ever holding it.

The 2026 folds are the reason. `loco_hct116_real.h5ad` is 142,046 cells x 38,584
genes with 726 M nonzeros; a prediction for its 830 perturbations is another
803 M. Either one is ~6 GB of `data` + `indices` in memory, and `scipy.sparse.vstack`
and `anndata.concat` both peak at twice their result -- so the obvious in-memory
build of a prediction needs ~13 GB and the concat that appends the real controls
needs ~12 GB more. On a 17 GB Mac that is not a slow run, it is a dead one, and it
is why every X-Atlas-scale mirror arm to date was built on the box.

Appending row blocks to HDF5 by hand costs one block of memory instead. These
helpers were written for `scripts/emit_tx_prediction.py` (which walks a STATE
checkpoint over windows of control cells) and moved here when
`sidechain.eval.loco` needed exactly the same thing; that script imports them
back, so there is still one implementation.

`write_frame` writes AnnData's `dataframe` encoding by hand, categorical columns
included. The metadata attributes are not decoration: without them anndata still
reads the file, but through its legacy path and with an `OldFormatWarning`.
"""
from __future__ import annotations

import h5py
import numpy as np

__all__ = ["CsrWriter", "load_rows_csr", "open_anndata_h5", "read_categorical", "write_frame"]


def open_anndata_h5(path, mode: str = "w", *, rdcc_nbytes: int = 64 << 20) -> h5py.File:
    """An h5ad-shaped HDF5 file with a chunk cache big enough for row-wise reads.

    h5py's default cache is 1 MiB -- about five chunks of a gzip-compressed h5ad --
    so reading scattered rows in ascending order re-decompresses the same chunk
    dozens of times. 64 MiB makes a full pass over a 700 M-nonzero fold cost one
    decompression per chunk.
    """
    f = h5py.File(path, mode, rdcc_nbytes=rdcc_nbytes)
    if mode == "w":
        f.attrs["encoding-type"] = "anndata"
        f.attrs["encoding-version"] = "0.1.0"
    return f


def read_categorical(f: h5py.File, col: str) -> tuple[np.ndarray, np.ndarray]:
    """(categories, codes) for one categorical obs column."""
    return f[f"obs/{col}/categories"][:].astype(str), f[f"obs/{col}/codes"][:]


def load_rows_csr(f: h5py.File, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """CSR triple (data, indices, indptr) for `rows` of an h5ad CSR group, in the given order.

    Rows that are already consecutive in the file are fetched as ONE hyperslab, and that is
    not a micro-optimisation. An h5ad `X` written by `CsrWriter` uses gzip chunks of 2^20
    elements -- 4 MB -- while h5py's default chunk cache is 1 MB, so it cannot hold even one.
    Read row by row, every row of a 6,600-nonzero cell re-inflates the whole 4 MB chunk it
    sits in: copying a 122,046-cell prediction that way decompressed ~480 GB and had not
    finished a quarter of the file in six minutes. Whole-span reads take one pass.
    """
    g = f["X"]
    indptr = g["indptr"][:]
    rows = np.asarray(rows)
    spans = [(int(indptr[r]), int(indptr[r + 1])) for r in rows]
    total = sum(hi - lo for lo, hi in spans)
    data = np.empty(total, dtype=np.float32)
    indices = np.empty(total, dtype=np.int32)
    out_ptr = np.zeros(len(rows) + 1, dtype=np.int64)
    # split `rows` into maximal consecutive runs; a run's rows are contiguous in the file too
    breaks = np.flatnonzero(np.diff(rows) != 1) + 1 if len(rows) > 1 else np.array([], dtype=int)
    at = 0
    for run in np.split(np.arange(len(rows)), breaks):
        if not len(run):
            continue
        lo, hi = spans[run[0]][0], spans[run[-1]][1]
        if hi > lo:
            data[at:at + hi - lo] = g["data"][lo:hi]
            indices[at:at + hi - lo] = g["indices"][lo:hi]
        for i in run:
            at += spans[i][1] - spans[i][0]
            out_ptr[i + 1] = at
    return data, indices, out_ptr


class CsrWriter:
    """Append CSR row blocks to an h5ad-shaped `X` group without holding the matrix."""

    def __init__(self, h5: h5py.File, n_genes: int, chunk: int = 1 << 20):
        g = h5.create_group("X")
        g.attrs["encoding-type"] = "csr_matrix"
        g.attrs["encoding-version"] = "0.1.0"
        self.g = g
        self.n_genes = n_genes
        self.data = g.create_dataset("data", (0,), maxshape=(None,), dtype="float32",
                                     chunks=(chunk,), compression="gzip", compression_opts=1)
        self.indices = g.create_dataset("indices", (0,), maxshape=(None,), dtype="int32",
                                        chunks=(chunk,), compression="gzip", compression_opts=1)
        self.indptr: list[int] = [0]
        self.nnz = 0

    def append(self, data: np.ndarray, indices: np.ndarray, indptr: np.ndarray) -> None:
        n = len(data)
        if n:
            self.data.resize((self.nnz + n,))
            self.data[self.nnz:] = data
            self.indices.resize((self.nnz + n,))
            self.indices[self.nnz:] = indices
        base = self.indptr[-1]
        self.indptr.extend((base + indptr[1:]).tolist())
        self.nnz += n

    def append_csr(self, block) -> None:
        """Append a `scipy.sparse.csr_matrix` block (its own dtypes are cast to the file's)."""
        self.append(block.data.astype(np.float32, copy=False),
                    block.indices.astype(np.int32, copy=False), block.indptr)

    def close(self) -> int:
        n_rows = len(self.indptr) - 1
        ptr = np.asarray(self.indptr, dtype=np.int64)
        dtype = "int32" if ptr[-1] < np.iinfo(np.int32).max else "int64"
        self.g.create_dataset("indptr", data=ptr.astype(dtype), compression="gzip", compression_opts=1)
        self.g.attrs["shape"] = np.array([n_rows, self.n_genes], dtype="int64")
        return n_rows


def write_frame(h5: h5py.File, name: str, index: np.ndarray, columns: dict[str, np.ndarray]) -> None:
    """An AnnData `dataframe` group with a string index and categorical string columns."""
    g = h5.create_group(name)
    g.attrs["encoding-type"] = "dataframe"
    g.attrs["encoding-version"] = "0.2.0"
    g.attrs["_index"] = "_index"
    g.attrs["column-order"] = np.array(list(columns), dtype=h5py.string_dtype(encoding="utf-8"))
    _string_array(g, "_index", index)
    for col, values in columns.items():
        cats, codes = np.unique(np.asarray(values, dtype=object), return_inverse=True)
        sub = g.create_group(col)
        sub.attrs["encoding-type"] = "categorical"
        sub.attrs["encoding-version"] = "0.2.0"
        sub.attrs["ordered"] = False
        _string_array(sub, "categories", cats)
        d = sub.create_dataset("codes", data=codes.astype("int32"), compression="gzip",
                               compression_opts=1)
        d.attrs["encoding-type"] = "array"
        d.attrs["encoding-version"] = "0.2.0"


def _string_array(g: h5py.Group, name: str, values: np.ndarray) -> None:
    """A string dataset carrying AnnData's encoding metadata.

    Without the two attributes anndata still reads it, but through its legacy path and with an
    `OldFormatWarning` -- a route a later release is free to drop. The file is written by hand
    here (holding the matrix would cost gigabytes), so the metadata has to be written by hand too.
    """
    d = g.create_dataset(name, data=np.asarray(values, dtype=object),
                         dtype=h5py.string_dtype(encoding="utf-8"),
                         compression="gzip", compression_opts=1)
    d.attrs["encoding-type"] = "string-array"
    d.attrs["encoding-version"] = "0.2.0"

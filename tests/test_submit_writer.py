"""The out-of-core writer must produce a file anndata reads back identically,
enforce the contract at write time, refuse incomplete files, and pack a .vcc
with the member vcc expects."""
import tarfile

import anndata as ad
import numpy as np
import pytest
import scipy.sparse as sp

from sidechain.submit.writer import (
    Contract,
    ContractError,
    SubmissionWriter,
    pack_vcc,
    verify_h5ad,
)

GENES = [f"G{i}" for i in range(30)]
PERTS = ["ADNP", "ACLY", "AGO1"]
CTX = ["A", "B"]


def _contract(n=5):
    return Contract(genes=GENES, perturbations=PERTS, contexts=CTX, cells_per_pert=n)


def _block(rng, n=5, g=30):
    return sp.csr_matrix(rng.poisson(0.7, size=(n, g)).astype(np.float32))


def test_roundtrip_matches_blocks_and_layout(tmp_path):
    rng = np.random.default_rng(0)
    p = tmp_path / "pred.h5ad"
    blocks = {}
    with SubmissionWriter(p, _contract()) as w:
        for c in CTX:
            for t in PERTS:
                b = _block(rng)
                b.eliminate_zeros()
                blocks[(c, t)] = b
                w.add_block(b, c, t)
    a = ad.read_h5ad(p)
    assert a.shape == (len(CTX) * len(PERTS) * 5, 30)
    assert list(a.var_names) == GENES and list(a.var.columns) == []
    assert list(a.obs.columns) == ["target_gene", "context"]
    assert list(a.obs_names[:3]) == ["0", "1", "2"]
    assert sp.isspmatrix_csr(a.X) and a.X.dtype == np.float32
    i = 0
    for c in CTX:
        for t in PERTS:
            got = a.X[i:i + 5]
            assert (got != blocks[(c, t)]).nnz == 0
            assert (a.obs["context"][i:i + 5] == c).all() and (a.obs["target_gene"][i:i + 5] == t).all()
            i += 5
    info = verify_h5ad(p, _contract())
    assert info["n_cells"] == 30 and info["nnz"] == a.X.nnz


@pytest.mark.parametrize("bad, msg", [
    (lambda b: b * 1.5, "fractional"),
    (lambda b: -b, "negative"),
    (lambda b: b * 1e9, "totals"),
])
def test_rejects_bad_values(tmp_path, bad, msg):
    rng = np.random.default_rng(1)
    w = SubmissionWriter(tmp_path / "x.h5ad", _contract())
    with pytest.raises(ContractError, match=msg):
        w.add_block(bad(_block(rng)), "A", "ADNP")


def test_rejects_unknown_labels_and_incomplete_files(tmp_path):
    rng = np.random.default_rng(2)
    w = SubmissionWriter(tmp_path / "x.h5ad", _contract())
    with pytest.raises(ContractError, match="not in the perturbation list"):
        w.add_block(_block(rng), "A", "non-targeting")
    with pytest.raises(ContractError, match="unknown context"):
        w.add_block(_block(rng), "D", "ADNP")
    w.add_block(_block(rng), "A", "ADNP")
    with pytest.raises(ContractError, match="incomplete"):
        w.close()


def test_explicit_zeros_are_not_stored(tmp_path):
    w = SubmissionWriter(tmp_path / "z.h5ad", _contract(n=2))
    dense = np.zeros((2, 30), dtype=np.float32); dense[0, 3] = 2; dense[1, 7] = 1
    M = sp.csr_matrix(dense); M.data[0] = 0  # an explicitly stored zero
    w.add_block(M, "A", "ADNP")
    assert w._nnz == 1


def test_pack_vcc_has_the_member_vcc_expects(tmp_path):
    rng = np.random.default_rng(3)
    p = tmp_path / "pred.h5ad"
    with SubmissionWriter(p, _contract()) as w:
        for c in CTX:
            for t in PERTS:
                w.add_block(_block(rng), c, t)
    v = pack_vcc(p, tmp_path / "pred.vcc")
    with tarfile.open(v) as tar:
        names = tar.getnames()
        assert names == ["pred.h5ad.zst"]
        m = tar.getmember("pred.h5ad.zst")
        assert m.size > 0 and m.uid == 0 and m.uname == "" and m.mtime == 0

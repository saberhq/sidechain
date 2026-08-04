"""Contract tests: every PriorSource must stay aligned to the master gene index and
never emit a dense matrix. These guard the extensibility invariant.
"""
import numpy as np
import pytest

from sidechain.priors.base import PriorSource, PriorArtifact


class _Dummy(PriorSource):
    def fetch(self):
        pass

    def build(self):
        # two directed edges among the first genes
        return PriorArtifact(
            kind="edge", relation="dummy", layer="posttx",
            edge_index=np.array([[0, 1], [2, 3]], dtype=np.int64),
        )


def test_artifact_is_sparse_and_aligned():
    gi = {f"g{i}": i for i in range(10)}
    art = _Dummy(spec={"name": "d", "layer": "posttx", "enabled": True}, gene_index=gi).build()
    assert art.kind == "edge"
    assert art.edge_index.shape[0] == 2          # COO, not dense
    assert art.edge_index.max() < len(gi)        # aligned to gene index


def test_enabled_flag_respected():
    gi = {"g0": 0}
    src = _Dummy(spec={"name": "d", "layer": "posttx", "enabled": False}, gene_index=gi)
    assert src.enabled() is False


# TODO(dev): add one real test per source once fetch()/build() are implemented.

"""Contract tests: every PriorSource must stay aligned to the master gene index
and never emit a dense matrix. These guard the extensibility invariant.

Note on the assertion that used to live here: `edge_index.shape[0] == 2` against
the fixture [[0,1],[2,3]] passes for a 2-edge COO array AND for a dense 2-gene
adjacency matrix -- so it passed for exactly the thing it claimed to forbid. The
tests below assert integer dtype, that the edge count equals the input pair count
after filtering, and behaviour when genes fall outside the index.
"""
import numpy as np
import pytest

from sidechain.priors.base import PriorArtifact, PriorSource


class _EdgeSource(PriorSource):
    """Builds edges from whatever pairs it is handed, via to_edge_index."""

    def __init__(self, spec, gene_index, srcs=(), dsts=(), attrs=None):
        super().__init__(spec, gene_index)
        self._srcs, self._dsts, self._attrs = list(srcs), list(dsts), attrs

    def fetch(self):
        pass

    def build(self):
        ei, attr = self.to_edge_index(self._srcs, self._dsts, self._attrs)
        return PriorArtifact(
            kind="edge",
            relation=self.relation,
            layer=self.layer,
            edge_index=ei,
            edge_attr=attr,
        )


def _spec(name="d", layer="posttx", relation="dummy", kind="edge", enabled=True):
    return {
        "name": name,
        "layer": layer,
        "relation": relation,
        "kind": kind,
        "enabled": enabled,
    }


GI = {f"g{i}": i for i in range(10)}


# ---------------------------------------------------------------- contract --


def test_kind_must_be_declared():
    """A source without an explicit kind: is a config error, not a default."""
    with pytest.raises(ValueError, match="must declare kind"):
        _EdgeSource({"name": "d", "layer": "posttx"}, GI)


def test_edge_index_is_sparse_coo_of_integers():
    src = _EdgeSource(_spec(), GI, srcs=["g0", "g1"], dsts=["g2", "g3"])
    art = src.build_checked()
    assert art.edge_index.shape == (2, 2)
    assert np.issubdtype(art.edge_index.dtype, np.integer)
    # A dense n_genes x n_genes adjacency would be (10, 10), never (2, n_edges).
    assert art.edge_index.shape[0] == 2 and art.edge_index.shape[1] != len(GI)
    np.testing.assert_array_equal(art.edge_index, np.array([[0, 1], [2, 3]]))


def test_edge_count_equals_surviving_pair_count():
    src = _EdgeSource(_spec(), GI, srcs=["g0", "g1", "g2"], dsts=["g3", "g4", "g5"])
    assert src.build_checked().edge_index.shape[1] == 3


# ------------------------------------------------- pair-aware ID filtering --


def test_unknown_endpoint_drops_the_whole_pair():
    """The bug this guards: filtering src and dst independently keeps 'g1' from
    one list and 'g3' from the other and pairs them into an edge that was never
    in the source data."""
    src = _EdgeSource(
        _spec(),
        GI,
        srcs=["g0", "NOT_A_GENE", "g1"],
        dsts=["g2", "g3", "ALSO_MISSING"],
    )
    art = src.build_checked()
    assert art.edge_index.shape[1] == 1, "only the g0->g2 pair has both endpoints"
    np.testing.assert_array_equal(art.edge_index, np.array([[0], [2]]))


def test_to_positions_and_to_edge_index_disagree_by_design():
    """to_positions drops independently; to_edge_index masks on the pair. This is
    exactly why edge sources must not build edge_index out of two to_positions
    calls."""
    src = _EdgeSource(_spec(), GI)
    srcs, dsts = ["g0", "MISSING"], ["MISSING", "g5"]
    naive_s, naive_d = src.to_positions(srcs), src.to_positions(dsts)
    assert len(naive_s) == len(naive_d) == 1  # same length here, but MIS-PAIRED:
    assert (naive_s[0], naive_d[0]) == (0, 5)  # an edge g0->g5 that never existed
    assert src.to_edge_index(srcs, dsts)[0].shape[1] == 0  # correct: no valid pair


def test_edge_attr_is_masked_with_the_edges():
    attrs = np.array([[0.1], [0.2], [0.3]])
    src = _EdgeSource(
        _spec(), GI, srcs=["g0", "MISSING", "g2"], dsts=["g1", "g3", "g4"], attrs=attrs
    )
    art = src.build_checked()
    assert art.edge_index.shape[1] == 2
    assert art.edge_attr.shape[0] == 2
    # The dropped middle row must be gone, not shifted.
    np.testing.assert_allclose(art.edge_attr.ravel(), [0.1, 0.3])


def test_mismatched_src_dst_lengths_raise():
    src = _EdgeSource(_spec(), GI)
    with pytest.raises(ValueError, match="same length"):
        src.to_edge_index(["g0", "g1"], ["g2"])


def test_all_genes_outside_index_yields_empty_not_malformed():
    src = _EdgeSource(_spec(), GI, srcs=["X", "Y"], dsts=["Z", "W"])
    art = src.build_checked()
    assert art.edge_index.shape == (2, 0)
    assert np.issubdtype(art.edge_index.dtype, np.integer)


def test_self_loops_dropped_on_request():
    src = _EdgeSource(_spec(), GI)
    ei, _ = src.to_edge_index(["g0", "g1"], ["g0", "g2"], drop_self_loops=True)
    np.testing.assert_array_equal(ei, np.array([[1], [2]]))


# ------------------------------------------------------- declared vs built --


def test_declared_kind_must_match_build_output():
    class _Liar(_EdgeSource):
        def build(self):
            return PriorArtifact(
                kind="node_feature",
                relation="x",
                layer="cis",
                features=np.zeros((len(GI), 3)),
            )

    with pytest.raises(ValueError, match="declared kind"):
        _Liar(_spec(kind="edge"), GI).build_checked()


def test_out_of_range_edge_index_is_rejected():
    class _OutOfRange(_EdgeSource):
        def build(self):
            return PriorArtifact(
                kind="edge",
                relation="x",
                layer="posttx",
                edge_index=np.array([[0, 999], [1, 2]], dtype=np.int64),
            )

    with pytest.raises(ValueError, match="out of range"):
        _OutOfRange(_spec(), GI).build_checked()


def test_float_edge_index_is_rejected():
    class _Floaty(_EdgeSource):
        def build(self):
            return PriorArtifact(
                kind="edge",
                relation="x",
                layer="posttx",
                edge_index=np.array([[0.0, 1.0], [2.0, 3.0]]),
            )

    with pytest.raises(ValueError, match="must be integer"):
        _Floaty(_spec(), GI).build_checked()


def test_enabled_flag_respected():
    src = _EdgeSource(_spec(enabled=False), {"g0": 0})
    assert src.enabled() is False


# -------------------------------------------------------------- edge types --


def test_edge_type_key_disambiguates_shared_relations():
    """mirbind2 and targetscan both declare relation 'mirna_target'; keying on
    relation alone would drop one when both are enabled."""
    a = _EdgeSource(_spec(name="mirbind2", relation="mirna_target"), GI)
    b = _EdgeSource(_spec(name="targetscan", relation="mirna_target"), GI)
    assert a.relation == b.relation
    assert a.edge_type_key != b.edge_type_key


def test_graph_builder_keeps_both_sources_of_a_shared_relation():
    from sidechain.priors.graph_builder import build_prior_graph

    a = _EdgeSource(
        _spec(name="mirbind2", relation="mirna_target"), GI, srcs=["g0"], dsts=["g1"]
    )
    b = _EdgeSource(
        _spec(name="targetscan", relation="mirna_target"), GI, srcs=["g2"], dsts=["g3"]
    )
    graph = build_prior_graph([a, b], n_genes=len(GI), to_pyg=False)
    assert len(graph["edges"]) == 2, "a shared relation name must not collapse sources"

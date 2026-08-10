# Developer (subagent)

**Goal:** implement against the interfaces in `src/sidechain/`. **Access:** write repo,
branch + PR only (never self-approve; the Critic gates).

Rules:
- Climb the rung ladder in `configs/model.yaml`; don't build Rung 3 before Rung 1 is scored.
- New biology = a `PriorSource` subclass + a `data_sources.yaml` block. Nothing else changes.
- Every source's `build()` returns a `PriorArtifact` aligned to the master gene index.
- Sparse only (`edge_index`/COO). Never allocate a dense gene x gene matrix.
- A change isn't done until `sidechain.eval.local_mirror` scores it and `tests/` pass.

**Assignments come from the task queue, not from this file.** This brief describes the role;
what to build next is decided outside it.

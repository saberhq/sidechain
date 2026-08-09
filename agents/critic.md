# Critic / Reviewer (subagent)

**Goal:** cheap gate before merge. Reads the diff + the metric deltas.

Block if: gain only on the public split; single-metric optimization that regresses
others; a "win" without a `local_mirror` score; a prior added without a contract test;
any dense-matrix allocation. Approve only well-rounded, reproducible improvements.

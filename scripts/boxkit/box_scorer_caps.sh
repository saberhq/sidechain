#!/bin/bash
# Put in front of cell-eval2 so every scoring step runs with its thread pools capped. On a box with many CPUs an uncapped step spreads
# over all of them: 8 cores alone and about 16 each with four at once on 124 CPUs, against 1.2 with every pool at 4 (same scores, same
# wall time alone, four at once 1.6 times faster). Capping some pools does nothing; all of them are listed here on purpose.
#   in Python:  sidechain.eval.mirror2026.CE2 = ["bash", ".../box_scorer_caps.sh"] + sidechain.eval.mirror2026.CE2
# BOX_SCORER_THREADS sets the cap (default 4). `gather_threads` is added only to `cell-eval2 run`; num_threads is not set: the GPU
# backend never reads it.
N=${BOX_SCORER_THREADS:-4}
export OMP_NUM_THREADS=$N OPENBLAS_NUM_THREADS=$N MKL_NUM_THREADS=$N POLARS_MAX_THREADS=$N \
       NUMBA_NUM_THREADS=$N RAYON_NUM_THREADS=$N NUMEXPR_NUM_THREADS=$N
ce=${1:?usage: box_scorer_caps.sh <cell-eval2> <args...>}; shift
if [ "${1:-}" = run ]; then exec "$ce" "$@" --set "gather_threads=$N"; fi
exec "$ce" "$@"

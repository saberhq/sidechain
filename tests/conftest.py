"""Session setup for the test suite.

**torch and scanpy fight over OpenMP on the Mac, and the loser segfaults.** Importing `torch`
-- which loads its own bundled `libomp` -- before scanpy's numba-compiled `normalize_total`
runs kills the interpreter outright: `Fatal Python error: Segmentation fault`, no exception,
no traceback of ours. It went unnoticed until 2026-09-05 because nothing under `tests/`
imported torch. `sidechain.models.masked_axis` does, and pytest imports every test module
during collection, so adding one test file took the whole suite down inside
`tests/test_data.py` -- a file that has nothing to do with either.

Pointing numba at its own thread pool instead of OpenMP removes the clash. Two things that do
NOT fix it, both tried on 2026-09-05: `KMP_DUPLICATE_LIB_OK=TRUE`, and importing scanpy before
torch. And setting the environment variable alone is not enough either -- a pytest plugin has
already imported numba by the time a conftest runs, and numba reads its threading layer at
import -- hence the reload.
"""

import os
import sys

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
if "numba" in sys.modules:
    import numba

    numba.config.reload_config()

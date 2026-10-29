"""Test-suite setup: numba uses its workqueue threading layer, not OpenMP.

    .venv/bin/python -m pytest -q tests/

Sets NUMBA_THREADING_LAYER and reloads numba's config if a plugin has already imported it.
Writes nothing. Notes: private/code-notes/tests/conftest.py.md
"""

import os
import sys

os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
if "numba" in sys.modules:
    import numba

    numba.config.reload_config()

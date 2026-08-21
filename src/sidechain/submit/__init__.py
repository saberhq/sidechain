"""Build a challenge submission without holding it in memory.

A 2026 submission is 360,000 cells x 18,533 genes; at realistic density that
is ~2e9 nonzeros, ~24 GiB as a CSR matrix -- more than this machine has. So
the prediction is written block by block with h5py into the exact layout
`vcc prep` would produce, checked against the submission contract as it
goes, and packed into the `.vcc` container (tar of pred.h5ad.zst) that
`vcc submit` accepts as-is.
"""
from sidechain.submit.writer import (  # noqa: F401
    SubmissionWriter,
    pack_vcc,
    verify_h5ad,
)

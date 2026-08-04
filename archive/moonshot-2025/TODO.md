# moonshot – next tasks

1) Environment & deps
   - Create a Python env for moonshot (conda or venv).
   - Add a minimal `requirements.txt` with:
     - anndata, scanpy, pandas, numpy, torch, cell-eval, pyyaml, tqdm
   - Install locally and verify imports.

2) Basic data loading
   - Create `src/data/loader.py` with functions to:
     - Load train/val `.h5ad` from a path (e.g., `../vcc_data/train.h5ad`).
     - Print shapes: `adata.shape`, key columns in `obs` and `var`.
   - Add a tiny script `scripts/check_data.py` that calls this and logs info.

3) QC & data sanity
   - Using scanpy, recompute QC metrics for the training data:
     - `sc.pp.calculate_qc_metrics(adata, inplace=True)`
   - Save as `train_qc.h5ad` to reuse.
   - Decide what subset (genes/cells) you want for the baseline.

4) Baseline model (minimum viable)
   - Implement `src/models/baseline.py`:
     - Start super simple: predict the **global gene mean** (or perturbation-wise mean) across training cells.
   - Implement `scripts/make_submission.py`:
     - Load val/test AnnData.
     - Apply baseline model to generate predictions.
     - Write predictions in the format expected by `cell-eval` / `vcc.ipynb`.

5) Local scoring & format validation
   - Install and import `cell-eval`.
   - Use the VCC tutorial notebook (`vcc.ipynb` from `cell-eval`) as a guide to:
     - Wrap your predicted matrix into a predicted AnnData.
     - Run the scoring functions on the **validation** set to ensure:
       - Your format is correct.
       - Metrics can be computed without errors.
   - Once this passes → you’re ready to push a first baseline submission to the VCC leaderboard.

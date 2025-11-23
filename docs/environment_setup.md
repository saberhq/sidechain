# Environment Setup

## 1. Install Miniforge

Miniforge provides a lightweight Conda distribution with the `conda-forge` channel pre-configured.

1. Download the latest installer for your platform from <https://conda-forge.org/miniforge/>.
2. Run the installer and follow the prompts (`bash Miniforge3-MacOSX-arm64.sh` on Apple Silicon).
3. After installation, open a new terminal so that `conda` is on your `PATH`.

## 2. Create the VCC environment

All project dependencies are described in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate vcc
```

The specification pins Python 3.11, core scverse packages (Scanpy/AnnData, muon, scvi-tools, pertpy, decoupler), PyTorch (CPU build), W&B, and installs `cell-eval` via `pip`.

## 3. Point to local data

Most utilities will ask for the data directory the first time they run (expects the folder containing `adata_Training.h5ad`, `gene_names.csv`, and `pert_counts_Validation.csv`). When prompted, paste the absolute path:

```bash
python scripts/inspect_dataset.py info --dataset train
# -> Enter the path to the VCC data directory:
```

If you prefer to skip the prompt, export `VCC_DATA_DIR` (and optionally `VCC_RESOURCES_DIR`) in your shell profile.

## 4. Optional: Update packages

To install GPU-enabled PyTorch builds or additional libraries, modify `environment.yml` and re-run:

```bash
conda env update -f environment.yml --prune
```

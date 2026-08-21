#!/usr/bin/env bash
# Bootstrap a Sidechain GPU box on NVIDIA Brev. Run AS THE LOGIN USER, after the
# instance is ready (this is not a root startup script any more -- the first version
# ran as root and resolved "the user" to uid 1000, which on Brev/Shadeform images is
# `ubuntu`, not the `shadeform` login you actually get; everything landed in the
# wrong home):
#
#   brev create sidechain-gpu --type hyperstack_A100_80G
#   brev exec sidechain-gpu @scripts/brev_bootstrap.sh
#   brev copy ~/data/sidechain/vcc2026/vcc_2026_controls.zip sidechain-gpu:~/data/sidechain/vcc2026/
#
# Idempotent. Needs passwordless sudo for apt only. The vcc login (API key) is
# Saber's to run, in his own shell: `printf '%s' "$KEY" | vcc login --token-stdin`.
set -euo pipefail
export PATH="$HOME/.local/bin:$PATH"

sudo apt-get update -qq && sudo apt-get install -y -qq git curl unzip build-essential >/dev/null

# uv (Python + project env) and the vcc CLI with its agent skill
command -v uv >/dev/null || curl -LsSf https://astral.sh/uv/install.sh | sh
uv tool install --quiet vcc-cli || uv tool upgrade --quiet vcc-cli
vcc skill install --agent claude >/dev/null 2>&1 || true

# the public repo; the private repo is cloned by hand after an SSH key is added
mkdir -p ~/code ~/data/sidechain/{vcc2026,cache/vcc2026,runs}
cd ~/code
[ -d sidechain ] || git clone --quiet https://github.com/saberhq/sidechain.git
cd sidechain && git pull --quiet && uv sync --quiet

# GPU extras for the 2026 scorer: CUDA kernels + GPU differential expression.
# Competition-rule scoring resolves DE on the GPU; a CPU (pdex) run builds a
# "diagnostic" bundle whose config hash differs from the vcc2026 preset.
# `--torch-backend=auto` picked a CUDA-13 torch (2.13.0+cu130) on a box whose
# driver (570.x) tops out at CUDA 12.8 -> torch.cuda.is_available() == False.
# Pin the CUDA-12.6 wheels instead; they run on every driver >= 560.
uv pip install --quiet --torch-backend=cu126 "cell-eval2[gpu,gpudge]" \
  || echo "cell-eval2 GPU extras failed; the CPU path still works"
uv pip install --quiet --reinstall --index-url https://download.pytorch.org/whl/cu126 torch

echo "sidechain bootstrap done on $(hostname) at $(date -u): $(uv run python -c 'import torch; print("cuda", torch.cuda.is_available())' 2>/dev/null)"

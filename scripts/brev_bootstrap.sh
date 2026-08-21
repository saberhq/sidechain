#!/usr/bin/env bash
# Startup script for a Sidechain GPU box on NVIDIA Brev.
#
#   brev create sidechain-gpu -g A100 --min-disk 500 --stoppable \
#        --startup-script @scripts/brev_bootstrap.sh
#
# Runs as root on first boot (Brev convention); everything lands under the
# default user's home. Idempotent. After it finishes:
#
#   brev copy ~/data/sidechain/vcc2026/vcc_2026_controls.zip sidechain-gpu:~/data/sidechain/vcc2026/
#   brev shell sidechain-gpu
#   ... then on the box: unzip, `printf '%s' "$KEY" | vcc login --token-stdin` (Saber only).
#
# UNTESTED as of 2026-08-21: written before the first `brev login` on this Mac.
set -euo pipefail
USER_HOME=$(getent passwd 1000 | cut -d: -f6 || echo /home/ubuntu)
run_as_user() { sudo -u "$(basename "$USER_HOME")" -H bash -lc "$*"; }

apt-get update -qq && apt-get install -y -qq git curl unzip build-essential >/dev/null

# uv (Python + project env), the vcc CLI, and its agent skill
run_as_user 'curl -LsSf https://astral.sh/uv/install.sh | sh'
run_as_user 'export PATH="$HOME/.local/bin:$PATH" && uv tool install vcc-cli && vcc skill install --agent claude || true'

# the public repo; the private repo is cloned by hand after an SSH key is added
run_as_user 'mkdir -p ~/code ~/data/sidechain/{vcc2026,cache,runs} && cd ~/code && \
  { [ -d sidechain ] || git clone https://github.com/saberhq/sidechain.git; } && \
  cd sidechain && export PATH="$HOME/.local/bin:$PATH" && uv sync'

# GPU extras for the 2026 scorer: CUDA kernels + GPU differential expression
run_as_user 'cd ~/code/sidechain && export PATH="$HOME/.local/bin:$PATH" && \
  uv pip install --torch-backend=auto "cell-eval2[gpu,gpudge]" || echo "cell-eval2 GPU extras failed; CPU path still works"'

echo "sidechain bootstrap done: $(date -u)"

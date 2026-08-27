#!/usr/bin/env bash
# Bootstrap a Sidechain GPU box on NVIDIA Brev. Run AS THE LOGIN USER, after the
# instance is ready (this is not a root startup script any more -- the first version
# ran as root and resolved "the user" to uid 1000, which on Brev/Shadeform images is
# `ubuntu`, not the `shadeform` login you actually get; everything landed in the
# wrong home):
#
#   brev search --min-total-vram 24 --min-disk 200   # pick from a FRESH list, cheapest first
#   brev create sidechain-gpu --type massedcompute_L40S
#   brev copy scripts/brev_bootstrap.sh sidechain-gpu:~/brev_bootstrap.sh
#   brev exec sidechain-gpu -- 'bash ~/brev_bootstrap.sh'
#   brev copy ~/data/sidechain/vcc2026/vcc_2026_controls.zip sidechain-gpu:~/data/sidechain/vcc2026/
#
# Idempotent. Needs passwordless sudo for apt only. The vcc login (API key) is
# Saber's to run, in his own shell: `printf '%s' "$KEY" | vcc login --token-stdin`.
#
# ---------------------------------------------------------------------------
# FIVE THINGS THAT FAIL SILENTLY ON THIS BOX (measured 2026-08-24). Full runbook:
# the `sidechain-brev` skill.
#
# 1. ANY SCRIPT YOU RUN AFTER THIS ONE MUST RE-EXPORT PATH.
#    The `export PATH="$HOME/.local/bin:$PATH"` below applies to THIS shell only.
#    A non-interactive `brev exec` does not source the profile, so a later script
#    hits `uv: command not found` on its first `uv run`, dies instantly under
#    `set -e`, writes no output directory, and looks exactly like "never launched".
#      export PATH="$HOME/.local/bin:$PATH"     <- first line of every remote script
#
# 2. `brev exec -- bash -s < script.sh` RUNS THIS SCRIPT ON YOUR MAC, not on the box.
#    The redirect never reaches the remote shell. The errors mention SSH and quote
#    script lines as instance names, so it reads like a connection problem -- but the
#    lines are executing locally. On 2026-08-24 that ran line 59 below
#    (`[ -d sidechain ] || git clone ...`) inside ~/code/sidechain and cloned the repo
#    into itself, untracked and unignored. `/sidechain/` is in .gitignore now.
#    NEVER pipe or redirect a script into `brev exec`. Copy it over and run it there.
#
# 3. `nohup ... &` DOES NOT SURVIVE `brev exec`. Long jobs need tmux:
#      brev exec BOX -- 'tmux new-session -d -s job "bash ~/run.sh > ~/run.log 2>&1"'
#    then poll `tail ~/run.log` from separate short execs.
#
# 4. REPEATED `--type` FLAGS DO NOT ACCUMULATE in `brev create` -- only the last is
#    used, so a "fallback chain" is silently a single attempt.
#
# 5. `CREATE_FAILED` IS USUALLY THE PROVIDER. hyperstack_L40 refused a valid config;
#    massedcompute_L40S took it minutes later. Delete the failed instance, wait for
#    `brev ls` to stop listing it, and pick again from a FRESH `brev search` -- the
#    availability list goes stale and a stale type comes back "not recognized".
#
# And: the box BILLS WHILE IT EXISTS. `brev stop` or `brev delete` when done.
# ---------------------------------------------------------------------------
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
# What the GPU buys is SPEED, not rule-compliance -- corrected 2026-08-26 after
# this comment sent a session the wrong way. The competition rule hashes the
# CONTROL LABEL, not the DE backend (`pert_col`, `de.backend` and `outdir` are
# excluded from the digest), so a CPU `pdex` bundle is rule-exact too: both
# bundles rebuilt that day with `--control non-targeting` and `--de-backend
# pdex` came back with `rule_mismatches: []`. A bundle is diagnostic when its
# held-out line's controls are still labelled `control`.
#
# But the CONFIG digest is stricter than the rule digest: it includes
# `resolved_device`. A bundle built on CPU refuses arms scored on this box
# unless the GPU is hidden -- `export CUDA_VISIBLE_DEVICES=""` in the run
# script for any arm whose bundle records `resolved_device: cpu` (2026-08-27,
# four arms died on it after their DE had already computed; skill trap 10).
#
# These extras install into WHICHEVER venv is active. A second clone of this
# repo -- pinning a SHA for a reproducible run, say -- gets none of them from
# `uv sync --frozen`, and the gap only shows up when an arm asks for the gpudge
# backend its bundle was built with. Re-run these two lines in that clone.
# `--torch-backend=auto` picked a CUDA-13 torch (2.13.0+cu130) on a box whose
# driver (570.x) tops out at CUDA 12.8 -> torch.cuda.is_available() == False.
# Pin the CUDA-12.6 wheels instead; they run on every driver >= 560.
uv pip install --quiet --torch-backend=cu126 "cell-eval2[gpu,gpudge]" \
  || echo "cell-eval2 GPU extras failed; the CPU path still works"
uv pip install --quiet --reinstall --index-url https://download.pytorch.org/whl/cu126 torch

echo "sidechain bootstrap done on $(hostname) at $(date -u): $(uv run python -c 'import torch; print("cuda", torch.cuda.is_available())' 2>/dev/null)"

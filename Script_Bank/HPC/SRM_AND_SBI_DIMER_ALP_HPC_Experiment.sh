#!/bin/bash
# =============================================================================
# Slurm HPC application submitter: MAP estimation on real microscopy videos.
# =============================================================================
# Single GPU node. Reads the trained posterior + the .tif
# recordings under <data_bank>/Experiment/, writes inferred-parameter
# distributions per condition (Posit/..._MAP_Experiment/).
# Overridable via --export: KINDS (e.g. ALP,BET), MAX_CELLS (0=all),
#   CHUNK_STEP (seconds), SUMMARY (map|posterior|both), POOL_MODE, TOTAL_TIME.
#   Non-deterministic (no seed).
# Submit from the repo root and forward REPO: Slurm spools this script to
# /var/spool, so the child must be told where the repo is (--export=ALL,REPO=$PWD).
# --job-name follows the data-file naming convention
# SRM_AND_SBI_DIMER_ALP_<timing_label>_Experiment; with no TOTAL_TIME set the
# launcher default (2.0 s) gives timing_label 2S_50FPS, so swap the token (e.g.
# 5S_50FPS) whenever you pass TOTAL_TIME=5.0.
# CAVEAT: Slurm's --export splits its value on commas, so a multi-value KINDS
# CANNOT go inside the --export string (--export=ALL,KINDS=ALP,BET would parse as
# KINDS=ALP plus a stray, value-less BET). Either leave KINDS at the script default
# (ALP,BET), or pre-export it in the submitting shell and let --export=ALL carry it:
# Example (default KINDS):
#   cd /path/to/srm-and-sbi-dimer-alp
#   sbatch --job-name=SRM_AND_SBI_DIMER_ALP_2S_50FPS_Experiment --export=ALL,REPO=$PWD,SUMMARY=both Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Experiment.sh
# Example (multi-value KINDS via the environment, NOT inside --export):
#   cd /path/to/srm-and-sbi-dimer-alp
#   export KINDS=ALP,BET
#   sbatch --job-name=SRM_AND_SBI_DIMER_ALP_2S_50FPS_Experiment --export=ALL,REPO=$PWD,SUMMARY=both Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Experiment.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_DIMER_ALP_Experiment   # fallback; per-run --job-name (with timing_label) overrides this
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=%x_%A.out   # submit-directory; the controller overrides this via MON_OUT for packed jobs

set -eo pipefail

# Locate the repo root robustly. Slurm runs a SPOOLED COPY of this batch script
# from /var/spool, so BASH_SOURCE is unreliable for a directly-submitted job.
# Resolve REPO from, in order: an explicit REPO (e.g. --export=ALL,REPO=...), the
# Slurm submit directory, or this script's own location (for a non-Slurm
# `bash <script>`); accept the first that actually contains the package, and fail
# loud otherwise rather than crashing cryptically on a /var/spool path.
_find_repo() {
    local c
    for c in "${REPO:-}" "${SLURM_SUBMIT_DIR:-}" "${SLURM_SUBMIT_DIR:+$SLURM_SUBMIT_DIR/../..}" \
             "$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." 2>/dev/null && pwd)"; do
        [ -n "$c" ] || continue
        if [ -f "$c/pyproject.toml" ] && [ -d "$c/srm_and_sbi_dimer_alp" ]; then
            (cd "$c" && pwd); return 0
        fi
    done
    return 1
}
REPO="$(_find_repo)" || {
    echo "FATAL: cannot locate the srm-and-sbi-dimer-alp repo root (Slurm spools this" >&2
    echo "  script, so its own path is unreliable). Submit with an explicit REPO, e.g.:" >&2
    echo "    cd /path/to/srm-and-sbi-dimer-alp && sbatch --export=ALL,REPO=\$PWD,... <this-script>" >&2
    exit 1
}
cd "$REPO"

# Per-machine HPC config (gitignored; copy from hpc_local.env.example): sets
# MACHINE_PROFILE / CONDA_SETUP / etc. Sourced via the resolved REPO so it is
# found even under Slurm spooling. Falls back to the defaults below if absent.
HPC_ENV="${HPC_ENV:-$REPO/Script_Bank/HPC/hpc_local.env}"
if [ -f "$HPC_ENV" ]; then . "$HPC_ENV"; fi

source "${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate SRM_AND_SBI_ENVY_V0
export MACHINE_PROFILE="${MACHINE_PROFILE:?set MACHINE_PROFILE (via hpc_local.env or --export) to a profile in your machine_profiles.toml}"

KINDS="${KINDS:-ALP,BET}"
MAX_CELLS="${MAX_CELLS:-0}"
CHUNK_STEP="${CHUNK_STEP:-2}"
SUMMARY="${SUMMARY:-both}"
POOL_MODE="${POOL_MODE:-bounded}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"

echo "=== Experiment | kinds=${KINDS} max_cells=${MAX_CELLS} chunk_step=${CHUNK_STEP}s summary=${SUMMARY} seed=None | node $(hostname) ==="

python -u "$REPO/Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Experiment.py" \
    --kinds "$KINDS" --max-cells "$MAX_CELLS" --chunk-step-seconds "$CHUNK_STEP" \
    --summary "$SUMMARY" --pool-mode "$POOL_MODE" \
    --total-time-seconds "$TOTAL_TIME"

echo "=== Experiment complete ==="

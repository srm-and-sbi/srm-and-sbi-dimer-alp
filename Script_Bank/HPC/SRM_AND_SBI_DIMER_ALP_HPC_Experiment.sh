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
# Example:
#   sbatch --export=ALL,KINDS=ALP,BET,SUMMARY=both SRM_AND_SBI_DIMER_ALP_HPC_Experiment.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_DIMER_ALP_HPC_Experiment
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=%x_%A.out   # submit-directory; the controller overrides this via MON_OUT for packed jobs

set -eo pipefail

source "${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate SRM_AND_SBI_ENVY_V0
export MACHINE_PROFILE="${MACHINE_PROFILE:?set MACHINE_PROFILE to the profile name in your machine_profiles.toml}"

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"   # self-derived: each script lives at REPO/Script_Bank/HPC/
cd "$REPO"

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

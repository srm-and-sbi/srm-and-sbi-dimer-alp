#!/bin/bash
# =============================================================================
# Slurm HPC validation submitter: MAP recovery on the held-out EVAL set.
# =============================================================================
# Single GPU node. Reads the trained posterior + EVAL data,
# writes the recovery report (Posit/..._MAP_Recovery/).
# Overridable via --export: EVAL_TASKS, SUMMARY (map|posterior|both), POOL_MODE
#   (bounded|unrestricted), TOTAL_TIME. Non-deterministic (no seed).
# Example:
#   sbatch --export=ALL,EVAL_TASKS=1,SUMMARY=both SRM_AND_SBI_DIMER_ALP_HPC_Evaluation.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_DIMER_ALP_HPC_Evaluation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=12:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=%x_%A.out   # submit-directory; the controller overrides this via MON_OUT for packed jobs

set -eo pipefail

source "${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate SRM_AND_SBI_ENVY_V0
export MACHINE_PROFILE="${MACHINE_PROFILE:?set MACHINE_PROFILE to the profile name in your machine_profiles.toml}"

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"   # self-derived: each script lives at REPO/Script_Bank/HPC/
cd "$REPO"

EVAL_TASKS="${EVAL_TASKS:-1}"
SUMMARY="${SUMMARY:-both}"
POOL_MODE="${POOL_MODE:-bounded}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"

echo "=== Evaluation | eval_tasks=${EVAL_TASKS} summary=${SUMMARY} pool=${POOL_MODE} time=${TOTAL_TIME}s seed=None | node $(hostname) ==="

python -u "$REPO/Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Evaluation.py" \
    --eval-tasks "$EVAL_TASKS" --summary "$SUMMARY" --pool-mode "$POOL_MODE" \
    --total-time-seconds "$TOTAL_TIME"

echo "=== Evaluation complete ==="

#!/bin/bash
# =============================================================================
# Slurm HPC validation submitter: MAP recovery on the held-out EVAL set.
# =============================================================================
# Adapts to the allocated GPUs: with >1 GPU it shards the EVAL set across one
# worker per GPU (torchrun) then merges the per-shard results into one report;
# with 1 GPU it is the original single-GPU path. Reads the trained posterior +
# EVAL data, writes the recovery report (Posit/..._MAP_Recovery/).
# Overridable via --export: EVAL_TASKS, SUMMARY (map|posterior|both), POOL_MODE
#   (bounded|unrestricted), TOTAL_TIME, SRM_AND_SBI_GPUS (cap the GPUs used;
#   default = all allocated). Non-deterministic (no seed).
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

# Per-machine HPC config (gitignored; copy from hpc_local.env.example): sets
# MACHINE_PROFILE / CONDA_SETUP / etc. so this generic script runs unchanged
# on any cluster. Falls back to the defaults below if absent.
HPC_ENV="${HPC_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hpc_local.env}"
if [ -f "$HPC_ENV" ]; then . "$HPC_ENV"; fi

source "${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate SRM_AND_SBI_ENVY_V0
export MACHINE_PROFILE="${MACHINE_PROFILE:?set MACHINE_PROFILE to the profile name in your machine_profiles.toml}"

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"   # self-derived: each script lives at REPO/Script_Bank/HPC/
cd "$REPO"

EVAL_TASKS="${EVAL_TASKS:-1}"
SUMMARY="${SUMMARY:-both}"
POOL_MODE="${POOL_MODE:-bounded}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"

# GPU count for sharding: SRM_AND_SBI_GPUS override, else allocated GPUs, else 1.
GPUS="${SRM_AND_SBI_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
EVAL_PY="$REPO/Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Evaluation.py"
EVAL_ARGS=( --eval-tasks "$EVAL_TASKS" --summary "$SUMMARY" --pool-mode "$POOL_MODE"
            --total-time-seconds "$TOTAL_TIME" )

echo "=== Evaluation | eval_tasks=${EVAL_TASKS} summary=${SUMMARY} pool=${POOL_MODE} time=${TOTAL_TIME}s gpus=${GPUS} seed=None | node $(hostname) ==="

if [ "${GPUS:-1}" -gt 1 ]; then
    # Shard recovery across $GPUS workers (one GPU each), then merge to one report.
    torchrun --standalone --nproc_per_node="$GPUS" "$EVAL_PY" "${EVAL_ARGS[@]}"
    python -u "$EVAL_PY" "${EVAL_ARGS[@]}" --merge
else
    # Single GPU: the original path (writes the report directly; no merge).
    python -u "$EVAL_PY" "${EVAL_ARGS[@]}"
fi

echo "=== Evaluation complete ==="

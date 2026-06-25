#!/bin/bash
# =============================================================================
# Slurm HPC training submitter: train the posterior on TRAIN, select on TEST.
# =============================================================================
# Single GPU node. The `gpu` partition allocates a whole node (8 GPUs); our code
# uses one GPU (gpu_device_index=0 in the active machine profile).
# Overridable via --export: TRAIN_TASKS, TEST_TASKS, EPOCHS, TOTAL_TIME, BATCH.
# BATCH is optional: leave unset to use the script default (PARAMETERS batch_size,
# 32). Long videos (e.g. 10 s = 500 frames) need a smaller batch -- one early
# conv3d activation is batch * ~1 GiB at 500 frames, so batch 32 OOMs a 64 GB GPU.
# Training is non-deterministic (no seed; consistent with generation).
# Example:
#   sbatch --export=ALL,TRAIN_TASKS=8,TEST_TASKS=2,EPOCHS=50 SRM_AND_SBI_DIMER_ALP_HPC_Inference.sh
# Quick smoke on a shorter-lived test GPU partition (set <gpu-partition> to your cluster's):
#   sbatch --partition=<gpu-partition> --gres=gpu:1 --time=01:00:00 --export=ALL,TRAIN_TASKS=8,TEST_TASKS=2,EPOCHS=1 SRM_AND_SBI_DIMER_ALP_HPC_Inference.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_DIMER_ALP_HPC_Inference
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
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

TRAIN_TASKS="${TRAIN_TASKS:-8}"
TEST_TASKS="${TEST_TASKS:-2}"
EPOCHS="${EPOCHS:-50}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"
BATCH="${BATCH:-}"   # empty -> script default (PARAMETERS batch_size, 32)

BATCH_ARG=()
[ -n "$BATCH" ] && BATCH_ARG=(--batch-size "$BATCH")

echo "=== Inference | train_tasks=${TRAIN_TASKS} test_tasks=${TEST_TASKS} epochs=${EPOCHS} time=${TOTAL_TIME}s batch=${BATCH:-default} seed=None | node $(hostname) ==="

python -u "$REPO/Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Inference.py" \
    --tasks "$TRAIN_TASKS" --test-tasks "$TEST_TASKS" --epochs "$EPOCHS" \
    --total-time-seconds "$TOTAL_TIME" "${BATCH_ARG[@]}"

echo "=== Inference complete ==="

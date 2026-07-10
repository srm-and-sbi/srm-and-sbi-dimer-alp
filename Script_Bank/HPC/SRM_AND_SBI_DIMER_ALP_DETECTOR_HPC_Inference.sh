#!/bin/bash
# =============================================================================
# Detector-workflow training (B3): train the imaging-parameter posterior estimator.
# Part of the Detector calibration workflow's OWN committed submission machinery
# (Script_Bank/HPC/), parallel to the canonical pipeline and never wired
# into the canonical Submit.sh dispatcher.
# =============================================================================
# Mirrors the canonical SRM_AND_SBI_DIMER_ALP_HPC_Inference.sh: adapts to the
# allocated GPUs — with >1 GPU it trains data-parallel via DistributedDataParallel
# (torchrun, one process per GPU); with 1 GPU it is the single-GPU path. It drives
# the Detector B3 entry point, which reuses the canonical setup_training /
# train_loop by import and saves the version-portable A5 estimator artifact.
# Overridable via --export: TRAIN_TASKS, TEST_TASKS, EPOCHS, TOTAL_TIME, BATCH,
#   RESURRECT (1 = continue from the existing Detector checkpoint), HEARTBEAT,
#   SRM_AND_SBI_GPUS (cap the GPUs used; default = all allocated). On Goethe the
#   two GPU modes are gpu_test (4 GPUs, checks) and gpu (8 GPUs, production); the
#   Detector Submit helper pins SRM_AND_SBI_GPUS to match the chosen partition.
# Training is non-deterministic (no seed; consistent with generation). Submit from
# the repo root or forward REPO explicitly.
# --job-name: SRM_AND_SBI_DIMER_ALP_DETECTOR_<timing_label>_Inference.
# Example (2 s smoke on a 4-GPU test partition):
#   cd /path/to/srm-and-sbi-dimer-alp
#   sbatch --partition=gpu_test --gres=gpu:4 --time=02:00:00 \
#     --job-name=SRM_AND_SBI_DIMER_ALP_DETECTOR_2S_50FPS_Inference \
#     --export=ALL,REPO=$PWD,SRM_AND_SBI_GPUS=4,TRAIN_TASKS=16,TEST_TASKS=4,EPOCHS=5,BATCH=8,TOTAL_TIME=2.0 \
#     Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_Inference.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_DIMER_ALP_DETECTOR_Inference   # fallback; per-run --job-name overrides
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=1-00:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=%x_%A.out

set -eo pipefail

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
    echo "FATAL: cannot locate the srm-and-sbi-dimer-alp repo root. Submit with an explicit REPO." >&2
    exit 1
}
cd "$REPO"

HPC_ENV="${HPC_ENV:-$REPO/Script_Bank/HPC/hpc_local.env}"
if [ -f "$HPC_ENV" ]; then . "$HPC_ENV"; fi

source "${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate SRM_AND_SBI_ENVY_V0
export MACHINE_PROFILE="${MACHINE_PROFILE:?set MACHINE_PROFILE (via hpc_local.env or --export)}"

TRAIN_TASKS="${TRAIN_TASKS:-8}"
TEST_TASKS="${TEST_TASKS:-2}"
EPOCHS="${EPOCHS:-50}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"
BATCH="${BATCH:-}"           # empty -> Detector B3 default (PARAMETERS batch_size)
HEARTBEAT="${HEARTBEAT:-}"
RESURRECT="${RESURRECT:-}"

BATCH_ARG=();     [ -n "$BATCH" ]     && BATCH_ARG=(--batch-size "$BATCH")
HEARTBEAT_ARG=(); [ -n "$HEARTBEAT" ] && HEARTBEAT_ARG=(--heartbeat "$HEARTBEAT")
RESURRECT_ARG=(); [ "$RESURRECT" = 1 ] && RESURRECT_ARG=(--resurrect)

# GPU count for data-parallel training: SRM_AND_SBI_GPUS override, else allocated, else 1.
GPUS="${SRM_AND_SBI_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
INFER_PY="$REPO/Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_DETECTOR_Inference.py"
INFER_ARGS=( --tasks "$TRAIN_TASKS" --test-tasks "$TEST_TASKS" --epochs "$EPOCHS"
             --total-time-seconds "$TOTAL_TIME" "${BATCH_ARG[@]}" "${HEARTBEAT_ARG[@]}" "${RESURRECT_ARG[@]}" )

echo "=== Detector Inference | train=${TRAIN_TASKS} test=${TEST_TASKS} epochs=${EPOCHS} time=${TOTAL_TIME}s batch=${BATCH:-default} resurrect=${RESURRECT:-0} gpus=${GPUS} | node $(hostname) ==="

if [ "${GPUS:-1}" -gt 1 ]; then
    torchrun --standalone --nproc_per_node="$GPUS" "$INFER_PY" "${INFER_ARGS[@]}"
else
    python -u "$INFER_PY" "${INFER_ARGS[@]}"
fi

echo "=== Detector Inference complete ==="

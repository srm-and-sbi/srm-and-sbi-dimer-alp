#!/bin/bash
# =============================================================================
# Detector-workflow evaluation (B5): MAP recovery of the imaging parameters on the
# held-out _DETECTOR EVAL set. Part of the Detector calibration workflow's OWN
# committed submission machinery (Script_Bank/HPC/), parallel to the
# canonical pipeline and never wired into the canonical Submit.sh dispatcher.
# =============================================================================
# Mirrors the canonical SRM_AND_SBI_DIMER_ALP_HPC_Evaluation.sh: with >1 GPU it
# shards the EVAL set across one worker per GPU (torchrun) then runs a separate
# --merge step to combine the per-shard arrays into one report; with 1 GPU it is
# the single-GPU path. Workers are auto-capped at EVAL_TASKS by the B5 script
# (an idle worker does no recovery), so EVAL_TASKS<=1 runs the single-process path.
# Overridable via --export: EVAL_TASKS, POOL_MODE (bounded|unrestricted;
#   default = the config value, as in the canonical Evaluation — 'unrestricted'
#   for an undertrained/smoke posterior that would stall bounded rejection),
#   TOTAL_TIME, SRM_AND_SBI_GPUS (cap the GPUs used; default = all allocated).
# Non-deterministic (no seed). Submit from the repo root or forward REPO.
# --job-name: SRM_AND_SBI_DIMER_ALP_DETECTOR_<timing_label>_Evaluation.
# Example (2 s smoke, 2 EVAL tasks -> 2-way shard + merge, 4-GPU test partition):
#   sbatch --partition=gpu_test --gres=gpu:4 --time=01:00:00 \
#     --job-name=SRM_AND_SBI_DIMER_ALP_DETECTOR_2S_50FPS_Evaluation \
#     --export=ALL,REPO=$PWD,SRM_AND_SBI_GPUS=4,EVAL_TASKS=2,POOL_MODE=unrestricted,TOTAL_TIME=2.0 \
#     Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_Evaluation.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_DIMER_ALP_DETECTOR_Evaluation   # fallback; per-run --job-name overrides
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=12:00:00
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

EVAL_TASKS="${EVAL_TASKS:-1}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"
POOL_MODE="${POOL_MODE:-}"   # empty -> Detector B5 default (the config pool_mode, i.e. bounded)

POOL_ARG=(); [ -n "$POOL_MODE" ] && POOL_ARG=(--pool-mode "$POOL_MODE")

# GPU count for sharding: SRM_AND_SBI_GPUS override, else allocated, else 1.
GPUS="${SRM_AND_SBI_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
# Never launch more workers than EVAL tasks -- an idle worker does no recovery.
if [ "$EVAL_TASKS" -lt "$GPUS" ]; then GPUS="$EVAL_TASKS"; fi
EVAL_PY="$REPO/Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_DETECTOR_Evaluation.py"
EVAL_ARGS=( --eval-tasks "$EVAL_TASKS" --total-time-seconds "$TOTAL_TIME" "${POOL_ARG[@]}" )

echo "=== Detector Evaluation | eval_tasks=${EVAL_TASKS} pool=${POOL_MODE:-default} time=${TOTAL_TIME}s gpus=${GPUS} | node $(hostname) ==="

if [ "${GPUS:-1}" -gt 1 ]; then
    # Shard recovery across $GPUS workers (one GPU each), then merge to one report.
    torchrun --standalone --nproc_per_node="$GPUS" "$EVAL_PY" "${EVAL_ARGS[@]}"
    python -u "$EVAL_PY" "${EVAL_ARGS[@]}" --merge
else
    # Single GPU: the original path (writes the report directly; no merge).
    python -u "$EVAL_PY" "${EVAL_ARGS[@]}"
fi

echo "=== Detector Evaluation complete ==="

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
#   default = all allocated). Workers are auto-capped at EVAL_TASKS (an idle
#   worker would do no recovery), so EVAL_TASKS<=1 runs the single-process path.
#   Non-deterministic (no seed).
# Submit from the repo root and forward REPO: Slurm spools this script to
# /var/spool, so the child must be told where the repo is (--export=ALL,REPO=$PWD).
# --job-name follows the data-file naming convention
# SRM_AND_SBI_DIMER_ALP_<timing_label>_Evaluation; with no TOTAL_TIME set the
# launcher default (2.0 s) gives timing_label 2S_50FPS, so swap the token (e.g.
# 5S_50FPS) whenever you pass TOTAL_TIME=5.0. POOL_MODE defaults to bounded (a
# trained posterior); use POOL_MODE=unrestricted for an undertrained/smoke posterior.
# Example:
#   cd /path/to/srm-and-sbi-dimer-alp
#   sbatch --job-name=SRM_AND_SBI_DIMER_ALP_2S_50FPS_Evaluation --export=ALL,REPO=$PWD,EVAL_TASKS=1,SUMMARY=both Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Evaluation.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_DIMER_ALP_Evaluation   # fallback; per-run --job-name (with timing_label) overrides this
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=12:00:00
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

EVAL_TASKS="${EVAL_TASKS:-1}"
SUMMARY="${SUMMARY:-both}"
POOL_MODE="${POOL_MODE:-bounded}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"

# GPU count for sharding: SRM_AND_SBI_GPUS override, else allocated GPUs, else 1.
GPUS="${SRM_AND_SBI_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
# Never launch more workers than EVAL tasks -- an idle worker does no recovery.
# EVAL_TASKS < GPUS caps GPUS; EVAL_TASKS <= 1 collapses to the single-process path.
if [ "$EVAL_TASKS" -lt "$GPUS" ]; then GPUS="$EVAL_TASKS"; fi
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

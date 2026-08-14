#!/bin/bash
# =============================================================================
# Slurm HPC application submitter: emit the Nuisance_DLI spec template (pool build).
# =============================================================================
# Runs the Detector Nuisance_DLI analysis in --emit-template mode: it reads the
# trained posterior + the .tif recordings under <data_bank>/Experiment/, builds the
# pooled posterior-sample pool (the GPU cost), caches it, and emits the value-based
# spec pre-filled with the calibrated-imaging percentiles for a person to finalize.
# Adapts to the allocation: with >1 node it shards the (kind, cell) work across one
# worker per GPU on EVERY node (srun places one torchrun launcher per node) into
# per-rank pool shards on the shared filesystem, then a single-process, no-GPU
# --merge step concatenates every shard into the cached pool + spec; with 1 node
# and >1 GPU it shards across that node's GPUs (torchrun --standalone) then merges;
# with 1 GPU it is the original single-process path. --gres is per node, so
# --nodes=N --gres=gpu:G gives N*G shard workers (world_size = N*G); the sharding is
# embarrassingly parallel (workers just need the shared output dir for the merge). A
# worker that draws no cells writes no shard. (This stage is submitted directly by
# sbatch, not via Submit.sh, so pass --nodes=N on the sbatch line for multi-node.)
# This does NOT build the artifact -- edit the emitted _Nuisance_DLI_Spec.toml, then
# run the analysis with --build (single process; it reuses the cached pool, no GPU).
# Overridable via --export: POOL_MODE (bounded|unrestricted; default unrestricted),
#   TOTAL_TIME (model/recording seconds; default 2.0), SPAN (recording span seconds;
#   default 20), SRM_AND_SBI_GPUS (cap the GPUs used; default = all allocated).
# Submit from the repo root and forward REPO: Slurm spools this script to
# /var/spool, so the child must be told where the repo is (--export=ALL,REPO=$PWD).
# --job-name follows the data-file naming convention
# SRM_AND_SBI_DIMER_ALP_DETECTOR_<timing_label>_Nuisance_DLI; with no TOTAL_TIME set the
# launcher default (2.0 s) gives timing_label 2S_50FPS, so swap the token (e.g.
# 5S_50FPS) whenever you pass TOTAL_TIME=5.0.
# Example (defaults; unrestricted pool):
#   cd /path/to/srm-and-sbi-dimer-alp
#   sbatch --job-name=SRM_AND_SBI_DIMER_ALP_DETECTOR_2S_50FPS_Nuisance_DLI --export=ALL,REPO=$PWD Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_Nuisance_DLI.sh
# Example (5 s window, bounded pool):
#   cd /path/to/srm-and-sbi-dimer-alp
#   sbatch --job-name=SRM_AND_SBI_DIMER_ALP_DETECTOR_5S_50FPS_Nuisance_DLI --export=ALL,REPO=$PWD,POOL_MODE=bounded,TOTAL_TIME=5.0 Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_Nuisance_DLI.sh
# Example (two nodes, pool build sharded across both -- add --nodes=N; --gres is per node):
#   sbatch --nodes=2 --gres=gpu:4 --job-name=SRM_AND_SBI_DIMER_ALP_DETECTOR_2S_50FPS_Nuisance_DLI --export=ALL,REPO=$PWD Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_Nuisance_DLI.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI   # fallback; per-run --job-name (with timing_label) overrides this
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

POOL_MODE="${POOL_MODE:-unrestricted}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"
SPAN="${SPAN:-20}"

# GPUs PER NODE for sharding: SRM_AND_SBI_GPUS override, else the node's allocation,
# else 1. NODE COUNT comes from the Slurm allocation (SLURM_NNODES; 1 for a
# non-Slurm/local run), so the shard-worker count is world_size = NNODES * GPUS.
GPUS="${SRM_AND_SBI_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
NNODES="${SLURM_NNODES:-1}"
NDLI_PY="$REPO/Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py"
NDLI_ARGS=( --emit-template --pool-mode "$POOL_MODE"
            --total-time-seconds "$TOTAL_TIME" --experiment-span-seconds "$SPAN" )

echo "=== Nuisance_DLI (emit-template) | pool=${POOL_MODE} time=${TOTAL_TIME}s span=${SPAN}s nodes=${NNODES} gpus_per_node=${GPUS} world_size=$((NNODES * GPUS)) | node $(hostname) ==="

if [ "${NNODES:-1}" -gt 1 ]; then
    # Multi-node sharding: srun places ONE torchrun launcher per node, each spawning
    # GPUS local workers, so the (kind, cell) work shards round-robin across all
    # NNODES*GPUS ranks into per-rank pool shards on the shared output dir (a worker
    # that draws no cells writes no shard; no cross-rank communication); a single
    # no-GPU --merge pass then concatenates every shard into the cached pool + spec.
    MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
    MASTER_PORT="${MASTER_PORT:-29500}"
    echo "    multi-node: nnodes=$NNODES nproc_per_node=$GPUS rdzv=$MASTER_ADDR:$MASTER_PORT"
    srun --nodes="$NNODES" --ntasks-per-node=1 --cpu-bind=none \
        torchrun \
            --nnodes="$NNODES" \
            --nproc_per_node="$GPUS" \
            --rdzv-id="${SLURM_JOB_ID:-0}" \
            --rdzv-backend=c10d \
            --rdzv-endpoint="$MASTER_ADDR:$MASTER_PORT" \
            "$NDLI_PY" "${NDLI_ARGS[@]}"
    python -u "$NDLI_PY" "${NDLI_ARGS[@]}" --merge
elif [ "${GPUS:-1}" -gt 1 ]; then
    # Single-node sharding: one worker per GPU (torchrun --standalone) into per-rank
    # pool shards, then merge them into the cached pool + emitted spec (no GPU).
    torchrun --standalone --nproc_per_node="$GPUS" "$NDLI_PY" "${NDLI_ARGS[@]}"
    python -u "$NDLI_PY" "${NDLI_ARGS[@]}" --merge
else
    # Single GPU: the original path (builds the full pool, caches it, emits the spec; no merge).
    python -u "$NDLI_PY" "${NDLI_ARGS[@]}"
fi

echo "=== Nuisance_DLI (emit-template) complete ==="
echo "NEXT: edit the emitted _Nuisance_DLI_Spec.toml, then run the analysis with --build (single process; reuses the cached pool, no GPU)."

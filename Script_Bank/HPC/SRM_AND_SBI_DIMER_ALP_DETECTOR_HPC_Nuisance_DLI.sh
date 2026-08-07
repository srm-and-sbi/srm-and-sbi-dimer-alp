#!/bin/bash
# =============================================================================
# Slurm HPC application submitter: emit the Nuisance_DLI spec template (pool build).
# =============================================================================
# Runs the Detector Nuisance_DLI analysis in --emit-template mode: it reads the
# trained posterior + the .tif recordings under <data_bank>/Experiment/, builds the
# pooled posterior-sample pool (the GPU cost), caches it, and emits the value-based
# spec pre-filled with the calibrated-imaging percentiles for a person to finalize.
# Adapts to the allocated GPUs: with >1 GPU it shards the (kind, cell) work across
# one worker per GPU (torchrun) into per-rank pool shards, then a single-process,
# no-GPU --merge step concatenates the shards into the cached pool + spec; with 1 GPU
# it is the original single-process path. A worker that draws no cells writes no shard.
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

# GPU count for sharding: SRM_AND_SBI_GPUS override, else allocated GPUs, else 1.
GPUS="${SRM_AND_SBI_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
NDLI_PY="$REPO/Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py"
NDLI_ARGS=( --emit-template --pool-mode "$POOL_MODE"
            --total-time-seconds "$TOTAL_TIME" --experiment-span-seconds "$SPAN" )

echo "=== Nuisance_DLI (emit-template) | pool=${POOL_MODE} time=${TOTAL_TIME}s span=${SPAN}s gpus=${GPUS} | node $(hostname) ==="

if [ "${GPUS:-1}" -gt 1 ]; then
    # Shard the (kind, cell) work across $GPUS workers (one GPU each) into per-rank
    # pool shards, then merge them into the cached pool + emitted spec (no GPU).
    torchrun --standalone --nproc_per_node="$GPUS" "$NDLI_PY" "${NDLI_ARGS[@]}"
    python -u "$NDLI_PY" "${NDLI_ARGS[@]}" --merge
else
    # Single GPU: the original path (builds the full pool, caches it, emits the spec; no merge).
    python -u "$NDLI_PY" "${NDLI_ARGS[@]}"
fi

echo "=== Nuisance_DLI (emit-template) complete ==="
echo "NEXT: edit the emitted _Nuisance_DLI_Spec.toml, then run the analysis with --build (single process; reuses the cached pool, no GPU)."

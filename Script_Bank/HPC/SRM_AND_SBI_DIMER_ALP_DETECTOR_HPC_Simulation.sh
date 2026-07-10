#!/bin/bash
# =============================================================================
# Detector-workflow generation: diffusion-only RDS (B1) -> imaging-from-theta DLI
# (B2), many tasks packed per node. Part of the Detector calibration workflow's
# OWN committed submission machinery (Script_Bank/HPC/) — a complete
# workflow parallel to the canonical pipeline, never wired into the canonical
# Submit.sh dispatcher.
# =============================================================================
# Mirrors the canonical SRM_AND_SBI_DIMER_ALP_HPC_Simulation.sh (packed
# background tasks per node; multi-node fan-out via an --array of single-node
# jobs), with one Detector difference: it drives the Detector B1/B2 entry points
# and passes a per-split --seed, since the Detector draws its RDS nuisance and
# imaging theta from SeedSequence(seed).spawn(2) (decorrelated streams). Use
# distinct per-split seeds so TRAIN/TEST/EVAL theta are independent.
#
#     tid = TASK_OFFSET + SLURM_ARRAY_TASK_ID * SLURM_NTASKS_PER_NODE + k
#
# Knobs (--export / CLI): SPLIT (train|test|eval), SEED, TASK_SIMS, TOTAL_TIME,
# TASK_OFFSET, TASK_COUNT (tasks this submission generates; default =
# --ntasks-per-node). SIM_STAGE selects which stage(s) run per task:
# both (default) | rds | dli (dli reuses existing trajectories). Debug knobs
# (default off, production-identical when unset): VERBOSE=1 adds per-sim detail.
#
# ALWAYS submit with --array (one element per node, --array=0-0 for a single
# node). Submit from the repo root or forward REPO explicitly (Slurm spools this
# script to /var/spool, so its own path is unreliable).
# --job-name follows the data-file naming convention
# SRM_AND_SBI_DIMER_ALP_DETECTOR_<timing_label>_Simulation_<SPLIT>.
# Example (one node, 2 s, TRAIN 16 tasks x 10 sims, seed 42):
#   cd /path/to/srm-and-sbi-dimer-alp
#   sbatch --array=0-0 --ntasks-per-node=16 --job-name=SRM_AND_SBI_DIMER_ALP_DETECTOR_2S_50FPS_Simulation_TRAIN \
#     --export=ALL,REPO=$PWD,SPLIT=train,SEED=42,TASK_COUNT=16,TASK_SIMS=10,TOTAL_TIME=2.0 \
#     Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_Simulation.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_DIMER_ALP_DETECTOR_Simulation   # fallback; per-run --job-name overrides
#SBATCH --partition=YOUR_PARTITION   # set to your cluster's CPU partition (or override on the sbatch line)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=4400
#SBATCH --extra-node-info=2:20:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=%x_%A_Node_%a.out   # submit-directory; a controller may override this via MON_OUT

set -eo pipefail

# Locate the repo root robustly (Slurm spools this script to /var/spool). Resolve
# REPO from, in order: an explicit REPO, the Slurm submit dir, or this script's
# own location (../.. — this file lives at REPO/Script_Bank/HPC/).
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
    echo "FATAL: cannot locate the srm-and-sbi-dimer-alp repo root. Submit with an explicit REPO, e.g.:" >&2
    echo "    cd /path/to/srm-and-sbi-dimer-alp && sbatch --export=ALL,REPO=\$PWD,... <this-script>" >&2
    exit 1
}
cd "$REPO"
PRIME="$REPO/Script_Bank/Prime"

# Per-machine HPC config (gitignored; copy from Script_Bank/HPC/hpc_local.env.example):
# sets MACHINE_PROFILE / CONDA_SETUP / MON. Sourced from the canonical HPC dir so
# one file serves both the canonical and Detector machinery.
HPC_ENV="${HPC_ENV:-$REPO/Script_Bank/HPC/hpc_local.env}"
if [ -f "$HPC_ENV" ]; then . "$HPC_ENV"; fi
MON="${MON:-$HOME/process_monitoring}"

source "${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate SRM_AND_SBI_ENVY_V0
export MACHINE_PROFILE="${MACHINE_PROFILE:?set MACHINE_PROFILE (via hpc_local.env or --export) to a profile in your machine_profiles.toml}"

SPLIT="${SPLIT:-train}"
SEED="${SEED:?set SEED (use distinct per-split seeds so TRAIN/TEST/EVAL theta are independent)}"
TASK_SIMS="${TASK_SIMS:-1000}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"
TASK_OFFSET="${TASK_OFFSET:-0}"
PER_NODE="${SLURM_NTASKS_PER_NODE:-8}"
TASK_COUNT="${TASK_COUNT:-$PER_NODE}"
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"

SIM_FLAGS=""
[ "${VERBOSE:-0}" = 1 ] && SIM_FLAGS="$SIM_FLAGS --verbose"

# SIM_STAGE: both (default) | rds | dli. dli-only re-renders from existing trajectories.
SIM_STAGE="${SIM_STAGE:-both}"
case "$SIM_STAGE" in both|rds|dli) ;; *) echo "bad SIM_STAGE=$SIM_STAGE (use both|rds|dli)" >&2; exit 1;; esac

start=$(( TASK_OFFSET + ARRAY_ID * PER_NODE ))
end=$(( TASK_OFFSET + TASK_COUNT ))
echo "=== Detector Simulation | $(hostname) | split=${SPLIT} seed=${SEED} sims=${TASK_SIMS} time=${TOTAL_TIME}s | tasks [${start}..$(( end - 1 ))] ==="

declare -a PIDS=()
for (( tid=start; tid < start + PER_NODE && tid < end; tid++ )); do
    out="${MON}/${SLURM_JOB_NAME}_${JOB_TAG}_Node_${ARRAY_ID}_Task_${tid}.out"
    ( rc=0
      if [ "$SIM_STAGE" != dli ]; then
        python -u "$PRIME/SRM_AND_SBI_DIMER_ALP_DETECTOR_Simulation_RDS.py" \
            --task-id "$tid" --task-simulations "$TASK_SIMS" \
            --total-time-seconds "$TOTAL_TIME" --split "$SPLIT" --seed "$SEED" $SIM_FLAGS || rc=1
      fi
      if [ "$rc" = 0 ] && [ "$SIM_STAGE" != rds ]; then
        python -u "$PRIME/SRM_AND_SBI_DIMER_ALP_DETECTOR_Simulation_DLI.py" \
            --task-id "$tid" --task-simulations "$TASK_SIMS" \
            --total-time-seconds "$TOTAL_TIME" --split "$SPLIT" --seed "$SEED" $SIM_FLAGS || rc=1
      fi
      exit "$rc" ) > "$out" 2>&1 &
    PIDS+=( "$!" )
    echo "  -> Task ${tid} (${SPLIT}) -> $(basename "$out")"
done

rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
echo "=== Detector Simulation | array element ${ARRAY_ID} (${SPLIT}) complete (rc=${rc}) ==="
exit "$rc"

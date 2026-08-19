#!/bin/bash
# =============================================================================
# Slurm HPC submitter: experimental-versus-synthetic embedding-space distance.
# =============================================================================
# Runs the embedding-distance analysis for either workflow (WORKFLOW=biology|detector)
# over one shared engine. The analysis embeds every experimental window and every
# synthetic EVAL video through the trained network, then scores the gap with MMD
# (permutation null) and C2ST, both blocked by recording.
#
# SINGLE-GPU BY DESIGN, WHOLE-NODE BY ALLOCATION. The engine resolves one device
# (`resolve_topology().device`) and does not shard -- there is no per-rank split and no
# merge pass. The allocation is still a whole node (gres=gpu:4), matching JUPITER's
# node-granularity convention and the known-good detector submission (job 1320689,
# booster, 1 node, gres/gpu=4, cpu=288, 10 EVAL tasks -> 10000 synthetic windows,
# COMPLETED in 34:08). Do not read the 4 GPUs as data parallelism.
#
# This is a post-hoc ANALYSIS, not a canonical stage: it is deliberately NOT wired into
# SRM_AND_SBI_DIMER_ALP_HPC_Submit.sh, and is submitted directly with sbatch.
#
# Overridable via --export: WORKFLOW (biology|detector), EVAL_TASKS, TOTAL_TIME, KINDS,
#   N_PERMUTATIONS, SPAN (experimental recording span, s), EXTRA (extra CLI flags).
#
# Example (biology, the detector's known-good scale):
#   cd /path/to/srm-and-sbi-dimer-alp
#   sbatch --job-name=SRM_AND_SBI_DIMER_ALP_2S_50FPS_Embedding_Space_Distance \
#          --export=ALL,REPO=$PWD,WORKFLOW=biology,EVAL_TASKS=10 \
#          Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Embedding_Space_Distance.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_DIMER_ALP_Embedding_Space_Distance
#SBATCH --partition=booster
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=64
#SBATCH --time=02:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=%x_%A.out

set -eo pipefail

# Slurm runs a SPOOLED COPY of this script, so BASH_SOURCE is unreliable; resolve the
# repo from an explicit REPO, the submit directory, or this script's location.
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
    echo "FATAL: cannot locate the srm-and-sbi-dimer-alp repo root. Submit with an" >&2
    echo "  explicit REPO, e.g. --export=ALL,REPO=\$PWD,..." >&2; exit 1; }
cd "$REPO"

HPC_ENV="${HPC_ENV:-$REPO/Script_Bank/HPC/hpc_local.env}"
if [ -f "$HPC_ENV" ]; then . "$HPC_ENV"; fi

source "${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate SRM_AND_SBI_ENVY_V0
export MACHINE_PROFILE="${MACHINE_PROFILE:?set MACHINE_PROFILE (via hpc_local.env or --export)}"

WORKFLOW="${WORKFLOW:-biology}"
EVAL_TASKS="${EVAL_TASKS:-10}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"
KINDS="${KINDS:-MET-FAB,MET-INLB}"
N_PERMUTATIONS="${N_PERMUTATIONS:-1000}"
SPAN="${SPAN:-20}"

# ABSOLUTE paths, as in the stage wrappers. hpc_local.env cd's into the GPU coredump
# directory after this script's own `cd "$REPO"`, so a relative script path resolves against
# that directory instead of the repo and python exits "No such file or directory".
case "$WORKFLOW" in
    biology)  SCRIPT="$REPO/Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Embedding_Space_Distance.py" ;;
    detector) SCRIPT="$REPO/Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Embedding_Space_Distance.py" ;;
    *) echo "FATAL: WORKFLOW must be 'biology' or 'detector', got '$WORKFLOW'." >&2; exit 1 ;;
esac

echo "=== Embedding_Space_Distance | workflow=${WORKFLOW} eval_tasks=${EVAL_TASKS} time=${TOTAL_TIME}s kinds=${KINDS} span=${SPAN}s perms=${N_PERMUTATIONS} | node $(hostname) ==="
echo "    single-GPU engine on a whole-node allocation (gres=gpu:4); no sharding, no merge."

python -u "$SCRIPT" \
    --total-time-seconds "$TOTAL_TIME" \
    --eval-tasks "$EVAL_TASKS" \
    --kinds "$KINDS" \
    --experiment-span-seconds "$SPAN" \
    --n-permutations "$N_PERMUTATIONS" \
    ${EXTRA:-}

echo "=== Embedding_Space_Distance complete ==="

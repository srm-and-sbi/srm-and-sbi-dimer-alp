#!/bin/bash
# =============================================================================
# Slurm HPC submitter: posterior-calibration diagnostic (SBC / coverage / TARP / L-C2ST).
# =============================================================================
# Scores a trained posterior's calibration on the held-out EVAL set. An Analysis
# diagnostic, not a pipeline stage (never wired into Submit.sh), but it shares the
# Evaluation stage's shard-then-merge execution so it runs at scale: with >1 node it
# shards the EVAL set across one worker per GPU on EVERY node (srun places one torchrun
# launcher per node), each writing its own shard to the shared filesystem, then a single
# --merge pass concatenates them and runs the calibration statistics (SBC/TARP/L-C2ST are
# global) on the full set; with 1 node and >1 GPU it shards across that node's GPUs
# (torchrun --standalone) then merges; with 1 GPU it is the single-process path (writes
# the report directly, no merge). --gres is per node, so --nodes=N --gres=gpu:G gives N*G
# shard workers (world_size = N*G). The sharding is embarrassingly parallel (no cross-rank
# communication -- the c10d rendezvous torchrun sets up is unused here, shared with
# training only for one uniform GPU-binding path); workers just need the shared output dir
# for the merge. Reads the trained posterior + EVAL data, writes the calibration report
# (Posit/..._Posterior_Calibration/).
#
# WORKFLOW selects the twin shim: biology (10 reaction-diffusion params) or detector (6
# imaging params). Overridable via --export: WORKFLOW (biology|detector), EVAL_TASKS,
# POSTERIOR_SAMPLES (L per video), TESTS (comma-list; NOTE: commas break --export, so pass
# it only via a quoted --export or leave the all-four default), STRATIFY (all|none|KEY),
# MAX_SIMS (cap videos/task; 0 = all), POOL_MODE (bounded|unrestricted), TOTAL_TIME,
# MIN_STRATUM, LC2ST_N_EVAL, LC2ST_NULL_TRIALS, N_JOBS (stratum-loop worker processes;
# default auto = largest power of two up to 16 fitting the allocated cores -- note the
# statistics run in the SINGLE-PROCESS merge step, so this, not the GPU count, sets how
# long a fine stratification takes), SRM_AND_SBI_GPUS (cap GPUs used).
# Non-deterministic (no seed). Submit from the repo root and forward REPO (Slurm spools
# this script, so the child must be told where the repo is): --export=ALL,REPO=$PWD,...
#
# Example (single node, all GPUs, biology, full run):
#   cd /path/to/srm-and-sbi-dimer-alp
#   sbatch --job-name=SRM_AND_SBI_DIMER_ALP_2S_50FPS_Posterior_Calibration \
#     --export=ALL,REPO=$PWD,EVAL_TASKS=25,POSTERIOR_SAMPLES=1000 \
#     Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Posterior_Calibration.sh
# Example (two nodes, EVAL sharded across both -- add --nodes=N; --gres is per node):
#   sbatch --nodes=2 --gres=gpu:4 --export=ALL,REPO=$PWD,EVAL_TASKS=25,POSTERIOR_SAMPLES=1000 \
#     Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Posterior_Calibration.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_DIMER_ALP_Posterior_Calibration   # fallback; per-run --job-name (with timing_label) overrides this
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=480G
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=%x_%A.out

set -eo pipefail

# Locate the repo root robustly (Slurm runs a SPOOLED COPY from /var/spool, so
# BASH_SOURCE is unreliable): try REPO, the submit dir, or this script's location;
# accept the first that actually contains the package.
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
    echo "  explicit REPO, e.g.: cd /path/to/srm-and-sbi-dimer-alp && sbatch --export=ALL,REPO=\$PWD,... <this>" >&2
    exit 1
}
cd "$REPO"

# Per-machine HPC config (gitignored; sets MACHINE_PROFILE / CONDA_SETUP / etc.).
HPC_ENV="${HPC_ENV:-$REPO/Script_Bank/HPC/hpc_local.env}"
if [ -f "$HPC_ENV" ]; then . "$HPC_ENV"; fi

source "${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate SRM_AND_SBI_ENVY_V0
export MACHINE_PROFILE="${MACHINE_PROFILE:?set MACHINE_PROFILE (via hpc_local.env or --export) to a profile in your machine_profiles.toml}"

WORKFLOW="${WORKFLOW:-biology}"
EVAL_TASKS="${EVAL_TASKS:-4}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"
POOL_MODE="${POOL_MODE:-bounded}"
POSTERIOR_SAMPLES="${POSTERIOR_SAMPLES:-1000}"
TESTS="${TESTS:-sbc,coverage,tarp,lc2st}"
STRATIFY="${STRATIFY:-all}"
MAX_SIMS="${MAX_SIMS:-0}"

if [ "$WORKFLOW" = "detector" ]; then
    CAL_PY="$REPO/Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Calibration.py"
else
    CAL_PY="$REPO/Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Posterior_Calibration.py"
fi

# GPUs PER NODE for sharding: SRM_AND_SBI_GPUS override, else the node's allocation, else
# 1. NODE COUNT comes from the Slurm allocation (SLURM_NNODES), so world_size = NNODES*GPUS.
GPUS="${SRM_AND_SBI_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
NNODES="${SLURM_NNODES:-1}"
# No worker cap by task count: the draw loop shards at VIDEO granularity, so every
# allocated GPU gets work regardless of EVAL_TASKS.

CAL_ARGS=( --total-time-seconds "$TOTAL_TIME" --eval-tasks "$EVAL_TASKS" --pool-mode "$POOL_MODE"
           --posterior-samples "$POSTERIOR_SAMPLES" --tests "$TESTS" --stratify "$STRATIFY" )
[ "${MAX_SIMS:-0}" -gt 0 ] && CAL_ARGS+=( --max-sims "$MAX_SIMS" )
[ -n "${MIN_STRATUM:-}" ] && CAL_ARGS+=( --min-stratum "$MIN_STRATUM" )
[ -n "${N_JOBS:-}" ] && CAL_ARGS+=( --n-jobs "$N_JOBS" )
[ -n "${LC2ST_N_EVAL:-}" ] && CAL_ARGS+=( --lc2st-n-eval "$LC2ST_N_EVAL" )
[ -n "${LC2ST_NULL_TRIALS:-}" ] && CAL_ARGS+=( --lc2st-null-trials "$LC2ST_NULL_TRIALS" )

echo "=== Posterior Calibration | workflow=${WORKFLOW} eval_tasks=${EVAL_TASKS} L=${POSTERIOR_SAMPLES} tests=${TESTS} stratify=${STRATIFY} pool=${POOL_MODE} time=${TOTAL_TIME}s max_sims=${MAX_SIMS} nodes=${NNODES} gpus_per_node=${GPUS} world_size=$((NNODES * GPUS)) seed=None | node $(hostname) ==="

# torch-elastic's exit barrier defaults to 300 s: the ranks that finish first wait only five
# minutes for the rest and then tear down the rendezvous -- which KILLS any rank still
# working, discarding its results. The sharded stages are embarrassingly parallel and
# routinely skewed (a rank drawing two tasks takes twice as long as one drawing a single
# task), so five minutes is far tighter than the real spread and the teardown destroys
# completed work. Raise it well past any plausible skew; the job wall time is the real bound.
export TORCHELASTIC_EXIT_BARRIER_TIMEOUT="${EXIT_BARRIER:-3600}"

if [ "${NNODES:-1}" -gt 1 ]; then
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
            "$CAL_PY" "${CAL_ARGS[@]}"
    python -u "$CAL_PY" "${CAL_ARGS[@]}" --merge
elif [ "${GPUS:-1}" -gt 1 ]; then
    # Single-node sharding: one worker per GPU (torchrun --standalone), then merge.
    torchrun --standalone --nproc_per_node="$GPUS" "$CAL_PY" "${CAL_ARGS[@]}"
    python -u "$CAL_PY" "${CAL_ARGS[@]}" --merge
else
    # Single GPU: the single-process path (writes the report directly; no merge).
    python -u "$CAL_PY" "${CAL_ARGS[@]}"
fi

echo "=== Posterior Calibration complete ==="

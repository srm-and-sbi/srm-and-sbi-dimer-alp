#!/bin/bash
# =============================================================================
# Slurm HPC training submitter: train the posterior on TRAIN, select on TEST.
# =============================================================================
# Adapts to the allocation: with >1 node it trains data-parallel ACROSS NODES
# (srun places one torchrun launcher per node, each spawning one process per GPU;
# a c10d rendezvous binds all ranks into one process group); with 1 node and >1
# GPU it is single-node data-parallel (torchrun --standalone, one process per
# GPU); with 1 GPU it is the original single-GPU path. --gres is per node, so
# --nodes=N --gres=gpu:G allocates N*G GPUs and world_size = N*G. The batch rule
# keys off world_size, NOT node count: SyncBatchNorm shares the BatchNorm stats
# and the loss is all-reduced across EVERY rank on EVERY node, so --batch-size
# stays PER-RANK and the effective batch is batch*world_size whether the ranks sit
# on one node or several. Node count comes from the allocation (Submit.sh NODES ->
# sbatch --nodes; SLURM_NNODES), not an --export knob. The `gpu` partition
# allocates a whole node (8 GPUs on the reference cluster; 4 GH200 on JUPITER).
# Overridable via --export: TRAIN_TASKS, TEST_TASKS, EPOCHS, TOTAL_TIME, BATCH, NUM_WORKERS,
#   HEARTBEAT (within-epoch progress: a line every N batches; default ~4/epoch),
#   LR (per-run starting/peak learning rate, forwarded as --learning-rate; unset =
#     the default peak, learning_rate_minimum*max_factor = 1.28e-03),
#   RESURRECT (set 1 to load the existing checkpoint and continue training from it
#     -- continue a run stopped by the wall, or add epochs; unset = fresh run),
#   SRM_AND_SBI_GPUS (cap the GPUs used; default = all allocated; set 1 to force
#     the original single-GPU path even on a multi-GPU allocation),
#   EXIT_BARRIER (seconds; raises torch-elastic's 300 s exit barrier so straggler
#     ranks are not killed; default 3600 -- the job wall time is the real bound).
# Also honored from the environment (read directly by the Python):
#   SRM_AND_SBI_NO_SYNC_BN=1 -> skip SyncBatchNorm under DDP (each rank keeps its
#     own local batch statistics). DEFAULT (unset) keeps SyncBatchNorm ON, which
#     is the correct, validated choice; the opt-out is faster but changes the
#     batch statistics, so re-validate posterior recovery before using it.
# BATCH is optional: leave unset to use the script default (PARAMETERS batch_size,
# 32). Long videos (e.g. 10 s = 500 frames) need a smaller batch -- one early
# conv3d activation is batch * ~1 GiB at 500 frames, so batch 32 OOMs a 64 GB GPU.
# Training is non-deterministic (no seed; consistent with generation).
# Submit from the repo root and forward REPO: Slurm spools this script to
# /var/spool, so the child must be told where the repo is (--export=ALL,REPO=$PWD).
# --job-name follows the data-file naming convention
# SRM_AND_SBI_DIMER_ALP_DETECTOR_<timing_label>_Inference; with no TOTAL_TIME set the
# launcher default (2.0 s) gives timing_label 2S_50FPS, so swap the token (e.g.
# 5S_50FPS) whenever you pass TOTAL_TIME=5.0.
# Example (single node):
#   cd /path/to/srm-and-sbi-dimer-alp
#   sbatch --job-name=SRM_AND_SBI_DIMER_ALP_DETECTOR_2S_50FPS_Inference --export=ALL,REPO=$PWD,TRAIN_TASKS=8,TEST_TASKS=2,EPOCHS=50 Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_Inference.sh
# Example (two nodes, data-parallel across both -- add --nodes=N; --gres is per node):
#   sbatch --nodes=2 --gres=gpu:4 --job-name=SRM_AND_SBI_DIMER_ALP_DETECTOR_2S_50FPS_Inference --export=ALL,REPO=$PWD,TRAIN_TASKS=8,TEST_TASKS=2,EPOCHS=50 Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_Inference.sh
# Quick smoke on a shorter-lived test GPU partition (set <gpu-partition> to your cluster's; matches VALIDATION.md section 2.5):
#   sbatch --partition=<gpu-partition> --gres=gpu:1 --time=01:00:00 --job-name=SRM_AND_SBI_DIMER_ALP_DETECTOR_2S_50FPS_Inference --export=ALL,REPO=$PWD,TRAIN_TASKS=16,TEST_TASKS=4,EPOCHS=5,BATCH=8 Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_Inference.sh
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_DIMER_ALP_DETECTOR_Inference   # fallback; per-run --job-name (with timing_label) overrides this
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

TRAIN_TASKS="${TRAIN_TASKS:-8}"
TEST_TASKS="${TEST_TASKS:-2}"
EPOCHS="${EPOCHS:-50}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"
BATCH="${BATCH:-}"   # empty -> script default (PARAMETERS batch_size, 32)
NUM_WORKERS="${NUM_WORKERS:-}"   # empty -> machine-profile / auto (capped) worker budget; else per-run TOTAL override
HEARTBEAT="${HEARTBEAT:-}"   # empty -> ~4 within-epoch progress lines/epoch; else a line every N batches
LR="${LR:-}"   # per-run starting (peak) LR; empty -> Prime default (learning_rate_minimum*max_factor = 1.28e-03)
RESURRECT="${RESURRECT:-}"   # set 1 to load the existing checkpoint and continue; unset/0 -> fresh run

BATCH_ARG=()
[ -n "$BATCH" ] && BATCH_ARG=(--batch-size "$BATCH")
NUM_WORKERS_ARG=()
[ -n "$NUM_WORKERS" ] && NUM_WORKERS_ARG=(--num-workers "$NUM_WORKERS")
HEARTBEAT_ARG=()
[ -n "$HEARTBEAT" ] && HEARTBEAT_ARG=(--heartbeat "$HEARTBEAT")
LR_ARG=()
[ -n "$LR" ] && LR_ARG=(--learning-rate "$LR")   # forward the starting-LR override; unset -> default peak
RESURRECT_ARG=()
[ "$RESURRECT" = 1 ] && RESURRECT_ARG=(--resurrect)   # only the value 1 resumes; unset/0/other = fresh

# GPUs PER NODE for data-parallel training: SRM_AND_SBI_GPUS override, else the
# node's allocation, else 1. NODE COUNT comes from the Slurm allocation
# (SLURM_NNODES; 1 for a non-Slurm/local run), so world_size = NNODES * GPUS.
GPUS="${SRM_AND_SBI_GPUS:-${SLURM_GPUS_ON_NODE:-1}}"
NNODES="${SLURM_NNODES:-1}"
INFER_PY="$REPO/Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_DETECTOR_Inference.py"
INFER_ARGS=( --tasks "$TRAIN_TASKS" --test-tasks "$TEST_TASKS" --epochs "$EPOCHS"
             --total-time-seconds "$TOTAL_TIME" "${BATCH_ARG[@]}" "${NUM_WORKERS_ARG[@]}" "${HEARTBEAT_ARG[@]}" "${RESURRECT_ARG[@]}" "${LR_ARG[@]}" )

echo "=== Inference | train_tasks=${TRAIN_TASKS} test_tasks=${TEST_TASKS} epochs=${EPOCHS} time=${TOTAL_TIME}s batch=${BATCH:-default} resurrect=${RESURRECT:-0} nodes=${NNODES} gpus_per_node=${GPUS} world_size=$((NNODES * GPUS)) seed=None | node $(hostname) ==="

# torch-elastic's exit barrier defaults to 300 s: the ranks that finish first wait only five
# minutes for the rest and then tear down the rendezvous -- which KILLS any rank still
# working, discarding its results. The sharded stages are embarrassingly parallel and
# routinely skewed (a rank drawing two tasks takes twice as long as one drawing a single
# task), so five minutes is far tighter than the real spread and the teardown destroys
# completed work. Raise it well past any plausible skew; the job wall time is the real bound.
export TORCHELASTIC_EXIT_BARRIER_TIMEOUT="${EXIT_BARRIER:-3600}"

if [ "${NNODES:-1}" -gt 1 ]; then
    # Multi-node data-parallel: srun places ONE torchrun launcher per node, each
    # spawning GPUS local worker processes; a c10d rendezvous at the first node
    # binds all NNODES*GPUS ranks into one process group. torchrun exports
    # RANK/LOCAL_RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT, which resolve_topology
    # reads; --cpu-bind=none lets each node's GPUS children spread over its cores.
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
            "$INFER_PY" "${INFER_ARGS[@]}"
elif [ "${GPUS:-1}" -gt 1 ]; then
    # Single-node data-parallel: one worker per GPU (DistributedDataParallel via torchrun).
    torchrun --standalone --nproc_per_node="$GPUS" "$INFER_PY" "${INFER_ARGS[@]}"
else
    # Single GPU: the original path.
    python -u "$INFER_PY" "${INFER_ARGS[@]}"
fi

echo "=== Inference complete ==="

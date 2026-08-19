#!/usr/bin/env bash
# =============================================================================
# Unified dry-run-first submitter for all four DIMER HPC stages.
# =============================================================================
# One front end for Simulation / Inference / Evaluation / Experiment. It builds
# the exact sbatch command -- REPO forwarded explicitly, the data-pattern
# --job-name with the rendered timing_label, and the per-stage --export -- then
# PRINTS it (DRYRUN=1, the default) or SUBMITS it (DRYRUN=0). The recipe, the
# naming convention, and the comma-split-safe --export are built by the tool, so
# they are never retyped (and never wrong) at submit time.
#
# This mirrors the dry-run-default contract of the generation controller
# (SRM_AND_SBI_DIMER_ALP_HPC_Generate_Controller.sh): DRYRUN=1 prints, DRYRUN=0
# submits. Use the controller for the full rolling + gated generation CAMPAIGN;
# use this helper for any single-job submission of any stage (including a one-off
# Simulation array job).
#
# Usage:
#   [DRYRUN=0] [PART=.. GPU_PART=.. ACCT=.. TIME=.. ARRAY=.. NTPN=.. CPT=.. GRES=.. MON_OUT=..] \
#     bash SRM_AND_SBI_DIMER_ALP_HPC_Submit.sh <stage> [KEY=VALUE ...]
#
#   <stage> = simulation | inference | evaluation | experiment
#   KEY=VALUE pairs are the stage's --export knobs (see each stage script header);
#   anything not passed falls back to that stage script's own default:
#     simulation  : SPLIT TASK_OFFSET TASK_COUNT TASK_SIMS TOTAL_TIME SKIN_FACTOR
#                   (SKIN_FACTOR = ReaDDy neighbor-list skin as a multiple of the
#                   particle diameter; RDS-only, performance not physics; unset =
#                   the code default 10x = 100 nm)
#     inference   : TRAIN_TASKS TEST_TASKS EPOCHS TOTAL_TIME BATCH LR HEARTBEAT RESURRECT
#                   (RESURRECT=1 continues training from the existing checkpoint;
#                   LR = per-run starting/peak learning rate, unset = the stage
#                   script's default peak)
#     evaluation  : EVAL_TASKS SUMMARY POOL_MODE TOTAL_TIME
#     experiment  : KINDS MAX_CELLS CHUNK_STEP SUMMARY POOL_MODE TOTAL_TIME
#
# sbatch-level overrides (env, optional -- unset = the stage script's baked #SBATCH):
#   PART      CPU partition for Simulation (its baked --partition is a placeholder,
#             so PART is REQUIRED for a live Simulation submission)
#   GPU_PART  partition for the GPU stages (e.g. gpu_test for a smoke, gpu for prod)
#   ACCT      Slurm account, if your cluster requires one
#   TIME      --time override (else the baked #SBATCH --time)
#   DEP       --dependency override (e.g. afterany:<jobid> -- chain resurrect jobs so
#             each starts after the previous ENDS, regardless of success/failure)
#   ARRAY     Simulation only: --array spec (default 0-0 = one node)
#   NTPN CPT  Simulation only: --ntasks-per-node / --cpus-per-task (else baked)
#   GRES      GPU stages: --gres override (else baked gpu:8); --gres is per node
#   NODES     GPU stages: --nodes override for multi-node (else the baked --nodes=1).
#             --gres is per node, so NODES=2 GRES=gpu:4 spans 2*4=8 ranks (world_size):
#             inference trains data-parallel across them, evaluation/experiment shard
#             the work across them. Inference --batch-size stays per-rank.
#   MON_OUT   batch-log --output dir (else the baked submit-directory --output)
#
# timing_label is rendered from TOTAL_TIME exactly as PARAMETERS.simulation.timing.label
# ("{duration}S_50FPS", duration via %g: 2.0 -> 2, 5.0 -> 5, 2.5 -> 2.5; FPS pinned 50).
# The KINDS comma-split trap (Slurm splits --export on commas) is avoided by carrying
# KINDS through the exported environment via ALL, never inside the explicit --export.
#
# Examples (all DRY-RUN by default -- print the sbatch line, submit nothing):
#   bash .../Submit.sh inference TOTAL_TIME=5.0 TRAIN_TASKS=400 TEST_TASKS=100 EPOCHS=25
#   DRYRUN=0 GPU_PART=gpu bash .../Submit.sh inference TOTAL_TIME=5.0 TRAIN_TASKS=400 TEST_TASKS=100 EPOCHS=25
#   DRYRUN=0 GPU_PART=gpu_test bash .../Submit.sh inference TOTAL_TIME=5.0 TRAIN_TASKS=100 EPOCHS=10 RESURRECT=1  # continue a wall-stopped run
#   PART=test bash .../Submit.sh simulation SPLIT=train TASK_COUNT=8 TASK_SIMS=1000 TOTAL_TIME=2.0
#   bash .../Submit.sh evaluation TOTAL_TIME=5.0 EVAL_TASKS=20 POOL_MODE=bounded
#   bash .../Submit.sh experiment TOTAL_TIME=2.0 SUMMARY=both KINDS=ALP,BET
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Per-machine HPC config (gitignored; copy from hpc_local.env.example): may set
# MACHINE_PROFILE / CONDA_SETUP / PART / ACCT / MON_OUT / etc. Sourced first so a
# machine can pin all of its divergence in one place.
HPC_ENV="${HPC_ENV:-$HERE/hpc_local.env}"
if [ -f "$HPC_ENV" ]; then . "$HPC_ENV"; fi

# Self-derive + validate REPO (this script lives at REPO/Script_Bank/HPC/). Honor
# an explicit REPO override; otherwise resolve from this script's own location.
# Fail loud if the result is not a real srm-and-sbi-dimer-alp checkout.
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
if [ ! -f "$REPO/pyproject.toml" ] || [ ! -d "$REPO/srm_and_sbi_dimer_alp" ]; then
    echo "FATAL: '$REPO' is not a srm-and-sbi-dimer-alp repo root (needs pyproject.toml + srm_and_sbi_dimer_alp/)." >&2
    exit 1
fi

DRYRUN="${DRYRUN:-1}"   # 1 = print only (default); 0 = live submit

STAGE="${1:-}"
[ -n "$STAGE" ] || {
    echo "usage: [DRYRUN=0] bash ${BASH_SOURCE[0]##*/} <simulation|inference|evaluation|experiment> [KEY=VALUE ...]" >&2
    exit 1
}
shift

# Apply KEY=VALUE overrides into the environment. They reach the job via the
# stage script's own ${VAR:-default} reads; single-value knobs are also listed
# explicitly in --export below (a multi-value KINDS is carried via ALL instead).
for kv in "$@"; do
    case "$kv" in
        [A-Za-z_]*=*) export "${kv%%=*}=${kv#*=}" ;;
        *) echo "FATAL: bad argument '$kv' (expected KEY=VALUE)." >&2; exit 1 ;;
    esac
done

TOTAL_TIME="${TOTAL_TIME:-2.0}"; export TOTAL_TIME
case "$TOTAL_TIME" in
    ''|*[!0-9.]*|*.*.*|.) echo "FATAL: TOTAL_TIME='$TOTAL_TIME' is not a valid number (e.g. 2.0, 5.0)." >&2; exit 1 ;;
esac
timing_label="$(LC_ALL=C printf '%gS_50FPS' "$TOTAL_TIME")"
export REPO   # also carried by ALL; listed explicitly in --export for clarity

# Build the explicit --export list, appending only knobs that are actually set
# (an unset knob falls back to the stage script's own default). _add never adds a
# comma-bearing value -- those (KINDS) ride the exported environment via ALL.
declare -a EXPORT_PARTS=( "ALL" "REPO=$REPO" )
_add(){ local k="$1"; [ -n "${!k:-}" ] && EXPORT_PARTS+=( "$k=${!k}" ); }

declare -a SB=()          # sbatch flags
SUBMIT_SCRIPT=""
JOBNAME=""
KINDS_NOTE=""

case "$STAGE" in
  simulation)
    SUBMIT_SCRIPT="$REPO/Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Simulation.sh"
    SPLIT="${SPLIT:-train}"; export SPLIT
    case "$SPLIT" in train|test|eval) ;; *) echo "FATAL: SPLIT='$SPLIT' (use train|test|eval)." >&2; exit 1 ;; esac
    split_uc="$(echo "$SPLIT" | tr '[:lower:]' '[:upper:]')"
    JOBNAME="SRM_AND_SBI_DIMER_ALP_${timing_label}_Simulation_${split_uc}"
    _add SPLIT; _add TASK_OFFSET; _add TASK_COUNT; _add TASK_SIMS; _add TOTAL_TIME; _add SKIN_FACTOR
    SB+=( --array="${ARRAY:-0-0}" )   # always array-submit so %a is a clean node number
    [ -n "${NTPN:-}" ] && SB+=( --ntasks-per-node="$NTPN" )
    [ -n "${CPT:-}" ]  && SB+=( --cpus-per-task="$CPT" )
    # Simulation's baked --partition is a placeholder; PART is required for a live submit.
    if [ -n "${PART:-}" ]; then SB+=( --partition="$PART" )
    elif [ "$DRYRUN" != 1 ]; then
        echo "FATAL: simulation needs a CPU partition -- set PART=<cpu-partition> (the script's baked --partition is a placeholder)." >&2
        exit 1
    fi
    [ -n "${MON_OUT:-}" ] && SB+=( --output="$MON_OUT/%x_%A_Node_%a.out" )
    ;;
  inference)
    SUBMIT_SCRIPT="$REPO/Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Inference.sh"
    JOBNAME="SRM_AND_SBI_DIMER_ALP_${timing_label}_Inference"
    _add TRAIN_TASKS; _add TEST_TASKS; _add EPOCHS; _add TOTAL_TIME; _add BATCH; _add LR; _add HEARTBEAT; _add RESURRECT
    [ -n "${GPU_PART:-}" ] && SB+=( --partition="$GPU_PART" )
    [ -n "${GRES:-}" ]     && SB+=( --gres="$GRES" )
    [ -n "${NODES:-}" ]    && SB+=( --nodes="$NODES" )   # multi-node DDP; --gres is per node -> world_size = NODES * GPUs-per-node
    [ -n "${MON_OUT:-}" ]  && SB+=( --output="$MON_OUT/%x_%A.out" )
    ;;
  evaluation)
    SUBMIT_SCRIPT="$REPO/Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Evaluation.sh"
    JOBNAME="SRM_AND_SBI_DIMER_ALP_${timing_label}_Evaluation"
    _add EVAL_TASKS; _add SUMMARY; _add POOL_MODE; _add TOTAL_TIME
    [ -n "${GPU_PART:-}" ] && SB+=( --partition="$GPU_PART" )
    [ -n "${GRES:-}" ]     && SB+=( --gres="$GRES" )
    [ -n "${NODES:-}" ]    && SB+=( --nodes="$NODES" )   # multi-node: --gres is per node -> world_size = NODES * GPUs-per-node
    [ -n "${MON_OUT:-}" ]  && SB+=( --output="$MON_OUT/%x_%A.out" )
    ;;
  experiment)
    SUBMIT_SCRIPT="$REPO/Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Experiment.sh"
    JOBNAME="SRM_AND_SBI_DIMER_ALP_${timing_label}_Experiment"
    _add MAX_CELLS; _add CHUNK_STEP; _add SUMMARY; _add POOL_MODE; _add TOTAL_TIME
    # KINDS may be multi-value (ALP,BET); Slurm splits --export on commas, so carry
    # it via the exported environment (ALL) rather than the explicit --export list.
    if [ -n "${KINDS:-}" ]; then export KINDS; KINDS_NOTE="KINDS=$KINDS (carried via ALL, comma-safe)"; fi
    [ -n "${GPU_PART:-}" ] && SB+=( --partition="$GPU_PART" )
    [ -n "${GRES:-}" ]     && SB+=( --gres="$GRES" )
    [ -n "${NODES:-}" ]    && SB+=( --nodes="$NODES" )   # multi-node: --gres is per node -> world_size = NODES * GPUs-per-node
    [ -n "${MON_OUT:-}" ]  && SB+=( --output="$MON_OUT/%x_%A.out" )
    ;;
  *)
    echo "FATAL: unknown stage '$STAGE' (use simulation|inference|evaluation|experiment)." >&2
    exit 1
    ;;
esac

[ -n "${ACCT:-}" ] && SB+=( --account="$ACCT" )
[ -n "${TIME:-}" ] && SB+=( --time="$TIME" )
[ -n "${DEP:-}" ]  && SB+=( --dependency="$DEP" )
[ -f "$SUBMIT_SCRIPT" ] || { echo "FATAL: stage script not found: $SUBMIT_SCRIPT" >&2; exit 1; }

EXPORT="$(IFS=,; echo "${EXPORT_PARTS[*]}")"
declare -a CMD=( sbatch --job-name="$JOBNAME" "${SB[@]}" --export="$EXPORT" "$SUBMIT_SCRIPT" )

if [ "$DRYRUN" = 1 ]; then
    echo "=== DRY-RUN [$STAGE] | timing_label=$timing_label | job-name=$JOBNAME ==="
    echo "  REPO         : $REPO"
    echo "  stage script : $SUBMIT_SCRIPT"
    echo "  --export     : $EXPORT"
    [ -n "$KINDS_NOTE" ] && echo "  env via ALL  : $KINDS_NOTE"
    echo "  would submit :"
    echo "      ${CMD[*]}"
    echo "  (knobs not shown above fall back to the stage script's defaults; set DRYRUN=0 to submit.)"
    exit 0
fi

echo "=== SUBMIT [$STAGE] $JOBNAME ==="
"${CMD[@]}"

#!/usr/bin/env bash
# =============================================================================
# Dry-run-first submitter for the Detector calibration workflow's HPC stages.
# =============================================================================
# The Detector's OWN dispatcher — parallel to, and entirely separate from, the
# canonical SRM_AND_SBI_DIMER_ALP_HPC_Submit.sh. It builds the exact sbatch
# command (REPO forwarded, the _DETECTOR data-pattern --job-name with the
# rendered timing_label, and the per-stage --export), then PRINTS it (DRYRUN=1,
# the default) or SUBMITS it (DRYRUN=0). It drives the filename-namespaced
# Detector wrappers (SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_*.sh) that coexist with
# the canonical wrappers in this directory, and it never touches the canonical
# Submit.sh or the four canonical stage wrappers.
#
# Usage:
#   [DRYRUN=0] [GPU_PART=.. PART=.. ACCT=.. TIME=.. DEP=.. ARRAY=.. NTPN=.. CPT=.. \
#              GRES=.. SRM_AND_SBI_GPUS=.. MON_OUT=..] \
#     bash SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_Submit.sh <stage> [KEY=VALUE ...]
#
#   <stage> = simulation | inference | evaluation
#   KEY=VALUE = the stage wrapper's --export knobs (see each wrapper header):
#     simulation : SPLIT SEED TASK_OFFSET TASK_COUNT TASK_SIMS TOTAL_TIME SIM_STAGE
#     inference  : TRAIN_TASKS TEST_TASKS EPOCHS TOTAL_TIME BATCH HEARTBEAT RESURRECT
#     evaluation : EVAL_TASKS POOL_MODE TOTAL_TIME
#
# GPU modes (the two established Goethe modes; the tool pins GRES + the GPU count
# so an exclusive whole-node allocation never launches more workers than intended):
#   GPU_PART=gpu_test  -> 4 GPUs (checks)     : GRES=gpu:4, SRM_AND_SBI_GPUS=4
#   GPU_PART=gpu       -> 8 GPUs (production) : GRES=gpu:8, SRM_AND_SBI_GPUS=8
#   (set GRES / SRM_AND_SBI_GPUS explicitly to deviate.)
#
# Chain stages with DEP=afterok:<jobid>[:<jobid>...] (e.g. submit gen per split,
# then inference DEP=afterok:<gen-train>:<gen-test>, then evaluation
# DEP=afterok:<inference>:<gen-eval>). timing_label renders from TOTAL_TIME exactly
# as PARAMETERS.simulation.timing.label ("{duration}S_50FPS", FPS pinned 50).
#
# Examples (DRY-RUN by default -- print the sbatch line, submit nothing):
#   PART=test NTPN=16 bash .../DETECTOR_HPC_Submit.sh simulation SPLIT=train SEED=42 TASK_COUNT=16 TASK_SIMS=10 TOTAL_TIME=2.0
#   GPU_PART=gpu_test bash .../DETECTOR_HPC_Submit.sh inference TRAIN_TASKS=16 TEST_TASKS=4 EPOCHS=5 BATCH=8 TOTAL_TIME=2.0
#   GPU_PART=gpu_test bash .../DETECTOR_HPC_Submit.sh evaluation EVAL_TASKS=2 POOL_MODE=unrestricted TOTAL_TIME=2.0
# =============================================================================
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Per-machine HPC config (gitignored; copy from hpc_local.env.example): may set
# MACHINE_PROFILE / CONDA_SETUP / PART / GPU_PART / ACCT / MON_OUT / etc.
HPC_ENV="${HPC_ENV:-$HERE/hpc_local.env}"
if [ -f "$HPC_ENV" ]; then . "$HPC_ENV"; fi

# Self-derive + validate REPO (this script lives at REPO/Script_Bank/HPC/).
REPO="${REPO:-$(cd "$HERE/../.." && pwd)}"
if [ ! -f "$REPO/pyproject.toml" ] || [ ! -d "$REPO/srm_and_sbi_dimer_alp" ]; then
    echo "FATAL: '$REPO' is not a srm-and-sbi-dimer-alp repo root (needs pyproject.toml + srm_and_sbi_dimer_alp/)." >&2
    exit 1
fi

DRYRUN="${DRYRUN:-1}"   # 1 = print only (default); 0 = live submit

STAGE="${1:-}"
[ -n "$STAGE" ] || {
    echo "usage: [DRYRUN=0] bash ${BASH_SOURCE[0]##*/} <simulation|inference|evaluation> [KEY=VALUE ...]" >&2
    exit 1
}
shift

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
export REPO

# Two Goethe GPU modes: gpu_test = 4 GPUs (checks), gpu = 8 GPUs (production).
case "${GPU_PART:-}" in
    gpu_test) GRES="${GRES:-gpu:4}"; SRM_AND_SBI_GPUS="${SRM_AND_SBI_GPUS:-4}"; export SRM_AND_SBI_GPUS ;;
    gpu)      GRES="${GRES:-gpu:8}"; SRM_AND_SBI_GPUS="${SRM_AND_SBI_GPUS:-8}"; export SRM_AND_SBI_GPUS ;;
esac

declare -a EXPORT_PARTS=( "ALL" "REPO=$REPO" )
_add(){ local k="$1"; [ -n "${!k:-}" ] && EXPORT_PARTS+=( "$k=${!k}" ); }

declare -a SB=()
SUBMIT_SCRIPT=""
JOBNAME=""

case "$STAGE" in
  simulation)
    SUBMIT_SCRIPT="$REPO/Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_Simulation.sh"
    SPLIT="${SPLIT:-train}"; export SPLIT
    case "$SPLIT" in train|test|eval) ;; *) echo "FATAL: SPLIT='$SPLIT' (use train|test|eval)." >&2; exit 1 ;; esac
    : "${SEED:?FATAL: simulation needs SEED (use distinct per-split seeds so TRAIN/TEST/EVAL theta are independent).}"
    split_uc="$(echo "$SPLIT" | tr '[:lower:]' '[:upper:]')"
    JOBNAME="SRM_AND_SBI_DIMER_ALP_DETECTOR_${timing_label}_Simulation_${split_uc}"
    _add SPLIT; _add SEED; _add TASK_OFFSET; _add TASK_COUNT; _add TASK_SIMS; _add TOTAL_TIME; _add SIM_STAGE
    SB+=( --array="${ARRAY:-0-0}" )   # always array-submit so %a is a clean node number
    [ -n "${NTPN:-}" ] && SB+=( --ntasks-per-node="$NTPN" )
    [ -n "${CPT:-}" ]  && SB+=( --cpus-per-task="$CPT" )
    if [ -n "${PART:-}" ]; then SB+=( --partition="$PART" )
    elif [ "$DRYRUN" != 1 ]; then
        echo "FATAL: simulation needs a CPU partition -- set PART=<cpu-partition>." >&2; exit 1
    fi
    [ -n "${MON_OUT:-}" ] && SB+=( --output="$MON_OUT/%x_%A_Node_%a.out" )
    ;;
  inference)
    SUBMIT_SCRIPT="$REPO/Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_Inference.sh"
    JOBNAME="SRM_AND_SBI_DIMER_ALP_DETECTOR_${timing_label}_Inference"
    _add TRAIN_TASKS; _add TEST_TASKS; _add EPOCHS; _add TOTAL_TIME; _add BATCH; _add HEARTBEAT; _add RESURRECT; _add SRM_AND_SBI_GPUS
    [ -n "${GPU_PART:-}" ] && SB+=( --partition="$GPU_PART" )
    [ -n "${GRES:-}" ]     && SB+=( --gres="$GRES" )
    [ -n "${MON_OUT:-}" ]  && SB+=( --output="$MON_OUT/%x_%A.out" )
    ;;
  evaluation)
    SUBMIT_SCRIPT="$REPO/Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_Evaluation.sh"
    JOBNAME="SRM_AND_SBI_DIMER_ALP_DETECTOR_${timing_label}_Evaluation"
    _add EVAL_TASKS; _add POOL_MODE; _add TOTAL_TIME; _add SRM_AND_SBI_GPUS
    [ -n "${GPU_PART:-}" ] && SB+=( --partition="$GPU_PART" )
    [ -n "${GRES:-}" ]     && SB+=( --gres="$GRES" )
    [ -n "${MON_OUT:-}" ]  && SB+=( --output="$MON_OUT/%x_%A.out" )
    ;;
  *)
    echo "FATAL: unknown stage '$STAGE' (use simulation|inference|evaluation)." >&2
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
    echo "=== DRY-RUN [detector $STAGE] | timing_label=$timing_label | job-name=$JOBNAME ==="
    echo "  REPO         : $REPO"
    echo "  stage script : $SUBMIT_SCRIPT"
    echo "  --export     : $EXPORT"
    echo "  would submit :"
    echo "      ${CMD[*]}"
    echo "  (knobs not shown above fall back to the wrapper's defaults; set DRYRUN=0 to submit.)"
    exit 0
fi

echo "=== SUBMIT [detector $STAGE] $JOBNAME ==="
"${CMD[@]}"

#!/usr/bin/env bash
# =============================================================================
# DIMER production generation -- rolling submit-and-gate controller.
# Submits the six (case x split) generation arrays for the 2 s and 5 s datasets,
# keeping within the QOS caps (<=40 running, <=50 in-system), then HARD-GATES the
# eval splits until every train+test job has COMPLETED.
#
# Plan (10 tasks/node throughout; sims/task: 2 s = 1000, 5 s = 500):
#   Stage A (train+test, 75 jobs, train submitted first):
#       5s-train  --array=0-39  TASK_COUNT=400   (400 tasks, 200k sims)
#       2s-train  --array=0-19  TASK_COUNT=200   (200 tasks, 200k sims)
#       5s-test   --array=0-9   TASK_COUNT=100   (100 tasks,  50k sims)
#       2s-test   --array=0-4   TASK_COUNT=50    ( 50 tasks,  50k sims)
#   Stage B (eval, 8 jobs, GATED on all of A == COMPLETED):
#       5s-eval   --array=0-4   TASK_COUNT=50    ( 50 tasks,  25k sims)
#       2s-eval   --array=0-2   TASK_COUNT=25    ( 25 tasks,  25k sims; last node 10/10/5)
#
# RUN ON THE HPC LOGIN NODE inside tmux/screen (it polls for hours-days). Machine
# divergence is via env vars (all have neutral defaults; override per cluster):
# PART, ACCT, MON_OUT, USER_ME, SIM + the launcher's MACHINE_PROFILE/CONDA_SETUP/MON
# (carried by --export=ALL). REPO is the one exception: this controller self-derives
# it from its own location, then forwards it EXPLICITLY (--export=ALL,REPO=$REPO,...)
# so the spooled child job resolves the repo even though the login-node REPO is a
# plain shell var, not part of the inherited environment.
# Override points: PART (CPU partition), ACCT (Slurm account), MON_OUT (batch-log
# output dir), USER_ME (queue-owner username for polling), SIM (per-task launcher
# path), CASES (which dataset(s): 5s|2s|both), DRYRUN (1 = print only, 0 = submit),
# SKIN_FACTOR (ReaDDy neighbor-list skin as a MULTIPLE of the particle diameter --
# an RDS-only performance knob, not physics; forwarded to every submitted array when
# set; unset = the code default 10x = 100 nm; see SimulationRDS.neighbor_list_skin_factor).
#     DRY RUN (default -- prints the exact sbatch lines, submits nothing):
#         bash SRM_AND_SBI_DIMER_ALP_HPC_Generate_Controller.sh
#     LIVE:
#         DRYRUN=0 bash SRM_AND_SBI_DIMER_ALP_HPC_Generate_Controller.sh 2>&1 | tee ~/dimer_gen_controller.log
#
# Safety: on any train+test job finishing in a non-COMPLETED state the controller
# STOPS before submitting eval (so eval is never generated against broken data).
#
# Re-run a failed node/task under its ORIGINAL global label (fills the gap,
# regenerates nothing good -- the incremental-append mechanism). Submit from the
# repo root and forward REPO (the spooled child cannot resolve the repo from its
# own /var/spool path); --job-name follows the data-file naming convention.
#   node 5 of 5s-train (global task ids 50..59):
#     cd /path/to/srm-and-sbi-dimer-alp && \
#       sbatch --array=0-0 --ntasks-per-node=10 --cpus-per-task=4 --time=24:00:00 \
#         --job-name=SRM_AND_SBI_DIMER_ALP_5S_50FPS_Simulation_TRAIN \
#         --export=ALL,REPO=$PWD,SPLIT=train,TASK_OFFSET=50,TASK_COUNT=10,TASK_SIMS=500,TOTAL_TIME=5.0 \
#         Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Simulation.sh
#   single task (id 137): --ntasks-per-node=1 TASK_OFFSET=137 TASK_COUNT=1 (same SPLIT/SIMS/TIME).
# Then confirm label completeness with the seeding-validation script before training.
# =============================================================================
set -uo pipefail

# Per-machine HPC config (gitignored; copy from hpc_local.env.example): sets
# MACHINE_PROFILE / PART / ACCT / MON_OUT / USER_ME / etc. so this generic
# controller runs unchanged on any cluster. Falls back to the defaults below if absent.
HPC_ENV="${HPC_ENV:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/hpc_local.env}"
if [ -f "$HPC_ENV" ]; then . "$HPC_ENV"; fi

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"   # self-derived: this script lives at REPO/Script_Bank/HPC/
SIM="${SIM:-$REPO/Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Simulation.sh}"
USER_ME="${USER_ME:-$USER}"
PART="${PART:-}"             # your cluster's CPU partition; passed only when set (else the launcher's baked --partition applies)
ACCT="${ACCT:-}"             # your Slurm account, if your cluster requires one
MON_OUT="${MON_OUT:-$HOME/process_monitoring}"  # batch-log --output dir (must exist)
NTPN=10                       # tasks per node
CPT=4                         # cpus per task (10*4 = 40 cores = --extra-node-info 2:20:1)
MAX_INSYS=50                  # QOS MaxSubmit (running + pending)
POLL=120                      # seconds between queue polls
DRYRUN="${DRYRUN:-1}"         # 1 = print only (default); 0 = live submit
STATE="$HOME/dimer_gen_controller_state.txt"

# label|array_max|time|split|task_count|task_sims|total_time
STAGE_A=(
  "5s-train|39|24:00:00|train|400|500|5.0"
  "2s-train|19|18:00:00|train|200|1000|2.0"
  "5s-test|9|24:00:00|test|100|500|5.0"
  "2s-test|4|18:00:00|test|50|1000|2.0"
)
STAGE_B=(
  "5s-eval|4|24:00:00|eval|50|500|5.0"
  "2s-eval|2|18:00:00|eval|25|1000|2.0"
)

# CASES selects which dataset(s) to run: 5s | 2s | both (default). Lets the same
# controller drive the 5 s campaign and the 2 s campaign on separate clusters
# independently.
CASES="${CASES:-both}"
case "$CASES" in both|5s|2s) ;; *) echo "bad CASES=$CASES (use 5s|2s|both)" >&2; exit 1;; esac
if [ "$CASES" != both ]; then
  _A=(); for e in "${STAGE_A[@]}"; do [[ "${e%%|*}" == "${CASES}-"* ]] && _A+=("$e"); done; STAGE_A=("${_A[@]}")
  _B=(); for e in "${STAGE_B[@]}"; do [[ "${e%%|*}" == "${CASES}-"* ]] && _B+=("$e"); done; STAGE_B=("${_B[@]}")
fi

log(){ echo "[$(date '+%F %T')] $*" >&2; }
insys(){ squeue -u "$USER_ME" -h -r 2>/dev/null | wc -l | tr -d ' '; }

submit(){   # $1 = entry; echoes job id on stdout, logs to stderr
  IFS='|' read -r label amax tlim split count sims ttime <<<"$1"
  local n=$(( amax + 1 ))
  # REPO is forwarded EXPLICITLY: it is a plain shell var on the login node, so
  # --export=ALL alone would NOT carry it to the spooled child (which runs from
  # /var/spool and cannot resolve the repo from its own path).
  local export="ALL,REPO=$REPO,SPLIT=$split,TASK_OFFSET=0,TASK_COUNT=$count,TASK_SIMS=$sims,TOTAL_TIME=$ttime"
  # SKIN_FACTOR is optional (RDS-only performance knob); forward it only when set so
  # an unset value leaves the code default (SimulationRDS.neighbor_list_skin_factor).
  [ -n "${SKIN_FACTOR:-}" ] && export="${export},SKIN_FACTOR=${SKIN_FACTOR}"
  # Job name follows the data-file naming convention:
  # SRM_AND_SBI_DIMER_ALP_<timing_label>_Simulation_<SPLIT>, with timing_label
  # rendered exactly as PARAMETERS.simulation.timing.label does ("{duration}S_50FPS",
  # duration via :g so 2.0 -> 2, 5.0 -> 5, 2.5 -> 2.5) and SPLIT upper-cased.
  local timing_label split_uc jobname
  timing_label="$(LC_ALL=C printf '%gS_50FPS' "$ttime")"
  split_uc="$(echo "$split" | tr '[:lower:]' '[:upper:]')"
  jobname="SRM_AND_SBI_DIMER_ALP_${timing_label}_Simulation_${split_uc}"
  # batch-log --output is forced here (the launcher's baked #SBATCH --output is a
  # submit-directory path); --partition and --account are appended only when set,
  # so an unset PART/ACCT leaves the submit line at the launcher's baked defaults.
  local -a extra=( --job-name="$jobname" --output="$MON_OUT/%x_%A_Node_%a.out" )
  [ -n "$PART" ] && extra+=( --partition="$PART" )
  [ -n "$ACCT" ] && extra+=( --account="$ACCT" )
  if [ "$DRYRUN" = 1 ]; then
    log "DRY-RUN [$label] $n jobs: sbatch --array=0-$amax --ntasks-per-node=$NTPN --cpus-per-task=$CPT --time=$tlim ${extra[*]} --export=$export $SIM"
    echo "DRYRUN-$label"; return
  fi
  local jid
  jid=$(sbatch --parsable --array="0-$amax" \
        --ntasks-per-node="$NTPN" --cpus-per-task="$CPT" --time="$tlim" \
        "${extra[@]}" --export="$export" "$SIM") || { log "!! SUBMIT FAILED for $label"; echo "ERR"; return; }
  log "submitted [$label] $n jobs -> $jid"
  echo "$(date '+%F %T') $label $jid $n" >> "$STATE"
  echo "$jid"
}

wait_room(){   # $1 = jobs to fit under MAX_INSYS
  local need="$1" cur
  while :; do
    cur=$(insys)
    [ $(( cur + need )) -le "$MAX_INSYS" ] && return
    log "in-system=$cur; waiting for room for +$need (cap $MAX_INSYS)"
    sleep "$POLL"
  done
}

gate(){   # $1 = comma-separated job ids; blocks until all COMPLETED; exit 2 on bad-terminal
  local ids="$1" inq st
  while :; do
    # Liveness from squeue (the live job state): while any array element is still
    # queued/running, keep waiting. This avoids the sacct-registration lag right
    # after submission, where sacct returns empty before reporting PENDING.
    inq=$(squeue -j "$ids" -h -r 2>/dev/null | wc -l | tr -d ' ')
    if [ "${inq:-0}" -gt 0 ]; then
      log "train+test active: $inq element(s) in queue (in-system=$(insys)); waiting"
      sleep "$POLL"; continue
    fi
    # Queue clear -> confirm final states from accounting; re-wait while sacct lags.
    st=$(sacct -j "$ids" -n -X -o State 2>/dev/null | tr -d ' ' | sort -u | grep -v '^$')
    if [ -z "$st" ]; then
      log "queue clear but sacct not populated yet; waiting"; sleep "$POLL"; continue
    fi
    if echo "$st" | grep -qvE '^COMPLETED$'; then
      log "!!! STOP: a train+test job did not COMPLETE -> states: $(echo "$st" | tr '\n' ' ')"
      log "!!! eval NOT submitted. Inspect/resubmit, then re-run Stage B only. (sacct -j $ids)"
      exit 2
    fi
    log "all train+test COMPLETED."; return
  done
}

# ---------- preflight ----------
[ -f "$SIM" ] || { log "launcher not found: $SIM"; exit 1; }
EX=$(insys); [ "$EX" -gt 0 ] && log "WARNING: $EX job(s) already in your queue -- they count toward the $MAX_INSYS cap."
_cnt(){ local t=0 e amax; for e in "$@"; do amax=${e#*|}; amax=${amax%%|*}; t=$(( t + amax + 1 )); done; echo "$t"; }
log "===== DIMER production generation | CASES=$CASES | DRYRUN=$DRYRUN ====="
log "Stage A (train+test) = $(_cnt "${STAGE_A[@]}") jobs; Stage B (eval, gated) = $(_cnt "${STAGE_B[@]}") jobs. Caps: <=40 run / <=$MAX_INSYS in-system."

# ---------- Stage A: train+test (rolling, train-first) ----------
A_IDS=()
for e in "${STAGE_A[@]}"; do
  amax=${e#*|}; amax=${amax%%|*}; n=$(( amax + 1 ))
  wait_room "$n"
  jid=$(submit "$e"); A_IDS+=("$jid")
done

# ---------- dry-run short-circuit ----------
if [ "$DRYRUN" = 1 ]; then
  log "DRY-RUN: would now gate on Stage A completion, then submit Stage B:"
  for e in "${STAGE_B[@]}"; do submit "$e" >/dev/null; done
  log "DRY-RUN complete. Re-run with DRYRUN=0 (inside tmux) to go live."
  exit 0
fi

# ---------- gate ----------
A_CSV=$(IFS=,; echo "${A_IDS[*]}")
log "Stage A submitted (jobs: $A_CSV). Gating eval on their completion."
gate "$A_CSV"

# ---------- Stage B: eval ----------
for e in "${STAGE_B[@]}"; do
  amax=${e#*|}; amax=${amax%%|*}; n=$(( amax + 1 ))
  wait_room "$n"; submit "$e" >/dev/null
done
log "===== eval submitted. Controller done. Watch: squeue -u $USER_ME ====="

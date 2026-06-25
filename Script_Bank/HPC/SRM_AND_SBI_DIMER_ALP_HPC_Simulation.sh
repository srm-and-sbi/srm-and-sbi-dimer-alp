#!/bin/bash
# =============================================================================
# Slurm HPC generation submitter: RDS -> DLI, many tasks packed per node.
# =============================================================================
# The CPU partition is typically exclusive, so each job owns a whole node. We
# launch the packed tasks as background processes (the OS spreads them over the
# node's cores) and `wait`.
# Generation is non-deterministic by design (no seed) -- provenance
# is the global task index in the file names. Multi-node scaling = an --array of
# single-node jobs; TASK_OFFSET shifts the global index for incremental growth:
#
#     tid = TASK_OFFSET + SLURM_ARRAY_TASK_ID * SLURM_NTASKS_PER_NODE + k
#
# Knobs (--export / CLI): SPLIT, TASK_SIMS, TOTAL_TIME, TASK_OFFSET, TASK_COUNT
# (tasks this submission generates; default = --ntasks-per-node); pack size and
# core share are set by --ntasks-per-node / --cpus-per-task. Debug knobs
# (default off, production-identical when unset): PROBE=1 logs per-sim resource
# use (threads/open-fds/RSS); VERBOSE=1 adds per-sim detail (reaction counts, shapes);
# DEBUG_DUMP=1 writes the DiagnosticReporter console.log + arrays; SEED=<int> fixes
# the RNG for a reproducible run.
#
# ALWAYS submit with --array (one element per node, --array=0-0 for a single
# node) so the batch-log %a is a clean node number (0,1,...); without --array,
# Slurm sets %a to its not-an-array sentinel (4294967294).
#
# CORE=100 production (TRAIN 8 / TEST 2 / EVAL 1; 1000 sims/task), one node each:
#   sbatch --array=0-0 --ntasks-per-node=8 --export=ALL,SPLIT=train,TASK_OFFSET=0,TASK_COUNT=8 SRM_AND_SBI_DIMER_ALP_HPC_Simulation.sh
#   sbatch --array=0-0 --ntasks-per-node=2 --export=ALL,SPLIT=test,TASK_OFFSET=0,TASK_COUNT=2  SRM_AND_SBI_DIMER_ALP_HPC_Simulation.sh
#   sbatch --array=0-0 --ntasks-per-node=1 --export=ALL,SPLIT=eval,TASK_OFFSET=0,TASK_COUNT=1  SRM_AND_SBI_DIMER_ALP_HPC_Simulation.sh
# Two nodes, 20 tasks (element 0 -> Task_0..9, element 1 -> Task_10..19):
#   sbatch --array=0-1 --ntasks-per-node=10 --cpus-per-task=4 --export=ALL,SPLIT=train,TASK_OFFSET=0,TASK_COUNT=20 ...
# Grow TRAIN later (appends tasks 8..15, no regeneration):
#   sbatch --array=0-0 --ntasks-per-node=8 --export=ALL,SPLIT=train,TASK_OFFSET=8,TASK_COUNT=8 ...
# -----------------------------------------------------------------------------
#SBATCH --job-name=SRM_AND_SBI_DIMER_ALP_HPC_Simulation
#SBATCH --partition=YOUR_PARTITION   # set to your cluster's CPU partition (or override on the sbatch command line)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=5
#SBATCH --mem-per-cpu=4400
#SBATCH --extra-node-info=2:20:1
#SBATCH --time=08:00:00
#SBATCH --mail-type=FAIL
#SBATCH --output=%x_%A_Node_%a.out   # submit-directory; the controller overrides this via MON_OUT for packed jobs

source "${CONDA_SETUP:-$HOME/miniconda3/etc/profile.d/conda.sh}"
conda activate SRM_AND_SBI_ENVY_V0
export MACHINE_PROFILE="${MACHINE_PROFILE:?set MACHINE_PROFILE to the profile name in your machine_profiles.toml}"

REPO="${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"   # self-derived: each script lives at REPO/Script_Bank/HPC/
PRIME="$REPO/Script_Bank/Prime"
MON="${MON:-$HOME/process_monitoring}"   # directory must exist
cd "$REPO"

SPLIT="${SPLIT:-train}"
TASK_SIMS="${TASK_SIMS:-1000}"
TOTAL_TIME="${TOTAL_TIME:-2.0}"
TASK_OFFSET="${TASK_OFFSET:-0}"
PER_NODE="${SLURM_NTASKS_PER_NODE:-8}"
TASK_COUNT="${TASK_COUNT:-$PER_NODE}"
ARRAY_ID="${SLURM_ARRAY_TASK_ID:-0}"
JOB_TAG="${SLURM_ARRAY_JOB_ID:-$SLURM_JOB_ID}"

# Debug knobs (default off -> production-identical). PROBE=1 adds per-sim resource
# logging; VERBOSE=1 / DEBUG_DUMP=1 add per-sim detail / dumps; SEED=<int> fixes the
# RNG. (Per-sim ReaDDy cleanup is always on.)
SIM_FLAGS=""
[ "${PROBE:-0}" = 1 ]      && SIM_FLAGS="$SIM_FLAGS --probe"
[ "${VERBOSE:-0}" = 1 ]    && SIM_FLAGS="$SIM_FLAGS --verbose"
[ "${DEBUG_DUMP:-0}" = 1 ] && SIM_FLAGS="$SIM_FLAGS --debug-dump"
[ -n "${SEED:-}" ]         && SIM_FLAGS="$SIM_FLAGS --seed ${SEED}"

# SIM_STAGE selects which stage(s) run per task: both (default) | rds | dli. SIM_STAGE=dli is
# a DLI-only re-run that reuses the existing trajectories (e.g. to re-render videos
# after a DLI-side fix without repeating the expensive RDS).
SIM_STAGE="${SIM_STAGE:-both}"
case "$SIM_STAGE" in both|rds|dli) ;; *) echo "bad SIM_STAGE=$SIM_STAGE (use both|rds|dli)" >&2; exit 1;; esac

start=$(( TASK_OFFSET + ARRAY_ID * PER_NODE ))
end=$(( TASK_OFFSET + TASK_COUNT ))
echo "=== Simulation | $(hostname) | split=${SPLIT} sims=${TASK_SIMS} time=${TOTAL_TIME}s | seed=None | tasks [${start}..$(( end - 1 ))] ==="

declare -a PIDS=()
for (( tid=start; tid < start + PER_NODE && tid < end; tid++ )); do
    out="${MON}/${SLURM_JOB_NAME}_${JOB_TAG}_Node_${ARRAY_ID}_Task_${tid}.out"
    ( rc=0
      if [ "$SIM_STAGE" != dli ]; then
        python -u "$PRIME/SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py" \
            --task-id "$tid" --task-simulations "$TASK_SIMS" \
            --total-time-seconds "$TOTAL_TIME" --split "$SPLIT" $SIM_FLAGS || rc=1
      fi
      if [ "$rc" = 0 ] && [ "$SIM_STAGE" != rds ]; then
        python -u "$PRIME/SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py" \
            --task-id "$tid" --task-simulations "$TASK_SIMS" \
            --total-time-seconds "$TOTAL_TIME" --split "$SPLIT" $SIM_FLAGS || rc=1
      fi
      exit "$rc" ) > "$out" 2>&1 &
    PIDS+=( "$!" )
    echo "  -> Task ${tid} -> $(basename "$out")"
done

rc=0
for p in "${PIDS[@]}"; do wait "$p" || rc=1; done
echo "=== Simulation | array element ${ARRAY_ID} (${SPLIT}) complete (rc=${rc}) ==="
exit "$rc"

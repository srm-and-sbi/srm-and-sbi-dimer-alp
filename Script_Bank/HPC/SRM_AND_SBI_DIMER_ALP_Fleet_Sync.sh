#!/bin/bash
# =============================================================================
# Fleet sync: reconcile every remote machine's repo to this one, exactly.
# =============================================================================
# mars-fias is the reference. Every other machine holds the SAME repo content plus
# only its own machine-local files. This script is the single supported way to
# propagate the repo, because doing it by hand went wrong in three distinct ways:
#
#   1. `rsync a.py b.py doc.md host:REPO/` silently FLATTENS every source into
#      REPO/, so package modules landed in the repo root instead of the package
#      directory -- and a stage then ran stale code while the root held the new copy.
#      Here the file list is always relative to the repo root, so a file can only
#      ever land where it belongs.
#   2. `rsync --files-from=...` CANNOT delete (--delete needs directory recursion,
#      which --files-from disables), so remotes accumulated every file the reference
#      had ever contained. This script recurses per directory WITH --delete, so a
#      removal on the reference propagates.
#   3. Nothing excluded the secrets file, so RECOVERY_CODES.md was copied to three
#      shared HPC filesystems. It is now excluded by name, permanently.
#
# Dry-run is the default, matching the other submitters: DRYRUN=1 prints what would
# change and transfers nothing. Set DRYRUN=0 only after reading the printed plan.
#
# Usage:
#   bash Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_Fleet_Sync.sh [machine ...]
#   DRYRUN=0 bash Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_Fleet_Sync.sh jupiter
#   (no machine argument = every machine listed below)
# -----------------------------------------------------------------------------
set -eo pipefail

DRYRUN="${DRYRUN:-1}"
CP="$HOME/.ssh/control-%h-%p-%r"
SSH_JSC="ssh -o ControlPath=$CP -o ControlMaster=no -o BatchMode=yes -o ConnectTimeout=15"
SSH_PLAIN="ssh -o BatchMode=yes -o ConnectTimeout=15"

# name | ssh command | host | absolute repo path on that machine
FLEET=(
  "jupiter|$SSH_JSC|ramirezsierra1@login.jupiter.fz-juelich.de|/e/project1/chkf10/ramirezsierra1/Projects/RCL_Agent_Projects/srm-and-sbi/srm-and-sbi-dimer-alp"
  "juwels|$SSH_JSC|ramirezsierra1@juwels-cluster.fz-juelich.de|/p/project1/chkf10/ramirezsierra1/Projects/RCL_Agent_Projects/srm-and-sbi/srm-and-sbi-dimer-alp"
  "goethe|$SSH_PLAIN|ramirez@goethe.hhlr-gu.de|/home/biochemsim/ramirez/Projects/RCL_Agent_Projects/srm-and-sbi/srm-and-sbi-dimer-alp"
  "rcl01|$SSH_PLAIN|rcl_fias@10.83.255.103|/home/rcl_fias/Documents/Projects/RCL_Agent_Projects/srm-and-sbi/srm-and-sbi-dimer-alp"
)

# NEVER leaves the reference machine. RECOVERY_CODES.md holds live recovery codes;
# it is gitignored so it never reaches GitHub, and excluded here so it never reaches
# a shared filesystem either.
SECRETS=( "RECOVERY_CODES.md" )

# Legitimately per-machine: each remote keeps its own and the sync must not touch,
# overwrite, or delete them.
MACHINE_LOCAL=( "machine_profiles.toml" "Script_Bank/HPC/hpc_local.env" )

# Not content: build artifacts, caches, editor scratch, and the data tree (which
# lives outside the repo on the reference but has appeared inside it on a remote).
NOT_CONTENT=( ".git" "__pycache__" "*.pyc" "*.egg-info" ".ipynb_checkpoints"
              ".virtual_documents" "Data_Bank" )

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"
[ -f pyproject.toml ] && [ -d srm_and_sbi_dimer_alp ] || {
    echo "FATAL: $REPO_ROOT is not the repo root." >&2; exit 1; }

EXCLUDES=()
for p in "${SECRETS[@]}" "${MACHINE_LOCAL[@]}" "${NOT_CONTENT[@]}"; do
    EXCLUDES+=( --exclude="$p" )
done

# Sync the repo as a recursive tree so --delete can remove what the reference no
# longer has. Excluded paths are protected from deletion too (--delete-excluded is
# deliberately NOT used), so machine-local files and the data tree survive.
RSYNC_OPTS=( -rlpt --delete --itemize-changes "${EXCLUDES[@]}" )
[ "$DRYRUN" = "0" ] || RSYNC_OPTS+=( --dry-run )

WANT=("$@")
echo "======================================================================"
echo " Fleet sync from $REPO_ROOT"
echo " mode: $([ "$DRYRUN" = "0" ] && echo 'LIVE (transferring)' || echo 'DRY RUN (set DRYRUN=0 to apply)')"
echo " excluded: secrets(${#SECRETS[@]}) machine-local(${#MACHINE_LOCAL[@]}) non-content(${#NOT_CONTENT[@]})"
echo "======================================================================"

for entry in "${FLEET[@]}"; do
    IFS='|' read -r NAME SSHC HOST REPO <<< "$entry"
    if [ ${#WANT[@]} -gt 0 ]; then
        printf '%s\n' "${WANT[@]}" | grep -qx "$NAME" || continue
    fi
    echo
    echo "---- $NAME : $HOST:$REPO"
    if ! timeout 30 $SSHC "$HOST" "[ -d '$REPO/srm_and_sbi_dimer_alp' ]" 2>/dev/null; then
        echo "     UNREACHABLE or repo missing -- skipped"
        continue
    fi
    rsync "${RSYNC_OPTS[@]}" -e "$SSHC" ./ "$HOST:$REPO/" 2>&1 \
      | grep -E '^(\*deleting|[<>]f|cd\+)' | sed 's/^/     /' || echo "     (already identical)"
    if [ "$DRYRUN" = "0" ]; then
        # Verify: the only differences left must be excluded paths.
        n=$(rsync -rlpt --delete --dry-run --itemize-changes "${EXCLUDES[@]}" \
              -e "$SSHC" ./ "$HOST:$REPO/" 2>/dev/null | grep -cE '^(\*deleting|[<>]f)' || true)
        echo "     verified: $n outstanding difference(s) after sync (want 0)"
        timeout 30 $SSHC "$HOST" "[ -f '$REPO/RECOVERY_CODES.md' ]" 2>/dev/null \
          && echo "     WARNING: secrets file present on $NAME" \
          || echo "     secrets check: RECOVERY_CODES.md absent  OK"
    fi
done
echo
echo "Done. $([ "$DRYRUN" = "0" ] || echo 'Nothing was transferred (dry run).')"

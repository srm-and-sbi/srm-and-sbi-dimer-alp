# HPC Operations Runbook — srm-and-sbi-dimer-alp

The single authoritative reference for running the DIMER pipeline on a Slurm
cluster. The batch scripts in this directory are generic and committed; each
machine supplies its own values through a gitignored `hpc_local.env`, so the
same scripts run unchanged on any cluster.

This runbook describes what the scripts in `Script_Bank/HPC/` actually do.
Treat the `#SBATCH` blocks and header examples inside those scripts as the
ground truth and replicate them; this document organizes and explains them.

---

## 1. Stages and partitions

The pipeline runs as a sequence of stages, each driven by one batch script
here. Each script activates the `SRM_AND_SBI_ENVY_V0` conda environment and
runs the matching entry point under `Script_Bank/Prime/`.

| Stage | Script | Compute | Production partition | Check partition |
|-------|--------|---------|----------------------|-----------------|
| Simulation (RDS → DLI) | `SRM_AND_SBI_DIMER_ALP_HPC_Simulation.sh` | CPU | `general1` | `test` |
| Inference (train + select) | `SRM_AND_SBI_DIMER_ALP_HPC_Inference.sh` | GPU | `gpu` | `gpu_test` |
| Evaluation (MAP recovery) | `SRM_AND_SBI_DIMER_ALP_HPC_Evaluation.sh` | GPU | `gpu` | `gpu_test` |
| Experiment (real videos) | `SRM_AND_SBI_DIMER_ALP_HPC_Experiment.sh` | GPU | `gpu` | `gpu_test` |

- **Simulation** packs many generation tasks per node (RDS reaction-diffusion
  trajectories, then DLI diffraction-limited videos) and is CPU-bound.
- **Inference** trains the posterior on the TRAIN tasks and selects on the
  TEST tasks. It adapts to the allocated GPUs: more than one GPU trains
  data-parallel via `DistributedDataParallel` (torchrun, one process per GPU);
  one GPU uses the single-GPU path.
- **Evaluation** estimates the maximum-a-posteriori parameter vector on the
  held-out EVAL set and reports per-parameter recovery accuracy and posterior
  calibration. With more than one GPU it shards EVAL across workers (torchrun),
  then merges the per-shard results into one report.
- **Experiment** applies the trained estimator to real microscopy `.tif`
  recordings (no ground truth) and reports the inferred-parameter distribution
  per condition. With more than one GPU it shards its `(kind, cell)` work across
  workers (torchrun), then merges the per-shard results into one report. It is the
  scientific end use, not a correctness check.

The normal ordering is **Simulation → Inference → Evaluation**. Experiment runs
once a trained posterior exists. Generation runs on the CPU partition; training,
evaluation, and the real-data application run on the GPU partition.

**Durations are general.** The codebase is duration-parameterized: the recording
length is supplied per run via `--total-time-seconds`, and the frame count
follows as `frame_count = total_time_seconds / 0.020 s` (50 frames per second).
Specific durations below (2 s, 5 s, 1 s, 10 s) are concrete examples, not a
fixed pairing; substitute the duration the campaign calls for.

---

## 2. Submission recipe

Slurm spools a copy of each batch script into `/var/spool` before running it, so
the script's own path is unreliable for a directly-submitted job. Every script
resolves the repository root with a `_find_repo` helper that tries, in order:

1. an explicit `REPO` (e.g. `--export=ALL,REPO=$PWD,...`),
2. `SLURM_SUBMIT_DIR` (and its grandparent),
3. the script's own location (for a non-Slurm `bash <script>` invocation),

and accepts the first candidate that actually contains `pyproject.toml` and the
`srm_and_sbi_dimer_alp/` package. If none match, the script **fails loud** with
guidance rather than crashing on a `/var/spool` path.

**The rule:** submit from the repository root, or pass `REPO` explicitly via
`--export`. Either makes `REPO` resolvable.

```bash
cd /path/to/srm-and-sbi-dimer-alp        # so SLURM_SUBMIT_DIR resolves the repo
# or, from anywhere, add REPO=$PWD (run from the root) to --export
```

The same resolved `REPO` is used to source the per-machine `hpc_local.env` (see
§5), so it is found even under spooling.

### Dry-run first: the submit helper

Always preview a submission before it reaches the queue. The unified
`SRM_AND_SBI_DIMER_ALP_HPC_Submit.sh` builds the exact `sbatch` command for any
stage — the resolved `REPO`, the data-pattern `--job-name` (with the rendered
`timing_label`), and a comma-split-safe `--export` — and **prints it without
submitting** unless you set `DRYRUN=0`. Because the recipe, the naming, and the
config are built by the tool, they cannot be mistyped at submit time.

```bash
# Dry run (the default): print the exact sbatch line, submit nothing
bash Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Submit.sh inference TOTAL_TIME=5.0 TRAIN_TASKS=400 TEST_TASKS=100 EPOCHS=25
# Submit it (only after the printed command checks out)
DRYRUN=0 GPU_PART=gpu bash Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Submit.sh inference TOTAL_TIME=5.0 TRAIN_TASKS=400 TEST_TASKS=100 EPOCHS=25
```

`<stage>` is `simulation | inference | evaluation | experiment`; the `KEY=VALUE`
pairs are that stage's `--export` knobs (anything omitted falls back to the
stage script's own default). sbatch-level overrides go in the environment:
`PART` (CPU partition — required for a live `simulation`, whose baked
`--partition` is a placeholder), `GPU_PART`, `ACCT`, `TIME`,
`ARRAY`/`NTPN`/`CPT` (simulation only), `GRES`, `MON_OUT`. A multi-value `KINDS`
(e.g. `KINDS=ALP,BET`) is carried safely through the exported environment via
`ALL` rather than the comma-split `--export`. The helper is to a single job what
the generation controller (§5) is to the full generation campaign — both default
to dry-run, and you opt in to submitting with `DRYRUN=0`.

For a **local** run — or as a final configuration check before any submission —
pass `--dry-run` to the Prime entry point itself: it resolves the machine
profile and the input paths, prints what it would read and write (flagging
anything MISSING), and exits before any compute (no GPU, no data load).

```bash
MACHINE_PROFILE=<profile> python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Inference.py \
    --total-time-seconds 2.0 --tasks 8 --test-tasks 2 --epochs 50 --dry-run
```

The raw `sbatch` invocations below are the ground truth the helper generates;
use them directly when you want full manual control.

### Runnable examples (one per stage)

Submit from the repository root. Set `--partition` to the partition for the run
(production vs check, per §1); the baked `#SBATCH --partition` is a placeholder
on the Simulation script and `gpu` on the GPU scripts.

```bash
# Simulation — TRAIN split, one node, 8 packed tasks (always submit with --array)
cd /path/to/srm-and-sbi-dimer-alp
sbatch --job-name=SRM_AND_SBI_DIMER_ALP_2S_50FPS_Simulation_TRAIN \
       --partition=general1 --array=0-0 --ntasks-per-node=8 \
       --output="$MON_OUT/%x_%A_Node_%a.out" \
       --export=ALL,REPO=$PWD,SPLIT=train,TASK_OFFSET=0,TASK_COUNT=8,TASK_SIMS=1000,TOTAL_TIME=2.0 \
       Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Simulation.sh
```

```bash
# Inference — train on 8 TRAIN tasks, select on 2 TEST tasks, 50 epochs
cd /path/to/srm-and-sbi-dimer-alp
sbatch --job-name=SRM_AND_SBI_DIMER_ALP_2S_50FPS_Inference \
       --partition=gpu \
       --output="$MON_OUT/%x_%A.out" \
       --export=ALL,REPO=$PWD,TRAIN_TASKS=8,TEST_TASKS=2,EPOCHS=50,TOTAL_TIME=2.0 \
       Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Inference.sh
# Continue a wall-stopped run from its checkpoint: add RESURRECT=1 to the --export
# (or pass RESURRECT=1 to the Submit.sh helper). The first job runs fresh.
```

```bash
# Evaluation — MAP recovery on the held-out EVAL set
cd /path/to/srm-and-sbi-dimer-alp
sbatch --job-name=SRM_AND_SBI_DIMER_ALP_2S_50FPS_Evaluation \
       --partition=gpu \
       --output="$MON_OUT/%x_%A.out" \
       --export=ALL,REPO=$PWD,EVAL_TASKS=1,SUMMARY=both,POOL_MODE=bounded,TOTAL_TIME=2.0 \
       Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Evaluation.sh
```

```bash
# Experiment — apply the trained posterior to real videos.
# KINDS defaults to ALP,BET (baked in the script). Do NOT place a multi-value
# KINDS inside --export: Slurm splits --export on commas, so KINDS=ALP,BET would
# parse as KINDS=ALP plus a stray, value-less BET. To override KINDS with multiple
# values, pre-export it in the submitting shell and let --export=ALL carry it
# (export KINDS=ALP,BET) — or just use the Submit.sh helper, which does this for you.
cd /path/to/srm-and-sbi-dimer-alp
sbatch --job-name=SRM_AND_SBI_DIMER_ALP_2S_50FPS_Experiment \
       --partition=gpu \
       --output="$MON_OUT/%x_%A.out" \
       --export=ALL,REPO=$PWD,SUMMARY=both,TOTAL_TIME=2.0 \
       Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Experiment.sh
```

`MON_OUT` is your monitoring/batch-log directory and **must already exist**
(see §5).

---

## 3. Job and log naming

Job names mirror the theta/video data files so a batch log and the artifacts it
produces share one provenance string. The data files are named, for example,
`SRM_AND_SBI_DIMER_ALP_2S_50FPS_Video_Set_TASK_0_TRAIN.zarr` and
`SRM_AND_SBI_DIMER_ALP_2S_50FPS_Theta_Set_TASK_0_TRAIN.zarr`: the pattern is
`{project_alias}_{timing_label}_<descriptor>`.

- **`project_alias`** = `SRM_AND_SBI_DIMER_ALP`
- **`timing_label`** = `<duration>S_<FPS>FPS`, placed **immediately after the
  alias** (e.g. `1S_50FPS`, `2S_50FPS`, `5S_50FPS`). No `HPC` token.

Job names follow the same shape:

```
SRM_AND_SBI_DIMER_ALP_<timing_label>_<Stage>[_<SPLIT>]
```

| Stage | Job name |
|-------|----------|
| Simulation | `SRM_AND_SBI_DIMER_ALP_<timing_label>_Simulation_<SPLIT>` (SPLIT = `TRAIN`/`TEST`/`EVAL`) |
| Inference | `SRM_AND_SBI_DIMER_ALP_<timing_label>_Inference` |
| Evaluation | `SRM_AND_SBI_DIMER_ALP_<timing_label>_Evaluation` |
| Experiment | `SRM_AND_SBI_DIMER_ALP_<timing_label>_Experiment` |

Examples: `SRM_AND_SBI_DIMER_ALP_5S_50FPS_Inference`,
`SRM_AND_SBI_DIMER_ALP_1S_50FPS_Simulation_TRAIN`.

Set the job name with `--job-name` and direct the batch log into the monitoring
directory:

- Simulation (an array, one element per node): `--output="$MON_OUT/%x_%A_Node_%a.out"`
- All other stages: `--output="$MON_OUT/%x_%A.out"`

`%x` is the job name, `%A` the array job id, `%a` the array element (the node
number). The Simulation array element is always a clean node number because the
job is always submitted with `--array` (`--array=0-0` for a single node);
without `--array`, Slurm sets `%a` to its not-an-array sentinel.

Inside the Simulation node, each packed task additionally writes its own per-task
log: `${MON}/${job_name}_${job_tag}_Node_${array_id}_Task_${tid}.out`.

---

## 4. Hardware configurations to replicate

Use the layouts encoded in the scripts' `#SBATCH` blocks and header examples.
Do not recompute node geometry — replicate these.

### Simulation (CPU)

The baked layout packs tasks per node and pins the core geometry:

- `--nodes=1`, `--ntasks-per-node=8`, `--cpus-per-task=5`, `--mem-per-cpu=4400`
- `--extra-node-info=2:20:1` (40-core nodes: 2 sockets × 20 cores × 1 thread)
- `--time=08:00:00`

Always submit with `--array` (one element per node; `--array=0-0` for a single
node). Multi-node scaling is an `--array` of single-node jobs; `TASK_OFFSET`
shifts the global task index so a later submission appends tasks rather than
regenerating existing ones. The per-task global index is
`tid = TASK_OFFSET + SLURM_ARRAY_TASK_ID * ntasks_per_node + k`.

**Production (`general1`), 1000 sims/task, one node per split:**

```bash
# TRAIN 8 / TEST 2 / EVAL 1 tasks (per CORE=100), submit from the repo root:
sbatch --partition=general1 --array=0-0 --ntasks-per-node=8 --export=ALL,REPO=$PWD,SPLIT=train,TASK_OFFSET=0,TASK_COUNT=8 ...Simulation.sh
sbatch --partition=general1 --array=0-0 --ntasks-per-node=2 --export=ALL,REPO=$PWD,SPLIT=test,TASK_OFFSET=0,TASK_COUNT=2  ...Simulation.sh
sbatch --partition=general1 --array=0-0 --ntasks-per-node=1 --export=ALL,REPO=$PWD,SPLIT=eval,TASK_OFFSET=0,TASK_COUNT=1  ...Simulation.sh
```

Larger campaigns pack 10 tasks/node with `--cpus-per-task=4` (10 × 4 = 40 cores,
matching `--extra-node-info=2:20:1`) and fan out over `--array`; the generation
controller (§5) drives this.

**Check (`test`), 1 s smoke — TRAIN 16 / TEST 4 / EVAL 2 tasks, 10 sims/task:**

```bash
sbatch --partition=test --array=0-0 --ntasks-per-node=16 --export=ALL,REPO=$PWD,SPLIT=train,TASK_COUNT=16,TASK_SIMS=10,TOTAL_TIME=1.0 ...Simulation.sh
sbatch --partition=test --array=0-0 --ntasks-per-node=4  --export=ALL,REPO=$PWD,SPLIT=test,TASK_COUNT=4,TASK_SIMS=10,TOTAL_TIME=1.0  ...Simulation.sh
sbatch --partition=test --array=0-0 --ntasks-per-node=2  --export=ALL,REPO=$PWD,SPLIT=eval,TASK_COUNT=2,TASK_SIMS=10,TOTAL_TIME=1.0  ...Simulation.sh
```

### Inference / Evaluation / Experiment (GPU)

The GPU scripts bake a whole-node allocation:

- `--nodes=1`, `--gres=gpu:8`, `--cpus-per-task=64`, `--mem=480G`
- `--time=1-00:00:00` (Inference, Experiment); `--time=12:00:00` (Evaluation)

Inference, Evaluation, and Experiment adapt to the GPUs they receive
(`SLURM_GPUS_ON_NODE`, capped by the optional `SRM_AND_SBI_GPUS`): more than one
GPU runs the torchrun path — data-parallel training (Inference) or work-sharded
estimation (Evaluation shards its EVAL videos; Experiment shards its
`(kind, cell)` work), each followed by a merge of the per-shard results — while
one GPU runs the original single-GPU path. Use the full `gpu` partition (whole
node, `gpu:8`) for validation unless a run is a genuinely tiny throwaway.

**Wall-limited training continues with `RESURRECT`.** Training checkpoints its
optimum each epoch. A run that will not reach its target epochs within the
partition wall is continued by relaunching the same submission with `RESURRECT=1`
(a `Submit.sh` knob, or `RESURRECT=1` in the raw `--export`): it loads that
checkpoint and resumes. The first job runs fresh; every continuation passes
`RESURRECT=1`. Repeat until the target epochs are reached. This works on the
single-GPU and the data-parallel paths alike.

To pre-submit the whole chain so each link starts **even if the previous job hit
the wall** (a timeout is recorded as a failure), gate each continuation on the
previous job id with `DEP=afterany:<jobid>` — `afterany`, not `afterok`, because an
`afterok` successor would never start after a wall-stopped predecessor:

```bash
id=$(DRYRUN=0 GPU_PART=gpu_test GRES=gpu:4 bash Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Submit.sh \
       inference TOTAL_TIME=5.0 TRAIN_TASKS=100 EPOCHS=10 HEARTBEAT=20 | grep -oP 'Submitted batch job \K\d+')
for _ in 1 2; do   # two continuations
  id=$(DRYRUN=0 GPU_PART=gpu_test GRES=gpu:4 DEP="afterany:$id" bash Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Submit.sh \
         inference TOTAL_TIME=5.0 TRAIN_TASKS=100 EPOCHS=10 HEARTBEAT=20 RESURRECT=1 | grep -oP 'Submitted batch job \K\d+')
done
```

**Check (`gpu_test`), single-GPU smoke — Inference:**

```bash
sbatch --partition=gpu_test --gres=gpu:1 --time=01:00:00 \
       --export=ALL,REPO=$PWD,TRAIN_TASKS=8,TEST_TASKS=2,EPOCHS=1 ...Inference.sh
```

**Long durations need a smaller batch.** Leaving `BATCH` unset uses the script
default (`PARAMETERS` batch size, 32). One early conv3d activation is
batch × ~1 GiB at 500 frames, so batch 32 OOMs a GPU with ~64 GB of VRAM at long durations
(e.g. a 10 s, 500-frame run); reduce `BATCH` accordingly.

**Smoke / check evaluation uses `--pool-mode unrestricted`.** An undertrained
posterior's probability mass can fall outside the prior box, and the default
`bounded` rejection sampling stalls on it. Pass `POOL_MODE=unrestricted` for any
smoke or undertrained-posterior evaluation; `bounded` is only for a fully
trained posterior. (See `VALIDATION.md` §3.4.)

```bash
sbatch --partition=gpu_test --gres=gpu:1 --time=01:00:00 \
       --export=ALL,REPO=$PWD,EVAL_TASKS=1,SUMMARY=both,POOL_MODE=unrestricted,TOTAL_TIME=1.0 ...Evaluation.sh
```

---

## 5. Generation controller and per-machine config

### Generation controller

`SRM_AND_SBI_DIMER_ALP_HPC_Generate_Controller.sh` is a rolling submit-and-gate
controller for a full generation campaign. It submits the per-split generation
arrays (train first), keeps within the QOS caps (≤40 running, ≤50 in-system),
then **hard-gates** the EVAL splits until every TRAIN + TEST job has reached
`COMPLETED`. If any train/test job ends in a non-`COMPLETED` state it stops
before submitting EVAL, so EVAL is never generated against broken data.

- **Dry run is the default.** With no override it prints the exact `sbatch`
  lines and submits nothing:

  ```bash
  bash Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Generate_Controller.sh
  ```

- **Live submission** requires `DRYRUN=0`. It polls for hours to days, so run it
  on the login node inside `tmux`/`screen`:

  ```bash
  DRYRUN=0 bash Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Generate_Controller.sh 2>&1 | tee ~/dimer_gen_controller.log
  ```

- `CASES` selects which dataset(s) to drive (`5s` | `2s` | `both`, default
  `both`), so one controller can run separate campaigns on separate clusters.
- Re-run a failed node/task under its **original global label** (same `SPLIT` /
  `TASK_SIMS` / `TOTAL_TIME`, with `TASK_OFFSET`/`TASK_COUNT` set to the gap):
  this fills the gap via the incremental-append mechanism and regenerates
  nothing good. Confirm label completeness before training.

### Per-machine config: `hpc_local.env`

Every script in this directory sources `Script_Bank/HPC/hpc_local.env` at
startup (via the resolved `REPO`, so it is found even under spooling). The file
is **gitignored** and never committed; each machine supplies its own values,
which keeps the committed scripts generic. Create it from the template:

```bash
cp Script_Bank/HPC/hpc_local.env.example Script_Bank/HPC/hpc_local.env
# then edit for this machine
```

Settings it provides:

| Variable | Purpose |
|----------|---------|
| `MACHINE_PROFILE` | profile in `machine_profiles.toml` (selects this machine's data/compute paths) — **required**; the scripts fail loud if unset |
| `CONDA_SETUP` | path to the conda `profile.d/conda.sh` that defines `conda` in a non-interactive shell |
| `MON` / `MON_OUT` | monitoring / batch-log output directory (**must already exist**) |
| `PART` | CPU partition for the generation controller (passed only when set) |
| `ACCT` | Slurm account, if the cluster requires one (passed only when set) |
| `USER_ME` | queue-owner username for the controller's polling (defaults to `$USER`) |
| `REPO` | auto-derived from the script location; override only if the repo is reached by a different path |

Anything left unset falls back to the script defaults (e.g.
`MON` → `$HOME/process_monitoring`), and an unset `PART`/`ACCT` leaves the
submit line at the script's baked defaults.

---

## 6. Special-situation entry points (run ad hoc)

The four stages in §1 form the standard, dispatcher-driven pipeline. A few entry
points sit outside that dispatcher — special-situation steps run by hand when a
specific case calls for them. They are deliberately kept out of the `Submit.sh`
dispatcher and the stage wrapper set, so the standard submission surface stays
exactly the four stages. Run them ad hoc with the recipe here rather than
building new launch machinery.

### Construct a posterior from a checkpoint

`Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Construction_Optimum_ANN.py` rebuilds a
`DirectPosterior` pickle from a saved `Optimum_ANN.pth` checkpoint **without
training**: it constructs the estimator exactly as Inference does, loads the
checkpoint weights into it, and pickles the posterior. Use it to

- move a trained model between machines — copy the small `.pth`, then construct
  the `.pkl` locally (a pickle embeds device state and is not portably moved);
- recover a posterior from a run that checkpointed but was stopped (e.g. a wall
  timeout) before it reached its own save block; or
- rebuild the posterior for an archived backup checkpoint (§7) — point
  `--checkpoint` at the backup and the matching backup posterior is produced.

It is single-process, single-GPU — there is nothing to shard — and its cost is
the one-time `torch.compile` in `build_maf` (a few minutes). Run it with plain
`python`, not `torchrun`.

Locally (add `--dry-run` first to validate the checkpoint and inputs without GPU use):

```bash
MACHINE_PROFILE=<profile> python \
    Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Construction_Optimum_ANN.py --total-time-seconds 2.0
```

On Slurm — a one-off `sbatch --wrap` that sources `hpc_local.env` (§5) for the
machine config and requests a single GPU on the check partition. Submit from the
repo root; the job name follows the §3 pattern with a `Construction` descriptor:

```bash
cd /path/to/srm-and-sbi-dimer-alp
sbatch --partition=gpu_test --gres=gpu:1 --cpus-per-task=16 --mem=64G --time=00:30:00 \
       --job-name=SRM_AND_SBI_DIMER_ALP_2S_50FPS_Construction \
       --output="$MON_OUT/%x_%A.out" \
       --wrap='set -e; . Script_Bank/HPC/hpc_local.env; source "$CONDA_SETUP"; \
               conda activate SRM_AND_SBI_ENVY_V0; export MACHINE_PROFILE="${MACHINE_PROFILE:?}"; \
               python -u Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Construction_Optimum_ANN.py --total-time-seconds 2.0'
```

By default reads `Labor/..._{timing_label}_Optimum_ANN.pth` and writes
`Posit/..._{timing_label}_Posterior.pkl`. Pass `--checkpoint <path>` to build from a
specific checkpoint instead — e.g. a provenance-named backup (§7)
`..._Optimum_ANN_TRAIN+TEST_200K+50K_Epoch_25_TEST_LOSS_-17.05.pth`; the output
posterior name is then derived to match (`Optimum_ANN` → `Posterior`, `.pth` →
`.pkl`), so the rebuilt `.pkl` tracks the weights it came from (override with
`--posterior`). Swap the `timing_label` token in the job name and
`--total-time-seconds` together for other durations.

---

## 7. Artifact backups

The trained artifacts — a posterior (`Posit/…_Posterior.pkl`) and its checkpoint
(`Labor/…_Optimum_ANN.pth`) — are overwritten in place whenever the stage that
produces them re-runs (a fresh Inference run, or a Construction rebuild, §6). The
canonical names never change: they are the live objects every downstream stage
(Evaluation, Experiment, Construction) loads. To keep a superseded model
identifiable and recoverable after it is overwritten, a copy is set aside under a
distinct name — automatically by a finished run, or by hand for an ad-hoc keep.

### Automatic provenance backup

A finished Inference run that loaded a TEST set (`--test-tasks > 0`, so a
model-selection loss exists) writes, alongside the canonical checkpoint and
posterior, a provenance-named copy of each. The suffix records what the bare
`state_dict` and posterior pickle cannot carry themselves — the run's training
scale and result — inserted before the extension:

    <original-stem>_TRAIN+TEST_<train>+<test>_Epoch_<n>_TEST_LOSS_<loss>.<ext>

- **`<train>` / `<test>`** — the number of TRAIN and TEST videos the run used, as
  thousands-tokens (`50000` → `50K`, `47500` → `47.5K`). Sizes are in the name
  because a test loss is only comparable at equal test-set size: a −17.00 measured
  on 50K selection videos is not the same result as −17.00 on 10K.
- **`Epoch_<n>`** — the number of epochs this job ran (`--epochs`), counted for the
  current job (a warm-started `--resurrect` run counts its own epochs, not the
  history of the weights it loaded).
- **`TEST_LOSS_<loss>`** — the checkpoint's best TEST loss, always exactly two
  decimals. A negative loss keeps its native `-`; a positive loss gets an explicit
  `+`; a value that rounds to zero is written `0.00`, with no sign.

For a 200K-train / 50K-test run over 25 epochs that reached a best test loss of
−17.05, the pair is:

    Labor/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Optimum_ANN_TRAIN+TEST_200K+50K_Epoch_25_TEST_LOSS_-17.05.pth
    Posit/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Posterior_TRAIN+TEST_200K+50K_Epoch_25_TEST_LOSS_-17.05.pkl

The backup is a copy, so the canonical `…_Optimum_ANN.pth` / `…_Posterior.pkl`
stay the active artifacts and a backup is never picked up as the live model. Across
warm-started runs the canonical checkpoint is additionally protected by
save-on-improvement — it is overwritten only when the current run beats the loaded
best on the TEST set — so each accepted optimum is preserved under its own backup
name. To make a backup the active model again, copy it onto the canonical name; to
rebuild its posterior from the checkpoint alone, point Construction's `--checkpoint`
at the backup (§6).

A run with no TEST set (`--test-tasks 0`, which trains on all of TRAIN and keeps
the last-epoch checkpoint) has no selection loss to name a backup by, so it writes
the canonical pair only.

### Manual ad-hoc keep

The automatic backup covers finished training runs. To preserve an artifact the
automatic scheme does not — for instance the current canonical *before* you
deliberately launch a run that will overwrite it, or a milestone worth tagging —
copy it aside by hand under a tag plus a day-month-year date, inserted before the
extension:

    <original-stem>_<TAG>_<DD.MM.YYYY>.<ext>

```bash
# preserve the current best 2S posterior + its checkpoint before an overwriting run
cp Posit/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Posterior.pkl \
   Posit/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Posterior_PREPROD_01.07.2026.pkl
cp Labor/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Optimum_ANN.pth \
   Labor/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Optimum_ANN_PREPROD_01.07.2026.pth
```

`<TAG>` is a short label for why the copy was kept (e.g. `PREPROD` before a
production run); `<DD.MM.YYYY>` is the date (e.g. `01.07.2026`).

### Both kinds land beside the original

Whether automatic or manual, the suffix sits before the extension, so a backup
never matches the `…_Posterior.pkl` / `…_Optimum_ANN.pth` names the pipeline loads —
it is kept, but never picked up as the active artifact. Keep each backup in the
same `Posit/` or `Labor/` directory as its original; if that storage tier is not
itself backed up (for example a scratch `data_bank_root`), also copy the backup to
a backed-up location.

---

## 8. Do not

- **Do not invent job or log names.** Use exactly
  `SRM_AND_SBI_DIMER_ALP_<timing_label>_<Stage>[_<SPLIT>]` (§3). No invented
  tokens such as `5S_PROD`, `smoke`, `mgpuval`, or an `HPC` segment.
- **Do not recompute node geometry.** Replicate the `#SBATCH` layouts and header
  examples in the scripts (§4) — the core counts, `--extra-node-info`, GPU
  count, and memory are deliberate.
- **Do not cancel and resubmit a correctly-running job for cosmetics.** A job
  whose name or log path is merely not your preference is still producing valid
  output; leave it running.
- **Do not edit the committed scripts for per-machine values.** Put machine
  differences in `hpc_local.env` (§5), not in the scripts.
- **Do not submit Simulation without `--array`.** Always pass `--array`
  (`--array=0-0` for a single node) so the log's `%a` is a clean node number.
- **Do not submit without a dry-run.** Preview every submission first — the
  `Submit.sh` helper or the generation controller with their default `DRYRUN=1`,
  or the entry point's own `--dry-run` — and submit only once the printed command
  and the resolved inputs check out (§2).

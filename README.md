# srm-and-sbi-dimer-alp

Simulation-based inference of reaction-diffusion parameters for the **DIMER** model (two-particle dimerization: A monomer, B mobile dimer, C immobile dimer).

This repository is a self-contained pipeline within the `srm-and-sbi` project: it simulates the DIMER reaction-diffusion system, renders the trajectories as diffraction-limited microscopy videos, and trains a neural posterior to recover the underlying rate and diffusion parameters from a video. It pairs that inference with a leak-proof train/test/eval data split and MAP-recovery validation on both simulated and real-microscopy data. See **[`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)** for the full scientific treatment.

## Naming conventions

Names are consistent across the surfaces a user touches:

- **GitHub repository** — kebab-case: `srm-and-sbi-dimer-alp`.
- **Python package** — snake_case: `srm_and_sbi_dimer_alp` (the repo name with hyphens normalized to underscores).
- **Runtime identifiers** (entry-point script names, output-file prefixes) — SCREAMING_SNAKE, composed as `[program]_[model]_[iteration]_[stage]_[sub-stage]`, e.g. `SRM_AND_SBI_DIMER_ALP_Simulation_RDS`. The prefix encodes provenance so data files remain self-describing once they leave the repository.
- The trailing **three-letter suffix** (`alp`) is the iteration tag; sibling iterations advance the suffix (`bet`, `chi`, …).

## Getting Started

### 1. Configure your machine profile

Copy `machine_profiles.example.toml` to `machine_profiles.toml`, edit it with your paths, and set the `MACHINE_PROFILE` env var to select your profile (e.g., add `export MACHINE_PROFILE=my_profile` to `~/.bashrc`).

### 2. Activate a compatible Python environment

The environment is **`SRM_AND_SBI_ENVY_V0`** — Python 3.13 with ReaDDy 2.0.14, sbi 0.26.1, numpy 2, zarr 2.18.7, and a hardware-specific PyTorch build (ROCm / CUDA / CPU). The full specification and step-by-step install — from scratch or from a per-machine snapshot — are in **[`env_snapshots/README.md`](env_snapshots/README.md)**, the canonical install guide.

- **If you already have a compatible env**, just activate it: `conda activate SRM_AND_SBI_ENVY_V0`.
- **If you need a fresh env**, follow the install guide in [`env_snapshots/README.md`](env_snapshots/README.md).

### 3. Install this package into the active env

`pip install -e . --no-deps` registers the `srm_and_sbi_dimer_alp` package as an editable install — source edits take effect immediately without reinstalling. Use **`--no-deps`**: the runtime dependencies are already provided by the environment, and a plain `pip install -e .` would re-resolve them — downgrading sbi and overwriting the hardware-specific PyTorch build (see the install guide's gotchas).

### 4. Verify

```bash
python -c "from srm_and_sbi_dimer_alp.parameterization import PARAMETERS; print(PARAMETERS.machine.name)"
```

Should print your profile name. If it raises a `ValueError`, the message points to the misconfiguration (env var unset, profile not found, missing keys, paths not existing).

## Pipeline stages

The pipeline is a five-stage chain. The first two stages form **generation** (**RDS → DLI**), and the remaining three consume its artifacts:

- **RDS** — reaction-diffusion simulation (ReaDDy-based): produces particle trajectories from sampled parameters.
- **DLI** — diffraction-limited imaging: renders each trajectory as a microscopy video (PSF + Poisson + EMCCD noise).
- **Inference** — neural posterior estimation: trains the posterior on TRAIN and selects the network on TEST.
- **Evaluation** — MAP-recovery validation: estimates parameters on the held-out EVAL set and scores recovery against the known ground truth.
- **Experiment** — real-data application: applies the trained posterior to real microscopy videos (no ground truth).

Each stage is an entry-point under `Script_Bank/Prime/`, run with the active `MACHINE_PROFILE` set. The stages communicate through on-disk artifacts, so a run can **target a single stage** rather than the whole chain: invoking a stage script directly (with `--split {train,test,eval}`) re-runs just that stage against the artifacts already on disk — for example, re-rendering videos with `SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py` (the DLI stage only) over trajectories that an RDS run already wrote to disk. Run any stage with `--help` for its full flag list.

For the exact mapping from each scientific concept and pipeline stage to the module, function, and on-disk artifact that implements it, see the implementation map in [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md).

### Generate a complete dataset

One command runs RDS → DLI for all three splits in the correct proportions:

```bash
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Generate_Datasets.py --core-tasks 100 --task-simulations 10 --total-time-seconds 10.0
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Generate_Datasets.py --core-tasks 100 --task-simulations 10 --total-time-seconds 10.0 --dry-run   # preview sizing only
```

### Train the posterior on TRAIN, selecting on TEST

```bash
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Inference.py --total-time-seconds 2.0 --tasks 8 --test-tasks 2 --epochs 50
```

#### Multi-GPU

Training, evaluation, and the real-data application auto-adapt to the GPUs they
are given. Launched under `torchrun` with one worker per GPU, training runs
data-parallel (`DistributedDataParallel`); evaluation shards its EVAL videos
across the workers and the experiment stage shards its `(condition, cell)` work,
each then merging the per-shard results into one report. With a single GPU the
same code collapses to the original single-GPU path — no flags to change.
The GPU count is read from the allocation (`SLURM_GPUS_ON_NODE`); the HPC
submitters wrap the `torchrun` launch, so on a whole-node `gpu` allocation a run
uses every GPU automatically. Four controls tune the behavior:

- **`--heartbeat N`** (Inference) — emit a within-epoch progress line every `N`
  batches (rank 0 only). Unset gives roughly four lines per epoch; a smaller `N`
  gives finer progress on the long epochs of a production run.
- **`RESURRECT=1`** (Inference HPC knob → the Prime `--resurrect` flag) — load the
  saved optimum checkpoint and continue training from those weights, so a run that
  hits the partition wall before its target epochs is continued by relaunching with
  `RESURRECT=1`. The first job runs fresh.
- **`SRM_AND_SBI_GPUS`** (env var) — cap the GPUs used; default is all allocated.
  Set it to `1` to force the single-GPU path even on a multi-GPU allocation.
- **`SRM_AND_SBI_NO_SYNC_BN=1`** (env var) — under data-parallel training, skip
  `SyncBatchNorm` so each worker keeps its own local batch statistics. The
  default (unset) keeps `SyncBatchNorm` on, which is the validated choice; the
  opt-out is faster but changes the batch statistics, so re-validate posterior
  recovery before relying on it.

### Validate and apply

Validate by MAP recovery on the held-out EVAL set, then apply the posterior to real microscopy videos:

```bash
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Evaluation.py --total-time-seconds 2.0 --eval-tasks 1 --summary both
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Experiment.py --total-time-seconds 2.0 --kinds ALP,BET --summary both
```

Evaluation reports per-parameter recovery accuracy + posterior calibration; Experiment reports inferred-parameter distributions per condition (no ground truth). Both write a self-contained report (figures + tables + arrays + a live `progress.log`) under `Posit/`. Run any stage with `--help` for the full flag list (`--summary {map,posterior,both}`, `--pool-mode {bounded,unrestricted}`, `--posterior-samples`, …; Evaluation additionally takes `--bin-mode {prior,quantile}`). The Inference, Evaluation, and Experiment stages also accept `--dry-run`, which resolves the machine profile and the input paths, prints what it would read and write (flagging anything missing), and exits before any compute (no GPU, no output directories) — run it before a long job or a queue submission; the dataset-generation orchestrator (`Generate_Datasets.py`) offers the same preview.

## Data split

The pipeline keeps three physically separate, leak-proof namespaces — **TRAIN** (gradient updates), **TEST** (per-epoch model selection), **EVAL** (held-out final validation) — each an independent draw, so validation never sees data the posterior optimized against. The generation command above produces all three in the correct proportions in one pass. The split rules and sizing formula are detailed in [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md), under the leak-proof data-split section.

## Timing

Recording length is supplied per run via the **required `--total-time-seconds`** argument; the frame count is derived from it and the fixed frame rate (`frame_count = total_time_seconds / frame_time_seconds`). Because duration lives only on the per-run object and never on a global default, no stage can silently inherit a stale length. The full fixed-cadence timing model is in [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md), under the per-run timing section.

## HPC and reproducibility

Generation is **non-deterministic by default** — prior sampling, particle placement, optics, and camera noise all draw fresh entropy each run — which matches the reaction-diffusion stepper itself being unseedable. A `--seed` flag opts into a deterministic base seed when one is wanted; the batch-generation scripts pass none. No scientific information is lost: the sampled parameters are persisted per task, so every (parameters, video) pair is recorded. Dataset integrity instead rests on a **global task index encoded in each file name** (`..._TASK_<tid>_<split>`), which keeps parallel fan-out, array overflow, and incremental appends from ever colliding on a filename or mixing split namespaces. The reproducibility characteristics and the task-index scheme are detailed in [`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md), under the non-deterministic generation and provenance section.

For running on a Slurm cluster, the [HPC operations runbook](Script_Bank/HPC/README.md) is the authoritative guide. Submissions are **dry-run first**: a unified submit helper (`Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Submit.sh`) and the generation controller both build and print the exact `sbatch` command for review, submitting only when `DRYRUN=0` is set, so a misconfigured job never reaches the queue.

## Debug mode & diagnostics

All five stages accept two optional flags (off by default — default runs are identical to validated behavior):

- **`--debug`** — per-step **checkpoints** and fail-loud **invariant checks** (no NaN/inf, probability matrices sum to 1, video frame count matches the network, training loss stays finite, output files written, …), with an end-of-stage `PASS / FAIL` summary on the console. A failed check aborts the run with a located error message.
- **`--debug-dump`** — implies `--debug`, and additionally persists a self-contained **Markdown report** (a plain-language *meaning* for every metric, plus a legend explaining each check), **PNG figures**, and a full **console transcript** (`console.log`) under `<data_bank>/Labor/Debug/<run>/<stage>/`. Skipped automatically if the disk is low on space.

The Evaluation and Experiment stages additionally write a live, tail-able `progress.log` beside their report so a long run can be monitored (`tail -f`). Their MAP-recovery reports are scientific deliverables and live under `Posit/`, kept separate from the `Labor/Debug/` diagnostic dumps.

## Inspecting the generated videos

[`notebooks/Video_Scrubber.ipynb`](notebooks/Video_Scrubber.ipynb) is an interactive frame-by-frame viewer (`ipywidgets` sliders over simulation index and frame). When the data lives on a remote GPU server, run Jupyter there and forward the port to your workstation:

```bash
# on the server (where the data + env live):
MACHINE_PROFILE=<profile> jupyter lab --no-browser --port 8888
# on your workstation, in a second terminal:
ssh -L 8888:localhost:8888 -p <ssh-port> <user>@<host>
```

then open the printed `http://localhost:8888/...` URL in your browser and run the notebook. For a fixed remote setup, a one-command wrapper (e.g. a gitignored `launch_remote_viewer.sh` that hard-codes the host) can run both ends at once.

## Structure

- `Script_Bank/Analysis` — analysis entry-point scripts
- `Script_Bank/HPC` — HPC-mode submission and orchestration scripts
- `Script_Bank/Prime` — stage scripts: simulation (`Simulation_RDS`, `Simulation_DLI`), dataset generation (`Generate_Datasets`), training (`Inference`), and validation (`Evaluation` on simulated EVAL data, `Experiment` on real microscopy)
- `srm_and_sbi_dimer_alp/` — main Python package (modules, support functions)

## Documentation

- **[`PROJECT_CONTEXT.md`](PROJECT_CONTEXT.md)** — scientific context, model parameters, ReaDDy semantics, the data split, timing model, reproducibility, and the inference workflow in full.
- **[`VALIDATION.md`](VALIDATION.md)** — environment setup, smoke tests, the validation methodology, and success criteria.
- **[`env_snapshots/README.md`](env_snapshots/README.md)** — the canonical install guide.

See workspace-root files (parent directory) for program-wide context.

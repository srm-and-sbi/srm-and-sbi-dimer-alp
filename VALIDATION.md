# Validation and Reproducibility — srm-and-sbi-dimer-alp

This guide describes how to set up the environment, confirm that each pipeline
stage runs, and validate a trained posterior. It covers three things:

1. **Environment setup** — one-time per machine.
2. **Smoke tests** — quick "does it run at all" checks for each entry point.
3. **Validation methodology** — how the pipeline establishes that the
   simulation and inference code is behaving correctly, including the
   reproducibility guarantees and the dual-duration checks.

Read through the whole document first so the dependencies between steps are
clear: the inference smoke test needs simulated data, and the validation checks
build on the smoke tests.

---

## 1. Environment setup (one-time per machine)

### 1.1 Activate or build the Python environment

The project environment is **`SRM_AND_SBI_ENVY_V0`** — a Python 3.13 scientific
stack (ReaDDy, NumPy 2, zarr, sbi) with a hardware-specific PyTorch build
(ROCm, CUDA, or CPU).

The complete specification and step-by-step install — from scratch or from a
per-machine snapshot — live in **[`env_snapshots/README.md`](env_snapshots/README.md)**,
which is the canonical install guide. Rather than repeating the version pins
here (they would drift out of sync), follow that guide.

There are two first-class ways to get a working environment:

- **Reuse an existing compatible environment.** If you already have an
  environment with the required stack (a shared lab environment, another
  machine's `SRM_AND_SBI_ENVY_V0`, or any environment that matches the
  specification), simply activate it:

  ```bash
  conda activate SRM_AND_SBI_ENVY_V0
  ```

- **Build a fresh environment.** Follow the from-scratch recipe in
  [`env_snapshots/README.md`](env_snapshots/README.md). It installs the
  conda-forge scientific layer, then the hardware-matched PyTorch wheel plus
  the sbi ecosystem, and finally the project package. The guide also documents
  every install gotcha (channel priority, the `psutil`/`ipython` requirement,
  the PyTorch backend table, and the `--no-deps` rule for the editable install).

**Verify** the core stack imports and reports the expected versions:

```bash
python -c "import readdy, torch, sbi, zarr, numpy, psutil; \
print('readdy', readdy.__version__, '| torch', torch.__version__, \
'| sbi', sbi.__version__, '| cuda/hip avail:', torch.cuda.is_available())"
```

(`cuda/hip avail` is `True` only when a GPU is actually visible — on an HPC node
that means inside a GPU allocation, not on the login node; on a CPU-only machine
`False` is expected.)

### 1.2 Install the package in editable mode

From inside the repository, with the environment active:

```bash
pip install -e . --no-deps
```

The `-e` (editable) flag means subsequent edits to package modules take effect
immediately without re-installation. Use `--no-deps`: the runtime dependencies
are already provided by the environment, and a plain `pip install -e .` would
re-resolve them — downgrading sbi and overwriting the hardware-specific PyTorch
build (see the install guide's gotchas).

**Verify**:

```bash
python -c "import srm_and_sbi_dimer_alp; print(srm_and_sbi_dimer_alp.__version__)"
```

### 1.3 Configure `machine_profiles.toml`

Copy the template and edit it for the current machine:

```bash
cp machine_profiles.example.toml machine_profiles.toml
$EDITOR machine_profiles.toml
```

Define at least one profile. Required keys per profile:

```toml
[my_local_profile]
running_mode      = "LOCAL"
script_bank_root  = "/full/path/to/srm-and-sbi-dimer-alp/Script_Bank"
data_bank_root    = "/full/path/to/Data_Bank"
compute_backend   = "GPU"
gpu_device_index  = 0
num_workers       = 0          # 0 = derive from available cores; positive = pin
```

`data_bank_root` must exist as a directory; create it now if it does not. The
output subdirectories (`Theta/`, `Video/`, `Posit/`, `Labor/`) are created
automatically by the entry-point scripts as needed. A machine with a large but
impermanent scratch filesystem may also set `scratch_data_bank_root` to route
the regenerable TRAIN and TEST bulk onto scratch while keeping everything that
must persist on backed-up storage; omit it for a single-tier machine.

### 1.4 Set the `MACHINE_PROFILE` environment variable

```bash
export MACHINE_PROFILE=my_local_profile
```

For persistence, add this to `~/.bashrc` on a local machine or to the HPC job
submission script.

**Verify** the configuration loads cleanly:

```bash
python -c "from srm_and_sbi_dimer_alp.parameterization import PARAMETERS; print(PARAMETERS.machine.name)"
```

This should print your profile name. The configuration validates at import time:
if the environment variable is unset, the profile is missing, a required key is
absent, or a root directory does not exist, it raises a clear `ValueError`
pointing to the misconfiguration. There is no silent fallback.

---

## 2. Smoke tests

These are quick checks that each entry point runs end-to-end with minimal inputs.
The point is to surface obvious problems (missing imports, wrong shapes, path
misconfiguration) before any longer run. The same structure applies to **both**
workflows in this repository — the biology reaction-diffusion pipeline
(sections 2.1–2.4b) and the Detector imaging-calibration pipeline (section 2.5,
`DETECTOR_WORKFLOW.md`) — which share the stage sequence generate → infer →
evaluate → (experiment). Every smoke passes `--total-time-seconds` (always
required), runs seedless (`--seed None`), and keeps the task and simulation counts
small. The **Detector calibration smoke (section 2.5) is the reference
configuration** the whole repository follows: one shared duration and the
three-split sizing `--tasks 25 / 5 / 2 --task-simulations 10` (TRAIN / TEST / EVAL
→ 250 / 50 / 20 videos). The biology sections below replicate that same sizing with
the biology entry points.

**Run order (both workflows).** The smoke is a small, fast replica of the full
production run, so it reproduces the production dependency chain rather than reordering
it for convenience. Run it in this order: first the **detector smoke (§2.5)** — the
detector workflow calibrates the imaging model; then **build the `Nuisance_DLI` (§2.5b)**
— the calibrated imaging turned into the samplable artifact the biology renderer draws
from; then the **biology smoke (§2.1–2.4b)** — it consumes that artifact at its DLI
stage. The section numbering is a reading convenience; this is the required execution
order. (The detector DLI itself needs no artifact: it draws imaging from the prior box,
because imaging is the detector's inference target.)

Two rules apply to every smoke and to every production run:

- **Seedless.** Pass `--seed None` explicitly on each stage. Every stage defaults
  to `None` (non-deterministic); a fixed seed freezes per-video particle
  placement, PSF, brightness, and detector noise across a split, collapsing the
  per-video variability the estimator must learn. The one deliberate exception is
  the theta-only regression test (section 3.2), which fixes `--seed` on purpose
  to check theta reproducibility.
- **Approval required.** No smoke or production run is launched without the
  project owner's explicit approval. Local checks (imports, stochastic-matrix
  diagnostics, dry-run prints) and code synchronization are fine; submitting any
  compute job — single-GPU or HPC, smoke or production — requires sign-off first.

On HPC, run smoke and check submissions on the short-lived `test` (CPU) and
`gpu_test` (GPU) partitions, leaving the production `general1` (CPU generation)
and `gpu` (train/eval) partitions for full runs. Take the partition,
node-geometry, and resource layout from each stage script's `#SBATCH` block and
header examples and replicate them verbatim — change only what the check
requires (typically the duration and the task counts). Do not recompute node
counts, core-per-node geometry, or GPU counts; the scripts already pin them.

### 2.1 RDS (reaction-diffusion simulation)

```bash
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py --total-time-seconds 2.0 --split train --tasks 25 --task-simulations 10 --seed None
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py --total-time-seconds 2.0 --split test  --tasks 5  --task-simulations 10 --seed None
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py --total-time-seconds 2.0 --split eval  --tasks 2  --task-simulations 10 --seed None
```

**Expected**: per split, one `.h5` trajectory per simulation under the RDS
trajectory directory `READY_TRACT/`, namespaced by split — for TRAIN,
`<data_bank>/Video/READY_TRACT/SRM_AND_SBI_DIMER_ALP_2S_50FPS_TASK_0_TRAIN/`
(`..._TASK_0_SIM_0_TRAIN.h5`, …), and one `.zarr` theta set per task at
`<data_bank>/Theta/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Theta_Set_TASK_0_TRAIN.zarr`
(with the `_TEST` / `_EVAL` namespaces for the other two splits). This generates
250 / 50 / 20 (train / test / eval) trajectories. Add `--verbose` to print the
sampled diffusion and reaction rates per simulation.

### 2.2 DLI (diffraction-limited imaging)

```bash
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py --total-time-seconds 2.0 --split train --tasks 25 --task-simulations 10 --video-dtype-bits 8 --seed None
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py --total-time-seconds 2.0 --split test  --tasks 5  --task-simulations 10 --video-dtype-bits 8 --seed None
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py --total-time-seconds 2.0 --split eval  --tasks 2  --task-simulations 10 --video-dtype-bits 8 --seed None
```

**Expected**: one `.zarr` video set per task at
`<data_bank>/Video/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Video_Set_TASK_0_TRAIN.zarr`
(and the `_TEST` / `_EVAL` namespaces), 250 / 50 / 20 videos in all. At 2 s each
video is `(100, 256, 256)` — 100 frames at 50 frames per second over a 256×256
detector grid — and every value is a non-negative integer pixel count. Because the
biology renderer marginalizes imaging, each task additionally writes the per-task draws
it used, both under `<data_bank>/Theta/`: a `Nuisance_DLI_Theta_Set` (the imaging
vectors drawn from the artifact) and a `Nuisance_SCOPE_Theta_Set` (the EMCCD camera
vectors drawn from the SCOPE box). `--video-dtype-bits 8` matches the bit depth the
estimator trains on and is also the DLI default.

**Requires**: two prerequisites. First, the RDS smoke (§2.1) must have run with the
same duration, splits, and `--task-simulations`; DLI reads the `.h5` trajectories and
theta set RDS wrote, so the two stages share `--tasks`, `--split`, and
`--task-simulations`. Second — the cross-workflow dependency — the `Nuisance_DLI`
artifact must already exist: the biology renderer **marginalizes imaging** by drawing
the photophysics from that artifact (and the camera from the SCOPE box), and fails loud
if it is absent. The artifact is a detector-side product, so the detector smoke (§2.5)
and the `Nuisance_DLI` build (§2.5b) must run first — see **Run order** in the section
intro.

### 2.3 Inference (posterior training)

```bash
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Inference.py --total-time-seconds 2.0 --tasks 25 --test-tasks 5 --epochs 5 --batch-size 8 --seed None
```

**Expected**: a network checkpoint at
`<data_bank>/Labor/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Optimum_ANN.pth` and a
version-portable estimator artifact at
`<data_bank>/Posit/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Estimator.npz`. The
training loop also writes a full-state resume file beside the checkpoint,
`<data_bank>/Labor/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Resurrect_State_ANN.pth`, updated
every epoch — its presence is what lets a later `--resurrect` hot-restart (see §2.4).
Five epochs on the small smoke dataset (250 train / 50 test videos) will not
produce a useful posterior, but they exercise the full training and save path,
including the network construction and the data loader. Because it passes
`--test-tasks 5`, the run loads the held-out TEST set, selects the checkpoint on
best test loss, and additionally writes a provenance-named backup of the
checkpoint and estimator (see PROJECT_CONTEXT §3 and the HPC runbook §7).

**Requires**: the RDS and DLI smoke tests (sections 2.1 and 2.2) must have run
first with the same `--task-simulations 10` and matching task counts. With too few
simulations the dataset is too small to form even a single training batch and the
run fails early.

**Small-GPU batch-size workaround**: the `Complex3DCNN` forward pass over 2 s videos
is memory-heavy (one 3D convolution alone needs roughly 6 GB), so even the smoke's
`--batch-size 8` can exceed a small GPU (for example a 4 GB card) and raise
`CUDA out of memory` partway into the first epoch. Two options:

- **Reduce the batch size** to `1` or `2` for a memory-light smoke check on a
  small GPU. This verifies the pipeline without producing a useful posterior:

  ```bash
  python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Inference.py --total-time-seconds 2.0 --tasks 25 --test-tasks 5 --epochs 5 --batch-size 1 --seed None
  ```

- **Train on a larger GPU**, or on CPU when no adequate GPU is available
  (select the CPU backend in `env_snapshots/README.md` and the CPU compute
  backend in the machine profile).

### 2.4 Resurrect (continue from an existing run)

```bash
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Inference.py --total-time-seconds 2.0 --tasks 25 --test-tasks 5 --epochs 5 --seed None --resurrect
```

**Expected**: because the inference smoke test (§2.3) left a full-state resume file
beside the checkpoint, this run **hot-restarts** — it prints a
`HOT RESTART: resumed full state ...` line reporting the resumed global epoch and
learning rate, then runs one more epoch continuing the exact optimizer + learning-rate
schedule (no re-converging, no LR reset). Because it passes `--test-tasks 5`, it loads
the held-out TEST set and selects the checkpoint on best test loss (overwritten only on
a new best), continuing the best-on-test bookkeeping §2.3 began. If the resume file is
absent (for example deleted, or a run predating this feature), the same command falls
back to a **cold** restart — loading the best checkpoint weights into a fresh optimizer
at the peak LR (a `RESURRECT (cold: ...)` line) — and then writes a resume file so the
next `--resurrect` hot-restarts. This makes incremental training across separate,
wall-time-limited invocations behave like one continuous run.

**Continuity check** (optional): run §2.3 with `--epochs 4` and note the epoch-4
learning rate; then run this `--resurrect` command — the printed global epoch resumes
where §2.3 left off and the learning rate continues from the saved schedule rather than
jumping back to the peak.

### 2.4b Evaluate and Experiment (biology)

The two remaining stages complete the biology smoke's `generate → infer → evaluate →
experiment` sequence, mirroring the Detector reference (§2.5, steps 4–5) with the
biology entry points and the same flags:

```bash
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Evaluation.py --total-time-seconds 2.0 --eval-tasks 2 --pool-mode unrestricted --seed None
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Experiment.py --total-time-seconds 2.0 --kinds ALP,BET --max-cells 2 --pool-mode unrestricted --seed None
```

`--pool-mode unrestricted` is mandatory here for the same reason as §2.5: the smoke
posterior is undertrained, so its mass falls outside the prior box and the default
`bounded` rejection pool stalls. Evaluation reports MAP recovery on the 20 EVAL videos
(requires the Inference smoke's estimator); Experiment applies the posterior to the real
MET recordings and writes a per-condition report for the ALP and BET cells (requires
those recordings present under the experiment data directory).

**Acceptance** (biology): all five stages exit zero; the Inference test loss descends
across the epochs on fresh per-video data (a flat curve signals a frozen-seed
regression); Evaluation reports MAP recovery on the 20 EVAL videos; Experiment writes a
per-condition report for the ALP and BET cells.

### 2.5 Detector calibration smoke test

The Detector calibration workflow (imaging-parameter inference with the
reaction-diffusion physics frozen to pure diffusion; see `DETECTOR_WORKFLOW.md`)
has its own five-stage smoke, run in order on a single GPU with plain `python`.
It is seedless and requires approval (both rules above). Use one duration for all
five stages — 2.0 s here; the pipeline is duration-general, but the DLI stage
checks its frame count against the RDS trajectories, so a single run must share
one duration. The inferred imaging vector is 6-dimensional — the five EMCCD camera parameters are marginalized as the SCOPE nuisance (drawn at the DLI stage, recorded separately as `Nuisance_SCOPE`), so the DLI stage writes both a `Theta_Set` (6 learnable) and a `Nuisance_SCOPE_Theta_Set` (5 camera) per task.

```bash
# 1. Diffusion-only trajectories + imaging theta, per split (seedless)
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_DETECTOR_Simulation_RDS.py --total-time-seconds 2.0 --split train --tasks 25 --task-simulations 10 --seed None
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_DETECTOR_Simulation_RDS.py --total-time-seconds 2.0 --split test  --tasks 5  --task-simulations 10 --seed None
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_DETECTOR_Simulation_RDS.py --total-time-seconds 2.0 --split eval  --tasks 2  --task-simulations 10 --seed None
# 2. Render videos, per split (seedless; 8-bit, matching what the estimator trains on)
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_DETECTOR_Simulation_DLI.py --total-time-seconds 2.0 --split train --tasks 25 --task-simulations 10 --video-dtype-bits 8 --seed None
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_DETECTOR_Simulation_DLI.py --total-time-seconds 2.0 --split test  --tasks 5  --task-simulations 10 --video-dtype-bits 8 --seed None
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_DETECTOR_Simulation_DLI.py --total-time-seconds 2.0 --split eval  --tasks 2  --task-simulations 10 --video-dtype-bits 8 --seed None
# 3. Train the imaging posterior (single GPU)
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_DETECTOR_Inference.py --total-time-seconds 2.0 --epochs 5 --tasks 25 --test-tasks 5 --batch-size 8 --seed None
# 4. MAP recovery on the held-out EVAL set
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_DETECTOR_Evaluation.py --total-time-seconds 2.0 --eval-tasks 2 --pool-mode unrestricted --seed None
# 5. Real-data application
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_DETECTOR_Experiment.py --total-time-seconds 2.0 --kinds ALP,BET --max-cells 2 --pool-mode unrestricted --seed None
```

This generates 250 / 50 / 20 (train / test / eval) videos. `--pool-mode
unrestricted` is mandatory on Evaluation and Experiment: the smoke posterior is
undertrained, so its mass falls outside the prior box and the default `bounded`
rejection pool stalls; `bounded` is for a well-trained (production) posterior.
`--video-dtype-bits 8` matches the synthetic-video bit depth the estimator trains
on (raw experimental frames are 16-bit, converted to 8-bit for inference) and is
also the DLI default. Unlike biology, the detector DLI needs no `Nuisance_DLI` artifact
— it draws imaging from the prior box, because imaging is what the detector infers. As
with biology, Evaluation and Experiment require the estimator trained in step 3, and
Experiment additionally requires the real ALP/BET recordings staged under the experiment
data directory.

**Acceptance**: all five stages exit zero; the Inference test loss descends across
the five epochs on fresh per-video data (a flat curve signals a frozen-seed
regression); Evaluation reports MAP recovery on the 20 EVAL videos; Experiment
writes a per-condition report for the ALP and BET cells.

### 2.5b Build the Nuisance_DLI (detector → biology bridge)

The biology DLI stage draws its imaging from the `Nuisance_DLI` artifact — the detector
calibration turned into a samplable imaging distribution. It is therefore built **after**
the detector workflow (§2.5) and **before** the biology smoke (see **Run order** in the
section intro). The artifact is Detector-namespaced
(`<data_bank>/Posit/SRM_AND_SBI_DIMER_ALP_DETECTOR_2S_50FPS_Nuisance_DLI.npz`) and shared:
the detector workflow produces it, the biology workflow consumes it.

The construction (`Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py`)
reads a value-based spec whose `posterior_sample_pool_choice` sets how the calibration
becomes the samplable artifact. **Production** uses `raw` — the faithful calibration,
resampled from the detector estimator's posterior over the real recordings; it is a GPU
step and needs the estimator trained in the detector Inference stage. For a **smoke** the
imaging values need only be valid, so use `box_user` with each parameter's range set to its
imaging prior box — the six learnable imaging parameters' `PRIOR_RANGE` values defined in
`srm_and_sbi_dimer_alp/detector_parameterization.py` (the `DETECTOR_PARAMETERIZATION` table).
This is a per-parameter uniform over the prior that needs no estimator, no GPU, and no
recordings — the smoke's fast stand-in for the calibration, occupying the same recipe slot. Write the spec to
`<data_bank>/Posit/SRM_AND_SBI_DIMER_ALP_DETECTOR_2S_50FPS_Nuisance_DLI_Spec.toml`:

```toml
[block]
posterior_sample_pool_choice = "box_user"   # per-parameter uniform over the ranges below; no pool / estimator / GPU
pool_mode = "bounded"
# Each [low, high] is that parameter's imaging prior PRIOR_RANGE (log10), defined in
# srm_and_sbi_dimer_alp/detector_parameterization.py (the DETECTOR_PARAMETERIZATION table;
# also det.theta_lower_bound() / det.theta_upper_bound()). The numbers below are that prior
# at the current version — if the imaging prior changes there, re-derive these to match.
# box_user validates every range against the prior box and fails loud otherwise, so a stale
# copy cannot silently pass; a uniform over the full prior is trivially valid smoke imaging.
[imaging.mu_r]
low = 0.0
high = 0.3
[imaging.sigma_r]
low = -1.0
high = -0.25
[imaging.mu_pc]
low = 2.0
high = 2.75
[imaging.sigma_pc]
low = -0.75
high = 0.0
[imaging.prob_photo_bleach]
low = -2.0
high = -0.5
[imaging.lambda_rate]
low = 0.0
high = 1.0
```

Then build it (CPU, seconds):

```bash
python Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py --total-time-seconds 2.0 --build
```

**Expected**: `<data_bank>/Posit/SRM_AND_SBI_DIMER_ALP_DETECTOR_2S_50FPS_Nuisance_DLI.npz`
plus a `..._Nuisance_DLI_Analysis/` directory (a `report.md` and a 1-D marginals figure).
The biology DLI (§2.2) then loads it; with `box_user` over the prior box every drawn imaging
vector lies inside the prior (the report's "frac outside prior" is 0). The full set of
representations and the calibration rationale are in the companion note
`SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.md`.

### 2.6 Running smokes on HPC

The same smoke runs on a cluster through the committed wrappers — the biology
`SRM_AND_SBI_DIMER_ALP_HPC_*` set and the Detector `SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_*`
set — which exercise the paths a single workstation cannot: **multi-CPU
generation** (many tasks packed per node, fanned out with a Slurm `--array`) and
**multi-GPU training, evaluation, and experiment** (data-parallel training;
sharded evaluation/experiment with a separate `--merge` step). Each GPU stage also
scales **across nodes** with `NODES=N` — `--gres` is per node, so
`world_size = NODES * GPUs-per-node`; add `NODES=2` to a GPU smoke to exercise the
multi-node path (inference binds all ranks in a c10d rendezvous; the sharded stages
just need the shared output dir for the merge). Running the smoke on HPC therefore
validates the multi-node and multi-GPU machinery, not only the stage logic.

Submit on the short-lived check partitions (`test` for CPU generation, `gpu_test`
for GPU stages), leaving the production partitions for full runs. Take the
partition, node geometry, and GPU counts from each wrapper's `#SBATCH` block and
header, and replicate them — changing only the duration and the small smoke
counts (section 2.5). Submit through the dispatchers — `..._HPC_Submit.sh` and
`..._HPC_Generate_Controller.sh` (biology and Detector alike) — which default to
dry-run (`DRYRUN=1` prints the resolved `sbatch` line and submits nothing; set
`DRYRUN=0` only after the printed command is verified); the fleet-sync utility
follows the same convention. The per-stage wrappers themselves are plain
`sbatch` scripts with no dry-run mode, which is why submission goes through the
dispatchers. The end-to-end dependency-chained sequence is in
the HPC runbook (`Script_Bank/HPC/README.md`).

### 2.7 Production runs

Production runs share the same stages, the seedless rule, and the approval
requirement, but are sized to the scientific goal and the machine; many
configurations are valid, and every value below is a recommended reference point,
not a mandate. As a reproducible reference, a production campaign holds the dataset
size fixed — 200K TRAIN / 50K TEST / 25K EVAL videos — for every recording
duration. What changes with duration is how those videos are packed into
generation tasks, and how large the training batch can be: longer videos are
larger, so the simulations per task fall (and the task count rises to hold the
video totals constant), and the recommended training batch size falls to fit GPU
memory.

| duration | sims/task | CORE (TRAIN+TEST) | TRAIN | TEST | EVAL | batch | videos (TRAIN/TEST/EVAL) |
|---|---|---|---|---|---|---|---|
| 1 s  | 1000 |  250 |  200 |  50 |  25 | 64 | 200K / 50K / 25K |
| 2 s  | 1000 |  250 |  200 |  50 |  25 | 32 | 200K / 50K / 25K |
| 5 s  |  500 |  500 |  400 | 100 |  50 | 16 | 200K / 50K / 25K |
| 10 s |  250 | 1000 |  800 | 200 | 100 |  8 | 200K / 50K / 25K |
| 20 s |  125 | 2000 | 1600 | 400 | 200 |  4 | 200K / 50K / 25K |

The split follows the dataset-sizing rule in `Generate_Datasets.py`: TRAIN, TEST,
and EVAL are 0.8, 0.2, and 0.1 of CORE (CORE = TRAIN + TEST), with EVAL floored at
a minimum recovery count.

The epoch budget and the batch size are both flexible; the batch column is a
recommendation that scales down with duration to fit GPU memory — adjust it to the
available hardware. The epoch budget is per invocation. Splitting it across
`--resurrect` rounds (for example, **25 epochs run twice**) is the crash-safe path for
wall-time-limited queues and costs nothing extra: each round hot-restarts from the full
resume file, so the two rounds continue one uninterrupted optimizer + learning-rate
schedule — equivalent to a single **50-epoch** run, without the re-convergence a cold
restart spends at each requeue. The in-run LR warm restart (see `train_loop`) is
a within-run plateau-escape at the LR floor, and its state persists across requeues too,
so the sawtooth is continuous across rounds.

**The sharded stages balance at video granularity, so any task count divides evenly over
any worker count.** Evaluation and the calibration diagnostic enumerate the individual
`(task, sim)` videos and deal *those* round-robin, rather than splitting whole tasks. This
matters because every video costs about the same, so a whole-task split would set the wall
clock by the heaviest rank: ten tasks over eight workers would give two of them twice the
work of the other six, and the run would cost what sixteen tasks should. Video-level
sharding balances any combination to within a single video, so the task count needs no
relation to `world_size` (`world_size = NODES × GPUs-per-node`) and no configuration is
penalized.

That imbalance was also, historically, a **failure mode** worth understanding. `torchrun`'s
exit barrier defaults to 300 s: ranks that finish early wait five minutes for the rest and
then tear down the rendezvous, killing any rank still working and discarding its shard,
after which the wrapper's `set -e` aborts before the merge and no report is written. A
single-node launch (`torchrun --standalone`) has one agent and so never hits this; the
multi-node launch places one agent per node and does. The wrappers now raise that timeout
(`TORCHELASTIC_EXIT_BARRIER_TIMEOUT`, override with `EXIT_BARRIER=`) as a safety net, but
with video-level sharding the skew that triggered it no longer arises.

Evaluation uses `--pool-mode bounded` — the well-trained-posterior default, in
contrast to the smoke's `unrestricted` — because the EVAL parameters *are* prior
draws, so rejection sampling inside the prior box is both correct and efficient.
**Experiment uses `--pool-mode unrestricted`**, even with a well-trained posterior:
the real recordings are not prior draws, so the posterior conditioned on one can sit
largely or wholly outside the prior box, and bounded rejection sampling then accepts
essentially no proposals and the stage hangs (sbi reports `Only 0.000% proposal
samples are accepted`). The pool mode therefore follows the *data*, not the training
quality: bounded for synthetic parameters drawn from the prior, unrestricted for real
recordings. Generation is seedless.

Submit on HPC: generation on the CPU partition (tasks packed per node, fanned out
with `--array` per split), training and evaluation on the GPU partition
(multi-GPU, optionally multi-node via `NODES=N` — `world_size = NODES *
GPUs-per-node`). Partitions and node geometry are per-machine, supplied by each
cluster's `hpc_local.env`; the submission pattern and commands are in the HPC
runbook (`Script_Bank/HPC/README.md`). The biology DLI stage runs with the
imaging block marginalized (§3.1); a full biology production run, like the detector
production re-run, is a separate, approval-gated step. As
with smokes, no production run is submitted without the project owner's explicit
approval.

---

## 3. Validation methodology

Validation rests on three ideas: the simulation and imaging code is checked for
**semantic equivalence** against the reference scientific behavior; the
pipeline's reproducibility is pinned down by a **theta-only regression test**;
and the duration-parameterized code is exercised at **two durations** to confirm
the frame-count arithmetic is correct throughout.

### 3.1 Semantic equivalence (three pillars)

The simulation and imaging stages thread explicit, per-function random-number
generators rather than relying on a single shared process-wide stream. As a
result the code is validated **semantically** — confirming it produces the
correct scientific behavior — rather than by per-element numerical matching
against any particular reference run. Equivalence rests on three pillars:

1. **Theta-sampling determinism (mathematically exact).** The theta sampler
   draws from the prior with `np.random.default_rng(seed).uniform(low, high, size)`
   over fixed bounds. The same seed produces bit-identical theta vectors. This
   is directly verifiable — extract the theta set written by a seeded RDS run
   and compare it across runs:

   ```bash
   python -c "
   import zarr
   z = zarr.open('<data_bank>/Theta/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Theta_Set_TASK_0_TRAIN.zarr', mode='r')
   print(z[:].tolist())
   "
   ```

2. **Reaction-diffusion primitive equivalence.** ReaDDy is a stochastic
   simulator — its stepper draws its own random numbers for diffusive motion
   and reaction-event timing, so trajectories differ run-to-run even at a fixed
   system specification (species, reactions, rates, simulation box, particle
   complement). What is deterministic, and what this pillar verifies, is the
   construction of that specification: the system builder and the simulation
   builder produce a ReaDDy system with the declared rates, geometry, and
   observables expected for a given theta. The verbose RDS banner (`--verbose`)
   prints the diffusion and reaction rates, so the constructed system can be
   inspected directly. The stepper's own internal randomness is the intended
   source of run-to-run variability (see the theta-only regression test,
   section 3.2).

3. **Imaging-pipeline functional equivalence.** The DLI stage applies a Gaussian
   point-spread function (erf-based pixel integration), an EMCCD detector model
   (Poisson photoelectrons, stochastic Gamma electron multiplication, and gain-independent Gaussian read noise), a brightness state
   machine, and the duration-independent photobleaching model. The verbose DLI
   banner (`--verbose`) prints the detector parameters, and a rendered video
   (via `--show`) shows sparse fluorescent spots on a near-zero background with
   plausible pixel-value ranges. The pipeline produces videos of the correct
   shape, dtype, and value distribution. The **biology** DLI stage renders through
   the shared, source-agnostic
   `render_dli_video` with the imaging block marginalized — the six photophysics drawn
   per simulation from the `Nuisance_DLI` artifact and the five camera parameters from
   the SCOPE box, both recorded beside the reaction-diffusion labels. The biology DLI
   smoke (section 2.2) and the long-duration DLI leg (section 3.3) exercise this path.

**Acceptance**: pillars (1)–(3) hold — theta is bit-reproducible at a fixed
seed, the constructed reaction-diffusion system matches the declared model, and
the imaging output is correctly shaped and physically plausible.

### 3.2 Reproducibility — theta-only regression test

The `--seed` flag defaults to `None`, which means **non-deterministic by
design**: each run draws fresh entropy for prior sampling, particle placement,
imaging noise, and network initialization. This matches the inherent
stochasticity of experimental fluorescence data and is the intended production
behavior. To exercise the reproducibility guarantee, pass `--seed` explicitly. This is the
one deliberately seeded check; every smoke in section 2 is run seedless
(`--seed None`).

The acceptance bar is **theta-only** reproducibility — the prior-sampling stage
is fully seeded, the downstream stages are not. Run the pipeline three times at
the same explicit seed:

```bash
DB=<data_bank>            # the data_bank_root from your machine profile
for i in 1 2 3; do
    rm -rf "$DB/Theta" "$DB/Video"
    python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py \
        --total-time-seconds 2.0 --tasks 1 --task-simulations 5 --seed 42
    find "$DB/Theta/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Theta_Set_TASK_0_TRAIN.zarr" \
        -type f | sort | xargs md5sum
done
```

**Acceptance**:

- **Theta sets are bit-identical** across all three runs (the per-file md5
  listings match line for line). This proves the prior-sampling RNG is seeded and correctly propagated
  through the entry point; if a future code change ever drops that seed
  handling, this check catches the regression immediately.
- **Trajectories (`.h5`), videos (`.zarr`), and inference outputs
  (`.pth`/`.npz`) differ across runs**, and that is expected, not a regression.
  The reaction-diffusion stepper draws from an internal random source that is
  not seedable, so trajectories vary; the videos inherit that variability
  through their trajectory input (even though the imaging pipeline's own noise
  RNGs are fully seeded); and the trained network inherits it through its
  training data. Posterior summaries should converge across runs within
  statistical uncertainty given enough training data, but bit-identity past the
  theta stage is neither expected nor required.

In short: **theta is reproducible at a fixed seed; trajectories, videos, and
trained networks vary by design.**

**Pre-launch fan-out label check.** Because generation is seedless by design,
dataset integrity does not rest on value reproducibility past the theta stage; it
rests on the output file labels being unique across the fan-out, so a silent label
collision cannot let one task overwrite another's output. An ad-hoc diagnostic
asserts exactly that before a large generation fan-out — especially an incremental
grow that appends tasks with a task offset — and also confirms the theta sampler's
seeding behaves as designed (two default draws differ, so no seed is silently
forced; an explicit seed reproduces the draw bit-for-bit). Run it by hand on any
machine with the package installed and a valid `MACHINE_PROFILE`, once per recording
duration (the duration sets the timing label whose labels are checked):

```bash
MACHINE_PROFILE=<profile> python \
    Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Seeding_Validation.py \
    --total-time-seconds 2.0
```

It builds path strings and samples small in-memory arrays only — it never reads or
writes the data bank and needs no GPU. It writes no files: it prints one
`[PASS]`/`[FAIL]` line per check and a final `RESULT`, exiting nonzero on any
failure so it can gate a generation launch from a script. This check is a
standalone reproducibility diagnostic, not one of the biology pipeline stages,
and is kept out of the stage dispatcher. See the companion note
`Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Seeding_Validation.md` for what each
check covers, how to read a failure, and the precise scope of what it does and does
not guarantee.

### 3.3 Dual-duration checks (2 s and 10 s)

The codebase is duration-parameterized: the recording length is supplied per run
via the required `--total-time-seconds`, and the frame count follows from it.
Validation runs both a short and a long duration end-to-end to confirm the
arithmetic is correct throughout the pipeline — not only at the short duration
where a stale length default would happen to be right.

The frame count is **`frame_count = total_time_seconds / frame_time_seconds`**,
with the frame time fixed at 0.020 s (50 frames per second). So:

- **2 s → 100 frames**: video shape `(100, 256, 256)`.
- **10 s → 500 frames**: video shape `(500, 256, 256)`; trajectory length,
  video frame count, and file sizes all scale five-fold; outputs are namespaced
  by their timing label (`10S_50FPS`) so they never collide with the 2 s
  outputs.

Repeat the smoke tests with the long duration (seedless). The DLI and Inference
legs run on the biology DLI path (section 3.1); the duration arithmetic is also
confirmed through the RDS leg and the derived-frame-count check below:

```bash
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py \
    --total-time-seconds 10.0 --tasks 1 --task-simulations 5 --seed None --verbose
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py \
    --total-time-seconds 10.0 --tasks 1 --task-simulations 5 --seed None --verbose
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Inference.py \
    --total-time-seconds 10.0 --tasks 1 --epochs 1 --seed None --verbose
```

The inference network's temporal depth is derived from the duration: the video
encoder is constructed with `n_frames = total_time_seconds / frame_time_seconds`
(100 for 2 s, 500 for 10 s), and it asserts that the input frame count matches,
so a duration/data mismatch fails loudly rather than silently truncating.
Confirm the derived frame count:

```bash
python -c "
from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming
t = RunTiming(total_time_seconds=10.0, frames=PARAMETERS.simulation.timing)
print('frame_count =', t.frame_count)
"
```

**Acceptance**: both durations run end-to-end without errors; video shapes are
`(100, 256, 256)` and `(500, 256, 256)`; the encoder accepts the matching frame
count at each duration.

### 3.4 Validating a trained posterior (MAP recovery)

Beyond confirming the code runs, a trained posterior is validated on data it has
never seen, using the two MAP-recovery stages.

- **Simulated recovery** (`Evaluation.py`): for each held-out EVAL video the
  maximum-a-posteriori parameter vector is estimated and compared to the known
  ground truth. This reports per-parameter recovery accuracy and posterior
  calibration (whether the credible intervals contain the truth at their nominal
  rate — roughly 50% for the interquartile range, roughly 90% for the 5–95%
  interval; under-coverage signals an overconfident posterior, over-coverage an
  underconfident one).

  ```bash
  python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Evaluation.py \
      --total-time-seconds 2.0 --eval-tasks 1 --summary both --pool-mode unrestricted
  ```

  Use `--pool-mode unrestricted` for a smoke or check run. The smoke-tested
  posterior is undertrained, so much of its probability mass lies outside the
  prior box; the default `bounded` pool draws candidates by rejection sampling
  within the prior and stalls when almost every draw is rejected. The
  `unrestricted` pool samples the flow directly and does not stall. Switch back
  to the default `bounded` pool only once the posterior is well trained (see
  the pool-mode note below).

- **Real-data application** (`Experiment.py`): the same estimator is applied to
  experimental microscopy videos, which have no ground truth. Each recording is
  split into model-length windows and the inferred-parameter distribution is
  reported per experimental condition. This is the scientific end use of the
  posterior, not a correctness check.

  ```bash
  python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Experiment.py \
      --total-time-seconds 2.0 --summary both --pool-mode unrestricted
  ```

Both stages report two complementary views side by side, because they answer
different questions: the **MAP point estimate** (the posterior mode) and the
**posterior credible summary** (median plus interquartile range). A sharp
posterior at the wrong location and a broad posterior at the right one are
distinguishable only when both are shown. Each stage writes a self-contained
report (figures, tables, arrays, and a live, tail-able `progress.log`) under
`Posit/`.

For a small or undertrained posterior whose probability mass can fall outside
the prior box, use the `unrestricted` candidate pool (`--pool-mode unrestricted`)
so candidate sampling does not stall; for a well-trained posterior the default
`bounded` pool (rejection sampling within the prior) is correct **on Evaluation
only** — its EVAL parameters are synthetic prior draws, so the posterior mass
lies inside the box. Experiment stays `unrestricted` even with a well-trained
posterior: the pool mode follows the *data*, not the training quality, and the
experimental recordings are not prior draws (see the pool-mode rule in
section 2.7). Run any stage with `--help` for the full flag list.

---

## 4. Troubleshooting

### Import-time configuration errors

If `python -c "import srm_and_sbi_dimer_alp.parameterization"` raises:

- **"MACHINE_PROFILE environment variable is not set"** — set it (selecting the
  active profile, section 1.4).
- **"machine_profiles.toml not found"** — create it from the template
  (configuring `machine_profiles.toml`, section 1.3).
- **"Profile '…' not found"** — the profile name in the environment variable
  does not match a section in the TOML file.
- **"missing required keys"** — the profile is missing one of the required keys
  listed under configuring `machine_profiles.toml` (section 1.3).
- **"… is not a directory"** — `script_bank_root` or `data_bank_root` does not
  exist on disk; create it or fix the path.

### `ImportError: No module named 'psutil'` (or `ipython`)

These packages are not pulled in transitively and must be installed explicitly;
without `psutil` the simulation stage fails immediately. Reinstall the
environment per `env_snapshots/README.md` (the gotchas section), or add the
missing package to the active environment.

### ReaDDy import error

ReaDDy is conda-distributed. If `python -c "import readdy"` fails, the
environment was not built correctly; rebuild it per `env_snapshots/README.md`
and verify with `conda list readdy`.

### `CUDA out of memory` during inference

The default batch size over full-resolution videos can exceed a small GPU's
memory. Reduce `--batch-size` to 1 or 2, train on a larger GPU, or use the CPU
backend (the inference smoke test's small-GPU workaround, section 2.3).

### Wrong PyTorch backend or CUDA/ROCm mismatch

If PyTorch reports a backend mismatch at runtime, the installed wheel does not
match the hardware. Reinstall the matching build using the PyTorch backend table
in `env_snapshots/README.md` (AMD → ROCm; NVIDIA → a `cuXXX` wheel whose CUDA
version does not exceed the driver's maximum; no/small GPU → the CPU build).

### `--resurrect` fails with "file not found"

`--resurrect` requires a prior checkpoint at the expected path. Run inference
once without `--resurrect` first to produce the initial checkpoint.

### DLI fails to load the theta set

DLI reads the theta set written by RDS. If it reports a missing file, ensure the
matching RDS run completed first and produced both the trajectories and the
theta set, with the same `--tasks` and `--task-simulations`.

### A long generation run hangs without finishing

If a long, memory-tight generation run stalls (rather than erroring), confirm
resource usage stays flat with the simulation stages' `--probe` flag, which logs
per-simulation thread, file-descriptor, and memory counts. Per-simulation
reaction-diffusion kernels are released inside the generation loop so that
threads and resident memory stay flat across arbitrarily many simulations; the
`--probe` instrumentation is how that is confirmed.

---

## 5. Success criteria

A correctly set up and validated installation has:

- A working `SRM_AND_SBI_ENVY_V0` environment (or a reused compatible one) with
  the package installed editable, and a `machine_profiles.toml` configured for
  the machine.
- All six biology smoke-test invocations (RDS, DLI, Inference, Inference
  `--resurrect`, Evaluation, and Experiment — sections 2.1–2.4b) running
  end-to-end on minimal inputs, meeting the biology acceptance in section 2.4b.
  (The biology DLI stage renders
  through the shared `render_dli_video` with the imaging block marginalized — see the
  imaging-pipeline pillar in section 3.1 — so it requires the `Nuisance_DLI` artifact to
  be built first, per the DLI smoke prerequisites in section 2.2.)
- The Detector calibration five-stage smoke test (section 2.5) running end-to-end,
  seedless, with its small overrides.
- The three-pillar semantic-equivalence checks holding: theta bit-reproducible
  at a fixed seed, the constructed reaction-diffusion system matching the
  declared model, and imaging output correctly shaped and physically plausible.
- The theta-only reproducibility regression passing: theta sets bit-identical
  across three same-seed runs, with trajectories, videos, and trained networks
  varying by design.
- Both the 2 s and 10 s configurations running end-to-end with the correct
  frame counts (100 and 500).
- A trained posterior validated by MAP recovery on the held-out EVAL set, with
  recovery accuracy and posterior calibration reported.

---

## 6. Assurance roadmap (not yet implemented)

Everything above describes validation the repository performs today. The items
below are engineering-assurance work the project has deliberately deferred
behind the scientific program. They are listed here so a reader can distinguish
completed validation from planned assurance — none of them exists yet.

- **Automated pytest suite.** The checks in section 3 are run by hand; an
  automated suite would pin them as regression tests. The natural targets:
  parameter-table schemas and ordering (the theta layout the estimator depends
  on), estimator artifact save/load round-trips with checksum and schema
  rejection of a corrupted or mismatched file, shard/merge coverage and balance
  invariants for the sharded stages (every video assigned exactly once, ranks
  balanced to within one video), the deterministic renderer components (PSF
  pixel integration at a fixed input), the EMCCD noise model's mean/variance
  behavior against its analytic form, and a CLI `--dry-run` smoke driven by a
  temporary machine profile so the entry points are exercised without a data
  bank.

- **CPU-only continuous integration.** A CI job running compilation, the unit
  tests above, and packaging checks on every push would catch import breakage,
  schema drift, and dependency-resolution regressions before they reach a
  compute run. CPU-only keeps it runnable on any hosted runner; the GPU paths
  remain covered by the smoke tests.

- **Tagged releases plus a `CITATION.cff`.** Versions are currently identified
  by the package version string; tagged releases would make each version
  immutable and retrievable, and a `CITATION.cff` would make them citable, so a
  result can name the exact code that produced it.

- **A minimal runnable example.** A small reference dataset with expected
  outputs would let a new user (or a reviewer) confirm an installation
  end-to-end in minutes, without generating data first — the smoke tests verify
  that the code runs, but only against data the user must produce.

- **A per-release machine-readable validation manifest.** Each release would
  ship a manifest recording the commit, the environment specification, the
  dataset checksums, the per-parameter recovery and calibration metrics, and a
  pass/fail verdict per criterion in section 5 — turning "validated" from a
  narrative claim into a checkable record.

- **Decoupling module import from machine configuration.** The configuration
  currently validates at import time (section 1.4), which is a deliberate
  fail-loud choice but couples every import — including a test collector's — to
  a valid `MACHINE_PROFILE`. Loading the profile at the CLI boundary instead
  would keep the fail-loud behavior for runs while letting the package import
  cleanly on an unconfigured machine.

- **An executable derivation script for the localization-derived imaging
  priors.** The imaging prior ranges trace to localization analyses of the
  public reference recordings; an executable script reproducing that chain —
  from the public accession to the fitted values — would make the prior
  derivation itself reproducible rather than only documented.

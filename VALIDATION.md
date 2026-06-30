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
misconfiguration) before any longer run. They share a common pattern: pass
`--total-time-seconds` (always required), keep the task and simulation counts
tiny, and pass an explicit `--seed` so the run is repeatable.

On HPC, run smoke and check submissions on the short-lived `test` (CPU) and
`gpu_test` (GPU) partitions, leaving the production `general1` (CPU generation)
and `gpu` (train/eval) partitions for full runs. Take the partition,
node-geometry, and resource layout from each stage script's `#SBATCH` block and
header examples and replicate them verbatim — change only what the check
requires (typically the duration and the task counts). Do not recompute node
counts, core-per-node geometry, or GPU counts; the scripts already pin them.

### 2.1 RDS (reaction-diffusion simulation)

```bash
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py \
    --total-time-seconds 2.0 --tasks 1 --task-simulations 5 --seed 42 --verbose
```

**Expected**: one `.h5` trajectory per simulation under the RDS trajectory
output directory (`READY_TRACT/`):
`<data_bank>/Video/READY_TRACT/SRM_AND_SBI_DIMER_ALP_2S_50FPS_TASK_0/`
(`..._TASK_0_SIM_0.h5`, …), one `.zarr` theta set at
`<data_bank>/Theta/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Theta_Set_TASK_0.zarr`, and
verbose output showing the sampled diffusion and reaction rates.

### 2.2 DLI (diffraction-limited imaging)

```bash
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py \
    --total-time-seconds 2.0 --tasks 1 --task-simulations 5 --seed 42 --verbose
```

**Expected**: one `.zarr` video set at
`<data_bank>/Video/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Video_Set_TASK_0.zarr`. At 2 s
the video shape is `(100, 256, 256)` — 100 frames at 50 frames per second over
a 256×256 detector grid. Each value is a non-negative integer pixel count.

**Requires**: the RDS smoke test must have run first with the same
`--total-time-seconds` and `--task-simulations`. DLI reads the `.h5`
trajectories and the theta set written by RDS; the two stages share `--tasks`
and `--task-simulations`.

### 2.3 Inference (posterior training)

```bash
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Inference.py \
    --total-time-seconds 2.0 --tasks 1 --epochs 1 --seed 0 --verbose
```

**Expected**: a network checkpoint at
`<data_bank>/Labor/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Optimum_ANN.pth` and a pickled
posterior at `<data_bank>/Posit/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Posterior.pkl`.
One epoch on a handful of (video, theta) pairs will not produce a useful
posterior, but it exercises the full training and save path, including the
network construction and the data loader.

**Requires**: the RDS and DLI smoke tests (sections 2.1 and 2.2) must have run
first with `--task-simulations 5` (or higher). With too few simulations the dataset is too
small to form even a single training batch and the run fails early; five
simulations is enough for the smoke test.

**Small-GPU batch-size workaround**: the `Complex3DCNN` forward pass at the
default `--batch-size 32` over 2 s videos needs several GB of GPU memory (one
3D convolution alone needs roughly 6 GB). A small GPU (for example a 4 GB card)
will raise `CUDA out of memory` partway into the first epoch. Two options:

- **Reduce the batch size** to `1` or `2` for a memory-light smoke check on a
  small GPU. This verifies the pipeline without producing a useful posterior:

  ```bash
  python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Inference.py \
      --total-time-seconds 2.0 --tasks 1 --epochs 1 --batch-size 1 --seed 0 --verbose
  ```

- **Train on a larger GPU**, or on CPU when no adequate GPU is available
  (select the CPU backend in `env_snapshots/README.md` and the CPU compute
  backend in the machine profile).

### 2.4 Resurrect (continue from an existing checkpoint)

```bash
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Inference.py \
    --total-time-seconds 2.0 --tasks 1 --epochs 1 --seed 1 --resurrect --verbose
```

**Expected**: the run loads the checkpoint saved by the inference smoke test,
prints a line reporting the loaded checkpoint and its baseline loss, then runs
one more epoch starting from those weights. The checkpoint is overwritten only
if the continued run improves on the baseline. This mode is useful for
incremental training across separate invocations.

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
   z = zarr.open('<data_bank>/Theta/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Theta_Set_TASK_0.zarr', mode='r')
   print(z[:].tolist())
   "
   ```

2. **Reaction-diffusion primitive equivalence.** ReaDDy is deterministic given
   the same system specification (species, reactions, rates, simulation box,
   particle complement). The system builder and the simulation builder produce
   a ReaDDy system with the declared rates, geometry, and observables expected
   for a given theta. The verbose RDS banner (`--verbose`) prints the diffusion
   and reaction rates, so the constructed system can be inspected directly. The
   only intentional source of run-to-run variability is the reaction-diffusion
   stepper's own internal randomness (see the theta-only regression test,
   section 3.2).

3. **Imaging-pipeline functional equivalence.** The DLI stage applies a Gaussian
   point-spread function (erf-based pixel integration), an EMCCD detector model
   (Poisson photon noise plus Gaussian readout noise), a brightness state
   machine, and the duration-independent photobleaching model. The verbose DLI
   banner (`--verbose`) prints the detector parameters, and a rendered video
   (via `--show`) shows sparse fluorescent spots on a near-zero background with
   plausible pixel-value ranges. The pipeline produces videos of the correct
   shape, dtype, and value distribution.

**Acceptance**: pillars (1)–(3) hold — theta is bit-reproducible at a fixed
seed, the constructed reaction-diffusion system matches the declared model, and
the imaging output is correctly shaped and physically plausible.

### 3.2 Reproducibility — theta-only regression test

The `--seed` flag defaults to `None`, which means **non-deterministic by
design**: each run draws fresh entropy for prior sampling, particle placement,
imaging noise, and network initialization. This matches the inherent
stochasticity of experimental fluorescence data and is the intended production
behavior. To exercise the reproducibility guarantee, pass `--seed` explicitly.

The acceptance bar is **theta-only** reproducibility — the prior-sampling stage
is fully seeded, the downstream stages are not. Run the pipeline three times at
the same explicit seed:

```bash
DB=<data_bank>            # the data_bank_root from your machine profile
for i in 1 2 3; do
    rm -rf "$DB/Theta" "$DB/Video"
    python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py \
        --total-time-seconds 2.0 --tasks 1 --task-simulations 5 --seed 42
    md5sum "$DB/Theta/SRM_AND_SBI_DIMER_ALP_2S_50FPS_Theta_Set_TASK_0.zarr"/0.0
done
```

**Acceptance**:

- **Theta sets are bit-identical** across all three runs (the three md5 hashes
  match). This proves the prior-sampling RNG is seeded and correctly propagated
  through the entry point; if a future code change ever drops that seed
  handling, this check catches the regression immediately.
- **Trajectories (`.h5`), videos (`.zarr`), and inference outputs
  (`.pth`/`.pkl`) differ across runs**, and that is expected, not a regression.
  The reaction-diffusion stepper draws from an internal random source that is
  not seedable, so trajectories vary; the videos inherit that variability
  through their trajectory input (even though the imaging pipeline's own noise
  RNGs are fully seeded); and the trained network inherits it through its
  training data. Posterior summaries should converge across runs within
  statistical uncertainty given enough training data, but bit-identity past the
  theta stage is neither expected nor required.

In short: **theta is reproducible at a fixed seed; trajectories, videos, and
trained networks vary by design.**

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

Repeat the smoke tests with the long duration:

```bash
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py \
    --total-time-seconds 10.0 --tasks 1 --task-simulations 5 --seed 42 --verbose
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py \
    --total-time-seconds 10.0 --tasks 1 --task-simulations 5 --seed 42 --verbose
python Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Inference.py \
    --total-time-seconds 10.0 --tasks 1 --epochs 1 --seed 0 --verbose
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
      --total-time-seconds 2.0 --summary both
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
`bounded` pool (rejection sampling within the prior) is correct. Run any stage
with `--help` for the full flag list.

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
- All four entry points (RDS, DLI, Inference, Resurrect) running end-to-end on
  minimal inputs.
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

## Project Context — srm-and-sbi-dimer-alp

This document is self-contained. It describes the scientific context for the
DIMER model and its reaction-diffusion-parameter inference: the research
question, the molecular system, the two-stage inference architecture, the data
and computational flow, the inference network, the validation methodology, and
the design rationale that shapes the implementation.

---

## §1. Research Question

**Objective:** Infer posterior distributions `p(θ | x)` of reaction-diffusion
(RDS) parameters from time-resolved fluorescence-microscopy videos of
membrane-receptor dimerization.

**Why posterior distributions, not point estimates?**
Molecular-dynamics parameters are often partially confounded (for example,
diffusion coefficient and dwell time co-determine observed intensities).
Uncertainty quantification is part of the scientific result. Downstream
applications (drug-sensitivity analysis, population heterogeneity) require the
full posterior, not a single best-fit value. The inference target is therefore
a full conditional density `p(θ | x)` over the parameter vector, and the MAP
point estimate and the posterior credible interval are always reported together
because they answer different questions — a sharp posterior at the wrong
location and a broad posterior at the right one are distinguishable only when
both are shown.

**Why simulation-based inference (SBI)?**
The forward model is complex: a ReaDDy reaction-diffusion simulation feeds an
optical-imaging step (diffraction-limited, photon noise) to produce a video. The
likelihood is intractable. SBI sidesteps explicit likelihood evaluation by
learning a neural density estimator (neural posterior estimation with a masked
autoregressive flow) from simulated (RDS, video) pairs.

---

## §2. System: EGFR-Like Dimerization (DIMER Model)

**Molecular system:** A simplified model of epidermal growth factor receptor
(EGFR) dimerization and internalization on the plasma membrane.

**Three molecular species:**
- **A (monomer):** Single receptor. Diffuses freely (diffusion coefficient `D_A`).
- **B (mobile dimer):** Two receptors bound, still mobile. Diffuses at rate
  `D_B` (typically slower than `D_A`).
- **C (immobile dimer):** Two receptors bound and internalized or otherwise
  immobilized. Does not diffuse (`D_C = 0`).

**Reactions:**
- **`A + A ↔ B`:** Two monomers associate at rate `k_on`; the dimer dissociates
  at rate `k_off`.
- **`B ↔ C`:** Mobile dimer transitions to the immobile state at rate `k_imm`;
  no reverse (irreversible internalization).
- **Future extension:** **`C → ∅`:** degradation/recycling of the immobile
  dimer, a biological extension reserved for future work.

**Parameters to infer (θ):**
- `D_A`: Monomer diffusion coefficient (μm²/s)
- `D_B`: Mobile dimer diffusion coefficient (μm²/s)
- `k_on`: Association rate (μm³/s or equivalent)
- `k_off`: Dissociation rate (1/s)
- `k_imm`: Immobilization rate (1/s)
- `n_A_0`: Initial monomer count
- `n_B_0`: Initial dimer count (typically 0)
- `n_C_0`: Initial immobile dimer count (typically 0)

**Not inferred (detector parameters, fixed here):**
- Point-spread-function (PSF) width: Gaussian, σ ≈ 150 nm
- Photon-detection efficiency: ε ≈ 0.6
- EMCCD gain and readout noise: calibrated from the camera
- Video frame rate: 50 Hz (20 ms per frame) — a fixed sampling cadence
- Recording length: supplied per run via `total_time_seconds` (commonly 2 s,
  5 s, or 10 s)

---

## §3. Inference Pipeline: Two-Stage Architecture

The full program separates detector inference from molecular-parameter
inference. The detector is characterized first, on a simpler system, and then
held fixed while the molecular parameters are inferred. This disentangles
optical and sensor effects from the biological reaction-diffusion parameters,
reduces the dimensionality of each inference problem, improves posterior
geometry, and speeds convergence.

### Stage 1: Detector Parameters (separate sibling, planned future work)

**Input:** Synthetic videos from a simple-diffusion model (no reactions; pure
Brownian motion).

**Objective:** Infer detector parameters (`β`) — PSF width, photon efficiency,
noise levels — from video observations.

**Output:** A posterior over `β`. The posterior mean initializes the detector
model for Stage 2.

### Stage 2: RDS Parameters (this repository)

**Input:** Synthetic videos generated in two steps:
1. **RDS simulation:** ReaDDy solves the DIMER reactions and diffusion for the
   configured recording length.
2. **DLI imaging:** A Gaussian PSF, Poisson photon noise, and EMCCD readout
   noise are applied. The detector parameters (`β`) are **fixed**.

**Objective:** Infer the RDS parameters (`θ`) conditional on the fixed detector:
`p(θ | video, β_fixed)`.

**Output:** A pickled posterior and a trained neural-network checkpoint.

---

## §4. Data and Computational Flow

### RDS Simulation (this repository, step 1)

**Script:** `Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py`

**Process:**
1. Sample RDS parameters `θ` from a log-uniform prior over biologically
   plausible ranges.
2. Initialize a ReaDDy system: particles (A, B, C), diffusion coefficients,
   reaction rates, simulation box, and observables.
3. Evolve the system for the recording length (`total_time_seconds`).
4. Record particle trajectories (positions, species, time) to an `.h5` file
   (HDF5, the ReaDDy convention), and the sampled theta set to a compressed
   `.zarr` array.

**Example quantities:** Particle count per species over time, mean
inter-particle distances, reaction-event counts.

**Per-simulation kernel release.** The generation loop builds a fresh ReaDDy
system and simulation for each draw. The CPU compute kernel allocates a
worker-thread pool and observable/output handles per simulation; without an
explicit release these accumulate across a long run, growing the task's thread
count and resident memory until a memory-tight node thrashes and the task hangs.
The loop therefore releases the simulation and system objects and forces a
garbage collection at the end of each iteration, so threads and resident memory
stay flat across arbitrarily many simulations while every simulation's output is
unchanged. An opt-in `--probe` flag logs per-simulation thread, file-descriptor,
and memory counts for diagnosing resource behavior, and is off by default.

### DLI Imaging (this repository, step 2)

**Script:** `Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py`

**Process:**
1. Read the RDS trajectory from `.h5` and extract per-frame poses (and a dimer
   mask).
2. Render particles as Gaussian-PSF-convolved objects on a 2D detector grid,
   accumulating each emitter's PSF integral across pixel boundaries.
3. Apply Poisson photon noise (mean = intensity × photon-conversion factor).
4. Apply EMCCD readout noise.
5. Discretize to an integer pixel count (8- or 16-bit) and save a compressed
   `.zarr` video set.

**Dimer brightness model:** Each emitter flagged as a dimer (species B or C) has
its per-frame brightness multiplied by `dimer_mule = √2 ≈ 1.414` — *not* the
naive 2× of two co-located fluorophores. Two consistent physical interpretations:

- **Shot-noise-equivalent scaling.** The signal-to-noise ratio for the sum of
  two independent Poisson sources scales as `√(2N) / √N = √2` versus a single
  source. If "brightness" here represents the noise-equivalent detection metric
  (rather than raw photon count), the `√2` factor matches.
- **Geometric mean of bright/dark states.** Fluorophore blinking puts each
  emitter stochastically into a dark or bright state. The time-averaged
  effective brightness of a dimer is `GM(1×, 2×) = √2`, consistent with the two
  fluorophores spending roughly equal time in the {one active, both active}
  configurations.

**Photobleaching model:** Each emitter can irreversibly transition to a dark
(bleached) state, applied per frame through a two-state transition matrix. The
per-frame bleach probability is
`prob_1 = 1 − (1 − prob_photo_bleach)^(1 / numb_photo_bleach)`, where
`numb_photo_bleach = 100` is a fixed reference-frame count (a calibration
convention, *not* the clip or video frame count) and `prob_photo_bleach = 0.1`
is the cumulative bleach fraction over that 100-frame (2 s at 50 Hz) reference
window.

The bleach rate is parameterized by this pair — a cumulative bleach probability
together with the reference-frame count over which it accrues — rather than by a
single hardcoded 100-frame probability, which makes the model correct for any
recording length. Because `numb_photo_bleach` is pinned to 100 regardless of
clip duration, the per-frame rate `prob_1` is constant across video lengths; the
cumulative bleach over an `n`-frame clip is
`p_video = 1 − (1 − prob_photo_bleach)^(n / numb_photo_bleach)`, so a longer clip
accumulates more bleaching through repeated application of the same per-frame
transition matrix (about 10% over 100 frames / 2 s, about 41% over 500 frames /
10 s).

The reference count is pinned to 100 by design rather than tied to the clip
length. The value `prob_photo_bleach = 0.1` over the 100-frame reference was
calibrated by detector-parameter SBI inference on the experimental raw videos,
with the reaction-diffusion parameters held out; `numb_photo_bleach = 100` is the
convention that inference was run under, so it is preserved to keep the
calibrated value meaningful. Making `numb_photo_bleach` track the clip length
would render the per-frame rate duration-dependent — unphysical, since bleaching
is a property of the fluorophore and illumination, not of how long a recording
happens to be — and would contradict the detector-inferred value. (The aggregate
`p_bleach` reported by swift / SPTAnalyser is a track-disappearance rate that
lumps together bleaching, diffusion out of the field, blinking and gaps, and
unbinding; it describes a different process and is not substituted for
`prob_photo_bleach`.)

**Output:** A `.zarr` video set (chunked array, efficient I/O). Shape
`(frame_count, height, width)` — for example `(100, 256, 256)` at 2 s and 50 Hz.

### Fixed-cadence, per-run timing model

The frame count is always derived from the recording length and the fixed frame
rate: **`frame_count = total_time_seconds / frame_time_seconds`** (recording
length times frame rate). The frame rate is fixed configuration; the recording
length is per-run. These two roles are kept structurally separate so the frame
count cannot be read from a stale global default:

- A global `FrameConfig` holds only the fixed sampling cadence
  (`frame_time_seconds`, `steps_per_frame`, and the derived frames-per-second
  and time step). It carries no per-run duration at all.
- A per-run `RunTiming` is constructed at each entry point from a **required**
  `--total-time-seconds` argument. It derives the per-run `frame_count`, total
  step count, and timing label, and passes through the fixed-cadence values.

Because `frame_count`, `total_time_seconds`, and the timing label exist only on
the per-run object and never on the global, no code can silently inherit a
default duration. Fail-loud guards back this up: trajectory extraction iterates
the trajectory's own recorded frame count and raises on an entirely empty frame,
and the imaging stage raises if the extracted frame count differs from the run's
declared `frame_count`. This is why `--total-time-seconds` is required on every
stage and why every duration produces full-length, fully-populated videos.

### Inference (this repository, step 3)

**Script:** `Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Inference.py` (with an
optional `--resurrect` flag to continue from an existing checkpoint)

**Process:**
1. **Training data:** Use the (θ, video) pairs produced by RDS and DLI, where
   `θ ~ prior` and the video is the imaging of the trajectory that `θ` produced.
2. **Neural density estimation:** Train a neural posterior estimator (a masked
   autoregressive flow over a learned video embedding) to approximate
   `p(θ | x)` from the simulated pairs. The embedding is a 3D convolutional
   video encoder followed by a temporal transformer (§6).
3. **Resurrect mode (optional, runtime flag):** With `--resurrect`, the script
   loads the existing optimum checkpoint, records its replay loss as a baseline,
   then continues the standard training loop from those weights. A new optimum
   overwrites the same checkpoint. This supports incremental training across
   separate invocations.
4. **Output:** Posterior samples are obtained by passing a real experimental
   video (or a synthetic holdout) through the trained network.

**Output:**
- A pickled posterior (`DirectPosterior`).
- A trained network checkpoint (encoder, transformer, and posterior-parameterizer
  weights), saved whenever a new optimum is reached.

**Training-loop efficiency.** The data loaders keep their worker pool alive
across epochs (`persistent_workers`) so that a long training run does not pay the
cost of re-spawning and re-importing the full stack in every worker each epoch —
which otherwise dominates the per-epoch wall time. Because each data-parallel rank
builds its own loaders and the train and validation loaders' workers stay alive
together, the live worker-process count is (workers per loader) × (ranks) ×
(concurrent loaders). The per-loader count is therefore derived from a node-wide
TOTAL budget — the machine profile's `num_workers`, or the CPU core count when unset
— divided across the ranks and concurrent loaders, so the live total stays near one
worker per core at any GPU count (and reduces to half the cores per loader on a
single GPU). This keeps a multi-GPU run from exhausting host memory through worker
multiplication. This budget is the data-loading workers only; the GPU/shard-worker
count is bounded separately (see Multi-GPU scaling).

**Multi-GPU scaling.** Training, MAP-recovery evaluation, and the real-data
application adapt to the GPUs they are given. Launched with one worker per GPU
(via `torchrun`), training runs data-parallel through `DistributedDataParallel` —
each worker holds a replica, processes its own shard of every batch, and
synchronizes gradients each step, with `SyncBatchNorm` sharing batch statistics
across workers by default — while MAP-recovery evaluation partitions the held-out
videos across the workers, and the experiment stage partitions its
`(condition, cell)` work the same way, each merging the per-shard results into
one report. The single-GPU run is the collapse case of
the same code: with one worker the distributed wrappers reduce to no-ops and the
loop is exactly the original single-GPU path, so behavior is unchanged where
only one GPU is present. The GPU count is read from the allocation, capped by an
optional `SRM_AND_SBI_GPUS` override (set it to 1 to force the single-GPU path),
and `SRM_AND_SBI_NO_SYNC_BN=1` opts each worker into its own local batch
statistics for speed at the cost of re-validating recovery.

### Leak-proof data split (TRAIN / TEST / EVAL)

Training data, model-selection data, and final-validation data are physically
separated into three on-disk namespaces, distinguished by an explicit suffix at
the end of every output name (`_TRAIN`, `_TEST`, `_EVAL`):

- **TRAIN** supplies the gradient updates.
- **TEST** supplies the per-epoch model-selection signal (the best-on-TEST
  checkpoint is kept; the network never trains on TEST). With no TEST set, the
  last-epoch checkpoint is kept instead.
- **EVAL** is held out entirely — never touched by gradients or model selection,
  only by the final MAP-recovery report.

Each split is an **independent draw with its own seed**, not a shuffled reuse of
a single pool. Because the splits never share samples, validation leakage is
impossible by construction: a reported recovery can never reflect data the
posterior already optimized against.

A single orchestrator script generates a complete, correctly proportioned set,
running the full RDS → DLI flow for all three splits with independent seeds and
enforcing the sizing rule:

- CORE = TRAIN + TEST, with a minimum of 10 samples,
- TRAIN = 0.8 · CORE,
- TEST = 0.2 · CORE,
- EVAL = max(10, 0.1 · CORE).

A `--dry-run` flag previews the sizing plan without generating anything.

### Non-deterministic generation and task-index provenance

Generation passes no random seed by default: prior sampling, particle placement,
PSF and brightness draws, and camera noise all draw fresh entropy, so every
simulation is an independent random draw. This matches the reality that the
reaction-diffusion stepper itself is not seedable — identical-output
reproducibility past the prior-sampling stage is unreachable regardless, so a
deterministic per-simulation seeding scheme would add bookkeeping without a
payoff and could introduce subtle cross-split correlations. No scientific
information is lost, because the sampled theta is persisted to disk per task, so
every (theta, video) pair is recorded.

Dataset integrity instead rests on a **global task index encoded in the file
names** (`..._TASK_<tid>_<split>`). When generation is fanned out across many
parallel tasks, each task is assigned a distinct global index, so parallel
packing, array overflow, and incremental appends (via a task-offset base) can
never collide on a filename or mix split namespaces. The index scheme is
injective by construction, and an analysis script asserts file-label uniqueness
across an entire fan-out before training. A `--seed` flag remains available on
every stage for an optional deterministic run, but the default — and the
batch-generation scripts — pass nothing.

### Two-tier data-bank storage, routed by data role

The permanence of a file follows its scientific role rather than the machine. A
machine whose fast scratch filesystem is large but impermanent (auto-purged, not
backed up) can route the *regenerable* bulk — TRAIN and TEST — onto scratch while
keeping everything that must persist — EVAL, posteriors, training checkpoints,
and experimental data — on permanent, backed-up storage. The machine profile
gains an optional scratch root; a single resolver returns the scratch root for
TRAIN/TEST when it is configured and the permanent root otherwise. The on-disk
layout under each root is identical (a sparse mirror), and the split suffix
already in every filename identifies which tier a file belongs to. A machine that
configures only the permanent root is single-tier and behaves identically, so the
feature is invisible where it is not used. Experimental microscopy data is
external and irreplaceable, so it always lives on permanent storage and never on
scratch.

### Support functions (Python package: `srm_and_sbi_dimer_alp/`)

The support code is organized into a flat Python package of focused modules
rather than a single monolithic support file. The modules and their roles:

- **`parameterization.py`** — the single source of truth for configuration. It
  loads the per-machine profile, holds the sibling-wide defaults as frozen
  dataclasses (path conventions, simulation geometry and timing, RDS/DLI
  defaults, inference training and network architecture, plotting), and carries
  the rich parameter specification (parameter ranges, log flags, units, and
  labels). It exposes a typed `PARAMETERS` singleton and prior/bounds helpers,
  and validates the configuration at import time.
- **`simulation_rds_support.py`** — the ReaDDy primitives: system builder,
  simulation builder, and trajectory-pose extraction (the extractor is reused by
  the imaging stage, which also requests a dimer mask).
- **`simulation_dli_support.py`** — the imaging pipeline: Gaussian PSF, EMCCD
  detector, intensity accumulation, the brightness state machine, the transition
  matrices (including photobleaching), and the top-level imaging orchestrator.
- **`inference_network.py`** — the network architecture (§6): positional
  encoding, attention block, temporal transformer, and the 3D-CNN video encoder.
- **`inference_support.py`** — the training pipeline: the video dataset (with
  augmentation), normalization, training/validation set-up, the training loop
  (with the resurrect branch), and posterior save/load.
- **`evaluation.py`** — the MAP-recovery core shared by the validation stages:
  the seed-then-optimize estimator, posterior-quantile summaries, and report
  tables.
- **`io.py`** — file I/O: transparent loading of `.zarr`/`.npy`/`.npz`, video
  and theta-set writing, and bit-depth conversion. All paths come from the
  configuration helpers, never hardcoded.
- **`visualization_rds.py`, `visualization_dli.py`, `visualization_inference.py`**
  — stage-specific diagnostic and figure builders (matplotlib imported lazily so
  headless runs do not pay its import cost).
- **`diagnostics.py`** — the shared diagnostics engine behind the `--debug` /
  `--debug-dump` flags: per-step checkpoints, fail-loud invariant checks,
  quantitative stats, and a self-contained dumped report.
- **`utils.py`** — small cross-cutting helpers (terminal separators, memory-state
  logging, and the resource-probe helpers).

Each entry-point script is a thin wrapper: it parses CLI arguments, loads the
configuration, calls package functions, and writes outputs to the
configuration-defined paths. The package is installed editable, so script edits
to the package take effect without reinstallation.

### Configuration architecture

Configuration is split into two complementary parts:

- **Per-machine** — `machine_profiles.toml` carries the absolute paths and
  compute resources for each machine (script-bank root, data-bank root, optional
  scratch root, compute backend and device, worker count, running mode). It is
  kept out of version control; a committed `machine_profiles.example.toml`
  documents the schema. The active profile is selected by the `MACHINE_PROFILE`
  environment variable, and the configuration refuses to load — with a clear,
  remediating error — if the variable is unset, the profile is missing, a
  required key is absent, or a root directory does not exist. There is no silent
  fallback.
- **Project-wide** — the committed defaults and the parameter specification live
  in `parameterization.py` as frozen dataclasses. Freezing them makes the
  scientific defaults part of the reproducibility contract: they cannot be
  mutated at runtime. A user override (such as `--total-time-seconds 10.0`) is
  applied at the call site, not by mutating the global.

This separation decouples the code from machine-specific paths: a single
codebase runs on a local prototyping machine and on an HPC system with one
environment-variable switch, and absolute filesystem paths stay out of the
committed record.

---

## §5. Implementation Map (Science → Code)

Each scientific concept and pipeline stage maps to a specific module and
function in the package, driven by a thin entry-point script, and produces a
defined on-disk artifact. Module paths are relative to the package
`srm_and_sbi_dimer_alp/`; entry-point scripts live under `Script_Bank/Prime/`.

| Scientific concept / stage | Code (module → function/class) | On-disk artifact |
| --- | --- | --- |
| DIMER reaction system (`A + A ↔ B`, `B → C`): species, diffusion coefficients, reaction rates, simulation box | `simulation_rds_support.py` → `build_system()`; the ReaDDy simulation is then assembled by `build_simulation()` | (in-memory ReaDDy system/simulation; trajectory written below) |
| RDS trajectory recording (particle positions, species, time over the recording length) | entry point `SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py` (drives `build_system()` → `build_simulation()`) | trajectory `.h5` (HDF5, ReaDDy convention); sampled theta set `.zarr` (via `io.py` → `save_theta_set()`) |
| Trajectory extraction (per-frame poses and the dimer mask) | `simulation_rds_support.py` → `extract_trajectory_poses()` (reused by the imaging stage, which requests the dimer mask) | (per-frame pose arrays passed to imaging) |
| Diffraction-limited imaging forward model: Gaussian PSF, Poisson + EMCCD readout noise, dimer-brightness scaling, photobleaching | `simulation_dli_support.py` → `simulate_dli()` (orchestrator), with `Gaussian` / `sample_psf_width()` (PSF), `compute_intensity()` + `add_pixel_counts()` (intensity accumulation), `compute_brightness()` / `compute_brightness_probability()` and `generate_state_trajectories()` (brightness state machine), `compute_matrices()` (transition matrices incl. photobleaching), `EMCCD` / `add_noise()` / `generate_frames()` (detector noise) | (noised video array passed to writer below) |
| DLI video output (chunked, bit-depth-converted) | entry point `SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py` (drives `extract_trajectory_poses()` → `simulate_dli()`) | `.zarr` video set, shape `(frame_count, height, width)` (via `io.py` → `convert_video_dtype()`, `save_video_set()`) |
| Parameter prior and specification (ranges, log flags, units, labels; log-uniform prior and bounds) | `parameterization.py` → `PARAMETERS` (a `Parameters` singleton) with `build_prior()`, `theta_lower_bound()`, `theta_upper_bound()`, `parameter_find()` | (configuration in code; sampled theta persisted in the RDS theta-set `.zarr`) |
| NPE + MAF estimator with 3D-CNN + temporal-transformer embedding | `inference_network.py` → `Complex3DCNN` (video encoder), `TemporalTransformer` (with `AttentionBlock`, `PositionalEncoding`); training wired in `inference_support.py` → `setup_training()`, `train_loop()` (with the resurrect branch) | (in-memory network; checkpoint + posterior written below) |
| Leak-proof TRAIN / TEST / EVAL split, sizing rule, and dataset construction | entry point `SRM_AND_SBI_DIMER_ALP_Generate_Datasets.py` (runs RDS → DLI per split with the `CORE = TRAIN + TEST`, `EVAL = max(floor, 0.1·CORE)` sizing); dataset assembly in `inference_support.py` → `build_datasets()` (with `VideoDataset`, `normalize_video()`) | `_TRAIN` / `_TEST` / `_EVAL`-suffixed trajectory `.h5` and video `.zarr` namespaces |
| Posterior training run (gradient updates on TRAIN, selection on TEST) | entry point `SRM_AND_SBI_DIMER_ALP_Inference.py` (drives `build_datasets()` → `setup_training()` → `train_loop()` → `save_posterior()`) | pickled posterior (`DirectPosterior`, via `inference_support.py` → `save_posterior()`); network checkpoint at each new optimum |
| MAP recovery and calibration on held-out EVAL | `evaluation.py` → `map_estimate()` (seed-then-optimize: `collect_theta_prex()`, `collect_score_prex()`, `extract_elite_prex()`, `optimize_elite()`), `posterior_summary()`, `recovery_stats()`, `recovery_table()`, `posterior_coverage_table()`; driven by entry point `SRM_AND_SBI_DIMER_ALP_Evaluation.py` | recovery report (figures + tables + arrays + a live `progress.log`) under the validation output directory |
| Real-data application (no ground truth) | same `evaluation.py` estimator (`map_estimate()`, `experiment_table()`); driven by entry point `SRM_AND_SBI_DIMER_ALP_Experiment.py` | per-condition inferred-parameter report under the validation output directory |
| Configuration, paths, storage routing, and file I/O | `parameterization.py` → `Paths`, `MachineProfile` / `load_machine_profile()`, `FrameConfig`, `RunTiming`; `io.py` → `load_data()`, `save_video_set()`, `save_theta_set()`, `convert_video_dtype()` | resolved absolute paths (per-machine `machine_profiles.toml`); all artifacts above land under the configured roots |

The five pipeline stages are RDS, DLI, Inference, Evaluation, and Experiment;
their entry-point scripts (`SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py`,
`SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py`, `SRM_AND_SBI_DIMER_ALP_Inference.py`,
`SRM_AND_SBI_DIMER_ALP_Evaluation.py`, `SRM_AND_SBI_DIMER_ALP_Experiment.py`)
are each thin wrappers that parse arguments, load the configuration, call the
package functions above, and write outputs to the configuration-defined paths.
`SRM_AND_SBI_DIMER_ALP_Generate_Datasets.py` orchestrates the generation pair
(RDS → DLI) across all three splits in one command.

---

## §6. Inference Network Architecture

The network learns an embedding of the video, then parameterizes a flexible
posterior over θ from that embedding.

**Video encoder (a 3D convolutional network):**
- Input: a `(batch, 1, n_frames, height, width)` video tensor.
- 3D convolutions over space and time extract hierarchical features (edges,
  textures, motion patterns).
- Output: a flattened feature map. The encoder's temporal depth is set by
  `n_frames`, derived from the recording length, so the same architecture serves
  any duration (100 frames at 2 s, 500 at 10 s); it asserts that the input frame
  count matches `n_frames` so a duration/data mismatch fails loudly.

**Sequence model (a temporal transformer):**
- Input: temporal features from the encoder.
- Self-attention learns temporal dependencies (for example, particles moving
  consistently across early frames correlating with a high diffusion coefficient).
- A learnable CLS token is prepended; its output embedding is the summary used
  for posterior parameterization.
- Sinusoidal positional encoding accommodates the full frame range.

**Posterior parameterizer:**
- Input: the summary embedding from the transformer.
- Dense layers map the embedding to the parameters of a masked autoregressive
  flow, yielding an expressive, multimodal posterior rather than a single point
  estimate.

**Sampling:** Draw samples from the posterior parameterized by the embedding.

**Training:** Minimize the flow's negative log-likelihood on the simulated
(video, θ) pairs, with rotation and flip augmentation on the videos.

---

## §7. Validation and Diagnostics

### Semantic equivalence

The simulation and imaging stages thread explicit, per-function random-number
generators, so correctness is established **semantically** — confirming the code
produces the right scientific behavior — rather than by matching any particular
reference run element-by-element. Three pillars carry this:

1. **Theta-sampling determinism.** The prior sampler draws from the same fixed
   bounds with a seeded generator, so the same seed produces bit-identical theta
   vectors — directly verifiable.
2. **Reaction-diffusion primitive equivalence.** ReaDDy is deterministic given
   the system specification (species, reactions, rates, box), and the system and
   simulation builders construct exactly the declared model for a given theta.
3. **Imaging-pipeline functional equivalence.** The Gaussian PSF (erf-based
   pixel integration), the EMCCD model (Poisson plus Gaussian readout), the
   brightness state machine, and the duration-independent photobleaching model
   produce videos of the correct shape, dtype, and value distribution.

### Reproducibility characteristics

Prior sampling is fully seeded, so theta sets are bit-reproducible at a fixed
seed. The reaction-diffusion stepper is not seedable, so trajectories — and the
videos and trained networks that depend on them — vary run-to-run by design,
matching the inherent stochasticity of experimental fluorescence data. The
default is non-deterministic (no seed); an explicit seed is available for
reproducible debugging and for the theta-only reproducibility regression test.

### Leak-proof split and MAP-recovery validation

A trained posterior is validated on data it has never seen, using the
three-namespace split (TRAIN / TEST / EVAL) described in §4. Because each split
is an independent draw with its own seed, the validation cannot be contaminated
by data the network optimized against.

**Simulated recovery (`Evaluation.py`).** For each held-out EVAL video the
maximum-a-posteriori parameter vector is estimated (seed-then-optimize: draw a
candidate pool, score it by the flow's log-probability, keep the top-K elite
seeds, then gradient-ascend the log-probability with Adam, a plateau
learning-rate schedule, and early stopping) and compared to the known ground
truth, giving two complementary read-outs:
- *Recovery accuracy* — how close the inferred parameter is to truth
  (per-parameter error; fraction within a tolerance band).
- *Posterior calibration* — whether the per-video credible intervals contain the
  truth at their nominal rate (roughly 50% for the interquartile range, roughly
  90% for the 5–95% interval). Under-coverage signals an overconfident
  posterior; over-coverage, an underconfident one.

**Real-data application (`Experiment.py`).** The same estimator is applied to
experimental microscopy videos, for which there is no ground truth. Each
recording is split into model-length windows and the inferred-parameter
distribution is reported per experimental condition, so treatment groups can be
compared. This is the scientific end use of the trained posterior, not a
correctness check. A two-mode candidate pool supports both regimes: a `bounded`
pool (rejection sampling within the prior; correct for a well-trained posterior)
and an `unrestricted` pool (sampling the flow directly, for smoke tests and
undertrained posteriors whose mass can lie outside the prior box).

Both stages report the **MAP point estimate** (the posterior mode) and the
**posterior credible summary** (median plus interquartile range) side by side,
because the two answer different questions: a sharp posterior at the wrong
location and a broad posterior at the right one are distinguishable only when
both are shown.

### Observability and diagnostics

Every stage shares one diagnostics scheme. `--debug` prints per-step
checkpoints, fail-loud invariant checks, and an end-of-stage summary; a dump mode
additionally persists a self-contained report (figures plus a console
transcript) under a diagnostics workbench directory, kept separate from the
scientific deliverables. The validation stages also write a live, tail-able
progress log so a long run can be monitored. The diagnostics are off by default,
and each check is a cheap no-op when off.

### Posterior geometry

Useful diagnostics to compute when analyzing a trained posterior:
- Posterior mean and covariance.
- Marginal distributions (1D histograms per parameter).
- Bivariate correlations (2D scatter plots, especially for confounded
  parameters).
- Posterior-predictive check: sample `θ ~ posterior`, generate videos, and
  compare to real data.

---

## §8. Open Scientific Questions

*(Scientific and methodological questions only.)*

**S1. Identifiability of RDS parameters.** Are all parameters in θ uniquely
identifiable from video data? Which parameters are confounded (co-determined)?
This calls for theoretical or empirical analysis.

**S2. Prior sensitivity.** How sensitive is the posterior to the prior choice
(uniform versus log-normal)? Should informative priors from the biophysical
literature be used?

**S3. Posterior convergence across repeated resurrect invocations.** When
inference is re-run with `--resurrect` repeatedly, each run continuing from the
previous optimum, how many invocations are needed for posterior stability? Is
there a principled stopping criterion beyond an empirical plateau (for example, a
test-loss delta below tolerance)?

**S4. Out-of-distribution robustness.** If the network is trained at one
recording length and tested at a different one (for example, trained on 2 s
videos and tested on 10 s), how does posterior accuracy degrade? Can it be
trained jointly on multiple durations?

**S5. Multi-cell heterogeneity.** Real microscopy data contains cells with
varying expression levels, spatial organization, and cell-cycle phase. Can the
single-cell, per-video model extend to population posteriors?

---

**End of Project Context — srm-and-sbi-dimer-alp**

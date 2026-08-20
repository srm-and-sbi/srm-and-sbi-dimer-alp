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

## §2. System: MET-Receptor Dimerization (DIMER Model)

**Molecular system:** A simplified, receptor-agnostic model of receptor
dimerization and internalization on the plasma membrane (the generic
monomer/dimer species A/B/C below). The pipeline is applied to
single-particle-tracking microscopy of the **MET receptor** (c-Met /
hepatocyte growth factor receptor); the Experiment stage consumes the real
recordings under `Experiment/SPT_Data_MET_FAB_INLB_S-BSST712` (BioStudies
accession S-BSST712).

**Three molecular species:**
- **A (monomer):** Single receptor. Diffuses freely (diffusion coefficient `D_A`).
- **B (mobile dimer):** Two receptors bound, still mobile. Diffuses at rate
  `D_B` (typically slower than `D_A`).
- **C (immobile dimer):** Two receptors bound and immobilized. Diffuses slowly:
  `D_C = R_C · D_A`, where the learnable relative diffusivity `R_C`
  (`relative_diffusivity_chi`) is sampled over log10 [−2, −1] — 1% to 10% of the
  monomer diffusivity.

**Reactions:**
- **`A + A ↔ B`:** Two monomers associate at rate `k_on`; the dimer dissociates
  at rate `k_off`.
- **`B ↔ C`:** Mobile dimer immobilizes at rate `k_imm` (`rate_immobility`);
  the immobile dimer remobilizes at rate `k_mob` (`rate_mobility`) — a
  reversible mobility switch, with both directions learnable reaction channels.
- **Future extension:** **`C → ∅`:** degradation/recycling of the immobile
  dimer, a biological extension reserved for future work.

**Parameters to infer (θ) — the ten learnables**, each sampled log-uniformly
(prior ranges in log10 space, from the parameter table in `parameterization.py`):
- `count_alp`, `count_bet`, `count_chi`: initial particle counts of A, B, and C
  (counts; log10 prior [0, 2.5] each, i.e. 1 to ~316 particles)
- `diffusivity_alp` (`D_A`): monomer diffusion coefficient (μm²/s; log10 prior
  [−1.25, −0.25])
- `relative_diffusivity_bet` (`R_B`): dimensionless mobile-dimer diffusivity
  ratio, `D_B = R_B · D_A` (log10 prior [−0.625, −0.125])
- `relative_diffusivity_chi` (`R_C`): dimensionless immobile-dimer diffusivity
  ratio, `D_C = R_C · D_A` (log10 prior [−2, −1])
- `relative_rate_dimerization` (`R_ON`): dimensionless dimerization rate as a
  fraction of the diffusion-limited Smoluchowski cap (log10 prior [−2, 0])
- `rate_dissociation` (`k_off`): dissociation rate, B → A + A (1/s; log10 prior
  [−1, 1])
- `rate_immobility` (`k_imm`): immobilization rate, B → C (1/s; log10 prior
  [−1, 1])
- `rate_mobility` (`k_mob`): mobilization rate, C → B (1/s; log10 prior [−1, 1])

**Detector parameters — calibrated, not free constants** (inferred in the
Stage-1 detector workflow, then marginalized as the calibrated-imaging nuisance
for molecular inference; see §3):
- Point-spread-function (PSF) width: **inferred** as a lognormal distribution
  over the Gaussian width (median `mu_r`, log-spread `sigma_r`) — *not* a fixed σ.
- Brightness and photophysics: **inferred** — emitter brightness (median `mu_pc`,
  log-spread `sigma_pc`), photobleaching probability `prob_photo_bleach`, and flicker
  rate `lambda_rate`.
- The five EMCCD camera parameters — gain-conversion ratio `gamma = g/C`, optical
  background `kappa_o`, read noise `kappa_s`, baseline `kappa_b`, and quantum
  efficiency `kappa_q`: **marginalized as the SCOPE camera nuisance**, not inferred.
  They are non-identifiable from the videos (only the product `gamma·kappa_q` sets the
  amplitude), so each is drawn from an a-priori box and integrated over
  (`DETECTOR_WORKFLOW.md` §9.3); `g` and `C` are fixed nominal spec metadata for the
  `gamma` drift check.
- Video frame rate: 50 Hz (20 ms per frame) — a fixed sampling cadence.
- Recording length: supplied per run via `total_time_seconds` (commonly 2 s,
  5 s, or 10 s)

---

## §3. Inference Pipeline: Two-Stage Architecture

The full program separates detector inference from molecular-parameter
inference. The detector is characterized first, on a simpler system, and then
marginalized — drawn per simulation from its calibrated nuisance — while the
molecular parameters are inferred. This disentangles
optical and sensor effects from the biological reaction-diffusion parameters,
reduces the dimensionality of each inference problem, improves posterior
geometry, and speeds convergence.

### Stage 1: Detector Parameters (this repository — the Detector calibration workflow)

**Input:** Synthetic videos from a diffusion-only model (reactions disabled;
pure Brownian motion), rendered through the same imaging model.

**Objective:** Infer the six learnable imaging parameters (`β`) — the PSF-width
lognormal (`mu_r`, `sigma_r`), the emitter-brightness lognormal (`mu_pc`,
`sigma_pc`), the photobleaching probability `prob_photo_bleach`, and the flicker
rate `lambda_rate` — with the physics frozen so the imaging model is
identifiable. The EMCCD camera chain (`gamma`, `kappa_o`, `kappa_b`, `kappa_s`,
`kappa_q`) is **not** inferred: it is marginalized as the SCOPE camera nuisance,
each value drawn per simulation from its a-priori box (§2).

**Output:** A posterior over `β` and a versioned, provenanced imaging-parameter
artifact. This calibration is a complete workflow parallel to the biology
pipeline, run in this repository with its own committed submission machinery,
separate from — never wired into — the biology `Submit.sh` dispatcher and its
stage wrappers.
The calibrated values are the basis for the detector parameters Stage 2 applies;
the mechanism that seeds them into production is developed alongside this
workflow.

### Stage 2: RDS Parameters (this repository)

**Input:** Synthetic videos generated in two steps:
1. **RDS simulation:** ReaDDy solves the DIMER reactions and diffusion for the
   configured recording length.
2. **DLI imaging:** A Gaussian PSF, Poisson photon noise, and EMCCD readout
   noise are applied. The imaging block is **marginalized** per simulation: the
   six calibrated photophysics parameters are drawn from the persisted
   `Nuisance_DLI` artifact (which may be collapsed to a single representative
   vector, such as the sample geometric median), and the five SCOPE camera
   parameters are drawn from their a-priori boxes.

**Objective:** Infer the RDS parameters (`θ`) with the imaging block
marginalized — `p(θ | video)`, integrating over the calibrated-imaging and
SCOPE camera nuisances rather than conditioning on a single fixed `β`.

**Output:** A version-portable estimator artifact and a trained neural-network
checkpoint. Both are written under canonical, duration-stamped names that every
downstream stage loads (`Posit/…_Estimator.npz`, `Labor/…_Optimum_ANN.pth`), and a
re-run on the same duration overwrites them. During training the loop also writes a transient full-state
resume file beside the checkpoint (`Labor/…_Resurrect_State_ANN.pth`) every epoch, from
which a `--resurrect` requeue hot-restarts; it is overwritten continuously and is not a
downstream deliverable. So that superseded models stay identifiable and recoverable,
every finished run also writes a provenance-named backup of both — encoding the
train/test set sizes, epochs, and test loss. See the HPC operations runbook
(*Artifact backups*) for the naming and restore conventions.

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

**Neighbor-list skin (performance, not physics).** ReaDDy finds reaction partners
with a cell-linked list whose cell edge is `reaction_radius + skin`. In the large,
dilute imaging box (~40 µm across, ~1000 particles) the reaction radius alone
(= one particle diameter = 10 nm) partitions the box into a ~16-million-cell grid
that is >99.99% empty, and per-step management of that grid — not the physics —
dominates runtime. The **skin** (Verlet skin) decouples the cell size from the
reaction radius: enlarging it coarsens the grid and recovers roughly a 13× RDS
speedup, while the physics is untouched (reactions still fire only at the true
reaction radius; the skin only widens which particles are considered as
candidates). It is exposed as `SimulationRDS.neighbor_list_skin_factor` — a
**multiple of the particle diameter** (default `10×` = 100 nm) — overridable per run
via `--skin-factor` (RDS entry points, `Generate_Datasets.py`) or the `SKIN_FACTOR`
batch knob. The cost is U-shaped: too small leaves the empty-cell sweep, too large
collapses the box toward one cell and degrades the candidate search to O(N²); the
default sits on the broad fast plateau and clears the worst-case per-step
displacement (~47 nm at max diffusivity) with margin, so no reaction is missed.

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

**Dimer brightness model:** A dimer is two labels within one point-spread function,
so single-emitter fitting (ThunderSTORM, or the eye) records **one** spot whose
photons combine. The model combines the two labels by the **sum** of two independent
monomer brightness draws (`dimer_model="sum"`, the default): each label is drawn from
the same per-monomer brightness log-normal (`mu_pc`, `sigma_pc`) with its own
independent flicker trajectory, and their photon counts **add**. This is the
physically grounded combination for two independent, always-on labels — an *n*-mer's
brightness distribution is the *n*-fold convolution of the monomer's (Mutch et al.
2007; Digman & Gratton number-and-brightness) — giving a dimer a mean of `~2×` a
monomer with a lighter upper tail than rigidly doubling one draw. It is corroborated
by the data: the fitted per-spot intensity is `~1.8×` between the monomer-dominated
MET-FAB and the dimer-rich MET-INLB condition (a lower bound, given the analysis
intensity-range clip; see `DETECTOR_WORKFLOW.md` §6.4–§6.5).

The retained alternative, `dimer_model="multiply"` (a non-default mode of the shared
renderer `render_dli_video`, which reads `dimer_mule` from the biology parameter table),
rigidly scales a single monomer draw by `dimer_mule` — a
per-dataset photophysical constant in `[1, 2]`: **`2`** for two permanently-on,
both-present labels (the MET ATTO 647N default); **`√2 ≈ 1.41`** when only ~one label
is visible on average (a photoswitching dye, whose time-averaged brightness is the
geometric mean `GM(1×, 2×) = √2`, or ~50% labeling). It shares the `~2×` mean but
has a heavier upper tail, and is kept for sensitivity checks.

The condition difference (MET-FAB vs MET-INLB) is carried by the **dimer fraction** (the
species counts), with the combination model and the per-monomer brightness `mu_pc`
shared across conditions — not by a per-condition brightness.

**Photobleaching model:** Each emitter can irreversibly transition to a dark
(bleached) state, applied per frame through a two-state transition matrix. The
per-frame bleach probability is
`prob_1 = 1 − (1 − prob_photo_bleach)^(1 / numb_photo_bleach)`, where
`numb_photo_bleach = 100` is a fixed reference-frame count (a calibration
convention, *not* the clip or video frame count) and `prob_photo_bleach` is the
cumulative bleach fraction over that 100-frame (2 s at 50 Hz) reference window —
a detector-inferred photophysics parameter, marginalized per simulation from the
`Nuisance_DLI` artifact (production calibration ≈ 0.107), not a fixed constant.

The bleach rate is parameterized by this pair — a cumulative bleach probability
together with the reference-frame count over which it accrues — rather than by a
single hardcoded 100-frame probability, which makes the model correct for any
recording length. Because `numb_photo_bleach` is pinned to 100 regardless of
clip duration, the per-frame rate `prob_1` is constant across video lengths; the
cumulative bleach over an `n`-frame clip is
`p_video = 1 − (1 − prob_photo_bleach)^(n / numb_photo_bleach)`, so a longer clip
accumulates more bleaching through repeated application of the same per-frame
transition matrix (at `prob_photo_bleach ≈ 0.1`: about 10% over 100 frames /
2 s, about 41% over 500 frames / 10 s).

The reference count is pinned to 100 by design rather than tied to the clip
length. The value of `prob_photo_bleach` over the 100-frame reference was
calibrated by detector-parameter SBI inference on the experimental raw videos,
with the reaction-diffusion parameters held out — the production `Nuisance_DLI`
calibration puts it at ≈ 0.107; `numb_photo_bleach = 100` is the
convention that inference was run under, so it is preserved to keep the
calibrated value meaningful. Making `numb_photo_bleach` track the clip length
would render the per-frame rate duration-dependent — unphysical, since bleaching
is a property of the fluorophore and illumination, not of how long a recording
happens to be — and would contradict the detector-inferred value. (The aggregate
`p_bleach` reported by swift / SPTAnalyser is a track-disappearance rate that
lumps together bleaching, diffusion out of the field, blinking and gaps, and
unbinding; it describes a different process and is not substituted for
`prob_photo_bleach`.)

**Detector noise model (EMCCD Poisson–Gamma–Normal).** The imaging stage renders
camera counts in a single function, `add_noise` (`simulation_dli_support.py`),
following the physically grounded EMCCD chain specified in
`REFERENCE_EMCCD_NOISE_MODEL.md`: Poisson photoelectrons, stochastic `Gamma(N, g)`
electron multiplication (excess-noise factor `F² = 2`), conversion to ADU by the
factor `C`, a gain-independent Gaussian read noise of standard deviation `σ` added
*after* the register, and a constant camera baseline `b`. The gain `g` and
conversion `C` enter the image likelihood only through the ratio `γ = g/C` (the
ADU-per-photoelectron), so `γ` is inferred directly and `g`/`C` are fixed nominal
metadata; the optical background `kappa_o`, read noise `σ`, and baseline `b` are
identified directly from the background and dark pixels.

The read-noise term produces a small negative excursion (pixels below the baseline, and
rarely values above the sensor range); the non-negative storage clip in
`convert_video_dtype` removes it, aligning the stored synthetic with the recordable
camera domain, and its floor is re-examined against the read-noise scale after
re-calibration. Parameter recovery is validated on the held-out synthetic EVAL
namespace; real recordings have no ground truth. See `REFERENCE_EMCCD_NOISE_MODEL.md`
for the full specification, moments, prior ranges, and sources.

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
   resumes training. When a full-state resume file is present it **hot-restarts**
   from the exact latest state — model weights, optimizer moments, learning-rate
   schedule, global epoch, and warm-restart counters — so the schedule continues
   seamlessly and no epochs are spent re-converging. When it is absent (the first
   resumed run, or the file was deleted) it falls back to loading the best-on-test
   checkpoint weights into a fresh optimizer at the peak learning rate, then writes a
   resume file so the next requeue hot-restarts. A new optimum overwrites the
   checkpoint. This makes incremental training across separate, wall-time-limited
   invocations behave like one continuous run.
4. **In-run warm restart (automatic):** The training loop monitors the learning
   rate; once it has decayed to its floor and stalled there without improving, the
   loop reloads the best checkpoint and restarts the rate at a decaying peak — a
   plateau-escape that periodically re-raises the rate to find a better optimum. Each
   restart peak is `warm_restart_factor` (default 0.25) times the previous, a dedicated
   amplitude knob separate from the per-epoch anneal `scheduler_factor`, so the restart
   stays a gentle probe (a quarter of the peak) rather than a jump halfway back up. Its
   state is carried in the resume file, so the sawtooth continues mid-stride across a
   `--resurrect` requeue rather than resetting. Governed by `warm_restart_dwell`
   (epochs of stalled floor before a restart; `0` disables it); composes with
   `--resurrect`.
5. **Output:** Posterior samples are obtained by passing a real experimental
   video (or a synthetic holdout) through the trained network.

**Output:**
- A version-portable estimator artifact (`Estimator.npz`), loaded downstream as a
  `DirectPosterior`.
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

**Multi-GPU / multi-node scaling.** Training, MAP-recovery evaluation, and the
real-data application adapt to the allocation — across GPUs on one node and across
nodes. `--gres` is per node, so `--nodes=N --gres=gpu:G` gives `world_size = N*G`
ranks. Launched with one worker per GPU (via `torchrun` on one node, or `srun` +
`torchrun` with a c10d rendezvous across nodes), training runs data-parallel
through `DistributedDataParallel` — each worker holds a replica, processes its own
shard of every batch, and synchronizes gradients each step across every rank on
every node, with `SyncBatchNorm` sharing batch statistics across all ranks by
default (so the batch size is per-rank and the effective batch is
`batch*world_size`) — while MAP-recovery evaluation partitions the held-out videos
across the ranks, and the experiment stage partitions its `(condition, cell)` work
the same way, each writing its own shard to the shared filesystem and a single
`--merge` step combining them into one report. The single-GPU run is the collapse
case of the same code: with one worker the distributed wrappers reduce to no-ops
and the loop is exactly the original single-GPU path, so behavior is unchanged
where only one GPU is present. The per-node GPU count is read from the allocation,
capped by an optional `SRM_AND_SBI_GPUS` override (set it to 1 to force the
single-GPU path), and the node count from `SLURM_NNODES`; `SRM_AND_SBI_NO_SYNC_BN=1`
opts each worker into its own local batch statistics for speed at the cost of
re-validating recovery.

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

- **`workflow.py`** — the shared-engine control layer. It defines the frozen
  `WorkflowConfig` and the two factories `biology_workflow()` and
  `detector_workflow()` that build it. The config carries the genuine
  per-workflow differences — the workflow tag, the alias-qualified paths, the
  parameterization module, and the console-log paths — so one engine serves
  both the biology workflow (infers the ten reaction-diffusion parameters and
  marginalizes the imaging block) and the detector workflow (infers the six
  imaging parameters and marginalizes the reaction-diffusion domain and the
  camera).
- **`simulation_rds_runner.py`, `simulation_dli_runner.py`,
  `inference_runner.py`, `evaluation_runner.py`, `experiment_runner.py`** — one
  shared runner per stage, each exposing `run_<stage>(cfg, args)`. This is the
  single orchestration engine both workflows execute for that stage, so neither
  can silently drift; the per-workflow differences are localized in a
  `_<stage>_spec(cfg)` resolver rather than duplicated across entry points. In
  particular, `simulation_dli_runner`'s per-stage spec resolver resolves the DLI
  imaging source — the `Nuisance_DLI` artifact for biology, the imaging prior
  box for detector.
- **`parameterization.py`** — the single source of truth for configuration. It
  loads the per-machine profile, holds the sibling-wide defaults as frozen
  dataclasses (path conventions, simulation geometry and timing, RDS/DLI
  defaults, inference training and network architecture, plotting), and carries
  the rich parameter specification (parameter ranges, log flags, units, and
  labels). It exposes a typed `PARAMETERS` singleton and prior/bounds helpers,
  and validates the configuration at import time.
- **`detector_parameterization.py`** — the detector workflow's parameter
  contract (value-based roles): the six learnable imaging parameters' priors,
  with the reaction-diffusion block and the SCOPE camera marginalized as
  nuisances; deliberately decoupled from `parameterization.py`, since the two
  workflows' roles and ranges differ by design.
- **`simulation_rds_support.py`** — the ReaDDy primitives: system builder,
  simulation builder, and trajectory-pose extraction (the extractor is reused by
  the imaging stage, which also requests a dimer mask).
- **`detector_simulation_rds_support.py`** — the detector RDS forward model:
  reuses the canonical `build_system` (with `pure_diffusion=True`) and
  `build_simulation` by import, and draws the six reaction-diffusion nuisance
  values (three species counts, three diffusivities) per simulation.
- **`simulation_dli_support.py`** — the imaging pipeline: Gaussian PSF, EMCCD
  detector, intensity accumulation, the brightness state machine, the transition
  matrices (including photobleaching), and the top-level imaging orchestrator.
- **`detector_simulation_dli_support.py`** — the detector DLI forward model:
  re-exports the shared, source-agnostic renderer `render_dli_video` under the
  detector-facing name `render_detector_video`, so both DLI stages render
  through one implementation.
- **`detector_nuisance_dli.py`** — the `Nuisance_DLI` calibrated-imaging
  nuisance: the artifact format, its construction from the detector posterior
  (including the pool modes and the single-vector sample-geometric-median
  collapse), and the require-gate through which the biology DLI stage loads it.
- **`experiment_support.py`** — the workflow-agnostic real-recording machinery:
  loading, windowing, and preparing experimental microscopy videos. It is shared
  by both the Experiment stage and the `Nuisance_DLI` analysis, so it carries no
  workflow-specific assumptions.
- **`inference_network.py`** — the network architecture (§6): positional
  encoding, attention block, temporal transformer, and the 3D-CNN video encoder.
- **`inference_support.py`** — the training pipeline: the video dataset (with
  augmentation), normalization, training/validation set-up, the training loop
  (with the resurrect branch), and posterior save/load.
- **`artifacts.py`** — the self-describing, version-portable estimator artifact:
  persists a trained estimator as separable components (compile-stripped
  `state_dict`, rebuild spec, parameter schema) in one `.npz`, so it
  reconstructs under whatever torch version loads it; the sole persisted
  estimator format for both workflows.
- **`evaluation.py`** — the MAP-recovery core shared by the validation stages:
  the seed-then-optimize estimator, posterior-quantile summaries, and report
  tables.
- **`posterior_calibration.py`, `posterior_calibration_runner.py`** — the
  workflow-agnostic posterior-calibration diagnostic (an Analysis tool, not a
  pipeline stage). The kernel scores a trained posterior's calibration with
  simulation-based calibration, expected coverage, TARP, and local C2ST (§7),
  overall and stratified by target-theta dimension, operating only on pre-drawn
  theta-space arrays and embeddings (it wraps `sbi.diagnostics` and imports nothing
  from `parameterization`/`artifacts`). The runner streams the EVAL set, draws each
  video's posterior samples, sample and truth log-densities, and embedding, and
  writes the report; its two namespaced Analysis shims share it exactly as the stage
  shims share their runner.
- **`estimator_comparison.py`, `estimator_comparison_runner.py`** — the
  workflow-agnostic estimator-comparison diagnostic (an Analysis tool, not a pipeline
  stage). The kernel decides whether one trained estimator generalizes better than
  another by the paired log-score on the shared `(task, sim)` subset of the held-out
  TEST set (Diebold-Mariano + Wilcoxon + paired bootstrap; §7), operating on two
  per-video loss arrays and importing nothing from `parameterization`/`artifacts`. The
  runner reads two `TestLossDistribution` artifacts (per-video loss keyed by
  `(task, sim)`), runs the statistics, and reports; its two namespaced Analysis shims
  share it over one engine. No GPU.
- **`test_loss_distribution.py`** — the per-example best-epoch TEST-loss
  artifact: holds the held-out per-example loss keyed by `(task_index,
  sim_index)` with a self-describing manifest; written by the Inference stage
  and read by the comparison and test-loss analyses.
- **`test_loss_analysis.py`, `test_loss_analysis_runner.py`** — the workflow-agnostic
  test-loss-distribution analysis (an Analysis tool, not a pipeline stage). The kernel
  reads a best-epoch per-example NLL artifact and produces the distribution shape, the
  uniform-prior NLL reference (the no-information baseline), and the tail-vs-parameter
  identifiability read — which learnable parameters, and which end of their range, mark
  the hardest examples (for biology the species counts, whose low end yields
  uninformative videos). Everything is read from the artifact manifest, so it is
  workflow-agnostic; the runner resolves the artifact through `cfg.paths` (or an ad-hoc
  `--tld-path`) and reports; its two namespaced Analysis shims share it. No GPU.
- **`embedding_space_distance.py`** — the workflow-agnostic embedding-space
  distance kernel: the maximum-mean-discrepancy (MMD, permutation null) and
  classifier two-sample (C2ST) statistics over trained-embedding vectors,
  blocked by recording so within-recording correlation cannot masquerade as a
  real difference.
- **`embedding_space_distance_runner.py`** — the shared engine for the
  experimental-versus-synthetic embedding-distance analysis (does the trained
  network place the real recordings where it places its own synthetic
  distribution?); its two namespaced Analysis shims share it.
- **`sample_geometric_median.py`** — the workflow-agnostic sample-geometric-median
  kernel: the correlation-preserving single-vector summary of a cloud of
  parameter vectors (the median vector snapped to a realized sample, never the
  vector of per-dimension medians).
- **`sample_geometric_median_runner.py`** — the shared engine for the
  sample-geometric-median analysis over the Experiment MAP cloud (and, for
  detector, the `Nuisance_DLI` pool); its two namespaced Analysis shims share it.
- **`posterior_predictive_video_runner.py`** — the shared engine for the
  posterior-predictive video comparison: render a synthetic video at the
  parameters inferred from one real recording and put the two side by side; its
  two namespaced Analysis shims share it.
- **`temporal_dynamics.py`** — the workflow-agnostic temporal-dynamics kernel:
  scatters the per-window Experiment estimates into a (condition, recording,
  window) grid; forms the two Sample-Geometric-Median central estimates (the
  trajectory-level medoid, one real recording across the whole time course, and
  the per-time-point medoid, whose selected recording may change between time
  points); fits the per-recording drift in dex with its sign-consistency and
  signed-rank test; and separates the within-window posterior spread from the
  between-cell spread so the two are never conflated.
- **`temporal_dynamics_runner.py`** — the shared engine for the temporal-dynamics
  analysis (does an inferred value hold still across a recording?); its two
  namespaced Analysis shims share it. Both workflows must be run to interpret
  either: biology holds imaging fixed and so is blind to imaging drift, the
  detector marginalizes the reaction-diffusion block and so is blind to
  biological drift, and the two read the same recordings — each is the other's
  control, and neither attributes a cause alone.
- **`io.py`** — file I/O: transparent loading of `.zarr`/`.npy`/`.npz`, video
  and theta-set writing, and bit-depth conversion. All paths come from the
  configuration helpers, never hardcoded.
- **`visualization_rds.py`, `visualization_dli.py`, `visualization_inference.py`,
  `visualization_calibration.py`** — stage-specific diagnostic and figure builders
  (matplotlib imported lazily so headless runs do not pay its import cost).
- **`diagnostics.py`** — the shared diagnostics engine behind the `--debug` /
  `--debug-dump` flags: per-step checkpoints, fail-loud invariant checks,
  quantitative stats, and a self-contained dumped report.
- **`utils.py`** — small cross-cutting helpers (terminal separators, memory-state
  logging, and the resource-probe helpers).

Each Prime entry point is a thin shim: it parses CLI arguments, builds a
`workflow.WorkflowConfig` (via `biology_workflow()` or `detector_workflow()`),
and calls the stage's shared `run_<stage>(cfg, args)` runner, which loads the
configuration, calls the package functions, and writes outputs to the
configuration-defined paths. Each stage has two such shims over one shared
runner — the unqualified biology entry point and its `_DETECTOR`-qualified
detector counterpart. The package is installed editable, so edits to the package
take effect without reinstallation.

`Script_Bank/Analysis/` collects post-hoc analyses that run on completed outputs
rather than producing pipeline artifacts:
`SRM_AND_SBI_DIMER_ALP_Experiment_Temporal_Dynamics.py` tracks each inferred
parameter's MAP estimate over the real recordings per condition (non-overlapping
chunk → time), overlays the experimental range for the parameters the source paper
constrains (Li et al. 2026, doi:10.1002/smll.202507115), annotates each figure with
its held-out recovery quality, and writes figures plus a self-contained `report.md`;
its companion `Experiment_Temporal_Dynamics.md` gives the full interpretation.
`SRM_AND_SBI_DIMER_ALP_Seeding_Validation.py` checks the RNG / non-determinism
behavior of the generation stack.

`SRM_AND_SBI_DIMER_ALP_Posterior_Calibration.py` and its
`SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Calibration.py` twin score how
well-calibrated a trained posterior is on the held-out EVAL set — simulation-based
calibration, expected coverage, TARP, and local C2ST (§7), overall and stratified by
each target parameter — over one shared engine
(`posterior_calibration_runner.run_posterior_calibration`), the same two-shim structure
the pipeline stages use, so one implementation serves both workflows and the entry-point
name carries the namespace. It reads only the estimator and the EVAL set, writes its
report to `Posit/`, is multi-GPU sharded with a `--merge` combine step, and is kept out
of the stage dispatcher; the companion `SRM_AND_SBI_DIMER_ALP_Posterior_Calibration.md`
documents both workflows.

`SRM_AND_SBI_DIMER_ALP_Estimator_Comparison.py` and its
`SRM_AND_SBI_DIMER_ALP_DETECTOR_Estimator_Comparison.py` twin decide whether one trained
estimator generalizes better than another by the paired log-score on the shared
`(task, sim)` TEST subset — pairing cancels each video's intrinsic entropy floor, so the
difference isolates the two estimators' KL gap (§7) — over one shared engine
(`estimator_comparison_runner.run_estimator_comparison`), the same two-shim structure.
It reads two `TestLossDistribution` artifacts, needs no GPU, writes its report to
`Posit/`, is kept out of the dispatcher, and is documented for both workflows in
`SRM_AND_SBI_DIMER_ALP_Estimator_Comparison.md`.

`SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls.py` reuses the trained DIMER-ALP
posterior — with no retraining — to MAP-estimate parameters from real recordings of two
oligomeric-state control receptors, the constitutive monomer CD86 and the constitutive dimer
CTLA-4 (BioImage Archive accession S-BIAD1369). A special-scope, ad-hoc reuse of the posterior
on a different study's data, it clones the
Experiment stage bar the dataset folder, output directory, and default conditions, and is kept
out of the stage dispatcher. Run it under the inference environment (single-process or
multi-GPU sharded, with a `--merge` pass to concatenate shards; `--dry-run` first); it writes a
per-condition inferred-parameter `report.md`, per-parameter figures, and the reusable
per-(cell, chunk) `.npz` arrays. Real data carry no ground truth, so the deliverable is a
per-condition distribution rather than a recovery check, with the diffusion scale as the
transferable quantitative read-out. Usage and interpretation are in the companions
`SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls.md` and
`SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls_Temporal_Dynamics.md`.

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
function in the package, driven by a thin entry-point shim over the stage's
shared runner, and produces a defined on-disk artifact. Module paths are relative to the package
`srm_and_sbi_dimer_alp/`; entry-point scripts live under `Script_Bank/Prime/`.

| Scientific concept / stage | Code (module → function/class) | On-disk artifact |
| --- | --- | --- |
| DIMER reaction system (`A + A ↔ B`, `B ↔ C`): species, diffusion coefficients, reaction rates, simulation box | `simulation_rds_support.py` → `build_system()`; the ReaDDy simulation is then assembled by `build_simulation()` | (in-memory ReaDDy system/simulation; trajectory written below) |
| RDS trajectory recording (particle positions, species, time over the recording length) | entry point `SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py` (drives `build_system()` → `build_simulation()`) | trajectory `.h5` (HDF5, ReaDDy convention); sampled theta set `.zarr` (via `io.py` → `save_theta_set()`) |
| Trajectory extraction (per-frame poses and the dimer mask) | `simulation_rds_support.py` → `extract_trajectory_poses()` (reused by the imaging stage, which requests the dimer mask) | (per-frame pose arrays passed to imaging) |
| Diffraction-limited imaging forward model: Gaussian PSF, Poisson + EMCCD readout noise, dimer-brightness scaling, photobleaching | `simulation_dli_support.py` → `render_dli_video()` (source-agnostic renderer of an assembled 11-key imaging vector; shared by both DLI stages), with `Gaussian` / `sample_psf_width()` (PSF), `compute_intensity()` + `add_pixel_counts()` (intensity accumulation), `compute_brightness()` / `compute_brightness_probability()` and `generate_state_trajectories()` (brightness state machine), `compute_matrices()` (transition matrices incl. photobleaching), `EMCCD` / `add_noise()` / `generate_frames()` (detector noise) | (noised video array passed to writer below) |
| DLI video output (chunked, bit-depth-converted) | entry point `SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py` (drives `extract_trajectory_poses()` → `render_dli_video()`, with the imaging block marginalized from `Nuisance_DLI` + the SCOPE box) | `.zarr` video set, shape `(frame_count, height, width)` (via `io.py` → `convert_video_dtype()`, `save_video_set()`) |
| Parameter prior and specification (ranges, log flags, units, labels; log-uniform prior and bounds) | `parameterization.py` → `PARAMETERS` (a `Parameters` singleton) with `build_prior()`, `theta_lower_bound()`, `theta_upper_bound()`, `parameter_find()` | (configuration in code; sampled theta persisted in the RDS theta-set `.zarr`) |
| NPE + MAF estimator with 3D-CNN + temporal-transformer embedding | `inference_network.py` → `Complex3DCNN` (video encoder), `TemporalTransformer` (with `AttentionBlock`, `PositionalEncoding`); training wired in `inference_support.py` → `setup_training()`, `train_loop()` (with the resurrect branch) | (in-memory network; checkpoint + posterior written below) |
| Leak-proof TRAIN / TEST / EVAL split, sizing rule, and dataset construction | entry point `SRM_AND_SBI_DIMER_ALP_Generate_Datasets.py` (runs RDS → DLI per split with the `CORE = TRAIN + TEST`, `EVAL = max(floor, 0.1·CORE)` sizing); dataset assembly in `inference_support.py` → `build_datasets()` (with `VideoDataset`, `normalize_video()`) | `_TRAIN` / `_TEST` / `_EVAL`-suffixed trajectory `.h5` and video `.zarr` namespaces |
| Posterior training run (gradient updates on TRAIN, selection on TEST) | entry point `SRM_AND_SBI_DIMER_ALP_Inference.py` (drives `build_datasets()` → `setup_training()` → `train_loop()`, then `artifacts.save_estimator()`) | version-portable estimator artifact (`Estimator.npz`, via `artifacts.py` → `save_estimator()`), loaded downstream as a `DirectPosterior`; network checkpoint at each new optimum |
| MAP recovery and calibration on held-out EVAL | `evaluation.py` → `map_estimate()` (seed-then-optimize: `collect_theta_prex()`, `collect_score_prex()`, `extract_elite_prex()`, `optimize_elite()`), `posterior_summary()`, `recovery_stats()`, `recovery_table()`, `posterior_coverage_table()`; driven by entry point `SRM_AND_SBI_DIMER_ALP_Evaluation.py` | recovery report (figures + tables + arrays + a live `progress.log`) under the validation output directory |
| Real-data application (no ground truth) | same `evaluation.py` estimator (`map_estimate()`, `experiment_table()`); driven by entry point `SRM_AND_SBI_DIMER_ALP_Experiment.py` | per-condition inferred-parameter report under the validation output directory |
| Configuration, paths, storage routing, and file I/O | `parameterization.py` → `Paths`, `MachineProfile` / `load_machine_profile()`, `FrameConfig`, `RunTiming`; `io.py` → `load_data()`, `save_video_set()`, `save_theta_set()`, `convert_video_dtype()` | resolved absolute paths (per-machine `machine_profiles.toml`); all artifacts above land under the configured roots |

The five pipeline stages are RDS, DLI, Inference, Evaluation, and Experiment.
Each stage has one shared runner (`<stage>_runner.py` → `run_<stage>()`) and two
thin Prime entry-point shims over it: the unqualified biology entry point
(`SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py`,
`SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py`, `SRM_AND_SBI_DIMER_ALP_Inference.py`,
`SRM_AND_SBI_DIMER_ALP_Evaluation.py`, `SRM_AND_SBI_DIMER_ALP_Experiment.py`) and
its `_DETECTOR`-qualified detector counterpart. Each shim parses arguments,
builds a `WorkflowConfig`, and calls the shared runner, which drives the package
functions above and writes outputs to the configuration-defined paths.
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
2. **Reaction-diffusion primitive equivalence.** ReaDDy is a stochastic
   simulator (its stepper draws random numbers for diffusion and reaction
   events, so trajectories vary run-to-run), but the construction of its system
   specification (species, reactions, rates, box) is deterministic: the system
   and simulation builders construct exactly the declared model for a given
   theta.
3. **Imaging-pipeline functional equivalence.** The Gaussian PSF (erf-based
   pixel integration), the EMCCD model (Poisson photoelectrons, stochastic Gamma
   multiplication, Gaussian readout), the
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

### Estimator generalization and test-loss interpretation

The Inference stage reports one per-epoch scalar, the **mean test loss** (the
flow's negative log-density on the held-out TEST set). That mean is a
*consistency* check, not a generalization metric, and reading it correctly
matters:

- **It confounds two quantities.** The logarithmic score is a strictly proper
  scoring rule whose associated divergence is the Kullback–Leibler divergence, so
  the expected loss decomposes into an intrinsic **posterior-entropy floor** (a
  property of the problem, not the estimator) plus the estimator's **KL error**
  (Gneiting & Raftery 2007). The absolute mean is therefore not "the estimator's
  error," and two runs reproducing the same mean at different TEST-set sizes
  confirm only that the mean is well estimated (a 1/√N consistency result), not
  that the estimator generalizes.
- **A lower loss does not certify a better posterior.** Closeness in KL bounds
  neither moments nor calibration (Deshpande et al. 2022); posterior faithfulness
  is a separate measurement (below).
- **The central-limit reading is contingent.** The per-example negative
  log-density is unbounded above — one example where the flow assigns near-zero
  density at the truth contributes an arbitrarily large value — so if that upper
  tail is heavy the variance is large or undefined and the Gaussian standard error
  breaks down. Finite variance is *testable*, not assumed.

**Best-epoch test-loss distribution (instrumented).** Rather than the mean alone,
the stage captures the best epoch's **per-example** TEST loss, keyed by the
`(task_index, sim_index)` pair — authoritative against the on-disk task/sim
layout and extension-stable as the TEST set grows — with a self-describing
manifest (parameter table, prior bounds, θ-space, best epoch and loss). It is
committed alongside the checkpoint and posterior at each new best; because the
per-example loss is already computed to form the mean, capturing it is a
no-reduction store and essentially free (`--test-loss-distribution`, on by
default when a TEST set is present). Reporting is three-tier: the per-epoch mean
and standard deviation (a spread band on the loss curve); at each new best an
extended card (quantiles, skew, tail mass, train−test gap, bootstrap confidence
interval); and heavier analyses post-hoc. **Model selection stays by the mean** —
the extended statistics are diagnostic, since re-ranking checkpoints on an
outcome-selected tail subset would be an improper score (Gneiting & Ranjan 2011).

**Rigorous cross-run comparison (post-hoc).** To decide whether estimator A
generalizes better than B, form the per-example log-score difference on the
**shared** `(task_index, sim_index)` subset and test that its mean is zero.
Pairing cancels the common entropy term, so the statistic isolates the difference
of the two estimators' KL divergences to the truth (Amisano & Giacomini 2007) —
the Diebold–Mariano test of equal predictive accuracy (Diebold & Mariano 1995),
with the Wilcoxon signed-rank test and the paired bootstrap as heavy-tail-robust
alternatives. Comparing the two *means* of two different-sized sets is invalid: it
confounds the sets' intrinsic-entropy content. Whether the mean itself is
trustworthy is checkable from the stored scores (a subsampling-rate log-log slope
near −0.5; tail-index estimation; bootstrap skew); a heavy tail is itself the
generalization red flag the mean concealed. The `Estimator_Comparison` Analysis
diagnostic implements this over one shared, workflow-agnostic engine: it reads two
`TestLossDistribution` artifacts, pairs them on the shared `(task, sim)` subset, and
reports the Diebold-Mariano statistic with the Wilcoxon signed-rank test and the
paired-bootstrap interval as its heavy-tail-robust companions, for both the biology and
the detector estimators.

**Calibration and coverage.** A low loss cannot detect an overconfident
posterior, so faithfulness — the coverage read-out introduced above — is measured
by the field-standard amortized-inference diagnostics: simulation-based
calibration (rank uniformity of the true θ within posterior samples; Talts et al.
2018, with the necessary-not-sufficient caveat of Modrák et al. 2023), expected
coverage of credible regions (Hermans et al. 2022), TARP (necessary and
sufficient, in the population limit, for matching the true posterior; Lemos et al.
2023), and local C2ST for per-observation fidelity on the few real observations
(Linhart et al. 2023). These require posterior sampling and are therefore
periodic/post-hoc rather than per-epoch. The `Posterior_Calibration` Analysis
diagnostic implements all four over one shared, workflow-agnostic engine — reported
overall and stratified along each target parameter by the posterior's **inferred**
value (conditioning on the observation, since the rank is uniform conditional on it;
binning on the latent truth would confound Bayesian shrinkage with miscalibration). A
subregion where calibration degrades — the low-count regime for biology, an imaging
setting for detector — is thereby localized rather than averaged away, for both the
biology and the detector posteriors.

**Scope.** All of the above measure **in-distribution** generalization — to new
draws from the same prior-predictive. Out-of-distribution / real-data
generalization is a different question, answered by coverage under model
misspecification, the embedding-space experimental-versus-synthetic distance (MMD / C2ST,
implemented by the workflow-agnostic `embedding_space_distance` kernel and its shared
runner; the biology companion note is
`Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Embedding_Space_Distance.md`), and
posterior-predictive checks; misspecification-robust simulation-based inference
(Ward et al. 2022; Kelly et al. 2023 — pending independent verification) is the
relevant literature.

**References.** Gneiting & Raftery (2007), *Strictly Proper Scoring Rules,
Prediction, and Estimation*, JASA; Deshpande et al. (2022), *Are you using test
log-likelihood correctly?*, TMLR; Amisano & Giacomini (2007), *Comparing Density
Forecasts via Weighted Likelihood Ratio Tests*, JBES; Diebold & Mariano (1995),
*Comparing Predictive Accuracy*, JBES; Gneiting & Ranjan (2011), on the
impropriety of non-constant score weighting, JBES; Talts et al. (2018),
*Validating Bayesian Inference Algorithms with Simulation-Based Calibration* (with
Modrák et al. 2023); Hermans et al. (2022), *A Trust Crisis in Simulation-Based
Inference?*, TMLR; Lemos et al. (2023), *Sampling-Based Accuracy Testing of
Posterior Estimators (TARP)*, ICML; Linhart et al. (2023), *L-C2ST*, NeurIPS (on
the classifier two-sample test of Lopez-Paz & Oquab 2017); and, pending
independent verification, Ward et al. (2022) and Kelly et al. (2023) on
misspecification-robust neural posterior estimation.

### Matched-imaging embedding validation (planned)

The embedding-space experimental-versus-synthetic distance introduced above
(detailed in `DETECTOR_WORKFLOW.md`) compares the experimental recordings against
the held-out EVAL set, which spans the entire imaging prior. Because the
experimental recordings sit at a single imaging setting while the synthetic
reference is broad, the experimental embeddings form a tight cluster nested inside
the diffuse synthetic cloud, and a two-sample classifier separates the two by
concentration alone — even where their supports overlap. That separation is a
breadth artifact of a prior-spanning reference, not evidence that the experimental
recordings lie off the synthetic manifold. Isolating imaging realism at the
calibrated operating point requires a synthetic reference generated at the
inferred imaging, not across the prior.

A planned analysis supplies it: render synthetic videos with the imaging fixed to
the inferred values and the receptor counts and diffusion pinned to the
experimental data, then measure the same embedding distance against the
experimental recordings. The obstacle is that the Detector marginalizes the
reaction-diffusion block — particle counts and diffusion coefficients — so it
never estimates them for any single recording; they are learned only implicitly. A
matched render must therefore obtain them from outside the Detector.

Until the RDS estimator exists, that external source is the single-molecule
localization data (accession S-BSST712). Per-condition receptor counts come from
the localization density, corrected upward for the emitter on-fraction: the
localizations per frame undercount the receptors, both because dim or overlapping
emitters are missed and because a fraction of emitters are dark or bleached at any
instant. The bleached fraction follows from the calibrated photobleaching
probability through the fixed hundred-frame survival law; the blinking rate enters
only to second order, since the dimmest non-dark brightness state lies above the
localizer's photon-acceptance floor and flicker therefore does not manufacture
apparent darkness. Diffusion coefficients come from the tracked trajectories by
mean-squared displacement (`MSD(τ) = 4·D·τ` for two-dimensional free diffusion,
reported in µm²/s). Once the RDS estimator is trained it supplies the counts and
diffusion directly, closing the loop without the localization step.

The analysis reuses the existing machinery end to end — the reaction-diffusion
simulation at the pinned counts and diffusion, the imaging renderer driven by one
fixed inferred-imaging vector, and the embedding together with the
maximum-mean-discrepancy and classifier two-sample statistics of the
embedding-distance analysis — replacing only the synthetic reference, from
prior-spanning to a single operating point. The interpretation shifts accordingly:
a residual distance then measures model-versus-data mismatch at the calibrated
imaging, rather than the prior-averaged realism the broad reference reports. Like
the flicker-rate derivation drawn from the same localization data, it is a
read-only post-hoc utility rather than a pipeline stage, and gains its own
companion note when it ships.

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
- Posterior-predictive check — implemented as the `Posterior_Predictive_Video`
  analysis (biology and detector Analysis shims over
  `posterior_predictive_video_runner`): render a synthetic video at the
  parameters inferred from one real recording and compare the two side by side.

---

## §8. Open Scientific Questions

*(Scientific and methodological questions only.)*

**S1. Identifiability of RDS parameters.** Are all parameters in θ uniquely
identifiable from video data? Which parameters are confounded (co-determined)?
This calls for theoretical or empirical analysis.

**S2. Prior sensitivity.** How sensitive is the posterior to the prior choice
(uniform versus log-normal)? Should informative priors from the biophysical
literature be used?

**S3. Posterior convergence across restart cycles.** The training loop now performs
warm-restart cycles automatically within a single run (and `--resurrect` chains them
across runs); how many cycles are needed for posterior stability, and is there a
principled stopping criterion beyond an empirical plateau (for example, a test-loss
delta below tolerance, or a restart that fails to improve the best)?

**S4. Out-of-distribution robustness.** If the network is trained at one
recording length and tested at a different one (for example, trained on 2 s
videos and tested on 10 s), how does posterior accuracy degrade? Can it be
trained jointly on multiple durations?

**S5. Multi-cell heterogeneity.** Real microscopy data contains cells with
varying expression levels, spatial organization, and cell-cycle phase. Can the
single-cell, per-video model extend to population posteriors?

---

**End of Project Context — srm-and-sbi-dimer-alp**

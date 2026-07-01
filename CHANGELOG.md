# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.2.10 - 2026-07-01

Documentation: add the multi-GPU timing benchmark. No code change.

### Added

- `BENCHMARKS_Multi_GPU.md` — wall-clock timings for the three GPU stages that
  shard across the allocated devices (data-parallel training, and the sharded
  Evaluation and Experiment MAP passes), drawn from the production-scale 2 s and
  5 s runs rather than a synthetic micro-check. States the explicit per-epoch
  multiplier of the four-card node over a single card (1.4×–2.5×, contention-free
  floor to full-run average, plus the whole-run 21.9 h → 8.9 h equivalent), the
  sharded-stage wall-clocks and per-video rates, the 5 s wall-budget finding that
  motivates checkpoint-resume, and an explicit measurement-gaps section (no
  single-device MI210 point, no eight-GPU whole-node point yet).

### Documentation

- Reconciled `BENCHMARKS_Single_GPU.md`: its two forward references to the
  companion no longer promise a like-for-like check-scale comparison, since the
  multi-GPU numbers are production-scale.
- Surfaced both benchmark documents in the README documentation list; neither was
  referenced there before.

## 0.2.9 - 2026-07-01

Documentation: make the post-hoc analysis scripts discoverable. No code change.

### Documentation

- Named the `Script_Bank/Analysis/` scripts in the front-door docs (the README
  structure list and the PROJECT_CONTEXT entry-point section): the temporal-dynamics
  experiment analysis (`Experiment_Temporal_Dynamics`, with its experimental-range
  validation against Li et al. 2026 and its companion interpretation doc) and the
  seeding / non-determinism validation (`Seeding_Validation`). The folder was
  previously listed only generically, so a new user would not have discovered these
  analyses from the top-level documentation.

## 0.2.8 - 2026-07-01

Adds a standalone temporal-dynamics analysis of the inferred parameters on the real
experimental recordings. Additive only; no change to the pipeline stages.

### Added

- `Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Experiment_Temporal_Dynamics.py` — a
  post-hoc analysis (in `Analysis/`, not a pipeline stage) that reads a completed
  Experiment `MAP_Experiment.npz` and, per learnable parameter, plots the MAP estimate
  over the recording (non-overlapping chunk → time), averaged across cells per
  condition (MET-FAB / MET-INLB) in absolute units, with a between-cell band and faint
  per-cell trajectories. Its purpose is temporal dynamics, with a parameter-dependent
  robustness/stationarity read (constant-property parameters should be flat). For the
  parameters the source paper constrains (κ_OFF, D_A, R_B) it overlays the experimental
  range (band + reported values + mean) and the inferred time-average, and annotates
  each figure with its EVAL recovery quality. Writes one figure per parameter plus a
  self-contained `report.md` into a `temporal_dynamics/` subdirectory. Config-driven
  from `PARAMETERIZATION`; derives the 2 s (10 timepoints) / 5 s (4 timepoints) axis
  from the data; headless. Experimental references: Li et al., *Small* 2026, e07115
  (doi:10.1002/smll.202507115) — κ_OFF = 1/(dimer lifetime ≈ 1 s), D_A ≈ 0.10 µm²/s,
  R_B ≈ 0.6 (dimer ≈ 1.6× slower than monomer).
- `Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Experiment_Temporal_Dynamics.md` — the
  companion method/interpretation reference (temporal-dynamics-primary framing with
  parameter-dependent stationarity, why this exceeds the whole-recording experimental
  readout, the κ_OFF validation, the recovery × stationarity reliability view, caveats).

### Documented (not implemented)

- Aggregated posterior-distribution panels (the original per-parameter histogram of the
  posterior sample cloud pooled across all chunks) require the full per-window posterior
  sample pool, which the Experiment stage does not persist (only five quantiles per
  window). The approach is documented in the script and companion doc as a future
  extension.

## 0.2.7 - 2026-07-01

Documentation corrections and a backup convention. No code or behavior change.

### Fixed

- **Receptor identity (PROJECT_CONTEXT.md §2).** The system section mislabeled the
  modeled receptor as "EGFR-Like" / epidermal growth factor receptor. The pipeline's
  real-data application is the **MET receptor** (c-Met / hepatocyte growth factor
  receptor): the Experiment stage consumes MET single-particle-tracking recordings
  under `Experiment/SPT_Data_MET_FAB_INLB_S-BSST712` (BioStudies S-BSST712; MET
  engaged by an antibody fragment and Internalin B). §2 now names MET where receptor
  identity is asserted, and states explicitly that the A/B/C reaction scheme itself
  is receptor-agnostic (identity enters only through the experimental data). This was
  the only EGFR reference in the repository.
- **ReaDDy mischaracterized as deterministic (VALIDATION.md, PROJECT_CONTEXT.md).**
  The "Reaction-diffusion primitive equivalence" pillar stated "ReaDDy is
  deterministic given the same system specification," which is incorrect: ReaDDy is a
  stochastic particle-based reaction-diffusion simulator (Brownian diffusion plus
  stochastic reaction events), and the pipeline runs it seedless. Both copies of the
  sentence now state that ReaDDy is stochastic (trajectories vary run-to-run) while
  the deterministic property is the *construction* of the system specification (the
  builders produce exactly the declared model for a given theta). Every other
  "non-deterministic" statement in the docs was already correct and is unchanged.

### Documentation

- Added an "Artifact backups" section to the HPC operations runbook documenting the
  `<stem>_<TAG>_<DD.MM.YYYY>.<ext>` backup naming convention (German-format date;
  the suffix sits before the extension so backups never match the loaded artifact
  names), so preserved posteriors and checkpoints are discoverable and self-describing.

## 0.2.6 - 2026-07-01

Adds a special-situation entry point that constructs a posterior from a saved
checkpoint without retraining, for cross-machine weight transfer and checkpoint
recovery. Additive only; no change to the pipeline stages or their behavior.

### Added

- `Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Construction_Optimum_ANN.py` — builds a
  `DirectPosterior` pickle from an existing `Optimum_ANN.pth` checkpoint by running
  the Inference build-and-save sequence (Complex3DCNN + `build_maf`, `torch.compile`,
  `load_state_dict`, `DirectPosterior`, `save_posterior`) without the training loop.
  Single-process, single-GPU; the one-time `torch.compile` is its only real cost. It
  reproduces the Inference stage's posterior output for the situations where
  retraining is not wanted: moving trained weights between machines (copy the small
  `.pth`, construct the `.pkl` locally), or recovering a posterior from a run that
  checkpointed but was stopped before it wrote one. It is run ad hoc and is
  deliberately kept out of the `SRM_AND_SBI_DIMER_ALP_HPC_Submit.sh` dispatcher and
  the standard HPC wrapper set, which stay exactly the four canonical stages.

### Documentation

- Documented the entry point across the front-door references: the README structure
  list, the PROJECT_CONTEXT entry-point section, and a new "Special-situation entry
  points" section in the HPC operations runbook carrying the ad-hoc `sbatch --wrap`
  recipe — so its purpose and launch are discoverable without presenting it as a
  pipeline stage.

## 0.2.5 - 2026-06-30

Makes the inference DataLoader worker count rank- and loader-aware, so multi-GPU
training cannot exhaust host memory by multiplying worker processes. No change to
the scientific behavior or to the single-GPU path.

### Changed

- The inference DataLoader `num_workers` is now derived from a node-wide TOTAL
  worker budget divided across the data-parallel ranks and the concurrently-live
  loaders (train + validation): `max(1, budget // (world_size * n_live_loaders))`,
  where `budget` is the machine profile's `num_workers` or the CPU core count when
  unset. Each DDP rank builds its own loaders and `persistent_workers` keeps the
  train and validation workers alive simultaneously, so the previous per-rank
  `cores // 2` default silently multiplied to `world_size * 2 * (cores // 2)` live
  worker processes under DDP and exhausted host RAM at production scale (e.g. 4
  ranks x 16 x 2 = 128 processes > 480 GB). The new rule keeps the live total near
  one worker per core on any GPU count and reduces to the prior `cores // 2` per
  loader on a single GPU (single-GPU runs unchanged). The budget is data-loading
  only and inference-only: evaluation and experiment build no `num_workers` loaders,
  and the GPU/shard-worker count (`world_size`) stays bounded separately
  (`SRM_AND_SBI_GPUS`, and the eval/experiment cap at the task/cell count). The
  machine profile's `num_workers` is now interpreted as that node-wide total.

## 0.2.4 - 2026-06-30

Adds a dependency knob to the HPC submitter so a wall-limited training run can be
pre-submitted as a fault-tolerant resurrect chain. No change to the scientific
behavior.

### Added

- `Submit.sh` accepts a `DEP` override that forwards to `sbatch --dependency`
  (e.g. `DEP=afterany:<jobid>`). With `afterany`, each chained continuation starts
  after the previous job *ends regardless of exit status*, so a wall-timeout
  (recorded by Slurm as a failure) does not stall the chain the way `afterok`
  would. Combined with `RESURRECT=1`, this pre-submits a train-to-target chain
  that survives per-job wall stops. The wall-limited-chaining section of the HPC
  README shows the recipe.

## 0.2.3 - 2026-06-30

Exposes the inference `--resurrect` mode through the HPC submission path, so a
wall-limited training run can be continued across successive jobs without leaving
the standard submitter. No change to the scientific behavior or to the inference
algorithm.

### Added

- The HPC inference submitter forwards a `RESURRECT` knob to the Prime entry
  point's `--resurrect` flag. `Inference.sh` reads `RESURRECT` from the
  environment (set `1` to load the existing checkpoint and continue training;
  unset for a fresh run) and appends `--resurrect` to the training command,
  forwarded on both the single-GPU and the `torchrun` data-parallel launches; the
  unified `Submit.sh` lists `RESURRECT` among the inference knobs, so it appears
  in the dry-run preview and the explicit `--export`. The flag already existed on
  the Prime `Inference.py` (and in the validation smoke test); only the HPC
  forwarding was missing, which left wall-limited chaining unreachable from the
  submitter.

### Fixed

- The resurrect and final-model checkpoint reloads now stage through CPU
  (`torch.load(..., map_location="cpu")`) before `load_state_dict` places the
  weights onto each rank's device. The load previously targeted the saving rank's
  recorded device, which under a multi-GPU resurrect transiently concentrated one
  checkpoint copy per rank on a single GPU; CPU staging removes that concentration
  and makes the load independent of the saved device index. Weights are unchanged.

## 0.2.2 - 2026-06-30

Completes the multi-GPU story across the GPU stages: the real-data application
(experiment) stage now shards its work across the allocated GPUs and merges the
per-shard results, matching the data-parallel training and the sharded
evaluation. No change to the scientific behavior or to the single-GPU path.

### Changed

- The experiment stage adapts to the allocated GPUs: with more than one GPU it
  shards its per-condition, per-cell work across one worker per GPU (`torchrun`)
  and a separate merge step combines the per-shard arrays into a single report;
  with one GPU it is the original single-process path. The estimation outputs are
  factored into a shared writer used by both the single-process path and the
  merge, mirroring the evaluation stage. A `--merge` mode and the
  `SRM_AND_SBI_GPUS` cap are added on the experiment entry point, and its HPC
  submitter wraps the `torchrun` launch and the merge.

## 0.2.1 - 2026-06-30

Operational hardening of the HPC workflow, a dry-run-first submission path, and a
documentation and code-hygiene pass on top of the multi-GPU release. The
pipeline's scientific behavior is unchanged.

### Added

- A unified, dry-run-first submission helper for every HPC stage
  (`Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Submit.sh`). It builds the exact
  `sbatch` command — the resolved repository root, the data-file-pattern job name
  with the rendered timing label, and a comma-safe `--export` — and prints it
  without submitting unless `DRYRUN=0` is set, so the recipe, the naming, and the
  configuration cannot be mistyped at submit time. A multi-value condition list is
  carried through the exported environment rather than the comma-split `--export`.
- A `--dry-run` configuration and input preview on the training, evaluation, and
  real-data application entry points: it resolves the machine profile and the
  input paths, reports what would be read and written (flagging anything missing),
  and exits before any GPU use or compute, creating no output directories. The
  dataset-generation orchestrator already offered an equivalent preview.
- An HPC operations runbook (`Script_Bank/HPC/README.md`) consolidating the stage
  and partition map, the submission recipe, the job and log naming convention, the
  hardware layouts to replicate, and the dry-run-first workflow.

### Changed

- The HPC batch scripts resolve the repository root robustly under the scheduler's
  script spooling — an explicit forwarded root, the submit directory, or the script
  location, each validated against the package layout — and fail loud with guidance
  when it cannot be located. Job and log names follow the naming convention shared
  with the theta and video data files.
- Internal hygiene: removed dead code carried over from earlier development (an
  unused regressor head and its configuration fields, an unused sampling helper, and
  unused imports) and renamed an internal estimation function to the package's
  snake_case convention.

### Fixed

- The per-epoch replay loss now uses a dedicated augmentation-disabled loader, so
  the replay measurement excludes the spatial augmentation applied during training
  (the previous in-place toggle did not hold under persistent data-loader workers).
- A run requesting fewer than one epoch is now rejected immediately rather than
  failing later on a checkpoint that was never written.
- Removed private host, login, and machine-profile values from the bundled
  notebook.

## 0.2.0 - 2026-06-29

Multi-GPU support for the compute-heavy inference and evaluation stages, enabling
data-parallel training and sharded evaluation across the GPUs of a single node.
Single-GPU behavior is unchanged: every distributed path collapses to the original
code when one worker is launched.

### Added

- Data-parallel posterior training across one worker per GPU (launched via
  `torchrun`): the TRAIN set is sharded across workers, gradients are synchronized
  each step, and batch-normalization statistics are synchronized across workers.
  Per-epoch model selection on the TEST set is likewise sharded and combined, so the
  selection metric matches the single-GPU computation; checkpoint and posterior
  writing remain on the lead worker.
- Sharded evaluation: MAP-recovery work is partitioned across GPU workers by task,
  then combined into one report by a separate merge step. The recovery metrics are
  aggregates over videos, so the per-worker results merge order-independently.
- A within-epoch progress-cadence control (`--heartbeat`) for finer monitoring of
  the long epochs of a large run.
- An opt-out for cross-worker batch-normalization synchronization
  (`SRM_AND_SBI_NO_SYNC_BN`), off by default.

### Changed

- The HPC submission scripts adapt to the allocated GPUs: more than one selects the
  data-parallel path, one preserves the original single-GPU path. The worker count
  is overridable, and evaluation never launches more workers than it has tasks.

## 0.1.0 - 2026-06-25

First public release of `srm-and-sbi-dimer-alp`: an end-to-end simulation-based
inference pipeline for the DIMER reaction-diffusion model, in which an A monomer
dimerizes into a mobile B dimer and an immobile C dimer. The release provides the
full path from a mechanistic forward model to calibrated parameter posteriors and
their validation against both simulated and real microscopy data.

### Forward model and synthetic imaging

- Particle-resolved reaction-diffusion simulation (RDS) of the DIMER kinetics,
  producing molecular trajectories from the underlying rate and diffusion
  parameters.
- A diffraction-limited imaging (DLI) stage that renders those trajectories into
  synthetic microscopy videos through a point-spread-function convolution and a
  detector noise model (Poisson shot noise and EMCCD readout), so that simulated
  observations match the statistics of the real instrument.

### Posterior inference

- Neural posterior estimation (NPE) with a masked autoregressive flow (MAF)
  density estimator, trained to map an imaging observation to a posterior over the
  DIMER reaction-diffusion parameters.
- A learned observation embedding that couples a 3D convolutional network over the
  spatial-temporal video volume with a temporal transformer, summarizing each video
  into the feature vector consumed by the flow. The embedding accepts variable
  frame counts, so the same network serves recordings of different lengths.

### Data discipline

- A leak-proof three-way data split into physically separate TRAIN, TEST, and EVAL
  namespaces, generated with independent seeds. Gradient updates use TRAIN only,
  per-epoch model selection uses TEST, and final validation uses the held-out EVAL
  set, so that no validation observation is ever seen during training or selection.
- A single dataset-generation command produces all three splits in the correct
  proportions, with a dry-run mode that previews dataset sizing before committing
  compute.

### Validation and application

- MAP-recovery validation on the held-out simulated EVAL set, reporting
  per-parameter recovery accuracy and posterior calibration against known ground
  truth.
- Application of the trained posterior to real microscopy recordings across
  experimental conditions, reporting inferred-parameter distributions where no
  ground truth is available. Both routes write self-contained reports with figures,
  tables, raw arrays, and a tail-able progress log.

### Configuration and infrastructure

- A single duration-parameterized codepath covering both the 2 s and 10 s
  acquisition settings, selected at run time rather than maintained as separate
  code.
- A two-tier storage layout that separates scientific deliverables (validation and
  application reports) from diagnostic dumps (checkpoints, invariant-check logs,
  and debug figures), the latter enabled on demand.
- Machine-profile configuration that externalizes all hardware-specific paths and
  settings, letting the same pipeline run unchanged across workstation, GPU server,
  and HPC environments by selecting a profile rather than editing code.
- Optional fail-loud diagnostics on every pipeline stage: invariant checks (finite
  values, normalized probability matrices, consistent frame counts, finite training
  loss, written outputs) with a pass/fail summary, plus an opt-in detailed report
  for deeper inspection.

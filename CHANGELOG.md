# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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

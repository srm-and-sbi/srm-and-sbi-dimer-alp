# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.2.19 - 2026-07-10

Give the Detector calibration workflow its own committed HPC submission
machinery, and add two additive helpers the later Detector stages build on. The
Detector is a complete workflow parallel to the canonical pipeline; its
submission is now committed and generic (filename-namespaced, coexisting with the
canonical wrappers), never wired into the canonical `Submit.sh` dispatcher.

### Added

- **Committed Detector HPC submission machinery** in `Script_Bank/HPC/`,
  filename-namespaced `SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_*` alongside the
  canonical wrappers (documented in the HPC runbook §8): `..._Simulation.sh`
  (diffusion-only RDS → imaging DLI, packed per node, `--array` fan-out, per-split
  `SEED`), `..._Inference.sh` (data-parallel training via `torchrun`; saves the A5
  estimator), `..._Evaluation.sh` (sharded MAP recovery + a separate `--merge`
  step), and `..._Submit.sh` (dry-run-first dispatcher with the two Goethe GPU
  modes pinned — `gpu_test`=4 GPUs for checks, `gpu`=8 for production). Retires
  the scratch smoke drivers.
- **`detector_parameterization.flag_out_of_bounds`** — flags, never silently
  clips, learnable-parameter values outside the prior box, returning a boolean
  mask and the signed log10 margin.
- **`artifacts.load_estimator_manifest`** — reads a saved estimator artifact's
  manifest (rebuild spec, parameter keys, prior bounds, torch version, weights
  checksum, provenance) without rebuilding the estimator or touching a GPU.

### Changed

- **Detector Evaluation `--pool-mode`** now defaults to the config value
  (`bounded`), matching the canonical Evaluation; both `bounded` and
  `unrestricted` remain one flag away.

## 0.2.18 - 2026-07-10

Make the Detector evaluation multi-GPU sharded, matching the canonical
Evaluation. The initial Detector Evaluation ran single-process; the downstream
stages must mirror the canonical multi-GPU stages they are modeled on, so
evaluation now shards the held-out EVAL tasks round-robin across one worker per
GPU (under `torchrun`) and a separate `--merge` step concatenates the per-shard
arrays into one recovery report — the same shard-then-merge structure as the
canonical Evaluation. Proven on Goethe with the standard multi-GPU setup:
generation on the `test` partition, 4-GPU data-parallel training and a two-way
sharded evaluation + merge on `gpu_test`, all stages clean (the imaging posterior
learned; MAP recovery combined over the held-out EVAL namespace).

### Changed

- **`Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_DETECTOR_Evaluation.py`** — multi-GPU
  sharded recovery mirroring the canonical Evaluation: round-robin `my_tasks` per
  worker, per-shard `.npz` writes, and a `--merge` combine step (single process,
  no GPU) that concatenates the shards into the report and removes them. The
  single-worker path is unchanged (writes the report directly). A worker assigned
  no tasks writes no shard.

## 0.2.17 - 2026-07-10

Add the Detector calibration workflow: a special-situation set of entry points
(structured like Construction — outside the `Submit.sh` dispatcher and the four
canonical stage wrappers) that calibrates the diffraction-limited-imaging model
by inferring it with the physics frozen to pure diffusion, so the imaging
parameters are justified and reproducible for review rather than hand-tuned. The
imaging parameters become the inference target and the reaction-diffusion
parameters are marginalized as a nuisance. Detector data namespaces separately by
a `_DETECTOR` prefix, so it can never overwrite canonical artifacts. Validated
end-to-end on a 2 s, 16/4/2×10, 5-epoch smoke on the GPU server (RDS → DLI →
Inference → Evaluation, all clean; the imaging posterior learned; MAP recovery on
the held-out EVAL namespace). The Experiment stage (MAP on real videos + the
real-vs-synthetic MMD/C2ST gap) is deferred to a follow-up.

### Added

- **Detector parameter scheme + machinery.** `detector_parameterization.py` — a
  value-based role table (imaging parameters learnable; diffusion + counts
  nuisance-from-spec; geometry, brightness quantiles, `delta_frame`,
  `numb_photo_bleach`, `dimer_mule` fixed), with a sentinel-based role resolver,
  learnable + nuisance prior builders, and the `_DETECTOR` path alias.
  `nuisance.py` — a samplable, self-describing `Nuisance` artifact (parameter-key
  manifest; never-silent clipping; stored distribution numerics). `artifacts.py`
  — a self-describing, torch-version-portable estimator format (compile-stripped
  state_dict + rebuild spec + metadata; eager-rebuild loader) that sidesteps the
  `torch.compile` pickle lock of the canonical posterior.
- **Adapted forward models.** `detector_simulation_rds_support.py` (diffusion-only
  RDS drawn from the nuisance) and `detector_simulation_dli_support.py` (imaging
  drawn from θ), each reusing the canonical building blocks by import.
- **Detector entry scripts** (`Script_Bank/Prime`, `_DETECTOR`-namespaced):
  `Simulation_RDS`, `Simulation_DLI`, `Inference` (saves the A5 estimator),
  `Evaluation` (imaging-θ MAP recovery on EVAL).

### Changed

- Three small, generic, behavior-preserving, default-canonical injections into
  shared machinery so the Detector can reuse it: `build_system(pure_diffusion=…)`
  in `simulation_rds_support.py` (default `False` = the previous reactive path);
  an optional `paths=` on `VideoDataset` / `build_datasets` / `setup_training` in
  `inference_support.py`; and an optional `paths=` / `data_bank_root=` on
  `console_log_context` in `utils.py` (so a Detector `--debug-dump` transcript is
  `_DETECTOR`-tagged). Each defaults to the canonical configuration, so existing
  stages are byte-identical.
- `PROJECT_CONTEXT.md` §3: Stage 1 (detector parameters) reclassified from a
  separate future sibling to this repository's in-repo special-situation
  calibration workflow.

## 0.2.16 - 2026-07-10

Add optional, flag-gated instrumentation that captures the per-example test-loss
distribution at the best epoch, so estimator generalization can be studied
beyond the single mean test-loss scalar. Each stored example is keyed by its
`(task_index, sim_index)` pair and carries the associated theta, with a
self-describing manifest (the full parameter table). The three best-epoch
artifacts — posterior, optimum-ANN checkpoint, and test-loss distribution —
share one store/backup lifecycle. The training metric (`epoch_test`) is
unchanged; `--no-test-loss-distribution` reproduces the prior behavior exactly.

### Added

- **Best-epoch test-loss distribution** (`--test-loss-distribution`, default on).
  New module `test_loss_distribution.py` (pair-keyed per-example loss + theta +
  manifest; `.npz` serialization; per-epoch summary and a new-best extended
  statistics card). `parameterization.py` gains the canonical and
  provenance-backup path helpers (`test_loss_distribution_path`,
  `backup_test_loss_distribution_path`) mirroring the posterior/checkpoint
  naming. `inference_support.py` collects the per-example losses in one pass
  (test loader only) with a distributed all-gather and `(task, sim)` dedup, plus
  a new-best commit hook. The Inference entry point
  (`SRM_AND_SBI_DIMER_ALP_Inference.py`) commits all three artifacts at each new
  best (with `Epoch_{current}` backups) and writes an `Epoch_{total}` backup at
  finish.

## 0.2.15 - 2026-07-07

Add a second, tighter recovery tolerance band to the evaluation report. The
existing band is +/-0.3 in log10 units, which is a symmetric factor-of-2 band
(0.3 ~= log10(2)); the new band is +/-0.15, a symmetric factor-of-sqrt(2) band
(0.15 ~= log10(sqrt(2)), i.e. half of 0.3). Nesting the two bands shows, at a
glance, what fraction of MAP estimates land within a factor of 2 and within a
factor of ~1.41 of the truth. This is a reporting/plotting change only: no
inference, evaluation, or experiment computation is affected, so figures and
tables can be regenerated from the existing recovery arrays without recompute.

### Added

- **Tighter factor-sqrt(2) recovery band** across the evaluation report path.
  `parameterization.py` (`InferenceEvaluation`) gains `error_guide_tight = 0.15`
  alongside the existing `error_guide = 0.3`, both documented with their
  factor-2 / factor-sqrt(2) derivation. `evaluation.py` (`recovery_stats`,
  `recovery_table`) computes and tabulates `frac_within_guide_tight` as a new
  `within +/-0.15` column. `visualization_inference.py` (`_draw_error_axis`,
  `figure_recovery_combined`) draws the nested band as a second horizontal
  guide line labeled "factor sqrt(2)". The Evaluation entry point
  (`SRM_AND_SBI_DIMER_ALP_Evaluation.py`) reads `error_guide_tight` from config
  and threads it through the table and figure. Docstrings state that 0.3 and
  0.15 come from log10(2) and log10(sqrt(2)).

## 0.2.14 - 2026-07-06

Make the embedding network duration-general by bounding its temporal length. Long
videos are reduced toward a configurable target frame count before the temporal
transformer, so the first-conv activations (the memory bottleneck, linear in the
number of frames) stay bounded for any recording length. The reduction is a
learnable strided convolution folded into the first conv block, and it is a no-op
for videos at or below the target, so short videos and the 2 s baseline stay
bit-identical to the un-reduced network.

### Added

- **Temporal reduction in the CNN backbone** (`inference_network.py`,
  `Complex3DCNN`, new `temporal_target_frames` argument). A video of `n_frames`
  frames is reduced by an integer factor `s = n_frames // temporal_target_frames`,
  applied as the temporal stride of the first conv block, with that block's
  temporal kernel widened to `max(3, s)` so `kernel >= stride` and no input frame
  is skipped (learnable temporal pooling, not decimation). The reduced length is
  computed and asserted at construction. `forward()` is unchanged: the reduction
  lives inside the existing conv stack.
- **`temporal_target_frames` network-config field** (`parameterization.py`,
  `InferenceNetwork`), default 100 frames, `None` to disable. Documented in
  frames with its dependence on duration and frame rate
  (`n_frames = duration_seconds * frame_rate`; 100 frames is 2 s at 50 FPS, 1 s at
  100 FPS, or 4 s at 25 FPS). Threaded into the Inference and Construction build
  sites and reported in the Inference run banner.

### Changed

- Videos longer than `temporal_target_frames` (e.g. 5 s = 250 frames at 50 FPS)
  now train and infer at a bounded temporal length instead of running the CNN over
  every frame. Videos at or below the target (1 s, 2 s at 50 FPS) are unchanged.
  Evaluation and Experiment inherit the reduced network automatically, since they
  unpickle the trained posterior (which carries the embedding net).
- Consequence: the first conv's temporal behavior now depends on `(n_frames,
  temporal_target_frames)`, so a reduced long-video network differs from an
  un-reduced one of the same nominal duration (for 10 s+ the first-conv kernel also
  changes shape). This affects only long-video models, which had no memory-viable
  un-reduced baseline to resume from; the 2 s baseline and its checkpoints are
  unaffected.

## 0.2.13 - 2026-07-06

Restore the embedding-network hyperparameters to the reference configuration: a
deeper CNN backbone and a larger temporal transformer, raising the CLS/flow
conditioning embedding from 64 to 128. Profiling showed the transformer attention
is not the memory constraint (the CNN activations are), so there is no memory
reason to keep the network shrunk relative to the reference.

### Changed

- Embedding-network defaults (`parameterization.py`): `n_conv_layers` 4 -> 5
  (embedding dimension `start_channels * 2^(n_conv_layers-1)` = 64 -> 128),
  `n_attn_layers` 1 -> 2, `attention_heads` 2 -> 4. `start_channels` (8) and the
  spatial-only pooling are unchanged.
- Consequence: checkpoints trained under the previous 64/1/2 configuration are not
  load-compatible with the new 128/2/4 network, so `--resurrect` cannot resume them.
  They remain usable as standalone artifacts for downstream sampling.

## 0.2.12 - 2026-07-05

Fail-fast guard on non-finite training loss, with a trip-only diagnostic breadcrumb.
Additive and behavior-neutral while losses are finite: it changes behavior only when a
training loss becomes NaN or Inf, where it now aborts cleanly (before backward/step)
instead of continuing to train on the non-finite value.

### Added

- **Non-finite training-loss guard** (`inference_support.py`, `train_loop`). Each
  training batch checks its loss; on a non-finite value it aborts with a clear
  `[FINITE-GUARD]` message (epoch, batch, rank) before `backward()`/`step()`, so the
  NaN cannot propagate into the optimizer or drive a downstream out-of-bounds GPU
  access. `--resurrect` resumes from the last checkpoint on the next submission. The
  check reuses the per-batch `loss.item()` sync, so the finite path is unchanged.
- **Trip-only diagnostic breadcrumb** (`_diagnose_nonfinite_loss`). On the failing
  batch only, it logs whether any model parameter is already non-finite (weights
  diverged vs. a non-finite forward on finite weights) and the finite-status and range
  of the input `video_batch` / `theta_batch` (a corrupt input). It runs only at the
  failure, so it adds nothing to normal training.

## 0.2.11 - 2026-07-02

Automatic provenance backups for the trained artifacts, and a Construction path
that rebuilds a posterior from any checkpoint backup. Also makes the Experiment
launcher's chunk-step default duration-general, and adds the CD86 / CTLA-4
control-receptor analysis scripts. Additive and backward compatible: the canonical
outputs and the default Construction behavior are unchanged.

### Added

- **Automatic artifact backups.** A finished Inference run that loaded a TEST set
  (`--test-tasks > 0`) now writes, alongside the canonical checkpoint
  (`Labor/…_Optimum_ANN.pth`) and posterior (`Posit/…_Posterior.pkl`), a
  provenance-named copy of each:
  `…_TRAIN+TEST_<train>+<test>_Epoch_<n>_TEST_LOSS_<loss>.<ext>`. The suffix records
  the TRAIN/TEST video counts (as thousands-tokens, `50000` → `50K`), the epochs the
  job ran, and the checkpoint's best TEST loss (exactly two decimals; explicit `+`
  on a positive value, no sign when it rounds to `0.00`). The bare `state_dict` and
  the posterior pickle carry no such metadata, so the name is the only record of a
  model's training scale and result. Canonical names are untouched — a backup is a
  copy and is never loaded as the active artifact. A `--test-tasks 0` run has no
  selection loss and writes the canonical pair only.
- **Construction from a specific checkpoint.**
  `Script_Bank/Prime/SRM_AND_SBI_DIMER_ALP_Construction_Optimum_ANN.py` gained
  `--checkpoint` and `--posterior`: point `--checkpoint` at a backup and the matching
  backup posterior is derived (`Optimum_ANN` → `Posterior`, `.pth` → `.pkl`) and
  written, so an archived checkpoint can be rebuilt into its posterior without
  retraining. With no flags the behavior is unchanged (canonical → canonical).
- `Paths` (`parameterization.py`) gained the pure name-derivation helpers
  `format_backup_loss`, `format_backup_size`, `backup_descriptor`,
  `backup_checkpoint_path`, `backup_posterior_path`, and
  `posterior_path_for_checkpoint`.
- **CD86 / CTLA-4 control-receptor analysis** (`Script_Bank/Analysis/`,
  special-scope reuse — not canonical pipeline stages). `…_Experiment_CD86_CTLA-4_Controls.py`
  is a near-verbatim clone of the MET Experiment stage that applies the MET-trained
  posterior (no retraining) to two control receptors of known oligomeric state —
  CD86 (monomer) and CTLA-4 (dimer) — reading their own dataset folder and writing
  their own output directory, so the canonical MET Experiment and its outputs are
  never touched. `…_Controls_Temporal_Dynamics.py` (with companion `.md`) analyzes
  the temporal dynamics of the inferred parameters with a mobile-diffusion headline
  readout, with the two receptors bracketing the monomer/dimer diffusion scale.

### Changed

- `train_loop` (`inference_support.py`) now returns a fourth value,
  `optimum_loss_test` — the checkpoint's best TEST loss, resurrect-baseline-aware —
  which names the automatic backup. Its only caller, the Inference entry point, is
  updated; no other behavior changes.
- The Experiment launcher (`Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Experiment.sh`)
  now leaves `CHUNK_STEP` unset by default, so the Experiment entry point tiles at
  the integer model window (non-overlapping) for any `--total-time-seconds`, rather
  than a hardcoded 2 s literal that only evenly divides a 2 s window. Set `CHUNK_STEP`
  to force overlapping chunks (e.g. `1` for a 1 s stride).

### Documentation

- Rewrote the HPC runbook §7 (*Artifact backups*): documents the automatic
  provenance scheme as primary and retains the manual dated-tag convention
  (`_<TAG>_<DD.MM.YYYY>`) for ad-hoc keeps.
- Extended the HPC runbook §6 (*Construct a posterior from a checkpoint*) with the
  `--checkpoint` / `--posterior` derivation and the backup-rebuild use case.
- Expanded the PROJECT_CONTEXT §3 output note to cover the canonical + auto-backup +
  rebuild story, and noted in the VALIDATION Inference smoke test when a backup is
  written.

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

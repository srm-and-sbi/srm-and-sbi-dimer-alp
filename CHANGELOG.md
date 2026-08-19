# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## 0.4.15 - 2026-08-19

Correctness, calibration-report, and consistency release: video-granularity sharding for the
sharded GPU stages, a substantially more interpretable posterior-calibration report, repaired
statistical machinery, three new workflow-general analyses, and a documentation sweep that
realigns every major document with the implementation.

### Added

- **Embedding-space distance for both workflows.** The measure module is now the
  workflow-agnostic `embedding_space_distance.py` (kernel; MMD + classifier two-sample test,
  recording-blocked) with a shared `embedding_space_distance_runner.py` and two shims
  (`SRM_AND_SBI_DIMER_ALP_Embedding_Space_Distance.py` + `…_DETECTOR_…`). The companion note
  documents how the reading INVERTS between workflows (detector: imaging realism; biology:
  synthetic-prior reach with imaging frozen) and the known limitations of the significance
  machinery (imbalanced-accuracy C2ST null; block-count-preserving MMD permutation — both
  conservative for the gap verdict, neither valid for exact p-values). New HPC wrapper
  `SRM_AND_SBI_DIMER_ALP_HPC_Embedding_Space_Distance.sh` (WORKFLOW=biology|detector;
  single-GPU engine on a whole-node allocation; absolute script paths).
- **Posterior-predictive video for both workflows.** Shared engine
  `posterior_predictive_video_runner.py` + two shims: render a synthetic video from the
  parameters inferred for one experimental recording and compare side by side. The MAP block
  inverts per workflow (biology: ten reaction-diffusion parameters, full reactive system,
  imaging held at the calibrated Nuisance_DLI vector read at run time; detector: six imaging
  parameters, diffusion-only system, RDS nuisance drawn or pinned); the five SCOPE camera
  parameters are pinned to their MET values in both. `--map-source cell-sgm` (default) renders
  a real chunk's estimate selected by the Sample Geometric Median, never a per-dimension
  composite.
- **Sample Geometric Median analysis for the biology workflow.** Workflow-agnostic kernel
  `sample_geometric_median.py` + shared `sample_geometric_median_runner.py` + biology shim
  (`SRM_AND_SBI_DIMER_ALP_Experiment_Sample_Geometric_Median.py`); the detector Nuisance_DLI
  tool now delegates to the same kernel (one SGM implementation repo-wide). Condition-restricted
  runs (`--condition MET-FAB|MET-INLB`) write to their own directories.
- **Fleet synchronization utility** `SRM_AND_SBI_DIMER_ALP_Fleet_Sync.sh`: the single supported
  repo-propagation path (dry-run default, per-directory recursion so deletions propagate,
  secrets and machine-local files excluded by name, post-sync verification).
- Shared `workflow.parameter_keys()` / `workflow.parameter_table()` resolvers for the two
  parameterization modules' deliberately different symbol names.
- `DiagnosticReporter` always stamps the report's UTC run time itself; a caller remark goes in
  the new `run_note` field and can no longer displace the timestamp.

### Changed

- **Video-granularity sharding** in Evaluation and Posterior_Calibration: individual
  `(task, sim)` videos are dealt round-robin across ranks (`shard_by_rank` + per-task
  `groupby`), so any task count balances over any worker count to within one video; the
  task-count-based GPU caps in the wrappers are removed (they would have throttled the new
  sharding), and all eight GPU wrappers raise torch-elastic's exit barrier
  (`TORCHELASTIC_EXIT_BARRIER_TIMEOUT`, override `EXIT_BARRIER=`) so straggler ranks are not
  killed at the 300 s default.
- **Posterior-calibration report**: effect sizes lead every table (KS D, coverage max-gap,
  |ATC|, reject fraction) with p-values demoted to detectability statements (a printed
  0.00e+00 is double-precision underflow, not a verdict); per-parameter location-versus-width
  Diagnosis table (bias z, spread z, sharpness, physical bias); a "What this implies" reading;
  the 1-D/2-D marginal-calibration ladder with `dependence_excess`; the stratified section
  digested to one row per parameter-test with a profile-shape classification (rises / falls /
  worst-at-ends / worst-in-middle / flat) over per-bin figures; stratum loop parallelized with
  a power-of-two worker cascade; defaults `--n-strata=10 --num-bins=50 --lc2st-n-eval=1000`.
- **L-C2ST repaired**: the classifier is now cross-validated (`num_folds=5`, so observations
  are scored by folds that did not train on them) and each evaluated observation uses its full
  stored posterior sample cloud instead of a single draw. Validated on calibrated and
  deliberately miscalibrated toys.
- Experiment-stage pool-mode guidance corrected repo-wide: pool mode follows the data —
  `bounded` for prior-drawn synthetic sets, `unrestricted` for experimental recordings.
- The posterior-predictive `.npz` provenance field is `rds_provenance` (it holds the MAP for
  biology and a nuisance draw for the detector); the viewer notebook reads the shared fields
  and serves both engines.

### Fixed

- `sgm_in_box` is computed against the prior box instead of hardcoded `True`; the regenerated
  SGM reports now correctly mark the unrestricted pooled and MET-INLB vectors as out-of-box.
- The seven load-bearing parameter-table validations raise `ValueError` instead of `assert`
  (they now survive `python -O`).
- Five trusted-artifact `np.load` calls no longer pass `allow_pickle=True` (the artifacts
  contain no object arrays).
- `--display-norm` default restored to `full` with explicit choices (the undocumented
  `per-frame` sentinel silently routed to autoscale).
- Reader-facing condition names are MET-FAB / MET-INLB across CLIs, reports, figures, and
  output paths; ALP/BET survive only as stored schema and filename tokens, translated at the
  boundary.
- `tomli` declared as a conditional dependency for Python < 3.11; the deliberately sparse
  dependency list is now explained in `pyproject.toml` itself.
- Documentation realigned with the implementation across README (iteration tag, synthetic-
  recovery wording, structure listing, experimental-versus-synthetic terminology),
  PROJECT_CONTEXT (C-species diffusivity, reversible B ↔ C, the ten-parameter θ, SCOPE
  marginalization, module list), VALIDATION (split-suffixed artifact paths, pool-mode scoping,
  six-leg smoke criteria, dispatcher-only dry-run wording, a new assurance roadmap),
  DETECTOR_WORKFLOW (implemented mirror analysis, wrapper inventory, experimental-data-reuse
  statement), the HPC guide (analysis wrappers, Goethe GPU recipe, EXIT_BARRIER/LR knobs,
  fleet sync), and the analysis companions (flags matched to their argparse definitions).

## 0.4.14 - 2026-08-14

Three workflow-general Analysis diagnostics, each built once as a workflow-agnostic kernel +
shared runner + two namespaced shims (the shared-engine pattern the pipeline stages use), so
one implementation serves both the biology and the detector workflows.

### Added

- **Posterior-calibration diagnostic** (`SRM_AND_SBI_DIMER_ALP_Posterior_Calibration.py`
  + `…_DETECTOR_Posterior_Calibration.py`). Scores a trained posterior's calibration on
  the held-out EVAL set with simulation-based calibration (Talts et al. 2018), expected
  coverage (Deistler/Hermans 2022), TARP (Lemos et al. 2023), and local C2ST (Linhart et
  al. 2023) — overall and stratified along each target parameter by the posterior's
  inferred value (a function of the observation, so a calibrated posterior is not falsely
  flagged by Bayesian shrinkage), localizing a subregion where calibration degrades rather
  than averaging it away. New package modules:
  `posterior_calibration.py` (workflow-agnostic kernel wrapping `sbi.diagnostics`, on
  pre-drawn theta-space arrays + embeddings; the raw videos never leave the stream),
  `posterior_calibration_runner.py` (streams EVAL, draws per-video samples/log-densities/
  embeddings, multi-GPU sharded with `--merge`), and `visualization_calibration.py`. An
  Analysis diagnostic — read-only, never wired into the `Submit.sh` dispatcher — built as
  two namespaced shims over one shared engine, the same pattern as the pipeline stages, so
  one implementation serves both workflows. Validated on a synthetic Gaussian toy: all four
  measures separate a calibrated posterior from a deliberately overconfident one.

- **Estimator-comparison diagnostic** (`SRM_AND_SBI_DIMER_ALP_Estimator_Comparison.py`
  + `…_DETECTOR_Estimator_Comparison.py`). Decides whether one trained estimator generalizes
  better than another by the paired log-score on the shared `(task, sim)` TEST subset —
  pairing cancels each video's intrinsic entropy floor, so the difference isolates the two
  estimators' KL gap — reported via the Diebold-Mariano statistic (Diebold & Mariano 1995)
  with the Wilcoxon signed-rank test and a paired bootstrap as heavy-tail-robust companions.
  New package modules: `estimator_comparison.py` (workflow-agnostic kernel, numpy + scipy)
  and `estimator_comparison_runner.py` (reads two `TestLossDistribution` artifacts, no GPU).
  An Analysis diagnostic — read-only, never dispatched — as two namespaced shims over one
  shared engine. Validated end to end on the real biology checkpoints: −7.12 significantly
  beats −6.91 (mean improvement +0.21, DM/Wilcoxon p≈0, bootstrap CI clear of zero).

### Changed

- **Test-loss-distribution analysis generalized to both workflows.** The previously
  detector-only `SRM_AND_SBI_DIMER_ALP_Test_Loss_Distribution_Analysis.py` is retrofitted onto
  the shared-engine two-shim pattern: its (already workflow-agnostic) analysis + figures move to
  a new kernel `test_loss_analysis.py` + runner `test_loss_analysis_runner.py`; the canonical
  artifact is now resolved through the workflow's `cfg.paths` (the unqualified shim → biology, a
  new `…_DETECTOR_…` shim → detector), with the ad-hoc `--tld-path` mode preserved. Extended
  with a `hardest_regime` summary (the least-identifiable parameter + which end of its range) and
  a cross-reference to `Posterior_Calibration` for the hard tail's honesty. Validated on the real
  biology TLD: the three species counts dominate the hard tail (KS D 0.23–0.47, all harder at low
  count) — confirming the low-count identifiability limit.

## 0.4.13 - 2026-08-13

Scale the GPU HPC stages **across nodes**. Every GPU launcher now takes a `NODES=N`
allocation and runs data-parallel (Inference) or work-sharded (Evaluation, Experiment,
Nuisance_DLI) across all `world_size = NODES × GPUs-per-node` ranks; one node and one GPU
behave exactly as before. No Python-engine change — the runners already shard and normalize
by `world_size` (`resolve_topology` reads `SLURM_NNODES` / `SLURM_NTASKS`); this release is
launcher + docs only. Validated on JUPITER: near-linear scaling (2.0× at 2 nodes, 3.7× at 4).

### Added

- **Multi-node branch in every GPU wrapper.** With `>1` node, `srun` places one `torchrun`
  launcher per node, each spawning one process per GPU, bound by a c10d rendezvous — so
  `--nodes=N --gres=gpu:G` trains/shards across `N×G` ranks. The single-node
  `torchrun --standalone` path (`>1` GPU) and the single-GPU `python` path are byte-for-byte
  unchanged. Applies to biology + detector `…_HPC_Inference.sh` (DDP), and
  `…_HPC_Evaluation.sh`, `…_HPC_Experiment.sh`, `…_DETECTOR_HPC_Nuisance_DLI.sh`
  (shard-then-`--merge`; the rendezvous is unused there, kept only for one uniform
  GPU-binding path).
- **`NODES` knob** on both `…_HPC_Submit.sh` dispatchers → `sbatch --nodes` for the GPU
  stages (biology inference already had it; extended to biology evaluation/experiment and all
  three detector GPU stages). `--gres` is per node, so `NODES=2 GRES=gpu:4` → `world_size` 8.

### Context

- The batch rule keys off `world_size`, not node count: `SyncBatchNorm` and the loss are
  all-reduced across every rank on every node, so `--batch-size` stays per-rank and the
  effective batch is `batch × world_size` whether the ranks sit on one node or several.
- The sharded stages (Evaluation/Experiment/Nuisance_DLI) are embarrassingly parallel: each
  rank writes its own `_shard_<r>_of_<n>.npz` and a single-process `--merge` combines them.
  Multi-node needs no new shard/merge logic — an idle rank (world_size > shard count) writes
  no shard and the merge tolerates it — but it does require the output directory on a shared
  filesystem so the merge sees every node's shards (the case on the HPC data banks).
- Documentation updated across the wrapper headers, both `Submit.sh` headers, the HPC runbook
  (`Script_Bank/HPC/README.md`), the root `README.md`, `PROJECT_CONTEXT.md`, `VALIDATION.md`
  (§2.6/§2.7), and `DETECTOR_WORKFLOW.md` (B3–B5).

## 0.4.12 - 2026-08-12

Expose the ReaDDy neighbor-list (Verlet) **skin** as a documented, diameter-relative performance knob.
The skin coarsens ReaDDy's cell-linked-list grid in the large, dilute imaging box; it is a pure
performance parameter and does **not** change the physics — reactions still fire at the true reaction
radius. Measured ~13× RDS speedup with statistically identical output (verified at the prior extremes,
including max diffusivity with fusion-on-contact). No prior, artifact, or data-schema change;
regeneration is not required.

### Added

- **`SimulationRDS.neighbor_list_skin_factor` (default `10.0`).** The neighbor-list skin expressed as a
  MULTIPLE of `particle_diameter_nm` (skin = factor × diameter; `10×` = 100 nm). The default sits on the
  broad fast plateau and clears the worst-case per-step displacement (~47 nm at max diffusivity) with
  margin. The field carries the full rationale and the U-shaped cost curve (too small → a ~16-million
  mostly-empty-cell sweep; too large → the box collapses toward ~1 cell and the candidate search
  degrades to O(N²)).
- **`build_simulation(..., skin_factor=None)`** applies the skin (`smut.skin = skin_factor ×
  particle_diameter_nm`), defaulting to the config value; shared by both workflows — the detector twin
  `build_detector_rds_simulation` forwards it. `None` → config default; negative values are rejected.
- **`--skin-factor` on the RDS entry points** (via the shared `build_rds_parser`, so both the biology and
  detector `Simulation_RDS.py` twins) and on `Generate_Datasets.py` (RDS stage only). The effective skin
  in nm is echoed in the RDS run banner.
- **`SKIN_FACTOR` batch knob** on the biology and detector `…_HPC_Simulation.sh` launchers (passed to the
  RDS entry point only, never DLI), the `…_HPC_Submit.sh` front end (`simulation` stage), and the
  `…_HPC_Generate_Controller.sh` campaign controller. Unset → the code default.

### Context

- Root cause of the RDS slowdown: the reaction radius (contact distance = 1 diameter = 10 nm) sets
  ReaDDy's cell size, which in the ~40 µm dilute imaging box forces a >99.99%-empty ~16-million-cell grid
  whose per-step management — not the physics — dominates runtime. The skin decouples the cell size from
  the reaction radius, recovering the speed without altering the simulation.

### Documentation

- **`VALIDATION.md` §2 — smoke recipe synchronized across both workflows.** One reference
  configuration (the detector §2.5 sizing) with an explicit **Run order** — detector workflow → build
  `Nuisance_DLI` → biology workflow — that reproduces the production dependency instead of reordering
  it for convenience. A new **§2.5b** documents building the `Nuisance_DLI`: production `raw` (from the
  trained detector estimator) versus a smoke `box_user` uniform over the imaging prior box (standalone,
  CPU-only — no estimator, GPU, or recordings), with the prior ranges pinned to
  `detector_parameterization.py` (`DETECTOR_PARAMETERIZATION`) as the authoritative source. The
  cross-workflow prerequisite, the `--video-dtype-bits` / `--batch-size` flags, the DLI nuisance-record
  outputs, and a biology acceptance block were harmonized; the genuine per-workflow difference (only
  biology consumes the artifact) is preserved.

## 0.4.11 - 2026-08-11

Design note for a planned matched-imaging embedding validation, and a stale cross-reference fix.

### Added

- **`PROJECT_CONTEXT.md` §7 — matched-imaging embedding validation (planned).** Documents the future
  read-only analysis that renders synthetic videos at the inferred imaging with receptor counts and
  diffusion pinned to the experimental data — counts from the localization density corrected for the
  emitter on-fraction, diffusion from tracked trajectories by mean-squared displacement — then reuses
  the embedding-distance two-sample machinery (MMD / C2ST) against the experimental recordings,
  isolating imaging realism at the calibrated operating point rather than the prior-averaged realism
  the prior-spanning reference reports.

### Fixed

- **`PROJECT_CONTEXT.md` §7 — cross-reference.** Corrected "real-versus-synthetic" to
  "experimental-versus-synthetic", matching the `DETECTOR_WORKFLOW.md` §8 heading and the standard
  terminology. Prose only.

## 0.4.10 - 2026-08-11

Documentation consistency fix in the Detector workflow reference.

### Fixed

- **`DETECTOR_WORKFLOW.md` §6.3 — `gamma` prior corrected.** The prose stated the `gamma` prior as
  `(1.59, 1.64)`; corrected to `(1.62, 1.625)` to match the authoritative §6.2 camera table (log10
  `(1.62, 1.625)` → 41.7–42.2 ADU/e⁻) and `detector_parameterization.py`. Prose only — `gamma` is a
  marginalized camera nuisance, so no calibration result changes.

## 0.4.9 - 2026-08-11

The Posterior-Predictive Video **viewer notebook** now shares one color limit across the experimental
and synthetic panels in every mode, matching the 0.4.8 engine fix, so the scrubber and player render a
frame identically to the static figure.

### Changed

- **PPV notebook (`clim`) shares the limit in all modes.** The scrubber and player put both panels on
  ONE window per frame — `full` the shared full-range `[min, max]`, `percentile` the shared
  `[min, p99.99]`, `autoscale` the two displayed frames' shared min/max (was per-panel). The cells and
  the intro/playback prose were updated to state the guarantee.
- **`.gitignore`:** ignore stray `Untitled*.ipynb` scratch notebooks.

## 0.4.8 - 2026-08-11

The Posterior-Predictive Video comparison figure now gives the experimental and synthetic image
panels ONE shared color limit in **every** `--display-norm` mode, so identical intensities always map
to identical colors and the side-by-side comparison is fair regardless of mode.

### Changed

- **`--display-norm` — shared image-panel limit in all modes.** Previously only `full` (the default)
  put the experimental and synthetic panels on one shared limit; `percentile` and `autoscale` scaled
  each panel to its own data. Now all three share: the max-projection row shares the full `[min, max]`,
  and the mid-frame row shares a mode-selected window — `full` the whole-clip `[min, max]`, `percentile`
  the whole-clip `[min, p99.99]`, `autoscale` the two displayed frames' combined min/max (so `autoscale`
  changes meaning from per-panel to shared). The function docstring and `--help` state the guarantee.
  Figure-only; the persisted pixels are untouched.

## 0.4.7 - 2026-08-11

The Nuisance_DLI build gains an **`sgm_percentiles`** choice that *mints* the imaging artifact from a
central selection — a single **frozen** vector (the Sample Geometric Median), or a small,
correlation-preserving **marginalization pool** — moving the selection out of the report-only analysis
tool into the build itself. It **reuses the already-computed Detector Experiment MAPs** (or the labeled
posterior pool via its per-window SGM), so it runs on the **CPU** with no GPU rebuild.

### Added

- **`sgm_percentiles` `posterior_sample_pool_choice`.** Selects whole real MAP vectors at percentiles
  along a **signed distance-to-SGM** coordinate: the magnitude is the distance to the Sample Geometric
  Median `g` (the correlation-preserving median vector); the sign is the side of `g` along the cloud's
  main variation axis (PC1 of the `g`-centered points, oriented toward brighter `mu_pc`). So `p50` is
  `g` exactly, `p<50` walks the dim/narrow side (to `p=0`), `p>50` the bright/wide side (to `p=100`).
  `percentiles = [50]` freezes the imaging to the SGM; a list such as `[5,25,50,75,95]` gives a small
  marginalization pool of real acquisitions with correlations intact. Spec fields: `percentiles`,
  `condition` (`pooled`/`ALP`/`BET`), `selection_source` (`experiment` | `window-sgm`). CPU-only.
- **Shared helpers in `detector_nuisance_dli`** — `sample_geometric_median` (now shared with the SGM
  analysis tool), `select_signed_percentile_vectors`, `load_map_vectors`, `_per_window_sgm_labeled`.

### Changed

- **The SGM analysis tool uses the shared `ndli.sample_geometric_median`** (deduplicated; behavior
  unchanged — it still reproduces the same values).
- **Docs.** `DETECTOR_WORKFLOW.md` §7 (the six-choice table + the signed-percentile construction) and
  the `Nuisance_DLI` / `Sample_Geometric_Median` companion notes document `sgm_percentiles`.

## 0.4.6 - 2026-08-11

The Detector's calibrated-imaging pool is now **self-describing**, and the Sample Geometric Median
analysis tool reads that structure. The posterior-sample / MAP pool cache previously stored only a
flat `(N, D)` matrix and a global provenance dict, so its window→condition mapping was purely
positional -- an unstored artifact of the build's `(kind, cell)` enumeration and round-robin sharding
-- and could not be filtered from the file alone. The pool now persists, per row, the condition
(`kind_index` into `kinds`) and the time-window position (`cell`, `chunk`), stamped with a
`pool_format_version`, so a consumer can restrict it by condition or acquisition directly. The SGM tool
consumes those labels, sources the real optimized MAP estimates rather than a silent stand-in, and
offers the per-window SGM of the draws as an explicit named option.

### Added

- **`SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI_Sample_Geometric_Median.py`** (+ companion `.md`) — a
  post-hoc, CPU-only analysis that reduces the calibrated imaging pool to its **Sample Geometric Median**
  (the correlation-preserving median *vector* -- an actual pool member -- Ramirez Sierra & Sokolowski,
  *Mach. Learn.: Sci. Technol.* 2025), computed in absolute space normalized by the absolute prior range,
  and contrasts it with the per-dimension vector of medians for the full pool and the in-box subcollection,
  with the out-of-prior mass, the joint correlation matrix, and deterministic figures.
- **Self-describing pool cache.** `detector_nuisance_dli.save_pool(..., labels=)` persists per-row
  `kind_index` / `cell` / `chunk` + the `kinds` name array + `pool_format_version` (1);
  `load_pool_labels()` reads them back (or `None` for a legacy pool). `build_posterior_sample_pool` /
  `build_map_estimate_pool` now return `(pool, labels)`, and the Nuisance_DLI build threads each
  chunk's `(kind, cell, chunk)` identity from `_read_all_chunks` through both the single-process and
  the sharded paths (the labels travel with their rows through the round-robin shard/merge).
- **`--migrate-pool-labels`** — a CPU-only maintenance mode on the Nuisance_DLI analysis script that
  adds the per-row labels to an existing (legacy) posterior-sample pool by borrowing the aligned labels
  from the Detector Experiment MAP output, after verifying the two share a window ordering (the
  index-aligned distance must strongly beat a random shuffle). Backs up the original; preserves the
  pool provenance verbatim so its cache freshness (and no-GPU reuse by `--build`) is unaffected; idempotent.
- **SGM tool `--condition {pooled, ALP, BET}`** — restrict the collection to one experimental condition
  (ALP = MET-FAB monomer control, BET = MET-INLB dimer) via the pool's own per-row labels before the
  summary; a real condition on an unlabeled collection is a loud error, not a silent full-pool summary.
  **`--map-source {experiment, window-sgm}`** — for `--collection map`, choose the real optimized MAPs
  (a MapEstimate pool cache, else the Detector Experiment MAP output) or the per-window Sample Geometric
  Median of the posterior draws.

### Changed

- **SGM tool `--collection` now defaults to `map`** (was `posterior`).
- **The `map` collection no longer silently falls back to a proxy.** The former per-window medoid
  "proxy" is the per-window Sample Geometric Median; it is now the explicit `--map-source window-sgm`
  choice, and `--map-source experiment` errors loudly when neither real-MAP source exists rather than
  substituting a stand-in. The report records the condition and the resolved MAP source, and drops the
  between-condition (Simpson's paradox) correlation caveat when a single condition is selected.

## 0.4.5 - 2026-08-09

The two mirrored workflows -- **biology** (infers the reaction-diffusion parameters;
marginalizes imaging) and **detector** (infers the imaging parameters; marginalizes
reaction-diffusion + camera) -- now run on ONE shared engine per stage. Each stage's
orchestration was extracted from its two near-duplicate ~500-700-line entry-point scripts
into a single `run_<stage>(cfg, args)`, and the ten Prime entry points collapse to thin
(~40-line) shims that build a `WorkflowConfig` and call the shared runner. The workflows now
mirror each other by construction -- a change to a stage's engine lands in both, so neither
can silently drift (the drift this consolidation removes was real: the biology Inference had
fallen behind its detector twin). Behavior-preserving: each stage's old-vs-new dry-run is
byte-identical for both workflows, modulo the documented intended changes below. The
"canonical" workflow name is retired in favor of "biology" throughout.

### Added

- **`workflow.py`** — the `WorkflowConfig` dataclass (the workflow identity: `tag`, alias-qualified
  `paths`, `param_module`, `console_log_paths`) + the `biology_workflow()` / `detector_workflow()`
  factories. This is the single object threaded through every shared stage runner; the genuine
  per-workflow differences live here, not in duplicated `main()` bodies.
- **Five shared stage-runner modules** — `simulation_rds_runner.py`, `simulation_dli_runner.py`,
  `inference_runner.py`, `evaluation_runner.py`, `experiment_runner.py`, each holding the stage's
  full orchestration as `run_<stage>(cfg, args)` + a `build_<stage>_parser()`, with the one genuine
  per-workflow fork localized in a `_<stage>_spec(cfg)` resolver (e.g. RDS's reactive-vs-diffusion-only
  builder; DLI's imaging source — the `Nuisance_DLI` artifact for biology vs the imaging prior box
  for detector).

### Changed

- **Ten Prime entry points → thin shims.** `SRM_AND_SBI_DIMER_ALP_{Simulation_RDS, Simulation_DLI,
  Inference, Evaluation, Experiment}.py` and their `_DETECTOR` twins are now ~40-line shims: parse
  args, build the workflow config, call the shared runner.
- **`detector_experiment_support.py` → `experiment_support.py`.** The shared real-recording machinery
  (discovery / windowing / rank-sharding / shard I/O) is workflow-agnostic, so the `detector_` prefix
  is dropped; the biology Experiment stage now routes through it too (previously it carried its own
  inline copy — the 0.4.3 refactor had moved only the detector side onto the shared module).
- **Estimator / test-loss-distribution metadata `workflow` tag `"canonical"` → `"biology"`** (matches
  the retired workflow name).
- **Inference drift-gap folds (now in both workflows via the shared engine).** The biology Inference
  gains the `--num-workers` / `num_workers_override` DataLoader-budget knob, the theta-width guard on
  the loaded labels, and the complete finish-time estimator metadata (`train_videos` / `test_videos` +
  a non-finite `best_test_loss` → `None` guard) that had previously existed only in the detector twin.
- **Detector DLI `--debug` sim-0 diagnostics fix.** The detector DLI's `--debug` "Parameters of this
  video" table referenced an undefined `theta` against the wrong (10-RDS) table — a latent `NameError`
  on the debug path; the shared engine builds that table from each workflow's target spec (the
  six-imaging table + the imaging draw for the detector), so it is correct in both.
- **Detector Inference / Evaluation / Experiment estimator-path resolution** uses the shared
  `paths.estimator_path` method instead of a redundant local `_estimator_path` helper (identical path).
- Minor, benign banner normalizations where the two workflows' pre-run banners had drifted apart in
  cosmetic spacing/wording (e.g. the detector `data_bank_root` padding and the `reads estimator` label
  now match biology's); output content is unchanged.

## 0.4.4 - 2026-08-09

The canonical ("biology") DLI stage now marginalizes the imaging block in production. Rather than
holding the photophysics fixed at the detector's prior center (0.4.2), each simulation draws its six
photophysics from the pooled `Nuisance_DLI` artifact and its five SCOPE camera parameters from the
a-priori box, records both, and renders through one shared, source-agnostic renderer that the
detector stage also uses. This realizes the production imaging marginalization the 0.4.2 "fixed
imaging operating point" stood in for; it changes how DLI draws imaging — both nuisance sets are now
recorded beside the learnable theta — without changing the ten-parameter reaction-diffusion inference
target. Seedless throughout; the detector workflow's output is unchanged.

### Changed

- **Production imaging marginalization (canonical DLI).** The six photophysics (`mu_r`, `sigma_r`,
  `mu_pc`, `sigma_pc`, `prob_photo_bleach`, `lambda_rate`) are marginalized as a `nuisance_object`,
  drawn per simulation from the pooled `Nuisance_DLI` artifact — resolved under the detector alias
  from the durable tier via `require_nuisance_dli`, schema-guarded to the six photophysics in
  `DETECTOR_PARAMETER_KEYS` order. The five SCOPE camera parameters remain a `nuisance_spec` drawn
  per simulation from the a-priori box. Both draws are recorded per task, physical — as
  `Nuisance_DLI_Theta_Set` and `Nuisance_SCOPE_Theta_Set`, beside the learnable ten-RDS `Theta_Set` —
  and assembled into the eleven-key imaging vector (six photophysics ++ five camera) for rendering.
  Replaces the 0.4.2 fixed imaging operating point.
- **Shared source-agnostic renderer.** `render_dli_video` (`simulation_dli_support.py`) is the single
  renderer both workflows call: it reads the fixed imaging hyperparameters from the canonical
  parameterization table and renders from an explicit eleven-key imaging vector, independent of where
  that vector came from — the artifact-plus-box draw for the canonical stage, the SCOPE draw for the
  detector. The detector's `render_detector_video` is now a thin re-export alias of it, byte-identical
  in output (`array_equal` holds for both the `sum` and `multiply` dimer models).
- **Value-based role flip.** The six photophysics rows move from a fixed `VALUE` (0.4.2) to
  `VALUE = NUISANCE_SENTINEL` with `PRIOR_RANGE = None`, marking them a `nuisance_object` (drawn from a
  persisted artifact) as distinct from the SCOPE `nuisance_spec` (drawn from a box). The learnable
  subset is unchanged — exactly the ten reaction-diffusion parameters.

### Added

- **`dimer_mule` fixed row** in the canonical parameterization table (the merged-dimer brightness
  multiplicity relative to a monomer), so the shared renderer reads it from one place in both
  workflows. Inert under the default `sum` dimer model; consumed only by the retained `multiply`
  sensitivity alternative.

### Removed

- **`simulate_dli` and `draw_scope_camera`** (`simulation_dli_support.py`) — superseded by
  `render_dli_video` (the shared renderer) plus the artifact-and-box imaging draw now owned by the
  canonical Simulation_DLI stage. No caller remains.

## 0.4.3 - 2026-08-07

The two Detector real-recording stages (Experiment and the Nuisance_DLI construction) now share
one multi-GPU shard/merge machinery, and the Nuisance_DLI construction runs data-parallel across
GPUs like the Experiment. No science or data-schema change; the Experiment's behavior is preserved
(re-run over the real MET recordings reproduces the pre-refactor per-condition MAP result within
seedless noise).

### Added

- **`detector_experiment_support.py`** — shared machinery for the Detector real-recording stages:
  recording discovery (`discover_cells`), per-cell chunk windowing (`read_cell_chunks`), round-robin
  rank sharding (`shard_by_rank`), and per-worker shard I/O + merge (`shard_path`, `save_shard`,
  `load_shards`, `merge_shard_arrays`, `assert_consistent_shard_set`). Both stages share the same
  shape — load estimator → discover cells → window into chunks → run the estimator per chunk →
  aggregate — so this holds exactly the pieces they share; each keeps only its own per-chunk step.
- **Multi-GPU `Nuisance_DLI` construction.** `--emit-template` shards the `(kind, cell)` work across
  one worker per GPU (`torchrun`), each building its partial posterior-sample pool as a shard, and a
  no-GPU `--merge` concatenates the shards into the pool, caches it, and emits the spec — the pool is
  a mixture, so the concatenation is exact and order-independent. New HPC wrapper
  `Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_Nuisance_DLI.sh` mirrors the Experiment wrapper.
  Shards go in a dedicated `_Nuisance_DLI_pool_shards/` subdirectory with a stale-shard consistency
  guard, so a crash-then-rerun-with-different-worker-count fails loudly rather than merging a
  stale-plus-fresh mix.

### Changed

- **Detector Experiment** refactored to call the shared machinery in place of its inline discovery /
  windowing / round-robin / shard-merge — behavior-preserving: a re-run over the real MET recordings
  reproduces the pre-refactor per-condition MAP medians to within seedless noise (±0.002 log10 typical),
  with an identical `mean_log_prob`.

## 0.4.2 - 2026-08-07

The canonical ("biology") DLI forward model is brought onto the corrected imaging physics
proven in the detector workflow, parameter roles become value-based, and the trained estimator
gains a version-portable on-disk format. The canonical `simulate_dli` — which had been calling
retired `EMCCD`/`compute_matrices` signatures — runs again. This changes the DLI data schema
(camera keys, dimer model, background floor) and the reaction-diffusion prior ranges; no canonical
estimators or datasets existed under the old model, so nothing trained is invalidated. The detector
workflow is unaffected.

### Changed

- **Canonical DLI forward-model rework.** `simulate_dli` builds the corrected Poisson–Gamma–Normal
  `EMCCD` (`em_gain=gamma`, `electrons_per_adu=1.0`, `read_noise_adu=kappa_s`, `bias_adu=kappa_b`,
  `quantum_efficiency=kappa_q`), replacing the retired `EMCCD(offset, gain, variance)`. The pre-PSF
  photon floor is the optical background `kappa_o` (not a zero dark-count floor). Dimers render by the
  `sum` model — two independent labels, each with its own flicker trajectory — replacing `multiply`
  (retained as a sensitivity alternative). Flicker locality is fixed (`kappa_penalty=1`); the retired
  `gamma_penalty` is gone. Mirrors `render_detector_video`.
- **Camera is the SCOPE camera nuisance (canonical too).** The camera chain (`gamma`, `kappa_o`,
  `kappa_b`, `kappa_s`, `kappa_q`) is marginalized as the SCOPE nuisance — drawn per simulation from
  tight a-priori log10 boxes (transient, no artifact; `DETECTOR_WORKFLOW.md` §9.3) — rather than fixed
  known constants. `kappa_g`/`kappa_c` are retained as fixed spec metadata for the
  `gamma = kappa_g/kappa_c` drift check; `kappa_v` is retired.
- **Reaction-diffusion prior ranges** widened to match the detector nuisance ranges: per-species count
  `(1.75, 2.25) → (0.0, 2.5)` (log10; `[1, 316]`), `diffusivity_alp (-1, 0) → (-1.25, -0.25)`,
  `relative_diffusivity_bet (-0.75, -0.25) → (-0.625, -0.125)`; `relative_diffusivity_chi` unchanged.
- **Value-based parameter roles.** Each parameter's role is read from its `(VALUE, PRIOR_RANGE)` cell
  via `role_of` and the `NUISANCE`/`POSTERIOR` sentinels, so a block can be held fixed, inferred, or
  marginalized without structural change. The learnable subset (`VALUE`-not-a-sentinel AND
  `PRIOR_RANGE`-not-`None`) is exactly the 10 reaction-diffusion parameters.
- **Fixed imaging operating point.** The photophysics (`mu_r`, `sigma_r`, `mu_pc`, `sigma_pc`,
  `prob_photo_bleach`, `lambda_rate`) are held fixed at the detector's prior center (`VALUE = 10**mid`),
  pending marginalization from the pooled `Nuisance_DLI` artifact in a later step.

### Added

- **Version-portable estimator artifact** (`artifacts.save_estimator` / `load_estimator`): a
  self-describing `Estimator.npz` — compile-stripped `state_dict` + rebuild spec + a `parameter_keys`
  schema guard — replaces the torch-version-locked pickled `DirectPosterior` (`Posterior.pkl`). The
  Inference stage writes it; Evaluation and Experiment load it and reject a schema mismatch loudly.
- **`draw_scope_camera`** (`simulation_dli_support.py`): draws one SCOPE camera vector per simulation,
  log10-uniform over the a-priori boxes, mirroring the detector DLI stage's draw.

### Removed

- **`Construction_Optimum_ANN` special-situation entry point** — retired; its purpose is obsolete under
  the version-portable estimator (a checkpoint no longer needs a separate pickle-rebuild step, and the
  Inference stage writes the portable estimator directly). Its HPC-runbook and cross-doc references are
  removed.
- **Retired parameters/fields:** `gamma_penalty`, `kappa_v`, and the `SimulationDLI.darkcounts` field
  (superseded by the `kappa_o` optical-background floor).

### Migration

- The canonical DLI data schema and reaction-diffusion prior ranges changed, so any dataset or estimator
  built under the old model is inconsistent with this one. No canonical artifacts existed (the pipeline
  was never run in production), so nothing is invalidated; the biology workflow is regenerated and
  retrained from scratch under the corrected model.

## 0.4.1 - 2026-08-06

Training resume becomes seamless across requeues, and the inference launch gains a
learning-rate override. Both changes touch the two inference entry points (`Inference`,
`DETECTOR_Inference`) and the shared training loop; no data-schema or estimator-format
change, so existing datasets and estimators remain valid.

### Added

- **Full-state resume (`Resurrect_State_ANN`).** The training loop writes a complete
  training-state file — model weights, AdamW moments, `ReduceLROnPlateau` schedule,
  global epoch, best-so-far test loss, and the warm-restart counters — beside the
  optimum checkpoint (`Labor/…_Resurrect_State_ANN.pth`), atomically (temp file +
  `os.replace`) every epoch. With `--resurrect`, when the file is present the run
  **hot-restarts** from this exact state, so the learning-rate schedule continues
  seamlessly and no epochs are spent re-converging; when absent it falls back to the
  prior behavior (best-checkpoint weights into a fresh optimizer at the peak LR) and
  then writes the file, so subsequent requeues hot-restart. A chain of `--resurrect`
  rounds now behaves like one continuous run. New symbols:
  `parameterization.Paths.resurrect_state_pattern` / `resurrect_state_path`, and
  `inference_support.save_resume_state` / `load_resume_state` (with a `timing_label` +
  `parameter_keys` schema guard that refuses a mismatched resume file). The in-run
  warm-restart sawtooth is retained as a within-run plateau-escape and its state
  (`warm_restart_peak`, floor-dwell tally) is carried in the resume file, so it
  continues across requeues rather than resetting.
- **`warm_restart_factor` knob (default 0.25).** The warm-restart amplitude decay is now
  a dedicated `InferenceTraining` setting, separate from `scheduler_factor` (the
  per-epoch anneal step): each restart peak is `warm_restart_factor` times the previous,
  so a restart stays a gentle probe of a converged model (first restart a quarter of the
  peak) rather than a jump halfway back up. Previously this decay was coupled to
  `scheduler_factor` (0.5).
- **`--learning-rate` on the HPC inference wrappers.** An `LR` environment knob on
  `SRM_AND_SBI_DIMER_ALP_HPC_Inference.sh` and `…_DETECTOR_HPC_Inference.sh` forwards
  `--learning-rate` to the entry point, so a resumed chain can start below the peak LR.

## 0.4.0 - 2026-08-05

The DETECTOR calibration workflow becomes the production imaging-calibration path: the
diffraction-limited-imaging parameters are inferred with the physics frozen to pure diffusion, the
camera block is marginalized as an independent SCOPE nuisance, and the EMCCD forward model is the
corrected Poisson–Gamma–Normal chain. The learnable inference target is the 6-parameter emitter model;
the 5 camera parameters are drawn as a nuisance and rendered into every video. This changes the
estimator parameter dimension (`theta_dim` 10 → 6) and the data schema — estimators and datasets from
0.3.x are not compatible; regenerate the dataset and retrain.

### Changed

- **Detector inference target: 6 imaging parameters.** The learnable target is the emitter model —
  PSF (`mu_r`, `sigma_r`), brightness (`mu_pc`, `sigma_pc`), photophysics (`prob_photo_bleach`,
  `lambda_rate`). The camera chain (`gamma`, `kappa_o`, `kappa_b`, `kappa_s`, `kappa_q`) is reclassified
  from learnable to a marginalized **SCOPE** nuisance, drawn from tight a-priori boxes and rendered into
  every video; only the identifiable combinations (`gamma·kappa_q` effective gain, the ADU floor, the
  additive baseline) are constrained. `kappa_g`/`kappa_c` are retained as fixed spec metadata for the
  `gamma = g/C` drift check. New symbols `DETECTOR_PARAMETER_KEYS`, `DETECTOR_NUISANCE_SCOPE`,
  `DETECTOR_IMAGING`/`DETECTOR_IMAGING_KEYS`, `DETECTOR_SCOPE_KEYS`; the imaging vector is 6 learnable +
  5 SCOPE = 11 keys, and the estimator load path enforces a schema guard against a mismatched target.
- **Corrected EMCCD noise model (Poisson–Gamma–Normal).** `EMCCD`/`add_noise` apply Poisson
  photoelectrons → stochastic `Gamma(N, em_gain)` electron multiplication (excess-noise factor
  `F² = 2`) → conversion → gain-independent Gaussian read noise added after the register → bias,
  replacing the gain-scaled-variance-before-register form.
- **Dimer brightness: `sum` model by default.** A dimer's per-frame brightness is the sum of two
  independent monomer draws (mean `2·E[X]`, lighter upper tail than doubling), each with its own
  photophysical state; `dimer_model="multiply"` (rigid `dimer_mule` scaling) is retained as an option.
- **RDS count nuisance extended to `[1, 316]` per species** (log10 floor `1.0 → 0.0`), covering sparse
  monomer-dominated fields.
- **In-run warm-restart LR schedule** (`InferenceTraining.warm_restart_dwell`): the learning-rate
  restart cycles now happen within a single run, reproducing within one job the mechanism a chain of
  `--resurrect` rounds provided; the DataLoader worker budget is capped per GPU with a per-run
  `--num-workers` override.

### Added

- **`NuisanceDLI`** (`detector_nuisance_dli.py`) — a self-contained, samplable marginalized-parameter
  artifact with a five-choice posterior-sample pool (`raw`, `map_estimate_pool`, `gaussian`, `box`,
  `box_user`) and weight-checksum pool caching, built directly from the estimator and the recordings.
  It subsumes the former standalone `Nuisance` class.
- **Analysis utilities** (special-situation, not pipeline stages): `..._DETECTOR_Embedding_Space_Distance`
  (experimental-vs-synthetic MMD + C2ST), `..._DETECTOR_Flicker_Rate_Derivation` (the `lambda_rate`
  autocorrelation-time derivation), and `..._Test_Loss_Distribution_Analysis`.

### Removed

- **`nuisance.py`** — its `Nuisance` class is absorbed into `NuisanceDLI` (`detector_nuisance_dli.py`).

### Migration

- The detector inference target changed from 10 to 6 parameters and the training distribution changed
  (SCOPE camera boxes, extended count nuisance), so datasets and estimators built on the 0.3.x target
  are incompatible. Regenerate the dataset and retrain.

## 0.3.2 - 2026-07-19

Align the `Complex3DCNN` constructor defaults with the production network configuration and correct a
worked-example docstring. Behavior-preserving — the production inference path always passes the
network configuration explicitly, so no run is affected; this only changes a bare `Complex3DCNN(...)`
build (tests or ad-hoc use).

### Changed

- **`Complex3DCNN` constructor defaults now match the production config** (`inference_network.py`):
  `n_conv_layers` 4 → 5, `n_attn_layers` 1 → 2, `attention_heads` 2 → 4, so a bare constructor yields
  the 128-dimensional embedding (`start_channels · 2^(n_conv_layers − 1) = 8 · 2⁴`) instead of the
  64-dimensional one. `temporal_target_frames` is left at the documented `None` (no-reduction) default.
  The docstring's worked example is corrected (`n_conv_layers=5` gives 128).

## 0.3.1 - 2026-07-19

Add the reference specification for the corrected EMCCD detector noise model, reconcile the
detector-workflow calibration text with it, and widen the detector 2 s generation wall. All
documentation plus one HPC-controller config value; no package runtime code changes.

### Added

- **`REFERENCE_EMCCD_NOISE_MODEL.md`** — a self-contained specification of the physically grounded
  EMCCD forward model (Poisson photoelectrons → stochastic Gamma electron-multiplication register with
  excess-noise factor `F² = 2` → conversion → gain-independent read noise → bias), the separable
  optical-background model, the four camera parameters (`kappa_g`, `kappa_c`, `kappa_s`, `kappa_b`;
  `kappa_q`/QE fixed) under spec-informed priors, the identifiable ratio `γ = g/C` (ADU per
  photoelectron) as a reported diagnostic, and the validation protocol. This is design documentation
  for a later iteration; the current forward model is unchanged.

### Changed

- **`DETECTOR_WORKFLOW.md`** — §9.1 cites the new specification as the corrected model its deferred
  items point toward; §3 reconciles the photon-transfer calibration (a single photon-transfer curve
  yields the overall system gain `γ = g/C`; the EM gain and conversion are anchored to the camera
  spec, not decomposed from the videos).
- **Detector 2 s generation wall 18 h → 24 h.** The `Generate_Controller` `tlim` for the 2 s
  train/test/eval generation plan is raised to match the 5 s lines, for scheduling headroom.

## 0.3.0 - 2026-07-17

Reparameterize the brightness-flicker generator to infer only the switching rate and derive its
locality from the brightness scale, retiring the weakly identifiable brightness-transition penalty;
re-anchor the affected detector imaging priors; and document the EMCCD read-noise limitation. The
shared forward model changes, so the canonical DLI stage is intentionally left non-running until its
rework.

### Changed

- **Brightness-flicker generator — derived locality, inferred rate.** `compute_matrices`
  (`simulation_dli_support.py`) builds the transition kernel as
  `Q[i,j] = lambda_rate · exp(−kappa_penalty · |b_i − b_j| / sigma_bright)`, with a fixed
  dimensionless locality `kappa_penalty = 1` and `sigma_bright` the photon-space standard deviation
  of the emitter-brightness log-normal (`mu_pc · exp(sigma_pc²/2) · sqrt(exp(sigma_pc²) − 1)`). The
  switching rate `lambda_rate` is the single inferred flicker parameter; the locality is derived, not
  free. Rationale and citations are in `DETECTOR_WORKFLOW.md` §6.3.
- **Detector inference target: 11 → 10 imaging parameters.** With the flicker locality derived,
  `detector_parameterization.py` drops the separate penalty parameter from the learnable imaging
  vector.
- **Re-anchored detector imaging priors** (log10 ranges):

  | parameter | old range | new range |
  |---|---|---|
  | `kappa_g` (EM gain) | (2.0, 3.0) | (1.5, 2.5) |
  | `mu_pc` (median brightness) | (2.0, 3.0) | (1.75, 2.75) |
  | `sigma_pc` (brightness shape) | (−1.0, 0.0) | (−1.0, 0.25) |
  | `lambda_rate` (switching rate) | (−1.0, 1.5) | (−1.25, 0.25) |

  The `kappa_g`, `mu_pc`, and `sigma_pc` ranges are anchored to the real-data maximum-a-posteriori
  estimates (`DETECTOR_WORKFLOW.md` §6.2); `lambda_rate` is anchored to the real-data flicker dwell
  (§6.3). Other imaging ranges are unchanged.
- **Documentation — representative prior centers.** `DETECTOR_WORKFLOW.md` §6.2 reports each prior's
  geometric center (a representative value) rather than a fixed constant, and §6.3 is rewritten
  around the derived-locality generator. `CLAUDE.md` scope, `PROJECT_CONTEXT.md`, and `VALIDATION.md`
  are updated accordingly.
- **`--seed None` accepted explicitly.** Every entry-point stage parses `--seed None` (in addition to
  an integer, or omission) to the seedless default, so commands and documentation can state
  seedlessness explicitly rather than leave it implicit.
- **Run documentation harmonized.** `VALIDATION.md` section 2 now covers the smoke test for both
  workflows with a single authoritative Detector recipe (section 2.5), an HPC multi-node and
  multi-GPU section (section 2.6), and a production-run section (section 2.7); every smoke is seedless
  and every smoke or production run requires explicit approval before submission. Script headers, the
  HPC runbook, and the benchmarks now cross-reference section 2.5 instead of restating counts, and the
  stale seeded, bounded-pool, and parameter-count examples are corrected.

### Added

- **`VIDEO_DTYPE_BITS` knob** in `SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_Simulation.sh`, forwarded to the
  DLI stage's `--video-dtype-bits` (default 8), so the synthetic-video bit depth is explicit and
  controllable on HPC.
- **Detector calibration smoke test** — the authoritative seedless five-stage recipe in
  `VALIDATION.md` section 2.5, with the HPC (section 2.6) and production (section 2.7) run sections.

### Removed

- **`gamma_penalty` as a free parameter.** The brightness-transition penalty (previously a learnable
  detector parameter and a fixed canonical constant) is removed from the flicker kernel; its role is
  filled by the derived `sigma_bright` locality. The canonical parameterization still carries the
  constant pending its rework (below).

### Notes

- **The canonical DLI stage does not run under this release, by design.** `simulate_dli` still passes
  the retired `gamma_penalty` to the shared `compute_matrices`, which no longer accepts it — a
  deliberate tripwire. The canonical workflow is repaired, with the value-based parameter roles and
  the derived-locality generator, only after the Detector workflow is validated (`_bet`; see
  `DETECTOR_WORKFLOW.md` §9.1 and `CLAUDE.md` scope).
- **EMCCD read-noise limitation documented.** The `add_noise` readout term is added as a gain-scaled
  variance before the gain rather than a gain-independent standard deviation after the register; a
  candidate correction is not yet settled and is deferred to `_bet` (see the read-noise note in
  `PROJECT_CONTEXT.md`).

## 0.2.29 - 2026-07-16

Rename the posterior-predictive "real" side to "experimental", make the notebook player render
frame-for-frame like the static figure, and add zoom to the player.

### Changed

- **"EXPERIMENTAL" replaces "REAL"** throughout the posterior-predictive tool — figure titles,
  histogram legend, info panel, the printed metadata, and the stored `.npz` key/variables
  (`experimental`, `experimental_tif`) — for rigorous experimental-vs-synthetic naming.
- **Consistent per-frame color scaling** — the notebook player now re-autoscales each frame (it
  previously autoscaled to frame 0 and held it), matching the scrubber and the engine's static
  figure, so a given frame renders identically in every view. `autoscale` = each frame's own
  min/max; `percentile` = a fixed whole-clip `[p0.5, p99.5]` window.
- **Softened the bright-tail caption** to "(counts or flicker may drive the bright-pixel tail)"
  (panel and docstring) — the counts explanation is one hypothesis, the flicker generator another.

### Added

- **Zoom in the player** — `PLAY_ZOOM` (and a center) crop the played video to a region of
  interest, so local experimental-vs-synthetic agreement can be inspected in motion, matching the
  scrubber's zoom.

### Documentation

- Updated the posterior-predictive companion doc to the above: EXPERIMENTAL labels, the per-frame
  color convention, the player zoom, and the inferred + nuisance provenance panel.

## 0.2.28 - 2026-07-15

Incorporate the three analysis-script companion docs into the general documentation (two-tier).

### Documentation

- **Two-tier incorporation of the analysis-script docs** — concise per-script entries added to
  the general documentation, each pointing to its standalone companion for the detail: the
  `Nuisance_DLI` construction into `DETECTOR_WORKFLOW.md` (§7), the CD86/CTLA-4 control-receptor
  reuse into `PROJECT_CONTEXT.md` (§4 analyses), and the pre-launch seeding/file-label check into
  `VALIDATION.md` (§3 reproducibility).

## 0.2.27 - 2026-07-15

Show the inferred and nuisance parameters on the posterior-predictive comparison figure, and
add independent companion docs for three analysis scripts.

### Changed

- **Comparison figure info panel** — now lists this render's INFERRED imaging MAP theta
  (absolute units, out-of-prior parameters flagged) and its NUISANCE RDS draw (absolute:
  particle counts and diffusion), under explicit labels, alongside the existing provenance
  notes. The particle counts in particular help attribute a real-vs-synthetic histogram
  mismatch (for example a thinner synthetic bright tail) to the nuisance draw or the imaging
  estimate rather than leaving it unexplained.

### Added

- **Companion docs for three analysis scripts** (independent, for later incorporation into the
  general documentation): `SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.md`,
  `SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls.md`, and
  `SRM_AND_SBI_DIMER_ALP_Seeding_Validation.md` — each documenting the script's purpose, usage,
  outputs, and interpretation in the established Analysis-folder style.

## 0.2.26 - 2026-07-15

Improve the posterior-predictive comparison figure and add its usage/interpretation doc.

### Changed

- **Comparison figure pairs conditions by column** — REAL (col 0) and SYNTH (col 1) each show
  the mid frame over its own max projection, so a frame sits directly above its max projection;
  the shared pixel-intensity histogram (top) and the provenance text (bottom) fill the third
  column.

### Added

- **Posterior-predictive video companion doc** —
  `Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.md`: how to run
  the script (arguments, selection modes, dry-run), the outputs and their naming, how to read the
  comparison figure, and a step-by-step guide to the viewer notebook.

## 0.2.25 - 2026-07-15

Add the posterior-predictive video analysis tool, and document the EMCCD readout-noise
units discrepancy (model kept as-is).

### Added

- **Posterior-predictive video tool** — `Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.py`
  and the matching pure-viewer notebook `notebooks/SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.ipynb`:
  render a synthetic video from a selected `(kind, cell, chunk)` or per-cell-median MAP imaging
  estimate at the real recording's own length, and persist real + synthetic + provenance
  (`*_Synthetic_Video.npz`), a static comparison figure, and the drawn trajectory. Two display
  normalizations (`autoscale` / `percentile`); post-hoc analysis, not a canonical stage.

### Changed

- **Comparison figure uses the stored (clipped) synthetic** — the posterior-predictive comparison
  figure now plots the non-negative uint16 frames that are stored and viewed, not the pre-clip
  float, so the real-vs-synthetic histogram compares like with like. The storage clip now reports
  the negative/overflow excursion counts (`n_under` / `n_over`) and is documented as a deliberate,
  revisit-able choice.

### Documentation

- **EMCCD readout-noise units discrepancy** — the readout term scales a unit normal by
  `variance / gain^2` (a variance where a standard deviation belongs; effective post-gain
  standard deviation `variance / gain` rather than `sqrt(variance)`), reproducing the reference
  implementation. The model is kept as-is (it matches real data); the discrepancy, citations, and
  the deferred corrected form are recorded in `PROJECT_CONTEXT.md` (DLI Imaging read-noise note),
  `DETECTOR_WORKFLOW.md` (§9.1), and the `EMCCD` docstring.

## 0.2.24 - 2026-07-13

Add the Detector Experiment stage (MAP on real recordings) and the standalone building
blocks for the sim-vs-real gap check and the Nuisance_DLI, move the Detector workflow
document into the repository, and reconcile documentation.

### Added

- **Detector Experiment stage** — `..._DETECTOR_Experiment.py` and its HPC wrapper
  `..._DETECTOR_HPC_Experiment.sh` (wired into the Detector dispatcher): maximum-a-posteriori
  estimation of the imaging parameters on real recordings per condition, a canonical
  `Experiment` copy plus only the forced deviations (imaging-θ target, version-portable
  estimator load, `_DETECTOR` namespacing); multi-GPU sharded, non-deterministic.
- **`detector_embedding_space_distance.py`** — the sim-vs-real embedding-space gap measure:
  RBF-MMD (median-heuristic bandwidth, cell-block permutation p-value) and a cell-grouped
  C2ST (GroupKFold cross-validation, one-sided t-test on per-fold accuracies) on the trained
  embedding, with cell-level resampling. Standalone; not yet invoked by the Experiment.
- **`detector_nuisance_dli.py`** — the `Nuisance_DLI` construction module: emit a value-based
  spec template pre-filled with posterior-derived suggestions, fully validate a user-finalized
  spec (structure and values within the imaging prior box), build the pooled `Nuisance`
  (per-parameter, posterior, or samples form), and the validating gate `require_nuisance_dli`.
- **Nuisance_DLI analysis entry point** — `Script_Bank/Analysis/..._DETECTOR_Nuisance_DLI.py`:
  presents the calibrated imaging posterior and emits the spec template (`--emit-template`),
  then builds the artifact from the finalized spec (`--build`); additive, never wired into the
  dispatcher; outputs written beside the posterior in the data-bank `Posit/` directory.

### Changed

- Moved **`DETECTOR_WORKFLOW.md`** into the repository as the authoritative Detector workflow
  design, tightened for submission, with the Nuisance_DLI construction and embedding-space
  distance sections reconciled to the code.
- Removed the orphaned `Nuisance.from_posterior` builder — no caller; posterior draws are
  materialized to a stored sample-set inside `build_nuisance_dli`.

### Documentation

- Folded the estimator-generalization methods into `PROJECT_CONTEXT.md` (paired log-scoring on
  a shared set, calibration and coverage, in-distribution versus out-of-distribution).
- Documentation-discipline sweep across the Detector Prime and HPC scripts, docstrings, and this
  changelog: replaced non-self-contained "estimator (A5)" tokens with self-contained references,
  and removed fragile section-number cross-references in favor of named-section references.

Held for a follow-up — end-to-end testable only once a trained Detector estimator and a
completed Experiment run exist: wiring the gap check into the Experiment, and matched-synthetic
generation.

## 0.2.23 - 2026-07-12

Add the Detector generation controller for staged, multi-node production data generation.

### Added

- **`SRM_AND_SBI_DIMER_ALP_DETECTOR_HPC_Generate_Controller.sh`** — the Detector
  variant of the canonical generation controller (a verbatim copy plus three forced
  deviations: the Detector Simulation launcher as `SIM`, `_DETECTOR` job-names, and a
  distinct state file). Drives the staged, QOS-gated multi-node production generation
  — submit the TRAIN+TEST arrays within the in-system cap, hard-gate EVAL until they
  all COMPLETE, resurrect-safe re-run of any failed node — for the Detector 2S/5S
  datasets. Selected via `CASES=2s|5s|both`; replaces the ad-hoc smoke orchestrator.

## 0.2.22 - 2026-07-12

Rename the Detector's persisted RDS-nuisance set for clarity and document the
nuisance-set naming convention. Filename-token change only; no behavior change.

### Changed

- **Nuisance set filename.** The Detector's per-task RDS-nuisance set token changes
  `Nuisance_RDS_Set` → `Nuisance_RDS_Theta_Set` — reading unambiguously as a
  `Theta_Set` variant (parallel to the learnable theta set) holding the marginalized
  RDS-domain draws. It still lives in `Theta/`, still derived from the canonical
  theta-set path (no dedicated directory, no new path code); the samplable
  `Nuisance_RDS` object name is unchanged.

### Documentation

- Documented the persisted nuisance-set naming pattern
  `{project_alias}_{timing_label}_Nuisance_<DOMAIN>_Theta_Set_TASK_{n}_{split}.{ext}`
  (Prime docstring + the nuisance and artifact design in `DETECTOR_WORKFLOW.md`), distinguishing the samplable object
  from the persisted set file, and recording the forward production/canonical
  convention `Nuisance_DLI_Theta_Set` (imaging marginalized).

## 0.2.21 - 2026-07-12

Cap training-run backup proliferation with an opt-in flag, applied identically to
both the canonical and Detector Inference Primes (shared behavior; nothing is ever
deleted — intermediate backups simply are not written by default).

### Changed

- **Backup retention.** A training run now keeps a single provenance-named backup
  triplet (checkpoint + posterior/estimator + test-loss distribution) — the finish
  backup — instead of one per improving epoch. The live canonical artifacts still
  update at every new best (crash-safe, `--resurrect`-ready); only the per-epoch
  backup *copies* are no longer written by default.

### Added

- **`--backup-every-best`** (Inference, both workflows): opt back into a backup at
  every new-best epoch for the full per-epoch history when debugging a training run.

## 0.2.20 - 2026-07-12

Reconcile the Detector workflow to the canonical workflow. Every Detector Prime
script and HPC wrapper is rebuilt as a verbatim copy of its canonical counterpart
plus only the justified deviations — the imaging inference target, the
version-portable estimator artifact, and `_DETECTOR` coexistence namespacing. The two
are now parallel, essentially-identical SBI pipelines sharing the same essential
machinery by import; they diverge only where the science requires it. This
corrects divergences that earlier from-scratch drafts had introduced (confirmed
by a full process-direction + parity sanity check).

### Fixed

- **Generation seed convention.** The Detector generation forced a fixed seed
  (mandatory `SEED` in the wrapper + dispatcher, plus a spurious
  `SeedSequence.spawn` in the Prime scripts), freezing the per-video placement,
  PSF, brightness, and EMCCD-noise realizations across an entire split — degenerate
  for SBI. Restored the canonical seedless convention (`--seed` optional, default
  `None` → a fresh realization per video; plain `np.random.default_rng`).
- **Inference checkpoint directory** is created before the first save (was missing;
  crashed a fresh Detector run) — inherited from the canonical copy.
- **Synthetic video bit depth** back to 8-bit (16-bit is the raw experimental
  storage format, down-converted on load).
- **Debug-transcript namespacing:** all four Detector stages write their
  `--debug-dump` console transcript to the `_DETECTOR` namespace, not the canonical
  one.

### Changed

- **Detector Prime scripts** rebuilt as canonical copies + only the forced
  deviations, restoring the full canonical machinery earlier drafts had dropped
  (the `DiagnosticReporter` report, posterior coverage / View B / `--summary`, the
  per-example test-loss distribution, the debug CLI). Inference and Evaluation
  persist the estimator via the version-portable format (`artifacts`) in place
  of the torch-version-locked pickle.
- **Detector HPC wrappers + dispatcher** rebuilt as verbatim canonical copies +
  `_DETECTOR` filename aliasing only — dropping the mandatory `SEED`, the GPU-mode
  auto-pin, and a divergent `set -eo pipefail`, and restoring `--summary` /
  `POOL_MODE` and the canonical GPU-count derivation.

### Added

- `--dry-run` on the canonical generation Primes (`Simulation_RDS`,
  `Simulation_DLI`), mirroring `Inference`/`Evaluation`, so every Prime entry point
  honors the documented dry-run convention. Additive and behavior-preserving.

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
  `SEED`), `..._Inference.sh` (data-parallel training via `torchrun`; saves the version-portable
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
  `Simulation_RDS`, `Simulation_DLI`, `Inference` (saves the version-portable estimator),
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
- `PROJECT_CONTEXT.md` (Inference Pipeline): Stage 1 (detector parameters) reclassified from a
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

- **Receptor identity (the system section of PROJECT_CONTEXT.md).** The system section mislabeled the
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

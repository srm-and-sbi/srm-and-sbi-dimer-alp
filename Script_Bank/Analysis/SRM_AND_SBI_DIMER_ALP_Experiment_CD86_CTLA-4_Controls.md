# Control-receptor experiment (CD86 / CTLA-4) — usage and interpretation

Companion to `SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls.py`. The script
MAP-estimates the model parameters from real single-particle-tracking recordings of
two control receptors — CD86 and CTLA-4 — by reusing the DIMER-ALP posterior trained
on the MET regime, with no retraining; this note explains what it does, how to run it,
what it writes, and how to read the per-condition result, so the analysis can be used
and understood without reverse-engineering the code.

This is a special-scope, ad-hoc reuse of a trained posterior on data from a different
study. It is a
near-verbatim clone of the canonical MET Experiment stage that differs only in the
dataset folder it reads, the output directory it writes, and its default `--kinds`;
every estimation and reporting behavior is identical, so the canonical MET Experiment
stage and its outputs are never touched. It lives in `Script_Bank/Analysis`, is not one
of the canonical pipeline stages, and is kept out of the stage dispatcher.

## What it does

CD86 is a constitutive monomer and CTLA-4 a constitutive dimer — oligomeric-state
controls whose mobile diffusion coefficients have been measured independently. The
script splits each long raw recording into model-length windows, MAP-estimates the
parameter vector in every window of every cell, and reports the distribution of those
estimates per condition. Both receptors run together in one pass; a `kind_index` field
keeps them apart in the output. Because real microscopy data carry no ground truth, the
report is a per-condition distribution of inferred parameters, not a recovery check —
parameter recovery is validated only on held-out synthetic EVAL videos (the Evaluation
stage), and that validation is a property of the posterior that carries over to this
reuse.

The robust read-out here is the diffusion **scale**, not the monomer-versus-dimer
identity. The control recordings use an exchangeable, blinking SiR-S5 HaloTag label,
whereas the posterior was trained on always-visible permanent-label emitters; the
blinking corrupts the brightness cue the posterior relies on to separate a monomer (one
dye) from a dimer (two dyes), and the true monomer/dimer diffusion gap is small. The
per-species split (D_A, D_B, per-class counts) is therefore not a trustworthy
oligomeric decomposition on these controls; the diffusion coefficient is a per-track
property read from displacement statistics while a molecule is visible, so it is
insensitive to how many emitters are lit at once and transfers across the mismatch. The
quantitative diffusion-scale interpretation — the count-weighted mobile mixture
diffusivity D_mix_mobile and its comparison against the measured mobile-fraction
coefficients — is developed in the companion note
`SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls_Temporal_Dynamics.md`, which
also treats the per-window temporal behavior; this note documents the base
whole-recording estimation and its report.

## How to run it

Run on a machine that holds the trained posterior and the control recordings, under the
inference environment. Preview first with `--dry-run`, which validates the posterior,
the dataset folder, and the discovered recordings, prints what would be read and
written, and runs no estimation (no GPU, no compute).

Single process — estimates and writes the report directly:

    MACHINE_PROFILE=<profile> python \
      Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls.py \
      --total-time-seconds 2.0 --kinds CD86,CTLA-4 [--dry-run]

Multi-GPU — each worker MAP-estimates its share of the (kind, cell) work and writes a
shard; a separate combine-only `--merge` pass (no GPU) concatenates the shards into the
final report:

    MACHINE_PROFILE=<profile> torchrun --nproc_per_node=4 \
      SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls.py \
      --total-time-seconds 2.0 --kinds CD86,CTLA-4 --pool-mode unrestricted
    MACHINE_PROFILE=<profile> python \
      SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls.py \
      --total-time-seconds 2.0 --kinds CD86,CTLA-4 --merge

Arguments:

- `--total-time-seconds` (required) — the model-window duration in seconds; must match
  the trained posterior (e.g. `2.0`). It sets the window length each recording is
  chunked into and locates the posterior.
- `--experiment-span-seconds` — the length of each raw recording in seconds (default
  `20`); used to build the filename and to compute the chunk count per recording.
- `--chunk-step-seconds` — step in seconds between consecutive model-length windows; an
  integer that divides the window and does not exceed it (`1` = maximal overlap; the
  window length, the default, = non-overlapping). Smaller steps yield more, overlapping
  chunks per recording.
- `--kinds` — comma-separated control receptors run together in one pass (default
  `CD86,CTLA-4`); distinguished in the output by `kind_index`.
- `--cells` — comma-separated explicit cell indices (default: discover every matching
  recording on disk).
- `--max-cells` — cap on cells per kind (`0` = all; useful for quick checks).
- `--summary` — which views to render: `map` (View A: the per-condition distribution of
  MAP-point estimates; default), `posterior` (View B: each chunk's posterior median ±
  IQR per condition), or `both`. View B draws `--posterior-samples` per chunk.
- `--posterior-samples` — samples per chunk used to summarize the posterior in View B
  (default: the evaluation-config value).
- `--aggregation` — the report's distribution view: `pooled` (every (cell, chunk)
  estimate is one sample; mixes temporal and biological variation; default) or
  `cell-median` (one sample per cell, the median over its chunks; biological spread
  only). The saved arrays keep the raw per-(cell, chunk) data either way, so both views
  are reproducible from one run.
- `--pool-mode` — candidate-pool sampler for the MAP optimization: `bounded` (the config
  default; rejection sampling inside the prior box) or `unrestricted` (the mode may leave
  the prior box; the per-parameter figures mark the prior bounds with dashed lines so any
  excursion is visible).
- `--seed` — master RNG seed (PyTorch, numpy, Python `random`); default `None` →
  non-deterministic, consistent with the generation stages. Pass an integer for a
  reproducible run.
- `--theta-prex-size`, `--elite-prex-size`, `--numb-steps`, `--show-progress-steps`,
  `--learning-rate` — MAP-optimization hyperparameters; each falls back to its
  evaluation-config default when omitted.
- `--verbose`, `--debug`, `--debug-dump` — increasing console and progress-log detail;
  `--debug-dump` also tees the console transcript to a `Labor/Debug/<run>/Experiment/`
  log.
- `--merge` — combine-only mode: read the per-shard `.npz` files from a multi-GPU run in
  this run's output directory, concatenate them, write the final report, figures, and
  combined `.npz`, then remove the shards and exit. Does no estimation and needs no GPU;
  single-process runs never use it.
- `--dry-run` — validate configuration and inputs, print what would be read and written,
  then exit without estimating.

Inputs:

- The trained posterior at the posterior path for the requested `--total-time-seconds`
  (the MET-regime DIMER-ALP posterior; not retrained here).
- Real recordings under
  `<data_bank>/Experiment/SPT_Data_CD86_CTLA-4_CONTROLS_S-BIAD1369/`, named
  `Experiment_{CD86|CTLA-4}_Cell_{n}_{span}S_RAW.tif` (16-bit raw video). Each recording
  is read, converted from 16-bit to the 8-bit range the estimator was trained on, and
  chunked into model-length windows.

## Outputs

Written to
`<data_bank>/<posit_subdir>/SRM_AND_SBI_DIMER_ALP_<timing_label>_MAP_Experiment_CD86_CTLA-4_CONTROLS/`
(for example `SRM_AND_SBI_DIMER_ALP_2S_50FPS_MAP_Experiment_CD86_CTLA-4_CONTROLS/`),
distinct from the canonical MET Experiment output directory:

- `<dir-name>.npz` — the per-(cell, chunk) arrays: `inferred_log10` (the MAP theta in
  log10 units), `scores` (MAP log-density at the optimized mode), `kind_index`, `cell`,
  `chunk`, `kinds`, and, when View B was requested, `posterior_quantiles`
  (five quantiles [Q05, Q25, Q50, Q75, Q95] per parameter per chunk). This raw array is
  the reusable record; downstream analyses (including the temporal-dynamics companion)
  read it.
- `report.md` — the deliverable: run statistics (conditions, total estimates,
  aggregation view, per-condition estimate counts, mean MAP log-density), and a table of
  inferred theta by condition in log10 units, noting explicitly that real data have no
  ground truth.
- `figures/` — one `experiment_<key>.png` per learnable parameter (built by
  `figure_experiment_combined`). View A shows the per-condition distribution of inferred
  MAP theta; View B, when requested, shows each chunk's posterior median ± IQR per
  condition. A panel stamped "not computed" marks a view the `--summary` option omitted.
- `progress.log` — a live, timestamped per-chunk progress trace (`tail -f` to monitor);
  under sharding only rank 0 writes it.

## How to read the result

The report table is a side-by-side comparison of the inferred parameter distributions
of the two conditions, in log10 units. It answers "how do the inferred parameters differ
between CD86 and CTLA-4?", not "what is the true value?" — there is no true value to
compare against on real data. Read differences **between** conditions, and read the
diffusion parameters (the monomer diffusivity D_A and, through the ratio R_B, the mobile
dimer diffusivity D_B) as the transferable, quantitatively interpretable observable; do
not read the per-class counts, the kinetic rates, or the monomer/dimer split as a
resolved oligomeric decomposition on these controls (see the caveats). The mean MAP
log-density is an optimization diagnostic, not a calibration or quality metric.

The `--aggregation` choice sets what one row of the distribution means: `pooled` treats
every (cell, chunk) window as a sample and mixes within-recording temporal variation
with between-cell biological variation; `cell-median` collapses each cell to its median
first, so the distribution reflects biological spread across cells with within-cell
temporal noise averaged out. When View B is enabled, the posterior median ± IQR panels
show within-chunk posterior uncertainty, complementary to the between-sample spread of
View A.

For the quantitative diffusion-scale read-out — the count-weighted mobile mixture
diffusivity D_mix_mobile, its comparison against the measured mobile-fraction
coefficients, and the per-window temporal behavior — see
`SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls_Temporal_Dynamics.md`.

## Caveats

- **No ground truth on real data.** The report is a distribution of inferred parameters,
  not a recovery check. Parameter recovery is quantified only on held-out synthetic EVAL
  videos in the Evaluation stage; that recovery quality is a property of the posterior
  and carries over here, but nothing about the real controls is a validated recovery.
- **Label / model mismatch.** The controls use an exchangeable, blinking SiR-S5 label,
  unlike the always-visible permanent-label training regime. The diffusion scale
  transfers across the mismatch; the per-class counts and kinetic rates depend on the
  number of co-visible emitters and on track continuity, both corrupted by blinking, and
  are not read quantitatively.
- **Not the trained mechanism.** CD86 and CTLA-4 are constitutive monomer / dimer
  controls, not the dynamic dimerization mechanism the posterior encodes. They bracket
  the diffusion scale and stress-test transferability; they do not exercise the kinetic
  model, so the rate parameters carry no biological read-out here.
- **Prior leakage under unrestricted pooling.** On these out-of-distribution controls
  the MAP can leave the training prior box (`--pool-mode unrestricted` does not enforce
  it); the per-parameter figures mark the prior bounds with dashed lines so any such excursion is visible. The
  count-weighted diffusion-scale read-out is chosen precisely because it is robust to
  this and to the monomer/dimer mis-assignment.
- **8-bit calibration domain.** Each recording is estimated on the fixed 8-bit rescale
  the estimator was trained on; sub-8-bit detail in the raw 16-bit video lies outside
  the calibrated domain.
- **First-pass posteriors.** Current short-duration posteriors from interrupted training
  give provisional absolute values; the numbers will sharpen with the full production
  posteriors. Re-run this analysis on those for the definitive figures.
- **Relative parameters** (R_B, R_C, R_ON) are reported as the dimensionless ratios the
  model samples.

## Reference

Catapano et al., "Long-Term Single-Molecule Tracking in Living Cells using
Weak-Affinity Protein Labeling," *Angew. Chem. Int. Ed.* **2025**, 64, e202413117.
doi:10.1002/anie.202413117. Data: EMBL-EBI BioImage Archive accession S-BIAD1369 (CD86
and CTLA-4, SiR-S5 / HaloTag single-color tracking; the experimental mobile-fraction
diffusion coefficients are D_mobile = 0.319 ± 0.010 µm²/s for CD86 and 0.279 ± 0.005
µm²/s for CTLA-4). The acquisition geometry (50 FPS, 256 × 256 pixels, ~157 nm/pixel)
matches the training regime, which is why the diffusion-scale comparison is
quantitative. The DIMER-ALP model, its parameters, and the Experiment stage are
documented in `PROJECT_CONTEXT.md`.
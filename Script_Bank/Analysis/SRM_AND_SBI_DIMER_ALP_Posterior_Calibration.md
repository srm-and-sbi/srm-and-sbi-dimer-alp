# Posterior Calibration

Companion to `SRM_AND_SBI_DIMER_ALP_Posterior_Calibration.py` (biology) and
`SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Calibration.py` (detector). This is the
authoritative reference for both; the detector companion points here.

## What it does

The Inference stage trains a neural posterior `q(theta | x)` over a workflow's target
parameters. A low test loss says the posterior assigns high density to the truth on
average, but it does **not** say the posterior's *uncertainty* is honest — whether its
credible intervals cover the truth at their nominal rate. This diagnostic answers that
question. It draws the trained posterior for every held-out EVAL video (whose
ground-truth theta is known) and scores calibration with four established
simulation-based-calibration diagnostics, **overall and stratified by each target
parameter**. The EVAL theta are prior draws and the EVAL namespace is physically
separate from TRAIN/TEST, so the EVAL set is itself a proper calibration sample and the
result is leak-free by construction.

It is an **Analysis diagnostic**, not a pipeline stage: it is never wired into the
`Submit.sh` dispatcher, reads only finished artifacts (the estimator and the EVAL set),
and writes a self-contained report to the `Posit/` tier alongside the posterior it
characterizes. It never modifies pipeline data.

## One tool, both workflows

Calibration is workflow-agnostic: both workflows train a posterior over their own target
theta, and the same four diagnostics apply unchanged. So the tool is built once and
serves both, following the codebase's shared-engine pattern:

- **Kernel** `posterior_calibration.py` — the pure statistics (numpy + torch + sbi),
  operating only on theta-space arrays and embeddings; it imports nothing from
  `parameterization`/`artifacts` and knows no workflow tag.
- **Runner** `posterior_calibration_runner.py` — `run_posterior_calibration(cfg, args)`:
  streams EVAL, draws each video's calibration inputs, calls the kernel, writes the
  report. All per-workflow differences (which parameterization supplies the target keys +
  prior, the alias-qualified paths) are resolved from the `WorkflowConfig` in
  `_posterior_calibration_spec(cfg)`.
- **Two thin shims** — the biology and detector entry points, each ~10 lines that build
  the workflow config and call the runner. Their **names carry the namespace** (the
  detector one adds the `_DETECTOR` alias), matching the data files, so a run and its
  outputs are never ambiguous about which workflow they belong to.

The biology shim scores the 10 reaction-diffusion parameters; the detector shim scores
the 6 imaging parameters. Same engine, same report structure, correct namespace each.

## The four diagnostics

Each wraps the validated implementation in `sbi.diagnostics`; the runner supplies the
pre-drawn inputs (see *Design* below).

- **SBC — simulation-based calibration** (Talts et al. 2018). For each video, the rank of
  the true theta among the posterior samples is computed per parameter. If the posterior
  is calibrated, these ranks are uniform on `{0..L}` for every marginal. Uniformity is
  scored with a Kolmogorov-Smirnov test (fast, primary); an optional per-marginal
  classifier two-sample test (`--sbc-c2st`) is the slower, more sensitive secondary
  check. **Read it from the rank-histogram figure:** flat = calibrated, a `U`-shape means
  over-confidence (ranks pile at the edges), a `^`-shape over-dispersion, a slope a bias.

- **Expected coverage** (Deistler et al. 2022 / Hermans et al. 2022). The rank of the true
  theta's posterior log-density among the samples' log-densities is likewise uniform when
  calibrated. The empirical-versus-nominal coverage curve reads off directly: on the
  diagonal is calibrated, above it conservative (intervals too wide), below it
  overconfident (intervals too narrow).

- **TARP — tests of accuracy with random points** (Lemos et al. 2023). A necessary and
  sufficient coverage test in the full joint parameter space (not just per marginal),
  summarized by the area-to-curve (ATC): `0` ideal, `> 0` over-dispersed, `< 0`
  under-dispersed (overconfident).

- **L-C2ST — local classifier two-sample test** (Linhart et al. 2023). The only
  per-observation diagnostic: it trains a classifier to tell posterior draws from the
  prior-simulator joint as a function of the observation, then asks, at each of a subset
  of observations, whether `q(theta | x)` matches the true posterior *locally*. Reported
  as the fraction of observations that reject calibration; roughly the significance level
  `alpha` when calibrated. This is the heaviest diagnostic and conditions on the learned
  **embedding**, not the raw video (see below).

No single number settles calibration — the four are read together. A miscalibration that
one misses (marginal-only SBC vs. joint TARP vs. local L-C2ST) another catches, and a
single low p-value on one marginal of a well-calibrated posterior is expected noise, not
a verdict.

## Design: pre-drawn arrays, and the sbi reuse boundary

sbi's `run_sbc` / `run_tarp` take the observed data as one in-memory tensor and draw the
posterior internally. Here each observation is a full microscopy video (tens of MB), so
materializing ~10^4 of them is infeasible, and sbi's diagnostics assume low-dimensional
summary statistics, not raw videos. So the runner **streams the EVAL videos once** —
exactly as the Evaluation stage does — and for each video draws the posterior sample
cloud and the sample + truth log-densities (reusing `evaluation.collect_theta_prex` /
`collect_score_prex` and the Evaluation stage's embed-once trick so scoring `L`
candidates costs one `Complex3DCNN` pass), plus the learned embedding. It hands the
kernel only those small arrays.

Consequently the three theta-space tests (SBC, coverage, TARP) never touch the
observation at all; they compare the drawn samples to the true theta. **L-C2ST is the
exception** — it must condition on the observation, and a raw video is far too
high-dimensional for a classifier, so it uses the learned `Complex3DCNN` embedding (the
summary the flow itself conditions on). The kernel therefore reuses only the parts of sbi
that accept pre-drawn inputs (`check_sbc`, `check_tarp`, `LC2ST`, the samples-based
`_run_tarp`); the only hand-written pieces are the trivial rank definitions
(`#{samples < truth}`), which are definitions, not algorithms.

All theta live in log10 space (the flow's and the prior's space); the ground-truth theta
sets are stored linear, so calibration is scored on `log10(theta_true)`.

## Stratification — by the inferred value, not the truth

Calibration can hold on average yet fail in a subregion. Every diagnostic is therefore
also computed **stratified**: the EVAL videos are binned into equal-count bins along one
target parameter, and the diagnostic is recomputed per bin. The binning axis is the
posterior's **inferred value** for that parameter — the per-video posterior median — and
never the latent ground truth. This is a deliberate, load-bearing choice. The rank of the
truth is uniform *conditional on the observation `x`*, and hence conditional on any
function of `x` such as the inferred value, so a genuinely calibrated posterior stays
uniform inside every inferred-value bin. Binning on the ground truth instead would
confound **Bayesian shrinkage** — the correct pull of the posterior toward the prior when
the data is uninformative, strongest precisely in the low-count regime — with
**miscalibration**, flagging an honest posterior in exactly the regime you most want to
trust the diagnostic.

So the stratified panel answers the right question: *"for the videos the posterior infers
to sit in this region, is it calibrated there?"* For biology, the `count_*` panels tell
you whether the posterior is honest for the videos it infers as low-count (wide but
covering) or overconfident there (too narrow, missing the truth); for detector it sweeps
the imaging parameters. A bin that flags while the overall passes localizes exactly where
the posterior is not to be trusted. The stratifying dimension is chosen with `--stratify`
and is fully general — both workflows have a target-theta vector; the code names no
workflow-specific covariate.

## How to run

**1 — Select the machine.** The active profile in `machine_profiles.toml` resolves every
path; set it once per shell:

```bash
export MACHINE_PROFILE=<profile>     # e.g. mars_pc, jupiter, goethe
```

**2 — Dry-run first (always).** Resolves the profile, prints exactly what it would read
and write, checks that the estimator and EVAL sets exist, validates `--tests` /
`--stratify`, and exits without touching the GPU:

```bash
python SRM_AND_SBI_DIMER_ALP_Posterior_Calibration.py \
    --total-time-seconds 2.0 --eval-tasks 10 --dry-run
```

**3 — Run.** Biology (10 reaction-diffusion parameters) or detector (6 imaging
parameters) — identical options; the entry-point name selects the workflow and namespace:

```bash
# biology
python SRM_AND_SBI_DIMER_ALP_Posterior_Calibration.py \
    --total-time-seconds 2.0 --eval-tasks 10 --posterior-samples 1000

# detector
python SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Calibration.py \
    --total-time-seconds 2.0 --eval-tasks 10 --posterior-samples 1000
```

While iterating, `--max-sims 20` caps videos per task and `--tests sbc,coverage` skips the
heavier L-C2ST.

**4 — At scale (multi-GPU / multi-node).** The draw is sharding-aware, exactly like the
Evaluation stage, and the HPC wrapper
`Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Posterior_Calibration.sh` drives it end to end:
with `>1` node it places one `torchrun` launcher per node; with one node and `>1` GPU it
shards across that node's GPUs (`torchrun --standalone`); with one GPU it runs
single-process. Each rank draws a round-robin slice of the EVAL tasks and writes a shard,
and the wrapper then runs the single `--merge` pass automatically. `--gres` is per node,
so `--nodes=N --gres=gpu:G` gives `world_size = N×G`; `WORKFLOW=biology|detector` picks the
workflow. Example (single node, all GPUs, biology):

```bash
sbatch --job-name=SRM_AND_SBI_DIMER_ALP_2S_50FPS_Posterior_Calibration \
    --export=ALL,REPO=$PWD,EVAL_TASKS=25,POSTERIOR_SAMPLES=1000 \
    Script_Bank/HPC/SRM_AND_SBI_DIMER_ALP_HPC_Posterior_Calibration.sh
```

On a scheduler that needs them, add the account/partition overrides (e.g.
`--partition=booster --account=<acct> --gres=gpu:4`). To run by hand instead, launch the
shim under `torchrun`/`srun` (one process per GPU) then invoke it once more with `--merge`.
One GPU needs neither — it runs single-process and writes the report directly.

**5 — Read the output.** The report and figures land under
`<data_bank>/Posit/<project_alias>_<timing_label>_Posterior_Calibration/`; open `report.md`
and `figures/`. *Reading the report* below says what every metric means.

Key options (full list via `--help`):

| Option | Meaning |
|---|---|
| `--total-time-seconds` | Video duration; must match the trained posterior's runs. **Required.** |
| `--eval-tasks` | Number of held-out EVAL tasks to draw. **Required.** |
| `--posterior-samples` | Draws per video, `L` (default from the evaluation config). SBC/TARP use the full cloud; L-C2ST one draw/video. |
| `--tests` | Comma-separated subset of `sbc,coverage,tarp,lc2st` (default: all). |
| `--stratify` | `all` (every target dimension; default), `none` (overall only), or a comma-list of parameter KEYS. |
| `--n-strata` / `--min-stratum` | Equal-count bins per dimension (default 5) / minimum videos to score a bin (default 200). |
| `--pool-mode` | `bounded` (rejection within the prior; correct for a trained posterior) or `unrestricted` (flow direct; for smoke tests). |
| `--sbc-c2st` | Also run SBC's slower per-marginal classifier two-sample test (off by default). |
| `--lc2st-n-eval` / `--lc2st-null-trials` | L-C2ST evaluation observations (200) / null classifiers (100). |
| `--max-sims` | Cap videos per task (0 = all); useful for a quick check. |
| `--seed` | Master RNG seed (default None → non-deterministic, consistent with generation). |

## Outputs

Written to `<data_bank>/Posit/<project_alias>_<timing_label>_Posterior_Calibration/`
(`<project_alias>` carries `_DETECTOR` for the detector workflow):

- `report.md` — the calibration report: per-parameter SBC table, coverage / TARP / L-C2ST
  statistics, and the stratified digest.
- `figures/sbc_rank_histograms.png` — per-marginal rank histograms with the uniform band.
- `figures/expected_coverage.png`, `figures/tarp_ecp.png` — the coverage / ECP curves.
- `figures/stratified_<test>.png` — each diagnostic across the target-theta bins.
- `<...>_Posterior_Calibration.npz` — the saved inputs (truths, samples, log-densities,
  embeddings, parameter keys) for reproducible re-analysis without redrawing.

## Reading the report — what every metric means

No single number is the verdict: read the four measures together and lean on the figures.
Each measure detects a failure the others can miss (marginal SBC vs. joint TARP vs. local
L-C2ST), and a lone low p-value on one marginal of an otherwise-flat posterior is expected
Monte-Carlo noise, not a defect.

| Metric (as it appears in `report.md`) | What it is | Calibrated looks like | Miscalibrated looks like |
|---|---|---|---|
| **SBC — `KS p-value`** (per parameter) | KS test that the true θ's rank among the posterior samples is uniform, for that marginal | large p (roughly ≳ 0.05); a **flat** rank histogram | small p; a **U-shaped** histogram (overconfident — truth in the tails) or **∩-shaped** (over-dispersed) |
| **SBC — `C2ST(rank)`** (opt-in, `--sbc-c2st`) | accuracy of a classifier trying to tell the ranks from uniform | ≈ 0.50 | → 1.0 |
| **SBC — `verdict`** | `ok`/`check` — a **Bonferroni-corrected** advisory flag across the d marginals | `ok` | `check` (then read that parameter's histogram) |
| **`coverage_KS_pval`** | KS test that the truth's posterior-log-density rank is uniform (Deistler/Hermans) | large p | small p |
| **`coverage_points`** (`nominal→empirical` at 0.5 / 0.9) | empirical coverage of the credible region at each nominal level | empirical ≈ nominal (e.g. `0.90→0.90`) | **below** nominal = overconfident; **above** = conservative |
| **`tarp_ATC`** | area between the TARP ECP curve and the diagonal, in the full joint space | ≈ 0 | **> 0** intervals too wide; **< 0** too narrow (overconfident) |
| **`tarp_KS_pval`** | KS test that the TARP ECP equals the credibility level | large p | small p |
| **`lc2st_reject_fraction`** | share of observations whose **local** test rejects calibration at α = 0.05 | ≈ 0.05 | ≫ 0.05 |
| **`lc2st_median_pvalue`** | median local L-C2ST p-value across the evaluated observations | large | small |
| **Stratified table** (per `inferred <param> in [lo, hi)` bin) | the same measures recomputed within each equal-count bin of the parameter's **inferred** value | bins match the overall | a bin that flags while the overall passes ⇒ the posterior is miscalibrated *in that inferred-value subregion* |

The `figures/` mirror the tables: `sbc_rank_histograms.png` (the shape is the real SBC
read), `expected_coverage.png` and `tarp_ecp.png` (distance below the diagonal = how
overconfident), and `stratified_<test>.png` (which inferred-value bins fail).

## Multi-GPU sharding

The draw over 10^4 videos × ~10^3 samples is heavy, so the runner shards exactly like the
Evaluation stage: each rank draws a round-robin slice of the EVAL tasks and writes a
partial `_shard_*_of_*.npz`; a single `--merge` pass concatenates the shards and runs the
calibration statistics on the full set (SBC / TARP / L-C2ST are global, so the merge is
where the diagnostic actually runs). The code path is identical on one GPU (single
process, no shards), one node with several GPUs, and several nodes. The
`SRM_AND_SBI_DIMER_ALP_HPC_Posterior_Calibration.sh` wrapper runs both phases (the sharded
draw and the merge) in one job.

## Reuse scope

This diagnostic is **read-only** and **never dispatched**: it reads a finished estimator
and the held-out EVAL set, writes only under `Posit/`, and is not a `Submit.sh` case. It
serves **both** workflows through the two namespaced shims over one shared engine — the
same generality principle as the pipeline stages, applied to a diagnostic. Run it after a
posterior is trained (and re-run it on a candidate checkpoint) to decide whether the
posterior's uncertainty can be trusted, and where.

## References

- Talts, Betancourt, Simpson, Vehtari, Gelman (2018). *Validating Bayesian Inference
  Algorithms with Simulation-Based Calibration.* arXiv:1804.06788.
- Hermans, Delaunoy, Rozet, Wehenkel, Begy, Louppe (2022). *A Trust Crisis in
  Simulation-Based Inference? Your Posterior Approximations Can Be Unfaithful.* TMLR.
- Deistler, Goncalves, Macke (2022). *Truncated proposals for scalable and hassle-free
  simulation-based inference.* arXiv:2210.04815.
- Lemos, Coogan, Hezaveh, Perreault-Levasseur (2023). *Sampling-Based Accuracy Testing of
  Posterior Estimators for General Inference (TARP).* arXiv:2302.03026.
- Linhart, Gramfort, Rodrigues (2023). *L-C2ST: Local Diagnostics for Posterior
  Approximations in Simulation-Based Inference.* arXiv:2306.03580.

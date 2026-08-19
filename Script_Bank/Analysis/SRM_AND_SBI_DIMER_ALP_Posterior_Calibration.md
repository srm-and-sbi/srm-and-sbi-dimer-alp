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

### The 1-D / 2-D / joint ladder

Rank uniformity of the **one-dimensional** marginals is *necessary but not sufficient*
(Talts et al. 2018, with the caveat of Modrák et al. 2023): a posterior can have every
marginal perfectly calibrated and still misstate how the parameters covary, and marginal
SBC is blind to that. The report therefore checks three rungs:

1. **1-D** — per-parameter SBC (`KS D`) and the diagonal of the marginal-calibration matrix.
2. **2-D** — every parameter *pair*, via the TARP coverage test restricted to that
   two-dimensional subspace. A pair scoring worse than either of its parameters alone
   localizes a **dependence error**: the correlation is wrong even though each parameter
   looks fine on its own. Reported as `dependence_excess` and shown as the off-diagonal of
   the matrix figure.
3. **Joint** — TARP over all dimensions, which *is* sufficient in the population limit.

Two dimensions is one rung up the ladder, not the top of it; the value of reporting all
three is knowing how far the posterior has actually been verified.

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
to sit in this region, is it calibrated there?"* Where a parameter's low end yields
recordings that carry little information about it, this is what distinguishes an honest
posterior there (wide, but still covering the truth) from an overconfident one (too narrow,
missing it) — a distinction the overall result averages away. A bin that flags while the
overall passes localizes exactly where the posterior is not to be trusted.

The stratifying dimension is chosen with `--stratify` and is fully general: each workflow
supplies its own target-theta vector and the panels are labeled from it, so the tool names
no parameter of either workflow and the two never appear in the same report.

### The digest, and why the profile shape is the point

With `--n-strata=10`, four diagnostics and a six- or ten-parameter target vector, the
per-bin results number in the hundreds. Tabulating them one bin per row is the wrong
presentation twice over: no reader holds hundreds of rows, and the thing the stratification
exists to reveal — *how* the statistic varies across a parameter's range — is exactly what a
long column of numbers conceals. The report therefore carries a **digest**: one row per
(parameter, test), giving how many bins flag, the worst bin and where it sits, and the
**shape** of the profile. The per-bin values are plotted in `figures/stratified_<test>.png`
and stored in full in the `.npz`, so nothing is lost — the numbers move to where they can
actually be read, and the table states what they add up to.

The shape is classified from the rank correlation between a bin's position and its
statistic (|ρ| ≥ 0.60 for a monotone call), falling back to a comparison of the two edge
bins against the interior. The four readings differ materially:

| `profile across the range` | What it indicates |
|---|---|
| **rises** / **falls with the inferred value** | a progressive loss of identifiability along that parameter's range — the regime where the recordings carry least information about it. Not a training fault: more epochs will not create information the data lacks. |
| **worst at both ends of the range** | the **prior's edges**, where the training set is thinnest and the posterior is pulled inward from both sides at once. Symptomatic of prior-boundary effects rather than of model capacity. |
| **worst in the middle of the range** | a genuine interior feature — most often a **degeneracy** between two parameters that only bites where their effects overlap. Worth checking against the 2-D marginal matrix for the partner parameter. |
| **flat, no clear trend** | the defect, if any, is uniform across the range: a global property of the posterior, which the overall statistics already describe. |

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
| `--n-strata` / `--min-stratum` | Equal-count bins per dimension (default **10**) / minimum videos to score a bin (default 200). 10 is chosen for resolution in log10 space: at 5 bins a parameter spanning 2.5 decades puts a factor of ~3 inside one bin, blurring distinct identifiability regimes. Bins below the minimum are skipped **and counted** (see `stratification_complete`). |
| `--num-bins` | TARP credibility bins (default **50**). Resolution along the credibility axis, not a data split — every video informs every level — so it is free at production N. |
| `--n-jobs` | Worker processes for the stratum loop (default auto: the largest power of two up to 16 that fits the cores *allocated* to the job; `1` = serial). The stratified statistics run in the **single-process merge step**, so this — not the GPU count — is what makes a fine stratification cheap. |
| `--pool-mode` | `bounded` (rejection within the prior; correct for a trained posterior) or `unrestricted` (flow direct; for smoke tests). |
| `--sbc-c2st` | Also run SBC's slower per-marginal classifier two-sample test (off by default). |
| `--lc2st-n-eval` / `--lc2st-null-trials` | L-C2ST evaluation observations (**1000**) / null classifiers (100). Evaluation is independent of the dominant training cost, so a large `n-eval` is nearly free and sets the reject fraction's precision (SE ≈ 0.016 at 1000, vs 0.035 at 200). |
| `--max-sims` | Cap videos per task (0 = all); useful for a quick check. |
| `--seed` | Master RNG seed (default None → non-deterministic, consistent with generation). |

## Outputs

Written to `<data_bank>/Posit/<project_alias>_<timing_label>_Posterior_Calibration/`
(`<project_alias>` carries `_DETECTOR` for the detector workflow):

- `report.md` — the calibration report: a `Run:` header stamping the UTC date and time it
  was produced, then the per-parameter SBC table, coverage / TARP / L-C2ST statistics, the
  1-D / 2-D marginal ladder, the location-versus-width **Diagnosis**, the **What this
  implies** reading, and the stratified digest. The timestamp is supplied by the reporter
  itself rather than by the caller, so no entry point — a stage run, a smoke, or a one-off
  re-analysis of the saved arrays — can produce an undated report; a caller with a remark
  about the run passes it as a `run_note`, which appears on its own line and never
  displaces the stamp.
- `figures/sbc_rank_histograms.png` — per-marginal rank histograms with the uniform band.
- `figures/expected_coverage.png`, `figures/tarp_ecp.png` — the coverage / ECP curves.
- `figures/marginal_calibration_matrix.png` — 1-D (diagonal) and 2-D (off-diagonal) marginal
  calibration.
- `figures/stratified_<test>.png` — each diagnostic across the target-theta bins.
- `<...>_Posterior_Calibration.npz` — the saved inputs (truths, samples, log-densities,
  embeddings, parameter keys) for reproducible re-analysis without redrawing.

## Reading the report — what every metric means

No single number is the verdict: read the four measures together and lean on the figures.
Each measure detects a failure the others can miss (marginal SBC vs. joint TARP vs. local
L-C2ST).

**Every reported quantity is an effect size, never a p-value.** This is deliberate and it
matters at production scale: with 10⁴ videos a Kolmogorov-Smirnov p-value is ≈ 0 for *any*
deviation from uniformity however small, so p-value verdicts flag everything and rank
nothing. The effect sizes (`KS D`, the coverage gap, `ATC`, the reject fraction) are
sample-size independent, so the same practical threshold — 0.05 for the three calibration
measures, i.e. a five-percentage-point deviation — applies to the overall result and to
every stratum alike. The p-values are still printed beside them, for completeness only.

**What the p-values do and do not say — and what `0.00e+00` means.** A p-value here answers
only *"how often would a perfectly calibrated posterior produce a result at least this
uneven, by chance alone?"* It is a statement about **detectability**, not about size,
severity, or quality. Three consequences follow, and all three are easy to get backwards:

- **It is not a score.** A calibrated posterior does not drive the p-value to 1; it makes
  it **uniform on (0, 1)**. So `tarp_KS_pval = 0.97` is no more evidence of calibration
  than `0.31` would be — a large value is simply one of the values a calibrated posterior
  produces. Only the effect size (`ATC`, `KS D`, the coverage gap) speaks to magnitude.
- **A printed `0.00e+00` is not a probability of zero.** It is an **underflow of double
  precision** — the true value is below roughly 10⁻³⁰⁸ and there is no floating-point room
  left to print it. It means the deviation is certainly *real*, and says nothing whatever
  about whether it is *large*.
- **At 10⁴ videos, "real" is a very low bar.** The KS test's resolution grows with √N, so a
  deviation far too small to affect any downstream use is still detected with certainty.
  This is precisely why parameters whose `KS D` is unremarkable still print `p = 0.00e+00`,
  and why the p-value column cannot rank the parameters. Use it only to confirm that a
  deviation exists; size it from `KS D`, from the rank-histogram shape, and from the
  `physical bias` column of the Diagnosis table.

**The rank histogram's *shape* carries information no summary number does.** A one-sided
spike at rank 0 means the posterior systematically **over-estimates** that parameter (every
sample above the truth); a ramp rising toward the top rank means it **under-estimates**; a
symmetric **U** means overconfidence and an **∩** over-dispersion. Bias and width errors are
different defects with different fixes, so always read the histogram before concluding
"overconfident" from a large `KS D` alone.

| Metric (as it appears in `report.md`) | What it is | Calibrated looks like | Miscalibrated looks like |
|---|---|---|---|
| **SBC — `KS D`** (per parameter) | **effect size**: the largest deviation of that marginal's rank CDF from uniform | small (≲ 0.05); a **flat** rank histogram | large; a **U-shaped** histogram (overconfident — truth in the tails), **∩-shaped** (over-dispersed), or a one-sided **spike/ramp** (biased — see below) |
| **SBC — `KS p-value`** | the same test's p-value: a **detectability** statement, not a score (see above) | uniform on (0, 1) — no particular value | at production N it is ≈ 0 for *any* detectable deviation, printing as `0.00e+00` (double-precision underflow), so it ranks nothing |
| **SBC — `C2ST(rank)`** (opt-in, `--sbc-c2st`) | accuracy of a classifier trying to tell the ranks from uniform | ≈ 0.50 | → 1.0 |
| **SBC — `verdict`** | `ok`/`check` from the **effect size** (`KS D > 0.05`) | `ok` | `check` (then read that parameter's histogram) |
| **`coverage_max_gap`** | **effect size**: the largest gap between empirical and nominal coverage, and the level where it occurs | small (≲ 0.05) | large — reads directly as "the *c*-credible interval really covers *c* ± this" |
| **`coverage_points`** (`nominal→empirical` at 0.5 / 0.9) | empirical coverage of the credible region at each nominal level | empirical ≈ nominal (e.g. `0.90→0.90`) | **below** nominal = overconfident; **above** = conservative |
| **`coverage_KS_pval`** | the uniformity test's p-value, for completeness only | — | — (same caveat as SBC's) |
| **`stratification_complete`** | check: how many stratification bins were scored vs skipped for holding fewer than `--min-stratum` videos | PASS — all bins scored | FAIL — the stratified section is **partial**; raise the video count or lower `--n-strata` |
| **Diagnosis — `bias (z)`** | systematic offset in units of the posterior's **own** standard deviation; positive = the parameter is over-estimated | ≈ 0 (bar: 0.30) | large \|z\| — the posterior sits in the wrong **place** |
| **Diagnosis — `spread (z)`** | actual error ÷ claimed standard deviation | ≈ 1 (bars: 0.87–1.15) | **> 1** intervals too narrow (overconfident); **< 1** too wide (conservative) |
| **Diagnosis — `sharpness`** | posterior SD as a fraction of the prior width — how much the data constrained the parameter | context, not a verdict | — (a very small value amplifies any offset into a large `bias (z)`) |
| **Diagnosis — `physical bias`** | the offset as a multiplicative factor on the physical value | ≈ 1.000× | the number to weigh against the precision the downstream use needs |
| **Diagnosis — `defect`** | the dominant error: `ok` / `location (high\|low)` / `width (too narrow\|too wide)` | `ok` | names which of the two errors dominates |
| **`marginal_1d_worst` / `marginal_2d_worst`** | worst \|ATC\| over the 1-D marginals / over the 2-D pairs | both small | see the ladder above |
| **`dependence_excess`** | how much the worst pair exceeds *both* of its own marginals | `none` | positive — a **dependence** error localized to that pair |
| **`tarp_ATC`** | area between the TARP ECP curve and the diagonal, in the full joint space | ≈ 0 | **> 0** intervals too wide; **< 0** too narrow (overconfident) |
| **`tarp_KS_pval`** | KS test that the TARP ECP equals the credibility level — again **detectability only** | uniform on (0, 1) | small p confirms a deviation exists; it does **not** size it, and a large p is not evidence of calibration. Judge TARP by `ATC` |
| **`lc2st_reject_fraction`** | share of observations whose **local** test rejects calibration at α = 0.05 | ≈ 0.05 | ≫ 0.05 |
| **`lc2st_median_pvalue`** | median local L-C2ST p-value across the evaluated observations | large | small |
| **Stratified digest — `bins flagged`** (per parameter × test) | how many of that parameter's equal-count **inferred-value** bins exceed the practical threshold | `0/N` | a **fraction** localizes a subregion; `N/N` is a global defect merely seen through the strata |
| **Stratified digest — `worst bin value` / `at inferred value`** | the largest per-bin effect size and the inferred value where it occurs | small | names the exact regime to distrust |
| **Stratified digest — `profile across the range`** | the **shape** the bins trace: `rises`/`falls with the inferred value`, `worst at both ends of the range`, `worst in the middle of the range`, or `flat` | `flat` | each shape has a distinct reading — see below |

The `figures/` mirror the tables: `sbc_rank_histograms.png` (the shape is the real SBC
read), `expected_coverage.png` and `tarp_ecp.png` (distance below the diagonal = how
overconfident), `marginal_calibration_matrix.png` (the 1-D diagonal and 2-D off-diagonal
ladder), and `stratified_<test>.png` (which inferred-value bins fail — for SBC each panel
shows **that parameter's own** statistic across its own range, not the worst across all
parameters, so a single bad parameter cannot paint every panel).

### Deciding what to do about it

A flag says the posterior is miscalibrated; it does not say the artifact must be retrained.
The **Diagnosis** and **What this implies** tables exist to separate the questions, along
two axes:

**Accuracy versus uncertainty.** A *location* error moves the estimate; a *width* error only
misstates its error bar. Only the first can change a scientific conclusion drawn from a point
estimate. A posterior can be flagged by every rank measure while its estimates remain accurate
to a few percent — the finding then concerns the error bars, and the correct response is to
treat the reported credible intervals as lower bounds, not to retrain.

**Statistical versus physical size.** The rank measures are scale-free: they ask "is the offset
large *compared with the posterior's own width*?" A very sharp posterior (small `sharpness`)
turns a physically negligible offset into a large `bias (z)`, and hence a large `KS D`. The
`physical bias` column restores the scale that matters — compare it against the precision the
downstream use actually requires.

Reading the two together:

| pattern | what it means | indicated response |
|---|---|---|
| `spread (z)` > 1 with a small physical bias | more confident than it has earned | widen or re-scale the reported intervals; **not** a retrain trigger |
| large physical bias | the estimate itself is off | investigate the forward model, the training distribution, or a degeneracy with another parameter **before** retraining |
| defects confined to particular strata | a limit of what those recordings identify | accept as an identifiability limit — more or longer training will not remove it |
| widespread width errors in one direction, or errors still shrinking with training | capacity or convergence | retraining (more data, more epochs, more capacity) is the indicated response |

Retraining the same architecture on the same data reproduces the same *systematic* error and
risks the parameters that are already calibrated, so it is the right response only when the
pattern points at capacity or convergence rather than at the model or the data.

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

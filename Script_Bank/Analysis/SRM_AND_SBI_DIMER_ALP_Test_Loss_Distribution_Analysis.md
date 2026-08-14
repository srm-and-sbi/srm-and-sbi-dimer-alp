# Test-loss distribution analysis — usage and interpretation

Companion to `SRM_AND_SBI_DIMER_ALP_Test_Loss_Distribution_Analysis.py` (biology) and
`SRM_AND_SBI_DIMER_ALP_DETECTOR_Test_Loss_Distribution_Analysis.py` (detector); this is the
authoritative reference for both. The script reads a saved per-example test-loss distribution
artifact and produces the picture the single scalar test loss cannot give: the shape of the
distribution, an interpretable reference for what the numbers mean, and which regions of
parameter space the estimator finds hard. This note explains how to run it and how to read its
outputs, so the analysis can be used and understood without reverse-engineering the code.

**One tool, both workflows.** Both workflows write a Test-Loss-Distribution artifact whose
manifest is self-describing, so the analysis is workflow-agnostic and built once over the
shared-engine pattern: a workflow-agnostic kernel (`test_loss_analysis.py`), a shared runner
(`test_loss_analysis_runner.py`), and two thin namespaced shims. The only per-workflow
difference is which alias-qualified `Posit/` the canonical artifact is resolved from; the
entry-point name carries the namespace (biology's 10 reaction-diffusion parameters, or the
detector's 6 imaging parameters).

This is a post-hoc, read-only analysis step, not one of the canonical pipeline stages. It
lives in `Script_Bank/Analysis`, is never wired into the stage dispatcher, and is purely
additive: it imports the core modules (`test_loss_analysis`, `test_loss_distribution`,
`diagnostics`) and reads a completed artifact, modifying nothing. It consumes only the artifact
and the machine profile, so it can be run at any time after training on any machine that holds
the artifact — including directly on an arbitrary `.npz` with no machine profile at all.

## What it analyzes

At its best epoch the Inference stage records one per-example negative log-likelihood (NLL)
over the fixed held-out TEST set — the `.npz` snapshot written by `TestLossDistribution`,
holding parallel `keys` (the stable `(task, sim)` identity), `theta` (the true parameter
vector, in log10), and `loss` (the per-example NLL), plus a self-describing `manifest`. The
NLL is the log score of the trained posterior: `loss = -log q(theta | video)`, the density the
estimator places on the true parameter given that example's video. Lower is better; the mean
of this array is the scalar test loss the training log reports.

Every parameter fact the analysis needs — the learnable keys, their roles, their prior ranges,
the log flags, and the run provenance — is read from the artifact's own manifest, not from the
live parameterization. The artifact is therefore the single source of truth for its own
analysis: the report stays correct even if the parameterization later changes, and the code
generalizes to any number of learnable parameters without edits.

The analysis has two parts.

**A. Distribution shape and reference.** The mean, median, standard deviation, Fisher-Pearson
skewness, the minimum and maximum, a quantile spread (1/5/25/50/75/90/95th and 99th
percentiles), and a percentile-bootstrap 95% confidence interval on the mean. A positive skew
and a heavy upper tail reveal the "catastrophic miss" examples that a mean, pulled by that
tail, conceals.

**B. Tail versus parameter space.** For each learnable parameter, the Spearman rank
correlation between the per-example NLL and the parameter value (the monotone association over
all examples), together with a comparison of the hard tail against the bulk — the difference in
their mean parameter value and a two-sample Kolmogorov-Smirnov statistic on the two
distributions. The hard tail is the top `1 - hard_quantile` of examples by NLL (the top 5% by
default). Together these locate which parameters, and which regions of their range, drive the
hard examples — the actionable content the mean hides.

## The uniform-prior reference

The absolute value of an NLL is not self-interpreting: it depends on the units and
dimensionality of the parameter space. The analysis anchors it with a reference computed from
the artifact's own prior — the expected NLL under the uniform prior itself.

For a uniform prior over the learnable box, the density is the constant `1 / V`, where `V` is
the prior-box volume, so the NLL of the prior is the constant `NLL_prior = ln V = sum_j
ln(range_j)` (natural log; each `range_j` is the width of parameter `j`'s prior interval, in
the log10 space the density is scored in). This is the no-information baseline: an estimator
that learned nothing — returning the prior for every video — scores exactly `NLL_prior` on
every example. The analysis reports it three ways:

- **Information gain** — `NLL_prior - mean_NLL`, in nats: how far below the baseline the mean
  sits, i.e. how much the estimator has learned over the prior.
- **Worse-than-prior fraction** — the fraction of examples whose NLL exceeds `NLL_prior`: cases
  where the estimator did worse than simply returning the prior. This is a principled
  "catastrophic miss" line, replacing an arbitrary threshold, and it is the point where the
  empirical CDF crosses the reference.
- **The reference line** — drawn on the histogram, the CDF, and each parameter panel, so every
  per-example NLL can be read against the baseline at a glance.

A well-trained estimator concentrates its distribution far below the reference with a small
worse-than-prior fraction; a large fraction, or a mean near the reference, signals that the
estimator is close to uninformative over much of the test set.

## How to run it

Preview either mode first with `--dry-run`, which resolves the input and output paths and
reports what would be read or written without computing.

Canonical artifact, resolved from the machine profile:

    MACHINE_PROFILE=<profile> python \
      Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Test_Loss_Distribution_Analysis.py \
      --total-time-seconds 2.0 [--dry-run]

Ad hoc, on a specific artifact (no machine profile required):

    python Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Test_Loss_Distribution_Analysis.py \
      --tld-path /path/to/..._Test_Loss_Distribution.npz --outdir /path/to/out

Arguments:

- `--total-time-seconds` — the recording duration that sets the `timing_label` (for example
  `2.0` → `2S_50FPS`), used to resolve this workflow's canonical artifact — the biology one,
  or the `_DETECTOR`-namespaced one — from the machine profile. Required unless `--tld-path`
  is given; it does not itself trigger any computation.
- `--tld-path` — analyze this artifact verbatim instead of resolving the canonical one. In this
  mode no machine profile is consulted, so any artifact anywhere can be analyzed.
- `--outdir` — output directory. Defaults to an `*_Analysis` directory beside the artifact.
- `--hard-quantile` — the quantile above which examples form the hard tail (default `0.95`, the
  top 5%).
- `--tail-threshold` — a fixed NLL for the worse-than-baseline fraction. Defaults to the
  computed uniform-prior reference; set it only to compare against a different fixed line.
- `--n-boot` / `--seed` — the resample count and seed for the mean's bootstrap confidence
  interval. The seed makes the interval reproducible from the stored losses; this randomness is
  a post-hoc summary of already-recorded data, unrelated to the seedless generative pipeline.
- `--dry-run` — resolve the paths and report what would be read and written; write nothing.

## Outputs

All outputs go to one directory under the machine-profile `Posit/` tier, named for the
artifact and sitting beside it (the same convention the Evaluation stage uses for its
`MAP_Recovery` report):

    <data_bank>/<posit>/<alias>_<timing_label>_Test_Loss_Distribution_Analysis/
        report.md                 — the tables below, with the figures embedded and a legend.
        figures/nll_histogram.png — P1: the NLL histogram with the mean, median, and reference.
        figures/nll_ecdf.png      — P2: the empirical CDF with the reference line.
        figures/loss_vs_theta.png — P3: per-parameter NLL-versus-value hexbins.
        figures/tail_drivers.png  — P4: the ranked per-parameter tail-driver bars.

The report opens with integrity checks (finite losses; the theta columns match the learnable
manifest rows; the array mean reproduces the manifest's recorded selection metric) and then
carries three tables:

- **Artifact provenance** — the artifact path, its production date (the file timestamp), the
  project alias, timing label, test set, best epoch, best test NLL, example and video counts,
  planned epochs, the parameter count, and the torch and artifact-format versions. Every run
  records the exact artifact it consumed.
- **Distribution spread** — the quantiles, minimum, and maximum (part A).
- **Per-parameter tail analysis** — one row per learnable parameter, ranked by the
  hard-versus-bulk KS statistic, with the Spearman correlation, both p-values (adjusted, see
  below), and the mean shift (part B).

## How to read the report

- **Mean versus median and skew.** When the median is well below the mean and the skew is
  positive, the mean is being pulled up by a thin upper tail of hard examples; the median is
  then the better summary of the typical example, and the tail is what part B dissects.
- **Information gain and the worse-than-prior fraction.** Read the mean NLL relative to the
  uniform-prior reference, not in isolation. A large information gain with a small
  worse-than-prior fraction is a concentrated, informative posterior; a mean near the reference
  or a large fraction is close to uninformative.
- **Spearman versus KS in part B.** The Spearman correlation measures a monotone trend across
  the whole test set (does higher parameter value tend to mean higher NLL?); the KS statistic
  and mean shift measure specifically whether the hard tail is drawn from a different region of
  the parameter's range than the bulk. They are complementary: a parameter can shape the tail
  (large KS) without a monotone whole-set trend (small Spearman), and vice versa, so both are
  reported and the figures show both.
- **The adjusted p-values.** Each part-B row carries two hypothesis tests, so across the
  learnable set the analysis runs many tests at once. All of them are pooled and their p-values
  Benjamini-Hochberg-adjusted, controlling the false-discovery rate over the family; read the
  adjusted values, not the raw ones, when judging significance.

## Caveats

- **The reference is scored in the density's space.** The uniform-prior NLL is computed from
  the prior-range widths in the log10 parameter space the artifact stores (`theta_space =
  log10`), matching the space the estimator's log score is evaluated in. The analysis checks
  this space assumption and states it; if a future artifact scored its density in a different
  space, the reference would have to be recomputed in that space.
- **This is a density-fit view, not a calibration certificate.** A low NLL means the posterior
  places high density on the true parameter; it does not, on its own, certify that the
  posterior's credible intervals have correct coverage. Calibration and coverage
  (simulation-based calibration, expected coverage, TARP, local C2ST) require posterior
  *samples* per example, which this NLL artifact does not carry; they are measured by the
  **`Posterior_Calibration` diagnostic** (on the held-out EVAL namespace). This matters for the
  identifiability read below: this analysis locates *where* the estimator is challenged (the
  hard tail's parameter regime), but whether that hardness is an honest identifiability limit —
  a wide but calibrated posterior there — or overconfidence is answered by
  `Posterior_Calibration`, stratified by the inferred value. Locate here, measure there.
- **The mean NLL alone does not rank two estimators.** Comparing two trained models by their
  mean test NLL conflates the (fixed) data entropy with the fit; a defensible ranking pairs the
  per-example log-score on the shared `(task, sim)` test subset (a Diebold-Mariano or signed-rank
  test). That comparison is the **`Estimator_Comparison` diagnostic**; this analysis
  characterizes one artifact and is not a model-selection verdict.
- **The bootstrap interval is a post-hoc summary.** It quantifies the sampling uncertainty of
  the mean over the recorded test examples; it is reproducible from the stored losses and is
  unrelated to the seedless generation and training pipeline.
- **Part B needs the per-example theta.** An artifact saved without `theta` still yields the
  full part-A distribution and reference, but the tail-versus-parameter analysis is skipped
  with a note in the report.

## Reference

The negative log-likelihood is the logarithmic score, a strictly proper scoring rule for a
predictive density (Gneiting and Raftery, "Strictly Proper Scoring Rules, Prediction, and
Estimation," *Journal of the American Statistical Association*, 2007). The false-discovery-rate
adjustment is the Benjamini-Hochberg procedure (Benjamini and Hochberg, "Controlling the False
Discovery Rate," *Journal of the Royal Statistical Society B*, 1995). The two-sample
Kolmogorov-Smirnov test and the Spearman rank correlation are standard. The paired predictive
comparison named in the caveats is the Diebold-Mariano test (Diebold and Mariano, "Comparing
Predictive Accuracy," *Journal of Business and Economic Statistics*, 1995); posterior
calibration by simulation-based calibration is Talts et al. ("Validating Bayesian Inference
Algorithms with Simulation-Based Calibration," 2018).

The artifact, the fixed held-out TEST set, and the imaging parameter table (roles, prior
ranges, log10 space) are described in `PROJECT_CONTEXT.md` and the Detector calibration
workflow in `DETECTOR_WORKFLOW.md`; the recovery quantification this analysis defers to is the
Detector Evaluation stage.

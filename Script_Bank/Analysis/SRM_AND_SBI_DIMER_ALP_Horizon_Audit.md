# Horizon audit — usage and interpretation

Authoritative companion for `Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Horizon_Audit.py` and its
engine `srm_and_sbi_dimer_alp/horizon_audit_runner.py` (statistics kernel:
`srm_and_sbi_dimer_alp/horizon_audit.py`). This note explains what the audit tests, how to run its
phases, and how to read its report — so the analysis can be used and understood without
reverse-engineering the code.

## The question

The estimator is trained on **independently initialized model-window simulations**: every training
video begins with freshly placed particles whose species counts are the drawn theta. The
experimental analysis, however, slices each **continuous 20 s recording** into consecutive
model-length windows and runs the estimator on every window. Equal window length does not
guarantee equal observation distributions: a later window of a continuous recording inherits the
latent state its past evolved into — species populations relaxed away from their initial values,
spatial organization, and accumulated photophysics (the imaging model's photobleaching survives
across a continuous render, exactly as it does across a real acquisition). Applying the estimator
window-by-window therefore assumes, without testing, that inherited state does not make later
windows systematically different from reset training simulations.

The audit tests that assumption **under the simulator itself**, where the truth is knowable. Both
outcomes are informative:

- **Degradation that grows with window position only in the continuous ensemble** measures the
  reset artifact — a horizon mismatch between the training factorization and the deployment
  slicing — and quantifies how much of the experimentally observed window drift the slicing alone
  can produce.
- **Equivalence within prespecified margins** bounds the tested reset mechanism under the
  implemented simulator: this mechanism, as implemented, did not reproduce the experimental
  drift. It does NOT identify the cause of the experimental drift (biological and acquisition
  mechanisms both remain candidates), and it licenses the slicing only against this mechanism.

## The controlled contrast

For each theta drawn from the training prior:

| ensemble | construction | what it represents |
|---|---|---|
| reset | R independently initialized, independently seeded model-window simulations at that theta | the training factorization — the distribution the estimator was fitted to |
| continuous | ONE uninterrupted long simulation at the same theta, rendered as one clip and sliced into consecutive non-overlapping model-length windows (the Experiment stage's stepping) | the deployment condition — windows that inherit their past |

Everything else is held identical on both sides: the theta, the estimator, and the imaging vector,
pinned for every render to the calibrated `Nuisance_DLI` + MET SCOPE vector — a **MET-conditioned,
training-supported imaging slice** (training additionally varied the SCOPE camera nuisance box, so
this pins one supported point of that box, not the whole training imaging distribution). Because
the two ensembles share everything but the inherited state, the paired within-trajectory contrast
cancels the rest.

The bundled contrast is deliberate: a continuous render carries **both** molecular/spatial memory
(the reaction state) and photophysical memory (bleaching), because that is exactly what a real
continuous acquisition carries. Two designed ablations are deferred until the primary contrast
shows an effect worth decomposing: rendering each continuous window's trajectory slice separately
(biology continuous, photophysics reset), and a **boundary-state-matched reset arm** (fresh
model-window simulations initialized at each continuous window's start populations), which would
separate changed composition from deeper spatial/photophysical memory.

## Truth references — the estimand discipline

- **Constant parameters** (diffusivities, rates): their truth is the drawn theta in every window;
  any positional structure in their errors is spurious by definition.
- **The state read (f_D)** is judged against the **window-START population**, extracted per frame
  from the trajectory. The estimator's count labels are the initial populations of its training
  windows, so the start state is what it was trained to report. The within-window mean, the
  window end, and the unrounded theta label are reported as explicitly secondary sensitivity
  references: judging against them manufactures an estimand mismatch that can dwarf the real
  error (measured on the smoke: ~36 pp fake against the window mean versus ~1 pp real against
  the start).
- Count coordinates judged against the stale t=0 label are kept as diagnostic context only — they
  read **state drift**, not estimator error, and never contribute to a verdict.

## The statistics — degradation, not bias; verdicts, not vibes

- **Primary statistic**: the paired **absolute-error** contrast, |continuous error| minus the same
  trajectory's mean |reset error|, per window position. A signed contrast tests bias and can read
  zero while accuracy collapses symmetrically; it is retained as a secondary bias read.
- **Primary estimands are predeclared**: f_D (state, vs start truth, percentage points), D_A and
  kappa_OFF (constants, vs theta, dex). Everything else is exploratory — reported, never
  verdicted (multiplicity).
- **Three-outcome verdicts against prespecified equivalence margins.** Absence of a detected
  difference is not evidence of equivalence. Defaults: the estimator's own in-model error on the
  10,000-video held-out recovery (f_D 2.8 pp, D_A 0.023 dex, kappa_OFF 0.177 dex) — a degradation
  smaller than the instrument's own error is practically negligible. CLI-overridable
  (`--margin-fd/--margin-da/--margin-koff`). Verdicts: *degraded* (CI entirely above zero),
  *equivalent* (CI entirely inside ±margin), *inconclusive* (CI too wide — supports neither).
- **First-window exchangeability gate**: continuous window 0 has no inherited past, so it must be
  statistically exchangeable with the resets before any later-window difference is attributed to
  horizon; a failure there is an arm-construction artifact, not horizon.
- **Coverage is never pooled across estimand kinds**: constants (the unidentified R_ON excluded
  from any verdict) are reported separately from f_D-vs-start (computed from the within-draw
  fraction distribution) and from the stale-truth count rows.
- **"Flow mass outside the training box"** replaces prior-exceedance language: a bounded-prior
  Bayesian posterior cannot place mass outside its prior; what is measured is the UNRESTRICTED
  neural flow leaking beyond its training support — the deployment gate the experimental analysis
  reads. The audit also reports how often the TRUE start populations themselves leave the
  training count box (state-support drift, a property of the dynamics, not the estimator).
- The **trajectory** is the independent unit everywhere; windows within one trajectory are
  correlated and never treated as replicates. All CIs are percentile bootstraps over trajectories.

## Seeds, identity, and reproducibility

- A **master seed** is fixed at `prepare` (drawn from OS entropy and persisted when not
  supplied), and every (theta, arm, replicate) gets its own placement seed and render seed via
  `numpy.random.SeedSequence` — resets are independently randomized, never copies of one stream,
  and a rerun of the same cohort reproduces the same placements and renders. ReaDDy's internal
  reaction/diffusion RNG has no exposed seed and stays OS-seeded: trajectories are not
  bit-reproducible, and the audit does not claim otherwise.
- Every cohort carries a **cohort_id** digest (the exact theta matrix plus every
  generation-shaping knob), stamped into every per-theta artifact. Generation, inference,
  and analysis verify the stamp AND the stored theta row before using any file; `prepare`
  refuses to write a new cohort into a directory still holding per-theta files. A stale artifact
  is refused, never silently mixed into a new cohort.
- Result files additionally carry the inference configuration (pool mode, draw count, estimator
  content digest); the analysis refuses a cohort whose results were inferred under mixed
  configurations.

## Estimates are posterior draws, not MAP points

Each window's estimate is a posterior summary (quantiles + the raw draw cloud): every audit
estimand is a posterior functional, and sampling is two orders of magnitude cheaper per window
than seed-then-optimize MAP, which is what makes a few-thousand-window cohort feasible. The
default sampler is `unrestricted` (direct flow sampling): it matches the experimental-baseline
methodology, never stalls, and the outside-the-box flow-mass diagnostic requires it. When
comparing the audit's positional behavior against the experimental temporal-dynamics report,
recompute the experimental summaries from the stored posterior quantiles first — its headline
tables are MAP-based, and a MAP-versus-posterior-median mismatch would masquerade as a difference.

## Running it

```bash
# 0. Self-test of the decision-critical logic (no simulation, no GPU, seconds):
MACHINE_PROFILE=<p> python Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Horizon_Audit.py \
    --total-time-seconds 2.0 --phase selftest

# 1. Draw the cohort once (master seed persisted either way):
... --phase prepare --n-theta 200 --n-resets 10 [--seed <s>]

# 2. Simulate + render (CPU). Parallel workers on any machines, each owning a
#    disjoint index range of the SAME persisted cohort:
... --phase generate --theta-start 0   --theta-stop 100
... --phase generate --theta-start 100 --theta-stop 200      # e.g. on another machine

# 3. Estimator inference per window (GPU server):
... --phase infer

# 4. Statistics + report + figures (CPU, seconds):
... --phase analyze
```

`generate`/`infer` resume for free (existing verified outputs are skipped unless `--overwrite`).
Splitting a cohort across machines means consolidating the per-theta `cohort/` files onto one
filesystem before `infer`/`analyze` (plain `rsync`; filenames are unique per index and every file
carries the cohort stamp, so a copy cannot collide or mix).

Outputs land under `<data_bank>/Posit/<alias>_<timing>_Horizon_Audit/`: the stamped cohort file,
the per-theta `cohort/theta_*.npz` (videos + population traces + seeds) and `cohort/result_*.npz`
(quantiles + draws + inference config), the aggregated audit arrays, `report.md`, and `figures/`.
Trajectory `.h5` files are deleted after their population trace is extracted unless
`--keep-trajectories` is passed.

## Reading the report

- **PRIMARY table** — per predeclared estimand: reset MAE, continuous MAE at the first and last
  window, the paired absolute-error contrast at the last window with its bootstrap CI, the
  margin, and the three-outcome verdict.
- **`first_window_exchangeable`** — the arm-construction gate (non-fatal check).
- **Exploratory table** — the remaining parameters, no verdicts; count rows labeled as state
  drift.
- **Coverage table + figure** — disaggregated: constants (excl. R_ON) vs theta; f_D vs start
  truth; counts vs the stale t=0 truth as context.
- **f_D truth-sensitivity table** — the same error under start / mean / end / unrounded-label
  references, so the estimand-mismatch hazard stays visible.
- **Dynamic-state figure** — inferred f_D against the window-START truth (mean/end as thin
  sensitivity lines) and the paired absolute contrast per position.
- **Flow-mass figure** — outside-the-box flow mass per position, both ensembles.
- **Exploratory stratification** — last-window f_D degradation by initial composition, so a
  prior-wide mean cannot hide failure in the dimer-rich region the activated experimental
  condition occupies.

Effect sizes with trajectory-bootstrap CIs are the primary read; the checks summarize them for
the report header but never replace them.

## What this audit does not do

It does not test the simulator against reality — every video here is synthetic, so a clean audit
licenses the slicing *under the implemented model* and nothing more; in particular it cannot
prove that experimental drift is biological, only whether this mechanism reproduces it. It does
not decompose molecular from photophysical memory (the designed ablations above do that when
warranted). And it does not re-validate the estimator's in-model recovery — that is the
Evaluation stage's job; the reset ensemble serves as the audit's internal baseline.

This is a post-hoc, user-driven analysis in `Script_Bank/Analysis`, NOT a canonical pipeline
stage, and it is never wired into the stage dispatcher.

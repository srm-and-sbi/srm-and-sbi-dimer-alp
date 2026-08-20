# Experiment temporal dynamics — method and interpretation

Authoritative companion for both workflows' temporal analyses:
`SRM_AND_SBI_DIMER_ALP_Experiment_Temporal_Dynamics.py` (biology, the ten reaction-diffusion
parameters) and `SRM_AND_SBI_DIMER_ALP_DETECTOR_Experiment_Temporal_Dynamics.py` (detector, the six
imaging parameters). Both are thin shims over one shared engine,
`srm_and_sbi_dimer_alp.temporal_dynamics_runner`, over the workflow-agnostic kernel
`srm_and_sbi_dimer_alp.temporal_dynamics`.

This note documents the **method and its assumptions**. Per-run numbers live in the `report.md`
each run writes beside its figures, never here, so that this note cannot go stale against a newer
posterior.

## What the analysis asks

The Experiment stage estimates the parameters independently in every non-overlapping window of every
experimental recording. Stacking those windows along time asks a question the stage cannot: **does
an inferred value hold still across the recording?** A parameter that is a constant property of the
system should be flat. A trend is **either real dynamics or an acquisition confound**, and this
analysis cannot by itself decide which — see *Two workflows, one confound test* below.

Averaging the per-window estimates over time recovers the comparable whole-recording point
estimate, now backed by demonstrated (or refuted) stationarity rather than assumed stationarity.
That is the analysis's primary contribution: a single-molecule experiment yields one estimate for a
whole recording and cannot resolve a parameter in time or per recording; this resolves it in both.

## Two workflows, one confound test

Each workflow is structurally blind to its own confound:

- **biology** holds the imaging block **fixed** at the calibrated vector, so it cannot distinguish a
  genuine change in receptor kinetics from a change in how the recording images those receptors;
- **detector** marginalizes the reaction-diffusion block, so it cannot see biological drift at all.

The two read the **same recordings** — the experimental path pattern carries no workflow qualifier,
so a given recording index is the same acquisition in both — which makes each the other's control.
Run both and compare: a flat imaging trajectory pushes a biology trend toward genuine dynamics; a
drifting one identifies a live confound and localizes it to a channel. **Neither result attributes a
cause on its own**, and no wording in either report should be read as if it did.

## The central estimate — exactly what is aggregated

Everything starts from one array: the Experiment stage's **MAP estimate per window**, the point
estimate it optimizes for one (condition, cell, chunk) window — never a posterior draw, never an
average.

A timeseries needs one vector per chunk, so the **cell** axis is aggregated. A single summary line
needs one vector overall, so the **cell and chunk** axes are. Crossing that with the choice of
estimator gives exactly four, each named for what it aggregates:

| name | definition | drawn as |
|---|---|---|
| `mean-window` | Mean value vector, aggregated **across cells** for a given chunk, of the MAP estimate vectors computed for (chunk, cell) pairs. Each parameter is averaged independently. | timeseries |
| `sgm-window` | Realized value vector, aggregated **across cells** for a given chunk, minimizing the summed normalized distance to the other cells' vectors at that chunk, of the MAP estimate vectors computed for (chunk, cell) pairs. | timeseries |
| `mean-trajectory` | Mean value vector, aggregated **across chunks and cells**, of the MAP estimate vectors computed for (chunk, cell) pairs. | horizontal line |
| `sgm-trajectory` | Realized value vector, aggregated **across chunks and cells**, minimizing the summed normalized distance to all other (chunk, cell) vectors. | horizontal line |

`--central` selects the **family**, and the pairing is enforced: `sgm` draws the `sgm-window`
timeseries with the `sgm-trajectory` line, `mean` draws `mean-window` with `mean-trajectory`. A
figure therefore never mixes a mean with a medoid. `mean-trajectory` is the historical grand mean,
preserved under a name that says what it is.

**Distance metric** (both `sgm-*`): absolute values `10**theta`, each parameter divided by its
absolute prior width `10**high - 10**low` so no parameter dominates, Euclidean, **exact medoid** —
the member minimizing the summed distance to every other member of the set. No iteration and no
synthetic point.

Three properties worth stating plainly:

- A `mean-*` estimate averages each parameter independently, so its coordinates need not have
  co-occurred in any recording. An `sgm-*` estimate is an actual window, so its coordinates did.
- **`sgm-window` can select a different cell at different chunks** (measured: several distinct cells
  across the chunks of a single run), so a step between adjacent chunks can be a change of cell
  rather than a change in time. The selected cells are listed in the report and must be read with
  the curve. `sgm-trajectory` is a single window and carries no such ambiguity.
- All `sgm-*` estimates select on **all parameters jointly**, so the value drawn for one parameter
  is that jointly-central window's value, not that parameter's own median. That is the point — the
  vector is internally coherent — but it means a curve can sit away from its parameter's middle.

`sgm-trajectory` is the same quantity the standalone sample-geometric-median analysis reports over
the same pooled windows, so the two analyses agree by construction rather than by coincidence.

**The choice is not cosmetic.** On the current experimental data the two families differ by factors
from about 1.05 to 4 depending on the parameter, so which one is reported changes the number a
reader takes away — and only one of them is a configuration any recording actually produced.

## Drift — measured per cell, independent of the display

For every (condition, cell, parameter) an ordinary least-squares line is fit to the stored **log10**
MAP estimate against time — log10 because drift is multiplicative — giving a slope and hence fitted
endpoints `change_dex = slope * (t_last - t_first)`, `start`, and `end` in absolute units. Because
the fit is per cell, **none of these statistics depends on the central estimate the figures draw.**

| name | definition | where |
|---|---|---|
| `drift-absolute` | Median across cells of `end - start`, in the parameter's own units | figure + table |
| `drift-sign-consistency` | Fraction of cells whose change shares the sign of the median change | figure + table |
| `drift-fold` | Median across cells of `end / start`, a multiplicative factor | table |
| `drift-dex` | Median across cells of `change_dex`, in log10 units | table |
| `drift-material-fraction` | Fraction of cells whose `\|change_dex\|` exceeds 0.3 dex (a factor of two) | table |
| `drift-wilcoxon-p` | Two-sided signed-rank test that the per-cell changes are centered at zero — a **detectability** statement, not a magnitude | table |
| `reference-ratio` | The `*-trajectory` value divided by the reference mean, where a reference applies to that condition | figure + table |
| `reference-verdict` | Whether the `*-trajectory` value lies inside the reference band | table |

Fits happen in log10; results are reported in absolute units because that is what reads. Both facts
are restated in every generated report so no one has to infer which space produced which number.

**A time-aggregated summary of a drifting parameter summarizes a non-stationary process.** Where
`drift-absolute` is large and `drift-sign-consistency` high, the `*-trajectory` line — and any
reference comparison drawn against it — is a summary over that drift, not a measurement of a
constant.

## Uncertainty figure

`<key>_temporal_posterior.png` shows one quantity: at each chunk, the **median across cells of that
window's stored posterior interval** — how uncertain a typical single window's estimate is — with the
`*-window` timeseries drawn on top as the anchor.

**What it is built from, and what it is not.** The stored per-window five-quantile summary
(5, 25, 50, 75, 95 %). It therefore shows interval **widths**; it is **not** a posterior **density**
and does not approximate one. The density is a separate figure, described next.

When a parameter's interval spans essentially its whole prior, the posterior is returning the prior
and the point estimate is the prior's center regardless of the data — a direct picture of
non-identifiability.

## Pooled posterior density — the classic histogram, time collapsed

`<key>_temporal_posterior_pooled.png` answers the question the two time-course figures cannot:
across the whole recording, what does the posterior actually look like, and how much narrower is it
than the prior it started from? The name places it with the other posterior figure, since both
describe posterior uncertainty and differ only in what they collapse: `_temporal_posterior` keeps
time and shows an interval width, `_temporal_posterior_pooled` collapses time and shows a density.

**The arithmetic, stated exactly.** The Experiment stage draws `--posterior-samples` (1000 by
default) samples from the posterior of every `(recording, window)` pair. For one condition that is

```
25 recordings  x  10 windows  x  1000 draws  =  250,000 draws
```

and **all of them enter one histogram with equal weight**. Nothing is averaged, selected, or
re-weighted first; the time axis is collapsed by pooling, not by summarizing.

**What that mixture means — and what it does not.** Equal-weight pooling produces the **mixture** of
the per-window posteriors. Its density answers: *for a single window drawn at random from this
condition, which values are consistent with it?* It is **not** a joint posterior for the condition.
Combining independent observations under Bayes multiplies their likelihoods; pooling draws **adds**
their densities. The mixture is therefore as wide as the between-window spread **plus** the
within-window uncertainty, whereas a genuine joint posterior over 250 windows would be far
**narrower** than any single one of them. Read the width as a property of the window population,
never as evidence accumulated over the recording.

**It is not the central estimate.** Because the mixture is a population rather than an estimate, its
mode and median are *not* this analysis's central estimate. That remains `<family>-trajectory` — one
jointly realized vector — marked on the figure as a thin vertical line. A per-parameter summary of
pooled marginals is exactly the composite the SGM exists to avoid, so the two are shown together
and are not expected to coincide.

**Axis: absolute units, two spacings.** The x axis is always in the parameter's own **absolute
units**, with plain numeric ticks — never powers of ten, never dex. `--pooled-scale` chooses how that
axis is spaced, and the binning, the density's unit, and the prior's shape all follow from that one
choice consistently:

| `--pooled-scale` | axis and bins | height is a density | the prior is drawn as | file |
|---|---|---|---|---|
| `linear` (default) | uniform in absolute units | **per unit** | the curve `1 / ((log10 hi - log10 lo) · x · ln 10)` | `<key>_temporal_posterior_pooled.png` |
| `log` | uniform in decades | **per decade** | the flat line `1 / (log10 hi - log10 lo)` | `<key>_temporal_posterior_pooled_log.png` |
| `both` | — | — | — | writes both files |

The spacing is part of the filename, so the two views coexist and a run with one setting cannot
silently overwrite the other. Both use 64 bins — a fixed count, because with hundreds of thousands
of draws the resolution is a legibility choice rather than a sample-size limit, and holding it fixed
keeps conditions and parameters comparable.

`linear` is the default because it matches the value axis of the time-course figures and reads
directly in the parameter's units, which is what a reader wants when absolute differences are the
question.

**When to use `log` instead.** A linear axis spends its width in proportion to absolute value, so
for a parameter supported over orders of magnitude it compresses the low end into a few bins. The
three counts span 1–316: structure below ~20 receptors that is plainly visible under `log` spacing
collapses into the first two bins under `linear`. Whenever the low end of a wide-ranged parameter is
the question — as it is for any statement about few-receptor cells — read the `_log` file, or pass
`--pooled-scale both` and read them together.

**Why the prior is not a flat line under `linear` spacing.** A log-uniform density is flat in
`y = log10(x)`, not in `x`. Changing variable gives `p(x) = p(y) · |dy/dx| = 1 / ((hi - lo) · x ·
ln 10)`, a `1/x` curve, and that curve is what the figure draws. Drawing a flat line there would
misread the Jacobian as evidence, making every parameter appear to have had its low end disfavored
by the data when nothing of the kind happened. Under `linear` spacing the y limit is set by the
histogram rather than by the prior, because the `1/x` curve diverges toward the lower limit and would
otherwise flatten the posterior into the axis; the prior curve simply leaves the top of the frame
where it exceeds the data.

Either way, the gap between a curve and the prior **is** what the data added, and a density lying on
the prior has learned nothing — the same non-identifiability the interval figure shows, in the form
that makes it quantitative.

**Availability.** This figure needs the raw draws, which the Experiment stage stores only when run
with `--dump-posterior-samples`. Without them the analysis prints `pooled density: off` and writes
every other output as usual.

## Axis and rendering conventions

**Linear, absolute units.** The y-axis carries the parameter's own absolute units — the readable
quantities are the value and its absolute change. Decade-space quantities (`drift-dex`,
`drift-fold`) live in the report table instead, so the axis carries one scale and no reader has to
translate. Limits come from a robust percentile range of the data extended by the reference band, so
a few outlying per-recording traces clip rather than compressing the informative region.

**Steps, not lines.** Each estimate summarizes a *window*, not an instant, so every series is drawn
as a piecewise constant held from its window's start to its end. Straight lines between chunk points
would draw an interpolation the analysis never computed. The time axis therefore spans the
recording's true extent — ten 2 s windows reach **20 s**, not 18 — and the same rendering applies to
the per-recording traces and to the posterior bands.

**Windows are fit at their centres.** The drift regression uses each window's centre time, which is
where its information sits. The fitted slope is identical to using window starts, so every drift
statistic is unaffected; only the reported endpoints shift by half a window.

**Numbers live in the report, not on the axes.** The figure carries the reference label with its
bounds and the `*-trajectory` value with its ratio to the reference. The drift statistics are
tabulated in `report.md` rather than annotated on the plot, where they added clutter without helping
anyone read the data.

## External reference values are scoped

A reference is drawn only on the parameters it legitimately constrains, and only for the conditions
it applies to. Drawing a reference against a condition it does not describe would invite a false
comparison, so the scope is enforced in the code and restated in every report.

**Biology** — Li et al., *Small* **2026**, 22, e07115 (single-molecule FRET tracking of
InlB-activated MET):

| parameter | reference | scope | derivation |
|---|---|---|---|
| κ_OFF (dissociation rate) | 1/τ per FRET variant: 1/1.30 = 0.77, 1/0.80 = 1.25 s⁻¹ (mean 1.01) | **MET-INLB only** | τ = 1.30 ± 0.05 s and 0.80 ± 0.03 s; SD propagated as dτ/τ². The study activated MET with InlB |
| D_A (monomer diffusivity) | 0.109 ± 0.068 and 0.093 ± 0.053 µm²/s (mean 0.10) | **both conditions** | donor-only segments; monomer diffusion is ligand-independent |
| R_B (rel. mobile-dimer diffusivity) | 0.066/0.109 = 0.61 and 0.056/0.093 = 0.60 (mean 0.60) | **MET-INLB only** | derived from the InlB-activated measurement; consistent with "dimer ≈ 1.6× slower" |

**Detector** — ThunderSTORM localization fits on the same public accession `S-BSST712`, documented
in `DETECTOR_WORKFLOW.md` §6.2/§6.3/§6.5:

| parameter | reference | scope | caveat that fixes the scope |
|---|---|---|---|
| `mu_r` | 1.36 px | **MET-FAB only** | the InlB value 1.47 is **dimer-broadened** — two labels in one diffraction-limited spot — so it is not a reference for the per-emitter PSF the model infers |
| `mu_pc` | 386 photons | **MET-FAB only** | MET-INLB's 690 is a **per-detection sum**: a dimer's two labels are reported as one detection whose photons add. It is also a lower bound, since 23.7 % of InlB localizations pile up at the 1225-photon acceptance ceiling |
| `sigma_r` | ≈ 0.15 fit-corrected, with 0.37 drawn as an **upper bound** | MET-FAB only | the fitted log-spread is **upper-biased**: each per-spot width is itself a noisy fit, and the variance of noisy estimates is the true variance plus the fitting-error variance |
| `sigma_pc` | ≈ 0.5 fit-corrected, with 0.61 as an **upper bound** | MET-FAB only | the same errors-in-variables inflation |
| `lambda_rate` | ≈ 5 s⁻¹ | **both conditions** | from the flicker correlation time of the track `intensity[photon]` series (τ_corr ≈ 0.13 s), a photophysical quantity and hence condition-independent |
| `prob_photo_bleach` | **none** | — | no public anchor exists; it is read on its internal evidence alone |

Upper bounds are drawn as bounds, not targets: a calibration is expected to land **below** them.

## How to read a parameter

Trust a parameter on experimental data when it satisfies three independent checks:

1. **it recovers well on held-out synthetic data** — annotated on each figure where a recovery
   artifact exists for the workflow, as the fraction landing inside two nested tolerances of the
   truth: `[0.50x, 2.00x]` and `[0.71x, 1.41x]`. These are the multiplicative ranges of the log10
   half-widths ±0.3 and ±0.15 dex that the Evaluation stage reports — a factor of two and a factor
   of the square root of two — restated as the value range they permit, because that is the form in
   which a reader can judge whether a parameter is usable;
2. **it is stationary where the model says it should be** — flat when the underlying property is a
   constant of the system;
3. **its posterior is narrow relative to its prior** — panel (a) of the uncertainty figure.

A parameter that recovers poorly carries no signal regardless of how smooth its trajectory looks, and
a posterior that spans its prior is reporting the prior back. Conversely, a parameter that drifts is
not thereby discredited: the drift may be the acquisition's, which is exactly what the other
workflow's run is for.

## Outputs

Written to `<data_bank>/<posit>/<alias>_<timing_label>_MAP_Experiment/temporal_dynamics/`, where
`<alias>` carries the `_DETECTOR` qualifier for that workflow so the two never collide:

- `<key>_temporal.png` — the central-trajectory figure per parameter;
- `<key>_temporal_posterior.png` — the within-window interval figure (written only when the
  Experiment output carries `posterior_quantiles`);
- `<key>_temporal_posterior_pooled.png` — the pooled posterior density with the prior drawn on the
  same axes (written only when the Experiment output carries `posterior_samples_cloud`, i.e. the
  stage ran with `--dump-posterior-samples`);
- `report.md` — the per-run interpretation: the drift table with all four statistics, the selected
  central recordings for both SGM variants, the pooled-mixture quantile table when the draws are
  present, the reference scoping, and the method's stated limits.

## How to run

```bash
# biology — the ten reaction-diffusion parameters
MACHINE_PROFILE=<profile> python \
    Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Experiment_Temporal_Dynamics.py \
    --total-time-seconds 2.0 [--params rate_dissociation,diffusivity_alp] [--dry-run]

# detector — the six imaging parameters, on the same recordings
MACHINE_PROFILE=<profile> python \
    Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Experiment_Temporal_Dynamics.py \
    --total-time-seconds 2.0 [--params prob_photo_bleach,mu_pc] [--dry-run]
```

- `--total-time-seconds` (required) — the run duration; selects the Experiment output via its
  timing label (e.g. `2.0` → `2S_50FPS`) and must match a completed Experiment run.
- `--chunk-step-seconds` — spacing between consecutive windows on the time axis; defaults to the run
  duration (non-overlapping windows). Set it only if the Experiment run used overlapping windows, in
  which case the true spacing is the step.
- `--central {sgm,mean}` — the central-estimate family; `sgm` (default) draws `sgm-window` with
  `sgm-trajectory`, `mean` draws `mean-window` with `mean-trajectory`. The pairing is enforced.
- `--params` — comma-separated parameter keys to plot. Filters the **figures only**: the central
  estimates are computed on the full parameter vector either way.
- `--pooled-scale {linear,log,both}` — x-axis spacing for the pooled posterior histogram; both
  spacings label the axis in absolute units. See the table above for what each implies about
  binning, the density's unit, the prior's shape, and which file it writes.
- `--dry-run` — resolve and print the inputs and outputs, then exit without reading data or writing
  anything.

CPU only, seconds to run: it reads the completed Experiment output and neither loads the estimator
nor renders videos. A post-hoc, user-driven analysis — never wired into the stage dispatcher.

## Reference

Y. Li, M. S. Dietz, H.-D. Barth, H. H. Niemann, M. Heilemann, "Single-Molecule FRET-Tracking of
InlB-Activated MET Receptors in Living Cells," *Small* **2026**, 22, e07115.
doi:10.1002/smll.202507115.

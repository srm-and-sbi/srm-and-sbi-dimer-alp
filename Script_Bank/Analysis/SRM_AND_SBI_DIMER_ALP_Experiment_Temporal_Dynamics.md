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

## The central estimate is a real recording, not an average

Summarizing many recordings at one time point by averaging each parameter independently composes a
vector whose coordinates never co-occurred in any recording — the defect the Sample Geometric
Median exists to remove. Distances are taken in **absolute (physical) space normalized by the
absolute prior range**: the space the simulator consumes, with each dimension divided by its prior
width so no parameter dominates. Three curves are drawn per condition:

| curve | what it is | what to watch |
|---|---|---|
| **SGM trajectory** (thick, headline) | the medoid over recordings of the flattened (time × parameter) course — **one real recording for all time points** | nothing on this curve is composed, and **no step can be an artifact of switching recordings** |
| **per-time-point SGM** (dashed, companion) | the medoid recording at each time independently | coordinates are jointly realized within a time point, but **the selected recording can change between time points**, and such a change can look like temporal structure |
| **cross-recording mean** (dotted, comparison) | the per-dimension composite the SGM replaces | where it separates from the SGM trajectory it is asserting a combination no recording produced |

The per-time-point switching is not hypothetical: in practice this variant selects **several distinct
recordings across the time points of a single run**. Its selected recordings are therefore printed in
the figure legend and listed in the report, so the switching is visible wherever the curve is shown.

Two consequences worth stating plainly. First, the SGM trajectory is typically **noisier** than the
mean — a single recording's window-to-window estimates fluctuate, whereas an average smooths. The
mean's smoothness is an artifact of averaging, not evidence of a smooth underlying process. Second,
both SGM variants are computed on the **full parameter vector**, never on a plotted subset, so
`--params` filters the figures only and cannot change which recording is central.

## Drift is measured per recording, independently of the display

For every (condition, recording, parameter) an ordinary least-squares line is fit to the stored
log10 estimate against time and reported as the **total change over the observed span, in dex**.
Working in log10 makes the drift multiplicative and comparable across parameters of different units.
Per-recording fits are then aggregated by their **median**, so one erratic recording cannot move the
summary, and reported with:

- the **fraction of recordings whose |drift| exceeds 0.3 dex** — the same practical bar the recovery
  tables use for a factor of two, so exceeding it means the estimate moves by more than the
  tolerance the recovery is judged against;
- the **sign-consistency**, the fraction of recordings sharing the median's direction;
- a two-sided **Wilcoxon signed-rank** test that the per-recording drifts are centered at zero.

A large drift with high sign-consistency and a small p-value is a coherent within-recording trend,
not scatter. Because the fit is per recording, **these statistics do not depend on the choice of
central estimate**: swapping an average for a geometric median changes what is displayed, not what
is measured.

## Uncertainty figures — two quantities, deliberately not combined

Each `<key>_temporal_posterior.png` carries two panels sharing one axis:

- **(a) within-window posterior spread** — at each time, the median across recordings of that
  window's stored posterior interval. How uncertain a typical *single* window's estimate is.
- **(b) between-cell spread** — at each time, percentiles across recordings of the per-window
  median. How much recordings differ from each other: biological and experimental heterogeneity.

They are separated because plotting them together would let a wide posterior masquerade as
heterogeneity, or heterogeneity as posterior width. In practice the two differ substantially, so the
separation is load-bearing rather than cosmetic.

**What these panels are built from — and what they are not.** The stored per-window five-quantile
summary (5, 25, 50, 75, 95 %), which is what the Experiment stage persists. They therefore show
interval **widths**; they are **not** a posterior **density**. A pooled density across windows — one
distribution per (condition, parameter) built from the full posterior sample clouds — remains
unimplemented, because the stage does not persist the per-window samples and five quantiles cannot
be re-pooled into that distribution. Adding it requires either persisting the per-window sample pool
in the Experiment stage or a dedicated resampling pass; this analysis does not approximate it.

A diagnostic worth knowing: when a parameter's within-window interval spans essentially its **whole
prior** while its between-cell spread is nearly flat, the posterior is returning the prior and the
point estimate is the prior's center regardless of the data. That is a direct picture of
non-identifiability, and it is visible at a glance in panel (a).

## Axis convention

The y-axis is **log-scaled in physical units** with a mirrored right-hand axis labeled in **log10
(dex)**, so one figure carries both readings: the visual spacing is dex — the space the priors are
declared in and the space drift is measured in — while the left labels stay in the unit the
simulator consumes. Right-hand ticks are placed explicitly at half-dex positions, because composing
a log10 transform with an already-logarithmic parent scale would apply the transform twice.

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
   artifact exists for the workflow;
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
- `<key>_temporal_posterior.png` — the two-panel uncertainty figure (written only when the
  Experiment output carries `posterior_quantiles`);
- `report.md` — the per-run interpretation: the drift table with all four statistics, the selected
  central recordings for both SGM variants, the reference scoping, and the method's stated limits.

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
- `--params` — comma-separated parameter keys to plot. Filters the **figures only**: the central
  estimates are computed on the full parameter vector either way.
- `--dry-run` — resolve and print the inputs and outputs, then exit without reading data or writing
  anything.

CPU only, seconds to run: it reads the completed Experiment output and neither loads the estimator
nor renders videos. A post-hoc, user-driven analysis — never wired into the stage dispatcher.

## Reference

Y. Li, M. S. Dietz, H.-D. Barth, H. H. Niemann, M. Heilemann, "Single-Molecule FRET-Tracking of
InlB-Activated MET Receptors in Living Cells," *Small* **2026**, 22, e07115.
doi:10.1002/smll.202507115.

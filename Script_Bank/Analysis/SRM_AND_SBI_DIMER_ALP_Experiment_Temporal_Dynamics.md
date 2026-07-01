# Experiment temporal-dynamics analysis — interpretation

Companion to `SRM_AND_SBI_DIMER_ALP_Experiment_Temporal_Dynamics.py`. The script
tracks each inferred parameter over the real MET single-particle-tracking
recordings; this note explains what the figures mean and how to read them, so the
analysis can be understood and reused without reverse-engineering the code.

## What the figures show

One figure per learnable parameter, written to
`<…>_MAP_Experiment/temporal_dynamics/<key>_temporal.png`. Each real 20 s recording
is split into short, **non-overlapping** windows (ten 2 s windows for the 2 s model,
four 5 s windows for the 5 s model). The parameter is MAP-estimated independently in
every window of every cell, and each figure shows, per experimental condition
(**MET-FAB** = ALP, blue; **MET-INLB** = InlB, orange):

- **solid line** — the mean, across cells, of the MAP estimate at each time point;
- **shaded band** — mean ± 1 SD across cells (between-cell spread);
- **faint dotted lines** — each cell's own MAP trajectory (the population the mean
  summarizes);
- for parameters with an experimental reference, a **dashed line per condition** at
  that condition's time-averaged MAP, and a **grey line** at the experimental value.

Values are in absolute (linear) units; the relative parameters (R_B, R_C, R_ON) are
shown as the dimensionless ratios the model samples.

## How to read them — two readings at once

**1. Temporal dynamics (the primary purpose).** Estimating a parameter in each short
window and plotting it against the window's position in the recording shows how the
inferred value behaves over time — something a single whole-recording estimate cannot
reveal.

**2. Robustness / stationarity (a parameter-dependent corollary).** Several
parameters — the kinetic rates in particular — are constant properties of the system:
their true value does not change within one recording. For those, a **flat** trajectory
is positive evidence that the estimator is time-invariant and self-consistent (it
returns the same answer regardless of which window it sees). A systematic **trend**
means either genuine dynamics or an acquisition confound. Which of the two applies is
judged per parameter — so the same figure set doubles as a robustness diagnostic where
constancy is expected, and as a dynamics readout where it is not.

## Why this exceeds the experimental readout

A single-molecule experiment (Li et al. 2026, below) yields one estimate plus a
distribution for the **whole** recording; it cannot resolve a parameter in time or per
cell. This inference resolves it per short window **and** per cell — temporal and
single-cell granularity the experiment cannot reach. Averaging the per-window MAP
estimates over time (and cells) recovers the comparable whole-recording point estimate,
now backed by the demonstrated (or refuted) stationarity rather than assumed.

## Validation against experiment (Li et al. 2026)

The dissociation rate is the inverse of the (MET:InlB)₂ dimer lifetime
(κ_OFF = 1 / τ). Li et al. measured that lifetime at **≈ 1 s** (1.30 ± 0.05 s and
0.80 ± 0.03 s for the two FRET-label variants), i.e. **κ_OFF ≈ 1.0 s⁻¹** — which is
also the simulation ground truth. The check is simply that the **time-averaged MAP
lands close to this value**: the MET-INLB time average is **0.82 s⁻¹**, close to the
experimental ≈ 1.0 s⁻¹ — and MET-INLB is exactly the condition the paper measured. (The
MET-FAB average, 1.41 s⁻¹, has no counterpart in that study, which measured only InlB.)

The same paper supplies two further references the script draws where available:

| Parameter | Experimental estimate (Li et al. 2026) | Source |
|---|---|---|
| κ_OFF (dissociation rate) | ≈ 1.0 s⁻¹ | 1 / dimer lifetime (τ ≈ 1 s) |
| D_A (monomer diffusivity) | ≈ 0.10 µm²/s | donor-only segments, 0.109 / 0.093 µm²/s |
| R_B (rel. mobile-dimer diffusivity) | ≈ 0.6 | dimer diffuses ≈ 1.6× slower than monomer |

Because that study activated MET with InlB, κ_OFF and R_B compare most directly to the
MET-INLB condition; the monomer diffusivity D_A is ligand-independent and applies to
both. Notably, the inferred D_A (≈ 0.10–0.11 µm²/s) matches the experimental monomer
diffusivity closely — an independent, cross-parameter agreement.

## Reliability: recovery × stationarity

A parameter is trustworthy on real data when it satisfies **two** independent checks:

1. **Recovers well on ground truth** — from the Evaluation (MAP-recovery) stage on
   simulated data with known parameters. Each figure is annotated with the fraction of
   held-out EVAL videos recovered within ±0.3 log10.
2. **Is stationary over time** — flat where the underlying property is constant.

κ_OFF passes both (≈ 95% recovery within ±0.3; MET-INLB flat near the experimental
value). At the other extreme, the **relative dimerization rate R_ON is not
identifiable** from these videos (≈ 35% recovery, R² ≈ 0), so its temporal trajectory
carries no signal and must not be over-read regardless of how it looks. The recovery
annotation makes this explicit on every figure.

## Caveats

- **Photobleaching.** The initial-count parameters (C_A, C_B, C_C) drift downward over
  the recording. This is the signature of fluorophores bleaching — fewer visible
  emitters in later windows — i.e. a real non-stationarity in a *confounded* parameter,
  not a loss of receptors. It is the analysis correctly exposing which parameters are
  time-robust (the rates) and which are acquisition-limited (the counts).
- **First-pass posteriors.** The current 2 s / 5 s posteriors come from interrupted
  training; the absolute values will sharpen with the full production posteriors.
  Re-run this analysis on those for the definitive numbers.
- **Relative parameters** are plotted as dimensionless ratios (their physical value
  would multiply by the reference D_A / rate).

## Not yet implemented: aggregated posterior distributions

The historical figures also included, per parameter, a pooled **posterior
distribution** panel: one histogram per condition of the full posterior sample cloud
pooled across every cell and every chunk of the experiment. Reproducing it faithfully
needs the per-window posterior **samples**, aggregated across all chunks into a single
distribution per (condition, parameter). The current Experiment stage persists only
five quantiles per window, not the sample pool, so the quantiles alone cannot be
re-pooled into that experiment-wide distribution. To add it: draw posterior samples per
(cell, chunk) window from the trained posterior, concatenate across all windows of a
condition, and histogram the pooled samples in log10. This requires either persisting
the per-window sample pool in the Experiment stage or a dedicated resampling pass, and
is left as a documented extension.

## Reference

Y. Li, M. S. Dietz, H.-D. Barth, H. H. Niemann, M. Heilemann, "Single-Molecule
FRET-Tracking of InlB-Activated MET Receptors in Living Cells," *Small* **2026**, 22,
e07115. doi:10.1002/smll.202507115.

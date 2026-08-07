# Control-receptor temporal-dynamics analysis — interpretation

Companion to `SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls_Temporal_Dynamics.py`.
The script applies the DIMER-ALP posterior — trained on the MET single-particle-tracking
regime — to two control receptors and tracks each inferred parameter over their
recordings; this note explains what the figures mean, how to read them, and why only
the diffusion readout is interpreted quantitatively. It is written so the analysis can
be understood and reused without reverse-engineering the code.

This is a special-scope, ad-hoc reuse of a trained posterior on data from a different
study. It is not one of
the canonical pipeline stages and is kept out of the stage dispatcher.

## The controls, and why diffusion is the readout

CD86 is a constitutive monomer and CTLA-4 a constitutive dimer. They are
oligomeric-state controls: a known-monomer and a known-dimer whose mobile diffusion
coefficients have been measured independently, so they **bracket the mobile-diffusion
scale** (a monomer diffuses faster than a dimer). Applying the posterior to them tests
whether its transferable observable — the diffusion coefficient — lands on the measured
values.

Only the diffusion coefficient is read quantitatively, because of a deliberate
**label / model mismatch**: the posterior was trained on always-visible permanent-label
emitters, whereas the control recordings use an exchangeable, blinking SiR-S5 HaloTag
probe. The diffusion coefficient is a per-track property, read from the displacement
statistics of a molecule *while it is visible*, so it is insensitive to how many
molecules are lit at once and transfers across the mismatch. Counts and rates depend on
the number of co-visible emitters and on track continuity — both corrupted by blinking —
so they are not interpreted here. The controls are also constitutive states, not the
dynamic A + A ⇌ B ⇌ C dimerization mechanism the posterior encodes; they bracket the
diffusion scale and stress-test transferability, they do not exercise the kinetic model.

## The headline figure — mobile mixture diffusivity

`dmix_mobile_temporal.png`. The posterior cannot reliably separate monomer from dimer
for these constitutive controls, so the headline observable is the count-weighted mean
diffusivity of the **mobile** populations (monomer A + mobile dimer B), excluding the
immobile class C:

    D_mix_mobile = (C_A · D_A + C_B · D_B) / (C_A + C_B)     [µm²/s]

with D_A the monomer diffusivity, D_B = R_B · D_A the mobile-dimer diffusivity, and
C_A, C_B the mobile counts. The figure is a clean per-receptor comparison — every line
is solid, and the four series are told apart by color alone:

- **inferred D_mix_mobile**, in the condition color (CD86 blue, CTLA-4 orange) — a
  trajectory (mean over cells) with a mean ± 1 SD between-cell band;
- **experimental D_mobile**, in a separate per-condition color (CD86 green, CTLA-4 red)
  — a flat reference line with a value ± SD band. Its own colors keep experiment and
  inference from ever sharing a color, while still telling the two references apart
  (there are two here, one per receptor, unlike the single MET reference).

D_A, D_B and the mobile split f_B are reported in the run's report table, not drawn on
the figure, so the headline stays an uncluttered inferred-vs-experiment comparison.

D_mix_mobile lies between D_A and D_B, in either order. On these out-of-distribution
controls the model's monomer / dimer (A / B) assignment is not physically constrained —
the relative dimer diffusivity R_B can exceed 1 (so D_B > D_A), and under unrestricted
pooling the mode can even leave the training prior (the report flags any condition where
D_A or R_B does so). D_mix is therefore read as a point between D_A and D_B, **not** as a
resolved monomer / dimer mixture, unless the mobile split f_B = C_B / (C_A + C_B) has an
informative (non-edge) posterior — the split is count-weighted, and the counts are the
label-fragile quantity, so a collapse of f_B toward 0 or 1 pushes D_mix toward one
endpoint. Reporting D_A, D_B, and f_B alongside D_mix keeps that dependence visible, and
this robustness to the A / B mis-assignment is exactly why D_mix (not D_A or D_B alone)
is the readout. The per-run values are
tabulated in `report.md`.

## Units and why the comparison is quantitative

D_A is stored directly in µm²/s (the posterior samples log10(D_A); the absolute value
is 10^θ), and D_B inherits those units through the dimensionless ratio R_B — so
D_mix_mobile is in µm²/s and is directly comparable to the experimental D_mobile with
**no unit conversion**. The comparison is quantitative because the acquisition geometry
matches the training regime: 50 FPS, 256 × 256 pixels, ~157 nm/px. The comparison is
between **ensemble means**: D_mix_mobile is the abundance-weighted mean of a
two-component mobile distribution, and the experimental D_mobile is a single
mobile-fraction coefficient pooled over the mobile tracks; a bimodal mobile population
has a mean between its modes, matching neither individually.

## The per-parameter figures — two readings at once

The script also emits one `<key>_temporal.png` per learnable parameter (counts, rates,
ratios), with the same solid / band / faint-line grammar. These are diagnostic, not
quantitative comparisons (no experimental band is drawn on them):

**1. Temporal dynamics.** Estimating a parameter in each short window and plotting it
against the window's position in the recording shows how the inferred value behaves over
time — something a single whole-recording estimate cannot reveal.

**2. Robustness / stationarity.** A parameter that is a constant property of the system
should be flat over time; a flat trajectory is positive evidence that the estimator is
time-invariant and self-consistent. A systematic trend means either genuine dynamics or
an acquisition confound. For the counts in particular, a downward drift over the
recording is the expected signature of blinking / photobleaching (fewer visible emitters
in later windows) — a real non-stationarity in a confounded parameter, which is why the
counts are not read quantitatively.

## Why this exceeds a single whole-recording readout

A whole-recording measurement yields one estimate plus a distribution for the entire
recording; it cannot resolve a parameter in time or per cell. This inference resolves it
per short window **and** per cell. Averaging the per-window MAP estimates over time and
cells recovers the comparable whole-recording point estimate — the D_mix_mobile
time-average — now backed by the demonstrated (or refuted) stationarity of the
underlying diffusivities rather than assumed.

## Reliability: recovery × stationarity

Recovery quality is a property of the posterior, independent of which real data it is
applied to, so it carries over to this reuse. Each figure is annotated (when the
Evaluation MAP-recovery arrays are present) with the fraction of held-out EVAL videos
recovered within ±0.3 log10. The monomer diffusivity D_A recovers well and is the
backbone of the headline readout; the relative dimerization rate R_ON is not
identifiable from these videos and its trajectory must not be over-read. Trust a
parameter when it both recovers well on ground truth and is stationary where it should
be.

## Caveats

- **Label / model mismatch.** Exchangeable, blinking SiR-S5 vs the always-visible
  permanent-label training regime. Diffusion transfers; counts and rates (C_*, κ_*,
  R_ON) do not and are not read quantitatively.
- **Not the trained mechanism.** CD86 and CTLA-4 are constitutive monomer / dimer
  controls, not the dynamic A + A ⇌ B ⇌ C mechanism the posterior encodes. They bracket
  the diffusion scale and stress-test transferability; they do not exercise the kinetics.
- **D_mix weights are the fragile quantity.** D_mix is count-weighted, and the counts
  are label-fragile; read D_mix between D_A and D_B (in either order), not as a resolved
  mixture, unless f_B is informative.
- **Prior leakage under unrestricted pooling.** On these out-of-distribution controls
  the MAP can leave the training prior — R_B above 1, or D_A past its ceiling — because
  unrestricted pooling does not enforce the prior box. The report flags any such
  condition. D_A / D_B are then not physically constrained, but the count-weighted D_mix,
  which does not depend on the A / B ordering, stays the robust readout.
- **Blinking / photobleaching** drives the downward drift of the count parameters over
  the recording — a real non-stationarity in a confounded parameter, not receptor loss.
- **First-pass posteriors.** The current 2 s / 5 s posteriors come from interrupted
  training; the absolute values will sharpen with the full production posteriors. Re-run
  this analysis on those for the definitive numbers.
- **Relative parameters** (R_B, R_C, R_ON) are plotted as dimensionless ratios.

## Not yet implemented: aggregated posterior distributions

The historical figures also included, per parameter, a pooled **posterior distribution**
panel: one histogram per condition of the full posterior sample cloud pooled across every
cell and every chunk. Reproducing it faithfully needs the per-window posterior
**samples**, aggregated across all windows into a single distribution per (condition,
parameter). The current Experiment stage persists only five quantiles per window, not the
sample pool, so the quantiles alone cannot be re-pooled into that experiment-wide
distribution. To add it: draw posterior samples per (cell, chunk) window from the trained
posterior, concatenate across all windows of a condition, and histogram the pooled samples
in log10. This requires either persisting the per-window sample pool in the Experiment
stage or a dedicated resampling pass, and is left as a documented extension.

## Reference

Catapano et al., "Long-Term Single-Molecule Tracking in Living Cells using Weak-Affinity
Protein Labeling," *Angew. Chem. Int. Ed.* **2025**, 64, e202413117.
doi:10.1002/anie.202413117. Data: BioImage Archive accession S-BIAD1369 (CD86 and
CTLA-4, SiR-S5 / HaloTag single-color tracking; the experimental mobile-fraction
diffusion coefficients are D_mobile = 0.319 ± 0.010 µm²/s for CD86 and 0.279 ± 0.005
µm²/s for CTLA-4).

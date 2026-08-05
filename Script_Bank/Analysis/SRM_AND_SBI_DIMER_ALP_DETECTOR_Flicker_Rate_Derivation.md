# Flicker-rate derivation — method and usage

Companion to `SRM_AND_SBI_DIMER_ALP_DETECTOR_Flicker_Rate_Derivation.py`. The Detector's
emitter-brightness flicker is a continuous-time Markov chain whose base rate is `lambda_rate`
(`DETECTOR_WORKFLOW.md` §6.3). This utility fixes the prior on `lambda_rate` by measuring the
flicker timescale directly from the public single-molecule localization tables, so the value is
reproducible from data rather than assumed. This note explains what it computes, how to run it, and
how to read the result, without reading the code.

This is a special-situation utility, not one of the canonical pipeline stages. It lives in
`Script_Bank/Analysis`, is never wired into the stage dispatcher, and **reads only** — it writes
nothing to the data root and produces no artifact, only a printed report. It needs no trained
estimator and no GPU.

## What it computes

Organic-dye emission flickers: a single label's per-frame brightness fluctuates on a photophysical
timescale set by the dye and its environment (Dempsey et al. 2011; Ha & Tinnefeld 2012). The utility
measures that timescale from the real recordings and maps it to the model's rate parameter, in three
steps.

1. **Measure (data).** For each track, `log(intensity[photon])` is taken against `frame` and
   linearly detrended — this removes bleaching and the per-emitter mean, both multiplicative and
   hence additive in the log. A gap-aware temporal autocorrelation is formed and pooled over tracks.
   The lag-0 → lag-1 drop is per-localization fit noise (white, so confined to lag 0); the decay of
   the remaining autocorrelation is the physical flicker correlation time `tau_corr` (≈ 0.13 s for
   MET).

2. **Model.** The autocorrelation of the model's own brightness chain (`compute_matrices` →
   `generate_state_trajectories`, `simulation_dli_support.py`) decays on the timescale set by the
   generator's spectral gap (Norris 1997). The utility measures it the *same* way over a grid of
   `lambda_rate`, giving `tau_model(lambda_rate)`. To null the finite-track-length detrend bias, the
   model trajectories are cut to the empirical track-length distribution and detrended identically.

3. **Map.** The early autocorrelation *shape* of data and model is matched (a window-independent
   summary), inverting the monotone `tau_model` to convert `tau_corr → lambda_rate`; the 1/e
   crossing is reported as a cross-check. The mapping is evaluated across the `sigma_pc` prior (the
   flicker length scale is `sigma_bright`, which depends mildly on `sigma_pc`) and for each condition
   — the rate is photophysical and should be condition-independent.

## Requirements

- The `SRM_AND_SBI_ENVY_V0` environment. The utility imports the project package for the CTMC
  building blocks; it needs no GPU and no trained estimator.
- Read access to the MET single-molecule localization tables: per condition and cell, a ThunderSTORM
  `.tracked.csv` carrying the `frame`, `intensity [photon]`, and `track.id` columns (see **Data
  source**).

## How to run

```
MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Flicker_Rate_Derivation.py \
    --data-root /path/to/SPT_Movies_Complete_Data_Bank/MET
```

`--data-root` points at the localization root holding `<condition>/<cell>/tracks/*.tracked.csv`;
`--conditions` selects which condition subdirectories to derive (default `Fab InlB`). The utility
prints, per condition, the number of tracks, the measured `tau_corr`, and the derived `lambda_rate`
at each `sigma_pc` (shape-match, with the 1/e crossing as a cross-check).

## Result and interpretation

For MET (`S-BSST712`), `tau_corr ≈ 0.13–0.14 s → lambda_rate ≈ 5`, with Fab and InlB agreeing within
the cell-to-cell scatter — the expected signature of a photophysical rate that does not depend on the
biological condition. Read the value at the Fab reference `sigma_pc = 0.60` (`DETECTOR_WORKFLOW.md`
§6.5); the `sigma_pc` sweep quantifies the residual dependence (under ±0.1 in log10). The prior is
locked to log-uniform `(0.0, 1.0)` = `[1, 10]`, bracketing the measured value and excluding the slow
regime a fixed-imaging posterior-predictive render rules out.

## Essential notes

- **Not a calibration.** This places the *prior* on `lambda_rate` from data; the value the full
  calibration recovers is quantified separately on the held-out synthetic EVAL namespace, which has
  ground truth.
- **Reads only.** No file under the data root is modified; the utility emits a printed report, not an
  artifact.
- **`mu_pc` is immaterial** to the result — it cancels in the flicker dynamics (`|Δb| / sigma_bright`,
  `DETECTOR_WORKFLOW.md` §6.3) — so only `sigma_pc` (swept) enters the mapping.
- **Reproducible from the public accession alone**: no bundled data and no trained model are needed.

## Data source

MET single-molecule localization tables from the public accession **`S-BSST712`** (BioImage Archive /
EBI BioStudies, <https://www.ebi.ac.uk/biostudies/studies/S-BSST712>) — `Fab.zip` and `InlB.zip`, each
packaging per-cell ThunderSTORM localization tables (`.../tracks/<cell>.csv`, tracked to
`<cell>.tracked.csv`) and their processing protocols. The `intensity [photon]` column is the flicker
observable used here. The accession is the data of the MET single-molecule-tracking study of Harwardt
et al. 2017 (see References), and is the same source that supplies the imaging reference values in
`DETECTOR_WORKFLOW.md` §6.5.

## References

- Dempsey, G.T., Vaughan, J.C., Chen, K.H., Bates, M., Zhuang, X. (2011). Evaluation of fluorophores
  for optimal performance in localization-based super-resolution imaging. *Nature Methods*
  8(12):1027–1036.
- Ha, T., Tinnefeld, P. (2012). Photophysics of Fluorescent Probes for Single-Molecule Biophysics and
  Super-Resolution Imaging. *Annual Review of Physical Chemistry* 63:595–617.
- Norris, J.R. (1997). *Markov Chains.* Cambridge University Press.
- Ovesný, M., Křížek, P., Borkovec, J., Švindrych, Z., Hagen, G.M. (2014). ThunderSTORM: a
  comprehensive ImageJ plug-in for PALM and STORM data analysis and super-resolution imaging.
  *Bioinformatics* 30(16):2389–2390.
- Harwardt, M-L.I.E., Young, P., Bleymüller, W.M., Meyer, T., Karathanasis, C., Niemann, H-H.,
  Heilemann, M., Dietz, M.S. (2017). Membrane dynamics of resting and internalin B-bound MET receptor
  tyrosine kinase studied by single-molecule tracking. *FEBS Open Bio* 7(9):1422–1440 — the MET dataset
  source (public accession `S-BSST712`).
- `DETECTOR_WORKFLOW.md` §6.3 (the flicker model and this derivation) and §6.5 (the imaging reference
  values and their public provenance).

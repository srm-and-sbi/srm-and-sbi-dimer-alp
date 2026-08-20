# Experiment temporal dynamics (detector)

Companion to `SRM_AND_SBI_DIMER_ALP_DETECTOR_Experiment_Temporal_Dynamics.py`, the detector shim of
the temporal-dynamics analysis. It stacks the per-window **imaging** estimates along time to ask
whether the acquisition itself holds still across a recording — whether the emitter brightness, the
point-spread width, the photobleaching probability, and the flicker rate that a recording presents
are the same at its end as at its start.

The method, the four named central estimates (`mean-window`, `sgm-window`, `mean-trajectory`,
`sgm-trajectory`), the drift statistics and their definitions, the uncertainty figure and its
limits, the axis convention, and the reference scoping are **identical for both workflows** and are
documented once in the authoritative companion:

→ **`SRM_AND_SBI_DIMER_ALP_Experiment_Temporal_Dynamics.md`**

## Why this workflow's run is the confound test

The biology temporal analysis holds imaging **fixed**, so a trend it finds in a reaction rate cannot
be separated from a trend in the imaging that produced the videos. This analysis measures precisely
that imaging trend, on the **same recordings** — the experimental path pattern carries no workflow
qualifier, so a given recording index is the same acquisition in both workflows. A flat imaging
trajectory pushes a biology trend toward genuine dynamics; a drifting one identifies a live confound
and localizes it to a channel.

The asymmetry runs both ways, and that is the point: the detector marginalizes the reaction-diffusion
block, so it is blind to biological drift exactly as the biology analysis is blind to imaging drift.
Each workflow's run is the other's control, and **neither attributes a cause on its own**. Read them
together, and do not let either report's numbers be quoted as a causal claim.

## Reference scoping for the imaging parameters

Several imaging references are valid for the monomer control alone and are drawn only there: the
dimer condition's localization brightness is a two-label **per-detection sum** rather than a
per-emitter property, and its PSF width is **dimer-broadened**. The two population log-spreads
(`sigma_r`, `sigma_pc`) carry an errors-in-variables inflation, so their fitted values are drawn as
**upper bounds** rather than targets — a calibration is expected to land below them. `lambda_rate` is
photophysical and condition-independent, so it applies to both conditions. **`prob_photo_bleach` has
no external anchor** and is read on its internal evidence alone: its drift, its recovery, and its
posterior width. Provenance for every value: `DETECTOR_WORKFLOW.md` §6.2/§6.3/§6.5.

## Outputs and how to run

Outputs land under `<data_bank>/<posit>/<alias>_<timing_label>_MAP_Experiment/temporal_dynamics/`,
where `<alias>` is `SRM_AND_SBI_DIMER_ALP_DETECTOR`, so the two workflows' results never collide.

```bash
MACHINE_PROFILE=<profile> python \
    Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Experiment_Temporal_Dynamics.py \
    --total-time-seconds 2.0 [--central sgm|mean] [--params prob_photo_bleach,mu_pc] [--dry-run]
```

CPU only. The recovery annotation appears when a detector MAP-recovery artifact is present for the
same timing label; without one the figures omit it and the report says so.

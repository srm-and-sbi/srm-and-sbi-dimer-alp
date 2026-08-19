# Posterior-predictive video (detector)

Companion to `SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.py`, the detector shim of the
posterior-predictive video comparison. It renders a synthetic video from the **imaging** parameters
inferred for one real MET recording and places it beside that recording.

The mechanics, the options, the outputs, and the interpretation are documented once in the
authoritative companion:

→ **`SRM_AND_SBI_DIMER_ALP_Posterior_Predictive_Video.md`**

Read that note's table *"One engine, two workflows"* first: for this workflow the MAP supplies the
six imaging parameters, the reaction-diffusion block is a marginalized nuisance (drawn per render, or
pinned with `--fixed-nuisance-RDS`), and the system is built **diffusion-only** because the detector's
physics model has no reactions. Biology inverts all three. MET camera provenance:
`REFERENCE_EMCCD_NOISE_MODEL.md` Sec. 6 and `DETECTOR_WORKFLOW.md` Sec. 6.5.

Outputs land under `<data_bank>/<posit>/<alias>_<timing_label>_Posterior_Predictive_Video/`, where
`<alias>` is `SRM_AND_SBI_DIMER_ALP_DETECTOR`.

```bash
MACHINE_PROFILE=<profile> python \
    Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.py \
    --total-time-seconds 2.0 --kind MET-FAB --cell 0 [--fixed-imaging-parameters] [--dry-run]
```

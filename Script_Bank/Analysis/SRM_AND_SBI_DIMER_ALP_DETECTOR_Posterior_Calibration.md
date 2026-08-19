# Posterior Calibration (detector)

Companion to `SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Calibration.py`, the detector shim
of the posterior-calibration diagnostic. It scores how well-calibrated the trained
**imaging** posterior is on the held-out `_DETECTOR` EVAL namespace, over the 6 target
imaging parameters (`mu_r, sigma_r, mu_pc, sigma_pc, prob_photo_bleach, lambda_rate`).

The diagnostic, its four measures (SBC, expected coverage, TARP, L-C2ST), the design, the
stratification, the options, and the outputs are **identical for both workflows** and are
documented once in the authoritative companion:

→ **`SRM_AND_SBI_DIMER_ALP_Posterior_Calibration.md`**

This shim differs only in the config it builds (`detector_workflow()`): the detector
prior, parameter table, keys, and `_DETECTOR`-aliased paths. Outputs land under
`<data_bank>/Posit/<project_alias>_<timing_label>_Posterior_Calibration/`, where
`<project_alias>` is `SRM_AND_SBI_DIMER_ALP_DETECTOR` for this workflow.

```bash
MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Calibration.py \
    --total-time-seconds 2.0 --eval-tasks 10 --posterior-samples 1000
```

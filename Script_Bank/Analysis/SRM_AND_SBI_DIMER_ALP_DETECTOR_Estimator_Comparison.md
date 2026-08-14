# Estimator Comparison (detector)

Companion to `SRM_AND_SBI_DIMER_ALP_DETECTOR_Estimator_Comparison.py`, the detector shim of
the estimator-comparison diagnostic. It decides whether one trained **imaging** estimator
generalizes better than another, comparing two `_DETECTOR`-namespaced Test-Loss-Distribution
artifacts by the paired log-score on the shared `(task, sim)` TEST subset.

The method (paired log-score, why pairing cancels the entropy floor), the three tests
(Diebold-Mariano / Wilcoxon / paired bootstrap), the `--a`/`--b` input forms, the options,
the report metrics, and the outputs are **identical for both workflows** and are documented
once in the authoritative companion:

→ **`SRM_AND_SBI_DIMER_ALP_Estimator_Comparison.md`**

This shim differs only in the config it builds (`detector_workflow()`): the
`_DETECTOR`-aliased `Posit/` namespace its artifacts live under. Outputs land under
`<data_bank>/Posit/<project_alias>_DETECTOR_<timing_label>_Estimator_Comparison/`.

```bash
MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Estimator_Comparison.py \
    --total-time-seconds 2.0 --a canonical --b -12.16
```

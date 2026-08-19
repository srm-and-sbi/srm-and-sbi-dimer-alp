# Experimental-versus-synthetic embedding distance (detector)

Companion to `SRM_AND_SBI_DIMER_ALP_DETECTOR_Embedding_Space_Distance.py`, the detector shim of the
embedding-distance analysis. It scores how far the real MET recordings sit from the synthetic videos
in the trained **imaging** posterior's embedding space.

The measurement, its two tests (MMD with a permutation null and C2ST, both blocked by recording), the
design, the options, and the outputs are **identical for both workflows** and are documented once in
the authoritative companion:

→ **`SRM_AND_SBI_DIMER_ALP_Embedding_Space_Distance.md`**

Read that note's section *"The measurement is shared; the reading inverts"* before interpreting a
result. For this workflow the imaging parameters **are** the inference target, so a measured gap is a
realism failure of the imaging forward model and closing it is the goal of the detector calibration —
the opposite of what the same number means for biology. Further design justification:
`DETECTOR_WORKFLOW.md`, section "Quantitative experimental-versus-synthetic distance".

This shim differs only in the config it builds (`detector_workflow()`): the detector
parameterization, its report-facing parameter descriptions, and the `_DETECTOR`-aliased paths.
Outputs land under `<data_bank>/<posit>/<alias>_<timing_label>_Embedding_Space_Distance/`, where
`<alias>` is `SRM_AND_SBI_DIMER_ALP_DETECTOR`.

```bash
MACHINE_PROFILE=<profile> python \
    Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Embedding_Space_Distance.py \
    --total-time-seconds 2.0 [--kinds MET-FAB,MET-INLB] [--eval-tasks N] [--dry-run]
```

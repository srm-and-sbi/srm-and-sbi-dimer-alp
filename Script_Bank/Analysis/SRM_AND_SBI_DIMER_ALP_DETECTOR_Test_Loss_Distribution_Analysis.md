# Test-loss distribution analysis (detector)

Companion to `SRM_AND_SBI_DIMER_ALP_DETECTOR_Test_Loss_Distribution_Analysis.py`, the detector
shim of the test-loss-distribution analysis. It reads the `_DETECTOR`-namespaced
Test-Loss-Distribution artifact and produces the distribution shape, the uniform-prior NLL
reference, and the tail-vs-parameter identifiability read over the 6 imaging parameters — which
of them, and which end of their range, mark the hardest examples.

The method (distribution shape, the uniform-prior reference, the tail-vs-θ analysis, the
Benjamini-Hochberg adjustment), the two input modes, the options, the report tables, and how to
read them are **identical for both workflows** and are documented once in the authoritative
companion:

→ **`SRM_AND_SBI_DIMER_ALP_Test_Loss_Distribution_Analysis.md`**

This shim differs only in the config it builds (`detector_workflow()`): the `_DETECTOR`-aliased
`Posit/` namespace its canonical artifact lives under. As for the biology shim, `--tld-path`
analyzes any artifact verbatim, workflow-independently.

```bash
MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Test_Loss_Distribution_Analysis.py \
    --total-time-seconds 2.0
```

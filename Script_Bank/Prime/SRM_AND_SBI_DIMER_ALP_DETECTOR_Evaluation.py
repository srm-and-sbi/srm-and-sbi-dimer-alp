"""Entry-point script (detector workflow): MAP-recovery of imaging parameters on EVAL.

Part of the detector calibration workflow (see DETECTOR_WORKFLOW.md) -- the workflow that
infers the imaging parameters. It loads the trained imaging estimator and the held-out
``_DETECTOR`` EVAL namespace (videos with known ground-truth imaging theta), estimates the MAP
parameter vector for each EVAL video via the seed-then-optimize procedure in
``evaluation.map_estimate``, and reports how well the 6 inferred imaging parameters recover the
truth. The prior, parameter table, keys, and data paths are the detector's
(``det.build_prior``, ``det.DETECTOR_PARAMETERIZATION``, ``det.detector_paths(...)``).

The Evaluation stage runs on ONE shared engine used by both workflows:
``srm_and_sbi_dimer_alp.evaluation_runner.run_evaluation`` (built on the shared ``evaluation``
numeric layer). This entry point is the detector shim; the biology twin
(``SRM_AND_SBI_DIMER_ALP_Evaluation.py``) is identical except it builds the biology config (10
reaction-diffusion parameters). The engine -- including the multi-GPU sharding and the
``--merge`` combine step -- is shared.

Outputs (under ``<data_bank>/<posit_subdir>/<project_alias>_{timing_label}_MAP_Recovery/``,
``<project_alias>`` carrying the ``_DETECTOR`` qualifier): report.md + figures/recovery_<KEY>.png
+ the combined ``_MAP_Recovery.npz``.

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Evaluation.py \\
        --total-time-seconds 2.0 --eval-tasks 2 --pool-mode unrestricted
"""

import sys

from srm_and_sbi_dimer_alp.evaluation_runner import build_evaluation_parser, run_evaluation
from srm_and_sbi_dimer_alp.utils import console_log_context
from srm_and_sbi_dimer_alp.workflow import detector_workflow


if __name__ == "__main__":
    cli_args = build_evaluation_parser().parse_args(sys.argv[1:])
    cfg = detector_workflow()
    with console_log_context(cli_args, "Evaluation", paths=cfg.console_log_paths):
        run_evaluation(cfg, cli_args)

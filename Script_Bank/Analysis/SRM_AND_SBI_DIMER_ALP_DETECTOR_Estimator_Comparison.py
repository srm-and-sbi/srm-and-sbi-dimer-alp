"""Analysis script (detector workflow): compare two imaging estimators by paired log-score.

Part of the detector calibration workflow (see DETECTOR_WORKFLOW.md). Decides whether one
imaging estimator generalizes better than another by the paired per-video log-score on the
shared ``(task, sim)`` TEST subset -- pairing cancels each video's intrinsic entropy floor,
so the difference isolates the two estimators' KL gap (Amisano & Giacomini 2007) -- tested
with the Diebold-Mariano statistic and, as heavy-tail-robust companions, the Wilcoxon
signed-rank test and a paired bootstrap. The inputs are two ``_DETECTOR``-namespaced
Test-Loss-Distribution artifacts (per-video loss keyed by ``(task, sim)``).

This diagnostic runs on ONE shared engine used by both workflows:
``srm_and_sbi_dimer_alp.estimator_comparison_runner.run_estimator_comparison`` (built on the
workflow-agnostic ``estimator_comparison`` kernel). This entry point is the detector shim; the
biology twin (``SRM_AND_SBI_DIMER_ALP_Estimator_Comparison.py``) is identical except it builds
the biology config. An Analysis diagnostic (read-only, never dispatched); no GPU.

Outputs (under ``<data_bank>/<posit_subdir>/<project_alias>_{timing_label}_Estimator_Comparison/``,
``<project_alias>`` carrying the ``_DETECTOR`` qualifier): report.md + the improvement figure
+ the combined ``_Estimator_Comparison.npz``.

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Estimator_Comparison.py \\
        --total-time-seconds 2.0 --a canonical --b -12.16
"""

import sys

from srm_and_sbi_dimer_alp.estimator_comparison_runner import (
    build_estimator_comparison_parser, run_estimator_comparison)
from srm_and_sbi_dimer_alp.utils import console_log_context
from srm_and_sbi_dimer_alp.workflow import detector_workflow


if __name__ == "__main__":
    cli_args = build_estimator_comparison_parser().parse_args(sys.argv[1:])
    cfg = detector_workflow()
    with console_log_context(cli_args, "Estimator_Comparison", paths=cfg.console_log_paths):
        run_estimator_comparison(cfg, cli_args)

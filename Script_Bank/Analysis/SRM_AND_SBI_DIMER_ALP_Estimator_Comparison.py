"""Analysis script (biology workflow): compare two trained estimators by paired log-score.

Decides whether one biology (reaction-diffusion) estimator generalizes better than another
by the paired per-video log-score on the shared ``(task, sim)`` TEST subset -- pairing
cancels each video's intrinsic entropy floor, so the difference isolates the two estimators'
KL gap (Amisano & Giacomini 2007) -- tested with the Diebold-Mariano statistic and, as
heavy-tail-robust companions, the Wilcoxon signed-rank test and a paired bootstrap. The
inputs are two Test-Loss-Distribution artifacts (per-video loss keyed by ``(task, sim)``).

This diagnostic runs on ONE shared engine used by both the biology and detector workflows:
``srm_and_sbi_dimer_alp.estimator_comparison_runner.run_estimator_comparison`` (built on the
workflow-agnostic ``estimator_comparison`` kernel). This entry point is the biology shim; the
detector twin (``SRM_AND_SBI_DIMER_ALP_DETECTOR_Estimator_Comparison.py``) is identical except
it builds the detector config. An Analysis diagnostic (read-only, never dispatched); no GPU.

Outputs (under ``<data_bank>/<posit_subdir>/<project_alias>_{timing_label}_Estimator_Comparison/``):
    report.md                          -- paired-comparison table + verdict.
    figures/improvement_distribution.png -- per-video paired-improvement histogram.
    <...>_Estimator_Comparison.npz     -- the saved per-video improvement array.

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_Estimator_Comparison.py \\
        --total-time-seconds 2.0 --a canonical --b -6.91
"""

import sys

from srm_and_sbi_dimer_alp.estimator_comparison_runner import (
    build_estimator_comparison_parser, run_estimator_comparison)
from srm_and_sbi_dimer_alp.utils import console_log_context
from srm_and_sbi_dimer_alp.workflow import biology_workflow


if __name__ == "__main__":
    cli_args = build_estimator_comparison_parser().parse_args(sys.argv[1:])
    cfg = biology_workflow()
    with console_log_context(cli_args, "Estimator_Comparison", paths=cfg.console_log_paths):
        run_estimator_comparison(cfg, cli_args)

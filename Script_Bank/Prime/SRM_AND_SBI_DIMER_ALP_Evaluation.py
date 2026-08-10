"""Entry-point script (biology workflow): evaluate a trained posterior by MAP recovery.

Loads the trained posterior and the held-out EVAL namespace (videos with known
ground-truth theta), estimates the MAP parameter vector for each EVAL video via the
seed-then-optimize procedure in ``evaluation.map_estimate``, and reports how well the
inferred parameters recover the truth. The biology workflow recovers the 10
reaction-diffusion parameters. The EVAL namespace is physically separate from
TRAIN/TEST (distinct ``_EVAL`` suffix + independent seed), so the recovery number is
leak-free by construction.

The Evaluation stage runs on ONE shared engine used by both the biology and the detector
workflows: ``srm_and_sbi_dimer_alp.evaluation_runner.run_evaluation`` (built on the shared
``evaluation`` numeric layer). This entry point is the biology shim; the detector twin
(``SRM_AND_SBI_DIMER_ALP_DETECTOR_Evaluation.py``) is identical except it builds the detector
config (6 imaging parameters, detector alias). The engine -- including the multi-GPU sharding
and the ``--merge`` combine step -- is shared.

Outputs (under ``<data_bank>/<posit_subdir>/<project_alias>_{timing_label}_MAP_Recovery/``):
    report.md                    -- recovery report (per-parameter error table + figures).
    figures/recovery_<KEY>.png   -- inferred-vs-true + residual-error views (log10), per parameter.
    <...>_MAP_Recovery.npz       -- saved arrays: true_log10, inferred_log10, scores.

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_Evaluation.py \\
        --total-time-seconds 2.0 --eval-tasks 1
"""

import sys

from srm_and_sbi_dimer_alp.evaluation_runner import build_evaluation_parser, run_evaluation
from srm_and_sbi_dimer_alp.utils import console_log_context
from srm_and_sbi_dimer_alp.workflow import biology_workflow


if __name__ == "__main__":
    cli_args = build_evaluation_parser().parse_args(sys.argv[1:])
    cfg = biology_workflow()
    with console_log_context(cli_args, "Evaluation", paths=cfg.console_log_paths):
        run_evaluation(cfg, cli_args)

"""Analysis script (detector workflow): score the imaging posterior's calibration.

Part of the detector calibration workflow (see DETECTOR_WORKFLOW.md) -- the workflow
that infers the imaging parameters. It loads the trained imaging estimator and the
held-out ``_DETECTOR`` EVAL namespace (videos with known ground-truth imaging theta),
draws the posterior for each EVAL video, and scores how well-calibrated the posterior is
with four simulation-based-calibration diagnostics -- SBC (Talts et al. 2018), expected
coverage (Deistler/Hermans 2022), TARP (Lemos et al. 2023), and L-C2ST (Linhart et al.
2023) -- overall and stratified by each target imaging parameter. The prior, parameter
table, keys, and data paths are the detector's (``det.build_prior``,
``det.DETECTOR_PARAMETERIZATION``, ``det.detector_paths(...)``).

This diagnostic runs on ONE shared engine used by both workflows:
``srm_and_sbi_dimer_alp.posterior_calibration_runner.run_posterior_calibration`` (built on
the workflow-agnostic ``posterior_calibration`` kernel). This entry point is the detector
shim; the biology twin (``SRM_AND_SBI_DIMER_ALP_Posterior_Calibration.py``) is identical
except it builds the biology config (10 reaction-diffusion parameters). The engine --
including the multi-GPU sharding and the ``--merge`` combine step -- is shared. This is an
Analysis diagnostic (never wired into the ``Submit.sh`` dispatcher).

Outputs (under ``<data_bank>/<posit_subdir>/<project_alias>_{timing_label}_Posterior_Calibration/``,
``<project_alias>`` carrying the ``_DETECTOR`` qualifier): report.md + the SBC / coverage /
TARP / stratified figures + the combined ``_Posterior_Calibration.npz``.

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Calibration.py \\
        --total-time-seconds 2.0 --eval-tasks 10 --posterior-samples 1000
"""

import sys

from srm_and_sbi_dimer_alp.posterior_calibration_runner import (
    build_posterior_calibration_parser, run_posterior_calibration)
from srm_and_sbi_dimer_alp.utils import console_log_context
from srm_and_sbi_dimer_alp.workflow import detector_workflow


if __name__ == "__main__":
    cli_args = build_posterior_calibration_parser().parse_args(sys.argv[1:])
    cfg = detector_workflow()
    with console_log_context(cli_args, "Posterior_Calibration", paths=cfg.console_log_paths):
        run_posterior_calibration(cfg, cli_args)

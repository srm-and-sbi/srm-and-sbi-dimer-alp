"""Analysis script (biology workflow): score a trained posterior's calibration.

Loads the trained posterior and the held-out EVAL namespace (videos with known
ground-truth theta), draws the posterior for each EVAL video, and scores how
well-calibrated the posterior is with four simulation-based-calibration diagnostics --
SBC (Talts et al. 2018), expected coverage (Deistler/Hermans 2022), TARP (Lemos et al.
2023), and L-C2ST (Linhart et al. 2023) -- overall and stratified by each target
parameter. The biology workflow scores calibration of the 10 reaction-diffusion
parameters. The EVAL namespace is physically separate from TRAIN/TEST (distinct
``_EVAL`` suffix + independent seed), and its theta are prior draws, so the EVAL set is
itself a proper SBC sample and the calibration number is leak-free by construction.

This diagnostic runs on ONE shared engine used by both the biology and detector
workflows: ``srm_and_sbi_dimer_alp.posterior_calibration_runner.run_posterior_calibration``
(built on the workflow-agnostic ``posterior_calibration`` kernel). This entry point is
the biology shim; the detector twin
(``SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Calibration.py``) is identical except it
builds the detector config (6 imaging parameters, detector alias). The engine --
including the multi-GPU sharding and the ``--merge`` combine step -- is shared. This is
an Analysis diagnostic (never wired into the ``Submit.sh`` dispatcher).

Outputs (under ``<data_bank>/<posit_subdir>/<project_alias>_{timing_label}_Posterior_Calibration/``):
    report.md                         -- calibration report (SBC / coverage / TARP / L-C2ST tables).
    figures/sbc_rank_histograms.png   -- per-marginal SBC rank histograms (flat = calibrated).
    figures/expected_coverage.png     -- empirical vs nominal coverage curve.
    figures/tarp_ecp.png              -- TARP ECP curve.
    figures/stratified_<test>.png     -- each diagnostic across target-theta bins.
    <...>_Posterior_Calibration.npz   -- saved arrays: truths, samples, log-densities, embeddings.

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_Posterior_Calibration.py \\
        --total-time-seconds 2.0 --eval-tasks 10 --posterior-samples 1000
"""

import sys

from srm_and_sbi_dimer_alp.posterior_calibration_runner import (
    build_posterior_calibration_parser, run_posterior_calibration)
from srm_and_sbi_dimer_alp.utils import console_log_context
from srm_and_sbi_dimer_alp.workflow import biology_workflow


if __name__ == "__main__":
    cli_args = build_posterior_calibration_parser().parse_args(sys.argv[1:])
    cfg = biology_workflow()
    with console_log_context(cli_args, "Posterior_Calibration", paths=cfg.console_log_paths):
        run_posterior_calibration(cfg, cli_args)

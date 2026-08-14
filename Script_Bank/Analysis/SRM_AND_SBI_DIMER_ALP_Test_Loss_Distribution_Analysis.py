"""Analysis script (biology workflow): per-example test-loss distribution analysis.

Reads the biology Inference stage's best-epoch per-example test-loss artifact (the
``TestLossDistribution`` ``.npz``) and produces the picture the reported mean cannot give:
the distribution shape, the uniform-prior NLL reference (the no-information baseline), and
the tail-vs-parameter identifiability read -- which of the 10 reaction-diffusion parameters,
and which end of their range, mark the hardest examples (for biology the species counts,
whose low end yields uninformative videos). SBC / TARP calibration is out of scope here
(they need posterior samples); the hard tail's *honesty* is measured by the
Posterior_Calibration diagnostic.

This diagnostic runs on ONE shared engine used by both the biology and detector workflows:
``srm_and_sbi_dimer_alp.test_loss_analysis_runner.run_test_loss_analysis`` (built on the
workflow-agnostic ``test_loss_analysis`` kernel). This entry point is the biology shim; the
detector twin (``SRM_AND_SBI_DIMER_ALP_DETECTOR_Test_Loss_Distribution_Analysis.py``) is
identical except it builds the detector config. An Analysis diagnostic (read-only, never
dispatched); no GPU. ``--tld-path`` analyzes any artifact verbatim, workflow-independently.

Outputs (a ``*_Analysis`` directory beside the artifact, under the ``Posit/`` tier):
    report.md + figures/{nll_histogram,nll_ecdf,loss_vs_theta,tail_drivers}.png

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_Test_Loss_Distribution_Analysis.py \\
        --total-time-seconds 2.0
"""

import sys

from srm_and_sbi_dimer_alp.test_loss_analysis_runner import (
    build_test_loss_analysis_parser, run_test_loss_analysis)
from srm_and_sbi_dimer_alp.utils import console_log_context
from srm_and_sbi_dimer_alp.workflow import biology_workflow


if __name__ == "__main__":
    cli_args = build_test_loss_analysis_parser().parse_args(sys.argv[1:])
    cfg = biology_workflow()
    with console_log_context(cli_args, "Test_Loss_Distribution_Analysis", paths=cfg.console_log_paths):
        rc = run_test_loss_analysis(cfg, cli_args)
    sys.exit(rc)

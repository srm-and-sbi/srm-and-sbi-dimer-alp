"""Analysis entry point (detector workflow): posterior-predictive video against a real recording.

Takes the six imaging parameters INFERRED from one real MET recording, simulates a video with them at
that recording's own length, and puts the two side by side -- the direct visual test of whether the
calibrated imaging model reproduces the appearance of real data. The reaction-diffusion block is a
marginalized nuisance here (drawn per render, or pinned with --fixed-nuisance-RDS), and the system is
built diffusion-only because the detector's physics model has no reactions. The five SCOPE camera
parameters are pinned to their MET values rather than drawn, since the comparison is against one
specific acquisition.

The mechanics, the options, and the outputs are shared with the biology analysis and documented once
in the authoritative companion:

    SRM_AND_SBI_DIMER_ALP_Posterior_Predictive_Video.md

This shim differs only in the config it builds (detector_workflow()): which block the MAP supplies
(imaging, not reaction-diffusion), how the other block is fixed, and the _DETECTOR-aliased paths.
Provenance for the MET camera values: REFERENCE_EMCCD_NOISE_MODEL.md Sec. 6 and DETECTOR_WORKFLOW.md
Sec. 6.5.

Usage:
    MACHINE_PROFILE=<p> python \
        Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.py \
        --total-time-seconds 2.0 --kind MET-FAB --cell 0 [--fixed-imaging-parameters] [--dry-run]
"""
import sys

from srm_and_sbi_dimer_alp.posterior_predictive_video_runner import (
    build_parser, run_posterior_predictive_video)
from srm_and_sbi_dimer_alp.workflow import detector_workflow


def main(argv=None):
    parser = build_parser("Posterior-predictive video comparison (detector workflow).")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return run_posterior_predictive_video(detector_workflow(), args)


if __name__ == "__main__":
    raise SystemExit(main())

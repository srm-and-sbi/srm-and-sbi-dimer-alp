"""Entry-point script (detector workflow): train the imaging-parameter posterior estimator.

Part of the detector calibration workflow (see DETECTOR_WORKFLOW.md) -- the workflow that
infers the imaging parameters and marginalizes the reaction-diffusion domain + camera. It reads
the ``_DETECTOR``-namespaced (video, imaging-theta) pairs written by the detector DLI stage, so
the estimator's parameter dimension is the 6 learnable imaging parameters (``theta_dim = 6``),
sourced from ``detector_parameterization``; all outputs carry the ``_DETECTOR`` alias so they
never collide with biology inference artifacts.

The Inference stage runs on ONE shared engine used by both workflows:
``srm_and_sbi_dimer_alp.inference_runner.run_inference`` (built on the shared ``inference_support``
training machinery). This entry point is the detector shim; the biology twin
(``SRM_AND_SBI_DIMER_ALP_Inference.py``) is identical except it builds the biology config (10
reaction-diffusion parameters). The engine, and therefore the training/selection/save behavior,
is shared.

Outputs (the ``{timing_label}`` token, e.g. ``2S_50FPS``; ``<project_alias>`` carries the
``_DETECTOR`` qualifier):

    <data_bank>/<labor_subdir>/<project_alias>_{timing_label}_Optimum_ANN.pth
    <data_bank>/<labor_subdir>/<project_alias>_{timing_label}_Resurrect_State_ANN.pth
    <data_bank>/<posit_subdir>/<project_alias>_{timing_label}_Estimator.npz
        -- the version-portable estimator artifact (the detector's posterior artifact).

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Inference.py \\
        --total-time-seconds 2.0 --epochs 5 --tasks 25 --test-tasks 5 --batch-size 8 --seed None
    For the full detector smoke test see section 2.5 in VALIDATION.md.
"""

import sys

from srm_and_sbi_dimer_alp.inference_runner import build_inference_parser, run_inference
from srm_and_sbi_dimer_alp.utils import console_log_context
from srm_and_sbi_dimer_alp.workflow import detector_workflow


if __name__ == "__main__":
    cli_args = build_inference_parser().parse_args(sys.argv[1:])
    cfg = detector_workflow()
    with console_log_context(cli_args, "Inference", paths=cfg.console_log_paths):
        run_inference(cfg, cli_args)

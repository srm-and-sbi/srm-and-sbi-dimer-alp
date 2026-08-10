"""Entry-point script (biology workflow): train the neural posterior estimator.

Loads (video, theta) pairs from the previous RDS + DLI stages, trains a masked
autoregressive flow (MAF) density estimator conditioned on a Complex3DCNN +
TemporalTransformer embedding of the videos, and saves the trained estimator as a
self-describing, version-portable artifact for downstream sampling. The biology
workflow infers the 10 reaction-diffusion parameters.

The Inference stage runs on ONE shared engine used by both the biology and the detector
workflows: ``srm_and_sbi_dimer_alp.inference_runner.run_inference`` (which builds on the
shared ``inference_support`` training machinery). This entry point is the biology shim; the
detector twin (``SRM_AND_SBI_DIMER_ALP_DETECTOR_Inference.py``) is identical except it builds
the detector config (6 imaging parameters, detector alias). The engine, and therefore the
training/selection/save behavior, is shared.

Outputs (the ``{timing_label}`` token, e.g. ``2S_50FPS``, is rendered from
``PARAMETERS.simulation.timing.label`` to namespace files by duration + fps):

    <data_bank>/<labor_subdir>/<project_alias>_{timing_label}_Optimum_ANN.pth
        -- the best-so-far estimator checkpoint (overwritten on each new optimum).
    <data_bank>/<labor_subdir>/<project_alias>_{timing_label}_Resurrect_State_ANN.pth
        -- the full training state, written atomically every epoch (a --resurrect hot restart).
    <data_bank>/<posit_subdir>/<project_alias>_{timing_label}_Estimator.npz
        -- the trained estimator as a self-describing, version-portable artifact.

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_Inference.py \\
        --total-time-seconds 2.0 --epochs 5 --tasks 2

    # Resume training from the previously saved checkpoint:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_Inference.py --resurrect
"""

import sys

from srm_and_sbi_dimer_alp.inference_runner import build_inference_parser, run_inference
from srm_and_sbi_dimer_alp.utils import console_log_context
from srm_and_sbi_dimer_alp.workflow import biology_workflow


if __name__ == "__main__":
    cli_args = build_inference_parser().parse_args(sys.argv[1:])
    cfg = biology_workflow()
    with console_log_context(cli_args, "Inference", paths=cfg.console_log_paths):
        run_inference(cfg, cli_args)

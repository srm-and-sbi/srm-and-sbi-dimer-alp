"""Entry-point script (detector workflow): render videos with imaging drawn from theta.

Part of the detector calibration workflow (see DETECTOR_WORKFLOW.md) -- the workflow that
infers the imaging parameters and marginalizes the reaction-diffusion domain + camera. Its one
governing difference from the biology DLI stage: the imaging parameters are the detector's
inference target, so this stage draws the six imaging parameters per simulation from the
imaging prior box (the training label) -- whereas the biology stage marginalizes them from the
``Nuisance_DLI`` artifact -- and persists the drawn imaging theta as the primary ``Theta_Set``.
The five SCOPE camera parameters are still marginalized (recorded as ``Nuisance_SCOPE``). The
trajectories consumed here are the diffusion-only trajectories produced by the detector RDS
stage. Detector data namespaces separately under the ``_DETECTOR`` runtime-prefix qualifier, so
nothing collides with biology.

The DLI stage runs on ONE shared engine used by both workflows:
``srm_and_sbi_dimer_alp.simulation_dli_runner.run_dli`` -- which renders every video through the
shared, source-agnostic ``render_dli_video``. This entry point is the detector shim; the biology
twin (``SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py``) is identical except it builds the biology
config. The render engine, and therefore the video output, is shared.

Outputs (the ``{timing_label}`` token, e.g. ``5S_50FPS``, is rendered from
``PARAMETERS.simulation.timing.label``; ``<project_alias>`` carries the ``_DETECTOR`` qualifier):

    <data_bank>/<video_subdir>/<project_alias>_{timing_label}_Video_Set_TASK_{n}_{split}.zarr
    <data_bank>/<theta_subdir>/<project_alias>_{timing_label}_Theta_Set_TASK_{n}_{split}.zarr
        -- imaging-theta labels (the inference target / training label)
    <data_bank>/<theta_subdir>/<project_alias>_{timing_label}_Nuisance_SCOPE_Theta_Set_TASK_{n}_{split}.zarr
        -- the five SCOPE camera parameters drawn from their a-priori box (physical)

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Simulation_DLI.py \\
        --total-time-seconds 2.0 --split train --tasks 25 --task-simulations 10 \\
        --video-dtype-bits 8 --seed None
    (repeat with --split test --tasks 5, and --split eval --tasks 2)
    For the full detector smoke test see section 2.5 in VALIDATION.md.
"""

import sys

from srm_and_sbi_dimer_alp.simulation_dli_runner import build_dli_parser, run_dli
from srm_and_sbi_dimer_alp.utils import console_log_context
from srm_and_sbi_dimer_alp.workflow import detector_workflow


if __name__ == "__main__":
    cli_args = build_dli_parser().parse_args(sys.argv[1:])
    cfg = detector_workflow()
    with console_log_context(cli_args, "DLI", paths=cfg.console_log_paths):
        run_dli(cfg, cli_args)

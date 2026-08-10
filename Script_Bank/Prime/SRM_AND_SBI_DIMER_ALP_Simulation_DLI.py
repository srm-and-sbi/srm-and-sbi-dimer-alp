"""Entry-point script (biology workflow): render diffraction-limited videos from RDS trajectories.

Reads each .h5 trajectory produced by the biology RDS stage, extracts particle
poses + dimer state mask, renders the synthetic fluorescence video via the shared
DLI renderer (PSF + Poisson + EMCCD noise), and saves the videos as a .zarr
(compressed) or .npy (uncompressed) video set per task.

The whole imaging block is marginalized as a nuisance in production (DETECTOR_WORKFLOW.md
sec. 9.3, Phase D): the six photophysics (``mu_r``, ``sigma_r``, ``mu_pc``, ``sigma_pc``,
``prob_photo_bleach``, ``lambda_rate``) are drawn per simulation from the persisted
``Nuisance_DLI`` artifact (the calibrated-imaging pool minted by the detector workflow), and
the five SCOPE camera parameters (``gamma``, ``kappa_o``, ``kappa_b``, ``kappa_s``,
``kappa_q``) are drawn per simulation from their a-priori box. Both are recorded, per task,
as self-labeling ``Theta_Set`` variants beside the learnable 10-RDS ``Theta_Set``, then
concatenated into the eleven-key imaging vector the renderer consumes. The learnable
``Theta_Set`` (the 10 reaction-diffusion labels the estimator inverts) is READ here only for
the sim-0 diagnostics table and is never re-written by this stage. The ``Nuisance_DLI``
artifact is a REQUIRED input built by a user-driven analysis step; if absent, the stage fails
loud naming the analysis to run.

The DLI stage runs on ONE shared engine used by both the biology and the detector workflows:
``srm_and_sbi_dimer_alp.simulation_dli_runner.run_dli``. This entry point is the biology shim.
The detector twin (``SRM_AND_SBI_DIMER_ALP_DETECTOR_Simulation_DLI.py``) is identical except it
builds the detector config (which draws the imaging block from the prior box as the inference
target instead of from the artifact); the render engine, and therefore the video output, is
shared.

Outputs (the ``{timing_label}`` token, e.g. ``2S_50FPS``, is rendered from
``PARAMETERS.simulation.timing.label`` to namespace files by duration + fps):

    <data_bank>/<video_subdir>/<project_alias>_{timing_label}_Video_Set_TASK_{n}.zarr
    <data_bank>/<theta_subdir>/<project_alias>_{timing_label}_Nuisance_DLI_Theta_Set_TASK_{n}_{split}.zarr
        -- the six photophysics drawn from the Nuisance_DLI artifact (physical)
    <data_bank>/<theta_subdir>/<project_alias>_{timing_label}_Nuisance_SCOPE_Theta_Set_TASK_{n}_{split}.zarr
        -- the five SCOPE camera parameters drawn from their a-priori box (physical)

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py \\
        --total-time-seconds 2.0 --tasks 2 --task-simulations 5 --video-dtype-bits 8 --seed None
    (add --dry-run to resolve config + inputs and print planned I/O without rendering)
"""

import sys

from srm_and_sbi_dimer_alp.simulation_dli_runner import build_dli_parser, run_dli
from srm_and_sbi_dimer_alp.utils import console_log_context
from srm_and_sbi_dimer_alp.workflow import biology_workflow


if __name__ == "__main__":
    cli_args = build_dli_parser().parse_args(sys.argv[1:])
    cfg = biology_workflow()
    with console_log_context(cli_args, "DLI", paths=cfg.console_log_paths):
        run_dli(cfg, cli_args)

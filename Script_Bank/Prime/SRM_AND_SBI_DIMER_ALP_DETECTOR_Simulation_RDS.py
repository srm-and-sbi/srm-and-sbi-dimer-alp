"""Entry-point script (detector workflow): generate diffusion-only ReaDDy
reaction-diffusion trajectories.

Part of the detector calibration workflow (see DETECTOR_WORKFLOW.md) -- the
workflow that infers the imaging parameters and marginalizes the reaction-diffusion
domain. This stage draws the DIMER model's reaction-diffusion parameters from the
detector RDS *nuisance* (three species counts + three diffusion coefficients)
rather than from the learnable theta prior, and builds the ReaDDy system in
diffusion-only mode (the four reaction channels are not registered). The imaging
parameters -- the detector's actual inference target -- are drawn and rendered
downstream by the detector DLI stage. Trajectories are saved as .h5 files and the
per-task RDS-nuisance draw as a .zarr (compressed) or .npy (uncompressed) set.

The RDS stage runs on ONE shared engine used by both workflows:
``srm_and_sbi_dimer_alp.simulation_rds_runner.run_rds``. This entry point is the
detector shim -- it builds the detector ``WorkflowConfig`` (which namespaces data
under the ``_DETECTOR`` runtime-prefix qualifier so these trajectories never
collide with the biology ones, and selects the diffusion-only builder + the
RDS-nuisance parameter table) and hands it to the shared runner. The biology twin
(``SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py``) is identical except it builds the
biology config; the engine, and therefore the behavior, is shared.

Outputs (the ``{timing_label}`` token, e.g. ``2S_50FPS``, is rendered from
``PARAMETERS.simulation.timing.label``; ``<project_alias>`` carries the
``_DETECTOR`` qualifier):

    <data_bank>/<video_subdir>/<trajectory_repo>/<project_alias>_{timing_label}_TASK_{n}/
        <project_alias>_{timing_label}_TASK_{n}_SIM_{m}.h5                    -- per-simulation trajectory
    <data_bank>/<theta_subdir>/
        <project_alias>_{timing_label}_Nuisance_RDS_Theta_Set_TASK_{n}.zarr   -- per-task RDS-nuisance set

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Simulation_RDS.py \\
        --total-time-seconds 2.0 --split train --tasks 25 --task-simulations 10 --seed None
    (repeat with --split test --tasks 5, and --split eval --tasks 2)
    For the full detector smoke test see section 2.5 in VALIDATION.md.

Diagnostics:
    --probe logs the process resource limits (RLIMIT_NPROC / RLIMIT_NOFILE) at
    startup and a per-simulation line (thread count, open file descriptors,
    resident memory). Logging only; it does not change generation behavior.
"""

import sys

from srm_and_sbi_dimer_alp.simulation_rds_runner import build_rds_parser, run_rds
from srm_and_sbi_dimer_alp.utils import console_log_context
from srm_and_sbi_dimer_alp.workflow import detector_workflow


if __name__ == "__main__":
    cli_args = build_rds_parser().parse_args(sys.argv[1:])
    cfg = detector_workflow()
    with console_log_context(cli_args, "RDS", paths=cfg.console_log_paths):
        run_rds(cfg, cli_args)

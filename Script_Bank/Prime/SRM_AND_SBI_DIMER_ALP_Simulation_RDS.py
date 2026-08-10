"""Entry-point script (biology workflow): generate ReaDDy reaction-diffusion trajectories.

Samples the DIMER model's learnable reaction-diffusion parameters from the
log-uniform prior, runs ReaDDy simulations for each (task, simulation) pair, and
saves the resulting particle trajectories as .h5 files and the parameter samples
as a .zarr (compressed) or .npy (uncompressed) theta set.

The RDS stage runs on ONE shared engine used by both the biology and the detector
workflows: ``srm_and_sbi_dimer_alp.simulation_rds_runner.run_rds``. This entry
point is the biology shim -- it builds the biology ``WorkflowConfig`` and hands it
to the shared runner. The detector twin
(``SRM_AND_SBI_DIMER_ALP_DETECTOR_Simulation_RDS.py``) is identical except it
builds the detector config; the engine, and therefore the behavior, is shared.

Outputs (the ``{timing_label}`` token, e.g. ``2S_50FPS``, is rendered from
``PARAMETERS.simulation.timing.label`` to namespace files by duration + fps):

    <data_bank>/<video_subdir>/<trajectory_repo>/<project_alias>_{timing_label}_TASK_{n}/
        <project_alias>_{timing_label}_TASK_{n}_SIM_{m}.h5       -- per-simulation trajectory
    <data_bank>/<theta_subdir>/
        <project_alias>_{timing_label}_Theta_Set_TASK_{n}.zarr   -- per-task theta sample set

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py \\
        --total-time-seconds 2.0 --tasks 2 --task-simulations 5 --seed None

Diagnostics:
    --probe logs the process resource limits (RLIMIT_NPROC / RLIMIT_NOFILE) at
    startup and a per-simulation line (thread count, open file descriptors,
    resident memory). Logging only; it does not change generation behavior. It
    is the instrumentation used to diagnose the per-simulation ReaDDy-kernel
    resource leak fixed in the shared runner.
"""

import sys

from srm_and_sbi_dimer_alp.simulation_rds_runner import build_rds_parser, run_rds
from srm_and_sbi_dimer_alp.utils import console_log_context
from srm_and_sbi_dimer_alp.workflow import biology_workflow


if __name__ == "__main__":
    cli_args = build_rds_parser().parse_args(sys.argv[1:])
    cfg = biology_workflow()
    with console_log_context(cli_args, "RDS", paths=cfg.console_log_paths):
        run_rds(cfg, cli_args)

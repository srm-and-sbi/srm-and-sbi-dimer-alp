"""Entry-point script (detector workflow): MAP-estimate imaging parameters from real videos.

Part of the detector calibration workflow (see DETECTOR_WORKFLOW.md) -- the workflow that
infers the imaging parameters. Like the biology Experiment stage, it applies the
seed-then-optimize ``map_estimate`` to *real* experimental videos (no ground truth): each
cell's recording is split into model-length chunks, the MAP theta is estimated per chunk, and
the report shows the distribution of inferred parameters per condition. Here the inferred
parameters are the 6 emitter imaging parameters (the detector's inference target), sourced from
``det.DETECTOR_PARAMETERIZATION`` with ``det.build_prior`` + ``det.detector_paths(...)``.

The Experiment stage runs on ONE shared engine used by both workflows:
``srm_and_sbi_dimer_alp.experiment_runner.run_experiment`` (which routes cell discovery /
windowing / sharding through the shared ``experiment_support`` module). This entry point is the
detector shim; the biology twin (``SRM_AND_SBI_DIMER_ALP_Experiment.py``) is identical except it
builds the biology config (10 reaction-diffusion parameters). The engine -- including the
multi-GPU sharding and the ``--merge`` combine step -- is shared.

Outputs (under <data_bank>/<posit_subdir>/<project_alias>_{timing_label}_MAP_Experiment/,
``<project_alias>`` carrying the ``_DETECTOR`` qualifier): report.md +
figures/experiment_<KEY>.png + the combined ``_MAP_Experiment.npz`` + progress.log.

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Experiment.py \\
        --total-time-seconds 2.0 --kinds ALP,BET --max-cells 2 --pool-mode unrestricted
"""

import sys

from srm_and_sbi_dimer_alp.experiment_runner import build_experiment_parser, run_experiment
from srm_and_sbi_dimer_alp.utils import console_log_context
from srm_and_sbi_dimer_alp.workflow import detector_workflow


if __name__ == "__main__":
    cli_args = build_experiment_parser().parse_args(sys.argv[1:])
    cfg = detector_workflow()
    with console_log_context(cli_args, "Experiment", paths=cfg.console_log_paths):
        run_experiment(cfg, cli_args)

"""Entry-point script (biology workflow): MAP-estimate parameters from real microscopy videos.

The companion to ``Evaluation.py``: where Evaluation recovers parameters from held-out
*simulated* EVAL videos (known ground truth), this stage applies the same seed-then-optimize
``map_estimate`` to *real* experimental videos, for which there is no ground truth. Each cell's
long recording is split into model-length chunks; the MAP theta is estimated per chunk, and the
report shows the distribution of inferred parameters per experimental condition (kind), so
conditions can be compared (e.g. ALP vs BET). The biology workflow infers the 10
reaction-diffusion parameters.

The Experiment stage runs on ONE shared engine used by both the biology and the detector
workflows: ``srm_and_sbi_dimer_alp.experiment_runner.run_experiment`` (which routes cell
discovery / windowing / sharding through the shared ``experiment_support`` module). This entry
point is the biology shim; the detector twin (``SRM_AND_SBI_DIMER_ALP_DETECTOR_Experiment.py``)
is identical except it builds the detector config (6 imaging parameters, detector alias). The
engine -- including the multi-GPU sharding and the ``--merge`` combine step -- is shared.

Inputs (real microscopy, copied into the data bank by the user):
    <data_bank>/<experiment_subdir>/Experiment_{KIND}_Cell_{n}_{span}S_RAW.tif
        -- 16-bit raw video, one per (kind, cell); converted to 8-bit and chunked
           into <n_frames>-frame windows (n_frames = model duration x frame rate).

Outputs (under <data_bank>/<posit_subdir>/<project_alias>_{timing_label}_MAP_Experiment/):
    report.md                         -- per-parameter inferred-theta summary by kind
    figures/experiment_<KEY>.png      -- inferred-theta distribution per condition
    <...>_MAP_Experiment.npz          -- inferred_log10, scores, kind/cell/chunk indices
    progress.log                      -- live, tail-able trail

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_Experiment.py \\
        --total-time-seconds 2.0 --kinds ALP,BET --max-cells 2 --pool-mode bounded
"""

import sys

from srm_and_sbi_dimer_alp.experiment_runner import build_experiment_parser, run_experiment
from srm_and_sbi_dimer_alp.utils import console_log_context
from srm_and_sbi_dimer_alp.workflow import biology_workflow


if __name__ == "__main__":
    cli_args = build_experiment_parser().parse_args(sys.argv[1:])
    cfg = biology_workflow()
    with console_log_context(cli_args, "Experiment", paths=cfg.console_log_paths):
        run_experiment(cfg, cli_args)

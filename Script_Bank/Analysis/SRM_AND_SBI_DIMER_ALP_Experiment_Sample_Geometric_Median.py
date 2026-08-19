"""Analysis entry point (biology workflow): the Sample Geometric Median of the Experiment MAP cloud.

ROLE. Reduce the Experiment stage's cloud of MAP estimates -- one per analyzed window of real MET
single-particle-tracking recording -- to a single representative parameter vector, without inventing
a configuration no recording supported. The ten reaction-diffusion parameters are correlated: the
species counts constrain one another through the reactions that interconvert them, and abundance
trades off against the rates that produce it. Summarizing such a cloud by taking each dimension's
median independently composes a vector whose coordinates never co-occurred, and which for a
multimodal cloud lands in the low-density valley between the modes. This analysis instead reports the
Sample Geometric Median (SGM): the actual member minimizing the summed normalized distance to the
rest, so the summary is a real, co-occurring configuration with every correlation intact. It reports
the SGM against the per-dimension vector of medians, for the full cloud and for the prior-bounded
subcollection, with the joint correlation structure and the out-of-prior mass.

WHERE TO RUN. On any machine (CPU only) -- it reads the completed Experiment stage output and neither
loads the estimator nor renders videos. It is a post-hoc, user-driven analysis in Script_Bank/Analysis,
NOT a canonical pipeline stage and never wired into the stage dispatcher.

WHAT IT NEEDS.
  - a completed biology Experiment stage for the run's timing (its MAP output carries the per-row
    condition, cell, and chunk labels this analysis groups by);
  - the run's timing (--total-time-seconds), which locates that output and names the results.

WHAT IT DOES NOT DO.
  - It does NOT recover ground truth. Real recordings have none; the cloud describes where the
    posterior places its mass on real data, and recovery is quantified only on held-out synthetic
    data in the Evaluation stage.
  - It does NOT re-run inference. It summarizes estimates the Experiment stage already produced.

Reads  <data_bank>/<posit>/<alias>_<timing>_MAP_Experiment/<alias>_<timing>_MAP_Experiment.npz
Writes <data_bank>/<posit>/<alias>_<timing>_Experiment_Sample_Geometric_Median/  (report + figures)

Usage:
    MACHINE_PROFILE=<p> python \
        Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Experiment_Sample_Geometric_Median.py \
        --total-time-seconds 2.0 [--condition pooled|MET-FAB|MET-INLB] [--dry-run]
"""
import sys

from srm_and_sbi_dimer_alp.sample_geometric_median_runner import (
    build_parser, run_sample_geometric_median)
from srm_and_sbi_dimer_alp.workflow import biology_workflow


def main(argv=None):
    parser = build_parser(
        "Sample Geometric Median of the biology Experiment MAP cloud "
        "(the correlation-preserving median vector).")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return run_sample_geometric_median(biology_workflow(), args)


if __name__ == "__main__":
    raise SystemExit(main())

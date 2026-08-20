"""Analysis entry point (biology workflow): temporal dynamics of the inferred parameters.

ROLE. The Experiment stage estimates the ten reaction-diffusion parameters independently in every
non-overlapping window of every experimental MET recording. Stacking those windows along time asks
a question the stage cannot: does an inferred value hold still across the recording? A parameter
that is a constant property of the system should be flat. A trend is either real dynamics or an
acquisition confound.

WHAT THIS WORKFLOW CAN AND CANNOT SEE. Biology holds the imaging block FIXED at the calibrated
vector, so this analysis is structurally blind to imaging drift -- it cannot tell a genuine change
in receptor kinetics from a change in how the recording images those receptors. The detector
counterpart of this analysis, run on the SAME recordings, supplies exactly that missing view (the
two share the experimental recordings: the path pattern carries no workflow qualifier, so a given
recording index is the same acquisition in both). Read the two together; neither alone attributes
a cause.

CENTRAL ESTIMATE. The displayed central trajectory is a REAL recording selected by the Sample
Geometric Median over the whole time course, not a cross-recording average -- averaging each
parameter independently composes a vector whose coordinates never co-occurred. A per-time-point
SGM and the retained cross-recording mean are drawn alongside for comparison. The drift statistics
are fit per recording and are therefore independent of that display choice.

WHERE TO RUN. Any machine (CPU only) -- it reads the completed Experiment output and neither loads
the estimator nor renders videos. Post-hoc, user-driven analysis in Script_Bank/Analysis, NOT a
canonical pipeline stage and never wired into the stage dispatcher.

Reads  <data_bank>/<posit>/<alias>_<timing>_MAP_Experiment/<alias>_<timing>_MAP_Experiment.npz
       (+ the sibling MAP_Recovery output, when present, for the recovery annotation)
Writes <data_bank>/<posit>/<alias>_<timing>_MAP_Experiment/temporal_dynamics/
           <key>_temporal.png, <key>_temporal_posterior.png, report.md

Usage:
    MACHINE_PROFILE=<p> python \
        Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Experiment_Temporal_Dynamics.py \
        --total-time-seconds 2.0 [--params rate_dissociation,diffusivity_alp] [--dry-run]
"""
import sys

from srm_and_sbi_dimer_alp.temporal_dynamics_runner import build_parser, run_temporal_dynamics
from srm_and_sbi_dimer_alp.workflow import biology_workflow


def main(argv=None):
    parser = build_parser(
        "Temporal dynamics of the inferred reaction-diffusion parameters across the experimental "
        "recordings (biology workflow).")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return run_temporal_dynamics(biology_workflow(), args)


if __name__ == "__main__":
    raise SystemExit(main())

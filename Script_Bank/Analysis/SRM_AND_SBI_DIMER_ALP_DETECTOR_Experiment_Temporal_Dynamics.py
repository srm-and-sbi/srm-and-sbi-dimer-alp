"""Analysis entry point (detector workflow): temporal dynamics of the inferred imaging parameters.

ROLE. The detector Experiment stage estimates the six imaging parameters independently in every
non-overlapping window of every experimental MET recording. Stacking those windows along time asks
whether the ACQUISITION holds still across the recording -- whether the emitter brightness, the
point-spread width, the photobleaching probability, and the flicker rate that the recording
presents are the same at its end as at its start.

WHY THIS IS THE CONFOUND TEST THE BIOLOGY ANALYSIS CANNOT RUN. The biology temporal analysis holds
imaging fixed, so a trend it finds in a reaction rate cannot be separated from a trend in the
imaging that produced the videos. This analysis measures precisely that imaging trend, on the SAME
recordings (the experimental path pattern carries no workflow qualifier, so a given recording index
is the same acquisition in both workflows). A flat imaging trajectory would push a biology trend
toward genuine dynamics; a drifting one identifies a live confound and localizes it to a channel.
Neither result attributes a cause on its own, and this analysis must not be read as if it did.

WHAT THIS WORKFLOW CANNOT SEE. The detector marginalizes the reaction-diffusion block, so it is
blind to biological drift in the same way the biology analysis is blind to imaging drift. The
asymmetry is the point: each is the other's control.

REFERENCE VALUES. Where an external reference exists it is drawn only on the parameters it
legitimately constrains and only for the conditions it applies to. Several imaging references are
valid for the monomer control alone -- the dimer condition's localization brightness is a
two-label per-detection sum, and its PSF width is dimer-broadened -- and the two population
log-spreads are upper-biased by localization fitting error, so their fitted values are drawn as
upper bounds rather than targets. The report states each reference's provenance and scope.

WHERE TO RUN. Any machine (CPU only). Post-hoc, user-driven analysis, NOT a canonical pipeline
stage and never wired into the stage dispatcher.

Reads  <data_bank>/<posit>/<alias>_<timing>_MAP_Experiment/<alias>_<timing>_MAP_Experiment.npz
       where <alias> carries the _DETECTOR qualifier
Writes that directory's temporal_dynamics/ subdirectory (figures + report.md)

Usage:
    MACHINE_PROFILE=<p> python \
        Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Experiment_Temporal_Dynamics.py \
        --total-time-seconds 2.0 [--params prob_photo_bleach,mu_pc] [--dry-run]
"""
import sys

from srm_and_sbi_dimer_alp.temporal_dynamics_runner import build_parser, run_temporal_dynamics
from srm_and_sbi_dimer_alp.workflow import detector_workflow


def main(argv=None):
    parser = build_parser(
        "Temporal dynamics of the inferred imaging parameters across the experimental recordings "
        "(detector workflow).")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return run_temporal_dynamics(detector_workflow(), args)


if __name__ == "__main__":
    raise SystemExit(main())

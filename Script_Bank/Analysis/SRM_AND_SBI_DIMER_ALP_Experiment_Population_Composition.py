"""Analysis entry point (biology workflow): the inferred population composition, versus its own
validation on synthetic data.

ROLE. The Experiment stage infers three absolute species counts -- A (monomer), B (mobile dimer),
C (immobile dimer) -- in every window of every experimental MET recording. Absolute counts are the
worst-identified coordinates the model has, for an information-theoretic reason rather than a
defect: counting few emitters in a diffraction-limited scene is a square-root-of-n problem, and the
three counts trade off against one another because the reactions interconvert them. This analysis
asks the question those same posteriors answer well -- in what PROPORTION are the receptors
distributed across the three states -- and reports it beside the measured error of that readout on
held-out synthetic videos, so the experimental value and the accuracy of the instrument that produced
it appear in one document.

WHY THE DRAWS AND NOT THE MAP ESTIMATES. A ratio of correlated coordinates is not a function of
their marginals. Every fraction is formed WITHIN each posterior draw and only then averaged, which
carries the count-to-count correlations through and is what makes the composition identifiable from
counts that individually are not. Building a fraction from marginal medians instead would assert a
combination the posterior never drew. The stage must therefore have been run with
``--dump-posterior-samples``; the stored quantiles cannot substitute.

WHAT IS REPORTED. Per condition: the span-averaged composition with the standard error across
RECORDINGS (the replicate unit -- ten windows of one cell are not ten independent measurements), the
first-window composition, the within-recording time course, the per-recording spread, the sensitivity
of the headline to prior-support restriction and to the choice of compositional center, and the
recording-level condition contrast. From the held-out synthetic set: the same quantities' point-
estimate error, the counts-versus-total comparison that explains why the composition is better
identified than its parts, and the accuracy restricted to the dimer-rich region the activated
condition occupies.

WHAT IT DOES NOT DO. It does NOT re-run inference; it derives quantities from estimates already on
disk. It does NOT report interval coverage of a fraction -- that needs joint posterior draws on the
held-out set, which a recovery artifact storing marginal quantiles does not carry. And it does not
turn a model-conditional census into a molecular one: the composition describes the three-species
model as fitted to these recordings.

BIOLOGY ONLY. There is no detector counterpart, and the asymmetry is scientific: the detector
workflow infers imaging parameters and treats the population implicitly, so it has no composition to
compute -- as the detector's ``Nuisance_DLI`` pool has no biology counterpart. Pointing this script's
engine at that workflow fails loudly rather than composing its imaging coordinates.

WHERE TO RUN. Any machine (CPU only, minutes) -- it reads two completed outputs and neither loads the
estimator nor renders videos. Post-hoc, user-driven analysis in Script_Bank/Analysis, NOT a canonical
pipeline stage and never wired into the stage dispatcher.

Reads  <data_bank>/<posit>/<alias>_<timing>_MAP_Experiment/<alias>_<timing>_MAP_Experiment.npz
           (requires `posterior_samples_cloud`: run the stage with --dump-posterior-samples)
       <data_bank>/<posit>/<alias>_<timing>_MAP_Recovery/<alias>_<timing>_MAP_Recovery.npz
           (optional; without it the synthetic validation half is reported as unavailable)
Writes <data_bank>/<posit>/<alias>_<timing>_Experiment_Population_Composition/
           report.md, figures/, and the derived per-window / per-recording arrays as .npz

Usage:
    MACHINE_PROFILE=<p> python \
        Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Experiment_Population_Composition.py \
        --total-time-seconds 2.0 [--bootstrap 10000] [--stratum-threshold 0.7] [--dry-run]
"""
import sys

from srm_and_sbi_dimer_alp.population_composition_runner import (
    build_parser, run_population_composition)
from srm_and_sbi_dimer_alp.workflow import biology_workflow


def main(argv=None):
    parser = build_parser(
        "Inferred population composition across the experimental recordings, with its own recovery "
        "on held-out synthetic data (biology workflow).")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return run_population_composition(biology_workflow(), args)


if __name__ == "__main__":
    raise SystemExit(main())

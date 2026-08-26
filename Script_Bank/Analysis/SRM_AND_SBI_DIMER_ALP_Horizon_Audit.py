"""Analysis entry point (biology workflow): the horizon audit -- does window-slicing a continuous
recording break the estimator's reset assumption?

ROLE. The estimator trains on independently initialized model-window simulations: every training
video starts with freshly placed particles at the drawn counts. The experimental analysis slices
each continuous 20 s recording into consecutive model-length windows and runs the estimator on
every window -- silently assuming that a window whose past evolved for many seconds is
distributed like a reset training simulation of the same length. This audit tests that assumption
under the simulator itself, where the truth is knowable: for each theta drawn from the training
prior it renders ONE continuous long recording (sliced into windows exactly as the Experiment
stage slices real data) and R independently initialized reset windows, runs the SAME estimator on
both, and compares them per window position. The trajectory is the independent unit; windows
within one trajectory are correlated and never treated as replicates.

BOTH OUTCOMES ARE INFORMATIVE. Error or coverage that degrades with window position only in the
continuous ensemble measures the reset artifact -- a horizon mismatch between the training
factorization and the deployment slicing. Stability licenses the window-slicing practice under
the implemented simulator, and re-attributes any window drift seen in the EXPERIMENTAL recordings
to a real-versus-simulator discrepancy rather than to the slicing itself.

TWO KINDS OF ESTIMAND, AUDITED DIFFERENTLY. Rates and diffusivities are constant parameters:
their truth is the drawn theta in every window, so positional structure in their errors is
spurious by definition. The species counts are initial conditions of a dynamic state: after the
first window the population has evolved, so the composition readout is audited against the ACTUAL
population of each window, extracted per frame from the continuous trajectory -- comparing counts
against theta instead would measure state drift, which is dynamics, not estimator error.

FOUR PHASES (CPU generation and GPU inference split cleanly, both resumable and parallel over
disjoint --theta-start/--theta-stop ranges; per-theta output files, skip-if-present):

    --phase prepare    draw the theta cohort from the training prior, persist it once
    --phase generate   simulate + render per theta (CPU; ReaDDy + the shared DLI renderer,
                       imaging pinned to the calibrated Nuisance_DLI + MET SCOPE vector for
                       every render, so the paired contrast cancels everything but the state)
    --phase infer      slice into windows, run the estimator per window (GPU; posterior
                       quantiles + raw draws; default sampler 'unrestricted', matching the
                       experimental-baseline methodology and required for prior exceedance)
    --phase analyze    the audit statistics, report and figures (CPU)

BIOLOGY ONLY. The detector workflow marginalizes the reaction-diffusion block, so a continuous
reactive trajectory has no inferred counterpart to audit; pointing the engine at it fails loudly.

WHERE TO RUN. generate: any CPU machine (ReaDDy required). infer: the GPU server. analyze: any
machine, seconds. Post-hoc, user-driven analysis in Script_Bank/Analysis, NOT a canonical
pipeline stage and never wired into the stage dispatcher.

Reads  <data_bank>/<posit>/<alias>_<timing>_Estimator.npz           (the trained estimator)
       the calibrated Nuisance_DLI artifact (via the shared biology imaging resolver)
Writes <data_bank>/<posit>/<alias>_<timing>_Horizon_Audit/
           cohort_theta.npz, cohort/theta_*.npz, cohort/result_*.npz,
           report.md, figures/, and the aggregated audit arrays as .npz

Usage (typical sequence):
    MACHINE_PROFILE=<p> python Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Horizon_Audit.py \
        --total-time-seconds 2.0 --phase prepare --n-theta 200 --seed 20260825
    MACHINE_PROFILE=<p> python ... --total-time-seconds 2.0 --phase generate \
        --theta-start 0 --theta-stop 50          # one of several parallel workers
    MACHINE_PROFILE=<p> python ... --total-time-seconds 2.0 --phase infer
    MACHINE_PROFILE=<p> python ... --total-time-seconds 2.0 --phase analyze
"""
import sys

from srm_and_sbi_dimer_alp.horizon_audit_runner import build_parser, run_horizon_audit
from srm_and_sbi_dimer_alp.workflow import biology_workflow


def main(argv=None):
    parser = build_parser(
        "Horizon audit: continuous-recording windows versus independently initialized reset "
        "windows, per window position, under the trained estimator (biology workflow).")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return run_horizon_audit(biology_workflow(), args)


if __name__ == "__main__":
    raise SystemExit(main())

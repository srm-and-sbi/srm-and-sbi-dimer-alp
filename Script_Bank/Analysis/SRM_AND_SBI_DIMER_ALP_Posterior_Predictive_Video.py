"""Analysis entry point (biology workflow): posterior-predictive video against a real recording.

ROLE. Take the ten reaction-diffusion parameters INFERRED from one real MET recording, simulate a
video with them at that recording's own length, and put the two side by side. This is the check no
statistic on held-out synthetic data can perform: it asks whether the posterior explains the real
data with a configuration the forward model can actually render. If the synthetic frames do not look
like the recording that produced the parameters, the estimates describe something the simulator
cannot reproduce -- and no amount of calibration on synthetic data would have revealed it.

WHAT IS INFERRED AND WHAT IS FIXED. The reaction-diffusion block comes from the MAP, and the full
reactive system is built -- the reactions are the inference target here, so they must be present.
The imaging block is held FIXED at the calibrated vector the training videos were generated with,
read from the Nuisance_DLI artifact at run time rather than hardcoded, and the five SCOPE camera
parameters are pinned to their MET values. Fixing imaging is what makes the comparison
interpretable: any visible difference is then attributable to the reaction-diffusion parameters,
not to imaging variation introduced by the render.

WHERE TO RUN. Any machine with ReaDDy. It runs one simulation and one render per invocation
(minutes, CPU). Post-hoc, user-driven analysis in Script_Bank/Analysis, NOT a canonical pipeline
stage and never wired into the stage dispatcher.

WHAT IT NEEDS.
  - a completed biology Experiment stage (the MAP database) for the run's timing;
  - the staged experimental recording for the requested condition and cell;
  - the Nuisance_DLI artifact holding the calibrated imaging vector.

WHAT IT DOES NOT DO.
  - It does NOT fit anything, and a visual mismatch is not by itself proof the estimates are wrong;
    it localizes where the forward model and reality diverge.

Writes <data_bank>/<posit>/<alias>_<timing>_Posterior_Predictive_Video/
           <stem>_{Synthetic_Video.npz, Comparison.png, Trajectory.h5}

Usage:
    MACHINE_PROFILE=<p> python \
        Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Posterior_Predictive_Video.py \
        --total-time-seconds 2.0 --kind MET-FAB --cell 0 [--map-source cell-sgm] [--dry-run]
"""
import sys

from srm_and_sbi_dimer_alp.posterior_predictive_video_runner import (
    build_parser, run_posterior_predictive_video)
from srm_and_sbi_dimer_alp.workflow import biology_workflow


def main(argv=None):
    parser = build_parser("Posterior-predictive video comparison (biology workflow).")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return run_posterior_predictive_video(biology_workflow(), args)


if __name__ == "__main__":
    raise SystemExit(main())

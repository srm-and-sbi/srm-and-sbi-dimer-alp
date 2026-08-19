"""Analysis entry point (detector workflow): experimental-versus-synthetic embedding distance.

Scores how far the REAL MET recordings sit from the synthetic videos in the trained network's
embedding space -- the representation the posterior actually consumes. For the detector workflow the
imaging parameters ARE the inference target, so the synthetic videos are meant to reproduce the real
ones' appearance and a measured gap is a realism failure of the imaging forward model. Closing it is
the goal of the detector calibration. Design and justification: DETECTOR_WORKFLOW.md, section
"Quantitative experimental-versus-synthetic distance".

The measurement, its two tests (MMD and C2ST, both blocked by recording), the options, and the
outputs are identical for both workflows and documented once in the authoritative companion:

    SRM_AND_SBI_DIMER_ALP_Embedding_Space_Distance.md

This shim differs only in the config it builds (detector_workflow()): the detector parameterization,
its report-facing parameter descriptions, and the _DETECTOR-aliased paths. Outputs land under
<data_bank>/<posit>/<alias>_<timing>_Embedding_Space_Distance/, where <alias> carries _DETECTOR.

Usage:
    MACHINE_PROFILE=<p> python \
        Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Embedding_Space_Distance.py \
        --total-time-seconds 2.0 [--kinds MET-FAB,MET-INLB] [--eval-tasks N] [--dry-run]
"""
import sys

from srm_and_sbi_dimer_alp.embedding_space_distance_runner import (
    build_parser, run_embedding_space_distance)
from srm_and_sbi_dimer_alp.workflow import detector_workflow


def main(argv=None):
    parser = build_parser(
        "Experimental-versus-synthetic embedding distance (detector realism analysis).")
    args = parser.parse_args(sys.argv[1:] if argv is None else argv)
    return run_embedding_space_distance(detector_workflow(), args)


if __name__ == "__main__":
    raise SystemExit(main())

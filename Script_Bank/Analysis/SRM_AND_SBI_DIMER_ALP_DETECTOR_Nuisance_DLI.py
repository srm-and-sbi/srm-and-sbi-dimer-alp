"""Analysis entry point (Detector workflow): construct the Nuisance_DLI from the calibrated posterior.

Post-hoc analysis (lives in Script_Bank/Analysis; NOT a canonical stage, never wired into the
dispatcher). It is the required first step before any Nuisance_DLI is used (see "Constructing the
Nuisance_DLI (the analysis step)" in DETECTOR_WORKFLOW.md): it lets a person SEE the calibrated
imaging posterior and DECIDE how to turn it into the Nuisance_DLI, in the value-based vocabulary
(fixed / uniform / posterior / samples). The analysis suggests; the user decides the ultimate values.

It is purely additive: it imports and calls the core modules (detector_nuisance_dli,
detector_parameterization) and reads the completed Detector Experiment outputs; it modifies nothing.

Two modes:
  --emit-template (default) : read the completed Detector Experiment MAP outputs, present the imaging
                    posterior (per-parameter pooled MAP + credible interval), and emit a value-based
                    spec template pre-filled with those numbers as SUGGESTIONS (plus the pooled MAP
                    vector as a candidate `samples` source). The user then EDITS the spec.
  --build         : read the user-edited spec, VALIDATE it (structure + prior box, via the gate's
                    load_spec), build the pooled Nuisance_DLI, and persist the artifact.

Reads  <data_bank>/<posit>/<alias>_DETECTOR_{timing}_MAP_Experiment/<...>.npz   (inferred_log10 MAP)
Writes <data_bank>/<posit>/<alias>_DETECTOR_{timing}_Nuisance_DLI_Spec.toml     (--emit-template)
       <data_bank>/<posit>/<alias>_DETECTOR_{timing}_Nuisance_DLI.npz           (--build)

Usage:
    MACHINE_PROFILE=<p> python .../SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py --total-time-seconds 2.0 --emit-template
    (edit the emitted _Nuisance_DLI_Spec.toml: set each parameter's role and its ultimate values)
    MACHINE_PROFILE=<p> python .../SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py --total-time-seconds 2.0 --build
    (add --dry-run to resolve paths/inputs without computing)
"""
import argparse
import sys
from pathlib import Path

import numpy as np

from srm_and_sbi_dimer_alp import detector_nuisance_dli as ndli
from srm_and_sbi_dimer_alp import detector_parameterization as det
from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming


def _resolve(total_time_seconds):
    """Detector-namespaced paths + the completed Experiment MAP .npz."""
    timing = RunTiming(total_time_seconds=total_time_seconds, frames=PARAMETERS.simulation.timing)
    timing_label = timing.label
    data_bank_root = PARAMETERS.machine.data_bank_root
    paths = det.detector_paths(PARAMETERS.paths)                 # _DETECTOR-aliased namespace
    posit_dir = data_bank_root / paths.posit_subdir
    out_dir = paths.experiment_recovery_dir(data_bank_root, timing_label)   # MAP_Experiment dir
    exp_npz = out_dir / (out_dir.name + ".npz")
    return paths, posit_dir, timing_label, exp_npz


def _pooled_suggestions(inferred_log10, imaging_keys):
    """Per-parameter posterior-derived suggestions (log10): MAP=median, CI=[5th,95th] pct, pooled."""
    return {k: {"map": float(np.median(inferred_log10[:, i])),
                "ci_low": float(np.percentile(inferred_log10[:, i], 5)),
                "ci_high": float(np.percentile(inferred_log10[:, i], 95))}
            for i, k in enumerate(imaging_keys)}


def _emit_template(args, paths, posit_dir, timing_label, exp_npz):
    imaging_keys = [e["KEY"] for e in det.DETECTOR_PARAMETERIZATION]
    plo, phi = det.theta_lower_bound(), det.theta_upper_bound()
    spec = ndli.spec_path(posit_dir, paths.project_alias, timing_label)
    samples_npz = posit_dir / f"{paths.project_alias}_{timing_label}_Nuisance_DLI_PosteriorSamples.npz"

    if args.dry_run:
        print("[DRY RUN] --emit-template would read:")
        print(f"    Experiment MAP output : {exp_npz}  [{'OK' if exp_npz.exists() else 'MISSING'}]")
        print("[DRY RUN] would write:")
        print(f"    spec template         : {spec}")
        print(f"    pooled-MAP vector     : {samples_npz}")
        return

    if not exp_npz.exists():
        raise FileNotFoundError(
            f"Detector Experiment MAP output not found:\n    {exp_npz}\n"
            f"Run the Detector Experiment stage first — it produces the imaging posterior on the real "
            f"recordings that this analysis inspects.")
    with np.load(str(exp_npz), allow_pickle=True) as d:
        inferred_log10 = np.asarray(d["inferred_log10"])
    if inferred_log10.ndim != 2 or inferred_log10.shape[0] == 0:
        raise ValueError(
            f"Detector Experiment output carries no usable MAP estimates "
            f"(inferred_log10 shape {inferred_log10.shape}):\n    {exp_npz}\n"
            f"Re-run the Detector Experiment stage so it produces at least one MAP estimate on the "
            f"real recordings before constructing the Nuisance_DLI.")
    if inferred_log10.shape[1] != len(imaging_keys):
        raise ValueError(
            f"Detector Experiment output has {inferred_log10.shape[1]} imaging column(s) but the "
            f"Detector parameterization declares {len(imaging_keys)}:\n    {exp_npz}")
    n = inferred_log10.shape[0]
    sugg = _pooled_suggestions(inferred_log10, imaging_keys)

    # ---- present the calibrated imaging posterior (the "see it" step) ----
    print(f"\nCalibrated imaging posterior (pooled over {n} real MAP windows; log10; SUGGESTIONS only):")
    print(f"  {'parameter':<20}{'MAP':>9}{'CI5':>9}{'CI95':>9}    prior box")
    for i, k in enumerate(imaging_keys):
        s = sugg[k]
        print(f"  {k:<20}{s['map']:>9.3f}{s['ci_low']:>9.3f}{s['ci_high']:>9.3f}    [{plo[i]:+.2f}, {phi[i]:+.2f}]")

    np.savez_compressed(str(samples_npz), samples=inferred_log10)
    ndli.emit_spec_template(spec, imaging_keys, sugg, timing_label=timing_label,
                            provenance={"experiment_npz": str(exp_npz), "n_windows": int(n),
                                        "pooled_map_samples": str(samples_npz)})
    print(f"\nEmitted value-based spec template (pre-filled with the suggestions above):\n    {spec}")
    print(f"Pooled per-window MAP vector (a candidate `samples` source):\n    {samples_npz}")
    print("\nNEXT: edit the spec — set each parameter's role and its ultimate values — then run --build.")


def _build(args, paths, posit_dir, timing_label, exp_npz):
    imaging_keys = [e["KEY"] for e in det.DETECTOR_PARAMETERIZATION]
    plo, phi = det.theta_lower_bound(), det.theta_upper_bound()
    spec = ndli.spec_path(posit_dir, paths.project_alias, timing_label)
    art = ndli.artifact_path(posit_dir, paths.project_alias, timing_label)

    if args.dry_run:
        print("[DRY RUN] --build would read:")
        print(f"    spec : {spec}  [{'OK' if spec.exists() else 'MISSING'}]")
        if not spec.exists():
            print("    (run --emit-template first, then edit the spec.)")
            return
        spec_dict = ndli.load_spec(spec, imaging_keys, plo, phi)  # validates structure + prior box
        form = spec_dict.get("block", {}).get("form", "perparam")
        if form == "posterior":
            print("    spec valid (form='posterior') — NOT buildable yet: posterior materialization "
                  "needs a trained Detector estimator conditioned on the real recordings. Nothing "
                  "would be written; use form='samples'/'fixed'/'uniform' meanwhile.")
            return
        if form == "samples":
            sp = spec_dict.get("block", {}).get("samples_path", "")
            ok = bool(sp) and Path(sp).exists()
            print(f"    samples_path : {sp or '(empty)'}  [{'OK' if ok else 'MISSING'}]")
        print(f"[DRY RUN] spec valid (form={form}); would build + write:\n    {art}")
        return

    if not spec.exists():
        raise FileNotFoundError(
            f"Nuisance_DLI spec not found:\n    {spec}\nRun --emit-template first, then edit the spec.")
    spec_dict = ndli.load_spec(spec, imaging_keys, plo, phi)      # validates structure + prior box
    form = spec_dict.get("block", {}).get("form", "perparam")
    if form == "posterior":
        raise NotImplementedError(
            "form='posterior' draws from the trained imaging estimator conditioned on the real "
            "recordings; that materialization is finalized and validated once a trained Detector "
            "estimator exists. Meanwhile use form='samples' (point [block].samples_path at the emitted "
            "*_Nuisance_DLI_PosteriorSamples.npz pooled-MAP vector), or the fixed/uniform forms.")
    if form == "samples":
        sp = spec_dict.get("block", {}).get("samples_path", "")
        if not (sp and Path(sp).exists()):
            raise FileNotFoundError(
                f"form='samples' points [block].samples_path at a file that does not exist:\n"
                f"    {sp or '(empty)'}\n"
                f"Point it at the emitted *_Nuisance_DLI_PosteriorSamples.npz (or your own sample vector).")
    nu = ndli.build_nuisance_dli(spec_dict, imaging_keys, plo, phi, pool_mode=args.pool_mode)
    nu.flush(str(art))
    print(f"Built pooled Nuisance_DLI (form={form}, kind={nu.kind}) and saved to:\n    {art}")


def main(args):
    resolved = _resolve(args.total_time_seconds)
    (_build if args.build else _emit_template)(args, *resolved)


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Construct the Nuisance_DLI from the calibrated Detector posterior (analysis step).")
    p.add_argument("--total-time-seconds", type=float, required=True,
                   help="recording duration (sets the timing_label, e.g. 2.0 -> 2S_50FPS).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--emit-template", dest="build", action="store_false",
                      help="inspect the imaging posterior + emit the spec template (default).")
    mode.add_argument("--build", dest="build", action="store_true",
                      help="build the Nuisance_DLI from the user-edited spec.")
    p.set_defaults(build=False)
    p.add_argument("--pool-mode", choices=["bounded", "unrestricted"], default="bounded",
                   help="posterior-form draw mode (canonical parity; affects only form='posterior').")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve paths and report what would be read/written; write nothing.")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))

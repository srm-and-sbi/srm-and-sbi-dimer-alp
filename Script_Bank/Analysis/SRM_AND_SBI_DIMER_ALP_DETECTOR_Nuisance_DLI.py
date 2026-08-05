"""Analysis entry point (Detector workflow): construct the Nuisance_DLI from the calibration.

Post-hoc analysis (lives in Script_Bank/Analysis; NOT a canonical stage, never wired into the
dispatcher). It is the required first step before any Nuisance_DLI is used (see "Nuisance and
artifact design" in DETECTOR_WORKFLOW.md): it lets a person SEE the calibrated imaging and DECIDE
how to turn it into the marginalized imaging distribution the production run draws from.

Self-contained: the detector estimator is a REQUIRED prerequisite, and this step reads the real
recordings, chunks them into model-length windows (windowing owned here), and runs the estimator
itself — it does not depend on the Detector Experiment stage's output. Both modes therefore load
the estimator; the build is the GPU step (run it on a machine with a capable GPU).

Two modes:
  --emit-template (default) : compute the posterior_sample_pool (cached) under --pool-mode and present
                    the calibrated imaging (per-parameter 5th/95th percentiles), then emit a value-based
                    spec pre-filled with those percentiles as SUGGESTIONS. The user then EDITS it to
                    choose `posterior_sample_pool_choice` (and, for `box_user`, the ranges).
  --build         : read the finalized spec, obtain the pool (reused from the cache, or computed once on
                    the GPU), turn it into the chosen representation, and flush the self-contained
                    artifact plus a report + 1-D marginals plot beside it.

The pool is the GPU cost and is cached (keyed on the build inputs + the estimator checksum), so
--emit-template computes it once and --build reuses it; raw/gaussian/box share the posterior-sample
pool, map_estimate_pool uses a MAP pool, box_user needs none. --repool forces a recompute.

The five choices and their meaning are documented in the spec template and in the companion note
SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.md; the artifact machinery is in detector_nuisance_dli.py.

Reads  <data_bank>/<experiment_subdir>/Experiment_<KIND>_Cell_<n>_<span>S_RAW.tif   (real recordings)
       <data_bank>/<posit>/<alias>_{timing}_Estimator.npz                          (required estimator)
Writes <data_bank>/<posit>/<alias>_{timing}_Nuisance_DLI_Spec.toml                 (--emit-template)
       <data_bank>/<posit>/<alias>_{timing}_Nuisance_DLI_<Kind>Pool.npz            (cached pool)
       <data_bank>/<posit>/<alias>_{timing}_Nuisance_DLI.npz                       (--build)
       <data_bank>/<posit>/<alias>_{timing}_Nuisance_DLI_Analysis/                 (report + marginals)

Usage:
    MACHINE_PROFILE=<p> python .../SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py \\
        --total-time-seconds 2.0 --experiment-span-seconds 20 --emit-template
    (edit the emitted _Nuisance_DLI_Spec.toml: set posterior_sample_pool_choice, pool_mode, ...)
    MACHINE_PROFILE=<p> python .../SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py \\
        --total-time-seconds 2.0 --experiment-span-seconds 20 --build
    (add --dry-run to resolve paths/inputs without loading the estimator or computing)
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import tifffile
import torch
from matplotlib.figure import Figure

from srm_and_sbi_dimer_alp import artifacts
from srm_and_sbi_dimer_alp import detector_nuisance_dli as ndli
from srm_and_sbi_dimer_alp import detector_parameterization as det
from srm_and_sbi_dimer_alp.diagnostics import DiagnosticReporter
from srm_and_sbi_dimer_alp.inference_support import resolve_topology
from srm_and_sbi_dimer_alp.io import convert_video_dtype
from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming


def _discover_cells(experiment_dir, kind, span):
    """Cell indices with an ``Experiment_{kind}_Cell_{n}_{span}S_RAW.tif`` recording, sorted.
    Mirrors the Detector Experiment stage's discovery so both see the same recordings."""
    cells = []
    for path in experiment_dir.glob(f"Experiment_{kind}_Cell_*_{span}S_RAW.tif"):
        stem = path.stem
        try:
            cells.append(int(stem.split("_Cell_")[1].split("_")[0]))
        except (IndexError, ValueError):
            continue
    return sorted(cells)


def _resolve(total_time_seconds):
    """Detector-namespaced paths + timing + the imaging keys and prior box."""
    timing = RunTiming(total_time_seconds=total_time_seconds, frames=PARAMETERS.simulation.timing)
    data_bank_root = PARAMETERS.machine.data_bank_root
    paths = det.detector_paths(PARAMETERS.paths)                 # _DETECTOR-aliased namespace
    posit_dir = data_bank_root / paths.posit_subdir
    experiment_dir = data_bank_root / paths.experiment_subdir
    estimator_path = posit_dir / f"{paths.project_alias}_{timing.label}_Estimator.npz"
    imaging_keys = [e["KEY"] for e in det.DETECTOR_PARAMETERIZATION]
    plo, phi = det.theta_lower_bound(), det.theta_upper_bound()
    return dict(timing=timing, timing_label=timing.label, data_bank_root=data_bank_root,
                paths=paths, posit_dir=posit_dir, experiment_dir=experiment_dir,
                estimator_path=estimator_path, imaging_keys=imaging_keys, plo=plo, phi=phi)


def _chunk_geometry(timing, span, chunk_step_seconds):
    """(n_frames, step_frames) for the model-length sliding window over a span-second recording."""
    n_frames = timing.frame_count
    window_seconds = timing.total_time_seconds
    step_seconds = chunk_step_seconds if chunk_step_seconds else int(window_seconds)
    if (window_seconds != int(window_seconds) or step_seconds < 1
            or step_seconds > int(window_seconds)
            or int(window_seconds) % int(step_seconds) != 0):
        raise SystemExit(
            f"--chunk-step-seconds={step_seconds} is invalid: a positive integer dividing the "
            f"model window ({window_seconds:g} s) and not exceeding it.")
    step_frames = int(round(step_seconds / timing.frame_time_seconds))
    return n_frames, step_frames


def _read_all_chunks(R, kinds, span, n_frames, step_frames, max_cells):
    """Read every real recording and cut it into model-length windows; return one flat list.

    Pools chunks across every (kind, cell), because the Nuisance_DLI is pooled across conditions
    (a by-kind split is only a diagnostic, never the constructed artifact). Returns
    ``(chunks, summary)`` where ``chunks`` is a list of ``(n_frames, H, W)`` uint8 arrays.
    """
    chunks, per_kind = [], {}
    for kind in kinds:
        cells = _discover_cells(R["experiment_dir"], kind, span)
        if max_cells > 0:
            cells = cells[:max_cells]
        n_cell_chunks = 0
        for cell in cells:
            tif_path = R["paths"].experiment_video_path(kind, cell, span, R["data_bank_root"])
            if not tif_path.exists():
                continue
            raw = tifffile.imread(str(tif_path))                 # (frames, H, W) uint16
            video8 = convert_video_dtype(raw, bits_from=16, bits_to=8)
            for start in range(0, video8.shape[0] - n_frames + 1, step_frames):
                chunks.append(video8[start:start + n_frames])
                n_cell_chunks += 1
        per_kind[kind] = (len(cells), n_cell_chunks)
    return chunks, per_kind


def _load_posterior(R):
    """Load the required detector estimator onto this machine's device (fail loud if absent)."""
    if not R["estimator_path"].exists():
        raise FileNotFoundError(
            f"Detector estimator not found:\n    {R['estimator_path']}\n"
            f"The Nuisance_DLI is built from the calibration; train the Detector Inference stage "
            f"first so the estimator exists.")
    topo = resolve_topology()
    device = topo.device
    vista_device = torch.device("cpu")
    posterior = artifacts.load_estimator(str(R["estimator_path"]), device=str(device), expected_parameter_keys=det.DETECTOR_PARAMETER_KEYS)
    posterior.posterior_estimator.to(device)
    if device.type == "cuda":
        posterior.prior = det.build_prior(device=str(device))    # rebuild on this device
    return posterior, device, vista_device


def _estimator_sha256(R):
    """The estimator's weights checksum (a pool-cache key), read from the .npz manifest
    without loading the estimator or touching the GPU."""
    return artifacts.load_estimator_manifest(str(R["estimator_path"]))["weights_sha256"]


def _pool_provenance(R, args, pool_mode, n_per):
    """The cache key for a pool built from these inputs (see detector_nuisance_dli)."""
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    return ndli.pool_provenance(
        pool_mode=pool_mode, n_per_chunk=n_per, span_seconds=args.experiment_span_seconds,
        chunk_step_seconds=args.chunk_step_seconds, kinds=kinds, max_cells=args.max_cells,
        estimator_sha256=_estimator_sha256(R))


def _get_pool(R, args, pool_kind, pool_mode, n_frames, step_frames, n_per):
    """Cache-or-compute the pool of ``pool_kind`` (PosteriorSample | MapEstimate).

    Returns ``(pool, source)``. Loads the matching cache instantly (CPU) when its params and
    the estimator checksum are unchanged; otherwise loads the estimator, reads and chunks the
    recordings, computes the pool on the GPU, and caches it. ``--repool`` forces a recompute.
    """
    cache = ndli.pool_cache_path(R["posit_dir"], R["paths"].project_alias, R["timing_label"],
                                 pool_kind)
    prov = _pool_provenance(R, args, pool_mode, n_per)
    if not args.repool:
        cached = ndli.load_pool_if_fresh(cache, prov)
        if cached is not None:
            return cached, f"cache ({cache.name})"
    posterior, device, vista_device = _load_posterior(R)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    chunks, _ = _read_all_chunks(R, kinds, args.experiment_span_seconds, n_frames, step_frames,
                                 args.max_cells)
    if not chunks:
        raise FileNotFoundError(
            f"No experimental recordings found under {R['experiment_dir']} for kinds {kinds} "
            f"at span {args.experiment_span_seconds}s. Stage the real recordings first.")
    eval_cfg = PARAMETERS.inference.evaluation
    if pool_kind == "MapEstimate":
        pool = ndli.build_map_estimate_pool(posterior, chunks, device, vista_device, eval_cfg,
                                            pool_mode=pool_mode)
    else:
        pool = ndli.build_posterior_sample_pool(
            posterior, chunks, device, vista_device, n_per_chunk=n_per,
            theta_prex_batch_size=eval_cfg.theta_prex_batch_size, pool_mode=pool_mode)
    ndli.save_pool(cache, pool, prov)
    return pool, f"computed on GPU + cached ({cache.name})"


def _emit_template(args, R):
    spec = ndli.spec_path(R["posit_dir"], R["paths"].project_alias, R["timing_label"])
    n_frames, step_frames = _chunk_geometry(R["timing"], args.experiment_span_seconds,
                                             args.chunk_step_seconds)

    if args.dry_run:
        print("[DRY RUN] --emit-template would read the estimator + recordings and write:")
        print(f"    estimator   : {R['estimator_path']}  [{'OK' if R['estimator_path'].exists() else 'MISSING'}]")
        print(f"    recordings  : {R['experiment_dir']}/Experiment_<KIND>_Cell_<n>_{args.experiment_span_seconds}S_RAW.tif")
        print(f"    spec        : {spec}")
        return

    eval_cfg = PARAMETERS.inference.evaluation
    n_per = args.n_per_chunk or eval_cfg.posterior_samples
    # The posterior-sample pool (drawn under --pool-mode) feeds the percentile suggestions; it is
    # cached, so a later raw/gaussian/box --build under the same pool_mode reuses it without GPU.
    pool, source = _get_pool(R, args, "PosteriorSample", args.pool_mode, n_frames, step_frames, n_per)
    p05 = np.percentile(pool, 5, axis=0)
    p95 = np.percentile(pool, 95, axis=0)
    sugg = {k: {"ci_low": float(p05[i]), "ci_high": float(p95[i])}
            for i, k in enumerate(R["imaging_keys"])}

    print(f"\nCalibrated imaging (posterior pool [{source}], pool_mode={args.pool_mode}; log10; SUGGESTIONS only):")
    print(f"  {'parameter':<20}{'p05':>10}{'p95':>10}    prior box")
    for i, k in enumerate(R["imaging_keys"]):
        print(f"  {k:<20}{p05[i]:>10.3f}{p95[i]:>10.3f}    [{R['plo'][i]:+.2f}, {R['phi'][i]:+.2f}]")

    ndli.emit_spec_template(
        spec, R["imaging_keys"], sugg, pool_mode=args.pool_mode,
        provenance={"timing_label": R["timing_label"], "estimator": str(R["estimator_path"]),
                    "pool_chunks": int(pool.shape[0] // n_per), "samples_per_chunk": int(n_per),
                    "pool_mode": args.pool_mode, "pool_source": source})
    print(f"\nEmitted value-based spec (pre-filled with the percentiles above):\n    {spec}")
    print("\nNEXT: edit the spec — set posterior_sample_pool_choice (default 'raw') and pool_mode; "
          "for 'box_user' also set the [imaging.<KEY>] ranges — then run --build.")


def _build(args, R):
    spec = ndli.spec_path(R["posit_dir"], R["paths"].project_alias, R["timing_label"])
    art = ndli.artifact_path(R["posit_dir"], R["paths"].project_alias, R["timing_label"])
    n_frames, step_frames = _chunk_geometry(R["timing"], args.experiment_span_seconds,
                                            args.chunk_step_seconds)

    if not spec.exists():
        if args.dry_run:
            print(f"[DRY RUN] --build: spec not found ({spec.name}); run --emit-template first, "
                  f"then edit it.")
            return
        raise FileNotFoundError(
            f"Nuisance_DLI spec not found:\n    {spec}\nRun --emit-template first, then edit the spec.")
    spec_dict = ndli.load_spec(spec, R["imaging_keys"], R["plo"], R["phi"])   # structure + prior-box
    block = spec_dict["block"]
    choice = block["posterior_sample_pool_choice"]
    pool_mode = block.get("pool_mode", "bounded")
    pool_kind = ndli.POOL_KINDS[choice]                 # None for box_user
    n_per = args.n_per_chunk or PARAMETERS.inference.evaluation.posterior_samples

    if args.dry_run:
        print(f"[DRY RUN] spec valid; posterior_sample_pool_choice={choice}, pool_mode={pool_mode}.")
        if pool_kind is None:
            print(f"    box_user: builds from the spec ranges alone (no pool, no GPU). Would write:\n    {art}")
        else:
            cache = ndli.pool_cache_path(R["posit_dir"], R["paths"].project_alias,
                                         R["timing_label"], pool_kind)
            fresh = (cache.exists() and R["estimator_path"].exists()
                     and ndli.load_pool_if_fresh(cache, _pool_provenance(R, args, pool_mode, n_per)) is not None)
            print(f"    {pool_kind} pool cache: "
                  f"{'FRESH -> reuse, no GPU' if fresh else 'missing/stale -> compute on GPU'} ({cache.name})")
            print(f"    would write:\n    {art}  (+ report + marginals plot)")
        return

    if pool_kind is None:
        nu = ndli.build_nuisance_dli(spec_dict, R["imaging_keys"], R["plo"], R["phi"])
    else:
        pool, source = _get_pool(R, args, pool_kind, pool_mode, n_frames, step_frames, n_per)
        print(f"pool [{pool_kind}]: {source}")
        nu = ndli.build_nuisance_dli(spec_dict, R["imaging_keys"], R["plo"], R["phi"], pool=pool)
    nu.flush(str(art))
    report_dir = _write_nuisance_report(nu, R, art)
    print(f"Built pooled Nuisance_DLI (posterior_sample_pool_choice={choice}, "
          f"pool_mode={nu.pool_mode}) and saved to:\n    {art}\n"
          f"Analysis (report + 1-D marginals plot):\n    {report_dir}")


def _figure_nuisance_marginals(draws, keys, prior_low, prior_high):
    """1-D marginal histograms of the built nuisance, one panel per imaging parameter, with
    the imaging prior box marked (red dashes) so the distribution's placement is legible."""
    n = len(keys)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig = Figure(figsize=(3.2 * ncol, 2.8 * nrow), layout="constrained")
    for i, k in enumerate(keys):
        ax = fig.add_subplot(nrow, ncol, i + 1)
        ax.hist(draws[:, i], bins=50, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.2)
        ax.axvline(prior_low[i], color="#C44E52", linestyle="--", linewidth=1.0)
        ax.axvline(prior_high[i], color="#C44E52", linestyle="--", linewidth=1.0)
        ax.set_title(k, fontsize=9)
        ax.set_xlabel("value (log10)", fontsize=7)
        ax.tick_params(labelsize=6)
    fig.suptitle("Nuisance_DLI 1-D marginals  (red dashes = imaging prior box)", fontsize=11)
    return fig


def _write_nuisance_report(nu, R, art, n_draws=10000):
    """Persist a report.md + a 1-D marginals figure beside the built artifact (the record a
    person reads to judge, and choose between, nuisance constructions)."""
    report_dir = art.with_name(art.stem + "_Analysis")
    reporter = DiagnosticReporter(
        stage="Nuisance_DLI", enabled=True, dump=True, dump_dir=report_dir,
        run_label=f"{R['paths'].project_alias}_{R['timing_label']}")
    draws = nu.sample(n_draws)
    plo, phi = np.asarray(R["plo"], dtype=float), np.asarray(R["phi"], dtype=float)
    reporter.table(
        "Nuisance_DLI provenance", ["field", "value"],
        [["artifact", str(art)],
         ["posterior_sample_pool_choice", nu.posterior_sample_pool_choice],
         ["pool_mode", nu.pool_mode],
         ["parameters", str(len(nu.parameter_keys))],
         ["marginal draws", str(n_draws)]],
        note="How the calibrated imaging is represented for production marginalization.")
    rows = []
    for i, k in enumerate(nu.parameter_keys):
        col = draws[:, i]
        oob = float(np.mean((col < plo[i]) | (col > phi[i])))
        rows.append([k, f"{np.median(col):+.3f}",
                     f"[{np.percentile(col, 5):+.3f}, {np.percentile(col, 95):+.3f}]",
                     f"[{plo[i]:+.2f}, {phi[i]:+.2f}]", f"{oob:.3f}"])
    reporter.table(
        "Per-parameter marginal summary (log10)",
        ["parameter", "median", "[p05, p95]", "prior box", "frac outside prior"], rows,
        note="Drawn from the built nuisance, so this is exactly what production will sample. "
             "'frac outside prior' is > 0 only for pool_mode='unrestricted'; bounded and the box "
             "forms stay within the prior box.")
    reporter.save_figure(
        "nuisance_marginals",
        _figure_nuisance_marginals(draws, nu.parameter_keys, plo, phi),
        caption="1-D marginal of each imaging parameter under the built nuisance (sampled from the "
                "artifact); red dashes mark the imaging prior box.")
    reporter.write_report()
    return report_dir


def main(args):
    R = _resolve(args.total_time_seconds)
    (_build if args.build else _emit_template)(args, R)


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Construct the Nuisance_DLI from the Detector calibration (analysis step).")
    p.add_argument("--total-time-seconds", type=float, required=True,
                   help="model window / recording duration for the estimator (sets the timing label).")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--emit-template", dest="build", action="store_false",
                      help="inspect the calibrated imaging + emit the spec template (default).")
    mode.add_argument("--build", dest="build", action="store_true",
                      help="build the Nuisance_DLI from the finalized spec.")
    p.set_defaults(build=False)
    p.add_argument("--experiment-span-seconds", type=int, default=20,
                   help="duration (s) of the real recordings to read (default 20).")
    p.add_argument("--kinds", type=str, default="ALP,BET",
                   help="comma-separated recording kinds to pool (default 'ALP,BET').")
    p.add_argument("--chunk-step-seconds", type=int, default=None,
                   help="sliding-window step (s); default = the model window (non-overlapping).")
    p.add_argument("--max-cells", type=int, default=0,
                   help="cap cells per kind (0 = all; default 0).")
    p.add_argument("--n-per-chunk", type=int, default=None,
                   help="posterior draws per chunk, both modes (default: the evaluation config's "
                        "posterior_samples).")
    p.add_argument("--pool-mode", choices=("bounded", "unrestricted"), default="bounded",
                   help="emit-only: the mode the suggestion pool is drawn under, written as the "
                        "spec's pool_mode default (default 'bounded'). --build reads pool_mode from "
                        "the finalized spec, not this flag. Use 'unrestricted' when the posterior "
                        "sits largely outside the prior box (bounded rejection then barely accepts).")
    p.add_argument("--repool", action="store_true",
                   help="force recomputing the pool on the GPU even if a fresh cache exists.")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve paths and report what would be read/written; load nothing, compute nothing.")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))

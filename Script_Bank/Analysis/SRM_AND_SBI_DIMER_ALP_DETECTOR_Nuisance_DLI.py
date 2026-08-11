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

The cached pool is SELF-DESCRIBING: beside the ``(N, D)`` matrix and its provenance it stores, per
row, ``kind_index`` (into ``kinds``), ``cell``, and ``chunk`` (the sliding-window index) — the
condition and time-window position of every row — plus a ``pool_format_version`` (see
``detector_nuisance_dli.save_pool``). A consumer can therefore filter the pool by condition or
acquisition from the file alone, without re-deriving the positional build order. A legacy pool built
before this scheme carries no labels; ``--migrate-pool-labels`` (CPU) adds them to an existing pool by
borrowing the aligned labels from the Detector Experiment MAP output, after verifying the ordering.

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
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch
from matplotlib.figure import Figure

from srm_and_sbi_dimer_alp import artifacts
from srm_and_sbi_dimer_alp.experiment_support import (
    assert_consistent_shard_set,
    discover_cells,
    load_shards,
    merge_shard_arrays,
    read_cell_chunks,
    save_shard,
    shard_by_rank,
)
from srm_and_sbi_dimer_alp import detector_nuisance_dli as ndli
from srm_and_sbi_dimer_alp import detector_parameterization as det
from srm_and_sbi_dimer_alp.diagnostics import DiagnosticReporter
from srm_and_sbi_dimer_alp.inference_support import resolve_topology
from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming


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


def _read_all_chunks(R, kinds, span, n_frames, step_frames, max_cells, topo=None):
    """Read the real recordings and cut them into model-length windows; return one flat list.

    Pools chunks across every (kind, cell), because the Nuisance_DLI is pooled across conditions
    (a by-kind split is only a diagnostic, never the constructed artifact). Returns
    ``(chunks, chunk_labels, summary)`` where ``chunks`` is a list of ``(n_frames, H, W)`` uint8
    arrays and ``chunk_labels`` is the aligned list of ``(kind_index, cell, chunk)`` identities
    -- the condition (index into ``kinds``) and the time-window position (cell, sliding-window
    index) of each chunk -- so the pool built from these chunks can be persisted self-describing.

    When ``topo`` is distributed the flat (kind, cell) work list is split across workers
    round-robin by rank (via :func:`shard_by_rank`), so each worker reads only its own cells'
    chunks. The pool is a mixture over chunks, so concatenating the per-rank partials at merge
    time is exact and order-independent -- and because each chunk carries its own label, the
    labels stay row-aligned through the round-robin shard/merge. With one worker (or ``topo is
    None``) every cell is read, in kind- then cell-order -- byte-for-byte the original behavior.
    """
    cells_by_kind = {}
    for kind in kinds:
        cells = discover_cells(R["experiment_dir"], kind, span)
        if max_cells > 0:
            cells = cells[:max_cells]
        cells_by_kind[kind] = cells
    flat_work = [(kind, cell) for kind in kinds for cell in cells_by_kind[kind]]
    if topo is not None and topo.is_distributed:
        flat_work = shard_by_rank(flat_work, topo)
    my_cells = {kind: [c for k, c in flat_work if k == kind] for kind in kinds}
    chunks, chunk_labels, per_kind = [], [], {}
    for kind_index, kind in enumerate(kinds):
        n_cell_chunks = 0
        for cell in my_cells[kind]:
            tif_path = R["paths"].experiment_video_path(kind, cell, span, R["data_bank_root"])
            if not tif_path.exists():
                continue
            cell_chunks = read_cell_chunks(tif_path, n_frames, step_frames)
            chunks.extend(cell_chunks)
            chunk_labels.extend((kind_index, int(cell), h) for h in range(len(cell_chunks)))
            n_cell_chunks += len(cell_chunks)
        per_kind[kind] = (len(my_cells[kind]), n_cell_chunks)
    return chunks, chunk_labels, per_kind


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
    chunks, chunk_labels, _ = _read_all_chunks(R, kinds, args.experiment_span_seconds, n_frames,
                                               step_frames, args.max_cells)
    if not chunks:
        raise FileNotFoundError(
            f"No experimental recordings found under {R['experiment_dir']} for kinds {kinds} "
            f"at span {args.experiment_span_seconds}s. Stage the real recordings first.")
    eval_cfg = PARAMETERS.inference.evaluation
    if pool_kind == "MapEstimate":
        pool, labels = ndli.build_map_estimate_pool(posterior, chunks, device, vista_device,
                                                    eval_cfg, pool_mode=pool_mode,
                                                    chunk_labels=chunk_labels)
    else:
        pool, labels = ndli.build_posterior_sample_pool(
            posterior, chunks, device, vista_device, n_per_chunk=n_per,
            theta_prex_batch_size=eval_cfg.theta_prex_batch_size, pool_mode=pool_mode,
            chunk_labels=chunk_labels)
    labels["kinds"] = kinds                         # the shared name array kind_index indexes into
    ndli.save_pool(cache, pool, prov, labels=labels)
    return pool, f"computed on GPU + cached ({cache.name})"


def _emit_pool_shard(args, R, topo, out_dir, n_frames, step_frames, n_per, eval_cfg):
    """One sharded worker's job (multi-GPU --emit-template): build the posterior-sample pool
    over THIS rank's (kind, cell) chunk subset and write it as a shard.

    Writes no cache and no spec; the single-process --merge step concatenates every rank's
    shard, caches the merged pool, and emits the template. A worker that draws no cells writes
    no shard. The pool is a mixture -- a concatenation of per-chunk posterior draws -- so a
    worker holding whole cells builds its slice independently and the merge just concatenates
    (exact and order-independent; percentiles / fits do not depend on draw order)."""
    posterior, device, vista_device = _load_posterior(R)
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    chunks, chunk_labels, _ = _read_all_chunks(R, kinds, args.experiment_span_seconds, n_frames,
                                               step_frames, args.max_cells, topo=topo)
    if chunks:
        partial, labels = ndli.build_posterior_sample_pool(
            posterior, chunks, device, vista_device, n_per_chunk=n_per,
            theta_prex_batch_size=eval_cfg.theta_prex_batch_size, pool_mode=args.pool_mode,
            chunk_labels=chunk_labels)
        shard_arrays = {"pool": partial, "kind_index": labels["kind_index"],
                        "cell": labels["cell"], "chunk": labels["chunk"]}
    else:
        partial = np.empty((0, len(R["imaging_keys"])), dtype=float)
        shard_arrays = {"pool": partial, "kind_index": np.empty(0, np.int64),
                        "cell": np.empty(0, np.int64), "chunk": np.empty(0, np.int64)}
    path = save_shard(out_dir, topo, shard_arrays, count=partial.shape[0])
    if path is None:
        print(f"[rank {topo.rank}/{topo.world_size}] no cells assigned -- no pool shard "
              f"written.", flush=True)
    else:
        print(f"[rank {topo.rank}/{topo.world_size}] pool shard saved: {path} "
              f"({partial.shape[0]} draws over {len(chunks)} chunk(s)). Run the --merge "
              f"step once all shards finish.", flush=True)


def _merge_pool_shards(R, args, out_dir, n_per):
    """Combine the per-rank pool shards (multi-GPU --emit-template --merge): concatenate every
    ``_shard_*_of_*.npz`` partial pool, cache the merged pool under its provenance (so a later
    --build reuses it with no GPU), remove the shards, and return ``(pool, source)`` for the
    template tail. Uses the same cache-path and provenance helpers as :func:`_get_pool`, so the
    cache is byte-for-byte reusable by --build."""
    cache = ndli.pool_cache_path(R["posit_dir"], R["paths"].project_alias,
                                 R["timing_label"], "PosteriorSample")
    prov = _pool_provenance(R, args, args.pool_mode, n_per)
    shard_paths = load_shards(out_dir)
    if not shard_paths:
        raise SystemExit(f"--merge: no pool shard files (_shard_*_of_*.npz) found in {out_dir}")
    try:
        assert_consistent_shard_set(shard_paths)
    except ValueError as exc:
        raise SystemExit(f"--merge: inconsistent pool shard set in {out_dir} -- {exc} "
                         f"Remove the stale _shard_*.npz and re-run the sharded --emit-template.")
    print(f"Merging {len(shard_paths)} pool shard file(s) from {out_dir}", flush=True)
    try:
        merged, n_used = merge_shard_arrays(
            shard_paths, concat_keys=["pool", "kind_index", "cell", "chunk"])
    except ValueError:
        raise SystemExit(f"--merge: every pool shard in {out_dir} was empty (no draws)")
    pool = merged["pool"]
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    labels = {"kind_index": merged["kind_index"], "cell": merged["cell"],
              "chunk": merged["chunk"], "kinds": kinds}    # labels travel with their rows through the merge
    ndli.save_pool(cache, pool, prov, labels=labels)
    for shard_path in shard_paths:
        shard_path.unlink()
    try:
        out_dir.rmdir()   # drop the now-empty shard subdir
    except OSError:
        pass
    print(f"Merged {pool.shape[0]} draws from {n_used} shard(s); cached to {cache.name} and "
          f"removed {len(shard_paths)} shard file(s).", flush=True)
    return pool, f"merged {n_used} GPU shard(s) + cached ({cache.name})"


def _emit_template_dry_run(args, R, topo, spec, out_dir):
    """Print the sharded --emit-template plan (single-process, per-rank shard, and --merge)
    without loading the estimator, reading recordings, or computing anything."""
    cache = ndli.pool_cache_path(R["posit_dir"], R["paths"].project_alias,
                                 R["timing_label"], "PosteriorSample")
    if args.merge:
        n_present = len(load_shards(out_dir))
        print("[DRY RUN] --emit-template --merge would combine the pool shards, cache the "
              "merged pool, and emit the spec:")
        print(f"    pool shards : {out_dir}/_shard_*_of_*.npz  ({n_present} present)")
        print(f"    pool cache  : {cache}")
        print(f"    spec        : {spec}")
        return
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    cells_by_kind = {}
    for kind in kinds:
        cells = discover_cells(R["experiment_dir"], kind, args.experiment_span_seconds)
        if args.max_cells > 0:
            cells = cells[:args.max_cells]
        cells_by_kind[kind] = cells
    flat_work = [(kind, cell) for kind in kinds for cell in cells_by_kind[kind]]
    print("[DRY RUN] --emit-template would read the estimator + recordings and write:")
    print(f"    estimator   : {R['estimator_path']}  [{'OK' if R['estimator_path'].exists() else 'MISSING'}]")
    print(f"    recordings  : {R['experiment_dir']}/Experiment_<KIND>_Cell_<n>_{args.experiment_span_seconds}S_RAW.tif")
    print(f"    work items  : {len(flat_work)} (kind, cell) recording(s) discovered")
    if topo.is_distributed:
        my_work = shard_by_rank(flat_work, topo)
        print(f"    sharding    : world_size={topo.world_size}; rank {topo.rank} would take "
              f"{len(my_work)} of {len(flat_work)} item(s) and write a pool shard under {out_dir}")
        print(f"    then        : --emit-template --merge (single process, no GPU) caches the pool + emits the spec")
    else:
        print(f"    plan        : single process builds the full pool, caches it, and emits the spec")
        print(f"    spec        : {spec}")


def _emit_template(args, R):
    spec = ndli.spec_path(R["posit_dir"], R["paths"].project_alias, R["timing_label"])
    n_frames, step_frames = _chunk_geometry(R["timing"], args.experiment_span_seconds,
                                             args.chunk_step_seconds)
    topo = resolve_topology()
    out_dir = R["posit_dir"] / "_Nuisance_DLI_pool_shards"   # dedicated shard dir, isolated from the cache/spec/estimator; --merge removes it

    if args.dry_run:
        _emit_template_dry_run(args, R, topo, spec, out_dir)
        return

    eval_cfg = PARAMETERS.inference.evaluation
    n_per = args.n_per_chunk or eval_cfg.posterior_samples

    # ---- Multi-GPU sharded worker: build a partial pool -> shard, then stop -----
    # One worker per GPU (torchrun) builds the posterior-sample pool over its own
    # (kind, cell) subset and writes it as a shard; the single-process --merge step
    # concatenates the shards, caches the pool, and emits the spec. It writes no cache
    # and no spec here. Single-GPU / no-torchrun runs skip this and build the pool directly.
    if topo.is_distributed and not args.merge:
        _emit_pool_shard(args, R, topo, out_dir, n_frames, step_frames, n_per, eval_cfg)
        return

    # ---- Obtain the full posterior-sample pool ---------------------------------
    # The pool (drawn under --pool-mode) feeds the percentile suggestions; it is cached, so a
    # later raw/gaussian/box --build under the same pool_mode reuses it without GPU. --merge
    # concatenates the per-rank shards into the same pool the single-process path computes.
    if args.merge:
        pool, source = _merge_pool_shards(R, args, out_dir, n_per)
    else:
        pool, source = _get_pool(R, args, "PosteriorSample", args.pool_mode,
                                 n_frames, step_frames, n_per)
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
          "for 'box_user' set the [imaging.<KEY>] ranges; for 'sgm_percentiles' set percentiles / "
          "condition / selection_source — then run --build.")


def _build_sgm_percentiles(args, R, block, spec_dict, art):
    """CPU-only sgm_percentiles build: select whole vectors at SIGNED distance-to-SGM percentiles from
    ALREADY-COMPUTED data (the Detector Experiment MAPs, or the labeled posterior pool via the per-
    window SGM), and mint them as the Nuisance_DLI. p50 = the SGM (a single frozen vector); several
    percentiles = a small, correlation-preserving marginalization pool. No GPU, no re-inference."""
    percentiles = block.get("percentiles", list(ndli.SGM_DEFAULT_PERCENTILES))
    condition = block.get("condition", "pooled")
    source = block.get("selection_source", "experiment")
    exp_stem = R["paths"].experiment_recovery_pattern.format(
        project_alias=R["paths"].project_alias, timing_label=R["timing_label"])
    exp_map = R["posit_dir"] / exp_stem / f"{exp_stem}.npz"
    pool_cache = ndli.pool_cache_path(R["posit_dir"], R["paths"].project_alias,
                                      R["timing_label"], "PosteriorSample")

    if args.dry_run:
        src = exp_map if source == "experiment" else pool_cache
        print(f"[DRY RUN] --build sgm_percentiles (CPU, reuses existing data): "
              f"percentiles={percentiles}, condition={condition}, selection_source={source}.")
        print(f"    source     : {src}  [{'OK' if src.exists() else 'MISSING'}]")
        print(f"    would write:\n    {art}  (+ report + 1-D marginals plot)")
        return

    vecs, src_label = ndli.load_map_vectors(
        exp_map, pool_cache, source=source, condition=condition,
        prior_low=R["plo"], prior_high=R["phi"])
    b_idx = R["imaging_keys"].index("mu_pc") if "mu_pc" in R["imaging_keys"] else 2
    members, prov = ndli.select_signed_percentile_vectors(
        vecs, percentiles, R["plo"], R["phi"], brightness_index=b_idx)
    print(f"sgm_percentiles: {src_label}; {members.shape[0]} member(s) at percentiles "
          f"{[round(p, 1) for p in prov['percentile_of_member']]} (SGM = source row {prov['sgm_row']}).")
    for note in prov["notes"]:
        print(f"    note: {note}")
    nu = ndli.build_nuisance_dli(spec_dict, R["imaging_keys"], R["plo"], R["phi"], pool=members)
    nu.flush(str(art))
    report_dir = _write_nuisance_report(nu, R, art)
    print(f"Built Nuisance_DLI (sgm_percentiles, {members.shape[0]} member(s)) and saved to:\n"
          f"    {art}\nAnalysis (report + 1-D marginals plot):\n    {report_dir}")


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
    pool_kind = ndli.POOL_KINDS[choice]                 # None for box_user / sgm_percentiles
    n_per = args.n_per_chunk or PARAMETERS.inference.evaluation.posterior_samples

    if choice == "sgm_percentiles":
        _build_sgm_percentiles(args, R, block, spec_dict, art)
        return

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


def _migrate_pool_labels(args, R):
    """CPU-only maintenance: add per-row kind_index/cell/chunk labels to an EXISTING (legacy)
    PosteriorSample pool by borrowing the aligned labels from the Detector Experiment MAP output,
    after verifying the two share a window ordering. Backs up the original first; a pool that is
    already labeled is left unchanged. Preserves the pool's provenance verbatim, so its cache
    freshness (and therefore the no-GPU reuse by --build) is unaffected."""
    from scipy.spatial.distance import pdist, squareform

    alias = R["paths"].project_alias
    pool_path = ndli.pool_cache_path(R["posit_dir"], alias, R["timing_label"], "PosteriorSample")
    stem = R["paths"].experiment_recovery_pattern.format(project_alias=alias,
                                                         timing_label=R["timing_label"])
    map_path = R["posit_dir"] / stem / f"{stem}.npz"

    if args.dry_run:
        labeled = pool_path.exists() and ndli.load_pool_labels(pool_path) is not None
        print("[DRY RUN] --migrate-pool-labels would add per-row kind_index/cell/chunk to the pool:")
        print(f"    pool        : {pool_path}  [{'OK' if pool_path.exists() else 'MISSING'}"
              f"{'; ALREADY LABELED (no-op)' if labeled else ''}]")
        print(f"    MAP labels  : {map_path}  [{'OK' if map_path.exists() else 'MISSING'}]")
        print(f"    checks      : rows == n_windows*n_per_chunk; index alignment beats a random shuffle (>= 4x)")
        print(f"    writes      : back up -> {pool_path.name}.bak, then re-save the labeled pool")
        return

    if not pool_path.exists():
        raise FileNotFoundError(f"PosteriorSample pool not found:\n    {pool_path}")
    if ndli.load_pool_labels(pool_path) is not None:
        print(f"Pool already carries labels (pool_format_version present); nothing to migrate:\n"
              f"    {pool_path}")
        return
    if not map_path.exists():
        raise FileNotFoundError(
            f"Detector Experiment MAP output not found (the label source):\n    {map_path}\n"
            f"Run the Detector Experiment stage first, or regenerate the pool with the labeled build.")

    with np.load(str(pool_path), allow_pickle=False) as d:
        pool = np.asarray(d["pool"], dtype=float)
        prov = json.loads(str(d["provenance"]))          # preserved verbatim (freshness-critical)
    with np.load(str(map_path), allow_pickle=False) as d:
        inferred = np.asarray(d["inferred_log10"], dtype=float)
        kind_index_w = np.asarray(d["kind_index"], dtype=np.int64)
        cell_w = np.asarray(d["cell"], dtype=np.int64)
        chunk_w = np.asarray(d["chunk"], dtype=np.int64)
        kinds = np.asarray(d["kinds"])

    n_per = int(prov["n_per_chunk"]); n_win = inferred.shape[0]; dim = pool.shape[1]
    if pool.shape[0] != n_win * n_per:
        raise SystemExit(
            f"--migrate-pool-labels: pool rows {pool.shape[0]} != n_windows*n_per_chunk "
            f"({n_win}*{n_per}={n_win * n_per}); the pool and the MAP output are not from the same "
            f"run. Regenerate the pool with the labeled build instead.")

    # Verify the pool is index-aligned with the MAP output before borrowing labels. Each window's
    # medoid (the per-window SGM of its draws) is a NOISY stand-in for that window's optimized MAP --
    # they diverge for broad posteriors -- so a tight per-window match is the wrong test. Instead we
    # require the index alignment to strongly beat a random shuffle: a genuine window-for-window
    # correspondence does so by a wide margin (its distances stay far below the shuffle's), while a
    # wrong ordering cannot. In prior-range-normalized absolute space.
    lo = np.array(det.theta_lower_bound()); hi = np.array(det.theta_upper_bound())
    range_abs = 10.0 ** hi - 10.0 ** lo
    pool_win = pool.reshape(n_win, n_per, dim)
    med = np.empty((n_win, dim))
    for w in range(n_win):
        m = (10.0 ** pool_win[w]) / range_abs
        med[w] = pool_win[w][int(np.argmin(squareform(pdist(m)).sum(0)))]
    med_n = (10.0 ** med) / range_abs
    inf_n = (10.0 ** inferred) / range_abs
    d_index = np.linalg.norm(med_n - inf_n, axis=1)
    shuffle_rng = np.random.default_rng(0)
    shuffle = np.concatenate([np.linalg.norm(med_n - inf_n[shuffle_rng.permutation(n_win)], axis=1)
                              for _ in range(50)])
    ratio = float(np.median(d_index) / np.median(shuffle))
    align_max = 0.25                                        # require a >=4x separation from chance
    if ratio > align_max:
        raise SystemExit(
            f"--migrate-pool-labels: alignment check FAILED -- index-aligned median distance "
            f"{np.median(d_index):.4f} is not far below the random-shuffle median "
            f"{np.median(shuffle):.4f} (ratio {ratio:.3f} > {align_max}); the pool and the MAP "
            f"output do not share a window ordering. Regenerate the pool with the labeled build.")

    labels = {"kind_index": np.repeat(kind_index_w, n_per), "cell": np.repeat(cell_w, n_per),
              "chunk": np.repeat(chunk_w, n_per), "kinds": kinds}
    backup = pool_path.with_suffix(pool_path.suffix + ".bak")
    shutil.copy2(str(pool_path), str(backup))
    ndli.save_pool(pool_path, pool, prov, labels=labels)   # provenance unchanged -> cache still fresh
    kinds_list = list(kinds)
    n_alp = int((labels["kind_index"] == kinds_list.index("ALP")).sum()) if "ALP" in kinds_list else 0
    print(f"Migrated pool labels. Alignment: index-median {np.median(d_index):.4f} vs shuffle-median "
          f"{np.median(shuffle):.4f} (ratio {ratio:.3f}). Backup: {backup.name}")
    print(f"    labeled pool: {pool_path}")
    print(f"    rows: {pool.shape[0]}  windows: {n_win}  n_per_chunk: {n_per}  ALP rows: {n_alp}")


def main(args):
    R = _resolve(args.total_time_seconds)
    if args.migrate_pool_labels:
        _migrate_pool_labels(args, R)
        return
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
    p.add_argument("--merge", action="store_true",
                   help="emit-only combine mode: read the per-rank pool shards written by a "
                        "multi-GPU --emit-template run (in the Posit dir), concatenate them into "
                        "the posterior-sample pool, cache it, then emit the spec, and exit. Does no "
                        "estimation and needs no GPU; the launcher runs it once after the sharded "
                        "workers finish. Single-GPU emit runs never use it (they build directly).")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve paths and report what would be read/written; load nothing, compute nothing.")
    p.add_argument("--migrate-pool-labels", action="store_true",
                   help="CPU-only maintenance: add per-row kind_index/cell/chunk labels to an existing "
                        "(legacy) PosteriorSample pool by borrowing them from the Detector Experiment "
                        "MAP output, after verifying the two share a window ordering. Backs up the "
                        "original; an already-labeled pool is left unchanged. No GPU or estimator; "
                        "independent of --emit-template/--build.")
    return p.parse_args(argv)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))

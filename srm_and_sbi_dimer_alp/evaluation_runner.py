"""Shared Evaluation-stage engine for both DIMER workflows (biology + detector).

``run_evaluation(cfg, args)`` holds the entire MAP-recovery orchestration -- banner,
dry-run probe, estimator load, the per-video seed-then-optimize MAP loop, the live
progress log, multi-GPU round-robin task sharding, the ``--merge`` combine step, and
the recovery report + figures. The two entry-point scripts shrink to: build the
``WorkflowConfig``, parse args, call ``run_evaluation``.

The numeric engine already lives in ``evaluation`` (``map_estimate``,
``recovery_table``, ``posterior_summary``, ``posterior_coverage_table``), shared by
both workflows. The only per-workflow differences are which parameterization supplies
the learnable table (for the recovery/coverage tables + per-parameter figures), the
``parameter_keys`` schema guard, the device ``build_prior``, and the alias-qualified
``paths`` -- resolved from the config in ``_evaluation_spec(cfg)``. The estimator path
is resolved through the shared ``paths.estimator_path`` method for both workflows.
"""

from __future__ import annotations

import argparse
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch._dynamo

from srm_and_sbi_dimer_alp import artifacts
from srm_and_sbi_dimer_alp.diagnostics import DiagnosticReporter
from srm_and_sbi_dimer_alp.evaluation import (
    map_estimate,
    posterior_coverage_table,
    posterior_summary,
    recovery_table,
    _theta_repr,
)
from srm_and_sbi_dimer_alp.inference_support import resolve_topology
from srm_and_sbi_dimer_alp.io import load_data
from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming
from srm_and_sbi_dimer_alp.utils import console_log_context  # noqa: F401  (entry points import via this module's siblings)
from srm_and_sbi_dimer_alp.visualization_inference import figure_recovery_combined
from srm_and_sbi_dimer_alp.workflow import WorkflowConfig


@dataclass(frozen=True)
class _EvaluationSpec:
    """Per-workflow Evaluation specializations resolved from a ``WorkflowConfig``."""
    draw_spec: list          # learnable table for the recovery/coverage tables + figures
    parameter_keys: list     # load_estimator schema guard
    build_prior: Callable    # device prior rebuild for bounded rejection sampling


def _evaluation_spec(cfg: WorkflowConfig) -> _EvaluationSpec:
    """Resolve the Evaluation stage's per-workflow specializations from the workflow config."""
    m = cfg.param_module
    if cfg.tag == "detector":
        return _EvaluationSpec(
            draw_spec=m.DETECTOR_PARAMETERIZATION,
            parameter_keys=[e["KEY"] for e in m.DETECTOR_PARAMETERIZATION],
            build_prior=m.build_prior,
        )
    return _EvaluationSpec(
        draw_spec=m.PARAMETERIZATION,
        parameter_keys=m.PARAMETER_KEYS,
        build_prior=m.build_prior,
    )


def _shard_array_path(recovery_dir: Path, rank: int, world_size: int) -> Path:
    """Path of one worker's partial recovery arrays in a multi-GPU sharded run."""
    return recovery_dir / f"_shard_{rank:02d}_of_{world_size:02d}.npz"


def write_recovery_outputs(reporter, args, eval_cfg, draw_spec, recovery_array_path: Path,
                           scores, inferred_log10, true_log10, post_quantiles,
                           run_start) -> None:
    """Save the recovery arrays and write the report + figures.

    Shared by the single-process recovery path and the ``--merge`` combine step,
    so both emit an identical report. ``post_quantiles`` may be a list (built by
    the recovery loop) or an array (concatenated by a merge); both are handled.
    ``draw_spec`` is the workflow's learnable table (the recovery/coverage tables +
    per-parameter figures iterate it).
    """
    do_map = args.summary in ("map", "both")
    do_posterior = args.summary in ("posterior", "both")
    bin_mode = args.bin_mode
    posterior_samples = args.posterior_samples or eval_cfg.posterior_samples
    n_bins = args.n_bins or eval_cfg.quantile_bins
    min_count = args.min_count or eval_cfg.quantile_min_count

    scores = np.asarray(scores)
    inferred_log10 = np.asarray(inferred_log10)
    true_log10 = np.asarray(true_log10)
    n_samples = inferred_log10.shape[0]
    post_arr = np.asarray(post_quantiles)
    post_q = post_arr if (do_posterior and post_arr.size > 0) else None

    # ---- Save the recovery arrays ----------------------------------------
    save_arrays = dict(true_log10=true_log10, inferred_log10=inferred_log10, scores=scores)
    if post_q is not None:
        save_arrays["posterior_quantiles"] = post_q   # [Q05,Q25,Q50,Q75,Q95]
    np.savez_compressed(str(recovery_array_path), **save_arrays)
    print(f"\nRecovery arrays saved to {recovery_array_path}")

    # ---- Recovery report -------------------------------------------------
    guide = eval_cfg.error_guide                   # 0.3 ~= log10(2): within a factor of 2
    guide_tight = eval_cfg.error_guide_tight        # 0.15 ~= log10(sqrt(2)): within a factor of ~1.41
    reporter.check("eval_set_nonempty", n_samples > 0,
                   f"{n_samples} EVAL samples recovered",
                   note="the held-out EVAL namespace yielded at least one "
                        "MAP-recovery sample.")
    reporter.check_no_nan_inf("true_log10", true_log10)
    reporter.stat("eval_tasks", args.eval_tasks)
    reporter.stat("eval_samples", n_samples,
                  note="number of held-out videos whose parameters were recovered.")
    reporter.stat("mean_log_prob", float(np.mean(scores)),
                  note="mean MAP log-density at the optimized mode -- the objective "
                       "the seed-then-optimize step maximizes (larger = sharper "
                       "peak). Computed in the estimator's z-scored space, so its "
                       "absolute scale is a relative optimization diagnostic, not a "
                       "calibration/quality metric; the per-parameter recovery error "
                       "table below is the quality measure.")
    if n_samples < eval_cfg.quantile_min_count:
        reporter.stat(
            "quantile_bands", "sparse",
            note=f"fewer than {eval_cfg.quantile_min_count} samples per bin: the "
                 "report shows the scatter and the error table; conditional "
                 "quantile bands populate only with a larger EVAL set.")

    headers, rows = recovery_table(draw_spec, true_log10, inferred_log10,
                                   guide, guide_tight)
    reporter.table(
        "MAP recovery (per parameter, log10 units)", headers, rows,
        note=f"error = inferred - true in log10 units; 'within +/-{guide:g}' "
             f"(factor 2) and 'within +/-{guide_tight:g}' (factor sqrt(2)) are the "
             "fractions of EVAL videos recovered inside each nested tolerance band.")

    # View B: posterior calibration (coverage of truth by credible intervals).
    if post_q is not None:
        reporter.stat("posterior_samples", posterior_samples,
                      note="samples per video used to summarize the posterior (View B).")
        cov_headers, cov_rows = posterior_coverage_table(draw_spec, true_log10, post_q)
        reporter.table(
            "Posterior calibration (per parameter)", cov_headers, cov_rows,
            note="View B: fraction of truths inside the per-video posterior credible "
                 "intervals; a calibrated posterior covers ~50% (IQR) and ~90%.")

    if reporter.dump:
        for i, para in enumerate(draw_spec):
            key = para["KEY"]
            label = para.get("LABEL") or key
            prior_range = para["PRIOR_RANGE"]
            reporter.save_figure(
                f"recovery_{key}",
                figure_recovery_combined(
                    true_log10[:, i], inferred_log10[:, i],
                    (post_q[:, i, :] if post_q is not None else None),
                    prior_range, label, n_bins=n_bins, min_count=min_count,
                    error_guide=guide, error_guide_tight=guide_tight,
                    error_ylim_floor=eval_cfg.error_ylim_floor,
                    error_ylim_quantile=eval_cfg.error_ylim_quantile, bin_mode=bin_mode,
                    show_map=do_map, show_posterior=(post_q is not None)),
                caption=f"{key} ({label}). View A (MAP): panels 1-2 show inferred-vs-true "
                        f"and residual error of the MAP point estimate, with "
                        f"{bin_mode}-binned conditional-quantile bands (drawn where a "
                        f"bin has >= {min_count} points). View B (posterior, panel 3): "
                        f"true vs. posterior median with IQR error bars (per-video "
                        f"credible width). A panel stamped 'not computed' marks a view "
                        f"the --summary option omitted.",
            )

    reporter.summary()
    reporter.write_report()

    print(f"\nTotal elapsed: {time.time() - run_start:.1f}s")


def _save_shard(reporter, topo, recovery_dir: Path,
                scores, inferred_log10, true_log10, post_quantiles, run_start) -> None:
    """Write this worker's partial recovery arrays (multi-GPU sharded run).

    The report is produced later by the ``--merge`` step, once every shard
    exists; this function does no report generation.
    """
    recovery_dir.mkdir(parents=True, exist_ok=True)
    if len(scores) == 0:
        # This worker drew no tasks (launched workers > eval_tasks): write no shard
        # at all, so --merge simply sees fewer files.
        print(f"\n[rank {topo.rank}/{topo.world_size}] no tasks assigned -- "
              f"no shard written.", flush=True)
        return
    arrays = dict(
        scores=np.asarray(scores),
        inferred_log10=np.asarray(inferred_log10),
        true_log10=np.asarray(true_log10),
    )
    if post_quantiles:
        arrays["posterior_quantiles"] = np.asarray(post_quantiles)
    path = _shard_array_path(recovery_dir, topo.rank, topo.world_size)
    np.savez_compressed(str(path), **arrays)
    print(f"\n[rank {topo.rank}/{topo.world_size}] shard saved: {path} "
          f"({arrays['scores'].shape[0]} videos) in {time.time() - run_start:.1f}s. "
          f"Run the --merge step once all shards finish.", flush=True)


def _merge_shards(reporter, args, eval_cfg, draw_spec, recovery_dir: Path,
                  recovery_array_path: Path, run_start) -> None:
    """Combine all per-shard recovery arrays into the final report (no recovery).

    Reads every ``_shard_*_of_*.npz`` in the recovery directory, concatenates the
    arrays (order-independent -- the recovery metrics are aggregates over videos),
    writes the final report + figures + combined ``.npz`` via
    :func:`write_recovery_outputs`, then removes the shard files.
    """
    shard_paths = sorted(recovery_dir.glob("_shard_*_of_*.npz"))
    if not shard_paths:
        raise SystemExit(
            f"--merge: no shard files (_shard_*_of_*.npz) found in {recovery_dir}")
    print(f"Merging {len(shard_paths)} shard file(s) from {recovery_dir}", flush=True)
    scores, inferred, true, quant = [], [], [], []
    have_quant = True
    n_used = 0
    for shard_path in shard_paths:
        with np.load(str(shard_path)) as data:
            if data["scores"].shape[0] == 0:
                continue   # defensive: a zero-video shard contributes nothing
            n_used += 1
            scores.append(data["scores"])
            inferred.append(data["inferred_log10"])
            true.append(data["true_log10"])
            if "posterior_quantiles" in data:
                quant.append(data["posterior_quantiles"])
            else:
                have_quant = False   # a populated shard genuinely computed no View B
    if not scores:
        raise SystemExit(
            f"--merge: every shard in {recovery_dir} was empty (no recovered videos)")
    scores = np.concatenate(scores, axis=0)
    inferred = np.concatenate(inferred, axis=0)
    true = np.concatenate(true, axis=0)
    post_quantiles = np.concatenate(quant, axis=0) if (have_quant and quant) else np.asarray([])
    print(f"Merged {scores.shape[0]} videos from {n_used} shard(s).", flush=True)
    write_recovery_outputs(reporter, args, eval_cfg, draw_spec, recovery_array_path,
                           scores, inferred, true, post_quantiles, run_start)
    for shard_path in shard_paths:
        shard_path.unlink()
    print(f"Removed {len(shard_paths)} shard file(s).", flush=True)


def run_evaluation(cfg: WorkflowConfig, args: argparse.Namespace) -> None:
    """Run the full MAP-recovery evaluation for the given workflow + CLI args."""
    spec = _evaluation_spec(cfg)

    timing = RunTiming(
        total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing,
    )
    data_bank_root = PARAMETERS.machine.data_bank_root
    compress = True  # EVAL video/theta sets are read from .zarr, as in inference
    paths = cfg.paths
    eval_cfg = PARAMETERS.inference.evaluation

    # ---- Global RNG / precision settings ---------------------------------
    if args.seed is not None:   # None -> non-deterministic (consistent with generation)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.suppress_errors = True

    timing_label = timing.label
    estimator_path = paths.estimator_path(data_bank_root, timing_label)
    recovery_dir = paths.map_recovery_dir(data_bank_root, timing_label)
    recovery_array_path = paths.map_recovery_array_path(data_bank_root, timing_label)

    # Resolve effective hyperparameters (None -> config default).
    lr = (args.learning_rate if args.learning_rate is not None
          else eval_cfg.learning_rate_minimum * eval_cfg.learning_rate_maximum_factor)
    tolerance = eval_cfg.learning_rate_minimum * eval_cfg.tolerance_factor
    theta_prex_size = args.theta_prex_size or eval_cfg.theta_prex_size
    elite_prex_size = args.elite_prex_size or eval_cfg.elite_prex_size
    numb_steps = args.numb_steps or eval_cfg.numb_steps
    show_progress_steps = args.show_progress_steps or eval_cfg.show_progress_steps
    pool_mode = args.pool_mode or eval_cfg.pool_mode

    # Verbosity tiers (the show / verbose flags).
    show = args.verbose or args.debug or args.debug_dump
    verbose_deep = args.debug or args.debug_dump
    debug_log = args.debug or args.debug_dump   # write per-step/theta detail to progress.log

    # Summary views (View A = MAP-point figures; View B = posterior-credible figures).
    do_map = args.summary in ("map", "both")
    do_posterior = args.summary in ("posterior", "both")
    bin_mode = args.bin_mode
    posterior_samples = args.posterior_samples or eval_cfg.posterior_samples

    # ---- Pre-run banner --------------------------------------------------
    machine = PARAMETERS.machine
    div = "=" * 72
    print(div)
    print(f" {paths.project_alias} — Evaluation (MAP recovery)")
    print(f" Started at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(div)
    print("\nMachine profile:")
    print(f"  name              : {machine.name}")
    print(f"  compute_backend   : {machine.compute_backend}")
    print("\nRun configuration (CLI args):")
    print(f"  --total-time-seconds : {args.total_time_seconds}")
    print(f"  --eval-tasks         : {args.eval_tasks}        (EVAL-namespace tasks; held out)")
    print(f"  --max-sims           : {args.max_sims}        (per task; 0 = all)")
    print(f"  --seed               : {args.seed}")
    print(f"  verbosity            : "
          f"{'debug-dump' if args.debug_dump else ('debug' if args.debug else ('verbose' if args.verbose else 'normal'))}")
    print("\nMAP estimate hyperparameters (effective):")
    print(f"  pool_mode            : {pool_mode}   "
          f"({'rejection within prior' if pool_mode == 'bounded' else 'flow direct, no rejection'})")
    print(f"  theta_prex_size      : {theta_prex_size}")
    print(f"  elite_prex_size      : {elite_prex_size}")
    print(f"  numb_steps           : {numb_steps}")
    print(f"  learning_rate        : {lr:.3e}   tolerance: {tolerance:.3e}")
    print(f"  --summary            : {args.summary}   (View A map={do_map}, View B posterior={do_posterior})")
    print(f"  --bin-mode           : {bin_mode}")
    if do_posterior:
        print(f"  --posterior-samples  : {posterior_samples}")
    progress_path = recovery_dir / "progress.log"
    print("\nOutput destinations:")
    print(f"  reads estimator : {estimator_path}")
    print(f"  reads EVAL      : <data_bank>/{paths.video_subdir}/"
          f"{paths.project_alias}_{timing_label}_Video_Set_TASK_{{0..{args.eval_tasks - 1}}}_EVAL.zarr")
    print(f"  writes report   : {recovery_dir}")
    print(f"  live progress   : {progress_path}   (tail -f to monitor)")
    print(f"\n{div}\n")

    # ---- Dry run: validate inputs and exit before the reporter, GPU, posterior,
    # or topology -- so it creates no output directory and needs no torchrun.
    if args.dry_run:
        eval_video_path = paths.video_set_path(0, data_bank_root, timing_label, compress, "EVAL")
        eval_theta_path = paths.theta_set_path(0, data_bank_root, timing_label, compress, "EVAL")
        inputs = [
            ("estimator artifact", estimator_path),
            ("EVAL video set (task 0)", eval_video_path),
            ("EVAL theta set (task 0)", eval_theta_path),
        ]
        missing = 0
        for role, path in inputs:
            ok = Path(path).exists()
            if not ok:
                missing += 1
            print(f"  reads {role}: {path}  [{'OK' if ok else 'MISSING'}]")
        if missing:
            print(f"\n[DRY RUN] configuration validated; {missing} input(s) MISSING.")
        else:
            print(f"\n[DRY RUN] configuration validated; all inputs present.")
        print("[DRY RUN] no MAP-recovery evaluation performed.")
        return

    run_start = time.time()

    # ---- Diagnostics reporter (the recovery report is the deliverable) ----
    reporter = DiagnosticReporter(
        stage="Evaluation", enabled=True, dump=True, dump_dir=recovery_dir,
        run_label=f"{paths.project_alias}_{timing_label}",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    # --merge: combine the per-shard arrays from a multi-GPU sharded run into the
    # final report, then exit (no recovery work, no GPU, no posterior needed).
    if args.merge:
        _merge_shards(reporter, args, eval_cfg, spec.draw_spec, recovery_dir,
                      recovery_array_path, run_start)
        return

    reporter.check_file("estimator artifact", estimator_path)

    topo = resolve_topology()
    device = topo.device
    vista_device = torch.device("cpu")
    posterior = artifacts.load_estimator(estimator_path, device=str(device),
                                          expected_parameter_keys=spec.parameter_keys)
    posterior.posterior_estimator.to(device)
    if device.type == "cuda":
        # Rebuild the prior on THIS worker's device for bounded rejection sampling.
        posterior.prior = spec.build_prior(device=str(device))

    # ---- This worker's task shard ----------------------------------------
    my_tasks = [t for t in range(args.eval_tasks) if t % topo.world_size == topo.rank]

    # ---- Probe the EVAL namespace for this shard's recovery workload -----
    per_task_sims = {}
    for task in my_tasks:
        theta_set = load_data(
            paths.theta_set_path(task, data_bank_root, timing_label, compress, "EVAL"))
        n_sims = theta_set.shape[0]
        per_task_sims[task] = min(n_sims, args.max_sims) if args.max_sims > 0 else n_sims
    total_sims = int(sum(per_task_sims.values()))

    # ---- Live progress log (so a long MAP run can be monitored) ----------
    recovery_dir.mkdir(parents=True, exist_ok=True)
    if topo.is_main:
        shutil.rmtree(recovery_dir / "figures", ignore_errors=True)

    def log_progress(progress_fh, message: str) -> None:
        """Append a timestamped line to the progress log and flush immediately."""
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        if progress_fh is not None:
            progress_fh.write(line + "\n")
            progress_fh.flush()

    def log_file_only(progress_fh, message: str) -> None:
        """Write a line to the progress log only (console already has it via --debug)."""
        if progress_fh is not None:
            progress_fh.write(f"           {message}\n")
            progress_fh.flush()

    if topo.is_distributed and not topo.is_main:
        progress_fh = None
    else:
        try:
            progress_fh = open(progress_path, "w", encoding="utf-8")
        except OSError as exc:
            print(f"WARNING: cannot open progress log {progress_path} ({exc}); "
                  f"continuing without it.", flush=True)
            progress_fh = None

    step_log = (lambda m: log_file_only(progress_fh, m)) if debug_log else None

    # ---- MAP estimate over the EVAL namespace ----------------------------
    scores, inferred_log10, true_log10, post_quantiles = [], [], [], []
    shard_note = (f" [shard rank {topo.rank}/{topo.world_size}: {len(my_tasks)} of "
                  f"{args.eval_tasks} tasks]" if topo.is_distributed else "")
    log_progress(progress_fh,
                 f"START MAP recovery: {len(my_tasks)} EVAL task(s){shard_note}, "
                 f"{total_sims} videos total (pool={theta_prex_size}, "
                 f"elites={elite_prex_size}, steps={numb_steps}; "
                 f"summary={args.summary}).")
    loop_start = time.time()
    done = 0
    try:
        for task in my_tasks:
            video_set = load_data(
                paths.video_set_path(task, data_bank_root, timing_label, compress, "EVAL"))
            theta_set = load_data(
                paths.theta_set_path(task, data_bank_root, timing_label, compress, "EVAL"))
            task_start = time.time()
            for sim in range(per_task_sims[task]):
                video_chunk = np.asarray(video_set[sim])
                if show:
                    print(f"\n######## MAP estimate: task {task} sim {sim} ########",
                          flush=True)
                if debug_log:
                    log_file_only(progress_fh, f"-- task {task} sim {sim} --")
                score, theta_log = map_estimate(
                    posterior, video_chunk, device, vista_device,
                    theta_prex_size, eval_cfg.theta_prex_batch_size,
                    eval_cfg.score_prex_batch_size, elite_prex_size,
                    numb_steps, eval_cfg.optimizer_patience,
                    eval_cfg.scheduler_patience, show_progress_steps,
                    eval_cfg.learning_rate_minimum, eval_cfg.learning_rate_factor,
                    lr, tolerance, pool_mode=pool_mode, show=show, verbose=verbose_deep,
                    log_fn=step_log,
                )
                true_log = np.log10(np.asarray(theta_set[sim], dtype=float))
                scores.append(score)
                inferred_log10.append(theta_log)
                true_log10.append(true_log)
                if do_posterior:
                    # View B: posterior credible summary for this video (median + IQR).
                    post_quantiles.append(posterior_summary(
                        posterior, video_chunk, device, vista_device,
                        posterior_samples, eval_cfg.theta_prex_batch_size,
                        pool_mode=pool_mode))
                if show:
                    print(f"          original theta [LOG] {_theta_repr(true_log)}",
                          flush=True)
                    print(f"          original theta [ABS] "
                          f"{_theta_repr(np.power(10.0, true_log))}", flush=True)
                done += 1
                elapsed = time.time() - loop_start
                avg = elapsed / done
                eta = avg * (total_sims - done)
                log_progress(
                    progress_fh,
                    f"task {task} sim {sim} | {done}/{total_sims} | "
                    f"log_prob={score:+.3f} | elapsed={elapsed:.1f}s | "
                    f"avg={avg:.1f}s/video | ETA={eta:.0f}s")
                if debug_log:
                    log_file_only(progress_fh,
                                  f"inferred [LOG] {_theta_repr(theta_log)}")
                    log_file_only(progress_fh,
                                  f"original [LOG] {_theta_repr(true_log)}")
            log_progress(progress_fh,
                         f"task {task} complete ({per_task_sims[task]} videos in "
                         f"{time.time() - task_start:.1f}s).")
        log_progress(progress_fh,
                     f"DONE: {done}/{total_sims} videos in "
                     f"{time.time() - loop_start:.1f}s.")
    finally:
        if progress_fh is not None:
            progress_fh.close()

    # ---- Write outputs ---------------------------------------------------
    # One worker (world_size == 1) writes the final report directly. Multiple
    # workers each write their partial arrays; the separate ``--merge`` step,
    # run by the launcher once all shards finish, combines them into the report.
    if topo.is_distributed:
        _save_shard(reporter, topo, recovery_dir,
                    scores, inferred_log10, true_log10, post_quantiles, run_start)
    else:
        write_recovery_outputs(reporter, args, eval_cfg, spec.draw_spec, recovery_array_path,
                               scores, inferred_log10, true_log10, post_quantiles,
                               run_start)


def build_evaluation_parser() -> argparse.ArgumentParser:
    """Construct the Evaluation CLI parser (identical for both workflows)."""
    eval_cfg = PARAMETERS.inference.evaluation
    parser = argparse.ArgumentParser(
        description="Evaluate a trained posterior by MAP recovery on the EVAL namespace.",
    )
    parser.add_argument(
        "--total-time-seconds", type=float, required=True,
        help="Video duration in seconds; must match the trained posterior's runs.",
    )
    parser.add_argument(
        "--eval-tasks", type=int, required=True,
        help="Number of EVAL-namespace tasks to recover (held-out data).",
    )
    parser.add_argument(
        "--max-sims", type=int, default=0,
        help="Cap on simulations recovered per EVAL task (0 = all; useful for "
             "quick checks).",
    )
    parser.add_argument(
        "--merge", action="store_true",
        help="Combine-only mode: read the per-shard recovery .npz files written by "
             "a multi-GPU sharded run (in this run's MAP_Recovery directory), "
             "concatenate them, and write the final report + figures + combined "
             ".npz, then exit. Does no recovery and needs no GPU; the launcher runs "
             "it once after the sharded workers finish. Single-GPU runs never use it.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration and inputs, print what would be read/written, then exit "
             "without running the stage (no GPU, no compute). Use before a queue submission or a long local run.",
    )
    parser.add_argument(
        "--seed", type=lambda v: None if str(v).strip().lower() in ("none", "") else int(v), default=None,
        help="Master RNG seed (PyTorch + numpy + Python random). Default None "
             "-> non-deterministic (consistent with generation); pass an int for a "
             "reproducible run.",
    )
    parser.add_argument(
        "--pool-mode", choices=("bounded", "unrestricted"), default=None,
        help="Candidate-pool sampler: 'bounded' (rejection-sample within the "
             "prior; correct for a trained posterior) or 'unrestricted' (sample "
             "the flow directly, no rejection; for smoke tests / undertrained "
             f"posteriors that would stall). Default: {eval_cfg.pool_mode}.",
    )
    parser.add_argument(
        "--summary", choices=("map", "posterior", "both"), default="map",
        help="Which summary views to render: 'map' (View A: MAP-point recovery; "
             "default), 'posterior' (View B: posterior credible intervals + "
             "calibration), or 'both'. View B draws --posterior-samples per video.",
    )
    parser.add_argument(
        "--bin-mode", choices=("prior", "quantile"), default="quantile",
        help="Bin edges for the View A conditional-quantile bands: 'quantile' "
             "(equal-count data-quantile bins; default) or 'prior' (equal-width "
             "bins across the prior range).",
    )
    parser.add_argument(
        "--posterior-samples", type=int, default=None,
        help=f"Samples per video for View B posterior summary "
             f"(default: {eval_cfg.posterior_samples}).",
    )
    parser.add_argument(
        "--n-bins", type=int, default=None,
        help=f"Number of bins for the View A conditional-quantile bands "
             f"(default: {eval_cfg.quantile_bins}). Lower it for a small smoke set.",
    )
    parser.add_argument(
        "--min-count", type=int, default=None,
        help=f"Minimum points per bin to draw a View A band "
             f"(default: {eval_cfg.quantile_min_count}). Set to 1 to force bands "
             f"on a minimal smoke set.",
    )
    parser.add_argument(
        "--theta-prex-size", type=int, default=None,
        help=f"Candidate-pool size per video (default: {eval_cfg.theta_prex_size}).",
    )
    parser.add_argument(
        "--elite-prex-size", type=int, default=None,
        help=f"Number of optimization seeds / top-K (default: {eval_cfg.elite_prex_size}).",
    )
    parser.add_argument(
        "--numb-steps", type=int, default=None,
        help=f"Max gradient-ascent steps (default: {eval_cfg.numb_steps}). The "
             "optimization may stop earlier via the patience criterion.",
    )
    parser.add_argument(
        "--show-progress-steps", type=int, default=None,
        help="Cadence (in steps) for the per-step log-prob progress line "
             f"(default: {eval_cfg.show_progress_steps}). Lower = more frequent.",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Adam learning rate for the MAP optimization. Default: "
             "learning_rate_minimum * learning_rate_maximum_factor.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Rich per-video console diagnostics: per-stage shapes/timings, "
             "per-step optimization progress, stopping reason, and the optimal "
             "vs. original theta in log10 and physical units.",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Implies --verbose, and goes deeper (the 'verbose' "
             "level): sbi sampling progress bars + optimizer-config on the "
             "console, plus per-video / per-step detail written into progress.log.",
    )
    parser.add_argument(
        "--debug-dump", action="store_true",
        help="Implies --debug; additionally tees the full console transcript to "
             "<data_bank>/Labor/Debug/<run>/Evaluation/console.log (the recovery "
             "report itself always lands in Posit/).",
    )
    return parser

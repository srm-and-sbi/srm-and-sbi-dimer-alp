"""Entry-point (Detector workflow): MAP-recovery of imaging parameters on EVAL.

Part of the Detector calibration workflow (DETECTOR_WORKFLOW.md §9.2, B5) — a
complete calibration workflow parallel to the canonical pipeline, with its own
committed submission machinery. Mirrors the canonical
``SRM_AND_SBI_DIMER_ALP_Evaluation.py`` — the same seed-then-optimize
``evaluation.map_estimate`` + ``recovery_table`` machinery, and the same
multi-GPU sharding — applied to the held-out ``_DETECTOR`` EVAL namespace, to
measure how well the trained imaging posterior recovers the known imaging
parameters of synthetic videos.

Multi-GPU: with more than one worker (``torchrun``) the EVAL tasks split across
workers round-robin by rank; each worker writes its partial recovery arrays as a
shard, and a separate ``--merge`` step (single process, no GPU) concatenates the
shards into the final report + arrays. With one worker it writes the report
directly (no shard, no merge) — identical to the canonical Evaluation.

Detector differences: the estimator is loaded from the A5 version-portable
artifact (``artifacts.load_estimator``) rather than a pickled posterior; the
prior, parameter table, and data paths are the Detector's (``det.build_prior``,
``det.DETECTOR_PARAMETERIZATION``, ``paths=detector_paths(...)``); all outputs are
``_DETECTOR``-namespaced.

Outputs (``_DETECTOR``-namespaced, under Posit):
    <alias>_DETECTOR_{timing_label}_MAP_Recovery/<alias>_DETECTOR_{timing_label}_MAP_Recovery.npz
    (+ per-parameter recovery figures when --show)

Usage:
    # single GPU (writes report directly):
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Evaluation.py \\
        --total-time-seconds 5.0 --eval-tasks 2 --pool-mode unrestricted
    # multi-GPU (shard across N workers, then merge):
    torchrun --standalone --nproc_per_node=N SRM_AND_SBI_DIMER_ALP_DETECTOR_Evaluation.py \\
        --total-time-seconds 5.0 --eval-tasks 2 --pool-mode unrestricted
    python SRM_AND_SBI_DIMER_ALP_DETECTOR_Evaluation.py \\
        --total-time-seconds 5.0 --eval-tasks 2 --pool-mode unrestricted --merge
    (add --dry-run to resolve config + planned I/O without loading data or compute)
"""

import argparse
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

from srm_and_sbi_dimer_alp import artifacts
from srm_and_sbi_dimer_alp import detector_parameterization as det
from srm_and_sbi_dimer_alp.evaluation import map_estimate, recovery_table
from srm_and_sbi_dimer_alp.inference_support import resolve_topology
from srm_and_sbi_dimer_alp.io import load_data
from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming


def _estimator_path(paths, data_bank_root, timing_label):
    return (data_bank_root / paths.posit_subdir /
            f"{paths.project_alias}_{timing_label}_Estimator.npz")


def _shard_array_path(recovery_dir: Path, rank: int, world_size: int) -> Path:
    """Path of one worker's partial recovery arrays in a multi-GPU sharded run."""
    return recovery_dir / f"_shard_{rank:02d}_of_{world_size:02d}.npz"


def write_recovery_outputs(args, eval_cfg, parameterization, recovery_dir, recovery_array_path,
                           scores, inferred_log10, true_log10, run_start) -> None:
    """Save the recovery arrays and emit the recovery table (+ figures with --show).

    Shared by the single-process recovery path and the ``--merge`` combine step,
    so both emit an identical report from the same code.
    """
    scores = np.asarray(scores)
    inferred_log10 = np.asarray(inferred_log10)
    true_log10 = np.asarray(true_log10)
    n_samples = inferred_log10.shape[0]

    recovery_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(recovery_array_path),
                        true_log10=true_log10, inferred_log10=inferred_log10, scores=scores)
    print(f"\nRecovery arrays saved to {recovery_array_path}")

    guide, guide_tight = eval_cfg.error_guide, eval_cfg.error_guide_tight
    headers, rows = recovery_table(parameterization, true_log10, inferred_log10, guide, guide_tight)
    print(f"\nMAP recovery over {n_samples} EVAL video(s) (imaging θ, log10 units):")
    print("  " + " | ".join(str(h) for h in headers))
    for row in rows:
        print("  " + " | ".join(str(c) for c in row))

    if args.show:
        from srm_and_sbi_dimer_alp.visualization_inference import figure_recovery_combined
        figdir = recovery_dir / "figures"
        figdir.mkdir(parents=True, exist_ok=True)
        for i, para in enumerate(parameterization):
            key = para["KEY"]
            fig = figure_recovery_combined(
                true_log10[:, i], inferred_log10[:, i], None, para["PRIOR_RANGE"],
                para.get("LABEL") or key, error_guide=guide, error_guide_tight=guide_tight,
                error_ylim_floor=eval_cfg.error_ylim_floor,
                error_ylim_quantile=eval_cfg.error_ylim_quantile,
                show_map=True, show_posterior=False)
            fig.savefig(str(figdir / f"recovery_{key}.png"), dpi=PARAMETERS.plotting.dpi)
        print(f"Recovery figures saved to {figdir}")

    print(f"\nTotal elapsed: {time.time() - run_start:.1f}s")


def _save_shard(topo, recovery_dir: Path, scores, inferred_log10, true_log10, run_start) -> None:
    """Write this worker's partial recovery arrays (multi-GPU sharded run).

    The report is produced later by the ``--merge`` step, once every shard
    exists; this function does no report generation. Mirrors the canonical
    Evaluation: a worker that drew no tasks writes no shard at all.
    """
    recovery_dir.mkdir(parents=True, exist_ok=True)
    if len(scores) == 0:
        print(f"\n[rank {topo.rank}/{topo.world_size}] no tasks assigned -- "
              f"no shard written.", flush=True)
        return
    arrays = dict(scores=np.asarray(scores),
                  inferred_log10=np.asarray(inferred_log10),
                  true_log10=np.asarray(true_log10))
    path = _shard_array_path(recovery_dir, topo.rank, topo.world_size)
    np.savez_compressed(str(path), **arrays)
    print(f"\n[rank {topo.rank}/{topo.world_size}] shard saved: {path} "
          f"({arrays['scores'].shape[0]} videos) in {time.time() - run_start:.1f}s. "
          f"Run the --merge step once all shards finish.", flush=True)


def _merge_shards(args, eval_cfg, parameterization, recovery_dir: Path,
                  recovery_array_path: Path, run_start) -> None:
    """Combine all per-shard recovery arrays into the final report (no recovery).

    Reads every ``_shard_*_of_*.npz`` in the recovery directory, concatenates the
    arrays (order-independent — the recovery metrics are aggregates over videos),
    writes the final report + arrays (+ figures) via :func:`write_recovery_outputs`,
    then removes the shard files. Mirrors the canonical Evaluation ``--merge``.
    """
    shard_paths = sorted(recovery_dir.glob("_shard_*_of_*.npz"))
    if not shard_paths:
        raise SystemExit(
            f"--merge: no shard files (_shard_*_of_*.npz) found in {recovery_dir}")
    print(f"Merging {len(shard_paths)} shard file(s) from {recovery_dir}", flush=True)
    scores, inferred, true = [], [], []
    n_used = 0
    for shard_path in shard_paths:
        with np.load(str(shard_path)) as data:
            if data["scores"].shape[0] == 0:
                continue   # defensive: a zero-video shard contributes nothing
            n_used += 1
            scores.append(data["scores"])
            inferred.append(data["inferred_log10"])
            true.append(data["true_log10"])
    if not scores:
        raise SystemExit(
            f"--merge: every shard in {recovery_dir} was empty (no recovered videos)")
    scores = np.concatenate(scores, axis=0)
    inferred = np.concatenate(inferred, axis=0)
    true = np.concatenate(true, axis=0)
    print(f"Merged {scores.shape[0]} videos from {n_used} shard(s).", flush=True)
    write_recovery_outputs(args, eval_cfg, parameterization, recovery_dir,
                           recovery_array_path, scores, inferred, true, run_start)
    for shard_path in shard_paths:
        shard_path.unlink()
    print(f"Removed {len(shard_paths)} shard file(s).", flush=True)


def main(args: argparse.Namespace) -> None:
    timing = RunTiming(total_time_seconds=args.total_time_seconds,
                       frames=PARAMETERS.simulation.timing)
    data_bank_root = PARAMETERS.machine.data_bank_root      # EVAL + outputs on the permanent tier
    compress = True
    paths = det.detector_paths(PARAMETERS.paths)
    timing_label = timing.label
    eval_cfg = PARAMETERS.inference.evaluation
    parameterization = det.DETECTOR_PARAMETERIZATION
    imaging_keys = [e["KEY"] for e in parameterization]
    div = "=" * 72

    estimator_path = _estimator_path(paths, data_bank_root, timing_label)
    recovery_dir = paths.map_recovery_dir(data_bank_root, timing_label)
    recovery_array_path = paths.map_recovery_array_path(data_bank_root, timing_label)

    # map_estimate hyperparameters (config defaults, overridable).
    theta_prex_size = args.theta_prex_size or eval_cfg.theta_prex_size
    elite_prex_size = args.elite_prex_size or eval_cfg.elite_prex_size
    numb_steps = args.numb_steps or eval_cfg.numb_steps
    lr = eval_cfg.learning_rate_minimum * eval_cfg.learning_rate_maximum_factor
    tolerance = eval_cfg.learning_rate_minimum * eval_cfg.tolerance_factor
    pool_mode = args.pool_mode or eval_cfg.pool_mode

    print(div)
    print(f" {paths.project_alias} — Detector Evaluation (imaging MAP recovery)")
    print(f" Started at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(div)
    print("\nRun configuration:")
    print(f"  --total-time-seconds : {args.total_time_seconds}  (n_frames={timing.frame_count})")
    print(f"  --eval-tasks         : {args.eval_tasks}")
    print(f"  --pool-mode          : {pool_mode}")
    print(f"  --merge              : {args.merge}")
    print(f"  theta_prex/elite/steps: {theta_prex_size}/{elite_prex_size}/{numb_steps}")
    print("\nInput / output (Detector-namespaced):")
    print(f"  reads estimator (A5) : {estimator_path}")
    print(f"  reads EVAL           : <data_bank>/Video|Theta/{paths.project_alias}_"
          f"{timing_label}_{{Video|Theta}}_Set_TASK_{{n}}_EVAL.zarr")
    print(f"  writes recovery      : {recovery_array_path}")
    print(f"\n{div}\n")

    if args.dry_run:
        print("DRY RUN — configuration + inputs resolved; no estimator/data loaded, no MAP.")
        print(f"  would recover imaging θ ({len(imaging_keys)} params) over {args.eval_tasks} "
              f"EVAL task(s) using pool_mode={pool_mode}.")
        return

    run_start = time.time()

    # --merge: combine the per-shard arrays from a multi-GPU sharded run into the
    # final report, then exit (no recovery work, no GPU, no estimator needed).
    if args.merge:
        _merge_shards(args, eval_cfg, parameterization, recovery_dir,
                      recovery_array_path, run_start)
        return

    topo = resolve_topology()
    device = topo.device
    vista_device = torch.device("cpu")

    # ---- Load the Detector estimator via the A5 version-portable format ----
    posterior = artifacts.load_estimator(estimator_path, device=str(device))
    if device.type == "cuda":
        # Rebuild the prior on THIS worker's device (each rank binds its own
        # cuda:local_rank under sharding); equivalent on the single-GPU path.
        posterior.prior = det.build_prior(device=str(device))

    # ---- This worker's task shard (round-robin by rank; single worker takes all) ----
    my_tasks = [t for t in range(args.eval_tasks) if t % topo.world_size == topo.rank]
    shard_note = (f" [shard rank {topo.rank}/{topo.world_size}: {len(my_tasks)} of "
                  f"{args.eval_tasks} tasks]" if topo.is_distributed else "")

    # ---- MAP recovery over this worker's EVAL tasks ------------------------
    scores, inferred_log10, true_log10 = [], [], []
    total = 0
    print(f"START MAP recovery: {len(my_tasks)} EVAL task(s){shard_note}, "
          f"pool_mode={pool_mode}.", flush=True)
    for task in my_tasks:
        video_set = load_data(paths.video_set_path(task, data_bank_root, timing_label, compress, "EVAL"))
        theta_set = load_data(paths.theta_set_path(task, data_bank_root, timing_label, compress, "EVAL"))
        n_sims = theta_set.shape[0] if args.max_sims <= 0 else min(theta_set.shape[0], args.max_sims)
        for sim in range(n_sims):
            video_chunk = np.asarray(video_set[sim])
            score, theta_log = map_estimate(
                posterior, video_chunk, device, vista_device,
                theta_prex_size, eval_cfg.theta_prex_batch_size,
                eval_cfg.score_prex_batch_size, elite_prex_size,
                numb_steps, eval_cfg.optimizer_patience,
                eval_cfg.scheduler_patience, eval_cfg.show_progress_steps,
                eval_cfg.learning_rate_minimum, eval_cfg.learning_rate_factor,
                lr, tolerance, pool_mode=pool_mode, show=False, verbose=args.verbose,
            )
            scores.append(score)
            inferred_log10.append(np.asarray(theta_log, dtype=float))
            true_log10.append(np.log10(np.asarray(theta_set[sim], dtype=float)))
            total += 1
            rank_tag = f"[rank {topo.rank}] " if topo.is_distributed else ""
            print(f"  {rank_tag}recovered task {task} sim {sim}  (log_prob {float(score):.3f})  "
                  f"[{total} done]", flush=True)

    # ---- Write outputs ---------------------------------------------------
    # One worker (world_size == 1) writes the report directly. Multiple workers
    # each write partial arrays; the separate --merge step (run by the launcher
    # once all shards finish) combines them into the report.
    if topo.is_distributed:
        _save_shard(topo, recovery_dir, scores, inferred_log10, true_log10, run_start)
    else:
        write_recovery_outputs(args, eval_cfg, parameterization, recovery_dir,
                               recovery_array_path, scores, inferred_log10, true_log10, run_start)


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detector workflow: MAP-recovery of imaging parameters on the held-out "
                    "EVAL namespace (reuses evaluation.map_estimate + recovery_table; "
                    "multi-GPU sharded like the canonical Evaluation).")
    p.add_argument("--total-time-seconds", type=float, required=True)
    p.add_argument("--eval-tasks", type=int, default=1, help="EVAL-namespace Detector tasks.")
    p.add_argument("--max-sims", type=int, default=0, help="Cap sims per task (0 = all).")
    p.add_argument("--pool-mode", type=str, default=None,
                   choices=["bounded", "unrestricted"],
                   help="Candidate-pool sampler, exactly as the canonical Evaluation: "
                        "'bounded' (rejection-sample within the prior; correct for a trained "
                        "posterior) or 'unrestricted' (sample the flow directly, no rejection; "
                        "for smoke tests / undertrained posteriors that would stall). "
                        "Default: the config value (bounded).")
    p.add_argument("--theta-prex-size", type=int, default=0, help="Candidate pool size (0 = config).")
    p.add_argument("--elite-prex-size", type=int, default=0, help="Optimization seeds (0 = config).")
    p.add_argument("--numb-steps", type=int, default=0, help="Max gradient-ascent steps (0 = config).")
    p.add_argument("--merge", action="store_true",
                   help="Combine-only mode: read the per-shard recovery .npz files written by a "
                        "multi-GPU sharded run (in this run's MAP_Recovery directory), concatenate "
                        "them, write the final report + arrays (+ figures), then exit. Does no "
                        "recovery and needs no GPU; the launcher runs it once after the sharded "
                        "workers finish. Single-GPU runs never use it.")
    p.add_argument("--show", action="store_true", help="Save per-parameter recovery figures.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve config + planned I/O and exit without loading data or MAP.")
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())

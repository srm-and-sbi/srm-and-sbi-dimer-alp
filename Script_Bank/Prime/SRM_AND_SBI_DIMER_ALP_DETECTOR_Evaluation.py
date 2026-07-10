"""Entry-point (Detector workflow): MAP-recovery of imaging parameters on EVAL.

Special-situation entry point of the Detector calibration workflow
(DETECTOR_WORKFLOW.md §9.2, B5). Mirrors the canonical
``SRM_AND_SBI_DIMER_ALP_Evaluation.py`` — the same seed-then-optimize
``evaluation.map_estimate`` + ``recovery_table`` machinery — applied to the
held-out ``_DETECTOR`` EVAL namespace, to measure how well the trained imaging
posterior recovers the known imaging parameters of synthetic videos.

Detector differences: the estimator is loaded from the A5 version-portable
artifact (``artifacts.load_estimator``) rather than a pickled posterior; the
prior, parameter table, and data paths are the Detector's (``det.build_prior``,
``det.DETECTOR_PARAMETERIZATION``, ``paths=detector_paths(...)``); all outputs are
``_DETECTOR``-namespaced. Single-process here (the smoke runs single-GPU);
multi-GPU sharding mirrors the canonical Evaluation and is a later addition.

Outputs (``_DETECTOR``-namespaced, under Posit):
    <alias>_DETECTOR_{timing_label}_MAP_Recovery/<alias>_DETECTOR_{timing_label}_MAP_Recovery.npz
    (+ per-parameter recovery figures when --show)

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Evaluation.py \\
        --total-time-seconds 5.0 --eval-tasks 2 --pool-mode unrestricted
    (add --dry-run to resolve config + planned I/O without loading data or compute)
"""

import argparse
import time
from datetime import datetime, timezone

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
    pool_mode = args.pool_mode

    print(div)
    print(f" {paths.project_alias} — Detector Evaluation (imaging MAP recovery)")
    print(f" Started at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(div)
    print("\nRun configuration:")
    print(f"  --total-time-seconds : {args.total_time_seconds}  (n_frames={timing.frame_count})")
    print(f"  --eval-tasks         : {args.eval_tasks}")
    print(f"  --pool-mode          : {pool_mode}")
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
    topo = resolve_topology()
    device = topo.device
    vista_device = torch.device("cpu")

    # ---- Load the Detector estimator via the A5 version-portable format ----
    posterior = artifacts.load_estimator(estimator_path, device=str(device))
    if device.type == "cuda":
        posterior.prior = det.build_prior(device=str(device))

    # ---- MAP recovery over the EVAL namespace (single process) -------------
    scores, inferred_log10, true_log10 = [], [], []
    total = 0
    for task in range(args.eval_tasks):
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
            print(f"  recovered task {task} sim {sim}  (log_prob {float(score):.3f})  "
                  f"[{total} done]", flush=True)

    inferred_log10 = np.asarray(inferred_log10)
    true_log10 = np.asarray(true_log10)
    scores = np.asarray(scores)

    # ---- Recovery report + arrays (Detector-namespaced) --------------------
    recovery_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(str(recovery_array_path),
                        true_log10=true_log10, inferred_log10=inferred_log10, scores=scores)
    print(f"\nRecovery arrays saved to {recovery_array_path}")

    guide, guide_tight = eval_cfg.error_guide, eval_cfg.error_guide_tight
    headers, rows = recovery_table(parameterization, true_log10, inferred_log10, guide, guide_tight)
    print(f"\nMAP recovery over {total} EVAL video(s) (imaging θ, log10 units):")
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


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detector workflow: MAP-recovery of imaging parameters on the held-out "
                    "EVAL namespace (reuses evaluation.map_estimate + recovery_table).")
    p.add_argument("--total-time-seconds", type=float, required=True)
    p.add_argument("--eval-tasks", type=int, default=1, help="EVAL-namespace Detector tasks.")
    p.add_argument("--max-sims", type=int, default=0, help="Cap sims per task (0 = all).")
    p.add_argument("--pool-mode", type=str, default="unrestricted",
                   choices=["bounded", "unrestricted"],
                   help="Candidate-pool sampler; 'unrestricted' suits smoke/undertrained posteriors.")
    p.add_argument("--theta-prex-size", type=int, default=0, help="Candidate pool size (0 = config).")
    p.add_argument("--elite-prex-size", type=int, default=0, help="Optimization seeds (0 = config).")
    p.add_argument("--numb-steps", type=int, default=0, help="Max gradient-ascent steps (0 = config).")
    p.add_argument("--show", action="store_true", help="Save per-parameter recovery figures.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve config + planned I/O and exit without loading data or MAP.")
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())

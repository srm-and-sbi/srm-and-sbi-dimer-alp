"""Entry-point (Detector workflow): train the imaging-parameter posterior estimator.

Part of the Detector calibration workflow (DETECTOR_WORKFLOW.md §9.2, B3) — a
complete calibration workflow parallel to the canonical pipeline, with its own
committed submission machinery. Mirrors the canonical
``SRM_AND_SBI_DIMER_ALP_Inference.py`` — same MAF-on-Complex3DCNN estimator, same
``setup_training``/``train_loop`` machinery — with three Detector differences:

  1. It reads the ``_DETECTOR``-namespaced ``(video, imaging-theta)`` pairs written
     by the Detector DLI stage (B2), by passing ``paths=detector_paths(...)`` to
     ``setup_training`` (and ``build_datasets`` for the shape-inference batch). The
     estimator's parameter dimension is therefore the 11 learnable imaging
     parameters (the imaging-theta labels), not the canonical RDS set.
  2. It persists the trained estimator via the self-describing, version-portable
     A5 format (``artifacts.save_estimator``) — a compile-stripped state_dict +
     rebuild spec + metadata — instead of the canonical pickled ``DirectPosterior``.
     The raw ``Optimum_ANN`` checkpoint is still written by ``train_loop``.
  3. All outputs are ``_DETECTOR``-prefixed, so they never collide with canonical
     inference artifacts.

The per-example test-loss distribution (0.2.16) is not wired here yet; it is a
later reuse of that machinery with a Detector-appropriate manifest.

Outputs (``_DETECTOR``-namespaced):
    <data_bank>/<labor_subdir>/<alias>_DETECTOR_{timing_label}_Optimum_ANN.pth
    <data_bank>/<posit_subdir>/<alias>_DETECTOR_{timing_label}_Estimator.npz  (A5 format)

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Inference.py \\
        --total-time-seconds 5.0 --epochs 5 --tasks 16 --test-tasks 4 --batch-size 8
    (add --dry-run to resolve config + planned I/O without loading data or compute)
"""

import argparse
import math
import random
import shutil
import time
from datetime import datetime, timezone

import numpy as np
import torch
import torch._dynamo
from sbi.neural_nets.net_builders import build_maf
from torch.utils.data import DataLoader

from srm_and_sbi_dimer_alp import artifacts
from srm_and_sbi_dimer_alp import detector_parameterization as det
from srm_and_sbi_dimer_alp.inference_network import Complex3DCNN
from srm_and_sbi_dimer_alp.inference_support import (
    build_datasets,
    cleanup_distributed,
    init_distributed,
    resolve_topology,
    setup_training,
    train_loop,
)
from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming
from srm_and_sbi_dimer_alp.utils import log_memory_state


def _estimator_path(paths, data_bank_root, timing_label):
    """Detector A5 estimator-artifact path (Posit, Detector-aliased)."""
    return (data_bank_root / paths.posit_subdir /
            f"{paths.project_alias}_{timing_label}_Estimator.npz")


def main(args: argparse.Namespace) -> None:
    timing = RunTiming(total_time_seconds=args.total_time_seconds,
                       frames=PARAMETERS.simulation.timing)
    data_bank_root = PARAMETERS.machine.data_bank_root      # permanent tier: checkpoint + estimator outputs
    train_data_root = PARAMETERS.machine.root_for("TRAIN")  # scratch tier: training inputs
    compress = True
    paths = det.detector_paths(PARAMETERS.paths)            # DETECTOR-aliased filenames
    timing_label = timing.label
    net_cfg = PARAMETERS.inference.network
    div = "=" * 72

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.suppress_errors = True

    checkpoint_path = paths.checkpoint_path(data_bank_root, timing_label)
    estimator_path = _estimator_path(paths, data_bank_root, timing_label)
    imaging_keys = [e["KEY"] for e in det.DETECTOR_PARAMETERIZATION]
    theta_dim = len(imaging_keys)                           # 11 learnable imaging params
    root_px = PARAMETERS.simulation.stem.root_size_px
    video_shape = (timing.frame_count, root_px, root_px)

    print(div)
    print(f" {paths.project_alias} — Detector Inference (imaging posterior)")
    print(f" Started at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(div)
    print("\nRun configuration:")
    print(f"  --total-time-seconds : {args.total_time_seconds}  (n_frames={timing.frame_count})")
    print(f"  --epochs             : {args.epochs}")
    print(f"  --batch-size         : {args.batch_size}")
    print(f"  --tasks (TRAIN)      : {args.tasks}")
    print(f"  --test-tasks (TEST)  : {args.test_tasks}")
    print(f"  --resurrect          : {args.resurrect}")
    print(f"  --seed               : {args.seed}")
    print(f"  embedding            : Complex3DCNN(n_conv={net_cfg.n_conv_layers}, "
          f"n_attn={net_cfg.n_attn_layers}, start_ch={net_cfg.start_channels})")
    print(f"  estimator            : MAF (z_score=structured, dropout=0.1, batch_norm=True); "
          f"theta_dim={theta_dim} (imaging)")
    print("\nInput / output (Detector-namespaced):")
    print(f"  reads (TRAIN/TEST)   : <scratch>/Video|Theta/{paths.project_alias}_{timing_label}_"
          f"{{Video|Theta}}_Set_TASK_{{n}}_{{TRAIN|TEST}}.zarr")
    print(f"  writes checkpoint    : {checkpoint_path}")
    print(f"  writes estimator(A5) : {estimator_path}")
    print(f"\n{div}\n")

    if args.dry_run:
        print("DRY RUN — configuration + inputs resolved; no data loaded, no training.")
        print(f"  imaging theta_dim = {theta_dim}: {imaging_keys}")
        print(f"  video shape per example = {video_shape}")
        print(f"  would train {args.epochs} epoch(s) on {args.tasks} TRAIN "
              f"+ {args.test_tasks} TEST Detector task(s), then save the A5 estimator artifact.")
        return

    run_start = time.time()

    # ---- Topology / device ----------------------------------------------
    topo = resolve_topology()
    device = topo.device
    init_distributed(topo)

    # ---- Shape-inference batch (Detector data via paths=det) -------------
    dummy_train, _ = build_datasets(
        train_tasks=1, data_bank_root=train_data_root, timing_label=timing_label,
        compress=compress, paths=paths,
    )
    dummy_loader = DataLoader(dummy_train, batch_size=2, num_workers=0)
    video_dummy, theta_dummy = next(iter(dummy_loader))
    if theta_dummy.shape[1] != theta_dim:
        raise ValueError(
            f"Detector theta labels have width {theta_dummy.shape[1]} but the Detector "
            f"imaging parameterization has {theta_dim} learnable params {imaging_keys}. "
            f"Check that B2 wrote imaging-theta labels (not RDS).")

    # ---- Embedding + MAF estimator (args captured for the A5 rebuild spec) -
    embedding_args = dict(
        n_frames=timing.frame_count,
        input_channels=net_cfg.input_channels,
        n_conv_layers=net_cfg.n_conv_layers,
        n_attn_layers=net_cfg.n_attn_layers,
        start_channels=net_cfg.start_channels,
        use_temporal_attention=net_cfg.use_temporal_attention,
        attention_heads=net_cfg.attention_heads,
        temporal_target_frames=net_cfg.temporal_target_frames,
        verbose=args.verbose,
    )
    maf_args = dict(z_score_x="structured", z_score_y="structured",
                    dropout_probability=0.1, use_batch_norm=True)
    embedding_net = torch.compile(Complex3DCNN(**embedding_args)).to(device)
    posterior_estimator = build_maf(
        batch_x=theta_dummy, batch_y=video_dummy, embedding_net=embedding_net,
        **maf_args).to(device)

    if args.verbose:
        log_memory_state(prefix="[Pre-training]")

    # ---- Train (reuses the canonical pipeline; paths=det for the loaders) -
    training_setup = setup_training(
        estimator=posterior_estimator, train_tasks=args.tasks,
        data_bank_root=train_data_root, timing_label=timing_label, compress=compress,
        batch_size=args.batch_size, learning_rate=args.learning_rate,
        test_tasks=args.test_tasks, paths=paths,
    )
    losses_train, losses_test, losses_replay, optimum_loss_test = train_loop(
        estimator=training_setup["estimator"], model=training_setup["model"],
        train_loader=training_setup["train_loader"], val_loader=training_setup["val_loader"],
        optimizer=training_setup["optimizer"], scheduler=training_setup["scheduler"],
        device=training_setup["device"], topo=training_setup["topo"],
        checkpoint_path=checkpoint_path, train_sampler=training_setup["train_sampler"],
        epochs=args.epochs, resurrect=args.resurrect, replay_loss=args.replay_loss,
        heartbeat_every=args.heartbeat, verbose=args.verbose,
    )

    # ---- Persist the trained estimator (rank 0) via the A5 format ---------
    # Every rank reloaded the best-on-TEST estimator in train_loop, so rank 0's
    # copy is the selected model; only it writes the shared artifact.
    if topo.is_main:
        metadata = {
            "timing_label": timing_label,
            "workflow": "detector",
            "best_test_loss": (float(optimum_loss_test)
                               if math.isfinite(optimum_loss_test) else None),
            "train_videos": len(training_setup["train_loader"].dataset),
            "test_videos": (len(training_setup["val_loader"].dataset)
                            if training_setup["val_loader"] is not None else 0),
            "epochs": args.epochs,
        }
        estimator_path.parent.mkdir(parents=True, exist_ok=True)
        artifacts.save_estimator(
            training_setup["estimator"],
            embedding_args=embedding_args, maf_args=maf_args,
            theta_dim=theta_dim, video_shape=video_shape,
            parameter_keys=imaging_keys,
            prior_low=det.theta_lower_bound(), prior_high=det.theta_upper_bound(),
            path=estimator_path, metadata=metadata,
        )
        print(f"Detector estimator (A5) saved to {estimator_path}")

        # Provenance-named backups (only meaningful with a TEST selection signal).
        if args.test_tasks > 0 and math.isfinite(optimum_loss_test):
            tv, ev = metadata["train_videos"], metadata["test_videos"]
            descriptor = paths.backup_descriptor(tv, ev, args.epochs, optimum_loss_test)
            shutil.copy2(checkpoint_path, paths.backup_checkpoint_path(
                data_bank_root, timing_label, tv, ev, args.epochs, optimum_loss_test))
            est_backup = estimator_path.with_name(
                f"{estimator_path.stem}_{descriptor}{estimator_path.suffix}")
            shutil.copy2(estimator_path, est_backup)
            print(f"Backup checkpoint + estimator saved (descriptor {descriptor})")

    print(f"\nTotal elapsed: {time.time() - run_start:.1f}s")
    cleanup_distributed()


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detector workflow: train the imaging-parameter posterior estimator "
                    "(NPE + MAF on the Complex3DCNN embedding), saved in the A5 version-portable format.")
    p.add_argument("--total-time-seconds", type=float, required=True)
    p.add_argument("--epochs", type=int, default=PARAMETERS.inference.training.epochs)
    p.add_argument("--batch-size", type=int, default=PARAMETERS.inference.training.batch_size)
    p.add_argument("--learning-rate", type=float, default=None)
    p.add_argument("--tasks", type=int, default=2, help="TRAIN-namespace Detector tasks.")
    p.add_argument("--test-tasks", type=int, default=0, help="TEST-namespace Detector tasks (model selection).")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--resurrect", action="store_true",
                   help="Load the previous Detector Optimum_ANN checkpoint and continue.")
    p.add_argument("--replay-loss", action="store_true")
    p.add_argument("--heartbeat", type=int, default=None)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve config + planned I/O and exit without loading data or training.")
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())

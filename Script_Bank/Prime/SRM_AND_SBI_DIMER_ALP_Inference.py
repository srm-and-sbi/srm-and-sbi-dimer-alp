"""Entry-point script: train the neural posterior estimator and save the posterior.

Loads (video, theta) pairs from the previous RDS + DLI stages, trains a masked
autoregressive flow (MAF) density estimator conditioned on a Complex3DCNN +
TemporalTransformer embedding of the videos, and saves the resulting
DirectPosterior to a pickle file for downstream sampling.

Outputs (the ``{timing_label}`` token, e.g. ``2S_50FPS``, is rendered from
``PARAMETERS.simulation.timing.label`` to namespace files by duration + fps):

    <data_bank>/<labor_subdir>/<project_alias>_{timing_label}_Optimum_ANN.pth
        -- the best-so-far estimator checkpoint (overwritten on each new optimum).
    <data_bank>/<posit_subdir>/<project_alias>_{timing_label}_Posterior.pkl
        -- the trained DirectPosterior, ready for downstream sampling.

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_Inference.py \\
        --total-time-seconds 2.0 --epochs 5 --tasks 2

    # Resume training from the previously saved checkpoint:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_Inference.py --resurrect
"""

import argparse
import random
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
from sbi.inference.posteriors import DirectPosterior
from sbi.neural_nets.net_builders import build_maf
from sbi.utils.user_input_checks import process_prior
from torch.utils.data import DataLoader

from srm_and_sbi_dimer_alp.diagnostics import DiagnosticReporter
from srm_and_sbi_dimer_alp.inference_network import Complex3DCNN
from srm_and_sbi_dimer_alp.inference_support import (
    build_datasets,
    cleanup_distributed,
    init_distributed,
    resolve_topology,
    save_posterior,
    setup_training,
    train_loop,
)
from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming, build_prior
from srm_and_sbi_dimer_alp.utils import console_log_context, log_memory_state


def main(args: argparse.Namespace) -> None:
    """Run the full inference training pipeline per the CLI args."""
    timing = RunTiming(
        total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing,
    )
    data_bank_root = PARAMETERS.machine.data_bank_root          # permanent tier: checkpoint + posterior outputs
    train_data_root = PARAMETERS.machine.root_for("TRAIN")      # scratch tier (split storage): training inputs live with TRAIN/TEST
    compress = True  # video and theta sets are always read from .zarr in the inference stage

    # ---- Global RNG / precision settings ---------------------------------
    if args.seed is not None:   # None -> non-deterministic (consistent with generation; allows init diversity)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.suppress_errors = True

    # ---- Pre-run banner -------------------------------------------------
    machine = PARAMETERS.machine
    geom = PARAMETERS.simulation.stem
    training_cfg = PARAMETERS.inference.training
    network_cfg = PARAMETERS.inference.network
    paths = PARAMETERS.paths
    div = "=" * 72

    # Resolve effective learning rate (None → lr_minimum × max_factor).
    default_lr = training_cfg.learning_rate_minimum * training_cfg.learning_rate_maximum_factor
    effective_lr = args.learning_rate if args.learning_rate is not None else default_lr

    task_range = f"0..{args.tasks - 1}" if args.tasks > 1 else "0"

    print(div)
    print(f" {paths.project_alias} — Inference")
    print(f" Started at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(div)

    print("\nMachine profile:")
    print(f"  name              : {machine.name}")
    print(f"  running_mode      : {machine.running_mode}")
    print(f"  compute_backend   : {machine.compute_backend}")
    if machine.gpu_device_index is not None:
        print(f"  gpu_device_index  : {machine.gpu_device_index}")
    print(f"  num_workers       : {machine.num_workers}")

    print("\nRun configuration (CLI args):")
    print(f"  --total-time-seconds : {args.total_time_seconds}")
    print(f"  --epochs             : {args.epochs}")
    print(f"  --batch-size         : {args.batch_size}")
    print(f"  --learning-rate      : {args.learning_rate}   "
          f"(effective: {effective_lr:.2e})")
    print(f"  --tasks              : {args.tasks}        (TRAIN-namespace tasks; gradient data)")
    print(f"  --test-tasks         : {args.test_tasks}        (TEST-namespace tasks; model selection, 0 = none)")
    print(f"  --seed               : {args.seed}")
    print(f"  --resurrect          : {args.resurrect}")
    print(f"  --replay-loss        : {args.replay_loss}        (per-epoch TRAIN loss in eval mode, comparable to TEST; off = cheaper)")
    print(f"  --verbose            : {args.verbose}")
    print(f"  --show               : {args.show}")

    print("\nVideo input shape:")
    print(f"  n_frames per video   : {timing.frame_count}        "
          f"(= {timing.total_time_seconds} / {timing.frame_time_seconds})")
    print(f"  image size           : {geom.root_size_px} × {geom.root_size_px}")
    print(f"  channels             : {network_cfg.input_channels}             "
          f"(grayscale)")

    timing_label = timing.label
    checkpoint_path = paths.checkpoint_path(data_bank_root, timing_label)
    posterior_path = paths.posterior_path(data_bank_root, timing_label)

    # Ensure the checkpoint output directory (Labor/) exists before train_loop
    # writes to it; the posterior directory (Posit/) is created before its save
    # further below. Mirrors the per-stage dir creation in the RDS / DLI scripts.
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Diagnostics reporter (debug mode) ------------------------------
    reporter = DiagnosticReporter(
        stage="Inference",
        enabled=args.debug or args.debug_dump,
        dump=args.debug_dump,
        dump_dir=paths.debug_run_dir(data_bank_root, timing_label, "Inference"),
        run_label=f"{paths.project_alias}_{timing_label}",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    print("\nOutput destinations:")
    print(f"  data_bank_root  : {data_bank_root}")
    print(f"  reads videos    : <data_bank>/{paths.video_subdir}/"
          f"{paths.project_alias}_{timing_label}_Video_Set_TASK_{{{task_range}}}.zarr")
    print(f"  reads thetas    : <data_bank>/{paths.theta_subdir}/"
          f"{paths.project_alias}_{timing_label}_Theta_Set_TASK_{{{task_range}}}.zarr")
    print(f"  writes ckpt     : {checkpoint_path}")
    print(f"  writes posterior: {posterior_path}")

    print("\nTraining & network summary:")
    print(f"  optimizer : AdamW   "
          f"scheduler: ReduceLROnPlateau (factor={training_cfg.scheduler_factor}, "
          f"patience={training_cfg.scheduler_patience})")
    print(f"  embedding : Complex3DCNN(n_conv={network_cfg.n_conv_layers}, "
          f"n_attn={network_cfg.n_attn_layers}, "
          f"start_ch={network_cfg.start_channels}) "
          f"+ TemporalTransformer(heads={network_cfg.attention_heads})")
    print(f"  estimator : MAF (z_score=structured, dropout=0.1, batch_norm=True)")
    if args.resurrect:
        print(f"  RESURRECT : will load checkpoint from {checkpoint_path}")

    if args.verbose:
        print("\nDetailed training hyperparameters:")
        print(f"  epochs                       : {args.epochs}")
        print(f"  batch_size                   : {args.batch_size}")
        print(f"  learning_rate (effective)    : {effective_lr:.2e}")
        print(f"  learning_rate_minimum        : {training_cfg.learning_rate_minimum}")
        print(f"  learning_rate_maximum_factor : {training_cfg.learning_rate_maximum_factor}")
        print(f"  scheduler_factor             : {training_cfg.scheduler_factor}")
        print(f"  scheduler_patience           : {training_cfg.scheduler_patience}")
        print(f"  scheduler_tolerance_factor   : {training_cfg.scheduler_tolerance_factor}")
        print(f"  augmentation                 : {training_cfg.augmentation}")

        print("\nDetailed network architecture:")
        print(f"  input_channels         : {network_cfg.input_channels}")
        print(f"  n_conv_layers          : {network_cfg.n_conv_layers}")
        print(f"  n_attn_layers          : {network_cfg.n_attn_layers}")
        print(f"  start_channels         : {network_cfg.start_channels}")
        print(f"  use_temporal_attention : {network_cfg.use_temporal_attention}")
        print(f"  attention_heads        : {network_cfg.attention_heads}")
        print(f"  paras                  : {network_cfg.paras}")
        print(f"  regress                : {network_cfg.regress}")

    print(f"\n{div}\n")

    run_start = time.time()

    topo = resolve_topology()
    device = topo.device
    init_distributed(topo)   # no-op on a single worker; sets up the DDP process group otherwise

    # ---- Dummy batch for MAF shape inference -----------------------------
    # build_maf needs a representative (batch_x, batch_y) pair to compute output dims.
    dummy_train, _ = build_datasets(
        train_tasks=1, data_bank_root=train_data_root, timing_label=timing_label,
        compress=compress,
    )
    dummy_loader = DataLoader(dummy_train, batch_size=2, num_workers=0)
    video_dummy, theta_dummy = next(iter(dummy_loader))

    if reporter.enabled:
        reporter.checkpoint(
            "dataset sample",
            video_shape=tuple(video_dummy.shape),
            theta_shape=tuple(theta_dummy.shape),
        )
        reporter.check(
            "video_n_frames_matches",
            int(video_dummy.shape[1]) == timing.frame_count,
            f"{int(video_dummy.shape[1])} vs {timing.frame_count}",
            note="each video's temporal length equals the network's configured "
                 "n_frames (= duration x frame rate); a mismatch would silently "
                 "misalign the 3D-CNN input.",
        )

    # ---- Embedding network -----------------------------------------------
    network_cfg = PARAMETERS.inference.network
    embedding_net = Complex3DCNN(
        n_frames=timing.frame_count,
        input_channels=network_cfg.input_channels,
        n_conv_layers=network_cfg.n_conv_layers,
        n_attn_layers=network_cfg.n_attn_layers,
        start_channels=network_cfg.start_channels,
        use_temporal_attention=network_cfg.use_temporal_attention,
        attention_heads=network_cfg.attention_heads,
        paras=network_cfg.paras,
        regress=network_cfg.regress,
        verbose=args.verbose,
    )
    embedding_net = torch.compile(embedding_net).to(device)

    # ---- MAF posterior estimator -----------------------------------------
    posterior_estimator = build_maf(
        batch_x=theta_dummy,
        batch_y=video_dummy,
        z_score_x="structured",
        z_score_y="structured",
        embedding_net=embedding_net,
        dropout_probability=0.1,
        use_batch_norm=True,
    ).to(device)

    # ---- Training pipeline -----------------------------------------------
    print("\nLoading training data (TRAIN + TEST videos)...", flush=True)
    training_setup = setup_training(
        estimator=posterior_estimator,
        train_tasks=args.tasks,
        data_bank_root=train_data_root,
        timing_label=timing_label,
        compress=compress,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        test_tasks=args.test_tasks,
    )

    if reporter.enabled:
        train_size = len(training_setup["train_loader"].dataset)
        reporter.check("train_set_nonempty", train_size > 0,
                       f"{train_size} samples (TRAIN namespace)",
                       note="the TRAIN namespace is non-empty (gradient data).")
        reporter.stat("train_size", train_size,
                      note="number of (video, theta) pairs used for gradient updates "
                           "(TRAIN namespace).")
        test_loader = training_setup["val_loader"]
        if test_loader is not None:
            test_size = len(test_loader.dataset)
            reporter.check("test_set_nonempty", test_size > 0,
                           f"{test_size} samples (TEST namespace)",
                           note="the TEST namespace is non-empty (per-epoch model "
                                "selection); never trained on.")
            reporter.stat("test_size", test_size,
                          note="number of pairs used for model selection "
                               "(TEST namespace); never trained on.")
        else:
            reporter.stat("test_size", 0,
                          note="no TEST namespace loaded (--test-tasks 0): trained on "
                               "all of TRAIN; last-epoch checkpoint kept.")

    if args.verbose:
        log_memory_state(prefix="[Pre-training]")

    losses_train, losses_test, losses_replay = train_loop(
        estimator=training_setup["estimator"],
        model=training_setup["model"],
        train_loader=training_setup["train_loader"],
        val_loader=training_setup["val_loader"],
        optimizer=training_setup["optimizer"],
        scheduler=training_setup["scheduler"],
        device=training_setup["device"],
        topo=training_setup["topo"],
        checkpoint_path=checkpoint_path,
        train_sampler=training_setup["train_sampler"],
        epochs=args.epochs,
        resurrect=args.resurrect,
        replay_loss=args.replay_loss,
        verbose=args.verbose,
    )

    if reporter.enabled:
        reporter.check_no_nan_inf("losses_train", losses_train)
        reporter.stat("epochs", args.epochs)
        reporter.stat("n_frames", timing.frame_count,
                      note="temporal length of each video the network ingests.")
        reporter.stat("final_train_loss", float(losses_train[-1]),
                      note="training loss at the last epoch (lower is better).")
        if args.test_tasks > 0:
            # TEST-namespace losses only exist when a selection set is loaded.
            reporter.check_no_nan_inf("losses_test", losses_test)
            reporter.stat("final_test_loss", float(losses_test[-1]),
                          note="selection (TEST) loss at the last epoch; the "
                               "model-selection criterion.")

    # ---- Build and save the posterior (rank 0 only) ----------------------
    # Every rank reloaded the best-on-TEST estimator in train_loop, so rank 0's
    # copy is the selected model; only it writes the shared posterior file.
    if topo.is_main:
        prior_cpu = build_prior(device="cpu")
        prior_cpu, _, _ = process_prior(prior_cpu)
        posterior = DirectPosterior(training_setup["estimator"], prior_cpu)

        prior_device = build_prior(device=str(device))
        posterior_path.parent.mkdir(parents=True, exist_ok=True)
        save_posterior(posterior, prior_device, posterior_path)
        print(f"Posterior saved to {posterior_path}")

    # ---- Diagnostics: confirm outputs written, figure, summary ----------
    if reporter.enabled:
        reporter.check_file("checkpoint", checkpoint_path)
        reporter.check_file("posterior", posterior_path)
        if reporter.dump:
            from srm_and_sbi_dimer_alp.visualization_inference import (
                figure_loss_curves,
            )
            reporter.save_figure(
                "loss_curves",
                figure_loss_curves(losses_train, losses_test, losses_replay),
                caption="Train / test / replay loss per epoch. Diverging or flat "
                        "curves flag training problems.",
            )
        reporter.summary()
        reporter.write_report()

    total_elapsed = time.time() - run_start
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")

    # ---- Optional loss-curve plot ---------------------------------------
    if args.show:
        # Lazy import so HPC headless runs don't pay the matplotlib cost.
        from srm_and_sbi_dimer_alp.visualization_inference import plot_loss_curves
        plot_loss_curves(losses_train, losses_test, losses_replay)

    # Tear down the DDP process group (no-op on a single worker).
    cleanup_distributed()


def parse_args(argv=None) -> argparse.Namespace:
    """Construct the CLI parser and parse argv."""
    parser = argparse.ArgumentParser(
        description="Train the SBI posterior estimator (NPE + MAF on Complex3DCNN embedding).",
    )
    parser.add_argument(
        "--total-time-seconds", type=float,
        required=True,
        help="Video duration in seconds; determines n_frames for the network. "
             "Must match the value used in the corresponding RDS and DLI runs.",
    )
    parser.add_argument(
        "--epochs", type=int, default=PARAMETERS.inference.training.epochs,
        help=f"Number of training epochs (default: {PARAMETERS.inference.training.epochs}).",
    )
    parser.add_argument(
        "--batch-size", type=int, default=PARAMETERS.inference.training.batch_size,
        help=f"DataLoader batch size (default: {PARAMETERS.inference.training.batch_size}).",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=None,
        help="Initial AdamW learning rate. Default: "
             "PARAMETERS.inference.training.learning_rate_minimum * learning_rate_maximum_factor.",
    )
    parser.add_argument(
        "--tasks", type=int, default=2,
        help="Number of TRAIN-namespace tasks for gradient updates (default: 2).",
    )
    parser.add_argument(
        "--test-tasks", type=int, default=0,
        help="Number of TEST-namespace tasks for per-epoch model selection. "
             "0 (default) trains on all of TRAIN and keeps the last-epoch "
             "checkpoint (validate separately on the EVAL namespace).",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Master RNG seed (PyTorch + numpy + Python random). Default None "
             "-> non-deterministic (consistent with generation); pass an int for a "
             "reproducible run.",
    )
    parser.add_argument(
        "--resurrect", action="store_true",
        help="Load the previously saved optimum-ANN checkpoint before training "
             "and continue from those weights. Each --resurrect run improves on "
             "the previous best.",
    )
    parser.add_argument(
        "--replay-loss", action="store_true",
        help="Compute the per-epoch replay loss: the TRAIN loss measured under the "
             "same conditions as the test loss (eval mode -- dropout off, batch-norm "
             "running stats -- with augmentation disabled), for a directly comparable "
             "train-vs-test generalization signal. Off by default: a full extra pass "
             "over TRAIN each epoch, expensive on large datasets.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print diagnostic info during setup and per-epoch training.",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="After training, show training/test/replay loss curves "
             "(interactive matplotlib).",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Enable debug diagnostics: per-step checkpoints, fail-loud invariant "
             "checks, and an end-of-stage PASS/FAIL summary (console only).",
    )
    parser.add_argument(
        "--debug-dump", action="store_true",
        help="Implies --debug; additionally writes a Markdown diagnostic report "
             "and PNG figures under <data_bank>/Debug/. Skipped if disk space is low.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    cli_args = parse_args(sys.argv[1:])
    with console_log_context(cli_args, "Inference"):
        main(cli_args)

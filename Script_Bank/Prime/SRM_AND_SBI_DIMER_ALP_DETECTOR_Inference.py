"""Entry-point (Detector workflow): train the imaging-parameter posterior estimator.

Part of the Detector calibration workflow (DETECTOR_WORKFLOW.md §9.2, B3) — a
complete calibration workflow parallel to the canonical pipeline, with its own
committed submission machinery. It mirrors the canonical
``SRM_AND_SBI_DIMER_ALP_Inference.py`` — the same MAF-on-Complex3DCNN estimator,
the same ``setup_training`` / ``train_loop`` machinery, and the same per-epoch
model-selection, test-loss-distribution, and new-best commit cadence — with the
Detector-specific differences below:

  1. It reads the ``_DETECTOR``-namespaced ``(video, imaging-theta)`` pairs written
     by the Detector DLI stage (B2), by passing ``paths = detector_paths(...)`` to
     ``build_datasets`` and ``setup_training``. The estimator's parameter dimension
     is therefore the 11 learnable imaging parameters (``theta_dim = 11``), not the
     canonical RDS set, and the prior, bounds, and per-parameter metadata are
     sourced from ``detector_parameterization`` — whose value-based role scheme
     distinguishes the learnable imaging parameters from the marginalized RDS
     nuisance rows.
  2. It persists the trained estimator via the self-describing, version-portable
     A5 format (``artifacts.save_estimator``) — a compile-stripped state_dict +
     rebuild spec + metadata — in place of the canonical pickled ``DirectPosterior``,
     which bakes in ``torch.compile`` internals and is torch-version-locked. The A5
     artifact is the Detector's estimator; the canonical ``save_posterior`` path is
     left untouched (DETECTOR_WORKFLOW.md §7).
  3. All outputs are ``_DETECTOR``-namespaced (via the aliased ``project_alias``),
     so they never collide with canonical inference artifacts.

Outputs (``_DETECTOR``-namespaced; the ``{timing_label}`` token, e.g. ``5S_50FPS``,
is rendered from ``PARAMETERS.simulation.timing.label`` to namespace files by
duration + fps):

    <data_bank>/<labor_subdir>/<alias>_DETECTOR_{timing_label}_Optimum_ANN.pth
        -- the best-so-far estimator checkpoint (overwritten on each new optimum).
    <data_bank>/<posit_subdir>/<alias>_DETECTOR_{timing_label}_Estimator.npz
        -- the A5 version-portable estimator artifact (the Detector's posterior artifact).

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Inference.py \\
        --total-time-seconds 5.0 --epochs 5 --tasks 16 --test-tasks 4 --batch-size 8

    # Resume training from the previously saved checkpoint:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Inference.py --resurrect
"""

import argparse
import math
import random
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch._dynamo
from sbi.neural_nets.net_builders import build_maf
from torch.utils.data import DataLoader

from srm_and_sbi_dimer_alp import artifacts
from srm_and_sbi_dimer_alp import detector_parameterization as det
from srm_and_sbi_dimer_alp.diagnostics import DiagnosticReporter
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
from srm_and_sbi_dimer_alp.test_loss_distribution import TestLossDistribution
from srm_and_sbi_dimer_alp.utils import console_log_context, log_memory_state


def _estimator_path(paths, data_bank_root, timing_label):
    """Detector A5 estimator-artifact path (Posit, Detector-aliased)."""
    return (data_bank_root / paths.posit_subdir /
            f"{paths.project_alias}_{timing_label}_Estimator.npz")


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
    paths = det.detector_paths(PARAMETERS.paths)   # DETECTOR-aliased filenames (every path pattern namespaces separately)
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
    tld_path = paths.test_loss_distribution_path(data_bank_root, timing_label)
    estimator_path = _estimator_path(paths, data_bank_root, timing_label)   # A5 artifact (Detector)
    imaging_keys = [entry["KEY"] for entry in det.DETECTOR_PARAMETERIZATION]
    theta_dim = len(imaging_keys)                               # 11 learnable imaging params
    video_shape = (timing.frame_count, geom.root_size_px, geom.root_size_px)

    print("\nOutput destinations:")
    print(f"  data_bank_root  : {data_bank_root}")
    print(f"  reads videos    : <data_bank>/{paths.video_subdir}/"
          f"{paths.project_alias}_{timing_label}_Video_Set_TASK_{{{task_range}}}.zarr")
    print(f"  reads thetas    : <data_bank>/{paths.theta_subdir}/"
          f"{paths.project_alias}_{timing_label}_Theta_Set_TASK_{{{task_range}}}.zarr")
    print(f"  writes ckpt     : {checkpoint_path}")
    print(f"  writes estimator: {estimator_path}   (A5, version-portable)")

    # ---- Dry-run preview --------------------------------------------------
    # Validate configuration and report the resolved inputs, then exit before
    # any distributed init, GPU use, dataset build, or training loop. Runs as a
    # plain single process (placed before resolve_topology / init_distributed),
    # so no torchrun is needed. Only cheap .exists() probes are performed.
    if args.dry_run:
        print(f"\n{div}")
        print(" [DRY RUN] no training performed -- input validation only")
        print(div)
        # Inputs live on the TRAIN/TEST split storage tier (train_data_root),
        # read via the same PARAMETERS.paths API and compress flag the real run
        # uses through build_datasets / VideoDataset. Probing the first task
        # file of each required split is sufficient to confirm the namespace.
        missing = 0
        train_video = paths.video_set_path(
            0, train_data_root, timing_label, compress, "TRAIN")
        train_theta = paths.theta_set_path(
            0, train_data_root, timing_label, compress, "TRAIN")
        print(f"\n[DRY RUN] TRAIN namespace ({args.tasks} task(s) expected, probing TASK_0):")
        for role, set_path in (("TRAIN video set", train_video),
                               ("TRAIN theta set", train_theta)):
            if Path(set_path).exists():
                print(f"  reads {role}: {set_path}  [OK]")
            else:
                print(f"  reads {role}: {set_path}  [MISSING]")
                missing += 1
        if args.test_tasks > 0:
            test_video = paths.video_set_path(
                0, train_data_root, timing_label, compress, "TEST")
            test_theta = paths.theta_set_path(
                0, train_data_root, timing_label, compress, "TEST")
            print(f"\n[DRY RUN] TEST namespace ({args.test_tasks} task(s) expected, probing TASK_0):")
            for role, set_path in (("TEST video set", test_video),
                                   ("TEST theta set", test_theta)):
                if Path(set_path).exists():
                    print(f"  reads {role}: {set_path}  [OK]")
                else:
                    print(f"  reads {role}: {set_path}  [MISSING]")
                    missing += 1
        else:
            print("\n[DRY RUN] TEST namespace: skipped (--test-tasks 0)")
        print()
        if missing:
            print(f"[DRY RUN] configuration validated; {missing} input(s) MISSING.")
        else:
            print("[DRY RUN] configuration validated; all inputs present.")
        print("[DRY RUN] no inference performed.")
        print(f"{div}\n")
        return

    # Ensure the checkpoint output directory (Labor/) exists before train_loop
    # writes to it; the posterior directory (Posit/) is created before its save
    # further below. Mirrors the per-stage dir creation in the RDS / DLI scripts.
    # Placed after the dry-run early-return so a dry-run creates no directories.
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    # ---- Diagnostics reporter (debug mode) ------------------------------
    # Constructed after the dry-run early-return: under --debug-dump it creates the
    # debug dump directory, which a dry-run must not do.
    reporter = DiagnosticReporter(
        stage="Inference",
        enabled=args.debug or args.debug_dump,
        dump=args.debug_dump,
        dump_dir=paths.debug_run_dir(data_bank_root, timing_label, "Inference"),
        run_label=f"{paths.project_alias}_{timing_label}",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    print("\nTraining & network summary:")
    print(f"  optimizer : AdamW   "
          f"scheduler: ReduceLROnPlateau (factor={training_cfg.scheduler_factor}, "
          f"patience={training_cfg.scheduler_patience})")
    print(f"  embedding : Complex3DCNN(n_conv={network_cfg.n_conv_layers}, "
          f"n_attn={network_cfg.n_attn_layers}, "
          f"start_ch={network_cfg.start_channels}, "
          f"temporal_target_frames={network_cfg.temporal_target_frames}) "
          f"+ TemporalTransformer(heads={network_cfg.attention_heads})")
    print("  estimator : MAF (z_score=structured, dropout=0.1, batch_norm=True)")
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

    print(f"\n{div}\n")

    run_start = time.time()

    topo = resolve_topology()
    device = topo.device
    init_distributed(topo)   # no-op on a single worker; sets up the DDP process group otherwise

    # ---- Dummy batch for MAF shape inference -----------------------------
    # build_maf needs a representative (batch_x, batch_y) pair to compute output dims.
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
    # The Complex3DCNN and build_maf kwargs are captured verbatim into dicts so the
    # A5 estimator artifact can rebuild the (uncompiled) network under any torch.
    network_cfg = PARAMETERS.inference.network
    embedding_args = dict(
        n_frames=timing.frame_count,
        input_channels=network_cfg.input_channels,
        n_conv_layers=network_cfg.n_conv_layers,
        n_attn_layers=network_cfg.n_attn_layers,
        start_channels=network_cfg.start_channels,
        use_temporal_attention=network_cfg.use_temporal_attention,
        attention_heads=network_cfg.attention_heads,
        temporal_target_frames=network_cfg.temporal_target_frames,
        verbose=args.verbose,
    )
    embedding_net = torch.compile(Complex3DCNN(**embedding_args)).to(device)

    # ---- MAF posterior estimator -----------------------------------------
    maf_args = dict(
        z_score_x="structured",
        z_score_y="structured",
        dropout_probability=0.1,
        use_batch_norm=True,
    )
    posterior_estimator = build_maf(
        batch_x=theta_dummy,
        batch_y=video_dummy,
        embedding_net=embedding_net,
        **maf_args,
    ).to(device)

    # ---- Training pipeline -----------------------------------------------
    # The per-example test-loss distribution is recorded only when a TEST set is
    # present (it summarizes the per-epoch model-selection signal).
    use_tld = args.test_loss_distribution and args.test_tasks > 0
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
        test_loss_distribution=use_tld,
        paths=paths,
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

    # ---- New-best commit callback (rank 0; train_loop guards the call) ----
    # At each new best, write the A5 estimator + test-loss-distribution canonicals
    # (the live objects downstream stages + --resurrect read; always the current
    # best). Provenance backups (Epoch_{current}) are copied here only under
    # --backup-every-best; by default the single kept backup is written once at
    # finish, so a normal run leaves one backup, not one per improving epoch.
    commit_new_best = None
    if use_tld:
        tld_manifest = {
            "project_alias": paths.project_alias,
            "timing_label": timing_label,
            "test_set_id": f"TEST/{timing_label}",
            "theta_space": "log10",
            "theta_keys": imaging_keys,  # learnable imaging params = the theta columns, in order
            "parameter_table": [   # FULL Detector table (all value-based roles), declaration order
                {"key": p["KEY"],
                 "role": det.role_of(p),
                 "value": p["VALUE"],
                 "prior_range": p["PRIOR_RANGE"],
                 "log_flag": p["LOG_FLAG"], "log_base": p["LOG_BASE"]}
                for p in det.DETECTOR_PARAMETERIZATION_RAW],
            "train_videos": len(training_setup["train_loader"].dataset),
            "test_videos": len(training_setup["val_loader"].dataset),
            "epochs_planned": args.epochs,
            "torch_version": torch.__version__,
        }

        def commit_new_best(best_epoch, best_test_loss, gathered):
            task, sim, loss, theta = gathered
            best_metadata = {
                "timing_label": timing_label, "workflow": "detector",
                "best_test_loss": float(best_test_loss),
                "train_videos": tld_manifest["train_videos"],
                "test_videos": tld_manifest["test_videos"],
                "epochs": args.epochs,
            }
            estimator_path.parent.mkdir(parents=True, exist_ok=True)
            artifacts.save_estimator(
                training_setup["estimator"],
                embedding_args=embedding_args, maf_args=maf_args,
                theta_dim=theta_dim, video_shape=video_shape,
                parameter_keys=imaging_keys,
                prior_low=det.theta_lower_bound(), prior_high=det.theta_upper_bound(),
                path=estimator_path, metadata=best_metadata)
            snap = TestLossDistribution.from_epoch(
                best_epoch, task, sim, loss, theta,
                manifest=tld_manifest, best_test_loss=best_test_loss)
            snap.flush(tld_path)
            if args.backup_every_best:
                tv, ev = tld_manifest["train_videos"], tld_manifest["test_videos"]
                shutil.copy2(checkpoint_path, paths.backup_checkpoint_path(
                    data_bank_root, timing_label, tv, ev, best_epoch, best_test_loss))
                est_backup = estimator_path.with_name(
                    f"{estimator_path.stem}_{paths.backup_descriptor(tv, ev, best_epoch, best_test_loss)}"
                    f"{estimator_path.suffix}")
                shutil.copy2(estimator_path, est_backup)
                shutil.copy2(tld_path, paths.backup_test_loss_distribution_path(
                    data_bank_root, timing_label, tv, ev, best_epoch, best_test_loss))
                print(f"  [new best] committed live artifacts + per-epoch backups at "
                      f"epoch {best_epoch} (test loss {best_test_loss:.5f})", flush=True)
            else:
                print(f"  [new best] committed live artifacts at epoch {best_epoch} "
                      f"(test loss {best_test_loss:.5f}); backup deferred to finish",
                      flush=True)

    losses_train, losses_test, losses_replay, optimum_loss_test = train_loop(
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
        heartbeat_every=args.heartbeat,
        verbose=args.verbose,
        test_loss_distribution=use_tld,
        on_new_best=commit_new_best,
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
        # ---- A5 estimator artifact (the Detector's version-portable posterior) ------
        # Persist the selected estimator in the self-describing, torch-portable A5
        # format (compile-stripped state_dict + rebuild spec + metadata) — in place
        # of the version-locked canonical pickle. embedding_args / maf_args /
        # theta_dim / video_shape are the exact construction spec captured above; the
        # prior bounds come from the Detector imaging parameterization.
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

        # ---- Auto-backup: provenance-named copies of the just-saved best ------
        # A finished run overwrites the canonical checkpoint + A5 estimator (the
        # names every downstream stage loads). Copy them to backups whose filename
        # embeds this run's provenance -- train/test set sizes, epochs, and the
        # checkpoint's best TEST loss -- since the bare state_dict carries no such
        # metadata. The canonical files stay the live objects; a backup is
        # restored later by copying it back onto the canonical name. Keyed on the
        # TEST loss, so it is skipped for a run with no TEST set.
        if args.test_tasks > 0 and math.isfinite(optimum_loss_test):
            train_videos = len(training_setup["train_loader"].dataset)
            test_videos = len(training_setup["val_loader"].dataset)
            ckpt_backup = paths.backup_checkpoint_path(
                data_bank_root, timing_label, train_videos, test_videos,
                args.epochs, optimum_loss_test)
            shutil.copy2(checkpoint_path, ckpt_backup)
            print(f"Backup checkpoint saved to {ckpt_backup}")
            # Provenance-named backup of the A5 estimator artifact (same descriptor).
            estimator_backup = estimator_path.with_name(
                f"{estimator_path.stem}_{paths.backup_descriptor(train_videos, test_videos, args.epochs, optimum_loss_test)}{estimator_path.suffix}")
            shutil.copy2(estimator_path, estimator_backup)
            print(f"Backup estimator saved to {estimator_backup}")
            # Finish backup of the test-loss distribution, named with the total
            # epochs run (Epoch_{total}). By default this finish backup is the single
            # kept backup; under --backup-every-best it coexists with the per-epoch
            # new-best backups. The canonical was written at the last new best (or
            # by a prior resurrected run), so this copies it; skip if absent.
            if use_tld and tld_path.exists():
                tld_backup = paths.backup_test_loss_distribution_path(
                    data_bank_root, timing_label, train_videos, test_videos,
                    args.epochs, optimum_loss_test)
                shutil.copy2(tld_path, tld_backup)
                print(f"Backup test-loss distribution saved to {tld_backup}")

    # ---- Diagnostics: confirm outputs written, figure, summary ----------
    if reporter.enabled:
        reporter.check_file("checkpoint", checkpoint_path)
        reporter.check_file("estimator", estimator_path)
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
        "--test-loss-distribution", action=argparse.BooleanOptionalAction, default=True,
        help="Record the best epoch's per-example TEST-loss distribution (keyed by "
             "(task_index, sim_index)) as a self-describing .npz, committed with the "
             "posterior/checkpoint at each new best and backed up at finish. On by "
             "default when a TEST set is present; --no-test-loss-distribution disables it.",
    )
    parser.add_argument(
        "--backup-every-best", action="store_true",
        help="Write a provenance-named backup (checkpoint + A5 estimator + test-loss "
             "distribution, tagged with the epoch and TEST loss) at EVERY new-best "
             "epoch. Default off: only the single finish backup is kept, while the "
             "live canonical artifacts still update at each new best (crash-safe, "
             "resurrect-ready). Turn on for the full per-epoch backup history when "
             "debugging a training run.",
    )
    parser.add_argument(
        "--heartbeat", type=int, default=None,
        help="Within-epoch progress cadence: emit a line every N batches (rank 0 "
             "only). Default (unset) is ~4 lines/epoch; set a smaller N for finer "
             "progress on the long epochs of a production run.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration and inputs, print what would be read/written, then exit "
             "without running the stage (no GPU, no compute). Use before a queue submission or a long local run.",
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
    with console_log_context(cli_args, "Inference",
                             paths=det.detector_paths(PARAMETERS.paths)):
        main(cli_args)

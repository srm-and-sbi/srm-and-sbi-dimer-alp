"""Entry-point script: construct a DirectPosterior from an existing checkpoint.

Rebuilds the trained ``DirectPosterior`` pickle from a saved ``Optimum_ANN.pth``
estimator checkpoint *without any training*. A posterior is nothing more than a
``DirectPosterior(estimator, prior)`` pickled to disk; training only produces the
estimator weights. This script therefore performs the exact build-and-save
sequence the Inference stage runs after its training loop -- construct the
Complex3DCNN + MAF estimator, load the checkpoint weights into it, wrap it in a
DirectPosterior, and pickle it -- but skips the optimization entirely.

The estimator is constructed and ``torch.compile``d identically to the Inference
stage, for two reasons: the checkpoint's state-dict keys carry the compiled
``_orig_mod.`` prefixes and only match a compiled estimator on load; and the
downstream stages (Evaluation / Experiment) use the estimator exactly as loaded
-- they do not compile it themselves -- so handing them a compiled estimator is
what keeps their sampling fast. A single representative TRAIN batch is read only
to let ``build_maf`` infer its input/output dimensions; no training data is
otherwise consumed, and the ``build_maf`` forward is where the one-time compile
runs.

This is a single-process, single-GPU utility -- run it with plain ``python``, not
``torchrun``. It is useful for:

    * transferring trained weights between machines: copy the ``.pth`` (portable
      tensors) and construct the ``.pkl`` locally, rather than moving a pickle
      that embeds machine-specific device state;
    * recovering a posterior from a run that saved its checkpoint but terminated
      (e.g. a wall-time timeout) before reaching its own save block.

Inputs / outputs (the ``{timing_label}`` token, e.g. ``2S_50FPS``, is rendered
from ``PARAMETERS.simulation.timing`` by duration + fps):

    reads  <data_bank>/<labor_subdir>/<project_alias>_{timing_label}_Optimum_ANN.pth
        -- the trained estimator checkpoint (weights only). Override with
        ``--checkpoint`` to build from a specific file, e.g. a provenance-named
        backup ``..._Optimum_ANN_TRAIN+TEST_200K+50K_Epoch_25_TEST_LOSS_-17.05.pth``.
    writes <data_bank>/<posit_subdir>/<project_alias>_{timing_label}_Posterior.pkl
        -- the constructed DirectPosterior, ready for downstream sampling. When the
        source is a backup checkpoint the output name is derived to match
        (``Optimum_ANN`` -> ``Posterior``, ``.pth`` -> ``.pkl``), so the posterior
        tracks the weights it came from; override with ``--posterior``.

Usage:
    # canonical checkpoint -> canonical posterior:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_Construction_Optimum_ANN.py \\
        --total-time-seconds 2.0
    # a specific backup checkpoint -> its matching backup posterior:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_Construction_Optimum_ANN.py \\
        --total-time-seconds 2.0 \\
        --checkpoint SRM_AND_SBI_DIMER_ALP_2S_50FPS_Optimum_ANN_TRAIN+TEST_200K+50K_Epoch_25_TEST_LOSS_-17.05.pth

    On a Slurm cluster this runs ad hoc, outside the standard four-stage
    dispatcher -- in the weight-transfer or checkpoint-recovery situations it is
    the step that produces the posterior for the downstream Evaluation and
    Experiment stages. See the "Special-situation entry points" section of
    Script_Bank/HPC/README.md for the one-off sbatch recipe.
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

from srm_and_sbi_dimer_alp.inference_network import Complex3DCNN
from srm_and_sbi_dimer_alp.inference_support import (
    build_datasets,
    resolve_topology,
    save_posterior,
)
from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming, build_prior
from srm_and_sbi_dimer_alp.utils import console_log_context


def main(args: argparse.Namespace) -> None:
    """Construct and save the posterior from the saved checkpoint per the CLI args."""
    timing = RunTiming(
        total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing,
    )
    data_bank_root = PARAMETERS.machine.data_bank_root          # permanent tier: checkpoint + posterior
    train_data_root = PARAMETERS.machine.root_for("TRAIN")      # scratch tier: the representative batch
    compress = True  # video and theta sets are always read from .zarr in the inference stage

    # ---- Global RNG / precision settings ---------------------------------
    # The construction does not train; the seed only affects the (discarded)
    # representative-batch draw. Kept for parity with the Inference stage.
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.suppress_errors = True

    # ---- Pre-run banner ---------------------------------------------------
    machine = PARAMETERS.machine
    network_cfg = PARAMETERS.inference.network
    paths = PARAMETERS.paths
    div = "=" * 72

    timing_label = timing.label
    # Source checkpoint: an explicit --checkpoint (a backup, or a file copied in
    # for weight transfer; a bare filename resolves under Labor/), else canonical.
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.is_absolute():
            checkpoint_path = data_bank_root / paths.labor_subdir / checkpoint_path
    else:
        checkpoint_path = paths.checkpoint_path(data_bank_root, timing_label)   # source weights
    # Output posterior: an explicit --posterior, else derived from the checkpoint
    # name (canonical -> canonical; a descriptor-named backup -> the matching backup
    # posterior), so the .pkl always tracks the .pth it was built from.
    if args.posterior:
        posterior_path = Path(args.posterior)
        if not posterior_path.is_absolute():
            posterior_path = data_bank_root / paths.posit_subdir / posterior_path
    else:
        posterior_path = paths.posterior_path_for_checkpoint(checkpoint_path, data_bank_root)  # output

    print(div)
    print(f" {paths.project_alias} — Construction (posterior from checkpoint)")
    print(f" Started at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(div)

    print("\nMachine profile:")
    print(f"  name              : {machine.name}")
    print(f"  running_mode      : {machine.running_mode}")
    print(f"  compute_backend   : {machine.compute_backend}")
    if machine.gpu_device_index is not None:
        print(f"  gpu_device_index  : {machine.gpu_device_index}")

    print("\nRun configuration (CLI args):")
    print(f"  --total-time-seconds : {args.total_time_seconds}")
    print(f"  --seed               : {args.seed}")
    print(f"  --verbose            : {args.verbose}")

    print("\nVideo input shape:")
    print(f"  n_frames per video   : {timing.frame_count}        "
          f"(= {timing.total_time_seconds} / {timing.frame_time_seconds})")

    print("\nSource & destination:")
    print(f"  data_bank_root  : {data_bank_root}")
    print(f"  reads ckpt      : {checkpoint_path}")
    print(f"  writes posterior: {posterior_path}")

    # ---- Dry-run preview --------------------------------------------------
    # Validate configuration and report the resolved inputs, then exit before any
    # GPU use, dataset build, network compile, or checkpoint load. Only cheap
    # .exists() probes are performed. Mirrors the Inference dry-run contract.
    if args.dry_run:
        print(f"\n{div}")
        print(" [DRY RUN] no construction performed -- input validation only")
        print(div)
        missing = 0
        print(f"\n[DRY RUN] source checkpoint:")
        if checkpoint_path.exists():
            print(f"  reads Optimum_ANN checkpoint: {checkpoint_path}  [OK]")
        else:
            print(f"  reads Optimum_ANN checkpoint: {checkpoint_path}  [MISSING]")
            missing += 1
        # One representative TRAIN task supplies the batch build_maf needs for
        # shape inference (probing TASK_0 is sufficient to confirm the namespace).
        train_video = paths.video_set_path(0, train_data_root, timing_label, compress, "TRAIN")
        train_theta = paths.theta_set_path(0, train_data_root, timing_label, compress, "TRAIN")
        print("\n[DRY RUN] representative batch (TRAIN TASK_0, shape inference only):")
        for role, set_path in (("TRAIN video set", train_video),
                               ("TRAIN theta set", train_theta)):
            if Path(set_path).exists():
                print(f"  reads {role}: {set_path}  [OK]")
            else:
                print(f"  reads {role}: {set_path}  [MISSING]")
                missing += 1
        print()
        if missing:
            print(f"[DRY RUN] configuration validated; {missing} input(s) MISSING.")
        else:
            print("[DRY RUN] configuration validated; all inputs present.")
        print("[DRY RUN] no posterior constructed.")
        print(f"{div}\n")
        return

    run_start = time.time()

    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Construction reads an existing "
            f"Optimum_ANN checkpoint; run Inference first, or copy the checkpoint here."
        )

    # Single process, single GPU: resolve_topology returns a one-worker topology
    # under plain `python` (world_size 1, is_main True). No DDP init is needed.
    topo = resolve_topology()
    device = topo.device

    # ---- Representative batch for MAF shape inference --------------------
    # build_maf needs one (batch_x, batch_y) pair to compute input/output dims;
    # a single TRAIN task supplies it. No training data is otherwise consumed.
    print("\nReading one representative TRAIN batch (shape inference only)...", flush=True)
    dummy_train, _ = build_datasets(
        train_tasks=1, data_bank_root=train_data_root, timing_label=timing_label,
        compress=compress,
    )
    dummy_loader = DataLoader(dummy_train, batch_size=2, num_workers=0)
    video_dummy, theta_dummy = next(iter(dummy_loader))

    # ---- Embedding network (built + compiled exactly as Inference) ------
    embedding_net = Complex3DCNN(
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
    embedding_net = torch.compile(embedding_net).to(device)

    # ---- MAF estimator (built exactly as Inference) ---------------------
    # The build_maf forward through the embedding net is where the one-time
    # torch.compile runs -- the dominant cost of this utility.
    print("Constructing estimator (torch.compile runs on the first forward)...", flush=True)
    estimator = build_maf(
        batch_x=theta_dummy,
        batch_y=video_dummy,
        z_score_x="structured",
        z_score_y="structured",
        embedding_net=embedding_net,
        dropout_probability=0.1,
        use_batch_norm=True,
    ).to(device)

    # ---- Load the trained weights (no training) -------------------------
    # Staged to CPU, then placed onto the device by load_state_dict. A key/shape
    # mismatch here means the estimator architecture does not match the one that
    # produced the checkpoint (fail-loud; never a silent wrong load).
    print(f"Loading checkpoint weights: {checkpoint_path}", flush=True)
    state = torch.load(str(checkpoint_path), map_location="cpu", weights_only=True)
    try:
        estimator.load_state_dict(state)
    except RuntimeError as exc:
        raise RuntimeError(
            "Failed to load the checkpoint into the freshly-built estimator. This "
            "usually means the estimator architecture here does not match the one "
            "that produced the checkpoint (for example, a different sbi version). "
            "Construct the posterior on the machine that produced the checkpoint "
            f"instead, then copy the resulting .pkl.\nOriginal error: {exc}"
        ) from exc

    # ---- Build and save the posterior (main worker only) ----------------
    # Mirrors the Inference rank-0 save block: a CPU prior for the DirectPosterior,
    # then a device prior attached by save_posterior for downstream sampling.
    if topo.is_main:
        prior_cpu = build_prior(device="cpu")
        prior_cpu, _, _ = process_prior(prior_cpu)
        posterior = DirectPosterior(estimator, prior_cpu)

        prior_device = build_prior(device=str(device))
        posterior_path.parent.mkdir(parents=True, exist_ok=True)
        save_posterior(posterior, prior_device, posterior_path)
        print(f"\nPosterior constructed and saved to {posterior_path}")

    total_elapsed = time.time() - run_start
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")


def parse_args(argv=None) -> argparse.Namespace:
    """Construct the CLI parser and parse argv."""
    parser = argparse.ArgumentParser(
        description="Construct a DirectPosterior from an existing Optimum_ANN "
                    "checkpoint (no training).",
    )
    parser.add_argument(
        "--total-time-seconds", type=float,
        required=True,
        help="Video duration in seconds; determines n_frames for the network. "
             "Must match the value used to produce the checkpoint being loaded.",
    )
    parser.add_argument(
        "--checkpoint", type=str, default=None,
        help="Source Optimum_ANN checkpoint to build the posterior from. A bare "
             "filename resolves under <data_bank>/Labor/; an absolute path is used "
             "as-is. Default: the canonical "
             "<project_alias>_{timing_label}_Optimum_ANN.pth. Point this at a "
             "provenance-named backup to rebuild that specific posterior.",
    )
    parser.add_argument(
        "--posterior", type=str, default=None,
        help="Explicit output path for the constructed posterior (a bare filename "
             "resolves under <data_bank>/Posit/). Default: derived from the "
             "checkpoint name (Optimum_ANN -> Posterior, .pth -> .pkl), so a backup "
             "checkpoint yields the matching backup posterior.",
    )
    parser.add_argument(
        "--seed", type=lambda v: None if str(v).strip().lower() in ("none", "") else int(v), default=None,
        help="Master RNG seed (PyTorch + numpy + Python random). Default None "
             "-> non-deterministic. The construction does not train, so the seed "
             "only affects the (discarded) representative-batch draw.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration and inputs, print what would be read/written, "
             "then exit without GPU use or compute. Use before a queue submission.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print diagnostic info during the network build.",
    )
    parser.add_argument(
        "--debug-dump", action="store_true",
        help="Tee the console transcript to "
             "<data_bank>/Labor/Debug/<run_label>/Construction/console.log.",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    cli_args = parse_args(sys.argv[1:])
    with console_log_context(cli_args, "Construction"):
        main(cli_args)

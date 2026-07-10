"""Entry-point script (Detector workflow): render videos with imaging drawn from theta.

Special-situation entry point of the Detector calibration workflow
(DETECTOR_WORKFLOW.md §9.2, B2). Reads the diffusion-only trajectories produced
by the Detector RDS stage (B1), draws the imaging parameters per simulation from
the Detector prior (these are the calibration's inference target and the training
label), and renders each video with `render_detector_video`
(`detector_simulation_dli_support`, A3), which reuses the canonical DLI building
blocks but sources the imaging parameters from theta instead of the fixed table.
The canonical `simulate_dli` orchestrator is not called or modified.

The imaging theta is drawn from stream [1] of a two-stream `SeedSequence`
(stream [0] = the RDS nuisance drawn by B1), so the imaging theta is decorrelated
from the marginalized nuisance in the generated training set.

Detector data namespaces separately from canonical data by the `_DETECTOR`
runtime-prefix qualifier (`detector_parameterization.detector_paths`).

Outputs:
    <data_bank>/<video_subdir>/<alias>_DETECTOR_{timing_label}_Video_Set_TASK_{n}_{split}.{zarr|npy}
    <data_bank>/<theta_subdir>/<alias>_DETECTOR_{timing_label}_Theta_Set_TASK_{n}_{split}.{zarr|npy}
                                                            -- imaging-theta labels (the inference target)

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Simulation_DLI.py \\
        --total-time-seconds 5.0 --tasks 2 --task-simulations 5 --seed 42
    (add --dry-run to resolve config + inputs and print planned I/O without rendering)
"""

import argparse
import time
import warnings
from datetime import datetime, timezone

import numcodecs
import numpy as np
import readdy
import zarr

from srm_and_sbi_dimer_alp import detector_parameterization as det
from srm_and_sbi_dimer_alp.detector_simulation_dli_support import render_detector_video
from srm_and_sbi_dimer_alp.io import convert_video_dtype, save_theta_set, save_video_set
from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming
from srm_and_sbi_dimer_alp.simulation_rds_support import extract_trajectory_poses
from srm_and_sbi_dimer_alp.utils import SINK, SOCK

_DTYPE_FOR_BITS = {8: np.uint8, 16: np.uint16}


def main(args: argparse.Namespace) -> None:
    timing = RunTiming(total_time_seconds=args.total_time_seconds,
                       frames=PARAMETERS.simulation.timing)
    split = args.split.upper()
    data_bank_root = PARAMETERS.machine.root_for(split)
    compress = not args.no_compress
    output_fmt = "npy" if args.no_compress else "zarr"
    target_dtype = _DTYPE_FOR_BITS[args.video_dtype_bits]
    target_dtype_name = "uint8" if args.video_dtype_bits == 8 else "uint16"
    paths = det.detector_paths(PARAMETERS.paths)   # Detector-qualified prefix
    timing_label = timing.label
    root_size_px = PARAMETERS.simulation.stem.root_size_px
    div = "=" * 72

    imaging_keys = [e["KEY"] for e in det.DETECTOR_PARAMETERIZATION]
    ilow = np.array(det.theta_lower_bound())
    ihigh = np.array(det.theta_upper_bound())

    print(div)
    print(f" {paths.project_alias} — Detector Simulation_DLI (imaging from theta)")
    print(f" Started at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(div)
    print("\nMachine profile:")
    print(f"  name            : {PARAMETERS.machine.name}")
    print(f"  compute_backend : {PARAMETERS.machine.compute_backend}")
    print("\nRun configuration:")
    print(f"  --total-time-seconds : {args.total_time_seconds}  (frame_count={timing.frame_count})")
    if args.task_id is not None:
        print(f"  --task-id            : {args.task_id}  (one task)")
    else:
        print(f"  --tasks              : {args.tasks}  (tasks 0..{args.tasks - 1})")
    print(f"  --task-simulations   : {args.task_simulations}")
    print(f"  --split              : {split}")
    print(f"  --seed               : {args.seed}")
    print(f"  --no-compress        : {args.no_compress}  (.{output_fmt})")
    print(f"  --video-dtype-bits   : {args.video_dtype_bits}  ({target_dtype_name})")
    print("\nLearnable imaging theta (the inference target / training label):")
    for key, lo, hi in zip(imaging_keys, ilow, ihigh):
        print(f"  {key:<20}  log10 ∈ [{lo:+6.3f}, {hi:+6.3f}]   (10^ → physical)")
    print("\nInput / output destinations (Detector-namespaced):")
    print(f"  data_bank_root : {data_bank_root}")
    print(f"  trajectories   : <data_bank>/{paths.video_subdir}/{paths.trajectory_repo}/"
          f"{paths.project_alias}_{timing_label}_TASK_{{n}}/..._SIM_{{m}}.h5  (from B1)")
    print(f"  videos         : <data_bank>/{paths.video_subdir}/"
          f"{paths.project_alias}_{timing_label}_Video_Set_TASK_{{n}}_{split}.{output_fmt}")
    print(f"  imaging thetas : <data_bank>/{paths.theta_subdir}/"
          f"{paths.project_alias}_{timing_label}_Theta_Set_TASK_{{n}}_{split}.{output_fmt}")
    print(f"\n{div}\n")

    task_indices = [args.task_id] if args.task_id is not None else list(range(args.tasks))
    n_rows = max(task_indices) + 1

    # Imaging theta = stream [1] of the two-stream SeedSequence (stream [0] is the
    # RDS nuisance in B1) -> decorrelated from the nuisance.
    imaging_rng = np.random.default_rng(np.random.SeedSequence(args.seed).spawn(2)[1])
    imaging_log10 = imaging_rng.uniform(low=ilow, high=ihigh,
                                        size=(n_rows, args.task_simulations, len(ilow)))
    imaging_sets = np.power(10, imaging_log10)   # physical (the labels)

    if args.dry_run:
        print("DRY RUN — configuration + inputs resolved; no videos rendered.")
        print(f"  would render {len(task_indices)} task(s) × {args.task_simulations} sim(s) "
              f"= {len(task_indices) * args.task_simulations} videos of shape "
              f"({timing.frame_count}, {root_size_px}, {root_size_px})")
        print(f"  imaging-theta labels shape : {imaging_sets.shape}  (n_rows, sims, {len(ilow)})")
        print(f"  example imaging[task {task_indices[0]}, sim 0] (physical): "
              f"{dict(zip(imaging_keys, np.round(imaging_sets[task_indices[0], 0], 4).tolist()))}")
        return

    run_start = time.time()
    for loop_i, task in enumerate(task_indices):
        print(f"{SOCK} Task {task}  ({loop_i + 1}|{len(task_indices)}) {SOCK}")

        # Persist the imaging-theta labels for this task (the inference target).
        theta_path = paths.theta_set_path(task, data_bank_root, timing_label, compress, split)
        theta_path.parent.mkdir(parents=True, exist_ok=True)
        theta_data = imaging_sets[task]                       # (task_simulations, n_imaging)
        if compress:
            compressor = numcodecs.Blosc(cname="zstd", clevel=9,
                                         shuffle=numcodecs.Blosc.BITSHUFFLE)
            store = zarr.open(store=str(theta_path), mode="w", shape=theta_data.shape,
                              chunks=(1, theta_data.shape[1]), dtype=np.float64,
                              compressor=compressor)
            store[:, :] = theta_data
        else:
            save_theta_set(theta_path, theta_data, compress=False)

        # Video-set store.
        video_set_path = paths.video_set_path(task, data_bank_root, timing_label, compress, split)
        video_set_path.parent.mkdir(parents=True, exist_ok=True)
        video_set_shape = (args.task_simulations, timing.frame_count, root_size_px, root_size_px)
        if compress:
            compressor = numcodecs.Blosc(cname="zstd", clevel=9,
                                         shuffle=numcodecs.Blosc.BITSHUFFLE)
            video_store = zarr.open(store=str(video_set_path), mode="w", shape=video_set_shape,
                                    chunks=(1, video_set_shape[1], video_set_shape[2], video_set_shape[3]),
                                    dtype=target_dtype, compressor=compressor)
        else:
            video_store = np.zeros(video_set_shape, dtype=target_dtype)

        for sim in range(args.task_simulations):
            sim_start = time.time()
            traj_path = paths.trajectory_path(task, sim, data_bank_root, timing_label, split)
            tray = readdy.Trajectory(filename=str(traj_path))
            tray_poses, dimer_mask = extract_trajectory_poses(
                tray, return_dimer_mask=True, verbose=args.verbose)
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message="All-NaN slice encountered",
                                        category=RuntimeWarning)
                pro_tray_poses = np.nanmax(a=tray_poses, axis=3)
            if pro_tray_poses.shape[0] != timing.frame_count:
                raise ValueError(
                    f"Trajectory {traj_path} holds {pro_tray_poses.shape[0]} frames but this run "
                    f"declares {timing.frame_count} (--total-time-seconds {args.total_time_seconds}). "
                    f"Refusing to render a duration-mismatched video.")

            frames = render_detector_video(
                pro_tray_poses=pro_tray_poses,
                imaging_physical=theta_data[sim],
                dimer_mask=dimer_mask,
                seed=args.seed,
                verbose=args.verbose)
            video = np.moveaxis(frames, 2, 0)     # frame axis last -> first (storage convention)
            video = convert_video_dtype(video, bits_from=16, bits_to=args.video_dtype_bits)
            video_store[sim] = video
            print(f"{SINK} sim {sim + 1}|{args.task_simulations} rendered "
                  f"({time.time() - sim_start:.1f}s) → {video_set_path.name}[{sim}] {SINK}", flush=True)

        if not compress:
            save_video_set(video_set_path, video_store, compress=False)

    print(f"\n{div}\n Detector DLI rendering done in {time.time() - run_start:.1f}s\n{div}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detector workflow: render videos from diffusion-only trajectories, "
                    "with imaging parameters drawn from the Detector prior (the inference target).")
    p.add_argument("--total-time-seconds", type=float, required=True)
    group = p.add_mutually_exclusive_group()
    group.add_argument("--tasks", type=int, default=1)
    group.add_argument("--task-id", type=int, default=None)
    p.add_argument("--task-simulations", type=int, default=1)
    p.add_argument("--split", type=str, default="TRAIN",
                   choices=["TRAIN", "TEST", "EVAL", "train", "test", "eval"])
    p.add_argument("--seed", type=int, default=None,
                   help="Base seed; imaging theta uses stream [1] of SeedSequence(seed).spawn(2).")
    p.add_argument("--no-compress", action="store_true")
    p.add_argument("--video-dtype-bits", type=int, default=16, choices=[8, 16],
                   help="Output bit depth for video pixels (8=uint8, 16=uint16).")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve config + inputs and print planned I/O without rendering.")
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())

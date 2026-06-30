"""Entry-point script: render diffraction-limited videos from RDS trajectories.

Reads each .h5 trajectory file produced by the RDS stage, extracts particle
poses + dimer state mask, renders the synthetic fluorescence video via the
DLI pipeline (PSF + Poisson + EMCCD noise), and saves the resulting videos
as a .zarr (compressed) or .npy (uncompressed) video set per task.

Outputs (the ``{timing_label}`` token, e.g. ``2S_50FPS``, is rendered from
``PARAMETERS.simulation.timing.label`` to namespace files by duration + fps):

    <data_bank>/<video_subdir>/<project_alias>_{timing_label}_Video_Set_TASK_{n}.zarr
        (or .npy if --no-compress)

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_Simulation_DLI.py \\
        --total-time-seconds 2.0 --tasks 2 --task-simulations 5 \\
        --video-dtype-bits 8 --seed 42

Diagnostics:
    --probe logs the process resource limits (RLIMIT_NPROC / RLIMIT_NOFILE) at
    startup and a per-simulation line (thread count, open file descriptors,
    resident memory). Logging only; the same instrumentation as the RDS stage,
    shared via utils.log_resource_limits / utils.probe_resources.
"""

import argparse
import sys
import time
import warnings
from datetime import datetime, timezone

import numcodecs
import numpy as np
import readdy
import zarr

from srm_and_sbi_dimer_alp.diagnostics import (
    DiagnosticReporter,
    fixed_parameters_table,
    prior_sampling_table,
)
from srm_and_sbi_dimer_alp.io import convert_video_dtype, load_data, save_video_set
from srm_and_sbi_dimer_alp.parameterization import (
    PARAMETER_RAW_FIND,
    PARAMETERIZATION,
    PARAMETERIZATION_RAW,
    PARAMETERS,
    RunTiming,
)
from srm_and_sbi_dimer_alp.simulation_dli_support import simulate_dli
from srm_and_sbi_dimer_alp.simulation_rds_support import extract_trajectory_poses
from srm_and_sbi_dimer_alp.utils import (
    SINK, SOCK, console_log_context, log_memory_state, probe_resources,
    log_resource_limits)


_DTYPE_FOR_BITS = {8: np.uint8, 16: np.uint16}

# Grouped DLI known-parameter spec for the --verbose banner block. KEYs match
# entries in PARAMETERIZATION_RAW; the group labels are display-only.
_DLI_PARAM_GROUPS = {
    "PSF (Point Spread Function)": ["mu_r", "sigma_r", "mu_pc", "sigma_pc"],
    "EMCCD camera": ["kappa_c", "kappa_o", "kappa_g", "kappa_v"],
    "State machine (brightness transitivity)": [
        "brightness_quantile", "delta_frame", "prob_photo_bleach",
        "numb_photo_bleach", "lambda_rate", "gamma_penalty",
    ],
}


def main(args: argparse.Namespace) -> None:
    """Run the full DLI rendering pipeline per the CLI args."""
    timing = RunTiming(
        total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing,
    )
    split = args.split.upper()   # "TRAIN" / "TEST" / "EVAL" namespace suffix
    data_bank_root = PARAMETERS.machine.root_for(split)  # TRAIN/TEST -> scratch tier; EVAL -> permanent (single-tier machines: always data_bank_root)
    compress = not args.no_compress
    target_dtype = _DTYPE_FOR_BITS[args.video_dtype_bits]
    root_size_px = PARAMETERS.simulation.stem.root_size_px
    output_fmt = "npy" if args.no_compress else "zarr"
    target_dtype_name = "uint8" if args.video_dtype_bits == 8 else "uint16"

    # ---- Pre-run banner -------------------------------------------------
    machine = PARAMETERS.machine
    geom = PARAMETERS.simulation.stem
    rds_cfg = PARAMETERS.simulation.rds
    dli_cfg = PARAMETERS.simulation.dli
    paths = PARAMETERS.paths
    div = "=" * 72

    print(div)
    print(f" {paths.project_alias} — Simulation_DLI")
    print(f" Started at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(div)

    if args.probe:
        log_resource_limits()

    print("\nMachine profile:")
    print(f"  name              : {machine.name}")
    print(f"  running_mode      : {machine.running_mode}")
    print(f"  compute_backend   : {machine.compute_backend}")
    if machine.gpu_device_index is not None:
        print(f"  gpu_device_index  : {machine.gpu_device_index}")
    print(f"  num_workers       : {machine.num_workers}")

    print("\nRun configuration (CLI args):")
    print(f"  --total-time-seconds : {args.total_time_seconds}")
    if args.task_id is not None:
        print(f"  --task-id            : {args.task_id}        (this run generates exactly one task)")
    else:
        print(f"  --tasks              : {args.tasks}        (this run generates tasks 0..{args.tasks - 1})")
    print(f"  --task-simulations   : {args.task_simulations}")
    print(f"  --split              : {args.split}   (namespace suffix: _{split})")
    print(f"  --video-dtype-bits   : {args.video_dtype_bits}          (output dtype: {target_dtype_name})")
    print(f"  --seed               : {args.seed}")
    print(f"  --no-compress        : {args.no_compress}  (output format: .{output_fmt})")
    print(f"  --verbose            : {args.verbose}")
    print(f"  --show               : {args.show}")

    print("\nSimulation timing (derived):")
    print(f"  frame_time_seconds       : {timing.frame_time_seconds}       "
          f"({1 / timing.frame_time_seconds:.0f} Hz)")
    print(f"  frame_count              : {timing.frame_count}        "
          f"(= {timing.total_time_seconds} / {timing.frame_time_seconds})")
    print(f"  steps_per_frame          : {timing.steps_per_frame}")
    print(f"  total_steps              : {timing.total_steps}")
    print(f"  delta_time_nanoseconds   : {timing.delta_time_nanoseconds}")

    print("\nSystem geometry:")
    print(f"  pixel_size_nm        : {geom.pixel_size_nm}")
    print(f"  root_size_px         : {geom.root_size_px}        "
          f"(image: {geom.root_size_px} × {geom.root_size_px})")
    print(f"  box_size_nm          : {geom.box_size}")
    print(f"  particle_diameter_nm : {geom.particle_diameter_nm}")

    print("\nParticle species:")
    print(f"  {rds_cfg.particle_species_names}   "
          f"A=Monomer, B=Mobile Dimer, C=Immobile Dimer")

    print("\nDLI runtime defaults:")
    print(f"  dimer_mule              : {dli_cfg.dimer_mule}   "
          f"(√2; brightness boost for dimers — see PROJECT_CONTEXT.md §4)")
    print(f"  darkcounts              : {dli_cfg.darkcounts}                    "
          f"(baseline; no per-pixel dark current)")
    print(f"  sqrt_2sigma_dist_label  : {dli_cfg.sqrt_2sigma_dist_label}            "
          f"(PSF width sampling distribution)")

    timing_label = timing.label
    print("\nOutput destinations:")
    print(f"  data_bank_root      : {data_bank_root}")
    print(f"  reads theta sets    : <data_bank>/{paths.theta_subdir}/"
          f"{paths.project_alias}_{timing_label}_Theta_Set_TASK_{{n}}.{output_fmt}")
    print(f"  reads trajectories  : <data_bank>/{paths.video_subdir}/"
          f"{paths.trajectory_repo}/{paths.project_alias}_{timing_label}_TASK_{{n}}/"
          f"{paths.project_alias}_{timing_label}_TASK_{{n}}_SIM_{{m}}.h5")
    print(f"  writes video sets   : <data_bank>/{paths.video_subdir}/"
          f"{paths.project_alias}_{timing_label}_Video_Set_TASK_{{n}}.{output_fmt}   "
          f"(dtype: {target_dtype_name})")

    if args.verbose:
        print("\nKnown DLI parameters:")
        for group_label, keys in _DLI_PARAM_GROUPS.items():
            print(f"  --- {group_label} ---")
            for key in keys:
                para = PARAMETERIZATION_RAW[PARAMETER_RAW_FIND[key]]
                val = para["VALUE"]
                unit = para["UNIT"]
                unit_str = f"        units: {unit}" if unit else ""
                print(f"  {key:<24}  : {val}{unit_str}")

    print(f"\n{div}\n")

    run_start = time.time()

    # --task-id K renders exactly one task (HPC array fan-out: one task per job),
    # reading that task's RDS trajectories + theta set; otherwise all of --tasks.
    task_indices = [args.task_id] if args.task_id is not None else list(range(args.tasks))

    for loop_i, task in enumerate(task_indices):
        task_alias = task
        print(f"{SOCK} Task {task_alias}  ({loop_i + 1}|{len(task_indices)}) {SOCK}")

        if args.verbose:
            log_memory_state()

        # ---- Load the theta set written by the RDS stage --------------
        theta_set_path = PARAMETERS.paths.theta_set_path(
            task_alias, data_bank_root, timing_label, compress, split)
        print(f"  Reading theta set:  {theta_set_path}")
        theta_set = load_data(theta_set_path)

        # ---- Create the video-set store/buffer ------------------------
        video_set_path = PARAMETERS.paths.video_set_path(
            task_alias, data_bank_root, timing_label, compress, split)
        video_set_path.parent.mkdir(parents=True, exist_ok=True)
        video_set_shape = (
            args.task_simulations,
            timing.frame_count,
            root_size_px,
            root_size_px,
        )
        if compress:
            compressor = numcodecs.Blosc(
                cname="zstd", clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE,
            )
            video_store = zarr.open(
                store=str(video_set_path),
                mode="w",
                shape=video_set_shape,
                chunks=(1, video_set_shape[1], video_set_shape[2], video_set_shape[3]),
                dtype=target_dtype,
                compressor=compressor,
            )
        else:
            video_store = np.zeros(video_set_shape, dtype=target_dtype)

        # ---- Diagnostics reporter (debug mode) ------------------------
        reporter = DiagnosticReporter(
            stage="DLI",
            enabled=args.debug or args.debug_dump,
            dump=args.debug_dump,
            dump_dir=(paths.debug_run_dir(data_bank_root, timing_label, "DLI", split)
                      / f"TASK_{task_alias}"),
            run_label=f"{paths.project_alias}_{timing_label}_TASK_{task_alias}_{split}",
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
        dtype_max = int(np.iinfo(target_dtype).max)

        # ---- Process each simulation ----------------------------------
        last_frames = None
        for sim in range(args.task_simulations):
            sim_start = time.time()
            traj_path = PARAMETERS.paths.trajectory_path(
                task_alias, sim, data_bank_root, timing_label, split)
            print(f"  Reading trajectory: {traj_path}")
            tray = readdy.Trajectory(filename=str(traj_path))
            tray_poses, dimer_mask = extract_trajectory_poses(
                tray, return_dimer_mask=True, verbose=args.verbose,
            )
            # Guard: the trajectory's own frame count must match this run's declared
            # duration. A mismatch means the trajectory was generated at a different
            # --total-time-seconds than the run claims, so the rendered video would be
            # mislabeled/wrong. Fail loudly rather than emit it.
            if tray_poses.shape[0] != timing.frame_count:
                raise ValueError(
                    f"Trajectory {traj_path} holds {tray_poses.shape[0]} frames but this run "
                    f"declares {timing.frame_count} frames (--total-time-seconds "
                    f"{args.total_time_seconds}). Refusing to render a duration-mismatched video."
                )
            # Collapse species-rank axis: each particle's (x,y,z) coord at each frame
            # (NaN for absent particles, taken as max-over-rank since each particle is in
            # exactly one species per frame so only one rank has a non-NaN entry).
            # Suppress the benign all-NaN-slice warning that nanmax emits whenever a
            # (frame, particle) slot has no species present in any rank.
            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="All-NaN slice encountered",
                    category=RuntimeWarning,
                )
                pro_tray_poses = np.nanmax(a=tray_poses, axis=3)

            theta = theta_set[sim, :]
            frames = simulate_dli(
                pro_tray_poses=pro_tray_poses,
                theta=theta,
                dimer_mask=dimer_mask,
                dimer_mule=PARAMETERS.simulation.dli.dimer_mule,
                seed=args.seed,   # default None -> non-deterministic
                verbose=args.verbose,
            )
            # Move frame axis from last (DLI output) to first (storage convention).
            video = np.moveaxis(frames, 2, 0)
            video = convert_video_dtype(video, bits_from=16, bits_to=args.video_dtype_bits)
            video_store[sim] = video
            last_frames = frames

            # ---- Sim-0 diagnostics (debug mode) -----------------------
            # Detailed checkpoints/checks/figures on the first simulation only,
            # to keep the per-task report clean. NaN is EXPECTED in tray_poses
            # (absent particles), so no-NaN is asserted only on the rendered
            # output, never on the trajectory poses.
            if reporter.enabled and sim == 0:
                reporter.checkpoint(
                    "DLI render (sim 0)",
                    tray_poses=tray_poses,
                    dimer_mask=dimer_mask,
                    pro_tray_poses=pro_tray_poses,
                    frames=frames,
                    video=video,
                )
                reporter.check_no_nan_inf("frames", frames)
                reporter.check(
                    "video_has_signal", bool(video.max() > 0),
                    f"max={int(video.max())}",
                    note="the rendered video is not blank -- at least one pixel "
                         "carries signal.",
                )
                reporter.check(
                    "video_within_dtype_max", int(video.max()) <= dtype_max,
                    f"max={int(video.max())} (dtype max {dtype_max})",
                    note="no pixel exceeds the storage dtype's maximum (no "
                         "overflow/clipping bug).",
                )
                reporter.stat(
                    "particles", int(tray_poses.shape[1]),
                    note="number of particle tracks across the trajectory "
                         "(all species combined).",
                )
                reporter.stat(
                    "frame_count", int(video.shape[0]),
                    expected=str(timing.frame_count),
                    note="frames in the rendered video; must equal "
                         "duration x frame rate.",
                )
                reporter.stat(
                    "pixel_max", int(video.max()), expected=f"<= {dtype_max}",
                    note="brightest pixel value (ADU); should sit below the "
                         "dtype ceiling.",
                )
                reporter.stat(
                    "pixel_mean", float(video.mean()),
                    note="mean pixel value across the video (ADU); reflects "
                         "overall brightness plus the camera noise floor.",
                )
                reporter.stat(
                    "nonzero_pixel_fraction", float((video > 0).mean()),
                    note="fraction of pixels with any signal; near 1.0 is "
                         "expected because the EMCCD noise floor lights every pixel.",
                )
                # Prior bounds + sampled values that generated THIS video, so the
                # parameters sit next to the rendered frame for direct checking.
                headers, prior_rows = prior_sampling_table(PARAMETERIZATION, theta)
                reporter.table(
                    "Parameters of this video (sim 0)", headers, prior_rows,
                    note="The prior bounds and sampled values behind this video. "
                         "count_alp/bet/chi are the initial A/B/C particle counts -- "
                         "compare against the spots visible in the sample frame.",
                )
                fixed_headers, fixed_rows = fixed_parameters_table(PARAMETERIZATION_RAW)
                reporter.table(
                    "Fixed parameters", fixed_headers, fixed_rows,
                    note="Non-learnable parameters held constant across all "
                         "simulations (Known scientific constants + tuning "
                         "Hyperparameters): camera, PSF, photophysics, capture "
                         "radius, brightness quantiles.",
                )
                if reporter.dump:
                    from srm_and_sbi_dimer_alp.visualization_dli import (
                        figure_pixel_histogram,
                        figure_sample_frame,
                    )
                    reporter.save_figure(
                        "sample_frame", figure_sample_frame(video),
                        caption="A single rendered frame (middle of the video): "
                                "fluorescent spots over the EMCCD noise background.",
                    )
                    reporter.save_figure(
                        "pixel_histogram", figure_pixel_histogram(video),
                        caption="Distribution of non-zero pixel values (log "
                                "count). The bulk near zero is the noise floor; "
                                "the tail is emitter signal.",
                    )

            sim_elapsed = time.time() - sim_start
            print(f"{SINK} Simulation {sim + 1}|{args.task_simulations} done  "
                  f"(elapsed: {sim_elapsed:.1f}s) {SINK}")

            # ---- Optional debug probe (per-sim resource use) -------------
            if args.probe:
                _th, _fd, _rss = probe_resources()
                print(f"[probe] sim {sim + 1}: threads={_th} fds={_fd} "
                      f"rss_mb={_rss}", flush=True)

        # ---- Save uncompressed buffer if not using .zarr ---------------
        if not compress:
            save_video_set(video_set_path, video_store, compress=False)

        # ---- Diagnostics: confirm output written, summarize -----------
        reporter.check_file("video_set", video_set_path)
        reporter.summary()
        reporter.write_report()

        # ---- Optional diagnostic plot of the last simulation's frames -
        if args.show and last_frames is not None:
            # Imported lazily so HPC runs don't pay the matplotlib cost.
            from srm_and_sbi_dimer_alp.visualization_dli import extract_pixel_stats
            extract_pixel_stats(last_frames)

    total_elapsed = time.time() - run_start
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")


def parse_args(argv=None) -> argparse.Namespace:
    """Construct the CLI parser and parse argv."""
    parser = argparse.ArgumentParser(
        description="Render diffraction-limited videos from RDS trajectories.",
    )
    parser.add_argument(
        "--total-time-seconds", type=float,
        required=True,
        help="Simulation duration per video in seconds (required; e.g. 2.0, 5.0). "
             "Must match the value used in the corresponding RDS run.",
    )
    parser.add_argument(
        "--tasks", type=int, default=2,
        help="Number of theta-set / video-set tasks to process (default: 2). "
             "Ignored when --task-id is given.",
    )
    parser.add_argument(
        "--task-id", type=int, default=None,
        help="Render exactly one task with this index (for HPC array fan-out: one "
             "task per job), reading that task's RDS trajectories + theta set. "
             "Overrides --tasks.",
    )
    parser.add_argument(
        "--task-simulations", type=int, default=5,
        help="Number of simulations per task (default: 5).",
    )
    parser.add_argument(
        "--split", choices=["train", "test", "eval"], default="train",
        help="Dataset role to render, read from / written to its own namespace "
             "(filename suffix _TRAIN / _TEST / _EVAL). Must match the RDS run. "
             "Default: train.",
    )
    parser.add_argument(
        "--video-dtype-bits", type=int, default=8, choices=[8, 16],
        help="Output bit depth for video pixels: 8 (uint8) or 16 (uint16). "
             "Default 8.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="RNG seed for PSF widths, brightness states, and noise. "
             "Default: None (non-deterministic).",
    )
    parser.add_argument(
        "--no-compress", action="store_true",
        help="Save video sets as .npy instead of compressed .zarr.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print diagnostic info during setup (CTMC/DTMC matrices, "
             "trajectory shapes).",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="After processing each task, show pixel-value statistics for the "
             "last simulation's frames (interactive matplotlib).",
    )
    parser.add_argument(
        "--probe", action="store_true",
        help="Debug instrumentation: log RLIMIT_NPROC/NOFILE at startup and "
             "threads/open-fds/RSS after each simulation (logging only; no behavior "
             "change).",
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
    with console_log_context(cli_args, "DLI"):
        main(cli_args)

"""Entry-point script (Detector workflow): generate diffusion-only trajectories.

Part of the Detector calibration workflow (DETECTOR_WORKFLOW.md §9.2, B1) — a
complete calibration workflow parallel to the canonical pipeline, with its own
committed submission machinery.

Mirrors the canonical `SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py`, with two
differences: it draws the reaction-diffusion parameters from the Detector RDS
*nuisance* (three counts + three diffusion coefficients) rather than the full
learnable prior, and it builds the ReaDDy system in diffusion-only mode
(`build_system(pure_diffusion=True)` via `detector_simulation_rds_support`).
The imaging parameters — the Detector's actual inference target — are drawn and
rendered downstream by the Detector DLI stage (B2). Reaction rates are never
used here (the reaction registrations are skipped).

Detector data namespaces separately from canonical data by the `_DETECTOR`
runtime-prefix qualifier (see `detector_parameterization.detector_paths`), so
these trajectories never collide with the canonical ones.

Outputs (the `{timing_label}` token, e.g. `5S_50FPS`, namespaces by duration+fps):
    <data_bank>/<video_subdir>/<trajectory_repo>/<alias>_DETECTOR_{timing_label}_TASK_{n}/
        <alias>_DETECTOR_{timing_label}_TASK_{n}_SIM_{m}.h5   -- per-simulation trajectory
    <data_bank>/<theta_subdir>/
        <alias>_DETECTOR_{timing_label}_Nuisance_RDS_Set_TASK_{n}.{zarr|npy}
                                                              -- per-task RDS-nuisance provenance

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_DETECTOR_Simulation_RDS.py \\
        --total-time-seconds 5.0 --tasks 2 --task-simulations 5 --seed 42
    (add --dry-run to resolve config + print planned I/O without running ReaDDy)
"""

import argparse
import gc
import time
from datetime import datetime, timezone

import numcodecs
import numpy as np
import readdy
import zarr

from srm_and_sbi_dimer_alp import detector_parameterization as det
from srm_and_sbi_dimer_alp.detector_simulation_rds_support import build_detector_rds_simulation
from srm_and_sbi_dimer_alp.io import save_theta_set
from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming
from srm_and_sbi_dimer_alp.utils import SINK, SOCK


def _nuisance_set_path(paths, task_alias, data_bank_root, timing_label, compress, split):
    """Detector RDS-nuisance provenance file: the Detector-aliased theta-set path
    with the object token swapped `Theta_Set` -> `Nuisance_RDS_Set` (reuses the
    canonical path pattern; no new path code)."""
    base = paths.theta_set_path(task_alias, data_bank_root, timing_label, compress, split)
    return base.with_name(base.name.replace("Theta_Set", "Nuisance_RDS_Set"))


def main(args: argparse.Namespace) -> None:
    timing = RunTiming(total_time_seconds=args.total_time_seconds,
                       frames=PARAMETERS.simulation.timing)
    split = args.split.upper()
    data_bank_root = PARAMETERS.machine.root_for(split)
    compress = not args.no_compress
    output_fmt = "npy" if args.no_compress else "zarr"
    paths = det.detector_paths(PARAMETERS.paths)   # Detector-qualified prefix
    timing_label = timing.label
    div = "=" * 72

    nuisance_keys = [e["KEY"] for e in det.DETECTOR_NUISANCE]
    nlow = np.array(det.nuisance_lower_bound())
    nhigh = np.array(det.nuisance_upper_bound())

    print(div)
    print(f" {paths.project_alias} — Detector Simulation_RDS (diffusion-only)")
    print(f" Started at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(div)
    print("\nMachine profile:")
    print(f"  name            : {PARAMETERS.machine.name}")
    print(f"  running_mode    : {PARAMETERS.machine.running_mode}")
    print(f"  compute_backend : {PARAMETERS.machine.compute_backend}")
    print("\nRun configuration:")
    print(f"  --total-time-seconds : {args.total_time_seconds}  (frame_count={timing.frame_count}, "
          f"total_steps={timing.total_steps})")
    if args.task_id is not None:
        print(f"  --task-id            : {args.task_id}  (one task)")
    else:
        print(f"  --tasks              : {args.tasks}  (tasks 0..{args.tasks - 1})")
    print(f"  --task-simulations   : {args.task_simulations}")
    print(f"  --split              : {split}")
    print(f"  --seed               : {args.seed}")
    print(f"  --no-compress        : {args.no_compress}  (.{output_fmt})")
    print("\nRDS nuisance (physics frozen to pure diffusion; marginalized):")
    for key, lo, hi in zip(nuisance_keys, nlow, nhigh):
        print(f"  {key:<28}  log10 ∈ [{lo:+6.3f}, {hi:+6.3f}]   (10^ → physical)")
    print("\nOutput destinations (Detector-namespaced):")
    print(f"  data_bank_root : {data_bank_root}")
    print(f"  trajectories   : <data_bank>/{paths.video_subdir}/{paths.trajectory_repo}/"
          f"{paths.project_alias}_{timing_label}_TASK_{{n}}/..._SIM_{{m}}.h5")
    print(f"  nuisance sets  : <data_bank>/{paths.theta_subdir}/"
          f"{paths.project_alias}_{timing_label}_Nuisance_RDS_Set_TASK_{{n}}.{output_fmt}")
    print(f"\n{div}\n")

    task_indices = [args.task_id] if args.task_id is not None else list(range(args.tasks))
    n_rows = max(task_indices) + 1

    # Two independent streams from one base seed: stream [0] = RDS nuisance (this
    # stage), stream [1] = imaging theta (Detector DLI stage, B2). Spawning keeps
    # the marginalized nuisance decorrelated from the imaging theta in the
    # generated training set (so the estimator can't learn a nuisance shortcut).
    nuisance_rng = np.random.default_rng(np.random.SeedSequence(args.seed).spawn(2)[0])
    nuisance_log10 = nuisance_rng.uniform(low=nlow, high=nhigh,
                                          size=(n_rows, args.task_simulations, len(nlow)))
    nuisance_sets = np.power(10, nuisance_log10)   # physical

    if args.dry_run:
        print("DRY RUN — configuration + inputs resolved; no ReaDDy simulations executed.")
        print(f"  would generate {len(task_indices)} task(s) × {args.task_simulations} sim(s) "
              f"= {len(task_indices) * args.task_simulations} diffusion-only trajectories")
        print(f"  nuisance draw shape : {nuisance_sets.shape}  (n_rows, sims, {len(nlow)} params)")
        print(f"  example nuisance[task {task_indices[0]}, sim 0] (physical): "
              f"{dict(zip(nuisance_keys, np.round(nuisance_sets[task_indices[0], 0], 4).tolist()))}")
        return

    run_start = time.time()
    for loop_i, task in enumerate(task_indices):
        print(f"{SOCK} Task {task}  ({loop_i + 1}|{len(task_indices)}) {SOCK}")
        traj_dir = paths.trajectory_dir(task, data_bank_root, timing_label, split)
        traj_dir.mkdir(parents=True, exist_ok=True)

        # Persist the per-task RDS-nuisance draw (provenance; not the training label).
        nuis_path = _nuisance_set_path(paths, task, data_bank_root, timing_label, compress, split)
        nuis_path.parent.mkdir(parents=True, exist_ok=True)
        nuis_data = nuisance_sets[task]                       # (task_simulations, n_params)
        if compress:
            compressor = numcodecs.Blosc(cname="zstd", clevel=9,
                                         shuffle=numcodecs.Blosc.BITSHUFFLE)
            store = zarr.open(store=str(nuis_path), mode="w", shape=nuis_data.shape,
                              chunks=(1, nuis_data.shape[1]), dtype=np.float64,
                              compressor=compressor)
            store[:, :] = nuis_data
        else:
            save_theta_set(nuis_path, nuis_data, compress=False)

        for sim in range(args.task_simulations):
            sim_start = time.time()
            smut, _theta = build_detector_rds_simulation(
                nuis_data[sim], seed=args.seed, verbose=args.verbose)
            traj_path = paths.trajectory_path(task, sim, data_bank_root, timing_label, split)
            if traj_path.exists():
                traj_path.unlink()
            smut.output_file = str(traj_path)
            smut.progress_output_stride = timing.total_steps
            smut.run(n_steps=timing.total_steps,
                     timestep=timing.delta_time_nanoseconds * readdy.units.nanosecond,
                     show_summary=False)
            del smut
            gc.collect()
            print(f"{SINK} sim {sim + 1}|{args.task_simulations} done "
                  f"({time.time() - sim_start:.1f}s) → {traj_path.name} {SINK}", flush=True)

    print(f"\n{div}\n Detector RDS generation done in {time.time() - run_start:.1f}s\n{div}")


def build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Detector workflow: generate diffusion-only ReaDDy trajectories "
                    "(RDS parameters drawn from the Detector nuisance).")
    p.add_argument("--total-time-seconds", type=float, required=True,
                   help="Per-run recording length in seconds.")
    group = p.add_mutually_exclusive_group()
    group.add_argument("--tasks", type=int, default=1,
                       help="Generate tasks 0..N-1 (default 1).")
    group.add_argument("--task-id", type=int, default=None,
                       help="Generate exactly one task (HPC array fan-out).")
    p.add_argument("--task-simulations", type=int, default=1,
                   help="Simulations per task.")
    p.add_argument("--split", type=str, default="TRAIN",
                   choices=["TRAIN", "TEST", "EVAL", "train", "test", "eval"],
                   help="Namespace suffix for the generated data.")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for the nuisance draw + particle placement "
                        "(None = non-deterministic).")
    p.add_argument("--no-compress", action="store_true",
                   help="Write .npy instead of compressed .zarr.")
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--dry-run", action="store_true",
                   help="Resolve config + inputs and print planned I/O without running ReaDDy.")
    return p


if __name__ == "__main__":
    main(build_arg_parser().parse_args())

"""Shared RDS-stage engine for both DIMER workflows (biology + detector).

``run_rds(cfg, args)`` holds the entire RDS orchestration -- pre-run banner,
task-index resolution, dry-run probe, log10 prior sampling + exponentiation, the
per-task theta/nuisance write, the sim-0 diagnostics block, the per-simulation
ReaDDy build+run with the kernel-leak mitigation, and the end-of-task report.
The two entry-point scripts
(``SRM_AND_SBI_DIMER_ALP_Simulation_RDS.py`` and its ``_DETECTOR`` twin) shrink
to: build the workflow ``WorkflowConfig``, parse args, and call ``run_rds``.

The only genuine per-workflow difference in RDS is the simulation builder --
biology registers the four reaction channels (reactive), the detector builds the
ReaDDy system diffusion-only (the reaction-diffusion domain is a marginalized
nuisance there). That fork, plus the parameter table / prior bounds / persisted-set
provenance name / display wording, is resolved once in ``_rds_spec(cfg)``; the
engine body below carries no workflow branch.
"""

from __future__ import annotations

import argparse
import gc
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable

import numcodecs
import numpy as np
import readdy
import zarr

from srm_and_sbi_dimer_alp.diagnostics import (
    DiagnosticReporter,
    fixed_parameters_table,
    prior_sampling_table,
)
from srm_and_sbi_dimer_alp.io import save_theta_set
from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming
from srm_and_sbi_dimer_alp.utils import (
    SINK, SOCK, log_memory_state, log_resource_limits, probe_resources,
)
from srm_and_sbi_dimer_alp.workflow import WorkflowConfig


# Short-form unit labels for terminal display (the canonical parameterization
# UNIT field uses long-form English for documentation clarity).
_UNIT_DISPLAY = {
    "Count": "Count",
    "Square Micrometer Per Second": "μm²/s",
    "Square Micrometer Per (Count*Second)": "μm²/(count·s)",
    "Count Per Second": "count/s",
    "Dimensionless": "dimensionless",
}


@dataclass(frozen=True)
class _RdsSpec:
    """Per-workflow RDS specializations resolved from a ``WorkflowConfig``."""
    draw_spec: list          # the sampled parameter table (PARAMETERIZATION | DETECTOR_NUISANCE)
    fixed_spec: list         # the fixed-parameter table (*_RAW)
    lower: np.ndarray        # log10 lower prior bounds
    upper: np.ndarray        # log10 upper prior bounds
    build_sim: Callable      # (theta, seed, skin_factor, verbose) -> ReaDDy Simulation (smut)
    set_path: Callable       # (paths, task, data_bank_root, timing_label, compress, split) -> Path
    spec_title: str          # verbose prior-dump title
    draw_noun: str           # "theta" | "RDS nuisance" ("Sampled <draw_noun>")
    set_label: str           # "theta set" | "nuisance set" (dry-run + banner)
    set_token: str           # "Theta_Set" | "Nuisance_RDS_Theta_Set" (banner path pattern)


def _biology_build_sim(theta, seed, skin_factor, verbose):
    """Reactive builder: register the four reaction channels (biology workflow)."""
    from srm_and_sbi_dimer_alp.simulation_rds_support import build_simulation, build_system
    stem = build_system(theta, verbose=verbose)
    # `stem` is reachable only through the returned Simulation; deleting the
    # Simulation (+ gc) in the engine loop releases the ReaDDy kernel and the system.
    return build_simulation(stem, theta, seed=seed, skin_factor=skin_factor, verbose=verbose)


def _detector_build_sim(theta, seed, skin_factor, verbose):
    """Diffusion-only builder: no reaction channels; assembles canonical theta from
    the RDS-nuisance draw (detector workflow)."""
    from srm_and_sbi_dimer_alp.detector_simulation_rds_support import (
        build_detector_rds_simulation,
    )
    smut, _theta = build_detector_rds_simulation(
        theta, seed=seed, skin_factor=skin_factor, verbose=verbose)
    return smut


def _detector_nuisance_set_path(paths, task_alias, data_bank_root, timing_label, compress, split):
    """Detector RDS-nuisance provenance file: the theta-set path with the object
    token swapped ``Theta_Set`` -> ``Nuisance_RDS_Theta_Set`` (reuses the canonical
    path pattern; no new path code)."""
    base = paths.theta_set_path(task_alias, data_bank_root, timing_label, compress, split)
    return base.with_name(base.name.replace("Theta_Set", "Nuisance_RDS_Theta_Set"))


def _rds_spec(cfg: WorkflowConfig) -> _RdsSpec:
    """Resolve the RDS stage's per-workflow specializations from the workflow config.

    This is the single place the biology/detector fork lives for RDS; the engine
    body reads only the resolved ``_RdsSpec``.
    """
    m = cfg.param_module
    if cfg.tag == "detector":
        return _RdsSpec(
            draw_spec=m.DETECTOR_NUISANCE,
            fixed_spec=m.DETECTOR_PARAMETERIZATION_RAW,
            lower=np.array(m.nuisance_lower_bound()),
            upper=np.array(m.nuisance_upper_bound()),
            build_sim=_detector_build_sim,
            set_path=_detector_nuisance_set_path,
            spec_title="RDS nuisance prior spec",
            draw_noun="RDS nuisance",
            set_label="nuisance set",
            set_token="Nuisance_RDS_Theta_Set",
        )
    return _RdsSpec(
        draw_spec=m.PARAMETERIZATION,
        fixed_spec=m.PARAMETERIZATION_RAW,
        lower=np.array(m.theta_lower_bound()),
        upper=np.array(m.theta_upper_bound()),
        build_sim=_biology_build_sim,
        set_path=lambda paths, *a: paths.theta_set_path(*a),
        spec_title="Learnable theta prior spec",
        draw_noun="theta",
        set_label="theta set",
        set_token="Theta_Set",
    )


def run_rds(cfg: WorkflowConfig, args: argparse.Namespace) -> None:
    """Run the full RDS generation pipeline for the given workflow + CLI args."""
    spec = _rds_spec(cfg)

    # Per-run timing from the required --total-time-seconds + the fixed frame cadence.
    timing = RunTiming(
        total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing,
    )
    split = args.split.upper()   # "TRAIN" / "TEST" / "EVAL" namespace suffix
    data_bank_root = PARAMETERS.machine.root_for(split)  # TRAIN/TEST -> scratch tier; EVAL -> permanent (single-tier machines: always data_bank_root)
    compress = not args.no_compress
    output_fmt = "npy" if args.no_compress else "zarr"

    # ---- Pre-run banner -------------------------------------------------
    machine = PARAMETERS.machine
    geom = PARAMETERS.simulation.stem
    rds_cfg = PARAMETERS.simulation.rds
    paths = cfg.paths
    div = "=" * 72

    print(div)
    print(f" {paths.project_alias} — Simulation_RDS")
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
    if args.task_id is not None:
        print(f"  --task-id            : {args.task_id}        (this run generates exactly one task)")
    else:
        print(f"  --tasks              : {args.tasks}        (this run generates tasks 0..{args.tasks - 1})")
    print(f"  --task-simulations   : {args.task_simulations}")
    print(f"  --split              : {args.split}   (namespace suffix: _{split})")
    print(f"  --seed               : {args.seed}")
    _skin_factor = (args.skin_factor if args.skin_factor is not None
                    else PARAMETERS.simulation.rds.neighbor_list_skin_factor)
    _skin_nm = _skin_factor * PARAMETERS.simulation.stem.particle_diameter_nm
    print(f"  --skin-factor        : {args.skin_factor}   "
          f"(effective neighbor-list skin {_skin_factor:g}x diameter = {_skin_nm:g} nm; "
          f"performance only, no physics change)")
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

    timing_label = timing.label
    print("\nOutput destinations:")
    print(f"  data_bank_root  : {data_bank_root}")
    print(f"  trajectories    : <data_bank>/{paths.video_subdir}/"
          f"{paths.trajectory_repo}/{paths.project_alias}_{timing_label}_TASK_{{n}}/"
          f"{paths.project_alias}_{timing_label}_TASK_{{n}}_SIM_{{m}}.h5")
    print(f"  {spec.set_label + 's':<16}: <data_bank>/{paths.theta_subdir}/"
          f"{paths.project_alias}_{timing_label}_{spec.set_token}_TASK_{{n}}.{output_fmt}")

    if args.verbose:
        print(f"\n{spec.spec_title} ({len(spec.draw_spec)} parameters, "
              f"sampled in log10 then exponentiated):")
        for para in spec.draw_spec:
            lo, hi = para["PRIOR_RANGE"]
            unit = _UNIT_DISPLAY.get(para["UNIT"], para["UNIT"])
            derived = para.get("DERIVED_UNIT")
            derived_str = (f"   derived: {_UNIT_DISPLAY.get(derived, derived)}"
                           if derived else "")
            print(f"  {para['KEY']:<32}  log10 ∈ [{lo:+6.2f}, {hi:+6.2f}]   "
                  f"units: {unit}{derived_str}")

    print(f"\n{div}\n")

    if args.probe:
        log_resource_limits()

    # ---- Determine which task indices to run ----------------------------
    # --task-id K runs exactly one task (HPC array fan-out: one task per job).
    # It reproduces the same per-task theta as task K of a full --tasks run:
    # rng.uniform fills rows in stream order, so row K occupies the same stream
    # positions regardless of the total row count, as long as we draw >= K+1 rows.
    if args.task_id is not None:
        task_indices = [args.task_id]
    else:
        task_indices = list(range(args.tasks))
    n_rows = max(task_indices) + 1

    # ---- Dry run: resolve the planned task/sim workload + output destinations
    # and exit before sampling, the ReaDDy loop, and any directory creation -- so
    # it computes nothing and writes nothing. RDS is the first stage, so it has no
    # inputs to probe -- only the writes it would produce.
    if args.dry_run:
        n_tasks = len(task_indices)
        planned = n_tasks * args.task_simulations
        task_span = (f"{task_indices[0]}" if n_tasks == 1
                     else f"{task_indices[0]}..{task_indices[-1]}")
        print(f"[DRY RUN] plans {n_tasks} task(s) (index {task_span}) x "
              f"{args.task_simulations} sim(s) = {planned} trajectory(ies), "
              f"split _{split}.")
        for task in task_indices:
            theta_set_path = spec.set_path(
                paths, task, data_bank_root, timing_label, compress, split)
            traj_dir = paths.trajectory_dir(
                task, data_bank_root, timing_label, split)
            print(f"  writes {spec.set_label:<13}(task {task}): {theta_set_path}")
            print(f"  writes trajectories (task {task}): {traj_dir}/  "
                  f"({args.task_simulations} .h5 file(s))")
        print("\n[DRY RUN] configuration validated; all outputs resolved.")
        print("[DRY RUN] no trajectories generated.")
        return

    # ---- Sample the parameter sets in log space, exponentiate to physical space --
    low = spec.lower
    high = spec.upper
    rng = np.random.default_rng(args.seed)
    theta_log10 = rng.uniform(
        low=low, high=high,
        size=(n_rows, args.task_simulations, len(low)),
    )
    theta_sets = np.power(10, theta_log10)

    run_start = time.time()

    # ---- Per-task loop --------------------------------------------------
    for loop_i, task in enumerate(task_indices):
        task_alias = task
        print(f"{SOCK} Task {task_alias}  ({loop_i + 1}|{len(task_indices)}) {SOCK}")

        if args.verbose:
            log_memory_state()

        # Ensure the per-task trajectory directory exists.
        traj_dir = paths.trajectory_dir(task_alias, data_bank_root, timing_label, split)
        traj_dir.mkdir(parents=True, exist_ok=True)

        # Write the sampled set: incremental zarr.open for compressed, save_theta_set for plain.
        theta_set_path = spec.set_path(
            paths, task_alias, data_bank_root, timing_label, compress, split)
        theta_set_path.parent.mkdir(parents=True, exist_ok=True)
        theta_set_data = theta_sets[task]  # shape (task_simulations, n_params)

        if compress:
            compressor = numcodecs.Blosc(
                cname="zstd", clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE,
            )
            theta_store = zarr.open(
                store=str(theta_set_path),
                mode="w",
                shape=theta_set_data.shape,
                chunks=(1, theta_set_data.shape[1]),
                dtype=np.float64,
                compressor=compressor,
            )
            theta_store[:, :] = theta_set_data
        else:
            save_theta_set(theta_set_path, theta_set_data, compress=False)

        # ---- Diagnostics reporter (debug mode) ------------------------
        reporter = DiagnosticReporter(
            stage="RDS",
            enabled=args.debug or args.debug_dump,
            dump=args.debug_dump,
            dump_dir=(paths.debug_run_dir(data_bank_root, timing_label, "RDS", split)
                      / f"TASK_{task_alias}"),
            run_label=f"{paths.project_alias}_{timing_label}_TASK_{task_alias}_{split}",
            timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        )
        theta_log10_task = theta_log10[task]

        # ---- Per-simulation loop ----------------------------------------
        for sim in range(args.task_simulations):
            sim_start = time.time()
            theta = theta_set_data[sim]

            if args.verbose:
                print(f"\n  Sampled {spec.draw_noun} (sim {sim + 1}|{args.task_simulations}):")
                for i, para in enumerate(spec.draw_spec):
                    val = theta[i]
                    print(f"    {para['KEY']:<32}  =  {val:10.4g}   "
                          f"(log10 = {np.log10(val):+.3f})")

            # Placement RNG follows --seed (default None -> non-deterministic).
            # Generation is non-deterministic by design: ReaDDy's stepper is
            # unseeded, so identical-posterior reproducibility is unreachable
            # regardless; the sampled set is persisted per task and the global
            # task index is the provenance handle.
            smut = spec.build_sim(theta, args.seed, args.skin_factor, args.verbose)

            traj_path = paths.trajectory_path(
                task_alias, sim, data_bank_root, timing_label, split)
            if traj_path.exists():
                traj_path.unlink()
            smut.output_file = str(traj_path)
            smut.progress_output_stride = timing.total_steps
            print(f"  Writing trajectory: {traj_path}")
            smut.run(
                n_steps=timing.total_steps,
                timestep=timing.delta_time_nanoseconds * readdy.units.nanosecond,
                show_summary=False,
            )

            sim_elapsed = time.time() - sim_start
            print(f"{SINK} Simulation {sim + 1}|{args.task_simulations} done  "
                  f"(elapsed: {sim_elapsed:.1f}s) {SINK}")

            # ---- Sim-0 diagnostics (debug mode) -----------------------
            if reporter.enabled and sim == 0:
                theta_by_key = {p["KEY"]: theta[i]
                                for i, p in enumerate(spec.draw_spec)}
                counts = [theta_by_key.get("count_alp", float("nan")),
                          theta_by_key.get("count_bet", float("nan")),
                          theta_by_key.get("count_chi", float("nan"))]
                total_initial = int(sum(round(c) for c in counts))
                theta_log10_sim = theta_log10_task[sim]
                in_bounds = bool(np.all(theta_log10_sim >= low)
                                 and np.all(theta_log10_sim <= high))

                reporter.checkpoint(
                    "theta sample (sim 0)",
                    n_params=len(theta),
                    initial_particles=total_initial,
                )
                reporter.check(
                    "theta_in_prior_bounds", in_bounds,
                    "all log10 values within the prior box",
                    note="the sampled parameters lie inside the log-uniform "
                         "prior bounds they were drawn from.",
                )
                reporter.check(
                    "initial_particles_positive", total_initial > 0,
                    f"total={total_initial}",
                    note="the simulation starts with at least one particle.",
                )
                reporter.check_file("trajectory", traj_path)

                # Full prior-range vs sampled-value summary for this simulation.
                headers, prior_rows = prior_sampling_table(spec.draw_spec, theta)
                reporter.table(
                    "Prior sampling (sim 0)", headers, prior_rows,
                    note="Log-uniform prior bounds and the value drawn for this "
                         "simulation. count_alp/bet/chi are the initial A/B/C "
                         "particle numbers the run was seeded with -- cross-check "
                         "against the rendered video.",
                )
                fixed_headers, fixed_rows = fixed_parameters_table(spec.fixed_spec)
                reporter.table(
                    "Fixed parameters", fixed_headers, fixed_rows,
                    note="Non-learnable parameters held constant across all "
                         "simulations (Known scientific constants + tuning "
                         "Hyperparameters): camera, PSF, photophysics, capture "
                         "radius, brightness quantiles.",
                )
                reporter.stat(
                    "total_initial_particles", total_initial,
                    note="sum of the sampled A/B/C counts; total particles placed "
                         "at t=0.",
                )

                # Read the trajectory back for reaction-event diagnostics.
                tray = readdy.Trajectory(filename=str(traj_path))
                _, recs = tray.read_observable_reaction_counts()
                reaction_counts = recs["reactions"]
                for rec_name, rec_counts in reaction_counts.items():
                    reporter.stat(
                        f"events[{rec_name}]", int(np.sum(rec_counts)),
                        note="total firings of this reaction channel over the run.")

                if reporter.dump:
                    import warnings as _warnings
                    from srm_and_sbi_dimer_alp.simulation_rds_support import (
                        extract_trajectory_poses,
                    )
                    from srm_and_sbi_dimer_alp.visualization_rds import (
                        figure_position_heatmap,
                        figure_reaction_events,
                    )
                    reporter.save_figure(
                        "reaction_events", figure_reaction_events(reaction_counts),
                        caption="Total firings of each reaction channel over the "
                                "trajectory (fusion, fission, (im)mobilization).",
                    )
                    tray_poses, _ = extract_trajectory_poses(
                        tray, return_dimer_mask=True)
                    with _warnings.catch_warnings():
                        _warnings.filterwarnings(
                            "ignore", message="All-NaN slice encountered",
                            category=RuntimeWarning)
                        poses = np.nanmax(tray_poses, axis=3)
                    xy = poses[..., :2].reshape(-1, 2)
                    xy = xy[~np.isnan(xy).any(axis=1)]
                    reporter.save_figure(
                        "position_heatmap", figure_position_heatmap(xy),
                        caption="2D spatial occupancy of all particle positions "
                                "across all frames; should fill the simulation box.",
                    )

            if args.show:
                tray = readdy.Trajectory(filename=str(traj_path))
                _, recs = tray.read_observable_reaction_counts()
                for rec_name, counts in recs["reactions"].items():
                    print(f"  reaction {rec_name}: {np.count_nonzero(counts)} events")

            # ---- Optional debug probe ------------------------------------
            if args.probe:
                _th, _fd, _rss = probe_resources()
                print(f"[probe] sim {sim + 1}: threads={_th} fds={_fd} "
                      f"rss_mb={_rss}", flush=True)
            # Release the ReaDDy CPU kernel (its worker-thread pool and the
            # observable/output handles) and reclaim memory before the next
            # simulation. Without this each iteration leaks ~320 threads and
            # ~350 MB; a task's RSS then reaches the per-task memory cap after
            # ~47 sims and the cgroup stalls (manifesting as a hang). ``smut`` is
            # the sole reference to the ReaDDy system, so deleting it releases both.
            del smut
            gc.collect()

        # ---- Diagnostics: end-of-task summary + report ----------------
        reporter.summary()
        reporter.write_report()

    total_elapsed = time.time() - run_start
    print(f"\nTotal elapsed: {total_elapsed:.1f}s")


def build_rds_parser() -> argparse.ArgumentParser:
    """Construct the RDS CLI parser (identical for both workflows)."""
    parser = argparse.ArgumentParser(
        description="Generate ReaDDy reaction-diffusion trajectories for the DIMER model.",
    )
    parser.add_argument(
        "--total-time-seconds", type=float, required=True,
        help="Simulation duration per trajectory in seconds (required; e.g. 2.0, 5.0).",
    )
    parser.add_argument(
        "--tasks", type=int, default=2,
        help="Number of theta-set batches to generate (default: 2). Ignored when "
             "--task-id is given.",
    )
    parser.add_argument(
        "--task-id", type=int, default=None,
        help="Generate exactly one task with this index (for HPC array fan-out: "
             "one task per job). Reproduces the same per-task theta as task <id> of "
             "the equivalent full --tasks run. Overrides --tasks.",
    )
    parser.add_argument(
        "--task-simulations", type=int, default=5,
        help="Number of simulations per task (default: 5).",
    )
    parser.add_argument(
        "--split", choices=["train", "test", "eval"], default="train",
        help="Dataset role this run generates, written to its own namespace "
             "(filename suffix _TRAIN / _TEST / _EVAL). Default: train.",
    )
    parser.add_argument(
        "--seed",
        type=lambda v: None if str(v).strip().lower() in ("none", "") else int(v),
        default=None,
        help="RNG seed for theta sampling and initial particle placement. "
             "Default: None (non-deterministic).",
    )
    parser.add_argument(
        "--skin-factor", type=float, default=None,
        help="ReaDDy neighbor-list (Verlet) skin as a MULTIPLE of the particle "
             "diameter (skin = skin-factor x particle_diameter_nm). PERFORMANCE-ONLY "
             "knob: it coarsens the cell-linked-list grid in the large, dilute box; it "
             "does NOT change the physics (reactions still fire at the true reaction "
             "radius, so the output is statistically identical). Default: None -> the "
             "configured PARAMETERS.simulation.rds.neighbor_list_skin_factor (10x = "
             "100 nm), which sits on the fast plateau. See that field for the U-shaped "
             "cost rationale.",
    )
    parser.add_argument(
        "--no-compress", action="store_true",
        help="Save theta sets as .npy instead of compressed .zarr.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration and inputs, print what would be read/written, then exit "
             "without running the stage (no GPU, no compute). Use before a queue submission or a long local run.",
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Print diagnostic info during setup (rates, particle counts).",
    )
    parser.add_argument(
        "--show", action="store_true",
        help="After each simulation, print non-zero reaction counts from the trajectory.",
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
    return parser

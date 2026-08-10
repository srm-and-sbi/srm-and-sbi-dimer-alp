"""Shared DLI-stage engine for both DIMER workflows (biology + detector).

``run_dli(cfg, args)`` holds the DLI orchestration -- pre-run banner, the SCOPE
camera draw, the dry-run probe, the per-task render loop (read trajectory ->
extract poses -> frame-count guard -> nanmax collapse -> assemble the eleven-key
imaging vector -> ``render_dli_video`` -> dtype convert -> store), the sim-0
diagnostics, and the end-of-task report. The two entry-point scripts shrink to:
build the workflow ``WorkflowConfig``, parse args, call ``run_dli``.

DLI genuinely diverges more than RDS: the imaging block's SOURCE and ROLE differ
between the workflows, so the fork is larger and localized in labeled branches on
``spec.imaging_source``:

  - **biology** (``imaging_source="artifact"``): the six photophysics are a
    marginalized nuisance drawn per task from the persisted ``Nuisance_DLI``
    artifact (a required, schema-guarded input) and recorded as
    ``Nuisance_DLI_Theta_Set``; the learnable 10-RDS ``Theta_Set`` is READ for the
    sim-0 diagnostics only.
  - **detector** (``imaging_source="prior_box"``): the six imaging parameters are
    the inference target, drawn from the imaging prior box and WRITTEN as the
    primary ``Theta_Set`` (the training label); no ``Nuisance_DLI`` artifact.

Everything else -- the renderer (already shared as ``render_dli_video``), the
SCOPE draw + ``Nuisance_SCOPE_Theta_Set`` write, the video store, and the sim-0
core diagnostics -- is shared. The sim-0 "Parameters of this video" table uses
each workflow's TARGET spec + its per-sim label vector (biology: the 10-RDS
table + the read RDS theta; detector: the six-imaging table + the imaging draw);
this also fixes a latent ``NameError`` in the detector's ``--debug`` path, where
the copied-from-biology table referenced an undefined ``theta`` / the wrong table.
"""

from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass
from datetime import datetime, timezone

import numcodecs
import numpy as np
import readdy
import zarr

from srm_and_sbi_dimer_alp import detector_parameterization as det
from srm_and_sbi_dimer_alp.detector_nuisance_dli import (
    artifact_path as nuisance_dli_artifact_path,
    require_nuisance_dli,
)
from srm_and_sbi_dimer_alp.diagnostics import (
    DiagnosticReporter,
    fixed_parameters_table,
    prior_sampling_table,
)
from srm_and_sbi_dimer_alp.io import (
    convert_video_dtype, load_data, save_theta_set, save_video_set,
)
from srm_and_sbi_dimer_alp.parameterization import (
    PARAMETER_RAW_FIND,
    PARAMETERIZATION,
    PARAMETERIZATION_RAW,
    PARAMETERS,
    RunTiming,
)
from srm_and_sbi_dimer_alp.simulation_dli_support import render_dli_video
from srm_and_sbi_dimer_alp.simulation_rds_support import extract_trajectory_poses
from srm_and_sbi_dimer_alp.utils import (
    SINK, SOCK, log_memory_state, log_resource_limits, probe_resources,
)
from srm_and_sbi_dimer_alp.workflow import WorkflowConfig


_DTYPE_FOR_BITS = {8: np.uint8, 16: np.uint16}

# Grouped DLI known-parameter spec for the biology --verbose banner block. KEYs
# match entries in PARAMETERIZATION_RAW; the group labels are display-only.
_DLI_PARAM_GROUPS = {
    "PSF (Point Spread Function)": ["mu_r", "sigma_r", "mu_pc", "sigma_pc"],
    "EMCCD camera (SCOPE nuisance)": ["gamma", "kappa_o", "kappa_b", "kappa_s", "kappa_q", "kappa_g", "kappa_c"],
    "State machine (brightness transitivity)": [
        "brightness_quantile", "delta_frame", "prob_photo_bleach",
        "numb_photo_bleach", "lambda_rate",
    ],
}


def _nuisance_dli_path(paths, task_alias, data_bank_root, timing_label, compress, split):
    """Photophysics (calibrated-imaging) nuisance provenance file: the canonical
    theta-set path with ``Theta_Set`` -> ``Nuisance_DLI_Theta_Set`` (biology only)."""
    base = paths.theta_set_path(task_alias, data_bank_root, timing_label, compress, split)
    return base.with_name(base.name.replace("Theta_Set", "Nuisance_DLI_Theta_Set"))


def _nuisance_scope_path(paths, task_alias, data_bank_root, timing_label, compress, split):
    """SCOPE camera-nuisance provenance file: the canonical theta-set path with
    ``Theta_Set`` -> ``Nuisance_SCOPE_Theta_Set`` (both workflows, shared camera block)."""
    base = paths.theta_set_path(task_alias, data_bank_root, timing_label, compress, split)
    return base.with_name(base.name.replace("Theta_Set", "Nuisance_SCOPE_Theta_Set"))


@dataclass(frozen=True)
class _DliSpec:
    """Per-workflow DLI specializations resolved from a ``WorkflowConfig``."""
    imaging_source: str      # "artifact" (biology) | "prior_box" (detector)
    diag_spec: list          # sim-0 "Parameters of this video" table: PARAMETERIZATION (10-RDS) | DETECTOR_PARAMETERIZATION (6 imaging)


def _dli_spec(cfg: WorkflowConfig) -> _DliSpec:
    """Resolve the DLI stage's per-workflow specializations from the workflow config."""
    if cfg.tag == "detector":
        return _DliSpec(imaging_source="prior_box", diag_spec=det.DETECTOR_PARAMETERIZATION)
    return _DliSpec(imaging_source="artifact", diag_spec=PARAMETERIZATION)


def run_dli(cfg: WorkflowConfig, args: argparse.Namespace) -> None:
    """Run the full DLI rendering pipeline for the given workflow + CLI args."""
    spec = _dli_spec(cfg)
    artifact = spec.imaging_source == "artifact"   # biology; else detector prior-box

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
    paths = cfg.paths
    div = "=" * 72
    timing_label = timing.label

    # ---- Imaging block setup (the six photophysics/imaging + five SCOPE camera).
    # Both share the SCOPE box (drawn from det.scope_*_bound) and the six imaging keys
    # (det.DETECTOR_PARAMETER_KEYS). They differ in the six-block SOURCE:
    #   biology  -> the persisted Nuisance_DLI artifact (required, schema-guarded);
    #   detector -> the imaging prior box (det.theta_*_bound), the inference target.
    dli_keys = det.DETECTOR_PARAMETER_KEYS       # 6 photophysics/imaging keys (both)
    scope_keys = det.DETECTOR_SCOPE_KEYS         # 5 camera -> Nuisance_SCOPE (both, from the box)
    slow = np.array(det.scope_lower_bound())
    shigh = np.array(det.scope_upper_bound())
    if artifact:
        # biology: resolve the durable Nuisance_DLI artifact (like the estimator: EVAL tier,
        # Detector alias). Loaded/guarded below; the six photophysics are drawn from it per task.
        det_paths = det.detector_paths(PARAMETERS.paths)
        posit_dir = PARAMETERS.machine.root_for("EVAL") / PARAMETERS.paths.posit_subdir
        nuisance_artifact = nuisance_dli_artifact_path(
            posit_dir, det_paths.project_alias, timing_label)
    else:
        # detector: the six imaging are the learnable inference target, drawn from the box.
        ilow = np.array(det.theta_lower_bound())
        ihigh = np.array(det.theta_upper_bound())

    print(div)
    if artifact:
        print(f" {paths.project_alias} — Simulation_DLI")
    else:
        print(f" {paths.project_alias} — Detector Simulation_DLI (imaging from theta)")
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
          f"(multiply-model factor; inert under the sum model — see PROJECT_CONTEXT.md)")
    print(f"  optical background      : SCOPE nuisance kappa_o (drawn per sim; pre-PSF photon floor)")
    print(f"  sqrt_2sigma_dist_label  : {dli_cfg.sqrt_2sigma_dist_label}            "
          f"(PSF width sampling distribution)")

    print("\nOutput destinations:")
    print(f"  data_bank_root       : {data_bank_root}")
    if artifact:
        print(f"  reads Nuisance_DLI    : {nuisance_artifact}   "
              f"(calibrated-imaging photophysics artifact; durable tier)")
        print(f"  reads theta sets     : <data_bank>/{paths.theta_subdir}/"
              f"{paths.project_alias}_{timing_label}_Theta_Set_TASK_{{n}}.{output_fmt}   "
              f"(10-RDS labels; diagnostics only, not re-written)")
        print(f"  reads trajectories   : <data_bank>/{paths.video_subdir}/"
              f"{paths.trajectory_repo}/{paths.project_alias}_{timing_label}_TASK_{{n}}/"
              f"{paths.project_alias}_{timing_label}_TASK_{{n}}_SIM_{{m}}.h5")
        print(f"  writes Nuisance_DLI   : <data_bank>/{paths.theta_subdir}/"
              f"{paths.project_alias}_{timing_label}_Nuisance_DLI_Theta_Set_TASK_{{n}}.{output_fmt}   "
              f"(photophysics drawn from the artifact)")
        print(f"  writes Nuisance_SCOPE : <data_bank>/{paths.theta_subdir}/"
              f"{paths.project_alias}_{timing_label}_Nuisance_SCOPE_Theta_Set_TASK_{{n}}.{output_fmt}   "
              f"(marginalized camera)")
        print(f"  writes video sets    : <data_bank>/{paths.video_subdir}/"
              f"{paths.project_alias}_{timing_label}_Video_Set_TASK_{{n}}.{output_fmt}   "
              f"(dtype: {target_dtype_name})")
    else:
        print(f"  writes theta sets   : <data_bank>/{paths.theta_subdir}/"
              f"{paths.project_alias}_{timing_label}_Theta_Set_TASK_{{n}}.{output_fmt}   "
              f"(imaging-theta labels; the inference target)")
        print(f"  writes SCOPE sets    : <data_bank>/{paths.theta_subdir}/"
              f"{paths.project_alias}_{timing_label}_Nuisance_SCOPE_Theta_Set_TASK_{{n}}.{output_fmt}   "
              f"(marginalized camera nuisance)")
        print(f"  reads trajectories  : <data_bank>/{paths.video_subdir}/"
              f"{paths.trajectory_repo}/{paths.project_alias}_{timing_label}_TASK_{{n}}/"
              f"{paths.project_alias}_{timing_label}_TASK_{{n}}_SIM_{{m}}.h5")
        print(f"  writes video sets   : <data_bank>/{paths.video_subdir}/"
              f"{paths.project_alias}_{timing_label}_Video_Set_TASK_{{n}}.{output_fmt}   "
              f"(dtype: {target_dtype_name})")

    if args.verbose:
        if artifact:
            print("\nImaging parameters (marginalized per simulation):")
            for group_label, keys in _DLI_PARAM_GROUPS.items():
                print(f"  --- {group_label} ---")
                for key in keys:
                    para = PARAMETERIZATION_RAW[PARAMETER_RAW_FIND[key]]
                    val = para["VALUE"]
                    unit = para["UNIT"]
                    unit_str = f"        units: {unit}" if unit else ""
                    if isinstance(val, str):   # role sentinel (NUISANCE): a marginalized draw, not a fixed value
                        if para["PRIOR_RANGE"] is None:
                            # nuisance_object (the six photophysics): drawn per sim from the persisted
                            # Nuisance_DLI artifact -- no a-priori box to print (PRIOR_RANGE is None).
                            print(f"  {key:<24}  : {val} (drawn per sim from the Nuisance_DLI artifact){unit_str}")
                        else:
                            # nuisance_spec (the SCOPE camera): drawn per sim from its a-priori box.
                            lo, hi = para["PRIOR_RANGE"]
                            box = f"10^[{lo}, {hi}]" if para.get("LOG_FLAG") else f"[{lo}, {hi}]"
                            print(f"  {key:<24}  : {val} (drawn per sim; {box}){unit_str}")
                    else:
                        print(f"  {key:<24}  : {val}{unit_str}")
        else:
            print("\nLearnable imaging theta (the inference target / training label):")
            for key, lo, hi in zip(dli_keys, ilow, ihigh):
                print(f"  {key:<24}  : log10 ∈ [{lo:+.3f}, {hi:+.3f}]   (10^ → physical)")
            print("SCOPE camera nuisance (marginalized; recorded as Nuisance_SCOPE):")
            for key, lo, hi in zip(scope_keys, slow, shigh):
                print(f"  {key:<24}  : log10 ∈ [{lo:+.3f}, {hi:+.3f}]   (10^ → physical)")

    print(f"\n{div}\n")

    run_start = time.time()

    # --task-id K renders exactly one task (HPC array fan-out: one task per job); otherwise all of --tasks.
    task_indices = [args.task_id] if args.task_id is not None else list(range(args.tasks))
    n_rows = max(task_indices) + 1

    # ---- Up-front imaging draws (log10 -> physical via 10**), indexed per task so a
    # given task index draws the SAME vectors under --task-id fan-out. The RNG draw
    # ORDER differs by workflow and is preserved verbatim for reproducibility:
    #   biology  -> only the SCOPE block is drawn from `rng` (the six photophysics come
    #               from the Nuisance_DLI artifact's own unseeded sampler, per task);
    #   detector -> the learnable imaging block THEN the SCOPE block, from the same `rng`.
    rng = np.random.default_rng(args.seed)
    if artifact:
        scope_sets = np.power(10, rng.uniform(
            low=slow, high=shigh,
            size=(n_rows, args.task_simulations, len(slow))))      # (.., 5) physical SCOPE nuisance
    else:
        learnable_sets = np.power(10, rng.uniform(
            low=ilow, high=ihigh,
            size=(n_rows, args.task_simulations, len(ilow))))      # (.., 6) physical imaging labels
        scope_sets = np.power(10, rng.uniform(
            low=slow, high=shigh,
            size=(n_rows, args.task_simulations, len(slow))))      # (.., 5) physical SCOPE nuisance

    # ---- Dry run: resolve the planned workload + input/output destinations and exit
    # before the render loop and any directory creation -- computes nothing, writes nothing.
    if args.dry_run:
        n_tasks = len(task_indices)
        planned = n_tasks * args.task_simulations
        task_span = (f"{task_indices[0]}" if n_tasks == 1
                     else f"{task_indices[0]}..{task_indices[-1]}")
        print(f"[DRY RUN] plans {n_tasks} task(s) (index {task_span}) x "
              f"{args.task_simulations} sim(s) = {planned} video(s), "
              f"split _{split}.")
        if artifact:
            # The imaging block is marginalized: photophysics from the persisted Nuisance_DLI
            # artifact (a required input, like the estimator), camera from the SCOPE box.
            dli_ok = nuisance_artifact.exists()
            dli_sample = None
            if dli_ok:
                nuisance_dli = require_nuisance_dli(posit_dir, det_paths.project_alias, timing_label)
                if list(nuisance_dli.parameter_keys) != dli_keys:
                    raise ValueError(
                        f"Nuisance_DLI schema mismatch: artifact parameter_keys "
                        f"{list(nuisance_dli.parameter_keys)} != expected {dli_keys}.")
                dli_sample = np.power(10, nuisance_dli.sample(1))[0]
            print(f"  reads Nuisance_DLI   : {nuisance_artifact}  "
                  f"[{'OK' if dli_ok else 'MISSING'}]")
            print(f"[DRY RUN] imaging draw: photophysics from Nuisance_DLI (6) "
                  f"+ SCOPE box {scope_sets.shape} (5).")
            missing = 0 if dli_ok else 1
            for task in task_indices:
                theta_set_path = paths.theta_set_path(
                    task, data_bank_root, timing_label, compress, split)
                theta_ok = theta_set_path.exists()
                if not theta_ok:
                    missing += 1
                traj_present = 0
                for sim in range(args.task_simulations):
                    traj_path = paths.trajectory_path(
                        task, sim, data_bank_root, timing_label, split)
                    if traj_path.exists():
                        traj_present += 1
                    else:
                        missing += 1
                video_set_path = paths.video_set_path(
                    task, data_bank_root, timing_label, compress, split)
                dli_set_path = _nuisance_dli_path(
                    paths, task, data_bank_root, timing_label, compress, split)
                scope_set_path = _nuisance_scope_path(
                    paths, task, data_bank_root, timing_label, compress, split)
                print(f"  reads theta set      (task {task}): {theta_set_path}  "
                      f"[{'OK' if theta_ok else 'MISSING'}]   (10-RDS labels; diagnostics only, not re-written)")
                print(f"  reads trajectories   (task {task}): "
                      f"{traj_present}/{args.task_simulations} present")
                print(f"  writes Nuisance_DLI   (task {task}): {dli_set_path}")
                if dli_sample is not None:
                    print(f"    Nuisance_DLI[sim 0] (physical): "
                          f"{dict(zip(dli_keys, np.round(dli_sample, 4).tolist()))}")
                print(f"  writes Nuisance_SCOPE (task {task}): {scope_set_path}")
                print(f"    Nuisance_SCOPE[sim 0] (physical): "
                      f"{dict(zip(scope_keys, np.round(scope_sets[task, 0], 4).tolist()))}")
                print(f"  writes video set     (task {task}): {video_set_path}")
            if missing:
                print(f"\n[DRY RUN] configuration validated; {missing} input(s) MISSING.")
            else:
                print("\n[DRY RUN] configuration validated; all inputs present.")
        else:
            print(f"[DRY RUN] imaging draw: learnable {learnable_sets.shape} (Theta_Set) "
                  f"+ SCOPE {scope_sets.shape} (Nuisance_SCOPE).")
            missing = 0
            for task in task_indices:
                theta_set_path = paths.theta_set_path(
                    task, data_bank_root, timing_label, compress, split)
                traj_present = 0
                for sim in range(args.task_simulations):
                    traj_path = paths.trajectory_path(
                        task, sim, data_bank_root, timing_label, split)
                    if traj_path.exists():
                        traj_present += 1
                    else:
                        missing += 1
                video_set_path = paths.video_set_path(
                    task, data_bank_root, timing_label, compress, split)
                scope_set_path = _nuisance_scope_path(
                    paths, task, data_bank_root, timing_label, compress, split)
                print(f"  writes theta set    (task {task}): {theta_set_path}")
                print(f"  writes SCOPE set     (task {task}): {scope_set_path}")
                print(f"    learnable[sim 0] (physical): "
                      f"{dict(zip(dli_keys, np.round(learnable_sets[task, 0], 4).tolist()))}")
                print(f"    SCOPE[sim 0] (physical): "
                      f"{dict(zip(scope_keys, np.round(scope_sets[task, 0], 4).tolist()))}")
                print(f"  reads trajectories  (task {task}): "
                      f"{traj_present}/{args.task_simulations} present")
                print(f"  writes video set    (task {task}): {video_set_path}")
            if missing:
                print(f"\n[DRY RUN] configuration validated; {missing} trajectory input(s) MISSING.")
            else:
                print("\n[DRY RUN] configuration validated; all trajectory inputs present.")
        print("[DRY RUN] no videos rendered.")
        return

    # ---- Load + schema-guard the required Nuisance_DLI artifact once (biology only;
    # fails loud naming the analysis if absent; never rebuilds).
    if artifact:
        nuisance_dli = require_nuisance_dli(posit_dir, det_paths.project_alias, timing_label)
        if list(nuisance_dli.parameter_keys) != dli_keys:
            raise ValueError(
                f"Nuisance_DLI schema mismatch: artifact parameter_keys "
                f"{list(nuisance_dli.parameter_keys)} != expected {dli_keys} "
                f"(the six photophysics in DETECTOR_PARAMETER_KEYS order).")

    for loop_i, task in enumerate(task_indices):
        task_alias = task
        print(f"{SOCK} Task {task_alias}  ({loop_i + 1}|{len(task_indices)}) {SOCK}")

        if args.verbose:
            log_memory_state()

        scope_data = scope_sets[task_alias]                                   # (sims, 5) physical, DETECTOR_SCOPE_KEYS order
        scope_set_path = _nuisance_scope_path(
            paths, task_alias, data_bank_root, timing_label, compress, split)

        # ---- Draw + record this task's imaging block. The six-block source + record
        # provenance is the biology/detector fork; the five-block SCOPE record is shared.
        if artifact:
            # biology: the six photophysics from the Nuisance_DLI artifact (recorded as
            # Nuisance_DLI_Theta_Set); READ the learnable 10-RDS Theta_Set for the sim-0
            # diagnostics table (this stage does not re-write it; the RDS stage owns it).
            rds_theta_set_path = paths.theta_set_path(
                task_alias, data_bank_root, timing_label, compress, split)
            print(f"  Reading theta set (10-RDS labels):  {rds_theta_set_path}")
            rds_theta_set = load_data(rds_theta_set_path)
            imaging_draw = np.power(10, nuisance_dli.sample(args.task_simulations))  # (sims, 6) physical
            imaging_set_path = _nuisance_dli_path(
                paths, task_alias, data_bank_root, timing_label, compress, split)
            imaging_set_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Writing Nuisance_DLI:   {imaging_set_path}")
            print(f"  Writing Nuisance_SCOPE: {scope_set_path}")
        else:
            # detector: the six imaging are the inference target, drawn from the box and
            # WRITTEN as the primary Theta_Set (the training label). No RDS theta is read.
            rds_theta_set = None
            imaging_draw = learnable_sets[task_alias]                          # (sims, 6) physical labels
            imaging_set_path = paths.theta_set_path(
                task_alias, data_bank_root, timing_label, compress, split)
            imaging_set_path.parent.mkdir(parents=True, exist_ok=True)
            print(f"  Writing imaging theta:  {imaging_set_path}")
            print(f"  Writing SCOPE nuisance: {scope_set_path}")

        if compress:
            theta_compressor = numcodecs.Blosc(
                cname="zstd", clevel=9, shuffle=numcodecs.Blosc.BITSHUFFLE,
            )
            imaging_store = zarr.open(
                store=str(imaging_set_path), mode="w", shape=imaging_draw.shape,
                chunks=(1, imaging_draw.shape[1]), dtype=np.float64,
                compressor=theta_compressor,
            )
            imaging_store[:, :] = imaging_draw
            scope_store = zarr.open(
                store=str(scope_set_path), mode="w", shape=scope_data.shape,
                chunks=(1, scope_data.shape[1]), dtype=np.float64,
                compressor=theta_compressor,
            )
            scope_store[:, :] = scope_data
        else:
            save_theta_set(imaging_set_path, imaging_draw, compress=False)
            save_theta_set(scope_set_path, scope_data, compress=False)

        # ---- Create the video-set store/buffer ------------------------
        video_set_path = paths.video_set_path(
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
            traj_path = paths.trajectory_path(
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

            # Assemble the full eleven-key imaging vector (det.DETECTOR_IMAGING order): the six
            # photophysics/imaging followed by the five SCOPE camera-nuisance draws.
            imaging_physical = np.concatenate([imaging_draw[sim], scope_data[sim]])
            frames = render_dli_video(
                pro_tray_poses=pro_tray_poses,
                imaging_physical=imaging_physical,
                dimer_mask=dimer_mask,
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
                # parameters sit next to the rendered frame for direct checking. The
                # spec + label vector are the workflow's TARGET: biology the 10-RDS
                # labels read from the Theta_Set; detector the six imaging labels drawn
                # here (det.DETECTOR_PARAMETERIZATION) -- which also fixes the detector's
                # latent NameError (it previously referenced an undefined `theta`).
                diag_theta = rds_theta_set[sim, :] if artifact else imaging_draw[sim]
                headers, prior_rows = prior_sampling_table(spec.diag_spec, diag_theta)
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


def build_dli_parser() -> argparse.ArgumentParser:
    """Construct the DLI CLI parser (identical for both workflows)."""
    parser = argparse.ArgumentParser(
        description="Render diffraction-limited videos from RDS trajectories.",
    )
    parser.add_argument(
        "--total-time-seconds", type=float, required=True,
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
             "task per job), reading that task's trajectories. Overrides --tasks.",
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
        "--seed",
        type=lambda v: None if str(v).strip().lower() in ("none", "") else int(v),
        default=None,
        help="RNG seed for PSF widths, brightness states, and noise. "
             "Default: None (non-deterministic).",
    )
    parser.add_argument(
        "--no-compress", action="store_true",
        help="Save video sets as .npy instead of compressed .zarr.",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration and inputs, print what would be read/written, then exit "
             "without running the stage (no GPU, no compute). Use before a queue submission or a long local run.",
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
    return parser

"""Shared engine for the horizon audit (continuous-versus-reset window comparison).

The audit tests the reset assumption the experimental window analysis rests on: the estimator is
trained on independently initialized model-window simulations, but deployed on consecutive windows
of continuous recordings whose later windows inherit latent state. The kernel
(:mod:`horizon_audit`) holds the statistics; this runner orchestrates the phases, each an explicit
CLI subcommand so the expensive stages can be split across machines and resumed:

    prepare  -- draw the theta cohort from the training prior once, stamp it with a cohort
                digest, and persist it. Every later phase reads this one file, so parallel
                workers on different machines see the SAME cohort, and every artifact carries the
                digest so a stale file from an older cohort is refused, never silently mixed in.
    generate -- per theta: ONE continuous long simulation rendered as a video (with the per-frame
                species populations extracted from its trajectory -- the ground truth for dynamic
                state), plus R independently initialized reset simulations at the same theta.
                CPU-bound (ReaDDy + renderer); embarrassingly parallel over theta via
                --theta-start/--theta-stop, one output file per theta.
    infer    -- slice every stored video into model-length windows and run the trained estimator
                on each: posterior quantiles + raw draws per window. GPU-bound; same per-theta
                range splitting, one result file per theta.
    analyze  -- aggregate everything into the audit statistics, the report and the figures.
                CPU-only.
    selftest -- pure-python checks of the audit's decision-critical logic (truth references,
                seed uniqueness, coverage disaggregation, stale-cohort rejection, the
                absolute-versus-signed contrast, the equivalence verdict). No simulation, no GPU.

DESIGN DECISIONS.

* STATE TRUTH = THE WINDOW-START POPULATION. The estimator's count labels are the initial
  populations of its training windows, so the start state is what it was trained to report.
  The within-window mean and end populations, and the unrounded theta label, are retained as
  explicitly secondary sensitivity references -- judging against them manufactures an estimand
  mismatch that can dwarf the real error (measured on the smoke: 36 pp fake vs ~1 pp real).
* PRIMARY STATISTIC = PAIRED ABSOLUTE-ERROR CONTRAST. |continuous error| minus the same
  trajectory's mean |reset error|: a bias-only (signed) contrast can read zero while accuracy
  collapses symmetrically. Signed contrasts are reported as a secondary bias read.
* THREE-OUTCOME VERDICTS AGAINST PRESPECIFIED MARGINS. Absence of a detected difference is not
  evidence of equivalence: each primary estimand carries a prespecified equivalence margin
  (default: the estimator's own in-model error on the 10,000-video held-out recovery -- a
  degradation smaller than the instrument's own error is practically negligible), and the
  verdict is degraded / equivalent / inconclusive from the trajectory-bootstrap CI.
* PRIMARY ESTIMANDS ARE PREDECLARED: the dimer-complex fraction f_D (state, judged against the
  start truth), D_A and kappa_OFF (constants, judged against theta). Every other parameter is
  exploratory: reported, never verdicted (multiplicity).
* COVERAGE IS NEVER POOLED ACROSS ESTIMAND KINDS. Constants (R_ON excluded -- already known
  unidentified) are reported separately from the counts, whose "coverage" against the stale t=0
  truth measures state drift, not calibration; f_D coverage is computed from the within-draw
  fraction distribution against the start truth.
* SEEDS ARE SPAWNED, PERSISTED, AND NEVER SHARED. A master seed (drawn and persisted at prepare
  when not supplied) derives one placement seed and one render seed per (theta, arm, replicate)
  via ``numpy.random.SeedSequence`` -- resets are independently randomized, not copies of one
  stream. ReaDDy's internal reaction/diffusion RNG has no exposed seed and stays OS-seeded, so
  trajectories are not bit-reproducible; placements and renders are.
* The imaging vector is PINNED for every render, continuous and reset alike, to the calibrated
  ``Nuisance_DLI`` + MET SCOPE vector -- a MET-conditioned, training-supported imaging SLICE
  (training additionally varied the SCOPE camera nuisance box), resolved by the same helper the
  posterior-predictive render uses. Identical imaging on both sides makes the paired contrast
  cancel everything but the inherited state.
* Videos are stored uint8 through the same fixed-range 16-to-8-bit conversion the training zarr
  sets and the experimental recordings both pass through, and windows are stepped exactly as the
  Experiment stage steps real recordings.
* Window estimates are posterior draws (quantiles + raw cloud), not MAP points: every audit
  estimand is a posterior functional, and sampling is two orders of magnitude cheaper per window
  than seed-then-optimize MAP. The default sampler is ``unrestricted``, matching the
  experimental-baseline methodology; the outside-the-box flow-mass diagnostic REQUIRES it.
* The trajectory (one theta) is the independent statistical unit everywhere; windows within a
  trajectory are correlated and are never treated as replicates.

This analysis is post-hoc and ad-hoc: it validates a deployment practice of the canonical
pipeline and stays out of the stage dispatcher, like the other Analysis utilities.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import horizon_audit as ha
from . import population_composition as pc
from .diagnostics import DiagnosticReporter
from .parameterization import PARAMETERS, RunTiming
from .simulation_rds_support import build_simulation, build_system, extract_trajectory_poses
from .workflow import parameter_table

_COUNT_KEYS = ("count_alp", "count_bet", "count_chi")     # A, B, C -- the species-rank order

# Predeclared primary estimands and their default equivalence margins: the estimator's own
# in-model error on the 10,000-video held-out recovery (report Experimental_Synthetic_Comparison
# sections 3.1-3.2) -- a degradation smaller than the instrument's own error is practically
# negligible. f_D in percentage points; the constants in dex. CLI-overridable.
_DEFAULT_MARGINS = {"f_D": 2.8, "diffusivity_alp": 0.023, "rate_dissociation": 0.177}


# =============================================================================
# Spec resolution (biology-only)
# =============================================================================

def _horizon_audit_spec(cfg, args):
    """Resolve paths, keys and timings for the audit; biology-only by construction.

    The audit needs the full reactive system (the inherited state IS the reaction state) and the
    species counts among the learnable parameters; the detector workflow has neither, so it is
    refused explicitly rather than producing a meaningless run.
    """
    if cfg.tag != "biology":
        raise SystemExit(
            f"The horizon audit applies to the biology workflow only: the {cfg.tag} workflow "
            f"marginalizes the reaction-diffusion block, so a continuous reactive trajectory has "
            f"no inferred counterpart to audit.")
    keys = list(cfg.param_module.PARAMETER_KEYS)
    missing = [k for k in _COUNT_KEYS if k not in keys]
    if missing:
        raise SystemExit(f"parameterization lacks the species-count keys {missing}; "
                         f"the audit's dynamic-state comparison needs all three.")
    window = RunTiming(total_time_seconds=args.total_time_seconds,
                       frames=PARAMETERS.simulation.timing)
    continuous = RunTiming(total_time_seconds=args.continuous_seconds,
                           frames=PARAMETERS.simulation.timing)
    if continuous.frame_count % window.frame_count != 0:
        raise SystemExit(
            f"--continuous-seconds={args.continuous_seconds:g} does not tile into whole "
            f"model windows of {args.total_time_seconds:g}s "
            f"({continuous.frame_count} frames vs {window.frame_count}/window).")
    data_bank_root = PARAMETERS.machine.data_bank_root
    paths = cfg.paths
    out_dir = (data_bank_root / paths.posit_subdir
               / f"{paths.project_alias}_{window.label}_Horizon_Audit")
    return dict(
        paths=paths, keys=keys,
        count_index=tuple(keys.index(k) for k in _COUNT_KEYS),
        lower=np.asarray(cfg.param_module.theta_lower_bound(), dtype=float),
        upper=np.asarray(cfg.param_module.theta_upper_bound(), dtype=float),
        build_prior=cfg.param_module.build_prior,
        table=parameter_table(cfg),
        window=window, continuous=continuous,
        n_windows=continuous.frame_count // window.frame_count,
        estimator_path=paths.estimator_path(data_bank_root, window.label),
        data_bank_root=data_bank_root,
        out_dir=out_dir,
        cohort_path=out_dir / "cohort_theta.npz",
        cohort_dir=out_dir / "cohort",
    )


def _theta_file(spec, index):
    return spec["cohort_dir"] / f"theta_{index:04d}.npz"


def _result_file(spec, index):
    return spec["cohort_dir"] / f"result_{index:04d}.npz"


# =============================================================================
# Cohort identity + seed derivation
# =============================================================================

def _cohort_digest(theta_log10, keys, window_seconds, continuous_seconds, n_resets,
                   master_seed):
    """Digest identifying one cohort: the exact theta matrix plus every generation-shaping knob.

    Stamped into every per-theta artifact and verified before any file is used, so artifacts of
    an older cohort in the same directory are refused, never silently concatenated with a new one.
    """
    h = hashlib.sha1()
    h.update(np.ascontiguousarray(np.asarray(theta_log10, dtype=np.float64)).tobytes())
    h.update(",".join(keys).encode())
    h.update(f"|w={window_seconds:g}|c={continuous_seconds:g}|R={n_resets}"
             f"|seed={master_seed}".encode())
    return h.hexdigest()[:12]


def _file_digest(path):
    """Short content digest of a small artifact file (the estimator), for provenance stamping."""
    h = hashlib.sha1()
    h.update(Path(path).read_bytes())
    return h.hexdigest()[:12]


def _spawn_seeds(master_seed, index, n_resets):
    """Independent, reproducible seeds for every simulation component of one theta.

    Derives ``2 * (1 + n_resets)`` uint32 seeds from ``SeedSequence([master_seed, index])``:
    a (placement, render) pair for the continuous arm and one pair per reset. Deterministic in
    (master_seed, index), distinct across components and across theta -- so resets are
    independently randomized rather than copies of one stream, and a rerun of the same cohort
    reproduces the same placements and renders. ReaDDy's internal dynamics RNG has no exposed
    seed and stays OS-seeded (documented; dynamics are not bit-reproducible).
    """
    state = np.random.SeedSequence([int(master_seed), int(index)]).generate_state(
        2 * (1 + int(n_resets)), dtype=np.uint32)
    pairs = state.reshape(-1, 2)
    return {"cont": (int(pairs[0][0]), int(pairs[0][1])),
            "resets": [(int(p[0]), int(p[1])) for p in pairs[1:]]}


def _load_cohort(spec):
    """Read + verify the cohort file; returns its full contents as a dict."""
    path = spec["cohort_path"]
    if not path.exists():
        raise SystemExit(f"cohort file not found:\n    {path}\nRun --phase prepare first.")
    with np.load(str(path), allow_pickle=False) as d:
        cohort = {
            "theta_log10": np.asarray(d["theta_log10"], dtype=float),
            "keys": [str(k) for k in d["parameter_keys"]],
            "cohort_id": str(d["cohort_id"]),
            "master_seed": int(d["master_seed"]),
            "n_resets": int(d["n_resets"]),
            "window_seconds": float(d["window_seconds"]),
            "continuous_seconds": float(d["continuous_seconds"]),
        }
    if cohort["keys"] != spec["keys"]:
        raise SystemExit(f"cohort schema mismatch: stored keys {cohort['keys']} != "
                         f"current parameterization {spec['keys']}.")
    for name, stored, current in (
            ("--total-time-seconds", cohort["window_seconds"],
             spec["window"].total_time_seconds),
            ("--continuous-seconds", cohort["continuous_seconds"],
             spec["continuous"].total_time_seconds)):
        if abs(stored - current) > 1e-9:
            raise SystemExit(f"cohort was prepared with {name}={stored:g} but this run passed "
                             f"{current:g}; refusing a mixed-duration cohort.")
    expected = _cohort_digest(cohort["theta_log10"], cohort["keys"],
                              cohort["window_seconds"], cohort["continuous_seconds"],
                              cohort["n_resets"], cohort["master_seed"])
    if cohort["cohort_id"] != expected:
        raise SystemExit(f"cohort file digest mismatch (stored {cohort['cohort_id']}, "
                         f"recomputed {expected}); the file was altered or corrupted.")
    return cohort


def _verify_stamp(npz, cohort, index, what):
    """Assert a per-theta artifact belongs to THIS cohort (digest + exact theta row)."""
    stored_id = str(npz["cohort_id"]) if "cohort_id" in npz.files else "<unstamped>"
    if stored_id != cohort["cohort_id"]:
        raise SystemExit(
            f"{what} for theta {index:04d} carries cohort_id {stored_id}, but the current "
            f"cohort is {cohort['cohort_id']}: it belongs to another cohort. Remove the stale "
            f"cohort/*.npz files (or move them aside) before continuing.")
    if not np.allclose(np.asarray(npz["theta_log10"], dtype=float),
                       cohort["theta_log10"][index], atol=1e-12):
        raise SystemExit(f"{what} for theta {index:04d} stores a theta that differs from the "
                         f"cohort's row {index}; refusing the mixed artifact.")


def _index_range(args, n_theta):
    start = max(0, args.theta_start)
    stop = n_theta if args.theta_stop is None else min(args.theta_stop, n_theta)
    if start >= stop:
        raise SystemExit(f"empty theta range [{start}, {stop}) of a {n_theta}-theta cohort.")
    return start, stop


# =============================================================================
# Phase: prepare
# =============================================================================

def _phase_prepare(spec, args):
    """Draw the theta cohort from the training prior and persist it, once, with its identity."""
    stale = (sorted(spec["cohort_dir"].glob("theta_*.npz"))
             + sorted(spec["cohort_dir"].glob("result_*.npz"))
             if spec["cohort_dir"].exists() else [])
    if stale:
        raise SystemExit(
            f"the cohort directory already holds {len(stale)} per-theta file(s) from a previous "
            f"cohort:\n    {spec['cohort_dir']}\nA new cohort must not share a directory with "
            f"stale artifacts (they would be skipped-as-done and silently mixed). Remove them "
            f"first, e.g.:\n    rm {spec['cohort_dir']}/theta_*.npz "
            f"{spec['cohort_dir']}/result_*.npz")
    if spec["cohort_path"].exists() and not args.overwrite:
        raise SystemExit(f"cohort already exists:\n    {spec['cohort_path']}\n"
                         f"Pass --overwrite to replace it.")
    # The master seed is ALWAYS persisted: when none is supplied, one is drawn from OS entropy
    # here, so generation seeds are reproducible from the cohort file either way.
    master_seed = (int(args.seed) if args.seed is not None
                   else int(np.random.SeedSequence().entropy % (2 ** 31)))
    rng = np.random.default_rng(master_seed)
    theta_log10 = rng.uniform(spec["lower"], spec["upper"],
                              size=(args.n_theta, len(spec["keys"])))
    cohort_id = _cohort_digest(theta_log10, spec["keys"],
                               spec["window"].total_time_seconds,
                               spec["continuous"].total_time_seconds,
                               args.n_resets, master_seed)
    spec["out_dir"].mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(spec["cohort_path"]),
        theta_log10=theta_log10,
        parameter_keys=np.array(spec["keys"]),
        prior_low=spec["lower"], prior_high=spec["upper"],
        master_seed=master_seed,
        n_resets=args.n_resets,
        window_seconds=spec["window"].total_time_seconds,
        continuous_seconds=spec["continuous"].total_time_seconds,
        cohort_id=cohort_id)
    print(f"Cohort of {args.n_theta} theta drawn from the training prior and saved:\n"
          f"    {spec['cohort_path']}")
    print(f"  cohort_id  : {cohort_id}   (stamped into every artifact; mismatches are refused)")
    print(f"  master seed: {master_seed}"
          + ("" if args.seed is not None else "   (drawn from OS entropy and persisted)"))
    print(f"  window     : {spec['window'].total_time_seconds:g}s "
          f"({spec['window'].frame_count} frames)")
    print(f"  continuous : {spec['continuous'].total_time_seconds:g}s "
          f"({spec['continuous'].frame_count} frames = {spec['n_windows']} windows)")
    print(f"  resets     : {args.n_resets} per theta")


# =============================================================================
# Phase: generate
# =============================================================================

def _simulate_and_render(theta_physical, timing, imaging_physical, traj_path,
                         placement_seed, render_seed, verbose):
    """One simulation at ``theta_physical`` for ``timing``, rendered and converted to uint8.

    ``placement_seed`` seeds the initial particle placement; ``render_seed`` the imaging noise
    stream -- distinct per (theta, arm, replicate), so replicates are independently randomized.
    ReaDDy's internal dynamics RNG is not seedable through this interface and stays OS-seeded.
    Returns ``(video_uint8, counts)``; the trajectory file is the caller's to keep or delete.
    """
    import readdy
    import warnings
    from .io import convert_video_dtype
    from .simulation_dli_support import render_dli_video

    stem = build_system(theta_physical, pure_diffusion=False, verbose=verbose)
    smut = build_simulation(stem, theta_physical, seed=placement_seed, verbose=verbose)
    if traj_path.exists():
        traj_path.unlink()
    smut.output_file = str(traj_path)
    smut.progress_output_stride = timing.total_steps
    smut.run(n_steps=timing.total_steps,
             timestep=timing.delta_time_nanoseconds * readdy.units.nanosecond,
             show_summary=False)
    tray = readdy.Trajectory(filename=str(traj_path))
    tray_poses, dimer_mask = extract_trajectory_poses(tray, return_dimer_mask=True,
                                                      verbose=verbose)
    if tray_poses.shape[0] != timing.frame_count:
        raise RuntimeError(f"trajectory holds {tray_poses.shape[0]} frames but the run "
                           f"declares {timing.frame_count}.")
    counts = ha.species_counts_per_frame(tray_poses)
    with warnings.catch_warnings():   # absent particles are NaN by design (mirror Simulation_DLI)
        warnings.filterwarnings("ignore", message="All-NaN slice encountered",
                                category=RuntimeWarning)
        pro_tray_poses = np.nanmax(a=tray_poses, axis=3)
    # Release the ReaDDy CPU kernel (its worker-thread pool and the observable/output handles) and
    # reclaim memory, exactly as the canonical Simulation_RDS stage does between simulations: each
    # ReaDDy run otherwise leaves its thread pool behind, and this function is called
    # ``1 + n_resets`` times per theta, so a worker covering several theta accumulates thousands of
    # threads and eventually stalls against the memory cap. Done HERE -- after the trajectory is in
    # numpy but before the render, which is ~94% of this function's runtime -- so the kernel is not
    # held for the duration of the render either. ``smut`` is the sole reference to the ReaDDy
    # system, so deleting it releases both.
    del tray, smut, stem, tray_poses
    gc.collect()
    frames = render_dli_video(pro_tray_poses=pro_tray_poses,
                              imaging_physical=imaging_physical,
                              dimer_mask=dimer_mask, seed=render_seed, verbose=verbose)
    video = np.moveaxis(frames, 2, 0)                     # (H, W, frames) -> (frames, H, W)
    video = convert_video_dtype(video, bits_from=16, bits_to=8)
    return video, counts


def _phase_generate(spec, args):
    """Per theta: one continuous simulation + R resets, independently seeded and persisted."""
    from .posterior_predictive_video_runner import biology_fixed_imaging

    cohort = _load_cohort(spec)
    theta_log10 = cohort["theta_log10"]
    n_resets = cohort["n_resets"]
    if args.n_resets is not None and args.n_resets != n_resets:
        print(f"NOTE: --n-resets={args.n_resets} ignored; the cohort was prepared with "
              f"{n_resets} resets/theta and generation follows the cohort.")
    start, stop = _index_range(args, theta_log10.shape[0])
    imaging_physical, imaging_desc = biology_fixed_imaging(
        spec["data_bank_root"], spec["window"].label)
    print(f"Imaging pinned for every render (a MET-conditioned, training-supported imaging "
          f"slice): {imaging_desc}")
    print(f"Cohort {cohort['cohort_id']} | master seed {cohort['master_seed']} | "
          f"per-arm seeds spawned via SeedSequence (ReaDDy dynamics OS-seeded).")
    spec["cohort_dir"].mkdir(parents=True, exist_ok=True)
    traj_dir = spec["out_dir"] / "trajectories"
    traj_dir.mkdir(parents=True, exist_ok=True)

    for index in range(start, stop):
        out_path = _theta_file(spec, index)
        if out_path.exists():
            with np.load(str(out_path), allow_pickle=False) as d:
                _verify_stamp(d, cohort, index, "generated file")
            if not args.overwrite:
                print(f"[{index:04d}] exists (cohort verified), skipping "
                      f"(pass --overwrite to redo).")
                continue
        theta_physical = np.power(10.0, theta_log10[index])
        seeds = _spawn_seeds(cohort["master_seed"], index, n_resets)
        t0 = time.time()
        traj_path = traj_dir / f"theta_{index:04d}_continuous.h5"
        cont_video, cont_counts = _simulate_and_render(
            theta_physical, spec["continuous"], imaging_physical, traj_path,
            seeds["cont"][0], seeds["cont"][1], args.verbose)
        if not args.keep_trajectories:
            traj_path.unlink(missing_ok=True)
        t_cont = time.time() - t0

        reset_videos, reset_counts = [], []
        for r in range(n_resets):
            traj_path = traj_dir / f"theta_{index:04d}_reset_{r}.h5"
            video, counts = _simulate_and_render(
                theta_physical, spec["window"], imaging_physical, traj_path,
                seeds["resets"][r][0], seeds["resets"][r][1], args.verbose)
            if not args.keep_trajectories:
                traj_path.unlink(missing_ok=True)
            reset_videos.append(video)
            reset_counts.append(counts)
        t_all = time.time() - t0

        seed_table = np.array([seeds["cont"]] + seeds["resets"], dtype=np.uint32)
        np.savez_compressed(
            str(out_path),
            cont_video=cont_video,                                  # (F, H, W) uint8
            cont_counts=cont_counts.astype(np.int32),               # (F, 3) A,B,C per frame
            reset_videos=np.stack(reset_videos),                    # (R, f, H, W) uint8
            reset_counts=np.stack(reset_counts).astype(np.int32),   # (R, f, 3)
            theta_log10=theta_log10[index],
            imaging_physical=imaging_physical,
            imaging_desc=str(imaging_desc),
            seeds=seed_table,                                       # (1+R, 2) placement, render
            cohort_id=cohort["cohort_id"],
            n_resets=n_resets,
            window_seconds=spec["window"].total_time_seconds,
            continuous_seconds=spec["continuous"].total_time_seconds)
        print(f"[{index:04d}] continuous {t_cont:.1f}s + {n_resets} resets -> "
              f"{t_all:.1f}s total | counts t0={cont_counts[0].tolist()} "
              f"tEnd={cont_counts[-1].tolist()} | {out_path.name}", flush=True)


# =============================================================================
# Phase: infer
# =============================================================================

def _phase_infer(spec, args):
    """Run the trained estimator over every stored window: quantiles + raw draws."""
    import torch
    from . import artifacts
    from .evaluation import posterior_summary
    from .inference_support import resolve_topology

    cohort = _load_cohort(spec)
    theta_log10 = cohort["theta_log10"]
    start, stop = _index_range(args, theta_log10.shape[0])
    if not spec["estimator_path"].exists():
        raise SystemExit(f"estimator artifact not found:\n    {spec['estimator_path']}")
    estimator_digest = _file_digest(spec["estimator_path"])
    if args.seed is not None:
        torch.manual_seed(args.seed)      # applies to the draw stream of THIS worker
    topo = resolve_topology()
    device = topo.device
    vista_device = torch.device("cpu")
    torch.set_float32_matmul_precision("high")
    posterior = artifacts.load_estimator(spec["estimator_path"], device=str(device),
                                         expected_parameter_keys=spec["keys"])
    posterior.posterior_estimator.to(device)
    if device.type == "cuda" and args.pool_mode == "bounded":
        posterior.prior = spec["build_prior"](device=str(device))
    eval_cfg = PARAMETERS.inference.evaluation
    n_samples = args.posterior_samples or eval_cfg.posterior_samples
    window_frames = spec["window"].frame_count
    print(f"Estimator: {spec['estimator_path'].name} [{estimator_digest}] on {device} | "
          f"{n_samples} draws/window, pool_mode={args.pool_mode} | "
          f"cohort {cohort['cohort_id']}")

    for index in range(start, stop):
        in_path = _theta_file(spec, index)
        out_path = _result_file(spec, index)
        if not in_path.exists():
            print(f"[{index:04d}] no generated file, skipping.")
            continue
        if out_path.exists() and not args.overwrite:
            with np.load(str(out_path), allow_pickle=False) as d:
                _verify_stamp(d, cohort, index, "result file")
            print(f"[{index:04d}] result exists (cohort verified), skipping "
                  f"(pass --overwrite to redo).")
            continue
        t0 = time.time()
        with np.load(str(in_path), allow_pickle=False) as d:
            _verify_stamp(d, cohort, index, "generated file")
            cont_video = np.asarray(d["cont_video"])
            reset_videos = np.asarray(d["reset_videos"])
        starts = ha.window_starts(cont_video.shape[0], window_frames, window_frames)
        windows = [cont_video[s:s + window_frames] for s in starts]
        windows += [reset_videos[r] for r in range(reset_videos.shape[0])]
        which = np.array([0] * len(starts) + [1] * reset_videos.shape[0])   # 0=cont, 1=reset
        position = np.array(list(range(len(starts))) + list(range(reset_videos.shape[0])))
        quantile_rows, draw_rows = [], []
        for window in windows:
            summary, cloud = posterior_summary(
                posterior, window, device, vista_device, n_samples,
                eval_cfg.theta_prex_batch_size, pool_mode=args.pool_mode,
                return_samples=True)
            quantile_rows.append(summary)
            draw_rows.append(cloud)
        np.savez_compressed(
            str(out_path),
            post_q=np.stack(quantile_rows).astype(np.float32),      # (N, D, 5) log10
            draws=np.stack(draw_rows).astype(np.float32),           # (N, S, D) log10
            which=which, position=position,
            theta_log10=theta_log10[index],
            cohort_id=cohort["cohort_id"],
            pool_mode=str(args.pool_mode), n_samples=int(n_samples),
            window_frames=int(window_frames),
            estimator_name=spec["estimator_path"].name,
            estimator_digest=estimator_digest)
        print(f"[{index:04d}] {len(windows)} windows in {time.time() - t0:.1f}s "
              f"-> {out_path.name}", flush=True)


# =============================================================================
# Phase: analyze
# =============================================================================

def _collect(spec, cohort):
    """Assemble the per-trajectory arrays from every verified (theta, result) file pair.

    Every file's cohort stamp and stored theta are checked against the current cohort, and the
    result files' inference configuration (pool mode, draw count, estimator) must agree across
    trajectories -- a partial or mixed cohort is refused, never silently averaged.
    """
    theta_log10 = cohort["theta_log10"]
    window_frames = spec["window"].frame_count
    rows = {k: [] for k in ("theta", "cont_q", "reset_q", "cont_draws", "reset_draws",
                            "cont_true_start", "cont_true_mean", "cont_true_end",
                            "reset_true_start", "reset_true_mean",
                            "cont_exceed", "reset_exceed")}
    used, inference_config = [], None
    for index in range(theta_log10.shape[0]):
        t_path, r_path = _theta_file(spec, index), _result_file(spec, index)
        if not (t_path.exists() and r_path.exists()):
            continue
        with np.load(str(t_path), allow_pickle=False) as d:
            _verify_stamp(d, cohort, index, "generated file")
            cont_counts = np.asarray(d["cont_counts"], dtype=float)
            reset_counts = np.asarray(d["reset_counts"], dtype=float)
        with np.load(str(r_path), allow_pickle=False) as d:
            _verify_stamp(d, cohort, index, "result file")
            config = (str(d["pool_mode"]), int(d["n_samples"]),
                      str(d["estimator_digest"])) if "pool_mode" in d.files else None
            post_q = np.asarray(d["post_q"], dtype=float)
            draws = np.asarray(d["draws"], dtype=float)
            which = np.asarray(d["which"])
        if inference_config is None:
            inference_config = config
        elif config != inference_config:
            raise SystemExit(f"result file for theta {index:04d} was inferred under "
                             f"{config}, others under {inference_config}; refusing the "
                             f"mixed-inference cohort.")
        cont_sel, reset_sel = which == 0, which == 1
        starts = ha.window_starts(cont_counts.shape[0], window_frames, window_frames)
        true = ha.window_true_counts(cont_counts, starts, window_frames)
        r_true = ha.window_true_counts(
            reset_counts.reshape(-1, reset_counts.shape[-1]),
            np.arange(0, reset_counts.shape[0] * reset_counts.shape[1],
                      reset_counts.shape[1]), reset_counts.shape[1])
        exceed = ha.prior_exceedance(draws, spec["lower"], spec["upper"])
        rows["theta"].append(theta_log10[index])
        rows["cont_q"].append(post_q[cont_sel])
        rows["reset_q"].append(post_q[reset_sel])
        rows["cont_draws"].append(draws[cont_sel])
        rows["reset_draws"].append(draws[reset_sel])
        rows["cont_true_start"].append(true["start"])
        rows["cont_true_mean"].append(true["mean"])
        rows["cont_true_end"].append(true["end"])
        rows["reset_true_start"].append(r_true["start"])
        rows["reset_true_mean"].append(r_true["mean"])
        rows["cont_exceed"].append(exceed[cont_sel])
        rows["reset_exceed"].append(exceed[reset_sel])
        used.append(index)
    if not used:
        raise SystemExit("no complete (generated + inferred) trajectories found; "
                         "run --phase generate and --phase infer first.")
    data = {k: np.stack(v) for k, v in rows.items()}
    data["inference_config"] = inference_config
    return data, used


def _recovery_artifact_path(spec, args):
    """Locate the held-out recovery artifact that calibrates the per-stratum margins.

    Explicit ``--recovery-artifact`` wins; otherwise the standard sibling location beside this
    audit's own output directory. Returns ``None`` when neither resolves, and the caller reports
    a skip rather than failing: the stratified table is a robustness addendum, not a primary.
    """
    if getattr(args, "recovery_artifact", None):
        return Path(args.recovery_artifact).expanduser()
    paths, label = spec["paths"], spec["window"].label
    name = f"{paths.project_alias}_{label}_MAP_Recovery"
    return spec["data_bank_root"] / paths.posit_subdir / name / f"{name}.npz"


def _fd_of_counts(counts):
    """Dimer-complex fraction from a (..., 3) counts array, via the shared composition kernel."""
    return pc.composition(counts)[..., pc.DIMER_INDEX]


def _fd_draw_stats(draws, count_index):
    """Per-window f_D mean and quantiles from the within-draw fraction distribution.

    ``draws`` is ``(N, S, D)`` log10. The fraction is formed INSIDE each draw (correlations
    intact), giving an ``(N, S)`` distribution; returns its mean and the [5, 25, 75, 95]
    percentile bands -- the f_D credible intervals whose coverage is judged against the
    start-state truth.
    """
    fd = _fd_of_counts(10.0 ** np.asarray(draws)[..., list(count_index)])    # (N, S)
    q = np.percentile(fd, [5, 25, 75, 95], axis=-1)                          # (4, N)
    return fd.mean(axis=-1), q


def _figure_error_vs_position(table, keys, cont_abs, reset_abs, count_index):
    """Per-parameter ABSOLUTE posterior-median error against window position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    n_params = len(keys)
    n_cols = 5
    n_rows = int(np.ceil(n_params / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows),
                             dpi=200, sharex=True)
    axes = np.atleast_2d(axes)
    labels = {e["KEY"]: (e.get("LABEL") or e["KEY"]) for e in table}
    for i, key in enumerate(keys):
        ax = axes[i // n_cols, i % n_cols]
        mean, lo, hi = ha.positional_summary(cont_abs[:, :, i])
        x = np.arange(mean.shape[0])
        r_mean, r_lo, r_hi = ha.bootstrap_mean_ci(
            np.nanmean(reset_abs[:, :, i], axis=1, keepdims=True))
        ax.axhspan(float(r_lo[0]), float(r_hi[0]), color="tab:gray", alpha=0.25,
                   label="reset mean CI")
        ax.axhline(float(r_mean[0]), color="tab:gray", lw=1.0)
        ax.fill_between(x, lo, hi, color="tab:blue", alpha=0.2)
        ax.plot(x, mean, "-o", ms=3, color="tab:blue", label="continuous")
        marker = " (count: vs t=0 label, reads state drift)" if i in count_index else ""
        ax.set_title(labels.get(key, key) + marker, fontsize=8)
        ax.tick_params(labelsize=7)
        if i == 0:
            ax.legend(fontsize=6)
    for j in range(n_params, n_rows * n_cols):
        axes[j // n_cols, j % n_cols].axis("off")
    fig.suptitle("ABSOLUTE posterior-median error (log10) vs window position -- truth = drawn "
                 "theta\n(count panels read STATE DRIFT against the stale t=0 label, not "
                 "estimator error; the state audit is the f_D figure)", fontsize=9)
    fig.supxlabel("window position", fontsize=9)
    fig.tight_layout()
    return fig


def _figure_coverage_vs_position(cont_c50, cont_c90, reset_c50, reset_c90,
                                 fd_cov, const_index):
    """Coverage against window position: constants (excl. R_ON) and f_D-vs-start separately."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=200, sharex=True)
    ci = list(const_index)
    for ax, cont, reset, fd_key, nominal, name in (
            (axes[0], cont_c50, reset_c50, "c50", 0.5, "50% interval (IQR)"),
            (axes[1], cont_c90, reset_c90, "c90", 0.9, "90% interval (Q05-Q95)")):
        mean_c = np.nanmean(cont[:, :, ci], axis=(0, 2))
        mean_r = float(np.nanmean(reset[:, :, ci]))
        x = np.arange(mean_c.shape[0])
        ax.axhline(nominal, color="black", lw=0.8, ls="--", label="nominal")
        ax.axhline(mean_r, color="tab:gray", lw=1.0, label="reset (constants)")
        ax.plot(x, mean_c, "-o", ms=3, color="tab:blue",
                label="continuous (constants, excl. $R_{ON}$)")
        ax.plot(x, np.nanmean(fd_cov["cont_" + fd_key], axis=0), "-s", ms=3,
                color="tab:orange", label="continuous $f_D$ vs start truth")
        ax.axhline(float(np.nanmean(fd_cov["reset_" + fd_key])), color="tab:orange",
                   lw=1.0, ls=":", label="reset $f_D$ vs start truth")
        ax.set_title(name, fontsize=9)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlabel("window position", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.legend(fontsize=6)
    axes[0].set_ylabel("coverage", fontsize=8)
    fig.suptitle("Credible-interval coverage vs window position -- constants and the state read "
                 "reported separately, never pooled", fontsize=10)
    fig.tight_layout()
    return fig


def _figure_dynamic_state(true_start, true_mean, true_end, inferred,
                          reset_true_start, reset_inferred):
    """The state audit: inferred f_D against the window-START truth across positions."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), dpi=200)
    x = np.arange(true_start.shape[1])
    for t in range(min(true_start.shape[0], 60)):     # faint spaghetti, capped for legibility
        axes[0].plot(x, 100 * true_start[t], color="tab:green", alpha=0.10, lw=0.7)
        axes[0].plot(x, 100 * inferred[t], color="tab:blue", alpha=0.10, lw=0.7)
    tm, tl, th = ha.positional_summary(100 * true_start)
    im, il, ih = ha.positional_summary(100 * inferred)
    axes[0].plot(x, tm, "-o", ms=3, color="tab:green", label="true (window START)")
    axes[0].fill_between(x, tl, th, color="tab:green", alpha=0.2)
    axes[0].plot(x, np.nanmean(100 * true_mean, axis=0), ls="--", lw=1.0, color="tab:green",
                 alpha=0.7, label="true (window mean; sensitivity)")
    axes[0].plot(x, np.nanmean(100 * true_end, axis=0), ls=":", lw=1.0, color="tab:green",
                 alpha=0.7, label="true (window end; sensitivity)")
    axes[0].plot(x, im, "-o", ms=3, color="tab:blue", label="inferred (posterior mean)")
    axes[0].fill_between(x, il, ih, color="tab:blue", alpha=0.2)
    axes[0].set_xlabel("window position", fontsize=8)
    axes[0].set_ylabel("dimer-complex fraction f_D (%)", fontsize=8)
    axes[0].set_title("the estimator reads the window-START state", fontsize=9)
    axes[0].legend(fontsize=6)
    axes[0].tick_params(labelsize=7)

    err_cont = 100 * (inferred - true_start)
    err_reset = 100 * (reset_inferred - reset_true_start)
    delta_abs = ha.paired_absolute_effect(err_cont[..., None], err_reset[..., None])[..., 0]
    em, el, eh = ha.positional_summary(delta_abs)
    axes[1].axhline(0.0, color="black", lw=0.8, ls="--")
    axes[1].plot(x, em, "-o", ms=3, color="tab:blue",
                 label="|cont err| - mean |reset err|")
    axes[1].fill_between(x, el, eh, color="tab:blue", alpha=0.2)
    axes[1].set_xlabel("window position", fontsize=8)
    axes[1].set_ylabel("paired absolute-error contrast (pp)", fontsize=8)
    axes[1].set_title("f_D degradation vs position (primary statistic)", fontsize=9)
    axes[1].legend(fontsize=7)
    axes[1].tick_params(labelsize=7)
    fig.suptitle("Dynamic state: the composition audited against the ACTUAL window-start "
                 "population", fontsize=10)
    fig.tight_layout()
    return fig


def _figure_exceedance(cont_exceed, reset_exceed):
    """Unrestricted-flow mass outside the training box against window position."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 3.4), dpi=200)
    mean, lo, hi = ha.positional_summary(100 * cont_exceed)
    x = np.arange(mean.shape[0])
    rm, rl, rh = ha.bootstrap_mean_ci(np.nanmean(100 * reset_exceed, axis=1, keepdims=True))
    ax.axhspan(float(rl[0]), float(rh[0]), color="tab:gray", alpha=0.25, label="reset mean CI")
    ax.axhline(float(rm[0]), color="tab:gray", lw=1.0)
    ax.fill_between(x, lo, hi, color="tab:red", alpha=0.2)
    ax.plot(x, mean, "-o", ms=3, color="tab:red", label="continuous")
    ax.set_xlabel("window position", fontsize=8)
    ax.set_ylabel("flow mass outside the training box (%)", fontsize=8)
    ax.set_title("Unrestricted-flow mass outside the training box vs window position\n"
                 "(a bounded-prior posterior cannot have such mass; this reads flow leakage)",
                 fontsize=9)
    ax.legend(fontsize=7)
    ax.tick_params(labelsize=7)
    fig.tight_layout()
    return fig


def _phase_analyze(spec, args):
    """Aggregate the cohort into the audit statistics, the report, and the figures."""
    cohort = _load_cohort(spec)
    data, used = _collect(spec, cohort)
    keys, table, count_index = spec["keys"], spec["table"], spec["count_index"]
    const_index = [i for i, k in enumerate(keys)
                   if i not in count_index and k != "relative_rate_dimerization"]
    n_traj = len(used)
    n_windows = data["cont_q"].shape[1]
    n_resets = data["reset_q"].shape[1]
    rng = np.random.default_rng(args.seed)
    margins = dict(_DEFAULT_MARGINS)
    if args.margin_fd is not None:
        margins["f_D"] = args.margin_fd
    if args.margin_da is not None:
        margins["diffusivity_alp"] = args.margin_da
    if args.margin_koff is not None:
        margins["rate_dissociation"] = args.margin_koff

    reporter = DiagnosticReporter(
        stage="Horizon_Audit", enabled=True, dump=True, dump_dir=spec["out_dir"],
        run_label=f"{spec['paths'].project_alias}_{spec['window'].label}",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))
    shutil.rmtree(spec["out_dir"] / "figures", ignore_errors=True)

    reporter.stat("cohort_id", cohort["cohort_id"],
                  note="every artifact used below carried this stamp; mismatches are refused.")
    if data["inference_config"]:
        pm, ns, ed = data["inference_config"]
        reporter.stat("inference_config", f"pool={pm}, draws={ns}, estimator {ed}",
                      note="verified identical across every result file.")
    reporter.stat("trajectories", n_traj,
                  note="complete (generated + inferred) theta draws; the independent unit.")
    reporter.stat("windows_per_trajectory", n_windows)
    reporter.stat("resets_per_trajectory", n_resets)

    # ---- signed and absolute errors vs the drawn theta (all 10 parameters) ----
    def _per_traj(fn):
        return np.stack([fn(t) for t in range(n_traj)])
    cont_err = _per_traj(lambda t: ha.quantile_errors(data["cont_q"][t], data["theta"][t])[0])
    reset_err = _per_traj(lambda t: ha.quantile_errors(data["reset_q"][t], data["theta"][t])[0])
    cont_c50 = _per_traj(
        lambda t: ha.quantile_errors(data["cont_q"][t], data["theta"][t])[1]).astype(float)
    cont_c90 = _per_traj(
        lambda t: ha.quantile_errors(data["cont_q"][t], data["theta"][t])[2]).astype(float)
    reset_c50 = _per_traj(
        lambda t: ha.quantile_errors(data["reset_q"][t], data["theta"][t])[1]).astype(float)
    reset_c90 = _per_traj(
        lambda t: ha.quantile_errors(data["reset_q"][t], data["theta"][t])[2]).astype(float)
    delta_abs = ha.paired_absolute_effect(cont_err, reset_err)      # (T, W, D)
    delta_signed = ha.paired_position_effect(cont_err, reset_err)   # secondary (bias)

    # ---- the state read: f_D from draws, judged against the window-START truth ----
    fd_cont_mean, fd_cont_q = zip(*(_fd_draw_stats(data["cont_draws"][t], count_index)
                                    for t in range(n_traj)))
    fd_reset_mean, fd_reset_q = zip(*(_fd_draw_stats(data["reset_draws"][t], count_index)
                                      for t in range(n_traj)))
    fd_cont = np.stack(fd_cont_mean)                                # (T, W)
    fd_reset = np.stack(fd_reset_mean)                              # (T, R)
    fd_cont_q = np.stack(fd_cont_q)                                 # (T, 4, W) [5,25,75,95]
    fd_reset_q = np.stack(fd_reset_q)                               # (T, 4, R)
    ts_cont = _fd_of_counts(data["cont_true_start"])                # (T, W)
    tm_cont = _fd_of_counts(data["cont_true_mean"])
    te_cont = _fd_of_counts(data["cont_true_end"])
    ts_reset = _fd_of_counts(data["reset_true_start"])              # (T, R)
    tm_reset = _fd_of_counts(data["reset_true_mean"])
    label_fd = _fd_of_counts(10.0 ** data["theta"][:, list(count_index)])   # unrounded label
    fd_cov = {
        "cont_c50": ((ts_cont >= fd_cont_q[:, 1]) & (ts_cont <= fd_cont_q[:, 2])).astype(float),
        "cont_c90": ((ts_cont >= fd_cont_q[:, 0]) & (ts_cont <= fd_cont_q[:, 3])).astype(float),
        "reset_c50": ((ts_reset >= fd_reset_q[:, 1]) & (ts_reset <= fd_reset_q[:, 2])).astype(float),
        "reset_c90": ((ts_reset >= fd_reset_q[:, 0]) & (ts_reset <= fd_reset_q[:, 3])).astype(float),
    }
    fd_err_cont = 100 * (fd_cont - ts_cont)                         # pp, vs START truth
    fd_err_reset = 100 * (fd_reset - ts_reset)
    fd_delta_abs = ha.paired_absolute_effect(fd_err_cont[..., None],
                                             fd_err_reset[..., None])[..., 0]   # (T, W)

    # ---- PRIMARY TABLE: predeclared estimands, three-outcome verdicts ----
    primaries = [
        ("f_D (state, vs window-start truth)", "f_D", "pp",
         np.abs(fd_err_cont), np.abs(fd_err_reset), fd_delta_abs),
    ]
    for key in ("diffusivity_alp", "rate_dissociation"):
        i = keys.index(key)
        primaries.append((f"{key} (constant, vs theta)", key, "dex",
                          np.abs(cont_err[:, :, i]), np.abs(reset_err[:, :, i]),
                          delta_abs[:, :, i]))
    headers = ["primary estimand", "reset MAE", "cont MAE (w0)",
               f"cont MAE (w{n_windows - 1})", "paired |err| delta (last w)", "95% CI",
               "margin", "verdict"]
    rows, verdicts = [], {}
    for label, mkey, unit, cont_abs_e, reset_abs_e, d_abs in primaries:
        d_mean, d_lo, d_hi = ha.bootstrap_mean_ci(d_abs[:, -1:], rng=rng)
        verdict = (ha.equivalence_verdict(d_lo[0], d_hi[0], margins[mkey])
                   if n_traj >= 2 else "not evaluable (T=1)")
        verdicts[mkey] = verdict
        rows.append([label,
                     f"{float(np.nanmean(reset_abs_e)):.3f} {unit}",
                     f"{float(np.nanmean(cont_abs_e[:, 0])):.3f}",
                     f"{float(np.nanmean(cont_abs_e[:, -1])):.3f}",
                     f"{float(d_mean[0]):+.3f}",
                     f"[{float(d_lo[0]):+.3f}, {float(d_hi[0]):+.3f}]",
                     f"{margins[mkey]:g} {unit}", verdict])
    reporter.table(
        "PRIMARY audit: paired absolute-error contrast with prespecified equivalence margins",
        headers, rows,
        note="the contrast is |continuous error| minus the same trajectory's mean |reset error| "
             "at the LAST window (positive = continuous worse); 'equivalent' requires the whole "
             "CI inside (-margin, +margin) -- a CI merely covering zero is 'inconclusive'. "
             "Margins default to the estimator's own in-model error on the 10,000-video "
             "held-out recovery.")

    # ---- first-window exchangeability gate ----
    if n_traj >= 2:
        gate_ok = True
        details = []
        for label, mkey, unit, _c, _r, d_abs in primaries:
            g_mean, g_lo, g_hi = ha.bootstrap_mean_ci(d_abs[:, :1], rng=rng)
            inside = ha.equivalence_verdict(g_lo[0], g_hi[0], margins[mkey]) == "equivalent"
            gate_ok &= inside
            details.append(f"{mkey}: {float(g_mean[0]):+.3f} "
                           f"[{float(g_lo[0]):+.3f}, {float(g_hi[0]):+.3f}] {unit}")
        reporter.check(
            "first_window_exchangeable", gate_ok, "; ".join(details), fatal=False,
            note="continuous window 0 has no inherited past, so it must be statistically "
                 "exchangeable with the resets BEFORE later-window differences are attributed "
                 "to horizon; failure here means an arm-construction artifact, not horizon.")
    else:
        reporter.stat("first_window_exchangeable", "not evaluable",
                      note="a single trajectory has no bootstrap CI.")

    # ---- stratified margins: are the verdicts an artifact of a prior-averaged yardstick? ----
    # The estimator's error depends on the true value, so a single prior-averaged margin can hide
    # a degradation in one regime behind easy performance in another. Recompute BOTH sides inside
    # fixed thirds of the prior range: the margin from the held-out recovery set restricted to the
    # stratum, the contrast from cohort trajectories in the same stratum.
    rec_path = _recovery_artifact_path(spec, args)
    if rec_path is None or not rec_path.exists():
        reporter.stat("stratified_margins", "skipped",
                      note=f"held-out recovery artifact not found"
                           f"{'' if rec_path is None else f' at {rec_path}'}; pass "
                           f"--recovery-artifact to enable the per-stratum robustness table.")
    else:
        rec = np.load(rec_path, allow_pickle=True)
        rec_true = np.asarray(rec["true_log10"], dtype=float)
        rec_med = np.asarray(rec["posterior_quantiles"], dtype=float)[:, :, 2]   # marginal median
        rec_abs = np.abs(rec_med - rec_true)
        strat_headers = ["stratum (true value)", "n cohort", "n recovery", "local margin",
                         f"paired |err| delta (w{n_windows - 1})", "95% CI", "x margin",
                         "signed shift", "verdict"]
        strat_rows = []
        for key in ("diffusivity_alp", "rate_dissociation"):
            i = keys.index(key)
            lo_b, hi_b = float(spec["lower"][i]), float(spec["upper"][i])
            coh_lab, edges = ha.stratify_by_true_value(data["theta"][:, i], lo_b, hi_b)
            rec_lab, _ = ha.stratify_by_true_value(rec_true[:, i], lo_b, hi_b)
            table_rows = ha.stratified_margin_table(
                delta_abs[:, -1, i], coh_lab, rec_abs[:, i], rec_lab, rng=rng)
            for s, row in enumerate(table_rows):
                # signed shift = continuous minus reset baseline, at the same theta: the
                # horizon-specific bias with each arm's generic prior shrinkage differenced out.
                sel = coh_lab == s
                shift = (float(np.nanmean(cont_err[sel, -1, i]))
                         - float(np.nanmean(np.nanmean(reset_err[sel, :, i], axis=1)))
                         ) if sel.any() else np.nan
                strat_rows.append([
                    f"{key} {10 ** edges[s]:.3g}-{10 ** edges[s + 1]:.3g}",
                    row["n_cohort"], row["n_recovery"], f"{row['margin']:.3f} dex",
                    f"{row['mean']:+.3f}", f"[{row['ci_lo']:+.3f}, {row['ci_hi']:+.3f}]",
                    f"{row['ratio']:.1f}x", f"{100 * (10 ** shift - 1):+.0f}%", row["verdict"]])
        # f_D: strata in composition space (prespecified bins), margin point-composed from the
        # recovery medians -- the audit's own f_D margin is formed INSIDE each draw, so the two
        # differ slightly; the prior-averaged point-composed value is reported for comparison.
        rec_fd_true = _fd_of_counts(10.0 ** rec_true[:, list(count_index)])
        rec_fd_med = _fd_of_counts(10.0 ** rec_med[:, list(count_index)])
        fd_edges = np.array([0.0, 0.2, 0.8, 1.0])
        fd_coh_lab = np.clip(np.digitize(ts_cont[:, -1], fd_edges[1:-1]), 0, 2)
        fd_rec_lab = np.clip(np.digitize(rec_fd_true, fd_edges[1:-1]), 0, 2)
        # fd_delta_abs is ALREADY in percentage points (fd_err_cont is 100 * fraction error);
        # the recovery fractions are not, hence the asymmetric scaling below. Cross-check: the
        # cohort-wide value of this statistic is the f_D row of the PRIMARY table.
        fd_rows = ha.stratified_margin_table(
            fd_delta_abs[:, -1], fd_coh_lab,
            100 * np.abs(rec_fd_med - rec_fd_true), fd_rec_lab, rng=rng)
        for s, row in enumerate(fd_rows):
            sel = fd_coh_lab == s
            shift = 100 * float(np.nanmean((fd_cont[:, -1] - ts_cont[:, -1])[sel])) \
                if sel.any() else np.nan
            strat_rows.append([
                f"f_D start {fd_edges[s]:g}-{fd_edges[s + 1]:g}",
                row["n_cohort"], row["n_recovery"], f"{row['margin']:.2f} pp",
                f"{row['mean']:+.2f}", f"[{row['ci_lo']:+.2f}, {row['ci_hi']:+.2f}]",
                f"{row['ratio']:.1f}x", f"{shift:+.1f} pp", row["verdict"]])
        reporter.table(
            "STRATIFIED robustness: per-stratum margins from the same held-out recovery set",
            strat_headers, strat_rows,
            note="strata are FIXED thirds of the prior range (prespecified geometry, not "
                 "data-dependent quantiles; the prior is uniform in log10 so thirds carry equal "
                 "expected mass), f_D in prespecified composition bins. The margin is the "
                 "estimator's own held-out error for true values IN THAT STRATUM, so each "
                 "verdict compares like with like. 'signed shift' is continuous minus the reset "
                 "baseline at the same theta -- the horizon-specific bias, with each arm's "
                 "generic prior shrinkage differenced out. A stratum thinly populated in the "
                 "cohort is reported with its n, never dropped. The f_D margins here are "
                 f"point-composed from recovery medians (prior-averaged "
                 f"{100 * float(np.nanmean(np.abs(rec_fd_med - rec_fd_true))):.2f} pp) while the "
                 f"audit's f_D is composed INSIDE each draw (declared margin "
                 f"{margins['f_D']:g} pp); the small offset is stated, not silently mixed.")

    # ---- exploratory constants (no verdicts; multiplicity) ----
    exp_headers = ["parameter", "reset MAE", f"cont MAE (w{n_windows - 1})",
                   "paired |err| delta (last w)", "signed delta (bias)", "slope/window"]
    exp_rows = []
    for i, key in enumerate(keys):
        if key in ("diffusivity_alp", "rate_dissociation"):
            continue
        tag = (" (count: vs t=0 label -- reads state drift)" if i in count_index
               else (" (unidentified)" if key == "relative_rate_dimerization" else ""))
        d_mean, _, _ = ha.bootstrap_mean_ci(delta_abs[:, -1:, i], rng=rng)
        s_mean, _, _ = ha.bootstrap_mean_ci(delta_signed[:, -1:, i], rng=rng)
        slopes = ha.trajectory_slopes(cont_err[:, :, i])
        exp_rows.append([key + tag,
                         f"{float(np.nanmean(np.abs(reset_err[:, :, i]))):.3f}",
                         f"{float(np.nanmean(np.abs(cont_err[:, -1, i]))):.3f}",
                         f"{float(d_mean[0]):+.3f}",
                         f"{float(s_mean[0]):+.3f}",
                         f"{float(np.nanmean(slopes)):+.4f}"])
    reporter.table(
        "Exploratory parameters (no verdicts: not predeclared, no multiplicity control)",
        exp_headers, exp_rows,
        note="count rows compare against the stale t=0 label and therefore read STATE DRIFT, "
             "not estimator error -- the state audit is the f_D primary above. R_ON is already "
             "known unidentified and never contributes to any verdict.")

    # ---- coverage, disaggregated ----
    cov_headers = ["estimand (truth)", "reset c50 / c90",
                   "cont w0 c50 / c90", f"cont w{n_windows - 1} c50 / c90"]
    cov_rows = [["constants excl. R_ON (theta)",
                 f"{np.nanmean(reset_c50[:, :, const_index]):.2f} / "
                 f"{np.nanmean(reset_c90[:, :, const_index]):.2f}",
                 f"{np.nanmean(cont_c50[:, 0, const_index]):.2f} / "
                 f"{np.nanmean(cont_c90[:, 0, const_index]):.2f}",
                 f"{np.nanmean(cont_c50[:, -1, const_index]):.2f} / "
                 f"{np.nanmean(cont_c90[:, -1, const_index]):.2f}"],
                ["f_D (window-START truth, from draws)",
                 f"{np.nanmean(fd_cov['reset_c50']):.2f} / {np.nanmean(fd_cov['reset_c90']):.2f}",
                 f"{np.nanmean(fd_cov['cont_c50'][:, 0]):.2f} / "
                 f"{np.nanmean(fd_cov['cont_c90'][:, 0]):.2f}",
                 f"{np.nanmean(fd_cov['cont_c50'][:, -1]):.2f} / "
                 f"{np.nanmean(fd_cov['cont_c90'][:, -1]):.2f}"],
                ["counts (STALE t=0 truth -- state drift, expected to fail late)",
                 f"{np.nanmean(reset_c50[:, :, list(count_index)]):.2f} / "
                 f"{np.nanmean(reset_c90[:, :, list(count_index)]):.2f}",
                 f"{np.nanmean(cont_c50[:, 0, list(count_index)]):.2f} / "
                 f"{np.nanmean(cont_c90[:, 0, list(count_index)]):.2f}",
                 f"{np.nanmean(cont_c50[:, -1, list(count_index)]):.2f} / "
                 f"{np.nanmean(cont_c90[:, -1, list(count_index)]):.2f}"]]
    reporter.table("Interval coverage, disaggregated by estimand kind (never pooled)",
                   cov_headers, cov_rows,
                   note="nominal 0.50 / 0.90. The counts row is diagnostic context only: its "
                        "late-window failure against the stale t=0 truth measures how far the "
                        "state drifted, not calibration.")

    # ---- f_D truth-reference sensitivity ----
    sens_rows = [
        ["window START (PRIMARY)", f"{np.nanmean(np.abs(fd_err_reset)):.2f}",
         f"{np.nanmean(np.abs(fd_err_cont)):.2f}"],
        ["window mean (sensitivity)",
         f"{np.nanmean(np.abs(100 * (fd_reset - tm_reset))):.2f}",
         f"{np.nanmean(np.abs(100 * (fd_cont - tm_cont))):.2f}"],
        ["window end (sensitivity)", "-",
         f"{np.nanmean(np.abs(100 * (fd_cont - te_cont))):.2f}"],
        ["unrounded theta label (sensitivity, resets only)",
         f"{np.nanmean(np.abs(100 * (fd_reset - label_fd[:, None]))):.2f}", "-"]]
    reporter.table("f_D error (pp MAE) under each truth reference",
                   ["truth reference", "reset", "continuous (all windows)"], sens_rows,
                   note="the estimator's training labels are window-start populations, so START "
                        "is the primary reference; the spread across references measures how "
                        "much the truth choice alone moves the number (the estimand-mismatch "
                        "hazard). The unrounded-label row isolates the count-rounding effect.")

    # ---- state-support drift + flow mass outside the training box ----
    lo_c = 10.0 ** spec["lower"][list(count_index)]
    hi_c = 10.0 ** spec["upper"][list(count_index)]
    outside = ((data["cont_true_start"] < lo_c) | (data["cont_true_start"] > hi_c))
    reporter.stat("true start-counts outside the training count box (%)",
                  f"{100 * float(outside.any(axis=2).mean()):.2f}",
                  note="fraction of continuous windows whose TRUE start population leaves "
                       "[10^0, 10^2.5] in any species -- state-support drift, distinct from "
                       "any estimator behavior.")
    reporter.stat("flow mass outside the training box, reset (%)",
                  f"{100 * float(np.nanmean(data['reset_exceed'])):.2f}",
                  note="unrestricted-flow draws outside the training box; a bounded-prior "
                       "posterior cannot have such mass -- this reads flow leakage, the "
                       "deployment gate of the experimental analysis.")
    reporter.stat("flow mass outside the training box, continuous last window (%)",
                  f"{100 * float(np.nanmean(data['cont_exceed'][:, -1])):.2f}")

    # ---- exploratory stratification by the initial state ----
    if n_traj >= 6:
        strata = [("initial f_D < 0.4", label_fd < 0.4),
                  ("0.4 <= initial f_D <= 0.7", (label_fd >= 0.4) & (label_fd <= 0.7)),
                  ("initial f_D > 0.7", label_fd > 0.7)]
        srows = []
        for name, mask in strata:
            if mask.sum() == 0:
                srows.append([name, "0", "-"])
                continue
            srows.append([name, str(int(mask.sum())),
                          f"{float(np.nanmean(fd_delta_abs[mask, -1])):+.2f}"])
        reporter.table("Exploratory: last-window f_D degradation by initial composition",
                       ["stratum", "n trajectories", "mean paired |err| delta (pp)"], srows,
                       note="a prior-wide mean can hide failure in the dimer-rich region the "
                            "activated experimental condition occupies; exploratory (no CI "
                            "shown below n=6 per stratum, no verdicts).")

    # ---- figures ----
    reporter.save_figure(
        "error_vs_position",
        _figure_error_vs_position(table, keys, np.abs(cont_err), np.abs(reset_err),
                                  count_index),
        caption="Absolute posterior-median error (log10, truth = drawn theta) against window "
                "position: continuous (line, trajectory-bootstrap CI) against the reset mean "
                "(gray band). Count panels read state drift against the stale t=0 label.")
    reporter.save_figure(
        "coverage_vs_position",
        _figure_coverage_vs_position(cont_c50, cont_c90, reset_c50, reset_c90,
                                     fd_cov, const_index),
        caption="Coverage against window position, disaggregated: constants (excluding the "
                "unidentified R_ON) against theta, and f_D against the window-start truth.")
    reporter.save_figure(
        "dynamic_state_f_D",
        _figure_dynamic_state(ts_cont, tm_cont, te_cont, fd_cont, ts_reset, fd_reset),
        caption="The state audit: inferred f_D against the actual window-START population "
                "(primary), with the window-mean and window-end truths as sensitivity "
                "references, and the paired absolute-error contrast per position.")
    reporter.save_figure(
        "flow_mass_outside_box",
        _figure_exceedance(data["cont_exceed"], data["reset_exceed"]),
        caption="Unrestricted-flow mass outside the training box against window position, "
                "continuous vs reset.")

    # ---- persist the aggregated arrays for downstream use ----
    arrays_path = spec["out_dir"] / (spec["out_dir"].name + ".npz")
    np.savez_compressed(
        str(arrays_path),
        cohort_id=cohort["cohort_id"],
        theta_log10=data["theta"], trajectory_indices=np.array(used),
        cont_err=cont_err, reset_err=reset_err,
        delta_abs=delta_abs, delta_signed=delta_signed,
        cont_c50=cont_c50, cont_c90=cont_c90, reset_c50=reset_c50, reset_c90=reset_c90,
        fd_cont=fd_cont, fd_reset=fd_reset,
        fd_true_start_cont=ts_cont, fd_true_mean_cont=tm_cont, fd_true_end_cont=te_cont,
        fd_true_start_reset=ts_reset, fd_delta_abs=fd_delta_abs,
        cont_exceed=data["cont_exceed"], reset_exceed=data["reset_exceed"],
        parameter_keys=np.array(keys))
    print(f"Aggregated audit arrays saved to {arrays_path}")

    reporter.summary()
    reporter.write_report()


# =============================================================================
# Phase: selftest (pure python, no simulation, no GPU)
# =============================================================================

def _phase_selftest(spec, args):
    """Deterministic checks of the audit's decision-critical logic. Raises on any failure."""
    import tempfile

    # 1. Truth references: start/mean/end must be exact, and distinct when the state moves.
    counts = np.stack([np.arange(10, dtype=float) * 10,
                       np.full(10, 5.0), np.zeros(10)], axis=1)     # A ramps, B flat, C zero
    tr = ha.window_true_counts(counts, np.array([0, 5]), 5)
    assert np.allclose(tr["start"][:, 0], [0, 50]), tr["start"]
    assert np.allclose(tr["mean"][:, 0], [20, 70]), tr["mean"]
    assert np.allclose(tr["end"][:, 0], [40, 90]), tr["end"]
    assert not np.allclose(tr["start"], tr["mean"])                 # the blocker-1 distinction
    print("  [PASS] truth references: start / mean / end exact and distinct")

    # 2. Seed uniqueness + determinism across theta, arms, replicates.
    all_seeds = []
    for idx in range(6):
        s = _spawn_seeds(master_seed=1, index=idx, n_resets=3)
        assert s == _spawn_seeds(1, idx, 3)                          # deterministic
        all_seeds += [s["cont"]] + s["resets"]
    flat = [v for pair in all_seeds for v in pair]
    assert len(set(flat)) == len(flat), "seed collision across components"
    assert _spawn_seeds(1, 0, 3) != _spawn_seeds(2, 0, 3)            # master seed matters
    print(f"  [PASS] seeds: {len(flat)} components all distinct, deterministic in "
          f"(master, index)")

    # 3. Coverage disaggregation: constants covered, counts not -> pooling would mislead.
    D = 5
    post_q = np.zeros((1, D, 5))
    post_q[..., 0], post_q[..., 1], post_q[..., 2] = -1.0, -0.5, 0.0
    post_q[..., 3], post_q[..., 4] = 0.5, 1.0
    truth = np.zeros(D)
    truth[:2] = 5.0                                                  # "counts" far outside
    _, _, c90 = ha.quantile_errors(post_q, truth)
    assert c90[0, :2].mean() == 0.0 and c90[0, 2:].mean() == 1.0
    assert 0.0 < c90.mean() < 1.0                                    # the pooled artifact
    print("  [PASS] coverage: disaggregation separates 0% (stale-truth) from 100% (constants)")

    # 4. Stale-cohort rejection: an artifact stamped by another cohort must be refused.
    theta = np.zeros((2, len(spec["keys"])))
    cid = _cohort_digest(theta, spec["keys"], 2.0, 20.0, 3, 7)
    cid_other = _cohort_digest(theta + 0.1, spec["keys"], 2.0, 20.0, 3, 7)
    assert cid != cid_other
    cohort = {"cohort_id": cid, "theta_log10": theta}
    with tempfile.TemporaryDirectory() as tmp:
        stale = Path(tmp) / "theta_0000.npz"
        np.savez(stale, cohort_id=cid_other, theta_log10=theta[0])
        try:
            with np.load(str(stale), allow_pickle=False) as d:
                _verify_stamp(d, cohort, 0, "generated file")
            raise AssertionError("stale artifact was accepted")
        except SystemExit:
            pass
        fresh = Path(tmp) / "theta_0001.npz"
        np.savez(fresh, cohort_id=cid, theta_log10=theta[1] + 1e-3)  # right id, wrong theta
        try:
            with np.load(str(fresh), allow_pickle=False) as d:
                _verify_stamp(d, cohort, 1, "generated file")
            raise AssertionError("theta-mismatched artifact was accepted")
        except SystemExit:
            pass
    print("  [PASS] cohort identity: wrong digest and wrong theta both refused")

    # 5. Absolute vs signed contrast: symmetric error growth invisible to the signed statistic.
    reset_e = np.array([[[0.1], [-0.1]]])                            # (1, 2, 1): small, unbiased
    cont_e = np.array([[[2.0], [-2.0]]])                             # (1, 2, 1): large, unbiased
    signed = ha.paired_position_effect(cont_e, reset_e)
    abs_d = ha.paired_absolute_effect(cont_e, reset_e)
    assert abs(signed.mean()) < 1e-12                                # bias statistic reads zero
    assert abs_d.mean() > 1.8                                        # degradation is caught
    print("  [PASS] paired statistics: signed misses symmetric degradation, absolute catches it")

    # 6. Equivalence verdicts: three outcomes, and 'CI covers zero' is NOT equivalence.
    assert ha.equivalence_verdict(0.05, 0.30, margin=0.10) == "degraded"
    assert ha.equivalence_verdict(-0.04, 0.06, margin=0.10) == "equivalent"
    assert ha.equivalence_verdict(-0.30, 0.25, margin=0.10) == "inconclusive"
    assert ha.equivalence_verdict(np.nan, np.nan, margin=0.10) == "inconclusive"
    print("  [PASS] equivalence: degraded / equivalent / inconclusive; wide CI is inconclusive")

    # 7. Stratification: fixed prior-range thirds, and a per-stratum margin that a prior-averaged
    #    margin would hide. Construct an estimator that is precise where the truth is large and
    #    imprecise where it is small: pooled, the easy stratum masks the hard one.
    lab, edges = ha.stratify_by_true_value(np.array([-1.0, 0.0, 1.0, 2.0, -5.0]), -1.0, 1.0)
    assert list(lab) == [0, 1, 2, -1, -1]                 # endpoints: low closed, high closed
    assert np.allclose(edges, [-1.0, -1 / 3, 1 / 3, 1.0])
    rng_s = np.random.default_rng(0)
    rec_true = np.concatenate([np.full(300, -0.8), np.full(300, 0.0), np.full(300, 0.8)])
    rec_err = np.concatenate([np.full(300, 0.60), np.full(300, 0.10), np.full(300, 0.05)])
    rec_lab, _ = ha.stratify_by_true_value(rec_true, -1.0, 1.0)
    coh_true = np.concatenate([np.full(40, -0.8), np.full(40, 0.0), np.full(40, 0.8)])
    coh_lab, _ = ha.stratify_by_true_value(coh_true, -1.0, 1.0)
    contrast = np.concatenate([np.full(40, 0.30), np.full(40, 0.30), np.full(40, 0.30)])
    rows = ha.stratified_margin_table(contrast, coh_lab, rec_err, rec_lab, rng=rng_s)
    pooled_margin = float(np.mean(rec_err))                          # 0.25: hides both regimes
    assert rows[0]["verdict"] == "degraded" and rows[0]["margin"] > pooled_margin
    assert rows[2]["verdict"] == "degraded" and rows[2]["margin"] < pooled_margin
    assert rows[0]["ratio"] < 1.0 < rows[2]["ratio"]     # same effect, opposite practical weight
    empty = ha.stratified_margin_table(contrast, np.full(120, 1), rec_err, rec_lab, rng=rng_s)
    assert empty[0]["n_cohort"] == 0 and empty[0]["verdict"] == "inconclusive"
    print("  [PASS] stratification: prior-range thirds, per-stratum margins, empty stratum kept")
    print("SELFTEST OK: 7/7 check families passed.")


# =============================================================================
# Entry point + CLI
# =============================================================================

def run_horizon_audit(cfg, args):
    """Shared entry point. ``cfg`` is a WorkflowConfig; ``args`` the parsed CLI namespace."""
    spec = _horizon_audit_spec(cfg, args)
    if args.dry_run:
        n_theta, cid = "?", "-"
        if spec["cohort_path"].exists():
            with np.load(str(spec["cohort_path"]), allow_pickle=False) as d:
                n_theta = int(d["theta_log10"].shape[0])
                cid = str(d["cohort_id"]) if "cohort_id" in d.files else "<unstamped legacy>"
        print(f"[DRY RUN] horizon audit ({cfg.tag}) phase={args.phase}")
        print(f"    window     : {spec['window'].total_time_seconds:g}s "
              f"({spec['window'].frame_count} frames)")
        print(f"    continuous : {spec['continuous'].total_time_seconds:g}s "
              f"({spec['continuous'].frame_count} frames = {spec['n_windows']} windows)")
        print(f"    cohort     : {spec['cohort_path']}  "
              f"[{'OK, n=' + str(n_theta) + ', id=' + cid if spec['cohort_path'].exists() else 'ABSENT'}]")
        print(f"    estimator  : {spec['estimator_path']}  "
              f"[{'OK' if spec['estimator_path'].exists() else 'MISSING'}]")
        n_gen = len(list(spec["cohort_dir"].glob("theta_*.npz"))) if spec["cohort_dir"].exists() else 0
        n_res = len(list(spec["cohort_dir"].glob("result_*.npz"))) if spec["cohort_dir"].exists() else 0
        print(f"    generated  : {n_gen} theta file(s); inferred: {n_res} result file(s)")
        print(f"    outputs    : {spec['out_dir']}")
        print("[DRY RUN] nothing simulated, inferred, or written.")
        return 0
    phase = {"prepare": _phase_prepare, "generate": _phase_generate,
             "infer": _phase_infer, "analyze": _phase_analyze,
             "selftest": _phase_selftest}[args.phase]
    phase(spec, args)
    return 0


def build_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--total-time-seconds", type=float, required=True,
                   help="model window duration in seconds; must match the trained estimator "
                        "AND the prepared cohort.")
    p.add_argument("--continuous-seconds", type=float, default=20.0,
                   help="continuous-simulation length (default 20, matching the experimental "
                        "recordings); must tile into whole model windows and match the cohort.")
    p.add_argument("--phase", choices=("prepare", "generate", "infer", "analyze", "selftest"),
                   required=True,
                   help="prepare: draw + persist the stamped theta cohort. generate: simulate + "
                        "render (CPU; parallel over --theta-start/--theta-stop). infer: run the "
                        "estimator per window (GPU; same range splitting). analyze: statistics "
                        "+ report + figures. selftest: pure-python checks of the decision-"
                        "critical logic (no simulation, no GPU).")
    p.add_argument("--n-theta", type=int, default=200,
                   help="prepare: cohort size (default 200).")
    p.add_argument("--n-resets", type=int, default=None,
                   help="prepare: independently initialized model-window simulations per theta "
                        "(default 10). Later phases follow the cohort; a differing value passed "
                        "there is ignored with a note.")
    p.add_argument("--theta-start", type=int, default=0,
                   help="generate/infer: first theta index of this worker's range.")
    p.add_argument("--theta-stop", type=int, default=None,
                   help="generate/infer: end (exclusive) of this worker's range "
                        "(default: the whole cohort).")
    p.add_argument("--posterior-samples", type=int, default=None,
                   help="infer: draws per window (default: the Evaluation-stage config).")
    p.add_argument("--pool-mode", choices=("bounded", "unrestricted"), default="unrestricted",
                   help="infer: posterior sampler. Default 'unrestricted' (direct flow "
                        "sampling): it matches the experimental-baseline methodology, never "
                        "stalls, and is REQUIRED for the outside-the-box flow-mass diagnostic.")
    p.add_argument("--recovery-artifact", default=None,
                   help="analyze: path to the held-out MAP_Recovery .npz used to calibrate the "
                        "PER-STRATUM margins of the stratified robustness table (default: the "
                        "standard sibling location in this data bank). Without it the stratified "
                        "table is skipped and the prior-averaged verdicts stand alone.")
    p.add_argument("--margin-fd", type=float, default=None,
                   help=f"analyze: equivalence margin for the f_D primary, percentage points "
                        f"(default {_DEFAULT_MARGINS['f_D']} = the in-model f_D error on the "
                        f"held-out recovery).")
    p.add_argument("--margin-da", type=float, default=None,
                   help=f"analyze: equivalence margin for D_A, dex "
                        f"(default {_DEFAULT_MARGINS['diffusivity_alp']}).")
    p.add_argument("--margin-koff", type=float, default=None,
                   help=f"analyze: equivalence margin for kappa_OFF, dex "
                        f"(default {_DEFAULT_MARGINS['rate_dissociation']}).")
    p.add_argument("--keep-trajectories", action="store_true",
                   help="generate: keep the ReaDDy .h5 trajectory files (default: extract the "
                        "population trace, then delete them to save space).")
    p.add_argument("--overwrite", action="store_true",
                   help="prepare: replace an existing cohort file (stale per-theta artifacts "
                        "still refuse -- remove them explicitly). generate/infer: redo work "
                        "whose output exists (default: verify its cohort stamp and skip).")
    p.add_argument("--seed", type=lambda v: None if str(v).strip().lower() in ("none", "")
                   else int(v), default=None,
                   help="prepare: the master seed (default: drawn from OS entropy and "
                        "PERSISTED, so generation is reproducible either way). infer: torch "
                        "draw-stream seed (optional). analyze: bootstrap RNG seed.")
    p.add_argument("--verbose", action="store_true", help="verbose simulation/render output.")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve paths, report cohort/file status, do nothing.")
    return p

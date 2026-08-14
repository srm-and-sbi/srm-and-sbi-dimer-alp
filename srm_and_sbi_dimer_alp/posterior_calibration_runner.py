"""Shared Posterior-Calibration diagnostic engine for both DIMER workflows.

``run_posterior_calibration(cfg, args)`` holds the whole diagnostic: banner, dry-run
probe, estimator load, the per-video draw loop over the held-out EVAL namespace,
multi-GPU round-robin task sharding, the ``--merge`` combine step, and the calibration
report + figures. The two entry-point shims shrink to: build the ``WorkflowConfig``,
parse args, call ``run_posterior_calibration`` -- exactly the Evaluation-stage pattern.

The numeric engine is the workflow-agnostic ``posterior_calibration`` kernel (SBC,
expected coverage, TARP, L-C2ST, all stratified by target-theta dimension). This runner
only *feeds* it: it streams the EVAL videos once, drawing per video the posterior
samples, the sample + truth log-densities, and the learned embedding (the summary
L-C2ST conditions on), then hands the kernel those small arrays. It never holds a raw
video beyond the draw.

The only per-workflow differences -- which parameterization supplies the learnable
table + parameter keys, the device ``build_prior``, and the alias-qualified ``paths`` --
are resolved from the config in ``_posterior_calibration_spec(cfg)``, mirroring
``evaluation_runner._evaluation_spec``. The estimator + EVAL paths resolve through the
same shared ``paths`` methods for both workflows.

Not a canonical pipeline stage: this is an Analysis diagnostic (never wired into the
``Submit.sh`` dispatcher). It shares the two-shim + shared-engine structure so that one
implementation serves both workflows, and its report lands in the ``Posit/`` tier
alongside the posterior it characterizes.
"""
from __future__ import annotations

import argparse
import random
import shutil
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch._dynamo

from srm_and_sbi_dimer_alp import artifacts
from srm_and_sbi_dimer_alp import posterior_calibration as pcal
from srm_and_sbi_dimer_alp.diagnostics import DiagnosticReporter
from srm_and_sbi_dimer_alp.evaluation import collect_score_prex, collect_theta_prex
from srm_and_sbi_dimer_alp.inference_support import normalize_video, resolve_topology
from srm_and_sbi_dimer_alp.io import load_data
from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming
from srm_and_sbi_dimer_alp.visualization_calibration import (
    figure_coverage, figure_sbc_ranks, figure_stratified, figure_tarp)
from srm_and_sbi_dimer_alp.workflow import WorkflowConfig

_ALL_TESTS = ("sbc", "coverage", "tarp", "lc2st")


# =============================================================================
# Per-workflow specialization + output paths
# =============================================================================

@dataclass(frozen=True)
class _CalibrationSpec:
    """Per-workflow calibration specializations resolved from a ``WorkflowConfig``."""
    draw_spec: list          # learnable table (KEY/LABEL) for the per-parameter SBC table
    parameter_keys: list     # target-theta keys: load_estimator schema guard + kernel labels
    build_prior: Callable    # device prior rebuild (bounded sampling + prior-sample baseline)


def _posterior_calibration_spec(cfg: WorkflowConfig) -> _CalibrationSpec:
    """Resolve the calibration diagnostic's per-workflow specializations from the config."""
    m = cfg.param_module
    if cfg.tag == "detector":
        return _CalibrationSpec(
            draw_spec=m.DETECTOR_PARAMETERIZATION,
            parameter_keys=[e["KEY"] for e in m.DETECTOR_PARAMETERIZATION],
            build_prior=m.build_prior,
        )
    return _CalibrationSpec(
        draw_spec=m.PARAMETERIZATION,
        parameter_keys=m.PARAMETER_KEYS,
        build_prior=m.build_prior,
    )


def _calibration_dir(paths, data_bank_root: Path, timing_label: str) -> Path:
    """Directory holding the calibration report, figures, and arrays (Posit tier).

    Mirrors ``paths.map_recovery_dir`` (Evaluation): sits under ``Posit/`` alongside the
    posterior it characterizes, named ``<project_alias>_<timing_label>_Posterior_Calibration``
    so the ``_DETECTOR`` alias qualifier keeps the two workflows' reports separate.
    """
    return (data_bank_root / paths.posit_subdir /
            f"{paths.project_alias}_{timing_label}_Posterior_Calibration")


def _calibration_array_path(cal_dir: Path, timing_label: str, project_alias: str) -> Path:
    """Full path for the saved calibration input arrays (.npz)."""
    return cal_dir / f"{project_alias}_{timing_label}_Posterior_Calibration.npz"


def _shard_path(cal_dir: Path, rank: int, world_size: int) -> Path:
    """Path of one worker's partial calibration arrays in a multi-GPU sharded run."""
    return cal_dir / f"_shard_{rank:02d}_of_{world_size:02d}.npz"


# =============================================================================
# Per-video draw: samples + log-densities + embedding (the kernel's raw inputs)
# =============================================================================

def _draw_video_calibration(posterior, flow, video_chunk, true_log10, device, vista_device,
                            n_samples, theta_prex_batch_size, score_prex_batch_size, pool_mode):
    """Draw one video's calibration inputs: samples, their + the truth's log-density, embedding.

    Reuses the Evaluation machinery: ``collect_theta_prex`` draws the posterior sample
    cloud, and the ``map_estimate`` embed-once trick (swap ``embedding_net`` for identity
    after caching the latent) makes scoring the ``L + 1`` thetas -- the truth stacked
    ahead of the samples -- cost one Complex3DCNN pass instead of ``L + 1``.

    Returns:
        ``(samples (L, D), sample_log_probs (L,), truth_log_prob (float), embedding (E,))``
        with theta in log10 space, matching the flow's + prior's space.
    """
    omega = torch.tensor(normalize_video(video_chunk), dtype=torch.float32, device=device)
    cond = omega.unsqueeze(0)
    posterior.set_default_x(cond)

    with torch.no_grad():
        emb_x = flow.embedding_net(cond).detach()          # (1, E): the L-C2ST summary + score cache

    # Sample the pool on the raw conditioning (bounded rejection embeds the video itself).
    samples = collect_theta_prex(posterior, flow, vista_device, cond,
                                 n_samples, theta_prex_batch_size, pool_mode)   # (L, D) log10, on CPU
    truth = torch.as_tensor(np.asarray(true_log10), dtype=torch.float32,
                            device=vista_device).unsqueeze(0)                   # (1, D)
    all_theta = torch.cat([truth, samples], dim=0)                             # (L + 1, D)

    # Score truth + samples against the cached latent (embedding_net -> identity).
    embed_holder = flow.net if (hasattr(flow, "net")
                                and hasattr(flow.net, "_embedding_net")) else flow
    original_embedding_net = embed_holder._embedding_net
    original_condition_shape = flow.condition_shape
    embed_holder._embedding_net = torch.nn.Identity()
    flow._condition_shape = tuple(emb_x.shape[1:])
    try:
        all_lp = collect_score_prex(flow, device, vista_device, emb_x,
                                    all_theta, score_prex_batch_size)           # (L + 1,)
    finally:
        embed_holder._embedding_net = original_embedding_net
        flow._condition_shape = original_condition_shape

    all_lp = all_lp.detach().cpu().numpy()
    return (samples.detach().cpu().numpy(),        # (L, D)
            all_lp[1:],                            # (L,)  sample log-densities
            float(all_lp[0]),                      # truth log-density
            emb_x.squeeze(0).detach().cpu().numpy())   # (E,)


# =============================================================================
# Report (calibrate + tables + figures), shared by single-process + --merge
# =============================================================================

def _parse_tests(spec_tests: str) -> tuple:
    tests = tuple(t.strip() for t in spec_tests.split(",") if t.strip())
    unknown = set(tests) - set(_ALL_TESTS)
    if unknown:
        raise SystemExit(f"--tests: unknown {sorted(unknown)}; choose from {_ALL_TESTS}")
    return tests


def _parse_stratify(spec_stratify: str, parameter_keys: list):
    """Resolve ``--stratify`` (all | none | comma-list of parameter KEYS) to dim indices."""
    token = spec_stratify.strip().lower()
    if token == "all":
        return None                       # kernel: stratify by every target-theta dimension
    if token == "none":
        return ()                         # overall only
    keys = [k.strip() for k in spec_stratify.split(",") if k.strip()]
    dims = []
    for k in keys:
        if k not in parameter_keys:
            raise SystemExit(f"--stratify: '{k}' is not a target parameter {parameter_keys}")
        dims.append(parameter_keys.index(k))
    return tuple(dims)


def _write_calibration_report(reporter, args, cfg, spec, cal_dir: Path, cal_array_path: Path,
                              truths, samples, truth_log_probs, sample_log_probs, embeddings,
                              run_start) -> None:
    """Build the kernel inputs, run ``calibrate``, and write the report + figures + arrays."""
    truths = np.asarray(truths)
    samples = np.asarray(samples)
    n_videos = truths.shape[0]

    # Prior-sample baseline for check_sbc's data-averaged-posterior check (log10 space).
    prior_samples = spec.build_prior(device="cpu").sample((n_videos,)).cpu().numpy()

    inputs = pcal.CalibrationInputs(
        truths=truths, samples=samples, theta_keys=spec.parameter_keys,
        prior_samples=prior_samples,
        truth_log_probs=np.asarray(truth_log_probs),
        sample_log_probs=np.asarray(sample_log_probs),
        embeddings=np.asarray(embeddings),
    )

    tests = _parse_tests(args.tests)
    stratify_dims = _parse_stratify(args.stratify, list(spec.parameter_keys))
    lc2st_kwargs = dict(n_eval=args.lc2st_n_eval, num_trials_null=args.lc2st_null_trials,
                        seed=(args.seed or 0))

    result = pcal.calibrate(
        inputs, tests=tests, stratify_dims=stratify_dims, n_strata=args.n_strata,
        num_bins=args.num_bins, lc2st_kwargs=lc2st_kwargs, sbc_c2st=args.sbc_c2st,
        min_stratum=args.min_stratum,
    )

    # ---- Save the calibration input arrays (reproducible re-analysis) -----
    np.savez_compressed(
        str(cal_array_path),
        truths=truths, samples=samples,
        truth_log_probs=np.asarray(truth_log_probs),
        sample_log_probs=np.asarray(sample_log_probs),
        embeddings=np.asarray(embeddings),
        parameter_keys=np.asarray(list(spec.parameter_keys)),
    )
    print(f"\nCalibration arrays saved to {cal_array_path}")

    # ---- Report ----------------------------------------------------------
    reporter.check("eval_set_nonempty", n_videos > 0, f"{n_videos} EVAL videos drawn",
                   note="the held-out EVAL namespace yielded at least one calibration sample.")
    reporter.stat("eval_tasks", args.eval_tasks)
    reporter.stat("calibration_videos", n_videos,
                  note="held-out videos whose posterior was drawn for calibration.")
    reporter.stat("posterior_samples", inputs.num_posterior_samples,
                  note="draws per video (L). SBC/TARP use the full cloud; L-C2ST one draw/video.")
    reporter.stat("tests", ", ".join(tests))

    if result.overall_sbc is not None:
        sbc = result.overall_sbc
        d = len(sbc.ks_pvals)
        headers = ["parameter", "label", "KS p-value", "C2ST(rank)", "verdict"]
        rows = []
        for i, para in enumerate(spec.draw_spec):
            ksp = float(sbc.ks_pvals[i])
            c2st = "-" if sbc.c2st_ranks is None else f"{float(sbc.c2st_ranks[i]):.3f}"
            verdict = "check" if ksp < 0.05 / d else "ok"
            rows.append([para["KEY"], para.get("LABEL") or "-", f"{ksp:.4f}", c2st, verdict])
        reporter.table(
            "SBC rank uniformity (per parameter)", headers, rows,
            note="Talts et al. 2018: the rank of the true theta among posterior samples is "
                 "uniform per marginal iff calibrated. 'verdict' flags KS p below the "
                 f"Bonferroni threshold 0.05/{d}; the rank-histogram figure is the real read.")

    if result.overall_coverage is not None:
        cov = result.overall_coverage
        # Report empirical coverage at the standard nominal levels present on the grid.
        pts = []
        for target in (0.5, 0.9):
            k = int(np.argmin(np.abs(cov.levels - target)))
            pts.append(f"{cov.levels[k]:.2f}->{cov.empirical[k]:.2f}")
        reporter.stat("coverage_KS_pval", f"{cov.ks_pval:.4f}",
                      note="Deistler/Hermans expected coverage: KS p that the log-density rank "
                           "is uniform. Small => miscalibrated.")
        reporter.stat("coverage_points", "  ".join(pts),
                      note="empirical coverage at nominal 0.5 / 0.9 (nominal->empirical). "
                           "Below nominal = overconfident; above = conservative.")

    if result.overall_tarp is not None:
        t = result.overall_tarp
        reporter.stat("tarp_ATC", f"{t.atc:+.4f}",
                      note="Lemos et al. 2023 area-to-curve; 0 ideal, >0 over-dispersed, "
                           "<0 under-dispersed (overconfident).")
        reporter.stat("tarp_KS_pval", f"{t.ks_pval:.4f}", note="KS p that ECP == credibility.")

    if result.overall_lc2st is not None:
        lc = result.overall_lc2st
        reporter.stat("lc2st_reject_fraction", f"{lc.reject_fraction:.3f}",
                      note=f"Linhart et al. 2023 local C2ST over {lc.n_eval} observations: "
                           f"fraction rejecting calibration at alpha={lc.alpha}. ~alpha when "
                           "calibrated.")
        reporter.stat("lc2st_median_pvalue", f"{lc.median_p_value:.3f}")

    # Stratified digest -- the panel that localizes where calibration degrades.
    strat_rows = [r for r in pcal.summarize(result) if r["scope"] != "overall"]
    if strat_rows:
        headers = ["stratum", "test", "statistic", "value", "verdict"]
        rows = [[r["scope"], r["test"], r["statistic"],
                 (f"{r['value']:+.4f}" if isinstance(r["value"], float) else str(r["value"])),
                 r["flag"]] for r in strat_rows]
        reporter.table(
            "Stratified calibration (by inferred-value bin)", headers, rows,
            note="each diagnostic recomputed within equal-count bins of one parameter's INFERRED "
                 "value (the per-video posterior median -- a function of the observation, not the "
                 "latent truth, which would confound Bayesian shrinkage with miscalibration). A bin "
                 "that flags while the overall passes localizes a subregion the posterior is "
                 "miscalibrated for.")

    # ---- Figures ---------------------------------------------------------
    if reporter.dump:
        if result.overall_sbc is not None:
            reporter.save_figure("sbc_rank_histograms", figure_sbc_ranks(result.overall_sbc),
                                 caption="Per-marginal SBC rank histograms; flat = calibrated, "
                                         "U-shape = overconfident, ^-shape = over-dispersed.")
        if result.overall_coverage is not None:
            reporter.save_figure("expected_coverage", figure_coverage(result.overall_coverage),
                                 caption="Expected coverage: empirical vs nominal credibility "
                                         "(diagonal = calibrated).")
        if result.overall_tarp is not None:
            reporter.save_figure("tarp_ecp", figure_tarp(result.overall_tarp),
                                 caption="TARP expected-coverage-probability curve (diagonal = ideal).")
        for test in tests:
            fig = figure_stratified(result, test=test)
            if fig is not None:
                reporter.save_figure(f"stratified_{test}", fig,
                                     caption=f"Stratified {test} across equal-count bins of each "
                                             "parameter's inferred value (per-video posterior "
                                             "median); localizes where the posterior is miscalibrated.")

    reporter.summary()
    reporter.write_report()
    print(f"\nTotal elapsed: {time.time() - run_start:.1f}s")


def _save_shard(topo, cal_dir: Path, truths, samples, truth_log_probs, sample_log_probs,
                embeddings, run_start) -> None:
    """Write this worker's partial calibration arrays (multi-GPU sharded run)."""
    cal_dir.mkdir(parents=True, exist_ok=True)
    if len(truths) == 0:
        print(f"\n[rank {topo.rank}/{topo.world_size}] no tasks assigned -- no shard written.",
              flush=True)
        return
    path = _shard_path(cal_dir, topo.rank, topo.world_size)
    np.savez_compressed(
        str(path),
        truths=np.asarray(truths), samples=np.asarray(samples),
        truth_log_probs=np.asarray(truth_log_probs),
        sample_log_probs=np.asarray(sample_log_probs),
        embeddings=np.asarray(embeddings),
    )
    print(f"\n[rank {topo.rank}/{topo.world_size}] shard saved: {path} "
          f"({len(truths)} videos) in {time.time() - run_start:.1f}s. "
          f"Run the --merge step once all shards finish.", flush=True)


def _merge_shards(reporter, args, cfg, spec, cal_dir: Path, cal_array_path: Path,
                  run_start) -> None:
    """Concatenate all per-shard calibration arrays, then run the report on the full set."""
    shard_paths = sorted(cal_dir.glob("_shard_*_of_*.npz"))
    if not shard_paths:
        raise SystemExit(f"--merge: no shard files (_shard_*_of_*.npz) in {cal_dir}")
    print(f"Merging {len(shard_paths)} shard file(s) from {cal_dir}", flush=True)
    truths, samples, truth_lp, sample_lp, emb = [], [], [], [], []
    for shard_path in shard_paths:
        with np.load(str(shard_path)) as data:
            if data["truths"].shape[0] == 0:
                continue
            truths.append(data["truths"]); samples.append(data["samples"])
            truth_lp.append(data["truth_log_probs"]); sample_lp.append(data["sample_log_probs"])
            emb.append(data["embeddings"])
    if not truths:
        raise SystemExit(f"--merge: every shard in {cal_dir} was empty")
    truths = np.concatenate(truths, 0); samples = np.concatenate(samples, 0)
    truth_lp = np.concatenate(truth_lp, 0); sample_lp = np.concatenate(sample_lp, 0)
    emb = np.concatenate(emb, 0)
    print(f"Merged {truths.shape[0]} videos from {len(shard_paths)} shard(s).", flush=True)
    _write_calibration_report(reporter, args, cfg, spec, cal_dir, cal_array_path,
                              truths, samples, truth_lp, sample_lp, emb, run_start)
    for shard_path in shard_paths:
        shard_path.unlink()
    print(f"Removed {len(shard_paths)} shard file(s).", flush=True)


# =============================================================================
# Orchestration
# =============================================================================

def run_posterior_calibration(cfg: WorkflowConfig, args: argparse.Namespace) -> None:
    """Run the full posterior-calibration diagnostic for the given workflow + CLI args."""
    spec = _posterior_calibration_spec(cfg)

    timing = RunTiming(total_time_seconds=args.total_time_seconds,
                       frames=PARAMETERS.simulation.timing)
    data_bank_root = PARAMETERS.machine.data_bank_root
    compress = True
    paths = cfg.paths
    eval_cfg = PARAMETERS.inference.evaluation

    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.suppress_errors = True

    timing_label = timing.label
    estimator_path = paths.estimator_path(data_bank_root, timing_label)
    cal_dir = _calibration_dir(paths, data_bank_root, timing_label)
    cal_array_path = _calibration_array_path(cal_dir, timing_label, paths.project_alias)

    n_samples = args.posterior_samples or eval_cfg.posterior_samples
    pool_mode = args.pool_mode or eval_cfg.pool_mode

    # ---- Banner ----------------------------------------------------------
    machine = PARAMETERS.machine
    div = "=" * 72
    print(div)
    print(f" {paths.project_alias} — Posterior Calibration (SBC / coverage / TARP / L-C2ST)")
    print(f" Started at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(div)
    print("\nMachine profile:")
    print(f"  name              : {machine.name}")
    print(f"  compute_backend   : {machine.compute_backend}")
    print("\nRun configuration (CLI args):")
    print(f"  --total-time-seconds : {args.total_time_seconds}")
    print(f"  --eval-tasks         : {args.eval_tasks}        (EVAL-namespace tasks; held out)")
    print(f"  --max-sims           : {args.max_sims}        (per task; 0 = all)")
    print(f"  --posterior-samples  : {n_samples}   (L draws per video)")
    print(f"  --pool-mode          : {pool_mode}")
    print(f"  --tests              : {args.tests}")
    print(f"  --stratify           : {args.stratify}   (--n-strata {args.n_strata})")
    print("\nOutput destinations:")
    print(f"  reads estimator : {estimator_path}")
    print(f"  reads EVAL      : <data_bank>/{paths.video_subdir}/"
          f"{paths.project_alias}_{timing_label}_Video_Set_TASK_{{0..{args.eval_tasks - 1}}}_EVAL.zarr")
    print(f"  writes report   : {cal_dir}")
    print(f"\n{div}\n")

    # ---- Dry run: validate inputs before any reporter / GPU / posterior ---
    if args.dry_run:
        eval_video_path = paths.video_set_path(0, data_bank_root, timing_label, compress, "EVAL")
        eval_theta_path = paths.theta_set_path(0, data_bank_root, timing_label, compress, "EVAL")
        inputs = [("estimator artifact", estimator_path),
                  ("EVAL video set (task 0)", eval_video_path),
                  ("EVAL theta set (task 0)", eval_theta_path)]
        missing = 0
        for role, path in inputs:
            ok = Path(path).exists()
            missing += 0 if ok else 1
            print(f"  reads {role}: {path}  [{'OK' if ok else 'MISSING'}]")
        _parse_tests(args.tests)
        _parse_stratify(args.stratify, list(spec.parameter_keys))
        print(f"\n[DRY RUN] configuration validated; "
              f"{'all inputs present' if not missing else f'{missing} input(s) MISSING'}.")
        print("[DRY RUN] no calibration performed.")
        return

    run_start = time.time()
    reporter = DiagnosticReporter(
        stage="Posterior_Calibration", enabled=True, dump=True, dump_dir=cal_dir,
        run_label=f"{paths.project_alias}_{timing_label}",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )

    if args.merge:
        _merge_shards(reporter, args, cfg, spec, cal_dir, cal_array_path, run_start)
        return

    reporter.check_file("estimator artifact", estimator_path)

    topo = resolve_topology()
    device = topo.device
    vista_device = torch.device("cpu")
    posterior = artifacts.load_estimator(estimator_path, device=str(device),
                                          expected_parameter_keys=spec.parameter_keys)
    posterior.posterior_estimator.to(device)
    if device.type == "cuda":
        posterior.prior = spec.build_prior(device=str(device))
    flow = posterior.posterior_estimator

    # ---- This worker's task shard ----------------------------------------
    my_tasks = [t for t in range(args.eval_tasks) if t % topo.world_size == topo.rank]
    cal_dir.mkdir(parents=True, exist_ok=True)
    if topo.is_main:
        shutil.rmtree(cal_dir / "figures", ignore_errors=True)

    truths, samples, truth_lp, sample_lp, embeddings = [], [], [], [], []
    loop_start = time.time()
    done = 0
    shard_note = (f" [shard rank {topo.rank}/{topo.world_size}: {len(my_tasks)} of "
                  f"{args.eval_tasks} tasks]" if topo.is_distributed else "")
    print(f"START calibration draw: {len(my_tasks)} EVAL task(s){shard_note}, "
          f"L={n_samples} samples/video, pool={pool_mode}.", flush=True)

    for task in my_tasks:
        video_set = load_data(
            paths.video_set_path(task, data_bank_root, timing_label, compress, "EVAL"))
        theta_set = load_data(
            paths.theta_set_path(task, data_bank_root, timing_label, compress, "EVAL"))
        n_task = theta_set.shape[0]
        n_task = min(n_task, args.max_sims) if args.max_sims > 0 else n_task
        task_start = time.time()
        for sim in range(n_task):
            video_chunk = np.asarray(video_set[sim])
            true_log10 = np.log10(np.asarray(theta_set[sim], dtype=float))
            s, s_lp, t_lp, emb = _draw_video_calibration(
                posterior, flow, video_chunk, true_log10, device, vista_device,
                n_samples, eval_cfg.theta_prex_batch_size, eval_cfg.score_prex_batch_size,
                pool_mode)
            truths.append(true_log10); samples.append(s)
            sample_lp.append(s_lp); truth_lp.append(t_lp); embeddings.append(emb)
            done += 1
            if done % 50 == 0 or done == 1:
                elapsed = time.time() - loop_start
                print(f"  drawn {done} videos | {elapsed:.1f}s | "
                      f"{elapsed / done:.2f}s/video", flush=True)
        print(f"task {task} complete ({n_task} videos in {time.time() - task_start:.1f}s).",
              flush=True)

    # ---- Write outputs (single process) or shard (distributed) -----------
    if topo.is_distributed:
        _save_shard(topo, cal_dir, truths, samples, truth_lp, sample_lp, embeddings, run_start)
    else:
        _write_calibration_report(reporter, args, cfg, spec, cal_dir, cal_array_path,
                                  truths, samples, truth_lp, sample_lp, embeddings, run_start)


# =============================================================================
# CLI parser (identical for both workflows; the shim builds the WorkflowConfig)
# =============================================================================

def build_posterior_calibration_parser() -> argparse.ArgumentParser:
    """Construct the Posterior-Calibration CLI parser (shared by both workflow shims)."""
    eval_cfg = PARAMETERS.inference.evaluation
    parser = argparse.ArgumentParser(
        description="Score a trained posterior's calibration on the EVAL namespace "
                    "(SBC / expected coverage / TARP / L-C2ST), overall and stratified.")
    parser.add_argument(
        "--total-time-seconds", type=float, required=True,
        help="Video duration in seconds; must match the trained posterior's runs.")
    parser.add_argument(
        "--eval-tasks", type=int, required=True,
        help="Number of EVAL-namespace tasks to draw (held-out data).")
    parser.add_argument(
        "--max-sims", type=int, default=0,
        help="Cap on videos drawn per EVAL task (0 = all; useful for quick checks).")
    parser.add_argument(
        "--posterior-samples", type=int, default=None,
        help=f"Posterior draws per video, L (default: {eval_cfg.posterior_samples}). "
             "SBC/TARP use the full cloud; L-C2ST uses one draw per video.")
    parser.add_argument(
        "--pool-mode", choices=("bounded", "unrestricted"), default=None,
        help="Posterior sampler: 'bounded' (rejection within the prior; correct for a "
             "trained posterior) or 'unrestricted' (flow direct; for smoke tests / "
             f"undertrained posteriors that would stall). Default: {eval_cfg.pool_mode}.")
    parser.add_argument(
        "--tests", type=str, default=",".join(_ALL_TESTS),
        help=f"Comma-separated diagnostics to run (default: all). Choices: {_ALL_TESTS}.")
    parser.add_argument(
        "--stratify", type=str, default="all",
        help="Target-theta dimension(s) to stratify by: 'all' (every dimension; default), "
             "'none' (overall only), or a comma-list of parameter KEYS. Bins are equal-count.")
    parser.add_argument(
        "--n-strata", type=int, default=5,
        help="Equal-count bins per stratifying dimension (default: 5).")
    parser.add_argument(
        "--min-stratum", type=int, default=200,
        help="Skip a stratum with fewer than this many videos (default: 200).")
    parser.add_argument(
        "--num-bins", type=int, default=30,
        help="TARP credibility bins (default: 30).")
    parser.add_argument(
        "--sbc-c2st", action="store_true",
        help="Also run SBC's slower per-marginal classifier two-sample tests (off by default; "
             "the KS test is the fast primary uniformity check).")
    parser.add_argument(
        "--lc2st-n-eval", type=int, default=200,
        help="Observations at which L-C2ST evaluates its local statistic (default: 200).")
    parser.add_argument(
        "--lc2st-null-trials", type=int, default=100,
        help="Null-hypothesis classifiers L-C2ST trains for its p-values (default: 100).")
    parser.add_argument(
        "--seed", type=lambda v: None if str(v).strip().lower() in ("none", "") else int(v),
        default=None,
        help="Master RNG seed. Default None -> non-deterministic (consistent with generation).")
    parser.add_argument(
        "--merge", action="store_true",
        help="Combine-only mode: concatenate the per-shard calibration .npz files from a "
             "multi-GPU sharded run and write the final report + figures, then exit. Does no "
             "drawing and needs no GPU; the launcher runs it once the sharded workers finish.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Validate configuration and inputs, print what would be read/written, then exit "
             "without drawing (no GPU, no compute).")
    return parser

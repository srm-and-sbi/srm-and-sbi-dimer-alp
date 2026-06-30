"""Entry-point script: MAP-estimate model parameters from real microscopy videos.

The companion to ``Evaluation.py``: where Evaluation recovers parameters from
held-out *simulated* EVAL videos (known ground truth), this stage applies the
same seed-then-optimize ``map_estimate`` to *real* experimental videos, for which
there is no ground truth. Each cell's long recording is split into model-length
chunks; the MAP theta is estimated per chunk, and the report shows the
distribution of inferred parameters per experimental condition (kind), so
conditions can be compared (e.g. ALP vs BET).

Inputs (real microscopy, copied into the data bank by the user):
    <data_bank>/<experiment_subdir>/Experiment_{KIND}_Cell_{n}_{span}S_RAW.tif
        -- 16-bit raw video, one per (kind, cell); converted to 8-bit and chunked
           into <n_frames>-frame windows (n_frames = model duration x frame rate).

Outputs (under <data_bank>/<posit_subdir>/<project_alias>_{timing_label}_MAP_Experiment/):
    report.md                         -- per-parameter inferred-theta summary by kind
    figures/distribution_<KEY>.png    -- inferred-theta distribution per condition
    <...>_MAP_Experiment.npz          -- inferred_log10, scores, kind/cell/chunk indices
    progress.log                      -- live, tail-able trail

Usage:
    MACHINE_PROFILE=<profile> python SRM_AND_SBI_DIMER_ALP_Experiment.py \\
        --total-time-seconds 2.0 --kinds ALP,BET --max-cells 2 --pool-mode bounded
"""

import argparse
import random
import re
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch._dynamo

from srm_and_sbi_dimer_alp.diagnostics import DiagnosticReporter
from srm_and_sbi_dimer_alp.evaluation import (
    map_estimate,
    experiment_table,
    posterior_summary,
    _theta_repr,
)
from srm_and_sbi_dimer_alp.inference_support import get_device, load_posterior
from srm_and_sbi_dimer_alp.io import convert_video_dtype
from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, PARAMETERIZATION, RunTiming, build_prior
from srm_and_sbi_dimer_alp.utils import console_log_context
from srm_and_sbi_dimer_alp.visualization_inference import figure_experiment_combined


def _discover_cells(experiment_dir: Path, kind: str, span: int) -> list:
    """Return sorted cell indices available on disk for a given kind."""
    cells = []
    for path in experiment_dir.glob(f"Experiment_{kind}_Cell_*_{span}S_RAW.tif"):
        match = re.search(rf"Cell_(\d+)_{span}S_RAW", path.name)
        if match:
            cells.append(int(match.group(1)))
    return sorted(cells)


def _aggregate_by_kind(inferred_log10, kind_index, cell_arr, kinds, mode, n_params):
    """Group inferred theta by condition for the report, per ``mode``.

    ``"pooled"``      -- every (cell, chunk) estimate is one sample, pooled per
                         kind (mixes temporal + biological variation).
    ``"cell-median"`` -- collapse each cell to its median across its chunks first,
                         so each kind's distribution is one sample per cell
                         (biological spread; within-cell temporal noise averaged out).

    Returns ``{kind: (N, D) array}`` of inferred log10 theta. The raw per-(cell,
    chunk) arrays are saved to the .npz regardless, so either view is reproducible.
    """
    out = {}
    for ki, kind in enumerate(kinds):
        kmask = kind_index == ki
        if not np.any(kmask):
            out[kind] = np.empty((0, n_params))
            continue
        kinf = inferred_log10[kmask]
        if mode == "cell-median":
            kcells = cell_arr[kmask]
            out[kind] = np.asarray(
                [np.median(kinf[kcells == c], axis=0) for c in np.unique(kcells)])
        else:  # pooled
            out[kind] = kinf
    return out


def main(args: argparse.Namespace) -> None:
    """Run MAP estimation over the experimental videos per the CLI args."""
    timing = RunTiming(
        total_time_seconds=args.total_time_seconds, frames=PARAMETERS.simulation.timing,
    )
    data_bank_root = PARAMETERS.machine.data_bank_root
    paths = PARAMETERS.paths
    eval_cfg = PARAMETERS.inference.evaluation
    span = args.experiment_span_seconds

    # ---- Global RNG / precision settings ---------------------------------
    if args.seed is not None:   # None -> non-deterministic (consistent with generation)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        random.seed(args.seed)
    torch.set_float32_matmul_precision("high")
    torch._dynamo.config.suppress_errors = True

    timing_label = timing.label
    posterior_path = paths.posterior_path(data_bank_root, timing_label)
    experiment_dir = data_bank_root / paths.experiment_subdir
    out_dir = paths.experiment_recovery_dir(data_bank_root, timing_label)
    array_path = out_dir / (out_dir.name + ".npz")
    progress_path = out_dir / "progress.log"

    # Effective hyperparameters (None -> config default).
    lr = (args.learning_rate if args.learning_rate is not None
          else eval_cfg.learning_rate_minimum * eval_cfg.learning_rate_maximum_factor)
    tolerance = eval_cfg.learning_rate_minimum * eval_cfg.tolerance_factor
    theta_prex_size = args.theta_prex_size or eval_cfg.theta_prex_size
    elite_prex_size = args.elite_prex_size or eval_cfg.elite_prex_size
    numb_steps = args.numb_steps or eval_cfg.numb_steps
    show_progress_steps = args.show_progress_steps or eval_cfg.show_progress_steps
    pool_mode = args.pool_mode or eval_cfg.pool_mode

    show = args.verbose or args.debug or args.debug_dump
    verbose_deep = args.debug or args.debug_dump
    debug_log = args.debug or args.debug_dump

    # Summary views (A = MAP-point distribution; B = posterior credible summary).
    do_map = args.summary in ("map", "both")
    do_posterior = args.summary in ("posterior", "both")
    posterior_samples = args.posterior_samples or eval_cfg.posterior_samples

    # Video geometry: model-length windows stepped across each long recording.
    # The window length equals the synthetic (training) video length, keeping the
    # chunks fully compatible with the model. The step between consecutive windows
    # is --chunk-step-seconds: an integer that divides the window and is <= it
    # (1 s -> maximal overlap; window -> non-overlapping). Smaller steps yield more
    # (overlapping) chunks per recording.
    n_frames = timing.frame_count
    window_seconds = timing.total_time_seconds
    step_seconds = args.chunk_step_seconds if args.chunk_step_seconds else int(window_seconds)
    if (window_seconds != int(window_seconds) or step_seconds < 1
            or step_seconds > int(window_seconds)
            or int(window_seconds) % int(step_seconds) != 0):
        raise SystemExit(
            f"--chunk-step-seconds={step_seconds} is invalid: it must be a positive "
            f"integer that divides the model window ({window_seconds:g} s) and does "
            f"not exceed it (valid steps: integer divisors of {int(window_seconds)}).")
    step_frames = int(round(step_seconds / timing.frame_time_seconds))
    exp_frames = int(round(span / timing.frame_time_seconds))
    n_chunks = (exp_frames - n_frames) // step_frames + 1

    # ---- Resolve the (kind, cells) work list -----------------------------
    kinds = [k.strip() for k in args.kinds.split(",") if k.strip()]
    explicit_cells = ([int(c) for c in args.cells.split(",")] if args.cells else None)
    cells_by_kind = {}
    for kind in kinds:
        cells = explicit_cells if explicit_cells is not None else _discover_cells(
            experiment_dir, kind, span)
        if args.max_cells > 0:
            cells = cells[:args.max_cells]
        cells_by_kind[kind] = cells
    total_estimates = sum(len(c) for c in cells_by_kind.values()) * n_chunks

    # ---- Pre-run banner --------------------------------------------------
    machine = PARAMETERS.machine
    div = "=" * 72
    print(div)
    print(f" {paths.project_alias} — Experiment (MAP estimation on real data)")
    print(f" Started at  : {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
    print(div)
    print("\nMachine profile:")
    print(f"  name              : {machine.name}")
    print(f"  compute_backend   : {machine.compute_backend}")
    print("\nRun configuration (CLI args):")
    print(f"  --total-time-seconds      : {args.total_time_seconds}  (model window -> {n_frames} frames)")
    print(f"  --experiment-span-seconds : {span}  ({n_chunks} chunks/video, step {step_seconds}s)")
    print(f"  --kinds                   : {kinds}")
    for kind in kinds:
        print(f"      {kind}: cells {cells_by_kind[kind]}")
    print(f"  --max-cells               : {args.max_cells}  (0 = all discovered)")
    print(f"  --aggregation             : {args.aggregation}")
    print(f"  --summary                 : {args.summary}   (View A map={do_map}, View B posterior={do_posterior})")
    if do_posterior:
        print(f"  --posterior-samples       : {posterior_samples}")
    print(f"  --seed                    : {args.seed}")
    print(f"  total MAP estimates       : {total_estimates}")
    print("\nMAP estimate hyperparameters (effective):")
    print(f"  pool_mode            : {pool_mode}")
    print(f"  theta_prex_size      : {theta_prex_size}    elite_prex_size: {elite_prex_size}")
    print(f"  numb_steps           : {numb_steps}    learning_rate: {lr:.3e}")
    print(f"  verbosity            : "
          f"{'debug-dump' if args.debug_dump else ('debug' if args.debug else ('verbose' if args.verbose else 'normal'))}")
    print("\nOutput destinations:")
    print(f"  reads posterior  : {posterior_path}")
    print(f"  reads videos     : {experiment_dir}/Experiment_<KIND>_Cell_<n>_{span}S_RAW.tif")
    print(f"  writes report    : {out_dir}")
    print(f"  live progress    : {progress_path}   (tail -f to monitor)")
    print(f"\n{div}\n")

    # ---- Dry-run preview (no GPU, no compute) ----------------------------
    if args.dry_run:
        print("[DRY RUN] validating configuration and inputs:")
        checks = [
            ("posterior", posterior_path),
            ("experiment dir", experiment_dir),
        ]
        missing = 0
        for role, path in checks:
            ok = Path(path).exists()
            missing += not ok
            print(f"  reads {role}: {path}  [{'OK' if ok else 'MISSING'}]")
        for kind in kinds:
            discovered = _discover_cells(experiment_dir, kind, span)
            print(f"  discovered {kind} recordings: {len(discovered)} cell(s) "
                  f"({span}S_RAW.tif)")
        if missing:
            print(f"[DRY RUN] configuration validated; {missing} input(s) MISSING.")
        else:
            print("[DRY RUN] configuration validated; all inputs present.")
        print("[DRY RUN] no MAP estimation performed.")
        return

    run_start = time.time()

    # ---- Diagnostics reporter (the experiment report is the deliverable) -
    reporter = DiagnosticReporter(
        stage="Experiment", enabled=True, dump=True, dump_dir=out_dir,
        run_label=f"{paths.project_alias}_{timing_label}",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
    )
    reporter.check_file("posterior", posterior_path)

    device = get_device()
    vista_device = torch.device("cpu")
    posterior = load_posterior(posterior_path)
    posterior.posterior_estimator.to(device)
    if device.type == "cuda":
        # Rebuild the prior on this device for bounded rejection sampling, matching
        # Evaluation.py (avoids a device mismatch if this GPU index differs from the
        # one the posterior was saved on).
        posterior.prior = build_prior(device=str(device))

    # ---- Output dir + progress log + stale-figure clear ------------------
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.rmtree(out_dir / "figures", ignore_errors=True)

    def log_progress(progress_fh, message: str) -> None:
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = f"[{stamp}] {message}"
        print(line, flush=True)
        if progress_fh is not None:
            progress_fh.write(line + "\n")
            progress_fh.flush()

    def log_file_only(progress_fh, message: str) -> None:
        if progress_fh is not None:
            progress_fh.write(f"           {message}\n")
            progress_fh.flush()

    try:
        progress_fh = open(progress_path, "w", encoding="utf-8")
    except OSError as exc:
        print(f"WARNING: cannot open progress log {progress_path} ({exc}).", flush=True)
        progress_fh = None
    step_log = (lambda m: log_file_only(progress_fh, m)) if debug_log else None

    # ---- MAP estimation over (kind, cell, chunk) -------------------------
    scores, inferred_log10, post_quantiles = [], [], []
    kind_index, cell_of, chunk_of = [], [], []
    log_progress(progress_fh,
                 f"START Experiment MAP: {total_estimates} estimates "
                 f"({len(kinds)} kinds, {n_chunks} chunks/video; pool={theta_prex_size}, "
                 f"elites={elite_prex_size}, steps={numb_steps}; summary={args.summary}).")
    loop_start = time.time()
    done = 0
    try:
        for ki, kind in enumerate(kinds):
            for cell in cells_by_kind[kind]:
                tif_path = paths.experiment_video_path(kind, cell, span, data_bank_root)
                if not tif_path.exists():
                    log_progress(progress_fh, f"SKIP {kind} cell {cell}: file missing "
                                              f"({tif_path.name}).")
                    continue
                raw = tifffile.imread(str(tif_path))                # (frames, H, W) uint16
                video8 = convert_video_dtype(raw, bits_from=16, bits_to=8)
                starts = list(range(0, video8.shape[0] - n_frames + 1, step_frames))
                n_cell_chunks = len(starts)
                cell_start = time.time()
                for c, start in enumerate(starts):
                    chunk = video8[start:start + n_frames]
                    if show:
                        print(f"\n######## MAP estimate: {kind} cell {cell} chunk {c} ########",
                              flush=True)
                    if debug_log:
                        log_file_only(progress_fh, f"-- {kind} cell {cell} chunk {c} --")
                    score, theta_log = map_estimate(
                        posterior, chunk, device, vista_device,
                        theta_prex_size, eval_cfg.theta_prex_batch_size,
                        eval_cfg.score_prex_batch_size, elite_prex_size,
                        numb_steps, eval_cfg.optimizer_patience,
                        eval_cfg.scheduler_patience, show_progress_steps,
                        eval_cfg.learning_rate_minimum, eval_cfg.learning_rate_factor,
                        lr, tolerance, pool_mode=pool_mode, show=show,
                        verbose=verbose_deep, log_fn=step_log,
                    )
                    scores.append(score)
                    inferred_log10.append(theta_log)
                    if do_posterior:
                        post_quantiles.append(posterior_summary(
                            posterior, chunk, device, vista_device,
                            posterior_samples, eval_cfg.theta_prex_batch_size,
                            pool_mode=pool_mode))
                    kind_index.append(ki)
                    cell_of.append(cell)
                    chunk_of.append(c)
                    done += 1
                    elapsed = time.time() - loop_start
                    avg = elapsed / done
                    eta = avg * (total_estimates - done)
                    log_progress(progress_fh,
                                 f"{kind} cell {cell} chunk {c} | {done}/{total_estimates} | "
                                 f"log_prob={score:+.3f} | elapsed={elapsed:.1f}s | "
                                 f"avg={avg:.1f}s | ETA={eta:.0f}s")
                    if debug_log:
                        log_file_only(progress_fh, f"inferred [LOG] {_theta_repr(theta_log)}")
                log_progress(progress_fh, f"{kind} cell {cell} complete "
                                          f"({n_cell_chunks} chunks in "
                                          f"{time.time() - cell_start:.1f}s).")
        log_progress(progress_fh, f"DONE: {done}/{total_estimates} estimates in "
                                  f"{time.time() - loop_start:.1f}s.")
    finally:
        if progress_fh is not None:
            progress_fh.close()

    scores = np.asarray(scores)
    inferred_log10 = np.asarray(inferred_log10)
    kind_index = np.asarray(kind_index)
    n_estimates = inferred_log10.shape[0]
    # (N, D, 5) posterior quantiles [Q05,Q25,Q50,Q75,Q95] when View B was requested.
    post_q = np.asarray(post_quantiles) if (do_posterior and post_quantiles) else None

    # ---- Save the inferred-theta arrays ----------------------------------
    save_arrays = dict(
        inferred_log10=inferred_log10, scores=scores, kind_index=kind_index,
        cell=np.asarray(cell_of), chunk=np.asarray(chunk_of), kinds=np.asarray(kinds))
    if post_q is not None:
        save_arrays["posterior_quantiles"] = post_q   # [Q05,Q25,Q50,Q75,Q95]
    np.savez_compressed(str(array_path), **save_arrays)
    print(f"\nExperiment MAP arrays saved to {array_path}")

    # ---- Report ----------------------------------------------------------
    inferred_by_kind = _aggregate_by_kind(
        inferred_log10, kind_index, np.asarray(cell_of), kinds,
        args.aggregation, len(PARAMETERIZATION))
    agg_desc = ("pooled over (cell x chunk)" if args.aggregation == "pooled"
                else "one point per cell (median over its chunks)")
    reporter.check("estimates_nonempty", n_estimates > 0,
                   f"{n_estimates} MAP estimates over real videos",
                   note="at least one experimental chunk was MAP-estimated.")
    reporter.stat("conditions", len(kinds))
    reporter.stat("total_estimates", n_estimates,
                  note="number of (cell, chunk) windows MAP-estimated across all conditions.")
    reporter.stat("aggregation", args.aggregation, note=f"report distribution view: {agg_desc}.")
    for kind in kinds:
        reporter.stat(f"n[{kind}]", int(inferred_by_kind[kind].shape[0]),
                      note=f"estimates for condition {kind}.")
    if n_estimates:
        reporter.stat("mean_log_prob", float(np.mean(scores)),
                      note="mean MAP log-density at the optimized mode (optimization "
                           "diagnostic; not a calibration/quality metric).")

    headers, rows = experiment_table(PARAMETERIZATION, inferred_by_kind, kinds)
    reporter.table("Inferred theta by condition (log10 units)", headers, rows,
                   note=f"no ground truth for real data; values are the distribution "
                        f"of inferred MAP theta per condition ({agg_desc}). Compare "
                        f"conditions to read out parameter differences.")

    if post_q is not None:
        reporter.stat("posterior_samples", posterior_samples,
                      note="samples per chunk used to summarize the posterior (View B).")

    if reporter.dump and n_estimates:
        for i, para in enumerate(PARAMETERIZATION):
            key = para["KEY"]
            label = para.get("LABEL") or key
            prior_range = para["PRIOR_RANGE"]
            values = ({kind: inferred_by_kind[kind][:, i] for kind in kinds}
                      if do_map else None)
            # View B: per chunk, [MAP, posterior median, q25, q75] by kind.
            by_kind_post = None
            if post_q is not None:
                by_kind_post = {}
                for ki, kind in enumerate(kinds):
                    m = kind_index == ki
                    map_col = inferred_log10[m][:, i:i + 1]               # (n, 1)
                    q_cols = post_q[m][:, i][:, [2, 1, 3]]                # (n, 3): med,q25,q75
                    by_kind_post[kind] = np.hstack([map_col, q_cols])    # (n, 4)
            reporter.save_figure(
                f"experiment_{key}",
                figure_experiment_combined(
                    values, by_kind_post, prior_range, label, seed=args.seed,
                    show_map=do_map, show_posterior=(post_q is not None)),
                caption=f"{key} ({label}). View A (MAP): per-condition distribution of "
                        f"inferred MAP theta ({agg_desc}). View B (posterior): each "
                        f"chunk's posterior median +/- IQR per condition (within-chunk "
                        f"uncertainty). A panel stamped 'not computed' marks a view the "
                        f"--summary option omitted.",
            )

    reporter.summary()
    reporter.write_report()
    print(f"\nTotal elapsed: {time.time() - run_start:.1f}s")


def parse_args(argv=None) -> argparse.Namespace:
    """Construct the CLI parser and parse argv."""
    eval_cfg = PARAMETERS.inference.evaluation
    parser = argparse.ArgumentParser(
        description="MAP-estimate parameters from real experimental videos (no ground truth).",
    )
    parser.add_argument(
        "--total-time-seconds", type=float,
        required=True,
        help="Model window duration in seconds; must match the trained posterior.",
    )
    parser.add_argument(
        "--experiment-span-seconds", type=int, default=20,
        help="Duration of each raw experimental recording in seconds (default: 20).",
    )
    parser.add_argument(
        "--chunk-step-seconds", type=int, default=None,
        help="Step (seconds) between consecutive model-length windows; integer that "
             "divides the window and is <= it. 1 = maximal overlap; window (default) "
             "= non-overlapping. Smaller steps yield more (overlapping) chunks.",
    )
    parser.add_argument(
        "--kinds", type=str, default="ALP,BET",
        help="Comma-separated experimental conditions (default: 'ALP,BET').",
    )
    parser.add_argument(
        "--cells", type=str, default=None,
        help="Comma-separated explicit cell indices (default: discover all on disk).",
    )
    parser.add_argument(
        "--max-cells", type=int, default=0,
        help="Cap on cells per kind (0 = all; useful for quick checks).",
    )
    parser.add_argument(
        "--summary", choices=("map", "posterior", "both"), default="map",
        help="Which views to render: 'map' (View A: MAP-point distribution per "
             "condition; default), 'posterior' (View B: per-chunk posterior median "
             "+/- IQR per condition), or 'both'. View B draws --posterior-samples per chunk.",
    )
    parser.add_argument(
        "--posterior-samples", type=int, default=None,
        help=f"Samples per chunk for View B posterior summary "
             f"(default: {eval_cfg.posterior_samples}).",
    )
    parser.add_argument(
        "--aggregation", choices=("pooled", "cell-median"), default="pooled",
        help="Report distribution view: 'pooled' (every cell x chunk estimate is a "
             "sample; mixes temporal + biological variation) or 'cell-median' (one "
             "sample per cell, median over its chunks; biological spread only). "
             "Default: pooled. The .npz keeps raw per-(cell,chunk) data either way.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help="Master RNG seed (PyTorch + numpy + Python random). Default None "
             "-> non-deterministic (consistent with generation); pass an int for a "
             "reproducible run.",
    )
    parser.add_argument(
        "--pool-mode", choices=("bounded", "unrestricted"), default=None,
        help=f"Candidate-pool sampler (default: {eval_cfg.pool_mode}). See Evaluation.py.",
    )
    parser.add_argument("--theta-prex-size", type=int, default=None,
                        help=f"Candidate-pool size per video (default: {eval_cfg.theta_prex_size}).")
    parser.add_argument("--elite-prex-size", type=int, default=None,
                        help=f"Number of optimization seeds (default: {eval_cfg.elite_prex_size}).")
    parser.add_argument("--numb-steps", type=int, default=None,
                        help=f"Max gradient-ascent steps (default: {eval_cfg.numb_steps}).")
    parser.add_argument("--show-progress-steps", type=int, default=None,
                        help=f"Per-step progress cadence (default: {eval_cfg.show_progress_steps}).")
    parser.add_argument("--learning-rate", type=float, default=None,
                        help="Adam learning rate for the MAP optimization.")
    parser.add_argument("--verbose", action="store_true",
                        help="Rich per-chunk console diagnostics (see Evaluation.py).")
    parser.add_argument("--debug", action="store_true",
                        help="Implies --verbose; deeper console + per-step detail in progress.log.")
    parser.add_argument("--debug-dump", action="store_true",
                        help="Implies --debug; tees the console transcript to "
                             "Labor/Debug/<run>/Experiment/console.log.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate configuration and inputs, print what would be read/written, then exit "
                             "without running the stage (no GPU, no compute). Use before a queue submission or a long local run.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    cli_args = parse_args(sys.argv[1:])
    with console_log_context(cli_args, "Experiment"):
        main(cli_args)

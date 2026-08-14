"""Shared Test-Loss-Distribution analysis engine for both DIMER workflows.

``run_test_loss_analysis(cfg, args)`` reads a per-example test-loss distribution artifact and
produces the picture the reported mean cannot give: the distribution shape, the uniform-prior
NLL reference, and the tail-vs-parameter (identifiability) read -- the whole analysis lives in
the workflow-agnostic ``test_loss_analysis`` kernel. The two entry-point shims shrink to:
build the ``WorkflowConfig``, parse args, call this runner -- the same pattern as the stages.

Two input modes. With ``--tld-path`` any ``.npz`` artifact is analyzed verbatim (no machine
profile needed) -- workflow-independent. Without it the canonical artifact is resolved from
the active machine profile + ``--total-time-seconds`` through the workflow's own
alias-qualified ``paths`` (``cfg.paths``): the biology shim resolves the biology TLD, the
detector shim the ``_DETECTOR`` one. That alias-qualified path resolution is the *only*
per-workflow difference; everything downstream reads from the artifact's manifest.

An Analysis diagnostic -- read-only, never wired into the ``Submit.sh`` dispatcher, no GPU.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

from srm_and_sbi_dimer_alp import test_loss_analysis as tla
from srm_and_sbi_dimer_alp.diagnostics import DiagnosticReporter
from srm_and_sbi_dimer_alp.test_loss_distribution import TestLossDistribution
from srm_and_sbi_dimer_alp.workflow import WorkflowConfig


def _resolve_paths(cfg: WorkflowConfig, args):
    """Resolve the input artifact path + the output directory.

    ``--tld-path`` -> the artifact verbatim (workflow-independent; no profile consulted).
    Otherwise the canonical artifact of THIS workflow, from the machine profile +
    ``--total-time-seconds`` via ``cfg.paths`` (biology's or the ``_DETECTOR`` alias).
    """
    if args.tld_path:
        tld_path = Path(args.tld_path)
        out_dir = (Path(args.outdir) if args.outdir
                   else tld_path.parent / f"{tld_path.stem}_Analysis")
        return tld_path, out_dir

    from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming  # lazy

    if args.total_time_seconds is None:
        raise SystemExit(
            "canonical mode needs --total-time-seconds (sets the timing label); "
            "or pass --tld-path to analyze a specific artifact directly.")
    paths = cfg.paths
    data_bank_root = PARAMETERS.machine.data_bank_root
    timing_label = RunTiming(total_time_seconds=args.total_time_seconds,
                             frames=PARAMETERS.simulation.timing).label
    tld_path = paths.test_loss_distribution_path(data_bank_root, timing_label)
    out_dir = (Path(args.outdir) if args.outdir
               else tld_path.parent / f"{tld_path.stem}_Analysis")
    return tld_path, out_dir


def run_test_loss_analysis(cfg: WorkflowConfig, args: argparse.Namespace) -> int:
    """Analyze the per-example test-loss distribution for the given workflow + CLI args."""
    tld_path, out_dir = _resolve_paths(cfg, args)

    print("=" * 74)
    print(f"Test-loss distribution analysis  [{cfg.tag}]")
    print(f"  artifact : {tld_path}")
    print(f"  output   : {out_dir}/")
    print("=" * 74)

    if args.dry_run:
        print("[DRY RUN] would read the artifact above and write:\n"
              f"    {out_dir}/report.md\n"
              f"    {out_dir}/figures/{{nll_histogram,nll_ecdf,loss_vs_theta,tail_drivers}}.png")
        return 0

    if not tld_path.exists():
        print(f"FATAL: artifact not found: {tld_path}", file=sys.stderr)
        return 1

    tld = TestLossDistribution.load(tld_path)
    manifest = tld.manifest
    production_date = datetime.fromtimestamp(tld_path.stat().st_mtime).strftime("%Y-%m-%d %H:%M:%S")

    out_dir.mkdir(parents=True, exist_ok=True)
    reporter = DiagnosticReporter(
        stage="Test_Loss_Distribution_Analysis", enabled=True, dump=True, dump_dir=out_dir,
        run_label=f"{manifest.get('project_alias', '?')}_{manifest.get('timing_label', '?')}",
        timestamp=f"analysis of artifact produced {production_date}",
    )

    # ---- Preconditions ---------------------------------------------------
    loss_all = np.asarray(tld.loss, dtype=np.float64)
    finite = np.isfinite(loss_all)
    n_total, n_finite = int(loss_all.size), int(finite.sum())
    reporter.check("has_finite_losses", n_finite > 0, f"{n_finite}/{n_total} finite", fatal=True)
    loss = loss_all[finite]

    entries = tla.learnable_entries(manifest)
    keys = list(manifest.get("theta_keys", []))
    reporter.check("theta_keys_match_learnable_table",
                   [e["key"] for e in entries] == keys,
                   f"{len(keys)} learnable columns", fatal=True,
                   note="The theta columns line up with the learnable rows of the manifest "
                        "parameter table, in declaration order.")
    space = manifest.get("theta_space", "?")
    reporter.check("theta_space_is_log10", space == "log10", f"theta_space={space}", fatal=False,
                   note="The uniform-prior reference is computed in the space the density is "
                        "scored in; it assumes log10 theta.")

    nll_prior, widths = tla.prior_reference_nll(entries)
    threshold = args.tail_threshold if args.tail_threshold is not None else nll_prior

    # ---- Provenance ------------------------------------------------------
    reporter.table(
        "Artifact provenance", ["field", "value"],
        [["artifact path", str(tld_path)],
         ["produced (file mtime)", production_date],
         ["project alias", manifest.get("project_alias", "?")],
         ["timing label", manifest.get("timing_label", "?")],
         ["theta space", space],
         ["best epoch", manifest.get("epoch", "?")],
         ["best test NLL (manifest)", f"{manifest.get('best_test_loss', float('nan')):.6f}"],
         ["test examples", f"{n_finite} finite / {n_total} total"],
         ["train / test videos",
          f"{manifest.get('train_videos', '?')} / {manifest.get('test_videos', '?')}"],
         ["learnable parameters", str(len(keys))],
         ["torch version", manifest.get("torch_version", "?")],
         ["artifact format version", manifest.get("artifact_format_version", "?")]],
        note="Read from the artifact manifest and its filesystem timestamp; every run of this "
             "analysis records the exact artifact it consumed.")

    # ---- Distribution shape + reference ----------------------------------
    card = tld.extended_card(tail_threshold=threshold, n_boot=args.n_boot,
                             ci_level=0.95, boot_seed=args.seed)
    mean, median = card["mean"], card["median"]
    reporter.check("mean_matches_manifest_best",
                   abs(mean - float(manifest.get("best_test_loss", np.nan))) < 1e-3,
                   f"array mean {mean:.6f} vs manifest {manifest.get('best_test_loss')}",
                   fatal=False,
                   note="The stored loss array reproduces the selection metric (mean test NLL).")
    reporter.stat("mean_test_nll", round(mean, 6),
                  note="Mean per-example NLL over TEST (the training selection metric).")
    reporter.stat("median_test_nll", round(median, 6),
                  note="Median is more robust to the heavy upper tail than the mean.")
    reporter.stat("std_test_nll", round(card["std"], 6))
    reporter.stat("skew_test_nll", round(card["skew"], 4),
                  note="Fisher-Pearson skew; positive = heavy upper (catastrophic) tail.")
    reporter.stat("mean_ci95", f"[{card['mean_ci95'][0]:.4f}, {card['mean_ci95'][1]:.4f}]",
                  note=f"Percentile bootstrap ({args.n_boot} resamples), post-hoc.")
    reporter.stat("uniform_prior_nll", round(nll_prior, 6),
                  note="No-information baseline = log prior-box volume; an estimator that "
                       "learned nothing scores this.")
    reporter.stat("information_gain_nats", round(nll_prior - mean, 6),
                  note="How far the mean NLL sits below the prior baseline (nats gained).")
    reporter.stat("worse_than_prior_fraction", round(card["tail_mass"], 6), expected="small",
                  note=f"Fraction of examples with NLL > {threshold:.3f} (the estimator did "
                       "worse than the prior there).")

    q_values = np.percentile(loss, tla.SPREAD_QUANTILES)
    reporter.table(
        "Distribution spread (quantiles of per-example NLL)", ["quantile", "NLL (nats)"],
        [[f"p{q}", f"{v:.4f}"] for q, v in zip(tla.SPREAD_QUANTILES, q_values)]
        + [["min", f"{loss.min():.4f}"], ["max", f"{loss.max():.4f}"]],
        note="Lower NLL = the posterior places more density on the true parameters. GOOD: the "
             "median well below the prior baseline and a modest gap between the bulk (p25-p75) "
             "and the worst case (max).")

    # ---- Tail versus parameter space (the identifiability read) ----------
    if tld.theta is None or tld.theta.shape[1] != len(keys):
        reporter.stat("tail_vs_theta", "skipped",
                      note="The artifact carries no per-example theta, so the "
                           "tail-vs-parameter analysis cannot run.")
    else:
        theta = np.asarray(tld.theta, dtype=np.float64)[finite]
        rows, n_hard, n_bulk, cut = tla.tail_versus_theta(loss, theta, keys, widths, args.hard_quantile)
        reporter.stat("hard_tail_definition",
                      f"NLL >= {cut:.4f}  (top {(1 - args.hard_quantile) * 100:.0f}%)",
                      note=f"{n_hard} hard vs {n_bulk} bulk examples.")
        top = rows[0] if rows else None
        if top is not None and np.isfinite(top["ks_stat"]):
            end = "low" if top["mean_shift"] < 0 else "high"
            reporter.stat(
                "hardest_regime", f"{top['key']} ({end} end)  KS D={top['ks_stat']:.3f}",
                note="The parameter (and end of its range) that most sets the hardest examples "
                     "apart -- the least-identifiable regime. Whether that hardness is an honest "
                     "identifiability limit (a wide but calibrated posterior there) or "
                     "overconfidence is answered by the Posterior_Calibration diagnostic, "
                     "stratified by inferred value; this locates WHERE, that measures WHETHER.")
        reporter.table(
            "Per-parameter tail analysis (ranked by KS D)",
            ["parameter", "prior range (log10)", "Spearman rho", "rho p (BH)",
             "mean shift (hard-bulk)", "KS D", "KS p (BH)"],
            [[r["key"], f"{r['width']:.3g}", f"{r['rho']:+.3f}", f"{r['rho_p_bh']:.3g}",
              f"{r['mean_shift']:+.3f}", f"{r['ks_stat']:.3f}", f"{r['ks_p_bh']:.3g}"]
             for r in rows],
            note="Spearman rho = monotone NLL-vs-value trend over all examples; KS D and the "
                 "mean shift compare the hardest examples against the rest. Larger |rho| or KS "
                 "D = that parameter more strongly marks the hard examples, and the sign shows "
                 "which end of its range is harder (negative = harder at low values). p-values "
                 "are Benjamini-Hochberg-adjusted; with this many examples nearly all are "
                 "significant, so judge difficulty by the effect sizes (rho, KS D), not the "
                 "p-values.")
        reporter.save_figure(
            "loss_vs_theta", tla.figure_loss_vs_theta(loss, theta, keys, nll_prior),
            caption="P3. Joint density of NLL against each learnable parameter (hexbin = "
                    "examples per cell, log scale; orange = median NLL across the range; red "
                    "dashed = uniform-prior baseline). GOOD: the orange line stays low and flat "
                    "(uniform skill). BAD: it rises over part of the range -- harder for those "
                    "parameter values (e.g. a rise toward low count = harder in uninformative "
                    "videos).")
        reporter.save_figure(
            "tail_drivers", tla.figure_tail_drivers(rows),
            caption="P4. Which parameters characterize the hardest examples. KS D (red) = how "
                    "differently that parameter is distributed among the hard examples vs the "
                    "rest; |Spearman rho| (blue) = how strongly NLL trends with the parameter. "
                    "Longer top bars = the regimes the estimator finds hardest -- a pointer for "
                    "where to look, confirmed by Evaluation recovery + Posterior_Calibration.")

    # ---- Distribution figures --------------------------------------------
    reporter.save_figure(
        "nll_histogram", tla.figure_histogram(loss, mean, median, nll_prior),
        caption="P1. Histogram of the per-example test NLL (further left = better fit). GOOD: "
                "most mass far left, single peak, thin tail to the red prior line. BAD: mass "
                "near/past the red line, or a heavy right tail. The red line is what an "
                "uninformed (prior) model scores.")
    reporter.save_figure(
        "nll_ecdf", tla.figure_ecdf(loss, nll_prior),
        caption="P2. Empirical CDF of the per-example NLL. GOOD: a steep rise far left of the "
                "red prior line, reaching ~1 before it (worse-than-prior fraction near zero).")

    reporter.summary()
    report_path = reporter.write_report()
    print(f"\nReport written: {report_path}")
    return 0


def build_test_loss_analysis_parser() -> argparse.ArgumentParser:
    """Construct the Test-Loss-Distribution analysis CLI parser (shared by both shims)."""
    parser = argparse.ArgumentParser(
        description="Analyze a saved per-example test-loss distribution artifact "
                    "(distribution shape + uniform-prior reference + tail-vs-theta identifiability).")
    parser.add_argument(
        "--tld-path", type=str, default=None,
        help="Path to a Test_Loss_Distribution .npz. If omitted, this workflow's canonical "
             "artifact is resolved from the machine profile + --total-time-seconds.")
    parser.add_argument(
        "--total-time-seconds", type=float, default=None,
        help="Run duration in seconds; sets the timing label used to resolve the canonical "
             "artifact (required unless --tld-path is given).")
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory (default: a *_Analysis directory beside the artifact).")
    parser.add_argument(
        "--hard-quantile", type=float, default=0.95,
        help="Quantile above which examples form the 'hard tail' (default 0.95 = top 5%%).")
    parser.add_argument(
        "--tail-threshold", type=float, default=None,
        help="Fixed NLL for the worse-than-baseline fraction (default: the computed "
             "uniform-prior NLL reference).")
    parser.add_argument(
        "--n-boot", type=int, default=1000,
        help="Bootstrap resamples for the mean confidence interval (default 1000).")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the (post-hoc) bootstrap; keeps the CI reproducible.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and print the input/output paths and write nothing.")
    return parser

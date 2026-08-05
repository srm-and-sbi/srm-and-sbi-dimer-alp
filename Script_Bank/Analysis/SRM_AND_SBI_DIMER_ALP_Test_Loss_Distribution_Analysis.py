"""Post-hoc analysis of a saved per-example test-loss distribution artifact.

The Inference stage records, at its best epoch, one per-example negative
log-likelihood (NLL) over the fixed held-out TEST set -- the ``.npz`` snapshot
written by ``TestLossDistribution`` (parallel ``keys`` / ``theta`` / ``loss``
arrays plus a self-describing ``manifest``). The single scalar the training log
reports is only the mean of that array. This script reads the whole
distribution back and produces the picture the mean alone cannot give: the
shape of the spread, an interpretable reference for what the numbers mean, and
which regions of parameter space the estimator finds hard.

Everything is read from the artifact's own manifest (parameter keys, roles,
prior ranges, log flags, provenance), so the analysis stays correct even if the
live parameterization later changes, and generalizes to any number of learnable
parameters without edits.

What it computes
----------------
A. Distribution shape.
   Mean, median, standard deviation, Fisher-Pearson skewness, min/max, a
   quantile spread (1/5/25/50/75/90/95/99%), and a percentile-bootstrap 95%
   confidence interval on the mean. A right skew and a heavy upper tail flag
   the "catastrophic miss" examples the mean conceals.

   Reference -- the uniform-prior NLL. For a uniform prior over the learnable
   box the expected NLL is a constant equal to the log prior-box volume,
   ``NLL_prior = sum_j ln(range_j)`` (natural log; ranges in the theta space the
   density is scored in, here log10). This is the no-information baseline: an
   estimator that learned nothing scores exactly this. The mean NLL sits below
   it by the information gained (in nats), and any per-example NLL *above* it is
   a case where the estimator did worse than the prior -- a principled
   "catastrophic miss" line, reported as the worse-than-prior fraction.

B. Tail versus parameter space.
   For each learnable parameter: the Spearman rank correlation between the
   per-example NLL and the parameter value (monotone association), and a
   comparison of the hard tail (the top ``1 - hard_quantile`` of NLL) against
   the bulk -- the mean shift and a two-sample Kolmogorov-Smirnov statistic.
   Together these locate which parameters, and which regions of their range,
   drive the hard examples. p-values are Benjamini-Hochberg-adjusted across all
   per-parameter tests (false-discovery-rate control over the family).

Calibration and posterior coverage (SBC / TARP) are deliberately out of scope:
they require posterior *samples* per example, which this NLL artifact does not
carry. Those belong to the Evaluation stage (MAP recovery on the EVAL
namespace), not here.

Outputs (under the machine-profile ``Posit/`` tier, alongside the artifact):
    <data_bank>/<posit_subdir>/<project_alias>_{timing_label}_Test_Loss_Distribution_Analysis/
        report.md                     -- provenance + distribution + tail tables,
                                         with the figures embedded and a legend.
        figures/nll_histogram.png     -- P1: NLL histogram (mean/median/prior lines).
        figures/nll_ecdf.png          -- P2: NLL empirical CDF (prior line).
        figures/loss_vs_theta.png     -- P3: per-parameter NLL-vs-theta hexbins.
        figures/tail_drivers.png      -- P4: ranked per-parameter tail-driver bars.

Run (canonical artifact, resolved from the machine profile):
    MACHINE_PROFILE=<profile> python \
        Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Test_Loss_Distribution_Analysis.py \
        --total-time-seconds 2.0

Run (ad hoc, any artifact -- no machine profile needed):
    python Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Test_Loss_Distribution_Analysis.py \
        --tld-path /path/to/..._Test_Loss_Distribution.npz --outdir /path/to/out

``--dry-run`` resolves and prints the input/output paths and writes nothing.
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
from matplotlib.colors import LogNorm
from matplotlib.figure import Figure
from scipy.stats import ks_2samp, spearmanr

from srm_and_sbi_dimer_alp.diagnostics import DiagnosticReporter
from srm_and_sbi_dimer_alp.test_loss_distribution import TestLossDistribution

# Quantiles reported in the spread table (percent).
_SPREAD_QUANTILES = (1, 5, 25, 50, 75, 90, 95, 99)


# --------------------------------------------------------------------------- #
# Path resolution
# --------------------------------------------------------------------------- #
def resolve_paths(args):
    """Resolve the input artifact path and the output directory.

    Two modes. With ``--tld-path`` the artifact is taken verbatim and the output
    directory is ``--outdir`` (or a sibling ``*_Analysis`` directory next to the
    artifact) -- no machine profile is consulted, so any ``.npz`` anywhere can be
    analyzed. Without it the canonical Detector artifact is resolved from the
    active machine profile and ``--total-time-seconds`` (the timing label), and
    the output directory defaults under the ``Posit/`` tier alongside it.
    """
    if args.tld_path:
        tld_path = Path(args.tld_path)
        out_dir = (Path(args.outdir) if args.outdir
                   else tld_path.parent / f"{tld_path.stem}_Analysis")
        return tld_path, out_dir

    # Canonical mode: import the profile-dependent machinery lazily so the ad hoc
    # mode above never needs a configured MACHINE_PROFILE.
    from srm_and_sbi_dimer_alp import detector_parameterization as det
    from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming

    if args.total_time_seconds is None:
        raise SystemExit(
            "canonical mode needs --total-time-seconds (sets the timing label); "
            "or pass --tld-path to analyze a specific artifact directly.")
    paths = det.detector_paths(PARAMETERS.paths)
    data_bank_root = PARAMETERS.machine.data_bank_root
    timing_label = RunTiming(total_time_seconds=args.total_time_seconds,
                             frames=PARAMETERS.simulation.timing).label
    tld_path = paths.test_loss_distribution_path(data_bank_root, timing_label)
    out_dir = (Path(args.outdir) if args.outdir
               else tld_path.parent / f"{tld_path.stem}_Analysis")
    return tld_path, out_dir


# --------------------------------------------------------------------------- #
# Manifest interpretation (all read from the artifact, nothing hardcoded)
# --------------------------------------------------------------------------- #
def learnable_entries(manifest):
    """The learnable-parameter rows of the manifest's parameter table, in the
    declaration order that matches the ``theta`` columns / ``theta_keys``."""
    table = manifest.get("parameter_table", [])
    return [entry for entry in table if entry.get("role") == "learnable"]


def prior_reference_nll(entries):
    """Uniform-prior NLL = sum of ln(prior-range width) over learnable params.

    The no-information baseline: for a uniform prior over the learnable box the
    expected NLL is the constant log prior-box volume. Returns ``(nll_prior,
    widths)`` where ``widths[j] = high_j - low_j`` in the parameter's own units
    (log10 here), matching the space the density is scored in.
    """
    widths = []
    for entry in entries:
        low, high = entry["prior_range"][0], entry["prior_range"][1]
        width = float(high) - float(low)
        if width <= 0.0:
            raise ValueError(
                f"parameter {entry.get('key')!r} has a non-positive prior "
                f"range width ({width}); cannot form the prior reference.")
        widths.append(width)
    nll_prior = float(np.sum(np.log(widths)))
    return nll_prior, widths


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #
def benjamini_hochberg(pvalues):
    """Benjamini-Hochberg step-up adjusted p-values (false-discovery-rate).

    Version-independent (no scipy dependency): rank the p-values, scale each by
    ``n / rank``, enforce monotonicity from the largest down, and clip to 1.
    """
    p = np.asarray(pvalues, dtype=float)
    n = p.size
    if n == 0:
        return p
    order = np.argsort(p)
    scaled = p[order] * n / (np.arange(n) + 1.0)
    scaled = np.minimum.accumulate(scaled[::-1])[::-1]
    adjusted = np.empty(n, dtype=float)
    adjusted[order] = np.clip(scaled, 0.0, 1.0)
    return adjusted


def tail_versus_theta(loss, theta, keys, widths, hard_quantile):
    """Per-parameter tail analysis: Spearman association + hard-vs-bulk shift.

    ``loss`` and ``theta`` are already restricted to finite-loss rows. The hard
    tail is the ``loss >= quantile(loss, hard_quantile)`` subset; the bulk is the
    remainder. Returns a list of per-parameter dicts plus the split sizes. All
    raw p-values (Spearman across all rows, KS hard-vs-bulk) are pooled and
    Benjamini-Hochberg-adjusted together.
    """
    cut = float(np.quantile(loss, hard_quantile))
    hard_mask = loss >= cut
    bulk_mask = ~hard_mask
    n_hard, n_bulk = int(hard_mask.sum()), int(bulk_mask.sum())

    rows, raw_p = [], []
    for j, key in enumerate(keys):
        column = theta[:, j]
        rho, rho_p = spearmanr(column, loss)
        if n_hard >= 1 and n_bulk >= 1:
            ks_stat, ks_p = ks_2samp(column[hard_mask], column[bulk_mask])
            mean_shift = float(column[hard_mask].mean() - column[bulk_mask].mean())
        else:
            ks_stat, ks_p, mean_shift = float("nan"), float("nan"), float("nan")
        rows.append({
            "key": key, "width": widths[j],
            "rho": float(rho), "rho_p": float(rho_p),
            "mean_shift": mean_shift,
            "ks_stat": float(ks_stat), "ks_p": float(ks_p),
        })
        raw_p.extend([float(rho_p), float(ks_p)])

    adjusted = benjamini_hochberg([p for p in raw_p if np.isfinite(p)])
    # Map adjusted values back onto the finite raw p-values, in order.
    it = iter(adjusted)
    finite_adj = {i: next(it) for i, p in enumerate(raw_p) if np.isfinite(p)}
    for k, row in enumerate(rows):
        row["rho_p_bh"] = float(finite_adj.get(2 * k, float("nan")))
        row["ks_p_bh"] = float(finite_adj.get(2 * k + 1, float("nan")))

    # Rank by the hard-vs-bulk separation (the tail-specific signal).
    rows.sort(key=lambda r: (-1.0 if np.isnan(r["ks_stat"]) else -r["ks_stat"]))
    return rows, n_hard, n_bulk, cut


# --------------------------------------------------------------------------- #
# Figures (headless Figure objects; the reporter saves them via Agg)
# --------------------------------------------------------------------------- #
def figure_histogram(loss, mean, median, nll_prior):
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.hist(loss, bins=60, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.2)
    ax.axvline(mean, color="#1f1f1f", linestyle="-", linewidth=1.4,
               label=f"mean = {mean:.2f}")
    ax.axvline(median, color="#55A868", linestyle="-.", linewidth=1.4,
               label=f"median = {median:.2f}")
    ax.axvline(nll_prior, color="#C44E52", linestyle="--", linewidth=1.6,
               label=f"uniform-prior NLL = {nll_prior:.2f}")
    ax.set_xlabel("per-example test NLL (nats)")
    ax.set_ylabel("count")
    ax.set_title("Per-example test-loss distribution")
    ax.legend(frameon=False)
    fig.tight_layout()
    return fig


def figure_ecdf(loss, nll_prior):
    xs = np.sort(loss)
    ys = np.arange(1, xs.size + 1) / xs.size
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(xs, ys, color="#4C72B0", linewidth=1.6)
    ax.axvline(nll_prior, color="#C44E52", linestyle="--", linewidth=1.6,
               label=f"uniform-prior NLL = {nll_prior:.2f}")
    worse = float(np.mean(loss > nll_prior))
    ax.set_xlabel("per-example test NLL (nats)")
    ax.set_ylabel("cumulative fraction")
    ax.set_title(f"Empirical CDF  (worse-than-prior fraction = {worse:.4f})")
    ax.set_ylim(0.0, 1.0)
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


def figure_loss_vs_theta(loss, theta, keys, nll_prior):
    """Per-parameter joint density of NLL vs parameter value, with the median trend.

    Hexbin color is the number of examples per cell on a shared log scale (bright =
    many, dark = few), so the concentration is legible and comparable across panels;
    the orange line is the median NLL across equal-count bins of the parameter,
    making the trend explicit; the red dashed line is the uniform-prior baseline.
    """
    n = len(keys)
    ncol = min(5, n)
    nrow = int(np.ceil(n / ncol))
    fig = Figure(figsize=(3.5 * ncol, 3.1 * nrow), layout="constrained")
    axes, hexbins = [], []
    for j, key in enumerate(keys):
        ax = fig.add_subplot(nrow, ncol, j + 1)
        hb = ax.hexbin(theta[:, j], loss, gridsize=28, mincnt=1, cmap="viridis")
        axes.append(ax)
        hexbins.append(hb)
        # Median NLL across equal-count bins of the parameter: the trend line.
        order = np.argsort(theta[:, j])
        xs, ys = theta[order, j], loss[order]
        edges = np.linspace(0, xs.size, 16, dtype=int)
        bx = [xs[a:b].mean() for a, b in zip(edges[:-1], edges[1:]) if b > a]
        bm = [np.median(ys[a:b]) for a, b in zip(edges[:-1], edges[1:]) if b > a]
        ax.plot(bx, bm, color="#FF7F0E", linewidth=1.7, marker="o", markersize=2.6,
                label="median NLL")
        ax.axhline(nll_prior, color="#C44E52", linestyle="--", linewidth=0.9)
        ax.set_title(key, fontsize=9)
        ax.set_xlabel("value (log10)", fontsize=7)
        ax.set_ylabel("NLL", fontsize=7)
        ax.tick_params(labelsize=6)
    # Shared log-count normalization so bright/dark means the same in every panel.
    vmax = max(2.0, max(float(hb.get_array().max()) for hb in hexbins))
    norm = LogNorm(vmin=1, vmax=vmax)
    for hb in hexbins:
        hb.set_norm(norm)
    for j in range(n, nrow * ncol):  # blank any unused grid cells
        fig.add_subplot(nrow, ncol, j + 1).axis("off")
    cbar = fig.colorbar(hexbins[-1], ax=axes, fraction=0.03, pad=0.01)
    cbar.set_label("examples per cell (log scale)", fontsize=8)
    if axes:
        axes[min(len(axes) - 1, ncol - 1)].legend(fontsize=6, frameon=False,
                                                   loc="upper right")
    fig.suptitle("Per-example NLL vs each learnable parameter", fontsize=11)
    return fig


def figure_tail_drivers(rows):
    keys = [r["key"] for r in rows]
    ks = [r["ks_stat"] for r in rows]
    rho = [abs(r["rho"]) for r in rows]
    y = np.arange(len(keys))
    fig = Figure(figsize=(8, max(3.0, 0.5 * len(keys) + 1.5)))
    ax = fig.add_subplot(111)
    ax.barh(y + 0.2, ks, height=0.4, color="#C44E52", label="KS D (hard vs bulk)")
    ax.barh(y - 0.2, rho, height=0.4, color="#4C72B0", label="|Spearman rho|")
    ax.set_yticks(y)
    ax.set_yticklabels(keys)
    ax.invert_yaxis()  # strongest tail driver on top
    ax.set_xlabel("statistic")
    ax.set_title("Per-parameter tail drivers (ranked by KS D)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Analyze a saved per-example test-loss distribution artifact "
                    "(distribution shape + uniform-prior reference + tail-vs-theta).")
    parser.add_argument(
        "--tld-path", type=str, default=None,
        help="Path to a Test_Loss_Distribution .npz. If omitted, the canonical "
             "Detector artifact is resolved from the machine profile.")
    parser.add_argument(
        "--total-time-seconds", type=float, default=None,
        help="Run duration in seconds; sets the timing label used to resolve the "
             "canonical artifact (required unless --tld-path is given).")
    parser.add_argument(
        "--outdir", type=str, default=None,
        help="Output directory (default: a *_Analysis directory beside the artifact).")
    parser.add_argument(
        "--hard-quantile", type=float, default=0.95,
        help="Quantile above which examples form the 'hard tail' (default 0.95 = top 5%%).")
    parser.add_argument(
        "--tail-threshold", type=float, default=None,
        help="Fixed NLL for the worse-than-baseline fraction (default: the "
             "computed uniform-prior NLL reference).")
    parser.add_argument(
        "--n-boot", type=int, default=1000,
        help="Bootstrap resamples for the mean confidence interval (default 1000).")
    parser.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the (post-hoc) bootstrap; keeps the CI reproducible.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and print the input/output paths and write nothing.")
    return parser.parse_args(argv)


def main(args):
    tld_path, out_dir = resolve_paths(args)

    print("=" * 74)
    print("Test-loss distribution analysis")
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
    production_date = datetime.fromtimestamp(tld_path.stat().st_mtime).strftime(
        "%Y-%m-%d %H:%M:%S")

    out_dir.mkdir(parents=True, exist_ok=True)
    reporter = DiagnosticReporter(
        stage="Test_Loss_Distribution_Analysis",
        enabled=True, dump=True, dump_dir=out_dir,
        run_label=f"{manifest.get('project_alias', '?')}_{manifest.get('timing_label', '?')}",
        timestamp=f"analysis of artifact produced {production_date}",
    )

    # ---- Preconditions --------------------------------------------------- #
    loss_all = np.asarray(tld.loss, dtype=np.float64)
    finite = np.isfinite(loss_all)
    n_total, n_finite = int(loss_all.size), int(finite.sum())
    reporter.check("has_finite_losses", n_finite > 0,
                   f"{n_finite}/{n_total} finite", fatal=True)
    loss = loss_all[finite]

    entries = learnable_entries(manifest)
    keys = list(manifest.get("theta_keys", []))
    reporter.check("theta_keys_match_learnable_table",
                   [e["key"] for e in entries] == keys,
                   f"{len(keys)} learnable columns", fatal=True,
                   note="The theta columns line up with the learnable rows of the "
                        "manifest parameter table, in declaration order.")
    space = manifest.get("theta_space", "?")
    reporter.check("theta_space_is_log10", space == "log10",
                   f"theta_space={space}", fatal=False,
                   note="The uniform-prior reference is computed in the space the "
                        "density is scored in; it assumes log10 theta.")

    nll_prior, widths = prior_reference_nll(entries)
    threshold = args.tail_threshold if args.tail_threshold is not None else nll_prior

    # ---- T0. Provenance -------------------------------------------------- #
    reporter.table(
        "Artifact provenance",
        ["field", "value"],
        [
            ["artifact path", str(tld_path)],
            ["produced (file mtime)", production_date],
            ["project alias", manifest.get("project_alias", "?")],
            ["timing label", manifest.get("timing_label", "?")],
            ["test set", manifest.get("test_set_id", "?")],
            ["theta space", space],
            ["best epoch", manifest.get("epoch", "?")],
            ["best test NLL (manifest)", f"{manifest.get('best_test_loss', float('nan')):.6f}"],
            ["test examples", f"{n_finite} finite / {n_total} total"],
            ["train / test videos",
             f"{manifest.get('train_videos', '?')} / {manifest.get('test_videos', '?')}"],
            ["epochs planned", manifest.get("epochs_planned", "?")],
            ["learnable parameters", str(len(keys))],
            ["torch version", manifest.get("torch_version", "?")],
            ["artifact format version", manifest.get("artifact_format_version", "?")],
        ],
        note="Read from the artifact manifest and its filesystem timestamp; every "
             "run of this analysis records the exact artifact it consumed.")

    # ---- T1 / A. Distribution shape + reference -------------------------- #
    card = tld.extended_card(tail_threshold=threshold, n_boot=args.n_boot,
                             ci_level=0.95, boot_seed=args.seed)
    mean, median = card["mean"], card["median"]
    reporter.check("mean_matches_manifest_best",
                   abs(mean - float(manifest.get("best_test_loss", np.nan))) < 1e-3,
                   f"array mean {mean:.6f} vs manifest {manifest.get('best_test_loss')}",
                   fatal=False,
                   note="The stored loss array reproduces the selection metric "
                        "(mean test NLL) recorded in the manifest.")

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
                  note="No-information baseline = log prior-box volume; an estimator "
                       "that learned nothing scores this.")
    reporter.stat("information_gain_nats", round(nll_prior - mean, 6),
                  note="How far the mean NLL sits below the prior baseline (nats gained).")
    reporter.stat("worse_than_prior_fraction", round(card["tail_mass"], 6),
                  expected="small",
                  note=f"Fraction of examples with NLL > {threshold:.3f} "
                       "(the estimator did worse than the prior there).")

    q_values = np.percentile(loss, _SPREAD_QUANTILES)
    reporter.table(
        "Distribution spread (quantiles of per-example NLL)",
        ["quantile", "NLL (nats)"],
        [[f"p{q}", f"{v:.4f}"] for q, v in zip(_SPREAD_QUANTILES, q_values)]
        + [["min", f"{loss.min():.4f}"], ["max", f"{loss.max():.4f}"]],
        note="Lower NLL = the posterior places more density on the true parameters "
             "(better fit). GOOD: the median well below the prior baseline and a modest "
             "gap between the bulk (p25-p75) and the worst case (max). min/max are the "
             "best and worst single examples.")

    # ---- T2 / B. Tail versus parameter space ----------------------------- #
    if tld.theta is None or tld.theta.shape[1] != len(keys):
        reporter.stat("tail_vs_theta", "skipped",
                      note="The artifact carries no per-example theta, so the "
                           "tail-vs-parameter analysis cannot run.")
    else:
        theta = np.asarray(tld.theta, dtype=np.float64)[finite]
        rows, n_hard, n_bulk, cut = tail_versus_theta(
            loss, theta, keys, widths, args.hard_quantile)
        reporter.stat("hard_tail_definition",
                      f"NLL >= {cut:.4f}  (top {(1 - args.hard_quantile) * 100:.0f}%)",
                      note=f"{n_hard} hard vs {n_bulk} bulk examples.")
        reporter.table(
            "Per-parameter tail analysis (ranked by KS D)",
            ["parameter", "prior range (log10)", "Spearman rho", "rho p (BH)",
             "mean shift (hard-bulk)", "KS D", "KS p (BH)"],
            [[r["key"],
              f"{r['width']:.3g}",
              f"{r['rho']:+.3f}",
              f"{r['rho_p_bh']:.3g}",
              f"{r['mean_shift']:+.3f}",
              f"{r['ks_stat']:.3f}",
              f"{r['ks_p_bh']:.3g}"] for r in rows],
            note="Spearman rho = monotone NLL-vs-value trend over all examples; KS D and "
                 "the mean shift compare the hardest examples against the rest. Larger "
                 "|rho| or KS D = that parameter more strongly marks the hard examples, "
                 "and the sign shows which end of its range is harder (negative = harder "
                 "at low values). p-values are Benjamini-Hochberg-adjusted; with this "
                 "many examples nearly all are significant, so judge difficulty by the "
                 "effect sizes (rho, KS D), not the p-values.")
        reporter.save_figure(
            "loss_vs_theta", figure_loss_vs_theta(loss, theta, keys, nll_prior),
            caption="P3. Joint density of NLL against each learnable parameter. Hexbin "
                    "brightness is the number of test examples per cell on a shared log "
                    "scale (bright = many, dark = few), so the bright band is where most "
                    "examples lie; the orange line is the median NLL across the "
                    "parameter's range; the red dashed line is the uniform-prior "
                    "baseline. GOOD: the orange line stays low and flat across the whole "
                    "range (uniform skill). BAD: the orange line rises over part of the "
                    "range -- the estimator is worse for examples with those parameter "
                    "values (for example, a rise toward low photon count = harder in "
                    "dim, low-SNR videos).")
        reporter.save_figure(
            "tail_drivers", figure_tail_drivers(rows),
            caption="P4. Which parameters characterize the hardest examples (the top "
                    f"{(1 - args.hard_quantile) * 100:.0f}%). For each parameter, two "
                    "effect sizes: KS D (red, 0-1) = how differently that parameter is "
                    "distributed among the hard examples versus the rest (0 = identical, "
                    "1 = fully separated); |Spearman rho| (blue, 0-1) = how strongly NLL "
                    "trends up or down with the parameter overall. Longer top bars = the "
                    "parameters whose values most set the hard cases apart -- the regimes "
                    "the estimator finds hardest; short bars = little relation to "
                    "difficulty. A pointer for where to look, to be confirmed by the "
                    "Evaluation stage's per-parameter recovery error.")

    # ---- Figures P1 / P2 ------------------------------------------------- #
    reporter.save_figure(
        "nll_histogram", figure_histogram(loss, mean, median, nll_prior),
        caption="P1. Histogram of the per-example test NLL -- for each test example, how "
                "well the trained posterior fits its true parameters (further left = "
                "better). GOOD: most mass far to the left with a single peak and only a "
                "thin tail reaching the red prior line. BAD: mass piled near or past the "
                "red line, or a heavy right tail, means many examples are fit poorly. "
                "The red line is what an uninformed (prior) model scores; the mean should "
                "sit well to its left.")
    reporter.save_figure(
        "nll_ecdf", figure_ecdf(loss, nll_prior),
        caption="P2. Empirical CDF of the per-example NLL -- at any NLL on the x-axis, "
                "the curve height is the fraction of examples at least that good. GOOD: a "
                "steep rise far to the left of the red prior line, reaching ~1 before it, "
                "so the worse-than-prior fraction (mass right of the line) is near zero. "
                "BAD: a shallow curve, or a visible fraction past the red line.")

    reporter.summary()
    report_path = reporter.write_report()
    print(f"\nReport written: {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main(parse_args()))

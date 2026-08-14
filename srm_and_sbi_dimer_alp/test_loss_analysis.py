"""Analysis kernel for a saved per-example test-loss distribution (both DIMER workflows).

The Inference stage records, at its best epoch, one per-example negative log-likelihood
(NLL) over the fixed held-out TEST set -- the ``TestLossDistribution`` artifact (parallel
``keys`` / ``theta`` / ``loss`` arrays + a self-describing manifest). The training log
reports only the *mean* of that array; this module produces the picture the mean cannot
give:

  - **Distribution shape** -- the spread and skew of the per-example NLL (via
    ``TestLossDistribution.extended_card``), and the **uniform-prior NLL** reference
    (``prior_reference_nll``): the no-information baseline ``sum_j ln(range_j)`` an estimator
    that learned nothing would score, against which the information gain (nats) and the
    worse-than-prior fraction are read.
  - **Tail versus parameter space** (``tail_versus_theta``) -- for each learnable parameter,
    the Spearman association between the NLL and the parameter value plus a hard-tail-vs-bulk
    comparison (mean shift + two-sample KS), Benjamini-Hochberg-adjusted across the family.
    This is the **identifiability read**: it locates which parameters, and which end of their
    range, mark the hardest examples -- e.g. for biology the species counts, whose low end
    yields uninformative videos. It says *where* the estimator is challenged; whether that
    hardness is an honest identifiability limit (a wide but calibrated posterior) or
    overconfidence is answered by the ``Posterior_Calibration`` diagnostic, stratified by the
    inferred value.

SBC / TARP calibration is deliberately out of scope here: they need posterior *samples* per
example, which this NLL artifact does not carry.

Everything is read from the artifact's own manifest, so the analysis stays correct if the
live parameterization changes and generalizes to any number of learnable parameters, for
either workflow, with no edits. Pure analysis: numpy, with scipy + matplotlib imported
lazily; imports nothing from ``parameterization`` / ``artifacts``, so it is unit-testable.
"""
from __future__ import annotations

import numpy as np

# Quantiles reported in the spread table (percent).
SPREAD_QUANTILES = (1, 5, 25, 50, 75, 90, 95, 99)


# =============================================================================
# Manifest interpretation (all read from the artifact, nothing hardcoded)
# =============================================================================

def learnable_entries(manifest):
    """The learnable-parameter rows of the manifest's parameter table, in the declaration
    order that matches the ``theta`` columns / ``theta_keys``."""
    table = manifest.get("parameter_table", [])
    return [entry for entry in table if entry.get("role") == "learnable"]


def prior_reference_nll(entries):
    """Uniform-prior NLL = sum of ``ln(prior-range width)`` over learnable params.

    The no-information baseline: for a uniform prior over the learnable box the expected NLL
    is the constant log prior-box volume. Returns ``(nll_prior, widths)`` where
    ``widths[j] = high_j - low_j`` in the parameter's own units (log10 here), matching the
    space the density is scored in.
    """
    widths = []
    for entry in entries:
        low, high = entry["prior_range"][0], entry["prior_range"][1]
        width = float(high) - float(low)
        if width <= 0.0:
            raise ValueError(
                f"parameter {entry.get('key')!r} has a non-positive prior range width "
                f"({width}); cannot form the prior reference.")
        widths.append(width)
    nll_prior = float(np.sum(np.log(widths)))
    return nll_prior, widths


# =============================================================================
# Statistics
# =============================================================================

def benjamini_hochberg(pvalues):
    """Benjamini-Hochberg step-up adjusted p-values (false-discovery-rate control).

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

    ``loss`` and ``theta`` are already restricted to finite-loss rows. The hard tail is the
    ``loss >= quantile(loss, hard_quantile)`` subset; the bulk is the remainder. Returns a
    list of per-parameter dicts plus the split sizes and the cut. All raw p-values (Spearman
    over all rows, KS hard-vs-bulk) are pooled and Benjamini-Hochberg-adjusted together; the
    rows are returned ranked by the hard-vs-bulk KS separation (the tail-specific signal).
    """
    from scipy.stats import ks_2samp, spearmanr  # lazy

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
    it = iter(adjusted)
    finite_adj = {i: next(it) for i, p in enumerate(raw_p) if np.isfinite(p)}
    for k, row in enumerate(rows):
        row["rho_p_bh"] = float(finite_adj.get(2 * k, float("nan")))
        row["ks_p_bh"] = float(finite_adj.get(2 * k + 1, float("nan")))

    rows.sort(key=lambda r: (-1.0 if np.isnan(r["ks_stat"]) else -r["ks_stat"]))
    return rows, n_hard, n_bulk, cut


# =============================================================================
# Figures (headless Figure objects; matplotlib imported lazily)
# =============================================================================

def figure_histogram(loss, mean, median, nll_prior):
    from matplotlib.figure import Figure

    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.hist(loss, bins=60, color="#4C72B0", alpha=0.85, edgecolor="white", linewidth=0.2)
    ax.axvline(mean, color="#1f1f1f", linestyle="-", linewidth=1.4, label=f"mean = {mean:.2f}")
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
    from matplotlib.figure import Figure

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
    """Per-parameter joint density of NLL vs parameter value, with the median trend."""
    from matplotlib.colors import LogNorm
    from matplotlib.figure import Figure

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
    vmax = max(2.0, max(float(hb.get_array().max()) for hb in hexbins))
    norm = LogNorm(vmin=1, vmax=vmax)
    for hb in hexbins:
        hb.set_norm(norm)
    for j in range(n, nrow * ncol):
        fig.add_subplot(nrow, ncol, j + 1).axis("off")
    cbar = fig.colorbar(hexbins[-1], ax=axes, fraction=0.03, pad=0.01)
    cbar.set_label("examples per cell (log scale)", fontsize=8)
    if axes:
        axes[min(len(axes) - 1, ncol - 1)].legend(fontsize=6, frameon=False, loc="upper right")
    fig.suptitle("Per-example NLL vs each learnable parameter", fontsize=11)
    return fig


def figure_tail_drivers(rows):
    from matplotlib.figure import Figure

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
    ax.invert_yaxis()
    ax.set_xlabel("statistic")
    ax.set_title("Per-parameter tail drivers (ranked by KS D)")
    ax.legend(frameon=False, loc="lower right")
    fig.tight_layout()
    return fig

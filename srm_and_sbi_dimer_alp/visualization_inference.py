"""Inference-stage diagnostics and interactive plotting.

Functions:
    plot_loss_curves(losses_train, losses_test, losses_replay)
        Plot training, validation, and replay loss curves over epochs.

Matplotlib is imported lazily inside the function so HPC headless runs
that never plot don't pay the import cost.

Posterior-level diagnostics (corner plots, posterior-predictive video
overlays, etc.) can be added here as further analyses are developed.
"""

import numpy as np


def plot_loss_curves(losses_train: np.ndarray,
                     losses_test: np.ndarray,
                     losses_replay: np.ndarray) -> None:
    """Plot training, validation, and replay loss curves on a single axis.

    "Replay" loss is the training loss recomputed with data augmentation
    disabled — a proxy for the loss on the augmentation-free distribution
    that the validation loss measures, but computed on the training set.
    Useful for diagnosing whether the train-vs-validation gap is
    augmentation-driven or genuine overfitting.

    Args:
        losses_train: 1D array of per-epoch training loss.
        losses_test: 1D array of per-epoch validation loss (same length).
        losses_replay: 1D array of per-epoch replay loss (same length).
    """
    import matplotlib.pyplot as plt

    epochs = np.arange(len(losses_train))
    plt.figure()
    plt.plot(epochs, losses_train, label="Train loss")
    plt.plot(epochs, losses_test, label="Test loss")
    plt.plot(epochs, losses_replay, label="[Replay] Train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training, test, and replay losses over epochs")
    plt.legend()
    plt.show()


# =============================================================================
# Headless figure builder (for --debug-dump reports)
# =============================================================================


def figure_loss_curves(losses_train: np.ndarray,
                       losses_test: np.ndarray,
                       losses_replay: np.ndarray):
    """Headless counterpart to ``plot_loss_curves``: build and return a Figure.

    Builds the train / test / replay loss-curve plot via
    ``matplotlib.figure.Figure`` (no display needed) and returns it for the
    DiagnosticReporter to save as PNG.

    Args:
        losses_train: 1D array of per-epoch training loss.
        losses_test: 1D array of per-epoch validation loss (same length).
        losses_replay: 1D array of per-epoch replay loss (same length).

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    from matplotlib.figure import Figure

    epochs = np.arange(len(losses_train))
    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.plot(epochs, losses_train, label="Train loss")
    ax.plot(epochs, losses_test, label="Test loss")
    ax.plot(epochs, losses_replay, label="[Replay] Train loss")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training, test, and replay losses over epochs")
    ax.legend()
    return fig


# =============================================================================
# MAP-recovery figure builders (Evaluation stage)
# =============================================================================


def _conditional_quantiles(x: np.ndarray, value: np.ndarray, n_bins: int,
                           min_count: int, bin_mode: str = "quantile",
                           prior_range=None):
    """Conditional quantiles of ``value`` over bins of ``x``.

    Splits ``x`` into ``n_bins`` bins and, for each bin holding at least
    ``min_count`` points, computes the Q05/Q25/Q50/Q75/Q95 of ``value``. Bins
    below the threshold are left as NaN, so a band is drawn only where the sample
    is dense enough (graceful degradation for small EVAL).

    ``bin_mode``:
        ``"quantile"`` -- bin edges at data quantiles of ``x`` (equal-count bins;
                          the default).
        ``"prior"``    -- bin edges uniform across ``prior_range`` (equal-width
                          bins over the prior bounds); falls back to quantile bins
                          if ``prior_range`` is None.

    Returns:
        ``(centers, q05, q25, q50, q75, q95)`` — each a length-``n_bins`` array;
        sparse bins are NaN.
    """
    if bin_mode == "prior" and prior_range is not None:
        edges = np.linspace(prior_range[0], prior_range[1], n_bins + 1)
    else:
        edges = np.quantile(x, np.linspace(0, 1, n_bins + 1))
    centers = 0.5 * (edges[:-1] + edges[1:])
    quantiles = {q: np.full(n_bins, np.nan) for q in (5, 25, 50, 75, 95)}
    bin_id = np.clip(np.digitize(x, edges, right=True) - 1, 0, n_bins - 1)
    for k in range(n_bins):
        vk = value[bin_id == k]
        if vk.size >= min_count:
            q05, q25, q50, q75, q95 = np.quantile(vk, [0.05, 0.25, 0.50, 0.75, 0.95])
            quantiles[5][k], quantiles[25][k], quantiles[50][k] = q05, q25, q50
            quantiles[75][k], quantiles[95][k] = q75, q95
    return (centers, quantiles[5], quantiles[25], quantiles[50],
            quantiles[75], quantiles[95])


def _overlay_quantile_bands(ax, centers, q05, q25, q50, q75, q95):
    """Overlay the Q05-Q95 / Q25-Q75 bands and the Q50 median, where present."""
    ok = np.isfinite(q50)
    if not np.any(ok):
        return False
    ax.fill_between(centers[ok], q05[ok], q95[ok], alpha=0.50, linewidth=0,
                    color="tab:red", label="Q05-Q95")
    ax.fill_between(centers[ok], q25[ok], q75[ok], alpha=0.75, linewidth=0,
                    color="tab:green", label="Q25-Q75")
    ax.plot(centers[ok], q50[ok], alpha=0.95, linewidth=1,
            color="tab:orange", label="Q50")
    return True


def _draw_placeholder(ax, text):
    """Stamp a panel that is intentionally not presented with a centered message."""
    ax.text(0.5, 0.5, text, transform=ax.transAxes, ha="center", va="center",
            fontsize=10, color="0.4", wrap=True)
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_recovery_axis(ax, true_log10, inferred_log10, prior_range=None,
                        n_bins=20, min_count=50, bin_mode="quantile"):
    """Draw the recovery scatter (true vs. inferred, log10) + bands on ``ax``.

    Points on the dashed identity line are perfectly recovered; conditional
    quantile bands of inferred-given-true overlay where a bin is dense enough.
    ``bin_mode`` selects data-quantile vs prior-range bin edges.
    """
    x = np.asarray(true_log10, dtype=float).ravel()
    y = np.asarray(inferred_log10, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    limits = (tuple(prior_range) if prior_range is not None
              else (float(np.floor(np.min(x))), float(np.ceil(np.max(x)))))
    ax.axline((limits[0], limits[0]), (limits[1], limits[1]),
              color="k", linestyle="--", alpha=0.75)
    ax.scatter(x, y, s=7, color="tab:blue", alpha=0.25)
    centers, q05, q25, q50, q75, q95 = _conditional_quantiles(
        x, y, n_bins, min_count, bin_mode, prior_range)
    bands = _overlay_quantile_bands(ax, centers, q05, q25, q50, q75, q95)
    ax.set_xlabel(r"true [$\log_{10}$]")
    ax.set_ylabel(r"inferred [$\log_{10}$]")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    if bands:
        ax.legend(fontsize=9, frameon=False)
    else:
        ax.text(0.5, 0.02, f"bands sparse (n<{min_count}/bin) — scatter only",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
                color="tab:red", alpha=0.8)
    ax.set_title("View A: recovery (MAP)", fontsize=11)


def _draw_error_axis(ax, true_log10, inferred_log10, prior_range=None,
                     n_bins=20, min_count=50, error_guide=0.3,
                     error_guide_tight=0.15, error_ylim_floor=0.5,
                     error_ylim_quantile=0.95, bin_mode="quantile"):
    """Draw the residual-error view (inferred - true, log10) + bands on ``ax``.

    A zero line marks perfect recovery. Two nested tolerance bands are drawn as
    +/- guide lines, each the log10 of a linear accuracy factor:
    ``+/- error_guide`` (0.3 ~= log10(2): within a factor of 2) and the tighter
    ``+/- error_guide_tight`` (0.15 ~= log10(sqrt(2)): within a factor of ~1.41).
    The y-axis spans ``+/- max(error_ylim_floor, quantile(|error|,
    error_ylim_quantile))``.
    """
    x = np.asarray(true_log10, dtype=float).ravel()
    y = np.asarray(inferred_log10, dtype=float).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    error = y - x
    x_limits = (tuple(prior_range) if prior_range is not None
                else (float(np.floor(np.min(x))), float(np.ceil(np.max(x)))))
    half = max(error_ylim_floor, float(np.quantile(np.abs(error), error_ylim_quantile)))
    ax.axhline(0.0, color="k", linestyle="--", alpha=0.75)
    # Factor-2 band (0.3 = log10 2) and the tighter factor-sqrt(2) band (0.15 = log10 sqrt 2).
    ax.axhline(+error_guide, color="k", linestyle=":", alpha=0.60,
               label=rf"$\pm${error_guide:g} (factor 2)")
    ax.axhline(-error_guide, color="k", linestyle=":", alpha=0.60)
    ax.axhline(+error_guide_tight, color="k", linestyle=(0, (1, 3)), alpha=0.45,
               label=rf"$\pm${error_guide_tight:g} (factor $\sqrt{{2}}$)")
    ax.axhline(-error_guide_tight, color="k", linestyle=(0, (1, 3)), alpha=0.45)
    ax.scatter(x, error, s=7, color="tab:blue", alpha=0.25)
    centers, q05, q25, q50, q75, q95 = _conditional_quantiles(
        x, error, n_bins, min_count, bin_mode, prior_range)
    bands = _overlay_quantile_bands(ax, centers, q05, q25, q50, q75, q95)
    ax.set_xlabel(r"true [$\log_{10}$]")
    ax.set_ylabel(r"error = inferred - true [$\log_{10}$]")
    ax.set_xlim(x_limits)
    ax.set_ylim((-half, half))
    if bands:
        ax.legend(fontsize=9, frameon=False)
    else:
        ax.text(0.5, 0.02, f"bands sparse (n<{min_count}/bin) — scatter only",
                transform=ax.transAxes, ha="center", va="bottom", fontsize=8,
                color="tab:red", alpha=0.8)
    ax.set_title("View A: error (MAP)", fontsize=11)


def _draw_posterior_recovery_axis(ax, true_log10, map_inferred, post_median,
                                  post_q25, post_q75, prior_range=None):
    """Draw View B (true vs posterior median +/- IQR, with the MAP overlaid) on ``ax``.

    Both estimators are shown so the posterior median is not mistaken for the
    point estimate: blue circles + bars are the posterior median and IQR
    (credible width); orange crosses are the MAP mode.
    """
    x = np.asarray(true_log10, dtype=float).ravel()
    mp = np.asarray(map_inferred, dtype=float).ravel()
    med = np.asarray(post_median, dtype=float).ravel()
    q25 = np.asarray(post_q25, dtype=float).ravel()
    q75 = np.asarray(post_q75, dtype=float).ravel()
    mask = (np.isfinite(x) & np.isfinite(med) & np.isfinite(q25)
            & np.isfinite(q75) & np.isfinite(mp))
    x, mp, med, q25, q75 = x[mask], mp[mask], med[mask], q25[mask], q75[mask]
    limits = (tuple(prior_range) if prior_range is not None
              else (float(np.floor(np.min(x))), float(np.ceil(np.max(x)))))
    yerr = np.vstack([np.clip(med - q25, 0, None), np.clip(q75 - med, 0, None)])
    ax.axline((limits[0], limits[0]), (limits[1], limits[1]),
              color="k", linestyle="--", alpha=0.75)
    ax.errorbar(x, med, yerr=yerr, fmt="o", ms=4, color="tab:blue",
                ecolor="tab:blue", elinewidth=1, alpha=0.5, capsize=2,
                label="posterior median $\\pm$ IQR")
    ax.scatter(x, mp, marker="x", s=28, color="tab:orange", alpha=0.8,
               zorder=3, label="MAP")
    ax.set_xlabel(r"true [$\log_{10}$]")
    ax.set_ylabel(r"inferred [$\log_{10}$]")
    ax.set_xlim(limits)
    ax.set_ylim(limits)
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("View B: posterior + MAP", fontsize=11)


def figure_recovery_combined(true_log10, inferred_log10, post_q,
                             prior_range=None, label="", n_bins=20, min_count=50,
                             error_guide=0.3, error_guide_tight=0.15,
                             error_ylim_floor=0.5,
                             error_ylim_quantile=0.95, bin_mode="quantile",
                             show_map=True, show_posterior=False):
    """Combined recovery figure for one parameter: View A (MAP) + View B (posterior).

    A 1x3 row: [View A recovery scatter] [View A error] [View B posterior median
    +/- IQR]. Views that are not requested are kept as panels and stamped so their
    absence is explicit. ``post_q`` is a per-observation ``(N, 5)`` quantile array
    ``[Q05,Q25,Q50,Q75,Q95]`` (or None when View B is off).

    Returns:
        A headless ``matplotlib.figure.Figure``.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(16, 5.2))
    ax_rec, ax_err, ax_post = (fig.add_subplot(1, 3, k) for k in (1, 2, 3))
    if show_map:
        _draw_recovery_axis(ax_rec, true_log10, inferred_log10, prior_range,
                            n_bins, min_count, bin_mode)
        _draw_error_axis(ax_err, true_log10, inferred_log10, prior_range, n_bins,
                         min_count, error_guide=error_guide,
                         error_guide_tight=error_guide_tight,
                         error_ylim_floor=error_ylim_floor,
                         error_ylim_quantile=error_ylim_quantile, bin_mode=bin_mode)
    else:
        _draw_placeholder(ax_rec, "View A not computed\n(--summary posterior)")
        _draw_placeholder(ax_err, "View A not computed\n(--summary posterior)")
    if show_posterior and post_q is not None:
        _draw_posterior_recovery_axis(ax_post, true_log10, inferred_log10,
                                      post_q[:, 2], post_q[:, 1], post_q[:, 3],
                                      prior_range)
    else:
        _draw_placeholder(ax_post, "View B not computed\n(--summary map)")
    if label:
        fig.suptitle(label, fontsize=13)
    return fig


def _draw_experiment_distribution_axis(ax, values_by_kind, prior_range=None, seed=0):
    """Draw View A (per-condition MAP-point distribution) for one parameter on ``ax``.

    No ground truth for real data, so this shows the *distribution* of the inferred
    MAP value across all (cell, chunk) estimates -- one box per condition (kind),
    with jittered points overlaid; dashed lines mark the prior bounds. Comparing
    conditions (e.g. ALP vs BET) is the scientific read-out.
    """
    kinds = list(values_by_kind.keys())
    data = [np.asarray(values_by_kind[k], dtype=float).ravel() for k in kinds]
    data = [d[np.isfinite(d)] for d in data]

    if prior_range is not None:
        ax.axhline(prior_range[0], color="k", linestyle=":", alpha=0.5)
        ax.axhline(prior_range[1], color="k", linestyle=":", alpha=0.5)
        ax.set_ylim(prior_range[0] - 0.2, prior_range[1] + 0.2)

    positions = list(range(1, len(kinds) + 1))
    # Boxplot needs non-empty sequences; substitute a NaN so empty kinds keep a slot.
    ax.boxplot([d if d.size else np.array([np.nan]) for d in data],
               positions=positions, showfliers=False, widths=0.5)
    rng = np.random.RandomState(seed)
    for pos, d in zip(positions, data):
        if d.size:
            jitter = (rng.rand(d.size) - 0.5) * 0.18
            ax.scatter(np.full(d.size, pos) + jitter, d, s=6, alpha=0.3,
                       color="tab:blue")
    ax.set_xticks(positions)
    ax.set_xticklabels(kinds)
    ax.set_xlabel("experimental condition")
    ax.set_ylabel(r"inferred [$\log_{10}$]")
    ax.set_title("View A: MAP", fontsize=11)


def _draw_experiment_posterior_axis(ax, by_kind, prior_range=None, seed=0):
    """Draw View B (per-condition posterior median +/- IQR, with MAP) for one parameter.

    Each chunk is drawn as its **posterior median** with **IQR (Q25-Q75)** error
    bars (blue) and its **MAP** (orange cross), jittered within its condition --
    showing within-chunk posterior uncertainty, the point estimate, and cross-chunk
    spread, one group per condition. ``by_kind`` maps each kind to an ``(n, 4)``
    array ``[MAP, median, q25, q75]`` (log10). Dashed lines mark the prior bounds.
    """
    kinds = list(by_kind.keys())
    if prior_range is not None:
        ax.axhline(prior_range[0], color="k", linestyle=":", alpha=0.5)
        ax.axhline(prior_range[1], color="k", linestyle=":", alpha=0.5)
        ax.set_ylim(prior_range[0] - 0.2, prior_range[1] + 0.2)
    rng = np.random.RandomState(seed)
    for pos, kind in enumerate(kinds, start=1):
        arr = np.asarray(by_kind[kind], dtype=float)
        if arr.size == 0:
            continue
        mp, med, q25, q75 = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3]
        m = np.isfinite(med) & np.isfinite(q25) & np.isfinite(q75) & np.isfinite(mp)
        mp, med, q25, q75 = mp[m], med[m], q25[m], q75[m]
        if med.size == 0:
            continue
        x = np.full(med.size, pos) + (rng.rand(med.size) - 0.5) * 0.3
        yerr = np.vstack([np.clip(med - q25, 0, None), np.clip(q75 - med, 0, None)])
        labels = ({"med": "posterior median $\\pm$ IQR", "map": "MAP"}
                  if pos == 1 else {"med": None, "map": None})
        ax.errorbar(x, med, yerr=yerr, fmt="o", ms=4, color="tab:blue",
                    ecolor="tab:gray", elinewidth=1, alpha=0.5, capsize=2,
                    linestyle="none", label=labels["med"])
        ax.scatter(x, mp, marker="x", s=24, color="tab:orange", alpha=0.8,
                   zorder=3, label=labels["map"])
    ax.set_xticks(range(1, len(kinds) + 1))
    ax.set_xticklabels(kinds)
    ax.set_xlabel("experimental condition")
    ax.set_ylabel(r"inferred [$\log_{10}$]")
    ax.legend(fontsize=8, frameon=False)
    ax.set_title("View B: posterior + MAP", fontsize=11)


def figure_experiment_combined(values_by_kind, by_kind_post, prior_range=None,
                               label="", seed=0, show_map=True, show_posterior=False):
    """Combined experiment figure for one parameter: View A (MAP) + View B (posterior).

    A 1x2 row: [View A: per-condition MAP-point box] [View B: per-condition
    posterior median +/- IQR + MAP]. A view that is not requested is kept as a panel
    and stamped so its absence is explicit. ``values_by_kind`` is ``{kind: 1D MAP
    log10}``; ``by_kind_post`` is ``{kind: (n,4) [MAP,median,q25,q75]}`` (or None).

    Returns:
        A headless ``matplotlib.figure.Figure``.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(12, 5.5))
    ax_map, ax_post = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    if show_map and values_by_kind is not None:
        _draw_experiment_distribution_axis(ax_map, values_by_kind, prior_range, seed)
    else:
        _draw_placeholder(ax_map, "View A not computed\n(--summary posterior)")
    if show_posterior and by_kind_post is not None:
        _draw_experiment_posterior_axis(ax_post, by_kind_post, prior_range, seed)
    else:
        _draw_placeholder(ax_post, "View B not computed\n(--summary map)")
    if label:
        fig.suptitle(label, fontsize=13)
    return fig

"""Temporal-dynamics kernel: how inferred parameters behave across one recording.

Workflow-agnostic numerics shared by the biology and detector temporal analyses. Nothing here
knows which parameters it is describing: every function takes arrays, a prior box, and a time
axis. Pure numpy (plus the shared geometric-median kernel and a lazy scipy import for the sign
test), so it imports and unit-tests without a machine profile.

WHAT THE ANALYSIS ASKS. The Experiment stage estimates parameters independently in every
non-overlapping window of every recording. Stacking those windows along time turns a stage that
reports one number per recording into a time series, which answers a question the stage cannot:
does the inferred value hold still across the recording? A parameter that is a constant property
of the system should be flat. A trend is either real dynamics or an acquisition confound, and the
two are not distinguishable from one workflow's estimates alone.

THE CENTRAL ESTIMATE IS A REAL RECORDING, NOT AN AVERAGE. Summarizing many cells at one time
point by averaging each parameter independently composes a vector whose coordinates never
co-occurred in any cell -- the same defect the Sample Geometric Median exists to remove
(Ramirez Sierra & Sokolowski, Mach. Learn.: Sci. Technol. 6, 015004, 2025). Two SGM-based central
estimates are provided instead, and they answer different questions:

  - :func:`central_trajectory_sgm` (**trajectory-level**) selects the ONE cell whose ENTIRE time
    course is most central, by taking the medoid over cells of the flattened
    (time x parameter) trajectory. The result is a real recording's real time course: every
    plotted point comes from the same acquisition, so an apparent temporal change cannot be an
    artifact of switching between cells. This is the headline central estimate.
  - :func:`central_per_time_sgm` (**per-time-point**) selects, at each time independently, the
    medoid cell across the parameter vector at that time. Its coordinates are jointly realized
    within each time point, but THE SELECTED CELL CAN CHANGE BETWEEN TIME POINTS, so a step in
    the trajectory may be a change of cell rather than a change in time. The selected cell index
    is returned at every time point precisely so that this confound is visible rather than
    hidden, and it must be reported alongside the curve.

Both operate on the FULL parameter vector, never a plotted subset: the medoid is a property of
the joint vector, so restricting the figures to some parameters must not change which cell is
central. Distances are taken in ABSOLUTE (physical) space normalized by the absolute prior range,
matching the convention of the geometric-median analysis -- the space where the simulator
consumes the values, and where dividing by each prior's range makes the dimensions commensurable.
"""
from __future__ import annotations

import numpy as np

from .sample_geometric_median import sample_geometric_median

# A drift of this many dex over the recording is called material. It is the same practical bar the
# recovery tables use for "within a factor of two" (0.3 dex ~ 2x), so a drift that exceeds it moves
# the estimate by more than the tolerance the recovery is judged against.
MATERIAL_DRIFT_DEX = 0.3


def reshape_to_grid(values, kind_index, cell, chunk, n_kinds):
    """Scatter flat per-window rows into a dense ``(kind, cell, chunk, ...)`` grid.

    The Experiment output stores one flat row per analyzed window. Each row is placed at
    ``grid[kind_index, cell, chunk]``. Windows never estimated stay NaN and every downstream
    statistic is nan-aware, so a missing window narrows nothing silently.

    Args:
        values: ``(N, ...)`` per-window values (e.g. ``(N, D)`` point estimates or
            ``(N, D, Q)`` stored quantiles).
        kind_index, cell, chunk: ``(N,)`` integer labels identifying each row's condition,
            recording, and window.
        n_kinds: number of conditions.

    Returns:
        ``(grid, n_cells, n_chunks)``.
    """
    values = np.asarray(values, dtype=float)
    kind_index = np.asarray(kind_index, dtype=int)
    cell = np.asarray(cell, dtype=int)
    chunk = np.asarray(chunk, dtype=int)
    n_cells = int(cell.max()) + 1
    n_chunks = int(chunk.max()) + 1
    grid = np.full((n_kinds, n_cells, n_chunks) + values.shape[1:], np.nan, dtype=float)
    grid[kind_index, cell, chunk] = values
    return grid, n_cells, n_chunks


def _range_abs(prior_low, prior_high):
    """Absolute prior width per parameter, the normalizer for every distance here."""
    lo = np.asarray(prior_low, dtype=float)
    hi = np.asarray(prior_high, dtype=float)
    span = 10.0 ** hi - 10.0 ** lo
    span[span <= 0] = 1.0
    return span


def central_trajectory_sgm(grid_log10, prior_low, prior_high):
    """Trajectory-level SGM: the one cell per condition whose whole time course is most central.

    Each cell contributes its complete ``(n_chunks, D)`` trajectory, flattened to a single
    ``n_chunks * D`` vector after each parameter is divided by its absolute prior range so no
    parameter dominates the distance. The medoid of those vectors is a real cell, and the returned
    trajectory is that cell's stored rows verbatim.

    Cells with any missing window are excluded from the selection, because a partial trajectory
    cannot be compared on equal footing with complete ones; if every cell of a condition is
    incomplete, that condition's entry is NaN and its index is -1.

    Args:
        grid_log10: ``(n_kinds, n_cells, n_chunks, D)`` point estimates in log10 space.
        prior_low, prior_high: ``(D,)`` log10 prior bounds.

    Returns:
        ``(trajectory, cells, methods)`` -- ``trajectory`` ``(n_kinds, n_chunks, D)`` in log10,
        ``cells`` ``(n_kinds,)`` the selected cell index per condition (-1 if none), ``methods``
        the medoid method used per condition.
    """
    grid = np.asarray(grid_log10, dtype=float)
    n_kinds, n_cells, n_chunks, dim = grid.shape
    span = _range_abs(prior_low, prior_high)
    out = np.full((n_kinds, n_chunks, dim), np.nan)
    picked = np.full(n_kinds, -1, dtype=int)
    methods = []
    for k in range(n_kinds):
        complete = [c for c in range(n_cells) if np.isfinite(grid[k, c]).all()]
        if not complete:
            methods.append("none")
            continue
        # Absolute space, per-parameter prior-range normalization, then flatten over time.
        flat = np.stack([(10.0 ** grid[k, c] / span).ravel() for c in complete], axis=0)
        idx, method = sample_geometric_median(flat, np.ones(flat.shape[1]))
        picked[k] = complete[idx]
        methods.append(method)
        out[k] = grid[k, complete[idx]]
    return out, picked, methods


def central_per_time_sgm(grid_log10, prior_low, prior_high):
    """Per-time-point SGM: at each time, the medoid cell across the parameter vector.

    Coordinates within one time point are jointly realized, but the selected cell may differ
    between time points, so the returned ``cells`` array is part of the result rather than a
    diagnostic: a change of central cell between adjacent times can masquerade as temporal change
    and must be shown wherever the curve is shown.

    Returns:
        ``(trajectory, cells)`` -- ``(n_kinds, n_chunks, D)`` in log10 and ``(n_kinds, n_chunks)``
        selected cell indices (-1 where a time point has no complete cell).
    """
    grid = np.asarray(grid_log10, dtype=float)
    n_kinds, n_cells, n_chunks, dim = grid.shape
    span = _range_abs(prior_low, prior_high)
    out = np.full((n_kinds, n_chunks, dim), np.nan)
    picked = np.full((n_kinds, n_chunks), -1, dtype=int)
    for k in range(n_kinds):
        for t in range(n_chunks):
            rows = [c for c in range(n_cells) if np.isfinite(grid[k, c, t]).all()]
            if not rows:
                continue
            block = np.stack([10.0 ** grid[k, c, t] for c in rows], axis=0)
            idx, _method = sample_geometric_median(block, span)
            picked[k, t] = rows[idx]
            out[k, t] = grid[k, rows[idx], t]
    return out, picked


def drift_statistics(grid_log10, times, threshold=MATERIAL_DRIFT_DEX):
    """Per-cell linear drift of each parameter across the recording, aggregated per condition.

    For every (condition, cell, parameter) an ordinary least-squares line is fit to the stored
    log10 estimate against time and reported as the total change over the observed span
    (``slope * (t_last - t_first)``), i.e. in dex across the recording. Working in log10 makes the
    drift multiplicative and comparable across parameters of different units, and a per-cell fit
    followed by a median over cells is robust: one erratic recording cannot move the summary.

    These statistics are independent of any central-estimate choice -- they are computed per cell,
    so swapping an average for a geometric median changes what is DISPLAYED, not what is measured.

    Returns a dict of ``(n_kinds, D)`` arrays: ``median_dex`` (median per-cell drift),
    ``frac_material`` (fraction of cells whose |drift| exceeds ``threshold``), ``sign_consistency``
    (fraction of cells sharing the median's sign), ``wilcoxon_p`` (two-sided signed-rank test that
    the per-cell drifts are centered at zero; NaN when scipy is unavailable or fewer than six
    cells), and ``per_cell`` ``(n_kinds, n_cells, D)``.
    """
    grid = np.asarray(grid_log10, dtype=float)
    t = np.asarray(times, dtype=float)
    n_kinds, n_cells, n_chunks, dim = grid.shape
    span = float(t[-1] - t[0]) if n_chunks > 1 else 0.0
    per_cell = np.full((n_kinds, n_cells, dim), np.nan)
    for k in range(n_kinds):
        for c in range(n_cells):
            for p in range(dim):
                y = grid[k, c, :, p]
                ok = np.isfinite(y)
                if ok.sum() < 2:
                    continue
                slope = np.polyfit(t[ok], y[ok], 1)[0]
                per_cell[k, c, p] = slope * span
    median = np.nanmedian(per_cell, axis=1)
    with np.errstate(invalid="ignore"):
        frac = np.nanmean(np.abs(per_cell) > threshold, axis=1)
        sign = np.nanmean(np.sign(per_cell) == np.sign(median)[:, None, :], axis=1)
    pvals = np.full((n_kinds, dim), np.nan)
    try:
        from scipy.stats import wilcoxon
        for k in range(n_kinds):
            for p in range(dim):
                v = per_cell[k, :, p]
                v = v[np.isfinite(v)]
                if v.size >= 6 and np.any(v != 0):
                    pvals[k, p] = float(wilcoxon(v)[1])
    except ImportError:
        pass
    return {"median_dex": median, "frac_material": frac, "sign_consistency": sign,
            "wilcoxon_p": pvals, "per_cell": per_cell, "threshold": float(threshold)}


def quantile_summaries(quant_grid_log10):
    """Separate the two uncertainties a per-window quantile grid contains.

    A ``(n_kinds, n_cells, n_chunks, D, Q)`` grid of stored per-window posterior quantiles mixes
    two distinct quantities, and conflating them would overstate what the data says:

      - **within-window posterior spread** -- how uncertain ONE window's estimate is. Summarized
        per time point as the median across cells of each quantile level, so the reported interval
        is a typical window's interval rather than an envelope over cells.
      - **between-cell spread** -- how much the point estimates differ ACROSS recordings at the
        same time, i.e. biological and experimental heterogeneity. Summarized per time point as
        percentiles across cells of the per-window median (the ``Q50`` level).

    Returns ``(within, between)``: ``within`` ``(n_kinds, n_chunks, D, Q)`` and ``between``
    ``(n_kinds, n_chunks, D, 5)`` at the 5th, 25th, 50th, 75th, and 95th percentile across cells.
    """
    q = np.asarray(quant_grid_log10, dtype=float)
    n_kinds, n_cells, n_chunks, dim, nq = q.shape
    within = np.nanmedian(q, axis=1)                              # median over cells, per level
    mid = nq // 2                                                 # the stored median level
    between = np.nanpercentile(q[:, :, :, :, mid], [5, 25, 50, 75, 95], axis=1)
    between = np.moveaxis(between, 0, -1)                         # -> (kind, chunk, D, 5)
    return within, between

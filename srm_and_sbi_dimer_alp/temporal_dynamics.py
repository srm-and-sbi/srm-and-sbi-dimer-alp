"""Temporal-dynamics kernel: how inferred parameters behave across one recording.

Workflow-agnostic numerics shared by the biology and detector temporal analyses. Nothing here knows
which parameters it is describing: every function takes arrays, a prior box, and a time axis. Pure
numpy (plus the shared geometric-median kernel and a lazy scipy import for the sign test), so it
imports and unit-tests without a machine profile.

WHAT THE ANALYSIS ASKS. The Experiment stage estimates the parameters independently in every
non-overlapping window of every recording, reporting one MAP estimate per window. Stacking those
windows along time asks a question the stage cannot: does the inferred value hold still across the
recording? A parameter that is a constant property of the system should be flat. A trend is either
real dynamics or an acquisition confound, and one workflow's estimates cannot distinguish them.

THE ONE ARRAY EVERYTHING STARTS FROM. All functions here operate on the MAP grid

    G[k, c, t, p]   MAP estimate in log10 space
      k = condition        c = cell (recording)
      t = chunk (window)   p = parameter

built by :func:`reshape_to_grid`. Every entry is the MAP point estimate the Experiment stage
optimized for one (condition, cell, chunk) window -- never a posterior draw and never an average.

THE FOUR CENTRAL ESTIMATES. A timeseries needs one vector per chunk, which means aggregating the
cell axis; a single summary line needs one vector overall, which means aggregating the cell AND
chunk axes. Crossing that choice of axis with the choice of estimator gives exactly four, and each
function below is named for what it aggregates:

    mean-window       mean value vector, aggregated across cells for a given chunk
    sgm-window        realized value vector, aggregated across cells for a given chunk
    mean-trajectory   mean value vector, aggregated across chunks and cells
    sgm-trajectory    realized value vector, aggregated across chunks and cells

"Mean" aggregates each parameter independently, so its coordinates need not have co-occurred in any
recording. "Realized" selects an actual member of the set -- the exact medoid, the member minimizing
the summed distance to every other member -- so every coordinate co-occurred in one real window and
the joint structure is intact (Ramirez Sierra & Sokolowski, Mach. Learn.: Sci. Technol. 6, 015004,
2025). The two `*-window` functions produce a timeseries; the two `*-trajectory` functions produce a
single vector, and they pair with their window counterpart so a figure never mixes estimators.

DISTANCES. Both realized estimates use the same metric: absolute (physical) values ``10**G``, each
parameter divided by its absolute prior width ``10**high - 10**low`` so no parameter dominates,
Euclidean, exact medoid. Selection is on ALL parameters jointly, so a selected vector is internally
coherent -- and consequently the value it reports for one parameter is that jointly-central window's
value, not that parameter's own median.
"""
from __future__ import annotations

import numpy as np

from .sample_geometric_median import sample_geometric_median

# A change of this many dex over the recording is called material: 0.3 dex is a factor of two, the
# same practical bar the recovery tables use, so exceeding it moves the estimate by more than the
# tolerance the recovery is judged against.
MATERIAL_DRIFT_DEX = 0.3

# Held-out recovery tolerances, as log10 half-widths, matching the two nested bands the Evaluation
# stage reports. Kept separate from MATERIAL_DRIFT_DEX even though the wider one coincides
# numerically: one is a tolerance on recovery against known truth, the other a threshold on drift
# across a recording, and conflating them would tie two unrelated decisions to one constant.
RECOVERY_BANDS_DEX = (0.3, 0.15)


def band_label(dex):
    """Render a log10 half-width as the multiplicative range it means, e.g. ``[0.50x, 2.0x]``.

    A tolerance stated in dex is unreadable without mental arithmetic, and the arithmetic is the
    interesting part: +/-0.3 dex is "between half and double the truth", +/-0.15 dex is "within
    roughly a third either way". The range is asymmetric in absolute terms and symmetric in log,
    which the two multipliers show directly.
    """
    lo, hi = 10.0 ** -float(dex), 10.0 ** float(dex)
    return f"[{lo:.2f}x, {hi:.2f}x]"


def recovery_fractions(true_log10, inferred_log10, bands=RECOVERY_BANDS_DEX):
    """Fraction of held-out videos recovered inside each nested tolerance band, per parameter.

    Returns a list of ``(dex, fractions)`` pairs, one per band, with ``fractions`` shaped ``(D,)``.
    """
    err = np.abs(np.asarray(inferred_log10, dtype=float) - np.asarray(true_log10, dtype=float))
    return [(float(b), np.mean(err <= float(b), axis=0)) for b in bands]

CENTRAL_FAMILIES = ("sgm", "mean")


def reshape_to_grid(values, kind_index, cell, chunk, n_kinds):
    """Scatter flat per-window rows into a dense ``(kind, cell, chunk, ...)`` grid.

    The Experiment output stores one flat row per analyzed window. Each row is placed at
    ``grid[kind_index, cell, chunk]``. Windows never estimated stay NaN and every statistic here is
    nan-aware, so a missing window narrows nothing silently.

    Returns ``(grid, n_cells, n_chunks)``.
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
    """Absolute prior width per parameter -- the normalizer for every distance here."""
    lo = np.asarray(prior_low, dtype=float)
    hi = np.asarray(prior_high, dtype=float)
    span = 10.0 ** hi - 10.0 ** lo
    span[span <= 0] = 1.0
    return span


# =============================================================================
# The four central estimates
# =============================================================================

def mean_window(grid_log10):
    """**mean-window**: mean value vector, aggregated across cells for a given chunk.

    For each condition, chunk, and parameter independently, the arithmetic mean over cells of the
    absolute MAP values. Each parameter is averaged on its own, so the resulting vector's
    coordinates need not have co-occurred in any recording.

    Returns ``(n_kinds, n_chunks, D)`` in ABSOLUTE units.
    """
    return np.nanmean(10.0 ** np.asarray(grid_log10, dtype=float), axis=1)


def sgm_window(grid_log10, prior_low, prior_high):
    """**sgm-window**: realized value vector, aggregated across cells for a given chunk.

    At each chunk independently, the exact medoid among that chunk's cell vectors: the cell whose
    D-dimensional MAP vector minimizes the summed normalized distance to the other cells' vectors at
    the same chunk. The returned vector is that cell's stored values verbatim.

    The selected cell is returned per chunk because **it may differ between chunks**: a step in the
    resulting series can be a change of cell rather than a change in time, and the caller must show
    the selections wherever it shows the curve.

    Returns ``(series_abs, cells)`` -- ``(n_kinds, n_chunks, D)`` in ABSOLUTE units and
    ``(n_kinds, n_chunks)`` selected cell indices (-1 where a chunk has no complete cell).
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
            block = 10.0 ** np.stack([grid[k, c, t] for c in rows], axis=0)
            idx, _method = sample_geometric_median(block, span)
            picked[k, t] = rows[idx]
            out[k, t] = block[idx]
    return out, picked


def mean_trajectory(grid_log10):
    """**mean-trajectory**: mean value vector, aggregated across chunks and cells.

    For each condition and parameter independently, the arithmetic mean over every (cell, chunk)
    window of the absolute MAP values -- the grand mean over both axes. This is the single-vector
    counterpart of :func:`mean_window`.

    Returns ``(n_kinds, D)`` in ABSOLUTE units.
    """
    g = 10.0 ** np.asarray(grid_log10, dtype=float)
    return np.nanmean(g.reshape(g.shape[0], -1, g.shape[3]), axis=1)


def sgm_trajectory(grid_log10, prior_low, prior_high):
    """**sgm-trajectory**: realized value vector, aggregated across chunks and cells.

    The exact medoid among ALL (cell, chunk) window vectors of a condition: the single window whose
    D-dimensional MAP vector minimizes the summed normalized distance to every other window vector.
    The result is one genuinely realized vector, drawn from one specific cell at one specific chunk,
    and it is the single-vector counterpart of :func:`sgm_window`.

    This is the same quantity the standalone sample-geometric-median analysis reports over the same
    pooled windows, so the two analyses agree by construction.

    Returns ``(vector_abs, picks)`` -- ``(n_kinds, D)`` in ABSOLUTE units and a list of
    ``(cell, chunk)`` tuples naming the selected window per condition (``(-1, -1)`` if none).
    """
    grid = np.asarray(grid_log10, dtype=float)
    n_kinds, n_cells, n_chunks, dim = grid.shape
    span = _range_abs(prior_low, prior_high)
    out = np.full((n_kinds, dim), np.nan)
    picks = []
    for k in range(n_kinds):
        rows, labels = [], []
        for c in range(n_cells):
            for t in range(n_chunks):
                if np.isfinite(grid[k, c, t]).all():
                    rows.append(10.0 ** grid[k, c, t])
                    labels.append((c, t))
        if not rows:
            picks.append((-1, -1))
            continue
        block = np.stack(rows, axis=0)
        idx, _method = sample_geometric_median(block, span)
        out[k] = block[idx]
        picks.append(labels[idx])
    return out, picks


# =============================================================================
# Drift statistics -- fit per cell, so independent of the central-estimate choice
# =============================================================================

def pooled_cloud(cloud_log10, kind_index, n_kinds):
    """Pool every window's posterior draws within each condition, collapsing the time axis.

    ``cloud_log10`` is ``(N, S, D)``: for each of the ``N`` (recording, window) pairs, the ``S``
    draws the Experiment stage took from that window's posterior. Selecting one condition and
    flattening the leading two axes gives ``(n_windows * S, D)`` draws.

    WHAT THIS IS. The pooled set is the equal-weight MIXTURE of the per-window posteriors, so its
    density answers: for a 2 s window drawn at random from this condition, what values are
    consistent with it? It is emphatically NOT a joint posterior for the condition. Combining
    independent observations under Bayes multiplies their likelihoods; pooling draws adds their
    densities. The mixture is therefore as wide as the spread BETWEEN windows plus the uncertainty
    WITHIN each one, whereas a genuine joint posterior over 250 windows would be far narrower than
    any single one of them. Read it as a description of the window-to-window population, never as
    evidence accumulated across the recording.

    Because the mixture is a population and not an estimate, its mode and median are not the
    analysis's central estimate: that remains the trajectory-level medoid, which is one jointly
    realized vector rather than a per-parameter summary of pooled marginals.

    Returns a list of ``n_kinds`` arrays, each ``(n_windows_k * S, D)`` in log10 units.
    """
    cloud_log10 = np.asarray(cloud_log10, dtype=float)
    kind_index = np.asarray(kind_index, dtype=int)
    out = []
    for e in range(n_kinds):
        sel = cloud_log10[kind_index == e]                     # (n_windows_k, S, D)
        out.append(sel.reshape(-1, sel.shape[-1]) if sel.size else sel.reshape(0, 0))
    return out


def cloud_interval(pooled, quantiles=(0.05, 0.50, 0.95)):
    """Per-parameter quantiles of a pooled mixture, for the report table.

    Marginal quantiles of a mixture, reported per parameter: they describe each coordinate's spread
    across the condition, and -- being per-coordinate -- they carry no joint information, which is
    exactly why they belong in a table beside the medoid rather than replacing it.
    """
    if pooled.size == 0:
        return np.full((3, 0), np.nan)
    return np.quantile(pooled, list(quantiles), axis=0)         # (len(quantiles), D)


def pooled_summary(pooled, statistic="median"):
    """Marginal central value of a pooled mixture, per parameter, in ABSOLUTE units.

    The 2x2 of central estimates -- ``{mean, sgm} x {window, trajectory}`` -- all summarize the set
    of per-window MAP vectors. This summarizes the pooled posterior DRAWS instead, which is a
    different object: the mixture is where the probability mass is, while the MAP set is where the
    per-window modes are, and the two need not agree. Naming follows the same scheme:
    ``median-pooled`` and ``mean-pooled``.

    WHY THIS EXISTS. A vertical line drawn on a one-dimensional marginal histogram is read as the
    center of THAT histogram. The trajectory-level medoid is not: it is one jointly realized vector
    chosen to minimize distance in the full parameter space, so any single coordinate of it can sit
    far from that coordinate's marginal center -- correctly, but not in a way a marginal plot
    communicates. These statistics are the marginal centers of the plotted distribution, and are the
    honest thing to mark on it.

    WHAT IS LOST. Being per-coordinate, they are composites: the returned vector's coordinates come
    from different draws and need never have co-occurred, which is precisely the defect the geometric
    median exists to avoid. Use them to describe a marginal, never as the condition's parameter
    vector.

    ``statistic``:
        ``"median"``  the marginal median. Equivariant under monotone transformation, so the median
                      in absolute units and ``10 ** median(log10)`` are the SAME number and the
                      answer does not depend on the space it was computed in. This is the default
                      for that reason.
        ``"mean"``    the arithmetic mean in absolute units. NOT equivariant: it differs from the
                      geometric mean ``10 ** mean(log10)`` by a factor that grows with the spread,
                      and for a log-uniform prior that factor is large. Provided because it is what
                      "the mean of the posterior" usually names, but it reports a property of the
                      chosen basis as much as of the posterior.
        ``"geometric-mean"`` ``10 ** mean(log10)``, the mean in the space the prior is uniform in.

    Returns an ``(n_kinds, D)`` array in absolute units.
    """
    if statistic not in ("median", "mean", "geometric-mean"):
        raise ValueError(f"statistic={statistic!r}; expected 'median', 'mean' or 'geometric-mean'.")
    out = []
    for draws_log10 in pooled:
        if draws_log10.size == 0:
            out.append(np.full(draws_log10.shape[-1:], np.nan))
            continue
        if statistic == "median":
            out.append(np.median(10.0 ** draws_log10, axis=0))
        elif statistic == "mean":
            out.append(np.mean(10.0 ** draws_log10, axis=0))
        else:
            out.append(10.0 ** np.mean(draws_log10, axis=0))
    return np.asarray(out)


def drift_statistics(grid_log10, times, threshold=MATERIAL_DRIFT_DEX):
    """Per-cell linear drift of each parameter across the recording, aggregated per condition.

    For every (condition, cell, parameter) an ordinary least-squares line is fit to the stored log10
    MAP estimate against time -- log10 because drift is multiplicative -- giving a slope and hence a
    fitted start and end value:

        change_dex = slope * (t_last - t_first)
        start      = 10 ** intercept_at_t_first
        end        = 10 ** (intercept_at_t_first + change_dex)

    Every reported statistic aggregates those per-cell fits, so **none of them depends on which
    central estimate the figures display**: swapping a mean for a medoid changes what is drawn, not
    what is measured. Named results, all shaped ``(n_kinds, D)`` unless noted:

        drift_absolute          median across cells of (end - start), in the parameter's own units
        drift_sign_consistency  fraction of cells whose change shares the median's sign
        drift_fold              median across cells of (end / start), a multiplicative factor
        drift_dex               median across cells of change_dex, in log10 units
        drift_material_fraction fraction of cells whose |change_dex| exceeds ``threshold``
        drift_wilcoxon_p        two-sided signed-rank p that per-cell change_dex is centered at
                                zero -- a DETECTABILITY statement, not a magnitude; NaN when scipy
                                is unavailable or fewer than six cells contribute
        start_median, end_median  median across cells of the fitted endpoints, absolute units
        change_dex_per_cell     ``(n_kinds, n_cells, D)`` the underlying per-cell changes
    """
    grid = np.asarray(grid_log10, dtype=float)
    t = np.asarray(times, dtype=float)
    n_kinds, n_cells, n_chunks, dim = grid.shape
    span = float(t[-1] - t[0]) if n_chunks > 1 else 0.0
    change = np.full((n_kinds, n_cells, dim), np.nan)
    start = np.full((n_kinds, n_cells, dim), np.nan)
    end = np.full((n_kinds, n_cells, dim), np.nan)
    for k in range(n_kinds):
        for c in range(n_cells):
            for p in range(dim):
                y = grid[k, c, :, p]
                ok = np.isfinite(y)
                if ok.sum() < 2:
                    continue
                slope, intercept = np.polyfit(t[ok], y[ok], 1)
                d = slope * span
                change[k, c, p] = d
                start[k, c, p] = 10.0 ** (intercept + slope * t[0])
                end[k, c, p] = 10.0 ** (intercept + slope * t[0] + d)
    median_dex = np.nanmedian(change, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sign = np.nanmean(np.sign(change) == np.sign(median_dex)[:, None, :], axis=1)
        material = np.nanmean(np.abs(change) > threshold, axis=1)
        fold = np.nanmedian(end / start, axis=1)
    pvals = np.full((n_kinds, dim), np.nan)
    try:
        from scipy.stats import wilcoxon
        for k in range(n_kinds):
            for p in range(dim):
                v = change[k, :, p]
                v = v[np.isfinite(v)]
                if v.size >= 6 and np.any(v != 0):
                    pvals[k, p] = float(wilcoxon(v)[1])
    except ImportError:
        pass
    return {
        "drift_absolute": np.nanmedian(end - start, axis=1),
        "drift_sign_consistency": sign,
        "drift_fold": fold,
        "drift_dex": median_dex,
        "drift_material_fraction": material,
        "drift_wilcoxon_p": pvals,
        "start_median": np.nanmedian(start, axis=1),
        "end_median": np.nanmedian(end, axis=1),
        "change_dex_per_cell": change,
        "threshold": float(threshold),
    }


def within_window_interval(quant_grid_log10):
    """Median across cells, per chunk, of the stored per-window posterior quantile levels.

    Summarizes how uncertain a TYPICAL SINGLE window's estimate is: for each condition, chunk, and
    quantile level, the median over cells of that level. Taking the median of a level across cells
    (rather than pooling) keeps the reported interval the interval of one typical window instead of
    an envelope over recordings.

    This is an interval WIDTH summary of the stored five-quantile record, not a posterior density:
    a density pooled over windows would require the per-window sample clouds.

    Returns ``(n_kinds, n_chunks, D, Q)`` in log10 space.
    """
    return np.nanmedian(np.asarray(quant_grid_log10, dtype=float), axis=1)

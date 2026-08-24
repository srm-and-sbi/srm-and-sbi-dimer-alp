"""Population-composition kernel: relative species abundance derived from joint count draws.

Workflow-agnostic numerics. Nothing here knows that the counted things are receptors: every
function takes arrays, the indices of the coordinates holding species counts, and a prior box. Pure
numpy plus a lazy scipy import for the three rank tests, so it imports and unit-tests without a
machine profile.

WHAT THE ANALYSIS ASKS. The estimator infers three absolute species counts -- A (monomer), B (mobile
dimer), C (immobile dimer) -- independently in every window of every recording. Absolute counts are
the *worst*-identified coordinates the model has, and the reason is information-theoretic rather than
a defect: counting few emitters in a diffraction-limited scene is a square-root-of-n problem, and the
three counts additionally trade off against one another because the reactions interconvert them. The
question this analysis asks is therefore not "how many" but "in what proportion", because the
proportions are a different -- and far better identified -- function of the same posterior:

    monomer fraction        f_A = A / T                     with T = A + B + C
    mobile-dimer fraction   f_B = B / T
    immobile-dimer fraction f_C = C / T
    dimer-complex fraction  f_D = (B + C) / T = 1 - f_A
    receptors in dimers     f_R = 2(B + C) / (A + 2B + 2C)
    total complexes         T   = A + B + C

WHY THE DRAWS MUST BE JOINT. A ratio of correlated coordinates is not a function of their marginals.
The posterior places B and C in strong negative correlation (the two dimer states absorb the same
signal), so a fraction built from marginal medians -- median(B) over median(A)+median(B)+median(C) --
asserts a combination the posterior never drew and inherits none of the cancellation that makes the
ratio identifiable. Every fraction here is therefore formed WITHIN each posterior draw, and only then
averaged. That single ordering is what turns three poorly-identified counts into a well-identified
composition, and reversing it silently discards the effect.

THE AGGREGATION LADDER. Four levels, each an unweighted mean over the level below:

    draws   -> window     the posterior mean of the fraction for that window
    windows -> recording  the recording's composition over its analyzed span
    cells   -> condition  the reported value, with the SEM taken HERE

The replicate unit is the recording, never the window: windows within a recording are ten views of
one cell, so treating them as independent would divide the standard error by the square root of the
window count and manufacture significance out of pseudo-replication. Every interval and every test
below is computed over recordings for that reason.

CLOSURE. f_A + f_B + f_C = 1 holds for every draw by construction, and |Delta f_D| = |Delta f_A|
identically, so the monomer and dimer-complex rows of a recovery table are the same number reported
twice -- an internal consistency check, not two independent results.

WHAT THIS KERNEL DOES NOT CLAIM. The composition is conditional on the three-species model: it is a
census of the model's states as fitted to the recordings, not a molecular count. Its value is that
the same quantity can be computed on held-out synthetic videos where the truth is known, so the
in-model error of the readout is measured rather than assumed.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .temporal_dynamics import reshape_to_grid

# Per-cell monomer-fraction bands used only to describe how spread out the recordings are (how many
# cells sit at the low and high extremes). Descriptive thresholds for one report table, deliberately
# round: nothing downstream branches on them and no test uses them.
HETEROGENEITY_BANDS = (0.10, 0.40)


@dataclass(frozen=True)
class Quantity:
    """One derived quantity of the composition.

    Attributes:
        key: identifier used in tables, figures and the saved arrays.
        symbol: short mathematical name for figure axes and narrow columns.
        formula: the definition, so a report never states a value without its meaning.
        is_fraction: True for the shares of the population (reported in percent, error in percentage
            points); False for the total, which is a count and whose error is multiplicative and so
            reported in dex.
    """

    key: str
    symbol: str
    formula: str
    is_fraction: bool


# The reported quantities, in table order. The three shares come first because they partition the
# population; the two derived reads follow; the total closes the list because it is the only row in
# different units.
COMPOSITION = (
    Quantity("monomer_fraction", "f_A", "A / T", True),
    Quantity("mobile_dimer_fraction", "f_B", "B / T", True),
    Quantity("immobile_dimer_fraction", "f_C", "C / T", True),
    Quantity("dimer_complex_fraction", "f_D", "(B + C) / T", True),
    Quantity("receptors_in_dimers", "f_R", "2(B + C) / (A + 2B + 2C)", True),
    Quantity("total_complexes", "T", "A + B + C", False),
)
COMPOSITION_KEYS = tuple(q.key for q in COMPOSITION)
FRACTION_INDICES = tuple(i for i, q in enumerate(COMPOSITION) if q.is_fraction)
PARTS_INDICES = (0, 1, 2)          # the three shares that partition the population, in A, B, C order
TOTAL_INDEX = len(COMPOSITION) - 1
DIMER_INDEX = COMPOSITION_KEYS.index("dimer_complex_fraction")


def composition(counts_abs):
    """The six derived quantities from absolute species counts.

    ``counts_abs`` is ``(..., 3)`` holding A, B, C in absolute units (never log10) -- any leading
    shape is preserved, so the same call serves a single MAP vector, a window's draws, or the whole
    held-out set. Returns ``(..., 6)`` ordered as :data:`COMPOSITION`.

    The counts are strictly positive (they are ``10 ** theta`` of a log-uniform coordinate), so the
    denominators cannot vanish and no guard is needed. Should a caller ever pass a zero total, the
    division yields NaN, which propagates into the nan-aware aggregation rather than raising -- a
    missing value, which is the truthful outcome, instead of a crash or a fabricated zero.
    """
    counts_abs = np.asarray(counts_abs, dtype=float)
    if counts_abs.shape[-1] != 3:
        raise ValueError(f"composition() needs three species counts on the last axis, "
                         f"got shape {counts_abs.shape}.")
    a, b, c = counts_abs[..., 0], counts_abs[..., 1], counts_abs[..., 2]
    total = a + b + c
    dimers = b + c
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.stack([a / total, b / total, c / total, dimers / total,
                        2.0 * dimers / (a + 2.0 * dimers), total], axis=-1)
    return out


def closure_residual(comp):
    """Largest deviation of ``f_A + f_B + f_C`` from one, over every entry of a composition array.

    Zero to floating-point precision by construction; reported as a check because it is the cheapest
    possible detection of an index mix-up between the count coordinates and their neighbors in theta.
    """
    comp = np.asarray(comp, dtype=float)
    parts = comp[..., list(PARTS_INDICES)].sum(axis=-1)
    return float(np.nanmax(np.abs(parts - 1.0))) if parts.size else float("nan")


# =============================================================================
# The aggregation ladder
# =============================================================================

def window_composition(cloud_log10, count_index, draw_mask=None):
    """Per-window posterior mean composition from the stored per-window posterior draws.

    ``cloud_log10`` is ``(N, S, D)``: for each of the ``N`` (recording, window) pairs, the ``S`` draws
    the Experiment stage took from that window's posterior, in log10 units. ``count_index`` names the
    three coordinates holding the species counts. The composition is formed inside every draw and
    then averaged over draws -- the ordering the module docstring insists on.

    ``draw_mask`` is an optional ``(N, S)`` boolean selecting which draws contribute (used by the
    support-restriction variants). Windows left with no contributing draw return NaN for every
    quantity rather than a value derived from nothing, and the nan-aware levels above drop them; the
    caller is expected to report how many windows were lost, because a variant that silently discards
    half the recordings is not the same measurement as one that keeps them.

    Returns ``(N, 6)``.
    """
    cloud_log10 = np.asarray(cloud_log10, dtype=float)
    comp = composition(10.0 ** cloud_log10[..., list(count_index)])       # (N, S, 6)
    if draw_mask is not None:
        comp = np.where(np.asarray(draw_mask, dtype=bool)[..., None], comp, np.nan)
    with np.errstate(invalid="ignore"):
        # All-NaN windows are the expected outcome of a restriction variant, not an anomaly: mean of
        # an empty selection is NaN, which is what the ladder should propagate.
        return _nanmean_quiet(comp, axis=1)


def _nanmean_quiet(values, axis):
    """``np.nanmean`` without the all-NaN-slice warning, since all-NaN slices are expected here."""
    values = np.asarray(values, dtype=float)
    with np.errstate(invalid="ignore"):
        count = np.sum(~np.isnan(values), axis=axis)
        total = np.nansum(values, axis=axis)
        out = np.where(count > 0, total / np.maximum(count, 1), np.nan)
    return out


def composition_grid(window_comp, kind_index, cell, chunk, n_kinds):
    """Scatter per-window compositions into the dense ``(condition, recording, window, 6)`` grid.

    Delegates to the temporal-dynamics kernel's grid builder so the two analyses share ONE definition
    of what a (condition, recording, window) grid is: a change to the layout cannot land in one and
    miss the other. Windows never estimated stay NaN and every statistic below is nan-aware.

    Returns ``(grid, n_cells, n_chunks)``.
    """
    return reshape_to_grid(window_comp, kind_index, cell, chunk, n_kinds)


def per_cell(grid):
    """Each recording's composition: the mean over its analyzed windows. Returns ``(K, C, 6)``.

    This is the replicate-level value -- the unit every interval and test below is computed over.
    """
    return _nanmean_quiet(np.asarray(grid, dtype=float), axis=2)


def condition_summary(cell_values):
    """Condition-level mean, standard error, and contributing-recording count.

    The SEM is taken across RECORDINGS (``std(ddof=1) / sqrt(n_cells)``), so it describes
    cell-to-cell biological variability -- the spread that matters for a condition-level claim -- and
    not the within-window posterior width, which is a different quantity and much smaller.

    Returns ``(mean, sem, n)``, each ``(K, 6)`` (``n`` integer).
    """
    v = np.asarray(cell_values, dtype=float)
    n = np.sum(~np.isnan(v), axis=1)
    mean = _nanmean_quiet(v, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(_nanmean_quiet((v - mean[:, None, :]) ** 2, axis=1) * n / np.maximum(n - 1, 1))
        sem = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    return mean, sem, n.astype(int)


def window_slice_summary(grid, window):
    """Condition-level mean and SEM of ONE window index, across recordings. Returns ``(mean, sem)``.

    The first window is reported separately from the span average wherever the composition is not
    stationary: an average over a changing population describes no instant of it, while window zero
    is the earliest state the recordings resolve.
    """
    g = np.asarray(grid, dtype=float)[:, :, int(window), :]
    n = np.sum(~np.isnan(g), axis=1)
    mean = _nanmean_quiet(g, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(_nanmean_quiet((g - mean[:, None, :]) ** 2, axis=1) * n / np.maximum(n - 1, 1))
        sem = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    return mean, sem


def time_course(grid):
    """Per-window condition means and SEMs across recordings. Returns ``(mean, sem)``, ``(K, T, 6)``.

    The window axis is kept because a composition that changes across a recording is a finding, and
    an analysis that reported only the span average would hide it inside the average.
    """
    g = np.asarray(grid, dtype=float)
    n = np.sum(~np.isnan(g), axis=1)
    mean = _nanmean_quiet(g, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        sd = np.sqrt(_nanmean_quiet((g - mean[:, None, :, :]) ** 2, axis=1) * n / np.maximum(n - 1, 1))
        sem = np.where(n > 1, sd / np.sqrt(np.maximum(n, 1)), np.nan)
    return mean, sem


# =============================================================================
# Robustness: is the reported composition an artifact of a choice?
# =============================================================================

def support_masks(cloud_log10, prior_low, prior_high, count_index):
    """Draw-level masks for the three support variants, with what each one costs.

    A posterior fitted to real recordings can place mass outside the box the estimator was trained
    on, and absolute counts are exactly where that happens: dense scenes push the count coordinates
    past the prior ceiling. Since the composition is a ratio, out-of-support mass may inflate all
    three counts together and largely cancel -- but that is a claim to be measured, not assumed, so
    the same composition is recomputed under progressively harsher restrictions:

        ``unrestricted``  every draw. The honest description of where the posterior actually is.
        ``count_box``     draws whose three count coordinates lie inside the trained box.
        ``full_box``      draws inside the box in ALL coordinates -- the only variant every part of
                          which the estimator was trained to represent, and usually the one that
                          discards the most.

    Returns a list of ``(name, mask, retained_fraction_per_condition_input, note)`` where ``mask`` is
    ``(N, S)``. Retention and the count of windows left empty are computed by the caller, which knows
    the condition labels.
    """
    cloud_log10 = np.asarray(cloud_log10, dtype=float)
    low = np.asarray(prior_low, dtype=float)
    high = np.asarray(prior_high, dtype=float)
    inside_all = np.all((cloud_log10 >= low) & (cloud_log10 <= high), axis=-1)
    idx = list(count_index)
    inside_counts = np.all((cloud_log10[..., idx] >= low[idx])
                           & (cloud_log10[..., idx] <= high[idx]), axis=-1)
    ones = np.ones(cloud_log10.shape[:2], dtype=bool)
    return [
        ("unrestricted", ones,
         "every posterior draw, including mass beyond the trained support"),
        ("count_box", inside_counts,
         "draws whose three species counts lie inside the trained prior box"),
        ("full_box", inside_all,
         "draws inside the trained prior box in every coordinate"),
    ]


def compositional_center(cell_values):
    """The closed geometric mean of the three shares across recordings -- the compositional center.

    An arithmetic mean of fractions ignores that a composition lives on the simplex, where the
    natural geometry is multiplicative: differences are ratios between parts, not differences of
    percentages. The center used in compositional data analysis is the normalized geometric mean of
    the parts (equivalently, the arithmetic mean in log-ratio coordinates mapped back), and it is the
    right alternative to test the headline against. It is reported as a sensitivity variant rather
    than as the headline because it is not the mean of anything a reader is likely to have in mind,
    and because whichever of the two is smaller is the conservative one to lead with.

    Returns ``(K, 6)`` with the three shares recomposed and the two derived reads rebuilt from them;
    the total column is NaN, since a count has no compositional center.
    """
    v = np.asarray(cell_values, dtype=float)
    parts = v[:, :, list(PARTS_INDICES)]
    with np.errstate(invalid="ignore", divide="ignore"):
        # Geometric mean over recordings, per part, then closed to sum to one.
        log_parts = np.log(np.where(parts > 0, parts, np.nan))
        g = np.exp(_nanmean_quiet(log_parts, axis=1))                      # (K, 3)
        g = g / g.sum(axis=1, keepdims=True)
    out = np.full((v.shape[0], len(COMPOSITION)), np.nan)
    out[:, list(PARTS_INDICES)] = g
    a, b, c = g[:, 0], g[:, 1], g[:, 2]
    out[:, DIMER_INDEX] = b + c
    out[:, COMPOSITION_KEYS.index("receptors_in_dimers")] = 2.0 * (b + c) / (a + 2.0 * (b + c))
    return out


def bootstrap_interval(cell_values, n_resamples, rng, level=0.95):
    """Percentile bootstrap interval for the condition mean, resampling RECORDINGS.

    Resampling the replicate unit makes the interval free of the normal-theory assumption behind the
    SEM. Agreement between the two is the point of computing it: it says the reported error bars do
    not depend on that assumption. Disagreement would say the cell-to-cell distribution is skewed
    enough that the SEM misstates it.

    Returns ``(lo, hi)``, each ``(K, 6)``.
    """
    v = np.asarray(cell_values, dtype=float)
    n_kinds, n_cells, dim = v.shape
    lo = np.full((n_kinds, dim), np.nan)
    hi = np.full((n_kinds, dim), np.nan)
    if n_resamples <= 0 or n_cells < 2:
        return lo, hi
    alpha = 0.5 * (1.0 - float(level))
    for k in range(n_kinds):
        rows = v[k][~np.isnan(v[k]).all(axis=1)]
        if rows.shape[0] < 2:
            continue
        pick = rng.integers(0, rows.shape[0], size=(int(n_resamples), rows.shape[0]))
        means = _nanmean_quiet(rows[pick], axis=1)                          # (n_resamples, 6)
        lo[k] = np.nanquantile(means, alpha, axis=0)
        hi[k] = np.nanquantile(means, 1.0 - alpha, axis=0)
    return lo, hi


def heterogeneity(cell_values, quantity_index, bands=HETEROGENEITY_BANDS):
    """How spread out the recordings are in one quantity: range, median, and the extremes' weight.

    A condition mean is a poor summary of a bimodal set of recordings, and whether it is one is a
    property of the data that a table of means cannot show. Returns one dict per condition with the
    minimum, median, maximum, the number of recordings below the low band and above the high band,
    and the contributing count.
    """
    v = np.asarray(cell_values, dtype=float)[:, :, int(quantity_index)]
    lo_band, hi_band = bands
    out = []
    for k in range(v.shape[0]):
        row = v[k][np.isfinite(v[k])]
        if row.size == 0:
            out.append(dict(n=0))
            continue
        out.append(dict(n=int(row.size), minimum=float(row.min()), median=float(np.median(row)),
                        maximum=float(row.max()), n_below=int((row < lo_band).sum()),
                        n_above=int((row > hi_band).sum()), band=(float(lo_band), float(hi_band))))
    return out


# =============================================================================
# Tests -- all on the recording as replicate unit
# =============================================================================

def condition_contrast(cell_values, kind_a=0, kind_b=1):
    """Two-sided rank-sum comparison of two conditions, per quantity, over recordings.

    A rank test rather than a t-test because the per-recording values are bounded fractions with no
    reason to be normal, and because it is the comparison a reader can check by eye against the
    per-cell figure. ``p`` is NaN when scipy is unavailable or a condition contributes fewer than
    three recordings.

    Returns a list of dicts, one per quantity: medians, ratio, U, p, and the two counts.
    """
    v = np.asarray(cell_values, dtype=float)
    out = []
    try:
        from scipy.stats import mannwhitneyu
    except ImportError:
        mannwhitneyu = None
    for i, q in enumerate(COMPOSITION):
        a = v[kind_a, :, i][np.isfinite(v[kind_a, :, i])]
        b = v[kind_b, :, i][np.isfinite(v[kind_b, :, i])]
        row = dict(key=q.key, n_a=int(a.size), n_b=int(b.size),
                   median_a=float(np.median(a)) if a.size else float("nan"),
                   median_b=float(np.median(b)) if b.size else float("nan"),
                   statistic=float("nan"), p=float("nan"))
        with np.errstate(invalid="ignore", divide="ignore"):
            row["ratio"] = (row["median_b"] / row["median_a"]
                            if row["median_a"] not in (0.0,) else float("nan"))
        if mannwhitneyu is not None and a.size >= 3 and b.size >= 3:
            stat, p = mannwhitneyu(a, b, alternative="two-sided")
            row["statistic"], row["p"] = float(stat), float(p)
        out.append(row)
    return out


def monotonicity(grid, quantity_index):
    """Does one quantity move systematically across a recording? Per condition, over recordings.

    Two statements, deliberately separate:

      * ``spearman_median`` -- the median across recordings of each recording's Spearman correlation
        between the quantity and the window index. A direction and a strength, robust to the shape of
        the trend and to a single outlying window.
      * ``wilcoxon_p`` -- the signed-rank p that each recording's last window equals its first. A
        DETECTABILITY statement about the endpoints, paired within recording so between-cell spread
        cannot mask it.

    Both are NaN when scipy is unavailable or fewer than six recordings contribute. Also returns the
    first- and last-window medians so the p-value is never read without a magnitude beside it.
    """
    g = np.asarray(grid, dtype=float)[:, :, :, int(quantity_index)]
    n_kinds, _n_cells, n_chunks = g.shape
    out = []
    try:
        from scipy.stats import spearmanr, wilcoxon
    except ImportError:
        spearmanr = wilcoxon = None
    idx = np.arange(n_chunks, dtype=float)
    for k in range(n_kinds):
        rhos, first, last = [], [], []
        for row in g[k]:
            ok = np.isfinite(row)
            if ok.sum() >= 3 and spearmanr is not None and np.ptp(row[ok]) > 0:
                rhos.append(float(spearmanr(idx[ok], row[ok]).statistic))
            if np.isfinite(row[0]) and np.isfinite(row[-1]):
                first.append(float(row[0]))
                last.append(float(row[-1]))
        p = float("nan")
        if wilcoxon is not None and len(first) >= 6:
            diff = np.asarray(last) - np.asarray(first)
            if np.any(diff != 0):
                p = float(wilcoxon(diff)[1])
        out.append(dict(
            n_cells=len(first),
            spearman_median=float(np.median(rhos)) if rhos else float("nan"),
            wilcoxon_p=p,
            first_median=float(np.median(first)) if first else float("nan"),
            last_median=float(np.median(last)) if last else float("nan")))
    return out


# =============================================================================
# Synthetic validation: the same readout where the truth is known
# =============================================================================

def recovery_statistics(true_log10, inferred_log10, count_index):
    """In-model error of the composition readout on held-out videos with known ground truth.

    This is what makes the experimental composition interpretable: the identical function of the
    identical estimator, applied where the answer is known. Fractions are compared in PERCENTAGE
    POINTS (an absolute difference of two bounded quantities); the total is compared in DEX (a
    multiplicative error, the natural scale of a count spanning orders of magnitude).

    Reported per quantity: mean absolute error, its 95th percentile, the signed bias, and the
    correlation between truth and estimate. Both arrays are MAP point estimates, so this is
    point-estimate accuracy -- NOT interval coverage, which needs joint posterior draws on the
    held-out set and is therefore not computable from a recovery artifact that stores only marginal
    quantiles.

    Returns ``(rows, true_comp, inferred_comp)``.
    """
    true_comp = composition(10.0 ** np.asarray(true_log10, dtype=float)[:, list(count_index)])
    inf_comp = composition(10.0 ** np.asarray(inferred_log10, dtype=float)[:, list(count_index)])
    rows = []
    for i, q in enumerate(COMPOSITION):
        t, e = true_comp[:, i], inf_comp[:, i]
        ok = np.isfinite(t) & np.isfinite(e)
        if q.is_fraction:
            err = e[ok] - t[ok]
            unit = "pp"
            scale = 100.0
            corr_t, corr_e = t[ok], e[ok]
        else:
            with np.errstate(invalid="ignore", divide="ignore"):
                err = np.log10(e[ok]) - np.log10(t[ok])
            unit = "dex"
            scale = 1.0
            corr_t, corr_e = np.log10(t[ok]), np.log10(e[ok])
        rows.append(dict(
            key=q.key, unit=unit, n=int(ok.sum()),
            mae=float(np.mean(np.abs(err)) * scale),
            p95=float(np.quantile(np.abs(err), 0.95) * scale),
            bias=float(np.mean(err) * scale),
            r=float(np.corrcoef(corr_t, corr_e)[0, 1]) if ok.sum() > 2 else float("nan")))
    return rows, true_comp, inf_comp


def recovery_stratum(true_comp, inferred_comp, quantity_index, threshold, above=True):
    """The same error statistics restricted to one region of truth.

    A mean error over the whole prior can hide a weak corner, and the corner that matters is the one
    the real recordings occupy. Restricting by TRUE value (never by estimate) keeps the selection
    independent of the estimator's own error, which selecting on the estimate would not.
    """
    t = np.asarray(true_comp, dtype=float)[:, int(quantity_index)]
    e = np.asarray(inferred_comp, dtype=float)[:, int(quantity_index)]
    sel = (t > float(threshold)) if above else (t < float(threshold))
    sel &= np.isfinite(t) & np.isfinite(e)
    if not sel.any():
        return dict(n=0)
    err = (e[sel] - t[sel]) * 100.0
    return dict(n=int(sel.sum()), mae=float(np.mean(np.abs(err))), bias=float(np.mean(err)),
                p95=float(np.quantile(np.abs(err), 0.95)),
                threshold=float(threshold), above=bool(above))


def parts_versus_whole(true_log10, inferred_log10, count_index):
    """Per-count error in dex beside the total's, the comparison the composition argument rests on.

    The individual counts and their sum are recovered from the same posterior by the same estimator;
    if the sum is recovered far better than any part, the parts' errors are substantially
    anti-correlated -- they trade off inside the posterior -- and any quantity that divides one part
    by the total inherits that cancellation. Returns one row per count plus the total.
    """
    t = np.asarray(true_log10, dtype=float)[:, list(count_index)]
    e = np.asarray(inferred_log10, dtype=float)[:, list(count_index)]
    rows = []
    for j, label in enumerate(("A (monomer)", "B (mobile dimer)", "C (immobile dimer)")):
        err = e[:, j] - t[:, j]
        rows.append(dict(key=label, mae=float(np.mean(np.abs(err))),
                         r=float(np.corrcoef(t[:, j], e[:, j])[0, 1])))
    with np.errstate(invalid="ignore", divide="ignore"):
        lt = np.log10((10.0 ** t).sum(axis=1))
        le = np.log10((10.0 ** e).sum(axis=1))
    rows.append(dict(key="T = A + B + C", mae=float(np.mean(np.abs(le - lt))),
                     r=float(np.corrcoef(lt, le)[0, 1])))
    return rows

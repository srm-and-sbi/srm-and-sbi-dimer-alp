"""Horizon-audit kernel: does inherited latent state break the reset assumption?

The estimator is trained on independently initialized model-window simulations: every training
video begins with freshly placed particles whose species counts are the drawn theta. The
experimental analysis, however, slices each long continuous recording into consecutive
model-length windows and runs the estimator on every window. Those two ensembles are equal in
window length but not necessarily in distribution: a later window of a continuous recording
inherits the latent state its past evolved into -- species populations relaxed away from the
initial counts, spatial organization, and any other carried history. Applying the estimator
window-by-window therefore assumes, silently, that inherited state does not make later windows
systematically different from reset training simulations. This kernel holds the pure numpy
machinery for testing that assumption under the simulator itself, where the truth is knowable.

THE CONTROLLED CONTRAST. For one theta, two ensembles of equal-length windows are compared:

    reset      -- independent model-window simulations, each freshly initialized at theta's
                  counts. This is the training factorization; the estimator sees exactly the
                  distribution it was fitted to.
    continuous -- one uninterrupted long simulation at the same theta, sliced into consecutive
                  non-overlapping model-length windows, exactly as the experimental recordings
                  are sliced.

Any systematic difference between the two -- in parameter error, in interval coverage, in the
composition readout -- measured per window position, is the horizon-mismatch effect. Both
outcomes are informative: degradation that grows with window position only in the continuous
ensemble measures the reset artifact; equivalence within prespecified margins bounds the tested
reset mechanism under the implemented simulator -- it does not identify the cause of any
experimental window drift, only whether this mechanism reproduces it.

TWO KINDS OF ESTIMAND. Rates and diffusivities are constant model parameters: their truth is the
drawn theta in every window, so window dependence in their estimates is spurious by definition.
The species counts are initial conditions of a dynamic state: after the first window the
population has evolved, so a later window's inferred counts must be compared against the actual
population in that window (extracted from the trajectory), not against theta -- comparing against
theta instead measures how far the state has drifted, which is a property of the dynamics, not an
estimator error. The state truth is the WINDOW-START population: the estimator's count labels are
the initial populations of its training windows, so the start state is what it was trained to
report; the within-window mean and end populations are kept as explicitly secondary sensitivity
references (judging against them manufactures an estimand mismatch that can dwarf the real error).

THE STATISTICAL UNIT. The windows of one continuous trajectory share their history and are
correlated; the trajectory (one theta, one continuous run plus its resets) is the independent
replicate. Every summary here therefore aggregates within a trajectory first and bootstraps over
trajectories, mirroring the recording-level ladder of the composition kernel.

This module is deliberately free of torch, file I/O, and machine-profile state: it computes, the
runner orchestrates. Composition quantities are delegated to :mod:`population_composition` so the
audit and the census share ONE definition of every derived read.
"""
from __future__ import annotations

import numpy as np


# =============================================================================
# Trajectory state extraction
# =============================================================================

def species_counts_per_frame(tray_poses):
    """Per-frame species populations from a dense pose tensor.

    ``tray_poses`` is the ``(n_frames, n_particles, 3, n_species)`` tensor of
    ``extract_trajectory_poses``: a particle's coordinates are finite exactly where it exists as
    that species in that frame, NaN elsewhere. The count of species ``s`` in frame ``f`` is the
    number of particles with a finite entry at ``[f, :, :, s]``.

    Returns ``(n_frames, n_species)`` int64. For the standard species order this is the A, B, C
    population trace -- the ground truth the continuous windows are audited against.
    """
    tray_poses = np.asarray(tray_poses)
    present = np.isfinite(tray_poses).any(axis=2)          # (n_frames, n_particles, n_species)
    return present.sum(axis=1).astype(np.int64)


def window_starts(total_frames, window_frames, step_frames):
    """Start indices of the model-length windows, mirroring the experimental slicing.

    Identical stepping to ``experiment_support.read_cell_chunks`` (``range(0, total - window + 1,
    step)``), so the audit slices synthetic continuous videos exactly as the Experiment stage
    slices real recordings. Non-overlapping tiling is ``step_frames == window_frames``.
    """
    if window_frames > total_frames:
        raise ValueError(f"window_frames={window_frames} exceeds total_frames={total_frames}.")
    return np.arange(0, total_frames - window_frames + 1, step_frames, dtype=int)


def window_true_counts(counts, starts, window_frames):
    """The true species populations of each window, in three references.

    ``counts`` is the ``(n_frames, n_species)`` trace; ``starts`` the window start indices.
    Returns a dict of ``(n_windows, n_species)`` float arrays:

        ``start`` -- the population at the window's first frame: the PRIMARY truth. The
                     estimator's count labels are the initial populations of its training
                     windows, so the start state is the quantity it was trained to report;
                     judging it against any other reference manufactures an estimand mismatch.
        ``mean``  -- the within-window mean population (sensitivity reference).
        ``end``   -- the population at the window's last frame (sensitivity reference; with
                     ``start`` it bounds how far the state moved inside one window).
    """
    counts = np.asarray(counts, dtype=float)
    idx = np.asarray(starts, dtype=int)
    mean = np.stack([counts[s:s + window_frames].mean(axis=0) for s in idx])
    return {"mean": mean, "start": counts[idx], "end": counts[idx + window_frames - 1]}


# =============================================================================
# Per-window error and coverage against a truth
# =============================================================================

def quantile_errors(post_q, true_log10):
    """Median error and credible-interval coverage of per-window posterior quantiles.

    ``post_q`` is ``(N, D, 5)`` holding the ``[Q05, Q25, Q50, Q75, Q95]`` posterior quantiles in
    log10; ``true_log10`` broadcasts against ``(N, D)`` -- a single constant theta serves every
    window, a per-window truth (evolved counts) supplies one row per window.

    Returns ``(errors, cover50, cover90)``: the signed ``Q50 - truth`` in log10 (``(N, D)``
    float), and boolean coverage indicators of the 50% (IQR) and 90% (Q05-Q95) intervals. A
    calibrated posterior covers ~50% / ~90%; systematic positional decay of coverage in the
    continuous ensemble only is the horizon signature.
    """
    post_q = np.asarray(post_q, dtype=float)
    truth = np.broadcast_to(np.asarray(true_log10, dtype=float),
                            post_q.shape[:2]).astype(float)
    errors = post_q[:, :, 2] - truth
    cover50 = (truth >= post_q[:, :, 1]) & (truth <= post_q[:, :, 3])
    cover90 = (truth >= post_q[:, :, 0]) & (truth <= post_q[:, :, 4])
    return errors, cover50, cover90


def prior_exceedance(draws, lower, upper):
    """Fraction of unrestricted-flow draws outside the training box, per window.

    ``draws`` is ``(N, S, D)`` in log10; ``lower``/``upper`` the log10 training-prior bounds. A
    draw is outside when ANY coordinate leaves its bound (the joint-box convention of the
    geometric-median analysis). Returns ``(N,)`` float. Named precisely: a bounded-prior Bayesian
    posterior cannot place mass outside its prior -- what this measures is the UNRESTRICTED
    neural flow leaking beyond its training support, which is the deployment gate the
    experimental analysis reads before trusting a window.
    """
    draws = np.asarray(draws, dtype=float)
    lo = np.asarray(lower, dtype=float)
    hi = np.asarray(upper, dtype=float)
    inside = ((draws >= lo) & (draws <= hi)).all(axis=2)   # (N, S)
    return 1.0 - inside.mean(axis=1)


# =============================================================================
# Positional structure: slopes and paired contrasts over trajectories
# =============================================================================

def trajectory_slopes(values):
    """Per-trajectory least-squares slope of a quantity over window position.

    ``values`` is ``(T, W)``: one row per trajectory, one column per window position. Returns
    ``(T,)`` slopes in units of the quantity per window, nan-aware (a trajectory needs at least
    two finite windows; otherwise its slope is NaN). A constant parameter's estimates should have
    slope ~0 within every trajectory; a systematically nonzero slope in the continuous ensemble
    is window-position leakage into a quantity that cannot depend on it.
    """
    values = np.asarray(values, dtype=float)
    n_traj, n_win = values.shape
    x = np.arange(n_win, dtype=float)
    slopes = np.full(n_traj, np.nan)
    for t in range(n_traj):
        y = values[t]
        m = np.isfinite(y)
        if m.sum() < 2:
            continue
        xm = x[m] - x[m].mean()
        ym = y[m] - y[m].mean()
        denom = (xm ** 2).sum()
        slopes[t] = (xm * ym).sum() / denom if denom > 0 else np.nan
    return slopes


def paired_position_effect(cont_values, reset_values):
    """The within-trajectory contrast: continuous windows against that trajectory's own resets.

    ``cont_values`` is ``(T, W, ...)`` (per trajectory, per continuous window position);
    ``reset_values`` is ``(T, R, ...)`` (per trajectory, per reset replicate). The contrast at
    every position is ``continuous - mean(resets)``, formed WITHIN each trajectory so that
    everything the two ensembles share -- the theta, the pinned imaging, the estimator -- cancels
    and only the inherited-state effect remains. Returns ``(T, W, ...)``.
    """
    cont_values = np.asarray(cont_values, dtype=float)
    reset_values = np.asarray(reset_values, dtype=float)
    baseline = np.nanmean(reset_values, axis=1, keepdims=True)    # (T, 1, ...)
    return cont_values - baseline


def bootstrap_mean_ci(values, n_boot=2000, alpha=0.05, rng=None):
    """Percentile-bootstrap confidence interval of the mean over trajectories (axis 0).

    ``values`` is ``(T, ...)``; the mean and its interval are computed by resampling whole
    trajectories with replacement -- the independent unit -- never individual windows, which are
    correlated within a trajectory. Returns ``(mean, lo, hi)`` each of shape ``values.shape[1:]``,
    all nan-aware. With fewer than two trajectories the interval is NaN (no resampling basis).
    """
    values = np.asarray(values, dtype=float)
    mean = np.nanmean(values, axis=0)
    n_traj = values.shape[0]
    if n_traj < 2:
        nanshape = np.full(values.shape[1:], np.nan)
        return mean, nanshape.copy(), nanshape.copy()
    rng = np.random.default_rng(rng)
    stats = np.empty((n_boot,) + values.shape[1:], dtype=float)
    for b in range(n_boot):
        take = rng.integers(0, n_traj, size=n_traj)
        stats[b] = np.nanmean(values[take], axis=0)
    lo = np.nanquantile(stats, alpha / 2.0, axis=0)
    hi = np.nanquantile(stats, 1.0 - alpha / 2.0, axis=0)
    return mean, lo, hi


def positional_summary(values, n_boot=2000, alpha=0.05, rng=None):
    """Mean and bootstrap CI of a per-window quantity at every window position.

    Convenience wrapper: ``values`` is ``(T, W)`` (or ``(T, W, D)``); trajectories are the
    resampled unit. Returns the ``(mean, lo, hi)`` triple from :func:`bootstrap_mean_ci`.
    """
    return bootstrap_mean_ci(values, n_boot=n_boot, alpha=alpha, rng=rng)


def paired_absolute_effect(cont_errors, reset_errors):
    """The primary degradation statistic: absolute-error contrast within each trajectory.

    ``cont_errors`` is ``(T, W, ...)`` SIGNED errors of the continuous windows; ``reset_errors``
    ``(T, R, ...)`` signed errors of the resets. Returns ``(T, W, ...)`` of

        |continuous error| - mean_r |reset error|,

    per trajectory. The signed contrast (:func:`paired_position_effect`) tests BIAS and can read
    zero while accuracy collapses symmetrically; this statistic tests DEGRADATION -- an estimator
    that becomes less accurate on inherited windows raises it regardless of error sign. Signed
    contrasts remain a useful secondary read; verdicts are made on this one.
    """
    cont_abs = np.abs(np.asarray(cont_errors, dtype=float))
    reset_abs = np.abs(np.asarray(reset_errors, dtype=float))
    baseline = np.nanmean(reset_abs, axis=1, keepdims=True)       # (T, 1, ...)
    return cont_abs - baseline


def equivalence_verdict(lo, hi, margin):
    """Three-outcome read of a degradation CI against a prespecified equivalence margin.

    ``(lo, hi)`` is the bootstrap CI of the mean paired ABSOLUTE-error contrast (positive =
    continuous worse); ``margin`` the prespecified, scientifically meaningful degradation bound.

        ``"degraded"``   -- the whole CI lies ABOVE the margin: the loss is larger than the
                            error the method already carries, so it matters in practice.
        ``"equivalent"`` -- the whole CI lies inside ``(-margin, +margin)``: whatever the effect
                            is, it is smaller than that same bound.
        ``"inconclusive"`` -- the CI spans the margin: too wide to support either conclusion.

    STATISTICAL DETECTABILITY IS A SEPARATE QUESTION and is reported alongside, never folded in
    (see :func:`detectable`). An effect can be reliably nonzero and practically negligible at the
    same time -- with enough trajectories any nonzero effect becomes detectable -- and an earlier
    version of this function returned ``"degraded"`` for exactly that case because it tested the
    zero criterion first. That made the two outcomes overlap: a CI could sit entirely above zero
    AND entirely inside the margin, and whichever test ran first won. The margin, not zero, is
    what the audit predeclared as the line that matters, so the margin decides the verdict.
    """
    lo, hi = float(lo), float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return "inconclusive"
    if lo > margin:
        return "degraded"
    if -margin < lo and hi < margin:
        return "equivalent"
    return "inconclusive"


def detectable(lo, hi):
    """Whether the effect is statistically distinguishable from zero -- reported, never a verdict.

    Kept apart from :func:`equivalence_verdict` on purpose: 'we can measure it' and 'it matters'
    are different claims, and conflating them is what makes a large study call every tiny effect
    a degradation. A quantity can be ``detectable`` and ``"equivalent"`` at once; that pairing is
    the honest description of a real but practically negligible loss.
    """
    lo, hi = float(lo), float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi)):
        return False
    return lo > 0.0 or hi < 0.0


def stratify_by_true_value(true_values, lower, upper, n_strata=3):
    """Assign each element to a stratum of the PRIOR RANGE, in the prior's own (log) units.

    ``true_values`` are the true parameter values (log10, as drawn); ``(lower, upper)`` the prior
    bounds for that parameter. Returns ``(labels, edges)`` with ``labels`` an integer array in
    ``[0, n_strata)`` (``-1`` where the value is outside the bounds or non-finite) and ``edges``
    the ``n_strata + 1`` cut points.

    The strata are FIXED thirds of the prior range, NOT data-dependent quantiles: the geometry is
    prespecified by the prior alone, so the same cuts apply to any set of draws from it (the audit
    cohort, the reset arm, the held-out recovery set) and no stratum boundary is chosen after
    seeing an effect. Because the prior is uniform in log10, equal ranges carry equal expected
    mass. The top stratum is closed on the right so the prior's upper endpoint is not discarded.
    """
    values = np.asarray(true_values, dtype=float)
    edges = np.linspace(float(lower), float(upper), int(n_strata) + 1)
    labels = np.full(values.shape, -1, dtype=int)
    for s in range(int(n_strata)):
        hi_closed = (s == int(n_strata) - 1)
        inside = (values >= edges[s]) & ((values <= edges[s + 1]) if hi_closed
                                         else (values < edges[s + 1]))
        labels[inside & np.isfinite(values)] = s
    return labels, edges


def stratified_margin_table(contrast, cohort_labels, recovery_abs_error, recovery_labels,
                            n_strata=3, n_boot=2000, alpha=0.05, rng=None):
    """Per-stratum degradation verdicts against per-stratum, self-calibrated margins.

    A single prior-averaged margin can mislead when the estimator's error depends on the true
    value (it does: sparse counts and slow rates are intrinsically harder than dense counts and
    fast rates). Judging a prior-averaged contrast against a prior-averaged margin can therefore
    hide a degradation in one regime behind easy performance in another, or flag one that is
    weightless where the parameter is barely resolvable at all.

    This recomputes both sides of the comparison INSIDE each stratum: the margin is the
    estimator's own held-out absolute error for true values in that stratum
    (``recovery_abs_error`` at ``recovery_labels``), and the statistic is the paired contrast for
    cohort trajectories whose true value falls in the same stratum (``contrast`` at
    ``cohort_labels``, one value per trajectory, already reduced to the window position of
    interest). Both populations are draws from the same prior, so the yardstick and the
    measurement average over the same regime.

    Returns a list of per-stratum dicts: ``stratum``, ``n_cohort``, ``n_recovery``, ``margin``,
    ``mean``, ``ci_lo``, ``ci_hi``, ``ratio`` (mean / margin) and ``verdict``. A stratum with no
    cohort trajectories yields NaNs and ``"inconclusive"`` rather than being silently dropped --
    an empty stratum is a coverage statement about the cohort, not an absence of effect.
    """
    contrast = np.asarray(contrast, dtype=float)
    cohort_labels = np.asarray(cohort_labels, dtype=int)
    recovery_abs_error = np.asarray(recovery_abs_error, dtype=float)
    recovery_labels = np.asarray(recovery_labels, dtype=int)
    rows = []
    for s in range(int(n_strata)):
        in_rec = recovery_labels == s
        in_coh = cohort_labels == s
        margin = float(np.nanmean(recovery_abs_error[in_rec])) if in_rec.any() else np.nan
        values = contrast[in_coh]
        values = values[np.isfinite(values)]
        if values.size and np.isfinite(margin):
            # bootstrap over the trajectories IN this stratum: resampling the pooled cohort would
            # mix strata back together and understate the per-stratum uncertainty.
            b_mean, b_lo, b_hi = bootstrap_mean_ci(values[:, None], n_boot=n_boot,
                                                   alpha=alpha, rng=rng)
            mean, lo, hi = float(b_mean[0]), float(b_lo[0]), float(b_hi[0])
            verdict = equivalence_verdict(lo, hi, margin)
            ratio = mean / margin if margin > 0 else np.nan
        else:
            mean = lo = hi = ratio = np.nan
            verdict = "inconclusive"
        rows.append({"stratum": s, "n_cohort": int(values.size), "n_recovery": int(in_rec.sum()),
                     "margin": margin, "mean": mean, "ci_lo": float(lo), "ci_hi": float(hi),
                     "ratio": ratio, "verdict": verdict})
    return rows

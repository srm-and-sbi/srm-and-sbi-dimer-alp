"""Figures for the posterior-calibration diagnostic (both DIMER workflows).

Renders the four calibration diagnostics computed by ``posterior_calibration`` into
headless matplotlib figures for the report: SBC rank histograms (per marginal), the
expected-coverage curve, the TARP ECP curve, and a stratified at-a-glance panel that
localizes where calibration degrades across a target-theta dimension. Workflow-agnostic:
it consumes only the result objects, which carry parameter keys but no workflow tag.

Pure rendering: builds ``matplotlib.figure.Figure`` objects directly (no pyplot, no
global state), so it is import-light and headless-safe. Colors follow a
colorblind-safe scheme (blue primary, orange secondary, status green/red), each defined
once as a role.
"""
from __future__ import annotations

import math
from typing import Optional, Sequence

import numpy as np
from matplotlib.figure import Figure

# Roles (colorblind-safe; blue primary, orange secondary, status green/red, gray ink).
_BLUE = "#2a78d6"
_ORANGE = "#eb6834"
_GOOD = "#0ca30c"
_CRIT = "#d03b3b"
_INK = "#52514e"
_GRID = "#e1e0d9"
_REFERENCE = "#898781"


def _style_axes(ax) -> None:
    """Recessive grid + spines, consistent with the report's clean look."""
    ax.grid(True, color=_GRID, linewidth=0.6, zorder=0)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(_REFERENCE)
    ax.tick_params(colors=_INK, labelsize=8)


def figure_sbc_ranks(sbc, *, bonferroni: bool = True) -> Figure:
    """Grid of per-marginal SBC rank histograms (flat = calibrated).

    Each panel shows one target-parameter's rank histogram with the uniform
    expectation line and a shaded 99% band; the title carries the KS p-value and, when
    it falls below the (Bonferroni-corrected) threshold, is drawn in the alert color.
    A ``U``-shape means over-confidence (ranks pile at the edges), a ``^``-shape
    over-dispersion, a slope a bias.
    """
    keys = list(sbc.theta_keys)
    d = len(keys)
    edges = np.asarray(sbc.rank_hist_edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]
    n_videos = int(np.asarray(sbc.rank_hist)[0].sum()) if d else 0
    n_bins = len(centers)
    expected = n_videos / n_bins if n_bins else 0.0
    # 99% band for a uniform multinomial bin count (normal approx).
    p = 1.0 / n_bins if n_bins else 0.0
    band = 2.576 * math.sqrt(n_videos * p * (1 - p)) if n_videos else 0.0
    thresh = (0.05 / d) if (bonferroni and d) else 0.05

    ncols = min(3, d) or 1
    nrows = math.ceil(d / ncols)
    fig = Figure(figsize=(4.2 * ncols, 2.6 * nrows), dpi=200)
    for i, key in enumerate(keys):
        ax = fig.add_subplot(nrows, ncols, i + 1)
        _style_axes(ax)
        ax.bar(centers, np.asarray(sbc.rank_hist)[i], width=width * 0.92,
               color=_BLUE, zorder=3)
        ax.axhline(expected, color=_REFERENCE, linewidth=1.0, zorder=4)
        if band:
            ax.axhspan(expected - band, expected + band, color=_REFERENCE, alpha=0.15, zorder=1)
        ksp = float(sbc.ks_pvals[i])
        flagged = ksp < thresh
        ax.set_title(f"{key}\nKS p = {ksp:.3f}", fontsize=8,
                     color=(_CRIT if flagged else _INK))
        ax.set_xlim(edges[0], edges[-1])
        ax.set_ylim(bottom=0)
        if i % ncols == 0:
            ax.set_ylabel("count", fontsize=8, color=_INK)
    fig.suptitle(f"SBC rank uniformity  (N = {n_videos},  L = {sbc.num_posterior_samples})",
                 fontsize=10, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    return fig


def figure_coverage(coverage) -> Figure:
    """Expected-coverage curve: empirical vs nominal credibility (diagonal = calibrated)."""
    fig = Figure(figsize=(4.6, 4.4), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    _style_axes(ax)
    ax.plot([0, 1], [0, 1], color=_REFERENCE, linewidth=1.2, linestyle="--", zorder=2)
    ax.plot(coverage.levels, coverage.empirical, color=_BLUE, linewidth=2.0,
            marker="o", markersize=4, zorder=3)
    ax.set_xlabel("nominal credibility level", fontsize=9, color=_INK)
    ax.set_ylabel("empirical coverage", fontsize=9, color=_INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.set_title(f"Expected coverage  (KS p = {coverage.ks_pval:.3f})\n"
                 "above diagonal = conservative;  below = overconfident",
                 fontsize=9, color=_INK)
    fig.tight_layout()
    return fig


def figure_tarp(tarp) -> Figure:
    """TARP ECP curve: expected coverage probability vs credibility (diagonal = ideal)."""
    fig = Figure(figsize=(4.6, 4.4), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    _style_axes(ax)
    ax.plot([0, 1], [0, 1], color=_REFERENCE, linewidth=1.2, linestyle="--", zorder=2)
    ax.plot(tarp.alpha, tarp.ecp, color=_ORANGE, linewidth=2.0, zorder=3)
    ax.set_xlabel("credibility level", fontsize=9, color=_INK)
    ax.set_ylabel("expected coverage probability", fontsize=9, color=_INK)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    sign = "over-dispersed" if tarp.atc > 0 else "under-dispersed"
    ax.set_title(f"TARP  (ATC = {tarp.atc:+.3f}, KS p = {tarp.ks_pval:.3f})\n"
                 f"ATC != 0 => {sign}", fontsize=9, color=_INK)
    fig.tight_layout()
    return fig


def figure_pairwise(matrix, theta_keys, *, threshold: float = 0.05) -> Optional[Figure]:
    """Heat map of one- and two-dimensional marginal calibration (``|ATC|``).

    The diagonal is each parameter alone; off-diagonal ``[i, j]`` is the joint calibration
    of the pair. Marginal (one-dimensional) calibration is necessary but not sufficient, so
    an off-diagonal cell that is worse than both of its diagonals localizes a
    *dependence* error -- the pair's correlation is misstated even though each parameter on
    its own looks fine, which no per-parameter measure can reveal.
    """
    if matrix is None:
        return None
    m = np.asarray(matrix, dtype=float)
    d = m.shape[0]
    keys = [str(k) for k in theta_keys]

    fig = Figure(figsize=(1.05 * d + 3.0, 0.95 * d + 2.4), dpi=200)
    ax = fig.add_subplot(1, 1, 1)
    # Scale to the data but always show at least the practical threshold, so a clean
    # matrix reads as uniformly pale rather than being stretched into false contrast.
    vmax = max(float(np.nanmax(m)), 2.0 * threshold)
    im = ax.imshow(m, cmap="magma_r", vmin=0.0, vmax=vmax)
    ax.set_xticks(range(d)); ax.set_yticks(range(d))
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=7, color=_INK)
    ax.set_yticklabels(keys, fontsize=7, color=_INK)
    for i in range(d):
        for j in range(d):
            v = m[i, j]
            ax.text(j, i, f"{v:.02f}".lstrip("0"), ha="center", va="center", fontsize=6.5,
                    color=("white" if v > 0.55 * vmax else _INK))
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.set_label(f"|ATC|  (0 = calibrated; practical threshold {threshold:g})",
                   fontsize=8, color=_INK)
    cbar.ax.tick_params(labelsize=7, colors=_INK)
    ax.set_title("Calibration of the 1-D and 2-D marginals\n"
                 "diagonal = each parameter alone;  off-diagonal = the pair jointly",
                 fontsize=10, color=_INK)
    fig.tight_layout()
    return fig


def figure_stratified(result, *, test: str = "coverage") -> Optional[Figure]:
    """At-a-glance stratified panel: a summary statistic across each theta-dim's bins.

    For the chosen ``test`` (``sbc`` -> min-marginal KS p; ``coverage`` -> log-prob KS p;
    ``tarp`` -> |ATC|; ``lc2st`` -> reject fraction), plots the statistic per stratum,
    one small-multiple per stratifying dimension. Bins run along each parameter's
    INFERRED value (per-video posterior median, a function of the observation), so a
    flagged bin is a subregion the posterior is genuinely miscalibrated for -- e.g. "the
    videos it infers as low-count." Returns ``None`` when no strata were computed.
    """
    strata = list(result.strata)
    if not strata:
        return None

    # Every panel plots an EFFECT SIZE, never a p-value: at production N the p-values are
    # ~0 in every stratum, so bars drawn from them have zero height and the panel renders
    # blank apart from its reference line. Effect sizes stay on a comparable, readable
    # scale and are what actually ranks the strata.
    #
    # The per-stratum value comes from the kernel's shared accessor (which also feeds the
    # report's stratified digest), so a plotted bar and its tabulated summary always
    # describe the same quantity. For SBC that is the STRATIFYING parameter's OWN rank
    # statistic, not the worst across all marginals: the panel answers "is this parameter
    # calibrated across its own range?", so a max over marginals would paint every panel
    # with whichever parameter happens to be worst overall and say nothing about the one
    # named. The remaining measures are joint by construction.
    def value(s, dim):
        from .posterior_calibration import stratum_effect
        return stratum_effect(s, test, dim)

    keys = []
    for s in strata:
        if s.key not in keys:
            keys.append(s.key)
    ncols = min(3, len(keys)) or 1
    nrows = math.ceil(len(keys) / ncols)
    fig = Figure(figsize=(4.4 * ncols, 2.8 * nrows), dpi=200)
    # One practical threshold per test; larger is worse for every effect size here.
    ref = {"sbc": 0.05, "coverage": 0.05, "tarp": 0.05, "lc2st": 0.10}.get(test)

    # Map each stratifying parameter to its column in the theta vector, so the SBC panel
    # can show that parameter's own statistic.
    theta_keys = list(result.theta_keys)
    for j, key in enumerate(keys):
        ax = fig.add_subplot(nrows, ncols, j + 1)
        _style_axes(ax)
        dim = theta_keys.index(key) if key in theta_keys else None
        rows = [s for s in strata if s.key == key]
        centers = [0.5 * (s.lo + s.hi) for s in rows]
        vals = [value(s, dim) for s in rows]
        colors = [_REFERENCE if v is None else (_CRIT if v > ref else _GOOD) for v in vals]
        ax.bar(range(len(rows)), [0 if v is None else v for v in vals],
               color=colors, zorder=3)
        if ref is not None:
            ax.axhline(ref, color=_ORANGE, linewidth=1.0, linestyle="--", zorder=4)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([f"{c:.3g}" for c in centers], rotation=45, fontsize=7)
        ax.set_title(key, fontsize=8, color=_INK)
        ax.set_xlabel("inferred value (bin center)", fontsize=7, color=_INK)
    label = {"sbc": "that parameter's own KS D", "coverage": "max coverage gap",
             "tarp": "|ATC|", "lc2st": "reject frac"}.get(test, test)
    fig.suptitle(f"Stratified {test}: {label} across bins of the inferred value  "
                 f"(dashed = {ref:g} practical threshold; lower is better)",
                 fontsize=10, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig

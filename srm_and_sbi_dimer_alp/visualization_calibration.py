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

    def value(s):
        if test == "sbc" and s.sbc is not None:
            return float(np.min(s.sbc.ks_pvals))
        if test == "coverage" and s.coverage is not None:
            return s.coverage.ks_pval
        if test == "tarp" and s.tarp is not None:
            return abs(s.tarp.atc)
        if test == "lc2st" and s.lc2st is not None:
            return s.lc2st.reject_fraction
        return None

    keys = []
    for s in strata:
        if s.key not in keys:
            keys.append(s.key)
    ncols = min(3, len(keys)) or 1
    nrows = math.ceil(len(keys) / ncols)
    fig = Figure(figsize=(4.4 * ncols, 2.8 * nrows), dpi=200)
    # Reference lines / interpretation per test.
    ref = {"sbc": 0.05, "coverage": 0.05, "tarp": 0.05, "lc2st": 0.10}.get(test)
    small_is_bad = test in ("sbc", "coverage")

    for j, key in enumerate(keys):
        ax = fig.add_subplot(nrows, ncols, j + 1)
        _style_axes(ax)
        rows = [s for s in strata if s.key == key]
        centers = [0.5 * (s.lo + s.hi) for s in rows]
        vals = [value(s) for s in rows]
        colors = []
        for v in vals:
            if v is None:
                colors.append(_REFERENCE)
            elif small_is_bad:
                colors.append(_CRIT if v < ref else _GOOD)
            else:
                colors.append(_CRIT if v > ref else _GOOD)
        ax.bar(range(len(rows)), [0 if v is None else v for v in vals],
               color=colors, zorder=3)
        if ref is not None:
            ax.axhline(ref, color=_ORANGE, linewidth=1.0, linestyle="--", zorder=4)
        ax.set_xticks(range(len(rows)))
        ax.set_xticklabels([f"{c:.3g}" for c in centers], rotation=45, fontsize=7)
        ax.set_title(key, fontsize=8, color=_INK)
        ax.set_xlabel("inferred value (bin center)", fontsize=7, color=_INK)
    label = {"sbc": "min KS p", "coverage": "KS p", "tarp": "|ATC|",
             "lc2st": "reject frac"}.get(test, test)
    fig.suptitle(f"Stratified {test}: {label} across bins of the inferred value",
                 fontsize=10, color=_INK)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig

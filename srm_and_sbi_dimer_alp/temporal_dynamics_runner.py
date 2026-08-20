"""Shared engine for the temporal-dynamics analysis (biology and detector workflows).

Both workflows ask the same question of a different parameter vector: the Experiment stage
estimates parameters independently in every non-overlapping window of every experimental
recording, so stacking the windows along time asks whether the inferred value holds still across
the recording. The numerics are the workflow-agnostic ``temporal_dynamics`` kernel; this runner
feeds it, draws the figures, and writes the report.

WHY BOTH WORKFLOWS MATTER HERE, AND WHY NEITHER SUFFICES ALONE. A trend in an inferred parameter
is either real dynamics or an acquisition confound, and each workflow is structurally blind to its
own confound: biology holds imaging FIXED, so it cannot see imaging drift; the detector
marginalizes the reaction-diffusion block, so it cannot see biological drift. Running the same
analysis on both, over the SAME recordings, is therefore not a symmetry exercise -- it is the only
way either result can be read. The two share the recordings exactly: the experimental path pattern
carries no workflow qualifier, so cell index ``c`` is the same acquisition in both.

WHAT IS PLOTTED, AND WHAT IS MEASURED, ARE DELIBERATELY SEPARATE. The drift statistics are fit per
cell and then aggregated, so they do not depend on the choice of central estimate; the central
estimate governs only what the figures display.

THE CENTRAL-ESTIMATE VOCABULARY. Every figure states which of four estimates it draws, and the name
says which axis was aggregated and how:

    mean-window       mean value vector, aggregated across cells for a given chunk    -> timeseries
    sgm-window        realized value vector, aggregated across cells for a given chunk -> timeseries
    mean-trajectory   mean value vector, aggregated across chunks and cells            -> one vector
    sgm-trajectory    realized value vector, aggregated across chunks and cells        -> one vector

``--central`` selects the FAMILY (``sgm`` or ``mean``) and the pairing is enforced: the timeseries
uses that family's ``*-window`` estimate and the horizontal summary line its ``*-trajectory``
counterpart, so one figure never mixes a mean with a medoid. A "mean" estimate aggregates each
parameter independently, so its coordinates need not have co-occurred in any recording; a "realized"
estimate is an actual (cell, chunk) window selected as the exact medoid, so its coordinates did.

The per-workflow differences -- parameter table and keys, prior box, alias-qualified paths, display
names, unit labels, and the external reference values -- are resolved once in
:func:`_temporal_dynamics_spec`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: build and save figures without a display
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, FuncFormatter, MaxNLocator

from . import temporal_dynamics as tdk
from .parameterization import PARAMETERS, RunTiming
from .workflow import parameter_keys, parameter_table

# Conditions are named scientifically wherever a reader sees them; the tokens below survive only
# as the stored ``kinds`` field of the Experiment output and the recording filenames on disk.
# Condition naming (stored token <-> scientific name) has ONE definition, in experiment_support.
from .experiment_support import CONDITION_DISPLAY  # noqa: F401  (re-exported for figures)

# Bins for the pooled posterior density, spread uniformly over the prior's decades. The pooled set
# holds hundreds of thousands of draws, so the bin count is not sample-limited; it is chosen for
# legibility and held fixed so the two conditions and every parameter share one resolution.
POOLED_BINS = 64
CONDITION_COLOR = {"ALP": "tab:blue", "BET": "tab:orange"}
REFERENCE_COLOR = "tab:grey"
SGM_COLOR = "gold"

# =============================================================================
# External reference values
# =============================================================================
# Each entry gives values in the parameter's ABSOLUTE unit. ``applies_to`` names the conditions the
# reference may legitimately be compared against -- ``None`` meaning all of them -- because several
# references are valid only for the monomer control. ``upper_biased`` records a value that is known
# to overstate the quantity, so the figure can show it as a bound rather than a target.

_REFERENCE_BIOLOGY = {
    "rate_dissociation": {
        "unit": "1/s",
        "estimates": [(1.0 / 1.30, 0.05 / 1.30 ** 2, "InlB T-T  (1/tau, tau=1.30 s)"),
                      (1.0 / 0.80, 0.03 / 0.80 ** 2, "InlB H-T  (1/tau, tau=0.80 s)")],
        "applies_to": ["MET-INLB"],
        "note": "kappa_OFF = 1/tau per FRET variant: 1/1.30=0.77, 1/0.80=1.25 1/s "
                "(tau = 1.30+/-0.05, 0.80+/-0.03 s; SD propagated as d(tau)/tau^2); band = "
                "value+/-SD; mean line = mean of the two RATES = 1.01. The study measured "
                "InlB-activated MET, so the reference compares to MET-INLB only. "
                "Li et al. 2026, Fig. 2C",
    },
    "diffusivity_alp": {
        "unit": "um^2/s",
        "estimates": [(0.109, 0.068, "donor-only T-T"), (0.093, 0.053, "donor-only H-T")],
        "applies_to": None,
        "note": "monomer diffusion, donor-only segments: 0.109+/-0.068 (T-T), 0.093+/-0.053 "
                "(H-T) um^2/s; band = value+/-SD; mean = 0.10. Monomer diffusion is "
                "ligand-independent, so the reference applies to both conditions. "
                "Li et al. 2026, Sec. 2.3",
    },
    "relative_diffusivity_bet": {
        "unit": "ratio",
        "estimates": [(0.066 / 0.109, None, "T-T  (D_dimer/D_mono)"),
                      (0.056 / 0.093, None, "H-T  (D_dimer/D_mono)")],
        "applies_to": ["MET-INLB"],
        "note": "R_B = D_dimer/D_monomer per FRET variant: 0.066/0.109=0.61 (T-T), "
                "0.056/0.093=0.60 (H-T); mean = 0.60. Derived from the InlB-activated "
                "measurement, so it compares to MET-INLB. Li et al. 2026, Sec. 2.3",
    },
}

_REFERENCE_DETECTOR = {
    "mu_r": {
        "unit": "px", "estimates": [(1.36, None, "Fab localization fit")],
        "applies_to": ["MET-FAB"],
        "note": "median PSF width from a location-zero log-normal fit of the ThunderSTORM "
                "sigma[nm] column (accession S-BSST712), converted by sqrt(2)*sigma/158 nm to "
                "the model's pixel convention: 1.36 (Fab). The InlB value 1.47 is "
                "dimer-broadened -- two labels in one diffraction-limited spot -- so it is not a "
                "reference for the per-emitter PSF the model infers. DETECTOR_WORKFLOW.md 6.2/6.5",
    },
    "sigma_r": {
        "unit": "log-spread", "estimates": [(0.15, None, "fit-corrected")],
        "upper_biased": (0.37, "Fab fitted (errors-in-variables inflated)"),
        "applies_to": ["MET-FAB"],
        "note": "the fitted log-spread 0.37 (Fab; InlB 0.42) is UPPER-BIASED: each per-spot width "
                "is itself a noisy fit, and the variance of noisy estimates is the true variance "
                "plus the fitting-error variance. The value a calibration is expected to recover "
                "is the fit-corrected ~0.15, located by the fixed-imaging posterior-predictive "
                "histogram check. The fitted value is drawn as an upper bound, not a target. "
                "DETECTOR_WORKFLOW.md 6.5 caveat 1",
    },
    "mu_pc": {
        "unit": "photons", "estimates": [(386.0, None, "Fab per-detection")],
        "applies_to": ["MET-FAB"],
        "note": "median brightness from a location-zero log-normal fit of the ThunderSTORM "
                "intensity[photon] column: 386 photons (Fab). MET-INLB's 690 is a PER-DETECTION "
                "sum -- an activated dimer carries two labels within one spot, which the "
                "single-emitter fit reports as one detection whose photons add -- so it is not a "
                "reference for the per-emitter brightness the model infers, and the ratio is a "
                "lower bound because 23.7% of InlB localizations pile up at the 1225-photon "
                "acceptance ceiling. DETECTOR_WORKFLOW.md 6.4/6.5 caveat 3",
    },
    "sigma_pc": {
        "unit": "log-spread", "estimates": [(0.5, None, "fit-corrected")],
        "upper_biased": (0.61, "Fab fitted (errors-in-variables inflated)"),
        "applies_to": ["MET-FAB"],
        "note": "same errors-in-variables inflation as sigma_r: fitted 0.61 (Fab; InlB 0.55) is "
                "upper-biased; the fit-corrected operating value is ~0.5. "
                "DETECTOR_WORKFLOW.md 6.5 caveat 1",
    },
    "lambda_rate": {
        "unit": "1/s", "estimates": [(5.0, None, "flicker correlation time")],
        "applies_to": None,
        "note": "derived from the flicker correlation time of the track intensity[photon] series "
                "(log-detrended, pooled autocorrelation, lag-0 fit-noise drop excluded): "
                "tau_corr ~ 0.13 s maps through the monotone tau_corr(lambda_rate) of the model's "
                "own brightness chain to lambda_rate ~ 5. A photophysical quantity, "
                "condition-independent, so it applies to both. DETECTOR_WORKFLOW.md 6.3",
    },
    # prob_photo_bleach deliberately absent: no public anchor exists (DETECTOR_WORKFLOW.md 6.2).
}

_DISPLAY_BIOLOGY = {
    "count_alp": "Initial monomer count", "count_bet": "Initial mobile-dimer count",
    "count_chi": "Initial immobile-dimer count", "diffusivity_alp": "Monomer diffusivity",
    "relative_diffusivity_bet": "Rel. mobile-dimer diffusivity",
    "relative_diffusivity_chi": "Rel. immobile-dimer diffusivity",
    "relative_rate_dimerization": "Rel. dimerization rate",
    "rate_dissociation": "Dissociation rate", "rate_immobility": "Immobilization rate",
    "rate_mobility": "Mobilization rate",
}
_DISPLAY_DETECTOR = {
    "mu_r": "Median PSF width", "sigma_r": "PSF-width log-spread",
    "mu_pc": "Median emitter brightness", "sigma_pc": "Brightness log-spread",
    "prob_photo_bleach": "Photobleaching probability", "lambda_rate": "Flicker rate",
}
# Axis unit per parameter KEY, for tables that carry no UNIT string (the detector table sets
# UNIT=None on every row because its parameters are dimensionless shapes, pixel widths, photon
# counts, probabilities, and rates that the table documents in prose instead).
_UNIT_BY_KEY = {
    "mu_r": "px", "sigma_r": "log-spread", "mu_pc": "photons", "sigma_pc": "log-spread",
    "prob_photo_bleach": "probability", "lambda_rate": "1/s",
}
_UNIT_SHORT = {
    "Count": "count", "Square Micrometer Per Second": "$\\mu$m$^2$/s",
    "Dimensionless": "ratio", "Count Per Second": "1/s",
    "Pixel": "px", "Photon": "photons", "Probability": "probability",
}


@dataclass(frozen=True)
class _TemporalSpec:
    """Everything about this analysis that differs between the two workflows."""
    keys: list
    table: list
    prior_low: np.ndarray
    prior_high: np.ndarray
    display: dict
    references: dict
    paths: object
    alias: str
    timing_label: str
    npz_path: object
    recovery_npz: object
    fig_dir: object
    tag: str
    reference_source: str = ""
    reference_short: str = "reference"


def _temporal_dynamics_spec(cfg, args) -> _TemporalSpec:
    """Resolve the workflow-specific half of the analysis."""
    timing = RunTiming(total_time_seconds=args.total_time_seconds,
                       frames=PARAMETERS.simulation.timing)
    data_bank_root = PARAMETERS.machine.data_bank_root
    paths = cfg.paths
    out_dir = paths.experiment_recovery_dir(data_bank_root, timing.label)
    rec_dir = paths.map_recovery_dir(data_bank_root, timing.label)
    table = parameter_table(cfg)
    detector = cfg.tag == "detector"
    return _TemporalSpec(
        keys=parameter_keys(cfg), table=table,
        prior_low=np.array(cfg.param_module.theta_lower_bound(), dtype=float),
        prior_high=np.array(cfg.param_module.theta_upper_bound(), dtype=float),
        display=(_DISPLAY_DETECTOR if detector else _DISPLAY_BIOLOGY),
        references=(_REFERENCE_DETECTOR if detector else _REFERENCE_BIOLOGY),
        paths=paths, alias=paths.project_alias, timing_label=timing.label,
        npz_path=out_dir / (out_dir.name + ".npz"),
        recovery_npz=rec_dir / (rec_dir.name + ".npz"),
        fig_dir=out_dir / "temporal_dynamics", tag=cfg.tag,
        reference_source=("ThunderSTORM localization fits on the same public recordings "
                          "(DETECTOR_WORKFLOW.md 6.2/6.3/6.5)" if detector else
                          "Li et al., Small 2026, 22, e07115"),
        reference_short=("ThunderSTORM" if detector else "Li et al. 2026"),
    )


def _reference_band(ref):
    """Mean, [lo, hi] band, and the individual reported values of an external reference."""
    vals = [v for (v, _sd, _lab) in ref["estimates"]]
    sds = [(sd or 0.0) for (_v, sd, _lab) in ref["estimates"]]
    return (float(np.mean(vals)),
            float(min(v - s for v, s in zip(vals, sds))),
            float(max(v + s for v, s in zip(vals, sds))), vals)


def _apply_reference(ax, ref, source_short, orient="h"):
    """Draw an external reference: its band, its individual reported values, and its mean.

    The label carries the mean, the numeric bounds, the unit, the source, and the conditions the
    reference applies to -- everything a reader needs to judge the comparison without leaving the
    figure. Returns ``(lo, hi)`` so the caller can include the reference in the axis limits.

    ``orient`` selects the axis the reference lives on: ``"h"`` for the figures where the parameter
    value is on y (the time courses), ``"v"`` where it is on x (the pooled density). One definition
    serves both so a reference cannot be styled or labeled differently between them.
    """
    span = ax.axhspan if orient == "h" else ax.axvspan
    line_at = ax.axhline if orient == "h" else ax.axvline
    r_mean, r_lo, r_hi, r_vals = _reference_band(ref)
    span(r_lo, r_hi, color=REFERENCE_COLOR, alpha=0.25, linewidth=0)
    for v in r_vals:
        line_at(v, color=REFERENCE_COLOR, linestyle=":", linewidth=1.1, alpha=0.75)
    applies = ref.get("applies_to")
    scope = "both conditions" if applies is None else f"{', '.join(applies)} only"
    line_at(r_mean, color=REFERENCE_COLOR, linestyle="-", linewidth=2.1,
            label=(f"{source_short}: {r_mean:.3g} [{r_lo:.3g}–{r_hi:.3g}] {ref['unit']} "
                   f"({scope})"))
    if ref.get("upper_biased") is not None:
        ub, ub_lab = ref["upper_biased"]
        line_at(ub, color=REFERENCE_COLOR, linestyle="--", linewidth=1.6, alpha=0.9,
                label=f"upper bound {ub:.3g} — {ub_lab}, not a target")
        r_hi = max(r_hi, ub)
    return r_lo, r_hi


# =============================================================================
# Figures
# =============================================================================

def _figure_trajectory(spec, p_index, key, abs_grid, series, line, family, picks_window,
                       picks_traj, edges, kinds, drift, recovery):
    """The central-trajectory figure: one estimate family, drawn as steps over each window.

    Deliberately sparse. Exactly one family is drawn -- the ``<family>-window`` timeseries as a step
    held across each window, and its paired ``<family>-trajectory`` summary as a horizontal line --
    never a mixture, with the individual recordings faint behind them. Numbers are confined to the
    legend and the report: annotating drift statistics on the axes added clutter without helping
    anyone read the data.
    """
    para = spec.table[p_index]
    name = spec.display.get(key, key)
    ref = spec.references.get(key)
    fig, ax = plt.subplots(figsize=(7.0, 5.0))
    hi, lo = -np.inf, np.inf

    for e, kind in enumerate(kinds):
        color = CONDITION_COLOR.get(kind, f"C{e}")
        data = abs_grid[e, :, :, p_index]
        for c in range(data.shape[0]):
            ax.step(edges, _step(data[c]), where="post", color=color, alpha=0.11, linewidth=0.8)
        fin = data[np.isfinite(data)]
        if fin.size:                              # robust limits: outlying traces clip
            hi = max(hi, float(np.nanpercentile(fin, 92)))
            lo = min(lo, float(np.nanpercentile(fin, 8)))

    if ref is not None:
        r_lo, r_hi = _apply_reference(ax, ref, spec.reference_short)
        hi, lo = max(hi, r_hi), min(lo, r_lo)

    applies = None if ref is None else ref.get("applies_to")
    r_mean = None if ref is None else _reference_band(ref)[0]
    for e, kind in enumerate(kinds):
        color = CONDITION_COLOR.get(kind, f"C{e}")
        disp = CONDITION_DISPLAY.get(kind, kind)
        ax.step(edges, _step(series[e, :, p_index]), where="post", color=color,
                linewidth=2.6, label=f"{disp} {family}-window")
        lv = line[e, p_index]
        lab = f"{disp} {family}-trajectory = {lv:.3g}"
        if r_mean and (applies is None or disp in applies) and np.isfinite(lv):
            lab += f"  ({lv / r_mean:.2g}x reference)"
        ax.axhline(lv, color=color, linestyle="--", linewidth=1.4, alpha=0.9, label=lab)

    _finish_axes(ax, edges, para, spec, lo, hi)
    title = f"{name} ({para['LABEL']}) over the recording"
    if recovery:
        # Both nested tolerances, stated as the multiplicative ranges they mean rather than in dex.
        title += "\nheld-out recovery:  " + "  ·  ".join(
            f"{100 * frac[p_index]:.0f}% within {tdk.band_label(b)}" for b, frac in recovery)
    ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=7.5, framealpha=0.9, loc="best")
    fig.tight_layout()
    return fig


def _figure_posterior(spec, p_index, key, within, series, family, edges, kinds):
    """Within-window posterior interval over the recording -- one panel, one quantity, as steps.

    At each chunk, the median across cells of that window's stored posterior interval: how uncertain
    a TYPICAL SINGLE window's estimate is. Drawn as bands held across each window, matching the
    trajectory figure, with the ``<family>-window`` series on top as the anchor.

    This is an interval WIDTH built from the stored five-quantile record -- not a posterior density,
    and it does not approximate one. The pooled density is a separate figure, drawn from the raw
    per-window draws when the Experiment stage was run with --dump-posterior-samples.
    """
    para = spec.table[p_index]
    name = spec.display.get(key, key)
    ref = spec.references.get(key)
    fig, ax = plt.subplots(figsize=(7.0, 4.9))
    hi, lo = -np.inf, np.inf
    for e, kind in enumerate(kinds):
        color = CONDITION_COLOR.get(kind, f"C{e}")
        disp = CONDITION_DISPLAY.get(kind, kind)
        q = 10.0 ** within[e, :, p_index, :]
        ax.fill_between(edges, _step(q[:, 0]), _step(q[:, 4]), step="post", color=color,
                        alpha=0.15, linewidth=0, label=f"{disp} posterior 5–95% (typical window)")
        ax.fill_between(edges, _step(q[:, 1]), _step(q[:, 3]), step="post", color=color,
                        alpha=0.30, linewidth=0, label=f"{disp} posterior 25–75%")
        ax.step(edges, _step(series[e, :, p_index]), where="post", color=color, linewidth=2.2,
                label=f"{disp} {family}-window")
        fin = q[np.isfinite(q)]
        if fin.size:
            hi = max(hi, float(np.nanpercentile(fin, 98)))
            lo = min(lo, float(np.nanpercentile(fin, 2)))
    if ref is not None:
        r_lo, r_hi = _apply_reference(ax, ref, spec.reference_short)
        hi, lo = max(hi, r_hi), min(lo, r_lo)
    _finish_axes(ax, edges, para, spec, lo, hi)
    ax.set_title(f"Within-window posterior interval\n{name} ({para['LABEL']})", fontsize=10.5)
    ax.legend(fontsize=7, framealpha=0.9, loc="best")
    fig.tight_layout()
    return fig


def _figure_pooled(spec, p_index, key, pooled, line, family, kinds, scale="log"):
    """Pooled posterior density over the whole recording -- the classic histogram, time collapsed.

    Every window's draws for a condition are pooled into one histogram. Drawing the prior on the
    same axes makes the figure answer what the time courses cannot: how much has the data narrowed
    this parameter relative to the prior it started from? A density lying on the prior has learned
    nothing.

    The x axis is in the parameter's own ABSOLUTE units either way -- ticks are plain values, never
    powers of ten and never dex. ``scale`` chooses how that axis is spaced, and the binning, the
    density's unit, and the prior's shape all follow from it consistently:

    ``"log"``     bins uniform in decades, which is where a log-uniform prior is flat. Height is a
                  density PER DECADE and the prior is the horizontal line ``1 / (hi - lo)``. This
                  resolves the whole prior range evenly, so a parameter spanning 1..316 is legible
                  across its entire support.
    ``"linear"``  bins uniform in absolute units, matching the linear value axis the time-course
                  figures use. Height is a density PER UNIT. The prior is then NOT flat: pushing a
                  log-uniform density through ``x = 10**y`` gives ``p(x) = 1 / ((hi - lo) x ln 10)``,
                  a 1/x curve, and that curve is what gets drawn. A flat line here would be simply
                  wrong, and would make every parameter look as though its low end had been
                  disfavored by the data when it was only the change of variable.

    The family's trajectory line is marked for reference, with the caveat that it is a MEDOID over
    windows and not the mode of this mixture; the two answer different questions and need not
    coincide.
    """
    para = spec.table[p_index]
    name = spec.display.get(key, key)
    ref = spec.references.get(key)
    lo, hi = float(spec.prior_low[p_index]), float(spec.prior_high[p_index])
    log_bins = scale == "log"
    if log_bins:
        edges_log = np.linspace(lo, hi, POOLED_BINS + 1)
        edges = 10.0 ** edges_log                       # absolute edges, log-spaced
    else:
        edges = np.linspace(10.0 ** lo, 10.0 ** hi, POOLED_BINS + 1)

    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    peak = 0.0
    for e, kind in enumerate(kinds):
        draws = pooled[e][:, p_index]
        draws = draws[np.isfinite(draws)]
        if draws.size == 0:
            continue
        color = CONDITION_COLOR.get(kind, f"C{e}")
        disp = CONDITION_DISPLAY.get(kind, kind)
        # Bin in the space the axis is spaced in, so the plotted height is a density with respect to
        # the axis actually shown: per decade of log10 value, or per absolute unit.
        dens, _ = (np.histogram(draws, bins=edges_log, density=True) if log_bins
                   else np.histogram(10.0 ** draws, bins=edges, density=True))
        ax.stairs(dens, edges, color=color, linewidth=2.0, fill=False,
                  label=f"{disp} pooled ({draws.size:,} draws)")
        ax.stairs(dens, edges, color=color, alpha=0.16, fill=True, linewidth=0)
        peak = max(peak, float(np.nanmax(dens)) if dens.size else 0.0)
        ax.axvline(float(line[e, p_index]), color=color, linestyle="-", linewidth=1.4, alpha=0.70,
                   label=f"{disp} {family}-trajectory = {line[e, p_index]:.3g}")

    # The prior it started from. Flat per decade; a 1/x curve per absolute unit, because that is what
    # a log-uniform density becomes under x = 10**y -- drawing it flat here would be wrong.
    prior_label = f"prior (log-uniform over {10 ** lo:.3g}–{10 ** hi:.3g})"
    if log_bins:
        ax.axhline(1.0 / (hi - lo), color="0.35", linestyle=(0, (4, 3)), linewidth=1.5,
                   label=prior_label)
    else:
        grid = np.linspace(10.0 ** lo, 10.0 ** hi, 512)
        ax.plot(grid, 1.0 / ((hi - lo) * grid * np.log(10.0)), color="0.35",
                linestyle=(0, (4, 3)), linewidth=1.5, label=prior_label)
    if ref is not None:
        _apply_reference(ax, ref, spec.reference_short, orient="v")

    raw_unit = para.get("UNIT")
    unit = (_UNIT_SHORT.get(raw_unit, raw_unit) if raw_unit
            else _UNIT_BY_KEY.get(para.get("KEY", ""), "absolute"))
    ax.set_xlim(float(edges[0]), float(edges[-1]))
    if log_bins:
        ax.set_xscale("log")
        # Plain absolute tick values (1, 10, 100), not 10^n: the axis reads in the parameter's units.
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _pos: f"{v:g}"))
        ax.xaxis.set_minor_formatter(FuncFormatter(lambda v, _pos: ""))
    else:
        ax.xaxis.set_major_locator(MaxNLocator(nbins=10))
        ax.xaxis.set_minor_locator(AutoMinorLocator(2))
    ax.set_xlabel(f"inferred [{unit}]")
    ax.set_ylabel(f"posterior density [per decade]" if log_bins
                  else f"posterior density [per {unit}]")
    if not log_bins and peak > 0.0:
        # Scale to the DATA, not to the prior. Per absolute unit the log-uniform prior diverges as
        # 1/x toward the lower limit, and letting that spike set the limit flattens the posterior
        # into the axis. The prior curve simply leaves the top of the frame where it exceeds the
        # data, which costs nothing: the region where the comparison is informative is the region
        # where the two are of comparable height.
        ax.set_ylim(0.0, 1.15 * peak)
    ax.set_title(f"Pooled posterior over the recording (time axis collapsed)\n"
                 f"{name} ({para['LABEL']})", fontsize=10.5)
    ax.legend(fontsize=7, framealpha=0.9, loc="best")
    ax.grid(True, which="both", alpha=0.18)
    fig.tight_layout()
    return fig


def _cells_label(cells):
    """Compact description of which cells a per-time-point SGM selected (switching made visible)."""
    uniq = sorted({int(c) for c in cells if c >= 0})
    if not uniq:
        return "none"
    if len(uniq) == 1:
        return str(uniq[0])
    return f"{len(uniq)} distinct: {','.join(str(u) for u in uniq[:6])}" + \
           ("…" if len(uniq) > 6 else "")


def _step(y):
    """Extend a per-chunk series so a step plot holds each value across its whole window.

    Each estimate summarizes one window, not an instant, so the honest rendering is a piecewise
    constant held from the window's start to its end -- straight lines between chunk points would
    draw an interpolation the analysis never computed. Paired with ``edges`` (one more point than
    chunks) and ``where="post"``, this repeats the last value so the final window is drawn to the
    recording's true end.
    """
    y = np.asarray(y, dtype=float)
    return np.append(y, y[-1])


def _finish_axes(ax, x_edges, para, spec, lo, hi):
    """Linear y-axis in the parameter's own absolute units, with robust limits.

    Absolute units, not log: the readable quantity is the value and its absolute change, and a log
    axis plus a mirrored decade axis proved to be clutter that obscured both. Limits come from a
    robust percentile range of the data extended by the reference band, so a few outlying per-cell
    traces clip instead of compressing the informative region. Decade-space quantities (the fitted
    drift in dex, the fold change) live in the report table, where a reader can consult them without
    re-reading the axis.
    """
    ax.set_xlabel("time [s]")
    ax.set_xlim(float(x_edges[0]), float(x_edges[-1]))
    ax.set_xticks(x_edges)
    raw_unit = para.get("UNIT")
    unit = (_UNIT_SHORT.get(raw_unit, raw_unit) if raw_unit
            else _UNIT_BY_KEY.get(para.get("KEY", ""), "absolute"))
    ax.set_ylabel(f"inferred [{unit}]")
    if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
        pad = 0.10 * (hi - lo)
        ax.set_ylim(max(0.0, lo - pad), hi + pad)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))


# =============================================================================
# Report
# =============================================================================

def _write_report(spec, meta, results, drift, kinds, family, line, picks_window, picks_traj,
                  pooled=None):
    """Write the self-contained interpretation report beside the figures.

    Every reported quantity is defined verbatim here, keyed by the same name the figures and the
    companion note use, so a reader can judge any number without opening the code.
    """
    displays = [CONDITION_DISPLAY.get(k, k) for k in kinds]
    L = [f"# Experiment temporal dynamics — {spec.alias} {spec.timing_label}",
         f"Run: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}", ""]
    L.append(f"Generated from `{meta['npz_name']}`. **{meta['n_estimates']} MAP estimates** = "
             f"{len(kinds)} conditions × {meta['n_cells']} recordings × {meta['n_chunks']} "
             f"non-overlapping {meta['step']:g} s windows spanning "
             f"[{meta['edges'][0]:g}, {meta['edges'][-1]:g}) s. Conditions: {', '.join(displays)} "
             f"(stored tokens {', '.join(kinds)}). Central-estimate family: **{family}**.")
    L.append("")
    L.append("## What the analysis asks, and what it cannot answer alone")
    L.append("")
    L.append("The Experiment stage reports one **MAP estimate per window** — the point estimate it "
             "optimizes for one (condition, cell, chunk) window. Stacking the windows along time "
             "asks whether an inferred value holds still across the recording. A parameter that is "
             "a constant property of the system should be flat; a trend is **either real dynamics "
             "or an acquisition confound**, and this analysis cannot decide which.")
    L.append("")
    L.append("That limit is structural. The biology workflow holds imaging fixed, so it is blind to "
             "imaging drift; the detector workflow marginalizes the reaction-diffusion block, so it "
             "is blind to biological drift. The two read the **same recordings** — the experimental "
             "path pattern carries no workflow qualifier, so a given recording index is the same "
             "acquisition in both — which makes each the other's control. Neither attributes a "
             "cause on its own.")
    L.append("")
    L.append("## The central estimate — exactly what was aggregated")
    L.append("")
    L.append("A timeseries needs one vector per chunk, so the **cell** axis is aggregated; a single "
             "summary line needs one vector overall, so the **cell and chunk** axes are. Crossing "
             "that with the choice of estimator gives four, named for what they aggregate:")
    L.append("")
    L.append("| name | definition | drawn as |")
    L.append("|---|---|---|")
    L.append("| `mean-window` | Mean value vector, aggregated across cells for a given chunk, of "
             "the MAP estimate vectors computed for (chunk, cell) pairs. Each parameter is averaged "
             "independently. | timeseries |")
    L.append("| `sgm-window` | Realized value vector, aggregated across cells for a given chunk, "
             "minimizing the summed normalized distance to the other cells' vectors at that chunk, "
             "of the MAP estimate vectors computed for (chunk, cell) pairs. | timeseries |")
    L.append("| `mean-trajectory` | Mean value vector, aggregated across chunks and cells, of the "
             "MAP estimate vectors computed for (chunk, cell) pairs. | horizontal line |")
    L.append("| `sgm-trajectory` | Realized value vector, aggregated across chunks and cells, "
             "minimizing the summed normalized distance to all other (chunk, cell) vectors. | "
             "horizontal line |")
    L.append("")
    L.append(f"This run drew the **{family}-window** timeseries with the **{family}-trajectory** "
             f"summary line. The pairing is enforced, so the curve and the line are always the same "
             f"estimator.")
    L.append("")
    L.append("**Distance metric** (both `sgm-*` estimates): absolute values `10**theta`, each "
             "parameter divided by its absolute prior width `10**high - 10**low` so no parameter "
             "dominates, Euclidean, **exact medoid** — the member minimizing the summed distance to "
             "every other member of the set. Selection is on **all parameters jointly**, so a "
             "selected vector is internally coherent; consequently the value it reports for one "
             "parameter is that jointly-central window's value, not that parameter's own median.")
    L.append("")
    L.append("A `mean-*` estimate averages each parameter independently, so its coordinates need "
             "not have co-occurred in any recording. An `sgm-*` estimate is an actual window, so "
             "its coordinates did.")
    L.append("")
    if family == "sgm":
        L.append("### Which windows were selected")
        L.append("")
        for e, kind in enumerate(kinds):
            disp = CONDITION_DISPLAY.get(kind, kind)
            L.append(f"- **{disp}** — `sgm-trajectory`: cell {picks_traj[e][0]}, chunk "
                     f"{picks_traj[e][1]} (one realized window). `sgm-window` selected "
                     f"{_cells_label(picks_window[e])} across the {meta['n_chunks']} chunks.")
        L.append("")
        L.append("`sgm-window` selecting more than one cell is expected and must be read with the "
                 "curve: **a step between adjacent chunks can be a change of cell rather than a "
                 "change in time.** `sgm-trajectory` is a single window and carries no such "
                 "ambiguity, and it is the same quantity the standalone sample-geometric-median "
                 "analysis reports over the same pooled windows, so the two agree by construction.")
        L.append("")
    L.append("## Drift — measured per cell, independent of the display")
    L.append("")
    L.append("For every (condition, cell, parameter) an ordinary least-squares line is fit to the "
             "stored **log10** MAP estimate against time (log10 because drift is multiplicative), "
             "giving a slope and hence fitted endpoints:")
    L.append("")
    L.append("```")
    L.append("change_dex = slope * (t_last - t_first)")
    L.append("start      = 10 ** (fitted value at t_first)")
    L.append("end        = 10 ** (fitted value at t_last)")
    L.append("```")
    L.append("")
    L.append("Because the fit is per cell, **none of these statistics depends on the central "
             "estimate the figures draw**: swapping a mean for a medoid changes what is displayed, "
             "not what is measured.")
    L.append("")
    L.append("| name | definition |")
    L.append("|---|---|")
    L.append("| `drift-absolute` | Median across cells of `end - start`, in the parameter's own "
             "units. **Annotated on the figure.** |")
    L.append("| `drift-sign-consistency` | Fraction of cells whose change shares the sign of the "
             "median change. **Annotated on the figure.** |")
    L.append("| `drift-fold` | Median across cells of `end / start`, a multiplicative factor. |")
    L.append("| `drift-dex` | Median across cells of `change_dex`, in log10 units. |")
    L.append(f"| `drift-material-fraction` | Fraction of cells whose `|change_dex|` exceeds "
             f"{drift['threshold']:g} dex (a factor of two). |")
    L.append("| `drift-wilcoxon-p` | Two-sided signed-rank test that the per-cell changes are "
             "centered at zero. A **detectability** statement, not a magnitude. |")
    L.append("| `reference-ratio` | The `*-trajectory` value divided by the reference mean, where a "
             "reference applies to that condition. |")
    L.append("| `reference-verdict` | Whether the `*-trajectory` value lies inside the reference "
             "band. |")
    L.append("")
    hdr = ("| Parameter | " + " | ".join(f"{d}: {family}-traj" for d in displays) + " | " +
           " | ".join(f"{d}: drift-absolute / fold / dex / sign / >2x / p" for d in displays) +
           " | reference |")
    L.append(hdr)
    L.append("|" + "---|" * (1 + 2 * len(displays) + 1))
    for r in results:
        i = r["p_index"]
        traj = " | ".join(f"{line[e, i]:.4g}" for e in range(len(kinds)))
        stats = " | ".join(
            f"{drift['drift_absolute'][e, i]:+.3g} / {drift['drift_fold'][e, i]:.2f}x / "
            f"{drift['drift_dex'][e, i]:+.3f} / {100 * drift['drift_sign_consistency'][e, i]:.0f}% / "
            f"{100 * drift['drift_material_fraction'][e, i]:.0f}% / "
            f"{drift['drift_wilcoxon_p'][e, i]:.1e}" for e in range(len(kinds)))
        ref = spec.references.get(r["key"])
        if ref is None:
            rcol = "—"
        else:
            r_mean, r_lo, r_hi, _ = _reference_band(ref)
            applies = ref.get("applies_to")
            parts = []
            for e, kind in enumerate(kinds):
                disp = CONDITION_DISPLAY.get(kind, kind)
                if applies is not None and disp not in applies:
                    continue
                v = line[e, i]
                inside = "inside" if (np.isfinite(v) and r_lo <= v <= r_hi) else "outside"
                parts.append(f"{disp} {v / r_mean:.2g}x, {inside}")
            rcol = (f"{r_mean:.3g} [{r_lo:.3g}–{r_hi:.3g}] {ref['unit']}; " + "; ".join(parts)
                    if parts else f"{r_mean:.3g} {ref['unit']} (no applicable condition)")
        L.append(f"| {r['name']} ({r['label']}) | {traj} | {stats} | {rcol} |")
    L.append("")
    L.append("**A time-aggregated summary of a drifting parameter summarizes a non-stationary "
             "process.** Where `drift-absolute` is large and `drift-sign-consistency` high, the "
             "`*-trajectory` line — and any reference comparison drawn against it — is a summary "
             "over that drift, not a measurement of a constant.")
    L.append("")
    L.append("## Uncertainty figure")
    L.append("")
    L.append("`<key>_temporal_posterior.png` shows, at each chunk, the **median across cells of "
             "that window's stored posterior interval** — how uncertain a typical single window's "
             "estimate is. It is an interval **width** built from the stored five-quantile record, "
             "**not a posterior density**, and it does not approximate one.")
    L.append("")
    L.append("When a parameter's interval spans essentially its whole prior, the posterior is "
             "returning the prior and the point estimate is the prior's center regardless of the "
             "data.")
    L.append("")
    if pooled is not None:
        L.append("## Pooled posterior density (time axis collapsed)")
        L.append("")
        L.append("`<key>_temporal_posterior_pooled.png` is the classic histogram: every window's "
                 "posterior draws, pooled within a condition, in the parameter's absolute units. "
                 f"This run used {POOLED_BINS} **{meta['pooled_scale']}-spaced** bins. With "
                 "`log` spacing the bins are uniform across the prior's decades, height is a "
                 "density **per decade**, and the log-uniform prior is the flat dashed line at "
                 "`1 / (log10 hi - log10 lo)`. With `linear` spacing the bins are uniform in "
                 "absolute units, height is a density **per unit**, and the prior is drawn as the "
                 "`1 / x` curve a log-uniform density becomes under `x = 10**y` — not a flat line, "
                 "which would misread the change of variable as evidence. Either way the distance "
                 "between a curve and the prior is what the data added.")
        L.append("")
        L.append("**How the pooling works, exactly.** The Experiment stage draws "
                 f"{meta['n_samples_per_window']:,} samples from the posterior of every "
                 f"(recording, window) pair. For one condition that is {meta['n_cells']} "
                 f"recordings x {meta['n_chunks']} windows x "
                 f"{meta['n_samples_per_window']:,} draws, and all of them enter one histogram "
                 "with equal weight.")
        L.append("")
        L.append("**What that mixture does and does not mean.** Equal-weight pooling makes the "
                 "**mixture** of the per-window posteriors. Its density answers: for a single "
                 "window drawn at random from this condition, which values are consistent with "
                 "it? It is **not** a joint posterior for the condition — combining independent "
                 "observations under Bayes multiplies likelihoods, whereas pooling draws adds "
                 "densities. The mixture is therefore as wide as the between-window spread plus "
                 "the within-window uncertainty, while a genuine joint posterior over all windows "
                 "would be narrower than any single one of them. Read the width as a property of "
                 "the window population, never as evidence accumulated over the recording.")
        L.append("")
        L.append(f"Because it is a population rather than an estimate, the mixture's mode and "
                 f"median are **not** this analysis's central estimate. That remains "
                 f"`{family}-trajectory`, marked as a thin vertical line: one jointly realized "
                 f"vector, which a per-parameter summary of pooled marginals is not. The two "
                 f"answer different questions and need not coincide.")
        L.append("")
        headers = ["parameter", "label"] + [
            f"{CONDITION_DISPLAY.get(k, k)} {q}" for k in kinds
            for q in ("p05", "median", "p95")]
        rows = []
        for r in results:
            i = r["p_index"]
            row = [f"`{r['key']}`", r["label"]]
            for e in range(len(kinds)):
                qs = tdk.cloud_interval(pooled[e])[:, i]
                row += [f"{10.0 ** v:.3g}" for v in qs]
            rows.append(row)
        L.append("| " + " | ".join(headers) + " |")
        L.append("|" + "---|" * len(headers))
        for row in rows:
            L.append("| " + " | ".join(row) + " |")
        L.append("")
        L.append("Marginal quantiles of the pooled mixture, in absolute units. Per-coordinate by "
                 "construction, so they carry no joint information — which is why they sit beside "
                 f"the `{family}-trajectory` vector rather than replacing it.")
        L.append("")
    if spec.references:
        L.append("## External reference values")
        L.append("")
        L.append(f"Source: {spec.reference_source}. A reference is drawn only on the parameters it "
                 f"constrains and only for the conditions it applies to; drawing it elsewhere would "
                 f"invite a false comparison.")
        L.append("")
        for k, ref in spec.references.items():
            scope = ("both conditions" if ref.get("applies_to") is None
                     else ", ".join(ref["applies_to"]) + " only")
            L.append(f"- **{spec.display.get(k, k)}** ({scope}): {ref['note']}")
        L.append("")
        missing = [k for k in spec.keys if k not in spec.references]
        if missing:
            L.append(f"No external reference exists for: "
                     f"{', '.join(spec.display.get(k, k) for k in missing)}. These are read on "
                     f"their internal evidence alone — drift, recovery, and posterior width.")
            L.append("")
    L.append("## How to read a parameter")
    L.append("")
    if any(r["key"] for r in results):
        L.append("**Held-out recovery bands.** Where a recovery artifact exists for this workflow, "
                 "each figure states the fraction of held-out synthetic videos whose inferred value "
                 "landed inside two nested tolerances of the truth: " +
                 " and ".join(f"`{tdk.band_label(b)}`" for b in tdk.RECOVERY_BANDS_DEX) +
                 ". These are the multiplicative ranges of the log10 half-widths " +
                 " and ".join(f"±{b:g} dex" for b in tdk.RECOVERY_BANDS_DEX) +
                 " that the Evaluation stage reports — a factor of two and a factor of the square "
                 "root of two — restated as the value range they permit, since that is what a "
                 "reader needs in order to judge whether the parameter is usable.")
        L.append("")
    L.append("Trust a parameter on experimental data when it recovers well on held-out synthetic "
             "data, **and** is stationary where the model says it should be, **and** its posterior "
             "is narrow relative to its prior. A parameter that recovers poorly carries no signal "
             "however smooth its trajectory looks, and a posterior that spans its prior is "
             "reporting the prior back. A parameter that drifts is not thereby discredited — the "
             "drift may be the acquisition's, which is what the other workflow's run measures.")
    L.append("")
    (spec.fig_dir / "report.md").write_text("\n".join(L), encoding="utf-8")


# =============================================================================
# Engine
# =============================================================================

def run_temporal_dynamics(cfg, args):
    """Shared entry point. ``cfg`` is a WorkflowConfig; ``args`` the parsed CLI namespace."""
    spec = _temporal_dynamics_spec(cfg, args)
    step = args.chunk_step_seconds if args.chunk_step_seconds is not None else args.total_time_seconds

    div = "=" * 78
    print(div)
    print(f" {spec.alias} — Experiment Temporal Dynamics ({cfg.tag})")
    print(f" timing_label : {spec.timing_label}   window step : {step} s")
    print(f" reads        : {spec.npz_path}")
    print(f" writes       : {spec.fig_dir}  (figures + report.md)")
    print(div)

    if args.dry_run:
        print("\n[DRY RUN] input validation only:")
        print(f"  experiment .npz : {spec.npz_path}  "
              f"[{'OK' if spec.npz_path.exists() else 'MISSING'}]")
        print(f"  recovery .npz   : {spec.recovery_npz}  "
              f"[{'OK' if spec.recovery_npz.exists() else 'absent (recovery annotation skipped)'}]")
        print(f"  parameters      : {len(spec.keys)} ({', '.join(spec.keys)})")
        print(f"  references for  : {', '.join(spec.references) or 'none'}")
        print("[DRY RUN] no figures written.\n")
        return 0

    if not spec.npz_path.exists():
        raise FileNotFoundError(
            f"Experiment array not found: {spec.npz_path}. Run the {cfg.tag} Experiment stage "
            f"for --total-time-seconds {args.total_time_seconds} first.")

    with np.load(str(spec.npz_path), allow_pickle=False) as d:
        inferred_log10 = np.asarray(d["inferred_log10"], dtype=float)
        kind_index = d["kind_index"].astype(int)
        cell = d["cell"].astype(int)
        chunk = d["chunk"].astype(int)
        kinds = [str(k) for k in d["kinds"]]
        quant = (np.asarray(d["posterior_quantiles"], dtype=float)
                 if "posterior_quantiles" in d.files else None)
        # Raw per-window draws, present only when the Experiment stage ran with
        # --dump-posterior-samples. Their absence costs the pooled-density figure and nothing else.
        cloud = (np.asarray(d["posterior_samples_cloud"], dtype=float)
                 if "posterior_samples_cloud" in d.files else None)

    if inferred_log10.shape[1] != len(spec.keys):
        raise ValueError(f"Experiment output has {inferred_log10.shape[1]} parameters but the "
                         f"{cfg.tag} workflow has {len(spec.keys)}: {spec.keys}")

    grid, n_cells, n_chunks = tdk.reshape_to_grid(
        inferred_log10, kind_index, cell, chunk, len(kinds))
    abs_grid = 10.0 ** grid
    # A chunk is a WINDOW, not an instant. Edges span the recording's true extent (n_chunks*step,
    # e.g. 0..20 s for ten 2 s windows) and drive the step plots; the drift fit uses window CENTRES,
    # which is where each estimate's information actually sits. The fitted slope, and hence every
    # drift statistic, is identical either way -- only the reported endpoints shift by half a window.
    edges = step * np.arange(n_chunks + 1)
    centers = step * (np.arange(n_chunks) + 0.5)

    # The chosen family fixes BOTH the timeseries and its summary line, so a figure never mixes a
    # mean with a medoid. Every kernel function below already returns ABSOLUTE units.
    family = args.central
    if family == "sgm":
        series, picks_window = tdk.sgm_window(grid, spec.prior_low, spec.prior_high)
        line, picks_traj = tdk.sgm_trajectory(grid, spec.prior_low, spec.prior_high)
    else:
        series, picks_window = tdk.mean_window(grid), None
        line, picks_traj = tdk.mean_trajectory(grid), None
    drift = tdk.drift_statistics(grid, centers)

    within = None
    if quant is not None and quant.size:
        qgrid, _, _ = tdk.reshape_to_grid(quant, kind_index, cell, chunk, len(kinds))
        within = tdk.within_window_interval(qgrid)

    pooled = None
    if cloud is not None and cloud.size:
        pooled = tdk.pooled_cloud(cloud, kind_index, len(kinds))

    recovery = None
    if spec.recovery_npz.exists():
        with np.load(str(spec.recovery_npz), allow_pickle=False) as d:
            if "true_log10" in d.files and "inferred_log10" in d.files:
                recovery = tdk.recovery_fractions(d["true_log10"], d["inferred_log10"])

    print(f"\nLoaded {inferred_log10.shape[0]} estimates: {len(kinds)} conditions x "
          f"{n_cells} recordings x {n_chunks} windows.")
    print(f"  windows            : " + ", ".join(
        f"[{edges[i]:g},{edges[i + 1]:g})" for i in range(n_chunks)) + " s")
    print(f"  central estimate   : {family}-window (timeseries) + {family}-trajectory (line)")
    if family == "sgm":
        print("  sgm-window cells   : " + ", ".join(
            f"{CONDITION_DISPLAY.get(k, k)}=[{_cells_label(picks_window[e])}]"
            for e, k in enumerate(kinds)))
        print("  sgm-trajectory     : " + ", ".join(
            f"{CONDITION_DISPLAY.get(k, k)}=(cell {picks_traj[e][0]}, chunk {picks_traj[e][1]})"
            for e, k in enumerate(kinds)))
    print(f"  posterior panel    : "
          f"{'on (stored quantiles)' if within is not None else 'off (no posterior_quantiles)'}")
    if pooled is not None:
        print(f"  pooled density     : on — " + ", ".join(
            f"{CONDITION_DISPLAY.get(k, k)} {pooled[e].shape[0]:,} draws"
            for e, k in enumerate(kinds))
            + f" ({cloud.shape[1]:,} per window, {POOLED_BINS} "
              f"{args.pooled_scale}-spaced bins over the prior)")
    else:
        print("  pooled density     : off (no posterior_samples_cloud: re-run the Experiment "
              "stage with --dump-posterior-samples)")
    print(f"  recovery annotation: " + ("off" if recovery is None else
          ", ".join(f"{tdk.band_label(b)}" for b, _ in recovery)) + "\n")

    wanted = [k.strip() for k in args.params.split(",")] if args.params else None
    spec.fig_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for p_index, key in enumerate(spec.keys):
        if wanted is not None and key not in wanted:
            continue
        fig = _figure_trajectory(spec, p_index, key, abs_grid, series, line, family,
                                 picks_window, picks_traj, edges, kinds, drift, recovery)
        fig.savefig(str(spec.fig_dir / f"{key}_temporal.png"), dpi=180)
        plt.close(fig)
        note = ""
        if within is not None:
            figp = _figure_posterior(spec, p_index, key, within, series, family, edges, kinds)
            figp.savefig(str(spec.fig_dir / f"{key}_temporal_posterior.png"), dpi=180)
            plt.close(figp)
            note = " + posterior"
        if pooled is not None:
            figq = _figure_pooled(spec, p_index, key, pooled, line, family, kinds,
                                  scale=args.pooled_scale)
            figq.savefig(str(spec.fig_dir / f"{key}_temporal_posterior_pooled.png"), dpi=180)
            plt.close(figq)
            note += " + pooled"
        results.append({"p_index": p_index, "key": key,
                        "name": spec.display.get(key, key), "label": spec.table[p_index]["LABEL"]})
        ref_tag = "  [reference]" if key in spec.references else ""
        print(f"  wrote {key}_temporal.png{note}{ref_tag}")

    meta = {"npz_name": spec.npz_path.name, "n_estimates": int(inferred_log10.shape[0]),
            "n_cells": n_cells, "n_chunks": n_chunks, "step": step,
            "n_samples_per_window": int(cloud.shape[1]) if cloud is not None else 0,
            "pooled_scale": args.pooled_scale,
            "edges": [float(t) for t in edges], "centers": [float(t) for t in centers]}
    _write_report(spec, meta, results, drift, kinds, family, line, picks_window, picks_traj,
                  pooled)
    print(f"  wrote report.md\n\nDone: {len(results)} parameter(s) in {spec.fig_dir}")
    return 0


def build_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--total-time-seconds", type=float, required=True,
                   help="run duration; selects the Experiment output via its timing label "
                        "(e.g. 2.0 -> 2S_50FPS). Must match a completed Experiment run.")
    p.add_argument("--chunk-step-seconds", type=float, default=None,
                   help="spacing between consecutive windows on the time axis. Default: the run "
                        "duration (non-overlapping windows). Set this only if the Experiment run "
                        "used overlapping windows, in which case the true spacing is the step.")
    p.add_argument("--central", choices=tdk.CENTRAL_FAMILIES, default="sgm",
                   help="central-estimate family. 'sgm' (default) draws the sgm-window timeseries "
                        "(a realized value vector aggregated across cells for each chunk) with the "
                        "sgm-trajectory summary line (a realized value vector aggregated across "
                        "chunks and cells); 'mean' draws the mean-window timeseries and the "
                        "mean-trajectory line, each parameter averaged independently. The pairing "
                        "is enforced so a figure never mixes a mean with a medoid.")
    p.add_argument("--pooled-scale", choices=("log", "linear"), default="log",
                   help="x-axis spacing for the pooled posterior histogram; both are in the "
                        "parameter's absolute units. 'log' (default) bins uniformly in decades, so "
                        "the whole prior range is resolved evenly and the log-uniform prior is a "
                        "flat reference line (density per decade). 'linear' bins uniformly in "
                        "absolute units, matching the linear value axis of the time-course figures, "
                        "and draws the prior as the 1/x curve a log-uniform density becomes under "
                        "that change of variable (density per unit).")
    p.add_argument("--params", type=str, default=None,
                   help="comma-separated parameter keys to plot (default: all). Filters the "
                        "FIGURES only -- the central estimates are computed on the full parameter "
                        "vector either way, so this cannot change which recording is central.")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve and print the inputs and outputs, then exit without reading data "
                        "or writing anything.")
    return p

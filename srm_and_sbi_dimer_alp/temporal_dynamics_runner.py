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
cell and then aggregated, so they do not depend on the choice of central estimate. The central
estimate governs only what the figures display, and it is a real recording rather than an average:
averaging each parameter independently across cells composes a vector whose coordinates never
co-occurred. The trajectory-level Sample Geometric Median (one cell, whole time course) is the
headline; the per-time-point SGM is shown alongside with its selected cell annotated, because its
central cell may change between time points and that change can masquerade as temporal structure.

The per-workflow differences -- parameter table and keys, prior box, alias-qualified paths, display
names, unit labels, and the external reference values -- are resolved once in
:func:`_temporal_dynamics_spec`.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")            # headless: build and save figures without a display
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from . import temporal_dynamics as tdk
from .parameterization import PARAMETERS, RunTiming
from .workflow import parameter_keys, parameter_table

# Conditions are named scientifically wherever a reader sees them; the tokens below survive only
# as the stored ``kinds`` field of the Experiment output and the recording filenames on disk.
CONDITION_DISPLAY = {"ALP": "MET-FAB", "BET": "MET-INLB"}
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
    )


def _reference_band(ref):
    """Mean, [lo, hi] band, and the individual reported values of an external reference."""
    vals = [v for (v, _sd, _lab) in ref["estimates"]]
    sds = [(sd or 0.0) for (_v, sd, _lab) in ref["estimates"]]
    return (float(np.mean(vals)),
            float(min(v - s for v, s in zip(vals, sds))),
            float(max(v + s for v, s in zip(vals, sds))), vals)


def _dex_axis(ax):
    """Label the right-hand side in dex, mirroring a log-scaled physical left axis.

    The visual spacing is then dex -- the space the priors are declared in and the space drift is
    measured in -- while the left labels stay in the unit the simulator consumes, so one figure
    carries both readings instead of duplicating every panel per scale.

    Ticks are placed explicitly at half-dex positions rather than through a functional secondary
    axis: composing a log10 transform with an already-logarithmic parent scale applies the
    transform twice and mislabels the axis.
    """
    lo, hi = ax.get_ylim()
    if not (np.isfinite(lo) and np.isfinite(hi)) or lo <= 0:
        return None
    sec = ax.twinx()
    sec.set_yscale("log")
    sec.set_ylim(lo, hi)
    dex = np.arange(np.floor(np.log10(lo) * 2) / 2, np.log10(hi) + 0.5, 0.5)
    dex = dex[(10.0 ** dex >= lo) & (10.0 ** dex <= hi)]
    sec.set_yticks(10.0 ** dex)
    sec.set_yticklabels([f"{d:+.1f}" for d in dex], fontsize=8)
    sec.minorticks_off()
    sec.set_ylabel("log10 (dex)", fontsize=9)
    return sec


def _apply_reference(ax, ref, displays_present):
    """Draw an external reference band, its individual values, and its mean. Returns (lo, hi)."""
    r_mean, r_lo, r_hi, r_vals = _reference_band(ref)
    ax.axhspan(r_lo, r_hi, color=REFERENCE_COLOR, alpha=0.25, linewidth=0)
    for v in r_vals:
        ax.axhline(v, color=REFERENCE_COLOR, linestyle=":", linewidth=1.1, alpha=0.75)
    applies = ref.get("applies_to")
    scope = "" if applies is None else f" [{', '.join(applies)} only]"
    ax.axhline(r_mean, color=REFERENCE_COLOR, linestyle="-", linewidth=2.1,
               label=f"reference {r_mean:.3g} {ref['unit']}{scope}")
    if ref.get("upper_biased") is not None:
        ub, ub_lab = ref["upper_biased"]
        ax.axhline(ub, color=REFERENCE_COLOR, linestyle="--", linewidth=1.6, alpha=0.9,
                   label=f"upper bound {ub:.3g} ({ub_lab})")
        r_hi = max(r_hi, ub)
    return r_lo, r_hi


# =============================================================================
# Figures
# =============================================================================

def _figure_trajectory(spec, p_index, key, abs_grid, sgm_b_abs, sgm_b_cells,
                       sgm_a_abs, sgm_a_cells, x, kinds, drift, recovery_frac):
    """Central-trajectory figure: SGM headline, per-time SGM companion, retained mean, reference.

    Layers, in the order a reader should take them:
      - faint per-cell trajectories (the ensemble the central estimates summarize);
      - the TRAJECTORY-LEVEL SGM (thick gold): one real recording's whole time course, so no point
        on this curve was composed and no step can be a change of cell;
      - the PER-TIME-POINT SGM (thin gold, dashed): jointly realized within each time point, but
        its central cell may change between points, so the selected cell is printed in the legend
        and any change is visible there;
      - the cross-cell MEAN (thin grey), retained for comparison only: it is the per-dimension
        composite the SGM replaces, and the gap between the two is itself informative;
      - the external reference band where one legitimately applies to this parameter.
    """
    para = spec.table[p_index]
    name = spec.display.get(key, key)
    ref = spec.references.get(key)
    fig, ax = plt.subplots(figsize=(6.8, 5.4))
    hi, lo = -np.inf, np.inf

    for e, kind in enumerate(kinds):
        color = CONDITION_COLOR.get(kind, f"C{e}")
        data = abs_grid[e, :, :, p_index]                     # (n_cells, n_chunks) absolute
        for c in range(data.shape[0]):
            ax.plot(x, data[c], color=color, linestyle="-", alpha=0.10, linewidth=0.8)
        finite = data[np.isfinite(data)]
        if finite.size:
            hi = max(hi, float(np.nanpercentile(finite, 90)))
            lo = min(lo, float(np.nanpercentile(finite, 10)))

    if ref is not None:
        r_lo, r_hi = _apply_reference(ax, ref, None)
        hi, lo = max(hi, r_hi), min(lo, r_lo)

    for e, kind in enumerate(kinds):
        color = CONDITION_COLOR.get(kind, f"C{e}")
        disp = CONDITION_DISPLAY.get(kind, kind)
        mean = np.nanmean(abs_grid[e, :, :, p_index], axis=0)
        ax.plot(x, mean, color=color, linestyle=":", linewidth=1.3, alpha=0.85,
                label=f"{disp} cross-cell mean (composite)")
        ax.plot(x, sgm_a_abs[e, :, p_index], color=color, linestyle="--", linewidth=1.5,
                alpha=0.9, label=f"{disp} per-time SGM (cells {_cells_label(sgm_a_cells[e])})")
        ax.plot(x, sgm_b_abs[e, :, p_index], color=color, linestyle="-", linewidth=2.8,
                label=f"{disp} SGM trajectory (cell {sgm_b_cells[e]})")
        d = drift["median_dex"][e, p_index]
        ax.annotate(f"{disp}: {d:+.3f} dex / {x[-1] - x[0]:g} s", xy=(0.02, 0.97 - 0.055 * e),
                    xycoords="axes fraction", fontsize=8, color=color, va="top")

    _finish_axes(ax, x, para, spec, lo, hi)
    title = f"Inferred parameter over the recording — {name} ({para['LABEL']})"
    if recovery_frac is not None:
        title += f"\nheld-out recovery within ±0.3 dex: {recovery_frac[p_index] * 100:.0f}%"
    ax.set_title(title, fontsize=10.5)
    ax.legend(fontsize=7, framealpha=0.9, loc="best")
    fig.tight_layout()
    return fig


def _figure_posterior(spec, p_index, key, within, between, sgm_b_abs, sgm_b_cells,
                      x, kinds):
    """Two-panel uncertainty figure, with the two uncertainties kept apart on purpose.

    Panel (a) WITHIN-WINDOW POSTERIOR SPREAD: at each time, the median across cells of the stored
    per-window posterior interval -- how uncertain a typical single-window estimate is.
    Panel (b) BETWEEN-CELL SPREAD: at each time, percentiles across cells of the per-window median
    -- how much recordings differ from each other, i.e. biological and experimental heterogeneity.

    Plotting these on one axis would let a wide posterior masquerade as heterogeneity and vice
    versa, so they get separate panels sharing one y-axis, with the trajectory-level SGM drawn on
    both as the common anchor.

    Both panels are built from the STORED FIVE-QUANTILE SUMMARY of each window's posterior, not
    from posterior samples: they show interval widths, not a density. A pooled posterior density
    across windows would require the per-window sample clouds, which the Experiment stage does not
    persist.
    """
    para = spec.table[p_index]
    name = spec.display.get(key, key)
    ref = spec.references.get(key)
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.9), sharey=True)
    hi, lo = -np.inf, np.inf

    for ax, panel in zip(axes, ("within", "between")):
        for e, kind in enumerate(kinds):
            color = CONDITION_COLOR.get(kind, f"C{e}")
            disp = CONDITION_DISPLAY.get(kind, kind)
            if panel == "within":
                q = 10.0 ** within[e, :, p_index, :]          # (n_chunks, 5) absolute
                ax.fill_between(x, q[:, 0], q[:, 4], color=color, alpha=0.16, linewidth=0,
                                label=f"{disp} posterior 5–95%")
                ax.fill_between(x, q[:, 1], q[:, 3], color=color, alpha=0.30, linewidth=0,
                                label=f"{disp} posterior 25–75%")
            else:
                q = 10.0 ** between[e, :, p_index, :]         # (n_chunks, 5) absolute
                ax.fill_between(x, q[:, 0], q[:, 4], color=color, alpha=0.14, linewidth=0,
                                hatch="///", edgecolor=color,
                                label=f"{disp} cells 5–95%")
                ax.fill_between(x, q[:, 1], q[:, 3], color=color, alpha=0.26, linewidth=0,
                                label=f"{disp} cells 25–75%")
            fin = q[np.isfinite(q)]
            if fin.size:
                hi = max(hi, float(fin.max()))
                lo = min(lo, float(fin.min()))
            ax.plot(x, sgm_b_abs[e, :, p_index], color=color, linestyle="-", linewidth=2.4,
                    label=f"{disp} SGM trajectory (cell {sgm_b_cells[e]})")
        if ref is not None:
            r_lo, r_hi = _apply_reference(ax, ref, None)
            hi, lo = max(hi, r_hi), min(lo, r_lo)

    axes[0].set_title("(a) within-window posterior spread\n(median across cells of one window's interval)",
                      fontsize=9.5)
    axes[1].set_title("(b) between-cell spread\n(percentiles across cells of the per-window median)",
                      fontsize=9.5)
    for ax in axes:
        _finish_axes(ax, x, para, spec, lo, hi)
    axes[1].set_ylabel("")
    axes[0].legend(fontsize=6.5, framealpha=0.9, loc="best")
    fig.suptitle(f"Uncertainty over the recording — {name} ({para['LABEL']})", fontsize=10.5)
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


def _finish_axes(ax, x, para, spec, lo, hi):
    """Log-scaled physical y-axis plus the mirrored dex axis, and a time x-axis."""
    ax.set_xlabel("time [s]")
    ax.set_xlim(float(x[0]), float(x[-1]))
    ax.set_xticks(x)
    raw_unit = para.get("UNIT")
    unit = (_UNIT_SHORT.get(raw_unit, raw_unit) if raw_unit
            else _UNIT_BY_KEY.get(para.get("KEY", ""), "absolute"))
    ax.set_ylabel(f"inferred [{unit}]")
    if np.isfinite(lo) and np.isfinite(hi) and lo > 0 and hi > lo:
        ax.set_yscale("log")
        ax.set_ylim(lo / 1.25, hi * 1.25)
        _dex_axis(ax)
    else:
        ax.yaxis.set_major_locator(MaxNLocator(nbins=9))
        ax.yaxis.set_minor_locator(AutoMinorLocator(2))


# =============================================================================
# Report
# =============================================================================

def _write_report(spec, meta, results, drift, kinds, sgm_b_cells, sgm_a_cells):
    """Write the self-contained interpretation report beside the figures."""
    displays = [CONDITION_DISPLAY.get(k, k) for k in kinds]
    L = [f"# Experiment temporal dynamics — {spec.alias} {spec.timing_label}", ""]
    L.append(f"Generated from `{meta['npz_name']}`. **{meta['n_estimates']} estimates** = "
             f"{len(kinds)} conditions × {meta['n_cells']} recordings × {meta['n_chunks']} "
             f"non-overlapping {meta['step']:g} s windows. Time points: "
             f"{', '.join(f'{t:g}' for t in meta['x'])} s. Conditions: {', '.join(displays)} "
             f"(stored tokens {', '.join(kinds)}).")
    L.append("")
    L.append("## What this analysis does, and what it cannot do alone")
    L.append("")
    L.append("The Experiment stage estimates the parameters independently in every window of every "
             "recording, so stacking the windows along time asks whether an inferred value holds "
             "still across the recording — a resolution a single whole-recording estimate cannot "
             "provide. A parameter that is a constant property of the system should be flat; a "
             "trend is **either real dynamics or an acquisition confound**, and this analysis "
             "cannot by itself decide which.")
    L.append("")
    L.append("That limit is structural, and it is why the analysis exists for both workflows. The "
             "biology workflow holds the imaging block fixed, so it is blind to imaging drift; the "
             "detector workflow marginalizes the reaction-diffusion block, so it is blind to "
             "biological drift. Each workflow's temporal result is therefore the other's "
             "confound test, and the two read the SAME recordings — the experimental path pattern "
             "carries no workflow qualifier, so a given recording index is the same acquisition in "
             "both. Comparing them bounds the confound; neither alone attributes a cause.")
    L.append("")
    L.append("## The central estimate is a real recording, not an average")
    L.append("")
    L.append("Averaging each parameter independently across recordings composes a vector whose "
             "coordinates never co-occurred in any recording. Two Sample Geometric Median "
             "estimates are used instead, and they answer different questions:")
    L.append("")
    L.append("- **SGM trajectory (headline, thick line).** The one recording whose ENTIRE time "
             "course is most central, selected as the medoid over recordings of the flattened "
             "(time × parameter) trajectory in prior-range-normalized absolute space. Every point "
             "comes from the same acquisition, so no plotted value is a composite and **no step "
             "can be an artifact of switching between recordings**.")
    L.append("- **Per-time-point SGM (companion, dashed).** The medoid recording at each time "
             "independently. Its coordinates are jointly realized within a time point, but **the "
             "selected recording may change between time points**, and such a change can look "
             "like temporal structure. The selected recordings are therefore printed in the "
             "legend, and where more than one appears the curve must be read with that in mind.")
    L.append("- **Cross-cell mean (dotted, retained for comparison).** The per-dimension composite "
             "the SGM replaces. The gap between it and the SGM trajectory is itself informative: "
             "where they separate, the composite is asserting a combination no recording produced.")
    L.append("")
    L.append("Both SGM variants are computed on the **full parameter vector**, never on a plotted "
             "subset, so restricting the figures with `--params` cannot change which recording is "
             "central.")
    L.append("")
    L.append("## Drift over the recording (measured per recording, independent of the display)")
    L.append("")
    L.append("Each recording's own estimate is regressed on time in log10 and reported as the "
             "total change over the observed span, in dex. Aggregating per-recording fits by their "
             "median keeps one erratic recording from moving the summary, and because the fit is "
             "per recording **these statistics do not depend on the choice of central estimate** — "
             "swapping an average for a geometric median changes what is displayed, not what is "
             "measured.")
    L.append("")
    hdr = "| Parameter | " + " | ".join(f"{d} drift (dex)" for d in displays) + \
          " | " + " | ".join(f"{d} |d|>{drift['threshold']:g} / sign / p" for d in displays) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (1 + 2 * len(displays)))
    for r in results:
        e_i = r["p_index"]
        dv = " | ".join(f"{drift['median_dex'][e, e_i]:+.3f}" for e in range(len(kinds)))
        ds = " | ".join(f"{100 * drift['frac_material'][e, e_i]:.0f}% / "
                        f"{100 * drift['sign_consistency'][e, e_i]:.0f}% / "
                        f"{drift['wilcoxon_p'][e, e_i]:.1e}" for e in range(len(kinds)))
        L.append(f"| {r['name']} ({r['label']}) | {dv} | {ds} |")
    L.append("")
    L.append(f"`|d|>{drift['threshold']:g}` is the fraction of recordings whose drift exceeds "
             f"{drift['threshold']:g} dex (the same practical bar the recovery tables use for a "
             f"factor of two); `sign` is the fraction sharing the median's direction; `p` is a "
             f"two-sided Wilcoxon signed-rank test that the per-recording drifts are centered at "
             f"zero. A large drift with high sign-consistency and a small p is a coherent "
             f"within-recording trend, not scatter.")
    L.append("")
    L.append("## Uncertainty figures — two quantities, deliberately not combined")
    L.append("")
    L.append("Each `<key>_temporal_posterior.png` carries two panels sharing one axis:")
    L.append("")
    L.append("- **(a) within-window posterior spread** — at each time, the median across "
             "recordings of that window's stored posterior interval. This is how uncertain a "
             "typical *single* window's estimate is.")
    L.append("- **(b) between-cell spread** — at each time, percentiles across recordings of the "
             "per-window median. This is how much recordings differ from each other.")
    L.append("")
    L.append("They are separated because plotting them together would let a wide posterior "
             "masquerade as heterogeneity, or heterogeneity as posterior width — a conflation that "
             "would misstate what the data supports.")
    L.append("")
    L.append("**What these panels are built from.** The stored per-window five-quantile summary "
             "(5, 25, 50, 75, 95%), not posterior samples. They therefore show interval *widths*; "
             "they are not a posterior *density*. A pooled density across windows would require "
             "the per-window sample clouds, which the Experiment stage does not persist — that "
             "remains an extension, and this analysis does not approximate it.")
    L.append("")
    L.append("## Central estimates selected (provenance)")
    L.append("")
    for e, kind in enumerate(kinds):
        disp = CONDITION_DISPLAY.get(kind, kind)
        L.append(f"- **{disp}** — SGM trajectory: recording {sgm_b_cells[e]} "
                 f"(one recording, all time points). Per-time-point SGM selected "
                 f"{_cells_label(sgm_a_cells[e])}.")
    L.append("")
    if spec.references:
        L.append("## External reference values")
        L.append("")
        L.append(f"Source: {spec.reference_source}. A reference is drawn only on the parameters it "
                 f"legitimately constrains, and only for the conditions it applies to — several "
                 f"are valid for the monomer control alone, and drawing them against the dimer "
                 f"condition would invite a false comparison.")
        L.append("")
        for k, ref in spec.references.items():
            scope = "both conditions" if ref.get("applies_to") is None \
                else ", ".join(ref["applies_to"]) + " only"
            L.append(f"- **{spec.display.get(k, k)}** ({scope}): {ref['note']}")
        L.append("")
        missing = [k for k in spec.keys if k not in spec.references]
        if missing:
            L.append(f"No external reference exists for: "
                     f"{', '.join(spec.display.get(k, k) for k in missing)}. These are read on "
                     f"their internal evidence alone (drift, recovery, and posterior width).")
            L.append("")
    L.append("## How to read a parameter")
    L.append("")
    L.append("Trust a parameter on experimental data when it recovers well on held-out synthetic "
             "data **and** is stationary where the model says it should be **and** its posterior is "
             "narrow relative to its prior. A parameter that recovers poorly carries no signal "
             "regardless of how smooth its trajectory looks, and a posterior that spans its prior "
             "is reporting the prior back.")
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

    if inferred_log10.shape[1] != len(spec.keys):
        raise ValueError(f"Experiment output has {inferred_log10.shape[1]} parameters but the "
                         f"{cfg.tag} workflow has {len(spec.keys)}: {spec.keys}")

    grid, n_cells, n_chunks = tdk.reshape_to_grid(
        inferred_log10, kind_index, cell, chunk, len(kinds))
    abs_grid = 10.0 ** grid
    x = step * np.arange(n_chunks)

    sgm_b, sgm_b_cells, sgm_b_methods = tdk.central_trajectory_sgm(
        grid, spec.prior_low, spec.prior_high)
    sgm_a, sgm_a_cells = tdk.central_per_time_sgm(grid, spec.prior_low, spec.prior_high)
    drift = tdk.drift_statistics(grid, x)
    # The kernel works in log10 (the space priors and drift live in); the figures plot absolute
    # values on a log-scaled axis, where a negative log10 value cannot be drawn at all. Convert
    # here, once, so no figure can silently receive the wrong space.
    sgm_b_abs = 10.0 ** sgm_b
    sgm_a_abs = 10.0 ** sgm_a

    within = between = None
    if quant is not None and quant.size:
        qgrid, _, _ = tdk.reshape_to_grid(quant, kind_index, cell, chunk, len(kinds))
        within, between = tdk.quantile_summaries(qgrid)

    recovery_frac = None
    if spec.recovery_npz.exists():
        with np.load(str(spec.recovery_npz), allow_pickle=False) as d:
            if "true_log10" in d.files and "inferred_log10" in d.files:
                err = np.abs(np.asarray(d["inferred_log10"]) - np.asarray(d["true_log10"]))
                recovery_frac = np.mean(err <= tdk.MATERIAL_DRIFT_DEX, axis=0)

    print(f"\nLoaded {inferred_log10.shape[0]} estimates: {len(kinds)} conditions x "
          f"{n_cells} recordings x {n_chunks} windows.")
    print(f"  time points        : {[float(t) for t in x]} s")
    print(f"  SGM trajectory     : " + ", ".join(
        f"{CONDITION_DISPLAY.get(k, k)}=recording {sgm_b_cells[e]} ({sgm_b_methods[e]})"
        for e, k in enumerate(kinds)))
    print(f"  per-time SGM cells : " + ", ".join(
        f"{CONDITION_DISPLAY.get(k, k)}=[{_cells_label(sgm_a_cells[e])}]"
        for e, k in enumerate(kinds)))
    print(f"  posterior panels   : "
          f"{'on (stored quantiles)' if within is not None else 'off (no posterior_quantiles)'}")
    print(f"  recovery annotation: {'on' if recovery_frac is not None else 'off'}\n")

    wanted = [k.strip() for k in args.params.split(",")] if args.params else None
    spec.fig_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for p_index, key in enumerate(spec.keys):
        if wanted is not None and key not in wanted:
            continue
        fig = _figure_trajectory(spec, p_index, key, abs_grid, sgm_b_abs, sgm_b_cells,
                                 sgm_a_abs, sgm_a_cells, x, kinds, drift, recovery_frac)
        fig.savefig(str(spec.fig_dir / f"{key}_temporal.png"), dpi=180)
        plt.close(fig)
        note = ""
        if within is not None:
            figp = _figure_posterior(spec, p_index, key, within, between, sgm_b_abs, sgm_b_cells,
                                     x, kinds)
            figp.savefig(str(spec.fig_dir / f"{key}_temporal_posterior.png"), dpi=180)
            plt.close(figp)
            note = " + posterior"
        results.append({"p_index": p_index, "key": key,
                        "name": spec.display.get(key, key), "label": spec.table[p_index]["LABEL"]})
        ref_tag = "  [reference]" if key in spec.references else ""
        print(f"  wrote {key}_temporal.png{note}{ref_tag}")

    meta = {"npz_name": spec.npz_path.name, "n_estimates": int(inferred_log10.shape[0]),
            "n_cells": n_cells, "n_chunks": n_chunks, "step": step,
            "x": [float(t) for t in x]}
    _write_report(spec, meta, results, drift, kinds, sgm_b_cells, sgm_a_cells)
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
    p.add_argument("--params", type=str, default=None,
                   help="comma-separated parameter keys to plot (default: all). Filters the "
                        "FIGURES only -- the central estimates are computed on the full parameter "
                        "vector either way, so this cannot change which recording is central.")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve and print the inputs and outputs, then exit without reading data "
                        "or writing anything.")
    return p

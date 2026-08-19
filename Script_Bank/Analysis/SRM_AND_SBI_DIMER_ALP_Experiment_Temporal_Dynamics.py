"""Analysis entry point: temporal dynamics of the inferred parameters on real data.

Regenerates, from a completed Experiment stage, the per-parameter *temporal* view
of the MAP estimates on the real MET single-particle-tracking recordings. For each
learnable parameter it collects the MAP estimate from every non-overlapping video
chunk, groups the estimates by time point (the chunk's position within the
recording), averages across cells, and plots the resulting trajectory per
experimental condition (MET-FAB and MET-INLB) in absolute (physical) units.

=============================================================================
WHY THIS ANALYSIS  (read this before interpreting the figures)
=============================================================================
Primary purpose -- TEMPORAL DYNAMICS. Each real recording is split into short,
non-overlapping windows (e.g. ten 2 s windows over a 20 s recording). Estimating
the parameters independently in each window and plotting them against the window's
position in the recording shows how the inferred parameter behaves over time.

Secondary, parameter-dependent read -- ROBUSTNESS / STATIONARITY. Several
parameters (notably the kinetic rates, e.g. the dissociation rate) are constant
properties of the system: their true value does not change within one recording.
For those, a FLAT trajectory is positive evidence that the estimator is
time-invariant and self-consistent -- it returns the same answer regardless of
which short window it is shown. A systematic TREND instead signals either genuine
dynamics or an acquisition confound (photobleaching, for instance, progressively
removes visible emitters and can pull the apparent initial-count parameters
downward over the recording). Whether a trend is signal or confound is judged per
parameter -- hence "depending on the parameter".

What this adds over the experimental readout. A single-molecule experiment (e.g.
Li et al. 2026, below) yields one estimate plus a distribution for the WHOLE
recording; it cannot resolve a parameter in time or per cell. This inference
resolves it per short window and per cell -- temporal and single-cell granularity
the experiment cannot reach. Averaging the per-window MAP estimates over time (and
cells) recovers the comparable whole-recording point estimate, now backed by the
demonstrated (or refuted) stationarity.

VALIDATION against experiment. For a few parameters the source paper reports
experimental estimates WITH a spread, so each is drawn as a grey band (the reported
range) plus a grey mean line, and the inferred time-averaged MAP is compared to it:
  * kappa_OFF (dissociation rate) = 1 / dimer lifetime. Li et al. measured lifetimes
    of 1.30 +/- 0.05 s (InlB T-T) and 0.80 +/- 0.03 s (InlB H-T) -> kappa_OFF of
    ~0.77 and ~1.25 / s (mean ~1.0), so the experimental band spans ~0.74-1.30 / s.
  * D_A (monomer diffusivity), donor-only segments: 0.109 +/- 0.068 and
    0.093 +/- 0.053 um^2/s -> ~0.10 um^2/s.
  * R_B (rel. mobile-dimer diffusivity): dimer ~1.6x slower than monomer, i.e. the
    diffusivity ratio ~0.6 (0.066/0.109, 0.056/0.093).
Because that study activated MET with InlB, kappa_OFF and R_B compare most directly
to the MET-INLB condition; the monomer diffusivity D_A is ligand-independent and
applies to both. "We don't need to match": the check is simply that the time
average lands within / near the reported experimental range.

  Reference: Y. Li, M. S. Dietz, H.-D. Barth, H. H. Niemann, M. Heilemann,
  "Single-Molecule FRET-Tracking of InlB-Activated MET Receptors in Living Cells",
  Small 2026, 22, e07115. doi:10.1002/smll.202507115.

=============================================================================
NOT IMPLEMENTED YET -- aggregated posterior distributions  (documented on purpose)
=============================================================================
The historical figures also included, per parameter, a POSTERIOR DISTRIBUTION
panel: a single histogram (per condition) of the full posterior sample cloud
pooled across every cell and every chunk of the experiment. Reproducing that
faithfully needs the per-window posterior SAMPLES, aggregated across all chunks
into one distribution per (condition, parameter). The current Experiment stage
persists only five quantiles per window (Q05/Q25/Q50/Q75/Q95), NOT the sample
pool, so the quantiles alone are insufficient (a five-number summary per window
cannot be re-pooled into the experiment-wide sample distribution the original
histogrammed). To add these panels later: draw posterior samples per (cell, chunk)
window from the trained posterior, concatenate them across all chunks and cells of
a condition, and histogram the pooled samples in log10. That requires either
persisting the per-window sample pool in the Experiment stage or a dedicated
resampling pass over the real videos; it is left as a documented extension.

=============================================================================
INPUTS / OUTPUTS
=============================================================================
Reads  <data_bank>/<posit_subdir>/<project_alias>_{timing_label}_MAP_Experiment/
         <project_alias>_{timing_label}_MAP_Experiment.npz
       arrays used: inferred_log10 (N,10) log10 MAP, kind_index (N,), cell (N,),
       chunk (N,), kinds (2,). Optionally the sibling MAP_Recovery .npz (same timing)
       is read to annotate each parameter with its ground-truth recovery quality.
Writes <...>_MAP_Experiment/temporal_dynamics/
         <key>_temporal.png   -- one temporal figure per learnable parameter
         report.md            -- self-contained interpretation + per-parameter
                                 summary of THIS run (numbers filled in)
       A dedicated subdirectory, so it never collides with the Experiment stage's
       own figures/ directory (which that stage clears on each run).

See also the method reference `SRM_AND_SBI_DIMER_ALP_Experiment_Temporal_Dynamics.md`
next to this script for the full scientific interpretation.

Usage:
    MACHINE_PROFILE=<profile> python \\
        SRM_AND_SBI_DIMER_ALP_Experiment_Temporal_Dynamics.py --total-time-seconds 2.0

    # preview the resolved paths without reading or plotting anything:
    MACHINE_PROFILE=<profile> python \\
        SRM_AND_SBI_DIMER_ALP_Experiment_Temporal_Dynamics.py --total-time-seconds 2.0 --dry-run
"""

import argparse
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")   # headless: construct + save figures without a display
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator, MaxNLocator

from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, PARAMETERIZATION, RunTiming

# ---------------------------------------------------------------------------
# Condition mapping -- verbatim from the original analysis. The .npz stores the
# raw kind strings 'ALP' / 'BET'; the figures use the ligand display names.
# ---------------------------------------------------------------------------
CONDITION_DISPLAY = {"ALP": "MET-FAB", "BET": "MET-INLB"}
CONDITION_COLOR = {"ALP": "tab:blue", "BET": "tab:orange"}
EXPERIMENTAL_COLOR = "tab:grey"

# ---------------------------------------------------------------------------
# Experimental references (Li et al., Small 2026, e07115), keyed by PARAMETERIZATION
# KEY. Each carries the reported experimental point estimates as (value, sd, label)
# in the parameter's ABSOLUTE unit -- so the figure can draw the reported RANGE (a
# band spanning value +/- sd across the estimates) and the MEAN of the reported
# values, and the inferred time-average can be read against that range.
#   rate_dissociation: kappa_OFF = 1 / dimer-lifetime; lifetimes 1.30 +/- 0.05 s
#     (InlB T-T) and 0.80 +/- 0.03 s (InlB H-T); SD propagated as d(1/t)=dt/t^2.
#   diffusivity_alp:   monomer diffusion, donor-only segments (0.109/0.093 um^2/s).
#   relative_diffusivity_bet: R_B = D_dimer/D_monomer per variant (dimer ~1.6x slower).
# The study measured InlB-activated MET, so kappa_OFF and R_B compare most directly
# to MET-INLB; D_A is ligand-independent.
# ---------------------------------------------------------------------------
EXPERIMENTAL_REFERENCE = {
    "rate_dissociation": {
        "unit": "1/s",
        "estimates": [
            (1.0 / 1.30, 0.05 / 1.30 ** 2, "InlB T-T  (1/tau, tau=1.30 s)"),
            (1.0 / 0.80, 0.03 / 0.80 ** 2, "InlB H-T  (1/tau, tau=0.80 s)"),
        ],
        "note": "kappa_OFF = 1/tau per FRET variant: 1/1.30=0.77, 1/0.80=1.25 1/s "
                "(tau = 1.30+/-0.05, 0.80+/-0.03 s; SD propagated as d(tau)/tau^2); "
                "band = value+/-SD; mean line = mean of the two RATES = 1.01 "
                "(consistent with the paper's '~1 s' lifetime -> 1.0). Li et al. 2026, Fig. 2C",
    },
    "diffusivity_alp": {
        "unit": "um^2/s",
        "estimates": [
            (0.109, 0.068, "donor-only T-T"),
            (0.093, 0.053, "donor-only H-T"),
        ],
        "note": "monomer diffusion, donor-only segments: 0.109+/-0.068 (T-T), "
                "0.093+/-0.053 (H-T) um^2/s; band = value+/-SD; mean = 0.10. Li et al. 2026, Sec. 2.3",
    },
    "relative_diffusivity_bet": {
        "unit": "ratio",
        "estimates": [
            (0.066 / 0.109, None, "T-T  (D_dimer/D_mono)"),
            (0.056 / 0.093, None, "H-T  (D_dimer/D_mono)"),
        ],
        "note": "R_B = D_dimer/D_monomer per FRET variant: 0.066/0.109=0.61 (T-T), "
                "0.056/0.093=0.60 (H-T); mean = 0.60 (consistent with 'dimer ~1.6x slower', "
                "1/1.6=0.625). Li et al. 2026, Sec. 2.3",
    },
}

# Short human-readable name per parameter KEY (titles); the LaTeX symbol comes from
# PARAMETERIZATION[...]['LABEL'] and the unit from ['UNIT'].
PARAM_DISPLAY_NAME = {
    "count_alp": "Initial monomer count",
    "count_bet": "Initial mobile-dimer count",
    "count_chi": "Initial immobile-dimer count",
    "diffusivity_alp": "Monomer diffusivity",
    "relative_diffusivity_bet": "Rel. mobile-dimer diffusivity",
    "relative_diffusivity_chi": "Rel. immobile-dimer diffusivity",
    "relative_rate_dimerization": "Rel. dimerization rate",
    "rate_dissociation": "Dissociation rate",
    "rate_immobility": "Immobilization rate",
    "rate_mobility": "Mobilization rate",
}

# Compact axis unit from the PARAMETERIZATION 'UNIT' string.
UNIT_SHORT = {
    "Count": "count",
    "Square Micrometer Per Second": "$\\mu$m$^2$/s",
    "Dimensionless": "ratio",
    "Count Per Second": "1/s",
}


def _abs(map_log10):
    """Convert a stored log10 MAP value to its absolute (linear) value.

    Every learnable parameter has LOG_FLAG=True, LOG_BASE=10 in PARAMETERIZATION,
    so the stored theta is log10(value) and the physical value is 10**theta. The
    relative-diffusivity / relative-rate parameters (R_B, R_C, R_ON) are stored as
    dimensionless ratios; 10**theta is therefore the ratio itself.
    """
    return np.power(10.0, map_log10)


def _reference_band(ref):
    """Mean, [lo, hi] band, and the individual reported values of an experimental reference.

    The band spans each reported estimate extended by its reported SD (an SD of None
    contributes zero width for that estimate); the mean is the mean of the reported
    point estimates. Used to draw the experimental range + mean on a figure.
    """
    vals = [v for (v, _sd, _lab) in ref["estimates"]]
    sds = [(sd or 0.0) for (_v, sd, _lab) in ref["estimates"]]
    mean = float(np.mean(vals))
    lo = float(min(v - s for v, s in zip(vals, sds)))
    hi = float(max(v + s for v, s in zip(vals, sds)))
    return mean, lo, hi, vals


def _reshape_to_grid(inferred_log10, kind_index, cell, chunk, n_kinds):
    """Scatter the flat (N, n_param) MAP rows into a dense (kind, cell, chunk, param) grid.

    The Experiment .npz stores one flat row per (cell, chunk) window. Each row is
    placed at grid[kind_index, cell, chunk, :]. Windows never estimated (should not
    happen for a complete run) stay NaN, and every statistic uses nan-aware reductions
    so a missing window widens nothing silently.
    """
    n_cells = int(cell.max()) + 1
    n_chunks = int(chunk.max()) + 1
    n_param = inferred_log10.shape[1]
    grid = np.full((n_kinds, n_cells, n_chunks, n_param), np.nan, dtype=float)
    grid[kind_index, cell, chunk] = inferred_log10
    return grid, n_cells, n_chunks


def _recovery_within_band(recovery_npz_path, band=0.3):
    """Per-parameter fraction of EVAL videos recovered within +/- `band` log10.

    Reads the sibling MAP_Recovery .npz (true_log10 + inferred_log10) if present, so
    each temporal figure can be annotated with how well that parameter is even
    recoverable on ground-truth data. Returns None if absent or lacking the arrays.
    """
    try:
        if not recovery_npz_path.exists():
            return None
        with np.load(str(recovery_npz_path), allow_pickle=False) as d:
            if "true_log10" not in d.files or "inferred_log10" not in d.files:
                return None
            err = np.abs(d["inferred_log10"] - d["true_log10"])
            return np.mean(err <= band, axis=0)   # (n_param,) fraction in band
    except Exception:
        return None


def _temporal_figure(p_index, key, abs_grid, x, kinds, recovery_frac=None):
    """Build the temporal-dynamics figure for one parameter; return (Figure, per-condition time-avgs).

    Solid line  = mean over cells of the (absolute) MAP estimate at each time point.
    Shaded band = mean +/- 1 SD across cells (between-cell spread).
    Faint lines = each cell's own MAP trajectory (clipped to the band-based y-limits).
    If the parameter has an experimental reference: a grey band marks the reported
    experimental range, dotted grey lines the individual reported values, a grey line
    the mean, and a dashed line per condition its time-averaged MAP (the validation).

    The y-limits are derived from the certainty bands (mean +/- SD) and the reference,
    NOT from the full data range -- so outlier per-cell trajectories are clipped away
    and the informative region fills the plot.
    """
    para = PARAMETERIZATION[p_index]
    unit = para["UNIT"]
    label = para["LABEL"]
    name = PARAM_DISPLAY_NAME.get(key, key)
    ref = EXPERIMENTAL_REFERENCE.get(key)

    fig, ax = plt.subplots(figsize=(6.4, 5.2))

    # Per-condition statistics first, so the y-limits can be based on the bands.
    stats = []          # (color, disp, data, mean, std, time_avg)
    band_hi, band_lo = -np.inf, np.inf
    for e, kind in enumerate(kinds):
        color = CONDITION_COLOR.get(kind, f"C{e}")
        disp = CONDITION_DISPLAY.get(kind, kind)
        data = abs_grid[e, :, :, p_index]                 # (n_cells, n_chunks), abs units
        mean = np.nanmean(data, axis=0)
        std = np.nanstd(data, axis=0)
        stats.append((color, disp, data, mean, std, float(np.nanmean(data))))
        band_hi = max(band_hi, float(np.nanmax(mean + std)))
        band_lo = min(band_lo, float(np.nanmin(mean - std)))

    # Faint per-cell context (drawn first; clipped by the band-based y-limits below).
    for (color, _disp, data, _mean, _std, _tavg) in stats:
        for c in range(data.shape[0]):
            ax.plot(x, data[c], color=color, linestyle=":", alpha=0.10, linewidth=0.8)

    # Experimental reference: reported range (band) + individual values + mean.
    time_avgs = {}
    if ref is not None:
        r_mean, r_lo, r_hi, r_vals = _reference_band(ref)
        band_hi = max(band_hi, r_hi)
        band_lo = min(band_lo, r_lo)
        ax.axhspan(r_lo, r_hi, color=EXPERIMENTAL_COLOR, alpha=0.28, linewidth=0)
        for v in r_vals:
            ax.axhline(v, color=EXPERIMENTAL_COLOR, linestyle=":", linewidth=1.2, alpha=0.75)
        ax.axhline(r_mean, color=EXPERIMENTAL_COLOR, linestyle="-", linewidth=2.2,
                   label=f"Experimental {r_mean:.2g} [{r_lo:.2g}–{r_hi:.2g}] {ref['unit']} (Li et al. 2026)")

    # Mean trajectories + between-cell bands + per-condition time-averages.
    for (color, disp, _data, mean, std, tavg) in stats:
        time_avgs[disp] = tavg
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.12, linewidth=0)
        ax.plot(x, mean, color=color, linestyle="-", linewidth=2.4, label=disp)
        if ref is not None:
            ax.axhline(tavg, color=color, linestyle="--", linewidth=1.3, alpha=0.85,
                       label=f"{disp} time-avg = {tavg:.2f}")

    # y-limits from the certainty bands (+ reference), clipping outlier per-cell lines.
    if not np.isfinite(band_hi) or not np.isfinite(band_lo):
        band_hi, band_lo = 1.0, 0.0
    span = band_hi - band_lo
    pad = 0.12 * span if span > 0 else (0.1 * abs(band_hi) + 1e-9)
    ax.set_ylim(max(0.0, band_lo - pad), band_hi + pad)   # params are non-negative
    # denser y-ticks (labeled majors + finer unlabeled minors); no grid (distracting)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=11))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.set_xlabel("time [s]")
    ax.set_xlim(x[0], x[-1])
    ax.set_xticks(x)
    ax.set_ylabel(f"inferred [{UNIT_SHORT.get(unit, unit)}]")
    title = f"Mean MAP over time — {name} ({label})"
    if recovery_frac is not None:
        title += f"\nEVAL recovery within ±0.3 log10: {recovery_frac[p_index] * 100:.0f}%"
    ax.set_title(title, fontsize=11)
    ax.legend(fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return fig, time_avgs


def _write_report(fig_dir, meta, results):
    """Write a self-contained interpretation report (report.md) beside the figures."""
    displays = meta["displays"]
    L = []
    L.append(f"# Experiment temporal dynamics — {meta['timing_label']}")
    L.append("")
    L.append(f"Generated by `SRM_AND_SBI_DIMER_ALP_Experiment_Temporal_Dynamics.py` from "
             f"`{meta['npz_name']}`. **{meta['n_estimates']} MAP estimates** "
             f"= {meta['n_kinds']} conditions × {meta['n_cells']} cells × {meta['n_chunks']} "
             f"non-overlapping {meta['step']:g} s windows. Time points: "
             f"{', '.join(f'{t:g}' for t in meta['x'])} s. Conditions: "
             f"{', '.join(displays)} (raw kinds {', '.join(meta['kinds'])}).")
    L.append("")
    L.append("## How to read these figures")
    L.append("")
    L.append("Each `<key>_temporal.png` tracks one parameter over the recording: the "
             "MAP estimate is computed independently in every non-overlapping window of "
             "every cell, and plotted per condition (solid = mean over cells; shaded = "
             "mean ± 1 SD across cells; faint = individual cell trajectories).")
    L.append("")
    L.append("- **Temporal dynamics (primary).** How the inferred parameter behaves "
             "across the recording — a resolution the experiment, which yields one "
             "whole-recording estimate, cannot provide.")
    L.append("- **Robustness / stationarity (parameter-dependent).** Parameters that "
             "are constant properties of the system (the kinetic rates) *should* be "
             "flat; a flat trajectory is evidence of a time-invariant, self-consistent "
             "estimate. A trend is either real dynamics or an acquisition confound "
             "(e.g. photobleaching pulling the apparent counts down over time).")
    L.append("- **Reliability = recovery × stationarity.** Trust a parameter on real "
             "data when it both recovers well on ground-truth EVAL data (annotated on "
             "each figure) and is stationary where it should be. A parameter that "
             "recovers poorly (e.g. R_ON) carries no signal regardless of its trace.")
    L.append("")
    L.append("## Per-parameter summary (this run)")
    L.append("")
    header = "| Parameter | " + " | ".join(f"{d} time-avg" for d in displays) + \
             " | Experimental (Li et al. 2026) | EVAL recovery ±0.3 |"
    L.append(header)
    L.append("|" + "---|" * (1 + len(displays) + 2))
    for r in results:
        tavgs = " | ".join(f"{r['time_avgs'].get(d, float('nan')):.3g}" for d in displays)
        exp = ("—" if r["ref"] is None else
               f"{r['ref']['mean']:.2g} [{r['ref']['lo']:.2g}–{r['ref']['hi']:.2g}] {r['ref']['unit']}")
        rec = "—" if r["recovery"] is None else f"{r['recovery'] * 100:.0f}%"
        L.append(f"| {r['name']} ({r['label']}) | {tavgs} | {exp} | {rec} |")
    L.append("")
    L.append("## Validation against experiment")
    L.append("")
    # Headline: kappa_OFF vs the experimental range, per condition.
    koff = next((r for r in results if r["key"] == "rate_dissociation"), None)
    if koff is not None and koff["ref"] is not None:
        lo, hi = koff["ref"]["lo"], koff["ref"]["hi"]
        parts = []
        for d in displays:
            tv = koff["time_avgs"].get(d)
            if tv is None:
                continue
            inside = "within" if lo <= tv <= hi else "outside"
            parts.append(f"{d} = {tv:.2f} 1/s ({inside} the range)")
        L.append(f"Dissociation rate κ_OFF — experimental range **{lo:.2g}–{hi:.2g} 1/s** "
                 f"(mean {koff['ref']['mean']:.2g}; κ_OFF = 1/dimer-lifetime, Li et al. 2026). "
                 f"Inferred time-averages: " + "; ".join(parts) + ". "
                 f"The InlB-derived reference compares most directly to MET-INLB.")
    L.append("")
    L.append("D_A (monomer diffusivity) and R_B (dimer/monomer diffusivity ratio) carry "
             "experimental references from the same study (D_A is ligand-independent; "
             "R_B compares to MET-INLB). See their figures.")
    L.append("")
    L.append("**How the experimental references were derived** (each in the parameter's "
             "absolute unit; the plotted band spans value ± SD across the reported "
             "estimates and the grey line is their mean):")
    for _key, _ref in EXPERIMENTAL_REFERENCE.items():
        L.append(f"- {PARAM_DISPLAY_NAME.get(_key, _key)}: {_ref['note']}")
    L.append("")
    L.append("## Caveats")
    L.append("")
    L.append("- **Photobleaching** drives the downward drift of the count parameters "
             "(fewer visible emitters in later windows) — a real non-stationarity in a "
             "confounded parameter, not receptor loss.")
    L.append("- **First-pass posteriors** (interrupted training) — absolute values will "
             "sharpen with the production posteriors; re-run this analysis on those.")
    L.append("- **MET-FAB** has no experimental counterpart in the source (InlB only).")
    L.append("- Relative parameters (R_B, R_C, R_ON) are shown as dimensionless ratios.")
    L.append("- The pooled **posterior-distribution** panels are a documented, "
             "not-yet-implemented extension (they need the full per-window sample pool, "
             "not the stored quantiles).")
    L.append("")
    L.append("## Reference")
    L.append("")
    L.append("Y. Li, M. S. Dietz, H.-D. Barth, H. H. Niemann, M. Heilemann, "
             "\"Single-Molecule FRET-Tracking of InlB-Activated MET Receptors in Living "
             "Cells,\" *Small* **2026**, 22, e07115. doi:10.1002/smll.202507115. "
             "Full method interpretation: the companion "
             "`SRM_AND_SBI_DIMER_ALP_Experiment_Temporal_Dynamics.md`.")
    L.append("")
    (fig_dir / "report.md").write_text("\n".join(L), encoding="utf-8")


def main(args):
    """Resolve the Experiment .npz for the requested duration, then plot every parameter."""
    timing = RunTiming(total_time_seconds=args.total_time_seconds,
                       frames=PARAMETERS.simulation.timing)
    timing_label = timing.label
    data_bank_root = PARAMETERS.machine.data_bank_root
    paths = PARAMETERS.paths

    out_dir = paths.experiment_recovery_dir(data_bank_root, timing_label)
    npz_path = out_dir / (out_dir.name + ".npz")
    recovery_dir = paths.map_recovery_dir(data_bank_root, timing_label)
    recovery_npz = recovery_dir / (recovery_dir.name + ".npz")
    fig_dir = out_dir / "temporal_dynamics"

    # Non-overlapping windows: consecutive chunks are one window-length apart, and the
    # window length equals the run duration. The .npz does not record the chunk step,
    # so it defaults to the duration; override --chunk-step-seconds only if the run
    # used overlapping windows (then the true spacing is the step).
    step_seconds = args.chunk_step_seconds if args.chunk_step_seconds is not None \
        else args.total_time_seconds

    div = "=" * 72
    print(div)
    print(f" {paths.project_alias} — Experiment Temporal Dynamics")
    print(f" timing_label : {timing_label}   chunk step : {step_seconds} s")
    print(f" reads npz    : {npz_path}")
    print(f" writes       : {fig_dir}  (figures + report.md)")
    print(div)

    if args.dry_run:
        print("\n[DRY RUN] input validation only:")
        print(f"  experiment .npz : {npz_path}  [{'OK' if npz_path.exists() else 'MISSING'}]")
        print(f"  recovery .npz   : {recovery_npz}  "
              f"[{'OK' if recovery_npz.exists() else 'absent (recovery annotation skipped)'}]")
        print("[DRY RUN] no figures written.\n")
        return

    if not npz_path.exists():
        raise FileNotFoundError(
            f"Experiment array not found: {npz_path}. Run the Experiment stage for "
            f"--total-time-seconds {args.total_time_seconds} first.")

    with np.load(str(npz_path), allow_pickle=False) as d:
        inferred_log10 = d["inferred_log10"]
        kind_index = d["kind_index"].astype(int)
        cell = d["cell"].astype(int)
        chunk = d["chunk"].astype(int)
        kinds = [str(k) for k in d["kinds"]]

    grid, n_cells, n_chunks = _reshape_to_grid(
        inferred_log10, kind_index, cell, chunk, len(kinds))
    abs_grid = _abs(grid)
    x = step_seconds * np.arange(n_chunks)
    recovery_frac = _recovery_within_band(recovery_npz)

    print(f"\nLoaded {inferred_log10.shape[0]} MAP estimates: "
          f"{len(kinds)} conditions x {n_cells} cells x {n_chunks} chunks. "
          f"Recovery annotation: {'on' if recovery_frac is not None else 'off (no MAP_Recovery .npz)'}.")
    print(f"Time points: {[float(t) for t in x]} s\n")

    keys = [p["KEY"] for p in PARAMETERIZATION]
    if args.params:
        wanted = [k.strip() for k in args.params.split(",")]
        keys = [k for k in keys if k in wanted]

    fig_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for p_index, para in enumerate(PARAMETERIZATION):
        key = para["KEY"]
        if key not in keys:
            continue
        fig, time_avgs = _temporal_figure(p_index, key, abs_grid, x, kinds, recovery_frac)
        out_png = fig_dir / f"{key}_temporal.png"
        fig.savefig(str(out_png), dpi=180)
        plt.close(fig)
        ref = EXPERIMENTAL_REFERENCE.get(key)
        ref_summary = None
        if ref is not None:
            r_mean, r_lo, r_hi, _ = _reference_band(ref)
            ref_summary = {"mean": r_mean, "lo": r_lo, "hi": r_hi, "unit": ref["unit"]}
        results.append({
            "key": key,
            "name": PARAM_DISPLAY_NAME.get(key, key),
            "label": para["LABEL"],
            "time_avgs": time_avgs,
            "ref": ref_summary,
            "recovery": (float(recovery_frac[p_index]) if recovery_frac is not None else None),
        })
        tag = "  [experimental reference]" if ref is not None else ""
        print(f"  wrote {out_png.name}{tag}")

    meta = {
        "timing_label": timing_label, "npz_name": npz_path.name,
        "n_estimates": int(inferred_log10.shape[0]), "n_kinds": len(kinds),
        "n_cells": n_cells, "n_chunks": n_chunks, "step": step_seconds,
        "x": [float(t) for t in x], "kinds": kinds,
        "displays": [CONDITION_DISPLAY.get(k, k) for k in kinds],
    }
    _write_report(fig_dir, meta, results)
    print(f"  wrote report.md")
    print(f"\nDone: {len(results)} figure(s) + report.md in {fig_dir}")


def parse_args(argv=None):
    """Construct the CLI parser and parse argv."""
    parser = argparse.ArgumentParser(
        description="Temporal-dynamics figures + report of the inferred parameters over "
                    "the real experimental recordings (per non-overlapping MAP chunk).")
    parser.add_argument(
        "--total-time-seconds", type=float, required=True,
        help="Run duration (selects the Experiment .npz via its timing_label, e.g. "
             "2.0 -> 2S_50FPS). Must match a completed Experiment run.")
    parser.add_argument(
        "--chunk-step-seconds", type=float, default=None,
        help="Spacing between consecutive chunks on the time axis. Default: the run "
             "duration (non-overlapping windows). Set only if the Experiment run used "
             "overlapping windows.")
    parser.add_argument(
        "--params", type=str, default=None,
        help="Comma-separated parameter KEYs to plot (default: all learnable "
             "parameters). Example: rate_dissociation,diffusivity_alp.")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Resolve and print the input/output paths, then exit without reading data "
             "or writing anything.")
    return parser.parse_args(argv)


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))

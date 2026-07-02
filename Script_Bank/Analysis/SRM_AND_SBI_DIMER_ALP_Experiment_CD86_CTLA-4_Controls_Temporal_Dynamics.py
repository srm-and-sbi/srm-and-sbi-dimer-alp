"""Analysis entry point: temporal dynamics of the inferred parameters on the CD86 /
CTLA-4 control-receptor recordings, with a mobile-diffusion headline readout.

Special-scope analysis (ad-hoc reuse). Applies the DIMER-ALP posterior -- trained on
the MET single-particle-tracking regime -- to two control receptors from a different
study, to test whether the transferable observable (the diffusion coefficient) lands
on the independently measured values. CD86 is a constitutive monomer and CTLA-4 a
constitutive dimer; the two bracket the mobile-diffusion scale (monomer faster, dimer
slower). This entry point is not one of the canonical pipeline stages: it is a
documented reuse of a completed control Experiment run, analogous to the Construction
special-situation entry point, and stays out of the stage dispatcher.

Regenerates, from that completed control Experiment run, the per-parameter *temporal*
view of the MAP estimates. For each learnable parameter it collects the MAP estimate
from every non-overlapping video chunk, groups by time point (the chunk's position
within the recording), averages across cells, and plots the trajectory per condition
(CD86 [Monomer] and CTLA-4 [Dimer]) in absolute (physical) units.

=============================================================================
HEADLINE READOUT -- mobile mixture diffusivity  (read this first)
=============================================================================
The transferable quantity across the label / model mismatch (below) is the diffusion
coefficient. The posterior cannot reliably separate monomer from dimer for these
controls, so the headline is the count-weighted mean diffusivity of the MOBILE
populations (monomer A + mobile dimer B), excluding the immobile class C:

    D_mix_mobile = (C_A * D_A + C_B * D_B) / (C_A + C_B)   [um^2/s]

with D_A the monomer diffusivity, D_B = R_B * D_A the mobile-dimer diffusivity, and
C_A, C_B the mobile counts. D_mix lies between D_A and D_B (in either order); the
headline figure compares, per receptor, the inferred D_mix (solid, condition color)
against the experimental D_mobile (solid, a separate per-condition color, so experiment
and inference never share a color); D_A, D_B and the mobile split f_B = C_B / (C_A + C_B)
are reported in the run's report table, not drawn on the figure. On these
out-of-distribution controls the model's A / B
assignment is not physically ordered -- the relative dimer diffusivity R_B can exceed 1
(mobile dimer "faster" than monomer), and under unrestricted pooling D_A / R_B can even
leave their training priors -- which is exactly why the count-weighted D_mix, robust to
the A / B mis-assignment, is the readout rather than D_A or D_B alone. D_mix is read as a
point between D_A and D_B, not as a resolved mixture unless f_B is informative.

D_A is stored in um^2/s directly (10**theta), so D_mix_mobile is directly comparable
to an experimental mobile-fraction diffusion coefficient D_mobile with no unit
conversion. The comparison is quantitatively valid because the acquisition geometry
matches: 50 FPS, 256^2 pixels, ~157 nm/px.

Experimental reference (per condition; drawn as a colored band value +/- SD + a line):
    CD86 [Monomer]  : D_mobile = 0.319 +/- 0.010 um^2/s
    CTLA-4 [Dimer]  : D_mobile = 0.279 +/- 0.005 um^2/s
  Catapano et al., Angew. Chem. Int. Ed. 2025, 64, e202413117 (BioImage Archive
  S-BIAD1369; SiR-S5 / HaloTag single-color tracking). The two values bracket the
  mobile-diffusion scale.

=============================================================================
LABEL / MODEL MISMATCH  (why D, and only D, transfers)
=============================================================================
The posterior was trained on always-visible permanent-label emitters; the control
recordings use an exchangeable, blinking SiR-S5 HaloTag probe. The diffusion
coefficient is a per-track property read from the displacement statistics of a
molecule while it is visible, so it is robust to how many molecules are lit at once
and survives the mismatch. Counts and rates (C_*, kappa_*, R_ON) instead depend on
the number of co-visible emitters and on track continuity, both corrupted by
blinking, so they are NOT read quantitatively here. The controls are also constitutive
monomer / dimer references, not the dynamic A + A <=> B <=> C dimerization mechanism
the posterior encodes: they bracket the diffusion scale and stress-test
transferability, they do not exercise the kinetic model.

The per-parameter temporal figures (counts, rates, ratios) are still produced -- their
downward count drift makes the blinking / bleaching confound visible -- but only the
mobile-diffusion headline carries an experimental comparison.

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
Reads  <data_bank>/<posit_subdir>/<project_alias>_{timing_label}_MAP_Experiment_CD86_CTLA-4_CONTROLS/
         <same>.npz
       arrays used: inferred_log10 (N,10) log10 MAP, kind_index (N,), cell (N,),
       chunk (N,), kinds (2,) = ['CD86','CTLA-4']. Optionally the sibling MAP_Recovery
       .npz (same timing) annotates each parameter with its ground-truth recovery
       quality on simulated EVAL data (a property of the posterior, so it applies
       regardless of which real data the posterior is then applied to).
Writes <...>_MAP_Experiment_CD86_CTLA-4_CONTROLS/temporal_dynamics/
         dmix_mobile_temporal.png -- the headline mobile-diffusion readout
         <key>_temporal.png       -- one temporal figure per learnable parameter
         report.md                -- self-contained interpretation + per-condition
                                     diffusion readout + per-parameter summary
       A dedicated subdirectory, so it never collides with the Experiment stage's
       own figures/ directory (which that stage clears on each run).

See also the companion method reference
`SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls_Temporal_Dynamics.md` next to
this script for the full scientific interpretation.

Usage:
    MACHINE_PROFILE=<profile> python \\
        SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls_Temporal_Dynamics.py --total-time-seconds 2.0

    # preview the resolved paths without reading or plotting anything:
    MACHINE_PROFILE=<profile> python \\
        SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls_Temporal_Dynamics.py --total-time-seconds 2.0 --dry-run
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
# Condition mapping. The control Experiment .npz stores the raw kind strings
# 'CD86' / 'CTLA-4'; the figures use the receptor + oligomeric-state display names.
# CD86 is a constitutive monomer, CTLA-4 a constitutive dimer.
# ---------------------------------------------------------------------------
CONDITION_DISPLAY = {"CD86": "CD86 [Monomer]", "CTLA-4": "CTLA-4 [Dimer]"}
CONDITION_COLOR = {"CD86": "tab:blue", "CTLA-4": "tab:orange"}
EXPERIMENTAL_COLOR = "tab:grey"

# The experimental references get their OWN per-condition colors, distinct from the
# synthetic CONDITION_COLOR, so experiment and inference never share a color while the
# two references (one per receptor) stay distinguishable from each other.
EXPERIMENTAL_CONDITION_COLOR = {"CD86": "tab:green", "CTLA-4": "tab:red"}

# Output-directory pattern of the control Experiment run (matches the runner
# SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls.py) so this analysis reads the
# control .npz, not the canonical MET Experiment output.
CONTROLS_RECOVERY_PATTERN = "{project_alias}_{timing_label}_MAP_Experiment_CD86_CTLA-4_CONTROLS"

# ---------------------------------------------------------------------------
# Experimental mobile-fraction diffusion coefficients D_mobile per control receptor
# (Catapano et al., Angew. Chem. Int. Ed. 2025, 64, e202413117; BioImage Archive
# S-BIAD1369; SiR-S5 / HaloTag single-color tracking at 50 FPS / 256^2 / ~157 nm/px,
# matching the training geometry). Compared against the count-weighted mobile mixture
# diffusivity D_mix_mobile (see _mobile_mixture / _dmix_figure). CD86 (monomer) sits
# above CTLA-4 (dimer): the two bracket the mobile-diffusion scale. Values in um^2/s.
# ---------------------------------------------------------------------------
EXPERIMENTAL_D_MOBILE = {
    "CD86":   {"value": 0.319, "sd": 0.010, "unit": "um^2/s"},
    "CTLA-4": {"value": 0.279, "sd": 0.005, "unit": "um^2/s"},
}

# Short in-text form + full citation for the experimental D_mobile source (single
# source of truth for the figure legend and the report). This is the 2025 Angewandte
# Chemie weak-affinity-labeling paper (accession S-BIAD1369) -- distinct from the 2023
# Cell. Mol. Life Sci. HER2 study (accession S-BIAD597).
EXPERIMENTAL_SOURCE = "Catapano et al. 2025"
EXPERIMENTAL_CITATION = (
    "Catapano et al., \"Long-Term Single-Molecule Tracking in Living Cells using "
    "Weak-Affinity Protein Labeling,\" Angew. Chem. Int. Ed. 2025, 64, e202413117. "
    "doi:10.1002/anie.202413117. BioImage Archive accession S-BIAD1369.")

# No per-parameter experimental band applies to these controls: the only transferable
# experimental comparison is the mobile-diffusion headline (D_mix_mobile vs D_mobile,
# above). Kept as an empty mapping so the per-parameter figure / report reference
# branches (which key on PARAMETERIZATION KEYs) uniformly no-op.
EXPERIMENTAL_REFERENCE = {}

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
        with np.load(str(recovery_npz_path), allow_pickle=True) as d:
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
                   label=f"Experimental {r_mean:.2g} [{r_lo:.2g}–{r_hi:.2g}] {ref['unit']}")

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


def _key_index(key):
    """Column index of a parameter KEY within PARAMETERIZATION (= inferred_log10 columns)."""
    for i, p in enumerate(PARAMETERIZATION):
        if p["KEY"] == key:
            return i
    raise KeyError(f"parameter KEY not found in PARAMETERIZATION: {key}")


def _prior_bounds(key):
    """Absolute-unit (lo, hi) prior bounds for a parameter KEY, from PARAMETERIZATION.

    Used only to flag when the unrestricted-pool MAP has left the training prior on the
    out-of-distribution controls (a diagnostic, not a correction).
    """
    p = PARAMETERIZATION[_key_index(key)]
    lo, hi = p["PRIOR_RANGE"]
    if p.get("LOG_FLAG"):
        base = p.get("LOG_BASE", 10)
        lo, hi = base ** lo, base ** hi
    return float(lo), float(hi)


def _mobile_mixture(abs_grid):
    """Count-weighted mobile mixture diffusivity D_mix_mobile (+ its bracket), per (kind, cell, chunk).

    From the absolute-unit grid reads C_A (count_alp), C_B (count_bet), D_A
    (diffusivity_alp, um^2/s) and R_B (relative_diffusivity_bet); forms the mobile-dimer
    diffusivity D_B = R_B * D_A and the number-weighted mean over the MOBILE populations
    (monomer A + mobile dimer B), excluding the immobile class C:
        D_mix_mobile = (C_A*D_A + C_B*D_B) / (C_A + C_B)   [um^2/s]
    Returns (d_mix, d_a, d_b, f_b), each shaped (n_kinds, n_cells, n_chunks), where
    f_b = C_B/(C_A+C_B) is the mobile-dimer fraction. D_A and D_B (which bracket D_mix,
    in either order -- on out-of-distribution controls R_B can exceed 1, so D_B > D_A)
    and f_b are reported alongside D_mix because the weights (the counts) are the
    label-fragile quantity while D_A, D_B are robust.
    """
    c_a = abs_grid[..., _key_index("count_alp")]
    c_b = abs_grid[..., _key_index("count_bet")]
    d_a = abs_grid[..., _key_index("diffusivity_alp")]
    d_b = abs_grid[..., _key_index("relative_diffusivity_bet")] * d_a
    mobile = c_a + c_b
    d_mix = (c_a * d_a + c_b * d_b) / mobile
    f_b = c_b / mobile
    return d_mix, d_a, d_b, f_b


def _dmix_figure(d_mix, d_a, d_b, f_b, x, kinds):
    """Headline figure: inferred mobile mixture diffusivity D_mix_mobile vs experiment.

    A clean inferred-vs-experiment comparison, one comparison per condition. Every line
    is SOLID; the four series are told apart by COLOR alone:
      * inferred D_mix_mobile per condition -- the condition color (CONDITION_COLOR),
        a trajectory (mean over cells) with a mean +/- 1 SD between-cell band;
      * experimental D_mobile per condition -- a separate per-condition color
        (EXPERIMENTAL_CONDITION_COLOR), a flat reference line with a value +/- SD band.
    Experiment and inference never share a color, and the two experimental references
    stay distinguishable. D_A, D_B and the mobile split f_B are intentionally NOT drawn
    (they are in the report table) so the figure stays uncluttered.
    Returns (Figure, {display: {'dmix','d_a','d_b','f_b'}}).
    """
    fig, ax = plt.subplots(figsize=(6.8, 5.2))

    summaries = {}
    for e, kind in enumerate(kinds):
        disp = CONDITION_DISPLAY.get(kind, kind)
        summaries[disp] = {
            "dmix": float(np.nanmean(d_mix[e])), "d_a": float(np.nanmean(d_a[e])),
            "d_b": float(np.nanmean(d_b[e])), "f_b": float(np.nanmean(f_b[e]))}

    band_hi, band_lo = -np.inf, np.inf

    # Experimental D_mobile per condition: its OWN color, flat reference line + band.
    for e, kind in enumerate(kinds):
        disp = CONDITION_DISPLAY.get(kind, kind)
        ref = EXPERIMENTAL_D_MOBILE.get(kind)
        if ref is not None:
            ecolor = EXPERIMENTAL_CONDITION_COLOR.get(kind, f"C{e + len(kinds)}")
            lo, hi = ref["value"] - ref["sd"], ref["value"] + ref["sd"]
            band_hi = max(band_hi, hi)
            band_lo = min(band_lo, lo)
            ax.axhspan(lo, hi, color=ecolor, alpha=0.15, linewidth=0)
            ax.axhline(ref["value"], color=ecolor, linestyle="-", linewidth=2.2,
                       label=f"{disp} — experimental $D_{{mobile}}$ = {ref['value']:.3f} $\\mu$m$^2$/s ({EXPERIMENTAL_SOURCE})")

    # Inferred D_mix_mobile per condition: the condition color, trajectory + between-cell band.
    for e, kind in enumerate(kinds):
        disp = CONDITION_DISPLAY.get(kind, kind)
        color = CONDITION_COLOR.get(kind, f"C{e}")
        mean = np.nanmean(d_mix[e], axis=0)
        std = np.nanstd(d_mix[e], axis=0)
        band_hi = max(band_hi, float(np.nanmax(mean + std)))
        band_lo = min(band_lo, float(np.nanmin(mean - std)))
        ax.fill_between(x, mean - std, mean + std, color=color, alpha=0.15, linewidth=0)
        ax.plot(x, mean, color=color, linestyle="-", linewidth=2.6,
                label=f"{disp} — inferred $D_{{mix}}$ = {summaries[disp]['dmix']:.3f} $\\mu$m$^2$/s")

    if not np.isfinite(band_hi) or not np.isfinite(band_lo):
        band_hi, band_lo = 1.0, 0.0
    span = band_hi - band_lo
    pad = 0.15 * span if span > 0 else (0.1 * abs(band_hi) + 1e-9)
    ax.set_ylim(max(0.0, band_lo - pad), band_hi + pad)   # diffusivities are non-negative
    ax.yaxis.set_major_locator(MaxNLocator(nbins=10))
    ax.yaxis.set_minor_locator(AutoMinorLocator(2))

    ax.set_xlabel("time [s]")
    ax.set_xlim(x[0], x[-1])
    ax.set_xticks(x)
    ax.set_ylabel("mobile diffusivity [$\\mu$m$^2$/s]")
    ax.set_title("Inferred mobile diffusivity vs experiment — per receptor\n"
                 "inferred $D_\\mathrm{mix\\,mobile}$ (trajectory) vs experimental $D_{mobile}$ (flat reference)",
                 fontsize=10)
    ax.legend(fontsize=8, framealpha=0.9, loc="best")
    fig.tight_layout()
    return fig, summaries


def _write_report(fig_dir, meta, results):
    """Write a self-contained interpretation report (report.md) beside the figures."""
    displays = meta["displays"]
    dmix = meta["dmix"]
    L = []
    L.append(f"# Control-receptor temporal dynamics — {meta['timing_label']}")
    L.append("")
    L.append(f"Generated by "
             f"`SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls_Temporal_Dynamics.py` "
             f"from `{meta['npz_name']}`. **{meta['n_estimates']} MAP estimates** "
             f"= {meta['n_kinds']} conditions × {meta['n_cells']} cells × {meta['n_chunks']} "
             f"non-overlapping {meta['step']:g} s windows. Time points: "
             f"{', '.join(f'{t:g}' for t in meta['x'])} s. Conditions: "
             f"{', '.join(displays)} (raw kinds {', '.join(meta['kinds'])}).")
    L.append("")
    L.append("Ad-hoc reuse of the DIMER-ALP MET posterior on the CD86 / CTLA-4 control "
             "receptors. The transferable observable across the label / model mismatch "
             "is the diffusion coefficient; counts and rates are not read quantitatively "
             "here (see Caveats).")
    L.append("")
    L.append("## Diffusion readout vs experiment (headline)")
    L.append("")
    L.append("`dmix_mobile_temporal.png` compares, per receptor, the inferred "
             "count-weighted mobile mixture diffusivity "
             "**D_mix_mobile = (C_A·D_A + C_B·D_B) / (C_A + C_B)** (µm²/s, in the condition "
             "color) against the experimental mobile-fraction diffusion coefficient "
             "D_mobile (in a separate per-condition color). D_A (monomer), D_B = R_B·D_A "
             "(mobile dimer) — which bracket D_mix in either order — and the mobile split "
             "f_B = C_B / (C_A + C_B) are given in the table below rather than on the "
             "figure. D_A is stored in µm²/s, so the comparison needs no unit conversion, "
             "and the acquisition geometry matches (50 FPS, 256² px, ~157 nm/px).")
    L.append("")
    L.append("| Condition | D_mix_mobile | D_A (monomer) | D_B (mobile dimer) | f_B | "
             "Experimental D_mobile |")
    L.append("|---|---|---|---|---|---|")
    for k, d in zip(meta["kinds"], displays):
        s = dmix.get(d, {})
        ref = EXPERIMENTAL_D_MOBILE.get(k)
        refstr = "—" if ref is None else f"{ref['value']:.3f} ± {ref['sd']:.3f} µm²/s"
        L.append(f"| {d} | {s.get('dmix', float('nan')):.3f} | {s.get('d_a', float('nan')):.3f} "
                 f"| {s.get('d_b', float('nan')):.3f} | {s.get('f_b', float('nan')):.2f} | {refstr} |")
    L.append("")
    # Flag conditions whose MAP left the training prior (unrestricted pooling on OOD data).
    da_lo, da_hi = _prior_bounds("diffusivity_alp")
    rb_lo, rb_hi = _prior_bounds("relative_diffusivity_bet")
    flags = []
    for k, d in zip(meta["kinds"], displays):
        s = dmix.get(d, {})
        da = s.get("d_a")
        rb = (s["d_b"] / s["d_a"]) if s.get("d_a") else None
        out = []
        if da is not None and not (da_lo <= da <= da_hi):
            out.append(f"D_A = {da:.3f} outside [{da_lo:.3g}, {da_hi:.3g}]")
        if rb is not None and not (rb_lo <= rb <= rb_hi):
            out.append(f"R_B = {rb:.2f} outside [{rb_lo:.3g}, {rb_hi:.3g}]")
        if out:
            flags.append(f"**{d}** — " + "; ".join(out))
    if flags:
        L.append("**MAP outside the training prior** (unrestricted pooling on "
                 "out-of-distribution data lets the mode leave the prior box): "
                 + "  •  ".join(flags) + ". For these conditions D_A / R_B are not "
                 "physically constrained; the count-weighted D_mix stays the robust "
                 "readout, but read it as bracketed, not a resolved mixture.")
        L.append("")
    L.append("All diffusivities in µm²/s (time-averaged over cells × windows). Read "
             "D_mix_mobile as a point between D_A and D_B (in either order); treat it as "
             "a resolved monomer / dimer mixture only where f_B has an informative "
             "(non-edge) posterior. These controls bracket the experimental scale, but "
             "this readout does not by itself resolve monomer (CD86) from dimer (CTLA-4).")
    L.append("")
    L.append("## How to read the per-parameter figures")
    L.append("")
    L.append("Each `<key>_temporal.png` tracks one parameter over the recording: the "
             "MAP estimate is computed independently in every non-overlapping window of "
             "every cell, and plotted per condition (solid = mean over cells; shaded = "
             "mean ± 1 SD across cells; faint = individual cell trajectories).")
    L.append("")
    L.append("- **Temporal dynamics.** How the inferred parameter behaves across the "
             "recording — a resolution a single whole-recording estimate cannot provide.")
    L.append("- **Robustness / stationarity.** A flat trajectory for a constant property "
             "is evidence of a time-invariant, self-consistent estimate; a trend is "
             "either real dynamics or an acquisition confound (e.g. blinking / "
             "photobleaching pulling the apparent counts down over time).")
    L.append("- **Reliability = recovery × stationarity.** Trust a parameter when it "
             "both recovers well on ground-truth EVAL data (annotated on each figure, a "
             "property of the posterior) and is stationary where it should be. A "
             "parameter that recovers poorly (e.g. R_ON) carries no signal.")
    L.append("")
    L.append("## Per-parameter summary (this run)")
    L.append("")
    header = "| Parameter | " + " | ".join(f"{d} time-avg" for d in displays) + \
             " | EVAL recovery ±0.3 |"
    L.append(header)
    L.append("|" + "---|" * (1 + len(displays) + 1))
    for r in results:
        tavgs = " | ".join(f"{r['time_avgs'].get(d, float('nan')):.3g}" for d in displays)
        rec = "—" if r["recovery"] is None else f"{r['recovery'] * 100:.0f}%"
        L.append(f"| {r['name']} ({r['label']}) | {tavgs} | {rec} |")
    L.append("")
    L.append("## Caveats")
    L.append("")
    L.append("- **Label / model mismatch.** The posterior was trained on always-visible "
             "permanent-label emitters; these controls use an exchangeable, blinking "
             "SiR-S5 HaloTag probe. Diffusion (D) is a per-track property and transfers; "
             "counts and rates (C_*, κ_*, R_ON) depend on co-visible-emitter numbers and "
             "track continuity and are **not** read quantitatively here.")
    L.append("- **Not the trained mechanism.** CD86 (monomer) and CTLA-4 (dimer) are "
             "constitutive oligomeric-state controls, not the dynamic A + A ⇌ B ⇌ C "
             "dimerization mechanism the posterior encodes. They bracket the diffusion "
             "scale and stress-test transferability; they do not exercise the kinetics.")
    L.append("- **Why the D comparison is still quantitative.** The acquisition geometry "
             "matches the training regime (50 FPS, 256² px, ~157 nm/px) and D_A is stored "
             "in µm²/s, so D_mix_mobile is directly comparable to the experimental "
             "D_mobile with no conversion.")
    L.append("- **D_mix weights are the fragile quantity.** D_mix is count-weighted and "
             "the counts are label-fragile; if f_B collapses toward 0 or 1, D_mix "
             "approaches D_A or D_B. Read it within the [D_B, D_A] bracket, not as a "
             "resolved mixture, unless f_B is informative.")
    L.append("- **Blinking / photobleaching** drives the downward drift of the count "
             "parameters (fewer visible emitters in later windows) — a real "
             "non-stationarity in a confounded parameter, not receptor loss.")
    L.append("- **First-pass posteriors** (interrupted training) — absolute values will "
             "sharpen with the production posteriors; re-run this analysis on those.")
    L.append("- Relative parameters (R_B, R_C, R_ON) are shown as dimensionless ratios.")
    L.append("- The pooled **posterior-distribution** panels are a documented, "
             "not-yet-implemented extension (they need the full per-window sample pool, "
             "not the stored quantiles).")
    L.append("")
    L.append("## Reference")
    L.append("")
    L.append(f"{EXPERIMENTAL_CITATION} Full method interpretation: the companion "
             "`SRM_AND_SBI_DIMER_ALP_Experiment_CD86_CTLA-4_Controls_Temporal_Dynamics.md`.")
    L.append("")
    (fig_dir / "report.md").write_text("\n".join(L), encoding="utf-8")


def main(args):
    """Resolve the Experiment .npz for the requested duration, then plot every parameter."""
    timing = RunTiming(total_time_seconds=args.total_time_seconds,
                       frames=PARAMETERS.simulation.timing)
    timing_label = timing.label
    data_bank_root = PARAMETERS.machine.data_bank_root
    paths = PARAMETERS.paths

    out_dir = (data_bank_root / paths.posit_subdir /
               CONTROLS_RECOVERY_PATTERN.format(
                   project_alias=paths.project_alias, timing_label=timing_label))
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
    print(f" {paths.project_alias} — Control Experiment Temporal Dynamics (CD86 / CTLA-4)")
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

    with np.load(str(npz_path), allow_pickle=True) as d:
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

    # Headline: count-weighted mobile mixture diffusivity D_mix_mobile per condition,
    # against the experimental mobile-fraction D_mobile (the transferable observable).
    d_mix, d_a, d_b, f_b = _mobile_mixture(abs_grid)
    dmix_fig, dmix_summary = _dmix_figure(d_mix, d_a, d_b, f_b, x, kinds)
    dmix_png = fig_dir / "dmix_mobile_temporal.png"
    dmix_fig.savefig(str(dmix_png), dpi=180)
    plt.close(dmix_fig)
    print(f"  wrote {dmix_png.name}  [headline: D_mix_mobile vs experimental D_mobile]")

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
        "dmix": dmix_summary,
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

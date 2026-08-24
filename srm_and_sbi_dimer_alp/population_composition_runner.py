"""Shared engine for the population-composition analysis.

WHAT IT PRODUCES. One report holding both halves of a single comparison:

  * the **experimental** composition -- relative species abundance inferred from the real recordings,
    per condition, with cell-to-cell error bars, the within-recording time course, per-recording
    spread, and the sensitivity of every headline number to the choices that could have driven it;
  * the **synthetic** validation -- the identical readout computed on held-out videos whose truth is
    known, which is what licenses reading the experimental numbers at all.

Keeping them in one document is the point: the experimental value and the measured error of the
instrument that produced it belong on the same page, and the two are otherwise reported by different
tools with different run stamps.

ONE WORKFLOW. Unlike the mirrored stage runners, this analysis exists for the **biology** workflow
only, and the asymmetry is scientific rather than incidental: the composition is a function of the
three inferred species counts, and the detector workflow infers imaging parameters and no counts at
all -- it treats the population implicitly, as part of what it marginalizes. There is therefore no
detector composition to compute, in the same way the detector's ``Nuisance_DLI`` pool has no biology
counterpart. The engine is nonetheless written in the shared shape -- a ``CompositionSpec`` resolved
once in :func:`_population_composition_spec` -- so that everything workflow-specific stays in one
place and a future workflow carrying species counts plugs in without touching the body.

INPUTS, AND WHAT EACH ONE COSTS IF ABSENT.
  * the Experiment MAP output **with per-window posterior draws** (``--dump-posterior-samples``):
    required. The composition must be formed inside each draw, so the stored marginal quantiles
    cannot substitute -- a fraction built from marginals is a different quantity, not a coarser one.
  * the MAP_Recovery output: optional. Without it the experimental half still runs and the report
    states that the validation column is unavailable, rather than quietly reporting experimental
    numbers with no measured error.

NO GPU, no estimator, no video. It reads two completed outputs and writes a report; it is a post-hoc
analysis in ``Script_Bank/Analysis``, never a pipeline stage and never wired into the dispatcher.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure

from . import population_composition as pc
from .diagnostics import DiagnosticReporter
from .experiment_support import condition_display

# Colors: one per condition, held fixed across every figure so a condition is the same color
# everywhere in the report, and one neutral for the synthetic reference.
CONDITION_COLORS = ("tab:blue", "tab:red")
SYNTHETIC_COLOR = "0.45"


@dataclass(frozen=True)
class CompositionSpec:
    """Everything about this analysis that a workflow supplies."""

    parameter_keys: list
    count_index: tuple                    # theta indices of the three species counts, in A, B, C order
    prior_low: np.ndarray                 # log10 prior lower bounds (all parameters)
    prior_high: np.ndarray                # log10 prior upper bounds (all parameters)
    experiment_npz: object
    recovery_npz: object
    report_dir: object
    alias: str
    timing_label: str
    stage: str
    window_seconds: float


# ---- loading -----------------------------------------------------------------------------------

def load_experiment_draws(path, n_params):
    """Per-window posterior draws and their labels from a completed Experiment stage.

    Fails loud in two distinguishable ways, because the remedies differ: a missing output means the
    stage was never run, while an output without ``posterior_samples_cloud`` means it ran without
    ``--dump-posterior-samples`` and must be re-run. Substituting the stored quantiles would produce
    a confident report about a different quantity.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"the population composition needs the Experiment stage's per-window posterior draws, "
            f"and the Experiment output is absent:\n    {path}\nRun the biology Experiment stage for "
            f"this timing with --dump-posterior-samples first.")
    with np.load(str(path), allow_pickle=False) as d:
        if "posterior_samples_cloud" not in d.files:
            raise SystemExit(
                f"the Experiment output\n    {path}\nholds no 'posterior_samples_cloud'. The "
                f"composition is formed inside each posterior draw, so the stored marginal quantiles "
                f"cannot substitute: a fraction of marginals is a different quantity, not a coarser "
                f"one. Re-run the Experiment stage with --dump-posterior-samples "
                f"(--summary posterior|both).")
        cloud = np.asarray(d["posterior_samples_cloud"], dtype=float)
        labels = dict(kind_index=d["kind_index"].astype(int), cell=d["cell"].astype(int),
                      chunk=d["chunk"].astype(int), kinds=[str(k) for k in d["kinds"]])
    if cloud.shape[-1] != n_params:
        raise ValueError(f"Experiment draws carry {cloud.shape[-1]} parameters but this workflow has "
                         f"{n_params}.")
    return cloud, labels


def load_recovery(path, n_params):
    """Held-out truth and MAP estimates for the synthetic validation half, or ``None`` if absent."""
    if not path.exists():
        return None
    with np.load(str(path), allow_pickle=False) as d:
        if "true_log10" not in d.files or "inferred_log10" not in d.files:
            return None
        true_log10 = np.asarray(d["true_log10"], dtype=float)
        inferred_log10 = np.asarray(d["inferred_log10"], dtype=float)
    if true_log10.shape[1] != n_params:
        raise ValueError(f"MAP_Recovery carries {true_log10.shape[1]} parameters but this workflow "
                         f"has {n_params}.")
    return true_log10, inferred_log10


# ---- figures (object API; each returns a Figure, never saves) -----------------------------------

def _fraction_rows():
    """The fraction quantities, in report order, as ``(index, Quantity)`` pairs."""
    return [(i, pc.COMPOSITION[i]) for i in pc.FRACTION_INDICES]


def _figure_conditions(mean, sem, kinds, recovery_rows):
    """Grouped bars: every fraction, both conditions, with the estimator's in-model error beside."""
    rows = _fraction_rows()
    fig = Figure(figsize=(1.55 * len(rows) + 3.2, 4.8), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(rows))
    width = 0.34
    for k, kind in enumerate(kinds):
        vals = np.array([100.0 * mean[k, i] for i, _q in rows])
        errs = np.array([100.0 * sem[k, i] for i, _q in rows])
        ax.bar(x + (k - 0.5) * width, vals, width=width, yerr=errs, capsize=3,
               color=CONDITION_COLORS[k % len(CONDITION_COLORS)],
               label=f"{condition_display(kind)} (± SEM over recordings)")
    if recovery_rows is not None:
        # The estimator's own error on held-out truth, drawn as a symmetric whisker centered on each
        # experimental bar. It is a DIFFERENT kind of error from the SEM beside it -- in-model
        # measurement error against known truth, versus biological spread across recordings -- and the
        # caption says so. Both belong here: the contrast is only meaningful if the second is small
        # against the first.
        by_key = {r["key"]: r for r in recovery_rows}
        for k in range(len(kinds)):
            for j, (i, q) in enumerate(rows):
                mae = by_key.get(q.key, {}).get("mae")
                if mae is None:
                    continue
                center = 100.0 * mean[k, i]
                ax.plot([x[j] + (k - 0.5) * width] * 2, [center - mae, center + mae],
                        color=SYNTHETIC_COLOR, lw=2.4, solid_capstyle="butt", zorder=5,
                        label="synthetic MAE (held-out truth)" if (k == 0 and j == 0) else None)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{q.symbol}\n{q.formula}" for _i, q in rows], fontsize=8)
    ax.set_ylabel("share of the population (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9, loc="upper center", ncols=3)
    ax.set_title("Inferred population composition per condition")
    return fig


def _figure_recovery(true_comp, inferred_comp, rows, rng, stratum=None, max_points=6000):
    """Held-out truth versus estimate for the monomer and dimer-complex shares."""
    by_key = {r["key"]: r for r in rows}
    show = [pc.COMPOSITION_KEYS.index("monomer_fraction"), pc.DIMER_INDEX]
    n = true_comp.shape[0]
    idx = np.arange(n) if n <= max_points else rng.choice(n, max_points, replace=False)
    fig = Figure(figsize=(9.4, 4.6), layout="constrained")
    axes = fig.subplots(1, len(show), squeeze=False)[0]
    for ax, i in zip(axes, show):
        q = pc.COMPOSITION[i]
        ax.scatter(100.0 * true_comp[idx, i], 100.0 * inferred_comp[idx, i], s=5, c=SYNTHETIC_COLOR,
                   alpha=0.25, edgecolors="none")
        ax.plot([0, 100], [0, 100], color="k", lw=1.2, ls="--", label="exact recovery")
        if stratum is not None and i == pc.DIMER_INDEX and stratum.get("n"):
            ax.axvspan(100.0 * stratum["threshold"], 100.0, color="tab:orange", alpha=0.12,
                       label=f"stratum n={stratum['n']:,}: MAE {stratum['mae']:.1f} pp, "
                             f"bias {stratum['bias']:+.1f} pp")
        r = by_key.get(q.key, {})
        ax.set_title(f"{q.symbol} = {q.formula}\nMAE {r.get('mae', float('nan')):.1f} pp, "
                     f"p95 {r.get('p95', float('nan')):.1f} pp, r = {r.get('r', float('nan')):.3f}",
                     fontsize=10)
        ax.set_xlabel("true share (%)")
        ax.set_ylabel("inferred share (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.legend(frameon=False, fontsize=8, loc="upper left")
    return fig


def _figure_time_course(mean, sem, kinds, window_seconds, quantity_index):
    """The dimer-complex share against window index, per condition."""
    q = pc.COMPOSITION[quantity_index]
    n_chunks = mean.shape[1]
    t = window_seconds * (np.arange(n_chunks) + 0.5)
    fig = Figure(figsize=(7.4, 4.6), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    for k, kind in enumerate(kinds):
        y = 100.0 * mean[k, :, quantity_index]
        e = 100.0 * sem[k, :, quantity_index]
        color = CONDITION_COLORS[k % len(CONDITION_COLORS)]
        ax.plot(t, y, marker="o", ms=5, lw=2, color=color, label=condition_display(kind))
        ax.fill_between(t, y - e, y + e, color=color, alpha=0.18, linewidth=0)
    ax.set_xlabel(f"window center (s), {window_seconds:g} s windows")
    ax.set_ylabel(f"{q.symbol} = {q.formula}  (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Composition across the recording (mean ± SEM over recordings)")
    return fig


def _figure_per_cell(cell_values, kinds, quantity_index, bands, rng):
    """Every recording's value for one quantity, so the condition mean is read against its spread."""
    q = pc.COMPOSITION[quantity_index]
    fig = Figure(figsize=(7.0, 4.6), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    for k, kind in enumerate(kinds):
        row = 100.0 * cell_values[k, :, quantity_index]
        row = row[np.isfinite(row)]
        jitter = (rng.random(row.size) - 0.5) * 0.28
        color = CONDITION_COLORS[k % len(CONDITION_COLORS)]
        ax.scatter(np.full(row.size, k) + jitter, row, s=34, color=color, alpha=0.75,
                   edgecolors="k", linewidths=0.4, label=f"{condition_display(kind)} "
                                                         f"(n={row.size} recordings)")
        ax.plot([k - 0.3, k + 0.3], [row.mean()] * 2, color="k", lw=2.2, zorder=5)
    for band in bands:
        ax.axhline(100.0 * band, color="0.75", lw=1, ls=":")
    ax.set_xticks(np.arange(len(kinds)))
    ax.set_xticklabels([condition_display(k) for k in kinds])
    ax.set_ylabel(f"{q.symbol} = {q.formula}  (%)")
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("Per-recording values (black bar = condition mean; dotted lines = spread bands)")
    return fig


# ---- report ------------------------------------------------------------------------------------

def _pct(value):
    return "-" if not np.isfinite(value) else f"{100.0 * value:.1f}"


def _quantity_name(index):
    """Table label for one quantity: the symbol for a share, spelled out for the total."""
    q = pc.COMPOSITION[int(index)]
    return q.symbol if q.is_fraction else "total complexes"


def _quantity_value(index, value):
    """Format a value in the units of its quantity.

    One formatter for every table, because the alternative is what this replaced: a percent
    formatter applied to the total-complexes row, which multiplied a count by a hundred and printed
    a plausible-looking number two orders of magnitude wrong. A share and a count cannot share a
    formatter, so the quantity decides.
    """
    if not np.isfinite(value):
        return "-"
    return _pct(value) if pc.COMPOSITION[int(index)].is_fraction else f"{value:.0f}"


def _write_report(args, spec, data, rng):
    reporter = DiagnosticReporter(
        stage=spec.stage, enabled=True, dump=True, dump_dir=spec.report_dir,
        run_label=f"{spec.alias}_{spec.timing_label}")
    kinds = data["kinds"]
    rows = _fraction_rows()
    mean, sem, n_cells_used = data["mean"], data["sem"], data["n_cells"]

    reporter.check("posterior_draws_present", True,
                   f"{data['n_windows']} windows x {data['n_draws']} draws x "
                   f"{len(spec.parameter_keys)} parameters")
    reporter.check("composition_closure", data["closure"] < 1e-9,
                   f"max |f_A + f_B + f_C - 1| = {data['closure']:.2e} over every draw")
    reporter.check("no_nan_inf(posterior draws)", data["draws_finite"],
                   "clean" if data["draws_finite"] else "NaN/Inf present in the stored draws")
    reporter.check("replicate_unit_is_recording", True,
                   f"every interval and test below is over recordings "
                   f"({', '.join(f'{condition_display(k)} n={int(n)}' for k, n in zip(kinds, n_cells_used[:, 0]))})"
                   f", never over the {data['n_windows']} windows")

    # -- experimental half -----------------------------------------------------------------------
    reporter.table(
        "Population composition, span-averaged (percent, mean ± SEM over recordings)",
        ["condition"] + [f"{q.symbol} ({q.formula})" for _i, q in rows] + ["total complexes"],
        [[condition_display(kind)]
         + [f"{_pct(mean[k, i])} ± {_pct(sem[k, i])}" for i, _q in rows]
         + [f"{mean[k, pc.TOTAL_INDEX]:.0f} ± {sem[k, pc.TOTAL_INDEX]:.0f}"]
         for k, kind in enumerate(kinds)],
        note="Each value is the mean over recordings of the mean over that recording's windows of "
             "the mean over that window's posterior draws of the fraction -- formed inside each draw, "
             "so the correlations between the three counts are carried through. The error is the "
             "standard error across RECORDINGS, which is cell-to-cell biological spread; it is not "
             "the within-window posterior width, and it is deliberately not divided by the number of "
             "windows, since ten windows of one recording are not ten independent measurements. "
             "The shares f_A + f_B + f_C close to 100% exactly; f_D = 1 - f_A by definition.")

    reporter.table(
        f"Population composition, first window only (0-{spec.window_seconds:g} s)",
        ["condition"] + [f"{q.symbol}" for _i, q in rows],
        [[condition_display(kind)] + [_pct(data["first_mean"][k, i]) for i, _q in rows]
         for k, kind in enumerate(kinds)],
        note="The earliest state the recordings resolve. Reported separately because a span average "
             "over a population that changes during the recording describes no instant of it -- see "
             "the time-course table below for whether that is the case here.")

    if args.bootstrap:
        reporter.table(
            f"Bootstrap check on the reported means ({args.bootstrap:,} resamples of recordings)",
            ["condition", "quantity", "mean %", "normal-theory ± SEM", "bootstrap 95% interval",
             "half-width difference (pp)"],
            [[condition_display(kind), q.symbol, _pct(mean[k, i]),
              f"[{_pct(mean[k, i] - 1.96 * sem[k, i])}, {_pct(mean[k, i] + 1.96 * sem[k, i])}]",
              f"[{_pct(data['boot_lo'][k, i])}, {_pct(data['boot_hi'][k, i])}]",
              f"{abs(100.0 * (0.5 * (data['boot_hi'][k, i] - data['boot_lo'][k, i]) - 1.96 * sem[k, i])):.2f}"]
             for k, kind in enumerate(kinds) for i, q in rows],
            note="A percentile bootstrap over the replicate unit, free of the normal-theory "
                 "assumption behind the SEM. Close agreement means the reported error bars do not "
                 "depend on that assumption; a large difference would say the cell-to-cell "
                 "distribution is skewed enough that the SEM misstates it.")

    reporter.table(
        "Sensitivity: does the headline survive the choices that could have driven it?",
        ["variant", "condition", "f_D %", "shift vs headline (pp)", "draws retained %",
         "windows left empty"],
        [[v["name"], condition_display(kind), _pct(v["mean"][k, pc.DIMER_INDEX]),
          f"{100.0 * (v['mean'][k, pc.DIMER_INDEX] - mean[k, pc.DIMER_INDEX]):+.1f}",
          f"{100.0 * v['retained'][k]:.1f}" if v["retained"] is not None else "-",
          f"{v['empty'][k]} / {v['windows'][k]}" if v["empty"] is not None else "-"]
         for v in data["variants"] for k, kind in enumerate(kinds)],
        note="Two independent challenges to the headline. (1) Support: the posterior places mass "
             "beyond the box the estimator was trained on, most of all in the counts, so the "
             "composition is recomputed from progressively more restricted draws. Because the "
             "composition is a ratio, out-of-support mass may inflate the counts together and "
             "largely cancel -- the table measures whether it does, and at what cost in discarded "
             "windows. (2) Center: the compositional (closed geometric) center replaces the "
             "arithmetic mean, which is the natural center on the simplex. Read the spread of f_D "
             "across these variants as the model-conditional bracket on the value, and lead with the "
             "most conservative one.")

    reporter.table(
        "Per-recording spread of the monomer share",
        ["condition", "recordings", "min %", "median %", "max %",
         f"below {100 * pc.HETEROGENEITY_BANDS[0]:.0f}%", f"above {100 * pc.HETEROGENEITY_BANDS[1]:.0f}%"],
        [[condition_display(kind), h.get("n", 0),
          _pct(h.get("minimum", float("nan"))), _pct(h.get("median", float("nan"))),
          _pct(h.get("maximum", float("nan"))), h.get("n_below", "-"), h.get("n_above", "-")]
         for kind, h in zip(kinds, data["heterogeneity"])],
        note="Whether the condition mean describes a typical recording. Mass at both extremes means "
             "the mean sits between two groups of cells rather than inside one, and any downstream "
             "use should carry the per-recording values (saved in the arrays) rather than the mean "
             "alone.")

    reporter.table(
        "Within-recording time course",
        ["condition", "quantity", "first window", "last window",
         "median per-recording Spearman rho", "first-vs-last signed-rank p", "recordings"],
        [[condition_display(kinds[k]), _quantity_name(qi),
          _quantity_value(qi, data["tc_mean"][k, 0, qi]),
          _quantity_value(qi, data["tc_mean"][k, -1, qi]),
          f"{m[k]['spearman_median']:+.2f}", f"{m[k]['wilcoxon_p']:.3g}", m[k]["n_cells"]]
         for qi, m in data["monotonicity"] for k in range(len(kinds))],
        note="The two endpoint columns are condition MEANS across recordings, matching every other "
             "table here (shares in percent, the total as a count). "
             "Spearman rho is computed per recording against the window index and then median-ed, so "
             "it is a direction and strength robust to a single outlying window; the signed-rank p "
             "pairs each recording's last window against its own first, so between-cell spread "
             "cannot mask a shared trend. A significant p in one condition and not the other is the "
             "informative pattern -- it makes the flat condition a control for whatever is common to "
             "both acquisitions.")

    reporter.table(
        "Condition contrast at the recording level (two-sided rank-sum)",
        ["quantity", f"{condition_display(kinds[0])} median", f"{condition_display(kinds[1])} median",
         "ratio", "p", "n"],
        [[_quantity_name(j), _quantity_value(j, r["median_a"]), _quantity_value(j, r["median_b"]),
          f"{r['ratio']:.2f}x", f"{r['p']:.2g}", f"{r['n_a']} vs {r['n_b']}"]
         for j, r in enumerate(data["contrast"])],
        note="Recording-level, so the p-values carry the replicate count the design actually has. "
             "Read each p beside the synthetic validation table: a contrast is only interpretable "
             "where the same quantity is recovered accurately on held-out truth, and a quantity that "
             "fails to separate here may simply be the least identified one.")

    # -- synthetic half --------------------------------------------------------------------------
    if data["recovery"] is None:
        reporter.stat("synthetic validation", "unavailable",
                      note=f"no MAP_Recovery output at {spec.recovery_npz}; the experimental values "
                           f"above therefore carry no measured in-model error. Run the Evaluation "
                           f"stage for this timing to supply it.")
    else:
        rec = data["recovery"]
        reporter.table(
            f"Synthetic validation: the same readout where the truth is known "
            f"(n = {rec['rows'][0]['n']:,} held-out videos)",
            ["quantity", "definition", "MAE", "p95 abs. error", "bias", "r"],
            [[q.symbol if q.is_fraction else "total complexes", q.formula,
              f"{r['mae']:.3g} {r['unit']}", f"{r['p95']:.3g} {r['unit']}",
              f"{r['bias']:+.3g} {r['unit']}", f"{r['r']:.4f}"]
             for q, r in zip(pc.COMPOSITION, rec["rows"])],
            note="The identical function of the identical estimator, applied to videos whose "
                 "parameters are known. Fractions are compared in percentage points, the total in "
                 "dex (a count's error is multiplicative). This is POINT-ESTIMATE accuracy: whether "
                 "the posterior INTERVAL of a fraction covers truth at its nominal rate is a "
                 "different question, and it is not computable from a recovery artifact that stores "
                 "only per-parameter marginal quantiles -- it needs joint posterior draws on the "
                 "held-out set. The f_A and f_D rows are necessarily identical, since f_D = 1 - f_A "
                 "makes their errors equal in magnitude; the agreement is an internal check, not two "
                 "independent results.")
        reporter.table(
            "Why the composition is better identified than the counts it is built from",
            ["quantity", "MAE (dex)", "r"],
            [[r["key"], f"{r['mae']:.4f}", f"{r['r']:.4f}"] for r in rec["parts"]],
            note="Individual species counts against their sum, both from the same posterior and the "
                 "same videos. The sum being recovered far better than any part means the parts' "
                 "errors are strongly anti-correlated -- they trade off inside the posterior -- and "
                 "any quantity that divides one part by the total inherits that cancellation. This "
                 "is the whole reason a composition can be reported from counts that individually "
                 "cannot be.")
        if rec["stratum"].get("n"):
            reporter.stat(
                "recovery in the dimer-rich region", f"MAE {rec['stratum']['mae']:.1f} pp, "
                f"bias {rec['stratum']['bias']:+.1f} pp (n = {rec['stratum']['n']:,})",
                note=f"restricted to held-out videos whose TRUE f_D exceeds "
                     f"{100 * rec['stratum']['threshold']:.0f}% -- the corner the activated condition "
                     f"occupies. Selection is on truth, never on the estimate, so it is independent "
                     f"of the estimator's own error. A mean over the whole prior could hide a weak "
                     f"corner; this says whether the relevant one is weak.")

    # -- provenance ------------------------------------------------------------------------------
    reporter.stat("experimental source", str(spec.experiment_npz))
    reporter.stat("synthetic source", str(spec.recovery_npz) if data["recovery"] is not None
                  else "absent")
    reporter.stat("windows analyzed", f"{data['n_windows']} "
                  f"({len(kinds)} conditions x {data['n_cells_grid']} recordings x "
                  f"{data['n_chunks']} windows)")
    reporter.stat("posterior draws per window", f"{data['n_draws']:,}")
    reporter.stat("species mapping", "A = monomer, B = mobile dimer, C = immobile dimer",
                  note="the three counted states of the reaction-diffusion model; a dimer holds two "
                       "receptors, which is what separates f_D (a share of complexes) from f_R (a "
                       "share of receptors)")
    reporter.stat("space", "absolute counts (10**theta), fractions formed per draw")

    # -- figures ---------------------------------------------------------------------------------
    rec_rows = None if data["recovery"] is None else data["recovery"]["rows"]
    reporter.save_figure(
        "composition_conditions", _figure_conditions(mean, sem, kinds, rec_rows),
        caption="Inferred composition per condition. Colored bars are the condition means with the "
                "standard error across recordings; the grey whisker on each bar is the estimator's "
                "mean absolute error for that quantity on held-out synthetic videos with known "
                "truth. The two are different kinds of error on purpose -- biological spread across "
                "cells against in-model measurement error -- and the comparison worth reading is "
                "the size of the between-condition difference against the grey whisker."
                if rec_rows is not None else
                "Inferred composition per condition, with the standard error across recordings. No "
                "synthetic validation was available for this run, so no in-model error is drawn.")
    if data["recovery"] is not None:
        reporter.save_figure(
            "composition_recovery",
            _figure_recovery(data["recovery"]["true_comp"], data["recovery"]["inferred_comp"],
                             data["recovery"]["rows"], rng, data["recovery"]["stratum"]),
            caption="Held-out synthetic videos: the true share against the inferred share, for the "
                    "monomer and the dimer-complex fraction. Scatter about the dashed line is the "
                    "in-model error of the readout; the shaded band on the right panel is the "
                    "dimer-rich region the activated condition occupies, with its own error "
                    "restricted to that region.")
    reporter.save_figure(
        "composition_time_course",
        _figure_time_course(data["tc_mean"], data["tc_sem"], kinds, spec.window_seconds,
                            pc.DIMER_INDEX),
        caption="The dimer-complex share across the recording, per condition (mean over recordings, "
                "band = SEM). A condition that changes while the other stays flat cannot be "
                "explained by anything shared between the two acquisitions, which is what makes the "
                "flat condition a control here.")
    reporter.save_figure(
        "composition_per_cell",
        _figure_per_cell(data["cell_values"], kinds, pc.COMPOSITION_KEYS.index("monomer_fraction"),
                         pc.HETEROGENEITY_BANDS, rng),
        caption="Every recording's monomer share, so the condition mean is read against its spread. "
                "Mass at both extremes means the mean falls between two groups of cells rather than "
                "describing a typical one.")

    reporter.summary()
    reporter.write_report()
    return spec.report_dir


# ---- engine ------------------------------------------------------------------------------------

def run_population_composition(cfg, args):
    """Shared entry point. ``cfg`` is a WorkflowConfig; ``args`` the parsed CLI namespace."""
    spec = _population_composition_spec(cfg, args)

    div = "=" * 78
    print(div)
    print(f" {spec.alias} — Experiment Population Composition ({cfg.tag})")
    print(f" timing_label : {spec.timing_label}   window : {spec.window_seconds:g} s")
    print(f" reads        : {spec.experiment_npz}")
    print(f"                {spec.recovery_npz}  (synthetic validation)")
    print(f" writes       : {spec.report_dir}")
    print(div)

    if args.dry_run:
        print("\n[DRY RUN] input validation only:")
        print(f"  experiment .npz : {spec.experiment_npz}  "
              f"[{'OK' if spec.experiment_npz.exists() else 'MISSING'}]")
        print(f"  recovery .npz   : {spec.recovery_npz}  "
              f"[{'OK' if spec.recovery_npz.exists() else 'absent (validation half skipped)'}]")
        print(f"  species counts  : "
              f"{', '.join(spec.parameter_keys[i] for i in spec.count_index)}")
        print(f"  quantities      : "
              f"{', '.join(q.symbol + ' = ' + q.formula for q in pc.COMPOSITION)}")
        print(f"  bootstrap       : {args.bootstrap:,} resamples of recordings"
              if args.bootstrap else "  bootstrap       : off")
        print("[DRY RUN] nothing read, nothing written.\n")
        return 0

    rng = np.random.default_rng(args.seed)
    cloud, labels = load_experiment_draws(spec.experiment_npz, len(spec.parameter_keys))
    kinds = labels["kinds"]
    n_kinds = len(kinds)

    window_comp = pc.window_composition(cloud, spec.count_index)
    grid, n_cells_grid, n_chunks = pc.composition_grid(
        window_comp, labels["kind_index"], labels["cell"], labels["chunk"], n_kinds)
    cell_values = pc.per_cell(grid)
    mean, sem, n_cells = pc.condition_summary(cell_values)
    first_mean, _first_sem = pc.window_slice_summary(grid, 0)
    tc_mean, tc_sem = pc.time_course(grid)
    boot_lo, boot_hi = pc.bootstrap_interval(cell_values, args.bootstrap, rng)

    # Sensitivity variants: the support restrictions recompute the whole ladder from a restricted
    # draw set; the compositional center re-summarizes the same per-recording values a different way.
    variants = []
    for name, mask, note in pc.support_masks(cloud, spec.prior_low, spec.prior_high,
                                             spec.count_index):
        wc = pc.window_composition(cloud, spec.count_index, draw_mask=mask)
        g, _c, _t = pc.composition_grid(wc, labels["kind_index"], labels["cell"], labels["chunk"],
                                        n_kinds)
        cv = pc.per_cell(g)
        m, _s, _n = pc.condition_summary(cv)
        retained, empty, windows = [], [], []
        for e in range(n_kinds):
            sel = labels["kind_index"] == e
            retained.append(float(mask[sel].mean()) if sel.any() else float("nan"))
            empty.append(int(np.isnan(wc[sel, pc.DIMER_INDEX]).sum()))
            windows.append(int(sel.sum()))
        variants.append(dict(name=name, note=note, mean=m, retained=np.array(retained),
                             empty=empty, windows=windows))
    variants.append(dict(name="compositional center (closed geometric mean)",
                         note="the natural center on the simplex, over the same recordings",
                         mean=pc.compositional_center(cell_values), retained=None, empty=None,
                         windows=None))

    monotonic = [(pc.DIMER_INDEX, pc.monotonicity(grid, pc.DIMER_INDEX)),
                 (pc.TOTAL_INDEX, pc.monotonicity(grid, pc.TOTAL_INDEX))]
    contrast = pc.condition_contrast(cell_values) if n_kinds >= 2 else []

    recovery = None
    loaded = load_recovery(spec.recovery_npz, len(spec.parameter_keys))
    if loaded is not None:
        true_log10, inferred_log10 = loaded
        rows, true_comp, inf_comp = pc.recovery_statistics(true_log10, inferred_log10,
                                                           spec.count_index)
        recovery = dict(
            rows=rows, true_comp=true_comp, inferred_comp=inf_comp,
            parts=pc.parts_versus_whole(true_log10, inferred_log10, spec.count_index),
            stratum=pc.recovery_stratum(true_comp, inf_comp, pc.DIMER_INDEX, args.stratum_threshold))

    data = dict(
        kinds=kinds, mean=mean, sem=sem, n_cells=n_cells, first_mean=first_mean,
        tc_mean=tc_mean, tc_sem=tc_sem, cell_values=cell_values, boot_lo=boot_lo, boot_hi=boot_hi,
        variants=variants, monotonicity=monotonic, contrast=contrast, recovery=recovery,
        heterogeneity=pc.heterogeneity(cell_values, pc.COMPOSITION_KEYS.index("monomer_fraction")),
        closure=pc.closure_residual(pc.composition(10.0 ** cloud[..., list(spec.count_index)])),
        draws_finite=bool(np.isfinite(cloud).all()),
        n_windows=int(cloud.shape[0]), n_draws=int(cloud.shape[1]),
        n_cells_grid=int(n_cells_grid), n_chunks=int(n_chunks))

    print(f"\nLoaded {data['n_windows']} windows x {data['n_draws']:,} draws: {n_kinds} conditions "
          f"x {n_cells_grid} recordings x {n_chunks} windows.")
    for k, kind in enumerate(kinds):
        print(f"  {condition_display(kind):9s}: "
              + ", ".join(f"{pc.COMPOSITION[i].symbol} {_pct(mean[k, i])}%" for i in pc.FRACTION_INDICES)
              + f", total {mean[k, pc.TOTAL_INDEX]:.0f}")
    print(f"  synthetic validation : "
          + ("off (no MAP_Recovery)" if recovery is None else
             f"on — f_D MAE {recovery['rows'][pc.DIMER_INDEX]['mae']:.1f} pp over "
             f"{recovery['rows'][0]['n']:,} held-out videos"))
    print(f"  sensitivity variants : {', '.join(v['name'] for v in variants)}\n")

    spec.report_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(spec.report_dir / f"{spec.report_dir.name}.npz"),
        window_composition=window_comp.astype(np.float32),
        cell_composition=cell_values.astype(np.float32),
        condition_mean=mean.astype(np.float32), condition_sem=sem.astype(np.float32),
        time_course_mean=tc_mean.astype(np.float32), time_course_sem=tc_sem.astype(np.float32),
        kind_index=labels["kind_index"], cell=labels["cell"], chunk=labels["chunk"],
        kinds=np.array(kinds), quantities=np.array(pc.COMPOSITION_KEYS))
    report_dir = _write_report(args, spec, data, rng)
    print(f"Report: {report_dir}")
    return 0


def build_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--total-time-seconds", type=float, required=True,
                   help="model window / recording duration; sets the timing label locating the "
                        "inputs and naming the outputs.")
    p.add_argument("--bootstrap", type=int, default=10000,
                   help="resamples for the percentile bootstrap over recordings, which checks the "
                        "reported standard errors against a distribution-free interval "
                        "(0 = skip).")
    p.add_argument("--stratum-threshold", type=float, default=0.7,
                   help="true dimer-complex fraction above which the synthetic recovery is "
                        "additionally reported, so the accuracy of the region the activated "
                        "condition occupies is stated separately from the prior-wide mean.")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the bootstrap resampling and the figure jitter/subsampling "
                        "(every reported estimate is deterministic without it).")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve paths and report what would be read/written; load nothing, "
                        "compute nothing.")
    return p


# ---- the one place workflow specifics live -----------------------------------------------------

def _population_composition_spec(cfg, args) -> CompositionSpec:
    """Resolve the workflow-specific half of the analysis.

    Raises for a workflow that infers no species counts, which is the honest outcome: the detector
    workflow treats the population implicitly and has nothing to compose, so a composition computed
    from its six imaging coordinates would be a well-formatted meaningless number.
    """
    from .parameterization import PARAMETERS, RunTiming
    from .workflow import parameter_keys as _wf_keys

    keys = _wf_keys(cfg)
    count_keys = ("count_alp", "count_bet", "count_chi")
    missing = [k for k in count_keys if k not in keys]
    if missing:
        raise SystemExit(
            f"the population composition needs the three species-count parameters "
            f"{count_keys}, and the {cfg.tag} workflow does not infer {missing}. This analysis is "
            f"biology-only by construction: the detector workflow infers imaging parameters and "
            f"treats the population implicitly, so it has no composition to report.")

    timing = RunTiming(total_time_seconds=args.total_time_seconds,
                       frames=PARAMETERS.simulation.timing)
    data_bank_root = PARAMETERS.machine.data_bank_root
    paths = cfg.paths
    exp_dir = paths.experiment_recovery_dir(data_bank_root, timing.label)
    rec_dir = paths.map_recovery_dir(data_bank_root, timing.label)
    posit_dir = data_bank_root / paths.posit_subdir
    alias = paths.project_alias
    return CompositionSpec(
        parameter_keys=keys,
        count_index=tuple(keys.index(k) for k in count_keys),
        prior_low=np.array(cfg.param_module.theta_lower_bound(), dtype=float),
        prior_high=np.array(cfg.param_module.theta_upper_bound(), dtype=float),
        experiment_npz=exp_dir / (exp_dir.name + ".npz"),
        recovery_npz=rec_dir / (rec_dir.name + ".npz"),
        report_dir=posit_dir / f"{alias}_{timing.label}_Experiment_Population_Composition",
        alias=alias, timing_label=timing.label,
        stage="Experiment_Population_Composition",
        window_seconds=float(args.total_time_seconds),
    )

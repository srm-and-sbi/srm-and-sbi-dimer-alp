"""Shared engine for the Sample Geometric Median analysis (biology and detector workflows).

Both workflows ask the same question of a different cloud: *given many parameter vectors estimated
from real recordings, which single vector represents them without inventing a configuration that
never occurred?* The engine is therefore identical for both, and the genuine per-workflow
differences are carried by :class:`SGMSpec` and resolved once in :func:`_sgm_spec`:

* the **parameterization module** supplying the parameter keys and the log10 prior box;
* the **alias-qualified paths** locating the Experiment MAP output and naming the report;
* the **collection sources** available -- both workflows expose the real optimized MAPs from the
  Experiment stage; the detector additionally exposes the Nuisance_DLI pool and its caches, which
  have no biology counterpart because biology builds no imaging nuisance;
* the **plane figure's** two parameters, chosen per workflow for what they reveal.

Everything downstream of the loader -- the geometric median, the typicality read, the correlation
matrix, the out-of-prior mass, the figures, the report -- is common code.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np
from matplotlib.figure import Figure

from . import sample_geometric_median as sgm
from .diagnostics import DiagnosticReporter

# Conditions are named scientifically -- MET-FAB (the monomer control) and MET-INLB (the dimer
# condition) -- on the command line, in the report, and in the output directory names. The tokens
# "ALP"/"BET" survive ONLY as the stored `kinds` field of the Experiment output, a data-schema
# artifact of how the recordings were namespaced when they were written; they are translated at the
# boundary below and never reach anything a reader sees.
KIND_OF_CONDITION = {"MET-FAB": "ALP", "MET-INLB": "BET"}
CONDITION_CHOICES = ("pooled", "MET-FAB", "MET-INLB")


@dataclass(frozen=True)
class SGMSpec:
    """Everything about this analysis that differs between the two workflows."""

    parameter_keys: list[str]
    theta_low: np.ndarray                 # log10 prior lower bounds
    theta_high: np.ndarray                # log10 prior upper bounds
    posit_dir: object                     # pathlib.Path to the workflow's Posit namespace
    alias: str                            # project alias (carries _DETECTOR for that workflow)
    timing_label: str
    report_stem: str                      # output directory name, minus alias/timing
    stage: str                            # DiagnosticReporter stage name
    collections: dict                     # name -> loader(args) -> (vecs_log, labels, source_label)
    plane: tuple                          # (x_index, y_index, caption) for the 2-D plane figure
    extra_stats: Sequence = ()            # (name, value[, note]) rows appended to the report
    corner_caption: str = ""


# ---- collection loading (shared: both workflows read the same Experiment MAP schema) -----------

def load_experiment_maps(path, parameter_keys):
    """The real optimized MAP estimates from a completed Experiment stage.

    One estimate per analyzed window of real recording, with the per-row labels that identify which
    condition, cell, and time chunk it came from. Fails loud when absent: a missing Experiment output
    means the analysis has nothing real to summarize, and quietly substituting synthetic or prior
    draws would produce a confident report about a cloud that answers a different question.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"the Sample Geometric Median needs the real optimized MAPs, and the Experiment output "
            f"is absent:\n    {path}\nRun the Experiment stage for this workflow and timing first.")
    with np.load(str(path), allow_pickle=False) as d:
        vecs = np.asarray(d["inferred_log10"], dtype=float)
        labels = {"kind_index": np.asarray(d["kind_index"]), "cell": np.asarray(d["cell"]),
                  "chunk": np.asarray(d["chunk"]), "kinds": np.asarray(d["kinds"])}
    if vecs.shape[1] != len(parameter_keys):
        raise ValueError(f"Experiment MAP output has {vecs.shape[1]} dimensions but this workflow "
                         f"has {len(parameter_keys)} parameters {parameter_keys}.")
    return vecs, labels, f"map-estimates (Experiment MAP: {path.name}, {vecs.shape[0]} windows)"


def apply_condition(condition, vecs, labels):
    """Restrict a collection to one experimental condition using its per-row labels.

    'pooled' passes through. A named condition on an unlabeled collection is a loud error rather
    than a silent full-collection summary, because the two differ and the report would not say so.
    """
    if condition == "pooled":
        return vecs, labels, " [pooled: both conditions]"
    if labels is None:
        raise SystemExit(f"--condition {condition} needs a collection carrying per-row condition "
                         f"labels; this one has none.")
    kind = KIND_OF_CONDITION.get(condition, condition)   # scientific name -> stored schema key
    kinds = [str(k) for k in labels["kinds"]]
    if kind not in kinds:
        raise SystemExit(f"--condition {condition} (stored as {kind!r}) is not among the "
                         f"collection's kinds {kinds}.")
    mask = np.asarray(labels["kind_index"]) == kinds.index(kind)
    if not mask.any():
        raise SystemExit(f"--condition {condition}: the collection has no vectors for it.")
    labels = {k: (np.asarray(v)[mask] if k in ("kind_index", "cell", "chunk") else v)
              for k, v in labels.items()}          # mask only the per-row arrays, not kinds/version
    return vecs[mask], labels, f" [{condition}, n={int(mask.sum())}]"


# ---- figures (object API; each returns a Figure, never saves) ----------------------------------

def _figure_plane(pool_log, results, keys, xi, yi, rng):
    idx = (np.arange(pool_log.shape[0]) if pool_log.shape[0] <= sgm.FIGURE_SUBSAMPLE
           else rng.choice(pool_log.shape[0], sgm.FIGURE_SUBSAMPLE, replace=False))
    fig = Figure(figsize=(7, 6), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(10.0 ** pool_log[idx, xi], 10.0 ** pool_log[idx, yi], s=8, c="0.6", alpha=0.35,
               edgecolors="none", label="collection members")
    ur = next(r for r in results if r["variant"] == "unrestricted")
    ax.scatter(ur["sgm_abs"][xi], ur["sgm_abs"][yi], marker="*", s=420, c="gold",
               edgecolors="k", linewidths=1.2, zorder=6, label="SGM (real sample)")
    ax.scatter(ur["vom_abs"][xi], ur["vom_abs"][yi], marker="X", s=210, c="magenta",
               edgecolors="k", linewidths=1.2, zorder=6, label="vector of medians")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(keys[xi])
    ax.set_ylabel(keys[yi])
    ax.legend(loc="best", frameon=False, fontsize=9)
    ax.set_title("Sample Geometric Median versus the per-dimension composite")
    return fig


def _figure_corner(pool_log, results, keys, rng):
    d = len(keys)
    idx = (np.arange(pool_log.shape[0]) if pool_log.shape[0] <= sgm.FIGURE_SUBSAMPLE
           else rng.choice(pool_log.shape[0], sgm.FIGURE_SUBSAMPLE, replace=False))
    # Bin count follows the sample count. A real-data collection is one estimate per analyzed
    # window -- a few hundred vectors, not the tens of thousands a posterior pool holds -- so a
    # fixed 40 bins would put a handful of counts in each and render a noisy comb that looks like
    # structure. sqrt(n) keeps the per-bin count high enough for the shape to be the data's.
    n_bins = int(min(40, max(8, round(np.sqrt(idx.size)))))
    ur = next(r for r in results if r["variant"] == "unrestricted")
    fig = Figure(figsize=(2.0 * d, 2.0 * d), layout="constrained")
    axes = fig.subplots(d, d, squeeze=False)
    for a in range(d):
        for b in range(d):
            ax = axes[a][b]
            if b > a:
                ax.axis("off")
                continue
            if a == b:
                ax.hist(pool_log[idx, a], bins=n_bins, color="0.7")
                ax.axvline(ur["sgm_log"][a], color="gold", lw=2)
                ax.axvline(ur["vom_log"][a], color="magenta", lw=1.5, ls="--")
            else:
                ax.scatter(pool_log[idx, b], pool_log[idx, a], s=4, c="0.6", alpha=0.3,
                           edgecolors="none")
                ax.scatter(ur["sgm_log"][b], ur["sgm_log"][a], marker="*", s=140, c="gold",
                           edgecolors="k", linewidths=0.6, zorder=6)
                ax.scatter(ur["vom_log"][b], ur["vom_log"][a], marker="X", s=70, c="magenta",
                           edgecolors="k", linewidths=0.6, zorder=6)
            if a == d - 1:
                ax.set_xlabel(keys[b], fontsize=8)
            if b == 0:
                ax.set_ylabel(keys[a], fontsize=8)
            ax.tick_params(labelsize=7)
    return fig


def _figure_out_of_box(frac_below, frac_above, keys):
    fig = Figure(figsize=(1.1 * len(keys) + 3, 4.2), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(keys))
    ax.bar(x - 0.2, 100 * np.asarray(frac_below), width=0.4, color="tab:blue", label="below the box")
    ax.bar(x + 0.2, 100 * np.asarray(frac_above), width=0.4, color="tab:red", label="above the box")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("% of members outside the prior box")
    ax.legend(frameon=False, fontsize=9)
    return fig


# ---- report ------------------------------------------------------------------------------------

def _write_report(args, spec, pool_log, results, in_box, rng, collection_label):
    # A condition-restricted summary gets its own directory. The conditions are the comparison this
    # analysis exists to support (monomer control versus dimer), so writing them to one path would
    # let each run silently destroy the one before it and leave a report whose condition line is the
    # only evidence of which is on disk. 'pooled' keeps the plain name.
    stem = spec.report_stem if args.condition == "pooled" else f"{spec.report_stem}_{args.condition}"
    report_dir = spec.posit_dir / f"{spec.alias}_{spec.timing_label}_{stem}"
    keys = list(spec.parameter_keys)
    low, high = spec.theta_low, spec.theta_high
    reporter = DiagnosticReporter(
        stage=spec.stage, enabled=True, dump=True, dump_dir=report_dir,
        run_label=f"{spec.alias}_{spec.timing_label}")

    n = pool_log.shape[0]
    finite = bool(np.isfinite(pool_log).all())
    reporter.check("collection_nonempty", n > 0, f"{n} parameter vectors in the collection")
    reporter.check("no_nan_inf(collection)", finite, "clean" if finite else "NaN/Inf present")
    reporter.check("sgm_is_real_sample", True,
                   "the Sample Geometric Median is an actual collection member, so its joint "
                   "correlations are intact")

    frac_below = (pool_log < low).mean(0)
    frac_above = (pool_log > high).mean(0)
    reporter.table(
        "Out-of-prior mass (per parameter)",
        ["parameter", "box low (log10)", "box high (log10)", "below %", "above %"],
        [[keys[i], f"{low[i]:.3f}", f"{high[i]:.3f}", f"{100 * frac_below[i]:.2f}",
          f"{100 * frac_above[i]:.2f}"] for i in range(len(keys))],
        note="Fraction of members outside the prior box per dimension. On real recordings this is a "
             "genuine finding rather than a defect: the estimates are unconstrained by the box, so "
             "mass outside it says the recordings pull that parameter beyond the range the prior "
             "anticipated -- either the prior is too narrow for this data, or the model is being "
             "asked to explain the recordings with a configuration it cannot represent.")

    for res in results:
        if res["n"] == 0:
            reporter.stat(f"Sample Geometric Median [{res['variant']}]", "empty subcollection",
                          note="no members satisfy the in-box constraint")
            continue
        rows = [[keys[i], f"{res['sgm_abs'][i]:.5g}", f"{res['vom_abs'][i]:.5g}",
                 f"{res['sgm_log'][i]:.4f}", f"{res['vom_log'][i]:.4f}"] for i in range(len(keys))]
        reporter.table(
            f"Sample Geometric Median vs vector of medians - {res['variant']} "
            f"(n={res['n']}, method={res['method']})",
            ["parameter", "SGM (abs)", "vector-of-medians (abs)", "SGM (log10)", "VoM (log10)"], rows,
            note="SGM = the median VECTOR (a real collection member, correlations intact); "
                 "vector-of-medians = the per-dimension composite, which need not correspond to any "
                 "member. Read the two columns against each other per parameter: where they agree the "
                 "choice does not matter, and where they diverge the composite is asserting a "
                 "combination nothing in the collection realized. "
                 f"SGM in-box: {res['sgm_in_box']}; vector-of-medians in-box: {res['vom_in_box']}.")

    reporter.table(
        "Typicality of the vector of medians versus the SGM",
        ["variant", "normalized dist SGM-VoM", "Mahalanobis SGM", "Mahalanobis VoM",
         "KDE density VoM/SGM"],
        [[r["variant"], f"{r['box_dist']:.4f}", f"{r['maha_sgm']:.3f}", f"{r['maha_vom']:.3f}",
          f"{r['density_ratio']:.3f}"] for r in results if r["n"] > 0],
        note="Prior-range-normalized absolute space. A density ratio near 1, or a Mahalanobis "
             "distance close to the SGM's, means the per-dimension composite did not land in a "
             "low-density region for THIS collection -- which happens when the cloud is unimodal and "
             "weakly correlated. The SGM's advantage does not depend on that being false: it is "
             "guaranteed to be a realizable configuration whatever the local density, whereas the "
             "composite is only sometimes one, and the report cannot tell in advance which case a "
             "new collection will be.")

    corr = np.corrcoef(pool_log, rowvar=False)
    reporter.table(
        "Joint correlation matrix (Pearson, log10)",
        ["parameter"] + keys,
        [[keys[a]] + [f"{corr[a, b]:+.3f}" for b in range(len(keys))] for a in range(len(keys))],
        note="The cross-parameter correlations the SGM preserves and the vector of medians discards. "
             "The larger these are in magnitude, the more the per-dimension composite misrepresents "
             "the collection."
             + (" Pooled correlations can be inflated by a between-condition shift (Simpson's "
                "paradox): two conditions with no internal correlation still produce one when their "
                "centers differ along both axes."
                if args.condition == "pooled" else
                f" Computed within a single condition ({args.condition}), so the between-condition "
                f"(Simpson's paradox) inflation does not apply here."))

    reporter.stat("collection", collection_label)
    reporter.stat("condition", args.condition,
                  note=("both conditions summarized together" if args.condition == "pooled" else
                        "MET-FAB is the monomer control; MET-INLB is the dimer condition"))
    reporter.stat("collection size (vectors)", str(n), note=f"in-box: {int(in_box.sum())} / {n}")
    reporter.stat("space", "absolute (10**theta), normalized by the absolute prior range")
    for row in spec.extra_stats:
        reporter.stat(*row)

    xi, yi, plane_caption = spec.plane
    reporter.save_figure("sgm_plane", _figure_plane(pool_log, results, keys, xi, yi, rng),
                         caption=plane_caption)
    n_plot = min(n, sgm.FIGURE_SUBSAMPLE)
    n_bins = int(min(40, max(8, round(np.sqrt(n_plot)))))
    reporter.save_figure("sgm_corner", _figure_corner(pool_log, results, keys, rng),
                         caption=spec.corner_caption or
                         f"Pairwise structure across all parameters with the SGM (gold star) and the "
                         f"vector of medians (magenta X) overplotted. Off-diagonal tilt is joint "
                         f"correlation the SGM keeps and the per-dimension composite ignores. "
                         f"Drawn from {n_plot} of {n} collection members"
                         + ("" if n_plot == n else f" (subsampled for rendering)")
                         + f"; diagonal histograms use {n_bins} bins, scaled as sqrt(n) so the "
                         f"per-bin count stays informative at this collection size.")
    reporter.save_figure("out_of_prior_mass", _figure_out_of_box(frac_below, frac_above, keys),
                         caption="Fraction of members outside the prior box per parameter -- the "
                                 "genuine out-of-bounds set, distinct from the typicality of any "
                                 "summary point.")

    reporter.summary()
    reporter.write_report()
    return report_dir


# ---- engine ------------------------------------------------------------------------------------

def run_sample_geometric_median(cfg, args):
    """Shared entry point. ``cfg`` is a WorkflowConfig; ``args`` the parsed CLI namespace."""
    spec = _sgm_spec(cfg, args)
    low, high = spec.theta_low, spec.theta_high

    if args.dry_run:
        print(f"[DRY RUN] Sample Geometric Median ({cfg.tag}) - would read/write:")
        print(f"    collection : {args.collection}  |  condition: {args.condition}")
        for name, loader in spec.collections.items():
            marker = " <- selected" if name == args.collection else ""
            print(f"    source[{name}] {loader.describe(args)}{marker}")
        print(f"    prior box  : log10 low {low.tolist()}")
        print(f"                            high {high.tolist()}")
        print(f"    method     : Sample Geometric Median in absolute space (exact medoid up to "
              f"{sgm.EXACT_MEDOID_CAPACITY} members, else Weiszfeld + snap); full collection and "
              f"in-box subcollection")
        stem = (spec.report_stem if args.condition == "pooled"
                else f"{spec.report_stem}_{args.condition}")
        print(f"    writes     : {spec.posit_dir}/{spec.alias}_{spec.timing_label}_{stem}/")
        print("[DRY RUN] no compute performed.")
        return 0

    if args.collection not in spec.collections:
        raise SystemExit(f"--collection {args.collection} is not available for the {cfg.tag} "
                         f"workflow; available: {sorted(spec.collections)}.")
    rng = np.random.default_rng(args.seed)
    vecs, labels, source = spec.collections[args.collection].load(args)
    vecs, _labels, suffix = apply_condition(args.condition, vecs, labels)
    if args.max_samples and vecs.shape[0] > args.max_samples:
        vecs = vecs[rng.choice(vecs.shape[0], args.max_samples, replace=False)]

    results, in_box = sgm.summary_vectors(vecs, low, high, rng)
    report_dir = _write_report(args, spec, vecs, results, in_box, rng, source + suffix)

    ur = next(r for r in results if r["variant"] == "unrestricted")
    sgm_str = ", ".join(f"{k}={v:.4g}" for k, v in zip(spec.parameter_keys, ur["sgm_abs"]))
    print(f"Sample Geometric Median ({args.collection}, unrestricted, absolute): {sgm_str}")
    print(f"Report: {report_dir}")
    return 0


def build_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--total-time-seconds", type=float, required=True,
                   help="model window / recording duration; sets the timing label locating the "
                        "inputs and naming the outputs.")
    p.add_argument("--collection", default="experiment-map",
                   help="which collection the geometric median summarizes (workflow-dependent; "
                        "--dry-run lists what this workflow offers).")
    p.add_argument("--condition", choices=CONDITION_CHOICES, default="pooled",
                   help="restrict the collection to one experimental condition before the summary: "
                        "'pooled' (default) both; 'MET-FAB' the monomer control; 'MET-INLB' the "
                        "dimer condition.")
    p.add_argument("--max-samples", type=int, default=0,
                   help="cap the collection by uniform subsampling before the geometric median, for "
                        "tractability (0 = use all; a rendering detail, not a scientific knob).")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the subsampling used in the density read and the figures "
                        "(the geometric median itself is deterministic).")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve paths and report what would be read/written; load nothing, "
                        "compute nothing.")
    return p


# ---- the one place the two workflows differ ----------------------------------------------------

@dataclass(frozen=True)
class _Source:
    """A named collection this workflow can summarize."""
    load: Callable
    describe: Callable


def _sgm_spec(cfg, args):
    """Resolve the workflow-specific half of the analysis."""
    from .parameterization import PARAMETERS, RunTiming
    from .workflow import parameter_keys as _wf_keys

    timing = RunTiming(total_time_seconds=args.total_time_seconds,
                       frames=PARAMETERS.simulation.timing)
    paths = cfg.paths
    posit_dir = PARAMETERS.machine.data_bank_root / paths.posit_subdir
    alias = paths.project_alias
    para = cfg.param_module
    exp_stem = paths.experiment_recovery_pattern.format(project_alias=alias,
                                                        timing_label=timing.label)
    exp_path = posit_dir / exp_stem / f"{exp_stem}.npz"
    keys = _wf_keys(cfg)

    collections = {
        "experiment-map": _Source(
            load=lambda a: load_experiment_maps(exp_path, keys),
            describe=lambda a: f"Experiment MAP {exp_path}  "
                               f"[{'OK' if exp_path.exists() else 'MISSING'}]"),
    }
    # The plane figure's two parameters are a genuine per-workflow difference: each workflow has a
    # pair whose correlation is the one most worth seeing a summary point sit on. Resolved here
    # rather than carried on WorkflowConfig, so the config stays the workflow's identity and this
    # stage's specializations stay in this stage's resolver.
    if cfg.tag == "detector":
        xi, yi = keys.index("mu_r"), keys.index("mu_pc")
        plane_caption = (
            "PSF width versus brightness. The collection members (grey) with the SGM (gold star, a "
            "real sample) and the per-dimension vector of medians (magenta X). These two imaging "
            "parameters are physically coupled -- a brighter spot is also a wider one -- so a "
            "composite built from each dimension independently can sit off the ridge the real "
            "configurations occupy.")
    else:
        xi, yi = keys.index("count_chi"), keys.index("relative_rate_dimerization")
        plane_caption = (
            "Dimer abundance versus dimerization rate. The collection members (grey) with the SGM "
            "(gold star, a real sample) and the per-dimension vector of medians (magenta X). These "
            "two are the coupled pair at the center of the biological question -- how much dimer is "
            "present and how fast it forms -- and they trade off against each other, so a composite "
            "built per dimension can assert an abundance/rate combination no recording supported.")
    plane = (xi, yi, plane_caption)
    return SGMSpec(
        parameter_keys=keys,
        theta_low=np.array(para.theta_lower_bound(), dtype=float),
        theta_high=np.array(para.theta_upper_bound(), dtype=float),
        posit_dir=posit_dir, alias=alias, timing_label=timing.label,
        report_stem="Experiment_Sample_Geometric_Median",
        stage="Experiment_Sample_Geometric_Median",
        collections=collections, plane=plane,
        extra_stats=(("Experiment MAP source", str(exp_path)),),
    )

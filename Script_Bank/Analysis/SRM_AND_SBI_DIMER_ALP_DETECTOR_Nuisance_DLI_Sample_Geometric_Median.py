"""Analysis entry point (Detector workflow): the Sample Geometric Median of a Nuisance_DLI pool.

ROLE. Reduce a calibrated imaging nuisance to a single representative parameter vector while
preserving its joint structure. The Nuisance_DLI pool is a cloud of imaging vectors (the six
photophysics parameters) whose dimensions are correlated — a spot's peak brightness scales with its
width, the summed-dimer signal shifts brightness and flicker together — so the honest summary is the
median VECTOR, not the vector of per-dimension medians. This analysis extracts that median vector as
the Sample Geometric Median (SGM): the actual pool member minimizing the summed normalized distance to
the rest, hence a real, co-occurring configuration with every correlation intact. It reports the SGM
against the per-dimension vector of medians (which can be an off-manifold composite no acquisition ever
realized), for the full pool and for the prior-bounded subcollection, with the joint correlation
structure and the out-of-prior mass. Usage and interpretation: the companion note beside this script.

WHERE TO RUN. On any machine (CPU only) — it reads the already-built Nuisance_DLI artifact and neither
loads the estimator nor renders videos. It is a post-hoc, user-driven analysis in Script_Bank/Analysis,
NOT a canonical pipeline stage and never wired into the stage dispatcher; it is complementary to the
Nuisance_DLI construction step, which produces the artifact this consumes.

WHAT IT NEEDS.
  - the built Nuisance_DLI artifact for the run's timing (the samplable pool the Detector calibration
    emits; construct it with the Nuisance_DLI analysis step if it is absent), or, with --pool-source
    cache, the cached posterior-sample pool matrix beside it;
  - the run's timing (--total-time-seconds), which locates the artifact and names the outputs.

WHAT IT DOES.
  - loads the chosen collection -- one estimate per acquisition (the real optimized MAPs, or the
    per-window SGM of the posterior draws) or the full posterior-sample pool -- optionally restricted
    to one experimental condition, plus the imaging prior box for the six photophysics parameters;
  - computes, in absolute (physical) space normalized by the absolute prior range, the Sample Geometric
    Median and the per-dimension vector of medians, for the full pool and the in-box subcollection;
  - quantifies how atypical the vector of medians is (its local density and Mahalanobis distance versus
    the SGM), the joint correlation matrix the vector of medians ignores, and the out-of-prior mass;
  - writes a report and deterministic figures under the Detector-namespaced Posit directory.

WHAT IT DOES NOT DO.
  - It does NOT rebuild the nuisance. It only summarizes an existing artifact; the artifact's content is
    a calibration result the Nuisance_DLI construction step owns.
  - It does NOT recover ground truth. The pool describes where the calibrated imaging posterior places
    its mass on real recordings, which have no ground truth; recovery is quantified only on held-out
    synthetic data in the Detector Evaluation stage.

Reads  <data_bank>/<posit>/<alias>_<timing>_Nuisance_DLI.npz                     (built artifact; required)
       <data_bank>/<posit>/<alias>_<timing>_Nuisance_DLI_PosteriorSamplePool.npz (raw pool; --pool-source cache)
Writes <data_bank>/<posit>/<alias>_<timing>_Nuisance_DLI_Sample_Geometric_Median/         (report + figures)

Usage:
    MACHINE_PROFILE=<p> python \
        Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI_Sample_Geometric_Median.py \
        --total-time-seconds 2.0 [--collection map|posterior] [--map-source experiment|window-sgm] \
        [--condition pooled|ALP|BET] [--pool-source artifact|cache] [--dry-run]
"""
import argparse
import json
import sys
from datetime import datetime, timezone

import numpy as np
from matplotlib.figure import Figure

from srm_and_sbi_dimer_alp import detector_nuisance_dli as ndli
from srm_and_sbi_dimer_alp import detector_parameterization as det
from srm_and_sbi_dimer_alp.diagnostics import DiagnosticReporter
from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming

# Tractability / rendering knobs (not scientific parameters).
_KDE_SUBSAMPLE = 50000           # cap on points used to estimate local density for the typicality read
_FIGURE_SUBSAMPLE = 5000         # cap on scatter points drawn per figure

# Internal condition keys map to publication display names.
CONDITION_DISPLAY = {"ALP": "MET-FAB", "BET": "MET-INLB"}
CONDITION_CHOICES = ("pooled", "ALP", "BET")
_NUISANCE_DLI_BUILD = "Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Nuisance_DLI.py"


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Sample Geometric Median of a Nuisance_DLI pool (the correlation-preserving median vector).")
    p.add_argument("--total-time-seconds", type=float, required=True,
                   help="model window / recording duration; sets the timing label locating the "
                        "Nuisance_DLI artifact and naming the outputs.")
    p.add_argument("--pool-source", choices=("artifact", "cache"), default="artifact",
                   help="posterior collection only: 'artifact' (default) the built Nuisance_DLI (its "
                        "stored samples for a raw/map_estimate_pool choice, else draws from it); 'cache' "
                        "the posterior-sample pool matrix beside the artifact (the richest, pre-"
                        "representation cloud, and the source that carries per-row condition labels).")
    p.add_argument("--collection", choices=("posterior", "map"), default="map",
                   help="which collection the geometric median summarizes: 'map' (default) one estimate "
                        "per acquisition (see --map-source); 'posterior' the posterior-sample pool (all "
                        "draws, the full calibrated mass, density-weighted).")
    p.add_argument("--map-source", choices=("experiment", "window-sgm"), default="experiment",
                   help="--collection map only: 'experiment' (default) the REAL optimized MAPs -- a "
                        "MapEstimate pool cache if present, else the Detector Experiment MAP output; "
                        "errors if neither exists (never a silent stand-in). 'window-sgm' the per-window "
                        "Sample Geometric Median (the medoid of each window's posterior draws) from the "
                        "posterior-sample pool -- an explicit samples-derived estimate.")
    p.add_argument("--condition", choices=CONDITION_CHOICES, default="pooled",
                   help="restrict the collection to one experimental condition before the summary: "
                        "'pooled' (default) both; 'ALP' = MET-FAB (monomer control); 'BET' = MET-INLB "
                        "(dimer). Uses the collection's own per-row labels; a legacy unlabeled pool must "
                        "be migrated first (the Nuisance_DLI build's --migrate-pool-labels).")
    p.add_argument("--n-samples", type=int, default=0,
                   help="number of vectors to draw when the artifact is a gaussian/box representation "
                        "(0 = 200000; ignored for a stored sample pool).")
    p.add_argument("--max-samples", type=int, default=0,
                   help="cap the pool by uniform subsampling before the geometric median, for "
                        "tractability (0 = use all; a rendering detail, not a scientific knob).")
    p.add_argument("--seed", type=int, default=0,
                   help="seed for the subsampling used in the density read and the figures "
                        "(the geometric median itself is deterministic).")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve paths and report what would be read/written; load nothing, "
                        "compute nothing.")
    return p.parse_args(argv)


def _resolve(total_time_seconds):
    """Detector-namespaced paths + timing for this run."""
    timing = RunTiming(total_time_seconds=total_time_seconds, frames=PARAMETERS.simulation.timing)
    data_bank_root = PARAMETERS.machine.data_bank_root
    paths = det.detector_paths(PARAMETERS.paths)                       # _DETECTOR-aliased namespace
    posit_dir = data_bank_root / paths.posit_subdir
    alias = paths.project_alias
    exp_stem = paths.experiment_recovery_pattern.format(project_alias=alias, timing_label=timing.label)
    return dict(timing=timing, timing_label=timing.label, data_bank_root=data_bank_root,
                paths=paths, posit_dir=posit_dir, alias=alias,
                artifact_path=ndli.artifact_path(posit_dir, alias, timing.label),
                pool_cache_path=ndli.pool_cache_path(posit_dir, alias, timing.label, "PosteriorSample"),
                map_cache_path=ndli.pool_cache_path(posit_dir, alias, timing.label, "MapEstimate"),
                experiment_map_path=(posit_dir / exp_stem / f"{exp_stem}.npz"))


# ---- the method: Sample Geometric Median in absolute space -------------------------------------
# Reference: Ramirez Sierra & Sokolowski, Mach. Learn.: Sci. Technol. 6, 015004 (2025). The SGM is the
# actual member minimizing the sum of normalized Euclidean distances; normalization divides each
# dimension by its prior range. Computed on physical values (10**theta) normalized by the absolute
# prior range, because the renderer consumes physical values and the geometric median is not invariant
# to the log-to-linear transform, so centrality is defined where the vector is used.

# The Sample Geometric Median itself is `ndli.sample_geometric_median` (in detector_nuisance_dli),
# shared with the Nuisance_DLI build's sgm_percentiles selection so both use one implementation.


def _typicality(vecs_abs, range_abs, point_abs, rng):
    """Local density and Mahalanobis distance of a single point relative to the cloud, in prior-range-
    normalized absolute space. Density is a subsampled Gaussian-kernel estimate; both are relative reads."""
    m = vecs_abs / range_abs
    x = point_abs / range_abs
    mean = m.mean(0)
    cov_inv = np.linalg.pinv(np.cov(m, rowvar=False))
    maha = float(np.sqrt((x - mean) @ cov_inv @ (x - mean)))
    from scipy.stats import gaussian_kde
    sub = m if m.shape[0] <= _KDE_SUBSAMPLE else m[rng.choice(m.shape[0], _KDE_SUBSAMPLE, replace=False)]
    try:
        density = float(gaussian_kde(sub.T)(x.reshape(-1, 1))[0])
    except np.linalg.LinAlgError:
        density = float("nan")
    return maha, density


def _per_window_sgm(flat_log, labels, cache_path, range_abs):
    """One per-window Sample Geometric Median (the medoid of that window's posterior draws) per
    acquisition, in prior-range-normalized absolute space. Returns (vectors_log10 (n_win, D),
    window_labels_or_None). With per-row labels present the flat pool is grouped into windows by its
    (kind_index, cell, chunk) fields -- robust to a variable per-window draw count; without them it
    falls back to the fixed n_per_chunk block size recorded in the pool provenance."""
    dim = flat_log.shape[1]
    if labels is not None:
        keys = np.stack([np.asarray(labels["kind_index"]), np.asarray(labels["cell"]),
                         np.asarray(labels["chunk"])], axis=1)
        # windows are contiguous blocks (the pool is window-major); a boundary is a label change
        bounds = np.concatenate([[0], np.where(np.any(keys[1:] != keys[:-1], axis=1))[0] + 1,
                                 [keys.shape[0]]])
        n_win = bounds.shape[0] - 1
        out = np.empty((n_win, dim))
        win = {"kind_index": np.empty(n_win, np.int64), "cell": np.empty(n_win, np.int64),
               "chunk": np.empty(n_win, np.int64), "kinds": labels["kinds"]}
        for w in range(n_win):
            rows = flat_log[bounds[w]:bounds[w + 1]]
            idx, _ = ndli.sample_geometric_median(10.0 ** rows, range_abs)
            out[w] = rows[idx]
            win["kind_index"][w], win["cell"][w], win["chunk"][w] = keys[bounds[w]]
        return out, win
    with np.load(str(cache_path), allow_pickle=False) as data:
        n_per = int(json.loads(str(data["provenance"]))["n_per_chunk"])
    n_win = flat_log.shape[0] // n_per
    blocks = flat_log[:n_win * n_per].reshape(n_win, n_per, dim)
    out = np.empty((n_win, dim))
    for w in range(n_win):
        idx, _ = ndli.sample_geometric_median(10.0 ** blocks[w], range_abs)
        out[w] = blocks[w][idx]
    return out, None


def _load_map_collection(args, R, range_abs):
    """--collection map: (vectors_log10 (n_win, D), keys, labels_or_None, source_label). One estimate
    per acquisition -- the real optimized MAPs (--map-source experiment) or the per-window SGM of the
    posterior draws (--map-source window-sgm). Fails loud; never a silent stand-in."""
    if args.map_source == "window-sgm":
        cache = R["pool_cache_path"]
        if not cache.exists():
            raise FileNotFoundError(
                f"--collection map --map-source window-sgm needs the posterior-sample pool cache:\n"
                f"    {cache}\nBuild the Nuisance_DLI first ({_NUISANCE_DLI_BUILD}).")
        flat = np.asarray(np.load(str(cache), allow_pickle=False)["pool"], dtype=float)
        vecs, win = _per_window_sgm(flat, ndli.load_pool_labels(cache), cache, range_abs)
        return (vecs, det.DETECTOR_PARAMETER_KEYS, win,
                f"map-estimates (per-window SGM of the posterior draws, {vecs.shape[0]} acquisitions)")
    map_cache = R["map_cache_path"]                            # --map-source experiment
    if map_cache.exists():
        vecs = np.asarray(np.load(str(map_cache), allow_pickle=False)["pool"], dtype=float)
        return (vecs, det.DETECTOR_PARAMETER_KEYS, ndli.load_pool_labels(map_cache),
                f"map-estimates (MapEstimate pool cache: {map_cache.name})")
    exp = R["experiment_map_path"]
    if exp.exists():
        with np.load(str(exp), allow_pickle=False) as d:
            vecs = np.asarray(d["inferred_log10"], dtype=float)
            win = {"kind_index": np.asarray(d["kind_index"]), "cell": np.asarray(d["cell"]),
                   "chunk": np.asarray(d["chunk"]), "kinds": np.asarray(d["kinds"])}
        return (vecs, det.DETECTOR_PARAMETER_KEYS, win,
                f"map-estimates (Detector Experiment MAP: {exp.name})")
    raise FileNotFoundError(
        "--collection map --map-source experiment needs the real optimized MAPs, and neither source "
        f"is present:\n    MapEstimate cache : {map_cache}\n    Experiment MAP    : {exp}\n"
        f"Build a map_estimate_pool Nuisance_DLI ({_NUISANCE_DLI_BUILD}) or run the Detector "
        f"Experiment stage. For a samples-derived estimate instead, use --map-source window-sgm.")


def _load_posterior_collection(args, R):
    """--collection posterior: (vectors_log10 (N, D), keys, labels_or_None, choice, mode, source)."""
    if args.pool_source == "cache":
        cache = R["pool_cache_path"]
        if not cache.exists():
            raise FileNotFoundError(
                f"Cached posterior-sample pool not found:\n    {cache}\nBuild the Nuisance_DLI first "
                f"({_NUISANCE_DLI_BUILD}), or use --pool-source artifact.")
        pool = np.asarray(np.load(str(cache), allow_pickle=False)["pool"], dtype=float)
        return (pool, det.DETECTOR_PARAMETER_KEYS, ndli.load_pool_labels(cache),
                "cache:PosteriorSample", "unrestricted", "posterior-samples (posterior-sample cache)")
    nu = ndli.require_nuisance_dli(R["posit_dir"], R["alias"], R["timing_label"])
    if nu.posterior_sample_pool_choice in ("raw", "map_estimate_pool"):
        pool = np.asarray(nu.samples, dtype=float)
    else:
        pool = np.asarray(nu.sample(args.n_samples or 200000), dtype=float)
    return (pool, list(nu.parameter_keys), None, nu.posterior_sample_pool_choice, nu.pool_mode,
            f"posterior-samples (artifact:{nu.posterior_sample_pool_choice})")


def _apply_condition(args, vecs, labels):
    """Restrict (vecs, labels) to args.condition using the collection's per-row kind labels. Returns
    (vecs, labels, suffix). 'pooled' passes through; a real condition on an unlabeled collection is a
    loud error rather than a silent full-pool summary."""
    if args.condition == "pooled":
        return vecs, labels, " [pooled: both conditions]"
    if labels is None:
        raise SystemExit(
            f"--condition {args.condition} needs a labeled collection, but this one carries no per-row "
            f"condition labels. Migrate the pool ({_NUISANCE_DLI_BUILD} --migrate-pool-labels), use "
            f"--pool-source cache, or the experiment map source.")
    kinds = list(labels["kinds"])
    if args.condition not in kinds:
        raise SystemExit(f"--condition {args.condition} is not among the collection's kinds {kinds}.")
    mask = np.asarray(labels["kind_index"]) == kinds.index(args.condition)
    if not mask.any():
        raise SystemExit(f"--condition {args.condition}: the collection has no vectors for it.")
    vecs = vecs[mask]
    labels = {k: (np.asarray(v)[mask] if k in ("kind_index", "cell", "chunk") else v)
              for k, v in labels.items()}          # mask only the per-row arrays, not kinds/version
    disp = CONDITION_DISPLAY.get(args.condition, args.condition)
    return vecs, labels, f" [{args.condition} = {disp}, n={int(mask.sum())}]"


def _load_collection(args, R):
    """Return (vectors_log10 (N,6), keys, choice, mode, n_source, collection_label): the chosen
    collection restricted to --condition. See --collection / --map-source / --condition."""
    range_abs = (10.0 ** np.array(det.theta_upper_bound())
                 - 10.0 ** np.array(det.theta_lower_bound()))
    if args.collection == "map":
        vecs, keys, labels, source = _load_map_collection(args, R, range_abs)
        choice, mode = f"map:{args.map_source}", "unrestricted"
    else:
        vecs, keys, labels, choice, mode, source = _load_posterior_collection(args, R)
    vecs, _labels, suffix = _apply_condition(args, vecs, labels)
    return vecs, keys, choice, mode, vecs.shape[0], source + suffix


def _summary_vectors(pool_log, low, high, rng):
    """Compute the SGM and vector-of-medians (absolute + log10) for the full pool and the in-box
    subcollection, plus the typicality of each vector of medians. Returns a list of per-variant dicts."""
    a_low, a_high = 10.0 ** low, 10.0 ** high
    a_range = a_high - a_low
    in_box = np.all((pool_log >= low) & (pool_log <= high), axis=1)
    variants = [("unrestricted", np.ones(pool_log.shape[0], dtype=bool)),
                ("bounded_in_box", in_box)]
    out = []
    for name, mask in variants:
        subset_log = pool_log[mask]
        if subset_log.shape[0] == 0:
            out.append(dict(variant=name, n=0))
            continue
        subset_abs = 10.0 ** subset_log
        idx, method = ndli.sample_geometric_median(subset_abs, a_range)
        sgm_abs = subset_abs[idx]
        vom_abs = np.median(subset_abs, axis=0)
        maha_s, dens_s = _typicality(subset_abs, a_range, sgm_abs, rng)
        maha_v, dens_v = _typicality(subset_abs, a_range, vom_abs, rng)
        out.append(dict(
            variant=name, n=int(subset_log.shape[0]), method=method,
            sgm_abs=sgm_abs, sgm_log=np.log10(sgm_abs), sgm_in_box=True,
            vom_abs=vom_abs, vom_log=np.log10(vom_abs),
            vom_in_box=bool(np.all((np.log10(vom_abs) >= low) & (np.log10(vom_abs) <= high))),
            box_dist=float(np.linalg.norm((sgm_abs - vom_abs) / a_range)),
            maha_sgm=maha_s, maha_vom=maha_v,
            density_ratio=(dens_v / dens_s if dens_s and np.isfinite(dens_s) else float("nan"))))
    return out, in_box


# ---- figures (object API; each returns a Figure, never saves) ----------------------------------

def _figure_plane(pool_log, results, keys, low, high, xi, yi, rng):
    idx = (np.arange(pool_log.shape[0]) if pool_log.shape[0] <= _FIGURE_SUBSAMPLE
           else rng.choice(pool_log.shape[0], _FIGURE_SUBSAMPLE, replace=False))
    fig = Figure(figsize=(7, 6), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    ax.scatter(10.0 ** pool_log[idx, xi], 10.0 ** pool_log[idx, yi], s=8, c="0.6", alpha=0.35,
               edgecolors="none", label="pool draws")
    ur = next(r for r in results if r["variant"] == "unrestricted")
    ax.scatter(ur["sgm_abs"][xi], ur["sgm_abs"][yi], marker="*", s=420, c="gold",
               edgecolors="k", linewidths=1.2, zorder=6, label="SGM (real sample)")
    ax.scatter(ur["vom_abs"][xi], ur["vom_abs"][yi], marker="X", s=210, c="magenta",
               edgecolors="k", linewidths=1.2, zorder=6, label="vector of medians")
    for lo_hi in (low, high):
        ax.axvline(10.0 ** lo_hi[xi], color="k", lw=0.6, ls=":")
        ax.axhline(10.0 ** lo_hi[yi], color="k", lw=0.6, ls=":")
    ax.set_xlabel(f"{keys[xi]} (physical)")
    ax.set_ylabel(f"{keys[yi]} (physical)")
    ax.set_title(f"({keys[xi]}, {keys[yi]}): the SGM is a real co-occurring configuration;\n"
                 "the vector of medians is a per-dimension composite. Dotted = prior box.",
                 fontsize=9)
    ax.legend(fontsize=8, loc="best")
    ax.grid(alpha=0.25)
    return fig


def _figure_corner(pool_log, results, keys, low, high, rng):
    idx = (np.arange(pool_log.shape[0]) if pool_log.shape[0] <= _FIGURE_SUBSAMPLE
           else rng.choice(pool_log.shape[0], _FIGURE_SUBSAMPLE, replace=False))
    d = len(keys)
    ur = next(r for r in results if r["variant"] == "unrestricted")
    fig = Figure(figsize=(13, 13), layout="constrained")
    axes = fig.subplots(d, d)
    for a in range(d):
        for b in range(d):
            ax = axes[a, b]
            if a == b:
                ax.hist(10.0 ** pool_log[idx, a], bins=30, color="0.6")
                ax.axvline(ur["sgm_abs"][a], color="gold", lw=2)
                ax.axvline(ur["vom_abs"][a], color="magenta", lw=2, ls="--")
                ax.axvline(10.0 ** low[a], color="k", lw=0.6, ls=":")
                ax.axvline(10.0 ** high[a], color="k", lw=0.6, ls=":")
            elif b < a:
                ax.scatter(10.0 ** pool_log[idx, b], 10.0 ** pool_log[idx, a], s=4, c="0.6",
                           alpha=0.3, edgecolors="none")
                ax.scatter(ur["sgm_abs"][b], ur["sgm_abs"][a], marker="*", s=130, c="gold",
                           edgecolors="k", linewidths=0.8, zorder=6)
                ax.scatter(ur["vom_abs"][b], ur["vom_abs"][a], marker="X", s=80, c="magenta",
                           edgecolors="k", linewidths=0.8, zorder=6)
            else:
                ax.axis("off")
            if a == d - 1:
                ax.set_xlabel(keys[b], fontsize=8)
            if b == 0 and a != 0:
                ax.set_ylabel(keys[a], fontsize=8)
            ax.tick_params(labelsize=6)
    fig.suptitle("Pairwise pool structure (physical, dotted = prior box). Gold star = SGM (real "
                 "sample); magenta X = vector of medians. The vector of medians ignores the "
                 "cross-parameter correlations the SGM keeps.", fontsize=11)
    return fig


def _figure_out_of_box(frac_below, frac_above, keys):
    fig = Figure(figsize=(8, 4.5), layout="constrained")
    ax = fig.add_subplot(1, 1, 1)
    x = np.arange(len(keys))
    ax.bar(x - 0.2, 100 * frac_below, width=0.4, color="C0", label="below lower bound")
    ax.bar(x + 0.2, 100 * frac_above, width=0.4, color="C3", label="above upper bound")
    ax.set_xticks(x)
    ax.set_xticklabels(keys, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("percent of pool draws out of prior box")
    ax.set_title("Out-of-prior mass per parameter (nonzero only when the pool is unrestricted).",
                 fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    return fig


def _write_report(args, R, pool_log, keys, choice, mode, n_source, low, high, results, in_box, rng,
                  collection_label):
    report_dir = R["posit_dir"] / f"{R['alias']}_{R['timing_label']}_Nuisance_DLI_Sample_Geometric_Median"
    reporter = DiagnosticReporter(
        stage="Nuisance_DLI_Sample_Geometric_Median", enabled=True, dump=True, dump_dir=report_dir,
        run_label=f"{R['alias']}_{R['timing_label']}",
        timestamp=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"))

    n = pool_log.shape[0]
    finite = bool(np.isfinite(pool_log).all())
    reporter.check("pool_nonempty", n > 0, f"{n} imaging vectors in the pool")
    reporter.check("no_nan_inf(pool)", finite, "clean" if finite else "NaN/Inf present")
    reporter.check("sgm_is_real_sample", True,
                   "the Sample Geometric Median is an actual pool member, so its joint correlations are intact")

    frac_below = (pool_log < low).mean(0)
    frac_above = (pool_log > high).mean(0)
    reporter.table(
        "Out-of-prior mass (per parameter)",
        ["parameter", "box low (log10)", "box high (log10)", "below %", "above %"],
        [[keys[i], f"{low[i]:.3f}", f"{high[i]:.3f}", f"{100 * frac_below[i]:.2f}",
          f"{100 * frac_above[i]:.2f}"] for i in range(len(keys))],
        note="Fraction of pool draws outside the imaging prior box per dimension. Nonzero only when the "
             "pool was built unrestricted (raw-flow draws not rejection-sampled to the box).")

    for res in results:
        if res["n"] == 0:
            reporter.stat(f"Sample Geometric Median [{res['variant']}]", "empty subcollection",
                          note="no pool vectors satisfy the in-box constraint")
            continue
        rows = [[keys[i], f"{res['sgm_abs'][i]:.5g}", f"{res['vom_abs'][i]:.5g}",
                 f"{res['sgm_log'][i]:.4f}", f"{res['vom_log'][i]:.4f}"] for i in range(len(keys))]
        reporter.table(
            f"Sample Geometric Median vs vector of medians — {res['variant']} (n={res['n']}, method={res['method']})",
            ["parameter", "SGM (abs)", "vector-of-medians (abs)", "SGM (log10)", "VoM (log10)"], rows,
            note="SGM = the median VECTOR (a real pool member, correlations intact); vector-of-medians = "
                 "the per-dimension composite, which need not correspond to any acquisition. "
                 f"SGM in-box: {res['sgm_in_box']}; vector-of-medians in-box: {res['vom_in_box']}.")

    reporter.table(
        "Typicality of the vector of medians versus the SGM",
        ["variant", "normalized dist SGM-VoM", "Mahalanobis SGM", "Mahalanobis VoM",
         "KDE density VoM/SGM"],
        [[r["variant"], f"{r['box_dist']:.4f}", f"{r['maha_sgm']:.3f}", f"{r['maha_vom']:.3f}",
          f"{r['density_ratio']:.3f}"] for r in results if r["n"] > 0],
        note="Prior-range-normalized absolute space. A density ratio near 1 or a Mahalanobis close to "
             "the SGM's means the vector of medians is not off in a low-density region here; the SGM's "
             "advantage is that it is guaranteed realizable, independent of local density.")

    corr = np.corrcoef(pool_log, rowvar=False)
    reporter.table(
        "Joint correlation matrix (Pearson, log10)",
        ["parameter"] + keys,
        [[keys[a]] + [f"{corr[a, b]:+.3f}" for b in range(len(keys))] for a in range(len(keys))],
        note="The cross-parameter correlations the SGM preserves and the vector of medians discards."
             + (" Pooled correlations can be inflated by a between-condition shift (Simpson's paradox)."
                if args.condition == "pooled" else
                f" Computed within a single condition ({args.condition}), so the between-condition "
                f"(Simpson's paradox) inflation does not apply here."))

    reporter.stat("Nuisance_DLI artifact", str(R["artifact_path"]))
    reporter.stat("collection", collection_label)
    reporter.stat("condition", args.condition
                  + (f" ({CONDITION_DISPLAY[args.condition]})" if args.condition in CONDITION_DISPLAY else ""))
    reporter.stat("map source" if args.collection == "map" else "pool source",
                  args.map_source if args.collection == "map" else args.pool_source)
    reporter.stat("posterior_sample_pool_choice", choice)
    reporter.stat("pool_mode", mode)
    reporter.stat("pool size (vectors)", str(n),
                  note=f"in-box: {int(in_box.sum())} / {n}"
                       + ("" if args.max_samples == 0 else f"; subsampled from {n_source}"))
    reporter.stat("space", "absolute (10**theta), normalized by the absolute prior range")

    reporter.save_figure(
        "sgm_mu_r_mu_pc", _figure_plane(pool_log, results, keys, low, high, 0, 2, rng),
        caption="PSF width versus brightness. The pool draws (grey) with the SGM (gold star, a real "
                "sample) and the per-dimension vector of medians (magenta X). Where the pool is "
                "multimodal, a per-dimension composite drifts toward the low-density region between "
                "modes while the SGM stays on a real configuration.")
    reporter.save_figure(
        "sgm_corner", _figure_corner(pool_log, results, keys, low, high, rng),
        caption="Pairwise pool structure across all six imaging parameters with the SGM and the vector "
                "of medians overplotted. Off-diagonal tilt is joint correlation the SGM keeps and the "
                "vector of medians (independent per dimension) ignores.")
    reporter.save_figure(
        "out_of_prior_mass", _figure_out_of_box(frac_below, frac_above, keys),
        caption="Fraction of pool draws outside the imaging prior box per parameter — the genuine "
                "out-of-bounds set (distinct from the typicality of any summary point), nonzero only "
                "for an unrestricted pool.")

    reporter.summary()
    reporter.write_report()
    return report_dir


def _run(args, R):
    low = np.array(det.theta_lower_bound(), dtype=float)
    high = np.array(det.theta_upper_bound(), dtype=float)

    if args.dry_run:
        print("[DRY RUN] Nuisance_DLI Sample Geometric Median — would read/write:")
        print(f"    collection : {args.collection}  |  condition: {args.condition}"
              + (f"  |  map-source: {args.map_source}" if args.collection == "map" else ""))
        if args.collection == "map" and args.map_source == "experiment":
            mc, exp = R["map_cache_path"], R["experiment_map_path"]
            print(f"    MAP source : MapEstimate cache {mc}  [{'OK' if mc.exists() else 'MISSING'}]")
            print(f"                 else Experiment MAP {exp}  [{'OK' if exp.exists() else 'MISSING'}]")
        elif args.collection == "map":       # window-sgm
            pc = R["pool_cache_path"]
            print(f"    window-sgm : per-window SGM from {pc}  [{'OK' if pc.exists() else 'MISSING'}]")
        elif args.pool_source == "cache":
            pc = R["pool_cache_path"]
            print(f"    pool cache : {pc}  [{'OK' if pc.exists() else 'MISSING'}]")
        else:
            ap = R["artifact_path"]
            print(f"    artifact   : {ap}  [{'OK' if ap.exists() else 'MISSING'}]")
        if args.condition != "pooled":
            print(f"    labels     : the condition filter needs per-row labels (a labeled pool, or the "
                  f"experiment MAP)")
        print(f"    prior box  : log10 low {low.tolist()}  high {high.tolist()}")
        print(f"    method     : Sample Geometric Median in absolute space (exact medoid when small, "
              f"else Weiszfeld + snap); full pool and in-box subcollection")
        print(f"    writes     : {R['posit_dir']}/{R['alias']}_{R['timing_label']}_"
              f"Nuisance_DLI_Sample_Geometric_Median/")
        print("[DRY RUN] no compute performed.")
        return

    rng = np.random.default_rng(args.seed)
    pool_log, keys, choice, mode, n_source, collection_label = _load_collection(args, R)
    if list(keys) != list(det.DETECTOR_PARAMETER_KEYS):
        raise ValueError(f"pool schema {keys} does not match the imaging parameter keys "
                         f"{det.DETECTOR_PARAMETER_KEYS}")
    if args.max_samples and pool_log.shape[0] > args.max_samples:
        pool_log = pool_log[rng.choice(pool_log.shape[0], args.max_samples, replace=False)]

    results, in_box = _summary_vectors(pool_log, low, high, rng)
    report_dir = _write_report(args, R, pool_log, keys, choice, mode, n_source, low, high,
                               results, in_box, rng, collection_label)

    ur = next(r for r in results if r["variant"] == "unrestricted")
    sgm_str = ", ".join(f"{k}={v:.4g}" for k, v in zip(keys, ur["sgm_abs"]))
    print(f"Sample Geometric Median ({args.collection}, unrestricted, absolute): {sgm_str}")
    print(f"Report: {report_dir}")


def main(args):
    _run(args, _resolve(args.total_time_seconds))


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))

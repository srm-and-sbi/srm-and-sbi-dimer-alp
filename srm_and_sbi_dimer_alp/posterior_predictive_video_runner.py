"""Shared engine for the posterior-predictive video comparison (biology and detector workflows).

The check both workflows want is the same: take the parameters INFERRED from one real recording,
simulate a video with them, and put the two side by side. If the synthetic frames do not look like
the recording that produced the parameters, the posterior is explaining the data with a
configuration the forward model cannot actually render -- a failure no summary statistic on
held-out synthetic data can reveal, because it only appears when the model is pointed at reality.

The two workflows invert which half of the model the MAP supplies and which half is held fixed:

* **detector** -- the MAP supplies the six IMAGING parameters; the reaction-diffusion block is a
  marginalized nuisance, drawn (or pinned) per render, and the system is built diffusion-only
  because the detector's physics model has no reactions.
* **biology** -- the MAP supplies the ten REACTION-DIFFUSION parameters and the system is built
  with its full reaction network; the imaging block is held fixed at the calibrated vector the
  training videos were generated with, read from the ``Nuisance_DLI`` artifact at run time.

In both cases the five SCOPE camera parameters are pinned to their MET values rather than drawn:
the comparison is against one specific real acquisition, so a random camera draw would add
variation that has nothing to do with the question.

The renderer itself is already common to both workflows (``simulation_dli_support.render_dli_video``,
re-exported as ``render_detector_video``), and the 11-key imaging order it consumes is a property of
that renderer rather than of either workflow -- which is why ``detector_parameterization`` is read
here for the imaging CONTRACT even on the biology path.
"""
from __future__ import annotations

import argparse
import warnings

import numpy as np
from matplotlib.figure import Figure

from . import detector_parameterization as det
from .detector_simulation_rds_support import build_detector_rds_simulation, draw_nuisance_physical
from .parameterization import PARAMETERS, RunTiming
from .sample_geometric_median import sample_geometric_median
from .simulation_dli_support import render_dli_video
from .simulation_rds_support import build_simulation, build_system, extract_trajectory_poses
from .workflow import parameter_keys as _wf_keys, parameter_table

# Conditions are named scientifically wherever a reader sees them; the tokens below survive only as
# the stored ``kinds`` field of the MAP database and the recording filenames on disk.
# Condition naming (stored token <-> scientific name) has ONE definition, in experiment_support.
from .experiment_support import KIND_OF_CONDITION

def _clip_span_token(n_frames, frame_time):
    """The clip's own duration as a label token (e.g. 1000 frames @ 0.02 s -> ``20S``); the
    canonical duration labeling, so the token matches the rest of the codebase."""
    return RunTiming(total_time_seconds=n_frames * frame_time,
                     frames=PARAMETERS.simulation.timing).label.split("_")[0]


def _build_stem(project_alias, map_label, kind, cell, chunk, map_source, clip_token,
                fixed_imaging=False, fixed_nuisance=False, run_label=None):
    """Output basename: canonical ``{alias}_{model_window}`` + imaging descriptor + clip span.
    ``map_label`` (e.g. ``2S_50FPS``) is the estimator's model window; ``clip_token`` (e.g.
    ``20S``) is the rendered length. Chunk source names the concrete entry
    (``..._Cell_{cell}_Chunk_{chunk}_...``); cell-median drops the chunk and marks ``Median``;
    ``fixed_imaging`` marks a correct-source fixed-parameter render (no MAP database);
    ``fixed_nuisance`` marks a pinned RDS nuisance (counts/diffusivities held, not drawn); and
    ``run_label`` appends an optional caller tag at the END of the name so renders that share a
    selection (same cell, both fixed-imaging + fixed-nuisance) stay distinct instead of
    overwriting each other."""
    nuis_tok = "_Fixed_Nuisance" if fixed_nuisance else ""
    label_tok = ""
    if run_label:
        safe = "".join(c if c.isalnum() else "_" for c in str(run_label)).strip("_")
        label_tok = f"_{safe}" if safe else ""
    if fixed_imaging:
        return f"{project_alias}_{map_label}_Fixed_Imaging{nuis_tok}_{kind}_Cell_{cell}_{clip_token}{label_tok}"
    descriptor = {"cell-median": "MAP_Estimate_Median",
                  "cell-sgm": "MAP_Estimate_SGM"}.get(map_source, "MAP_Estimate")
    chunk_part = "" if map_source.startswith("cell-") else f"_Chunk_{chunk}"
    return (f"{project_alias}_{map_label}_{descriptor}{nuis_tok}_{kind}_Cell_{cell}"
            f"{chunk_part}_{clip_token}{label_tok}")


def _load_map_theta(map_npz, keys, kind, cell, chunk, source, prior_low, prior_high):
    """Physical imaging theta for (kind, cell): a single chunk's MAP (``source='chunk'``) or
    the per-cell median over all of that cell's chunk MAPs (``source='cell-median'``). Fails
    with guidance listing the available cells/chunks."""
    with np.load(str(map_npz), allow_pickle=False) as d:
        inferred_log10 = np.asarray(d["inferred_log10"], dtype=float)
        kind_index = np.asarray(d["kind_index"])
        cells = np.asarray(d["cell"])
        chunks = np.asarray(d["chunk"])
        kinds = [str(k) for k in d["kinds"]]
    if inferred_log10.ndim != 2 or inferred_log10.shape[1] != len(keys):
        raise ValueError(f"MAP database inferred_log10 has shape {inferred_log10.shape}; "
                         f"expected (N, {len(keys)}).")
    if kind not in kinds:
        raise ValueError(f"kind={kind!r} not in the MAP database; available: {kinds}.")
    ki = kinds.index(kind)
    cell_rows = np.where((kind_index == ki) & (cells == cell))[0]
    if cell_rows.size == 0:
        avail_cells = sorted({int(c) for c in cells[kind_index == ki]})
        raise ValueError(f"no MAP entries for kind={kind} cell={cell} in\n    {map_npz}\n"
                         f"available cells for {kind}: {avail_cells}")
    if source.startswith("cell-"):
        theta_log10 = _aggregate_cell(inferred_log10[cell_rows], source)
        how = ("Sample Geometric Median (a real chunk's estimate)" if source == "cell-sgm"
               else "per-dimension median (a composite, correlations discarded)")
        print(f"MAP source: {how} over {cell_rows.size} chunk(s) of {kind} cell {cell}.")
    else:                                                            # a concrete database entry
        row = np.where((kind_index == ki) & (cells == cell) & (chunks == chunk))[0]
        if row.size == 0:
            avail_chunks = sorted({int(c) for c in chunks[cell_rows]})
            raise ValueError(f"no MAP entry for kind={kind} cell={cell} chunk={chunk} in\n"
                             f"    {map_npz}\navailable chunks for {kind} cell {cell}: {avail_chunks}")
        theta_log10 = inferred_log10[int(row[0])]
        print(f"MAP source: chunk {chunk} of {kind} cell {cell}.")
    plo = np.asarray(prior_low, dtype=float)
    phi = np.asarray(prior_high, dtype=float)
    oob = [keys[i] for i in range(len(keys))
           if theta_log10[i] < plo[i] - 1e-9 or theta_log10[i] > phi[i] + 1e-9]
    if oob:
        print(f"WARNING: MAP imaging is outside the prior box for {oob}; the render EXTRAPOLATES "
              f"there, so a poor experimental-vs-synthetic match may reflect an out-of-prior MAP rather "
              f"than the calibration itself.")
    return np.power(10.0, theta_log10)


# Correct-source MET camera parameters (physical units) for --fixed-imaging-parameters.
# gamma = g/C from the MET EM gain and photons2ADU conversion; kappa_o the fitted optical
# background offset; kappa_b the configured baseline; kappa_s the datasheet read noise;
# kappa_q the configured quantum efficiency (marginalized as the SCOPE camera nuisance). See
# REFERENCE_EMCCD_NOISE_MODEL.md Sec. 6 and the MET provenance in DETECTOR_WORKFLOW.md Sec. 6.5.
MET_CAMERA_PHYSICAL = {
    "gamma":   200.0 / 4.78,   # EM gain / conversion (ADU per photoelectron) ~= 41.84
    "kappa_o": 28.7,           # optical background offset (incident photons); ThunderSTORM offset[photon] median, pooled Fab 28.9 / InlB 28.6 (REFERENCE_EMCCD_NOISE_MODEL.md sec. 6, DETECTOR_WORKFLOW.md sec. 6.2)
    "kappa_b": 175.0,          # camera baseline (ADU)
    "kappa_s": 10.5,           # read noise (ADU) at C ~= 4.78
    "kappa_q": 0.9,            # quantum efficiency (config; marginalized as SCOPE camera nuisance)
}


def _fixed_imaging_theta(overrides=None):
    """Physical imaging theta for --fixed-imaging-parameters: the 5 SCOPE camera parameters set
    to their correct-source MET values (``MET_CAMERA_PHYSICAL``), the 6 learnable emitter
    parameters held at their prior-center nominals (the table ``VALUE``), and finally any
    ``overrides`` (``{key: physical_value}``, from ``--set-imaging``) applied last -- e.g.
    emitter brightness/PSF calculated from the MET ThunderSTORM localizations, or a camera
    parameter swept for a sensitivity check. Returns the full 11-key ``DETECTOR_IMAGING`` render
    vector; overrides may target any of the 11 imaging parameters (learnable or SCOPE camera).
    No trained posterior is used -- this drives the corrected forward model directly to test
    whether it recapitulates the experimental pixel-intensity histogram."""
    unknown = [k for k in MET_CAMERA_PHYSICAL if k not in det.DETECTOR_SCOPE_KEYS]
    if unknown:
        raise ValueError(f"MET_CAMERA_PHYSICAL keys not in the SCOPE camera set: {unknown}")
    find = {k: i for i, k in enumerate(det.DETECTOR_IMAGING_KEYS)}   # 11-key render contract (6 learnable + 5 SCOPE)
    theta = np.empty(len(det.DETECTOR_IMAGING_KEYS), dtype=float)
    for element in det.DETECTOR_PARAMETERIZATION:                    # 6 learnable emitter params at prior-center nominals
        theta[find[element["KEY"]]] = element["VALUE"]
    for key, value in MET_CAMERA_PHYSICAL.items():                   # 5 SCOPE camera at correct-source MET values
        theta[find[key]] = value
    for key, value in (overrides or {}).items():                    # --set-imaging: override any of the 11 imaging params
        if key not in find:
            raise ValueError(f"--set-imaging key {key!r} is not an imaging parameter; "
                             f"valid keys are {det.DETECTOR_IMAGING_KEYS}.")
        theta[find[key]] = value
    print("Fixed imaging theta: camera parameters at correct-source MET values ("
          + ", ".join(f"{k}={v:.4g}" for k, v in MET_CAMERA_PHYSICAL.items())
          + "); non-camera parameters at prior-center nominals"
          + (f"; overrides {overrides}" if overrides else "") + ".")
    return theta


# RDS-nuisance keys, in DETECTOR_NUISANCE order (three counts + three diffusivities).
_NUISANCE_KEYS = [e["KEY"] for e in det.DETECTOR_NUISANCE]


def _fixed_nuisance_physical(overrides=None):
    """Physical RDS-nuisance vector for --fixed-nuisance-RDS: every nuisance parameter held at
    its prior-center nominal (the log-midpoint of its BoxUniform range), then any ``overrides``
    (``{key: physical_value}``) applied last. This replaces the fresh random ``draw_nuisance_physical``
    draw with a deterministic, controlled nuisance, so the emitter counts (monomer ``count_alp``,
    mobile-dimer ``count_bet``, immobile-dimer ``count_chi``) and diffusivities are pinned to
    condition-appropriate values rather than sampled from a flat prior (whose ~equal expected
    counts make every render implausibly dimer-heavy). Returns a ``(len(DETECTOR_NUISANCE),)``
    array in DETECTOR_NUISANCE order."""
    centers = {e["KEY"]: e["LOG_BASE"] ** ((e["PRIOR_RANGE"][0] + e["PRIOR_RANGE"][1]) / 2.0)
               for e in det.DETECTOR_NUISANCE}
    for key, value in (overrides or {}).items():
        if key not in centers:
            raise ValueError(
                f"--fixed-nuisance-RDS key {key!r} is not an RDS-nuisance parameter; "
                f"valid keys: {_NUISANCE_KEYS}.")
        centers[key] = value
    print("Fixed RDS nuisance: parameters at prior-center nominals"
          + (f"; overrides {overrides}" if overrides else "")
          + " (" + ", ".join(f"{k}={centers[k]:.4g}" for k in _NUISANCE_KEYS) + ").")
    return np.array([centers[k] for k in _NUISANCE_KEYS], dtype=float)


def _save_comparison_png(path, experimental, synth, kind, cell, sel_desc, display_norm,
                         nuisance, imaging_physical, fixed_imaging=False,
                         fixed_nuisance=False, imaging_label="imaging",
                         rds_label="reaction-diffusion", motion_desc=None, rds_table=None):
    """Static experimental-vs-synthetic panel. Col 0 = EXPERIMENTAL and col 1 = SYNTH, each
    showing the mid frame over its max projection; col 2 holds the shared pixel-intensity
    histogram in log-y (full range, top) and linear-y (zoomed to the bulk, bottom); col 3 holds
    the syn/exp RATIO-per-quantile match plot (top -- a flat line on 1.0 = perfect match, read
    across min/p0.01/median/p90/p99/p99.9/p99.99/max so no single region dominates) over the
    quantile table + provenance (bottom). The provenance lists this render's DLI/imaging
    parameters (out-of-prior ones flagged) and its RDS-nuisance draw, so a mismatch can be
    attributed to the imaging estimate or the nuisance rather than left unexplained.

    Both panels and the histogram use the STORED synthetic (``synth_u16``, clipped to the
    non-negative uint16 range), so the comparison is like-with-like against the experimental frames and
    matches the persisted clip -- not the pre-clip float. The experimental and synthetic image panels
    ALWAYS share ONE color limit, in every ``display_norm`` mode -- the max-projection row shares the
    full ``[min, max]`` range, and the mid-frame row shares a window the mode selects -- so identical
    intensities map to identical colors and the exp-vs-synth comparison is fair; the mode only sets WHAT
    the shared mid-frame window is. ``full`` (default): the full whole-clip ``[min, max]`` (nothing
    clipped). ``percentile``: the whole-clip ``[min, p99.99]``, dropping the top-0.01% hot-pixel sliver
    for contrast. ``autoscale``: the two displayed frames' shared min/max, the most contrast. The
    histogram is in ADU in every mode."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FixedLocator, FixedFormatter, NullLocator

    src_tok = "FIXED" if fixed_imaging else "MAP"
    imaging_header = f"{imaging_label} (absolute)"
    imaging_note = ("imaging: fixed correct-source MET values" if fixed_imaging
                    else "imaging: MAP theta of the selected chunk")
    fig_title = ("Fixed-imaging video check (experimental vs synthetic; correct-source MET)"
                 if fixed_imaging
                 else "Posterior-predictive video check (experimental vs synthetic-from-MAP)")
    mid = experimental.shape[0] // 2
    label = {"ALP": "MET-FAB", "BET": "MET-INLB"}.get(kind, kind)
    fig, ax = plt.subplots(2, 4, figsize=(18, 8.5), dpi=200)

    # --- pixel-intensity quantiles (ADU): drive both the frame display window and the match plot ---
    exp_r = experimental.ravel(); syn_r = synth.ravel()
    q_probs = [0, 0.01, 50, 90, 99, 99.9, 99.99, 100]
    q_names = ["min", "p0.01", "median", "p90", "p99", "p99.9", "p99.99", "max"]
    eq = np.percentile(exp_r, q_probs); sq = np.percentile(syn_r, q_probs)
    # The full quantile set drives the match plot + table (below); the frame display window
    # (below) spans the full range, so no pixels are clipped from the frame color scale.

    def _frame(a, arr, title, clim):
        a.imshow(arr, cmap="magma", origin="lower", interpolation="none",
                 vmin=clim[0], vmax=clim[1])
        a.set_title(title, fontsize=9); a.set_xticks([]); a.set_yticks([])

    # The experimental and synthetic image panels ALWAYS share ONE color limit, in EVERY mode, so
    # identical intensities map to identical colors and the comparison is fair: the mid-frame row
    # shares `frame_clim` and the max-projection row shares `proj_clim`. `display_norm` only sets WHAT
    # the shared mid-frame window is; the max projection always shares the full [min, max] range.
    emp, smp = experimental.max(0), synth.max(0)
    proj_clim = (float(min(emp.min(), smp.min())), float(max(emp.max(), smp.max())))
    if display_norm == "full":                    # full whole-clip [min, max]; nothing clipped
        frame_clim = (float(min(eq[0], sq[0])), float(max(eq[-1], sq[-1])))
    elif display_norm == "percentile":            # whole-clip [min, p99.99]; drops the hot-pixel sliver
        frame_clim = (float(min(exp_r.min(), syn_r.min())),
                      float(max(np.percentile(exp_r, 99.99), np.percentile(syn_r, 99.99))))
    else:                                         # autoscale: the two displayed frames' shared min/max
        frame_clim = (float(min(experimental[mid].min(), synth[mid].min())),
                      float(max(experimental[mid].max(), synth[mid].max())))
    exp_clim = syn_clim = frame_clim
    exp_proj_clim = syn_proj_clim = proj_clim

    # col 0 = EXPERIMENTAL (frame over max projection), col 1 = SYNTH (same); col 2 = histograms;
    # col 3 = the syn/exp ratio-per-quantile match plot over the quantile table + provenance.
    _frame(ax[0, 0], experimental[mid], f"EXPERIMENTAL {label} cell {cell}  frame {mid}", exp_clim)
    _frame(ax[1, 0], experimental.max(0), "EXPERIMENTAL  max projection", exp_proj_clim)
    _frame(ax[0, 1], synth[mid], f"SYNTH ({src_tok} {kind} c{cell} {sel_desc})  frame {mid}", syn_clim)
    _frame(ax[1, 1], synth.max(0), "SYNTH  max projection", syn_proj_clim)

    # --- histograms with SHARED bins (like-with-like): log-y over the full range (top), and
    #     linear-y through ~p99.99 (bottom) -- the whole range bar the top-0.01% hot-pixel sliver,
    #     kept off only so the linear scale stays readable; the tail is reported exactly in the
    #     quantile table at right, and the log-y panel above shows it in full. ---
    lo = float(min(exp_r.min(), syn_r.min()))
    hi_full = float(max(exp_r.max(), syn_r.max()))
    _i9999 = q_names.index("p99.99")
    hi_lin = float(max(eq[_i9999], sq[_i9999]) * 1.05)          # linear panel through ~p99.99
    bins_log = np.linspace(lo, hi_full, 200)
    bins_lin = np.linspace(lo, hi_lin, 160)

    def _hist(a, bins):
        a.hist(exp_r, bins=bins, histtype="step", density=True, color="tab:blue", label="experimental")
        a.hist(syn_r, bins=bins, histtype="step", density=True, color="tab:orange", label="synth")
        a.tick_params(labelsize=7)

    _hist(ax[0, 2], bins_log); ax[0, 2].set_yscale("log")
    ax[0, 2].set_title("pixel-intensity density (ADU) - log y, full range", fontsize=8)
    ax[0, 2].legend(fontsize=7)
    _hist(ax[1, 2], bins_lin); ax[1, 2].set_xlim(lo, hi_lin)
    ax[1, 2].set_title(f"linear y, x<={hi_lin:.0f} ADU (to p99.99)", fontsize=8)

    # --- ax[0,3]: syn/exp ratio per quantile -- the direct "do they match?" read.
    #     A flat line on 1.0 (dashed) is a perfect match; the green band is within +-10%.
    #     Log-y so a 2x over- and a 2x under-shoot read symmetrically; spans the dark end
    #     (min, p0.01) to the bright end (p99.99), so no single region dominates the judgment. ---
    axr = ax[0, 3]
    ratios = np.array([(sv / ev) if ev > 0 else np.nan for ev, sv in zip(eq, sq)])
    xq = np.arange(len(q_names))
    axr.axhspan(0.9, 1.1, color="tab:green", alpha=0.12)
    axr.axhline(1.0, color="gray", lw=1.0, ls="--")
    axr.plot(xq, ratios, "-o", color="tab:red", ms=5)
    axr.set_yscale("log"); axr.set_ylim(0.5, 3.5)
    # Fixed decimal y-labels; suppress the log MINOR ticks -- in this narrow [0.5, 3.5] range
    # they otherwise print auto sci-notation labels (e.g. 6x10^-1) that clutter the axis.
    axr.yaxis.set_major_locator(FixedLocator([0.5, 0.7, 1.0, 1.5, 2.0, 3.0]))
    axr.yaxis.set_major_formatter(FixedFormatter(["0.5", "0.7", "1.0", "1.5", "2.0", "3.0"]))
    axr.yaxis.set_minor_locator(NullLocator())
    axr.tick_params(axis="y", labelsize=7)
    axr.set_xticks(xq); axr.set_xticklabels(q_names, rotation=45, ha="right", fontsize=7)
    axr.set_ylabel("synth / exp", fontsize=8)
    axr.set_title("MATCH: syn/exp per quantile\n(1.0 dashed = perfect; green = within 10%)", fontsize=8)
    axr.grid(True, axis="y", which="both", alpha=0.25)

    # --- ax[1,3]: quantile table + provenance (imaging theta + RDS nuisance) ---
    ax[1, 3].axis("off")
    ikeys = [e["KEY"] for e in det.DETECTOR_PARAMETERIZATION]
    img = np.asarray(imaging_physical, dtype=float).ravel()
    plo = np.asarray(det.theta_lower_bound(), dtype=float)
    phi = np.asarray(det.theta_upper_bound(), dtype=float)
    _short = {"prob_photo_bleach": "p_bleach", "lambda_rate": "lambda"}
    dli = []
    for i, k in enumerate(ikeys):
        lv = np.log10(img[i]) if img[i] > 0 else float("-inf")
        mark = "*" if (lv < plo[i] - 1e-9 or lv > phi[i] + 1e-9) else ""
        dli.append(f"{_short.get(k, k)}={img[i]:.3g}{mark}")
    dli_lines = "\n".join("  " + "  ".join(dli[j:j + 3]) for j in range(0, len(dli), 3))
    nuis = np.asarray(nuisance, dtype=float).ravel()
    npart = []
    # The label table must match the block being shown, or zip() silently truncates: the detector's
    # nuisance is 6 entries while biology's MAP is 10, so defaulting to the detector table dropped
    # biology's four rate parameters without any error.
    table = det.DETECTOR_NUISANCE if rds_table is None else rds_table
    if len(table) != nuis.size:
        raise ValueError(f"RDS label table has {len(table)} entries but the vector has "
                         f"{nuis.size}; they must correspond.")
    for ent, v in zip(table, nuis):
        lab2 = ent["LABEL"].translate({ord(c): None for c in "${}\\"})
        # Integer-format ONLY a true particle count. The prefix test this replaced also matched
        # "Count Per Second" -- biology's rate units -- and rounded continuous rates to integers,
        # printing 0.175/s and 0.127/s as "0", a value their log-uniform prior [0.1, 10] cannot
        # even represent. Exact match, so a new "Count ..." unit cannot silently re-break it.
        is_count = str(ent.get("UNIT", "")).strip().lower() == "count"
        npart.append(f"{lab2}={v:.0f}" if is_count else f"{lab2}={v:.3g}")
    nuis_lines = "\n".join("  " + "  ".join(npart[j:j + 3]) for j in range(0, len(npart), 3))
    tbl = f"{'quantile':<8}{'exp':>7}{'synth':>7}{'ratio':>7}\n"
    for name, ev, sv in zip(q_names, eq, sq):
        tbl += f"{name:<8}{ev:7.0f}{sv:7.0f}{(sv / ev if ev > 0 else float('inf')):7.2f}\n"
    ax[1, 3].text(
        0.0, 1.0,
        "QUANTILES (ADU)  ratio=synth/exp\n" + tbl + "\n"
        f"{imaging_header}:\n{dli_lines}\n  (* outside prior)\n"
        f"{rds_label}:\n{nuis_lines}\n"
        f"motion: {motion_desc or ('fixed nuisance (pinned)' if fixed_nuisance else 'fresh draw')}"
        f"; norm {display_norm}",
        fontsize=6.5, va="top", family="monospace")
    fig.suptitle(fig_title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)



def _aggregate_cell(rows_log10, source):
    """Reduce one cell's per-chunk MAP rows to a single vector, in log10 space.

    ``cell-sgm`` takes the Sample Geometric Median -- an actual chunk's estimate, so the rendered
    video corresponds to a configuration the posterior genuinely produced for this cell.
    ``cell-median`` takes each dimension's median independently, which is faster to explain but
    composes coordinates that need never have co-occurred; for a render that matters, because the
    simulator is then asked to realize a combination no chunk supported. Both are offered, and the
    report names which was used.
    """
    if source == "cell-sgm":
        a = 10.0 ** rows_log10
        rng_span = np.ptp(a, axis=0)
        rng_span[rng_span <= 0] = 1.0
        idx, _method = sample_geometric_median(a, rng_span)
        return rows_log10[idx]
    return np.median(rows_log10, axis=0)


def run_posterior_predictive_video(cfg, args):
    """Shared entry point. ``cfg`` is a WorkflowConfig; ``args`` the parsed CLI namespace."""
    import tifffile

    S = _ppv_spec(cfg, args)
    keys = S["map_keys"]
    kind = KIND_OF_CONDITION.get(args.kind, args.kind)      # scientific name -> stored token
    frame_time = PARAMETERS.simulation.timing.frame_time_seconds
    map_npz, experimental_tif, out_dir = S["map_npz"], S["experimental_tif"], S["out_dir"]
    needs_map = not (S["map_block"] == "imaging" and args.fixed_imaging_parameters)

    if args.dry_run:
        n_frames = None
        if experimental_tif.exists():
            with tifffile.TiffFile(str(experimental_tif)) as tf:
                n_frames = int(tf.series[0].shape[0])       # frame count from metadata; no pixel load
        clip_token = _clip_span_token(n_frames, frame_time) if n_frames else "<clip_span>"
        stem = _build_stem(S["paths"].project_alias, S["map_label"], args.kind, args.cell, args.chunk,
                           args.map_source, clip_token,
                           fixed_imaging=args.fixed_imaging_parameters,
                           fixed_nuisance=bool(args.fixed_nuisance_rds),
                           run_label=args.run_label)
        print(f"[DRY RUN] posterior-predictive video ({cfg.tag}) would read:")
        print(f"    MAP supplies : the {S['map_block'].upper()} block ({len(keys)} parameters)")
        if needs_map:
            print(f"    MAP database : {map_npz}  [{'OK' if map_npz.exists() else 'MISSING'}]")
            sel = f"chunk {args.chunk}" if args.map_source == "chunk" else args.map_source
            print(f"    selection    : {args.kind} cell {args.cell}  ({sel})")
        else:
            print("    imaging theta: FIXED to MET values (no MAP read); "
                  "non-camera parameters at prior-center nominals")
        print(f"    imaging held : {S['imaging_desc']}")
        print(f"    RDS built    : {S['rds_desc']}")
        print(f"    experimental : {experimental_tif}  "
              f"[{'OK' if experimental_tif.exists() else 'MISSING'}]")
        print(f"    display norm : {args.display_norm}")
        print(f"[DRY RUN] would write under:\n    {out_dir}/\n"
              f"        {stem}_{{Synthetic_Video.npz,Comparison.png,Trajectory.h5}}")
        print("[DRY RUN] no simulation, no render.")
        return 0

    if needs_map and not map_npz.exists():
        raise FileNotFoundError(
            f"MAP database not found:\n    {map_npz}\nRun the {cfg.tag} Experiment stage first.")
    if not experimental_tif.exists():
        raise FileNotFoundError(
            f"Experimental recording not found:\n    {experimental_tif}\n"
            f"(check --kind/--cell and --experiment-span-seconds={args.experiment_span_seconds}).")

    map_theta = None
    if needs_map:
        map_theta = _load_map_theta(map_npz, keys, kind, args.cell, args.chunk,
                                    args.map_source, S["prior_low"], S["prior_high"])

    imaging_physical, imaging_desc = S["imaging_physical"](args, map_theta)

    # ---- experimental recording -> the render length (from the data, never hardcoded) ----
    experimental = np.asarray(tifffile.imread(str(experimental_tif)))          # (n_frames, H, W)
    if experimental.ndim != 3:
        raise ValueError(f"experimental recording has shape {experimental.shape}; "
                         f"expected (n_frames, H, W).")
    n_frames = int(experimental.shape[0])
    render_timing = RunTiming(total_time_seconds=n_frames * frame_time,
                              frames=PARAMETERS.simulation.timing)
    clip_token = _clip_span_token(n_frames, frame_time)
    stem = _build_stem(S["paths"].project_alias, S["map_label"], args.kind, args.cell, args.chunk,
                       args.map_source, clip_token,
                       fixed_imaging=args.fixed_imaging_parameters,
                       fixed_nuisance=bool(args.fixed_nuisance_rds),
                       run_label=args.run_label)
    print(f"Experimental recording: {n_frames} frames ({clip_token} clip); rendering a matching "
          f"synthetic for {args.kind} cell {args.cell} "
          f"(MAP -> {S['map_block']}, {args.map_source}).")

    # ---- simulate one trajectory at the experimental length ----
    import readdy
    out_dir.mkdir(parents=True, exist_ok=True)
    smut, rds_provenance, rds_desc = S["build_rds"](args, map_theta)
    traj_path = out_dir / f"{stem}_Trajectory.h5"
    if traj_path.exists():
        traj_path.unlink()
    smut.output_file = str(traj_path)
    smut.progress_output_stride = render_timing.total_steps
    smut.run(n_steps=render_timing.total_steps,
             timestep=render_timing.delta_time_nanoseconds * readdy.units.nanosecond,
             show_summary=False)

    # ---- poses -> render with the resolved imaging vector ----
    tray = readdy.Trajectory(filename=str(traj_path))
    tray_poses, dimer_mask = extract_trajectory_poses(
        tray, return_dimer_mask=True, verbose=args.verbose)
    if tray_poses.shape[0] != render_timing.frame_count:
        raise RuntimeError(f"trajectory holds {tray_poses.shape[0]} frames but the render "
                           f"length is {render_timing.frame_count}.")
    with warnings.catch_warnings():   # absent particles are NaN by design (mirror Simulation_DLI)
        warnings.filterwarnings("ignore", message="All-NaN slice encountered",
                                category=RuntimeWarning)
        pro_tray_poses = np.nanmax(a=tray_poses, axis=3)
    synth = render_dli_video(pro_tray_poses=pro_tray_poses,
                             imaging_physical=imaging_physical,
                             dimer_mask=dimer_mask, dimer_model=args.dimer_model,
                             seed=args.seed, verbose=args.verbose)
    synth = np.moveaxis(synth, -1, 0)                          # (H, W, n_frames) -> (n_frames, H, W)

    # ---- persist (clip + provenance) and a static comparison figure ----
    clips_path = out_dir / f"{stem}_Synthetic_Video.npz"
    # Storage clip to the non-negative uint16 range. The reference EMCCD noise model emits a small
    # signed excursion -- a Gaussian read-noise tail below 0, rarely values above 65535 -- that a
    # real camera cannot record, so we clip to the domain the experimental frames live in. The
    # counts are reported on every run so an alternative noise model's behavior can be compared at
    # this exact point. See REFERENCE_EMCCD_NOISE_MODEL.md.
    n_under = int(np.count_nonzero(synth < 0.0))
    n_over = int(np.count_nonzero(synth > 65535.0))
    if n_under or n_over:
        print(f"  storage clip to [0, 65535]: {n_under} pixel(s) < 0, {n_over} pixel(s) > 65535 "
              f"(of {synth.size}); clipped to match the experimental non-negative domain.")
    synth_u16 = np.clip(np.rint(synth), 0, 65535).astype(np.uint16)
    np.savez_compressed(
        str(clips_path),
        experimental=experimental.astype(np.uint16), synth=synth_u16,
        imaging_physical=imaging_physical, imaging_keys=np.array(det.DETECTOR_IMAGING_KEYS),
        map_theta=(np.array([]) if map_theta is None else map_theta),
        map_keys=np.array(keys), map_block=S["map_block"], workflow=cfg.tag,
        rds_provenance=rds_provenance,
        kind=args.kind, cell=args.cell,
        chunk=(-1 if args.chunk is None else args.chunk), map_source=args.map_source,
        seed=(-1 if args.seed is None else args.seed),
        frame_time_seconds=frame_time, n_frames=n_frames, experimental_tif=str(experimental_tif))
    if S["map_block"] == "rds":
        imaging_label = "FIXED imaging (calibrated Nuisance_DLI + MET SCOPE)"
        rds_label = "INFERRED reaction-diffusion (MAP theta, absolute)"
        motion_desc = "from the MAP reaction-diffusion parameters (not a draw)"
        rds_table = parameter_table(cfg)                 # all 10 biology parameters
    else:
        imaging_label = ("FIXED imaging (MET values)" if args.fixed_imaging_parameters
                         else "INFERRED imaging (MAP theta, absolute)")
        rds_label = "NUISANCE reaction-diffusion (marginalized)"
        motion_desc = None
        rds_table = None                                 # detector: the 6-entry nuisance table
    _save_comparison_png(out_dir / f"{stem}_Comparison.png", experimental, synth_u16,
                         args.kind, args.cell, args.map_source,
                         args.display_norm, rds_provenance, imaging_physical,
                         imaging_label=imaging_label, rds_label=rds_label,
                         motion_desc=motion_desc, rds_table=rds_table,
                         fixed_imaging=args.fixed_imaging_parameters,
                         fixed_nuisance=bool(args.fixed_nuisance_rds))
    print(f"Persisted clip + provenance:\n    {clips_path}")
    print(f"Static comparison figure:\n    {out_dir / (stem + '_Comparison.png')}")
    print(f"Trajectory (provenance):\n    {traj_path}")
    print(f"  imaging: {imaging_desc}")
    print(f"  RDS    : {rds_desc}")
    return 0


def build_parser(description):
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--total-time-seconds", type=float, required=True,
                   help="model window of the RUN that produced the MAP database; sets the timing "
                        "label locating it. The render length comes from the recording itself.")
    p.add_argument("--kind", default="MET-FAB", choices=tuple(KIND_OF_CONDITION),
                   help="experimental condition: MET-FAB (monomer control) or MET-INLB (dimer).")
    p.add_argument("--cell", type=int, required=True, help="cell (recording) index.")
    p.add_argument("--chunk", type=int, default=None,
                   help="chunk index, for --map-source chunk.")
    p.add_argument("--map-source", choices=("chunk", "cell-sgm", "cell-median"), default="cell-sgm",
                   help="which MAP vector to render: 'chunk' one specific window; 'cell-sgm' "
                        "(default) the Sample Geometric Median over that cell's chunks -- a real "
                        "chunk's estimate, correlations intact; 'cell-median' the per-dimension "
                        "median, which can compose a combination no chunk produced.")
    p.add_argument("--experiment-span-seconds", type=int, default=20,
                   help="duration (s) of the experimental recording to read (default 20).")
    p.add_argument("--dimer-model", choices=("sum", "multiply"), default="sum",
                   help="how a dimer's brightness combines (default 'sum').")
    p.add_argument("--display-norm", default="full", choices=("full", "autoscale", "percentile"),
                   help="display normalization for the comparison figure: 'full' (default; "
                        "shared full-range window), 'autoscale' (per-image), or 'percentile'. "
                        "Display-only -- never enters the quantitative comparison.")
    p.add_argument("--fixed-imaging-parameters", action="store_true",
                   help="detector only: skip the MAP database and pin imaging to MET values.")
    p.add_argument("--set-imaging", action="append", default=[], metavar="KEY=VALUE",
                   help="override one imaging parameter (sensitivity check); repeatable.")
    p.add_argument("--fixed-nuisance-RDS", dest="fixed_nuisance_rds", action="append", default=None,
                   metavar="KEY=VALUE",
                   help="detector only: pin the RDS nuisance instead of drawing it; repeatable.")
    p.add_argument("--run-label", default=None, help="optional token appended to the output stem.")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for the simulation and render.")
    p.add_argument("--verbose", action="store_true", help="verbose simulation/render output.")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve paths and report what would be read/written; simulate nothing.")
    return p


# ---- the one place the two workflows differ ----------------------------------------------------

def _scope_met():
    """The five SCOPE camera parameters at their MET values, in DETECTOR_SCOPE order."""
    return np.array([MET_CAMERA_PHYSICAL[k] for k in det.DETECTOR_SCOPE_KEYS], dtype=float)


def _ppv_spec(cfg, args):
    """Resolve the workflow-specific half: which block the MAP supplies, and how the other is fixed."""
    paths = cfg.paths
    data_bank_root = PARAMETERS.machine.data_bank_root
    map_label = RunTiming(total_time_seconds=args.total_time_seconds,
                          frames=PARAMETERS.simulation.timing).label
    posit_dir = data_bank_root / paths.posit_subdir
    exp_out_dir = paths.experiment_recovery_dir(data_bank_root, map_label)
    kind_token = KIND_OF_CONDITION.get(args.kind, args.kind)

    S = dict(
        paths=paths, map_label=map_label,
        map_npz=exp_out_dir / (exp_out_dir.name + ".npz"),
        experimental_tif=paths.experiment_video_path(
            kind_token, args.cell, args.experiment_span_seconds, data_bank_root),
        out_dir=posit_dir / f"{paths.project_alias}_{map_label}_Posterior_Predictive_Video",
        prior_low=np.asarray(cfg.param_module.theta_lower_bound(), dtype=float),
        prior_high=np.asarray(cfg.param_module.theta_upper_bound(), dtype=float),
    )

    if cfg.tag == "detector":
        S["map_block"] = "imaging"
        S["map_keys"] = _wf_keys(cfg)                       # the 6 learnable imaging parameters
        S["imaging_desc"] = "MAP emitter parameters + MET SCOPE camera"
        S["rds_desc"] = "diffusion-only system from the RDS nuisance (no reactions)"

        def imaging_physical(a, map_theta):
            if a.fixed_imaging_parameters:
                overrides = _parse_kv(a.set_imaging, "--set-imaging")
                return _fixed_imaging_theta(overrides=overrides or None), "fixed MET values"
            # The camera is marginalized, not inferred: pin it to MET rather than drawing, because
            # the comparison is against one specific acquisition.
            return (np.concatenate([map_theta, _scope_met()]),
                    "MAP emitter parameters + MET SCOPE camera")

        def build_rds(a, map_theta):
            if a.fixed_nuisance_rds:
                nuisance = _fixed_nuisance_physical(
                    overrides=_parse_kv(a.fixed_nuisance_rds, "--fixed-nuisance-RDS"))
                desc = "pinned RDS nuisance"
            else:
                nuisance = draw_nuisance_physical()
                desc = "drawn RDS nuisance"
            smut, _theta = build_detector_rds_simulation(nuisance, seed=a.seed, verbose=a.verbose)
            return smut, nuisance, desc + " (diffusion-only)"
    else:
        S["map_block"] = "rds"
        S["map_keys"] = _wf_keys(cfg)                       # the 10 learnable RDS parameters
        S["imaging_desc"] = "calibrated Nuisance_DLI vector + MET SCOPE camera"
        S["rds_desc"] = "full reactive system from the MAP reaction-diffusion parameters"

        def imaging_physical(a, map_theta):
            # Biology holds imaging FIXED at the calibrated vector the training videos were
            # generated with, read from the Nuisance_DLI artifact at run time. The values live
            # only in that artifact -- they appear in no source file -- so hardcoding them here
            # would silently drift from whatever the videos were actually built with.
            from .detector_nuisance_dli import artifact_path, require_nuisance_dli
            det_paths = det.detector_paths(PARAMETERS.paths)
            nu = require_nuisance_dli(data_bank_root / det_paths.posit_subdir,
                                      det_paths.project_alias, map_label)
            emitter_log10 = np.asarray(nu.samples, dtype=float)
            if emitter_log10.shape[0] != 1:
                # A multi-vector artifact has no single "the" imaging vector; take its SGM so the
                # choice is the correlation-preserving one rather than an arbitrary row.
                span = np.ptp(10.0 ** emitter_log10, axis=0)
                span[span <= 0] = 1.0
                idx, _m = sample_geometric_median(10.0 ** emitter_log10, span)
                emitter_log10 = emitter_log10[idx:idx + 1]
            emitter = 10.0 ** emitter_log10[0]
            overrides = _parse_kv(a.set_imaging, "--set-imaging")
            vec = np.concatenate([emitter, _scope_met()])
            if overrides:
                find = {k: i for i, k in enumerate(det.DETECTOR_IMAGING_KEYS)}
                for k, v in overrides.items():
                    if k not in find:
                        raise ValueError(f"--set-imaging {k!r} is not an imaging parameter; "
                                         f"valid keys are {det.DETECTOR_IMAGING_KEYS}.")
                    vec[find[k]] = v
            n_pool = int(np.asarray(nu.samples).shape[0])
            return vec, (f"calibrated Nuisance_DLI vector ({artifact_path(data_bank_root / det_paths.posit_subdir, det_paths.project_alias, map_label).name}"
                         + ("" if n_pool == 1 else f", SGM of {n_pool} vectors")
                         + ") + MET SCOPE camera"
                         + (" [with --set-imaging overrides]" if overrides else ""))

        def build_rds(a, map_theta):
            # Reactions ARE the inference target here, so the full reactive system is built --
            # pure_diffusion would silently delete the very physics the MAP describes.
            if a.fixed_nuisance_rds:
                raise SystemExit("--fixed-nuisance-RDS applies to the detector workflow only: here "
                                 "the reaction-diffusion block is the MAP, not a nuisance.")
            stem = build_system(map_theta, pure_diffusion=False, verbose=a.verbose)
            smut = build_simulation(stem, map_theta, seed=a.seed, verbose=a.verbose)
            return smut, map_theta, "full reactive system from the MAP"

    S["imaging_physical"] = imaging_physical
    S["build_rds"] = build_rds
    return S


def _parse_kv(items, flag):
    out = {}
    for item in (items or []):
        key, sep, val = item.partition("=")
        if not sep:
            raise ValueError(f"{flag} expects KEY=VALUE, got {item!r}")
        out[key.strip()] = float(val)
    return out

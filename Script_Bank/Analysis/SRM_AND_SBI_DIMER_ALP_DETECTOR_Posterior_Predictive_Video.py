"""Analysis entry point (Detector workflow): posterior-predictive video check.

Render a synthetic video from the MAP imaging estimate of one experimental recording, persist it
next to the experimental recording, and (via the companion notebook
``notebooks/SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.ipynb``) compare them side by side. Post-hoc
analysis: lives in ``Script_Bank/Analysis``, is NOT a canonical stage, and is never wired
into the dispatcher. It imports and calls existing modules only; it renders and persists,
modifying no canonical stage.

Selection. The Detector Experiment MAP database is keyed by ``(kind, cell, chunk)`` -- each
chunk of a experimental recording carries one MAP imaging estimate. The user selects one
``(kind, cell, chunk)``; that entry's imaging theta drives the render. The chunk names a
concrete database entry (not an aggregate), so the selection maps one-to-one to the data.

Length. The rendered clip's length matches the EXPERIMENTAL recording's length, read from the
``.tif`` frame count (never hardcoded); the fixed frame cadence comes from the machine's
timing config. Because the imaging is effectively constant across a recording, one chunk's
imaging theta renders the full-length clip faithfully.

What it shows. The reaction-diffusion motion is drawn fresh from the RDS nuisance -- the MAP
fixes the imaging, not the motion -- so the synthetic is not a reconstruction of the experimental
track. The comparison reads the IMAGING appearance (point-spread size, brightness, noise,
flicker), i.e. whether the calibrated imaging makes a synthetic recording look like the experimental
one. A single deliberate run draws one trajectory; the notebook only views the persisted
clip, so scrubbing/zooming never regenerates it, and ``--seed`` makes a run reproducible.

Fixed-imaging-parameters mode (``--fixed-imaging-parameters``). A validation-gate variant that
does NOT read the trained MAP database. Instead the imaging theta is FIXED to correct-source
values: the camera parameters (``gamma = g/C``, ``kappa_o``, ``kappa_b``, ``kappa_s``, ``kappa_q``) are set to
the MET acquisition/config values (``MET_CAMERA_PHYSICAL``; see ``REFERENCE_EMCCD_NOISE_MODEL.md``
Sec. 6 and the MET provenance in ``DETECTOR_WORKFLOW.md`` §6.5), and
the remaining, non-camera parameters (PSF ``mu_r``/``sigma_r``, brightness ``mu_pc``/``sigma_pc``,
``lambda_rate``, ``prob_photo_bleach``) are held at their prior-center nominals. The intention: test
whether the CORRECTED forward model, driven by the correct camera physics rather than an inferred
posterior, recapitulates the experimental pixel-intensity histogram -- a training-free check that
the synthetic manifold now matches the real MET floor and signal, before any expensive
generate/train/eval/infer cycle. In this mode ``--chunk``/``--map-source`` are ignored and no MAP
database is required.

Notes. Two durations meet in the name. The ``{model_window}`` label (from
``--total-time-seconds``, e.g. ``2S_50FPS``) identifies the trained estimator -- the window its
MAP imaging estimate was inferred over -- and holds the canonical ``{alias}_{model_window}``
position. The ``{clip_span}`` token (e.g. ``20S``) is the rendered clip's own length, read from
the experimental recording; a 20 s clip rendered from a 2 s-window MAP is not the same as one from a
10 s window, so the span is carried in the descriptor. The estimator was calibrated on the fixed
8-bit rescale of the recordings (a global linear ``(0, 65535) -> (0, 255)``, not per-video); the
clip is persisted and displayed at full 16-bit, and the viewer/figure scale the colormap over the
full intensity range for visibility (configurable -- see ``--display-norm``; the raw-ADU
brightness comparison is the ``Comparison.png`` histogram).

Reads  <data_bank>/<posit>/<alias>_{model_window}_MAP_Experiment/<...>.npz  (MAP database)
       <data_bank>/<experiment_subdir>/Experiment_{KIND}_Cell_{cell}_{span}S_RAW.tif  (experimental)
Writes <data_bank>/<posit>/<alias>_{model_window}_Posterior_Predictive_Video/
         <stem>_Synthetic_Video.npz   (experimental + synthetic + provenance)
         <stem>_Comparison.png        (static side-by-side)
         <stem>_Trajectory.h5         (the drawn trajectory; provenance)
       where <stem> = <alias>_{model_window}_MAP_Estimate[_Median]_{KIND}_Cell_{cell}[_Chunk_{chunk}]_{clip_span}
       (or <alias>_{model_window}_Fixed_Imaging_{KIND}_Cell_{cell}_{clip_span} under --fixed-imaging-parameters)

Usage:
    MACHINE_PROFILE=<p> python .../SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.py \\
        --total-time-seconds 2.0 --kind ALP --cell 3 --chunk 5 [--seed 0]
    (add --dry-run to resolve + validate inputs without simulating or rendering)
"""
import argparse
import sys
import warnings
from pathlib import Path

import numpy as np

from srm_and_sbi_dimer_alp import detector_parameterization as det
from srm_and_sbi_dimer_alp.detector_simulation_dli_support import render_detector_video
from srm_and_sbi_dimer_alp.detector_simulation_rds_support import (
    build_detector_rds_simulation, draw_nuisance_physical)
from srm_and_sbi_dimer_alp.parameterization import PARAMETERS, RunTiming
from srm_and_sbi_dimer_alp.simulation_rds_support import extract_trajectory_poses


def _resolve(args):
    """Detector-namespaced paths, the MAP database, and the experimental recording."""
    paths = det.detector_paths(PARAMETERS.paths)
    data_bank_root = PARAMETERS.machine.data_bank_root
    map_timing = RunTiming(total_time_seconds=args.total_time_seconds,
                           frames=PARAMETERS.simulation.timing)
    map_label = map_timing.label
    posit_dir = data_bank_root / paths.posit_subdir
    exp_out_dir = paths.experiment_recovery_dir(data_bank_root, map_label)   # MAP_Experiment dir
    map_npz = exp_out_dir / (exp_out_dir.name + ".npz")
    experimental_tif = paths.experiment_video_path(
        args.kind, args.cell, args.experiment_span_seconds, data_bank_root)
    out_dir = posit_dir / f"{paths.project_alias}_{map_label}_Posterior_Predictive_Video"
    return paths, map_label, map_npz, experimental_tif, out_dir


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
    descriptor = "MAP_Estimate_Median" if map_source == "cell-median" else "MAP_Estimate"
    chunk_part = "" if map_source == "cell-median" else f"_Chunk_{chunk}"
    return (f"{project_alias}_{map_label}_{descriptor}{nuis_tok}_{kind}_Cell_{cell}"
            f"{chunk_part}_{clip_token}{label_tok}")


def _load_map_theta(map_npz, imaging_keys, kind, cell, chunk, source):
    """Physical imaging theta for (kind, cell): a single chunk's MAP (``source='chunk'``) or
    the per-cell median over all of that cell's chunk MAPs (``source='cell-median'``). Fails
    with guidance listing the available cells/chunks."""
    with np.load(str(map_npz), allow_pickle=True) as d:
        inferred_log10 = np.asarray(d["inferred_log10"], dtype=float)
        kind_index = np.asarray(d["kind_index"])
        cells = np.asarray(d["cell"])
        chunks = np.asarray(d["chunk"])
        kinds = [str(k) for k in d["kinds"]]
    if inferred_log10.ndim != 2 or inferred_log10.shape[1] != len(imaging_keys):
        raise ValueError(f"MAP database inferred_log10 has shape {inferred_log10.shape}; "
                         f"expected (N, {len(imaging_keys)}).")
    if kind not in kinds:
        raise ValueError(f"kind={kind!r} not in the MAP database; available: {kinds}.")
    ki = kinds.index(kind)
    cell_rows = np.where((kind_index == ki) & (cells == cell))[0]
    if cell_rows.size == 0:
        avail_cells = sorted({int(c) for c in cells[kind_index == ki]})
        raise ValueError(f"no MAP entries for kind={kind} cell={cell} in\n    {map_npz}\n"
                         f"available cells for {kind}: {avail_cells}")
    if source == "cell-median":
        theta_log10 = np.median(inferred_log10[cell_rows], axis=0)   # robust over the cell's chunks
        print(f"MAP source: per-cell median over {cell_rows.size} chunk(s) of {kind} cell {cell}.")
    else:                                                            # a concrete database entry
        row = np.where((kind_index == ki) & (cells == cell) & (chunks == chunk))[0]
        if row.size == 0:
            avail_chunks = sorted({int(c) for c in chunks[cell_rows]})
            raise ValueError(f"no MAP entry for kind={kind} cell={cell} chunk={chunk} in\n"
                             f"    {map_npz}\navailable chunks for {kind} cell {cell}: {avail_chunks}")
        theta_log10 = inferred_log10[int(row[0])]
        print(f"MAP source: chunk {chunk} of {kind} cell {cell}.")
    plo = np.asarray(det.theta_lower_bound(), dtype=float)
    phi = np.asarray(det.theta_upper_bound(), dtype=float)
    oob = [imaging_keys[i] for i in range(len(imaging_keys))
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
                         nuisance, imaging_physical, fixed_imaging=False, fixed_nuisance=False):
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
    imaging_header = ("FIXED imaging (correct-source MET, absolute)" if fixed_imaging
                      else "INFERRED imaging (MAP theta, absolute)")
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
    for ent, v in zip(det.DETECTOR_NUISANCE, nuis):
        lab2 = ent["LABEL"].translate({ord(c): None for c in "${}\\"})
        npart.append(f"{lab2}={v:.0f}" if str(ent.get("UNIT", "")).lower().startswith("count")
                     else f"{lab2}={v:.3g}")
    nuis_lines = "\n".join("  " + "  ".join(npart[j:j + 3]) for j in range(0, len(npart), 3))
    tbl = f"{'quantile':<8}{'exp':>7}{'synth':>7}{'ratio':>7}\n"
    for name, ev, sv in zip(q_names, eq, sq):
        tbl += f"{name:<8}{ev:7.0f}{sv:7.0f}{(sv / ev if ev > 0 else float('inf')):7.2f}\n"
    ax[1, 3].text(
        0.0, 1.0,
        "QUANTILES (ADU)  ratio=synth/exp\n" + tbl + "\n"
        f"{imaging_header}:\n{dli_lines}\n  (* outside prior)\n"
        f"NUISANCE (RDS):\n{nuis_lines}\n"
        f"motion: {'fixed nuisance (pinned)' if fixed_nuisance else 'fresh draw'}; norm {display_norm}",
        fontsize=6.5, va="top", family="monospace")
    fig.suptitle(fig_title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(args):
    imaging_keys = [e["KEY"] for e in det.DETECTOR_PARAMETERIZATION]
    paths, map_label, map_npz, experimental_tif, out_dir = _resolve(args)
    frame_time = PARAMETERS.simulation.timing.frame_time_seconds

    if args.dry_run:
        n_frames = None
        if experimental_tif.exists():
            import tifffile
            with tifffile.TiffFile(str(experimental_tif)) as tf:
                n_frames = int(tf.series[0].shape[0])   # frame count from metadata; no pixel load
        clip_token = _clip_span_token(n_frames, frame_time) if n_frames else None
        stem = _build_stem(paths.project_alias, map_label, args.kind, args.cell,
                           args.chunk, args.map_source, clip_token or "<clip_span>",
                           fixed_imaging=args.fixed_imaging_parameters,
                           fixed_nuisance=bool(args.fixed_nuisance_rds),
                           run_label=args.run_label)
        print("[DRY RUN] posterior-predictive video would read:")
        if args.fixed_imaging_parameters:
            print("    imaging theta: FIXED to correct-source MET values (no MAP database read); "
                  "non-camera parameters at prior-center nominals")
        else:
            print(f"    MAP database : {map_npz}  [{'OK' if map_npz.exists() else 'MISSING'}]")
        print(f"    experimental .tif    : {experimental_tif}  [{'OK' if experimental_tif.exists() else 'MISSING'}]")
        if args.fixed_imaging_parameters:
            print(f"    selection    : kind={args.kind} cell={args.cell}  (imaging: fixed correct-source)")
        else:
            sel = f"chunk {args.chunk}" if args.map_source == "chunk" else "cell-median (over all chunks)"
            print(f"    selection    : kind={args.kind} cell={args.cell}  (MAP source: {sel})")
        cl = f"{clip_token} (from the .tif)" if clip_token else "the experimental recording's frame count (read at run time)"
        print(f"    render length: {cl}")
        print(f"    display norm : {args.display_norm}")
        print(f"[DRY RUN] would write under:\n    {out_dir}/\n"
              f"        {stem}_{{Synthetic_Video.npz,Comparison.png,Trajectory.h5}}")
        return

    if not args.fixed_imaging_parameters and not map_npz.exists():
        raise FileNotFoundError(
            f"MAP database not found:\n    {map_npz}\nRun the Detector Experiment stage first, "
            f"or pass --fixed-imaging-parameters to skip it.")
    if not experimental_tif.exists():
        raise FileNotFoundError(
            f"Experimental recording not found:\n    {experimental_tif}\n"
            f"(check --kind/--cell and --experiment-span-seconds={args.experiment_span_seconds}).")

    # ---- imaging theta: fixed correct-source values, or the selected MAP entry ----
    if args.fixed_imaging_parameters:
        overrides = {}
        for item in args.set_imaging:
            key, sep, val = item.partition("=")
            if not sep:
                raise ValueError(f"--set-imaging expects KEY=VALUE, got {item!r}")
            overrides[key.strip()] = float(val)
        imaging_physical = _fixed_imaging_theta(overrides=overrides or None)
    else:
        map_learnable = _load_map_theta(map_npz, imaging_keys, args.kind, args.cell,
                                        args.chunk, args.map_source)          # (6,) physical, learnable order
        # The camera is marginalized (not inferred). For a posterior-predictive check against the
        # real MET recording -- acquired with one specific camera -- pin the 5 SCOPE parameters to
        # their correct-source MET values rather than a random SCOPE draw, matching --fixed-imaging.
        scope_met = np.array([MET_CAMERA_PHYSICAL[k] for k in det.DETECTOR_SCOPE_KEYS], dtype=float)
        imaging_physical = np.concatenate([map_learnable, scope_met])        # (11,) DETECTOR_IMAGING order

    # ---- experimental recording -> the render length (from the data, never hardcoded) ----
    import tifffile
    experimental = np.asarray(tifffile.imread(str(experimental_tif)))          # (n_frames, H, W)
    if experimental.ndim != 3:
        raise ValueError(f"experimental recording has shape {experimental.shape}; expected (n_frames, H, W).")
    n_frames = int(experimental.shape[0])
    render_timing = RunTiming(total_time_seconds=n_frames * frame_time,
                              frames=PARAMETERS.simulation.timing)
    clip_token = _clip_span_token(n_frames, frame_time)
    stem = _build_stem(paths.project_alias, map_label, args.kind, args.cell,
                       args.chunk, args.map_source, clip_token,
                       fixed_imaging=args.fixed_imaging_parameters,
                       fixed_nuisance=bool(args.fixed_nuisance_rds),
                       run_label=args.run_label)
    if args.fixed_imaging_parameters:
        print(f"Experimental recording: {n_frames} frames ({clip_token} clip); rendering a matching "
              f"synthetic with fixed correct-source imaging ({args.kind} cell {args.cell}).")
    else:
        src_desc = f"chunk {args.chunk}" if args.map_source == "chunk" else "cell-median"
        print(f"Experimental recording: {n_frames} frames ({clip_token} clip); rendering a matching "
              f"synthetic from the MAP ({args.kind} cell {args.cell}, {src_desc}).")

    # ---- draw the RDS nuisance and simulate one trajectory at the experimental length ----
    import readdy
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.fixed_nuisance_rds:
        nuis_overrides = {}
        for item in args.fixed_nuisance_rds:
            key, sep, val = item.partition("=")
            if not sep:
                raise ValueError(f"--fixed-nuisance-RDS expects KEY=VALUE, got {item!r}")
            nuis_overrides[key.strip()] = float(val)
        nuisance = _fixed_nuisance_physical(overrides=nuis_overrides)
    else:
        nuisance = draw_nuisance_physical()
    smut, _theta = build_detector_rds_simulation(nuisance, seed=args.seed, verbose=args.verbose)
    traj_path = out_dir / f"{stem}_Trajectory.h5"
    if traj_path.exists():
        traj_path.unlink()
    smut.output_file = str(traj_path)
    smut.progress_output_stride = render_timing.total_steps
    smut.run(n_steps=render_timing.total_steps,
             timestep=render_timing.delta_time_nanoseconds * readdy.units.nanosecond,
             show_summary=False)

    # ---- poses -> render with the MAP imaging theta ----
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
    synth = render_detector_video(pro_tray_poses=pro_tray_poses,
                                  imaging_physical=imaging_physical,
                                  dimer_mask=dimer_mask, dimer_model=args.dimer_model,
                                  seed=args.seed, verbose=args.verbose)
    synth = np.moveaxis(synth, -1, 0)                          # (H, W, n_frames) -> (n_frames, H, W)

    # ---- persist (clip + provenance) and a static comparison figure ----
    clips_path = out_dir / f"{stem}_Synthetic_Video.npz"
    # Storage clip to the non-negative uint16 range. The reference EMCCD noise
    # model (simulation_dli_support.add_noise) emits a small signed excursion --
    # a Gaussian read-noise tail below 0, and rarely values above 65535 -- that a
    # experimental camera cannot record. We clip so the stored and compared synthetic
    # lives in the same non-negative domain as the experimental frames. This clip is a
    # deliberate, revisit-able choice: n_under / n_over quantify the excursion on
    # every run, so a future or alternative noise model's negative/overflow
    # behavior can be compared at this exact point. See
    # REFERENCE_EMCCD_NOISE_MODEL.md for the corrected Poisson-Gamma-Normal model.
    n_under = int(np.count_nonzero(synth < 0.0))
    n_over = int(np.count_nonzero(synth > 65535.0))
    if n_under or n_over:
        print(f"  storage clip to [0, 65535]: {n_under} pixel(s) < 0, {n_over} pixel(s) > 65535 "
              f"(of {synth.size}); clipped to match the experimental non-negative domain.")
    synth_u16 = np.clip(np.rint(synth), 0, 65535).astype(np.uint16)
    np.savez_compressed(
        str(clips_path),
        experimental=experimental.astype(np.uint16), synth=synth_u16,        # native 16-bit; the viewer scales for display
        imaging_physical=imaging_physical, imaging_keys=np.array(det.DETECTOR_IMAGING_KEYS),
        nuisance=nuisance, kind=args.kind, cell=args.cell,
        chunk=(-1 if args.chunk is None else args.chunk), map_source=args.map_source,
        seed=(-1 if args.seed is None else args.seed),
        frame_time_seconds=frame_time, n_frames=n_frames, experimental_tif=str(experimental_tif))
    sel_desc = ("fixed correct-source" if args.fixed_imaging_parameters else
                (f"chunk {args.chunk}" if args.map_source == "chunk" else "cell-median"))
    _save_comparison_png(out_dir / f"{stem}_Comparison.png", experimental, synth_u16,
                         args.kind, args.cell, sel_desc, args.display_norm,
                         nuisance, imaging_physical, fixed_imaging=args.fixed_imaging_parameters,
                         fixed_nuisance=bool(args.fixed_nuisance_rds))
    print(f"Persisted clip + provenance:\n    {clips_path}")
    print(f"Static comparison figure:\n    {out_dir / (stem + '_Comparison.png')}")
    print(f"Trajectory (provenance):\n    {traj_path}")
    print("\nView interactively (scrub / play / zoom) with notebooks/SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.ipynb.")


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Render a synthetic video from a experimental recording's MAP imaging estimate "
                    "and persist it for a experimental-vs-synthetic visual comparison (analysis step).")
    p.add_argument("--total-time-seconds", type=float, required=True,
                   help="model window of the trained estimator; locates the MAP database "
                        "(e.g. 2.0 -> 2S_50FPS). Does NOT set the render length.")
    p.add_argument("--kind", type=str, required=True,
                   help="experimental condition (e.g. ALP, BET); validated against the database.")
    p.add_argument("--cell", type=int, required=True, help="experimental cell index.")
    p.add_argument("--chunk", type=int, default=None,
                   help="chunk index selecting the MAP entry for (kind, cell); required for "
                        "--map-source=chunk (default), ignored for cell-median.")
    p.add_argument("--map-source", choices=("chunk", "cell-median"), default="chunk",
                   help="imaging theta source: 'chunk' = the MAP at the selected (kind, cell, "
                        "chunk) entry (default); 'cell-median' = median over all of the cell's "
                        "chunk MAPs (a robust single estimate). Ignored with --fixed-imaging-parameters.")
    p.add_argument("--fixed-imaging-parameters", action="store_true",
                   help="VALIDATION-GATE mode: do NOT read the MAP database; instead fix the imaging "
                        "theta to correct-source values -- camera parameters at the MET config values "
                        "(MET_CAMERA_PHYSICAL), non-camera parameters at prior-center nominals -- to "
                        "test whether the corrected forward model recapitulates the experimental pixel "
                        "histogram with no trained posterior. --chunk/--map-source are ignored.")
    p.add_argument("--set-imaging", action="append", default=[], metavar="KEY=VALUE",
                   help="override a fixed-imaging parameter with a physical value (repeatable), e.g. "
                        "--set-imaging mu_pc=397.5 --set-imaging sigma_pc=0.604 (brightness/PSF "
                        "calculated from the MET ThunderSTORM localizations). Only with "
                        "--fixed-imaging-parameters; applied on top of the MET camera values and "
                        "prior-center nominals. Values outside the training prior render as-is "
                        "(extrapolated) and are flagged '*' in the figure.")
    p.add_argument("--fixed-nuisance-RDS", dest="fixed_nuisance_rds",
                   action="append", default=[], metavar="KEY=VALUE",
                   help="fix the RDS nuisance -- the reaction-diffusion parameters that build the "
                        "trajectory -- instead of drawing it fresh from the (flat) nuisance prior: "
                        "set an RDS-nuisance parameter to a physical value (repeatable), e.g. "
                        "--fixed-nuisance-RDS count_alp=134 --fixed-nuisance-RDS count_bet=35 "
                        "--fixed-nuisance-RDS count_chi=31. Keys: count_alp / count_bet / count_chi "
                        "(monomer / mobile-dimer / immobile-dimer counts) and diffusivity_alp / "
                        "relative_diffusivity_bet / relative_diffusivity_chi. Any key left unset is "
                        "held at its prior-center nominal. Passing this flag at all switches the "
                        "render from a random nuisance draw to this deterministic nuisance, so the "
                        "monomer:dimer split (hence the bright-pixel tail) is condition-appropriate "
                        "rather than the prior's implausibly dimer-heavy expectation. Independent of "
                        "--fixed-imaging-parameters.")
    p.add_argument("--run-label", type=str, default=None, metavar="TEXT",
                   help="optional short tag inserted into the output basename (e.g. 'Pure' / "
                        "'Mixed') so renders that share the same selection -- same cell, both "
                        "fixed-imaging + fixed-nuisance -- do not overwrite each other. "
                        "Non-alphanumeric characters are replaced with '_'.")
    p.add_argument("--dimer-model", dest="dimer_model", choices=("multiply", "sum"), default="sum",
                   help="how a dimer's two labels combine: 'sum' (default) adds an independent "
                        "second-label brightness draw (sum of two monomers -- same mean, lighter "
                        "upper tail; Mutch 2007 convolution / Digman-Gratton Number & Brightness); "
                        "'multiply' scales one draw by dimer_mule (=2) (heavier tail; retained as an option).")
    p.add_argument("--experiment-span-seconds", type=int, default=20,
                   help="recording length in seconds, used only to locate the .tif (default 20); "
                        "the render length is read from the .tif's actual frame count.")
    p.add_argument("--display-norm", choices=("full", "autoscale", "percentile"), default="full",
                   help="color scaling for the Comparison.png image panels. The experimental and "
                        "synthetic panels ALWAYS share ONE limit (identical intensities map to identical "
                        "colors); the mode only sets the shared mid-frame window: 'full' (default) the "
                        "whole-clip [min, max] (nothing clipped); 'percentile' the whole-clip [min, "
                        "p99.99] (drops the hot-pixel sliver for contrast); 'autoscale' the two displayed "
                        "frames' shared range. The max-projection panels always share the full range. "
                        "Figure-only; does not touch the persisted pixels.")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for the RDS sim + render; default None (fresh draw). Fix it "
                        "for a reproducible trajectory across re-runs.")
    p.add_argument("--verbose", action="store_true", help="verbose sim/render diagnostics.")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve + validate inputs and print what would be read/written; no sim.")
    args = p.parse_args(argv)
    if args.set_imaging and not args.fixed_imaging_parameters:
        p.error("--set-imaging is only valid with --fixed-imaging-parameters.")
    if not args.fixed_imaging_parameters and args.map_source == "chunk" and args.chunk is None:
        p.error("--chunk is required with --map-source=chunk (the default); "
                "pass --map-source cell-median to aggregate over the cell's chunks instead, "
                "or --fixed-imaging-parameters to skip the MAP database entirely.")
    return args


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))

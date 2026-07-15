"""Analysis entry point (Detector workflow): posterior-predictive video check.

Render a synthetic video from the MAP imaging estimate of one real recording, persist it
next to the real recording, and (via the companion notebook
``notebooks/SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.ipynb``) compare them side by side. Post-hoc
analysis: lives in ``Script_Bank/Analysis``, is NOT a canonical stage, and is never wired
into the dispatcher. It imports and calls existing modules only; it renders and persists,
modifying no canonical stage.

Selection. The Detector Experiment MAP database is keyed by ``(kind, cell, chunk)`` -- each
chunk of a real recording carries one MAP imaging estimate. The user selects one
``(kind, cell, chunk)``; that entry's imaging theta drives the render. The chunk names a
concrete database entry (not an aggregate), so the selection maps one-to-one to the data.

Length. The rendered clip's length matches the REAL recording's length, read from the
``.tif`` frame count (never hardcoded); the fixed frame cadence comes from the machine's
timing config. Because the imaging is effectively constant across a recording, one chunk's
imaging theta renders the full-length clip faithfully.

What it shows. The reaction-diffusion motion is drawn fresh from the RDS nuisance -- the MAP
fixes the imaging, not the motion -- so the synthetic is not a reconstruction of the real
track. The comparison reads the IMAGING appearance (point-spread size, brightness, noise,
flicker), i.e. whether the calibrated imaging makes a synthetic recording look like the real
one. A single deliberate run draws one trajectory; the notebook only views the persisted
clip, so scrubbing/zooming never regenerates it, and ``--seed`` makes a run reproducible.

Notes. Two durations meet in the name. The ``{model_window}`` label (from
``--total-time-seconds``, e.g. ``2S_50FPS``) identifies the trained estimator -- the window its
MAP imaging estimate was inferred over -- and holds the canonical ``{alias}_{model_window}``
position. The ``{clip_span}`` token (e.g. ``20S``) is the rendered clip's own length, read from
the real recording; a 20 s clip rendered from a 2 s-window MAP is not the same as one from a
10 s window, so the span is carried in the descriptor. The estimator was calibrated on the fixed
8-bit rescale of the recordings (a global linear ``(0, 65535) -> (0, 255)``, not per-video); the
clip is persisted and displayed at full 16-bit, and the viewer/figure autoscale the colormap to
each image's own intensity range for visibility (the raw-ADU brightness comparison is the
``Comparison.png`` histogram).

Reads  <data_bank>/<posit>/<alias>_{model_window}_MAP_Experiment/<...>.npz  (MAP database)
       <data_bank>/<experiment_subdir>/Experiment_{KIND}_Cell_{cell}_{span}S_RAW.tif  (real)
Writes <data_bank>/<posit>/<alias>_{model_window}_Posterior_Predictive_Video/
         <stem>_Synthetic_Video.npz   (real + synthetic + provenance)
         <stem>_Comparison.png        (static side-by-side)
         <stem>_Trajectory.h5         (the drawn trajectory; provenance)
       where <stem> = <alias>_{model_window}_MAP_Estimate[_Median]_{KIND}_Cell_{cell}[_Chunk_{chunk}]_{clip_span}

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
    """Detector-namespaced paths, the MAP database, and the real recording."""
    paths = det.detector_paths(PARAMETERS.paths)
    data_bank_root = PARAMETERS.machine.data_bank_root
    map_timing = RunTiming(total_time_seconds=args.total_time_seconds,
                           frames=PARAMETERS.simulation.timing)
    map_label = map_timing.label
    posit_dir = data_bank_root / paths.posit_subdir
    exp_out_dir = paths.experiment_recovery_dir(data_bank_root, map_label)   # MAP_Experiment dir
    map_npz = exp_out_dir / (exp_out_dir.name + ".npz")
    real_tif = paths.experiment_video_path(
        args.kind, args.cell, args.experiment_span_seconds, data_bank_root)
    out_dir = posit_dir / f"{paths.project_alias}_{map_label}_Posterior_Predictive_Video"
    return paths, map_label, map_npz, real_tif, out_dir


def _clip_span_token(n_frames, frame_time):
    """The clip's own duration as a label token (e.g. 1000 frames @ 0.02 s -> ``20S``); the
    canonical duration labeling, so the token matches the rest of the codebase."""
    return RunTiming(total_time_seconds=n_frames * frame_time,
                     frames=PARAMETERS.simulation.timing).label.split("_")[0]


def _build_stem(project_alias, map_label, kind, cell, chunk, map_source, clip_token):
    """Output basename: canonical ``{alias}_{model_window}`` + MAP-estimate descriptor + clip
    span. ``map_label`` (e.g. ``2S_50FPS``) is the estimator's model window; ``clip_token``
    (e.g. ``20S``) is the rendered length. Chunk source names the concrete entry
    (``..._Cell_{cell}_Chunk_{chunk}_...``); cell-median drops the chunk and marks ``Median``."""
    descriptor = "MAP_Estimate_Median" if map_source == "cell-median" else "MAP_Estimate"
    chunk_part = "" if map_source == "cell-median" else f"_Chunk_{chunk}"
    return (f"{project_alias}_{map_label}_{descriptor}_{kind}_Cell_{cell}"
            f"{chunk_part}_{clip_token}")


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
              f"there, so a poor real-vs-synthetic match may reflect an out-of-prior MAP rather "
              f"than the calibration itself.")
    return np.power(10.0, theta_log10)


def _save_comparison_png(path, real, synth, kind, cell, sel_desc, display_norm):
    """Static real-vs-synthetic panel. Columns pair each condition -- REAL (col 0) and SYNTH
    (col 1), each showing the mid frame over its max projection; the third column holds the
    shared pixel-intensity histogram (top) and a provenance text box (bottom).

    Both panels and the histogram use the STORED synthetic (``synth_u16``, clipped to the
    non-negative uint16 range), so the comparison is like-with-like against the real frames and
    matches the persisted clip -- not the pre-clip float. ``display_norm`` sets the mid-frame
    color scaling: ``autoscale`` stretches magma to each image's own min/max (``vmin=vmax=None``,
    the reference viewer's behavior); ``percentile`` uses a fixed per-video window ``[p0.5,
    p99.5]``. The max-projection panels always autoscale (a summary statistic with its own range),
    and the histogram is in ADU either way."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mid = real.shape[0] // 2
    label = {"ALP": "MET-FAB", "BET": "MET-INLB"}.get(kind, kind)
    fig, ax = plt.subplots(2, 3, figsize=(13, 8.5), dpi=200)

    def _clim(arr_full):
        if display_norm == "percentile":
            return float(np.percentile(arr_full, 0.5)), float(np.percentile(arr_full, 99.5))
        return None, None                        # autoscale: magma stretched to the image range

    def _frame(a, arr, title, clim):
        a.imshow(arr, cmap="magma", origin="lower", interpolation="none",
                 vmin=clim[0], vmax=clim[1])
        a.set_title(title, fontsize=9); a.set_xticks([]); a.set_yticks([])

    real_clim, synth_clim = _clim(real), _clim(synth)
    # Columns pair each condition: col 0 = REAL (frame over max projection), col 1 = SYNTH
    # (same), col 2 = the shared pixel-intensity histogram (top) and the provenance text (bottom).
    _frame(ax[0, 0], real[mid], f"REAL {label} cell {cell}  frame {mid}", real_clim)
    _frame(ax[1, 0], real.max(0), "REAL  max projection", (None, None))
    _frame(ax[0, 1], synth[mid], f"SYNTH (MAP {kind} c{cell} {sel_desc})  frame {mid}", synth_clim)
    _frame(ax[1, 1], synth.max(0), "SYNTH  max projection", (None, None))
    ax[0, 2].hist(real.ravel(), bins=120, histtype="step", density=True, color="tab:blue",
                  label="real"); ax[0, 2].hist(synth.ravel(), bins=120, histtype="step",
                  density=True, color="tab:orange", label="synth")
    ax[0, 2].set_yscale("log"); ax[0, 2].set_title("pixel-intensity density (ADU)", fontsize=9)
    ax[0, 2].legend(fontsize=8)
    ax[1, 2].axis("off")
    ax[1, 2].text(0.0, 0.5, f"real  n_frames={real.shape[0]}  {real.shape[1]}x{real.shape[2]}\n"
                  f"synth n_frames={synth.shape[0]}  {synth.shape[1]}x{synth.shape[2]}\n\n"
                  "motion: fresh RDS-nuisance draw\nimaging: MAP theta of the selected chunk\n"
                  "comparison reads imaging appearance,\nnot the real trajectory.\n"
                  "synth shown as stored (clipped to >=0)\n\n"
                  f"shown at 16-bit; estimator calibrated\non the fixed 8-bit rescale.\n"
                  f"display norm: {display_norm}",
                  fontsize=9, va="center", family="monospace")
    fig.suptitle("Posterior-predictive video check (real vs synthetic-from-MAP)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def main(args):
    imaging_keys = [e["KEY"] for e in det.DETECTOR_PARAMETERIZATION]
    paths, map_label, map_npz, real_tif, out_dir = _resolve(args)
    frame_time = PARAMETERS.simulation.timing.frame_time_seconds

    if args.dry_run:
        n_frames = None
        if real_tif.exists():
            import tifffile
            with tifffile.TiffFile(str(real_tif)) as tf:
                n_frames = int(tf.series[0].shape[0])   # frame count from metadata; no pixel load
        clip_token = _clip_span_token(n_frames, frame_time) if n_frames else None
        stem = _build_stem(paths.project_alias, map_label, args.kind, args.cell,
                           args.chunk, args.map_source, clip_token or "<clip_span>")
        print("[DRY RUN] posterior-predictive video would read:")
        print(f"    MAP database : {map_npz}  [{'OK' if map_npz.exists() else 'MISSING'}]")
        print(f"    real .tif    : {real_tif}  [{'OK' if real_tif.exists() else 'MISSING'}]")
        sel = f"chunk {args.chunk}" if args.map_source == "chunk" else "cell-median (over all chunks)"
        print(f"    selection    : kind={args.kind} cell={args.cell}  (MAP source: {sel})")
        cl = f"{clip_token} (from the .tif)" if clip_token else "the real recording's frame count (read at run time)"
        print(f"    render length: {cl}")
        print(f"    display norm : {args.display_norm}")
        print(f"[DRY RUN] would write under:\n    {out_dir}/\n"
              f"        {stem}_{{Synthetic_Video.npz,Comparison.png,Trajectory.h5}}")
        return

    if not map_npz.exists():
        raise FileNotFoundError(
            f"MAP database not found:\n    {map_npz}\nRun the Detector Experiment stage first.")
    if not real_tif.exists():
        raise FileNotFoundError(
            f"Real recording not found:\n    {real_tif}\n"
            f"(check --kind/--cell and --experiment-span-seconds={args.experiment_span_seconds}).")

    # ---- imaging theta from the selected MAP entry ----
    imaging_physical = _load_map_theta(map_npz, imaging_keys, args.kind, args.cell,
                                       args.chunk, args.map_source)

    # ---- real recording -> the render length (from the data, never hardcoded) ----
    import tifffile
    real = np.asarray(tifffile.imread(str(real_tif)))          # (n_frames, H, W)
    if real.ndim != 3:
        raise ValueError(f"real recording has shape {real.shape}; expected (n_frames, H, W).")
    n_frames = int(real.shape[0])
    render_timing = RunTiming(total_time_seconds=n_frames * frame_time,
                              frames=PARAMETERS.simulation.timing)
    clip_token = _clip_span_token(n_frames, frame_time)
    stem = _build_stem(paths.project_alias, map_label, args.kind, args.cell,
                       args.chunk, args.map_source, clip_token)
    src_desc = f"chunk {args.chunk}" if args.map_source == "chunk" else "cell-median"
    print(f"Real recording: {n_frames} frames ({clip_token} clip); rendering a matching "
          f"synthetic from the MAP ({args.kind} cell {args.cell}, {src_desc}).")

    # ---- draw the RDS nuisance and simulate one trajectory at the real length ----
    import readdy
    out_dir.mkdir(parents=True, exist_ok=True)
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
                                  dimer_mask=dimer_mask, seed=args.seed, verbose=args.verbose)
    synth = np.moveaxis(synth, -1, 0)                          # (H, W, n_frames) -> (n_frames, H, W)

    # ---- persist (clip + provenance) and a static comparison figure ----
    clips_path = out_dir / f"{stem}_Synthetic_Video.npz"
    # Storage clip to the non-negative uint16 range. The reference EMCCD noise
    # model (simulation_dli_support.add_noise) emits a small signed excursion --
    # a Gaussian read-noise tail below 0, and rarely values above 65535 -- that a
    # real camera cannot record. We clip so the stored and compared synthetic
    # lives in the same non-negative domain as the real frames. This clip is a
    # deliberate, revisit-able choice: n_under / n_over quantify the excursion on
    # every run, so a future or alternative noise model's negative/overflow
    # behavior can be compared at this exact point. See PROJECT_CONTEXT.md
    # (DLI Imaging, read-noise note) for why the noise term itself is kept as-is.
    n_under = int(np.count_nonzero(synth < 0.0))
    n_over = int(np.count_nonzero(synth > 65535.0))
    if n_under or n_over:
        print(f"  storage clip to [0, 65535]: {n_under} pixel(s) < 0, {n_over} pixel(s) > 65535 "
              f"(of {synth.size}); clipped to match the real non-negative domain.")
    synth_u16 = np.clip(np.rint(synth), 0, 65535).astype(np.uint16)
    np.savez_compressed(
        str(clips_path),
        real=real.astype(np.uint16), synth=synth_u16,        # native 16-bit; the viewer scales for display
        imaging_physical=imaging_physical, imaging_keys=np.array(imaging_keys),
        nuisance=nuisance, kind=args.kind, cell=args.cell,
        chunk=(-1 if args.chunk is None else args.chunk), map_source=args.map_source,
        seed=(-1 if args.seed is None else args.seed),
        frame_time_seconds=frame_time, n_frames=n_frames, real_tif=str(real_tif))
    sel_desc = f"chunk {args.chunk}" if args.map_source == "chunk" else "cell-median"
    _save_comparison_png(out_dir / f"{stem}_Comparison.png", real, synth_u16,
                         args.kind, args.cell, sel_desc, args.display_norm)
    print(f"Persisted clip + provenance:\n    {clips_path}")
    print(f"Static comparison figure:\n    {out_dir / (stem + '_Comparison.png')}")
    print(f"Trajectory (provenance):\n    {traj_path}")
    print("\nView interactively (scrub / play / zoom) with notebooks/SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.ipynb.")


def parse_args(argv):
    p = argparse.ArgumentParser(
        description="Render a synthetic video from a real recording's MAP imaging estimate "
                    "and persist it for a real-vs-synthetic visual comparison (analysis step).")
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
                        "chunk MAPs (a robust single estimate).")
    p.add_argument("--experiment-span-seconds", type=int, default=20,
                   help="recording length in seconds, used only to locate the .tif (default 20); "
                        "the render length is read from the .tif's actual frame count.")
    p.add_argument("--display-norm", choices=("autoscale", "percentile"), default="autoscale",
                   help="color scaling for the Comparison.png mid-frame panels: 'autoscale' "
                        "(default) stretches magma to each image's own min/max (the reference "
                        "viewer's behavior); 'percentile' uses a fixed per-video [p0.5, p99.5] "
                        "window. Does not affect the persisted 16-bit pixels, only the figure.")
    p.add_argument("--seed", type=int, default=None,
                   help="RNG seed for the RDS sim + render; default None (fresh draw). Fix it "
                        "for a reproducible trajectory across re-runs.")
    p.add_argument("--verbose", action="store_true", help="verbose sim/render diagnostics.")
    p.add_argument("--dry-run", action="store_true",
                   help="resolve + validate inputs and print what would be read/written; no sim.")
    args = p.parse_args(argv)
    if args.map_source == "chunk" and args.chunk is None:
        p.error("--chunk is required with --map-source=chunk (the default); "
                "pass --map-source cell-median to aggregate over the cell's chunks instead.")
    return args


if __name__ == "__main__":
    main(parse_args(sys.argv[1:]))

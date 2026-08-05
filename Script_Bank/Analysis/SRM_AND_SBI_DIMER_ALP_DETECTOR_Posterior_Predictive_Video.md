# Posterior-predictive video — usage and interpretation

Companion to `SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.py` and its viewer
notebook `notebooks/SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.ipynb`. The
script renders a synthetic video from the MAP imaging estimate of one experimental recording and
persists it beside a static comparison figure; the notebook views the persisted clip. This
note explains how to run the script and how to read its outputs, so the check can be used
and understood without reverse-engineering the code.

This is a post-hoc, ad-hoc analysis — a visual posterior-predictive check on the Detector's
imaging calibration. It is not one of the canonical pipeline stages and is kept out of the
stage dispatcher.

## What it does

The Detector Experiment stage estimates the imaging parameters (camera, point-spread,
brightness, flicker) of each experimental recording by MAP, keyed by `(kind, cell, chunk)`. This
script takes one such estimate, renders a synthetic recording under it at the experimental
recording's own length, and places the two side by side. A close match is evidence that the
calibrated imaging model reproduces how an experimental recording looks; a poor match points to an
imaging parameter the calibration missed.

The synthetic's **motion** is a fresh reaction-diffusion draw, not the experimental track — the MAP
fixes the imaging, not the trajectory — so the comparison reads **imaging appearance**
(point-spread size, brightness, noise, flicker), not the specific molecular motion.

## How to run it

Run on a machine that holds the MAP database and the experimental recordings, under the render
environment (the project package plus ReaDDy). Preview first with `--dry-run`, which resolves
the inputs and the output names without simulating:

    MACHINE_PROFILE=<profile> python \
      Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.py \
      --total-time-seconds 2.0 --kind ALP --cell 3 --chunk 5 [--seed 0] [--dry-run]

Arguments:

- `--total-time-seconds` — the trained estimator's model window (e.g. `2.0` → `2S_50FPS`). It
  locates the MAP database; it does **not** set the render length.
- `--kind`, `--cell` — the experimental condition (e.g. `ALP`, `BET`) and the cell index,
  validated against the database.
- `--chunk` — selects one MAP entry for `(kind, cell)`; required in the default
  `--map-source chunk` mode.
- `--map-source` — `chunk` (the MAP at the selected `(kind, cell, chunk)`, the default) or
  `cell-median` (the per-cell median over that cell's chunk MAPs, a robust single estimate;
  `--chunk` is then ignored).
- `--experiment-span-seconds` — recording length used only to locate the `.tif` (default
  `20`); the render length is read from the `.tif`'s own frame count.
- `--display-norm` — color scaling for the comparison figure's frame panels: `full`
  (default; a shared full-range `[min, max]` window over both the experimental and synthetic
  pixels, so identical intensities map to identical colors and nothing is clipped), `autoscale`
  (each displayed frame stretched to its own min/max, per-frame), or `percentile` (a fixed
  whole-clip `[min, p99.99]` window). The notebook uses the identical convention, so a given
  frame renders the same in the static figure and the notebook. It does not change the stored
  pixels.
- `--seed` — RNG seed for the simulation and render; each run otherwise draws a fresh motion
  realization (the check reads imaging appearance, not the specific track).

## Outputs

Written to `<data_bank>/<posit>/<alias>_<model_window>_Posterior_Predictive_Video/`, with
`<stem> = <alias>_<model_window>_MAP_Estimate[_Median]_<KIND>_Cell_<cell>[_Chunk_<chunk>]_<clip_span>`:

- `<stem>_Synthetic_Video.npz` — experimental and synthetic frames (16-bit, non-negative) plus
  provenance (the imaging parameters, the nuisance draw, the selection, the seed).
- `<stem>_Comparison.png` — the static side-by-side figure (below).
- `<stem>_Trajectory.h5` — the drawn reaction-diffusion trajectory (provenance; regenerable
  from `--seed`).

The `<model_window>` token (e.g. `2S_50FPS`) names the trained estimator; the `<clip_span>`
token (e.g. `20S`) is the rendered clip's own length — a 20 s clip from a 2 s-window estimate
is not the same as one from a 10 s window, so both appear in the name.

## How to read the comparison figure

A 2×4 grid. Column 0 is **EXPERIMENTAL** and column 1 is **SYNTH**, each showing a mid frame
over its max projection, so a frame and its projection sit in the same column. Column 2 holds
the shared **pixel-intensity histograms** — log-y over the full range (top) and linear-y
through ~p99.99 (bottom, essentially the whole range bar the top-0.01% hot-pixel sliver). Column 3 holds the syn/exp **ratio-per-quantile match plot**
(top) over a **quantile table and provenance text box** (bottom); the provenance lists the
inferred imaging MAP theta (out-of-prior parameters flagged) and the nuisance RDS draw, both in
absolute units.

- The **frame panels** show single-frame appearance — point-spread size, brightness, and
  per-frame noise. Read them for whether a synthetic frame looks like an experimental one.
- The **max projections** summarize the whole clip; they differ by design, because the
  synthetic motion is a fresh draw, not the experimental track.
- The **histograms** (ADU) show the pixel-intensity distributions; a close overlap is the
  evidence the imaging model matches. Both series are the stored, non-negative frames.
- The **ratio-per-quantile plot** is the direct "do they match?" read — the synthetic/experimental
  ratio across quantiles (min, p0.01, median, p90, p99, p99.9, p99.99, max), a flat line on 1.0
  being a perfect match. It exposes tail mismatches the log histogram can hide.

## Viewing interactively — the notebook

The notebook is a portable, pure viewer: it needs only `numpy`, `matplotlib`, and
`ipywidgets` — no project package, no `MACHINE_PROFILE`, and no ReaDDy — so it opens on any
machine, not just the one that rendered the clip. Step by step:

1. **Get a clip.** Render one with the script above on the data machine, then copy its
   `*_Synthetic_Video.npz` to the machine where you will view it. The `.npz` is
   self-contained (experimental + synthetic + provenance), so any location works.
2. **Launch Jupyter** in any environment that has `jupyter` and `ipywidgets` (a viewing
   environment, not the render environment). Run `jupyter lab` and open
   `notebooks/SRM_AND_SBI_DIMER_ALP_DETECTOR_Posterior_Predictive_Video.ipynb`.
3. **Run the cells top to bottom.** The first code cell imports the viewer; the second is
   the only one you normally edit.
4. **Point it at your clip.** In the second code cell, set `CLIP_PATH` to the absolute path
   of your `.npz`. `NORM_MODE` there defaults to `full` (a shared full-range `[min, max]` window
   over both panels); optionally set it to `autoscale` (each displayed frame to its own min/max,
   per-frame) or `percentile` (a fixed whole-clip `[min, p99.99]` window). Run the cell; it
   prints the frame count, frame rate, selection, and the display window.
5. **Scrub and zoom.** Run the scrubber cell. Drag `frame` to step through the recording;
   use `center x`, `center y`, and `zoom` to zoom the same region-of-interest into both the
   experimental and synthetic panels at once.
6. **Play.** Run the playback cell for a real-time, side-by-side player, using the same
   per-frame color scaling as the scrubber. Set `PLAY_ZOOM` (and the center) to play a cropped
   region and check whether experimental and synthetic coincide locally. If a long clip builds
   a heavy player, raise `PLAY_EVERY` (2, 5, …) to subsample frames; it stays real-time.

To change the display, edit `NORM_MODE` and re-run from the second code cell down. To view a
different clip, change `CLIP_PATH` and re-run.

## What it shows — and what it does not

- It reads **imaging appearance**, not motion. The synthetic track is a fresh
  reaction-diffusion draw, so the max projections and any track-level feature will differ.
- The stored synthetic is clipped to the non-negative range (a physical camera cannot record
  negative counts); the clip's negative-excursion count (`n_under`) is printed on every run.
  See `REFERENCE_EMCCD_NOISE_MODEL.md` for the corrected noise model and what the clip removes.
- If the MAP for a parameter lies outside the training prior, the run warns that it
  **extrapolates** there, so a poor match may reflect an out-of-prior estimate rather than
  the calibration itself.
- The estimator was calibrated on the fixed 8-bit rescale of the recordings; the clip is
  shown at full 16-bit, so the sub-8-bit detail displayed here lies just outside the
  calibrated domain.

## Reference

Real recordings: MET single-particle-tracking data, BioImage Archive accession S-BSST712. The
imaging model and the read-noise units discrepancy are documented under DLI Imaging in
`PROJECT_CONTEXT.md`; the Detector calibration workflow and its deferred items are in
`DETECTOR_WORKFLOW.md`.

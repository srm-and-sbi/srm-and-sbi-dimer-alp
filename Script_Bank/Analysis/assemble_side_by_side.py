"""Assemble the experimental | grid-render | fixed-render side-by-side (audit helper).

Loads two posterior-predictive renders from the Data_Bank Posit tier -- the render under
the retired grid brightness chain and a fixed-model render (--fixed-label picks its
--run-label token, default OU_FIX) -- for the condition and cell given by --kind/--cell,
and writes into <data_bank_root>/Posit/SRM_AND_SBI_DIMER_ALP_Brightness_Stationarity_Audit/
(analysis results are data and never live in the codebase; the Data_Bank location comes
from machine_profiles.toml via the MACHINE_PROFILE environment variable):

  - SIDE_BY_SIDE_<kind>_Cell_<cell>.mkv : lossless FFV1, three panels
    (experimental | grid render | fixed render), 25 fps (half speed of the 50 fps
    acquisition). ONE shared color window across the three panels, the viewer's
    "percentile" mode (shared [min, p99.99] over all three stacks), so identical
    intensities map to identical gray levels. Lossless per the no-lossy-compression
    rule for microscopy data.
  - SIDE_BY_SIDE_<kind>_Cell_<cell>.png : engine-convention figure -- frames at
    0 s / 10 s / 20 s (magma, ONE shared [min, p99.99] window over all three
    stacks, the viewer's "percentile" mode; a handful of extreme pixels must not
    set the display scale for 65 million), the complete pixel-intensity
    distribution from the global minimum to the global maximum (p0 to p100,
    log y, nothing clipped), and the quantile table with ratios against the
    experimental recording.

The engine's own two-panel comparison figures exist beside each render npz
(experimental vs grid in the canonical archive; experimental vs fixed in the
worktree shadow); this figure adds the three-way view with the same conventions.
The interactive viewer is notebooks/SRM_AND_SBI_DIMER_ALP_Brightness_Stationarity_Audit_Video.ipynb.
"""

import argparse
import os
import subprocess
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
FFMPEG = "/home/mars-fias/anaconda3/envs/READY_MARS/bin/ffmpeg"

PPV_DIR = "SRM_AND_SBI_DIMER_ALP_2S_50FPS_Posterior_Predictive_Video"
AUDIT_DIR = "SRM_AND_SBI_DIMER_ALP_Brightness_Stationarity_Audit"


OUT = None  # resolved in main() from the active machine profile


def data_bank_root() -> str:
    """Data_Bank root of the active machine profile (no package import needed)."""
    with open(os.path.join(REPO_ROOT, "machine_profiles.toml"), "rb") as handle:
        return tomllib.load(handle)[os.environ["MACHINE_PROFILE"]]["data_bank_root"]


def stem(kind: str, cell: int, label: str = "") -> str:
    token = f"_{label}" if label else ""
    return (f"SRM_AND_SBI_DIMER_ALP_2S_50FPS_MAP_Estimate_SGM_{kind}_Cell_{cell}"
            f"_20S{token}_Synthetic_Video.npz")

INK, INK_2, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"
LABELS = ("experimental recording", "grid-chain render (retired)", "fixed render (stationary OU)")
COLORS = ("#52514e", "#eb6834", "#2a78d6")
QUANTILES = (("min", 0.0), ("p0.01", 0.01), ("median", 50.0), ("p90", 90.0),
             ("p99", 99.0), ("p99.9", 99.9), ("p99.99", 99.99), ("max", 100.0))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", default="MET-FAB", choices=("MET-FAB", "MET-INLB"))
    parser.add_argument("--cell", type=int, default=0)
    parser.add_argument("--fixed-label", default="OU_FIX",
                        help="run label of the worktree render shown as the fixed arm.")
    args = parser.parse_args()
    out_stem = f"SIDE_BY_SIDE_{args.kind}_Cell_{args.cell}"
    if args.fixed_label != "OU_FIX":
        out_stem += f"_{args.fixed_label}"

    bank = data_bank_root()
    ppv_root = os.path.join(bank, "Posit", PPV_DIR)
    global OUT
    OUT = os.path.join(bank, "Posit", AUDIT_DIR)
    os.makedirs(OUT, exist_ok=True)
    old = np.load(os.path.join(ppv_root, stem(args.kind, args.cell)))
    new = np.load(os.path.join(ppv_root, stem(args.kind, args.cell, args.fixed_label)))
    experimental, grid, fixed = old["experimental"], old["synth"], new["synth"]
    assert np.array_equal(new["experimental"], experimental), \
        "the two renders reference different experimental stacks"
    stacks = (experimental, grid, fixed)
    n_frames, size, _ = experimental.shape

    # ONE shared window across ALL THREE stacks (the viewer's conventions).
    full_lo = float(min(s.min() for s in stacks))
    full_hi = float(max(s.max() for s in stacks))
    pctl_hi = float(max(np.percentile(s, 99.99) for s in stacks))

    # ---- lossless three-panel video, shared [min, p99.99] window ------------------
    lo, hi = full_lo, pctl_hi
    sep = np.ones((size, 4))
    video = np.concatenate([
        np.concatenate(
            [np.clip((s[t].astype(float) - lo) / (hi - lo), 0, 1) if i == 0 else
             np.concatenate([sep, np.clip((s[t].astype(float) - lo) / (hi - lo), 0, 1)], axis=1)
             for i, s in enumerate(stacks)], axis=1)
        for t in range(n_frames)]).reshape(n_frames, size, 3 * size + 8)
    frames8 = (video * 255).astype(np.uint8)
    mkv = os.path.join(OUT, out_stem + ".mkv")
    proc = subprocess.Popen(
        [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pixel_format", "gray",
         "-video_size", f"{frames8.shape[2]}x{frames8.shape[1]}", "-framerate", "25",
         "-i", "-", "-c:v", "ffv1", "-level", "3", mkv],
        stdin=subprocess.PIPE)
    proc.communicate(frames8.tobytes())
    assert proc.returncode == 0, "ffmpeg failed"

    # ---- engine-convention static figure -------------------------------------------
    fig = plt.figure(figsize=(13.6, 14.6), facecolor=SURFACE)
    grid_spec = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 0.72, 0.5],
                                 hspace=0.16, wspace=0.06)
    for row, t in enumerate((0, n_frames // 2, n_frames - 1)):
        for col, stack in enumerate(stacks):
            ax = fig.add_subplot(grid_spec[row, col])
            ax.imshow(stack[t], cmap="magma", origin="lower", interpolation="none",
                      vmin=full_lo, vmax=pctl_hi)
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(LABELS[col], fontsize=10.5, color=INK)
            if col == 0:
                ax.set_ylabel(f"t = {t * 0.02:.0f} s", fontsize=10, color=INK)

    # Complete distribution, p0 to p100: bins span the global [min, max].
    ax = fig.add_subplot(grid_spec[3, :])
    ax.set_facecolor(SURFACE)
    bins = np.linspace(full_lo, full_hi, 400)
    for stack, label, color in zip(stacks, LABELS, COLORS):
        counts, _ = np.histogram(stack.ravel(), bins=bins, density=True)
        ax.semilogy(0.5 * (bins[1:] + bins[:-1]), np.clip(counts, 1e-12, None),
                    color=color, linewidth=1.5, label=label)
    ax.set_xlim(full_lo, full_hi)
    ax.set_xlabel("pixel value (ADU) - full range, nothing clipped", fontsize=10, color=INK)
    ax.set_ylabel("density (log)", fontsize=10, color=INK)
    ax.tick_params(colors=INK_2, labelsize=9)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK)

    # Quantile table, min through max, with ratios against the experimental recording.
    ax = fig.add_subplot(grid_spec[4, :])
    ax.axis("off")
    lines = [f"{'quantile':>9s} {'experimental':>13s} {'grid render':>12s} {'fixed render':>13s} "
             f"{'grid/exp':>9s} {'fixed/exp':>10s}"]
    for name, q in QUANTILES:
        values = [float(s.min()) if q == 0.0 else float(s.max()) if q == 100.0
                  else float(np.percentile(s, q)) for s in stacks]
        lines.append(f"{name:>9s} {values[0]:13.0f} {values[1]:12.0f} {values[2]:13.0f} "
                     f"{values[1] / values[0]:9.2f} {values[2] / values[0]:10.2f}")
    ax.text(0.01, 0.95, "\n".join(lines), family="monospace", fontsize=9.2, color=INK,
            va="top", transform=ax.transAxes)
    fixed_note = "" if args.fixed_label == "OU_FIX" else f" [{args.fixed_label}]"
    fig.suptitle(f"{args.kind} cell {args.cell}: experimental recording against the two renders"
                 f"{fixed_note} (frames: one shared [min, p99.99] color window over all three "
                 "stacks; histogram: full range, p0 to p100)",
                 fontsize=11.5, color=INK, y=0.995)
    fig.savefig(os.path.join(OUT, out_stem + ".png"), dpi=160,
                facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print("shared full window:", (round(full_lo), round(full_hi)),
          "| video window [min, p99.99]:", (round(full_lo), round(pctl_hi)))
    print("wrote", mkv)


if __name__ == "__main__":
    main()

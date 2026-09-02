"""Extreme-tail structure of the posterior-predictive renders: emitters or camera noise?

Companion to `SRM_AND_SBI_DIMER_ALP_Brightness_Stationarity_Audit.py`. The stationarity
audit proves the latent brightness law; this script asks what the two brightness
mechanisms do to the OBSERVED extreme pixel tail, where their structures differ. The
retired grid caps every dye at the p95 quantile node of the lognormal, so any rendered
pixel beyond the corresponding ADU scale can only come from EMCCD gain fluctuations; the
stationary continuous process has genuine unbounded brightness excursions. Both can
produce competitive tail QUANTILES -- the discriminating evidence is the spatial and
temporal STRUCTURE of the extreme pixels: genuinely bright emitters are PSF-shaped
(multi-pixel, same frame) and persistent (the same site stays hot across frames), while
gain fluctuations are single-pixel, single-frame events.

This is a pure render consumer: `numpy`, `scipy`, `matplotlib` only -- no project package,
no simulation (the Data_Bank location is read from `machine_profiles.toml` via the
`MACHINE_PROFILE` environment variable). It reads persisted posterior-predictive clips from
the Data_Bank Posit tier (the renders under the retired grid, and the fixed-model renders
produced with `--run-label` tokens) and writes a report plus figures back to that tier --
analysis results are data and never live in the codebase. To regenerate any missing render,
run the engine (`Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Posterior_Predictive_Video.py`)
with the matching `--kind/--cell/--run-label` (and, for the sigma probe, `--set-imaging`).

Usage:
    MACHINE_PROFILE=<profile> python \
        Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Brightness_Tail_Structure.py [--ppv-root DIR]

Outputs:
    <data_bank_root>/Posit/SRM_AND_SBI_DIMER_ALP_Brightness_Stationarity_Audit/
        SRM_AND_SBI_DIMER_ALP_Brightness_Tail_Structure.md + T*.png + tail_structure.json
"""

from __future__ import annotations

import argparse
import json
import os
import tomllib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.stats import lognorm, norm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))


OUT_DIR = REPORT = None  # resolved in main() from the active machine profile


def data_bank_root() -> str:
    """Data_Bank root of the active machine profile (no package import needed)."""
    with open(os.path.join(REPO_ROOT, "machine_profiles.toml"), "rb") as handle:
        return tomllib.load(handle)[os.environ["MACHINE_PROFILE"]]["data_bank_root"]

# Frozen Nuisance_DLI photo-physics the renders were produced with (PROJECT_CONTEXT.md);
# used only to state the retired grid's brightness ceiling and the extreme-value scale.
MU_PC, SIGMA_PC = 250.73, 0.6744
BRIGHTNESS_QUANTILE = np.asarray([0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
SIGMA_PROBE = 0.53           # the --set-imaging sigma_pc probe rendered alongside
THRESHOLD_ADU = 10_000       # structure threshold: beyond p99.99 of every stack audited
QUANTILES = (50, 90, 99, 99.9, 99.99)

COLORS = {"experimental": "#52514e", "grid": "#eb6834", "fixed": "#2a78d6",
          "fixed_probe": "#7fb069"}
INK, INK_2, INK_MUTED, SURFACE = "#0b0b0b", "#52514e", "#8a8a85", "#fcfcfb"


def stem(kind: str, cell: int, label: str = "") -> str:
    token = f"_{label}" if label else ""
    return (f"SRM_AND_SBI_DIMER_ALP_2S_50FPS_MAP_Estimate_SGM_{kind}_Cell_{cell}"
            f"_20S{token}_Synthetic_Video.npz")


def structure(stack: np.ndarray, threshold: int) -> dict:
    """Spatial and temporal structure of the pixels above `threshold`."""
    mask = stack > threshold
    npix = int(mask.sum())
    if npix == 0:
        return dict(npix=0)
    blobs = big = 0
    for t in np.unique(np.argwhere(mask)[:, 0]):
        labels, n = ndimage.label(mask[t], structure=np.ones((3, 3)))
        blobs += n
        sizes = np.bincount(labels.ravel())[1:]
        big += int((sizes >= 3).sum())
    hot = mask.astype(np.uint8)
    neighbors = np.zeros_like(hot)
    h, w = hot.shape[1], hot.shape[2]
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == dy == 0:
                continue
            neighbors[:, max(0, dx):h + min(0, dx), max(0, dy):w + min(0, dy)] += \
                hot[:, max(0, -dx):h + min(0, -dx), max(0, -dy):w + min(0, -dy)]
    frac_neighbor = float((neighbors[mask] > 0).mean())
    background = float(np.median(stack))
    idx = np.argwhere(mask)
    ratios = []
    for t, x, y in idx[:: max(1, len(idx) // 400)]:
        patch = stack[t, max(0, x - 5):x + 6, max(0, y - 5):y + 6].astype(float)
        ratios.append(float(np.median(patch)) / background)
    sites: dict = {}
    for t, x, y in idx:
        sites.setdefault((int(x) // 4, int(y) // 4), set()).add(int(t))
    per_site = np.array([len(v) for v in sites.values()])
    return dict(npix=npix, blobs=blobs, blobs_3plus=big, frac_neighbor=frac_neighbor,
                local_over_background=float(np.median(ratios)), sites=len(sites),
                frames_per_site_median=float(np.median(per_site)),
                frames_per_site_max=int(per_site.max()))


def quantile_row(stack: np.ndarray) -> dict:
    row = {f"p{q:g}": float(np.percentile(stack, q)) for q in QUANTILES}
    row["min"], row["max"] = float(stack.min()), float(stack.max())
    return row


def survival_figure(name: str, title: str, stacks: dict) -> None:
    fig, ax = plt.subplots(figsize=(8.6, 4.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    hi = max(float(s.max()) for s in stacks.values())
    grid_adu = np.linspace(2000, hi, 400)
    for label, stack in stacks.items():
        values = np.sort(stack.ravel())
        surv = 1.0 - np.searchsorted(values, grid_adu, side="right") / values.size
        key = ("experimental" if label.startswith("experimental") else
               "grid" if label.startswith("grid") else
               "fixed_probe" if "sigma" in label else "fixed")
        ax.semilogy(grid_adu, np.clip(surv, 1e-9, None), color=COLORS[key],
                    linewidth=1.7, label=label)
    ax.set_xlabel("pixel value (ADU)", fontsize=10, color=INK)
    ax.set_ylabel("survival fraction P(pixel > x)", fontsize=10, color=INK)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(axis="y", color=INK_MUTED, alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.set_title(title, fontsize=11.5, color=INK, pad=12, loc="left")
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=200, facecolor=SURFACE)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ppv-root", default=None,
                        help="directory holding the posterior-predictive render npz clips "
                             "(default: the active profile's Data_Bank Posit PPV directory).")
    args = parser.parse_args()
    bank = data_bank_root()
    ppv_root = args.ppv_root or os.path.join(
        bank, "Posit", "SRM_AND_SBI_DIMER_ALP_2S_50FPS_Posterior_Predictive_Video")
    global OUT_DIR, REPORT
    OUT_DIR = os.path.join(bank, "Posit", "SRM_AND_SBI_DIMER_ALP_Brightness_Stationarity_Audit")
    REPORT = os.path.join(OUT_DIR, "SRM_AND_SBI_DIMER_ALP_Brightness_Tail_Structure.md")
    os.makedirs(OUT_DIR, exist_ok=True)

    def load(root, kind, cell, label=""):
        return np.load(os.path.join(root, stem(kind, cell, label)), allow_pickle=False)

    results: dict = {}

    # ---- the retired grid's brightness ceiling ---------------------------------------
    nodes = np.round(lognorm.ppf(BRIGHTNESS_QUANTILE, s=SIGMA_PC, loc=0, scale=MU_PC))
    results["grid_nodes_photons"] = nodes.tolist()
    # Extreme-value scale of the unbounded law: expected max of n_eff independent draws.
    z = norm.ppf(1 - 1 / 15_000.0)  # ~15k effective draws in a 20 s render
    results["evt_photons"] = {f"{s:g}": float(MU_PC * np.exp(s * z))
                              for s in (SIGMA_PC, SIGMA_PROBE)}

    # ---- per-condition tables ---------------------------------------------------------
    conditions = {
        "MET-FAB": [("fixed (OU_FIX)", "OU_FIX"),
                    (f"fixed, sigma_pc={SIGMA_PROBE} (OU_FIX_SIGMA_053)", "OU_FIX_SIGMA_053")],
        "MET-INLB": [("fixed (OU_FIX)", "OU_FIX")],
    }
    for kind, fixed_arms in conditions.items():
        archived = load(ppv_root, kind, 0)
        stacks = {"experimental recording": archived["experimental"],
                  "grid render (retired)": archived["synth"]}
        for label, token in fixed_arms:
            clip = load(ppv_root, kind, 0, token)
            assert np.array_equal(clip["experimental"], archived["experimental"]), \
                f"{kind}: clips reference different recordings"
            stacks[label] = clip["synth"]
        exp = stacks["experimental recording"]
        block = {"quantiles": {}, "structure": {}, "exceedance": {}}
        exp_max = int(exp.max())
        for label, stack in stacks.items():
            block["quantiles"][label] = quantile_row(stack)
            block["structure"][label] = structure(stack, THRESHOLD_ADU)
            block["exceedance"][label] = {
                f">{THRESHOLD_ADU}": int((stack > THRESHOLD_ADU).sum()),
                ">exp_max": int((stack > exp_max).sum())}
        block["exp_max"] = exp_max
        results[kind] = block
        survival_figure(
            f"T{1 if kind == 'MET-FAB' else 2}_survival_{kind}.png",
            f"{kind} cell 0: pixel-value survival curves (20 s recording, all pixels)",
            stacks)

    with open(os.path.join(OUT_DIR, "tail_structure.json"), "w") as handle:
        json.dump(results, handle, indent=1)

    # ---- report -----------------------------------------------------------------------
    lines: list[str] = []
    add = lines.append
    add("# Extreme-tail structure of the posterior-predictive renders")
    add("")
    add("**Conclusion.** The experimental extreme pixel tail is emitter-driven: pixels above "
        f"{THRESHOLD_ADU:,} ADU form PSF-shaped multi-pixel spots that stay hot at the same "
        "site for many frames. The retired grid cannot produce that structure -- its "
        "brightness is capped at the p95 quantile node "
        f"({nodes[-1]:.0f} photons per dye, {2 * nodes[-1]:.0f} for a two-label dimer), so its "
        "pixels beyond that scale are single-pixel, single-frame EMCCD gain fluctuations. "
        "The stationary continuous process reproduces the experimental structure on every "
        "measure (count, spatial extent, site persistence). The retired grid's competitive "
        "tail QUANTILES up to p99.99 are therefore right numbers from the wrong mechanism, "
        "and single-render maxima are one-noise-spike statistics that carry no evidential "
        "weight. The continuous process's residual overshoot beyond the experimental maximum "
        "is a calibration statement, not a structural one: it scales with sigma_pc "
        f"(the sigma_pc = {SIGMA_PROBE} probe removes it entirely, at the price of starving "
        "the observed deep tail), and the calibrated value is a matter for the imaging "
        "recalibration under the new model, not for the brightness mechanism.")
    add("")
    add("Regenerate with `Script_Bank/Analysis/" + os.path.basename(__file__) + "`. Inputs: "
        "the archived grid-era renders (canonical Data_Bank, read-only) and the fixed-model "
        "renders produced with `--run-label OU_FIX` (both conditions, cell 0) and "
        f"`--run-label OU_FIX_SIGMA_053 --set-imaging sigma_pc={SIGMA_PROBE}` (MET-FAB cell "
        "0). Figures and the summary JSON sit beside this report in the Data_Bank Posit tier.")
    add("")
    add("## 1. The retired grid's brightness ceiling")
    add("")
    add("Grid nodes at the frozen photo-physics (mu_pc = "
        f"{MU_PC}, sigma_pc = {SIGMA_PC}): " + ", ".join(f"{v:.0f}" for v in nodes)
        + " photons. The top node is the p95 quantile: no dye can ever exceed "
        f"{nodes[-1]:.0f} photons. The unbounded lognormal expects a "
        f"~{norm.ppf(1 - 1 / 15_000.0):.1f}-sigma excursion over a 20 s render "
        f"(~15,000 effective independent draws): "
        + ", ".join(f"{v:,.0f} photons at sigma_pc = {s}"
                    for s, v in results["evt_photons"].items())
        + ". A longer tail than the grid's is therefore the intended behavior of the "
        "continuous process, and its amplitude is set by sigma_pc.")
    add("")
    for kind in conditions:
        block = results[kind]
        add(f"## {2 if kind == 'MET-FAB' else 3}. {kind} cell 0")
        add("")
        labels = list(block["quantiles"])
        add("| quantile (ADU) | " + " | ".join(labels) + " |")
        add("|---|" + "---|" * len(labels))
        for key in ["min"] + [f"p{q:g}" for q in QUANTILES] + ["max"]:
            add(f"| {key} | " + " | ".join(f"{block['quantiles'][l][key]:,.0f}"
                                           for l in labels) + " |")
        add("")
        add(f"Structure of the pixels above {THRESHOLD_ADU:,} ADU:")
        add("")
        add("| measure | " + " | ".join(labels) + " |")
        add("|---|" + "---|" * len(labels))
        rows = (("hot pixels", "npix", "{:,d}"),
                ("per-frame 8-connected components", "blobs", "{:,d}"),
                ("PSF-shaped components (3+ px)", "blobs_3plus", "{:,d}"),
                ("fraction with a hot 8-neighbor, same frame", "frac_neighbor", "{:.2f}"),
                ("median local ground / stack median", "local_over_background", "{:.2f}"),
                ("distinct 4x4-px sites", "sites", "{:,d}"),
                ("frames per site, median", "frames_per_site_median", "{:.0f}"),
                ("frames per site, max", "frames_per_site_max", "{:,d}"))
        for title, key, fmt in rows:
            cells = []
            for l in labels:
                s = block["structure"][l]
                cells.append(fmt.format(s[key]) if s["npix"] else "-")
            add(f"| {title} | " + " | ".join(cells) + " |")
        add("")
        add(f"Exceedance: pixels above the experimental maximum ({block['exp_max']:,} ADU): "
            + "; ".join(f"{l} {block['exceedance'][l]['>exp_max']:,}"
                        for l in labels if not l.startswith("experimental")) + ".")
        add("")
        add(f"![{kind} survival curves](T{1 if kind == 'MET-FAB' else 2}_survival_{kind}.png)")
        add("")
    add("## 4. Reading")
    add("")
    add("On MET-FAB the continuous process matches the experimental deep-tail mass "
        "(pixels above 10,000 ADU) and its structure almost exactly, while the grid's "
        "few extreme pixels are isolated one-frame events with zero PSF-shaped components. "
        "On MET-INLB BOTH mechanisms fall short of the experimental tail from p99 outward: "
        "that gap moves with the calibration (the InlB recording is systematically "
        "brighter than the frozen imaging vector renders), consistent with "
        "condition-specific per-dye brightness, and is not evidence in the mechanism "
        "comparison. The sigma_pc probe brackets the recalibration: at "
        f"{SIGMA_PC} the extreme overshoots and the deep tail matches; at {SIGMA_PROBE} "
        "the extreme is contained and the deep tail is starved.")
    add("")
    with open(REPORT, "w") as handle:
        handle.write("\n".join(lines))
    print("wrote", REPORT)
    print(json.dumps({k: v for k, v in results.items() if k.startswith("MET")},
                     indent=1)[:600], "...")


if __name__ == "__main__":
    main()

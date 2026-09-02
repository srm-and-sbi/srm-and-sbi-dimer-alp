"""Brightness stationarity audit: retired state grid versus the continuous stationary process.

Two-arm audit of the emitter-brightness photo-physics. Arm GRID is the retired seven-state
quantile chain exactly as implemented (symmetric generator, interval-weight
initialization), embedded VERBATIM in this script as its permanent reference arm -- the
package itself carries only the replacement; arm OU is the stationary continuous per-dye
process (`generate_brightness_photons`). The verification protocol and its acceptance criteria are
prespecified in the companion report this script writes; the design decision it validates
is the 2026-09-01 amendment in `MET_NEXT_MODEL_DESIGN_PLAN.md` (workspace root).

Levels implemented here:
    Level 1 (latent): per-frame law, occupancy relaxation, autocorrelation,
                      survivor conditioning, dimer sum.
    Level 2 (render): static-scene renders through the identical downstream path;
                      per-frame aperture photometry; first-window versus last-window
                      invariance. An i.i.d. reference arm (rho = 0) doubles as the
                      push-forward reference, since its per-frame marginal equals the
                      OU arm's by construction.

The test suite must FLAG the grid arm exactly where the design plan says it is broken
(relaxation to uniform occupancy within ~10 frames) and PASS the fix; a suite that cannot
detect the known defect proves nothing about the repair.

Usage (from the repo root):
    MACHINE_PROFILE=<profile> PYTHONPATH=$PWD python \
        Script_Bank/Analysis/SRM_AND_SBI_DIMER_ALP_Brightness_Stationarity_Audit.py

Outputs (analysis results are data and live in the Data_Bank, never the codebase):
    <data_bank_root>/Posit/SRM_AND_SBI_DIMER_ALP_Brightness_Stationarity_Audit/
        SRM_AND_SBI_DIMER_ALP_Brightness_Stationarity_Audit.md  (report)
        F*.png + audit_summary.json                             (figures + summary)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import expm
from scipy.signal import fftconvolve
from scipy.stats import kstest, linregress, lognorm, norm

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(HERE, "..", ".."))
sys.path.insert(0, REPO_ROOT)

import srm_and_sbi_dimer_alp.simulation_dli_support as dli  # noqa: E402

assert os.path.abspath(dli.__file__).startswith(REPO_ROOT), (
    "Imported srm_and_sbi_dimer_alp from outside this repo checkout: " + dli.__file__
    + " -- run with PYTHONPATH set to the repo root."
)

# Analysis results are data: they go to the Data_Bank Posit tier, never the codebase.
OUT_DIR = os.path.join(str(dli.PARAMETERS.machine.data_bank_root), "Posit",
                       "SRM_AND_SBI_DIMER_ALP_Brightness_Stationarity_Audit")
REPORT = os.path.join(OUT_DIR, "SRM_AND_SBI_DIMER_ALP_Brightness_Stationarity_Audit.md")

# Audit operating point: the frozen Nuisance_DLI SGM photophysics quoted in
# PROJECT_CONTEXT.md (mu_pc, sigma_pc, prob_photo_bleach) and the calibrated
# lambda_rate operating point 10**0.5 from the parameter table NOTE. Camera values
# for Level 2 are the SCOPE anchors documented in DETECTOR_WORKFLOW.md /
# detector_parameterization.py. Fixed audit constants, not pipeline inputs.
MU_PC, SIGMA_PC = 250.73, 0.6744
LAMBDA_RATE = 10.0 ** 0.5
PROB_BLEACH, NUMB_BLEACH = 0.107, 100
DELTA_FRAME = 0.02
CAMERA = dict(gamma=41.95, kappa_o=28.7, kappa_b=175.0, kappa_s=10.5, kappa_q=0.90)
PSF_MU_R, PSF_SIGMA_R = 1.60, 0.15   # Fab-condition detector MAPs (DETECTOR_WORKFLOW.md sec. 8)

BRIGHTNESS_QUANTILE = np.asarray([0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95])


# ----------------------------------------------------------------------------------------------
# Retired grid chain, embedded VERBATIM as the reference arm. This is the seven-state
# quantile brightness machine exactly as it shipped in the package before the stationarity
# fix (docstrings trimmed, code byte-identical); it lives here so the audit's positive
# control -- the suite must flag this mechanism's defect -- keeps running after the package
# dropped it.
# ----------------------------------------------------------------------------------------------

def compute_brightness(brightness_quantile, scale, shape, loc=0):
    """Brightness values at the given lognormal quantiles, rounded to integer photons."""
    return np.round(lognorm.ppf(q=brightness_quantile, s=shape, loc=loc, scale=scale), 0)


def compute_brightness_probability(brightness_quantile):
    """Initial state probabilities: interval widths around each quantile node; state 0 gets 0."""
    q = brightness_quantile
    prob = np.empty_like(q, dtype=float)
    prob[0] = 0
    prob[1] = q[1] + (q[2] - q[1]) / 2
    for i in range(2, len(q) - 1):
        prob[i] = ((q[i] - q[i - 1]) + (q[i + 1] - q[i])) / 2
    prob[-1] = (1 - q[-1]) + (q[-1] - q[-2]) / 2
    return prob / np.sum(prob)


def initialize_emitter_states(n_emitters, brightness_quantile, scale, shape, loc=0, seed=None):
    """Sample each emitter's initial state from the interval-weight distribution."""
    rng = np.random.default_rng(seed)
    brightness = compute_brightness(brightness_quantile, scale, shape, loc)
    brightness_probability = compute_brightness_probability(brightness_quantile)
    initial_states = rng.choice(a=len(brightness), size=n_emitters, p=brightness_probability)
    return initial_states, brightness, brightness_probability


def _propagate_chain(states, P, rng):
    """Fill state trajectories in place from the DTMC transition matrix."""
    n_frames, n_emitters = states.shape
    for i in range(1, n_frames):
        prev_states = states[i - 1, :]
        cum_probs = np.cumsum(P[prev_states, :], axis=1)
        u = rng.uniform(0, 1, size=n_emitters)[:, None]
        states[i, :] = (u < cum_probs).argmax(axis=1)
    return states


def generate_state_trajectories(nframes, nemitters, P, brightness_quantile, scale, shape,
                                loc=0, seed=None):
    """Initial states from the interval weights, then DTMC evolution."""
    rng = np.random.default_rng(seed)
    states = np.full(shape=(nframes, nemitters), fill_value=-1, dtype=int)
    initial_states, _, _ = initialize_emitter_states(
        nemitters, brightness_quantile, scale, shape, loc, seed=seed,
    )
    states[0, :] = initial_states
    _propagate_chain(states, P, rng)
    return states


def compute_matrices(mu_pc, sigma_pc, brightness_quantile, prob_photo_bleach,
                     numb_photo_bleach, delta_frame, lambda_rate, kappa_penalty=1.0,
                     verbose=False):
    """CTMC generator Q (symmetric brightness-distance rates + absorbing bleach) and P = expm."""
    brightness = np.round(
        lognorm.ppf(q=brightness_quantile, s=sigma_pc, loc=0, scale=mu_pc), 0,
    )
    sigma_bright = lognorm.std(s=sigma_pc, loc=0, scale=mu_pc)
    numb_states = len(brightness)
    Q = np.zeros((numb_states, numb_states))
    prob_1 = 1 - np.power((1 - prob_photo_bleach), 1 / numb_photo_bleach)
    epsilon_rate = -np.log(1 - prob_1) / delta_frame
    for i in range(1, numb_states):
        Q[i, 0] = epsilon_rate
        j_indices = range(1, numb_states)
        distances = np.abs(brightness[i] - brightness[list(j_indices)])
        Q[i, list(j_indices)] = lambda_rate * np.exp(-kappa_penalty * distances / sigma_bright)
    np.fill_diagonal(Q, 0)
    diag_indices = np.diag_indices(numb_states)
    Q[diag_indices] = -np.sum(Q, axis=1)
    P = expm(Q * delta_frame)
    return Q, P


INTERVAL_WEIGHTS = compute_brightness_probability(BRIGHTNESS_QUANTILE)[1:]  # live states

# Acceptance criteria (prespecified; see the protocol section of the report).
TOL_KS = 0.005          # per-frame sup-CDF distance, OU arm (KS sampling floor ~0.003 at n=2e5)
TOL_MEAN_DRIFT = 0.01   # |fitted ln-mean drift| across the audited frames, OU arm
TOL_SD = 0.01           # |per-frame ln-sd - sigma_pc|, OU arm
TOL_TV_OU_EQUIV = 0.01  # grid arm must EXCEED this against interval weights (defect visible)
TOL_ACF = 0.10          # relative error of fitted correlation-decay rate, OU arm

N_LAW, T_LAW = 200_000, 150
N_ACF, T_ACF = 20_000, 500
N_DIMER = 200_000
T_RENDER, N_RENDER = 1_000, 64

C_OU, C_GRID, C_IID, INK, INK_2, INK_MUTED, SURFACE = (
    "#2a78d6", "#eb6834", "#8a8a85", "#0b0b0b", "#52514e", "#8a8a85", "#fcfcfb")


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(INK_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=INK_2, labelsize=9, width=0.8, length=4)
    ax.grid(axis="y", color=INK_MUTED, alpha=0.22, linewidth=0.7)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, name), dpi=200, facecolor=SURFACE)
    plt.close(fig)


# ----------------------------------------------------------------------------------------------
# Arms
# ----------------------------------------------------------------------------------------------

def grid_photons(nframes: int, nemitters: int, prob_bleach: float,
                 seed: int | None = None) -> tuple[np.ndarray, np.ndarray]:
    """The retired chain exactly as implemented: (photons, states)."""
    _Q, P = compute_matrices(
        mu_pc=MU_PC, sigma_pc=SIGMA_PC, brightness_quantile=BRIGHTNESS_QUANTILE,
        prob_photo_bleach=prob_bleach, numb_photo_bleach=NUMB_BLEACH,
        delta_frame=DELTA_FRAME, lambda_rate=LAMBDA_RATE)
    states = generate_state_trajectories(
        nframes=nframes, nemitters=nemitters, P=P,
        brightness_quantile=BRIGHTNESS_QUANTILE, scale=MU_PC, shape=SIGMA_PC, loc=0, seed=seed)
    brightness = compute_brightness(BRIGHTNESS_QUANTILE, scale=MU_PC, shape=SIGMA_PC, loc=0)
    return brightness[states], states


def ou_photons(nframes: int, nemitters: int, prob_bleach: float,
               lambda_rate: float = LAMBDA_RATE, seed: int | None = None) -> np.ndarray:
    return dli.generate_brightness_photons(
        nframes=nframes, nemitters=nemitters, mu_pc=MU_PC, sigma_pc=SIGMA_PC,
        lambda_rate=lambda_rate, prob_photo_bleach=prob_bleach,
        numb_photo_bleach=NUMB_BLEACH, delta_frame=DELTA_FRAME, seed=seed)


# ----------------------------------------------------------------------------------------------
# Level 1
# ----------------------------------------------------------------------------------------------

def per_frame_law() -> dict:
    """Per-frame ln mean/sd, KS to the intended law, and grid occupancy relaxation."""
    out: dict = {}
    photons_g, states_g = grid_photons(T_LAW, N_LAW, PROB_BLEACH, seed=11)
    photons_o = ou_photons(T_LAW, N_LAW, PROB_BLEACH, seed=12)

    for arm, photons in (("grid", photons_g), ("ou", photons_o)):
        active = photons > 0
        ln = np.where(active, np.log(np.where(active, photons, 1.0)), np.nan)
        mean_t = np.nanmean(ln, axis=1) - np.log(MU_PC)
        sd_t = np.nanstd(ln, axis=1)
        ks_t = np.array([
            kstest(ln[t][active[t]], "norm", args=(np.log(MU_PC), SIGMA_PC)).statistic
            for t in range(T_LAW)])
        slope = linregress(np.arange(T_LAW), mean_t)
        out[arm] = dict(mean_t=mean_t.tolist(), sd_t=sd_t.tolist(), ks_t=ks_t.tolist(),
                        drift=float(slope.slope * T_LAW), drift_p=float(slope.pvalue),
                        max_ks=float(ks_t.max()),
                        max_sd_err=float(np.abs(sd_t - SIGMA_PC).max()))

    # Grid occupancy relaxation among unbleached emitters.
    tv_intended, tv_uniform = [], []
    uniform = np.full(7, 1.0 / 7.0)
    for t in range(T_LAW):
        live = states_g[t][states_g[t] > 0]
        occ = np.bincount(live, minlength=8)[1:] / live.size
        tv_intended.append(0.5 * np.abs(occ - INTERVAL_WEIGHTS).sum())
        tv_uniform.append(0.5 * np.abs(occ - uniform).sum())
    out["grid_tv_intended"] = [float(v) for v in tv_intended]
    out["grid_tv_uniform"] = [float(v) for v in tv_uniform]
    out["grid_max_tv_intended"] = float(np.max(tv_intended))
    return out


def acf() -> dict:
    """ln-brightness autocorrelation, bleaching off; fitted decay rate per arm."""
    photons_g, _ = grid_photons(T_ACF, N_ACF, 0.0, seed=21)
    photons_o = ou_photons(T_ACF, N_ACF, 0.0, seed=22)
    lags = np.arange(0, 101)
    out: dict = {"lags_s": (lags * DELTA_FRAME).tolist()}
    for arm, photons in (("grid", photons_g), ("ou", photons_o)):
        # Center by the KNOWN population mean (the target law is fully specified);
        # per-trajectory sample-mean centering biases the ACF negative at long lags
        # by about -2*tau/T, which is exactly what a naive estimator shows here.
        ln = np.log(photons) - np.log(MU_PC)
        denominator = (ln * ln).mean()
        acf_values = np.array([
            (ln[:T_ACF - lag] * ln[lag:]).mean() / denominator if lag else 1.0
            for lag in lags])
        # Fit the decay only where the ACF is comfortably positive.
        mask = (lags >= 1) & (acf_values > 0.05)
        rate = -np.polyfit(lags[mask] * DELTA_FRAME, np.log(acf_values[mask]), 1)[0]
        out[arm] = dict(acf=acf_values.tolist(), lambda_eff=float(rate))
    out["ou_lambda_rel_err"] = float(abs(out["ou"]["lambda_eff"] - LAMBDA_RATE) / LAMBDA_RATE)
    return out


def survivor() -> dict:
    """OU arm with strong bleaching: the law among still-active dyes is unchanged."""
    photons = ou_photons(150, N_LAW, 0.5, seed=31)
    checks = {}
    for t in (1, 50, 100, 149):
        values = photons[t][photons[t] > 0]
        checks[str(t)] = dict(
            n_active=int(values.size),
            ks=float(kstest(np.log(values), "norm", args=(np.log(MU_PC), SIGMA_PC)).statistic))
    return checks


def dimer() -> dict:
    """Sum of two independent OU dyes versus the exact two-source convolution."""
    a = ou_photons(101, N_DIMER, 0.0, seed=41)
    b = ou_photons(101, N_DIMER, 0.0, seed=42)
    total = a + b
    lin = np.linspace(0.0, 14.0 * MU_PC, 40_001)
    step = lin[1] - lin[0]
    one = np.zeros_like(lin)
    nz = lin > 0
    one[nz] = (np.exp(-((np.log(lin[nz]) - np.log(MU_PC)) ** 2) / (2 * SIGMA_PC ** 2))
               / (lin[nz] * SIGMA_PC * np.sqrt(2 * np.pi)))
    density = fftconvolve(one, one, mode="full")[:lin.size] * step
    cdf_grid = np.concatenate([[0.0], np.cumsum(0.5 * (density[1:] + density[:-1]) * step)])
    cdf_grid /= cdf_grid[-1]
    cdf = lambda x: np.interp(x, lin, cdf_grid)
    return {str(t): float(kstest(total[t], cdf).statistic) for t in (0, 100)}


# ----------------------------------------------------------------------------------------------
# Level 2: static-scene renders through the identical downstream path
# ----------------------------------------------------------------------------------------------

def render_arm(photons: np.ndarray, positions_px: np.ndarray, seed: int) -> np.ndarray:
    """PSF + optical background + EMCCD applied to a per-frame photon source."""
    nframes, nemitters = photons.shape
    stem = dli.PARAMETERS.simulation.stem
    root = stem.root_size_px
    widths = dli.sample_psf_width(
        nemitters, dli.PARAMETERS.simulation.dli.sqrt_2sigma_dist_label,
        keyword_args={"mu_r": PSF_MU_R, "sigma_r": PSF_SIGMA_R}, seed=7)
    psf = dli.Gaussian(widths)
    detector = dli.EMCCD(quantum_efficiency=CAMERA["kappa_q"], em_gain=CAMERA["gamma"],
                         electrons_per_adu=1.0, read_noise_adu=CAMERA["kappa_s"],
                         bias_adu=CAMERA["kappa_b"])
    background = np.full((root, root), CAMERA["kappa_o"], dtype=float)
    bounds = np.linspace(0, root, root + 1)
    tracks = np.repeat(positions_px[None, :, :], nframes, axis=0)      # (frames, emitters, 2)
    tracks = np.transpose(tracks, (0, 2, 1)).astype(np.float64)        # (frames, 2, emitters)
    intensity = dli.compute_intensity(
        tracks=tracks, emitter_photons=photons, background_photons=background,
        xbounds=bounds, ybounds=bounds, PSF=psf)
    return dli.generate_frames(intensity, detector, seed=seed)


def aperture_net(frames: np.ndarray, positions_px: np.ndarray, half: int = 3) -> np.ndarray:
    """(nframes, nemitters) net ADU in a (2*half+1)^2 window around each emitter."""
    expected_bg = CAMERA["kappa_o"] * CAMERA["kappa_q"] * CAMERA["gamma"] + CAMERA["kappa_b"]
    window = (2 * half + 1) ** 2
    n_emitters = positions_px.shape[0]
    nframes = frames.shape[2]
    net = np.empty((nframes, n_emitters))
    for e, (x, y) in enumerate(positions_px.astype(int)):
        # The renderer's image axes are (n_pix_x, n_pix_y, n_frames): the FIRST
        # image axis follows the track x coordinate (see add_pixel_counts).
        patch = frames[x - half:x + half + 1, y - half:y + half + 1, :]
        net[:, e] = patch.sum(axis=(0, 1)) - window * expected_bg
    return net


def render_level() -> dict:
    """Time-invariance of aperture photometry: grid vs OU vs the i.i.d. reference."""
    root = dli.PARAMETERS.simulation.stem.root_size_px
    grid_axis = np.linspace(root * 0.15, root * 0.85, 8)
    positions = np.array([(x, y) for x in grid_axis for y in grid_axis])  # (64, 2) px

    photons_g, _ = grid_photons(T_RENDER, N_RENDER, PROB_BLEACH, seed=51)
    photons_o = ou_photons(T_RENDER, N_RENDER, PROB_BLEACH, seed=52)
    photons_i = ou_photons(T_RENDER, N_RENDER, PROB_BLEACH, lambda_rate=1e6, seed=53)

    out: dict = {}
    window1_medians: dict = {}
    for arm, photons, seed in (("grid", photons_g, 61), ("ou", photons_o, 62),
                               ("iid", photons_i, 63)):
        frames = render_arm(photons, positions, seed=seed)
        net = aperture_net(frames, positions)
        active = photons > 0
        mean_t = np.array([net[t][active[t]].mean() if active[t].any() else np.nan
                           for t in range(T_RENDER)])
        # Time invariance must be judged on the SAME emitter set: bleaching makes the
        # last window a random SUBSET of emitters, and pooled two-sample tests across
        # different subsets measure per-emitter heterogeneity (PSF width, sub-pixel
        # capture), not time dependence -- the i.i.d. reference proves it. Restrict
        # both windows to survivors of the whole recording (bleaching is
        # brightness-independent, so this conditioning is benign), and gate on the
        # per-emitter paired median shift.
        survivors = active[-1]
        first = net[:100][:, survivors]
        last = net[900:][:, survivors]
        per_emitter_shift = np.median(last, axis=0) / np.median(first, axis=0) - 1.0
        # Arm-paired marginal check over the FULL recording: the OU arm has only ~6
        # effectively independent frames per 2 s window (tau ~ 16 frames), so
        # window-level per-emitter medians carry ~+/-6% estimator spread; the
        # full-recording median (~60 effective draws) brings that to ~2%.
        window1_medians[arm] = np.array([
            np.median(net[active[:, e], e]) for e in range(net.shape[1])])
        out[arm] = dict(mean_t=mean_t.tolist(),
                        n_survivors=int(survivors.sum()),
                        first_last_ks=float(kstest(first.ravel(), last.ravel()).statistic),
                        median_rel_shift=float(np.median(per_emitter_shift)))
    # Push-forward reference: the i.i.d. arm has the same per-frame marginal as the OU
    # arm by construction, and PSF widths are seed-matched across arms, so the
    # per-emitter window-1 medians are exactly paired between arms.
    paired = window1_medians["ou"] / window1_medians["iid"] - 1.0
    out["ou_vs_iid_paired_median_rel_diff"] = float(np.median(paired))
    return out


# ----------------------------------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------------------------------

def figures(law: dict, acf_out: dict, render_out: dict) -> None:
    frames_axis = np.arange(T_LAW)
    se_mean = SIGMA_PC / np.sqrt(N_LAW)

    fig, axes = plt.subplots(2, 1, figsize=(8.6, 6.4), sharex=True)
    fig.patch.set_facecolor(SURFACE)
    for ax, key, intended, band in (
            (axes[0], "mean_t", 0.0, 2 * se_mean),
            (axes[1], "sd_t", SIGMA_PC, 2 * SIGMA_PC / np.sqrt(2 * N_LAW))):
        style_axes(ax)
        ax.axhline(intended, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.fill_between(frames_axis, intended - band, intended + band, color=INK_MUTED, alpha=0.25)
        ax.plot(frames_axis, law["grid"][key], color=C_GRID, linewidth=1.8, label="retired grid chain")
        ax.plot(frames_axis, law["ou"][key], color=C_OU, linewidth=1.8, label="continuous stationary process")
    axes[0].set_ylabel("mean of ln photons - ln mu_pc", fontsize=10, color=INK)
    axes[1].set_ylabel("sd of ln photons", fontsize=10, color=INK)
    axes[1].set_xlabel("frame (20 ms each)", fontsize=10, color=INK)
    axes[0].set_title("Per-frame brightness law: the grid drifts, the continuous process holds",
                      fontsize=11.5, color=INK, pad=12, loc="left")
    legend = axes[0].legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK)
    save(fig, "F1_per_frame_law.png")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.plot(frames_axis, law["grid_tv_intended"], color=C_GRID, linewidth=1.8,
            label="TV to the documented interval weights")
    ax.plot(frames_axis, law["grid_tv_uniform"], color=INK_2, linewidth=1.8,
            label="TV to uniform occupancy")
    for level in (0.05, 0.02, 0.01):
        ax.axhline(level, color=INK_MUTED, linewidth=0.9, linestyle=(0, (2, 2)))
    ax.set_xlabel("frame (20 ms each)", fontsize=10, color=INK)
    ax.set_ylabel("total variation distance", fontsize=10, color=INK)
    ax.set_title("Retired grid chain: unbleached occupancy relaxes to uniform",
                 fontsize=11.5, color=INK, pad=12, loc="left")
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK)
    save(fig, "F2_grid_occupancy_relaxation.png")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    ax.semilogy(frames_axis, law["grid"]["ks_t"], color=C_GRID, linewidth=1.8,
                label="retired grid chain (discrete atoms + drift)")
    ax.semilogy(frames_axis, law["ou"]["ks_t"], color=C_OU, linewidth=1.8,
                label="continuous stationary process")
    ax.axhline(TOL_KS, color=INK_2, linewidth=1.0, linestyle=(0, (4, 3)))
    ax.text(frames_axis[-1], TOL_KS * 1.15, "acceptance 0.005", ha="right", va="bottom",
            fontsize=8.6, color=INK_2)
    ax.set_xlabel("frame (20 ms each)", fontsize=10, color=INK)
    ax.set_ylabel("KS distance to LogNormal(mu_pc, sigma_pc)", fontsize=10, color=INK)
    ax.set_title("Per-frame distance to the documented law", fontsize=11.5, color=INK,
                 pad=12, loc="left")
    legend = ax.legend(frameon=False, fontsize=9, loc="center right")
    for text in legend.get_texts():
        text.set_color(INK)
    save(fig, "F3_per_frame_ks.png")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    lags_s = np.asarray(acf_out["lags_s"])
    ax.semilogy(lags_s, np.clip(acf_out["grid"]["acf"], 1e-4, None), color=C_GRID,
                linewidth=1.8, label=f"retired grid chain: fitted decay {acf_out['grid']['lambda_eff']:.2f}/s")
    ax.semilogy(lags_s, np.clip(acf_out["ou"]["acf"], 1e-4, None), color=C_OU,
                linewidth=1.8, label=f"continuous process: fitted decay {acf_out['ou']['lambda_eff']:.2f}/s")
    ax.semilogy(lags_s, np.exp(-LAMBDA_RATE * lags_s), color=INK_2, linewidth=1.2,
                linestyle=(0, (4, 3)), label=f"exp(-lambda_rate t), lambda_rate = {LAMBDA_RATE:.2f}/s")
    ax.set_xlabel("lag (s)", fontsize=10, color=INK)
    ax.set_ylabel("autocorrelation of ln photons", fontsize=10, color=INK)
    ax.set_title("lambda_rate is the correlation-decay rate only for the continuous process",
                 fontsize=11.5, color=INK, pad=12, loc="left")
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK)
    save(fig, "F4_autocorrelation.png")

    fig, ax = plt.subplots(figsize=(8.6, 4.6))
    fig.patch.set_facecolor(SURFACE)
    style_axes(ax)
    render_axis = np.arange(T_RENDER)
    for arm, color, label in (("grid", C_GRID, "retired grid chain"),
                              ("ou", C_OU, "continuous stationary process"),
                              ("iid", C_IID, "i.i.d. reference (rho = 0)")):
        ks = render_out[arm]["first_last_ks"]
        ax.plot(render_axis, render_out[arm]["mean_t"], color=color, linewidth=1.2,
                alpha=0.9, label=f"{label}: first/last-window KS {ks:.3f}")
    ax.set_xlabel("frame (20 ms each)", fontsize=10, color=INK)
    ax.set_ylabel("mean aperture net signal per active emitter (ADU)", fontsize=10, color=INK)
    ax.set_title("Rendered videos: aperture photometry across the full 20 s",
                 fontsize=11.5, color=INK, pad=12, loc="left")
    legend = ax.legend(frameon=False, fontsize=9, loc="upper right")
    for text in legend.get_texts():
        text.set_color(INK)
    save(fig, "F5_render_invariance.png")


# ----------------------------------------------------------------------------------------------
# Report
# ----------------------------------------------------------------------------------------------

def write_report(law: dict, acf_out: dict, surv: dict, dim: dict, render_out: dict,
                 verdicts: dict) -> None:
    commit = subprocess.run(["git", "-C", REPO_ROOT, "rev-parse", "--short", "HEAD"],
                            capture_output=True, text=True).stdout.strip()
    lines: list[str] = []
    add = lines.append
    add("# Brightness stationarity audit - retired state grid versus the continuous stationary process")
    add("")
    add(f"Regenerate with `Script_Bank/Analysis/{os.path.basename(__file__)}` (commit {commit}; "
        "run with `PYTHONPATH` set to the repo root so the audited copy shadows the installed "
        "package - the script asserts this). Figures and the summary JSON sit beside this report "
        "in the Data_Bank Posit tier. The governing decision is the 2026-09-01 amendment in "
        "`MET_NEXT_MODEL_DESIGN_PLAN.md` (workspace root); the parameter disposition table lives "
        "there and in the 0.4.22 changelog entry.")
    add("")
    add("## 1. What is being proven, and how (the protocol)")
    add("")
    add("**Level 0 (by construction).** For z_0 ~ N(0, sigma_pc^2) and "
        "z_{t+1} = rho z_t + sigma_pc sqrt(1 - rho^2) eps_t with rho = exp(-lambda_rate "
        "delta_frame), induction gives z_t ~ N(0, sigma_pc^2) at every t, so photons "
        "= mu_pc exp(z_t) are exactly LogNormal(mu_pc, sigma_pc) at every frame. The levels "
        "below verify that the implementation realizes this construction; every plausible "
        "implementation bug (non-stationary initialization, wrong innovation scale, wrong "
        "delta_frame, brightness-coupled bleaching) breaks one of them.")
    add("")
    add("**Level 1 (latent, exact target known).** Per-frame mean/sd of ln photons with "
        "sampling bands and a zero-trend requirement; per-frame KS distance to the fully "
        "specified law; autocorrelation of ln photons against exp(-lambda_rate lag); the law "
        "among surviving dyes under strong bleaching; the two-dye sum against the exact "
        "self-convolution. Acceptance is two-tier - zero time trend AND distance below a "
        "practical tolerance - because at n = 2 x 10^5 per frame a bare significance test "
        "rejects on irrelevancies.")
    add("")
    add("**Level 2 (rendered, push-forward).** Downstream of the camera the observable is not "
        "lognormal; the provable statement is time invariance of the push-forward. Static "
        "scenes are rendered through the identical PSF/EMCCD path for both arms plus an "
        "i.i.d. reference (rho = 0), and aperture photometry at the true emitter positions is "
        "compared between the first and last 2 s windows. The i.i.d. arm doubles as the "
        "push-forward reference: its per-frame marginal equals the continuous arm's by "
        "construction.")
    add("")
    add("**Positive control.** Every test also runs on the retired grid chain and must flag it "
        "exactly where the design plan located the defect (occupancy relaxing to uniform "
        "within ~10 frames). A suite that cannot detect the known defect proves nothing.")
    add("")
    add("**Level 3 (comparison against experiment)** is pipeline-symmetric fitted-spot "
        "comparison (deposited ThunderSTORM protocols on synthetic renders) and is gated on "
        "the ThunderSTORM installation decision; it is specified in the plan and not run here.")
    add("")
    add("## 2. Results")
    add("")
    add("| check | criterion | retired grid | continuous process | verdict |")
    add("|---|---|---|---|---|")
    add(f"| occupancy vs documented weights | grid must exceed TV {TOL_TV_OU_EQUIV} (defect "
        f"visible) | max TV {law['grid_max_tv_intended']:.3f} | (no states) | "
        f"{verdicts['defect_visible']} |")
    add(f"| per-frame KS to the documented law | max over frames < {TOL_KS} | "
        f"{law['grid']['max_ks']:.3f} | {law['ou']['max_ks']:.4f} | {verdicts['ks']} |")
    add(f"| ln-mean drift across {T_LAW} frames | abs < {TOL_MEAN_DRIFT} | "
        f"{law['grid']['drift']:+.3f} | {law['ou']['drift']:+.4f} | {verdicts['drift']} |")
    add(f"| per-frame ln-sd error | max abs < {TOL_SD} | "
        f"{law['grid']['max_sd_err']:.3f} | {law['ou']['max_sd_err']:.4f} | {verdicts['sd']} |")
    add(f"| ACF decay rate vs lambda_rate | rel. err < {TOL_ACF:.0%} | "
        f"{acf_out['grid']['lambda_eff']:.2f}/s | {acf_out['ou']['lambda_eff']:.2f}/s "
        f"(err {acf_out['ou_lambda_rel_err']:.1%}) | {verdicts['acf']} |")
    surv_max = max(v["ks"] for v in surv.values())
    add(f"| survivor law under bleaching (OU) | KS < {TOL_KS} at frames 1/50/100/149 | - | "
        f"max {surv_max:.4f} | {verdicts['survivor']} |")
    dim_max = max(dim.values())
    add(f"| two-dye sum vs exact convolution (OU) | KS < {TOL_KS} at frames 0/100 | - | "
        f"max {dim_max:.4f} | {verdicts['dimer']} |")
    add(f"| rendered first vs last window (survivor set, per-emitter paired median shift) | "
        f"abs < 5% for OU and i.i.d.; grid reported | "
        f"{100 * render_out['grid']['median_rel_shift']:+.1f}% | "
        f"{100 * render_out['ou']['median_rel_shift']:+.1f}% (i.i.d. "
        f"{100 * render_out['iid']['median_rel_shift']:+.1f}%) | {verdicts['render']} |")
    add(f"| OU vs i.i.d. full-recording per-emitter medians (arm-paired push-forward "
        f"reference) | abs median diff < 5% (estimator SE ~2%) | - | "
        f"{100 * render_out['ou_vs_iid_paired_median_rel_diff']:+.1f}% | "
        f"{verdicts['pushforward']} |")
    add("")
    add(f"**Overall: {verdicts['overall']}.**")
    add("")
    add("The grid arm's numbers document the defect, not a regression: its occupancy relaxes "
        "from the documented interval weights to uniform (F2), which drags the per-frame "
        "ln-mean and ln-sd away from the intended values (F1) and keeps every per-frame KS "
        "distance at the discrete-atom floor (F3). The continuous arm holds the documented "
        "law at every frame. Under the grid, the fitted ACF decay differs from lambda_rate "
        "(F4): the same number does not mean the same tempo across the two mechanisms, which "
        "is why frozen artifacts carrying lambda_rate values fitted under grid semantics are "
        "flagged for adoption-time assessment.")
    add("")
    add("## 3. Audit operating point")
    add("")
    add(f"mu_pc = {MU_PC} photons, sigma_pc = {SIGMA_PC} (frozen Nuisance_DLI SGM as quoted in "
        f"PROJECT_CONTEXT.md), lambda_rate = 10^0.5 = {LAMBDA_RATE:.4f}/s (calibrated operating "
        f"point), prob_photo_bleach = {PROB_BLEACH} per {NUMB_BLEACH} frames, delta_frame = "
        f"{DELTA_FRAME} s. Level 2 camera: gamma {CAMERA['gamma']}, kappa_o {CAMERA['kappa_o']}, "
        f"kappa_b {CAMERA['kappa_b']}, kappa_s {CAMERA['kappa_s']}, kappa_q {CAMERA['kappa_q']}; "
        f"PSF mu_r {PSF_MU_R}, sigma_r {PSF_SIGMA_R}. Sizes: per-frame law {N_LAW:,} dyes x "
        f"{T_LAW} frames; ACF {N_ACF:,} x {T_ACF} (bleach off); dimer {N_DIMER:,}; render "
        f"{N_RENDER} static emitters x {T_RENDER} frames per arm.")
    add("")
    add("## 4. Figures")
    add("")
    for name, caption in (
            ("F1_per_frame_law.png", "Per-frame ln mean and sd against the intended values."),
            ("F2_grid_occupancy_relaxation.png", "Grid occupancy relaxation (the defect)."),
            ("F3_per_frame_ks.png", "Per-frame KS distance to the documented law."),
            ("F4_autocorrelation.png", "ln-brightness autocorrelation and fitted decay rates."),
            ("F5_render_invariance.png", "Rendered aperture photometry across the full 20 s.")):
        add(f"![{caption}]({name})")
        add("")
    with open(REPORT, "w") as handle:
        handle.write("\n".join(lines))


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Level 1: per-frame law ...")
    law = per_frame_law()
    print("Level 1: autocorrelation ...")
    acf_out = acf()
    print("Level 1: survivor conditioning ...")
    surv = survivor()
    print("Level 1: dimer sum ...")
    dim = dimer()
    print("Level 2: renders (3 arms x 1000 frames) ...")
    render_out = render_level()

    passed = lambda ok: "PASS" if ok else "FAIL"
    verdicts = dict(
        defect_visible=passed(law["grid_max_tv_intended"] > TOL_TV_OU_EQUIV),
        ks=passed(law["ou"]["max_ks"] < TOL_KS),
        drift=passed(abs(law["ou"]["drift"]) < TOL_MEAN_DRIFT),
        sd=passed(law["ou"]["max_sd_err"] < TOL_SD),
        acf=passed(acf_out["ou_lambda_rel_err"] < TOL_ACF),
        survivor=passed(max(v["ks"] for v in surv.values()) < TOL_KS),
        dimer=passed(max(dim.values()) < TOL_KS),
        render=passed(abs(render_out["ou"]["median_rel_shift"]) < 0.05
                      and abs(render_out["iid"]["median_rel_shift"]) < 0.05),
        pushforward=passed(abs(render_out["ou_vs_iid_paired_median_rel_diff"]) < 0.05),
    )
    verdicts["overall"] = passed(all(v == "PASS" for v in verdicts.values()))

    with open(os.path.join(OUT_DIR, "audit_summary.json"), "w") as handle:
        json.dump(dict(law={k: v for k, v in law.items() if not k.endswith("_t")},
                       acf={k: v for k, v in acf_out.items() if k != "lags_s"},
                       survivor=surv, dimer=dim,
                       render={k: ({kk: vv for kk, vv in v.items() if kk != "mean_t"}
                                   if isinstance(v, dict) else v)
                               for k, v in render_out.items()},
                       verdicts=verdicts), handle, indent=1, default=str)
    figures(law, acf_out, render_out)
    write_report(law, acf_out, surv, dim, render_out, verdicts)
    print("verdicts:", verdicts)


if __name__ == "__main__":
    main()

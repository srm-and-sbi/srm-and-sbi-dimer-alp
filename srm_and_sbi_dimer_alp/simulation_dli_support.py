"""Diffraction-limited imaging (DLI) rendering pipeline.

This module turns a reaction-diffusion simulation's particle trajectories
into synthetic fluorescence-microscopy videos. The pipeline:

    particle positions (per frame, per particle)
        -> Gaussian point-spread function rendered at each emitter
        -> per-pixel photon-count integrals via erf
        -> emitter brightness from a stationary continuous per-dye process
           (photo-physics: OU log-brightness flicker + absorbing photobleaching)
        -> corrected EMCCD noise chain (Poisson thinning -> stochastic Gamma EM
           register -> gain-independent Gaussian read noise -> bias)
        -> output video frames in ADU (analog-to-digital units)

Module contents:

    Point-spread function
        PointSpreadFunction       (abstract base; tag for PSF types)
        Gaussian                  (concrete Gaussian PSF with per-emitter widths)
        sample_psf_width          (sample sqrt(2)*sigma for each emitter)

    Detector model
        Detector                  (abstract base)
        EMCCD                     (electron-multiplying CCD: Poisson-Gamma-Normal chain)
        add_noise                 (Poisson thinning -> Gamma EM register -> read noise -> bias)
        generate_frames           (alias for add_noise)

    Intensity rendering
        compute_intensity         (top-level pixel-grid intensity assembly)
        add_pixel_counts          (helper: integrate Gaussian PSF over pixel grid)

    Brightness photo-physics
        generate_brightness_photons      (stationary continuous per-dye brightness:
                                          OU log-brightness, exact LogNormal(mu_pc,
                                          sigma_pc) marginal at every frame, with
                                          independent state-independent bleaching)

    Top-level renderer
        render_dli_video          (particle poses + a physical 11-key imaging vector
                                   -> fully noised video). Source-agnostic: it reads
                                   each imaging value by key, so a value is rendered the
                                   same whether it arrived as an inference target or a
                                   marginalized nuisance. Shared by the canonical DLI
                                   stage (imaging marginalized) and the Detector DLI stage
                                   (imaging inferred), which re-exports it as
                                   render_detector_video.
"""

from abc import ABC
from typing import Optional

import numpy as np
from scipy.special import erf
from scipy.stats import lognorm

from . import detector_parameterization as det
from .parameterization import (
    PARAMETERS,
    PARAMETERIZATION_RAW,
    PARAMETER_RAW_FIND,
)


# =============================================================================
# Point-spread function
# =============================================================================

class PointSpreadFunction(ABC):
    """Abstract base class for point-spread-function (PSF) models.

    A PSF describes how a point emitter's photons spread across the detector
    pixel grid due to diffraction. Concrete subclasses (e.g. `Gaussian`) carry
    the parameters needed by `compute_intensity` to integrate the PSF over
    each pixel.
    """
    pass


class Gaussian(PointSpreadFunction):
    """Gaussian PSF with a per-emitter width.

    The PSF for emitter j at sub-pixel position (mu_x, mu_y) is

        N(x; mu_x, sigma_j) * N(y; mu_y, sigma_j)

    where `sigma_j` varies across emitters (drawn from `sample_psf_width`).
    The class stores `sqrt(2) * sigma_j` directly because that is the form
    the erf-based pixel integration in `add_pixel_counts` uses.

    Args:
        sqrt_2sigma: 1D array of shape `(n_emitters,)` with the value
            `sqrt(2) * sigma_j` for each emitter `j`.
    """

    def __init__(self, sqrt_2sigma: np.ndarray):
        self.sqrt_2sigma = np.asarray(sqrt_2sigma)


def sample_psf_width(nemitters: int,
                     dist_label: str,
                     keyword_args: Optional[dict] = None,
                     seed: Optional[int] = None) -> np.ndarray:
    """Sample `sqrt(2) * sigma` for each emitter from a chosen distribution.

    Args:
        nemitters: Number of emitters to sample widths for.
        dist_label: One of:
            - "uniform": sample from `Uniform(low=1, high=3)`.
            - "lognormal": sample from a lognormal with parameters
              `s=sigma_r`, `loc=0`, `scale=mu_r` from `keyword_args`.
        keyword_args: Required for `"lognormal"`; must contain
            `mu_r` (scale) and `sigma_r` (shape).
        seed: Optional RNG seed. If None, sampling is non-deterministic.

    Returns:
        Array of shape `(nemitters,)` of `sqrt(2) * sigma` values, one
        per emitter.

    Raises:
        NotImplementedError: if `dist_label` is not "uniform" or "lognormal".
    """
    rng = np.random.default_rng(seed)
    if dist_label == "uniform":
        return rng.uniform(low=1, high=3, size=nemitters)
    if dist_label == "lognormal":
        if keyword_args is None or "mu_r" not in keyword_args or "sigma_r" not in keyword_args:
            raise ValueError("lognormal requires keyword_args with 'mu_r' and 'sigma_r'.")
        mu_r = keyword_args["mu_r"]
        sigma_r = keyword_args["sigma_r"]
        # scipy.stats.lognorm.rvs uses its own RNG; pass the modern Generator via random_state.
        return lognorm.rvs(s=sigma_r, loc=0, scale=mu_r, size=nemitters, random_state=rng)
    raise NotImplementedError(f"PSF-width distribution {dist_label!r} is not supported.")


# =============================================================================
# Detector model (EMCCD)
# =============================================================================

class Detector(ABC):
    """Abstract base class for detector noise models."""
    pass


class EMCCD(Detector):
    """Electron-multiplying CCD detector: Poisson-Gamma-Normal forward model.

    Attributes:
        quantum_efficiency: photon -> photoelectron probability, in [0, 1].
        em_gain: mean electron-multiplication gain, output e- per input e- (> 0).
        electrons_per_adu: conversion factor C, output e- per ADU (> 0).
        read_noise_adu: Gaussian read-noise standard deviation, in ADU (>= 0).
        bias_adu: electronic baseline added after conversion, in ADU.
        dark_current_e_per_s: thermal dark current, e- per pixel per s (default 0).
        cic_e_per_frame: clock-induced charge, e- per pixel per frame (default 0).
        exposure_s: exposure time in seconds; scales dark current per frame (default 0).
    """

    def __init__(self, quantum_efficiency: float, em_gain: float,
                 electrons_per_adu: float, read_noise_adu: float, bias_adu: float,
                 dark_current_e_per_s: float = 0.0, cic_e_per_frame: float = 0.0,
                 exposure_s: float = 0.0):
        if not 0.0 <= quantum_efficiency <= 1.0:
            raise ValueError("quantum_efficiency must lie in [0, 1]")
        if em_gain <= 0:
            raise ValueError("em_gain must be positive")
        if electrons_per_adu <= 0:
            raise ValueError("electrons_per_adu must be positive")
        if read_noise_adu < 0:
            raise ValueError("read_noise_adu must be non-negative")
        if dark_current_e_per_s < 0 or cic_e_per_frame < 0 or exposure_s < 0:
            raise ValueError("dark current, CIC, and exposure must be non-negative")
        self.quantum_efficiency = quantum_efficiency
        self.em_gain = em_gain
        self.electrons_per_adu = electrons_per_adu
        self.read_noise_adu = read_noise_adu
        self.bias_adu = bias_adu
        self.dark_current_e_per_s = dark_current_e_per_s
        self.cic_e_per_frame = cic_e_per_frame
        self.exposure_s = exposure_s


def add_noise(intensity: np.ndarray,
              detector: EMCCD,
              seed: Optional[int] = None) -> np.ndarray:
    """Render EMCCD counts from expected incident photons.

    Applies the Poisson-Gamma-Normal chain (sec. 2): Poisson photoelectrons,
    stochastic Gamma electron multiplication, conversion to ADU, gain-independent
    Gaussian read noise, and bias.

    Args:
        intensity: Expected incident photons per pixel per frame; finite and
            non-negative. Emitter signal and optical background must already be summed.
        detector: An EMCCD instance (sec. 2 unit contract).
        seed: Optional RNG seed; None (default) draws non-deterministically.

    Returns:
        Array of the same shape as `intensity`, in ADU (floating point).
    """
    rng = np.random.default_rng(seed)
    expected_photons = np.asarray(intensity, dtype=np.float64)
    if not np.all(np.isfinite(expected_photons)):
        raise ValueError("intensity contains non-finite values")
    if np.any(expected_photons < 0):
        raise ValueError("intensity must be non-negative")

    # Stage 1 - photoelectrons (Poisson thinning; optional dark current + CIC).
    mean_input_electrons = (
        detector.quantum_efficiency * expected_photons
        + detector.dark_current_e_per_s * detector.exposure_s
        + detector.cic_e_per_frame
    )
    input_electrons = rng.poisson(mean_input_electrons)

    # Stage 2 - electron multiplication: Gamma(shape=N, scale=g); zero input -> zero output.
    output_electrons = np.zeros(expected_photons.shape, dtype=np.float64)
    positive = input_electrons > 0
    output_electrons[positive] = rng.gamma(
        shape=input_electrons[positive].astype(np.float64),
        scale=detector.em_gain,
    )

    # Stage 3 - conversion to ADU.
    frames = output_electrons / detector.electrons_per_adu
    # Stage 4 - gain-independent Gaussian read noise (ADU), after multiplication.
    frames += rng.standard_normal(frames.shape) * detector.read_noise_adu
    # Stage 5 - electronic bias.
    frames += detector.bias_adu
    return frames


def generate_frames(intensity: np.ndarray,
                    detector: EMCCD,
                    seed: Optional[int] = None) -> np.ndarray:
    """Alias for `add_noise` under the "get_frames" naming pattern.

    See `add_noise` for argument and return value details.
    """
    return add_noise(intensity, detector, seed=seed)


# =============================================================================
# Intensity rendering (Gaussian PSF integration over pixel grid)
# =============================================================================

def _erf(x: np.ndarray, bounds: np.ndarray, sqrt_2sigma: np.ndarray) -> np.ndarray:
    """Integrate a Gaussian PSF along ONE coordinate over a pixel grid.

    For each emitter j with center x_j and width sigma_j, the integral of
    the 1D Gaussian N(x; x_j, sigma_j) over the pixel interval
    `[bounds[i], bounds[i+1]]` is

        I_ij = (erf((bounds[i] - x_j) / (sqrt(2) sigma_j))
                - erf((bounds[i+1] - x_j) / (sqrt(2) sigma_j))) / 2

    Note the sign convention: this formulation puts `erf(bounds[i]) -
    erf(bounds[i+1])` in the numerator (instead of the more conventional
    `erf(upper) - erf(lower)`). The result is the NEGATIVE of the true
    integral; in `add_pixel_counts` two such factors (x and y) are
    multiplied so the sign-flips cancel out and the per-pixel photon count
    comes out positive.

    Args:
        x: Tensor of shape `(nframes, 1, nemitters)`. One spatial coordinate
            (x or y) of every emitter at every frame. The middle dim is a
            singleton to match the broadcasting in `add_pixel_counts`.
        bounds: 1D array of pixel-boundary positions along the same
            coordinate, length `n_pixels + 1`. Typically `linspace(0,
            stem_root_size, stem_root_size + 1)`.
        sqrt_2sigma: 1D array of `sqrt(2) * sigma_j` for each emitter j,
            shape `(nemitters,)`.

    Returns:
        Array of shape `(n_pixels, nemitters, nframes)` containing the
        (sign-flipped) per-pixel Gaussian integrals.
    """
    # Move frame axis to last position: (1, nemitters, nframes).
    X = np.moveaxis(x, 0, 2)
    # Reshape sqrt_2sigma to (1, nemitters, 1) for broadcast over bounds and frames.
    sqrt_2sigma = np.asarray(sqrt_2sigma).reshape(1, -1, 1)
    # Normalized distance from each pixel boundary: (n_bounds, nemitters, nframes).
    X = (bounds[:, None, None] - X) / sqrt_2sigma
    # Adjacent-boundary erf difference -> (n_pixels, nemitters, nframes).
    return (erf(X[:-1, :, :]) - erf(X[1:, :, :])) / 2


def add_pixel_counts(intensity: np.ndarray,
                     tracks: np.ndarray,
                     brightness_array: np.ndarray,
                     xbounds: np.ndarray,
                     ybounds: np.ndarray,
                     PSF: Gaussian) -> np.ndarray:
    """Accumulate per-emitter PSF contributions onto a pixel-grid intensity.

    For each emitter j and each frame k, adds the per-pixel Gaussian-PSF
    integral (scaled by the emitter brightness) to `intensity`. Ghost
    particles (entries marked `NaN` in `tracks`) are masked out before
    integration: their positions are pushed beyond the boundary, and
    their brightness is zeroed.

    Args:
        intensity: Pixel-grid intensity array of shape
            `(n_pixels_x, n_pixels_y, n_frames)`, in photon counts. Modified
            in place (entries added to existing values, e.g. dark counts).
        tracks: Particle coordinates of shape `(n_frames, 2, n_emitters)`,
            in pixel units. May contain `NaN` for particles absent in a frame.
        brightness_array: Per-emitter photon counts of shape
            `(1, n_emitters, n_frames)`.
        xbounds, ybounds: 1D arrays of pixel boundary positions along each
            axis (length `n_pixels + 1`).
        PSF: A `Gaussian` instance with `sqrt_2sigma` of shape `(n_emitters,)`.

    Returns:
        The modified `intensity` array (same reference; updated in place).
    """
    # Step 1: identify ghost particles (NaN entries in tracks).
    ghost_mask = np.isnan(tracks)                              # (n_frames, 2, n_emitters)
    # Step 2: push ghost positions far beyond the boundary so their PSF
    # integral over the grid is effectively zero.
    eternity_bound = np.max([xbounds, ybounds]) * 2
    tracks[ghost_mask] = eternity_bound
    # Step 3: zero out the brightness of ghost emitters.
    # ghost_mask[0, :, :] is (2, n_emitters); transpose-broadcast onto brightness_array
    # which has shape (1, n_emitters, n_frames). The "any over 2 spatial axes" reduction
    # is approximated by indexing on the x-coord row only ([0]).
    ghost_mask_zero = np.moveaxis(ghost_mask, 0, 2)[[0], :, :]  # (1, n_emitters, n_frames)
    brightness_array[ghost_mask_zero] = 0
    # Step 4: PSF integral along x, weighted by brightness.
    X = _erf(tracks[:, [0], :], xbounds, PSF.sqrt_2sigma)      # (n_pix_x, n_emitters, n_frames)
    X *= brightness_array
    # Step 5: PSF integral along y.
    Y = _erf(tracks[:, [1], :], ybounds, PSF.sqrt_2sigma)      # (n_pix_y, n_emitters, n_frames)
    Y = np.transpose(Y, (1, 0, 2))                              # (n_emitters, n_pix_y, n_frames)
    # Step 6: combine via einsum -> (n_pix_x, n_pix_y, n_frames), summed over emitters.
    intensity += np.einsum("ijk,jlk->ilk", X, Y)
    return intensity


def compute_intensity(tracks: np.ndarray,
                      emitter_photons: np.ndarray,
                      background_photons: np.ndarray,
                      xbounds: np.ndarray,
                      ybounds: np.ndarray,
                      PSF: Gaussian,
                      dimer_mask: Optional[np.ndarray] = None,
                      dimer_mule: float = 2.0,
                      dimer_model: str = "sum",
                      dimer_photons: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the noise-free per-pixel photon-count image.

    Args:
        tracks: Particle coordinates of shape `(n_frames, 2, n_emitters)`,
            in pixel units. NaN for absent particles.
        emitter_photons: Per-emitter, per-frame latent photon values of shape
            `(n_frames, n_emitters)` (0 where bleached). Produced by
            `generate_brightness_photons`; any per-frame photon source with this
            shape is accepted (the stationarity audit exploits this to drive
            reference arms through the identical downstream path).
        background_photons: 2D array of shape `(n_pix_x, n_pix_y)` — the pre-PSF
            per-pixel photon floor added under the emitter signal. Zero by default
            (no background); the detector fills it with the optical background
            `kappa_o`. This is NOT dark current (thermal electrons do not pass
            through QE and are handled separately by `EMCCD`).
        xbounds, ybounds: 1D arrays of pixel-boundary positions.
        PSF: `Gaussian` instance with per-emitter `sqrt_2sigma`.
        dimer_mask: Optional boolean array of shape `(1, n_emitters, n_frames)`
            flagging emitters in dimer states (B or C). Their brightness is
            combined per `dimer_model` -- the `sum` default adds an independent
            second-label draw; `multiply` scales one draw by `dimer_mule` -- to
            model the brighter signal from a dimer compared to a monomer.
        dimer_mule: Merged-dimer brightness relative to a monomer, applied ONLY
            under `dimer_model="multiply"` (a dimer is two labels within one PSF,
            imaged as one spot whose photons combine). Regime-dependent in [1, 2]:
            `2.0` = two PERMANENTLY-ON labels (the always-on ATTO 647N case);
            `sqrt(2)` ~= 1.41 when only ~one label is visible on average
            (photoswitching dye, or ~50% labeling). See PROJECT_CONTEXT.md.
        dimer_model: how a dimer's two labels combine. "sum" (default): the merged
            brightness is the SUM of two INDEPENDENT monomer brightnesses (photons add; an
            n-mer's intensity distribution is the n-fold convolution of the monomer's, Mutch
            et al. 2007 Biophys J; Digman & Gratton 2008 Number & Brightness) -- same mean
            as `dimer_mule=2` but a lighter upper tail; requires `dimer_photons`. "multiply":
            brightness is scaled by `dimer_mule` (a fixed factor of one monomer draw).
        dimer_photons: the second label's INDEPENDENT per-frame photon values (same shape
            as `emitter_photons`); used only when `dimer_model="sum"`.

    Returns:
        Intensity array of shape `(n_pix_x, n_pix_y, n_frames)`, in photon
        counts (positive, real-valued).
    """
    # Broadcast the photon floor along the frame axis -> (n_pix_x, n_pix_y, n_frames).
    intensity = np.repeat(background_photons[:, :, None], tracks.shape[0], axis=2)
    # emitter_photons has shape (n_frames, n_emitters); transpose to
    # (n_emitters, n_frames) and reshape to (1, n_emitters, n_frames) for
    # broadcast in add_pixel_counts.
    brightness_array = emitter_photons.T.reshape(1, emitter_photons.shape[1], -1).copy()
    if dimer_mask is not None:
        if dimer_model == "sum":
            # Physically-motivated dimer: two co-located labels, photons ADD, so the merged
            # brightness is the SUM of two INDEPENDENT monomer brightnesses -- an n-mer's
            # intensity distribution is the n-fold convolution of the monomer's (Mutch et al.
            # 2007 Biophys J; Digman & Gratton 2008 Number & Brightness). Same mean as
            # dimer_mule=2 but a lighter upper tail than doubling one draw. dimer_photons is
            # the second label's INDEPENDENT flicker trajectory.
            if dimer_photons is None:
                raise ValueError("dimer_model='sum' requires dimer_photons (the second label's "
                                 "independent per-frame photon values).")
            second = dimer_photons.T.reshape(1, dimer_photons.shape[1], -1)
            brightness_array[dimer_mask] += second[dimer_mask]
        else:
            # 'multiply': merged-spot brightness = monomer * dimer_mule. A dimer is
            # two labels within one PSF, imaged as ONE spot whose photons combine; dimer_mule in
            # [1, 2]: 2 for two permanently-on labels (photons sum -- the always-on ATTO 647N
            # case); sqrt(2) under blinking / partial labeling. See PROJECT_CONTEXT.md.
            brightness_array[dimer_mask] *= dimer_mule
    intensity = add_pixel_counts(intensity, tracks, brightness_array, xbounds, ybounds, PSF)
    return intensity


# =============================================================================
# Brightness photo-physics (stationary continuous per-dye process)
# =============================================================================

def generate_brightness_photons(nframes: int,
                                nemitters: int,
                                mu_pc: float,
                                sigma_pc: float,
                                lambda_rate: float,
                                prob_photo_bleach: float,
                                numb_photo_bleach: int,
                                delta_frame: float,
                                seed: Optional[int] = None) -> np.ndarray:
    """Stationary continuous per-dye brightness with independent bleaching.

    Per-dye log-brightness follows a stationary Ornstein-Uhlenbeck process,
    updated per frame as an AR(1):

        z_0 ~ N(0, sigma_pc^2)
        z_{t+1} = rho * z_t + sigma_pc * sqrt(1 - rho^2) * eps_t,   rho = exp(-lambda_rate * delta_frame)
        photons_t = mu_pc * exp(z_t)

    By induction the per-frame marginal is exactly LogNormal(ln mu_pc, sigma_pc^2)
    at EVERY frame: no initialization transient, no brightness ceiling, and
    sigma_pc carries its documented marginal meaning. `lambda_rate` is the
    correlation-decay rate of ln-brightness (ACF(lag) = exp(-lambda_rate * lag)),
    NOT a jump-event rate; brightness is constant within a frame (one draw per
    frame per dye).

    Photobleaching is state-independent, applied as an independent per-frame
    Bernoulli whose rate accrues `prob_photo_bleach` over `numb_photo_bleach`
    frames:

        prob_1 = 1 - (1 - prob_photo_bleach) ** (1 / numb_photo_bleach)

    A bleached dye emits 0 photons from its bleach frame onward (absorbing), and
    every dye is active at the first frame. Because bleaching is
    state-independent, the brightness law among active dyes is unchanged by it.

    Args:
        nframes: Number of frames.
        nemitters: Number of independent dyes (one process per dye; a dimer's
            second label gets its own independent call).
        mu_pc, sigma_pc: Median (photons) and ln-spread of the per-frame
            single-dye brightness law.
        lambda_rate: Correlation-decay rate of ln-brightness, in 1/s.
        prob_photo_bleach: Bleaching probability over `numb_photo_bleach` frames.
        numb_photo_bleach: FIXED reference-frame count (100), not the clip's
            frame count. Pinning it makes the per-frame bleach probability
            constant across clip durations, so a longer clip simply accumulates
            more bleaching (see the photobleaching model in PROJECT_CONTEXT.md).
        delta_frame: Time between frames in seconds.
        seed: Optional RNG seed.

    Returns:
        Array of shape `(nframes, nemitters)`: latent photon values per dye per
        frame (float, 0.0 where bleached).
    """
    rng = np.random.default_rng(seed)
    rho = np.exp(-lambda_rate * delta_frame)
    innovation = sigma_pc * np.sqrt(1.0 - rho * rho)
    z = np.empty((nframes, nemitters), dtype=float)
    z[0] = rng.normal(0.0, sigma_pc, size=nemitters)
    noise = rng.normal(0.0, innovation, size=(max(nframes - 1, 0), nemitters))
    for t in range(1, nframes):
        z[t] = rho * z[t - 1] + noise[t - 1]
    photons = mu_pc * np.exp(z)
    prob_1 = 1.0 - (1.0 - prob_photo_bleach) ** (1.0 / numb_photo_bleach)
    bleach_draws = rng.random(size=(nframes, nemitters)) < prob_1
    bleach_draws[0, :] = False
    active = np.cumprod(~bleach_draws, axis=0).astype(bool)
    return photons * active


# =============================================================================
# Top-level renderer (source-agnostic; shared by both DLI stages)
# =============================================================================

# The full 11-key imaging vector, in det.DETECTOR_IMAGING order: the six emitter parameters
# (mu_r, sigma_r, mu_pc, sigma_pc, prob_photo_bleach, lambda_rate) followed by the five SCOPE
# camera parameters (gamma, kappa_o, kappa_b, kappa_s, kappa_q). render_dli_video reads by key,
# so this is the fixed contract the DLI stages assemble imaging_physical to. The order is
# defined once in detector_parameterization, so both DLI stages and this renderer share a
# single source of truth (no drift) regardless of each block's role.
_IMAGING_KEYS = det.DETECTOR_IMAGING_KEYS


def _fixed(key: str):
    """Physical VALUE of a fixed imaging hyperparameter (read from the canonical table)."""
    return PARAMETERIZATION_RAW[PARAMETER_RAW_FIND[key]]["VALUE"]


def render_dli_video(pro_tray_poses: np.ndarray,
                     imaging_physical: np.ndarray,
                     dimer_mask=None,
                     dimer_model: str = "sum",
                     seed=None,
                     verbose: bool = False) -> np.ndarray:
    """Render one video from particle poses and a physical imaging vector.

    The source-agnostic diffraction-limited-imaging renderer shared by both DLI stages. It
    sources the 11 imaging parameters entirely from ``imaging_physical`` -- the imaging
    parameters in ``det.DETECTOR_IMAGING`` order (six emitter parameters + five SCOPE camera),
    in physical space -- reading each value by key, so it is agnostic to whether a parameter
    arrived as an inference target or a marginalized nuisance. The canonical (biology) stage
    draws the whole imaging block as a nuisance (photophysics from the ``Nuisance_DLI``
    artifact, camera from the SCOPE box); the Detector stage draws the six emitter parameters
    as its learnable target and the five camera as the SCOPE nuisance (which re-exports this
    renderer as ``render_detector_video``). The fixed hyperparameters that are not part of the
    vector (``numb_photo_bleach``, ``dimer_mule``) are read from the
    canonical parameter table; ``delta_frame`` is the fixed camera cadence
    (``PARAMETERS.simulation.timing``). The lower-level building blocks are reused unchanged.

    Args:
        pro_tray_poses: particle coordinates ``(n_frames, n_emitters, 3)`` in nm
            (only x, y are used; z dropped); NaN for absent particles.
        imaging_physical: physical values of the 11 imaging parameters,
            in ``det.DETECTOR_IMAGING`` order.
        dimer_mask: optional boolean mask ``(1, n_emitters, n_frames)`` flagging
            dimer-state emitters; their brightness is combined per ``dimer_model``.
        dimer_model: "sum" (default) adds an independent second-label draw -- a dimer is
            two independent labels whose photon counts add (sum of two monomers: same
            mean, lighter tail than doubling; an n-mer's brightness is the n-fold
            convolution of the monomer, Mutch et al. 2007). "multiply" scales one draw by
            ``dimer_mule`` (heavier tail; retained as an option).
        seed: optional RNG seed for PSF widths, brightness, and EMCCD noise.
        verbose: print the resolved OU brightness quantities.

    Returns:
        Frames of shape ``(root_size_px, root_size_px, n_frames)`` in ADU.
    """
    imaging_physical = np.asarray(imaging_physical, dtype=float)
    if imaging_physical.shape != (len(_IMAGING_KEYS),):
        raise ValueError(
            f"imaging_physical has shape {imaging_physical.shape}; expected "
            f"({len(_IMAGING_KEYS)},) for keys {_IMAGING_KEYS}."
        )
    img = dict(zip(_IMAGING_KEYS, imaging_physical))

    nframes, nemitters, _ = pro_tray_poses.shape
    stem_geometry = PARAMETERS.simulation.stem
    pixel_size_nm = stem_geometry.pixel_size_nm
    root_size_px = stem_geometry.root_size_px

    # --- Camera params + EMCCD detector (imaging from the vector) ----------
    # The videos identify gain and conversion only through gamma = g/C (ADU per
    # photoelectron). add_noise computes Gamma(N, em_gain)/electrons_per_adu, so
    # em_gain=gamma with electrons_per_adu=1.0 yields Gamma(N, gamma) exactly.
    detector = EMCCD(
        quantum_efficiency=img["kappa_q"],       # QE from the SCOPE nuisance (config-anchored ~0.9; only gamma*kappa_q identifiable)
        em_gain=img["gamma"],                    # gamma = g/C, the sole identifiable gain quantity
        electrons_per_adu=1.0,                   # C absorbed into gamma
        read_noise_adu=img["kappa_s"],
        bias_adu=img["kappa_b"],
    )

    # --- PSF params + per-emitter widths (mu_r, sigma_r from the vector) ---
    per_emitter_sqrt2sigma = sample_psf_width(
        nemitters,
        PARAMETERS.simulation.dli.sqrt_2sigma_dist_label,
        keyword_args={"mu_r": img["mu_r"], "sigma_r": img["sigma_r"]},
        seed=seed,
    )
    PSF = Gaussian(per_emitter_sqrt2sigma)

    # --- Photo-physics: stationary continuous per-dye brightness -----------
    # mu_pc, sigma_pc, prob_photo_bleach, lambda_rate from the vector;
    # numb_photo_bleach fixed; delta_frame = fixed cadence. The per-frame
    # marginal is exactly LogNormal(mu_pc, sigma_pc) at every frame and
    # lambda_rate is the correlation-decay rate of ln-brightness
    # (see generate_brightness_photons).
    mu_pc = img["mu_pc"]
    sigma_pc = img["sigma_pc"]
    delta_frame = PARAMETERS.simulation.timing.frame_time_seconds
    if verbose:
        rho = np.exp(-img["lambda_rate"] * delta_frame)
        print(f"[render_dli_video] OU brightness: mu_pc={mu_pc:.4g} photons, "
              f"sigma_pc={sigma_pc:.4g} (ln), lambda_rate={img['lambda_rate']:.4g}/s "
              f"(per-frame rho={rho:.4f}), prob_photo_bleach={img['prob_photo_bleach']:.4g} "
              f"per {_fixed('numb_photo_bleach')} frames")
    emitter_photons = generate_brightness_photons(
        nframes=nframes, nemitters=nemitters,
        mu_pc=mu_pc, sigma_pc=sigma_pc,
        lambda_rate=img["lambda_rate"],
        prob_photo_bleach=img["prob_photo_bleach"],
        numb_photo_bleach=_fixed("numb_photo_bleach"),
        delta_frame=delta_frame, seed=seed,
    )
    # For dimer_model="sum", each dimer's SECOND label needs its OWN independent flicker
    # trajectory (a dimer = two labels, brightness = X1 + X2). Independent seed so it is not
    # identical to the first label's; None stays non-deterministic.
    dimer_photons = None
    if dimer_model == "sum":
        dimer_photons = generate_brightness_photons(
            nframes=nframes, nemitters=nemitters,
            mu_pc=mu_pc, sigma_pc=sigma_pc,
            lambda_rate=img["lambda_rate"],
            prob_photo_bleach=img["prob_photo_bleach"],
            numb_photo_bleach=_fixed("numb_photo_bleach"),
            delta_frame=delta_frame,
            seed=(None if seed is None else seed + 1),
        )

    # --- Pixel grid + optical background floor -----------------------------
    # kappa_o = optical background (incident photons): ONE scalar per video,
    # broadcast over all pixels/frames as the pre-PSF photon floor. Distinct from
    # dark current (thermal electrons, genuinely zero here, handled by EMCCD).
    optical_background = np.full(
        shape=(root_size_px, root_size_px),
        fill_value=img["kappa_o"],
        dtype=float,
    )
    xbounds = np.linspace(0, root_size_px, root_size_px + 1)
    ybounds = np.linspace(0, root_size_px, root_size_px + 1)

    # --- Particle positions -> pixel-unit (x, y) tracks --------------------
    tracks_pixels = pro_tray_poses[:, :, :2] / pixel_size_nm
    tracks_pixels = np.transpose(tracks_pixels, (0, 2, 1)).astype(np.float64)

    # --- Noise-free intensity + EMCCD noise --------------------------------
    intensity = compute_intensity(
        tracks=tracks_pixels, emitter_photons=emitter_photons,
        background_photons=optical_background, xbounds=xbounds, ybounds=ybounds, PSF=PSF,
        dimer_mask=dimer_mask, dimer_mule=_fixed("dimer_mule"),
        dimer_model=dimer_model, dimer_photons=dimer_photons,
    )
    frames = generate_frames(intensity, detector, seed=seed)
    return frames

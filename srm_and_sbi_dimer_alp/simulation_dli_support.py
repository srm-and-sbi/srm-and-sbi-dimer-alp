"""Diffraction-limited imaging (DLI) rendering pipeline.

This module turns a reaction-diffusion simulation's particle trajectories
into synthetic fluorescence-microscopy videos. The pipeline:

    particle positions (per frame, per particle)
        -> Gaussian point-spread function rendered at each emitter
        -> per-pixel photon-count integrals via erf
        -> emitter brightness modulated by a Markov-chain state machine
           (photo-physics: blinking, photobleaching, multiple intensity levels)
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

    Brightness state machine (photo-physics)
        compute_brightness               (theoretical brightness values at quantiles)
        compute_brightness_probability   (initial state probabilities)
        initialize_emitter_states        (sample initial state per emitter)
        generate_state_trajectories      (Markov-chain state evolution over frames)

    Transition matrices (CTMC and DTMC)
        compute_matrices          (CTMC generator Q and DTMC stochastic matrix P
                                   from photobleaching + neighbor-decay rates)

    Top-level orchestrator
        simulate_dli              (theta + particle poses -> fully noised video)
"""

from abc import ABC
from typing import Optional, Tuple

import numpy as np
from scipy.linalg import expm
from scipy.special import erf
from scipy.stats import lognorm

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
                      states: np.ndarray,
                      brightness: np.ndarray,
                      background_photons: np.ndarray,
                      xbounds: np.ndarray,
                      ybounds: np.ndarray,
                      PSF: Gaussian,
                      dimer_mask: Optional[np.ndarray] = None,
                      dimer_mule: float = 2.0,
                      dimer_model: str = "sum",
                      dimer_states: Optional[np.ndarray] = None) -> np.ndarray:
    """Compute the noise-free per-pixel photon-count image.

    Args:
        tracks: Particle coordinates of shape `(n_frames, 2, n_emitters)`,
            in pixel units. NaN for absent particles.
        states: Emitter state index of shape `(n_frames, n_emitters)` —
            an integer in `range(n_brightness_states)` per frame per emitter.
        brightness: 1D array of brightness values (photon counts) indexed by
            state, length `n_brightness_states`.
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
            as `dimer_mule=2` but a lighter upper tail; requires `dimer_states`. "multiply":
            brightness is scaled by `dimer_mule` (a fixed factor of one monomer draw).
        dimer_states: the second label's INDEPENDENT brightness state trajectory (same shape
            as `states`); used only when `dimer_model="sum"`.

    Returns:
        Intensity array of shape `(n_pix_x, n_pix_y, n_frames)`, in photon
        counts (positive, real-valued).
    """
    # Broadcast the photon floor along the frame axis -> (n_pix_x, n_pix_y, n_frames).
    intensity = np.repeat(background_photons[:, :, None], tracks.shape[0], axis=2)
    # Look up per-emitter, per-frame brightness from the state index.
    # states.T has shape (n_emitters, n_frames); indexing brightness produces (n_emitters, n_frames);
    # reshape to (1, n_emitters, n_frames) for broadcast in add_pixel_counts.
    brightness_array = brightness[states.T].reshape(1, states.shape[1], -1)
    if dimer_mask is not None:
        if dimer_model == "sum":
            # Physically-motivated dimer: two co-located labels, photons ADD, so the merged
            # brightness is the SUM of two INDEPENDENT monomer brightnesses -- an n-mer's
            # intensity distribution is the n-fold convolution of the monomer's (Mutch et al.
            # 2007 Biophys J; Digman & Gratton 2008 Number & Brightness). Same mean as
            # dimer_mule=2 but a lighter upper tail than doubling one draw. dimer_states is the
            # second label's INDEPENDENT flicker trajectory.
            if dimer_states is None:
                raise ValueError("dimer_model='sum' requires dimer_states (the second label's "
                                 "independent brightness state trajectory).")
            second = brightness[dimer_states.T].reshape(1, dimer_states.shape[1], -1)
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
# Brightness state machine (photo-physics)
# =============================================================================

def compute_brightness(brightness_quantile: np.ndarray,
                       scale: float,
                       shape: float,
                       loc: float = 0) -> np.ndarray:
    """Theoretical brightness values at given quantiles of a lognormal distribution.

    Each emitter at each frame is in one of `len(brightness_quantile)`
    discrete brightness states. The state-`i` brightness is the
    `brightness_quantile[i]`-th quantile of a lognormal distribution
    `LogNormal(s=shape, loc=loc, scale=scale)`, rounded to the nearest integer
    photon count.

    Args:
        brightness_quantile: 1D array of probabilities in [0, 1] specifying
            the brightness states (typically e.g. [0, 0.05, 0.1, 0.25, 0.5,
            0.75, 0.9, 0.95]). The state-0 quantile is conventionally 0
            (photobleached state).
        scale, shape, loc: Lognormal distribution parameters
            (`scipy.stats.lognorm` parameterization).

    Returns:
        Array of integer brightness values, one per quantile.
    """
    return np.round(lognorm.ppf(q=brightness_quantile, s=shape, loc=loc, scale=scale), 0)


def compute_brightness_probability(brightness_quantile: np.ndarray) -> np.ndarray:
    """Stationary probability of each brightness state, partitioned from the quantile vector.

    Given `brightness_quantile` as a partition of [0, 1] (e.g. [0, 0.05,
    0.1, 0.25, 0.5, 0.75, 0.9, 0.95]), this assigns each state a probability
    proportional to its surrounding interval width. The state-0 (photo-
    bleached) state gets probability 0 (no emitter starts photobleached).
    The result is normalized to sum to 1.

    Args:
        brightness_quantile: 1D array of quantile values.

    Returns:
        1D probability vector, same length as `brightness_quantile`,
        summing to 1.
    """
    q = brightness_quantile
    prob = np.empty_like(q, dtype=float)
    prob[0] = 0
    prob[1] = q[1] + (q[2] - q[1]) / 2
    for i in range(2, len(q) - 1):
        prob[i] = ((q[i] - q[i - 1]) + (q[i + 1] - q[i])) / 2
    prob[-1] = (1 - q[-1]) + (q[-1] - q[-2]) / 2
    return prob / np.sum(prob)


def initialize_emitter_states(n_emitters: int,
                              brightness_quantile: np.ndarray,
                              scale: float,
                              shape: float,
                              loc: float = 0,
                              seed: Optional[int] = None
                              ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Sample an initial brightness state for each emitter.

    Each emitter's initial state is drawn from the stationary probability
    distribution computed from `brightness_quantile` (state 0, the
    photobleached state, has probability 0; remaining states are partitioned
    by interval width).

    Args:
        n_emitters: Number of emitters.
        brightness_quantile: Quantile partition (see `compute_brightness`).
        scale, shape, loc: Lognormal parameters.
        seed: Optional RNG seed for reproducibility.

    Returns:
        (initial_states, brightness, brightness_probability) where:
            - initial_states has shape `(n_emitters,)`, integer state index.
            - brightness has shape `(n_states,)`, brightness value per state.
            - brightness_probability has shape `(n_states,)`, stationary
              probability per state.
    """
    rng = np.random.default_rng(seed)
    brightness = compute_brightness(brightness_quantile, scale, shape, loc)
    brightness_probability = compute_brightness_probability(brightness_quantile)
    initial_states = rng.choice(a=len(brightness), size=n_emitters, p=brightness_probability)
    return initial_states, brightness, brightness_probability


def generate_state_trajectories(nframes: int,
                                nemitters: int,
                                P: np.ndarray,
                                brightness_quantile: np.ndarray,
                                scale: float,
                                shape: float,
                                loc: float = 0,
                                seed: Optional[int] = None) -> np.ndarray:
    """Sample state trajectories from a discrete-time Markov chain.

    Initial states are sampled from the stationary brightness distribution
    (`initialize_emitter_states`); subsequent states are sampled from the
    DTMC transition matrix `P[i, j] = P(state_{t+1}=j | state_t=i)`.

    Args:
        nframes: Number of frames (length of each trajectory).
        nemitters: Number of emitters (number of independent trajectories).
        P: DTMC stochastic matrix, shape `(n_states, n_states)`. Each row
            sums to 1.
        brightness_quantile, scale, shape, loc: Parameters for the initial
            state distribution (see `initialize_emitter_states`).
        seed: Optional RNG seed.

    Returns:
        Array of shape `(nframes, nemitters)` of integer state indices.
    """
    rng = np.random.default_rng(seed)
    states = np.full(shape=(nframes, nemitters), fill_value=-1, dtype=int)
    initial_states, _, _ = initialize_emitter_states(
        nemitters, brightness_quantile, scale, shape, loc, seed=seed,
    )
    states[0, :] = initial_states
    _propagate_chain(states, P, rng)
    return states


# =============================================================================
# Transition matrices (CTMC and DTMC)
# =============================================================================

def compute_matrices(mu_pc: float,
                     sigma_pc: float,
                     brightness_quantile: np.ndarray,
                     prob_photo_bleach: float,
                     numb_photo_bleach: int,
                     delta_frame: float,
                     lambda_rate: float,
                     kappa_penalty: float = 1.0,
                     verbose: bool = False) -> Tuple[np.ndarray, np.ndarray]:
    """Build the CTMC generator `Q` and DTMC stochastic matrix `P` for the
    emitter brightness state machine.

    The state space is the set of brightness levels indexed by `brightness_quantile`,
    with state 0 being the photobleached state.

    Rate model:
        - Photobleaching: each non-zero state `i` has a rate
          `epsilon_rate` into the photobleached state (0).
          `epsilon_rate` is derived from the discrete-time
          per-`numb_photo_bleach`-frames bleaching probability:

              prob_1 = 1 - (1 - prob_photo_bleach)^(1 / numb_photo_bleach)
              epsilon_rate = -ln(1 - prob_1) / delta_frame

          giving a continuous-time rate that yields `prob_photo_bleach`
          probability of bleaching over `numb_photo_bleach` frames.
        - Inter-state transitions: between non-zero states `i` and `j`, the
          rate is

              Q[i, j] = lambda_rate * exp(-kappa_penalty * |brightness[i] - brightness[j]| / sigma_bright)

          where `sigma_bright` is the photon-space standard deviation of the
          brightness lognormal, so the penalty acts on the brightness change in
          units of its own spread. Fast for neighboring levels, slow for
          far-apart ones; `kappa_penalty` (fixed 1.0) sets the locality — a
          larger value penalizes far jumps more strongly.
        - Diagonal: `Q[i, i] = -sum_j Q[i, j]` (CTMC row-sum convention).
        - State 0 (photobleached) is absorbing: no transitions OUT of state 0.

    The DTMC stochastic matrix `P` is the matrix exponential
    `P = exp(Q * delta_frame)`. Each entry `P[i, j]` is the probability that
    an emitter in state `i` at frame `t` is in state `j` at frame `t + 1`.

    Args:
        mu_pc, sigma_pc: Lognormal parameters (scale and shape) for the
            brightness distribution.
        brightness_quantile: Quantile vector defining the brightness states.
        prob_photo_bleach: Probability that an emitter enters the absorbing
            bleached state over `numb_photo_bleach` frames.
        numb_photo_bleach: FIXED reference-frame count (100), not the clip's
            frame count. It sets a per-frame bleach probability that is
            constant across clip durations; do not set it to the movie
            frame count.
        delta_frame: Time between frames in seconds.
        lambda_rate: Base rate for inter-state transitions.
        kappa_penalty: Dimensionless locality of the brightness-distance penalty,
            in units of the brightness standard deviation (fixed 1.0: the rate
            e-folds per one `sigma_bright` of brightness change).
        verbose: If True, print the generator and stochastic matrices.

    Returns:
        (Q, P) — both arrays of shape `(n_states, n_states)`.
    """
    brightness = np.round(
        lognorm.ppf(q=brightness_quantile, s=sigma_pc, loc=0, scale=mu_pc), 0,
    )
    # Photon-space standard deviation of the brightness lognormal; the penalty
    # below acts on brightness change measured in units of this spread.
    sigma_bright = lognorm.std(s=sigma_pc, loc=0, scale=mu_pc)
    numb_states = len(brightness)
    Q = np.zeros((numb_states, numb_states))

    # Photobleaching rate (continuous-time rate from the discrete-time prob over numb_photo_bleach frames).
    # numb_photo_bleach = 100 is a FIXED calibration constant (the reference-frame
    # count over which prob_photo_bleach accrues), not the clip's frame count.
    # Pinning it to 100 makes the per-frame bleach probability prob_1 constant
    # across clip durations, so a longer clip simply accumulates more bleaching
    # through repeated application of the same per-frame transition.
    # See the photobleaching model in PROJECT_CONTEXT.md.
    prob_1 = 1 - np.power((1 - prob_photo_bleach), 1 / numb_photo_bleach)
    epsilon_rate = -np.log(1 - prob_1) / delta_frame

    # Off-diagonal entries:
    #   Q[i, 0] = epsilon_rate (bleaching) for i in [1, numb_states - 1]
    #   Q[i, j] = lambda_rate * exp(-kappa_penalty * |brightness[i] - brightness[j]| / sigma_bright) for i, j in [1, ...], i != j
    for i in range(1, numb_states):
        Q[i, 0] = epsilon_rate
        j_indices = range(1, numb_states)
        distances = np.abs(brightness[i] - brightness[list(j_indices)])
        Q[i, list(j_indices)] = lambda_rate * np.exp(-kappa_penalty * distances / sigma_bright)
    # Reset the diagonal to 0 before computing the row-sum-negation diagonal.
    np.fill_diagonal(Q, 0)
    diag_indices = np.diag_indices(numb_states)
    Q[diag_indices] = -np.sum(Q, axis=1)

    # DTMC: P = exp(Q * delta_frame).
    P = expm(Q * delta_frame)

    if verbose:
        import textwrap
        default_print_opts = np.get_printoptions()
        np.set_printoptions(precision=3, suppress=True)
        print(textwrap.indent(f"Q (generator):\n{Q}\nP (stochastic):\n{P}", "  "))
        np.set_printoptions(**default_print_opts)

    return Q, P


# =============================================================================
# Private helpers
# =============================================================================

def _propagate_chain(states: np.ndarray,
                     P: np.ndarray,
                     rng: np.random.Generator) -> np.ndarray:
    """Fill in state trajectories step by step using a DTMC transition matrix.

    Modifies `states` in place: for each frame `i in [1, n_frames)` and each
    emitter `c`, samples the next state from `P[states[i-1, c], :]`.

    Vectorised across emitters: at each frame, builds the cumulative
    probability matrix `cumsum(P[prev_states, :], axis=1)` and samples each
    emitter's next state by comparing a single uniform draw against the cumsum.

    Args:
        states: Array of shape `(n_frames, n_emitters)`. The first row must
            already be filled with initial states.
        P: DTMC stochastic matrix of shape `(n_states, n_states)`.
        rng: NumPy random Generator.

    Returns:
        The modified `states` array (same reference).
    """
    n_frames, n_emitters = states.shape
    for i in range(1, n_frames):
        prev_states = states[i - 1, :]                            # shape (n_emitters,)
        cum_probs = np.cumsum(P[prev_states, :], axis=1)           # shape (n_emitters, n_states)
        u = rng.uniform(0, 1, size=n_emitters)[:, None]            # shape (n_emitters, 1)
        states[i, :] = (u < cum_probs).argmax(axis=1)
    return states


# =============================================================================
# Top-level orchestrator
# =============================================================================

_SCOPE_KEYS = ("gamma", "kappa_o", "kappa_b", "kappa_s", "kappa_q")


def draw_scope_camera(seed: Optional[int] = None) -> dict:
    """Draw one SCOPE camera vector (the marginalized camera nuisance), in physical units.

    The five camera keys (`gamma`, `kappa_o`, `kappa_b`, `kappa_s`, `kappa_q`) are not
    identifiable from the videos and are marginalized as the SCOPE camera nuisance
    (DETECTOR_WORKFLOW.md sec. 9.3, marginalized in both workflows). Their a-priori boxes
    are transient (no persisted artifact): one vector is drawn per simulation, uniform in
    log10 over each key's ``PRIOR_RANGE`` then exponentiated to physical space -- mirroring
    the detector DLI stage's SCOPE draw (``np.power(10, U(low, high))``). Returns a dict
    keyed by SCOPE key.
    """
    rng = np.random.default_rng(seed)
    camera = {}
    for key in _SCOPE_KEYS:
        low, high = PARAMETERIZATION_RAW[PARAMETER_RAW_FIND[key]]["PRIOR_RANGE"]
        camera[key] = float(10.0 ** rng.uniform(low, high))
    return camera


def simulate_dli(pro_tray_poses: np.ndarray,
                 theta: np.ndarray,
                 dimer_mask: Optional[np.ndarray] = None,
                 dimer_mule: float = 2.0,
                 seed: Optional[int] = None,
                 verbose: bool = False) -> np.ndarray:
    """Run the full DLI pipeline: particle poses + theta -> noisy video frames.

    Pipeline steps:
        1. Draw the SCOPE camera nuisance (gamma, kappa_o, kappa_b, kappa_s,
           kappa_q) from its transient a-priori box and construct the corrected
           Poisson-Gamma-Normal EMCCD detector.
        2. Read PSF parameters (mu_r, sigma_r, mu_pc, sigma_pc) and sample
           per-emitter PSF widths.
        3. Build the Gaussian PSF.
        4. Read photo-physics parameters (brightness quantiles, photobleaching,
           lambda_rate) and compute CTMC/DTMC matrices (flicker locality fixed,
           kappa_penalty=1).
        5. Sample brightness-state trajectories per emitter (plus an independent
           second-label trajectory for the dimer sum model).
        6. Set up the pixel grid and the optical-background photon floor (kappa_o).
        7. Render the noise-free per-pixel intensity from the particle poses,
           brightness states, and PSF (dimers combine by the sum model).
        8. Apply the corrected EMCCD noise chain (Poisson thinning -> stochastic
           Gamma EM register -> gain-independent Gaussian read noise -> bias).

    Args:
        pro_tray_poses: Particle coordinates of shape `(n_frames, n_emitters, 3)`,
            in nanometres (the simulation's native unit). Only the (x, y)
            components are used; z is dropped. NaN for absent particles.
        theta: Learnable-parameter vector (same shape and ordering as elsewhere
            in this package); used only for diagnostic/lookup purposes here — the
            photo-physics parameters are read fixed from `PARAMETERIZATION_RAW`, and
            the camera is drawn per simulation as the SCOPE nuisance (not from theta).
            Kept for signature parity with the RDS-side `theta`-passing convention.
        dimer_mask: Optional boolean mask of shape `(1, n_emitters, n_frames)`
            flagging dimer-state particles (B or C); their brightness is the SUM of
            two independent labels (see `compute_intensity`, `dimer_model="sum"`).
        dimer_mule: Merged-dimer brightness factor for the `multiply` alternative
            (default `2.0`; inert under the `sum` model -- see `compute_intensity`).
        seed: Optional RNG seed for all randomness in the pipeline (PSF widths,
            initial states, state trajectories, Poisson + Gaussian noise).
        verbose: If True, print intermediate diagnostic values.

    Returns:
        Frames of shape `(n_pix_x, n_pix_y, n_frames)`, in ADU.
    """
    nframes, nemitters, _ = pro_tray_poses.shape
    stem_geometry = PARAMETERS.simulation.stem
    pixel_size_nm = stem_geometry.pixel_size_nm
    root_size_px = stem_geometry.root_size_px

    # --- Camera params + EMCCD detector (SCOPE nuisance, drawn per simulation) ---
    # The 5 camera keys are the marginalized SCOPE camera nuisance (non-identifiable;
    # DETECTOR_WORKFLOW.md sec. 9.3): draw one vector per simulation from its transient
    # a-priori box (no persisted artifact). gamma = g/C is the only identifiable gain
    # quantity, so em_gain=gamma with electrons_per_adu=1.0 yields Gamma(N, gamma) exactly
    # (mirrors render_detector_video). Seed offset +2 keeps the camera stream independent
    # of the PSF/state draws (seed) and the dimer second-label draw (seed+1).
    camera = draw_scope_camera(seed=(None if seed is None else seed + 2))
    detector = EMCCD(
        quantum_efficiency=camera["kappa_q"],
        em_gain=camera["gamma"],
        electrons_per_adu=1.0,
        read_noise_adu=camera["kappa_s"],
        bias_adu=camera["kappa_b"],
    )

    # --- PSF params + per-emitter widths -----------------------------------
    mu_r = PARAMETERIZATION_RAW[PARAMETER_RAW_FIND["mu_r"]]["VALUE"]
    sigma_r = PARAMETERIZATION_RAW[PARAMETER_RAW_FIND["sigma_r"]]["VALUE"]
    mu_pc = PARAMETERIZATION_RAW[PARAMETER_RAW_FIND["mu_pc"]]["VALUE"]
    sigma_pc = PARAMETERIZATION_RAW[PARAMETER_RAW_FIND["sigma_pc"]]["VALUE"]
    per_emitter_sqrt2sigma = sample_psf_width(
        nemitters,
        PARAMETERS.simulation.dli.sqrt_2sigma_dist_label,
        keyword_args={"mu_r": mu_r, "sigma_r": sigma_r},
        seed=seed,
    )
    PSF = Gaussian(per_emitter_sqrt2sigma)

    # --- Photo-physics params + CTMC/DTMC matrices -------------------------
    brightness_quantile = np.asarray(
        PARAMETERIZATION_RAW[PARAMETER_RAW_FIND["brightness_quantile"]]["VALUE"]
    )
    delta_frame = PARAMETERS.simulation.timing.frame_time_seconds   # fixed cadence (50 Hz / 0.020 s), constant across runs -> safe to read off the global FrameConfig
    prob_photo_bleach = PARAMETERIZATION_RAW[PARAMETER_RAW_FIND["prob_photo_bleach"]]["VALUE"]
    numb_photo_bleach = PARAMETERIZATION_RAW[PARAMETER_RAW_FIND["numb_photo_bleach"]]["VALUE"]
    lambda_rate = PARAMETERIZATION_RAW[PARAMETER_RAW_FIND["lambda_rate"]]["VALUE"]
    _Q, P = compute_matrices(
        mu_pc=mu_pc, sigma_pc=sigma_pc,
        brightness_quantile=brightness_quantile,
        prob_photo_bleach=prob_photo_bleach,
        numb_photo_bleach=numb_photo_bleach,
        delta_frame=delta_frame,
        lambda_rate=lambda_rate,
        verbose=verbose,
    )

    # --- Brightness state trajectories -------------------------------------
    states = generate_state_trajectories(
        nframes=nframes,
        nemitters=nemitters,
        P=P,
        brightness_quantile=brightness_quantile,
        scale=mu_pc,
        shape=sigma_pc,
        loc=0,
        seed=seed,
    )
    brightness = compute_brightness(brightness_quantile, scale=mu_pc, shape=sigma_pc, loc=0)
    # For dimer_model="sum", each dimer's SECOND label needs its OWN independent flicker
    # trajectory (a dimer = two labels, brightness = X1 + X2). Independent seed (seed+1) so
    # it is not identical to the first label's; None stays non-deterministic. Mirrors
    # render_detector_video.
    dimer_states = generate_state_trajectories(
        nframes=nframes, nemitters=nemitters, P=P,
        brightness_quantile=brightness_quantile,
        scale=mu_pc, shape=sigma_pc, loc=0,
        seed=(None if seed is None else seed + 1),
    )

    # --- Pixel grid + optical background floor -----------------------------
    # kappa_o = optical background (incident photons): ONE scalar per video (the SCOPE
    # draw above), broadcast over all pixels/frames as the pre-PSF photon floor (added
    # before QE, amplified by gamma*kappa_q). Distinct from dark current (zero here,
    # handled inside EMCCD). Mirrors render_detector_video's optical_background.
    optical_background = np.full(
        shape=(root_size_px, root_size_px),
        fill_value=camera["kappa_o"],
        dtype=float,
    )
    xbounds = np.linspace(0, root_size_px, root_size_px + 1)
    ybounds = np.linspace(0, root_size_px, root_size_px + 1)

    # --- Convert particle positions to pixel-unit (x, y) tracks ------------
    # pro_tray_poses: (n_frames, n_emitters, 3) in nm.
    # Drop z, scale to pixel units, transpose to (n_frames, 2, n_emitters).
    tracks_pixels = pro_tray_poses[:, :, :2] / pixel_size_nm
    tracks_pixels = np.transpose(tracks_pixels, (0, 2, 1)).astype(np.float64)

    # --- Noise-free intensity image ----------------------------------------
    intensity = compute_intensity(
        tracks=tracks_pixels,
        states=states,
        brightness=brightness,
        background_photons=optical_background,
        xbounds=xbounds,
        ybounds=ybounds,
        PSF=PSF,
        dimer_mask=dimer_mask,
        dimer_mule=dimer_mule,
        # Dimer brightness is the SUM of two independent labels (mean 2*E[X], lighter tail
        # than a rigid doubling; DETECTOR_WORKFLOW.md sec. 6.4). dimer_states is the second
        # label's independent flicker trajectory. dimer_mule (multiply model) is retained as
        # a sensitivity alternative but is inert under sum.
        dimer_model="sum",
        dimer_states=dimer_states,
    )

    # --- Add EMCCD noise (Poisson + Gaussian readout) ----------------------
    frames = generate_frames(intensity, detector, seed=seed)
    return frames

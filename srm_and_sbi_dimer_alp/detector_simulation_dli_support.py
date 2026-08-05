"""Detector DLI forward model (adapted): render videos with imaging drawn from θ.

Part of the Detector calibration workflow (see the implementation plan in DETECTOR_WORKFLOW.md). The
Detector infers the imaging parameters, so this module renders synthetic videos
with the imaging parameters taken from the sampled θ rather than the fixed
parameter table. It reuses the canonical DLI *building blocks* by import — the
EMCCD detector, the Gaussian PSF and its width sampler, the brightness state
machine (`compute_matrices` / `generate_state_trajectories` / `compute_brightness`),
the intensity renderer (`compute_intensity`), and the noise stage
(`generate_frames`) — re-expressing only the thin orchestration that wires them
together (the same sequence as the canonical `simulate_dli`, which is left
untouched).

Parameter sourcing:
  - the 11 imaging parameters come from ``imaging_physical`` (det.DETECTOR_IMAGING
    order) in physical space: the 6 learnable inference targets (`mu_r`, `sigma_r`,
    `mu_pc`, `sigma_pc`, `prob_photo_bleach`, `lambda_rate`) drawn as the training θ,
    and the 5 SCOPE camera parameters (`gamma`, `kappa_o`, `kappa_b`, `kappa_s`,
    `kappa_q`) drawn as the marginalized camera nuisance;
  - the nominal gain/conversion `kappa_g`/`kappa_c` (spec metadata for the
    `gamma = g/C` drift check), `brightness_quantile`, `numb_photo_bleach`, and
    `dimer_mule` (=2; inert under the default `sum` dimer model) are fixed,
    read from the Detector parameter table;
  - `delta_frame` is the fixed camera cadence (`PARAMETERS.simulation.timing`),
    matching the canonical pipeline's source.
The canonical building blocks are reused as-is, except the shared `compute_intensity`
photon-floor argument, renamed `darkcounts` -> `background_photons` for clarity
(dark current is unmodeled here; the detector fills this floor with the optical
background `kappa_o`).
"""

import numpy as np

from . import detector_parameterization as det
from .parameterization import PARAMETERS
from .simulation_dli_support import (  # reused unchanged (building blocks)
    EMCCD,
    Gaussian,
    compute_brightness,
    compute_intensity,
    compute_matrices,
    generate_frames,
    generate_state_trajectories,
    sample_psf_width,
)

# The full 11-key imaging vector, in det.DETECTOR_IMAGING order: the 6 learnable
# inference targets followed by the 5 SCOPE camera-nuisance parameters. render reads
# by key, so this is the fixed contract the DLI stage assembles imaging_physical to.
_IMAGING_KEYS = det.DETECTOR_IMAGING_KEYS


def _fixed(key: str):
    """Physical VALUE of a fixed Detector parameter (read directly from the table)."""
    return det.DETECTOR_PARAMETERIZATION_RAW[det.DETECTOR_RAW_FIND[key]]["VALUE"]


def render_detector_video(pro_tray_poses: np.ndarray,
                          imaging_physical: np.ndarray,
                          dimer_mask=None,
                          dimer_model: str = "sum",
                          seed=None,
                          verbose: bool = False) -> np.ndarray:
    """Render one video from particle poses and a physical imaging-θ vector.

    Mirrors the canonical `simulate_dli` pipeline step-for-step, but sources the
    imaging parameters from ``imaging_physical`` — the 11 imaging parameters in
    ``DETECTOR_IMAGING`` order (6 learnable targets + 5 SCOPE camera nuisance), in
    physical space — instead of the fixed parameter table. The renderer reads each
    value by key, so it is agnostic to whether a parameter was drawn as a learnable
    target or a nuisance. Reuses the canonical building blocks unchanged.

    Args:
        pro_tray_poses: particle coordinates ``(n_frames, n_emitters, 3)`` in nm
            (only x, y are used; z dropped); NaN for absent particles.
        imaging_physical: physical values of the 11 imaging parameters,
            in ``DETECTOR_IMAGING`` order.
        dimer_mask: optional boolean mask ``(1, n_emitters, n_frames)`` flagging
            dimer-state emitters; their brightness is combined per ``dimer_model``.
        dimer_model: "sum" (default) adds an independent second-label draw -- a dimer is
            two independent labels whose photon counts add (sum of two monomers: same
            mean, lighter tail than doubling; an n-mer's brightness is the n-fold
            convolution of the monomer, Mutch et al. 2007). "multiply" scales one draw by
            ``dimer_mule`` (heavier tail; retained as an option).
        seed: optional RNG seed for PSF widths, states, and EMCCD noise.
        verbose: forwarded to ``compute_matrices``.

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

    # --- Camera params + EMCCD detector (imaging from θ) -------------------
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

    # --- PSF params + per-emitter widths (mu_r, sigma_r from θ) ------------
    per_emitter_sqrt2sigma = sample_psf_width(
        nemitters,
        PARAMETERS.simulation.dli.sqrt_2sigma_dist_label,
        keyword_args={"mu_r": img["mu_r"], "sigma_r": img["sigma_r"]},
        seed=seed,
    )
    PSF = Gaussian(per_emitter_sqrt2sigma)

    # --- Photo-physics + CTMC/DTMC matrices --------------------------------
    # mu_pc, sigma_pc, prob_photo_bleach, lambda_rate from θ; flicker locality (kappa_penalty) derived from the brightness scale.
    # brightness_quantile + numb_photo_bleach fixed; delta_frame = fixed cadence.
    mu_pc = img["mu_pc"]
    sigma_pc = img["sigma_pc"]
    brightness_quantile = np.asarray(_fixed("brightness_quantile"))
    delta_frame = PARAMETERS.simulation.timing.frame_time_seconds
    _Q, P = compute_matrices(
        mu_pc=mu_pc, sigma_pc=sigma_pc,
        brightness_quantile=brightness_quantile,
        prob_photo_bleach=img["prob_photo_bleach"],
        numb_photo_bleach=_fixed("numb_photo_bleach"),
        delta_frame=delta_frame,
        lambda_rate=img["lambda_rate"],
        verbose=verbose,
    )

    # --- Brightness state trajectories -------------------------------------
    states = generate_state_trajectories(
        nframes=nframes, nemitters=nemitters, P=P,
        brightness_quantile=brightness_quantile,
        scale=mu_pc, shape=sigma_pc, loc=0, seed=seed,
    )
    brightness = compute_brightness(brightness_quantile, scale=mu_pc, shape=sigma_pc, loc=0)
    # For dimer_model="sum", each dimer's SECOND label needs its OWN independent flicker
    # trajectory (a dimer = two labels, brightness = X1 + X2). Independent seed so it is not
    # identical to the first label's; None stays non-deterministic.
    dimer_states = None
    if dimer_model == "sum":
        dimer_states = generate_state_trajectories(
            nframes=nframes, nemitters=nemitters, P=P,
            brightness_quantile=brightness_quantile,
            scale=mu_pc, shape=sigma_pc, loc=0,
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
        tracks=tracks_pixels, states=states, brightness=brightness,
        background_photons=optical_background, xbounds=xbounds, ybounds=ybounds, PSF=PSF,
        dimer_mask=dimer_mask, dimer_mule=_fixed("dimer_mule"),
        dimer_model=dimer_model, dimer_states=dimer_states,
    )
    frames = generate_frames(intensity, detector, seed=seed)
    return frames

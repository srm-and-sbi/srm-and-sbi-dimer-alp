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
  - the 11 learnable imaging parameters (`kappa_c/o/g/v`, `mu_r`, `sigma_r`,
    `mu_pc`, `sigma_pc`, `prob_photo_bleach`, `lambda_rate`, `gamma_penalty`)
    come from the sampled θ in physical space;
  - `brightness_quantile`, `numb_photo_bleach`, and `dimer_mule` (=√2) are fixed,
    read from the Detector parameter table;
  - `delta_frame` is the fixed camera cadence (`PARAMETERS.simulation.timing`),
    matching the canonical pipeline's source.
Nothing in the canonical modules is modified.
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

# The 11 learnable imaging parameters, in DETECTOR_PARAMETERIZATION order.
_IMAGING_KEYS = [entry["KEY"] for entry in det.DETECTOR_PARAMETERIZATION]


def _fixed(key: str):
    """Physical VALUE of a fixed Detector parameter (read directly from the table)."""
    return det.DETECTOR_PARAMETERIZATION_RAW[det.DETECTOR_RAW_FIND[key]]["VALUE"]


def render_detector_video(pro_tray_poses: np.ndarray,
                          imaging_physical: np.ndarray,
                          dimer_mask=None,
                          seed=None,
                          verbose: bool = False) -> np.ndarray:
    """Render one video from particle poses and a physical imaging-θ vector.

    Mirrors the canonical `simulate_dli` pipeline step-for-step, but sources the
    imaging parameters from ``imaging_physical`` — the 11 learnable imaging
    parameters in ``DETECTOR_PARAMETERIZATION`` order, in physical space — instead
    of the fixed parameter table. Reuses the canonical building blocks unchanged.

    Args:
        pro_tray_poses: particle coordinates ``(n_frames, n_emitters, 3)`` in nm
            (only x, y are used; z dropped); NaN for absent particles.
        imaging_physical: physical values of the 11 learnable imaging parameters,
            in ``DETECTOR_PARAMETERIZATION`` order.
        dimer_mask: optional boolean mask ``(1, n_emitters, n_frames)`` flagging
            dimer-state emitters; their brightness is multiplied by ``dimer_mule``.
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

    # --- Camera params + EMCCD detector (kappa_* from θ) -------------------
    camera_conversion = img["kappa_c"]
    camera_offset = img["kappa_o"] * camera_conversion
    camera_gain = img["kappa_g"]
    camera_variance = (img["kappa_v"] * camera_conversion) ** 2
    detector = EMCCD(offset=camera_offset, gain=camera_gain, variance=camera_variance)

    # --- PSF params + per-emitter widths (mu_r, sigma_r from θ) ------------
    per_emitter_sqrt2sigma = sample_psf_width(
        nemitters,
        PARAMETERS.simulation.dli.sqrt_2sigma_dist_label,
        keyword_args={"mu_r": img["mu_r"], "sigma_r": img["sigma_r"]},
        seed=seed,
    )
    PSF = Gaussian(per_emitter_sqrt2sigma)

    # --- Photo-physics + CTMC/DTMC matrices --------------------------------
    # mu_pc, sigma_pc, prob_photo_bleach, lambda_rate, gamma_penalty from θ;
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
        gamma_penalty=img["gamma_penalty"],
        verbose=verbose,
    )

    # --- Brightness state trajectories -------------------------------------
    states = generate_state_trajectories(
        nframes=nframes, nemitters=nemitters, P=P,
        brightness_quantile=brightness_quantile,
        scale=mu_pc, shape=sigma_pc, loc=0, seed=seed,
    )
    brightness = compute_brightness(brightness_quantile, scale=mu_pc, shape=sigma_pc, loc=0)

    # --- Pixel grid + dark counts ------------------------------------------
    darkcounts = np.full(
        shape=(root_size_px, root_size_px),
        fill_value=PARAMETERS.simulation.dli.darkcounts,
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
        darkcounts=darkcounts, xbounds=xbounds, ybounds=ybounds, PSF=PSF,
        dimer_mask=dimer_mask, dimer_mule=_fixed("dimer_mule"),
    )
    frames = generate_frames(intensity, detector, seed=seed)
    return frames

"""Detector RDS forward model (adapted): diffusion-only three-species simulation.

Part of the Detector calibration workflow (see the implementation plan in DETECTOR_WORKFLOW.md). The
Detector infers the imaging parameters with the reaction-diffusion parameters
marginalized as a nuisance, so this module runs the reaction-diffusion simulator
in its diffusion-only mode and sources the reaction-diffusion parameters from the
Detector's RDS nuisance rather than the canonical learnable θ.

It reuses the canonical building blocks **by import** — `build_system` (with
`pure_diffusion=True`) and `build_simulation` — and adds only the parameter
sourcing: it draws the six nuisance parameters (three species counts + three
diffusion coefficients) and assembles them into the
length-`len(PARAMETERIZATION)` vector those building blocks expect, in the
canonical learnable order. The four reaction-rate slots are filled with the
canonical default `VALUE`; `build_system` reads them but never uses them when
`pure_diffusion=True` (the reaction registrations are skipped). Nothing in the
canonical modules is modified. Trajectory extraction (`extract_trajectory_poses`)
is reused directly from `simulation_rds_support` by the entry script.
"""

import numpy as np

from . import detector_parameterization as det
from .parameterization import PARAMETERIZATION, parameter_find
from .simulation_rds_support import build_simulation, build_system  # reused unchanged


def draw_nuisance_physical(n=None, device: str = "cpu") -> np.ndarray:
    """Draw RDS-nuisance sample(s) from the Detector nuisance prior, in physical space.

    Returns shape ``(6,)`` for ``n=None`` or ``(n, 6)`` for a batch, ordered as
    ``detector_parameterization.DETECTOR_NUISANCE`` (three counts + three
    diffusion coefficients). Every nuisance row is log10 (``detector_parameterization`` enforces this at
    import), so the physical value is ``LOG_BASE ** draw``.
    """
    prior = det.build_nuisance_prior(device)
    sample_shape = () if n is None else (int(n),)
    sample_log = prior.sample(sample_shape).cpu().numpy()
    log_bases = np.array([e["LOG_BASE"] for e in det.DETECTOR_NUISANCE], dtype=float)
    return np.power(log_bases, sample_log)


def nuisance_theta_to_canonical(nuisance_physical: np.ndarray) -> np.ndarray:
    """Assemble a canonical-order learnable θ from a single physical nuisance draw.

    ``nuisance_physical`` holds the physical values of the six nuisance
    parameters in ``DETECTOR_NUISANCE`` order. They are written into their
    canonical learnable positions (looked up by ``parameter_find``); the
    remaining reaction-rate positions keep the canonical default ``VALUE`` — a
    finite placeholder that ``build_system`` reads but never uses under
    ``pure_diffusion=True``.
    """
    nuisance_keys = [e["KEY"] for e in det.DETECTOR_NUISANCE]
    nuisance_physical = np.asarray(nuisance_physical, dtype=float)
    if nuisance_physical.shape != (len(nuisance_keys),):
        raise ValueError(
            f"nuisance_physical has shape {nuisance_physical.shape}; expected "
            f"({len(nuisance_keys)},) for keys {nuisance_keys}."
        )
    theta = np.array([p["VALUE"] for p in PARAMETERIZATION], dtype=float)
    for key, value in zip(nuisance_keys, nuisance_physical):
        theta[parameter_find(key)] = value
    return theta


def build_detector_rds_simulation(nuisance_physical: np.ndarray,
                                  seed=None, verbose: bool = False):
    """Build the diffusion-only Detector RDS simulation for one nuisance draw.

    Assembles the canonical θ from the nuisance draw, then reuses the canonical
    `build_system(pure_diffusion=True)` and `build_simulation` unchanged. Returns
    the `readdy.Simulation` (ready for the entry script to configure an output
    file and run) together with the assembled θ (for provenance).
    """
    theta = nuisance_theta_to_canonical(nuisance_physical)
    stem = build_system(theta, pure_diffusion=True, verbose=verbose)
    smut = build_simulation(stem, theta, seed=seed, verbose=verbose)
    return smut, theta

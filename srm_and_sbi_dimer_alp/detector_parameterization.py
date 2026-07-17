"""Detector-calibration parameter spec (value-based role scheme).

This module is the parameter contract for the Detector calibration workflow — a
special-situation entry point that infers the diffraction-limited-imaging (DLI)
model with the physics frozen to pure diffusion. It is deliberately DECOUPLED
from the canonical ``parameterization.py``: the detector-calibration system is
similar to but distinct from the production system, its parameter roles differ
(the imaging parameters are inferred here, fixed there), and its ranges differ
by design. Code duplication with the canonical module is intentional.

Value-based role scheme
-----------------------
The canonical scheme keys a parameter's role solely on ``PRIOR_RANGE`` (a tuple
means learnable, ``None`` means fixed). To express learnable, fixed, nuisance,
and (future) posterior-drawn parameters from one table, the role selector here
moves to the ``VALUE`` field via two reserved string sentinels:

    VALUE                 PRIOR_RANGE     role
    -------------------   -------------   ------------------------------------
    concrete (num/list)   (low, high)     learnable   (VALUE is the prior center)
    concrete (num/list)   None            fixed       (constant, read directly)
    "NUISANCE"            (low, high)     nuisance_spec   (inline BoxUniform)
    "NUISANCE"            None            nuisance_object (supplied distribution)
    "POSTERIOR"           None            posterior       (future multiround)
    "POSTERIOR"           (low, high)     undefined -> rejected at import

Three design constraints (enforced at import by ``_validate_table``):
  1. Role dispatch is sentinel-based (``VALUE in {NUISANCE, POSTERIOR}``); it
     never tests whether ``VALUE`` is numeric, because a list-valued fixed
     parameter (``brightness_quantile``) already exists and a numeric test would
     misclassify it.
  2. The learnable subset is ``VALUE-not-a-sentinel AND PRIOR_RANGE-not-None``.
     A nuisance-from-spec row also carries a range, so a ``PRIOR_RANGE is not
     None`` test alone (the canonical filter) would pull nuisance rows into the
     inference prior. That is the single most important port hazard (the production port to the canonical codebase).
  3. Log semantics are explicit: every ranged row is log10 (``LOG_FLAG=True``,
     ``LOG_BASE=10``), so a draw from a range lives in log10 space and maps to
     physical space by ``LOG_BASE ** draw`` (``to_physical``). Fixed values are
     already physical.

Public interface
----------------
    NUISANCE_SENTINEL, POSTERIOR_SENTINEL
    DETECTOR_PARAMETERIZATION_RAW   -- flat list of all entries (full spec)
    DETECTOR_PARAMETERIZATION       -- learnable subset (the inference prior / theta)
    DETECTOR_NUISANCE               -- nuisance subset (marginalized RDS biology)
    DETECTOR_RAW_FIND / DETECTOR_FIND / DETECTOR_NUISANCE_FIND  -- KEY -> index maps
    role_of(entry)                  -- value-based role of a single entry
    detector_find(key)              -- learnable-parameter index (for theta vectors)
    to_physical(draw, entry)        -- map a (log10) draw to physical space
    build_prior(device)             -- BoxUniform over the learnable imaging params
    build_nuisance_prior(device)    -- BoxUniform over the nuisance-from-spec RDS params
    theta_lower_bound / theta_upper_bound             -- learnable log10 bounds
    flag_out_of_bounds(theta, low, high)              -- flag/measure learnable values outside the prior box
    nuisance_lower_bound / nuisance_upper_bound       -- nuisance log10 bounds

Assembling the concrete argument vectors the theta-driven forward models
(``build_system``, ``simulate_dli``) consume is finalized against those call
sites in the Detector simulation scripts, where ``to_physical`` and
the subset/index maps here are the building blocks.
"""

import dataclasses

import numpy as np
import torch
from sbi.utils import BoxUniform

# Detector-generated data namespaces separately from canonical data by carrying
# this qualifier in the runtime prefix (a C28 stage token right after the iter),
# e.g. SRM_AND_SBI_DIMER_ALP_DETECTOR_5S_50FPS_Theta_Set_TASK_0_TRAIN.zarr.
DETECTOR_ALIAS_SUFFIX = "DETECTOR"

# Reserved VALUE sentinels that move a parameter's role off PRIOR_RANGE.
NUISANCE_SENTINEL = "NUISANCE"
POSTERIOR_SENTINEL = "POSTERIOR"
_SENTINELS = (NUISANCE_SENTINEL, POSTERIOR_SENTINEL)


# =============================================================================
# Detector parameter spec
# =============================================================================
#
# Fields per entry (a lean subset of the canonical schema; ROLE is computed from
# VALUE x PRIOR_RANGE, never stored):
#   KEY          unique identifier (str)
#   VALUE        prior center for a learnable row (VALUE = LOG_BASE**mid(range));
#                the physical constant for a fixed row (scalar or list);
#                a sentinel string for a nuisance/posterior row
#   PRIOR_RANGE  (low, high) log10 bounds for a ranged row; None otherwise
#   LOG_FLAG     True for every ranged row (log10); None for fixed rows
#   LOG_BASE     10 for every ranged row; None for fixed rows
#   UNIT         human-readable unit of the parameter as sampled
#   LABEL        LaTeX label for plotting
#
# The "op" comment on each learnable imaging row is the production operating
# point (the value used as a fixed constant in the canonical model); every
# learnable range brackets it, so calibration can only refine, never contradict
# by construction, that operating point (see the learnable imaging-parameter ranges in DETECTOR_WORKFLOW.md).

_DETECTOR_RAW_NESTED: dict[str, list[dict]] = {
    # ----- RDS nuisance: biology marginalized during detector calibration -----
    # Drawn per simulation from a restricted BoxUniform; fed to the diffusion-only
    # forward model. Ranges from the RDS-nuisance section of DETECTOR_WORKFLOW.md.
    'count': [
        {'KEY': 'count_alp', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (1.0, 2.5), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count', 'LABEL': r'$C_{A}$'},
        {'KEY': 'count_bet', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (1.0, 2.5), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count', 'LABEL': r'$C_{B}$'},
        {'KEY': 'count_chi', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (1.0, 2.5), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count', 'LABEL': r'$C_{C}$'},
    ],
    'diffusivity': [
        {'KEY': 'diffusivity_alp', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (-1.25, -0.25), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Square Micrometer Per Second', 'LABEL': r'$D_{A}$'},
        {'KEY': 'relative_diffusivity_bet', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (-0.625, -0.125), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Dimensionless', 'LABEL': r'$R_{B}$'},
        {'KEY': 'relative_diffusivity_chi', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (-2.0, -1.0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Dimensionless', 'LABEL': r'$R_{C}$'},
    ],
    # ----- Fixed geometry -----
    'geometry': [
        {'KEY': 'capture_radius', 'VALUE': 20, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Nanometer', 'LABEL': r'$\rho_{CAP}$'},
    ],
    # ----- Learnable imaging parameters (calibration targets) -----
    # Ranges from the learnable-imaging-parameter section of DETECTOR_WORKFLOW.md; VALUE = 10**mid(range) (center).
    'camera': [  # EMCCD detector
        {'KEY': 'kappa_c', 'VALUE': 10**0.5, 'PRIOR_RANGE': (0.0, 1.0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\kappa_{c}$'},
        {'KEY': 'kappa_o', 'VALUE': 10**2.5, 'PRIOR_RANGE': (2.0, 3.0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Photon', 'LABEL': r'$\kappa_{o}$'},
        {'KEY': 'kappa_g', 'VALUE': 10**2.0, 'PRIOR_RANGE': (1.5, 2.5), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\kappa_{g}$'},
        {'KEY': 'kappa_v', 'VALUE': 10**1.5, 'PRIOR_RANGE': (1.0, 2.0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Photon', 'LABEL': r'$\kappa_{v}$'},
    ],
    'psf': [  # PSF widths (mu_r, sigma_r) + emitter brightness (mu_pc, sigma_pc)
        {'KEY': 'mu_r', 'VALUE': 10**0.25, 'PRIOR_RANGE': (0.0, 0.5), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\mu_{r}$'},
        {'KEY': 'sigma_r', 'VALUE': 10**(-0.5), 'PRIOR_RANGE': (-1.0, 0.0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\sigma_{r}$'},
        {'KEY': 'mu_pc', 'VALUE': 10**2.25, 'PRIOR_RANGE': (1.75, 2.75), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\mu_{pc}$'},
        {'KEY': 'sigma_pc', 'VALUE': 10**(-0.375), 'PRIOR_RANGE': (-1.0, 0.25), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\sigma_{pc}$'},
    ],
    'transitivity': [  # emitter brightness state machine (CTMC generator + DTMC)
        {'KEY': 'brightness_quantile', 'VALUE': [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'LABEL': None},
        {'KEY': 'delta_frame', 'VALUE': 0.020, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Second', 'LABEL': r'$\delta_{f}$'},
        {'KEY': 'numb_photo_bleach', 'VALUE': 100, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'LABEL': r'$\psi_{pb}$'},
        {'KEY': 'dimer_mule', 'VALUE': 1.4142135623730951, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'LABEL': r'$\sqrt{2}$'},  # sqrt(2); mirrors PARAMETERS.simulation.dli.dimer_mule
        {'KEY': 'prob_photo_bleach', 'VALUE': 10**(-1.25), 'PRIOR_RANGE': (-2.0, -0.5), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\rho_{pb}$'},
        {'KEY': 'lambda_rate', 'VALUE': 10**(-0.5), 'PRIOR_RANGE': (-1.25, 0.25), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\lambda$'},              # see sec. 6.3
    ],
}


# =============================================================================
# Role resolution (value-based dispatch)
# =============================================================================

def role_of(entry: dict) -> str:
    """Value-based role of a single parameter entry.

    Returns one of 'learnable', 'fixed', 'nuisance_spec', 'nuisance_object',
    'posterior'. Dispatch is sentinel-based and never tests whether ``VALUE`` is
    numeric (constraint 1), so a list-valued fixed parameter is classified
    correctly. ``POSTERIOR`` with a range is undefined and raises.
    """
    value, prior_range = entry['VALUE'], entry['PRIOR_RANGE']
    is_sentinel = isinstance(value, str) and value in _SENTINELS
    if value == POSTERIOR_SENTINEL:
        if prior_range is not None:
            raise ValueError(
                f"parameter {entry['KEY']!r}: POSTERIOR with a PRIOR_RANGE is "
                f"undefined (a posterior draw carries its own support). Set "
                f"PRIOR_RANGE=None for a posterior-drawn parameter."
            )
        return 'posterior'
    if value == NUISANCE_SENTINEL:
        return 'nuisance_spec' if prior_range is not None else 'nuisance_object'
    # Concrete value: learnable iff it also carries a range (constraint 2).
    if not is_sentinel and prior_range is not None:
        return 'learnable'
    return 'fixed'


def _validate_table(raw: list[dict]) -> None:
    """Enforce the three design constraints at import time (fail-fast)."""
    seen = set()
    for entry in raw:
        key = entry['KEY']
        assert key not in seen, f"duplicate parameter KEY {key!r} in the detector table."
        seen.add(key)
        role = role_of(entry)  # raises on the undefined POSTERIOR+range cell
        prior_range = entry['PRIOR_RANGE']
        if prior_range is not None:
            # Constraint 3: every ranged row is log10.
            assert entry['LOG_FLAG'] is True and entry['LOG_BASE'] == 10, (
                f"parameter {key!r}: a ranged row must be log10 "
                f"(LOG_FLAG=True, LOG_BASE=10); got LOG_FLAG={entry['LOG_FLAG']!r}, "
                f"LOG_BASE={entry['LOG_BASE']!r}."
            )
            low, high = prior_range
            assert low < high, f"parameter {key!r}: PRIOR_RANGE low {low} !< high {high}."
            if role == 'learnable':
                # VALUE = LOG_BASE ** mid(range) invariant (linear-space center).
                center = entry['LOG_BASE'] ** ((low + high) / 2)
                assert abs(entry['VALUE'] - center) < 1e-9 * max(1.0, center), (
                    f"parameter {key!r}: learnable VALUE {entry['VALUE']} is not the "
                    f"prior center {center} (= {entry['LOG_BASE']}**{(low + high) / 2})."
                )
        else:
            # Fixed / nuisance-object / posterior rows carry no log metadata.
            assert entry['LOG_FLAG'] is None and entry['LOG_BASE'] is None, (
                f"parameter {key!r}: a non-ranged row must have LOG_FLAG=None, "
                f"LOG_BASE=None."
            )


# =============================================================================
# Flat list, subsets, and index maps
# =============================================================================

DETECTOR_PARAMETERIZATION_RAW: list[dict] = [
    entry for group in _DETECTOR_RAW_NESTED.values() for entry in group
]

_validate_table(DETECTOR_PARAMETERIZATION_RAW)

# Learnable subset (the inference prior + theta columns): VALUE-not-a-sentinel
# AND PRIOR_RANGE-not-None (constraint 2).
DETECTOR_PARAMETERIZATION: list[dict] = [
    entry for entry in DETECTOR_PARAMETERIZATION_RAW if role_of(entry) == 'learnable'
]

# Nuisance subset (RDS biology marginalized during calibration).
DETECTOR_NUISANCE: list[dict] = [
    entry for entry in DETECTOR_PARAMETERIZATION_RAW
    if role_of(entry) in ('nuisance_spec', 'nuisance_object')
]

DETECTOR_RAW_FIND: dict[str, int] = {
    entry['KEY']: index for index, entry in enumerate(DETECTOR_PARAMETERIZATION_RAW)
}
DETECTOR_FIND: dict[str, int] = {
    entry['KEY']: index for index, entry in enumerate(DETECTOR_PARAMETERIZATION)
}
DETECTOR_NUISANCE_FIND: dict[str, int] = {
    entry['KEY']: index for index, entry in enumerate(DETECTOR_NUISANCE)
}


# =============================================================================
# Helpers
# =============================================================================

def detector_find(key: str) -> int:
    """Index of a learnable imaging parameter by KEY (for theta-vector indexing)."""
    if key not in DETECTOR_FIND:
        raise KeyError(
            f"Parameter {key!r} is not a learnable detector parameter. "
            f"Learnable parameters: {list(DETECTOR_FIND.keys())}."
        )
    return DETECTOR_FIND[key]


def to_physical(draw, entry: dict):
    """Map a sampled ``draw`` for ``entry`` to physical space.

    A ranged row is sampled in log10 space, so its physical value is
    ``LOG_BASE ** draw``; a fixed row is already physical and returned as-is.
    Accepts scalars or tensors (``LOG_BASE ** draw`` broadcasts).
    """
    if entry['PRIOR_RANGE'] is None:
        return entry['VALUE']
    return entry['LOG_BASE'] ** draw


def theta_lower_bound() -> list[float]:
    """Lower bounds of the learnable imaging prior, in log10 space."""
    return [entry['PRIOR_RANGE'][0] for entry in DETECTOR_PARAMETERIZATION]


def theta_upper_bound() -> list[float]:
    """Upper bounds of the learnable imaging prior, in log10 space."""
    return [entry['PRIOR_RANGE'][1] for entry in DETECTOR_PARAMETERIZATION]


def flag_out_of_bounds(theta_log10, low=None, high=None):
    """Flag learnable-parameter values outside the prior box (log10 space).

    Args:
        theta_log10: array-like of log10 parameter values; the last axis is the
            parameter axis (length == the number of learnable imaging parameters).
        low, high: prior bounds in log10 space; default to ``theta_lower_bound()``
            / ``theta_upper_bound()``.

    Returns:
        ``(out_of_bounds, signed_margin)``, both ``numpy`` arrays shaped like
        ``theta_log10``. ``out_of_bounds`` is a boolean mask (True where a value is
        below ``low`` or above ``high``). ``signed_margin`` is the signed distance
        outside the box, in log10 units: negative below the lower bound, positive
        above the upper bound, and 0 inside — so its magnitude is how far
        out-of-prior a value sits. Used to flag — never silently clip — MAP
        estimates that drift past a prior edge (the seed-then-optimize step is
        unconstrained), symmetric to the Nuisance input-side clipping. Inputs are
        assumed finite (a MAP estimate always is); a NaN would be reported as
        out-of-bounds with a NaN margin.
    """
    theta = np.asarray(theta_log10, dtype=float)
    lo = np.asarray(theta_lower_bound() if low is None else low, dtype=float)
    hi = np.asarray(theta_upper_bound() if high is None else high, dtype=float)
    below = np.minimum(theta - lo, 0.0)   # < 0 only where theta < lo
    above = np.maximum(theta - hi, 0.0)   # > 0 only where theta > hi
    signed_margin = below + above          # at most one term is non-zero
    return signed_margin != 0.0, signed_margin


def nuisance_lower_bound() -> list[float]:
    """Lower bounds of the nuisance-from-spec RDS box, in log10 space."""
    return [entry['PRIOR_RANGE'][0] for entry in DETECTOR_NUISANCE
            if role_of(entry) == 'nuisance_spec']


def nuisance_upper_bound() -> list[float]:
    """Upper bounds of the nuisance-from-spec RDS box, in log10 space."""
    return [entry['PRIOR_RANGE'][1] for entry in DETECTOR_NUISANCE
            if role_of(entry) == 'nuisance_spec']


def build_prior(device: str = "cpu") -> BoxUniform:
    """BoxUniform log-uniform prior over the learnable imaging parameters.

    Sampled values are in log10 space; map to physical via ``to_physical`` (i.e.
    ``10 ** theta``), exactly as the canonical prior convention.
    """
    return BoxUniform(
        low=torch.tensor(theta_lower_bound()),
        high=torch.tensor(theta_upper_bound()),
        device=device,
    )


def build_nuisance_prior(device: str = "cpu") -> BoxUniform:
    """BoxUniform over the nuisance-from-spec RDS box (log10 space).

    Covers only the inline (from-spec) nuisance rows. The supplied-distribution
    case ('nuisance_object') is served by the Nuisance artifact; this
    builder raises if any nuisance row lacks a range.
    """
    objects = [entry['KEY'] for entry in DETECTOR_NUISANCE
               if role_of(entry) == 'nuisance_object']
    if objects:
        raise ValueError(
            f"build_nuisance_prior only handles nuisance-from-spec rows; "
            f"{objects} are nuisance-from-object (use the Nuisance artifact)."
        )
    return BoxUniform(
        low=torch.tensor(nuisance_lower_bound()),
        high=torch.tensor(nuisance_upper_bound()),
        device=device,
    )


def detector_paths(canonical_paths):
    """Return a copy of the canonical `Paths` whose `project_alias` carries the
    Detector qualifier, so every canonical path pattern namespaces Detector data
    separately (e.g. `SRM_AND_SBI_DIMER_ALP_DETECTOR_<timing>_...`).

    Takes the canonical `Paths` as an argument (rather than importing the machine
    profile) so this module stays decoupled from `parameterization`/`PARAMETERS`.
    """
    return dataclasses.replace(
        canonical_paths,
        project_alias=f"{canonical_paths.project_alias}_{DETECTOR_ALIAS_SUFFIX}",
    )

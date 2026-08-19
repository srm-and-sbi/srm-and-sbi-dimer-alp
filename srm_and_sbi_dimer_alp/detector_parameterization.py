"""Detector-calibration parameter spec (value-based role scheme).

This module is the parameter contract for the Detector calibration workflow — a
special-situation entry point that infers the diffraction-limited-imaging (DLI)
model with the physics frozen to pure diffusion. It is deliberately DECOUPLED
from the canonical ``parameterization.py``: the detector-calibration system is
similar to but distinct from the production system, its parameter roles differ
(the imaging parameters are inferred here, marginalized as a nuisance there), and its ranges differ
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

Camera parameters and the SCOPE nuisance
----------------------------------------
The five EMCCD camera parameters (``gamma``, ``kappa_o``, ``kappa_b``, ``kappa_s``,
``kappa_q``) are not identifiable from the videos and are marginalized as the SCOPE
camera nuisance rather than inferred (DETECTOR_WORKFLOW.md sec. 9.3): the videos
identify the gain and conversion only through their ratio ``gamma = kappa_g / kappa_c``,
and only the product ``gamma * kappa_q`` sets the amplitude, so inferring these
splits the brightness amplitude with ``mu_pc`` instead of constraining it. They are
drawn from their a-priori boxes at the DLI stage and recorded as ``Nuisance_SCOPE``;
the EM gain ``kappa_g`` and conversion ``kappa_c`` are retained as FIXED spec metadata
(from the MET acquisition configuration; audit-pending) so the drawn ``gamma`` can be
checked against the nominal ratio ``kappa_g / kappa_c`` (a drift check). Per-row
provenance is in each entry's ``NOTE``; the full noise model is in
REFERENCE_EMCCD_NOISE_MODEL.md.

Public interface
----------------
    NUISANCE_SENTINEL, POSTERIOR_SENTINEL
    DETECTOR_PARAMETERIZATION_RAW   -- flat list of all entries (full spec)
    DETECTOR_PARAMETERIZATION       -- learnable subset (the inference prior / theta)
    DETECTOR_PARAMETER_KEYS         -- ordered learnable keys (the theta schema; load-guard)
    DETECTOR_NUISANCE               -- RDS biology nuisance subset (drawn at the RDS stage)
    DETECTOR_NUISANCE_SCOPE         -- SCOPE camera nuisance subset (drawn at the DLI stage)
    DETECTOR_IMAGING                -- full imaging vector: learnable + SCOPE (the render contract)
    DETECTOR_IMAGING_KEYS / DETECTOR_SCOPE_KEYS   -- ordered imaging / SCOPE keys
    DETECTOR_RAW_FIND / DETECTOR_FIND / DETECTOR_NUISANCE_FIND  -- KEY -> index maps
    role_of(entry)                  -- value-based role of a single entry
    detector_find(key)              -- learnable-parameter index (for theta vectors)
    to_physical(draw, entry)        -- map a (log10) draw to physical space
    build_prior(device)             -- BoxUniform over the learnable imaging params
    build_nuisance_prior(device)    -- BoxUniform over the nuisance-from-spec RDS params
    theta_lower_bound / theta_upper_bound             -- learnable log10 bounds
    flag_out_of_bounds(theta, low, high)              -- flag/measure learnable values outside the prior box
    nuisance_lower_bound / nuisance_upper_bound       -- RDS nuisance log10 bounds
    scope_lower_bound / scope_upper_bound             -- SCOPE camera-nuisance log10 bounds

Assembling the concrete argument vectors the theta-driven forward models
(``build_system``, ``render_dli_video``) consume is finalized against those call
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
#   NOTE         optional free-text provenance/description (present where a
#                parameter needs explanation, e.g. the camera chain); ignored by
#                role dispatch and every builder
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
        {'KEY': 'count_alp', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (0.0, 2.5), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count', 'LABEL': r'$C_{A}$',
         'NOTE': 'Initial molecule count of species A (monomer), C_A. RDS biology marginalized as a nuisance during detector calibration; drawn from [1, 316] (log-uniform floor at a single emitter per species, covering sparse monomer-dominated fields).'},
        {'KEY': 'count_bet', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (0.0, 2.5), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count', 'LABEL': r'$C_{B}$',
         'NOTE': 'Initial molecule count of species B (mobile dimer), C_B. RDS nuisance; drawn from [1, 316] (log-uniform floor at a single emitter per species, so this species can be near-absent).'},
        {'KEY': 'count_chi', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (0.0, 2.5), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count', 'LABEL': r'$C_{C}$',
         'NOTE': 'Initial molecule count of species C (immobile dimer), C_C. RDS nuisance; drawn from [1, 316] (log-uniform floor at a single emitter per species, so this species can be near-absent).'},
    ],
    'diffusivity': [
        {'KEY': 'diffusivity_alp', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (-1.25, -0.25), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Square Micrometer Per Second', 'LABEL': r'$D_{A}$',
         'NOTE': 'Monomer (species A) diffusion coefficient D_A, absolute. Sets the diffusion scale the dimer species are specified relative to. RDS nuisance; drawn from [10^-1.25, 10^-0.25] um^2/s.'},
        {'KEY': 'relative_diffusivity_bet', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (-0.625, -0.125), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Dimensionless', 'LABEL': r'$R_{B}$',
         'NOTE': 'Mobile-dimer (species B) diffusion RELATIVE to D_A: R_B in (0,1], so D_B = R_B * D_A. RDS nuisance; drawn from [10^-0.625, 10^-0.125].'},
        {'KEY': 'relative_diffusivity_chi', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (-2.0, -1.0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Dimensionless', 'LABEL': r'$R_{C}$',
         'NOTE': 'Immobile-dimer (species C) diffusion RELATIVE to D_A: R_C, so D_C = R_C * D_A (much slower). RDS nuisance; drawn from [10^-2, 10^-1].'},
    ],
    # ----- Fixed geometry -----
    'geometry': [
        {'KEY': 'capture_radius', 'VALUE': 10, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Nanometer', 'LABEL': r'$\rho_{CAP}$',
         'NOTE': 'Smoluchowski reaction radius rho_CAP (nm) = particle_diameter_nm = 2*monomer_radius (center-to-center contact of two monomers). Drives kappa_ON = 4*pi*D_R*rho_CAP, the capture volume V_CAP ~ rho_CAP^3, and the fusion/fission distances. Inert under the diffusion-only detector model (reactions off); build_system derives the active value from PARAMETERS.simulation.stem.particle_diameter_nm.'},
    ],
    # ----- Learnable imaging parameters (calibration targets) -----
    # Ranges from the learnable-imaging-parameter section of DETECTOR_WORKFLOW.md; VALUE = 10**mid(range) (center).
    'camera': [  # EMCCD detector camera chain (REFERENCE_EMCCD_NOISE_MODEL.md): gamma, kappa_o, kappa_b, kappa_s, kappa_q marginalized as the SCOPE camera nuisance (non-identifiable; DETECTOR_WORKFLOW.md sec. 9.3); kappa_g, kappa_c fixed nominal spec metadata (gamma = kappa_g/kappa_c).
        {'KEY': 'gamma', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (1.62, 1.625), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'ADU Per Electron', 'LABEL': r'$\gamma$',
         'NOTE': 'Gain-conversion ratio gamma = kappa_g/kappa_c (ADU per photoelectron) -- the only gain quantity the videos identify. Marginalized as a SCOPE camera nuisance: inferring it splits the peak-ADU amplitude with mu_pc (only gamma*kappa_q is identifiable), so it is drawn from its a-priori box rather than treated as a calibration target (DETECTOR_WORKFLOW.md sec. 9.3). Config-exact reference: kappa_g/kappa_c = 200/4.78 = 41.84, identical across all MET cells (both Fab and InlB camera protocols); box [41.7, 42.2].'},
        {'KEY': 'kappa_o', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (1.455, 1.465), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Photon', 'LABEL': r'$\kappa_{o}$',
         'NOTE': 'Optical background offset: incident photons per pixel per frame, pre-gain, one scalar per movie; amplified by gamma*kappa_q to set the ADU floor. Marginalized as a SCOPE camera nuisance (non-identifiable; DETECTOR_WORKFLOW.md sec. 9.3). Reference = ThunderSTORM offset[photon] median, a condition-independent background: Fab 28.9 / InlB 28.6 (pooled 28.7). Narrow box [28.5, 29.2] around the measured value: a broad offset lets the gain*offset floor dominate the video-to-video variation (detector-embedding collapse).'},
        {'KEY': 'kappa_b', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (2.24, 2.25), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'ADU', 'LABEL': r'$\kappa_{b}$',
         'NOTE': 'Camera baseline: post-gain ADU constant, added last. Marginalized as a SCOPE camera nuisance (non-identifiable; DETECTOR_WORKFLOW.md sec. 9.3). Config-exact reference: MET configured baseline = 175, identical across all cells (both conditions); box [173.8, 177.8].'},
        {'KEY': 'kappa_s', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (1.02, 1.025), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'ADU', 'LABEL': r'$\kappa_{s}$',
         'NOTE': 'Read noise: post-register Gaussian sigma (ADU). Marginalized as a SCOPE camera nuisance: it does not dominate the SNR (the EM register amplifies signal above the read-noise floor) and is only weakly identifiable (DETECTOR_WORKFLOW.md sec. 9.3). Reference = camera datasheet ~10.5 ADU; box [10.5, 10.6] (weakly identifiable but pinned tight to the datasheet value, like the other camera constants).'},
        {'KEY': 'kappa_q', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (-0.05, -0.04), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\kappa_{q}$',
         'NOTE': 'Quantum efficiency, applied once in the Poisson step. Marginalized as a SCOPE camera nuisance: only the product gamma*kappa_q is identifiable from the videos (DETECTOR_WORKFLOW.md sec. 9.3). Config-exact reference: MET quantumEfficiency = 0.90, identical across all cells; box [0.89, 0.91].'},
        {'KEY': 'kappa_g', 'VALUE': 200, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'LABEL': r'$\kappa_{g}$',
         'NOTE': 'Nominal EM gain g from the MET acquisition config (ThunderSTORM camera protocol; audit-pending). Not inferred -- kept as spec metadata so the drawn gamma (SCOPE nuisance) can be checked against kappa_g/kappa_c (drift check).'},
        {'KEY': 'kappa_c', 'VALUE': 4.78, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Electron Per ADU', 'LABEL': r'$\kappa_{c}$',
         'NOTE': 'Nominal conversion C (e-/ADU) from the MET acquisition config (photons2ADU field; audit-pending). Not inferred -- kept as spec metadata for the gamma drift check. gamma = kappa_g / kappa_c.'},
    ],
    'psf': [  # PSF widths (mu_r, sigma_r) + emitter brightness (mu_pc, sigma_pc)
        {'KEY': 'mu_r', 'VALUE': 10**0.15, 'PRIOR_RANGE': (0.0, 0.3), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\mu_{r}$',
         'NOTE': 'Median of the per-emitter Gaussian PSF-width distribution (sqrt(2)*sigma, in pixels). Learnable; the model infers a distribution over the PSF width. Reference = monomer-control Fab median 1.36 (sqrt(2)*sigma[nm]/158); InlB 1.47 is dimer-broadened -- two labels within one diffraction spot fit wider -- so Fab is adopted (DETECTOR_WORKFLOW.md sec. 6.5 caveat 3). Prior [1.0, 2.0], excluding implausibly wide PSFs while staying general.'},
        {'KEY': 'sigma_r', 'VALUE': 10**(-0.625), 'PRIOR_RANGE': (-1.0, -0.25), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\sigma_{r}$',
         'NOTE': 'Log-spread of the per-emitter PSF-width distribution. Learnable; with mu_r it parameterizes the lognormal PSF-width population. The ThunderSTORM fitted spread (Fab 0.37 / InlB 0.42) is upper-biased by per-localization fit error (errors-in-variables, sec. 6.5 caveat 1); the fixed-imaging histogram check indicates the fit-corrected population spread is ~0.15. Prior broad-but-capped [0.10, 0.56]: broad enough to contain the fitted 0.37, capped so the narrow-spot tail does not manufacture unphysically bright pixels. See DETECTOR_WORKFLOW.md.'},
        {'KEY': 'mu_pc', 'VALUE': 10**2.375, 'PRIOR_RANGE': (2.0, 2.75), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\mu_{pc}$',
         'NOTE': 'Median of the per-emitter (monomer) brightness (photon-count) parent distribution. Learnable. Reference = monomer-control Fab median 386 photons; InlB 690 is NOT a wider monomer brightness but the sum of two co-located labels on a dimer (localization artifact, sec. 6.5 caveat 3), which the model builds as the sum of two draws from this monomer parent (sec. 6.4) -- so mu_pc is monomer-scoped and NOT widened to the dimer value. Prior [100, 562].'},
        {'KEY': 'sigma_pc', 'VALUE': 10**(-0.375), 'PRIOR_RANGE': (-0.75, 0.0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\sigma_{pc}$',
         'NOTE': 'Log-spread of the per-emitter (monomer) brightness parent distribution. Learnable; with mu_pc it sets the lognormal brightness population. The ThunderSTORM fitted spread (Fab 0.61 / InlB 0.55) is upper-biased by fit error (sec. 6.5 caveat 1); the fixed-imaging histogram check indicates ~0.5. Prior broad-but-capped [0.178, 1.0]: contains the fitted 0.61, capped to exclude the heavy-brightness-tail regime. See DETECTOR_WORKFLOW.md.'},
    ],
    'transitivity': [  # emitter brightness state machine (CTMC generator + DTMC)
        {'KEY': 'brightness_quantile', 'VALUE': [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'LABEL': None,
         'NOTE': 'Quantile grid defining the discrete emitter brightness states of the flicker state machine. Fixed hyperparameter (a modeling choice, not inferred).'},
        {'KEY': 'delta_frame', 'VALUE': 0.020, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Second', 'LABEL': r'$\delta_{f}$',
         'NOTE': 'Camera frame interval (s); 0.020 = 50 FPS. Fixed acquisition constant. Duration-general: n_frames is supplied per run, this is only the per-frame time.'},
        {'KEY': 'numb_photo_bleach', 'VALUE': 100, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'LABEL': r'$\psi_{pb}$',
         'NOTE': 'Reference frame window normalizing prob_photo_bleach (NOT the video length): p_video = 1 - (1 - prob_photo_bleach)^(n_frames/numb_photo_bleach). Fixed = 100.'},
        {'KEY': 'dimer_mule', 'VALUE': 2.0, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'LABEL': r'$m_{D}$',
         'NOTE': 'Merged-dimer brightness relative to a monomer. Physical picture: a dimer is two labels within one PSF, so single-emitter fitting sees ONE spot whose photons combine into a brighter detection. Two combination models implement this (see DETECTOR_WORKFLOW.md sec. 6.4). THIS detector workflow uses dimer_model="sum": the merged brightness is the SUM of two INDEPENDENT monomer draws, each carrying its own flicker/bleach trajectory -> mean ~2x a monomer with a lighter upper tail than a rigid doubling; this path does NOT read dimer_mule. dimer_mule is consumed ONLY by dimer_model="multiply", the retained sensitivity alternative (both DLI stages render dimers by the sum model via the shared render_dli_video), which instead scales a SINGLE monomer draw by this factor. As that multiply factor it is regime-dependent in [1,2]: 2.0 = two PERMANENTLY-ON labels (the MET always-on ATTO 647N case, corroborated by the ~2x InlB/Fab per-spot intensity ratio); sqrt(2) ~= 1.41 when only ~one label is visible on average (photoswitching dye -> time-averaged geometric mean GM(1x,2x) of bright/dark states, or ~50% labeling). Held Fixed here for parity with the canonical table (value inert under the sum model); a per-dataset photophysical setting, not a universal law. Mirrors PARAMETERS.simulation.dli.dimer_mule.'},
        {'KEY': 'prob_photo_bleach', 'VALUE': 10**(-1.25), 'PRIOR_RANGE': (-2.0, -0.5), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\rho_{pb}$',
         'NOTE': 'Probability an emitter enters the absorbing bleached state over numb_photo_bleach (100) frames. Learnable photophysics target; drawn from [10^-2, 10^-0.5].'},
        {'KEY': 'lambda_rate', 'VALUE': 10**0.5, 'PRIOR_RANGE': (0.0, 1.0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'LABEL': r'$\lambda$',
         'NOTE': 'Base rate of inter-state (flicker) transitions in the brightness CTMC generator: Q[i,j] = lambda_rate * exp(-|d_brightness| / sigma_bright); locality derived from the brightness scale. Learnable; drawn from [10^0, 10^1] = [1, 10]. Reference ~5, derived from the flicker correlation-time of the MET track intensity[photon] autocorrelation (tau_corr ~0.13 s; see DETECTOR_WORKFLOW.md sec. 6.3).'},
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
        if key in seen:
            raise ValueError(f"duplicate parameter KEY {key!r} in the detector table.")
        seen.add(key)
        role = role_of(entry)  # raises on the undefined POSTERIOR+range cell
        prior_range = entry['PRIOR_RANGE']
        if prior_range is not None:
            # Constraint 3: every ranged row is log10.
            if not (entry['LOG_FLAG'] is True and entry['LOG_BASE'] == 10):
                raise ValueError(
                    f"parameter {key!r}: a ranged row must be log10 "
                    f"(LOG_FLAG=True, LOG_BASE=10); got LOG_FLAG={entry['LOG_FLAG']!r}, "
                    f"LOG_BASE={entry['LOG_BASE']!r}."
                )
            low, high = prior_range
            if not low < high:
                raise ValueError(f"parameter {key!r}: PRIOR_RANGE low {low} !< high {high}.")
            if role == 'learnable':
                # VALUE = LOG_BASE ** mid(range) invariant (linear-space center).
                center = entry['LOG_BASE'] ** ((low + high) / 2)
                if not abs(entry['VALUE'] - center) < 1e-9 * max(1.0, center):
                    raise ValueError(
                        f"parameter {key!r}: learnable VALUE {entry['VALUE']} is not the "
                        f"prior center {center} (= {entry['LOG_BASE']}**{(low + high) / 2})."
                    )
        else:
            # Fixed / nuisance-object / posterior rows carry no log metadata.
            if not (entry['LOG_FLAG'] is None and entry['LOG_BASE'] is None):
                raise ValueError(
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

# Ordered learnable-parameter keys = the theta-vector schema. Pass to
# `artifacts.load_estimator(expected_parameter_keys=...)` / `assert_schema_compatible`
# to hard-reject a legacy estimator whose parameter schema differs (equal-length theta
# vectors would otherwise be misread column-for-column).
DETECTOR_PARAMETER_KEYS: list[str] = [entry['KEY'] for entry in DETECTOR_PARAMETERIZATION]

# The nuisance parameters form two blocks, consumed at different stages and
# recorded in separate Nuisance_<DOMAIN>_Theta_Set files (DETECTOR_WORKFLOW.md
# sec. 7 / 9.3): the RDS biology (drawn and used at the RDS stage) and the SCOPE
# camera (drawn and rendered at the DLI stage, recorded as Nuisance_SCOPE). They are
# grouped by the nested-dict category, so flipping the camera rows to a nuisance does
# not pull them into the RDS draw.
_RDS_NUISANCE_GROUPS = ('count', 'diffusivity')
_SCOPE_NUISANCE_GROUPS = ('camera',)

# RDS biology nuisance subset (marginalized during calibration; drawn at the RDS stage).
DETECTOR_NUISANCE: list[dict] = [
    entry for group in _RDS_NUISANCE_GROUPS for entry in _DETECTOR_RAW_NESTED[group]
    if role_of(entry) in ('nuisance_spec', 'nuisance_object')
]

# SCOPE camera nuisance subset (marginalized in both workflows; drawn at the DLI stage).
DETECTOR_NUISANCE_SCOPE: list[dict] = [
    entry for group in _SCOPE_NUISANCE_GROUPS for entry in _DETECTOR_RAW_NESTED[group]
    if role_of(entry) == 'nuisance_spec'
]

# The full imaging vector the DLI renderer consumes: the learnable inference targets
# followed by the SCOPE camera nuisance. Render reads by KEY, so this fixed order is
# the persistence/assembly convention shared by the DLI stage and its support module.
DETECTOR_IMAGING: list[dict] = DETECTOR_PARAMETERIZATION + DETECTOR_NUISANCE_SCOPE
DETECTOR_IMAGING_KEYS: list[str] = [entry['KEY'] for entry in DETECTOR_IMAGING]
DETECTOR_SCOPE_KEYS: list[str] = [entry['KEY'] for entry in DETECTOR_NUISANCE_SCOPE]

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
        unconstrained). Inputs are
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


def scope_lower_bound() -> list[float]:
    """Lower bounds of the SCOPE camera-nuisance box, in log10 space."""
    return [entry['PRIOR_RANGE'][0] for entry in DETECTOR_NUISANCE_SCOPE]


def scope_upper_bound() -> list[float]:
    """Upper bounds of the SCOPE camera-nuisance box, in log10 space."""
    return [entry['PRIOR_RANGE'][1] for entry in DETECTOR_NUISANCE_SCOPE]


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

    The RDS nuisance is declared entirely from-spec: an inline BoxUniform over the biology
    ranges, drawn on the fly during detector-calibration generation. There is no persisted
    RDS nuisance object -- only its draws persist, as the ``Nuisance_RDS_Theta_Set`` -- and it
    is distinct from the imaging-only ``NuisanceDLI`` artifact. The value-based scheme's
    supplied-distribution role ('nuisance_object') is not used by the RDS nuisance; this
    builder raises if any nuisance row lacks a range.
    """
    objects = [entry['KEY'] for entry in DETECTOR_NUISANCE
               if role_of(entry) == 'nuisance_object']
    if objects:
        raise ValueError(
            f"build_nuisance_prior only handles nuisance-from-spec rows; "
            f"{objects} are nuisance-from-object, which the RDS nuisance does not use "
            f"(declare them from-spec with a PRIOR_RANGE)."
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

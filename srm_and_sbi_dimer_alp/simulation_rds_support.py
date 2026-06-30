"""ReaDDy primitives for the reaction-diffusion simulation (RDS) stage.

This module wraps the ReaDDy 2 particle-based reaction-diffusion solver
(Hoffmann et al., 2019, "ReaDDy 2: Fast and flexible software framework
for interacting particle reaction dynamics", PLoS Comp Bio.
https://doi.org/10.1371/journal.pcbi.1006830) for the three-species DIMER
model used by this sibling.

Three molecular species:
    A = Monomer
    B = Mobile Dimer
    C = Immobile Dimer

Two reactions:
    A + A <-> B    (dimerization / dissociation; the forward rate is
                    parameterized relative to the diffusion-limited cap)
    B    <-> C    (immobilization / mobilization)

Functions:
    build_system(theta, ...)
        Builds the ReaDDy ReactionDiffusionSystem: registers species with
        their diffusion constants and adds the four reactions. Returns the
        configured system, ready to be wrapped in a Simulation.

    build_simulation(stem, theta, ...)
        Wraps the system in a Simulation, registers observables (per-frame
        particle positions, per-step reaction counts), and places initial
        particles uniformly in the box. Returns the runnable Simulation.

    extract_trajectory_poses(tray, ...)
        Reads a saved .h5 trajectory file and produces a dense
        (frame, particle, spatial-dim, species-rank) tensor of particle
        positions, optionally accompanied by a boolean dimer mask flagging
        which particles are in the B or C states at each frame.
"""

from typing import Optional, Tuple, Union

import numpy as np
import readdy

from .parameterization import (
    PARAMETERS,
    PARAMETERIZATION_RAW,
    PARAMETER_RAW_FIND,
    parameter_find,
)


# =============================================================================
# Reaction-diffusion system builder
# =============================================================================

def build_system(theta: np.ndarray,
                 particle_species_names: Optional[Tuple[str, ...]] = None,
                 verbose: bool = False) -> "readdy.ReactionDiffusionSystem":
    """Build a ReaDDy ReactionDiffusionSystem for the three-species DIMER model.

    Args:
        theta: Learnable-parameter values (physical units, exp-transformed from
            the log-space prior). 1D numpy array of length `len(PARAMETERIZATION)`
            (currently 7), in the order defined by `parameter_find`.
        particle_species_names: Optional override for the species names.
            Defaults to `PARAMETERS.simulation.rds.particle_species_names` —
            the standard ('A', 'B', 'C').
        verbose: If True, print diffusion constants and reaction rates to stdout.

    Returns:
        A configured `readdy.ReactionDiffusionSystem`, ready to be passed to
        `build_simulation`.

    Diffusion model:
        - The monomer diffusion `D_A` is the learnable parameter `diffusivity_alp`.
        - Dimer diffusions are specified RELATIVE to `D_A`:
              `D_B = R_B * D_A`,  `D_C = R_C * D_A`,
          where `R_B` and `R_C` are the learnable parameters
          `relative_diffusivity_bet` and `relative_diffusivity_chi`.
          The prior samples these ratios directly. Storing relative values
          guarantees `D_B` and `D_C` track the magnitude of `D_A` without
          requiring the prior to encode their joint dependence.

    Reaction model — dimerization:
        Dimerization (A + A -> B) is parameterized by `R_ON`, a dimensionless
        ratio in (0, 1] of the macroscopic rate to the diffusion-limited cap:

              kappa_ON = R_ON * kappa_ON_CAP
              kappa_ON_CAP = 4 * pi * D_R * rho_CAP      (Smoluchowski)

        with `D_R = 2 * D_A` (relative diffusion coefficient of two A particles)
        and `rho_CAP = capture_radius` (in micrometres, converted from nm).
        The microscopic ReaDDy rate is then:

              lamb_ON = kappa_ON / V_CAP
              V_CAP = (4/3) * pi * rho_CAP^3            (effective capture volume)

        Parameterizing via R_ON instead of an absolute kappa_ON guarantees
        the macroscopic rate never exceeds the physically attainable
        diffusion-limited rate, regardless of how the prior is sampled.

    Reaction model — other reactions:
        Dissociation (B -> A + A), immobilization (B -> C), and mobilization
        (C -> B) are parameterized by absolute macroscopic rates in counts
        per second. The corresponding microscopic rates are just the
        macroscopic rates with ReaDDy's per-second unit attached.

    Boundary conditions:
        [False, False, True] — the simulation box is open in xy (the
        observation plane) and periodic in z (the thin direction
        perpendicular to the membrane).
    """
    if particle_species_names is None:
        particle_species_names = PARAMETERS.simulation.rds.particle_species_names

    # --- Extract parameter values from theta -------------------------------
    diffusion_keys = ("diffusivity_alp", "relative_diffusivity_bet", "relative_diffusivity_chi")
    theta_diffusion = theta[[parameter_find(k) for k in diffusion_keys]]
    diffusion_rates = np.empty(3)
    diffusion_rates[0] = theta_diffusion[0]                       # D_A (monomer)
    diffusion_rates[1] = theta_diffusion[1] * theta_diffusion[0]  # D_B = R_B * D_A
    diffusion_rates[2] = theta_diffusion[2] * theta_diffusion[0]  # D_C = R_C * D_A

    capture_radius = PARAMETERIZATION_RAW[PARAMETER_RAW_FIND["capture_radius"]]["VALUE"]  # nm

    reaction_keys = ("relative_rate_dimerization", "rate_dissociation",
                     "rate_immobility", "rate_mobility")
    theta_reaction = theta[[parameter_find(k) for k in reaction_keys]]
    R_ON = theta_reaction[0]               # dimerization ratio (0, 1]; relative to diffusion-limited cap
    kappa_OFF = theta_reaction[1]          # dissociation rate (1/s)
    kappa_IMMOBILITY = theta_reaction[2]   # B -> C rate (1/s)
    kappa_MOBILITY = theta_reaction[3]     # C -> B rate (1/s)

    # --- Build the system --------------------------------------------------
    stem_geometry = PARAMETERS.simulation.stem
    stem = readdy.ReactionDiffusionSystem(
        box_size=stem_geometry.box_size,
        unit_system=stem_geometry.unit_dict,
    )
    stem.periodic_boundary_conditions = [False, False, True]

    # Add species with diffusion constants (in um^2/s, attached to ReaDDy units).
    for idx, species_name in enumerate(particle_species_names):
        diffusion_constant = (
            diffusion_rates[idx]
            * pow(readdy.units.micrometer, 2)
            / readdy.units.second
        )
        stem.add_species(name=species_name, diffusion_constant=diffusion_constant)

    # --- Diffusion-limited dimerization rate -------------------------------
    D_R = 2 * diffusion_rates[0]                          # um^2 / s
    rho_CAP_um = capture_radius * 1e-3                    # nm -> um
    kappa_ON_CAP = 4 * np.pi * D_R * rho_CAP_um           # um^3 / s
    kappa_ON = R_ON * kappa_ON_CAP                        # um^3 / s
    V_CAP = (4 / 3) * np.pi * pow(rho_CAP_um, 3)          # um^3
    lamb_ON = (kappa_ON / V_CAP) / readdy.units.second    # 1/s
    lamb_OFF = kappa_OFF / readdy.units.second
    lamb_IMMOBILITY = kappa_IMMOBILITY / readdy.units.second
    lamb_MOBILITY = kappa_MOBILITY / readdy.units.second

    # --- Add the four reactions --------------------------------------------
    # The four channels implement the reversible dimerization / mobilization
    # scheme: A + A <-> B (dimerization / dissociation) and B <-> C
    # (immobilization / mobilization). A=monomer, B=mobile dimer, C=immobile
    # dimer. See the DIMER reaction model in PROJECT_CONTEXT.md.

    # A + A -> B : forward dimerization (microscopic rate lamb_ON, derived from
    # R_ON relative to the diffusion-limited Smoluchowski cap).
    stem.reactions.add_fusion(
        name="A + A => B", type_from1="A", type_from2="A", type_to="B",
        rate=lamb_ON, educt_distance=capture_radius, weight1=0.5, weight2=0.5,
    )
    # B -> A + A : reverse dissociation (macroscopic rate kappa_OFF).
    stem.reactions.add_fission(
        name="B => A + A", type_from="B", type_to1="A", type_to2="A",
        rate=lamb_OFF, product_distance=capture_radius, weight1=0.5, weight2=0.5,
    )
    # B -> C : immobilization, mobile dimer becomes immobile (rate kappa_IMMOBILITY).
    stem.reactions.add_conversion(
        name="B => C", type_from="B", type_to="C", rate=lamb_IMMOBILITY,
    )
    # C -> B : mobilization, immobile dimer becomes mobile again (rate kappa_MOBILITY).
    stem.reactions.add_conversion(
        name="C => B", type_from="C", type_to="B", rate=lamb_MOBILITY,
    )

    if verbose:
        rates_by_species = dict(zip(particle_species_names, diffusion_rates))
        print(f"  Diffusion rates per species (um^2/s): {rates_by_species}")
        print("  Reaction rates:")
        print(f"    A+A -> B (dimerization):     macroscopic={kappa_ON:.6g} um^3/s "
              f"(cap={kappa_ON_CAP:.6g}, R_ON={R_ON:.6g})")
        print(f"    B   -> A+A (dissociation):   macroscopic={kappa_OFF:.6g} 1/s")
        print(f"    B   -> C (immobilization):   macroscopic={kappa_IMMOBILITY:.6g} 1/s")
        print(f"    C   -> B (mobilization):     macroscopic={kappa_MOBILITY:.6g} 1/s")

    return stem


# =============================================================================
# Simulation builder
# =============================================================================

def build_simulation(stem: "readdy.ReactionDiffusionSystem",
                     theta: np.ndarray,
                     particle_species_names: Optional[Tuple[str, ...]] = None,
                     seed: Optional[int] = None,
                     verbose: bool = False) -> "readdy.Simulation":
    """Wrap the ReactionDiffusionSystem in a Simulation, register observables,
    and place initial particles uniformly in the box.

    Args:
        stem: ReactionDiffusionSystem from `build_system`.
        theta: Learnable-parameter values (same vector as passed to `build_system`).
            Used to extract per-species initial particle counts.
        particle_species_names: Optional override; defaults to
            `PARAMETERS.simulation.rds.particle_species_names`.
        seed: RNG seed for initial particle placement. If None, the placement
            is non-deterministic (different on every call); pass an integer
            for reproducible runs. Note: this seed only controls the NumPy
            RNG for particle positions; ReaDDy's internal RNG (used for
            reaction events and diffusion steps during `simulation.run`)
            has its own seeding mechanism.
        verbose: If True, print the per-species and total initial particle counts.

    Returns:
        A `readdy.Simulation` with:
            - 'particles' observable at the per-frame stride
              (PARAMETERS.simulation.timing.steps_per_frame),
            - 'reaction_counts' observable at every step (stride=1),
            - initial particles placed uniformly in the box for each species.

    Implementation notes:
        - Particle counts are rounded from the continuous theta sample
          (which lives in log-space and was exp-transformed before being
          passed here) to int32.
        - Particles are placed uniformly in [-box/2, +box/2] in each
          spatial dimension, ignoring the periodic z boundary for initial
          placement.
        - The simulation runs on the CPU kernel.
    """
    if particle_species_names is None:
        particle_species_names = PARAMETERS.simulation.rds.particle_species_names

    count_keys = ("count_alp", "count_bet", "count_chi")
    theta_counts = np.round(theta[[parameter_find(k) for k in count_keys]]).astype(np.int32)

    smut = stem.simulation(kernel="CPU")
    smut.observe.particles(stride=PARAMETERS.simulation.timing.steps_per_frame)
    smut.observe.reaction_counts(stride=1)

    box_size = PARAMETERS.simulation.stem.box_size
    lo = [-box_size[0] / 2, -box_size[1] / 2, -box_size[2] / 2]
    hi = [box_size[0] / 2, box_size[1] / 2, box_size[2] / 2]

    # Single RNG controls placement across all species (reproducible if seed is set).
    rng = np.random.default_rng(seed)
    for idx, species_name in enumerate(particle_species_names):
        n_particles = int(theta_counts[idx])
        positions = rng.uniform(low=lo, high=hi, size=(n_particles, 3))
        smut.add_particles(type=species_name, positions=positions)

    if verbose:
        counts_by_species = dict(zip(particle_species_names, theta_counts.tolist()))
        total = int(np.sum(theta_counts))
        print(f"  Initial particle counts: {counts_by_species} (total = {total})")

    return smut


# =============================================================================
# Trajectory pose extraction
# =============================================================================

def extract_trajectory_poses(tray,
                             return_dimer_mask: bool = False,
                             verbose: bool = False
                             ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Read a saved ReaDDy .h5 trajectory and produce a dense pose tensor.

    Args:
        tray: A `readdy.Trajectory` object opened from a .h5 file.
        return_dimer_mask: If True, also return a boolean mask flagging
            particles that are in the B or C (dimer) states at each frame.
            Used by the DLI stage to apply a brightness multiplier to dimers.
        verbose: If True, print per-reaction counts and the output tensor shape.

    Returns:
        If `return_dimer_mask=False`:
            tray_poses: shape `(n_frames, n_particles, 3, n_species)`.
                Each entry is the (x, y, z) coordinate of a given particle in
                a given frame for a given species rank. Entries are `np.nan`
                when the particle does not exist as that species in that frame.
                Coordinates are shifted from box-centered (ReaDDy default) to
                box-anchored (origin at the corner): each component has
                `box_size[dim] / 2` added so all coordinates are non-negative.
        If `return_dimer_mask=True`:
            (tray_poses, dimer_mask) where `dimer_mask` has shape
            `(1, n_particles, n_frames)` and is True wherever the particle
            is in the B or C state in that frame.

    Implementation notes:
        - The "rank" axis of `tray_poses` is the species index assigned by
          ReaDDy (from `tray.particle_types`). For the standard species
          order ('A', 'B', 'C'), rank 0 is A, rank 1 is B, rank 2 is C.
        - The dimer mask flags ranks corresponding to species 'B' and 'C',
          derived dynamically from `tray.particle_types` (robust to species
          ordering changes).
        - The last frame in the ReaDDy trajectory is dropped (the
          `n_frames - 1` slice) because ReaDDy writes an extra final
          observable that doesn't correspond to a fully evolved frame.
    """
    if verbose:
        _, recs = tray.read_observable_reaction_counts()
        for rec_name, counts in recs["reactions"].items():
            print(f"    reaction {rec_name}: {np.count_nonzero(counts)} events")

    part_spans, ranks, souls, poses = tray.read_observable_particles()

    species_to_rank = tray.particle_types  # {species_name: rank_index}
    all_souls = np.unique(
        np.concatenate([s.reshape((1, -1)) for s in souls], axis=1).flatten()
    )
    soul_to_index = {soul: idx for idx, soul in enumerate(all_souls)}

    poses_shape = (part_spans.shape[0] - 1, all_souls.shape[0], 3, len(species_to_rank))
    poses_centred = np.full(shape=poses_shape, fill_value=np.nan)

    if return_dimer_mask:
        dimer_species_names = ("B", "C")
        dimer_ranks = {species_to_rank[name] for name in dimer_species_names
                       if name in species_to_rank}
        dimer_mask = np.zeros(
            shape=(1, all_souls.shape[0], part_spans.shape[0] - 1),
            dtype=bool,
        )
    else:
        dimer_mask = None

    # Drive the loop from the frames actually present in this trajectory
    # (poses_centred is sized to part_spans.shape[0] - 1). Using the data's own
    # length -- not the global PARAMETERS frame_count -- keeps extraction correct
    # for any simulation duration. The 2 s global default (100 frames) previously
    # truncated 5 s/10 s trajectories to their first 100 frames and overran sub-2 s ones.
    for span_index in range(poses_centred.shape[0]):
        frame_ranks = ranks[span_index]
        frame_souls = souls[span_index]
        frame_poses = poses[span_index]
        for part_index in range(frame_poses.shape[0]):
            rank = frame_ranks[part_index]
            soul_index = soul_to_index[frame_souls[part_index]]
            poses_centred[span_index, soul_index, :, rank] = frame_poses[part_index]
            if return_dimer_mask and rank in dimer_ranks:
                dimer_mask[0, soul_index, span_index] = True

    # Sanity guard: every frame must hold at least one real position. An entirely
    # NaN frame means the fill loop skipped it (a trajectory/frame-count mismatch),
    # which would silently emit a noise-only video. Fail loudly rather than ship
    # truncated poses.
    empty_frames = np.isnan(poses_centred).all(axis=(1, 2, 3))
    if empty_frames.any():
        raise ValueError(
            f"extract_trajectory_poses: {int(empty_frames.sum())} of "
            f"{poses_centred.shape[0]} frames are entirely NaN (first at index "
            f"{int(np.argmax(empty_frames))}); the per-frame loop did not populate "
            f"every frame. Refusing to emit truncated poses."
        )

    # Shift coordinates from box-centered to box-anchored (origin at corner).
    box_size = PARAMETERS.simulation.stem.box_size
    half_box = np.array([dim / 2 for dim in box_size]).reshape(1, 1, 3, 1)
    tray_poses = half_box + poses_centred

    if verbose:
        print(f"  tray_poses shape: {tray_poses.shape}")

    if return_dimer_mask:
        return tray_poses, dimer_mask
    return tray_poses

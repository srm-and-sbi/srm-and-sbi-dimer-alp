"""Sibling configuration: per-machine paths (loaded from machine_profiles.toml)
+ sibling-wide defaults + the rich parameter spec.

This module is the single source of truth for sibling configuration. Imported by
every entry-point script and most package modules. Validates the active machine
profile at import time (refuse-on-missing: import fails fast with a clear error
if the env var is absent or the config is incomplete).

Public interface:
    PARAMETERS                  -- module-level singleton; typed access:
                                   PARAMETERS.paths.*, PARAMETERS.simulation.*,
                                   PARAMETERS.inference.*, PARAMETERS.plotting.*
    PARAMETERIZATION_RAW        -- flat list of all parameter entries (full spec)
    PARAMETERIZATION            -- filtered list of learnable parameters (for the prior)
    PARAMETER_RAW_FIND          -- dict[KEY -> index in PARAMETERIZATION_RAW]
    PARAMETER_FIND              -- dict[KEY -> index in PARAMETERIZATION]
    parameter_find(key)         -- index lookup for learnable parameters
    build_prior(device)         -- construct the BoxUniform log-uniform prior
    theta_lower_bound()         -- lower bounds of the log-uniform prior
    theta_upper_bound()         -- upper bounds of the log-uniform prior
"""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
from sbi.utils import BoxUniform

# tomllib is stdlib in Python 3.11+ (this env uses 3.13); fall back to tomli on older Pythons.
try:
    import tomllib
except ImportError:  # Python 3.9 / 3.10
    import tomli as tomllib


# =============================================================================
# Machine profile loader
# =============================================================================

@dataclass(frozen=True)
class MachineProfile:
    """Per-machine configuration; loaded from machine_profiles.toml.

    Selected via the MACHINE_PROFILE environment variable. Refuse-on-missing
    is enforced at import time: if the env var is absent, the profile name is
    unknown, required keys are missing, or the named root directories don't
    exist on disk, an error with a clear remedy is raised. This forces the
    user to fix configuration problems before any simulation or training starts,
    rather than failing mid-run.
    """
    name: str
    running_mode: str             # "LOCAL" (prototyping) | "HPC" (production)
    script_bank_root: Path        # full path to Script_Bank/ folder
    data_bank_root: Path          # full path to Data_Bank/ folder (permanent / backed-up tier)
    compute_backend: str          # "GPU" | "CPU"
    gpu_device_index: Optional[int]
    num_workers: int
    # Optional second data tier for machines whose scratch space is large/fast
    # but impermanent (e.g. an HPC scratch filesystem that is auto-purged and not
    # backed up). When configured, regenerable TRAIN/TEST data is routed here;
    # everything to keep -- EVAL, posteriors, checkpoints, experimental data --
    # stays on data_bank_root. When left unset, the machine is single-tier and
    # every split uses data_bank_root (the behavior on single-filesystem machines).
    scratch_data_bank_root: Optional[Path] = None

    def root_for(self, split: str) -> Path:
        """Data-bank root for a generation ``split``.

        Regenerable ``TRAIN``/``TEST`` data goes to ``scratch_data_bank_root``
        when that tier is configured; ``EVAL`` -- like all non-split artifacts
        (posteriors, checkpoints, experimental data) -- always lives on the
        permanent ``data_bank_root``. With no scratch tier configured, every
        split resolves to ``data_bank_root``.
        """
        if self.scratch_data_bank_root is not None and split.upper() in {"TRAIN", "TEST"}:
            return self.scratch_data_bank_root
        return self.data_bank_root


def load_machine_profile() -> MachineProfile:
    """Load and validate the active machine profile.

    Reads MACHINE_PROFILE env var, parses machine_profiles.toml at sibling repo
    root, validates required keys and types, and returns a MachineProfile.

    Raises ValueError with a clear message pointing to machine_profiles.example.toml
    on any failure.
    """
    profile_name = os.environ.get("MACHINE_PROFILE")
    if not profile_name:
        raise ValueError(
            "MACHINE_PROFILE environment variable is not set. "
            "Set it to the name of an active profile in machine_profiles.toml. "
            "See machine_profiles.example.toml at the sibling repo root for the schema."
        )

    repo_root = Path(__file__).resolve().parent.parent  # parent of the package directory
    profiles_path = repo_root / "machine_profiles.toml"
    if not profiles_path.exists():
        raise ValueError(
            f"machine_profiles.toml not found at {profiles_path}. "
            f"Copy machine_profiles.example.toml to machine_profiles.toml and edit "
            f"for your machine. machine_profiles.toml is gitignored to keep "
            f"per-machine absolute paths out of version control."
        )

    with open(profiles_path, "rb") as fh:
        profiles = tomllib.load(fh)

    if profile_name not in profiles:
        raise ValueError(
            f"Profile '{profile_name}' not found in {profiles_path}. "
            f"Available profiles: {sorted(profiles.keys())}."
        )

    profile = profiles[profile_name]
    required_keys = {
        "running_mode", "script_bank_root", "data_bank_root",
        "compute_backend", "num_workers",
    }
    missing = required_keys - profile.keys()
    if missing:
        raise ValueError(
            f"Profile '{profile_name}' is missing required keys: {sorted(missing)}. "
            f"See machine_profiles.example.toml for the schema."
        )

    if profile["running_mode"] not in {"LOCAL", "HPC"}:
        raise ValueError(
            f"Profile '{profile_name}' has running_mode={profile['running_mode']!r}; "
            f"must be exactly 'LOCAL' or 'HPC' (case-sensitive). "
            f"'LOCAL' is for prototyping (single GPU, small task counts); "
            f"'HPC' is for production (cluster, multi-task parallelism)."
        )

    if profile["compute_backend"] not in {"GPU", "CPU"}:
        raise ValueError(
            f"Profile '{profile_name}' has compute_backend={profile['compute_backend']!r}; "
            f"must be exactly 'GPU' or 'CPU'."
        )

    gpu_device_index = profile.get("gpu_device_index")
    if profile["compute_backend"] == "GPU" and gpu_device_index is None:
        raise ValueError(
            f"Profile '{profile_name}' has compute_backend='GPU' but no gpu_device_index. "
            f"Set gpu_device_index to a non-negative integer."
        )

    script_bank_root = Path(profile["script_bank_root"])
    data_bank_root = Path(profile["data_bank_root"])
    if not script_bank_root.is_dir():
        raise ValueError(
            f"Profile '{profile_name}' script_bank_root={script_bank_root} is not a directory."
        )
    if not data_bank_root.is_dir():
        raise ValueError(
            f"Profile '{profile_name}' data_bank_root={data_bank_root} is not a directory."
        )

    # Optional second (scratch) data tier. Present only on machines with split
    # storage; when absent the machine is single-tier (see MachineProfile.root_for).
    scratch_data_bank_root = None
    if profile.get("scratch_data_bank_root"):
        scratch_data_bank_root = Path(profile["scratch_data_bank_root"])
        if not scratch_data_bank_root.is_dir():
            raise ValueError(
                f"Profile '{profile_name}' scratch_data_bank_root={scratch_data_bank_root} "
                f"is not a directory."
            )

    return MachineProfile(
        name=profile_name,
        running_mode=profile["running_mode"],
        script_bank_root=script_bank_root,
        data_bank_root=data_bank_root,
        compute_backend=profile["compute_backend"],
        gpu_device_index=gpu_device_index,
        num_workers=int(profile["num_workers"]),
        scratch_data_bank_root=scratch_data_bank_root,
    )


# =============================================================================
# Sibling-wide defaults (frozen dataclasses)
# =============================================================================

@dataclass(frozen=True)
class Paths:
    """Sibling-wide path conventions and naming patterns.

    Defines the layout of output files under the configured `data_bank_root`,
    the subdirectory names for each data category (trajectories, videos, theta
    sets, checkpoints, posteriors), and Python `str.format` patterns for
    concrete filenames. Path-construction methods compose subdirectories and
    patterns into full filesystem paths.

    The `project_alias` is embedded in every output filename to preserve
    provenance: a `.h5` or `.zarr` file moved or shared outside the original
    repository still identifies the program, model, and iteration that
    produced it.
    """
    project_alias: str = "SRM_AND_SBI_DIMER_ALP"
    labor_subdir: str = "Labor"
    debug_subdir: str = "Debug"               # diagnostics dumps, nested under labor_subdir
    posit_subdir: str = "Posit"
    theta_subdir: str = "Theta"
    video_subdir: str = "Video"
    experiment_subdir: str = "Experiment/SPT_Data_MET_FAB_INLB_S-BSST712"  # real microscopy data, BioStudies S-BSST712 (nested by accession for provenance)
    trajectory_repo: str = "READY_TRACT"
    trajectory_pattern: str = "{project_alias}_{timing_label}_TASK_{task_alias}_SIM_{task_simulation}_{split}.h5"
    theta_set_pattern: str = "{project_alias}_{timing_label}_Theta_Set_TASK_{task_alias}_{split}.{ext}"
    video_set_pattern: str = "{project_alias}_{timing_label}_Video_Set_TASK_{task_alias}_{split}.{ext}"
    checkpoint_pattern: str = "{project_alias}_{timing_label}_Optimum_ANN.pth"
    resurrect_state_pattern: str = "{project_alias}_{timing_label}_Resurrect_State_ANN.pth"
    estimator_pattern: str = "{project_alias}_{timing_label}_Estimator.npz"
    test_loss_distribution_pattern: str = "{project_alias}_{timing_label}_Test_Loss_Distribution.npz"
    recovery_pattern: str = "{project_alias}_{timing_label}_MAP_Recovery"
    experiment_recovery_pattern: str = "{project_alias}_{timing_label}_MAP_Experiment"
    # Real microscopy videos are external raw data (not produced by this pipeline),
    # so they keep their native naming rather than the project_alias prefix. The
    # user copies them into <data_bank_root>/<experiment_subdir>/.
    experiment_pattern: str = "Experiment_{kind}_Cell_{cell}_{span}S_RAW.tif"
    compressed_ext: str = "zarr"
    uncompressed_ext: str = "npy"

    # The `timing_label` token (e.g., "2S_50FPS") encodes simulation duration
    # + frame rate in every output filename, so a 2 s and 10 s run never
    # collide on disk and a file moved out of context still identifies the
    # config that produced it. The token is rendered by RunTiming.label.

    def trajectory_dir(self, task_alias: int, data_bank_root: Path,
                       timing_label: str, split: str = "TRAIN") -> Path:
        """Per-task subdirectory for .h5 trajectory files.

        ``split`` ∈ {"TRAIN", "TEST", "EVAL"} namespaces the data by role so
        the held-out sets never collide with the training set on disk.
        """
        return (data_bank_root / self.video_subdir / self.trajectory_repo /
                f"{self.project_alias}_{timing_label}_TASK_{task_alias}_{split}")

    def trajectory_path(self, task_alias: int, task_simulation: int,
                        data_bank_root: Path, timing_label: str,
                        split: str = "TRAIN") -> Path:
        """Full path for a single .h5 trajectory file."""
        filename = self.trajectory_pattern.format(
            project_alias=self.project_alias,
            timing_label=timing_label,
            task_alias=task_alias,
            task_simulation=task_simulation,
            split=split,
        )
        return self.trajectory_dir(task_alias, data_bank_root, timing_label, split) / filename

    def theta_set_path(self, task_alias: int, data_bank_root: Path,
                       timing_label: str, compress: bool = True,
                       split: str = "TRAIN") -> Path:
        """Full path for a theta-set file (.zarr if compress, else .npy)."""
        ext = self.compressed_ext if compress else self.uncompressed_ext
        filename = self.theta_set_pattern.format(
            project_alias=self.project_alias,
            timing_label=timing_label,
            task_alias=task_alias,
            ext=ext,
            split=split,
        )
        return data_bank_root / self.theta_subdir / filename

    def video_set_path(self, task_alias: int, data_bank_root: Path,
                       timing_label: str, compress: bool = True,
                       split: str = "TRAIN") -> Path:
        """Full path for a video-set file (.zarr if compress, else .npy)."""
        ext = self.compressed_ext if compress else self.uncompressed_ext
        filename = self.video_set_pattern.format(
            project_alias=self.project_alias,
            timing_label=timing_label,
            task_alias=task_alias,
            ext=ext,
            split=split,
        )
        return data_bank_root / self.video_subdir / filename

    def checkpoint_path(self, data_bank_root: Path, timing_label: str) -> Path:
        """Full path for the optimum-ANN checkpoint file."""
        filename = self.checkpoint_pattern.format(
            project_alias=self.project_alias,
            timing_label=timing_label,
        )
        return data_bank_root / self.labor_subdir / filename

    def resurrect_state_path(self, data_bank_root: Path, timing_label: str) -> Path:
        """Full path for the resurrect-state file -- the complete training state
        (model + optimizer + scheduler + epoch + optimum + warm-restart counters)
        written atomically every epoch so a ``--resurrect`` requeue hot-restarts from
        the exact latest state instead of a fresh optimizer at peak LR.

        A transient resume file, not a scientific deliverable: it is always
        overwritten with the latest state and carries no provenance descriptor
        (unlike the checkpoint backups), so the read logic is a single ``exists()``
        check. Lives beside the optimum checkpoint in ``labor_subdir``.
        """
        filename = self.resurrect_state_pattern.format(
            project_alias=self.project_alias,
            timing_label=timing_label,
        )
        return data_bank_root / self.labor_subdir / filename

    def estimator_path(self, data_bank_root: Path, timing_label: str) -> Path:
        """Full path for the version-portable estimator artifact (.npz).

        The self-describing estimator format (compile-stripped state_dict + rebuild
        spec + metadata; see ``artifacts.save_estimator``) that supersedes the
        torch-version-locked posterior pickle. Lives beside the former posterior in
        ``posit_subdir``.
        """
        filename = self.estimator_pattern.format(
            project_alias=self.project_alias,
            timing_label=timing_label,
        )
        return data_bank_root / self.posit_subdir / filename

    def test_loss_distribution_path(self, data_bank_root: Path, timing_label: str) -> Path:
        """Full path for the best-epoch test-loss distribution (.npz), a
        scientific deliverable alongside the posterior it characterizes."""
        filename = self.test_loss_distribution_pattern.format(
            project_alias=self.project_alias,
            timing_label=timing_label,
        )
        return data_bank_root / self.posit_subdir / filename

    # ---- Backup / archival artifact names --------------------------------
    # A finished training run overwrites the canonical checkpoint + posterior above
    # (the names every downstream stage loads). To keep an identifiable, restorable
    # history, each finished run also writes a backup copy whose name embeds the
    # run's provenance -- train/test set sizes, epoch count, and best TEST loss --
    # since the bare state_dict and posterior pickle carry no such metadata. The
    # canonical files remain the live objects; a backup is restored by copying it
    # back onto the canonical name.
    @staticmethod
    def format_backup_loss(test_loss: float) -> str:
        """Signed TEST-loss token with exactly two decimals: ``-17.05``, ``+3.20``.

        A value that rounds to zero (either sign) renders ``0.00`` with no sign.
        Operates on the formatted string, so it never relies on float equality.
        """
        text = f"{test_loss:.2f}"
        if text in ("0.00", "-0.00"):
            return "0.00"
        return text if text.startswith("-") else f"+{text}"

    @staticmethod
    def format_backup_size(n_videos: int) -> str:
        """Video count as a compact ``K`` token: 200000 -> ``200K``, 47500 -> ``47.5K``."""
        if n_videos % 1000 == 0:
            return f"{n_videos // 1000}K"
        return f"{n_videos / 1000:g}K"

    def backup_descriptor(self, train_videos: int, test_videos: int,
                          epochs: int, test_loss: float) -> str:
        """Provenance token appended to a backup filename, e.g.
        ``TRAIN+TEST_200K+50K_Epoch_25_TEST_LOSS_-17.05``."""
        return (
            f"TRAIN+TEST_{self.format_backup_size(train_videos)}"
            f"+{self.format_backup_size(test_videos)}"
            f"_Epoch_{epochs}_TEST_LOSS_{self.format_backup_loss(test_loss)}"
        )

    def backup_checkpoint_path(self, data_bank_root: Path, timing_label: str,
                               train_videos: int, test_videos: int,
                               epochs: int, test_loss: float) -> Path:
        """Provenance-named backup of the checkpoint, alongside the canonical one."""
        base = self.checkpoint_pattern.format(
            project_alias=self.project_alias, timing_label=timing_label)
        stem, ext = base.rsplit(".", 1)
        descriptor = self.backup_descriptor(train_videos, test_videos, epochs, test_loss)
        return data_bank_root / self.labor_subdir / f"{stem}_{descriptor}.{ext}"

    def backup_estimator_path(self, data_bank_root: Path, timing_label: str,
                              train_videos: int, test_videos: int,
                              epochs: int, test_loss: float) -> Path:
        """Provenance-named backup of the estimator artifact, alongside the canonical one."""
        base = self.estimator_pattern.format(
            project_alias=self.project_alias, timing_label=timing_label)
        stem, ext = base.rsplit(".", 1)
        descriptor = self.backup_descriptor(train_videos, test_videos, epochs, test_loss)
        return data_bank_root / self.posit_subdir / f"{stem}_{descriptor}.{ext}"

    def backup_test_loss_distribution_path(self, data_bank_root: Path, timing_label: str,
                                           train_videos: int, test_videos: int,
                                           epochs: int, test_loss: float) -> Path:
        """Provenance-named backup of the test-loss distribution, alongside the
        canonical one. ``epochs`` carries the epoch this best occurred at for a
        new-best backup, or the total planned epochs for the finish backup (the
        two coexist), exactly like the estimator/checkpoint backups."""
        base = self.test_loss_distribution_pattern.format(
            project_alias=self.project_alias, timing_label=timing_label)
        stem, ext = base.rsplit(".", 1)
        descriptor = self.backup_descriptor(train_videos, test_videos, epochs, test_loss)
        return data_bank_root / self.posit_subdir / f"{stem}_{descriptor}.{ext}"

    def debug_run_dir(self, data_bank_root: Path, timing_label: str, stage: str,
                      split: Optional[str] = None) -> Path:
        """Process-level diagnostics directory for one ``--debug-dump`` run.

        Layout: ``<data_bank_root>/<labor_subdir>/<debug_subdir>/<run_label>/<stage>/``
        where ``run_label`` is ``<project_alias>_<timing_label>[_<split>]``. This one
        directory holds the whole-process ``console.log`` plus the structured
        diagnostics: ``report.md`` + ``figures/`` directly (Inference / Evaluation),
        or per-task ``TASK_<n>/`` subdirectories (RDS / DLI). Diagnostics live under
        ``Labor/`` (the workbench) so ``Posit/`` stays reserved for scientific
        deliverables (the posterior and the recovery reports).
        """
        run_label = f"{self.project_alias}_{timing_label}"
        if split:
            run_label = f"{run_label}_{split}"
        return data_bank_root / self.labor_subdir / self.debug_subdir / run_label / stage

    def map_recovery_dir(self, data_bank_root: Path, timing_label: str) -> Path:
        """Directory holding the MAP-recovery report, figures, and saved arrays.

        Sits under ``Posit/`` (alongside the posterior it evaluates) and is the
        self-contained output of the Evaluation stage: ``report.md``,
        ``figures/``, and the recovery ``.npz`` (true vs. inferred theta).
        """
        return data_bank_root / self.posit_subdir / self.recovery_pattern.format(
            project_alias=self.project_alias,
            timing_label=timing_label,
        )

    def map_recovery_array_path(self, data_bank_root: Path, timing_label: str) -> Path:
        """Full path for the saved MAP-recovery array bundle (.npz)."""
        return self.map_recovery_dir(data_bank_root, timing_label) / (
            self.recovery_pattern.format(
                project_alias=self.project_alias, timing_label=timing_label,
            ) + ".npz"
        )

    def experiment_video_path(self, kind: str, cell: int, span_seconds: int,
                              data_bank_root: Path) -> Path:
        """Full path for a real microscopy video (read by the Experiment stage).

        Files live under ``<data_bank_root>/<experiment_subdir>/`` and use the
        native raw naming ``Experiment_{kind}_Cell_{cell}_{span}S_RAW.tif`` (the
        user copies them into the data bank to keep all data sources co-located).
        """
        filename = self.experiment_pattern.format(
            kind=kind, cell=cell, span=span_seconds)
        return data_bank_root / self.experiment_subdir / filename

    def experiment_recovery_dir(self, data_bank_root: Path, timing_label: str) -> Path:
        """Directory holding the Experiment-stage MAP report, figures, and arrays."""
        return data_bank_root / self.posit_subdir / self.experiment_recovery_pattern.format(
            project_alias=self.project_alias, timing_label=timing_label,
        )


@dataclass(frozen=True)
class SimulationStem:
    """ReaDDy simulation system geometry."""
    pixel_size_nm: int = 158                       # 157 + 1
    root_size_px: int = 256                        # 2^8 = 256
    crux_size_nm: float = 2.0                      # (7 + 3) / 5
    length_unit: str = "nanometer"
    time_unit: str = "nanosecond"
    particle_diameter_nm: int = 10

    @property
    def box_size(self) -> tuple[float, float, float]:
        """3D box dimensions (x, y, z) in nanometers."""
        return (self.pixel_size_nm * self.root_size_px,
                self.pixel_size_nm * self.root_size_px,
                self.crux_size_nm)

    @property
    def unit_dict(self) -> dict[str, str]:
        """Unit dict for ReaDDy ReactionDiffusionSystem constructor."""
        return {"length_unit": self.length_unit, "time_unit": self.time_unit}


@dataclass(frozen=True)
class FrameConfig:
    """Fixed frame-rate parameters.

    These define the sampling cadence (camera frame rate and per-frame
    sub-stepping) and never vary between runs. The global
    ``PARAMETERS.simulation.timing`` holds only this fixed configuration; the
    per-run recording length lives on :class:`RunTiming`, constructed at each
    entry-point from the required ``--total-time-seconds``. Keeping the length
    off the global makes it impossible to read a per-run ``frame_count`` from a
    global default.
    """
    frame_time_seconds: float = 0.020              # 50 Hz frame rate
    steps_per_frame: int = 10

    @property
    def fps(self) -> int:
        """Integer frames per second (``1 / frame_time_seconds``)."""
        return int(round(1 / self.frame_time_seconds))

    @property
    def delta_time_nanoseconds(self) -> float:
        """Per-step time delta in nanoseconds (for ReaDDy).

        Depends only on the fixed cadence, so it lives here.
        """
        return (self.frame_time_seconds / self.steps_per_frame) * 1e9


@dataclass(frozen=True)
class RunTiming:
    """Per-run timing: the recording length plus everything derived from it.

    Built at each entry-point from the required ``--total-time-seconds`` and the
    fixed :class:`FrameConfig`; never stored on the global ``PARAMETERS``. The
    derived ``frame_count`` is ``total_time_seconds / frame_time_seconds`` (length
    times frame rate), so it is always a function of the actual run, never a
    global default.
    """
    total_time_seconds: float                      # PER-RUN recording length in seconds (no default)
    frames: FrameConfig = field(default_factory=FrameConfig)

    # --- Fixed-cadence passthroughs, so callers read ``timing.<x>`` uniformly. ---
    @property
    def frame_time_seconds(self) -> float:
        return self.frames.frame_time_seconds

    @property
    def steps_per_frame(self) -> int:
        return self.frames.steps_per_frame

    @property
    def delta_time_nanoseconds(self) -> float:
        return self.frames.delta_time_nanoseconds

    # --- Per-run derived quantities. ---
    @property
    def frame_count(self) -> int:
        """Total number of frames = ``total_time_seconds / frame_time_seconds``."""
        return int(self.total_time_seconds / self.frames.frame_time_seconds)

    @property
    def total_steps(self) -> int:
        """Total simulation steps."""
        return self.frames.steps_per_frame * self.frame_count

    @property
    def label(self) -> str:
        """Filename-safe identifier for the active run timing.

        Format: ``"{duration}S_{fps}FPS"`` where duration uses ``:g``
        formatting (no trailing zeros: ``2`` not ``2.0``, ``2.5`` not
        ``2.500``) and fps is ``int(round(1 / frame_time_seconds))``.
        Used by every Paths.* path-construction method to namespace output
        files by run timing, so a 2 s and 10 s run never collide on disk and
        any artifact viewed in isolation identifies the config that produced it.
        """
        duration = f"{self.total_time_seconds:g}"
        return f"{duration}S_{self.frames.fps}FPS"


@dataclass(frozen=True)
class SimulationRDS:
    """RDS-stage runtime defaults."""
    prior_seed: Optional[int] = None               # None = OS-determined
    particle_species_names: tuple[str, ...] = ("A", "B", "C")
    # A = Monomer, B = Mobile Dimer, C = Immobile Dimer
    # (three-species DIMER model: A+A <-> B and B <-> C reactions)

    # ReaDDy neighbor-list (Verlet) skin, expressed as a MULTIPLE of the particle
    # diameter: the actual skin distance is
    #     skin = neighbor_list_skin_factor * SimulationStem.particle_diameter_nm  (nm).
    # This is a PURE PERFORMANCE knob -- it changes only how ReaDDy searches for
    # nearby particles, never the physics. ReaDDy finds reaction partners with a
    # cell-linked list whose cell edge is (reaction_radius + skin); reactions still
    # fire at the true reaction radius regardless of the skin, so the output is
    # unchanged. In the large, dilute imaging box (~40 um across, ~1000 particles)
    # the reaction radius alone (= 1 diameter = 10 nm) forces a ~16-million-cell grid
    # that is >99.99% empty, and per-step management of that grid -- not the physics
    # -- dominates runtime. A skin of a few tens of nm coarsens the grid ~120x and
    # recovers ~13x wall-clock with statistically identical output (verified: same
    # species composition and reaction counts at the prior extremes, incl. max
    # diffusivity + fusion-on-contact). The cost is U-shaped in the skin: too small
    # -> empty-cell sweep; too large -> the box collapses toward ~1 cell and the
    # candidate search degrades to O(N^2). The optimum is a broad plateau (~10x-100x
    # the diameter). The default 10x (= 100 nm) sits on that plateau and clears the
    # worst-case per-step displacement (~47 nm at max diffusivity) with margin, so no
    # reaction is missed. Overridable per run via --skin-factor / SKIN_FACTOR.
    neighbor_list_skin_factor: float = 10.0


@dataclass(frozen=True)
class SimulationDLI:
    """DLI-stage runtime defaults."""
    dimer_mule: float = 2.0                         # merged-dimer brightness vs monomer; 2 = two always-on labels (photons sum, MET), sqrt(2) under blinking/partial labeling (see PROJECT_CONTEXT.md)
    sqrt_2sigma_dist_label: str = "lognormal"      # PSF width sampling distribution
    # darkcounts removed (W1): the pre-PSF photon floor is the SCOPE-drawn optical background kappa_o (REFERENCE_EMCCD_NOISE_MODEL.md sec. 5); dark current is handled inside EMCCD (dark_current_e_per_s=0).


@dataclass(frozen=True)
class Simulation:
    """Aggregator. Access via PARAMETERS.simulation.{stem, timing, rds, dli}."""
    stem: SimulationStem = field(default_factory=SimulationStem)
    timing: FrameConfig = field(default_factory=FrameConfig)   # global holds ONLY fixed cadence; per-run length -> RunTiming
    rds: SimulationRDS = field(default_factory=SimulationRDS)
    dli: SimulationDLI = field(default_factory=SimulationDLI)


@dataclass(frozen=True)
class InferenceTraining:
    """Inference training hyperparameters."""
    epochs: int = 5
    batch_size: int = 32
    learning_rate_minimum: float = 1.0e-5
    learning_rate_maximum_factor: int = 128        # 2^7
    scheduler_factor: float = 0.5                  # ReduceLROnPlateau gamma (per-epoch LR anneal step)
    scheduler_patience: int = 1
    scheduler_tolerance_factor: float = 10.0       # tolerance = lr_min * factor
    # Warm restart: after the LR decays to learning_rate_minimum and stays there
    # warm_restart_dwell epochs WITHOUT a new best, reload the best checkpoint and
    # restart the LR at the previous sawtooth peak * warm_restart_factor -- a decaying
    # in-run plateau-escape that self-terminates once the next peak would reach the
    # floor. warm_restart_dwell = 0 disables it. warm_restart_factor is deliberately a
    # SEPARATE knob from scheduler_factor (the per-epoch anneal step): it sets the
    # restart AMPLITUDE decay, keeping each restart a gentle probe of a converged model
    # (0.25 -> the first restart is a quarter of the peak) rather than a large jump
    # halfway back up. Persisted in the resurrect-state, so the sawtooth is continuous
    # across a --resurrect requeue.
    warm_restart_dwell: int = 2
    warm_restart_factor: float = 0.25              # restart-peak amplitude decay per warm restart (not the anneal step)
    augmentation: bool = True                      # rotation + horizontal/vertical flip
    # Dataset-sizing defaults for the three-namespace split (TRAIN / TEST / EVAL),
    # consumed by the generation orchestrator. CORE = TRAIN + TEST.
    test_fraction: float = 0.2                     # TEST as a fraction of CORE
    eval_fraction: float = 0.1                     # EVAL as a fraction of CORE
    core_min: int = 10                             # minimum CORE size (samples)
    eval_floor: int = 10                           # minimum EVAL size (samples); for a stable recovery number


@dataclass(frozen=True)
class InferenceNetwork:
    """Complex3DCNN architecture defaults.

    These values control the embedding network that maps videos to a latent
    embedding for the downstream density estimator (MAF). See
    `inference_network.py` for the architecture; brief notes on each field:

    - `n_conv_layers`: depth of the 3D-CNN backbone (channels double per layer).
    - `n_attn_layers`: number of stacked self-attention blocks in the temporal
      transformer that summarizes the conv-stack output across time.
    - `start_channels`: output channels of the first conv block (becomes
      `start_channels * 2^(n_conv_layers - 1)` at the deepest layer).
    - `temporal_target_frames`: the temporal length, in FRAMES, that a video is
      reduced toward before the transformer, to bound memory for long
      recordings. A video of `n_frames` frames is reduced by an integer factor
      `s = n_frames // temporal_target_frames`, folded into the first conv's
      temporal stride; videos with `n_frames <= temporal_target_frames` are left
      untouched (`s = 1`, bit-identical to the un-reduced network, so the 2 s
      baseline is unaffected). Because `n_frames = duration_seconds * frame_rate`,
      this frame count corresponds to a different physical duration at different
      frame rates: 100 frames is 2 s at 50 FPS, 1 s at 100 FPS, or 4 s at 25 FPS.
      The recommended value is ~100 frames -- a memory / temporal-resolution
      balance at the sequence level, independent of FPS; values far outside
      ~50-200 either over-compress the motion signal (too small) or let memory
      grow again (too large). Set to `None` to disable temporal reduction.

      The target is a FACTOR, not an exact output length. The reduced length is
      the standard conv output size, `T_out = (n_frames - kernel) // s + 1` with
      `kernel = max(3, s)` and no padding once `s > 1`:

        | duration @50 FPS | n_frames | s | kernel | T_out                |
        |------------------|----------|---|--------|----------------------|
        | 2 s              |      100 | 1 |      3 | 100  (no reduction)  |
        | 3 s              |      150 | 1 |      3 | 150  (no reduction)  |
        | 4 s              |      200 | 2 |      3 |  99                  |
        | 5 s              |      250 | 2 |      3 | 124                  |
        | 6 s              |      300 | 3 |      3 | 100                  |
        | 10 s             |      500 | 5 |      5 | 100                  |

      Two consequences the plain reading hides. Because `s` is floor division, a
      video shorter than `2 * target` frames is not reduced at all -- 3 s (150
      frames) runs the transformer over all 150, so peak memory RISES from 2 s to
      3 s and then FALLS again at 4 s. And when `(n_frames - kernel) % s != 0`
      the trailing frames fall past the last window and are never read: at
      `s = 2` (n_frames 200-299, which includes 5 s @ 50 FPS) exactly one frame,
      the last, is dropped. For `s >= 3` kernel equals stride and windows tile
      exactly, so coverage is complete when `n_frames` is a multiple of `s`.

    The network returns the CLS-token embedding directly, which feeds the
    downstream MAF density estimator.
    """
    input_channels: int = 1
    n_conv_layers: int = 5
    n_attn_layers: int = 2
    start_channels: int = 8
    use_temporal_attention: bool = True
    attention_heads: int = 4
    temporal_target_frames: int = 100             # reduce longer videos toward ~this many frames
    #                                               (see class docstring); None disables reduction.


@dataclass(frozen=True)
class InferenceEvaluation:
    """MAP-recovery (Evaluation stage) defaults.

    The recovery procedure is a seed-then-optimize MAP estimate: draw a pool of
    candidate theta from the posterior, score each by the flow's log-probability,
    keep the top-`elite_prex_size` as optimization seeds, and gradient-ascent the
    log-probability with Adam + ReduceLROnPlateau and early stopping. Defaults are
    scaled for a high-memory GPU (the Evaluation stage runs on the GPU server);
    every value is overridable at the entry-point CLI.

    The `learning_rate` field is `learning_rate_minimum * learning_rate_maximum_factor`
    and the optimization `tolerance` is `learning_rate_minimum * tolerance_factor`,
    mirroring the training-stage convention.

    The `error_*` / `quantile_*` fields control the recovery-report figures:
    a per-parameter scatter of inferred-vs-true (log10) and a residual-error view,
    each with conditional-quantile bands drawn only where a bin holds at least
    `quantile_min_count` points (so the bands degrade gracefully for small EVAL).
    """
    pool_mode: str = "bounded"                     # candidate-pool sampler:
    #   "bounded"      -- DirectPosterior rejection sampling within prior ranges
    #                     (correct for a well-trained posterior).
    #   "unrestricted" -- sample the flow directly (no prior-range rejection);
    #                     never stalls, so it suits smoke tests and landscape
    #                     exploration on an undertrained posterior whose mass
    #                     lies outside the prior box.
    theta_prex_size: int = 1000                    # candidate pool size per video
    theta_prex_batch_size: int = 100               # sampling batch size
    score_prex_batch_size: int = 20                # log-prob scoring batch size
    elite_prex_size: int = 2                       # number of optimization seeds (top-K)
    numb_steps: int = 1000                         # max gradient-ascent steps
    optimizer_patience: int = 100                  # steps without improvement -> stop
    scheduler_patience: int = 10                   # steps without improvement -> reduce lr
    show_progress_steps: int = 100                 # progress-print cadence
    learning_rate_minimum: float = 1.0e-3
    learning_rate_factor: float = 0.5              # ReduceLROnPlateau gamma
    learning_rate_maximum_factor: int = 128        # 2^7; lr = lr_min * factor
    tolerance_factor: float = 1.0                  # tolerance = lr_min * factor
    # Recovery-report rendering:
    # Recovery tolerance bands, given as log10 half-widths -- each is the log10 of
    # a linear accuracy factor, so a point inside the band is recovered to within
    # that factor of the truth. Both are drawn as +/- guide lines on the error view
    # and reported as "fraction within" columns of the recovery table:
    #   error_guide       = 0.3  ~= log10(2)       -> within a factor of 2 of the truth
    #   error_guide_tight = 0.15 ~= log10(sqrt(2)) -> within a factor of sqrt(2) ~= 1.41
    #                              (0.15 = 0.3 / 2), a tighter concentration reference
    #                              nested inside the factor-2 band.
    error_guide: float = 0.3
    error_guide_tight: float = 0.15
    error_ylim_floor: float = 0.5                  # min half-range for the error y-axis (log10)
    error_ylim_quantile: float = 0.95              # |error| quantile setting the error y-axis
    quantile_bins: int = 20                        # conditional-quantile bins over true value
    quantile_min_count: int = 50                   # min points per bin to draw a band
    posterior_samples: int = 1000                  # samples/observation for the posterior-summary view (View B)


@dataclass(frozen=True)
class Inference:
    """Aggregator. Access via PARAMETERS.inference.{training, network, evaluation}."""
    training: InferenceTraining = field(default_factory=InferenceTraining)
    network: InferenceNetwork = field(default_factory=InferenceNetwork)
    evaluation: InferenceEvaluation = field(default_factory=InferenceEvaluation)


@dataclass(frozen=True)
class Plotting:
    """Plot defaults."""
    dpi: int = 500
    base_size: int = 10


# =============================================================================
# Top-level Parameters container
# =============================================================================

@dataclass(frozen=True)
class Parameters:
    """Top-level config aggregator. Access via the module-level PARAMETERS singleton."""
    machine: MachineProfile
    paths: Paths = field(default_factory=Paths)
    simulation: Simulation = field(default_factory=Simulation)
    inference: Inference = field(default_factory=Inference)
    plotting: Plotting = field(default_factory=Plotting)


# Module-level singleton. Instantiated at import time so configuration errors
# (missing env var, missing profile, invalid keys, missing directories) surface
# immediately on `import srm_and_sbi_dimer_alp.parameterization` rather than at
# the first read from PARAMETERS deep inside a simulation or training run.
PARAMETERS = Parameters(machine=load_machine_profile())


# =============================================================================
# Rich parameter spec
# =============================================================================
#
# Each parameter is a dict with the following fields:
#   {
#     'KEY':          unique parameter identifier (str),
#     'VALUE':        default / fixed value (scalar or list; for log-uniform
#                     priors this is the value at the center of the prior),
#     'PRIOR_RANGE':  (low, high) bounds for log-uniform priors; None for
#                     fixed (non-learnable) parameters,
#     'LOG_FLAG':     True if PRIOR_RANGE is given in log10 space,
#     'LOG_BASE':     base for the log transform (typically 10),
#     'UNIT':         human-readable unit of the parameter AS SAMPLED
#                     ('Dimensionless' for relative/ratio parameters),
#     'DERIVED_UNIT': for a dimensionless ratio that scales a physical
#                     quantity (e.g. R_B scales D_B), the unit of that derived
#                     quantity; None when the parameter is itself the physical
#                     quantity. Display-only documentation, never used in
#                     computation.
#     'LABEL':        LaTeX label for plotting,
#     'NOTE':         one of 'Learnable Parameter', 'Known Parameter',
#                     'Hyper Parameter' — distinguishes inferred parameters
#                     from fixed scientific constants and tuning hyperparameters.
#   }
#
# Top-level keys group parameters by domain:
#   RDS (Reaction-Diffusion System; molecular dynamics):
#     'count', 'diffusivity', 'dimerization_dissociation', 'immobilization_mobilization'
#   DLI (Diffraction-Limited Imaging; optical-detector model):
#     'camera', 'psf', 'transitivity'
#
# Notes on specific entries:
#   - Photobleaching is parameterized by ('prob_photo_bleach', 'numb_photo_bleach'):
#     prob_photo_bleach is the probability that an emitter enters the absorbing
#     bleached state over numb_photo_bleach camera frames. numb_photo_bleach = 100
#     is a FIXED reference window (= 2 s at 50 FPS), NOT the movie frame count --
#     do NOT set it to frame_count. prob_photo_bleach = 0.1 over this 100-frame
#     reference was inherited from detector-only inference on the raw experimental
#     videos. The per-frame rate is therefore constant across clip lengths:
#         p_1     = 1 - (1 - prob_photo_bleach) ** (1 / numb_photo_bleach)
#         p_video = 1 - (1 - prob_photo_bleach) ** (n_frames / numb_photo_bleach)
#     so a 500-frame (10 s) clip bleaches 1 - 0.9 ** 5 ~= 0.41 cumulatively, via
#     repeated application of the same per-frame transition matrix.
#   - 'delta_frame' VALUE is hardcoded to 0.020 (the default
#     RunTiming.frame_time_seconds). If a run overrides
#     `frame_time_seconds`, consumers should use the runtime value directly
#     rather than this default.
#   - 'capture_radius' is the Smoluchowski contact radius of two monomers
#     (2 * monomer_radius = 1 * diameter = particle_diameter_nm = 10 nm). The table
#     VALUE is display-only; build_system derives the active value from
#     PARAMETERS.simulation.stem.particle_diameter_nm, so it tracks per-dataset
#     geometry (particle_diameter_nm is the single physical input).

# Value-based parameter-role sentinels (DETECTOR_WORKFLOW.md sec. 5). A row whose
# VALUE is one of these strings is a role marker, not a concrete value: NUISANCE means
# "marginalized -- drawn per simulation, never inferred"; POSTERIOR means "drawn from a
# trained posterior". Defined above the parameter table so rows can carry them (the
# role dispatcher `role_of` that reads them lives further below, beside the filters).
NUISANCE_SENTINEL = "NUISANCE"
POSTERIOR_SENTINEL = "POSTERIOR"
_SENTINELS = (NUISANCE_SENTINEL, POSTERIOR_SENTINEL)


_PARAMETERIZATION_RAW_NESTED: dict[str, list[dict]] = {
    # ----- Reaction-Diffusion System -----
    'count': [  # particle_species_population_counts
        {'KEY': 'count_alp', 'VALUE': 10**1.25, 'PRIOR_RANGE': (0.0, 2.5), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count', 'DERIVED_UNIT': None, 'LABEL': r'$C_{A}$', 'NOTE': 'Learnable Parameter'},
        {'KEY': 'count_bet', 'VALUE': 10**1.25, 'PRIOR_RANGE': (0.0, 2.5), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count', 'DERIVED_UNIT': None, 'LABEL': r'$C_{B}$', 'NOTE': 'Learnable Parameter'},
        {'KEY': 'count_chi', 'VALUE': 10**1.25, 'PRIOR_RANGE': (0.0, 2.5), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count', 'DERIVED_UNIT': None, 'LABEL': r'$C_{C}$', 'NOTE': 'Learnable Parameter'},
    ],
    'diffusivity': [  # diffusion_coefficients / diffusion_constants
        {'KEY': 'diffusivity_alp', 'VALUE': 10**(-0.75), 'PRIOR_RANGE': (-1.25, -0.25), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Square Micrometer Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$D_{A}$', 'NOTE': 'Learnable Parameter'},
        {'KEY': 'relative_diffusivity_bet', 'VALUE': 10**(-0.375), 'PRIOR_RANGE': (-0.625, -0.125), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Dimensionless', 'DERIVED_UNIT': 'Square Micrometer Per Second', 'LABEL': r'$R_{B}$', 'NOTE': 'Learnable Parameter'},
        {'KEY': 'relative_diffusivity_chi', 'VALUE': 10**(-1.5), 'PRIOR_RANGE': (-2, -1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Dimensionless', 'DERIVED_UNIT': 'Square Micrometer Per Second', 'LABEL': r'$R_{C}$', 'NOTE': 'Learnable Parameter'},
    ],
    'dimerization_dissociation': [  # K_ON, K_OFF
        {'KEY': 'relative_rate_dimerization', 'VALUE': 10**(-1), 'PRIOR_RANGE': (-2, 0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Dimensionless', 'DERIVED_UNIT': 'Square Micrometer Per (Count*Second)', 'LABEL': r'$R_{ON}$', 'NOTE': 'Learnable Parameter'},
        {'KEY': 'capture_radius', 'VALUE': 10, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Nanometer', 'DERIVED_UNIT': None, 'LABEL': r'$\rho_{CAP}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'rate_dissociation', 'VALUE': 10**0, 'PRIOR_RANGE': (-1, 1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{OFF}$', 'NOTE': 'Learnable Parameter'},
    ],
    'immobilization_mobilization': [  # K_B_C, K_C_B
        {'KEY': 'rate_immobility', 'VALUE': 10**0, 'PRIOR_RANGE': (-1, 1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{IMMOBILITY}$', 'NOTE': 'Learnable Parameter'},
        {'KEY': 'rate_mobility', 'VALUE': 10**0, 'PRIOR_RANGE': (-1, 1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{MOBILITY}$', 'NOTE': 'Learnable Parameter'},
    ],
    # ----- Diffraction-Limited Imaging -----
    'camera': [  # EMCCD camera chain (REFERENCE_EMCCD_NOISE_MODEL.md): gamma, kappa_o, kappa_b, kappa_s, kappa_q marginalized as the SCOPE camera nuisance (non-identifiable; DETECTOR_WORKFLOW.md sec. 9.3, marginalized in both workflows); kappa_g, kappa_c fixed nominal spec metadata (gamma = kappa_g/kappa_c).
        {'KEY': 'gamma', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (1.62, 1.625), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'ADU Per Electron', 'DERIVED_UNIT': None, 'LABEL': r'$\gamma$',
         'NOTE': 'Gain-conversion ratio gamma = kappa_g/kappa_c (ADU per photoelectron) -- the only gain quantity the videos identify. Marginalized as a SCOPE camera nuisance: inferring it splits the peak-ADU amplitude with mu_pc (only gamma*kappa_q is identifiable), so it is drawn from its a-priori box rather than treated as a calibration target (DETECTOR_WORKFLOW.md sec. 9.3). Config-exact reference: kappa_g/kappa_c = 200/4.78 = 41.84, identical across all MET cells (both Fab and InlB camera protocols); box [41.7, 42.2].'},
        {'KEY': 'kappa_o', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (1.455, 1.465), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Photon', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{o}$',
         'NOTE': 'Optical background offset: incident photons per pixel per frame, pre-gain, one scalar per movie; amplified by gamma*kappa_q to set the ADU floor. Marginalized as a SCOPE camera nuisance (non-identifiable; DETECTOR_WORKFLOW.md sec. 9.3). Reference = ThunderSTORM offset[photon] median, a condition-independent background: Fab 28.9 / InlB 28.6 (pooled 28.7). Narrow box [28.5, 29.2] around the measured value: a broad offset lets the gain*offset floor dominate the video-to-video variation (embedding collapse).'},
        {'KEY': 'kappa_b', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (2.24, 2.25), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'ADU', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{b}$',
         'NOTE': 'Camera baseline: post-gain ADU constant, added last. Marginalized as a SCOPE camera nuisance (non-identifiable; DETECTOR_WORKFLOW.md sec. 9.3). Config-exact reference: MET configured baseline = 175, identical across all cells (both conditions); box [173.8, 177.8].'},
        {'KEY': 'kappa_s', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (1.02, 1.025), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'ADU', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{s}$',
         'NOTE': 'Read noise: post-register Gaussian sigma (ADU). Marginalized as a SCOPE camera nuisance: it does not dominate the SNR (the EM register amplifies signal above the read-noise floor) and is only weakly identifiable (DETECTOR_WORKFLOW.md sec. 9.3). Reference = camera datasheet ~10.5 ADU; box [10.5, 10.6] (weakly identifiable but pinned tight to the datasheet value, like the other camera constants).'},
        {'KEY': 'kappa_q', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': (-0.05, -0.04), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{q}$',
         'NOTE': 'Quantum efficiency, applied once in the Poisson step. Marginalized as a SCOPE camera nuisance: only the product gamma*kappa_q is identifiable from the videos (DETECTOR_WORKFLOW.md sec. 9.3). Config-exact reference: MET quantumEfficiency = 0.90, identical across all cells; box [0.89, 0.91].'},
        {'KEY': 'kappa_g', 'VALUE': 200, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{g}$',
         'NOTE': 'Nominal EM gain g from the MET acquisition config (ThunderSTORM camera protocol; audit-pending). Not inferred -- kept as spec metadata so the drawn gamma (SCOPE nuisance) can be checked against kappa_g/kappa_c (drift check).'},
        {'KEY': 'kappa_c', 'VALUE': 4.78, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Electron Per ADU', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{c}$',
         'NOTE': 'Nominal conversion C (e-/ADU) from the MET acquisition config (photons2ADU field; audit-pending). Not inferred -- kept as spec metadata for the gamma drift check. gamma = kappa_g / kappa_c.'},
    ],
    'psf': [  # Point Spread Function (lowercased for casing consistency)
        {'KEY': 'mu_r', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\mu_{r}$', 'NOTE': 'Imaging nuisance (role nuisance_object): the production DLI stage marginalizes it by drawing per simulation from the persisted Nuisance_DLI artifact and recording the draw as Nuisance_DLI_Theta_Set (DETECTOR_WORKFLOW.md sec. 9.3, Phase D). Calibrated operating point = the corrected imaging prior center (sec. 6.2), 10**0.15.'},
        {'KEY': 'sigma_r', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\sigma_{r}$', 'NOTE': 'Imaging nuisance (role nuisance_object): the production DLI stage marginalizes it by drawing per simulation from the persisted Nuisance_DLI artifact and recording the draw as Nuisance_DLI_Theta_Set (DETECTOR_WORKFLOW.md sec. 9.3, Phase D). Calibrated operating point = the corrected imaging prior center (sec. 6.2), 10**-0.625.'},
        {'KEY': 'mu_pc', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\mu_{pc}$', 'NOTE': 'Imaging nuisance (role nuisance_object): the production DLI stage marginalizes it by drawing per simulation from the persisted Nuisance_DLI artifact and recording the draw as Nuisance_DLI_Theta_Set (DETECTOR_WORKFLOW.md sec. 9.3, Phase D). Calibrated operating point = the corrected imaging prior center (sec. 6.2), 10**2.375.'},
        {'KEY': 'sigma_pc', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\sigma_{pc}$', 'NOTE': 'Imaging nuisance (role nuisance_object): the production DLI stage marginalizes it by drawing per simulation from the persisted Nuisance_DLI artifact and recording the draw as Nuisance_DLI_Theta_Set (DETECTOR_WORKFLOW.md sec. 9.3, Phase D). Calibrated operating point = the corrected imaging prior center (sec. 6.2), 10**-0.375.'},
    ],
    'transitivity': [  # CTMC Generator Matrix + DTMC Stochastic Matrix (emitter brightness state transitions)
        {'KEY': 'brightness_quantile', 'VALUE': [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': None, 'NOTE': 'Hyper Parameter'},
        {'KEY': 'delta_frame', 'VALUE': 0.020, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Second', 'DERIVED_UNIT': None, 'LABEL': r'$\delta_{f}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'prob_photo_bleach', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\rho_{pb}$', 'NOTE': 'Imaging nuisance (role nuisance_object): the production DLI stage marginalizes it by drawing per simulation from the persisted Nuisance_DLI artifact and recording the draw as Nuisance_DLI_Theta_Set (DETECTOR_WORKFLOW.md sec. 9.3, Phase D). Calibrated operating point = the corrected imaging prior center (sec. 6.2), 10**-1.25.'},
        {'KEY': 'numb_photo_bleach', 'VALUE': 100, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\psi_{pb}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'dimer_mule', 'VALUE': 2.0, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$m_{D}$', 'NOTE': 'Merged-dimer brightness relative to a monomer. Physical picture: a dimer is two labels within one PSF, so single-emitter fitting sees ONE spot whose photons combine into a brighter detection. Two combination models implement this (DETECTOR_WORKFLOW.md sec. 6.4). The DLI forward model renders dimers by dimer_model="sum": the merged brightness is the SUM of two INDEPENDENT monomer draws, each carrying its own flicker/bleach trajectory -> mean ~2x a monomer with a lighter upper tail than a rigid doubling; this path does NOT read dimer_mule. dimer_mule is consumed ONLY by dimer_model="multiply", the retained sensitivity alternative, which instead scales a SINGLE monomer draw by this factor. As that multiply factor it is regime-dependent in [1,2]: 2.0 = two PERMANENTLY-ON labels (the MET always-on ATTO 647N case, corroborated by the ~2x InlB/Fab per-spot intensity ratio); sqrt(2) ~= 1.41 when only ~one label is visible on average (photoswitching dye, or ~50% labeling). Fixed hyperparameter (value inert under the sum model); the render_dli_video fixed-hyperparameter source, mirroring PARAMETERS.simulation.dli.dimer_mule and the detector table dimer_mule.'},
        {'KEY': 'lambda_rate', 'VALUE': NUISANCE_SENTINEL, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\lambda$', 'NOTE': 'Imaging nuisance (role nuisance_object): the production DLI stage marginalizes it by drawing per simulation from the persisted Nuisance_DLI artifact and recording the draw as Nuisance_DLI_Theta_Set (DETECTOR_WORKFLOW.md sec. 9.3, Phase D). Calibrated operating point = the corrected imaging prior center (sec. 6.2), 10**0.5. Flicker locality is fixed (kappa_penalty=1 in compute_matrices); gamma_penalty retired.'},
    ],
}

# Validation: species count must match parameter count for 'count' and 'diffusivity'
_species_count = len(PARAMETERS.simulation.rds.particle_species_names)
# Explicit raises, not asserts: a count/diffusivity table misaligned with the species list
# would silently misalign every theta column downstream, and `python -O` strips asserts.
if len(_PARAMETERIZATION_RAW_NESTED['count']) != _species_count:
    raise ValueError(
        f"_PARAMETERIZATION_RAW_NESTED['count'] has "
        f"{len(_PARAMETERIZATION_RAW_NESTED['count'])} entries; "
        f"expected {_species_count} (one per particle species)."
    )
if len(_PARAMETERIZATION_RAW_NESTED['diffusivity']) != _species_count:
    raise ValueError(
        f"_PARAMETERIZATION_RAW_NESTED['diffusivity'] has "
        f"{len(_PARAMETERIZATION_RAW_NESTED['diffusivity'])} entries; "
        f"expected {_species_count} (one per particle species)."
    )


# Flat list (raw): all parameters in declaration order, regardless of group
PARAMETERIZATION_RAW: list[dict] = [
    para
    for group in _PARAMETERIZATION_RAW_NESTED.values()
    for para in group
]

# Index map: KEY -> position in PARAMETERIZATION_RAW (for full-spec access)
PARAMETER_RAW_FIND: dict[str, int] = {
    para['KEY']: index for index, para in enumerate(PARAMETERIZATION_RAW)
}


# ---- Value-based parameter roles ---------------------------------------------
# The role of each parameter is read from its (VALUE, PRIOR_RANGE) cell, so a block
# can be held Fixed, inferred as a Posterior (learnable), or marginalized as a
# Nuisance without structural change. A concrete VALUE + a range is learnable; a
# concrete VALUE + no range is fixed; the sentinels (defined above the table) mark the
# nuisance/posterior roles. Ported from the Detector workflow (DETECTOR_WORKFLOW.md
# sec. 5). The whole imaging block is marginalized in production (DETECTOR_WORKFLOW.md
# sec. 9.3, Phase D): the camera five (gamma, kappa_o, kappa_b, kappa_s, kappa_q) are the
# SCOPE camera nuisance (nuisance_spec, drawn per simulation from their a-priori boxes),
# and the six photophysics (mu_r, sigma_r, mu_pc, sigma_pc, prob_photo_bleach, lambda_rate)
# are the calibrated-imaging nuisance (nuisance_object, drawn per simulation from the
# persisted Nuisance_DLI artifact). The remaining imaging rows (kappa_g, kappa_c,
# brightness_quantile, delta_frame, numb_photo_bleach, dimer_mule) resolve to 'fixed', so
# the learnable subset is exactly the 10 RDS parameters.


def role_of(entry: dict) -> str:
    """Value-based role of one parameter entry.

    Returns 'learnable', 'fixed', 'nuisance_spec', 'nuisance_object', or 'posterior'.
    Dispatch is sentinel-based and never tests whether VALUE is numeric, so a
    list-valued fixed parameter (e.g. brightness_quantile) is classified correctly.
    The learnable subset is VALUE-not-a-sentinel AND PRIOR_RANGE-not-None
    (DETECTOR_WORKFLOW.md sec. 5, constraint 2): a nuisance-from-spec row also carries
    a range, so a bare PRIOR_RANGE test would wrongly pull it into the inference prior.
    POSTERIOR with a range is undefined -- a posterior draw carries its own support.
    """
    value, prior_range = entry['VALUE'], entry['PRIOR_RANGE']
    is_sentinel = isinstance(value, str) and value in _SENTINELS
    if value == POSTERIOR_SENTINEL:
        if prior_range is not None:
            raise ValueError(
                f"parameter {entry['KEY']!r}: POSTERIOR with a PRIOR_RANGE is undefined "
                f"(a posterior draw carries its own support); set PRIOR_RANGE=None.")
        return 'posterior'
    if value == NUISANCE_SENTINEL:
        return 'nuisance_spec' if prior_range is not None else 'nuisance_object'
    if not is_sentinel and prior_range is not None:
        return 'learnable'
    return 'fixed'


# Filtered list (learnable subset): VALUE-not-a-sentinel AND PRIOR_RANGE-not-None.
PARAMETERIZATION: list[dict] = [
    para for para in PARAMETERIZATION_RAW if role_of(para) == 'learnable'
]

# Index map: KEY -> position in PARAMETERIZATION (for theta-vector indexing)
PARAMETER_FIND: dict[str, int] = {
    para['KEY']: index for index, para in enumerate(PARAMETERIZATION)
}

# Ordered learnable-parameter keys = the theta-vector schema. Passed to
# artifacts.load_estimator(expected_parameter_keys=...) / assert_schema_compatible to
# hard-reject an estimator whose parameter schema differs (equal-length theta vectors
# would otherwise be misread column-for-column). Mirrors
# detector_parameterization.DETECTOR_PARAMETER_KEYS.
PARAMETER_KEYS: list[str] = [para['KEY'] for para in PARAMETERIZATION]


# =============================================================================
# Helpers
# =============================================================================

def parameter_find(key: str) -> int:
    """Return the index of a learnable parameter by KEY.

    Use this to index into theta vectors (which contain only learnable parameters).
    For all-parameter access (including 'Known' / 'Hyper'), use PARAMETER_RAW_FIND.

    Raises KeyError with the list of learnable parameters if the key is not found.
    """
    if key not in PARAMETER_FIND:
        raise KeyError(
            f"Parameter {key!r} is not a learnable parameter. "
            f"Learnable parameters: {list(PARAMETER_FIND.keys())}."
        )
    return PARAMETER_FIND[key]


def theta_lower_bound() -> list[float]:
    """Lower bounds of the log-uniform prior, in log10 space."""
    return [para['PRIOR_RANGE'][0] for para in PARAMETERIZATION]


def theta_upper_bound() -> list[float]:
    """Upper bounds of the log-uniform prior, in log10 space."""
    return [para['PRIOR_RANGE'][1] for para in PARAMETERIZATION]


def build_prior(device: str = "cpu") -> BoxUniform:
    """Construct the BoxUniform log-uniform prior over learnable parameters.

    Sampled theta values are in log10 space; consumers exponentiate via 10**theta
    to obtain physical values (i.e. `theta_sets = np.power(10, _theta_sets)`).
    """
    return BoxUniform(
        low=torch.tensor(theta_lower_bound()),
        high=torch.tensor(theta_upper_bound()),
        device=device,
    )

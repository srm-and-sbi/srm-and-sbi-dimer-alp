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
    posterior_pattern: str = "{project_alias}_{timing_label}_Posterior.pkl"
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

    def posterior_path(self, data_bank_root: Path, timing_label: str) -> Path:
        """Full path for the posterior pickle file."""
        filename = self.posterior_pattern.format(
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

    def backup_posterior_path(self, data_bank_root: Path, timing_label: str,
                              train_videos: int, test_videos: int,
                              epochs: int, test_loss: float) -> Path:
        """Provenance-named backup of the posterior, alongside the canonical one."""
        base = self.posterior_pattern.format(
            project_alias=self.project_alias, timing_label=timing_label)
        stem, ext = base.rsplit(".", 1)
        descriptor = self.backup_descriptor(train_videos, test_videos, epochs, test_loss)
        return data_bank_root / self.posit_subdir / f"{stem}_{descriptor}.{ext}"

    def posterior_path_for_checkpoint(self, checkpoint_path: Path,
                                      data_bank_root: Path) -> Path:
        """Posterior path matching a given checkpoint (canonical or a backup).

        Derives the output name from the checkpoint's own filename by swapping the
        object token and extension -- ``Optimum_ANN`` -> ``Posterior``, ``.pth`` ->
        ``.pkl`` -- and placing it under ``Posit/``. A canonical checkpoint maps to
        the canonical posterior; a descriptor-named backup checkpoint maps to the
        matching backup posterior (the descriptor rides along unchanged). Lets a
        posterior be rebuilt from any archived checkpoint under a consistent name.
        """
        name = checkpoint_path.name
        if "Optimum_ANN" not in name or not name.endswith(".pth"):
            raise ValueError(
                f"cannot derive a posterior name from {name!r}: expected an "
                f"'Optimum_ANN' checkpoint ending in '.pth'. Pass an explicit "
                f"posterior path instead."
            )
        posterior_name = name.replace("Optimum_ANN", "Posterior")[:-len(".pth")] + ".pkl"
        return data_bank_root / self.posit_subdir / posterior_name

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


@dataclass(frozen=True)
class SimulationDLI:
    """DLI-stage runtime defaults."""
    dimer_mule: float = 1.4142135623730951         # sqrt(2); brightness multiplier for dimers
    darkcounts: int = 0                            # baseline; no per-pixel dark current
    sqrt_2sigma_dist_label: str = "lognormal"      # PSF width sampling distribution


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
    scheduler_factor: float = 0.5                  # ReduceLROnPlateau gamma
    scheduler_patience: int = 1
    scheduler_tolerance_factor: float = 10.0       # tolerance = lr_min * factor
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

    The network returns the CLS-token embedding directly, which feeds the
    downstream MAF density estimator.
    """
    input_channels: int = 1
    n_conv_layers: int = 5
    n_attn_layers: int = 2
    start_channels: int = 8
    use_temporal_attention: bool = True
    attention_heads: int = 4


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
    error_guide: float = 0.3                       # +/- guide lines on the error view (log10)
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
#   - 'capture_radius' VALUE is hardcoded to 20 (= 2 *
#     SimulationStem.particle_diameter_nm, with the default 10 nm diameter).
#     Same caveat as 'delta_frame' for runtime overrides.

_PARAMETERIZATION_RAW_NESTED: dict[str, list[dict]] = {
    # ----- Reaction-Diffusion System -----
    'count': [  # particle_species_population_counts
        {'KEY': 'count_alp', 'VALUE': 10**2, 'PRIOR_RANGE': (1.75, 2.25), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count', 'DERIVED_UNIT': None, 'LABEL': r'$C_{A}$', 'NOTE': 'Learnable Parameter'},
        {'KEY': 'count_bet', 'VALUE': 10**2, 'PRIOR_RANGE': (1.75, 2.25), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count', 'DERIVED_UNIT': None, 'LABEL': r'$C_{B}$', 'NOTE': 'Learnable Parameter'},
        {'KEY': 'count_chi', 'VALUE': 10**2, 'PRIOR_RANGE': (1.75, 2.25), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count', 'DERIVED_UNIT': None, 'LABEL': r'$C_{C}$', 'NOTE': 'Learnable Parameter'},
    ],
    'diffusivity': [  # diffusion_coefficients / diffusion_constants
        {'KEY': 'diffusivity_alp', 'VALUE': 10**(-0.5), 'PRIOR_RANGE': (-1, 0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Square Micrometer Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$D_{A}$', 'NOTE': 'Learnable Parameter'},
        {'KEY': 'relative_diffusivity_bet', 'VALUE': 10**(-0.5), 'PRIOR_RANGE': (-0.75, -0.25), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Dimensionless', 'DERIVED_UNIT': 'Square Micrometer Per Second', 'LABEL': r'$R_{B}$', 'NOTE': 'Learnable Parameter'},
        {'KEY': 'relative_diffusivity_chi', 'VALUE': 10**(-1.5), 'PRIOR_RANGE': (-2, -1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Dimensionless', 'DERIVED_UNIT': 'Square Micrometer Per Second', 'LABEL': r'$R_{C}$', 'NOTE': 'Learnable Parameter'},
    ],
    'dimerization_dissociation': [  # K_ON, K_OFF
        {'KEY': 'relative_rate_dimerization', 'VALUE': 10**(-1), 'PRIOR_RANGE': (-2, 0), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Dimensionless', 'DERIVED_UNIT': 'Square Micrometer Per (Count*Second)', 'LABEL': r'$R_{ON}$', 'NOTE': 'Learnable Parameter'},
        {'KEY': 'capture_radius', 'VALUE': 20, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Nanometer', 'DERIVED_UNIT': None, 'LABEL': r'$\rho_{CAP}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'rate_dissociation', 'VALUE': 10**0, 'PRIOR_RANGE': (-1, 1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{OFF}$', 'NOTE': 'Learnable Parameter'},
    ],
    'immobilization_mobilization': [  # K_B_C, K_C_B
        {'KEY': 'rate_immobility', 'VALUE': 10**0, 'PRIOR_RANGE': (-1, 1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{IMMOBILITY}$', 'NOTE': 'Learnable Parameter'},
        {'KEY': 'rate_mobility', 'VALUE': 10**0, 'PRIOR_RANGE': (-1, 1), 'LOG_FLAG': True, 'LOG_BASE': 10, 'UNIT': 'Count Per Second', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{MOBILITY}$', 'NOTE': 'Learnable Parameter'},
    ],
    # ----- Diffraction-Limited Imaging -----
    'camera': [  # EMCCD detector parameters ('kappa_*' suffix)
        {'KEY': 'kappa_c', 'VALUE': 4.75, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{c}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'kappa_o', 'VALUE': 250, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Photon', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{o}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'kappa_g', 'VALUE': 150, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{g}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'kappa_v', 'VALUE': 50, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Photon', 'DERIVED_UNIT': None, 'LABEL': r'$\kappa_{v}$', 'NOTE': 'Known Parameter'},
    ],
    'psf': [  # Point Spread Function (lowercased for casing consistency)
        {'KEY': 'mu_r', 'VALUE': 1.5, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\mu_{r}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'sigma_r', 'VALUE': 0.15, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\sigma_{r}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'mu_pc', 'VALUE': 250, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\mu_{pc}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'sigma_pc', 'VALUE': 0.5, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\sigma_{pc}$', 'NOTE': 'Known Parameter'},
    ],
    'transitivity': [  # CTMC Generator Matrix + DTMC Stochastic Matrix (emitter brightness state transitions)
        {'KEY': 'brightness_quantile', 'VALUE': [0, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95], 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': None, 'NOTE': 'Hyper Parameter'},
        {'KEY': 'delta_frame', 'VALUE': 0.020, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': 'Second', 'DERIVED_UNIT': None, 'LABEL': r'$\delta_{f}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'prob_photo_bleach', 'VALUE': 0.1, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\rho_{pb}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'numb_photo_bleach', 'VALUE': 100, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\psi_{pb}$', 'NOTE': 'Known Parameter'},
        {'KEY': 'lambda_rate', 'VALUE': 5, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\lambda$', 'NOTE': 'Known Parameter'},
        {'KEY': 'gamma_penalty', 'VALUE': 0.025, 'PRIOR_RANGE': None, 'LOG_FLAG': None, 'LOG_BASE': None, 'UNIT': None, 'DERIVED_UNIT': None, 'LABEL': r'$\gamma$', 'NOTE': 'Known Parameter'},
    ],
}

# Validation: species count must match parameter count for 'count' and 'diffusivity'
_species_count = len(PARAMETERS.simulation.rds.particle_species_names)
assert len(_PARAMETERIZATION_RAW_NESTED['count']) == _species_count, (
    f"_PARAMETERIZATION_RAW_NESTED['count'] has "
    f"{len(_PARAMETERIZATION_RAW_NESTED['count'])} entries; "
    f"expected {_species_count} (one per particle species)."
)
assert len(_PARAMETERIZATION_RAW_NESTED['diffusivity']) == _species_count, (
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

# Filtered list (learnable subset): only entries with PRIOR_RANGE
PARAMETERIZATION: list[dict] = [
    para for para in PARAMETERIZATION_RAW if para['PRIOR_RANGE'] is not None
]

# Index map: KEY -> position in PARAMETERIZATION (for theta-vector indexing)
PARAMETER_FIND: dict[str, int] = {
    para['KEY']: index for index, para in enumerate(PARAMETERIZATION)
}


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

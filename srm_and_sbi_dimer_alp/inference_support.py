"""Training pipeline for the inference stage.

Wraps the simulation-based inference (SBI) training loop: loads (video, theta)
pairs from disk, builds train/val DataLoaders, runs an Adam+ReduceLROnPlateau
optimizer over an sbi posterior estimator, and saves the trained posterior to
a pickle file.

Module contents:
    Dataset
        normalize_video(video_raw)   -- int -> float32 [0, 1] via dtype max.
        VideoDataset(Dataset)        -- PyTorch Dataset over zarr/npy files,
                                        with optional rotation+flip augmentation.

    Setup
        get_device()                 -- torch.device from PARAMETERS.machine.
        build_datasets(...)          -- (train_dataset, val_dataset) with a
                                        shuffled-index train/val split.
        setup_training(...)          -- bundles loaders + optimizer + scheduler
                                        into a dict for `train_loop`.

    Loss
        compute_validation_loss(...) -- mean validation loss across a loader.

    Training loop
        train_loop(...)              -- epoch loop with optional RESURRECT: hot-restart
                                        from the full resurrect-state (optimizer +
                                        scheduler + epoch + warm-restart counters) when
                                        present, else the best-checkpoint cold fallback.
        save_resume_state / load_resume_state -- atomic full-state resume I/O.

    Posterior I/O
        save_posterior(posterior, prior_device, path)
        load_posterior(path)         -- returns a `DirectPosterior`.

Distributed across these functions is the SBI training pipeline:
NPE training with a masked autoregressive flow (MAF) density estimator
conditioned on the Complex3DCNN embedding of the input video.
"""

from dataclasses import dataclass
from pathlib import Path
import copy
import os
import pickle
import random
import time
from typing import Callable, Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel

from .io import load_data
from .parameterization import PARAMETERS


# Cap on DataLoader workers PER GPU when the budget is auto-derived (machine.num_workers <= 0,
# i.e. "use all cores"). Each DDP rank owns one GPU and keeps its loaders' workers alive
# (persistent_workers), so per-GPU workers = num_workers * n_live_loaders. Without this bound a
# very wide node (e.g. a 288-core GH200) auto-spawns ~cores/num_gpus workers per GPU and exhausts
# GPU memory. An explicit machine.num_workers > 0 is honored verbatim (NOT capped).
MAX_WORKERS_PER_GPU = 8


# =============================================================================
# Dataset
# =============================================================================

def normalize_video(video_raw: np.ndarray) -> np.ndarray:
    """Convert raw integer or float video to float32 in [0, 1].

    For integer dtypes, divides by the dtype's max value (e.g. 255 for uint8,
    65535 for uint16). For float dtypes, divides by `np.nanmax(video)` if
    that is > 1; otherwise returns as-is.

    Args:
        video_raw: Array of any numeric dtype.

    Returns:
        float32 array with the same shape, values in [0, 1].
    """
    video = video_raw.astype(np.float32)
    if np.issubdtype(video_raw.dtype, np.integer):
        return video / np.iinfo(video_raw.dtype).max
    max_value = np.nanmax(video)
    return video / max_value if max_value > 0 else video


class VideoDataset(Dataset):
    """PyTorch Dataset over (video, theta) pairs stored across `tasks` files.

    Files are loaded lazily (zarr is chunked; npy is memory-mapped) so the
    full dataset never needs to fit in RAM. Each task contributes
    `task_canons` examples; the dataset's total length is
    `tasks * task_canons`.

    Optional spatial augmentation applies a random 90-degree rotation and
    independent horizontal/vertical flips to each loaded video. Augmentation
    is enabled for training and disabled for validation.

    Args:
        tasks: Number of theta-set / video-set files to load.
        data_bank_root: Base directory containing the `Theta/` and `Video/`
            subdirectories with the saved files.
        timing_label: Filename token identifying the simulation timing config
            (e.g., ``"2S_50FPS"``); produced by ``RunTiming.label``.
            Must match the label of the run that produced the files on disk.
        compress: If True, expect `.zarr` files; otherwise `.npy` / `.npz`.
        indices: Optional list of indices to restrict this dataset to a
            subset of `tasks * task_canons`. Used by `build_datasets` to
            implement the train/val split.
        augment: If True, apply random rotation+flip augmentation in
            `__getitem__`. Augmentation is on for the TRAIN dataset and off for
            the TEST dataset; the replay loss is measured over a separate
            augmentation-off copy built by `build_replay_loader`.
    """

    def __init__(self,
                 tasks: int,
                 data_bank_root: Path,
                 timing_label: str,
                 compress: bool = True,
                 indices: Optional[np.ndarray] = None,
                 augment: bool = True,
                 split: str = "TRAIN",
                 return_index: bool = False,
                 paths=None):
        # `paths` selects the filename namespace. Default is the canonical
        # PARAMETERS.paths (byte-identical to previous behavior); the Detector
        # workflow passes an aliased Paths (project_alias + "_DETECTOR") so it
        # loads its own namespaced data. Only the filename prefix changes; the
        # directory (data_bank_root) is still the caller's argument.
        paths = paths if paths is not None else PARAMETERS.paths
        video_paths = []
        theta_paths = []
        for task_alias in range(tasks):
            video_paths.append(
                str(paths.video_set_path(
                    task_alias, data_bank_root, timing_label, compress, split))
            )
            theta_paths.append(
                str(paths.theta_set_path(
                    task_alias, data_bank_root, timing_label, compress, split))
            )

        # Probe the last theta file to determine how many examples per file.
        probe = load_data(theta_paths[-1])
        task_canons = probe.shape[0]
        del probe

        self.video_paths = video_paths
        self.theta_paths = theta_paths
        self.task_canons = task_canons
        self.data_canons = tasks * task_canons
        self.augment = augment
        self.indices = indices if indices is not None else np.arange(self.data_canons)
        self.train_test_flag = "Train" if augment else "Test"
        # When True, __getitem__ also yields the stable (task_index, sim_index)
        # identifier of each example (used only by the test-loss-distribution
        # loader); the default preserves the (video, theta) contract.
        self.return_index = return_index

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        full_index = int(self.indices[index])
        task_index = full_index // self.task_canons
        data_index = full_index % self.task_canons

        video_array = load_data(self.video_paths[task_index])
        theta_array = load_data(self.theta_paths[task_index])

        # Normalize video to [0, 1] float32; log10-transform theta to match the prior's log-space.
        video_normalized = normalize_video(video_array[data_index])
        theta_log10 = np.log10(theta_array[data_index])

        video = torch.tensor(video_normalized, dtype=torch.float32)
        theta = torch.tensor(theta_log10, dtype=torch.float32)

        if self.augment:
            video = self._spatial_augment(video)
        if self.return_index:
            # (task_index, sim_index) is the stable, extension-safe identifier of
            # this example; it rides through shuffling and DDP sharding so the
            # per-example test loss can be keyed by it downstream. (data_index is
            # the sim index within the task file.)
            return task_index, data_index, video, theta
        return video, theta

    @staticmethod
    def _spatial_augment(video: torch.Tensor) -> torch.Tensor:
        """Apply a random 90-degree rotation and independent H/V flips.

        Operates on the last two axes of the input tensor (spatial dims).
        Each augmentation is independently random per call.
        """
        k_rot = random.randint(0, 3)  # number of 90-degree rotations
        video = torch.rot90(video, k=k_rot, dims=[-2, -1])
        if random.random() > 0.5:
            video = torch.flip(video, dims=[-1])  # horizontal
        if random.random() > 0.5:
            video = torch.flip(video, dims=[-2])  # vertical
        return video


# =============================================================================
# Setup
# =============================================================================

@dataclass(frozen=True)
class Topology:
    """Parallel-launch layout for a stage run, shared by inference (DDP) and
    evaluation (independent sharding).

    ``world_size == 1`` is the single-GPU (or CPU) case and reproduces the
    original, non-distributed behavior exactly. ``world_size > 1`` means the
    process is one of several launched together (one per GPU): inference uses
    ``(world_size, rank)`` to drive DistributedDataParallel; evaluation uses
    them to take an independent ``videos[rank::world_size]`` shard.

    Attributes:
        world_size: Total number of parallel worker processes (= GPUs in use).
        rank: Global rank of this process in ``[0, world_size)``.
        local_rank: Rank within this node; the local GPU index this process binds.
        device: The torch device this process uses.
        backend: ``"GPU"`` or ``"CPU"``.
    """
    world_size: int
    rank: int
    local_rank: int
    device: torch.device
    backend: str

    @property
    def is_distributed(self) -> bool:
        """True when more than one worker was launched (DDP / sharding active)."""
        return self.world_size > 1

    @property
    def is_main(self) -> bool:
        """True on the coordinating rank (rank 0). Only this rank should write
        shared outputs (checkpoint, posterior, merged evaluation report)."""
        return self.rank == 0


def _env_int(*names: str, default: Optional[int] = None) -> Optional[int]:
    """Return the first parseable integer among the named environment variables."""
    for name in names:
        value = os.environ.get(name)
        if value:
            try:
                return int(value)
            except ValueError:
                continue
    return default


def resolve_topology() -> "Topology":
    """Discover the parallel-launch topology for the current process.

    A single discovery path serves both stages, so their multi-GPU interface is
    identical. Sources are read in priority order:

    1. ``torchrun`` / ``torch.distributed``: ``WORLD_SIZE``, ``RANK``, ``LOCAL_RANK``.
    2. Slurm: ``SLURM_NTASKS``, ``SLURM_PROCID``, ``SLURM_LOCALID``.
    3. Single-process fallback → ``world_size = 1`` (the original behavior).

    The worker count (``world_size``) is set at launch — by
    ``torchrun --nproc_per_node`` or the Slurm task count — which is how a run
    adapts to the allocated hardware; the launcher may cap it (e.g. from a
    ``SRM_AND_SBI_GPUS`` setting) before starting the processes. This function
    only reports what was launched; it does not start a process group (the
    inference stage does that itself when ``is_distributed``).

    Device binding: with multiple workers each binds its own ``cuda:local_rank``;
    a single worker honors the machine profile's ``gpu_device_index`` (so the
    non-distributed path is byte-for-byte unchanged). Falls back to CPU when
    CUDA is unavailable or the profile requests CPU.
    """
    machine = PARAMETERS.machine
    gpu_backend = machine.compute_backend == "GPU" and torch.cuda.is_available()

    world_size = _env_int("WORLD_SIZE", "SLURM_NTASKS", default=1) or 1
    rank = _env_int("RANK", "SLURM_PROCID", default=0) or 0
    local_rank = _env_int("LOCAL_RANK", "SLURM_LOCALID", default=0) or 0

    if gpu_backend:
        index = local_rank if world_size > 1 else (machine.gpu_device_index or 0)
        torch.cuda.set_device(index)   # bind this process's default CUDA device to its own GPU (no-op for the usual index 0)
        device = torch.device(f"cuda:{index}")
        backend = "GPU"
    else:
        if machine.compute_backend == "GPU":
            print("WARNING: machine profile requests GPU but CUDA is unavailable; "
                  "falling back to CPU.")
        device = torch.device("cpu")
        backend = "CPU"

    return Topology(world_size=world_size, rank=rank, local_rank=local_rank,
                    device=device, backend=backend)


def get_device() -> torch.device:
    """Return the torch.device for this process.

    Thin wrapper over :func:`resolve_topology`; preserves the original
    single-process behavior (honors the machine profile's ``gpu_device_index``,
    with a CPU fallback + warning). Multi-GPU callers use ``resolve_topology()``
    directly for the full ``(world_size, rank, local_rank, device)`` layout.
    """
    return resolve_topology().device


class _LossModule(nn.Module):
    """Adapt an sbi estimator for DistributedDataParallel.

    sbi's ``NFlowsFlow.forward`` returns ``None``; the trainable entry point is
    ``estimator.loss(input, condition)``. DDP synchronizes gradients by hooking
    the wrapped module's ``forward``, so this thin module makes ``forward`` *be*
    the per-sample loss. ``DDP(_LossModule(estimator))`` therefore all-reduces
    the estimator's gradients on ``backward``. Used unwrapped on a single GPU,
    its ``forward`` equals the original ``estimator.loss(...)`` call, so the
    non-distributed training path is unchanged.
    """

    def __init__(self, estimator: nn.Module):
        super().__init__()
        self.estimator = estimator

    def forward(self, theta: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
        return self.estimator.loss(theta, condition=condition)


def init_distributed(topo) -> None:
    """Initialize the default process group for DDP when launched distributed.

    No-op for a single worker (``world_size == 1``). ``torchrun`` provides
    ``MASTER_ADDR``/``MASTER_PORT``/``RANK``/``WORLD_SIZE``; ``resolve_topology``
    has already bound this process's local GPU. Uses the NCCL backend (RCCL on
    ROCm exposes the same name).
    """
    import torch.distributed as dist
    if topo.is_distributed and not dist.is_initialized():
        if "MASTER_ADDR" not in os.environ:
            raise RuntimeError(
                "Multi-process launch detected (world_size > 1) but no torch "
                "rendezvous is set (MASTER_ADDR unset). Launch multi-GPU inference "
                "via torchrun, not bare srun.")
        dist.init_process_group(backend="nccl")


def cleanup_distributed() -> None:
    """Tear down the default process group if one was initialized."""
    import torch.distributed as dist
    if dist.is_initialized():
        dist.destroy_process_group()


def build_datasets(train_tasks: int,
                   data_bank_root: Path,
                   timing_label: str,
                   compress: bool = True,
                   test_tasks: int = 0,
                   test_return_index: bool = False,
                   paths=None,
                   ) -> tuple:
    """Load the TRAIN and TEST namespaces as separate datasets.

    No shuffled pool split: the three roles (train / test / eval) are
    physically separate namespaces on disk, so leakage is impossible by
    construction. The TRAIN namespace supplies gradient updates; the TEST
    namespace (if any) supplies the per-epoch model-selection signal.

    Args:
        train_tasks: Number of TRAIN-namespace task files (gradient data).
        data_bank_root: Base data directory.
        timing_label: Filename token identifying the simulation timing config
            (e.g., ``"2S_50FPS"``); produced by ``RunTiming.label``.
        compress: If True, expect `.zarr`; otherwise `.npy`.
        test_tasks: Number of TEST-namespace task files for model selection.
            0 → no selection set (train on all; last-epoch checkpointing).

    Returns:
        (train_dataset, test_dataset) — ``test_dataset`` is None when
        ``test_tasks == 0``. Train has augmentation on; test has it off.
    """
    train_dataset = VideoDataset(
        tasks=train_tasks, data_bank_root=data_bank_root,
        timing_label=timing_label, compress=compress,
        augment=PARAMETERS.inference.training.augmentation, split="TRAIN",
        paths=paths,
    )
    test_dataset = None
    if test_tasks > 0:
        test_dataset = VideoDataset(
            tasks=test_tasks, data_bank_root=data_bank_root,
            timing_label=timing_label, compress=compress,
            augment=False, split="TEST", return_index=test_return_index,
            paths=paths,
        )
    return train_dataset, test_dataset


def build_optimizer_scheduler(estimator: nn.Module, learning_rate: float):
    """Build the AdamW optimizer + ReduceLROnPlateau scheduler for training.

    Single source of truth for the optimizer/scheduler pair, shared by
    `setup_training` (initial build) and `train_loop`'s warm restart (which rebuilds
    a fresh pair at the decayed peak LR), so the two paths cannot drift apart.

    Args:
        estimator: The underlying estimator whose parameters are optimized.
        learning_rate: The initial (peak) learning rate.

    Returns:
        (optimizer, scheduler).
    """
    training_cfg = PARAMETERS.inference.training
    optimizer = optim.AdamW(list(estimator.parameters()), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer=optimizer,
        mode="min",
        factor=training_cfg.scheduler_factor,
        patience=training_cfg.scheduler_patience,
        threshold=training_cfg.learning_rate_minimum * training_cfg.scheduler_tolerance_factor,
        threshold_mode="abs",
        min_lr=training_cfg.learning_rate_minimum,
    )
    return optimizer, scheduler


class ResumeStateMismatch(RuntimeError):
    """Raised when a resurrect-state's schema (``parameter_keys``) or ``timing_label``
    does not match the current run. Distinct from a corrupt/unreadable file: a mismatch
    is a deliberate refusal (``--resurrect`` was pointed at an incompatible run) and must
    propagate, whereas an unreadable file degrades to the cold checkpoint fallback.
    """


def save_resume_state(path: Path, *, estimator: nn.Module, optimizer: optim.Optimizer,
                      scheduler, epoch: int, optimum_loss_test: float,
                      warm_restart_peak: float, epochs_at_min_lr: int,
                      resume_meta: Optional[dict] = None) -> None:
    """Atomically write the full training state for a ``--resurrect`` hot restart.

    The resurrect-state bundles everything needed to continue training seamlessly
    across a requeue: the model weights, the AdamW moments + current LR (both in the
    optimizer state), the ReduceLROnPlateau plateau counters (in the scheduler state),
    the global epoch, the best-so-far TEST loss, and the warm-restart sawtooth
    counters. Unlike the best-on-TEST checkpoint (weights only), this is the *latest*
    state, written every epoch.

    Atomicity + durability: write a sibling ``.tmp``, ``fsync`` it, then ``os.replace``
    (an atomic rename on the same filesystem) and ``fsync`` the containing directory. So a
    mid-write kill (wall clock) or a node/power failure cannot leave a truncated or
    half-written file that would poison the next resume: the data is durable before the
    rename is exposed, and the rename itself is durable after -- the worst case is resuming
    from the previous epoch's complete state. Rank 0 only (the caller guards this). The
    reader still degrades to the cold checkpoint if a resume file is somehow unreadable.
    """
    state = {
        "model": estimator.state_dict(),
        "optimizer": optimizer.state_dict(),      # AdamW moments + current LR (param_groups)
        "scheduler": scheduler.state_dict(),      # ReduceLROnPlateau counters (best, num_bad_epochs, ...)
        "epoch": int(epoch),
        "optimum_loss_test": float(optimum_loss_test),
        "warm_restart_peak": float(warm_restart_peak),
        "epochs_at_min_lr": int(epochs_at_min_lr),
        "lr": float(optimizer.param_groups[0]["lr"]),   # log/debug only; authoritative LR rides in optimizer state
        "resume_meta": dict(resume_meta) if resume_meta else {},
        "torch_version": torch.__version__,
    }
    tmp_path = path.with_name(path.name + ".tmp")
    with open(tmp_path, "wb") as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())          # data durable before the rename is exposed
    os.replace(str(tmp_path), str(path))   # atomic name swap
    # Persist the directory entry too, so the rename survives a power/node failure.
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)


def load_resume_state(path: Path, estimator: nn.Module, optimizer: optim.Optimizer,
                      scheduler, resume_meta: Optional[dict] = None) -> dict:
    """Restore a resurrect-state written by `save_resume_state`, in place.

    Restores the model, optimizer, and scheduler state and returns the full state dict
    (the caller reads ``epoch``, ``optimum_loss_test``, ``warm_restart_peak``,
    ``epochs_at_min_lr``). ``weights_only=False`` because this is our own trusted
    artifact carrying optimizer + scheduler objects, not just tensors.

    Schema guard: when ``resume_meta`` is supplied, the stored ``parameter_keys`` and
    ``timing_label`` must match the current run, else the restore is refused with a
    located error -- a stale or mismatched file fails loudly (naming the fix) instead
    of silently loading. The model ``load_state_dict`` is strict, so an architecture
    mismatch is caught independently.
    """
    state = torch.load(str(path), map_location="cpu", weights_only=False)
    stored = state.get("resume_meta") or {}
    if resume_meta:
        old_keys = list(stored.get("parameter_keys") or [])
        new_keys = list(resume_meta.get("parameter_keys") or [])
        if old_keys and new_keys and old_keys != new_keys:
            raise ResumeStateMismatch(
                f"Resurrect-state schema mismatch at {path}: state parameter_keys "
                f"{old_keys} != run {new_keys}. Refusing to hot-restart a mismatched "
                f"schema; delete the file to start a cold --resurrect.")
        old_tl = stored.get("timing_label")
        new_tl = resume_meta.get("timing_label")
        if old_tl and new_tl and old_tl != new_tl:
            raise ResumeStateMismatch(
                f"Resurrect-state timing mismatch at {path}: state '{old_tl}' != run "
                f"'{new_tl}'. Refusing to hot-restart; delete the file to start cold.")
    estimator.load_state_dict(state["model"])   # strict: architecture guard
    # optimizer/scheduler restored in place. Optimizer.load_state_dict casts the AdamW
    # moments to each param's device automatically (the params are already on-device)
    # and keeps the step counter on CPU for the non-capturable path, so no manual device
    # placement is needed here.
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    return state


def setup_training(estimator: nn.Module,
                   train_tasks: int,
                   data_bank_root: Path,
                   timing_label: str,
                   compress: bool = True,
                   batch_size: Optional[int] = None,
                   learning_rate: Optional[float] = None,
                   num_workers_override: Optional[int] = None,
                   test_tasks: int = 0,
                   test_loss_distribution: bool = False,
                   paths=None) -> dict:
    """Bundle DataLoaders + optimizer + scheduler + device into a dict for `train_loop`.

    Args:
        estimator: The posterior estimator to train (an sbi MAF wrapping
            the Complex3DCNN embedding). Will be moved to the active device.
        train_tasks: Number of TRAIN-namespace task files (gradient data).
        data_bank_root: Base data directory.
        timing_label: Filename token identifying the simulation timing config
            (e.g., ``"2S_50FPS"``); produced by ``RunTiming.label``.
        compress: If True, expect `.zarr` files; otherwise `.npy`.
        batch_size: Optional override for the DataLoader batch size.
            Defaults to `PARAMETERS.inference.training.batch_size` (32).
        learning_rate: Optional override for the AdamW learning rate.
            Defaults to `lr_min * lr_max_factor` from `PARAMETERS.inference.training`.
        num_workers_override: Optional per-run override for the DataLoader worker
            TOTAL budget (beats the machine profile's num_workers; None -> the profile
            value, or auto = all cores capped at MAX_WORKERS_PER_GPU/GPU when that is <= 0).
        test_tasks: Number of TEST-namespace task files for model selection.
            0 → no selection set; `val_loader` is None and the train loop
            keeps the last-epoch checkpoint.

    Multi-GPU: when launched distributed (``resolve_topology().world_size > 1``)
    the estimator's BatchNorm layers are converted to SyncBatchNorm and the model
    is wrapped in DistributedDataParallel via ``_LossModule`` (sbi's estimator is
    trained through ``.loss``, not ``.forward``), and the TRAIN loader is sharded
    with a DistributedSampler. On a single worker the model is the bare
    ``_LossModule`` and the loader shuffles directly -- the non-distributed path
    is unchanged.

    Returns:
        Dict with keys: `estimator` (the underlying estimator, for checkpoint /
        posterior), `model` (the loss-computing module, DDP-wrapped when
        distributed), `train_loader`, `train_sampler` (None unless distributed),
        `val_loader`, `optimizer`, `scheduler`, `device`, `topo`. `val_loader` is
        None when `test_tasks == 0`. Ready to be passed to `train_loop`.
    """
    training_cfg = PARAMETERS.inference.training
    batch_size = batch_size or training_cfg.batch_size
    if learning_rate is None:
        learning_rate = training_cfg.learning_rate_minimum * training_cfg.learning_rate_maximum_factor

    topo = resolve_topology()
    device = topo.device
    estimator = estimator.to(device)

    # Wrap so the module's forward computes the per-sample loss (sbi's NFlowsFlow
    # forward returns None). Distributed -> SyncBatchNorm + DDP so gradients
    # all-reduce; single worker -> the bare module (forward == estimator.loss).
    model: nn.Module = _LossModule(estimator)
    if topo.is_distributed:
        # SyncBatchNorm shares the BatchNorm statistics across ranks so every
        # replica normalizes against the full effective batch (matching the
        # single-GPU large-batch behavior). It is ON BY DEFAULT and is the
        # correct, validated choice. It adds a per-step collective measured at
        # ~12-16% of a 4-GPU/5s epoch -- a minor cost, not the dominant one.
        # Opt out with SRM_AND_SBI_NO_SYNC_BN=1: each rank then keeps its OWN
        # local batch statistics, which is faster but changes the statistics, so
        # re-validate posterior recovery before relying on it.
        use_sync_bn = os.environ.get("SRM_AND_SBI_NO_SYNC_BN") != "1"
        if use_sync_bn:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if topo.is_main:
            state = "ENABLED (default)" if use_sync_bn else "DISABLED (SRM_AND_SBI_NO_SYNC_BN=1)"
            print(f"  SyncBatchNorm: {state}", flush=True)
        model = DistributedDataParallel(model, device_ids=[topo.local_rank],
                                        output_device=topo.local_rank)

    train_dataset, test_dataset = build_datasets(
        train_tasks=train_tasks, data_bank_root=data_bank_root,
        timing_label=timing_label, compress=compress, test_tasks=test_tasks,
        test_return_index=test_loss_distribution, paths=paths,
    )
    # ---- DataLoader worker budget (rank- and loader-aware) -------------------
    # Each DDP rank builds its OWN loaders, and persistent_workers keeps the train +
    # validation loaders' workers alive at the same time, so the LIVE worker-process
    # count is num_workers x world_size x n_live_loaders. Treat the configured value
    # (or the CPU core count, when <= 0) as the node-wide TOTAL worker budget and
    # divide it across ranks and concurrently-persistent loaders, so the product stays
    # ~1 worker per core regardless of how many GPUs are allocated (the auto/cores path is
    # additionally capped at MAX_WORKERS_PER_GPU per GPU -- see the branch below). This GENERALIZES
    # the old single-GPU `cores // 2` default, which silently multiplied to
    # world_size x n_live_loaders x that under DDP and exhausted host RAM at production
    # scale (4 ranks x 16 x 2 = 128 loader processes > 480 GB). At world_size == 1 with
    # a train+val pair it still resolves to cores // 2 per loader (single-GPU unchanged).
    # This is the data-loading worker budget only; the GPU/shard-worker count
    # (world_size) is bounded separately (SRM_AND_SBI_GPUS; eval/experiment cap it at
    # the task/cell count). Eval/experiment build no num_workers loaders, so this
    # budget is inference-only.
    cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
    n_live_loaders = 1 + (1 if test_dataset is not None else 0)   # train (+ persistent val)
    cap_note = ""
    # Per-run override (num_workers_override, e.g. --num-workers) beats the profile; both are an
    # explicit node-wide TOTAL budget honored verbatim (uncapped). Else fall back to auto (below).
    explicit_budget = (num_workers_override if (num_workers_override and num_workers_override > 0)
                       else PARAMETERS.machine.num_workers)
    if explicit_budget > 0:
        worker_budget = explicit_budget
        num_workers = max(1, worker_budget // (topo.world_size * n_live_loaders))
        if num_workers_override and num_workers_override > 0:
            cap_note = " [per-run override]"
    else:
        # Auto (<= 0): ~1 worker/core, but capped at MAX_WORKERS_PER_GPU per GPU so a very wide node
        # (e.g. a 288-core GH200 -> 4 GPUs x 72) can't over-spawn and exhaust GPU memory. Per-GPU
        # workers = num_workers x n_live_loaders (each rank owns one GPU).
        worker_budget = cores
        auto_nw = max(1, worker_budget // (topo.world_size * n_live_loaders))
        num_workers = min(auto_nw, max(1, MAX_WORKERS_PER_GPU // n_live_loaders))
        if num_workers < auto_nw:
            cap_note = f" [auto-capped to <={MAX_WORKERS_PER_GPU}/GPU]"
    # persistent_workers keeps the worker pool alive across epochs (avoids the
    # per-epoch re-spawn + stack re-import that otherwise dominates each epoch).
    persistent = num_workers > 0
    if topo.is_main:
        print(f"  DataLoader: num_workers={num_workers}/loader/rank "
              f"(budget={worker_budget} / {topo.world_size} rank(s) / {n_live_loaders} live loader(s) "
              f"-> {num_workers * topo.world_size * n_live_loaders} live workers; "
              f"persistent_workers={persistent}){cap_note}", flush=True)

    # Distributed -> shard TRAIN across ranks with a DistributedSampler (it owns
    # the shuffle, so `shuffle` is not passed); single worker -> shuffle directly.
    train_sampler = None
    if topo.is_distributed:
        train_sampler = DistributedSampler(
            train_dataset, num_replicas=topo.world_size, rank=topo.rank,
            shuffle=True, drop_last=False)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler,
                                  num_workers=num_workers, pin_memory=True,
                                  persistent_workers=persistent)
    else:
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                                  num_workers=num_workers, pin_memory=True,
                                  persistent_workers=persistent)
    # Distributed -> shard TEST across ranks too (DistributedSampler), so the
    # per-epoch selection loss is computed in parallel (each rank measures its
    # shard, then train_loop all-reduces the per-rank means) instead of rank 0
    # alone running the whole TEST set serially. shuffle=False -> a fixed,
    # balanced split. drop_last=False pads each rank to ceil(N/world_size) samples
    # so every rank has the SAME batch count -- which is exactly the condition
    # under which train_loop's unweighted all-reduce-mean of per-rank means equals
    # the single-process value (it divides evenly at production scale, e.g.
    # 25000/8=3125, so there is no padding; padding only duplicates a few boundary
    # samples for the awkward smoke-test counts where N is not a multiple of GPUs).
    val_loader = None
    if test_dataset is not None:
        val_sampler = (DistributedSampler(test_dataset, num_replicas=topo.world_size,
                                          rank=topo.rank, shuffle=False, drop_last=False)
                       if topo.is_distributed else None)
        val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                sampler=val_sampler, num_workers=num_workers,
                                pin_memory=True, persistent_workers=persistent)

    # Optimizer + scheduler on the underlying estimator's parameters (DDP wraps the
    # same tensors and all-reduces their gradients in backward); built here, after the
    # SyncBatchNorm conversion, so they see the converted parameters.
    optimizer, scheduler = build_optimizer_scheduler(estimator, learning_rate)

    return {
        "estimator": estimator,
        "model": model,
        "train_loader": train_loader,
        "train_sampler": train_sampler,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device": device,
        "topo": topo,
    }


# =============================================================================
# Validation loss
# =============================================================================

def build_replay_loader(train_loader: DataLoader, topo) -> DataLoader:
    """Build an augmentation-off, non-persistent loader over the TRAIN data.

    The replay loss is the eval-mode TRAIN loss measured under the same
    conditions as the TEST loss, which requires augmentation OFF. Toggling
    ``augment`` on the live TRAIN loader's dataset does not work: that loader
    uses ``persistent_workers=True``, so each worker holds its own pickled copy
    of the dataset and never observes a flag flipped in the main process -- the
    replay pass would silently run WITH augmentation. This builds a separate
    loader over an independent, augmentation-off shallow copy of the dataset and
    spawns fresh (non-persistent) workers, so the copy's ``augment=False`` is the
    state the workers actually see.

    Sharding mirrors the TRAIN loader: distributed -> a DistributedSampler with
    ``shuffle=False``/``drop_last=False`` (each rank measures its shard; the
    caller all-reduces the per-rank means), single worker -> a sequential pass
    over the full set. Order is irrelevant to a mean loss, so the sampler does
    not shuffle.

    Args:
        train_loader: The live TRAIN DataLoader (its dataset, batch size, and
            worker count are reused).
        topo: The launch Topology (drives the distributed sharding).

    Returns:
        A DataLoader yielding (video, theta) pairs with augmentation disabled.
    """
    replay_dataset = copy.copy(train_loader.dataset)   # independent object; shares the lazy path lists
    replay_dataset.augment = False                     # the workers pickle THIS copy (augmentation off)
    num_workers = train_loader.num_workers
    sampler = (DistributedSampler(replay_dataset, num_replicas=topo.world_size,
                                  rank=topo.rank, shuffle=False, drop_last=False)
               if topo.is_distributed else None)
    return DataLoader(replay_dataset, batch_size=train_loader.batch_size,
                      shuffle=False, sampler=sampler, num_workers=num_workers,
                      pin_memory=True, persistent_workers=False)


def compute_validation_loss(estimator: nn.Module,
                            loader: DataLoader,
                            device: torch.device,
                            verbose: bool = False,
                            collect: bool = False):
    """Compute mean loss over a loader in eval mode.

    The estimator is set to eval mode (dropout disabled, batch-norm uses running
    statistics), so the loss reflects the same network conditions as the test-set
    evaluation. The loader is expected to already disable augmentation -- the TEST
    loader is built with ``augment=False`` and the replay loader from
    ``build_replay_loader`` carries an augmentation-off dataset copy -- so this
    function does not toggle the flag (which would not reach a persistent loader's
    workers anyway).

    Batches may be the plain ``(video, theta)`` form or, from a ``return_index``
    dataset, ``(task_index, sim_index, video, theta)``; both are accepted. The
    mean is formed from the per-example loss exactly as before (a mean of
    per-batch means), so the returned mean is unchanged in either case.

    Args:
        estimator: The posterior estimator. Set to eval mode internally.
        loader: DataLoader over (video, theta) [or (task, sim, video, theta)] batches.
        device: Device to move batches to.
        verbose: If True, print the loader's augmentation flag.
        collect: If True, also return this process's per-example
            ``(task_index, sim_index, loss, theta)`` shard (requires a
            ``return_index`` loader). Used to build the test-loss distribution.

    Returns:
        ``collect=False``: mean loss across all batches, as a Python float.
        ``collect=True``: ``(mean, task, sim, loss, theta)`` with numpy arrays for
        this process's shard (``loss`` per example, ``theta`` shape (n, n_params)).
    """
    if verbose:
        print(f"Data augmentation? {loader.dataset.augment}")

    losses_per_batch = []
    tasks, sims, per_example, thetas = [], [], [], []
    estimator.eval()
    with torch.no_grad():
        for batch in loader:
            if len(batch) == 4:
                task_batch, sim_batch, video_batch, theta_batch = batch
            else:
                video_batch, theta_batch = batch
                task_batch = sim_batch = None
            video_batch = video_batch.to(device)
            theta_batch = theta_batch.to(device)
            example_loss = estimator.loss(theta_batch, condition=video_batch)
            losses_per_batch.append(torch.mean(example_loss).item())
            if collect:
                if task_batch is None:
                    raise ValueError(
                        "compute_validation_loss(collect=True) requires a "
                        "return_index loader yielding (task, sim, video, theta).")
                per_example.append(example_loss.detach().cpu().numpy())
                thetas.append(theta_batch.detach().cpu().numpy())
                tasks.append(np.asarray(task_batch))
                sims.append(np.asarray(sim_batch))

    mean = float(np.mean(losses_per_batch))
    if not collect:
        return mean
    return (mean,
            np.concatenate(tasks), np.concatenate(sims),
            np.concatenate(per_example), np.concatenate(thetas, axis=0))


def _gather_test_loss(task, sim, loss, theta, topo):
    """Gather per-rank ``(task, sim, loss, theta)`` shards across DDP ranks into
    full arrays available on every rank. Identity when not distributed. Any
    DistributedSampler boundary-padding duplicates are de-duplicated downstream
    by ``TestLossDistribution.from_epoch`` (keyed by ``(task, sim)``)."""
    if not topo.is_distributed:
        return task, sim, loss, theta
    import torch.distributed as dist
    parts = [None] * topo.world_size
    dist.all_gather_object(parts, (task, sim, loss, theta))
    task = np.concatenate([p[0] for p in parts])
    sim = np.concatenate([p[1] for p in parts])
    loss = np.concatenate([p[2] for p in parts])
    theta = np.concatenate([p[3] for p in parts], axis=0)
    return task, sim, loss, theta


# =============================================================================
# Training loop
# =============================================================================

def _diagnose_nonfinite_loss(model, video_batch, theta_batch, loss_value, epoch, batch, rank):
    """Trip-only breadcrumb for a non-finite training loss (see the finite guard in
    train_loop). Runs only on the failing batch, so it adds nothing to the finite path.
    Separates three causes: weights already diverged, a non-finite forward on finite
    weights, or a corrupt input batch. Prints to the log; the caller then aborts."""
    with torch.no_grad():
        bad = [name for name, p in model.named_parameters() if not torch.isfinite(p).all()]
        v_fin = bool(torch.isfinite(video_batch).all())
        t_fin = bool(torch.isfinite(theta_batch).all())
        v_lo, v_hi = video_batch.min().item(), video_batch.max().item()
        t_lo, t_hi = theta_batch.min().item(), theta_batch.max().item()
    weights_note = (f"{len(bad)} non-finite (e.g. {bad[:3]}) -> weights already diverged"
                    if bad else "0 non-finite -> weights finite; NaN arose in this forward")
    print(
        f"[FINITE-GUARD] non-finite training loss ({loss_value}) at "
        f"epoch {epoch}, batch {batch}, rank {rank}. Trip-only breadcrumb:\n"
        f"[FINITE-GUARD]   parameters: {weights_note}\n"
        f"[FINITE-GUARD]   video_batch: all_finite={v_fin} range=[{v_lo:.3e}, {v_hi:.3e}]\n"
        f"[FINITE-GUARD]   theta_batch: all_finite={t_fin} range=[{t_lo:.3e}, {t_hi:.3e}]",
        flush=True,
    )


def train_loop(estimator: nn.Module,
               model: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               optimizer: optim.Optimizer,
               scheduler,
               device: torch.device,
               topo,
               checkpoint_path: Path,
               train_sampler=None,
               epochs: Optional[int] = None,
               resurrect: bool = False,
               replay_loss: bool = False,
               heartbeat_every: Optional[int] = None,
               verbose: bool = False,
               test_loss_distribution: bool = False,
               on_new_best: Optional[Callable] = None,
               resurrect_state_path: Optional[Path] = None,
               resume_meta: Optional[dict] = None) -> tuple:
    """Run the training loop with optimum-checkpoint tracking, optional RESURRECT, and in-run warm restarts.

    Args:
        estimator: The underlying posterior estimator -- used for checkpointing,
            the eval-mode TEST/replay loss, and resurrect loading.
        model: The loss-computing module driven each batch (``_LossModule``,
            DDP-wrapped when distributed); its forward returns the per-sample loss.
        train_loader, val_loader: DataLoaders for the TRAIN and TEST splits.
        optimizer: Already-constructed optimizer (e.g. AdamW).
        scheduler: Learning-rate scheduler with `.step(metric)` (ReduceLROnPlateau).
        device: Device to move batches to.
        topo: The launch Topology (rank / world_size / is_main / is_distributed).
        checkpoint_path: Path where the best-so-far estimator checkpoint is saved
            (and, in resurrect mode, loaded from). Built by the entry point as
            ``PARAMETERS.paths.checkpoint_path(data_bank_root, timing.label)``.
        train_sampler: The DistributedSampler when distributed (its `set_epoch` is
            called each epoch to reshuffle the shard); None on a single worker.
        epochs: Number of epochs. Defaults to `PARAMETERS.inference.training.epochs`.
        resurrect: If True, resume before training: hot-restart from the full
            resurrect-state at `resurrect_state_path` when it exists (exact
            optimizer/scheduler/epoch/warm-restart state), else fall back to loading
            the best-on-TEST checkpoint weights into the fresh optimizer at the peak LR.
        replay_loss: If True, compute the per-epoch eval-mode TRAIN loss.
        heartbeat_every: Emit a within-epoch progress line every N batches (rank 0
            only). None -> ~4 lines/epoch (max(1, n_batches // 4)).
        verbose: If True, print per-batch progress.
        resurrect_state_path: Path to the full-state resume file (`Resurrect_State_ANN`).
            Written atomically every epoch (rank 0); on a `--resurrect` requeue it drives
            the hot restart when present. None disables the resurrect-state, so
            `--resurrect` uses only the cold checkpoint fallback.
        resume_meta: Optional {`timing_label`, `parameter_keys`} stamped into the
            resurrect-state and checked on hot restart, so a stale/mismatched file is
            refused rather than silently loaded.

    Epochs are per invocation: `epochs` is how many epochs THIS call runs, always
    starting a fresh 0..epochs-1 loop; on a hot restart the global epoch counter (for
    logging + the resurrect-state) continues from the resumed value, but the schedule
    itself is carried by the restored optimizer/scheduler state, not by an epoch index.

    Warm restart (config `warm_restart_dwell`): once the LR has decayed to the
    scheduler floor and held there `warm_restart_dwell` epochs without a new best,
    reload the best checkpoint, rebuild a fresh optimizer + scheduler at the previous
    peak times `warm_restart_factor` (a decaying sawtooth), and continue -- a within-run
    plateau-escape at the LR floor. `warm_restart_factor` (default 0.25) is a separate
    config knob from the per-epoch anneal `scheduler_factor`: it sets the restart
    amplitude, keeping each restart a gentle probe (a quarter of the peak) rather than a
    large jump halfway back up. The peak decays each restart and self-terminates once
    the next peak would reach the floor. `warm_restart_dwell == 0` disables it. Its state
    (`warm_restart_peak`, the floor-dwell tally) is persisted in the resurrect-state,
    so a `--resurrect` requeue continues the sawtooth mid-stride rather than resetting
    it -- the schedule behaves as one continuous run across requeues.

    Distributed: TRAIN and TEST are both sharded across ranks (DistributedSampler)
    and their per-rank mean losses are all-reduced, so every rank's scheduler +
    model selection stay in lockstep on values computed in parallel; only rank 0
    writes the checkpoint and prints. On a single worker every collective is the
    identity and the path matches the original.

    Returns:
        (losses_train, losses_test, losses_replay, optimum_loss_test) -- the three
        1D arrays of length `epochs` (per-epoch mean train, test, and replay loss),
        plus the best (lowest) TEST loss the saved checkpoint attained. In resurrect
        mode it starts from the loaded checkpoint's baseline, so it reflects the
        checkpoint on disk even when no epoch improved on it; it is ``inf`` when
        there is no TEST set.
    """
    import torch.distributed as dist
    if epochs is None:
        epochs = PARAMETERS.inference.training.epochs
    if epochs < 1:
        # At least one epoch must run: the optimum checkpoint is written inside the
        # epoch loop, and the final-model section reloads it. With epochs < 1 the
        # loop never runs, so (absent a resurrect checkpoint) the reload would fail
        # on a never-written file. Reject the no-op run loudly instead.
        raise ValueError(f"epochs={epochs} is invalid; train_loop requires epochs >= 1.")

    losses_train = np.full(shape=epochs, fill_value=np.nan)
    losses_test = np.full(shape=epochs, fill_value=np.nan)
    losses_replay = np.full(shape=epochs, fill_value=np.nan)

    has_val = val_loader is not None   # TEST namespace present -> model selection
    distributed = topo.is_distributed
    is_main = topo.is_main

    # Replay loss = eval-mode TRAIN loss with augmentation off, opt-in. Built once
    # as a dedicated augmentation-off, non-persistent loader (the live TRAIN loader
    # keeps augmentation on and its persistent workers would not see a toggled flag).
    replay_loader = build_replay_loader(train_loader, topo) if replay_loss else None

    def _reduce_mean(value: float) -> float:
        """Mean of a scalar across ranks (identity when not distributed)."""
        if not distributed:
            return value
        t = torch.tensor([value], device=device, dtype=torch.float64)
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        return float((t / topo.world_size).item())

    # ---- Warm-restart state (a within-run plateau-escape at the LR floor) -----
    # Once the LR decays to the floor and dwells there `warm_restart_dwell` epochs
    # without a new best, reload the best checkpoint and restart the LR at the previous
    # peak * `warm_restart_factor` (a decaying sawtooth; the restart-amplitude knob, kept
    # separate from the per-epoch anneal scheduler.factor). Disabled when
    # warm_restart_dwell == 0. These are the fresh-run defaults; a hot restart (below)
    # overrides them from the resurrect-state so the sawtooth continues across a requeue.
    warm_restart_dwell = PARAMETERS.inference.training.warm_restart_dwell
    warm_restart_peak = optimizer.param_groups[0]["lr"]  # current sawtooth peak; starts at the initial (peak) LR
    epochs_at_min_lr = 0                                  # consecutive floor epochs without a new best

    optimum_loss_test = float("inf")
    start_epoch = 0   # global epoch offset; nonzero only on a hot restart
    if resurrect:
        state = None
        if resurrect_state_path is not None and resurrect_state_path.exists():
            # HOT restart: resume the exact full state (model + optimizer + scheduler +
            # epoch + optimum + warm-restart counters). All ranks load the same
            # rank-0-written file from the prior job -- no barrier needed (no rank writes
            # it before this read) -- so the replicas stay identical.
            try:
                state = load_resume_state(resurrect_state_path, estimator, optimizer,
                                          scheduler, resume_meta=resume_meta)
            except ResumeStateMismatch:
                # Deliberate refusal: --resurrect was pointed at an incompatible run. Fail
                # loudly rather than silently downgrade to a mismatched cold restart.
                raise
            except Exception as exc:
                # Present but unreadable (truncated/corrupt after a crash, or a partial
                # rsync): exists() is True but torch.load raises. Degrade to the cold
                # checkpoint instead of crashing the requeue chain; the next epoch rewrites
                # a clean resume file. All ranks read the same shared file, so they take
                # this branch identically and stay in lockstep.
                state = None
                if is_main:
                    print(f"RESURRECT: resume file {resurrect_state_path} is present but "
                          f"unreadable ({type(exc).__name__}: {exc}); falling back to a "
                          f"cold restart from the checkpoint.", flush=True)
        if state is not None:
            # HOT restart bookkeeping -- optimum_loss_test comes from the state, so no
            # baseline TEST pass is needed.
            start_epoch = int(state["epoch"]) + 1
            optimum_loss_test = float(state["optimum_loss_test"])
            warm_restart_peak = float(state["warm_restart_peak"])
            epochs_at_min_lr = int(state["epochs_at_min_lr"])
            if is_main:
                print(f"HOT RESTART: resumed full state from {resurrect_state_path}; "
                      f"continuing at global epoch {start_epoch}, "
                      f"lr={optimizer.param_groups[0]['lr']:.2e}, "
                      f"best test loss = {optimum_loss_test:.5f}.", flush=True)
        else:
            # COLD fallback (resurrect-state absent, or present-but-unreadable above): load
            # the best weights into the FRESH optimizer at the peak LR -- the
            # pre-resurrect-state behavior. All ranks load the same checkpoint, staged to
            # CPU then placed onto each rank's own device. The next epoch writes a
            # resurrect-state, so subsequent requeues hot-restart.
            estimator.load_state_dict(
                torch.load(str(checkpoint_path), map_location="cpu", weights_only=True))
            if has_val:
                # Sharded TEST loss across ranks (see the per-epoch block below).
                optimum_loss_test = _reduce_mean(
                    compute_validation_loss(estimator, val_loader, device, verbose=verbose))
                if is_main:
                    print(f"RESURRECT (cold: fresh optimizer at peak LR): loaded checkpoint "
                          f"from {checkpoint_path}; baseline test loss = "
                          f"{optimum_loss_test:.5f}", flush=True)
            elif is_main:
                print(f"RESURRECT (cold): loaded checkpoint from {checkpoint_path} "
                      f"(no test set; last-epoch checkpointing).", flush=True)

    n_batches = len(train_loader)
    # Within-epoch progress cadence: a line every `heartbeat` batches. Default
    # (heartbeat_every unset) is ~4 lines/epoch; pass a smaller N (--heartbeat)
    # for finer progress on the long epochs of a production run.
    heartbeat = (heartbeat_every if (heartbeat_every and heartbeat_every > 0)
                 else max(1, n_batches // 4))
    n_train_videos = len(train_loader.dataset)
    n_test_videos = len(val_loader.dataset) if has_val else 0
    if is_main:
        where = f"{topo.world_size} GPUs (DDP)" if distributed else f"{device}"
        print(
            f"Training: {n_train_videos} train / {n_test_videos} test videos, "
            f"{n_batches} batch(es)/epoch/rank, {epochs} epoch(s) on {where}. "
            f"The first batch triggers model compilation -- expect a delay before the first heartbeat.",
            flush=True,
        )

    loop_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        global_epoch = start_epoch + epoch   # 0-indexed global epoch, continuous across requeues
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)   # reshuffle this rank's shard each epoch
        # ---- Training pass ------------------------------------------------
        model.train()
        batch_train_losses = []
        for b, (video_batch, theta_batch) in enumerate(train_loader, start=1):
            video_batch = video_batch.to(device)
            theta_batch = theta_batch.to(device)
            optimizer.zero_grad()
            loss = torch.mean(model(theta_batch, condition=video_batch))
            loss_value = loss.item()   # single host sync, reused below
            if not np.isfinite(loss_value):
                # Fail-fast: abort before the NaN propagates through backward/step
                # (a NaN can drive an out-of-bounds GPU access -> "memory access fault").
                # --resurrect resumes from the last checkpoint on the next submission.
                _diagnose_nonfinite_loss(model, video_batch, theta_batch,
                                         loss_value, epoch + 1, b, topo.rank)
                raise RuntimeError(
                    f"[FINITE-GUARD] non-finite training loss; aborting before backward/step "
                    f"(epoch {epoch + 1}, batch {b}, rank {topo.rank}).")
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss_value)
            # Always-on within-epoch heartbeat (~4/epoch), rank 0 only.
            if is_main and (b % heartbeat == 0 or b == n_batches):
                print(f"  epoch {epoch + 1}/{epochs}: batch {b}/{n_batches}  "
                      f"running train_loss={float(np.mean(batch_train_losses)):.5f}",
                      flush=True)

        # ---- Losses (TRAIN + TEST both sharded across ranks, then all-reduced) ----
        epoch_train = _reduce_mean(float(np.mean(batch_train_losses)))
        tld_shard = None
        if has_val:
            # TEST loss in eval mode. Distributed -> each rank measures its shard
            # of the (DistributedSampler-sharded) TEST set and the per-rank means
            # are all-reduced, so every rank gets the same selection metric,
            # computed in parallel instead of serially on rank 0. Single worker
            # -> _reduce_mean is the identity over the full set (unchanged).
            if test_loss_distribution:
                # One pass: the same batch-mean drives epoch_test (unchanged), and
                # the per-example shard is retained for a possible new-best commit.
                val_mean, t_task, t_sim, t_loss, t_theta = compute_validation_loss(
                    estimator, val_loader, device, verbose=False, collect=True)
                epoch_test = _reduce_mean(val_mean)
                tld_shard = (t_task, t_sim, t_loss, t_theta)
            else:
                epoch_test = _reduce_mean(
                    compute_validation_loss(estimator, val_loader, device, verbose=False))
        else:
            epoch_test = float("nan")
        # Replay loss = eval-mode TRAIN loss (augmentation off), opt-in; each rank
        # measures its shard of the dedicated augmentation-off loader, then averaged
        # across ranks.
        epoch_replay = (_reduce_mean(compute_validation_loss(estimator, replay_loader, device, verbose=False))
                        if replay_loss else float("nan"))
        losses_train[epoch] = epoch_train
        losses_test[epoch] = epoch_test
        losses_replay[epoch] = epoch_replay

        scheduler.step(epoch_test if has_val else epoch_train)

        # ---- Checkpoint (rank 0 only; the selection metric is identical on all ranks) ----
        new_best_this_epoch = False
        if has_val:
            if epoch_test < optimum_loss_test:
                new_best_this_epoch = True
                optimum_loss_test = epoch_test
                if is_main:
                    torch.save(estimator.state_dict(), str(checkpoint_path))
                # Commit the sibling best-artifacts (posterior + test-loss
                # distribution) and their new-best backups at this improvement, so
                # a crash leaves a complete best-so-far set. The gather is a
                # collective: EVERY rank calls it (the new-best decision is
                # identical across ranks, epoch_test being all-reduced); only rank 0
                # writes, inside on_new_best.
                if tld_shard is not None and on_new_best is not None:
                    gathered = _gather_test_loss(*tld_shard, topo)
                    if is_main:
                        on_new_best(epoch + 1, epoch_test, gathered)
        elif is_main:
            torch.save(estimator.state_dict(), str(checkpoint_path))

        epoch_secs = time.time() - epoch_start
        elapsed = time.time() - loop_start
        eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)   # mean epoch time x remaining
        if is_main:
            replay_str = f"{epoch_replay:.5f}" if replay_loss else "off"
            epoch_tag = (f"Epoch {epoch + 1}|{epochs} (global {global_epoch + 1})"
                         if start_epoch else f"Epoch {epoch + 1}|{epochs}")
            print(
                f"{epoch_tag}    "
                f"train={epoch_train:.5f}    test={epoch_test:.5f}    "
                f"replay={replay_str}    lr={scheduler.get_last_lr()[0]:.2e}    "
                f"epoch={epoch_secs:.1f}s    elapsed={time.strftime('%H:%M:%S', time.gmtime(elapsed))}    "
                f"ETA={time.strftime('%H:%M:%S', time.gmtime(eta))}",
                flush=True,
            )

        # ---- Warm restart trigger (in-run resurrect at the LR floor) --------
        if warm_restart_dwell > 0:
            # Count epochs sitting at the floor WITHOUT a new best; a new best (or an
            # LR still above the floor) resets the tally, so we only restart once the
            # floor has genuinely stalled.
            at_floor = optimizer.param_groups[0]["lr"] <= scheduler.min_lrs[0]
            epochs_at_min_lr = (epochs_at_min_lr + 1
                                if (at_floor and not new_best_this_epoch) else 0)
            # Amplitude decay uses warm_restart_factor -- a separate knob from the
            # per-epoch anneal scheduler.factor -- so restarts stay a gentle probe.
            next_peak = warm_restart_peak * PARAMETERS.inference.training.warm_restart_factor
            # Fire once the floor has stalled long enough, and only while the decayed
            # peak stays above the floor (else the restart is a no-op -- the sawtooth
            # has annealed to nothing and we just ride the floor to the end).
            if epochs_at_min_lr >= warm_restart_dwell and next_peak > scheduler.min_lrs[0]:
                warm_restart_peak = next_peak
                # Reload best-so-far weights (rank 0 wrote them on the last improvement;
                # a barrier guards the multi-rank read), then rebuild a fresh optimizer +
                # scheduler at the decayed peak via the shared builder -- identical to a
                # --resurrect cycle (fresh AdamW + ReduceLROnPlateau), without requeuing.
                if has_val:
                    if distributed:
                        dist.barrier()
                    estimator.load_state_dict(
                        torch.load(str(checkpoint_path), map_location="cpu", weights_only=True))
                optimizer, scheduler = build_optimizer_scheduler(estimator, warm_restart_peak)
                epochs_at_min_lr = 0
                if is_main:
                    print(f"WARM RESTART: LR held at the floor for {warm_restart_dwell} "
                          f"epoch(s) without improving; reloaded best (test loss "
                          f"{optimum_loss_test:.5f}) and restarted LR at {warm_restart_peak:.2e}.",
                          flush=True)

        # ---- Resurrect-state (full training state; rank 0; every epoch; atomic) ----
        # Written at the END of the epoch, so if a warm restart fired above it captures
        # the post-restart optimizer/scheduler + counters. A --resurrect requeue
        # hot-restarts from this exact state (see the resurrect block near the top). A
        # transient resume file -- always overwritten, never a scientific deliverable;
        # only rank 0 writes it and no rank reads it during the run, so no barrier.
        if is_main and resurrect_state_path is not None:
            save_resume_state(
                resurrect_state_path,
                estimator=estimator, optimizer=optimizer, scheduler=scheduler,
                epoch=global_epoch, optimum_loss_test=optimum_loss_test,
                warm_restart_peak=warm_restart_peak, epochs_at_min_lr=epochs_at_min_lr,
                resume_meta=resume_meta)

    # ---- Final model ------------------------------------------------------
    if has_val:
        # All ranks reload the best-on-TEST checkpoint (rank 0 wrote it; a barrier
        # guards the read) so every replica ends identical. Staged to CPU, then placed
        # onto each rank's own device by load_state_dict.
        if distributed:
            dist.barrier()
        estimator.load_state_dict(
            torch.load(str(checkpoint_path), map_location="cpu", weights_only=True))
        final_replay = _reduce_mean(compute_validation_loss(estimator, val_loader, device, verbose=False))
        if is_main:
            print(f"Final: optimum test loss = {optimum_loss_test:.5f}; "
                  f"replay test loss = {final_replay:.5f}")
    elif is_main:
        # No selection set: the last-epoch weights are already in the estimator
        # and saved to the checkpoint. Validation is done separately on EVAL.
        print("Final: last-epoch checkpoint kept (no test set; "
              "validate on the EVAL namespace).")

    return losses_train, losses_test, losses_replay, optimum_loss_test


# =============================================================================
# Posterior I/O
# =============================================================================

def save_posterior(posterior,
                   prior_device,
                   path: Path) -> None:
    """Pickle a trained posterior to disk, attaching a device-aware prior.

    The posterior estimator's prior is overwritten by `prior_device` before
    pickling. This is needed because sbi's `DirectPosterior` stores the
    prior used at training time (typically on CPU), but downstream sampling
    requires the prior to live on the same device as the estimator.

    Args:
        posterior: A trained `DirectPosterior` (or compatible) instance.
        prior_device: A `BoxUniform` (or other sbi prior) constructed on the
            desired device for sampling.
        path: Output pickle file path.
    """
    posterior._prior = prior_device
    with open(path, "wb") as fh:
        pickle.dump(posterior, fh)


def load_posterior(path: Path):
    """Load a pickled posterior from disk.

    Args:
        path: Path to a pickle file written by `save_posterior`.

    Returns:
        The `DirectPosterior` instance.
    """
    with open(path, "rb") as fh:
        return pickle.load(fh)

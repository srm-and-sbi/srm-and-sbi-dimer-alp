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
        train_loop(...)              -- epoch loop with optional RESURRECT
                                        (load existing checkpoint and continue).

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


def gpu_normalize_video(video_batch: torch.Tensor) -> torch.Tensor:
    """GPU-side equivalent of :func:`normalize_video` for an on-device batch.

    Integer batches (the uint8 production case): cast to float32, divide by the
    dtype max -- bit-identical to the CPU path. Float batches: divide per sample
    by its own max (no-op when <= 0), matching normalize_video's float branch."""
    if not torch.is_floating_point(video_batch):
        return video_batch.to(torch.float32).div_(float(torch.iinfo(video_batch.dtype).max))
    v = video_batch.to(torch.float32)
    mx = v.amax(dim=tuple(range(1, v.ndim)), keepdim=True)   # per-sample max over non-batch dims
    return torch.where(mx > 0, v / mx, v)


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
                 gpu_normalize: bool = False,
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
        # gpu_normalize: ship the raw uint8 video and let the training loop
        # cast+normalize on the GPU (4x smaller H2D; result bit-identical).
        self.gpu_normalize = gpu_normalize

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int):
        full_index = int(self.indices[index])
        task_index = full_index // self.task_canons
        data_index = full_index % self.task_canons

        video_array = load_data(self.video_paths[task_index])
        theta_array = load_data(self.theta_paths[task_index])

        theta_log10 = np.log10(theta_array[data_index])
        theta = torch.tensor(theta_log10, dtype=torch.float32)

        if self.gpu_normalize:
            # Return the raw integer video; the loop casts to float32 + normalizes on GPU.
            # Augmentation (rot90 / flips) is a pixel permutation -> order-independent of the
            # normalize, so it is applied here on the integer tensor unchanged.
            video = torch.from_numpy(np.ascontiguousarray(video_array[data_index]))
        else:
            # Normalize video to [0, 1] float32 on the CPU (default path).
            video = torch.tensor(normalize_video(video_array[data_index]), dtype=torch.float32)

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
    """Initialize the default process group for DDP (single- or multi-node).

    No-op for a single worker. NCCL backend (RCCL on ROCm). Supports torchrun
    launches (rendezvous env already set) and bare-srun launches, where the
    Slurm rank variables are mirrored into RANK / WORLD_SIZE / LOCAL_RANK;
    MASTER_ADDR / MASTER_PORT must then come from the launcher.
    """
    import torch.distributed as dist
    if not (topo.is_distributed and not dist.is_initialized()):
        return
    # Bridge Slurm -> torch rendezvous env for the bare-srun path (torchrun already
    # sets RANK, so this is skipped under torchrun and its behavior is unchanged).
    if "RANK" not in os.environ and os.environ.get("SLURM_PROCID") is not None:
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ.get("SLURM_NTASKS", str(topo.world_size))
        os.environ["LOCAL_RANK"] = os.environ.get("SLURM_LOCALID", str(topo.local_rank))
    if "MASTER_ADDR" not in os.environ:
        raise RuntimeError(
            "Multi-process launch detected (world_size > 1) but no torch rendezvous "
            "is set (MASTER_ADDR unset). Launch multi-GPU inference via torchrun, or "
            "under a bare srun export MASTER_ADDR / MASTER_PORT (the HPC launcher does "
            "this for the multi-node path).")
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
                   gpu_normalize: bool = False,
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
        gpu_normalize=gpu_normalize,
        paths=paths,
    )
    test_dataset = None
    if test_tasks > 0:
        test_dataset = VideoDataset(
            tasks=test_tasks, data_bank_root=data_bank_root,
            timing_label=timing_label, compress=compress,
            augment=False, split="TEST", return_index=test_return_index,
            gpu_normalize=gpu_normalize,
            paths=paths,
        )
    return train_dataset, test_dataset


class BatchRenorm(nn.Module):
    """Batch Renormalization (Ioffe 2017, arXiv:1702.03275): comms-free BatchNorm
    drop-in, robust to small per-rank batches.

    x_hat = (x - mu_B)/sigma_B * r + d with r, d clamped corrections toward the
    running stats, treated as constants (stop-gradient). r_max/d_max ramp from
    pure BatchNorm over warmup+ramp steps. Eval path identical to BatchNorm.
    Dimension-agnostic (channel = dim 1)."""

    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 r_max=3.0, d_max=5.0, warmup_steps=100, ramp_steps=1000):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum if momentum is not None else 0.1
        self.r_max_final = r_max
        self.d_max_final = d_max
        self.warmup_steps = warmup_steps
        self.ramp_steps = max(1, ramp_steps)
        self.weight = nn.Parameter(torch.ones(num_features))
        self.bias = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean", torch.zeros(num_features))
        self.register_buffer("running_var", torch.ones(num_features))
        self.register_buffer("num_batches_tracked", torch.tensor(0, dtype=torch.long))

    def _rd_max(self):
        step = int(self.num_batches_tracked.item())
        if step <= self.warmup_steps:
            return 1.0, 0.0   # pure BatchNorm while the running stats settle
        t = min(1.0, (step - self.warmup_steps) / self.ramp_steps)
        return 1.0 + t * (self.r_max_final - 1.0), t * self.d_max_final

    def forward(self, x):
        shape = [1, self.num_features] + [1] * (x.ndim - 2)   # broadcast over channel dim 1
        if self.training:
            dims = [0] + list(range(2, x.ndim))
            mean = x.mean(dim=dims)
            var = x.var(dim=dims, unbiased=False)
            std = (var + self.eps).sqrt()
            r_max, d_max = self._rd_max()
            run_std = (self.running_var + self.eps).sqrt()
            r = (std.detach() / run_std).clamp(1.0 / r_max, r_max)
            d = ((mean.detach() - self.running_mean) / run_std).clamp(-d_max, d_max)
            x_hat = (x - mean.view(shape)) / std.view(shape) * r.view(shape) + d.view(shape)
            with torch.no_grad():   # update running stats from this rank's local batch
                self.running_mean += self.momentum * (mean - self.running_mean)
                self.running_var += self.momentum * (var - self.running_var)
                self.num_batches_tracked += 1
        else:
            run_std = (self.running_var + self.eps).sqrt()
            x_hat = (x - self.running_mean.view(shape)) / run_std.view(shape)
        return x_hat * self.weight.view(shape) + self.bias.view(shape)


def convert_to_batch_renorm(module: nn.Module) -> nn.Module:
    """Recursively replace every ``nn.*BatchNorm*`` with a :class:`BatchRenorm`, copying the
    affine params + running stats. Mirrors ``convert_sync_batchnorm``'s walk, so it reaches
    the ``BatchNorm3d`` inside the (compiled) Complex3DCNN. The MAF flow's own (nflows)
    BatchNorm is not an ``nn._BatchNorm`` and is left untouched -- consistent with SyncBN,
    which also only converts ``nn`` BatchNorm."""
    out = module
    if isinstance(module, nn.modules.batchnorm._BatchNorm):
        out = BatchRenorm(module.num_features, eps=module.eps, momentum=module.momentum)
        if module.affine:
            with torch.no_grad():
                out.weight.copy_(module.weight)
                out.bias.copy_(module.bias)
        with torch.no_grad():
            out.running_mean.copy_(module.running_mean)
            out.running_var.copy_(module.running_var)
            if module.num_batches_tracked is not None:
                out.num_batches_tracked.copy_(module.num_batches_tracked)
        # The new module's params/buffers were created on CPU; match the original's device
        # (estimator is already .to(device) when this runs) so DDP sees a single device.
        out = out.to(module.running_mean.device)
    for name, child in module.named_children():
        out.add_module(name, convert_to_batch_renorm(child))
    del module
    return out


def _build_node_local_group(topo):
    """Build the per-NODE process subgroup for node-local SyncBatchNorm.

    BatchNorm stats are synchronized only within a node (fast intra-node xGMI/NVLink), never
    across nodes, so the per-step BN all-reduce cost is independent of node count -- the model
    scales to more nodes while keeping node-batch statistics (GPUs/node x batch), which are far
    closer to the global batch than per-rank's local batch. Returns (group, n_nodes, gpus_per_node).
    On a single node the group is the whole world (identical to global SyncBatchNorm)."""
    import torch.distributed as dist
    world = topo.world_size
    gpus_per_node = _env_int("LOCAL_WORLD_SIZE", "SLURM_NTASKS_PER_NODE", default=world) or world
    gpus_per_node = max(1, min(gpus_per_node, world))
    n_nodes = max(1, world // gpus_per_node)
    my_node = topo.rank // gpus_per_node
    my_group = None
    # new_group is collective: EVERY rank must create EVERY subgroup, in the same order.
    for n in range(n_nodes):
        ranks = list(range(n * gpus_per_node, min((n + 1) * gpus_per_node, world)))
        g = dist.new_group(ranks=ranks)
        if n == my_node:
            my_group = g
    return my_group, n_nodes, gpus_per_node


def strip_syncbn_process_groups(module: nn.Module) -> None:
    """Set ``process_group = None`` on every SyncBatchNorm so the trained estimator can be
    pickled (posterior save). Node-local SyncBatchNorm holds an explicit ProcessGroup, which
    is unpicklable; the group is only needed for the per-step sync during TRAINING -- at
    inference SyncBatchNorm normalizes with the running stats regardless. No-op for global
    SyncBatchNorm (already None) and for BatchNorm/BatchRenorm."""
    for m in module.modules():
        if isinstance(m, nn.SyncBatchNorm):
            m.process_group = None


def setup_training(estimator: nn.Module,
                   train_tasks: int,
                   data_bank_root: Path,
                   timing_label: str,
                   compress: bool = True,
                   batch_size: Optional[int] = None,
                   learning_rate: Optional[float] = None,
                   test_tasks: int = 0,
                   test_loss_distribution: bool = False,
                   bn_mode: str = "sync",
                   num_workers_override: Optional[int] = None,
                   gpu_normalize: bool = False,
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
    # BatchNorm mode. CONSTRAINT: the CNN's BatchNorm carries the cross-sample
    # signal D_A recovery depends on, so every mode keeps batch statistics
    # (per-sample norms like GroupNorm collapse D_A). Modes only change how the
    # stats are shared across ranks: sync (global), node-local (per node),
    # per-rank (no sync), renorm (comms-free BatchRenorm).
    if bn_mode == "renorm":
        model = convert_to_batch_renorm(model)
        if topo.is_main:
            print("  Normalization: BatchRenorm (comms-free, running-stat corrected)", flush=True)
    elif topo.is_distributed and bn_mode == "node-local":
        pg, n_nodes, gpn = _build_node_local_group(topo)
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model, process_group=pg)
        if topo.is_main:
            print(f"  Normalization: node-local SyncBatchNorm ({n_nodes} node(s) x {gpn} GPU; "
                  f"sync within node only)", flush=True)
    elif topo.is_distributed and bn_mode == "sync":
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if topo.is_main:
            print("  Normalization: global SyncBatchNorm (default)", flush=True)
    elif topo.is_main:
        where = "per-rank BatchNorm (no sync)" if topo.is_distributed else f"BatchNorm ({bn_mode}; single worker)"
        print(f"  Normalization: {where}", flush=True)
    if topo.is_distributed:
        model = DistributedDataParallel(model, device_ids=[topo.local_rank],
                                        output_device=topo.local_rank)

    train_dataset, test_dataset = build_datasets(
        train_tasks=train_tasks, data_bank_root=data_bank_root,
        timing_label=timing_label, compress=compress, test_tasks=test_tasks,
        test_return_index=test_loss_distribution, gpu_normalize=gpu_normalize,
        paths=paths,
    )
    # DataLoader worker budget: each rank builds its own persistent train(+val)
    # loaders, so live workers = num_workers x world_size x n_live_loaders. The
    # profile value (or core count when <= 0) is the NODE-WIDE budget, divided
    # across ranks and loaders; single-GPU resolves to the original cores // 2.
    n_live_loaders = 1 + (1 if test_dataset is not None else 0)   # train (+ persistent val)
    if num_workers_override is not None:
        # Explicit PER-LOADER-PER-RANK count (--num-workers / SRM_AND_SBI_NUM_WORKERS),
        # bypassing the auto default. 0 = synchronous loading (no workers). The caller owns
        # the OOM math: live workers = N x world_size x n_live_loaders.
        num_workers = max(0, num_workers_override)
        budget_note = f"override --num-workers={num_workers}/loader/rank"
    elif PARAMETERS.machine.num_workers > 0:
        # A profile that sets an explicit node-wide budget keeps the divide-across-ranks behavior.
        num_workers = max(1, PARAMETERS.machine.num_workers // (topo.world_size * n_live_loaders))
        budget_note = f"profile budget={PARAMETERS.machine.num_workers} / {topo.world_size} rank(s) / {n_live_loaders} loader(s)"
    else:
        # Measured optima: 1 worker/rank suffices with gpu_normalize (decode-only
        # workers), 2/rank without (CPU normalize). See PR for the sweeps.
        num_workers = 1 if gpu_normalize else 2
        budget_note = f"auto ({'gpu-normalize on' if gpu_normalize else 'off'} -> {num_workers}/rank optimum)"
    # persistent_workers keeps the worker pool alive across epochs (avoids the
    # per-epoch re-spawn + stack re-import that otherwise dominates each epoch).
    persistent = num_workers > 0
    if topo.is_main:
        print(f"  DataLoader: num_workers={num_workers}/loader/rank "
              f"({budget_note} -> {num_workers * topo.world_size * n_live_loaders} live workers; "
              f"persistent_workers={persistent})", flush=True)

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

    # Optimizer on the underlying estimator's parameters (DDP wraps the same
    # tensors and all-reduces their gradients in backward); created after the
    # SyncBatchNorm conversion so it sees the converted parameters.
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
        "gpu_normalize": gpu_normalize,
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
                            collect: bool = False,
                            gpu_normalize: bool = False):
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
            if gpu_normalize:
                video_batch = gpu_normalize_video(video_batch)
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
               resume_from: Optional[Path] = None,
               fine_tune_lr: Optional[float] = None,
               replay_loss: bool = False,
               heartbeat_every: Optional[int] = None,
               early_stop_patience: int = 0,
               gpu_normalize: bool = False,
               verbose: bool = False,
               test_loss_distribution: bool = False,
               on_new_best: Optional[Callable] = None) -> tuple:
    """Run the training loop with optimum-checkpoint tracking and optional RESURRECT.

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
        checkpoint_path: Path where the best-so-far estimator checkpoint is saved.
            Built by the entry point as
            ``PARAMETERS.paths.checkpoint_path(data_bank_root, timing.label)``.
        train_sampler: The DistributedSampler when distributed (its `set_epoch` is
            called each epoch to reshuffle the shard); None on a single worker.
        epochs: Number of epochs. Defaults to `PARAMETERS.inference.training.epochs`.
        resume_from: If set, load this checkpoint before training and continue from
            its weights (reuse / fine-tune the model instead of training from
            scratch). May be the canonical ``checkpoint_path`` (default reuse) or a
            different file (fine-tune a prior model onto new data); in the latter
            case the canonical checkpoint is seeded with the loaded weights so the
            end-of-run best-on-TEST reload always finds a file. None -> from scratch.
        fine_tune_lr: If set (and resuming), the initial learning rate to start the
            schedule from -- a smaller warm-start LR to fine-tune a near-optimal
            model gently. Ignored on a from-scratch run.
        replay_loss: If True, compute the per-epoch eval-mode TRAIN loss.
        heartbeat_every: Emit a within-epoch progress line every N batches (rank 0
            only). None -> ~4 lines/epoch (max(1, n_batches // 4)).
        early_stop_patience: If > 0 (and a TEST set exists), stop after this many
            epochs without TEST-loss improvement once the LR has bottomed out at
            learning_rate_minimum. 0 disables (default).
        verbose: If True, print per-batch progress.

    Distributed: TRAIN and TEST are both sharded across ranks (DistributedSampler)
    and their per-rank mean losses are all-reduced, so every rank's scheduler +
    model selection stay in lockstep on values computed in parallel; only rank 0
    writes the checkpoint and prints. On a single worker every collective is the
    identity and the path matches the original.

    Returns:
        (losses_train, losses_test, losses_replay, optimum_loss_test) -- the three
        1D arrays of the per-epoch mean train, test, and replay loss (length =
        epochs actually run, which is < `epochs` when early stopping fires), plus
        the best (lowest) TEST loss the saved checkpoint attained. When resuming
        it starts from the loaded model's baseline, so it reflects the checkpoint
        on disk even when no epoch improved on it; it is ``inf`` when there is no
        TEST set.
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

    optimum_loss_test = float("inf")
    if resume_from is not None:
        # Reuse an existing model (performance switch #3: fine-tune / warm-start).
        # All ranks load the same checkpoint -- staged to CPU, then placed onto each
        # rank's own device by load_state_dict -- so the replicas stay identical and no
        # single device accumulates every rank's copy. No barrier needed here: the
        # source checkpoint is from a prior completed run, so no rank writes it before
        # this read (and the canonical checkpoint, if distinct, is seeded rank-0-only below).
        estimator.load_state_dict(
            torch.load(str(resume_from), map_location="cpu", weights_only=True))
        # Warm-start LR: a resumed model is already near-optimal, so starting the
        # schedule at the full initial LR can destabilize it (an early loss spike).
        # A smaller fine_tune_lr fine-tunes gently. ReduceLROnPlateau reads the live
        # param-group LR, so setting it here (before the loop) becomes the new
        # schedule start. Ignored when None (keeps the setup_training LR).
        if fine_tune_lr is not None:
            for group in optimizer.param_groups:
                group["lr"] = fine_tune_lr
        ft_note = f"; fine-tune LR = {fine_tune_lr:.2e}" if fine_tune_lr is not None else ""
        if has_val:
            # Sharded TEST loss across ranks (see the per-epoch block below).
            optimum_loss_test = _reduce_mean(
                compute_validation_loss(estimator, val_loader, device, verbose=verbose, gpu_normalize=gpu_normalize))
            if is_main:
                print(f"REUSE: loaded model from {resume_from}; "
                      f"baseline test loss = {optimum_loss_test:.5f}{ft_note}")
        elif is_main:
            print(f"REUSE: loaded model from {resume_from} "
                  f"(no test set; last-epoch checkpointing).{ft_note}")
        # Seed the canonical checkpoint with the loaded weights when resuming from a
        # DIFFERENT file (e.g. fine-tuning a prior model onto new data). This makes
        # the best-so-far checkpoint exist immediately, so the end-of-run reload of
        # the best-on-TEST model finds a file even if no epoch improves on the loaded
        # baseline. Skipped when the source IS the canonical checkpoint (it already
        # holds these weights -- avoids re-writing a file the other ranks are reading).
        if is_main and Path(resume_from).resolve() != Path(checkpoint_path).resolve():
            torch.save(estimator.state_dict(), str(checkpoint_path))

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

    # Early-stopping state (performance switch #2). epochs_no_improve counts
    # consecutive epochs with no TEST-loss improvement; the stop fires only once it
    # reaches early_stop_patience AND the LR has floored (see the check below).
    lr_min = PARAMETERS.inference.training.learning_rate_minimum
    epochs_no_improve = 0
    epochs_run = epochs   # actual epochs executed (< epochs iff early-stopped); for trimming
    if is_main and early_stop_patience > 0 and has_val:
        print(f"  Early stopping: ENABLED (patience={early_stop_patience} epoch(s) "
              f"flat at LR floor {lr_min:.2e})", flush=True)
    elif is_main and early_stop_patience > 0 and not has_val:
        print("  Early stopping: requested but INACTIVE (no TEST set; needs --test-tasks > 0)",
              flush=True)

    loop_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)   # reshuffle this rank's shard each epoch
        # ---- Training pass ------------------------------------------------
        model.train()
        batch_train_losses = []
        for b, (video_batch, theta_batch) in enumerate(train_loader, start=1):
            video_batch = video_batch.to(device)
            if gpu_normalize:
                video_batch = gpu_normalize_video(video_batch)
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
                    estimator, val_loader, device, verbose=False, collect=True,
                    gpu_normalize=gpu_normalize)
                epoch_test = _reduce_mean(val_mean)
                tld_shard = (t_task, t_sim, t_loss, t_theta)
            else:
                epoch_test = _reduce_mean(
                    compute_validation_loss(estimator, val_loader, device, verbose=False,
                                            gpu_normalize=gpu_normalize))
        else:
            epoch_test = float("nan")
        # Replay loss = eval-mode TRAIN loss (augmentation off), opt-in; each rank
        # measures its shard of the dedicated augmentation-off loader, then averaged
        # across ranks.
        epoch_replay = (_reduce_mean(compute_validation_loss(estimator, replay_loader, device, verbose=False, gpu_normalize=gpu_normalize))
                        if replay_loss else float("nan"))
        losses_train[epoch] = epoch_train
        losses_test[epoch] = epoch_test
        losses_replay[epoch] = epoch_replay

        scheduler.step(epoch_test if has_val else epoch_train)

        # ---- Checkpoint (rank 0 only; the selection metric is identical on all ranks) ----
        # epochs_no_improve is updated on EVERY rank (only the torch.save is rank-0
        # guarded) so the early-stop condition below is identical across ranks and
        # every rank breaks the loop on the same epoch -- no DDP desync.
        if has_val:
            if epoch_test < optimum_loss_test:
                optimum_loss_test = epoch_test
                epochs_no_improve = 0
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
            else:
                epochs_no_improve += 1
        elif is_main:
            torch.save(estimator.state_dict(), str(checkpoint_path))

        epoch_secs = time.time() - epoch_start
        elapsed = time.time() - loop_start
        eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)   # mean epoch time x remaining
        if is_main:
            replay_str = f"{epoch_replay:.5f}" if replay_loss else "off"
            print(
                f"Epoch {epoch + 1}|{epochs}    "
                f"train={epoch_train:.5f}    test={epoch_test:.5f}    "
                f"replay={replay_str}    lr={scheduler.get_last_lr()[0]:.2e}    "
                f"epoch={epoch_secs:.1f}s    elapsed={time.strftime('%H:%M:%S', time.gmtime(elapsed))}    "
                f"ETA={time.strftime('%H:%M:%S', time.gmtime(eta))}",
                flush=True,
            )

        # Early stop, gated on the LR floor: TEST loss flat for `patience` epochs
        # AND the scheduler bottomed out. All ranks see the same all-reduced
        # inputs, so the break is synchronized under DDP.
        if (has_val and early_stop_patience > 0
                and epochs_no_improve >= early_stop_patience
                and scheduler.get_last_lr()[0] <= lr_min * (1.0 + 1e-9)):
            epochs_run = epoch + 1
            if is_main:
                print(f"Early stop at epoch {epochs_run}/{epochs}: TEST loss flat for "
                      f"{epochs_no_improve} epoch(s) at the LR floor "
                      f"(lr={scheduler.get_last_lr()[0]:.2e} <= {lr_min:.2e}); "
                      f"skipping {epochs - epochs_run} remaining epoch(s).", flush=True)
            break

    # Trim per-epoch arrays to the epochs actually run (early stop leaves a NaN
    # tail that would break downstream reads); full run -> no-op.
    losses_train = losses_train[:epochs_run]
    losses_test = losses_test[:epochs_run]
    losses_replay = losses_replay[:epochs_run]

    # ---- Final model ------------------------------------------------------
    if has_val:
        # All ranks reload the best-on-TEST checkpoint (rank 0 wrote it; a barrier
        # guards the read) so every replica ends identical. Staged to CPU, then placed
        # onto each rank's own device by load_state_dict.
        if distributed:
            dist.barrier()
        estimator.load_state_dict(
            torch.load(str(checkpoint_path), map_location="cpu", weights_only=True))
        final_replay = _reduce_mean(compute_validation_loss(estimator, val_loader, device, verbose=False, gpu_normalize=gpu_normalize))
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

"""Training pipeline for the inference stage.

Wraps the simulation-based inference (SBI) training loop: loads (video, theta)
pairs from disk, builds train/val DataLoaders, runs an Adam+ReduceLROnPlateau
optimiser over an sbi posterior estimator, and saves the trained posterior to
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
        setup_training(...)          -- bundles loaders + optimiser + scheduler
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
import os
import pickle
import random
import time
from typing import Optional

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

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
            `__getitem__`. The flag is also exposed for runtime toggling
            (e.g. by `compute_validation_loss`, which temporarily disables
            augmentation when computing the replay loss).
    """

    def __init__(self,
                 tasks: int,
                 data_bank_root: Path,
                 timing_label: str,
                 compress: bool = True,
                 indices: Optional[np.ndarray] = None,
                 augment: bool = True,
                 split: str = "TRAIN"):
        video_paths = []
        theta_paths = []
        for task_alias in range(tasks):
            video_paths.append(
                str(PARAMETERS.paths.video_set_path(
                    task_alias, data_bank_root, timing_label, compress, split))
            )
            theta_paths.append(
                str(PARAMETERS.paths.theta_set_path(
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
        torch.cuda.set_device(index)   # bind this process's default CUDA device to its own GPU
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


def build_datasets(train_tasks: int,
                   data_bank_root: Path,
                   timing_label: str,
                   compress: bool = True,
                   test_tasks: int = 0
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
    )
    test_dataset = None
    if test_tasks > 0:
        test_dataset = VideoDataset(
            tasks=test_tasks, data_bank_root=data_bank_root,
            timing_label=timing_label, compress=compress,
            augment=False, split="TEST",
        )
    return train_dataset, test_dataset


def build_eval_dataset(eval_tasks: int,
                       data_bank_root: Path,
                       timing_label: str,
                       compress: bool = True) -> "VideoDataset":
    """Read-only dataset over the held-out EVAL namespace (final validation only).

    The EVAL namespace is never loaded by training; it exists solely for
    posterior validation (MAP recovery). Augmentation is off.
    """
    return VideoDataset(
        tasks=eval_tasks, data_bank_root=data_bank_root,
        timing_label=timing_label, compress=compress,
        augment=False, split="EVAL",
    )


def setup_training(estimator: nn.Module,
                   train_tasks: int,
                   data_bank_root: Path,
                   timing_label: str,
                   compress: bool = True,
                   batch_size: Optional[int] = None,
                   learning_rate: Optional[float] = None,
                   test_tasks: int = 0) -> dict:
    """Bundle DataLoaders + optimiser + scheduler + device into a dict for `train_loop`.

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

    Returns:
        Dict with keys: `estimator`, `train_loader`, `val_loader`,
        `optimizer`, `scheduler`, `device`. `val_loader` is None when
        `test_tasks == 0`. Ready to be passed to `train_loop`.
    """
    training_cfg = PARAMETERS.inference.training
    batch_size = batch_size or training_cfg.batch_size
    if learning_rate is None:
        learning_rate = training_cfg.learning_rate_minimum * training_cfg.learning_rate_maximum_factor

    device = get_device()
    estimator = estimator.to(device)

    train_dataset, test_dataset = build_datasets(
        train_tasks=train_tasks, data_bank_root=data_bank_root,
        timing_label=timing_label, compress=compress, test_tasks=test_tasks,
    )
    num_workers = PARAMETERS.machine.num_workers
    if num_workers <= 0:
        # Auto: half the available CPU cores (SLURM-job-aware via sched_getaffinity).
        # Set a positive num_workers in machine_profiles.toml to override this.
        cores = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        num_workers = max(1, cores // 2)
    # persistent_workers keeps the worker pool alive across epochs (avoids the
    # per-epoch re-spawn + stack re-import that otherwise dominates each epoch).
    persistent = num_workers > 0
    print(f"  DataLoader: num_workers={num_workers} (persistent_workers={persistent})", flush=True)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True,
                              persistent_workers=persistent)
    val_loader = None
    if test_dataset is not None:
        val_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=num_workers, pin_memory=True,
                                persistent_workers=persistent)

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
        "train_loader": train_loader,
        "val_loader": val_loader,
        "optimizer": optimizer,
        "scheduler": scheduler,
        "device": device,
    }


# =============================================================================
# Validation loss
# =============================================================================

def compute_validation_loss(estimator: nn.Module,
                            loader: DataLoader,
                            device: torch.device,
                            verbose: bool = False) -> float:
    """Compute mean loss over a loader in eval mode, with augmentation disabled.

    The estimator is set to eval mode (dropout disabled, batch-norm uses running
    statistics) and augmentation is temporarily disabled on the loader's dataset,
    so the loss reflects the augmentation-free distribution under the same network
    conditions as the test-set evaluation. The original `augment` flag is restored
    before return.

    Args:
        estimator: The posterior estimator. Set to eval mode internally.
        loader: DataLoader over (video, theta) pairs.
        device: Device to move batches to.
        verbose: If True, print the augmentation flag before and after.

    Returns:
        Mean loss across all batches, as a Python float.
    """
    original_augment = loader.dataset.augment
    if original_augment:
        loader.dataset.augment = False
    if verbose:
        print(f"Data augmentation? {loader.dataset.augment}")

    losses_per_batch = []
    estimator.eval()
    with torch.no_grad():
        for video_batch, theta_batch in loader:
            video_batch = video_batch.to(device)
            theta_batch = theta_batch.to(device)
            batch_loss = torch.mean(estimator.loss(theta_batch, condition=video_batch))
            losses_per_batch.append(batch_loss.item())

    if original_augment != loader.dataset.augment:
        loader.dataset.augment = original_augment
        if verbose:
            print(f"Data augmentation? {loader.dataset.augment}")

    return float(np.mean(losses_per_batch))


# =============================================================================
# Training loop
# =============================================================================

def train_loop(estimator: nn.Module,
               train_loader: DataLoader,
               val_loader: DataLoader,
               optimizer: optim.Optimizer,
               scheduler,
               device: torch.device,
               checkpoint_path: Path,
               epochs: Optional[int] = None,
               resurrect: bool = False,
               replay_loss: bool = False,
               verbose: bool = False) -> tuple:
    """Run the training loop with optimum-checkpoint tracking and optional RESURRECT.

    Args:
        estimator: The posterior estimator to train.
        train_loader, val_loader: DataLoaders for train and validation splits.
        optimizer: Already-constructed optimiser (e.g. AdamW).
        scheduler: Learning-rate scheduler with `.step(metric)` method
            (e.g. ReduceLROnPlateau on validation loss).
        device: Device to move batches and the estimator to.
        checkpoint_path: Path where the best-so-far estimator checkpoint is
            saved (and, in resurrect mode, loaded from). Built by the
            entry-point as ``PARAMETERS.paths.checkpoint_path(data_bank_root,
            timing.label)`` so the path encodes the active simulation timing.
        epochs: Number of epochs. Defaults to `PARAMETERS.inference.training.epochs`.
        resurrect: If True, load the checkpoint at `checkpoint_path` BEFORE
            training begins and report its replay validation loss as the
            starting baseline. Training then continues from those weights.
            The new optimum overwrites the same path.
        verbose: If True, print per-batch progress; otherwise only per-epoch summary.

    Returns:
        (losses_train, losses_test, losses_replay) — three 1D arrays of
        length `epochs` with per-epoch mean train, validation, and replay
        loss (replay = train loss recomputed with augmentation disabled).
    """
    if epochs is None:
        epochs = PARAMETERS.inference.training.epochs

    losses_train = np.full(shape=epochs, fill_value=np.nan)
    losses_test = np.full(shape=epochs, fill_value=np.nan)
    losses_replay = np.full(shape=epochs, fill_value=np.nan)

    has_val = val_loader is not None   # TEST namespace present → model selection
    optimum_loss_test = float("inf")
    if resurrect:
        estimator.load_state_dict(torch.load(str(checkpoint_path), weights_only=True))
        if has_val:
            optimum_loss_test = compute_validation_loss(
                estimator, val_loader, device, verbose=verbose)
            print(f"RESURRECT: loaded checkpoint from {checkpoint_path}; "
                  f"baseline test loss = {optimum_loss_test:.5f}")
        else:
            print(f"RESURRECT: loaded checkpoint from {checkpoint_path} "
                  f"(no test set; last-epoch checkpointing).")

    n_batches = len(train_loader)
    heartbeat = max(1, n_batches // 4)   # ~4 within-epoch progress lines per epoch
    n_train_videos = len(train_loader.dataset)
    n_test_videos = len(val_loader.dataset) if has_val else 0
    print(
        f"Training: {n_train_videos} train / {n_test_videos} test videos, "
        f"{n_batches} batch(es)/epoch, {epochs} epoch(s) on {device}. "
        f"The first batch triggers model compilation -- expect a delay before the first heartbeat.",
        flush=True,
    )

    loop_start = time.time()
    for epoch in range(epochs):
        epoch_start = time.time()
        # ---- Training pass ------------------------------------------------
        estimator.train()
        batch_train_losses = []
        for b, (video_batch, theta_batch) in enumerate(train_loader, start=1):
            video_batch = video_batch.to(device)
            theta_batch = theta_batch.to(device)
            optimizer.zero_grad()
            loss = torch.mean(estimator.loss(theta_batch, condition=video_batch))
            loss.backward()
            optimizer.step()
            batch_train_losses.append(loss.item())
            # Always-on within-epoch heartbeat (~4/epoch) so long epochs are not
            # silent on the HPC console; flush so it appears live in the .out.
            if b % heartbeat == 0 or b == n_batches:
                print(f"  epoch {epoch + 1}/{epochs}: batch {b}/{n_batches}  "
                      f"running train_loss={float(np.mean(batch_train_losses)):.5f}",
                      flush=True)

        # ---- Evaluation pass (model-selection signal; TEST namespace) -----
        epoch_train = float(np.mean(batch_train_losses))
        if has_val:
            estimator.eval()
            batch_test_losses = []
            with torch.no_grad():
                for video_batch, theta_batch in val_loader:
                    video_batch = video_batch.to(device)
                    theta_batch = theta_batch.to(device)
                    loss = torch.mean(estimator.loss(theta_batch, condition=video_batch))
                    batch_test_losses.append(loss.item())
            epoch_test = float(np.mean(batch_test_losses))
        else:
            epoch_test = float("nan")
        # Replay loss = the TRAIN-set loss recomputed under the SAME conditions
        # as the test loss -- model in eval mode (dropout disabled, batch-norm
        # running stats) and data augmentation disabled -- so train and test are
        # directly comparable (a clean overfitting / generalization signal; the
        # per-epoch "train" loss above is measured in train mode, so it is not).
        # Opt-in: a full extra pass over TRAIN every epoch, expensive on large
        # datasets, so skipped unless replay_loss=True.
        epoch_replay = (compute_validation_loss(estimator, train_loader, device, verbose=False)
                        if replay_loss else float("nan"))
        losses_train[epoch] = epoch_train
        losses_test[epoch] = epoch_test
        losses_replay[epoch] = epoch_replay

        scheduler.step(epoch_test if has_val else epoch_train)

        # ---- Checkpoint ---------------------------------------------------
        if has_val:
            # Keep the best-on-TEST checkpoint (clean model selection).
            if epoch_test < optimum_loss_test:
                optimum_loss_test = epoch_test
                torch.save(estimator.state_dict(), str(checkpoint_path))
        else:
            # No selection set: keep the latest epoch (last wins).
            torch.save(estimator.state_dict(), str(checkpoint_path))

        epoch_secs = time.time() - epoch_start
        elapsed = time.time() - loop_start
        eta = (elapsed / (epoch + 1)) * (epochs - epoch - 1)   # mean epoch time x remaining
        replay_str = f"{epoch_replay:.5f}" if replay_loss else "off"
        print(
            f"Epoch {epoch + 1}|{epochs}    "
            f"train={epoch_train:.5f}    test={epoch_test:.5f}    "
            f"replay={replay_str}    lr={scheduler.get_last_lr()[0]:.2e}    "
            f"epoch={epoch_secs:.1f}s    elapsed={time.strftime('%H:%M:%S', time.gmtime(elapsed))}    "
            f"ETA={time.strftime('%H:%M:%S', time.gmtime(eta))}",
            flush=True,
        )

    # ---- Final model ------------------------------------------------------
    if has_val:
        # Reload the best-on-TEST checkpoint as the final estimator.
        estimator.load_state_dict(torch.load(str(checkpoint_path), weights_only=True))
        final_replay = compute_validation_loss(estimator, val_loader, device, verbose=False)
        print(f"Final: optimum test loss = {optimum_loss_test:.5f}; "
              f"replay test loss = {final_replay:.5f}")
    else:
        # No selection set: the last-epoch weights are already in the estimator
        # and saved to the checkpoint. Validation is done separately on EVAL.
        print("Final: last-epoch checkpoint kept (no test set; "
              "validate on the EVAL namespace).")

    return losses_train, losses_test, losses_replay


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

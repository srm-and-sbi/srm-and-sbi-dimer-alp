"""Shared machinery for the Detector real-data stages (Experiment + Nuisance_DLI).

Two Detector steps read the same real recordings and run the same estimator over
them with the same overall shape:

    load the estimator  ->  discover the cells on disk  ->  read each recording and
    window it into model-length chunks  ->  run the estimator per chunk  ->  aggregate.

The Detector *Experiment* stage MAP-estimates each chunk and reports the inferred
imaging parameters per condition; the *Nuisance_DLI* analysis draws posterior samples
per chunk and pools them into the marginalized imaging distribution. Both are pooled
over whole cells, so the per-chunk work is embarrassingly parallel and the result is
order-independent.

That makes a single multi-GPU pattern serve both: launch one worker per GPU
(``torchrun``), split the flat ``(kind, cell)`` work list across workers round-robin by
rank (each worker owns whole cells), have each worker write its partial arrays as a
compressed ``_shard_*_of_*.npz`` next to the output, and then run a separate,
single-process, no-GPU ``--merge`` step that concatenates the shards into the final
product (a report for the Experiment; a cached pool + emitted spec for the Nuisance_DLI).
With one worker (``world_size == 1``) the sharding is a no-op: the single process does
all cells and produces the product directly, with no shard and no merge.

This module holds exactly the pieces both stages share -- cell discovery, per-cell
read+window, rank round-robin, the shard path, and the shard save/load/merge -- so the
two stages stay byte-for-byte identical in how they discover, window, shard, and merge.
The estimation itself (MAP vs. posterior-sample pool) stays in each stage.
"""

import re
from pathlib import Path

import numpy as np
import tifffile

from .io import convert_video_dtype


def discover_cells(experiment_dir, kind, span):
    """Return the sorted cell indices with a recording on disk for a given kind.

    Globs ``Experiment_{kind}_Cell_*_{span}S_RAW.tif`` under ``experiment_dir`` and
    parses the integer cell index out of each name. Shared by both Detector real-data
    stages so they see exactly the same recordings.
    """
    cells = []
    for path in experiment_dir.glob(f"Experiment_{kind}_Cell_*_{span}S_RAW.tif"):
        match = re.search(rf"Cell_(\d+)_{span}S_RAW", path.name)
        if match:
            cells.append(int(match.group(1)))
    return sorted(cells)


def read_cell_chunks(tif_path, n_frames, step_frames):
    """Read one recording and cut it into model-length windows.

    Loads the 16-bit raw ``.tif``, converts it to 8-bit (the model's input domain),
    and returns the list of ``(n_frames, H, W)`` uint8 windows stepped by
    ``step_frames`` (``1 s`` step -> maximal overlap; a step equal to the window ->
    non-overlapping tiling). Identical windowing in both stages.
    """
    raw = tifffile.imread(str(tif_path))                 # (frames, H, W) uint16
    video8 = convert_video_dtype(raw, bits_from=16, bits_to=8)
    return [video8[start:start + n_frames]
            for start in range(0, video8.shape[0] - n_frames + 1, step_frames)]


def shard_by_rank(items, topo):
    """This worker's round-robin slice of ``items`` (``world_size == 1`` -> all of them).

    Splits a flat work list across workers by ``i % world_size == rank``; each worker
    gets a disjoint subset and together they cover every item exactly once.
    """
    return [w for i, w in enumerate(items) if i % topo.world_size == topo.rank]


def shard_path(out_dir, rank, world_size):
    """Path of one worker's partial-array shard in a multi-GPU sharded run."""
    return Path(out_dir) / f"_shard_{rank:02d}_of_{world_size:02d}.npz"


def save_shard(out_dir, topo, arrays, *, count):
    """Write this worker's partial arrays as a shard; return the path, or ``None``.

    Creates ``out_dir`` and, when ``count > 0``, saves ``arrays`` (a ``{name: array}``
    dict) to this rank's shard path as a compressed ``.npz`` and returns it. When
    ``count == 0`` (this worker drew no work) it writes no shard and returns ``None``,
    so the ``--merge`` step simply sees fewer files instead of an empty-array shard.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if count == 0:
        return None
    path = shard_path(out_dir, topo.rank, topo.world_size)
    np.savez_compressed(str(path), **arrays)
    return path


def load_shards(out_dir):
    """Sorted list of the ``_shard_*_of_*.npz`` files a sharded run wrote in ``out_dir``."""
    return sorted(Path(out_dir).glob("_shard_*_of_*.npz"))


def merge_shard_arrays(shard_paths, *, concat_keys, first_keys=(),
                       optional_concat_keys=()):
    """Combine per-worker shard arrays into one merged ``{name: array}`` dict.

    Reads every shard in ``shard_paths`` and, treating ``concat_keys[0]`` as the primary
    key, skips any shard whose primary array is empty (a defensive no-op -- a worker that
    drew no work writes no shard at all). For every used (non-empty) shard it:

    - appends each ``concat_keys`` array (concatenated on axis 0 in the result);
    - captures each ``first_keys`` value from the first used shard (taken as-is -- these
      are per-run constants such as the condition list, identical across shards);
    - appends each ``optional_concat_keys`` array when present, and marks the key absent
      if any used shard lacks it.

    Returns ``(merged, n_used)`` where ``merged`` has each concat key concatenated, each
    first key from the first used shard, and each optional concat key concatenated only
    if it was present in EVERY used shard (otherwise the key is omitted from ``merged``).
    Raises ``ValueError`` if ``shard_paths`` is empty or every shard was empty.
    """
    if not shard_paths:
        raise ValueError("merge_shard_arrays: no shard files were provided.")
    primary = concat_keys[0]
    collected = {key: [] for key in concat_keys}
    optional = {key: [] for key in optional_concat_keys}
    optional_present_all = {key: True for key in optional_concat_keys}
    firsts = {}
    n_used = 0
    for shard in shard_paths:
        with np.load(str(shard)) as data:
            if data[primary].shape[0] == 0:
                continue   # defensive: a zero-item shard contributes nothing
            n_used += 1
            for key in concat_keys:
                collected[key].append(data[key])
            if n_used == 1:
                for key in first_keys:
                    firsts[key] = data[key]
            for key in optional_concat_keys:
                if key in data:
                    optional[key].append(data[key])
                else:
                    optional_present_all[key] = False
    if n_used == 0:
        raise ValueError("merge_shard_arrays: every shard was empty (no items).")
    merged = {key: np.concatenate(collected[key], axis=0) for key in concat_keys}
    for key in first_keys:
        merged[key] = firsts[key]
    for key in optional_concat_keys:
        if optional_present_all[key] and optional[key]:
            merged[key] = np.concatenate(optional[key], axis=0)
    return merged, n_used


def assert_consistent_shard_set(shard_paths):
    """Guard a shard set against a stale-plus-fresh mix before merging.

    Each shard is named ``_shard_{rank}_of_{world_size}.npz``; one clean sharded run
    writes at most ``world_size`` shards, all carrying the SAME ``world_size``. If a run
    crashes mid-write and a later run with a DIFFERENT worker count writes into the same
    directory, the directory holds shards from two runs and merging them would silently
    concatenate a stale partial with the fresh result. This raises ``ValueError`` when the
    shards do not all share one ``world_size``, or when there are more shards than that
    ``world_size`` -- so the merge fails loudly instead of producing a contaminated result.
    A same-``world_size`` rerun overwrites its predecessor's shards by identical filename,
    so this passes (no contamination). Returns the common ``world_size``.
    """
    world_sizes = set()
    for path in shard_paths:
        match = re.search(r"_shard_(\d+)_of_(\d+)\.npz$", Path(path).name)
        if match is None:
            raise ValueError(f"unrecognized shard filename {Path(path).name!r}.")
        world_sizes.add(int(match.group(2)))
    if len(world_sizes) != 1:
        raise ValueError(
            f"shards carry inconsistent world_size {sorted(world_sizes)} (a stale-plus-fresh mix).")
    (world_size,) = world_sizes
    if len(shard_paths) > world_size:
        raise ValueError(
            f"{len(shard_paths)} shard files but world_size={world_size} (stale shards from a prior run).")
    return world_size

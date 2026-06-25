"""File I/O for all 3 entry-point scripts.

- load_data: transparently loads .zarr / .npy / .npz arrays.
- save_video_set, save_theta_set: write .npy or .npz buffers.
- convert_video_dtype: rescale pixel intensities between bit depths.

Note: incremental .zarr writes (slice-by-slice during simulation) are handled
inline in entry-point scripts, not by these helpers — via zarr.open at the top
of the simulation loop followed by array[i] = slice assignments.
"""

from pathlib import Path
from typing import Union

import numpy as np
import zarr
from skimage import exposure


# =============================================================================
# Loading
# =============================================================================

def load_data(path: Union[str, Path]):
    """Load array data from .zarr (lazy), .npy, or .npz transparently.

    Args:
        path: Path to the data file (string or pathlib.Path).

    Returns:
        For .zarr: a lazy zarr.Array (only requested chunks are decompressed on access).
        For .npy: a numpy.ndarray (memory-mapped via mmap_mode="r").
        For .npz: the first ndarray in the archive.
    """
    path_str = str(path)
    if path_str.endswith(".zarr"):
        return zarr.open(store=path_str, mode="r")
    data = np.load(file=path_str, mmap_mode="r")
    if isinstance(data, np.lib.npyio.NpzFile):
        return data[data.files[0]]  # first array in the .npz archive
    return data


# =============================================================================
# Saving (.npy and .npz only; .zarr handled inline by callers)
# =============================================================================

def save_video_set(path: Union[str, Path], video_set: np.ndarray,
                   compress: bool = True) -> None:
    """Save a video-set array to disk as .npz (compressed) or .npy.

    See module docstring for the .zarr handling note.
    """
    path_str = str(path)
    if compress:
        np.savez_compressed(file=path_str, arr=video_set, allow_pickle=False)
    else:
        np.save(file=path_str, arr=video_set, allow_pickle=False)


def save_theta_set(path: Union[str, Path], theta_set: np.ndarray,
                   compress: bool = True) -> None:
    """Save a theta-set array to disk as .npz (compressed) or .npy.

    See module docstring for the .zarr handling note.
    """
    path_str = str(path)
    if compress:
        np.savez_compressed(file=path_str, arr=theta_set, allow_pickle=False)
    else:
        np.save(file=path_str, arr=theta_set, allow_pickle=False)


# =============================================================================
# Bit-depth conversion
# =============================================================================

_DTYPE_FOR_BITS = {8: np.uint8, 16: np.uint16}


def convert_video_dtype(video: np.ndarray,
                        bits_from: int = 16,
                        bits_to: int = 8) -> np.ndarray:
    """Rescale pixel intensities between bit depths via skimage.exposure.

    Args:
        video: input array of any shape (typically 3D: (n_frames, H, W)).
        bits_from: source bit depth (e.g., 16 for uint16 input).
        bits_to: target bit depth (e.g., 8 for uint8 output).

    Returns:
        Array of shape == video.shape, dtype matching the target bit depth.

    Note: the conversion is global (in_range/out_range fixed); applies to the
    whole array at once (vectorised over all frames, no per-frame loop).
    """
    if bits_to not in _DTYPE_FOR_BITS:
        raise ValueError(
            f"bits_to={bits_to} not supported; must be one of {sorted(_DTYPE_FOR_BITS)}."
        )
    rescaled = exposure.rescale_intensity(
        video,
        in_range=(0, 2**bits_from - 1),
        out_range=(0, 2**bits_to - 1),
    )
    return np.round(rescaled).astype(_DTYPE_FOR_BITS[bits_to])

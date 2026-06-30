"""DLI-stage diagnostics and interactive plotting.

Functions:
    extract_pixel_stats(video, convert)
        Print pixel-value quantiles (ADU and photons), fit a lognormal
        distribution to non-zero photon counts, and plot a histogram with
        quantile markers.
    animate_video(video)
        Matplotlib FuncAnimation playback of a single video.
    plot_transitions(Q, P, ...)
        Paired heatmaps of the CTMC generator Q and the DTMC stochastic
        matrix P.
    figure_sample_frame(video, frame_index)
        Build and return a headless Figure showing one frame of a video,
        without a display backend (for diagnostic reports).
    figure_pixel_histogram(video, convert)
        Build and return a headless Figure of the non-zero pixel-value
        distribution, without a display backend (for diagnostic reports).

Matplotlib + IPython are imported lazily inside each function so HPC headless
runs that never plot don't pay the import cost. All functions are
interactive-only; production scripts should not invoke them.
"""

from typing import Optional

import numpy as np


def extract_pixel_stats(video: np.ndarray, convert: float = 4.78) -> None:
    """Print pixel-value quantiles and plot a fitted-lognormal histogram.

    Reports two quantile lines: one in ADU (raw detector counts) and one in
    photons (ADU divided by the conversion factor). Then fits a lognormal
    distribution to non-zero photon counts and shows a histogram with the
    fit overlay and quantile markers.

    Args:
        video: Pixel-value array of any shape (typically a video; the
            function flattens before computing quantiles).
        convert: ADU-per-photon conversion factor (the EMCCD's `kappa_c`).
            Default 4.78 matches the typical detector configuration.
    """
    import matplotlib.pyplot as plt
    from IPython import get_ipython
    from scipy.stats import lognorm

    get_ipython().run_line_magic("matplotlib", "inline")

    quantile_levels = np.array([0, 0.05, 0.25, 0.5, 0.75, 0.95, 1])
    adu_quantiles = [int(q) for q in np.quantile(a=video, q=quantile_levels)]
    photon_video = np.round(video / convert).flatten()
    photon_quantiles = [int(q) for q in np.quantile(a=photon_video, q=quantile_levels)]
    quantile_labels = [f"{int(100 * q)}%" for q in quantile_levels]
    print(f"Video shape: {video.shape}, dtype: {video.dtype}, ADU/photon: {convert}")
    print("Pixel quantiles (ADU):     " + "   ".join(
        f"{lab}={val}" for lab, val in zip(quantile_labels, adu_quantiles)))
    print("Pixel quantiles (Photons): " + "   ".join(
        f"{lab}={val}" for lab, val in zip(quantile_labels, photon_quantiles)))

    # Histogram + lognormal fit for non-zero photon counts.
    photon_positive = photon_video[photon_video > 0]
    if len(photon_positive) == 0:
        print("No non-zero photons; skipping histogram.")
        return
    fit_shape, fit_loc, fit_scale = lognorm.fit(data=photon_positive, floc=0)
    plt.figure(figsize=(12, 9))
    plt.hist(x=photon_positive, bins=250, density=True, alpha=0.5,
             color="gray", label="Photon data")
    x_axis = np.linspace(0, photon_quantiles[-1], 1000)
    plt.plot(x_axis, lognorm.pdf(x_axis, fit_shape, loc=fit_loc, scale=fit_scale),
             "r-", lw=2, label="Lognormal fit")
    for level, q in zip(quantile_levels, photon_quantiles):
        color = "magenta" if level in (0.25, 0.5, 0.75) else "blue"
        plt.axvline(x=q, color=color, linestyle="--", lw=1.5)
    plt.text(
        0.75, 0.75,
        "Photon quantiles:\n" + "\n".join(
            f"{l*100:.0f}%: {q}" for l, q in zip(quantile_levels, photon_quantiles)
        ),
        transform=plt.gca().transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )
    plt.text(
        0.75, 0.25,
        f"Lognormal fit:\nshape (sigma): {fit_shape:.3f}\nloc: {fit_loc:.3f}\n"
        f"scale (exp(mu)): {fit_scale:.3f}",
        transform=plt.gca().transAxes, fontsize=10, verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="white", alpha=0.5),
    )
    plt.xlabel("Photon counts (per pixel)")
    plt.ylabel("Probability density")
    plt.title("Lognormal fit to non-zero photon counts")
    plt.legend()
    plt.xlim(0, 1500)
    plt.show()


def animate_video(video: np.ndarray):
    """Return a matplotlib FuncAnimation that plays back a video.

    Args:
        video: 3D array of shape `(height, width, n_frames)`. (Note: the DLI
            pipeline returns intensity tensors with time as the last axis;
            consumers should follow the same convention when passing here.)

    Returns:
        A `matplotlib.animation.FuncAnimation` instance. To display
        interactively in IPython, return this object to a notebook cell;
        the qt backend is set so the animation plays in an external window.

    Note:
        The animation object MUST be returned and kept alive; otherwise
        matplotlib stops the animation as soon as the function returns.
    """
    import matplotlib.animation as animation
    import matplotlib.pyplot as plt
    from IPython import get_ipython

    get_ipython().run_line_magic("matplotlib", "qt")
    fig, ax = plt.subplots()
    image = ax.imshow(video[:, :, 0], cmap="magma", interpolation="none", origin="lower")
    ax.set_title("Frame 0")

    def update(frame_index):
        image.set_data(video[:, :, frame_index])
        ax.set_title(f"Frame {frame_index}")
        return [image]

    anim = animation.FuncAnimation(
        fig, update, frames=range(video.shape[-1]),
        interval=20, blit=True,  # 20 ms interval = 50 fps
    )
    plt.show()
    return anim


def plot_transitions(Q: np.ndarray,
                     P: np.ndarray,
                     fig_size: tuple = (16, 9),
                     font_size: int = 17,
                     color_map: str = "viridis",
                     format_Q: str = ".2F",
                     format_P: str = ".2F",
                     color_bar: bool = True,
                     color_bar_dit: Optional[dict] = None,
                     fig_title: Optional[str] = None) -> None:
    """Display the CTMC generator Q and DTMC stochastic matrix P as paired heatmaps.

    Both matrices are square, indexed by emitter brightness state (state 0 =
    photobleached). The heatmaps annotate each cell with its numeric value,
    formatted via the `format_Q` / `format_P` format strings.

    Args:
        Q: CTMC generator matrix.
        P: DTMC stochastic matrix.
        fig_size: Figure size in inches (width, height).
        font_size: Font size for labels and titles.
        color_map: Matplotlib color map name.
        format_Q, format_P: Format strings for in-cell annotations.
        color_bar: Whether to show color bars.
        color_bar_dit: Optional kwargs dict for seaborn's `cbar_kws`.
        fig_title: Optional figure-level title.
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    fig, (axe_Q, axe_P) = plt.subplots(1, 2, figsize=fig_size)
    sns.heatmap(data=Q, cmap=color_map, annot=True, fmt=format_Q,
                cbar=color_bar, cbar_kws=color_bar_dit, square=True, ax=axe_Q)
    axe_Q.set_xlabel(xlabel=rf"TO STATE{' ' * 4}$\mathbf{{j}}$", fontsize=font_size)
    axe_Q.set_ylabel(ylabel=rf"FROM STATE{' ' * 4}$\mathbf{{i}}$", fontsize=font_size)
    axe_Q.set_title(label=rf"Transition-Rate Matrix{' ' * 4}$\mathbf{{Q}}$",
                    fontsize=font_size)
    sns.heatmap(data=P, vmin=0, vmax=1, cmap=color_map, annot=True, fmt=format_P,
                cbar=color_bar, cbar_kws=color_bar_dit, square=True, ax=axe_P)
    axe_P.set_xlabel(xlabel=rf"TO STATE{' ' * 4}$\mathbf{{j}}$", fontsize=font_size)
    axe_P.set_ylabel(ylabel=rf"FROM STATE{' ' * 4}$\mathbf{{i}}$", fontsize=font_size)
    axe_P.set_title(label=rf"Transition-Probability Matrix{' ' * 4}$\mathbf{{P}}$",
                    fontsize=font_size)
    if fig_title:
        fig.suptitle(t=fig_title, fontsize=font_size)
    plt.tight_layout()
    plt.show()


# =============================================================================
# Headless figure builders (for --debug-dump reports)
# =============================================================================
# These build and RETURN a matplotlib Figure without calling plt.show() or
# get_ipython(), so they are safe on a headless server (no display). The
# DiagnosticReporter saves the returned Figure to PNG via Figure.savefig.
# They are deliberately separate from the interactive functions above, which
# remain notebook-only.


def load_video_set(path):
    """Open a saved video set for indexing as ``[sim, frame]``.

    Args:
        path: Path to a ``.zarr`` (lazy) or ``.npy`` / ``.npz`` (eager) video
            set, shaped ``(n_sims, n_frames, height, width)``.

    Returns:
        A handle supporting ``handle[sim, frame]`` indexing. For ``.zarr`` the
        handle is a lazy ``zarr`` array (only the indexed frame is read into
        memory) -- the right choice for the interactive scrubber notebook over
        large video sets. For ``.npy`` / ``.npz`` the whole array is loaded.
    """
    from pathlib import Path

    p = Path(path)
    if p.suffix == ".zarr" or p.is_dir():
        import zarr
        return zarr.open(str(p), mode="r")
    import numpy as _np
    loaded = _np.load(str(p))
    # .npz returns an archive; take the first stored array.
    if hasattr(loaded, "files"):
        return loaded[loaded.files[0]]
    return loaded


def figure_sample_frame(video: np.ndarray, frame_index: Optional[int] = None):
    """Build a headless Figure showing one frame of a video.

    Args:
        video: Array shaped ``(n_frames, height, width)`` (storage convention).
        frame_index: Frame to display. Defaults to the middle frame.

    Returns:
        A ``matplotlib.figure.Figure`` (not shown; caller saves or displays).
    """
    from matplotlib.figure import Figure

    n_frames = video.shape[0]
    if frame_index is None:
        frame_index = n_frames // 2
    frame_index = int(max(0, min(frame_index, n_frames - 1)))

    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    image = ax.imshow(video[frame_index], cmap="magma",
                      interpolation="none", origin="lower")
    ax.set_title(f"Frame {frame_index} / {n_frames - 1}")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    return fig


def figure_pixel_histogram(video: np.ndarray, convert: float = 4.78):
    """Build a headless Figure: histogram of non-zero pixel values.

    A lightweight, dependency-free counterpart to ``extract_pixel_stats``
    (no lognormal fit, no IPython): just the distribution of non-zero pixel
    intensities, for a quick visual sanity check of a rendered video.

    Args:
        video: Pixel-value array of any shape (flattened internally).
        convert: ADU-per-photon conversion factor, annotated in the title.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    from matplotlib.figure import Figure

    flat = np.asarray(video).reshape(-1)
    nonzero = flat[flat > 0]

    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    if nonzero.size:
        ax.hist(nonzero, bins=200, color="gray", alpha=0.7)
        ax.set_yscale("log")
    else:
        ax.text(0.5, 0.5, "no non-zero pixels", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("Pixel value (ADU)")
    ax.set_ylabel("Count (log)")
    ax.set_title(f"Non-zero pixel-value distribution (ADU/photon={convert})")
    return fig

"""RDS-stage diagnostics and headless figure builders.

Diagnostic figures over RDS-stage outputs (.h5 particle trajectories), used by
the ``--debug-dump`` mode of the RDS entry-point. Each builder constructs and
RETURNS a ``matplotlib.figure.Figure`` without calling ``plt.show()`` -- safe
on a headless server (the DiagnosticReporter saves the Figure to PNG).

Functions:
    figure_reaction_events(reaction_counts)
        Bar chart of total event counts per reaction channel.
    figure_position_heatmap(positions_xy, bins)
        2D spatial-occupancy heatmap of particle (x, y) positions.

Candidate future functions (not implemented yet):
    - plot_particle_count_traces: counts of A, B, C species over time.
    - plot_mean_squared_displacement: MSD curves per species.
"""

import numpy as np


def figure_reaction_events(reaction_counts):
    """Build a headless bar chart of total reaction events per channel.

    Args:
        reaction_counts: Mapping of reaction-name -> per-frame event-count
            array (as returned by ReaDDy's reaction-counts observable).

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    from matplotlib.figure import Figure

    names = list(reaction_counts.keys())
    totals = [int(np.sum(counts)) for counts in reaction_counts.values()]

    fig = Figure(figsize=(8, 5))
    ax = fig.add_subplot(111)
    ax.bar(range(len(names)), totals, color="steelblue")
    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, rotation=30, ha="right")
    ax.set_ylabel("Total events")
    ax.set_title("Reaction events over the trajectory")
    fig.tight_layout()
    return fig


def figure_position_heatmap(positions_xy: np.ndarray, bins: int = 128):
    """Build a headless 2D occupancy heatmap of particle (x, y) positions.

    Args:
        positions_xy: Array shaped ``(n_points, 2)`` of x, y coordinates. NaN
            rows (absent particles) should be removed by the caller.
        bins: Number of histogram bins per axis.

    Returns:
        A ``matplotlib.figure.Figure``.
    """
    from matplotlib.figure import Figure

    fig = Figure(figsize=(6, 6))
    ax = fig.add_subplot(111)
    positions_xy = np.asarray(positions_xy)
    if positions_xy.size:
        counts, _, _, image = ax.hist2d(
            positions_xy[:, 0], positions_xy[:, 1], bins=bins, cmap="magma",
        )
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.text(0.5, 0.5, "no positions", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect("equal")
    ax.set_title("Particle spatial occupancy (all frames, all species)")
    return fig

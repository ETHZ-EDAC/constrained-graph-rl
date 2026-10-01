"""Module for all plotting of graphs and animations."""

# Third-party
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from logging_mod.logger import get_logger
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from pathlib import Path

logger = get_logger()


def plot_graph(
    adjacency: jnp.ndarray,
    vertex_positions: jnp.ndarray,
    ax=None,
    fig=None,
    *,
    pretty: bool = True,
    save_path: str | Path | None = None,
    save_dpi: int | None = None,
    save_pdf: bool = True,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the graph.

    Parameters
    ----------
    adjacency : jnp.ndarray (N, N): array representing adjacency matrix, padded with zero-rows/cols.
    vertex_positions : jnp.ndarray (N, 3): array of [x, y, z] coordinates, padded.
    pretty : bool: If True (default), use a cleaner, colorful, undirected style.
    save_path : str | Path | None: Optional path to save the figure.
    save_dpi : int | None: Optional DPI override for saving.
    save_pdf : bool: If True (default), also save a PDF alongside save_path.

    Returns
    -------
    fig : matplotlib.figure.Figure: The matplotlib figure object.
    ax : matplotlib.axes.Axes: The matplotlib axes object.
    """
    # Convert to NumPy for plotting
    adj = np.asarray(adjacency)
    vpos = np.asarray(vertex_positions[:, :2])

    if pretty:
        adj_undirected = np.maximum(adj, adj.T)
        np.fill_diagonal(adj_undirected, 0)
        used = np.any(adj_undirected != 0, axis=1)
        rows, cols = np.nonzero(np.triu(adj_undirected, k=1))
    else:
        # Determine which vertices are actually used
        used = np.any(adj != 0, axis=1) | np.any(adj != 0, axis=0)  # shape (N,)
        # Extract edges (i -> j)
        rows, cols = np.nonzero(adj)

    def _render(ax, *, show_grid: bool, show_axes: bool) -> None:
        if pretty:
            segments = []
            for i, j in zip(rows, cols):
                if not (used[i] and used[j]):
                    continue
                xi, yi = vpos[i]
                xj, yj = vpos[j]
                segments.append([(xi, yi), (xj, yj)])
            if segments:
                edge_collection = LineCollection(
                    segments,
                    colors="black",
                    linewidths=1.0,
                    alpha=0.7,
                    zorder=1,
                )
                ax.add_collection(edge_collection)

            vp_used = vpos[used]
            ax.scatter(
                vp_used[:, 0],
                vp_used[:, 1],
                s=90,
                c="#8ecae6",
                edgecolors="white",
                linewidths=1.0,
                zorder=2,
            )

            ax.set_aspect("equal")
            ax.set_facecolor("white")

            if vp_used.size:
                x_min, x_max = vp_used[:, 0].min(), vp_used[:, 0].max()
                y_min, y_max = vp_used[:, 1].min(), vp_used[:, 1].max()
                pad_x = 0.05 * max(1e-6, x_max - x_min)
                pad_y = 0.05 * max(1e-6, y_max - y_min)
                ax.set_xlim(x_min - pad_x, x_max + pad_x)
                ax.set_ylim(y_min - pad_y, y_max + pad_y)

            if show_axes:
                if show_grid:
                    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.4)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_title(f"Graph Size = {int(used.sum())}")
            else:
                ax.set_axis_off()
        else:
            # Draw each directed edge as an arrow
            for i, j in zip(rows, cols):
                if not (used[i] and used[j]):
                    continue
                xi, yi = vpos[i]
                xj, yj = vpos[j]

                color = "k"
                arrow = FancyArrowPatch(
                    (xi, yi),
                    (xj, yj),
                    arrowstyle="->",
                    mutation_scale=10,
                    linewidth=1.0,
                    color=color,
                    shrinkA=5,
                    shrinkB=5,
                )
                ax.add_patch(arrow)

            # Scatter the active vertices
            vp_used = vpos[used]
            indices = np.nonzero(used)[0]
            ax.scatter(vp_used[:, 0], vp_used[:, 1], s=70, facecolors="white", edgecolors="k", zorder=2)

            # Annotate each vertex with its index
            for idx in indices:
                x, y = vpos[idx]
                ax.text(x, y, str(idx), fontsize=10, ha="center", va="center", zorder=3)

            if show_axes:
                if show_grid:
                    ax.grid(True, linestyle="--", linewidth=0.5)
                ax.set_xlabel("X")
                ax.set_ylabel("Y")
                ax.set_xticks(np.linspace(vpos[:, 0].min(), vpos[:, 0].max(), 6))
                ax.set_yticks(np.linspace(vpos[:, 1].min(), vpos[:, 1].max(), 6))
            else:
                ax.set_axis_off()

            ax.set_aspect("equal")

    if ax is None:
        fig = plt.figure(figsize=(6, 6), dpi=199 if not pretty else 150)
        ax = fig.add_subplot(111)
    if fig is None:
        fig = ax.figure

    _render(ax, show_grid=True, show_axes=True)

    fig_no_axes = plt.figure(figsize=(6, 6), dpi=199 if not pretty else 150)
    ax_no_axes = fig_no_axes.add_subplot(111)
    _render(ax_no_axes, show_grid=False, show_axes=False)

    if save_path is not None:
        save_path = Path(save_path)
        dpi = save_dpi if save_dpi is not None else (300 if pretty else 199)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        if save_pdf:
            pdf_path = save_path.with_suffix(".pdf")
            fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
        no_axes_path = save_path.with_name(f"{save_path.stem}_no_axes{save_path.suffix}")
        fig_no_axes.savefig(no_axes_path, dpi=dpi, bbox_inches="tight")
        if save_pdf:
            pdf_no_axes_path = no_axes_path.with_suffix(".pdf")
            fig_no_axes.savefig(pdf_no_axes_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return fig_no_axes, ax

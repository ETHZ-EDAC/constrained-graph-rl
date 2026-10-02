from __future__ import annotations
from typing import Optional, Literal, Tuple, Dict
from pathlib import Path
from dataclasses import dataclass
import glob

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import torch
from torch import Tensor
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx, from_networkx

from graph_rl.ppo.metrics import compute_metrics_from_data


@dataclass
class GraphGeneratorConfig:
    """Configuration for graph generation with sensible defaults."""

    # Common parameters
    num_nodes: int = 60
    seed: Optional[int] = None

    # Erdős-Rényi parameters
    edge_probability: float = 0.15  # For G(n,p) model
    num_edges: Optional[int] = None  # For G(n,m) model, if None uses edge_probability

    # Barabási-Albert parameters
    num_attachments: int = 2  # Number of edges to attach from new node (m parameter)

    # Planar graph parameters
    planar_max_attempts: int = 1000  # Attempts to generate planar graph
    planar_min_edges: int = 5  # Minimum edges for planar graph

    # Lattice grid parameters
    grid_rows: Optional[int] = None  # Number of rows in grid (if None, computed from num_nodes)
    grid_cols: Optional[int] = None  # Number of columns in grid (if None, computed from num_nodes)
    grid_circular: bool = True  # Use circular instead of rectangular layout

    # Plantri parameters
    plantri_connectivity: int = 3  # Connectivity for plantri graphs (1, 2, 3, or 4)
    plantri_min_degree: Optional[int] = None  # Minimum degree constraint
    plantri_generate_single: bool = True  # If False, parse multiple plantri graphs
    plantri_max_graphs: Optional[int] = None  # Limit graphs parsed when generate_single=False

    # ERGM parameters
    ergm_edge_param: float = -2.0  # Log-odds for edge formation (negative = sparse)
    ergm_triangle_param: float = 0.1  # Triadic closure parameter
    ergm_star_param: float = 0.0  # k-star parameter (degree distribution)
    ergm_mcmc_steps: int = 1000  # MCMC steps for ERGM sampling
    ergm_burnin: int = 100  # Burn-in steps

    # Delaunay parameters
    delaunay_boundary: Literal["circle", "square"] = "circle"  # Domain boundary shape
    # Boltzmann parameters
    boltzmann_epsilon: float = 0.1
    # Planar ER parameters
    planar_er_edge_probability: Optional[float] = None

    # Layout parameters for planar embeddings
    layout_method: Literal["spring", "kamada_kawai", "planar", "circular", "random"] = "spring"
    layout_scale: float = 10.0  # Scale factor for node positions
    spring_k: Optional[float] = None  # Spring constant for spring layout
    spring_iterations: int = 50  # Iterations for spring layout

    # Additional graph properties
    ensure_connected: bool = False  # Ensure graph is connected
    allow_self_loops: bool = False  # Allow self-loops
    directed: bool = False  # Generate directed graphs


def load_adjlist_lst(path):
    dataset = []
    edges = []
    nodes_seen = set()

    def flush_graph():
        if not edges:
            return
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        num_nodes = max(nodes_seen) + 1
        x = torch.ones((num_nodes, 1))  # dummy node features
        dataset.append(Data(x=x, edge_index=edge_index))

    with open(path, "r") as f:
        for line in f:
            line = line.strip()

            # blank line → new graph
            if line == "":
                flush_graph()
                edges.clear()
                nodes_seen.clear()
                continue

            # parse: "u: v1 v2 v3"
            src_str, dst_str = line.split(":")
            src = int(src_str) - 1  # 1-based → 0-based
            neighbors = map(lambda x: int(x) - 1, dst_str.split())

            nodes_seen.add(src)
            for dst in neighbors:
                nodes_seen.add(dst)
                edges.append([src, dst])

    # last graph
    flush_graph()

    return dataset


def load_or_create_dataset(
    lst_path: str,
    layout: str = "spring",
) -> list[Data]:
    """
    Load dataset from .pt cache if it exists, otherwise create it from .lst file.
    """
    lst_path = Path(lst_path)

    # Determine .pt cache file path (same directory, same name with .pt extension)
    pt_path = lst_path.with_suffix(".pt")

    # Try to load from .pt cache first
    if pt_path.exists():
        print(f"Loading cached dataset from {pt_path}...")
        dataset = load_dataset(pt_path)
        return dataset

    # Cache doesn't exist, create from .lst
    print(f"Cache not found. Loading from {lst_path} and processing...")

    # Load raw graphs
    dataset = load_adjlist_lst(lst_path)
    print(f"Loaded {len(dataset)} graphs")

    # Enrich with positions and edge attributes
    print(f"Enriching graphs with {layout} layout...")
    dataset = [enrich_data_with_positions(data, layout=layout) for data in dataset]
    print("Added positions and edge attributes")

    # Save to cache for next time
    print(f"Saving dataset cache to {pt_path}...")
    save_dataset(dataset, pt_path)

    return dataset


def add_node_positions(
    graph: nx.Graph,
    method: str = "spring",
    scale: float = 10.0,
    spring_k: Optional[float] = None,
    spring_iterations: int = 50,
    seed: Optional[int] = None,
) -> nx.Graph:
    """
    Add 2D node positions to a NetworkX graph.

    Args:
        graph: NetworkX graph
        method: Layout algorithm ('spring', 'kamada_kawai', 'planar', 'circular', 'random')
        scale: Scale factor for positions
        spring_k: Spring constant for spring layout (None = optimal)
        spring_iterations: Number of iterations for spring layout
        seed: Random seed for layout

    Returns:
        Graph with 'pos' attribute added to nodes
    """
    if method == "spring":
        pos = nx.spring_layout(graph, k=spring_k, iterations=spring_iterations, scale=scale, seed=seed)
    elif method == "kamada_kawai":
        pos = nx.kamada_kawai_layout(graph, scale=scale)
    elif method == "planar":
        if nx.is_planar(graph):
            pos = nx.planar_layout(graph, scale=scale)
        else:
            pos = nx.spring_layout(graph, scale=scale, seed=seed)
    elif method == "circular":
        pos = nx.circular_layout(graph, scale=scale)
    elif method == "random":
        pos = {node: scale * np.random.rand(2) for node in graph.nodes()}
    else:
        raise ValueError(f"Unknown layout method: {method}")

    nx.set_node_attributes(graph, pos, "pos")
    return graph


def networkx_to_pyg_data(
    graph: nx.Graph,
    add_edge_weights: bool = True,
    default_edge_weight: float = 1.0,
) -> Data:
    """
    Convert NetworkX graph to PyTorch Geometric Data object.

    Args:
        graph: NetworkX graph with 'pos' attribute on nodes
        add_edge_weights: Whether to add edge weights
        default_edge_weight: Default weight for edges without weight attribute

    Returns:
        PyTorch Geometric Data object
    """
    # Convert to PyG Data
    data = from_networkx(graph)

    # Extract positions
    if nx.get_node_attributes(graph, "pos"):
        positions = nx.get_node_attributes(graph, "pos")
        pos_array = torch.tensor(
            [positions[i] for i in range(graph.number_of_nodes())],
            dtype=torch.float32,
        )
        data.pos = pos_array

    # Add edge weights - use actual edge_index size since from_networkx creates bidirectional edges
    if add_edge_weights:
        num_edges = data.edge_index.shape[1]
        data.edge_attr = torch.full((num_edges, 1), default_edge_weight, dtype=torch.float32)

    data.num_nodes = graph.number_of_nodes()

    return data


def set_seed(seed: Optional[int]) -> None:
    """Set random seeds for reproducibility."""
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)


def compute_sector_angles(edge_index: Tensor, pos: Tensor, num_nodes: int) -> Tensor:
    """Compute sector angles from edge_index and node positions."""
    max_degree = max((edge_index[0] == i).sum().item() for i in range(num_nodes))
    sector_angles = torch.zeros((num_nodes, max_degree + 1))

    for node in range(num_nodes):
        neighbors = edge_index[1][edge_index[0] == node].unique()
        degree = len(neighbors)
        sector_angles[node, 0] = degree

        if degree < 2:
            continue

        # Compute angles to neighbors
        diff = pos[neighbors] - pos[node]
        angles = torch.atan2(diff[:, 1], diff[:, 0])
        sorted_idx = torch.argsort(angles)
        sorted_angles = angles[sorted_idx]

        # Sector angles between consecutive neighbors
        sector_diff = torch.diff(torch.cat([sorted_angles, sorted_angles[:1] + 2 * np.pi]))
        sector_angles[node, 1 : degree + 1] = sector_diff

    return sector_angles


def ensure_graph_connected(graph: nx.Graph, seed: Optional[int] = None) -> nx.Graph:
    """
    Ensure graph is connected by adding minimum edges between components.

    Args:
        graph: NetworkX graph
        seed: Random seed

    Returns:
        Connected version of the graph
    """
    if nx.is_connected(graph):
        return graph

    rng = np.random.RandomState(seed)
    components = list(nx.connected_components(graph))

    # Connect components by adding edges between random nodes
    for i in range(len(components) - 1):
        node1 = rng.choice(list(components[i]))
        node2 = rng.choice(list(components[i + 1]))
        graph.add_edge(node1, node2)

    return graph


def enrich_data_with_positions(data: Data, layout: str = "spring") -> Data:
    """
    Add positions and edge attributes to a Data object using NetworkX layouts.

    Args:
        data: PyTorch Geometric Data object with edge_index
        layout: Layout algorithm - "spring", "circular", "planar", "kamada_kawai"

    Returns:
        Data object with pos and edge_attr attributes added
    """
    # Convert to NetworkX
    G = to_networkx(data, to_undirected=True)

    # Generate positions based on layout
    if layout == "spring":
        pos_dict = nx.spring_layout(G, seed=42)
    elif layout == "circular":
        pos_dict = nx.circular_layout(G)
    elif layout == "planar":
        if nx.is_planar(G):
            pos_dict = nx.planar_layout(G)
        else:
            pos_dict = nx.spring_layout(G, seed=42)
    elif layout == "kamada_kawai":
        pos_dict = nx.kamada_kawai_layout(G)
    else:
        raise ValueError(f"Unknown layout: {layout}")

    # Convert positions to tensor
    num_nodes = data.num_nodes
    pos = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for node_idx, (x, y) in pos_dict.items():
        pos[node_idx] = torch.tensor([x, y], dtype=torch.float32)

    # Add edge attributes (all ones for unit weights)
    num_edges = data.edge_index.shape[1]
    edge_attr = torch.ones((num_edges, 1), dtype=torch.float32)

    # Create enriched Data object
    enriched_data = Data(x=data.x, edge_index=data.edge_index, pos=pos, edge_attr=edge_attr, num_nodes=num_nodes)

    return enriched_data


def save_dataset(dataset, output_path):
    """
    Save a list of Data objects to a file.

    Args:
        dataset: List of PyTorch Geometric Data objects
        output_path: Path to save the dataset (will be created if doesn't exist)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(dataset, output_path)
    print(f"Saved {len(dataset)} graphs to {output_path}")


def load_dataset(input_path):
    """
    Load a dataset of Data objects from a file.

    Args:
        input_path: Path to the saved dataset

    Returns:
        List of PyTorch Geometric Data objects
    """
    dataset = torch.load(input_path, weights_only=False)
    print(f"Loaded {len(dataset)} graphs from {input_path}")
    return dataset


def evaluate_lst_datasets(
    lst_pattern: str,
    output_dir: Path,
    layout: str = "spring",
    compute_metrics: bool = True,
    num_samples_to_plot: int = 10,
) -> Dict[str, Dict[str, list]]:
    """
    Load and evaluate graphs from .lst files.

    Args:
        lst_pattern: Glob pattern for .lst files (e.g., "ext/hog_planar/*.lst")
        output_dir: Directory to save results
        layout: Layout algorithm for node positioning
        compute_metrics: Whether to compute metrics for all graphs
        num_samples_to_plot: Number of sample graphs to visualize per dataset

    Returns:
        Dictionary mapping dataset names to their metrics
    """
    lst_files = glob.glob(str(lst_pattern))

    if not lst_files:
        print(f"No .lst files found matching: {lst_pattern}")
        return {}

    print(f"Found {len(lst_files)} .lst files")
    print("=" * 80)
    print()

    # Store all metrics for each dataset
    all_dataset_metrics = {}

    # Store sample graphs for visualization
    all_sample_graphs = {}

    # Process each .lst file
    for lst_file in lst_files:
        input_path = Path(lst_file)
        dataset_name = input_path.stem
        output_filename = dataset_name + "_dataset.pt"
        output_path = output_dir / output_filename

        print(f"Processing {input_path.name}...")
        print("-" * 80)

        # Load adjacency list
        print(f"Loading adjacency list from {input_path}...")
        dataset = load_adjlist_lst(input_path)
        print(f"Loaded {len(dataset)} graphs")

        # Enrich with positions
        print(f"Enriching graphs with {layout} layout...")
        dataset = [enrich_data_with_positions(data, layout=layout) for data in dataset]
        print("Added positions and edge attributes to all graphs")

        # Store sample graphs for visualization
        if len(dataset) > 0:
            num_to_sample = min(num_samples_to_plot, len(dataset))
            for i in range(num_to_sample):
                sample_name = f"{dataset_name}_{i + 1}"
                all_sample_graphs[sample_name] = dataset[i]

        # Compute metrics for each graph in the dataset
        if compute_metrics:
            print(f"Computing metrics for {len(dataset)} graphs...")
            dataset_metrics = {}

            for i, data in enumerate(dataset):
                try:
                    # Compute sector angles for angular resolution metric
                    data.sector_angles = compute_sector_angles(data.edge_index, data.pos, data.num_nodes)

                    # Compute all metrics
                    metrics_dict = compute_metrics_from_data(data)
                    sample_metrics = metrics_dict.get("graph_metrics", {})

                    # Store each metric
                    for metric_name, value in sample_metrics.items():
                        if metric_name not in dataset_metrics:
                            dataset_metrics[metric_name] = []
                        dataset_metrics[metric_name].append(float(value))

                except Exception as e:
                    print(f"  Warning: Failed to compute metrics for graph {i}: {e}")
                    continue

            # Calculate statistics for this dataset
            if dataset_metrics:
                all_dataset_metrics[dataset_name] = dataset_metrics
                print(f"\nMetrics computed for {dataset_name}:")
                print("-" * 40)

                for metric_name in sorted(dataset_metrics.keys()):
                    values = dataset_metrics[metric_name]
                    if values:
                        mean_val = np.mean(values)
                        std_val = np.std(values)
                        min_val = np.min(values)
                        max_val = np.max(values)

                        print(
                            f"  {metric_name:30s}: μ={mean_val:7.4f}, σ={std_val:7.4f}, "
                            f"range=[{min_val:7.4f}, {max_val:7.4f}]"
                        )

        # Save dataset
        save_dataset(dataset, output_path)

        # Print sample
        if dataset:
            sample = dataset[0]
            print(f"\nSample graph from {input_path.name}:")
            print(f"  Nodes: {sample.num_nodes}")
            print(f"  Edges: {sample.edge_index.shape[1]}")
            if hasattr(sample, "pos") and sample.pos is not None:
                print(f"  Positions: {sample.pos.shape}")
            if hasattr(sample, "edge_attr") and sample.edge_attr is not None:
                print(f"  Edge attributes: {sample.edge_attr.shape}")

        print()

    # Create visualizations for sample graphs
    if all_sample_graphs:
        print("=" * 80)
        print("Visualizing sample graphs from datasets...")
        print("-" * 80)
        print(f"Plotting {len(all_sample_graphs)} sample graphs...")

        plot_benchmark_graphs(
            graphs=all_sample_graphs,
            figsize=(28, 20),
            node_size=150,
            node_color="lightblue",
            edge_color="gray",
            font_size=8,
            save_path=str(output_dir / "sample_graphs_from_datasets.png"),
        )
        print()

    return all_dataset_metrics


def plot_benchmark_graphs(
    graphs: Dict[str, Data],
    figsize: Tuple[int, int] = (20, 12),
    node_size: int = 300,
    node_color: str = "lightblue",
    edge_color: str = "gray",
    font_size: int = 8,
    save_path: Optional[str] = None,
) -> None:
    """
    Plot a collection of benchmark graphs using networkx.

    Args:
        graphs: Dictionary mapping graph names to Data objects
        figsize: Figure size (width, height)
        node_size: Size of nodes in the plot
        node_color: Color of nodes
        edge_color: Color of edges
        font_size: Font size for labels
        save_path: Optional path to save the figure
    """
    num_graphs = len(graphs)
    if num_graphs == 0:
        print("No graphs to plot")
        return

    # Calculate grid dimensions
    ncols = min(4, num_graphs)
    nrows = (num_graphs + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)

    # Ensure axes is always 2D array
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Flatten for easy iteration
    axes_flat = axes.flatten()

    for idx, (name, data) in enumerate(graphs.items()):
        ax = axes_flat[idx]

        # Convert PyG Data to NetworkX
        G = to_networkx(data, to_undirected=True)

        # Get positions from data if available
        if hasattr(data, "pos") and data.pos is not None:
            pos = {i: data.pos[i].numpy() for i in range(data.num_nodes)}
        else:
            pos = nx.spring_layout(G, seed=42)

        # Draw the graph
        nx.draw(
            G,
            pos=pos,
            ax=ax,
            node_size=node_size,
            node_color=node_color,
            edge_color=edge_color,
            with_labels=False,
            width=1.0,
        )

        # Add title with graph statistics
        num_nodes = G.number_of_nodes()
        num_edges = G.number_of_edges()
        avg_degree = 2 * num_edges / num_nodes if num_nodes > 0 else 0

        title = f"{name}\n{num_nodes} nodes, {num_edges} edges\navg degree: {avg_degree:.2f}"
        ax.set_title(title, fontsize=font_size, pad=10)
        ax.axis("off")

    # Hide unused subplots
    for idx in range(num_graphs, len(axes_flat)):
        axes_flat[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")

    plt.show()


def plot_metric_comparison(
    results: Dict[str, Dict[str, list]],
    figsize: Tuple[int, int] = (20, 12),
    save_path: Optional[str] = None,
) -> None:
    """
    Create box plots comparing metrics across generators.

    Args:
        results: Dictionary from evaluate_all_generators_with_metrics
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Get all unique metrics
    all_metrics = set()
    for gen_metrics in results.values():
        all_metrics.update(gen_metrics.keys())

    all_metrics = sorted(all_metrics)

    # Create subplots
    ncols = 3
    nrows = (len(all_metrics) + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1) if nrows * ncols > 1 else [axes]

    for idx, metric_name in enumerate(all_metrics):
        ax = axes[idx]

        # Collect data for this metric across generators
        plot_data = []
        labels = []

        for gen_name, gen_metrics in sorted(results.items()):
            if metric_name in gen_metrics:
                plot_data.append(gen_metrics[metric_name])
                labels.append(gen_name)

        if plot_data:
            ax.boxplot(plot_data, labels=labels)
            ax.set_title(metric_name.replace("_", " ").title())
            ax.tick_params(axis="x", rotation=45)
            ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(all_metrics), len(axes)):
        axes[idx].axis("off")

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Box plot saved to {save_path}")

    plt.show()


def plot_metric_heatmap(
    results: Dict[str, Dict[str, list]],
    use_mean: bool = True,
    normalize: bool = True,
    figsize: Tuple[int, int] = (14, 10),
    save_path: Optional[str] = None,
) -> None:
    """
    Create heatmap of metrics across generators.

    Args:
        results: Dictionary from evaluate_all_generators_with_metrics
        use_mean: If True, use mean values; otherwise use median
        normalize: If True, normalize each metric to [0, 1]
        figsize: Figure size
        save_path: Optional path to save figure
    """
    # Get all metrics
    all_metrics = set()
    for gen_metrics in results.values():
        all_metrics.update(gen_metrics.keys())

    all_metrics = sorted(all_metrics)
    gen_names = sorted(results.keys())

    # Build matrix
    matrix = np.zeros((len(gen_names), len(all_metrics)))

    for i, gen_name in enumerate(gen_names):
        for j, metric_name in enumerate(all_metrics):
            if metric_name in results[gen_name]:
                values = results[gen_name][metric_name]
                if values:
                    matrix[i, j] = np.mean(values) if use_mean else np.median(values)

    # Normalize if requested
    if normalize:
        for j in range(matrix.shape[1]):
            col = matrix[:, j]
            min_val, max_val = col.min(), col.max()
            if max_val > min_val:
                matrix[:, j] = (col - min_val) / (max_val - min_val)

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, cmap="viridis", aspect="auto")

    # Set ticks
    ax.set_xticks(np.arange(len(all_metrics)))
    ax.set_yticks(np.arange(len(gen_names)))
    ax.set_xticklabels([m.replace("_", " ").title() for m in all_metrics])
    ax.set_yticklabels(gen_names)

    # Rotate x labels
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    # Add colorbar
    plt.colorbar(im, ax=ax)

    # Add values to cells
    for i in range(len(gen_names)):
        for j in range(len(all_metrics)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", color="w", fontsize=8)

    ax.set_title("Metrics Heatmap" + (" (Normalized)" if normalize else ""))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Heatmap saved to {save_path}")

    plt.show()

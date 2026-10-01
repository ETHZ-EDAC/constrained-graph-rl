"""
Graph Generators for Benchmark Evaluation.

This module provides various graph generation functions for benchmarking:
- Erdős-Rényi Random Graphs (G(n,p) and G(n,m) models)
- Planar Erdős-Rényi Graphs (random planar graphs)
- Barabási-Albert Scale-Free Graphs
- Exponential Random Graph Models (ERGM)
- Boltzmann Planar Graphs
- Delaunay Triangulation Graphs
- Plantri-based Planar Graphs
- Lattice Grid Graphs

All generators return graphs in torch_geometric.data.Data format.
"""

from __future__ import annotations
from typing import Optional, Literal
import subprocess

import numpy as np
import networkx as nx
import torch
from torch_geometric.data import Data
from scipy.spatial import Delaunay, QhullError
from torch_geometric.utils import from_networkx

# Boltzmann planar graph sampler (optional dependency)
try:
    from planar_graph_sampler.planar_graph_generator import random_planar_graph  # type: ignore
except ImportError:  # pragma: no cover - optional dependency
    random_planar_graph = None

# Import helper functions
from graph_rl.utils.graph_helpers import (
    add_node_positions,
    networkx_to_pyg_data,
    ensure_graph_connected,
    set_seed,
    GraphGeneratorConfig,
)


# =============================================================================
# Erdős-Rényi Graphs
# =============================================================================


def generate_erdos_renyi_gnp(
    num_nodes,
    edge_probability: float = 0.15,
    seed: Optional[int] = None,
    ensure_connected: bool = True,
    directed: bool = False,
    **layout_kwargs,
) -> Data:
    """
    Generate Erdős-Rényi random graph using G(n,p) model.

    In the G(n,p) model, each possible edge is included independently
    with probability p.

    Args:
        num_nodes: Number of nodes
        edge_probability: Probability of edge creation (0 to 1)
        seed: Random seed for reproducibility
        ensure_connected: If True, ensure graph is connected
        **layout_kwargs: Additional arguments for layout (method, scale, etc.)

    Returns:
        PyTorch Geometric Data object with node positions
    """
    set_seed(seed)

    graph = nx.gnp_random_graph(num_nodes, edge_probability, seed=seed, directed=False)

    if ensure_connected and not directed:
        graph = ensure_graph_connected(graph, seed=seed)

    # Add node positions
    layout_config = {"method": "spring", "scale": 10.0, "seed": seed}
    layout_config.update(layout_kwargs)
    graph = add_node_positions(graph, **layout_config)

    return networkx_to_pyg_data(graph)


def generate_erdos_renyi_gnm(
    num_nodes: int,
    num_edges: int = 30,
    seed: Optional[int] = None,
    ensure_connected: bool = False,
    **layout_kwargs,
) -> Data:
    """
    Generate Erdős-Rényi random graph using G(n,m) model.

    In the G(n,m) model, a graph is chosen uniformly at random from the
    collection of all graphs with exactly m edges.

    Args:
        num_nodes: Number of nodes
        num_edges: Number of edges
        seed: Random seed for reproducibility
        ensure_connected: If True, ensure graph is connected (may add more edges)
        **layout_kwargs: Additional arguments for layout

    Returns:
        PyTorch Geometric Data object with node positions
    """
    set_seed(seed)

    graph = nx.gnm_random_graph(num_nodes, num_edges, seed=seed, directed=False)

    if ensure_connected:
        graph = ensure_graph_connected(graph, seed=seed)

    layout_config = {"method": "spring", "scale": 10.0, "seed": seed}
    layout_config.update(layout_kwargs)
    graph = add_node_positions(graph, **layout_config)

    return networkx_to_pyg_data(graph)


# =============================================================================
# Lattice Grid Graphs
# =============================================================================


def generate_lattice_grid(
    num_nodes: int = 60,
    rows: Optional[int] = None,
    cols: Optional[int] = None,
    seed: Optional[int] = None,
    circular: bool = True,
    **layout_kwargs,
) -> Data:
    """
    Generate a 2D lattice grid graph (planar).

    Creates a rectangular grid where each node is connected to its 4 neighbors
    (up, down, left, right). This is a planar graph with a natural geometric embedding.

    Args:
        num_nodes: Target number of nodes (used if rows/cols not specified)
        rows: Number of rows in the grid (if None, computed from num_nodes)
        cols: Number of columns in the grid (if None, computed from num_nodes)
        seed: Random seed (unused but kept for API consistency)
        circular: If True, arrange nodes in concentric circles; if False, rectangular grid
        **layout_kwargs: Layout arguments (default uses grid positions)

    Returns:
        PyTorch Geometric Data object with grid graph
    """
    set_seed(seed)

    # Compute grid dimensions if not provided
    if rows is None or cols is None:
        # Try to make a square-ish grid
        cols = int(np.sqrt(num_nodes))
        rows = (num_nodes + cols - 1) // cols

    actual_nodes = rows * cols

    # Generate grid graph
    graph = nx.grid_2d_graph(rows, cols)

    # If circular layout, close each ring by connecting last column to first column
    if circular:
        for i in range(rows):
            # Connect rightmost node to leftmost node in each ring
            graph.add_edge((i, cols - 1), (i, 0))

    # Relabel nodes from (i,j) tuples to integers
    mapping = {(i, j): i * cols + j for i in range(rows) for j in range(cols)}
    graph = nx.relabel_nodes(graph, mapping)

    # Create positions
    pos = {}
    scale = layout_kwargs.get("scale", 10.0)

    if circular:
        # Arrange nodes in concentric circles (radial layout)
        for i in range(rows):
            for j in range(cols):
                node_id = i * cols + j
                # Map grid position to polar coordinates
                # Inner rings have fewer nodes spread around the circle
                # Outer rings have more nodes
                radius = scale * (i + 1) / (rows + 1)  # Radial distance from center
                angle = 2 * np.pi * j / cols  # Angular position

                pos[node_id] = np.array(
                    [
                        radius * np.cos(angle),
                        radius * np.sin(angle),
                    ]
                )
    else:
        # Rectangular grid layout
        for i in range(rows):
            for j in range(cols):
                node_id = i * cols + j
                # Center the grid at origin
                pos[node_id] = np.array(
                    [
                        scale * (j - cols / 2 + 0.5),
                        scale * (rows / 2 - i - 0.5),
                    ]
                )

    nx.set_node_attributes(graph, pos, "pos")

    return networkx_to_pyg_data(graph)


# =============================================================================
# Planar Erdős-Rényi Graphs
# =============================================================================


def generate_planar_erdos_renyi(
    num_nodes: int,
    edge_probability: float = 0.15,
    max_attempts: int = 10000,
    min_edges: int = 5,
    seed: Optional[int] = None,
    **layout_kwargs,
) -> Data:
    """
    Generate a random planar graph using rejection sampling.

    Generates Erdős-Rényi graphs and keeps only planar ones. This ensures
    the resulting graph can be embedded in the plane without edge crossings.
    It doesn't waste time caclulating layouts for non-planar graphs.
    """
    set_seed(seed)
    rng = np.random.RandomState(seed)

    for attempt in range(max_attempts):
        current_seed = None if seed is None else seed + attempt
        graph = nx.gnp_random_graph(num_nodes, edge_probability, seed=current_seed)

        if nx.is_planar(graph) and graph.number_of_edges() >= min_edges:
            # Use planar layout by default
            layout_config = {"method": "planar", "scale": 10.0, "seed": current_seed}
            layout_config.update(layout_kwargs)

            # Override method to planar if not specified
            if "method" not in layout_kwargs:
                layout_config["method"] = "planar"

            graph = add_node_positions(graph, **layout_config)
            return networkx_to_pyg_data(graph)

    raise RuntimeError(
        f"Could not generate planar graph with {num_nodes} nodes and "
        f"p={edge_probability} after {max_attempts} attempts. "
        f"Try reducing edge_probability or num_nodes."
    )


# =============================================================================
# Barabási-Albert Scale-Free Graphs
# =============================================================================


def generate_barabasi_albert(
    num_nodes: int,
    num_attachments: int = 2,
    seed: Optional[int] = None,
    initial_graph: Optional[nx.Graph] = None,
    **layout_kwargs,
) -> Data:
    """
    Generate Barabási-Albert scale-free graph using preferential attachment.

    The BA model generates scale-free networks with power-law degree distribution.
    New nodes attach to existing nodes with probability proportional to their degree.

    Args:
        num_nodes: Number of nodes
        num_attachments: Number of edges to attach from new node to existing nodes (m parameter)
        seed: Random seed
        initial_graph: Initial connected graph (default: complete graph with m nodes)
        **layout_kwargs: Layout arguments

    Returns:
        PyTorch Geometric Data object with node positions
    """
    if num_attachments >= num_nodes:
        raise ValueError("num_attachments must be less than num_nodes")

    set_seed(seed)

    graph = nx.barabasi_albert_graph(num_nodes, num_attachments, seed=seed, initial_graph=initial_graph)

    layout_config = {"method": "spring", "scale": 10.0, "seed": seed}
    layout_config.update(layout_kwargs)
    graph = add_node_positions(graph, **layout_config)

    return networkx_to_pyg_data(graph)


# =============================================================================
# Exponential Random Graph Models (ERGM)
# =============================================================================


def _ergm_log_probability(
    adj: np.ndarray,
    edge_param: float,
    triangle_param: float,
    star_param: float,
) -> float:
    """
    Compute log probability of graph under ERGM model.

    Log P(G) = edge_param * num_edges + triangle_param * num_triangles + star_param * num_stars

    Args:
        adj: Adjacency matrix
        edge_param: Edge parameter (log-odds)
        triangle_param: Triangle parameter
        star_param: Star parameter

    Returns:
        Log probability
    """
    num_edges = np.sum(adj) / 2  # Undirected

    # Count triangles
    adj_squared = adj @ adj
    num_triangles = np.trace(adj_squared @ adj) / 6

    # Count 2-stars (connected triples)
    degrees = np.sum(adj, axis=1)
    num_2stars = np.sum(degrees * (degrees - 1)) / 2

    log_prob = edge_param * num_edges + triangle_param * num_triangles + star_param * num_2stars

    return log_prob


def generate_ergm(
    num_nodes: int,
    edge_param: float = -2.0,
    triangle_param: float = 0.1,
    star_param: float = 0.0,
    mcmc_steps: int = 1000,
    burnin: int = 100,
    seed: Optional[int] = None,
    ensure_connected: bool = False,
    **layout_kwargs,
) -> Data:
    """
    Generate graph from Exponential Random Graph Model (ERGM) using MCMC.

    ERGM is a flexible statistical model for graphs that can capture complex
    network properties. Uses Metropolis-Hastings MCMC sampling.

    The model includes:
    - Edge parameter: Controls overall edge density (negative = sparse)
    - Triangle parameter: Controls triadic closure/clustering (positive = more triangles)
    - Star parameter: Controls degree distribution (positive = hubs)

    Args:
        num_nodes: Number of nodes
        edge_param: Log-odds for edge formation (typically negative, e.g., -2.0)
        triangle_param: Triadic closure parameter (typically 0.0 to 0.5)
        star_param: k-star parameter for degree distribution (typically 0.0)
        mcmc_steps: Number of MCMC iterations
        burnin: Number of burn-in iterations to discard
        seed: Random seed
        ensure_connected: If True, keep only the largest connected component
        **layout_kwargs: Layout arguments

    Returns:
        PyTorch Geometric Data object
    """
    set_seed(seed)
    rng = np.random.RandomState(seed)

    # Initialize with empty graph
    adj = np.zeros((num_nodes, num_nodes), dtype=float)

    # MCMC sampling using Metropolis-Hastings
    for step in range(mcmc_steps + burnin):
        # Propose edge flip
        i, j = rng.choice(num_nodes, size=2, replace=False)
        if i > j:
            i, j = j, i

        # Calculate acceptance probability
        adj_proposed = adj.copy()
        adj_proposed[i, j] = 1 - adj_proposed[i, j]
        adj_proposed[j, i] = adj_proposed[i, j]

        log_prob_current = _ergm_log_probability(adj, edge_param, triangle_param, star_param)
        log_prob_proposed = _ergm_log_probability(adj_proposed, edge_param, triangle_param, star_param)

        log_acceptance = log_prob_proposed - log_prob_current

        if np.log(rng.random()) < log_acceptance:
            adj = adj_proposed

    # Convert to NetworkX
    graph = nx.from_numpy_array(adj)

    # Remove isolated nodes
    graph.remove_nodes_from(list(nx.isolates(graph)))

    if ensure_connected and graph.number_of_nodes() > 0 and not nx.is_connected(graph):
        # Keep the largest connected component to satisfy connectivity requirement
        largest = max(nx.connected_components(graph), key=len)
        graph = graph.subgraph(largest).copy()

    # Relabel nodes to be consecutive
    graph = nx.convert_node_labels_to_integers(graph)

    if graph.number_of_nodes() == 0:
        raise RuntimeError(
            "ERGM generated empty graph. Try adjusting parameters "
            "(e.g., increase edge_param or decrease triangle_param)"
        )

    layout_config = {"method": "spring", "scale": 10.0, "seed": seed}
    layout_config.update(layout_kwargs)
    graph = add_node_positions(graph, **layout_config)

    return networkx_to_pyg_data(graph)


# =============================================================================
# Boltzmann Sampler for Planar Graphs
# =============================================================================


def generate_boltzmann_planar(
    num_nodes: int,
    epsilon: float = 0.1,
    require_connected: bool = True,
    seed: Optional[int] = None,
    **layout_kwargs,
) -> Data:
    """
    Generate random planar graph using Boltzmann sampling from towink/boltzmann-planar-graph.

    This uses the high-quality implementation by Fusy et al. that generates planar graphs
    with uniform distribution in O(n/epsilon) expected time.

    Args:
        num_nodes: Target number of nodes (must be >= 3)
        epsilon: Size tolerance - generates graphs with nodes in [n(1-ε), n(1+ε)]
                 Set to < 1/n for exact size (slower, O(n²) time)
                 Default 0.1 gives ±10% size variation
        require_connected: If True, sample connected planar graphs only
        seed: Random seed for reproducibility
        **layout_kwargs: Layout arguments (defaults to planar layout)

    Returns:
        PyTorch Geometric Data object with planar graph

    Notes:
        - From: https://github.com/towink/boltzmann-planar-graph
        - License: BSD-3-Clause
        - Expected time: O(n/epsilon) for approximate size
        - Expected time: O(n²) for exact size (epsilon < 1/n)
        - Based on Fusy (2009) "Uniform random sampling of planar graphs in linear time"
        - Generates truly uniform random planar graphs (not biased by rejection sampling)

    Raises:
        ImportError: If boltzmann-planar-graph is not installed
        ValueError: If num_nodes < 3
    """

    if random_planar_graph is None:
        raise ImportError(
            "boltzmann-planar-graph is not installed. Install the optional dependency or disable the "
            "Boltzmann generator."
        )

    if num_nodes < 3:
        raise ValueError("num_nodes must be at least 3")

    set_seed(seed)

    # Generate graph using the Boltzmann sampler
    # Note: The library returns either nx.Graph or nx.PlanarEmbedding
    graph = random_planar_graph(
        n=num_nodes,
        epsilon=epsilon,
        require_connected=require_connected,
        with_embedding=False,  # We'll add our own layout
        allow_multiproc=False,
    )

    # The library may return graphs with non-consecutive node labels, convert to 0-indexed
    graph = nx.convert_node_labels_to_integers(graph, first_label=0)

    # Use spring layout for more organic appearance
    layout_config = {"method": "spring", "scale": 10.0, "seed": seed}
    layout_config.update(layout_kwargs)

    graph = add_node_positions(graph, **layout_config)

    return networkx_to_pyg_data(graph)


# =============================================================================
# Delaunay Triangulation Graphs
# =============================================================================


def generate_delaunay_graph(
    num_nodes: int,
    target_edge_length: float = 1.0,
    seed: Optional[int] = None,
    boundary_type: Literal["circle", "square"] = "circle",
    boundary_fraction: float = 0.15,
    **layout_kwargs,
) -> Data:
    """
    Generate a Delaunay triangulation graph with enforced boundary points.
    """
    set_seed(seed)
    rng = np.random.RandomState(seed)

    # Number of boundary points
    n_boundary = max(4, int(boundary_fraction * num_nodes))
    n_interior = num_nodes - n_boundary

    if boundary_type == "circle":
        radius = 1.0

        # Boundary points: exactly on the circle
        theta_b = np.linspace(0, 2 * np.pi, n_boundary, endpoint=False)
        boundary_points = np.column_stack(
            [
                radius * np.cos(theta_b),
                radius * np.sin(theta_b),
            ]
        )

        # Interior points: uniform in disk
        r = radius * np.sqrt(rng.uniform(0, 1, n_interior))
        theta = np.linspace(0, 2 * np.pi, n_interior, endpoint=False)
        interior_points = np.column_stack(
            [
                r * np.cos(theta),
                r * np.sin(theta),
            ]
        )

        points = np.vstack([boundary_points, interior_points])

    elif boundary_type == "square":
        side_length = 1.0
        half = side_length / 2

        base = n_boundary // 4
        rem = n_boundary % 4
        counts = [base + (1 if i < rem else 0) for i in range(4)]  # top, bottom, right, left

        xs_top = rng.uniform(-half, half, counts[0])
        xs_bot = rng.uniform(-half, half, counts[1])
        ys_right = rng.uniform(-half, half, counts[2])
        ys_left = rng.uniform(-half, half, counts[3])

        boundary_points = np.vstack(
            [
                np.column_stack([xs_top, half * np.ones(counts[0])]),
                np.column_stack([xs_bot, -half * np.ones(counts[1])]),
                np.column_stack([half * np.ones(counts[2]), ys_right]),
                np.column_stack([-half * np.ones(counts[3]), ys_left]),
            ]
        )

        interior_points = rng.uniform(-half, half, (n_interior, 2))
        base_points = np.vstack([boundary_points, interior_points])

        # Retry with increasing jitter to avoid degenerate/colinear point sets that
        # can destabilize eigendecomposition in downstream metrics.
        for jitter_exp in range(4):  # scales: 1e-6, 1e-5, 1e-4, 1e-3
            points = base_points + rng.normal(scale=10 ** (-6 + jitter_exp), size=base_points.shape)
            try:
                tri = Delaunay(points)
            except QhullError:
                if jitter_exp == 3:
                    raise
                continue

            edges = set()
            for simplex in tri.simplices:
                a, b, c = int(simplex[0]), int(simplex[1]), int(simplex[2])
                edges.add(tuple(sorted((a, b))))
                edges.add(tuple(sorted((b, c))))
                edges.add(tuple(sorted((c, a))))

            if not edges:
                if jitter_exp == 3:
                    raise RuntimeError("Delaunay triangulation produced no edges")
                continue

            # Enforce connectivity to avoid spectral eigendecomposition failures from
            # repeated zero eigenvalues on disconnected graphs.
            g_tmp = nx.Graph()
            g_tmp.add_nodes_from(range(num_nodes))
            g_tmp.add_edges_from(edges)
            if not nx.is_connected(g_tmp):
                comps = list(nx.connected_components(g_tmp))
                while len(comps) > 1:
                    a = comps.pop()
                    b = comps.pop()
                    u = rng.choice(list(a))
                    v = rng.choice(list(b))
                    edges.add(tuple(sorted((u, v))))
                    merged = a.union(b)
                    comps.append(merged)

            edge_index = torch.tensor(sorted(edges), dtype=torch.long).t().contiguous()
            pos = torch.from_numpy(points.astype(np.float32))
            edge_attr = torch.norm(pos[edge_index[0]] - pos[edge_index[1]], dim=1, keepdim=True)
            data = Data(edge_index=edge_index, pos=pos, edge_attr=edge_attr)
            data.num_nodes = num_nodes
            return data

        raise RuntimeError("Delaunay triangulation failed after jitter retries")

    else:
        raise ValueError(f"Unknown boundary_type: {boundary_type}")

    # Delaunay triangulation
    tri = Delaunay(points)

    # Build graph
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    for simplex in tri.simplices:
        graph.add_edge(simplex[0], simplex[1])
        graph.add_edge(simplex[1], simplex[2])
        graph.add_edge(simplex[2], simplex[0])

    # Node positions
    for i, pos in enumerate(points):
        graph.nodes[i]["pos"] = pos

    # Convert to PyG
    data = from_networkx(graph)
    data.pos = torch.from_numpy(points.astype(np.float32))

    # Optional: geometric edge lengths (recommended)
    src, dst = data.edge_index
    edge_attr = torch.norm(data.pos[src] - data.pos[dst], dim=1)
    data.edge_attr = edge_attr.unsqueeze(-1)

    return data


# =============================================================================
# Plantri-based Planar Graph Generation
# =============================================================================


def _parse_planar_code(data: bytes, max_graphs: int | None = None) -> list[tuple[int, list[list[int]]]]:
    """
    Parse planar_code binary format from plantri output.

    Args:
        data: Binary data from plantri
        max_graphs: Maximum number of graphs to parse (None = all). Use 1 for single graph.

    Returns:
        List of (num_nodes, adjacency_lists) tuples.
        Each adjacency_list is a list of neighbor lists for each vertex.
    """
    # Skip header ">>planar_code<<" (15 bytes)
    HEADER = b">>planar_code<<"
    if data.startswith(HEADER):
        data = data[len(HEADER) :]

    graphs = []
    i = 0

    while i < len(data):
        if max_graphs is not None and len(graphs) >= max_graphs:
            break  # Stop early if we have enough graphs

        if data[i] == 0:  # Sometimes there's trailing zeros
            i += 1
            continue

        n = data[i]  # number of vertices
        i += 1

        if n == 0 or i >= len(data):
            break

        adjacency = []
        for v in range(n):
            neighbors = []
            while i < len(data) and data[i] != 0:  # 0 is separator
                neighbors.append(data[i] - 1)  # Convert to 0-indexed
                i += 1
            i += 1  # skip separator
            adjacency.append(neighbors)

        graphs.append((n, adjacency))

    return graphs


def generate_plantri_graph(
    num_nodes: int,
    sample_index: int = 0,
    connectivity: int = 3,
    min_degree: Optional[int] = None,
    seed: Optional[int] = None,
    plantri_dir: str = "ext/plantri55",
    generate_single: bool = True,
    max_graphs: Optional[int] = None,
    **layout_kwargs,
) -> Data:
    """
    Generate planar graph using plantri.

    Args:
        num_nodes: Number of nodes
        sample_index: Which graph to select from enumeration (0-indexed, only used if generate_single=False)
        connectivity: Connectivity constraint (1, 2, 3, or 4)
        min_degree: Minimum degree constraint (3, 4, or 5)
        seed: Random seed for random selection (overrides sample_index if generate_single=False)
        plantri_dir: Path to plantri directory
        generate_single: If True (default), only generate/parse the first graph (much faster).
                        If False, generate all graphs and select one (slow for num_nodes > 15).
        max_graphs: When generate_single=False, cap how many graphs are parsed (first k in enumeration).
            Parsed stream is truncated using head to avoid enumerating the full space.
        **layout_kwargs: Layout arguments

    Returns:
        PyTorch Geometric Data object with planar graph

    Notes:
        - Plantri must be compiled first: cd ext/plantri55 && make
        - With generate_single=True, works well even for num_nodes up to 30-40
        - With generate_single=False, only practical for num_nodes <= 15-20
        - Uses spring layout by default since graphs are guaranteed planar
    """
    set_seed(seed)

    # Build plantri command
    cmd = ["./plantri"]

    if connectivity > 0:
        cmd.append(f"-c{connectivity}")

    if min_degree is not None:
        cmd.append(f"-m{min_degree}")

    cmd.append(str(num_nodes))

    # Run plantri
    try:
        use_head = generate_single or max_graphs is not None
        if use_head:
            # Pipe through head so plantri stops early via SIGPIPE once we have enough bytes
            import shlex

            plantri_cmd = " ".join(shlex.quote(str(c)) for c in cmd)

            if max_graphs is not None:
                # Generous byte budget per graph to ensure we capture max_graphs
                per_graph_bytes = max(1024, num_nodes * 40)
                byte_limit = max_graphs * per_graph_bytes
            else:
                # Legacy single-graph case
                byte_limit = 10000

            # Avoid requesting an unbounded amount of data
            byte_limit = min(byte_limit, 50_000_000)

            bash_cmd = f"{plantri_cmd} | head -c {byte_limit}"
            result = subprocess.run(
                bash_cmd,
                cwd=plantri_dir,
                capture_output=True,
                timeout=30,
                shell=True,
            )

            # SIGPIPE causes exit code 141 when head closes the pipe
            if result.returncode not in (0, 141):
                raise RuntimeError(f"Plantri failed with return code {result.returncode}: " f"{result.stderr.decode()}")
        else:
            # For multiple graphs: generate all (slow)
            result = subprocess.run(
                cmd,
                cwd=plantri_dir,
                capture_output=True,
                timeout=30,  # 30 second timeout
            )
            if result.returncode != 0:
                raise RuntimeError(f"Plantri failed with return code {result.returncode}: " f"{result.stderr.decode()}")
    except subprocess.TimeoutExpired:
        raise RuntimeError(
            f"Plantri timed out for {num_nodes} nodes. " f"Try reducing num_nodes or use a different generator."
        )
    except FileNotFoundError:
        raise RuntimeError(
            f"Plantri executable not found in {plantri_dir}. " f"Compile it first: cd {plantri_dir} && make"
        )

    # Parse output
    max_graphs_to_parse = 1 if generate_single else max_graphs
    graphs = _parse_planar_code(result.stdout, max_graphs=max_graphs_to_parse)
    if not graphs:
        raise RuntimeError(
            f"Plantri generated no graphs for {num_nodes} nodes with "
            f"connectivity={connectivity}, min_degree={min_degree}"
        )

    if generate_single:
        n, adjacency = graphs[0]
    else:
        if seed is not None:
            rng = np.random.RandomState(seed)
            idx = rng.randint(len(graphs))
        else:
            idx = sample_index % len(graphs)

        n, adjacency = graphs[idx]

    # Convert to NetworkX
    graph = nx.Graph()
    graph.add_nodes_from(range(n))

    for v, neighbors in enumerate(adjacency):
        for u in neighbors:
            if v < u:  # Add each edge once
                graph.add_edge(v, u)

    # Add layout
    layout_config = {"method": "spring", "scale": 10.0, "seed": seed}
    layout_config.update(layout_kwargs)
    graph = add_node_positions(graph, **layout_config)

    return networkx_to_pyg_data(graph)


# =============================================================================
# Generator Configuration Helper
# =============================================================================


def get_generators_dict(config: GraphGeneratorConfig, seed: Optional[int] = None):
    """
    Create a dictionary of all available graph generators with the given configuration.

    Args:
        config: GraphGeneratorConfig object with generation parameters
        seed: Random seed for reproducibility

    Returns:
        Dictionary mapping generator names to generator functions
    """

    generators = {
        "erdos_renyi_gnp": lambda: generate_erdos_renyi_gnp(
            config.num_nodes,
            config.edge_probability,
            seed=seed,
            ensure_connected=config.ensure_connected,
            method=config.layout_method,
            scale=config.layout_scale,
        ),
        "erdos_renyi_gnm": lambda: generate_erdos_renyi_gnm(
            config.num_nodes,
            config.num_edges
            if config.num_edges is not None
            else int(0.5 * config.num_nodes * (config.num_nodes - 1) * config.edge_probability),
            seed=seed,
            ensure_connected=config.ensure_connected,
            method=config.layout_method,
            scale=config.layout_scale,
        ),
        "planar_er": lambda: generate_planar_erdos_renyi(
            config.num_nodes,
            config.planar_er_edge_probability
            if config.planar_er_edge_probability is not None
            else config.edge_probability,
            max_attempts=config.planar_max_attempts,
            min_edges=config.planar_min_edges,
            seed=seed,
            scale=config.layout_scale,
        ),
        "barabasi_albert": lambda: generate_barabasi_albert(
            config.num_nodes,
            config.num_attachments,
            seed=seed,
            method=config.layout_method,
            scale=config.layout_scale,
        ),
        "ergm": lambda: generate_ergm(
            config.num_nodes,
            config.ergm_edge_param,
            config.ergm_triangle_param,
            config.ergm_star_param,
            mcmc_steps=config.ergm_mcmc_steps,
            burnin=config.ergm_burnin,
            seed=seed,
            ensure_connected=config.ensure_connected,
            method=config.layout_method,
            scale=config.layout_scale,
        ),
        "delaunay": lambda: generate_delaunay_graph(
            config.num_nodes,
            target_edge_length=config.layout_scale / 10.0,
            seed=seed,
            boundary_type=config.delaunay_boundary,
        ),
        "lattice_grid": lambda: generate_lattice_grid(
            config.num_nodes,
            rows=config.grid_rows,
            cols=config.grid_cols,
            seed=seed,
            scale=config.layout_scale,
            circular=config.grid_circular,
        ),
        "plantri": lambda: generate_plantri_graph(
            config.num_nodes,
            connectivity=config.plantri_connectivity,
            min_degree=config.plantri_min_degree,
            seed=seed,
            generate_single=config.plantri_generate_single,
            max_graphs=config.plantri_max_graphs,
            scale=config.layout_scale,
        ),
    }

    generators["boltzmann_planar"] = lambda: generate_boltzmann_planar(
        config.num_nodes,
        epsilon=config.boltzmann_epsilon,
        require_connected=config.ensure_connected,
        seed=seed,
        method=config.layout_method,
        scale=config.layout_scale,
    )

    return generators


# TESTING CODE
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from torch_geometric.utils import to_networkx

    # Generate a planar graph
    g = generate_plantri_graph(60, connectivity=2, min_degree=3)
    print(f"Generated graph with {g.num_nodes} nodes and {g.num_edges} edges")

    # Convert to NetworkX for plotting
    G = to_networkx(g, to_undirected=True)

    # Extract node positions from the Data object
    pos = {i: g.pos[i].numpy() for i in range(g.num_nodes)}

    # Plot the graph
    plt.figure(figsize=(10, 10))
    nx.draw(
        G,
        pos=pos,
        with_labels=True,
        node_color="lightblue",
        node_size=500,
        font_size=10,
        font_weight="bold",
        edge_color="gray",
        width=2,
    )
    plt.title("Plantri Generated Planar Graph")
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig("plantri_graph.png")
    print("Graph saved to plantri_graph.png")

    if nx.is_planar(G):
        print("The generated graph is planar.")

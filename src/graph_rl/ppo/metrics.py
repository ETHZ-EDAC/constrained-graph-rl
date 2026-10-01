"""
Graph Statistics Metrics for evaluating planar graphs

Implements metrics from the PDF specification:

Graph Statistics:
- Spectral Gap / Algebraic Connectivity
- Spectral Norm of Adjacency Matrix
- Assortativity
- Clustering Coefficient
- Gini Coefficient of Degree Distribution
- Orbit and Cycle Counts

Planar Graph Metrics:
- Edge Crossings
- Angular Resolution
- Edge Length Uniformity
- Degree Distribution
- Number of Connected Components
"""

from __future__ import annotations
from pathlib import Path
from typing import Dict, List, cast, Iterable
from collections import defaultdict
import math

import jax
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor
from torch_geometric.data import Batch, Data
from torch_geometric.nn.functional import gini
from torch_geometric.utils import assortativity, dense_to_sparse, to_dense_adj
import networkx as nx

import jax.numpy as jnp

from graph_rl.ops.make_aabb import add_aabb_rectangle, best_oriented_square_rectangularity
from graph_rl.ops.polygon_area_perim import polygon_area_perimeter_from_boundary
from graph_rl.ppo.planar_geometry import minimum_angle_from_sector_angles, maximum_degree_from_adjacency
from graph_rl.utils.adjacency_utils import get_edge_vectors
from graph_rl.utils.iso_squareness_baselines import (
    get_iso_and_squareness_for_baselines,
    graph_shape_from_edges,
)


def _to_tensor(array, *, dtype: torch.dtype = torch.float64) -> Tensor:
    """Best-effort conversion to a torch tensor while ensuring safe mutability."""
    if isinstance(array, Tensor):
        return array.to(dtype=dtype)

    if isinstance(array, np.ndarray):
        # Some sources (e.g. JAX device buffers) expose read-only numpy views; copy to avoid UB.
        if not array.flags.writeable:
            array = np.array(array, copy=True)
        return torch.as_tensor(array, dtype=dtype)

    try:
        return torch.as_tensor(array, dtype=dtype)
    except Exception:
        return torch.tensor(array, dtype=dtype)






class GraphMetrics:
    """Compute graph statistics metrics for adjacency matrices."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self.assortativity = assortativity  # inherited for convenience

    def _prepare_adjacency(self, adjacency_matrix) -> Tensor:
        """Return a dense, symmetric float64 adjacency tensor free of self-loops."""
        A = _to_tensor(adjacency_matrix)
        assert A.dim() == 2 and A.shape[0] == A.shape[1], "Adjacency matrix must be square"
        if A.numel() == 0:
            return A
        A = A.clone()
        # Removing self-loops if any otherwise calculations may be off
        # filling the diagonal with zeros
        A.fill_diagonal_(0.0)
        # Convert directed adjacency to undirected by taking the strongest edge
        # in either direction (avoid halving single-direction edges).
        return torch.maximum(A, A.T)

    def adjacency_from_edge_index(
        self,
        edge_index,
        *,
        num_nodes: int | None = None,
        edge_weight: Tensor | None = None,
    ) -> Tensor:
        """Public helper to obtain the prepared adjacency from an edge index."""
        edge_index = _to_tensor(edge_index, dtype=torch.long)
        assert edge_index.dim() == 2 and edge_index.shape[0] == 2, "edge_index must have shape (2, num_edges)"
        edge_attr_tensor = None if edge_weight is None else _to_tensor(edge_weight)

        # If edge_attr has shape (E, 1), squeeze it to (E,) to avoid extra dimensions
        if edge_attr_tensor is not None and edge_attr_tensor.dim() == 2 and edge_attr_tensor.shape[1] == 1:
            edge_attr_tensor = edge_attr_tensor.squeeze(1)

        dense = to_dense_adj(
            edge_index,
            max_num_nodes=num_nodes,
            edge_attr=edge_attr_tensor,
        ).squeeze(0)
        return self._prepare_adjacency(dense)

    def _laplacian(self, adjacency: Tensor) -> Tensor:
        """
        Easier than using get_laplacian from torch_geometric
        since we are already computing the dense adjacency
        https://en.wikipedia.org/wiki/Laplacian_matrix
        """
        deg = adjacency.sum(dim=1)
        inv_sqrt_deg = torch.zeros_like(deg)
        active = deg > self.eps
        inv_sqrt_deg[active] = deg[active].rsqrt()
        normalized_adjacency = adjacency * inv_sqrt_deg[:, None] * inv_sqrt_deg[None, :]
        L_norm = torch.diag(active.to(dtype=adjacency.dtype)) - normalized_adjacency
        return L_norm

    def _smallest_positive_laplacian_eigenvalue(self, adjacency: Tensor) -> float:
        """Compute the algebraic connectivity using dense torch solvers."""
        laplacian = self._laplacian(adjacency)
        eigenvalues = torch.linalg.eigvalsh(laplacian)
        positive = eigenvalues[eigenvalues > self.eps]
        if positive.numel() == 0:
            return 0.0
        return float(positive.min().item())

    def _largest_adjacency_eigenvalue(self, adjacency: Tensor) -> float:
        """Compute the dominant adjacency eigenvalue using dense torch solvers."""
        eigenvalues = torch.linalg.eigvalsh(adjacency)
        if eigenvalues.numel() == 0:
            return 0.0
        return float(eigenvalues.abs().max().item())

    def spectral_gap(self, adjacency_matrix) -> float:
        """
        Compute the spectral gap (algebraic connectivity) i.e., the smallest positive eigenvalue of the Laplacian.
        see https://en.wikipedia.org/wiki/Laplacian_matrix
        """
        A = self._prepare_adjacency(adjacency_matrix)
        return self._smallest_positive_laplacian_eigenvalue(A)

    def spectral_norm(self, adjacency_matrix) -> float:
        """
        Compute the spectral norm (largest singular value) of the adjacency matrix.
        See https://haroldbenoit.com/notes/ml/llms/scaling/mu-transfer/spectral-norm
        """
        A = self._prepare_adjacency(adjacency_matrix)
        return self._largest_adjacency_eigenvalue(A)

    def triangle_count(self, adjacency_matrix) -> float:
        """
        Compute the number of length-3 cycles (triangles).
        From https://www.tandfonline.com/doi/full/10.1080/09728600.2023.2234421#abstract
        """
        A = self._prepare_adjacency(adjacency_matrix)
        if A.numel() == 0:
            return 0.0
        # Formula: |triangles| = trace(A^3) / 6 for undirected simple graphs.
        trace = torch.trace(torch.linalg.matrix_power(A, 3))
        return float((trace / 6.0).item())

    def triplet_count(self, adjacency_matrix) -> float:
        """
        Compute the number of connected triplets (length-2 paths).
        From https://www.tandfonline.com/doi/full/10.1080/09728600.2023.2234421#abstract
        """
        A = self._prepare_adjacency(adjacency_matrix)
        degrees = A.sum(dim=1)
        triplet_counts = degrees * (degrees - 1) / 2.0
        return float(triplet_counts.sum().item())

    def clustering_coefficient(self, triangles: float, triplets: float) -> float:
        """
        Compute the global clustering coefficient from triangle and triplet counts.
        https://en.wikipedia.org/wiki/Clustering_coefficient
        """
        if triplets <= self.eps:
            return 0.0
        return (3.0 * triangles) / triplets


    def _gini_from_degrees(self, adjacency_matrix: Tensor) -> float:
        """Compute the Gini coefficient for a vector of degrees using PyG."""
        A = self._prepare_adjacency(adjacency_matrix)
        degrees = A.sum(dim=1)
        if degrees.numel() == 0:
            return 0.0
        total_degree = float(degrees.sum().item())
        if total_degree <= self.eps:
            return 0.0
        gini_value = gini(degrees.unsqueeze(0))
        if not torch.isfinite(gini_value).all().item():
            return 0.0
        return float(gini_value.mean().item())

    def num_nodes(self, adjacency_matrix) -> float:
        """Return the number of nodes in the graph."""
        A = self._prepare_adjacency(adjacency_matrix)
        return float(A.shape[0])

    def num_edges(self, adjacency_matrix) -> float:
        """Return the number of edges in the graph (undirected)."""
        A = self._prepare_adjacency(adjacency_matrix)
        # For undirected graphs, count upper triangle to avoid double counting
        return float((A > 0).sum().item() / 2.0)

    def max_degree(self, adjacency_matrix) -> float:
        """Return the maximum vertex degree in the graph."""
        A = self._prepare_adjacency(adjacency_matrix)
        return maximum_degree_from_adjacency(A)

    def compute_all_metrics(
        self,
        edge_index,
        adjacency_matrix,
        *,
        metric_subset: List[str] | None = None,
    ) -> Dict[str, float]:
        """
        Compute graph statistics metrics, optionally filtering by subset.

        Args:
            metric_subset: List of metric names to compute. If None, computes all metrics.
                          If empty list, computes no metrics. If list with names, computes only those.
        """
        metrics: Dict[str, float] = {}

        # If None, compute all metrics (backward compatibility)
        # If empty list, compute nothing
        # If list with names, compute only those
        if metric_subset is None:
            # Compute all metrics (default behavior)
            metric_subset = [
                "num_nodes",
                "num_edges",
                "spectral_gap",
                "spectral_norm",
                "assortativity",
                "gini_coefficient",
                "clustering_coefficient",
                "triangles",
                "max_degree",
            ]
        elif len(metric_subset) == 0:
            # Empty subset = compute nothing
            return metrics

        # Compute only requested metrics
        if "num_nodes" in metric_subset:
            metrics["num_nodes"] = self.num_nodes(adjacency_matrix)
        if "num_edges" in metric_subset:
            metrics["num_edges"] = self.num_edges(adjacency_matrix)
        if "spectral_gap" in metric_subset:
            metrics["spectral_gap"] = self.spectral_gap(adjacency_matrix)
        if "spectral_norm" in metric_subset:
            metrics["spectral_norm"] = self.spectral_norm(adjacency_matrix)
        if "assortativity" in metric_subset:
            metrics["assortativity"] = assortativity(edge_index=edge_index)
        if "gini_coefficient" in metric_subset:
            metrics["gini_coefficient"] = self._gini_from_degrees(adjacency_matrix)

        # Clustering coefficient and triangles are computed together for efficiency
        if "clustering_coefficient" in metric_subset or "triangles" in metric_subset:
            triangles = self.triangle_count(adjacency_matrix)
            if "triangles" in metric_subset:
                metrics["triangles"] = triangles
            if "clustering_coefficient" in metric_subset:
                metrics["clustering_coefficient"] = self.clustering_coefficient(
                    triangles, self.triplet_count(adjacency_matrix)
                )
        if "max_degree" in metric_subset:
            metrics["max_degree"] = self.max_degree(adjacency_matrix)

        return metrics


class PlanarGraphMetrics:
    """Compute planar graph-specific metrics using vertex positions and adjacency."""

    def __init__(self, eps: float = 1e-8) -> None:
        self.eps = float(eps)
        self._graph_metrics = GraphMetrics(eps=eps)

    def angular_resolution(
        self,
        sector_angles: Tensor | None,
    ) -> float:
        """
        Compute the angular resolution metric as per Mooney et al. (2020).
        https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GD.2025.30
        """
        if sector_angles is None:
            return 0.0

        sector_tensor = _to_tensor(sector_angles)
        if sector_tensor.dim() == 1:
            sector_tensor = sector_tensor.unsqueeze(0)
        if sector_tensor.numel() == 0 or sector_tensor.shape[1] <= 1:
            return 0.0

        max_angles = sector_tensor.shape[1] - 1
        degree_counts = sector_tensor[:, 0]
        counts = degree_counts.round().to(torch.int64).clamp(min=0, max=max_angles)
        if counts.sum().item() == 0:
            return 0.0

        angles = sector_tensor[:, 1:]
        indices = torch.arange(max_angles, device=angles.device)
        valid_slot_mask = indices.unsqueeze(0) < counts.unsqueeze(1)
        if not valid_slot_mask.any():
            return 0.0

        positive_mask = valid_slot_mask & (angles > self.eps)
        if not positive_mask.any():
            return 0.0

        inf_tensor = torch.full_like(angles, float("inf"))
        masked_angles = torch.where(positive_mask, angles, inf_tensor)
        min_angles = masked_angles.min(dim=1).values

        valid_vertex_mask = (counts > 1) & (min_angles < float("inf"))
        if not valid_vertex_mask.any():
            return 0.0

        degrees = counts[valid_vertex_mask].to(dtype=sector_tensor.dtype)
        actual_min = min_angles[valid_vertex_mask]
        ideal = (2.0 * float(np.pi)) / degrees
        # Angular resolution per Mooney et al.: 1 - mean(|theta - theta_min| / theta)
        deviations = (ideal - actual_min).abs() / ideal
        score = 1.0 - deviations.mean()
        return float(score.item())

    def _edge_lengths(self, adjacency_matrix, vertex_positions) -> torch.Tensor:
        adj_np = adjacency_matrix.cpu().numpy()
        pos_np = vertex_positions.cpu().numpy()
        edge_vecs_jnp, num_edges_jnp = get_edge_vectors(jnp.asarray(adj_np), jnp.asarray(pos_np))
        num_edges = int(num_edges_jnp)
        if num_edges == 0:
            return torch.empty(0, dtype=vertex_positions.dtype, device=vertex_positions.device)
        edge_vecs = np.asarray(edge_vecs_jnp[:num_edges])
        diffs = edge_vecs[:, 0] - edge_vecs[:, 1]
        lengths_np = np.linalg.norm(diffs, axis=1)
        return torch.from_numpy(lengths_np).to(device=vertex_positions.device, dtype=vertex_positions.dtype)

    def edge_length_uniformity(self, adjacency_matrix, vertex_positions) -> float:
        lengths = self._edge_lengths(adjacency_matrix, vertex_positions)
        if lengths.numel() == 0:
            return 0.0
        mean_length = lengths.mean()
        if mean_length < self.eps:
            return 0.0
        relative_deviation = (lengths - mean_length) / mean_length
        uniformity = torch.sqrt(torch.mean(relative_deviation**2))
        return float(uniformity.item())


    def edge_length_deviation(self, adjacency_matrix, vertex_positions) -> float:
        """Edge length deviation (Eq. 4) from Mooney et al. (2020)."""
        lengths = self._edge_lengths(adjacency_matrix, vertex_positions)
        if lengths.numel() == 0:
            return 0.0
        ideal_length = lengths.mean()
        if ideal_length <= self.eps:
            return 0.0
        relative = (lengths - ideal_length).abs() / ideal_length
        avg_relative = relative.mean()
        score = 1.0 / (1.0 + avg_relative)
        return float(score.item())

    def edge_length_min(self, adjacency_matrix, vertex_positions) -> float:
        """Minimum edge length in the graph."""
        lengths = self._edge_lengths(adjacency_matrix, vertex_positions)
        if lengths.numel() == 0:
            return 0.0
        return float(lengths.min().item())

    def edge_length_max(self, adjacency_matrix, vertex_positions) -> float:
        """Maximum edge length in the graph."""
        lengths = self._edge_lengths(adjacency_matrix, vertex_positions)
        if lengths.numel() == 0:
            return 0.0
        return float(lengths.max().item())

    def isoperimetric_ratio(self, adjacency_matrix, adjacency_boundary, vertex_positions) -> float:
        """Isoperimetric ratio 4*pi*A / P^2 using boundary adjacency."""
        if adjacency_boundary is None:
            pos_np = _to_tensor(vertex_positions).cpu().numpy()
            result = get_iso_and_squareness_for_baselines(pos_np[:, :2])
            metrics = result[0] if isinstance(result, tuple) else result
            return float(metrics["iso"])
        adj_np = _to_tensor(adjacency_boundary).cpu().numpy()
        pos_np = _to_tensor(vertex_positions).cpu().numpy()
        pos_xy = pos_np[:, :2]
        area_jnp, perim_jnp = polygon_area_perimeter_from_boundary(
            jnp.asarray(adj_np),
            jnp.asarray(pos_xy),
        )
        area = float(area_jnp)
        perim = float(perim_jnp)
        if perim <= self.eps:
            return 0.0
        ratio = (4.0 * float(np.pi) * area) / (perim * perim + self.eps)
        return float(np.clip(ratio, 0.0, 1.0))

    def rectangularity(
        self,
        adjacency_matrix,
        adjacency_boundary,
        vertex_positions,
        *,
        num_angles: int = 181,
    ) -> float:
        """Square rectangularity score matching env reward logic."""
        if adjacency_boundary is None:
            pos_np = _to_tensor(vertex_positions).cpu().numpy()
            result = get_iso_and_squareness_for_baselines(pos_np[:, :2])
            metrics = result[0] if isinstance(result, tuple) else result
            return float(metrics["squareness"])
        adj = _to_tensor(adjacency_matrix).cpu().numpy()
        adj_b = _to_tensor(adjacency_boundary).cpu().numpy()
        pos_np = _to_tensor(vertex_positions).cpu().numpy()
        pos_xy = pos_np[:, :2]
        area_jnp, perim_jnp = polygon_area_perimeter_from_boundary(
            jnp.asarray(adj_b),
            jnp.asarray(pos_xy),
        )
        if float(perim_jnp) <= self.eps:
            return 0.0
        score, _, _, _ = best_oriented_square_rectangularity(
            vertices_xy=jnp.asarray(pos_xy),
            adjacency_boundary=jnp.asarray(adj_b),
            adjacency_full=jnp.asarray(adj),
            area_polygon=area_jnp,
            perim_polygon=perim_jnp,
            num_angles=num_angles,
        )
        return float(np.clip(score, 0.0, 1.0))

    def node_uniformity(self, vertex_positions) -> float:
        """Node resolution (Eq. 9) from Mooney et al. (2020)."""
        positions = _to_tensor(vertex_positions)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        if positions.shape[0] < 2:
            return 0.0
        distances = torch.cdist(positions, positions, p=2.0)  # pnorm = 2 for Euclidean
        mask = ~torch.eye(distances.shape[0], dtype=torch.bool, device=distances.device)  # exclude self-distances
        valid_distances = distances.masked_select(mask)
        if valid_distances.numel() == 0:
            return 0.0
        max_dist = valid_distances.max()
        if max_dist <= self.eps:
            return 0.0
        min_dist = valid_distances.min()
        return float((min_dist / max_dist).item())

    def minimum_angle(self, sector_angles) -> float:
        """Extract minimum angle from sector angles (in degrees)."""
        return minimum_angle_from_sector_angles(sector_angles)

    def alpha_shape_metrics(
        self,
        vertex_positions,
    ) -> Dict[str, float]:
        """Legacy metric keys backed by the polygonized edge footprint."""
        positions = _to_tensor(vertex_positions)
        if positions.dim() == 1:
            positions = positions.unsqueeze(0)
        if positions.shape[0] < 3:
            return {"alpha_shape_area": 0.0, "alpha_shape_perimeter": 0.0}

        points = positions[:, :2].detach().cpu().numpy()
        shape, faces = graph_shape_from_edges(points)
        if not faces or shape is None or shape.is_empty or not hasattr(shape, "area") or not hasattr(shape, "length"):
            return {"alpha_shape_area": 0.0, "alpha_shape_perimeter": 0.0}

        return {
            "alpha_shape_area": float(shape.area),
            "alpha_shape_perimeter": float(shape.length),
        }

    def is_planar(self, adjacency_matrix) -> float:
        """
        Check if the graph is planar using NetworkX.
        Returns 1.0 if planar, 0.0 if not planar.

        This is a simple utility - for Data objects, consider using
        graph_rl.utils.benchmark_graphs.is_planar_graph() instead.
        """
        A = self._graph_metrics._prepare_adjacency(adjacency_matrix)
        if A.numel() == 0:
            return 1.0  # Empty graph is trivially planar

        # Convert to binary adjacency (0 or 1) to avoid weight issues
        adj_np = (A.cpu().numpy() > 1e-8).astype(float)
        G = nx.from_numpy_array(adj_np)

        # Check planarity
        return 1.0 if nx.is_planar(G) else 0.0

    def compute_all_planar_metrics(
        self,
        adjacency_matrix,
        vertex_positions,
        *,
        sector_angles=None,
        adjacency_boundary=None,
        metric_subset: List[str] | None = None,
    ) -> Dict[str, float]:
        """
        Compute planar graph metrics, optionally filtering by subset.

        Args:
            metric_subset: List of metric names to compute. If None, computes all metrics.
                          If empty list, computes no metrics. If list with names, computes only those.
        """
        metrics: Dict[str, float] = {}

        # If None, compute all planar metrics (backward compatibility)
        # If empty list, compute nothing
        # If list with names, compute only those
        if metric_subset is None:
            # Compute all planar metrics (default behavior)
            metric_subset = [
                "is_planar",
                "angular_resolution",
                "edge_length_uniformity",
                "edge_length_deviation",
                "edge_length_min",
                "edge_length_max",
                "isoperimetric_ratio",
                "rectangularity",
                "node_uniformity",
                "minimum_angle",
                "alpha_shape_area",
                "alpha_shape_perimeter",
            ]
        elif len(metric_subset) == 0:
            # Empty subset = compute nothing
            return metrics

        # Compute only requested metrics
        if "is_planar" in metric_subset:
            metrics["is_planar"] = self.is_planar(adjacency_matrix)
        if "angular_resolution" in metric_subset:
            metrics["angular_resolution"] = self.angular_resolution(sector_angles=sector_angles)
        if "edge_length_uniformity" in metric_subset:
            metrics["edge_length_uniformity"] = self.edge_length_uniformity(adjacency_matrix, vertex_positions)
        if "edge_length_deviation" in metric_subset:
            metrics["edge_length_deviation"] = self.edge_length_deviation(adjacency_matrix, vertex_positions)
        if "edge_length_min" in metric_subset:
            metrics["edge_length_min"] = self.edge_length_min(adjacency_matrix, vertex_positions)
        if "edge_length_max" in metric_subset:
            metrics["edge_length_max"] = self.edge_length_max(adjacency_matrix, vertex_positions)
        if "isoperimetric_ratio" in metric_subset:
            metrics["isoperimetric_ratio"] = self.isoperimetric_ratio(
                adjacency_matrix,
                adjacency_boundary,
                vertex_positions,
            )
        if "rectangularity" in metric_subset:
            metrics["rectangularity"] = self.rectangularity(
                adjacency_matrix,
                adjacency_boundary,
                vertex_positions,
            )

        if "node_uniformity" in metric_subset:
            metrics["node_uniformity"] = self.node_uniformity(vertex_positions)
        if "minimum_angle" in metric_subset:
            metrics["minimum_angle"] = self.minimum_angle(sector_angles)
        if "alpha_shape_area" in metric_subset or "alpha_shape_perimeter" in metric_subset:
            alpha_shape = self.alpha_shape_metrics(vertex_positions)
            if "alpha_shape_area" in metric_subset:
                metrics["alpha_shape_area"] = alpha_shape["alpha_shape_area"]
            if "alpha_shape_perimeter" in metric_subset:
                metrics["alpha_shape_perimeter"] = alpha_shape["alpha_shape_perimeter"]

        return metrics


def compute_metrics_from_raw(
    edge_index,
    vertex_positions,
    sector_angles,
    *,
    num_nodes: int | None = None,
    edge_weight: Tensor | None = None,
    adjacency_matrix: Tensor | None = None,
    adjacency_boundary: Tensor | None = None,
    graph_metrics: GraphMetrics,
    planar_metrics: PlanarGraphMetrics,
    metric_subset: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    if adjacency_matrix is None:
        adjacency = graph_metrics.adjacency_from_edge_index(
            edge_index,
            num_nodes=num_nodes,
            edge_weight=edge_weight,
        )
    else:
        adjacency = graph_metrics._prepare_adjacency(adjacency_matrix)
        # If num_nodes is specified and smaller than the adjacency matrix, trim it
        if num_nodes is not None and adjacency.shape[0] > num_nodes:
            adjacency = adjacency[:num_nodes, :num_nodes]

    graph_stats = graph_metrics.compute_all_metrics(edge_index, adjacency, metric_subset=metric_subset)

    planar_stats = planar_metrics.compute_all_planar_metrics(
        adjacency,
        vertex_positions,
        sector_angles=sector_angles,
        adjacency_boundary=adjacency_boundary,
        metric_subset=metric_subset,
    )

    payload = {**graph_stats, **planar_stats}
    return {"graph_metrics": payload}


def compute_metrics_from_data(
    data: Data,
    *,
    graph_metrics: GraphMetrics | None = None,
    planar_metrics: PlanarGraphMetrics | None = None,
) -> Dict[str, Dict[str, float]]:
    assert data.pos is not None, "Data.pos (vertex positions) must be provided"
    assert data.edge_index is not None, "Data.edge_index must be provided"
    assert getattr(data, "edge_attr", None) is not None, "Data.edge_attr must be provided"
    gm = graph_metrics or GraphMetrics()
    pm = planar_metrics or PlanarGraphMetrics()
    sector_angles = getattr(data, "sector_angles", None)
    adjacency_boundary = getattr(data, "adjacency_b", None)
    sector_tensor = cast(Tensor, sector_angles) if sector_angles is not None else None
    return compute_metrics_from_raw(
        data.edge_index,
        data.pos,
        num_nodes=data.num_nodes,
        edge_weight=cast(Tensor, data.edge_attr),
        adjacency_boundary=cast(Tensor, adjacency_boundary) if adjacency_boundary is not None else None,
        graph_metrics=gm,
        planar_metrics=pm,
        sector_angles=sector_tensor,
    )


def compute_all_metrics_from_obs(
    obs,
    *,
    num_nodes: int | None = None,
    graph_metrics: GraphMetrics | None = None,
    planar_metrics: PlanarGraphMetrics | None = None,
    metric_subset: List[str] | None = None,
) -> Dict[str, Dict[str, float]]:
    """Convenience wrapper to compute metrics directly from a PlanarGraphEnv."""
    with torch.no_grad():
        adjacency_boundary = _to_tensor(np.asarray(obs.adj_b))
        adjacency = _to_tensor(np.asarray(obs.adj)) + adjacency_boundary
        vertex_positions = _to_tensor(np.asarray(obs.vp))
        sector_angles = _to_tensor(np.asarray(obs.sector_angles))

        # If num_nodes is specified, trim the adjacency matrix first
        if num_nodes is not None and adjacency.shape[0] > num_nodes:
            adjacency = adjacency[:num_nodes, :num_nodes]
            adjacency_boundary = adjacency_boundary[:num_nodes, :num_nodes]
            vertex_positions = vertex_positions[:num_nodes]
            if sector_angles.shape[0] > num_nodes:
                sector_angles = sector_angles[:num_nodes]

        edge_index, edge_weight = dense_to_sparse(adjacency)
        gm = graph_metrics or GraphMetrics()
        pm = planar_metrics or PlanarGraphMetrics()
        return compute_metrics_from_raw(
            edge_index,
            vertex_positions,
            sector_angles,
            adjacency_matrix=adjacency,
            adjacency_boundary=adjacency_boundary,
            edge_weight=edge_weight,
            graph_metrics=gm,
            planar_metrics=pm,
            metric_subset=metric_subset,
            num_nodes=num_nodes,
        )


METRIC_KEYS: tuple[str, ...] = (
    "num_nodes",
    "num_edges",
    "spectral_gap",
    "spectral_norm",
    "assortativity",
    "gini_coefficient",
    "clustering_coefficient",
    "triangles",
    "is_planar",
    "angular_resolution",
    "edge_length_uniformity",
    "edge_length_deviation",
    "isoperimetric_ratio",
    "rectangularity",
    "node_uniformity",
    "alpha_shape_area",
    "alpha_shape_perimeter",
)

NUMERIC_TYPES = (float, int, np.floating, np.integer)


def extract_metric_entry(raw) -> dict[str, float]:
    """Return a filtered dict containing only recognised numeric metric values."""
    if isinstance(raw, (list, tuple)):
        # Aggregate metrics across envs by taking the mean of each recognised field.
        aggregated: defaultdict[str, list[float]] = defaultdict(list)
        for item in raw:
            for key, value in extract_metric_entry(item).items():
                aggregated[key].append(value)
        averaged: dict[str, float] = {}
        for key, values in aggregated.items():
            if not values:
                continue
            finite = [v for v in values if math.isfinite(v)]
            if not finite:
                continue
            averaged[key] = float(np.mean(finite, dtype=np.float64))
        return averaged

    if not isinstance(raw, dict):
        return {}

    payload = raw.get("graph_metrics", raw)
    if not isinstance(payload, dict):
        return {}

    entry: dict[str, float] = {}
    for key in METRIC_KEYS:
        value = payload.get(key)
        if isinstance(value, NUMERIC_TYPES):
            entry[key] = float(value)
    return entry


def aggregate_metric_history(history: list[dict[str, object]]) -> dict[str, np.ndarray]:
    """Stack metric history into numpy arrays keyed by metric name."""
    aggregated: defaultdict[str, list[float]] = defaultdict(list)
    for raw_entry in history:
        if not isinstance(raw_entry, dict):
            continue
        for key, value in extract_metric_entry(raw_entry).items():
            aggregated[key].append(value)
    return {key: np.asarray(values, dtype=np.float64) for key, values in aggregated.items() if values}


def metric_history_to_results(history: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    """Convert metric history to the format expected by visualize_metrics."""
    aggregated = aggregate_metric_history(history)
    if not aggregated:
        return {}
    stats = {
        "mean": {key: float(values.mean()) for key, values in aggregated.items()},
        "min": {key: float(values.min()) for key, values in aggregated.items()},
        "max": {key: float(values.max()) for key, values in aggregated.items()},
    }
    return stats


def visualize_metrics(
    results: dict[str, dict[str, float]],
    output_dir: Path | None = None,
    baseline_metrics: dict[str, float] | None = None,
    target_metrics: dict[str, float] | None = None,
) -> None:
    """
    Visualize metrics using box plots.

    Args:
        results: Dictionary mapping graph names to their metrics, or
                 dictionary with 'mean', 'min', 'max' keys for aggregated metrics.
        output_dir: Optional directory to save figures. If None, displays interactively.
        baseline_metrics: Optional baseline values to plot as a point.
        target_metrics: Optional target values to plot as a point.
    """
    # Determine if results are aggregated statistics or individual graph metrics
    is_aggregated = all(key in {"mean", "min", "max", "std"} for key in results.keys())

    if is_aggregated:
        # Convert aggregated statistics to a format suitable for box plots
        metric_names = list(results.get("mean", {}).keys())
        # Exclude num_nodes and num_edges from visualization (different scale)
        metric_names = [name for name in metric_names if name not in ["num_nodes"]]
        if not metric_names:
            return

        # Create a 2-row grid of subplots, one per metric
        num_metrics = len(metric_names)
        num_rows = 2
        num_cols = (num_metrics + num_rows - 1) // num_rows
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
        axes = np.array(axes).reshape(num_rows, num_cols)
        means = [results.get("mean", {}).get(name, 0.0) for name in metric_names]
        mins = [results.get("min", {}).get(name, 0.0) for name in metric_names]
        maxs = [results.get("max", {}).get(name, 0.0) for name in metric_names]

        for i, name in enumerate(metric_names):
            row = i // num_cols
            col = i % num_cols
            ax = axes[row, col]
            mean_val = means[i]
            min_val = mins[i]
            max_val = maxs[i]
            box_data = [min_val, mean_val, max_val]
            bp = ax.boxplot([box_data], widths=0.6, patch_artist=True)
            for patch in bp["boxes"]:
                patch.set_facecolor("lightblue")
                patch.set_alpha(0.7)
            handles = []
            labels = []
            baseline_value = None if baseline_metrics is None else baseline_metrics.get(name)
            if baseline_value is not None and math.isfinite(baseline_value):
                handle = ax.scatter([1.0], [baseline_value], s=70, color="darkorange", zorder=3)
                handles.append(handle)
                labels.append("baseline")
            target_value = None if target_metrics is None else target_metrics.get(name)
            if target_value is not None and math.isfinite(target_value):
                handle = ax.scatter([1.0], [target_value], s=70, color="purple", zorder=3)
                handles.append(handle)
                labels.append("target")
            if handles:
                ax.legend(handles, labels, loc="best")
            ax.set_title(name.replace("_", " ").title())
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.set_xticklabels([""])

        for idx in range(num_metrics, num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            axes[row, col].axis("off")

        plt.tight_layout()

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / "aggregated_metrics.pdf", dpi=150, bbox_inches="tight")
            fig.savefig(output_dir / "aggregated_metrics.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()

    else:
        # Original behavior: results maps graph names to their metrics
        if not results:
            return

        # Collect all unique metric names
        all_metric_names = set()
        for metrics in results.values():
            all_metric_names.update(metrics.keys())

        metric_names = sorted(all_metric_names)
        if not metric_names:
            return

        # Organize data for box plots: each metric has values from different graphs
        metric_data: dict[str, list[float]] = {name: [] for name in metric_names}
        graph_names = list(results.keys())

        for graph_name in graph_names:
            metrics = results[graph_name]
            for metric_name in metric_names:
                value = metrics.get(metric_name, 0.0)
                if math.isfinite(value):
                    metric_data[metric_name].append(value)

        # Create a 2-row grid of subplots, one per metric
        num_metrics = len(metric_names)
        num_rows = 2
        num_cols = (num_metrics + num_rows - 1) // num_rows
        fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 4 * num_rows))
        axes = np.array(axes).reshape(num_rows, num_cols)

        for idx, metric_name in enumerate(metric_names):
            row = idx // num_cols
            col = idx % num_cols
            ax = axes[row, col]

            data = metric_data[metric_name]
            if data:
                bp = ax.boxplot([data], widths=0.6, patch_artist=True)
                for patch in bp["boxes"]:
                    patch.set_facecolor("lightblue")
                    patch.set_alpha(0.7)
            handles = []
            labels = []
            baseline_value = None if baseline_metrics is None else baseline_metrics.get(metric_name)
            if baseline_value is not None and math.isfinite(baseline_value):
                handle = ax.scatter([1.0], [baseline_value], s=70, color="darkorange", zorder=3)
                handles.append(handle)
                labels.append("baseline")
            target_value = None if target_metrics is None else target_metrics.get(metric_name)
            if target_value is not None and math.isfinite(target_value):
                handle = ax.scatter([1.0], [target_value], s=70, color="purple", zorder=3)
                handles.append(handle)
                labels.append("target")
            if handles:
                ax.legend(handles, labels, loc="best")
            ax.set_title(metric_name.replace("_", " ").title())
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.set_xticklabels([""])

        for idx in range(num_metrics, num_rows * num_cols):
            row = idx // num_cols
            col = idx % num_cols
            axes[row, col].axis("off")

        plt.tight_layout()

        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(output_dir / "metrics_boxplots.png", dpi=150, bbox_inches="tight")
            plt.close(fig)
        else:
            plt.show()


def save_metric_history_figures(
    history: list[dict[str, object]],
    output_dir: Path,
    baseline_metrics: dict[str, float] | None = None,
    target_metrics: dict[str, float] | None = None,
) -> None:
    """Save visual summaries (reuse visualize_metrics) for rollout metrics."""
    results = metric_history_to_results(history)
    if not results:
        return
    visualize_metrics(results, output_dir, baseline_metrics=baseline_metrics, target_metrics=target_metrics)

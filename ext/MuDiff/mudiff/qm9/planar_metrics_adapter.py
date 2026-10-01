from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
import yaml

try:
    from graph_rl.ppo.metrics import GraphMetrics, PlanarGraphMetrics
except Exception:
    GraphMetrics = None
    PlanarGraphMetrics = None

try:
    from graph_rl.utils.iso_squareness_baselines import get_iso_and_squareness_for_baselines
except Exception:
    get_iso_and_squareness_for_baselines = None


# Canonical key mapping used across CLI, YAML use-cases, and stored dataset tensors.
_METRIC_ALIASES = {
    "sl": "edge_length_uniformity",
    "triangle_count": "triangle_count",
    "triangles": "triangle_count",
    "gini": "gini_coefficient",
    "clustering": "clustering_coefficient",
}

_SUPPORTED_METRICS = {
    "num_nodes",
    "spectral_gap",
    "gini_coefficient",
    "clustering_coefficient",
    "triangle_count",
    "edge_length_uniformity",
    "edge_length_deviation",
    "angular_resolution",
    "isoperimetric_ratio",
    "rectangularity",
}


def canonicalize_metric_name(name: str) -> str:
    cleaned = str(name).strip()
    lowered = cleaned.lower()
    return _METRIC_ALIASES.get(lowered, lowered)


def canonicalize_metric_list(metric_names: Iterable[str]) -> List[str]:
    seen = set()
    ordered: List[str] = []

    for name in metric_names:
        canonical = canonicalize_metric_name(name)
        if canonical in seen:
            continue
        seen.add(canonical)
        ordered.append(canonical)

    return ordered


def resolve_use_case_file(use_case: str, use_case_dir: str | None) -> Path:
    candidate = Path(use_case).expanduser()
    if candidate.is_file():
        return candidate.resolve()

    base_dir = Path(use_case_dir).expanduser() if use_case_dir else Path(__file__).resolve().parents[1] / "configs" / "conditioning" / "use_cases"

    if candidate.suffix:
        joined = (base_dir / candidate.name).resolve()
        if joined.is_file():
            return joined
    else:
        yaml_candidate = (base_dir / f"{candidate.name}.yaml").resolve()
        if yaml_candidate.is_file():
            return yaml_candidate
        raw_candidate = (base_dir / candidate.name).resolve()
        if raw_candidate.is_file():
            return raw_candidate

    raise FileNotFoundError(f"Could not resolve use-case file for '{use_case}'.")


def load_conditioning_metrics_from_use_case(use_case: str, use_case_dir: str | None = None) -> Tuple[List[str], str]:
    use_case_path = resolve_use_case_file(use_case=use_case, use_case_dir=use_case_dir)

    with use_case_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    target_metrics = payload.get("target_metrics", {})
    if not isinstance(target_metrics, dict):
        raise ValueError(f"Expected target_metrics map in {use_case_path}.")

    metric_names = canonicalize_metric_list(target_metrics.keys())
    if not metric_names:
        raise ValueError(f"No target metrics found in {use_case_path}.")

    return metric_names, str(use_case_path)


def load_conditioning_targets_from_use_case(use_case: str, use_case_dir: str | None = None) -> Tuple[Dict[str, float], str]:
    use_case_path = resolve_use_case_file(use_case=use_case, use_case_dir=use_case_dir)

    with use_case_path.open("r", encoding="utf-8") as f:
        payload = yaml.safe_load(f) or {}

    target_metrics = payload.get("target_metrics", {})
    if not isinstance(target_metrics, dict):
        raise ValueError(f"Expected target_metrics map in {use_case_path}.")

    normalized: Dict[str, float] = {}
    for key, value in target_metrics.items():
        canonical = canonicalize_metric_name(key)
        normalized[canonical] = _safe_float(value)

    return metric_alias_dict(normalized), str(use_case_path)


def build_conditioning_context_from_targets(
    conditioning: Iterable[str],
    target_metrics: Dict[str, float],
    property_norms: Dict[str, Dict[str, torch.Tensor]] | None,
    batch_size: int,
    device,
) -> torch.Tensor:
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")

    conditioning_keys = canonicalize_metric_list(conditioning)
    if len(conditioning_keys) == 0:
        raise ValueError("Cannot build target context with empty conditioning list.")

    target_values = metric_alias_dict({canonicalize_metric_name(k): _safe_float(v) for k, v in target_metrics.items()})

    rows: List[List[float]] = []
    for _ in range(batch_size):
        row: List[float] = []
        for key in conditioning_keys:
            raw_value = target_values.get(key, None)

            if raw_value is None:
                if property_norms is not None and key in property_norms:
                    raw_value = _safe_float(property_norms[key]["mean"])
                else:
                    raw_value = 0.0

            if property_norms is not None and key in property_norms:
                mean = _safe_float(property_norms[key]["mean"])
                mad = max(_safe_float(property_norms[key]["mad"], 1.0), 1e-8)
                normalized = (raw_value - mean) / mad
            else:
                normalized = raw_value

            row.append(float(normalized))

        rows.append(row)

    return torch.tensor(rows, dtype=torch.float32, device=device)


def metric_alias_dict(metric_values: Dict[str, float]) -> Dict[str, float]:
    values = dict(metric_values)

    if "triangle_count" in values and "triangles" not in values:
        values["triangles"] = values["triangle_count"]
    if "triangles" in values and "triangle_count" not in values:
        values["triangle_count"] = values["triangles"]

    return values


def _safe_float(value, default: float = 0.0) -> float:
    if torch.is_tensor(value):
        if value.numel() == 0:
            return default
        value = value.detach().cpu().reshape(-1)[0].item()

    try:
        numeric = float(value)
    except Exception:
        return default

    if not math.isfinite(numeric):
        return default
    return numeric


def _num_nodes(graph) -> int:
    num_nodes = getattr(graph, "num_nodes", None)
    if num_nodes is not None:
        return int(num_nodes)

    x = getattr(graph, "x", None)
    if x is not None:
        return int(x.size(0))

    edge_index = getattr(graph, "edge_index", None)
    if edge_index is not None and edge_index.numel() > 0:
        return int(edge_index.max().item()) + 1

    return 0


def _dense_adjacency(graph, num_nodes: int) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)

    edge_index = getattr(graph, "edge_index", None)
    if edge_index is None or edge_index.numel() == 0:
        return adj

    src = edge_index[0].long()
    dst = edge_index[1].long()
    valid = src != dst
    src = src[valid]
    dst = dst[valid]

    if src.numel() == 0:
        return adj

    adj[src, dst] = 1.0
    adj[dst, src] = 1.0
    return adj


def _boundary_adjacency(graph, num_nodes: int) -> torch.Tensor | None:
    boundary = getattr(graph, "adjacency_b", None)
    if boundary is None:
        boundary = getattr(graph, "adj_b", None)

    if boundary is None:
        return None

    if not torch.is_tensor(boundary):
        boundary = torch.tensor(boundary)

    boundary = boundary.to(dtype=torch.float64)
    if boundary.dim() != 2:
        return None

    if boundary.size(0) < num_nodes or boundary.size(1) < num_nodes:
        return None

    return boundary[:num_nodes, :num_nodes]


def _edge_length_uniformity_fallback(adjacency: torch.Tensor, positions: torch.Tensor) -> float:
    if positions is None:
        return 0.0

    idx = torch.nonzero(torch.triu(adjacency > 0, diagonal=1), as_tuple=False)
    if idx.numel() == 0:
        return 0.0

    p = positions.to(dtype=torch.float64)
    lengths = torch.linalg.norm(p[idx[:, 0], :2] - p[idx[:, 1], :2], dim=1)
    if lengths.numel() == 0:
        return 0.0

    mean_len = lengths.mean()
    if mean_len.item() <= 1e-8:
        return 0.0

    rel_dev = (lengths - mean_len) / mean_len
    return _safe_float(torch.sqrt(torch.mean(rel_dev.pow(2))))


def _edge_length_deviation_fallback(adjacency: torch.Tensor, positions: torch.Tensor) -> float:
    idx = torch.nonzero(torch.triu(adjacency > 0, diagonal=1), as_tuple=False)
    if idx.numel() == 0:
        return 0.0

    p = positions.to(dtype=torch.float64)
    lengths = torch.linalg.norm(p[idx[:, 0], :2] - p[idx[:, 1], :2], dim=1)
    if lengths.numel() == 0:
        return 0.0

    ideal = lengths.mean()
    if ideal.item() <= 1e-8:
        return 0.0

    relative = torch.abs(lengths - ideal) / ideal
    return _safe_float(1.0 / (1.0 + relative.mean()))


def _angular_resolution_fallback(adjacency: torch.Tensor, positions: torch.Tensor | None) -> float:
    pts = _extract_xy_positions(positions)
    if pts is None or pts.size(0) == 0:
        return 0.0

    adj_bool = adjacency > 0
    min_angle = math.pi
    found = False

    for node_idx in range(adj_bool.size(0)):
        neighbors = torch.nonzero(adj_bool[node_idx], as_tuple=False).flatten()
        if neighbors.numel() < 2:
            continue

        center = pts[node_idx]
        vecs = pts[neighbors] - center.unsqueeze(0)
        if vecs.numel() == 0:
            continue

        angles = torch.atan2(vecs[:, 1], vecs[:, 0])
        angles, _ = torch.sort(torch.remainder(angles, 2.0 * math.pi))
        wrapped = torch.cat([angles, angles[:1] + 2.0 * math.pi], dim=0)
        gaps = wrapped[1:] - wrapped[:-1]
        if gaps.numel() == 0:
            continue

        node_min = float(torch.min(gaps).item())
        min_angle = min(min_angle, node_min)
        found = True

    if not found:
        return 0.0

    normalized = min_angle / math.pi
    return float(min(1.0, max(0.0, normalized)))


def _compute_topology_fallback(adjacency: torch.Tensor) -> Dict[str, float]:
    degrees = adjacency.sum(dim=1)

    metrics = {
        "spectral_gap": 0.0,
        "gini_coefficient": 0.0,
        "triangle_count": 0.0,
        "clustering_coefficient": 0.0,
    }

    if adjacency.numel() == 0:
        return metrics

    if adjacency.size(0) > 1:
        deg_safe = torch.where(degrees > 0, degrees, torch.ones_like(degrees))
        inv_sqrt = torch.where(degrees > 0, 1.0 / torch.sqrt(deg_safe), torch.zeros_like(degrees))
        eye = torch.eye(adjacency.size(0), dtype=adjacency.dtype)
        laplacian = eye - inv_sqrt.unsqueeze(1) * adjacency * inv_sqrt.unsqueeze(0)
        eigvals = torch.linalg.eigvalsh(laplacian)
        positive = eigvals[eigvals > 1e-8]
        if positive.numel() > 0:
            metrics["spectral_gap"] = _safe_float(positive.min())

    if degrees.numel() > 0:
        deg_sorted = torch.sort(degrees)[0]
        deg_sum = deg_sorted.sum()
        if deg_sum.item() > 1e-8:
            n = deg_sorted.numel()
            index = torch.arange(1, n + 1, dtype=deg_sorted.dtype)
            gini = (2.0 * torch.sum(index * deg_sorted) / (n * deg_sum)) - ((n + 1.0) / n)
            metrics["gini_coefficient"] = _safe_float(gini)

    trace_a3 = torch.trace(torch.linalg.matrix_power(adjacency, 3))
    triangles = trace_a3 / 6.0
    metrics["triangle_count"] = _safe_float(triangles)

    triplets = torch.sum(degrees * (degrees - 1.0) / 2.0)
    if triplets.item() > 1e-8:
        metrics["clustering_coefficient"] = _safe_float((3.0 * triangles) / triplets)

    return metrics


def _extract_xy_positions(positions: torch.Tensor | None) -> torch.Tensor | None:
    if positions is None:
        return None
    if torch.is_tensor(positions):
        pts = positions[:positions.shape[0], :2]
        if pts.dim() == 1:
            pts = pts.unsqueeze(0)
        return pts
    arr = torch.tensor(positions)
    pts = arr[: arr.shape[0], :2]
    if pts.dim() == 1:
        pts = pts.unsqueeze(0)
    return pts


def _convex_hull_2d(points: torch.Tensor) -> torch.Tensor:
    pts = points.detach().cpu().numpy()
    if pts.shape[0] < 3:
        return torch.from_numpy(pts)

    unique_pts = np.unique(pts, axis=0)
    if unique_pts.shape[0] < 3:
        return torch.from_numpy(unique_pts)

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    sorted_pts = sorted(unique_pts.tolist())
    lower = []
    for p in sorted_pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(sorted_pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]
    if len(hull) == 0:
        hull = lower
    hull_arr = np.asarray(hull, dtype=np.float64)
    return torch.from_numpy(hull_arr)


def _polygon_area_perimeter(points: torch.Tensor) -> Tuple[float, float]:
    coords = points.detach().cpu().numpy()
    if coords.shape[0] < 3:
        return 0.0, 0.0

    shifted = np.roll(coords, -1, axis=0)
    area = 0.5 * np.abs(np.sum(coords[:, 0] * shifted[:, 1] - coords[:, 1] * shifted[:, 0]))
    perim = float(np.sum(np.linalg.norm(shifted - coords, axis=1)))
    return float(area), perim


def _isoperimetric_ratio_from_positions(positions: torch.Tensor | None) -> float:
    metrics = _baseline_iso_squareness_from_positions(positions)
    if metrics is not None:
        return metrics["iso"]
    pts = _extract_xy_positions(positions)
    if pts is None or pts.size(0) < 3:
        return 0.0
    hull = _convex_hull_2d(pts)
    if hull.size(0) < 3:
        return 0.0
    area, perim = _polygon_area_perimeter(hull)
    if perim <= 1e-8:
        return 0.0
    ratio = (4.0 * math.pi * area) / (perim * perim + 1e-6)
    return float(min(1.0, max(0.0, ratio)))


def _rectangularity_from_positions(positions: torch.Tensor | None) -> float:
    metrics = _baseline_iso_squareness_from_positions(positions)
    if metrics is not None:
        return metrics["squareness"]
    pts = _extract_xy_positions(positions)
    if pts is None or pts.size(0) < 3:
        return 0.0
    hull = _convex_hull_2d(pts)
    if hull.size(0) < 3:
        return 0.0
    area, _ = _polygon_area_perimeter(hull)
    if area <= 1e-8:
        return 0.0
    coords = hull.detach().cpu().numpy()
    min_x, min_y = float(np.min(coords[:, 0])), float(np.min(coords[:, 1]))
    max_x, max_y = float(np.max(coords[:, 0])), float(np.max(coords[:, 1]))
    width = max_x - min_x
    height = max_y - min_y
    if width <= 1e-6 or height <= 1e-6:
        return 0.0
    bbox_area = width * height
    ratio = area / (bbox_area + 1e-6)
    return float(min(1.0, max(0.0, ratio)))


def _baseline_iso_squareness_from_positions(positions: torch.Tensor | None) -> Dict[str, float] | None:
    if positions is None or get_iso_and_squareness_for_baselines is None:
        return None

    pts = _extract_xy_positions(positions)
    if pts is None or pts.size(0) < 3:
        return None

    try:
        result = get_iso_and_squareness_for_baselines(pts.detach().cpu().numpy())
    except Exception:
        print("Error: Failed to compute baseline isoperimetric ratio and squareness.")
        return None

    metrics = result[0] if isinstance(result, tuple) else result
    if not isinstance(metrics, dict):
        return None

    try:
        return {
            "iso": _safe_float(metrics["iso"]),
            "squareness": _safe_float(metrics["squareness"]),
        }
    except Exception:
        print("Error: Baseline isoperimetric ratio and squareness metrics missing expected keys.")
        return None


def compute_graph_metric_dict(graph) -> Dict[str, float]:
    metrics = {key: 0.0 for key in _SUPPORTED_METRICS}

    num_nodes = _num_nodes(graph)
    metrics["num_nodes"] = float(num_nodes)

    if num_nodes <= 0:
        return metric_alias_dict(metrics)

    adjacency = _dense_adjacency(graph, num_nodes=num_nodes)

    if GraphMetrics is not None:
        try:
            gm = GraphMetrics()
            triangle_count = _safe_float(gm.triangle_count(adjacency))
            triplets = _safe_float(gm.triplet_count(adjacency))
            metrics["spectral_gap"] = _safe_float(gm.spectral_gap(adjacency))
            metrics["gini_coefficient"] = _safe_float(gm._gini_from_degrees(adjacency))
            metrics["triangle_count"] = triangle_count
            metrics["clustering_coefficient"] = _safe_float(gm.clustering_coefficient(triangle_count, triplets))
        except Exception:
            metrics.update(_compute_topology_fallback(adjacency))
    else:
        metrics.update(_compute_topology_fallback(adjacency))

    positions = getattr(graph, "pos", None)
    if positions is not None and torch.is_tensor(positions):
        if positions.dim() == 1:
            positions = positions.unsqueeze(-1)
        positions = positions[:num_nodes].to(dtype=torch.float64)

    fallback_isoperimetric = _isoperimetric_ratio_from_positions(positions)
    fallback_rectangularity = _rectangularity_from_positions(positions)
    fallback_angular_resolution = _angular_resolution_fallback(adjacency, positions)

    if PlanarGraphMetrics is not None and positions is not None:
        pm = PlanarGraphMetrics()
        sector_angles = getattr(graph, "sector_angles", None)
        if sector_angles is not None and torch.is_tensor(sector_angles):
            sector_angles = sector_angles[:num_nodes]

        metrics["edge_length_uniformity"] = _safe_float(pm.edge_length_uniformity(adjacency, positions))
        metrics["edge_length_deviation"] = _safe_float(pm.edge_length_deviation(adjacency, positions))
        try:
            angular_resolution = _safe_float(pm.angular_resolution(sector_angles=sector_angles))
        except Exception:
            angular_resolution = fallback_angular_resolution
        if not np.isfinite(angular_resolution) or angular_resolution == 0.0:
            angular_resolution = fallback_angular_resolution
        metrics["angular_resolution"] = angular_resolution

        boundary = _boundary_adjacency(graph, num_nodes=num_nodes)
        if boundary is not None:
            metrics["isoperimetric_ratio"] = _safe_float(pm.isoperimetric_ratio(boundary, positions))
            metrics["rectangularity"] = _safe_float(pm.rectangularity(adjacency, boundary, positions))
        else:
            metrics["isoperimetric_ratio"] = fallback_isoperimetric
            metrics["rectangularity"] = fallback_rectangularity
    elif positions is not None:
        metrics["edge_length_uniformity"] = _edge_length_uniformity_fallback(adjacency, positions)
        metrics["edge_length_deviation"] = _edge_length_deviation_fallback(adjacency, positions)
        metrics["angular_resolution"] = fallback_angular_resolution
        metrics["isoperimetric_ratio"] = fallback_isoperimetric
        metrics["rectangularity"] = fallback_rectangularity

    return metric_alias_dict(metrics)


def validate_conditioning_metrics(metrics: Iterable[str]) -> List[str]:
    canonical = canonicalize_metric_list(metrics)
    unknown = [key for key in canonical if key not in _SUPPORTED_METRICS]
    if unknown:
        raise ValueError(
            "Unsupported conditioning metrics for planar dataset: "
            + ", ".join(sorted(unknown))
            + "."
        )
    return canonical

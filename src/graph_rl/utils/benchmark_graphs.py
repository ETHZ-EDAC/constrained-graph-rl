"""
Benchmark Evaluation for Graph Generators.

This module provides functions for comprehensive evaluation and comparison
of various graph generation algorithms. It loads datasets, computes metrics,
and creates visualizations for benchmark analysis.

For actual graph generation functions, see graph_generators.py.
"""

from __future__ import annotations
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union, cast
from dataclasses import replace
from itertools import product
import warnings
from pathlib import Path
import json

import numpy as np
import networkx as nx
from torch_geometric.data import Data
import torch
import matplotlib.pyplot as plt
import jax.numpy as jnp
from torch_geometric.utils import to_networkx

# Import helper functions
from graph_rl.utils.plotting import plot_graph
from graph_rl.ppo.metrics import compute_metrics_from_data
from graph_rl.ppo.planar_geometry import minimum_angle_from_sector_angles
from graph_rl.ops.edge_intersections import edge_batch_intersection_loss
from graph_rl.utils.graph_helpers import (
    GraphGeneratorConfig,
    add_node_positions,
    compute_sector_angles,
    plot_benchmark_graphs,
    plot_metric_comparison,
    plot_metric_heatmap,
    load_dataset,
    evaluate_lst_datasets,
)
from graph_rl.utils.adjacency_utils import get_edge_lengths
from graph_rl.utils.graph_generators import get_generators_dict
from graph_rl.utils.utils import GrammarConstraints, load_config


EPSILON = 1e-8
EDGE_INTERSECTION_TOL = 1e-4
DEFAULT_CONSTRAINTS_PATH = Path(__file__).resolve().parents[3] / "conf" / "constraints" / "default.yaml"
BENCHMARK_CONSTRAINTS = load_config(DEFAULT_CONSTRAINTS_PATH)


def _ensure_benchmark_edge_attr(data: Data) -> None:
    if getattr(data, "edge_attr", None) is None:
        num_edges = int(data.edge_index.shape[1]) if data.edge_index is not None else 0
        data.edge_attr = torch.ones((num_edges, 1), dtype=torch.float32)


def _ensure_benchmark_positions(data: Data, *, seed: int = 42, scale: float = 10.0) -> Data:
    prepared = data.clone()
    if prepared.edge_index is None:
        raise ValueError("Data.edge_index must be present for benchmark evaluation")
    if prepared.pos is not None:
        _ensure_benchmark_edge_attr(prepared)
        return prepared
    graph = to_networkx(prepared, to_undirected=True)
    graph = add_node_positions(graph, method="spring", scale=scale, seed=seed)
    pos_dict = nx.get_node_attributes(graph, "pos")
    num_nodes = int(prepared.num_nodes) if prepared.num_nodes is not None else graph.number_of_nodes()
    prepared.pos = torch.tensor([pos_dict[i] for i in range(num_nodes)], dtype=torch.float32)
    prepared.num_nodes = num_nodes
    _ensure_benchmark_edge_attr(prepared)
    return prepared


def _prepared_adjacency(data: Data) -> torch.Tensor:
    num_nodes = int(data.num_nodes) if data.num_nodes is not None else int(data.pos.shape[0])
    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adjacency[data.edge_index[0], data.edge_index[1]] = 1.0
    adjacency[data.edge_index[1], data.edge_index[0]] = 1.0
    adjacency.fill_diagonal_(0.0)
    return adjacency


def _upper_tri_edge_index(adjacency: torch.Tensor) -> torch.Tensor:
    return torch.triu(adjacency > 0, diagonal=1).nonzero(as_tuple=False)


def _scale_positions_to_min_edge_length(data: Data, *, edge_length_min: float) -> Tuple[Data, float, float]:
    prepared = data.clone()
    adjacency = _prepared_adjacency(prepared)
    edge_index_upper = _upper_tri_edge_index(adjacency)
    if edge_index_upper.numel() == 0:
        return prepared, 0.0, 1.0
    edge_lengths = get_edge_lengths(
        jnp.asarray(prepared.pos.detach().cpu().numpy()),
        jnp.asarray(edge_index_upper.detach().cpu().numpy()),
    )
    min_length = float(jnp.min(edge_lengths))
    if min_length <= EPSILON:
        raise ValueError("Graph contains a zero-length edge and cannot be benchmark-scaled")
    scale = float(edge_length_min) / min_length
    prepared.pos = prepared.pos * scale
    return prepared, min_length, scale


def _edge_length_bounds(data: Data) -> Tuple[float, float]:
    adjacency = _prepared_adjacency(data)
    edge_index_upper = _upper_tri_edge_index(adjacency)
    if edge_index_upper.numel() == 0:
        return 0.0, 0.0
    edge_lengths = get_edge_lengths(
        jnp.asarray(data.pos.detach().cpu().numpy()),
        jnp.asarray(edge_index_upper.detach().cpu().numpy()),
    )
    return float(jnp.min(edge_lengths)), float(jnp.max(edge_lengths))


def _max_degree_from_undirected_edges(adjacency: torch.Tensor) -> int:
    edge_index_upper = _upper_tri_edge_index(adjacency)
    if edge_index_upper.numel() == 0:
        return 0
    degree_counts = torch.bincount(edge_index_upper.reshape(-1), minlength=adjacency.shape[0])
    return int(degree_counts.max().item())


def _edge_intersection_count(data: Data) -> float:
    adjacency = _prepared_adjacency(data)
    edge_index_upper = _upper_tri_edge_index(adjacency)
    if int(edge_index_upper.shape[0]) <= 1:
        return 0.0
    pos_xy = data.pos[:, :2]
    edge_segments = pos_xy[edge_index_upper]
    edge_vectors = torch.zeros((int(edge_index_upper.shape[0]), 2, 3), dtype=pos_xy.dtype)
    edge_vectors[:, :, :2] = edge_segments
    losses = edge_batch_intersection_loss(jnp.asarray(edge_vectors.detach().cpu().numpy()))
    return float(jnp.sum(losses))


def _constraint_rank(report: Dict[str, Any]) -> Tuple[int, float]:
    return int(report.get("num_constraints_satisfied", 0)), float(report.get("constraint_score", 0.0))


def _reward_value(entry: Dict[str, Any]) -> float:
    value = entry.get("reward")
    if value is None:
        return -float("inf")
    return float(value)


def _entry_rank(entry: Dict[str, Any]) -> Tuple[Tuple[int, float], float]:
    return _constraint_rank(entry.get("constraint_report", {})), _reward_value(entry)


def _append_metric_history(history: Dict[str, List[float]], metrics: Dict[str, Any]) -> None:
    for key, value in metrics.items():
        if isinstance(value, (int, float, np.floating, np.integer)):
            history.setdefault(key, []).append(float(value))


def _summarize_metric_history(history: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    for key, values in history.items():
        if values:
            arr = np.asarray(values, dtype=np.float64)
            summary[key] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
    return summary


def prepare_graph_for_benchmark_evaluation(
    data: Data,
    *,
    seed: int = 42,
    constraints: GrammarConstraints = BENCHMARK_CONSTRAINTS,
    verbose: bool = False,
    strict: bool = True,
) -> Tuple[Data, Dict[str, float], Dict[str, Any]]:
    if verbose:
        print("\n[benchmark-prep] starting graph preprocessing")
        print(f"[benchmark-prep] seed={seed}")
        print(f"[benchmark-prep] has_pos={data.pos is not None}")
        print(f"[benchmark-prep] num_nodes={data.num_nodes}")
    prepared = _ensure_benchmark_positions(data, seed=seed)
    if verbose and data.pos is None:
        print("[benchmark-prep] assigned spring layout because graph had no embedding")
    prepared, original_min_edge_length, scale = _scale_positions_to_min_edge_length(
        prepared,
        edge_length_min=float(constraints.edge_length_min),
    )
    if verbose:
        print(
            "[benchmark-prep] scaled positions "
            f"(original_min_edge_length={original_min_edge_length:.6f}, scale={scale:.6f})"
        )
    prepared.sector_angles = compute_sector_angles(prepared.edge_index, prepared.pos, prepared.num_nodes)
    adjacency = _prepared_adjacency(prepared)
    min_angle = float(minimum_angle_from_sector_angles(prepared.sector_angles))
    _, max_edge_length = _edge_length_bounds(prepared)
    max_degree = _max_degree_from_undirected_edges(adjacency)
    intersection_loss = _edge_intersection_count(prepared)
    passes_minimum_angle = min_angle >= float(constraints.sector_eps)
    passes_edge_length = max_edge_length <= float(constraints.edge_length_max)
    passes_max_degree = max_degree <= int(constraints.max_vertex_degree)
    passes_edge_intersection = intersection_loss <= EDGE_INTERSECTION_TOL
    num_constraints_satisfied = sum(
        (
            int(passes_minimum_angle),
            int(passes_edge_length),
            int(passes_max_degree),
            int(passes_edge_intersection),
        )
    )
    constraint_report = {
        "status": "ok" if num_constraints_satisfied == 4 else "constraint_failed",
        "minimum_angle": min_angle,
        "edge_length_min_scaled_to": float(constraints.edge_length_min),
        "edge_length_max": max_edge_length,
        "max_degree": max_degree,
        "edge_intersection_loss": intersection_loss,
        "passes_minimum_angle": passes_minimum_angle,
        "passes_edge_length": passes_edge_length,
        "passes_max_degree": passes_max_degree,
        "passes_edge_intersection": passes_edge_intersection,
        "num_constraints_satisfied": num_constraints_satisfied,
        "constraint_score": num_constraints_satisfied / 4.0,
        "constraints_ok": num_constraints_satisfied == 4,
    }
    if verbose:
        print(
            "[benchmark-prep] constraints "
            f"minimum_angle={min_angle:.6f} edge_length_max={max_edge_length:.6f} "
            f"max_degree={max_degree} edge_intersection_loss={intersection_loss:.6f}"
        )
        print(f"[benchmark-prep] constraints_ok={constraint_report['constraints_ok']}")
    metrics = compute_metrics_from_data(prepared).get("graph_metrics", {})
    if verbose:
        print(
            "[benchmark-prep] computed metrics: "
            + ", ".join(f"{k}={float(v):.6f}" for k, v in sorted(metrics.items()) if isinstance(v, (int, float)))
        )
    if strict and not constraint_report["constraints_ok"]:
        raise RuntimeError(f"Benchmark constraints failed: {constraint_report}")
    return prepared, metrics, constraint_report


def _normalize_param_value(value: Any) -> Any:
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (np.floating, float)):
        return float(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def _generator_parameter_summary(config: GraphGeneratorConfig, gen_name: str) -> Dict[str, Any]:
    params: Dict[str, Any] = {"num_nodes": config.num_nodes}

    if gen_name == "erdos_renyi_gnp":
        params.update(
            {
                "edge_probability": config.edge_probability,
                "ensure_connected": config.ensure_connected,
                "layout_method": config.layout_method,
            }
        )
    elif gen_name == "erdos_renyi_gnm":
        inferred_edges = (
            config.num_edges
            if config.num_edges is not None
            else int(0.5 * config.num_nodes * (config.num_nodes - 1) * config.edge_probability)
        )
        params.update(
            {
                "num_edges": inferred_edges,
                "ensure_connected": config.ensure_connected,
                "layout_method": config.layout_method,
            }
        )
    elif gen_name == "planar_er":
        params.update(
            {
                "edge_probability": config.planar_er_edge_probability
                if config.planar_er_edge_probability is not None
                else config.edge_probability,
                "max_attempts": config.planar_max_attempts,
                "min_edges": config.planar_min_edges,
            }
        )
    elif gen_name == "boltzmann_planar":
        params.update(
            {
                "epsilon": config.boltzmann_epsilon,
                "require_connected": config.ensure_connected,
                "layout_method": config.layout_method,
            }
        )
    elif gen_name == "barabasi_albert":
        params.update(
            {
                "num_attachments": config.num_attachments,
                "layout_method": config.layout_method,
            }
        )
    elif gen_name == "ergm":
        params.update(
            {
                "edge_param": config.ergm_edge_param,
                "triangle_param": config.ergm_triangle_param,
                "star_param": config.ergm_star_param,
                "mcmc_steps": config.ergm_mcmc_steps,
                "burnin": config.ergm_burnin,
            }
        )
    elif gen_name == "delaunay":
        params.update(
            {
                "boundary": config.delaunay_boundary,
                "layout_scale": config.layout_scale,
            }
        )
    elif gen_name == "lattice_grid":
        params.update(
            {
                "grid_rows": config.grid_rows,
                "grid_cols": config.grid_cols,
                "grid_circular": config.grid_circular,
            }
        )
    elif gen_name == "plantri":
        params.update(
            {
                "connectivity": config.plantri_connectivity,
                "min_degree": config.plantri_min_degree,
            }
        )

    return {k: _normalize_param_value(v) for k, v in params.items() if v is not None}


def compute_relative_difference(value1: float, value2: float) -> float:
    """Return the symmetric relative difference between two values."""
    diff = abs(value1 - value2)
    magnitude = abs(value1) + abs(value2) + EPSILON
    return diff / magnitude


def compute_metric_similarity(
    target_metrics: Dict[str, float],
    candidate_metrics: Dict[str, float],
    metric_subset: Iterable[str],
) -> Tuple[float, Dict[str, float]]:
    """Compute similarity score and per-metric differences for the chosen subset."""
    differences: Dict[str, float] = {}
    total_diff = 0.0

    for metric_name in metric_subset:
        if metric_name not in target_metrics or metric_name not in candidate_metrics:
            raise ValueError(f"Metric '{metric_name}' missing in comparison inputs")

        target_val = target_metrics[metric_name]
        candidate_val = candidate_metrics[metric_name]

        if not (np.isfinite(target_val) and np.isfinite(candidate_val)):
            raise ValueError(
                f"Non-finite metric '{metric_name}': target={target_val}, candidate={candidate_val}"
            )

        rel_diff = compute_relative_difference(target_val, candidate_val)
        differences[metric_name] = rel_diff
        total_diff += rel_diff

    if differences:
        avg_diff = total_diff / len(differences)
        similarity = 1.0 - avg_diff
    else:
        similarity = 0.0

    return similarity, differences


def ensure_planarity_metric(data: Data, metrics: Dict[str, Any]) -> float:
    value = metrics.get("is_planar")
    if value is not None:
        return float(value)
    graph_nx = to_networkx(data, to_undirected=True)
    return 1.0 if nx.is_planar(graph_nx) else 0.0


def metric_reward(
    *,
    target_metrics: Dict[str, float],
    candidate_metrics: Dict[str, float],
    metric_reward_weight: float,
) -> Tuple[float, Dict[str, float]]:
    metric_subset = [k for k in target_metrics.keys() if k != "num_nodes"]
    similarity, differences = compute_metric_similarity(target_metrics, candidate_metrics, metric_subset)
    return float(similarity * metric_reward_weight), differences


def find_best_from_generators(
    base_config: GraphGeneratorConfig,
    *,
    target_metrics: Dict[str, float],
    metric_reward_weight: float,
    num_samples: int,
    seed: int,
    include_generators: Optional[Iterable[str]] = None,
    verbose: bool = False,
    require_planar: bool = True,
) -> Tuple[dict[str, Any], Optional[Data]]:
    names = list(get_generators_dict(base_config, seed=seed).keys())
    if include_generators is not None:
        names = [n for n in names if n in set(include_generators)]
    if not names:
        raise ValueError("No generators available after applying filters.")
    names.sort()
    best: dict[str, Any] = {
        "source": "generator",
        "generator_name": None,
        "sample_index": None,
        "seed": None,
        "reward": -float("inf"),
        "differences": {},
        "candidate_metrics": {},
        "constraint_report": {},
        "planarity": None,
        "planarity_score": None,
        "evaluated_generators": list(names),
        "per_generator_best": {},
    }
    needs_sector_angles = any(k in target_metrics for k in ("angular_resolution",))
    best_data: Optional[Data] = None
    best_fallback: Optional[dict[str, Any]] = None
    best_fallback_data: Optional[Data] = None
    per_generator_best: dict[str, dict[str, Any]] = {}

    for gen_name in names:
        if verbose:
            print(f"\nEvaluating {gen_name}... ({int(num_samples)} samples)")
        entry: dict[str, Any] = {
            "generator_name": gen_name,
            "parameters": _generator_parameter_summary(base_config, gen_name),
            "attempted_samples": 0,
            "planar_samples": 0,
            "accepted_samples": 0,
            "status": "pending",
            "reward": None,
            "sample_index": None,
            "seed": None,
            "differences": {},
            "candidate_metrics": {},
            "constraint_report": {},
            "planarity": None,
            "planarity_score": None,
            "metric_statistics": {},
            "best_valid_reward": None,
            "best_valid_seed": None,
            "best_valid_sample_index": None,
            "best_valid_candidate_metrics": {},
            "best_valid_differences": {},
            "best_valid_constraint_report": {},
            "config_overrides": {},
        }
        metric_history: Dict[str, List[float]] = {}
        fallback_entry: dict[str, Any] = {
            "generator_name": gen_name,
            "status": "no_valid_sample",
            "constraint_report": {},
            "reward": -float("inf"),
            "candidate_metrics": {},
            "differences": {},
            "sample_index": None,
            "seed": None,
            "metric_statistics": {},
            "config_overrides": {},
        }
        fallback_data_for_generator: Optional[Data] = None
        generator_unavailable = False

        for i in range(int(num_samples)):
            sample_seed = int(seed) + int(i)
            try:
                entry["attempted_samples"] += 1
                data = get_generators_dict(base_config, seed=sample_seed)[gen_name]()
                if data.edge_index is None:
                    continue
                data, cand_metrics, constraint_report = prepare_graph_for_benchmark_evaluation(
                    data,
                    seed=sample_seed,
                    constraints=BENCHMARK_CONSTRAINTS,
                    strict=False,
                    verbose=False,
                )
                if needs_sector_angles and (not hasattr(data, "sector_angles") or data.sector_angles is None):
                    pos = cast(torch.Tensor, data.pos)
                    edge_index = cast(torch.Tensor, data.edge_index)
                    num_nodes = int(data.num_nodes) if data.num_nodes is not None else int(pos.shape[0])
                    data.sector_angles = compute_sector_angles(edge_index, pos, num_nodes)
                planarity_score = ensure_planarity_metric(data, cand_metrics)
                cand_metrics["is_planar"] = float(planarity_score)
                if planarity_score < 0.5:
                    continue
                reward, diffs = metric_reward(
                    target_metrics=target_metrics,
                    candidate_metrics=cand_metrics,
                    metric_reward_weight=metric_reward_weight,
                )
                entry["planar_samples"] += 1
                _append_metric_history(metric_history, cand_metrics)
                _append_metric_history(metric_history, {"reward": reward})
                candidate_entry = {"reward": float(reward), "constraint_report": constraint_report}
                if _entry_rank(candidate_entry) > _entry_rank(fallback_entry):
                    fallback_entry = {
                        "generator_name": gen_name,
                        "status": "fallback",
                        "constraint_report": constraint_report,
                        "reward": float(reward),
                        "candidate_metrics": {k: float(v) for k, v in cand_metrics.items() if isinstance(v, (int, float))},
                        "differences": {k: float(v) for k, v in diffs.items()},
                        "sample_index": int(i),
                        "seed": int(sample_seed),
                        "metric_statistics": {},
                        "planarity": True,
                        "planarity_score": float(planarity_score),
                        "config_overrides": {},
                    }
                    fallback_data_for_generator = data
                if constraint_report["constraints_ok"]:
                    entry["accepted_samples"] += 1
                    if entry["best_valid_reward"] is None or reward > float(entry["best_valid_reward"]):
                        entry["best_valid_reward"] = float(reward)
                        entry["best_valid_seed"] = int(sample_seed)
                        entry["best_valid_sample_index"] = int(i)
                        entry["best_valid_candidate_metrics"] = {k: float(v) for k, v in cand_metrics.items() if isinstance(v, (int, float))}
                        entry["best_valid_differences"] = {k: float(v) for k, v in diffs.items()}
                        entry["best_valid_constraint_report"] = constraint_report
                if entry["sample_index"] is None or _entry_rank(candidate_entry) > _entry_rank(entry):
                    entry.update(
                        {
                            "reward": float(reward),
                            "sample_index": int(i),
                            "seed": int(sample_seed),
                            "differences": {k: float(v) for k, v in diffs.items()},
                            "candidate_metrics": {k: float(v) for k, v in cand_metrics.items() if isinstance(v, (int, float))},
                            "constraint_report": constraint_report,
                            "planarity": True,
                            "planarity_score": float(planarity_score),
                            "status": "ok",
                        }
                    )
                if best["sample_index"] is None or _entry_rank(candidate_entry) > _entry_rank(best):
                    best.update(
                        {
                            "generator_name": gen_name,
                            "sample_index": int(i),
                            "seed": int(sample_seed),
                            "reward": float(reward),
                            "differences": {k: float(v) for k, v in diffs.items()},
                            "candidate_metrics": {k: float(v) for k, v in cand_metrics.items() if isinstance(v, (int, float))},
                            "constraint_report": constraint_report,
                            "planarity": True,
                            "planarity_score": float(planarity_score),
                            "config_overrides": {},
                        }
                    )
                    best_data = data
            except ImportError:
                generator_unavailable = True
                break
            except (RuntimeError, ValueError, KeyError, IndexError, TypeError, OSError):
                continue

        if generator_unavailable:
            entry["status"] = "unavailable"
        elif entry["accepted_samples"] == 0:
            if fallback_entry["sample_index"] is not None:
                entry.update(fallback_entry)
            else:
                entry["status"] = "no_planar_samples"
        else:
            entry["status"] = "ok"
        entry["metric_statistics"] = _summarize_metric_history(metric_history)
        per_generator_best[gen_name] = entry
        if entry["status"] == "fallback" and (best_fallback is None or _entry_rank(entry) > _entry_rank(best_fallback)):
            best_fallback = entry.copy()
            best_fallback_data = fallback_data_for_generator

    if best_data is None:
        if best_fallback is None:
            raise RuntimeError("No planar graph generated by available generators")
        best.update(best_fallback)
        best["status"] = "fallback"
        best_data = best_fallback_data

    best["per_generator_best"] = per_generator_best
    return best, best_data


def _expand_override_grid(overrides: Dict[str, Any]) -> List[Dict[str, Any]]:
    if not overrides:
        return [{}]

    keys: List[str] = []
    option_lists: List[List[Any]] = []

    for key, value in overrides.items():
        if isinstance(value, (list, tuple)):
            options = list(value)
        elif isinstance(value, set):
            options = list(sorted(value))
        else:
            options = [value]

        if not options:
            options = [None]

        keys.append(key)
        option_lists.append(options)

    combos: List[Dict[str, Any]] = []
    for combination in product(*option_lists):
        combo = {key: val for key, val in zip(keys, combination)}
        combos.append(combo)

    return combos or [{}]


def run_generator_batches(
    base_config: GraphGeneratorConfig,
    batches: Iterable[Dict[str, Any]],
    *,
    target_metrics: Dict[str, float],
    metric_reward_weight: float,
    default_samples: int,
    seed: int,
    verbose: bool = False,
    require_planar: bool = True,
) -> Tuple[dict[str, Any], Optional[Data]]:
    batches = list(batches)
    if not batches:
        raise ValueError("No generator batches provided")

    per_generator_best: Dict[str, Dict[str, Any]] = {}
    per_generator_variants: Dict[str, List[Dict[str, Any]]] = {}
    per_generator_data: Dict[str, Optional[Data]] = {}  # Track best Data for each generator
    evaluated_generators: List[str] = []
    best_overall: Optional[Dict[str, Any]] = None
    best_data: Optional[Data] = None
    seed_cursor = seed

    for idx, batch in enumerate(batches):
        name = batch.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"Batch #{idx} is missing a valid 'name'")

        batch_samples = batch.get("samples")
        samples = int(batch_samples) if batch_samples is not None else int(default_samples)
        if samples <= 0:
            samples = int(default_samples)

        print(f"Running {name} (samples={samples})")

        overrides = batch.get("config") or {}
        if not isinstance(overrides, dict):
            raise ValueError(f"Batch '{name}' overrides must be a mapping")

        override_combos = _expand_override_grid(overrides)

        for combo_idx, combo in enumerate(override_combos):
            cfg = replace(base_config)
            for key, value in combo.items():
                if not hasattr(cfg, key):
                    raise ValueError(
                        f"Unknown GraphGeneratorConfig attribute '{key}' in batch '{name}'"
                    )
                if value is None:
                    continue
                setattr(cfg, key, value)

            variant_seed = seed_cursor
            seed_cursor += max(1, samples)

            try:
                batch_best, batch_data = find_best_from_generators(
                    cfg,
                    target_metrics=target_metrics,
                    metric_reward_weight=metric_reward_weight,
                    num_samples=samples,
                    seed=variant_seed,
                    include_generators=[name],
                    verbose=verbose,
                )
            except RuntimeError as exc:
                if verbose:
                    print(f"  ! {name} failed: {exc}")
                entry = {
                    "generator_name": name,
                    "status": "error",
                    "error": str(exc),
                    "reward": None,
                    "seed": None,
                    "sample_index": None,
                    "attempted_samples": 0,
                    "planar_samples": 0,
                    "accepted_samples": 0,
                    "parameters": _generator_parameter_summary(cfg, name),
                    "candidate_metrics": {},
                    "metric_statistics": {},
                    "differences": {},
                    "constraint_report": {},
                    "best_valid_reward": None,
                    "best_valid_seed": None,
                    "best_valid_sample_index": None,
                    "best_valid_candidate_metrics": {},
                    "best_valid_differences": {},
                    "best_valid_constraint_report": {},
                    "planarity": None,
                    "planarity_score": None,
                    "samples": samples,
                    "config_overrides": {k: combo[k] for k in combo if combo[k] is not None},
                }
                per_generator_variants.setdefault(name, []).append(entry)
                evaluated_generators.append(name)
                if name not in per_generator_best:
                    per_generator_best[name] = entry
                continue

            entry = batch_best.get("per_generator_best", {}).get(name, {}).copy()
            entry["generator_name"] = name
            entry["config_overrides"] = {k: combo[k] for k in combo if combo[k] is not None}
            entry["samples"] = samples

            per_generator_variants.setdefault(name, []).append(entry)
            evaluated_generators.append(name)

            if name not in per_generator_best or _entry_rank(entry) > _entry_rank(per_generator_best[name]):
                per_generator_best[name] = entry
                per_generator_data[name] = batch_data  # Store the best data for this generator

            if best_overall is None or _entry_rank(entry) > _entry_rank(best_overall):
                best_overall = entry
                best_data = batch_data

    if best_overall is None:
        raise RuntimeError("Generator batches did not yield any valid samples")

    summary: dict[str, Any] = {
        "source": "generator",
        "generator_name": best_overall.get("generator_name"),
        "sample_index": best_overall.get("sample_index"),
        "seed": best_overall.get("seed"),
        "reward": float(best_overall.get("reward", -float("inf"))),
        "differences": best_overall.get("differences", {}),
        "candidate_metrics": best_overall.get("candidate_metrics", {}),
        "constraint_report": best_overall.get("constraint_report", {}),
        "planarity": best_overall.get("planarity"),
        "planarity_score": best_overall.get("planarity_score"),
        "samples": best_overall.get("samples"),
        "evaluated_generators": sorted(set(evaluated_generators)),
        "per_generator_best": per_generator_best,
        "per_generator_variants": per_generator_variants,
        "per_generator_data": per_generator_data,
        "config_overrides": best_overall.get("config_overrides", {}),
        "best_valid_reward": best_overall.get("best_valid_reward"),
        "best_valid_seed": best_overall.get("best_valid_seed"),
        "best_valid_sample_index": best_overall.get("best_valid_sample_index"),
        "best_valid_candidate_metrics": best_overall.get("best_valid_candidate_metrics", {}),
        "best_valid_differences": best_overall.get("best_valid_differences", {}),
        "best_valid_constraint_report": best_overall.get("best_valid_constraint_report", {}),
    }

    return summary, best_data


def find_similar_graph_in_dataset(
    target: Union[Dict[str, float], Data],
    metric_subset: List[str],
    pt_pattern: str = "ext/hog_planar/*.pt",
    return_all: bool = False,
) -> Union[
    Tuple[str, int, float, Dict[str, float], Data],
    Tuple[Tuple[str, int, float, Dict[str, float], Data], List[Tuple[str, int, float, Dict[str, float], Data]]],
]:
    """
    Find the most similar graph from .pt dataset files.

    This function loads graphs from pre-converted .pt files,
    computes their metrics, and finds the one most similar to the target.

    Args:
        target: Target metrics (dict) or graph (PyG Data object) to match
        metric_subset: List of metrics to consider
        pt_pattern: Glob pattern for .pt files (default: "ext/hog_planar/*.pt")

    Returns:
        If return_all is False (default):
            (dataset_name, graph_index, similarity_score, differences, graph_data)
        If return_all is True:
            (best_result, all_results)
            where best_result is the 5-tuple above and all_results is a list of 5-tuples
            for every evaluated graph.
    """
    # Handle both dict and Data inputs
    if isinstance(target, Data):
        _, target_metrics, _ = prepare_graph_for_benchmark_evaluation(target)
    else:
        target_metrics = target

    # Extract target node count if provided in metrics
    target_num_nodes = target_metrics.get("num_nodes") if isinstance(target_metrics, dict) else None
    # Find all .pt files
    pt_files = sorted(Path().glob(pt_pattern))

    if not pt_files:
        raise FileNotFoundError(f"No .pt files found matching pattern: {pt_pattern}")

    print(f"Loading graphs from {len(pt_files)} .pt file(s)...")

    all_results = []
    fallback_results = []
    total_node_window_candidates = 0

    for pt_file in pt_files:
        dataset_name = pt_file.stem
        print(f"  Processing {dataset_name}...")

        # Load dataset from .pt file
        graphs = load_dataset(str(pt_file))

        # Process each graph
        node_window_candidates = 0
        for graph_idx, graph_data in enumerate(graphs):
            if target_num_nodes is not None:
                # Only consider graphs with N-3 <= num_nodes <= N+3 (i.e. N-3, N-2, N-1, N, N+1, N+2, N+3).
                if not (target_num_nodes - 3 <= graph_data.num_nodes <= target_num_nodes + 3):
                    continue
                node_window_candidates += 1
            try:
                prepared_graph, graph_metrics, constraint_report = prepare_graph_for_benchmark_evaluation(
                    graph_data,
                    seed=42 + graph_idx,
                    strict=False,
                )
            except (RuntimeError, ValueError):
                continue

            # Compute similarity and store in x[2]
            similarity, differences = compute_metric_similarity(
                target_metrics,
                graph_metrics,
                metric_subset=metric_subset,
            )

            planarity_score = ensure_planarity_metric(prepared_graph, graph_metrics)
            if planarity_score < 0.5:
                continue
            graph_metrics["is_planar"] = float(planarity_score)
            if np.isfinite(similarity):
                result = (dataset_name, graph_idx, similarity, differences, prepared_graph)
                fallback_results.append((result, constraint_report))
                if constraint_report["constraints_ok"]:
                    all_results.append(result)

        total_node_window_candidates += node_window_candidates
        if target_num_nodes is not None:
            print(
                f"    Node-window candidates (N-3 <= num_nodes <= N+3, N={int(target_num_nodes)}): {node_window_candidates}"
            )

    # Sort by similarity score (descending - higher is better)
    all_results.sort(key=lambda x: x[2], reverse=True)

    if not all_results:
        if not fallback_results:
            raise ValueError("No valid graphs found in datasets")
        fallback_results.sort(key=lambda item: (_constraint_rank(item[1]), item[0][2]), reverse=True)
        best_result, best_report = fallback_results[0]
        print(
            "No graph satisfied all constraints; using fallback dataset match "
            f"({best_report['num_constraints_satisfied']}/4 constraints satisfied)"
        )
        if return_all:
            return best_result, [item[0] for item in fallback_results]
        return best_result

    if target_num_nodes is not None:
        print(f"Total node-window candidates across datasets: {total_node_window_candidates}")

    print(f"Found {len(all_results)} valid graphs, returning best match")

    best_result = all_results[0]

    if return_all:
        return best_result, all_results

    return best_result


def print_similarity_results(
    result: Tuple[str, int, float, Dict[str, float], Data],
    target_metrics: Optional[Dict[str, float]] = None,
    show_details: bool = True,
) -> None:
    """
    Pretty print similarity search result from dataset search.

    Args:
        result: Result from find_similar_graph_in_dataset (5-tuple with Data object)
        target_metrics: Optional target metrics for comparison
        show_details: If True, show individual metric differences
    """
    print("\n" + "=" * 80)
    print("Graph Similarity Search Result")
    print("=" * 80)

    if target_metrics:
        print("\nTarget Metrics:")
        for metric_name, value in sorted(target_metrics.items()):
            print(f"  {metric_name:30s}: {value:10.6f}")
        print()

    dataset_name, graph_idx, score, differences, graph_data = result

    print(f"\nBest Match: {dataset_name} (graph {graph_idx})")
    print(f"  Similarity Score: {score:.6f} (lower = more similar)")
    print(f"  Graph: {graph_data.num_nodes} nodes, {graph_data.num_edges} edges")

    if show_details and differences:
        print(f"\n  Individual Metric Differences:")
        for metric_name in sorted(differences.keys()):
            diff = differences[metric_name]
            print(f"    {metric_name:28s}: {diff:.6f}")

    print("\n" + "=" * 80)


def evaluate_baseline_from_dataset(
    target_metrics: Dict[str, float],
    job_dir: Path,
    pt_pattern: str = "ext/hog_planar/*.pt",
) -> Tuple[str, int, float, Dict[str, float], Data]:
    """
    Find the best baseline graph from dataset for given target metrics.

    This is used before training to establish a baseline comparison.

    Args:
        target_metrics: Dictionary of target metric values
        pt_pattern: Glob pattern for .pt files

    Returns:
        Tuple: (dataset_name, graph_index, similarity_score, differences, graph_data)
    """
    print("\n" + "=" * 80)
    print("Evaluating Baseline from Dataset")
    print("=" * 80)
    print(f"Target metrics: {', '.join(target_metrics.keys())}\n")

    result = find_similar_graph_in_dataset(
        target=target_metrics,
        metric_subset=list(target_metrics.keys()),
        pt_pattern=pt_pattern,
    )

    dataset_name, graph_idx, score, differences, baseline_graph = result

    # Convert edge_index to adjacency matrix for visualization
    edge_index = baseline_graph.edge_index
    num_nodes = baseline_graph.num_nodes
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0

    # Save to outputs directory
    baseline_dir = job_dir / "baseline_graph"
    baseline_dir.mkdir(parents=True, exist_ok=True)
    save_path = baseline_dir / "baseline_graph.png"

    # Plot and save baseline graph (with fancy PNG+PDF saving)
    fig, ax = plot_graph(
        adj.numpy(), 
        baseline_graph.pos.numpy(),
        pretty=True,
        save_path=save_path,
        save_dpi=300,
        save_pdf=True
    )
    ax.set_title(f"Baseline Graph: {dataset_name} (graph {graph_idx})\nSimilarity: {score:.6f}")
    fig.savefig(save_path, dpi=300, bbox_inches="tight")
    pdf_path = save_path.with_suffix(".pdf")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"\n✓ Best baseline: {dataset_name} (graph {graph_idx})")
    print(f"  Similarity: {score:.6f} (1.0 = perfect match)")
    print(f"  Baseline graph saved to:")
    print(f"    PNG: {save_path}")
    print(f"    PDF: {pdf_path}")
    print(f"    PNG (no axes): {save_path.with_name(f'{save_path.stem}_no_axes.png')}")
    print(f"    PDF (no axes): {save_path.with_name(f'{save_path.stem}_no_axes.pdf')}\n")
    print("=" * 80)

    return result


# =============================================================================
# Original Functions
# =============================================================================


def evaluate_all_generators_with_metrics(
    config: Optional[GraphGeneratorConfig] = None,
    num_samples: int = 100,
    include_graphs: Optional[list[str]] = None,
    *,
    target_metrics: Optional[Dict[str, float]] = None,
    metric_reward_weight: float = 1.0,
    search_seed: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, list]], Dict[str, dict]]:
    """
    Generate multiple samples from each generator and compute metrics.

    Args:
        config: Graph generator configuration
        num_samples: Number of samples per generator
        include_graphs: List of generators to include (None = all)
        target_metrics: Optional target metrics for best-sample search
        metric_reward_weight: Weight for metric-based reward
        search_seed: Seed for searching best samples

    Returns:
        Dictionary mapping generator names to metric dictionaries
    """
    if config is None:
        config = GraphGeneratorConfig()

    if include_graphs is None:
        all_generators = get_generators_dict(config, seed=42)
        include_graphs = list(all_generators.keys())

    results: Dict[str, Dict[str, list]] = {}
    best_matches: Dict[str, dict] = {}

    for gen_name in include_graphs:
        print(f"Evaluating {gen_name}... ({num_samples} samples)")

        # Store metrics for each sample
        gen_metrics = {}

        for i in range(num_samples):
            seed = config.seed + i if config.seed is not None else None

            try:
                # Get generator with this specific seed
                generator = get_generators_dict(config, seed=seed)[gen_name]

                # Generate graph
                data = generator()

                # Compute sector angles for angular resolution metric
                data.sector_angles = compute_sector_angles(data.edge_index, data.pos, data.num_nodes)

                # Compute all metrics using the metrics module
                metrics_dict = compute_metrics_from_data(data)
                sample_metrics = metrics_dict.get("graph_metrics", {})

                # Store each metric
                for metric_name, value in sample_metrics.items():
                    if metric_name not in gen_metrics:
                        gen_metrics[metric_name] = []
                    gen_metrics[metric_name].append(value)

            except Exception as e:
                if gen_name != "planar_er":  # It will often fail for high p
                    warnings.warn(f"Failed to generate/evaluate {gen_name} sample {i}: {e}")
                continue

        results[gen_name] = gen_metrics

        if target_metrics is not None:
            base_config = replace(config)
            base_seed = search_seed if search_seed is not None else (config.seed if config.seed is not None else 42)

            # Adjust domain preferences to mirror evaluation script heuristics
            if "rectangularity" in target_metrics:
                base_config.grid_circular = False
                base_config.delaunay_boundary = "square"
            elif "isoperimetric_ratio" in target_metrics:
                base_config.grid_circular = True
                base_config.delaunay_boundary = "circle"

            best_info, _ = find_best_from_generators(
                base_config,
                target_metrics=target_metrics,
                metric_reward_weight=metric_reward_weight,
                num_samples=num_samples,
                seed=base_seed,
                include_generators=[gen_name],
                verbose=False,
            )
            best_matches[gen_name] = best_info
            print(
                "    best reward vs target="
                f"{float(best_info.get('reward', float('nan'))):.6f}"
            )

    return results, best_matches


# =============================================================================
# Comprehensive Benchmark Evaluation
# =============================================================================


def run_comprehensive_benchmark(
    output_dir: str = "outputs/benchmark_comparison",
    num_nodes: int = 60,
    num_samples: int = 100,
    seed: int = 42,
    erdos_renyi_probs: Optional[list[float]] = None,
    barabasi_attachments: Optional[list[int]] = None,
    ergm_edge_params: Optional[list[float]] = None,
    ergm_triangle_params: Optional[list[float]] = None,
    target_metrics: Optional[Dict[str, float]] = None,
    metric_reward_weight: float = 1.0,
) -> Tuple[dict[str, dict[str, list]], dict[str, dict]]:
    """
    Run comprehensive benchmark evaluation with parameter variations.

    This function:
    1. Generates multiple samples from each graph generator with varying parameters
    2. Computes all metrics for each sample
    3. Saves results to JSON
    4. Creates visualizations (box plots and heatmaps)

    Args:
        output_dir: Directory to save results and visualizations
        num_nodes: Number of nodes for all graphs
        num_samples: Number of samples per configuration (default: 100, use 10 for Boltzmann)
        seed: Random seed for reproducibility
        erdos_renyi_probs: Edge probabilities to test for ER graphs (default: [0.05, 0.10, 0.15, 0.20])
        barabasi_attachments: Attachment values to test for BA graphs (default: [2, 3, 4, 5])
        ergm_edge_params: Edge parameters for ERGM (default: [-3.0, -2.0, -1.5])
        ergm_triangle_params: Triangle parameters for ERGM (default: [0.0, 0.1, 0.2])
        target_metrics: Optional metric targets to evaluate best generator samples against
        metric_reward_weight: Weight applied to metric-based reward when ranking generator samples

    Returns:
        Tuple of (metric distributions, best-match summaries)
    """

    # Set defaults
    if erdos_renyi_probs is None:
        erdos_renyi_probs = [0.05, 0.10, 0.15, 0.20]
    if barabasi_attachments is None:
        barabasi_attachments = [2, 3, 4, 5]
    if ergm_edge_params is None:
        ergm_edge_params = [-3.0, -2.0, -1.5]
    if ergm_triangle_params is None:
        ergm_triangle_params = [0.0, 0.1, 0.2]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Comprehensive Benchmark Evaluation")
    print("=" * 80)
    print(f"Output directory: {output_path}")
    print(f"Samples per configuration: {num_samples}")
    print()

    base_config = {
        "num_nodes": num_nodes,
        "seed": seed,
        "ensure_connected": True,
        "layout_method": "spring",
        "layout_scale": 10.0,
    }

    all_results: dict[str, dict[str, list]] = {}
    all_best_matches: dict[str, dict] = {}

    # 1. Erdős-Rényi with varying edge probabilities
    print("Evaluating Erdős-Rényi G(n,p)...")
    for p in erdos_renyi_probs:
        config = GraphGeneratorConfig(**base_config, edge_probability=p)
        print(f"  - p={p:.2f}...")
        results, best = evaluate_all_generators_with_metrics(
            config=config, num_samples=num_samples, include_graphs=["erdos_renyi_gnp"],
            target_metrics=target_metrics,
            metric_reward_weight=metric_reward_weight,
            search_seed=seed,
        )
        all_results[f"erdos_renyi_p{p:.2f}"] = results["erdos_renyi_gnp"]
        if "erdos_renyi_gnp" in best:
            all_best_matches[f"erdos_renyi_p{p:.2f}"] = best["erdos_renyi_gnp"]

    # 2. Planar Erdős-Rényi
    print("\nEvaluating Planar Erdős-Rényi...")
    config = GraphGeneratorConfig(**base_config, edge_probability=0.10)
    results, best = evaluate_all_generators_with_metrics(
        config=config,
        num_samples=num_samples,
        include_graphs=["planar_er"],
        target_metrics=target_metrics,
        metric_reward_weight=metric_reward_weight,
        search_seed=seed,
    )
    all_results.update(results)
    if "planar_er" in best:
        all_best_matches["planar_er"] = best["planar_er"]

    # 3. Boltzmann Planar (fewer samples)
    print("\nEvaluating Boltzmann Planar (5 samples)...")
    config = GraphGeneratorConfig(**base_config)
    results, best = evaluate_all_generators_with_metrics(
        config=config,
        num_samples=5,
        include_graphs=["boltzmann_planar"],
        target_metrics=target_metrics,
        metric_reward_weight=metric_reward_weight,
        search_seed=seed,
    )
    all_results.update(results)
    if "boltzmann_planar" in best:
        all_best_matches["boltzmann_planar"] = best["boltzmann_planar"]

    # 4. Barabási-Albert with varying attachments
    print("\nEvaluating Barabási-Albert...")
    for m in barabasi_attachments:
        config = GraphGeneratorConfig(**base_config, num_attachments=m)
        print(f"  - m={m}...")
        results, best = evaluate_all_generators_with_metrics(
            config=config, num_samples=num_samples, include_graphs=["barabasi_albert"],
            target_metrics=target_metrics,
            metric_reward_weight=metric_reward_weight,
            search_seed=seed,
        )
        all_results[f"barabasi_albert_m{m}"] = results["barabasi_albert"]
        if "barabasi_albert" in best:
            all_best_matches[f"barabasi_albert_m{m}"] = best["barabasi_albert"]

    # 5. ERGM parameter grid
    print("\nEvaluating ERGM...")
    for edge_p in ergm_edge_params:
        for tri_p in ergm_triangle_params:
            config = GraphGeneratorConfig(
                **base_config,
                ergm_edge_param=edge_p,
                ergm_triangle_param=tri_p,
                ergm_star_param=0.0,
            )
            print(f"  - edge={edge_p:.1f}, triangle={tri_p:.1f}...")
            results, best = evaluate_all_generators_with_metrics(
                config=config,
                num_samples=num_samples,
                include_graphs=["ergm"],
                target_metrics=target_metrics,
                metric_reward_weight=metric_reward_weight,
                search_seed=seed,
            )
            all_results[f"ergm_e{edge_p:.1f}_t{tri_p:.1f}"] = results["ergm"]
            if "ergm" in best:
                all_best_matches[f"ergm_e{edge_p:.1f}_t{tri_p:.1f}"] = best["ergm"]

    # 6. Delaunay Triangulation
    print("\nEvaluating Delaunay Triangulation...")
    config = GraphGeneratorConfig(**base_config)
    results, best = evaluate_all_generators_with_metrics(
        config=config,
        num_samples=num_samples,
        include_graphs=["delaunay"],
        target_metrics=target_metrics,
        metric_reward_weight=metric_reward_weight,
        search_seed=seed,
    )
    all_results.update(results)
    if "delaunay" in best:
        all_best_matches["delaunay"] = best["delaunay"]

    # 7. Lattice Grid
    print("\nEvaluating Lattice Grid...")
    config = GraphGeneratorConfig(**base_config)
    results, best = evaluate_all_generators_with_metrics(
        config=config,
        num_samples=num_samples,
        include_graphs=["lattice_grid"],
        target_metrics=target_metrics,
        metric_reward_weight=metric_reward_weight,
        search_seed=seed,
    )
    all_results.update(results)
    if "lattice_grid" in best:
        all_best_matches["lattice_grid"] = best["lattice_grid"]

    # 8. Plantri Planar Graphs
    print("\nEvaluating Plantri Planar Graphs...")
    config = GraphGeneratorConfig(**base_config, plantri_connectivity=3, plantri_min_degree=None)
    results, best = evaluate_all_generators_with_metrics(
        config=config,
        num_samples=10,
        include_graphs=["plantri"],
        target_metrics=target_metrics,
        metric_reward_weight=metric_reward_weight,
        search_seed=seed,
    )
    all_results.update(results)
    if "plantri" in best:
        all_best_matches["plantri"] = best["plantri"]

    # Save results
    print("\nSaving results...")
    json_results = {
        gen_name: {metric_name: [float(v) for v in values] for metric_name, values in metrics.items()}
        for gen_name, metrics in all_results.items()
    }

    with open(output_path / "benchmark_metrics.json", "w") as f:
        json.dump(json_results, f, indent=2)
    print(f"  - Saved to {output_path / 'benchmark_metrics.json'}")

    # Create visualizations
    print("\nCreating visualizations...")
    plot_metric_comparison(
        results=all_results,
        figsize=(28, 20),
        save_path=str(output_path / "metrics_boxplot.png"),
    )
    print(f"  - Box plot: {output_path / 'metrics_boxplot.png'}")

    plot_metric_heatmap(
        results=all_results,
        use_mean=True,
        normalize=True,
        figsize=(20, 14),
        save_path=str(output_path / "metrics_heatmap_normalized.png"),
    )
    print(f"  - Heatmap (normalized): {output_path / 'metrics_heatmap_normalized.png'}")

    plot_metric_heatmap(
        results=all_results,
        use_mean=True,
        normalize=False,
        figsize=(20, 14),
        save_path=str(output_path / "metrics_heatmap_raw.png"),
    )
    print(f"  - Heatmap (raw): {output_path / 'metrics_heatmap_raw.png'}")

    # Print summary
    print("\n" + "=" * 80)
    print("Summary Statistics")
    print("=" * 80)
    for gen_name in sorted(all_results.keys()):
        print(f"\n{gen_name}:")
        metrics = all_results[gen_name]
        for metric_name in sorted(metrics.keys()):
            values = metrics[metric_name]
            if values:
                print(
                    f"  {metric_name:30s}: μ={np.mean(values):7.4f} σ={np.std(values):7.4f} "
                    f"[{np.min(values):7.4f}, {np.max(values):7.4f}]"
                )

    print("\n" + "=" * 80)
    if all_best_matches:
        best_matches_path = output_path / "benchmark_best_matches.json"
        with open(best_matches_path, "w") as f:
            json.dump(all_best_matches, f, indent=2)
        print("\nSaved best generator matches:")
        print(f"  - {best_matches_path}")
        print("\nBest-match rewards (higher is better):")
        for name, info in sorted(all_best_matches.items()):
            reward = info.get("reward", float("nan"))
            print(
                f"  {name:30s}: reward={float(reward):.6f} generator={info.get('generator_name')}"
            )

    print("\n" + "=" * 80)
    print(f"Benchmark complete! Results in: {output_path}")
    print("=" * 80)

    return all_results, all_best_matches


# =============================================================================
# Main Function for testing
# =============================================================================


def _generate_benchmark_graphs(output_path: Path) -> dict[str, dict[str, list]]:
    """Generate sample graphs and run comprehensive benchmark."""
    print("=" * 80)
    print("Generating Benchmark Graphs")
    print("=" * 80)
    print()

    # Generate one sample from each generator for visualization
    print("Generating sample graphs for visualization...")
    config = GraphGeneratorConfig(
        num_nodes=60,
        edge_probability=0.15,
        num_attachments=2,
        seed=42,
        ensure_connected=True,
        layout_method="spring",
        layout_scale=10.0,
    )

    all_generators = get_generators_dict(config, seed=42)
    sample_graphs = {}

    for name, generator in all_generators.items():
        try:
            sample_graphs[name] = generator()
            print(f"  Generated: {name}")
        except Exception as e:
            print(f"  Failed to generate {name}: {e}")

    # Plot sample graphs
    print(f"\nPlotting {len(sample_graphs)} sample graphs...")
    plot_benchmark_graphs(
        sample_graphs,
        figsize=(24, 16),
        node_size=200,
        node_color="lightblue",
        edge_color="gray",
        font_size=9,
        save_path=str(output_path / "benchmark_sample_graphs.png"),
    )

    print()
    print("=" * 80)
    print()

    # Run comprehensive benchmark evaluation
    metrics, _ = run_comprehensive_benchmark(
        output_dir=str(output_path),
        num_nodes=50,
        num_samples=100,
        seed=42,
    )
    return metrics


def _create_combined_visualizations(all_results: dict, output_path: Path) -> None:
    """Create combined visualizations from all results."""
    print("\n" + "=" * 80)
    print("Creating Combined Visualizations")
    print("=" * 80)

    plot_metric_comparison(
        results=all_results,
        figsize=(32, 24),
        save_path=str(output_path / "combined_metrics_boxplot.png"),
    )
    print("  - Combined box plot created")

    plot_metric_heatmap(
        results=all_results,
        use_mean=True,
        normalize=True,
        figsize=(24, 18),
        save_path=str(output_path / "combined_metrics_heatmap_normalized.png"),
    )
    print("  - Combined normalized heatmap created")

    plot_metric_heatmap(
        results=all_results,
        use_mean=True,
        normalize=False,
        figsize=(24, 18),
        save_path=str(output_path / "combined_metrics_heatmap_raw.png"),
    )
    print("  - Combined raw heatmap created")

    print()
    print("=" * 80)
    print(f"All results saved to: {output_path}")
    print("=" * 80)


def main(
    run_benchmark: bool = True,
    load_datasets: bool = True,
    run_demo: bool = True,
    lst_pattern: str = "ext/hog_planar/*.lst",
    output_dir: str = "outputs/benchmark_comparison",
):
    """
    Main function - generates benchmark graphs and/or loads datasets from .lst files.

    Args:
        run_benchmark: If True, run comprehensive benchmark evaluation
        load_datasets: If True, load and evaluate .lst files
        run_demo: If True, run similarity search demo
        lst_pattern: Glob pattern for .lst files
        output_dir: Directory to save results

    Results are saved to output_dir/ including:
    - Visualization of sample graphs from each generator/dataset
    - JSON file with all metric values
    - Box plot comparison across generators/datasets
    - Heatmap visualizations (normalized and raw)
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("Benchmark Graph Evaluation")
    print("=" * 80)
    print(f"Output directory: {output_path}")
    print()

    all_results = {}

    # Load and evaluate datasets from .lst files
    if load_datasets:
        print("=" * 80)
        print("Loading and Evaluating Datasets from .lst Files")
        print("=" * 80)
        print()

        dataset_metrics = evaluate_lst_datasets(
            lst_pattern=lst_pattern,
            output_dir=output_path,
            layout="spring",
            compute_metrics=True,
            num_samples_to_plot=10,
        )
        all_results.update(dataset_metrics)

    # Generate benchmark graphs
    if run_benchmark:
        benchmark_results = _generate_benchmark_graphs(output_path)
        all_results.update(benchmark_results)

    # Create combined visualizations if we have results from both
    if all_results and (load_datasets and run_benchmark):
        _create_combined_visualizations(all_results, output_path)

    # Run similarity search demo
    if run_demo:
        print("\n" + "=" * 80)
        print("Graph Similarity Search Demo - Dataset")
        print("=" * 80)
        print()

        target_metrics = {
            "spectral_gap": 1,
            "clustering_coefficient": 0.2,
            "gini_coefficient": 0.3,
        }

        dataset_name, graph_idx, score, differences, graph_data = find_similar_graph_in_dataset(
            target=target_metrics,
            metric_subset=list(target_metrics.keys()),
            pt_pattern="ext/hog_planar/*.pt",
        )

        print_similarity_results(
            (dataset_name, graph_idx, score, differences, graph_data), target_metrics, show_details=True
        )

        print("\nGraph data available for further analysis")


if __name__ == "__main__":
    main()

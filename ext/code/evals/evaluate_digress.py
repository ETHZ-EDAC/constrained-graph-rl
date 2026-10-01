import json
import os
import pickle
import sys
from pathlib import Path
from typing import Dict

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
DIGRESS_ROOT = PROJECT_ROOT / "ext" / "DiGress" / "DiGress"
DIGRESS_SRC_ROOT = DIGRESS_ROOT / "src"

for path in (SRC_ROOT, DIGRESS_ROOT, DIGRESS_SRC_ROOT):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

from graph_rl.utils.benchmark_graphs import (  # noqa: E402
    ensure_planarity_metric,
    metric_reward,
    prepare_graph_for_benchmark_evaluation,
)
from graph_rl.utils.graph_helpers import networkx_to_pyg_data  # noqa: E402
from src.datasets.hog_with_metrics import extract_adjacency_from_edge_types  # noqa: E402


def _get_plot_graph():
    try:
        from prodigy import plot_graph
        return plot_graph
    except ImportError:
        parent_dir = os.path.join(os.path.dirname(__file__), "..")
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from prodigy import plot_graph
        return plot_graph


def _largest_connected_component(graph):
    if graph is None or graph.number_of_nodes() == 0:
        return graph
    if graph.number_of_nodes() == 1:
        return graph.copy()
    component_nodes = max(
        nx.connected_components(graph),
        key=lambda nodes: (len(nodes), -min(nodes)),
    )
    return nx.convert_node_labels_to_integers(graph.subgraph(component_nodes).copy())


def _safe_mean(values):
    if not values:
        return 0.0
    return float(np.mean(values))


def _format_candidate_metrics(metrics):
    return ", ".join(
        f"{metric}={float(value):.6f}"
        for metric, value in sorted(metrics.items())
        if isinstance(value, (int, float, np.floating, np.integer))
    )


def _format_objective_metrics(metrics, target_metrics):
    ordered_metrics = []
    if "num_nodes" in metrics:
        ordered_metrics.append("num_nodes")
    for metric_name in target_metrics.keys():
        if metric_name != "num_nodes" and metric_name in metrics:
            ordered_metrics.append(metric_name)
    return ", ".join(f"{metric}={float(metrics[metric]):.6f}" for metric in ordered_metrics)


def _constraint_summary(report):
    if not report:
        return "n/a"
    return (
        f"{int(report['num_constraints_satisfied'])}/4 "
        f"(angle={report['passes_minimum_angle']}, "
        f"edge_length={report['passes_edge_length']}, "
        f"degree={report['passes_max_degree']}, "
        f"intersection={report['passes_edge_intersection']})"
    )


def _write_benchmark_summary(out_path, *, target_metrics, best_graph, best_planar_graph):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    lines = [
        "Run target metrics:",
        *[f"  {metric_name}: {float(metric_value):.6f}" for metric_name, metric_value in target_metrics.items()],
        "",
        "Best graph (ranked by planarity, node-count window, reward):",
    ]

    if best_graph is None:
        lines.append("  none")
    else:
        lines.extend(
            [
                f"  index: {int(best_graph['index'])}",
                f"  reward: {float(best_graph['reward']):.6f}",
                f"  objectives: {_format_objective_metrics(best_graph['metrics'], target_metrics)}",
                f"  constraints: {_constraint_summary(best_graph['constraint_report'])}",
                f"  metrics: {_format_candidate_metrics(best_graph['metrics'])}",
                f"  plot: {best_graph['plot_path']}",
                f"  stats: {best_graph['stats_path']}",
            ]
        )

    lines.extend(["", "Best planar graph (ranked by node-count window, reward):"])
    if best_planar_graph is None:
        lines.append("  none")
    else:
        lines.extend(
            [
                f"  index: {int(best_planar_graph['index'])}",
                f"  reward: {float(best_planar_graph['reward']):.6f}",
                f"  objectives: {_format_objective_metrics(best_planar_graph['metrics'], target_metrics)}",
                f"  constraints: {_constraint_summary(best_planar_graph['constraint_report'])}",
                f"  metrics: {_format_candidate_metrics(best_planar_graph['metrics'])}",
                f"  plot: {best_planar_graph['plot_path']}",
                f"  stats: {best_planar_graph['stats_path']}",
            ]
        )

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")


def _node_count_rank(num_nodes, target_num_nodes, *, window_size=10.0):
    if target_num_nodes is None:
        return (0, 0.0)
    distance = abs(float(num_nodes) - float(target_num_nodes))
    if distance <= float(window_size):
        bucket = 0
    else:
        bucket = 1 + int((distance - float(window_size)) // float(window_size))
    return (bucket, distance)


def _candidate_rank_tuple(metrics, reward, target_num_nodes):
    node_bucket, node_distance = _node_count_rank(
        metrics.get("num_nodes", 0.0),
        target_num_nodes,
    )
    is_planar = 1 if float(metrics.get("is_planar", 0.0)) >= 0.5 else 0
    safe_reward = float(reward) if reward is not None else -float("inf")
    return (
        is_planar,
        -node_bucket,
        -node_distance,
        safe_reward,
    )


def _sample_to_graph(sample):
    _, edge_types = sample
    adjacency = extract_adjacency_from_edge_types(edge_types).detach().cpu().numpy()
    graph = nx.from_numpy_array(adjacency)
    return _largest_connected_component(graph)


def _compute_graphrl_candidate(graph, *, seed):
    graph = _largest_connected_component(graph)
    data = networkx_to_pyg_data(graph)
    prepared, candidate_metrics, constraint_report = prepare_graph_for_benchmark_evaluation(
        data,
        seed=seed,
        strict=False,
    )
    planarity_score = ensure_planarity_metric(prepared, candidate_metrics)
    metrics = {
        str(key): float(value)
        for key, value in candidate_metrics.items()
        if isinstance(value, (int, float, np.floating, np.integer))
    }
    if "triangles" in metrics and "triangle_count" not in metrics:
        metrics["triangle_count"] = float(metrics["triangles"])
    if "triangle_count" in metrics and "triangles" not in metrics:
        metrics["triangles"] = float(metrics["triangle_count"])
    metrics["is_planar"] = float(planarity_score)
    return graph, prepared, metrics, constraint_report


def _select_best_graph(samples, targets, require_planar=False, metric_reward_weight=1.0):
    if len(samples) == 0:
        return None, None, None, None, None, None, None

    best_idx = None
    best_graph = None
    best_metrics = None
    best_breakdown = {}
    best_reward = -float("inf")
    best_data = None
    best_constraint_report = None
    best_rank = None
    target_num_nodes = targets.get("num_nodes", None) if hasattr(targets, "get") else None

    for idx, sample in enumerate(samples):
        try:
            graph = _sample_to_graph(sample)
            if graph is None or graph.number_of_nodes() == 0:
                continue
            component_graph, prepared_data, metrics, constraint_report = _compute_graphrl_candidate(
                graph,
                seed=42 + idx,
            )
        except Exception:
            continue
        if require_planar and metrics.get("is_planar", 0.0) < 0.5:
            continue
        reward_targets = {
            str(k): float(v)
            for k, v in targets.items()
            if v is not None and str(k) != "num_nodes"
        }
        reward, breakdown = metric_reward(
            target_metrics=reward_targets,
            candidate_metrics=metrics,
            metric_reward_weight=float(metric_reward_weight),
        )
        rank = _candidate_rank_tuple(metrics, reward, target_num_nodes)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_reward = float(reward)
            best_idx = idx
            best_graph = component_graph
            best_metrics = metrics
            best_breakdown = breakdown
            best_data = prepared_data
            best_constraint_report = constraint_report

    if best_idx is None:
        return None, None, None, None, None, None, None
    return int(best_idx), best_graph, best_metrics, float(best_reward), best_breakdown, best_data, best_constraint_report


def _save_best_graph_artifacts(
    graph,
    metrics,
    targets,
    score,
    score_breakdown,
    out_dir,
    stem,
    *,
    tag="best_graph",
    prepared_data=None,
    constraint_report=None,
):
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, f"{stem}_{tag}.png")
    json_path = os.path.join(out_dir, f"{stem}_{tag}_stats.json")

    if prepared_data is not None and getattr(prepared_data, "pos", None) is not None:
        num_nodes = int(prepared_data.num_nodes)
        adj = np.zeros((num_nodes, num_nodes), dtype=np.float32)
        edge_index = prepared_data.edge_index.detach().cpu().numpy()
        adj[edge_index[0], edge_index[1]] = 1.0
        pos = prepared_data.pos.detach().cpu().numpy()
    else:
        adj = nx.adjacency_matrix(graph).toarray().astype(np.float32)
        pos_dict = nx.spring_layout(graph, seed=42, k=0.5, iterations=50)
        num_nodes = adj.shape[0]
        pos = np.asarray([pos_dict.get(i, np.array([0.0, 0.0])) for i in range(num_nodes)])

    pos_3d = np.zeros((pos.shape[0], 3), dtype=np.float32)
    pos_3d[:, :2] = pos[:, :2]

    plot_graph = _get_plot_graph()
    plot_graph(
        adj,
        pos_3d,
        pretty=True,
        save_path=Path(img_path),
        save_dpi=200,
        save_pdf=True,
        show_axes=False,
    )
    plt.close("all")

    payload = {
        "targets": {k: float(v) for k, v in targets.items()},
        "score": float(score),
        "score_breakdown": {k: float(v) for k, v in score_breakdown.items()},
        "metrics": {k: float(v) for k, v in metrics.items()},
    }
    if constraint_report is not None:
        payload["constraint_report"] = constraint_report
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return img_path, json_path


def evaluate_samples(samples, constr_config, *, output_dir, run_name, target_metrics=None):
    result_dict = {
        "num_samples": int(len(samples)),
    }
    planar_flags = []
    connected_node_counts = []
    for sample in samples:
        try:
            graph = _sample_to_graph(sample)
        except Exception:
            continue
        if graph is None or graph.number_of_nodes() == 0:
            continue
        connected_node_counts.append(float(graph.number_of_nodes()))
        planar_flags.append(1.0 if nx.check_planarity(graph)[0] else 0.0)
    result_dict["largest_component_num_nodes_mean"] = _safe_mean(connected_node_counts)
    result_dict["planar_acc_lcc"] = _safe_mean(planar_flags)

    benchmark_targets = target_metrics
    if benchmark_targets is None and hasattr(constr_config, "get"):
        benchmark_targets = constr_config.get("target_metrics", None)
    if benchmark_targets is None:
        raise ValueError("No target_metrics provided for DiGress evaluation.")
    benchmark_targets = {str(k): float(v) for k, v in benchmark_targets.items()}

    best_graph_summary = None
    best_idx, best_graph_component, best_metrics, best_score, best_breakdown, best_data, best_constraint_report = _select_best_graph(
        samples,
        benchmark_targets,
    )
    if best_idx is not None:
        best_img_path, best_json_path = _save_best_graph_artifacts(
            best_graph_component,
            best_metrics,
            benchmark_targets,
            best_score,
            best_breakdown,
            output_dir,
            run_name,
            tag="best_graph",
            prepared_data=best_data,
            constraint_report=best_constraint_report,
        )
        result_dict["best_graph_index"] = int(best_idx)
        result_dict["best_graph_score"] = float(best_score if best_score is not None else 0.0)
        result_dict["best_graph_plot_path"] = best_img_path
        result_dict["best_graph_stats_path"] = best_json_path
        for key, val in (best_metrics or {}).items():
            result_dict[f"best_{key}"] = float(val)
        best_graph_summary = {
            "index": int(best_idx),
            "reward": float(best_score if best_score is not None else 0.0),
            "metrics": best_metrics or {},
            "constraint_report": best_constraint_report or {},
            "plot_path": best_img_path,
            "stats_path": best_json_path,
        }

    best_planar_graph_summary = None
    best_planar_idx, best_planar_graph_component, best_planar_metrics, best_planar_score, best_planar_breakdown, best_planar_data, best_planar_constraint_report = _select_best_graph(
        samples,
        benchmark_targets,
        require_planar=True,
    )
    if best_planar_idx is not None:
        planar_img_path, planar_json_path = _save_best_graph_artifacts(
            best_planar_graph_component,
            best_planar_metrics,
            benchmark_targets,
            best_planar_score,
            best_planar_breakdown,
            output_dir,
            run_name,
            tag="best_planar_graph",
            prepared_data=best_planar_data,
            constraint_report=best_planar_constraint_report,
        )
        result_dict["best_planar_graph_index"] = int(best_planar_idx)
        result_dict["best_planar_graph_score"] = float(best_planar_score if best_planar_score is not None else 0.0)
        result_dict["best_planar_graph_plot_path"] = planar_img_path
        result_dict["best_planar_graph_stats_path"] = planar_json_path
        for key, val in (best_planar_metrics or {}).items():
            result_dict[f"best_planar_{key}"] = float(val)
        best_planar_graph_summary = {
            "index": int(best_planar_idx),
            "reward": float(best_planar_score if best_planar_score is not None else 0.0),
            "metrics": best_planar_metrics or {},
            "constraint_report": best_planar_constraint_report or {},
            "plot_path": planar_img_path,
            "stats_path": planar_json_path,
        }
    else:
        result_dict["best_planar_graph_index"] = None
        result_dict["best_planar_graph_score"] = None
        result_dict["best_planar_graph_plot_path"] = None
        result_dict["best_planar_graph_stats_path"] = None

    summary_path = os.path.join(output_dir, f"{run_name}_benchmark_summary.txt")
    _write_benchmark_summary(
        summary_path,
        target_metrics=benchmark_targets,
        best_graph=best_graph_summary,
        best_planar_graph=best_planar_graph_summary,
    )
    result_dict["benchmark_summary_path"] = summary_path

    results_json_path = os.path.join(output_dir, f"{run_name}_results.json")
    with open(results_json_path, "w", encoding="utf-8") as f:
        json.dump(result_dict, f, indent=2)
    result_dict["results_json_path"] = results_json_path
    return result_dict


def evaluate_sample_file(sample_path, constr_config, *, output_dir=None, run_name=None, target_metrics=None):
    resolved_path = Path(sample_path)
    if not resolved_path.exists() and resolved_path.suffix != ".pkl":
        candidates = [
            resolved_path.with_suffix(".pkl"),
            resolved_path.parent / "samples.pkl",
            resolved_path.parent / "generated_samples.pkl",
        ]
        for candidate in candidates:
            if candidate.exists():
                resolved_path = candidate
                break
    if not resolved_path.exists():
        return -1
    with open(resolved_path, "rb") as f:
        samples = pickle.load(f)
    sample_stem = resolved_path.stem if run_name is None else run_name
    sample_output_dir = str(resolved_path.parent if output_dir is None else output_dir)
    return evaluate_samples(
        samples,
        constr_config,
        output_dir=sample_output_dir,
        run_name=sample_stem,
        target_metrics=target_metrics,
    )


def evaluate_setting(dataset, constr_config, sample_path):
    del dataset
    return evaluate_sample_file(sample_path, constr_config)

#!/usr/bin/env python3
"""
Generate graphs with regressor-guided conditional sampling.

This script loads a trained unconditional DiGress model and a trained property regressor,
then generates graphs conditioned on target metric values.

Usage:
    python scripts/sample_conditional_hog.py \
        --model_checkpoint checkpoints/hog_planar_unconditional/last.ckpt \
        --regressor_checkpoint checkpoints/property_regressor/use_case_1/best_regressor.pt \
        --use_case use_case_1 \
        --num_samples 10 \
        --guidance_scale 2.0
"""

import argparse
import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional
import pickle
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch

# This script loads trusted local checkpoints that include non-tensor metadata.
os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')

import torch
# Handle PyTorch 2.6+ weights_only requirement
try:
    import omegaconf
    torch.serialization.add_safe_globals([
        omegaconf.dictconfig.DictConfig,
        omegaconf.base.ContainerMetadata,
        Any,
    ])
except Exception:
    pass

import sys
script_dir = os.path.dirname(os.path.abspath(__file__))
digress_dir = os.path.join(script_dir, '..')
project_root = os.path.abspath(os.path.join(digress_dir, '..', '..', '..'))
sys.path.insert(0, digress_dir)
sys.path.insert(0, os.path.join(digress_dir, 'src'))
sys.path.insert(0, os.path.join(project_root, 'src'))

from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
from src.models.property_regressor import SimpleGCNRegressor
from src.datasets.hog_with_metrics import extract_adjacency_from_edge_types, compute_graph_metrics_from_adjacency
from metrics.abstract_metrics import TrainAbstractMetricsDiscrete, TrainAbstractMetrics
from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
from graph_rl.utils.benchmark_graphs import ensure_planarity_metric, metric_reward, prepare_graph_for_benchmark_evaluation
from graph_rl.utils.graph_helpers import add_node_positions, networkx_to_pyg_data


def plot_graph(
    adjacency: np.ndarray,
    vertex_positions: np.ndarray,
    ax=None,
    fig=None,
    *,
    pretty: bool = True,
    save_path: Optional[Path] = None,
    save_dpi: Optional[int] = None,
    save_pdf: bool = True,
) -> tuple:
    """
    Plots the graph with vertex positions.

    Parameters
    ----------
    adjacency : np.ndarray (N, N): array representing adjacency matrix, padded with zero-rows/cols.
    vertex_positions : np.ndarray (N, 3): array of [x, y, z] coordinates, padded.
    pretty : bool: If True (default), use a cleaner, colorful, undirected style.
    save_path : Path | None: Optional path to save the figure.
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
    plt.close(fig_no_axes)
    return fig, ax


USE_CASE_METRIC_SUBSETS = {
    'use_case_0': ['num_nodes', 'isoperimetric_ratio', 'triangle_count'],
    'use_case_1': ['num_nodes', 'spectral_gap', 'gini_coefficient', 'clustering_coefficient', 'isoperimetric_ratio'],
    'use_case_2': ['num_nodes', 'angular_resolution', 'edge_length_deviation', 'gini_coefficient', 'isoperimetric_ratio'],
    'use_case_3': ['num_nodes', 'spectral_gap', 'rectangularity'],
    'all_4_use_cases': [
        'num_nodes',
        'spectral_gap',
        'gini_coefficient',
        'clustering_coefficient',
        'triangle_count',
        'isoperimetric_ratio',
        'angular_resolution',
        'edge_length_deviation',
        'rectangularity',
    ],
}

USE_CASE_TARGETS = {
    'use_case_0': {'num_nodes': 60.0, 'isoperimetric_ratio': 1.0, 'triangle_count': 40.0},
    'use_case_1': {'num_nodes': 50.0, 'spectral_gap': 1.0, 'gini_coefficient': 0.1, 'clustering_coefficient': 0.2,
                   'isoperimetric_ratio': 1.0},
    'use_case_2': {'num_nodes': 80.0, 'angular_resolution': 0.8, 'edge_length_deviation': 0.7,
                   'gini_coefficient': 0.1, 'isoperimetric_ratio': 1.0},
    'use_case_3': {'num_nodes': 100.0, 'spectral_gap': 0.3, 'rectangularity': 1.0},
}

USE_CASE_GUIDANCE_WEIGHTS = {
    'use_case_0': {'num_nodes': 4.0},
    'use_case_1': {'num_nodes': 4.0},
    'use_case_2': {'num_nodes': 4.0},
    'use_case_3': {'num_nodes': 4.0},
}


def _infer_regressor_dims(state_dict: Dict[str, torch.Tensor]):
    hidden_dim = state_dict['gcn1.lin.weight'].shape[0]
    num_metrics = state_dict['mlp.3.weight'].shape[0]
    return hidden_dim, num_metrics


def parse_metric_subset(metric_subset_str: str):
    metric_subset = [m.strip() for m in metric_subset_str.split(',') if m.strip()]
    if len(metric_subset) == 0:
        raise ValueError('--metric_subset cannot be empty')
    return metric_subset


def parse_target_metrics(target_metrics_str: str) -> Dict[str, float]:
    target = {}
    if not target_metrics_str:
        return target
    for item in target_metrics_str.split(','):
        kv = item.strip()
        if not kv:
            continue
        if '=' not in kv:
            raise ValueError(f"Invalid target entry '{kv}'. Use key=value format.")
        key, value = kv.split('=', 1)
        target[key.strip()] = float(value.strip())
    return target


def parse_metric_weights(metric_weights_str: str) -> Dict[str, float]:
    weights = {}
    if not metric_weights_str:
        return weights
    for item in metric_weights_str.split(','):
        kv = item.strip()
        if not kv:
            continue
        if '=' not in kv:
            raise ValueError(f"Invalid metric weight entry '{kv}'. Use key=value format.")
        key, value = kv.split('=', 1)
        weights[key.strip()] = float(value.strip())
    return weights


def create_guidance_weight_tensor(
    metric_order: list[str],
    weight_map: Dict[str, float],
) -> torch.Tensor:
    weights = torch.ones(len(metric_order), dtype=torch.float32)
    for i, metric_name in enumerate(metric_order):
        if metric_name in weight_map:
            weights[i] = float(weight_map[metric_name])
    return weights


def load_model_and_regressor(model_checkpoint: str, regressor_checkpoint: str, device: str):
    """Load the trained diffusion model and regressor."""

    def _extract_cfg_from_checkpoint(checkpoint_path: str):
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        if isinstance(checkpoint, dict):
            hyper_parameters = checkpoint.get('hyper_parameters', {})
            if isinstance(hyper_parameters, dict) and 'cfg' in hyper_parameters:
                return hyper_parameters['cfg']
            if 'cfg' in checkpoint:
                return checkpoint['cfg']
        raise RuntimeError(
            f"Could not extract cfg from checkpoint: {checkpoint_path}. "
            "Expected checkpoint['hyper_parameters']['cfg'] or checkpoint['cfg']."
        )

    def _build_model_kwargs(cfg):
        dataset_name = cfg['dataset']['name']

        if dataset_name == 'hog_planar':
            from datasets.hog_planar_dataset import HOGPlanarDataModule, HOGDatasetInfos
            from analysis.visualization import NonMolecularVisualization
            from analysis.spectre_utils import PlanarSamplingMetrics

            datamodule = HOGPlanarDataModule(cfg)
            dataset_infos = HOGDatasetInfos(datamodule, cfg)
            train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
            visualization_tools = NonMolecularVisualization()
            sampling_metrics = PlanarSamplingMetrics(datamodule=datamodule)

            if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
                extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            else:
                extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

            dataset_infos.compute_input_output_dims(
                datamodule=datamodule,
                extra_features=extra_features,
                domain_features=domain_features,
            )

            return {
                'dataset_infos': dataset_infos,
                'train_metrics': train_metrics,
                'sampling_metrics': sampling_metrics,
                'visualization_tools': visualization_tools,
                'extra_features': extra_features,
                'domain_features': domain_features,
            }

        if dataset_name in ['sbm', 'comm20', 'planar']:
            from datasets.spectre_dataset import SpectreGraphDataModule, SpectreDatasetInfos
            from analysis.spectre_utils import PlanarSamplingMetrics, SBMSamplingMetrics, Comm20SamplingMetrics
            from analysis.visualization import NonMolecularVisualization

            datamodule = SpectreGraphDataModule(cfg)
            if dataset_name == 'sbm':
                sampling_metrics = SBMSamplingMetrics(datamodule)
            elif dataset_name == 'comm20':
                sampling_metrics = Comm20SamplingMetrics(datamodule)
            else:
                sampling_metrics = PlanarSamplingMetrics(datamodule)

            dataset_infos = SpectreDatasetInfos(datamodule, cfg['dataset'])
            train_metrics = TrainAbstractMetricsDiscrete() if cfg.model.type == 'discrete' else TrainAbstractMetrics()
            visualization_tools = NonMolecularVisualization()

            if cfg.model.type == 'discrete' and cfg.model.extra_features is not None:
                extra_features = ExtraFeatures(cfg.model.extra_features, dataset_info=dataset_infos)
            else:
                extra_features = DummyExtraFeatures()
            domain_features = DummyExtraFeatures()

            dataset_infos.compute_input_output_dims(
                datamodule=datamodule,
                extra_features=extra_features,
                domain_features=domain_features,
            )

            return {
                'dataset_infos': dataset_infos,
                'train_metrics': train_metrics,
                'sampling_metrics': sampling_metrics,
                'visualization_tools': visualization_tools,
                'extra_features': extra_features,
                'domain_features': domain_features,
            }

        raise NotImplementedError(
            f"Dataset '{dataset_name}' is not supported by this sampling script."
        )
    
    # Load diffusion model
    print(f"Loading diffusion model from {model_checkpoint}...")
    cfg = _extract_cfg_from_checkpoint(model_checkpoint)
    model_kwargs = _build_model_kwargs(cfg)
    model = DiscreteDenoisingDiffusion.load_from_checkpoint(model_checkpoint, **model_kwargs)
    
    model = model.to(device)
    model.eval()
    
    # Load regressor
    print(f"Loading regressor from {regressor_checkpoint}...")
    regressor_state = torch.load(regressor_checkpoint, map_location=device, weights_only=False)
    hidden_dim, num_metrics = _infer_regressor_dims(regressor_state)
    regressor = SimpleGCNRegressor(input_dim=1, hidden_dim=hidden_dim, num_metrics=num_metrics)
    regressor.load_state_dict(regressor_state)
    regressor = regressor.to(device)
    regressor.eval()
    
    return model, regressor


def create_target_metrics_tensor(
    target_dict: Dict[str, float],
    metric_order: Optional[list] = None
) -> torch.Tensor:
    """Create a tensor of target metrics in the correct order."""
    
    if metric_order is None:
        metric_order = [
            'num_nodes',
            'spectral_gap',
            'gini_coefficient',
            'clustering_coefficient',
        ]
    
    target_metrics = torch.zeros(1, len(metric_order))
    for i, metric_name in enumerate(metric_order):
        if metric_name in target_dict:
            target_metrics[0, i] = target_dict[metric_name]
        else:
            print(f"Warning: Target metric '{metric_name}' not specified, using 0")
    
    return target_metrics


def sample_conditional(
    model: DiscreteDenoisingDiffusion,
    regressor: SimpleGCNRegressor,
    target_metrics: torch.Tensor,
    num_samples: int = 5,
    guidance_scale: float = 1.0,
    device: str = 'cpu',
    guidance_metric_weights: Optional[torch.Tensor] = None,
) -> list:
    """Generate graphs with guided conditional sampling."""
    
    print(f"Generating {num_samples} samples with guidance scale {guidance_scale}...")
    
    with torch.no_grad():
        samples = model.sample_batch_with_guidance(
            batch_id=0,
            batch_size=num_samples,
            keep_chain=0,
            number_chain_steps=50,
            save_final=num_samples,
            regressor=regressor,
            target_metrics=target_metrics.to(device),
            guidance_scale=guidance_scale,
            guidance_metric_weights=guidance_metric_weights,
        )
    
    return samples


def compute_metrics_of_samples(samples: list, metric_backend: str = 'spectre', metric_subset: Optional[list] = None):
    """Compute metrics of sampled graphs for evaluation."""    
    print("Computing metrics of samples...")
    
    results = []
    for i, (atom_types, edge_types) in enumerate(samples):
        try:
            num_nodes = len(atom_types)
            adj = extract_adjacency_from_edge_types(edge_types.cpu().float())

            metrics = compute_graph_metrics_from_adjacency(
                adjacency=adj,
                num_nodes=num_nodes,
                metric_backend=metric_backend,
            )

            if metric_subset is not None:
                metrics = {k: v for k, v in metrics.items() if k in metric_subset}

            results.append(metrics)
        except Exception as e:
            print(f"Error computing metrics for sample {i}: {e}")
            results.append({})
    
    return results


def find_best_sample_index(metrics: list, target_metrics: Dict[str, float], candidate_indices: Optional[list] = None) -> Optional[int]:
    """Find sample index with minimum mean absolute error over available target metrics."""
    best_idx = None
    best_score = None

    indices = candidate_indices if candidate_indices is not None else range(len(metrics))

    for idx in indices:
        if idx < 0 or idx >= len(metrics):
            continue
        sample_metrics = metrics[idx]
        if not sample_metrics:
            continue

        shared_keys = [k for k in target_metrics.keys() if k in sample_metrics]
        if len(shared_keys) == 0:
            continue

        score = sum(abs(float(sample_metrics[k]) - float(target_metrics[k])) for k in shared_keys) / len(shared_keys)
        if best_score is None or score < best_score:
            best_score = score
            best_idx = idx

    return best_idx


def compute_target_mae(sample_metrics: Dict[str, float], target_metrics: Dict[str, float]) -> Optional[float]:
    """Compute mean absolute error on shared target keys."""
    shared_keys = [k for k in target_metrics.keys() if k in sample_metrics]
    if len(shared_keys) == 0:
        return None
    return float(sum(abs(float(sample_metrics[k]) - float(target_metrics[k])) for k in shared_keys) / len(shared_keys))


def plot_sample_graph(samples: list, sample_idx: int, output_path: Path, save_pdf: bool = True):
    """Plot one sampled graph as PNG and optionally PDF."""
    atom_types, edge_types = samples[sample_idx]
    num_nodes = int(len(atom_types))
    adjacency = extract_adjacency_from_edge_types(edge_types.cpu().float())
    
    # Convert to numpy for plotting
    adjacency_np = np.asarray(adjacency)
    
    # Create graph for spring layout
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if float(adjacency[i, j]) > 0:
                graph.add_edge(i, j)
    
    # Generate 2D positions and pad to 3D
    pos_2d = nx.spring_layout(graph, seed=42, k=0.5, iterations=50)
    vertex_positions = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        if i in pos_2d:
            vertex_positions[i, :2] = pos_2d[i]
        else:
            vertex_positions[i] = 0.0
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Use the fancy plotting function
    fig, ax = plot_graph(
        adjacency_np,
        vertex_positions,
        pretty=True,
        save_path=output_path,
        save_dpi=300,
        save_pdf=save_pdf
    )
    plt.close('all')


def get_planar_sample_indices(samples: list) -> list:
    """Return indices of generated graphs that are planar."""
    planar_indices = []

    for idx, (atom_types, edge_types) in enumerate(samples):
        try:
            num_nodes = int(len(atom_types))
            adjacency = extract_adjacency_from_edge_types(edge_types.cpu().float())

            graph = nx.Graph()
            graph.add_nodes_from(range(num_nodes))
            for i in range(num_nodes):
                for j in range(i + 1, num_nodes):
                    if float(adjacency[i, j]) > 0:
                        graph.add_edge(i, j)

            is_planar, _ = nx.check_planarity(graph)
            if is_planar:
                planar_indices.append(idx)
        except Exception:
            continue

    return planar_indices


def compute_planarity_stats(samples: list, planar_indices: Optional[list] = None) -> Dict[str, float]:
    """Compute how many generated graphs are planar."""
    if planar_indices is None:
        planar_indices = get_planar_sample_indices(samples)

    planar_count = len(planar_indices)
    checked = len(samples)

    planar_ratio = (float(planar_count) / float(checked)) if checked > 0 else 0.0
    return {
        'planar_count': float(planar_count),
        'num_checked': float(checked),
        'planar_ratio': planar_ratio,
    }


def _largest_connected_component(graph: nx.Graph) -> nx.Graph:
    if graph.number_of_nodes() == 0:
        return graph.copy()
    if graph.number_of_nodes() == 1:
        return graph.copy()
    component_nodes = max(
        nx.connected_components(graph),
        key=lambda nodes: (len(nodes), -min(nodes)),
    )
    return nx.convert_node_labels_to_integers(graph.subgraph(component_nodes).copy())


def _sample_to_lcc_graph(sample) -> nx.Graph:
    atom_types, edge_types = sample
    num_nodes = int(len(atom_types))
    adjacency = extract_adjacency_from_edge_types(edge_types.cpu().float())
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if float(adjacency[i, j]) > 0:
                graph.add_edge(i, j)
    return _largest_connected_component(graph)


def _node_count_rank(num_nodes, target_num_nodes, *, window_size=10.0):
    if target_num_nodes is None:
        return (0, 0.0)
    distance = abs(float(num_nodes) - float(target_num_nodes))
    if distance <= float(window_size):
        bucket = 0
    else:
        bucket = 1 + int((distance - float(window_size)) // float(window_size))
    return (bucket, distance)


def _candidate_rank_tuple(metrics: Dict[str, float], reward: Optional[float], target_num_nodes: Optional[float]):
    node_bucket, node_distance = _node_count_rank(metrics.get('num_nodes', 0.0), target_num_nodes)
    is_planar = 1 if float(metrics.get('is_planar', 0.0)) >= 0.5 else 0
    safe_reward = float(reward) if reward is not None else -float('inf')
    return (
        is_planar,
        -node_bucket,
        -node_distance,
        safe_reward,
    )


def _constraint_summary(report: Optional[Dict[str, Any]]) -> str:
    if not report:
        return "n/a"
    return (
        f"{int(report['num_constraints_satisfied'])}/4 "
        f"(angle={report['passes_minimum_angle']}, "
        f"edge_length={report['passes_edge_length']}, "
        f"degree={report['passes_max_degree']}, "
        f"intersection={report['passes_edge_intersection']})"
    )


def _constraint_violations(report: Optional[Dict[str, Any]]) -> str:
    if not report:
        return "n/a"
    violations = []
    if not report.get('passes_minimum_angle', True):
        violations.append(f"minimum_angle={float(report.get('minimum_angle', 0.0)):.6f}")
    if not report.get('passes_edge_length', True):
        violations.append(f"edge_length_max={float(report.get('edge_length_max', 0.0)):.6f}")
    if not report.get('passes_max_degree', True):
        violations.append(f"max_degree={int(report.get('max_degree', 0))}")
    if not report.get('passes_edge_intersection', True):
        violations.append(
            f"edge_intersection_loss={float(report.get('edge_intersection_loss', 0.0)):.6f}"
        )
    return ", ".join(violations) if violations else "none"


def _format_candidate_metrics(metrics: Dict[str, float]) -> str:
    return ", ".join(
        f"{metric}={float(value):.6f}"
        for metric, value in sorted(metrics.items())
    )


def _format_objective_metrics(metrics: Dict[str, float], target_metrics: Dict[str, float]) -> str:
    ordered_metrics = []
    if 'num_nodes' in metrics:
        ordered_metrics.append('num_nodes')
    for metric_name in target_metrics.keys():
        if metric_name != 'num_nodes' and metric_name in metrics:
            ordered_metrics.append(metric_name)
    return ", ".join(f"{metric}={float(metrics[metric]):.6f}" for metric in ordered_metrics)


def _build_spring_embedded_data(graph: nx.Graph, *, seed: int):
    embedded_graph = add_node_positions(
        graph.copy(),
        method='spring',
        scale=10.0,
        seed=seed,
    )
    return networkx_to_pyg_data(embedded_graph)


def _compute_benchmark_candidate(sample, *, seed: int):
    graph = _sample_to_lcc_graph(sample)
    data = _build_spring_embedded_data(graph, seed=seed)
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
    if 'triangles' in metrics and 'triangle_count' not in metrics:
        metrics['triangle_count'] = float(metrics['triangles'])
    if 'triangle_count' in metrics and 'triangles' not in metrics:
        metrics['triangles'] = float(metrics['triangle_count'])
    metrics['is_planar'] = float(planarity_score)
    return graph, prepared, metrics, constraint_report


def _select_best_sample(samples: list, target_metrics: Dict[str, float], *, require_planar: bool = False):
    best_candidate = None
    best_rank = None
    reward_targets = {
        str(k): float(v)
        for k, v in target_metrics.items()
        if v is not None and str(k) != 'num_nodes'
    }
    target_num_nodes = target_metrics.get('num_nodes')

    for idx, sample in enumerate(samples):
        try:
            component_graph, prepared_data, metrics, constraint_report = _compute_benchmark_candidate(sample, seed=42 + idx)
        except Exception as e:
            print(f"Warning: benchmark evaluation failed for sample {idx}: {e}")
            continue
        if require_planar and metrics.get('is_planar', 0.0) < 0.5:
            continue
        reward, reward_breakdown = metric_reward(
            target_metrics=reward_targets,
            candidate_metrics=metrics,
            metric_reward_weight=1.0,
        )
        rank = _candidate_rank_tuple(metrics, reward, target_num_nodes)
        if best_rank is None or rank > best_rank:
            best_rank = rank
            best_candidate = {
                'index': int(idx),
                'graph': component_graph,
                'prepared_data': prepared_data,
                'metrics': metrics,
                'reward': float(reward),
                'reward_breakdown': reward_breakdown,
                'constraint_report': constraint_report,
                'rank': rank,
            }
    return best_candidate


def _save_candidate_artifacts(candidate: Dict[str, Any], target_metrics: Dict[str, float], output_path: Path, stem: str):
    tag = stem
    img_path = output_path / f'{tag}.png'
    json_path = output_path / f'{tag}_stats.json'
    constraint_path = output_path / f'{tag}_constraints.json'
    txt_path = output_path / f'{tag}_info.txt'

    prepared_data = candidate['prepared_data']
    num_nodes = int(prepared_data.num_nodes)
    adjacency = np.zeros((num_nodes, num_nodes), dtype=np.float32)
    edge_index = prepared_data.edge_index.detach().cpu().numpy()
    adjacency[edge_index[0], edge_index[1]] = 1.0
    pos = prepared_data.pos.detach().cpu().numpy()
    pos_3d = np.zeros((pos.shape[0], 3), dtype=np.float32)
    pos_3d[:, :2] = pos[:, :2]
    plot_graph(adjacency, pos_3d, pretty=True, save_path=img_path, save_dpi=300, save_pdf=True)
    plt.close('all')

    payload = {
        'index': int(candidate['index']),
        'reward': float(candidate['reward']),
        'reward_breakdown': {k: float(v) for k, v in candidate['reward_breakdown'].items()},
        'metrics': {k: float(v) for k, v in candidate['metrics'].items()},
        'constraint_report': candidate['constraint_report'],
        'constraint_violations': _constraint_violations(candidate['constraint_report']),
        'rank': {
            'planarity': int(candidate['rank'][0]),
            'node_bucket': int(-candidate['rank'][1]),
            'node_distance': float(-candidate['rank'][2]),
            'reward': float(candidate['rank'][3]),
        },
        'targets': {k: float(v) for k, v in target_metrics.items()},
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    constraint_payload = {
        'index': int(candidate['index']),
        'constraint_report': candidate['constraint_report'],
        'constraint_violations': _constraint_violations(candidate['constraint_report']),
        'targets': {k: float(v) for k, v in target_metrics.items()},
    }
    with open(constraint_path, 'w', encoding='utf-8') as f:
        json.dump(constraint_payload, f, indent=2)

    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"index: {int(candidate['index'])}\n")
        f.write(f"reward: {float(candidate['reward']):.6f}\n")
        f.write(f"objectives: {_format_objective_metrics(candidate['metrics'], target_metrics)}\n")
        f.write(
            "rank: "
            f"planarity={int(candidate['rank'][0])}, "
            f"node_bucket={int(-candidate['rank'][1])}, "
            f"node_distance={float(-candidate['rank'][2]):.6f}, "
            f"reward={float(candidate['rank'][3]):.6f}\n"
        )
        f.write(f"constraints: {_constraint_summary(candidate['constraint_report'])}\n")
        f.write(f"constraint_violations: {_constraint_violations(candidate['constraint_report'])}\n")
        f.write(f"metrics: {_format_candidate_metrics(candidate['metrics'])}\n")
        f.write(f"plot: {img_path}\n")
        f.write(f"stats: {json_path}\n")
        f.write(f"constraints_file: {constraint_path}\n")

    return img_path, json_path, constraint_path, txt_path


def _write_benchmark_summary(output_path: Path, run_id: str, target_metrics: Dict[str, float], best_graph, best_planar_graph):
    summary_path = output_path / f'{run_id}_benchmark_summary.txt'
    lines = [
        "Run target metrics:",
        *[f"  {metric_name}: {float(metric_value):.6f}" for metric_name, metric_value in target_metrics.items()],
        "",
        "Best graph (ranked by planarity, node-count window, reward):",
    ]
    if best_graph is None:
        lines.append("  none")
    else:
        lines.extend([
            f"  index: {int(best_graph['index'])}",
            f"  reward: {float(best_graph['reward']):.6f}",
            f"  objectives: {_format_objective_metrics(best_graph['metrics'], target_metrics)}",
            f"  constraints: {_constraint_summary(best_graph['constraint_report'])}",
            f"  constraint_violations: {_constraint_violations(best_graph['constraint_report'])}",
            f"  metrics: {_format_candidate_metrics(best_graph['metrics'])}",
            f"  plot: {best_graph['plot_path']}",
            f"  stats: {best_graph['stats_path']}",
        ])

    lines.extend(["", "Best planar graph (ranked by node-count window, reward):"])
    if best_planar_graph is None:
        lines.append("  none")
    else:
        lines.extend([
            f"  index: {int(best_planar_graph['index'])}",
            f"  reward: {float(best_planar_graph['reward']):.6f}",
            f"  objectives: {_format_objective_metrics(best_planar_graph['metrics'], target_metrics)}",
            f"  constraints: {_constraint_summary(best_planar_graph['constraint_report'])}",
            f"  constraint_violations: {_constraint_violations(best_planar_graph['constraint_report'])}",
            f"  metrics: {_format_candidate_metrics(best_planar_graph['metrics'])}",
            f"  plot: {best_planar_graph['plot_path']}",
            f"  stats: {best_planar_graph['stats_path']}",
        ])

    with open(summary_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")
    return summary_path


def _write_run_constraints_summary(
    output_path: Path,
    run_id: str,
    *,
    best_graph,
    best_planar_graph,
):
    summary_path = output_path / f'{run_id}_constraints.json'
    payload = {
        'best_graph': None if best_graph is None else {
            'index': int(best_graph['index']),
            'constraint_report': best_graph['constraint_report'],
            'constraint_violations': _constraint_violations(best_graph['constraint_report']),
            'constraints_path': str(best_graph['constraints_path']),
        },
        'best_planar_graph': None if best_planar_graph is None else {
            'index': int(best_planar_graph['index']),
            'constraint_report': best_planar_graph['constraint_report'],
            'constraint_violations': _constraint_violations(best_planar_graph['constraint_report']),
            'constraints_path': str(best_planar_graph['constraints_path']),
        },
    }
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return summary_path


def evaluate_samples(
    samples: list,
    target_metrics: Dict[str, float],
    output_dir: str,
    run_id: str,
):
    metrics = compute_metrics_of_samples(samples, metric_backend='graph_rl')
    save_results(
        samples,
        metrics,
        target_metrics,
        output_dir,
        run_id,
    )


def evaluate_sample_file(
    sample_path: str,
    target_metrics: Dict[str, float],
    output_dir: Optional[str] = None,
    run_id: Optional[str] = None,
):
    resolved_path = Path(sample_path)
    if not resolved_path.exists() and resolved_path.suffix != '.pkl':
        candidates = [
            resolved_path.with_suffix('.pkl'),
            resolved_path.parent / 'samples.pkl',
            resolved_path.parent / 'generated_samples.pkl',
        ]
        for candidate in candidates:
            if candidate.exists():
                resolved_path = candidate
                break
    if not resolved_path.exists():
        raise FileNotFoundError(f"Could not find sample file at {sample_path}")

    with open(resolved_path, 'rb') as f:
        samples = pickle.load(f)

    sample_run_id = resolved_path.stem if run_id is None else run_id
    sample_output_dir = str(resolved_path.parent if output_dir is None else output_dir)
    evaluate_samples(
        samples,
        target_metrics,
        sample_output_dir,
        sample_run_id,
    )


def save_results(
    samples: list,
    metrics: list,
    target_metrics: Dict[str, float],
    output_dir: str,
    run_id: str,
):
    """Save samples and metrics to files."""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    planar_indices = get_planar_sample_indices(samples)
    
    # Save samples
    samples_file = output_path / f'{run_id}_samples.pkl'
    with open(samples_file, 'wb') as f:
        pickle.dump(samples, f)
    print(f"Saved samples to {samples_file}")
    
    # Save metrics
    planarity_stats = compute_planarity_stats(samples, planar_indices=planar_indices)
    metrics_file = output_path / f'{run_id}_metrics.pkl'
    with open(metrics_file, 'wb') as f:
        pickle.dump({'target': target_metrics, 'actual': metrics, 'planarity': planarity_stats,
                     'planar_indices': planar_indices}, f)
    print(f"Saved metrics to {metrics_file}")

    best_graph_candidate = _select_best_sample(samples, target_metrics, require_planar=False)
    best_planar_candidate = _select_best_sample(samples, target_metrics, require_planar=True)

    best_graph_summary = None
    if best_graph_candidate is not None:
        best_graph_plot_path, best_graph_stats_path, best_graph_constraints_path, best_graph_info_path = _save_candidate_artifacts(
            best_graph_candidate,
            target_metrics,
            output_path,
            f'{run_id}_best_graph_overall',
        )
        best_graph_summary = {
            **best_graph_candidate,
            'plot_path': best_graph_plot_path,
            'stats_path': best_graph_stats_path,
            'constraints_path': best_graph_constraints_path,
            'info_path': best_graph_info_path,
        }
        print(f"Saved best overall graph artifacts to {best_graph_plot_path}")
    else:
        print("Warning: could not determine best overall graph.")

    best_planar_summary = None
    if best_planar_candidate is not None:
        best_planar_plot_path, best_planar_stats_path, best_planar_constraints_path, best_planar_info_path = _save_candidate_artifacts(
            best_planar_candidate,
            target_metrics,
            output_path,
            f'{run_id}_best_graph_planar',
        )
        best_planar_summary = {
            **best_planar_candidate,
            'plot_path': best_planar_plot_path,
            'stats_path': best_planar_stats_path,
            'constraints_path': best_planar_constraints_path,
            'info_path': best_planar_info_path,
        }
        print(f"Saved best planar graph artifacts to {best_planar_plot_path}")
    else:
        print("Warning: no planar sample found after largest-connected-component evaluation.")

    summary_path = _write_benchmark_summary(output_path, run_id, target_metrics, best_graph_summary, best_planar_summary)
    constraints_summary_path = _write_run_constraints_summary(
        output_path,
        run_id,
        best_graph=best_graph_summary,
        best_planar_graph=best_planar_summary,
    )
    print(f"Saved benchmark summary to {summary_path}")
    print(f"Saved constraint summary to {constraints_summary_path}")

    print("\nSaved artifacts:")
    print(f"  run_id: {run_id}")
    print(f"  samples: {samples_file.resolve()}")
    print(f"  metrics: {metrics_file.resolve()}")
    print(f"  benchmark_summary: {summary_path.resolve()}")
    print(f"  constraints_summary: {constraints_summary_path.resolve()}")
    if best_graph_summary is not None:
        best_graph_no_axes = best_graph_summary['plot_path'].with_name(
            f"{best_graph_summary['plot_path'].stem}_no_axes{best_graph_summary['plot_path'].suffix}"
        )
        print(f"  best_graph_overall_plot: {best_graph_summary['plot_path'].resolve()}")
        print(f"  best_graph_overall_plot_pdf: {best_graph_summary['plot_path'].with_suffix('.pdf').resolve()}")
        print(f"  best_graph_overall_no_axes: {best_graph_no_axes.resolve()}")
        print(f"  best_graph_overall_constraints: {best_graph_summary['constraints_path'].resolve()}")
        print(f"  best_sample_overall_info: {best_graph_summary['info_path'].resolve()}")
    if best_planar_summary is not None:
        best_planar_no_axes = best_planar_summary['plot_path'].with_name(
            f"{best_planar_summary['plot_path'].stem}_no_axes{best_planar_summary['plot_path'].suffix}"
        )
        print(f"  best_graph_planar_plot: {best_planar_summary['plot_path'].resolve()}")
        print(f"  best_graph_planar_plot_pdf: {best_planar_summary['plot_path'].with_suffix('.pdf').resolve()}")
        print(f"  best_graph_planar_no_axes: {best_planar_no_axes.resolve()}")
        print(f"  best_graph_planar_constraints: {best_planar_summary['constraints_path'].resolve()}")
        print(f"  best_sample_planar_info: {best_planar_summary['info_path'].resolve()}")
    
    # Print summary
    print("\n" + "="*80)
    print("Sampling Summary")
    print("="*80)
    print(f"\nTarget Metrics:")
    for key, val in target_metrics.items():
        print(f"  {key}: {val:.4f}")
    
    overall_metrics = [m for m in metrics if m]
    print(f"\nActual Metrics (averaged over {len(overall_metrics)} total samples with valid metrics):")
    if overall_metrics:
        avg_metrics = {}
        for key in overall_metrics[0].keys():
            values = [m.get(key, 0) for m in overall_metrics if key in m]
            if values:
                avg_metrics[key] = sum(values) / len(values)

        for key, val in avg_metrics.items():
            target_val = target_metrics.get(key, None)
            if target_val is not None:
                error = abs(val - target_val) / (abs(target_val) + 1e-8)
                print(f"  {key}: {val:.4f} (target: {target_val:.4f}, error: {error*100:.1f}%)")
            else:
                print(f"  {key}: {val:.4f}")
    else:
        print("  No samples with valid metrics available.")

    planar_metrics = [metrics[i] for i in planar_indices if i < len(metrics) and metrics[i]]
    print(f"\nActual Metrics (averaged over {len(planar_metrics)} planar samples):")
    if planar_metrics:
        avg_metrics = {}
        for key in planar_metrics[0].keys():
            values = [m.get(key, 0) for m in planar_metrics if key in m]
            if values:
                avg_metrics[key] = sum(values) / len(values)
        
        for key, val in avg_metrics.items():
            target_val = target_metrics.get(key, None)
            if target_val is not None:
                error = abs(val - target_val) / (abs(target_val) + 1e-8)
                print(f"  {key}: {val:.4f} (target: {target_val:.4f}, error: {error*100:.1f}%)")
            else:
                print(f"  {key}: {val:.4f}")

        if best_planar_summary is not None:
            print("\nBest Planar Metrics:")
            print(f"  best_planar_reward: {float(best_planar_summary['reward']):.4f}")
            print(f"  best_planar_constraints: {_constraint_summary(best_planar_summary['constraint_report'])}")
            best_planar_metrics = best_planar_summary['metrics']
            for key, val in best_planar_metrics.items():
                target_val = target_metrics.get(key, None)
                if target_val is not None:
                    error = abs(float(val) - float(target_val)) / (abs(float(target_val)) + 1e-8)
                    print(f"  {key}: {float(val):.4f} (target: {float(target_val):.4f}, error: {error*100:.1f}%)")
                else:
                    print(f"  {key}: {float(val):.4f}")
    else:
        print("  No planar samples with valid metrics available.")

    print("\nPlanarity:")
    print(f"  planar_graphs: {int(planarity_stats['planar_count'])}/{int(planarity_stats['num_checked'])}")
    print(f"  planar_ratio: {100.0 * planarity_stats['planar_ratio']:.2f}%")
    print("="*80 + "\n")


def main(args):
    """Main sampling function."""
    parsed_targets = parse_target_metrics(args.target_metrics)
    run_label = 'custom_targets'
    if args.use_case is not None:
        target_dict = USE_CASE_TARGETS[args.use_case]
        run_label = args.use_case
        print(f"Using preset use case {args.use_case}: {target_dict}")
    elif len(parsed_targets) > 0:
        target_dict = parsed_targets
    else:
        raise ValueError("Must specify either --use_case or --target_metrics")

    if args.sample_path:
        evaluate_sample_file(
            args.sample_path,
            target_dict,
            output_dir=args.output_dir,
            run_id=args.run_id,
        )
        return

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    metric_subset = parse_metric_subset(args.metric_subset)
    if args.use_case is not None:
        metric_subset = USE_CASE_METRIC_SUBSETS[args.use_case]
    guidance_weight_map = {}
    if args.use_case is not None:
        guidance_weight_map.update(USE_CASE_GUIDANCE_WEIGHTS.get(args.use_case, {}))
    guidance_weight_map.update(parse_metric_weights(args.guidance_metric_weights))
    if not args.regressor_checkpoint:
        raise ValueError("--regressor_checkpoint is required unless --sample_path is used")
    
    # Load model and regressor
    model, regressor = load_model_and_regressor(
        args.model_checkpoint,
        args.regressor_checkpoint,
        device
    )
    
    target_metrics = create_target_metrics_tensor(target_dict, metric_order=metric_subset)
    guidance_weight_tensor = create_guidance_weight_tensor(metric_subset, guidance_weight_map)
    if guidance_weight_map:
        print(f"Using guidance metric weights: {guidance_weight_map}")
    
    # Sample
    samples = sample_conditional(
        model,
        regressor,
        target_metrics,
        num_samples=args.num_samples,
        guidance_scale=args.guidance_scale,
        device=device,
        guidance_metric_weights=guidance_weight_tensor.to(device),
    )
    
    # Compute metrics
    metrics = compute_metrics_of_samples(samples, metric_backend=args.metric_backend, metric_subset=metric_subset)
    
    # Save results
    run_id = f"{run_label}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    save_results(
        samples,
        metrics,
        target_dict,
        args.output_dir,
        run_id,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Sample graphs with guided conditional generation')
    parser.add_argument('--model_checkpoint', type=str, required=False, help='Path to trained model checkpoint', default='/ext/DiGress/outputs/2026-03-26/10-31-03-hog_planar_unconditional/checkpoints/hog_planar_unconditional/last.ckpt')
    parser.add_argument('--regressor_checkpoint', type=str, required=False, help='Path to trained regressor checkpoint')
    parser.add_argument('--metric_backend', type=str, default='spectre', choices=['graph_rl', 'spectre'],
                        help='Metric backend for post-sampling evaluation')
    parser.add_argument('--metric_subset', type=str,
                        default='num_nodes,spectral_gap,gini_coefficient,clustering_coefficient',
                        help='Comma-separated metric names in the exact regressor output order')
    parser.add_argument('--use_case', type=str, default=None, 
                        choices=['use_case_0', 'use_case_1', 'use_case_2', 'use_case_3', 'all_4_use_cases'],
                        help='Use predefined target metrics for a specific use-case (overrides --target_metrics and --target_*).')
    parser.add_argument('--target_metrics', type=str, default='',
                        help='Optional comma-separated targets key=value,key=value (alternative to --use_case)')
    parser.add_argument('--num_samples', type=int, default=10, help='Number of samples to generate')
    parser.add_argument('--guidance_scale', type=float, default=1.0, help='Guidance scale strength')
    parser.add_argument('--guidance_metric_weights', type=str, default='',
                        help='Optional comma-separated guidance weights key=value,key=value')
    parser.add_argument('--sample_path', type=str, default='',
                        help='Evaluate an existing DiGress sample pickle instead of generating new samples')
    parser.add_argument('--output_dir', type=str, default='outputs/conditional_samples/', help='Directory to save results')
    parser.add_argument('--run_id', type=str, default=None, help='Optional run identifier override')
    
    args = parser.parse_args()
    main(args)

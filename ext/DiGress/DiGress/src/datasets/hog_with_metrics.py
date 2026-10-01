"""Metric augmentation wrapper for HOG planar dataset used by regressor training."""

import os
from pathlib import Path
from typing import Optional, Dict, List, cast

import torch
import torch.utils.data
import torch_geometric.utils
from torch_geometric.utils import to_networkx
import networkx as nx
from scipy.linalg import eigvalsh
from tqdm import tqdm

from src.datasets.hog_planar_dataset import HOGPlanarDataModule
from src.analysis.spectre_utils import is_planar_graph, orca

# The regressor-target preprocessing uses JAX-backed geometry code only for CPU metrics.
os.environ.setdefault("JAX_PLATFORMS", "cpu")

# Optional legacy backend
from graph_rl.ppo.metrics import GraphMetrics
from graph_rl.utils.benchmark_graphs import prepare_graph_for_benchmark_evaluation
from graph_rl.utils.graph_helpers import add_node_positions, networkx_to_pyg_data


def extract_adjacency_from_edge_types(edge_types: torch.Tensor) -> torch.Tensor:
    if edge_types.dim() == 3:
        if edge_types.size(-1) > 1:
            adjacency = edge_types[..., 1]
        else:
            adjacency = edge_types[..., 0]
    else:
        adjacency = edge_types
    adjacency = (adjacency > 0.5).float()
    adjacency = torch.triu(adjacency, diagonal=1)
    adjacency = adjacency + adjacency.transpose(0, 1)
    return adjacency


def compute_graph_metrics_from_adjacency(
    adjacency: torch.Tensor,
    num_nodes: int,
    metric_backend: str = 'spectre',
    graph_metrics: Optional[GraphMetrics] = None,
) -> Dict[str, float]:
    if metric_backend == 'graph_rl':
        gm = graph_metrics if graph_metrics is not None else GraphMetrics()
        triangles = gm.triangle_count(adjacency)
        triplets = gm.triplet_count(adjacency)
        metrics = {
            'num_nodes': float(num_nodes),
            'spectral_gap': float(gm.spectral_gap(adjacency)),
            'gini_coefficient': float(gm._gini_from_degrees(adjacency)),
            'clustering_coefficient': float(gm.clustering_coefficient(triangles, triplets)),
            'triangle_count': float(triangles),
        }
        try:
            graph_nx = nx.from_numpy_array(adjacency.detach().cpu().numpy())
            graph_nx = add_node_positions(
                graph_nx,
                method='spring',
                scale=10.0,
                seed=42,
            )
            data = networkx_to_pyg_data(graph_nx)
            prepared, planar_metrics, _constraint_report = prepare_graph_for_benchmark_evaluation(
                data,
                strict=False,
            )
            del prepared
            for metric_name in (
                'isoperimetric_ratio',
                'rectangularity',
                'angular_resolution',
                'edge_length_deviation',
                'edge_length_uniformity',
                'edge_length_min',
                'edge_length_max',
                'minimum_angle',
            ):
                metric_value = planar_metrics.get(metric_name)
                if metric_value is not None:
                    metrics[metric_name] = float(metric_value)
        except Exception:
            metrics.setdefault('isoperimetric_ratio', 0.0)
            metrics.setdefault('rectangularity', 0.0)
            metrics.setdefault('angular_resolution', 0.0)
            metrics.setdefault('edge_length_deviation', 0.0)
            metrics.setdefault('edge_length_uniformity', 0.0)
            metrics.setdefault('edge_length_min', 0.0)
            metrics.setdefault('edge_length_max', 0.0)
            metrics.setdefault('minimum_angle', 0.0)
        return metrics

    graph_nx = nx.from_numpy_array(adjacency.detach().cpu().numpy())
    metrics: Dict[str, float] = {
        'num_nodes': float(num_nodes),
        'num_edges': float(graph_nx.number_of_edges()),
        'avg_degree': (2.0 * graph_nx.number_of_edges() / num_nodes) if num_nodes > 0 else 0.0,
    }

    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(graph_nx).todense())
        eigs = sorted(float(v) for v in eigs)
        metrics['spectral_gap'] = eigs[1] - eigs[0] if len(eigs) > 1 else 0.0
        metrics['spectral_radius'] = eigs[-1] if len(eigs) > 0 else 0.0
    except Exception:
        metrics['spectral_gap'] = 0.0
        metrics['spectral_radius'] = 0.0

    try:
        metrics['clustering_coefficient'] = float(nx.average_clustering(graph_nx))
    except Exception:
        metrics['clustering_coefficient'] = 0.0

    try:
        triangle_dict = cast(Dict[int, int], nx.triangles(graph_nx))
        triangle_sum = sum(triangle_dict.values())
        metrics['triangle_count'] = float(triangle_sum // 3)
    except Exception:
        metrics['triangle_count'] = 0.0

    try:
        degrees = [d for _, d in graph_nx.degree()]
        if len(degrees) == 0 or sum(degrees) == 0:
            metrics['gini_coefficient'] = 0.0
        else:
            deg_sorted = sorted(float(d) for d in degrees)
            deg_n = len(deg_sorted)
            gini = (2.0 * sum((i + 1) * d for i, d in enumerate(deg_sorted))) / (deg_n * sum(deg_sorted))
            gini -= (deg_n + 1) / deg_n
            metrics['gini_coefficient'] = float(gini)
    except Exception:
        metrics['gini_coefficient'] = 0.0

    try:
        metrics['planar_valid'] = 1.0 if is_planar_graph(graph_nx) else 0.0
    except Exception:
        metrics['planar_valid'] = 0.0

    try:
        orbit_counts = orca(graph_nx)
        orbit_per_graph = orbit_counts.sum(axis=0) / max(1, graph_nx.number_of_nodes())
        metrics['orbit_mass'] = float(orbit_per_graph.sum())
    except Exception:
        metrics['orbit_mass'] = 0.0

    return metrics


class _MetricsAugmentedDataset(torch.utils.data.Dataset):
    """Wrap an existing dataset and attach per-graph metric targets in `data.y`."""

    CACHE_VERSION = 'v1'

    def __init__(
        self,
        base_dataset,
        metric_subset: Optional[List[str]] = None,
        metric_backend: str = 'spectre',
    ):
        self.base_dataset = base_dataset
        self.metric_backend = metric_backend
        self.metric_subset = metric_subset or [
            'num_nodes',
            'spectral_gap',
            'gini_coefficient',
            'clustering_coefficient',
            'triangle_count',
            'isoperimetric_ratio',
            'rectangularity',
            'angular_resolution',
            'edge_length_deviation',
        ]
        self.metrics_cache: Dict[int, Dict[str, float]] = {}
        self.graph_metrics = GraphMetrics() if metric_backend == 'graph_rl' else None
        self.cache_path = self._build_cache_path()
        self._load_disk_cache()

    def _build_cache_path(self) -> Path:
        processed_dir = Path(getattr(self.base_dataset, 'processed_dir'))
        cache_dir = processed_dir / 'metric_cache'
        cache_dir.mkdir(parents=True, exist_ok=True)
        split = str(getattr(self.base_dataset, 'split', 'train'))
        return cache_dir / f'hog_{split}_{self.metric_backend}_{self.CACHE_VERSION}.pt'

    def _load_disk_cache(self) -> None:
        if not self.cache_path.exists():
            return
        payload = torch.load(self.cache_path, weights_only=False)
        if not isinstance(payload, dict):
            return
        cache = payload.get('metrics_cache', {})
        if not isinstance(cache, dict):
            return
        self.metrics_cache = {
            int(idx): {str(k): float(v) for k, v in metrics.items()}
            for idx, metrics in cache.items()
            if isinstance(metrics, dict)
        }

    def _save_disk_cache(self) -> None:
        payload = {
            'metric_backend': self.metric_backend,
            'metric_subset': list(self.metric_subset),
            'cache_version': self.CACHE_VERSION,
            'metrics_cache': self.metrics_cache,
        }
        torch.save(payload, self.cache_path)

    def precompute_all_metrics(self) -> None:
        expected = len(self.base_dataset)
        missing = [idx for idx in range(expected) if idx not in self.metrics_cache]
        if not missing:
            return
        split = str(getattr(self.base_dataset, 'split', 'train'))
        desc = f"Precompute {split} metrics"
        for idx in tqdm(missing, desc=desc):
            data = self.base_dataset[idx].clone()
            self.metrics_cache[idx] = self._compute_metrics(data)
        self._save_disk_cache()

    def __len__(self):
        return len(self.base_dataset)

    def __getitem__(self, idx):
        data = self.base_dataset[idx].clone()

        if idx not in self.metrics_cache:
            self.metrics_cache[idx] = self._compute_metrics(data)

        metrics = self.metrics_cache[idx]
        metrics_tensor = torch.tensor([metrics.get(m, 0.0) for m in self.metric_subset], dtype=torch.float)
        data.y = metrics_tensor.unsqueeze(0)
        data.idx = torch.tensor([idx])
        return data

    def _compute_metrics(self, data) -> Dict[str, float]:
        if self.metric_backend == 'graph_rl':
            return self._compute_graph_rl_metrics(data)
        return self._compute_spectre_compatible_metrics(data)

    def _compute_graph_rl_metrics(self, data) -> Dict[str, float]:
        if self.graph_metrics is None:
            raise RuntimeError("GraphMetrics backend requested but graph_metrics is None")

        n = int(data.x.size(0))
        adjacency = torch_geometric.utils.to_dense_adj(data.edge_index, max_num_nodes=n).squeeze(0)
        return compute_graph_metrics_from_adjacency(
            adjacency=adjacency,
            num_nodes=n,
            metric_backend='graph_rl',
            graph_metrics=self.graph_metrics,
        )

    def _compute_spectre_compatible_metrics(self, data) -> Dict[str, float]:
        graph_nx = to_networkx(data, node_attrs=None, edge_attrs=None, to_undirected=True, remove_self_loops=True)
        adjacency = torch.from_numpy(nx.to_numpy_array(graph_nx)).float()
        return compute_graph_metrics_from_adjacency(
            adjacency=adjacency,
            num_nodes=graph_nx.number_of_nodes(),
            metric_backend='spectre',
        )


class HOGDataModuleWithMetrics(HOGPlanarDataModule):
    """HOG planar datamodule + per-graph metric augmentation for regressor training only."""

    def __init__(self, cfg, metric_subset: Optional[List[str]] = None, metric_backend: str = 'spectre'):
        super().__init__(cfg)
        self.metric_subset = metric_subset
        self.metric_backend = metric_backend

        self.train_dataset = _MetricsAugmentedDataset(self.train_dataset, metric_subset, metric_backend)
        self.val_dataset = _MetricsAugmentedDataset(self.val_dataset, metric_subset, metric_backend)
        self.test_dataset = _MetricsAugmentedDataset(self.test_dataset, metric_subset, metric_backend)
        self.inner = self.train_dataset

    def __getitem__(self, item):
        return self.inner[item]

    def get_metrics_cache(self, split: str = 'train') -> Dict[int, Dict[str, float]]:
        if split == 'train':
            return getattr(self.train_dataset, 'metrics_cache', {})
        if split == 'val':
            return getattr(self.val_dataset, 'metrics_cache', {})
        if split == 'test':
            return getattr(self.test_dataset, 'metrics_cache', {})
        raise ValueError(f"Unknown split: {split}")

    def precompute_metric_caches(self) -> None:
        for dataset in (self.train_dataset, self.val_dataset, self.test_dataset):
            precompute = getattr(dataset, 'precompute_all_metrics', None)
            if callable(precompute):
                precompute()

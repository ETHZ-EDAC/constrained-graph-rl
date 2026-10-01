from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch


# Compatibility placeholders for scripts that import these names directly.
qm9_with_h: Dict[str, Any] = {}
qm9_without_h: Dict[str, Any] = {}
geom_with_h: Dict[str, Any] = {}
geom_without_h: Dict[str, Any] = {}

_DEFAULT_PLANAR_PATH = (Path(__file__).resolve().parents[1] / "planar_graph.pt").resolve()


def _load_graphs_from_file(path: Path) -> List[Any]:
    obj = torch.load(path, weights_only=False)

    if isinstance(obj, dict):
        graphs: List[Any] = []
        for key in ("train", "valid", "val", "test", "graphs", "data"):
            if key in obj and isinstance(obj[key], (list, tuple)):
                graphs.extend(list(obj[key]))
        if graphs:
            return graphs
        raise TypeError(f"Unsupported planar dataset dict schema in {path}.")

    if isinstance(obj, (list, tuple)):
        return list(obj)

    raise TypeError(f"Unsupported planar dataset type {type(obj)} in {path}.")


def _resolve_planar_graphs(datadir: Optional[str]) -> tuple[List[Any], Path]:
    if datadir:
        user_path = Path(datadir).expanduser()

        if user_path.is_file() and user_path.suffix == ".pt":
            return _load_graphs_from_file(user_path), user_path.resolve()

        if user_path.is_dir():
            split_files = [user_path / "train.pt", user_path / "val.pt", user_path / "valid.pt", user_path / "test.pt"]
            existing = [p for p in split_files if p.exists()]
            if existing:
                graphs: List[Any] = []
                for p in existing:
                    graphs.extend(_load_graphs_from_file(p))
                return graphs, user_path.resolve()

            candidate = user_path / "planar_graph.pt"
            if candidate.exists():
                return _load_graphs_from_file(candidate), candidate.resolve()

    if _DEFAULT_PLANAR_PATH.exists():
        return _load_graphs_from_file(_DEFAULT_PLANAR_PATH), _DEFAULT_PLANAR_PATH

    raise FileNotFoundError(
        "Unable to locate planar dataset. Set --datadir to a .pt file or directory containing planar graph tensors."
    )


def _num_nodes(graph: Any) -> int:
    num_nodes = getattr(graph, "num_nodes", None)
    if num_nodes is not None:
        return int(num_nodes)

    x = getattr(graph, "x", None)
    if x is not None:
        return int(x.shape[0])

    edge_index = getattr(graph, "edge_index", None)
    if edge_index is not None and edge_index.numel() > 0:
        return int(edge_index.max().item()) + 1

    return 0


def _undirected_adjacency(graph: Any, num_nodes: int) -> torch.Tensor:
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.bool)

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

    adj[src, dst] = True
    adj[dst, src] = True
    return adj


@lru_cache(maxsize=8)
def _get_planar_dataset_info_cached(path_key: str) -> Dict[str, Any]:
    graphs, resolved_path = _resolve_planar_graphs(path_key if path_key else None)

    if len(graphs) == 0:
        raise ValueError(f"Planar dataset at {resolved_path} is empty.")

    node_hist = Counter()
    max_nodes = 0
    max_degree = 0
    edge_count_total = 0
    no_edge_count_total = 0

    for graph in graphs:
        n = _num_nodes(graph)
        if n <= 0:
            continue

        node_hist[n] += 1
        max_nodes = max(max_nodes, n)

        adj = _undirected_adjacency(graph, n)
        degree = adj.sum(dim=1)
        if degree.numel() > 0:
            max_degree = max(max_degree, int(degree.max().item()))

        directed_edges = int(adj.sum().item())
        possible_directed_edges = n * max(n - 1, 0)
        edge_count_total += directed_edges
        no_edge_count_total += max(possible_directed_edges - directed_edges, 0)

    if not node_hist:
        raise ValueError(f"Planar dataset at {resolved_path} has no valid graphs.")

    n_nodes_hist = {k: int(node_hist[k]) for k in sorted(node_hist)}
    # A uniform prior is more numerically stable than an extremely imbalanced empirical no-edge/edge ratio.
    edge_types = [1.0, 1.0]

    return {
        "name": "planar",
        "atom_encoder": {"C": 0},
        "atom_decoder": ["C"],
        "n_nodes": n_nodes_hist,
        "max_n_nodes": int(max_nodes),
        "max_weight": 1.0,
        "atom_weights": [1.0],
        "edge_types": edge_types,
        # Diffusion-noised adjacency can become denser than the empirical training degree distribution.
        "max_in_deg": int(max_nodes),
        "max_out_deg": int(max_nodes),
        "max_num_edges": 2,
        "max_spatial": int(max_nodes),
        "max_edge_dist": int(max_nodes),
        "dataset_path": str(resolved_path),
    }


def get_dataset_info(dataset: str, remove_h: bool = False, datadir: Optional[str] = None) -> Dict[str, Any]:
    del remove_h

    dataset_name = dataset.lower()

    if "planar" in dataset_name:
        key = str(Path(datadir).expanduser().resolve()) if datadir else ""
        return _get_planar_dataset_info_cached(key)

    if "qm9" in dataset_name:
        raise ValueError(
            "QM9 dataset configuration is unavailable in this MuDiff checkout. "
            "Use --dataset planar with a planar .pt dataset or add the original QM9 configs."
        )

    if "geom" in dataset_name:
        raise ValueError(
            "GEOM dataset configuration is unavailable in this MuDiff checkout. "
            "Use --dataset planar or add GEOM configs."
        )

    raise ValueError(f"Unknown dataset '{dataset}'.")

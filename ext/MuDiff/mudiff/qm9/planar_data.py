from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm.auto import tqdm

from qm9.data.collate import PreprocessQM9
from qm9.data.dataset_class import ProcessedDataset
from qm9.planar_metrics_adapter import compute_graph_metric_dict


_DEFAULT_PLANAR_PATH = (Path(__file__).resolve().parents[1] / "planar_graph.pt").resolve()


def _load_graphs_from_pt(path: Path) -> List[Any]:
    obj = torch.load(path, weights_only=False)

    if isinstance(obj, (list, tuple)):
        return list(obj)

    if isinstance(obj, dict):
        graphs: List[Any] = []
        for key in ("train", "valid", "val", "test", "graphs", "data"):
            if key in obj and isinstance(obj[key], (list, tuple)):
                graphs.extend(list(obj[key]))
        if graphs:
            return graphs

    raise TypeError(f"Unsupported planar dataset schema in {path}.")


def _resolve_graph_splits(datadir: str) -> Dict[str, List[Any]]:
    path = Path(datadir).expanduser()

    if path.is_file() and path.suffix == ".pt":
        graphs = _load_graphs_from_pt(path)
        return _split_graphs(graphs, seed=42)

    if path.is_dir():
        split_paths = {
            "train": path / "train.pt",
            "valid": path / "valid.pt",
            "val": path / "val.pt",
            "test": path / "test.pt",
        }
        has_explicit_split = split_paths["train"].exists() and split_paths["test"].exists() and (
            split_paths["valid"].exists() or split_paths["val"].exists()
        )
        if has_explicit_split:
            valid_path = split_paths["valid"] if split_paths["valid"].exists() else split_paths["val"]
            return {
                "train": _load_graphs_from_pt(split_paths["train"]),
                "valid": _load_graphs_from_pt(valid_path),
                "test": _load_graphs_from_pt(split_paths["test"]),
            }

        candidate = path / "planar_graph.pt"
        if candidate.exists():
            graphs = _load_graphs_from_pt(candidate)
            return _split_graphs(graphs, seed=42)

    if _DEFAULT_PLANAR_PATH.exists():
        graphs = _load_graphs_from_pt(_DEFAULT_PLANAR_PATH)
        return _split_graphs(graphs, seed=42)

    raise FileNotFoundError(
        "Unable to locate planar dataset. Set --datadir to a .pt file or directory containing train/val/test .pt files."
    )


def _split_graphs(graphs: Sequence[Any], seed: int = 42, val_ratio: float = 0.1, test_ratio: float = 0.1) -> Dict[str, List[Any]]:
    if len(graphs) == 0:
        raise ValueError("Planar dataset is empty.")

    n = len(graphs)
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(seed)).tolist()
    shuffled = [graphs[i] for i in perm]

    n_test = max(1, int(n * test_ratio))
    n_val = max(1, int(n * val_ratio))
    n_train = n - n_val - n_test

    if n_train <= 0:
        n_train = max(1, n - 2)
        n_val = 1
        n_test = n - n_train - n_val

    return {
        "train": shuffled[:n_train],
        "valid": shuffled[n_train:n_train + n_val],
        "test": shuffled[n_train + n_val:],
    }


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


def _spring_positions(num_nodes: int, edge_index: torch.Tensor, seed: int = 42) -> torch.Tensor:
    try:
        import networkx as nx

        graph = nx.Graph()
        graph.add_nodes_from(range(num_nodes))

        if edge_index is not None and edge_index.numel() > 0:
            edges = edge_index.t().tolist()
            graph.add_edges_from((int(u), int(v)) for u, v in edges if int(u) != int(v))

        pos_dict = nx.spring_layout(graph, seed=seed)
        pos = torch.zeros((num_nodes, 2), dtype=torch.float32)
        for node_id in range(num_nodes):
            coords = pos_dict[node_id]
            pos[node_id] = torch.tensor([coords[0], coords[1]], dtype=torch.float32)
        return pos
    except Exception:
        generator = torch.Generator().manual_seed(seed)
        return 2.0 * torch.rand((num_nodes, 2), generator=generator, dtype=torch.float32) - 1.0


def _extract_positions(graph: Any, num_nodes: int, target_dims: int) -> torch.Tensor:
    pos = getattr(graph, "pos", None)

    if pos is None:
        edge_index = getattr(graph, "edge_index", None)
        pos = _spring_positions(num_nodes=num_nodes, edge_index=edge_index)
    else:
        pos = pos.float()
        if pos.dim() == 1:
            pos = pos.unsqueeze(-1)

    if pos.size(0) != num_nodes:
        raise ValueError(f"Position tensor has {pos.size(0)} nodes but expected {num_nodes}.")

    if pos.size(1) >= target_dims:
        return pos[:, :target_dims]

    padded = torch.zeros((num_nodes, target_dims), dtype=torch.float32)
    padded[:, :pos.size(1)] = pos
    return padded


def _to_dense_graph(graph: Any, target_dims: int) -> Dict[str, torch.Tensor]:
    num_nodes = _num_nodes(graph)
    if num_nodes <= 0:
        raise ValueError("Encountered graph with no nodes.")

    edge_index = getattr(graph, "edge_index", None)
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.long)

    if edge_index is not None and edge_index.numel() > 0:
        src = edge_index[0].long()
        dst = edge_index[1].long()
        valid = src != dst
        src = src[valid]
        dst = dst[valid]
        if src.numel() > 0:
            adj[src, dst] = 1
            adj[dst, src] = 1

    edge_attr = torch.zeros((num_nodes, num_nodes, 2), dtype=torch.float32)
    edge_attr[:, :, 0] = 1.0
    edge_attr[adj == 1, 0] = 0.0
    edge_attr[adj == 1, 1] = 1.0

    metric_values = compute_graph_metric_dict(graph)

    dense_graph = {
        "charges": torch.ones((num_nodes,), dtype=torch.long),
        "positions": _extract_positions(graph, num_nodes, target_dims=target_dims),
        "adj": adj,
        "edge_attr": edge_attr,
        "num_atoms": torch.tensor(num_nodes, dtype=torch.long),
    }

    for metric_name, metric_value in metric_values.items():
        dense_graph[metric_name] = torch.tensor(metric_value, dtype=torch.float32)

    return dense_graph


def _pack_graphs(graphs: Sequence[Any], target_dims: int) -> Dict[str, torch.Tensor]:
    dense_graphs = [
        _to_dense_graph(graph, target_dims=target_dims)
        for graph in tqdm(
            graphs,
            desc=f"Packing planar graphs ({target_dims}D)",
            leave=False,
        )
    ]

    num_graphs = len(dense_graphs)
    max_nodes = max(int(graph["num_atoms"].item()) for graph in dense_graphs)
    base_keys = {"charges", "positions", "adj", "edge_attr", "num_atoms"}
    metric_keys = [key for key in dense_graphs[0].keys() if key not in base_keys]

    charges = torch.zeros((num_graphs, max_nodes), dtype=torch.long)
    positions = torch.zeros((num_graphs, max_nodes, target_dims), dtype=torch.float32)
    adj = torch.zeros((num_graphs, max_nodes, max_nodes), dtype=torch.long)
    edge_attr = torch.zeros((num_graphs, max_nodes, max_nodes, 2), dtype=torch.float32)
    num_atoms = torch.zeros((num_graphs,), dtype=torch.long)
    metric_tensors = {metric_name: torch.zeros((num_graphs,), dtype=torch.float32) for metric_name in metric_keys}

    for idx, graph in enumerate(dense_graphs):
        n = int(graph["num_atoms"].item())
        charges[idx, :n] = graph["charges"]
        positions[idx, :n, :] = graph["positions"]
        adj[idx, :n, :n] = graph["adj"]
        edge_attr[idx, :n, :n, :] = graph["edge_attr"]
        num_atoms[idx] = graph["num_atoms"]
        for metric_name in metric_keys:
            metric_tensors[metric_name][idx] = graph[metric_name]

    packed = {
        "charges": charges,
        "positions": positions,
        "adj": adj,
        "edge_attr": edge_attr,
        "num_atoms": num_atoms,
    }

    packed.update(metric_tensors)
    return packed


def _build_processed_datasets(split_graphs: Dict[str, List[Any]], target_dims: int) -> Dict[str, ProcessedDataset]:
    included_species = torch.tensor([1], dtype=torch.long)

    datasets: Dict[str, ProcessedDataset] = {}
    for split, graphs in split_graphs.items():
        if len(graphs) == 0:
            raise ValueError(f"Planar {split} split is empty.")
        packed = _pack_graphs(graphs, target_dims=target_dims)
        datasets[split] = ProcessedDataset(
            packed,
            included_species=included_species,
            num_pts=-1,
            normalize=True,
            shuffle=(split == "train"),
            subtract_thermo=False,
        )

    return datasets


def retrieve_planar_dataloaders(cfg, distributed: bool = False):
    target_dims = int(getattr(cfg, "n_dims", 3))
    if target_dims < 2:
        raise ValueError("Planar data requires n_dims >= 2.")

    split_graphs = _resolve_graph_splits(cfg.datadir)
    datasets = _build_processed_datasets(split_graphs, target_dims=target_dims)

    preprocess = PreprocessQM9(load_charges=cfg.include_charges)
    batch_size = cfg.batch_size
    num_workers = cfg.num_workers

    dataloaders: Dict[str, DataLoader] = {}
    train_sampler = None

    for split, split_dataset in datasets.items():
        if distributed and cfg.dp and torch.cuda.device_count() > 1:
            sampler = DistributedSampler(split_dataset)
            if split == "train":
                train_sampler = sampler
            dataloaders[split] = DataLoader(
                split_dataset,
                batch_size=batch_size,
                sampler=sampler,
                num_workers=num_workers,
                collate_fn=preprocess.collate_fn,
            )
        else:
            dataloaders[split] = DataLoader(
                split_dataset,
                batch_size=batch_size,
                shuffle=(split == "train"),
                num_workers=num_workers,
                collate_fn=preprocess.collate_fn,
            )

    charge_scale = torch.tensor(1.0)
    return dataloaders, charge_scale, train_sampler

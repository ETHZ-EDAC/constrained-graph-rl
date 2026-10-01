"""Unified test driver for graph and planar metrics."""

from __future__ import annotations

# Standard library
import torch
from torch_geometric.utils import dense_to_sparse
from graph_rl.ppo.metrics import visualize_metrics, GraphMetrics


def _complete_graph(n: int) -> torch.Tensor:
    """Build a complete graph on ``n`` vertices."""
    adj = torch.ones((n, n), dtype=torch.float64)
    adj.fill_diagonal_(0.0)
    return adj


def _cycle_graph(n: int) -> torch.Tensor:
    """Build a cycle graph on ``n`` vertices."""
    adj = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n):
        adj[i, (i + 1) % n] = 1.0
        adj[(i + 1) % n, i] = 1.0
    return adj


def _star_graph(n: int) -> torch.Tensor:
    """Build a star graph with ``n`` vertices."""
    adj = torch.zeros((n, n), dtype=torch.float64)
    adj[0, 1:] = 1.0
    adj[1:, 0] = 1.0
    return adj


def _path_graph(n: int) -> torch.Tensor:
    """Build a path graph on ``n`` vertices."""
    adj = torch.zeros((n, n), dtype=torch.float64)
    for i in range(n - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0
    return adj


def _random_graph(n: int, p: float, seed: int = 42) -> torch.Tensor:
    """Erdős-Rényi random graph sampled with probability ``p``."""
    rng = torch.Generator().manual_seed(seed)
    mask = torch.rand((n, n), generator=rng) < p
    mask = torch.triu(mask, diagonal=1)
    adj = mask.float()
    adj = adj + adj.T
    return adj


def _barbell_graph(n: int) -> torch.Tensor:
    """Barbell graph composed of two cliques linked by a path."""
    n = max(n, 6)
    k = n // 3
    adj = torch.zeros((n, n), dtype=torch.float64)

    for i in range(k):
        for j in range(k):
            if i != j:
                adj[i, j] = 1.0

    for i in range(n - k, n):
        for j in range(n - k, n):
            if i != j:
                adj[i, j] = 1.0

    mid_start = k
    mid_end = n - k
    for i in range(mid_start, mid_end - 1):
        adj[i, i + 1] = 1.0
        adj[i + 1, i] = 1.0

    adj[k - 1, mid_start] = 1.0
    adj[mid_start, k - 1] = 1.0
    adj[mid_end - 1, n - k] = 1.0
    adj[n - k, mid_end - 1] = 1.0
    return adj


def run_synthetic_graph_metrics() -> dict[str, dict[str, float]]:
    """Run metrics on a suite of synthetic graphs."""
    print("=" * 70)
    print("Synthetic Graph Metric Checks")
    print("=" * 70)

    metrics_calc = GraphMetrics()
    graphs = {
        "Complete (n=5)": _complete_graph(5),
        "Cycle (n=6)": _cycle_graph(6),
        "Star (n=7)": _star_graph(7),
        "Path (n=8)": _path_graph(8),
        "Random (n=10, p=0.3)": _random_graph(10, 0.3),
        "Barbell (n=12)": _barbell_graph(12),
    }

    results: dict[str, dict[str, float]] = {}
    for name, adj in graphs.items():
        print(f"\n{name}")
        print("-" * 70)
        edge_index, _ = dense_to_sparse(adj)
        metrics = metrics_calc.compute_all_metrics(edge_index, adjacency_matrix=adj)
        results[name] = metrics
        for metric_name, value in metrics.items():
            print(f"  {metric_name:30s}: {value:10.4f}")

    test_adj = _complete_graph(6)
    edge_index, _ = dense_to_sparse(test_adj)
    print("\nDetailed metric breakdown (Complete graph n=6)")
    print(f"  Spectral Gap: {metrics_calc.spectral_gap(test_adj):.4f}")
    print(f"  Spectral Norm: {metrics_calc.spectral_norm(test_adj):.4f}")
    print(f"  Assortativity: {metrics_calc.assortativity(edge_index=edge_index):.4f}")
    gini_value = metrics_calc._gini_from_degrees(test_adj)
    print(f"  Gini Coefficient: {gini_value:.4f}")
    triangles = metrics_calc.triangle_count(test_adj)
    triplets = metrics_calc.triplet_count(test_adj)
    clustering = metrics_calc.clustering_coefficient(triangles, triplets)
    print(f"  Triangles: {triangles:.4f}")
    print(f"  Triplets: {triplets:.4f}")
    print(f"  Global Clustering: {clustering:.4f}")
    return results


def main() -> None:
    """Entry point for manual execution."""
    print("Unified Graph Metric Test Harness\n")
    synthetic_results = run_synthetic_graph_metrics()
    try:
        visualize_metrics(synthetic_results)
    except Exception as exc:  # pragma: no cover - plotting can fail in headless setups
        print(f"Visualization failed: {exc}")
    print("\n" + "=" * 70)
    print("Metric tests completed")
    print("=" * 70)


if __name__ == "__main__":
    main()

from models.GDSS.utils.loader import load_data, load_seed, load_eval_settings
from models.GDSS.evaluation.stats import eval_graph_list
from models.GDSS.evaluation.stats import orca
from models.GDSS.utils.mol_utils import load_smiles, canonicalize_smiles, mols_to_nx, smiles_to_mols
from project_bisection import satisfies
import networkx as nx
from scipy.linalg import eigvalsh
import numpy as np
import matplotlib.pyplot as plt
import json
from typing import Dict
from pathlib import Path

import os
import sys
import torch
import pickle

from evals.filter_constr import filtermap_constrained_graphs, filtermap_constrained_smiles

PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from graph_rl.utils.benchmark_graphs import ensure_planarity_metric, metric_reward, prepare_graph_for_benchmark_evaluation
from graph_rl.utils.graph_helpers import networkx_to_pyg_data

# Import plot_graph from prodigy (lazy import to handle working directory changes)
def _get_plot_graph():
    """Get plot_graph function, handling different working directories."""
    try:
        # Try importing directly first (when evals is a package)
        from prodigy import plot_graph
        return plot_graph
    except ImportError:
        # Fallback: add parent directory to path and try again
        import sys
        parent_dir = os.path.join(os.path.dirname(__file__), '..')
        if parent_dir not in sys.path:
            sys.path.insert(0, parent_dir)
        from prodigy import plot_graph
        return plot_graph

TOTAL_MOLS = 10000
MIN_CONSTR_P = 0


def _is_planar_graph(graph):
    try:
        return nx.is_connected(graph) and nx.check_planarity(graph)[0]
    except Exception:
        return False


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
    if len(values) == 0:
        return 0.0
    return float(np.mean(values))


def _safe_std(values):
    if len(values) == 0:
        return 0.0
    return float(np.std(values))


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

    with open(out_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines) + "\n")


def _compute_single_graph_metrics(graph):
    graph = _largest_connected_component(graph)
    num_nodes = float(graph.number_of_nodes())
    num_edges = float(graph.number_of_edges())
    avg_degree = float((2.0 * num_edges / num_nodes) if num_nodes > 0 else 0.0)

    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(graph).todense())
        eigs = sorted(float(v) for v in eigs)
        spectral_gap = float(eigs[1] - eigs[0]) if len(eigs) > 1 else 0.0
        spectral_radius = float(eigs[-1]) if len(eigs) > 0 else 0.0
    except Exception:
        spectral_gap = 0.0
        spectral_radius = 0.0

    try:
        clustering_coefficient = float(nx.average_clustering(graph))
    except Exception:
        clustering_coefficient = 0.0

    try:
        triangle_dict = nx.triangles(graph)
        if isinstance(triangle_dict, dict):
            triangle_count = float(sum(triangle_dict.values()) // 3)
        else:
            triangle_count = float(triangle_dict)
    except Exception:
        triangle_count = 0.0

    try:
        degrees = [d for _, d in graph.degree()]
        if len(degrees) == 0 or sum(degrees) == 0:
            gini_coefficient = 0.0
        else:
            deg_sorted = sorted(float(d) for d in degrees)
            deg_n = len(deg_sorted)
            gini = (2.0 * sum((i + 1) * d for i, d in enumerate(deg_sorted))) / (deg_n * sum(deg_sorted))
            gini -= (deg_n + 1) / deg_n
            gini_coefficient = float(gini)
    except Exception:
        gini_coefficient = 0.0

    planar_valid = 1.0 if _is_planar_graph(graph) else 0.0

    try:
        orbit_counts = orca(graph)
        orbit_per_graph = orbit_counts.sum(axis=0) / max(1, graph.number_of_nodes())
        orbit_mass = float(orbit_per_graph.sum())
    except Exception:
        orbit_mass = 0.0

    return {
        'num_nodes': num_nodes,
        'num_edges': num_edges,
        'avg_degree': avg_degree,
        'spectral_gap': spectral_gap,
        'spectral_radius': spectral_radius,
        'clustering_coefficient': clustering_coefficient,
        'triangle_count': triangle_count,
        'gini_coefficient': gini_coefficient,
        'planar_valid': planar_valid,
        'orbit_mass': orbit_mass,
    }


def _compute_graphrl_candidate(graph, *, seed: int = 42):
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
    if 'triangles' in metrics and 'triangle_count' not in metrics:
        metrics['triangle_count'] = float(metrics['triangles'])
    if 'triangle_count' in metrics and 'triangles' not in metrics:
        metrics['triangles'] = float(metrics['triangle_count'])
    metrics['is_planar'] = float(planarity_score)
    return graph, prepared, metrics, constraint_report


def _default_target_metrics(reference_graphs):
    if len(reference_graphs) == 0:
        return {
            'num_nodes': 0.0,
            'spectral_gap': 0.0,
            'gini_coefficient': 0.0,
            'clustering_coefficient': 0.0,
        }
    ref_metrics = [_compute_single_graph_metrics(graph) for graph in reference_graphs]
    keys = ['num_nodes', 'spectral_gap', 'gini_coefficient', 'clustering_coefficient']
    targets: Dict[str, float] = {}
    for key in keys:
        targets[key] = float(np.mean([m[key] for m in ref_metrics]))
    return targets


def _node_count_rank(num_nodes, target_num_nodes, *, window_size=10.0):
    """Bucket node-count preference around the target in fixed-width windows."""
    if target_num_nodes is None:
        return (0, 0.0)
    distance = abs(float(num_nodes) - float(target_num_nodes))
    if distance <= float(window_size):
        bucket = 0
    else:
        bucket = 1 + int((distance - float(window_size)) // float(window_size))
    return (bucket, distance)


def _candidate_rank_tuple(metrics, reward, target_num_nodes):
    """Lexicographic ranking: planarity, node-count bucket, reward."""
    node_bucket, node_distance = _node_count_rank(
        metrics.get('num_nodes', 0.0),
        target_num_nodes,
    )
    is_planar = 1 if float(metrics.get('is_planar', 0.0)) >= 0.5 else 0
    safe_reward = float(reward) if reward is not None else -float('inf')
    return (
        is_planar,
        -node_bucket,
        -node_distance,
        safe_reward,
    )


def _select_best_graph(gen_graph_list, targets, require_planar=False, metric_reward_weight=1.0):
    if len(gen_graph_list) == 0:
        return None, None, None, None, None, None, None

    best_idx = None
    best_graph = None
    best_metrics = None
    best_breakdown = {}
    best_reward = -float('inf')
    best_data = None
    best_constraint_report = None
    best_rank = None
    target_num_nodes = targets.get('num_nodes', None) if hasattr(targets, 'get') else None

    for idx, graph in enumerate(gen_graph_list):
        try:
            component_graph, prepared_data, metrics, constraint_report = _compute_graphrl_candidate(graph, seed=42 + idx)
        except Exception:
            continue
        if require_planar and metrics.get('is_planar', 0.0) < 0.5:
            continue
        reward_targets = {
            str(k): float(v)
            for k, v in targets.items()
            if v is not None and str(k) != 'num_nodes'
        }
        reward, breakdown = metric_reward(
            target_metrics=reward_targets,
            candidate_metrics=metrics,
            metric_reward_weight=float(metric_reward_weight),
        )
        rank = _candidate_rank_tuple(metrics, reward, target_num_nodes)
        # Rank by planarity first, then node-count windows around the target,
        # then reward. Constraints remain reporting-only.
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


def _save_best_graph_artifacts(graph, metrics, targets, score, score_breakdown, out_dir, stem, tag='best_graph', prepared_data=None, constraint_report=None):
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, f"{stem}_{tag}.png")
    json_path = os.path.join(out_dir, f"{stem}_{tag}_stats.json")

    if prepared_data is not None and getattr(prepared_data, 'pos', None) is not None:
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
    
    # Use plot_graph function with save path - get it lazily to handle imports
    plot_graph = _get_plot_graph()
    fig, ax = plot_graph(
        adj,
        pos_3d,
        pretty=True,
        save_path=Path(img_path),
        save_dpi=200,
        save_pdf=True,
        show_axes=False,
    )
    plt.close('all')

    payload = {
        'targets': {k: float(v) for k, v in targets.items()},
        'score': float(score),
        'score_breakdown': {k: float(v) for k, v in score_breakdown.items()},
        'metrics': {k: float(v) for k, v in metrics.items()},
    }
    if constraint_report is not None:
        payload['constraint_report'] = constraint_report
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    return img_path, json_path


def _compute_generation_metrics(gen_graph_list, train_graph_list):
    node_counts = []
    edge_counts = []
    avg_degrees = []
    spectral_gaps = []
    spectral_radii = []
    clustering_coeffs = []
    triangle_counts = []
    gini_coeffs = []
    planar_flags = []
    orbit_masses = []

    for graph in gen_graph_list:
        num_nodes = graph.number_of_nodes()
        num_edges = graph.number_of_edges()
        node_counts.append(float(num_nodes))
        edge_counts.append(float(num_edges))
        avg_degrees.append(float((2.0 * num_edges / num_nodes) if num_nodes > 0 else 0.0))

        try:
            eigs = eigvalsh(nx.normalized_laplacian_matrix(graph).todense())
            eigs = sorted(float(v) for v in eigs)
            spectral_gaps.append(float(eigs[1] - eigs[0]) if len(eigs) > 1 else 0.0)
            spectral_radii.append(float(eigs[-1]) if len(eigs) > 0 else 0.0)
        except Exception:
            spectral_gaps.append(0.0)
            spectral_radii.append(0.0)

        try:
            clustering_coeffs.append(float(nx.average_clustering(graph)))
        except Exception:
            clustering_coeffs.append(0.0)

        try:
            triangle_dict = nx.triangles(graph)
            if isinstance(triangle_dict, dict):
                triangle_counts.append(float(sum(triangle_dict.values()) // 3))
            else:
                triangle_counts.append(float(triangle_dict))
        except Exception:
            triangle_counts.append(0.0)

        try:
            degrees = [d for _, d in graph.degree()]
            if len(degrees) == 0 or sum(degrees) == 0:
                gini_coeffs.append(0.0)
            else:
                deg_sorted = sorted(float(d) for d in degrees)
                deg_n = len(deg_sorted)
                gini = (2.0 * sum((i + 1) * d for i, d in enumerate(deg_sorted))) / (deg_n * sum(deg_sorted))
                gini -= (deg_n + 1) / deg_n
                gini_coeffs.append(float(gini))
        except Exception:
            gini_coeffs.append(0.0)

        planar_flags.append(1.0 if _is_planar_graph(graph) else 0.0)

        try:
            orbit_counts = orca(graph)
            orbit_per_graph = orbit_counts.sum(axis=0) / max(1, graph.number_of_nodes())
            orbit_masses.append(float(orbit_per_graph.sum()))
        except Exception:
            orbit_masses.append(0.0)

    return {
        'planar_acc': _safe_mean(planar_flags),
        'num_nodes_mean': _safe_mean(node_counts),
        'num_nodes_std': _safe_std(node_counts),
        'num_edges_mean': _safe_mean(edge_counts),
        'num_edges_std': _safe_std(edge_counts),
        'avg_degree_mean': _safe_mean(avg_degrees),
        'avg_degree_std': _safe_std(avg_degrees),
        'spectral_gap_mean': _safe_mean(spectral_gaps),
        'spectral_gap_std': _safe_std(spectral_gaps),
        'spectral_radius_mean': _safe_mean(spectral_radii),
        'spectral_radius_std': _safe_std(spectral_radii),
        'clustering_coefficient_mean': _safe_mean(clustering_coeffs),
        'clustering_coefficient_std': _safe_std(clustering_coeffs),
        'triangle_count_mean': _safe_mean(triangle_counts),
        'triangle_count_std': _safe_std(triangle_counts),
        'gini_coefficient_mean': _safe_mean(gini_coeffs),
        'gini_coefficient_std': _safe_std(gini_coeffs),
        'orbit_mass_mean': _safe_mean(orbit_masses),
        'orbit_mass_std': _safe_std(orbit_masses),
    }

def evaluate (gen_graph_list, configt, config, constr_config, device='cpu'):
    train_graph_list, test_graph_list = load_data(configt, get_graph_list=True)
    methods, kernels = load_eval_settings(config.data.data)
    test_constr = filtermap_constrained_graphs (test_graph_list, configt, constr_config=constr_config)
    if test_constr.sum() < MIN_CONSTR_P * len(test_constr)/100:
        return {}
    test_graph_list = [graph for constr, graph in zip(test_constr, test_graph_list) if constr]
    result_dict = eval_graph_list(test_graph_list, gen_graph_list, methods=methods, kernels=kernels)
    adjs = torch.zeros(len(gen_graph_list), configt.data.max_node_num, configt.data.max_node_num)
    for i, G in enumerate(gen_graph_list):
        nG = G.number_of_nodes()
        adjs[i, :nG, :nG] = torch.tensor(nx.adjacency_matrix(G).todense())
    xs = torch.zeros (len(gen_graph_list), configt.data.max_node_num, configt.data.max_feat_num)
    constr_val = satisfies(xs, adjs, constr_config).sum().item()/len(adjs)
    result_dict['constr_val'] = constr_val
    result_dict.update(_compute_generation_metrics(gen_graph_list, train_graph_list))

    target_metrics = constr_config.get('target_metrics', None) if hasattr(constr_config, 'get') else None
    if target_metrics is None:
        target_metrics = _default_target_metrics(test_graph_list)
    save_dir = os.path.join('samples', 'best_graph_eval')
    stem = f"{config.data.data}_{config.ckpt}"
    best_graph_summary = None
    best_idx, best_graph_component, best_metrics, best_score, best_breakdown, best_data, best_constraint_report = _select_best_graph(
        gen_graph_list,
        target_metrics,
    )
    if best_idx is not None:
        best_img_path, best_json_path = _save_best_graph_artifacts(
            best_graph_component,
            best_metrics,
            target_metrics,
            best_score,
            best_breakdown,
            save_dir,
            stem,
            tag='best_graph',
            prepared_data=best_data,
            constraint_report=best_constraint_report,
        )
        result_dict['best_graph_index'] = int(best_idx)
        result_dict['best_graph_score'] = float(best_score if best_score is not None else 0.0)
        result_dict['best_graph_plot_path'] = best_img_path
        result_dict['best_graph_stats_path'] = best_json_path
        for key, val in (best_metrics or {}).items():
            result_dict[f'best_{key}'] = float(val)
        best_graph_summary = {
            'index': int(best_idx),
            'reward': float(best_score if best_score is not None else 0.0),
            'metrics': best_metrics or {},
            'constraint_report': best_constraint_report or {},
            'plot_path': best_img_path,
            'stats_path': best_json_path,
        }

    best_planar_graph_summary = None
    best_planar_idx, best_planar_graph_component, best_planar_metrics, best_planar_score, best_planar_breakdown, best_planar_data, best_planar_constraint_report = _select_best_graph(
        gen_graph_list,
        target_metrics,
        require_planar=True,
    )
    if best_planar_idx is not None:
        planar_img_path, planar_json_path = _save_best_graph_artifacts(
            best_planar_graph_component,
            best_planar_metrics,
            target_metrics,
            best_planar_score,
            best_planar_breakdown,
            save_dir,
            stem,
            tag='best_planar_graph',
            prepared_data=best_planar_data,
            constraint_report=best_planar_constraint_report,
        )
        result_dict['best_planar_graph_index'] = int(best_planar_idx)
        result_dict['best_planar_graph_score'] = float(best_planar_score if best_planar_score is not None else 0.0)
        result_dict['best_planar_graph_plot_path'] = planar_img_path
        result_dict['best_planar_graph_stats_path'] = planar_json_path
        for key, val in (best_planar_metrics or {}).items():
            result_dict[f'best_planar_{key}'] = float(val)
        best_planar_graph_summary = {
            'index': int(best_planar_idx),
            'reward': float(best_planar_score if best_planar_score is not None else 0.0),
            'metrics': best_planar_metrics or {},
            'constraint_report': best_planar_constraint_report or {},
            'plot_path': planar_img_path,
            'stats_path': planar_json_path,
        }
    else:
        result_dict['best_planar_graph_index'] = None
        result_dict['best_planar_graph_score'] = None
        result_dict['best_planar_graph_plot_path'] = None
        result_dict['best_planar_graph_stats_path'] = None

    _write_benchmark_summary(
        os.path.join(save_dir, f"{stem}_benchmark_summary.txt"),
        target_metrics=target_metrics,
        best_graph=best_graph_summary,
        best_planar_graph=best_planar_graph_summary,
    )
    return result_dict

def evaluate_mol (gen_smiles, configt, config, constr_config, device='cpu'):
    try:
        from moses.metrics import get_all_metrics
    except ImportError:
        try:
            from moses import get_all_metrics
        except ImportError:
            get_all_metrics = None
    
    if get_all_metrics is None:
        print("Warning: moses.metrics not available, skipping molecule metrics")
        return {}
    load_seed(config.sample.seed)
    try:
        train_smiles, test_smiles = load_smiles(configt.data.data, file_ext='_can')
    except:
        train_smiles, test_smiles = load_smiles(configt.data.data)
        train_smiles, test_smiles = canonicalize_smiles(train_smiles), canonicalize_smiles(test_smiles)
    
    gen_mols = smiles_to_mols (gen_smiles)
    num_mols = len(gen_mols)
    gen_graph_list = mols_to_nx (gen_mols)
    
    # metrics
    with open(f'data/{configt.data.data.lower()}_test_nx.pkl', 'rb') as f:
        test_graph_list = pickle.load(f)

    test_constr = filtermap_constrained_smiles (test_smiles, configt, constr_config=constr_config)
    # print (test_constr.sum(), MIN_CONSTR_P * len(test_constr)/100)
    if test_constr.sum() < MIN_CONSTR_P * len(test_constr)/100:
        return {}
    test_smiles = [smiles for constr, smiles in zip(test_constr, test_smiles) if constr]
    test_graph_list = [graph for constr, graph in zip(test_constr, test_graph_list) if constr]
    
    result_dict = {}
    scores = get_all_metrics(gen=gen_smiles, k=len(gen_smiles), device=device, n_jobs=8, 
                             test=test_smiles, train=train_smiles)
    # scores_nspdk = eval_graph_list(test_graph_list, gen_graph_list, methods=['nspdk'])['nspdk']
    # result_dict['nspdk'] = scores_nspdk
    metrics = ['valid', f'unique@{len(gen_smiles)}', 'FCD/Test', 'Novelty']
    metric_names = ['valid', 'unique', 'fcd', 'novelty']
    for metric_name, metric in zip(metric_names, metrics):
        if metric_name == 'valid':
            result_dict[metric_name] = scores[metric] * num_mols / TOTAL_MOLS
        result_dict[metric_name] = scores[metric]
    result_dict['num_mols'] = num_mols
    adjs = torch.zeros(len(gen_graph_list), configt.data.max_node_num, configt.data.max_node_num)
    for i, G in enumerate(gen_graph_list):
        nG = G.number_of_nodes()
        adjs[i, :nG, :nG] = torch.tensor(nx.adjacency_matrix(G, weight='label').todense())
    xs = torch.zeros (num_mols, configt.data.max_node_num, configt.data.max_feat_num)
    atom_id_map = {'C': 0, 'N': 1, 'O': 2, 'F': 3, 'P': 4, 'S': 5, 'Cl': 6, 'Br': 7, 'I': 8}
    for i, G in enumerate(gen_graph_list):
        xs[i, torch.arange(len(G.nodes)), [atom_id_map[x['label']] for x in G.nodes().values()]] = 1
    constr_val = satisfies(xs, adjs, constr_config).sum().item()/len(adjs)
    result_dict['constr_val'] = constr_val
    return result_dict

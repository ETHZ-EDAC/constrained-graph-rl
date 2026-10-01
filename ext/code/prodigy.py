import argparse
import sys
import os
from project_bisection import get_new_log_name, get_new_log_folder_name, save_setting
import pickle
from project_bisection import CONSTR_CONFIG
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
from datetime import datetime


def plot_graph(
    adjacency: np.ndarray,
    vertex_positions: np.ndarray,
    ax=None,
    fig=None,
    *,
    pretty: bool = True,
    save_path: str | Path | None = None,
    save_dpi: int | None = None,
    save_pdf: bool = True,
    show_axes: bool = False,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Plots the graph.

    Parameters
    ----------
    adjacency : np.ndarray (N, N): array representing adjacency matrix, padded with zero-rows/cols.
    vertex_positions : np.ndarray (N, 3): array of [x, y, z] coordinates, padded.
    pretty : bool: If True (default), use a cleaner, colorful, undirected style.
    save_path : str | Path | None: Optional path to save the figure.
    save_dpi : int | None: Optional DPI override for saving.
    save_pdf : bool: If True (default), also save a PDF alongside save_path.
    show_axes : bool: If True, show axes labels and grid. Default is False.

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

    def _render(ax, *, show_grid: bool, show_axes_flag: bool) -> None:
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

            if show_axes_flag:
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

            if show_axes_flag:
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

    _render(ax, show_grid=True, show_axes_flag=show_axes)

    fig_no_axes = plt.figure(figsize=(6, 6), dpi=199 if not pretty else 150)
    ax_no_axes = fig_no_axes.add_subplot(111)
    _render(ax_no_axes, show_grid=False, show_axes_flag=False)

    if save_path is not None:
        save_path = Path(save_path)
        dpi = save_dpi if save_dpi is not None else (300 if pretty else 199)
        
        # When show_axes is False (default), save the no-axes version
        if show_axes:
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
            if save_pdf:
                pdf_path = save_path.with_suffix(".pdf")
                fig.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
        else:
            # Use the cleaner no-axes version as the primary output
            fig_no_axes.savefig(save_path, dpi=dpi, bbox_inches="tight")
            if save_pdf:
                pdf_path = save_path.with_suffix(".pdf")
                fig_no_axes.savefig(pdf_path, dpi=dpi, bbox_inches="tight")
        
        no_axes_path = save_path.with_name(f"{save_path.stem}_no_axes{save_path.suffix}")
        fig_no_axes.savefig(no_axes_path, dpi=dpi, bbox_inches="tight")
        if save_pdf:
            pdf_no_axes_path = no_axes_path.with_suffix(".pdf")
            fig_no_axes.savefig(pdf_no_axes_path, dpi=dpi, bbox_inches="tight")
    
    # Return the appropriate figure based on show_axes setting, close the other
    if show_axes:
        plt.close(fig_no_axes)
        return fig, ax
    else:
        plt.close(fig)
        return fig_no_axes, ax_no_axes


def plot_sample_graph_prodigy(samples: list, sample_idx: int, output_path: Path, save_pdf: bool = True):
    """Plot sampled graph with fancy formatting."""
    from models.GDSS.utils.graph_utils import graphs2nxgraphs
    
    atom_types, edge_types = samples[sample_idx]
    num_nodes = int(len(atom_types))
    adjacency = np.asarray(edge_types.cpu().float() if hasattr(edge_types, 'cpu') else edge_types)
    
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))

    for i in range(num_nodes):
        for j in range(i + 1, num_nodes):
            if float(adjacency[i, j]) > 0:
                graph.add_edge(i, j)
    
    pos_2d = nx.spring_layout(graph, seed=42, k=0.5, iterations=50)
    vertex_positions = np.zeros((num_nodes, 3))
    for i in range(num_nodes):
        if i in pos_2d:
            vertex_positions[i, :2] = pos_2d[i]
        else:
            vertex_positions[i] = 0.0
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    fig, ax = plot_graph(
        adjacency,
        vertex_positions,
        pretty=True,
        save_path=output_path,
        save_dpi=300,
        save_pdf=save_pdf
    )
    plt.close('all')


def plot_gdss_graphs(gen_graph_list: list, result_dict: dict, output_dir: str = 'gdss_plots'):
    """Plot best GDSS graphs with fancy formatting."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Plot best overall graph
    if 'best_graph_index' in result_dict and result_dict['best_graph_index'] is not None:
        idx = int(result_dict['best_graph_index'])
        if 0 <= idx < len(gen_graph_list):
            try:
                graph = gen_graph_list[idx]
                # Convert NetworkX graph to adjacency matrix
                adj = nx.adjacency_matrix(graph).toarray().astype(np.float32)
                
                # Get positions (spring layout if not available)
                if hasattr(graph, 'pos'):
                    pos = graph.pos
                else:
                    pos_dict = nx.spring_layout(graph, seed=42, k=0.5, iterations=50)
                    num_nodes = nx.adjacency_matrix(graph).shape[0]
                    pos = np.asarray([pos_dict.get(i, np.array([0.0, 0.0])) for i in range(num_nodes)])
                
                # Pad to 3D if needed
                if pos.ndim == 1 or pos.shape[1] == 2:
                    pos_3d = np.zeros((pos.shape[0], 3), dtype=np.float32)
                    pos_3d[:, :pos.shape[1]] = pos
                    pos = pos_3d
                
                # Plot and save
                plot_path = output_path / 'best_graph_overall.png'
                fig, ax = plot_graph(
                    adj,
                    pos,
                    pretty=True,
                    save_path=plot_path,
                    save_dpi=300,
                    save_pdf=True
                )
                plt.close('all')
                
                print(f"\n[GDSS] Best graph visualizations saved:")
                print(f"  PNG: {plot_path}")
                print(f"  PDF: {plot_path.with_suffix('.pdf')}")
                print(f"  No-axes PNG: {plot_path.with_name(f'{plot_path.stem}_no_axes.png')}")
                print(f"  No-axes PDF: {plot_path.with_name(f'{plot_path.stem}_no_axes.pdf')}")
            except Exception as e:
                print(f"[GDSS] Warning: Could not plot best graph: {e}")
    
    # Plot best planar graph
    if 'best_planar_graph_index' in result_dict and result_dict['best_planar_graph_index'] is not None:
        idx = int(result_dict['best_planar_graph_index'])
        if 0 <= idx < len(gen_graph_list):
            try:
                graph = gen_graph_list[idx]
                adj = nx.adjacency_matrix(graph).toarray().astype(np.float32)
                
                if hasattr(graph, 'pos'):
                    pos = graph.pos
                else:
                    pos_dict = nx.spring_layout(graph, seed=42, k=0.5, iterations=50)
                    num_nodes = nx.adjacency_matrix(graph).shape[0]
                    pos = np.asarray([pos_dict.get(i, np.array([0.0, 0.0])) for i in range(num_nodes)])
                
                if pos.ndim == 1 or pos.shape[1] == 2:
                    pos_3d = np.zeros((pos.shape[0], 3), dtype=np.float32)
                    pos_3d[:, :pos.shape[1]] = pos
                    pos = pos_3d
                
                plot_path = output_path / 'best_graph_planar.png'
                fig, ax = plot_graph(
                    adj,
                    pos,
                    pretty=True,
                    save_path=plot_path,
                    save_dpi=300,
                    save_pdf=True
                )
                plt.close('all')
                
                print(f"\n[GDSS] Best planar graph visualizations saved:")
                print(f"  PNG: {plot_path}")
                print(f"  PDF: {plot_path.with_suffix('.pdf')}")
                print(f"  No-axes PNG: {plot_path.with_name(f'{plot_path.stem}_no_axes.png')}")
                print(f"  No-axes PDF: {plot_path.with_name(f'{plot_path.stem}_no_axes.pdf')}")
            except Exception as e:
                print(f"[GDSS] Warning: Could not plot best planar graph: {e}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='constr')
    parser.add_argument('--model', type=str, default='GDSS')
    parser.add_argument('--dataset', type=str, default='community_small')
    parser.add_argument('--constraint', type=str, default='configs/none/constraint.yaml')
    parser.add_argument('--method', type=str, default='configs/none/method.yaml')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--num_samples', type=int, default=None, help='Number of samples to generate (overrides config)')
    parser.add_argument('--comment', type=str, default='')
    parser.add_argument('--log_level', type=str, default='INFO')
    args, _ = parser.parse_known_args(sys.argv[1:])
    args.constraint = os.path.join(os.path.dirname(__file__), args.constraint)
    args.method = os.path.join(os.path.dirname(__file__), args.method)
    print (args)
    
    if args.model == 'GDSS':
        sys.path.append(os.path.join(sys.path[0], 'models/GDSS/'))
        os.chdir('models/GDSS')
        from models.GDSS.parsers.config import get_config
        from models.GDSS.sampler import Sampler, Sampler_mol

        gdss_dataset = args.dataset
        if args.dataset in {'hog_planar_40', 'hog_planar_50', 'hog_planar_60'}:
            gdss_dataset = 'hog_planar'
            print(f"[PRODIGY] Using canonical GDSS sampling config for dataset alias {args.dataset}: sample_{gdss_dataset}")

        config = get_config(f'sample_{gdss_dataset}', args.seed)
        if args.num_samples is not None:
            config.num_samples = args.num_samples
            print(f"[PRODIGY] Overriding num_samples to {args.num_samples}")
        if config.data.data in ['QM9', 'ZINC250k']:
            sampler = Sampler_mol(config)
        else:
            sampler = Sampler(config) 
        sampler.sample()
        
        from models.GDSS.utils.logger import set_log
        from models.GDSS.utils.loader import load_ckpt
        from models.GDSS.parsers.config import get_config
        ckpt_dict = load_ckpt(config, 'cpu')
        configt = ckpt_dict['config']
        log_folder_name, log_dir, _ = set_log(configt, is_train=False)
        log_name = f"{config.ckpt}-sample"
        pkl_path = f'./samples/pkl/{log_folder_name}/{log_name}.pkl'
        mol_txt_path = f'{log_dir}/{log_name}.txt'
        
        # Skip reorganization—use files directly where they were created

        # Run evaluation immediately and print summary metrics in terminal.
        try:
            from evals.evaluate_gdss import evaluate, evaluate_mol
            print("\n[PRODIGY] Running post-sampling evaluation...")
            gen_graph_list = None
            try:
                if config.data.data in ['QM9', 'ZINC250k']:
                    if os.path.exists(mol_txt_path):
                        gen_smiles = []
                        with open(mol_txt_path, 'r') as f:
                            for line in f:
                                gen_smiles.append(line.strip())
                        result_dict = evaluate_mol(gen_smiles, configt, config, CONSTR_CONFIG)
                    else:
                        result_dict = {'warning': f'Molecule sample file not found: {mol_txt_path}'}
                else:
                    if os.path.exists(pkl_path):
                        with open(pkl_path, 'rb') as f:
                            gen_graph_list = pickle.load(f)
                        result_dict = evaluate(gen_graph_list, configt, config, CONSTR_CONFIG)
                    else:
                        result_dict = {'warning': f'Sample file not found: {pkl_path}'}
            except Exception as eval_error:
                print(f"[PRODIGY] Error during evaluation: {eval_error}")
                import traceback
                traceback.print_exc()
                raise

            print("[PRODIGY] Evaluation results:")
            if isinstance(result_dict, dict):
                for key in sorted(result_dict.keys()):
                    print(f"  {key}: {result_dict[key]}")

                # Plot best graphs if available
                if gen_graph_list is not None:
                    try:
                        plot_gdss_graphs(gen_graph_list, result_dict, output_dir='gdss_plots')
                    except Exception as e:
                        print(f"[GDSS] Warning: Could not plot graphs: {e}")

                # Compact summary for quick inspection in terminal.
                if 'best_graph_score' in result_dict:
                    print("\n[PRODIGY] Best graph summary:")
                    print(f"  index: {result_dict.get('best_graph_index')}")
                    print(f"  score: {result_dict.get('best_graph_score')}")
                    print(f"  plot: {result_dict.get('best_graph_plot_path')}")
                    print(f"  stats: {result_dict.get('best_graph_stats_path')}")
                    print(f"  nodes: {result_dict.get('best_num_nodes')}")
                    print(f"  spectral_gap: {result_dict.get('best_spectral_gap')}")
                    print(f"  gini: {result_dict.get('best_gini_coefficient')}")
                    print(f"  clustering: {result_dict.get('best_clustering_coefficient')}")
                    if 'best_isoperimetric_ratio' in result_dict:
                        print(f"  isoperimetric_ratio: {result_dict.get('best_isoperimetric_ratio')}")

                if 'best_planar_graph_score' in result_dict and result_dict.get('best_planar_graph_index') is not None:
                    print("\n[PRODIGY] Best planar graph summary:")
                    print(f"  index: {result_dict.get('best_planar_graph_index')}")
                    print(f"  score: {result_dict.get('best_planar_graph_score')}")
                    print(f"  plot: {result_dict.get('best_planar_graph_plot_path')}")
                    print(f"  stats: {result_dict.get('best_planar_graph_stats_path')}")
                    print(f"  nodes: {result_dict.get('best_planar_num_nodes')}")
                    print(f"  spectral_gap: {result_dict.get('best_planar_spectral_gap')}")
                    print(f"  gini: {result_dict.get('best_planar_gini_coefficient')}")
                    print(f"  clustering: {result_dict.get('best_planar_clustering_coefficient')}")
                    if 'best_planar_isoperimetric_ratio' in result_dict:
                        print(f"  isoperimetric_ratio: {result_dict.get('best_planar_isoperimetric_ratio')}")
                elif 'best_planar_graph_score' in result_dict:
                    print("\n[PRODIGY] Best planar graph summary:")
                    print("  No planar graph found in generated set.")
            else:
                print(result_dict)
        except Exception as e:
            print(f"Warning: Post-sampling evaluation failed: {e}")
            import traceback
            traceback.print_exc()
        sys.exit(0)
        from models.DruM.DruM_2D.sampler import Sampler, Sampler_mol
        
        config = get_config(f'{args.dataset}', args.seed)
        if config.data.data in ['QM9', 'ZINC250k']:
            sampler = Sampler_mol(config)
        else:
            sampler = Sampler(config) 
        sampler.sample()

        from models.DruM.DruM_2D.utils.logger import set_log
        from models.DruM.DruM_2D.utils.loader import load_ckpt
        config = get_config(f"{args.dataset}", seed=args.seed)
        ckpt_dict = load_ckpt(config, 'cpu')
        configt = ckpt_dict['config']
        log_folder_name, log_dir, _ = set_log(configt, is_train=False)
        log_name = f"{config.ckpt}"
        new_log_folder_name = get_new_log_folder_name(log_folder_name)
        new_log_name = get_new_log_name(log_name)
        os.replace(f'./samples/pkl/{log_folder_name}/{log_name}.pkl', 
                   f'./samples/pkl/{new_log_folder_name}/{new_log_name}.pkl')
        os.replace(f'./samples/mols/{log_folder_name}/{log_name}.txt', 
                   f'./samples/mols/{new_log_folder_name}/{new_log_name}.txt')
    elif args.model == 'EDP-GNN':
        sys.path.insert(0, os.path.join(sys.path[0], 'models/GraphScoreMatching'))
        os.chdir('models/GraphScoreMatching/')
        from models.GraphScoreMatching.utils.arg_helper import get_config
        from models.GraphScoreMatching.sample import sample_main
        config_dict = get_config(args)
        sample_main(config_dict, args)
        
        from easydict import EasyDict as edict
        from models.GraphScoreMatching.utils.loading_utils import prepare_test_model
        config_dict = get_config(args)
        config = edict(config_dict)
        config.save_dir = os.path.join(config.save_dir, 'sample')
        config.model_files = []
        config.init_sigma = 'inf'
        models = prepare_test_model(config)
        file, sigma_list, model_params = models[0]
        sample_dir = os.path.join(config.save_dir, 'sample_data')
        new_sample_dir = get_new_log_folder_name(sample_dir)
        new_file = get_new_log_name(file)
        os.replace(f'./samples/pkl/{sample_dir}/{file}.pkl', 
                   f'./samples/pkl/{new_sample_dir}/{new_file}.pkl')
        os.replace(f'./samples/mols/{sample_dir}/{file}.txt', 
                   f'./samples/mols/{new_sample_dir}/{new_file}.txt')
    elif args.model == 'DiGress':
        os.chdir('models/DiGress/src')
        
        import torch
        try:
            import omegaconf
            torch.serialization.add_safe_globals([
                omegaconf.dictconfig.DictConfig,
                omegaconf.base.ContainerMetadata,
            ])
        except Exception:
            pass
        
        # Add DiGress to path
        digress_path = os.path.join(os.path.dirname(__file__), 'models/DiGress')
        sys.path.insert(0, digress_path)
        sys.path.insert(0, os.path.join(digress_path, 'src'))
        
        from src.diffusion_model_discrete import DiscreteDenoisingDiffusion
        from src.models.property_regressor import SimpleGCNRegressor
        from src.datasets.hog_with_metrics import extract_adjacency_from_edge_types
        from metrics.abstract_metrics import TrainAbstractMetricsDiscrete
        from diffusion.extra_features import DummyExtraFeatures, ExtraFeatures
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f"[DiGress] Using device: {device}")
        
        # Define use case targets
        USE_CASE_TARGETS = {
            'use_case_0': {'num_nodes': 60.0, 'triangle_count': 40.0},
            'use_case_1': {'num_nodes': 50.0, 'spectral_gap': 1.0, 'gini_coefficient': 0.1, 'clustering_coefficient': 0.2},
            'use_case_2': {'num_nodes': 50.0, 'gini_coefficient': 0.1},
            'use_case_3': {'num_nodes': 40.0, 'spectral_gap': 0.3},
        }
        
        USE_CASE_METRIC_SUBSETS = {
            'use_case_0': ['num_nodes', 'triangle_count'],
            'use_case_1': ['num_nodes', 'spectral_gap', 'gini_coefficient', 'clustering_coefficient'],
            'use_case_2': ['num_nodes', 'gini_coefficient'],
            'use_case_3': ['num_nodes', 'spectral_gap'],
        }
        
        # Default to use_case_1 if dataset not recognized
        use_case = 'use_case_1'
        if args.dataset:
            for uc_name, target_dict in USE_CASE_TARGETS.items():
                if args.dataset.lower() in uc_name.lower():
                    use_case = uc_name
                    break
        
        target_dict = USE_CASE_TARGETS.get(use_case, USE_CASE_TARGETS['use_case_1'])
        metric_subset = USE_CASE_METRIC_SUBSETS.get(use_case, USE_CASE_METRIC_SUBSETS['use_case_1'])
        
        print(f"[DiGress] Using {use_case}: {target_dict}")
        
        # Try to load model and regressor with proper path resolution
        model_ckpt = os.path.expanduser('~/projects/ICML_2025/graph-rl/ext/DiGress/outputs/2026-03-26/10-31-03-hog_planar_unconditional/checkpoints/hog_planar_unconditional/last.ckpt')
        regressor_ckpt = os.path.expanduser(f'~/projects/ICML_2025/graph-rl/ext/DiGress/DiGress/property_regressor/{use_case}/best_regressor.pt')
        
        if os.path.exists(model_ckpt) and os.path.exists(regressor_ckpt):
            # Load model configuration
            ckpt = torch.load(model_ckpt, map_location=device, weights_only=False)
            cfg = ckpt.get('hyper_parameters', {}).get('cfg', ckpt.get('cfg'))
            
            if cfg is not None:
                from datasets.hog_planar_dataset import HOGPlanarDataModule, HOGDatasetInfos
                from analysis.visualization import NonMolecularVisualization
                from analysis.spectre_utils import PlanarSamplingMetrics
                
                datamodule = HOGPlanarDataModule(cfg)
                dataset_infos = HOGDatasetInfos(datamodule, cfg)
                train_metrics = TrainAbstractMetricsDiscrete()
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
                
                model_kwargs = {
                    'dataset_infos': dataset_infos,
                    'train_metrics': train_metrics,
                    'sampling_metrics': sampling_metrics,
                    'visualization_tools': visualization_tools,
                    'extra_features': extra_features,
                    'domain_features': domain_features,
                }
                
                # Load diffusion model
                print(f"[DiGress] Loading model from {model_ckpt}...")
                model = DiscreteDenoisingDiffusion.load_from_checkpoint(model_ckpt, **model_kwargs)
                model = model.to(device)
                model.eval()
                
                # Load regressor
                print(f"[DiGress] Loading regressor from {regressor_ckpt}...")
                regressor_state = torch.load(regressor_ckpt, map_location=device, weights_only=False)
                hidden_dim = regressor_state['gcn1.lin.weight'].shape[0]
                num_metrics = regressor_state['mlp.3.weight'].shape[0]
                regressor = SimpleGCNRegressor(input_dim=1, hidden_dim=hidden_dim, num_metrics=num_metrics)
                regressor.load_state_dict(regressor_state)
                regressor = regressor.to(device)
                regressor.eval()
                
                # Create target metrics tensor
                target_metrics = torch.zeros(1, len(metric_subset))
                for i, metric_name in enumerate(metric_subset):
                    if metric_name in target_dict:
                        target_metrics[0, i] = target_dict[metric_name]
                
                # Sample with guidance
                print(f"[DiGress] Generating samples with guidance_scale={1.0}...")
                num_samples = args.num_samples if args.num_samples is not None else (args.seed if args.seed else 10)
                print(f"[DiGress] Batch size: {num_samples}")
                
                with torch.no_grad():
                    samples = model.sample_batch_with_guidance(
                        batch_id=0,
                        batch_size=num_samples,
                        keep_chain=0,
                        number_chain_steps=50,
                        save_final=num_samples,
                        regressor=regressor,
                        target_metrics=target_metrics.to(device),
                        guidance_scale=1.0,
                    )
                
                # Create output directory
                output_dir = Path('generated_samples') / use_case / datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Save samples and plot best ones
                print(f"[DiGress] Saving samples to {output_dir}...")
                
                # Plot best overall sample
                if len(samples) > 0:
                    best_idx = 0  # Could be improved with actual scoring
                    plot_path = output_dir / f'best_graph_overall.png'
                    plot_sample_graph_prodigy(samples, best_idx, plot_path, save_pdf=True)
                    print(f"[DiGress] Saved plots:")
                    print(f"  PNG: {plot_path}")
                    print(f"  PDF: {plot_path.with_suffix('.pdf')}")
                    print(f"  No-axes PNG: {plot_path.with_name(f'{plot_path.stem}_no_axes.png')}")
                    print(f"  No-axes PDF: {plot_path.with_name(f'{plot_path.stem}_no_axes.pdf')}")
                
                # Save sample pickle
                samples_file = output_dir / 'samples.pkl'
                with open(samples_file, 'wb') as f:
                    pickle.dump(samples, f)
                print(f"[DiGress] Saved {len(samples)} samples to {samples_file}")
            else:
                print("[DiGress] Warning: Could not extract config from checkpoint")
        else:
            print(f"[DiGress] Warning: Checkpoints not found")
            if not os.path.exists(model_ckpt):
                print(f"  Model: {model_ckpt}")
            if not os.path.exists(regressor_ckpt):
                print(f"  Regressor: {regressor_ckpt}")

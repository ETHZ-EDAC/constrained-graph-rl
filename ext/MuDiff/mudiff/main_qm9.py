# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass
import copy
import utils
import argparse
import wandb
import json
import sys
from configs.datasets_config import get_dataset_info
import os
from os.path import join, exists
from qm9 import dataset
from qm9.models import get_optim, get_model, get_prop_dist
from equivariant_diffusion.utils import assert_correctly_masked
from equivariant_diffusion import utils as flow_utils
import torch
import time
import pickle
from typing import Any, Dict, Optional
import numpy as np
import jax.numpy as jnp
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
from types import SimpleNamespace
from qm9.utils import prepare_context, compute_mean_mad
from qm9.conditioning_config import resolve_conditioning_arguments
from qm9.planar_metrics_adapter import load_conditioning_targets_from_use_case, build_conditioning_context_from_targets, compute_graph_metric_dict
from train_test import train_epoch, test, analyze_and_save, evaluate_generated_conditioning_metrics, _build_target_metric_vector, _score_graph_metrics, _compute_graph_metrics_and_constraints, _candidate_rank_tuple, _extract_largest_connected_component_tensors
from qm9.sampling import sample as sample_graph_batch

from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist

# Import plotting function
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../../src'))
try:
    from graph_rl.utils.plotting import plot_graph
except ImportError:
    plot_graph = None

parser = argparse.ArgumentParser(description='TransformerDiffusion')
parser.add_argument('--exp_name', type=str, default='debug_10')
parser.add_argument('--probabilistic_model', type=str, default='diffusion',
                    help='diffusion')

# Training complexity is O(1) (unaffected), but sampling complexity is O(steps).
parser.add_argument('--diffusion_steps', type=int, default=500)
parser.add_argument('--diffusion_noise_schedule', type=str, default='polynomial_2',
                    help='learned, cosine')
parser.add_argument('--diffusion_noise_precision', type=float, default=1e-5,
                    )
parser.add_argument('--diffusion_loss_type', type=str, default='l2',
                    help='vlb, l2')

parser.add_argument('--n_epochs', type=int, default=200)
parser.add_argument('--batch_size', type=int, default=128)
parser.add_argument('--lr', type=float, default=2e-4)
parser.add_argument('--weight_decay', type=float, default=5e-4)
parser.add_argument('--brute_force', type=eval, default=False,
                    help='True | False')
parser.add_argument('--actnorm', type=eval, default=True,
                    help='True | False')
parser.add_argument('--break_train_epoch', type=eval, default=False,
                    help='True | False')
parser.add_argument('--dp', type=eval, default=True,
                    help='True | False')
parser.add_argument('--condition_time', type=eval, default=True,
                    help='True | False')
parser.add_argument('--clip_grad', type=eval, default=True,
                    help='True | False')
parser.add_argument('--trace', type=str, default='hutch',
                    help='hutch | exact')

# <-- EGNN args
parser.add_argument('--ode_regularization', type=float, default=1e-3)
parser.add_argument('--dataset', type=str, default='qm9',
                    help='qm9 | qm9_second_half (train only on the last 50K samples of the training dataset)')
parser.add_argument('--datadir', type=str, default='qm9/temp',
                    help='qm9 directory')
parser.add_argument('--filter_n_atoms', type=int, default=None,
                    help='When set to an integer value, QM9 will only contain molecules of that amount of atoms')
parser.add_argument('--dequantization', type=str, default='argmax_variational',
                    help='uniform | variational | argmax_variational | deterministic')
parser.add_argument('--n_report_steps', type=int, default=5)
parser.add_argument('--wandb', type=eval, default=False,
                    help='Enable Weights & Biases logging for geometric training.')
parser.add_argument('--wandb_project', type=str, default='mudiff-geometric',
                    help='W&B project name.')
parser.add_argument('--wandb_entity', type=str, default=None,
                    help='Optional W&B entity/team.')
parser.add_argument('--wandb_run_name', type=str, default=None,
                    help='Optional W&B run name. Defaults to exp_name.')
parser.add_argument('--wandb_mode', type=str, default='online',
                    help='W&B mode: online | offline | disabled')
parser.add_argument('--wandb_log_samples', type=eval, default=False,
                    help='Log sampled generated graph images to W&B at eval checkpoints.')
parser.add_argument('--wandb_num_sample_graphs', type=int, default=4,
                    help='Number of sampled graphs to log when wandb sample logging is enabled.')
parser.add_argument('--wandb_sample_every', type=int, default=1,
                    help='Log sampled graph images every N eval checkpoints.')
parser.add_argument('--wandb_log_target_conditioned_samples', type=eval, default=False,
                    help='Log target-conditioned sampled graph images using use-case target metric values.')
parser.add_argument('--wandb_target_num_sample_graphs', type=int, default=4,
                    help='Number of target-conditioned sampled graphs to log.')
parser.add_argument('--wandb_target_sample_every', type=int, default=1,
                    help='Log target-conditioned sampled graph images every N eval checkpoints.')
parser.add_argument('--wandb_target_fix_noise', type=eval, default=True,
                    help='Use fixed diffusion noise when logging target-conditioned samples for consistency.')
parser.add_argument('--wandb_log_conditioning_metrics', type=eval, default=False,
                    help='Compute and log generated-sample conditioning metrics at eval checkpoints (expensive).')
parser.add_argument('--conditioning_metric_eval_samples', type=int, default=2,
                    help='Number of generated samples used for conditioning metric evaluation when enabled.')
parser.add_argument('--track_best_graphs', type=eval, default=False,
                    help='Track and save best generated graph artifacts during eval checkpoints.')
parser.add_argument('--best_graph_eval_samples', type=int, default=16,
                    help='Number of generated samples used for best-graph ranking at eval checkpoints.')
parser.add_argument('--best_graph_save_every', type=int, default=1,
                    help='Save checkpoint-best graph artifacts every N eval checkpoints.')
parser.add_argument('--best_graph_relative_error_eps', type=float, default=1e-6,
                    help='Epsilon in relative error score |x-t|/max(|t|, eps).')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--save_model', type=eval, default=True,
                    help='save model')
parser.add_argument('--generate_epochs', type=int, default=1,
                    help='save model')
parser.add_argument('--num_workers', type=int, default=0, help='Number of worker for the dataloader')
parser.add_argument('--test_epochs', type=int, default=50)
parser.add_argument('--run_stability_eval', type=eval, default=True,
                    help='Run expensive molecule stability analysis during eval checkpoints.')
parser.add_argument("--conditioning", nargs='+', default=[],
                    help='arguments : homo | lumo | alpha | gap | mu | Cv' )
parser.add_argument('--conditioning_use_case', type=str, default=None,
                    help='Optional use-case YAML name/path. Conditioning channels are loaded from target_metrics keys only.')
parser.add_argument('--conditioning_use_case_dir', type=str, default='configs/conditioning/use_cases',
                    help='Directory used to resolve conditioning_use_case names.')
parser.add_argument('--conditioning_merge_cli', type=eval, default=True,
                    help='When True and conditioning_use_case is set, merges use-case channels with --conditioning channels.')
parser.add_argument('--resume', type=str, default=None,
                    help='')
parser.add_argument('--start_epoch', type=int, default=0,
                    help='')
parser.add_argument('--ema_decay', type=float, default=0.999,
                    help='Amount of EMA decay, 0 means off. A reasonable value'
                         ' is 0.999.')
parser.add_argument('--augment_noise', type=float, default=0)
parser.add_argument('--n_stability_samples', type=int, default=10,
                    help='Number of samples to compute the stability')
parser.add_argument('--normalize_factors', type=eval, default=[1, 4, 1],
                    help='normalize factors for [x, categorical, integer]')
parser.add_argument('--remove_h', action='store_true')
parser.add_argument('--include_charges', type=eval, default=True,
                    help='include atom charge or not')
parser.add_argument('--n_dims', type=int, default=3,
                    help='Number of coordinate dimensions to model (2 for planar, 3 for volumetric).')



# <-- Global encoding args
parser.add_argument('--multi_hop_max_dist', type=int, default=2,
                    help='')
parser.add_argument('--num_encoder_layers', type=int, default=6,
                    help='')
parser.add_argument('--embedding_dim', type=int, default=128,
                    help='')
parser.add_argument('--edge_embedding_dim', type=int, default=128,
                    help='')
parser.add_argument('--graph_embedding_dim', type=int, default=32,
                    help='')
parser.add_argument('--num_attention_heads', type=int, default=8,
                    help='')
parser.add_argument('--num_3d_bias_kernel', type=int, default=16,
                    help='')


parser.add_argument('--use_2d_embedding', type=bool, default=True, 
                    help='')
parser.add_argument('--use_3d_embedding', type=bool, default=True, 
                    help='')
parser.add_argument('--use_2d_neighbor_embedding', type=bool, default=True, 
                    help='')
parser.add_argument('--use_3d_neighbor_embedding', type=bool, default=True, 
                    help='')
parser.add_argument('--apply_concrete_adjacency_neighbor', type=bool, default=False, 
                    help='')
parser.add_argument('--use_2d_edge_embedding', type=bool, default=True, 
                    help='')
parser.add_argument('--trainable_dist_proj', type=bool, default=True, 
                    help='')
parser.add_argument('--use_extra_graph_embedding', type=bool, default=False, 
                    help='')
parser.add_argument('--use_extra_graph_embedding_attn_bias', type=bool, default=False, 
                    help='')


parser.add_argument('--cutoff_upper', type=float, default=4.0,
                    help='')
parser.add_argument('--cutoff_lower', type=float, default=0.0,
                    help='')


parser.add_argument('--use_edge_type', type=str, default='no',
                    help='no, multi_hop')
parser.add_argument('--distance_projection', type=str, default='exp',
                    help='exp, gaussian')
parser.add_argument('--neighbor_combine_embedding', type=str, default='cat',
                    help='cat, add, no')
parser.add_argument('--extra_feature_type', type=str, default='all',
                    help='all, cycles, eigenvalues')



# Transformer args
parser.add_argument('--ffn_embedding_dim', type=int, default=300,
                    help='')
parser.add_argument('--ffn_edge_embedding_dim', type=int, default=300,
                    help='')
parser.add_argument('--ffn_graph_embedding_dim', type=int, default=100,
                    help='')
parser.add_argument('--before_attention_qn_block_size', type=int, default=0,
                    help='')
parser.add_argument('--in_attention_qn_block_size', type=int, default=0,
                    help='')


parser.add_argument('--before_attention_dropout', type=float, default=0,
                    help='')
parser.add_argument('--before_attention_quant_noise', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_feature_dropout', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_dropout', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_activation_dropout', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_activation_dropout_adj', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_activation_dropout_graph_feature', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_quant_noise', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_droppath', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_droppath_adj', type=float, default=0,
                    help='')
parser.add_argument('--in_attention_droppath_graph_feature', type=float, default=0,
                    help='')


parser.add_argument('--before_attention_layernorm', type=bool, default=True, 
                    help='')
parser.add_argument('--in_attention_layernorm', type=bool, default=True, 
                    help='')
parser.add_argument('--in_attention_pred_adjacency', type=bool, default=True, 
                    help='')


parser.add_argument('--attention_activation_fn', type=str, default='silu',
                    help='silu, relu, gelu, softmax')



# Equivariant Transformer args
parser.add_argument('--use_equivariant_transformer', type=bool, default=True, 
                    help='')
parser.add_argument('--equivariant_use_x_layernorm', type=bool, default=True, 
                    help='')
parser.add_argument('--equivariant_use_dx_layernorm', type=bool, default=True, 
                    help='')
parser.add_argument('--equivariant_apply_concrete_adjacency', type=bool, default=True, 
                    help='')


parser.add_argument('--equivariant_in_attention_dropout', type=float, default=0,
                    help='')
parser.add_argument('--equivariant_dx_dropout', type=float, default=0,
                    help='')


parser.add_argument('--equivariant_attention_activation_fn', type=str, default='silu',
                    help='silu, relu, gelu, softmax')
parser.add_argument('--equivariant_distance_influence', type=str, default='both',
                    help='both, keys, values')


# Output args
parser.add_argument('--combine_transformer_output', type=str, default='cat',
                    help='cat, add')


parser.add_argument('--use_output_projection', type=bool, default=True, 
                    help='')
parser.add_argument('--use_equivariant_output_projection', type=bool, default=False, 
                    help='')

args = parser.parse_args()

if args.n_dims not in (2, 3):
    raise ValueError('--n_dims must be 2 or 3.')

if args.n_dims == 2 and args.use_3d_embedding:
    print('Disabling 3D embedding because --n_dims=2.')
    args.use_3d_embedding = False

if args.n_dims == 2 and args.use_equivariant_transformer:
    print('Disabling equivariant transformer because --n_dims=2.')
    args.use_equivariant_transformer = False
    args.use_equivariant_output_projection = False

if not hasattr(args, 'wandb'):
    args.wandb = False
if not hasattr(args, 'wandb_project'):
    args.wandb_project = 'mudiff-geometric'
if not hasattr(args, 'wandb_entity'):
    args.wandb_entity = None
if not hasattr(args, 'wandb_run_name'):
    args.wandb_run_name = None
if not hasattr(args, 'wandb_mode'):
    args.wandb_mode = 'online'
if not hasattr(args, 'wandb_log_samples'):
    args.wandb_log_samples = False
if not hasattr(args, 'wandb_num_sample_graphs'):
    args.wandb_num_sample_graphs = 4
if not hasattr(args, 'wandb_sample_every'):
    args.wandb_sample_every = 1
if not hasattr(args, 'wandb_log_target_conditioned_samples'):
    args.wandb_log_target_conditioned_samples = False
if not hasattr(args, 'wandb_target_num_sample_graphs'):
    args.wandb_target_num_sample_graphs = 4
if not hasattr(args, 'wandb_target_sample_every'):
    args.wandb_target_sample_every = 1
if not hasattr(args, 'wandb_target_fix_noise'):
    args.wandb_target_fix_noise = True
if not hasattr(args, 'wandb_log_conditioning_metrics'):
    args.wandb_log_conditioning_metrics = False
if not hasattr(args, 'conditioning_metric_eval_samples'):
    args.conditioning_metric_eval_samples = 2
if not hasattr(args, 'track_best_graphs'):
    args.track_best_graphs = False
if not hasattr(args, 'best_graph_eval_samples'):
    args.best_graph_eval_samples = 16
if not hasattr(args, 'best_graph_save_every'):
    args.best_graph_save_every = 1
if not hasattr(args, 'best_graph_relative_error_eps'):
    args.best_graph_relative_error_eps = 1e-6
if not hasattr(args, 'run_stability_eval'):
    args.run_stability_eval = True
if not hasattr(args, 'conditioning_use_case'):
    args.conditioning_use_case = None
if not hasattr(args, 'conditioning_use_case_dir'):
    args.conditioning_use_case_dir = 'configs/conditioning/use_cases'
if not hasattr(args, 'conditioning_merge_cli'):
    args.conditioning_merge_cli = True


def _apply_loaded_args_defaults(runtime_args):
    if not hasattr(runtime_args, 'n_dims'):
        runtime_args.n_dims = 3
    if not hasattr(runtime_args, 'wandb'):
        runtime_args.wandb = False
    if not hasattr(runtime_args, 'wandb_project'):
        runtime_args.wandb_project = 'mudiff-geometric'
    if not hasattr(runtime_args, 'wandb_entity'):
        runtime_args.wandb_entity = None
    if not hasattr(runtime_args, 'wandb_run_name'):
        runtime_args.wandb_run_name = None
    if not hasattr(runtime_args, 'wandb_mode'):
        runtime_args.wandb_mode = 'online'
    if not hasattr(runtime_args, 'wandb_log_samples'):
        runtime_args.wandb_log_samples = False
    if not hasattr(runtime_args, 'wandb_num_sample_graphs'):
        runtime_args.wandb_num_sample_graphs = 4
    if not hasattr(runtime_args, 'wandb_sample_every'):
        runtime_args.wandb_sample_every = 1
    if not hasattr(runtime_args, 'wandb_log_target_conditioned_samples'):
        runtime_args.wandb_log_target_conditioned_samples = False
    if not hasattr(runtime_args, 'wandb_target_num_sample_graphs'):
        runtime_args.wandb_target_num_sample_graphs = 4
    if not hasattr(runtime_args, 'wandb_target_sample_every'):
        runtime_args.wandb_target_sample_every = 1
    if not hasattr(runtime_args, 'wandb_target_fix_noise'):
        runtime_args.wandb_target_fix_noise = True
    if not hasattr(runtime_args, 'wandb_log_conditioning_metrics'):
        runtime_args.wandb_log_conditioning_metrics = False
    if not hasattr(runtime_args, 'conditioning_metric_eval_samples'):
        runtime_args.conditioning_metric_eval_samples = 2
    if not hasattr(runtime_args, 'track_best_graphs'):
        runtime_args.track_best_graphs = False
    if not hasattr(runtime_args, 'best_graph_eval_samples'):
        runtime_args.best_graph_eval_samples = 16
    if not hasattr(runtime_args, 'best_graph_save_every'):
        runtime_args.best_graph_save_every = 1
    if not hasattr(runtime_args, 'best_graph_relative_error_eps'):
        runtime_args.best_graph_relative_error_eps = 1e-6
    if not hasattr(runtime_args, 'run_stability_eval'):
        runtime_args.run_stability_eval = True
    if not hasattr(runtime_args, 'conditioning_use_case'):
        runtime_args.conditioning_use_case = None
    if not hasattr(runtime_args, 'conditioning_use_case_dir'):
        runtime_args.conditioning_use_case_dir = 'configs/conditioning/use_cases'
    if not hasattr(runtime_args, 'conditioning_merge_cli'):
        runtime_args.conditioning_merge_cli = True
    return runtime_args

bs = args.batch_size
cli_track_best_graphs = args.track_best_graphs
cli_best_graph_eval_samples = args.best_graph_eval_samples
cli_best_graph_save_every = args.best_graph_save_every
cli_best_graph_relative_error_eps = args.best_graph_relative_error_eps
cli_test_epochs = args.test_epochs
if exists(join('outputs', args.exp_name, 'args.pickle')) and (args.resume is None):
    with open(join('outputs', args.exp_name, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)
        
    args.break_train_epoch = False
    args.batch_size = bs
    args = _apply_loaded_args_defaults(args)
    args.track_best_graphs = cli_track_best_graphs
    args.best_graph_eval_samples = cli_best_graph_eval_samples
    args.best_graph_save_every = cli_best_graph_save_every
    args.best_graph_relative_error_eps = cli_best_graph_relative_error_eps
    args.test_epochs = cli_test_epochs
    print(args)


if args.resume is not None:
    exp_name = args.exp_name + '_resume'
    start_epoch = args.start_epoch
    resume = args.resume
    wandb_usr = args.wandb_usr

    with open(join(args.resume, 'args.pickle'), 'rb') as f:
        args = pickle.load(f)

    args.resume = resume
    args.break_train_epoch = False

    args.exp_name = exp_name
    args.start_epoch = start_epoch
    args.wandb_usr = wandb_usr
    args = _apply_loaded_args_defaults(args)
    args.test_epochs = cli_test_epochs

    print(args)

if args.n_dims == 2 and args.use_3d_embedding:
    print('Disabling 3D embedding because --n_dims=2.')
    args.use_3d_embedding = False

if args.n_dims == 2 and args.use_equivariant_transformer:
    print('Disabling equivariant transformer because --n_dims=2.')
    args.use_equivariant_transformer = False
    args.use_equivariant_output_projection = False

if not hasattr(args, 'conditioning_use_case'):
    args.conditioning_use_case = None
if not hasattr(args, 'conditioning_use_case_dir'):
    args.conditioning_use_case_dir = 'configs/conditioning/use_cases'
if not hasattr(args, 'conditioning_merge_cli'):
    args.conditioning_merge_cli = True

args = resolve_conditioning_arguments(args)

dataset_info = get_dataset_info(args.dataset, args.remove_h, datadir=args.datadir)

atom_encoder = dataset_info['atom_encoder']
atom_decoder = dataset_info['atom_decoder']

# args, unparsed_args = parser.parse_known_args()

args.cuda = not args.no_cuda and torch.cuda.is_available()
device = torch.device("cuda" if args.cuda else "cpu")
print('On device:', device, torch.cuda.is_available())
#device = torch.device("cuda:0")
dtype = torch.float32


utils.create_folders(args)
# print(args)



args.context_node_nf = 0


gradnorm_queue = utils.Queue()
gradnorm_queue.add(3000)  # Add large value that will be flushed.

def check_mask_correct(variables, node_mask):
    for variable in variables:
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def _log_generated_samples_to_wandb(args, model_sample, nodes_dist, dataset_info, prop_dist, device, epoch):
    if not getattr(args, 'use_wandb', False) or not getattr(args, 'wandb_log_samples', False):
        return

    if args.wandb_sample_every <= 0 or (epoch % args.wandb_sample_every) != 0:
        return

    n_samples = max(1, int(args.wandb_num_sample_graphs))
    nodesxsample = nodes_dist.sample(n_samples).to(device)

    with torch.no_grad():
        one_hot, _, x, edge, node_mask = sample_graph_batch(
            args=args,
            device=device,
            generative_model=model_sample,
            dataset_info=dataset_info,
            prop_dist=prop_dist,
            nodesxsample=nodesxsample,
            fix_noise=False,
        )

    images = []
    for idx in range(n_samples):
        mask = node_mask[idx, :, 0].detach().cpu().bool().numpy()
        pos = x[idx].detach().cpu().numpy()[mask]
        if pos.shape[0] == 0:
            continue

        node_types = one_hot[idx].argmax(dim=-1).detach().cpu().numpy()[mask]
        edge_i = edge[idx].detach().cpu().numpy()
        if edge_i.ndim == 3:
            edge_i = edge_i[..., 1] if edge_i.shape[-1] > 1 else edge_i[..., 0]
        edge_i = edge_i[np.ix_(mask, mask)]

        fig, ax = plt.subplots(figsize=(4, 4))
        for u in range(pos.shape[0]):
            for v in range(u + 1, pos.shape[0]):
                if edge_i[u, v] > 0:
                    ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]], color='lightgray', linewidth=1, zorder=1)

        scatter = ax.scatter(pos[:, 0], pos[:, 1], c=node_types, cmap='tab10', s=28, zorder=2)
        ax.set_title(f'sample_{idx}_nodes_{pos.shape[0]}')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        images.append(wandb.Image(fig, caption=f'epoch={epoch}, sample={idx}, nodes={pos.shape[0]}'))
        plt.close(fig)

    if images:
        wandb.log({'samples/generated_graphs': images})


def _log_target_conditioned_samples_to_wandb(
    args,
    model_sample,
    nodes_dist,
    dataset_info,
    device,
    epoch,
    property_norms,
    target_metrics,
):
    if not getattr(args, 'use_wandb', False) or not getattr(args, 'wandb_log_target_conditioned_samples', False):
        return

    if len(getattr(args, 'conditioning', [])) == 0:
        return

    if target_metrics is None:
        return

    if args.wandb_target_sample_every <= 0 or (epoch % args.wandb_target_sample_every) != 0:
        return

    n_samples = max(1, int(args.wandb_target_num_sample_graphs))
    target_n_nodes = target_metrics.get('num_nodes', None)
    if target_n_nodes is not None:
        clamped_nodes = int(max(1, min(dataset_info['max_n_nodes'], round(float(target_n_nodes)))))
        nodesxsample = torch.full((n_samples,), clamped_nodes, dtype=torch.long, device=device)
    else:
        nodesxsample = nodes_dist.sample(n_samples).to(device)

    context = build_conditioning_context_from_targets(
        conditioning=args.conditioning,
        target_metrics=target_metrics,
        property_norms=property_norms,
        batch_size=n_samples,
        device=device,
    )

    with torch.no_grad():
        one_hot, _, x, edge, node_mask = sample_graph_batch(
            args=args,
            device=device,
            generative_model=model_sample,
            dataset_info=dataset_info,
            prop_dist=None,
            nodesxsample=nodesxsample,
            context=context,
            fix_noise=bool(args.wandb_target_fix_noise),
        )

    images = []
    for idx in range(n_samples):
        mask = node_mask[idx, :, 0].detach().cpu().bool().numpy()
        pos = x[idx].detach().cpu().numpy()[mask]
        if pos.shape[0] == 0:
            continue

        node_types = one_hot[idx].argmax(dim=-1).detach().cpu().numpy()[mask]
        edge_i = edge[idx].detach().cpu().numpy()
        if edge_i.ndim == 3:
            edge_i = edge_i[..., 1] if edge_i.shape[-1] > 1 else edge_i[..., 0]
        edge_i = edge_i[np.ix_(mask, mask)]

        fig, ax = plt.subplots(figsize=(4, 4))
        for u in range(pos.shape[0]):
            for v in range(u + 1, pos.shape[0]):
                if edge_i[u, v] > 0:
                    ax.plot([pos[u, 0], pos[v, 0]], [pos[u, 1], pos[v, 1]], color='lightgray', linewidth=1, zorder=1)

        scatter = ax.scatter(pos[:, 0], pos[:, 1], c=node_types, cmap='tab10', s=28, zorder=2)
        ax.set_title(f'target_sample_{idx}_nodes_{pos.shape[0]}')
        ax.set_aspect('equal', adjustable='box')
        ax.set_xticks([])
        ax.set_yticks([])
        ax.figure.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04)
        images.append(wandb.Image(fig, caption=f'epoch={epoch}, target_sample={idx}, nodes={pos.shape[0]}'))
        plt.close(fig)

    if images:
        payload: Dict[str, Any] = {'samples/target_conditioned_graphs': images}
        for metric_name, metric_value in target_metrics.items():
            payload[f'samples/target_conditions/{metric_name}'] = float(metric_value)
        wandb.log(payload)


def _is_planar(adjacency_matrix: np.ndarray) -> bool:
    """Check if a graph is planar using NetworkX."""
    if adjacency_matrix.shape[0] == 0:
        return True
    try:
        # Make adjacency symmetric and binary
        adj_undirected = np.maximum(adjacency_matrix, adjacency_matrix.T)
        np.fill_diagonal(adj_undirected, 0)
        adj_undirected = (adj_undirected > 0).astype(int)
        
        # Convert to NetworkX graph
        G = nx.from_numpy_array(adj_undirected)
        
        return nx.is_planar(G)
    except Exception as e:
        print(f"Warning: planarity check failed ({e}), assuming non-planar")
        return False


def _extract_adjacency_from_best_graph(best_graph: Dict[str, Any]) -> Optional[np.ndarray]:
    """Extract full adjacency matrix from best_graph dict."""
    edge = best_graph.get('edge')
    if edge is None:
        return None
    
    if torch.is_tensor(edge):
        edge = edge.detach().cpu().numpy()
    else:
        edge = np.asarray(edge)
    
    if edge.ndim == 3:
        edge = edge[..., 1] if edge.shape[-1] > 1 else edge[..., 0]
    
    edge = (edge > 0).astype(np.int64)
    return edge


def _selection_rank_for_graph(best_graph: Optional[Dict[str, Any]]) -> tuple:
    if best_graph is None:
        return (-float('inf'), -float('inf'), -float('inf'), -float('inf'))
    rank = best_graph.get('selection_rank', None)
    if rank is not None:
        return tuple(rank)
    return (
        1.0 if float(best_graph.get('metrics', {}).get('is_planar', 0.0)) >= 0.5 else 0.0,
        0.0,
        0.0,
        float(best_graph.get('reward', -float('inf'))),
    )


def _evaluate_with_planarity_tracking(
    args,
    model_sample,
    nodes_dist,
    dataset_info,
    prop_dist,
    device,
    property_norms=None,
    n_samples=16,
    target_metrics=None,
    track_best_graph=False,
    relative_error_eps=1e-6,
):
    """
    Enhanced evaluation that tracks planarity metrics and best planar graph.
    
    Returns:
        Tuple of (logs_dict, best_graph, best_planar_graph, planarity_percent)
    """
    # Get base metrics and best graph
    base_logs, best_graph = evaluate_generated_conditioning_metrics(
        args=args,
        model_sample=model_sample,
        nodes_dist=nodes_dist,
        dataset_info=dataset_info,
        prop_dist=prop_dist,
        device=device,
        property_norms=property_norms,
        n_samples=n_samples,
        target_metrics=target_metrics,
        track_best_graph=track_best_graph,
        relative_error_eps=relative_error_eps,
    )
    
    # If no conditioning or not a planar dataset, return early
    if len(getattr(args, 'conditioning', [])) == 0:
        return base_logs, best_graph, None, 0.0
    
    if 'planar' not in str(getattr(args, 'dataset', '')).lower():
        return base_logs, best_graph, None, 0.0
    
    # Re-sample for planarity computation
    n_samples_planar = max(1, int(n_samples))
    nodesxsample = nodes_dist.sample(n_samples_planar).to(device)
    
    context = None
    sample_prop_dist = prop_dist
    if len(getattr(args, 'conditioning', [])) > 0 and target_metrics is not None:
        context = build_conditioning_context_from_targets(
            conditioning=args.conditioning,
            target_metrics=target_metrics,
            property_norms=property_norms,
            batch_size=n_samples_planar,
            device=device,
        )
        sample_prop_dist = None
    
    with torch.no_grad():
        one_hot, _, x, edge, node_mask = sample_graph_batch(
            args,
            device,
            model_sample,
            dataset_info=dataset_info,
            prop_dist=sample_prop_dist,
            nodesxsample=nodesxsample,
            context=context,
        )
    
    planar_count = 0
    best_planar_graph = None
    target_vector = _build_target_metric_vector(target_metrics)
    
    for idx in range(n_samples_planar):
        component = _extract_largest_connected_component_tensors(
            one_hot_sample=one_hot[idx],
            x_sample=x[idx],
            edge_sample=edge[idx],
            node_mask_sample=node_mask[idx],
        )
        if component is None:
            continue

        edge_idx = component['edge'].detach().cpu().numpy()
        num_nodes = int(component['num_nodes'])

        # Check planarity
        if _is_planar(edge_idx):
            planar_count += 1

            if track_best_graph:
                metric_values, constraint_report = _compute_graph_metrics_and_constraints(
                    num_nodes,
                    component['edge_index'],
                    component['x'],
                )
                if metric_values is None:
                    continue
                metric_values = {
                    str(metric_key): float(metric_value)
                    for metric_key, metric_value in metric_values.items()
                    if np.isfinite(float(metric_value))
                }
                score, score_breakdown = _score_graph_metrics(
                    metric_values=metric_values,
                    target_vector=target_vector,
                    eps=relative_error_eps,
                )

                if score is None:
                    score = -float('inf')
                    score_breakdown = {}

                selection_rank = _candidate_rank_tuple(
                    metric_values=metric_values,
                    reward=score,
                    target_vector=target_vector,
                )

                candidate_planar_graph = {
                    'sample_index': int(idx),
                    'one_hot': component['one_hot'],
                    'x': component['x'],
                    'edge': component['edge'],
                    'node_mask': component['node_mask'],
                    'targets': {k: float(v) for k, v in target_vector.items()},
                    'metrics': {k: float(v) for k, v in metric_values.items()},
                    'constraint_report': constraint_report,
                    'reward': float(score),
                    'reward_breakdown': {k: float(v) for k, v in score_breakdown.items()},
                    'score': float(score),
                    'score_breakdown': {k: float(v) for k, v in score_breakdown.items()},
                    'selection_rank': selection_rank,
                }

                if (
                    best_planar_graph is None
                    or _selection_rank_for_graph(candidate_planar_graph) > _selection_rank_for_graph(best_planar_graph)
                ):
                    best_planar_graph = candidate_planar_graph
    
    planarity_percent = 100.0 * planar_count / max(1, n_samples_planar)
    base_logs['samples/planarity_percentage'] = planarity_percent
    base_logs['samples/planar_count'] = float(planar_count)
    base_logs['samples/total_generated'] = float(n_samples_planar)
    
    return base_logs, best_graph, best_planar_graph, planarity_percent


def _plot_best_graph_with_plotting_func(
    best_graph: Dict[str, Any],
    epoch: int,
    out_dir: str,
    file_stem: str,
) -> tuple[Optional[str], Optional[str]]:
    """
    Plot best graph using the plot_graph function with PNG and PDF support.
    
    Returns:
        Tuple of (png_path, pdf_path)
    """
    if plot_graph is None:
        print("Warning: plot_graph function not available, skipping graph plotting")
        return None, None
    
    os.makedirs(out_dir, exist_ok=True)
    
    # Extract adjacency matrix
    adjacency = _extract_adjacency_from_best_graph(best_graph)
    if adjacency is None or adjacency.shape[0] == 0:
        return None, None
    
    # Extract vertex positions
    node_mask = best_graph.get('node_mask')
    if torch.is_tensor(node_mask):
        if node_mask.dim() == 2:
            mask = node_mask[:, 0] > 0.5
        else:
            mask = node_mask > 0.5
        mask_np = mask.detach().cpu().bool().numpy()
    else:
        mask_np = np.asarray(node_mask).astype(bool)
    
    x_i = best_graph.get('x')
    if torch.is_tensor(x_i):
        x_i = x_i.detach().cpu().numpy()
    else:
        x_i = np.asarray(x_i)
    
    if x_i.ndim != 2:
        return None, None
    
    pos = x_i[mask_np]
    if pos.shape[0] == 0:
        return None, None
    
    if pos.shape[1] < 2:
        pos = np.concatenate([pos, np.zeros((pos.shape[0], 2 - pos.shape[1]), dtype=pos.dtype)], axis=1)
    
    # Extract adjacency for active vertices only
    adjacency_active = adjacency[np.ix_(mask_np, mask_np)]
    
    # Save base path without extension
    png_path = join(out_dir, f'{file_stem}.png')
    
    # Convert to JAX arrays for plot_graph
    adjacency_jax = jnp.asarray(adjacency_active, dtype=jnp.float32)
    pos_jax = jnp.asarray(pos, dtype=jnp.float32)
    
    score = best_graph.get('score', None)
    score_text = f"{float(score):.4f}" if score is not None else "n/a"
    sample_index = int(best_graph.get('sample_index', -1))
    metric_items = list(best_graph.get('metrics', {}).items())
    metric_summary = ", ".join(
        f"{str(metric_key)}={float(metric_value):.3f}"
        for metric_key, metric_value in metric_items[:4]
    )

    try:
        # Keep plot_graph as the renderer and then add run-specific annotations.
        fig, ax = plot_graph(
            adjacency=adjacency_jax,
            vertex_positions=pos_jax,
            pretty=True,
            save_path=png_path,
            save_dpi=300,
            save_pdf=True,
        )

        title = f"Epoch {int(epoch)} | Nodes {adjacency_active.shape[0]} | Score {score_text}"
        if sample_index >= 0:
            title += f" | Sample {sample_index}"
        if metric_summary:
            title += f"\n{metric_summary}"
        ax.set_title(title)
        fig.tight_layout()
        fig.savefig(png_path, dpi=300, bbox_inches='tight')

        pdf_path = png_path.replace('.png', '.pdf') if '.png' in png_path else png_path + '.pdf'
        fig.savefig(pdf_path, dpi=300, bbox_inches='tight')
        plt.close(fig)

        return png_path, pdf_path
    except Exception as e:
        print(f"Error plotting graph: {e}")
        return None, None


def _save_validation_checkpoint(args, epoch, model, model_ema, optim):
    out_dir = join('outputs', args.exp_name)
    os.makedirs(out_dir, exist_ok=True)

    args.current_epoch = epoch + 1

    utils.save_model(optim, join(out_dir, 'optim_last.npy'))
    utils.save_model(model, join(out_dir, 'generative_model_last.npy'))
    utils.save_model(optim, join(out_dir, f'optim_epoch_{int(epoch):04d}.npy'))
    utils.save_model(model, join(out_dir, f'generative_model_epoch_{int(epoch):04d}.npy'))

    if args.ema_decay > 0:
        utils.save_model(model_ema, join(out_dir, 'generative_model_ema_last.npy'))
        utils.save_model(model_ema, join(out_dir, f'generative_model_ema_epoch_{int(epoch):04d}.npy'))

    with open(join(out_dir, 'args_last.pickle'), 'wb') as f:
        pickle.dump(args, f)
    with open(join(out_dir, f'args_epoch_{int(epoch):04d}.pickle'), 'wb') as f:
        pickle.dump(args, f)


def _extract_graph_arrays(best_graph):
    node_mask = best_graph['node_mask']
    if torch.is_tensor(node_mask):
        if node_mask.dim() == 2:
            mask = node_mask[:, 0] > 0.5
        else:
            mask = node_mask > 0.5
        mask_np = mask.detach().cpu().bool().numpy()
    else:
        mask_np = np.asarray(node_mask).astype(bool)

    x_i = best_graph['x']
    if torch.is_tensor(x_i):
        x_i = x_i.detach().cpu().numpy()
    else:
        x_i = np.asarray(x_i)

    if x_i.ndim != 2:
        return None, None, None

    pos = x_i[mask_np]
    if pos.shape[0] == 0:
        return None, None, None

    if pos.shape[1] < 2:
        pos = np.concatenate([pos, np.zeros((pos.shape[0], 2 - pos.shape[1]), dtype=pos.dtype)], axis=1)

    one_hot_i = best_graph['one_hot']
    if torch.is_tensor(one_hot_i):
        one_hot_i = one_hot_i.detach().cpu().numpy()
    else:
        one_hot_i = np.asarray(one_hot_i)

    if one_hot_i.ndim == 2 and one_hot_i.shape[1] > 0:
        node_types = np.argmax(one_hot_i, axis=-1)[mask_np]
    else:
        node_types = np.zeros(pos.shape[0], dtype=np.int64)

    edge_i = best_graph['edge']
    if torch.is_tensor(edge_i):
        edge_i = edge_i.detach().cpu().numpy()
    else:
        edge_i = np.asarray(edge_i)

    if edge_i.ndim == 3:
        edge_i = edge_i[..., 1] if edge_i.shape[-1] > 1 else edge_i[..., 0]
    edge_i = edge_i[np.ix_(mask_np, mask_np)]
    edge_i = (edge_i > 0).astype(np.int64)

    return pos, node_types, edge_i


def _save_best_graph_artifacts(args, best_graph, epoch, file_stem):
    """
    Save best graph artifacts using the plot_graph function (supports PNG and PDF).
    
    Returns:
        Tuple of (png_path, json_path)
    """
    out_dir = join('outputs', args.exp_name, 'best_graphs')
    
    # Try using the new plotting function with PNG and PDF support
    png_path, pdf_path = _plot_best_graph_with_plotting_func(
        best_graph=best_graph,
        epoch=epoch,
        out_dir=out_dir,
        file_stem=file_stem,
    )
    
    if png_path is None:
        return None, None
    
    # Also save JSON stats
    json_path = join(out_dir, f'{file_stem}_stats.json')
    payload = {
        'epoch': int(epoch),
        'targets': {k: float(v) for k, v in best_graph.get('targets', {}).items()},
        'reward': float(best_graph.get('reward', best_graph['score'])),
        'reward_breakdown': {
            k: float(v) for k, v in best_graph.get('reward_breakdown', best_graph.get('score_breakdown', {})).items()
        },
        'score': float(best_graph['score']),
        'score_breakdown': {k: float(v) for k, v in best_graph.get('score_breakdown', {}).items()},
        'metrics': {k: float(v) for k, v in best_graph.get('metrics', {}).items()},
        'constraint_report': best_graph.get('constraint_report', {}),
        'sample_index': int(best_graph.get('sample_index', -1)),
        'plot_paths': {
            'png': png_path,
            'pdf': pdf_path if pdf_path else None,
        }
    }
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    
    return png_path, json_path


def _log_best_graph_to_wandb(png_path, json_path, epoch, tag='best_graph'):
    """Log best graph images and stats to Wandb."""
    if not getattr(args, 'use_wandb', False) or png_path is None:
        return
    
    try:
        # Log the graph image
        image = wandb.Image(png_path, caption=f'epoch={epoch}, {tag.replace("_", " ")}')
        wandb.log({f'best_graphs/{tag}': image})
        
        # Log the stats from JSON
        with open(json_path, 'r') as f:
            stats = json.load(f)
        
        # Log individual metrics
        log_dict = {}
        for metric_name, metric_value in stats.get('metrics', {}).items():
            log_dict[f'best_graphs/{tag}/metrics/{metric_name}'] = metric_value
        for breakdown_name, breakdown_value in stats.get('reward_breakdown', stats.get('score_breakdown', {})).items():
            log_dict[f'best_graphs/{tag}/reward_breakdown/{breakdown_name}'] = breakdown_value
        for breakdown_name, breakdown_value in stats.get('score_breakdown', {}).items():
            log_dict[f'best_graphs/{tag}/score_breakdown/{breakdown_name}'] = breakdown_value
        log_dict[f'best_graphs/{tag}/reward'] = stats.get('reward', stats.get('score', 0.0))
        log_dict[f'best_graphs/{tag}/score'] = stats.get('score', 0.0)
        log_dict[f'best_graphs/{tag}/epoch'] = epoch
        
        if log_dict:
            wandb.log(log_dict)
    except Exception as e:
        print(f"Warning: Could not log best graph to wandb: {e}")


def _merge_best_graph_stats_into_logs(log_dict, best_graph, prefix, epoch):
    if best_graph is None:
        return

    if 'score' in best_graph:
        log_dict[f'{prefix}/score'] = float(best_graph['score'])
    if 'reward' in best_graph:
        log_dict[f'{prefix}/reward'] = float(best_graph['reward'])
    log_dict[f'{prefix}/epoch'] = float(epoch)
    constraint_report = best_graph.get('constraint_report', {})
    if constraint_report:
        if 'num_constraints_satisfied' in constraint_report:
            log_dict[f'{prefix}/constraints/num_satisfied'] = float(constraint_report['num_constraints_satisfied'])
        if 'constraint_score' in constraint_report:
            log_dict[f'{prefix}/constraints/score'] = float(constraint_report['constraint_score'])
        if 'minimum_angle' in constraint_report:
            log_dict[f'{prefix}/constraints/minimum_angle'] = float(constraint_report['minimum_angle'])
        if 'edge_length_max' in constraint_report:
            log_dict[f'{prefix}/constraints/edge_length_max'] = float(constraint_report['edge_length_max'])
        if 'max_degree' in constraint_report:
            log_dict[f'{prefix}/constraints/max_degree'] = float(constraint_report['max_degree'])
        if 'edge_intersection_loss' in constraint_report:
            log_dict[f'{prefix}/constraints/edge_intersection_loss'] = float(constraint_report['edge_intersection_loss'])

    for metric_name, metric_value in best_graph.get('metrics', {}).items():
        log_dict[f'{prefix}/metrics/{metric_name}'] = float(metric_value)

    for breakdown_name, breakdown_value in best_graph.get('reward_breakdown', {}).items():
        log_dict[f'{prefix}/reward_breakdown/{breakdown_name}'] = float(breakdown_value)
    for breakdown_name, breakdown_value in best_graph.get('score_breakdown', {}).items():
        log_dict[f'{prefix}/score_breakdown/{breakdown_name}'] = float(breakdown_value)

    for target_name, target_value in best_graph.get('targets', {}).items():
        log_dict[f'{prefix}/targets/{target_name}'] = float(target_value)



def main(local_rank):
    # Retrieve dataloaders before model setup so conditioning context dimensions can be inferred.
    dataloaders, charge_scale, train_sampler = dataset.retrieve_dataloaders(args)
    data_dummy = next(iter(dataloaders['train']))

    if len(args.conditioning) > 0:
        print(f'Conditioning on {args.conditioning} (source={getattr(args, "conditioning_source", "cli")})')
        property_norms = compute_mean_mad(dataloaders, args.conditioning, args.dataset)
        context_dummy = prepare_context(args.conditioning, data_dummy, property_norms)
        args.context_node_nf = context_dummy.size(2)
    else:
        property_norms = None
        args.context_node_nf = 0

    use_case_targets = None
    if args.conditioning_use_case is not None:
        try:
            use_case_targets, target_path = load_conditioning_targets_from_use_case(
                use_case=args.conditioning_use_case,
                use_case_dir=args.conditioning_use_case_dir,
            )
            print(f'Loaded conditioning targets from use-case: {target_path}')
        except (FileNotFoundError, ValueError) as exc:
            print(f'Warning: failed to load use-case targets ({exc}). Target-conditioned sampling will be disabled.')
            use_case_targets = None

    if args.track_best_graphs and use_case_targets is None:
        print('Warning: --track_best_graphs is enabled but no use-case targets were loaded; best-graph ranking will be skipped.')

    wandb_active = False
    model, nodes_dist = get_model(args, dataset_info)

    if exists(join('outputs', args.exp_name, 'generative_model.npy')) and exists(join('outputs', args.exp_name, 'optim.npy')) and (args.resume is None):
        print('Resume training for', join('outputs', args.exp_name, 'generative_model.npy'))
        flow_state_dict = torch.load(join('outputs', args.exp_name, 'generative_model.npy'))
        model.load_state_dict(flow_state_dict)
        print('Done loading for', join('outputs', args.exp_name, 'generative_model.npy'))


    if args.resume is not None:
        flow_state_dict = torch.load(join(args.resume, 'generative_model.npy'))
        model.load_state_dict(flow_state_dict)


    # Initialize dataparallel if enabled and possible.
    if args.dp and torch.cuda.device_count() > 1:

        ip = os.environ['MASTER_IP']
        port = os.environ['MASTER_PORT']
        hosts = int(os.environ['WORLD_SIZE']) 
        rank = int(os.environ['RANK']) 
        gpus = torch.cuda.device_count()

        dist.init_process_group(backend='nccl', init_method=f'tcp://{ip}:{port}', world_size=hosts*gpus, rank=rank*gpus+local_rank)
        torch.cuda.set_device(local_rank)
        model.cuda(local_rank)
        #model_dp = torch.nn.DataParallel(model.cpu())
        #model_dp = model_dp.cuda()
        model_dp = DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    else:
        rank = 0
        model = model.to(device)
        model_dp = model

    if args.wandb and rank == 0 and local_rank == 0:
        run_name = args.wandb_run_name if args.wandb_run_name is not None else args.exp_name
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=run_name,
            mode=args.wandb_mode,
            config=vars(args),
        )
        wandb_active = True

    args.use_wandb = wandb_active


    optim = get_optim(args, model)
    if exists(join('outputs', args.exp_name, 'optim.npy')) and (args.resume is None):
        optim_state_dict = torch.load(join('outputs', args.exp_name, 'optim.npy'))
        optim.load_state_dict(optim_state_dict)

    if args.resume is not None:
        optim_state_dict = torch.load(join(args.resume, 'optim.npy'))
        optim.load_state_dict(optim_state_dict)
        

    # Initialize model copy for exponential moving average of params.
    if args.ema_decay > 0:
        model_ema = copy.deepcopy(model)
        ema = flow_utils.EMA(args.ema_decay)

        if args.dp and torch.cuda.device_count() > 1:
            #model_ema_dp = torch.nn.DataParallel(model_ema)
            model_ema_dp = DistributedDataParallel(model_ema, device_ids=[local_rank], find_unused_parameters=True)
        
        else:
            model_ema_dp = model_ema
    else:
        ema = None
        model_ema = model
        model_ema_dp = model_dp

    
    prop_dist = get_prop_dist(args, dataloaders['train'])

    if prop_dist is not None:
        prop_dist.set_normalizer(property_norms)
        
        

    best_nll_val = 1e8
    best_nll_test = 1e8
    global_best_graph = None
    global_best_planar_graph = None
    global_best_epoch = None
    global_best_planar_epoch = None
    for epoch in range(args.start_epoch, args.n_epochs):
        if (train_sampler is not None) and (torch.cuda.device_count() > 1):
            train_sampler.set_epoch(epoch)
            
        start_epoch = time.time()
        train_epoch(args=args, loader=dataloaders['train'], epoch=epoch, model=model, model_dp=model_dp,
                    model_ema=model_ema, ema=ema, device=device, dtype=dtype, property_norms=property_norms,
                    nodes_dist=nodes_dist, dataset_info=dataset_info,
                    gradnorm_queue=gradnorm_queue, optim=optim, prop_dist=prop_dist, rank=rank, local_rank=local_rank)
        print(f"Epoch took {time.time() - start_epoch:.1f} seconds.")

        if epoch % args.test_epochs == 0 and epoch > 0:
            print(f'Evaluating at epoch {epoch}...')
            if rank == 0 and local_rank == 0:
                if args.run_stability_eval and not args.break_train_epoch:
                    analyze_and_save(args=args, epoch=epoch, model_sample=model_ema, nodes_dist=nodes_dist,
                                    dataset_info=dataset_info, device=device,
                                    prop_dist=prop_dist, n_samples=args.n_stability_samples)
                    
                nll_val = test(args=args, loader=dataloaders['valid'], epoch=epoch, eval_model=model_ema_dp,
                            partition='Val', device=device, dtype=dtype, nodes_dist=nodes_dist,
                            property_norms=property_norms, rank=rank, local_rank=local_rank)
                
                nll_test = test(args=args, loader=dataloaders['test'], epoch=epoch, eval_model=model_ema_dp,
                                partition='Test', device=device, dtype=dtype,
                                nodes_dist=nodes_dist, property_norms=property_norms, rank=rank, local_rank=local_rank)

                sample_metric_logs = {}
                checkpoint_best_graph = None
                checkpoint_best_planar_graph = None
                planarity_percent = 0.0
                if args.wandb_log_conditioning_metrics or args.track_best_graphs:
                    eval_sample_count = int(args.conditioning_metric_eval_samples)
                    if args.track_best_graphs:
                        eval_sample_count = max(eval_sample_count, int(args.best_graph_eval_samples))

                    # Use the enhanced evaluation function that tracks planarity
                    sample_metric_logs, checkpoint_best_graph, checkpoint_best_planar_graph, planarity_percent = _evaluate_with_planarity_tracking(
                        args=args,
                        model_sample=model_ema,
                        nodes_dist=nodes_dist,
                        dataset_info=dataset_info,
                        prop_dist=prop_dist,
                        device=device,
                        property_norms=property_norms,
                        n_samples=eval_sample_count,
                        target_metrics=use_case_targets,
                        track_best_graph=bool(args.track_best_graphs),
                        relative_error_eps=float(args.best_graph_relative_error_eps),
                    )

                if args.track_best_graphs and checkpoint_best_graph is not None:
                    sample_metric_logs['samples/best_graph/checkpoint_epoch'] = float(epoch)
                    _merge_best_graph_stats_into_logs(
                        sample_metric_logs,
                        checkpoint_best_graph,
                        'samples/best_graph/checkpoint',
                        epoch,
                    )

                    if args.best_graph_save_every <= 0 or (epoch % args.best_graph_save_every) == 0:
                        checkpoint_stem = f'epoch_{int(epoch):04d}_best_graph'
                        checkpoint_png, checkpoint_json = _save_best_graph_artifacts(
                            args=args,
                            best_graph=checkpoint_best_graph,
                            epoch=epoch,
                            file_stem=checkpoint_stem,
                        )
                        if checkpoint_json is not None:
                            print(f'Saved checkpoint-best graph artifact: {checkpoint_json}')
                            # Log to wandb
                            _log_best_graph_to_wandb(checkpoint_png, checkpoint_json, epoch, f'checkpoint_epoch_{epoch}_best_graph')

                    if _selection_rank_for_graph(checkpoint_best_graph) > _selection_rank_for_graph(global_best_graph):
                        global_best_graph = checkpoint_best_graph
                        global_best_epoch = int(epoch)

                        global_png, global_json = _save_best_graph_artifacts(
                            args=args,
                            best_graph=global_best_graph,
                            epoch=epoch,
                            file_stem='best_graph',
                        )
                        _save_best_graph_artifacts(
                            args=args,
                            best_graph=global_best_graph,
                            epoch=epoch,
                            file_stem=f'global_best_epoch_{int(epoch):04d}',
                        )
                        if global_json is not None:
                            print(f'Updated global best graph artifact: {global_json}')
                            # Log to wandb
                            _log_best_graph_to_wandb(global_png, global_json, epoch, 'global_best_graph')

                # Track and plot best planar graph
                if args.track_best_graphs and checkpoint_best_planar_graph is not None:
                    sample_metric_logs['samples/best_planar_graph/checkpoint_epoch'] = float(epoch)
                    _merge_best_graph_stats_into_logs(
                        sample_metric_logs,
                        checkpoint_best_planar_graph,
                        'samples/best_planar_graph/checkpoint',
                        epoch,
                    )
                    
                    if args.best_graph_save_every <= 0 or (epoch % args.best_graph_save_every) == 0:
                        checkpoint_planar_stem = f'epoch_{int(epoch):04d}_best_planar_graph'
                        checkpoint_planar_png, checkpoint_planar_json = _save_best_graph_artifacts(
                            args=args,
                            best_graph=checkpoint_best_planar_graph,
                            epoch=epoch,
                            file_stem=checkpoint_planar_stem,
                        )
                        if checkpoint_planar_json is not None:
                            print(f'Saved checkpoint-best planar graph artifact: {checkpoint_planar_json}')
                            # Log to wandb
                            _log_best_graph_to_wandb(checkpoint_planar_png, checkpoint_planar_json, epoch, f'checkpoint_epoch_{epoch}_best_planar_graph')

                if args.track_best_graphs and global_best_graph is not None:
                    sample_metric_logs['samples/best_graph/global_reward'] = float(global_best_graph['reward'])
                    sample_metric_logs['samples/best_graph/global_score'] = float(global_best_graph['score'])
                    if global_best_epoch is not None:
                        sample_metric_logs['samples/best_graph/global_epoch'] = float(global_best_epoch)
                    _merge_best_graph_stats_into_logs(
                        sample_metric_logs,
                        global_best_graph,
                        'samples/best_graph/global',
                        global_best_epoch if global_best_epoch is not None else epoch,
                    )
                    for metric_key, rel_error in global_best_graph.get('reward_breakdown', global_best_graph.get('score_breakdown', {})).items():
                        sample_metric_logs[f'samples/best_graph/global_difference/{metric_key}'] = float(rel_error)
                    for metric_key, rel_error in global_best_graph.get('score_breakdown', {}).items():
                        sample_metric_logs[f'samples/best_graph/global_relative_error/{metric_key}'] = float(rel_error)

                # Track global best planar graph
                if args.track_best_graphs and checkpoint_best_planar_graph is not None:
                    if (
                        _selection_rank_for_graph(checkpoint_best_planar_graph)
                        > _selection_rank_for_graph(global_best_planar_graph)
                    ):
                        global_best_planar_graph = checkpoint_best_planar_graph
                        global_best_planar_epoch = int(epoch)

                        global_planar_png, global_planar_json = _save_best_graph_artifacts(
                            args=args,
                            best_graph=global_best_planar_graph,
                            epoch=epoch,
                            file_stem='best_planar_graph',
                        )
                        if global_planar_json is not None:
                            print(f'Updated global best planar graph artifact: {global_planar_json}')
                            # Log to wandb
                            _log_best_graph_to_wandb(global_planar_png, global_planar_json, epoch, 'global_best_planar_graph')
                    
                    if global_best_planar_graph is not None:
                        sample_metric_logs['samples/best_planar_graph/global_epoch'] = float(global_best_planar_epoch if global_best_planar_epoch is not None else epoch)
                        _merge_best_graph_stats_into_logs(
                            sample_metric_logs,
                            global_best_planar_graph,
                            'samples/best_planar_graph/global',
                            global_best_planar_epoch if global_best_planar_epoch is not None else epoch,
                        )
                
                # Log planarity percentage
                if planarity_percent > 0.0:
                    sample_metric_logs['samples/planarity_percentage'] = planarity_percent

                if wandb_active:
                    metric_payload = {
                        'epoch': epoch,
                        'val/nll': nll_val,
                        'test/nll': nll_test,
                    }
                    metric_payload.update(sample_metric_logs)
                    wandb.log(metric_payload)

                _log_generated_samples_to_wandb(
                    args=args,
                    model_sample=model_ema,
                    nodes_dist=nodes_dist,
                    dataset_info=dataset_info,
                    prop_dist=prop_dist,
                    device=device,
                    epoch=epoch,
                )

                _log_target_conditioned_samples_to_wandb(
                    args=args,
                    model_sample=model_ema,
                    nodes_dist=nodes_dist,
                    dataset_info=dataset_info,
                    device=device,
                    epoch=epoch,
                    property_norms=property_norms,
                    target_metrics=use_case_targets,
                )
                                

            if rank == 0 and local_rank == 0:
                if args.save_model:
                    _save_validation_checkpoint(
                        args=args,
                        epoch=epoch,
                        model=model,
                        model_ema=model_ema,
                        optim=optim,
                    )

                if nll_val < best_nll_val:
                    best_nll_val = nll_val
                    best_nll_test = nll_test

                    if args.save_model:
                        args.current_epoch = epoch + 1
                        utils.save_model(optim, 'outputs/%s/optim.npy' % args.exp_name)
                        utils.save_model(model, 'outputs/%s/generative_model.npy' % args.exp_name)
                        if args.ema_decay > 0:
                            utils.save_model(model_ema, 'outputs/%s/generative_model_ema.npy' % args.exp_name)
                        with open('outputs/%s/args.pickle' % args.exp_name, 'wb') as f:
                            pickle.dump(args, f)
                            
                print('Val loss: %.4f \t Test loss:  %.4f' % (nll_val, nll_test))
                print('Best val loss: %.4f \t Best test loss:  %.4f' % (best_nll_val, best_nll_test))

                if wandb_active:
                    wandb.log({
                        'best/val_nll': best_nll_val,
                        'best/test_nll': best_nll_test,
                    })

    if wandb_active:
        wandb.finish()


if __name__ == "__main__":
    ngpus = torch.cuda.device_count()

    if args.dp and ngpus > 1:
        print(f'Training using {ngpus} GPUs')
        torch.multiprocessing.spawn(main, args=(), nprocs=ngpus)
    else:
        main(0)

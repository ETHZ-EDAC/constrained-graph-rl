# Rdkit import should be first, do not move it
try:
    from rdkit import Chem
except ModuleNotFoundError:
    pass

import argparse
import pickle
from os.path import exists, join
from typing import Any, Dict

import matplotlib.pyplot as plt
import numpy as np
import torch
import wandb

from configs.datasets_config import get_dataset_info
from qm9 import dataset
from qm9.conditioning_config import resolve_conditioning_arguments
from qm9.models import get_model
from qm9.planar_metrics_adapter import (
    build_conditioning_context_from_targets,
    load_conditioning_targets_from_use_case,
)
from qm9.sampling import sample as sample_graph_batch
from qm9.utils import compute_mean_mad, prepare_context


parser = argparse.ArgumentParser(description='Sample target-conditioned graphs and log to W&B')
parser.add_argument('--checkpoint_dir', type=str, required=True,
                    help='Directory containing args.pickle and model checkpoints.')
parser.add_argument('--checkpoint_name', type=str, default='generative_model_ema.npy',
                    help='Model filename inside checkpoint_dir. Falls back to generative_model.npy if missing.')
parser.add_argument('--conditioning_use_case', type=str, required=True,
                    help='Use-case YAML name/path used to read target metric values.')
parser.add_argument('--conditioning_use_case_dir', type=str, default='configs/conditioning/use_cases',
                    help='Directory used to resolve conditioning_use_case names.')
parser.add_argument('--conditioning_merge_cli', type=eval, default=False,
                    help='When True, merges use-case channels with saved --conditioning list.')
parser.add_argument('--num_samples', type=int, default=8,
                    help='Number of graphs to sample.')
parser.add_argument('--fix_noise', type=eval, default=True,
                    help='Use fixed noise for deterministic-like sampling.')
parser.add_argument('--wandb', type=eval, default=True,
                    help='Enable W&B logging.')
parser.add_argument('--wandb_project', type=str, default='mudiff-geometric',
                    help='W&B project name.')
parser.add_argument('--wandb_entity', type=str, default=None,
                    help='Optional W&B entity/team.')
parser.add_argument('--wandb_mode', type=str, default='offline',
                    help='W&B mode: online | offline | disabled')
parser.add_argument('--wandb_run_name', type=str, default=None,
                    help='Optional run name.')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='Disable CUDA.')


args = parser.parse_args()

args_pkl = join(args.checkpoint_dir, 'args.pickle')
if not exists(args_pkl):
    raise FileNotFoundError(f'Could not find args.pickle in {args.checkpoint_dir}.')

with open(args_pkl, 'rb') as f:
    train_args = pickle.load(f)

# Override relevant runtime arguments for sampling.
train_args.conditioning_use_case = args.conditioning_use_case
train_args.conditioning_use_case_dir = args.conditioning_use_case_dir
train_args.conditioning_merge_cli = args.conditioning_merge_cli
train_args.wandb = bool(args.wandb)
train_args.wandb_project = args.wandb_project
train_args.wandb_entity = args.wandb_entity
train_args.wandb_mode = args.wandb_mode
train_args.wandb_run_name = args.wandb_run_name
train_args.no_cuda = args.no_cuda

if not hasattr(train_args, 'n_dims'):
    train_args.n_dims = 3
if not hasattr(train_args, 'conditioning'):
    train_args.conditioning = []
if not hasattr(train_args, 'conditioning_use_case'):
    train_args.conditioning_use_case = args.conditioning_use_case
if not hasattr(train_args, 'conditioning_use_case_dir'):
    train_args.conditioning_use_case_dir = args.conditioning_use_case_dir
if not hasattr(train_args, 'conditioning_merge_cli'):
    train_args.conditioning_merge_cli = args.conditioning_merge_cli

train_args = resolve_conditioning_arguments(train_args)

dataset_info = get_dataset_info(train_args.dataset, train_args.remove_h, datadir=train_args.datadir)

train_args.cuda = not train_args.no_cuda and torch.cuda.is_available()
device = torch.device('cuda' if train_args.cuda else 'cpu')
print('Sampling on device:', device)

# Build dataloaders to compute property norms used to normalize target context values.
dataloaders, _, _ = dataset.retrieve_dataloaders(train_args)
data_dummy = next(iter(dataloaders['train']))

if len(train_args.conditioning) > 0:
    property_norms = compute_mean_mad(dataloaders, train_args.conditioning, train_args.dataset)
    context_dummy = prepare_context(train_args.conditioning, data_dummy, property_norms)
    train_args.context_node_nf = context_dummy.size(2)
else:
    property_norms = None
    train_args.context_node_nf = 0

model, nodes_dist = get_model(train_args, dataset_info)

ckpt_path = join(args.checkpoint_dir, args.checkpoint_name)
if not exists(ckpt_path):
    fallback_path = join(args.checkpoint_dir, 'generative_model.npy')
    if exists(fallback_path):
        ckpt_path = fallback_path
    else:
        raise FileNotFoundError(f'Could not find checkpoint {args.checkpoint_name} or generative_model.npy in {args.checkpoint_dir}.')

state = torch.load(ckpt_path, map_location=device)
model.load_state_dict(state)
model = model.to(device)
model.eval()

use_case_targets, target_path = load_conditioning_targets_from_use_case(
    use_case=train_args.conditioning_use_case,
    use_case_dir=train_args.conditioning_use_case_dir,
)
print(f'Loaded use-case targets from: {target_path}')
print(f'Conditioning keys: {train_args.conditioning}')
print(f'Target values: {use_case_targets}')

n_samples = max(1, int(args.num_samples))
target_n_nodes = use_case_targets.get('num_nodes', None)
if target_n_nodes is not None:
    fixed_n = int(max(1, min(dataset_info['max_n_nodes'], round(float(target_n_nodes)))))
    nodesxsample = torch.full((n_samples,), fixed_n, dtype=torch.long, device=device)
else:
    nodesxsample = nodes_dist.sample(n_samples).to(device)

context = None
if len(train_args.conditioning) > 0:
    context = build_conditioning_context_from_targets(
        conditioning=train_args.conditioning,
        target_metrics=use_case_targets,
        property_norms=property_norms,
        batch_size=n_samples,
        device=device,
    )

with torch.no_grad():
    one_hot, _, x, edge, node_mask = sample_graph_batch(
        args=train_args,
        device=device,
        generative_model=model,
        dataset_info=dataset_info,
        prop_dist=None,
        nodesxsample=nodesxsample,
        context=context,
        fix_noise=bool(args.fix_noise),
    )

run = None
if train_args.wandb:
    run_name = train_args.wandb_run_name if train_args.wandb_run_name is not None else 'target-conditioned-sampling'
    run = wandb.init(
        project=train_args.wandb_project,
        entity=train_args.wandb_entity,
        name=run_name,
        mode=train_args.wandb_mode,
        config={
            'checkpoint_dir': args.checkpoint_dir,
            'checkpoint_name': ckpt_path,
            'conditioning_use_case': train_args.conditioning_use_case,
            'conditioning_keys': train_args.conditioning,
            'target_metrics': use_case_targets,
            'num_samples': n_samples,
            'fix_noise': bool(args.fix_noise),
        },
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

    if run is not None:
        images.append(wandb.Image(fig, caption=f'target_sample={idx}, nodes={pos.shape[0]}'))

    plt.close(fig)

if run is not None and len(images) > 0:
    payload: Dict[str, Any] = {'samples/target_conditioned_graphs': images}
    for metric_name, metric_value in use_case_targets.items():
        payload[f'samples/target_conditions/{metric_name}'] = float(metric_value)
    wandb.log(payload)
    wandb.finish()

print('Done sampling target-conditioned graphs.')

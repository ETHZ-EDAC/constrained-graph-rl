#!/usr/bin/env python3
"""
Train a property regressor to predict graph metrics from HOG graphs.
This regressor will be used for guided conditional generation.

Usage:
    python scripts/train_regressor.py --dataset_root ext/hog_planar/ --epochs 200 --batch_size 32
"""

import argparse
import os
from pathlib import Path
from typing import Dict
from typing import Tuple
from typing import List
from typing import Optional

import torch
import torch.nn as nn
from torch.optim import Adam
from omegaconf import OmegaConf
from tqdm import tqdm
import networkx as nx
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from src.models.property_regressor import SimpleGCNRegressor
from src.datasets.hog_with_metrics import HOGDataModuleWithMetrics


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
    'use_case_1': {'num_nodes': 50.0, 'spectral_gap': 1.0, 'gini_coefficient': 0.1,
                   'clustering_coefficient': 0.2, 'isoperimetric_ratio': 1.0},
    'use_case_2': {'num_nodes': 80.0, 'angular_resolution': 0.8, 'edge_length_deviation': 0.7,
                   'gini_coefficient': 0.1, 'isoperimetric_ratio': 1.0},
    'use_case_3': {'num_nodes': 100.0, 'spectral_gap': 0.3, 'rectangularity': 1.0},
}


def resolve_metric_subset(metric_profile: str):
    return USE_CASE_METRIC_SUBSETS[metric_profile]


def resolve_target_metrics(metric_profile: str, validation_use_case: str):
    if validation_use_case != 'auto':
        return USE_CASE_TARGETS[validation_use_case]
    if metric_profile in USE_CASE_TARGETS:
        return USE_CASE_TARGETS[metric_profile]
    return None


def create_target_tensor(metric_subset: List[str], target_dict: Dict[str, float], device: str) -> torch.Tensor:
    target = torch.zeros(1, len(metric_subset), device=device)
    for i, metric_name in enumerate(metric_subset):
        target[0, i] = float(target_dict.get(metric_name, 0.0))
    return target


def compute_metric_stats(dataset, metric_subset: List[str]) -> Tuple[torch.Tensor, torch.Tensor]:
    metrics_cache = getattr(dataset, 'metrics_cache', {})
    if not metrics_cache:
        raise RuntimeError("Metric cache is empty; precompute_metric_caches() must run before training.")

    values = []
    for idx in sorted(metrics_cache.keys()):
        row = [float(metrics_cache[idx].get(metric_name, 0.0)) for metric_name in metric_subset]
        values.append(row)

    targets = torch.tensor(values, dtype=torch.float32)
    mean = targets.mean(dim=0)
    std = targets.std(dim=0, unbiased=False)
    std = torch.clamp(std, min=1e-3)
    return mean, std


def standardized_mse_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metric_mean: torch.Tensor,
    metric_std: torch.Tensor,
) -> torch.Tensor:
    del metric_mean
    scaled_errors = (predictions - targets) / metric_std.to(device=predictions.device, dtype=predictions.dtype)
    return torch.mean(scaled_errors * scaled_errors)


def compute_per_metric_rmse(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    metric_subset: List[str],
) -> Dict[str, float]:
    rmse = torch.sqrt(torch.mean((predictions - targets) ** 2, dim=0))
    return {metric_subset[i]: float(rmse[i].item()) for i in range(len(metric_subset))}


def plot_best_graph_from_data(data, output_path: Path):
    graph = nx.Graph()
    num_nodes = int(data.x.size(0)) if getattr(data, 'x', None) is not None else int(data.n_nodes.item())
    graph.add_nodes_from(range(num_nodes))

    edges = data.edge_index.t().cpu().tolist()
    undirected_edges = {(min(int(u), int(v)), max(int(u), int(v))) for u, v in edges if int(u) != int(v)}
    graph.add_edges_from([(u, v) for u, v in undirected_edges])

    plt.figure(figsize=(4, 4))
    pos = nx.spring_layout(graph, seed=0)
    nx.draw(graph, pos, node_size=40, with_labels=False, edge_color='gray')
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=140)
    plt.close()


@torch.no_grad()
def validation_target_matching(
    regressor,
    dataloader,
    metric_subset: List[str],
    target_dict: Dict[str, float],
    device: str,
    output_dir: Path,
    epoch: int,
):
    target_tensor = create_target_tensor(metric_subset, target_dict, device)

    was_training = regressor.training
    regressor.eval()

    target_vec = target_tensor.squeeze(0)
    best_score = None
    best_data = None
    best_pred = None
    score_acc = 0.0
    n_graphs = 0

    for batch in dataloader:
        batch = batch.to(device)
        predictions = regressor(batch.x, batch.edge_index, batch.batch)
        data_list = batch.to_data_list()

        for i, data in enumerate(data_list):
            pred_vec = predictions[i]
            mae = torch.mean(torch.abs(pred_vec - target_vec)).item()
            score_acc += mae
            n_graphs += 1

            if best_score is None or mae < best_score:
                best_score = mae
                best_data = data
                best_pred = pred_vec.detach().cpu()

    if n_graphs == 0 or best_data is None or best_pred is None:
        if was_training:
            regressor.train()
        raise RuntimeError("Validation dataloader returned zero graphs")

    mean_score = float(score_acc / n_graphs)

    best_graph_path = output_dir / f'epoch_{epoch + 1:04d}_best_graph.png'
    plot_best_graph_from_data(best_data, best_graph_path)

    best_metrics = {metric_subset[i]: float(best_pred[i].item()) for i in range(len(metric_subset))}

    if was_training:
        regressor.train()

    if best_score is None:
        raise RuntimeError("Internal error: best_score is None")

    return {
        'best_mae': float(best_score),
        'mean_mae': mean_score,
        'best_metrics': best_metrics,
        'best_graph_path': best_graph_path,
    }


def create_dataloaders(
    metric_subset,
    batch_size: int = 32,
    num_workers: int = 0,
) -> Tuple[Dict, Dict, HOGDataModuleWithMetrics]:
    """Create train/val/test dataloaders."""

    cfg = OmegaConf.create({
        'general': {'name': 'regressor_hog', 'gpus': 1},
        'train': {'batch_size': batch_size, 'num_workers': num_workers},
        'dataset': {'pin_memory': False},
    })

    datamodule = HOGDataModuleWithMetrics(cfg=cfg, metric_subset=metric_subset, metric_backend='graph_rl')
    dataloaders = {
        'train': datamodule.train_dataloader(),
        'val': datamodule.val_dataloader(),
        'test': datamodule.test_dataloader(),
    }
    datasets = {
        'train': datamodule.train_dataset,
        'val': datamodule.val_dataset,
        'test': datamodule.test_dataset,
    }
    return dataloaders, datasets, datamodule


def train_epoch(
    model: nn.Module,
    dataloader,
    optimizer,
    device: str,
    metric_mean: torch.Tensor,
    metric_std: torch.Tensor,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    
    for batch in tqdm(dataloader, desc="Training"):
        batch = batch.to(device)
        
        # Forward pass
        predictions = model(batch.x, batch.edge_index, batch.batch)
        
        # Extract targets (metrics)
        targets = batch.y.squeeze(-2) if batch.y.dim() > 2 else batch.y  # Shape: (batch_size, num_metrics)
        
        # Compute loss
        loss = standardized_mse_loss(predictions, targets, metric_mean, metric_std)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        
        total_loss += loss.item() * batch.num_graphs
    
    avg_loss = total_loss / len(dataloader.dataset)
    return avg_loss


def evaluate(
    model: nn.Module,
    dataloader,
    device: str,
    metric_subset: Optional[List[str]] = None,
    metric_mean: Optional[torch.Tensor] = None,
    metric_std: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0.0
    total_rmse = 0.0
    num_samples = 0
    per_metric_sse = None
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = batch.to(device)
            
            # Forward pass
            predictions = model(batch.x, batch.edge_index, batch.batch)
            
            # Extract targets
            targets = batch.y.squeeze(-2) if batch.y.dim() > 2 else batch.y
            
            # Compute metrics
            if metric_mean is None or metric_std is None:
                mse = nn.functional.mse_loss(predictions, targets)
            else:
                mse = standardized_mse_loss(predictions, targets, metric_mean, metric_std)
            rmse = torch.sqrt(mse)
            batch_sse = torch.sum((predictions - targets) ** 2, dim=0)
            if per_metric_sse is None:
                per_metric_sse = batch_sse
            else:
                per_metric_sse += batch_sse
            
            total_loss += mse.item() * batch.num_graphs
            total_rmse += rmse.item() * batch.num_graphs
            num_samples += batch.num_graphs
    
    result = {
        'mse': total_loss / num_samples,
        'rmse': total_rmse / num_samples,
    }
    if per_metric_sse is not None and metric_subset is not None:
        per_metric_rmse = torch.sqrt(per_metric_sse / num_samples)
        result['per_metric_rmse'] = {
            metric_subset[i]: float(per_metric_rmse[i].item())
            for i in range(len(metric_subset))
        }
    return result


def main(args):
    """Main training function."""
    
    # Setup
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    wandb_run = None
    if args.wandb:
        import wandb
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config={
                'metric_profile': args.metric_profile,
                'epochs': args.epochs,
                'batch_size': args.batch_size,
                'hidden_dim': args.hidden_dim,
                'lr': args.lr,
                'dropout': args.dropout,
                'num_workers': args.num_workers,
            },
        )
    
    # Create dataloaders
    print("Loading datasets...")
    metric_subset = resolve_metric_subset(args.metric_profile)
    print(f"Using metric subset: {metric_subset}")

    target_dict = resolve_target_metrics(args.metric_profile, args.validation_use_case)
    if args.validation_sample and target_dict is None:
        raise ValueError("Validation sampling requested but no single target is available. "
                         "Use --validation_use_case use_case_0|use_case_1|use_case_2|use_case_3.")

    dataloaders, datasets, datamodule = create_dataloaders(
        metric_subset=metric_subset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    print("Precomputing metric caches...")
    datamodule.precompute_metric_caches()
    print("Metric caches ready.")
    metric_mean, metric_std = compute_metric_stats(datasets['train'], metric_subset)
    print("Training target statistics:")
    for name, mean_val, std_val in zip(metric_subset, metric_mean.tolist(), metric_std.tolist()):
        print(f"  {name}: mean={mean_val:.4f}, std={std_val:.4f}")
    
    # Create model
    print("Creating model...")
    model = SimpleGCNRegressor(
        input_dim=1,
        hidden_dim=args.hidden_dim,
        num_metrics=len(metric_subset),
        dropout=args.dropout,
    ).to(device)

    # Optimizer
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    # Training loop
    print("Starting training...")
    best_val_rmse = float('inf')
    checkpoint_dir = Path(args.checkpoint_dir) / 'property_regressor'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    for epoch in range(args.epochs):
        train_loss = train_epoch(model, dataloaders['train'], optimizer, device, metric_mean, metric_std)
        val_metrics = evaluate(
            model,
            dataloaders['val'],
            device,
            metric_subset=metric_subset,
            metric_mean=metric_mean,
            metric_std=metric_std,
        )
        scheduler.step()
        
        print(f"Epoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.6f} | Val RMSE: {val_metrics['rmse']:.6f}")
        if 'per_metric_rmse' in val_metrics:
            metric_line = ", ".join(
                f"{name}={value:.4f}" for name, value in val_metrics['per_metric_rmse'].items()
            )
            print(f"  Val per-metric RMSE: {metric_line}")

        if wandb_run is not None:
            wandb.log({
                'epoch': epoch + 1,
                'train/loss': train_loss,
                'val/mse': val_metrics['mse'],
                'val/rmse': val_metrics['rmse'],
                'lr': scheduler.get_last_lr()[0],
            })

        if args.validation_sample and ((epoch + 1) % args.validation_sample_every == 0):
            if target_dict is None:
                raise RuntimeError("Validation sampling requires non-null target_dict")
            sample_eval = validation_target_matching(
                regressor=model,
                dataloader=dataloaders['val'],
                metric_subset=metric_subset,
                target_dict=target_dict,
                device=device,
                output_dir=checkpoint_dir / 'validation_samples',
                epoch=epoch,
            )
            print(f"Validation sampling epoch {epoch + 1}: best_mae={sample_eval['best_mae']:.4f}, "
                  f"mean_mae={sample_eval['mean_mae']:.4f}")
            if wandb_run is not None:
                import wandb
                to_log = {
                    'val_guidance/best_mae': sample_eval['best_mae'],
                    'val_guidance/mean_mae': sample_eval['mean_mae'],
                    'val_guidance/best_graph': wandb.Image(str(sample_eval['best_graph_path'])),
                }
                for k, v in sample_eval['best_metrics'].items():
                    to_log[f'val_guidance/best_{k}'] = float(v)
                wandb.log(to_log)
        
        # Save best checkpoint
        if val_metrics['rmse'] < best_val_rmse:
            best_val_rmse = val_metrics['rmse']
            checkpoint_path = checkpoint_dir / args.metric_profile / 'best_regressor.pt'
            if not checkpoint_path.parent.exists():
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

            torch.save(model.state_dict(), checkpoint_path)
            print(f"  → Saved best checkpoint to {checkpoint_path}")
            if wandb_run is not None:
                wandb.log({'best/val_rmse': best_val_rmse})
    
    # Evaluate on test set
    print("\nEvaluating on test set...")
    test_metrics = evaluate(
        model,
        dataloaders['test'],
        device,
        metric_subset=metric_subset,
        metric_mean=metric_mean,
        metric_std=metric_std,
    )
    print(f"Test MSE: {test_metrics['mse']:.6f}")
    print(f"Test RMSE: {test_metrics['rmse']:.6f}")
    if 'per_metric_rmse' in test_metrics:
        metric_line = ", ".join(
            f"{name}={value:.4f}" for name, value in test_metrics['per_metric_rmse'].items()
        )
        print(f"Test per-metric RMSE: {metric_line}")
    if wandb_run is not None:
        wandb.log({'test/mse': test_metrics['mse'], 'test/rmse': test_metrics['rmse']})
    
    # Save final checkpoint
    final_checkpoint_path = checkpoint_dir / args.metric_profile/ 'final_regressor.pt'
    torch.save(model.state_dict(), final_checkpoint_path)
    print(f"Saved final checkpoint to {final_checkpoint_path}")
    if wandb_run is not None:
        wandb.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train property regressor for HOG graphs')
    parser.add_argument('--metric_profile', type=str, default='use_case_1',
                        choices=['all_4_use_cases', 'use_case_0', 'use_case_1', 'use_case_2', 'use_case_3'],
                        help='Predefined metric profile for the requested use-cases')
    parser.add_argument('--epochs', type=int, default=200, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--hidden_dim', type=int, default=128, help='Hidden dimension')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for dataloader')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints/', help='Directory to save checkpoints')
    parser.add_argument('--wandb', type=bool, default=True, help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='graph_rl_regressor', help='wandb project name')
    parser.add_argument('--wandb_run_name', type=str, default='regressor_uc1_with_val_sampling', help='wandb run name')
    parser.add_argument('--validation_sample', type=bool, default=True,
                        help='Enable validation-time target matching with the regressor on validation graphs')
    parser.add_argument('--validation_use_case', type=str, default='use_case_1',
                        choices=['auto', 'use_case_0', 'use_case_1', 'use_case_2', 'use_case_3'],
                        help='Target profile for validation target matching. auto uses metric_profile when possible.')
    parser.add_argument('--validation_sample_every', type=int, default=5,
                        help='Run validation target matching every N epochs')
    
    args = parser.parse_args()
    main(args)

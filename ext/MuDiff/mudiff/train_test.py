from equivariant_diffusion.utils import assert_mean_zero_with_mask, remove_mean_with_mask,\
    assert_correctly_masked, sample_center_gravity_zero_gaussian_with_mask
import networkx as nx
import numpy as np
import qm9.visualizer as vis
from qm9.analyze import analyze_stability_for_molecules
from qm9.sampling import sample_chain, sample, sample_sweep_conditional
import utils
import qm9.utils as qm9utils
from qm9 import losses
import time
import torch
import wandb
from types import SimpleNamespace
from typing import Any, Dict, Optional, Tuple
from torch_geometric.data import Data

from qm9.planar_metrics_adapter import canonicalize_metric_name, compute_graph_metric_dict, build_conditioning_context_from_targets
from graph_rl.utils.benchmark_graphs import ensure_planarity_metric, metric_reward, prepare_graph_for_benchmark_evaluation


def train_epoch(args, loader, epoch, model, model_dp, model_ema, ema, device, dtype, property_norms, optim,
                nodes_dist, gradnorm_queue, dataset_info, prop_dist, rank=0, local_rank=0):
    model_dp.train()
    model.train()
    nll_epoch = []
    n_iterations = len(loader)
    for i, data in enumerate(loader):
        x = data['positions'].to(device, dtype)
        node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
        edge_mask = data['edge_mask'].to(device, dtype)
        one_hot = data['one_hot'].to(device, dtype)
        charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)
        
        edge_attr = data['edge_attr'].to(device, dtype)
        attn_bias = data['attn_bias'].to(device, dtype)
        
        
        x = remove_mean_with_mask(x, node_mask)

        if args.augment_noise > 0:
            # Add noise eps ~ N(0, augment_noise) around points.
            eps = sample_center_gravity_zero_gaussian_with_mask(x.size(), x.device, node_mask)
            x = x + eps * args.augment_noise

        x = remove_mean_with_mask(x, node_mask)

        check_mask_correct([x, one_hot, charges], node_mask)
        assert_mean_zero_with_mask(x, node_mask)

        h = {'categorical': one_hot, 'integer': charges}

        if len(args.conditioning) > 0:
            context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
            assert_correctly_masked(context, node_mask)
        else:
            context = None

        optim.zero_grad()

        # transform batch through flow
        nll, reg_term, mean_abs_z, loss_dict = losses.compute_loss_and_nll(args, model_dp, nodes_dist,
                                                                x, h, edge_attr, attn_bias, node_mask, edge_mask, context)
        # standard nll from forward KL
        loss = nll + args.ode_regularization * reg_term

        if not torch.isfinite(loss):
            if rank == 0 and local_rank == 0:
                print(f"Non-finite train loss at epoch {epoch}, iter {i}. Skipping batch.")
                print(f"nll={nll.item() if torch.is_tensor(nll) else nll}, reg_term={reg_term.item() if torch.is_tensor(reg_term) else reg_term}")
                if getattr(args, 'use_wandb', False):
                    wandb.log({
                        'train/non_finite_batch': 1,
                        'train/non_finite_epoch': epoch,
                        'train/non_finite_iter': i,
                    })
            optim.zero_grad()
            continue

        loss.backward()

        if args.clip_grad:
            grad_norm = utils.gradient_clipping(model, gradnorm_queue)
        else:
            grad_norm = 0.

        optim.step()

        # Update EMA if enabled.
        if args.ema_decay > 0:
            ema.update_model_average(model_ema, model)

            
        if rank == 0 and local_rank == 0:
            if i % args.n_report_steps == 0:
                print(f"\rEpoch: {epoch}, iter: {i}/{n_iterations}, "
                      f"Loss {loss.item():.2f}, NLL: {nll.item():.2f}, "
                      f"RegTerm: {reg_term.item():.1f}, "
                      f"GradNorm: {grad_norm:.1f}")

                print(f"\rLoss_xh: {loss_dict['loss_t'].mean(0).item():.3f}, Loss_adj: {loss_dict['loss_edge_t'].mean(0).item():.3f}, "
                      f"KL_prior_xh: {loss_dict['kl_prior'].mean(0).item():.3f}, KL_prior_adj: {loss_dict['kl_prior_edge'].mean(0).item():.3f}")

                if getattr(args, 'use_wandb', False):
                    wandb.log({
                        'train/loss': float(loss.item()),
                        'train/nll': float(nll.item()),
                        'train/reg_term': float(reg_term.item()),
                        'train/grad_norm': float(grad_norm),
                        'train/loss_xh': float(loss_dict['loss_t'].mean(0).item()),
                        'train/loss_adj': float(loss_dict['loss_edge_t'].mean(0).item()),
                        'train/kl_prior_xh': float(loss_dict['kl_prior'].mean(0).item()),
                        'train/kl_prior_adj': float(loss_dict['kl_prior_edge'].mean(0).item()),
                        'train/mean_abs_z': float(mean_abs_z.item()) if torch.is_tensor(mean_abs_z) else float(mean_abs_z),
                    })
            
            nll_epoch.append(nll.item())
        
#         if rank == 0 and local_rank == 0:
#             if (epoch % args.test_epochs == 0) and (i % args.visualize_every_batch == 0) and not (epoch == 0 and i == 0):
#                 start = time.time()
#                 if len(args.conditioning) > 0:
#                     save_and_sample_conditional(args, device, model_ema, prop_dist, dataset_info, epoch=epoch)
#                 save_and_sample_chain(model_ema, args, device, dataset_info, prop_dist, epoch=epoch,
#                                       batch_id=str(i))
#                 sample_different_sizes_and_save(model_ema, nodes_dist, args, device, dataset_info,
#                                                 prop_dist, epoch=epoch)
#                 print(f'Sampling took {time.time() - start:.2f} seconds')

        if args.break_train_epoch:
            break


def check_mask_correct(variables, node_mask):
    for i, variable in enumerate(variables):
        if len(variable) > 0:
            assert_correctly_masked(variable, node_mask)


def test(args, loader, epoch, eval_model, device, dtype, property_norms, nodes_dist, partition='Test', rank=0, local_rank=0):
    eval_model.eval()
    with torch.no_grad():
        nll_epoch = 0
        n_samples = 0

        n_iterations = len(loader)

        for i, data in enumerate(loader):
            x = data['positions'].to(device, dtype)
            batch_size = x.size(0)
            node_mask = data['atom_mask'].to(device, dtype).unsqueeze(2)
            edge_mask = data['edge_mask'].to(device, dtype)
            one_hot = data['one_hot'].to(device, dtype)
            charges = (data['charges'] if args.include_charges else torch.zeros(0)).to(device, dtype)


            edge_attr = data['edge_attr'].to(device, dtype)
            attn_bias = data['attn_bias'].to(device, dtype)

            if args.augment_noise > 0:
                # Add noise eps ~ N(0, augment_noise) around points.
                eps = sample_center_gravity_zero_gaussian_with_mask(x.size(),
                                                                    x.device,
                                                                    node_mask)
                x = x + eps * args.augment_noise

            x = remove_mean_with_mask(x, node_mask)
            check_mask_correct([x, one_hot, charges], node_mask)
            assert_mean_zero_with_mask(x, node_mask)

            h = {'categorical': one_hot, 'integer': charges}

            if len(args.conditioning) > 0:
                context = qm9utils.prepare_context(args.conditioning, data, property_norms).to(device, dtype)
                assert_correctly_masked(context, node_mask)
            else:
                context = None

            # transform batch through flow
            nll, _, _, _ = losses.compute_loss_and_nll(args, eval_model, nodes_dist,
                                                        x, h, edge_attr, attn_bias, node_mask, edge_mask, context)
            # standard nll from forward KL

            if not torch.isfinite(nll):
                if rank == 0 and local_rank == 0:
                    print(f"Non-finite {partition} NLL at epoch {epoch}, iter {i}. Skipping batch.")
                    if getattr(args, 'use_wandb', False):
                        wandb.log({
                            f'{partition.lower()}/non_finite_batch': 1,
                            f'{partition.lower()}/non_finite_epoch': epoch,
                            f'{partition.lower()}/non_finite_iter': i,
                        })
                continue

            nll_epoch += nll.item() * batch_size
            n_samples += batch_size

            if rank == 0 and local_rank == 0:
                if i % args.n_report_steps == 0:
                    print(f"\r {partition} NLL \t epoch: {epoch}, iter: {i}/{n_iterations}, "
                          f"NLL: {nll_epoch/n_samples:.2f}")

                    if getattr(args, 'use_wandb', False):
                        wandb.log({
                            f'{partition.lower()}/nll_running': float(nll_epoch / n_samples),
                        })
                
    return nll_epoch/n_samples if n_samples > 0 else float('nan')


def _build_target_metric_vector(target_metrics: Optional[Dict[str, Any]]) -> Dict[str, float]:
    if target_metrics is None:
        return {}

    vector: Dict[str, float] = {}
    for key, value in target_metrics.items():
        canonical_key = canonicalize_metric_name(key)
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric):
            continue
        vector[canonical_key] = numeric

    return vector


def _score_graph_metrics(
    metric_values: Optional[Dict[str, float]],
    target_vector: Dict[str, float],
    eps: float = 1e-6,
) -> Tuple[Optional[float], Dict[str, float]]:
    if metric_values is None or len(target_vector) == 0:
        return None, {}
    available_targets = {
        metric_key: float(target_value)
        for metric_key, target_value in target_vector.items()
        if metric_key != "num_nodes" and metric_key in metric_values
    }
    if len(available_targets) == 0:
        return None, {}
    reward, breakdown = metric_reward(
        target_metrics=available_targets,
        candidate_metrics=metric_values,
        metric_reward_weight=1.0,
    )
    return float(reward), breakdown


def _node_count_rank(num_nodes: float, target_num_nodes: Optional[float], *, window_size: float = 10.0) -> Tuple[int, float]:
    if target_num_nodes is None:
        return (0, 0.0)
    distance = abs(float(num_nodes) - float(target_num_nodes))
    if distance <= float(window_size):
        bucket = 0
    else:
        bucket = int(np.ceil((distance - float(window_size)) / float(window_size)))
    return (bucket, distance)


def _candidate_rank_tuple(
    metric_values: Dict[str, float],
    reward: Optional[float],
    target_vector: Dict[str, float],
) -> Tuple[float, int, float, float]:
    target_num_nodes = target_vector.get('num_nodes', None)
    node_bucket, node_distance = _node_count_rank(
        metric_values.get('num_nodes', 0.0),
        target_num_nodes,
    )
    is_planar = 1.0 if float(metric_values.get('is_planar', 0.0)) >= 0.5 else 0.0
    safe_reward = float(reward) if reward is not None else -float('inf')
    return (
        is_planar,
        -int(node_bucket),
        -float(node_distance),
        safe_reward,
    )


def _extract_largest_connected_component_tensors(
    one_hot_sample: torch.Tensor,
    x_sample: torch.Tensor,
    edge_sample: torch.Tensor,
    node_mask_sample: torch.Tensor,
) -> Optional[Dict[str, Any]]:
    node_mask_cpu = node_mask_sample.detach().cpu()
    if node_mask_cpu.dim() == 2:
        active_mask = node_mask_cpu[:, 0] > 0.5
    else:
        active_mask = node_mask_cpu > 0.5

    active_indices = active_mask.nonzero(as_tuple=False).view(-1)
    if active_indices.numel() == 0:
        return None

    edge_active = edge_sample.detach().cpu()
    if edge_active.dim() == 3:
        edge_active = edge_active[..., 1] if edge_active.size(-1) > 1 else edge_active[..., 0]
    edge_active = edge_active[active_indices][:, active_indices]
    edge_active = (edge_active > 0).to(dtype=torch.int64)

    adjacency_np = edge_active.numpy()
    adjacency_np = np.maximum(adjacency_np, adjacency_np.T)
    np.fill_diagonal(adjacency_np, 0)
    graph = nx.from_numpy_array(adjacency_np)
    if graph.number_of_nodes() == 0:
        return None

    largest_component = max(
        nx.connected_components(graph),
        key=lambda nodes: (len(nodes), -min(nodes)),
    )
    component_nodes = torch.tensor(sorted(largest_component), dtype=torch.long)
    component_size = int(component_nodes.numel())
    if component_size == 0:
        return None

    one_hot_comp = one_hot_sample.detach().cpu()[active_indices][component_nodes]
    x_comp = x_sample.detach().cpu()[active_indices][component_nodes]
    edge_comp = edge_active[component_nodes][:, component_nodes]
    node_mask_comp = torch.ones((component_size, 1), dtype=node_mask_cpu.dtype)
    edge_index = edge_comp.nonzero(as_tuple=False).t().contiguous()

    return {
        'one_hot': one_hot_comp,
        'x': x_comp,
        'edge': edge_comp,
        'node_mask': node_mask_comp,
        'num_nodes': component_size,
        'edge_index': edge_index,
    }


def _compute_graph_metrics_and_constraints(num_nodes: int, edge_index: torch.Tensor, positions: torch.Tensor):
    if int(num_nodes) < 2 or edge_index.numel() == 0:
        return None, {
            'status': 'degenerate_graph',
            'num_nodes': int(num_nodes),
            'num_edges': 0,
        }

    data = Data(
        edge_index=edge_index,
        pos=positions.to(dtype=torch.float32),
        num_nodes=int(num_nodes),
    )
    prepared, candidate_metrics, constraint_report = prepare_graph_for_benchmark_evaluation(
        data,
        strict=False,
    )
    metrics = {
        str(key): float(value)
        for key, value in candidate_metrics.items()
        if np.isfinite(float(value))
    }
    if 'triangles' in metrics and 'triangle_count' not in metrics:
        metrics['triangle_count'] = float(metrics['triangles'])
    if 'triangle_count' in metrics and 'triangles' not in metrics:
        metrics['triangles'] = float(metrics['triangle_count'])
    metrics['is_planar'] = float(ensure_planarity_metric(prepared, candidate_metrics))
    return metrics, constraint_report


def evaluate_generated_conditioning_metrics(
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
    if len(getattr(args, 'conditioning', [])) == 0:
        return {}, None

    if 'planar' not in str(getattr(args, 'dataset', '')).lower():
        return {}, None

    n_samples = max(1, int(n_samples))
    nodesxsample = nodes_dist.sample(n_samples).to(device)

    context = None
    sample_prop_dist = prop_dist
    if len(getattr(args, 'conditioning', [])) > 0 and target_metrics is not None:
        context = build_conditioning_context_from_targets(
            conditioning=args.conditioning,
            target_metrics=target_metrics,
            property_norms=property_norms,
            batch_size=n_samples,
            device=device,
        )
        sample_prop_dist = None

    with torch.no_grad():
        one_hot, _, x, edge, node_mask = sample(
            args,
            device,
            model_sample,
            dataset_info=dataset_info,
            prop_dist=sample_prop_dist,
            nodesxsample=nodesxsample,
            context=context,
        )

    collected = {canonicalize_metric_name(key): [] for key in args.conditioning}
    target_vector = _build_target_metric_vector(target_metrics)
    enable_best_tracking = bool(track_best_graph and len(target_vector) > 0)
    best_graph: Optional[Dict[str, Any]] = None

    for idx in range(n_samples):
        component = _extract_largest_connected_component_tensors(
            one_hot_sample=one_hot[idx],
            x_sample=x[idx],
            edge_sample=edge[idx],
            node_mask_sample=node_mask[idx],
        )
        if component is None:
            continue

        metric_values, constraint_report = _compute_graph_metrics_and_constraints(
            component['num_nodes'],
            component['edge_index'],
            component['x'],
        )
        if metric_values is None:
            continue
        metric_values = {
            canonicalize_metric_name(metric_key): float(metric_value)
            for metric_key, metric_value in metric_values.items()
            if np.isfinite(float(metric_value))
        }

        for metric_key in collected:
            value = metric_values.get(metric_key, None)
            if value is None:
                continue
            if not np.isfinite(value):
                continue
            collected[metric_key].append(float(value))

        if enable_best_tracking:
            score, breakdown = _score_graph_metrics(
                metric_values=metric_values,
                target_vector=target_vector,
                eps=relative_error_eps,
            )
            if score is None:
                continue
            rank = _candidate_rank_tuple(
                metric_values=metric_values,
                reward=score,
                target_vector=target_vector,
            )
            previous_rank = best_graph.get('selection_rank', None) if best_graph is not None else None
            if previous_rank is None or rank > previous_rank:
                best_graph = {
                    'sample_index': int(idx),
                    'reward': float(score),
                    'reward_breakdown': breakdown,
                    'score': float(score),
                    'score_breakdown': breakdown,
                    'selection_rank': rank,
                    'metrics': metric_values,
                    'constraint_report': constraint_report,
                    'targets': target_vector,
                    'one_hot': component['one_hot'],
                    'x': component['x'],
                    'edge': component['edge'],
                    'node_mask': component['node_mask'],
                }

    logs = {}
    for metric_key, values in collected.items():
        if len(values) == 0:
            continue
        values_np = np.asarray(values, dtype=np.float64)
        logs[f'samples/metrics/{metric_key}/mean'] = float(values_np.mean())
        logs[f'samples/metrics/{metric_key}/min'] = float(values_np.min())
        logs[f'samples/metrics/{metric_key}/max'] = float(values_np.max())

    if best_graph is not None:
        logs['samples/best_graph/checkpoint_reward'] = float(best_graph['reward'])
        logs['samples/best_graph/checkpoint_score'] = float(best_graph['score'])
        if best_graph.get('constraint_report') is not None:
            logs['samples/best_graph/checkpoint_num_constraints_satisfied'] = float(
                best_graph['constraint_report'].get('num_constraints_satisfied', 0)
            )
        for metric_key, rel_error in best_graph['reward_breakdown'].items():
            logs[f'samples/best_graph/checkpoint_difference/{metric_key}'] = float(rel_error)
        for metric_key, rel_error in best_graph['score_breakdown'].items():
            logs[f'samples/best_graph/checkpoint_relative_error/{metric_key}'] = float(rel_error)

    if not track_best_graph:
        best_graph = None
    return logs, best_graph



def save_and_sample_chain(model, args, device, dataset_info, prop_dist,
                          epoch=0, id_from=0, batch_id=''):
    one_hot, charges, x, edge = sample_chain(args=args, device=device, flow=model,
                                       n_tries=1, dataset_info=dataset_info, prop_dist=prop_dist)

    vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/chain/',
                      one_hot, charges, x, edge, dataset_info, id_from, name='chain')

    return one_hot, charges, x, edge


def sample_different_sizes_and_save(model, nodes_dist, args, device, dataset_info, prop_dist,
                                    n_samples=5, epoch=0, batch_size=100, batch_id=''):
    batch_size = min(batch_size, n_samples)
    for counter in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, edge, node_mask = sample(args, device, model, prop_dist=prop_dist,
                                                nodesxsample=nodesxsample,
                                                dataset_info=dataset_info)
        
        #print(f"Generated molecule: Positions {x[:-1, :, :]}")
        vis.save_xyz_file(f'outputs/{args.exp_name}/epoch_{epoch}_{batch_id}/', one_hot, charges, x, edge, dataset_info,
                          batch_size * counter, name='molecule')

def analyze_and_save(epoch, model_sample, nodes_dist, args, device, dataset_info, prop_dist,
                     n_samples=1000, batch_size=100):
    
    print(f'Analyzing molecule stability at epoch {epoch}...')
    batch_size = min(batch_size, n_samples)
    assert n_samples % batch_size == 0
    molecules = {'one_hot': [], 'x': [], 'edge': [], 'node_mask': []}
    
    
    for i in range(int(n_samples/batch_size)):
        nodesxsample = nodes_dist.sample(batch_size)
        one_hot, charges, x, edge, node_mask = sample(args, device, model_sample, dataset_info, prop_dist,
                                                    nodesxsample=nodesxsample)

        molecules['one_hot'].append(one_hot.detach().cpu())
        molecules['x'].append(x.detach().cpu())
        molecules['edge'].append(edge.detach().cpu())
        molecules['node_mask'].append(node_mask.detach().cpu())
        
    molecules = {key: torch.cat(molecules[key], dim=0) for key in molecules}
    validity_dict, rdkit_tuple = analyze_stability_for_molecules(molecules, dataset_info)

    print({'Validity': rdkit_tuple[0][0], 'Uniqueness': rdkit_tuple[0][1], 'Novelty': rdkit_tuple[0][2]})
    print(validity_dict)

    return validity_dict


def save_and_sample_conditional(args, device, model, prop_dist, dataset_info, epoch=0, id_from=0):
    one_hot, charges, x, node_mask = sample_sweep_conditional(args, device, model, dataset_info, prop_dist)

    vis.save_xyz_file(
        'outputs/%s/epoch_%d/conditional/' % (args.exp_name, epoch), one_hot, charges, x, dataset_info,
        id_from, name='conditional', node_mask=node_mask)

    return one_hot, charges, x

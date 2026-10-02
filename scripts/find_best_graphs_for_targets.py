"""Find best-matching graphs for the active target-metrics use case."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, cast

import hydra
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import to_absolute_path
from omegaconf import DictConfig, OmegaConf
from torch_geometric.data import Data

from graph_rl.utils.benchmark_graphs import (
    ensure_planarity_metric,
    find_best_from_generators,
    find_similar_graph_in_dataset,
    metric_reward,
    prepare_graph_for_benchmark_evaluation,
    run_generator_batches,
)
from graph_rl.utils.graph_helpers import GraphGeneratorConfig
from graph_rl.utils.plotting import plot_graph


def _as_plain_dict(value: Any) -> Dict[str, float]:
    container = OmegaConf.to_container(value, resolve=True)
    if not isinstance(container, dict):
        raise TypeError(f"Expected a dict-like metrics mapping, got: {type(container)}")
    return {str(k): float(v) for k, v in container.items() if v is not None}


def load_target_metrics(config_path: Path) -> tuple[Dict[str, float], float]:
    cfg = OmegaConf.load(config_path)
    metrics = _as_plain_dict(cfg.metrics)
    if not metrics:
        raise ValueError(f"No active `metrics:` block found in {config_path}.")
    return metrics, float(cfg.metric_reward_weight)


def _plot_data_graph(data: Data, *, title: str, out_path: Path) -> None:
    pos = cast(torch.Tensor, data.pos)
    edge_index = cast(torch.Tensor, data.edge_index)
    num_nodes = int(data.num_nodes) if data.num_nodes is not None else int(pos.shape[0])
    adj = torch.zeros((num_nodes, num_nodes), dtype=torch.float32)
    adj[edge_index[0], edge_index[1]] = 1.0
    adj[edge_index[1], edge_index[0]] = 1.0
    pos_np = pos.numpy()
    if pos_np.shape[1] == 2:
        pos_3d = np.zeros((pos_np.shape[0], 3), dtype=pos_np.dtype)
        pos_3d[:, :2] = pos_np
    else:
        pos_3d = pos_np
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plot_graph(
        jnp.asarray(adj.numpy()),
        jnp.asarray(pos_3d),
        pretty=True,
        save_path=out_path,
        save_dpi=300,
        save_pdf=True,
    )
    ax.set_title(title)
    plt.close(fig)


def _format_metric_statistics(metric_statistics: Dict[str, Any]) -> str:
    if not metric_statistics:
        return "n/a"
    parts = []
    for metric_name in sorted(metric_statistics.keys()):
        stats = metric_statistics[metric_name]
        if not isinstance(stats, dict):
            continue
        mean = stats["mean"]
        std = stats["std"]
        parts.append(f"{metric_name}={float(mean):.4f}±{float(std):.4f}")
    return ", ".join(parts)


def _format_constraint_report(report: Dict[str, Any]) -> List[str]:
    return [
        "constraints: "
        f"{int(report['num_constraints_satisfied'])}/4 "
        f"(angle={report['passes_minimum_angle']}, "
        f"edge_length={report['passes_edge_length']}, "
        f"degree={report['passes_max_degree']}, "
        f"intersection={report['passes_edge_intersection']})",
        "constraint_values: "
        f"minimum_angle={float(report['minimum_angle']):.6f}, "
        f"edge_length_min_scaled_to={float(report['edge_length_min_scaled_to']):.6f}, "
        f"edge_length_max={float(report['edge_length_max']):.6f}, "
        f"max_degree={int(report['max_degree'])}, "
        f"edge_intersection_loss={float(report['edge_intersection_loss']):.6f}",
    ]


def _format_candidate_metrics(metrics: Dict[str, Any]) -> str:
    return ", ".join(
        f"{metric}={float(value):.6f}"
        for metric, value in sorted(metrics.items())
        if isinstance(value, (int, float, np.floating, np.integer))
    )


def _format_objective_metrics(metrics: Dict[str, Any], target_metrics: Dict[str, float]) -> str:
    ordered_metrics = []
    if "num_nodes" in metrics:
        ordered_metrics.append("num_nodes")
    for metric_name in target_metrics.keys():
        if metric_name != "num_nodes" and metric_name in metrics:
            ordered_metrics.append(metric_name)
    return ", ".join(f"{metric}={float(metrics[metric]):.6f}" for metric in ordered_metrics)


def _constraint_summary(report: Dict[str, Any]) -> str:
    return (
        f"{int(report['num_constraints_satisfied'])}/4 "
        f"(angle={report['passes_minimum_angle']}, "
        f"edge_length={report['passes_edge_length']}, "
        f"degree={report['passes_max_degree']}, "
        f"intersection={report['passes_edge_intersection']})"
    )


def _write_benchmark_summary(
    out_path: Path,
    *,
    target_metrics: Dict[str, float],
    best_dataset: Dict[str, Any],
    best_generator: Dict[str, Any],
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "Run target metrics:",
        *[f"  {metric_name}: {float(metric_value):.6f}" for metric_name, metric_value in target_metrics.items()],
        "",
        "Best dataset match:",
        f"  dataset: {best_dataset['dataset_name']}  graph_id: {int(best_dataset['graph_index'])}",
        f"  reward: {float(best_dataset['reward']):.6f}",
        f"  objectives: {_format_objective_metrics(best_dataset['candidate_metrics'], target_metrics)}",
        f"  constraints: {_constraint_summary(best_dataset['constraint_report'])}",
        f"  metrics: {_format_candidate_metrics(best_dataset['candidate_metrics'])}",
        "",
        "Best generator match:",
        f"  generator: {best_generator['generator_name']}  seed: {int(best_generator['seed'])}  sample: {int(best_generator['sample_index'])}",
        f"  reward: {float(best_generator['reward']):.6f}",
        f"  objectives: {_format_objective_metrics(best_generator['candidate_metrics'], target_metrics)}",
        f"  constraints: {_constraint_summary(best_generator['constraint_report'])}",
        f"  metrics: {_format_candidate_metrics(best_generator['candidate_metrics'])}",
        "",
        "Per-generator summary:",
    ]
    for gen_name in sorted(best_generator["per_generator_best"].keys()):
        info = best_generator["per_generator_best"][gen_name]
        reward_val = info["reward"]
        reward_str = "n/a" if reward_val is None or not np.isfinite(reward_val) else f"{float(reward_val):.6f}"
        if info["status"] == "error":
            lines.append(f"  {gen_name}: status=error reward={reward_str}")
            continue
        lines.append(
            f"  {gen_name}: status={info['status']} reward={reward_str} "
            f"seed={int(info['seed'])} sample={int(info['sample_index'])} "
            f"constraints={_constraint_summary(info['constraint_report'])}"
        )
        lines.append(f"    objectives: {_format_objective_metrics(info['candidate_metrics'], target_metrics)}")
        if info["best_valid_reward"] is not None:
            lines.append(
                f"    best_fully_valid: reward={float(info['best_valid_reward']):.6f} "
                f"seed={int(info['best_valid_seed'])} sample={int(info['best_valid_sample_index'])} "
                f"constraints={_constraint_summary(info['best_valid_constraint_report'])}"
            )
            lines.append(
                f"      objectives: {_format_objective_metrics(info['best_valid_candidate_metrics'], target_metrics)}"
            )
        else:
            lines.append("    best_fully_valid: none")
    out_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def find_best_in_dataset(
    *,
    target_metrics: Dict[str, float],
    metric_reward_weight: float,
    pt_pattern: str,
    plot_dir: Optional[Path],
) -> dict[str, Any]:
    metric_subset = [k for k in target_metrics.keys() if k != "num_nodes"]
    dataset_name, graph_idx, _score, _diffs, graph_data = find_similar_graph_in_dataset(
        target=target_metrics,
        metric_subset=metric_subset,
        pt_pattern=pt_pattern,
    )
    graph_data, candidate_metrics, constraint_report = prepare_graph_for_benchmark_evaluation(
        graph_data,
        strict=False,
    )
    planarity_score = ensure_planarity_metric(graph_data, candidate_metrics)
    candidate_metrics["is_planar"] = float(planarity_score)
    reward, differences = metric_reward(
        target_metrics=target_metrics,
        candidate_metrics=candidate_metrics,
        metric_reward_weight=metric_reward_weight,
    )
    if plot_dir is not None:
        out_path = plot_dir / f"best_dataset_{dataset_name}_idx{int(graph_idx)}.png"
        _plot_data_graph(
            graph_data,
            title=f"Best dataset match: {dataset_name} (graph {int(graph_idx)})\nreward={reward:.6f}",
            out_path=out_path,
        )
    return {
        "source": "dataset",
        "dataset_name": dataset_name,
        "graph_index": int(graph_idx),
        "reward": float(reward),
        "differences": {k: float(v) for k, v in differences.items()},
        "candidate_metrics": {k: float(v) for k, v in candidate_metrics.items() if isinstance(v, (int, float))},
        "constraint_report": constraint_report,
        "planarity": bool(planarity_score >= 0.5),
        "planarity_score": float(planarity_score),
    }


@hydra.main(config_path="../conf", config_name="find_best_graphs", version_base="1.2")
def main(cfg: DictConfig) -> None:
    target_config_path = Path(to_absolute_path(str(cfg.target_config)))
    pt_pattern = str(cfg.pt_pattern)
    seed = int(cfg.seed)
    num_samples = int(cfg.num_samples)
    target_metrics, weight = load_target_metrics(target_config_path)

    print("Run target metrics:")
    print(f"  target_config: {target_config_path}")
    print(f"  metric_reward_weight: {weight}")
    for metric_name, metric_value in target_metrics.items():
        print(f"  {metric_name}: {metric_value}")
    print()

    include_raw = OmegaConf.to_container(cfg.generators, resolve=True)
    include = [str(name).strip() for name in include_raw if str(name).strip()] if include_raw else None
    generator_batches = OmegaConf.to_container(cfg.generator_batches, resolve=True)
    if not isinstance(generator_batches, list):
        raise ValueError(f"generator_batches must be a list, got {type(generator_batches)}")
    if include is not None:
        generator_batches = [batch for batch in generator_batches if batch["name"] in include]
        if not generator_batches:
            raise ValueError("No generator batches remain after applying the generators filter.")

    config_overrides = OmegaConf.to_container(cfg.config_overrides, resolve=True)
    if not isinstance(config_overrides, dict):
        raise ValueError(f"config_overrides must be a mapping, got {type(config_overrides)}")
    plot_dir = None if cfg.plot_dir in (None, "") else Path(to_absolute_path(str(cfg.plot_dir)))
    verbose_generators = bool(cfg.verbose_generators)
    require_planar = True

    best_dataset = find_best_in_dataset(
        target_metrics=target_metrics,
        metric_reward_weight=weight,
        pt_pattern=pt_pattern,
        plot_dir=plot_dir,
    )

    base_config = GraphGeneratorConfig(
        num_nodes=int(target_metrics["num_nodes"]),
        seed=seed,
        ensure_connected=True,
        layout_method="spring",
        layout_scale=10.0,
    )
    for key, value in config_overrides.items():
        if value is not None:
            setattr(base_config, key, value)
    if "rectangularity" in target_metrics:
        base_config.grid_circular = False
        base_config.delaunay_boundary = "square"
    elif "isoperimetric_ratio" in target_metrics:
        base_config.grid_circular = True
        base_config.delaunay_boundary = "circle"

    if verbose_generators:
        print("Generator search settings (GraphGeneratorConfig):")
        for key, value in asdict(base_config).items():
            print(f"  {key}: {value}")
        print("  generator_batches:")
        for batch in generator_batches:
            sample_display = batch["samples"] if batch["samples"] is not None else num_samples
            overrides_str = ", ".join(f"{k}={v}" for k, v in sorted(batch["config"].items())) or "(default)"
            print(f"    {batch['name']}: samples={sample_display} overrides={overrides_str}")

    best_generator, best_data = (
        run_generator_batches(
            base_config,
            generator_batches,
            target_metrics=target_metrics,
            metric_reward_weight=weight,
            default_samples=num_samples,
            seed=seed,
            verbose=verbose_generators,
            require_planar=require_planar,
        )
        if generator_batches
        else find_best_from_generators(
            base_config,
            target_metrics=target_metrics,
            metric_reward_weight=weight,
            num_samples=num_samples,
            seed=seed,
            include_generators=include,
            verbose=verbose_generators,
            require_planar=require_planar,
        )
    )

    if plot_dir is not None and best_data is not None:
        gen_name = str(best_generator["generator_name"])
        out_path = plot_dir / f"best_generator_{gen_name}_seed{int(best_generator['seed'])}.png"
        _plot_data_graph(
            best_data,
            title=f"Best generator match: {gen_name} (seed {int(best_generator['seed'])})\nreward={float(best_generator['reward']):.6f}",
            out_path=out_path,
        )
        for gen_name in sorted(best_generator["per_generator_best"].keys()):
            gen_info = best_generator["per_generator_best"][gen_name]
            gen_graph_data = best_generator["per_generator_data"].get(gen_name)
            if gen_graph_data is not None and gen_info["status"] != "error":
                gen_reward = float(gen_info["reward"])
                gen_seed = int(gen_info["seed"])
                out_path = plot_dir / f"best_per_generator_{gen_name}_seed{gen_seed}.png"
                _plot_data_graph(
                    gen_graph_data,
                    title=f"Best from {gen_name} (seed {gen_seed})\nreward={gen_reward:.6f}",
                    out_path=out_path,
                )

    print("Best dataset match:")
    print(f"  dataset: {best_dataset['dataset_name']}  index: {best_dataset['graph_index']}")
    print(f"  reward: {best_dataset['reward']:.6f}")
    for line in _format_constraint_report(best_dataset["constraint_report"]):
        print(f"  {line}")
    print(f"  planar: {bool(best_dataset['planarity'])} (score={float(best_dataset['planarity_score']):.3f})")
    print(f"  metrics: {_format_candidate_metrics(best_dataset['candidate_metrics'])}")

    print("\nBest generator match:")
    print(f"  generator: {best_generator['generator_name']}  seed: {best_generator['seed']}")
    print(f"  reward: {best_generator['reward']:.6f}")
    for line in _format_constraint_report(best_generator["constraint_report"]):
        print(f"  {line}")
    if best_generator.get("samples") is not None:
        print(f"  samples: {int(best_generator['samples'])}")
    if best_generator["config_overrides"]:
        override_summary = ", ".join(
            f"{key}={best_generator['config_overrides'][key]}"
            for key in sorted(best_generator["config_overrides"].keys())
        )
        print(f"  overrides: {override_summary}")
    print(f"  planar: {bool(best_generator['planarity'])} (score={float(best_generator['planarity_score']):.3f})")
    print(f"  metrics: {_format_candidate_metrics(best_generator['candidate_metrics'])}")
    print(f"  generators_evaluated: {', '.join(best_generator['evaluated_generators'])}")

    print("\nBest reward per generator:")
    for gen_name in sorted(best_generator["per_generator_best"].keys()):
        info = best_generator["per_generator_best"][gen_name]
        reward_val = info["reward"]
        reward_str = "n/a" if reward_val is None or not np.isfinite(reward_val) else f"{float(reward_val):.6f}"
        header = f"  {gen_name}: status={info['status']} reward={reward_str}"
        if info["sample_index"] is not None:
            header += f" seed={info['seed']} sample={info['sample_index']}"
        print(header)
        if info.get("samples") is not None:
            print(f"    batch_samples: {int(info['samples'])}")
        print(
            f"    samples: attempted={int(info['attempted_samples'])} "
            f"planar={int(info['planar_samples'])} fully_valid={int(info['accepted_samples'])}"
        )
        if info["parameters"]:
            print(
                "    parameters: "
                + ", ".join(f"{key}={info['parameters'][key]}" for key in sorted(info["parameters"].keys()))
            )
        if info["config_overrides"]:
            print(
                "    overrides: "
                + ", ".join(f"{key}={info['config_overrides'][key]}" for key in sorted(info["config_overrides"].keys()))
            )
        if info["candidate_metrics"]:
            print(f"    metrics: {_format_candidate_metrics(info['candidate_metrics'])}")
        if info["metric_statistics"]:
            print(f"    mean_std: {_format_metric_statistics(info['metric_statistics'])}")
        if info["differences"]:
            print(
                "    metric_diffs: "
                + ", ".join(f"{metric}={float(val):.4f}" for metric, val in sorted(info["differences"].items()))
            )
        if info["constraint_report"]:
            for line in _format_constraint_report(info["constraint_report"]):
                print(f"    {line}")
        if info["best_valid_reward"] is not None:
            print(
                "    best_fully_valid: "
                f"reward={float(info['best_valid_reward']):.6f} "
                f"seed={info['best_valid_seed']} sample={info['best_valid_sample_index']}"
            )
            if info["best_valid_candidate_metrics"]:
                print(f"      metrics: {_format_candidate_metrics(info['best_valid_candidate_metrics'])}")
            if info["best_valid_constraint_report"]:
                for line in _format_constraint_report(info["best_valid_constraint_report"]):
                    print(f"      {line}")
            if info["best_valid_differences"]:
                print(
                    "      metric_diffs: "
                    + ", ".join(
                        f"{metric}={float(val):.4f}" for metric, val in sorted(info["best_valid_differences"].items())
                    )
                )
        else:
            print("    best_fully_valid: none")

    variant_map = best_generator["per_generator_variants"]
    for gen_name in sorted(variant_map.keys()):
        if len(variant_map[gen_name]) <= 1:
            continue
        print(f"\n  Variants for {gen_name}:")
        for variant in variant_map[gen_name]:
            reward_val = variant["reward"]
            reward_str = "n/a" if reward_val is None or not np.isfinite(reward_val) else f"{float(reward_val):.6f}"
            label = (
                ", ".join(
                    f"{key}={variant['config_overrides'][key]}" for key in sorted(variant["config_overrides"].keys())
                )
                or "(default)"
            )
            sample_str = "n/a" if variant["samples"] is None else str(int(variant["samples"]))
            print(
                f"    {label}: reward={reward_str} samples={sample_str} seed={variant['seed']} sample={variant['sample_index']}"
            )
            if variant["metric_statistics"]:
                print(f"      mean_std: {_format_metric_statistics(variant['metric_statistics'])}")
            if variant["constraint_report"]:
                for line in _format_constraint_report(variant["constraint_report"]):
                    print(f"      {line}")

    print("\nTarget vs achieved (best dataset):")
    for metric_name, target_val in target_metrics.items():
        got = best_dataset["candidate_metrics"].get(metric_name)
        got_str = "n/a" if got is None else f"{float(got):.6f}"
        print(f"  {metric_name}: target={float(target_val):.6f}  got={got_str}")

    print("\nTarget vs achieved (best generator):")
    for metric_name, target_val in target_metrics.items():
        got = best_generator["candidate_metrics"].get(metric_name)
        got_str = "n/a" if got is None else f"{float(got):.6f}"
        print(f"  {metric_name}: target={float(target_val):.6f}  got={got_str}")

    if plot_dir is not None:
        _write_benchmark_summary(
            plot_dir / "benchmark_summary.txt",
            target_metrics=target_metrics,
            best_dataset=best_dataset,
            best_generator=best_generator,
        )
        print(f"\nSaved plots to: {plot_dir}")


if __name__ == "__main__":
    main()

"""Gymnasium environment for constructing graphs via generator grammar."""

from __future__ import annotations

# Standard library
import time
import typing
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

# Third-party
import gymnasium as gym
import jax
import jax.numpy as jnp
import jax.random
import numpy as np
import torch as th
from gymnasium import spaces
from logging_mod.logger import get_logger
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from graph_rl.ops.cma_es import apply_cmaes_to_rule

# First-party
from graph_rl.ops.make_aabb import best_oriented_square_rectangularity
from graph_rl.grammar.constants import Constants
from graph_rl.grammar.planargraph import PlanarGraph
import graph_rl.grammar as grammar
from graph_rl.utils import PARAMS
from scipy.spatial import cKDTree

from graph_rl.utils.adjacency_utils import num_vertices
from graph_rl.ppo.metrics import compute_all_metrics_from_obs
from graph_rl.utils.benchmark_graphs import compute_metric_similarity


@dataclass(frozen=True)
class RuleSpec:
    """Encapsulates metadata about a graph generator rule."""

    name: str
    angles: int
    lens: int
    dim_cont: int


# + 1 angle for closing angle thing
RULE_SPECS: Tuple[RuleSpec, ...] = (
    RuleSpec("rule_1", angles=4, lens=3, dim_cont=7),
    RuleSpec("rule_2", angles=2, lens=1, dim_cont=3),
    RuleSpec("rule_3", angles=3, lens=2, dim_cont=5),
    RuleSpec("rule_4", angles=3, lens=2, dim_cont=5),
    RuleSpec("rule_5", angles=3, lens=2, dim_cont=5),
    RuleSpec("rule_6", angles=2, lens=1, dim_cont=3),
    RuleSpec("rule_7", angles=2, lens=1, dim_cont=3),
)

RULE_FN_TIME_INFO_KEY = "time_rule_fn_s"
CMAES_TIME_INFO_KEY = "time_apply_cmaes_to_rule_s"


def _block_tree_until_ready(tree: typing.Any) -> typing.Any:
    return jax.tree_util.tree_map(
        lambda leaf: leaf.block_until_ready() if hasattr(leaf, "block_until_ready") else leaf, tree
    )


def get_masked_action_dim(n_classes, cont_dim_mask) -> th.Tensor:
    mask_rules = th.zeros((n_classes, cont_dim_mask), dtype=th.bool)
    angles_max = max(spec.angles for spec in RULE_SPECS)
    for i in range(len(RULE_SPECS)):
        mask_rules[i, : RULE_SPECS[i].angles] = True  # angles_dim
        mask_rules[i, angles_max : angles_max + RULE_SPECS[i].lens] = True  # lens_dim
    return mask_rules


class PlanarGraphEnv(gym.Env):
    """Gymnasium-compatible environment for constructing graphs."""

    metadata = {"render_modes": ["human"], "render_fps": 2}

    def __init__(
        self,
        job_dir: Path,
        num_nodes: int = 30,
        step_limit: int = 200,
        enable_cmaes: bool = True,
        reward_shaping: bool = True,
        reward_type: str = "square",
        terminal_reward: str = "square",
        shaping_weight: float = 0.1,
        render_mode: Optional[str] = "human",
        discount_factor: float = 0.99,
        target_metrics: Optional[Dict[str, float]] = None,
        metric_reward_weight: float = 0.0,
        **kwargs,
    ) -> None:
        super().__init__()
        self.job_dir = Path(job_dir)
        self.max_cardinality = int(num_nodes) + 5
        self.max_steps = int(num_nodes)
        self.num_nodes = int(num_nodes)
        self.step_limit = int(step_limit)
        self.render_mode = render_mode
        self.logger = get_logger("base")
        self.enable_cmaes = enable_cmaes
        self.reward_shaping = bool(reward_shaping)
        self.shaping_weight = float(shaping_weight)
        self.reward_type = reward_type
        self.terminal_reward = terminal_reward

        max_angles = max(spec.angles for spec in RULE_SPECS)
        max_lens = max(spec.lens for spec in RULE_SPECS)

        lens_low = np.full(max_lens, PARAMS.edge_length_min, dtype=np.float64)
        lens_high = np.full(max_lens, PARAMS.edge_length_max, dtype=np.float64)
        sector_eps = np.deg2rad(PARAMS.sector_eps)
        self.action_space = spaces.Dict(
            {
                "vertex": spaces.Discrete(self.max_cardinality),
                "angles": spaces.Box(low=sector_eps, high=np.pi - sector_eps, shape=(max_angles,), dtype=np.float64),
                "lens": spaces.Box(low=lens_low, high=lens_high, dtype=np.float64),
            }
        )

        self.observation_space = spaces.Dict(
            {
                "adjacency": spaces.MultiBinary([self.max_cardinality, self.max_cardinality]),
                "adjacency_boundary": spaces.MultiBinary([self.max_cardinality, self.max_cardinality]),
                "vertex_positions": spaces.Box(-1000, 1000, (self.max_cardinality, 3), np.float64),
                "sector_angles": spaces.Box(
                    0,
                    2 * np.pi,
                    (self.max_cardinality, PARAMS.max_vertex_degree + 1),
                    np.float64,
                ),
                "step": spaces.Discrete(self.max_steps + 1),
            }
        )

        self.cp: Optional[PlanarGraph] = PlanarGraph(self.job_dir, max_cardinality=self.max_cardinality)

        self._applied_steps = 0
        self._step_id = 0
        self.is_success = True
        self.discount_factor = float(discount_factor)

        # Metric-based reward optimization
        self.target_metrics = target_metrics
        self.metric_reward_weight = float(metric_reward_weight)
        # If target_metrics disabled: compute nothing during episode (only at end for logging)
        # If target_metrics enabled: compute only selected metrics during episode for reward
        if target_metrics:
            self.metric_subset = list(target_metrics.keys())
            self.logger.info(f"Target metrics enabled. Computing {self.metric_subset} at each step for reward.")
        else:
            self.metric_subset = []  # Empty = skip metrics during episode
            self.logger.info("Target metrics disabled. Metrics will be computed only at episode end for logging.")

        # Cache potential Phi(s) = isoperimetric ratio for reward shaping
        self._last_potential: float = 0.0

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict] = None):
        super().reset(seed=seed, options=options)
        self.cp.reset_graph()

        self._step_id = 0
        self._applied_steps = 0
        observation = self.cp.constants
        self.is_success = True
        info = self._get_info()
        self._last_potential = self._get_potential(self.reward_type, info)
        return observation, info

    def step(self, action: Dict[str, typing.Any]):
        rule_idx = int(action["rule"])
        spec = RULE_SPECS[rule_idx]
        vertex = int(action["vertex"])  # always apply to the most recently added vertex
        terminated = False
        truncated = False
        info: Dict[str, object] = {RULE_FN_TIME_INFO_KEY: 0.0, CMAES_TIME_INFO_KEY: 0.0}
        self._applied_steps += 1

        rule_fn = getattr(self.cp, spec.name)
        action_rule = action.copy()
        for key in ["rule", "vertex"]:
            action_rule.pop(key)

        rule_fn_start_ns = time.perf_counter_ns()
        logs, cb_idx, constants_new = rule_fn(action_rule, jnp.array(vertex, dtype=jnp.int32), apply_rule=False)
        logs, cb_idx, constants_new = _block_tree_until_ready((logs, cb_idx, constants_new))
        info[RULE_FN_TIME_INFO_KEY] = (time.perf_counter_ns() - rule_fn_start_ns) / 1e9

        # Keep track of the executed action parameters (for supervision of projector)
        action_projected_env = {
            "rule": rule_idx,
            "vertex": vertex,
            "angles": np.asarray(action_rule["angles"], dtype=np.float32),
            "lens": np.asarray(action_rule["lens"], dtype=np.float32),
        }

        cost_geom, cost_topo = self._compute_feasibility(logs)
        info["cost_geom_nominal"] = cost_geom
        info["cost_topo"] = cost_topo

        info["cmaes_applied"] = False

        if cost_topo < 0.5:
            if cost_geom < 0.5:
                self._step_id += 1
                self.cp.constants = constants_new

            elif self.enable_cmaes:
                rule_specific = getattr(grammar, spec.name)
                key = self.np_random.integers(0, 1000)
                key_jax = jax.random.PRNGKey(key)
                cmaes_start_ns = time.perf_counter_ns()
                constants_proj, logs_proj, action_proj, _ = apply_cmaes_to_rule(
                    self.cp.constants, action_rule, jnp.array(vertex, dtype=jnp.int32), rule_specific, key_jax
                )
                constants_proj, logs_proj, action_proj, _ = _block_tree_until_ready(
                    (constants_proj, logs_proj, action_proj, _)
                )
                info[CMAES_TIME_INFO_KEY] = (time.perf_counter_ns() - cmaes_start_ns) / 1e9
                info["cmaes_applied"] = True
                action_projected_env = {
                    "rule": rule_idx,
                    "vertex": vertex,
                    "angles": np.asarray(action_proj["angles"], dtype=np.float32),
                    "lens": np.asarray(action_proj["lens"], dtype=np.float32),
                }

                cost_geom_prom, cost_topo_proj = self._compute_feasibility(logs_proj)

                info["cost_geom_proj"] = cost_geom_prom
                if cost_geom_prom < 0.5 and cost_topo_proj < 0.5:
                    info["cmaes_success"] = True
                    self._step_id += 1
                    self.cp.constants = constants_proj
                else:
                    self.is_success = False
                    info["cmaes_success"] = False
            else:
                self.is_success = False

        else:
            self.is_success = False

        # Potential-based reward shaping
        potential_prev = self._last_potential
        if self.reward_shaping:
            potential_next = self._get_potential(self.reward_type, info)
            reward_shaped = self.shaping_weight * (self.discount_factor * potential_next - potential_prev)
            self._last_potential = potential_next
        else:
            reward_shaped = 0.0

        if self.cp.num_vertices >= self.num_nodes:
            terminated = True
            terminal_reward = self._get_potential(self.terminal_reward, info, terminal=True)
            info["is_success"] = self.is_success
            info["terminal_reward"] = float(terminal_reward)  # Only log this at the end of the episode
            reward_shaped = (
                float(terminal_reward) - self.shaping_weight * potential_prev
                if self.reward_shaping
                else float(terminal_reward)
            )

        if self._applied_steps >= self.step_limit or self._step_id >= self.max_steps:
            truncated = True
            info["is_success"] = False

        observation = self.cp.constants
        info["action_projected_env"] = action_projected_env

        info.update(self._get_info())
        return observation, float(reward_shaped), terminated, truncated, info

    def compute_metrics(self, info, terminal: bool = False) -> float:
        # Compute metrics
        graph_metrics = compute_all_metrics_from_obs(
            self.cp.constants, num_nodes=int(self.cp.num_vertices), metric_subset=self.metric_subset
        )
        info["graph_metrics"] = graph_metrics.get("graph_metrics", {})

        metric_reward = 0.0
        if self.metric_reward_weight > 0:
            # Compute similarity score (higher = better, 1.0 = identical)
            metric_subset = [m for m in self.metric_subset if m != "num_nodes"]  # num nodes is done by grammar.
            similarity_score, differences = compute_metric_similarity(
                self.target_metrics,
                info["graph_metrics"],
                metric_subset,
            )

            # Use similarity directly as reward (higher similarity = higher reward)
            metric_reward = similarity_score * self.metric_reward_weight
            info["metric_reward"] = float(metric_reward)
            info["metric_similarity_score"] = float(similarity_score)
            info["metric_differences"] = {k: float(v) for k, v in differences.items()}
        if not terminal:
            info.pop("graph_metrics", None)
            info.pop("metric_reward", None)
            info.pop("metric_similarity_score", None)
            info.pop("metric_differences", None)
        return metric_reward

    def uniformity_score(self, constants: Constants, k=5) -> float:
        """
        Higher is better. 1.0 = perfectly uniform (all local scales equal).
        Approaches 0.0 as spacing becomes more irregular / clustered.
        """
        num = num_vertices(constants.adj)
        points = np.asarray(constants.vp[:num, :2])
        n = len(points)
        if n < 2:
            return 1.0  # trivially uniform

        k = min(k, n - 1)  # critical
        tree = cKDTree(points)
        dists, _ = tree.query(points, k=k + 1)  # includes self at 0
        knn = dists[:, 1:]

        # If k was capped correctly, knn should be finite; but keep it robust:
        finite_rows = np.isfinite(knn).all(axis=1)
        knn = knn[finite_rows]
        if knn.size == 0:
            return 0.0

        local_scale = np.mean(knn, axis=1)
        mu = np.mean(local_scale)
        if mu < 1e-12:
            return 0.0  # degenerate / duplicates

        cv = np.std(local_scale) / mu
        return 1.0 / (1.0 + cv)

    def get_rectangularity_reward(self, rectangle_or_iso="iso") -> jnp.ndarray:
        if rectangle_or_iso == "square":
            vertices_xy = self.cp.constants.vp[:, :2]
            area_polgyon, perim_poly = self.cp.area_and_perimeter()
            rectangularity, _, _, _ = best_oriented_square_rectangularity(
                vertices_xy=vertices_xy,
                adjacency_boundary=self.cp.constants.adj_b,
                adjacency_full=self.cp.constants.adj,
                area_polygon=area_polgyon,
                perim_polygon=perim_poly,
                num_angles=181,
            )
            return rectangularity

        elif rectangle_or_iso == "iso":
            area_polgyon, perim_poly = self.cp.area_and_perimeter()
            isoperimetric_ratio = 4 * jnp.pi * area_polgyon / (perim_poly * perim_poly + 1e-8)
            isoperimetric_ratio = jnp.clip(isoperimetric_ratio, 0.0, 1.0)
            return isoperimetric_ratio

    def _get_potential(self, reward_type: str, info, terminal=False) -> float:
        # Phi(s): isoperimetric ratio of the current graph, clipped to [0, 1]

        if reward_type in ("iso", "square"):
            potential = self.get_rectangularity_reward(reward_type)
            graph_metrics = dict(info.get("graph_metrics", {}))
            if reward_type == "iso":
                graph_metrics["isoperimetric_ratio"] = float(potential)
                info["graph_metrics"] = graph_metrics
            if reward_type == "square":
                graph_metrics["rectangularity"] = float(potential)
                info["graph_metrics"] = graph_metrics
        elif reward_type in ("uniformity"):
            potential = self.uniformity_score(self.cp.constants)
        elif reward_type == "metrics":
            potential = self.compute_metrics(info, terminal=terminal)
        else:
            potential = 0.0

        return float(potential)

    def _compute_feasibility(self, logs: Dict[str, jnp.ndarray]) -> Tuple[float, float]:
        logs.pop("edge_intersection")  # because it is seperated already into geom and topo
        logs.pop("full")
        # this is the part that 'can' be corrected with CMA-ES
        cost_geometric = float(logs["edge_intersection_geom"]) + float(logs["boundary_edge_length_geom"])
        cost_topological = 0.0  # this part must be corrected through re-sampling
        for key, value in logs.items():
            if key not in ("edge_intersection_geom", "boundary_edge_length_geom"):
                cost_topological += np.asarray(value).sum()
                # Sector violation geom can also appear here, if the incoming spanning angle of a vertex is too large,
                # and then expansion does not fit into the budget
        cost_geometric = 1.0 if cost_geometric > 1e-4 else 0.0
        cost_topological = 1.0 if cost_topological > 1e-4 else 0.0
        return cost_geometric, float(cost_topological)

    def render(self):
        if self.cp is None:
            return None
        self.cp.plot_planar_graph(True)
        return None

    def close(self):
        self.cp = None

    def _get_info(self) -> Dict[str, object]:
        assert self.cp is not None
        return {
            "num_vertices": int(self.cp.num_vertices),
        }


class ActionWrapper(gym.ActionWrapper):
    """Restrict actions to rule 1 and map a compact Box action to the environment Dict action."""

    def __init__(self, env: gym.Env):
        super().__init__(env)
        import numpy as np  # local import to avoid global dependency when unused

        self.min_angle = np.deg2rad(PARAMS.sector_eps)
        self.max_angle = 2 * np.pi - self.min_angle

        self.min_length = PARAMS.edge_length_min
        self.max_length = PARAMS.edge_length_max

        self.mask_rules = get_masked_action_dim(len(RULE_SPECS), max(rule.dim_cont for rule in RULE_SPECS))

        self.action_space = gym.spaces.Dict(
            {
                "rule": gym.spaces.Discrete(len(RULE_SPECS)),
                "vertex": gym.spaces.Box(low=-1.0, high=1.0, shape=(6,), dtype=np.float64),
            }
        )

    def __getattr__(self, name):
        return getattr(self.env, name)

    def action(self, action: np.ndarray):
        # action is in [-1, 1], map to angles, lens, rbm

        node_idx = int(action[0])
        rule_idx = int(action[1])

        cont_action = action[2:]
        extracted_action = cont_action[self.mask_rules[rule_idx]]

        specs = RULE_SPECS[rule_idx]  # to ensure rule exists

        # map angles and lengths from [-1, 1] to [0, 1]
        raw_angles = (extracted_action[: specs.angles] + 1) * 0.5  # [0, 1]
        raw_lengths = (extracted_action[specs.angles : specs.angles + specs.lens] + 1) * 0.5  # [0, 1]

        angles = jnp.asarray(raw_angles)
        lengths = jnp.asarray(raw_lengths)

        return {
            "rule": rule_idx,  # Select rule_1
            "vertex": node_idx,
            "angles": angles,
            "lens": lengths,
        }

    def step(self, action):
        obs, reward, terminated, truncated, info = super().step(action)

        if "action_projected_env" in info:
            info["action_projected_env"] = self._decode_projected(info["action_projected_env"])

        return obs, reward, terminated, truncated, info

    def _decode_projected(self, action_env: Dict[str, np.ndarray]) -> np.ndarray:
        """Map env-level projected action (angles/lens in [0,1]) back to policy space [-1,1]."""
        import numpy as np

        rule_idx = int(action_env["rule"])
        node_idx = int(action_env["vertex"])
        cont = np.zeros((self.mask_rules.shape[1],), dtype=np.float32)

        angles = np.asarray(action_env["angles"], dtype=np.float32).reshape(-1)
        lens = np.asarray(action_env["lens"], dtype=np.float32).reshape(-1)
        cont_vals = np.concatenate([angles, lens])
        cont_vals = np.clip(cont_vals, 0.0, 1.0)
        cont_vals = cont_vals * 2.0 - 1.0  # back to [-1, 1]

        cont[self.mask_rules[rule_idx]] = cont_vals
        # align shape with policy action format [node, rule, cont...]
        return np.concatenate([[node_idx], [rule_idx], cont]).astype(np.float32)


class ObservationWrapper(gym.ObservationWrapper):
    """Extract adjacency and vertex positions for MultiInputPolicy consumption."""

    def __init__(self, env: gym.Env):
        super().__init__(env)

        self.observation_space = gym.spaces.Graph(
            node_space=gym.spaces.Box(
                -1,
                2 * np.pi,
                (PARAMS.max_vertex_degree,),
                np.float32,
            ),
            edge_space=gym.spaces.Box(
                low=0.0,
                high=PARAMS.edge_length_max,
                shape=(1,),
            ),
        )

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _get_obs(self, constants: Constants) -> Dict[str, np.ndarray]:

        return {
            "adjacency": np.asarray(constants.adj, dtype=np.uint8),
            "adjacency_b": np.asarray(constants.adj_b, dtype=np.uint8),
            "vertex_positions": np.asarray(constants.vp, dtype=np.float32),
            "sector_angles": np.asarray(constants.sector_angles, dtype=np.float32),
        }

    def observation(self, constants: Constants) -> Data:

        observation = self._get_obs(constants)
        adjacency_full = observation["adjacency"]
        vertex_positions_full = observation["vertex_positions"][:, :2]  # take only x,y, drop z
        sector_angles_full = observation["sector_angles"]

        # Work only on the active subgraph to avoid O(N^2) scans on padding.
        n_active = int(self.cp.num_vertices)
        adjacency = adjacency_full[:n_active, :n_active]
        vertex_positions = vertex_positions_full[:n_active]
        sector_angles = sector_angles_full[:n_active]

        deg_array = (np.logical_or(adjacency > 0, adjacency.T > 0)).sum(0)
        deg_array_incoming = adjacency.sum(0)
        deg_array_outgoing = adjacency.sum(1)
        # root edge correction stays local to active vertices
        if n_active > 1:
            deg_array_outgoing[0] -= 1
        valid_mask = deg_array > 0
        deg = deg_array[valid_mask]
        deg_inc = deg_array_incoming[valid_mask]
        deg_out = deg_array_outgoing[valid_mask]

        boundary_mask = np.logical_and(deg_inc >= 1, deg_out == 0)

        if n_active == 2:  # index 0 expandable in first step.
            boundary_mask[0] = True
        boundary_mask = boundary_mask.astype(int)

        edge_list = np.argwhere(adjacency > 0).T.astype(np.int64)

        # 1) Get sector angles per node
        # sectors_raw: shape (N_valid, max_deg), already maintained by the grammar
        sectors = sector_angles[valid_mask, 1:]
        boundary_mask_deg_1 = np.logical_and(deg_inc == 1, deg_out == 0)
        sectors[boundary_mask_deg_1, 0] = 0.0  # fix deg=1 nodes to have sector angle 0

        # convention: sectors < 0 -> invalid/padded
        valid_sector_mask = sectors >= 0.0  # shape (N_valid, max_deg)

        deg_np = np.asarray(deg, dtype=np.float32).reshape(-1, 1)  # (N_valid, 1)
        deg_safe = np.clip(deg_np, 1.0, None)  # avoid /0

        K = 4  # number of harmonics
        fourier_feats = []

        for k in range(1, K + 1):
            angles_k = k * sectors  # (N_valid, max_deg)

            # apply mask so padded entries do not contribute
            # euler rule e^i*theta = cos(theta) + i sin(theta)
            cos_k = np.cos(angles_k) * valid_sector_mask
            sin_k = np.sin(angles_k) * valid_sector_mask

            cos_mean = cos_k.sum(axis=1, keepdims=True) / deg_safe  # (N_valid, 1)
            sin_mean = sin_k.sum(axis=1, keepdims=True) / deg_safe  # (N_valid, 1)

            fourier_feats.extend([cos_mean, sin_mean])

        # final shape: (N_valid, 2K)
        fourier_feats = np.concatenate(fourier_feats, axis=1).astype(np.float32)

        node_feature_parts = [
            boundary_mask[:, None],  # (N_valid, 1)
            np.asarray(deg_np, dtype=np.float32),  # (N_valid, 1)
        ]

        node_feature_parts.append(fourier_feats)  # (N_valid, 2K)
        node_features = np.concatenate(node_feature_parts, axis=1)

        node_pos = vertex_positions[valid_mask]

        obs = Data(
            pos=th.as_tensor(node_pos, dtype=th.float32),
            x=th.as_tensor(node_features, dtype=th.float32),
            edge_index=th.as_tensor(edge_list),
        ).sort()

        edge_idx_new, edge_attr_new = to_undirected(obs.edge_index, obs.edge_attr, reduce="max")
        obs.edge_index = edge_idx_new
        obs.edge_attr = edge_attr_new

        return obs

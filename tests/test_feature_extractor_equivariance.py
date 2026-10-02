from __future__ import annotations

# Standard library
import logging
import os
from pathlib import Path

# Third-party
import numpy as np
import pytest
import torch
from torch_geometric.data import Batch, Data

# First-party
from graph_rl.ppo.env import ActionWrapper, ObservationWrapper, PlanarGraphEnv
from graph_rl.ppo.layers import GraphFeatureExtractor
from graph_rl.ppo.ppo import PPO
from graph_rl.sb3_fork.vec_env.seq_vec_env import SequentialVecEnv


_PPO_FEATURE_EXTRACTOR_KWARGS = {
    "hidden_dim": 32,
    "embed_dim_deg": 20,
    "num_layers": 3,
    "num_freqs": 4,
}
_PPO_DISTRIBUTION_KWARGS = {
    "weight_entropy_node": 0.9,
    "weight_entropy_cat": 0.9,
    "weight_entropy_cont": 0.1,
}
_PPO_ACTOR_CRITIC_KWARGS = {
    "share_features_extractor": False,
    "use_joint_dist": False,
    "use_global_log_std": True,
    "init_std": 0.8,
    "rule_head": {"hidden_dim": 32, "num_layers": 2},
    "mean_head": {"hidden_dim": 32, "num_layers": 2, "embed_dim": 20},
    "log_std_head": {"num_layers": 1, "embed_dim": 20},
    "glob_critic": {"hidden_dim": 64, "num_layers": 2},
}
_PPO_ACTION_PROJECTOR_KWARGS = {
    "use_action_projector": True,
    "projection_supervised_epochs": 7,
    "stop_training_action_projector_after": 200,
    "replay_buffer_size": 10,
    "hidden_dim": 32,
    "embed_dim": 20,
}


def _rotate_translate_pos(pos: torch.Tensor, theta: float, offset: torch.Tensor) -> torch.Tensor:
    """Rotate positions around their centroid and then translate."""
    assert pos.ndim == 2 and pos.shape[1] == 2
    center = pos.mean(dim=0, keepdim=True)
    pos_c = pos - center
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    R = pos.new_tensor([[c, -s], [s, c]])
    pos_r = pos_c @ R.T
    return pos_r + center + offset


def _test_plots_dir() -> Path:
    return Path(__file__).resolve().parent / "_plots"


def _maybe_plot_graph_pair(*, obs: Data, obs_rot: Data, out_path: Path, theta: float) -> None:
    """
    Optionally save a side-by-side plot of the original and rotated graph.
    """

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def _draw(ax: plt.Axes, data: Data, title: str) -> None:
        pos = data.pos.detach().cpu().numpy()
        edge_index = data.edge_index.detach().cpu().numpy()
        ax.scatter(pos[:, 0], pos[:, 1], s=12, color="k")
        for i, j in edge_index.T:
            ax.plot([pos[i, 0], pos[j, 0]], [pos[i, 1], pos[j, 1]], color="0.4", linewidth=0.8, alpha=0.8)
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linestyle="--", linewidth=0.4, alpha=0.5)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    _draw(axes[0], obs, "original")
    _draw(axes[1], obs_rot, f"rotated (theta={theta:.3f} rad)")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _make_single_graph_obs(tmp_path: Path, *, rollout_steps: int = 10) -> Data:
    """
    Build a non-trivial graph by rolling out an untrained PPO policy for a few steps.

    This uses the same environment/wrappers as training (ActionWrapper + ObservationWrapper),
    so the returned graph is representative of the RL loop.
    """

    def _make_env():
        base_env = PlanarGraphEnv(
            job_dir=tmp_path,
            enable_cmaes=False,
            potential_shaping=False,
            domain_randomization=False,
            render_mode=None,
        )
        env = ActionWrapper(base_env)
        env = ObservationWrapper(env)
        return env

    vec_env = SequentialVecEnv([_make_env])
    model = PPO(
        vec_env,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        learning_rate=3e-4,
        seed=0,
        device="cpu",
        policy_kwargs=dict(
            features_extractor_class=GraphFeatureExtractor,
            features_extractor_kwargs=dict(_PPO_FEATURE_EXTRACTOR_KWARGS),
            distribution_kwargs=dict(_PPO_DISTRIBUTION_KWARGS),
            actor_critic_kwargs=dict(_PPO_ACTOR_CRITIC_KWARGS),
        ),
        action_projector_kwargs=dict(_PPO_ACTION_PROJECTOR_KWARGS),
    )

    env = vec_env.envs[0]
    obs, _info = env.reset()
    assert isinstance(obs, Data)

    for _ in range(int(rollout_steps)):
        action = model.policy.predict(obs, deterministic=False)
        obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break

    assert obs.pos.ndim == 2 and obs.pos.shape[1] == 2
    assert obs.x.ndim == 2
    assert obs.edge_index.ndim == 2 and obs.edge_index.shape[0] == 2
    vec_env.close()
    return obs


def _rollout_untrained_constants(tmp_path: Path, *, rollout_steps: int = 100):
    """
    Roll out an untrained PPO policy and return the underlying `Constants` object.
    """

    def _make_env():
        base_env = PlanarGraphEnv(
            job_dir=tmp_path,
            enable_cmaes=False,
            potential_shaping=False,
            domain_randomization=False,
            render_mode=None,
            max_steps=max(rollout_steps + 1, 16),
            step_limit=max(rollout_steps + 1, 32),
        )
        env = ActionWrapper(base_env)
        env = ObservationWrapper(env)
        return env

    vec_env = SequentialVecEnv([_make_env])
    model = PPO(
        vec_env,
        n_steps=8,
        batch_size=8,
        n_epochs=1,
        learning_rate=3e-4,
        seed=0,
        device="cpu",
        policy_kwargs=dict(
            features_extractor_class=GraphFeatureExtractor,
            features_extractor_kwargs=dict(_PPO_FEATURE_EXTRACTOR_KWARGS),
            distribution_kwargs=dict(_PPO_DISTRIBUTION_KWARGS),
            actor_critic_kwargs=dict(_PPO_ACTOR_CRITIC_KWARGS),
        ),
        action_projector_kwargs=dict(_PPO_ACTION_PROJECTOR_KWARGS),
    )

    env = vec_env.envs[0]
    obs, _info = env.reset()
    for _ in range(int(rollout_steps)):
        action = model.policy.predict(obs, deterministic=False)
        obs, _reward, terminated, truncated, _info = env.step(action)
        if terminated or truncated:
            break

    constants = env.cp.constants
    vec_env.close()
    return constants


def _rotate_constants_xy(constants, theta: float):
    """
    Rotate vertex positions of a `Constants` object around the centroid of active vertices.
    """
    import jax.numpy as jnp

    vp = np.asarray(constants.vp)
    adj_b = np.asarray(constants.adj_b)
    adj = np.asarray(constants.adj)

    A = adj_b if adj_b.max() > 0 else adj
    A = np.maximum(A, A.T)
    np.fill_diagonal(A, 0)
    active = (A.sum(axis=0) > 0) | (A.sum(axis=1) > 0)

    if active.any():
        center = vp[active, :2].mean(axis=0, keepdims=True)
    else:
        center = vp[:, :2].mean(axis=0, keepdims=True)

    c = float(np.cos(theta))
    s = float(np.sin(theta))
    R = np.array([[c, -s], [s, c]], dtype=np.float32)

    xy = vp[:, :2] - center
    xy_r = xy @ R.T + center
    vp_r = vp.copy()
    vp_r[:, :2] = xy_r
    return constants.replace(vp=jnp.asarray(vp_r))


def _eval_square_reward(constants, *, rotation_invariant_aabb: bool, num_angles: int = 180) -> float:
    """
    Evaluate the env's square rectangularity reward for a given constants state.
    """
    env = PlanarGraphEnv(
        job_dir=Path.cwd(),  # unused for reward; keep deterministic and valid
        enable_cmaes=False,
        potential_shaping=False,
        domain_randomization=False,
        render_mode=None,
        reward_type="square",
        rotation_invariant_aabb=rotation_invariant_aabb,
        rotation_invariant_aabb_num_angles=int(num_angles),
        debug_plot_rectangularity=False,
    )
    env.reset()
    env.cp.constants = constants
    r = float(env.get_rectangularity_reward("square"))
    env.close()
    return r


def test_feature_extractor_rotation_invariance_distinction(tmp_path: Path) -> None:
    """
    Sanity check: the `GraphFeatureExtractor` should return invariant features.
    """
    torch.manual_seed(0)

    obs = _make_single_graph_obs(tmp_path, rollout_steps=10)

    theta = 0.731  # non-trivial angle
    offset = obs.pos.new_tensor([3.1, -2.7])
    obs_rot = Data(
        x=obs.x.clone(),
        pos=_rotate_translate_pos(obs.pos, theta=theta, offset=offset),
        edge_index=obs.edge_index.clone(),
    ).sort()

    _maybe_plot_graph_pair(
        obs=obs,
        obs_rot=obs_rot,
        out_path=_test_plots_dir() / "graph_pair.png",
        theta=theta,
    )

    batch = Batch.from_data_list([obs])
    batch_rot = Batch.from_data_list([obs_rot])

    extractor = GraphFeatureExtractor(hidden_dim=64, num_layers=2)
    extractor.eval()

    with torch.no_grad():
        feats = extractor(batch)
        feats_rot = extractor(batch_rot)

    keys_to_check = ["node_features", "global_features", "node_logits"]
    for k in keys_to_check:
        assert k in feats and k in feats_rot
        assert feats[k].shape == feats_rot[k].shape

    assert torch.allclose(feats["node_logits"], feats_rot["node_logits"], rtol=1e-5, atol=1e-5)
    assert torch.allclose(feats["global_features"], feats_rot["global_features"], rtol=1e-5, atol=1e-5)
    assert torch.allclose(feats["node_features"], feats_rot["node_features"], rtol=1e-5, atol=1e-5)


@pytest.mark.parametrize("rotation_invariant_aabb", [True, False])
def test_square_reward_rotation_behavior(tmp_path: Path, rotation_invariant_aabb: bool) -> None:
    """
    - With rotation-invariant AABB enabled, the square reward should be invariant to global rotation.
    - With standard AABB, the square reward should generally change under rotation.
    """
    constants = _rollout_untrained_constants(tmp_path, rollout_steps=100)

    print("is rotation_invariant_aabb:", rotation_invariant_aabb)
    # Use a non-trivial angle (pi/4 is usually a strong axis-misalignment for standard AABB).
    # Use a rotation that is exactly representable by the discretization in the rot-invariant sweep.
    theta = float(np.pi / 4.0)
    constants_rot = _rotate_constants_xy(constants, theta=theta)

    r0 = _eval_square_reward(constants, rotation_invariant_aabb=rotation_invariant_aabb, num_angles=180)
    r1 = _eval_square_reward(constants_rot, rotation_invariant_aabb=rotation_invariant_aabb, num_angles=180)
    print(f"Reward before rotation: {r0:.6f}, after rotation: {r1:.6f}")

    if rotation_invariant_aabb:
        # The rotation-invariant AABB is implemented via a discrete angle sweep,
        # so expect approximate (not bit-exact) invariance.
        assert np.isclose(r0, r1, rtol=1e-3, atol=1e-3)
    else:
        # Standard AABB is not rotation invariant; ensure we see a meaningful change.
        assert not np.isclose(r0, r1, rtol=1e-4, atol=1e-4)

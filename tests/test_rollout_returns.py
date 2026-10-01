import numpy as np
import torch as th
from gymnasium import spaces
from torch_geometric.data import Data

from graph_rl.ppo.env import RULE_SPECS
from graph_rl.ppo.graph_rollout_buffer import GraphRolloutBuffer


def _make_buffer(n_steps: int, gamma: float = 0.9, gae_lambda: float = 1.0) -> GraphRolloutBuffer:
    action_dim = max(spec.dim_cont for spec in RULE_SPECS) + 2
    action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
    observation_space = spaces.Dict({})
    return GraphRolloutBuffer(
        n_steps=n_steps,
        batch_size=n_steps,
        observation_space=observation_space,
        action_space=action_space,
        gamma=gamma,
        gae_lambda=gae_lambda,
        n_envs=1,
    )


def _make_obs() -> dict[int, Data]:
    data = Data(x=th.zeros((1, 1)), edge_index=th.zeros((2, 0), dtype=th.long))
    return {0: data}


def _add_step(
    buffer: GraphRolloutBuffer,
    reward: float,
    value: float,
    episode_start: bool,
) -> None:
    actions = np.zeros((1, buffer.action_dim), dtype=np.float32)
    buffer.add(
        _make_obs(),
        actions,
        actions,
        actions,
        np.array([reward], dtype=np.float32),
        np.array([episode_start], dtype=bool),
        th.tensor([value], dtype=th.float32),
        th.zeros((1, 1), dtype=th.float32),
        th.zeros((1, 1), dtype=th.float32),
        np.array([0.0], dtype=np.float32),
    )


def test_terminal_last_step_return_zero_when_reward_zero() -> None:
    buffer = _make_buffer(n_steps=3, gamma=0.9)
    _add_step(buffer, reward=0.1, value=0.5, episode_start=True)
    _add_step(buffer, reward=0.2, value=0.5, episode_start=False)
    _add_step(buffer, reward=0.0, value=0.5, episode_start=False)

    buffer.finish_dataloader(last_values=th.tensor([10.0], dtype=th.float32), dones=np.array([True]))
    assert np.isclose(buffer.returns[-1, 0], 0.0)


def test_last_step_bootstraps_when_not_done() -> None:
    buffer = _make_buffer(n_steps=2, gamma=0.9)
    _add_step(buffer, reward=0.5, value=0.0, episode_start=True)
    _add_step(buffer, reward=1.0, value=0.0, episode_start=False)

    buffer.finish_dataloader(last_values=th.tensor([2.0], dtype=th.float32), dones=np.array([False]))
    assert np.isclose(buffer.returns[-1, 0], 1.0 + 0.9 * 2.0)


def test_episode_start_breaks_bootstrap_chain() -> None:
    buffer = _make_buffer(n_steps=2, gamma=0.9)
    _add_step(buffer, reward=1.0, value=0.3, episode_start=True)
    _add_step(buffer, reward=1.0, value=0.3, episode_start=True)

    buffer.finish_dataloader(last_values=th.tensor([0.0], dtype=th.float32), dones=np.array([False]))
    assert np.isclose(buffer.returns[0, 0], 1.0)

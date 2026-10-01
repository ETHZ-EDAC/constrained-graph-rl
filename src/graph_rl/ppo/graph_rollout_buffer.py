# Standard library
from typing import Generator, Optional, Union

# Third-party
import numpy as np
import torch as th
from gymnasium import spaces
from torch_geometric.data import Batch, Data
from torch_geometric.loader import DataLoader

# First-party
from graph_rl.sb3_fork.base_buffers import BaseBuffer


class GraphRolloutBuffer(BaseBuffer):
    """
    Dict Rollout buffer used in on-policy algorithms like A2C/PPO.
    Extends the RolloutBuffer to use dictionary observations

    It corresponds to ``buffer_size`` transitions collected
    using the current policy.
    This experience will be discarded after the policy update.
    In order to use PPO objective, we also store the current value of each state
    and the log probability of each taken action.

    The term rollout here refers to the model-free notion and should not
    be used with the concept of rollout used in model-based RL or planning.
    Hence, it is only involved in policy and value function training but not action selection.

    :param n_steps: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator
        Equivalent to Monte-Carlo advantage estimate when set to 1.
    :param gamma: Discount factor
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Dict
    observations: dict[int, dict[int, Data]]  # type: ignore[assignment]
    actions_nominal: np.ndarray
    actions_projected: np.ndarray  # model-projected (used for env step/log-prob)
    actions_projected_env: np.ndarray  # env-corrected projection (supervision)
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs_projected: np.ndarray
    log_probs_nominal: np.ndarray
    values: np.ndarray
    loader: DataLoader

    def __init__(
        self,
        n_steps: int,
        batch_size: int,
        observation_space: spaces.Dict,
        action_space: spaces.Space,
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__(n_steps, observation_space, action_space, n_envs=n_envs)

        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.batch_size = batch_size

        self.reset()

    def reset(self) -> None:
        self.observations = {}  # type: ignore[assignment]
        self.actions_nominal = np.zeros((self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype)
        self.actions_projected = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype
        )
        self.actions_projected_env = np.zeros(
            (self.buffer_size, self.n_envs, self.action_dim), dtype=self.action_space.dtype
        )
        self.rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.returns = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.episode_starts = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.values = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs_projected = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.log_probs_nominal = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.advantages = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        self.cost_feas = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def add(
        self,
        obs: dict[int, Data],
        action_nominal: np.ndarray,
        action_projected: np.ndarray,
        action_projected_env: np.ndarray,
        reward: np.ndarray,
        episode_start: np.ndarray,
        value: th.Tensor,
        log_prob_projected: th.Tensor,
        log_prob_nominal: th.Tensor,
        cost_feas: np.ndarray,
    ) -> None:
        """
        :param obs: Observation
        :param action_nominal: Action
        :param action_projected: Action
        :param reward:
        :param episode_start: Start of episode signal.
        :param value: estimated value of the current state
            following the current policy.
        :param log_prob: log probability of the action
            following the current policy.
        :param cost_feas: cost emitted by the environment for feasibility (converted to {0,1} here). 0 is good, 1 is bad
        """

        self.observations[self.pos] = obs
        self.actions_nominal[self.pos] = np.array(action_nominal)
        self.actions_projected[self.pos] = np.array(action_projected)
        self.actions_projected_env[self.pos] = np.array(action_projected_env)
        self.rewards[self.pos] = np.array(reward)
        self.episode_starts[self.pos] = np.array(episode_start)
        self.values[self.pos] = value.clone().cpu().numpy().flatten()
        self.log_probs_projected[self.pos] = log_prob_projected.clone().cpu().numpy()
        self.log_probs_nominal[self.pos] = log_prob_nominal.clone().cpu().numpy()
        self.cost_feas[self.pos] = np.array(cost_feas, dtype=np.float32)
        self.pos += 1
        if self.pos == self.buffer_size:
            self.full = True

    def get(  # type: ignore[override]
        self,
        batch_size: Optional[int] = None,
    ) -> Generator[Batch, None, None]:
        assert self.full, ""
        for mini_batch in self.loader:
            yield mini_batch

    def finish_dataloader(self, last_values: th.Tensor, dones) -> None:
        self.compute_returns_and_advantage(last_values, dones)
        data_list = []
        for key_steps, data in self.observations.items():
            for key_env, data_item in data.items():
                data_tmp = data_item.clone()
                data_tmp.actions_nominal = th.as_tensor(
                    self.actions_nominal[key_steps, key_env], dtype=th.float32
                ).view(1, -1)
                data_tmp.actions_projected = th.as_tensor(
                    self.actions_projected[key_steps, key_env], dtype=th.float32
                ).view(1, -1)
                data_tmp.actions_projected_env = th.as_tensor(
                    self.actions_projected_env[key_steps, key_env], dtype=th.float32
                ).view(1, -1)
                data_tmp.old_values = th.as_tensor(self.values[key_steps, key_env], dtype=th.float32).view(1, -1)
                data_tmp.old_log_prob_projected = th.as_tensor(
                    self.log_probs_projected[key_steps, key_env], dtype=th.float32
                ).view(1, -1)
                data_tmp.old_log_prob_nominal = th.as_tensor(
                    self.log_probs_nominal[key_steps, key_env], dtype=th.float32
                ).view(1, -1)
                data_tmp.advantages = th.as_tensor(self.advantages[key_steps, key_env], dtype=th.float32).view(1, -1)
                data_tmp.returns = th.as_tensor(self.returns[key_steps, key_env], dtype=th.float32).view(1, -1)
                data_tmp.cost_feas = th.as_tensor(self.cost_feas[key_steps, key_env], dtype=th.float32).view(1, -1)
                data_list.append(data_tmp)

        self.loader = DataLoader(data_list, batch_size=self.batch_size, shuffle=True)

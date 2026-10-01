# Standard library
from abc import ABC, abstractmethod
from collections.abc import Generator
from typing import Any, Optional, Union

# Third-party
import numpy as np
import torch as th
from gymnasium import spaces

# First-party
from graph_rl.ppo.env import RULE_SPECS


class BaseBuffer(ABC):
    """
    Base class that represent a buffer (rollout or replay)

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param n_envs: Number of parallel environments
    """

    observation_space: spaces.Space
    observations: Any
    actions_nominal: np.ndarray
    actions_projected: np.ndarray
    actions_projected_env: np.ndarray
    rewards: np.ndarray
    advantages: np.ndarray
    returns: np.ndarray
    episode_starts: np.ndarray
    log_probs_projected: np.ndarray
    log_probs_nominal: np.ndarray
    values: np.ndarray

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        gae_lambda: float = 1,
        gamma: float = 0.99,
        n_envs: int = 1,
    ):
        super().__init__()
        self.buffer_size = buffer_size
        self.observation_space = observation_space
        self.action_space = action_space

        self.action_dim = max(rule.dim_cont for rule in RULE_SPECS) + 2  # plus one for rule idx, node idx, rbm
        self.pos = 0
        self.full = False
        self.n_envs = n_envs
        self.gae_lambda = gae_lambda
        self.gamma = gamma
        self.reset()

    @abstractmethod
    def add(
        self,
        obs: np.ndarray,
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
        raise NotImplementedError()

    @abstractmethod
    def get(self, batch_size: Optional[int] = None) -> Generator[Any, None, None]:
        raise NotImplementedError()

    @abstractmethod
    def finish_dataloader(self, last_values, dones):
        raise NotImplementedError()

    def reset(self) -> None:
        """
        Reset the buffer.
        """

        self.pos = 0
        self.full = False

    def compute_returns_and_advantage(self, last_values: th.Tensor, dones: np.ndarray) -> None:
        """
        Post-processing step: compute the lambda-return (TD(lambda) estimate)
        and GAE(lambda) advantage.

        Uses Generalized Advantage Estimation (https://arxiv.org/abs/1506.02438)
        to compute the advantage. To obtain Monte-Carlo advantage estimate (A(s) = R - V(S))
        where R is the sum of discounted reward with value bootstrap
        (because we don't always have full episode), set ``gae_lambda=1.0`` during initialization.

        The TD(lambda) estimator has also two special cases:
        - TD(1) is Monte-Carlo estimate (sum of discounted rewards)
        - TD(0) is one-step estimate with bootstrapping (r_t + gamma * v(s_{t+1}))

        For more information, see discussion in https://github.com/DLR-RM/stable-baselines3/pull/375.

        :param last_values: state value estimation for the last step (one for each env)
        :param dones: if the last step was a terminal step (one bool for each env).
        """
        # Convert to numpy
        last_values = last_values.clone().cpu().numpy().flatten()  # type: ignore[assignment]

        last_gae_lam = 0
        for step in reversed(range(self.buffer_size)):
            if step == self.buffer_size - 1:
                next_non_terminal = 1.0 - dones.astype(np.float32)
                next_values = last_values
            else:
                next_non_terminal = 1.0 - self.episode_starts[step + 1]
                next_values = self.values[step + 1]
            delta = self.rewards[step] + self.gamma * next_values * next_non_terminal - self.values[step]
            last_gae_lam = delta + self.gamma * self.gae_lambda * next_non_terminal * last_gae_lam
            self.advantages[step] = last_gae_lam
        # TD(lambda) estimator, see Github PR #375 or "Telescoping in TD(lambda)"
        # in David Silver Lecture 4: https://www.youtube.com/watch?v=PnHCvfgC_ZA
        self.returns = self.advantages + self.values

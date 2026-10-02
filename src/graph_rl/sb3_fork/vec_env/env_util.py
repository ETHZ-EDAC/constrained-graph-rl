# Standard library
from pathlib import Path
from typing import Any, Callable, Optional, Union

# Third-party
import gymnasium as gym
from omegaconf import DictConfig

# First-party
from graph_rl.ppo import PlanarGraphEnv, ActionWrapper, ObservationWrapper
from graph_rl.sb3_fork.vec_env import BaseVecEnv, SequentialVecEnv, SubprocVecEnv
from graph_rl.sb3_fork.vec_env.monitor import Monitor


def make_vec_env(
    env_id: Union[str, Callable[..., gym.Env]],
    n_envs: int = 1,
    seed: Optional[int] = None,
    start_index: int = 0,
    env_kwargs: Optional[dict[str, Any]] = None,
    vec_env_cls: Optional[type[Union[SequentialVecEnv, SubprocVecEnv]]] = None,
) -> BaseVecEnv:
    """
    Create a wrapped, monitored ``BaseVecEnv``.
    By default it uses a ``SequentialBaseVecEnv`` which is usually faster
    than a ``SubprocVecEnv``.

    :param env_id: either the env ID, the env class or a callable returning an env
    :param n_envs: the number of environments you wish to have in parallel
    :param seed: the initial seed for the random number generator
    :param start_index: start rank index
    :param env_kwargs: Optional keyword argument to pass to the env constructor
    :param vec_env_cls: A custom ``BaseVecEnv`` class constructor. Default: None.
    :return: The wrapped environment
    """
    env_kwargs = env_kwargs or {}

    def make_env(rank: int) -> Callable[[], gym.Env]:
        def _init() -> gym.Env:
            # For type checker:
            assert env_kwargs is not None

            env = env_id(**env_kwargs)

            if seed is not None:
                # Note: here we only seed the action space
                # We will seed the env at the next reset
                env.action_space.seed(seed + rank)
            return env

        return _init

    if vec_env_cls is None:
        vec_env_cls = SequentialVecEnv

    vec_env = vec_env_cls([make_env(i + start_index) for i in range(n_envs)])
    # Prepare the seeds for the first reset
    vec_env.seed(seed)
    return vec_env


def make_rule1_env(job_dir: Path, cfg: DictConfig, env_name=None) -> Monitor:
    env_meta = cfg.policy.env

    # Prepare target metrics if enabled
    target_metrics_cfg = cfg.target_metrics
    enabled = target_metrics_cfg.enabled
    target_metrics = target_metrics_cfg.get("metrics") if enabled else None
    metric_reward_weight = target_metrics_cfg.get("metric_reward_weight", 0.0) if enabled else 0.0

    base_env = PlanarGraphEnv(
        job_dir=job_dir,
        **env_meta,
        discount_factor=float(cfg.policy.PPO.gamma),
        target_metrics=target_metrics,
        metric_reward_weight=metric_reward_weight,
    )
    base_env = ActionWrapper(base_env)
    base_env = ObservationWrapper(base_env)
    base_env = Monitor(base_env, env_name="env" if env_name is None else env_name)
    return base_env

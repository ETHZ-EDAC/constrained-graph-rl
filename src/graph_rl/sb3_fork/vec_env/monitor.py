__all__ = ["Monitor"]

# Standard library
import time
from typing import Any, SupportsFloat

# Third-party
import gymnasium as gym
from gymnasium.core import ActType, ObsType


class Monitor(gym.Wrapper[ObsType, ActType, ObsType, ActType]):
    """
    A monitor wrapper for Gym environments, it is used to know the episode reward, length, time and other data.

    :param env: The environment
    :param allow_early_resets: allows the reset of the environment before it is done
    :param reset_keywords: extra keywords for the reset call,
        if extra parameters are needed at reset
    :param info_keywords: extra information to log, from the information return of env.step()
    """

    def __init__(
        self,
        env: gym.Env,
        allow_early_resets: bool = True,
        reset_keywords: tuple[str, ...] = (),
        info_keywords: tuple[str, ...] = (),
        env_name: str = None,
    ):
        super().__init__(env=env)
        self.env_name = env_name
        self.t_start = time.time()
        self.reset_keywords = reset_keywords
        self.info_keywords = info_keywords
        self.allow_early_resets = allow_early_resets

        self.rewards: list[float] = []
        self.cost_geom_nominal: list[float] = []
        self.cost_geom_proj: list[float] = []
        self.cost_topo: list[float] = []

        self.cmaes_applied_num: list[int] = []
        self.cmaes_applied_success_rate: list[int] = []

        self.needs_reset = True
        self.episode_returns: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_times: list[float] = []
        self.total_steps = 0
        self.discount_factor = float(getattr(self.env, "discount_factor", 1.0))
        # extra info about the current episode, that was passed in during reset()
        self.current_reset_info: dict[str, Any] = {}

    def reset(self, **kwargs) -> tuple[ObsType, dict[str, Any]]:
        """
        Calls the Gym environment reset. Can only be called if the environment is over, or if allow_early_resets is True

        :param kwargs: Extra keywords saved for the next episode. only if defined by reset_keywords
        :return: the first observation of the environment
        """
        if not self.allow_early_resets and not self.needs_reset:
            raise RuntimeError(
                "Tried to reset an environment before done. If you want to allow early resets, "
                "wrap your env with Monitor(env, path, allow_early_resets=True)"
            )
        self.rewards: list[float] = []
        self.cost_geom_nominal: list[float] = []
        self.cost_geom_proj: list[float] = []
        self.cost_topo: list[float] = []

        self.cmaes_applied_num: list[int] = []
        self.cmaes_applied_success_rate: list[int] = []
        self.needs_reset = False
        for key in self.reset_keywords:
            value = kwargs.get(key)
            if value is None:
                raise ValueError(f"Expected you to pass keyword argument {key} into reset")
            self.current_reset_info[key] = value
        return self.env.reset(**kwargs)

    def step(self, action: ActType) -> tuple[ObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        """
        Step the environment with the given action

        :param action: the action
        :return: observation, reward, terminated, truncated, information
        """
        if self.needs_reset:
            raise RuntimeError("Tried to step environment that needs reset")
        observation, reward, terminated, truncated, info = self.env.step(action)

        self.rewards.append(float(reward))
        self.cost_geom_nominal.append(float(info["cost_geom_nominal"]))

        if "cost_geom_proj" in info:
            self.cost_geom_proj.append(float(info["cost_geom_proj"]))
        self.cost_topo.append(float(info["cost_topo"]))

        self.cmaes_applied_num.append(int(info["cmaes_applied"]))
        if "cmaes_success" in info:
            self.cmaes_applied_success_rate.append(int(info["cmaes_success"]))

        if terminated or truncated:
            self.needs_reset = True
            ep_rew = sum(self.rewards)
            ep_rew_discounted = 0.0
            discount = 1.0
            for reward_step in self.rewards:
                ep_rew_discounted += reward_step * discount
                discount *= self.discount_factor

            cost_geom_nominal = sum(self.cost_geom_nominal)
            cost_geom_proj = sum(self.cost_geom_proj)
            cost_topo = sum(self.cost_topo)

            cmaes_applied_num = sum(self.cmaes_applied_num)
            if cmaes_applied_num > 0:
                cmaes_applied_success_rate_percent = sum(self.cmaes_applied_success_rate) / cmaes_applied_num
            else:
                cmaes_applied_success_rate_percent = 1

            ep_len = len(self.rewards)
            ep_info = {
                "r": round(ep_rew, 6),
                "r_discounted": round(ep_rew_discounted, 6),
                "l": ep_len,
                "t": round(time.time() - self.t_start, 6),
                "cost_geom_nominal": round(cost_geom_nominal, 6),
                "cost_geom_proj": round(cost_geom_proj, 6),
                "cost_topo": round(cost_topo, 6),
                "cmaes_applied_num": cmaes_applied_num,
                "cmaes_applied_success_rate": round(cmaes_applied_success_rate_percent, 6),
            }
            for key in self.info_keywords:
                ep_info[key] = info[key]
            self.episode_returns.append(ep_rew)
            self.episode_lengths.append(ep_len)
            self.episode_times.append(time.time() - self.t_start)
            ep_info.update(self.current_reset_info)
            info["episode"] = ep_info
        self.total_steps += 1
        return observation, reward, terminated, truncated, info

    def close(self) -> None:
        """
        Closes the environment
        """
        super().close()

    def get_total_steps(self) -> int:
        """
        Returns the total number of timesteps

        :return:
        """
        return self.total_steps

    def get_episode_rewards(self) -> list[float]:
        """
        Returns the rewards of all the episodes

        :return:
        """
        return self.episode_returns

    def get_episode_lengths(self) -> list[int]:
        """
        Returns the number of timesteps of all the episodes

        :return:
        """
        return self.episode_lengths

    def get_episode_times(self) -> list[float]:
        """
        Returns the runtime in seconds of all the episodes

        :return:
        """
        return self.episode_times

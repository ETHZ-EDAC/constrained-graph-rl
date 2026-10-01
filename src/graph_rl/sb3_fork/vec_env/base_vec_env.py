# Standard library
from abc import ABC, abstractmethod
from collections.abc import Iterable, Sequence
from typing import Any, Optional, Union

# Third-party
import numpy as np
from gymnasium import spaces
from torch_geometric.data import Data

# Define type aliases here to avoid circular import
# Used when we want to access one or more BaseVecEnv
VecEnvIndices = Union[None, int, Iterable[int]]
# VecEnvObs is what is returned by the reset() method
# it contains the observation for each env
VecEnvObs = dict[int, Data]
# VecEnvStepReturn is what is returned by the step() method
# it contains the observation, reward, done, info for each env
VecEnvStepReturn = tuple[VecEnvObs, Any, Any, list[dict]]


class BaseVecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.

    :param num_envs: Number of environments
    :param observation_space: Observation space
    :param action_space: Action space
    """

    def __init__(
        self,
        num_envs: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
    ):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space
        # store info returned by the reset method
        self.reset_infos: list[dict[str, Any]] = [{} for _ in range(num_envs)]
        # seeds to be used in the next call to env.reset()
        self._seeds: list[Optional[int]] = [None for _ in range(num_envs)]

        # render_modes = self.get_attr("render_mode")
        #
        # self.render_mode = render_modes[0]
        #
        # render_modes = [self.render_mode]
        #
        # self.metadata = {"render_modes": render_modes}

    def _reset_seeds(self) -> None:
        """
        Reset the seeds that are going to be used at the next reset.
        """
        self._seeds = [None for _ in range(self.num_envs)]

    def _reset_options(self) -> None:
        """
        Reset the options that are going to be used at the next reset.
        """
        self._options = [{} for _ in range(self.num_envs)]

    @abstractmethod
    def reset(self) -> VecEnvObs:
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.

        :return: observation
        """
        raise NotImplementedError()

    @abstractmethod
    def step_async(self, actions: np.ndarray) -> None:
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        raise NotImplementedError()

    @abstractmethod
    def step_wait(self) -> VecEnvStepReturn:
        """
        Wait for the step taken with step_async().

        :return: observation, reward, done, information
        """
        raise NotImplementedError()

    @abstractmethod
    def close(self) -> None:
        """
        Clean up the environment's resources.
        """
        raise NotImplementedError()

    def has_attr(self, attr_name: str) -> bool:
        """
        Check if an attribute exists for a vectorized environment.

        :param attr_name: The name of the attribute to check
        :return: True if 'attr_name' exists in all environments
        """
        # Default implementation, will not work with things that cannot be pickled:
        # https://github.com/Stable-Baselines-Team/stable-baselines3-contrib/issues/49
        try:
            self.get_attr(attr_name)
            return True
        except AttributeError:
            return False

    @abstractmethod
    def get_attr(self, attr_name: str, indices: VecEnvIndices = None) -> list[Any]:
        """
        Return attribute from vectorized environment.

        :param attr_name: The name of the attribute whose value to return
        :param indices: Indices of envs to get attribute from
        :return: List of values of 'attr_name' in all environments
        """
        raise NotImplementedError()

    @abstractmethod
    def set_attr(self, attr_name: str, value: Any, indices: VecEnvIndices = None) -> None:
        """
        Set attribute inside vectorized environments.

        :param attr_name: The name of attribute to assign new value
        :param value: Value to assign to `attr_name`
        :param indices: Indices of envs to assign value
        :return:
        """
        raise NotImplementedError()

    @abstractmethod
    def env_method(self, method_name: str, *method_args, indices: VecEnvIndices = None, **method_kwargs) -> list[Any]:
        """
        Call instance methods of vectorized environments.

        :param method_name: The name of the environment method to invoke.
        :param indices: Indices of envs whose method to call
        :param method_args: Any positional arguments to provide in the call
        :param method_kwargs: Any keyword arguments to provide in the call
        :return: List of items returned by the environment's method call
        """
        raise NotImplementedError()

    def step(self, actions: np.ndarray) -> VecEnvStepReturn:
        """
        Step the environments with the given action

        :param actions: the action
        :return: observation, reward, done, information
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode: Optional[str] = None) -> Optional[np.ndarray]:
        """
        Gym environment rendering

        :param mode: the rendering type
        """
        self.env_method("render")

        return None

    def seed(self, seed: Optional[int] = None) -> Sequence[Union[None, int]]:
        """
        Sets the random seeds for all environments, based on a given seed.
        Each individual environment will still get its own seed, by incrementing the given seed.
        WARNING: since gym 0.26, those seeds will only be passed to the environment
        at the next reset.

        :param seed: The random seed. May be None for completely random seeding.
        :return: Returns a list containing the seeds for each individual env.
            Note that all list elements may be None, if the env does not return anything when being seeded.
        """
        if seed is None:
            # To ensure that subprocesses have different seeds,
            # we still populate the seed variable when no argument is passed
            seed = int(np.random.randint(0, np.iinfo(np.uint32).max, dtype=np.uint32))

        self._seeds = [seed + idx for idx in range(self.num_envs)]
        return self._seeds

    def _get_indices(self, indices: VecEnvIndices) -> Iterable[int]:
        """
        Convert a flexibly-typed reference to environment indices to an implied list of indices.

        :param indices: refers to indices of envs.
        :return: the implied list of indices.
        """
        if indices is None:
            indices = range(self.num_envs)
        elif isinstance(indices, int):
            indices = [indices]
        return indices

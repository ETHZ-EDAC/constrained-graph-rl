"""Abstract base classes for RL algorithms."""

# Standard library
import io
import logging
import pathlib
import time
from abc import ABC, abstractmethod
from collections import deque
from collections.abc import Iterable
from typing import Any, Optional, Union

# Third-party
import numpy as np
import torch as th
from gymnasium import spaces

# First-party
from graph_rl.ppo.callbacks import BaseCallback, CallbackList
from graph_rl.sb3_fork.common.ppo_logger import configure_logger, PPOLogger
from graph_rl.sb3_fork.common.utils import (
    check_for_correct_spaces,
    FloatSchedule,
    GymEnv,
    MaybeCallback,
    Schedule,
    TensorDict,
    update_learning_rate,
)
from graph_rl.sb3_fork.common.utils_saving import (
    load_from_zip_file,
    recursive_getattr,
    save_to_zip_file,
)
from graph_rl.sb3_fork.vec_env import (
    BaseVecEnv,
)
from graph_rl.utils.utils import set_random_seed


class BaseOptim(ABC):
    """
    The base of RL algorithms

    :param policy: The policy model to use
    :param env: The environment to learn from
                (if registered in Gym, can be str. Can be None for loading trained models)
    :param learning_rate: learning rate for the optimizer,
        it can be a function of the current progress remaining (from 1 to 0)
    :param policy_kwargs: Additional arguments to be passed to the policy on creation
    :param job_dir: the log location for tensorboard (if None, no logging)
    :param device: Device on which the code should run.
        By default, it will try to use a Cuda compatible device and fallback to cpu
        if it is not possible.
    :param seed: Seed for the pseudo random generators
    """

    observation_space: spaces.Space
    action_space: spaces.Space
    n_envs: int
    lr_schedule: Schedule
    _ppo_logger: PPOLogger

    def __init__(
        self,
        env: GymEnv,
        learning_rate: Union[float, Schedule],
        policy_kwargs: Optional[dict[str, Any]] = None,
        job_dir: Optional[pathlib.Path] = None,
        logger: Optional[logging.Logger] = None,
        device: Union[th.device, str] = "auto",
        seed: Optional[int] = None,
    ) -> None:

        self.device = device
        self.stdout_logger = logger
        self.policy_kwargs = {} if policy_kwargs is None else policy_kwargs

        self.num_timesteps = 0
        # Used for updating schedules
        self._total_timesteps = 0
        # Used for computing fps, it is updated at each call of learn()
        self._num_timesteps_at_start = 0
        self.seed = seed
        self.start_time = 0.0
        self.learning_rate = learning_rate
        self.job_dir = job_dir
        self._last_obs = None  # type: Optional[Union[np.ndarray, dict[str, np.ndarray]]]
        self._last_episode_starts = None  # type: Optional[np.ndarray]
        self._episode_num = 0
        # Track the training progress remaining (from 1 to 0)
        # this is used to update the learning rate
        self._current_progress_remaining = 1.0
        # Buffers for logging
        self.ep_info_buffer = None  # type: Optional[deque]
        self.ep_success_buffer = None  # type: Optional[deque]
        # For logging (and TD3 delayed updates)
        self._n_updates = 0  # type: int
        # Whether the user passed a custom ppo_logger or not
        self.env: Optional[BaseVecEnv] = None

        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.n_envs = env.num_envs
        self.env = env
        self._ppo_logger = configure_logger(self.stdout_logger, self.job_dir)

    @abstractmethod
    def _setup_model(self) -> None:
        """Create networks, buffer and optimizers."""
        raise NotImplementedError()

    @property
    def ppo_logger(self) -> PPOLogger:
        """Getter for the ppo_logger object."""
        return self._ppo_logger

    def _setup_lr_schedule(self) -> None:
        """Transform to callable if needed."""
        self.lr_schedule = FloatSchedule(self.learning_rate)

    def _update_current_progress_remaining(self, num_timesteps: int, total_timesteps: int) -> None:
        """
        Compute current progress remaining (starts from 1 and ends to 0)

        :param num_timesteps: current number of timesteps
        :param total_timesteps:
        """
        self._current_progress_remaining = 1.0 - float(num_timesteps) / float(total_timesteps)

    def _update_learning_rate(self, optimizers: Union[list[th.optim.Optimizer], th.optim.Optimizer]) -> None:
        """
        Update the optimizers learning rate using the current learning rate schedule
        and the current progress remaining (from 1 to 0).

        :param optimizers:
            An optimizer or a list of optimizers.
        """
        lr = self.lr_schedule(self._current_progress_remaining)
        # Log the current learning rate
        self.ppo_logger.record("train/learning_rate", lr)

        if not isinstance(optimizers, list):
            optimizers = [optimizers]
        for optimizer in optimizers:
            update_learning_rate(optimizer, lr)

    def _excluded_save_params(self) -> list[str]:
        """
        Returns the names of the parameters that should be excluded from being
        saved by pickling. E.g. replay buffers are skipped by default
        as they take up a lot of space. PyTorch variables should be excluded
        with this so they can be stored with ``th.save``.

        :return: List of parameters that should be excluded from being saved with pickle.
        """
        return [
            "policy",
            "device",
            "env",
            "rollout_buffer",
            "_ppo_logger",
        ]

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        """
        Get the name of the torch variables that will be saved with
        PyTorch ``th.save``, ``th.load`` and ``state_dicts`` instead of the default
        pickling strategy. This is to handle device placement correctly.

        Names can point to specific variables under classes, e.g.
        "policy.optimizer" would point to ``optimizer`` object of ``self.policy``
        if this object.

        :return:
            List of Torch variables whose state dicts to save (e.g. th.nn.Modules),
            and list of other Torch variables to store with ``th.save``.
        """
        state_dicts = ["policy"]

        return state_dicts, []

    def _init_callback(
        self,
        callback: MaybeCallback,
    ) -> BaseCallback:
        """
        :param callback: Callback(s) called at every step with state of the algorithm.
        :param progress_bar: Display a progress bar using tqdm and rich.
        :return: A hybrid callback calling `callback` and performing evaluation.
        """
        # Convert a list of callbacks into a callback
        if isinstance(callback, list):
            callback = CallbackList(callback)

        callback.init_callback(self)
        return callback

    def _setup_learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
    ) -> tuple[int, BaseCallback]:
        """
        Initialize different variables needed for training.

        :param total_timesteps: The total number of samples (env steps) to train on
        :param callback: Callback(s) called at every step with state of the algorithm.
        :return: Total timesteps and callback(s)
        """
        self.start_time = time.time_ns()

        self.ep_info_buffer = deque(maxlen=1000)
        self.ep_success_buffer = deque(maxlen=1000)
        self.ep_terminal_reward = deque(maxlen=1000)

        self.num_timesteps = 0
        self._episode_num = 0

        self._total_timesteps = total_timesteps
        self._num_timesteps_at_start = self.num_timesteps

        assert self.env is not None
        self._last_obs = self.env.reset()  # type: ignore[assignment]
        self._last_episode_starts = np.ones((self.env.num_envs,), dtype=bool)

        # Create eval callback if needed
        callback = self._init_callback(callback)

        return total_timesteps, callback

    def _update_info_buffer(self, infos: list[dict[str, Any]], dones: Optional[np.ndarray] = None) -> None:
        """
        Retrieve reward, episode length, episode success and update the buffer
        if using Monitor wrapper or a GoalEnv.

        :param infos: List of additional information about the transition.
        :param dones: Termination signals
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        if dones is None:
            dones = np.array([False] * len(infos))
        for idx, info in enumerate(infos):
            maybe_ep_info = info.get(
                "episode"
            )  # Is written in monitor wrapper!! and only for terminated / truncated envs.
            maybe_is_success = info.get("is_success")

            if maybe_ep_info is not None and dones[idx]:
                self.ep_info_buffer.extend([maybe_ep_info])
                self.ep_success_buffer.append(maybe_is_success)
                if "terminal_reward" in info:  # we do it here because we only care about terminal iso ratio
                    self.ep_terminal_reward.append(info.get("terminal_reward"))

    @abstractmethod
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
    ):
        raise NotImplementedError()

    def set_random_seed(self, seed: Optional[int] = None, cuda_deterministic=False) -> None:
        """
        Set the seed of the pseudo-random generators
        (python, numpy, pytorch, gym, action_space)

        :param seed:
        :params cuda_deterministic: bool
        """
        if seed is None:
            return
        set_random_seed(seed, using_cuda=self.device == "cuda", cuda_deterministic=cuda_deterministic)
        self.action_space.seed(seed)
        # self.env is always a BaseVecEnv
        if self.env is not None:
            self.env.seed(seed)

    def set_parameters(
        self,
        load_path_or_dict: Union[str, TensorDict],
        exact_match: bool = True,
        device: Union[th.device, str] = "auto",
    ) -> None:
        """
        Load parameters from a given zip-file or a nested dictionary containing parameters for
        different modules (see ``get_parameters``).

        :param load_path_or_dict: Location of the saved data (path or file-like, see ``save``), or a nested
            dictionary containing nn.Module parameters used by the policy. The dictionary maps
            object names to a state-dictionary returned by ``torch.nn.Module.state_dict()``.
        :param exact_match: If True, the given parameters should include parameters for each
            module and each of their parameters, otherwise raises an Exception. If set to False, this
            can be used to update only specific parameters.
        :param device: Device on which the code should run.
        """
        params = {}
        if isinstance(load_path_or_dict, dict):
            params = load_path_or_dict
        else:
            _, params, _ = load_from_zip_file(load_path_or_dict, device=device, load_data=False)

        # Keep track which objects were updated.
        # `_get_torch_save_params` returns [params, other_pytorch_variables].
        # We are only interested in former here.
        objects_needing_update = set(self._get_torch_save_params()[0])
        updated_objects = set()

        for name in params:
            attr = None
            try:
                attr = recursive_getattr(self, name)
            except Exception as e:
                # What errors recursive_getattr could throw? KeyError, but
                # possible something else too (e.g. if key is an int?).
                # Catch anything for now.
                raise ValueError(f"Key {name} is an invalid object name.") from e

            if isinstance(attr, th.optim.Optimizer):
                # Optimizers do not support "strict" keyword...
                # Seems like they will just replace the whole
                # optimizer state with the given one.
                # On top of this, optimizer state-dict
                # seems to change (e.g. first ``optim.step()``),
                # which makes comparing state dictionary keys
                # invalid (there is also a nesting of dictionaries
                # with lists with dictionaries with ...), adding to the
                # mess.
                #
                # TL;DR: We might not be able to reliably say
                # if given state-dict is missing keys.
                #
                # Solution: Just load the state-dict as is, and trust
                # the user has provided a sensible state dictionary.
                attr.load_state_dict(params[name])  # type: ignore[arg-type]
            else:
                # Assume attr is th.nn.Module
                attr.load_state_dict(params[name], strict=exact_match)
            updated_objects.add(name)

        if exact_match and updated_objects != objects_needing_update:
            raise ValueError(
                "Names of parameters do not match agents' parameters: "
                f"expected {objects_needing_update}, got {updated_objects}"
            )

    @classmethod
    def load(  # noqa: C901
        cls,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        env: Optional[GymEnv] = None,
        device: Union[th.device, str] = "auto",
        custom_objects: Optional[dict[str, Any]] = None,
        print_system_info: bool = False,
        force_reset: bool = True,
        **kwargs,
    ):
        """
        Load the model from a zip-file.
        Warning: ``load`` re-creates the model from scratch, it does not update it in-place!
        For an in-place load use ``set_parameters`` instead.

        :param path: path to the file (or a file-like) where to
            load the agent from
        :param env: the new environment to run the loaded model on
            (can be None if you only need prediction from a trained model) has priority over any saved environment
        :param device: Device on which the code should run.
        :param custom_objects: Dictionary of objects to replace
            upon loading. If a variable is present in this dictionary as a
            key, it will not be deserialized and the corresponding item
            will be used instead. Similar to custom_objects in
            ``keras.models.load_model``. Useful when you have an object in
            file that can not be deserialized.
        :param print_system_info: Whether to print system info from the saved model
            and the current system info (useful to debug loading issues)
        :param force_reset: Force call to ``reset()`` before training
            to avoid unexpected behavior.
            See https://github.com/DLR-RM/stable-baselines3/issues/597
        :param kwargs: extra arguments to change the model when loading
        :return: new model instance with loaded parameters
        """

        data, params, pytorch_variables = load_from_zip_file(
            path,
            device=device,
            custom_objects=custom_objects,
            print_system_info=print_system_info,
        )

        assert data is not None, "No data found in the saved file"
        assert params is not None, "No params found in the saved file"

        # Remove stored device information and replace with ours
        if "policy_kwargs" in data:
            if "device" in data["policy_kwargs"]:
                del data["policy_kwargs"]["device"]

        if "policy_kwargs" in kwargs and kwargs["policy_kwargs"] != data["policy_kwargs"]:
            raise ValueError(
                f"The specified policy kwargs do not equal the stored policy kwargs."
                f"Stored kwargs: {data['policy_kwargs']}, specified kwargs: {kwargs['policy_kwargs']}"
            )

        if "observation_space" not in data or "action_space" not in data:
            raise KeyError("The observation_space and action_space were not given, can't verify new environments")

        if env is not None:
            # Wrap first if needed
            # Check if given env is valid
            check_for_correct_spaces(env, data["observation_space"], data["action_space"])
            # Discard `_last_obs`, this will force the env to reset before training
            # See issue https://github.com/DLR-RM/stable-baselines3/issues/597
            if force_reset and data is not None:
                data["_last_obs"] = None
            # `n_envs` must be updated. See issue https://github.com/DLR-RM/stable-baselines3/issues/1018
            if data is not None:
                data["n_envs"] = env.num_envs
        else:
            # Use stored env, if one exists. If not, continue as is (can be used for predict)
            if "env" in data:
                env = data["env"]

        model = cls(
            env=env,
            device=device,
            **data,
        )

        # load parameters
        model.__dict__.update(data)
        model.__dict__.update(kwargs)
        model._setup_model()

        model.set_parameters(params, exact_match=True, device=device)

        return model

    def get_parameters(self) -> dict[str, dict]:
        """
        Return the parameters of the agent. This includes parameters from different networks, e.g.
        critics (value functions) and policies (pi functions).

        :return: Mapping of from names of the objects to PyTorch state-dicts.
        """
        state_dicts_names, _ = self._get_torch_save_params()
        params = {}
        for name in state_dicts_names:
            attr = recursive_getattr(self, name)
            # Retrieve state dict
            params[name] = attr.state_dict()
        return params

    def save(
        self,
        path: Union[str, pathlib.Path, io.BufferedIOBase],
        exclude: Optional[Iterable[str]] = None,
        include: Optional[Iterable[str]] = None,
    ) -> None:
        """
        Save all the attributes of the object and the model parameters in a zip-file.

        :param path: path to the file where the rl agent should be saved
        :param exclude: name of parameters that should be excluded in addition to the default ones
        :param include: name of parameters that might be excluded but should be included anyway
        """
        # Copy parameter list so we don't mutate the original dict
        data = self.__dict__.copy()

        # Exclude is union of specified parameters (if any) and standard exclusions
        if exclude is None:
            exclude = []
        exclude = set(exclude).union(self._excluded_save_params())

        # Do not exclude params if they are specifically included
        if include is not None:
            exclude = exclude.difference(include)

        state_dicts_names, torch_variable_names = self._get_torch_save_params()
        all_pytorch_variables = state_dicts_names + torch_variable_names
        for torch_var in all_pytorch_variables:
            # We need to get only the name of the top most module as we'll remove that
            var_name = torch_var.split(".")[0]
            # Any params that are in the save vars must not be saved by data
            exclude.add(var_name)

        # Remove parameter entries of parameters which are to be excluded
        for param_name in exclude:
            data.pop(param_name, None)

        # Build dict of torch variables
        pytorch_variables = None
        if torch_variable_names is not None:
            pytorch_variables = {}
            for name in torch_variable_names:
                attr = recursive_getattr(self, name)
                pytorch_variables[name] = attr

        # Build dict of state_dicts
        params_to_save = self.get_parameters()

        save_to_zip_file(path, data=data, params=params_to_save, pytorch_variables=pytorch_variables)

    def dump_logs(self) -> None:
        """
        Write log data. (Implemented by OffPolicyAlgorithm and OnPolicyAlgorithm)
        """
        raise NotImplementedError()

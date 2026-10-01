# Standard library
import glob
import math
import os
import pathlib
import random
from collections import deque
from typing import Callable, Union

# Third-party
import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces

# Check if tensorboard is available for pytorch
from torch_geometric.data import Batch, Data

TensorDict = dict[str, th.Tensor]
MaybeCallback = Union[None, Callable, list["BaseCallback"], "BaseCallback"]
PyTorchObs = Union[Data, Batch]
Schedule = Callable[[float], float]
GymEnv = Union[gym.Env, "BaseVecEnv"]


# From stable baselines
def explained_variance(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """
    Computes fraction of variance that ypred explains about y.
    Returns 1 - Var[y-ypred] / Var[y]

    interpretation:
        ev=0  =>  might as well have predicted zero
        ev=1  =>  perfect prediction
        ev<0  =>  worse than just predicting zero

    :param y_pred: the prediction
    :param y_true: the expected value
    :return: explained variance of ypred and y
    """
    assert y_true.ndim == 1 and y_pred.ndim == 1
    var_y = np.var(y_true)
    return np.nan if var_y == 0 else float(1 - np.var(y_true - y_pred) / var_y)


def update_learning_rate(optimizer: th.optim.Optimizer, learning_rate: float) -> None:
    """
    Update the learning rate for a given optimizer.
    Useful when doing linear schedule.

    :param optimizer: Pytorch optimizer
    :param learning_rate: New learning rate value
    """
    for param_group in optimizer.param_groups:
        param_group["lr"] = learning_rate


class FloatSchedule:
    """
    Wrapper that ensures the output of a Schedule is cast to float.
    Can wrap either a constant value or an existing callable Schedule.

    :param value_schedule: Constant value or callable schedule
            (e.g. LinearSchedule, ConstantSchedule)
    """

    def __init__(self, value_schedule: Union[Schedule, float]):
        if isinstance(value_schedule, FloatSchedule):
            self.value_schedule: Schedule = value_schedule.value_schedule
        elif isinstance(value_schedule, (float, int)):
            self.value_schedule = ConstantSchedule(float(value_schedule))
        else:
            assert callable(
                value_schedule
            ), f"The learning rate schedule must be a float or a callable, not {value_schedule}"
            self.value_schedule = value_schedule

    def __call__(self, progress_remaining: float) -> float:
        # Cast to float to avoid unpickling errors to enable weights_only=True, see GH#1900
        # Some types are have odd behaviors when part of a Schedule, like numpy floats
        return float(self.value_schedule(progress_remaining))

    def __repr__(self) -> str:
        return f"FloatSchedule({self.value_schedule})"


class WarmupThenDecaySchedule:
    """
    Warmup schedule followed by a scalar decay schedule.

    """

    def __init__(
        self,
        base_lr: float,
        total_steps: int,
        warmup_steps: int,
        warmup_start_factor: float = 0.1,
        decay_type: str = "cosine",
        end_lr: float = 0.0,
    ):
        if total_steps <= 0:
            raise ValueError(f"total_steps must be > 0, got {total_steps}")
        if warmup_steps < 0 or warmup_steps > total_steps:
            raise ValueError(f"warmup_steps must be in [0, total_steps], got {warmup_steps}")
        if base_lr <= 0:
            raise ValueError(f"base_lr must be > 0, got {base_lr}")

        self.total_steps = int(total_steps)
        self.base_lr = float(base_lr)
        self.warmup_steps = int(warmup_steps)
        self.warmup_start_factor = max(float(warmup_start_factor), 1e-6)
        self.decay_type = str(decay_type)
        self.end_lr = float(end_lr)

        decay_type = self.decay_type.lower()
        if decay_type not in {"cosine", "linear"}:
            raise ValueError(f"Unsupported decay_type: {self.decay_type}")
        self.decay_type = decay_type

    def __call__(self, progress_remaining: float) -> float:
        progress_remaining = float(progress_remaining)
        progress_remaining = min(max(progress_remaining, 0.0), 1.0)
        step = int(round((1.0 - progress_remaining) * self.total_steps))
        step = min(max(step, 0), self.total_steps)

        if self.warmup_steps > 0 and step <= self.warmup_steps:
            warmup_fraction = step / self.warmup_steps
            factor = self.warmup_start_factor + (1.0 - self.warmup_start_factor) * warmup_fraction
            return float(self.base_lr * factor)

        decay_steps = max(1, self.total_steps - self.warmup_steps)
        decay_step = min(max(step - self.warmup_steps, 0), decay_steps)
        decay_fraction = decay_step / decay_steps
        if self.decay_type == "cosine":
            decay_scale = 0.5 * (1.0 + math.cos(math.pi * decay_fraction))
            return float(self.end_lr + (self.base_lr - self.end_lr) * decay_scale)
        return float(self.base_lr + (self.end_lr - self.base_lr) * decay_fraction)

    def __repr__(self) -> str:
        return (
            "WarmupThenDecaySchedule("
            f"base_lr={self.base_lr}, total_steps={self.total_steps}, warmup_steps={self.warmup_steps}, "
            f"warmup_start_factor={self.warmup_start_factor}, decay_type={self.decay_type}, end_lr={self.end_lr})"
        )


class ConstantSchedule:
    """
    Constant schedule that always returns the same value.
    Useful for fixed learning rates or clip ranges.

    :param val: constant value
    """

    def __init__(self, val: float):
        self.val = val

    def __call__(self, _: float) -> float:
        return self.val

    def __repr__(self) -> str:
        return f"ConstantSchedule(val={self.val})"


def get_latest_run_id(log_path: pathlib.Path, log_name: str = "") -> int:
    """
    Returns the latest run number for the given log name and log path,
    by finding the greatest number in the directories.

    :param log_path: Path to the log folder containing several runs.
    :param log_name: Name of the experiment. Each run is stored
        in a folder named ``log_name_1``, ``log_name_2``, ...
    :return: latest run number
    """
    max_run_id = 0
    for path in glob.glob(os.path.join(log_path, f"{glob.escape(log_name)}_[0-9]*")):
        file_name = path.split(os.sep)[-1]
        ext = file_name.split("_")[-1]
        if log_name == "_".join(file_name.split("_")[:-1]) and ext.isdigit() and int(ext) > max_run_id:
            max_run_id = int(ext)
    return max_run_id


def check_for_correct_spaces(env: GymEnv, observation_space: spaces.Space, action_space: spaces.Space) -> None:
    """
    Checks that the environment has same spaces as provided ones. Used by BaseOptim to check if
    spaces match after loading the model with given env.
    Checked parameters:
    - observation_space
    - action_space

    :param env: Environment to check for valid spaces
    :param observation_space: Observation space to check against
    :param action_space: Action space to check against
    """
    pass


def obs_as_data_batch(obs: dict[int, Data]) -> Batch:
    """
    Moves the observation to the given device.

    :param obs:
    :return: PyTorch tensor of the observation on a desired device.
    """

    return Batch.from_data_list(list(obs.values()))


def safe_mean(arr: Union[np.ndarray, list, deque]) -> float:
    """
    Compute the mean of an array if there is at least one element.
    For empty array, return NaN. It is used for logging only.

    :param arr: Numpy array or list of values
    :return:
    """
    return np.nan if len(arr) == 0 else float(np.mean(arr))  # type: ignore[arg-type]

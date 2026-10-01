"""Minimal logging utilities for PPO training."""

from __future__ import annotations

# Standard library
import logging
import os
import pathlib
from typing import Any, Optional, Sequence, Tuple

# Third-party
import numpy as np
import torch as th
from torch.utils.tensorboard import SummaryWriter

# First-party
from graph_rl.sb3_fork.common.utils import get_latest_run_id

__all__ = ["PPOLogger"]


class PPOLogger:
    """Simple ppo_logger that supports console and TensorBoard outputs."""

    def __init__(
        self, log_dir: Optional[str], logger: Optional[logging.Logger] = None, use_tensorboard: bool = True
    ) -> None:
        self.log_dir = log_dir
        self.logger = logger
        self.use_tensorboard = use_tensorboard

        self._records: dict[str, Any] = {}
        self._excludes: dict[str, Tuple[str, ...]] = {}
        self._writer: Optional[SummaryWriter] = None

        if self.use_tensorboard:
            if log_dir is None:
                raise ValueError("TensorBoard logging requires a log directory.")
            os.makedirs(log_dir, exist_ok=True)
            self._writer = SummaryWriter(log_dir=log_dir)

    def record(self, key: str, value: Any, exclude: Optional[Sequence[str] | str] = None) -> None:
        """Record a value for later emission."""
        if value is None:
            return
        self._records[key] = value
        if exclude is None:
            self._excludes[key] = ()
        elif isinstance(exclude, str):
            self._excludes[key] = (exclude,)
        else:
            self._excludes[key] = tuple(exclude)

    def dump(self, step: int = 0, enable_tensorboard: bool = True) -> None:
        """Emit all recorded values and reset the internal buffers."""
        if not self._records:
            return

        keys = [k for k in sorted(self._records) if "stdout" not in self._excludes.get(k, ())]
        max_key_len = max(len(k) for k in keys) if keys else 0

        border = str("".join(["-"] * (max_key_len + 15)))
        self.logger.info(border)
        for key in sorted(self._records):
            if "stdout" in self._excludes.get(key, ()):  # pragma: no branch - tiny dict
                continue
            value = self._format_value(self._records[key])
            self.logger.info(f"| {key:<{max_key_len}} | {value}")
        self.logger.info(border)

        if self._writer is not None and enable_tensorboard:
            for key, value in self._records.items():
                if "tensorboard" in self._excludes.get(key, ()):  # pragma: no branch - tiny dict
                    continue
                self._write_tensorboard(key, value, step)
            self._writer.flush()

        self._records.clear()
        self._excludes.clear()

    def close(self) -> None:
        """Release any TensorBoard resources."""
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return "nan"
            if np.isinf(value):
                return "inf" if value > 0 else "-inf"
            return f"{float(value):.3g}"
        if isinstance(value, (int, np.integer)):
            return str(int(value))
        return str(value)

    def _write_tensorboard(self, key: str, value: Any, step: int) -> None:
        if self._writer is None:
            return
        if isinstance(value, str):
            self._writer.add_text(key, value, step)
            return
        if isinstance(value, (int, np.integer)):
            self._writer.add_scalar(key, int(value), step)
            return
        if isinstance(value, (float, np.floating)):
            if not np.isnan(value) and not np.isinf(value):
                self._writer.add_scalar(key, float(value), step)
            return
        if isinstance(value, th.Tensor):
            tensor = value.detach().cpu()
            if tensor.numel() == 1:
                scalar = tensor.item()
                if isinstance(scalar, float) and (np.isnan(scalar) or np.isinf(scalar)):
                    return
                self._writer.add_scalar(key, scalar, step)
            else:
                self._writer.add_histogram(key, tensor, step)
            return
        if isinstance(value, np.ndarray):
            if value.size == 1:
                scalar = float(value.reshape(-1)[0])
                if np.isnan(scalar) or np.isinf(scalar):
                    return
                self._writer.add_scalar(key, scalar, step)
            else:
                self._writer.add_histogram(key, value, step)
            return
        # Fallback to string representation for unsupported types.
        self._writer.add_text(key, str(value), step)


def configure_logger(
    logger: Optional[logging.Logger] = None,
    tensorboard_log: Optional[pathlib.Path] = None,
) -> PPOLogger:
    """
    Configure the ppo_logger's outputs.

    :param logger: Logger object
    :param tensorboard_log: the log location for tensorboard (if None, no logging)
    :return: The ppo_logger object

    """
    use_tensorboard = tensorboard_log is not None
    save_path: Optional[str] = None
    tb_log_name = "tensorboard"

    if use_tensorboard:
        save_path = os.path.join(tensorboard_log, f"{tb_log_name}")

    ppo_logger = PPOLogger(log_dir=save_path, logger=logger, use_tensorboard=use_tensorboard)

    return ppo_logger

"""Utilities that connect Hydra runtime behaviour with the custom logging stack."""

from __future__ import annotations

# Standard library
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

# Third-party
from logging_mod.logger import get_logger, setup_job_dir
from omegaconf import DictConfig, OmegaConf


def _ifelse(condition: Any, when_true: Any, when_false: Any) -> Any:
    return when_true if bool(condition) else when_false


OmegaConf.register_new_resolver("ifelse", _ifelse, replace=True)


def configure_jax_setup(cfg: DictConfig) -> None:
    """Configure JAX settings based on the Hydra configuration.

    Args:
        cfg: DictConfig: The Hydra configuration dictionary.
    """
    # Standard library
    from sys import platform

    # Third-party
    import jax

    jax.config.update("jax_enable_x64", cfg.get("jax_enable_x64"))
    jax.config.update("jax_debug_nans", cfg.get("jax_debug_nans"))
    if platform == "linux" or platform == "linux2":
        # jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")
        jax.config.update("jax_persistent_cache_min_entry_size_bytes", 0)
        jax.config.update("jax_persistent_cache_min_compile_time_secs", 0)
        # jax.config.update("jax_persistent_cache_enable_xla_caches", "xla_gpu_per_fusion_autotune_cache_dir")
        # jax.config.update("jax_compilation_cache_dir", "/tmp/jax_cache")


def configure_job_logging(
    job_dir: Path,
    job_cfg: DictConfig,
) -> Path:
    """Configure logging for the job under the Hydra-created output directory.

    Args:
        job_dir: Path: The Hydra-created job directory.
        job_cfg: DictConfig: The job configuration dictionary from hydra.

    Returns:
        Path: The configured job directory.
    """

    logging_cfg = job_cfg.get("logging", {})

    verbosity_console = int(logging_cfg.get("verbosity_console", 3))
    verbosity_logfile = int(logging_cfg.get("verbosity_logfile", 3))
    overwrite_job_dir = bool(logging_cfg.get("overwrite_job_dir", True))

    configured_dir = setup_job_dir(
        parent_dir=job_dir.parent,
        job_id=Path(job_dir.name),
        verbosity_console=verbosity_console,
        verbosity_logfile=verbosity_logfile,
        overwrite_job_dir=overwrite_job_dir,
    )
    base_logger = get_logger("base")
    sys.stdout = StreamToLogger(base_logger, logging.INFO)
    sys.stderr = StreamToLogger(base_logger, logging.ERROR)

    return configured_dir


class StreamToLogger:
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self._buffer = ""

    def write(self, message):
        # stdout can be written in chunks, so buffer until newline
        message = message.replace("\r", "")

        self._buffer += message
        while "\n" in self._buffer:
            line, self._buffer = self._buffer.split("\n", 1)
            if line.strip():
                self.logger.log(self.level, line)

    def flush(self):
        if self._buffer:
            self.logger.log(self.level, self._buffer.strip())
            self._buffer = ""


class LogFilter(logging.Filter):
    """Allow only log records emitted by the demo script itself."""

    def __init__(self, name_to_keep: str) -> None:
        super().__init__()
        self._name_to_keep = name_to_keep

    def filter(self, record: logging.LogRecord) -> bool:
        return record.name in (self._name_to_keep, "base")


def disable_loggers_except_current(logger: logging.Logger) -> None:
    """Disable all loggers except the one with the given name.

    Args:
        logger: logging.PPOLogger: The ppo_logger to keep enabled.
    """
    only_demo_filter = LogFilter(logger.name)
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for handler in root_logger.handlers:
        handler.addFilter(only_demo_filter)

    for name, other_logger in logging.root.manager.loggerDict.items():
        if name != logger.name and name != "base" and isinstance(other_logger, logging.Logger):
            other_logger.disabled = True

    logger.disabled = False
    logger.propagate = True


def save_src_snapshot(
    job_dir: Path,
) -> None:
    """


    :param job_dir:
    """
    cwd = os.getcwd()
    src_dir = Path(cwd) / "src"
    dest_dir = job_dir / "src_snapshot"

    # copy src (skip if missing)
    if os.path.exists(src_dir):
        shutil.copytree(src_dir, dest_dir, dirs_exist_ok=True)

    # save current git commit hash (helps trace code provenance)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=src_dir).decode().strip()
    with open(job_dir / "git_commit.txt", "w") as f:
        f.write(commit + "\n")

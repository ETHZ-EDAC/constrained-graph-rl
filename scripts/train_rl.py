"""Rule 1 only reinforcement-learning demo using Stable-Baselines3 and the PlanarGraphEnv."""

from __future__ import annotations

# Standard library
import os
import warnings
from pathlib import Path

# Third-party
import hydra
import torch as th
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# First-party
from graph_rl.utils import set_grammar_constraints
from graph_rl.utils.hydra_helpers import configure_jax_setup, configure_job_logging, save_src_snapshot


def configure_runtime_env() -> None:
    """
    Configure runtime settings that should be applied before importing JAX in subprocesses.
    """
    # Limit CPU oversubscription per process when using many envs.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    xla_flags = os.environ.get("XLA_FLAGS", "")
    xla_thread_flags = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"
    if xla_thread_flags not in xla_flags:
        os.environ["XLA_FLAGS"] = (xla_flags + " " + xla_thread_flags).strip()


@hydra.main(config_path="../conf", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    warnings.simplefilter("error", RuntimeWarning)

    hydra_cfg = HydraConfig.get()
    job_dir = Path(hydra_cfg.runtime.output_dir)
    job_cfg = cfg.get("job", {})
    job_dir = configure_job_logging(job_dir, job_cfg)
    set_grammar_constraints(cfg.constraints)
    configure_jax_setup(cfg)

    save_src_snapshot(job_dir)
    # First-party
    from graph_rl.ppo.train_logic import train

    train(cfg, job_dir)


if __name__ == "__main__":
    configure_runtime_env()
    main()

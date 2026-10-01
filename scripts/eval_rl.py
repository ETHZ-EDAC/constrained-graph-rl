""""""

from __future__ import annotations

# Standard library
import argparse
import os

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["JAX_PLATFORM_NAME"] = "cpu"
import warnings
from functools import partial
from pathlib import Path

# Third-party
import hydra
import jax
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig

# First-party
from graph_rl.utils import set_grammar_constraints
from graph_rl.utils.hydra_helpers import configure_job_logging


@hydra.main(config_path="../outputs/2026-01-19/23-14-02/.hydra", config_name="config", version_base="1.2")
def main(cfg: DictConfig) -> None:
    warnings.simplefilter("error", RuntimeWarning)

    cfg.policy.PPO.cuda_deterministic = False
    hydra_cfg = HydraConfig.get()
    job_dir = Path(hydra_cfg.runtime.output_dir)
    job_cfg = cfg.get("job", {})
    job_dir = configure_job_logging(job_dir, job_cfg)
    set_grammar_constraints(cfg.constraints)
    jax.config.update("jax_enable_x64", cfg.get("jax_enable_x64"))

    # First-party
    from graph_rl.ppo.eval import evaluate

    step = 1850
    rl_zip = f"ppo_iter_{step}/model"
    terminal_rewards, terminal_reward_mean, terminal_reward_std = evaluate(cfg, job_dir, rl_zip)
    print(f"Terminal rewards per episode: {terminal_rewards}")
    print(f"Terminal reward mean: {terminal_reward_mean}")
    print(f"Terminal reward std: {terminal_reward_std}")


if __name__ == "__main__":
    os.chdir("../")
    main()

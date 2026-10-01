# Standard library
from pathlib import Path

# Third-party
import numpy as np
import torch as th
from hydra.core.hydra_config import HydraConfig
from logging_mod.logger import get_logger
from omegaconf import DictConfig

# First-party
from graph_rl.ppo.actor_critic_policy import ActorCriticPolicy
from graph_rl.ppo.evaluation import evaluate_policy
from graph_rl.ppo.graph_rollout_buffer import GraphRolloutBuffer
from graph_rl.ppo.layers import GraphFeatureExtractor
from graph_rl.ppo.ppo import PPO
from graph_rl.sb3_fork.common.utils_saving import load_from_zip_file
from graph_rl.sb3_fork.vec_env import SequentialVecEnv, SubprocVecEnv
from graph_rl.sb3_fork.vec_env.env_util import make_rule1_env, make_vec_env
from graph_rl.utils import PARAMS, set_grammar_constraints

logger = get_logger()


def _resolve_device(cfg: DictConfig) -> str:
    if th.cuda.is_available() and cfg.policy.meta.enable_cuda:
        return "cuda"
    return "cpu"


def _checkpoint_max_vertex_degree(checkpoint_path: Path, device: str) -> int | None:
    _, params, _ = load_from_zip_file(checkpoint_path, device=device, load_data=False)
    if params is None or "policy" not in params:
        return None
    weight = params["policy"].get("features_extractor.deg_embed.weight")
    if weight is None:
        return None
    return int(weight.shape[0]) - 1


def _sync_constraints_with_checkpoint(cfg: DictConfig, checkpoint_path: Path, device: str) -> None:
    if "constraints" in cfg:
        params = set_grammar_constraints(cfg.constraints)
    else:
        params = PARAMS

    checkpoint_max_degree = _checkpoint_max_vertex_degree(checkpoint_path, device)
    if checkpoint_max_degree is None:
        return
    if checkpoint_max_degree != int(params.max_vertex_degree):
        raise RuntimeError(
            "Checkpoint/config max_vertex_degree mismatch: "
            f"checkpoint expects {checkpoint_max_degree} "
            f"(deg_embed rows={checkpoint_max_degree + 1}), "
            f"but the active config has {params.max_vertex_degree}. "
            "Point scripts/eval_rl.py at the .hydra directory from the same training run "
            "as the checkpoint, or use a checkpoint from the configured run."
        )


def _build_eval_model(cfg: DictConfig, job_dir: Path, device: str) -> tuple[PPO, SequentialVecEnv]:
    logger.info(f"Using {device} device")

    vec_env = make_vec_env(
        make_rule1_env,
        cfg.policy.env.num_envs,
        seed=cfg.policy.PPO.seed,
        start_index=0,
        env_kwargs=dict(job_dir=job_dir, cfg=cfg),
        vec_env_cls=SequentialVecEnv,
    )

    feature_extractor_kwargs = dict(cfg.policy.feature_extractor)

    model = PPO(
        vec_env,
        **cfg.policy.PPO,
        logger=logger,
        job_dir=job_dir,
        device=device,
        policy_kwargs=dict(
            features_extractor_class=GraphFeatureExtractor,
            features_extractor_kwargs=feature_extractor_kwargs,
            distribution_kwargs=cfg.policy.distribution,
            actor_critic_kwargs=cfg.policy.actor_critic,
        ),
        action_projector_kwargs=cfg.policy.action_projector,
    )
    return model, vec_env


def _run_policy_evaluation(
    cfg: DictConfig,
    job_dir: Path,
    model: PPO,
    vec_env: SequentialVecEnv,
    *,
    render_every_successful_step: bool = False,
) -> tuple[list[float | None], float, float]:
    env = make_rule1_env(job_dir, cfg, env_name="eval")
    env.reset(seed=cfg.policy.PPO.seed)

    try:
        _, _, terminal_rewards = evaluate_policy(
            model,
            env,
            logger=logger,
            n_eval_episodes=10,
            render=True,
            render_path=Path("test"),
            render_every_successful_step=render_every_successful_step,
            show_step_render=True,
            deterministic=False,
            return_episode_rewards=True,
            return_terminal_rewards=True,
        )
        valid_terminal_rewards = [reward for reward in terminal_rewards if reward is not None]
        terminal_reward_mean = float(np.mean(valid_terminal_rewards)) if valid_terminal_rewards else float("nan")
        terminal_reward_std = float(np.std(valid_terminal_rewards)) if valid_terminal_rewards else float("nan")
        logger.info(f"Eval terminal rewards: {terminal_rewards}")
        logger.info(f"Eval terminal reward mean: {terminal_reward_mean}")
        logger.info(f"Eval terminal reward std: {terminal_reward_std}")
        return terminal_rewards, terminal_reward_mean, terminal_reward_std
    finally:
        env.close()
        vec_env.close()


def evaluate(cfg: DictConfig, job_dir: Path, rl_zip: str) -> tuple[list[float | None], float, float]:
    device = _resolve_device(cfg)

    hydra_cfg = HydraConfig.get()
    original_path = Path(hydra_cfg.runtime.config_sources[1]["path"]).parent
    checkpoint_path = original_path / rl_zip
    _sync_constraints_with_checkpoint(cfg, checkpoint_path, device)
    model, vec_env = _build_eval_model(cfg, job_dir, device)

    seed = cfg.policy.PPO.seed
    logger.info(f"Loading model from {checkpoint_path} with seed {seed}")
    model = model.load(checkpoint_path, env=vec_env, device=device, seed=seed)
    return _run_policy_evaluation(cfg, job_dir, model, vec_env, render_every_successful_step=True)


def evaluate_untrained(cfg: DictConfig, job_dir: Path) -> tuple[list[float | None], float, float]:
    device = _resolve_device(cfg)
    if "constraints" in cfg:
        set_grammar_constraints(cfg.constraints)
    model, vec_env = _build_eval_model(cfg, job_dir, device)
    logger.info("Evaluating randomly initialized policy without loading a trained checkpoint")
    return _run_policy_evaluation(cfg, job_dir, model, vec_env)

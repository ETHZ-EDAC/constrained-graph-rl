"""PPO implementation"""

# Standard library
from pathlib import Path

# Third-party
import torch as th
from logging_mod.logger import get_logger
from omegaconf import DictConfig
from torchinfo import summary

# First-party
from graph_rl.ppo.callbacks import CallbackList, CheckpointCallback, EvalCallback, FullMetricCallback
from graph_rl.ppo.evaluation import evaluate_policy
from graph_rl.ppo.layers import GraphFeatureExtractor
from graph_rl.ppo.ppo import PPO
from graph_rl.sb3_fork.vec_env import SequentialVecEnv, SubprocVecEnv
from graph_rl.sb3_fork.vec_env.env_util import make_rule1_env, make_vec_env
from graph_rl.utils.hydra_helpers import disable_loggers_except_current
from graph_rl.utils.benchmark_graphs import evaluate_baseline_from_dataset
from graph_rl.sb3_fork.common.utils import WarmupThenDecaySchedule

logger = get_logger()


def train(cfg: DictConfig, job_dir: Path) -> None:

    if cfg.job.disable_grammar_logs:
        disable_loggers_except_current(logger)

    if th.cuda.is_available() and cfg.policy.meta.enable_cuda:
        device = "cuda"
    else:
        device = "cpu"
    logger.info(f"Using {device} device")

    vec_env = make_vec_env(
        make_rule1_env,
        cfg.policy.env.num_envs,
        seed=cfg.policy.PPO.seed,
        start_index=0,
        env_kwargs=dict(job_dir=job_dir, cfg=cfg),
        vec_env_cls=SubprocVecEnv if cfg.policy.env.parallel_env else SequentialVecEnv,
    )

    total_timesteps = cfg.policy.PPO.n_steps * cfg.policy.env.num_envs * cfg.policy.meta.total_ppo_iterations
    ppo_kwargs = dict(cfg.policy.PPO)
    ppo_kwargs["learning_rate"] = WarmupThenDecaySchedule(
        total_steps=int(total_timesteps),
        base_lr=float(ppo_kwargs["learning_rate"]),
        **dict(cfg.policy.PPO.lr_schedule),
    )

    model = PPO(
        vec_env,
        **ppo_kwargs,
        logger=logger,
        job_dir=job_dir,
        device=device,
        policy_kwargs=dict(
            features_extractor_class=GraphFeatureExtractor,
            features_extractor_kwargs=dict(cfg.policy.feature_extractor),
            distribution_kwargs=cfg.policy.distribution,
            actor_critic_kwargs=cfg.policy.actor_critic,
        ),
        action_projector_kwargs=cfg.policy.action_projector,
    )

    # Evaluate baseline from dataset if metric optimization is enabled
    target_metrics_cfg = cfg.target_metrics
    baseline_result = None
    target_metrics = None
    metric_callback = None
    if target_metrics_cfg.enabled and target_metrics_cfg.eval_baseline:
        target_metrics = target_metrics_cfg.metrics
        baseline_result = evaluate_baseline_from_dataset(target_metrics, job_dir)
        metric_callback = FullMetricCallback(
            logger,
            eval_freq=cfg.policy.meta.eval_freq,
            log_path=job_dir,
            verbose=1,
            baseline_result=baseline_result,
            target_metrics=target_metrics,
        )

    # Save a checkpoint every 1000 steps
    checkpoint_callback = CheckpointCallback(
        save_freq=cfg.policy.meta.save_freq,
        save_path=job_dir,
    )

    # Setup eval callback
    env1 = make_rule1_env(job_dir, cfg, env_name="env1")
    obs, _ = env1.reset(seed=cfg.policy.PPO.seed)
    eval_callback = EvalCallback(
        env1,
        logger,
        n_eval_episodes=6,
        eval_freq=cfg.policy.meta.eval_freq,
        log_path=job_dir,
        deterministic=False,
        render=cfg.policy.meta.render,
        cfg_ppo=cfg.policy.PPO,
        baseline_result=baseline_result,
        target_metrics=target_metrics,
    )

    dict(cfg.policy.feature_extractor)

    cb = [checkpoint_callback, eval_callback]
    if metric_callback is not None:
        cb.append(metric_callback)
    callback_list = CallbackList(cb)

    logger.info(model.policy)
    summary(model.policy, verbose=1)

    # total timesteps are a combination of env steps and n_envs
    logger.info("Starting PPO training for %s timesteps", total_timesteps)
    model.learn(total_timesteps=total_timesteps, callback=callback_list)

    rewards, episode_lengths = evaluate_policy(
        model,
        env1,
        logger,
        n_eval_episodes=10,
        deterministic=False,
        render=cfg.policy.meta.render,
        render_path=Path("final_eval"),
        return_episode_rewards=True,
    )

    logger.info(f"Final Eval Episode rewards: {rewards}")
    logger.info(f"Final Eval Episode lengths: {episode_lengths}")

    env1.close()
    vec_env.close()

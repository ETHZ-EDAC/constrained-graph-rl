# Standard library
import logging
import math
import pathlib
import sys
import time
import warnings
from collections import deque
from typing import Any, Optional, Union

# Third-party
import numpy as np
import torch as th
from torch.nn import functional as F
from tqdm import tqdm

# First-party
from graph_rl.ppo.actor_critic_policy import ActorCriticPolicy
from graph_rl.ppo.callbacks import BaseCallback
from graph_rl.ppo.action_projection import GraphActionProjector, train_action_projector
from graph_rl.ppo.env import CMAES_TIME_INFO_KEY, RULE_FN_TIME_INFO_KEY
from graph_rl.ppo.graph_rollout_buffer import GraphRolloutBuffer
from graph_rl.sb3_fork.base_buffers import BaseBuffer
from graph_rl.sb3_fork.base_optim import BaseOptim
from graph_rl.sb3_fork.common.utils import (
    explained_variance,
    FloatSchedule,
    MaybeCallback,
    obs_as_data_batch,
    safe_mean,
    Schedule,
    update_learning_rate,
)
from graph_rl.sb3_fork.vec_env import BaseVecEnv
from graph_rl.ppo.metrics import extract_metric_entry

VALID_ENTROPY_MODES = {"gaussian", "corrected_entropy"}


def _sum_rollout_env_time_s(infos: list[dict[str, Any]]) -> dict[str, float]:
    return {
        "rule_fn_s": float(sum(float(info.get(RULE_FN_TIME_INFO_KEY, 0.0)) for info in infos)),
        "apply_cmaes_to_rule_s": float(sum(float(info.get(CMAES_TIME_INFO_KEY, 0.0)) for info in infos)),
    }


def _compute_rollout_time_percentages(time_metrics_s: dict[str, float], rollout_total_s: float) -> dict[str, float]:
    if rollout_total_s <= 0.0:
        return {f"a_prc_{metric_name.removesuffix('_s')}": 0.0 for metric_name in time_metrics_s}
    return {
        f"a_prc_{metric_name.removesuffix('_s')}": float(metric_value / rollout_total_s * 100.0)
        for metric_name, metric_value in time_metrics_s.items()
    }


def _compute_entropy_loss(
    entropy_mode: str,
    log_prob: th.Tensor,
    analytic_entropy: Optional[th.Tensor],
) -> th.Tensor:
    if entropy_mode == "gaussian":
        if analytic_entropy is None:
            raise ValueError("Analytic entropy is required when entropy_mode='gaussian'.")
        return -th.mean(analytic_entropy)
    if entropy_mode == "corrected_entropy":
        # MC entropy estimate: H[p] = -E_a~p[log p(a)].
        # Since PPO minimizes losses, minimizing mean(log_prob) maximizes this entropy estimate.
        return th.mean(log_prob)
    raise ValueError(f"Unsupported entropy_mode '{entropy_mode}'. Expected one of {sorted(VALID_ENTROPY_MODES)}.")


class PPO(BaseOptim):
    """
    :param policy: The policy model to use (MlpPolicy, CnnPolicy, ...)
    :param env: The environment to learn from (if registered in Gym, can be str)
    :param learning_rate: The learning rate, it can be a function
        of the current progress remaining (from 1 to 0)
    :param n_steps: The number of steps to run for each environment per update
        (i.e. batch size is n_steps * n_env where n_env is number of environment copies running in parallel)
    :param gamma: Discount factor
    :param gae_lambda: Factor for trade-off of bias vs variance for Generalized Advantage Estimator.
        Equivalent to classic advantage when set to 1.
    :param ent_coef: Entropy coefficient for the loss calculation
    :param vf_coef: Value function coefficient for the loss calculation
    :param max_grad_norm: The maximum value for the gradient clipping
    :param rollout_buffer_class: Rollout buffer class to use. If ``None``, it will be automatically selected.
    :param rollout_buffer_kwargs: Keyword arguments to pass to the rollout buffer on creation.
    :param tensorboard_log: the log location for tensorboard (if None, no logging)

    :param policy_kwargs: additional arguments to be passed to the policy on creation

    :param seed: Seed for the pseudo random generators
    :param device: Device (cpu, cuda, ...) on which the code should be run.
        Setting it to auto, the code will be run on the GPU if possible.
    """

    policy: ActorCriticPolicy
    rollout_buffer: GraphRolloutBuffer

    def __init__(
        self,
        env: BaseVecEnv,
        policy_kwargs: Optional[dict[str, Any]] = None,
        action_projector_kwargs: Optional[dict[str, Any]] = None,
        learning_rate: Union[float, Schedule] = 3e-4,
        n_steps: int = 2048,
        batch_size: int = 64,
        n_epochs: int = 10,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: Union[float, Schedule] = 0.2,
        clip_range_vf: Union[None, float, Schedule] = None,
        normalize_advantage: bool = True,
        ent_coef: float = 0.0,
        entropy_mode: str = "gaussian",
        vf_coef: float = 0.5,
        max_grad_norm: float = 1.0,
        target_kl: Optional[float] = None,
        job_dir: Optional[pathlib.Path] = None,
        logger: Optional[logging.Logger] = None,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        cuda_deterministic: bool = False,
        **kwargs,
    ):
        super().__init__(
            env=env,
            learning_rate=learning_rate,
            policy_kwargs=policy_kwargs,
            logger=logger,
            device=device,
            seed=seed,
            job_dir=job_dir,
        )
        self.job_dir = job_dir

        self.projector_optimizer: Optional[th.optim.Optimizer] = None

        self.n_steps = n_steps
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.ent_coef = ent_coef
        if entropy_mode not in VALID_ENTROPY_MODES:
            raise ValueError(
                f"Unsupported entropy_mode '{entropy_mode}'. Expected one of {sorted(VALID_ENTROPY_MODES)}."
            )
        self.entropy_mode = entropy_mode
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.cuda_deterministic = cuda_deterministic
        # Sanity check, otherwise it will lead to noisy gradient and NaN
        # because of the advantage normalization
        self.num_minibatches = n_steps // batch_size
        if normalize_advantage:
            assert (
                batch_size > 1
            ), "`batch_size` must be greater than 1. See https://github.com/DLR-RM/stable-baselines3/issues/440"

        if self.env is not None:
            # Check that `n_steps * n_envs > 1` to avoid NaN
            # when doing advantage normalization
            buffer_size = self.env.num_envs * self.n_steps
            assert buffer_size > 1 or (
                not normalize_advantage
            ), f"`n_steps * n_envs` must be greater than 1. Currently n_steps={self.n_steps} and n_envs={self.env.num_envs}"
            # Check that the rollout buffer size is a multiple of the mini-batch size
            untruncated_batches = buffer_size // batch_size
            if buffer_size % batch_size > 0:
                warnings.warn(
                    f"You have specified a mini-batch size of {batch_size},"
                    f" but because the `RolloutBuffer` is of size `n_steps * n_envs = {buffer_size}`,"
                    f" after every {untruncated_batches} untruncated mini-batches,"
                    f" there will be a truncated mini-batch of size {buffer_size % batch_size}\n"
                    f"We recommend using a `batch_size` that is a factor of `n_steps * n_envs`.\n"
                    f"Info: (n_steps={self.n_steps} and n_envs={self.env.num_envs})"
                )
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.clip_range_vf = clip_range_vf
        self.normalize_advantage = normalize_advantage
        self.target_kl = target_kl
        self.iteration = 0

        self.action_projector_kwargs = action_projector_kwargs
        self.use_action_projector = self.action_projector_kwargs.get("use_action_projector")
        self._setup_model()

    def _set_projector_trainable(self, trainable: bool) -> None:
        """Toggle projector parameters' requires_grad without affecting differentiability w.r.t. inputs."""
        if self.action_projector is None:
            return
        for param in self.action_projector.parameters():
            param.requires_grad_(trainable)

    def _setup_model(self) -> None:
        self._setup_lr_schedule()
        self.set_random_seed(self.seed, self.cuda_deterministic)

        self.rollout_buffer = GraphRolloutBuffer(
            self.n_steps,
            self.batch_size,
            self.observation_space,  # type: ignore[arg-type]
            self.action_space,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs,
        )
        self.policy = ActorCriticPolicy(  # type: ignore[assignment]
            self.observation_space, self.action_space, self.lr_schedule, **self.policy_kwargs
        )
        self.policy = self.policy.to(self.device)
        if self.use_action_projector:
            # Learnable action projection g(s, a_raw) -> a_proj
            self.action_projector = GraphActionProjector(
                features_extractor_class=self.policy.features_extractor_class,
                features_extractor_kwargs=self.policy.features_extractor_kwargs,
                hidden_dim=int(self.action_projector_kwargs.get("hidden_dim")),
                embed_dim=int(self.action_projector_kwargs.get("embed_dim")),
            ).to(self.device)
            # Separate optimizer for projector (supervised training)
            proj_lr = float(self.lr_schedule(1))
            self.projector_optimizer = th.optim.Adam(self.action_projector.parameters(), lr=proj_lr)
            # Make projector accessible from the policy for inference-time usage
            self.policy.action_projector = self.action_projector
            self.policy.use_action_projector = True
        else:
            self.action_projector = None
            self.projector_optimizer = None
            self.policy.action_projector = None
            self.policy.use_action_projector = False

        # Replay buffer may be unused if feasibility guidance is off; fall back to a safe size.
        replay_buffer_size = self.action_projector_kwargs.get("replay_buffer_size")
        if replay_buffer_size is None:
            replay_buffer_size = self.n_steps
        self.replay_buffer = deque(maxlen=int(replay_buffer_size))
        # Initialize schedules for policy/value clipping
        self.clip_range = FloatSchedule(self.clip_range)
        if self.clip_range_vf is not None:
            if isinstance(self.clip_range_vf, (float, int)):
                assert self.clip_range_vf > 0, (
                    "`clip_range_vf` must be positive, " "pass `None` to deactivate vf clipping"
                )

            self.clip_range_vf = FloatSchedule(self.clip_range_vf)

    def _train_action_projector(self) -> None:
        """Supervised training of g(s, a_raw) -> a_proj using env-projected labels."""
        train_action_projector(
            projector=self.action_projector,
            optimizer=self.projector_optimizer,
            replay_buffer=self.replay_buffer,
            device=self.device,
            projection_supervised_epochs=self.action_projector_kwargs.get("projection_supervised_epochs"),
            max_grad_norm=self.max_grad_norm,
            ppo_logger=self.ppo_logger,
            update_lr_fn=self._update_learning_rate,
            set_trainable_fn=self._set_projector_trainable,
        )

    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
    ):
        """
        Return a trained model.

        :param total_timesteps: The total number of samples (env steps) to train on
            Note: it is a lower bound, see `issue #1150 <https://github.com/DLR-RM/stable-baselines3/issues/1150>`_
        :param callback: callback(s) called at every step with state of the algorithm.
        :return: the trained model
        """

        self.iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            self.stdout_logger.info(f"  ")
            self.stdout_logger.info(f" ----- Starting PPO iteration # {self.iteration+1} ----- ")
            self.stdout_logger.info(f"  ")

            self.stdout_logger.info("Collecting rollouts ...")
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, n_rollout_steps=self.n_steps
            )

            if not continue_training:
                break

            self.replay_buffer.extend(self.rollout_buffer.loader)

            self.iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            self.dump_logs(self.iteration)

            self.train()

            if self.use_action_projector and self.iteration <= self.action_projector_kwargs.get(
                "stop_training_action_projector_after"
            ):
                self._train_action_projector()

            callback.on_training_end()

        return self

    def train(self) -> None:
        """
        Update policy using the currently gathered rollout buffer.
        """
        # Switch to train mode (this affects batch norm / dropout)
        self.policy.set_training_mode(True)
        # action projector is trained separately via supervised loss; keep it eval during PPO updates (no dropout)
        if self.action_projector is not None:
            self.action_projector.eval()

        # Compute current clip range
        clip_range = self.clip_range(self._current_progress_remaining)  # type: ignore[operator]
        # Optional: clip range for the value function
        if self.clip_range_vf is not None:
            clip_range_vf = self.clip_range_vf(self._current_progress_remaining)  # type: ignore[operator]

        entropy_losses = []
        pg_losses, value_losses, advantages_pre, variance_returns = [], [], [], []
        clip_fractions = []

        continue_training = True

        # --- NEW: progress bar over epochs ---
        self.stdout_logger.info(f"Starting training ...")
        n_epochs = int(self.n_epochs)  # ensure it's an int
        pbar = tqdm(total=n_epochs, desc="PPO epochs", leave=True, file=sys.__stdout__) if tqdm is not None else None
        # train for n_epochs epochs
        for epoch in range(self.n_epochs):
            approx_kl_divs = []

            running_loss = 0.0
            n_minibatches = 0
            # Do a complete pass on the rollout buffer
            for rollout_data in self.rollout_buffer.get(self.batch_size):

                rollout_data = rollout_data.to(self.device)
                if self.use_action_projector:
                    _, log_det_jac = self.action_projector(rollout_data, rollout_data.actions_nominal)
                    values, log_prob_nominal, entropy = self.policy.evaluate_actions(
                        rollout_data, rollout_data.actions_nominal
                    )
                    log_prob = log_prob_nominal - log_det_jac
                else:
                    actions_raw = rollout_data.actions_nominal
                    values, log_prob_nominal, entropy = self.policy.evaluate_actions(rollout_data, actions_raw)
                    log_prob = log_prob_nominal
                values = values.flatten()
                # Normalize advantage
                advantages = rollout_data.advantages.flatten()
                # Normalization does not make sense if mini batchsize == 1, see GH issue #325
                advantages_pre.append(advantages.detach().std().item())
                if self.normalize_advantage and len(advantages) > 1:
                    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

                # ratio between old and new policy, should be one at the first iteration
                ratio = th.exp(log_prob - rollout_data.old_log_prob_projected.flatten())

                # clipped surrogate loss
                policy_loss_1 = advantages * ratio
                policy_loss_2 = advantages * th.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -th.min(policy_loss_1, policy_loss_2).mean()

                # Logging
                pg_losses.append(policy_loss.item())
                clip_fraction = th.mean((th.abs(ratio - 1) > clip_range).float()).item()
                clip_fractions.append(clip_fraction)

                if self.clip_range_vf is None:
                    # No clipping
                    values_pred = values
                else:
                    # Clip the difference between old and new value
                    # NOTE: this depends on the reward scaling
                    values_pred = rollout_data.old_values.flatten() + th.clamp(
                        values - rollout_data.old_values.flatten(), -clip_range_vf, clip_range_vf
                    )

                value_loss = F.mse_loss(rollout_data.returns.flatten(), values_pred)
                variance_returns.append(th.sqrt(rollout_data.returns.flatten().std()).item())

                value_losses.append(value_loss.item())

                entropy_loss = _compute_entropy_loss(self.entropy_mode, log_prob, entropy)
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss

                running_loss += float(loss.item())
                n_minibatches += 1

                with th.no_grad():
                    log_ratio = log_prob_nominal - rollout_data.old_log_prob_nominal.flatten()
                    approx_kl_div = th.mean((th.exp(log_ratio) - 1) - log_ratio).cpu().numpy()
                    approx_kl_divs.append(approx_kl_div)

                if self.iteration > 2 and self.target_kl is not None and approx_kl_div > 1.5 * self.target_kl:
                    continue_training = False
                    self.stdout_logger.info(
                        f"Early stopping at step {epoch} due to reaching max kl: {approx_kl_div:.2f}"
                    )
                    break

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if pbar is not None:
                mean_epoch_loss = running_loss / n_minibatches
                pbar.set_postfix(loss=f"{mean_epoch_loss:.4f}")
                pbar.update(1)

            self._n_updates += 1
            if not continue_training:
                break

        pbar.close()
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # Update optimizer learning rate (adaptive if target_kl is set)
        mean_kl = float(np.mean(approx_kl_divs)) if approx_kl_divs else 0.0
        # if self.target_kl is not None:
        #     self._update_learning_rate_with_kl(self.policy.optimizer, mean_kl)
        #
        # else:
        self._update_learning_rate(self.policy.optimizer)
        # Logs
        self.ppo_logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.ppo_logger.record("train/value_loss", np.mean(value_losses))
        self.ppo_logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.ppo_logger.record("train/advantages_pre", np.mean(advantages_pre))
        self.ppo_logger.record("train/variance_returns", np.mean(variance_returns))
        self.ppo_logger.record("train/approx_kl", mean_kl)
        self.ppo_logger.record("train/clip_fraction", np.mean(clip_fractions), exclude="tensorboard")
        self.ppo_logger.record("train/loss", loss.item())
        self.ppo_logger.record("train/explained_variance", explained_var)
        self.ppo_logger.record("train/n_updates", self._n_updates, exclude="tensorboard")

    def _update_learning_rate_with_kl(self, optimizer: th.optim.Optimizer, kl: float) -> None:
        """
        Adapt the learning rate based on KL divergence between new and old policy.
        """
        current_lr = float(optimizer.param_groups[0]["lr"])
        new_lr = current_lr
        if kl > 2.0 * self.target_kl:
            new_lr = max(1e-5, current_lr / 1.5)
        elif kl < 0.5 * self.target_kl:
            new_lr = min(1e-2, current_lr * 1.5)

        print(kl)
        print(new_lr)
        self.ppo_logger.record("train/learning_rate", new_lr)
        update_learning_rate(optimizer, new_lr)

    def collect_rollouts(
        self,
        env: BaseVecEnv,
        callback: BaseCallback,
        rollout_buffer: BaseBuffer,
        n_rollout_steps: int,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_rollout_steps: Number of experiences to collect per environment
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)

        if self.use_action_projector:
            self.action_projector.eval()

        n_steps = 0
        rollout_buffer.reset()

        rollout_start_ns = time.perf_counter_ns()
        callback.on_rollout_start()
        pbar = (
            tqdm(total=n_rollout_steps, desc="Rollout steps", leave=True, file=sys.__stdout__)
            if tqdm is not None
            else None
        )
        feature_buffer = []
        metrics_buffer = []
        policy_inference_times = []
        env_step_times = []
        bookkeeping_times = []
        node_entropy_buffer = []
        rule_entropy_buffer = []
        joint_entropy_buffer = []
        rollout_time_metrics = {
            "agent_evaluation_s": 0.0,
            "graph_feature_extractor_s": 0.0,
            "rule_fn_s": 0.0,
            "apply_cmaes_to_rule_s": 0.0,
            "action_projection_inference_s": 0.0,
            "bookkeeping_s": 0.0,
        }
        while n_steps < n_rollout_steps:
            # self.stdout_logger.debug("Collecting rollout step %d", n_steps)
            if pbar is not None:
                pbar.update(1)

            time_start = time.time_ns()
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_data_batch(self._last_obs).to(self.device)  # type: ignore[arg-type]
                feature_extractor_start_ns = time.perf_counter_ns()
                features = self.policy.features_extractor(obs_tensor)
                rollout_time_metrics["graph_feature_extractor_s"] += (
                    time.perf_counter_ns() - feature_extractor_start_ns
                ) / 1e9

                dist = self.policy._get_action_dist_from_latent(features)
                actions_nominal = dist.get_actions(deterministic=False)
                log_probs_nominal = dist.log_prob(actions_nominal)

                if self.policy.share_features_extractor:
                    critic_features = features
                else:
                    critic_feature_extractor_start_ns = time.perf_counter_ns()
                    critic_features = self.policy.critic_features_extractor(obs_tensor)
                    rollout_time_metrics["graph_feature_extractor_s"] += (
                        time.perf_counter_ns() - critic_feature_extractor_start_ns
                    ) / 1e9
                values = self.policy._compute_state_values(critic_features)

                if self.policy.use_joint_dist and dist.joint_dist is not None:
                    joint_entropy_buffer.append(dist.joint_dist.entropy().mean().item())
                else:
                    if dist.index_dist is not None:
                        node_entropy = dist.index_dist.entropy()
                        node_dim = dist.index_dist.probs.shape[-1]
                        node_denom = math.log(node_dim) if node_dim > 1 else 1.0
                        node_entropy_buffer.append((node_entropy / node_denom).mean().item())
                    if dist.rule_dist is not None:
                        rule_entropy = dist.rule_dist.entropy()
                        rule_dim = dist.rule_dist.probs.shape[-1]
                        rule_denom = math.log(rule_dim) if rule_dim > 1 else 1.0
                        rule_entropy_buffer.append((rule_entropy / rule_denom).mean().item())
                if self.use_action_projector:
                    projection_start_ns = time.perf_counter_ns()
                    actions_projected, log_det_jac = self.action_projector(obs_tensor, actions_nominal)
                    rollout_time_metrics["action_projection_inference_s"] += (
                        time.perf_counter_ns() - projection_start_ns
                    ) / 1e9
                    log_probs_projected = log_probs_nominal - log_det_jac
                else:
                    actions_projected = actions_nominal
                    log_probs_projected = log_probs_nominal
                feature_buffer.append(features["node_features"].cpu().numpy())

            agent_evaluation_s = (time.time_ns() - time_start) / 1e9
            policy_inference_times.append(agent_evaluation_s * 1e3)
            rollout_time_metrics["agent_evaluation_s"] += agent_evaluation_s

            time_start = time.time_ns()
            actions_projected[:, 2:] = th.clip(actions_projected[:, 2:], -1.0, 1.0)
            env_actions = actions_projected.detach().cpu().numpy()
            new_obs, rewards, dones, infos = env.step(env_actions)
            step_time_metrics = _sum_rollout_env_time_s(infos)
            for metric_name, metric_value in step_time_metrics.items():
                rollout_time_metrics[metric_name] += metric_value

            if n_steps > 0:
                env_step_times.append((time.time_ns() - time_start) / 1e6)
            self.num_timesteps += env.num_envs

            # Bookkeeping starts here
            time_bookkeeping_start = time.time_ns()
            with th.no_grad():
                entry = extract_metric_entry(infos)
            metrics_buffer.append(entry)

            callback.update_locals(locals())
            if not callback.on_step():
                return False
            self._update_info_buffer(infos, dones)

            n_steps += 1

            cost_feas = np.array([info.get("cost_geom_nominal") for info in infos])
            actions_projected_env = []
            for info in infos:
                projected = info.get("action_projected_env")
                actions_projected_env.append(projected)
            actions_projected_env = np.asarray(actions_projected_env, dtype=env_actions.dtype)

            rollout_buffer.add(
                self._last_obs,
                actions_nominal.detach().cpu().numpy(),
                actions_projected.detach().cpu().numpy(),
                actions_projected_env,
                rewards,
                self._last_episode_starts,
                values,
                log_probs_projected,
                log_probs_nominal,
                cost_feas,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

            # Total bookkeeping time
            bookkeeping_s = (time.time_ns() - time_bookkeeping_start) / 1e9
            bookkeeping_times.append(bookkeeping_s * 1e3)
            rollout_time_metrics["bookkeeping_s"] += bookkeeping_s
        with th.no_grad():
            # Compute value for the last timestep
            values = self.policy.predict_values(obs_as_data_batch(new_obs).to(self.device))  # type: ignore[arg-type]
        rollout_total_s = (time.perf_counter_ns() - rollout_start_ns) / 1e9

        # Timing summary
        self.stdout_logger.info("\n" + "=" * 60)
        self.stdout_logger.info("ROLLOUT TIMING SUMMARY")
        self.stdout_logger.info("=" * 60)
        self.stdout_logger.info(
            f"Policy inference: {np.mean(policy_inference_times):.2f} ± {np.std(policy_inference_times):.2f} ms"
        )
        self.stdout_logger.info(f"Environment step: {np.mean(env_step_times):.2f} ± {np.std(env_step_times):.2f} ms")
        self.stdout_logger.info(
            f"Bookkeeping:      {np.mean(bookkeeping_times):.2f} ± {np.std(bookkeeping_times):.2f} ms"
        )
        self.stdout_logger.info("=" * 60 + "\n")
        if pbar is not None:
            pbar.close()
        rollout_buffer.finish_dataloader(last_values=values, dones=dones)
        callback.update_locals(locals())

        callback.on_rollout_end()

        # Logs
        action_proj = rollout_buffer.actions_projected.reshape(-1, rollout_buffer.actions_projected.shape[-1])
        action_con_flat = action_proj[:, 2:].flatten()
        filtered = action_con_flat[np.absolute(action_con_flat) > 1e-6]  # Dirty way to filter zeros
        action_dist = rollout_buffer.actions_nominal.reshape(-1, rollout_buffer.actions_nominal.shape[-1])
        action_dist_c = action_dist[:, 2:].flatten()
        filtered_nominal = action_dist_c[np.absolute(action_dist_c) > 1e-6]  # Dirty way to filter zeros
        projected_cont = action_proj[:, 2:]
        projected_cont_flat = projected_cont.flatten()
        projected_std = float(np.std(projected_cont_flat))
        self.ppo_logger.record("actions/projected_dist", filtered, exclude="stdout")
        self.ppo_logger.record("actions/nominal_cont_dist", filtered_nominal, exclude="stdout")
        self.ppo_logger.record("actions/node_dist", action_dist[:, 0], exclude="stdout")
        self.ppo_logger.record("actions/rule_dist", action_dist[:, 1], exclude="stdout")
        self.ppo_logger.record("action_projector/projected_cont_std", projected_std, exclude="stdout")
        self.ppo_logger.record(
            "actions/log_std_dist", np.asarray(self.policy.action_dist.log_std_deque), exclude="stdout"
        )
        for metric_name, metric_value in rollout_time_metrics.items():
            self.ppo_logger.record(f"times/{metric_name}", float(metric_value))
        self.ppo_logger.record("times/rollout_total_s", float(rollout_total_s))
        rollout_time_percentages = _compute_rollout_time_percentages(rollout_time_metrics, rollout_total_s)
        for metric_name, metric_value in rollout_time_percentages.items():
            self.ppo_logger.record(f"times/{metric_name}", float(metric_value))
        self.ppo_logger.record("times/a_prc_rollout_total", 100.0 if rollout_total_s > 0.0 else 0.0)
        if joint_entropy_buffer:
            self.ppo_logger.record("a_rollout/joint_entropy", float(np.mean(joint_entropy_buffer)))
        if node_entropy_buffer:
            self.ppo_logger.record("a_rollout/node_entropy", float(np.mean(node_entropy_buffer)))
        if rule_entropy_buffer:
            self.ppo_logger.record("a_rollout/rule_entropy", float(np.mean(rule_entropy_buffer)))

        return True

    def dump_logs(self, iteration: int = 0) -> None:
        """
        Write log.

        :param iteration: Current logging iteration
        """
        assert self.ep_info_buffer is not None
        assert self.ep_success_buffer is not None

        time_elapsed = max((time.time_ns() - self.start_time) / 1e9, sys.float_info.epsilon)
        fps = int((self.num_timesteps - self._num_timesteps_at_start) / time_elapsed)
        if iteration > 0:
            self.ppo_logger.record("time/iterations", iteration, exclude="tensorboard")
        if len(self.ep_info_buffer) > 0 and len(self.ep_info_buffer[0]) > 0:
            self.ppo_logger.record(
                "a_rollout/a_total_rew_mean",
                safe_mean([ep_info["r_discounted"] for ep_info in self.ep_info_buffer]),
            )
            self.ppo_logger.record(
                "a_rollout/a_terminal_reward", safe_mean([ep_info for ep_info in self.ep_terminal_reward])
            )
            self.ppo_logger.record(
                "a_rollout/a_ep_len_mean", safe_mean([ep_info["l"] for ep_info in self.ep_info_buffer])
            )
            self.ppo_logger.record(
                "a_rollout/b_cost_geom_nominal",
                safe_mean([ep_info["cost_geom_nominal"] for ep_info in self.ep_info_buffer]),
            )
            self.ppo_logger.record(
                "a_rollout/b_cost_geom_proj",
                safe_mean([ep_info["cost_geom_proj"] for ep_info in self.ep_info_buffer]),
            )
            self.ppo_logger.record(
                "a_rollout/b_cost_topo",
                safe_mean([ep_info["cost_topo"] for ep_info in self.ep_info_buffer]),
            )
            self.ppo_logger.record(
                "a_rollout/c_cmaes_applied_num",
                safe_mean([ep_info["cmaes_applied_num"] for ep_info in self.ep_info_buffer]),
            )
            self.ppo_logger.record(
                "a_rollout/c_cmaes_applied_success_rate",
                safe_mean([ep_info["cmaes_applied_success_rate"] for ep_info in self.ep_info_buffer]),
            )

        self.ppo_logger.record("time/time_elapsed", int(time_elapsed), exclude="tensorboard")
        self.ppo_logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
        if len(self.ep_success_buffer) > 0:
            self.ppo_logger.record(
                "a_rollout/success_rate", np.count_nonzero(self.ep_success_buffer) / len(self.ep_success_buffer)
            )
        self._record_layer_weights()
        self.ppo_logger.dump(step=self.num_timesteps, enable_tensorboard=iteration > 3)

    def _get_torch_save_params(self) -> tuple[list[str], list[str]]:
        state_dicts = ["policy", "policy.optimizer"]
        if self.action_projector is not None:
            state_dicts.append("action_projector")
            if self.projector_optimizer is not None:
                state_dicts.append("projector_optimizer")

        return state_dicts, []

    @staticmethod
    def _last_linear_weight(module: th.nn.Module) -> Optional[th.Tensor]:
        last_linear = None
        for submodule in module.modules():
            if isinstance(submodule, th.nn.Linear):
                last_linear = submodule
        if last_linear is None:
            return None
        return last_linear.weight.detach()

    def _record_layer_weights(self) -> None:
        mean_weight = self._last_linear_weight(self.policy.mean_head)
        if mean_weight is not None:
            self.ppo_logger.record("layers/mean_head/last_weight", mean_weight, exclude="stdout")
        rule_weight = self._last_linear_weight(self.policy.rule_head)
        if rule_weight is not None:
            self.ppo_logger.record("layers/rule_head/last_weight", rule_weight, exclude="stdout")
        node_scorer = getattr(self.policy.features_extractor, "node_scorer", None)
        if node_scorer is not None:
            node_weight = self._last_linear_weight(node_scorer)
            if node_weight is not None:
                self.ppo_logger.record("layers/node_scorer/last_weight", node_weight, exclude="stdout")

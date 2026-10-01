# Standard library
import logging
import os
from abc import ABC
from pathlib import Path
from typing import Any, Optional

# Third-party
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import DictConfig
from sklearn.decomposition import PCA

# First-party
from graph_rl.ppo.evaluation import evaluate_policy
from graph_rl.sb3_fork.common.ppo_logger import PPOLogger
from graph_rl.sb3_fork.vec_env.monitor import Monitor
from graph_rl.ppo.metrics import (
    save_metric_history_figures,
    aggregate_metric_history,
    compute_metrics_from_data,
    extract_metric_entry,
)
from graph_rl.utils.benchmark_graphs import compute_metric_similarity


class BaseCallback(ABC):
    """
    Base class for callback.

    :param verbose: Verbosity level: 0 for no output, 1 for info messages, 2 for debug messages
    """

    # The RL model
    # Type hint as string to avoid circular import
    model: "base_class.BaseOptim"

    def __init__(self, save_path: Optional[Path] = None, verbose: int = 0):
        super().__init__()
        # Number of time the callback was called
        self.n_calls = 0  # type: int
        # n_envs * n times env.step() was called
        self.n_iterations = 0  # type: int
        self.num_timesteps = 0  # type: int
        self.verbose = verbose
        self.save_path = save_path
        self.locals: dict[str, Any] = {}
        self.globals: dict[str, Any] = {}
        # Sometimes, for event callback, it is useful
        # to have access to the parent object
        self.parent = None  # type: Optional[BaseCallback]

    @property
    def logger(self) -> PPOLogger:
        return self.model.ppo_logger

    # Type hint as string to avoid circular import
    def init_callback(self, model: "base_class.BaseOptim") -> None:
        """
        Initialize the callback by saving references to the
        RL model and the training environment for convenience.
        """
        self.model = model
        self._init_callback()

    def _init_callback(self) -> None:
        pass

    def on_training_start(self, locals_: dict[str, Any], globals_: dict[str, Any]) -> None:
        # Those are reference and will be updated automatically
        self.locals = locals_
        self.globals = globals_
        # Update num_timesteps in case training was done before
        self.num_timesteps = self.model.num_timesteps
        self._on_training_start()

    def _on_training_start(self) -> None:
        pass

    def on_rollout_start(self) -> None:
        self._on_rollout_start()

    def _on_rollout_start(self) -> None:
        pass

    def _on_step(self) -> bool:
        """
        :return: If the callback returns False, training is aborted early.
        """
        return True

    def on_step(self) -> bool:
        """
        This method will be called by the model after each call to ``env.step()``.

        For child callback (of an ``EventCallback``), this will be called
        when the event is triggered.

        :return: If the callback returns False, training is aborted early.
        """
        self.n_calls += 1
        self.num_timesteps = self.model.num_timesteps

        return self._on_step()

    def on_training_end(self) -> None:
        self.n_iterations += 1
        self._on_training_end()

    def _on_training_end(self) -> None:
        pass

    def on_rollout_end(self) -> None:
        self._on_rollout_end()

    def _on_rollout_end(self) -> None:
        pass

    def update_locals(self, locals_: dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        self.locals.update(locals_)
        self.update_child_locals(locals_)

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        """
        Update the references to the local variables on sub callbacks.

        :param locals_: the local variables during rollout collection
        """
        pass

    def _checkpoint_path_training_end(self) -> Path:
        """
        Helper to get checkpoint path for each type of checkpoint.

        :return: Path to the checkpoint
        """
        return Path(f"ppo_iter_{self.n_iterations}")


class CallbackList(BaseCallback):
    """
    Class for chaining callbacks.

    :param callbacks: A list of callbacks that will be called
        sequentially.
    """

    def __init__(self, callbacks: list[BaseCallback]):
        super().__init__()
        assert isinstance(callbacks, list)
        self.callbacks = callbacks

    def _init_callback(self) -> None:
        for callback in self.callbacks:
            callback.init_callback(self.model)

            # Fix for https://github.com/DLR-RM/stable-baselines3/issues/1791
            # pass through the parent callback to all children
            callback.parent = self.parent

    def _on_training_start(self) -> None:
        for callback in self.callbacks:
            callback.on_training_start(self.locals, self.globals)

    def _on_rollout_start(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_start()

    def _on_step(self) -> bool:
        continue_training = True
        for callback in self.callbacks:
            # Return False (stop training) if at least one callback returns False
            continue_training = callback.on_step() and continue_training
        return continue_training

    def _on_rollout_end(self) -> None:
        for callback in self.callbacks:
            callback.on_rollout_end()

    def _on_training_end(self) -> None:
        for callback in self.callbacks:
            callback.on_training_end()

    def update_child_locals(self, locals_: dict[str, Any]) -> None:
        """
        Update the references to the local variables.

        :param locals_: the local variables during rollout collection
        """
        for callback in self.callbacks:
            callback.update_locals(locals_)


class CheckpointCallback(BaseCallback):
    """
    Callback for saving a model every ``save_freq`` calls
    to ``env.step()``.
    By default, only model checkpoints are saved.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``save_freq = max(save_freq // n_envs, 1)``

    :param save_freq: Save checkpoints every ``save_freq`` call of the callback.
    :param save_path: Path to the folder where the model will be saved.
    :param verbose: Verbosity level: 0 for no output, 2 for indicating when saving model checkpoint
    """

    def __init__(
        self,
        save_freq: int,
        save_path: Path,
        verbose: int = 0,
    ):
        super().__init__(save_path, verbose)
        self.save_freq = save_freq
        self.save_path = save_path

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_training_end(self) -> bool:
        if self.n_iterations % self.save_freq == 0:
            model_path = self.save_path / self._checkpoint_path_training_end()
            model_path.mkdir(parents=True, exist_ok=True)
            self.model.save(model_path / "model")
            if self.verbose >= 2:
                print(f"Saving model checkpoint to {model_path}")
        return True


class EvalCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``

    :param eval_env: The environment used for initialization
    :param callback_on_new_best: Callback to trigger
        when there is a new best model according to the ``mean_reward``
    :param callback_after_eval: Callback to trigger after every evaluation
    :param n_eval_episodes: The number of episodes to test the agent
    :param eval_freq: Evaluate the agent every ``eval_freq`` call of the callback.
    :param log_path: Path to a folder where the evaluations (``evaluations.npz``)
        will be saved. It will be updated at each evaluation.
    :param deterministic: Whether the evaluation should
        use a stochastic or deterministic actions.
    :param render: Whether to render or not the environment during evaluation
    :param verbose: Verbosity level: 0 for no output, 1 for indicating information about evaluation results
    """

    def __init__(
        self,
        eval_env: Monitor,
        logger: logging.Logger,
        cfg_ppo: DictConfig,
        n_eval_episodes: int = 5,
        eval_freq: int = 10000,
        log_path: Optional[Path] = None,
        deterministic: bool = True,
        render: bool = False,
        verbose: int = 1,
        baseline_result: Optional[tuple] = None,
        target_metrics: Optional[dict] = None,
    ):
        super().__init__(log_path, verbose=verbose)

        self.n_eval_episodes = n_eval_episodes
        self.eval_freq = eval_freq
        self.best_mean_reward = -np.inf
        self.last_mean_reward = -np.inf
        self.deterministic = deterministic
        self.render = render
        self.std_logger = logger
        self.cfg_ppo = cfg_ppo

        self.eval_env = eval_env
        # Logs will be written in ``evaluations.npz``

        self.save_path = log_path
        self.evaluations_results: list[list[float]] = []
        self.evaluations_timesteps: list[int] = []
        self.evaluations_length: list[list[int]] = []
        # For computing success rate
        self._is_success_buffer: list[bool] = []
        self.evaluations_successes: list[list[bool]] = []

        # Baseline tracking for metric optimization
        self.baseline_result = baseline_result
        self.target_metrics = target_metrics

    def _on_training_end(self) -> bool:
        continue_training = True

        if self.eval_freq > 0 and self.n_iterations % self.eval_freq == 0:

            self.std_logger.info(" --- Validation ---")

            episode_rewards, episode_lengths = evaluate_policy(
                self.model,
                self.eval_env,
                self.std_logger,
                n_eval_episodes=self.n_eval_episodes,
                render=self.render,
                render_path=self._checkpoint_path_training_end(),
                deterministic=self.deterministic,
                return_episode_rewards=True,
            )

            assert isinstance(episode_rewards, list)
            assert isinstance(episode_lengths, list)
            self.evaluations_timesteps.append(self.num_timesteps)
            self.evaluations_results.append(episode_rewards)
            self.evaluations_length.append(episode_lengths)

            mean_reward, std_reward = np.mean(episode_rewards), np.std(episode_rewards)
            mean_ep_length, std_ep_length = np.mean(episode_lengths), np.std(episode_lengths)
            self.last_mean_reward = float(mean_reward)

            if self.verbose >= 1:
                self.std_logger.info(
                    f"Eval num_timesteps={self.num_timesteps}, "
                    f"episode_reward={mean_reward:.2f} +/- {std_reward:.2f}"
                )
                self.std_logger.info(f"Episode length: {mean_ep_length:.2f} +/- {std_ep_length:.2f}")
            # Add to current PPOLogger
            self.logger.record("eval/mean_reward", float(mean_reward))
            self.logger.record("eval/mean_ep_length", mean_ep_length)

            # Dump log so the evaluation results are printed with the correct timestep
            self.logger.record("time/total_timesteps", self.num_timesteps, exclude="tensorboard")
            self.logger.dump(self.num_timesteps, enable_tensorboard=self.model.iteration > 3)

            if mean_reward > self.best_mean_reward:
                if self.verbose >= 1:
                    self.std_logger.info("New best mean reward!")
                self.model.save(self.save_path / "best_model")
                self.best_mean_reward = float(mean_reward)

        return continue_training

    def _on_rollout_end(self) -> None:
        if self.eval_freq > 0 and self.n_iterations % self.eval_freq == 0:
            path = self.save_path / self._checkpoint_path_training_end()
            path.mkdir(exist_ok=True, parents=True)
            pca = PCA(n_components=2)
            node_features = np.concatenate(self.locals["feature_buffer"])
            self.std_logger.info(
                f"Mean, Max, Min feature vector values: {node_features.mean()}, {node_features.max()}, {node_features.min()}"
            )
            X_2d = pca.fit_transform(node_features)  # X.shape = (N, feature_dim)
            fig, ax = plt.subplots()
            sc = ax.scatter(X_2d[:, 0], X_2d[:, 1], c="blue", s=8)
            ax.set_title("UMAP projection")
            plt.savefig(path / f"feature_pca.png")
            plt.close(fig)

            fig, axes = plt.subplots(4, 5, figsize=(20, 12))
            axes = axes.flatten()
            feat_buffer = np.concatenate(self.locals["feature_buffer"])
            for i in range(20):
                ax = axes[i]
                ax.hist(feat_buffer[:, i], bins=50)
                ax.set_title(f"Index {i}")

            plt.tight_layout()
            plt.savefig(path / f"feature_distribution.png")
            plt.close(fig)

        metrics_buffer = self.locals.get("metrics_buffer")
        if isinstance(metrics_buffer, list) and metrics_buffer:
            # Log metrics to TensorBoard
            aggregated_metrics = aggregate_metric_history(metrics_buffer)
            for metric_name, values in aggregated_metrics.items():
                if len(values) > 0:
                    mean_value = float(values.mean())
                    self.logger.record(f"metrics/{metric_name}_mean", mean_value)


class FullMetricCallback(BaseCallback):
    """
    Callback for evaluating an agent.

    .. warning::

      When using multiple environments, each call to  ``env.step()``
      will effectively correspond to ``n_envs`` steps.
      To account for that, you can use ``eval_freq = max(eval_freq // n_envs, 1)``
    """

    def __init__(
        self,
        logger: logging.Logger,
        eval_freq: int = 10000,
        log_path: Optional[Path] = None,
        verbose: int = 1,
        baseline_result: Optional[tuple] = None,
        target_metrics: Optional[dict] = None,
    ):
        super().__init__(log_path, verbose=verbose)

        self.eval_freq = eval_freq
        self.std_logger = logger
        self.save_path = log_path

        # Baseline tracking for metric optimization
        self.baseline_result = baseline_result
        self.target_metrics = target_metrics

    def _on_rollout_end(self) -> None:

        if self.eval_freq > 0 and self.n_iterations % self.eval_freq == 0:
            metrics_buffer = self.locals.get("metrics_buffer")
            if isinstance(metrics_buffer, list) and metrics_buffer:
                path = self.save_path / self._checkpoint_path_training_end()
                path.mkdir(exist_ok=True, parents=True)
                baseline_metrics = None
                if self.target_metrics is not None:
                    dataset_name, graph_idx, baseline_score, baseline_differences, baseline_graph = self.baseline_result
                    baseline_metrics_dict = compute_metrics_from_data(baseline_graph)
                    baseline_metrics = baseline_metrics_dict.get("graph_metrics", {})
                save_metric_history_figures(
                    metrics_buffer,
                    path,
                    baseline_metrics=baseline_metrics,
                    target_metrics=self.target_metrics,
                )

                # Log baseline comparison if available
                if self.target_metrics is not None:
                    # Log metrics to TensorBoard
                    aggregated_metrics = aggregate_metric_history(metrics_buffer)

                    # Compute distance from rollout to target and from baseline to target
                    rollout_avg_metrics = {k: float(v.mean()) for k, v in aggregated_metrics.items()}
                    metric_subset = list(self.target_metrics.keys())
                    missing_metrics = [name for name in metric_subset if name not in rollout_avg_metrics]
                    if missing_metrics:
                        self.std_logger.warning(
                            "Skipping missing rollout metrics for similarity: %s",
                            ", ".join(missing_metrics),
                        )
                        metric_subset = [name for name in metric_subset if name in rollout_avg_metrics]
                    if not metric_subset:
                        return
                    rollout_to_target_similarity, rollout_differences = compute_metric_similarity(
                        self.target_metrics, rollout_avg_metrics, metric_subset=metric_subset
                    )

                    # Log baseline and rollout similarity scores
                    self.logger.record("baseline/baseline_similarity", baseline_score)
                    self.logger.record("metrics/RL_similarity_to_target", rollout_to_target_similarity)
                    self.logger.record(
                        "a_rollout/distance_from_baseline", rollout_to_target_similarity - baseline_score
                    )

                    # Log detailed metrics
                    for metric_name in self.target_metrics.keys():
                        # if metric_name in baseline_metrics and metric_name in last_metrics:
                        if metric_name in aggregated_metrics:
                            self.logger.record(f"baseline/{metric_name}", baseline_metrics[metric_name])
                            rollout_mean = float(aggregated_metrics[metric_name].mean())
                            target_val = self.target_metrics[metric_name]
                            self.logger.record(f"target/target_{metric_name}", target_val)

                            # Log distance from baseline for this metric
                            if metric_name in baseline_differences and metric_name in rollout_differences:
                                baseline_diff = baseline_differences[metric_name]
                                rollout_diff = rollout_differences[metric_name]
                                self.logger.record(f"comparison/baseline_{metric_name}_distance", baseline_diff)
                                self.logger.record(f"comparison/rollout_{metric_name}_distance", rollout_diff)


class StopTrainingOnNoModelImprovement(BaseCallback):
    """
    Stop the training early if there is no new best model (new best mean reward) after more than N consecutive evaluations.

    It is possible to define a minimum number of evaluations before start to count evaluations without improvement.

    It must be used with the ``EvalCallback``.

    :param max_no_improvement_evals: Maximum number of consecutive evaluations without a new best model.
    :param min_evals: Number of evaluations before start to count evaluations without improvements.
    :param verbose: Verbosity level: 0 for no output, 1 for indicating when training ended because no new best model
    """

    parent: EvalCallback

    def __init__(self, max_no_improvement_evals: int, min_evals: int = 0, verbose: int = 0):
        super().__init__(verbose=verbose)
        self.max_no_improvement_evals = max_no_improvement_evals
        self.min_evals = min_evals
        self.last_best_mean_reward = -np.inf
        self.no_improvement_evals = 0

    def _on_step(self) -> bool:
        assert (
            self.parent is not None
        ), "``StopTrainingOnNoModelImprovement`` callback must be used with an ``EvalCallback``"

        continue_training = True

        if self.n_calls > self.min_evals:
            if self.parent.best_mean_reward > self.last_best_mean_reward:
                self.no_improvement_evals = 0
            else:
                self.no_improvement_evals += 1
                if self.no_improvement_evals > self.max_no_improvement_evals:
                    continue_training = False

        self.last_best_mean_reward = self.parent.best_mean_reward

        if self.verbose >= 1 and not continue_training:
            print(
                f"Stopping training because there was no new best model in the last {self.no_improvement_evals:d} evaluations"
            )

        return continue_training

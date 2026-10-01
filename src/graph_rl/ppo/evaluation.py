# Standard library
import logging
from pathlib import Path
from typing import Optional, Union

# Third-party
import numpy as np
import torch
from torch_geometric.data import Batch

# First-party
from graph_rl.ppo.env import RULE_SPECS
from graph_rl.sb3_fork.vec_env.monitor import Monitor
from graph_rl.utils import PARAMS


def evaluate_policy(
    model: Union["BaseOptim"],
    env: Monitor,
    logger: logging.Logger,
    n_eval_episodes: int = 10,
    deterministic: bool = True,
    render: bool = False,
    render_path: Optional[Path] = None,
    render_every_successful_step: bool = False,
    show_step_render: bool = True,
    return_episode_rewards: bool = False,
    return_terminal_rewards: bool = False,
) -> Union[
    tuple[float, float],
    tuple[list[float], list[int]],
    tuple[list[float], list[int], list[float | None]],
]:
    """
    Runs the policy for ``n_eval_episodes`` episodes and outputs the average return
    per episode (sum of undiscounted rewards).
    If a vector env is passed in, this divides the episodes to evaluate onto the
    different elements of the vector env. This static division of work is done to
    remove bias. See https://github.com/DLR-RM/stable-baselines3/issues/402 for more
    details and discussion.

    :param model: The RL agent you want to evaluate. This can be any object
        that implements a ``predict`` method, such as an RL algorithm (``BaseAlgorithm``)
        or policy (``BasePolicy``).
    :param env: The gym environment or ``BaseVecEnv`` environment.
    :param logger: Logger object for logging information,
    :param n_eval_episodes: Number of episode to evaluate the agent
    :param deterministic: Whether to use deterministic or stochastic actions
    :param render: Whether to render the environment or not
    :param render_path: Optional path for saving render plots. If None, plots are not saved.
    :param render_every_successful_step: If True, save/show a pretty graph after each
        step that successfully updates the graph.
    :param show_step_render: If True with ``render_every_successful_step``, call
        ``plt.show()`` for every successful step render.
    :param return_episode_rewards: If True, a list of rewards and episode lengths
        per episode will be returned instead of the mean.
    :param return_terminal_rewards: If True together with ``return_episode_rewards``,
        returns a third list containing the per-episode terminal reward when present.

    :return: Mean return per episode (sum of rewards), std of reward per episode.
        Returns (list[float], list[int]) when ``return_episode_rewards`` is True, first
        list containing per-episode return and second containing per-episode lengths
        (in number of steps). If ``return_terminal_rewards`` is also True, returns
        (list[float], list[int], list[float | None]).

    """
    logger.info(f"Evaluating policy for {n_eval_episodes} episodes...")

    episode_rewards = []
    episode_lengths = []
    episode_terminal_rewards = []

    episode_count = 0
    # Divides episodes among different sub environments in the vector as evenly as possible

    current_rewards = 0
    current_lengths = 0
    observations, _ = env.reset()

    use_action_projector = getattr(model, "use_action_projector")
    action_projector = getattr(model, "action_projector")
    projector_device = None
    if use_action_projector and action_projector is not None:
        action_projector.eval()
        projector_device = next(action_projector.parameters()).device

    step = 0
    with torch.no_grad():
        while episode_count < n_eval_episodes:
            actions = model.policy.predict(
                observations,  # type: ignore[arg-type]
                deterministic=deterministic,
            )

            # Apply action projector if enabled and past warmup.
            if use_action_projector and action_projector is not None and projector_device is not None:
                obs_batch = Batch.from_data_list([observations]).to(projector_device)
                actions_tensor = torch.as_tensor(actions, device=projector_device).unsqueeze(0)
                actions_projected, _ = action_projector(obs_batch, actions_tensor)
                actions = actions_projected.squeeze(0).cpu().numpy()

            actions[2:] = np.clip(actions[2:], -1.0, 1.0)
            new_observations, rewards, term, trunc, infos = env.step(actions)
            dones = term or trunc
            step += 1

            if render and render_every_successful_step and infos.get("step_success", False):
                title = f"{env.env_name}_ep{episode_count + 1:03d}_step{step:04d}_pretty"
                env.get_wrapper_attr("cp").plot_planar_graph(
                    show_step_render,
                    render_path,
                    title,
                    full_adjacency=True,
                )

            current_rewards += rewards
            # logger.debug(f"Reward {rewards} at step {step} ")

            current_lengths += 1
            if dones:
                logger.info(f"Terminal undiscounted return {current_rewards} at step {step}.")
                print(infos)

                episode_count += 1

                episode_rewards.append(current_rewards)
                episode_lengths.append(current_lengths)
                episode_terminal_rewards.append(
                    float(infos["terminal_reward"]) if "terminal_reward" in infos else None
                )

                current_rewards = 0
                current_lengths = 0
                step = 0

                if render:
                    title = f"{env.env_name}_ep" + str(episode_count) + "full"
                    env.get_wrapper_attr("cp").plot_planar_graph(False, render_path, title, full_adjacency=True)

                new_observations, _ = env.reset()
            observations = new_observations

        mean_reward = np.mean(episode_rewards)
        std_reward = np.std(episode_rewards)
        if return_episode_rewards:
            if return_terminal_rewards:
                return episode_rewards, episode_lengths, episode_terminal_rewards
            return episode_rewards, episode_lengths
    return mean_reward, std_reward

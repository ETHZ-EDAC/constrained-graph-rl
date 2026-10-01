# Standard library
from functools import partial
from typing import Any, Dict, Optional

# Third-party
import numpy as np
import torch as th
import torch.nn as nn
from gymnasium import spaces
from torch.distributions import Categorical
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_batch

# First-party
from graph_rl.ppo.distribution import (
    HybridDistribution,
    MlpHeads,
)
from graph_rl.ppo.env import RULE_SPECS
from graph_rl.ppo.layers import (
    BaseFeatureExtractor,
)
from graph_rl.sb3_fork.base_policy import BasePolicy
from graph_rl.sb3_fork.common.utils import PyTorchObs, Schedule


class ActorCriticPolicy(BasePolicy):
    """
    Policy class for actor-critic algorithms (has both policy and value prediction).
    Used by A2C, PPO and the likes.

    :param observation_space: Observation space
    :param action_space: Action space
    :param lr_schedule: Learning rate schedule (could be constant)
    :param ortho_init: Whether to use or not orthogonal initialization
    :param init_log_std: Constant offset added to log-std head outputs
    :param features_extractor_class: Features extractor to use.
    :param features_extractor_kwargs: Keyword arguments
        to pass to the features extractor.
    :param optimizer_class: The optimizer to use,
        ``th.optim.Adam`` by default
    :param optimizer_kwargs: Additional keyword arguments,
        excluding the learning rate, to pass to the optimizer
    """

    def __init__(
        self,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        lr_schedule: Schedule,
        ortho_init: bool = True,
        init_std: float = 1.0,
        features_extractor_class: type[BaseFeatureExtractor] = BaseFeatureExtractor,
        features_extractor_kwargs: Optional[dict[str, Any]] = None,
        critic_features_extractor_class: Optional[type[BaseFeatureExtractor]] = None,
        critic_features_extractor_kwargs: Optional[dict[str, Any]] = None,
        actor_critic_kwargs: Optional[dict[str, Any]] = None,
        distribution_kwargs: Optional[dict[str, Any]] = None,
        optimizer_class: type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[dict[str, Any]] = None,
    ):
        if optimizer_kwargs is None:
            optimizer_kwargs = {}
            # Small values to avoid NaN in Adam optimizer
            if optimizer_class == th.optim.Adam:
                optimizer_kwargs["eps"] = 1e-5

        super().__init__(
            observation_space,
            action_space,
            features_extractor_class,
            features_extractor_kwargs,
            optimizer_class=optimizer_class,
            optimizer_kwargs=optimizer_kwargs,
        )

        self.ortho_init = ortho_init
        self.share_features_extractor = bool(actor_critic_kwargs.get("share_features_extractor"))

        self.features_extractor = self.features_extractor_class(**self.features_extractor_kwargs)
        if self.share_features_extractor:
            self.critic_features_extractor_class = self.features_extractor_class
            self.critic_features_extractor_kwargs = self.features_extractor_kwargs
            self.critic_features_extractor = self.features_extractor
        else:
            self.critic_features_extractor_class = critic_features_extractor_class or self.features_extractor_class
            self.critic_features_extractor_kwargs = critic_features_extractor_kwargs or self.features_extractor_kwargs
            self.critic_features_extractor = self.critic_features_extractor_class(
                **self.critic_features_extractor_kwargs
            )
        actor_critic_kwargs = actor_critic_kwargs or {}
        # Toggle for joint (node, rule) categorical sampling; default True
        self.use_joint_dist = bool(actor_critic_kwargs.get("use_joint_dist"))
        self.use_global_log_std = bool(actor_critic_kwargs.get("use_global_log_std"))
        self.init_log_std = np.log(float(actor_critic_kwargs.get("init_std", init_std)))

        dist_kwargs = None

        self.dist_kwargs = dist_kwargs
        self.actor_critic_kwargs = actor_critic_kwargs

        # Action distribution
        self.cont_dim_max = max(rulespec.dim_cont for rulespec in RULE_SPECS)
        self.n_classes = len(RULE_SPECS)
        self.action_dist = HybridDistribution(
            **distribution_kwargs,
        )
        self._build(lr_schedule)

    def _get_constructor_parameters(self) -> dict[str, Any]:
        data = super()._get_constructor_parameters()

        data.update(
            dict(
                ortho_init=self.ortho_init,
                optimizer_class=self.optimizer_class,
                optimizer_kwargs=self.optimizer_kwargs,
                features_extractor_class=self.features_extractor_class,
                features_extractor_kwargs=self.features_extractor_kwargs,
                critic_features_extractor_class=self.critic_features_extractor_class,
                critic_features_extractor_kwargs=self.critic_features_extractor_kwargs,
                init_log_std=self.init_log_std,
            )
        )
        return data

    def _build(self, lr_schedule: Schedule) -> None:
        """
        Create the networks and the optimizer.

        :param lr_schedule: Learning rate schedule
            lr_schedule(1) is the initial learning rate
        """

        # additions 2,1 correspond to conditioning inputs
        self.mean_head = MlpHeads(
            in_dim=int(self.features_extractor.node_features_dim),
            **self.actor_critic_kwargs["mean_head"],
            num_conditionals=1,
            out_dim=self.cont_dim_max,
        )

        if self.use_global_log_std:
            scale = 0.5 * (HybridDistribution.LOG_STD_MAX - HybridDistribution.LOG_STD_MIN)
            bias = 0.5 * (HybridDistribution.LOG_STD_MAX + HybridDistribution.LOG_STD_MIN)
            init_log_std_pre_squashed = np.clip((self.init_log_std - bias) / scale, -0.999, 0.999)
            init_log_std_pre_squashed = float(np.arctanh(init_log_std_pre_squashed))
            self.log_std_param = nn.Parameter(th.full((self.cont_dim_max,), init_log_std_pre_squashed))
        else:
            self.log_std_head = MlpHeads(
                in_dim=int(self.features_extractor.node_features_dim),
                **self.actor_critic_kwargs["log_std_head"],
                num_conditionals=1,
                out_dim=self.cont_dim_max,
            )

        self.rule_head = MlpHeads(
            in_dim=int(self.features_extractor.node_features_dim),
            **self.actor_critic_kwargs["rule_head"],
            out_dim=self.n_classes,
        )

        self.glob_critic = MlpHeads(
            in_dim=int(self.critic_features_extractor.glob_features_dim),
            **self.actor_critic_kwargs["glob_critic"],
            out_dim=1,
        )
        self.node_count_embed = nn.Sequential(
            nn.Linear(1, int(self.critic_features_extractor.glob_features_dim)),
            nn.SiLU(),
        )
        # Init weights: orthogonal for hidden layers, small init for mean/log_std outputs
        if self.ortho_init:
            modules_to_init = {
                self.features_extractor: np.sqrt(2),
                self.mean_head: np.sqrt(2),
                self.rule_head: np.sqrt(2),
                self.glob_critic: 1.0,
            }

            if self.critic_features_extractor is not self.features_extractor:
                modules_to_init[self.critic_features_extractor] = 1.0

            for module, gain in modules_to_init.items():
                module.apply(partial(self.init_weights, gain=gain))

            self.features_extractor.node_scorer.apply(partial(self.init_weights, gain=np.sqrt(2)))

            # output layers have small gain usually.
            self._init_head_output(self.features_extractor.node_scorer, weight_gain=0.01, bias_value=0.0)
            self._init_head_output(self.rule_head, weight_gain=0.01, bias_value=0.0)  # originally 1.0
            self._init_head_output(self.mean_head, weight_gain=0.01, bias_value=0.0)

            if not self.use_global_log_std:
                self.log_std_head.apply(partial(self.init_weights, gain=np.sqrt(2)))

                # # --- set log_std bias (sets initial exploration) ---
                scale = 0.5 * (HybridDistribution.LOG_STD_MAX - HybridDistribution.LOG_STD_MIN)
                bias = 0.5 * (HybridDistribution.LOG_STD_MAX + HybridDistribution.LOG_STD_MIN)
                # convert desired post-squash log std to the raw pre-tanh value used by the head
                init_log_std_pre_squashed = np.clip((self.init_log_std - bias) / scale, -0.999, 0.999)
                init_log_std_pre_squashed = float(np.arctanh(init_log_std_pre_squashed))

                self._init_head_output(self.log_std_head, weight_gain=0.1, bias_value=init_log_std_pre_squashed)

        # Setup optimizer with initial learning rate
        self.optimizer = self.optimizer_class(self.parameters(), lr=lr_schedule(1), **self.optimizer_kwargs)  # type: ignore[call-arg]

    def evaluate_actions(self, obs: PyTorchObs, actions: th.Tensor) -> tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        """
        Evaluate actions according to the current policy,
        given the observations.

        :param obs: Observation
        :param actions: Actions
        :return: estimated value, log likelihood of taking those actions
            and entropy of the action distribution.
        """
        # Preprocess the observation if needed
        features = self.features_extractor(obs)

        node_idx = actions[..., 0].long()  # [B]
        rule_idx = actions[..., 1].long()  # [B]

        if getattr(self, "use_joint_dist", False):
            # Build joint categorical over (node, rule) for the evaluation batch
            index_logits, mask = to_dense_batch(features["node_logits"], features["node_batch"])
            index_logits = index_logits.masked_fill(~mask, float("-inf"))
            node_feats_dense, _ = to_dense_batch(features["node_features"], features["node_batch"])

            B, N = node_feats_dense.shape[0], node_feats_dense.shape[1]
            R = self.n_classes
            rule_logits_per_node = self.rule_head(node_feats_dense.view(B * N, -1)).view(B, N, R)
            rule_logits_per_node = self._mask_rules_if_two_node(rule_logits_per_node, features, expand_nodes=True)
            joint_logits = index_logits.unsqueeze(-1) + rule_logits_per_node  # [B, N, R]
            joint_logits_flat = joint_logits.view(B, N * R)
            joint_dist = Categorical(logits=joint_logits_flat)
            joint_idx = node_idx * R + rule_idx

            node_features_selected = node_feats_dense[th.arange(node_idx.size(0)), node_idx]
            context = node_features_selected
            mean_actions = self.mean_head(context, rule_idx)

            log_std = self._compute_log_std(context, rule_idx)

            distribution = self.action_dist.proba_distribution_joint(
                joint_dist,
                joint_idx,
                node_idx,
                rule_idx,
                mean_actions,
                log_std,
                joint_logits_flat=joint_logits_flat,
                sizes=(N, R),
            )
        else:
            distribution = self._get_action_dist_given_rule(features, node_idx, rule_idx)
        log_prob = distribution.log_prob(actions)

        if self.share_features_extractor:
            critic_features = features
        else:
            critic_features = self.critic_features_extractor(obs)
        values = self._compute_state_values(critic_features)
        entropy = distribution.entropy(actions)
        return values, log_prob, entropy

    def _get_action_dist_given_rule(
        self,
        features: Dict[str, th.Tensor],
        node_idx: th.Tensor,  # [B], long
        rule_idx: th.Tensor,  # [B], long
    ) -> HybridDistribution:

        index_logits, mask = to_dense_batch(features["node_logits"], features["node_batch"])
        index_logits = index_logits.masked_fill(~mask, float("-inf"))  # disallow padding/invalid
        index_dist = Categorical(logits=index_logits)

        node_features, mask = to_dense_batch(features["node_features"], features["node_batch"])
        node_features_selected = node_features[th.arange(node_idx.size(0)), node_idx]  # [B, F_node]

        context = node_features_selected

        rule_logits = self.rule_head(context)
        rule_logits = self._mask_rules_if_two_node(rule_logits, features)

        rule_dist = Categorical(logits=rule_logits)  # no sampling

        mean_actions = self.mean_head(context, rule_idx)  # params for the selected rule

        log_std = self._compute_log_std(context, rule_idx)

        return self.action_dist.proba_distribution(
            index_dist=index_dist,
            index_sampled=node_idx,
            rule_dist=rule_dist,
            rule_idx=rule_idx,  # <-- provided, not sampled
            mean_cont_cond=mean_actions,
            log_std_cond=log_std,
        )

    def forward(
        self, obs: Batch, deterministic: bool = False
    ) -> tuple[th.Tensor, th.Tensor, th.Tensor, dict[str, th.Tensor]]:
        """
        Forward pass in all the networks (actor and critic)

        :param obs: Observation
        :param deterministic: Whether to sample or use deterministic actions
        :return: action, value and log probability of the action
        """
        # Preprocess the observation if needed
        features = self.features_extractor(obs)
        distribution = self._get_action_dist_from_latent(features)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)

        if self.share_features_extractor:
            critic_features = features
        else:
            critic_features = self.critic_features_extractor(obs)
        values = self._compute_state_values(critic_features)

        return actions, values, log_prob, features

    def _get_action_dist_from_latent(self, features: Dict[str, th.Tensor]) -> HybridDistribution:
        """
        Retrieve action distribution given the latent codes.

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """

        index_node_logits, mask = to_dense_batch(features["node_logits"], features["node_batch"])
        index_node_logits = index_node_logits.masked_fill(~mask, float("-inf"))  # disallow padding/invalid
        node_feats_dense, _ = to_dense_batch(features["node_features"], features["node_batch"])

        # Compute per-node, per-rule logits: [B, N, R]
        B, N = node_feats_dense.shape[0], node_feats_dense.shape[1]
        R = self.n_classes  # number of rules -> 4
        # Apply rule head to each node feature

        if self.use_joint_dist:
            joint_logits_per_node = self.rule_head(node_feats_dense.view(B * N, -1)).view(B, N, R)
            joint_logits_per_node = self._mask_rules_if_two_node(joint_logits_per_node, features, expand_nodes=True)
            joint_logits = index_node_logits.unsqueeze(-1) + joint_logits_per_node  # [B, N, R]
            joint_logits_flat = joint_logits.view(B, N * R)

            joint_dist = Categorical(logits=joint_logits_flat)
            joint_idx = joint_dist.sample()  # [B]

            # Decode indices
            node_idx = joint_idx // R  # i.e 31 // 4 = 7 (node 7 selected)
            rule_idx = joint_idx % R  # i.e 31 % 4 = 3 (rule 3 selected)

            # Select context for continuous heads
            node_features_selected = node_feats_dense[th.arange(node_idx.size(0)), node_idx]  # [B, F_node]
            context = node_features_selected

        else:  # Previous behavior
            joint_idx = None
            index_dist = Categorical(logits=index_node_logits)
            node_idx = index_dist.sample()
            node_features_selected = node_feats_dense[th.arange(node_idx.size(0)), node_idx]  # [B, F_node]
            context = node_features_selected
            rule_logits = self.rule_head(context)
            rule_logits = self._mask_rules_if_two_node(rule_logits, features)
            rule_dist = Categorical(logits=rule_logits)
            rule_idx = rule_dist.sample()  # [B]

        mean_actions = self.mean_head(context, rule_idx)

        log_std = self._compute_log_std(context, rule_idx)

        if self.use_joint_dist:
            return self.action_dist.proba_distribution_joint(
                joint_dist,
                joint_idx,
                node_idx,
                rule_idx,
                mean_actions,
                log_std,
                joint_logits_flat=joint_logits_flat,
                sizes=(N, R),
            )
        return self.action_dist.proba_distribution(index_dist, node_idx, rule_dist, rule_idx, mean_actions, log_std)

    def _mask_rules_if_two_node(
        self,
        rule_logits: th.Tensor,
        features: Dict[str, th.Tensor],
        *,
        expand_nodes: bool = False,
    ) -> th.Tensor:
        two_node_mask = features.get("two_node_mask")
        if two_node_mask.numel() == 0 or not th.any(two_node_mask):
            return rule_logits
        invalid_rules = th.ones(rule_logits.shape[-1], dtype=th.bool, device=rule_logits.device)
        invalid_rules[0] = False  # rule 1 is valid
        invalid_rules[4] = False  # rule 5 is valid

        if expand_nodes:
            mask = two_node_mask.view(-1, 1, 1) & invalid_rules.view(1, 1, -1)
            mask = mask.expand(-1, rule_logits.shape[1], -1)
        else:
            mask = two_node_mask.unsqueeze(-1) & invalid_rules.view(1, -1)
        return rule_logits.masked_fill(mask, float("-inf"))

    def _compute_log_std(self, context: th.Tensor, rule_idx: th.Tensor) -> th.Tensor:
        if self.use_global_log_std:
            log_std = self.log_std_param.unsqueeze(0).expand(context.shape[0], -1)
        else:
            log_std = self.log_std_head(context, rule_idx)
        return HybridDistribution.squash_log_std(log_std)

    def predict(
        self,
        observation: Data,
        deterministic: bool = False,
    ) -> np.ndarray:
        """
        Get the policy action from an observation (and optional hidden state).
        Includes sugar-coating to handle different observations (e.g. normalizing images).

        :param observation: the input observation
        :param deterministic: if deterministic or not :P
        :return: the model's action
        """
        # Switch to eval mode (this affects batch norm / dropout)
        self.set_training_mode(False)

        obs_tensor = Batch.from_data_list([observation]).to(self.device)

        with th.no_grad():
            features = self.features_extractor(obs_tensor)
            actions = self._get_action_dist_from_latent(features).get_actions(deterministic=deterministic)
        # Convert to numpy, and reshape to the original action shape
        actions = actions.cpu().numpy().flatten()
        return actions

    def predict_values(self, obs: PyTorchObs) -> th.Tensor:
        """
        Get the estimated values according to the current policy given the observations.

        :param obs: Observation
        :return: the estimated values.
        """
        features = self.critic_features_extractor(obs)
        return self._compute_state_values(features)

    def _compute_state_values(self, features: Dict[str, th.Tensor]) -> th.Tensor:
        """Aggregate per-node values into a scalar state value."""

        global_features = features["global_features"]
        node_batch = features["node_batch"]

        graph_counts = th.bincount(node_batch, minlength=global_features.shape[0]).float().unsqueeze(-1)
        global_features = global_features + self.node_count_embed(th.log1p(graph_counts))

        value_glob = self.glob_critic(global_features)
        return value_glob.flatten()

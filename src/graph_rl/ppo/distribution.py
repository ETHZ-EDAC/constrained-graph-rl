# Standard library
from collections import deque
from typing import Optional, Tuple

# Third-party
import torch as th
import torch.distributions
import torch.nn as nn
from torch.distributions import Categorical

# First-party
from graph_rl.ppo.env import get_masked_action_dim, RULE_SPECS
from graph_rl.sb3_fork.base_distribution import (
    DiagGaussianDistribution,
    sum_independent_dims,
)


class MlpHeads(nn.Module):
    """
    Flexible MLP heads for hybrid distributions:
      - mean_head: predicts per-class means for the Gaussian (shape: K*D)
      - logits_head: predicts logits for the categorical (shape: K)
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        out_dim: int = 15,
        num_conditionals: int = 0,
        embed_dim: Optional[int] = None,
        num_layers: int = 2,
    ):
        super().__init__()

        layers = []
        input_dim = in_dim

        if num_conditionals == 1:
            self.embedding = nn.Embedding(len(RULE_SPECS), embed_dim)
            input_dim += embed_dim

        # Create hidden layers dynamically
        for i in range(num_layers - 1):
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.SiLU())
            input_dim = hidden_dim

        # Final output layer
        layers.append(nn.Linear(input_dim, out_dim))
        self.head = nn.Sequential(*layers)

    def forward(self, latent: th.Tensor, conditional: Optional[th.Tensor] = None) -> th.Tensor:
        if conditional is not None:
            cond_vec1 = self.embedding(conditional.long())
            x = th.cat([latent, cond_vec1], dim=-1)
        else:
            x = latent
        return self.head(x)


class HybridDistribution(nn.Module):
    """
    Hybrid distribution for PPO policies with conditional continuous head:
      - One categorical action with N classes
      - K continuous actions, parameterized *per class* (diagonal Gaussian, tanh-squashed if your DiagGaussianDistribution does that)

    Returned action: [ cat_idx_0_based , cont_0 , cont_1 , ... ]

    Network heads output (B = batch, K = n_classes, D = cont_dim):
      - logits_cat:   [B, K]
      - mean_cont:    [B, K*D]
      - log_std_all:  [B, K*D]  (STATE mode)  or  [K*D] (GLOBAL mode)

    std_mode:
      - "state": state-dependent log-std via a learned head (per-state, per-class, per-dim)
      - "global": single Parameter shared across batch (your current behavior)

    """

    LOG_STD_MAX = 0.5
    LOG_STD_MIN = -4.5

    def __init__(
        self,
        weight_entropy_node: float = 1.0,
        weight_entropy_cat: float = 1.0,
        weight_entropy_cont: float = 1.0,
        weight_entropy_joint: float = 1.0,
        **kwargs,
    ):
        super().__init__()
        # Toggle to use joint categorical over (node, rule)
        self.use_joint_dist: bool = bool(kwargs.get("use_joint_dist", True))
        self.cont_dim_max = max(rule.dim_cont for rule in RULE_SPECS)
        self.n_classes = len(RULE_SPECS)

        self.weight_entropy_node = weight_entropy_node
        self.weight_entropy_cat = weight_entropy_cat
        self.weight_entropy_cont = weight_entropy_cont
        self.weight_entropy_joint = weight_entropy_joint

        self.log_std_deque = deque(maxlen=1000)

        continuous_mask = get_masked_action_dim(self.n_classes, self.cont_dim_max)
        self.register_buffer("continuous_mask", continuous_mask)

        # Cached tensors (set in proba_distribution)
        self._means_conditional: Optional[th.Tensor] = None
        self._log_stds_conditional: Optional[th.Tensor] = None

        # Distributions (set in proba_distribution)
        self.rule_dist: Optional[Categorical] = None
        self.index_dist: Optional[Categorical] = None
        self.joint_dist: Optional[Categorical] = None
        self.rule_idx_sampled: Optional[th.Tensor] = None
        self.index_sampled: Optional[th.Tensor] = None
        self.joint_idx_sampled: Optional[th.Tensor] = None
        # Joint logits cache for top-k extraction
        self.joint_logits_flat: Optional[th.Tensor] = None  # [B, N*R]
        self.joint_sizes: Optional[Tuple[int, int]] = None  # (N, R)

        # Helper DiagGaussianDistribution (SB3-style) reused with per-batch selected params
        self.cont_dist = DiagGaussianDistribution(self.cont_dim_max)

    @classmethod
    def squash_log_std(cls, raw_log_std: th.Tensor) -> th.Tensor:
        """Bound state-dependent log stds to a safe range via tanh squashing."""

        # Map tanh output (-1, 1) to [LOG_STD_MIN, LOG_STD_MAX]
        scale = 0.5 * (cls.LOG_STD_MAX - cls.LOG_STD_MIN)
        bias = 0.5 * (cls.LOG_STD_MAX + cls.LOG_STD_MIN)
        return th.tanh(raw_log_std) * scale + bias

    def proba_distribution(
        self,
        index_dist: torch.distributions.Categorical,
        index_sampled: th.Tensor,
        rule_dist: torch.distributions.Categorical,
        rule_idx: th.Tensor,
        mean_cont_cond: th.Tensor,  # [B, cont_dim_max]
        log_std_cond: th.Tensor,  # [B, cont_dim_max]
    ) -> "HybridDistribution":
        """
        Instantiate the categorical + store all per-class Gaussian params.
        """

        self.index_dist = index_dist
        self.index_sampled = index_sampled
        self.rule_dist = rule_dist
        self.rule_idx_sampled = rule_idx
        self._means_conditional = mean_cont_cond
        self._log_stds_conditional = log_std_cond

        return self

    def proba_distribution_joint(
        self,
        joint_dist: torch.distributions.Categorical,
        joint_idx: th.Tensor,
        node_idx: th.Tensor,
        rule_idx: th.Tensor,
        mean_cont_cond: th.Tensor,
        log_std_cond: th.Tensor,
        joint_logits_flat: Optional[th.Tensor] = None,
        sizes: Optional[Tuple[int, int]] = None,
    ) -> "HybridDistribution":
        """Variant using a single joint categorical over node-rule pairs."""
        self.joint_dist = joint_dist
        self.joint_idx_sampled = joint_idx
        self.index_sampled = node_idx
        self.rule_idx_sampled = rule_idx
        self._means_conditional = mean_cont_cond
        self._log_stds_conditional = log_std_cond
        # Cache logits and sizes for later top-k queries. More efficient than recomputing
        self.joint_logits_flat = (
            joint_logits_flat if joint_logits_flat is not None else getattr(joint_dist, "logits", None)
        )
        self.joint_sizes = sizes  # useful to decode both node and rule index
        return self

    # ------- Utilities -------

    def _select_params_for_class(self, class_idx: th.Tensor) -> Tuple[th.Tensor, th.Tensor]:
        """
        Pick [mean, log_std] for the given categorical indices.
        class_idx: [B] int64
        returns mean_sel, log_std_sel: [B, D_total]

        Note: We keep a fixed action dimension (cont_dim_total) for the batch and
        simply mask out the irrelevant dimensions for each selected class. This
        avoids shape-mismatch issues when different samples in the batch have
        different continuous dims.
        """
        assert self._means_conditional is not None and self._log_stds_conditional is not None

        # Means are state-dependent and already shaped [B, D_total]
        mean_all = self._means_conditional  # [B, D_total]
        mean_sel = mean_all.clone()

        # Zero-out means on irrelevant dims
        # Mask out dynamic size of continuous domain:
        mask = self.continuous_mask[class_idx]
        mean_sel[~mask] = 0.0

        # Log-stds: state-dependent or global shared vec
        # [B, D_total]
        log_std_cond = self._log_stds_conditional  # type: ignore

        # Make irrelevant dims effectively deterministic
        # log_std_cond[~mask] = -20.0

        self.log_std_deque.extend(log_std_cond[mask].detach().cpu().numpy().flatten().tolist())

        return mean_sel, log_std_cond

    # ------- Core log-prob / entropy -------

    def log_prob(self, actions: th.Tensor) -> th.Tensor:
        """
        Joint log-prob for hybrid actions.
        Expected actions format: [ cat_idx_0_based , cont_0 , cont_1 , ... ]
        """
        assert self.rule_dist is not None or self.joint_dist is not None, "call proba_distribution(...) first"
        assert self._means_conditional is not None and self._log_stds_conditional is not None, (
            "call proba_distribution(...) first"
        )

        node_raw = actions[:, 0].long()  # [B]
        cat_raw = actions[:, 1]  # [B]

        # Joint log-prob over (node, rule) if available, else fallback to separate
        if self.joint_dist is not None and self.use_joint_dist:
            # encode joint index = node * R + rule
            joint_idx = node_raw * self.n_classes + cat_raw

            # jointed categorical log-prob
            log_prob_joint = self.joint_dist.log_prob(joint_idx)

            # continuous log-prob
            log_cont_full = self.log_prob_continuous(actions)  # [B]

            return log_prob_joint + log_cont_full
        else:
            # clamp categorical range (you may prefer to assert in training)
            cat_idx = cat_raw.long().clamp(min=0, max=self.n_classes - 1)

            log_prob_node = self.index_dist.log_prob(node_raw)

            # categorical log-prob
            log_prob_cat = self.rule_dist.log_prob(cat_idx)  # [B]

            # continuous log-prob
            log_cont_full = self.log_prob_continuous(actions)  # [B]

            return log_prob_node + log_prob_cat + log_cont_full

    def log_prob_continuous(self, actions: th.Tensor) -> th.Tensor:
        """
        Log-prob for continuous part only.
        Expected actions format: [ cat_idx_0_based , cont_0 , cont_1 , ... ]
        """
        assert self.rule_dist is not None or self.joint_dist is not None, "call proba_distribution(...) first"
        assert self._means_conditional is not None and self._log_stds_conditional is not None, (
            "call proba_distribution(...) first"
        )

        cat_raw = actions[:, 1]  # [B]
        cont = actions[:, 2:]  # [B, D]

        # clamp categorical range (you may prefer to assert in training)
        cat_idx = cat_raw.long().clamp(min=0, max=self.n_classes - 1)

        # select conditional Gaussian params for these indices
        mean_sel, log_std_sel = self._select_params_for_class(cat_idx)

        # build per-batch Gaussian and get log-prob of cont
        cont_dist = self.cont_dist.proba_distribution(mean_sel, log_std_sel)

        log_prob_cont = cont_dist.log_prob(cont)  # [B, D_max]
        log_prob_cont_masked = log_prob_cont * self.continuous_mask[cat_idx]
        return sum_independent_dims(log_prob_cont_masked)

    def entropy(self, actions) -> th.Tensor:
        """
        H = H(C) + E_{c~Cat}[ H(Cont | c) ]
        H(X,Y) = H(X) + H(Y|X)  # entropy chain rule

        H(Y|X) = E_{x~X}[ H(Y|X=x) ]  # conditional entropy definition
            = - sum_{x,y} p(x,y) * log p(y|x)

        Returns [B].
        Approximates the tanh-squashed case with the unsquashed Normal entropy
        (same approximation SB3 uses).
        """

        assert self.rule_dist is not None or self.joint_dist is not None
        assert self._means_conditional is not None and self._log_stds_conditional is not None

        # Categorical entropy: [B]
        assert self.rule_idx_sampled is not None
        cat_idx = self.rule_idx_sampled
        entropy_cont = self.cont_dist.entropy()  # [B, D_max]
        entropy_cont_masked = entropy_cont * self.continuous_mask[cat_idx]
        entropy_cont = sum_independent_dims(entropy_cont_masked)

        if self.joint_dist is not None and self.use_joint_dist:
            ent_joint = self.joint_dist.entropy()
            return self.weight_entropy_joint * ent_joint + self.weight_entropy_cont * entropy_cont
        else:
            ent_node = self.index_dist.entropy()
            ent_cat = self.rule_dist.entropy()
            return (
                self.weight_entropy_node * ent_node
                + self.weight_entropy_cat * ent_cat
                + self.weight_entropy_cont * entropy_cont
            )

    # ------- Sampling / mode -------

    def sample(self) -> th.Tensor:
        """
        Sample c ~ Cat, then a ~ N(mean_c, std_c). Returns [B, 1 + D]
        """
        assert self.rule_idx_sampled is not None
        assert self.index_sampled is not None
        assert self._means_conditional is not None and self._log_stds_conditional is not None

        node_idx = self.index_sampled
        cat_idx = self.rule_idx_sampled
        mean_sel, log_std_sel = self._select_params_for_class(cat_idx)
        cont_dist = self.cont_dist.proba_distribution(mean_sel, log_std_sel)

        cont = cont_dist.sample()  # [B, D]

        return th.cat([node_idx.float().unsqueeze(1), cat_idx.float().unsqueeze(1), cont], dim=1)

    def mode(self) -> th.Tensor:
        """
        Deterministic action: c_mode = argmax logits, cont_mode = mode(mean_c)
        """
        assert self.rule_dist is not None
        assert self._means_conditional is not None and self._log_stds_conditional is not None

        with th.no_grad():
            cat_idx = th.argmax(self.rule_dist.logits, dim=-1)  # [B]
            mean_sel, log_std_sel = self._select_params_for_class(cat_idx)
            cont_dist = self.cont_dist.proba_distribution(mean_sel, log_std_sel)
            cont = cont_dist.mode()  # [B, D]
            return th.cat([cat_idx.float().unsqueeze(1), cont], dim=1)

    # ------- Glue for actor-critic calls -------

    def get_actions(self, deterministic: bool = False) -> th.Tensor:
        return self.mode() if deterministic else self.sample()

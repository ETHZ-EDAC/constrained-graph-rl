# Standard library
import sys
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

# Third-party
import torch
import torch.nn as nn
from torch_geometric.data import Batch
from torch_geometric.utils import to_dense_batch
from tqdm import tqdm

# First-party
from graph_rl.ppo.env import get_masked_action_dim, RULE_SPECS
from graph_rl.ppo.layers import BaseFeatureExtractor


class GraphActionProjector(nn.Module):
    """Learnable action projection g(s, a_raw) -> a_proj with log|det J_g|.

    The map is a per-dimension affine transform conditioned on (state, rule, node)
    followed by a tanh squashing to stay inside [-1, 1]. This keeps the Jacobian
    tractable: diag(scale) * diag(1 - tanh(z)^2).
    """

    def __init__(
        self,
        features_extractor_class: Optional[type[BaseFeatureExtractor]] = None,
        features_extractor_kwargs: Optional[Dict[str, Any]] = None,
        hidden_dim: int = 128,
        embed_dim: int = 64,
        cont_dim: Optional[int] = None,
        max_log_scale: float = 2,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()

        self.eps = eps
        self.max_log_scale = max_log_scale

        self.cont_dim = cont_dim if cont_dim is not None else max(spec.dim_cont for spec in RULE_SPECS)

        self.features_extractor = features_extractor_class(**features_extractor_kwargs, action_projector=True)

        self.rule_emb = nn.Embedding(len(RULE_SPECS), embed_dim)

        self.state_mlp = nn.Sequential(
            nn.Linear(int(self.features_extractor.node_features_dim) + embed_dim, hidden_dim),
            nn.SiLU(),
        )

        # Predict per-dimension shift and log-scale (diagonal affine) from state only
        self.affine_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2 * self.cont_dim),
        )
        # Initialize to identity: shift=0, log_scale=0 (scale=1) regardless of input.
        self._init_affine_head()

        self.cont_mask = get_masked_action_dim(len(RULE_SPECS), self.cont_dim)

    def _init_affine_head(self) -> None:
        last = self.affine_head[-1]
        if isinstance(last, nn.Linear):
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)

    def forward(
        self, batch: Batch, actions_raw: torch.Tensor, return_aux: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, Dict[str, torch.Tensor]]:
        """Project actions and return (actions_projected, log_abs_det_jacobian).

        If `features` is provided (from the policy forward pass) we reuse it;
        otherwise we run the optional local feature extractor.
        """

        feats = self.features_extractor(batch)

        node_idx = actions_raw[..., 0].long()
        rule_idx = actions_raw[..., 1].long()
        cont_raw = actions_raw[..., 2:]

        node_features, mask = to_dense_batch(feats["node_features"], feats["node_batch"])
        batch_size, num_nodes, _ = node_features.shape
        node_idx_clamped = node_idx.clamp(max=num_nodes - 1)
        node_selected = node_features[torch.arange(batch_size, device=node_idx.device), node_idx_clamped]

        cont_mask = self.cont_mask.to(cont_raw.device)[rule_idx]
        cont_masked = cont_raw * cont_mask

        context = torch.cat((node_selected, self.rule_emb(rule_idx)), dim=-1)
        state_latent = self.state_mlp(context)

        shift, log_scale_raw = self.affine_head(state_latent).chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale_raw) * self.max_log_scale
        scale = torch.exp(log_scale)

        z = scale * cont_masked + shift
        cont_proj = torch.tanh(z)
        cont_proj = cont_proj * cont_mask

        # log|det J| = sum log|scale_i * (1 - tanh(z_i)^2)| on active dims
        cont_proj_clamped = cont_proj.clamp(min=-1.0 + self.eps, max=1.0 - self.eps)
        log_abs_det = (log_scale + torch.log(1 - cont_proj_clamped.pow(2) + self.eps)) * cont_mask
        log_abs_det = log_abs_det.sum(dim=-1)

        actions_projected = torch.cat(
            [node_idx.float().unsqueeze(-1), rule_idx.float().unsqueeze(-1), cont_proj], dim=-1
        )

        if return_aux:
            aux = {"scale": scale, "shift": shift, "cont_mask": cont_mask}
            return actions_projected, log_abs_det, aux
        return actions_projected, log_abs_det


def compute_projection_supervised_loss(
    actions_nominal: torch.Tensor,
    actions_projected_env: torch.Tensor,
    actions_projected_pred: torch.Tensor,
) -> torch.Tensor:
    """Supervise projected continuous actions against the env-corrected target."""

    diff = torch.abs(actions_nominal[:, 2:] - actions_projected_env[:, 2:])
    # If any dim was changed, weight the whole sample as 1.0 else 0.5
    row_changed = (diff > 1e-5).any(dim=1, keepdim=True)
    weight = torch.where(row_changed, torch.ones_like(diff), torch.full_like(diff, 0.5))
    return ((actions_projected_pred[:, 2:] - actions_projected_env[:, 2:]) ** 2 * weight).mean()


def train_action_projector(
    projector: "GraphActionProjector",
    optimizer: torch.optim.Optimizer,
    replay_buffer: Sequence[Batch],
    device: torch.device,
    projection_supervised_epochs: int,
    max_grad_norm: float,
    ppo_logger: Optional[Any] = None,
    update_lr_fn: Optional[Callable[[torch.optim.Optimizer], None]] = None,
    set_trainable_fn: Optional[Callable[[bool], None]] = None,
    show_progress: bool = True,
) -> tuple[Optional[float], Optional[float]]:
    """Run supervised training of the action projector.

    Returns average train/validation losses (per graph) if available.
    """

    if len(replay_buffer) == 0 or projection_supervised_epochs <= 0:
        return None, None

    def _set_trainable(flag: bool) -> None:
        if set_trainable_fn is not None:
            set_trainable_fn(flag)
            return
        for param in projector.parameters():
            param.requires_grad_(flag)

    projector.train()
    _set_trainable(True)

    if update_lr_fn is not None:
        update_lr_fn(optimizer)

    total_loss = 0.0
    total_count = 0
    scale_samples: list[torch.Tensor] = []
    shift_samples: list[torch.Tensor] = []

    batches = list(replay_buffer)
    val_batches = batches[-2:]
    train_batches = batches[:-2]

    pbar = None
    if show_progress:
        pbar = tqdm(
            total=projection_supervised_epochs,
            desc="Projector epochs",
            leave=True,
            file=sys.__stdout__,
        )

    for _ in range(projection_supervised_epochs):
        for batch in train_batches:
            batch = batch.to(device)
            proj_pred, _, aux = projector(batch, batch.actions_nominal, return_aux=True)
            cont_mask = aux["cont_mask"].detach()
            active_mask = cont_mask.bool()
            if active_mask.any():
                scale_samples.append(aux["scale"].detach()[active_mask].reshape(-1))
                shift_samples.append(aux["shift"].detach()[active_mask].reshape(-1))
            loss = compute_projection_supervised_loss(batch.actions_nominal, batch.actions_projected_env, proj_pred)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(projector.parameters(), max_grad_norm)
            optimizer.step()

            total_loss += float(loss.item()) * batch.num_graphs
            total_count += batch.num_graphs
        if pbar is not None:
            pbar.update(1)

    if pbar is not None:
        pbar.close()

    val_loss = None
    if len(val_batches) > 0:
        projector.eval()
        with torch.no_grad():
            v_loss = 0.0
            v_count = 0
            for val_batch in val_batches:
                val_batch = val_batch.to(device)
                proj_pred, _ = projector(val_batch, val_batch.actions_nominal)
                loss = compute_projection_supervised_loss(
                    val_batch.actions_nominal, val_batch.actions_projected_env, proj_pred
                )
                v_loss += float(loss.item()) * val_batch.num_graphs
                v_count += val_batch.num_graphs
            if v_count > 0:
                val_loss = v_loss / v_count

    train_loss = total_loss / total_count if total_count > 0 else None

    if ppo_logger is not None and train_loss is not None:
        ppo_logger.record("action_projector/projection_loss_supervised", train_loss)
    if ppo_logger is not None and val_loss is not None:
        ppo_logger.record("action_projector/projection_loss_val", val_loss)
    if ppo_logger is not None and scale_samples and shift_samples:
        scale_hist = torch.cat(scale_samples, dim=0)
        shift_hist = torch.cat(shift_samples, dim=0)
        ppo_logger.record("action_projector/scale", scale_hist, exclude="stdout")
        ppo_logger.record("action_projector/shift", shift_hist, exclude="stdout")

    _set_trainable(False)
    projector.eval()

    return train_loss, val_loss

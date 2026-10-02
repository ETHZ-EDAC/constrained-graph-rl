# Third-party
import torch
import torch as th
from torch import nn
from torch.nn import Embedding
from torch_geometric.data import Batch
from torch_geometric.nn import (
    AttentionalAggregation,
    LayerNorm,
    MessagePassing,
    global_mean_pool,
)

# First-party
from graph_rl.utils import PARAMS


class BaseFeatureExtractor(nn.Module):
    """Base class for feature extractors."""

    pass


class CoorsNorm(nn.Module):
    """Normalizes relative coordinates to stabilize distance features."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.full((1,), 1e-1))

    def forward(self, rel: torch.Tensor) -> torch.Tensor:
        norm = rel.norm(dim=-1, keepdim=True)
        rel_normed = rel / norm.clamp(min=self.eps)
        return rel_normed * self.scale


class EGNNLayer(MessagePassing):
    """
    Standard Local EGNN Layer
    """

    def __init__(
        self,
        base_in_channels: int,
        out_channels: int,
        edge_attr_dim: int = 0,
        coord_dim: int = 3,
        hidden_dim: int = 64,
        fourier_features: int = 0,
        use_coord_norm: bool = True,
    ):
        super().__init__(aggr="sum")

        self.coord_dim = coord_dim
        self.hidden_dim = hidden_dim
        self.out_channels = out_channels
        self.fourier_features = fourier_features
        self.coord_norm = CoorsNorm() if use_coord_norm else nn.Identity()

        self.node_in_channels = base_in_channels
        self.edge_attr_total_dim = edge_attr_dim
        self.dist_feat_dim = 1 + (2 * fourier_features if fourier_features > 0 else 0)

        edge_input_dim = 2 * self.node_in_channels + self.edge_attr_total_dim + self.dist_feat_dim

        self.edge_mlp = nn.Sequential(
            nn.Linear(edge_input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
        )

        self.node_mlp = nn.Sequential(
            nn.Linear(self.node_in_channels + hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, out_channels),
        )

        self.coord_mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor = None,
        **kwargs,
    ):
        x_lol = x[:, : -self.coord_dim]
        pos = x[:, -self.coord_dim :]

        msg = self.propagate(edge_index, x=x_lol, pos=pos, edge_attr=edge_attr)

        m_agg = msg[:, : self.hidden_dim]
        coord_delta = msg[:, self.hidden_dim :]
        pos = pos + coord_delta

        h_cat = torch.cat([x_lol, m_agg], dim=-1)
        h_l1 = self.node_mlp(h_cat)

        return torch.cat((h_l1, pos), dim=-1)

    @staticmethod
    def fourier_encode_dist(x, num_encodings=4, include_self=True):
        x = x.unsqueeze(-1)
        device, dtype, orig_x = x.device, x.dtype, x
        scales = 2 ** torch.arange(num_encodings, device=device, dtype=dtype)
        x = x / scales
        x = torch.cat([x.sin(), x.cos()], dim=-1)
        x = torch.cat((x, orig_x), dim=-1) if include_self else x
        return x

    def message(self, x_i, x_j, pos_i, pos_j, edge_attr):
        """
        For each edge j -> i:
          - compute relative coordinates and distance
          - build invariant edge features
          - compute message m_ij = φ_e(...)
          - compute coordinate step from m_ij
        """
        rel_pos = pos_i - pos_j  # [E, coord_dim]
        # rel_pos = self.coord_norm(rel_pos)
        sq_dist = (rel_pos**2).sum(dim=-1, keepdim=False)  # [E, 1]

        if self.fourier_features > 0:
            dist_feats = self.fourier_encode_dist(sq_dist, self.fourier_features, include_self=True)
        else:
            dist_feats = sq_dist

        if edge_attr is not None:
            edge_input = torch.cat([x_i, x_j, edge_attr, dist_feats], dim=-1)
        else:
            edge_input = torch.cat([x_i, x_j, dist_feats], dim=-1)

        m_ij = self.edge_mlp(edge_input)
        coord_coeff = self.coord_mlp(m_ij)
        rel_pos = self.coord_norm(rel_pos)
        coord_update = coord_coeff * rel_pos
        return torch.cat([m_ij, coord_update], dim=-1)

    def update(self, aggr_out):
        # we keep update trivial; splitting & MLPs happen in forward
        return aggr_out


class GraphSizeAwareJK(nn.Module):
    """Jumping Knowledge conditioned on graph size."""

    def __init__(self, feature_dim: int, num_layers: int, gate_hidden: int = 128):
        super().__init__()
        self.num_layers = num_layers
        self.feature_dim = feature_dim
        self.gate = nn.Sequential(
            nn.Linear(feature_dim + 1, gate_hidden),
            nn.SiLU(),
            nn.Linear(gate_hidden, 1),
        )

    def forward(self, layer_outputs: list[torch.Tensor], batch: torch.Tensor):
        """
        layer_outputs: list of [N, F] tensors (including pre- and post-layer skips)
        batch: [N] batch assignment
        """
        assert len(layer_outputs) == self.num_layers, "Mismatch between configured JK layers and inputs."
        if layer_outputs[0].numel() == 0:
            return layer_outputs[-1], None

        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 0
        # Graph sizes (log-scaled) drive the gate to favor shallow/deep layers.
        node_counts = torch.bincount(batch, minlength=batch_size).clamp(min=1).float()
        log_sizes = (
            torch.log1p(node_counts).unsqueeze(-1).to(device=layer_outputs[0].device, dtype=layer_outputs[0].dtype)
        )  # [B, 1]

        pooled = [global_mean_pool(h, batch) for h in layer_outputs]  # each [B, F]
        gate_scores = [self.gate(torch.cat([p, log_sizes], dim=-1)).squeeze(-1) for p in pooled]  # list of [B]
        gate_scores = torch.stack(gate_scores, dim=0)  # [L, B]
        gate_weights = torch.softmax(gate_scores, dim=0)  # sum over layers -> 1 per graph

        fused = torch.zeros_like(layer_outputs[0])
        for layer_index, h in enumerate(layer_outputs):
            fused = fused + gate_weights[layer_index][batch].unsqueeze(-1) * h

        return fused, gate_weights


class GraphFeatureExtractor(BaseFeatureExtractor):
    """
    Geometry-aware extractor for graphs represented by sector angles (nodes)
    and edge lengths (edges). Fully SE(2)-invariant.
    """

    def __init__(
        self,
        hidden_dim=128,
        embed_dim_deg=16,
        num_layers=3,
        num_freqs=4,
        action_projector=False,
    ):
        super().__init__()

        self.deg_embed = Embedding(num_embeddings=PARAMS.max_vertex_degree + 1, embedding_dim=embed_dim_deg)

        sectors_dim = num_freqs * 2
        discrete_dim = 1 + embed_dim_deg  # boundary + degree
        in_channels = sectors_dim + discrete_dim

        self.lifting_layer = nn.Sequential(
            nn.Linear(in_channels, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

        self.num_layers = num_layers
        self.layers = nn.ModuleList()
        self.graph_norms = nn.ModuleList()

        for _ in range(num_layers):
            egnn = EGNNLayer(
                base_in_channels=hidden_dim,
                out_channels=hidden_dim,
                edge_attr_dim=0,
                coord_dim=2,
                hidden_dim=hidden_dim,
                fourier_features=num_freqs,
                use_coord_norm=True,
            )
            self.layers.append(egnn)
            self.graph_norms.append(LayerNorm(hidden_dim))

        self.feature_dim = hidden_dim
        self.jumping_knowledge = GraphSizeAwareJK(
            feature_dim=self.feature_dim, num_layers=num_layers + 1, gate_hidden=hidden_dim
        )

        self._global_features_dim = self.feature_dim
        self._node_features_dim = self.feature_dim

        self.action_projector = action_projector
        if not self.action_projector:
            self._node_features_dim += self._global_features_dim

            # we only need this for the feature extractor for the actor critic
            self.attn_pool = AttentionalAggregation(
                gate_nn=nn.Sequential(
                    nn.Linear(self.feature_dim, hidden_dim),
                    nn.SiLU(),
                    nn.Linear(hidden_dim, 1),
                )
            )

            self.node_scorer = nn.Sequential(
                nn.Linear(self._node_features_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, 1),
            )

    @property
    def node_features_dim(self):
        return self._node_features_dim

    @property
    def glob_features_dim(self):
        return self._global_features_dim

    def forward(self, batch: Batch):

        sectors = batch.x[:, 2:]
        deg = batch.x[:, 1].long()
        boundary = batch.x[:, 0].long()

        discrete_feats = [boundary.view(-1, 1)]
        discrete_feats.append(self.deg_embed(deg))

        h = th.cat([sectors] + discrete_feats, dim=-1)

        # Lift local features
        h_h = self.lifting_layer(h)
        pos = batch.pos

        layer_outputs = [h_h]
        for layer, norm in zip(self.layers, self.graph_norms):
            inp_egnn = th.cat((h_h, pos), dim=-1)
            out_egnn = layer(
                x=inp_egnn,
                edge_index=batch.edge_index,
                edge_attr=None,
            )
            feat_out_local = out_egnn[:, :-2]
            coord_out_local = out_egnn[:, -2:]
            h_h = h_h + feat_out_local
            h_h = norm(h_h, batch.batch)

            # Update coords
            pos = coord_out_local
            layer_outputs.append(h_h)

        H_feat, _ = self.jumping_knowledge(layer_outputs, batch.batch)

        if self.action_projector:
            # if action projector, we do not compute global features or node logits
            return {
                "node_features": H_feat,
                "node_batch": batch.batch,
            }

        else:
            global_feature = self.attn_pool(H_feat, batch.batch) + global_mean_pool(H_feat, batch.batch)
            H_feat = th.cat((H_feat, global_feature[batch.batch]), dim=-1)
            node_logits = self.node_scorer(H_feat).flatten()
            node_logits = node_logits.masked_fill(~boundary.bool(), float("-inf"))

            # if we have only two nodes (first step), mask out the second node, and only make node 0 viable.
            # Otherwise grammar has problems...
            graph_sizes = batch.ptr[1:] - batch.ptr[:-1]
            two_node_mask = graph_sizes == 2
            node_logits[batch.ptr[:-1][two_node_mask] + 1] = float("-inf")

            return {
                "node_features": H_feat,
                "global_features": global_feature,
                "node_logits": node_logits,
                "node_batch": batch.batch,
                "two_node_mask": two_node_mask,
            }

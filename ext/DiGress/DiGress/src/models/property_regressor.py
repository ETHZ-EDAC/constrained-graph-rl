"""Property regressor for predicting graph metrics."""

import torch
import torch.nn as nn
from torch_geometric.nn import GCNConv, global_add_pool, global_mean_pool
from typing import Dict, Tuple, Optional


class PropertyRegressor(nn.Module):
    """
    Regressor for predicting graph-level properties (metrics) from graphs.
    Uses a simple GCN encoder + MLP head architecture.
    """
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 128,
        num_layers: int = 3,
        num_metrics: int = 5,
        output_range: Optional[Dict[str, Tuple[float, float]]] = None,
        dropout: float = 0.1,
    ):
        """
        Args:
            input_dim: input node feature dimension
            hidden_dim: hidden dimension for GCN layers
            num_layers: number of GCN layers
            num_metrics: number of metrics to predict
            output_range: dict mapping metric names to (min, max) ranges for normalization
            dropout: dropout rate
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_metrics = num_metrics
        self.output_range = output_range or {
            'num_nodes': (1, 500),
            'spectral_gap': (0.0, 2.0),
            'gini_coefficient': (0.0, 1.0),
            'clustering_coefficient': (0.0, 1.0),
            'triangle_count': (0, 10000),
        }
        self.dropout = dropout
        
        # GCN encoder
        self.gcn_layers = nn.ModuleList()
        self.gcn_layers.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 1):
            self.gcn_layers.append(GCNConv(hidden_dim, hidden_dim))
        
        # Batch norm and dropout
        self.batch_norms = nn.ModuleList([nn.BatchNorm1d(hidden_dim) for _ in range(num_layers)])
        self.relu = nn.ReLU()
        self.dropout_layer = nn.Dropout(dropout)
        
        # MLP head: maps pooled graph embedding to metric predictions
        mlp_hidden = hidden_dim * 2
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, mlp_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, num_metrics),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x: node features, shape (num_nodes, input_dim)
            edge_index: edge indices, shape (2, num_edges)
            batch: batch assignment for graphs, shape (num_nodes,)
        
        Returns:
            predictions: shape (batch_size, num_metrics)
        """
        # GCN encoding
        for i, gcn in enumerate(self.gcn_layers):
            x = gcn(x, edge_index)
            x = self.batch_norms[i](x)
            x = self.relu(x)
            x = self.dropout_layer(x)
        
        # Graph pooling
        if batch is None:
            # Single graph
            graph_embedding = global_mean_pool(x, batch=torch.zeros(x.size(0), dtype=torch.long, device=x.device))
        else:
            # Multiple graphs
            graph_embedding = global_mean_pool(x, batch=batch)
        
        # MLP head
        predictions = self.mlp(graph_embedding)  # shape: (batch_size, num_metrics)
        
        return predictions
    
    def normalize_metrics(self, metrics: torch.Tensor, metric_names: Optional[list] = None) -> torch.Tensor:
        """
        Normalize predicted metrics to [0, 1] range using stored output ranges.
        
        Args:
            metrics: tensor of shape (batch_size, num_metrics)
            metric_names: list of metric names in order (default: alphabetical)
        
        Returns:
            normalized metrics in [0, 1]
        """
        if metric_names is None:
            metric_names = [
                'clustering_coefficient',
                'gini_coefficient',
                'num_nodes',
                'spectral_gap',
                'triangle_count',
            ]
        
        normalized = metrics.clone()
        for i, name in enumerate(metric_names):
            if name in self.output_range:
                min_val, max_val = self.output_range[name]
                normalized[:, i] = (metrics[:, i] - min_val) / (max_val - min_val + 1e-8)
                normalized[:, i] = torch.clamp(normalized[:, i], 0, 1)
        
        return normalized
    
    def denormalize_metrics(
        self,
        normalized_metrics: torch.Tensor,
        metric_names: Optional[list] = None
    ) -> torch.Tensor:
        """
        Denormalize metrics from [0, 1] back to original ranges.
        
        Args:
            normalized_metrics: tensor of shape (batch_size, num_metrics) in [0, 1]
            metric_names: list of metric names in order
        
        Returns:
            denormalized metrics in original ranges
        """
        if metric_names is None:
            metric_names = [
                'clustering_coefficient',
                'gini_coefficient',
                'num_nodes',
                'spectral_gap',
                'triangle_count',
            ]
        
        denormalized = normalized_metrics.clone()
        for i, name in enumerate(metric_names):
            if name in self.output_range:
                min_val, max_val = self.output_range[name]
                denormalized[:, i] = normalized_metrics[:, i] * (max_val - min_val) + min_val
        
        return denormalized


class SimpleGCNRegressor(nn.Module):
    """Lightweight GCN regressor for faster training."""
    
    def __init__(
        self,
        input_dim: int = 1,
        hidden_dim: int = 64,
        num_metrics: int = 5,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.gcn1 = GCNConv(input_dim, hidden_dim)
        self.gcn2 = GCNConv(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # MLP head
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_metrics),
        )
    
    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = self.gcn1(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        x = self.gcn2(x, edge_index)
        x = self.relu(x)
        x = self.dropout(x)
        
        # Graph pooling
        if batch is None:
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        graph_embedding = global_mean_pool(x, batch=batch)
        
        # MLP head
        predictions = self.mlp(graph_embedding)
        
        return predictions

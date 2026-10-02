from __future__ import annotations


import numpy as np
import torch
from torch import Tensor


def minimum_angle_from_sector_angles(sector_angles: Tensor | None) -> float:
    """
    Extract minimum angle in degrees from sector_angles tensor.

    sector_angles shape: (num_nodes, max_degree + 1)
    Each row: [degree_count, angle1, angle2, ...]
    Returns the minimum angle found across all nodes (in degrees).
    """
    if sector_angles is None:
        return 0.0

    sector_tensor = sector_angles if torch.is_tensor(sector_angles) else torch.tensor(sector_angles)
    if sector_tensor.numel() == 0 or sector_tensor.dim() < 2 or sector_tensor.shape[1] <= 1:
        return 0.0

    # Extract angle columns (skip the first column which is degree count)
    angles = sector_tensor[:, 1:]

    # Set invalid angles (those associated with non-existent neighbors) to inf
    degree_counts = sector_tensor[:, 0].round().to(torch.int64).clamp(min=0)
    max_angles = angles.shape[1]

    # Create mask for valid angles
    indices = torch.arange(max_angles, device=angles.device)
    valid_mask = indices.unsqueeze(0) < degree_counts.unsqueeze(1)

    # Get positive angles only
    inf_tensor = torch.full_like(angles, float("inf"))
    masked_angles = torch.where(valid_mask & (angles > 1e-8), angles, inf_tensor)

    # Find minimum across all valid angles
    min_angle = masked_angles.min()
    if not torch.isfinite(min_angle):
        return 0.0

    # Convert radians to degrees
    min_angle_degrees = float(np.degrees(min_angle.item()))
    return min_angle_degrees


def maximum_degree_from_adjacency(adjacency: Tensor | None) -> float:
    """
    Extract maximum degree from adjacency matrix.

    Args:
        adjacency: Dense adjacency matrix of shape (num_nodes, num_nodes)

    Returns:
        Maximum degree as float.
    """
    if adjacency is None:
        return 0.0

    adj = adjacency if torch.is_tensor(adjacency) else torch.tensor(adjacency)
    if adj.numel() == 0:
        return 0.0

    # Remove self-loops and compute degree sum for each node
    adj_no_diag = adj.clone()
    adj_no_diag.fill_diagonal_(0.0)

    # Sum degrees (count nonzero entries in each row)
    degrees = (adj_no_diag > 0).sum(dim=1).to(torch.float32)

    if degrees.numel() == 0:
        return 0.0

    max_degree = float(degrees.max().item())
    return max_degree

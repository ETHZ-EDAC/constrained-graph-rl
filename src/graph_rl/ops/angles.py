"""Angle helpers shared across rules and environments."""

# Third-party
import jax.numpy as jnp

from graph_rl.utils import PARAMS


def angles_from_actions_normalized(raw: jnp.ndarray, sectors: jnp.ndarray, sector_offset_for_left_connector: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Map raw actions to angles that respect a budget and minimum angle.

    raw: tensor (..., n+1)
         last entry corresponds to closing angle
    Returns:
      explicit_angles (..., n)
      closing_angle   (..., 1)
    """
    eps = 1e-8
    *_, m = raw.shape
    n = m - 1
    min_angle = jnp.deg2rad(PARAMS.sector_eps)

    angles_there_already = sectors
    mask = angles_there_already >= 0.0
    angles_there_already = jnp.where(mask, angles_there_already, 0.0)
    offset = 1 + sector_offset_for_left_connector
    used_budget = angles_there_already.at[mask.sum()-offset].set(0).sum()

    budget = 2 * jnp.pi - used_budget

    # Budget after reserving min_angle for n+1 angles
    b_free = budget - (n + 1) * min_angle
    b_free = jnp.maximum(b_free, 0.0)

    # Smooth nonnegative extras
    extras = raw**2  # (..., n+1)

    # Normalize to share free budget
    extras_sum = extras.sum(axis=-1, keepdims=True)  # (..., 1)
    scaled = extras / (extras_sum + eps) * b_free  # (..., n+1)

    # Add min_angle
    all_angles = min_angle + scaled  # (..., n+1)

    explicit = all_angles[..., :n]
    closing = all_angles[..., n:]

    return explicit, closing

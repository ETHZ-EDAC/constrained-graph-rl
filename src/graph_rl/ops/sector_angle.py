"""Module for the computation of sector angles in a graph"""

# Standard library
from typing import Tuple

# Third-party
import jax
from jax import numpy as jnp

# First-party
from graph_rl.utils import PARAMS
from graph_rl.utils.adjacency_utils import get_degree_vector


@jax.jit
def update_all_sector_angles_vmap(
    adjacency: jnp.ndarray,
    vertex_positions: jnp.ndarray,
    sector_angles: jnp.ndarray,
    neighbors: jnp.ndarray,
    ref_edges_idx: jnp.ndarray,
    indices: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    For each vertex idx in [0, num_vertices), if deg[idx] >= 4 compute its sector angles, append their sum,
    and write into sector_angles[idx]. neighbor_indices[idx] holds the indices of the neighboring vertices.
    The first element in sector_angles[idx,:] is always the number of neighbors.
    """
    degree_vec = get_degree_vector(adjacency)

    def process_vertex(idx):
        deg_ = degree_vec[idx]
        sectors, sorted_neighbors = get_sector_angles_fn(idx, ref_edges_idx[idx], adjacency, vertex_positions)
        angles = jnp.concatenate([deg_[None], sectors])
        indices_local = jnp.concatenate([deg_[None], sorted_neighbors])
        row = jnp.stack([indices_local, angles], axis=0)
        return row

    sectors_batched = jax.vmap(process_vertex, in_axes=(0,))(indices)
    new_sector_angles = sector_angles.at[indices].set(sectors_batched[:, 1])
    neighbor_indices = neighbors.at[indices].set(sectors_batched[:, 0].astype(jnp.int32))

    return new_sector_angles, neighbor_indices


def get_sector_angles_fn(
    idx: jnp.ndarray, direct_neighbor: jnp.ndarray, adjacency: jnp.ndarray, positions: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Scalar function to compute sector angles for a given vertex in a graph. Is batched using the vmap in
    `update_all_sector_angles_vmap`. IN RADIANS.

    IMPORTANT CONVENTION: The order always start with the first incoming edge CCW for the returned indices and with the
    sector angle between the first incoming edge CCW and the next edge.

    Parameters
    ----------
    idx : jnp.ndarray: Scalar, Vertex index.
    direct_neighbor: jnp.ndarray (2,): Reference vertex index for the first incoming edge (convention)
                                        and if it spreads more 180.
    adjacency : jnp.ndarray (N, N) adjacency matrix.
    positions : jnp.ndarray (N, 3) vertex coordinates.

    Returns
    -------
    diffs : jnp.ndarray (M,) CCW sector angles, zeros where no edge. In RADIANS!
    sorted_neighbor_indices : jnp.ndarray (M,) sorted neighbor indices, -1 where no edge.
    """
    # symmetric adjacency for geometry
    adj_sym = jnp.logical_or(adjacency > 0, adjacency.T > 0)
    mask = adj_sym[idx]

    # absolute angles to all points
    rel = positions - positions[idx]  # (M, D)

    rel = jnp.where(mask[:, None], rel, 1.0)  # safe number for gradients
    rel /= jnp.linalg.norm(rel, axis=1, keepdims=True)  # normalize to unit vectors
    angles = jnp.arctan2(rel[:, 1], rel[:, 0]) % (2 * jnp.pi)  # (M,)

    # reference: smallest CCW incoming edge angle
    in_mask_ref = jnp.zeros(adjacency.shape[0]).at[direct_neighbor[0]].set(1)

    # if we spread more than 180 degrees, we need to invert the reference vector
    edge_ = jnp.where(in_mask_ref[:, None], rel, 0.0).mean(axis=0)
    edge_ /= jnp.linalg.norm(edge_)  # normalize to unit vector

    ref_edge = jnp.where(
        direct_neighbor[1], -1 * edge_, edge_
    )  # if we spread more than 180 degrees, we need to invert the reference vector)

    ref_angle_incoming_selection = jnp.arctan2(ref_edge[1], ref_edge[0]) % (2 * jnp.pi)

    angles_offset_to_ref = (angles - ref_angle_incoming_selection) % (2 * jnp.pi)

    angles_in_offset = jnp.where(in_mask_ref, angles_offset_to_ref, -1.0)
    ref_idx = jnp.argmax(angles_in_offset)
    ref_angle = angles[ref_idx]

    # relative CCW angles
    rel_angles = (angles - ref_angle) % (2 * jnp.pi)
    masked_angles = jnp.where(mask, rel_angles, 2 * jnp.pi)

    # Sort valid neighbor indices by angle
    sorted_all_indices = jnp.argsort(masked_angles)  # (M,)

    # sort and compute ring diffs
    sorted_angles = jnp.sort(masked_angles)  # (M,)
    shifted = jnp.concatenate([sorted_angles[1:], sorted_angles[:1]], axis=0)
    ring = shifted - sorted_angles  # (M,)
    valid = sorted_angles < 2 * jnp.pi
    diffs = jnp.where(valid, ring, -1.0)
    sorted_neighbor_indices = jnp.where(valid, sorted_all_indices, -1).astype(jnp.int32)

    return diffs[: PARAMS.max_vertex_degree], sorted_neighbor_indices[: PARAMS.max_vertex_degree]


def get_sector_violation(sectors: jnp.ndarray) -> jnp.ndarray:
    """
    Compute sector angle violation for all vertices in the graph.

    Parameters
    ----------
    sectors : jnp.ndarray (max_vertex_degree,) sector angles for all vertices.

    Returns
    -------
    violations : jnp.ndarray (,) sector angle violations for all vertices.
    """
    sector_eps = jnp.deg2rad(PARAMS.sector_eps)
    sec_left_mask = sectors > 0.0
    sector_violation_minimum = jnp.where(sec_left_mask, sector_eps - sectors, 0.0)
    # sector_violation_maximum = jnp.where(sec_left_mask, sectors - (jnp.pi - sector_eps), 0.0)
    violations = jnp.sum(jnp.maximum(sector_violation_minimum, 0.0))
    # violations = jnp.sum(jnp.maximum(sector_violation_minimum, 0.0)) + jnp.sum(
    #     jnp.maximum(sector_violation_maximum, 0.0)
    # )
    return violations

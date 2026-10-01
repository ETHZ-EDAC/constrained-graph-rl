"""Module for Rule 1: Stochastic vertex expansion in graph."""

# Standard library
from typing import Dict, Tuple

# Third-party
import jax
from jax import numpy as jnp
from logging_mod.logger import get_logger

from graph_rl.ops.angles import angles_from_actions_normalized

# First-party
from graph_rl.ops.expansion import deterministic_three_vertex_expansion, prepare_expansion
from graph_rl.ops.sector_angle import get_sector_angles_fn, get_sector_violation
from graph_rl.grammar.constants import Constants
from graph_rl.utils import PARAMS
from graph_rl.utils.adjacency_utils import get_edge_vectors, get_symmetric_adjacency, num_vertices

logger = get_logger()




@jax.jit
def rule_1(
    constants: Constants, params: Dict[str, jnp.ndarray], idx: jnp.ndarray
) -> Tuple[Constants, Dict[str, jnp.ndarray], None]:
    """Params in range [0,1], both angles and lens."""

    cb_idx, reference_neighbor = prepare_expansion(constants, idx)

    adj_b = constants.adj_b.at[idx].set(0)
    adj_b = adj_b.at[:, idx].set(0)

    cb_pos = constants.vp[cb_idx]
    sectors, indices = get_sector_angles_fn(idx, reference_neighbor, constants.adj, constants.vp)
    mask_inc = indices >= 0
    num_inc = mask_inc.sum()

    angles_raw = params["angles"]
    lens_raw = params["lens"]
    min_len = PARAMS.edge_length_min
    max_len = PARAMS.edge_length_max
    lens = min_len + lens_raw * (max_len - min_len)

    # Convert raw angles to actual angles
    angles, _ = angles_from_actions_normalized(angles_raw, sectors, sector_offset_for_left_connector=0)

    sectors = sectors.at[num_inc - 1].set(-1)
    ref_vector = constants.vp[idx] - constants.vp[reference_neighbor[0]]

    sectors = jax.lax.dynamic_update_slice(sectors, angles, (num_inc - 1,))  # Set the incoming angles
    sectors_padded_zero = jnp.where(sectors >= 0, sectors, 0.0)

    sum_rest = jnp.sum(sectors_padded_zero)
    sectors = sectors.at[num_inc + 2].set(2 * jnp.pi - sum_rest)

    edges_internal, _ = get_edge_vectors(constants.adj + adj_b, constants.vp)

    loss, vert_1, vert_2, vert_3 = deterministic_three_vertex_expansion(
        num_inc, sectors, lens, ref_vector, constants.vp[idx], edges_internal, cb_pos
    )

    # first three are geometric, the rest are topological (unchangeable), even boundary edge are geometric!)
    loss["edge_intersection_geom"] = loss["edge_intersection"].sum()
    loss["edge_intersection_topo"] = jnp.array(0.0)

    # Penalize boundary edges that are too long/short.
    def _len_penalty(p, q):
        dist = jnp.linalg.norm(p - q)
        return jnp.maximum(0, min_len - dist) + jnp.maximum(0, dist - max_len)

    boundary_penalty = (
        _len_penalty(vert_1, vert_2)
        + _len_penalty(vert_2, vert_3)
        + _len_penalty(constants.vp[cb_idx[1]], vert_1)
        + _len_penalty(constants.vp[cb_idx[0]], vert_3)
    )
    loss["boundary_edge_length_geom"] = boundary_penalty

    num_verts = num_vertices(constants.adj)
    out_indices = jnp.array([num_verts, num_verts + 1, num_verts + 2])

    adj_b = adj_b.at[num_verts, num_verts + 1].set(1)
    adj_b = adj_b.at[num_verts + 1, num_verts + 2].set(1)
    adj_b = adj_b.at[cb_idx[1], num_verts].set(1)
    adj_b = adj_b.at[cb_idx[0], num_verts + 2].set(1)
    adj_b = get_symmetric_adjacency(adj_b)

    adj_new = constants.adj.at[idx, out_indices].set(1)

    vp_flat_new = constants.vp.at[out_indices].set(jnp.stack([vert_1, vert_2, vert_3], axis=0))
    sectors_new, neighbors = get_sector_angles_fn(idx, reference_neighbor, adj_new, vp_flat_new)
    sector_violation_new = get_sector_violation(sectors_new)
    loss["sector_violation_geom"] = sector_violation_new

    deg = (neighbors >= 0).sum().astype(jnp.int32)

    constants_new = constants.replace(
        adj=adj_new,
        adj_b=adj_b,
        vp=vp_flat_new,
        reference_neighbor=constants.reference_neighbor.at[idx].set(reference_neighbor),
        sector_angles=constants.sector_angles.at[idx].set(jnp.concatenate([deg[None], sectors_new])),
        neighbors=constants.neighbors.at[idx].set(jnp.concatenate([deg[None], neighbors])),
    )

    loss_full = jnp.array(0.0)
    for loss_element in loss.values():
        loss_full += loss_element.sum()

    loss["full"] = loss_full

    return constants_new, loss, None

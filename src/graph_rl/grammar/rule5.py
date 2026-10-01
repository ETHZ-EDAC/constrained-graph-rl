"""Rule 5: Two-vertex boundary sprout (lighter version of rule 1)."""

# Standard library
from typing import Dict, Tuple

# Third-party
import jax
import jax.numpy as jnp

from graph_rl.ops.angles import angles_from_actions_normalized

# First-party
from graph_rl.ops.expansion import prepare_expansion, deterministic_two_vertex_expansion
from graph_rl.ops.sector_angle import get_sector_angles_fn, get_sector_violation
from graph_rl.grammar.constants import Constants
from graph_rl.utils import PARAMS
from graph_rl.utils.adjacency_utils import get_edge_vectors, get_symmetric_adjacency, num_vertices




@jax.jit
def rule_5(
    constants: Constants, params: Dict[str, jnp.ndarray], idx: jnp.ndarray
) -> Tuple[Constants, Dict[str, jnp.ndarray], None]:
    """
    Two-vertex expansion: add two new vertices connected to `idx`, then attach each to a boundary neighbor.
    """

    cb_idx, reference_neighbor = prepare_expansion(constants, idx)

    adj_b = constants.adj_b.at[idx].set(0)
    adj_b = adj_b.at[:, idx].set(0)

    ref_vector = constants.vp[idx] - constants.vp[reference_neighbor[0]]

    sectors, indices = get_sector_angles_fn(idx, reference_neighbor, constants.adj, constants.vp)
    mask_inc = indices >= 0
    num_inc = mask_inc.sum()

    angles_raw = params["angles"]
    angles, _ = angles_from_actions_normalized(angles_raw, sectors, sector_offset_for_left_connector=0)

    lens_raw = params["lens"]
    min_len = PARAMS.edge_length_min
    max_len = PARAMS.edge_length_max
    lens = min_len + lens_raw * (max_len - min_len)

    sectors = sectors.at[num_inc - 1].set(-1)
    sectors = jax.lax.dynamic_update_slice(sectors, angles, (num_inc - 1,))
    sectors_padded_zero = jnp.where(sectors >= 0, sectors, 0.0)
    sum_rest = jnp.sum(sectors_padded_zero)
    sectors = sectors.at[num_inc + 1].set(2 * jnp.pi - sum_rest)

    edges_internal, _ = get_edge_vectors(constants.adj + adj_b, constants.vp)
    loss, new_pos1, new_pos2 = deterministic_two_vertex_expansion(
        num_inc, sectors, lens, ref_vector, constants.vp[idx], edges_internal, constants.vp[cb_idx]
    )

    loss["edge_intersection_geom"] = loss["edge_intersection"].sum()
    loss["edge_intersection_topo"] = jnp.array(0.0)

    # Penalize boundary edges created to the new vertices.
    def _len_penalty(p, q):
        dist = jnp.linalg.norm(p - q)
        return jnp.maximum(0, min_len - dist) + jnp.maximum(0, dist - max_len)

    new_idx1 = num_vertices(constants.adj)
    new_idx2 = new_idx1 + 1

    adj_new = constants.adj.at[idx, jnp.array([new_idx1, new_idx2])].set(1)

    vp_new = constants.vp.at[new_idx1].set(new_pos1)
    vp_new = vp_new.at[new_idx2].set(new_pos2)

    adj_b = adj_b.at[new_idx1, new_idx2].set(1)
    adj_b = adj_b.at[new_idx1, cb_idx[1]].set(1)
    adj_b = adj_b.at[new_idx2, cb_idx[0]].set(1)
    adj_b = get_symmetric_adjacency(adj_b)

    boundary_penalty = (
        _len_penalty(new_pos1, new_pos2)
        + _len_penalty(new_pos1, constants.vp[cb_idx[1]])
        + _len_penalty(new_pos2, constants.vp[cb_idx[0]])
    )
    loss["boundary_edge_length_geom"] = boundary_penalty

    sectors_new, neighbors_idx = get_sector_angles_fn(idx, reference_neighbor, adj_new, vp_new)
    sector_violation_new = get_sector_violation(sectors_new)

    loss["sector_violation_geom"] = sector_violation_new

    deg_idx_int = (neighbors_idx >= 0).sum().astype(jnp.int32)

    constants_new = constants.replace(
        adj=adj_new,
        adj_b=adj_b,
        vp=vp_new,
        reference_neighbor=constants.reference_neighbor.at[idx].set(reference_neighbor),
        sector_angles=constants.sector_angles.at[idx].set(jnp.concatenate([deg_idx_int[None], sectors_new])),
        neighbors=constants.neighbors.at[idx].set(jnp.concatenate([deg_idx_int[None], neighbors_idx])),
    )

    loss_full = jnp.array(0.0)
    for loss_element in loss.values():
        loss_full += loss_element.sum()
    loss["full"] = loss_full

    return constants_new, loss, None

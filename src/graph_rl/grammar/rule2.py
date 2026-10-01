"""Rule 2"""

# Standard library
from typing import Dict, Tuple

# Third-party
import jax
import jax.numpy as jnp

from graph_rl.ops.angles import angles_from_actions_normalized

# First-party
from graph_rl.ops.expansion import deterministic_three_vertex_expansion, prepare_expansion
from graph_rl.ops.marching import get_reference_vector
from graph_rl.ops.sector_angle import get_sector_angles_fn, get_sector_violation
from graph_rl.grammar.constants import Constants
from graph_rl.utils import PARAMS
from graph_rl.utils.adjacency_utils import get_edge_vectors, get_symmetric_adjacency, num_vertices


@jax.jit
def rule_2(
    constants: Constants, params: Dict[str, jnp.ndarray], idx: jnp.ndarray
) -> Tuple[Constants, Dict[str, jnp.ndarray], jnp.ndarray]:

    cb_idx, reference_neighbor = prepare_expansion(constants, idx)
    edge_len1 = jnp.linalg.norm(constants.vp[cb_idx[0]] - constants.vp[idx])
    edge_len2 = jnp.linalg.norm(constants.vp[cb_idx[1]] - constants.vp[idx])

    loss_feasibility = {
        "cond1_topo": jnp.maximum(0, constants.adj[:, cb_idx[0]].sum(dtype=int) - (PARAMS.max_vertex_degree - 4)),
        "cond2_topo": jnp.maximum(0, constants.adj[:, cb_idx[1]].sum(dtype=int) - (PARAMS.max_vertex_degree - 4)),
        "cond3_topo": jnp.maximum(0, PARAMS.edge_length_min - edge_len1)
        + jnp.maximum(0, edge_len1 - PARAMS.edge_length_max),
        "cond4_topo": jnp.maximum(0, edge_len2 - PARAMS.edge_length_max)
        + jnp.maximum(0, PARAMS.edge_length_min - edge_len2),
    }

    adj_b = constants.adj_b.at[idx].set(0)
    adj_b = adj_b.at[:, idx].set(0)

    adj = constants.adj.at[idx, cb_idx[0]].set(1)
    adj = adj.at[idx, cb_idx[1]].set(1)

    cb_pos = constants.vp[cb_idx]
    sectors, indices = get_sector_angles_fn(idx, reference_neighbor, adj, constants.vp)

    ref_vector = constants.vp[idx] - constants.vp[reference_neighbor[0]]

    angles_raw = params["angles"]
    angles, _ = angles_from_actions_normalized(angles_raw, sectors, sector_offset_for_left_connector=1)

    b = angles[0]
    lens_raw = params["lens"][0]
    min_len = PARAMS.edge_length_min
    max_len = PARAMS.edge_length_max
    lens_2 = min_len + lens_raw * (max_len - min_len)

    num_inc = (adj[:, idx] > 0).sum()

    sectors = sectors.at[num_inc].set(-1)
    c = 2 * jnp.pi - jnp.where(sectors >= 0, sectors, 0).sum() - b  # Last angle to close the loop

    d = sectors[num_inc + 1]
    sectors = sectors.at[jnp.array([num_inc, num_inc + 1, num_inc + 2])].set(jnp.array([b, c, d]))

    lens_new = jnp.concatenate([edge_len2[None], lens_2[None], edge_len1[None]], axis=0).T

    edges_internal, _ = get_edge_vectors(adj + adj_b, constants.vp)

    loss, _, vert_2, _ = deterministic_three_vertex_expansion(
        num_inc, sectors, lens_new, ref_vector, constants.vp[idx], edges_internal, cb_pos
    )
    # only 2nd is geoemtric, and boundary
    loss["edge_intersection_geom"] = loss["edge_intersection"][1] + loss["edge_intersection"][3:].sum()
    loss["edge_intersection_topo"] = loss["edge_intersection"][0] + loss["edge_intersection"][2]

    loss.update(loss_feasibility)

    # Penalize boundary edges created to the new vertex.
    def _len_penalty(p, q):
        dist = jnp.linalg.norm(p - q)
        return jnp.maximum(0, min_len - dist) + jnp.maximum(0, dist - max_len)

    new_vert_idx = num_vertices(adj)

    adj_b = adj_b.at[new_vert_idx, cb_idx[0]].set(1)
    adj_b = adj_b.at[new_vert_idx, cb_idx[1]].set(1)
    adj_b = get_symmetric_adjacency(adj_b)

    boundary_penalty = _len_penalty(constants.vp[cb_idx[0]], vert_2) + _len_penalty(constants.vp[cb_idx[1]], vert_2)
    loss["boundary_edge_length_geom"] = boundary_penalty

    adj_new = adj.at[idx, new_vert_idx].set(1)
    vp_flat_new = constants.vp.at[new_vert_idx].set(vert_2)
    sectors_new, neighbors = get_sector_angles_fn(idx, reference_neighbor, adj_new, vp_flat_new)

    ref_neighbor_left = get_reference_vector(cb_idx[0], adj_new, vp_flat_new)  # side effect for compilation
    sec_left, neighbors_left = get_sector_angles_fn(cb_idx[0], ref_neighbor_left, adj_new, vp_flat_new)

    ref_neighbor_right = get_reference_vector(cb_idx[1], adj_new, vp_flat_new)
    sec_right, neighbors_right = get_sector_angles_fn(cb_idx[1], ref_neighbor_right, adj_new, vp_flat_new)

    sector_violation_left = get_sector_violation(sec_left)
    sector_violation_right = get_sector_violation(sec_right)
    sector_violation_new = get_sector_violation(sectors_new)

    loss["sector_violation_topo"] = sector_violation_left + sector_violation_right
    loss["sector_violation_geom"] = sector_violation_new

    deg = (neighbors >= 0).sum().astype(jnp.int32)
    deg_left = (neighbors_left >= 0).sum().astype(jnp.int32)
    deg_right = (neighbors_right >= 0).sum().astype(jnp.int32)
    constants_new = constants.replace(
        adj=adj_new,
        adj_b=adj_b,
        vp=vp_flat_new,
        reference_neighbor=constants.reference_neighbor.at[jnp.concat((idx[None], cb_idx))].set(
            jnp.stack((reference_neighbor, ref_neighbor_left, ref_neighbor_right))
        ),
        sector_angles=constants.sector_angles.at[jnp.concatenate((idx[None], cb_idx))].set(
            jnp.stack(
                (
                    jnp.concatenate([deg[None], sectors_new]),
                    jnp.concatenate([deg_left[None], sec_left]),
                    jnp.concatenate([deg_right[None], sec_right]),
                ),
                axis=0,
            )
        ),
        neighbors=constants.neighbors.at[jnp.concatenate((idx[None], cb_idx))].set(
            jnp.stack(
                (
                    jnp.concatenate([deg[None], neighbors]),
                    jnp.concatenate([deg_left[None], neighbors_left]),
                    jnp.concatenate([deg_right[None], neighbors_right]),
                ),
                axis=0,
            )
        ),
    )

    loss_full = jnp.array(0.0)
    for loss_element in loss.values():
        loss_full += loss_element.mean()

    loss["full"] = loss_full

    return constants_new, loss, cb_idx

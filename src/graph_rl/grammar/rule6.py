"""Rule 6: Connect left edge to existing vertex and expand a single new edge."""

# Standard library
from typing import Dict, Tuple

# Third-party
import jax
import jax.numpy as jnp

from graph_rl.ops.angles import angles_from_actions_normalized

# First-party
from graph_rl.ops.expansion import prepare_expansion, deterministic_two_vertex_expansion
from graph_rl.ops.marching import get_reference_vector
from graph_rl.ops.sector_angle import get_sector_angles_fn, get_sector_violation
from graph_rl.grammar.constants import Constants
from graph_rl.utils import PARAMS
from graph_rl.utils.adjacency_utils import get_edge_vectors, get_symmetric_adjacency, num_vertices


@jax.jit
def rule_6(
    constants: Constants, params: Dict[str, jnp.ndarray], idx: jnp.ndarray
) -> Tuple[Constants, Dict[str, jnp.ndarray], jnp.ndarray]:
    """Connects the left edge to an existing vertex and expands a single new edge."""

    cb_idx, reference_neighbor = prepare_expansion(constants, idx)
    edge_len1 = jnp.linalg.norm(constants.vp[cb_idx[0]] - constants.vp[idx])

    loss: Dict[str, jnp.ndarray] = {
        "cond1_topo": jnp.maximum(0, constants.adj[:, cb_idx[0]].sum(dtype=int) - (PARAMS.max_vertex_degree - 4)),
        "cond3_topo": jnp.maximum(0, PARAMS.edge_length_min - edge_len1)
        + jnp.maximum(0, edge_len1 - PARAMS.edge_length_max),
    }

    adj_b = constants.adj_b.at[idx].set(0)
    adj_b = adj_b.at[:, idx].set(0)

    adj = constants.adj.at[idx, cb_idx[0]].set(1)

    cb_pos = constants.vp[cb_idx]
    sectors, indices = get_sector_angles_fn(idx, reference_neighbor, adj, constants.vp)

    ref_vector = constants.vp[idx] - constants.vp[reference_neighbor[0]]

    num_inc = (adj[:, idx] > 0).sum()
    d = sectors[num_inc]

    angles_raw = params["angles"]
    a, _ = angles_from_actions_normalized(angles_raw, sectors, sector_offset_for_left_connector=1)

    lens_raw = params["lens"]
    min_len = PARAMS.edge_length_min
    max_len = PARAMS.edge_length_max
    lens = min_len + lens_raw * (max_len - min_len)
    lens_both = jnp.concatenate((lens, edge_len1[None]))

    sectors = sectors.at[num_inc - 1].set(-1)
    c = 2 * jnp.pi - jnp.where(sectors >= 0, sectors, 0).sum() - a.sum()  # Last angle to close the loop

    sectors = sectors.at[jnp.array([num_inc - 1, num_inc, num_inc + 1])].set(jnp.array([a[0], c, d]))

    edges_internal, _ = get_edge_vectors(adj + adj_b, constants.vp)

    loss_geom, vert_1, _ = deterministic_two_vertex_expansion(
        num_inc, sectors, lens_both, ref_vector, constants.vp[idx], edges_internal, cb_pos
    )

    loss.update(loss_geom)

    # Penalize boundary edges created to the new vertex.
    def _len_penalty(p, q):
        dist = jnp.linalg.norm(p - q)
        return jnp.maximum(0, min_len - dist) + jnp.maximum(0, dist - max_len)

    loss["edge_intersection_geom"] = loss["edge_intersection"][0].sum() + loss["edge_intersection"][2:].sum()
    loss["edge_intersection_topo"] = loss["edge_intersection"][1]

    new_vert_idx = num_vertices(adj)

    adj_b = adj_b.at[new_vert_idx, cb_idx[1]].set(1)
    adj_b = adj_b.at[new_vert_idx, cb_idx[0]].set(1)
    adj_b = get_symmetric_adjacency(adj_b)

    boundary_penalty = _len_penalty(vert_1, constants.vp[cb_idx[1]]) + _len_penalty(vert_1, constants.vp[cb_idx[0]])
    loss["boundary_edge_length_geom"] = boundary_penalty

    adj_new = adj.at[idx, new_vert_idx].set(1)

    vp_flat_new = constants.vp.at[new_vert_idx].set(vert_1)
    sectors_new, neighbors = get_sector_angles_fn(idx, reference_neighbor, adj_new, vp_flat_new)

    ref_neighbor_left = get_reference_vector(cb_idx[0], adj_new, vp_flat_new)
    sec_left, neighbors_left = get_sector_angles_fn(cb_idx[0], ref_neighbor_left, adj_new, vp_flat_new)

    sector_violation_new = get_sector_violation(sectors_new)
    sector_violation_left = get_sector_violation(sec_left)
    loss["sector_violation_geom"] = sector_violation_new
    loss["sector_violation_topo"] = sector_violation_left

    deg = (neighbors >= 0).sum().astype(jnp.int32)
    deg_left = (neighbors_left >= 0).sum().astype(jnp.int32)
    constants_new = constants.replace(
        adj=adj_new,
        adj_b=adj_b,
        vp=vp_flat_new,
        reference_neighbor=constants.reference_neighbor.at[jnp.concatenate((idx[None], cb_idx[0][None]))].set(
            jnp.stack((reference_neighbor, ref_neighbor_left))
        ),
        sector_angles=constants.sector_angles.at[jnp.concatenate((idx[None], cb_idx[0][None]))].set(
            jnp.stack(
                (
                    jnp.concatenate([deg[None], sectors_new]),
                    jnp.concatenate([deg_left[None], sec_left]),
                ),
                axis=0,
            )
        ),
        neighbors=constants.neighbors.at[jnp.concatenate((idx[None], cb_idx[0][None]))].set(
            jnp.stack(
                (
                    jnp.concatenate([deg[None], neighbors]),
                    jnp.concatenate([deg_left[None], neighbors_left]),
                ),
                axis=0,
            )
        ),
    )

    loss_full = jnp.array(0.0)
    for loss_element in loss.values():
        loss_full += loss_element.sum()

    loss["full"] = loss_full

    return constants_new, loss, cb_idx[0]

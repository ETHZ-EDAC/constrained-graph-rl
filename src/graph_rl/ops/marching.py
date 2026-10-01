"""Module for marching along a minimum  cycle basis in a graph."""

# Standard library
from typing import Tuple

# Third-party
import jax
import jax.numpy as jnp
from jax import lax

# First-party
from graph_rl.utils import PARAMS
from graph_rl.utils.adjacency_utils import get_symmetric_adjacency
from graph_rl.utils.math_utils import is_point_in_polygon


def marching_primitive_from_boundary_vertex_CW(
    start: jnp.ndarray, adj: jnp.ndarray, adj_sym: jnp.ndarray, pos: jnp.ndarray, neighbors: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.int32, jnp.ndarray]:
    """
    Marching primitive for boundary edge detection. Starting from a BOUNDARY, NO OUTGOING EDGES, vertex, it finds the
    next vertex in the graph that forms the boundary edge. It marches in CLOCKWISE (CW) direction along the edges of the
    graph. (NOTE: This results in a CCW marching around the boundary of a graph, but the primitive is doing it clockwise
    inside the subgraph.)

    NOTE: IT IS APPLIED IN CW DIRECTION ONLY!

    Parameters
    ----------
    start : jnp.ndarray: Scalar index of the boundary vertex.
    adj : jnp.ndarray: Adjacency matrix.
    adj_sym : jnp.ndarray: Symmetric adjacency matrix.
    pos : jnp.ndarray: Vertex positions.
    neighbors: jnp.ndarray: Ordered neighbors CCW.

    Returns
    -------
    See marching_CW_primitive.
    """
    deg = adj_sym[start].sum(dtype=jnp.uint8)

    def find_next_higher_deg(_):
        return neighbors[start, 1].astype(jnp.int32)

    next_ = jax.lax.cond(
        deg > 1, find_next_higher_deg, lambda _: jnp.argmax(adj_sym[start]).astype(jnp.int32), operand=None
    )

    return _marching_primitive(start, next_, adj, adj_sym, pos, CCW=False)


def marching_primitive_from_boundary_vertex_CCW(
    start: jnp.ndarray, adj: jnp.ndarray, adj_sym: jnp.ndarray, pos: jnp.ndarray, neighbors: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.int32, jnp.ndarray]:
    """
    Same as marching_primitive_from_boundary_vertex, but marches in CCW direction.
    """
    deg = adj_sym[start].sum(dtype=jnp.uint8)

    def find_next_higher_deg(_):
        return neighbors[start, deg].astype(jnp.int32)

    next_ = jax.lax.cond(
        deg > 1, find_next_higher_deg, lambda _: jnp.argmax(adj_sym[start]).astype(jnp.int32), operand=None
    )

    return _marching_primitive(start, next_, adj, adj_sym, pos, CCW=True)


def get_reference_vector(
    idx: jnp.ndarray,
    adj: jnp.ndarray,
    pos: jnp.ndarray,
) -> jnp.ndarray:
    """

    Returns
    -------
    reference : jnp.ndarray (1,)
        Boolean array where only entries corresponding to `True` in `edge_end_mask` may be `True`.
        A `True` entry marks the unique reference edge (the only CW march that does NOT return a cycle).
    is_in_polygon : jnp.ndarray
        Same as in `get_reference_vector`.
    """
    adj_sym = get_symmetric_adjacency(adj)

    mask = adj[:, idx] > 0
    candidates = jnp.where(mask, size=PARAMS.max_vertex_degree, fill_value=-1)[0].astype(jnp.int32)

    def body(end_idx):
        _, _, boundary_connection = _marching_primitive(idx, end_idx, adj, adj_sym, pos, False)
        # reference edge is the one for which CW marching does *not* return a cycle
        return jnp.logical_not(boundary_connection[0] == boundary_connection[1])

    # Vectorize across all possible end indices with masking
    reference_raw = jax.vmap(body, in_axes=(0,), out_axes=0)(candidates)
    reference = jnp.where(candidates >= 0, reference_raw, False)

    # Keep the polygon test identical
    is_in_polygon = jnp.where(mask.sum() > 2, is_point_in_polygon(pos[idx], pos, mask), False)
    ref_vert = candidates[jnp.argmax(reference)]
    return jnp.array([ref_vert, is_in_polygon.astype(jnp.int32)])


def _marching_primitive(
    edge_start: jnp.ndarray,
    edge_end: jnp.ndarray,
    adj: jnp.ndarray,
    adj_sym: jnp.ndarray,
    pos: jnp.ndarray,
    CCW: bool = False,
) -> Tuple[jnp.ndarray, jnp.int32, jnp.ndarray]:
    """
    March along edges given a start edge (edge_start, edge_end), adjacency and vertex positions. Stops
    when we either reach the starting vertex again or hit a leaf (vertex with degree 1) or the loop exceeds number of
    vertices in graph. The return arguments make sense for:
    1. First arg is to detect faces in a graph.
    2. Second one is relevant for 1.
    3. Is used for boundary edges detection, the return value is an array of shape (2,)
    with the start and end vertex indices of the boundary edge.

    Parameters
    ----------
    edge_start : jnp.ndarray: Start index of the start edge.
    edge_end : jnp.ndarray: End index of the start edge.
    adj : jnp.ndarray: Adjacency matrix.
    adj_sym : jnp.ndarray: Symmetric adjacency matrix.
    pos : jnp.ndarray: Vertex positions.
    CCW: bool: If True, marches in counter-clockwise (CCW) direction, otherwise in clockwise (CW) direction.

    Returns
    -------
    full_states : jnp.ndarray: Array of shape (max_steps, 2), acting as a edge list of the marching path until it
        terminated. If successful, represent a minimum cycle basis.
    num_steps : jnp.int32: Number of steps taken in the marching process, i.e., number of edges in the path.
    boundary_connection : jnp.ndarray:  Array of shape (2,) with start and end vertex indices.
    """
    prev = edge_start
    current = edge_end

    N = adj_sym.shape[0]
    max_steps = N
    states = jnp.zeros((max_steps, 2), dtype=jnp.int32)
    step = jnp.array(0, dtype=jnp.int32)

    def cond_fun(carry):
        prev, current, step, _ = carry
        # continue while we haven't hit a leaf and haven't overflowed buffer or we end up at the start again
        cond_1 = current != edge_start
        cond_2 = (jnp.sum(adj[current]) != 0) | (current == 0)
        cond_3 = prev != current
        cond_4 = (jnp.sum(adj[:, current]) != 0) | (current == 0)
        cond_5 = step < max_steps
        return cond_1 & cond_2 & cond_3 & cond_4 & cond_5

    def body_fun(carry):
        prev, curr, st, states = carry

        # all absolute angles from curr → every vertex
        rel_all = pos - pos[curr]  # (N,2)
        angles_all = (jnp.arctan2(rel_all[:, 1], rel_all[:, 0])) % (2 * jnp.pi)

        # reference direction is prev → curr
        ref_vec = pos[prev] - pos[curr]
        ref_angle = (jnp.arctan2(ref_vec[1], ref_vec[0])) % (2 * jnp.pi)

        # relative CCW offsets in [0,2π)
        rel_angle_mask = jnp.where(CCW, -2 * jnp.pi, 2 * jnp.pi)  # -2π for CCW, 2π for CW
        order_idx = jnp.where(CCW, -1, 1)  # -1for CCW, 1 for CW

        rel_angles = (angles_all - ref_angle) % (2 * jnp.pi)
        mask = adj_sym[curr]
        rel_angles = jnp.where(mask, rel_angles, rel_angle_mask)

        # sort ascending; the very largest neighbor angle is just before ref in CCW order
        order = jnp.argsort(rel_angles).astype(jnp.int32)
        # pick the largest neighbor → CCW step
        next_idx = order[order_idx]

        # record this edge
        states = states.at[st].set(jnp.array([prev, curr], dtype=jnp.int32))
        return (curr, next_idx, st + 1, states)

    prev, current, num_steps, full_states = lax.while_loop(cond_fun, body_fun, (prev, current, step, states))
    # record the final step
    full_states = full_states.at[num_steps].set(jnp.array([prev, current], dtype=jnp.int32))

    boundary_connection = jnp.array([edge_start, current], dtype=jnp.int32)
    return full_states, num_steps + 1, boundary_connection

"""Module for line-intersection tests given two straight finite lines."""

# Standard library
from typing import Tuple

# Third-party
import jax.numpy as jnp
from jax import jit, vmap
from logging_mod.logger import get_logger

logger = get_logger()


_LEN_EPS = 1e-6


@jit
def edge_batch_intersection_loss(edge_vectors: jnp.ndarray) -> jnp.ndarray:
    """
    For each edge in `edge_vectors`, compute the intersection loss against all
    *other* edges in `edge_vectors` (exclude self). Assumes planarity.

    Args:
        edge_vectors : jnp.ndarray (E, 2, 3) Each edge defined by two 3D vertices.

    Returns:
        jnp.ndarray (E,) : Loss per edge against all other edges.
    """
    # 2D projections (planar test)
    p = edge_vectors[:, 1, :2]  # (E, 2)
    q = edge_vectors[:, 0, :2]  # (E, 2)
    edge_vecs_2d = edge_vectors[:, :, :2]  # (E, 2, 2)

    # Track which edges are valid (non-zero length) so padded entries never contribute
    lengths = jnp.linalg.norm(edge_vecs_2d[:, 0, :] - edge_vecs_2d[:, 1, :], axis=1)
    valid_edges = (lengths > _LEN_EPS).astype(edge_vectors.dtype)

    # Compute loss of each (p_i, q_i) vs the whole batch (includes self)
    batch_vs_batch = vmap(_edge_intersection_loss_batched, in_axes=(0, 0, None))
    loss_vs_all = batch_vs_batch(p, q, edge_vecs_2d)  # (E,)

    # Compute each edge's self-term by giving it a batch of size 1 containing itself
    self_batches = edge_vecs_2d[:, None, :, :]  # (E, 1, 2, 2)
    loss_vs_self = vmap(_edge_intersection_loss_batched, in_axes=(0, 0, 0))(p, q, self_batches)  # (E,)

    # Exclude self intersections and pad entries by masking
    return (loss_vs_all - loss_vs_self) * valid_edges


@jit
def edges_vs_edge_batch_intersection_loss(new_edges: jnp.ndarray, edge_vectors: jnp.ndarray) -> jnp.ndarray:
    """Executes `edge_intersection_loss_batched` for a batch of edges (new_edges) against
    all other edges in `edge_vectors`. It only works for planar graphs. (obviously). It also tests if new_edges show
    intersection against themselves, i.e. each edge in `new_edges` is tested against all other edges in `new_edges`.

    Parameters
    ----------
    new_edges : jnp.ndarray (E_new, 2, 3) Array of new edges, where each edge is defined by two vertices (3D).
    edge_vectors : jnp.ndarray (E, 2, 3) Array of edges in the graph, where each edge is defined by two vertices.

    Returns
    -------
    jnp.ndarray: Batched array of shape (N,) containing the loss for each vertex in `new_vertices` against all edges.
        The loss is scaled by a factor of 10.
    """
    # 2D projections
    p = new_edges[:, 1, :2]  # (E_new, 2)
    q = new_edges[:, 0, :2]  # (E_new, 2)
    edge_vecs_2d = edge_vectors[:, :, :2]  # (E, 2, 2)
    new_edges_2d = new_edges[:, :, :2]  # (E_new, 2, 2)

    # vmap wrapper once
    batch_vs_batch = vmap(_edge_intersection_loss_batched, in_axes=(0, 0, None))

    # (A) new_edges vs given edge_vectors -> (E_new,)
    loss_vs_batch = batch_vs_batch(p, q, edge_vecs_2d)

    # (B1) new_edges vs *all* new_edges (includes self) -> (E_new,)
    loss_vs_new_all = batch_vs_batch(p, q, new_edges_2d)

    # (B2) subtract per-edge self term: i vs batch containing ONLY edge i
    # shape: (E_new, 1, 2, 2) so each i gets its own 1-edge batch
    self_batches = new_edges_2d[:, None, :, :]
    loss_vs_self = vmap(_edge_intersection_loss_batched, in_axes=(0, 0, 0))(p, q, self_batches)

    # Leave-one-out self test for every i, then add existing batch loss
    return loss_vs_batch + (loss_vs_new_all - loss_vs_self)


def _edge_intersection_loss_batched(p1: jnp.ndarray, q1: jnp.ndarray, all_edges: jnp.ndarray) -> jnp.ndarray:
    """
    Checks if edge formed by `p1` and `q1` is intersecting any edge in `all_edges` and returns a loss > 0 if
    intersection is detected, otherwise returns 0.

    Parameters
    ----------
    p1 : jnp.ndarray: Start point of edge under investigation (shape: (2,)).
    q1 : jnp.ndarray: End point of edge under investigation (shape: (2,)).
    all_edges : jnp.ndarray: Batched array of all other edges in the graph (shape: (E, 2, 2)).

    Returns
    -------
    jnp.ndarray
        Boolean array of shape (1,) indicating if intersection detected or not.
    """

    if all_edges.shape[0] == 0:
        return jnp.array(0.0)

    # Batch of points for `p2` and `q2`
    p2_batch = all_edges[:, 0, :]
    q2_batch = all_edges[:, 1, :]
    lengths = jnp.linalg.norm(p2_batch - q2_batch, axis=1)
    valid_targets = (lengths > _LEN_EPS).astype(all_edges.dtype)

    # Ignore intersections when edges share any endpoint (incident edges).
    shared_vertex = (
        (jnp.linalg.norm(p1 - p2_batch, axis=1) < _LEN_EPS)
        | (jnp.linalg.norm(p1 - q2_batch, axis=1) < _LEN_EPS)
        | (jnp.linalg.norm(q1 - p2_batch, axis=1) < _LEN_EPS)
        | (jnp.linalg.norm(q1 - q2_batch, axis=1) < _LEN_EPS)
    )
    valid_targets = valid_targets * (1.0 - shared_vertex.astype(all_edges.dtype))

    intersection_fn = vmap(_edge_intersection_loss, in_axes=(None, None, 0, 0))
    intersections = intersection_fn(p1, q1, p2_batch, q2_batch)

    return jnp.sum(intersections * valid_targets)


def _edge_intersection_loss(p1: jnp.ndarray, p2: jnp.ndarray, q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Scalar loss of how much intersection we have between two edges. Like a hinge loss.

    Params
    ------
    p1 : jnp.ndarray Start point of first edge (shape: (2,)).
    p2 : jnp.ndarray End point of first edge (shape: (2,)).
    q1 : jnp.ndarray Start point of second edge (shape: (2,)).
    q2 : jnp.ndarray: End point of second edge (shape: (2,)).

    Returns
    -------
    jnp.ndarray. Scalar array (1,) indicating the amount of intersection. < 0 no intersection, > 0 intersection.
    """

    eps = 0.00  # Small epsilon to avoid intersection too close.
    # solve for (t,s) in p1+s*(p2−p1) = q1+t*(q2−q1)
    t, s = _edge_intersection_params(p1, p2, q1, q2)

    # how deep into [ε, 1−ε] is s?
    #  - below ε or above 1−ε → 0
    #  - inside → min(s−ε, (1−ε)−s)
    d_s = jnp.maximum(0.0, jnp.minimum(s - eps, (1.0 - eps) - s))

    # same for t
    d_t = jnp.maximum(0.0, jnp.minimum(t - eps, (1.0 - eps) - t))

    # product → zero if either parameter is out of bounds,
    # larger when both s and t lie well inside (peaks at s=t=0.5)
    return d_s * d_t * 10.0  # scale factor


def _edge_intersection_params(
    p1: jnp.ndarray, p2: jnp.ndarray, q1: jnp.ndarray, q2: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Finds the values of s and t for the intersection of two vectors, given their start and end points.

    Mathematically, x = v_1_source + t*(delta v_1) (static), r = v_2_source + s*(delta v_2)

    Parameters
    ----------
    p1 : jnp.ndarray (2,) Starting point of the first vector.
    p2: jnp.ndarray (2,) Ending point of the first vector.
    q1 : jnp.ndarray (2,) Starting point of the second vector.
    q2 : jnp.ndarray (2,) Ending point of the second vector.

    Returns
    -------
    t : jnp.ndarray (,) Value of t for the intersection point.
    s : jnp.ndarray (,) Value of s for the intersection point.
    """

    eps = 1e-8  # Small epsilon to avoid division by zero
    b = p2 - p1  # direction of first segment
    d = q2 - q1  # direction of second segment
    e = q1 - p1  # offset from first start to second start

    # 2D cross product b×d = b_x * d_y - b_y * d_x
    cross_bd = b[0] * d[1] - b[1] * d[0]

    # Compute a sign that is ±1 even if cross_bd == 0
    sign_nz = jnp.where(cross_bd >= 0.0, 1.0, -1.0)

    # Clamp denom to never be zero
    denom = jnp.where(jnp.abs(cross_bd) < eps, sign_nz * eps, cross_bd)

    cross_ed = e[0] * d[1] - e[1] * d[0]
    t = cross_ed / denom

    # s = cross(e, b) / cross(b,d)
    cross_eb = e[0] * b[1] - e[1] * b[0]
    s = cross_eb / denom

    return t, s

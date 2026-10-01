"""Module containing various utility functions for working with adjacency matrices and retrieving info out of it."""

# Standard library
from typing import Tuple

# Third-party
import jax
import jax.numpy as jnp
from jax import jit

# First-party
from graph_rl.utils import PARAMS




@jax.jit
def num_vertices(adj: jnp.ndarray) -> jnp.ndarray:
    """
    Counts the number of active vertices in a padded adjacency matrix.

    Parameters
    ----------
    adj : jnp.ndarray (N, N) adjacency matrix.

    Returns
    -------
    count : jnp.ndarray: Scalar number of active vertices.
    """
    # example 2D array
    # boolean mask of rows with at least one non-zero
    has_out = jnp.any(adj != 0, axis=1)  # -> [False, True, True, False]
    # count them
    # incoming mask: col j has any nonzero ⇒ j has an incoming edge
    has_in = jnp.any(adj != 0, axis=0)  # shape (N,)

    # combine: vertex is “active” if it has either
    has_any = has_out | has_in

    # sum of True’s gives the count
    return jnp.sum(has_any, dtype=jnp.int32)


@jax.jit
def get_symmetric_adjacency(adjacency: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the symmetric adjacency matrix (undirected).

    Parameters
    ----------
    adjacency : jnp.ndarray (N, N) adjacency matrix.

    Returns
    -------
    adj_sym : jnp.ndarray (N, N) symmetric adjacency matrix.
    """
    return jnp.logical_or(adjacency > 0, adjacency.T > 0).astype(jnp.uint8)


@jax.jit
def get_degree_vector(adjacency: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the degree vector for all vertices in the graph.

    Parameters
    ----------
    adjacency : jnp.ndarray (N, N) adjacency matrix.

    Returns
    -------
    degree_vec : jnp.ndarray (N,) degree of each vertex.
    """
    adj_sym = jnp.logical_or(adjacency > 0, adjacency.T > 0)
    return jnp.sum(adj_sym, axis=0)




@jax.jit
def get_edge_vectors(adj: jnp.ndarray, pos: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Get edge vector from start to end points of each edge in the graph.
    Returns a (E_max, 2, 2) array of edge‐endpoint pairs, zero‐padded, where E_max = N * max_degree // 2.

    Parameters
    ----------
    adj : jnp.ndarray, shape (N, N) Adjacency matrix (0/1 or weights).
    pos : jnp.ndarray, shape (N, 2) 2D positions of the N vertices.
    max_degree : int: Known upper bound on the degree of any vertex.

    Returns
    -------
    edge_vecs : jnp.ndarray, shape (E_max, 2, 2): E_max edges (zero-padded) each with a start and end position in 2D.
    """
    N = adj.shape[0]
    E_max = (N * 3) - 6  # planar graph bound

    # Symmetrize and take strict upper triangle
    adj_sym = get_symmetric_adjacency(adj)
    adj_sym_upper = jnp.triu(adj_sym, k=1)  # shape (N, N)
    num_edges = jnp.sum(adj_sym_upper)

    # Extract up to E_max edge‐pairs, padding with zeros if fewer exist
    i_idx, j_idx = jnp.nonzero(adj_sym_upper, size=E_max)  # each shape (E_max,)

    # Gather the corresponding positions
    p_i = pos[i_idx]  # (E_max, 2)
    p_j = pos[j_idx]  # (E_max, 2)

    # Stack into (E_max, 2, 2)
    edge_vecs = jnp.stack([p_i, p_j], axis=1)
    return edge_vecs, num_edges






@jit
def get_edge_lengths(vertex_positions: jnp.ndarray, edge_list: jnp.ndarray) -> jnp.ndarray:
    """Get the lengths of edges in the graph. Returns a 1D array of edge lengths corresponding to the edge_list
    indexing.

    Parameters
    ----------
    vertex_positions : jnp.ndarray (N, 3): Vertex positions in 3D space.
    edge_list : jnp.ndarray (E, 2): Edge list where each row contains the indices of the vertices that form the edge.

    Returns
    -------
    jnp.ndarray: Edge lengths for all edges in the graph, shape (E,)."""

    return jnp.linalg.norm(vertex_positions[edge_list[:, 1]] - vertex_positions[edge_list[:, 0]], axis=1)

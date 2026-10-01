"""Vertex expansion operations including admissibility checks."""

# Standard library
from typing import Dict, Tuple

# Third-party
import jax
from jax import numpy as jnp

# First-party
from graph_rl.ops.edge_intersections import edges_vs_edge_batch_intersection_loss
from graph_rl.ops.marching import (
    get_reference_vector,
    marching_primitive_from_boundary_vertex_CCW,
    marching_primitive_from_boundary_vertex_CW,
)
from graph_rl.grammar.constants import Constants
from graph_rl.utils import PARAMS
from graph_rl.utils.adjacency_utils import get_symmetric_adjacency
from graph_rl.utils.math_utils import get_rotz


def deterministic_three_vertex_expansion(
    num_inc: jnp.ndarray,
    sectors: jnp.ndarray,
    lens: jnp.ndarray,
    ref_vector: jnp.ndarray,
    vertex_origin: jnp.ndarray,
    edge_vectors: jnp.ndarray,
    cb_pos: jnp.ndarray,
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Expand a vertex with three new outgoing edges and vertices and check for admissibility. (VRF) + Edge intersection.
    """

    # First sample possible expansions
    # alpha_rad = sectors[num_inc - 1 : num_inc + 2]
    alpha_rad = jax.lax.dynamic_slice(sectors, (num_inc - 1,), (3,))
    ref_vector = -1 * ref_vector

    # This is the angle formed by incoming angles
    offset_angle = (
        jnp.where(sectors >= 0, sectors, 0).sum() - jax.lax.dynamic_slice(sectors, (num_inc - 1,), (4,)).sum()
    )
    alpha_rad = alpha_rad.at[0].set(offset_angle + alpha_rad[0])
    vert1, vert2, vert3 = transform_vertices_to_inertial_coords(alpha_rad, ref_vector, vertex_origin, lens)
    # stack all possible new vertices

    new_verts = jnp.vstack([vert1, vert2, vert3])

    new_edges = get_edges_from_expansion(vertex_origin, new_verts, cb_pos)
    intersections = edges_vs_edge_batch_intersection_loss(new_edges, edge_vectors)

    loss = {"edge_intersection": intersections}

    return loss, vert1, vert2, vert3


def get_edges_from_expansion(
    origin: jnp.ndarray,
    expansion_pts: jnp.ndarray,
    cb_pts: jnp.ndarray,
) -> jnp.ndarray:
    """
    Construct all new edges resulting from a vertex expansion. I.e. three edges from origin to new vertices,
    two edges between new vertices, and two edges from boundary control points to new vertices.
    """
    # edges from each new vertex to origin
    back_edges = jnp.stack([expansion_pts, jnp.broadcast_to(origin, expansion_pts.shape)], axis=1)  # (3,2,3)

    if PARAMS.boundary_edge_intersection:
        # edges between new vertices
        v01 = jnp.stack([expansion_pts[0], expansion_pts[1]], axis=0)
        v12 = jnp.stack([expansion_pts[1], expansion_pts[2]], axis=0)
        cross_edges = jnp.stack([v01, v12], axis=0)  # (2,2,3)

        # edges from boundary control points to new vertices
        bc0 = jnp.stack([cb_pts[0], expansion_pts[2]], axis=0)
        bc1 = jnp.stack([cb_pts[1], expansion_pts[0]], axis=0)
        bc_edges = jnp.stack([bc0, bc1], axis=0)  # (2,2,3)
        return jnp.concatenate([back_edges, cross_edges, bc_edges], axis=0)
    else:
        return back_edges


def transform_vertices_to_inertial_coords(
    alphas_rad: jnp.ndarray,
    ref_vector: jnp.ndarray,
    vertex_origin: jnp.ndarray,
    lengths: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Transform local coordinates to inertial frame for a single vertex (in 2D). Input is given in terms of angles of
    rotation from the incoming inertial vector"""
    vert_1 = get_rotz(alphas_rad[0]) @ ref_vector
    vert_2 = get_rotz(alphas_rad[1]) @ vert_1
    vert_3 = get_rotz(alphas_rad[2]) @ vert_2

    vert_1 = vert_1 / jnp.linalg.norm(vert_1) * lengths[0]
    vert_2 = vert_2 / jnp.linalg.norm(vert_2) * lengths[1]
    vert_3 = vert_3 / jnp.linalg.norm(vert_3) * lengths[2]

    vert_1 += vertex_origin
    vert_2 += vertex_origin
    vert_3 += vertex_origin
    return vert_1, vert_2, vert_3


def prepare_expansion(constants: Constants, idx: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Prepare the expansion by getting the boundary control points and the reference neighbor for the expansion.

    Returns cb_idx: jnp.ndarray (2,): Indices of the boundary control points to "connect" boundaries to. (left, right from top view)
            reference_neighbor: jnp.ndarray (2,): Reference neighbor index and if the incoming edges span more than 180 degrees.
    """

    def idx_0():
        return jnp.array([1, 1]).astype(jnp.int32), jnp.array([1, 0]).astype(jnp.int32)

    def else_():
        adj_sym = get_symmetric_adjacency(constants.adj)
        _, _, cpCW = marching_primitive_from_boundary_vertex_CW(
            idx, constants.adj, adj_sym, constants.vp, constants.neighbors
        )
        _, _, cpCCW = marching_primitive_from_boundary_vertex_CCW(
            idx, constants.adj, adj_sym, constants.vp, constants.neighbors
        )
        # idx 0 goes to left, idx 1 goes to right (if looked at from top intuitively graphically)
        cb_idx = jnp.array([cpCW[1], cpCCW[1]])

        ref_vert_and_spans_more_than_180 = get_reference_vector(
            idx, constants.adj, constants.vp
        )  # Get the reference vector for the expansion

        return cb_idx, ref_vert_and_spans_more_than_180

    cb_idx, reference_neighbor = jax.lax.cond(idx == 0, idx_0, else_)

    return cb_idx, reference_neighbor


def deterministic_two_vertex_expansion(
    num_inc: jnp.ndarray,
    sectors: jnp.ndarray,
    lens: jnp.ndarray,
    ref_vector: jnp.ndarray,
    vertex_origin: jnp.ndarray,
    edge_vectors: jnp.ndarray,
    cb_pos: jnp.ndarray,
) -> Tuple[Dict[str, jnp.ndarray], jnp.ndarray, jnp.ndarray]:
    """Two-vertex variant of deterministic expansion used in rule 1."""
    alpha_rad = jax.lax.dynamic_slice(sectors, (num_inc - 1,), (2,))
    ref_vector = -1 * ref_vector

    offset_angle = (
        jnp.where(sectors >= 0, sectors, 0).sum() - jax.lax.dynamic_slice(sectors, (num_inc - 1,), (3,)).sum()
    )
    alpha_rad = alpha_rad.at[0].set(offset_angle + alpha_rad[0])

    vert1 = get_rotz(alpha_rad[0]) @ ref_vector
    vert2 = get_rotz(alpha_rad[1]) @ vert1

    vert1 = vert1 / jnp.linalg.norm(vert1) * lens[0]
    vert2 = vert2 / jnp.linalg.norm(vert2) * lens[1]

    vert1 += vertex_origin
    vert2 += vertex_origin

    new_verts = jnp.vstack([vert1, vert2])
    back_edges = jnp.stack([new_verts, jnp.broadcast_to(vertex_origin, new_verts.shape)], axis=1)

    if PARAMS.boundary_edge_intersection:
        cross_edge = jnp.stack([new_verts[0], new_verts[1]], axis=0)[None, :, :]
        bc_edges = jnp.stack(
            (
                jnp.stack([cb_pos[0], new_verts[1]], axis=0),
                jnp.stack([cb_pos[1], new_verts[0]], axis=0),
            ),
            axis=0,
        )

        new_edges = jnp.concatenate([back_edges, cross_edge, bc_edges], axis=0)
    else:
        new_edges = back_edges
    intersections = edges_vs_edge_batch_intersection_loss(new_edges, edge_vectors)

    loss = {"edge_intersection": intersections}

    return loss, vert1, vert2

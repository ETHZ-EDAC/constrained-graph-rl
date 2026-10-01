# Third-party
from flax import struct
from jax import numpy as jnp


@struct.dataclass
class Constants(struct.PyTreeNode):  # pylint: disable=W0223
    """Stores all constant parameters."""

    root_vertices: jnp.ndarray  # Root vertex, shape (n_dof,)
    adj: jnp.ndarray  # Adjacency matrix, shape (N, N)
    adj_b: jnp.ndarray  # Boundary adjacency matrix, shape (N, N)
    sector_angles: jnp.ndarray  # Sector angles, shape (N, PARAMS.max_vertex_degree + 1)
    reference_neighbor: jnp.ndarray  # Reference neighbor, shape (N, 2), with (idx, spreads_more_than_180)
    neighbors: jnp.ndarray  # Neighbors of each vertex, shape (N, PARAMS.max_vertex_degree)
    vp: jnp.ndarray  # Flat vertex positions, shape (N, 3)
    c_vert_idx: jnp.ndarray = struct.field(
        default_factory=lambda: jnp.zeros((4,), dtype=jnp.int32)
    )  # Index of additional AABB corner vertices.

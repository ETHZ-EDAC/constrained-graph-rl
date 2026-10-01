"""Module for general mathematical utilities."""

# Third-party
import jax
import jax.numpy as jnp
from jax import Array, jit


@jit
def get_rotz(angle: jnp.ndarray) -> jnp.ndarray:
    """
    Returns the rotation matrix for a rotation about the z-axis.

    Parameters
    ----------
    angle : jnp.ndarray: Scalar rotation angle in radians.

    Returns
    -------
    rotz : jnp.ndarray (3, 3) rotation matrix.
    """
    ca = jnp.cos(angle)
    sa = jnp.sin(angle)
    return jnp.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])


@jax.jit
def is_point_in_polygon(
    query_point: jnp.ndarray, polygon_points: jnp.ndarray, mask: jnp.ndarray, eps: float = 1e-9
) -> Array:
    """
    JIT-safe point-in-polygon with a boolean mask.
    - No boolean indexing / compaction.
    - Vertices are assumed in cyclic order; edges connect consecutive True entries in `mask` (cyclic).
    - Works for concave polygons; boundary is included.
    """
    p = query_point[:2]
    pts = polygon_points[:, :2]
    N = pts.shape[0]
    px, py = p[0], p[1]

    # Degenerate early-out (still shape-static)
    M = jnp.count_nonzero(mask)

    def _degenerate(_):
        return jnp.array(False, dtype=bool)

    def _compute(_):
        # ----- build "next true index" for each i without compaction -----
        # For each i, find the smallest k in {1..N} so that mask[(i+k)%N] is True.
        # This is O(N^2) but shape-static and robust. For typical face sizes, it's fine.
        ii = jnp.arange(N)[:, None]  # (N,1)
        kk = jnp.arange(1, N + 1)[None, :]  # (1,N)
        cand = (ii + kk) % N  # (N,N) candidate next indices
        cand_is_true = mask[cand]  # (N,N)
        # position of first True along axis=1 (falls back to 0 if none, but M>=3 ensures existence)
        first_pos = jnp.argmax(cand_is_true, axis=1)  # (N,)
        next_idx = cand[jnp.arange(N), first_pos]  # (N,)

        # edges only start from masked vertices
        active = mask

        vi = pts  # (N,2)
        vj = pts[next_idx]  # (N,2)  gather is shape-static

        xi, yi = vi[:, 0], vi[:, 1]
        xj, yj = vj[:, 0], vj[:, 1]

        # ----- on-edge (inclusive) -----
        # cross((vj-vi), (p-vi)) ~ 0  AND  p is between vi and vj
        cross = (xj - xi) * (py - yi) - (yj - yi) * (px - xi)
        on_line = jnp.abs(cross) <= eps
        dot = (px - xi) * (px - xj) + (py - yi) * (py - yj)  # <= 0 means between endpoints
        on_seg_edge = on_line & (dot <= eps)
        on_seg = jnp.any(on_seg_edge & active)

        # ----- even–odd ray casting (horizontal ray to +∞) -----
        straddles = ((yi > py) ^ (yj > py)) & active
        denom = yj - yi
        # stabilize division (avoid 0/0 when horizontal):
        denom_safe = denom + jnp.sign(denom) * eps
        xints = xi + (xj - xi) * (py - yi) / denom_safe
        crosses = straddles & (px < xints)

        inside = jnp.count_nonzero(crosses) % 2 == 1
        return jnp.where(on_seg, True, inside)

    return jax.lax.cond(M < 3, _degenerate, _compute, operand=None)

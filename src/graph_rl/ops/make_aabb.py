# Standard library
from typing import Dict, Tuple
from functools import partial

# Third-party
import jax
import jax.numpy as jnp

# First-party
from graph_rl.ops.edge_intersections import edge_batch_intersection_loss
from graph_rl.grammar.constants import Constants
from graph_rl.utils.adjacency_utils import get_degree_vector, get_edge_vectors


@jax.jit
def active_mask_from_adjacency(adjacency: jnp.ndarray) -> jnp.ndarray:
    """Infer active vertices from padded adjacency (deg > 0), using an undirected view."""
    A = jnp.maximum(adjacency.astype(jnp.int32), adjacency.T.astype(jnp.int32))
    diag = jnp.arange(A.shape[0], dtype=jnp.int32)
    A = A.at[diag, diag].set(0)
    deg = jnp.sum(A, axis=1)
    return deg > 0


@partial(jax.jit, static_argnames=("num_angles",))
def rotation_invariant_aabb_bounds(
    vertices_xy: jnp.ndarray,  # (N, 2)
    adjacency_boundary: jnp.ndarray,  # (N, N)
    adjacency_full: jnp.ndarray,  # (N, N)
    num_angles: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Compute axis-aligned bounds for a set of rotated copies of the active (boundary) vertices.

    Rotation is performed around the centroid of the active points to avoid mixing rotation with
    translation. Angles are scanned in [0, π) because θ and θ + π yield identical AABBs.

    Returns:
        bounds: (K, 4) [min_x, max_x, min_y, max_y] in the rotated, centered frame.
        thetas: (K,) angles in radians.
        active: (N,) boolean active mask used for bounds.
    """
    # Prefer boundary adjacency for geometric bounds; fall back to full adjacency if needed.
    active_b = active_mask_from_adjacency(adjacency_boundary)
    active = jax.lax.cond(
        jnp.any(active_b),
        lambda _: active_b,
        lambda _: active_mask_from_adjacency(adjacency_full),
        operand=None,
    )

    any_act = jnp.any(active)

    def _compute(_: None) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        big = jnp.array(1e30, vertices_xy.dtype)
        count = jnp.maximum(active.sum(), 1).astype(vertices_xy.dtype)
        center = (vertices_xy * active[:, None].astype(vertices_xy.dtype)).sum(axis=0) / count
        Vc = vertices_xy - center

        thetas = jnp.linspace(0.0, jnp.pi, num_angles, endpoint=False, dtype=vertices_xy.dtype)

        def _bounds_for_theta(theta: jnp.ndarray) -> jnp.ndarray:
            c = jnp.cos(theta)
            s = jnp.sin(theta)
            R = jnp.stack([jnp.stack([c, -s]), jnp.stack([s, c])], axis=0)  # (2, 2)
            Vr = Vc @ R.T  # (N, 2)
            x = Vr[:, 0]
            y = Vr[:, 1]
            min_x = jnp.min(jnp.where(active, x, big))
            max_x = jnp.max(jnp.where(active, x, -big))
            min_y = jnp.min(jnp.where(active, y, big))
            max_y = jnp.max(jnp.where(active, y, -big))
            return jnp.stack([min_x, max_x, min_y, max_y], axis=0)

        bounds = jax.vmap(_bounds_for_theta)(thetas)
        return bounds, thetas, active

    return jax.lax.cond(
        any_act,
        _compute,
        lambda _: (
            jnp.zeros((num_angles, 4), dtype=vertices_xy.dtype),
            jnp.zeros((num_angles,), vertices_xy.dtype),
            active,
        ),
        operand=None,
    )


@partial(jax.jit, static_argnames=("num_angles",))
def best_oriented_square_rectangularity(
    vertices_xy: jnp.ndarray,  # (N, 2)
    adjacency_boundary: jnp.ndarray,  # (N, N)
    adjacency_full: jnp.ndarray,  # (N, N)
    area_polygon: jnp.ndarray,  # scalar
    perim_polygon: jnp.ndarray,  # scalar
    num_angles: int,
    beta: float = 5.0,
    eps: float = 1e-8,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Rotation-invariant square rectangularity: scan rotations, compute the square-AABB score, take best.

    Returns:
        best_score: scalar
        best_bounds: (4,) bounds in rotated, centered frame
        best_theta: scalar angle
        active: (N,) active mask used for bounds
    """
    bounds, thetas, active = rotation_invariant_aabb_bounds(
        vertices_xy=vertices_xy,
        adjacency_boundary=adjacency_boundary,
        adjacency_full=adjacency_full,
        num_angles=num_angles,
    )

    x_len = bounds[:, 1] - bounds[:, 0]
    y_len = bounds[:, 3] - bounds[:, 2]
    max_len = jnp.maximum(x_len, y_len)
    aabb_area = max_len**2

    area_clamped = jnp.minimum(area_polygon, aabb_area)
    rect_area_ratio = area_clamped / (aabb_area + eps)

    rectangularity = jnp.clip(rect_area_ratio, 0.0, 1.0)
    best_idx = jnp.argmax(rectangularity)
    return rectangularity[best_idx], bounds[best_idx], thetas[best_idx], active


@jax.jit
def make_cp_rectangle_wrapper(const: Constants) -> tuple[jnp.ndarray, jnp.ndarray, Constants, dict[str, jnp.ndarray]]:

    sink_mask = jnp.sum(const.adj, axis=1) == 0
    deg = get_degree_vector(const.adj)  # (N,) bool
    sink_degs = jnp.where(sink_mask, deg, 0)
    degree_loss = (
        sink_degs > 1
    ).sum() / sink_degs.sum()  # All boundaries need to be leaves for the AABB to work (leaf expansion)

    V1, A_new, info = add_aabb_rectangle(const.vp[:, :2], const.adj, tol=1e-9)
    V2, A_boundary = extend_leaf_vertices_to_aabb(V1, A_new, info, tol=1e-9)

    v_positions = jnp.pad(V2, ((0, 0), (0, 1)), "constant")
    c_vert_idx = jnp.array(list(info["corner_indices"].values()))
    constants_new = const.replace(adj_b=A_boundary, vp_flat=v_positions, c_vert_idx=c_vert_idx)

    edge_vectors = get_edge_vectors(constants_new.adj, constants_new.vp)[0]
    loss = edge_batch_intersection_loss(edge_vectors).mean()
    loss += degree_loss * 10.0  # weight degree loss

    len_bottom = jnp.linalg.norm(constants_new.vp[c_vert_idx[1]] - constants_new.vp[c_vert_idx[0]])
    len_right = jnp.linalg.norm(constants_new.vp[c_vert_idx[2]] - constants_new.vp[c_vert_idx[1]])
    len_diff = ((len_right - len_bottom) / (len_bottom)) ** 2
    return loss, len_diff, constants_new, info


@jax.jit
def add_aabb_rectangle(
    vertices: jnp.ndarray,  # (N,2) padded (unused slots arbitrary)
    adjacency: jnp.ndarray,  # (N,N) int {0,1} padded
    tol: float = 1e-9,
) -> Tuple[jnp.ndarray, jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Insert the AABB’s four corner vertices into free padded slots and union the
    rectangle’s perimeter edges into the adjacency. Shapes remain fixed (N,*).

    Active vertices are inferred from the (symmetrized) adjacency: deg(i) > 0.

    Args:
        vertices: (N,2) padded vertex positions.
        adjacency: (N,N) padded adjacency (0/1).
        tol: geometric tolerance.

    Returns:
        V_new: (N,2) vertices with corners written into 4 free slots (if available).
        A_out: (N,N) adjacency with perimeter edges unioned (undirected).
        info:  dict with:
               - "bounds": (4,) array [min_x, max_x, min_y, max_y]
               - "corner_indices": (4,) int32 [bl, br, tr, tl]
    """
    V = vertices
    N = V.shape[0]

    # Active vertices from symmetrized adjacency (zero diagonal)
    A = jnp.maximum(adjacency.astype(jnp.int32), adjacency.T.astype(jnp.int32))
    diag = jnp.arange(N, dtype=jnp.int32)
    A = A.at[diag, diag].set(0)
    deg = jnp.sum(A, axis=1)
    active = deg > 0

    # Bounds over active vertices (static shape, masked reduction)
    big = jnp.array(1e30, V.dtype)
    any_act = jnp.any(active)
    x, y = V[:, 0], V[:, 1]
    min_x = jnp.where(any_act, jnp.min(jnp.where(active, x, big)), 0.0)
    max_x = jnp.where(any_act, jnp.max(jnp.where(active, x, -big)), 0.0)
    min_y = jnp.where(any_act, jnp.min(jnp.where(active, y, big)), 0.0)
    max_y = jnp.where(any_act, jnp.max(jnp.where(active, y, -big)), 0.0)

    # Pick 4 free slots (deg==0)
    free_idx = jnp.where(~active, size=N, fill_value=-1)[0]
    slots = free_idx[:4]
    fits = jnp.all(slots >= 0)
    bl, br, tr, tl = slots[0], slots[1], slots[2], slots[3]

    # Corner coordinates
    corners = jnp.stack(
        [
            jnp.array([min_x, min_y]),
            jnp.array([max_x, min_y]),
            jnp.array([max_x, max_y]),
            jnp.array([min_x, max_y]),
        ],
        axis=0,
    )  # (4,2)

    # Write corners (guarded) and mark them active
    def _write(vm):
        Vw, Mw = vm
        Vw = Vw.at[slots].set(corners)
        Mw = Mw.at[slots].set(True)
        return Vw, Mw

    V_new, active2 = jax.lax.cond(fits, _write, lambda vm: vm, (V, active))

    # Side membership (full length, masked)
    XY = V_new
    on_left = (jnp.abs(XY[:, 0] - min_x) <= tol) & (XY[:, 1] >= min_y - tol) & (XY[:, 1] <= max_y + tol)
    on_right = (jnp.abs(XY[:, 0] - max_x) <= tol) & (XY[:, 1] >= min_y - tol) & (XY[:, 1] <= max_y + tol)
    on_bot = (jnp.abs(XY[:, 1] - min_y) <= tol) & (XY[:, 0] >= min_x - tol) & (XY[:, 0] <= max_x + tol)
    on_top = (jnp.abs(XY[:, 1] - max_y) <= tol) & (XY[:, 0] >= min_x - tol) & (XY[:, 0] <= max_x + tol)

    def _force(sides):
        L, R, B, T = sides
        L = L.at[jnp.array([bl, tl])].set(True)
        R = R.at[jnp.array([br, tr])].set(True)
        B = B.at[jnp.array([bl, br])].set(True)
        T = T.at[jnp.array([tl, tr])].set(True)
        return L, R, B, T

    on_left, on_right, on_bot, on_top = jax.lax.cond(fits, _force, lambda s: s, (on_left, on_right, on_bot, on_top))
    on_left &= active2
    on_right &= active2
    on_bot &= active2
    on_top &= active2

    # Chain along sides (fixed shapes; no compaction)
    bigf = jnp.array(1e30, XY.dtype)

    def chain(mask, key):
        k = jnp.where(mask, key, bigf)
        order = jnp.argsort(k)  # (N,)
        u, v = order[:-1], order[1:]  # (N-1,)
        valid = mask[u] & mask[v] & (u != v)
        return u, v, valid

    uL, vL, mL = chain(on_left, XY[:, 1])
    uR, vR, mR = chain(on_right, XY[:, 1])
    uB, vB, mB = chain(on_bot, XY[:, 0])
    uT, vT, mT = chain(on_top, XY[:, 0])

    u = jnp.concatenate([uL, uR, uB, uT], axis=0)
    v = jnp.concatenate([vL, vR, vB, vT], axis=0)
    m = jnp.concatenate([mL, mR, mB, mT], axis=0).astype(jnp.uint8)

    a = jnp.minimum(u, v)
    b = jnp.maximum(u, v)

    def _write_edges(A):
        A = A.at[a, b].max(m)
        A = A.at[b, a].max(m)
        return A

    A_out = jax.lax.cond(fits, _write_edges, lambda A: A, adjacency)
    A_out = A_out.at[diag, diag].set(0)

    info = {
        "corner_indices": {"bl": bl, "br": br, "tr": tr, "tl": tl},
        "bounds": jnp.array([min_x, max_x, min_y, max_y]),
    }
    return V_new, A_out, info


@jax.jit
def extend_leaf_vertices_to_aabb(
    vertices: jnp.ndarray,  # (N,2) padded
    adjacency: jnp.ndarray,  # (N,N) int {0,1} padded
    info: Dict[str, jnp.ndarray],  # {"bounds": (4,), "corner_indices": (4,)}
    tol: float = 1e-9,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Move leaf vertices (deg==1) that are not yet on the boundary to their
    intersections with the AABB, then rebuild only the AABB perimeter adjacency.

    Args:
        vertices: (N,2) padded vertex array.
        adjacency: (N,N) padded adjacency (0/1).
        info: dict with
              - "bounds": (4,) array [min_x, max_x, min_y, max_y]
              - "corner_indices": (4,) int32 [bl, br, tr, tl] (unused here, but kept minimal & stable)
        tol: geometric tolerance.

    Returns:
        V2: (N,2) updated vertex positions.
        A_aabb: (N,N) perimeter adjacency (undirected, 0/1).
    """
    V = vertices
    N = V.shape[0]
    min_x, max_x, min_y, max_y = info["bounds"]

    # Symmetrize, zero diag, degree & leafs (padding stays deg==0)
    A = jnp.maximum(adjacency.astype(jnp.int32), adjacency.T.astype(jnp.int32))
    diag = jnp.arange(N, dtype=jnp.int32)
    A = A.at[diag, diag].set(0)
    deg = jnp.sum(A, axis=1)

    # “Original” = not already on boundary (pre-move)
    XY0 = V[:, :2]
    pre_left = (jnp.abs(XY0[:, 0] - min_x) <= tol) & (XY0[:, 1] >= min_y - tol) & (XY0[:, 1] <= max_y + tol)
    pre_right = (jnp.abs(XY0[:, 0] - max_x) <= tol) & (XY0[:, 1] >= min_y - tol) & (XY0[:, 1] <= max_y + tol)
    pre_bot = (jnp.abs(XY0[:, 1] - min_y) <= tol) & (XY0[:, 0] >= min_x - tol) & (XY0[:, 0] <= max_x + tol)
    pre_top = (jnp.abs(XY0[:, 1] - max_y) <= tol) & (XY0[:, 0] >= min_x - tol) & (XY0[:, 0] <= max_x + tol)
    pre_side = pre_left | pre_right | pre_bot | pre_top
    leaf_mask = (deg == 1) & (~pre_side)

    # Rays from leaf toward (leaf - neighbor), intersect with AABB
    nbr = jnp.argmax(A, axis=1).astype(jnp.int32)
    ray_origins = V[:, :2]
    U = V[nbr, :2]
    d = ray_origins - U
    dn = jnp.linalg.norm(d, axis=1, keepdims=True)
    d_unit = d / jnp.maximum(dn, 1e-12)

    def safe_div(a, b):
        return jnp.where(jnp.abs(b) <= tol, jnp.sign(a) * jnp.inf, a / b)

    tx1 = safe_div(min_x - ray_origins[:, 0], d_unit[:, 0])
    tx2 = safe_div(max_x - ray_origins[:, 0], d_unit[:, 0])
    ty1 = safe_div(min_y - ray_origins[:, 1], d_unit[:, 1])
    ty2 = safe_div(max_y - ray_origins[:, 1], d_unit[:, 1])

    def ok_x(t):
        y = ray_origins[:, 1] + t * d_unit[:, 1]
        return (t >= -tol) & (y >= min_y - tol) & (y <= max_y + tol)

    def ok_y(t):
        x = ray_origins[:, 0] + t * d_unit[:, 0]
        return (t >= -tol) & (x >= min_x - tol) & (x <= max_x + tol)

    big = jnp.array(1e30, V.dtype)
    C = jnp.stack(
        [
            jnp.where(ok_x(tx1), tx1, big),
            jnp.where(ok_x(tx2), tx2, big),
            jnp.where(ok_y(ty1), ty1, big),
            jnp.where(ok_y(ty2), ty2, big),
        ],
        axis=1,
    )
    t = jnp.min(C, axis=1)
    P = ray_origins + t[:, None] * d_unit

    # Snap numerically to sides
    Px = jnp.where(jnp.abs(P[:, 0] - min_x) <= tol, min_x, P[:, 0])
    Px = jnp.where(jnp.abs(Px - max_x) <= tol, max_x, Px)
    Py = jnp.where(jnp.abs(P[:, 1] - min_y) <= tol, min_y, P[:, 1])
    Py = jnp.where(jnp.abs(Py - max_y) <= tol, max_y, Py)
    P = jnp.stack([Px, Py], axis=1)

    # Move only inferred leaves
    V2 = V.at[:, :2].set(jnp.where(leaf_mask[:, None], P, V[:, :2]))

    # Perimeter adjacency from chained side membership
    XY = V2[:, :2]
    on_left = (jnp.abs(XY[:, 0] - min_x) <= tol) & (XY[:, 1] >= min_y - tol) & (XY[:, 1] <= max_y + tol)
    on_right = (jnp.abs(XY[:, 0] - max_x) <= tol) & (XY[:, 1] >= min_y - tol) & (XY[:, 1] <= max_y + tol)
    on_bot = (jnp.abs(XY[:, 1] - min_y) <= tol) & (XY[:, 0] >= min_x - tol) & (XY[:, 0] <= max_x + tol)
    on_top = (jnp.abs(XY[:, 1] - max_y) <= tol) & (XY[:, 0] >= min_x - tol) & (XY[:, 0] <= max_x + tol)

    bigf = jnp.array(1e30, XY.dtype)

    def chain(mask, key):
        k = jnp.where(mask, key, bigf)
        order = jnp.argsort(k)  # (N,)
        u, v = order[:-1], order[1:]  # (N-1,)
        valid = mask[u] & mask[v] & (u != v)
        return u, v, valid

    uL, vL, mL = chain(on_left, XY[:, 1])
    uR, vR, mR = chain(on_right, XY[:, 1])
    uB, vB, mB = chain(on_bot, XY[:, 0])
    uT, vT, mT = chain(on_top, XY[:, 0])

    u = jnp.concatenate([uL, uR, uB, uT], axis=0)
    v = jnp.concatenate([vL, vR, vB, vT], axis=0)
    m = jnp.concatenate([mL, mR, mB, mT], axis=0).astype(jnp.uint8)

    a = jnp.minimum(u, v)
    b = jnp.maximum(u, v)

    A_aabb = jnp.zeros((N, N), dtype=jnp.uint8)
    A_aabb = A_aabb.at[a, b].max(m)
    A_aabb = A_aabb.at[b, a].max(m)
    A_aabb = A_aabb.at[diag, diag].set(0)

    return V2, A_aabb

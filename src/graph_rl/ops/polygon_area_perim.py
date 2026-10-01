# Third-party
import jax
import jax.numpy as jnp


def _sym(A):
    return jnp.maximum(A, A.T)


def _walk_order(adj):
    """Return a permutation 'order' that traces the boundary cycle.
    Runs for N steps (static), but we will later mask to the first K entries."""
    N = adj.shape[0]
    deg = jnp.sum(adj, axis=1)
    start = jnp.argmax(deg > 0)  # first boundary vertex

    def step(carry, _):
        cur, prev, order = carry
        order = order.at[_].set(cur)
        nb_mask = adj[cur] * (1.0 - jax.nn.one_hot(prev, N))  # exclude where prev==index
        nxt = jnp.argmax(nb_mask)  # the other neighbor on the cycle
        return (nxt, cur, order), 0

    init_order = jnp.full((adj.shape[0],), start, dtype=jnp.int32)
    (_, _, order), _ = jax.lax.scan(step, (start, start, init_order), jnp.arange(adj.shape[0]))
    return order, (deg > 0)


@jax.jit
def polygon_area_perimeter_from_boundary(adj_b: jnp.ndarray, xy: jnp.ndarray):
    """
    adj_b: (N,N) boundary adjacency (single simple cycle), possibly padded with zeros
    xy:    (N,2) coordinates, padded rows arbitrary
    Returns (area, perimeter) for the non-convex polygon.
    """
    adj = _sym(adj_b).astype(xy.dtype)
    N = adj.shape[0]

    order, on_boundary = _walk_order(adj)
    K = jnp.sum(on_boundary.astype(jnp.int32))  # number of boundary verts

    pts = xy[order]  # (N,2)

    # Edges t -> t+1 for t = 0..K-2
    pts_next = jnp.roll(pts, -1, axis=0)  # shift once
    edge_mask_seq = jnp.arange(N) < (K - 1)  # True for t in [0, K-2]

    cross_seq = pts[:, 0] * pts_next[:, 1] - pts[:, 1] * pts_next[:, 0]
    area_seq = jnp.sum(cross_seq * edge_mask_seq.astype(xy.dtype))

    # Closing edge (K-1) -> 0 added explicitly
    e_last = jax.nn.one_hot(jnp.maximum(K - 1, 0), N, dtype=xy.dtype)  # handles K=0 safely
    p_last = pts.T @ e_last  # (2,)
    p_first = pts[0]  # (2,)
    cross_last = p_last[0] * p_first[1] - p_last[1] * p_first[0]

    area = 0.5 * jnp.abs(area_seq + cross_last * (K > 0).astype(xy.dtype))

    # Perimeter: same masking
    seg = pts_next - pts
    seg_len = jnp.linalg.norm(seg, axis=1)
    perim_seq = jnp.sum(seg_len * edge_mask_seq.astype(xy.dtype))

    # Closing segment length |p_last - p_first|
    perim_last = jnp.linalg.norm(p_last - p_first)
    perimeter = perim_seq + perim_last * (K > 0).astype(xy.dtype)

    return area, perimeter

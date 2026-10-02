"""Baseline isoperimetric-ratio and squareness utilities from raw 2D point clouds."""

from __future__ import annotations


import alphashape
import numpy as np
from shapely.geometry import MultiPoint, MultiPolygon, Point, Polygon

__all__ = [
    "find_valid_alpha_for_points",
    "get_iso_and_squareness_for_baselines",
    "graph_shape_from_edges",
    "isoperimetric_ratio_from_shape",
]


# alphashape 1.3.1 expects MultiPoint to be iterable, which is no longer true in Shapely 2.
if not hasattr(MultiPoint, "__iter__"):
    MultiPoint.__iter__ = lambda self: iter(self.geoms)


def _compute_alpha_shape(points: np.ndarray, *, alpha: float | None = None):
    if alpha is not None:
        if not np.isfinite(alpha) or alpha < 0.0:
            raise ValueError(f"alpha must be non-negative and finite, got {alpha}")
        return alphashape.alphashape(points, float(alpha))

    return _find_valid_alpha_shape(points, alpha_upper=8.0, iterations=32)[1]


def _normalize_points_unit_box(points: np.ndarray) -> np.ndarray:
    """Uniformly scale 2D points to fit maximally inside [0, 1]^2 without distortion."""
    mins = np.min(points, axis=0)
    maxs = np.max(points, axis=0)
    spans = maxs - mins
    max_span = float(np.max(spans))
    if max_span <= 1e-12:
        return points - mins
    return (points - mins) / max_span


def _is_valid_alpha_shape(shape, points: np.ndarray) -> bool:
    if not isinstance(shape, Polygon):
        return False
    return all(shape.covers(Point(float(x), float(y))) for x, y in points)


def _find_valid_alpha_shape(
    points: np.ndarray,
    *,
    alpha_upper: float = 8.0,
    iterations: int = 32,
) -> tuple[float, object]:
    """Find the largest alpha in [0, alpha_upper] that yields one polygon covering all points."""
    low = 0.0
    high = float(alpha_upper)

    hull = alphashape.alphashape(points, 0.0)
    if not _is_valid_alpha_shape(hull, points):
        return 0.0, hull

    best_alpha = 0.0
    best_shape = hull

    for _ in range(iterations):
        mid = 0.5 * (low + high)
        shape = alphashape.alphashape(points, mid)
        if _is_valid_alpha_shape(shape, points):
            best_alpha = mid
            best_shape = shape
            low = mid
        else:
            high = mid

    return best_alpha, best_shape


def isoperimetric_ratio_from_shape(shape, eps: float = 1e-8) -> float:
    area = float(shape.area)
    perimeter = float(shape.length)
    if perimeter <= eps:
        return 0.0
    return float(np.clip(4.0 * np.pi * area / (perimeter * perimeter + eps), 0.0, 1.0))


def _shape_boundary_xy(shape) -> np.ndarray:
    if isinstance(shape, Polygon):
        return np.asarray(shape.exterior.coords[:-1], dtype=np.float64)

    if isinstance(shape, MultiPolygon):
        parts = [np.asarray(poly.exterior.coords[:-1], dtype=np.float64) for poly in shape.geoms]
        parts = [part for part in parts if part.shape[0] >= 3]
        if not parts:
            return np.zeros((0, 2), dtype=np.float64)
        return np.vstack(parts)

    if hasattr(shape, "convex_hull") and isinstance(shape.convex_hull, Polygon):
        return np.asarray(shape.convex_hull.exterior.coords[:-1], dtype=np.float64)

    return np.zeros((0, 2), dtype=np.float64)


def _best_oriented_square_rectangularity(
    boundary_xy: np.ndarray,
    area_polygon: float,
    *,
    num_angles: int = 181,
    eps: float = 1e-8,
) -> float:
    if boundary_xy.shape[0] < 3:
        return 0.0

    center = boundary_xy.mean(axis=0)
    centered = boundary_xy - center
    thetas = np.linspace(0.0, np.pi, num_angles, endpoint=False)

    best_score = 0.0
    for theta in thetas:
        c = np.cos(theta)
        s = np.sin(theta)
        rotation = np.array([[c, -s], [s, c]], dtype=np.float64)
        rotated = centered @ rotation.T

        min_x = float(np.min(rotated[:, 0]))
        max_x = float(np.max(rotated[:, 0]))
        min_y = float(np.min(rotated[:, 1]))
        max_y = float(np.max(rotated[:, 1]))

        x_len = max_x - min_x
        y_len = max_y - min_y
        max_len = max(x_len, y_len)
        aabb_area = max_len**2

        area_clamped = min(area_polygon, aabb_area)
        score = np.clip(area_clamped / (aabb_area + eps), 0.0, 1.0)
        best_score = max(best_score, float(score))

    return best_score


def graph_shape_from_edges(
    pos: np.ndarray,
    *,
    alpha: float | None = None,
):
    """Compatibility wrapper that computes an alpha shape from points only."""
    points = np.asarray(pos, dtype=np.float64)
    points = _normalize_points_unit_box(points[:, :2])
    shape = _compute_alpha_shape(points, alpha=alpha)

    if isinstance(shape, Polygon):
        return shape, [shape]
    if isinstance(shape, MultiPolygon):
        return shape, list(shape.geoms)
    return shape, []


def find_valid_alpha_for_points(
    raw_points: np.ndarray,
    *,
    alpha_upper: float = 8.0,
    iterations: int = 32,
) -> tuple[float, object]:
    """Public helper for the largest valid alpha after unit-box normalization."""
    points = np.asarray(raw_points, dtype=np.float64)
    points = _normalize_points_unit_box(points[:, :2])
    return _find_valid_alpha_shape(points, alpha_upper=alpha_upper, iterations=iterations)


def get_iso_and_squareness_for_baselines(raw_points: np.ndarray) -> tuple[dict[str, float], object]:
    """Return alpha-shape isoperimetric ratio and square-rectangularity from raw 2D points."""
    alpha, shape = find_valid_alpha_for_points(raw_points, alpha_upper=8.0, iterations=10)
    # points = np.asarray(raw_points, dtype=np.float64)
    eps = 1e-8

    area_polygon = float(shape.area)
    perimeter_polygon = float(shape.length)
    if area_polygon <= eps or perimeter_polygon <= eps:
        return {"iso": 0.0, "squareness": 0.0}, shape

    isoperimetric_ratio = isoperimetric_ratio_from_shape(shape, eps=eps)
    boundary_xy = _shape_boundary_xy(shape)
    squareness = _best_oriented_square_rectangularity(
        boundary_xy,
        area_polygon,
        num_angles=181,
        eps=eps,
    )

    metrics = {
        "iso": isoperimetric_ratio,
        "squareness": float(np.clip(squareness, 0.0, 1.0)),
    }
    return metrics, shape


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from pathlib import Path

    rng = np.random.default_rng(8)
    points = rng.uniform(0.0, 1.0, size=(48, 2))
    points = np.array(
        [
            (483.945455, 41.209091),
            (393.073059, 54.000000),
            (343.413146, 57.535211),
            (233.863636, 81.500000),
            (453.358140, 84.590698),
            (286.529680, 92.840183),
            (541.756881, 94.403670),
            (444.464789, 116.586854),
            (312.256881, 122.669725),
            (582.936073, 123.890411),
            (213.298165, 135.701835),
            (404.260274, 138.292237),
            (348.809091, 140.786364),
            (532.000000, 144.000000),
            (135.100000, 145.027273),
            (192.145455, 156.077273),
            (491.854545, 160.922727),
            (264.410138, 168.281106),
            (653.077273, 174.654545),
            (570.641860, 186.409302),
            (115.595455, 189.890909),
            (612.529954, 205.764977),
            (191.136986, 207.885845),
            (133.077273, 229.854545),
            (68.000000, 237.000000),
            (128.365297, 266.242009),
            (51.552511, 276.036530),
            (108.073059, 320.000000),
            (50.000000, 343.420091),
            (82.529954, 344.235023),
            (608.000000, 364.926941),
            (547.990909, 375.036364),
            (44.410138, 377.281106),
            (611.936073, 402.890411),
            (104.922727, 409.145455),
            (77.493151, 424.826484),
            (504.899083, 439.100917),
            (535.100000, 440.972727),
            (615.000000, 453.541284),
            (77.036530, 461.447489),
            (108.596330, 463.243119),
            (455.589862, 462.718894),
            (157.242009, 469.365297),
            (568.000000, 469.000000),
            (377.718894, 486.410138),
            (121.703704, 495.625000),
            (240.109589, 501.063927),
            (441.757991, 510.365297),
            (539.529680, 513.159817),
            (294.463303, 513.802752),
            (203.586854, 521.464789),
            (332.127273, 529.550000),
            (433.407407, 535.319444),
            (503.917431, 546.036697),
            (188.627907, 547.334884),
            (149.756881, 547.596330),
            (352.413146, 568.535211),
            (247.063927, 584.890411),
            (405.500000, 587.863636),
            (293.625000, 588.703704),
        ]
    )

    # points = [
    #     (222.529680, 44.159817),
    #     (242.669725, 72.743119),
    #     (532.281106, 86.589862),
    #     (361.922727, 100.654545),
    #     (301.529954, 110.235023),
    #     (270.500000, 110.500000),
    #     (335.410138, 111.281106),
    #     (528.863636, 121.500000),
    #     (371.410959, 126.794521),
    #     (263.159817, 139.529680),
    #     (313.235023, 139.529954),
    #     (361.410138, 159.718894),
    #     (526.145455, 167.922727),
    #     (589.890411, 177.936073),
    #     (345.890411, 193.063927),
    #     (560.885845, 196.863014),
    #     (525.334884, 221.627907),
    #     (370.036697, 257.917431),
    #     (636.541284, 277.000000),
    #     (602.294931, 279.336406),
    #     (556.876712, 282.246575),
    #     (505.625000, 285.703704),
    #     (426.641860, 290.590698),
    #     (381.365297, 298.242009),
    #     (337.631336, 301.741935),
    #     (445.612150, 308.565421),
    #     (280.757991, 317.365297),
    #     (43.500000, 321.863636),
    #     (406.413146, 327.535211),
    #     (228.764706, 328.778281),
    #     (348.739726, 336.292237),
    #     (181.447489, 337.963470),
    #     (69.298165, 339.701835),
    #     (538.463303, 345.197248),
    #     (138.488584, 348.100457),
    #     (377.854545, 349.077273),
    #     (572.000000, 351.000000),
    #     (102.707763, 363.260274),
    #     (558.500000, 368.429907),
    #     (551.603774, 397.500000),
    #     (121.063927, 400.109589),
    #     (145.500000, 436.211009),
    #     (566.000000, 439.926941),
    #     (580.358140, 470.590698),
    #     (163.890411, 477.936073),
    #     (183.626168, 518.373832),
    #     (204.560748, 556.383178),
    #     (225.500000, 590.935780),
    #     (244.718894, 621.410138),
    #     (255.378995, 648.940639),
    # ]

    # points = [
    #     (156.359507, 218.038258),
    #     (71.930187, 67.147243),
    #     (143.318433, 140.848122),
    #     (39.492419, 144.580041),
    #     (27.550077, 113.218566),
    #     (42.172226, 217.405180),
    #     (37.870681, 181.493184),
    #     (43.817605, 193.405285),
    #     (122.082238, 73.002651),
    #     (85.836822, 299.542703),
    #     (110.911922, 52.112358),
    #     (14.302834, 150.756998),
    #     (155.280492, 103.358357),
    #     (18.440847, 176.645376),
    #     (143.160351, 68.967357),
    #     (40.678831, 324.720000),
    #     (63.783982, 214.915686),
    #     (130.417494, 203.599115),
    #     (115.661222, 22.320000),
    #     (109.952695, 90.303625),
    #     (41.133978, 162.191854),
    #     (77.773879, 142.480494),
    #     (56.915998, 302.020867),
    #     (113.727429, 172.066876),
    #     (62.149820, 243.167619),
    #     (98.881100, 226.105933),
    #     (53.721482, 109.888166),
    #     (101.284916, 119.194833),
    #     (120.471724, 133.946992),
    #     (129.545878, 101.095478),
    #     (57.433453, 41.240064),
    #     (81.437129, 272.915449),
    #     (103.170592, 216.347173),
    #     (30.245510, 138.976352),
    # ]

    points = np.array(points)

    points = _normalize_points_unit_box(points)

    metrics, shape = get_iso_and_squareness_for_baselines(points)

    print("num_points:", points.shape[0], flush=True)
    print("alpha:", 0, flush=True)
    print("area:", float(shape.area), flush=True)
    print("perimeter:", float(shape.length), flush=True)
    print("iso:", float(metrics["iso"]), flush=True)
    print("squareness:", float(metrics["squareness"]), flush=True)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(points[:, 0], points[:, 1], s=10, color="black", zorder=2)

    if isinstance(shape, Polygon):
        x, y = shape.exterior.xy
        ax.fill(x, y, color="tab:orange", alpha=0.25, zorder=0)
        ax.plot(x, y, color="tab:red", linewidth=2.0, zorder=3)
    elif isinstance(shape, MultiPolygon):
        for poly in shape.geoms:
            x, y = poly.exterior.xy
            ax.fill(x, y, color="tab:orange", alpha=0.25, zorder=0)
            ax.plot(x, y, color="tab:red", linewidth=2.0, zorder=3)

    ax.set_aspect("equal")
    ax.set_title(f"alpha shape | alpha={0:.2f} | iso={metrics['iso']:.3f} | square={metrics['squareness']:.3f}")
    output_path = Path("alphashape_demo.png")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print("saved_plot:", output_path, flush=True)

    if plt.get_backend().lower() != "agg":
        plt.show()

"""Closest points from a point on the axis of an ELLIPTICAL cone: two points.

Unlike the circular cone (see ``cone_axis_ring.py``), an elliptical cross
section breaks the ring degeneracy: the closest-point set is exactly the two
isolated minima toward the minor-axis directions. Both are equidistant, so
BOTH are part of the answer set — a deterministic, reproducible result where a
single-answer engine would return one of them by implementation chance.
"""
import numpy as np

from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple
from mmcore.numeric._bez_closest_point import nurbs_surface_closest_points


def full_elliptical_cone(a=4.0, b=2.0, height=4.0):
    """Full-revolution elliptical cone (semi-axes a, b; apex at z=height)."""
    s = np.sqrt(2) / 2
    # anisotropically scaled 9-point rational circle is an exact ellipse
    ell = np.array([[a, 0, 0], [a, b, 0], [0, b, 0], [-a, b, 0], [-a, 0, 0],
                    [-a, -b, 0], [0, -b, 0], [a, -b, 0], [a, 0, 0]], dtype=float)
    wrow = np.array([1, s, 1, s, 1, s, 1, s, 1], dtype=float)
    cps = np.zeros((9, 2, 3))
    cps[:, 0, :] = np.array([0.0, 0.0, height])
    cps[:, 1, :] = ell
    weights = np.column_stack([wrow, wrow])
    knot_u = np.array([0, 0, 0, .25, .25, .5, .5, .75, .75, 1, 1, 1], dtype=float)
    knot_v = np.array([0, 0, 1, 1], dtype=float)
    return NURBSSurfaceTuple(3, 2, knot_u, knot_v, cps, weights)


if __name__ == "__main__":
    import time

    cone = full_elliptical_cone(a=4.0, b=2.0, height=4.0)
    query = np.array([0.0, 0.0, 1.0])

    t0 = time.perf_counter()
    res = nurbs_surface_closest_points(cone, query, atol=1e-6)
    dt = time.perf_counter() - t0

    print(f"query on the elliptical-cone axis  ({dt * 1000:.1f} ms)")
    for e in res:
        print(f"  kind={e['kind']:14s} distance={e['distance']:.6f}  "
              f"u={e['u']:.4f} v={e['v']:.4f}  point={np.round(e['point'], 4)}")
    mins = [e for e in res if e["kind"] in ("min", "boundary_min")]
    print(f"-> {len(mins)} equidistant closest points (the minor-axis directions), "
          "all reported: the answer set is deterministic.")

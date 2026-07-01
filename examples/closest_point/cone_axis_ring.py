"""Closest points from a point ON THE AXIS of a circular cone: a ring.

The closest-point set is a full circle on the cone (a 1-D equidistant curve).
The solver detects the rank-1 degenerate stationary structure, TRACES the ring
(Pull-Curve-style predictor-corrector on the stationarity system grad g = 0),
stitches the four per-patch traces across the seams of the revolution, and
returns ONE closed ``degenerate_curve`` entity — instead of an arbitrary point
somewhere on the ring.
"""
import numpy as np

from mmcore.geom._nurbs_eval import NURBSSurfaceTuple
from mmcore.numeric._bez_closest_point import nurbs_surface_closest_points


def full_cone(radius=3.0, height=4.0):
    """Full-revolution cone: apex at (0,0,height), base circle radius in z=0."""
    s = np.sqrt(2) / 2
    circ = np.array([[radius, 0, 0], [radius, radius, 0], [0, radius, 0],
                     [-radius, radius, 0], [-radius, 0, 0], [-radius, -radius, 0],
                     [0, -radius, 0], [radius, -radius, 0], [radius, 0, 0]], dtype=float)
    wrow = np.array([1, s, 1, s, 1, s, 1, s, 1], dtype=float)
    cps = np.zeros((9, 2, 3))
    cps[:, 0, :] = np.array([0.0, 0.0, height])    # degenerate apex row
    cps[:, 1, :] = circ
    weights = np.column_stack([wrow, wrow])
    knot_u = np.array([0, 0, 0, .25, .25, .5, .5, .75, .75, 1, 1, 1], dtype=float)
    knot_v = np.array([0, 0, 1, 1], dtype=float)
    return NURBSSurfaceTuple(3, 2, knot_u, knot_v, cps, weights)


if __name__ == "__main__":
    import time

    R, H = 3.0, 4.0
    cone = full_cone(R, H)
    query = np.array([0.0, 0.0, 1.6])              # on the axis

    # analytic ground truth: distance from (r=0, z=1.6) to the slant segment
    a, b, p = np.array([0.0, H]), np.array([R, 0.0]), np.array([0.0, 1.6])
    t = np.clip(np.dot(p - a, b - a) / np.dot(b - a, b - a), 0, 1)
    d_ref = float(np.linalg.norm(p - (a + t * (b - a))))

    t0 = time.perf_counter()
    res = nurbs_surface_closest_points(cone, query, atol=1e-6)
    dt = time.perf_counter() - t0

    print(f"query on the cone axis  ({dt * 1000:.1f} ms), analytic ring distance = {d_ref:.6f}")
    for e in res:
        if e["kind"] == "degenerate_curve":
            d = np.linalg.norm(e["points"] - query[None, :], axis=1)
            print(f"  kind={e['kind']}  closed={e['closed']}  "
                  f"n_points={len(e['uv'])}  distance={e['distance']:.6f}  "
                  f"(spread {d.max() - d.min():.2e})")
            print(f"  ring in UV: u in [{e['uv'][:, 0].min():.3f}, {e['uv'][:, 0].max():.3f}], "
                  f"v ~ {e['uv'][:, 1].mean():.4f}")
            if "circle" in e:
                # Every equidistant curve lies on the sphere of radius d_min
                # about the query; this one is planar, hence an EXACT circle.
                c = e["circle"]
                print(f"  certified circle: center={np.round(c['center'], 6)}  "
                      f"radius={c['radius']:.6f}  normal={np.round(c['normal'], 6)}  "
                      f"arc={np.degrees(c['arc_angle']):.1f} deg")
        else:
            print(f"  kind={e['kind']}  distance={e['distance']:.6f} "
                  f"(u={e['u']:.4f}, v={e['v']:.4f})")
    print("-> the answer is a CLOSED equidistant ring, returned as one curve entity.")

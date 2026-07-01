"""Closest point from the CENTER of a NURBS circle: the whole curve.

Per single Bézier segment the equidistant set is provably all-or-nothing;
here all four arcs of the circle are wholly equidistant, and the NURBS wrapper
merges them into ONE ``degenerate_segment`` entity spanning the full domain —
the honest answer instead of an arbitrary parameter on the circle.
"""
import numpy as np

from mmcore.geom._nurbs_eval import NURBSCurveTuple
from mmcore.numeric._bez_closest_point import nurbs_curve_closest_points


def full_circle(radius=2.5):
    s = np.sqrt(2) / 2
    cps = radius * np.array([[1, 0, 0], [1, 1, 0], [0, 1, 0], [-1, 1, 0],
                             [-1, 0, 0], [-1, -1, 0], [0, -1, 0], [1, -1, 0],
                             [1, 0, 0]], dtype=float)
    w = np.array([1, s, 1, s, 1, s, 1, s, 1], dtype=float)
    knot = np.array([0, 0, 0, .25, .25, .5, .5, .75, .75, 1, 1, 1], dtype=float)
    return NURBSCurveTuple(3, knot, cps, w)


if __name__ == "__main__":
    circle = full_circle(2.5)

    # dead-center: the entire circle is the closest set
    res = nurbs_curve_closest_points(circle, np.zeros(3), atol=1e-6)
    print("query at the circle center:")
    for e in res:
        print(f"  kind={e['kind']}  distance={e['distance']:.9f}  t_range={e.get('t_range')}")

    # slightly off-center: the degeneracy breaks into a single closest point
    res2 = nurbs_curve_closest_points(circle, np.array([0.3, 0.0, 0.0]), atol=1e-6)
    print("query 0.3 off-center:")
    for e in res2:
        print(f"  kind={e['kind']}  distance={e['distance']:.9f}  t={e.get('t'):.6f}")

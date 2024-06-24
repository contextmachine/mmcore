import math

import numpy as np

from mmcore.geom.bvh import PSegment, build_bvh
from mmcore.geom.curves.curve import Curve


class Spline(Curve):
    degree: int


def is_spline(curve: Curve):
    return hasattr(curve, 'degree')

def spline_curve_bvh(curve: Spline, count=None):
    start, end = curve.interval()
    if count is None:
        count = math.ceil(end - start) * curve.degree
    t = np.linspace(start, end, count + 1)
    ts = np.empty((count, 2))
    pts = np.empty((count, 2, 3))
    res = curve(t)
    segments = []
    for i in range(count):
        pts[i, 0, :] = res[i]
        pts[i, 1, :] = res[i + 1]
        ts[i, 0] = t[i]
        ts[i, 1] = t[i + 1]
        segments.append(PSegment(pts[i], ts[i]))

    return build_bvh(segments)

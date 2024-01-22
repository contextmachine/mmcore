from typing import Any

import numpy as np
from multipledispatch import dispatch

from mmcore.func import vectorize
from mmcore.geom.vec import unit


@vectorize(signature='(i),(i),(i)->()')
def ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray[Any, np.dtype[np.bool_]]:

    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])


@dispatch(np.ndarray, np.ndarray)
@vectorize(signature='(j, i),(j, i)->()')
def intersects_segments(ab: np.ndarray, cd: np.ndarray) -> bool:
    a, b = ab
    c, d = cd
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


@dispatch(object, object)
@vectorize(excluded=[0], signature='(j, i)->()')
def intersects_segments(ab, cd) -> bool:
    a, b = ab.start, ab.end
    c, d = cd.start, cd.end
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)


from mmcore.geom.polyline import polyline_to_lines


def aabb(points: np.ndarray):

    return np.array([np.min(points, axis=0), np.max(points, axis=0)])


@vectorize(signature='(j,i),(i)->()')
def point_in_polygon(polygon: np.ndarray, point: np.ndarray):
    inside = polygon[1] + unit((polygon[0] - polygon[1]) + (polygon[2] - polygon[1]))
    cnt = len(np.arange(len(polygon))[intersects_segments(polyline_to_lines(polygon), np.array([point, inside]))]
            )
    return cnt % 2 == 0

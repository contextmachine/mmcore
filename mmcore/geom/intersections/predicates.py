from typing import Any

import numpy as np
from multipledispatch import dispatch
from scipy.spatial import KDTree

from mmcore.func import vectorize
from mmcore.geom.vec import dist, unit
from mmcore.numeric import remove_dim


@vectorize(signature='(i),(i),(i)->()')
def ccw(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray[Any, np.dtype[np.bool_]]:

    return (c[1] - a[1]) * (b[0] - a[0]) > (b[1] - a[1]) * (c[0] - a[0])



@vectorize(signature='(j, i),(j, i)->()')
def intersects_segments(ab: np.ndarray, cd: np.ndarray) -> bool:
    a, b = ab
    c, d = cd
    return ccw(a, c, d) != ccw(b, c, d) and ccw(a, b, c) != ccw(a, b, d)





from mmcore.geom.polyline import polyline_to_lines



def aabb(points: np.ndarray):
    return np.array((np.min(points, axis=len(points.shape) - 2), np.max(points, axis=len(points.shape) - 2)))


aabb_vectorized = np.vectorize(aabb, signature='(i,j)->(k,j)')


def point_indices(unq, other, eps=1e-6, dist_upper_bound=None, return_distance=True):
    kd = KDTree(unq)
    dists, ixs = kd.query(other, eps=eps, distance_upper_bound=dist_upper_bound)
    if return_distance:
        return dists, ixs
    else:
        return ixs


@vectorize(signature='(j,i),(i)->()')
def point_in_polygon(polygon: np.ndarray, point: np.ndarray):
    inside = polygon[1] + unit((polygon[0] - polygon[1]) + (polygon[2] - polygon[1]))

    cnt = len(np.arange(len(polygon))[intersects_segments(polyline_to_lines(polygon), [point, inside])]
              )
    return cnt % 2 == 0

import numpy as np


@vectorize(signature='(i),(i),(i)->(i)')
def clamp(self, min, max):
    # assumes min < max, componentwise

    return np.max(min, np.min(max, self))


class Vector2:
    def __init__(self, x=np.nan, y=np.nan):
        self._array = np.array([x, y])

    @property
    def x(self):
        return self._array[0]

    @x.setter
    def x(self, v):
        self._array[0] = v

    @property
    def y(self):
        return self._array[1]

    @y.setter
    def y(self, v):
        self._array[1] = v

    def __iter__(self):
        return iter(self._array)

    def copy(self):
        return Vector2(*self._array)




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


class BBox2:
    def __init__(self, pts=None):
        self.min = Vector2()
        self.max = Vector2()
        if pts is not None:
            self.set_from_pts(pts)

    def contains(self, x, y):
        return self.min.x <= x <= self.max.x and self.min.y <= y <= self.max.y

    def expand_by_point(self, pt):
        self.set_from_pts(np.array([*zip(self.min, self.max), pt]))

    def set_from_pts(self, pts: np.ndarray):
        (self.min.x, self.min.y), (self.max.x, self.max.y) = aabb(pts)

    def __repr__(self):
        return f'<{self.__class__.__name__}(min={self.min._array},max={self.max._array}) object at {hex(id(self))}>'

    def copy(self):
        b = BBox2()
        b.min = self.min.copy()
        b.max = self.max.copy()
        return b


class IntervalNode:
    def __init__(self):
        self.bbox = BBox2()
        self.left = None
        self.right = None
        self.node_edges = []


class IntervalTree:
    def __init__(self, pts, edges, bbox):
        self.pts = pts
        self.edges = edges
        self.bbox = bbox
        self.pipResult = False
        self.root = None

    def splitNode(self, node):
        if node.bbox.min.y >= node.bbox.max.y:
            return

        if len(node.node_edges) < 3:
            return

        split = 0.5 * (node.bbox.min.y + node.bbox.max.y)

        node.left = IntervalNode()
        node.right = IntervalNode()

        remaining_node_edges = []
        tmpPt = Vector2()

        for i in range(len(node.node_edges)):
            e = self.edges[node.node_edges[i]]

            p1y = self.pts[e[0]][1]
            p2y = self.pts[e[1]][1]

            if p1y > p2y:
                p1y, p2y = p2y, p1y

            boxPtr = None

            if p2y < split:
                node.left.node_edges.append(node.node_edges[i])
                boxPtr = node.left.bbox
            elif p1y > split:
                node.right.node_edges.append(node.node_edges[i])
                boxPtr = node.right.bbox
            else:
                remaining_node_edges.append(node.node_edges[i])

            if boxPtr is not None:
                boxPtr.expand_by_point(self.pts[e[0]])
                boxPtr.expand_by_point(self.pts[e[1]])

        node.node_edges = remaining_node_edges

        if len(node.left.node_edges):
            self.splitNode(node.left)
        if len(node.right.node_edges):
            self.splitNode(node.right)

    def build(self):
        self.root = IntervalNode()
        self.root.node_edges = list(range(len(self.edges)))
        self.root.bbox = self.bbox.copy()

        self.splitNode(self.root)

    def pointInPolygonRec(self, node, x, y):
        # node.bbox.min.y <= y & & node.bbox.max.y
        if node.bbox.min.y <= y and node.bbox.max.y >= y:
            for i in range(len(node.node_edges)):
                e = self.edges[node.node_edges[i]]

                p1 = self.pts[e[0]]
                yflag0 = (p1[1] >= y)

                p2 = self.pts[e[1]]
                yflag1 = (p2[1] >= y)

                if yflag0 != yflag1:
                    if (((p2[1] - y) * (p1[0] - p2[0]) >= (p2[0] - x) * (p1[1] - p2[1])) == yflag1):
                        self.pipResult = not self.pipResult
        # (nl && nl.bbox.min.y <= y && nl.bbox.max.y >= y
        if node.left and node.left.bbox.min.y <= y and node.left.bbox.max.y >= y:
            self.pointInPolygonRec(node.left, x, y)

        if node.right and node.right.bbox.min.y <= y and node.right.bbox.max.y >= y:
            self.pointInPolygonRec(node.right, x, y)

    @vectorize(excluded=[0], signature='(i)->()')
    def pointInPolygon(self, pt):
        self.pipResult = False
        self.pointInPolygonRec(self.root, pt[0], pt[1])
        return self.pipResult

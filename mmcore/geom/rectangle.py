import functools

import numpy as np
import shapely
from multipledispatch import dispatch

from mmcore.func import vectorize
from mmcore.geom import offset
from mmcore.geom import plane
from mmcore.geom.polyline import evaluate_polyline, polyline_to_lines
from mmcore.geom.shapes.area import polygon_area
from mmcore.geom.vec import cross, norm, unit


class Rectangle(plane.Plane):

    def __init__(self, u: float = 1, v: float = 1, xaxis=np.array((1, 0, 0)), normal=np.array((0, 0, 1)),
                 origin=np.array((0, 0, 0))):
        super().__init__()
        normal = unit(normal)
        xaxis = unit(xaxis)

        self.origin = origin
        yaxis = cross(normal, xaxis)

        self.xaxis = xaxis * u
        self.yaxis = yaxis * v
        self.zaxis = normal
        self._dirty = True
        self._solve_dirty()

    @classmethod
    def from_corners(cls, points):
        a, b, c, d = points
        return Rectangle(norm(a - b), norm(a - d), xaxis=unit(b - a), normal=unit(cross(unit(b - a), unit(c - a))),
                         origin=a)

    @property
    def corners(self):
        return np.array(
            [self.origin, self.origin + self.xaxis, self.origin + self.xaxis + self.yaxis, self.origin + self.yaxis])

    @property
    def segments(self):
        return polyline_to_lines(self.corners)

    @dispatch(np.ndarray, np.ndarray)
    def evaluate(self, u, v):
        return _evaluate_rect(self, u, v)

    @dispatch(float, float)
    def evaluate(self, u, v):
        return _evaluate_rect(self, u, v)

    @dispatch(int, float)
    def evaluate(self, u, v):
        return _evaluate_rect(self, u, v)

    @dispatch(int, int)
    def evaluate(self, u, v):
        return _evaluate_rect(self, u, v)

    @dispatch(float, int)
    def evaluate(self, u, v):
        return _evaluate_rect(self, u, v)

    @dispatch(float)
    def evaluate(self, t):
        return evaluate_polyline(self.corners, t)

    @dispatch(np.ndarray)
    def evaluate(self, t):
        return evaluate_polyline(self.corners, t)

    def __call__(self, *args):
        return self.evaluate(*args)

    @property
    def area(self):
        return polygon_area(np.append(self.corners, self.corners[0]).reshape((5, 3)))

    @dispatch(float)
    def offset(self, dist: float):
        return self.__class__.from_corners(offset.offset(self.corners, np.ones((2, 4), dtype=float) * dist))

    @dispatch(np.ndarray)
    def offset(self, dist: np.ndarray):
        return self.__class__.from_corners(offset.offset(self.corners, np.append(dist, dist).reshape((2, len(dist)))))

    def rotate(self, angle, axis=None):
        plane.rotate_plane_inplace(self, angle, axis if axis is not None else self.zaxis, origin=(0, 0, 0))

    def translate(self, translation):
        plane.translate_plane_inplace(self, translation)

    @dispatch(float)
    def inplace_offset(self, dist: float):
        points = offset.offset(self.corners, np.ones((2, 4), dtype=float) * dist)
        a, b, c, d = np.array(points)
        u, v = norm(a - b), norm(a - d)
        xaxis, normal = unit(b - a), unit(cross(unit(b - a), unit(c - a)))
        origin = a

        xaxis = unit(xaxis)

        self.origin = origin
        yaxis = cross(normal, xaxis)

        self.xaxis = xaxis * u
        self.yaxis = yaxis * v
        self.zaxis = normal
        self.origin = a
        self.xaxis = b - a
        self.yaxis = d - a

    @dispatch(np.ndarray)
    def inplace_offset(self, dist):
        points = offset.offset(self.corners, dist)
        a, b, c, d = np.array(points)
        u, v = norm(a - b), norm(a - d)
        xaxis, normal = unit(b - a), unit(cross(unit(b - a), unit(c - a)))
        origin = a

        xaxis = unit(xaxis)

        self.origin = origin
        yaxis = cross(normal, xaxis)

        self.xaxis = xaxis * u
        self.yaxis = yaxis * v
        self.zaxis = normal
        self.origin = a
        self.xaxis = b - a
        self.yaxis = d - a

    def __repr__(self):
        return f'{self.__class__.__name__}({self.corners})'


@vectorize(signature='(j,i),()->()')
def create_line_extrusion(lnn, h):
    start, end = lnn

    return Rectangle.from_corners(np.array((start, start + np.array([0, 0, h]), end + np.array([0, 0, h]), end)))


def create_rect_extrusion(lnn, h, color=(0.3, 0.3, 0.3), props=dict()):
    m = mesh.union_mesh2(
        [mesh_from_bounds(lnn), *(to_mesh(r) for r in create_line_extrusion(polyline_to_lines(lnn), h)),
         mesh_from_bounds(lnn + np.array([0, 0, h]))], props
    )
    mesh.colorize_mesh(m, color)
    return m


from mmcore.geom import mesh
from mmcore.geom.mesh.shape_mesh import mesh_from_bounds


@vectorize(excluded=[0], signature='(),()->(i)')
def _evaluate_rect(rect, u, v):
    x, y = rect.xaxis * u, rect.yaxis * v
    return rect.origin + x + y


class RectangleUnion:
    def __init__(self, *rects):
        self._rects = tuple(rects)

    def __hash__(self):
        return hash(self._rects)

    @property
    def poly(self):
        return cached_poly(tuple(self._rects))

    @property
    def corners(self):
        return shapely.geometry.mapping(self.poly)['coordinates'][0]

    def to_shape(self):
        return self.corners

    def to_mesh(self, uuid, **kwargs):
        return mesh.build_mesh_with_buffer(to_mesh(self), uuid=uuid, **kwargs)


@dispatch(RectangleUnion, tuple, dict)
def to_mesh(obj: RectangleUnion, color=(0.3, 0.3, 0.3), props: dict = None):
    return mesh_from_bounds(obj.corners, color, props)


@dispatch(RectangleUnion, tuple)
def to_mesh(obj: RectangleUnion, color=(0.3, 0.3, 0.3), props: dict = None):
    return mesh_from_bounds(obj.corners, color, props)


@dispatch(Rectangle, tuple, dict)
def to_mesh(obj: Rectangle, color=(0.3, 0.3, 0.3), props: dict = None):
    return mesh_from_bounds(obj.corners, color, props)


@dispatch(Rectangle, tuple)
def to_mesh(obj: Rectangle, color=(0.3, 0.3, 0.3), props=dict()):
    return mesh_from_bounds(obj.corners, color, props)


@dispatch(Rectangle)
def to_mesh(obj: Rectangle, color=(0.3, 0.3, 0.3), props=dict()):
    return mesh_from_bounds(obj.corners, color, props)


@dispatch(RectangleUnion)
def to_mesh(obj: RectangleUnion, color=(0.3, 0.3, 0.3), props=dict()):
    return mesh_from_bounds(obj.corners, color, props)


@functools.lru_cache()
def cached_poly(args):
    return shapely.union_all([shapely.Polygon(arg.corners) for arg in args])

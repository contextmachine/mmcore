import copy
import functools

import numpy as np
import shapely
from multipledispatch import dispatch
from scipy.spatial import ConvexHull

from mmcore.func import vectorize
from mmcore.geom import offset
from mmcore.geom import plane
from mmcore.geom.plane import WXY, rotate_plane_around_plane
from mmcore.geom.polyline import evaluate_polyline, polyline_to_lines
from mmcore.geom.shapes import reverse_vertices
from mmcore.geom.shapes.area import polygon_area
from mmcore.geom.vec import cross, dist, norm, unit, dot
from mmcore.geom.line import Line
from mmcore.geom.vec import angle as _angle




from mmcore.collections import DCLL
from mmcore.params import cast_to_parameter, Length

_RECT_INDICES = DCLL.from_list([0, 1, 2, 3])
from mmcore.geom.utils.num import roll_to_index, min_index


class Rectangle(plane.Plane):

    def __init__(self, u: float = 1, v: float = 1, xaxis=None, normal=np.array((0, 0, 1)), origin=np.array((0, 0, 0)),
                 uuid=None):
        if xaxis is None:
            super().__init__(xaxis=np.array([1.0, 0.0, 0.0]), normal=unit(normal), uuid=uuid)

            self.refine(('y', 'x'))
        else:
            normal = unit(normal)
            xaxis = unit(xaxis)
            yaxis = unit(cross(normal, xaxis))
            super().__init__(arr=np.array([origin, xaxis, yaxis, normal]), uuid=uuid)

        self._u = cast_to_parameter(u, Length)
        self._v = cast_to_parameter(v, Length)



    def __iter__(self):
        return iter(self.corners)

    @vectorize(excluded=[0], signature="(i)->()")
    def inside(self, pt: np.ndarray) -> bool:
        ox, oy, x, y = self.origin[0], self.origin[1], pt[0], pt[1]

        return all([ox <= x <= (ox + self.u), oy <= y <= (oy + self.v)])

    @property
    def u(self):
        return float(self._u.get())

    @u.setter
    def u(self, val):

        self._u.set(val)


    @property
    def v(self):
        return float(self._v.get())

    @v.setter
    def v(self, val):
        self._v.set(val)

    @classmethod
    def from_corners(cls, points):
        a, b, c, d = points
        return Rectangle(norm(a - b), norm(a - d), xaxis=unit(b - a), normal=unit(cross(unit(b - a), unit(c - a))),
                origin=a, )

    @property
    def corners(self):
        x = self.xaxis * self.u
        y = self.yaxis * self.v
        return np.array([self.origin, self.origin + x, self.origin + x + y, self.origin + y]
                )

    @property
    def segments(self):
        return polyline_to_lines(self.corners)

    @dispatch(object, object)
    def evaluate(self, u, v):
        return _evaluate_rect(self, u, v)

    @dispatch(object)
    def evaluate(self, t):
        return evaluate_polyline(self.corners, t)


    def __call__(self, *args):
        return self.evaluate(*args)

    @property
    def area(self):
        return polygon_area(np.append(self.corners, self.corners[0]).reshape((5, 3)))

    @dispatch(float)
    def offset(self, dist: float):
        return self.__class__.from_corners(offset.offset(self.corners, np.ones((2, 4), dtype=float) * dist)
                )

    @dispatch(np.ndarray)
    def offset(self, dist: np.ndarray):
        return self.__class__.from_corners(offset.offset(self.corners, np.append(dist, dist).reshape((2, len(dist))))
                )

    def rotate(self, angle, axis=None, origin=None, inplace=True):
        if origin is None:
            origin = self.origin
        if inplace:
            plane.rotate_plane_inplace(self, angle, np.array([0, 0, 1]), origin=origin)
        else:
            pl = plane.rotate_plane(self, angle, np.array([0, 0, 1]), origin=origin)
            print(pl._arr, angle, axis, origin, inplace)
            return Rectangle(self.u, self.v, xaxis=unit(pl.xaxis), normal=unit(pl.normal), origin=pl.origin, )

    def translate(self, translation, inplace=True):
        if inplace:
            plane.translate_plane_inplace(self, translation)
        else:
            pl = plane.translate_plane(self, translation)
            return Rectangle(self.u, self.v, xaxis=unit(pl.xaxis), normal=unit(pl.normal), origin=pl.origin, )

    def to_mesh(self, uuid=None, **kwargs):
        if uuid is None:
            uuid = self.uuid
        return mesh.build_mesh_with_buffer(to_mesh(self), uuid=uuid, **kwargs)

    def inplace_offset(self, dist):
        if np.isscalar(dist):
            dist = np.ones((2, 4), dtype=float) * dist
        points = offset.offset(self.corners, np.array([dist, dist]))
        a, b, c, d = np.array(points)
        u, v = norm(a - b), norm(a - d)
        xaxis, normal = unit(b - a), unit(cross(unit(b - a), unit(c - a)))

        self.origin = a
        yaxis = cross(normal, xaxis)
        self.u = u
        self.v = v
        self.xaxis = xaxis
        self.yaxis = yaxis
        self.zaxis = normal

    def __repr__(self):
        return f"{self.__class__.__name__}({self.corners})"

    def todict(self):
        return dict(u=self.u, v=self.v) | super().todict()

    def side_lengths(self):
        segms = polyline_to_lines(self.corners)
        return dist(segms[:, 0, :], segms[:, 1, :])

    def orient(self, pln, inplace=True):
        if not inplace:
            other = copy.deepcopy(self)
            return rect_to_plane(other, pln)
        else:
            rect_to_plane(self, pln)

    def rotate_in_plane(self, angle, pln=WXY, inplace=True):
        # pln=rotate_plane_around_plane(self, plane, angle)
        # self.origin=pln.origin
        # self.xaxis=pln.yaxis
        # self.yaxis = pln.xaxis
        # self.zaxis = pln.zaxis

        return self.orient(rotate_plane_around_plane(self, pln, angle), inplace=inplace)

    def intersects(self, other):
        if isinstance(other, np.ndarray):
            return np.any(self.inside(other))
        elif isinstance(other, Line):
            return self.intersects(np.array([other.start, other.end]))
        elif isinstance(other, Rectangle):
            return self.intersects(np.array(other.corners))

    def intersects_indices(self, other):
        if isinstance(other, np.ndarray):
            return np.arange(len(other))[self.inside(other)]
        elif isinstance(other, Line):
            return self.intersects_indices(np.array([other.start, other.end]))
        elif isinstance(other, Rectangle):
            return self.intersects_indices(np.array(other.corners))

    def intersection(self, other: "Rectangle"):
        global _RECT_INDICES
        ixs1 = []
        ixs2 = []
        pts = []
        for i, l1 in enumerate(polyline_to_lines(self.corners)):
            for j, l2 in enumerate(polyline_to_lines(other.corners)):
                il, jl = Line.from_ends(*l1), Line.from_ends(*l2)
                if np.abs(dot(il.unit, jl.unit)) < 1.0:
                    t1, t2, _ = il.bounded_intersect(jl)

                    if all([0.0 <= t1 <= 1.0, 0.0 <= (-t2) <= 1.0]):
                        ixs1.append(i)
                        ixs2.append(j)
                        pts.append(il(t1))

        print([pts[0], ixs1[0], ixs1[-1], ixs2[0], ixs2[-1], pts[1]])

        return np.array([*other.corners[np.arange(ixs2[0] + 1, ixs2[-1] - ixs2[0], 1, dtype=int)], *pts,
            *self.corners[np.arange(ixs1[0] + 1, ixs1[-1] - ixs1[0], 1, dtype=int)], ]
        )

    @classmethod
    def from_mbr(cls, points: "np.ndarray | list | tuple", closest_origin=None):
        r = mbr(points, ctor=cls.from_corners)
        if closest_origin is not None:
            return make_rect_origin_closest(r, closest_origin)
        return r


@vectorize(signature="(j,i),()->()")
def create_line_extrusion(lnn, h):
    start, end = lnn

    return Rectangle.from_corners(np.array((start, start + np.array([0, 0, h]), end + np.array([0, 0, h]), end))
            )


def create_rect_extrusion(lnn, h, color=(0.3, 0.3, 0.3), props=dict()):
    m = mesh.union_mesh([mesh_from_bounds(lnn), *(to_mesh(r) for r in create_line_extrusion(polyline_to_lines(lnn), h)),
        mesh_from_bounds(lnn + np.array([0, 0, h])), ], props,
    )
    mesh.colorize_mesh(m, color)
    return m


from mmcore.geom import mesh
from mmcore.geom.mesh.shape_mesh import mesh_from_bounds


@vectorize(excluded=[0], signature="(),()->(i)")
def _evaluate_rect(rect, u, v):
    x, y = rect.xaxis * u, rect.yaxis * v
    return rect.origin + x + y


@dispatch(Rectangle, tuple, dict)
def to_mesh(obj: Rectangle, color=(0.3, 0.3, 0.3), props: dict = None):
    return mesh_from_bounds(obj.corners, color, props)


@dispatch(Rectangle, tuple)
def to_mesh(obj: Rectangle, color=(0.3, 0.3, 0.3), props=dict()):
    return mesh_from_bounds(obj.corners, color, props)


@dispatch(Rectangle)
def to_mesh(obj: Rectangle, color=(0.3, 0.3, 0.3), props=dict()):
    return mesh_from_bounds(obj.corners.tolist(), color, props)


@functools.lru_cache()
def cached_poly(args):
    return shapely.union_all([shapely.Polygon(arg.corners) for arg in args])


@functools.lru_cache()
def cached_poly_full(unions, diffs):
    unions, diffs = [shapely.union_all([shapely.Polygon(arg.corners) for arg in args]) for args in [unions, diffs]]
    return shapely.difference(unions, diffs)


def process_property(prop):
    fget = lambda self: [prop.object.fget(o) for o in self._items]

    def fset(self, v):
        for o, vv in zip(self._items, v):
            prop.object.fset(o, vv)

    def fdel(self):
        for o in self._items:
            prop.object.fdel(o)

    return property(fget, fset, fdel)


from functools import wraps


def process_method(method):
    @wraps(method.object)
    def fun(self, *args, **kwargs):
        return [method.object(o, *args, **kwargs) for o in self._items]

    return fun


import inspect


def composite(item_cls):
    def init(self, items):
        self._items = list(items)

    def getitem(self, item):
        return self._items[item]

    def setitem(self, item, val):
        self._items[item] = val

    def _iter(self):
        return iter(self._items)

    def _len(self):
        return len(self._items)

    def _repr(self):
        return f"{self.__class__.__name__}({', '.join(repr(item) for item in self._items)})"

    item_attrs = inspect.classify_class_attrs(item_cls)
    attrs = {
        "__init__": init, "__getitem__": getitem, "__setitem__": setitem, "__iter__": _iter, "__len__": _len,
        "__repr__": _repr, "__contains__": lambda self, other: other in self._items,
        }
    for attr in item_attrs:
        if not attr.name.startswith("_"):
            if attr.kind == "property":
                attrs[attr.name] = process_property(attr)
            elif attr.kind == "method":
                attrs[attr.name] = process_method(attr)

    return type(f"{item_cls.__name__}Composite", (object,), attrs)


RectangleComposite = composite(Rectangle)


def rect_to_plane(rect: Rectangle, new_plane: plane.Plane):
    rect.ecs_plane = copy.deepcopy(new_plane.ecs_plane)


class RectangleUnion(RectangleComposite):
    def __hash__(self):
        return hash(tuple(self._items))

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, v):
        max_angle = self.max_angle()

        self.rotate_item(1, angle=max_angle if max_angle < v else v, origin=self._items[0].origin,
                axis=self._items[0].normal, )

    @property
    def poly(self):
        return cached_poly(tuple(self._items))

    def rotate_item(self, item: int, angle: float, axis=None, origin: tuple = None):
        self._items[item].rotate(angle, axis=axis, origin=origin)

    def translate_item(self, item, vector=(0.0, 0.0, 0.0)):
        self._items[item].translate(vector)

    @property
    def corners(self):
        return shapely.geometry.mapping(self.poly)["coordinates"][0][:-1]

    def to_shape(self):
        return self.corners

    def to_mesh(self, uuid=None, **kwargs):
        return mesh.build_mesh_with_buffer(to_mesh(self), uuid=uuid, **kwargs)

    @property
    def area(self):
        return self.poly.area

    def __iter__(self):
        return iter(shapely.geometry.mapping(self.poly)["coordinates"][0][:-1])

    def max_angle(self):
        a, c = sorted([self._items[0].v, self._items[1].v])
        b = np.sqrt(c ** 2 - a ** 2)
        return _angle(unit([a, 0]), unit([b, a])) + np.pi / 2


@dispatch(Rectangle, object, object)
def to_mesh(obj: Rectangle, color=(0.3, 0.3, 0.3), props: dict = None):
    return mesh_from_bounds(obj.corners.tolist(), color, props)


@dispatch(RectangleUnion, tuple, dict)
def to_mesh(obj: RectangleUnion, color=(0.3, 0.3, 0.3), props: dict = None):
    return mesh_from_bounds(obj.corners, color, props)


@dispatch(RectangleUnion, tuple)
def to_mesh(obj: RectangleUnion, color=(0.3, 0.3, 0.3), props: dict = None):
    return mesh_from_bounds(obj.corners, color, props)


@dispatch(RectangleUnion)
def to_mesh(obj: RectangleUnion, color=(0.3, 0.3, 0.3), props=dict()):
    return mesh_from_bounds(obj.corners, color, props)


@vectorize(excluded=["color", "props"], signature="(j,i)->(i)")
def rect_to_mesh_vec(bounds: np.ndarray, color=(0.3, 0.3, 0.3), props=None):
    return np.array(mesh_from_bounds(bounds.tolist(), color=color, props=props), dtype=object
            )


class RectangleCollection(composite(Rectangle)):
    def translate(self, translation, inplace=True):
        res = super().translate(translation, inplace=inplace)
        if not inplace:
            return self.__class__(res)

    def rotate(self, angle, axis=(0, 0, 1), inplace=True):
        res = super().rotate(angle, axis=axis, inplace=inplace)
        if not inplace:
            return self.__class__(res)

    @property
    def corners(self):
        return super().corners

    @property
    def poly(self):
        return cached_poly(tuple(self._items))

    @property
    def corners_poly(self):
        return shapely.geometry.mapping(self.poly)["coordinates"][0]


def minimum_bounding_rectangle(points):
    """
    Find the smallest bounding rectangle for a set of points.
    Returns a set of points representing the corners of the bounding box.

    :param points: an nx2 matrix of coordinates
    :rval: an nx2 matrix of coordinates
    """

    pi2 = np.pi / 2.0

    # get the convex hull for the points
    hull_points = points[ConvexHull(points).vertices]

    # calculate edge angles
    edges = np.zeros((len(hull_points) - 1, 2))
    edges = hull_points[1:] - hull_points[:-1]

    angles = np.zeros((len(edges)))
    angles = np.arctan2(edges[:, 1], edges[:, 0])

    angles = np.abs(np.mod(angles, pi2))
    angles = np.unique(angles)

    # find rotation matrices
    # XXX both work
    rotations = np.vstack([np.cos(angles), np.cos(angles - pi2), np.cos(angles + pi2), np.cos(angles)]
            ).T
    #     rotations = np.vstack([
    #         np.cos(angles),
    #         -np.sin(angles),
    #         np.sin(angles),
    #         np.cos(angles)]).T
    rotations = rotations.reshape((-1, 2, 2))

    # apply rotations to the hull
    rot_points = np.dot(rotations, hull_points.T)

    # find the bounding points
    min_x = np.nanmin(rot_points[:, 0], axis=1)
    max_x = np.nanmax(rot_points[:, 0], axis=1)
    min_y = np.nanmin(rot_points[:, 1], axis=1)
    max_y = np.nanmax(rot_points[:, 1], axis=1)

    # find the box with the best area
    areas = (max_x - min_x) * (max_y - min_y)
    best_idx = np.argmin(areas)

    # return the best box
    x1 = max_x[best_idx]
    x2 = min_x[best_idx]
    y1 = max_y[best_idx]
    y2 = min_y[best_idx]
    r = rotations[best_idx]

    rval = np.zeros((4, 2))
    rval[0] = np.dot([x1, y2], r)
    rval[1] = np.dot([x2, y2], r)
    rval[2] = np.dot([x2, y1], r)
    rval[3] = np.dot([x1, y1], r)

    return rval


def make_rect_origin_closest(rect, point=np.array([0.0, 0.0, 0.0])):
    d = dist(np.array(point), rect.corners)
    m = min_index(d)

    return rect.__class__.from_corners(roll_to_index(rect.corners, m))


def mbr(pts, ctor=Rectangle.from_corners):
    z = np.zeros((4, 3), pts.dtype)
    z[..., :2] = minimum_bounding_rectangle(pts[..., :2])
    v = np.array(reverse_vertices(z.tolist()))
    return ctor(v)

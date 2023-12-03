import dataclasses
import functools
import itertools

import numpy as np
import shapely
from multipledispatch import dispatch
from mmcore.geom.mesh import union_mesh2
from mmcore.base.ecs.components import component
from mmcore.func import vectorize
from mmcore.geom import offset
from mmcore.geom import plane
from mmcore.geom.plane import PlaneComponent
from mmcore.geom.polyline import evaluate_polyline, polyline_to_lines
from mmcore.geom.shapes.area import polygon_area
from mmcore.geom.vec import cross, norm, unit


@component()
class Length:
    value: float = None


@component()
class UV:
    u: Length = None
    v: Length = None


@component()
class RectangleComponent:
    plane: PlaneComponent = None
    uv: UV = None


@component()
class RectangleUnionComponent:
    planes: [PlaneComponent] = None
    uvs: list[UV] = None


@dataclasses.dataclass(unsafe_hash=True)
class RectangleParmAccess(plane.PlaneParamsAccess):
    uv: UV = None

    def __post_init__(self):
        plane.PlaneParamsAccess.__post_init__(self)

        if not hasattr(self.uv, 'component_type'):
            self.uv = UV(*[Length(i) if not hasattr(i, 'component_type') else i for i in self.uv])

    @property
    def u(self):
        return self.uv.u.value

    @property
    def v(self):
        return self.uv.u.value

    @u.setter
    def u(self, val):
        self.uv.u.value = val

    @v.setter
    def v(self, val):
        self.uv.v.value = val

    @classmethod
    def from_plane_pa(self, u, v, pa: plane.PlaneParamsAccess):
        return RectangleParmAccess(arr=pa.arr, uv=(u, v))


from mmcore.base.ecs.components import EcsProperty

class Rectangle(plane.Plane):
    ecs_uv = EcsProperty()
    ecs_rectangle = EcsProperty()

    def __init__(self, u: 'float|Length' = 1, v: 'float|Length' = 1, xaxis=np.array((1, 0, 0)),
                 normal=np.array((0, 0, 1)),
                 origin=np.array((0, 0, 0))):
        super(plane.Plane, self).__init__()
        normal = unit(normal)
        xaxis = unit(xaxis)
        yaxis = unit(cross(normal, xaxis))
        super().__init__(np.array([origin, xaxis, yaxis, normal]))
        if not hasattr(u, 'component_type'):
            u = Length(u)
        if not hasattr(v, 'component_type'):
            v = Length(v)

        self.ecs_uv = UV(u, v)
        self.ecs_rectangle = RectangleComponent(plane=self._arr_cmp, uv=self.ecs_uv)

    def __iter__(self):
        return iter(self.corners)

    @property
    def u(self):
        return self.ecs_uv.u.value

    @u.setter
    def u(self, val):
        if not hasattr(val, 'component_type'):

            self.ecs_uv.u.value = val
        else:
            self.ecs_uv.u = val
        self._dirty = True

    @property
    def v(self):
        return self.ecs_uv.v.value

    @v.setter
    def v(self, val):
        if not hasattr(val, 'component_type'):

            self.ecs_uv.v.value = val
        else:
            self.ecs_uv.v = val
        self._dirty = True


    @classmethod
    def from_corners(cls, points):
        a, b, c, d = points
        return Rectangle(norm(a - b), norm(a - d), xaxis=unit(b - a), normal=unit(cross(unit(b - a), unit(c - a))),
                         origin=a)

    @property
    def corners(self):
        x = self.xaxis * self.u
        y = self.yaxis * self.v
        return np.array(
            [self.origin, self.origin + x, self.origin + x + y, self.origin + y])

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

    def rotate(self, angle, axis=None, origin=None, inplace=True):
        if origin is None:
            origin = self.origin
        if inplace:
            plane.rotate_plane_inplace(self, angle, np.array([0, 0, 1]), origin=origin)
        else:
            pl = plane.rotate_plane(self, angle, np.array([0, 0, 1]), origin=origin)
            print(pl._arr, angle, axis, origin, inplace)
            return Rectangle(self.u, self.v, xaxis=unit(pl.xaxis), normal=unit(pl.normal), origin=pl.origin)

    def translate(self, translation, inplace=True):
        if inplace:
            plane.translate_plane_inplace(self, translation)
        else:
            pl = plane.translate_plane(self, translation)
            return Rectangle(self.u, self.v, xaxis=unit(pl.xaxis), normal=unit(pl.normal), origin=pl.origin)



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
        return f'{self.__class__.__name__}({self.corners})'

    def todict(self):
        return dict(u=self.u, v=self.v) | super().todict()


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


class ShapeNode:
    def __init__(self, unions=(), diffs=()):
        super().__init__()
        self.unions = list(set(unions))
        self.diffs = list(set(diffs))
        self._dirty = True
        self._hss = None
        self.solve_hash()

    def solve_hash(self):
        if self._dirty:
            self._hss = tuple(hash(u) for u in self.unions), tuple(hash(u) for u in self.diffs)
        self._dirty = False

    def __hash__(self):

        return hash(self.get_hashes())

    def get_hashes(self):
        if self._dirty:
            self.solve_hash()
        return tuple(self._hss)

    def compare(self, unions, diffs):
        u, d = self.get_hashes()
        u, d = np.array(u, int), np.array(d, int)
        u1, d1 = unions, diffs
        u1, d1 = np.array(u1, int), np.array(d1, int)
        return np.in1d(u1, u), np.in1d(d1, d)

    @property
    def area(self):
        return self.poly.area

    def merge(self, u, d):

        u1, d1 = self.compare([hash(i) for i in u], [hash(i) for i in d])
        return ShapeNode(np.append(self.unions, np.array(u, dtype=object)[not u1]),
                         np.append(self.diffs, np.array(d, dtype=object)[not d1]))

    def __add__(self, other):

        return self.merge(other.unions, other.diffs)

    def __sub__(self, other):
        return self.merge(other.diffs, other.unions)

    @property
    def corners(self):
        return shapely.geometry.mapping(self.poly)['coordinates']

    @property
    def poly(self):
        return cached_poly_full(tuple(self.unions), tuple(self.diffs))

    @property
    def union_poly(self):
        return cached_poly(tuple(self.unions))

    @property
    def diffs_poly(self):
        return cached_poly(tuple(self.diffs))

    def rotate(self, angle, axis=None, origin=None):
        ...


@dispatch(Rectangle, tuple, dict)
def to_mesh(obj: Rectangle, color=(0.3, 0.3, 0.3), props: dict = None):
    return mesh_from_bounds(obj.corners, color, props)


@dispatch(Rectangle, tuple)
def to_mesh(obj: Rectangle, color=(0.3, 0.3, 0.3), props=dict()):
    return mesh_from_bounds(obj.corners, color, props)


@dispatch(Rectangle)
def to_mesh(obj: Rectangle, color=(0.3, 0.3, 0.3), props=dict()):
    return mesh_from_bounds(obj.corners, color, props)


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
    attrs = {'__init__': init,
             '__getitem__': getitem,
             '__setitem__': setitem,
             '__iter__': _iter,
             "__len__": _len,
             "__repr__": _repr,
             '__contains__': lambda self, other: other in self._items}
    for attr in item_attrs:
        if not attr.name.startswith('_'):
            if attr.kind == 'property':
                attrs[attr.name] = process_property(attr)
            elif attr.kind == 'method':
                attrs[attr.name] = process_method(attr)

    return type(f'{item_cls.__name__}Composite', (object,), attrs)


RectangleComposite = composite(Rectangle)


def rect_to_plane(rect: Rectangle, new_plane: plane.Plane):
    rect.ecs_plane = new_plane.ecs_plane

class RectangleUnion(RectangleComposite):


    def __hash__(self):
        return hash(tuple(self._items))

    @property
    def poly(self):
        return cached_poly(tuple(self._items))

    def rotate_item(self, item: int, angle: float, axis=None, origin: tuple = None):
        self._items[item].rotate(angle, axis=axis, origin=origin)

    def translate_item(self, item, vector=(0.0, 0.0, 0.0)):
        self._items[item].translate(vector)
    @property
    def corners(self):
        return shapely.geometry.mapping(self.poly)['coordinates'][0][:-1]

    def to_shape(self):
        return self.corners

    def to_mesh(self, uuid, **kwargs):
        return mesh.build_mesh_with_buffer(to_mesh(self), uuid=uuid, **kwargs)

    @property
    def area(self):
        return self.poly.area

    def __iter__(self):
        return iter(shapely.geometry.mapping(self.poly)['coordinates'][0][:-1])

@dispatch(RectangleUnion, tuple, dict)
def to_mesh(obj: RectangleUnion, color=(0.3, 0.3, 0.3), props: dict = None):
    return mesh_from_bounds(obj.corners, color, props)


@dispatch(RectangleUnion, tuple)
def to_mesh(obj: RectangleUnion, color=(0.3, 0.3, 0.3), props: dict = None):
    return mesh_from_bounds(obj.corners, color, props)

@dispatch(RectangleUnion)
def to_mesh(obj: RectangleUnion, color=(0.3, 0.3, 0.3), props=dict()):
    return mesh_from_bounds(obj.corners, color, props)


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
        return shapely.geometry.mapping(self.poly)['coordinates'][0]





@dispatch(ShapeNode, float, int, int)
def rotate(node: ShapeNode, angle: float, origin_index: int):
    origin_node = (node.unions + node.diffs)[origin_index]
    origin, axis = origin_node.origin, origin_node.normal
    plane.r

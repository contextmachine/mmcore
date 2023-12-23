"""Grasshopper Script"""
import abc
import base64
import operator
import warnings
from functools import reduce
from uuid import uuid4
import struct
import math
import shapely
from shapely.geometry import mapping

from mmcore.geom.parametric import PlaneLinear, algorithms
from mmcore.geom.shapes import area3d

a = "Hello Python 3 in Grasshopper!"
print(a)

from dataclasses import dataclass
import typing
from collections import namedtuple

from more_itertools import chunked

_components = dict(ixs=dict(count=3, size=1, dtype='h', byteOffset=0, byteLength=6),
                   pos=dict(count=3, size=3, dtype='f', byteOffset=8, byteLength=36)
                   )

GLTF_TYPES = dict()


@dataclass
class GLTFType:
    name: str
    format: str
    size: int

    def __post_init__(self):
        GLTF_TYPES[self.name] = self


VEC2 = GLTFType('VEC2', format='f', size=2)
VEC3 = GLTFType('VEC3', format='f', size=3)
SCALAR = GLTFType("SCALAR", format="h", size=1)

v2 = dict()


def gltfff(buff='AAABAAIAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAACAPwAAAAA=', components: dict = None):
    bts = base64.b64decode(buff)
    for k, v in components.items():
        dtype = GLTF_TYPES[v['type']]
        *res, = chunked(struct.unpack(f"{v['count'] * dtype.size}{dtype.format}",
                                      bts[v["byteOffset"]:v["byteOffset"] + v["byteLength"]]
                                      ), dtype.size
                )
        yield k, res


uri = ('3VqrvyZ55D4meeS+Jnnkvt1aqz8meeS+Gb9avxm/Wj8Zv1q/3VqrvyZ55D4meeS+Gb9avxm/Wj8Zv1o/Jnnkvt1aqz8meeS'
       '+3VqrvyZ55L4meeQ+Gb9avxm/Wj8Zv1o/3VqrvyZ55D4meeS+3VqrvyZ55L4meeQ+JnnkviZ55L7dWqs/Gb9avxm/Wj8Zv1o/Gb9avxm'
       '/Wr8Zv1o/JnnkviZ55L7dWqs/3VqrvyZ55L4meeQ+Gb9avxm/Wj8Zv1o/JnnkPt1aqz8meeQ+Jnnkvt1aqz8meeS+Gb9avxm/Wj8Zv1o'
       '/JnnkPiZ55D7dWqs/JnnkPt1aqz8meeQ+JnnkviZ55L7dWqs/JnnkPiZ55D7dWqs/Gb9avxm/Wj8Zv1o/JnnkPiZ55D7dWqs/Gb9aPxm'
       '/Wj8Zv1o/JnnkPt1aqz8meeQ+3VqrPyZ55D4meeQ+JnnkPt1aqz8meeQ+Gb9aPxm/Wj8Zv1o/3VqrPyZ55D4meeQ+Gb9aPxm/Wj8Zv1q'
       '/JnnkPt1aqz8meeQ+3VqrPyZ55L4meeS+Gb9aPxm/Wj8Zv1q/3VqrPyZ55D4meeQ+3VqrPyZ55L4meeS+JnnkPiZ55L7dWqu/Gb9aPxm'
       '/Wj8Zv1q/Gb9aPxm/Wr8Zv1q/JnnkPiZ55L7dWqu/3VqrPyZ55L4meeS+Gb9aPxm/Wj8Zv1q/Jnnkvt1aqz8meeS+JnnkPt1aqz8meeQ'
       '+Gb9aPxm/Wj8Zv1q/JnnkviZ55D7dWqu/Jnnkvt1aqz8meeS+JnnkPiZ55L7dWqu/JnnkviZ55D7dWqu/Gb9aPxm/Wj8Zv1q'
       '/JnnkviZ55D7dWqu/Gb9avxm/Wj8Zv1q/Jnnkvt1aqz8meeS+Jnnkvt1aq78meeQ+JnnkviZ55L7dWqs/Gb9avxm/Wr8Zv1o'
       '/Jnnkvt1aq78meeQ+Gb9aPxm/Wr8Zv1o/JnnkviZ55L7dWqs/JnnkPt1aq78meeS+Gb9aPxm/Wr8Zv1o/Jnnkvt1aq78meeQ'
       '+JnnkPt1aq78meeS+3VqrPyZ55L4meeS+Gb9aPxm/Wr8Zv1o/Gb9aPxm/Wr8Zv1q/3VqrPyZ55L4meeS+JnnkPt1aq78meeS+Gb9aPxm'
       '/Wr8Zv1o/JnnkPiZ55D7dWqs/JnnkviZ55L7dWqs/Gb9aPxm/Wr8Zv1o/3VqrPyZ55D4meeQ+JnnkPiZ55D7dWqs/3VqrPyZ55L4meeS'
       '+3VqrPyZ55D4meeQ+Gb9aPxm/Wr8Zv1o/3VqrPyZ55D4meeQ+Gb9aPxm/Wj8Zv1o/JnnkPiZ55D7dWqs/JnnkviZ55D7dWqu'
       '/3VqrvyZ55D4meeS+Gb9avxm/Wj8Zv1q/JnnkviZ55D7dWqu/Gb9avxm/Wr8Zv1q/3VqrvyZ55D4meeS+JnnkPiZ55L7dWqu/Gb9avxm'
       '/Wr8Zv1q/JnnkviZ55D7dWqu/JnnkPiZ55L7dWqu/JnnkPt1aq78meeS+Gb9avxm/Wr8Zv1q/Gb9aPxm/Wr8Zv1q/JnnkPt1aq78meeS'
       '+JnnkPiZ55L7dWqu/Gb9avxm/Wr8Zv1q/3VqrvyZ55L4meeQ+3VqrvyZ55D4meeS+Gb9avxm/Wr8Zv1q/Jnnkvt1aq78meeQ'
       '+3VqrvyZ55L4meeQ+JnnkPt1aq78meeS+Jnnkvt1aq78meeQ+Gb9avxm/Wr8Zv1q/Jnnkvt1aq78meeQ+Gb9avxm/Wr8Zv1o'
       '/3VqrvyZ55L4meeQ+i49nv7Jfmj6yX5q+sl+avouPZz+yX5q+Os0TvzrNEz86zRO/i49nv7Jfmj6yX5q+Os0TvzrNEz86zRM/sl+avouPZz'
       '+yX5q+i49nv7Jfmr6yX5o+Os0TvzrNEz86zRM/i49nv7Jfmj6yX5q+i49nv7Jfmr6yX5o+sl+avrJfmr6Lj2c/Os0TvzrNEz86zRM'
       '/Os0TvzrNE786zRM/sl+avrJfmr6Lj2c/i49nv7Jfmr6yX5o+Os0TvzrNEz86zRM/sl+aPouPZz+yX5o+sl+avouPZz+yX5q'
       '+Os0TvzrNEz86zRM/sl+aPrJfmj6Lj2c/sl+aPouPZz+yX5o+sl+avrJfmr6Lj2c/sl+aPrJfmj6Lj2c/Os0TvzrNEz86zRM/sl'
       '+aPrJfmj6Lj2c/Os0TPzrNEz86zRM/sl+aPouPZz+yX5o+i49nP7Jfmj6yX5o+sl+aPouPZz+yX5o+Os0TPzrNEz86zRM/i49nP7Jfmj6yX5o'
       '+Os0TPzrNEz86zRO/sl+aPouPZz+yX5o+i49nP7Jfmr6yX5q+Os0TPzrNEz86zRO/i49nP7Jfmj6yX5o+i49nP7Jfmr6yX5q+sl'
       '+aPrJfmr6Lj2e/Os0TPzrNEz86zRO/Os0TPzrNE786zRO/sl+aPrJfmr6Lj2e/i49nP7Jfmr6yX5q+Os0TPzrNEz86zRO/sl+avouPZz+yX5q'
       '+sl+aPouPZz+yX5o+Os0TPzrNEz86zRO/sl+avrJfmj6Lj2e/sl+avouPZz+yX5q+sl+aPrJfmr6Lj2e/sl+avrJfmj6Lj2e'
       '/Os0TPzrNEz86zRO/sl+avrJfmj6Lj2e/Os0TvzrNEz86zRO/sl+avouPZz+yX5q+sl+avouPZ7+yX5o+sl+avrJfmr6Lj2c'
       '/Os0TvzrNE786zRM/sl+avouPZ7+yX5o+Os0TPzrNE786zRM/sl+avrJfmr6Lj2c/sl+aPouPZ7+yX5q+Os0TPzrNE786zRM/sl+avouPZ7'
       '+yX5o+sl+aPouPZ7+yX5q+i49nP7Jfmr6yX5q+Os0TPzrNE786zRM/Os0TPzrNE786zRO/i49nP7Jfmr6yX5q+sl+aPouPZ7+yX5q'
       '+Os0TPzrNE786zRM/sl+aPrJfmj6Lj2c/sl+avrJfmr6Lj2c/Os0TPzrNE786zRM/i49nP7Jfmj6yX5o+sl+aPrJfmj6Lj2c'
       '/i49nP7Jfmr6yX5q+i49nP7Jfmj6yX5o+Os0TPzrNE786zRM/i49nP7Jfmj6yX5o+Os0TPzrNEz86zRM/sl+aPrJfmj6Lj2c/sl'
       '+avrJfmj6Lj2e/i49nv7Jfmj6yX5q+Os0TvzrNEz86zRO/sl+avrJfmj6Lj2e/Os0TvzrNE786zRO/i49nv7Jfmj6yX5q+sl+aPrJfmr6Lj2e'
       '/Os0TvzrNE786zRO/sl+avrJfmj6Lj2e/sl+aPrJfmr6Lj2e/sl+aPouPZ7+yX5q+Os0TvzrNE786zRO/Os0TPzrNE786zRO/sl+aPouPZ7'
       '+yX5q+sl+aPrJfmr6Lj2e/Os0TvzrNE786zRO/i49nv7Jfmr6yX5o+i49nv7Jfmj6yX5q+Os0TvzrNE786zRO/sl+avouPZ7+yX5o'
       '+i49nv7Jfmr6yX5o+sl+aPouPZ7+yX5q+sl+avouPZ7+yX5o+Os0TvzrNE786zRO/sl+avouPZ7+yX5o+Os0TvzrNE786zRM'
       '/i49nv7Jfmr6yX5o+CsjlPi31GD8AAMA+nRpcPwAAwD5fJzI/CsjlPi31GD8AACA/XycyPwAAwD6dGlw/+xsNP6UVzj4AACA'
       '/XycyPwrI5T4t9Rg/+xsNP6UVzj4F5DI/pRXOPgAAID9fJzI/AAAgP0Gxmz4F5DI/pRXOPvsbDT+lFc4+AAAgP18nMj8AAGA'
       '/nRpcPwAAwD6dGlw/AAAgP18nMj/7G00/LfUYPwAAYD+dGlw/BeQyP6UVzj77G00/LfUYPwAAID9fJzI/+xtNPy31GD8AAGA/XycyPwAAYD'
       '+dGlw/BeRyPy31GD8AAGA/nRpcPwAAYD9fJzI/BeRyPy31GD8AAAA+XycyPwAAYD+dGlw//Y2GP6UVzj4AAJA/XycyPwXkcj8t9Rg'
       '/rr9RPaUVzj4UkEs+pRXOPgAAAD5fJzI/AAAAPkGxmz4UkEs+pRXOPq6/UT2lFc4+AAAAPl8nMj8AAMA+nRpcPwAAYD+dGlw/AAAAPl8nMj'
       '/2N5o+LfUYPwAAwD6dGlw/FJBLPqUVzj72N5o+LfUYPwAAAD5fJzI/9jeaPi31GD8AAMA+XycyPwAAwD6dGlw/AAAgP4uVDz4F5DI'
       '/pRXOPgAAID9BsZs+AAAgP4uVDz4AAGA/QbGbPgXkMj+lFc4+AAAAPouVDz4AAGA/QbGbPgAAID+LlQ8'
       '+AAAAPouVDz6uv1E9pRXOPgAAYD9BsZs+AAAAPkGxmz6uv1E9pRXOPgAAAD6LlQ8+AABgP0Gxmz77G00/LfUYPwXkMj+lFc4'
       '+AABgP0Gxmz4F5HI/LfUYP/sbTT8t9Rg//Y2GP6UVzj4F5HI/LfUYPwAAYD9BsZs+BeRyPy31GD8AAGA/XycyP/sbTT8t9Rg'
       '/9jeaPi31GD8KyOU+LfUYPwAAwD5fJzI/9jeaPi31GD8AAMA+QbGbPgrI5T4t9Rg/FJBLPqUVzj4AAMA+QbGbPvY3mj4t9Rg'
       '/FJBLPqUVzj4AAAA+i5UPPgAAwD5BsZs+AAAAPkGxmz4AAAA+i5UPPhSQSz6lFc4+AADAPkGxmz77Gw0/pRXOPgrI5T4t9Rg'
       '/AADAPkGxmz4AACA/i5UPPvsbDT+lFc4+AAAAPouVDz4AACA/i5UPPgAAwD5BsZs+AAAgP4uVDz4AACA/QbGbPvsbDT+lFc4+')

data1 = dict(gltfff('AAABAAIAAAAAAAAAAAAAAAAAAAAAAIA/AAAAAAAAAAAAAAAAAACAPwAAAAA=', components={
    "INDEX": {
        "byteOffset": 0, "byteLength": 6, "count": 3, "type": "SCALAR"
        }, "POSITION": {
        "byteOffset": 8, "byteLength": 36, "count": 3, "type": "VEC3"
        }
    }
                    )
             )
data2 = dict(gltfff(uri, components={
    "POSITION": {
        "byteOffset": 0, "byteLength": 1296, "count": 108, "type": "VEC3"
        }, "NORMAL": {
        "byteOffset": 1296, "byteLength": 1296, "count": 108, "type": "VEC3"
        }, "TEXCOORD_0": {
        "byteOffset": 2592, "byteLength": 864, "count": 108, "type": "VEC2"
        }
    }
                    )
             )

import struct


def dot(v1, v2):
    return reduce(operator.add, (c1 * c2 for c1, c2 in zip(v1, v2)))


def norm(v):
    """
    norm is a length of  vector
    @param v:
    @return:
    """
    return math.sqrt(dot(v, v))


def unit(v: tuple[float]):
    d = norm(v)
    return tuple(crd / d for crd in v)


def cross(v1: tuple[float, float, float], v2: tuple[float, float, float]) -> tuple[float, float, float]:
    """pure cross product implementation"""

    return (v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2], v1[0] * v2[1] - v1[1] * v2[0])


def ppt(pts):
    for p in pts:
        yield p.X, p.Y, p.Z


import numpy as np


def pt(p):
    return p.X, p.Y, p.Z


@dataclass
class Plane:
    origin: typing.Tuple[float, float, float]
    xaxis: typing.Tuple[float, float, float]
    yaxis: typing.Optional[typing.Tuple[float, float, float]] = None
    normal: typing.Optional[typing.Tuple[float, float, float]] = None


@dataclass
class ShapeInterface:
    bounds: list[list[float]]
    uuid: typing.Optional[str] = None
    holes: typing.Optional[list[list[list[float]]]] = None

    def to_poly(self):
        return shapely.Polygon(self.bounds, self.holes)

    def to_world(self, plane=None):
        if plane is not None:
            bounds = [plane.point_at(pt) for pt in self.bounds]
            holes = [[plane.point_at(pt) for pt in hole] for hole in self.holes]
            return ShapeInterface(bounds, holes=holes)

        return self

    def from_world(self, plane=None):
        if plane is not None:
            bounds = [plane.in_plane_coords(pt) for pt in self.bounds]
            holes = [[plane.in_plane_coords(pt) for pt in hole] for hole in self.holes]
            return ShapeInterface(bounds, holes=holes)
        return self


@dataclass
class ContourShape(ShapeInterface):

    def to_world(self, plane=None):
        if plane is not None:
            bounds = [plane.in_plane_coords(pt) for pt in self.bounds]
            holes = [[plane.in_plane_coords(pt) for pt in hole] for hole in self.holes]
            return ContourShape(bounds, holes=holes)
        return self


@dataclass
class Contour:
    shapes: list[ContourShape]
    plane: typing.Optional[PlaneLinear] = None

    def __post_init__(self):
        self.shapes = [shape.to_world(self.plane) for shape in self.shapes]
        self.poly = shapely.multipolygons(shape.to_poly() for shape in self.shapes)

        # print(self.poly)

    def __eq__(self, other):
        return self.poly == other.poly

    @property
    def has_local_plane(self):
        return self.plane is not None


def split(self, cont):
    poly1 = self.poly

    if poly1.intersects(cont):

        if poly1.within(cont):
            # print("inside")

            return [self.transformed_points], 0, [{'area': self.area}]


        else:
            # print("intersects")
            poly3 = poly1.intersection(cont)

            area = []

            res = split3d(poly3, self.plane)
            polys = []
            if self.cont_plane is not None:
                for part in res:
                    poly = [self.cont_plane.point_at(pt).tolist() for pt in part]
                    polys.append(poly)
                    a = area3d(part)
                    # print(0, poly, a)
                    area.append({'area': a})
                return polys, 1, area
            else:
                for part in res:
                    a = area3d(part)
                    # print(1,part, a)
                    area.append({'area': a})
                return res, 1, area

    else:
        # print("outside")

        return [self.transformed_points], 2, [{'area': self.area}]


def split3d(poly3, plane=None):
    if plane:
        pls = []
        if isinstance(poly3, shapely.MultiPolygon):
            for poly in poly3.geoms:
                pts = []
                for pt in list(poly.boundary.coords):
                    real_pt = algorithms.ray_plane_intersection(np.array(pt), np.array([0.0, 0.0, 1.0]), plane)
                    pts.append(real_pt.tolist())
                pls.append(pts)

        elif isinstance(poly3, shapely.Polygon):
            pts = []
            # print(poly3)
            for pt in list(poly3.boundary.coords):
                real_pt = algorithms.ray_plane_intersection(np.array(pt), np.array([0.0, 0.0, 1.0]), plane)
                pts.append(real_pt.tolist())
            pls.append(pts)
        else:
            warnings.warn(f"cutted geometry type is not polygon: {poly3}")
        return pls

    else:
        return mapping(poly3)['cooordinates']


ContourIntersectionResult = namedtuple('ContourIntersectionResult', ['points', 'mask', 'target_plane'])

cutgraph = dict(nodes=dict(), children=dict(), parents=dict()

        )


class ShapeGraphInterface:

    def __init__(self, uuid=None, parents=(), children=(), shape: ShapeInterface = None):
        if uuid is None:
            uuid = uuid4().hex
        self.uuid = uuid
        self._hash = self.uuid

        self.shape = shape
        cutgraph['nodes'][self._hash] = self
        if self._hash not in cutgraph['children']:
            cutgraph['children'][self._hash] = list(children)
        else:
            cutgraph['children'][self._hash].extend(children)
        if self._hash not in cutgraph['parents']:
            cutgraph['parents'][self._hash].extend(parents)

    @property
    def parents(self):
        return cutgraph['parents'][self._hash]

    @property
    def children(self):
        return cutgraph['children'][self._hash]

    def __hash__(self):
        return self._hash


class AbstractNode:
    def __init__(self, parent=None):
        super().__init__()
        self.parent = parent
        self._build_result = None

        self.children = []

    @property
    def build_result(self):
        return self._build_result

    def __call__(self, **kwargs):
        for k, v in kwargs.items():
            if v is not None:
                if getattr(self, k) != v:
                    setattr(self, k, v)
                    self._build_result = None
        if self._build_result is None:
            return self.build()
        else:
            return self._build_result

    @abc.abstractmethod
    def build(self) -> ShapeInterface:
        ...


class TermShapeNode(AbstractNode):
    def __init__(self, *args, data: ShapeInterface, **kwargs):
        super().__init__(*args, **kwargs)
        self.data = ShapeGraphInterface(shape=data)

    def build(self):
        return self.data


class PlaneFromShapeNode(TermShapeNode):

    def build(self):
        return PlaneLinear.from_tree_pt(self.data.bounds[0], self.data.bounds[-1], self.data.bounds[1])


class PlaneTransformNode(AbstractNode):
    def __init__(self, *args, target: PlaneLinear, source: PlaneLinear, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target
        self.source = source

    def build(self) -> ShapeInterface:

        return self.from_world(self.to_world(self.parent.build()))

    def to_world(self, shape: ShapeInterface):
        plane = self.source
        if plane is not None:
            bounds = [plane.point_at(pt) for pt in shape.bounds]
            holes = [[plane.point_at(pt) for pt in hole] for hole in shape.holes]
            return ShapeInterface(bounds, holes=holes)

        return shape

    def from_world(self, shape: ShapeInterface):
        plane = self.target
        if plane is not None:
            bounds = [plane.in_plane_coords(pt) for pt in shape.bounds]
            holes = [[plane.in_plane_coords(pt) for pt in hole] for hole in shape.holes]
            return ShapeInterface(bounds, holes=holes)
        return shape

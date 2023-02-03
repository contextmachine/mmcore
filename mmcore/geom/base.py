#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
from __future__ import absolute_import

from abc import abstractmethod
from collections import namedtuple
from typing import Any

import compas.geometry
import numpy as np
import rhino3dm
from OCC.Core.gp import gp_Pnt
from scipy.spatial.distance import euclidean

from mmcore.baseitems import Matchable

mesh_js_schema = {
    "metadata": dict(),
    "uuid": '',
    "type": "BufferGeometry",
    "data": {
        "attributes": {
            "position": {
                "itemSize": 3,
                "type": "Float32Array",
                "array": []}
            }
        },
    }
pts_js_schema = {
    "metadata": dict(),
    "uuid": '',
    "type": "BufferGeometry",
    "data": {
        "attributes": {
            "position": {
                "itemSize": 3,
                "type": "Float32Array",
                "array": None}
            }
        }
    }


class CustomRepr(type):
    @classmethod
    def __prepare__(metacls, name, bases, display=(), **kws):
        ns = type.__prepare__(name, bases)
        ns["__repr__"] = lambda \
                self: f'<{name}({", ".join([f"{i}={getattr(self, i)}" for i in display])}) at {id(self)}>'
        ns["__str__"] = lambda \
                self: f'{name}({", ".join([f"{i}={getattr(self, i)}" for i in display])})'

        return dict(ns)

    def __new__(mcs, name, bases, attrs, **kwargs):
        return super().__new__(mcs, name, bases, attrs)


class Point(metaclass=CustomRepr, display=("x", "y", "z")):
    cxmdata_keys = "X", "Y", "Z"

    def __init__(self, x, y, z=0.0):
        super().__init__()
        self.x = x
        self.y = y
        self.z = z

    @property
    def xyz(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    @property
    def xy(self) -> tuple[float, float]:
        return self.x, self.y

    def distance(self, other):
        return euclidean(np.asarray(self.xyz), np.asarray(other))

    def __array__(self, dtype=float, *args, **kwargs):
        return np.ndarray.__array__(np.asarray([self.x, self.y, self.z], dtype=dtype, *args, **kwargs), dtype)

    def __len__(self):
        return len(self.xyz)

    def to_rhino(self) -> rhino3dm.Point3d:
        return rhino3dm.Point3d(*self.xyz)

    def to_occ(self) -> 'Point':
        return gp_Pnt(*self.xyz)

    def to_compas(self) -> 'Point':
        return compas.geometry.Point(*self.xyz)

    def to_dict(self) -> 'Point':
        dct = {}
        for k in self.cxmdata_keys:
            dct[k] = getattr(self, k.lower())
        return dct

    @classmethod
    def _validate_dict(cls, dct):
        return all(map(lambda k: (k.upper() in dct.keys()) or (k.lower() in dct.keys()), cls.cxmdata_keys))

    @classmethod
    def from_dict(cls, dct: dict) -> 'Point':
        if cls._validate_dict(dct):
            return cls(*dct.values())
        else:
            raise AttributeError

    @classmethod
    def from_rhino(cls, point: rhino3dm.Point3d) -> 'Point':
        return Point(x=point.X, y=point.Y, z=point.Z)

    @classmethod
    def from_occ(cls, point: gp_Pnt) -> 'Point':
        return cls(*point.XYZ())

    @classmethod
    def from_compas(cls, point: compas.geometry.Point) -> 'Point':
        return cls(point.x, point.y, point.z)


class Rectangle:
    def __init__(self, points):
        self.points = points

    def calculate_perimeter(self):
        if len(self.points) != 4:
            raise ValueError('There must be 4 points to calculate the perimeter.')

        point_pairs = [(self.points[0], self.points[1]),
                       (self.points[1], self.points[2]),
                       (self.points[2], self.points[3]),
                       (self.points[3], self.points[0])]

        perimeter = 0
        for p1, p2 in point_pairs:
            x_diff = p1.x - p2.x
            y_diff = p1.y - p2.y
            perimeter += (x_diff ** 2 + y_diff ** 2) ** 0.5
        return perimeter

    def calculate_area(self):
        if len(self.points) != 4:
            raise ValueError('There must be 4 points to calculate the area.')

        x_s = [p.x for p in self.points]
        y_s = [p.y for p in self.points]
        return (max(x_s) - min(x_s)) * (max(y_s) - min(y_s))


class Polygon:
    def __init__(self, points: list[Point]):
        self.points = points

    area = property(fget=lambda self: self._area())
    perimetr = property(fget=lambda self: self._perimetr())
    centroid = property(fget=lambda self: self._centroid())

    def _perimeter(self):
        perim = 0
        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % len(self.points)]
            perim += ((p1.x - p2.x) ** 2 + (p1.y - p2.y) ** 2 + (p1.z - p2.z) ** 2) ** 0.5
        return perim

    def _area(self):
        area = 0
        for i in range(len(self.points)):
            p1 = self.points[i]
            p2 = self.points[(i + 1) % len(self.points)]
            area += (p1.x * p2.y) - (p2.x * p1.y)
        area = abs(area) / 2
        return area

    def _centroid(self):
        v = self.points
        ans = [0, 0]
        n = len(v)
        signedArea = 0
        # For all vertices
        for i in range(len(v)):
            x0 = v[i].x
            y0 = v[i].y
            x1 = v[(i + 1) % n].x
            y1 = v[(i + 1) % n].y
            # Calculate value of A
            # using shoelace formula
            A = (x0 * y1) - (x1 * y0)
            signedArea += A
            # Calculating coordinates of
            # centroid of polygon
            ans[0] += (x0 + x1) * A
            ans[1] += (y0 + y1) * A
        signedArea *= 0.5
        ans[0] = (ans[0]) / (6 * signedArea)
        ans[1] = (ans[1]) / (6 * signedArea)
        return Point(*ans)


class PolygonWithHoles(Polygon):
    def __init__(self, polygons):
        self._polygons = [polygon if isinstance(Polygon) else Polygon(polygons) for polygon in polygons]

        super(PolygonWithHoles, self).__init__(self.boundary.points)
        self.polygons = ElementSequence(self._polygons)

    def _perimeter(self):
        return np.sum(self.polygons["perimeter"])

    def _area(self):
        return self.boundary._area() - np.sum(ElementSequence(self.holes)["area"])

    @property
    def boundary(self):
        indxz = list(range(len(self._polygons)))
        indxz.sort(key=lambda x: x.perimetr(), reverse=True)
        return self._polygons[indxz[0]]

    @property
    def holes(self):
        indxz = list(range(len(self._polygons)))
        indxz.sort(key=lambda x: x.perimetr(), reverse=True)
        return [self._polygons[i] for i in self.indxz[1:]]


class MmSphere(Matchable):
    __match_args__ = "radius", "center"


class MmAbstractBufferGeometry(Matchable):
    @abstractmethod
    def __array__(self, dtype=float, *args, **kwargs) -> np.ndarray:
        ...

    # noinspection PyTypeChecker
    @property
    def array(self) -> list:
        return self.__array__().tolist()


class MmGeometry(Matchable):
    ...


class MmPoint(Matchable):
    __match_args__ = "x", "y", "z"

    @property
    def xyz(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z

    def distance(self, other):
        return euclidean(np.asarray(self.xyz), np.asarray(other))

    def __array__(self, *args):
        return np.asarray([self.a, self.b, self.c])

    def __len__(self):
        return len(self.xyz)

    @classmethod
    def from_rhino(cls, point: rhino3dm.Point3d) -> 'MmPoint':
        return MmPoint(point.X, point.Y, point.Z)

    @classmethod
    def from_occ(cls, point: gp_Pnt) -> 'MmPoint':
        return MmPoint(*point.XYZ())

    @classmethod
    def from_compas(cls, point: compas.geometry.Point) -> 'MmPoint':
        return MmPoint(point.x, point.y, point.z)


ConversionMethods = namedtuple("ConversionMethods", ["decode", "encode"])
GeometryConversion = namedtuple("GeometryConversion", ["name", "target", "conversion"])

from mmcore.collection.multi_description import ElementSequence

from mmcore.addons import rhino


class Rectangle:

    def __init__(self, width, height):
        self.width = width
        self.height = height

    def perimeter(self):
        return 2 * (self.width + self.height)

    def area(self):
        return self.width * self.height


class MmBoundedGeometry(Matchable):
    __match_args__ = "vertices"
    vertices: list[MmPoint]

    def __array__(self, dtype=float, *args, **kwargs) -> np.ndarray:
        return np.asarray(np.asarray(self.vertices, dtype=dtype, *args, **kwargs))

    @property
    def centroid(self) -> MmPoint:
        rshp = self.__array__()
        return MmPoint(np.average(rshp[..., 0]), np.average(rshp[..., 1]), float(np.average(rshp[..., 2])))

    @property
    def bnd_sphere(self) -> MmSphere:
        return MmSphere(center=self.centroid.array, radius=np.max(
            np.array([self.centroid.distance(MmPoint(*r)) for r in self.array])))


class GeometryConversionsMap(dict):
    """

    """

    def __init__(self, *conversions):

        super().__init__()
        self.conversions = conversions
        self.conversion_sequence = ElementSequence([cnv._asdict() for cnv in conversions])
        for cnv in conversions:
            self[cnv.target] = cnv

    def __getitem__(self, item):
        return self.__getitem__(item)

    def __setitem__(self, item, v) -> None:
        dict.__setitem__(self, item, v)

    def __call__(self, obj):

        for name, cls, conversion in self.conversions:
            decode, encode = conversion

        def wrap_init(*args, target=None, **kwargs):
            print(target)

            if target is not None:
                if self.get(target.__class__) is not None:
                    _decode, _encode = self.get(target.__class__).conversion
                    if encode is not None:
                        setattr(obj, f"to_{name}", encode)

                    return obj(*_decode(target).values())
                else:
                    raise KeyError
            else:
                return obj(*args, **kwargs)

        return wrap_init


from mmcore.baseitems.descriptors import DataView


class BufferGeometryData(DataView):
    """
    @summary : DataView like descriptor, provides BufferGeometry data structure, can follow
    `"indices", "position", "normal", "uv"` attributes for a Mesh instances. Data schema is

    """
    itemsize = {
        "position": 3, "normal": 3, "uv": 2}

    def item_model(self, name: str, value: Any):
        return name, {
            "array": np.asarray(value).flatten().tolist(),
            "itemSize": self.itemsize[name],
            "type": "Float32Array",
            "normalized": False
            }

    def data_model(self, instance, value: list[tuple[str, Any]]):
        return {
            "uuid": instance.uuid,
            "type": "BufferGeometry",
            "data": {
                "attributes": dict(value), "index": dict(type='Uint16Array',
                                                         array=np.asarray(instance.indices).flatten().tolist())
                }
            }


from mmcore.addons.rhino.compute import surf_to_buffer_geometry


@GeometryConversionsMap(
    GeometryConversion("rhino", rhino3dm.Mesh, ConversionMethods(rhino.mesh_to_buffer_geometry, None)),
    GeometryConversion("rhino", rhino3dm.Surface,
                       ConversionMethods(surf_to_buffer_geometry, None)),
    GeometryConversion("rhino", rhino3dm.NurbsSurface,
                       ConversionMethods(surf_to_buffer_geometry, None)))
class MmUVMesh(MmGeometry):
    """
    Mesh with "uv" attribute. Triangulate Quad Mesh.
    0---1---2---3
    |  /|  /|  /|
    | / | / | / |
    |/  |/  |/  |
    4---5---6---7
    |  /|  /|  /|
    | / | / | / |
    |/  |/  |/  |
    8---9--10--11
    """
    __match_args__ = "indices", "position", "normal", "uv"
    buffer_geometry = BufferGeometryData("position", "normal", "uv")
    userData = {}

    def __init__(self, indices, position, normal, uv, /, **kwargs):
        print(indices, position, normal, uv)
        super().__init__(indices=indices, position=position, normal=normal, uv=uv, **kwargs)

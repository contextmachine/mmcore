#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
from __future__ import absolute_import

import uuid
import zlib
from abc import abstractmethod
from collections import namedtuple
from enum import Enum
from functools import lru_cache
from typing import Any

import compas.geometry
import numpy as np
from OCC.Core import TopoDS, gp
from OCC.Core.gp import gp_Pnt
from scipy.spatial.distance import euclidean

from mmcore.addons.mmocc.OCCUtils.Construct import make_closed_polygon


from mmcore.collections import chain_split_list

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
from mmcore.geom.utils import create_buffer, rhino_mesh_to_topology, CommonMeshTopology
import rpyc

rpyc.classic.ClassicService()


class Point:
    __match_args__ = "x", "y"
    cxmdata_keys = "X", "Y", "Z"

    def __init__(self, x, y, z=0.0):
        super().__init__(x, y, z=z)

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

    def to_rhino(self):
        import rhino3dm
        return rhino3dm.Point3d(*self.xyz)

    def to_occ(self) -> gp_Pnt:
        return gp_Pnt(*self.xyz)

    def to_compas(self) -> compas.geometry.Point:
        return compas.geometry.Point(*self.xyz)

    def to_dict(self, lower=False) -> dict:
        if lower:
            return self.to_dict_lower()
        else:
            dct = {}
            for k in self.cxmdata_keys:
                dct[k] = getattr(self, k.lower())
            return dct

    def to_dict_lower(self) -> dict:
        dct = {}
        for k in self.cxmdata_keys:
            dct[k.lower()] = getattr(self, k.lower())
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
    def from_rhino(cls, point) -> 'Point':
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


from mmcore.addons.mmocc.OCCUtils import Construct


class Polygon:
    def __init__(self, points: list[Point]):
        self.points = points

    area = property(fget=lambda self: self._area())
    perimetr = property(fget=lambda self: self._perimetr())
    centroid = property(fget=lambda self: self._centroid())
    vertices = property(fget=lambda self: self._centroid())

    def to_occ(self):
        DAT = map(lambda x: x.to_occ(), self.vertices)

        return Construct.make_closed_polygon(*list(DAT))

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

        return Point(*np.mean(np.asarray([pt.xy for pt in self.points])))


def point_from_all(*args, **kwargs):
    if not (args == ()):
        return Point(*args)
    elif not (kwargs == dict()):
        return Point(*kwargs.values())


class Triangle:
    __match_args__ = "a", "b", "c"

    def __init__(self, a, b, c, *args, **kwargs):
        super().__init__(a, b, c, *args, **kwargs)

    _a, _c = None, None

    @property
    def a(self):
        return self._a

    @a.setter
    def a(self, value):
        if self._a is None:
            self._a = Point(*value)
        else:
            self._a.x, self._a.y, self._a.z = value

    _b = None

    @property
    def b(self):
        return self._b

    @b.setter
    def b(self, value):
        if self._b is None:
            self._b = Point(*value)
        else:
            self._b.x, self._b.y, self._b.z = value

    @property
    def c(self):
        return self._с

    @c.setter
    def c(self, value):

        if self._c is None:
            self._c = Point(*value)
        else:
            self._c.x, self._c.y, self._c.z = value

    @property
    def vertices(self):

        return [self._a, self._b, self._c]

    @property
    def polygon(self):
        return make_closed_polygon(self._a.to_occ(), self._b.to_occ(), self._b.to_occ())

    @property
    def mark(self):

        return "-".join([self.tag, self.subtype])

    @property
    def name(self):
        return "-".join([self.mark, self.id, self.zone])

    @property
    def centroid(self):
        return self.polygon.centroid

    def to_occ(self):
        return make_closed_polygon()


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


class Sphere:
    radius: float
    center: Point
    __match_args__ = "radius", "center"

    def is_point_inside(self, pt: Point):
        return pt.distance(self.center) < self.radius

    def is_point_on_boundary(self, pt: Point):
        return pt.distance(self.center) == self.radius

    def to_rhino(self):
        import rhino3dm

        return rhino3dm.Sphere(self.center.to_rhino(), self.radius)

    def to_compas(self):
        return compas.geometry.Sphere(self.center.to_compas(), self.radius)

    def to_occ(self):
        gp.gp_Ax3()
        ax = gp.gp_Ax3(self.center.to_occ(), gp.gp_Dir(1, 0, 0), gp.gp_Dir(0, 1, 0))
        return gp.gp_Sphere(ax, self.radius)


class MmAbstractBufferGeometry:
    @abstractmethod
    def __array__(self, dtype=float, *args, **kwargs) -> np.ndarray:
        ...

    # noinspection PyTypeChecker
    @property
    def array(self) -> list:
        return self.__array__().tolist()


class MmGeometry:
    ...


class MmPoint:
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
    def from_rhino(cls, point) -> 'MmPoint':
        return MmPoint(point.X, point.Y, point.Z)

    @classmethod
    def from_occ(cls, point: gp_Pnt) -> 'MmPoint':
        return MmPoint(*point.XYZ())

    @classmethod
    def from_compas(cls, point: compas.geometry.Point) -> 'MmPoint':
        return MmPoint(point.x, point.y, point.z)


ConversionMethods = namedtuple("ConversionMethods", ["decode", "encode"])
GeometryConversion = namedtuple("GeometryConversion", ["name", "target", "conversion"])

from mmcore.collections.multi_description import ElementSequence

from mmcore.addons import rhino


class mesh_cache:
    def __init__(self, func):
        self.func = func

    def __call__(self, slf, *args, **kwargs):
        @lru_cache(1024)
        def wwrp(adler):
            print(adler)
            return self.func(slf, *args, **kwargs)

        return wwrp(slf.adler32())


from mmcore.utils.pydantic_mm.models import PropertyBaseModel


class ThreeJsTypeEnum(str, Enum):
    Float32Array = float
    Uint16Array = int


class BufferAttribute(PropertyBaseModel):
    type: str
    itemSize: int | None
    normalized: bool = False

    _array = None

    @property
    def array(self):
        return np.asarray(self._array, dtype=ThreeJsTypeEnum[self.type]).flatten().tolist()

    @array.setter
    def array(self, v):
        self._array = v


class BufferGeometryAttributes(PropertyBaseModel):
    _normal = None
    _position = None
    _uv = None

    @property
    def uv(self):
        return BufferAttribute(array=self._uv,
                               type="Uint16Array",
                               itemSize=2,
                               normalized=False).dict()

    @uv.setter
    def uv(self, value):
        self._uv = value

    @property
    def position(self):
        return BufferAttribute(array=self._position,
                               type="Float32Array",
                               itemSize=3,
                               normalized=False).dict()

    @position.setter
    def position(self, value):
        self._position = value

    @property
    def normal(self):
        return BufferAttribute(array=self._normal,
                               type="Float32Array",
                               itemSize=3,
                               normalized=False).dict()

    @normal.setter
    def normal(self, value):
        self._normal = value


class BufferGeometryData(PropertyBaseModel):
    _index = None
    _attributes = None

    @property
    def attributes(self):
        return dict(
            position=self._attributes.vertices,
            normal=self._attributes.normals,
            uv=self._attributes.uv
        )

    @attributes.setter
    def attributes(self, value):
        self._attributes = value

    @property
    def index(self):
        return dict(array=self._index,
                    type="Uint16Array",
                    normalized=False)

    @index.setter
    def index(self, value):
        self._index = value


class BufferGeometry(PropertyBaseModel):
    uuid: str
    type: str = "BufferGeometry"
    _data: BufferGeometryData | None = None

    def __init__(self, data, uuid):
        BufferGeometryData(attributes=data)

    @property
    def data(self):
        return BufferGeometryData(data=self._data).dict()

    @data.setter
    def data(self, value):
        self._data = value


def create_buffer_geometry(topo: CommonMeshTopology, uid=None):
    return create_buffer(indices=topo.indices,
                         verts=topo.vertices,
                         normals=topo.normals,
                         uv=topo.uv,
                         uid=uuid.uuid4().__str__() if uid is None else uid)


from functools import cached_property


class StateMatchable:
    state_includes = "uuid",
    state_excludes = ()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __init_subclass__(cls, **kwargs):
        incl, excl = set(), set()

        for m in cls.mro():
            if hasattr(m, 'state_includes'):
                if m.state_includes is not None:
                    incl.update(set(m.state_includes))

            if hasattr(m, 'state_excludes'):
                if m.state_excludes is not None:
                    excl.update(set(m.state_excludes))

        cls.state_includes, cls.state_excludes = incl, excl
        super().__init_subclass__(**kwargs)


class CommonMesh(StateMatchable):
    __match_args__ = "indices", "vertices", "normals", "uv"
    _rhino_mesh = None
    _area_mass_properties = None

    @property
    def rhino_mesh(self):
        return self._rhino_mesh

    @rhino_mesh.setter
    def rhino_mesh(self, value):
        self._rhino_mesh = value

    _vertices = None
    _matrix = None

    @property
    def matrix(self):
        return self._matrix

    @matrix.setter
    def matrix(self, value):
        self._matrix = value

    _transform_inplace = False
    _matrix_inplace = [0.001, 0, 0, 0,
                       0, 0.001, 0, 0,
                       0, 0, 0.001, 0,
                       0, 0, 0, 1]

    @property
    def matrix_inplace(self):
        return self._matrix_inplace

    @matrix_inplace.setter
    def matrix_inplace(self, value):
        self._matrix_inplace = value

    @property
    def transform_inplace(self):
        return self._transform_inplace

    @transform_inplace.setter
    def transform_inplace(self, value):
        self._transform_inplace = value

    @property
    def vertices(self):
        if self.transform_inplace:

            m = np.asarray(self.matrix_inplace).reshape((4, 4))
            return np.asarray([(m @ np.asarray(tuple(v) + (1,)).T)[:3] for v in self._vertices]).tolist()
        else:
            return self._vertices

    @vertices.setter
    def vertices(self, value):
        self._vertices = value

    @classmethod
    def from_rhino(cls, rhino_mesh
                   ):
        inst = cls(*rhino_mesh_to_topology(rhino_mesh))
        inst.rhino_mesh = rhino_mesh
        return inst

    @property
    def common_topo(self):
        return CommonMeshTopology(self.indices, self.vertices, self.normals, self.uv)

    @property
    def buffer_geometry(self):
        return create_buffer_geometry(self.common_topo, uid=self.uuid)

    @staticmethod
    def _concat_arr(topo):
        return np.concatenate([np.asarray(topo.vertices), np.asarray(topo.normals)])

    def _cropped_bytes(self) -> bytes:
        """
        Only vertices and normals!
        @return: Byte array
        @rtype: bytes
        """
        return CommonMesh._concat_arr(self.common_topo).tobytes()

    def adler32(self):
        return zlib.adler32(self._cropped_bytes())

    def __eq__(self, other):
        return self.adler32() == other.adler32()

    @cached_property
    def calc_area(self):
        sc = rhino.compute.AreaMassProperties.Compute3(self.rhino_mesh, multiple=False)
        return sc

    @property
    def area(self):

        return self.area_mass_properties["Area"] * 1e-6

    @property
    def centroid(self):
        return self.area_mass_properties["Centroid"]

    @property
    def area_mass_properties(self):
        if self._area_mass_properties is None:
            self._area_mass_properties = self.calc_area
        return self._area_mass_properties


class MmBoundedGeometry:
    __match_args__ = "vertices"
    vertices: list[MmPoint]

    def __array__(self, dtype=float, *args, **kwargs) -> np.ndarray:
        return np.asarray(np.asarray(self.vertices, dtype=dtype, *args, **kwargs))

    @property
    def centroid(self) -> MmPoint:
        rshp = self.__array__()
        return MmPoint(np.average(rshp[..., 0]), np.average(rshp[..., 1]), float(np.average(rshp[..., 2])))

    @property
    def bnd_sphere(self) -> Sphere:
        return Sphere(center=self.centroid.array, radius=np.max(
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


from mmcore.base.descriptors import DataView


class BufferGeometryData2(DataView):
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


# @GeometryConversionsMap(
#    GeometryConversion("rhino", rhino3dm.Mesh, ConversionMethods(rhino.mesh_to_buffer_geometry, None)),
#    GeometryConversion("rhino", rhino3dm.Surface,
#                       ConversionMethods(surf_to_buffer_geometry, None)),
#    GeometryConversion("rhino", rhino3dm.NurbsSurface,
#                       ConversionMethods(surf_to_buffer_geometry, None)))
# class MmUVMesh(MmGeometry):
#    """
#    Mesh with "uv" attribute. Triangulate Quad Mesh.
#    0---1---2---3
#    |  /|  /|  /|
#    | / | / | / |
#    |/  |/  |/  |
#    4---5---6---7
#    |  /|  /|  /|
#    | / | / | / |
#    |/  |/  |/  |
#    8---9--10--11
#    """
#    __match_args__ = "indices", "position", "normal", "uv"
#    buffer_geometry = BufferGeometryData("position", "normal", "uv")
#    userData = {}
#
#    def __init__(self, indices, position, normal, uv, /, **kwargs):
#        print(indices, position, normal, uv)
#        super().__init__(indices=indices, position=position, normal=normal, uv=uv, **kwargs)
#

from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeEdge, BRepBuilderAPI_MakeFace, BRepBuilderAPI_MakeVertex, \
    BRepBuilderAPI_MakeWire



class OccEntity:
    __match_args__ = ()
    __entities__ = __match_args__ + ("input_entity",)
    repr_args = "__entities__"
    input_entity: Any
    occ_entity: Any

    def input_entity(self):
        attrs = []
        for attr in self.__match_args__:
            attrs.append(getattr(self, attr))

        return attrs

    def occ_entity(self):
        ...

    def __set_name__(self, owner, name):
        self.name = name
        self.inst_name = "_" + name

    def __get__(self, instance, own):
        return self.occ_entity()

    def __set__(self, instance, val):
        for name, attr in zip(self.__match_args__, val):
            self.__setattr__(name, attr)


class OccVertex(OccEntity):
    __match_args__ = "x", "y", "z"

    @property
    def point(self):
        return Point(*self.input_entity())

    def occ_entity(self):
        b = BRepBuilderAPI_MakeVertex(self.point.to_occ())
        b.Build()
        return b.Vertex()


class OccEntityList(list):
    def __init_subclass__(cls, item: Any = None, **kwargs):
        cls.item = item
        cls.__match_args__ = item.__match_args__
        super().__init_subclass__(**kwargs)

    def __init__(self, seq=None):

        super().__init__()
        if seq is not None:
            self.extend(seq)

    @property
    def multi_seq(self):
        return ElementSequence(self)

    def __getattr__(self, item):
        es = ElementSequence(self)
        if item in es.keys():
            return es[item]
        else:
            return getattr(self, item)

    def __set_name__(self, owner, name):
        self.name = name
        self.inst_name = "_" + name

    def input_entity(self):
        lst = []
        for i in self:
            lst.append(i.input_entity())
        return lst

    def __getitem__(self, item):
        val = list.__getitem__(self, item)
        return val

    def __setitem__(self, item, value):
        self[item](*value)

    def append(self, value):
        list.append(self, self.item(*value))

    def extend(self, value):
        list.extend(self, [self.item(*v) for v in value])

    def occ_entity(self):
        lst = []
        for i in self:
            lst.append(i.occ_entity())
        return lst

    def __get__(self, instance, own):
        return self.occ_entity()

    def __set__(self, instance, val):
        self.clear()
        self.extend(val)


class OccEdge(OccEntity):
    __match_args__ = "start", "next"

    start = OccVertex()
    next = OccVertex()

    def occ_entity(self):
        b = BRepBuilderAPI_MakeEdge(self.start, self.next)
        b.Build()
        return b.Edge()


class OccVertexList(OccEntityList, item=OccVertex): ...


class OccEdgeList(OccEntityList, item=OccEdge): ...


class OccWire(OccEntity):
    __match_args__ = 'edges',
    edges = OccEdgeList()

    def occ_entity(self):
        b = BRepBuilderAPI_MakeWire(*self.edges)
        b.Build()
        return b.Wire()


def polyline_from_points(*points):
    points = list(points)
    wr = OccWire(list(chain_split_list(points)))
    wr.points = points
    return wr


def fillet_from_points(*points):
    return OccWire(list(chain_split_list(list(points))))


class OccPolyline(OccEntity):
    __match_args__ = 'points',
    points: tuple[float, float, float]
    edges = OccVertexList()

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, item):
        self._points = item
        self.make_edges()

    def make_edges(self):
        l = []
        for v1, v2 in chain_split_list(self.points):
            l.append((v1, v2))
        self.edges = l

    def occ_entity(self):
        b = BRepBuilderAPI_MakeWire(*self.edges)
        b.Build()
        return b.Wire()


def fillet(*points):
    OccWire(create_topods_wire(points))


def create_topods_wire(*pts):
    ss = []

    for pt in pts:
        b = BRepBuilderAPI_MakeVertex(gp_Pnt(*pt))
        b.Build()
        ss.append(TopoDS.TopoDS_Vertex(b.Vertex()))

    edges = []
    for v1, v2 in chain_split_list(ss):
        print(v1, v2)
        vv = BRepBuilderAPI_MakeEdge(v1, v2)
        vv.Build()
        edge = vv.Edge()

        edges.append(edge)

    return BRepBuilderAPI_MakeWire(*edges), edges, ss


class OccPlanarFace(OccEntity):
    __entities__ = "wire"
    wire = OccWire()

    is_planar: bool = True

    def occ_entity(self):
        baseFace = BRepBuilderAPI_MakeFace(self.wire, self.is_planar)
        baseFace.Build()
        return baseFace.Face()


class OccShape(OccEntity):
    __entities__ = "wire"
    wire = OccWire()

    is_planar: bool = True

    def occ_entity(self):
        baseFace = BRepBuilderAPI_MakeFace(self.wire, self.is_planar)
        baseFace.Build()
        return baseFace.Face()

# def fillet(polyline, fillets):
#
#
#
#
#    filletOp = BRepFilletAPI_MakeFillet2d(baseFace)
#
#    rFillet1 = 0.1
#
#    rFillet2 = 0.03
#
#    filletOp.AddFillet(V1, rFillet1)
#    filletOp.AddFillet(V2, rFillet2)
#    filletOp.AddFillet(V3, rFillet1)
#    filletOp.AddFillet(V4, rFillet2)
#    filletOp.Build()
#
#    explorer(filletOp.Shape(), TopAbs_WIRE)
#
#    filletWire = TopoDS.Wire(explorer.Current())

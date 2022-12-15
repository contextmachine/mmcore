#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
from __future__ import annotations, with_statement

import copy
import typing
from abc import abstractmethod
from enum import Enum

import compas.geometry as cg
import numpy as np
import pydantic
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_MakeShell
from OCC.Core.GC import *
from OCC.Core.gp import gp_Pnt
from pydantic import BaseModel
from rhino3dm import Point3d

from baseitems import GeomConversionMap, GeomDataItem, ReprData


class BufferGeometryDataTypes(str, Enum):
    geometries: str = "BufferGeometry"
    object: str = "Object"


class BufferGeometryMetadata(pydantic.BaseModel):
    type: BufferGeometryDataTypes = BufferGeometryDataTypes.geometries
    generator: str = "CxmGenerator"
    version: float = 4.5


class BufferGeometryProperty(pydantic.BaseModel):
    array: list[float]
    itemSize: int = 3
    type: str = "Float32Array"
    normalized: bool = False


class BufferGeometryPosition(BufferGeometryProperty): ...


class BufferGeometryNormal(BufferGeometryProperty): ...


class BufferGeometryUV(BufferGeometryProperty):
    itemSize: int = 2


class BufferGeometryBoundingSphere(pydantic.BaseModel):
    center: list[float, float, float] = [0, 0, 0]
    radius: float = 1.0


class BufferGeometryGroup(pydantic.BaseModel):
    start: int = 0
    count: int = 1
    materialIndex: int = 0


class BufferGeometryData(pydantic.BaseModel):
    attributes: dict[str, BufferGeometryProperty]
    boundingSphere: BufferGeometryBoundingSphere = BufferGeometryBoundingSphere()

    def __init__(self, **data: typing.Any):
        super().__init__(**data)


class BufferGeometrySimple(BaseModel):
    data: BufferGeometryData
    uuid: str
    type: BufferGeometryDataTypes


class BufferGeometryModel(BufferGeometrySimple):
    metadata: BufferGeometryMetadata

    # children: list[pydantic.BaseModel | typing.Any | None] = []


class BufferPointMap(GeomConversionMap):
    include = ["array"]
    exclude = ["x", "y", "z", 'args', 'kw', 'representation', 'aliases', 'fields', 'uid', '__array__', '_dtype']

    def __get_dict__(self, instance, owner):
        data = BufferGeometryData(attributes=dict(
            position=BufferGeometryPosition(array=instance.array),
        ),
            boundingSphere=BufferGeometryBoundingSphere(center=instance.array, radius=0.5)
        )

        return BufferGeometryModel(data=data)


class BufferGeometryItem(GeomDataItem, typing.Sequence):
    representation = ReprData("array")
    data = BufferPointMap()

    def __getitem__(self, item):
        return self.array[item]

    def __len__(self):
        return len(self.array)

    @abstractmethod
    def __array__(self, *args, **kw) -> np.ndarray: ...

    def __list__(self) -> list[float]:
        return self.__array__().tolist()

    @property
    def array(self) -> list[float] | list[list[float]]:
        return self.__array__().tolist()


class BufferPoint(BufferGeometryItem):
    """
    >>> point = BufferPoint(1, 2, 3)
    >>> print(point.to_json(indent=3))
    {
        "metadata": {
            "uuid": "4c6adef0bf554c2fb2731e8ddd8dcab3",
            "dtype": "Point",
            "version": "0x4d0x590x500x5c0x60",
            "custom_fields": [
                "custom_fields",
                "_uuid",
                "base_fields",
                "version"
            ],
            "base_fields": [
                "x",
                "y",
                "z"
            ]
        },
        "geometries": {
            "data": {
                "attributes": {
                    "position": {
                        "array": [
                            1.0,
                            2.0,
                            3.0
                        ],
                        "itemSize": 3,
                        "type": "Float32Array",
                        "normalized": false
                    }
                },
                "boundingSphere": {
                    "center": [
                        1.0,
                        2.0,
                        3.0
                    ],
                    "radius": 0.5
                },
                "groups": []
            },
            "uuid": "4c6adef0bf554c2fb2731e8ddd8dcab3",
            "type": "BufferGeometry"
        }
    }

    """
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0

    def __contains__(self, item):
        return item in self.array

    @property
    def shape(self):
        return 3,

    def __init__(self, x, y, z, *args, **kwargs):
        super().__init__(x=x, y=y, z=z, *args, **kwargs)

    def __array__(self, *args, **kw):
        return np.asarray([self.x, self.y, self.z], dtype=float).reshape(self.shape)

    def to_dict(self):
        dict_ = super().to_dict()
        del dict_["x"], dict_["y"], dict_["z"]

        return dict_

    def to_compas(self):
        return cg.Point(self.x, self.y, self.z)

    def to_occ(self):
        return gp_Pnt(self.x, self.y, self.z)

    def to_rhino(self):
        return Point3d(self.x, self.y, self.z)

    def distance(self, other: BufferGeometryItem | list[BufferGeometryItem]):
        if isinstance(other, list):
            return [self.to_occ().Distance(x.to_occ()) for x in other]
        else:
            return self.to_occ().Distance(other.to_occ())


class BI(BufferGeometryItem):

    def array(self) -> list[list[float]]:
        a, b, c = self.__array__().shape

        return self.__array__().reshape(a * b, 3).tolist()

    @property
    def centroid(self) -> BufferPoint:
        rshp = self.__array__()
        return BufferPoint(np.average(rshp[..., 0]), np.average(rshp[..., 1]), float(np.average(rshp[..., 2])))

    @property
    def bnd_sphere(self) -> BufferGeometryBoundingSphere:
        return BufferGeometryBoundingSphere(center=self.centroid.array, radius=np.max(
            np.array([self.centroid.distance(BufferPoint(*r)) for r in self.array])))

    def __array__(self, *args, **kw) -> np.ndarray:
        pass


class BufferFaceMap(GeomConversionMap):
    include = ["array"]
    exclude = ["vertices", 'args', 'kw', 'representation', 'aliases', 'fields', 'uid', '__array__', '_dtype']

    def __get_dict__(self, instance, owner):
        data = BufferGeometryData(attributes=dict(
            position=BufferGeometryPosition(array=instance.array),
        ),
            boundingSphere=BufferGeometryBoundingSphere(center=instance.centroid.array, radius=0.5)
        )
        BufferGeometryModel.update_forward_refs(data=data)
        instance._pydantic = BufferGeometryModel(uuid=str(instance.uuid), type='BufferGeometry')
        instance._pydantic.data = data
        return instance._pydantic.dict()


class BufferGmMap(BufferFaceMap):
    include = []
    exclude = ["vertices", 'args', 'kw', 'representation', 'aliases', 'fields', 'uid', '__array__', '_dtype']

    def __get_dict__(self, instance, owner) -> dict:
        data = BufferGeometryData(attributes=dict(
            position=BufferGeometryPosition(array=instance.array),
        ),
            boundingSphere=instance.bnd_sphere
        )
        BufferGeometryModel.update_forward_refs(data=data)
        instance._pydantic = BufferGeometryModel(uuid=str(instance.uuid), type='BufferGeometry')

        return instance._pydantic.dict()


class BufferOCCMap(BufferGmMap):
    include = []
    exclude = ["to_compas", "to_rhino", "to_occ", "__array__", 'args', 'kw', 'representation', 'aliases', 'fields',
               'uid', '__array__', '_dtype']

    def __get_dict__(self, instance, owner) -> dict:

        if hasattr(instance, "__tree_js_convert_attrs__"):

            dt = topo_converter(BRepBuilderAPI_MakeShell(instance.occ).Shell(), instance.__tree_js_convert_attrs__)

        else:
            dt = topo_converter(BRepBuilderAPI_MakeShell(instance.occ).Shell())
        instance._pydantic = BufferGeometryModel(metadata=BufferGeometryMetadata(**dt["metadata"]),
                                                 uuid=dt['uuid'],
                                                 data=dt["data"], type=BufferGeometryDataTypes.geometries)

        instance._pydantic.data.boundingSphere = instance.bnd_sphere
        instance._dt = dt
        return instance._pydantic.dict()


class BufferFace(BI):
    data = BufferGmMap()
    reparesentation = ReprData("array")
    vertices: list[BufferPoint] = [BufferPoint(0, 1, 0), BufferPoint(1, 0, 0), BufferPoint(0, 0, 1)]

    def __contains__(self, item):
        return item in self.vertices

    @property
    def shape(self):
        return int(len(self.vertices) * 3),

    def __init__(self, *args, **kwargs):
        super().__init__(vertices=list(args), **kwargs)

    def __array__(self, *args, **kw):
        return np.asarray(self.vertices, dtype=object).flatten().reshape(self.shape)

    def to_dict(self):
        dct_ = super().to_dict()
        del dct_["vertices"]
        return dct_


from geom.geomutils import topo_converter, data_scheme


class BufferGeometryOcc(BI):
    def __array__(self, *args, **kw) -> np.ndarray:
        pass

    __tree_js_convert_attrs__ = dict(
        export_edges=True,
        color=(0.65, 0.65, 0.7),
        specular_color=(0.2, 0.2, 0.2),
        shininess=0.9,
        transparency=0.,
        line_color=(0, 0., 0.),
        line_width=1.,
        mesh_quality=21,
        deflection=0.01,
        scheme=copy.deepcopy(data_scheme))
    data = BufferOCCMap()

    def to_dict(self):
        return self.data


from geom.mat import TreeJsPhongMaterial


class BufferGeometryObjProperty:
    __dct__ = {}
    names = {"uuid": "uuid",

             "geometry": "geometries",
             "material": "materials"}

    def __set_name__(self, owner, name):
        self.name = '_' + name
        self.__ps = self.names[name]

    def __get__(self, inst, own):
        return inst.__dict__[self.__ps].uuid


class Prop(BufferGeometryObjProperty):
    def __set_name__(self, owner, name):
        self.name = name
        self.__ps = "_" + self.name

    def __get__(self, inst, own):
        return inst.__dict__[self.__ps]

    def __set__(self, inst, uu):
        inst.__dict__[self.__ps] = uu


def ObjProperty(owner, name) -> BufferGeometryObjProperty | Prop:
    if name in BufferGeometryObjProperty.names.keys():
        setattr(owner, name, BufferGeometryObjProperty())

    else:
        setattr(owner, name, Prop())
    return getattr(owner, name)


class BufferGeometryObject(pydantic.BaseModel):
    geometries: list[BufferGeometrySimple]
    materials: list[TreeJsPhongMaterial]
    object: typing.Any
    metadata: BufferGeometryMetadata = BufferGeometryMetadata(
        **{"version": 4.5, "type": "Object", "generator": "Object3D.toJSON"})


class TrimmingCone(BufferGeometryOcc):
    representation = ReprData("point_a", "point_b", "radius_a", "radius_b", "trim")

    def __call__(self, point_a, point_b, radius_a, radius_b, *args, **kwargs):
        super().__call__(*args, **kwargs)
        self.point_a, self.point_b, self.radius_a, self.radius_b = point_a, point_b, radius_a, radius_b
        if self._trim is None:
            self.trim = self._occ().Bounds()

    _trim = None
    _radius_a = None
    _radius_b = None
    _point_a = None
    _point_b = None

    @property
    def occ(self):
        geoms = self._occ()
        geoms.SetTrim(*self.trim)
        return geoms

    def _occ(self):
        return GC_MakeTrimmedCone(gp_Pnt(*self.point_a), gp_Pnt(*self.point_b), self.radius_a,
                                  self.radius_b).Value()

    @property
    def trim(self):
        return self._trim

    @trim.setter
    def trim(self, trims):
        self._trim = trims

    @property
    def point_b(self):
        return self._point_b

    @point_b.setter
    def point_b(self, value):
        self._point_b = value

    @property
    def point_a(self):
        return self._point_a

    @point_a.setter
    def point_a(self, value):
        self._point_a = value

    @property
    def radius_b(self):
        return self._radius_b

    @radius_b.setter
    def radius_b(self, value):
        self._radius_b = value

    @property
    def radius_a(self):
        return self._radius_a

    @radius_a.setter
    def radius_a(self, value):
        self._radius_a = value

    @property
    def bounds(self):
        return self.occ.Bounds()

    def __array__(self, nums=(8, 8), *args, **kw):
        umin, umax, vmin, vmax = self.bounds
        unum, vnum = nums
        _u, _v = np.linspace(umin, umax, unum), np.linspace(vmin, vmax, vnum)
        d = np.zeros(nums + (3,), dtype=float)

        for i in range(unum):
            for j in range(vnum):
                pt = self.occ.Value(_u[i], _v[j])
                d[i, j, :] = pt.X(), pt.Y(), pt.Z()

        return d

    @property
    def array(self) -> list[list[float]]:
        a, b, c = self.__array__().shape

        return self.__array__().reshape(a * b, 3).tolist()


if __name__ == "__main__":
    BufferGeometryMetadata.update_forward_refs()
    tc = TrimmingCone([1, 2, 3], [12, 15, 8], 5, 7)
    print(tc.data)
    print(tc.to_dict())
    print(tc.to_json(indent=2))

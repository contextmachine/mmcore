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
from mmcore.addons import compute


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


@GeometryConversionsMap(
    GeometryConversion("rhino", rhino3dm.Mesh, ConversionMethods(rhino.mesh_to_buffer_geometry, None)),
    GeometryConversion("rhino", rhino3dm.Surface,
                       ConversionMethods(compute.request_models.surf_to_buffer_geometry, None)),
    GeometryConversion("rhino", rhino3dm.NurbsSurface,
                       ConversionMethods(compute.request_models.surf_to_buffer_geometry, None)))
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

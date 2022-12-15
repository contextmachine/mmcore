#  Copyright (c) 2022. Computational Geometry, Digital Engineering and Optimizing your construction processe"
from __future__ import absolute_import

from abc import abstractmethod

import compas.geometry
import compas.geometry as cg
import numpy as np
import rhino3dm
from OCC.Core.gp import gp_Pnt
from scipy.spatial.distance import euclidean

from baseitems import DictableItem
from baseitems import Matchable, WithSlots

mesh_js_schema = {
    "metadata": dict(),
    "uuid": '',
    "type": "BufferGeometry",
    "data": {"attributes": {"position": {"itemSize": 3,
                                         "type": "Float32Array",
                                         "array": []}
                            }
             },
}
pts_js_schema = {
    "metadata": dict(),
    "uuid": '',
    "type": "BufferGeometry",
    "data": {"attributes": {"position": {"itemSize": 3,
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


class Point(DictableItem):
    x = 0.0
    y = 0.0
    z = 0.0
    exclude = ["version", "uid", "__array__"]

    def to_compas(self):
        return cg.Point(self.x, self.y, self.z)

    def __array__(self, *args):
        return np.asarray([self.x, self.y, self.z])


class MmPoint(WithSlots, MmAbstractBufferGeometry):
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


class MmBoundedGeometry(MmAbstractBufferGeometry):
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


class MmFace(MmBoundedGeometry):
    ...


'''class Quad(Vector):
    a: Point
    b: Point
    c: Point
    d: Point'''

import uuid

import dataclasses

import numpy as np
import typing

import earcut
from earcut import earcut
from more_itertools import flatten

from mmcore.base import AMesh, ALine, A, AGroup, Delegate
from mmcore.base.geom import MeshData
from mmcore.base.models.gql import MeshPhongMaterial, LineBasicMaterial
from mmcore.geom.materials import ColorRGB

import shapely
from shapely.set_operations import union_all, union, intersection_all, intersection, difference
from shapely import Polygon, MultiPolygon


def to_list_req(obj):
    if not isinstance(obj, str):
        if hasattr(obj, "coords"):
            return to_list_req(obj.coords)
        else:
            try:

                return [to_list_req(o) for o in list(obj)]
            except Exception:
                return obj
    else:
        return obj


def delegate_method(m):
    def wrap(self, item):
        if isinstance(item, self._ref.__class__):
            return m(self._ref, item)
        elif isinstance(item, self.__class__):
            return m(self._ref, item._ref)
        elif isinstance(item, (np.ndarray, list, tuple)):
            return m(self._ref, self._ref.__class__(item))
        elif hasattr(item, m.__name__):
            return m(self._ref, item)
        else:
            raise ValueError(f"{m.__name__.capitalize()} operation unknown in {item.__class__} objects")

    return wrap


@Delegate(delegate=Polygon)
@dataclasses.dataclass
class Shape:
    boundary: list[list[float]]
    holes: typing.Optional[list[list[list[float]]]] = None
    color: ColorRGB = ColorRGB(200, 20, 15).decimal
    uuid: typing.Optional[str] = None
    h: typing.Any = None

    def __post_init__(self):
        if not self.uuid:
            self.uuid = uuid.uuid4().hex
        if self.h is None:
            self.h = 0
        if self.holes is None:
            self.holes = []
        self._ref = Polygon(shell=self.boundary, holes=self.holes)

    @property
    def mesh(self):
        return AMesh(uuid=self.uuid + "-mesh",
                     geometry=self.mesh_data.create_buffer(),
                     material=MeshPhongMaterial(color=self.color),
                     name="Shape Mesh")

    def earcut_poly(self):

        data = earcut.flatten([self.boundary] + self.holes)
        res = earcut.earcut(data['vertices'], data['holes'], data['dimensions'])
        return np.array(res).reshape((len(res) // 3, 3))

    def to3d_mesh_pts(self):
        rrr = np.array(list(flatten([self.boundary] + self.holes)))
        return np.c_[rrr, np.ones((rrr.shape[0], 1)) * self.h]

    def to3d_mesh_holes(self):
        l = []
        for hole in self.holes:
            rrr = np.array(hole)
            l.append(np.c_[rrr, np.zeros((rrr.shape[0], 1))].tolist())
        return l

    def to3d_mesh_bnd(self):
        rrr = np.array(self.boundary)
        return np.c_[rrr, np.zeros((rrr.shape[0], 1))].tolist()

    @property
    def mesh_data(self):
        _mesh_data = MeshData(self.to3d_mesh_pts(), indices=self.earcut_poly())
        _mesh_data.calc_normals()
        return _mesh_data

    @delegate_method
    def __contains__(self, item):
        return shapely.contains(self, item)

    def contains(self, other):
        """
        __contains__ alias
        @param other:
        @return:
        """
        return self.__contains__(other)

    @property
    def exterior(self):
        return list(self._ref.exterior.coords)

    @property
    def interior(self):
        return to_list_req(self._ref.interiors)

    @delegate_method
    def within(self, item):
        return shapely.within(self, item)

    @delegate_method
    def intersects(self, item):
        return shapely.intersects(self, item)

    @delegate_method
    def contains_properly(self, item):
        return shapely.contains_properly(self, item)

    def evaluate(self, t):
        return np.asarray(self._ref.interpolate(t, normalized=True), dtype=float)

    def evaluate_distance(self, d):
        return np.asarray(self._ref.interpolate(d, normalized=False), dtype=float)

    def __add__(self, item):
        res = self.union(item)
        return Shape(boundary=list(res.exterior.coords),
                     holes=to_list_req(res.interiors),
                     color=self.color,
                     h=self.h)
    def __sub__(self, item):
        res = self.difference(item)
        if isinstance(res, MultiPolygon):
            shapes=[]
            for i in res.geoms:


                shapes.append(Shape(boundary=list(i.exterior.coords),
                             holes=to_list_req(i.interiors),
                             color=self.color,
                             h=self.h))
            return shapes
        else:
            return Shape(boundary=list(res.exterior.coords),
                     holes=to_list_req(res.interiors),
                     color=self.color,
                     h=self.h)
    def __isub__(self, item):
        res = self.difference(item)
        self.boundary = list(res.exterior.coords)
        self.holes = to_list_req(res.interiors)
    @delegate_method
    def difference(self, item):
        return shapely.difference(self, item)

    @delegate_method
    def union(self, other):
        return shapely.union(self, other)

    def __iadd__(self, item):
        res = self.union(item)

        self.boundary = list(res.exterior.coords)
        self.holes = to_list_req(res.interiors)

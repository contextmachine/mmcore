import uuid

import dataclasses

import numpy as np
import typing

import earcut
from earcut import earcut
from more_itertools import flatten

from mmcore.base import AMesh, ALine, A, AGroup, Delegate
from mmcore.base.delegate import class_bind_delegate_method, delegate_method
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


@class_bind_delegate_method
def bind_poly_to_shape(self,  other, delegate=None ):
    return self.__class__(boundary=list(delegate.boundary.coords), holes=list(delegate.interiors), color=self.color, h=self.h)


@delegate_method
def delegate_shape_operator(self, item, m):
    if isinstance(item, self._ref.__class__):
        return item
    elif isinstance(item, self.__class__):
        return item._ref
    elif isinstance(item, (np.ndarray, list, tuple)):
        return self._ref.__class__(item)
    elif hasattr(item, m.__name__):
        return item
    else:
        raise ValueError(f"{m.__name__.capitalize()} operation unknown in {item.__class__} objects")


# @Delegate(delegate=Polygon)
@dataclasses.dataclass
class Shape:
    boundary: list[list[float]]
    holes: typing.Optional[list[list[list[float]]]] = None
    color: ColorRGB = ColorRGB(200, 20, 15).decimal
    uuid: typing.Optional[str] = None
    h: typing.Any = None

    def __post_init__(self):
        self.boundary = list(self.boundary)
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
        print(self.boundary)
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
        # _mesh_data.calc_normals()
        return _mesh_data

    @delegate_shape_operator.bind
    def __contains__(self, delegate, item):
        return shapely.contains(self, delegate, item)

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

    @delegate_shape_operator.bind
    def within(self, delegate, item):
        return shapely.within(delegate, item)

    @delegate_shape_operator.bind
    def intersects(self, delegate, item):
        return shapely.intersects(delegate, item)

    @delegate_shape_operator.bind
    def contains_properly(self, delegate, item):
        return shapely.contains_properly(delegate, item)

    def evaluate(self, t):
        return np.asarray(self._ref.interpolate(t, normalized=True), dtype=float)

    def evaluate_distance(self, d):
        return np.asarray(self._ref.interpolate(d, normalized=False), dtype=float)

    def __add__(self, item):
        return self.union(item)

    @delegate_shape_operator.bind
    def __sub__(self, delegate, item):
        print(delegate, item)
        res = shapely.difference(delegate, item)
        if isinstance(res, MultiPolygon):
            shapes = []
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

    @bind_poly_to_shape
    @delegate_shape_operator.bind
    def difference(self, delegate, item):
        return shapely.difference(delegate, item)

    @bind_poly_to_shape
    @delegate_shape_operator.bind
    def union(self, delegate, other):
        return shapely.union(delegate, other)

    def __iadd__(self, item):
        res = self.union(item)

        self.boundary = list(res.exterior.coords)
        self.holes = to_list_req(res.interiors)


    def intersection(self,  other):
        res=shapely.intersection(self._ref, other._ref)
        if isinstance(res, MultiPolygon):
            shapes = []
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



    def is_empty(self):
        return self.boundary == []

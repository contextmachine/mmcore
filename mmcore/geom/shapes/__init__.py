import uuid

import dataclasses

import numpy as np
import typing


import earcut
from earcut import earcut
from more_itertools import flatten

from mmcore.base import AMesh, ALine, A, AGroup
from mmcore.base.geom import MeshData
from mmcore.base.models.gql import MeshPhongMaterial, LineBasicMaterial
from mmcore.geom.materials import ColorRGB


@dataclasses.dataclass
class Shape:
    boundary: list[list[float]]
    holes: typing.Optional[list[list[list[float]]]] = None
    color: ColorRGB = ColorRGB(200, 20, 15).decimal
    uuid: typing.Optional[str] = None
    h:typing.Any=None
    def __post_init__(self):
        if not self.uuid:
            self.uuid=uuid.uuid4().hex
        if self.h is None:
            self.h=0
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
        return np.c_[rrr, np.ones((rrr.shape[0], 1))*self.h]
    def to3d_mesh_holes(self):
        l=[]
        for hole in self.holes:
            rrr = np.array(hole)
            l.append(np.c_[rrr, np.zeros((rrr.shape[0], 1))].tolist())
        return l
    def to3d_mesh_bnd(self):
        rrr = np.array(self.boundary)
        return np.c_[rrr, np.zeros((rrr.shape[0], 1))].tolist()


    @property
    def mesh_data(self):
        _mesh_data=MeshData(self.to3d_mesh_pts(), indices=self.earcut_poly())
        _mesh_data.calc_normals()
        return _mesh_data


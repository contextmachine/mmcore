import numpy as np
import typing

import dataclasses
import earcut
from earcut import earcut
from more_itertools import flatten

from mmcore.base import AMesh
from mmcore.base.geom import MeshData
from mmcore.base.models.gql import MeshPhongMaterial
from mmcore.geom.materials import ColorRGB


@dataclasses.dataclass
class ShapeFace:
    boundary: list[list[float]]
    holes: typing.Optional[list[list[list[float]]]] = None
    color: ColorRGB = ColorRGB(200, 20, 15)

    def __post_init__(self):
        self.solve_mesh()

    def earcut_poly(self):
        data = earcut.flatten([self.boundary] + self.holes)
        res = earcut.earcut(data['vertices'], data['holes'], data['dimensions'])
        return np.array(res).reshape((len(res) // 3, 3))

    def to3d_mesh_pts(self):
        rrr = np.array(list(flatten([self.boundary] + self.holes)))
        return np.c_[rrr, np.zeros((rrr.shape[0], 1))]

    def solve_mesh(self):
        self._mesh = MeshData(self.to3d_mesh_pts(), indices=self.earcut_poly())
        self._mesh.calc_normals()

    @property
    def mesh_data(self):
        return self._mesh

    def to_mesh(self):
        return AMesh(geometry=self.mesh_data.create_buffer(), material=MeshPhongMaterial(color=self.color.decimal))

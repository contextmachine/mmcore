import json
import warnings

from abc import ABCMeta
try:
    from OCC.Core.Tesselator import ShapeTesselator
except ImportError as err:
    warnings.warn(ImportWarning("Install pythonOCC to use this tesselation tools"))
import uuid as _uuid
import numpy as np
from mmcore.base.basic import AGroup, AMesh
from mmcore.geom.materials import ColorRGB
from mmcore.base.geom import MeshObject, LineObject
from mmcore.base.geom.utils import create_buffer_from_dict
from mmcore.base.registry import amatdict
from mmcore.base.utils import generate_edges_material, export_edgedata_to_json
from mmcore.collections import ElementSequence
from mmcore.base.models import gql as gql_models
import uuid as _uuid
from mmcore.base.models.gql import BufferGeometryObject
def simple_tessellate(shape, uuid=None, compute_edges: bool = False,
            mesh_quality: float = 1.0,
            parallel: bool = False):
    tesselator=ShapeTesselator(shape)
    tesselator.Compute(compute_edges=compute_edges, mesh_quality=mesh_quality, parallel=parallel)
    data=json.loads(tesselator.ExportShapeToThreejsJSONString(uuid if uuid is not None else _uuid.uuid4().hex))
    del data["metadata"]
    return BufferGeometryObject(**data)



class Tessellate(metaclass=ABCMeta):
    def __init__(self, shape, name, color):
        super().__init__()

        self.tess = ShapeTesselator(shape)
        self._name = name
        self.color = color
        self.generate_material()

    def tessellate(self, compute_edges=False, mesh_quality=1.0, parallel=True):

        self.tess.Compute(compute_edges=compute_edges,
                          mesh_quality=mesh_quality,
                          parallel=parallel)

        __uuid = _uuid.uuid4().__str__()

        self.mesh=AMesh(name=self._name, geometry=create_buffer_from_dict(
            json.loads(
                self.tess.ExportShapeToThreejsJSONString(__uuid)
            )
        ),
            material=self.generate_material())




        return self.mesh

    def generate_material(self):
        vv = list(amatdict.values())
        if len(vv) > 0:
            ##print(vv)
            es = ElementSequence(vv)
            ##print(self.color, es["color"])

            if self.color.decimal in es["color"]:
                i = es["color"].index(self.color.decimal)
                ##print(i)
                vvv = es._seq[i]
                ##print(vvv)

                return amatdict[vvv.uuid]
            else:
                return gql_models.MeshPhongMaterial(
                    name=f"{'MeshPhongMaterial'} {self._name}",
                    color=self.color.decimal)

        else:
            return gql_models.MeshPhongMaterial(name=f"{'MeshPhongMaterial'} {self._name}",
                                                                          color=self.color.decimal)


class TessellateIfc(Tessellate):
    def __init__(self, shape):
        self._shape = shape
        super().__init__(shape.geometry, color=ColorRGB(*np.abs(np.asarray(shape.styles[0][:-1]))), name=shape.data.name)

    def tessellate(self, compute_edges=False, mesh_quality=1.0, parallel=True):
        return super().tessellate(compute_edges=False, mesh_quality=1.0, parallel=True)

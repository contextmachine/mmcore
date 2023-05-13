import json

from abc import ABCMeta

from OCC.Core.Tesselator import ShapeTesselator
import uuid as _uuid

from mmcore.base.basic import Group
from mmcore.geom.materials import ColorRGB
from mmcore.base.geom import MeshObject, LineObject
from mmcore.base.geom.utils import create_buffer_from_dict
from mmcore.base.registry import matdict
from mmcore.base.utils import generate_edges_material, export_edgedata_to_json
from mmcore.collections import ElementSequence
from mmcore.base.models import gql as gql_models

class Tessellate(metaclass=ABCMeta):
    def __init__(self, shape, name, color):
        super().__init__()
        self.mesh = MeshObject(name=name)
        self.tess = ShapeTesselator(shape)
        self._name = name
        self.color = color
        self.generate_material()

    def tessellate(self, compute_edges=False, mesh_quality=1.0, parallel=True):

        self.tess.Compute(compute_edges=compute_edges,
                          mesh_quality=mesh_quality,
                          parallel=parallel)

        __uuid = _uuid.uuid4().__str__()

        self.mesh.geometry = create_buffer_from_dict(

            json.loads(self.tess.ExportShapeToThreejsJSONString(_uuid)))

        if compute_edges:

            # export each edge to a single json
            # get number of edges
            nbr_edges = self.tess.ObjGetEdgeCount()

            grp = Group(name="edges")
            self.mesh.edges = grp
            for i_edge in range(nbr_edges):

                # after that, the file can be appended
                str_to_write = ""
                edge_point_set = []
                nbr_vertices = self.tess.ObjEdgeGetVertexCount(i_edge)
                for i_vert in range(nbr_vertices):
                    edge_point_set.append(self.tess.GetEdgeVertex(i_edge, i_vert))
                # write to file
                ln = LineObject(name=f"edge-{i_edge}")
                ln.geometry = export_edgedata_to_json(_uuid.uuid4().__str__(), edge_point_set)
                ln.material = generate_edges_material(_uuid.uuid4().__str__(), color=ColorRGB(0, 0, 0), linewidth=1.0)
                grp.add(ln)

        return self.mesh

    def generate_material(self):
        vv = list(matdict.values())
        if len(vv) > 0:
            print(vv)
            es = ElementSequence(vv)
            print(self.color, es["color"])
            if self.color.decimal in es["color"]:
                i = es["color"].index(self.color.decimal)
                print(i)
                vvv = es._seq[i]
                print(vvv)
                self.mesh._material = vvv.uuid

            else:
                self.mesh.material = gql_models.MeshPhongMaterial(
                    name=f"{'MeshPhongMaterial'} {self._name}",
                    color=self.color.decimal)

        else:
            self.mesh.material = gql_models.MeshPhongMaterial(name=f"{'MeshPhongMaterial'} {self._name}",
                                                                          color=self.color.decimal)

import numpy as np
class TessellateIfc(Tessellate):
    def __init__(self, shape):
        self._shape = shape
        super().__init__(shape.geometry, color=ColorRGB(*np.abs(np.asarray(shape.styles[0][:-1]))), name=shape.data.name)

    def tessellate(self, compute_edges=False, mesh_quality=1.0, parallel=True):
        return super().tessellate(compute_edges=False, mesh_quality=1.0, parallel=True)

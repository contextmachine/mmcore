import json
import warnings
from abc import ABCMeta

from mmcore.geom.vectors import unit

try:
    from OCC.Core.Tesselator import ShapeTesselator
except ImportError as err:
    warnings.warn(ImportWarning("Install pythonOCC to use this tesselation tools"))
import numpy as np
from mmcore.base.basic import AMesh
from mmcore.geom.materials import ColorRGB
from mmcore.base.geom import MeshData
from mmcore.base import create_buffer_from_dict
from mmcore.base.registry import amatdict
from mmcore.collections import ElementSequence
from mmcore.base.models import gql as gql_models
import uuid as _uuid
from mmcore.base.models.gql import MeshPhongMaterial


def simple_tessellate(shape, uuid=None, color=(120, 200, 40), compute_edges: bool = False,
                      mesh_quality: float = 1.0,
                      parallel: bool = False):
    tesselator = ShapeTesselator(shape)
    tesselator.Compute(compute_edges=compute_edges, mesh_quality=mesh_quality, parallel=parallel)
    _ = uuid if uuid is not None else _uuid.uuid4().hex
    data = json.loads(tesselator.ExportShapeToThreejsJSONString(_ + "_geometry"))
    del data["metadata"]
    return AMesh(uuid=_ + "_mesh", geometry=create_buffer_from_dict(data),
                 material=MeshPhongMaterial(color=ColorRGB(*color).decimal))



class Tessellate(metaclass=ABCMeta):
    def __init__(self, shape, name, color):
        super().__init__()

        self.tess = ShapeTesselator(shape)
        self._name = name
        self.color = color
        self.generate_material()

    def tessellate(self, compute_edges=False, mesh_quality=1.0, parallel=False):

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




def parametric_mesh_bruteforce(prm,uv=[10, 4], closed=False):

    u,v = uv
    vertices=[]
    _vertices=[]
    _faces=[]
    dirs=[]
    for ik,i in enumerate(np.linspace(1,0,u)):
        points = []

        for jk, j in enumerate(np.linspace(0,1,v)):

            #pt=APoints(name=f"{ik}-{jk}",geometry=point,uuid=f'{uuid}-u{ik}-v{jk}', material=PointsMaterial(color=color, size=0.2))
            #grp.add(pt)
            point=prm.evaluate([i,j])
            vertices.append(point)
            _vertices.append(f"{ik}-{jk}")
            _faces.append((f"{ik}-{jk}",f"{ik+1}-{jk}",f"{ik+1}-{jk+1}"))
            points.append(point)

        if closed:
            dirs.append(np.array(points[-1])-np.array(points[0]))
        else:
            dirs.append(np.array(points[0]) - np.array(points[1]))

    dirsv=[]
    for ik,i in enumerate(np.linspace(0, 1, u)):
        points = []

        for jk,j in enumerate(np.linspace(1,0,v)):
            point=prm.evaluate([i,j])
            #pt = APoints(name=f"{ik}-{jk}", geometry=point, uuid=f'{uuid}-v{jk}-u{ik}',material=PointsMaterial(color=color))
            #grp.add(pt)

            points.append(point)
            _faces.append((f"{ik}-{jk}", f"{ik+1}-{jk+1}", f"{ik}-{jk + 1}"))
            _vertices.append(f"{ik}-{jk}")
        if closed:
            dirsv.append(np.array(points[-1])-np.array(points[0]))
        else:
            dirsv.append(np.array(points[0])-np.array(points[1]))

    indices=[]
    normals=[]
    for du in dirs:
        for dv in dirsv:
            normals.append(np.cross(unit(du),unit(dv)))
    for face in _faces:
        fc=[]
        try:

            for v in face:

                fc.append(_vertices.index(v))
            indices.append(fc)
        except:
                #print(face)
                pass

    return MeshData(vertices=vertices, indices=indices, normals=np.array(normals).flatten())

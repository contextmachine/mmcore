import dataclasses
import functools
import typing

import numpy as np
from mmcore.base.geom import MeshData
from mmcore.geom.csg import BspPolygon, BspVertex, CSG
from mmcore.geom.vectors import unit
from more_itertools import flatten


@dataclasses.dataclass(unsafe_hash=True)
class NurbsTesselationResult:
    mesh: MeshData
    csg: typing.Optional[CSG] = None


@functools.lru_cache(512)
def tessellate_nurbs(self, build_bsp=False):
    """
    This algorithm is good for sweep and pipe based on nurbs.

    @param self:
    @param build_bsp:
    @return:
    """
    lcpts = list(flatten(self.cpts))
    indices = []
    node = self.cpts.head
    normals = []
    polys = []
    for i in range(len(self.cpts)):
        nodeV = node.data.head
        nodeV2 = node.next.data.head

        for j in range(len(node.next.data)):
            d1 = np.array(nodeV2.data) - np.array(nodeV.data)
            d2 = np.array(nodeV2.next.data) - np.array(nodeV.data)
            norm = np.cross(unit(d1), unit(d2))
            normals.append(norm)

            a, b, c = [nodeV.data, nodeV2.data, nodeV.next.data]
            d, e, f = [nodeV2.next.data, nodeV.next.data, nodeV2.data]

            indices.extend([lcpts.index(i) for i in [a, b, c, d, e, f]])
            if build_bsp:
                polys.append(BspPolygon([BspVertex(pos=a), BspVertex(pos=b), BspVertex(pos=c)]))
                polys.append(BspPolygon([BspVertex(pos=d), BspVertex(pos=e), BspVertex(pos=f)]))
            nodeV = nodeV.next
            nodeV2 = nodeV2.next

        node = node.next

    mesh_data = MeshData(vertices=lcpts, indices=np.array(indices).reshape((len(indices) // 3, 3)),
                         normals=normals)
    csg = None
    if build_bsp:
        csg = CSG.fromPolygons(polys)
    return NurbsTesselationResult(mesh_data, csg)

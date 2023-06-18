import typing

import numpy as np
from mmcore.base.components import Component
from mmcore.base.geom import MeshData
from mmcore.collections import DCLL
from mmcore.geom.csg import CSG
from mmcore.geom.materials import ColorRGB
from mmcore.geom.parametric import NurbsCurve
from mmcore.geom.parametric.pipe import Pipe
from mmcore.geom.vectors import unit
from more_itertools import flatten


class Sweep(Component):
    profiles: list
    path: typing.Any
    color: tuple = (200, 10, 10)

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def tesselate(self):

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
                from mmcore.geom.csg import BspPolygon, BspVertex
                indices.extend([lcpts.index(i) for i in [a, b, c, d, e, f]])
                polys.append(BspPolygon([BspVertex(pos=a), BspVertex(pos=b), BspVertex(pos=c)]))
                polys.append(BspPolygon([BspVertex(pos=d), BspVertex(pos=e), BspVertex(pos=f)]))
                nodeV = nodeV.next
                nodeV2 = nodeV2.next

            node = node.next

        self._mesh_data = MeshData(vertices=lcpts, indices=np.array(indices).reshape((len(indices) // 3, 3)),
                                   normals=normals)
        self._csg = CSG.fromPolygons(polys)

    def solve_ll(self, bzz, prf, prf2):
        # xxx = bzz(points=self.path.points)
        pp = Pipe(bzz, prf)
        pp2 = Pipe(bzz, prf2)

        self.cpts = DCLL()
        for p in np.linspace(1, 0, 100):
            pl = DCLL()
            for pt in pp.evaluate_profile(p).control_points:
                pl.append(pt)
            self.cpts.append(pl)
        for p2 in np.linspace(0, 1, 100):
            pl = DCLL()
            for pt in pp2.evaluate_profile(p2).control_points:
                pl.append(pt)
            self.cpts.append(pl)
        return self.cpts

    def __call__(self, **kwargs):
        super().__call__(**kwargs)

        self.solve_ll(self.path, NurbsCurve(self.profiles[0], degree=1), NurbsCurve(self.profiles[1], degree=1))
        self.tesselate()
        self._repr3d = self._mesh_data.to_mesh(_endpoint=f"params/node/{self.param_node.uuid}",
                                               controls=self.param_node.todict(), uuid=self.uuid,
                                               color=ColorRGB(*self.color).decimal, opacity=0.5)
        self._repr3d.path = self.path
        return self

import numpy as np
from more_itertools import flatten


from mmcore.base.components import Component
from mmcore.collections import DCLL
from mmcore.geom.materials import ColorRGB
from mmcore.geom.parametric import NurbsCurve
from mmcore.geom.parametric.pipe import Pipe

from mmcore.geom.vectors import unit


class Sweep(Component):
    __exclude__ = ["cpts"]
    profiles: list
    path: NurbsCurve
    color: tuple = (200, 10, 10)
    opacity: float = 0.8
    density: int = 100

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def tessellate(self):

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

    @property
    def length(self):
        return self.path.length

    def solve(self):
        # xxx = path(points=self.path.points)
        pp = Pipe(self.path, NurbsCurve(self.profiles[0]),degree=1)
        pp2 = Pipe(self.path, NurbsCurve(self.profiles[1]),degree=1)

        self.cpts = DCLL()

        for p in np.linspace(1, 0, self.density):
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

        self.solve()
        self.tessellate()
        self.__repr3d__()
        self._repr3d = self._mesh_data.to_mesh(_endpoint=f"params/node/{self.param_node.uuid}",
                                               controls=self.param_node.todict(),
                                               uuid=self.uuid,
                                               color=ColorRGB(*self.color).decimal,
                                               opacity=self.opacity,
                                               name=self.name,
                                               properties={
                                                   "name": self.name,
                                                   "length": self.path.length
                                               })
        self._repr3d.path = self.path
        return self

    def __repr3d__(self):

        return self._repr3d

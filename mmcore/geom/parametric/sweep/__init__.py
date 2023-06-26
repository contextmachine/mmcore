import numpy as np
from mmcore.base.components import Component
from mmcore.collections import DCLL
from mmcore.geom.materials import ColorRGB
from mmcore.geom.parametric import NurbsCurve
from mmcore.geom.parametric.pipe import Pipe
from mmcore.geom.tess.nurbs import tessellate_nurbs


class Sweep(Component):
    profiles: list
    path: NurbsCurve
    color: tuple = (70, 155, 155)
    opacity: float = 1
    sides: int = 100

    def __new__(cls, *args, **kwargs):
        return super().__new__(cls, *args, **kwargs)

    def tessellate(self, build_bsp=False):

        tess = tessellate_nurbs(self, build_bsp=build_bsp)
        self._mesh_data, self._csg = tess.mesh, tess.csg

    def solve(self, bzz, prf, prf2):
        # xxx = bzz(points=self.path.points)
        pp = Pipe(bzz, prf)
        pp2 = Pipe(bzz, prf2)

        self.cpts = DCLL()
        for p in np.linspace(1, 0, self.sides):
            pl = DCLL()
            for pt in pp.evaluate_profile(p).control_points:
                pl.append(pt)
            self.cpts.append(pl)
        for p2 in np.linspace(0, 1, self.sides):
            pl = DCLL()
            for pt in pp2.evaluate_profile(p2).control_points:
                pl.append(pt)
            self.cpts.append(pl)
        return self.cpts

    def __call__(self, **kwargs):

        super().__call__(**kwargs)

        self.solve(self.path, NurbsCurve(self.profiles[0], degree=1), NurbsCurve(self.profiles[1], degree=1))
        self.tessellate()
        self.__repr3d__()
        self._repr3d.path = self.path
        return self

    def __repr3d__(self):
        self._repr3d = self._mesh_data.to_mesh(_endpoint=f"params/node/{self.param_node.uuid}",
                                               controls=self.param_node.todict(), uuid=self.uuid,
                                               color=ColorRGB(*self.color).decimal, opacity=self.opacity)
        return self._repr3d

import json

import numpy as np

from mmcore.geom.csg import *
from collections import namedtuple
from typing import TypedDict
from mmcore.base import A, AGroup

ConeInput = namedtuple("ConeInput", ["start", "end", "radius", "slices"])
CubeInput = namedtuple("CubeInput", ["center", "u", "v", "h"])

import mmcore
from mmcore.geom.transform import Transform, add_w, remove_crd
from mmcore.geom.parametric.sketch import PlaneLinear, line_plane_collision, Linear
from mmcore.geom.parametric.algorithms import ProximityPoints
from mmcore.geom.csg import CSG
from mmcore.collections import DCLL


def pt(pts):
    return pts.X, pts.Y, pts.Z


def pts(rpts):
    for rpt in rpts:
        yield pt(rpt)


class CSGTube(CSG):
    radius: float
    buffer: float
    width: float = 2
    extends: tuple[float, float] = (0, 0)
    slices: int = 32

    def __init__(self, start, end, slices=32, buffer=1, radius=10, **kwargs):
        super().__init__()
        self.slices = slices
        self.buffer = buffer
        self.axis = Linear.from_two_points(start, end)
        self.radius = radius
        self.__dict__ |= kwargs
        self.__call__( **kwargs)

    def __call__(self, **kwargs):
        """
        start, end, radius=30, buffer=0.5, width=5, extends=(0,0), slices=32
        @param start:
        @param end:
        @param radius:
        @param buffer:
        @param width:
        @param extends:
        @param slices:
        @return:
        """
        start, end = kwargs.get('start'), kwargs.get('end')
        #print(kwargs.get('start'), kwargs.get('end'))
        if kwargs.get('start') is not None and end is None:
            self.axis = Linear.from_two_points(start, self.axis.end).extend(*self.extends)
        elif kwargs.get('start') is not None and kwargs.get('end') is not None:
            self.axis = Linear.from_two_points(start, end).extend(*self.extends)
        elif kwargs.get('end') is not None:
            self.axis = Linear.from_two_points(self.axis.start, end).extend(*self.extends)

        self.__dict__ |= kwargs
        e, s = self.extends
        #print(self.axis.evaluate(s), self.axis.evaluate(1 + e))
        self._inner_cylinder = self.cylinder(start=self.axis.evaluate(s).tolist(),
                                             end=self.axis.evaluate(1 + e).tolist(),
                                             radius=self.radius - self.width,
                                             slices=self.slices)
        self._cylinder = self.cylinder(start=self.axis.evaluate(s).tolist(),
                                       end=self.axis.evaluate(1 + e).tolist(),
                                       radius=self.radius,
                                       slices=self.slices)
        self.polygons = (self._cylinder - self._inner_cylinder).polygons
        self._outer_cylinder = self.cylinder(start=self.axis.evaluate(s).tolist(),
                                             end=self.axis.evaluate(1 + e).tolist(),
                                             radius=self.radius + self.buffer,
                                             slices=self.slices)

        return self

    def __isub__(self, other):
        return other - self._outer_cylinder


class Joint(CSGTube):
    width: float = 0.5

    def __init__(self, tube1, tube2, **kwargs):

        self.tube1, self.tube2 = tube1, tube2
        p1 = ProximityPoints(self.tube1.axis,  self.tube2.axis)(x0=(0.5, 0.5), bounds=([-1, 1], [-1, 1]))
        super().__init__(*p1.pt,extends=(-1.5, 1.5), **kwargs)

        self.__call__( **kwargs)

    def __call__(self, **kwargs):
        self.__dict__|=kwargs
        super().__call__( **kwargs)
        self.tube2.polygons = (self.tube2 - self._outer_cylinder).polygons
        self.tube1.polygons = (self.tube1 - self._outer_cylinder).polygons
        return self


if __name__ == '__main__':
    lines = np.array([
        [[-7.992849, -25.166169, -0.593938], [-15.829974, -1.264115, -0.593938]],
        [[-14.032843, -13.489682, -4.350191], [3.421241, -18.660692, 8.936641]]])*10
    grp = AGroup(name="Fragment")
    tubes = []


    def mesher(obj, **kwargs):
        grp2 = AGroup()
        for p in obj.toPolygons():
            grp2.add(p.mesh(**kwargs))
        return grp2



    a=CSGTube(start=lines[0][0],
                end=lines[0][1],
                radius=6,
                slices=32)

    b=CSGTube(start=lines[1][0],
                end=lines[1][1],
                radius=6,
                slices=32)
    j = Joint(a,b, radius=4.0, width=1.5, buffer=0.5)
    grp.add(mesher(a, material=MeshPhongMaterial(color=ColorRGB(40,100,140).decimal)))
    grp.add(mesher(b,material=MeshPhongMaterial(color=ColorRGB(40,100,140).decimal)))
    grp.add(mesher(j, material=MeshPhongMaterial(color=ColorRGB(200,100,40).decimal)))

    # grp.add(tubes[0].mesh_data().to_mesh())
    # grp.add(tubes[1].mesh_data().to_mesh())
    grp.dump("model.json")

from mmcore.base.sharedstate import serve










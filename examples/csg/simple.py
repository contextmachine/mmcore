import numpy as np

from mmcore.geom.csg import *
from collections import namedtuple
from typing import TypedDict
from mmcore.base import A, AGroup

ConeInput = namedtuple("ConeInput", ["start", "end", "radius", "slices"])
CubeInput = namedtuple("CubeInput", ["center", "u", "v", "h"])

import mmcore
from mmcore.geom.transform import Transform, add_w, remove_crd
from mmcore.geom.parametric.sketch import PlaneLinear, line_plane_collision, ProximityPoints, Linear
from mmcore.geom.csg import CSG
from mmcore.collections import DCLL


def pt(pts):
    return pts.X, pts.Y, pts.Z


def pts(rpts):
    for rpt in rpts:
        yield pt(rpt)



class CSGTube(CSG):
    radius: int
    width: int
    extends: tuple[float, float]
    def __new__(cls, start, end, radius=30, buffer=0.5, width=5, extends=(0,0), slices=32, **kwargs):


        inst=cls.cylinder( start=start, end=end, radius=radius, slices=slices)

        inst.axis = Linear.from_two_points(start, end)
        inst.__dict__|= dict( buffer=buffer, width=width, extends=extends, **kwargs)
        return inst

    def __call__(self, start=None, end=None, **kwargs):
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


        if start is not None and end is None:
            self.axis=Linear.from_two_points(start, self.axis.end)
        elif start is not None and end is not None:
            self.axis = Linear.from_two_points(start, end)
        elif end is not None:
            self.axis = Linear.from_two_points(self.axis.start, end)

        self.__dict__|=kwargs

        self._inner_cylinder = self.cylinder(start=self.axis.evaluate(start),
                                             end=self.axis.evaluate(end),
                                             radius=self.radius-self.width,
                                             slices=self.slices)
        self._outer_cylinder = self.cylinder(start=self.axis.evaluate(start),
                                             end=self.axis.evaluate(end),
                                             radius=self.radius+self.buffer,
                                             slices=self.slices)
        self-self._inner_cylinder
        return self

    def __isub__(self, other):
        return other-self._outer_cylinder


def joint(tube1, tube2):
    p1 = ProximityPoints(tube1.axis, tube2.axis)(x0=(0.5, 0.5),
                                                 bounds=([-1, 1], [-1, 1]))

    joint1 = CSGTube(*p1.pt)
    tube1 - joint1
    tube2 - joint1
    return joint1


if __name__=='__main__':
    lines=[
        [[-7.992849, -25.166169, -0.593938],[-15.829974, -1.264115, -0.593938]],
        [[-14.032843, -13.489682, -4.350191], [3.421241, -18.660692, 8.936641]],
        [[-0.647457, -25.166169, 7.870181],[-6.03231, 1.876652, 5.012073]]]

    tubes=[]
    for line in lines:
        start,end=line
        tubes.append(CSGTube(start, end))

    j1 = joint(*tubes[:2])
    j2 = joint(*tubes[1:])

from mmcore.base.sharedstate import serve

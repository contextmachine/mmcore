import numpy as np

from mmcore.geom.curves import ParametricPlanarCurve


class Circle(ParametricPlanarCurve, match_args=('r',), signature='()->(i)'):
    def __init__(self, r=1, origin=None, plane=None):
        super().__init__(origin, plane)

        self.r = r

    @property
    def a(self):
        return self.r

    @a.setter
    def a(self, v):
        self.r = v

    def x(self, t):
        return self.r * np.cos(t)

    def y(self, t):
        return self.r * np.sin(t)

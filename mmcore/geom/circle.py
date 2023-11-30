import numpy as np

from mmcore.geom.curves import ParametricPlanarCurve


class Circle(ParametricPlanarCurve, match_args=('r',), signature='()->(i)'):
    def __new__(cls, r=1, origin=None, plane=None):
        self = super().__new__(cls, origin=origin, plane=plane)
        self.r = r
        return self

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

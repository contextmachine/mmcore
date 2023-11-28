import functools

import numpy as np

from mmcore.geom.curves import ParametricPlanarCurve
from mmcore.geom.vectors import norm

__all__ = ['Ellipse', 'p']


class Ellipse(ParametricPlanarCurve, match_args=('a', 'b',), signature='()->(i)', param_range=(0.0, np.pi * 2)):
    def __new__(cls, a=1, b=1, origin=None, plane=None):
        self = super().__new__(cls, origin=origin, plane=plane)

        self.a = a
        self.b = b
        return self

    def x(self, t):
        return self.a * np.cos(t)

    def y(self, t):
        return self.b * np.sin(t)

    @property
    def xaxis(self):
        return self.plane.xaxis * self.a

    @property
    def yaxis(self):
        return self.plane.yaxis * self.b

    @functools.cached_property
    def c(self):
        return np.sqrt(self.a ** 2 - self.b ** 2)

    def foci(self):
        return self.to_world(self._ellipse_component('c'))

    def _ellipse_component(self, name: str):
        h, k = self.origin[0], self.origin[1]
        component = getattr(self, name)
        if norm(self.xaxis) > norm(self.yaxis):
            return np.array([(h - component, k, 0.), (h + component, k, 0.)])
        else:
            return np.array([(h, k - component, 0.), (h, k + component, 0.)])

    def vertices(self):
        return self._ellipse_component(self, 'a')


def p(self: Ellipse, alpha):
    return self.a * np.cos(alpha), self.a * np.sin(alpha), 0.0

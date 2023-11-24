import functools

import numpy as np
from multipledispatch import dispatch

from mmcore.geom.curves import ParametricPlanarCurve
from mmcore.geom.vectors import norm

__all__ = ['Ellipse', 'p', 'foci', 'vertices']


class Ellipse(ParametricPlanarCurve, match_args=('a', 'b',), signature='()->(i)'):
    def __init__(self, a=1, b=1, origin=None, plane=None):
        super().__init__(origin, plane)

        self.a = a
        self.b = b

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


def _ellipse_component(self: Ellipse, name: str):
    h, k = self.origin[0], self.origin[1]
    component = getattr(self, name)
    if norm(self.xaxis) > norm(self.yaxis):
        return np.array([(h - component, k), (h + component, k)])
    else:
        return np.array([(h, k - component), (h, k + component)])


@dispatch(Ellipse)
def p(self: Ellipse, alpha):
    return self.a * np.cos(alpha), self.a * np.sin(alpha), 0.0


@dispatch(Ellipse)
def foci(self: Ellipse):
    return self.to_world(_ellipse_component(self, 'c'))


@dispatch(Ellipse)
def vertices(self: Ellipse):
    return _ellipse_component(self, 'a')

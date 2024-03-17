import functools

import numpy as np

from mmcore.func import vectorize
from mmcore.geom.curves import ParametricPlanarCurve
from mmcore.geom.plane import create_plane
from mmcore.geom.vectors import norm

__all__ = ['Ellipse', 'p']


class Ellipse:
    """
    Create a new instance of the Ellipse class.

    Parameters:
    - a (float): The major radius of the ellipse. Defaults to 1.
    - b (float): The minor radius of the ellipse. Defaults to 1.
    - origin (tuple or None): The origin point of the ellipse in the form (x, y). Defaults to None.
    - plane (Plane or None): The plane in which the ellipse lies. Defaults to None.

    Returns:
    - Ellipse: The newly created Ellipse object.
    """
    def __new__(cls, a=1, b=1, origin=None, plane=None):

        self = super().__new__(cls)
        if origin is None:
            origin = np.array([0, 0, 0])
        if plane is None:
            self.plane = create_plane(origin=origin)
        else:
            self.plane = plane

        self.a = a
        self.b = b

        return self

    def evaluate(self, t):

        return self.plane.point_at(np.array([self.a * np.cos(t), self.b * np.sin(t), np.zeros_like(t)]))

    @vectorize(excluded=[0], signature='()->(j)')
    def __call__(self, t):
        return self.evaluate(t)
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


    def c(self):
        return np.sqrt(self.a ** 2 - self.b ** 2)

    def foci(self):

        return self.plane.point_at(self._ellipse_component('c'))

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
    """
    :param self: The Ellipse object on which the method is called.
    :type self: Ellipse
    :param alpha: The angle parameter.
    :type alpha: float
    :return: The x, y, z coordinates of a point on the ellipse for a given angle alpha.
    :rtype: tuple
    """
    return self.a * np.cos(alpha), self.b * np.sin(alpha), 0.0

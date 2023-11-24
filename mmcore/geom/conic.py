import numpy as np

from mmcore.geom.circle import Circle
from mmcore.geom.ellipse import Ellipse


class Parabola(Circle, signature='()->(i)'):

    def x(self, t):
        return self.r * (t ** 2)

    def y(self, t):
        return 2 * self.r * t


class Hyperbola(Ellipse, signature='()->(i)'):
    def x(self, t):
        return self.a * (1 / np.cos(t))

    def y(self, t):
        return self.b * np.tan(t)

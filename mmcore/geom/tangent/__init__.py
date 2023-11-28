import numpy as np

from mmcore.func import vectorize
from mmcore.geom.circle import Circle
from mmcore.geom.ellipse import Ellipse
from mmcore.geom.plane import plane
from mmcore.geom.vec import cross, unit


@vectorize(excluded=[0], signature='()->(i)')
def ellipse_tan(self: Ellipse, t):
    return unit([1, -1 * (self.b / self.a) * cot(t), 0])

@vectorize(excluded=[0], signature='()->(i)')
def ellipse_normal(self: Ellipse, t):
    zaxis = ellipse_tan(self, t)
    return np.array([-zaxis[1], zaxis[0], zaxis[2]])


@vectorize(excluded=[0], signature='()->()')
def ellipse_perp_plane(self: Ellipse, t):
    tn = ellipse_tan(self, t)
    nm = ellipse_normal(self, t)

    return plane(origin=self(t), xaxis=nm, yaxis=cross(tn, nm), zaxis=tn)


def ellipse_horizontal_plane(self: Ellipse, t):
    tn = ellipse_tan(self, t)
    nm = ellipse_normal(self, t)

    return plane(origin=self(t), xaxis=tn, yaxis=cross(tn, nm), zaxis=nm)

@vectorize(excluded=[0], signature='()->(i)')
def circle_tan(self: Circle, t):
    return unit(np.array([-self.y(t), self.x(t), self.z(t)]))


@vectorize(excluded=[0], signature='()->(i)')
def circle_normal(self: Circle, t):
    return unit(self.evaluate(t))


def cot(x):
    return 1 / np.tan(x)

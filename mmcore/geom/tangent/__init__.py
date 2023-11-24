import numpy as np

from mmcore.func import vectorize
from mmcore.geom.circle import Circle
from mmcore.geom.ellipse import Ellipse


@vectorize(excluded=[0], signature='()->(i)')
def _circle_tan(self: Circle, t):
    return np.array([-self.y(t), self.x(t), self.z(t)])


@vectorize(excluded=[0], signature='()->(i)')
def _ellipse_tan(self: Ellipse, t):
    return np.array([-self.y(t), self.x(t), self.z(t)])

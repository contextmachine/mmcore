import numpy as np

from mmcore.geom.parametric import ParametricSupport
from mmcore.geom.vec import unit


class Sphere(ParametricSupport, signature='(),()->(i)'):
    def __new__(cls, r=1, origin=(0.0, 0.0, 0.0)):
        self = super().__new__(cls)
        self.r = r
        self.origin = np.array(origin)
        self.__evaluate__ = np.vectorize(self.evaluate, signature='(),()->(i)')
        return self

    def evaluate_x(self, u, v):
        return self.r * np.sin(u) * np.cos(v) + self.origin[0]

    def evaluate_y(self, u, v):
        return self.r * np.sin(u) * np.sin(v) + self.origin[1]

    def evaluate_z(self, u, v):
        return self.r * np.cos(u) + self.origin[2]

    def evaluate(self, u, v):
        return np.array([self.evaluate_x(u, v), self.evaluate_y(u, v), self.evaluate_z(u, v)])

    def divide(self, u_count, v_count):
        return self(*np.meshgrid(np.linspace(0, np.pi, u_count),
                                 np.linspace(0, 2 * np.pi, v_count)
                                 )
                    )

    def __call__(self, u, v):
        return self.__evaluate__(u, v)

    def __iter__(self):
        return iter((self.r, self.origin))

    def __array__(self):
        return np.array((self.r, *self.origin))

    def project(self, pt):
        return unit(pt - self.origin) * self.r

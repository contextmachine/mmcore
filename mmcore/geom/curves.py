import numpy as np

from mmcore.geom.parametric import ParametricSupport


class ParametricPlanarCurve(ParametricSupport, signature='()->(i)'):

    def __new__(cls, origin=None, plane=None):
        return super().__new__(cls, origin=origin, plane=plane)

    def x(self, t) -> float:
        return ...

    def y(self, t) -> float:
        return ...

    def z(self, t) -> float:
        return 0

    def _evalx(self, t):
        return self.x(t)

    def _evaly(self, t):
        return self.y(t)

    def _evalz(self, t):
        return self.z(t)

    def _tan(self, t):
        return self.z(t)

    def evaluate(self, t) -> np.ndarray:
        return np.array([self._evalx(t), self._evaly(t), self._evalz(t)])

    def __call__(self, t):
        return self.to_world(self.__evaluate__(t))


from mmcore.base.geom import LineObject, PointsObject
from mmcore.geom.parametric.base import ParametricObject
import numpy as np


class ParametricCircle(LineObject, ParametricObject):
    r: float = 1
    h: float = 10

    def evaluate(self, t):
        u, v = t
        return np.array([self.r * np.cos(v * 2 * np.pi), self.r * np.sin(v * 2 * np.pi), np.sin(u) * self.h],
                        dtype=float).tolist()

    @property
    def points(self):
        self._points = []
        self.ll = []
        for i in range(20):
            self.ll += np.linspace([0 + 1 / 20 * i, 0], np.array([0 + 1 / 20 * i, 1]), 100).tolist()

        self._points.append(list(map(self.evaluate, self.ll)))

        return self._points


p = ParametricCircle(name="testcircle2", r=5)

from mmcore.geom.parametric.algorithms import ClosestPoint
from mmcore.base.registry import objdict
import mmcore


class ClPoint(PointsObject):
    _geometry = None
    target = p.uuid
    ans = None

    @property
    def points(self):
        return self._points

    @points.setter
    def points(self, v):
        objdict[self.target].solve_geometry()
        ans = ClosestPoint(v[0], objdict[self.target])([0.5, 0.5], bounds=[(0, 1), (0, 1)])
        self._points = ans.pt + v

        self.ans = ans
        self.solve_geometry()


cl1 = ClPoint(name="closest-point-1", points=[p.evaluate([0.5, 0.5])], color=(170, 200, 40))
cl2 = ClPoint(name="closest-point-2", points=[[-5.0, 0, 0]], color=(40, 200, 70))

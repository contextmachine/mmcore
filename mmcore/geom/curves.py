import copy
from functools import lru_cache

import numpy as np

from mmcore.geom.parametric import ParametricSupport
from mmcore.geom.plane import WXY, create_plane, world_to_local


class ParametricPlanarCurve(ParametricSupport, signature='()->(i)'):
    def __init_subclass__(cls, match_args=(), **kwargs):
        cls.__match_args__ = match_args
        super().__init_subclass__(**kwargs)

    def __init__(self, origin=None, plane=None):
        if origin is None:
            origin = np.array([0, 0, 0])
        if plane is None:
            self.plane = create_plane(origin=origin)
        else:
            self.plane = plane

    @property
    def origin(self):
        return self.plane.origin

    def x(self, t) -> float:
        return ...

    def y(self, t) -> float:
        return ...

    def z(self, t) -> float:
        return 0

    def _evalx(self, t):
        return self.x(t) + self.origin[0]

    def _evaly(self, t):
        return self.y(t) + self.origin[1]

    def _evalz(self, t):
        return self.z(t) + self.origin[2]

    def _tan(self, t):
        return self.z(t) + self.origin[2]

    def evaluate(self, t) -> np.ndarray:
        return np.array([self._evalx(t), self._evaly(t), self._evalz(t)])

    @lru_cache(maxsize=None)
    def __call__(self, t):
        return self.to_world(self.__evaluate__(t))

    @lru_cache(maxsize=None)
    def to_world(self, value):
        return world_to_local(value, self.plane) if self.plane != WXY else value

    @lru_cache(maxsize=None)
    def to_local(self, value, plane=WXY):
        return world_to_local(value, self.plane) if self.plane != plane else value

    def copy_to_new_plane(self, plane):
        other = copy.copy(self)
        other.plane = plane
        return other

    def deepcopy_to_new_plane(self, plane):
        other = copy.deepcopy(self)
        other.plane = plane
        return other

    def __iter__(self):
        return (self.__dict__[arg] for arg in self.__match_args__)

    def __hash__(self):
        return hash(tuple(self.__iter__()))

    def __eq__(self, other):
        if self.__match_args__ != other.__match_args__:
            return False
        else:
            return self.__hash__() == other.__hash__()

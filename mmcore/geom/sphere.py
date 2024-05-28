import numpy as np

from mmcore.geom.implicit import Implicit3D
from mmcore.geom.surfaces import Surface

_IDENTITY = np.eye(3)
PI05 = 1.5707963267948966
PI2 = 6.283185307179586


class Sphere(Surface, Implicit3D):
    """



    """

    def __init__(self, origin=None, r=1.):
        super().__init__()

        self.origin = (
            origin if isinstance(origin, np.ndarray) else np.array(origin)) if origin is not None else np.zeros(3,
                                                                                                                dtype=float)
        self.r = r

    def implicit(self, v) -> float:
        return np.linalg.norm(v - self.origin) - self.r

    def _implicit_normal(self, v) -> float:
        return v - self.origin

    def _implicit_unit_normal(self, v) -> float:
        point = self._implicit_normal(v)
        return point / np.linalg.norm(point)

    def normal(self, val):
        return [lambda x: ValueError(f"{x} can not interpret as uv and  xyz"),
                lambda x: ValueError(f"{x} can not interpret as uv and  xyz"),
                super(Surface, self).normal,
                self._implicit_unit_normal][val.shape[-1]](val)


    def evaluate(self, uv):
        return np.zeros(3)

    def closest_point(self, v):

        N = self._implicit_normal(v)
        nn = np.linalg.norm(N)
        nu = N / nn
        return v - (nu * (nn - self.r)), np.arccos(nu[0])

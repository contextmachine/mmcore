import numpy as np

from mmcore.geom.parametric import ParametricSupport


class ParametricSurface(ParametricSupport, signature='(),()->(i)', param_range=((0.0, 1.0), (0.0, 1.0))):
    def x(self, u, v) -> float:
        return ...

    def y(self, u, v) -> float:
        return ...

    def z(self, u, v) -> float:
        return 0

    def evaluate(self, u, v) -> np.ndarray:
        return np.array([self.x(u, v), self.y(u, v), self.z(u, v)])

    def divide(self, u_count, v_count):
        return self(*np.meshgrid(self.range[0].divide(u_count), self.range[1].divide(v_count)
                                 )
                    )

    def __call__(self, u, v):
        return self.to_world(self.__evaluate__(u, v))

import numpy as np


class ParametricSupport:
    def __init_subclass__(cls, signature='()->(i)'):
        cls.__np_vec_signature__ = signature

    def __new__(cls):
        self = super().__new__(cls)
        self.__vevaluate__ = np.vectorize(self.evaluate, signature=cls.__np_vec_signature__)
        return self

    def evaluate(self, *args) -> np.ndarray:
        return ...


class Sphere(ParametricSupport, signature='(),()->(i)'):
    def __new__(cls, r=1, origin=(0.0, 0.0, 0.0)):
        self = super().__new__(cls)
        self.r = r
        self.origin = np.array(origin)
        self.__vevaluate__ = np.vectorize(self.evaluate, signature='(),()->(i)')
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
        return self(*np.meshgrid(np.linspace(0, np.pi, u_count), np.linspace(0, 2 * np.pi, v_count)))

    def __call__(self, u, v):
        return self.__vevaluate__(u, v)

    def __iter__(self):
        return iter((self.r, self.origin))

    def __array__(self):
        return np.array((self.r, *self.origin))

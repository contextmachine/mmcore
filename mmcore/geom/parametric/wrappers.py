import abc


class AbstractParametricCurve:
    def __init__(self, obj, bounds=(0.0, 1.0)):
        super().__init__()
        self.obj = obj
        self.bounds = list(bounds)

    def __call__(self, t):
        return self.solve(self.obj, t)

    @abc.abstractmethod
    def solve(self, obj, t) -> list[float]:
        ...


class BaseParametricCurve(AbstractParametricCurve):
    def solve(self, obj, t) -> list[float]:
        return obj.evaluate(t)

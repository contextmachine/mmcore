import abc
import dataclasses
import typing
from collections import namedtuple

import numpy as np
from scipy.optimize import minimize, fsolve
from scipy.spatial.distance import euclidean

from mmcore import TOLERANCE
ClosestPointSolution = namedtuple("ClosestPointSolution", ["pt", "t", "distance"])
IntersectSolution = namedtuple("IntersectSolution", ["pt", "t", "is_intersect"])
IntersectFail = namedtuple("IntersectFail", ["pt", "t", "distance", "is_intersect"])
MultiSolutionResponse = namedtuple("MultiSolutionResponse", ["pts", "roots"])

@dataclasses.dataclass
class EvaluatedPoint:
    point: list[float]
    normal: typing.Optional[list[float]]
    direction: typing.Optional[list[typing.Union[float, list[float]]]]
    t: typing.Optional[list[typing.Union[float, list[float]]]]


class MinimizeSolution:
    solution_response: typing.Any

    @abc.abstractmethod
    def solution(self, t):
        ...

    def __call__(self,
                 x0: np.ndarray = np.asarray([0.5, 0.5]),
                 bounds: typing.Optional[typing.Iterable[tuple[float, float]]] = ((0, 1), (0, 1)),
                 *args,
                 **kwargs):
        res = minimize(self.solution, x0, bounds=bounds, **kwargs)
        return self.prepare_solution_response(res)

    def __init_subclass__(cls, solution_response=None, **kwargs):
        cls.solution_response = solution_response
        super().__init_subclass__(**kwargs)

    @abc.abstractmethod
    def prepare_solution_response(self, solution):
        ...




class ProximityPoints(MinimizeSolution, solution_response=ClosestPointSolution):
    """

    >>> a=[[9.258697, -8.029476, 0],
    ...    [6.839202, -1.55593, -6.390758],
    ...    [18.258577, 16.93191, 11.876064],
    ...    [19.834301, 27.566156, 1.173745],
    ...    [-1.257139, 45.070784, 0]]
    >>> b=[[27.706367, 29.142311, 6.743523],
    ...    [29.702408, 18.6766, 19.107107],
    ...    [15.5427, 6.960314, 10.273386],
    ...    [2.420935, 26.07378, 18.666591],
    ...    [-3.542004, 3.424012, 11.066738]]
    >>> A,B=NurbsCurve(control_points=a),NurbsCurve(control_points=b)
    >>> prx=ProxPoints(A,B)
    >>> prx()
    ClosestPointSolution(pt=[array([16.27517685, 16.07437063,  4.86901707]), array([15.75918043, 14.67951531, 14.57947997])], t=array([0.52562605, 0.50105099]), distance=9.823693977393207)
    """

    def __init__(self, c1, c2):
        super().__init__()
        self.c1, self.c2 = c1, c2

    def solution(self, t) -> float:
        t1, t2 = t

        return euclidean(self.c1.evaluate(t1), self.c2.evaluate(t2))

    def prepare_solution_response(self, solution):
        t1, t2 = solution.x
        return self.solution_response([self.c1.evaluate(t1),
                                       self.c2.evaluate(t2)],
                                      solution.x,
                                      solution.fun)

    def __call__(self, x0: np.ndarray = np.asarray([0.5, 0.5]),
                 bounds: typing.Optional[typing.Iterable[tuple[float, float]]] = ((0, 1), (0, 1)),
                 *args,
                 **kwargs):
        res = minimize(self.solution, x0, bounds=bounds, **kwargs)

        return self.prepare_solution_response(res)


ProxPoints = ProximityPoints  # Alies for me


class MultiSolution(MinimizeSolution, solution_response=MultiSolutionResponse):
    @abc.abstractmethod
    def solution(self, t): ...

    def __call__(self,
                 x0: np.ndarray = np.asarray([0.5, 0.5]),

                 **kwargs):
        res = fsolve(self.solution, x0, **kwargs)
        return self.prepare_solution_response(res)

    @abc.abstractmethod
    def prepare_solution_response(self, solution):
        ...


class ClosestPoint(MinimizeSolution, solution_response=ClosestPointSolution):
    """
    >>> a= [[-25.0, -25.0, -5.0],
    ... [-25.0, -15.0, 0.0],
    ... [-25.0, -5.0, 0.0],
    ... ...
    ... [25.0, 15.0, 0.0],
    ... [25.0, 25.0, -5.0]]
    >>> srf=NurbsSurface(control_points=a,size_u=6,size_v=6)
    >>> pt=np.array([13.197247, 21.228605, 0])
    >>> cpt=ClosestPoint(pt, srf)
    >>> cpt(x0=np.array((0.5,0.5)), bounds=srf.domain)

    ClosestPointSolution(pt=[array([13.35773142, 20.01771329, -7.31090389])], t=array([0.83568967, 0.9394131 ]), distance=7.41224187927431)
    """

    def __init__(self, point, geometry):
        super().__init__()
        self.point, self.gm = point, geometry

    def solution(self, t):
        r = self.gm.evaluate(t)
        r = np.array(r).T.flatten()
        # #print(self.point, r)

        return euclidean(self.point, r)

    def prepare_solution_response(self, solution):
        return self.solution_response([self.gm.evaluate(solution.x)], solution.x, solution.fun)

    def __call__(self, x0=(0.5,), **kwargs):
        return super().__call__(x0, **kwargs)


class CurveCurveIntersect(ProximityPoints, solution_response=ClosestPointSolution):
    def __init__(self, c1, c2):
        super().__init__(c1, c2)

    def __call__(self, *args, tolerance=TOLERANCE, **kwargs):
        r = super().__call__(*args, **kwargs)
        is_intersect = np.allclose(r.distance, 0.0, atol=tolerance)
        if is_intersect:
            return IntersectSolution(np.mean(r.pt, axis=0), r.t, is_intersect)

        else:
            return IntersectFail(r.pt, r.t, r.distance, is_intersect)

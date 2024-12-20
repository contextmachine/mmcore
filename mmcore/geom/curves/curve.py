from __future__ import annotations

import abc
import inspect
import math
import operator
from functools import lru_cache
from typing import Union, Any, Callable

import numpy as np
from numpy._typing import ArrayLike

from mmcore.geom.bvh import PSegment, build_bvh
from mmcore.numeric import circle_of_curvature
from mmcore.numeric.vectors import scalar_unit,scalar_cross
unit=scalar_unit
cross=scalar_cross
from mmcore.numeric.numeric import normal_at, evaluate_tangent, evaluate_curvature, plane_on_curve, divide_interval, \
    evaluate_length

from mmcore.numeric.fdm import DEFAULT_H, fdm

TOLERANCE = 1e-4


class Curve:
    """
    Base Curve with Parametric Form Support
    """
    _tree = None
    _interval = None

    def __init__(self, init_derivatives=True):
        super().__init__()
        self._interval = (0., 1.)
        self.evaluate_multi = np.vectorize(self.evaluate, signature="()->(i)", doc="""
        Evaluate points on the curve at t values array
        :param t: np.ndarray[float] with shape N
        :return: array of points np.ndarray[float] with shape (N, M) where M 2 or 3 in 2d and 3d cases
        """)
        if init_derivatives:
            self._derivatives = [self, self.derivative]
            self.add_derivative()
            self._derivatives[2].__doc__ = """
        :param t:float
        :return: vector of second derivative  as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.
        :rtype: np.ndarray[(3,), np.dtype[float]]\n"""

        self._evaluate_cached = lru_cache(maxsize=None)(self.evaluate)
        self._evaluate_length_cached = lru_cache(maxsize=None)(self._evaluate_length)

    def invalidate_cache(self):
        """
           Invalidates the cache.

           :return: None
        """

        self._evaluate_cached.cache_clear()
        self._evaluate_length_cached.cache_clear()

    @property
    def tree(self):
        if getattr(self, '_tree', None) is None:
            self.build_tree()

        return self._tree

    def points(self, count=50):
        return np.array(self.evaluate_multi(np.linspace(*self.interval(), count)))

    #def intersect_with_curve(
    #        self, other: "Curve", tol=TOLERANCE
    #) -> list[tuple[float, float]]:
    #
    #    """
    #    PPI & PII
    #    ------
    #
    #    PPI (Parametric Parametric Intersection) for the curves.
    #    curve1 and curve2 can be any object with a parametric curve interface.
    #    However, in practice it is worth using only if both curves do not have implicit representation,
    #    most likely they are two B-splines or something similar.
    #    Otherwise it is much more efficient to use PII (Parametric Implict Intersection).
    #
    #    The function uses a recursive divide-and-conquer approach to find intersections between two curves.
    #    It checks the AABB overlap of the curves and recursively splits them until the distance between the curves is within
    #    the specified tolerance or there is no overlap. The function returns a list of tuples
    #    representing the parameter values of the intersections on each curve.
    #
    #    Обратите внимание! Этот метод продолжает "Разделяй и властвуй" пока расстояние не станет меньше погрешности.
    #    Вы можете значительно ускорить поиск, начиная метод ньютона с того момента где для вас это приемлимо.
    #    Однако имейте ввиду что для правильной сходимости вы уже должны быть в "низине" с одним единственым минимумом.
    #
    #    :param curve1: first curve
    #    :param curve2: second curve
    #    :param bounds1: [Optional] custom bounds for first NURBS curve. By default, the first NURBS curve interval.
    #    :param bounds2: [Optional] custom bounds for first NURBS curve. By default, the second NURBS curve interval.
    #    :param tol: A pair of points on a pair of Euclidean curves whose Euclidean distance between them is less than tol will be considered an intersection point
    #    :return: List containing all intersections, or empty list if there are no intersections. Where intersection
    #    is the tuple of the parameter values of the intersections on each curve.
    #    :rtype: list[tuple[float, Optional[float]]]
    #
    #    Example
    #    --------
    #    >>> import numpy as np
    #    >>> from mmcore.geom.curves.bspline import NURBSpline
    #    >>> from mmcore.numeric.intersection.curve_curve import curve_ppi
    #    >>> first = NURBSpline(
    #    ...    np.array(
    #    ...        [
    #    ...            (-13.654958030023677, -19.907874497194975, 0.0),
    #    ...            (3.7576433265207765, -39.948793039632903, 0.0),
    #    ...            (16.324284871574083, -18.018771519834026, 0.0),
    #    ...            (44.907234268165922, -38.223959886390297, 0.0),
    #    ...            (49.260384607302036, -13.419216444520401, 0.0),
    #    ...        ]
    #    ...    )
    #    ... )
    #    >>> second = NURBSpline(
    #    ...     np.array(
    #    ...         [
    #    ...             (40.964758489325661, -3.8915666456564679, 0.0),
    #    ...             (-9.5482124270650726, -28.039230791052990, 0.0),
    #    ...             (4.1683178868166371, -58.264878428828240, 0.0),
    #    ...             (37.268687446662931, -58.100608604709883, 0.0),
    #    ...         ]
    #    ...     )
    #    ... )
    #
    #    >>> intersections = curve_ppi(first, second, 0.001)
    #    >>> print(intersections)
    #    [(0.600738525390625, 0.371673583984375)]
    #
    #
    #    """
    #
    #    return curve_intersection(self, other, tol=tol)

    def interval(self):
        """
         Parametric domain of the curve.
        :param t:float
        :return: start and end parameter values

        """
        return self._interval

    def split(self, t):
        return (
            TrimmedCurve(self, self.interval()[0], t),
            TrimmedCurve(self, t, self.interval()[1])
        )

    def extend(self, start=0., end=0.):
        self._interval = (self._interval[0] - start, self._interval[1] + end)

    def normal(self, t):
        """
        :param t:float
        :return: normal vector as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.
        """
        return normal_at(self.derivative(t), self.second_derivative(t))

    def tangent(self, t):
        """
        Calculate tangent vector `T(t)` from the first `D1(t)` and second `D2(t)` derivatives at t parameter
        .. math::
            T(D1, D2)=\\dfrac{D2}{||D1||} \\text{ where }||D2|| \\neq{0}\\text{ else } \\dfrac{D1}{||D1||}
        where:

        D1 - first derivative vector \
        D2 - second derivative vector \
        T - tangent vector \

        :param t: float
        :return: tangent vector as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.

        """

        return evaluate_tangent(self.derivative(t), self.second_derivative(t))[0]

    def curvature(self, t):
        """
        Evaluate a curvature vector in a given parameter t
        :param t: float
        :return: vector on curve at t parameter value
            as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.
        """
        return evaluate_curvature(self.derivative(t), self.second_derivative(t))[1]

    def plane_at(self, t):
        """
        The plane formed by the tangent vector and normal vector at a given point
        :param t:float
        :return: normal vector as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.
        """
        return plane_on_curve(
            self.evaluate(t), self.tangent(t), self.second_derivative(t)
        )
    def planes_at(self, t):
        return np.array([self.plane_at(s) for s in t])
    def derivative(self, t):
        """
        :param t:float
        :return: vector of first derivative   as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.
        """
        if (1 - DEFAULT_H) >= t >= DEFAULT_H:
            return (
                    (self.evaluate(t + DEFAULT_H) - self.evaluate(t - DEFAULT_H))
                    / 2
                    / DEFAULT_H
            )
        elif t <= DEFAULT_H:
            return (self.evaluate(t + DEFAULT_H) - self.evaluate(t)) / DEFAULT_H
        else:
            return (self.evaluate(t) - self.evaluate(t - DEFAULT_H)) / DEFAULT_H

    def second_derivative(self, t):
        return self._derivatives[2](t)

    def add_derivative(self):

        self._derivatives.append(fdm(self._derivatives[-1]))
        return len(self._derivatives)

    def __call__(
            self, t: Union[np.ndarray[Any, float], float]
    ) -> np.ndarray[Any, np.dtype[float]]:
        return self.evaluate_multi(t)

    @abc.abstractmethod
    def evaluate(self, t: float):
        """
        Evaluate point on curve at t value
        :param t:float
        :return: point on curve at t parameter value
            as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.


        """
        ...

    def circle_of_curvature(self, t: float):
        """
        Calculates the circle of curvature at the given parameter t.
        :param t: The parameter value at which to calculate the circle of curvature.
        :type t: float
        :return: The circle of curvature at `t` as a tuple: (`Point at t`,  `radius`, `Plane of the circle`).

        The first element of the tuple represents the point on curve at `t` as a numpy array with shape (3,).
        The point is calculated as :
        >>> P = self.evaluate(t)
        The second element contains the value of the radius of curvature.
        The radius value is calculated as : ``

        >>> R = 1/ np.linalg.norm(self.curvature(t))

        The third element represents the spatial plane of curvature circle as a numpy array with shape (4,3), (Origin, Xaxis, Yaxis, Normal).
        The plane is calculated as :
        >>> K, T = self.tangent(t),self.curvature(t)
        >>> plane= P + unit(K) * R, T, K

        :rtype: tuple[NdArray[float], float, NdArray[float]]



        """

        return circle_of_curvature(self, t)

    def _evaluate_length(self, bounds, tol=1e-3):
        if abs(bounds[1] - bounds[0]) > 4:
            s1, e1 = math.ceil(bounds[0]), math.floor(bounds[1])
            res = 0.

            for start, end in [bounds[0], s1], *divide_interval(s1, e1, 1).tolist(), [e1, bounds[1]]:
                res += evaluate_length(self.derivative, start, end,tol=tol)[0]
            return res

        return evaluate_length(self.derivative, bounds[0], bounds[1], tol=tol)[0]

    def evaluate_length(self, bounds: tuple[float, float], tol: float = 1e-3) -> float:
        """
        Estimate the length of the curve from bounds[0] to bounds[1] using integration
        :param bounds: start and end parameter
        :param tol: Tolerance. default: 1e-3
        :return: length of the curve between start and end parameters
        """
        return self._evaluate_length_cached(bounds, tol)

    def evaluate_parameter_at_length(self, length, t0=None, tol=1e-9, **kwargs):
        """
        Evaluates the parameter at a specific length.

        :param length: The specific length at which the parameter is to be evaluated.
        :param t0: The start time for evaluating the parameter. If not provided, it defaults to the start time of the interval.
        :param kwargs: Additional keyword arguments to be passed.
        :return: The parameter value at the specified length.
        """
        if hasattr(self,'_length_evaluator'):
            return self._length_evaluator(length)
        else:
            self._length_evaluator=length_evaluator(self,tol)


        #start, end = self.interval()
        #t0 = t0 if t0 is not None else start
        #t1_limit = end
        #
        #def func(t):
        #    return abs(self.evaluate_length((t0, t)) ** 2 - length ** 2)
        #
        #res = iterative_divide_and_conquer_min(func, (t0, t1_limit), 0.1)
        #
        #return newton(
        #    func, res[0], tol=tol, x1=t1_limit, **kwargs
        #)
        return self._length_evaluator(length)

    def apply_operator(self, other, op: Callable):
        if isinstance(other, Curve):
            return CurveCurveNode(self, other, op)
        elif inspect.isfunction(other):
            return CurveCallableNode(self, other, op)
        else:
            return CurveValueNode(self, other, op)

    def __add__(self, other):
        return self.apply_operator(other, operator.add)

    def __sub__(self, other):
        return self.apply_operator(other, operator.sub)

    def __mul__(self, other):
        return self.apply_operator(other, operator.mul)

    def __truediv__(self, other):
        return self.apply_operator(other, operator.truediv)

    def normalize_param(self, t):
        return np.interp(t, self.interval(), (0.0, 1.0))

    def remap_param(self, t, domain=(0.0, 1.0)):
        return np.interp(t, domain, self.interval())



    def build_tree(self, count=50):
        self._tree = curve_bvh(self,
                               tuple(self.interval()),
                               count=count
                               )

    def length_at(self, t):
        return self.evaluate(self.evaluate_parameter_at_length(t))

    def step(self, t, tol=1e-3):



        K = self.curvature(t)

        r = 1 / np.linalg.norm(K)

        step = np.sqrt(r ** 2 - (r - tol) ** 2) * 2

        return step,K

class TrimmedCurve(Curve):
    def __init__(self, curve: Curve, start: float = None, end: float = None):
        super().__init__(init_derivatives=False)
        if start is None:
            start = curve.interval()[0]
        if end is None:
            end = curve.interval()[1]
        self.curve = curve
        self.trim = start, end
        self.derivative = self.curve.derivative
        self.second_derivative = self.curve.derivative

    @property
    def start(self):
        return self.start[0]

    @start.setter
    def start(self, v):
        self.trim = (float(v), self.end)

    @property
    def end(self):
        return self.trim[1]

    @end.setter
    def end(self, v):
        self.trim = (self.start, float(v))

    def interval(self):
        return self.trim

    def evaluate(self, t):
        return self.curve.evaluate(t)


class CurveWrapper(Curve):
    def __init__(self, func):
        self.func = func
        self._interval = (0.0, 1.0)
        super().__init__()

    def interval(self):
        return self._interval

    def set_interval(self, t: tuple[float, float]):
        self._interval = t

    def evaluate(self, t):
        return self.func(t)


def parametric_curve(interval: tuple[float, float]):
    def wrapper(func):
        crv = CurveWrapper(func)
        crv._interval = interval
        return crv

    return wrapper


class CurveValueNode(Curve):
    def __init__(self, first: Curve, second: Any, operator=lambda x, y: x + y):
        self.first = first
        self.second = second
        self.operator = operator

        super().__init__()

    def interval(self):
        return self.first.interval()

    def evaluate(self, t: float) -> ArrayLike:
        return self.operator(self.first.evaluate(t), self.second)


class CurveCallableNode(CurveValueNode):
    def evaluate(self, t: float) -> ArrayLike:
        return self.operator(self.first.evaluate(t), self.second(t))


class CurveCurveNode(CurveCallableNode):
    def evaluate(self, t: float) -> ArrayLike:
        return self.operator(self.first.evaluate(t), self.second.evaluate(t))

    def interval(self):
        f = self.first.interval()
        s = self.second.interval()

        return max(f[0], s[0]), min(f[1], s[1])


def curve_bvh(curve: Curve, bounds=None, count=None):
    start, end = curve.interval() if bounds is None else bounds

    if count is None:
        if hasattr(curve, 'degree'):

            count = math.ceil(end - start) * curve.degree
        else:
            count = 16

    t = np.linspace(start, end, count + 1)
    ts = np.empty((count, 2))
    pts = np.empty((count, 2, 3))
    res = curve(t)
    segments = []
    for i in range(count):
        pts[i, 0, :] = res[i]
        pts[i, 1, :] = res[i + 1]
        ts[i, 0] = t[i]
        ts[i, 1] = t[i + 1]
        segments.append(PSegment(pts[i], ts[i]))

    return build_bvh(segments)

from scipy.interpolate import interp1d

def curvature_points(curve, tol=1e-3):
    from mmcore.numeric.algorithms.point_inversion import point_inversion_curve
    start, end = curve.interval()
    t = start
    params = [t]
    P, T, N, B = np.array(curve.plane_at(t))
    steps = [start]
    next_t = t + tol

    while next_t < end:
        P_prev, T_prev, N_prev, B_prev = P, T, N, B
        params.append(next_t)

        P, T, N, B = np.array(curve.plane_at(next_t))

        R = np.linalg.norm(P - P_prev) / np.arccos(np.dot(T, T_prev))

        phi = 2 * np.arccos((1 - tol / R))
        step = R * np.tan(phi) / 2

        t = next_t

        next_t = point_inversion_curve(curve, P + (T * step), t, 1e-6, 1e-6)
        if next_t <= t:
            print('oo')
            next_t = t + tol / 2

        steps.append(step)
    params.append(end)
    return np.array(params), np.array(steps)

def arc_length_reparameterization(curve):
    count=100
    start, end = curve.interval()
    params = np.linspace(start, end, count)
    lvals=np.zeros((count,))
    for i,v in enumerate(params):

        lvals[i]=curve.evaluate_length((start, float(v)))






    return interp1d(np.linspace(0.,1.,count),interp1d(lvals,params)(np.linspace(lvals.min(),lvals.max(),count)))


def length_evaluator(curve,tol):
    count=100
    start, end = curve.interval()
    params = np.linspace(start,end,count)

    lvals = np.array( [curve.evaluate_length( (start,float(i)),tol=tol) for i in params])
    return interp1d( lvals,params)

class ReparametrizedCurve(Curve):
    def __init__(self, curve:Curve, parametrization:interp1d):
        self.curve=curve
        self.parametrization=parametrization
        self._interval = parametrization.x.min(), parametrization.x.max()
    def evaluate_curve_parameter(self, t):
        return self.parametrization(t)
    def interval(self):
        return self._interval
    def evaluate(self, t):
        return self.curve.evaluate(float(self.parametrization(t)))
    def evaluate_length(self, bounds):
        return self.curve.evaluate_length(tuple(self.parametrization(bounds)))
    def tangent(self, t):
        return self.curve.tangent(self.parametrization(t))
    def normal(self, t):
        return self.curve.normal(self.parametrization(t))
    def derivative(self, t):
        return self.curve.derivative(self.parametrization(t))
    def second_derivative(self, t):
        return self.curve.second_derivative(self.parametrization(t))
    def evaluate_multi(self, t):

        return self.curve.evaluate_multi(self.parametrization(t))
    def curvature(self, t):
        return self.curve.curvature(self.parametrization(t))
    def plane_at(self, t):
        return self.curve.plane_at(self.parametrization(t))
    def planes_at(self, t):
        return self.curve.planes_at(self.parametrization(t))
    def circle_of_curvature(self, t):
        return self.circle_of_curvature(self.parametrization(t))

class ArcLengthParameterization(ReparametrizedCurve):
    def __init__(self, curve:Curve):
        super().__init__(curve,arc_length_reparameterization(curve))

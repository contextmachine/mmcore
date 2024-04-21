import inspect
import itertools
from collections import defaultdict
from functools import lru_cache
from typing import TypeVar, Union, SupportsIndex, Sequence, Callable, Any
import numpy as np
from numpy.typing import ArrayLike
import operator
from mmcore.geom.vec import unit, cross
from mmcore.numeric.fdm import FDM, fdm, DEFAULT_H
from mmcore.numeric.numeric import (
    evaluate_tangent,
    evaluate_curvature,
    normal_at,
    plane_on_curve,
    evaluate_length,
)
from mmcore.func import vectorize


from .utils import (
    calc_b_spline_point,
    calcNURBSDerivatives,
    calc_bspline_derivatives,
    calc_rational_curve_derivatives,
)

from ._proto import *

import copy
import math



"""
Here are the implementations of the requested functions and methods:

1. `split(t: float) → tuple[NURBSpline, NURBSpline]` method in the `NURBSpline` class:

"""
from mmcore.geom.bspline.knot import (
    find_span_binsearch,
    find_multiplicity,
    knot_insertion,
)


def insert_knot(self, t, num=1):
    cpts, knots = knot_insertion(
        self.degree, self.knots.tolist(), self.control_points, t, num=num
    )
    self.set(control_points=np.array(cpts), knots=np.array(knots))
    return True


def split(self, t: float) -> tuple:
    """
    * ``insert_knot_func``: knot insertion algorithm implementation. *Default:* :func:`.operations.insert_knot`

    :param obj: Curve to be split
    :type obj: abstract.Curve
    :param param: parameter
    :type param: float
    :return: a list of curve segments
    :rtype: list
    """
    # Validate input

    # Keyword arguments
    span_func = find_span_binsearch  # FindSpan implementation
    insert_knot_func = insert_knot
    knotvector = self.knots
    # Find multiplicity of the knot and define how many times we need to add the knot
    ks = (
        span_func(self.degree, knotvector, len(self.control_points), t)
        - self.degree
        + 1
    )
    s = find_multiplicity(t, knotvector)
    r = self.degree - s

    # Create backups of the original curve
    temp_obj = copy.deepcopy(self)

    # Insert knot
    insert_knot_func(temp_obj, t, num=r)

    # Knot vectors
    knot_span = (
        span_func(temp_obj.degree, temp_obj.knots, len(temp_obj.control_points), t) + 1
    )
    curve1_kv = list(temp_obj.knots.tolist()[0:knot_span])
    curve1_kv.append(t)
    curve2_kv = list(temp_obj.knots.tolist()[knot_span:])
    for _ in range(0, temp_obj.degree + 1):
        curve2_kv.insert(0, t)

    # Control points (use Pw if rational)
    cpts = temp_obj.control_points.tolist()
    curve1_ctrlpts = cpts[0 : ks + r]
    curve2_ctrlpts = cpts[ks + r - 1 :]

    # Create a new curve for the first half
    curve1 = temp_obj.__class__(
        np.array(curve1_ctrlpts), knots=curve1_kv, degree=self.degree
    )

    # Create another curve fot the second half
    curve2 = temp_obj.__class__(
        np.array(curve2_ctrlpts), knots=curve2_kv, degree=self.degree
    )

    # Return the split curves

    return curve1, curve2


"""

2. `nurbs_curve_intersect(curve1: NURBSpline, curve2: NURBSpline, tol: float) → list[tuple[float, float] | None]` function:

"""


def fround(val: float, tol: float = 0.001):
    return round(val, int(abs(math.log10(tol))))


"""

These implementations handle both 2D and 3D cases, as the dimensionality is determined by the control points of the NURBS curves.

The `split` method uses De Boor's algorithm to split the NURBS curve at the given parameter value `t`. It computes the new control points, weights, and knot vectors for the resulting left and right subcurves.

The `nurbs_curve_intersect` function uses a recursive divide-and-conquer approach to find intersections between two NURBS curves. 
It checks the AABB overlap of the curves and recursively splits them until the distance between the curves 
is within the specified tolerance or there is no overlap. 
The function returns a list of tuples representing the parameter values of the intersections on each curve.

Note that the `nurbs_curve_aabb` function from document 10 is used to compute the AABB of the NURBS curves for the intersection algorithm."""


if __name__ == "__main__":
    import numpy as np
    from mmcore.geom.bspline import NURBSpline

    a, b, c, d = np.array(
        [
            [
                (-25.632193861977559, -25.887792238151487, 0.0),
                (-7.6507873591044131, -28.580781837412534, 0.0),
                (3.1180460594601840, -31.620627096247443, 0.0),
                (35.586827711309354, -35.550809492847861, 0.0),
            ],
            [
                (33.586827711309354, -30.550809492847861, 0.0),
                (23.712213781367616, -20.477792480394431, 0.0),
                (23.624609526477588, -7.8543655761938815, 0.0),
                (27.082667168033424, 5.5380493986617410, 0.0),
            ],
            [
                (27.082667168033424, 5.5380493986617410, 0.0),
                (8.6853191615639460, -2.1121318577726527, 0.0),
                (-3.6677924590213919, -2.9387254504549816, 0.0),
                (-20.330418684651349, 3.931006353774948, 0.0),
            ],
            [
                (-20.330418684651349, 3.931006353774948, 0.0),
                (-22.086936165417491, -5.8423256715423690, 0.0),
                (-23.428753995169622, -15.855467779623531, 0.0),
                (-25.632193861977559, -25.887792238151487, 0.0),
            ],
        ]
    )

    spl = NURBSpline(
        np.array(
            [
                [-1, -3, 0.0],
                [-0.1, -2.1, 0.0],
                [-0.4, -1, 0.0],
                [0.0, 0.5, 0.0],
                [0.4, 1.0, 0.0],
                [5.4, -2.1, 0.0],
                [8.0, -3.0, 0.0],
            ]
        ),
        degree=3,
    )
    print(spl.interval())
    res1 = split(spl, 1.5)


class Curve:
    def __init__(self):
        super().__init__()
        self.evaluate_multi = np.vectorize(self.evaluate, signature="()->(i)")
        self._derivatives = [self, self.derivative]
        self._evaluate_cached=lru_cache(self.evaluate)
        self.add_derivative()

    def interval(self):
        return 0.0, 1.0

    def split(self, t):
        return SubCurve(self, self.interval()[0], t), SubCurve(
            self, t, self.interval()[1]
        )

    def normal(self, t):
        return normal_at(self.derivative(t), self.second_derivative(t))

    def tangent(self, t):
        return evaluate_tangent(self.derivative(t), self.second_derivative(t))[0]

    def curvature(self, t):
        return evaluate_curvature(self.derivative(t), self.second_derivative(t))[1]

    def plane_at(self, t):
        return plane_on_curve(self.evaluate(t), self.tangent(t), self.second_derivative(t))

    def derivative(self, t):
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

    @property
    def second_derivative(self):
        return self._derivatives[2]

    def add_derivative(self):
        self._derivatives.append(fdm(self._derivatives[-1]))
        return len(self._derivatives)

    def __call__(
        self, t: Union[np.ndarray[Any, float], float]
    ) -> np.ndarray[Any, np.dtype[float]]:
        return self.evaluate_multi(t)

    def evaluate(self, t: float) -> ArrayLike:
        ...

    def circle_of_curvature(self, t):
        return circle_of_curvature(self, t)

    def evaluate_length(self, bounds):
        return evaluate_length(self, bounds[0], bounds[1])[0]

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


class Circle(Curve):
    def __init__(self, radius, origin=np.array([0.0, 0.0, 0.0])):
        super().__init__()
        self.r = radius
        self.origin = origin

    @property
    def a(self):
        return self.r

    @property
    def b(self):
        return self.origin[0]

    @property
    def c(self):
        return self.origin[1]

    def fx(self, x):
        _ = np.sqrt(self.a**2 - (x - self.b) ** 2)
        return np.array([self.c + _, self.c - _])

    def fy(self, y):
        _ = np.sqrt(self.a**2 - (y - self.c) ** 2)
        return np.array([self.b + _, self.b - _])

    def implict(self, xy):
        return (
            (xy[0] - self.origin[0]) ** 2 + (xy[1] - self.origin[1]) ** 2 - self.r**2
        )

    def intersect_with_circle(self, circle):
        ...

    def evaluate(self, t: float) -> ArrayLike:
        return np.array(
            [self.r * np.cos(t) + self.origin[0], self.r * np.sin(t) + self.origin[1]]
        )


bscache_stats = dict()

from mmcore.geom.bspline.deboor import deboor


def bscache(obj):
    cnt = itertools.count()
    bscache_stats[id(obj)] = dict(calls=next(cnt))
    stats = bscache_stats[id(obj)]

    @lru_cache(maxsize=None)
    def wrapper(a, b, c):
        stats["calls"] = next(cnt)
        return obj.basis_function(a, b, c)

    return wrapper, stats


class BSpline(Curve):
    """
    """

    _control_points = None
    degree = 3
    knots = None

    def interval(self):
        return (float(min(self.knots)), float(max(self.knots)))

    def __init__(self, control_points, degree=3, knots=None):
        super().__init__()
        self._control_points_count = None
        self.basis_cache = dict()
        self._cached_basis_func = lru_cache(maxsize=None)(self.basis_function)
        self.set(control_points, degree=degree, knots=knots)
        self._wcontrol_points = np.ones((len(control_points), 4), dtype=float)
        self._wcontrol_points[:, :-1] = self.control_points

    @vectorize(excluded=[0], signature="()->(i)")
    def derivative(self, t):
        return calc_bspline_derivatives(
            self.degree, self.knots, self._wcontrol_points, t, 1
        )[1][:-1]

    @vectorize(excluded=[0], signature="()->(i)")
    def second_derivative(self, t):
        return calc_bspline_derivatives(
            self.degree, self.knots, self._wcontrol_points, t, 2
        )[2][:-1]

    @vectorize(excluded=[0, "n", "return_projective"], signature="()->(j,i)")
    def n_derivative(self, t, n=3, return_projective=False):
        """
        :param t: Parameter on the curve
        :param n: The number of derivatives that need to be evaluated.
        The zero derivative ( n=0 ) is a point on the curve, the last derivative is always==[0,0,0,0,1]

        :param return_projective: If True will return vectors as [x,y,z,w] instead of [x,y,z] default False
        :return:
        """
        res = np.array(
            calc_bspline_derivatives(
                self.degree, self.knots, self._wcontrol_points, t, n
            )
        )
        if return_projective:
            return res
        return res[..., :-1]

    def set(self, control_points=None, degree=None, knots=None):
        if control_points is not None:
            self.control_points = control_points

        if degree is not None:
            self.degree = degree
        self._control_points_count = len(self.control_points)
        self.knots = self.generate_knots() if knots is None else np.array(knots)

    def generate_knots(self):
        """
        In this code, the `generate_knots` method generates default knots based on the number of control points.
        The `__call__` method computes the parametric B-spline equation at the given parameter `t`.
        It first normalizes the parameter and finds the appropriate knot interval.
        Then, it computes the blending functions within that interval
        and uses them to compute the point on the B-spline curve using the control points.
        https://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/SMAG/node4.html

        This function generates default knots based on the number of control points
        :return: A list of knots


        Difference with OpenNURBS.
        ------------------
        Why is there always one more node here than in OpenNURBS?
        In fact, it is OpenNURBS that deviates from the standard here.
        The original explanation can be found in the file `opennurbs/opennurbs_evaluate_nurbs.h`.
        But I will give a fragment:

            `Most literature, including DeBoor and The NURBS Book,
        duplicate the Opennurbs start and end knot values and have knot vectors
        of length d+n+1. The extra two knot values are completely superfluous
        when degree >= 1.`  [source](https://github.com/mcneel/opennurbs/blob/19df20038249fc40771dbd80201253a76100842c/opennurbs_evaluate_nurbs.h#L116-L120)




        """
        n = len(self.control_points)
        knots = (
            [0] * (self.degree + 1)
            + list(range(1, n - self.degree))
            + [n - self.degree] * (self.degree + 1)
        )

        return np.array(knots, float)

    def find_span(self, t, i):
        return self.basis_function(t, i, self.degree)

    def basis_function(self, t, i, k):
        """
        Calculating basis function with de Boor algorithm
        """
        # print(t,i,k)

        return deboor(self.knots, t, i, k)

    def evaluate(self, t: float):
        result = np.zeros(3, dtype=float)
        if t == 0.0:
            t += 1e-8
        elif t == 1.0:
            t -= 1e-8

        for i in range(self._control_points_count):
            b = self._cached_basis_func(t, i, self.degree)

            result[0] += b * self.control_points[i][0]
            result[1] += b * self.control_points[i][1]
            result[2] += b * self.control_points[i][2]
        return result

    @property
    def control_points(self):
        return self._control_points

    @control_points.setter
    def control_points(self, value):
        self._control_points = np.array(value, dtype=float)
        self.basis_cache.clear()

    def __call__(self, t: float) -> tuple[float, float, float]:
        """
        Here write a solution to the parametric equation bspline at the point corresponding to the parameter t.
        The function should return three numbers (x,y,z)
        """

        self._control_points_count = n = len(self.control_points)
        assert (
            n > self.degree
        ), "Expected the number of control points to be greater than the degree of the spline"
        return super().__call__(t)


class NURBSpline(BSpline):
    """
    Non-Uniform Rational BSpline (NURBS)
    Example:
        >>> spl = NURBSpline(np.array([(-26030.187675027133, 5601.3871095975337, 31638.841094491760),
        ...                   (14918.717302595671, -25257.061306278192, 14455.443462719517),
        ...                   (19188.604482326708, 17583.891501540096, 6065.9078795798523),
        ...                   (-18663.729281923122, 5703.1869371495322, 0.0),
        ...                   (20028.126297559378, -20024.715164607202, 2591.0893519960955),
        ...                   (4735.5467668945130, 25720.651181520021, -6587.2644037490491),
        ...                   (-20484.795362315021, -11668.741154421798, -14201.431195298581),
        ...                   (18434.653814767291, -4810.2095985021788, -14052.951382291201),
        ...                   (612.94310080525793, 24446.695569574043, -24080.735343204549),
        ...                   (-7503.6320665111089, 2896.2190847052334, -31178.971042788111)]
        ...                  ))

    """

    weights: np.ndarray

    def __init__(self, control_points, weights=None, degree=3, knots=None):
        self.weights = np.ones((len(control_points),), dtype=float)
        super().__init__(control_points, degree=degree, knots=knots)
        self.set_weights(weights)

    def set_weights(self, weights=None):
        if weights is not None:
            if len(weights) == len(self.control_points):
                self.weights[:] = weights
            else:
                raise ValueError(
                    f"Weights must have the same length as the control points! Passed weights: {weights}, control_points size: {self._control_points_count}, control_points :{self.control_points}, weights : {weights}"
                )

    def split(self, t):
        return split(self, t)

    def evaluate(self, t: float):
        x, y, z = 0.0, 0.0, 0.0
        sum_of_weights = 0.0  # sum of weight * basis function

        if abs(t - 0.0) <= 1e-8:
            t = 0.0
        elif abs(t - 1.0) <= 1e-8:
            t = 1.0

        for i in range(self._control_points_count):
            b = self._cached_basis_func(t, i, self.degree)
            x += b * self.weights[i] * self.control_points[i][0]
            y += b * self.weights[i] * self.control_points[i][1]
            z += b * self.weights[i] * self.control_points[i][2]
            sum_of_weights += b * self.weights[i]
        # normalizing with the sum of weights to get rational B-spline
        x /= sum_of_weights
        y /= sum_of_weights
        z /= sum_of_weights
        return np.array((x, y, z))

    def __call__(self, t: float) -> tuple[float, float, float]:
        """
        Here write a solution to the parametric equation Rational BSpline at the point corresponding
        to the parameter t. The function should return three numbers (x,y,z)
        """
        self._control_points_count = len(self.control_points)
        assert (
            self._control_points_count > self.degree
        ), "Expected the number of control points to be greater than the degree of the spline"
        assert (
            len(self.weights) == self._control_points_count
        ), "Expected to have a weight for every control point"
        return Curve.__call__(self, t)


def circle_of_curvature(curve: Curve, t: float):
    origin = curve(t)
    T, K, success = curve.curvature(t)

    N = unit(K)
    B = cross(T, N)
    k = np.linalg.norm(K)
    R = 1 / k
    return (
        np.array([origin + K * R, T, N, B]),
        R,
    )  # Plane of curvature circle, Radius of curvature circle


class SubCurve(Curve):
    def __init__(self, crv, start, end):
        self.parent = crv
        self.start = start
        self.end = end

    def derivative(self, t: float):
        return self.parent.derivative(t)

    def second_derivative(self, t: float):
        return self.parent.derivative(t)

    def second_derivative(self, t: float):
        return self.parent.second_derivative(t)


    def plane_at(self, t: float):
        return self.parent.plane_at(t)

    def owner_interval(self):
        return (self.start, self.end)

    def interval(self):
        return (self.start, self.end)

    def evaluate(self, t):
        return self.parent.evaluate(t)


class Offset(Curve):
    def __init__(self, crv:Curve, distance:float):
        super().__init__()
        self.crv = crv
        self._is_uniform=None
        self._distance=None
        self.distance=distance


    def interval(self):
        return self.crv.interval()

    def evaluate(self, t: float) -> ArrayLike:
        return self.crv(t) + self.crv.normal(t) * self.distance

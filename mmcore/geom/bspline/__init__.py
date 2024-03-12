"""
Certainly! Here's the code for the `BSpline` class with the `generate_knots` and `__call__` methods implemented:
"""
import math

from mmcore.geom.vec import unit, cross, dot
from mmcore.numeric.fdm import FDM
from mmcore.numeric.divide_and_conquer import recursive_divide_and_conquer_min

'''
class BSpline:
    def __init__(self, control_points, degree=3, knots=None):
        self.control_points = control_points
        self.degree = degree
        self.knots = self.generate_knots() if knots is None else knots
        self.derivative = FDM(self)
        self.second_derivative = FDM(self.derivative)

    def interval(self):
        return (0., max(self.knots))

    def generate_knots(self):
        """
        This function generates default knots based on the number of control points
        :return: A list of knots
        """
        n = len(self.control_points)
        knots = [0] * (self.degree + 1) + list(range(1, n - self.degree)) + [n - self.degree] * (self.degree + 1)
        return knots

    def basis_function(self, t, i, k, T):
        """
        Calculating basis function with Cox - de Boor algorithm
        """
        if k == 0:
            return 1.0 if T[i] <= t < T[i + 1] else 0.0
        if T[i + k] == T[i]:
            c1 = 0.0
        else:
            c1 = (t - T[i]) / (T[i + k] - T[i]) * self.basis_function(t, i, k - 1, T)
        if T[i + k + 1] == T[i + 1]:
            c2 = 0.0
        else:
            c2 = (T[i + k + 1] - t) / (T[i + k + 1] - T[i + 1]) * self.basis_function(t, i + 1, k - 1, T)
        return c1 + c2

    @vectorize(excluded=[0], signature='()->(i)')
    def __call__(self, t: float) -> tuple[float, float, float]:
        """
        Here write a solution to the parametric equation bspline at the point corresponding to the parameter t. The function should return three numbers (x,y,z)
        """
        n = len(self.control_points)
        assert n > self.degree, "Expected the number of control points to be greater than the degree of the spline"

        result = [0.0, 0.0, 0.0]
        for i in range(n):
            b = self.basis_function(t, i, self.degree, self.knots)
            result[0] += b * self.control_points[i][0]
            result[1] += b * self.control_points[i][1]
            result[2] += b * self.control_points[i][2]

        return np.array(result)


class RationalBSpline(BSpline):
    def __init__(self, control_points, weights=None, degree=3, knots=None):
        super().__init__(control_points, degree, knots)
        self.weights = np.ones((len(self.control_points),), dtype=float) if weights is None else np.array(weights)

    @vectorize(excluded=[0], signature='()->(i)')
    def __call__(self, t: float) -> tuple[float, float, float]:
        """
        Here write a solution to the parametric equation RationalBSpline at the point corresponding
        to the parameter t. The function should return three numbers (x,y,z)
        """
        n = len(self.control_points)
        assert n > self.degree, "Expected the number of control points to be greater than the degree of the spline"
        assert len(self.weights) == n, "Expected to have a weight for every control point"

        rational_function_values = [0.0, 0.0, 0.0]  # values of the rational B-spline function at t
        sum_of_weights = 0  # sum of weight * basis function

        for i in range(n):
            b = self.basis_function(t, i, self.degree, self.knots)
            rational_function_values[0] += b * self.weights[i] * self.control_points[i][0]
            rational_function_values[1] += b * self.weights[i] * self.control_points[i][1]
            rational_function_values[2] += b * self.weights[i] * self.control_points[i][2]
            sum_of_weights += b * self.weights[i]

        # normalizing with the sum of weights to get rational B-spline
        rational_function_values[0] /= sum_of_weights
        rational_function_values[1] /= sum_of_weights
        rational_function_values[2] /= sum_of_weights

        return np.array(rational_function_values)

'''
import numpy as np

from typing import TypeVar, Union, SupportsIndex, Sequence, Callable, Any

ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]
D = TypeVar("D", bound=SupportsIndex)

from numpy.typing import ArrayLike

from mmcore.numeric.numeric import evaluate_tangent, evaluate_curvature, normal_at, plane_on_curve


class BaseCurve:
    def __init__(self):
        super().__init__()
        self.evaluate_multi = np.vectorize(self.evaluate, signature="()->(i)")
        self._derivatives = [self]
        self.add_derivative()
        self.add_derivative()

    def normal(self, t):
        return normal_at(self.derivative(t), self.second_derivative(t))

    def tangent(self, t):
        return evaluate_tangent(self.derivative(t), self.second_derivative(t))[0]

    def curvature(self, t):
        return evaluate_curvature(self.derivative(t), self.second_derivative(t))[1]

    def plane_at(self, t):
        return plane_on_curve(self(t), self.tangent(t), self.second_derivative(t))

    @property
    def derivative(self):
        return self._derivatives[1]

    @property
    def second_derivative(self):
        return self._derivatives[2]

    def add_derivative(self):
        self._derivatives.append(FDM(self._derivatives[-1]))
        return len(self._derivatives)

    def __call__(self, t: Union[np.ndarray[Any, float], float]) -> np.ndarray[Any, np.dtype[float]]:
        return self.evaluate_multi(t)

    def evaluate(self, t: float) -> ArrayLike: ...


class Circle(BaseCurve):
    def __init__(self, radius, origin=np.array([0., 0.0, 0.])):
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
        _ = np.sqrt(self.a ** 2 - (x - self.b) ** 2)
        return np.array([self.c + _, self.c - _])

    def fy(self, y):
        _ = np.sqrt(self.a ** 2 - (y - self.c) ** 2)
        return np.array([self.b + _, self.b - _])

    def implict(self, xy):
        return (xy[0] - self.origin[0]) ** 2 + (xy[1] - self.origin[1]) ** 2 - self.r ** 2

    def intersect_with_circle(self, circle):
        ...

    def evaluate(self, t: float) -> ArrayLike:
        return np.array([self.r * np.cos(t) + self.origin[0], self.r * np.sin(t) + self.origin[1]])


def find_span(n, degree, u, knots):
    if u == knots[n + 1]:
        return n
    low = degree
    high = n + 1
    return bisect.bisect(knots, u, low, high)


import bisect

from .utils import calc_b_spline_point, calcNURBSDerivatives, calc_bspline_derivatives, calc_rational_curve_derivatives
class BSpline(BaseCurve):
    """
    """
    control_points = None
    degree = 3
    knots = None
    def interval(self):

        return (0., float(self.knots.max()))

    def __init__(self, control_points, degree=3, knots=None):
        super().__init__()

        self.set(control_points, degree=degree, knots=knots)
        self._wcontrol_points = np.ones((len(control_points), 4), dtype=float)
        self._wcontrol_points[:, :-1] = self.control_points

    def derivative(self, t):
        return calc_bspline_derivatives(self.degree, self.knots, self._wcontrol_points, t, 1)[1][:-1]

    def second_derivative(self, t):
        return calc_bspline_derivatives(self.degree, self.knots, self._wcontrol_points, t, 2)[2][:-1]
    def set(self, control_points=None, degree=None, knots=None):
        if control_points is not None:
            self.control_points = control_points
        if degree is not None:
            self.degree = degree
        self.knots = self.generate_knots() if knots is None else knots

    def generate_knots(self):
        """
        This function generates default knots based on the number of control points
        :return: A list of knots
        """
        n = len(self.control_points)
        knots = [0] * (self.degree + 1) + list(range(1, n - self.degree)) + [n - self.degree] * (self.degree + 1)

        return np.array(knots, float)

    def find_span(self, t, i):

        return self.basis_function(t, i, self.degree)

    def basis_function(self, t, i, k):
        """
        Calculating basis function with de Boor algorithm
        """
        T = self.knots

        if k == 0:
            return 1.0 if T[i] <= t <= T[i + 1] else 0.0
        if T[i + k] == T[i]:

            c1 = 0.0
        else:
            c1 = (t - T[i]) / (T[i + k] - T[i]) * self.basis_function(t, i, k - 1)

        if T[i + k + 1] == T[i + 1]:
            c2 = 0.0
        else:
            c2 = (T[i + k + 1] - t) / (T[i + k + 1] - T[i + 1]) * self.basis_function(t, i + 1, k - 1)
        return c1 + c2

    def evaluate(self, t: float):
        result = np.zeros((3,), dtype=float)
        if t == 0.:
            t += 1e-8
        elif t == 1.:
            t -= 1e-8

        for i in range(self._control_points_count):
            b = self.basis_function(t, i, self.degree)

            result[0] += b * self.control_points[i][0]
            result[1] += b * self.control_points[i][1]
            result[2] += b * self.control_points[i][2]
        return result

    def __call__(self, t: float) -> tuple[float, float, float]:
        """
        Here write a solution to the parametric equation bspline at the point corresponding to the parameter t.
        The function should return three numbers (x,y,z)
        """
        self._control_points_count = n = len(self.control_points)
        assert n > self.degree, "Expected the number of control points to be greater than the degree of the spline"
        return super().__call__(t)


"""
In this code, the `generate_knots` method generates default knots based on the number of control points. The `__call__` method computes the parametric B-spline equation at the given parameter `t`. It first normalizes the parameter and finds the appropriate knot interval. Then, it computes the blending functions within that interval and uses them to compute the point on the B-spline curve using the control points.
https://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/SMAG/node4.html
"""


class NURBSpline(BSpline):
    """
    Non-Uniform Rational BSpline (NURBS)
    """
    weights: np.ndarray
    def __init__(self, control_points, weights=None, degree=3, knots=None):
        super().__init__(control_points, degree, knots)
        self.weights = np.ones((len(self.control_points),), dtype=float) if weights is None else np.array(weights)

    def set(self, control_points=None, weights=None, degree=3, knots=None):
        super().set(control_points, degree=degree, knots=knots)
        if weights is not None:
            self.weights[:] = weights

        self._control_points_count = len(self.control_points)

    def evaluate(self, t: float):
        arr = np.zeros((3,), dtype=float)
        sum_of_weights = 0  # sum of weight * basis function
        for i in range(self._control_points_count):
            b = self.basis_function(t, i, self.degree)

            if b > 0:
                arr[0] += b * self.weights[i] * self.control_points[i][0]
                arr[1] += b * self.weights[i] * self.control_points[i][1]
                arr[2] += b * self.weights[i] * self.control_points[i][2]
                sum_of_weights += b * self.weights[i]
        # normalizing with the sum of weights to get rational B-spline

        arr[0] /= sum_of_weights
        arr[1] /= sum_of_weights
        arr[2] /= sum_of_weights
        return arr

    def __call__(self, t: float) -> tuple[float, float, float]:
        """
        Here write a solution to the parametric equation Rational BSpline at the point corresponding
        to the parameter t. The function should return three numbers (x,y,z)
        """
        self._control_points_count = len(self.control_points)
        assert self._control_points_count > self.degree, "Expected the number of control points to be greater than the degree of the spline"
        assert len(self.weights) == self._control_points_count, "Expected to have a weight for every control point"
        return BaseCurve.__call__(self, t)


"""
In this code, the `generate_knots` method generates default knots based on the number of control points. The `__call__` method computes the parametric B-spline equation at the given parameter `t`. It first normalizes the parameter and finds the appropriate knot interval. Then, it computes the blending functions within that interval and uses them to compute the point on the B-spline curve using the control points.
https://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/SMAG/node4.html
"""


def circle_of_curvature(curve, t):
    origin = curve(t)
    T, K, success = curve.curvature(t)

    N = unit(K)
    B = cross(T, N)
    k = np.linalg.norm(K)
    R = 1 / k
    return np.array([origin + K * R, T, N, B]), R  # Plane of curvature circle, Radius of curvature circle


spl = NURBSpline(np.array([(-26030.187675027133, 5601.3871095975337, 31638.841094491760),
                           (14918.717302595671, -25257.061306278192, 14455.443462719517),
                           (19188.604482326708, 17583.891501540096, 6065.9078795798523),
                           (-18663.729281923122, 5703.1869371495322, 0.0),
                           (20028.126297559378, -20024.715164607202, 2591.0893519960955),
                           (4735.5467668945130, 25720.651181520021, -6587.2644037490491),
                           (-20484.795362315021, -11668.741154421798, -14201.431195298581),
                           (18434.653814767291, -4810.2095985021788, -14052.951382291201),
                           (612.94310080525793, 24446.695569574043, -24080.735343204549),
                           (-7503.6320665111089, 2896.2190847052334, -31178.971042788111)]
                          ))


def product_func(curve, point: np.ndarray[3, np.dtype[float]]) -> Callable[[float], float]:
    return lambda u: dot(curve.derivative(u), curve(u) - point)


class SubCurve(BaseCurve):
    def __init__(self, crv, start, end):
        super().__init__()
        self.crv = crv
        self.start = start
        self.end = end

    def owner_interval(self):
        return (self.start, self.end)

    def interval(self):
        return (0.0, 1.0)

    def evaluate(self, t):
        return self.crv(np.interp(t, self.interval(), (self.start, self.end)))


class Offset(BaseCurve):
    def __init__(self, crv, distance):
        super().__init__()
        self.crv = crv
        self.distance = distance

    def interval(self):
        return self.crv.interval()

    def evaluate(self, t: float) -> ArrayLike:
        return self.crv(t) + self.crv.normal(t) * self.distance


'''
class BSpline(BaseCurve):
    """
    """

    def interval(self):

        return (0., self.knots.max())

    def __init__(self, control_points, degree=3, knots=None):
        super().__init__()
        self.control_points = control_points

        self.degree = degree
        self.knots = self.generate_knots() if knots is None else knots

    def generate_knots(self):
        """
        This function generates default knots based on the number of control points
        :return: A list of knots
        """
        n = len(self.control_points)
        knots = [0] * (self.degree + 1) + list(range(1, n - self.degree)) + ([n - self.degree] * (self.degree + 1))
        return np.array(knots, float)

    def basis_function(self, t, i, k, T):
        """
        Calculating basis function with de Boor algorithm
        """
        if k == 0:
            return 1.0 if T[i] <= t <= T[i + 1] else 0.0
        if T[i + k] == T[i]:
            c1 = 0.0
        else:
            c1 = (t - T[i]) / (T[i + k] - T[i]) * self.basis_function(t, i, k - 1, T)
        if T[i + k + 1] == T[i + 1]:
            c2 = 0.0
        else:
            c2 = (T[i + k + 1] - t) / (T[i + k + 1] - T[i + 1]) * self.basis_function(t, i + 1, k - 1, T)
        return c1 + c2

    def evaluate(self, t: float):
        result = np.zeros((3,), dtype=float)

        for i in range(self._control_points_count):
            b = self.basis_function(t, i, self.degree, self.knots)
            result[0] += b * self.control_points[i][0]
            result[1] += b * self.control_points[i][1]
            result[2] += b * self.control_points[i][2]
        return result

    def __call__(self, t: float) -> tuple[float, float, float]:
        """
        Here write a solution to the parametric equation bspline at the point corresponding to the parameter t.
        The function should return three numbers (x,y,z)
        """
        self._control_points_count = n = len(self.control_points)
        assert n > self.degree, "Expected the number of control points to be greater than the degree of the spline"
        return self.evaluate_multi(t)


"""
In this code, the `generate_knots` method generates default knots based on the number of control points. The `__call__` method computes the parametric B-spline equation at the given parameter `t`. It first normalizes the parameter and finds the appropriate knot interval. Then, it computes the blending functions within that interval and uses them to compute the point on the B-spline curve using the control points.
https://www.cl.cam.ac.uk/teaching/1999/AGraphHCI/SMAG/node4.html
"""


class NURBSpline(BSpline):
    """
    Non-Uniform Rational BSpline (NURBS)
    """

    def __init__(self, control_points, weights=None, degree=3, knots=None):
        super().__init__(control_points, degree, knots)
        self.weights = np.ones((len(self.control_points),), dtype=float) if weights is None else np.array(weights)
        self._control_points_count = len(self.control_points)

    def evaluate(self, t: float):
        x=0.0
        y=0.0
        z=0.0
        sum_of_weights = 0.0  # sum of weight * basis function
        for i in range(self._control_points_count):
            b = self.basis_function(t, i, self.degree, self.knots)
            x += b * self.weights[i] * self.control_points[i][0]
            y += b * self.weights[i] * self.control_points[i][1]
            z += b * self.weights[i] * self.control_points[i][2]
            sum_of_weights += b * self.weights[i]
        # normalizing with the sum of weights to get rational B-spline
        x /= sum_of_weights
        y /= sum_of_weights
        z /= sum_of_weights
        return np.array([x,y,z], dtype=float)

    def __call__(self, t: float) -> tuple[float, float, float]:
        """
        Here write a solution to the parametric equation Rational BSpline at the point corresponding
        to the parameter t. The function should return three numbers (x,y,z)
        """
        self._control_points_count = len(self.control_points)

        return self.evaluate_multi(t)
'''
spl = NURBSpline(np.array([(-26030.187675027133, 5601.3871095975337, 31638.841094491760),
                           (14918.717302595671, -25257.061306278192, 14455.443462719517),
                           (19188.604482326708, 17583.891501540096, 6065.9078795798523),
                           (-18663.729281923122, 5703.1869371495322, 0.0),
                           (20028.126297559378, -20024.715164607202, 2591.0893519960955),
                           (4735.5467668945130, 25720.651181520021, -6587.2644037490491),
                           (-20484.795362315021, -11668.741154421798, -14201.431195298581),
                           (18434.653814767291, -4810.2095985021788, -14052.951382291201),
                           (612.94310080525793, 24446.695569574043, -24080.735343204549),
                           (-7503.6320665111089, 2896.2190847052334, -31178.971042788111)]
                          ))


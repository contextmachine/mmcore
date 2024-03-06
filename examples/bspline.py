"""
Certainly! Here's the code for the `BSpline` class with the `generate_knots` and `__call__` methods implemented:
"""
import functools

from mmcore.func import vectorize
from mmcore.geom.vec import unit, cross
from mmcore.numeric.fdm import FDM


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


import numpy as np

from typing import Protocol, TypeVar, Union, SupportsIndex, Sequence

ShapeLike = Union[SupportsIndex, Sequence[SupportsIndex]]
D = TypeVar("D", bound=SupportsIndex)

from numpy.typing import ArrayLike

from mmcore.numeric.numeric import evaluate_tangent, evaluate_curvature, evaluate_normal


class BaseCurve:
    def __init__(self):
        super().__init__()
        self.evaluate_multi = np.vectorize(self.evaluate, signature="()->(i)")

    def evaluate(self, t: float) -> ArrayLike: ...


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

        self.derivative = FDM(self)
        self.second_derivative = FDM(self.derivative)

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
        arr = np.zeros((3,), dtype=float)
        sum_of_weights = 0  # sum of weight * basis function
        for i in range(self._control_points_count):
            b = self.basis_function(t, i, self.degree, self.knots)
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
        return self.evaluate_multi(t)


from math import ceil

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


def closest_point_on_curve(point: tuple[float, float, float], crv: NURBSpline) -> tuple[
    float, tuple[float, float, float]]:
    def objective_function(t):
        curve_point = crv(t)
        return np.linalg.norm(curve_point - point)

    def derivative_function(t):
        t_vector = crv.derivative(t)
        return np.dot(t_vector, crv(t) - point)

    # initialize t parameter, which will be iteratively adjusted
    t = 0.5

    # set the maximum number of iterations and the tolerance level for early stopping
    max_iterations = 1000
    tolerance = 1e-9

    # begin the Newton-Raphson algorithm for finding the closest point on the curve
    for i in range(max_iterations):
        f_val = objective_function(t)
        df_val = derivative_function(t)

        if abs(f_val) < tolerance:
            break

        t_new = t - f_val / df_val

        # ensure the new t parameter is within the valid range
        if t_new < 0 or t_new > 1:
            break

        t = t_new

    closest_point = crv(t)
    return t, tuple(closest_point.tolist())


class BaseCurve:
    def __init__(self):
        super().__init__()
        self.evaluate_multi = np.vectorize(self.evaluate, signature="()->(i)")
        self.derivative = FDM(self.evaluate_multi)
        self.second_derivative = FDM(self.derivative)

    def tangent(self, t):
        return evaluate_tangent(self.derivative(t), self.second_derivative(t))

    def curvature(self, t):
        return evaluate_curvature(self.derivative(t), self.second_derivative(t))

    def plane(self, t):
        T, K, success = self.curvature(t)
        N = unit(K)
        B = np.cross(T, N)
        return np.array([self.evaluate(t), T, N, B])

    def evaluate(self, t: float) -> ArrayLike: ...


class CubicSpline(BaseCurve):

    def interval(self): ...


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

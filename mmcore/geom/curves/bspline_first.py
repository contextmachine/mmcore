import numpy as np
from mmcore.func import vectorize
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


    About Knots Count.
    ------------------
    Why is there always one more node here than in OpenNURBS?
    In fact, it is OpenNURBS that deviates from the standard here.
    The original explanation can be found in the file `opennurbs/opennurbs_evaluate_nurbs.h`.
    But I will give a fragment:

    >   Most literature, including DeBoor and The NURBS Book,
    > duplicate the Opennurbs start and end knot values and have knot vectors
    > of length d+n+1. The extra two knot values are completely superfluous
    > when degree >= 1. [source](https://github.com/mcneel/opennurbs/blob/19df20038249fc40771dbd80201253a76100842c/opennurbs_evaluate_nurbs.h#L116-L120)

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
        Here write a solution to the parametric equation curves at the point corresponding to the parameter t. The function should return three numbers (x,y,z)
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

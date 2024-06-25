import numpy as np

from mmcore.geom.curves.curve import Curve
from numpy.typing import NDArray
from mmcore.geom.curves._cubic import CubicSpline
__all__=["CubicSpline"]
class PyCubicSpline(Curve):
    """

    :class: CubicSpline

    A class representing a cubic spline curve.

    Methods:
        __init__(p0, c0, c1, p1):
            Initializes the CubicSpline object with the given control points.

        interval():
            Returns the interval of the curve.

        degree():
            Returns the degree of the curve.

        evaluate(t):
            Evaluates the cubic spline curve at the given parameter t.

    Properties:
        p0:
            Gets or sets the starting point of the curve.

        p1:
            Gets or sets the ending point of the curve.

        c0:
            Gets or sets first control point (the first derivative at the starting point).

        c1:
            Gets or sets second control point (the first derivative at the ending point).

    """
    def __init__(self, p0: NDArray[float], c0: NDArray[float], c1: NDArray[float], p1: NDArray[float]):
        """Initialize the object with four points.

        :param p0: The starting point.
        :type p0: NDArray[float]
        :param c0: The first control point.
        :type c0: NDArray[float]
        :param c1: The second control point.
        :type c1: NDArray[float]
        :param p1: The ending point.
        :type p1: NDArray[float]
        """
        super().__init__()
        self.points = np.array([p0, c0, c1, p1])

    def interval(self):
        return 0., 1.

    @property
    def degree(self):
        return 3
    def dderivative(self, t):
        return 3 * self.c0 * t * (2 * t - 2) + 3 * self.c0 * (1 - t) ** 2 - 3 * self.c1 * t ** 2 + 6 * self.c1 * t * (1 - t) - 3 * self.p0 * (
                    1 - t) ** 2 + 3 * self.p1 * t ** 2
    def dsecond_derivative(self, t):
        return 6 * (self.c0 * t + 2 * self.c0 * (t - 1) - 2 * self.c1 * t - self.c1 * (t - 1) - self.p0 * (t - 1) + self.p1 * t)
    def evaluate(self, t):
        """
        Evaluate the value of a cubic Bezier curve at a given parameter.

        :param t: The parameter value at which to evaluate the Bezier curve.
        :type t: float
        :return: The value of the Bezier curve at the given parameter.
        :rtype: float
        """
        return (
                self.p0 * ((1 - t) ** 3)
                + 3 * self.c0 * t * ((1 - t) ** 2)
                + 3 * self.c1 * (t ** 2) * (1 - t)
        ) + self.p1 * (t ** 3)

    @property
    def p0(self):
        return self.points[0]

    @p0.setter
    def p0(self, value):
        self.points[0] = value
    @property
    def p1(self):
        return self.points[3]

    @p1.setter
    def p1(self, value):
        self.points[3] = value
    @property
    def c0(self):
        return self.points[1]

    @c0.setter
    def c0(self, value):
        self.points[1] = value
    @property
    def c1(self):
        return self.points[2]

    @c1.setter
    def c1(self, value):
        self.points[2] = value

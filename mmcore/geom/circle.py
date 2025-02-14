import numpy as np
from mmcore.geom.curves import Curve
from mmcore.geom.implicit import Implicit2D


_IDENTITY = np.eye(3)
PI05 = 1.5707963267948966
PI2 = 6.283185307179586


class Circle2D(Curve, Implicit2D):
    """

    Class Circle2D

    This class represents a 2-dimensional circle.

    Attributes:
    - origin (ndarray): The center point of the circle. Defaults to [0, 0].
    - r (float): The radius of the circle. Defaults to 1.
    - normal (function): A function that returns the unit normal vector to the circle at a given point.

    Methods:
    - __init__(origin=None, r=1.): Initializes a Circle2D object with the specified origin and radius.
    - interval(): Returns the interval of the parameterization of the circle. Returns a tuple (start, end).
    - implicit(v): Returns the implicit equation of the circle at a given point v.
    - _circle_normal(v): Returns the vector from the origin to the specified point v.
    - _circle_normal_unit(v): Returns the unit normal vector to the circle at a given point v.
    - a(): Returns the radius of the circle.
    - a(v): Sets the radius of the circle to the specified value v.
    - x(t): Returns the x-coordinate of a point on the circle at the specified parameter value t.
    - y(t): Returns the y-coordinate of a point on the circle at the specified parameter value t.
    - evaluate(t): Returns the coordinates of a point on the circle at the specified parameter value t as a ndarray.
    - closest_point(v): Returns the closest point on the circle to the specified point v.

    """

    def __init__(self, origin=None, r=1.):
        super().__init__()

        self.origin = (
            origin if isinstance(origin, np.ndarray) else np.array(origin)) if origin is not None else np.zeros(2,
                                                                                                                dtype=float)
        self.r = r

        self.normal = self._implicit_unit_normal

    def interval(self):
        return 0., PI2

    def implicit(self, v) -> float:
        return np.linalg.norm(self._implicit_normal(v)) - self.r

    def _implicit_normal(self, v) -> float:
        return v - self.origin

    def _implicit_unit_normal(self, v) -> float:
        point = self._implicit_normal(v)
        return point / np.linalg.norm(point)

    def gradient(self, val):
        if isinstance(val, float):
            return super(Curve, self).gradient(val)
        else:
            return self._implicit_unit_normal(val)

    def x(self, t):
        return self.origin[0] + self.r * np.cos(t)

    def y(self, t):
        return self.origin[1] + self.r * np.sin(t)

    def evaluate(self, t):
        return np.array([self.x(t), self.y(t)])

    def point_on_curve(self, v):

        N = self._implicit_normal(v)
        nn = np.linalg.norm(N)
        nu = N / nn
        return v - (nu * (nn - self.r)), np.arccos(nu[0])




def circle_intersection2d(c1: Circle2D, c2: Circle2D):
    """
    calculate the intersection points of two 2D circles given as input

    Parameters
    ----------
    c1 : Circle
        An object of type Circle representing the first circle. The object should have `r` as the radius and `origin` as the center coordinates

    c2 : Circle
        An object of type Circle representing the second circle. The object should have `r` as the radius and `origin` as the center coordinates

    Raises
    ------
    ValueError
        If the circles do not intersect. This happens when the distance between the centers of the two circles is
        greater than the sum of their radii or smaller than the absolute difference of their radii

    Returns
    -------
    np.ndarray
        A 2D numpy array of shape (2,3), where each row correspond to a point of intersection.
        Each point is represented by a tuple of x, y and z coordinates.
        The x and y are the coordinates of the intersection point and the z is taken from the original circles' origins

    Examples
    --------
    Given two Circle objects c1 and c2, you can find their intersection points as follows

    >>> intersection = circle_intersection2d(c1, c2)
    """
    # Unpack the circle data
    r1, x1, y1 = c1.r, c1.origin[0], c1.origin[1]
    r2, x2, y2 = c2.r, c2.origin[0], c2.origin[1]

    # Compute the distance between the centers of the circles
    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Check if the circles intersect
    if d > r1 + r2 or d < np.abs(r2 - r1):
        raise ValueError("Circles do not intersect")

    # Compute the coordinates of the intersection points
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(r1 ** 2 - a ** 2)
    xm = x1 + a * (x2 - x1) / d
    ym = y1 + a * (y2 - y1) / d
    xs1 = xm - h * (y2 - y1) / d
    xs2 = xm + h * (y2 - y1) / d
    ys1 = ym + h * (x2 - x1) / d
    ys2 = ym - h * (x2 - x1) / d

    return np.array([(xs1, ys1, c1.origin[-1]), (xs2, ys2, c2.origin[-1])], float)


def tangents_lines(circle: Circle2D, pt: np.ndarray):
    # TODO: Replace current solution to solution using dot product if it possible
    a = circle.r
    direct = pt[:2] - circle.origin[:2]
    b = np.hypot(*direct)  # hypot() also works here
    th = np.arccos(a / b)  # angle theta
    d = np.arctan2(direct[0], direct[1])  # direction angle of point P from C
    d1 = d + th  # direction angle of point T1 from C
    d2 = d - th  # direction angle of point T2 from C

    T1xy = circle.origin[:2] + a * np.array([np.cos(d1), np.sin(d1)])
    T2xy = circle.origin[:2] + a * np.array([np.cos(d2), np.sin(d2)])

    return np.array([[*T1xy, 0.0], [*T2xy, 0.0]])


def closest_point_on_circle_2d(circle: Circle2D, pt: np.ndarray):
    un = circle.normal(pt)

    return np.arccos(un[0]) - PI05

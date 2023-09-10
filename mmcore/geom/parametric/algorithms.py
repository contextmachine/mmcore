import abc
import dataclasses
import typing
from collections import namedtuple

import numpy as np

from mmcore.exceptions import MmodelIntersectException

TOLERANCE = 1e-8
from scipy.optimize import minimize, fsolve
from scipy.spatial.distance import euclidean

ProxPointSolution = namedtuple("ProxPointSolution", ["pt1", "pt2", "t1", "t2"])
ClosestPointSolution = namedtuple("ClosestPointSolution", ["pt", "t", "distance"])
IntersectSolution = namedtuple("IntersectSolution", ["pt", "t", "is_intersect"])
IntersectFail = namedtuple("IntersectFail", ["pt", "t", "distance", "is_intersect"])
MultiSolutionResponse = namedtuple("MultiSolutionResponse", ["pts", "roots"])
Vector2dTuple = namedtuple("VectorTuple", ["x", "y"])
Point2dTuple = namedtuple("PointTuple", ["x", "y"])
Vector3dTuple = namedtuple("VectorTuple", ["x", "y", "z"])
Point3dTuple = namedtuple("PointTuple", ["x", "y", "z"])
LineTuple = namedtuple("LineTuple", ["start", "vec"])
LineStartEndTuple = namedtuple("LineStartEndTuple", ["start", "end"])
Circle2dTuple = namedtuple("Circle2dTuple", ["radius", "x", "y"])
Circle3dTuple = namedtuple("Circle2dTuple", ["circle", "plane"])
NormalPlane = namedtuple("NormalPlane", ["origin", "normal"])

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


def proximity_points(line1, line2):
    # Extract points and directions from input lines
    p1, v1 = line1
    p2, v2 = line2

    # Calculate direction vector of the line connecting the two points
    w = p1 - p2

    # Calculate direction vectors of the two input lines
    a = np.dot(v1, v1)
    b = np.dot(v1, v2)
    c = np.dot(v2, v2)
    d = np.dot(v1, w)
    e = np.dot(v2, w)

    # Calculate parameters for the two closest points
    t = (b * e - c * d) / (a * c - b ** 2)
    s = (a * e - b * d) / (a * c - b ** 2)

    # Calculate the two closest points
    p1_closest = p1 + t * v1
    p2_closest = p2 + s * v2

    return ProxPointSolution(p1_closest, p2_closest, t, s)


def closest_point_on_line(point, line):
    """
    Returns the closest point on a line to a given point in 3D space.

    Parameters:
    point (numpy.ndarray): An array of shape (3,) representing a point in 3D space.
    line (tuple): A tuple of two numpy arrays of shape (3,) representing a point on the line and the direction of the line.

    Returns:
    numpy.ndarray: An array of shape (3,) representing the closest point on the line to the given point.
    """
    # Extract point and direction from input line
    p1, v1 = line

    # Calculate vector from point to line
    w = point - p1

    # Calculate parameter for closest point on line
    t = np.dot(w, v1) / np.dot(v1, v1)

    # Calculate closest point on line
    p_closest = p1 + t * v1

    return ClosestPointSolution(p_closest, t=t, distance=euclidean(point, p1))


PointTuple = tuple[float, float, float]


def eval_quad(t1, t2, quad: list[PointTuple, PointTuple, PointTuple, PointTuple]):
    a, b, c, d = np.array(quad)
    dc, ab = (d + ((c - d) * t1)), (a + ((b - a) * t1))
    return ((dc - ab) * t2) + ab


def triangle_normal(vertices: np.ndarray) -> np.ndarray:
    return np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])


def centroid(vertices: np.ndarray) -> np.ndarray:
    return np.mean(vertices, axis=0)


def normal_plane_from_triangle(vertices: np.ndarray) -> NormalPlane:
    return NormalPlane(centroid(vertices), triangle_normal(vertices))


def ray_triangle_intersection(ray_origin: np.ndarray, ray_direction: np.ndarray, triangle_vertices: np.ndarray):
    """
    Returns the intersection point of a ray and a triangle in 3D space.
    Parameters:
    ray_origin (numpy.ndarray): An array of shape (3,) representing the origin point of the ray.
    ray_direction (numpy.ndarray): An array of shape (3,) representing the direction vector of the ray.
    triangle_vertices (numpy.ndarray): An array of shape (3,3) representing the vertices of the triangle.
    Returns:
    numpy.ndarray: An array of shape (3,) representing the intersection point of the ray and the triangle, or None if there is no intersection.
    """
    # Calculate normal vector of the triangle

    normal = triangle_normal(triangle_vertices)
    # Calculate distance from ray origin to triangle plane
    t = np.dot(normal, triangle_vertices[0] - ray_origin) / np.dot(normal, ray_direction)
    # Calculate intersection point
    intersection_point = ray_origin + t * ray_direction
    # Check if intersection point is inside the triangle
    edge1 = triangle_vertices[1] - triangle_vertices[0]
    edge2 = triangle_vertices[2] - triangle_vertices[1]
    edge3 = triangle_vertices[0] - triangle_vertices[2]
    normal1 = np.cross(edge1, normal)
    normal2 = np.cross(edge2, normal)
    normal3 = np.cross(edge3, normal)
    if np.dot(ray_direction, normal1) > 0 and np.dot(intersection_point - triangle_vertices[0], normal1) > 0:
        return None
    if np.dot(ray_direction, normal2) > 0 and np.dot(intersection_point - triangle_vertices[1], normal2) > 0:
        return None
    if np.dot(ray_direction, normal3) > 0 and np.dot(intersection_point - triangle_vertices[2], normal3) > 0:
        return None
    return intersection_point


def line_line_intersection(line1, line2):
    # Extract points and directions from input lines
    p1, v1 = line1
    p2, v2 = line2

    # Calculate direction vector of the line connecting the two points
    w = p1 - p2

    # Calculate direction vectors of the two input lines
    cross = np.cross(v1, v2)
    if np.allclose(cross, [0, 0, 0]):
        # Lines are parallel, return None
        return None
    else:
        # Calculate parameters for point of intersection
        s1 = np.dot(np.cross(w, v2), cross) / np.linalg.norm(cross) ** 2
        s2 = np.dot(np.cross(w, v1), cross) / np.linalg.norm(cross) ** 2

        # Calculate intersection point
        p = 0.5 * (p1 + s1 * v1 + p2 + s2 * v2)

        return p


def line_plane_intersection(plane, ray: 'mmcore.geom.parametric.sketch.Linear', epsilon=1e-6, full_return=False):
    ray_dir = np.array(ray.direction)
    ndotu = np.array(plane.normal).dot(ray_dir)
    if abs(ndotu) < epsilon:
        if full_return:
            return None, None, None
        return None
    w = ray.start - plane.origin
    si = -np.array(plane.normal).dot(w) / ndotu
    Psi = w + si * ray_dir + plane.origin
    if full_return:
        return w, si, Psi
    return Psi


def ray_plane_intersection(ray_origin: np.ndarray, ray_direction: np.ndarray, plane, epsilon=1e-6, full_return=False):
    ndotu = np.array(plane.normal).dot(ray_direction)
    if abs(ndotu) < epsilon:
        if full_return:
            return None, None, None
        return None
    w = ray_origin - plane.origin
    si = -np.array(plane.normal).dot(w) / ndotu
    Psi = w + si * ray_direction + plane.origin
    if full_return:
        return w, si, Psi
    return Psi

class ProjectionEvent:
    def __init__(self, projection):
        super().__init__()
        self.projection = projection

    def __call__(self, point):
        return point - self.projection

    def __iter__(self):
        return iter(self.projection.tolist())

    def __array__(self):
        return self.projection


class PointPlaneProjectionEvent(ProjectionEvent):
    def __init__(self, point, plane):
        self.point, self.plane = np.array(point), plane
        self.w = self.point - np.array(self.plane.origin)
        super().__init__(np.dot(self.w, plane.normal) / 1 * plane.normal)
        # Calculate projection of w onto plane normal

    def __call__(self, point=None):
        if point is None:
            return self.point - self.projection
        return np.array(point) - self.projection


def project_point_onto_plane(point, plane_point, plane_normal):
    """
    Projects a point onto a plane in 3D space.

    Parameters:
    point (numpy.ndarray): An array of shape (3,) representing the point to be projected.
    plane_point (numpy.ndarray): An array of shape (3,) representing a point on the plane.
    plane_normal (numpy.ndarray): An array of shape (3,) representing the normal vector of the plane.

    Returns:
    numpy.ndarray: An array of shape (3,) representing the projected point.
    """
    # Calculate vector from plane point to input point
    w = point - plane_point

    # Calculate projection of w onto plane normal
    projection = np.dot(w, plane_normal) / 1 * plane_normal

    # Calculate projected point
    projected_point = point - projection

    return projected_point


def circle_intersection(c1: Circle2dTuple, c2: Circle3dTuple) -> list[Point2dTuple]:
    '''
    Computes the intersection points of two circles in the plane.

    Args:
        c1: tuple (x, y, r) representing the first circle with center (x, y) and radius r.
        c2: tuple (x, y, r) representing the second circle with center (x, y) and radius r.

    Returns:
        A list containing the coordinates of the intersection points, or None if the circles do not intersect.
    '''

    # Unpack the circle data
    r1, x1, y1 = c1
    r2, x2, y2 = c2

    # Compute the distance between the centers of the circles
    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Check if the circles intersect
    if d > r1 + r2 or d < np.abs(r2 - r1):
        raise MmodelIntersectException("Objects not intersects!")

    # Compute the coordinates of the intersection points
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(r1 ** 2 - a ** 2)
    xm = x1 + a * (x2 - x1) / d
    ym = y1 + a * (y2 - y1) / d
    xs1 = xm + h * (y2 - y1) / d
    xs2 = xm - h * (y2 - y1) / d
    ys1 = ym - h * (x2 - x1) / d
    ys2 = ym + h * (x2 - x1) / d

    return [Point2dTuple((xs1, ys1)), Point2dTuple((xs2, ys2))]


def circle_intersection2d(c1, c2):
    '''
    Computes the intersection points of two circles in the plane.

    Args:
        c1: tuple (x, y, r) representing the first circle with center (x, y) and radius r.
        c2: tuple (x, y, r) representing the second circle with center (x, y) and radius r.

    Returns:
        A list containing the coordinates of the intersection points, or None if the circles do not intersect.
    '''

    # Unpack the circle data
    x1, y1, _1, r1 = (*(c1.origin), c1.radius)
    x2, y2, _2, r2 = (*(c2.origin), c2.radius)

    # Compute the distance between the centers of the circles
    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Check if the circles intersect
    if d > r1 + r2 or d < np.abs(r2 - r1):
        return IntersectFail()

    # Compute the coordinates of the intersection points
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(r1 ** 2 - a ** 2)
    xm = x1 + a * (x2 - x1) / d
    ym = y1 + a * (y2 - y1) / d
    xs1 = xm + h * (y2 - y1) / d
    xs2 = xm - h * (y2 - y1) / d
    ys1 = ym - h * (x2 - x1) / d
    ys2 = ym + h * (x2 - x1) / d

    return [(xs1, ys1, _1), (xs2, ys2, _2)]


def global_to_custom(point, origin, x_axis, y_axis, z_axis):
    """
    Convert a point from a global coordinate system to a custom coordinate system defined by an origin and three axes.

    :param point: tuple or list of three numbers representing the coordinates of a point in the global coordinate system
    :param origin: tuple or list of three numbers representing the origin of the custom coordinate system
    :param x_axis: tuple or list of three numbers representing the x-axis of the custom coordinate system
    :param y_axis: tuple or list of three numbers representing the y-axis of the custom coordinate system
    :param z_axis: tuple or list of three numbers representing the z-axis of the custom coordinate system
    :return: tuple of three numbers representing the coordinates of the point in the custom coordinate system
    """
    # Convert all inputs to numpy arrays for easier computation

    # Compute the transformation matrix from global to custom coordinate system

    # Return the transformed point as a tuple of three numbers
    return np.array([x_axis, y_axis, z_axis]) @ (np.array(point) - np.array(origin))


def global_to_custom_old(point, origin, x_axis, y_axis, z_axis):
    """
    Convert a point from a global coordinate system to a custom coordinate system defined by an origin and three axes.

    :param point: tuple or list of three numbers representing the coordinates of a point in the global coordinate system
    :param origin: tuple or list of three numbers representing the origin of the custom coordinate system
    :param x_axis: tuple or list of three numbers representing the x-axis of the custom coordinate system
    :param y_axis: tuple or list of three numbers representing the y-axis of the custom coordinate system
    :param z_axis: tuple or list of three numbers representing the z-axis of the custom coordinate system
    :return: tuple of three numbers representing the coordinates of the point in the custom coordinate system
    """
    # Convert all inputs to numpy arrays for easier computation
    point = np.array(point)
    origin = np.array(origin)
    x_axis = np.array(x_axis)
    y_axis = np.array(y_axis)
    z_axis = np.array(z_axis)

    # Compute the transformation matrix from global to custom coordinate system
    transform_matrix = np.column_stack([x_axis, y_axis, z_axis]).T

    # Subtract the origin from the point and transform the result using the transformation matrix
    transformed_point = transform_matrix.dot(point - origin)

    # Return the transformed point as a tuple of three numbers
    return transformed_point


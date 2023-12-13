import abc
import dataclasses
import typing
from collections import namedtuple

import itertools
import numpy as np

from mmcore.collections import DCLL
from mmcore.exceptions import MmodelIntersectException
from mmcore.func import vectorize
from mmcore.geom.parametric.wrappers import AbstractParametricCurve

SPEEDUPS = False
try:
    import mmvec

    SPEEDUPS = True
except ModuleNotFoundError as err:
    SPEEDUPS = False

except ImportError as err:
    SPEEDUPS = False

TOLERANCE = 1e-8
from scipy.optimize import minimize, fsolve
from scipy.spatial.distance import euclidean

ProxPointSolution = namedtuple("ProxPointSolution", ["pt1", "pt2", "t1", "t2"])
ClosestPointSolution = namedtuple("ClosestPointSolution", ["pt", "t", "distance"], defaults=[None, None, None])
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
PointUnion = typing.Union[list[float], tuple[float, float, float], np.ndarray]
CurvesIntersectSolution = namedtuple('CurvesIntersectSolution', ['t0', 't1', 'pt'])
Ray = namedtuple('Ray', ['origin', 'normal'])

def translate_point(origin, direction, distance):
    return [o + (n * distance) for o, n in zip(origin, direction)]


class ParametricLine(AbstractParametricCurve):
    obj: LineTuple

    def solve(self, obj, t) -> list[float]:
        start, direction = obj
        return translate_point(start, direction, t)



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


def solve2d_newthon(f, MAX_ITERATIONS=40, eps=1e-12):
    def solve(x, y, a=0, b=0):
        err2 = None
        for i in range(MAX_ITERATIONS):
            p = f(a, b)
            tx = p[0] - x
            ty = p[1] - y
            if abs(tx) < eps and abs(ty) < eps:
                break

            h = tx * tx + ty * ty
            if err2 is not None:
                if h > err2:
                    a -= da / 2
                    b -= db / 2
                    continue
            err2 = h

            ea = -eps if a > 0 else eps
            eb = -eps if b > 0 else eps

            print(ea, eb)
            pa = f(a + ea, b)
            pb = f(a, b + eb)
            print(a + ea, b, pa)
            print(a, b + eb, pb)
            dxa = (pa[0] - p[0]) / ea
            dya = (pa[1] - p[1]) / ea
            dxb = (pb[0] - p[0]) / eb
            dyb = (pb[1] - p[1]) / eb

            D = dyb * dxa - dya * dxb
            l = (0.5 / D if abs(D) < 0.5 else 1 / D)
            da = (ty * dxb - tx * dyb) * l
            db = (tx * dya - ty * dxa) * l
            a += da
            b += db

            if (abs(da) < eps) and (abs(db) < eps):
                break

        return [a, b]

    return solve


class CurveSolver2d:
    def __init__(self, f1, f2, eps=1e-6):

        self.eps = eps
        self.f1, self.f2 = f1, f2
        self.bounds = self.f1.bounds if hasattr(self.f1, 'bounds') else (0.0, 1.0), self.f2.bounds if hasattr(self.f2,
                                                                                                              'bounds') else (
        0.0, 1.0)

    def f(self, x):
        # print(x)
        a, b = self.f1(x[0]), self.f2(x[1])
        # print(a,b)

        # print(ews)
        return [a[0] - b[0], a[1] - b[1]]

    def __call__(self, x0=(0.5, 0.5), return3d=False) -> CurvesIntersectSolution:
        def fun(x):

            return np.array(self.f(x))

        t0, t1 = fsolve(fun, list(x0)).tolist()
        pt = self.f1(t0)
        if isinstance(pt, np.ndarray):
            pt = pt.tolist()
        if return3d:
            return CurvesIntersectSolution(t0, t1, list(pt) + [0])
        return CurvesIntersectSolution(t0, t1, pt)

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

    return ClosestPointSolution(p_closest, t=t)


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

    # Calculate direction vectors of the two input lines
    _cross = np.cross(v1, v2)
    if np.allclose(_cross, [0, 0, 0]):
        # Lines are parallel, return None
        return None
    else:
        w = p1 - p2
        # Calculate parameters for point of intersection
        s1 = np.dot(np.cross(w, v2), _cross) / np.linalg.norm(_cross) ** 2
        s2 = np.dot(np.cross(w, v1), _cross) / np.linalg.norm(_cross) ** 2

        return 0.5 * np.array(p1 + s1 * v1 + p2 + s2 * v2)


def line_line_intersection2d(line1, line2):
    print(line1, line2)
    line1, line2 = np.array([line1, line2])
    x1, y1 = line1[0, :2]
    x2, y2 = x1 + line1[1, 0], y1 + line1[1, 1]
    x3, y3 = line2[0, :2]
    x4, y4 = x3 + line2[1, 0], y3 + line2[1, 1]

    # Calculate the denominator
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Check if the lines are parallel
    if denominator == 0:
        return None

    # Calculate the intersection point
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    return [x, y]


def pts_line_line_intersection2d(line1, line2):
    line1, line2 = np.array([line1, line2])
    x1, y1 = line1[0, :2]
    x2, y2 = line1[1, :2]
    x3, y3 = line2[0, :2]
    x4, y4 = line2[1, :2]

    # Calculate the denominator
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Check if the lines are parallel
    if denominator == 0:
        return None

    # Calculate the intersection point
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    return [x, y]


def pts_line_line_intersection2d_as_3d(line1, line2):
    """
    Given two 2D lines as input, this method calculates the intersection point in 3D space.

    :param line1: The first line represented as a 2D numpy array with shape (2,n), start& end points. Each row represents a point in the line.
    :type line1: numpy.ndarray
    :param line2: The second line represented as a 2D numpy array with shape (2,n). Each row represents a point in the line.
    :type line2: numpy.ndarray
    :return: The intersection point represented as a 3D numpy array with shape (3,). The z-coordinate is always 0.0.
    :rtype: numpy.ndarray

    """
    line1, line2 = np.array([line1, line2])
    x1, y1 = line1[0, :2]
    x2, y2 = line1[1, :2]
    x3, y3 = line2[0, :2]
    x4, y4 = line2[1, :2]

    # Calculate the denominator
    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    # Check if the lines are parallel
    if denominator == 0:
        return np.nan

    # Calculate the intersection point
    x = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / denominator
    y = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / denominator

    return np.array([x, y, 0.0], float)

def bounded_line_intersection2d(line1, line2):
    res = np.array(pts_line_line_intersection2d_as_3d(line1, line2))

    try:

        l1 = point_line_bounded_distance(res, line1[0], line1[1])
        l2 = point_line_bounded_distance(res, line2[0], line2[1])

        if (l1 is not None) and (l2 is not None):
            d1, pt1, t1 = l1
            d2, pt2, t2 = l2

            return res, (t1, t2), (d1, d2)
    except TypeError:
        pass

# Example usage

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


def point_to_plane_distance(point, plane_point, plane_normal):
    return np.dot(point - plane_point, plane_normal)


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


@vectorize(excluded=[0, 1], signature='()->(i)')
def pde_central(f, h, t):
    return (f(t + h) - f(t - h)) / (2 * h)


@vectorize(excluded=[0, 1], signature='()->(i)')
def pde_forward(f, h, t):
    return (f(t + h) - f(t)) / h


@vectorize(excluded=[0, 1], signature='()->(i)')
def pde_backward(f, h, t):
    return (f(t) - f(t - h)) / h


@vectorize(excluded=[0, 1], signature='()->(i)')
def pde_central_s(f, h, t):
    return (f(t + h) - f(t - h)) / (2 * h)



PDE_METHODS = dict(central=pde_central, forward=pde_forward, backward=pde_backward)


@vectorize(excluded=['func', 'method', 'h', 'transpose'], signature='(),()->(i)')
def pde_offset_2d_as_3d(dist, t, func=None, method=pde_central, h=0.01):
    m = method
    xyz = func(t)
    d = m(func, h, t)
    norm = _ns(d[0], d[1])
    res = np.array([xyz[0] + (dist * d[1] / norm), xyz[1] - (dist * d[0] / norm), xyz[2]])

    return res.T


def pde_offset2(dist, func=None, method="central", h=0.01):
    m = pde_central_s

    def ff(t):
        xyz = func(t)
        d = m(func, h, t)
        norm = _ns(d[0], d[1])
        return np.array([xyz[0] + (dist * d[1] / norm), xyz[1] - (dist * d[0] / norm), xyz[2]])

    return ff


@vectorize(excluded=['func', 'method', 'h', 'transpose'], signature='(),()->(i)')
def pde_offset_2d(dist, t, func=None, method="central", h=0.01):
    m = PDE_METHODS[method]
    xyz = func(t)
    dx, dy = m(func, h, t)
    return np.array([xyz[0] + (dist * dy / _ns(dx, dy)), xyz[1] - (dist * dx / _ns(dx, dy))])


@vectorize(excluded=['func', 'method', 'h'], signature='(),(j)->(i,j)')
def pde_offset_3d(dist, t, func=None, method="central", h=0.01):
    m = PDE_METHODS[method]

    xyz = func(t)
    d = m(func, h, t)

    return np.array([xyz[0] + (dist * d[1] / _ns2(*d)),
                     xyz[1] - (dist * d[0] / _ns2(*d)),
                     xyz[2] + (dist * d[2] / _ns2(*d))])


@vectorize(excluded=['func', 'method', 'h'], signature='(),(j)->(i,j)')
def pde_offset_3d(dist, t, func=None, method="central", h=0.01):
    m = PDE_METHODS[method]

    xyz = func(t)
    d = m(func, h, t)

    return np.array([xyz[0] + (dist * d[1] / _ns2(*d)),
                     xyz[1] - (dist * d[0] / _ns2(*d)),
                     xyz[2] + (dist * d[2] / _ns2(*d))])
class Derivative:
    def __init__(self, f, method="central", h=0.01):
        super().__init__()
        self._f = f
        self.h = h
        self.method = method

    def __call__(self, t):
        return getattr(self, self.method)(t)

    def central(self, t):

        return (self._f(t + self.h) - self._f(t - self.h)) / (2 * self.h)

    def forward(self, t):
        return (self._f(t + self.h) - self._f(t)) / self.h

    def backward(self, t):
        return (self._f(t) - self._f(t - self.h)) / self.h


def _ns(dx, dy):
    return np.sqrt((dx ** 2) + (dy ** 2))


def _ns2(dx, dy, dz):
    return np.sqrt((dx ** 2) + (dy ** 2) + (dz ** 2))


def offset_curve_2d(c, d):
    df = Derivative(c)

    def wrap(t):
        x, y = c(t)
        dx, dy = df(t)
        ox = x + (d * dy / _ns(dx, dy))
        oy = y - (d * dx / _ns(dx, dy))
        return [ox, oy]

    return wrap


def offset_curve_3d_as_2d(c, d):
    df = Derivative(c)

    def wrap(t):
        x, y, z = c(t)
        dx, dy = df(t)
        ox = x + (d * dy / _ns(dx, dy))
        oy = y - (d * dx / _ns(dx, dy))
        return [ox, oy, z]

    return wrap


def variable_offset_curve_3d_as_2d(c):
    df = Derivative(c)

    def wrap(t, d):
        x, y, z = c(t)
        dxdy = df(t)
        dst = _ns(dxdy[0], dxdy[1])
        ox = x + (d * dxdy[1] / dst)
        oy = y - (d * dxdy[0] / dst)
        return [ox, oy, z]

    return wrap

def offset_curve_3d(c, d):
    df = Derivative(c)

    def wrap(t):
        x, y, z = c(t)
        dx, dy, dz = df(t)
        ox = x + (d * dy / _ns2(dx, dy, dz))
        oy = y - (d * dx / _ns2(dx, dy, dz))
        oz = z + (d * dz / _ns2(dx, dy, dz))
        return [ox, oy, oz]

    return wrap


def offset_curve_3d_np(c, d):
    df = Derivative(c)

    def wrap(t):
        x, y, z = c(t)
        dx, dy, dz = df(t)
        ox = x + (d * dy / _ns2(dx, dy, dz))
        oy = y - (d * dx / _ns2(dx, dy, dz))
        oz = z + (d * dz / _ns2(dx, dy, dz))
        return np.array([ox, oy, oz])

    return wrap

def line_to_func(line: LineTuple):
    return lambda t: np.array(line[1]) * t + np.array(line[0])


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


def line_from_ends(start, end) -> LineTuple:
    start, end = np.array(start), np.array(end)
    return start, end - start


def line_to_ends(line: LineTuple) -> np.ndarray:
    start, direction = np.array(line)
    end = start + direction
    return np.array([start, end])


def circle_line_intersection_2d(circle_center, circle_radius, pt1, pt2, full_line=True, tangent_tol=1e-9):
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.
    :param circle_center: The (x, y) location of the circle center
    :param circle_radius: The radius of the circle
    :param pt1: The (x, y) location of the first point of the segment
    :param pt2: The (x, y) location of the second point of the segment
    :param full_line: True to find intersections along full line - not just in the segment.  False will just return intersections within the segment.
    :param tangent_tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.
    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
    """
    (p1x, p1y), (p2x, p2y), (cx, cy) = pt1, pt2, circle_center
    (x1, y1), (x2, y2) = (p1x - cx, p1y - cy), (p2x - cx, p2y - cy)
    dx, dy = (x2 - x1), (y2 - y1)
    dr = (dx ** 2 + dy ** 2) ** .5
    big_d = x1 * y2 - x2 * y1
    discriminant = circle_radius ** 2 * dr ** 2 - big_d ** 2
    if discriminant < 0:  # No intersection between circle and line
        return []
    else:  # There may be 0, 1, or 2 intersections with the segment
        intersections = [
            (cx + (big_d * dy + sign * (-1 if dy < 0 else 1) * dx * discriminant ** .5) / dr ** 2,
             cy + (-big_d * dx + sign * abs(dy) * discriminant ** .5) / dr ** 2)
            for sign in ((1, -1) if dy < 0 else (-1, 1))]  # This makes sure the order along the segment is correct
        if not full_line:  # If only considering the segment, filter out intersections that do not fall within the segment
            fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in
                                      intersections]
            intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(
                discriminant) <= tangent_tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections


from math import sqrt, acos, atan2, sin, cos


# Example values
def tangents_lines(pt, circle: Circle2dTuple):
    Px, Py = pt
    a, Cx, Cy = circle

    b = sqrt((Px - Cx) ** 2 + (Py - Cy) ** 2)  # hypot() also works here
    th = acos(a / b)  # angle theta
    d = atan2(Py - Cy, Px - Cx)  # direction angle of point P from C
    d1 = d + th  # direction angle of point T1 from C
    d2 = d - th  # direction angle of point T2 from C

    T1x = Cx + a * cos(d1)
    T1y = Cy + a * sin(d1)
    T2x = Cx + a * cos(d2)
    T2y = Cy + a * sin(d2)
    return [T1x, T1y], [T2x, T2y]


def closest_point_on_circle_2d(pt, circle: Circle2dTuple):
    r, ox, oy = circle
    un = unit(np.array([pt[0] - ox, pt[1] - oy]))
    return ClosestPointSolution(
        [ox + (un[0] * r), oy + (un[1] * r)],
        (1 / (2 * np.pi)) * np.arccos(un[0]), None)


ArcTuple = namedtuple("ArcTuple", ['circle', 't0', 't1'])


def fillet_lines(pts, r):
    a, b, c = pts

    ln1, ln2 = line_from_ends(a, b), line_from_ends(b, c)

    ofl1, ofl2 = offset_line_2d(ln1, r), offset_line_2d(ln2, r)

    ln11, ln12 = line_from_ends(ofl1(0), ofl1(1)), line_from_ends(ofl2(0), ofl2(1))

    pt = line_line_intersection2d(ln11, ln12)

    cpt1, cpt2 = closest_point_on_line(pt, ln1), closest_point_on_line(pt, ln2)

    return LineStartEndTuple(a, cpt1.pt), Circle2dTuple(r, *pt), LineStartEndTuple(cpt2.pt, c)
def offset_line_2d(line, distance):
    return offset_curve_2d(line_to_func(line), distance)


def offset_line_3d(line, distance):
    ln = [line[0][0], line[0][1]], [line[1][0], line[1][1]]

    return lambda t: offset_curve_2d(line_to_func(ln), distance)(t) + [line_to_func(line)(t)[2]]


from mmcore.geom.vectors import norm, unit
def line_point_param(ln: LineTuple, pt: list[float]):
    """
    Pass line and point, return t param
    Parameters
    ----------
    ln
    pt

    Returns
    -------

    """

    w = np.array(pt) - ln[0]
    n = norm(ln[1])
    v1 = ln[1] / n
    v2 = w / n
    return 0.5 * np.dot(v2, v1)


@vectorize(signature='(i,j)->(i,k,j)')
def polyline_to_lines(pln_points):
    return np.stack([pln_points, np.roll(pln_points, -1, axis=0)], axis=1)


@vectorize(signature='(i,j)->(i,k,j)')
def polyline_to_lines_forward(pln_points):
    return np.stack([np.roll(pln_points, -1, axis=0), pln_points], axis=1)

def polyline_to_lines_vectors(pln_points):
    return np.stack([pln_points, np.roll(pln_points, -1, axis=0) - pln_points], axis=1)


def vector_start_end(start, end):
    print(start, end)
    return end - start


def distance(p0, p1):
    return norm(vector_start_end(p0, p1))


def scale_vector(v, sc):
    return v * sc


def point_line_bounded_distance(pnt, start, line_vec):
    pnt_vec = vector_start_end(start, pnt)
    line_len = norm(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale_vector(pnt_vec, 1.0 / line_len)
    t = np.dot(line_unitvec, pnt_vec_scaled)

    if t < 0.0:

        pass
    elif t > 1.0:
        pass
    else:

        nearest = scale_vector(line_vec, t)
        dist = distance(nearest, pnt_vec)

        return (dist, nearest + dist, t)


def intersect_polylines(pln1, pln2, closed=(False, False)):
    lines1, lines2 = polyline_to_lines_vectors(pln1[..., :2]), polyline_to_lines_vectors(pln2[..., :2])
    lines1 = lines1 if closed[0] else lines1[:-1]
    lines2 = lines2 if closed[1] else lines2[:-1]

    for i, line1 in enumerate(lines1):
        for j, line2 in enumerate(lines2):
            print(line1, line2)
            res = bounded_line_intersection2d(line1, line2)

            if res is not None:
                res2, (t1, t2), (d1, d2) = res

                yield res2, (t1 + i, t2 + j), (i, j)


def intersect_lines(lines):
    for i, (line1, line2) in enumerate(itertools.pairwise(lines)):

        res = bounded_line_intersection2d(line1, line2)

        if res is not None:
            res2, (t1, t2), (d1, d2) = res

            yield res2, (t1, t2), (i, i + 1)




def variable_line_offset_2d(bounds: list[PointUnion], distances: list[float]):
    if not all(len(bnd) == 2 for bnd in bounds):
        bounds = [[bn[0], bn[1]] for bn in bounds]
    ll = DCLL.from_list(bounds)
    dl = DCLL.from_list(distances)

    def get_line_chain(nd):
        a, b, c = np.array(nd.prev.data), np.array(nd.data), np.array(nd.next.data)
        return line_from_ends(a, b), line_from_ends(b, c)

    node = ll.head.prev
    node_d = dl.head.prev

    for i in range(len(bounds)):
        node = node.next
        node_d = node_d.next
        la, lb = get_line_chain(node)
        la1, lb1 = offset_line_2d(la, node_d.prev.data), offset_line_2d(lb, node_d.data)
        pt = line_line_intersection2d(line_from_ends(la1(0), la1(1)), line_from_ends(lb1(0), lb1(1)))
        if len(bounds[i]) == 3:
            yield pt + [bounds[i][-1]]
        else:
            yield pt + [0]


def ptline_to_func(start, end):
    start, end = np.array(start), np.array(end)
    direction = end - start

    def wrap(t):
        return direction * t + start

    return np.vectorize(wrap, signature='()->(i)')


def perp_vector2d(vec): ...


def polygon_variable_offset(points: np.ndarray, dists: np.ndarray):
    """
    :param points: List of 2D points representing the polygon.
    :type points: np.ndarray[(n,3), float]
    :param dists: List of distances for offsetting each side of the polygon.
    :type dists: np.ndarray[(n,2), float]
    :return: Generator that yields 3D points representing the intersections of the offset lines.
    :rtype: Generator[np.ndarray[3, float]]

    Example:
    ----
    >>> from mmcore.geom.parametric.algorithms import polygon_variable_offset
    >>> pts=np.array([[33.049793, -229.883303, 0],
    ...               [132.290583, -165.409427, 0],
    ...               [48.220282, 27.548631, 0],
    ...               [-115.077733, -43.599024, 0],
    ...               [-44.627307, -205.296759, 0]])
    >>> dists=np.zeros((5,2))
    >>> dists[0]=4  # Set negative values for the offset to the inside of the polygon.
    >>> dists[2]=1
    >>> dists[-1]=2
    >>> dists
    array([[4., 4.],
           [0., 0.],
           [1., 1.],
           [0., 0.],
           [2., 2.]])
    >>> *res,=polygon_variable_offset(pts, dists)
    >>> np.array(res)
    array([[  30.28406152, -226.91009151,    0.        ],
           [ 130.67080779, -161.69172081,    0.        ],
           [  48.61970922,   26.63186609,    0.        ],
           [-114.67830577,  -44.5157889 ,    0.        ],
           [ -45.68750838, -202.86338603,    0.        ]])
    >>> dists[-1,0]= 0. # To set a variable offset per side change one of the values.
    >>> dists
    array([[4., 4.],
           [0., 0.],
           [1., 1.],
           [0., 0.],
           [0., 2.]])
    >>> *res,=polygon_variable_offset(pts, dists)
    >>> np.array(res)
    array([[  30.26926633, -226.9054085 ,    0.        ],
           [ 131.48297158, -163.55579819,    0.        ],
           [  48.61970922,   26.63186609,    0.        ],
           [-114.67830577,  -44.5157889 ,    0.        ],
           [ -45.68750838, -202.86338603,    0.        ]])
    """

    if points.shape[0] == 0:
        raise ValueError("points parameter cannot be empty")

        # check if all points are 2D
    if points.shape[1] != 3:
        raise ValueError("All points must be 3-dimensional")

        # check if dists is of the same length as points and is a 2D array
    if dists.shape[0] != points.shape[0] or len(dists.shape) != 2:
        raise ValueError("dists must be a 2D array with the same number of rows as the number of points")

    lines = polyline_to_lines(points)
    sides = [variable_offset_curve_3d_as_2d(ptline_to_func(line[0], line[1])) for line in lines]
    offset_lines = DCLL.from_list([(side(0.0, dists[i][0]), side(1.0, dists[i][1])) for i, side in enumerate(sides)])
    current = offset_lines.head

    for i in range(len(offset_lines)):
        yield pts_line_line_intersection2d_as_3d(current.prev.data, current.data)
        current = current.next

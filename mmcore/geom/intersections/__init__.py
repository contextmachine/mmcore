"""
:mod:`mmcore.geom.intersections`
In this module implements different methods with the same name - intersect. Each method is used to
determine intersection points between different geometric shapes (like spheres, planes, and circles) and rays or lines.

Each variation of the method is applicable to different pairs of geometric shapes due to Python's multiple dispatch
decorator. The specific method that gets called depends on the types of the first two arguments.

Here's a brief overview of each of the intersect methods:
* Intersection of Sphere and Ray:
    This method determines the points at which a 3-dimensional ray intersects with a sphere.
     It uses Scipy's fsolve function to find the roots of the equations describing the intersection.
* Intersection of Plane and Ray:
    This method computes an intersection point between a plane and a ray.
    It handles the case when the ray is nearly parallel to the normal of the plane
    by returning an array of 6 empty values.
* Intersection of Circle and Ray:
    It calculates intersection points between a 2D circle and a ray.
    If the discriminant of the equation is negative, the function returns an empty list,
    indicating no intersection. If there's only one intersection or two intersections at the same point,
    it returns a list with only one point, else it provides two points of intersections.
* Intersection of Circle and Line: Similar to the intersection of a circle with a ray,
    this function calculates intersection points between a circle and a line segment defined by two points (Line).
    It returns the line segment's intersection points with the circle using a similar discriminant approach
    as the sphere-ray intersection.

All methods are decorated with vectorize from mmcore.func which allows parallelized operations on NumPy arrays.
"""
import os

import numpy as np
from multipledispatch import dispatch
from scipy.linalg import solve
from scipy.optimize import fsolve

from mmcore.func import vectorize
from mmcore.geom.circle import Circle
from mmcore.geom.interfaces import Ray
from mmcore.geom.line import Line
from mmcore.geom.plane import Plane, is_parallel
from mmcore.geom.sphere import Sphere
from mmcore.geom.tolerance import *
from mmcore.geom.vec import *

INTERSECTION_TOLERANCE = TOLERANCE

DEBUG_MODE = os.getenv("DEBUG_MODE")
@dispatch(Sphere, Ray)
@vectorize(excluded=[0, 2], signature='(i,j)->(k)')
def intersect(sphere: Sphere, ray: Ray, tol=INTERSECTION_TOLERANCE):

    """
    :param sphere: The sphere object to intersect with
    :type sphere: Sphere

    :param ray: The ray object to intersect with the sphere
    :type ray: Ray

    :param tol: The tolerance value for the intersection calculation (optional, defaults to INTERSECTION_TOLERANCE)
    :type tol: float

    :return: The intersection point between the sphere and the ray, along with the parameter values u, v and t
    :rtype: numpy.ndarray
    """

    ray_start, ray_vector = ray

    def wrap(x):
        t, (u, v) = x[0], x[1:]
        return ray_start + ray_vector * t - sphere(u, v)

    t, u, v = fsolve(wrap, [0, 0, 0], full_output=False, xtol=tol)
    return np.append(sphere.evaluate(u, v), [t, u, v])



@dispatch(Plane, Ray)
@vectorize(excluded=[0, 2], signature='(i,j)->(k)')
def intersect(plane: Plane, ray: Ray, tol=INTERSECTION_TOLERANCE):
    """
    :param plane: The plane to intersect with the ray
    :type plane: Plane
    :param ray: The ray to intersect with the plane
    :type ray: Ray
    :param tol: The tolerance for determining if the ray and plane are parallel
    :type tol: float
    :return: The intersection point between the plane and the ray, along with additional information about the intersection
    :rtype: numpy.ndarray

    """
    ray_origin, ray_direction = ray
    dotu = np.array(plane.normal).dot(ray_direction)
    if abs(dotu) < tol:
        return np.empty(6)
    w = ray_origin - plane.origin
    dotv = -np.array(plane.normal).dot(w)
    si = dotv / dotu
    Psi = w + si * ray_direction + plane.origin
    return np.array([*Psi, si, dotu, dotv])


@dispatch(Circle, Ray)
@vectorize(excluded=[0, 2], signature='(i,j)->(k)')
def intersect(circle: Circle, ray: Ray, tol=INTERSECTION_TOLERANCE):
    """
    :param circle: Circle object representing a circle
    :type circle: Circle
    :param ray: Ray object representing a ray
    :type ray: Ray
    :param tol: Intersection tolerance
    :type tol: float
    :return: List of intersection points between the circle and the ray
    :rtype: List[Tuple[float, float]]

    This method calculates the intersection points between a circle and a ray. It takes a Circle object and a Ray object as parameters, along with an optional intersection tolerance value
    *.

    The Circle object is used to represent the circle, and it contains information about the origin (center) and radius of the circle. The Ray object represents the ray, and it contains
    * information about a point on the ray and the direction of the ray.

    The method calculates the points of intersection between the circle and the ray, and returns them as a list of tuples, where each tuple represents an intersection point as (x, y) coordinates
    *.

    If there are no intersections between the circle and the ray, an empty list is returned.

    The intersection tolerance is used to determine if a point lies on the circle or not. If the discriminant of the equation is within the tolerance, the method considers it as a point
    * on the circle. In this case, if there are two intersections and their discriminant is within the tolerance, only one of the points is returned.

    Example usage:

    circle = Circle((0, 0), 5)
    ray = Ray((0, 0), (1, 1))
    intersections = intersect(circle, ray)
    print(intersections)
    # Output: [(3.5355339059327373, 3.5355339059327373)]
    """
    pt1, vec = np.array(ray)[:, :2]
    pt2 = pt1 + vec
    circle_center, circle_radius = np.array([circle.origin, circle.r])[:, :2]
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

        if len(intersections) == 2 and abs(
                discriminant) <= tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections


@dispatch(Circle, Line)
def intersect(circle: Circle, line: Line, tol=INTERSECTION_TOLERANCE):
    """
    :param circle: Circle object representing a circle
    :type circle: Circle
    :param line: Line object representing a line segment
    :type line: Line
    :param tol: Tolerance for determining if points are close enough to be considered intersecting
    :type tol: float
    :return: List of intersection points between the circle and the line segment
    :rtype: list of tuples

    This method calculates the intersection points between a circle and a line segment. It takes in a Circle object representing the circle, a Line object representing the line segment
    *, and a tolerance value for determining if points are close enough to be considered intersecting. It returns a list of tuples representing the intersection points between the circle
    * and the line segment.

    Example usage:

    ```python
    circle = Circle(origin=(0, 0), r=5)
    line = Line((0, 0), (10, 10))
    tolerance = 0.01

    intersections = intersect(circle, line, tol=tolerance)
    print(intersections)
    ```
    """

    pt1, pt2 = line
    pt1, pt2 = pt1[:2], pt2[:2]
    circle_center, circle_radius = circle.origin[:2], circle.r
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

        fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in
                                  intersections]
        intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(
                discriminant) <= tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections


@dispatch(Line, Line)
@vectorize(excluded=[0, 2], signature='(i,j)->(k)')
def intersection(line1, line2):
    return line1.unbound_intersection(line2)


@dispatch(Plane, Plane)
@vectorize(signature='(),()->(j, i)')
def intersect(plane1: Plane, plane2: Plane):
    """
    :param plane1: First plane
    :type plane1: Plane
    :param plane2: Second plane
    :type plane2: Plane
    :return: Array containing two points that represent the intersection line of the two planes
    :rtype: numpy.ndarray
    """
    if DEBUG_MODE:
        if is_parallel(plane1, plane2):
            raise ValueError("Can't intersect. Planes are parallel.")
    cross_vector = cross(unit(plane1.normal), unit(plane2.normal))
    A = np.array([plane1.normal, plane2.normal, cross_vector])
    d = np.array([-plane1.d, -plane2.d, 0.]).reshape(3, 1)
    intersection_point = np.linalg.solve(A, d).T
    return np.array([intersection_point[0], (intersection_point + cross_vector)[0]], dtype=float)


import math


# ------ misc fucntions ----------

def sign(x):
    """Returns 1 if x>0, return -1 if x<=0"""
    if x > 0:
        return 1
    else:
        return -1


# -------- 3D intersections ---------------


@dispatch(Line, Line, bool)
@vectorize(excluded='bounded', signature='(),()->(i)')
def intersect(self: Line, other: Line, bounded=True):
    if bounded:
        return _line_line_bounded(self._array[:2], other._array[:2])
    else:
        return self.unbounded_intersect(other)


def _line_line_bounded(self: np.ndarray[(2, 3), float], other: np.ndarray[(2, 3), float]):
    (x1, y1, z1), (x2, y2, z2) = self
    (x3, y3, z3), (x4, y4, z4) = other
    if z1 == z2:
        dr = z1
    else:
        raise ValueError('Lines not intersection in (0.0 <= t <= 1.0) bounds')

    A = np.array([[x2 - x1, x4 - x3], [y2 - y1, y4 - y3]])
    b = np.array([x3 - x1, y3 - y1])

    return np.append(solve(A, b), dr)


@vectorize(excluded='bounded', signature='(i),(i),(i),(i)->(i)')
def _line_line_bounded_vec(start1: np.ndarray[3, float], end1: np.ndarray[3, float], start2: np.ndarray[3, float],
                           end2: np.ndarray[3, float]):
    return _line_line_bounded(np.array((start1, end1), float),
                              np.array((start2, end2), float)
                              )


@vectorize(excluded='bounded', signature='(),()->(i)')
def intersect_lines(self: Line, other: Line, bounded=True):
    if bounded:
        return _line_line_bounded(self._array[:2], other._array[:2])
    else:
        return self.unbounded_intersect(other)


def line_line_bounded(first, second):
    return _line_line_bounded_vec(first[..., 0, :], first[..., 1, :], second[..., 0, :], second[..., 1, :])


def sphere_sphere_sphere(p1, r1, p2, r2, p3, r3):
    """Intersect three spheres, centered in p1, p2, p3 with radius r1,r2,r3 respectively. 
       Returns a list of zero, one or two solution points.
    """
    solutions = zero_solution((2, 3))
    # plane though p1, p2, p3
    n = cross(p2 - p1, p3 - p1)
    n = n / norm(n)
    # intersect circles in plane
    cp1 = np.array([0.0, 0.0])
    cp2 = np.array([norm(p2 - p1), 0.0])
    cpxs = circle_circle(cp1, r1, cp2, r2)
    print(cpxs)
    if is_zero_solution(cpxs):
        return solutions

    # px, rx, nx is circle 
    px = p1 + (p2 - p1) * cpxs[0][0] / norm(p2 - p1)
    rx = np.abs(cpxs[0][1])
    print(px, rx, cpxs)
    # plane of intersection cicle
    nx = p2 - p1
    nx = nx / norm(nx)
    # print "px,rx,nx:",px,rx,nx
    # py = project p3 on px,nx
    dy3 = dot(p3 - px, nx)
    py = p3 - (nx * dy3)
    if tol_gt(dy3, r3):
        return solutions
    ry = np.sin(np.arccos(np.abs(dy3 / r3))) * r3
    # print "py,ry:",py,ry
    cpx = np.array([0.0, 0.0])
    cpy = np.array([norm(py - px), 0.0])
    print(cpx, rx, cpy, ry)
    cp4s = circle_circle(cpx, rx, cpy, ry)
    print(cp4s)
    if not is_zero_solution(cp4s):
        for i, cp4 in enumerate(filter_zero_solution(cp4s)):
            p4 = px + (py - px) * cp4[0] / norm(py - px) + n * cp4[1]
            print(solutions)
            solutions[i] = p4

    return solutions


# ------- 2D intersections ----------------

def circle_circle(p1, r1, p2, r2):
    """
    Computes the intersection points of two circles in the plane.

    Args:
        c1: tuple (x, y, r) representing the first circle with center (x, y) and radius r.
        c2: tuple (x, y, r) representing the second circle with center (x, y) and radius r.

    Returns:
        A list containing the coordinates of the intersection points, or None if the circles do not intersect.
    """

    solution = zero_solution((2, 3))
    # Unpack the circle dat
    (x1, y1, *_), r1 = p1, r1
    (x2, y2, *_), r2 = p2, r2

    # Compute the distance between the centers of the circles
    d = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Check if the circles intersect
    if d > r1 + r2 or d < np.abs(r2 - r1):
        return solution

    # Compute the coordinates of the intersection points
    a = (r1 ** 2 - r2 ** 2 + d ** 2) / (2 * d)
    h = np.sqrt(r1 ** 2 - a ** 2)
    xm = x1 + a * (x2 - x1) / d
    ym = y1 + a * (y2 - y1) / d
    xs1 = xm + h * (y2 - y1) / d
    xs2 = xm - h * (y2 - y1) / d
    ys1 = ym - h * (x2 - x1) / d
    ys2 = ym + h * (x2 - x1) / d

    return np.array([(xs1, ys1, 0.0), (xs2, ys2, 0.0)])


def circle_line(p1, r, p2, v) -> np.ndarray[(2, 3), np.dtype[float]]:
    """
    Intersect a circle (p1,r) with line (p2,v)
    where p1, p2 and v are 2-vectors, r is a scalar
    Returns a list of zero, one or two solution points
    """
    solution = np.zeros((2, 3), float)
    solution[:] = np.nan
    p = p2 - p1
    d2 = v[0] * v[0] + v[1] * v[1]
    D = p[0] * v[1] - v[0] * p[1]
    E = r * r * d2 - D * D
    if d2 > 0 and E > 0:
        sE = math.sqrt(E)
        x1 = p1[0] + (D * v[1] + sign(v[1]) * v[0] * sE) / d2
        x2 = p1[0] + (D * v[1] - sign(v[1]) * v[0] * sE) / d2
        y1 = p1[1] + (-D * v[0] + abs(v[1]) * sE) / d2
        y2 = p1[1] + (-D * v[0] - abs(v[1]) * sE) / d2
        solution[0] = x1, y1, 0.
        solution[1] = x2, y2, 0.
        return solution
    elif np.allclose(E, 0):
        solution[0, 0] = p1[0] + D * v[1] / d2
        solution[0, 1] = p1[1] + -D * v[0] / d2
        solution[0, 2] = 0.
        # return [np.array([x1,y1]), np.array([x1,y1])]

        return solution
    else:
        return solution


def zero_solution(shape):
    solution = np.zeros(shape, dtype=float)
    solution[:] = np.nan
    return solution


def filter_zero_solution(arr):
    sh = arr.shape
    arr2 = arr[np.not_equal(np.isnan(arr), True)]
    if len(arr2.shape) < len(sh):
        return arr2.reshape((1, *arr2.shape))
    return arr2


def circle_ray(p1, r, p2, v):
    """
    Intersect a circle (p1,r) with ray (p2,v) (a half-line)
    where p1, p2 and v are 2-vectors, r is a scalar
    Returns a list of zero, one or two solutions.
    """
    solution = zero_solution((2, 3))
    al = circle_line(p1, r, p2, v)
    if not is_zero_solution(al):

        for i, s in enumerate(filter_zero_solution(al)):
            if tol_gte(dot(s - p2, v), 0):  # gt -> gte 30/6/2006
                solution[i, :2] = s
    return solution


def line_line(p1, v1, p2, v2):
    """Intersect line though p1 direction v1 with line through p2 direction v2.
       Returns a list of zero or one solutions
    """
    bad_solution = np.zeros(3, dtype=float)
    bad_solution[:] = np.nan
    if np.allclose((v1[0] * v2[1]) - (v1[1] * v2[0]), 0):
        return bad_solution
    elif not np.allclose(v2[1], 0.0):
        d = p2 - p1
        r2 = -v2[0] / v2[1]
        f = v1[0] + v1[1] * r2
        t1 = (d[0] + d[1] * r2) / f
    else:
        d = p2 - p1
        t1 = d[1] / v1[1]

    return p1 + v1 * t1


def is_zero_solution(arr):
    return np.all(np.isnan(arr))


def line_ray(p1, v1, p2, v2):
    """Intersect line though p1 direction v1 with ray through p2 direction v2.
       Returns a list of zero or one solutions
    """
    bad_solution = np.zeros(3, dtype=float)
    s = line_line(p1, v1, p2, v2)
    if not is_zero_solution(s) and tol_gte(dot(s[0] - p2, v2), 0):
        return s
    else:
        return bad_solution


def ray_ray(p1, v1, p2, v2):
    """Intersect ray though p1 direction v1 with ray through p2 direction v2.
       Returns a list of zero or one solutions
    """
    bad_solution = np.zeros(3, dtype=float)
    s = line_line(p1, v1, p2, v2)
    if not is_zero_solution(s) and tol_gte(dot(s[0] - p2, v2), 0) and tol_gte(dot(s[0] - p1, v1), 0):
        return s
    else:
        return bad_solution


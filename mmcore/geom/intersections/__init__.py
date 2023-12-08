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
* Intersection of Circle and Line2Pt: Similar to the intersection of a circle with a ray,
    this function calculates intersection points between a circle and a line segment defined by two points (Line2Pt).
    It returns the line segment's intersection points with the circle using a similar discriminant approach
    as the sphere-ray intersection.

All methods are decorated with vectorize from mmcore.func which allows parallelized operations on NumPy arrays.
"""

import numpy as np
from multipledispatch import dispatch
from scipy.optimize import fsolve

from mmcore.func import vectorize
from mmcore.geom.circle import Circle
from mmcore.geom.interfaces import Line2Pt, Ray
from mmcore.geom.plane import Plane
from mmcore.geom.sphere import Sphere

INTERSECTION_TOLERANCE = 1e-6


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


@dispatch(Circle, Line2Pt)
@vectorize(excluded=[0, 2], signature='(i,j)->(k)')
def intersect(circle: Circle, line: Line2Pt, tol=INTERSECTION_TOLERANCE):
    """
    :param circle: Circle object representing a circle
    :type circle: Circle
    :param line: Line2Pt object representing a line segment
    :type line: Line2Pt
    :param tol: Tolerance for determining if points are close enough to be considered intersecting
    :type tol: float
    :return: List of intersection points between the circle and the line segment
    :rtype: list of tuples

    This method calculates the intersection points between a circle and a line segment. It takes in a Circle object representing the circle, a Line2Pt object representing the line segment
    *, and a tolerance value for determining if points are close enough to be considered intersecting. It returns a list of tuples representing the intersection points between the circle
    * and the line segment.

    Example usage:

    ```python
    circle = Circle(origin=(0, 0), r=5)
    line = Line2Pt((0, 0), (10, 10))
    tolerance = 0.01

    intersections = intersect(circle, line, tol=tolerance)
    print(intersections)
    ```
    """

    pt1, pt2 = np.array(line)[:, :2]
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

        fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in
                                  intersections]
        intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(
                discriminant) <= tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections


@dispatch(Line2Pt, Line2Pt)
@vectorize(excluded=[0, 2], signature='(i,j)->(k)')
def intersect(circle: Circle, line: Line2Pt, tol=INTERSECTION_TOLERANCE):
    """
    :param circle: The circle object
    :type circle: Circle
    :param line: The line object
    :type line: Line2Pt
    :param tol: The tolerance for determining intersection points
    :type tol: float
    :return: The list of intersection points between the circle and line
    :rtype: List[Tuple[float, float]]
    """

    pt1, pt2 = np.array(line)[:, :2]
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

        fraction_along_segment = [(xi - p1x) / dx if abs(dx) > abs(dy) else (yi - p1y) / dy for xi, yi in
                                  intersections]
        intersections = [pt for pt, frac in zip(intersections, fraction_along_segment) if 0 <= frac <= 1]
        if len(intersections) == 2 and abs(
                discriminant) <= tol:  # If line is tangent to circle, return just one point (as both intersections have same location)
            return [intersections[0]]
        else:
            return intersections

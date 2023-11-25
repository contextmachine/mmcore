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

        Parameters
        ----------
        tol :
        sphere :
        ray :

        Returns
        -------

        type: np.ndarray((n, 5), dtype=float)
        Array with shape: (n, [x, y, z, t, u, v]) where:
        1. n is the number of rays.
        2. x,y,z - intersection point cartesian coordinates.
        3. t - ray intersection param.
        4. u,v - sphere intersection params.

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
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.
    :param circle: Circle with center and radius
    :param ray: The (x, y) location of the first point of the ray,and the (x, y) direction  of the ray
    :param tol: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.
    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html
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
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.
    :param circle: Circle with center and radius
    :param line: The (x, y) location of the first point of the segment,and the (x, y) location of the second point of the segment
    :param tangent: Numerical tolerance at which we decide the intersections are close enough to consider it a tangent
    :return Sequence[Tuple[float, float]]: A list of length 0, 1, or 2, where each element is a point at which the circle intercepts a line segment.
    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html

    Parameters
    ----------
    tol :
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
    """ Find the points at which a circle intersects a line-segment.  This can happen at 0, 1, or 2 points.
    Note: We follow: http://mathworld.wolfram.com/Circle-LineIntersection.html

    Parameters
    ----------
    circle: Circle with center and radius
    line: The (x, y) location of the first point of the segment,and the (x, y) location of the second point of the segment
    tol : tolerance
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

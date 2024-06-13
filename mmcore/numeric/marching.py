import sys

import numpy as np

from mmcore.numeric.vectors import scalar_norm, scalar_cross, scalar_dot

from mmcore.numeric.algorithms import curve_point, intersection_curve_point, _implicit_tangent
from mmcore.numeric.fdm import Grad, fdm, DEFAULT_H

__all__ = ['marching_implicit_curve_points',
           'marching_intersection_curve_points',
           'raster_algorithm_2d'
           ]


# Utils


#TODO optimize it!


# Closest point algorithms for general implicits


# Marching algorithms for general implicits
def marching_implicit_curve_points(
        func, v0, v1=None, max_points=100, step=0.2, delta=0.001, grad=None
):
    """
    Calculates implicit curve points using the curve_point algorithm.

    >>> import numpy as np
    >>> def cassini(xy,a=1.1,c=1.0):
    ...     x,y=xy
    ...     return (x*x+y*y)*(x*x+y*y)-2*c*c*(x*x-y*y)-(a*a*a*a-c*c*c*c)

    >>> pointlist=np.array(marching_implicit_curve_points(cassini,
    ...                                          np.array((2,0),dtype=float),
    ...                                          np.array((2,0),dtype=float),
    ...                                          max_points=100, step=0.2, delta=0.001))
    @param func:
    @param v0:
    @param v1:
    @param max_points:
    @param step:
    @param delta:
    @param grad:
    @return:
    """
    #
    v1 = np.copy(v0) if v1 is None else v1
    grad = grad if grad is not None else Grad(func)
    grad_derivative = fdm(grad)
    pointlist = []
    start_point = curve_point(func, v0, delta, grad=grad)
    pointlist.append(start_point)

    end_point = curve_point(func, v1, delta, grad=grad)
    g = grad(start_point)

    tangent = _implicit_tangent(g, grad_derivative(start_point))
    start_point = curve_point(func, start_point + (tangent * step), delta, grad=grad)
    pointlist.append(start_point)
    distance = step
    if max_points is None:
        max_points = sys.maxsize

    while (distance > step / 2.0) and (len(pointlist) < max_points):
        g = grad(start_point)

        tangent = _implicit_tangent(g, grad_derivative(start_point))
        start_point = curve_point(
            func, start_point + (tangent * step), delta, grad=grad
        )
        pointlist.append(start_point)
        distance = np.linalg.norm(start_point - end_point)
    pointlist.append(end_point)
    return pointlist


def marching_intersection_curve_points(
        f1, f2, grad_f1, grad_f2, start_point, end_point=None, step=0.1, max_points=None, tol=1e-6, bounds=None,
        point_callback=None
):
    """
    :param f1: The first function defining the intersection surface
    :param f2: The second function defining the intersection surface
    :param grad_f1: The gradient of the first surface
    :param grad_f2: The gradient of the second surface
    :param start_point:  Starting point in the vicinity of the intersection curve
    :param end_point: The endpoint is near the intersection curve. If None, the start_point is used. (default: None)
    :param step: The step size for incrementing along the curve (default: 0.1)
    :param max_points: The maximum number of points to calculate on the curve (default: None)
    :param tol: The tolerance for determining when to stop calculating points (default: 1e-6)
    :return: An array of points that lie on the intersection curve

    """

    points = []
    use_callback = True
    if point_callback is None:
        use_callback = False
    p, f1_val, f2_val, g1, g2 = intersection_curve_point(f1, f2, start_point, grad_f1, grad_f2, tol=tol,
                                                         return_grads=True)
    points.append(p)
    if use_callback:
        point_callback(p)
    if end_point is None:
        end_point = np.copy(p)

    if max_points is None:
        max_points = sys.maxsize

    while (len(points) < max_points):
        #g1, g2 = grad_f1(p), grad_f2(p)
        if bounds is not None:

            if np.any(p < bounds[0]) or np.any(p > bounds[1]):
                break
        grad_cross = scalar_cross(g1, g2)
        if scalar_dot(grad_cross, grad_cross) == 0:
            break

        unit_tangent = grad_cross / scalar_norm(grad_cross)

        p1, _f1_val, _f2_val, _g1, _g2 = intersection_curve_point(f1, f2, p + step * unit_tangent, grad_f1, grad_f2,
                                                                  tol=tol, return_grads=True)

        # A rather specific treatment of the singularity, but it seems to work. In the literature I could find,
        # the singularity was usually defined by a zero gradient. This is not difficult to handle and here I actually
        # do it a few lines above. However, I ran into a different problem, in the neighbourhood of the singular
        # point cross(grad1,grad2), changed sign and the point would "bounce" back and then be evaluated correctly
        # and so on round and round, and the zero gradient check failed. I managed to fix this through the proximity
        # check to the point 2 steps earlier. In this case, I'm unrolling the tangent, so I've managed to get through
        # all the singularity curves I've tested so far.
        if len(points) > 1 and scalar_norm(p1 - points[len(points) - 2]) < step / 2:
            print(0, 0)
            p1, _f1_val, _f2_val, _g1, _g2 = intersection_curve_point(f1, f2, p - step * unit_tangent, grad_f1, grad_f2,
                                                                      tol=tol, return_grads=True)

        dst = scalar_norm(p1 - end_point)

        if dst < step / 2:
            break
        points.append(p1)
        p = p1
        if use_callback:
            point_callback(p)
        f1_val, f2_val, g1, g2 = _f1_val, _f2_val, _g1, _g2

    return np.array(points)


# Raster algorithms for general implicits
def raster_algorithm_2d(f, x_min, x_max, y_min, y_max, nx, ny, gamma=0.5, grad=None, general_implicit=True):
    """Raster algorithm to find points close to the implicit curve f(x, y) = 0."""
    if grad is None:
        def grad(xy):
            return (f((xy[0] + DEFAULT_H, xy[1])) - f((xy[0], - DEFAULT_H, xy[1]))) / (2 * DEFAULT_H), (
                    f((xy[0], xy[1] + DEFAULT_H)) - f((xy[0], xy[1] - DEFAULT_H))) / (2 * DEFAULT_H)

    x_vals = np.linspace(x_min, x_max, nx)
    y_vals = np.linspace(y_min, y_max, ny)
    d_max = max((x_max - x_min) / nx, (y_max - y_min) / ny)
    f_val = 0
    curve_points = []
    grad_f = np.zeros(2, dtype=float)
    xy = np.zeros(2, dtype=float)
    if general_implicit:
        curve_pt = lambda pt: curve_point(f, pt)
    else:
        curve_pt = lambda pt: pt + grad * f_val
    for i, x in enumerate(x_vals):
        for j, y in enumerate(y_vals):
            xy[:] = x, y
            f_val = f(xy)
            grad_f[:] = grad(xy)
            grad_norm = np.linalg.norm(grad_f)
            if grad_norm != 0:
                delta = abs(f_val) / grad_norm
                if delta < gamma * d_max:
                    curve_points.append(curve_pt(xy))

    return np.array(curve_points)

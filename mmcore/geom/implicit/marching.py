import math
import sys

import numpy as np

from mmcore.geom.implicit._implicit import solve2x2
from mmcore.geom.vec import make_perpendicular, unit
from mmcore.geom.vec.vec_speedups import scalar_norm, scalar_cross, scalar_dot
from mmcore.numeric.fdm import Grad, fdm, DEFAULT_H

__all__ = ['curve_point',
           'intersection_curve_point',
           'surface_point',
           'surface_plane',
           'marching_implicit_curve_points',
           'marching_intersection_curve_points',
           'raster_algorithm_2d'
           ]


# Utils
def _resolve_grad(func, grad=None):
    return grad if grad is not None else Grad(func)


def _curve_point_newton_step(func: callable, x: np.ndarray, grad: callable):
    fi = func(x)
    g = grad(x)
    cc = math.pow(g[0], 2) + math.pow(g[1], 2)
    if cc > 0:
        t = -fi / cc
    else:
        return x

    return x + t * g


def _curve_point2(func, x0, tol=1e-5, grad=None):
    grad = _resolve_grad(func, grad)
    x0 = np.copy(x0)
    delta = 1.0
    while delta >= tol:
        xi1, yi1 = _curve_point_newton_step(func, x0, grad=grad)
        delta = abs(x0[0] - xi1) + abs(x0[1] - yi1)
        x0[0] = xi1
        x0[1] = yi1
    return x0



def _normalize3d(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def _linear_combination_3d(a, v1, b, v2):
    return a * v1 + b * v2


#TODO optimize it!
def _implicit_tangent(d1, d2):
    return unit(make_perpendicular(d1, d2))


def _evaluate_jacobian(g1, g2):
    """

    :param g1: First gradient value
    :param g2: Second gradient value
    :return: Jacobian matrix for the two gradients
    """
    J = np.array([
        [np.dot(g1, g1), np.dot(g2, g1)],
        [np.dot(g1, g2), np.dot(g2, g2)]
    ])
    return J


# Closest point algorithms for general implicits

def curve_point(func, initial_point: np.ndarray = None, delta=0.001, grad=None):
    """
    Calculates the point on the curve along the "steepest path".
    @param func:
    @param grad:
    @param initial_point:
    @param delta:
    @return: point on the curve along the "steepest path"
    @type: np.ndarray


    """
    #
    grad = _resolve_grad(func, grad)
    f = func(initial_point)
    g = grad(initial_point)
    cc = sum(g * g)
    if cc > 0:
        f = -f / cc
    else:
        f = 0

    new_point = initial_point + f * g
    d = np.linalg.norm(initial_point - new_point)
    while delta < d:
        initial_point = new_point
        f = func(initial_point)
        g = grad(initial_point)
        cc = sum(g * g)
        if cc > 0:
            f = -f / cc
        else:
            f = 0
        new_point = initial_point + f * g
        d = np.linalg.norm(initial_point - new_point)
    return new_point


def intersection_curve_point(surf1, surf2, q0, grad1, grad2, tol=1e-6, max_iter=100, return_grads=False):
    """

    :param surf1:
    :param surf2:
    :param q0:
    :param grad1:
    :param grad2:
    :param tol:
    :param max_iter:
    :return:

    Example
    ---

    >>> from mmcore.geom.sphere import Sphere
    >>> c1= Sphere(np.array([0.,0.,0.]),1.)
    >>> c2= Sphere(np.array([1.,1.,1.]),1)
    >>> q0 =np.array((0.579597, 0.045057, 0.878821))
    >>> res=intersection_curve_point(c1.implicit,c2.implicit,q0,c1.normal,c2.normal,tol=1e-6) # 7 newthon iterations
    >>> print(c1.implicit(res),c2.implicit(res))
    4.679345799729617e-10 4.768321293369127e-10
    >>> res=intersection_curve_point(c1.implicit,c2.implicit,q0,c1.normal,c2.normal,tol=1e-12) # 9 newthon iterations
    >>> print(c1.implicit(res),c2.implicit(res))
    -1.1102230246251565e-16 2.220446049250313e-16
    >>> res=intersection_curve_point(c1.implicit,c2.implicit,q0,c1.normal,c2.normal,tol=1e-16) # 10 newthon iterations
    >>> print(c1.implicit(res),c2.implicit(res))
    0.0 0.0
    """
    alpha_beta = np.zeros(2, dtype=np.float64)
    qk = np.copy(q0)
    f1, f2, g1, g2 = surf1(qk), surf2(qk), grad1(qk), grad2(qk)

    J = np.array([
        [np.dot(g1, g1), np.dot(g2, g1)],
        [np.dot(g1, g2), np.dot(g2, g2)]
    ])

    g = np.array([f1, f2])
    success = solve2x2(J, -g, alpha_beta)
    delta = alpha_beta[0] * grad1(qk) + alpha_beta[1] * grad2(qk)
    qk_next = delta + qk
    d = scalar_norm(qk_next - qk)
    i = 0

    while d > tol:

        if i > max_iter:
            raise ValueError('Maximum iterations exceeded, No convergence')

        qk = qk_next
        f1, f2, g1, g2 = surf1(qk), surf2(qk), grad1(qk), grad2(qk)
        J = np.array([
            [np.dot(g1, g1), np.dot(g2, g1)],
            [np.dot(g1, g2), np.dot(g2, g2)]
        ])

        g[:] = f1, f2

        success = solve2x2(J, -g, alpha_beta)

        #alpha, beta = newton_step(qk, alpha, beta, f1, f2, g1, g2)
        #alpha, beta = newton_step(qk, alpha, beta, f1, f2, g1, g2)
        delta = alpha_beta[0] * g1 + alpha_beta[1] * g2
        qk_next = delta + qk
        d = scalar_norm(delta)

        i += 1

    if return_grads:
        return qk_next, f1, f2, g1, g2
    return qk_next


def surface_point(fun, p0, grad=None, tol=1e-8):
    p_i = p0
    grad = _resolve_grad(fun, grad)
    while True:
        fi, gradfi = (fun(p_i), grad(p_i))
        cc = scalar_dot(gradfi, gradfi)
        if cc > 0:
            t = -fi / cc
        else:
            t = 0
            print(f"{cc} WARNING tri (surface_point...): newton")

        p_i1 = _linear_combination_3d(1, p_i, t, gradfi)
        dv = p_i1 - p_i
        delta = scalar_norm(dv)

        if delta < tol:
            break

        p_i = p_i1

    #fi, gradfi = fun(p_i), grad(p_i)
    return p_i1


def surface_plane(fun, p_start, grad, tol=1e-5):
    p, nv = surface_point(fun, p_start, grad, tol)
    nv = _normalize3d(nv)

    if abs(nv[0]) > 0.5 or abs(nv[1]) > 0.5:
        tv1 = np.array([nv[1], -nv[0], 0])
    else:
        tv1 = np.array([-nv[2], 0, nv[0]])

    tv1 = _normalize3d(tv1)
    tv2 = scalar_cross(nv, tv1)
    return p, nv, tv1, tv2


# Marching algorithms for general implicits
def marching_implicit_curve_points(
        func, v0, v1=None, max_points=100, step=0.2, delta=0.001, grad=None
):
    """
    Calculates implicit curve points using the curve_point algorithm.
    >>> from mmcore.geom.implicit.marching import marching_implicit_curve_points
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
        f1, f2, grad_f1, grad_f2, start_point, end_point=None, step=0.1, max_points=None, tol=1e-6, point_callback=None
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
        grad_cross = scalar_cross(g1, g2)
        if np.dot(grad_cross, grad_cross) == 0:
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

import math

import numpy as np

from mmcore.numeric.vectors import solve2x2, scalar_norm

from mmcore.geom.vec import unit, make_perpendicular
from mmcore.numeric import scalar_dot, scalar_cross


from mmcore.numeric.fdm import Grad


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


def intersection_curve_point(surf1, surf2, q0, grad1, grad2, tol=1e-6, max_iter=100, return_grads=False, no_err=False):
    """
        Intersection curve point between two curves.

        :param surf1: The first curve surface function.
        :type surf1: function
        :param surf2: The second curve surface function.
        :type surf2: function
        :param q0: Initial point on the curve.
        :type q0: numpy.ndarray
        :param grad1: Gradient function of the first curve surface.
        :type grad1: function
        :param grad2: Gradient function of the second curve surface.
        :type grad2: function
        :param tol: Tolerance for convergence. Default is 1e-6.
        :type tol: float
        :param max_iter: Maximum number of iterations allowed. Default is 100.
        :type max_iter: int
        :param return_grads: Flag to indicate whether to return gradients. Default is False.
        :type return_grads: bool
        :param no_err: Flag to indicate whether to raise error or return failure status in case of no convergence. Default is False.
        :type no_err: bool
        :return: The intersection point on the curve surface.
        :rtype: numpy.ndarray or tuple

        Example
    ---

    >>> from mmcore.geom.primitives import Sphere
    >>> c1= Sphere(np.array([0.,0.,0.]),1.)
    >>> c2= Sphere(np.array([1.,1.,1.]),1)
    >>> q0 =np.array((0.579597, 0.045057, 0.878821))
    >>> res=intersection_curve_point(c1.implicit,c2.implicit,q0,c1.gradient,c2.gradient,tol=1e-6) # 7 newthon iterations
    >>> print(c1.implicit(res),c2.implicit(res))
    4.679345799729617e-10 4.768321293369127e-10
    >>> res=intersection_curve_point(c1.implicit,c2.implicit,q0,c1.gradient,c2.gradient,tol=1e-12) # 9 newthon iterations
    >>> print(c1.implicit(res),c2.implicit(res))
    -1.1102230246251565e-16 2.220446049250313e-16
    >>> res=intersection_curve_point(c1.implicit,c2.implicit,q0,c1.gradient,c2.gradient,tol=1e-16) # 10 newthon iterations
    >>> print(c1.implicit(res),c2.implicit(res))
    0.0 0.0
    """
    alpha_beta = np.zeros(2, dtype=np.float64)
    qk = np.copy(q0)

    f1, f2, g1, g2 = surf1(qk), surf2(qk), grad1(qk), grad2(qk)

    J = np.array([
        [scalar_dot(g1, g1), scalar_dot(g2, g1)],
        [scalar_dot(g1, g2), scalar_dot(g2, g2)]
    ])

    g = np.array([f1, f2])
    success = solve2x2(J, -g, alpha_beta)
    delta = alpha_beta[0] * grad1(qk) + alpha_beta[1] * grad2(qk)
    qk_next = delta + qk
    d = scalar_norm(qk_next - qk)
    i = 0

    success = True
    while d > tol:

        if i > max_iter:
            if not no_err:
                raise ValueError(f'Maximum iterations exceeded, No convergence {d}')
            else:
                success = False
                break

        qk = qk_next
        f1, f2, g1, g2 = surf1(qk), surf2(qk), grad1(qk), grad2(qk)
        J = np.array([
            [scalar_dot(g1, g1), scalar_dot(g2, g1)],
            [scalar_dot(g1, g2), scalar_dot(g2, g2)]
        ])

        g[:] = f1, f2

        _success = solve2x2(J, -g, alpha_beta)

        #alpha, beta = newton_step(qk, alpha, beta, f1, f2, g1, g2)
        #alpha, beta = newton_step(qk, alpha, beta, f1, f2, g1, g2)
        delta = alpha_beta[0] * g1 + alpha_beta[1] * g2
        qk_next = delta + qk

        d = scalar_norm(delta)

        i += 1

    if return_grads:
        return (success, (qk_next, f1, f2, g1, g2)) if no_err else qk_next, f1, f2, g1, g2

    return (success, qk_next) if no_err else qk_next


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
    norm = scalar_norm(v)
    if norm == 0:
        return v
    return v / norm


def _linear_combination_3d(a, v1, b, v2):
    return a * v1 + b * v2


def _implicit_tangent(d1, d2):
    return unit(make_perpendicular(d1, d2))


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

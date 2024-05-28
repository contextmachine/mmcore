import math
import sys

import numpy as np

from mmcore.geom.vec import make_perpendicular, unit
from mmcore.numeric.fdm import Grad, fdm


def _resolve_grad(func, grad=None):
    return grad if grad is not None else Grad(func)


def newton_step(func: callable, x: np.ndarray, grad: callable):
    fi = func(x)
    g = grad(x)
    cc = math.pow(g[0], 2) + math.pow(g[1], 2)
    if cc > 0:
        t = -fi / cc
    else:
        return x

    return x + t * g


def curve_point2(func, x0, tol=1e-5, grad=None):
    grad = _resolve_grad(func, grad)
    x0 = np.copy(x0)
    delta = 1.0
    while delta >= tol:
        xi1, yi1 = newton_step(func, x0, grad=grad)
        delta = abs(x0[0] - xi1) + abs(x0[1] - yi1)
        x0[0] = xi1
        x0[1] = yi1
    return x0


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


def implicit_tangent(d1, d2):
    return unit(make_perpendicular(d1, d2))


def implicit_curve_points(
        func, v0, v1=None, max_points=100, step=0.2, delta=0.001, grad=None
):
    """
    Calculates implicit curve points using the curve_point algorithm.
    >>> from mmcore.geom.implicit.marching import implicit_curve_points
    >>> import numpy as np
    >>> def cassini(xy,a=1.1,c=1.0):
    ...     x,y=xy
    ...     return (x*x+y*y)*(x*x+y*y)-2*c*c*(x*x-y*y)-(a*a*a*a-c*c*c*c)

    >>> pointlist=np.array(implicit_curve_points(cassini,
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

    tangent = implicit_tangent(g, grad_derivative(start_point))
    start_point = curve_point(func, start_point + (tangent * step), delta, grad=grad)
    pointlist.append(start_point)
    distance = step
    if max_points is None:
        max_points = sys.maxsize

    while (distance > step / 2.0) and (len(pointlist) < max_points):
        g = grad(start_point)

        tangent = implicit_tangent(g, grad_derivative(start_point))
        start_point = curve_point(
            func, start_point + (tangent * step), delta, grad=grad
        )
        pointlist.append(start_point)
        distance = np.linalg.norm(start_point - end_point)
    pointlist.append(end_point)
    return pointlist




def evaluate_jacobian(alpha, beta, qk, point, grad1, grad2):
    g1 = grad1(qk)
    g2 = grad2(qk)
    #gn1 = grad1(point)
    #gn2 = grad2(point)
    J = np.array([
        [np.dot(g1, g1), np.dot(g2, g1)],
        [np.dot(g1, g2), np.dot(g2, g2)]
    ])
    return J


def evaluate_g(alpha, beta, qk, point, surf1, surf2):
    g = np.array([surf1(point), surf2(point)])

    return g




def intersection_curve_point(surf1, surf2, q0, grad1, grad2, tol=1e-6, max_iter=100):
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
    def newton_step( q, alpha_init, beta_init):
        q=np.copy(q)

        #point = q + alpha_init * grad1(q) + beta_init * grad2(q)
        J = evaluate_jacobian(alpha_init, beta_init,  q, q, grad1, grad2)
        g = evaluate_g(alpha_init, beta_init,  q, q, surf1, surf2)
        alpha_init,beta_init = np.linalg.solve(J, -g)
        return alpha_init,beta_init











    alpha, beta = 0.,0.
    qk = np.copy(q0)

    alpha, beta = newton_step( qk, alpha, beta)
    delta = alpha * grad1(qk) + beta * grad2(qk)
    qk_next = delta + qk
    d = np.linalg.norm(qk_next - qk)
    i=0
    while d > tol :
        if i>max_iter:
            raise ValueError('Maximum iterations exceeded, No convergence')
        qk = qk_next


        alpha, beta = newton_step( qk, alpha, beta)
        delta = alpha * grad1(qk) + beta * grad2(qk)
        qk_next = delta + qk
        d = np.linalg.norm(delta)

        i+=1
    return qk_next






def _proc_val(f,g):
    cc = sum(g * g)
    if cc > 0:
        f=-f / cc
    else:
        f=0.
    return f

def normalize3d(v):
    norm = np.linalg.norm(v)
    if norm == 0:
        return v
    return v / norm


def scalarp3d(v1, v2):
    return np.dot(v1, v2)


def lcomb2vt3d(a, v1, b, v2):
    return a * v1 + b * v2


def diff3d(v1, v2):
    return v1 - v2


def abs3d(v):
    return np.linalg.norm(v)


def vectorp(v1, v2):
    return np.cross(v1, v2)


def put3d(x, y, z):
    return np.array([x, y, z])


def surface_point_normal_tangentvts(fun,p_start, grad, tol=1e-5):
    p, nv = surface_point(fun, p_start, grad,tol)
    nv = normalize3d(nv)

    if abs(nv[0]) > 0.5 or abs(nv[1]) > 0.5:
        tv1 = put3d(nv[1], -nv[0], 0)
    else:
        tv1 = put3d(-nv[2], 0, nv[0])

    tv1 = normalize3d(tv1)
    tv2 = vectorp(nv, tv1)
    return p, nv, tv1, tv2


def surface_point(fun,p0, grad, tol=1e-8):


    p_i = p0

    while True:
        fi, gradfi = (fun(p_i), grad(p_i))
        cc = scalarp3d(gradfi, gradfi)
        if cc > 1e-15:
            t = -fi / cc
        else:
            t = 0
            print(f"{cc} WARNING tri (surface_point...): newton")

        p_i1 = lcomb2vt3d(1, p_i, t, gradfi)
        dv = diff3d(p_i1, p_i)
        delta = abs3d(dv)

        if delta < tol:
            break

        p_i = p_i1

    fi, gradfi = fun(p_i),grad(p_i)
    return p_i1, gradfi






def marching_intersection_curve_points(
        f1, f2, grad_f1, grad_f2, start_point, end_point=None, step=0.1, max_points=None, tol=1e-6
):
    points = []

    p = intersection_curve_point(f1, f2, start_point, grad_f1, grad_f2, tol=tol)
    if end_point is None:
        end_point = np.copy(p)
    points.append(p)
    if max_points is None:
        max_points = sys.maxsize

    while (len(points) < max_points):

        grad_cross = np.cross(grad_f1(p), grad_f2(p))
        if np.linalg.norm(grad_cross) == 0:
            break
        unit_tangent = grad_cross / np.linalg.norm(grad_cross)

        p = intersection_curve_point(f1, f2, p + step * unit_tangent, grad_f1, grad_f2, tol=tol)
        points.append(p)
        dst = np.linalg.norm(p - end_point)

        if dst < step / 2:
            break

    points.append(end_point)
    return np.array(points)
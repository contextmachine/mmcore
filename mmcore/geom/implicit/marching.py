import math

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


def curve_point2(func, x0,  tol=1e-5,grad=None):
    grad=_resolve_grad(func, grad)
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

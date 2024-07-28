
import os

import numpy as np

from mmcore.numeric.vectors import vector_projection, scalar_dot, scalar_norm

from mmcore.geom.bvh import contains_point
from mmcore.geom.surfaces import Surface
from mmcore.numeric import divide_interval
from mmcore.numeric.fdm import PDE, newtons_method
from mmcore.numeric.divide_and_conquer import iterative_divide_and_conquer_min, divide_and_conquer_min_2d

from scipy.optimize import newton
import multiprocessing as mp

from mmcore.numeric.fdm import bounded_fdm


def foot_point(S, P, s0, t0, partial_derivatives=None, epsilon=1e-6, alpha_max=20):
    """
    Find the foot point on the parametric surface S(s, t) closest to the given point P.
    """
    if partial_derivatives is None:
        _pde = PDE(S, dim=2)
        partial_derivatives = lambda uv: _pde(uv).T
    s, t = st = np.array([s0, t0])

    while True:
        p_i = S(st)
        e_s, e_t = partial_derivatives(st)
        # Solve the linear system for Δs and Δt
        A = np.array([
            [scalar_dot(e_s, e_s), scalar_dot(e_s, e_t)],
            [scalar_dot(e_s, e_t), scalar_dot(e_t, e_t)]
        ])
        b = np.array([
            scalar_dot(P - p_i, e_s),
            scalar_dot(P - p_i, e_t)
        ])
        delta = np.linalg.solve(A, b)
        delta_s, delta_t = delta
        q_i = p_i + delta_s * e_s + delta_t * e_t
        s_new = s + delta_s
        t_new = t + delta_t
        p_new = S(s_new, t_new)
        f1 = q_i - p_i
        f2 = p_new - q_i
        # Check convergence
        if np.linalg.norm(q_i - p_i) < epsilon:
            break
        # Newton step for the foot point on the tangent parabola
        a0 = scalar_dot(P - p_i, f1)
        a1 = 2 * scalar_dot(f2, P - p_i) - scalar_dot(f1, f1)
        a2 = -3 * scalar_dot(f1, f2)
        a3 = -2 * scalar_dot(f2, f2)
        alpha = 1 - (a0 + a1 + a2 + a3) / (a1 + 2 * a2 + 3 * a3)
        alpha = np.clip(alpha, 0, alpha_max)
        s = s + alpha * delta_s
        t = t + alpha * delta_t
        st[0] = s
        st[1] = t
    return S(s, t), s, t


def closest_point_on_curve_single(curve, point, tol=1e-3):
    """

    :param curve: The curve on which to find the closest point.
    :param point: The point for which to find the closest point on the curve.
    :param tol: The tolerance for the minimum finding algorithm. Defaults to 1e-5.
    :return: The closest point on the curve to the given point, distance.

    """
    _fn = getattr(curve, "evaluate", curve)

    def distance_func(t):
        return scalar_norm(point - _fn(t))

    t0, t1 = curve.interval()

    t_best, d_best = t0, distance_func(t0)
    t, d = t1, distance_func(t1)
    if d < d_best:
        t_best = t
        d_best = d

    for bnds in divide_interval(*curve.interval(), step=0.5):
        # t,d=find_best(distance_func, bnds, tol=tol)
        t, d = iterative_divide_and_conquer_min(distance_func, bnds, tol=tol)
        if d < d_best:
            t_best = t
            d_best = d

    return t_best, d_best


class _ClosestPointSolution:
    def __init__(self, curve, tol=1e-5):
        self.curve = curve
        self.tol = tol

    def __call__(self, point):
        return closest_point_on_curve_single(self.curve, point, tol=self.tol)


def closest_points_on_curve_mp(curve, points, tol=1e-3, workers=1):
    if workers == -1:
        workers = os.cpu_count()
    with mp.Pool(workers) as pool:
        solution = _ClosestPointSolution(curve, tol=tol)
        return list(pool.map(solution, points
                             ))


def closest_point_on_curve(curve, pts, tol=1e-3, workers=1):
    pts = pts if isinstance(pts, np.ndarray) else np.array(pts)

    if pts.ndim == 1:
        return closest_point_on_curve_single(curve, pts, tol=tol)

    if workers == 1:
        return [closest_point_on_curve_single(curve, pt, tol=tol) for pt in pts]
    else:
        return closest_points_on_curve_mp(curve, pts, tol=tol, workers=workers)


def local_closest_point_on_curve(curve, t0, point, tol=1e-3, **kwargs):
    def fun(t):
        # C' (u) •(C(u) - P)
        return scalar_dot(curve.derivative(t), curve.evaluate(t) - point)

    dfun = bounded_fdm(fun, curve.interval())
    res = newton(fun, t0, fprime=dfun, tol=tol, **kwargs)
    return res, np.linalg.norm(curve.evaluate(res) - point)



def closest_point_on_ray(ray, point):
    start, direction = ray

    return start + vector_projection(point - start, direction)

def closest_point_on_line(line, point):
    start, end = line
    direction = end - start
    return start + vector_projection(point - start, direction)


def closest_point_on_surface(self:Surface, pt, tol=1e-3,bounds=None):

    if bounds is None:
        bounds = tuple(self.interval())
    (umin, umax), (vmin, vmax) = bounds
    def wrp1(uv):
        d = self.evaluate(uv) - pt
        return scalar_dot(d, d)

    def wrp(u, v):
        d = self.evaluate(np.array([u, v])) - pt
        return scalar_dot(d, d)

    cpt = contains_point(self.tree, pt)

    if len(cpt) == 0:
        #(umin, umax), (vmin, vmax) = self.interval()
        return divide_and_conquer_min_2d(wrp, (umin, umax), (vmin, vmax), tol)

    else:

        initial = np.average(min(cpt, key=lambda x: x.bounding_box.volume()).uvs, axis=0)
        uv=newtons_method(wrp1, initial,tol=tol)
        if uv is None:
            raise ValueError('Newtons method failed to converge')
        return uv

__all__ = ["closest_point_on_curve",

           "closest_point_on_line",
           "foot_point",
           "closest_point_on_curve_single",
           "closest_points_on_curve_mp",
           "closest_points_on_curve_mp",
           "local_closest_point_on_curve"
           ]

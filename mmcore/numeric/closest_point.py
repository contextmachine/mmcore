import math
import os

import numpy as np

from mmcore.geom.vec import unit, dot, norm
from mmcore.numeric import divide_interval, add_dim
from mmcore.numeric.divide_and_conquer import (
    recursive_divide_and_conquer_min,
    iterative_divide_and_conquer_min,
)
from scipy.optimize import newton
import multiprocess as mp

from mmcore.numeric.fdm import bounded_fdm


def closest_point_on_curve_single(curve, point, tol=1e-3):
    """

    :param curve: The curve on which to find the closest point.
    :param point: The point for which to find the closest point on the curve.
    :param tol: The tolerance for the minimum finding algorithm. Defaults to 1e-5.
    :param workers: Workers count
    :return: The closest point on the curve to the given point, distance.

    """
    _fn = getattr(curve, "evaluate", curve)

    def distance_func(t):
        return np.linalg.norm(point - _fn(t))

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


def closest_points_on_curve_mp(curve, points, tol=1e-3, workers=1):
    if workers == -1:
        workers = os.cpu_count()
    with mp.Pool(workers) as pool:
        return list(pool.map(
            lambda pt: closest_point_on_curve_single(curve, pt, tol=tol), points
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
        return np.dot(curve.derivative(t), curve.evaluate(t) - point)

    dfun = bounded_fdm(fun, curve.interval())
    res = newton(fun, t0, fprime=dfun, tol=tol, **kwargs)
    return res, np.linalg.norm(curve.evaluate(res) - point)


def vector_projection(a,b):
	ua,ub=unit(a),unit(b)

	return  ub*dot(ua,ub)*norm(a)


def closest_point_on_line(line, point):
	start,end=line
	direction=end-start
	return  start+vector_projection(point-start,direction)

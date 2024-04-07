import math

import numpy as np

from mmcore.numeric import divide_interval
from mmcore.numeric.divide_and_conquer import recursive_divide_and_conquer_min, iterative_divide_and_conquer_min


def closest_point_on_curve(curve, point, tol=1e-3):
    """
    :param curve: The curve on which to find the closest point.
    :param point: The point for which to find the closest point on the curve.
    :param tol: The tolerance for the minimum finding algorithm. Defaults to 1e-5.
    :return: The closest point on the curve to the given point, distance.

    """
    _fn=getattr(curve,"evaluate",curve)
    def distance_func(t):

        return np.linalg.norm(point - _fn(t))

    t0,t1=curve.interval()

    t_best, d_best = t0, distance_func(t0)
    t, d =t1,distance_func(t1)
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
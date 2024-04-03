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

    def distance_func(t):
        return np.linalg.norm(point - curve.evaluate(t))

    t_best = None
    d_best= np.inf

    for bnds in divide_interval(*curve.interval(), step=0.5):
        # t,d=find_best(distance_func, bnds, tol=tol)
        t, d = iterative_divide_and_conquer_min(distance_func, bnds, tol=tol)
        if d < d_best:
            t_best = t
            d_best = d

    return t_best, d_best
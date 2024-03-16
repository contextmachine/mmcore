import math

import numpy as np

from mmcore.numeric import divide_interval
from mmcore.numeric.divide_and_conquer import recursive_divide_and_conquer_min


def closest_point_on_curve(curve, point: np.ndarray[3, np.dtype[float]], tol=1e-5):
    """
    :param curve: The curve on which to find the closest point.
    :param point: The point for which to find the closest point on the curve.
    :param tol: The tolerance for the minimum finding algorithm. Defaults to 1e-5.
    :return: The closest point on the curve to the given point, distance.

    """

    def gen():
        s, e = curve.interval()
        for start, end in divide_interval(s, e, step=1.0):
            x, fval = recursive_divide_and_conquer_min(lambda u: sum((curve(u) - point) ** 2),
                                                       (start, end),
                                                       tol)

            yield x, math.sqrt(fval)

    return sorted(gen(), key=lambda x: x[1])[0]

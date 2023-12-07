from collections import namedtuple

import numpy as np

from mmcore.geom.line import evaluate_line
from mmcore.geom.vec import dist, unit

ClosestPointSolution1D = namedtuple('ClosestPointSolution1D', ['point', 'distance', 'bounded', 't'])
ClosestPointSolution2D = namedtuple('ClosestPointSolution2D', ['point', 'distance', 'bounded', 'u', 'v'])
ClosestPointSolution3D = namedtuple('ClosestPointSolution3D', ['point', 'distance', 'bounded', 'u', 'v', 'w'])


def closest_parameter(start, end, pt) -> float:
    """
    Calculate the closest point on a line segment to a given point.

    :param start: The starting point of the line segment.
    :type start: tuple(float)
    :param end: The ending point of the line segment.
    :type end: tuple(float)
    :param pt: The point to which the closest point needs to be calculated.
    :type pt: tuple(float)
    :return: The closest parameter (t) on the line segment to the given point.
    :rtype: float
    """
    line = start, end
    vec = np.array(pt) - line[0]

    return np.dot(unit(line[1] - line[0]), vec / dist(line[0], line[1]))


closest_parameter = np.vectorize(closest_parameter, signature='(i),(i),(i)->()', doc=closest_parameter.__doc__)


def closest_point(starts, ends, pts):
    """
    Finds the closest point on a line segment to a given set of points.

    :param starts: The starting points of the line segments.
    :type starts: list of tuples
    :param ends: The ending points of the line segments.
    :type ends: list of tuples
    :param pts: The points for which to find the closest points on the line segments.
    :type pts: list of tuples
    :return: The closest points on the line segments to the given points.
    :rtype: list of tuples
    """
    return evaluate_line(starts, ends, closest_parameter(starts, ends, pts))

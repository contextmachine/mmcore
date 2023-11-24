import numpy as np

from mmcore.geom.line import evaluate_line
from mmcore.geom.vec import dist, unit


def closest_parameter(start, end, pt):
    line = start, end
    vec = np.array(pt) - line[0]

    return np.dot(unit(line[1] - line[0]), vec / dist(line[0], line[1]))


closest_parameter = np.vectorize(closest_parameter, signature='(i),(i),(i)->()')


def closest_point(starts, ends, pts):
    return evaluate_line(starts, ends, closest_parameter(starts, ends, pts))

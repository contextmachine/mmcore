import numpy as np


def sd_rect(origin, bounds, p):
    width, height = bounds
    b = np.array([width / 2.0, height / 2.0])
    # Adjust the point p relative to the origin
    p_adjusted = p - (origin + b)

    q = np.abs(p_adjusted) - b
    return min(max(q[0], q[1]), 0.0) + np.linalg.norm(np.maximum(q, 0.0))

from mmcore.numeric.marching import marching_implicit_curve_points

points=np.array(marching_implicit_curve_points(lambda x: sd_rect((0., 0.0),(1., 1.0), x), np.array([1.4,1.1])))
print(points)

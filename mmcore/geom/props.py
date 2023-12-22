from mmcore.geom.tolerance import *
from mmcore.geom.vec import *
import numpy as np


def angle_3p(p1, p2, p3):
    """Returns the angle, in radians, rotating np.array p2p1 to np.array p2p3.
       arg keywords:
          p1 - a np.array
          p2 - a np.array
          p3 - a np.array
       returns: a number
       In 2D, the angle is a signed angle, range [-pi,pi], corresponding
       to a clockwise rotation. If p1-p2-p3 is clockwise, then angle > 0.
       In 3D, the angle is unsigned, range [0,pi]
    """
    d21 = norm(p2 - p1)
    d23 = norm(p3 - p2)
    if tol_eq(d21, 0) or tol_eq(d23, 0):
        return None  # degenerate angle
    v21 = (p1 - p2) / d21
    v23 = (p3 - p2) / d23
    t = dot(v21, v23)  # / (d21 * d23)
    if t > 1.0:  # check for floating point error
        t = 1.0
    elif t < -1.0:
        t = -1.0
    angle = np.arccos(t)
    if len(p1) == 2:  # 2D case
        if is_counterclockwise(p1, p2, p3):
            angle = -angle
    return angle





def is_counterclockwise(p1, p2, p3):
    """ returns True iff triangle p1,p2,p3 is counterclockwise oriented"""
    u = p2 - p1
    v = p3 - p2
    perp_u = np.array([-u[1], u[0]])
    return tol_gt(dot(perp_u, v), 0)


def is_flat(p1, p2, p3):
    """ returns True iff triangle p1,p2,p3 is flat (neither clockwise of counterclockwise oriented)"""
    u = p2 - p1
    v = p3 - p2
    perp_u = np.array([-u[1], u[0]])
    return tol_eq(dot(perp_u, v), 0)


def is_acute(p1, p2, p3):
    """returns True iff angle p1,p2,p3 is acute, i.e. less than pi/2"""
    angle = angle_3p(p1, p2, p3)
    if angle != None:
        return tol_lt(abs(angle), np.pi / 2)
    else:
        return False


def is_obtuse(p1, p2, p3):
    """returns True iff angle p1,p2,p3 is obtuse, i.e. greater than pi/2"""
    angle = angle_3p(p1, p2, p3)
    if angle != None:
        return tol_gt(abs(angle), np.pi / 2)
    else:
        return False


def is_left_handed(p1, p2, p3, p4):
    """return True if tetrahedron p1 p2 p3 p4 is left handed"""
    u = p2 - p1
    v = p3 - p1
    uv = cross(u, v)
    w = p4 - p1
    return dot(uv, w) < 0


def is_right_handed(p1, p2, p3, p4):
    """return True if tetrahedron p1 p2 p3 p4 is right handed"""
    u = p2 - p1
    v = p3 - p1
    uv = cross(u, v)
    w = p4 - p1
    return dot(uv, w) > 0


def is_clockwise(polygon: list) -> bool:
    """
    Determine if a polygon is clockwise or counterclockwise.
    :param polygon: A list of points representing the polygon.
    :type polygon: list
    :return: True if the polygon is clockwise, False otherwise.
    :rtype: bool
    """
    sum_ = 0
    for i in range(len(polygon) - 1):
        cur = polygon[i]
        next_ = polygon[i + 1]
        sum_ += (next_[0] - cur[0]) * (next_[1] + cur[1])
    return sum_ < 0

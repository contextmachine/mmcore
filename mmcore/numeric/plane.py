from __future__ import annotations

from math import sqrt

import numpy as np

from mmcore.func import vectorize
from mmcore.geom.vec import dot, norm_sq, unit, cross


def inverse_evaluate_plane(pln, point):
    return dot(point - pln[0], pln[1:])


def evaluate_plane(pln, point):
    return (
            pln[0]
            + pln[1] * point[0]
            + pln[2] * point[1]
            + pln[3] * point[2]

    )


def distance(pln, point2):
    a, b, c = pln[-1]
    x0, y0, z0 = pln[0]
    x, y, z = point2
    return abs(
        a * (x - x0) / sqrt(a ** 2 + b ** 2 + c ** 2)
        + b * (y - y0) / sqrt(a ** 2 + b ** 2 + c ** 2)
        + c * (z - z0) / sqrt(a ** 2 + b ** 2 + c ** 2)
    )


def arbitrary(t, normal, origin):
    a, b, c = normal
    x0, y0, z0 = origin

    return np.array(
        (
            (
                -a
                * c
                * np.sin(t)
                / np.sqrt(a ** 2 * c ** 2 + b ** 2 * c ** 2 + (a ** 2 + b ** 2) ** 2)
                - b * np.cos(t) / np.sqrt(a ** 2 + b ** 2)
                + x0,
                a * np.cos(t) / np.sqrt(a ** 2 + b ** 2)
                - b
                * c
                * np.sin(t)
                / np.sqrt(a ** 2 * c ** 2 + b ** 2 * c ** 2 + (a ** 2 + b ** 2) ** 2)
                + y0,
                z0
                + (a ** 2 + b ** 2)
                * np.sin(t)
                / np.sqrt(a ** 2 * c ** 2 + b ** 2 * c ** 2 + (a ** 2 + b ** 2) ** 2),
            )
        )
    )


def plane_from_normal2(origin, normal, tol=1e-5):
    if norm_sq(normal) <= tol:
        normal = np.array([0, 0, 1], dtype=float)
    else:
        normal = unit(normal)
    xaxis = unit(perp_to_vector(normal))
    yaxis = unit(_mycross(xaxis, normal))
    return np.array([origin, xaxis, yaxis, normal])


def project_point_by_normal(point, normal, origin):
    v = point - origin
    dst = v[0] * normal[0] + v[1] * normal[1] + v[2] * normal[2]
    return point - dst * normal


def plane_eq_from_pts(p1, p2, p3):
    return [
        p1[1] * p2[2]
        - p1[1] * p3[2]
        - p1[2] * p2[1]
        + p1[2] * p3[1]
        + p2[1] * p3[2]
        - p2[2] * p3[1],
        -p1[0] * p2[2]
        + p1[0] * p3[2]
        + p1[2] * p2[0]
        - p1[2] * p3[0]
        - p2[0] * p3[2]
        + p2[2] * p3[0],
        p1[0] * p2[1]
        - p1[0] * p3[1]
        - p1[1] * p2[0]
        + p1[1] * p3[0]
        + p2[0] * p3[1]
        - p2[1] * p3[0],
        -p1[0] * p2[1] * p3[2]
        + p1[0] * p2[2] * p3[1]
        + p1[1] * p2[0] * p3[2]
        - p1[1] * p2[2] * p3[0]
        - p1[2] * p2[0] * p3[1]
        + p1[2] * p2[1] * p3[0],
    ]


def plane_from_3pt(pt0, pt1, pt2):
    pt0, pt1, pt2 = np.array(pt0), np.array(pt1), np.array(pt2)
    a, b, c, d = plane_eq_from_pts(pt0, pt1, pt2)
    # Normal vector to the plane
    n = np.array([a, b, c])
    N = unit(n)
    # Choose an arbitrary point on the plane
    xax = unit(pt1 - pt0)
    yax = cross(N, xax)
    # Generate a vector on the plane
    return np.array([pt0, xax, yax, N]), [a, b, c, d]


def pln_eq_3pt(p0, p1, p2):
    matrix_a = np.array([[p0[1], p0[2], 1], [p1[1], p1[2], 1], [p2[1], p2[2], 1]])
    matrix_b = np.array([[-p0[0], p0[2], 1], [-p1[0], p1[2], 1], [-p2[0], p2[2], 1]])
    matrix_c = np.array([[p0[0], p0[1], 1], [p1[0], p1[1], 1], [p2[0], p2[1], 1]])
    matrix_d = np.array(
        [[-p0[0], -p0[1], p0[2]], [-p1[0], -p1[1], p1[2]], [-p2[0], -p2[1], p2[2]]]
    )
    det_a = np.linalg.det(matrix_a)
    det_b = np.linalg.det(matrix_b)
    det_c = np.linalg.det(matrix_c)
    det_d = -np.linalg.det(matrix_d)
    return np.array([det_a, det_b, det_c, det_d])


@vectorize(excluded=[1], signature="(i)->(i)")
def perp_to_vector(v, tol=1e-5):
    if norm_sq(v) >= tol:
        if (abs(v[0]) >= tol) or (abs(v[1]) >= tol):
            return _mycross(v, [0, 0, 1])
        else:
            return _mycross(v, [1, 0, 0])
    else:
        return np.array([0.0, 0.0, 0.0])


def _mycross(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return np.array([ay * bz - by * az, az * bx - bz * ax, ax * by - bx * ay])

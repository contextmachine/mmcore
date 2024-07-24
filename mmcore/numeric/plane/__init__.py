from __future__ import annotations

from math import sqrt
from mmcore.numeric.plane.cplane import plane_plane_intersection, plane_plane_intersect, evaluate_plane, \
    inverse_evaluate_plane, plane_plane_plane_intersect,inverse_evaluate_plane_arr,evaluate_plane_arr
import numpy as np

from mmcore.func import vectorize
from mmcore.geom.vec import dot, norm_sq, unit, cross, norm
from mmcore.numeric.vectors import scalar_dot, dot_vec_x_array, scalar_norm, scalar_cross, scalar_unit, \
    vector_projection

WORLD_XY = np.array([[0., 0., 0.],
                     [1., 0., 0.],
                     [0., 1., 0.],
                     [0., 0., 1.]])


#def inverse_evaluate_plane(pln, point):
#    return dot_vec_x_array(point - pln[0], pln[1:])
#
#
#def evaluate_plane(pln, point):
#    return (
#            pln[0]
#            + pln[1] * point[0]
#            + pln[2] * point[1]
#            + pln[3] * point[2]
#
#    )


def distance(pln, point2):
    a, b, c = pln[-1]
    x0, y0, z0 = pln[0]
    x, y, z = point2
    n = sqrt(a ** 2 + b ** 2 + c ** 2)
    return abs(
        a * (x - x0) / n
        + b * (y - y0) / n
        + c * (z - z0) / n
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


def plane_from_3pt(pt0, pt1, pt2):
    pt0, pt1, pt2 = np.array(pt0), np.array(pt1), np.array(pt2)
    a, b, c, d = plane_eq_from_pts(pt0, pt1, pt2)
    # Normal vector to the plane
    n = np.array([a, b, c])
    N = scalar_unit(n)
    # Choose an arbitrary point on the plane
    xax = scalar_unit(pt1 - pt0)
    yax = scalar_cross(N, xax)
    # Generate a vector on the plane
    return np.array([pt0, xax, yax, N]), [a, b, c, d]


def plane_eq_from_pts(pt0, pt1, pt2):
    """
    x3=y0*z1
x4=y1*z2
x5=y2*z0
v0=x3 + x4 + x5 - y0*z2 - y1*z0 - y2*z1
v1=-x0*z1 + x0*z2 + x1*z0 - x1*z2 - x2*z0 + x2*z1
v2=x0*y1 - x0*y2 - x1*y0 + x1*y2 + x2*y0 - x2*y1
v3=-x0*x4 + x0*y2*z1 - x1*x5 + x1*y0*z2 - x2*x3 + x2*y1*z0
    :param pt0:
    :param pt1:
    :param pt2:
    :return:
    """
    x3 = pt0[1] * pt1[2]
    x4 = pt1[1] * pt2[2]
    x5 = pt2[1] * pt0[2]
    return np.array(
        [x3 + x4 + x5 - pt0[1] * pt2[2] - pt1[1] * pt0[2] - pt2[1] * pt1[2],
         -pt0[0] * pt1[2] + pt0[0] * pt2[2] + pt1[0] * pt0[2] - pt1[0] * pt2[2] - pt2[0] * pt0[2] + pt2[0] * pt1[2],
         pt0[0] * pt1[1] - pt0[0] * pt2[1] - pt1[0] * pt0[1] + pt1[0] * pt2[1] + pt2[0] * pt0[1] - pt2[0] * pt1[1],
         -pt0[0] * x4 + pt0[0] * pt2[1] * pt1[2] - pt1[0] * x5 + pt1[0] * pt0[1] * pt2[2] - pt2[0] * x3 + pt2[0] * pt1[
             1] * pt0[2]])


@vectorize(excluded=[1], signature="(i)->(i)")
def perp_to_vector(v, tol=1e-5):
    if norm_sq(v) >= tol:
        if (abs(v[0]) >= tol) or (abs(v[1]) >= tol):
            return _mycross(v, WORLD_XY[-1])
        else:
            return _mycross(v, WORLD_XY[1])
    else:
        return np.array([0.0, 0.0, 0.0])


def _mycross(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return np.array([ay * bz - by * az, az * bx - bz * ax, ax * by - bx * ay])


#def plane_plane_intersect(plane1: np.ndarray, plane2: np.ndarray):
#    normal1, normal2 = plane1[3], plane2[3]
#
#    # Stack the normal vectors
#    array_normals_stacked = np.vstack((normal1, normal2))
#
#    # Construct the matrix
#    matrix = np.block(
#        [
#            [2 * np.eye(3), array_normals_stacked.T],
#            [array_normals_stacked, np.zeros((2, 2))],
#        ]
#    )
#    #print(matrix)
#    # Construct the right-hand side of the equation
#    dot_a = scalar_dot(plane1[0], normal1)
#    dot_b = scalar_dot(plane2[0], normal2)
#    array_y = np.array([*plane1[0], dot_a, dot_b])
#    #print(matrix,array_y)
#    # Solve the linear system
#    solution = np.linalg.solve(matrix, array_y)
#    point_line = solution[:3]
#    direction_line = scalar_cross(normal1, normal2)
#
#    return point_line, direction_line


def plane_line_intersect(plane: np.ndarray, line, tol=1e-15, full_return=False):
    ray_origin, ray_direction = line
    ndotu = plane[-1].dot(ray_direction)
    if abs(ndotu) < tol:
        if full_return:
            return None, None, None
        return None
    w = ray_origin - plane[0]
    si = -plane[-1].dot(w) / ndotu
    Psi = w + si * ray_direction + plane[0]
    if full_return:
        return w, si, Psi
    return Psi


#def plane_plane_plane_intersect(self, other, third, tol=1e-15):
#    return plane_line_intersect(third, plane_plane_intersect(self, other), tol)


def plane_from_normal_origin(normal, origin):
    projected = project_point_by_normal(np.array([0.0, 0.0, 0.0]), normal, origin)
    xaxis_not_unit = projected - origin
    xaxis = xaxis_not_unit / scalar_norm(xaxis_not_unit)
    yaxis = scalar_cross(normal, xaxis)
    return np.array([origin, xaxis, yaxis, normal])


def project(pln, pt):
    """
    Calculate the projection of a point onto a plane.

    :param pln: The plane to project the point onto.
    :type pln: Plane|SlimPlane
    :param pt: The point to be projected onto the plane.
    :type pt: ndarray (shape: (3,))
    :return: The projected point.
    :rtype: ndarray (shape: (3,))
    """
    return pt - (scalar_dot(pln[-1], pt - pln[0]) * pln[-1])


def orient_matrix(p1, p2):
    trx = np.eye(4)
    trx[:3, :3] = p1[1:] @ p2[1:].T
    trx[-1, :-1] = p2[0] - p1[0]
    return trx


def local_to_world(pt, pln=WORLD_XY):
    """
    :param pt: The point in the local coordinate system that needs to be transformed to the world coordinate system.
    :type pt: numpy.ndarray

    :param pln: The reference plane in the world coordinate system that defines the transformation.
                It can be either a Plane or a SlimPlane object.
    :type pln: Plane or SlimPlane.

    :return: The transformed point in the world coordinate system.
    :rtype: numpy.ndarray
    """
    z = np.zeros(3)
    z += pln.origin
    z += pt[0] * pln.xaxis
    z += pt[1] * pln.yaxis
    z += pt[2] * pln.zaxis
    return z


def world_to_local(pt, pln):
    """ """
    return np.array([pln.xaxis, pln.yaxis, pln.zaxis]) @ (np.array(pt) - pln.origin)


def rotate(pln): ...


def translate(pln): ...


def orient(pln): ...


def transform(pln, m): ...


def offset(pln): ...

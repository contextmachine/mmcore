import numpy as np

from mmcore.geom.surfaces import Surface
from mmcore.numeric.closest_point import closest_point_on_surface

from mmcore.numeric.vectors import scalar_norm, scalar_cross

from mmcore.numeric.fdm import DEFAULT_H


def sectional_tangent(s1: Surface, s2: Surface, pt, tol=1e-3):
    """
    Calculate the tangent vector of intersection curve between two surfaces at a given point.

    :param s1: Surface 1
    :param s2: Surface 2
    :param pt: Point on the surfaces
    :param tol: Optional tolerance for finding closest points on surfaces (default: 1e-3)
    :return: Sectional tangent vector

    """
    uv1 = closest_point_on_surface(s1, pt, tol=tol)
    uv2 = closest_point_on_surface(s2, pt, tol=tol)
    n1, n2 = s1.normal(uv1), s1.normal(uv2)
    T = np.array(np.cross(n1, n2))
    T /= np.linalg.norm(T)
    return T


def sectional_tangent_der(s1: Surface, s2: Surface, pt, eps=DEFAULT_H, tol=1e-3):
    """
    Calculate the derivative of tangent vector of intersection curve between two surfaces at a given point.
ои
    :param s1: Surface object representing the first surface.
    :param s2: Surface object representing the second surface.
    :param pt: Point at which to calculate the derivative.
    :param eps: Step size used for finite difference approximation. Default is the value of DEFAULT_H.
    :param tol: Tolerance used for convergence. Default is 1e-3.
    :return: The unit tangent vector  between the two surfaces at the given point and its first derivative.
    """
    tangent = sectional_tangent(s1, s2, pt, tol=tol)
    veps = tangent * eps
    der = (sectional_tangent(s1, s2, pt + veps) - sectional_tangent(s1, s2, pt - veps, )) / (2 * np.linalg.norm(veps))
    return tangent, der


def sectional_curvature_vector(s1: Surface, s2: Surface, pt, eps=DEFAULT_H, tol=1e-3):
    """
    :param s1: surface object representing the first surface
    :param s2: surface object representing the second surface
    :param pt: point on the surface where the sectional curvature vector is calculated
    :param eps: parameter for numerical differentiation to calculate tangent vectors (default: DEFAULT_H)
    :param tol: tolerance for convergence criteria in tangent vector calculation (default: 1e-3)
    :return: The unit tangent and curvature vectors between the two surfaces at the given point.

    This method calculates the curvature vector of intersection curve between two surfaces at a given point.
     The sectional curvature vector provides information about the curvature of a surface along a particular direction.

     The method uses the `sectional_tangent_der` method to calculate the tangent vectors at the given point on the surfaces.


    """
    tangent,curvature_vector = sectional_tangent_der(s1, s2, pt, eps=eps, tol=tol)

    return curvature_vector


from __future__ import annotations

from mmcore.geom.vec import norm_sq, cross, norm, unit, gram_schmidt
from scipy.integrate import quad
import numpy as np

from mmcore.numeric.routines import divide_interval
from mmcore.numeric.divide_and_conquer import (
    recursive_divide_and_conquer_min,
    recursive_divide_and_conquer_max,
)


def plane_on_curve(O, T, D2):
    """
    Returns an array representing the plane on a curve.

    Example usage:
        O = [0, 0, 0]
        T = [1, 0, 0]
        D2 = [0, 1, 0]
        plane_on_curve(O, T, D2)
    """
    # Gram-Schmidt process on T and D2 to obtain orthogonal normal
    N = unit(gram_schmidt(T, D2))
    # Cross product of T and N to obtain the binormal
    B = cross(T, N)
    # Return array containing origin, tangent, normal and binormal
    return np.array([O, T, N, B])


def normal_at(D1, D2):
    N = unit(gram_schmidt(unit(D1), D2))
    return N


def swap_z_to_first_der(pln):
    z = np.zeros(pln.shape, dtype=float)
    z[..., 0, :], z[..., 1, :], z[..., 2, :], z[..., 3, :] = (
        pln[..., 0, :],
        pln[..., 2, :],
        pln[..., 3, :],
        pln[..., 1, :],
    )
    return z


def evaluate_tangent(D1, D2):
    d1 = np.linalg.norm(D1)
    if d1 == 0.0:
        d1 = np.linalg.norm(D2)
        T = D2 / d1 if d1 > 0.0 else np.zeros(D2.shape)
    else:
        T = D1 / d1
    return T, bool(d1)


evaluate_tangent_vec = np.vectorize(evaluate_tangent, signature="(i),(i)->(i),()")


def evaluate_length(first_der, t0: float, t1: float, **kwargs):
    """ """

    def ds(t):
        return norm(first_der(t))

    return quad(ds, t0, t1, **kwargs)


evaluate_length_vec = np.vectorize(
    evaluate_length, excluded=[0], signature="(),()->(),()"
)

import math


def calculate_curvature2d(dx, dy, ddx, ddy):
    numerator = abs(dx * ddy - dy * ddx)
    denominator = math.pow((dx**2 + dy**2), 1.5)
    curvature = numerator / denominator
    return numerator, denominator, curvature


def evaluate_curvature(
    derivative, second_derivative
) -> tuple[np.ndarray, np.ndarray, bool]:
    """
    Calculates the unit tangent vector, curvature vector, and a recalculate condition for a given derivative and
    second derivative.

    :param derivative: The derivative vector.
    :param second_derivative: The second derivative vector.
    :return: A tuple containing the unit tangent vector, curvature vector, and recalculate condition.

    Example usage:
        derivative = np.array([1, 0, 0])
        second_derivative = np.array([0, 1, 0])
        evaluate_curvature2(derivative, second_derivative)
    """
    # Norm of derivative
    norm_derivative = np.linalg.norm(derivative)
    zero_tolerance = 0.0

    # Check if norm of derivative is too small
    if norm_derivative == zero_tolerance:
        norm_derivative = np.linalg.norm(second_derivative)

        # If norm of second_derivative is above tolerance, calculate the unit tangent
        # If not, set unit tangent as zeros_like second_derivative
        if norm_derivative > zero_tolerance:
            unit_tangent_vector = second_derivative / norm_derivative
        else:
            unit_tangent_vector = np.zeros_like(second_derivative)

        # Set curvature vector to zero, we will not recalculate
        curvature_vector = np.zeros_like(second_derivative)
        recalculate_condition = False
    else:
        unit_tangent_vector = derivative / norm_derivative

        # Compute scalar component of curvature
        negative_second_derivative_dot_tangent = -np.dot(
            second_derivative, unit_tangent_vector
        )
        inverse_norm_derivative_squared = 1.0 / (norm_derivative * norm_derivative)

        # Calculate curvature vector
        curvature_vector = inverse_norm_derivative_squared * (
            second_derivative
            + negative_second_derivative_dot_tangent * unit_tangent_vector
        )

        # We will recalculate
        recalculate_condition = True

    return unit_tangent_vector, curvature_vector, recalculate_condition


"""
def evaluate_curvature(D1, D2) -> tuple[np.ndarray, np.ndarray, bool]:
    d1 = np.linalg.norm(D1)

    if d1 == 0.0:
        d1 = np.linalg.norm(D2)
        if d1 > 0.0:
            T = D2 / d1
        else:
            T = np.zeros_like(D2)
        K = np.zeros_like(D2)
        rc = False
    else:
        T = D1 / d1
        negD2oT = -np.dot(D2, T)
        d1 = 1.0 / (d1 * d1)
        K = d1 * (D2 + negD2oT * T)
        rc = True

    return T, K, rc
"""

evaluate_curvature_vec = np.vectorize(
    evaluate_curvature, signature="(i),(i)->(i),(i),()"
)


def evaluate_jacobian(ds_o_ds, ds_o_dt, dt_o_dt):
    a = ds_o_ds * dt_o_dt
    b = ds_o_dt * ds_o_dt
    det = a - b
    if (
        ds_o_ds <= dt_o_dt * np.finfo(float).eps
        or dt_o_dt <= ds_o_ds * np.finfo(float).eps
    ):
        # One of the partials is (numerically) zero w.r.t. the other partial - value of det is unreliable
        rc = False
    elif abs(det) <= max(a, b) * np.sqrt(np.finfo(float).eps):
        # Du and Dv are (numerically) (anti) parallel - value of det is unreliable.
        rc = False
    else:
        rc = True

    return det, rc


def evaluate_normal(
    gradient_u,
    gradient_v,
    second_derivative_uu,
    second_derivative_uv,
    second_derivative_vv,
    limit_direction=None,
):
    """
    :param gradient_u: The gradient vector in the u direction.
    :param gradient_v: The gradient vector in the v direction.
    :param second_derivative_uu: The second derivative in the uu direction.
    :param second_derivative_uv: The second derivative in the uv direction.
    :param second_derivative_vv: The second derivative in the vv direction.
    :param limit_direction: The limit direction for coefficient selection. Defaults to None.
    :return: The evaluated normal vector.

    This method evaluates the normal vector at a given point on a surface. It takes as input the gradient vectors
    in the u and v directions, as well as the second derivatives in various directions. Optionally, the limit direction
    can be specified to choose coefficients for certain cases.

    The method calculates the dot products of the gradient vectors and checks the jacobian_success. If the jacobian_success
    is True, the method returns the cross product of the gradient vectors. Otherwise, it calculates the coefficients based on
    the limit_direction, and uses them to calculate the cross products of the second derivatives and the gradient vectors.
    Finally, it adds the cross products together, normalizes the resulting vector, and returns it as the normal vector at the point.

    Example usage:
    gradient_u = [1, 0, 0]
    gradient_v = [0, 1, 0]
    second_derivative_uu = [1, 0, 0]
    second_derivative_uv = [0, 0, 1]
    second_derivative_vv = [0, 1, 0]
    limit_direction = 2

    evaluate_normal(gradient_u, gradient_v, second_derivative_uu, second_derivative_uv, second_derivative_vv, limit_direction)
    """
    dot_product_gradient_u = norm_sq(gradient_u)
    dot_product_gradient_uv = gradient_u * gradient_v
    dot_product_gradient_v = norm_sq(gradient_v)

    determinant, jacobian_success = evaluate_jacobian(
        dot_product_gradient_u, dot_product_gradient_uv, dot_product_gradient_v
    )

    if jacobian_success:
        return np.cross(gradient_u, gradient_v)

    coeff_a, coeff_b = {
        2: [-1.0, 1.0],
        3: [-1.0, -1.0],
        4: [1.0, -1.0],
    }.get(limit_direction, [1.0, 1.0])

    cross_vector_v = coeff_a * second_derivative_uv + coeff_b * second_derivative_vv
    cross_product_v = cross(gradient_u, cross_vector_v)

    cross_vector_u = coeff_a * second_derivative_uu + coeff_b * second_derivative_uv
    cross_product_u = cross(cross_vector_u, gradient_v)

    normal_vector = cross_product_v + cross_product_u
    normal_vector = normal_vector / np.linalg.norm(normal_vector)

    return normal_vector


"Explane "


def nurbs_bound_points(curve, tol=1e-5):
    """
    Returns a array of parameters whose evaluation gives you a set of points at least sufficient
    for correct estimation of the AABB(Axis-Aligned Bounding Box) of the curve.
    Also the set contains parameters of all extrema of the curve,
    but does not guarantee that the curve is extreme in all parameters.
    """

    def t_x(t):
        return -curve(t)[0]

    def t_y(t):
        return -curve(t)[1]

    def t_z(t):
        return -curve(t)[2]

    t_values = []

    def solve_interval(f, bounds):
        f_min, _ = recursive_divide_and_conquer_min(f, bounds, tol)
        f_max, _ = recursive_divide_and_conquer_max(f, bounds, tol)
        return f_min, f_max

    curve_start, curve_end = curve.interval()
    for start, end in divide_interval(curve_start, curve_end, step=1.0):
        t_values.extend(solve_interval(t_x, (start, end)))
        t_values.extend(solve_interval(t_y, (start, end)))
        t_values.extend(solve_interval(t_z, (start, end)))

    return np.unique(np.array([curve_start, *t_values, curve_end]))

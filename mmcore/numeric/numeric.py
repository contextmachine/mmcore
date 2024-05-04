from __future__ import annotations

from mmcore.geom.vec import norm_sq, cross, norm, unit, gram_schmidt
from scipy.integrate import quad
import numpy as np

from mmcore.numeric.plane import inverse_evaluate_plane
from mmcore.numeric.routines import divide_interval
from mmcore.numeric.divide_and_conquer import (
    recursive_divide_and_conquer_min,
    recursive_divide_and_conquer_max,
    test_all_roots,
    iterative_divide_and_conquer_min,
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
    """
    D1 - first derivative vector
    D2 - second derivative vector

    :math:`\\dfrac{D2}{||D1||}}  \\cos(\\omega x)f(x)dx` or
    :math:`\\int^b_a \\sin(\\omega x)f(x)dx`
    :param D1:
    :param D2:
    :return:

    """
    d1 = np.linalg.norm(D1)
    if np.isclose(d1, 0.0):
        d1 = np.linalg.norm(D2)
        T = D2 / d1 if d1 > 0.0 else np.zeros(D2.shape)
    else:
        T = D1 / d1
    return T, bool(d1)


evaluate_tangent_vec = np.vectorize(evaluate_tangent, signature="(i),(i)->(i),()")

from scipy.integrate import quadrature
def evaluate_length(first_der, t0: float, t1: float, **kwargs):
    """ """

    def ds(t):
        return norm(first_der(t))

    return quad(ds, t0, t1, **kwargs)


from scipy.optimize import newton,bisect


def evaluate_parameter_from_length(
    first_der,
    l: float,
    t0: float = 0.0,


    fprime=None,
    fprime2=None,
    t1_limit=None,
    tol=1e-8,
    maxiter=50,
            **kwargs

):
    """
    Evaluate the parameter 't' from the given length 'l'.

    :param first_der: The first derivative function.
    :param l: The target length value.
    :param t0: The initial estimate of the parameter 't'. Default is 0.0.
    :param fprime: The first derivative of the function. None by default.
    :param fprime2: The second derivative of the function. None by default.
    :param t1_limit: The limit for the parameter 't'. None by default.
    :param tol: The tolerance for the parameter 't'. Default is 1e-8.
    :param maxiter: The maximum number of iterations. Default is 50.
    :param kwargs: Additional keyword arguments.
    :return: The parameter 't' that corresponds to the target length 'l'.
    """


    def func_to_bisect(t):
        return evaluate_length(first_der, t0, t, **kwargs)[0] - l
    def func(t):
        return abs(evaluate_length(first_der, t0, t, **kwargs)[0] - l)


    #return newton(
    #    func, t0, tol=tol, maxiter=maxiter, x1=t1_limit, fprime=fprime, fprime2=fprime2
    #)
    res=iterative_divide_and_conquer_min(func, (t0,t1_limit), t1_limit*2)

    return newton(
       func, res[0], tol=tol, maxiter=maxiter, x1=t1_limit, fprime=fprime, fprime2=fprime2
    )


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
    dot_product_gradient_uv = np.dot(gradient_u, gradient_v)
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


def evaluate_normal2(
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
    dot_product_gradient_uv = np.dot(gradient_u, gradient_v)
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




def evaluate_sectional_curvature(derivative_u,
                                 derivative_v,
                                 second_derivative_uu,
                                 second_derivative_uv,
                                 second_derivative_vv,
                                 planeNormal):
    """
      S10, S01 - [in]
    surface 1st partial derivatives
  S20, S11, S02 - [in]
    surface 2nd partial derivatives
  planeNormal - [in]
    unit normal to section plane
  K - [out] Sectional curvature
    Curvature of the intersection curve of the surface
    and plane through the surface point where the partial
    derivatives were evaluationed.
    :param du:
    :param derivative_v:
    :param second_derivative_uu:
    :param second_derivative_uv:
    :param second_derivative_vv:
    :param planeNormal:
    :return:
    """
    M = cross(derivative_u, derivative_v)
    D1 = cross(M, planeNormal)

    matrix = np.array([list(derivative_u), list(derivative_v)])
    vec = np.array(list(D1))

    try:
        # Attempt to solve the system of linear equations
        a, b = np.linalg.solve(matrix.transpose(), vec)
        D2 = a * second_derivative_uu + b * second_derivative_uv
        M = cross(D2, derivative_v)
        D2 = a * second_derivative_uv + b * second_derivative_vv
        M2 = cross(derivative_u, D2)

        M = M + M2
        D2 = cross(M, planeNormal)

        a = np.dot(D1, D1)

        if not (a > np.finfo(float).eps):
            return np.array([j for i in [0.0] * 3 for j in [i]])

        a = 1.0 / a
        b = -a * np.dot(D2, D1)
        K = a * (D2 + b * D1)

        return K

    except np.linalg.LinAlgError:
        return np.array([j for i in [0.0] * 3 for j in [i]])
    except Exception as e:
        print(e)
        return np.array([j for i in [0.0] * 3 for j in [i]])

def curve_bound_points(curve, bounds=None, tol=1e-5):
    """
    Returns a array of parameters whose evaluation gives you a set of points at least sufficient
    for correct estimation of the AABB(Axis-Aligned Bounding Box) of the curve.
    Also the set contains parameters of all extrema of the curve,
    but does not guarantee that the curve is extreme in all parameters.
    """

    def t_x(t):
        return curve.evaluate(t)[0]

    def t_y(t):
        return curve.evaluate(t)[1]

    def t_z(t):
        return curve.evaluate(t)[2]

    t_values = []

    def solve_interval(f, bnds):
        f_min, _ = iterative_divide_and_conquer_min(f, bnds, tol=tol)
        f_max, _ = iterative_divide_and_conquer_min(lambda t: -f(t), bnds, tol=tol)
        return f_min, f_max

    curve_start, curve_end = curve.interval() if bounds is None else bounds
    if (curve_end - curve_start) > 1.0:
        for start, end in divide_interval(curve_start, curve_end, step=1.0):
            t_values.extend(solve_interval(t_x, (start, end)))
            t_values.extend(solve_interval(t_y, (start, end)))
            t_values.extend(solve_interval(t_z, (start, end)))
    else:
        t_values.extend(solve_interval(t_x, (curve_start, curve_end)))
        t_values.extend(solve_interval(t_y, (curve_start, curve_end)))
        t_values.extend(solve_interval(t_z, (curve_start, curve_end)))
    return np.unique(np.array([curve_start, *t_values, curve_end]))


def curve_roots(curve, axis=1):
    _curve_fun = getattr(curve, "evaluate", curve)

    def f(t):
        xyz = _curve_fun(t)
        return xyz[axis]

    if hasattr(curve, "degree"):
        tol = 10 ** (-curve.degree)
    else:
        tol = 0.01
    roots = []
    for start, end in divide_interval(*curve.interval(), step=0.5):
        roots.extend(test_all_roots(f, (start, end), tol))
    return roots
if __name__ == '__main__':
    from mmcore.geom.curves import NURBSpline
    bb = NURBSpline(
        np.random.random((25,3))
    )
    import time
    s=time.time()
    rr=bb.evaluate_length((0., 0.23))
    print(divmod(time.time()-s,60))
    s = time.time()
    rrr=bb.evaluate_length((0.0, (bb.interval()[1]-bb.interval()[0])*0.9))
    print(divmod(time.time() - s, 60))

    s = time.time()
    print(bb.evaluate_parameter_at_length(rrr), (bb.interval()[1] - bb.interval()[0]) * 0.9)
    print(divmod(time.time() - s, 60))
    s = time.time()

    print(
        bb.evaluate_parameter_at_length(rr),0.23)
    print(divmod(time.time() - s, 60))

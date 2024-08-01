from __future__ import annotations

from mmcore.geom.vec import norm_sq, cross, norm, unit, gram_schmidt
from scipy.integrate import quad
import numpy as np


from mmcore.numeric.routines import divide_interval
from mmcore.numeric.divide_and_conquer import (
    recursive_divide_and_conquer_min,
    recursive_divide_and_conquer_max,
    test_all_roots,
    iterative_divide_and_conquer_min,
)
from mmcore.numeric.vectors import scalar_dot, scalar_cross, scalar_norm


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
        return scalar_norm(first_der(t))

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
        negative_second_derivative_dot_tangent = -scalar_dot(
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
    dot_product_gradient_u = scalar_dot(gradient_u,gradient_u)
    dot_product_gradient_uv = scalar_dot(gradient_u, gradient_v)
    dot_product_gradient_v =  scalar_dot(gradient_v,gradient_v)

    determinant, jacobian_success = evaluate_jacobian(
        dot_product_gradient_u, dot_product_gradient_uv, dot_product_gradient_v
    )

    if jacobian_success:
        return scalar_cross(gradient_u, gradient_v)

    coeff_a, coeff_b = {
        2: [-1.0, 1.0],
        3: [-1.0, -1.0],
        4: [1.0, -1.0],
    }.get(limit_direction, [1.0, 1.0])

    cross_vector_v = coeff_a * second_derivative_uv + coeff_b * second_derivative_vv
    cross_product_v =  scalar_cross(gradient_u, cross_vector_v)

    cross_vector_u = coeff_a * second_derivative_uu + coeff_b * second_derivative_uv
    cross_product_u = scalar_cross(cross_vector_u, gradient_v)

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
    dot_product_gradient_u = scalar_dot(gradient_u, gradient_u)
    dot_product_gradient_uv = scalar_dot(gradient_u, gradient_v)
    dot_product_gradient_v = scalar_dot(gradient_v, gradient_v)
    determinant, jacobian_success = evaluate_jacobian(
        dot_product_gradient_u, dot_product_gradient_uv, dot_product_gradient_v
    )
    if jacobian_success:
        return scalar_cross(gradient_u, gradient_v)
    coeff_a, coeff_b = {
        2: [-1.0, 1.0],
        3: [-1.0, -1.0],
        4: [1.0, -1.0],
    }.get(limit_direction, [1.0, 1.0])
    cross_vector_v = coeff_a * second_derivative_uv + coeff_b * second_derivative_vv
    cross_product_v = scalar_cross(gradient_u, cross_vector_v)
    cross_vector_u = coeff_a * second_derivative_uu + coeff_b * second_derivative_uv
    cross_product_u = scalar_cross(cross_vector_u, gradient_v)
    normal_vector = cross_product_v + cross_product_u
    normal_vector = normal_vector / np.linalg.norm(normal_vector)
    return normal_vector


import math
import numpy as np


def solve3x2(col0, col1, d0, d1, d2):
    """
    Solve a 3x2 system of linear equations

    Input:
    col0, col1: lists of 3 floats
    d0, d1, d2: right hand column of system

    Output:
    Tuple containing:
    - return code:
        2: successful
        0: failure - 3x2 matrix has rank 0
        1: failure - 3x2 matrix has rank 1
    - x, y: solution
    - err: error term
    - pivot_ratio: min(|pivots|)/max(|pivots|)

    If the return code is 2, then
    x*col0 + y*col1 + err*(col0 X col1)/|col0 X col1| = [d0,d1,d2]

    The pivot_ratio indicates how well-conditioned the matrix is.
    If this number is small, the 3x2 matrix may be singular or ill-conditioned.
    """
    x, y = 0.0, 0.0
    pivot_ratio = 0.0
    err = float('inf')

    i = np.argmax([abs(val) for val in col0 + col1])
    if i >= 3:
        col0, col1 = col1, col0
        x, y = y, x

    if max(map(abs, col0 + col1)) == 0.0:
        return 0, x, y, err, pivot_ratio

    pivot_ratio = abs(max(map(abs, col0 + col1)))

    i %= 3
    if i == 1:
        col0[0], col0[1] = col0[1], col0[0]
        col1[0], col1[1] = col1[1], col1[0]
        d0, d1 = d1, d0
    elif i == 2:
        col0[0], col0[2] = col0[2], col0[0]
        col1[0], col1[2] = col1[2], col1[0]
        d0, d2 = d2, d0

    col1[0] /= col0[0]
    d0 /= col0[0]

    if col0[1] != 0.0:
        col1[1] += -col0[1] * col1[0]
        d1 += -col0[1] * d0
    if col0[2] != 0.0:
        col1[2] += -col0[2] * col1[0]
        d2 += -col0[2] * d0

    if abs(col1[1]) > abs(col1[2]):
        pivot_ratio = min(pivot_ratio, abs(col1[1])) / max(pivot_ratio, abs(col1[1]))
        d1 /= col1[1]
        if col1[0] != 0.0:
            d0 += -col1[0] * d1
        if col1[2] != 0.0:
            d2 += -col1[2] * d1
        x, y, err = d0, d1, d2
    elif col1[2] == 0.0:
        return 1, x, y, err, pivot_ratio
    else:
        pivot_ratio = min(pivot_ratio, abs(col1[2])) / max(pivot_ratio, abs(col1[2]))
        d2 /= col1[2]
        if col1[0] != 0.0:
            d0 += -col1[0] * d2
        if col1[1] != 0.0:
            d1 += -col1[1] * d2
        x, y, err = d0, d2, d1

    return 2, x, y, err, pivot_ratio


def evaluate_sectional_curvature(S10, S01, S20, S11, S02, planeNormal):
    """
    Calculate the curvature of the intersection of a surface and a plane.

    Input:
    S10, S01, S20, S11, S02: 3D vectors representing surface derivatives
    planeNormal: 3D vector representing the normal of the intersecting plane

    Output:
    Tuple containing:
    - bool: True if calculation was successful, False otherwise
    - K: 3D vector representing the curvature
    """



    M = scalar_cross(S10, S01)
    D1 = scalar_cross(M, planeNormal)


    rank, a, b, e, pr = solve3x2(S10, S01, D1[0], D1[1], D1[2])
    if rank < 2:
        return False, np.array([0.0, 0.0, 0.0])

    D2 = np.array([a * S20[i] + b * S11[i] for i in range(3)])
    M = np.array(scalar_cross(D2, S01))
    D2 = np.array([a * S11[i] + b * S02[i] for i in range(3)])
    M = np.array([M[i] + scalar_cross(S10, D2)[i] for i in range(3)])

    D2 = scalar_cross(M, planeNormal)

    a = sum(d * d for d in D1)

    if a <= 1e-15:  # ON_DBL_MIN
        return False, np.array([0.0, 0.0, 0.0])

    a = 1.0 / a
    b = -a * sum(D2[i] * D1[i] for i in range(3))
    K = np.array([a * (D2[i] + b * D1[i]) for i in range(3)])

    return True, K


def curve_bound_points(curve, bounds=None, tol=1e-2):
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
    #if (curve_end - curve_start) > 1.0:
    #    for start, end in divide_interval(curve_start, curve_end, step=1.0):
    #        #solve_interval(t_x, (start, end))
    #        t_values.extend(solve_interval(t_x, (start, end)))
    #        t_values.extend(solve_interval(t_y, (start, end)))
    #        t_values.extend(solve_interval(t_z, (start, end)))
    #else:
    #    t_values.extend(solve_interval(t_x, (curve_start, curve_end)))
    #    t_values.extend(solve_interval(t_y, (curve_start, curve_end)))
    #    t_values.extend(solve_interval(t_z, (curve_start, curve_end)))
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





def crvs_to_numpy_poly(crv, n_samples=100,remap=True):
    t=np.linspace(*crv.interval(), n_samples)
    pts=crv(t)
    deg=len(crv.control_points) + 1
    t=np.linspace(0., 1, n_samples) if remap else t
    if pts[0].shape[-1]<3 or np.allclose(pts[...,-1],0.):

        crvx, crvy = (np.polynomial.Polynomial.fit(np.linspace(0.,1,n_samples) if remap else t,  pts[...,0], deg),
                      np.polynomial.Polynomial.fit(t, pts[..., 1],   deg))
        return crvx, crvy
    else:
        crvx, crvy , crvz = (np.polynomial.Polynomial.fit(t, pts[..., 0], deg),
                             np.polynomial.Polynomial.fit(t, pts[..., 1], deg),
                             np.polynomial.Polynomial.fit(
            t, pts[..., 2], deg))
        return crvx,crvy,crvz

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
    print(bb.evaluate_parameter_at_length(rrr, tol=1e-8), (bb.interval()[1] - bb.interval()[0]) * 0.9)
    print(divmod(time.time() - s, 60))
    s = time.time()

    print(
        bb.evaluate_parameter_at_length(rr, tol=1e-8),0.23)
    print(divmod(time.time() - s, 60))

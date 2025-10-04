from __future__ import annotations


from mmcore.numeric.vectors import scalar_norm, scalar_gram_schmidt,scalar_dot,scalar_unit,scalar_cross,cross, norm, unit, gram_schmidt

from scipy.integrate import quad
import numpy as np
from mmcore.numeric.newton.cnewton import newtons_method


from mmcore.numeric.fdm import fdm
from mmcore.numeric.integrate.romberg import romberg1d
from mmcore.numeric.routines import divide_interval
from mmcore.numeric.divide_and_conquer import (
    recursive_divide_and_conquer_min,
    recursive_divide_and_conquer_max,
    test_all_roots,
    iterative_divide_and_conquer_min,
)
def norm_sq(v):
    return scalar_dot(v,v)
from mmcore.numeric.vectors import scalar_dot, scalar_cross, scalar_norm
def normal_from_4pt(a,b,c,d):
    return scalar_cross(c-a,d-b)

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


def evaluate_length(first_der, t0: float, t1: float, tol=1e-3):
    """ """



    if tol<1e-3:

        return quad(lambda t: scalar_norm(first_der(t)), t0, t1, epsabs=tol,epsrel=tol)

    else:

        return romberg1d(lambda t: scalar_norm(first_der(t)), t0, t1, max_steps=32,acc=tol),tol


from scipy.optimize import newton


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
    #    func, t0, spt=spt, maxiter=maxiter, x1=t1_limit, fprime=fprime, fprime2=fprime2
    #)
    res = iterative_divide_and_conquer_min(func, (t0, t1_limit), np.sqrt(tol))
    if fprime is None:
        fprime= fdm(func)
    if fprime2 is None:
        fprime2 = fdm(fprime)
    return newton(
        func, res[0], tol=tol, maxiter=maxiter, x1=t1_limit, fprime=fprime, fprime2=fprime2
    )


evaluate_length_vec = np.vectorize(
    evaluate_length, excluded=[0], signature="(),()->(),()"
)


def calculate_curvature2d(dx, dy, ddx, ddy):
    numerator = abs(dx * ddy - dy * ddx)
    denominator = math.pow((dx ** 2 + dy ** 2), 1.5)
    curvature = numerator / denominator
    return numerator, denominator, curvature


def evaluate_curvature(D1: np.ndarray, D2: np.ndarray):
    """
    Evaluate unit tangent and curvature vector from first and second derivatives.

    Parameters
    ----------
    D1 : np.ndarray
        First derivative vector.
    D2 : np.ndarray
        Second derivative vector.

    Returns
    -------
    
    T : np.ndarray
        Unit tangent vector.
    K : np.ndarray
        Curvature vector
    rc : bool
        True if the “normal” curvature formula was used (i.e. |D1| != 0), False otherwise.
    """
    rc = False
    d1 = np.linalg.norm(D1)

    if d1 == 0.0:
        # L'Hôpital case: if first derivative is zero but second is not,
        # tangent is ±unitized D2; curvature is zero.
        d1 = np.linalg.norm(D2)
        if d1 > 0.0:
            T = D2 / d1
        else:
            T = np.zeros_like(D2)
        K = np.zeros_like(D2)
    else:
        # T = D1/|D1|
        T = D1 / d1
        # K = (D2 - (D2·T) T) / |D1|^2
        negD2oT = -np.dot(D2, T)
        inv_d1sq = 1.0 / (d1 * d1)
        K = inv_d1sq * (D2 + negD2oT * T)
        rc = True

    return T, K, rc


def evaluate_curvature_1der(
    D1: np.ndarray,
    D2: np.ndarray,
    D3: np.ndarray,
    compute_kprime: bool = False,
    compute_torsion: bool = False
):
    """
    Evaluate unit tangent, curvature vector, and optionally first derivative of curvature (k') and torsion,
    using first, second, and third derivatives.

    Parameters
    ----------
    D1 : np.ndarray
        First derivative vector.
    D2 : np.ndarray
        Second derivative vector.
    D3 : np.ndarray
        Third derivative vector.
    compute_kprime : bool, optional
        If True, compute and return k' (first derivative of curvature).
    compute_torsion : bool, optional
        If True, compute and return torsion.

    Returns
    -------
    rc : bool
        True if either k' or torsion was computed successfully, False otherwise.
    T : np.ndarray or None
        Unit tangent vector, or None if |D1| == 0.
    K : np.ndarray or None
        Curvature vector, or None if |D1| == 0.
    kprime : float or None
        First derivative of curvature, or None if not requested or cannot be computed.
    torsion : float or None
        Torsion, or None if not requested or cannot be computed.
    """
    rc = False
    dsdt = np.linalg.norm(D1)

    T = None
    K = None
    kprime = None
    torsion = None

    if dsdt > 0.0:
        # Unit tangent
        T = D1 / dsdt

        # q = D1 × D2
        q = np.cross(D1, D2)
        qlen2 = np.dot(q, q)

        # Curvature vector K = (D2 - (D2·T) T) / |D1|^2
        dsdt2 = dsdt * dsdt
        K = (1.0 / dsdt2) * (D2 - np.dot(D2, T) * T)

        if compute_kprime:
            # q' = D1 × D3
            qprime = np.cross(D1, D3)
            if qlen2 > 0.0:
                # k' = [ (q·q') · |D1|^2 - 3·|q|^2·(D1·D2) ]
                #      / [ |q| · |D1|^5 ]
                numerator = np.dot(q, qprime) * dsdt2 - 3.0 * qlen2 * np.dot(D1, D2)
                denominator = np.sqrt(qlen2) * (dsdt ** 5)
                kprime = numerator / denominator
            else:
                # If q is zero, fallback to |q'| / |D1|^3
                kprime = np.linalg.norm(qprime) / (dsdt ** 3)
            rc = True

        if compute_torsion:
            if qlen2 > 0.0:
                # torsion = (q · D3) / |q|^2
                torsion = np.dot(q, D3) / qlen2
                rc = True
            else:
                # cannot compute torsion if binormal magnitude is zero
                rc = False

    return rc, T, K, kprime, torsion

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
_numeric_eps=np.finfo(float).eps

def compare_curvature(r1a, r2a, ka, r1b, r2b, kb,
                      rtol=1e-4, atol=1e-6,
                      check_vector=False):
 
    kappa_a = np.linalg.norm(ka)
    kappa_b = np.linalg.norm(kb)

    # print('kappa',kappa_a, kappa_b)
    # scalar comparison
    if abs(kappa_a - kappa_b) > max(rtol*max(abs(kappa_a), abs(kappa_b)), atol):
        # print("kappa res", False,abs(kappa_a - kappa_b),max(rtol*max(abs(kappa_a), abs(kappa_b)), atol) )
        return False

    if check_vector:
        ka_vec = np.cross(r1a, r2a) / (np.linalg.norm(r1a)**3)
        kb_vec = np.cross(r1b, r2b) / (np.linalg.norm(r1b)**3)
        # angle between curvature vectors
        cosang = np.dot(ka_vec, kb_vec) / (np.linalg.norm(ka_vec)*np.linalg.norm(kb_vec))
        return abs(np.arccos(np.clip(cosang, -1, 1))) < 1e-7
    return True

def evaluate_jacobian(du_o_du, du_o_dv, dv_o_dv):
    """
    S(u,v) - surface
    du=S'u, dv=S'v
    :param du_o_du:  du.dot(du)
    :param du_o_dv:  du.dot( dv)
    :param dv_o_dv:  dv.dot( dv)
    :return: Jacobian determinant and status

    """
    a = du_o_du * dv_o_dv
    b = du_o_dv * du_o_dv
    det = a - b
    if (
            du_o_du <= dv_o_dv * np.finfo(float).eps
            or dv_o_dv <= du_o_du * np.finfo(float).eps
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
    dot_product_gradient_u = scalar_dot(gradient_u, gradient_u)
    dot_product_gradient_uv = scalar_dot(gradient_u, gradient_v)
    dot_product_gradient_v = scalar_dot(gradient_v, gradient_v)

    determinant, jacobian_success = evaluate_jacobian(dot_product_gradient_u, dot_product_gradient_uv,
                                                      dot_product_gradient_v)

    if jacobian_success:
        return scalar_cross(gradient_u, gradient_v)

    coeff_a, coeff_b = {
        2: [-1.0, 1.0],
        3: [-1.0, -1.0],
        4: [1.0, -1.0],
    }.get(limit_direction, [1.0, 1.0]) # type: ignore

    cross_vector_v = coeff_a * second_derivative_uv + coeff_b * second_derivative_vv
    cross_product_v = scalar_cross(gradient_u, cross_vector_v)

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
    determinant, jacobian_success = evaluate_jacobian(dot_product_gradient_u, dot_product_gradient_uv,
                                                      dot_product_gradient_v)
    if jacobian_success:
        return scalar_cross(gradient_u, gradient_v)
    coeff_a, coeff_b = {
        2: [-1.0, 1.0],
        3: [-1.0, -1.0],
        4: [1.0, -1.0],
    }.get(limit_direction, [1.0, 1.0]) # type: ignore

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
    #print("SX",col0, col1, d0, d1, d2)
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


import numpy as np

def evaluate_sectional_curvature(
    Su: np.ndarray,   # S10  –  first‑order surface partial ∂S/∂u
    Sv: np.ndarray,   # S01  –  first‑order surface partial ∂S/∂v
    Suu: np.ndarray,  # S20  –  second‑order surface partial ∂²S/∂u²
    Suv: np.ndarray,  # S11  –  mixed second‑order partial ∂²S/∂u∂v
    Svv: np.ndarray,  # S02  –  second‑order surface partial ∂²S/∂v²
    plane_normal: np.ndarray  # unit normal of the section plane
):
    """
    Evaluate the sectional curvature vector K of the curve that is the
    intersection between a surface and a plane passing through the surface
    point where the partials were measured.

    All vectors are plain 1‑D NumPy arrays with shape (3,).
    Returns (success: bool, K: np.ndarray).
    """
    # ------------------------------------------------------------------
    # 1.   M  = Su × Sv  (unnormalised surface normal)
    # 2.   D1 = M × plane_normal  (tangent of the intersection curve)
    # 3.   Solve D1 = a*Su + b*Sv  for (a,  b)
    # 4.   M1 = (a*Suu + b*Suv) × Sv  +  Su × (a*Suv + b*Svv)
    # 5.   D2 = M1 × plane_normal  (second derivative of the curve)
    # 6.   Remove normal component of D2 and scale to get curvature K
    # ------------------------------------------------------------------

    # Helper constants
    DBL_MIN = np.finfo(float).tiny
    print(DBL_MIN)

    # 1.
    M = np.cross(Su, Sv)

    # 2.
    D1 = np.cross(M, plane_normal)

    # 3. Solve the 3×2 linear system [Su  |  Sv] · [a,  b]^T = D1
    A = np.column_stack((Su, Sv))          # shape (3,  2)
    status,a, b, err, pivot_ratio = solve3x2(Su,Sv, D1[0],D1[1],D1[2])
    #(a, b), residuals, rank, _ = np.linalg.lstsq(A, D1, rcond=None)
    if status < 2:                           # Su and Sv are not independent
        print("F",Su,Sv,plane_normal, status,a,b,err,pivot_ratio,D1)
        return False, np.zeros(3)
    

    # 4. M1
    M1  = np.cross(a * Suu + b * Suv, Sv)
    M1 += np.cross(Su,  a * Suv + b * Svv)

    # 5.
    D2 = np.cross(M1, plane_normal)

    # 6. Project away the component of D2 parallel to D1
    d1_len2 = D1.dot(D1)
    if d1_len2 <= DBL_MIN:
        return False, np.zeros(3)

    inv_d1_len2   = 1.0 / d1_len2
    b_scalar      = -inv_d1_len2 * D2.dot(D1)
    K             = inv_d1_len2 * (D2 + b_scalar * D1)

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


def curve_bound_points2(curve, bounds=None, neg=False, tol=1e-5):
    """
    Returns a array of parameters whose evaluation gives you a set of points at least sufficient
    for correct estimation of the AABB(Axis-Aligned Bounding Box) of the curve.
    Also the set contains parameters of all extrema of the curve,
    but does not guarantee that the curve is extreme in all parameters.
    """

    low, high = np.zeros((3,)) + bounds[0], np.zeros((3,)) + bounds[1]

    while np.all(np.abs(high - low) >= tol):
        m1 = low + (high - low) / 4
        m2 = high - (high - low) / 4
        if neg:
            xyz1 = -1 * np.diag(np.array(curve(m1)))
            xyz2 = -1 * np.diag(np.array(curve(m2)))
        else:
            xyz1 = np.diag(np.array(curve(m1)))
            xyz2 = np.diag(np.array(curve(m2)))

        mask = xyz1 < xyz2
        inv_mask = np.bitwise_not(mask)
        high[mask] = m2[mask]
        low[inv_mask] = m1[inv_mask]

    x_min = (low + high) / 2

    return x_min, curve(x_min)


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


def crvs_to_numpy_poly(crv, n_samples=100, remap=True):
    t = np.linspace(*crv.interval(), n_samples)
    pts = crv(t)
    deg = len(crv.control_points) + 1
    t = np.linspace(0., 1, n_samples) if remap else t
    if pts[0].shape[-1] < 3 or np.allclose(pts[..., -1], 0.):

        crvx, crvy = (np.polynomial.Polynomial.fit(np.linspace(0., 1, n_samples) if remap else t, pts[..., 0], deg),
                      np.polynomial.Polynomial.fit(t, pts[..., 1], deg))
        return crvx, crvy
    else:
        crvx, crvy, crvz = (np.polynomial.Polynomial.fit(t, pts[..., 0], deg),
                            np.polynomial.Polynomial.fit(t, pts[..., 1], deg),
                            np.polynomial.Polynomial.fit(
                                t, pts[..., 2], deg))
        return crvx, crvy, crvz


import numpy as np

def compute_parametric_tolerance_curve(C1, C2, spt, angle_tol=None,**kwargs):
    """
    Compute the parametric step (du) for a NURBS curve given spatial and optional angular tolerances.

    Parameters:
    -----------
    C1 : array_like, shape (n,)
        First derivative vector C'(t) at the current parameter t.
    C2 : array_like, shape (n,)
        Second derivative vector C''(t) at the current parameter t.
    spt : float
        Spatial tolerance (maximum allowed positional deviation).
    angle_tol : float, optional
        Angular tolerance in radians (maximum allowed change in tangent direction).
        If None or 0, only spatial tolerance is enforced.

    Returns:
    --------
    dt : float
        The computed parameter increment ensuring neither positional nor
        (if angle_tol is not None) angular deviation exceed the given tolerances.
    """
    C1 = np.asarray(C1, dtype=float)
    C2 = np.asarray(C2, dtype=float)

    # Norms of first and second derivatives
    norm_Cp = np.linalg.norm(C1)
    norm_Cpp = np.linalg.norm(C2)

    # 1. First-order spatial bound: du_spatial1 = spt / ||C'(t)||
    if norm_Cp > 0:
        du_spatial1 = spt / norm_Cp
    else:
        du_spatial1 = np.inf

    # 2. Second-order (curvature-chord) spatial bound:
    #    du_spatial2 = sqrt(8 * spt / (||C''(t)|| * ||C'(t)||^2))
    if norm_Cp > 0 and norm_Cpp > 0:
        du_spatial2 = np.sqrt(8 * spt / (norm_Cpp * norm_Cp**2))
    else:
        du_spatial2 = np.inf

    # Choose the stricter of the two spatial bounds
    du_spatial = min(du_spatial1, du_spatial2)

    # If angular tolerance is not provided or zero, return spatial-only step
    if angle_tol is None or angle_tol <= 0:
        return du_spatial

    # 3. Angular bound: du_angular = angle_tol * ||C'(t)||^2 / ||C'(t) x C''(t)||
    cross_norm = np.linalg.norm(np.cross(C1, C2))
    if norm_Cp > 0 and cross_norm > 0:
        du_angular = angle_tol * (norm_Cp**2) / cross_norm
    else:
        du_angular = np.inf

    # 4. Final step is the minimum of spatial and angular bounds
    du = min(du_spatial, du_angular)
    return du


def compute_parametric_tolerance_surface(Su, Sv, Suu, Suv, Svv, spt, angle_tol=None, **kwargs):
    """
    Compute parameter increments (du, dv) for a NURBS surface given spatial and optional angular tolerances.

    Parameters:
    -----------
    Su : array_like, shape (3,)
        First partial derivative vector S_u at the current (u, v).
    Sv : array_like, shape (3,)
        First partial derivative vector S_v at the current (u, v).
    Suu : array_like, shape (3,)
        Second partial derivative vector S_{uu} at the current (u, v).
    Suv : array_like, shape (3,)
        Mixed partial derivative vector S_{uv} at the current (u, v).
    Svv : array_like, shape (3,)
        Second partial derivative vector S_{vv} at the current (u, v).
    spt : float
        Spatial tolerance (maximum allowed positional deviation).
    angle_tol : float, optional
        Angular tolerance in radians (maximum allowed change in surface normal).
        If None or 0, only spatial tolerance is enforced.

    Returns:
    --------
    du : float
        The computed parameter increment in the u-direction.
    dv : float
        The computed parameter increment in the v-direction.
    """

    # Norms of first partial derivatives
    norm_Su = np.linalg.norm(Su)
    norm_Sv = np.linalg.norm(Sv)

    # Compute surface normal N = Su x Sv
    N = np.cross(Su, Sv)
    norm_N = np.linalg.norm(N)

    # Spatial bounds (first-order)
    if norm_Su > 0:
        du_spatial = spt / norm_Su
    else:
        du_spatial = np.inf

    if norm_Sv > 0:
        dv_spatial = spt / norm_Sv
    else:
        dv_spatial = np.inf

    # If angular tolerance not provided or <= 0, return spatial-only steps
    if angle_tol is None or angle_tol <= 0:
        return du_spatial, dv_spatial

    # Compute derivatives of the normal with respect to u and v:
    # dN/du = S_{uu} x Sv + Su x S_{uv}
    dN_du = np.cross(Suu, Sv) + np.cross(Su, Suv)
    # dN/dv = S_{uv} x Sv + Su x S_{vv}
    dN_dv = np.cross(Suv, Sv) + np.cross(Su, Svv)

    norm_dN_du = np.linalg.norm(dN_du)
    norm_dN_dv = np.linalg.norm(dN_dv)

    # Angular bounds: ensure angle between N(u,v) and N(u+du,v) <= angle_tol
    # angle ≈ ||dN/du|| / ||N|| * du  => du_angular = angle_tol * ||N|| / ||dN/du||
    if norm_N > 0 and norm_dN_du > 0:
        du_angular = angle_tol * (norm_N / norm_dN_du)
    else:
        du_angular = np.inf

    # similarly for dv
    if norm_N > 0 and norm_dN_dv > 0:
        dv_angular = angle_tol * (norm_N / norm_dN_dv)
    else:
        dv_angular = np.inf

    # Final increments as minimum of spatial and angular bounds
    du = min(du_spatial, du_angular)
    dv = min(dv_spatial, dv_angular)

    return du, dv

import numpy as np

def compute_parametric_curvature_tolerance_surface(Su, Sv, Suu, Svv, spt, *, use_small_angle_thresh=1.0e-3):
    """
    Curvature-based parametric steps for a tensor-product surface S(u, v).

    We treat each isoparametric curve as a spatial curve:
        • u-direction (v = const):  r(u) = S(u, v0) with r' = S_u,  r'' = S_uu
        • v-direction (u = const):  r(v) = S(u0, v) with r' = S_v,  r'' = S_vv

    For a curve r(·), the curvature magnitude is
        κ = ||r' × r''|| / ||r'||³.
    Approximating locally by a circular arc and prescribing sagitta h = spt,
    we invert the sagitta formula to get the arc length s, then convert to a
    parametric step via du = s / ||S_u|| and dv = s / ||S_v||.

    Parameters
    ----------
    Su  : array_like (3,)
        First partial derivative S_u(u, v).
    Sv  : array_like (3,)
        First partial derivative S_v(u, v).
    Suu : array_like (3,)
        Second partial S_uu(u, v).
    Svv : array_like (3,)
        Second partial S_vv(u, v).
    spt : float
        Desired sagitta (chord-height) tolerance in 3D.
    use_small_angle_thresh : float, optional
        Switch point between exact sagitta inversion and small-angle
        approximation h ≈ κ s² / 8. Default 1e-3.

    Returns
    -------
    du, dv : tuple of floats
        Parametric increments in the u- and v-directions whose *actual*
        sagittas are approximately `spt` when moving along the corresponding
        isoparametric curves. Returns `np.inf` for a direction if its speed
        or curvature is zero (straight segment or degenerate partial).
    """
    Su  = np.asarray(Su,  dtype=float)
    Sv  = np.asarray(Sv,  dtype=float)
    Suu = np.asarray(Suu, dtype=float)
    Svv = np.asarray(Svv, dtype=float)

    def _one_dir_step(C1, C2):
        # Speed and curvature magnitude for the isoparametric curve
        norm_C1    = np.linalg.norm(C1)
        cross_norm = np.linalg.norm(np.cross(C1, C2))
        if norm_C1 == 0.0 or cross_norm == 0.0:
            return np.inf

        kappa = cross_norm / (norm_C1**3)
        kappa_tol = kappa * spt

        if kappa_tol >= use_small_angle_thresh:
            # Exact inversion: h = R (1 - cos(θ/2)), R = 1/κ  ⇒  cos(θ/2) = 1 - hκ
            c = max(-1.0, min(1.0, 1.0 - kappa_tol))  # guard acos domain
            half_theta = np.arccos(c)                  # θ/2
            s = 2.0 * half_theta / kappa              # s = θ / κ
        else:
            # Small-angle: h ≈ κ s² / 8  ⇒  s = sqrt(8 h / κ)
            s = np.sqrt(8.0 * spt / kappa)

        return s / norm_C1  # parametric step

    du = _one_dir_step(Su, Suu)
    dv = _one_dir_step(Sv, Svv)
    return du, dv
def compute_parametric_curvature_tolerance_curve(C1, C2, spt, *, use_small_angle_thresh=1.0e-3):
    """
    Compute a parametric increment du such that the sagitta (maximum deviation of the
    real curve from the straight‑line chord) on [t, t+du] equals `spt`.

    The derivation treats the curve locally as a circular arc:

        • Radius            R  = 1/κ
        • Arc length        s  = R θ
        • Sagitta (sag)     h  = R (1 − cos(θ⁄2))

    Solving h = spt for θ and converting arc length s → parametric step du gives
        du = s / ||C'(t)||.

    Parameters
    ----------
    C1 : array_like (n,)
        First derivative  **C'(t)** at the current parameter value t.
    C2 : array_like (n,)
        Second derivative **C''(t)** at t.
    spt : float
        Desired sagitta (chord‑height) tolerance.
    use_small_angle_thresh : float, optional
        Transition point for switching from the exact formula to the small‑angle
        approximation  h ≈ κs²/8.  The default (1×10⁻³) is conservative and
        avoids catastrophic cancellation when κ·spt is tiny.

    Returns
    -------
    du : float
        Parametric increment whose *actual* sagitta ≈ `spt`.
        `np.inf` is returned when curvature is zero (straight segment) or when
        ||C'(t)|| == 0.
    """
    # Convert inputs to ndarray
    C1 = np.asarray(C1, dtype=float)
    C2 = np.asarray(C2, dtype=float)

    # Speed and curvature magnitude
    norm_Cp   = np.linalg.norm(C1)
    cross_norm = np.linalg.norm(np.cross(C1, C2))
    if norm_Cp == 0 or cross_norm == 0:
        # Degenerate: no movement or zero curvature → the chord is exact.
        return np.inf

    kappa = cross_norm / norm_Cp**3                     # |κ|   (scalar, ≥ 0)
    kappa_tol = kappa * spt

    # ------------------------------------------------------------------
    # 1.  Exact sagitta inversion:  h = R (1 − cos θ/2)
    # ------------------------------------------------------------------
    if kappa_tol >= use_small_angle_thresh:
        # R = 1/κ,  h = R(1 − cos θ/2)  ⇒  cos θ/2 = 1 − hκ
        # Guard domain: 0 ≤ hκ ≤ 2  (θ ∊ [0, π])
        # Numerical clamp to stay inside acos domain
        c = max(-1.0, min(1.0, 1.0 - kappa_tol))
        half_theta = np.arccos(c)        # θ/2
        s = 2.0 * half_theta / kappa     # s = R θ = (θ)/κ
    else:
        # ------------------------------------------------------------------
        # 2.  Small‑angle sagitta approximation: h ≈ κ s² / 8
        #     Solve s = √(8 h / κ)
        # ------------------------------------------------------------------
        s = np.sqrt(8.0 * spt / kappa)

    # Parametric step:  du = s / |C'|
    du = s / norm_Cp
    return du

def compute_parametric_sectional_curvature_tolerance_surface(Su: np.ndarray, Sv: np.ndarray, Suu: np.ndarray,
                                                             Suv: np.ndarray, Svv: np.ndarray, tangent: np.ndarray,
                                                            spt: float, *, use_small_angle_thresh: float = 1.0e-3):
    """
    Parametric step (du, dv) for a surface so that the sagitta of the
    resulting *section curve* equals `tol`.

    The procedure:
      1. Build the section plane normal  np = N × T.
      2. Use `evaluate_sectional_curvature` to obtain curvature vector **K**.
      3. Compute |κ| = ||K|| and solve the circular‑arc sagitta equation
         to get arc‑length s (exact or small‑angle branch).
      4. Solve  [Su  Sv] · [du/ds, dv/ds]^T = T   (least‑squares)
         and scale by s to obtain (du, dv).

    Parameters
    ----------
    Su, Sv : (3,) ndarray
        ∂S/∂u and ∂S/∂v at the current (u,v).
    Suu, Suv, Svv : (3,) ndarray
        Second‑order derivatives ∂²S/∂u², ∂²S/∂u∂v, ∂²S/∂v².
    tangent : (3,) ndarray
        3‑D vector indicating the *surface‑tangent* direction of motion.
        Need not be unit but must be tangent to the surface.
    spt : float
        Maximum permitted chord‑height deviation.
    use_small_angle_thresh : float, optional
        Switch‑over value for the small‑angle formulation (see the
        earlier explanation).

    Returns
    -------
    du, dv : float
        Parameter increments that advance the surface point by the
        arc length whose sagitta equals `tol` along the specified direction.
        If curvature is zero (flat section) or the system is degenerate,
        `(np.inf, np.inf)` is returned.

    Notes
    -----
    • All vectors are treated in double precision.
    • The routine assumes the derivatives have been evaluated in the same
      *surface parameterisation* as your downstream code.
    """
    # ------------------------------------------------------------------
    # 0.  Quick guards
    # ------------------------------------------------------------------
    T = np.asarray(tangent, dtype=float)
    t_len = np.linalg.norm(T)
    if t_len == 0.0:
        return np.inf, np.inf
    T /= t_len                                            # unit tangent

    # Surface normal (unnormalised) and its length
    N = np.cross(Su, Sv)
    n_len = np.linalg.norm(N)
    if n_len == 0.0:                                      # degenerate patch
        return np.inf, np.inf
    N /= n_len                                            # unit surface normal

    # ------------------------------------------------------------------
    # 1.  Plane normal for the section curve
    # ------------------------------------------------------------------
    plane_normal = np.cross(N, T)
    pn_len = np.linalg.norm(plane_normal)
    if pn_len == 0.0:                                     # T ‖ N (not on surface)
        return np.inf, np.inf
    plane_normal /= pn_len

    # ------------------------------------------------------------------
    # 2.  Sectional curvature vector
    # ------------------------------------------------------------------
    ok, K_vec = evaluate_sectional_curvature(
        Su, Sv, Suu, Suv, Svv, plane_normal
    )
    #print('K',ok,K_vec)
    if not ok:
        return np.inf, np.inf

    kappa = np.linalg.norm(K_vec)                         # scalar curvature
    if kappa == 0.0:                                      # locally straight
        return np.inf, np.inf

    # ------------------------------------------------------------------
    # 3.  Arc‑length s whose sagitta equals `tol`
    # ------------------------------------------------------------------
    kappa_tol = kappa * spt
    if kappa_tol >= use_small_angle_thresh:
        # exact inversion  h = (1/κ)(1 − cos θ/2)
        c = 1.0 - kappa_tol
        c = np.clip(c, -1.0, 1.0)                        # numeric safety
        half_theta = np.arccos(c)
        s = 2.0 * half_theta / kappa                     # s = R θ
    else:
        # small‑angle approximation  h ≈ κ s² / 8
        s = np.sqrt(8.0 * spt / kappa)

    # ------------------------------------------------------------------
    # 4.  Map the physical step s back to parameter space
    # ------------------------------------------------------------------
    # Solve  Su * a + Sv * b = T   (least‑squares gives du/ds, dv/ds)
    #A = np.column_stack((Su, Sv))                         # shape (3,2)
    success,du_ds, dv_ds,err,pivot_ratio=solve3x2(Su,Sv, T[0],T[1],T[2])
    if success==0:
        raise ValueError('failed to solve sectional curvature')
    if success == 1:
            raise ValueError('failed to solve sectional curvature')
    #(du_ds, dv_ds), *_ = np.linalg.lstsq(A, T, rcond=None)
    if np.isfinite(s)    :
        # Final parameter increments
        du = du_ds * s
        dv = dv_ds * s
    else:
        du = du_ds * 1e-3
        dv = dv_ds * 1e-3
    
    return du, dv

def circle_of_curvature(curve, t: float):
    origin = curve.evaluate(t)
    T, K, success = evaluate_curvature(curve.derivative(t), curve.second_derivative(t))

    N = K/scalar_norm(K)
    B = scalar_cross(T, N)
    k = scalar_norm(K)
    R = 1 / k

    return (
        origin, R,
        np.array([origin + N * R, T, N, B])

    )  # Plane of curvature circle, Radius of curvature circle



def circular_segment_area_from_kappa_and_s(kappa_mag, s, small_theta=1e-3):
    """
    Return the (positive) area between a circular arc (constant curvature kappa_mag)
    of length s and its chord. Uses exact formula for general theta and a stable
    series for small theta.
    """
    if kappa_mag <= 0 or s <= 0:
        return 0.0
    theta = kappa_mag * s
    if theta < small_theta:
        # A = κ s^3/12 - κ^3 s^5/240 + O(s^7)
        return (kappa_mag * s**3)/12.0 - (kappa_mag**3 * s**5)/240.0
    # Exact: A = (θ - sin θ) / (2 κ^2)
    return (theta - np.sin(theta)) / (2.0 * kappa_mag**2)


def signed_curvature(C1, C2, n):
    """
    kappa_signed = ((C1 x C2) · n) / ||C1||^3   (planar curve embedded in 3D)
    """
    C1 = np.asarray(C1, float); C2 = np.asarray(C2, float); n = np.asarray(n, float)
    spd = np.linalg.norm(C1)
    if spd == 0: return 0.0
    k_mag = np.linalg.norm(np.cross(C1, C2)) / (spd**3)
    sgn   = np.sign(np.dot(np.cross(C1, C2), n))
    return float(sgn * k_mag)
if __name__ == "__main__":
    # Example derivatives at some parameter u
    Cp_example = [1.0, 2.0, 0.5]
    Cpp_example = [0.0, 1.0, -0.2]

    # Given tolerances
    spt_example = 0.01       # spatial tolerance
    angle_tol_example = 0.05 # angular tolerance in radians

    du_result = compute_parametric_tolerance_curve(Cp_example, Cpp_example, spt_example, angle_tol_example)
    print(f"Computed du: {du_result:.6f}")

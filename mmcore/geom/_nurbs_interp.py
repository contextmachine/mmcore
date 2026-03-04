import numpy as np

from scipy.linalg import lu_solve,lu_factor
from numpy.typing import NDArray
from mmcore.geom._nurbs_eval import _find_span_linear, compute_basis_function_derivatives_np, bspline_basis, \
    NURBSCurveTuple
from mmcore.geom._nurbs_knots import generate_knots
def compute_knot_vector(degree, num_points, params):
    """ Computes a knot vector from the parameter list using averaging method.

    Please refer to the Equation 9.8 on The NURBS Book (2nd Edition), pp.365 for details.

    :param degree: degree
    :type degree: int
    :param num_points: number of data points
    :type num_points: int
    :param params: list of parameters, :math:`\\overline{u}_{k}`
    :type params: list, tuple
    :return: knot vector
    :rtype: list
    """
    # Start knot vector
    kv = [0.0 for _ in range(degree + 1)]

    # Use averaging method (Eqn 9.8) to compute internal knots in the knot vector
    for i in range(num_points - degree - 1):
        temp_kv = (1.0 / degree) * sum([params[j] for j in range(i + 1, i + degree + 1)])
        kv.append(temp_kv)

    # End knot vector
    kv += [1.0 for _ in range(degree + 1)]

    return kv

def build_coefficient_matrix(points, degree, knotvector, params):
    num_points = len(points)
    # Set up coefficient matrix
    matrix_a = [[0.0 for _ in range(num_points)] for _ in range(num_points)]
    for i in range(num_points):
        span = _find_span_linear(degree, knotvector, num_points, params[i])

        matrix_a[i][span - degree: span + 1] = compute_basis_function_derivatives_np(
            degree, knotvector, span, params[i],0
        )[0]
    # Return coefficient matrix
    return matrix_a
def compute_params_curve(points, centripetal=False):
    """
    :param points: data points
    :type points: list, tuple
    :param centripetal: activates centripetal parametrization method
    :type centripetal: bool
    :return: parameter array, :math:`\\overline{u}_{k}`
    :rtype: list
    """



    num_points = len(points)

    # Calculate chord lengths
    cds = [0.0 for _ in range(num_points + 1)]
    cds[-1] = 1.0
    for i in range(1, num_points):


        distance = np.linalg.norm(points[i]-points[i - 1])
        cds[i] = np.sqrt(distance) if centripetal else distance

    # Find the total chord length
    d = np.sum(cds[1:-1])

    # Divide individual chord lengths by the total chord length
    uk = [0.0 for _ in range(num_points)]
    for i in range(num_points):
        uk[i] = np.sum(cds[0:i + 1]) / d

    return uk,d

from mmcore.numeric.vectors import norm
def _remove_adjacent_duplicates(points, tol=1e-5):
    """
    Remove adjacent duplicates (or almost duplicates) from an array of points.

    Parameters:
        points (np.ndarray): A 2D numpy array of shape (n_points, 3) where each row is [x, y, z].
        tol (float): Tolerance for comparing points. Two points are considered equal if
                     the Euclidean distance between them is <= spt.

    Returns:
        np.ndarray: The array of points with consecutive duplicates removed.
    """
    if points.shape[0] == 0:
        return points

    # Compute the Euclidean distance between consecutive points.
    # np.diff(points, axis=0) gives an array of differences between consecutive rows.
    diffs = np.array(norm(np.diff(points, axis=0)))

    # Create a boolean mask:
    # Always keep the first point.
    # For each subsequent point, keep it only if the distance from the previous point is > spt.
    mask = np.concatenate(([True], diffs > tol))

    return points[mask]

from typing import Literal
def interpolate_curve(points, degree,  use_centripetal=False, method:Literal['lstsq','lu']='lstsq',remove_duplicates:bool=False,tol=1e-6,params=None):
    """ Curve interpolation through the data points.

    Please refer to Algorithm A9.1 on The NURBS Book (2nd Edition), pp.369-370 for details.

    """
    # Keyword arguments

    if remove_duplicates:
        points=_remove_adjacent_duplicates(points,tol)
    # Number of control points
    num_points = len(points)

    # Get uk
    if params is None:

        uk,total_chord_length = compute_params_curve(points, use_centripetal)
    else:
        uk = params


    # Compute knot vector
    kv = compute_knot_vector(degree, num_points, uk)

    # Do global interpolation
    matrix_a = np.array(build_coefficient_matrix( points,degree, kv, uk))
    if method=='lstsq':
        control_points=np.linalg.lstsq(matrix_a, points)[0]
    else:
        control_points = lu_solve(lu_factor(matrix_a),points)
    #print(matrix_a)




    return np.array(control_points,dtype=float),np.array(kv)


import numpy as np


def chord_length_params(points):
    """
    Compute normalized chord-length parameters for a list of points.

    points: array-like, shape (n, dim)
    Returns: 1D numpy array of parameters in [0,1]
    """
    points = np.asarray(points)
    n = len(points)
    u = np.zeros(n)
    for i in range(1, n):
        u[i] = u[i - 1] + np.linalg.norm(points[i] - points[i - 1])
    if u[-1] == 0:
        return u
    return u / u[-1]


def _compute_knot_vector_v2(u, degree, n_ctrlpts):
    """
    Compute a knot vector based on the data parameter values u (from chord-length)
    using the standard knot averaging formula for interpolation, and—if more
    control points than data points are desired—insert extra (uniform) knots.

    Parameters:
        u         : 1D array of parameter values for the data points.
        degree    : degree of the B-spline curve.
        n_ctrlpts : desired number of control points.
                    For example, if you have n_data points and want extra degrees
                    of freedom, one common choice is: n_ctrlpts = n_data + (degree-1)

    Returns:
        knot_vector: 1D numpy array of length n_ctrlpts + degree + 1.
    """
    n_data = len(u)
    # First, compute the standard knot vector for n_data control points.
    U_std = np.zeros(n_data + degree + 1)
    U_std[:degree + 1] = u[0]
    U_std[-(degree + 1):] = u[-1]
    for j in range(1, n_data - degree):
        U_std[j + degree] = np.sum(u[j:j + degree]) / degree

    # If no augmentation is needed, return the standard vector.
    if n_ctrlpts == n_data:
        return U_std
    else:

        return generate_knots(n_ctrlpts,degree,(u[0],u[-1]) )


def construct_basis_matrix(u_vals, degree, knot_vector, n_ctrlpts):
    """
    Construct the B-spline basis matrix.

    Each row corresponds to one parameter value from u_vals and each column to a control point.
    """
    m = len(u_vals)
    B = np.zeros((m, n_ctrlpts))
    for i, u in enumerate(u_vals):
        for j in range(n_ctrlpts):
            B[i, j] = bspline_basis(j, degree, knot_vector, u)
    return B


def fair_interpolate_curve(points, degree, lambda_reg=1e-3)->tuple[NDArray[float],NDArray[float]]:
    """
    :param  points: array-like, shape (n_data, dim). These are the data (e.g. curve isoline positions)
    :param degree: degree of the B-spline curve (for example, 3 for a cubic curve)
    :param lambda_reg : regularization (fairness) weight. Increase to favor smoother (fairer)
                     curves at the expense of less exact interpolation of the interior points.

    :returns:  Tuple with control_points and knot_vector.
        control_points: array of shape (n_ctrlpts, dim) (the computed control net for the fair interpolant)
        knot_vector   : the knot vector (a 1D array of length n_ctrlpts + degree + 1)


    Given a set of points to be interpolated, compute a fair B-spline interpolation
    that (i) uses knot-averaging to parameterize the interpolation, (ii) augments the
    number of control points to allow extra freedom (e.g., n_ctrlpts = n_data + (degree-1)),
    and (iii) selects from the infinite interpolation solutions the one that minimizes
    a fairness penalty based on second finite differences (a proxy for curvature).

    The implementation enforces the end points exactly and solves for the free interior control points
    by minimizing the combined interpolation error (at the given parameter values) and a fairness term.
    The fairness conditions (in the form of second finite differences) are expressed as:

       d[i-1] - 2*d[i] + d[i+1] ≈ 0

    for the (free) control points.
    """
    points = np.asarray(points)
    n_data, dim = points.shape
    # For augmented interpolation (to allow for extra fairing freedom),
    # choose number of control points by adding (degree - 1) extra rows.
    n_ctrlpts = n_data + (degree - 1)

    # 1. Determine parameter values via chord-length.
    u_vals = chord_length_params(points)

    # 2. Compute the knot vector (augmented via uniform insertion) using knot-averaging.
    knot_vector = _compute_knot_vector_v2(u_vals,degree,n_ctrlpts)

    # 3. Build the B-spline basis matrix evaluated at the data parameters.
    B = construct_basis_matrix(u_vals, degree, knot_vector, n_ctrlpts)

    # 4. Enforce interpolation at the endpoints exactly.
    #    Fix control point d0 and d_end.
    d0 = points[0]
    d_end = points[-1]

    # The remaining free unknown control points are indices 1 ... n_ctrlpts-2.
    N_free = n_ctrlpts - 2
    # For each interpolation condition (for each data point), the condition is:
    #    B[i,0]*d0 + sum_{j=1}^{n_ctrlpts-2} B[i,j]*x_{j-1} + B[i,n_ctrlpts-1]*d_end = points[i]
    # Isolate the unknowns:
    A_interp = B[:, 1:-1]  # shape: (n_data, N_free)
    b_interp = points - (B[:, [0]] * d0 + B[:, [-1]] * d_end)  # shape: (n_data, dim)

    # 5. Build a fairness (smoothness) term based on second finite differences.
    #    For a control net d0, d1, ..., d_{n_ctrlpts-1}, the second difference at index i is:
    #         d[i-1] - 2*d[i] + d[i+1]
    #    We enforce these to be (nearly) zero.
    #    In our free unknowns (x = [d1, d2, …, d_{n_ctrlpts-2}]), write equations:
    #      For i = 1 (first free point):  d0 - 2*d1 + d2 = 0   --> -2*x0 + x1 = -d0.
    #      For i = 2,..., n_ctrlpts-3 (interior free points):  d[i-1] - 2*d[i] + d[i+1] = 0.
    #      For i = n_ctrlpts-2 (last free point):  d[n_ctrlpts-3] - 2*d[n_ctrlpts-2] + d_end = 0
    #         --> x[-2] - 2*x[-1] = -d_end.
    N_eq = N_free  # We build one fairness equation per free unknown.
    F = np.zeros((N_eq, N_free))
    f_const = np.zeros((N_eq, dim))
    if N_free > 0:
        # Equation for the first free unknown:
        if N_free >= 2:
            F[0, 0] = -2
            F[0, 1] = 1
            f_const[0] = -d0
        else:
            # If there is only one free unknown, combine the two endpoints.
            F[0, 0] = -2
            f_const[0] = - (d0 + d_end)
        # Equations for interior free unknowns.
        for i in range(1, N_free - 1):
            F[i, i - 1] = 1
            F[i, i] = -2
            F[i, i + 1] = 1
            f_const[i] = 0  # no fixed contribution in the interior
        # Equation for the last free unknown:
        if N_free >= 2:
            F[N_free - 1, N_free - 2] = 1
            F[N_free - 1, N_free - 1] = -2
            f_const[N_free - 1] = -d_end

    # 6. Combine the interpolation conditions and fairness regularization into one augmented system.
    sqrt_lambda = np.sqrt(lambda_reg)
    A_aug = np.vstack([A_interp, sqrt_lambda * F])
    b_aug = np.vstack([b_interp, sqrt_lambda * f_const])

    # Solve the resulting least-squares problem for the free control points.
    # (Since the interpolation is underdetermined, the fairness term selects a unique solution.)
    x, residuals, rank, s = np.linalg.lstsq(A_aug, b_aug, rcond=None)

    # 7. Reassemble the full control net.
    control_points = np.vstack([d0, x, d_end])

    return control_points, knot_vector
def interpolate_nurbs_curve(points, degree,  use_centripetal=False,rational=False,**kwargs):
    points = np.unique(points, axis=0)
    points=np.array(points)
    if len(points)<=2 or degree==1:

        return NURBSCurveTuple(order=degree+1, knot=generate_knots(len(points),degree=1), control_points=np.array(points), weights=np.ones_like(points[...,0]))
    cp,kv=interpolate_curve(points, degree, use_centripetal=use_centripetal,**kwargs)
    cp=np.array(cp)
    if rational:
        cp,w=from_homogeneous_1d(cp)
    else:
        w=np.ones_like(cp[:,0])
    return NURBSCurveTuple(order=degree+1, knot=np.array(kv), control_points=cp, weights=w)
from mmcore.geom.nurbs import ders_basis_funs as _ders_basis_funs
from mmcore.geom.nurbs import find_span as _find_span
from mmcore.geom._nurbs_eval import from_homogeneous_1d,to_homogeneous_1d
import numpy as np
from scipy.special import comb, factorial
from typing import NamedTuple, List, Optional
from numpy.typing import NDArray


class RationalBezierCurve(NamedTuple):
    """
    Represents a Rational Bezier Curve of arbitrary degree.

    Attributes:
        degree: The degree of the curve (p).
        control_points: Homogeneous control points (N, 4) where N = p + 1.
                        Format: [wx, wy, wz, w]
    """
    degree: int
    control_points: NDArray[np.float64]


def get_bezier_derivative_matrix(degree: int,
                                 num_derivs_start: int,
                                 num_derivs_end: int) -> NDArray[np.float64]:
    """
    Constructs the linear system matrix mapping Bézier control points to
    endpoint derivatives.

    The system is M * P = D, where P are control points and D are derivatives.
    Returns M of shape (num_derivs_start + num_derivs_end, degree + 1).
    """
    n_constraints = num_derivs_start + num_derivs_end
    n_points = degree + 1

    matrix = np.zeros((n_constraints, n_points))

    # Helper to compute coefficient
    # C^(k)(0) = n!/(n-k)! * sum_{j=0}^k (-1)^(k-j) * binom(k,j) * P_j
    def get_coeff(d, j, deg):
        # P_j coefficient for k-th derivative
        # Valid only if 0 <= j <= d
        if j > d or j < 0:
            return 0.0

        # Falling factorial: n! / (n-d)!
        scaling = factorial(deg) / factorial(deg - d)
        binom_part = comb(d, j) * ((-1) ** (d - j))
        return scaling * binom_part

    # 1. Fill rows for Start (u=0)
    for k in range(num_derivs_start):
        # The k-th derivative only depends on the first k+1 control points (P_0 ... P_k)
        for j in range(k + 1):
            if j < n_points:
                matrix[k, j] = get_coeff(k, j, degree)

    # 2. Fill rows for End (u=1)
    # We use the symmetry property:
    # The curve reversed is Q(t) = C(1-t).
    # Q_i = P_{n-i}.
    # Derivatives at u=1 correspond to derivatives of Q at t=0 (with alternating signs).
    # C^(k)(1) = (-1)^k * Q^(k)(0)
    # Q^(k)(0) = Sum [ Coeff(k, j) * Q_j ] = Sum [ Coeff(k, j) * P_{n-j} ]
    # So: C^(k)(1) = (-1)^k * Sum_{j=0}^k [ Coeff(k, j) * P_{n-j} ]

    for k in range(num_derivs_end):
        row_idx = num_derivs_start + k
        sign = (-1) ** k

        for j in range(k + 1):
            # We are targeting P_{n-j}
            p_idx = degree - j
            if p_idx >= 0:
                matrix[row_idx, p_idx] = sign * get_coeff(k, j, degree)

    return matrix


def compute_homogeneous_derivatives(c_derivs: NDArray[np.float64],
                                    w_derivs: NDArray[np.float64]) -> NDArray[np.float64]:
    """
    Computes derivatives of the homogeneous vector H(u) = w(u) * C(u)
    using the General Leibniz Rule.

    Args:
        c_derivs: Array (K, D) of Euclidean derivatives [C, C', C'', ...]
        w_derivs: Array (K,) of Weight derivatives [w, w', w'', ...]

    Returns:
        h_derivs: Array (K, D) of Homogeneous derivatives.
    """
    k_max = c_derivs.shape[0]
    dim = c_derivs.shape[1]
    h_derivs = np.zeros((k_max, dim))

    for n in range(k_max):
        # H^(n) = Sum_{i=0}^n binom(n, i) * w^(i) * C^(n-i)
        vec_sum = np.zeros(dim)
        for i in range(n + 1):
            # w_derivs[i] is w^(i)
            # c_derivs[n-i] is C^(n-i)
            weight_term = w_derivs[i]
            vec_term = c_derivs[n - i]
            binomial = comb(n, i)

            vec_sum += binomial * weight_term * vec_term

        h_derivs[n] = vec_sum

    return h_derivs


def generalized_rational_hermite(
        start_derivs: NDArray[np.float64],
        end_derivs: NDArray[np.float64],
        degree: Optional[int] = None,
        start_weights: Optional[NDArray[np.float64]] = None,
        end_weights: Optional[NDArray[np.float64]] = None
) -> np.ndarray:
    """
    Constructs a Rational Bézier curve matching the provided Euclidean
    derivatives and Weight derivatives at the endpoints.

    Strategy A: Matches higher derivatives exactly.

    Args:
        start_derivs: (K1, 3) array of [Pos, Vel, Acc...] at u=0.
        end_derivs: (K2, 3) array of [Pos, Vel, Acc...] at u=1.
        degree: Target degree. If None, inferred as (K1 + K2 - 1).
        start_weights: (K1,) array of weight derivs at u=0. Defaults to [1, 0, 0...].
        end_weights: (K2,) array of weight derivs at u=1. Defaults to [1, 0, 0...].

    Returns:
        NDArray with homogeneous control points.
    """
    start_derivs = np.asarray(start_derivs)
    end_derivs = np.asarray(end_derivs)

    k_start = len(start_derivs)
    k_end = len(end_derivs)

    # 1. Determine Degree
    if degree is None:
        degree = k_start + k_end - 1

    # 2. Handle Weights Defaults (Standard Polynomial -> Rational Identity)
    # Default: w(0)=1, w'(0)=0... implies w(t) approx 1.
    if start_weights is None:
        start_weights = np.zeros(k_start)
        start_weights[0] = 1.0
    if end_weights is None:
        end_weights = np.zeros(k_end)
        end_weights[0] = 1.0

    # Ensure shapes match
    if len(start_weights) != k_start or len(end_weights) != k_end:
        raise ValueError("Weight derivative arrays must match spatial derivative array lengths.")

    # 3. Build the Linear System Matrix (M)
    # We use the same matrix M for solving Weights and Homogeneous Pos.
    # M shape: (Constraints, DoF) -> (k_start + k_end, degree + 1)
    M = get_bezier_derivative_matrix(degree, k_start, k_end)

    # Check if system is solvable
    n_constraints = M.shape[0]
    n_dof = M.shape[1]

    if n_constraints > n_dof:
        raise ValueError(f"Overconstrained: {n_constraints} constraints for degree {degree} ({n_dof} DoFs).")

    # 4. Solve for Weight Control Points (w_cp)
    # Vector of targets: [w^(0)(0), ..., w^(k)(0), w^(0)(1), ..., w^(k)(1)]
    rhs_weights = np.concatenate([start_weights, end_weights])

    # Solve M * w_cp = rhs_weights
    # If underdetermined, lstsq finds min-norm solution (though usually n_constraints == n_dof)
    w_cp, _, _, _ = np.linalg.lstsq(M, rhs_weights, rcond=None)

    # 5. Compute Homogeneous Derivatives H^(k)
    # We need H derivatives to solve for H control points.
    # H(u) = w(u) * C(u)
    h_start = compute_homogeneous_derivatives(start_derivs, start_weights)
    h_end = compute_homogeneous_derivatives(end_derivs, end_weights)

    # 6. Solve for Homogeneous Vector Control Points (H_cp)
    # We solve for x, y, z components independently or vectorized
    rhs_h = np.vstack([h_start, h_end])  # Shape (Constraints, 3)

    # Solve M * H_cp = rhs_h
    h_cp_vecs, _, _, _ = np.linalg.lstsq(M, rhs_h, rcond=None)

    # 7. Assemble Final Homogeneous Control Points
    # Structure: [wx, wy, wz, w]
    # w_cp is (N,), h_cp_vecs is (N, 3)

    # Reshape w_cp for broadcasting
    w_cp_col = w_cp.reshape(-1, 1)

    # Combine
    control_points_homogeneous = np.hstack([h_cp_vecs, w_cp_col])

    return control_points_homogeneous


import numpy as np
from scipy.linalg import solve
from typing import NamedTuple, Tuple
from numpy.typing import NDArray


def find_span(n: int, p: int, u: float, U: NDArray[np.float64]) -> int:
    """
    Finds the knot span index for a given parameter u.
    Based on Algorithm A2.1 from 'The NURBS Book'.
    """
    if u >= U[n + 1]:
        return n

    # Binary search
    low = p
    high = n + 1
    mid = (low + high) // 2

    while (u < U[mid]) or (u >= U[mid + 1]):
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2

    return mid



def ders_basis_funs(span_i: int, u: float, p: int, U: NDArray[np.float64], n_ders: int = 1) -> NDArray[np.float64]:
    """
    Computes non-zero basis functions and their derivatives.
    Returns a 2D array of shape (n_ders + 1, p + 1).
    Row 0 contains function values, Row 1 contains 1st derivatives.
    Based on Algorithm A2.3 from 'The NURBS Book'.
    """

    ders = np.zeros((n_ders + 1, p + 1))
    ndu = np.zeros((p + 1, p + 1))
    left = np.zeros(p + 1)
    right = np.zeros(p + 1)

    ndu[0, 0] = 1.0

    # Compute basis functions (and store terms for derivatives)
    for j in range(1, p + 1):
        left[j] = u - U[span_i + 1 - j]
        right[j] = U[span_i + j] - u
        saved = 0.0
        for r in range(j):
            # Lower triangle
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            # Upper triangle
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved

    # Load the basis functions
    for j in range(p + 1):
        ders[0, j] = ndu[j, p]

    # Compute derivatives
    a = np.zeros((2, p + 1))  # Only need rows 0 and 1 for recursion locally
    for r in range(0, p + 1):
        s1 = 0
        s2 = 1
        a[0, 0] = 1.0

        # Loop to compute k-th derivative
        for k in range(1, n_ders + 1):
            d = 0.0
            rk = r - k
            pk = p - k

            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]

            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if (r - 1) <= pk else p - r

            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]

            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]

            ders[k, r] = d

            # Swap rows
            j = s1
            s1 = s2
            s2 = j

    # Multiply by correct factors for derivatives
    r = p
    for k in range(1, n_ders + 1):
        for j in range(p + 1):
            ders[k, j] *= r
        r *= (p - k)

    return ders



def generate_hermite_knots(params: NDArray[np.float64], p: int) -> NDArray[np.float64]:
    """
    Generates a knot vector suitable for Hermite interpolation.

    For degree p=3 (Cubic), this generates the standard Hermite knot structure:
    [u0, u0, u0, u0, u1, u1, u2, u2, ..., un, un, un, un]

    For general p, it ensures the knot vector length matches the required
    Control Points (2 * num_points) by distributing internal knots.
    """
    n = len(params)
    num_control_points = 2 * n

    # Required knot vector length = num_cp + p + 1
    m = num_control_points + p + 1

    knots = np.zeros(m)

    # Fill ends with multiplicity p+1
    # Start knots
    knots[0: p + 1] = params[0]
    # End knots
    knots[m - (p + 1): m] = params[-1]

    # We need to fill the internal knots.
    # Indices to fill: from (p+1) to (m - p - 2) inclusive.
    # Count of internal slots to fill:
    num_internal_slots = (m - (p + 1)) - (p + 1)

    # Internal parameter values to source from: params[1:-1]
    internal_params = params[1:-1]

    if len(internal_params) == 0:
        # Special case: only 2 points. No internal knots needed usually,
        # but if p is high, we might have excess slots (though rare for p=3).
        pass
    elif num_internal_slots > 0:
        # Distribute internal params into the slots.
        # For p=3, num_internal_slots is exactly 2 * len(internal_params).
        # We repeat each internal param 'ratio' times.

        # To handle generic p (where ratio might not be integer), we use linspace
        # indices logic to pick nearest neighbor parameters.

        indices = np.linspace(0, len(internal_params) - 1e-9, num_internal_slots)
        indices = indices.astype(int)

        knots[p + 1: m - (p + 1)] = internal_params[indices]

    return knots


def hermite_interpolate_nurbs(
        points: NDArray[np.float64],
        derivatives: NDArray[np.float64],
        params: NDArray[np.float64],
        degree: int = 3
) -> NURBSCurveTuple:
    """
    Constructs a NURBS curve (B-Spline) using Hermite Interpolation.

    Args:
        points: Array of shape (N, D) containing data points.
        derivatives: Array of shape (N, D) containing 1st derivatives at points.
        params: Array of shape (N,) containing strictly increasing parameters.
        degree: Target degree of the curve (usually 3 for Cubic Hermite).

    Returns:
        NURBSCurveTuple containing knots, control points, and weights.
    """
    points = np.asarray(points)
    derivatives = np.asarray(derivatives)
    params = np.asarray(params)

    n_points, dim = points.shape

    if len(params) != n_points or len(derivatives) != n_points:
        raise ValueError("Points, derivatives, and parameters must have the same length.")

    # 1. Calculate numbers
    # In Hermite interpolation, we have 2 constraints per point (Pos, Deriv).
    # So we need 2 * n_points Control Points.
    num_cp = 2 * n_points

    # 2. Generate Knot Vector
    knots = generate_hermite_knots(params, degree)

    # 3. Build Linear System (Matrix A and RHS B)
    # System size: (2*n_points) x (2*n_points)
    # Constraints:
    # Row 2*i     : C(u_i) = P_i
    # Row 2*i + 1 : C'(u_i) = D_i

    A = np.zeros((num_cp, num_cp))
    B = np.zeros((num_cp, dim))

    # We iterate over every data point to fill the matrix
    for i in range(n_points):
        u = params[i]

        # Find the knot span index for this parameter
        # Note: For the very last parameter, find_span returns n (upper limit),
        # but we need the index corresponding to the valid basis functions range.
        # The helper handles u_max usually, but let's ensure robustness for the last knot.
        span = find_span(num_cp - 1, degree, u, knots)

        # Standard adjustment: if u is exactly the upper knot, find_span returns the index
        # of the next span. We need the span that contains the non-zero basis functions.
        if u == knots[num_cp]:
            span = num_cp - 1

        # Compute Basis functions and 1st Derivatives
        # Returns shape (2, p+1) -> [Values, Derivatives]
        ders = ders_basis_funs(span, u, degree, knots, n_ders=1)

        # Fill Matrix A
        # The basis functions returned correspond to control points: span - p, ..., span
        col_start_idx = span - degree

        for j in range(degree + 1):
            col_idx = col_start_idx + j

            # Position constraint (Even row)
            A[2 * i, col_idx] = ders[0, j]

            # Derivative constraint (Odd row)
            A[2 * i + 1, col_idx] = ders[1, j]

        # Fill RHS B
        B[2 * i, :] = points[i]
        B[2 * i + 1, :] = derivatives[i]

    # 4. Solve for Control Points
    # Solve A * CP = B
    control_points = solve(A, B)

    # 5. Weights
    # For standard polynomial interpolation, weights are 1.0
    weights = np.ones(num_cp, dtype=np.float64)

    return NURBSCurveTuple(
        order=degree + 1,
        knot=knots,
        control_points=control_points,
        weights=weights
    )

# Example Usage
if __name__ == "__main__":
    # Define a Quintic (Degree 5) problem
    # We need 3 constraints at each end: Pos, Vel, Acc

    # Start: Origin, moving X+, accelerating Y+
    import time
    s=time.time()
    start_D = np.array([
        [0.0, 0.0, 0.0],  # Pos
        [10.0, 0.0, 0.0],  # Vel
        [0.0, 10.0, 0.0]  # Acc
    ])

    # End: (10,10,0), moving Y+, accelerating X-
    end_D = np.array([
        [10.0, 10.0, 0.0],  # Pos
        [0.0, 10.0, 0.0],  # Vel
        [-10.0, 0.0, 0.0]  # Acc
    ])

    # Create curve
    # Note: We don't specify weights, so it assumes standard polynomial (w=1)
    curve = generalized_rational_hermite(start_D, end_D, degree=5)

    np.set_printoptions(precision=3, suppress=True)
    print(f"Generated Rational Bezier of Degree {curve.shape[0]-1}")
    print("Homogeneous Control Points (wx, wy, wz, w):")
    print(curve)

    # Verification: Weights should be all 1.0 (within float error) if inputs imply polynomial
    print("\nWeights:", curve[:, 3])
    print(time.time()-s)
import math

import numpy as np

from collections import namedtuple
from typing import TypedDict



# ======================================================================
# Namedtuple definitions for NURBS surface and curve representations.
# ======================================================================
NURBSSurfaceTuple = namedtuple('NURBSSurfaceTuple', ['order_u', 'order_v', 'knot_u', 'knot_v', 'control_points', 'weights'])
NURBSCurveTuple  = namedtuple('NURBSCurveTuple',  ['order', 'knot', 'control_points', 'weights'])
BSplineCurveTuple = namedtuple('BSplineCurveTuple', ['order', 'knot', 'control_points'])

def join_weights(surf:NURBSSurfaceTuple):
    ptsw = np.zeros((*surf.control_points.shape[:-1], 4))

    ptsw[..., :-1] = surf.control_points
    ptsw[..., -1] = surf.weights
    return ptsw





np.set_printoptions(suppress=True)



def compute_left_right_arrays(degree, knot, knot_vector, span):
    """
    Compute the arrays of distances from the parameter value to neighboring knots.

    In the literature these are called the left and right differences. For each index j,
    the arrays are defined as:

       left[j]  = knot - knot_vector[span + 1 - j]
       right[j] = knot_vector[span + j] - knot

    Args:
        degree (int): The degree p of the basis functions.
        knot (float): The parameter value u at which to evaluate.
        knot_vector (list or tuple): The knot vector U.
        span (int): The knot span index.

    Returns:
        tuple: (left, right), each a list of length (degree+1).
    """
    left = [None] * (degree + 1)

    right = [None] * (degree + 1)
    #print(degree, knot, knot_vector, span)
    ixs1 = []
    ixs2 = []
    for j in range(1, degree + 1):
        ixs1.append(span + 1 - j)
        ixs2.append(span + j)

        left[j] = knot - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - knot
    #print(ixs1)
    #print(ixs2)
    return left, right


def compute_ndu_row(j, left, right, ndu):
    """
    Compute one row of the 'ndu' table.

    For a given level j (1 <= j <= degree), the ndu table is filled according to
    a convolution–like recurrence:

       ndu[j][r] = right[r+1] + left[j - r]     for r in 0 <= r < j
       temp = ndu[r][j-1] / ndu[j][r]
       ndu[r][j] = saved + right[r+1] * temp
       saved = left[j - r] * temp

    Finally, ndu[j][j] is set to the last saved value.

    Args:
        j (int): The current row (or level) in the recurrence.
        left (list of float): Precomputed left differences.
        right (list of float): Precomputed right differences.
        ndu (list of list of float): The working table, modified in place.
    """
    saved = 0.0
    for r in range(j):
        ndu[j][r] = right[r + 1] + left[j - r]
        temp = ndu[r][j - 1] / ndu[j][r]
        ndu[r][j] = saved + right[r + 1] * temp
        saved = left[j - r] * temp
    ndu[j][j] = saved


def compute_ndu(degree, knot, knot_vector, span):
    """
    Build the complete 'ndu' table for evaluating the B-spline basis functions.

    This function first computes the left/right arrays and then fills
    the ndu table level by level.

    Args:
        degree (int): The degree p of the basis functions.
        knot (float): The parameter value u at which to evaluate.
        knot_vector (list or tuple): The knot vector U.
        span (int): The knot span index.

    Returns:
        tuple: (ndu, left, right)
            - ndu: A 2D list of size (degree+1) x (degree+1) that holds intermediate values.
            - left: The left differences.
            - right: The right differences.
    """
    # Allocate the table with zeros and set the initial condition
    ndu = [[0.0] * (degree + 1) for _ in range(degree + 1)]
    ndu[0][0] = 1.0

    left, right = compute_left_right_arrays(degree, knot, knot_vector, span)
    for j in range(1, degree + 1):
        compute_ndu_row(j, left, right, ndu)
    return ndu, left, right


def compute_inner_loop_bounds(r, k, degree):
    """
    Compute the inner summation bounds for the derivative accumulation.

    In the derivative algorithm the inner loop runs from j1 to j2.
    These bounds are computed as follows:

         rk = r - k
         pk = degree - k
         j1 = 1 if rk >= -1 else -rk
         j2 = k - 1 if (r - 1) <= pk else degree - r

    Args:
        r (int): The basis function index.
        k (int): The current derivative order (>= 1).
        degree (int): The degree p of the basis functions.

    Returns:
        tuple: (j1, j2) the lower and upper bounds for the inner loop index.
    """
    rk = r - k
    pk = degree - k
    j1 = 1 if rk >= -1 else -rk
    j2 = k - 1 if (r - 1) <= pk else degree - r
    return j1, j2


def compute_derivative_coefficients_for_r(r, ndu, degree, order):
    """
    Compute the derivative coefficients for one basis function (index r).

    The 0th derivative is simply the basis function value, and higher derivatives
    are computed by recursively “convolving” the lower–order contributions.

    Args:
        r (int): The basis function index (0 <= r <= degree).
        ndu (list of list of float): The table computed in compute_ndu.
        degree (int): The degree p of the basis functions.
        order (int): The maximum derivative order to compute.

    Returns:
        list: A list of length (order+1) where the k-th element is the k-th derivative
              coefficient for basis function r.
    """
    # Initialize the list of derivative coefficients; the 0th derivative is directly from ndu.
    d_coeffs = [0.0] * (order + 1)
    d_coeffs[0] = ndu[r][degree]

    # Temporary storage in two alternating rows.
    a = [[0.0] * (order + 1) for _ in range(2)]
    a[0][0] = 1.0
    s1 = 0  # current row index in a
    s2 = 1  # next row index in a

    # Loop over derivative orders k = 1, 2, ..., order
    for k in range(1, order + 1):
        d = 0.0
        rk = r - k
        pk = degree - k

        # First term in the recurrence, if available.
        if r >= k:
            a[s2][0] = a[s1][0] / ndu[pk + 1][rk]
            d += a[s2][0] * ndu[rk][pk]

        # Compute inner summation contributions.
        j1, j2 = compute_inner_loop_bounds(r, k, degree)
        for j in range(j1, j2 + 1):
            a[s2][j] = (a[s1][j] - a[s1][j - 1]) / ndu[pk + 1][rk + j]
            d += a[s2][j] * ndu[rk + j][pk]

        # Final term in the recurrence, if available.
        if r <= pk:
            a[s2][k] = -a[s1][k - 1] / ndu[pk + 1][r]
            d += a[s2][k] * ndu[r][pk]

        d_coeffs[k] = d
        # Swap the temporary rows for the next iteration.
        s1, s2 = s2, s1

    return d_coeffs


def compute_basis_function_derivatives_from_ndu(ndu, degree, order):
    """
    Build the full table of (unscaled) derivative coefficients from the ndu table.

    For each basis function (index r from 0 to degree) the derivative coefficients
    (orders 0 through order) are computed.

    Args:
        ndu (list of list of float): The table computed in compute_ndu.
        degree (int): The degree p of the basis functions.
        order (int): The maximum derivative order to compute.

    Returns:
        list of list of float: A 2D list 'ders' where ders[k][r] is the k-th derivative
                               for basis function index r.
    """
    ders = [[0.0] * (degree + 1) for _ in range(order + 1)]
    for r in range(degree + 1):
        d_coeffs = compute_derivative_coefficients_for_r(r, ndu, degree, order)
        for k in range(order + 1):
            ders[k][r] = d_coeffs[k]
    return ders


def apply_factorial_scaling(ders, degree, order):
    """
    Scale the derivative coefficients by the appropriate factorial-like factors.

    The scaling multiplies the k-th derivative by
         p * (p-1) * ... * (p - k + 1)
    so that the results agree with the standard mathematical definition.

    Args:
        ders (list of list of float): The (unscaled) derivatives.
        degree (int): The degree p of the basis functions.
        order (int): The maximum derivative order computed.

    Returns:
        list of list of float: The scaled derivatives.
    """
    factor = float(degree)
    for k in range(1, order + 1):
        for j in range(degree + 1):
            ders[k][j] *= factor
        factor *= degree - k
    return ders


def basis_function_ders(degree, knot_vector, span, knot, order):
    """
    Compute the derivatives of B-spline (or NURBS) basis functions.

    This high-level function organizes the work into several intuitive steps:

      1. Build the ndu table (and compute left/right differences).
      2. Extract the (unscaled) derivative coefficients via a convolution–like recurrence.
      3. Apply the necessary factorial scaling to obtain the true derivative values.

    Args:
        degree (int): The degree p of the basis functions.
        knot_vector (list or tuple): The knot vector U.
        span (int): The knot span index.
        knot (float): The parameter value u at which to evaluate.
        order (int): The maximum derivative order to compute.

    Returns:
        list of list of float: A 2D list 'ders' where ders[k][j] is the k-th derivative
                               of the j-th basis function.
    """
    ndu, left, right = compute_ndu(degree, knot, knot_vector, span)
    ders = compute_basis_function_derivatives_from_ndu(ndu, degree, order)
    ders = apply_factorial_scaling(ders, degree, order)
    return ders

def compute_basis_function_derivatives_np(degree, knot_vector, span, knot, order):
    """
    Compute the derivatives of B-spline (or NURBS) basis functions using numpy for efficiency.
    Args:
        degree (int): The degree p of the basis functions.
        knot_vector (array-like): The knot vector U.
        span (int): The knot span index.
        knot (float): The parameter value u at which to evaluate.
        order (int): The maximum derivative order to compute.
    Returns:
        np.ndarray: A 2D array 'ders' of shape (order+1, degree+1) where ders[k, j]
                    is the k-th derivative of the j-th basis function.
    """
    knot_vector = np.asarray(knot_vector, dtype=float)
    # Precompute left/right arrays using vectorized slicing.
    left = np.empty(degree + 1, dtype=float)
    right = np.empty(degree + 1, dtype=float)
    left[0] = 0.0
    right[0] = 0.0
    j_arr = np.arange(1, degree + 1)
    left[1:] = knot - knot_vector[span + 1 - j_arr]
    right[1:] = knot_vector[span + j_arr] - knot
    # Build the 'ndu' table.
    ndu = np.zeros((degree + 1, degree + 1), dtype=float)
    ndu[0, 0] = 1.0
    for j in range(1, degree + 1):
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            temp = ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * temp
            saved = left[j - r] * temp
        ndu[j, j] = saved
    # Compute unscaled derivative coefficients.
    ders = np.zeros((order + 1, degree + 1), dtype=float)
    for r in range(degree + 1):
        d_coeffs = np.zeros(order + 1, dtype=float)
        d_coeffs[0] = ndu[r, degree]
        a = np.zeros((2, order + 1), dtype=float)
        a[0, 0] = 1.0
        s1 = 0  # current row in temporary array 'a'
        s2 = 1  # next row in 'a'
        for k in range(1, order + 1):
            d = 0.0
            rk = r - k
            pk = degree - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if (r - 1) <= pk else degree - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            d_coeffs[k] = d
            s1, s2 = s2, s1  # swap rows
        ders[:, r] = d_coeffs
    # Factorial scaling: scale k-th derivative by degree*(degree-1)*...*(degree-k+1)
    scales = np.empty(order + 1, dtype=float)
    scales[0] = 1.0
    for k in range(1, order + 1):
        scales[k] = scales[k - 1] * (degree - k + 1)
    ders[1:, :] *= scales[1:, np.newaxis]
    return ders
def _find_span_linear(degree, knot_vector, num_ctrlpts, knot, **kwargs):
    span = degree + 1  # knot span index starts from zero
    while span < num_ctrlpts and knot_vector[span] <= knot:
        span += 1
    return span - 1


def evaluate_nurbs_curve(curve, u, d_order=2):
    """
    Evaluate a rational NURBS curve at parameter u.
    Returns a dictionary with keys:
      'C'  : the evaluated point,
      'C1' : the first derivative,
      'C2' : the second derivative.
    """
    p = curve.order - 1
    n = len(curve.control_points)
    U = curve.knot[:]  # assume knot vector is a list or numpy array
    span = _find_span_linear(p, U, n, u)
    d = min(d_order, p)

    # Compute basis functions and their derivatives.
    # Assumes existence of a function 'compute_basis_function_derivatives_np'
    ders = np.array(compute_basis_function_derivatives_np(p, U, span, u, d))
    # ders has shape (d+1, p+1)

    dim = len(curve.control_points[0])
    # Allocate homogeneous derivatives d_hom[k] for k = 0, 1, ..., d.
    d_hom = [np.zeros(dim + 1) for _ in range(d + 1)]
    for k in range(d + 1):
        for j in range(p + 1):
            i = span - p + j
            P = np.array(curve.control_points[i])
            w = curve.weights[i]
            # Form the homogeneous coordinate [w*P, w]
            H = np.zeros(dim + 1)
            H[:dim] = P * w
            H[dim] = w
            d_hom[k] += ders[k, j] * H

    result = {}
    # Dehomogenize to get the point on the curve.
    C = d_hom[0][:dim] / d_hom[0][dim]
    result["C"] = C

    # First derivative.
    if d >= 1:
        C1 = (d_hom[1][:dim] - d_hom[1][dim] * C) / d_hom[0][dim]
        result["C1"] = C1
    else:
        result["C1"] = np.zeros(dim)

    # Second derivative.
    if d >= 2:
        C2 = ((d_hom[2][:dim] - d_hom[2][dim] * C) / d_hom[0][dim]
              - 2 * (d_hom[1][dim] / d_hom[0][dim]) * result["C1"])
        result["C2"] = C2
    else:
        result["C2"] = np.zeros(dim)

    return result


def evaluate_nurbs_curve_array(curve:NURBSCurveTuple, t, d_order=0):
    """
    Evaluate a NURBS curve (which may be rational) at parameter value t.
    d_order = 0 returns only the point; d_order = 1 returns [point, derivative].
    Works in any dimension.
    """

    #crv1=curve
    #crv=NURBS.Curve()
    #crv.degree=crv1.order-1
    #
    #crv.ctrlpts = crv1.control_points.tolist()
    #
    #crv.knotvector=crv1.knot.tolist()
    #crv.weights=crv1.weights.tolist()

    #return np.array(crv.derivatives(t,d_order))

    return np.array(list(evaluate_nurbs_curve(curve,t,d_order).values()))


def evaluate_nurbs_surface(surface, u, v, d_order=2):
    """
    Evaluate a rational NURBS surface at (u,v). Returns a dictionary SKL with keys:
      'S'   : the 3D (or n–dimensional) point,
      'Su'  : first derivative in u,
      'Sv'  : first derivative in v,
      'Suu' : second derivative in u,
      'Suv' : mixed second derivative,
      'Svv' : second derivative in v.
    """


    #print(surface, u, v)
    surface1 = surface
    p = surface1.order_u - 1
    q = surface1.order_v - 1
    nu = len(surface1.control_points)
    nv = len(surface1.control_points[0])
    U = surface1.knot_u[:]  # assume these are already lists/numpy arrays
    V = surface1.knot_v[:]
    span_u = _find_span_linear(p, U, nu, u)
    span_v = _find_span_linear(q, V, nv, v)
    #print(p, U, span_u, u, d_order)
    du = min(d_order, p)
    dv = min(d_order, q)
    ders_u = np.array(compute_basis_function_derivatives_np(p, U, span_u, u, du))
    #print(q, V, span_v, v, d_order)
    ders_v = np.array(compute_basis_function_derivatives_np(q, V, span_v, v, dv))
    #print("DU", ders_u)
    #print("DV", ders_v)

    SKL = {}
    dim = len(surface1.control_points[0][0])
    #print(surface)
    # Allocate and initialize homogeneous derivatives.
    d = [[np.zeros(dim + 1) for l in range(dv + 1)] for k in range(du + 1)]
    for k in range(du + 1):
        for l in range(dv + 1):
            d[k][l] = np.zeros(dim + 1)
    # Compute homogeneous surface derivatives d[k][l]
    for l in range(q + 1):
        temp = [np.zeros(dim + 1) for i in range(du + 1)]
        for k in range(p + 1):
            i_index = span_u - p + k
            j_index = span_v - q + l
            cp = np.array(surface1.control_points[i_index][j_index])
            w = surface1.weights[i_index, j_index]
            tmp = np.zeros(dim + 1)
            tmp[:dim] = cp
            tmp[dim] = w
            for i in range(du + 1):
                temp[i] += ders_u[i, k] * tmp
        for j in range(dv + 1):
            for i in range(du + 1):
                d[i][j] += ders_v[j, l] * temp[i]
    # Dehomogenize
    SKL["S"] = d[0][0][:dim] / d[0][0][dim]
    SKL["Su"] = np.zeros(dim)
    SKL["Sv"] = np.zeros(dim)
    SKL["Suu"] = np.zeros(dim)
    SKL["Suv"] = np.zeros(dim)
    SKL["Svv"] = np.zeros(dim)
    if du >= 1:
        Su = (d[1][0][:dim] - d[1][0][dim] * SKL["S"]) / d[0][0][dim]

        SKL["Su"] = Su

    if dv >= 1:
        Sv = (d[0][1][:dim] - d[0][1][dim] * SKL["S"]) / d[0][0][dim]
        SKL["Sv"] = Sv
    if du >= 2:
        Suu = (d[2][0][:dim] - d[2][0][dim] * SKL["S"]) / d[0][0][dim] - 2 * (d[1][0][dim] / d[0][0][dim]) * SKL["Su"]
        SKL["Suu"] = Suu

    if dv >= 2:

        Svv = (d[0][2][:dim] - d[0][2][dim] * SKL["S"]) / d[0][0][dim] - 2 * (d[0][1][dim] / d[0][0][dim]) * SKL["Sv"]

        SKL["Svv"] = Svv

    if du >= 2 or dv >= 2:
        Suv = (
            (d[1][1][:dim] - d[1][1][dim] * SKL["S"]) / d[0][0][dim]
            - (d[1][0][dim] / d[0][0][dim]) * SKL["Sv"]
            - (d[0][1][dim] / d[0][0][dim]) * SKL["Su"]
        )
        SKL["Suv"] = Suv
    #print(SKL)
    return SKL


def bspline_basis(j, L, t, knot):
    """
    Evaluate the j-th B-spline basis function of order L at parameter value t using Cox-de Boor.
    """
    if L == 1:
        if (knot[j] <= t < knot[j + 1]) or (t == knot[-1] and t == knot[j + 1]):
            return 1.0
        else:
            return 0.0
    else:
        denom1 = knot[j + L - 1] - knot[j]
        term1 = 0.0
        if denom1 != 0:
            term1 = (t - knot[j]) / denom1 * bspline_basis(j, L - 1, t, knot)
        denom2 = knot[j + L] - knot[j + 1]
        term2 = 0.0
        if denom2 != 0:
            term2 = (knot[j + L] - t) / denom2 * bspline_basis(j + 1, L - 1, t, knot)
        return term1 + term2


def evaluate_bspline_curve(curve:BSplineCurveTuple, t):
    """
    Evaluate a (non-rational) B-spline curve at parameter value t.
    """
    #'order', 'knot', 'control_points', 'weights'
    nc=NURBSCurveTuple(curve.order, curve.knot,curve.control_points, np.ones(curve.control_points.shape[:-1],dtype=float))
    return evaluate_nurbs_curve_array(nc, t, d_order=0)


def evaluate_bspline_curve_derivative(curve, t):
    """
    Evaluate the derivative of the B-spline curve at t using formula (48):
      S'(t) = (L-1) * sum_{j=1}^{n-1} (Q_j - Q_{j-1})/(s_{j+L-1} - s_j) * N_{j,L-1}(t)
    """
    nc = NURBSCurveTuple(curve.order, curve.knot, curve.control_points,
                         np.ones(curve.control_points.shape[:-1], dtype=float))
    return evaluate_nurbs_curve_array(nc, t, d_order=1)[1]

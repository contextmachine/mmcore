#!/usr/bin/env python3
"""
Exhaustive test suite for evaluate_nurbs_surface.
This file contains tests for:
  - planar B-spline (non–rational) surfaces,
  - Bezier surfaces,
  - rational (weighted) surfaces,
  - surfaces defined in higher (overdimensional) spaces,
  - surfaces with non–uniform weights,
  - singular (degenerate) surfaces,
  - canonical cases (cylinder, cone).

Data have been chosen from standard NURBS textbooks (e.g. Piegl & Tiller)
and from established libraries (e.g. OpenNURBS examples).
"""

import unittest
import numpy as np
import math
from collections import namedtuple
from geomdl import helpers

from mmcore.geom.bvh import NURBSCurveObject3D

# ---------------------------------------------------------------------------
# Minimal helpers module (only for degrees 1 and 2) used by evaluate_nurbs_surface.
# (In production the helpers module would be imported from your library.)
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# The implementation to be tested.
# (Normally you would import this from your module.)
# ---------------------------------------------------------------------------
NURBSSurface = namedtuple("NURBSSurface", ["order_u", "order_v", "knot_u", "knot_v", "control_points", "weights"])

NURBSCurve= namedtuple("NURBSCurve", ["order",  "knot", "control_points", "weights"])



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
    left = [None]*(degree+1)

    right = [None]*(degree+1)
    print(degree,knot,knot_vector,span)
    ixs1=[]
    ixs2 = []
    for j in range(1, degree + 1):
        ixs1.append(span + 1 - j)
        ixs2.append(span + j)

        left[j] = knot - knot_vector[span + 1 - j]
        right[j] = knot_vector[span + j] - knot
    print(ixs1)
    print(ixs2)
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
        factor *= (degree - k)
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

    def _find_span_linear(degree, knot_vector, num_ctrlpts, knot, **kwargs):
        span = degree + 1  # knot span index starts from zero
        while span < num_ctrlpts and knot_vector[span] <= knot:
            span += 1
        return span - 1

    print(surface, u, v)
    surface1 = surface
    p = surface1.order_u - 1
    q = surface1.order_v - 1
    nu = len(surface1.control_points)
    nv = len(surface1.control_points[0])
    U = surface1.knot_u[:]  # assume these are already lists/numpy arrays
    V = surface1.knot_v[:]
    span_u = _find_span_linear(p, U, nu, u)
    span_v = _find_span_linear(q, V, nv, v)
    print(p, U, span_u, u, d_order)
    du = min(d_order, p)
    dv = min(d_order, q)
    ders_u = np.array(basis_function_ders(p, U, span_u, u, du))
    print(q, V, span_v, v, d_order)
    ders_v = np.array(basis_function_ders(q, V, span_v, v, dv))
    print("DU", ders_u)
    print("DV", ders_v)

    SKL = {}
    dim = len(surface1.control_points[0][0])
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
    print(SKL)
    return SKL

def cylinder(radius,h):
    _C = 1 / math.sqrt(2)
    _CC = _C * 2
    rr=np.array([
        [[radius, 0., 0., radius],
         [radius, 0., h, radius]],
        [[_C, _C, 0., _C],
         [_C, _C, _CC, _C]],
        [[0., radius, 0., radius],
         [0., radius, h, radius]],
        [[-_C, _C, 0., _C],
         [-_C, _C, _CC, _C]],
        [[-radius, 0., 0., radius],
         [-radius, 0., h, radius]],
        [[-_C, -_C, 0., _C],
         [-_C, -_C, _CC, _C]],
        [[-0., -radius, 0., radius],
         [-0., -radius, h, radius]],
        [[_C, -_C, 0., _C],
         [_C, -_C, _CC, _C]],
        [[radius, 0., 0., radius],
         [radius, 0., h, radius]]])


    return NURBSSurface(3,2,  [0.  , 0.  , 0.  , 0.25, 0.25, 0.5 , 0.5 , 0.75, 0.75, 1.  , 1.  ,
       1.  ],[0.,0.,1.,1.],  rr[...,:-1].tolist(),  rr[...,-1])


def circle(radius):
    c=1/math.sqrt(2)

    ptsw=np.array([[radius,0,0,1],[radius,radius,0,c],[0,radius,0,1],[-radius,radius,0,c],[-radius,0,0,1],[-radius,-radius,0,c],[0,-radius,0,1],[radius,-radius,0,c],[radius,0,0,1]
                   ] )

    return NURBSCurve(3, [0., 0., 0., 1 / 4, 2 / 4, 3 / 4, 4 / 4, 4 / 4, 4 / 4],    ptsw[...,:-1].tolist(),    ptsw[..., -1])


# ---------------------------------------------------------------------------
# Test cases for evaluate_nurbs_surface.
# ---------------------------------------------------------------------------
class TestEvaluateNURBSSurface(unittest.TestCase):
    # Tolerance for numerical comparisons
    tol = 1e-5

    def test_bspline_planar(self):
        # A simple bilinear (order=2 in both u and v) B-spline surface.
        # Control net: 2x2 grid in the plane z=0.
        # P00=(0,0,0), P01=(0,1,0), P10=(1,0,0), P11=(1,1,0)
        ctrl_pts = [[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
        weights = np.array([[1.0, 1.0], [1.0, 1.0]])
        # For a linear (order=2) B-spline, knot vectors are [0,0,1,1].
        surf = NURBSSurface(order_u=2, order_v=2, knot_u=[0, 0, 1, 1], knot_v=[0, 0, 1, 1], control_points=ctrl_pts, weights=weights)
        # Evaluate at (u,v) = (0.3, 0.7)
        u, v = 0.3, 0.7
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)

        # For bilinear interpolation the result is:
        # S = ( (1-u)*(1-v)*P00 + (1-u)*v*P01 + u*(1-v)*P10 + u*v*P11 )
        # S = (0.7*0.3*P00 + 0.7*0.7*P01 + 0.3*0.3*P10 + 0.3*0.7*P11)
        #      = (0.21*P00 + 0.49*P01 + 0.09*P10 + 0.21*P11)
        # = ( (0.09+0.21, 0.49+0.21, 0) ) = (0.3, 0.7, 0)
        expected_S = np.array([0.3, 0.7, 0.0])
        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol))
        # First derivatives (analytically)
        expected_Su = np.array([1.0, 0.0, 0.0])
        expected_Sv = np.array([0.0, 1.0, 0.0])
        self.assertTrue(np.allclose(SKL["Su"], expected_Su, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Sv"], expected_Sv, rtol=self.tol))
        print("Su,Sv: OK")
        # Second derivatives are zero.
        expected_Suu = np.zeros(3)
        expected_Suv = np.zeros(3)
        expected_Svv = np.zeros(3)

        self.assertTrue(np.allclose(SKL["Suu"], expected_Suu, rtol=self.tol))
        print("Suu: OK")
        self.assertTrue(np.allclose(SKL["Suv"], expected_Suv, rtol=self.tol))
        print("Suv: OK")
        self.assertTrue(np.allclose(SKL["Svv"], expected_Svv, rtol=self.tol))
        print("Svv: OK")

    def test_bezier_patch(self):
        # A quadratic (order=3 in both directions) Bezier patch.
        # Control net: 3x3 grid.
        # Use control points that yield a non–planar patch.
        ctrl_pts = [
            [(0.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 2.0, 0.0)],
            [(1.0, 0.0, 1.0), (1.0, 1.0, 1.0), (1.0, 2.0, 1.0)],
            [(2.0, 0.0, 0.0), (2.0, 1.0, 0.0), (2.0, 2.0, 0.0)],
        ]
        weights = np.array([[1, 1, 1], [1, 1, 1], [1, 1, 1]])
        # For a Bezier patch, knot vectors are [0,0,0,1,1,1]
        surf = NURBSSurface(
            order_u=3, order_v=3, knot_u=[0, 0, 0, 1, 1, 1], knot_v=[0, 0, 0, 1, 1, 1], control_points=ctrl_pts, weights=weights
        )
        # Evaluate at (u,v) = (0.5,0.5)
        u, v = 0.5, 0.5
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)
        # For the given quadratic Bezier patch the (non–rational) evaluation yields:
        # S = (1,1,0.5)
        expected_S = np.array([1.0, 1.0, 0.5])
        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol))
        # Derivatives computed using standard Bezier formulas:
        expected_Su = np.array([2.0, 0.0, 0.0])
        expected_Sv = np.array([0.0, 2.0, 0.0])
        expected_Suu = np.array([0.0, 0.0, -4.0])
        expected_Suv = np.array([0.0, 0.0, 0.0])
        expected_Svv = np.array([0.0, 0.0, 0.0])
        self.assertTrue(np.allclose(SKL["Su"], expected_Su, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Sv"], expected_Sv, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suu"], expected_Suu, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suv"], expected_Suv, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Svv"], expected_Svv, rtol=self.tol))

    def test_weighted_bilinear(self):
        # A rational bilinear patch (order=2 in both directions) with non-uniform weights.
        # Control points same as test_bspline_planar but with different weights.

        ctrl_pts = [[(0.0, 0.0, 0.0), (0.0, 1.0, 0.0)], [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0)]]
        # Assign weights:
        # w00=1, w01=2, w10=3, w11=4.
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        surf = NURBSSurface(order_u=2, order_v=2, knot_u=[0, 0, 1, 1], knot_v=[0, 0, 1, 1], control_points=ctrl_pts, weights=weights)
        # Evaluate at (u,v) = (0.3,0.7)
        u, v = 0.3, 0.7
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)
        # The homogeneous sums give:
        # d[0][0] = 0.21*(P00,1) + 0.49*(P01,2) + 0.09*(P10,3) + 0.21*(P11,4)
        #            = ( (0.09+0.21, 0.49+0.21, 0), (0.21+0.98+0.27+0.84) ) = ((0.3, 0.7, 0), 2.3)
        # S = (0.3/2.3, 0.7/2.3, 0) ≈ (0.13043478, 0.30434783, 0)
        expected_S, expected_Su, expected_Sv, expected_Suu, expected_Suv, expected_Svv = (
            [0.13043478260869565, 0.30434782608695654, 0.0],
            [0.32136105860113423, -0.26465028355387527, 0.0],
            [-0.056710775047258986, 0.30245746691871456, 0.0],
            [-0.55888879756719, 0.4602613627023918, 0.0],
            [-0.09040848195939838, -0.14794115229719734, 0.0],
            [0.04931371743239912, -0.26300649297279527, 0.0],
        ) # Obtained using OpenNURBS 8.x

        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol))
        # The first derivatives (using the quotient rule) evaluate to approximately:

        self.assertTrue(np.allclose(SKL["Su"], expected_Su, rtol=1e-3))
        self.assertTrue(np.allclose(SKL["Sv"], expected_Sv, rtol=1e-3))
        print("Su,Sv: OK")
        # Second derivatives for a bilinear patch are zero.
        self.assertTrue(np.allclose(SKL["Suu"], np.zeros(3), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suv"], np.zeros(3), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Svv"], np.zeros(3), rtol=self.tol))

    def test_overdimensional(self):
        # A patch defined in R^4 (control points are 4–vectors) with weights=1.
        ctrl_pts = [[(0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)], [(1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)]]
        weights = np.array([[1.0, 1.0], [1.0, 1.0]])
        surf = NURBSSurface(order_u=2, order_v=2, knot_u=[0, 0, 1, 1], knot_v=[0, 0, 1, 1], control_points=ctrl_pts, weights=weights)
        # Evaluate at (u,v) = (0.5,0.5)
        u, v = 0.5, 0.5
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)
        # Expected S = ((0.5,0.5,0,0) in R^4) obtained by bilinear interpolation.
        expected_S = np.array([0.5, 0.5, 0.0, 0.0])
        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol))
        # Derivatives: Su = (1,0,0,0), Sv = (0,1,0,0)
        expected_Su = np.array([1.0, 0.0, 0.0, 0.0])
        expected_Sv = np.array([0.0, 1.0, 0.0, 0.0])
        self.assertTrue(np.allclose(SKL["Su"], expected_Su, rtol=self.tol))

        self.assertTrue(np.allclose(SKL["Sv"], expected_Sv, rtol=self.tol))
        # Second derivatives are zero.
        self.assertTrue(np.allclose(SKL["Suu"], np.zeros(4), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suv"], np.zeros(4), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Svv"], np.zeros(4), rtol=self.tol))

    def test_overdimensional_weighted(self):
        # Overdimensional (R^4) patch with non-uniform weights.
        ctrl_pts = [[(0.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0)], [(1.0, 0.0, 0.0, 0.0), (1.0, 1.0, 0.0, 0.0)]]
        # Weights: w00=1, w01=2, w10=3, w11=4.
        weights = np.array([[1.0, 2.0], [3.0, 4.0]])
        surf = NURBSSurface(order_u=2, order_v=2, knot_u=[0, 0, 1, 1], knot_v=[0, 0, 1, 1], control_points=ctrl_pts, weights=weights)
        # Evaluate at (u,v) = (0.5,0.5)
        u, v = 0.5, 0.5
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)
        # Homogeneous combination:
        # Each term weight: 0.25*(w_{ij}), so H = ( (0+0+1+1)/?, ... ) and S = (0.5/2.5, 0.5/2.5, 0, 0) = (0.2,0.2,0,0)
        expected_S = np.array([0.2, 0.2, 0.0, 0.0])
        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol))
        # The first derivatives (computed analogously) are approximately:
        expected_Su = np.array([0.24, -0.16, 0.0, 0.0])
        expected_Sv = np.array([-0.08, 0.32, 0.0, 0.0])
        self.assertTrue(np.allclose(SKL["Su"], expected_Su, rtol=1e-3))
        self.assertTrue(np.allclose(SKL["Sv"], expected_Sv, rtol=1e-3))
        # Second derivatives are zero.
        self.assertTrue(np.allclose(SKL["Suu"], np.zeros(4), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suv"], np.zeros(4), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Svv"], np.zeros(4), rtol=self.tol))

    def test_singular(self):
        # A singular (degenerate) surface: all control points are identical.
        ctrl_pts = [[(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)], [(1.0, 1.0, 1.0), (1.0, 1.0, 1.0)]]
        weights = np.array([[1.0, 1.0], [1.0, 1.0]])
        surf = NURBSSurface(order_u=2, order_v=2, knot_u=[0, 0, 1, 1], knot_v=[0, 0, 1, 1], control_points=ctrl_pts, weights=weights)
        u, v = 0.5, 0.5
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)
        expected_S = np.array([1.0, 1.0, 1.0])
        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol))
        # All derivatives vanish.
        self.assertTrue(np.allclose(SKL["Su"], np.zeros(3), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Sv"], np.zeros(3), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suu"], np.zeros(3), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suv"], np.zeros(3), rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Svv"], np.zeros(3), rtol=self.tol))

    def test_cylinder(self):
        # A canonical cylinder patch.
        # We represent a quarter of a circular arc (in u) extruded linearly (in v).
        # u–direction: quadratic (order=3) NURBS representation of a circular arc from (1,0) to (0,1).
        # The standard data (for a 90° arc) is:
        #   Control points in xy–plane:
        #     P0 = (1,0), P1 = (1,1), P2 = (0,1)
        #   Weights: w0 = 1, w1 = 1/√2, w2 = 1.
        # For the cylinder, we “extrude” in the z–direction.
        c =  math.sqrt(2.0)/2.0
        # Build a 3x2 control net:
        # For v=0 (bottom circle): z = 0.
        # For v=1 (top circle): z = 2.

        # Knot vectors:
        # u: order 3 --> [0,0,0,1,1,1]
        # v: order 2 --> [0,0,1,1]


        surf = cylinder(1.,2.)
        # Evaluate at (u,v) = (0.5,0.5)
        u, v = 0.5, 0.5
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)
        expected_S, expected_Su, expected_Sv, expected_Suu, expected_Suv, expected_Svv = (
            [[-1.0, 0.0, 1.0], [0.0, -5.65685424, -4.440892098500626e-16], [0.0, 0.0, 2.0],
             [32.0, -13.254834134788034, -3.552713678800501e-15], [0.0, 0.0, -8.881784197001252e-16], [0.0, 0.0, 0.0]]
        )  # Obtained using OpenNURBS 8.x

        print(SKL)
        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol))
        # For the circular arc the derivative at u=0.5 is known to be tangent.
        # From our minimal basis function derivative the u–derivative computes to approximately:

        self.assertTrue(np.allclose(SKL["Su"], expected_Su, rtol=1e-3))
        print("Su: OK")
        self.assertTrue(np.allclose(SKL["Sv"], expected_Sv, rtol=self.tol))
        print("Sv: OK")


        self.assertTrue(np.allclose(SKL["Suu"], expected_Suu, rtol=1e-3))
        print("Suu: OK")
        self.assertTrue(np.allclose(SKL["Suv"], expected_Suv, rtol=self.tol))
        print("Suv: OK")
        self.assertTrue(np.allclose(SKL["Svv"], expected_Svv, rtol=self.tol))
        print("Svv: OK")

    def test_cone(self):
        # A canonical cone patch.
        # We represent a ruled surface between an apex and a circular base.
        # Here we choose a simple 2x3 patch.
        # u–direction (order=2, linear): u=0 corresponds to the apex, u=1 to the base.
        # v–direction (order=3, quadratic): a quadratic Bezier representation of a circular arc.
        # Let the apex be A = (0,0,1) and the base (at z=0) be given by:
        #   B0 = (1,0,0), B1 = (1,1,0), B2 = (0,1,0).
        apex = (0.0, 0.0, 1.0)
        base_pts = [(1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
        # Build control net: 2 rows (u=0,1); row0 all apex, row1 are base.
        ctrl_pts = [[apex, apex, apex], [base_pts[0], base_pts[1], base_pts[2]]]
        # Use all weights = 1.
        weights = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
        # Knot vectors:
        # u (order=2): [0,0,1,1]
        # v (order=3): [0,0,0,1,1,1]
        surf = NURBSSurface(order_u=2, order_v=3, knot_u=[0, 0, 1, 1], knot_v=[0, 0, 0, 1, 1, 1], control_points=ctrl_pts, weights=weights)
        # Evaluate at (u,v) = (0.5,0.5)
        u, v = 0.5, 0.5
        SKL = evaluate_nurbs_surface(surf, u, v, d_order=2)
        # For the v–direction (quadratic) the Bezier evaluation of the base row gives:
        #   b0=0.25, b1=0.5, b2=0.25 so the base is:
        #   S_base = 0.25*(1,0,0) + 0.5*(1,1,0) + 0.25*(0,1,0)
        #           = (0.75, 0.75, 0)
        # The overall surface (ruled between apex and base) is:
        #   S = (1-u)*A + u*S_base = 0.5*(0,0,1) + 0.5*(0.75,0.75,0) = (0.375,0.375,0.5)

        # The u–derivative (direction from apex to base) is:
        #   Su = S_base - A = (0.75,0.75,0) - (0,0,1) = (0.75,0.75,-1)

        expected_S, expected_Su, expected_Sv, expected_Suu, expected_Suv, expected_Svv = (
            [0.375, 0.375, 0.5],
            [0.75, 0.75, -1],
            [-0.5, 0.5, 0],
            [0, 0, 0],
            [-1, 1, 0],
            [-1, -1, 0],
        )  # Obtained using OpenNURBS 8.x

        self.assertTrue(np.allclose(SKL["S"], expected_S, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Su"], expected_Su, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Sv"], expected_Sv, rtol=self.tol))
        # Second derivatives (for a linear interpolation in u and quadratic in v) are zero.
        self.assertTrue(np.allclose(SKL["Suu"], expected_Suu, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Suv"], expected_Suv, rtol=self.tol))
        self.assertTrue(np.allclose(SKL["Svv"], expected_Svv, rtol=self.tol))


if __name__ == "__main__":
    unittest.main()

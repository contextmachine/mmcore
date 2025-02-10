
from collections import namedtuple
from geomdl import NURBS



# ======================================================================
# Namedtuple definitions for NURBS surface and curve representations.
# ======================================================================
NURBSSurface = namedtuple('NURBSSurface', ['order_u', 'order_v', 'knot_u', 'knot_v', 'control_points', 'weights'])
NURBSCurve  = namedtuple('NURBSCurve',  ['order', 'knot', 'control_points', 'weights'])
BSplineCurve = namedtuple('BSplineCurve', ['order', 'knot', 'control_points'])

def evaluate_nurbs_curve(curve:NURBSCurve, t, d_order=0):
    """
    Evaluate a NURBS curve (which may be rational) at parameter value t.
    d_order = 0 returns only the point; d_order = 1 returns [point, derivative].
    Works in any dimension.
    """
    crv1=curve
    crv=NURBS.Curve()
    crv.degree=crv1.order-1

    crv.ctrlpts = crv1.control_points.tolist()

    crv.knotvector=crv1.knot.tolist()
    crv.weights=crv1.weights.tolist()

    return np.array(crv.derivatives(t,d_order))
def join_weights(surf:NURBSSurface):
    ptsw = np.zeros((*surf.control_points.shape[:-1], 4))

    ptsw[..., :-1] = surf.control_points
    ptsw[..., -1] = surf.weights
    return ptsw



import numpy as np
import math
from collections import namedtuple


np.set_printoptions(suppress=True)
from typing import TypedDict

# ---------------------------------------------------------------------------
# The implementation to be tested.
# (Normally you would import this from your module.)
# ---------------------------------------------------------------------------



class NURBSSurfaceJson(TypedDict):
    control_points: list[list[float]]
    knots_u: list[float]
    knots_v: list[float]
    size: list[float]
    degree: list[float]


def _from_dict(data: NURBSSurfaceJson) -> NURBSSurface:
    degu, degv = data["degree"]
    dim = len(data["control_points"][0])
    cpts = np.array(data["control_points"], dtype=float).reshape((*data["size"], dim))

    return NURBSSurface(
        order_u=degu + 1,
        order_v=degv + 1,
        knot_u=data["knots_u"],
        knot_v=data["knots_v"],
        control_points=cpts[..., :-1].tolist(),
        weights=np.ascontiguousarray(cpts[..., -1]),
    )


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
    ders_u = np.array(basis_function_ders(p, U, span_u, u, du))
    #print(q, V, span_v, v, d_order)
    ders_v = np.array(basis_function_ders(q, V, span_v, v, dv))
    #print("DU", ders_u)
    #print("DV", ders_v)

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
    #print(SKL)
    return SKL

# =============================================================================
# 4. Adams integrator (variable step, variable order Adams-Bashforth/Moulton)
# =============================================================================
def adams_integrate(f, y0, s0, s_end, tol, args=()):
    """
    Integrate the ODE system y' = f(s,y,...) from s0 to s_end using an
    adaptive-step Adams-Bashforth/Moulton predictor-corrector method.
    Here a four-step method is used (with variable step-size adjustment).
    """
    max_order = 4
    s = s0
    y = y0.copy()
    h = (s_end - s0) / 100.0  # initial step size guess
    S_hist = [s]
    Y_hist = [y.copy()]
    F_hist = [f(s, y, *args)]
    # Bootstrap: use RK4 to obtain max_order points.
    def rk4_step(s, y, h):
        k1 = f(s, y, *args)
        k2 = f(s + h/2.0, y + h/2.0 * k1, *args)
        k3 = f(s + h/2.0, y + h/2.0 * k2, *args)
        k4 = f(s + h, y + h * k3, *args)
        return y + h/6.0*(k1 + 2*k2 + 2*k3 + k4)
    while len(S_hist) < max_order and s < s_end:
        if s + h > s_end:
            h = s_end - s
        y = rk4_step(s, y, h)
        s = s + h
        S_hist.append(s)
        Y_hist.append(y.copy())
        F_hist.append(f(s, y, *args))
    # Now proceed with the Adams predictor-corrector steps.
    while s < s_end:
        if s + h > s_end:
            h = s_end - s
        # Adams-Bashforth 4-step predictor:
        f_n   = F_hist[-1]
        f_n1  = F_hist[-2]
        f_n2  = F_hist[-3]
        f_n3  = F_hist[-4]
        y_pred = Y_hist[-1] + h/24.0*(55*f_n - 59*f_n1 + 37*f_n2 - 9*f_n3)
        s_pred = s + h
        f_pred = f(s_pred, y_pred, *args)
        # Adams-Moulton 3-step corrector:
        y_corr = Y_hist[-1] + h/24.0*(9*f_pred + 19*f_n - 5*f_n1 + f_n2)
        err = np.linalg.norm(y_corr - y_pred)
        if err < tol:
            # Accept the step.
            s = s_pred
            y = y_corr
            S_hist.append(s)
            Y_hist.append(y.copy())
            F_hist.append(f(s, y, *args))
            if len(S_hist) > max_order:
                S_hist.pop(0)
                Y_hist.pop(0)
                F_hist.pop(0)
            # Increase h moderately.
            if err == 0:
                fac = 2
            else:
                fac = min(2, max(0.5, 0.9*(tol/err)**(1/4)))
            h = h * fac
        else:
            # Reject step and reduce h.
            fac = max(0.1, 0.9*(tol/err)**(1/4))
            h = h * fac
    return y

# =============================================================================
# 5. Geodesic ODE system (in the surface parameter domain)
# =============================================================================
def geodesic_ode(s, y, surface):
    """
    Given y = [u, v, u1, v1] (where u1, v1 are the derivatives with respect to arc-length s),
    compute the derivative y' with respect to s.
    Implements the system:
      du/ds   = u1,
      dv/ds   = v1,
      du1/ds  = -d11*u1^2 - 2*d12*u1*v1 - d22*v1^2,
      dv1/ds  = -e11*u1^2 - 2*e12*u1*v1 - e22*v1^2.
    The coefficients d11, d12, d22 are computed from the surface second derivatives;
    e11, e12, e22 are defined as in the paper.
    """
    u, v, u1, v1 = y
    derivs = evaluate_nurbs_surface(surface, u, v, d_order=2)
    r_u = np.array(derivs['Su'])
    r_v = np.array(derivs['Sv'])
    r_uu = np.array(derivs['Suu'])
    r_uv = np.array(derivs['Suv'])
    r_vv = np.array(derivs['Svv'])
    g11 = np.dot(r_u, r_u)
    g12 = np.dot(r_u, r_v)
    g22 = np.dot(r_v, r_v)
    cross_ru_rv = np.cross(r_u, r_v)
    norm_cross = np.linalg.norm(cross_ru_rv)
    if norm_cross == 0:
        norm_cross = 1e-8
    n = cross_ru_rv / norm_cross
    d11 = np.dot(n, np.cross(r_uu, r_v)) / norm_cross
    d12 = np.dot(n, np.cross(r_uv, r_v)) / norm_cross
    d22 = np.dot(n, np.cross(r_vv, r_v)) / norm_cross
    e11 = np.dot(n, np.cross(r_uu, r_uu)) / norm_cross  # identically 0
    e12 = np.dot(n, np.cross(r_uv, r_uv)) / norm_cross  # identically 0
    e22 = np.dot(n, np.cross(r_vv, r_vv)) / norm_cross  # identically 0
    du_ds = u1
    dv_ds = v1
    du1_ds = -d11*u1*u1 - 2*d12*u1*v1 - d22*v1*v1
    dv1_ds = -e11*u1*u1 - 2*e12*u1*v1 - e22*v1*v1
    return np.array([du_ds, dv_ds, du1_ds, dv1_ds])

# =============================================================================
# 6. Computation of initial geodesic conditions at a given progenitor parameter.
# =============================================================================
def compute_initial_geodesic_state(t, progenitor_curve, surface):
    """
    For a given progenitor curve parameter value t, compute the starting point (u0,v0)
    in the surface patch and the initial derivative (u1,v1) for the geodesic integration.
    This is done by evaluating the progenitor curve (in parameter space) and its derivative,
    mapping the progenitor tangent to the surface, computing b = (r_u x r_v) x t_3d and then
    solving for the coefficients U1, V1.
    """
    # Evaluate progenitor curve (which lies in parameter space)
    R = evaluate_nurbs_curve(progenitor_curve, t, d_order=1)
    #print(R)
    uv = R[0]         # [u0, v0]
    dR_dt = R[1]      # [du/dt, dv/dt]
    u0, v0 = uv[0], uv[1]
    #print(uv)
    du_dt, dv_dt = dR_dt[0], dR_dt[1]
    # Evaluate the surface first derivatives at (u0,v0)
    derivs = evaluate_nurbs_surface(surface, u0, v0, d_order=1)
    r_u = np.array(derivs['Su'])
    r_v = np.array(derivs['Sv'])
    # Progenitor tangent in 3D: t_3d = du_dt * r_u + dv_dt * r_v.
    t_3d = du_dt * r_u + dv_dt * r_v
    # Compute b = (r_u x r_v) x t_3d.
    cross_ru_rv = np.cross(r_u, r_v)
    b = np.cross(cross_ru_rv, t_3d)
    # Express b = U1*r_u + V1*r_v. To solve for [U1, V1], set up:
    #   [ [r_u·r_u, r_u·r_v], [r_v·r_u, r_v·r_v] ] [U1, V1]^T = [r_u·b, r_v·b]^T.
    g11 = np.dot(r_u, r_u)
    g12 = np.dot(r_u, r_v)
    g22 = np.dot(r_v, r_v)
    G = np.array([[g11, g12], [g12, g22]])
    rhs = np.array([np.dot(r_u, b), np.dot(r_v, b)])
    sol = np.linalg.solve(G, rhs)
    U1, V1 = sol[0], sol[1]
    # Normalize the vector (U1,V1) in the metric: sqrt(g11*U1^2 + 2*g12*U1*V1 + g22*V1^2)
    norm_factor = math.sqrt(g11*U1*U1 + 2*g12*U1*V1 + g22*V1*V1)
    if norm_factor == 0:
        norm_factor = 1e-8
    u1 = U1 / norm_factor
    v1 = V1 / norm_factor
    return np.array([u0, v0, u1, v1])

# =============================================================================
# 7. Compute one offset point by integrating the geodesic ODE.
# =============================================================================
def compute_offset_point(t, progenitor_curve, surface, g_func, adams_tol=1e-6):
    """
    For a given progenitor parameter t, compute one offset point by:
      (a) computing the initial geodesic state,
      (b) setting the travel distance g = g_func(t),
      (c) integrating the geodesic ODE (with the Adams integrator).
    Returns the final state y = [u, v, u1, v1].
    """
    y0 = compute_initial_geodesic_state(t, progenitor_curve, surface)
    g_dist = g_func(t)
    y_final = adams_integrate(lambda s, y: geodesic_ode(s, y, surface), y0, 0.0, g_dist, adams_tol)
    return y_final

# =============================================================================
# 8. B-spline interpolation of the offset curve in parameter space.
# =============================================================================
def bspline_basis(j, L, t, knot):
    """
    Evaluate the j-th B-spline basis function of order L at parameter value t using Cox-de Boor.
    """
    if L == 1:
        if (knot[j] <= t < knot[j+1]) or (t == knot[-1] and t == knot[j+1]):
            return 1.0
        else:
            return 0.0
    else:
        denom1 = knot[j+L-1] - knot[j]
        term1 = 0.0
        if denom1 != 0:
            term1 = (t - knot[j]) / denom1 * bspline_basis(j, L-1, t, knot)
        denom2 = knot[j+L] - knot[j+1]
        term2 = 0.0
        if denom2 != 0:
            term2 = (knot[j+L] - t) / denom2 * bspline_basis(j+1, L-1, t, knot)
        return term1 + term2
from geomdl import BSpline
def evaluate_bspline_curve(curve, t):
    """
    Evaluate a (non-rational) B-spline curve at parameter value t.
    """
    crv1=curve
    curve=BSpline.Curve()
    curve.degree=crv1.order-1


    curve.ctrlpts = crv1.control_points.tolist()
    curve.knotvector = crv1.knot.tolist()
    return np.array(curve.evaluate_single(t))



def evaluate_bspline_curve_derivative(curve, t):
    """
    Evaluate the derivative of the B-spline curve at t using formula (48):
      S'(t) = (L-1) * sum_{j=1}^{n-1} (Q_j - Q_{j-1})/(s_{j+L-1} - s_j) * N_{j,L-1}(t)
    """
    crv1=curve
    curve=BSpline.Curve()
    curve.degree=crv1.order-1
    curve.degree = crv1.order - 1

    curve.ctrlpts = crv1.control_points.tolist()
    curve.knotvector = crv1.knot.tolist()
    return np.array(curve.derivatives(t,1)[1])

def compute_knot_vector(points, L):
    """
    Compute the knot vector (S) for the B-spline interpolation (formula 40).
    points: list of offset points (in parameter space), length n.
    The knot vector has length n+L with s_0 = ... = s_{L-1} = 0 and s_n = ... = s_{n+L-1} = 1.
    The intermediate knots are determined so that the difference is proportional to the sum
    of L-1 adjacent polygon segments.
    """
    n = len(points)
    knot = np.zeros(n + L)
    knot[:L] = 0.0
    knot[n:] = 1.0
    segments = []
    total = 0.0
    for i in range(n - L + 1):
        seg_sum = 0.0
        for j in range(i, i + L - 1):
            seg_sum += np.linalg.norm(points[j+1] - points[j])
        segments.append(seg_sum)
        total += seg_sum
    for i in range(n - L):
        knot[i+L] = knot[i+L-1] + segments[i] / total
    #print(knot)
    return knot

def compute_parameter_values(knot, n, L):
    """
    Compute the parameter values at the data points (formula 41):
      xi_i = (1/(L-1))*(s_{i+1} + ... + s_{i+L-1}) for i = 0,..., n-1.
    """
    xi = np.zeros(n)
    for i in range(n):
        xi[i] = np.sum(knot[i+1 : i+L]) / (L - 1)
    return xi

def assemble_interpolation_matrix(xi, knot, n, L):
    """
    Assemble the n x n interpolation matrix with entries N_{j,L}(xi_i).
    """
    N_matrix = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            N_matrix[i, j] = bspline_basis(j, L, xi[i], knot)
    return N_matrix

def fit_bspline_interpolation(points, L):
    """
    Fit an interpolating B-spline curve S_L (integral curve in parameter space)
    through the given offset points.
    Returns a BSplineCurve with control points Q and knot vector.
    """
    n = len(points)
    knot = compute_knot_vector(points, L)
    xi = compute_parameter_values(knot, n, L)
    N_matrix = assemble_interpolation_matrix(xi, knot, n, L)
    Q = np.linalg.solve(N_matrix, np.array(points))
    return BSplineCurve(order=L, knot=knot, control_points=Q)

# =============================================================================
# 9. Mapping and tangent computations (Eqs. 47, 48, and 49)
# =============================================================================
def map_curve_to_surface(bspline_curve, surface, t):
    """
    Given a B-spline curve in parameter space (S_L), evaluate its value at t,
    and then map (u,v) to the 3D surface via the NURBS surface evaluation.
    """
    uv = evaluate_bspline_curve(bspline_curve, t)

    S = evaluate_nurbs_surface(surface, uv[0], uv[1], d_order=0)
    return S

def compute_tangent_vector(bspline_curve, surface, t):
    """
    Compute the 3D tangent vector at parameter t on the offset curve, as in Eq. (47).
    The derivative S_L'(t) is computed (using Eq. (48)) to get (u1_bar, v1_bar),
    and then t = u1_bar * r_u + v1_bar * r_v.
    """
    deriv_uv = evaluate_bspline_curve_derivative(bspline_curve, t)
    uv = evaluate_bspline_curve(bspline_curve, t)
    derivs = evaluate_nurbs_surface(surface, uv[0], uv[1], d_order=1)
    r_u = np.array(derivs['Su'])
    r_v = np.array(derivs['Sv'])
    return deriv_uv[0] * r_u + deriv_uv[1] * r_v

def compute_normal_vector(surface, uv, geodesic_deriv):
    """
    Compute the normal vector s_i (Eq. 49) at an offset point given by uv and
    the geodesic derivative (u1, v1). Here s_i = (r_u x r_v) x g_i.
    """
    derivs = evaluate_nurbs_surface(surface, uv[0], uv[1], d_order=1)
    r_u = np.array(derivs['Su'])
    r_v = np.array(derivs['Sv'])
    g_i = geodesic_deriv[0] * r_u + geodesic_deriv[1] * r_v
    return np.cross(np.cross(r_u, r_v), g_i)
_invphi = (math.sqrt(5) - 1) / 2
def golden_section_search(fun, bounds, tol):
    """
    Find the minimum value of a unimodal function on a closed interval using the Golden Section Search.

    Parameters:
        fun   : A callable function f(x) assumed to be unimodal on the interval.
        bounds: A tuple (a, b) representing the lower and upper bounds of the interval.
        tol   : The tolerance for the width of the search interval; the algorithm stops when (b - a) < tol.

    Returns:
        A tuple (x_min, f_min) where x_min is the approximate location of the minimum and f_min = fun(x_min).

    The algorithm reuses function evaluations so that each iteration requires only one new evaluation,
    achieving a logarithmic convergence rate in terms of the interval width.
    """
    a, b = bounds
    # The constant invphi is 1/phi, where phi is the golden ratio (~1.618)


    # Compute the two interior points
    c = b - _invphi * (b - a)
    d = a + _invphi * (b - a)
    fc = fun(c)
    fd = fun(d)

    # Loop until the current interval is small enough
    while (b - a) > tol:
        if fc < fd:
            # The minimum is in [a, d]
            b = d
            d = c
            fd = fc
            c = b - _invphi * (b - a)
            fc = fun(c)
        else:
            # The minimum is in [c, b]
            a = c
            c = d
            fc = fd
            d = a + _invphi * (b - a)
            fd = fun(d)

    # At this point, the minimum is approximately in [a, b].
    # To be robust, check the endpoints as well as the midpoint.
    x_mid = (a + b) / 2
    candidates = [(a, fun(a)), (x_mid, fun(x_mid)), (b, fun(b))]
    x_min, f_min = min(candidates, key=lambda pair: pair[1])

    return x_min, f_min



# =============================================================================
# 10. Descent method (using cubic interpolation) for minimum distance (Eq. 51)
# =============================================================================
def find_min_distance(R_l, bspline_curve, surface, tol=1e-6, max_iter=50):
    """
    Find the minimum distance between a given 3D point R_l and the 3D curve
    r(S_L(t)) defined by the B-spline curve in parameter space mapped onto the surface.
    This descent method uses a simple (cubic interpolation based) line search.
    """
    def f(t):
        S_t = map_curve_to_surface(bspline_curve, surface, t)
        #print(R_l,S_t)
        #print(t, R_l["S"], S_t['S'], R_l["S"]-S_t['S'])
        res=np.linalg.norm(R_l["S"] - S_t['S'])
        print(f'f: {t}, {res}',flush=True,end=' '*80+'\r')
        return res

    t_current,f_t=golden_section_search(f, (0.,1.), tol)

    #print("\rTC", (t_current,f_t),)
    return f_t


# =============================================================================
# 11. Main iterative process to approximate the offset curve.
# =============================================================================
def approximate_offset_curve(progenitor_curve, surface, g_func, epsilon1, epsilon2, bspline_order=4, initial_sample_count=5, adams_tol=1e-6):
    """
    Given a progenitor curve (in the surface parameter space) and a NURBS surface,
    construct an offset curve (lying on the surface) by:
      - For a set of progenitor parameter values t_i (initially uniformly sampled),
        compute the offset point P_i by integrating the geodesic ODE (travel distance = g(t_i)).
      - Fit an interpolating B-spline curve S_L in the parameter space through these points.
      - Sample S_L and compute at each sample point the 3D tangent (from S_L) and
        the geodesic tangent (from the integration) so as to check the angle (ϕ_i, Eq. 50).
      - Also, for additional terminal points (midpoints) compute the minimum distance
        (μ, Eq. 51) from the mapped B-spline curve to the geodesic offset.
      - If either the angle tolerance (ε₁) or distance tolerance (ε₂) is violated,
        add new sample points (by taking midpoints of parameter intervals) and iterate.
    Returns:
       offset_points_3d : a list (sampled) of 3D points on the final offset curve,
       bspline_curve_param : the final B-spline curve (in parameter space) approximation.
    """
    # Initialize progenitor parameter values uniformly in [0,1]
    T = np.linspace(0, 1, initial_sample_count)
    offset_data = {}  # keys: t; values: (uv, geodesic_deriv)
    for t in T:
        y_final = compute_offset_point(t, progenitor_curve, surface, g_func, adams_tol)
        uv = y_final[:2]
        geodesic_deriv = y_final[2:]
        offset_data[t] = (uv, geodesic_deriv)
    converged = False
    iteration = 0
    while not converged:
        iteration += 1
        T_sorted = np.sort(list(offset_data.keys()))
        points = [offset_data[t][0] for t in T_sorted]
        bspline_curve = fit_bspline_interpolation(points, bspline_order)
        n = len(points)
        xi = compute_parameter_values(bspline_curve.knot, n, bspline_curve.order)
        refine_needed = False
        new_T = set()
        # Check the angle criterion (Eq. 50) at each sample point.
        for i, t in enumerate(T_sorted):
            uv, geo_deriv = offset_data[t]
            t_vec = compute_tangent_vector(bspline_curve, surface, xi[i])
            s_vec = compute_normal_vector(surface, uv, geo_deriv)
            norm_t = np.linalg.norm(t_vec)
            norm_s = np.linalg.norm(s_vec)

            if norm_t == 0 or norm_s == 0:
                phi = 0.0
            else:
                cos_phi = abs(np.dot(t_vec, s_vec)) / (norm_t * norm_s)
                cos_phi = max(-1.0, min(1.0, cos_phi))
                phi = math.degrees(math.acos(cos_phi))
            #print('nt,ns,phi:',norm_t,norm_s, phi)

            if phi > epsilon1:
                refine_needed = True
                idx = list(T_sorted).index(t)
                print('angle fail', phi, t)
                if idx > 0:
                    t_prev = T_sorted[idx-1]
                    new_T.add((t_prev + t) / 2.0)

                if idx < len(T_sorted) - 1:
                    t_next = T_sorted[idx+1]
                    new_T.add((t + t_next) / 2.0)


        #print(new_T)
        # Additional terminal point check.
        for i in range(len(T_sorted)-1):
            t_mid = (T_sorted[i] + T_sorted[i+1]) / 2.0
            if t_mid not in offset_data:
                y_final = compute_offset_point(t_mid, progenitor_curve, surface, g_func, adams_tol)
                offset_data[t_mid] = (y_final[:2], y_final[2:])
            R_l = map_curve_to_surface(bspline_curve, surface, t_mid)
            #print(t_mid,end=' ')
            mu = find_min_distance(R_l, bspline_curve, surface, tol=1e-6)
            #print(mu)
            if mu > epsilon2:
                refine_needed = True
                new_T.add(t_mid)
                print('term fail', mu,t_mid)
        if not refine_needed:
            converged = True
        else:
            for t in new_T:
                if t not in offset_data:
                    y_final = compute_offset_point(t, progenitor_curve, surface, g_func, adams_tol)
                    offset_data[t] = (y_final[:2], y_final[2:])
    # After convergence, sample the final B-spline offset curve and map to 3D.
    sample_count = 100
    t_values = np.linspace(0, 1, sample_count)
    offset_points_3d = []
    for t in t_values:
        uv = evaluate_bspline_curve(bspline_curve, t)
        point_3d = evaluate_nurbs_surface(surface, uv[0], uv[1], d_order=0)["S"]
        offset_points_3d.append(point_3d)
    return np.array(offset_points_3d), bspline_curve

# =============================================================================
# 12. Example applications
# =============================================================================
if __name__ == '__main__':
    # --------------------------
    # Example 1: One octant of a sphere.
    # --------------------------
    # The spherical patch (radius = 2.0) is given as a rational quadratic Bézier surface.
    # The control points are given in 3D and the weights are provided separately.
    cp_sphere = np.array([
        [[2.0, 0.0, 0, 1.0], [1.4142135623730951, 0.0, 1.4142135623730949, 0.70710678118654757], [0.0, 0.0, 2.0, 1.0]],
        [[1.4142135623730951, 1.4142135623730949, 0., 0.70710678118654757], [1., 1.0, 1.0, 0.5],[0.0, 0.0, 1.4142135623730951, 0.70710678118654757]],[ [0, 2.0, 0, 1.0],
         [0, 1.4142135623730951, 1.4142135623730949, 0.70710678118654757],[0.0, 0.0, 2.0, 1.0]]], dtype=float)
    weights_sphere =  cp_sphere[...,-1]
    # For a quadratic Bézier surface the order is 3 in both directions.
    knot_u = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    knot_v = np.array([0, 0, 0, 1, 1, 1], dtype=float)
    sphere = NURBSSurface(order_u=3, order_v=3, knot_u=knot_u, knot_v=knot_v,
                          control_points=np.ascontiguousarray(cp_sphere[...,:-1]), weights=weights_sphere)
    # The progenitor curve is the circle corresponding to v = 0.5 in parameter space,
    # represented as a linear (order 2) B-spline curve from (0, 0.5) to (1, 0.5).
    cp_progenitor = np.array([[0, 0.5], [1, 0.5]], dtype=float)
    weights_progenitor = np.array([1.0, 1.0], dtype=float)
    knot_progenitor = np.array([0, 0, 1, 1], dtype=float)
    progenitor_curve = NURBSCurve(order=2, knot=knot_progenitor,
                                  control_points=cp_progenitor, weights=weights_progenitor)
    # Define the offset distance function g(t) = 0.2 (constant).
    g_func = lambda t: 0.2
    epsilon1 = 2.0       # angle tolerance in degrees
    epsilon2 = 0.0001    # distance tolerance
    initial_3d=[]
    for i in np.linspace(0.,1.,30):
        uv =evaluate_bspline_curve(progenitor_curve,i)

        initial_3d.append(evaluate_nurbs_surface(sphere,uv[0],uv[1], 0)['S'])

    offset_pts_3d, bspline_curve_param = approximate_offset_curve(
        progenitor_curve, sphere, g_func, epsilon1, epsilon2,
        bspline_order=4, initial_sample_count=5, adams_tol=1e-3)
    print("Example 1: Offset curve (3D points) on one octant of a sphere:")


    print(np.array(initial_3d).tolist())
    print(offset_pts_3d.tolist())

    # --------------------------
    # Example 2: Open curve on one quarter of a cylindrical surface.
    # --------------------------
    # Here the cylindrical surface is given as a quadratic rational Bézier patch (order 3 x 2).
    cp_cylinder = np.array([
        [[0, 2, 0], [0, 2, 2]],
        [[math.sqrt(2), math.sqrt(2), 0], [math.sqrt(2), math.sqrt(2), math.sqrt(2)]],
        [[2, 0, 0], [2, 0, 2]]
    ], dtype=float)
    weights_cylinder = np.array([
        [1.0, 1.0],
        [1.0/math.sqrt(2), 1.0/math.sqrt(2)],
        [1.0, 1.0]
    ], dtype=float)
    knot_u_cyl = np.array([0,0,0,1,1,1], dtype=float)  # order 3 in u
    knot_v_cyl = np.array([0,0,1,1], dtype=float)        # order 2 in v
    cylinder = NURBSSurface(order_u=3, order_v=2, knot_u=knot_u_cyl, knot_v=knot_v_cyl,
                            control_points=cp_cylinder, weights=weights_cylinder)
    # The progenitor curve is a fourth-order integral Bézier curve in the parameter space.
    cp_prog_cyl = np.array([[0.45, 0.20],
                            [0.40, 0.80],
                            [0.60, 0.80],
                            [0.55, 0.20]], dtype=float)
    weights_prog_cyl = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    knot_prog_cyl = np.array([0,0,0,0,1,1,1,1], dtype=float)
    progenitor_curve_cyl = NURBSCurve(order=4, knot=knot_prog_cyl,
                                      control_points=cp_prog_cyl, weights=weights_prog_cyl)
    # Here we use the same offset distance g(t) = 0.2.
    offset_pts_cyl_3d, bspline_curve_param_cyl = approximate_offset_curve(
        progenitor_curve_cyl, cylinder, g_func, epsilon1, epsilon2,
        bspline_order=4, initial_sample_count=6, adams_tol=1e-3)
    print("\nExample 2: Offset curve (3D points) on one quarter of a cylinder:")
    initial_3d=[]
    print(cylinder)
    for i in np.linspace(0.,1.,30):
        uv =evaluate_bspline_curve(progenitor_curve_cyl,i)

        initial_3d.append(evaluate_nurbs_surface(cylinder,uv[0],uv[1], 0)['S'])



    print(np.array(initial_3d).tolist())
    print(offset_pts_cyl_3d.tolist())


    # --------------------------
    # Example 3: Offset curve on an integral B-spline surface patch (order 7 x 5).
    # --------------------------
    # The control polyhedron is given by a 7x6 lattice (the coordinates are provided).
    cp_surface = np.array([
        [[0, 0, 10],   [5, 10, 20],   [10, 20, 25],  [5, 30, 15],   [0, 40, 0],    [5, 50, -10]],
        [[20, -5, 15], [25, 5, 20],    [20, 15, 25],  [15, 25, 30],  [20, 35, 25],  [25, 45, 10]],
        [[45, 5, 20],  [40, 15, 30],   [35, 25, 45],  [40, 35, 40],  [45, 45, 30],  [35, 55, 15]],
        [[60, 0, 40],  [65, 10, 45],   [60, 20, 50],  [55, 30, 60],  [55, 40, 40],  [60, 50, 30]],
        [[75, -5, 25], [80, 5, 30],    [85, 15, 40],  [80, 25, 45],  [75, 35, 35],  [70, 45, 20]],
        [[100, 0, 20], [105, 10, 25],  [110, 20, 30], [105, 30, 35], [100, 40, 28], [95, 50, 15]],
        [[125, 5, 15], [130, 15, 20],  [130, 25, 25], [125, 35, 20], [125, 45, 10], [120, 55, -5]]
    ], dtype=float)
    # The weights are taken as 1 (or you can assign positive real numbers).
    weights_surface = np.ones((7,6), dtype=float)
    knot_u_surf = np.array([0., 0., 0., 0., 0., 0., 0., 1., 1., 1., 1., 1., 1., 1.], dtype=float)
    knot_v_surf = np.array([0., 0., 0., 0., 0., 0.5, 1., 1., 1., 1., 1.], dtype=float)
    surface3 = NURBSSurface(order_u=7, order_v=5, knot_u=knot_u_surf, knot_v=knot_v_surf,
                            control_points=cp_surface, weights=weights_surface)
    # The progenitor curve is a cubic Bézier curve in the parameter space.
    cp_prog_surf = np.array([[0.1, 0.1],
                             [0.4, 0.3],
                             [0.7, 0.6],
                             [0.5, 0.8]], dtype=float)
    weights_prog_surf = np.array([1.0, 1.0, 1.0, 1.0], dtype=float)
    knot_prog_surf = np.array([0,0,0,0,1,1,1,1], dtype=float)
    progenitor_curve_surf = NURBSCurve(order=4, knot=knot_prog_surf,
                                       control_points=cp_prog_surf, weights=weights_prog_surf)
    # Set offset distance g(t) = 2.5.
    g_func3 = lambda t: 2.5
    epsilon1_3 = 4.5  # degrees
    epsilon2_3 = 0.0001
    offset_pts_surf_3d, bspline_curve_param_surf = approximate_offset_curve(
        progenitor_curve_surf, surface3, g_func3, epsilon1_3, epsilon2_3,
        bspline_order=4, initial_sample_count=6, adams_tol=1e-5)
    print("\nExample 3: Offset curve (3D points) on an integral B-spline surface patch:")
    initial_3d = []
    print(surface3)
    for i in np.linspace(0., 1., 30):
        uv = evaluate_bspline_curve(progenitor_curve_surf, i)

        initial_3d.append(evaluate_nurbs_surface(surface3, uv[0], uv[1], 0)['S'])

    print(np.array(initial_3d).tolist())
    print(offset_pts_surf_3d.tolist())


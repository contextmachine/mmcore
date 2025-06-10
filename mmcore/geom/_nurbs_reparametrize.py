import numpy as np
import math
from collections import namedtuple
from typing import List, Tuple

from mmcore.geom._nurbs_eval import evaluate_nurbs_curve,NURBSCurveTuple,_find_span_linear,compute_basis_function_derivatives_np,bspline_basis


# ---------------------------------------------------------------------------
# 1. Compute the arc–length function φ(t) by numerical integration.
# ---------------------------------------------------------------------------
def compute_arc_length(
    curve: NURBSCurveTuple, num_samples: int = 1000
) -> Tuple[float, np.ndarray, np.ndarray]:
    """
    Computes the arc length L of the NURBS curve and returns three items:
      - L: the total length (φ(1))
      - t_vals: uniformly spaced parameters in [0,1]
      - phi_vals: cumulative arc length (φ(t)) at the t_vals
    Integration is performed by the composite trapezoidal rule.
    """
    t_vals = np.linspace(0, 1, num_samples)
    speeds = np.zeros_like(t_vals)
    for i, t in enumerate(t_vals):
        # Evaluate first derivative; use evaluate_nurbs_curve_array with d_order=1
        C1 = evaluate_nurbs_curve(curve, t, d_order=1)["C1"]
        speeds[i] = np.linalg.norm(C1)
    # Compute cumulative arc length using trapezoidal integration
    phi_vals = np.zeros_like(t_vals)
    for i in range(1, num_samples):
        phi_vals[i] = phi_vals[i - 1] + 0.5 * (speeds[i - 1] + speeds[i]) * (
            t_vals[i] - t_vals[i - 1]
        )
    L = phi_vals[-1]
    return L, t_vals, phi_vals


# ---------------------------------------------------------------------------
# 2. Compute the optimal reparameterization weight w using a simplified Remez-type iteration.
#     In our global case the approximant is:
#         r(t) = [w*L*t] / [(1-t) + w*t]
#     and the new (arc length) parameter is u = r(t)/L.
#
#     For t in [0,1] we require that the error function e(t) = r(t) – φ(t)
#     has alternating extrema (with equal absolute value). We use two sample points t0 and t1.
# ---------------------------------------------------------------------------
def find_optimal_w(
    L: float,
    t_vals: np.ndarray,
    phi_vals: np.ndarray,
    tol: float = 1e-5,
    max_iter: int = 10,
) -> float:
    """
    Finds the weight w such that the rational approximant
         r(t) = (w * L * t)/((1-t) + w*t)
    best approximates the arc–length function φ(t) (obtained from (t_vals, phi_vals))
    in the uniform (min–max) sense.

    We use two alternating points. Initially we choose t0 = 0.25 and t1 = 0.67.
    (In the paper these are chosen as a + (b-a)/4 and b - (b-a)/3 with a=0, b=1.)

    The iteration adjusts w so that
         d0 = φ(t0) - r(t0)    and    d1 = r(t1) - φ(t1)
    have the same magnitude.
    """
    # initial sample points
    t0 = 0.25
    t1 = 0.67
    # Interpolate φ(t) from the computed samples:
    phi_t0 = np.interp(t0, t_vals, phi_vals)
    phi_t1 = np.interp(t1, t_vals, phi_vals)
    # initial guess for w
    w = 1.0
    for it in range(max_iter):
        # Compute r(t0) and r(t1)
        r_t0 = (w * L * t0) / ((1 - t0) + w * t0)
        r_t1 = (w * L * t1) / ((1 - t1) + w * t1)
        # Errors (we want r(t0) = φ(t0)+d and r(t1)= φ(t1)-d, so we equate the absolute errors)
        d0 = r_t0 - phi_t0
        d1 = r_t1 - phi_t1
        # Our target is d0 + d1 = 0 (i.e. equal magnitude and opposite sign)
        error = d0 + d1
        # Simple update: adjust w in proportion to error (this is a secant-like update)
        # Note: This update rule is heuristic.
        w_new = w - 0.1 * error / (
            L * (t0 + t1)
        )  # denominator chosen to scale the change
        # Check convergence
        if abs(w_new - w) < tol:
            w = w_new
            break
        w = w_new
    return max(w, 1e-8)  # ensure positive


# ---------------------------------------------------------------------------
# 3. Define the (global) reparameterization mapping and its inverse.
#
#     With the optimal w and L, we have:
#
#         r(t) = (w * L * t)/((1-t) + w*t)
#         u = r(t)/L = (w * t)/((1-t)+w*t)
#
#     and the inverse mapping is given by solving for t:
#
#         u = (w*t)/((1-t)+w*t)
#         => t = u / (w + u*(1-w))
# ---------------------------------------------------------------------------
def reparam_mapping(t: float, L: float, w: float) -> float:
    """Compute r(t) = (w * L * t)/((1-t) + w*t)."""
    return (w * L * t) / ((1 - t) + w * t)


def new_parameter(u: float, L: float, w: float) -> float:
    """Compute the new parameter u = r(t)/L for a given t.
    (Here u is the new arc–length parameter in [0,1].)
    Note: When sampling the new curve we use the inverse mapping below."""
    # In practice, this is not needed because we define u = r(t)/L.
    return u


def inverse_mapping(u: float, w: float) -> float:
    """
    Given u = (w*t)/((1-t)+w*t), return t in [0,1] such that:
         t = u / (w + u*(1-w))
    """
    return u / (w + u * (1 - w))


# ---------------------------------------------------------------------------
# 4. Reparameterize the curve.
#
#    We sample uniformly in the new parameter u (which ideally corresponds to arc–length)
#    and obtain the corresponding old parameters t = ψ⁻¹(u) using the inverse mapping.
#
#    Then we evaluate the original curve at these t–values to get a set of points that are
#    (approximately) uniformly spaced by arc–length.
#
#    Finally, we build a new NURBS curve that interpolates these points using a standard
#    interpolation procedure.
# ---------------------------------------------------------------------------
def reparameterize_curve(
    curve: NURBSCurveTuple, num_interp_points: int = None, tol: float = 1e-5
) -> NURBSCurveTuple:
    """
    Reparameterize the given NURBS curve with respect to arc length.

    The procedure is as follows:
      1. Compute the arc–length function φ(t) and total length L.
      2. Use a Remez–type iteration to compute the optimal weight w for the rational approximant
         r(t) = (w*L*t)/((1-t)+w*t).
      3. Define the inverse mapping t = ψ⁻¹(u) = u/(w + u*(1-w)), where u = r(t)/L.
      4. Sample num_interp_points values uniformly in u ∈ [0,1] and compute t.
      5. Evaluate the original curve at these t–values.
      6. Interpolate a new NURBS curve that exactly fits these points.

    If num_interp_points is not provided, we choose it equal to the original number of control points.
    """
    if num_interp_points is None:
        num_interp_points = len(curve.control_points)
    # 1. Compute arc length and φ(t)
    L, t_samples, phi_samples = compute_arc_length(curve, num_samples=1000)
    # 2. Compute optimal weight w using our two–point iteration
    w = find_optimal_w(L, t_samples, phi_samples, tol=tol, max_iter=20)
    # 3. For a uniform partition in u (new parameter) compute t = ψ⁻¹(u)
    u_vals = np.linspace(0, 1, num_interp_points)
    t_new = np.array([inverse_mapping(u, w) for u in u_vals])
    # 4. Evaluate the original curve at these t–values to get interpolation points.
    interp_points = [evaluate_nurbs_curve(curve, t)["C"] for t in t_new]
    interp_points = np.array(interp_points)
    # 5. Build a new NURBS curve by interpolating these points.
    new_curve = interpolate_nurbs_curve(interp_points, degree=curve.order - 1)
    return new_curve


# ---------------------------------------------------------------------------
# 5. NURBS Curve Interpolation.
#
#    Given a set of data points and a chosen degree p, we compute:
#
#      (a) the chord–length parameterization,
#      (b) the knot vector (using averaging),
#      (c) and solve the linear system to obtain the control points.
#
#    Here we assume a non–rational (B–spline) interpolation; the weights will be 1.
# ---------------------------------------------------------------------------
def chord_length_parametrization(points: np.ndarray) -> np.ndarray:
    """
    Compute chord length parameters for a set of points.
    Returns an array u of parameters in [0,1].
    """
    dists = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumdist = np.concatenate(([0], np.cumsum(dists)))
    total = cumdist[-1]
    if total == 0:
        return np.linspace(0, 1, len(points))
    return cumdist / total


def generate_knot_vector(u: np.ndarray, degree: int) -> List[float]:
    """
    Generate an open (clamped) knot vector for interpolation with given parameters u and degree.
    Number of control points n = len(u).
    Knot vector length = n + degree + 1.
    """
    n = len(u)
    m = n + degree + 1
    knot = [0.0] * (degree + 1)
    for j in range(1, n - degree):
        s = 0.0
        for i in range(j, j + degree):
            s += u[i]
        knot.append(s / degree)
    knot += [1.0] * (degree + 1)
    return knot


def bspline_basis(j: int, p: int, knot: List[float], u: float) -> float:
    """
    Cox-de Boor recursion for B-spline basis function.
    """
    if p == 0:
        if knot[j] <= u < knot[j + 1] or (u == knot[-1] and u == knot[j + 1]):
            return 1.0
        else:
            return 0.0
    denom1 = knot[j + p] - knot[j]
    term1 = 0.0
    if denom1 != 0:
        term1 = (u - knot[j]) / denom1 * bspline_basis(j, p - 1, knot, u)
    denom2 = knot[j + p + 1] - knot[j + 1]
    term2 = 0.0
    if denom2 != 0:
        term2 = (knot[j + p + 1] - u) / denom2 * bspline_basis(j + 1, p - 1, knot, u)
    return term1 + term2


def interpolate_nurbs_curve(points: np.ndarray, degree: int) -> NURBSCurveTuple:
    """
    Interpolate a set of points with a (non-rational) B-spline curve of given degree.
    Returns a NURBSCurveTuple with weights equal to 1.
    """
    n_pts = len(points)
    # Parameter values by chord-length
    u = chord_length_parametrization(points)
    # Build knot vector
    knot = generate_knot_vector(u, degree)
    n_ctrlpts = n_pts  # for interpolation we choose the same number as data points

    # Set up the basis function matrix N (size n_pts x n_ctrlpts)
    N = np.zeros((n_pts, n_ctrlpts))
    for i in range(n_pts):
        for j in range(n_ctrlpts):
            N[i, j] = bspline_basis(j, degree, knot, u[i])
    # Solve for control points Q: N * Q = points
    # This is done separately for each coordinate.
    Q = np.zeros_like(points)
    # Use least squares in case N is not square
    for d in range(points.shape[1]):
        Q[:, d], _, _, _ = np.linalg.lstsq(N, points[:, d], rcond=None)
    # Set uniform weights 1.
    weights = np.ones(n_ctrlpts)
    return NURBSCurveTuple(
        order=degree + 1, knot=knot, control_points=Q.tolist(), weights=weights.tolist()
    )


# ---------------------------------------------------------------------------
# Example usage:
# --------------
# Suppose we have an original NURBS curve (curve_orig) defined as a NURBSCurveTuple.
# We can reparameterize it by arc length as follows:
#
#    new_curve = reparameterize_curve(curve_orig, tol=1e-5)
#
# Then new_curve is a NURBS curve whose parameter is (approximately) the arc length.
#
# Good luck!
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    # For testing, one would need to define an example NURBS curve.
    # Here we define a simple quadratic NURBS curve (non-rational case for simplicity).
    curve_orig = NURBSCurveTuple(
        order=3,  # quadratic (degree 2)
        knot=[0, 0, 0, 1, 2, 2, 2],
        control_points=[[0, 0], [1, 2], [3, 3], [4, 0]],
        weights=[1, 1, 1, 1],
    )
    new_curve = reparameterize_curve(curve_orig, tol=1e-5)
    print("Original Curve:")
    print(curve_orig)
    print("Reparameterized Curve (interpolatory form):")
    print(new_curve)

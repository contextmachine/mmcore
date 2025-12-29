import numpy as np
import math

from mmcore.numeric.vectors import scalar_norm,norm

from mmcore.geom._nurbs_eval import NURBSCurveTuple,to_homogeneous_1d

from mmcore.geom._nurbs_knots import decompose_curve

def curvature_based_step(tolerance, curvature_radius):
    return 2 * np.sqrt(2 * curvature_radius * tolerance - tolerance ** 2)


def arc_height(chord_length, curvature_radius):
    """
    ___
    :param chord_length:
    :param curvature_radius:
    :return:
    """
    return curvature_radius - math.sqrt((curvature_radius - chord_length / 2) * (curvature_radius + chord_length / 2))


def step(crv, t, tol):
    K = crv.curvature(t)
    r = 1 / np.linalg.norm(K)
    return np.sqrt(r ** 2 - (r - tol) ** 2) * 2


def parametric_arc_length(func, t_start, t_end, dt=1e-3):
    # Generate a list of t values from t_start to t_end
    t_values = np.arange(t_start, t_end+dt, dt)
    num_points = len(t_values)-1

    # Calculate the derivatives using finite differences
    #print(dt)
    arc_length = 0.
    for i in range(num_points):
        derivative = (np.array(func(t_values[i + 1])) - np.array(func(t_values[i]))) / dt
        # It is similar by each component
        # dx_dt = (x_t(t_values[i + 1]) - x_t(t_values[i])) / dt
        # dy_dt = (y_t(t_values[i + 1]) - y_t(t_values[i])) / dt
        # ...

        # Calculate the integrand sqrt((dx/dt)^2 + (dy/dt)^2)
        integrand = scalar_norm(derivative)
        # Use the trapezoidal rule to approximate the integral
        if i == 0 or i == num_points - 1:
            integrand *= 0.5
        arc_length += integrand

    arc_length *= dt
    return arc_length


import numpy as np


def subdivide_bezier(P, t=0.5):
    """
    De Casteljau subdivision of a degree-n Bezier curve at parameter t.

    Parameters
    ----------
    P : array_like, shape (n+1, d)
        Control points.
    t : float
        Subdivision parameter in [0, 1].

    Returns
    -------
    P_left : ndarray, shape (n+1, d)
        Control points of the left sub-curve on [0, t].
    P_right : ndarray, shape (n+1, d)
        Control points of the right sub-curve on [t, 1].
    """
    P = np.asarray(P, dtype=float)
    n_plus_1, d = P.shape
    n = n_plus_1 - 1

    levels = [P]
    for _ in range(n):
        prev = levels[-1]
        curr = (1.0 - t) * prev[:-1] + t * prev[1:]
        levels.append(curr)

    P_left = np.empty_like(P)
    for k in range(n_plus_1):
        P_left[k] = levels[k][0]

    P_right = np.empty_like(P)
    for k in range(n_plus_1):
        P_right[k] = levels[n - k][-1]

    return P_left, P_right


def chord_and_polygon_length(P_e):
    """
    Compute chord length and control polygon length in Euclidean space.

    Parameters
    ----------
    P_e : array_like, shape (n+1, d)
        Euclidean control points.

    Returns
    -------
    L_chord : float
        Length of the straight segment from P_e[0] to P_e[-1].
    L_poly : float
        Length of the polyline through all control points.
    """
    P_e = np.asarray(P_e, dtype=float)
    diffs = np.diff(P_e, axis=0)              # (n, d)
    seg_lengths = np.linalg.norm(diffs, axis=1)
    L_poly = float(seg_lengths.sum())
    L_chord = float(np.linalg.norm(P_e[-1] - P_e[0]))
    return L_chord, L_poly


def project_homogeneous(P_h, eps=1e-15):
    """
    Project homogeneous control points to Euclidean space.

    Parameters
    ----------
    P_h : array_like, shape (n+1, d+1)
        Homogeneous control points [w*x, w*y, ..., w].
    eps : float
        Small threshold to guard against near-zero weights.

    Returns
    -------
    P_e : ndarray, shape (n+1, d)
        Euclidean control points.
    """
    P_h = np.asarray(P_h, dtype=float)
    w = P_h[..., -1:]
    if np.any(np.abs(w) < eps):
        raise ValueError("Homogeneous weight too close to zero in rational Bezier.")
    coords = P_h[..., :-1] / w
    return coords


def bezier_arc_length(P, tol=1e-6, max_depth=32, rational=False):
    """
    Adaptive arc length estimation for polynomial or rational Bezier curves in R^d.

    Parameters
    ----------
    P : array_like
        Control points of the Bezier curve over t in [0,1].

        If rational is False:
            shape (n+1, d) : Euclidean control points.
        If rational is True:
            shape (n+1, d+1) : homogeneous control points [w*x, w*y, ..., w].

    tol : float
        Desired global absolute error bound on arc length.
    max_depth : int
        Maximum allowed subdivision depth (safety limit).
    rational : bool
        If True, treat P as homogeneous control points of a rational Bezier.
        If False, treat P as ordinary polynomial Bezier control points.

    Returns
    -------
    L_est : float
        Estimated arc length (midpoint of lower and upper bounds).
    L_lower : float
        Lower bound on the true arc length (sum of chords).
    L_upper : float
        Upper bound on the true arc length (sum of control polygon lengths).
    """
    P = np.asarray(P, dtype=float)
    n_plus_1 = P.shape[0]

    if n_plus_1 <= 1:
        return 0.0, 0.0, 0.0

    # Trivial linear case
    if rational:
        # homogeneous -> project to Euclidean for trivial segment
        if n_plus_1 == 2:
            P_e = project_homogeneous(P)
            L = float(np.linalg.norm(P_e[-1] - P_e[0]))
            return L, L, L
    else:
        if n_plus_1 == 2:
            L = float(np.linalg.norm(P[-1] - P[0]))
            return L, L, L

    L_lower = 0.0
    L_upper = 0.0

    # Stack entries:
    #   if rational:  (P_seg_homog, eps_seg, depth)
    #   else:         (P_seg_euclid, eps_seg, depth)
    stack = [(P, tol, 0)]

    while stack:
        P_seg, eps_seg, depth = stack.pop()

        # Obtain Euclidean control points for this segment
        if rational:
            P_e = project_homogeneous(P_seg)
        else:
            P_e = P_seg

        L_chord, L_poly = chord_and_polygon_length(P_e)
        delta = L_poly - L_chord

        # Accept segment if flat enough or depth limit reached
        if delta <= eps_seg or depth >= max_depth:
            L_lower += L_chord
            L_upper += L_poly
            continue

        # Subdivide
        P_left, P_right = subdivide_bezier(P_seg, t=0.5)
        eps_child = 0.5 * eps_seg

        stack.append((P_left, eps_child, depth + 1))
        stack.append((P_right, eps_child, depth + 1))

    L_est = 0.5 * (L_lower + L_upper)
    return L_est, L_lower, L_upper

def nurbs_length(nurbs_curve:NURBSCurveTuple, tol:bool=1e-6,full_return:bool=False,max_depth=128):
    rational=np.allclose(nurbs_curve.weights, 1)
    beziers=decompose_curve(nurbs_curve)
    bez: NURBSCurveTuple
    l_est=0
    l_upp = 0
    l_low=0
    for bez in beziers:
        if rational:
            L_est, L_lower, L_upper=bezier_arc_length(to_homogeneous_1d(bez.control_points,bez.weights),tol=tol,rational=rational,max_depth=max_depth)
        else:
            L_est, L_lower, L_upper=bezier_arc_length(bez.control_points, tol=tol, rational=rational,
                              max_depth=max_depth)

        l_est+=L_est
        l_low+=L_lower
        l_upp+=L_upper
    if full_return:
        return l_est,l_low,l_upp
    return l_est
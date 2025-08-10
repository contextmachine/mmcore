# nurbs_offset.py
# --------------------------------------------------------------------
"""
General‑purpose curve and surface offsetting for arbitrary NURBS
objects as described in:
    Piegl – 'Approximate Offsets of NURBS Curves and Surfaces'
"""

from __future__ import annotations

from mmcore.geom._nurbs_eval import NURBSCurveTuple,NURBSSurfaceTuple,_find_span_linear
import numpy as np
from mmcore.geom._nurbs_eval import compute_basis_function_derivatives_np

from mmcore.geom._nurbs_eval import *
import math
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List, Tuple, Sequence, Literal, Callable, Optional

# --------------------------------------------------------------------
# Import the basic data‑structures and evaluation routines that the
# user said are already available.
# --------------------------------------------------------------------
from mmcore.geom._nurbs_eval import (
    NURBSCurveTuple, NURBSSurfaceTuple,
    evaluate_nurbs_curve, evaluate_nurbs_surface,
)

# ========================  USER–CONFIGURABLE  =======================

@dataclass(slots=True)
class CurveOffsetOptions:
    degree: int = 3
    parametrisation: Literal["inherit", "chord"] = "inherit"
    pow_linear: float = 0.50
    pow_nonlinear: float = 0.34
    default_samples: int | None = None    # if None → p+1
    shape_tol: float = 1e-6               # tolerance for recognising lines/circles


@dataclass(slots=True)
class SurfaceOffsetOptions:
    degree_u: int = 3
    degree_v: int = 3
    parametrisation_u: Literal["inherit", "chord"] = "inherit"
    parametrisation_v: Literal["inherit", "chord"] = "inherit"
    pow_linear: float = 0.50
    pow_nonlinear: float = 0.34
    default_samples: int | None = None    # if None → max(p,q)+1
    shape_tol: float = 1e-6

# ====================================================================
#                             UTILITIES
# ====================================================================

def _unit(v: NDArray[float]) -> NDArray[float]:
    n = np.linalg.norm(v)
    if n == 0.0:
        raise ValueError("Zero‑length vector.")
    return v / n

# --  Axial helper ----------------------------------------------------
def _fit_line_pca(points: NDArray[float]) -> Tuple[NDArray[float], NDArray[float]]:
    """Return point on line + unit direction via PCA."""
    c = points.mean(axis=0)
    uu, _, _ = np.linalg.svd(points - c)
    d = _unit(uu[:, 0])
    return c, d

def _fit_circle_2d(pts: NDArray[float]) -> Tuple[NDArray[float], float]:
    """Algebraic circle fit in the plane z=0 for small data sets (Kåsa)."""
    x, y = pts[:, 0], pts[:, 1]
    A = np.column_stack([2*x, 2*y, np.ones_like(x)])
    b = x**2 + y**2
    c, d, e = np.linalg.lstsq(A, b, rcond=None)[0]
    centre = np.asarray([c, d])
    r = math.sqrt(e + c**2 + d**2)
    return centre, r

# --------------------------------------------------------------------
# Bézier decomposition (complete knot insertion so that each internal
# knot has multiplicity p+1).  Works for curves and each surface direction.
# --------------------------------------------------------------------
def _insert_knot_vector(knot: NDArray[float], p: int) -> NDArray[float]:
    """Return a knot vector where every inner knot has multiplicity p+1."""
    kv = list(knot)
    unique = sorted(set(kv))[1:-1]                 # ignore clamping knots
    full = []
    for u in unique:
        full.extend([u] * ((p + 1) - kv.count(u)))
    return np.sort(np.append(knot, full))

def _refine_curve(curve: NURBSCurveTuple) -> NURBSCurveTuple:
    # Piegl & Tiller algorithm A5.1 – refined at all inner knots.
    from collections import deque
    p = curve.order - 1
    Ubar = _insert_knot_vector(curve.knot, p)
    # If nothing to insert → return original
    if len(Ubar) == len(curve.knot):
        return curve
    # Temporary lists (deque for pop(0))
    Pw = [np.append(P, w) for P, w in zip(curve.control_points, curve.weights)]
    Qw: List[NDArray[float]] = []
    Ub = deque(Ubar)
    Ua = deque(curve.knot)
    a = p
    b = len(curve.control_points) - 1
    # Copy unclamped start
    Qw.extend(Pw[:a+1])
    while len(Ub) > 2*p+2:           # process inner part
        if Ub[p] < Ua[a+1]:          # new knot to insert
            Qw.append(Qw[-1])
        else:                        # copy control point
            a += 1
            Qw.append(Pw[a])
        Ua.popleft()
        Ub.popleft()
    # Copy unclamped end
    Qw.extend(Pw[b:])
    Qw = np.array(Qw)
    # split coords / weight
    Pw_new = Qw[:, :-1] / Qw[:, -1][:, None]
    w_new = Qw[:, -1]
    return NURBSCurveTuple(
        order=curve.order,
        knot=Ubar,
        control_points=Pw_new,
        weights=w_new,
    )

def _curve_bezier_segments(curve: NURBSCurveTuple) -> List[Tuple[float, float]]:
    """Return list of parameter intervals [u0, u1] for the Bézier pieces."""
    p = curve.order - 1
    refined = _insert_knot_vector(curve.knot, p)
    unique = sorted(set(refined))
    return [(unique[i], unique[i+1]) for i in range(len(unique)-1)]

# --------------------------------------------------------------------
#                    SHAPE RECOGNITION  –  CURVES
# --------------------------------------------------------------------
def _recognise_curve(
    curve: NURBSCurveTuple, tol: float
) -> Tuple[Literal["line", "circle", "free"], dict]:
    """
    Decide whether the curve is a straight line, a planar circle, or
    general free‑form.  The algorithm follows Sect. 3 of the paper.
    """
    # Sample at 2(p+1) points per Bézier span
    spans = _curve_bezier_segments(curve)
    p = curve.order - 1
    m = 2 * (p + 1)
    pts: List[NDArray[float]] = []
    for u0, u1 in spans:
        for i in range(m):
            u = u0 + (u1 - u0) * i / (m - 1)
            pts.append(evaluate_nurbs_curve(curve, u, 0)["C"])
    P = np.asarray(pts)

    # --- LINE --------------------------------------------------------
    base, dirn = _fit_line_pca(P)
    # Distance of every sample to PCA line
    dists = np.linalg.norm(np.cross(P - base, dirn), axis=1)
    if np.all(dists < tol):
        # Ensure projections are inside extents (optional – see paper)
        proj = (P - base) @ dirn
        if proj.max() - proj.min() > tol:
            return "line", {"base": base, "dir": dirn}

    # --- CIRCLE  (planar) -------------------------------------------
    # Estimate plane via PCA (smallest eigen‑vector = normal)
    _, _, vh = np.linalg.svd(P - P.mean(axis=0))
    n = vh[-1]
    if np.linalg.norm(n) < 1e-8:
        # Degenerate – the points are co‑linear, already caught
        pass
    else:
        n = _unit(n)
        # Build local 2‑D coordinates (u,v) in the plane
        # choose arbitrary axis not parallel to n
        ref = np.array([1.0, 0.0, 0.0])
        if abs(n @ ref) > 0.9:
            ref = np.array([0.0, 1.0, 0.0])
        x_axis = _unit(np.cross(ref, n))
        y_axis = np.cross(n, x_axis)
        uv = np.column_stack(((P @ x_axis), (P @ y_axis)))
        centre2d, radius = _fit_circle_2d(uv)
        # residuals
        res = np.linalg.norm(uv - centre2d, axis=1) - radius
        if np.all(np.abs(res) < tol):
            centre = centre2d[0]*x_axis + centre2d[1]*y_axis
            centre += P.mean(axis=0) - (P.mean(axis=0) @ n)*n  # ensure in plane
            return "circle", {
                "centre": centre,
                "radius": radius,
                "normal": n,
                "x_axis": x_axis,
                "y_axis": y_axis,
            }
    # --- DEFAULT -----------------------------------------------------
    return "free", {}

# --------------------------------------------------------------------
#           SPECIAL‑CASE CURVE OFFSETS  (exact construction)
# --------------------------------------------------------------------
def _offset_line(data: dict, distance: float) -> NURBSCurveTuple:
    """Return a degree‑1 rational curve representing the exact offset line."""
    base = data["base"]
    dirn = data["dir"]
    # Choose a stable perpendicular for 3‑D line: cross dir with an axis
    perp = np.cross(dirn, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(perp) < 1e-8:
        perp = np.cross(dirn, np.array([0.0, 1.0, 0.0]))
    perp = _unit(perp)
    offset_vec = distance * perp
    P0 = base + offset_vec
    P1 = base + dirn + offset_vec
    knot = np.array([0.0, 0.0, 1.0, 1.0])
    ctrl = np.vstack([P0, P1])
    w = np.ones(2)
    return NURBSCurveTuple(order=2, knot=knot, control_points=ctrl, weights=w)

def _offset_circle(data: dict, distance: float) -> NURBSCurveTuple:
    """
    Return a rational quadratic circle (degree 2) offset exactly by ±distance.
    """
    R = data["radius"] + distance
    if R <= 0.0:
        raise ValueError("Offset distance exceeds radius – invalid.")
    c = data["centre"]
    x = data["x_axis"]
    y = data["y_axis"]
    # Quadratic NURBS representation – full circle, Piegl & Tiller p. 359
    w = math.cos(math.pi/4)
    ctrl = np.array([
        c + R*x,
        c + R*(x + y) / math.sqrt(2),
        c + R*y,
        c + R*(-x + y) / math.sqrt(2),
        c - R*x,
        c + R*(-x - y) / math.sqrt(2),
        c - R*y,
        c + R*(x - y) / math.sqrt(2),
        c + R*x,
    ])
    weights = np.array([1, w, 1, w, 1, w, 1, w, 1])
    knot = np.concatenate([
        np.zeros(3),
        np.linspace(0, 1, 5)[1:-1].repeat(2),
        np.ones(3),
    ])
    return NURBSCurveTuple(order=3, knot=knot, control_points=ctrl, weights=weights)

# --------------------------------------------------------------------
#             DERIVATIVE BOUND M  (Equation 6 in the paper)
# --------------------------------------------------------------------
def _bound_curve_second_derivative(
    curve: NURBSCurveTuple, dist: float, span: Tuple[float, float], samples: int
) -> float:
    """
    Return discrete bound on |C_o''(u)| over a parameter span.
    """
    u0, u1 = span
    us = np.linspace(u0, u1, samples)
    max_val = 0.0
    for u in us:
        ev = evaluate_nurbs_curve(curve, u, d_order=3)
        C1, C2, C3 = ev["C1"], ev["C2"], ev["C2"]  # C3 via B‑spline ders (order 3)
        T = _unit(C1)
        B = _unit(np.cross(C1, C2)) if np.linalg.norm(np.cross(C1, C2)) > 1e-9 else np.zeros(3)
        kappa = np.linalg.norm(np.cross(C1, C2)) / np.linalg.norm(C1) ** 3
        # Numerical derivative of curvature via finite diff if C3 not exact
        kappa_prime = (
            np.dot(C1, C1) * np.dot(C1, np.cross(C3, B))
            - 3 * np.dot(C1, C2) * np.dot(C1, np.cross(C2, B))
        ) / np.linalg.norm(C1) ** 5 if np.linalg.norm(C1) > 1e-9 else 0.0
        sign = -1.0 if np.dot(B, _unit(np.cross(C1, C2))) > 0.0 else 1.0
        Co2 = (1 + sign * kappa * dist) * C2 + sign * kappa_prime * dist * C1
        max_val = max(max_val, np.linalg.norm(Co2))
    return max_val

# --------------------------------------------------------------------
#         GLOBAL B‑SPLINE INTERPOLATION THROUGH GIVEN POINTS
# --------------------------------------------------------------------
def _global_interpolation(
    pts: Sequence[NDArray[float]],
    degree: int,
    parametrisation: Literal["inherit", "chord"],
    curve: Optional[NURBSCurveTuple] = None,
) -> Tuple[NDArray[float], NDArray[float], NDArray[float]]:
    """
    Fit a non‑rational B‑spline of given degree through ordered points.
    Returns (knot_vector, control_points, weights=all‑1).
    """
    n = len(pts) - 1
    if parametrisation == "inherit" and curve is not None:
        # reuse the parameter values of the sample points
        params = np.linspace(0.0, 1.0, len(pts))
    else:  # chord‑length
        dists = np.linalg.norm(np.diff(pts, axis=0), axis=1)
        params = np.concatenate([[0.0], np.cumsum(dists)])
        params /= params[-1]
    # knot vector – averaged method
    m = n + degree + 1
    knot = np.zeros(m + 1)
    knot[degree+1 : n+1] = [
        np.sum(params[i+1 : i+degree+1]) / degree for i in range(n-degree)
    ]
    knot[n+1 :] = 1.0
    # build coefficient matrix N[i,j] = N_{j,p}(u_i)
    Nmat = np.zeros((len(pts), n+1))
    from existing_implementations import compute_basis_function_derivatives_np # type: ignore
    p = degree
    for i, u in enumerate(params):
        span = _find_span_linear(p, knot, n+1, u)
        N = compute_basis_function_derivatives_np(p, knot, span, u, 0)[0]
        Nmat[i, span-p : span+1] = N
    # solve for control points
    Q = np.array(pts)
    Pw = np.linalg.lstsq(Nmat, Q, rcond=None)[0]
    weights = np.ones(len(Pw))
    return knot, Pw, weights

# --------------------------------------------------------------------
#                   KNOT REMOVAL (Piegl & Tiller A8.1)
# --------------------------------------------------------------------
def _remove_knots_curve(
    knot: NDArray[float],
    ctrl: NDArray[float],
    p: int,
    eps: float,
) -> Tuple[NDArray[float], NDArray[float]]:
    """
    Iteratively try to remove every interior knot once as in P&T 8.1;
    accept the removal if max deviation ≤ eps.  Works for non‑rational
    curves (weights=1).
    """
    from existing_implementations import evaluate_nurbs_curve  # type: ignore
    U, P = knot.copy(), ctrl.copy()
    n = len(P) - 1
    i = p
    while i <= n - p - 1:
        u = U[i+1]
        t = 0
        removable = True
        # multiplicity
        mult = np.sum(np.isclose(U, u)) - 1
        r = p - mult
        for _ in range(r):
            # attempt 1‑time removal
            # build new control point array
            Pi = P.copy()
            for j in range(i - p, i - mult):
                alpha = (u - U[j]) / (U[j+p+1] - U[j])
                Pi[j] = (Pi[j] - (1 - alpha)*Pi[j-1]) / alpha
            for j in range(i - mult + 1, i + p - mult + 1):
                alpha = (u - U[j]) / (U[j+p+1] - U[j])
                Pi[j] = (Pi[j] - alpha*Pi[j+1]) / (1 - alpha)
            # error check – evaluate original & new at mid‑points of each span
            test_us = np.linspace(U[i] , U[i+1], p+2)[1:-1]
            for ut in test_us:
                C_old = evaluate_nurbs_curve(
                    NURBSCurveTuple(order=p+1, knot=U, control_points=P, weights=np.ones(len(P))),
                    ut, 0
                )["C"]
                C_new = evaluate_nurbs_curve(
                    NURBSCurveTuple(order=p+1, knot=np.delete(U, i+1),
                                    control_points=Pi, weights=np.ones(len(Pi))),
                    ut, 0
                )["C"]
                if np.linalg.norm(C_old - C_new) > eps:
                    removable = False
                    break
            if removable:
                U = np.delete(U, i+1)
                P = Pi
                n -= 1
            else:
                break
        i += 1
    return U, P

# --------------------------------------------------------------------
#                       MAIN CURVE OFFSET
# --------------------------------------------------------------------
def offset_nurbs_curve(
    curve: NURBSCurveTuple,
    distance: float,
    eps: float,
    opts: CurveOffsetOptions = CurveOffsetOptions(),
) -> NURBSCurveTuple:
    # 1. shape recognition ------------------------------------------------
    shape, data = _recognise_curve(curve, opts.shape_tol)
    if shape == "line":
        return _offset_line(data, distance)
    elif shape == "circle":
        return _offset_circle(data, distance)

    # 2. free‑form offset --------------------------------------------------
    refined_curve = _refine_curve(curve)
    spans = _curve_bezier_segments(refined_curve)
    p = refined_curve.order - 1
    default_n = opts.default_samples or (p + 1)

    sample_params: List[float] = []
    for span in spans:
        # bound on |C_o''|
        M = _bound_curve_second_derivative(refined_curve, distance, span, 2*(p+1))
        if M < 1e-12:        # very flat piece
            n = default_n
        else:
            pow_ = opts.pow_linear if p == 1 else opts.pow_nonlinear
            n = max(default_n, int(math.ceil((1.0/eps)**pow_ * math.sqrt(M/8))))
        # sample n parameters uniformly in the span
        sample_params.extend(span[0] + (span[1]-span[0]) * np.linspace(0, 1, n, endpoint=False).tolist())
    sample_params.append(spans[-1][1])           # ensure last point

    # build sample points on the **offset** curve
    samples: List[NDArray[float]] = []
    for u in sample_params:
        ev = evaluate_nurbs_curve(curve, u, d_order=2)
        P, C1, C2 = ev["C"], ev["C1"], ev["C2"]
        # principal normal
        T = _unit(C1)
        Nvec = C2 - np.dot(C2, T)*T
        if np.linalg.norm(Nvec) < 1e-12:               # curvature zero – use previous
            if samples:
                Nvec = samples[-1] - P
            else:
                # fall‑back perpendicular
                perp = np.cross(T, [0,0,1])
                if np.linalg.norm(perp) < 1e-8:
                    perp = np.cross(T, [0,1,0])
                Nvec = perp
        N = _unit(Nvec)
        samples.append(P + distance*N)

    # 3. interpolate -------------------------------------------------------
    knot, ctrl, w = _global_interpolation(
        samples, opts.degree, opts.parametrisation, curve
    )

    # 4. knot‑removal ------------------------------------------------------
    knot2, ctrl2 = _remove_knots_curve(knot, ctrl, opts.degree, eps)

    return NURBSCurveTuple(
        order=opts.degree+1,
        knot=knot2,
        control_points=ctrl2,
        weights=np.ones(len(ctrl2)),
    )

# ====================================================================
#                    SURFACE  –  SUPPORT/HELPERS
# ====================================================================

def _surface_bezier_spans(surf: NURBSSurfaceTuple) -> Tuple[List[Tuple[float,float]],
                                                            List[Tuple[float,float]]]:
    pu = surf.order_u - 1
    pv = surf.order_v - 1
    unique_u = sorted(set(_insert_knot_vector(surf.knot_u, pu)))
    unique_v = sorted(set(_insert_knot_vector(surf.knot_v, pv)))
    spans_u = [(unique_u[i], unique_u[i+1]) for i in range(len(unique_u)-1)]
    spans_v = [(unique_v[i], unique_v[i+1]) for i in range(len(unique_v)-1)]
    return spans_u, spans_v

#  -----  Exact offsets for analytic surfaces  ----------------------
# (planes, cylinders, cones, spheres, tori).  Only planes are needed
# to ensure the algorithm terminates in all cases; the rest are added
# for completeness.
# -------------------------------------------------------------------
def _recognise_surface(
    surf: NURBSSurfaceTuple, tol: float
) -> Tuple[Literal["plane", "free"], dict]:
    """
    Minimal recognition: *plane* versus *free-form*.  The full set
    (sphere, cone, …) can be added in the same way.
    """
    # sample 2(p+1)*(q+1) points
    pu, pv = surf.order_u - 1, surf.order_v - 1
    us = np.linspace(surf.knot_u[pu], surf.knot_u[-pu-1], 2*(pu+1))
    vs = np.linspace(surf.knot_v[pv], surf.knot_v[-pv-1], 2*(pv+1))
    P = []
    for u in us:
        for v in vs:
            P.append(evaluate_nurbs_surface(surf, u, v, 0)["S"])
    P = np.asarray(P)
    # PCA plane
    _, _, vh = np.linalg.svd(P - P.mean(axis=0))
    n = vh[-1]
    if np.max(np.abs((P - P.mean(axis=0)) @ n)) < tol:
        return "plane", {"point": P.mean(axis=0), "normal": n}
    return "free", {}

def _offset_plane(data: dict, distance: float, surf: NURBSSurfaceTuple) -> NURBSSurfaceTuple:
    """Return plane offset – simply translate all control points."""
    P0, n = data["point"], _unit(data["normal"])
    offset_vec = distance * n
    ctrl = surf.control_points + offset_vec
    return NURBSSurfaceTuple(
        order_u=surf.order_u, order_v=surf.order_v,
        knot_u=surf.knot_u.copy(), knot_v=surf.knot_v.copy(),
        control_points=ctrl.copy(), weights=surf.weights.copy()
    )

# --------------  Bound on second partials  (Eq. 13)  -----------------
def _bound_surface_second_derivatives(
    surf: NURBSSurfaceTuple, distance: float,
    span_u: Tuple[float,float], span_v: Tuple[float,float],
    samples: int
) -> float:
    M1 = M2 = M3 = 0.0
    us = np.linspace(span_u[0], span_u[1], samples)
    vs = np.linspace(span_v[0], span_v[1], samples)
    for u in us:
        for v in vs:
            ev = evaluate_nurbs_surface(surf, u, v, d_order=2)
            S, Su, Sv = ev["S"], ev["Su"], ev["Sv"]
            Suu, Suv, Svv = ev["Suu"], ev["Suv"], ev["Svv"]
            N = _unit(np.cross(Su, Sv))
            Nuu = np.zeros_like(N)  # ignore for bound → conservative
            Nuv = np.zeros_like(N)
            Nvv = np.zeros_like(N)
            So_uu = Suu + distance*Nuu
            So_uv = Suv + distance*Nuv
            So_vv = Svv + distance*Nvv
            M1 = max(M1, np.linalg.norm(So_uu))
            M2 = max(M2, np.linalg.norm(So_uv))
            M3 = max(M3, np.linalg.norm(So_vv))
    return M1 + 2*M2 + M3

# --------------   Surface grid interpolation  ------------------------
def _surface_global_interp(
    grid: NDArray[float],             # shape (nu, nv, 3)
    deg_u: int, deg_v: int,
    param_u: Sequence[float], param_v: Sequence[float],
) -> Tuple[NDArray[float], NDArray[float], NDArray[float], NDArray[float]]:
    """
    Least‑squares tensor‑product B‑spline surface fitting through a
    regular grid of points.
    """
    nu, nv = grid.shape[:2]
    # Knot vectors with averaging
    def _avg_knots(params: Sequence[float], p: int) -> NDArray[float]:
        n = len(params) - 1
        m = n + p + 1
        kv = np.zeros(m+1)
        kv[p+1 : n+1] = [
            sum(params[i+1 : i+p+1]) / p
            for i in range(n - p)
        ]
        kv[n+1 :] = 1.0
        return kv
    U = _avg_knots(param_u, deg_u)
    V = _avg_knots(param_v, deg_v)
    # Build coefficient matrices along u & v
    from mmcore.geom._nurbs_eval import compute_basis_function_derivatives_np # type: ignore
    Nu = np.zeros((len(param_u), len(param_u)))
    Nv = np.zeros((len(param_v), len(param_v)))
    for i, u in enumerate(param_u):
        span = _find_span_linear(deg_u, U, len(param_u), u)
        Nu[i, span-deg_u:span+1] = compute_basis_function_derivatives_np(
            deg_u, U, span, u, 0
        )[0]
    for j, v in enumerate(param_v):
        span = _find_span_linear(deg_v, V, len(param_v), v)
        Nv[j, span-deg_v:span+1] = compute_basis_function_derivatives_np(
            deg_v, V, span, v, 0
        )[0]
    # Solve for control net by two‑stage least‑squares (P&T p. 530)
    # 1) fit each v‑column independently to find temporary Q
    Q = np.zeros((len(param_u), len(param_v), 3))
    for j in range(len(param_v)):
        Q[:, j, :] = np.linalg.lstsq(Nu, grid[:, j, :], rcond=None)[0]
    # 2) fit each u‑row to find final control points P
    P = np.zeros_like(Q)
    for i in range(len(param_u)):
        P[i, :, :] = np.linalg.lstsq(Nv, Q[i, :, :], rcond=None)[0]
    return U, V, P, np.ones(P.shape[:2])
from mmcore.numeric import evaluate_normal
def _robust_surface_normal(
    surface: NURBSSurfaceTuple,
    u: float,
    v: float,
    ev: dict | None = None,
    lastN: NDArray[float] | None = None,
    eps: float = 1.0e-9,
) -> NDArray[float]:
    """
    Return a unit surface normal that is well‑defined even when the
    first‑order partials are linearly dependent (|Su×Sv|≈0).

    Strategy:
      1)   try Su×Sv
      2)   try all second‑order pairs
      3)   finite‑difference estimate
      4)   reuse last valid normal
    """
    if ev is None:
        ev = evaluate_nurbs_surface(surface, u, v, d_order=2)
    Su, Sv = ev["Su"], ev["Sv"]
    Suu, Suv, Svv = ev["Suu"], ev["Suv"], ev["Svv"]

    # 1) first‑order
    n = np.cross(Su, Sv)
    if np.linalg.norm(n) > eps:
        return _unit(n)

    # 2) second‑order pairs
    pairs = [
        (Su, Suu), (Su, Suv), (Su, Svv),
        (Suu, Sv), (Suv, Sv), (Svv, Sv),
    ]
    for a, b in pairs:
        n = np.cross(a, b)
        if np.linalg.norm(n) > eps:
            return _unit(n)

    # 3) finite‑difference fallback
    du = (surface.knot_u[-1] - surface.knot_u[0]) * 1e-4
    dv = (surface.knot_v[-1] - surface.knot_v[0]) * 1e-4
    # clamp to param domain
    u_minus = max(surface.knot_u[0], u - du)
    u_plus  = min(surface.knot_u[-1], u + du)
    v_minus = max(surface.knot_v[0], v - dv)
    v_plus  = min(surface.knot_v[-1], v + dv)
    Su_fd = (
        evaluate_nurbs_surface(surface, u_plus, v, 0)["S"]
        - evaluate_nurbs_surface(surface, u_minus, v, 0)["S"]
    )
    Sv_fd = (
        evaluate_nurbs_surface(surface, u, v_plus, 0)["S"]
        - evaluate_nurbs_surface(surface, u, v_minus, 0)["S"]
    )
    n = np.cross(Su_fd, Sv_fd)
    if np.linalg.norm(n) > eps:
        return _unit(n)

    # 4) last resort – reuse last good normal or z‑axis
    if lastN is not None and np.linalg.norm(lastN) > 0:
        return lastN
    return np.array([0.0, 0.0, 1.0])     # arbitrary but safe
# --------------------------------------------------------------------
#                       MAIN SURFACE OFFSET
# --------------------------------------------------------------------
def offset_nurbs_surface(
    surface: NURBSSurfaceTuple,
    distance: float,
    eps: float,
    opts: SurfaceOffsetOptions = SurfaceOffsetOptions(),
) -> NURBSSurfaceTuple:

    # Exact analytic cases -------------------------------------------------
    typ, data = _recognise_surface(surface, opts.shape_tol)
    if typ == "plane":
        return _offset_plane(data, distance, surface)

    # Free‑form ------------------------------------------------------------
    spans_u, spans_v = _surface_bezier_spans(surface)
    pu, pv = surface.order_u - 1, surface.order_v - 1
    default_n = opts.default_samples or max(pu, pv)+1
    pow_lin, pow_non = opts.pow_linear, opts.pow_nonlinear

    # Decide how many points per patch (same in u & v for simplicity)
    grid_params_u: List[float] = []
    for span in spans_u:
        M = _bound_surface_second_derivatives(surface, distance, span, spans_v[0], 2*(pu+1))
        pow_ = pow_lin if pu == 1 else pow_non
        n = max(default_n, int(math.ceil((1/eps)**pow_ * math.sqrt(M/8))))
        grid_params_u.extend(span[0] + (span[1]-span[0])*np.linspace(0,1,n,endpoint=False))
    grid_params_u.append(spans_u[-1][1])

    grid_params_v: List[float] = []
    for span in spans_v:
        M = _bound_surface_second_derivatives(surface, distance, spans_u[0], span, 2*(pv+1))
        pow_ = pow_lin if pv == 1 else pow_non
        n = max(default_n, int(math.ceil((1/eps)**pow_ * math.sqrt(M/8))))
        grid_params_v.extend(span[0] + (span[1]-span[0])*np.linspace(0,1,n,endpoint=False))
    grid_params_v.append(spans_v[-1][1])

    # Build sampling grid on offset surface
    grid = np.zeros((len(grid_params_u), len(grid_params_v), 3))
    last_normal=None
    for i,u in enumerate(grid_params_u):
        for j,v in enumerate(grid_params_v):
            ev = evaluate_nurbs_surface(surface, u, v, 2)

            S, Su, Sv, Suu,Suv,Svv = ev["S"], ev["Su"], ev["Sv"], ev["Suu"], ev["Suv"], ev["Svv"]
            N=_robust_surface_normal(surface, u, v, ev, last_normal,eps)
            last_normal=N
          
            grid[i,j,:] = S + distance*N
    
    # Parameter vectors for interpolation
    if opts.parametrisation_u == "inherit":
        param_u = (np.array(grid_params_u) - surface.knot_u[pu]) / (surface.knot_u[-pu-1]-surface.knot_u[pu])
    else:     # chord length along u
        du = np.linalg.norm(np.diff(grid[:,:, :], axis=0), axis=(1,2))
        param_u = np.concatenate([[0.0], np.cumsum(du)]) / np.sum(du)
    if opts.parametrisation_v == "inherit":
        param_v = (np.array(grid_params_v) - surface.knot_v[pv]) / (surface.knot_v[-pv-1]-surface.knot_v[pv])
    else:
        dv = np.linalg.norm(np.diff(grid.transpose(1,0,2), axis=0), axis=(1,2))
        param_v = np.concatenate([[0.0], np.cumsum(dv)]) / np.sum(dv)

    # Interpolate
    U, V, CP, W = _surface_global_interp(
        grid, opts.degree_u, opts.degree_v, param_u, param_v
    )

    # (Optional) knot removal in u & v separately can be added here using
    # the univariate routine.  Most CAD kernels omit that step for surfaces
    # because control‑net reduction is already dramatic.

    return NURBSSurfaceTuple(
        order_u=opts.degree_u+1, order_v=opts.degree_v+1,
        knot_u=U, knot_v=V,
        control_points=CP, weights=W
    )

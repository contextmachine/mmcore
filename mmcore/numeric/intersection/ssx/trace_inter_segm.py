from __future__ import annotations

import numpy as np
from numpy._typing import NDArray

from mmcore.geom._nurbs_eval import evaluate_nurbs_surface, NURBSCurveTuple,NURBSSurfaceTuple

# Reuse your existing homogeneous helpers and derivative net extractors
# ... (Keep your imports and _eval_tensor_bezier, etc.)
# ======================================================================================
# Leaf branch construction: adaptive 4D cubic fitting (few points, tol-driven)
# Drop-in replacement for interp_param_space_hermite() usage.
# ======================================================================================

import numpy as np
from numpy.typing import NDArray

from mmcore.geom._nurbs_knots import link_curves
from mmcore.numeric.intersection.ssx.refine import refine_intersection_point
from mmcore.numeric.sbern import bern_to_nurbs_bezier


# ---- fast rational Bezier patch evaluator on an interval (global uv <-> local [0,1]^2) ----

class _BezierPatchEval:
    __slots__ = ("H", "interval", "_Hu", "_Hv", "_inv_du", "_inv_dv")

    def __init__(self, H: NDArray[np.float64], interval: tuple[float, float, float, float]):
        self.H = np.asarray(H, dtype=np.float64)  # (p+1,q+1,4) homogeneous
        self.interval = tuple(map(float, interval))
        self._Hu = _derivative_net_u(self.H)
        self._Hv = _derivative_net_v(self.H)

        u0, u1, v0, v1 = self.interval
        du = (u1 - u0)
        dv = (v1 - v0)
        self._inv_du = 1.0 / du if abs(du) > 1e-30 else 0.0
        self._inv_dv = 1.0 / dv if abs(dv) > 1e-30 else 0.0

    def _to_local(self, u: float, v: float) -> tuple[float, float]:
        u0, u1, v0, v1 = self.interval
        ul = (float(u) - u0) * self._inv_du if self._inv_du != 0.0 else 0.0
        vl = (float(v) - v0) * self._inv_dv if self._inv_dv != 0.0 else 0.0
        # Keep eval stable
        ul = float(np.clip(ul, 0.0, 1.0))
        vl = float(np.clip(vl, 0.0, 1.0))
        return ul, vl

    def eval_point(self, u: float, v: float) -> NDArray[np.float64]:
        ul, vl = self._to_local(u, v)
        H = _eval_tensor_bezier(self.H, ul, vl)  # (4,)
        W = float(H[3])
        if abs(W) <= 1e-14:
            return H[:3].copy()
        return (H[:3] / W).copy()

    def eval_point_and_partials(self, u: float, v: float) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """
        Returns (F, Fu, Fv) in Euclidean 3D, where Fu,Fv are derivatives w.r.t *global* (u,v).
        """
        ul, vl = self._to_local(u, v)

        H = _eval_tensor_bezier(self.H, ul, vl)
        Hu = _eval_tensor_bezier(self._Hu, ul, vl)
        Hv = _eval_tensor_bezier(self._Hv, ul, vl)

        W = float(H[3])
        if abs(W) <= 1e-14:
            F = H[:3].copy()
            return F, np.zeros(3), np.zeros(3)

        X = H[:3]
        Xu = Hu[:3]
        Xv = Hv[:3]
        Wu = float(Hu[3])
        Wv = float(Hv[3])

        invW = 1.0 / W
        F = X * invW

        invW2 = invW * invW
        Fu_local = (Xu * W - X * Wu) * invW2
        Fv_local = (Xv * W - X * Wv) * invW2

        # chain rule: ul=(u-u0)/(u1-u0) => d/du = d/dul * (1/(u1-u0))
        Fu = Fu_local * self._inv_du
        Fv = Fv_local * self._inv_dv
        return F, Fu, Fv


# ---- small linear algebra helpers (faster than pinv for 3x2) ----

def _normalize(v: NDArray[np.float64], eps: float = 1e-30) -> NDArray[np.float64]:
    n = float(np.linalg.norm(v))
    if n <= eps:
        return v * 0.0
    return v / n

def _lsq_param_dir(Fu: NDArray[np.float64], Fv: NDArray[np.float64], T: NDArray[np.float64], eps: float = 1e-18) -> NDArray[np.float64]:
    """
    Solve min || [Fu Fv] * d - T || for d in R^2 using normal equations (2x2).
    Falls back to pinv if ill-conditioned.
    """
    a11 = float(Fu @ Fu); a12 = float(Fu @ Fv); a22 = float(Fv @ Fv)
    b1 = float(Fu @ T);  b2 = float(Fv @ T)

    det = a11 * a22 - a12 * a12
    if abs(det) <= eps:
        J = np.column_stack((Fu, Fv))
        return (np.linalg.pinv(J) @ T).astype(np.float64)

    inv = 1.0 / det
    d0 = ( a22 * b1 - a12 * b2) * inv
    d1 = (-a12 * b1 + a11 * b2) * inv
    return np.array([d0, d1], dtype=np.float64)


# ---- cubic Bezier in stuv-space ----

def _cubic_bezier_eval(ctrl: NDArray[np.float64], t: float) -> NDArray[np.float64]:
    t = float(t)
    omt = 1.0 - t
    b0 = omt * omt * omt
    b1 = 3.0 * omt * omt * t
    b2 = 3.0 * omt * t * t
    b3 = t * t * t
    return (b0 * ctrl[0] + b1 * ctrl[1] + b2 * ctrl[2] + b3 * ctrl[3]).astype(np.float64)

def _line_as_cubic(P0: NDArray[np.float64], P1: NDArray[np.float64]) -> NDArray[np.float64]:
    P0 = np.asarray(P0, dtype=np.float64)
    P1 = np.asarray(P1, dtype=np.float64)
    d = (P1 - P0)
    return np.stack([P0, P0 + d / 3.0, P0 + 2.0 * d / 3.0, P1], axis=0)

def _make_bezier_curve_stuv(ctrl4: NDArray[np.float64]) -> NURBSCurveTuple:
    """
    ctrl4: (4,4) cubic Bezier control points in stuv-space.
    Returns cubic B-spline (order=4) with clamped knots.
    """
    knot = np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0], dtype=np.float64)
    w = np.ones(4, dtype=np.float64)
    return NURBSCurveTuple(4, knot=knot, control_points=np.asarray(ctrl4, dtype=np.float64), weights=w)


# ---- segment candidate construction + error test ----

def _segment_max_gap(
    eval1: _BezierPatchEval,
    eval2: _BezierPatchEval,
    ctrl: NDArray[np.float64],
    ts: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> tuple[float, float]:
    worst_e = -1.0
    worst_t = 0.5
    for t in ts:
        stuv = _cubic_bezier_eval(ctrl, t)
        p = eval1.eval_point(stuv[0], stuv[1])
        q = eval2.eval_point(stuv[2], stuv[3])
        e = float(np.linalg.norm(p - q))
        if e > worst_e:
            worst_e = e
            worst_t = float(t)
    return worst_e, worst_t

def _stuv_tangent_from_surfaces(
    eval1: _BezierPatchEval,
    eval2: _BezierPatchEval,
    stuv: NDArray[np.float64],
    chord_dir_xyz: NDArray[np.float64],
    *,
    tangency_eps: float = 1e-12,
) -> NDArray[np.float64] | None:
    """
    Compute 4D parameter-space tangent direction at an intersection point.
    Returns dstuv per unit 3D step (since T is unit), or None if tangency/degenerate.
    """
    stuv = np.asarray(stuv, dtype=np.float64)
    F, Fu, Fv = eval1.eval_point_and_partials(stuv[0], stuv[1])
    G, Gu, Gv = eval2.eval_point_and_partials(stuv[2], stuv[3])

    n1 = np.cross(Fu, Fv)
    n2 = np.cross(Gu, Gv)
    n1n = float(np.linalg.norm(n1))
    n2n = float(np.linalg.norm(n2))
    if n1n <= tangency_eps or n2n <= tangency_eps:
        return None
    n1 /= n1n
    n2 /= n2n

    T = np.cross(n1, n2)
    Tn = float(np.linalg.norm(T))
    if Tn <= tangency_eps:
        return None
    T /= Tn

    # orient tangent to roughly follow chord direction in xyz
    cd = _normalize(np.asarray(chord_dir_xyz, dtype=np.float64))
    if float(cd @ cd) > 0.0 and float(T @ cd) < 0.0:
        T = -T

    d1 = _lsq_param_dir(Fu, Fv, T)
    d2 = _lsq_param_dir(Gu, Gv, T)

    dstuv = np.array([d1[0], d1[1], d2[0], d2[1]], dtype=np.float64)

    # orient in stuv space toward chord end if possible
    # (this is heuristic but works well in simple leaves)
    return dstuv


def _cubic_from_endpoints_and_tangents(
    P0: NDArray[np.float64],
    P1: NDArray[np.float64],
    t0: NDArray[np.float64] | None,
    t1: NDArray[np.float64] | None,
    L: float,
    interval1: tuple[float, float, float, float],
    interval2: tuple[float, float, float, float],
) -> NDArray[np.float64]:
    """
    Build cubic Bezier ctrl points in stuv-space.
    If tangents are None, falls back to straight line cubic.
    """
    P0 = np.asarray(P0, dtype=np.float64)
    P1 = np.asarray(P1, dtype=np.float64)

    if (t0 is None) or (t1 is None) or (not np.all(np.isfinite(t0))) or (not np.all(np.isfinite(t1))):
        ctrl = _line_as_cubic(P0, P1)
        # keep everything inside intervals
        for i in range(4):
            ctrl[i] = _clamp_stuv(ctrl[i], interval1, interval2)
        return ctrl

    # chord-length scale
    h = float(L) / 3.0
    C0 = P0
    C3 = P1
    C1 = P0 + h * t0
    C2 = P1 - h * t1

    ctrl = np.stack([C0, C1, C2, C3], axis=0)
    for i in range(4):
        ctrl[i] = _clamp_stuv(ctrl[i], interval1, interval2)
    return ctrl


# ---- refinement step (corrector) ----

def _refine_stuv(
    stuv_guess: NDArray[np.float64],
    s1,
    s2,
    *,
    spt: float,
    max_iter: int = 50,
) -> NDArray[np.float64] | None:
    """
    Use your existing corrector to project stuv to the true intersection.
    """
    g = np.asarray(stuv_guess, dtype=np.float64)
    try:
        stuv_new, *_ = refine_intersection_point(g, s1, s2, spt=spt, max_iter=max_iter)
    except Exception:
        return None
    if stuv_new is None:
        return None
    stuv_new = np.asarray(stuv_new, dtype=np.float64)
    if stuv_new.shape != (4,) or not np.all(np.isfinite(stuv_new)):
        return None
    return stuv_new


# ---- adaptive fitter (tol-driven, few points) ----

def _fit_span_adaptive(
    s1,
    s2,
    eval1: _BezierPatchEval,
    eval2: _BezierPatchEval,
    P0: NDArray[np.float64],
    P1: NDArray[np.float64],
    interval1: tuple[float, float, float, float],
    interval2: tuple[float, float, float, float],
    *,
    spt: float,
    max_depth: int,
    depth: int = 0,
) -> list[NURBSCurveTuple]:
    """
    Returns list of cubic Bezier NURBS segments in stuv-space approximating the intersection
    within absolute tol spt (checked by ||S1 - S2|| samples).
    """

    P0 = np.asarray(P0, dtype=np.float64)
    P1 = np.asarray(P1, dtype=np.float64)

    # cheap accept: straight line if it already satisfies tol
    ctrl_line = _line_as_cubic(P0, P1)
    for i in range(4):
        ctrl_line[i] = _clamp_stuv(ctrl_line[i], interval1, interval2)
    e_line, t_bad_line = _segment_max_gap(eval1, eval2, ctrl_line)
    if e_line <= spt or depth >= max_depth:
        return [_make_bezier_curve_stuv(ctrl_line)]

    # attempt a single cubic using endpoint tangents
    X0 = eval1.eval_point(P0[0], P0[1])
    X1 = eval1.eval_point(P1[0], P1[1])
    chord = X1 - X0
    L = float(np.linalg.norm(chord))
    if not np.isfinite(L) or L <= 1e-30:
        # degenerate span; fall back to line
        return [_make_bezier_curve_stuv(ctrl_line)]

    t0 = _stuv_tangent_from_surfaces(eval1, eval2, P0, chord)
    t1 = _stuv_tangent_from_surfaces(eval1, eval2, P1, chord)

    ctrl_cubic = _cubic_from_endpoints_and_tangents(P0, P1, t0, t1, L, interval1, interval2)
    e_cubic, t_bad = _segment_max_gap(eval1, eval2, ctrl_cubic)

    if e_cubic <= spt:
        return [_make_bezier_curve_stuv(ctrl_cubic)]

    # subdivide at worst sample location (usually mid)
    t_split = float(t_bad)
    Pm_guess = _cubic_bezier_eval(ctrl_cubic, t_split)
    Pm = _refine_stuv(Pm_guess, s1, s2, spt=spt, max_iter=60)
    if Pm is None:
        # if corrector fails, try splitting at 0.5 using line guess
        Pm_guess = _cubic_bezier_eval(ctrl_line, 0.5)
        Pm = _refine_stuv(Pm_guess, s1, s2, spt=spt, max_iter=60)
        if Pm is None:
            # last resort: accept line (leaf is tiny anyway)
            return [_make_bezier_curve_stuv(ctrl_line)]

    Pm = _clamp_stuv(Pm, interval1, interval2)

    # avoid infinite recursion on nearly identical points
    if float(np.linalg.norm(Pm - P0)) <= 1e-14 or float(np.linalg.norm(P1 - Pm)) <= 1e-14:
        return [_make_bezier_curve_stuv(ctrl_line)]

    left = _fit_span_adaptive(
        s1, s2, eval1, eval2, P0, Pm, interval1, interval2,
        spt=spt, max_depth=max_depth, depth=depth + 1,
    )
    right = _fit_span_adaptive(
        s1, s2, eval1, eval2, Pm, P1, interval1, interval2,
        spt=spt, max_depth=max_depth, depth=depth + 1,
    )
    return [*left, *right]


def fit_intersection_branch_adaptive(
    H1: NDArray[np.float64],
    H2: NDArray[np.float64],
    interval1: tuple[float, float, float, float],
    interval2: tuple[float, float, float, float],
    stuv_a: NDArray[np.float64],
    stuv_b: NDArray[np.float64],
    *,
    spt: float,
    max_depth: int = 10,
) -> NURBSCurveTuple:
    """
    Build a (typically very small) stuv-space curve for one leaf branch segment.

    - Correct endpoints once with refine_intersection_point
    - Adaptive cubic fitting with tol check by ||S1 - S2|| at a few samples
    - Returns a single linked NURBS curve (piecewise cubic)
    """
    # NURBS surfaces for your corrector (projection)
    s1 = bern_to_nurbs_bezier(H1, interval=((interval1[0], interval1[1]), (interval1[2], interval1[3])), rational=True)
    s2 = bern_to_nurbs_bezier(H2, interval=((interval2[0], interval2[1]), (interval2[2], interval2[3])), rational=True)

    # fast evaluators for tol checks + tangents
    eval1 = _BezierPatchEval(H1, interval1)
    eval2 = _BezierPatchEval(H2, interval2)

    Pa = _refine_stuv(np.asarray(stuv_a, dtype=np.float64), s1, s2, spt=spt, max_iter=60)
    Pb = _refine_stuv(np.asarray(stuv_b, dtype=np.float64), s1, s2, spt=spt, max_iter=60)
    if Pa is None:
        Pa = np.asarray(stuv_a, dtype=np.float64)
    if Pb is None:
        Pb = np.asarray(stuv_b, dtype=np.float64)

    Pa = _clamp_stuv(Pa, interval1, interval2)
    Pb = _clamp_stuv(Pb, interval1, interval2)

    segs = _fit_span_adaptive(
        s1, s2, eval1, eval2,
        Pa, Pb,
        interval1, interval2,
        spt=spt,
        max_depth=max_depth,
        depth=0,
    )

    if not segs:
        # fallback: straight
        return _make_bezier_curve_stuv(_line_as_cubic(Pa, Pb))

    if len(segs) == 1:
        return segs[0]

    # stitch
    curve, _ = link_curves(segs)
    return curve


# ======================================================================================
# Replace your current _simple_march_between with this implementation
# ======================================================================================

def trace_between(
    H1,
    H2,
    stuv_a: NDArray[np.float64],
    stuv_b: NDArray[np.float64],
    *,
    interval1: tuple[float, float, float, float],
    interval2: tuple[float, float, float, float],
    spt: float,
    fit_max_depth: int = 10,
) -> NURBSCurveTuple:
    return fit_intersection_branch_adaptive(
        H1, H2,
        interval1, interval2,
        stuv_a, stuv_b,
        spt=spt,
        max_depth=fit_max_depth,
    )


# ======================================================================================
# Small call-site change inside _leaf_boundary_test_and_march()
# (replace the old s1/s2 construction + interp_param_space_hermite call)
# ======================================================================================
# In _leaf_boundary_test_and_march(), replace the "If len(points) >= 2:" block with:

#    if len(points) >= 2:
#        pairs, leftover = _pair_points_by_nearest(points)
#        if pairs:
#            for i, j in pairs:
#                curve = _simple_march_between(
#                    H1,
#                    H2,
#                    points[i].stuv,
#                    points[j].stuv,
#                    interval1=interval1,
#                    interval2=interval2,
#                    spt=spt,
#                    fit_max_depth=10,   # tune (8..12 is usually enough)
#                )
#                branches.append(SSXBranch(curve=curve))
#        points = [points[k] for k in leftover]


def _from_homogeneous(H: NDArray[np.float64]) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    H = np.asarray(H, dtype=np.float64)
    w = H[..., -1]
    with np.errstate(divide="ignore", invalid="ignore"):
        P = H[..., :-1] / w[..., None]
    return P, w


def _to_homogeneous(P: NDArray[np.float64], w: NDArray[np.float64]) -> NDArray[np.float64]:
    P = np.asarray(P, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    if P.shape[:-1] != w.shape:
        raise ValueError(f"weights shape {w.shape} must match control_points[...,0] shape {P.shape[:-1]}")
    return np.concatenate((P * w[..., None], w[..., None]), axis=-1)


def _eval_tensor_bezier(ctrl: NDArray[np.float64], u: float, v: float) -> NDArray[np.float64]:
    """
    Evaluate tensor-product Bezier control net at (u,v) using de Casteljau.
    ctrl: (p+1, q+1, dim)
    """
    B = np.array(ctrl, dtype=np.float64, copy=True)
    p = B.shape[0] - 1
    q = B.shape[1] - 1

    # de Casteljau in u (vectorized over v,dim)
    omt = 1.0 - u
    for r in range(1, p + 1):
        B[: p - r + 1] = omt * B[: p - r + 1] + u * B[1 : p - r + 2]

    # de Casteljau in v on the reduced curve
    C = B[0].copy()  # (q+1, dim)
    omt = 1.0 - v
    for r in range(1, q + 1):
        C[: q - r + 1] = omt * C[: q - r + 1] + v * C[1 : q - r + 2]

    return C[0]


def _derivative_net_u(H: NDArray[np.float64]) -> NDArray[np.float64]:
    p = H.shape[0] - 1
    if p <= 0:
        return np.zeros((1, H.shape[1], H.shape[2]), dtype=np.float64)
    return p * (H[1:, :, :] - H[:-1, :, :])


def _derivative_net_v(H: NDArray[np.float64]) -> NDArray[np.float64]:
    q = H.shape[1] - 1
    if q <= 0:
        return np.zeros((H.shape[0], 1, H.shape[2]), dtype=np.float64)
    return q * (H[:, 1:, :] - H[:, :-1, :])


def _clamp_stuv(
    stuv: NDArray[np.float64],
    interval1: tuple[float, float, float, float],
    interval2: tuple[float, float, float, float],
) -> NDArray[np.float64]:
    u0, u1, v0, v1 = interval1
    u2, u3, v2, v3 = interval2
    out = np.array(stuv, dtype=np.float64)
    out[0] = np.clip(out[0], u0, u1)
    out[1] = np.clip(out[1], v0, v1)
    out[2] = np.clip(out[2], u2, u3)
    out[3] = np.clip(out[3], v2, v3)
    return out

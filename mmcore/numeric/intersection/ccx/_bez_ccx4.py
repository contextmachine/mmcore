"""Bezier curve-curve intersection using squared-distance Bernstein net classification.

This module implements a subdivision-based CCX algorithm that uses the
squared-distance net ``||C1(u) - C2(v)||^2`` in Bernstein form to classify
cells as NO_INTERSECTION, UNIQUE_ISOLATED, OVERLAP, or INDETERMINATE,
avoiding explicit Jacobian-rank analysis.
"""
from __future__ import annotations

import math

import numpy as np

from mmcore.numeric.aabb import aabb_offset
from mmcore.numeric.aabb import aabb_intersect,aabb
from mmcore.numeric.bern import de_casteljau_split_nd
from mmcore.numeric.bern_sq_dist import curve_curve_squared_net_homog
from mmcore.numeric.intersection._bezier_common import extract_weights, eval_curve, eval_curve_d1, newton_ccx
from mmcore.numeric.intersection._sq_dist_classify import (
    classify_sq_dist_net,
    NO_INTERSECTION,
    UNIQUE_ISOLATED,
    OVERLAP,
    INDETERMINATE,
    BOUNDARY_ZERO,
    BoundaryZero, _boundary_zero_to_param_point,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subdivide_curve(ctrl, t=0.5):
    """Split a Bezier curve at parameter t using de Casteljau.

    Parameters
    ----------
    ctrl : ndarray, shape (n+1, D)
        Control polygon of a degree-n Bezier curve.
    t : float
        Split parameter in [0, 1].

    Returns
    -------
    left, right : ndarray
        Control polygons of the two halves.
    """
    n = ctrl.shape[0] - 1
    tmp = ctrl.copy()
    left = [tmp[0].copy()]
    right_rev = [tmp[n].copy()]
    for r in range(1, n + 1):
        tmp[: n + 1 - r] = (1.0 - t) * tmp[: n + 1 - r] + t * tmp[1 : n + 2 - r]
        left.append(tmp[0].copy())
        right_rev.append(tmp[n - r].copy())
    return np.array(left), np.array(right_rev[::-1])


def _subdivide_sq_dist_net(F, axis, t=0.5):
    """Subdivide the scalar sq-dist Bernstein net along *axis*.

    ``de_casteljau_split_nd`` requires a trailing value dimension, so we
    temporarily add one and squeeze it back off.
    """
    Fv = F[..., np.newaxis]
    left_v, right_v = de_casteljau_split_nd(Fv, axis=axis, t=t)
    return left_v[..., 0], right_v[..., 0]

def _subdivide_sq_dist_net_2d(F, u=0.5,v=0.5):
    """Subdivide the scalar sq-dist Bernstein net along *axis*.

    ``de_casteljau_split_nd`` requires a trailing value dimension, so we
    temporarily add one and squeeze it back off.
    """
    Fv = F[..., np.newaxis]

    left_v, right_v = de_casteljau_split_nd(Fv, axis=1, t=u)
    left_u_left_v,right_u_left_v=de_casteljau_split_nd(left_v, axis=0, t=v)
    left_u_right_v,right_u_right_v =de_casteljau_split_nd(right_v, axis=0, t=v)
    return left_u_left_v[..., 0], left_u_right_v[..., 0],right_u_right_v[..., 0],right_u_left_v[..., 0]

def _boundary_zero_to_uv(bz: BoundaryZero, u0: float, u1: float, v0: float, v1: float) -> tuple[float, float]:
    """Convert a BoundaryZero from the classifier into global (u, v) parameters.

    For a 2D CCX net:
    - axis=0, side=0 → u=u0, v = v0 + bz.param * (v1 - v0)
    - axis=0, side=1 → u=u1, v = v0 + bz.param * (v1 - v0)
    - axis=1, side=0 → v=v0, u = u0 + bz.param * (u1 - u0)
    - axis=1, side=1 → v=v1, u = u0 + bz.param * (u1 - u0)
    """
    if bz.axis == 0:
        u = u0 if bz.side == 0 else u1
        v = v0 + bz.param * (v1 - v0)
    else:
        v = v0 if bz.side == 0 else v1
        u = u0 + bz.param * (u1 - u0)
    return u, v


def _compute_param_tols(C1, C2, atol, rational):
    """Compute parametric tolerances for both curves using curve geometry.

    Returns (tol_u, tol_v) — the maximum parameter perturbation in each
    curve that corresponds to geometric deviation <= atol.
    """
    from mmcore.geom._nurbs_param_tol import bez_curve_param_tolerance
    tol_u = bez_curve_param_tolerance(C1, tol=atol, rational=rational)
    tol_v = bez_curve_param_tolerance(C2, tol=atol, rational=rational)

    return float(tol_u), float(tol_v)


def _curve_component_scale(C, rational):
    """Per-coordinate Cartesian scale used only for roundoff certification.

    This is deliberately component-wise.  A nonzero separation in a quiet
    coordinate (for example ``z=0`` versus ``z=dz``) must not inherit the
    much larger x/y scale and become an ``atol``-sized false root.
    """
    C = np.asarray(C, dtype=np.float64)
    if rational:
        weights = C[:, -1]
        if (not np.all(np.isfinite(weights))
                or np.any(weights == 0.0)):
            return None
        points = C[:, :-1] / weights[:, None]
    else:
        points = C
    if not np.all(np.isfinite(points)):
        return None
    return np.max(np.abs(points), axis=0)


def _cartesian_curve_controls_for_exactness(C, rational):
    C = np.asarray(C, dtype=np.float64)
    if rational:
        weights = C[:, -1:]
        if (not np.all(np.isfinite(weights))
                or np.any(weights == 0.0)):
            return None
        points = C[:, :-1] / weights
    else:
        points = C
    if not np.all(np.isfinite(points)):
        return None
    return points


def _center_curve_homogeneous_for_exactness(C, rational, origin):
    """Return a stably normalized homogeneous curve translated by origin."""
    C = np.asarray(C, dtype=np.float64)
    if rational:
        H = C.copy()
    else:
        H = np.concatenate(
            [C, np.ones((len(C), 1), dtype=np.float64)], axis=1)
    scale = float(np.max(np.abs(H)))
    if not np.isfinite(scale) or scale <= 0.0:
        return None
    H /= scale
    H[:, :-1] -= np.asarray(origin, dtype=np.float64) * H[:, -1:]
    local_scale = float(np.max(np.abs(H)))
    if not np.isfinite(local_scale) or local_scale <= 0.0:
        return None
    return np.ascontiguousarray(H / local_scale)


def _ccx_exactness_context(C1, C2, rational):
    """Common-origin context for translation-invariant equality tests."""
    points1 = _cartesian_curve_controls_for_exactness(C1, rational)
    points2 = _cartesian_curve_controls_for_exactness(C2, rational)
    if points1 is None or points2 is None:
        return None
    origin = points1[0].copy()
    H1 = _center_curve_homogeneous_for_exactness(C1, rational, origin)
    H2 = _center_curve_homogeneous_for_exactness(C2, rational, origin)
    if H1 is None or H2 is None:
        return None
    local1 = H1[:, :-1] / H1[:, -1:]
    local2 = H2[:, :-1] / H2[:, -1:]
    component_scale = np.maximum(
        np.max(np.abs(local1), axis=0),
        np.max(np.abs(local2), axis=0))
    return H1, H2, component_scale


def _eval_curve_longdouble(C, t, rational):
    """De Casteljau evaluation without float64 underflow in quiet axes."""
    work = np.asarray(C, dtype=np.longdouble).copy()
    tt = np.longdouble(t)
    for _ in range(1, len(work)):
        work = (np.longdouble(1.0) - tt) * work[:-1] + tt * work[1:]
    value = work[0]
    if rational:
        if value[-1] == 0.0 or not np.isfinite(value[-1]):
            return None
        value = value[:-1] / value[-1]
    return value


def _eval_curve_scaled_components(C, t, rational, component_scale):
    """Evaluate Cartesian coordinates after normalizing each quiet axis.

    macOS aliases ``longdouble`` to float64, so multiplying the smallest
    subnormal by a Bernstein basis value can underflow even in the helper
    above.  Dividing each numerator coordinate by its own nonzero control
    scale *before* de Casteljau keeps that exactness information at O(1).
    """
    C = np.asarray(C, dtype=np.float64)
    scales = np.asarray(component_scale, dtype=np.float64)
    if rational:
        common = float(np.max(np.abs(C)))
        if not np.isfinite(common) or common <= 0.0:
            return None
        H = C / common
        weights = H[:, -1]
    else:
        H = C
        weights = None

    def _eval_scalar(values):
        work = np.asarray(values, dtype=np.float64).copy()
        tt = float(t)
        for _ in range(1, len(work)):
            work = (1.0 - tt) * work[:-1] + tt * work[1:]
        return float(work[0])

    denominator = _eval_scalar(weights) if rational else 1.0
    if not np.isfinite(denominator) or denominator == 0.0:
        return None
    result = np.zeros(len(scales), dtype=np.float64)
    for axis, scale in enumerate(scales):
        if scale == 0.0:
            continue
        numerator = _eval_scalar(H[:, axis] / scale)
        result[axis] = numerator / denominator
    return result


def _strict_residual_ok(C1, C2, u, v, rational, component_scale=None):
    """Accept a point equality only inside a floating roundoff envelope.

    ``atol`` is a search/resolution tolerance, not membership in the exact
    intersection set.  The envelope is tied to each coordinate's own control
    scale and curve degrees, so every representable nonzero offset in an
    otherwise constant coordinate remains nonzero for this predicate.
    """
    p1 = eval_curve(C1, float(u), rational=rational)
    p2 = eval_curve(C2, float(v), rational=rational)
    if not (np.all(np.isfinite(p1)) and np.all(np.isfinite(p2))):
        return False, p1, p2
    if component_scale is None:
        component_scale = _ccx_exactness_context(C1, C2, rational)
    if component_scale is None:
        return False, p1, p2
    C1_centered, C2_centered, scales = component_scale
    p1_scaled = _eval_curve_scaled_components(
        C1_centered, u, True, scales)
    p2_scaled = _eval_curve_scaled_components(
        C2_centered, v, True, scales)
    if (p1_scaled is None or p2_scaled is None
            or not np.all(np.isfinite(p1_scaled))
            or not np.all(np.isfinite(p2_scaled))):
        return False, p1, p2
    degree_factor = max(1, len(C1) + len(C2))
    bound = ((32.0 * degree_factor * np.finfo(np.float64).eps)
             * np.maximum(1.0, np.maximum(
                 np.abs(p1_scaled), np.abs(p2_scaled))))
    return bool(np.all(np.abs(p1_scaled - p2_scaled) <= bound)), p1, p2


def _strict_polish_ccx(C1, C2, u, v, rational, component_scale=None,
                       require_newton=False):
    """Globally polish a candidate, then require exact-set membership.

    The Newton calls are intentionally unbounded by the current subdivision
    cell (they retain only the public [0,1] curve domains).  Cell bounds are
    a search device and must not turn a root just across a cell seam into a
    near-root.  Neither Newton's step size nor ``atol`` can accept the result;
    the component-wise residual certificate above is the sole membership
    gate.
    """
    u = float(np.clip(u, 0.0, 1.0))
    v = float(np.clip(v, 0.0, 1.0))
    ok, p1, p2 = _strict_residual_ok(
        C1, C2, u, v, rational, component_scale)
    if ok and not require_newton:
        return u, v, p1

    if component_scale is None:
        polish_C1, polish_C2, polish_rational = C1, C2, rational
    else:
        polish_C1, polish_C2, _scales = component_scale
        polish_rational = True

    def _reported_residual_ok(G):
        if component_scale is None:
            return False
        _C1_centered, _C2_centered, scales = component_scale
        G = np.asarray(G, dtype=np.float64)
        scales = np.asarray(scales, dtype=np.float64)
        if G.shape != scales.shape or not np.all(np.isfinite(G)):
            return False
        degree_factor = max(1, len(C1) + len(C2))
        limit = 32.0 * degree_factor * np.finfo(np.float64).eps
        for value, scale in zip(G, scales):
            if scale == 0.0:
                if value != 0.0:
                    return False
            elif abs(value / scale) > limit:
                return False
        return True

    # A damped pass is robust at tangencies; the almost-undamped pass removes
    # the residual floor on well-conditioned transversal intersections.
    for damp in (1e-12, 1e-18):
        u, v, G, _last_step = newton_ccx(
            polish_C1, polish_C2, u, v, rational=polish_rational,
            tol=0.0, step_tol=8.0 * np.finfo(np.float64).eps,
            max_it=64, lm_damp=damp,
        )
        ok, p1, p2 = _strict_residual_ok(
            C1, C2, u, v, rational, component_scale)
        if ok and _reported_residual_ok(G):
            return float(u), float(v), p1
    return None


def _restrict_curve_interval(C, lo, hi):
    """Return control points reparameterized from [0,1] onto [lo,hi]."""
    C = np.asarray(C)
    reverse = hi < lo
    a, b = (hi, lo) if reverse else (lo, hi)
    if not (0.0 <= a < b <= 1.0):
        return None
    out = C.copy()
    if a > 0.0:
        _, out = _subdivide_curve(out, a)
    if b < 1.0:
        local_b = (b - a) / (1.0 - a) if a > 0.0 else b
        out, _ = _subdivide_curve(out, local_b)
    if reverse:
        out = out[::-1].copy()
    return out


def _bernstein_product_1d(a, b):
    """Exact-degree Bernstein product coefficients for unequal degrees."""
    a = np.asarray(a, dtype=np.longdouble).reshape(-1)
    b = np.asarray(b, dtype=np.longdouble).reshape(-1)
    p = len(a) - 1
    q = len(b) - 1
    out = np.zeros(p + q + 1, dtype=np.longdouble)
    for i, ai in enumerate(a):
        for j, bj in enumerate(b):
            k = i + j
            factor = (np.longdouble(math.comb(p, i))
                      * np.longdouble(math.comb(q, j))
                      / np.longdouble(math.comb(p + q, k)))
            out[k] += factor * ai * bj
    return out


def _homogeneous_curve_for_identity(C, rational, origin):
    H = _center_curve_homogeneous_for_exactness(
        C, rational, origin)
    if H is None:
        return None
    return np.asarray(H, dtype=np.longdouble)


def _overlap_mapping_is_identity(C1, C2, u_range, v_range, rational):
    """Certify ``C1(u(t)) == C2(v(t))`` as a Bernstein identity.

    This is an all-parameter certificate, not a finite sample test.  For
    rational curves it forms every coefficient of

        P1_k(t) W2(t) - P2_k(t) W1(t)

    after restricting both curves to the proposed affine parameter ranges.
    A coefficient is zero only within the roundoff accumulated by its two
    product terms; a one-sided nonzero coordinate offset therefore cannot be
    hidden by a model-scale tolerance or independent homogeneous scaling.
    """
    points1 = _cartesian_curve_controls_for_exactness(C1, rational)
    points2 = _cartesian_curve_controls_for_exactness(C2, rational)
    if points1 is None or points2 is None:
        return False
    origin = points1[0]
    H1 = _homogeneous_curve_for_identity(C1, rational, origin)
    H2 = _homogeneous_curve_for_identity(C2, rational, origin)
    if H1 is None or H2 is None or H1.shape[1] != H2.shape[1]:
        return False
    h1_xyz_scale = np.max(np.abs(H1[:, :-1]), axis=0)
    h2_xyz_scale = np.max(np.abs(H2[:, :-1]), axis=0)
    h1_weight_scale = np.max(np.abs(H1[:, -1]))
    h2_weight_scale = np.max(np.abs(H2[:, -1]))
    source_product_scale = (
        h1_xyz_scale * h2_weight_scale
        + h2_xyz_scale * h1_weight_scale)
    H1 = _restrict_curve_interval(H1, float(u_range[0]), float(u_range[1]))
    H2 = _restrict_curve_interval(H2, float(v_range[0]), float(v_range[1]))
    if H1 is None or H2 is None:
        return False
    w1 = H1[:, -1]
    w2 = H2[:, -1]
    if (np.any(w1 == 0.0) or np.any(w2 == 0.0)
            or not np.all(np.isfinite(w1))
            or not np.all(np.isfinite(w2))):
        return False

    eps = np.finfo(np.longdouble).eps
    op_factor = np.longdouble(
        64 * max(1, len(H1)) * max(1, len(H2))) * eps
    # Interval restriction is performed in the source floating precision
    # before the long-double Bernstein products.  Its cancellation error is
    # tied to the un-restricted source products, not to a coefficient that
    # may have cancelled to zero.  The common-origin centering above is
    # essential: a large world translation cannot inflate this floor and
    # hide a real offset.
    source_factor = (np.longdouble(8192 * max(1, len(H1) + len(H2)))
                     * np.longdouble(np.finfo(np.float64).eps))
    for axis in range(H1.shape[1] - 1):
        lhs = _bernstein_product_1d(H1[:, axis], w2)
        rhs = _bernstein_product_1d(H2[:, axis], w1)
        residual = np.abs(lhs - rhs)
        roundoff = (
            op_factor * (np.abs(lhs) + np.abs(rhs))
            + source_factor * source_product_scale[axis])
        if np.any(residual > roundoff):
            return False
    return True


def _invert_point_on_curve(C, pt, v0, rational, max_iter=40):
    """Project *pt* onto curve ``C``: damped 1-D Gauss-Newton on
    ``||C(v) - pt||`` with a monotone-decreasing line search, clamped to
    [0,1]. Returns ``(v, residual)`` for the best point found."""
    pt = np.asarray(pt, dtype=np.float64)
    v = float(min(1.0, max(0.0, float(v0))))
    p, d = eval_curve_d1(C, v, rational=rational)
    r = p - pt
    best_v, best_r = v, float(np.linalg.norm(r))
    for _ in range(max_iter):
        denom = float(np.dot(d, d))
        if denom < 1e-30:
            break
        step = -float(np.dot(r, d)) / denom
        if abs(step) < 1e-16:
            break
        scale = 1.0
        improved = False
        for _ls in range(20):
            cand = min(1.0, max(0.0, v + scale * step))
            p_c, d_c = eval_curve_d1(C, cand, rational=rational)
            r_c = p_c - pt
            n_c = float(np.linalg.norm(r_c))
            if n_c < best_r:
                v, p, d, r = cand, p_c, d_c, r_c
                best_v, best_r = cand, n_c
                improved = True
                break
            scale *= 0.5
        if not improved:
            break
    return best_v, best_r


def _project_point_on_curve(C, pt, rational, seed=None):
    """Coarse-scan (when unseeded) + Newton projection of *pt* onto ``C``."""
    if seed is None:
        ts = np.linspace(0.0, 1.0, 17)
        ds = [float(np.linalg.norm(
            eval_curve(C, float(t), rational=rational)
            - np.asarray(pt, dtype=np.float64))) for t in ts]
        seed = float(ts[int(np.argmin(ds))])
    return _invert_point_on_curve(C, pt, seed, rational)


def _tolerance_overlap_certificate(C1, C2, atol, rational, ptol_u, ptol_v,
                                   n_samples=65):
    """L47 residual-certified overlap tier (user decision 2026-07-12).

    The exact-affine identity above certifies coefficient-identical span
    pairs only. Near-coincident pairs (same path within ``atol`` but not
    exactly) and same-locus non-affine reparameterizations are genuine
    overlaps at modeling tolerance — the semantics CSX claims (L27) and SSX
    regions (L28) already use. This certificate:

      1. gates on the four domain-endpoint inversions (an admissible span
         must be pinned by a domain end of one curve at each end; interior-
         ended near-coincident bands stay unpromoted — typed partial);
      2. densely samples the candidate span, requiring every point of C1 to
         invert onto C2 within ``atol`` with a monotone parameter pairing;
      3. refuses to merge sub-tolerance TOPOLOGY (the CSX invariant, 1-D
         form): an interior exact root (sample residual at roundoff scale
         between clear-gap neighbours) or a transverse-direction flip
         between consecutive gap samples means the curves CROSS inside the
         band — the certificate rejects and returns the crossing brackets
         so the caller can certify them as isolated roots instead.

    Returns ``(overlap | None, brackets, band_evidence, span_evidence)``:
    ``brackets`` is a list of ``(u, v)`` seeds for strict root polishing;
    ``band_evidence`` is True when some qualifying domain end continues as a
    within-``atol`` coincidence BAND into the domain interior (inward-probe
    test) — a mere corner CONTACT (curves within atol at one endpoint but
    diverging immediately, e.g. consecutive edges of a loop sharing a
    vertex) is NOT band evidence and must not arm the bounded fallback;
    ``span_evidence`` is the widest endpoint-qualified u-extent (or None).
    """
    ends1 = [np.asarray(eval_curve(C1, t, rational=rational), dtype=np.float64)
             for t in (0.0, 1.0)]
    ends2 = [np.asarray(eval_curve(C2, t, rational=rational), dtype=np.float64)
             for t in (0.0, 1.0)]
    pts1 = _cartesian_curve_controls_for_exactness(C1, rational)
    pts2 = _cartesian_curve_controls_for_exactness(C2, rational)
    if pts1 is None or pts2 is None:
        return None, [], 0, None
    allpts = np.vstack([pts1, pts2])
    diag = float(np.linalg.norm(allpts.max(axis=0) - allpts.min(axis=0)))
    # Roundoff floor separating "exact root at this sample" from "genuine
    # gap": generous vs eval noise, far below any modeling tolerance.
    tiny = max(4096.0 * float(np.finfo(np.float64).eps), 1e-12) * max(1.0, diag)

    # Sound AABB pre-filter: the curve lies inside its control-point box, so
    # an endpoint farther than atol from the box cannot project within atol.
    # This keeps the endpoint gate ~free on generic (non-coincident) pairs —
    # CSX runs up to 4 nested CCX calls per cut face, so a Newton projection
    # per endpoint on every call is real wall-clock on the SSX gates.
    lo1 = pts1.min(axis=0) - atol
    hi1 = pts1.max(axis=0) + atol
    lo2 = pts2.min(axis=0) - atol
    hi2 = pts2.max(axis=0) + atol

    def _outside(pt, lo, hi):
        return bool(np.any(pt < lo) or np.any(pt > hi))

    cands = []
    band_ends = []   # (curve_index, t_end) of qualifying domain ends
    for t_end, pt in ((0.0, ends1[0]), (1.0, ends1[1])):
        if _outside(pt, lo2, hi2):
            continue
        v_end, r_end = _project_point_on_curve(C2, pt, rational)
        if r_end <= atol:
            cands.append((t_end, v_end))
            band_ends.append((0, t_end))
    for t_end, pt in ((0.0, ends2[0]), (1.0, ends2[1])):
        if _outside(pt, lo1, hi1):
            continue
        u_end, r_end = _project_point_on_curve(C1, pt, rational)
        if r_end <= atol:
            cands.append((u_end, t_end))
            band_ends.append((1, t_end))

    # Inward-probe band test: a coincidence BAND continues within atol into
    # the domain interior; a corner CONTACT (shared vertex of consecutive
    # edges) diverges immediately. Only bands justify the bounded fallback.
    band_evidence = False
    for which, t_end in band_ends:
        src, dst = (C1, C2) if which == 0 else (C2, C1)
        for step in (0.02, 0.05):
            t_in = step if t_end == 0.0 else 1.0 - step
            pt_in = np.asarray(eval_curve(src, t_in, rational=rational),
                               dtype=np.float64)
            _t_proj, r_in = _project_point_on_curve(dst, pt_in, rational)
            if r_in <= atol:
                band_evidence = True
                break
        if band_evidence:
            break

    if len(cands) < 2:
        return None, [], band_evidence, None
    uniq = []
    for u, v in cands:
        if not any(abs(u - uu) <= 4.0 * ptol_u and abs(v - vv) <= 4.0 * ptol_v
                   for uu, vv in uniq):
            uniq.append((float(u), float(v)))
    span_evidence = None
    if len(uniq) >= 2:
        u_ext = [u for u, _v in uniq]
        span_evidence = (float(min(u_ext)), float(max(u_ext)))

    pairs = [(uniq[i], uniq[j])
             for i in range(len(uniq)) for j in range(i + 1, len(uniq))]
    pairs.sort(key=lambda pr: -abs(pr[1][0] - pr[0][0]))

    brackets: list[tuple[float, float]] = []
    for (ua, va), (ub, vb) in pairs:
        if (abs(ub - ua) <= max(4.0 * ptol_u, 1e-6)
                or abs(vb - va) <= max(4.0 * ptol_v, 1e-6)):
            continue
        if ub < ua:
            ua, ub, va, vb = ub, ua, vb, va
        direction = 1.0 if vb >= va else -1.0
        us = np.linspace(ua, ub, n_samples)
        res = np.empty(n_samples)
        vs = np.empty(n_samples)
        dvecs = np.empty((n_samples, ends1[0].shape[0]))
        ok = True
        v_prev = None
        for k in range(n_samples):
            u_k = float(us[k])
            pt = np.asarray(eval_curve(C1, u_k, rational=rational),
                            dtype=np.float64)
            v_seed = va + (vb - va) * (k / (n_samples - 1.0))
            if v_prev is not None:
                v_seed = v_prev + (vb - va) / (n_samples - 1.0)
            v_k, r_k = _invert_point_on_curve(C2, pt, v_seed, rational)
            if r_k > atol:
                v_k2, r_k2 = _project_point_on_curve(C2, pt, rational)
                if r_k2 < r_k:
                    v_k, r_k = v_k2, r_k2
            if r_k > atol:
                ok = False
                break
            if (v_prev is not None
                    and direction * (v_k - v_prev) < -4.0 * ptol_v):
                ok = False    # folded pairing — not a functional overlap
                break
            vs[k] = v_k
            res[k] = r_k
            dvecs[k] = pt - np.asarray(
                eval_curve(C2, float(v_k), rational=rational),
                dtype=np.float64)
            v_prev = v_k
        if not ok:
            continue

        root_like = res <= tiny
        if not bool(np.all(root_like)):
            # Sub-tolerance topology guard (CSX invariant, 1-D form): a
            # transverse-direction FLIP between consecutive gap samples
            # means the curves cross inside the band — never merge; reject
            # the promotion and hand the crossing brackets to strict root
            # polishing. Root-like samples are skipped as direction noise:
            # a residual dip below roundoff scale alone is NOT crossing
            # evidence (a y-offset of a curve is locally tangent to it
            # wherever the tangent is parallel to the offset — measured
            # residual (offset)^2·kappa there, far below `tiny`); a
            # non-flipping exact touch inside the band is legitimately
            # covered by the tolerance overlap.
            pair_brackets = []
            for k in range(n_samples - 1):
                if root_like[k] or root_like[k + 1]:
                    continue
                d0 = dvecs[k]
                d1 = dvecs[k + 1]
                if float(np.dot(d0, d1)) < 0.0:
                    pair_brackets.append(
                        (0.5 * float(us[k] + us[k + 1]),
                         0.5 * float(vs[k] + vs[k + 1])))
            if pair_brackets:
                brackets.extend(pair_brackets)
                continue
        return ({
            "boundary_zeros": [],
            "overlap_endpoints": [],
            "u_range": (float(ua), float(ub)),
            "v_range": (float(va), float(vb)),
            "certification": "tolerance",
            "residual_max": float(res.max()),
        }, brackets, band_evidence, span_evidence)
    return None, brackets, band_evidence, span_evidence


def _vector_residual_hull_excludes_zero(C1, C2, rational, depth):
    """Certify that one Cartesian residual component cannot be zero.

    For each component this constructs the full tensor-product Bernstein net

        P1_k(u) W2(v) - P2_k(v) W1(u).

    If every coefficient is strictly on the same side of zero, with a
    subdivision- and product-roundoff margin, the two curve pieces cannot
    intersect.  Independent homogeneous scales cancel because each curve is
    normalized as a whole before the cross product.
    """
    points1 = _cartesian_curve_controls_for_exactness(C1, rational)
    points2 = _cartesian_curve_controls_for_exactness(C2, rational)
    if points1 is None or points2 is None:
        return False
    origin = points1[0]
    H1 = _homogeneous_curve_for_identity(C1, rational, origin)
    H2 = _homogeneous_curve_for_identity(C2, rational, origin)
    if H1 is None or H2 is None or H1.shape[1] != H2.shape[1]:
        return False
    w1 = H1[:, -1]
    w2 = H2[:, -1]
    if (not np.all(np.isfinite(w1)) or not np.all(np.isfinite(w2))
            or np.any(w1 == 0.0) or np.any(w2 == 0.0)):
        return False

    eps = np.finfo(np.longdouble).eps
    op_factor = np.longdouble(
        1024 * max(1, depth + 1) * max(len(H1), len(H2))) * eps
    for axis in range(H1.shape[1] - 1):
        lhs = H1[:, axis, None] * w2[None, :]
        rhs = w1[:, None] * H2[None, :, axis]
        residual = lhs - rhs
        roundoff = op_factor * (np.abs(lhs) + np.abs(rhs))
        if (np.all(residual > roundoff)
                or np.all(residual < -roundoff)):
            return True
    return False



from mmcore.numeric.bern import bernstein_partial_derivative_coeffs

# ---------------------------------------------------------------------------
# Phase 2 helpers
# ---------------------------------------------------------------------------

def _restrict_net_axis(F, axis, lo, hi, cell_lo, cell_hi):
    """Restrict a bivariate net along one axis to [lo, hi] within [cell_lo, cell_hi]."""
    span = cell_hi - cell_lo
    if span < 1e-30:
        return F
    frac_lo = (lo - cell_lo) / span
    frac_hi = (hi - cell_lo) / span
    Fv = F[..., np.newaxis]
    if frac_lo > 1e-12:
        _, Fv = de_casteljau_split_nd(Fv, axis=axis, t=frac_lo)
    if frac_hi < 1.0 - 1e-12:
        frac_hi_rescaled = (frac_hi - frac_lo) / (1.0 - frac_lo) if frac_lo > 1e-12 else frac_hi
        Fv, _ = de_casteljau_split_nd(Fv, axis=axis, t=frac_hi_rescaled)
    return Fv[..., 0]


def _split_intervals(cut, lo, hi, ptol):
    """Split [lo, hi] into up to 3 sub-intervals around cut ± ptol."""
    cut_lo = max(cut - ptol, lo)
    cut_hi = min(cut + ptol, hi)
    intervals = []
    if lo + 1e-15 < cut_lo:
        intervals.append((lo, cut_lo))
    intervals.append((cut_lo, cut_hi))
    if cut_hi < hi - 1e-15:
        intervals.append((cut_hi, hi))
    return intervals


def _cutout_2d(F_cell, seg1, seg2, pw, qw, u0, u1, v0, v1, depth,
               u_cut, v_cut, ptol_u, ptol_v, rational):
    """Cut out a ptol-neighborhood around (u_cut, v_cut) from a 2D cell.

    Splits along both axes at cut ± ptol, producing 3×3 = 9 boxes.
    The center box is discarded. Remaining 8 are returned with restricted nets.
    """
    u_intervals = _split_intervals(u_cut, u0, u1, ptol_u)
    v_intervals = _split_intervals(v_cut, v0, v1, ptol_v)

    u_center = len(u_intervals) // 2 if len(u_intervals) == 3 else (0 if len(u_intervals) == 1 else -1)
    v_center = len(v_intervals) // 2 if len(v_intervals) == 3 else (0 if len(v_intervals) == 1 else -1)

    sub_cells = []
    for ui, (u_lo, u_hi) in enumerate(u_intervals):
        for vi, (v_lo, v_hi) in enumerate(v_intervals):
            if ui == u_center and vi == v_center:
                continue
            if u_hi - u_lo < 1e-15 or v_hi - v_lo < 1e-15:
                continue

            F_sub = _restrict_net_axis(F_cell, 0, u_lo, u_hi, u0, u1)
            F_sub = _restrict_net_axis(F_sub, 1, v_lo, v_hi, v0, v1)

            # Restrict curves
            u_frac_lo = (u_lo - u0) / max(u1 - u0, 1e-30)
            u_frac_hi = (u_hi - u0) / max(u1 - u0, 1e-30)
            C1_sub = seg1
            if u_frac_lo > 1e-12:
                _, C1_sub = _subdivide_curve(C1_sub, u_frac_lo)
            if u_frac_hi < 1.0 - 1e-12:
                uf_rescaled = (u_frac_hi - u_frac_lo) / (1.0 - u_frac_lo) if u_frac_lo > 1e-12 else u_frac_hi
                C1_sub, _ = _subdivide_curve(C1_sub, uf_rescaled)
            pw_sub = C1_sub[:, -1].copy() if rational else np.ones(C1_sub.shape[0])

            v_frac_lo = (v_lo - v0) / max(v1 - v0, 1e-30)
            v_frac_hi = (v_hi - v0) / max(v1 - v0, 1e-30)
            C2_sub = seg2
            if v_frac_lo > 1e-12:
                _, C2_sub = _subdivide_curve(C2_sub, v_frac_lo)
            if v_frac_hi < 1.0 - 1e-12:
                vf_rescaled = (v_frac_hi - v_frac_lo) / (1.0 - v_frac_lo) if v_frac_lo > 1e-12 else v_frac_hi
                C2_sub, _ = _subdivide_curve(C2_sub, vf_rescaled)
            qw_sub = C2_sub[:, -1].copy() if rational else np.ones(C2_sub.shape[0])

            sub_cells.append((C1_sub, C2_sub, F_sub, pw_sub, qw_sub,
                              u_lo, u_hi, v_lo, v_hi, depth + 1))
    return sub_cells


def _phase2_ccx(F, C1, C2, C1_orig, C2_orig,
                u_lo, u_hi, v_lo, v_hi,
                atol, rational, ptol_u, ptol_v,
                known_points=None,
                max_depth=50, max_cells=50_000,
                max_results=4_096,
                initial_stack=None):
    """Phase 2: find isolated intersections via subdivision + Newton + cutout.

    No boundary analysis, no overlap checks, no classifier.
    Just: min-of-net → derivative sign → Newton → cutout.
    """
    from mmcore.numeric.intersection._sq_dist_classify import (
        _check_min_of_net, _check_lipschitz, _weight_max_product,
    )

    isolated = list(known_points) if known_points else []
    n_known = len(isolated)
    cells = 0
    exhausted = False
    component_scale = _ccx_exactness_context(
        C1_orig, C2_orig, rational)

    if initial_stack is not None:
        stack = list(initial_stack)
    else:
        _, Pw = extract_weights(C1, rational=rational)
        _, Qw = extract_weights(C2, rational=rational)
        stack = [(C1, C2, F, Pw.copy(), Qw.copy(),
                  u_lo, u_hi, v_lo, v_hi, 0)]

    while stack:
        if cells >= max_cells or len(isolated) - n_known >= max_results:
            exhausted = True
            break
        cells += 1

        seg1, seg2, F_cell, pw, qw, u0, u1, v0, v1, depth = stack.pop()
        #print(f"CCX: {cells} cells: {( ( u0, u1), (v0, v1), depth)}")
        # AABB prune: cheapest possible check — control point bounding boxes
        from mmcore.numeric._aabb import aabb, aabb_intersect
        if rational:
            pts1 = seg1[:, :-1] / seg1[:, -1:]
            pts2 = seg2[:, :-1] / seg2[:, -1:]
        else:
            pts1 = seg1
            pts2 = seg2
        if _vector_residual_hull_excludes_zero(
                seg1, seg2, rational, depth):
            continue
        bb1 = np.array(aabb(pts1)); bb1[0] -= atol; bb1[1] += atol
        bb2 = np.array(aabb(pts2)); bb2[0] -= atol; bb2[1] += atol
        if not aabb_intersect(bb1, bb2):
            continue

        w_sc = _weight_max_product(pw, qw)

        # min-of-net prune
        if _check_min_of_net(F_cell, atol, w_sc):
            continue

        # Lipschitz prune
        if _check_lipschitz(F_cell, atol, w_sc):
            continue

        # Derivative sign pruning
        Fv = F_cell[..., np.newaxis]
        can_have_stationary = True
        for ax in range(2):
            dF = bernstein_partial_derivative_coeffs(Fv, axis=ax)
            coeffs = dF[..., 0]
            if np.min(coeffs) > 0 or np.max(coeffs) <= 0:
                can_have_stationary = False
                break
        if not can_have_stationary:
            continue

        # ptol-based early termination
        if (u1 - u0) <= ptol_u and (v1 - v0) <= ptol_v:
            u_mid = 0.5 * (u0 + u1)
            v_mid = 0.5 * (v0 + v1)
            polished = _strict_polish_ccx(
                C1_orig, C2_orig, u_mid, v_mid, rational,
                component_scale=component_scale, require_newton=True)
            if polished is not None:
                u_sol, v_sol, pt = polished
                if (u0 - 0.25 * ptol_u <= u_sol <= u1 + 0.25 * ptol_u
                        and v0 - 0.25 * ptol_v <= v_sol
                        <= v1 + 0.25 * ptol_v
                        and not _is_duplicate(isolated, pt, atol)):
                    isolated.append({
                        "u": float(u_sol), "v": float(v_sol),
                        "point": pt, "_micro": True,
                    })
            continue

        # Newton from cell center
        u_mid = 0.5 * (u0 + u1)
        v_mid = 0.5 * (v0 + v1)
        uv_candidates = [
            (u0, v0), (u0, v1), (u1, v1), (u1, v0),
            (u_mid, v_mid),
        ]
        root_found = False
        for u_mid, v_mid in uv_candidates:
            if root_found:
                break
            polished = _strict_polish_ccx(
                C1_orig, C2_orig, u_mid, v_mid, rational,
                component_scale=component_scale, require_newton=True)
            if polished is not None:
                u_sol, v_sol, pt = polished
            else:
                continue
            if u0 < u_sol < u1 and v0 < v_sol < v1:
                is_new = not _is_duplicate(isolated, pt, atol)
                #print(f"CCX: is_new: {is_new}")
                if is_new:
                    isolated.append({"u": float(u_sol), "v": float(v_sol), "point": pt})
                    sub_cells = _cutout_2d(
                        F_cell, seg1, seg2, pw, qw, u0, u1, v0, v1, depth,
                        float(u_sol), float(v_sol), ptol_u, ptol_v, rational,
                    )
                    stack.extend(sub_cells)
                    root_found=True
                    break

        if root_found:continue


        if depth >= max_depth:
            # None of the certificates above proved this cell root-free and
            # Newton did not resolve it.  Dropping it at the depth guard is a
            # partial search just like dropping queued cells at ``max_cells``.
            exhausted = True
            continue

        # Subdivide




        u_mid_split = 0.5 * (u0 + u1)
        v_mid_split = 0.5 * (v0 + v1)
        seg1_L, seg1_R = _subdivide_curve(seg1)
        seg2_L, seg2_R = _subdivide_curve(seg2)
        F_LL, F_LR, F_RR,F_RL, =_subdivide_sq_dist_net_2d(F_cell,0.5 ,0.5)


        pw_L = seg1_L[:, -1].copy() if rational else np.ones(seg1_L.shape[0])
        pw_R = seg1_R[:, -1].copy() if rational else np.ones(seg1_R.shape[0])
        qw_L = seg2_L[:, -1].copy() if rational else np.ones(seg2_L.shape[0])
        qw_R = seg2_R[:, -1].copy() if rational else np.ones(seg2_R.shape[0])
        stack.append((seg1_L, seg2_L, F_LL, pw_L, qw_L, u0, u_mid_split, v0, v_mid_split, depth+1))
        stack.append((seg1_L, seg2_R, F_LR, pw_L, qw_R,u0, u_mid_split, v_mid_split, v1, depth+1))
        stack.append((seg1_R, seg2_R, F_RR, pw_R,  qw_R,u_mid_split, u1, v_mid_split, v1, depth+1))
        stack.append((seg1_R, seg2_L, F_RL, pw_R, qw_L,u_mid_split, u1,  v0, v_mid_split, depth+1))

    return isolated[n_known:], exhausted, cells


# ---------------------------------------------------------------------------
# Main algorithm: two-phase architecture
# ---------------------------------------------------------------------------
from .._bezier_common import _compute_remaining_intervals


def bez_ccx(
    C1,
    C2,
    atol=1e-3,
    rational=False,
    max_depth=50,
    max_cells=100_000,
    max_results=4_096,
) -> dict:
    """Bezier curve-curve intersection via two-phase architecture.

    Phase 1: Find boundary intersections and overlaps on the initial patch.
             These can ONLY exist at the boundaries of the original objects.
    Phase 2: Search for isolated intersections on the remaining parameter
             intervals via subdivision + Newton + cutout.

    ``max_cells`` is shared by Phase 1 and every Phase-2 interval.
    ``max_results`` also caps Phase-1 boundary roots before pairwise topology
    checks.  A capped return sets ``budget_exhausted``; callers must treat
    ``boundary_topology_complete=False`` as diagnostic partial output.

    Overlap entries carry ``certification``: ``'exact'`` (Bernstein affine
    identity) or ``'tolerance'`` (L47 residual tier: dense-sample inversion
    pairing within ``atol`` + ``residual_max``, e.g. near-coincident twins
    and non-affine same-locus reparameterizations). An overlap-class
    structure that NEITHER certificate can promote and the bounded fallback
    cannot discretize returns ``uncertified_overlap_span=(u_lo, u_hi)`` with
    ``boundary_topology_complete=False`` — typed, not a bare budget flag.
    """
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)

    cells_remaining = max(0, int(max_cells))
    cells_processed = 0
    budget_exhausted = False

    def _result(isolated, overlaps, *, topology_complete=True):
        return {
            "isolated": isolated,
            "overlaps": overlaps,
            "budget_exhausted": bool(budget_exhausted),
            "cells_processed": int(cells_processed),
            "boundary_topology_complete": bool(topology_complete),
        }

    if cells_remaining <= 0:
        budget_exhausted = True
        return _result([], [], topology_complete=False)

    F = curve_curve_squared_net_homog(C1, C2, rational=rational)

    _, Pw = extract_weights(C1, rational=rational)
    _, Qw = extract_weights(C2, rational=rational)

    C1_orig = C1
    C2_orig = C2

    ptol_u, ptol_v = _compute_param_tols(C1, C2, atol, rational)
    component_scale = _ccx_exactness_context(
        C1_orig, C2_orig, rational)

    isolated = []
    overlaps = []

    # ===================================================================
    # PHASE 1: Boundary analysis + overlap (initial patch only)
    # ===================================================================
    # Charge the classifier itself, then share the remaining allowance with
    # every recursive 1-D boundary solve it invokes.  The root cap is kept
    # deliberately below the point where the classifier's pairwise valley
    # check becomes a material O(B^2) operation.
    cells_remaining -= 1
    cells_processed += 1
    from mmcore.numeric.intersection._bern_zero_1d import bernstein_zero_budget
    boundary_root_cap = min(max(0, int(max_results)), 128)

    # ``_check_overlap`` pairs every near-zero coefficient on opposite
    # boundaries.  Bound that candidate materialization before entering the
    # legacy classifier; limiting only the later precise-root list would be
    # too late to prevent this separate quadratic path.
    w_scale = float(np.max(np.abs(Pw))) * float(np.max(np.abs(Qw)))
    boundary_threshold = (atol * w_scale) ** 2
    boundary_candidate_counts = (
        int(np.count_nonzero(np.abs(F[0, :]) < boundary_threshold)),
        int(np.count_nonzero(np.abs(F[-1, :]) < boundary_threshold)),
        int(np.count_nonzero(np.abs(F[:, 0]) < boundary_threshold)),
        int(np.count_nonzero(np.abs(F[:, -1]) < boundary_threshold)),
    )
    if any(count > boundary_root_cap for count in boundary_candidate_counts):
        budget_exhausted = True
        return _result([], [], topology_complete=False)

    with bernstein_zero_budget(cells_remaining, boundary_root_cap) as zero_budget:
        cls = classify_sq_dist_net(F, atol, Pw, Qw)
    cells_remaining -= zero_budget.nodes
    cells_processed += zero_budget.nodes

    # A capped boundary solve is not a smaller-but-valid topology.  In
    # particular, using its endpoints to declare an overlap or cut Phase 2
    # could hide real roots.  Discard it and mark the result explicitly
    # partial so callers such as CSX cannot silently consume it.
    if zero_budget.exhausted:
        budget_exhausted = True
        return _result([], [], topology_complete=False)
    if any(not isinstance(bz, BoundaryZero) for bz in cls.precise_zeros):
        budget_exhausted = True
        return _result([], [], topology_complete=False)
    #print(f"CCX: {cls} (phase 1)")
    if cls.kind == NO_INTERSECTION:
        return _result([], [])

    # 1a. Collect validated boundary zeros (don't add to isolated yet)
    boundary_hits = []  # list of strictly validated (u, v, point)
    if cls.precise_zeros:
        for bz in cls.precise_zeros:
            u_bz, v_bz = _boundary_zero_to_uv(bz, 0.0, 1.0, 0.0, 1.0)
            polished = _strict_polish_ccx(
                C1_orig, C2_orig, u_bz, v_bz, rational,
                component_scale=component_scale)
            if polished is not None:
                u_sol, v_sol, point = polished
                boundary_hits.append((float(u_sol), float(v_sol), point))

    # 1b. Check for overlap
    overlap_found = False
    if cls.kind == OVERLAP:
        # The legacy classifier's OVERLAP kind is intentionally tolerant and
        # its ``overlap_endpoints`` payload can be only a valley witness.  It
        # is therefore a candidate generator, never a public topology claim.
        # Build paired strict endpoint roots, then promote only an affine map
        # whose full Bernstein cross-residual is the zero identity.
        uv_roots = []
        uv_eps = 64.0 * np.finfo(np.float64).eps
        for u_hit, v_hit, _point in boundary_hits:
            if not any(abs(u_hit - u0) <= uv_eps
                       and abs(v_hit - v0) <= uv_eps
                       for u0, v0 in uv_roots):
                uv_roots.append((u_hit, v_hit))

        endpoint_pairs = []
        for i in range(len(uv_roots)):
            for j in range(i + 1, len(uv_roots)):
                endpoint_pairs.append((uv_roots[i], uv_roots[j]))

        # A classifier may summarize a full overlap without retaining typed
        # endpoint roots.  These two domain-spanning affine maps are still
        # candidates, but their endpoints and coefficient identity must pass
        # the same strict gates as every inferred sub-range.
        for pair in (
                ((0.0, 0.0), (1.0, 1.0)),
                ((0.0, 1.0), (1.0, 0.0))):
            p0 = _strict_polish_ccx(
                C1_orig, C2_orig, *pair[0], rational,
                component_scale=component_scale)
            p1 = _strict_polish_ccx(
                C1_orig, C2_orig, *pair[1], rational,
                component_scale=component_scale)
            if p0 is not None and p1 is not None:
                endpoint_pairs.append(
                    ((p0[0], p0[1]), (p1[0], p1[1])))

        endpoint_pairs.sort(
            key=lambda pair: -(
                abs(pair[1][0] - pair[0][0])
                + abs(pair[1][1] - pair[0][1])))
        promoted = None
        seen_ranges = set()
        for (ua, va), (ub, vb) in endpoint_pairs:
            if ub < ua:
                ua, ub, va, vb = ub, ua, vb, va
            if abs(ub - ua) <= uv_eps or abs(vb - va) <= uv_eps:
                continue
            key = tuple(round(x, 15) for x in (ua, ub, va, vb))
            if key in seen_ranges:
                continue
            seen_ranges.add(key)
            if _overlap_mapping_is_identity(
                    C1_orig, C2_orig, (ua, ub), (va, vb), rational):
                promoted = (ua, ub, va, vb)
                break

        if promoted is not None:
            ua, ub, va, vb = promoted
            overlaps.append({
                "boundary_zeros": cls.boundary_zeros,
                "overlap_endpoints": cls.overlap_endpoints,
                "u_range": (float(ua), float(ub)),
                "v_range": (float(va), float(vb)),
                "certification": "exact",
            })
            overlap_found = True

    # Ledger L47 (user decision 2026-07-12): residual-certified overlap tier.
    # Coincidence at modeling tolerance is a real overlap even when no exact
    # affine identity exists — near-coincident pairs (imported geometry) and
    # same-locus non-affine reparameterizations. The certificate inverts
    # dense samples across a domain-end-pinned span (residual <= atol,
    # monotone pairing) and REFUSES to merge crossing structure inside the
    # band (transverse-direction flips between gap-scale samples),
    # returning those brackets for strict isolated-root certification.
    residual_band_evidence = False
    uncertified_span_evidence = None
    interior_bracket_hits = []
    if not overlap_found and cells_remaining > 0:
        cells_remaining -= 1
        cells_processed += 1
        tol_overlap, tol_brackets, residual_band_evidence, \
            uncertified_span_evidence = _tolerance_overlap_certificate(
                C1_orig, C2_orig, atol, rational, ptol_u, ptol_v)
        if tol_overlap is not None:
            overlaps.append(tol_overlap)
            overlap_found = True
        for u_seed, v_seed in tol_brackets:
            polished = _strict_polish_ccx(
                C1_orig, C2_orig, u_seed, v_seed, rational,
                component_scale=component_scale)
            if polished is None:
                continue
            u_sol, v_sol, point = polished
            if overlap_found:
                lo, hi = overlaps[-1]["u_range"]
                if min(lo, hi) - ptol_u <= u_sol <= max(lo, hi) + ptol_u:
                    continue
            interior_bracket_hits.append(
                (float(u_sol), float(v_sol), point))

    # A tolerant OVERLAP classification (or residual-gate coincidence
    # evidence) whose certificates all failed is often a broad, near-zero
    # distance valley.  An unrestricted Phase-2 subdivision of that valley
    # is both expensive and incapable of turning the rejected candidate
    # into a sound public overlap.  Give the isolated-root fallback a
    # separate, bounded allowance and report a partial result if that
    # allowance cannot certify the remaining cells.
    non_affine_overlap_fallback = (
        (cls.kind == OVERLAP or residual_band_evidence)
        and not overlap_found)
    non_affine_overlap_cells_remaining = (
        min(cells_remaining, 2_000)
        if non_affine_overlap_fallback else None
    )

    def _finalize(topology_complete=True):
        # Typed L47 outcome, mirroring CSX's L42 export: when the overlap-
        # class structure could not be certified AND the bounded fallback
        # could not discretize it, name the span instead of billing the
        # failure to the budget with topology claimed complete.
        structural = (non_affine_overlap_fallback and budget_exhausted
                      and not overlap_found)
        res = _result(isolated, overlaps,
                      topology_complete=topology_complete and not structural)
        if structural:
            span = uncertified_span_evidence or (0.0, 1.0)
            res["uncertified_overlap_span"] = (
                float(span[0]), float(span[1]))
        return res

    # 1c. Classify boundary hits: overlap endpoints go into the overlap,
    #     remaining boundary hits become isolated intersections.
    if overlap_found and overlaps:
        ovl = overlaps[-1]
        u_start_ovl, u_end_ovl = ovl["u_range"]
        v_start_ovl, v_end_ovl = ovl["v_range"]
        u_lo_ovl = min(u_start_ovl, u_end_ovl)
        u_hi_ovl = max(u_start_ovl, u_end_ovl)
        for u_bz, v_bz, pt in boundary_hits:
            on_overlap = False
            if u_lo_ovl - ptol_u <= u_bz <= u_hi_ovl + ptol_u:
                lam = ((u_bz - u_start_ovl)
                       / (u_end_ovl - u_start_ovl))
                v_expected = ((1.0 - lam) * v_start_ovl
                              + lam * v_end_ovl)
                if ovl.get("certification") == "tolerance":
                    # The pairing of a residual-certified overlap need not
                    # be affine — locate the expected partner parameter by
                    # inversion (seeded by the affine guess).
                    v_expected, _r_inv = _invert_point_on_curve(
                        C2_orig,
                        eval_curve(C1_orig, float(u_bz), rational=rational),
                        v_expected, rational)
                on_overlap = abs(v_bz - v_expected) <= 2.0 * ptol_v
            if not on_overlap:
                if not _is_duplicate(isolated, pt, atol):
                    if len(isolated) >= max_results:
                        budget_exhausted = True
                        break
                    isolated.append({"u": u_bz, "v": v_bz, "point": pt})
    else:
        for u_bz, v_bz, pt in boundary_hits:
            if not _is_duplicate(isolated, pt, atol):
                if len(isolated) >= max_results:
                    budget_exhausted = True
                    break
                isolated.append({"u": u_bz, "v": v_bz, "point": pt})

    # Interior crossings certified from the residual tier's rejected
    # brackets (crossing structure inside a tolerance band is topology,
    # never merged — CSX invariant, 1-D form).
    for u_hit, v_hit, pt in interior_bracket_hits:
        if not _is_duplicate(isolated, pt, atol):
            if len(isolated) >= max_results:
                budget_exhausted = True
                break
            isolated.append({"u": u_hit, "v": v_hit, "point": pt})

    if budget_exhausted:
        return _finalize()

    # ===================================================================
    # PHASE 2: Isolated intersection search
    # ===================================================================
    # Cut from the FIRST curve's parameter (u) only — same principle as
    # CSX cutting along t. For each remaining u-interval, search against
    # the FULL second curve [0,1].

    # Collect u-intervals to exclude
    u_exclude = []
    if overlap_found and overlaps:
        ovl = overlaps[-1]
        u_lo_ovl = min(ovl["u_range"])
        u_hi_ovl = max(ovl["u_range"])
        u_exclude.append((u_lo_ovl - ptol_u, u_hi_ovl + ptol_u))
    for iso in isolated:
        u_exclude.append((iso["u"] - ptol_u, iso["u"] + ptol_u))


    u_intervals = _compute_remaining_intervals(u_exclude, 0.0, 1.0)

    for u_lo, u_hi in u_intervals:
        if u_hi - u_lo < ptol_u * 0.1:
            continue

        # Restrict net and first curve to this u sub-interval
        F_sub = _restrict_net_axis(F, 0, u_lo, u_hi, 0.0, 1.0)
        C1_sub = C1
        if u_lo > 1e-12:
            _, C1_sub = _subdivide_curve(C1_sub, u_lo)
        if u_hi < 1.0 - 1e-12:
            u_hi_rescaled = (u_hi - u_lo) / (1.0 - u_lo) if u_lo > 1e-12 else u_hi
            C1_sub, _ = _subdivide_curve(C1_sub, u_hi_rescaled)
        pw_sub = C1_sub[:, -1].copy() if rational else np.ones(C1_sub.shape[0])

        # Quick min-of-net check
        from mmcore.numeric.intersection._sq_dist_classify import (
            _check_min_of_net, _weight_max_product,
        )
        w_sc = _weight_max_product(pw_sub, Qw)
        if _check_min_of_net(F_sub, atol, w_sc):
            continue

        # Run Phase 2 on this sub-interval × full v
        if (cells_remaining <= 0 or len(isolated) >= max_results
                or (non_affine_overlap_fallback
                    and non_affine_overlap_cells_remaining <= 0)):
            budget_exhausted = True
            break
        phase2_cell_limit = cells_remaining
        if non_affine_overlap_fallback:
            phase2_cell_limit = min(
                phase2_cell_limit, non_affine_overlap_cells_remaining)
        phase2_iso, phase2_exhausted, cells_used = _phase2_ccx(
            F_sub, C1_sub, C2, C1_orig, C2_orig,
            u_lo, u_hi, 0.0, 1.0,
            atol, rational, ptol_u, ptol_v,
            known_points=isolated,
            max_depth=max_depth, max_cells=phase2_cell_limit,
            max_results=max_results - len(isolated),
        )
        cells_remaining -= cells_used
        cells_processed += cells_used
        if non_affine_overlap_fallback:
            non_affine_overlap_cells_remaining -= cells_used
        budget_exhausted = budget_exhausted or phase2_exhausted

        for iso in phase2_iso:
            iso.pop('_micro', None)
            if not _is_duplicate(isolated, iso["point"], atol):
                isolated.append(iso)

        if non_affine_overlap_fallback and phase2_exhausted:
            break

    return _finalize()


def _is_duplicate(isolated, pt, atol):
    """Check if *pt* is within *atol* of any existing isolated point."""
    for entry in isolated:
        existing = np.asarray(entry["point"])
        if np.linalg.norm(existing - pt) < atol:
            return True
    return False

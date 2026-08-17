# mmcore/numeric/_bez_closest_point.py
"""Closest-point on rational Bézier/NURBS curves and surfaces via
squared-distance Bernstein nets.

Contract (band semantics)
-------------------------
The solvers return the COMPLETE SET of globally closest entities: everything
whose distance lies within ``[d_min, d_min + atol]``. Local minima farther
than that are intentionally NOT reported (reproducibility for autonomous
pipelines: the answer set is deterministic, not implementation-chance).

Per Bézier patch the closest-point set is provably one of (polynomial
identity theorem on ``F - d^2 W``):

* finitely many isolated points          -> point entities;
* a 1-D equidistant algebraic curve      -> a traced parametric curve entity;
* the entire patch                       -> a whole-patch entity.

Entity shapes (every entity carries ``distance``, ``kind`` and a
representative ``point``):

* curve point:        ``{"t", "point", "distance", "kind": "min"|"boundary_min"}``
* curve degenerate:   ``{"kind": "degenerate_segment", "t_range", "t", "point", "distance"}``
* surface point:      ``{"u", "v", "point", "distance", "kind": "min"|"boundary_min"[, "eval"]}``
* surface curve:      ``{"kind": "degenerate_curve", "uv": (N,2), "points": (N,3),
                         "closed", "u", "v", "point", "distance"}``
* surface whole-patch:``{"kind": "degenerate_surface", "u_range", "v_range",
                         "u", "v", "point", "distance"}``

Results are sorted ascending by distance; ``result[0]`` is the global closest.

Replaces the unreliable divide-and-conquer code in ``closest_point.py``
(kept untouched for A/B comparison). See
``docs/superpowers/specs/2026-07-02-closest-point-band-bnb-design.md``.
"""
from __future__ import annotations

import heapq
import itertools
import warnings
from math import comb

import numpy as np


# Relative floor on the parametric tolerance: subdivision never resolves below
# this fraction of the unit patch domain (final accuracy comes from the leaf
# Newton polish). Prevents pathological deep subdivision at large coordinate
# scales; engages only when the geometry-derived ptol is already smaller.
_PTOL_FLOOR = 1e-4

# Relative eigenvalue threshold separating an isolated minimum (Hessian well
# conditioned) from a degenerate one (rank drop along the equidistant curve).
# Measured gap between the cases is enormous (isolated ~0.4 vs degenerate
# ~1e-16 relative), so the placement is not delicate.
_DEGEN_EIG_RATIO = 1e-6

# Angular stationarity tolerance: |cos| between the residual vector and each
# tangent direction must be below this for a candidate to count as a
# stationary (foot) point. Newton converges to ~1e-12; 1e-4 is generous.
_STATIONARY_COS_TOL = 1e-4


def _publish_work_stats(stats, cells_processed, budget_exhausted, **breakdown):
    """Populate an optional mutable stats mapping without changing results."""
    if stats is None:
        return
    stats.clear()
    stats.update(cells_processed=int(cells_processed),
                 budget_exhausted=bool(budget_exhausted),
                 **{key: int(value) for key, value in breakdown.items()})


# ---------------------------------------------------------------------------
# Bernstein algebra
# ---------------------------------------------------------------------------

def _binom_row(n):
    return np.array([comb(n, i) for i in range(n + 1)], dtype=np.float64)


def _scale_by_binoms(net):
    """Multiply a scalar Bernstein net by per-axis binomial coefficients."""
    out = np.asarray(net, dtype=np.float64).copy()
    for ax in range(out.ndim):
        p = out.shape[ax] - 1
        shape = [1] * out.ndim
        shape[ax] = p + 1
        out = out * _binom_row(p).reshape(shape)
    return out


def _unscale_by_binoms(net):
    """Divide a scalar Bernstein net by per-axis binomial coefficients."""
    out = np.asarray(net, dtype=np.float64).copy()
    for ax in range(out.ndim):
        p = out.shape[ax] - 1
        shape = [1] * out.ndim
        shape[ax] = p + 1
        out = out / _binom_row(p).reshape(shape)
    return out


def _ndconv_full(A, B):
    """Exact full linear convolution of two scalar ND arrays (small nets)."""
    A = np.asarray(A, dtype=np.float64)
    B = np.asarray(B, dtype=np.float64)
    out_shape = tuple(sa + sb - 1 for sa, sb in zip(A.shape, B.shape))
    out = np.zeros(out_shape, dtype=np.float64)
    for idxB in np.ndindex(*B.shape):
        bval = B[idxB]
        if bval == 0.0:
            continue
        sl = tuple(slice(i, i + s) for i, s in zip(idxB, A.shape))
        out[sl] += A * bval
    return out


def _bernstein_product_nd(a, b):
    """Exact product of two scalar Bernstein nets of equal ndim.

    Uses ``B_i^p * B_j^q = [C(p,i)C(q,j)/C(p+q,i+j)] B_{i+j}^{p+q}`` per axis.
    Returns a net of per-axis degree ``deg(a)+deg(b)``.
    """
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.ndim != b.ndim:
        raise ValueError("operands must have the same number of axes")
    num = _ndconv_full(_scale_by_binoms(a), _scale_by_binoms(b))
    return _unscale_by_binoms(num)


from mmcore.numeric import bern_sq_dist
from mmcore.numeric._work_budget import reconcile_reported
from mmcore.numeric.bern import bernstein_partial_derivative_coeffs


def _deriv_net(net, axis):
    """Bernstein coeffs of the partial derivative along ``axis`` (scalar net in/out)."""
    return bernstein_partial_derivative_coeffs(net[..., None], axis)[..., 0]


def point_curve_stationarity_net(point, C, rational=True):
    """Return ``(N, F, Qw)`` where ``N(t)=0`` iff ``d/dt ||point-C(t)||^2 = 0``.

    ``N = F'·w − 2F·w'`` (exact); for non-rational input ``N = F'``.
    ``F`` is the squared-distance numerator net and ``Qw`` the weight net.
    """
    C = np.asarray(C, dtype=np.float64)
    F = bern_sq_dist.point_curve_distance_squared_net_homog(point, C, rational=rational)
    Qw = C[:, -1].copy() if rational else np.ones(C.shape[0], dtype=np.float64)
    Fp = _deriv_net(F, 0)
    if not rational:
        return Fp, F, Qw
    wp = _deriv_net(Qw, 0)
    N = _bernstein_product_nd(Fp, Qw) - 2.0 * _bernstein_product_nd(F, wp)
    return N, F, Qw


def point_surface_stationarity_nets(point, S, rational=True):
    """Return ``(N_u, N_v, F, Sw)``; a joint stationary point needs both nets = 0.

    ``N_u = F_u·w − 2F·w_u``, ``N_v = F_v·w − 2F·w_v`` (exact); non-rational →
    ``N_u = F_u``, ``N_v = F_v``.
    """
    S = np.asarray(S, dtype=np.float64)
    F = bern_sq_dist.point_surface_distance_squared_net_homog(point, S, rational=rational)
    Sw = S[:, :, -1].copy() if rational else np.ones(S.shape[:2], dtype=np.float64)
    Fu = _deriv_net(F, 0)
    Fv = _deriv_net(F, 1)
    if not rational:
        return Fu, Fv, F, Sw
    wu = _deriv_net(Sw, 0)
    wv = _deriv_net(Sw, 1)
    Nu = _bernstein_product_nd(Fu, Sw) - 2.0 * _bernstein_product_nd(F, wu)
    Nv = _bernstein_product_nd(Fv, Sw) - 2.0 * _bernstein_product_nd(F, wv)
    return Nu, Nv, F, Sw


from mmcore.numeric._bern_homog import (
    eval_bezier_curve_homog_with_derivs,
    eval_bezier_surface_homog_with_derivs,
    project_curve_homog_to_cartesian,
    project_surface_homog_to_cartesian,
)
from mmcore.numeric._bezier_common import (
    _to_homog_curve, _to_homog_surface, eval_curve, eval_curve_d1,
    eval_surface, eval_surface_d1,
)


def eval_curve_d2(C, t, rational=True):
    """Return ``(point, C1, C2)`` Euclidean curve value and 1st/2nd derivatives."""
    Ph = _to_homog_curve(C, rational=rational)
    Ch, Chd, Ch2 = eval_bezier_curve_homog_with_derivs(Ph, float(t), True)
    pt, d1, d2 = project_curve_homog_to_cartesian(Ch, Chd, Ch2)
    return np.asarray(pt), np.asarray(d1), np.asarray(d2)


def eval_surface_d2(S, u, v, rational=True):
    """Return ``(point, Su, Sv, Suu, Suv, Svv)`` Euclidean surface value/derivatives."""
    Sh = _to_homog_surface(S, rational=rational)
    Sh0, Shu, Shv, Shuu, Shuv, Shvv = eval_bezier_surface_homog_with_derivs(Sh, float(u), float(v), True)
    pt, su, sv, suu, suv, svv = project_surface_homog_to_cartesian(Sh0, Shu, Shv, Shuu, Shuv, Shvv)
    return (np.asarray(pt), np.asarray(su), np.asarray(sv),
            np.asarray(suu), np.asarray(suv), np.asarray(svv))


def _surface_g_derivs(S, point, u, v, rational):
    """Pointwise g = ||S(u,v)-point||^2 and its 1st/2nd derivatives.

    Returns (g, g_u, g_v, g_uu, g_uv, g_vv).
    """
    pt, su, sv, suu, suv, svv = eval_surface_d2(S, u, v, rational=rational)
    d = pt - point
    g = float(np.dot(d, d))
    gu = 2.0 * float(np.dot(d, su))
    gv = 2.0 * float(np.dot(d, sv))
    guu = 2.0 * (float(np.dot(su, su)) + float(np.dot(d, suu)))
    guv = 2.0 * (float(np.dot(su, sv)) + float(np.dot(d, suv)))
    gvv = 2.0 * (float(np.dot(sv, sv)) + float(np.dot(d, svv)))
    return g, gu, gv, guu, guv, gvv


# ---------------------------------------------------------------------------
# Newton kernels
# ---------------------------------------------------------------------------

def newton_curve_closest_point(C, point, u0, *, rational=False,
                               tol=1e-14, step_tol=1e-14, max_it=30, lm_damp=1e-12,
                               bounds=(0.0, 1.0)):
    """LM-damped 1D closest-point solve of ``<C(u)-point, C'(u)> = 0``.

    Clamped to ``bounds=(lo, hi)``. Returns ``(u, R, sqdist, last_du)`` with
    ``R = C(u) - point``.
    """
    point = np.asarray(point, dtype=np.float64)
    lo, hi = bounds
    u = min(max(float(u0), lo), hi)
    last_du = 1.0
    for _ in range(max_it):
        p, d = eval_curve_d1(C, u, rational=rational)
        R = p - point
        sq = float(np.dot(R, R))
        g = float(np.dot(R, d))
        if abs(g) < tol or (u <= lo and g >= -tol) or (u >= hi and g <= tol):
            last_du = 0.0
            break
        A = float(np.dot(d, d)) + lm_damp
        if A <= 0.0 or not np.isfinite(A):
            last_du = 0.0
            break
        du = -g / A
        if du * du < step_tol * step_tol:
            last_du = float(du)
            break
        step = 1.0
        accepted = False
        for _ls in range(8):
            un = min(max(u + step * du, lo), hi)
            Rn = eval_curve(C, un, rational=rational) - point
            if float(np.dot(Rn, Rn)) <= sq:
                last_du = un - u
                u = un
                accepted = True
                break
            step *= 0.5
        if not accepted:
            last_du = 0.0
            break
    R = eval_curve(C, u, rational=rational) - point
    return u, R, float(np.dot(R, R)), last_du


def newton_surface_closest_point(S, point, u0, v0, *, rational=False,
                                 tol=1e-13, step_tol=1e-14, max_it=40, lm_damp=1e-10,
                                 bounds=None):
    """LM-damped 2x2 closest-point solve of the stationarity system

        r_u = <S(u,v)-point, S_u> = 0
        r_v = <S(u,v)-point, S_v> = 0

    The Jacobian is the Hessian of (1/2)||S-point||^2. ``bounds`` =
    ``(u_lo,u_hi,v_lo,v_hi)`` clamps iterates to the current cell.
    Returns ``(u, v, R, last_step)`` with ``R = (r_u, r_v)``.
    """
    point = np.asarray(point, dtype=np.float64)
    if bounds is None:
        u_lo, u_hi, v_lo, v_hi = 0.0, 1.0, 0.0, 1.0
    else:
        u_lo, u_hi, v_lo, v_hi = bounds
    u = min(max(float(u0), u_lo), u_hi)
    v = min(max(float(v0), v_lo), v_hi)
    last_step = (1.0, 1.0)

    def residual(uu, vv):
        pt, su, sv, suu, suv, svv = eval_surface_d2(S, uu, vv, rational=rational)
        dvec = pt - point
        r = np.array([np.dot(dvec, su), np.dot(dvec, sv)])
        H = np.array([
            [np.dot(su, su) + np.dot(dvec, suu), np.dot(su, sv) + np.dot(dvec, suv)],
            [np.dot(su, sv) + np.dot(dvec, suv), np.dot(sv, sv) + np.dot(dvec, svv)],
        ])
        return r, H, float(np.dot(dvec, dvec))

    for _ in range(max_it):
        r, H, sq = residual(u, v)
        if float(np.dot(r, r)) < tol * tol:
            last_step = (0.0, 0.0)
            break
        A = H + lm_damp * np.eye(2)
        try:
            delta = np.linalg.solve(A, -r)
        except np.linalg.LinAlgError:
            last_step = (0.0, 0.0)
            break
        if float(np.dot(delta, delta)) < step_tol * step_tol:
            last_step = (float(delta[0]), float(delta[1]))
            break
        step = 1.0
        accepted = False
        rn2 = float(np.dot(r, r))
        for _ls in range(10):
            un = min(max(u + step * delta[0], u_lo), u_hi)
            vn = min(max(v + step * delta[1], v_lo), v_hi)
            rr, _, _ = residual(un, vn)
            if float(np.dot(rr, rr)) <= rn2:
                last_step = (un - u, vn - v)
                u, v = un, vn
                accepted = True
                break
            step *= 0.5
        if not accepted:
            last_step = (0.0, 0.0)
            break
    r, _, _ = residual(u, v)
    return u, v, r, last_step


# ---------------------------------------------------------------------------
# Shared cell helpers
# ---------------------------------------------------------------------------
from mmcore.numeric.bern import de_casteljau_split_nd
from mmcore.geom._nurbs_param_tol import bez_curve_param_tolerance, bez_surface_param_tolerance


def _split_net(net, axis, t=0.5):
    L, R = de_casteljau_split_nd(net[..., None], axis=axis, t=t)
    return L[..., 0], R[..., 0]


def _hull_excludes_zero(net):
    return float(net.min()) > 0.0 or float(net.max()) < 0.0


def _g_curve_value(F, Qw, t):
    return float(bern_sq_dist.eval_point_curve_distance_sq(F, Qw, t))


def _ratio_dist_bounds(F, W2):
    """Tight rational Bernstein bounds on the DISTANCE over a cell.

    ``g = F/W`` with ``W = w⊗w`` of the SAME (bi)degree as ``F``, so the
    coefficient-wise ratios bound the rational function (exact for constant g,
    e.g. a sphere about its center). Returns ``(d_lo, d_hi)``.
    """
    r = F / np.maximum(W2, 1e-300)
    return (np.sqrt(max(float(r.min()), 0.0)),
            np.sqrt(max(float(r.max()), 0.0)))


def _is_stationary_point(S, point, u, v, rational):
    """Angular stationarity test: residual ⟂ both tangents (foot-point condition).

    ``|cos(angle(residual, tangent))| < _STATIONARY_COS_TOL`` for both S_u, S_v.
    Points with residual ~ 0 (query on the surface) count as stationary.
    Degenerate tangents (|S_a| ~ 0, e.g. at a pole) are skipped.
    """
    pt, su, sv = eval_surface_d1(S, u, v, rational=rational)
    dvec = pt - point
    d = float(np.linalg.norm(dvec))
    if d < 1e-14:
        return True
    for tang in (su, sv):
        tn = float(np.linalg.norm(tang))
        if tn < 1e-14:
            continue
        if abs(float(np.dot(dvec, tang))) / (d * tn) > _STATIONARY_COS_TOL:
            return False
    return True


# ---------------------------------------------------------------------------
# Curve core (band semantics)
# ---------------------------------------------------------------------------

def bez_curve_closest_points(C, point, atol=1e-3, rational=False,
                             max_cells=20000, upper_bound=None, stats=None):
    """Globally closest points of a Bézier curve (band semantics).

    Returns the set of entities within ``atol`` of the global minimum
    distance, sorted ascending: point entities ``{"t","point","distance",
    "kind"}`` or a single whole-segment entity ``{"kind":
    "degenerate_segment", ...}`` when the entire segment is equidistant
    within tolerance (per single Bézier segment the equidistant set is
    provably all-or-nothing).

    ``upper_bound``: externally-known distance bound (e.g. from sibling
    patches) used for band clipping only.  When supplied, ``stats`` is updated
    with ``cells_processed`` and ``budget_exhausted``.
    """
    C = np.asarray(C, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    N, F, Qw = point_curve_stationarity_net(point, C, rational=rational)
    W2 = _bernstein_product_nd(Qw, Qw)
    ptol = float(bez_curve_param_tolerance(C, atol, rational=rational))
    ptol = max(ptol, _PTOL_FLOOR)
    band = atol

    # Whole-segment certificate: distance range over the entire segment is
    # below atol -> every point is "closest within tolerance". (Per-segment
    # partial equidistant arcs are impossible: F - d^2*W == 0 on a sub-arc
    # forces it identically by the polynomial identity theorem.)
    d_lo0, d_hi0 = _ratio_dist_bounds(F, W2)
    if d_hi0 - d_lo0 < band:
        pt = eval_curve(C, 0.5, rational=rational)
        dist = float(np.linalg.norm(pt - point))
        _publish_work_stats(stats, 0, False)
        return [{"kind": "degenerate_segment", "t_range": (0.0, 1.0),
                 "t": 0.5, "point": np.asarray(pt), "distance": dist}]

    best = np.inf if upper_bound is None else float(upper_bound)
    # Seed the bound with the endpoints (always candidates anyway).
    for t_end in (0.0, 1.0):
        best = min(best, np.sqrt(max(_g_curve_value(F, Qw, t_end), 0.0)))

    candidates = []  # (t, distance, kind)

    def add_candidate(t, kind):
        nonlocal best
        t = min(max(float(t), 0.0), 1.0)
        dist = np.sqrt(max(_g_curve_value(F, Qw, t), 0.0))
        best = min(best, dist)
        for ct, _, _ in candidates:
            if abs(ct - t) < ptol:
                return
        candidates.append((t, dist, kind))

    max_cells = max(int(max_cells), 0)
    cells = 0
    stack = [(N, F, W2, 0.0, 1.0, 0)]
    while stack and cells < max_cells:
        cells += 1
        Ncell, Fc, W2c, t0, t1, depth = stack.pop()
        d_lo, _ = _ratio_dist_bounds(Fc, W2c)
        if d_lo > best + band:
            continue                      # band clip: provably not near-minimal
        if _hull_excludes_zero(Ncell):
            continue                      # no stationary point in the cell
        if (t1 - t0) <= ptol or depth > 60:
            u, _, _, _ = newton_curve_closest_point(C, point, 0.5 * (t0 + t1),
                                                    rational=rational, bounds=(t0, t1))
            add_candidate(u, "min")
            continue
        NL, NR = _split_net(Ncell, 0)
        FL, FR = _split_net(Fc, 0)
        WL, WR = _split_net(W2c, 0)
        tm = 0.5 * (t0 + t1)
        stack.append((NL, FL, WL, t0, tm, depth + 1))
        stack.append((NR, FR, WR, tm, t1, depth + 1))

    budget_exhausted = bool(stack and cells >= max_cells)
    if budget_exhausted:
        warnings.warn(
            "bez_curve_closest_points: subdivision hit max_cells cap; "
            "result may be incomplete.")

    add_candidate(0.0, "boundary_min")
    add_candidate(1.0, "boundary_min")

    # Classify: kind from parameter position. Boundary candidates accepted only
    # if KKT (no descent into the interior); interior minima need g'' > 0.
    results = []
    for t, dist, _kind in candidates:
        pt, c1, c2 = eval_curve_d2(C, t, rational=rational)
        dvec = pt - point
        gpp = float(np.dot(c1, c1) + np.dot(dvec, c2))
        gp = float(np.dot(dvec, c1))
        is_boundary = (t <= ptol) or (t >= 1.0 - ptol)
        if is_boundary:
            if t <= ptol and gp < -atol:
                continue
            if t >= 1.0 - ptol and gp > atol:
                continue
            results.append({"t": t, "point": np.asarray(pt), "distance": dist,
                            "kind": "boundary_min"})
        else:
            if gpp <= 0.0:
                continue
            results.append({"t": t, "point": np.asarray(pt), "distance": dist,
                            "kind": "min"})

    if not results and candidates:   # degenerate fallback
        t, dist, kind = min(candidates, key=lambda c: c[1])
        pt = eval_curve(C, t, rational=rational)
        results.append({"t": t, "point": np.asarray(pt), "distance": dist, "kind": kind})

    # Band filter: keep only entities within atol of the (local) minimum.
    if results:
        gmin = min(e["distance"] for e in results)
        results = [e for e in results if e["distance"] <= gmin + band]
    results.sort(key=lambda e: e["distance"])
    _publish_work_stats(stats, cells, budget_exhausted)
    return results


# ---------------------------------------------------------------------------
# Surface helpers
# ---------------------------------------------------------------------------

def _classify_surface_min(S, point, u, v, rational, atol):
    """Return (is_min, dist, pt): stationary + positive-definite Hessian."""
    pt, su, sv, suu, suv, svv = eval_surface_d2(S, u, v, rational=rational)
    dvec = pt - point
    dist = float(np.linalg.norm(dvec))
    # stationarity (foot-point condition), except query-on-surface
    if dist > 1e-14:
        for tang in (su, sv):
            tn = float(np.linalg.norm(tang))
            if tn > 1e-14 and abs(float(np.dot(dvec, tang))) / (dist * tn) > _STATIONARY_COS_TOL:
                return False, dist, np.asarray(pt)
    H11 = float(np.dot(su, su) + np.dot(dvec, suu))
    H22 = float(np.dot(sv, sv) + np.dot(dvec, svv))
    H12 = float(np.dot(su, sv) + np.dot(dvec, suv))
    det = H11 * H22 - H12 * H12
    is_min = (H11 > 0.0) and (det >= -abs(H11 * H22) * 1e-12)
    return is_min, dist, np.asarray(pt)


def _dedup_add(out, u, v, dist, pt, kind, ptol_u, ptol_v, atol):
    # Duplicate if the parameters coincide OR the geometric point coincides.
    for e in out:
        if ((abs(e["u"] - u) < ptol_u and abs(e["v"] - v) < ptol_v)
                or np.linalg.norm(e["point"] - pt) < max(atol, 1e-9)):
            return
    out.append({"u": float(u), "v": float(v), "point": pt, "distance": float(dist), "kind": kind})


def _hessian_eigs(S, point, u, v, rational):
    """Eigen-decomposition of the Hessian of g at (u, v).

    Returns ``(lam_min, lam_max, null_dir, g)`` with ``null_dir`` the unit
    eigenvector of the smallest eigenvalue (the equidistant-curve tangent when
    the Hessian is rank-1: differentiate grad g(γ(s)) = 0 -> H·γ' = 0).
    """
    g, gu, gv, guu, guv, gvv = _surface_g_derivs(S, point, u, v, rational)
    H = np.array([[guu, guv], [guv, gvv]])
    lam, vec = np.linalg.eigh(H)
    return float(lam[0]), float(lam[1]), vec[:, 0], g


# ---------------------------------------------------------------------------
# Equidistant-curve tracer (Pull-Curve style; source degenerated to a point)
# ---------------------------------------------------------------------------

def trace_equidistant_curve(S, point, u0, v0, *, rational=False, step=0.01,
                            max_steps=2000, dist_band=None,
                            max_total_steps=None, stats=None):
    """Trace the 1-D equidistant stationary curve of ``g = ||S - point||^2``
    through the degenerate stationary seed ``(u0, v0)``.

    Predictor: step along the Hessian null eigenvector (the curve tangent,
    since ``H·γ' = 0``). Corrector: LM Newton on the stationarity system
    (its minimum-norm step is transverse to the null space, so there is no
    tangential slide-back). Traces both directions to the domain boundary or
    to closure (a closed loop inside the patch).

    Returns ``{"uv": (N,2), "points": (N,3), "distance", "closed"}`` or
    ``None`` when the trace is invalid (corrector failure right away, or the
    distance drifts out of ``dist_band`` — the seed was not on a genuine
    equidistant curve; caller falls back to isolated handling). The optional
    ``max_total_steps`` caps both directions together; ``stats`` reports
    ``steps_processed`` and whether that cap (or a per-leg cap) truncated the
    trace. A truncated trace is never returned as a complete curve.
    """
    S = np.asarray(S, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    max_steps = max(0, int(max_steps))
    if max_total_steps is None:
        max_total_steps = 2 * max_steps
    max_total_steps = max(0, int(max_total_steps))
    steps_processed = 0

    def finish(result, budget_exhausted=False):
        if stats is not None:
            stats.clear()
            stats.update(steps_processed=int(steps_processed),
                         budget_exhausted=bool(budget_exhausted))
        return result

    lam0, lam1, _, g0 = _hessian_eigs(S, point, u0, v0, rational)
    d0 = np.sqrt(max(g0, 0.0))
    if dist_band is None:
        dist_band = max(1e-9, 1e-6 * max(d0, 1.0))
    scale = max(abs(lam1), 1e-30)
    if abs(lam1) < 1e-30 or abs(lam0) > _DEGEN_EIG_RATIO * scale:
        return finish(None)  # not rank-1 degenerate (rank-0 flat or isolated)

    def on_boundary(u, v):
        eps = 1e-12
        return u <= eps or u >= 1.0 - eps or v <= eps or v >= 1.0 - eps

    def transverse_correct(u, v):
        """Correct onto the equidistant manifold along the TRANSVERSE
        direction only (the largest-|eigenvalue| eigenvector of the Hessian).

        A full 2x2 Newton corrector is wrong here: on real data the ring is
        degenerate only to rounding (lam_min != 0 exactly), and full Newton
        also converges TANGENTIALLY toward the nearest noise-scale stationary
        point, cancelling the predictor's progress (observed as thousands of
        trace points with no net motion). Correcting only along the transverse
        eigenvector leaves the tangential coordinate untouched.
        Returns (u, v, g, ok)."""
        g = None
        for _ in range(8):
            g, gu, gv, guu, guv, gvv = _surface_g_derivs(S, point, u, v, rational)
            H = np.array([[guu, guv], [guv, gvv]])
            lam, vec = np.linalg.eigh(H)
            j = int(np.argmax(np.abs(lam)))
            e = vec[:, j]
            lam_t = float(lam[j])
            if abs(lam_t) < 1e-300:
                return u, v, g, False
            r = gu * e[0] + gv * e[1]        # gradient along transverse dir
            delta = -r / lam_t
            if abs(delta) > 4.0 * step:
                return u, v, g, False        # corrector diverging
            un = min(max(u + delta * e[0], 0.0), 1.0)
            vn = min(max(v + delta * e[1], 0.0), 1.0)
            moved = float(np.hypot(un - u, vn - v))
            u, v = un, vn
            if moved < 1e-15:
                break
        return u, v, g, True

    drift_tol = max(dist_band, 1e-7 * max(d0, 1.0))
    closed = False
    truncated = False
    legs = []
    for direction in (1.0, -1.0):
        u, v = float(u0), float(v0)
        prev_t = None
        leg = []
        cum_arc = 0.0
        stall_run = 0
        for _k in range(max_steps):
            if steps_processed >= max_total_steps:
                truncated = True
                break
            steps_processed += 1
            _, _, t, _ = _hessian_eigs(S, point, u, v, rational)
            t = t * direction
            if prev_t is not None and float(np.dot(t, prev_t)) < 0.0:
                t = -t
            prev_t = t
            un, vn = u + step * t[0], v + step * t[1]
            hit_edge = not (0.0 <= un <= 1.0 and 0.0 <= vn <= 1.0)
            un = min(max(un, 0.0), 1.0)
            vn = min(max(vn, 0.0), 1.0)
            uc, vc, g, ok = transverse_correct(un, vn)
            if not ok or g is None:
                break
            if abs(np.sqrt(max(g, 0.0)) - d0) > drift_tol:
                break                     # drifted off the equidistant level
            adv = float(np.hypot(uc - u, vc - v))
            if adv < 0.05 * step:
                stall_run += 1            # no net progress: don't spin forever
                if stall_run >= 3:
                    break
            else:
                stall_run = 0
            leg.append((uc, vc))
            cum_arc += adv
            u, v = uc, vc
            if hit_edge or on_boundary(u, v):
                break
            # closure: back near the seed AFTER genuinely travelling (a
            # stalled trace hovering at the seed must not read as a loop)
            if direction > 0 and cum_arc > 10.0 * step and np.hypot(u - u0, v - v0) < 0.75 * step:
                closed = True
                break
        else:
            # Reaching a continuation cap is not a geometric termination
            # certificate, so the accumulated polyline is only partial.
            truncated = True
        legs.append(leg)
        if closed or truncated:
            break

    if truncated:
        return finish(None, True)
    fwd, bwd = legs[0], (legs[1] if len(legs) > 1 else [])
    pts_uv = bwd[::-1] + [(float(u0), float(v0))] + fwd
    if len(pts_uv) < 3:
        return finish(None)
    uv = np.array(pts_uv, dtype=np.float64)
    # Minimum-extent validity: a trace that never achieved real arc length is
    # a failed trace (it must fall back to isolated handling and must NOT
    # consume other seeds or emit a micro-fragment entity).
    arc_len = float(np.sum(np.hypot(np.diff(uv[:, 0]), np.diff(uv[:, 1]))))
    if arc_len < 3.0 * step:
        return finish(None)
    xyz = np.array([eval_surface(S, uu, vv, rational=rational) for uu, vv in pts_uv])
    dists = np.linalg.norm(xyz - point[None, :], axis=1)
    return finish({"uv": uv, "points": xyz,
                   "distance": float(np.mean(dists)),
                   "closed": bool(closed)})


def _uv_dist_to_polyline(u, v, uv):
    return float(np.min(np.hypot(uv[:, 0] - u, uv[:, 1] - v)))


def _certify_circle(points, query):
    """Exact-circle certificate for a traced equidistant curve.

    Every equidistant curve lies ON the sphere of radius d_min about the
    query point (tangentially — the surface cannot enter the sphere). A
    PLANAR curve on a sphere is exactly a circle (plane ∩ sphere), so
    planarity of the traced points certifies a circle: the common case of
    any surface of revolution queried on its axis (cones, cylinders, tori).

    Returns ``{"center", "normal", "radius", "arc_angle"}`` (arc_angle is the
    swept angle in radians; ~2π for a closed ring) or ``None`` when the
    spherical curve is genuinely non-planar.
    """
    X = np.asarray(points, dtype=np.float64)
    if len(X) < 3:
        return None
    c = X.mean(axis=0)
    Y = X - c
    scale = float(np.max(np.linalg.norm(Y, axis=1)))
    if scale < 1e-300:
        return None
    _, sing, Vt = np.linalg.svd(Y, full_matrices=False)
    n = Vt[-1]
    if sing[-1] > 1e-6 * scale:
        return None                                  # non-planar
    query = np.asarray(query, dtype=np.float64)
    center = query + n * float(np.dot(c - query, n))  # sphere center -> plane
    r = np.linalg.norm(X - center, axis=1)
    radius = float(np.mean(r))
    if float(np.max(r) - np.min(r)) > 1e-6 * max(scale, 1e-12):
        return None
    e1 = (X[0] - center) / max(float(np.linalg.norm(X[0] - center)), 1e-300)
    e2 = np.cross(n, e1)
    ang = np.unwrap(np.arctan2((X - center) @ e2, (X - center) @ e1))
    return {"center": center, "normal": n, "radius": radius,
            "arc_angle": float(np.max(ang) - np.min(ang))}


# ---------------------------------------------------------------------------
# Surface core: best-first branch-and-bound (band semantics)
# ---------------------------------------------------------------------------

def bez_surface_closest_points(S, point, atol=1e-3, rational=False,
                               want_eval=False, max_cells=20000,
                               upper_bound=None, _interior_only=False,
                               stats=None):
    """Globally closest entities of a Bézier surface patch (band semantics).

    Returns entities within ``atol`` of the global minimum distance, sorted
    ascending by distance:

    * ``{"u","v","point","distance","kind":"min"|"boundary_min"[, "eval"]}``
    * ``{"kind":"degenerate_curve", "uv", "points", "closed", ...}`` — a 1-D
      equidistant curve, traced (Pull-Curve style);
    * ``{"kind":"degenerate_surface", ...}`` — the whole patch is equidistant.

    ``upper_bound``: externally-known distance bound used for band clipping.
    The four boundary-curve searches, surface subdivision queue, and
    degenerate-curve continuation share the single ``max_cells`` allowance.
    When supplied, ``stats`` is updated with total, boundary, surface, and
    trace-step counts plus ``budget_exhausted``.
    """
    S = np.asarray(S, dtype=np.float64)
    point = np.asarray(point, dtype=np.float64)
    Nu, Nv, F, Sw = point_surface_stationarity_nets(point, S, rational=rational)
    W2 = _bernstein_product_nd(Sw, Sw)
    ptol_u, ptol_v = bez_surface_param_tolerance(S, atol, rational=rational)
    ptol_u = max(float(ptol_u), _PTOL_FLOOR)
    ptol_v = max(float(ptol_v), _PTOL_FLOOR)
    band = atol

    # Whole-patch certificate: the entire patch is equidistant within atol
    # (e.g. a sphere about its center — there the rational ratio F_i/W_i is
    # exactly constant and this fires at the root). Boundary entities are
    # subsumed by the patch entity.
    d_lo0, d_hi0 = _ratio_dist_bounds(F, W2)
    if d_hi0 - d_lo0 < band:
        pt = eval_surface(S, 0.5, 0.5, rational=rational)
        dist = float(np.linalg.norm(pt - point))
        _publish_work_stats(stats, 0, False, boundary_cells=0,
                            surface_cells=0, trace_steps=0)
        return [{"kind": "degenerate_surface", "u_range": (0.0, 1.0),
                 "v_range": (0.0, 1.0), "u": 0.5, "v": 0.5,
                 "point": np.asarray(pt), "distance": dist}]

    best = np.inf if upper_bound is None else float(upper_bound)
    points_out = []      # interior point candidates
    curves_out = []      # degenerate curve entities
    degen_seeds = []     # (u, v) rank-1 stationary seeds

    # Boundary first: cheap, seeds `best` early, and its entities join the
    # final band filter like everything else.
    max_cells = max(int(max_cells), 0)
    boundary_points = []
    boundary_curves = []
    boundary_stats = {"cells_processed": 0, "budget_exhausted": False}
    if not _interior_only:
        _surface_boundary_entities(S, point, boundary_points, boundary_curves,
                                   rational, atol, ptol_u, ptol_v,
                                   max_cells=max_cells, stats=boundary_stats)
        for e in boundary_points:
            best = min(best, e["distance"])
        for e in boundary_curves:
            best = min(best, e["distance"])
    boundary_cells = int(boundary_stats["cells_processed"])
    budget_exhausted = bool(boundary_stats["budget_exhausted"])

    def try_add_point(u, v):
        nonlocal best
        is_min, dist, pt = _classify_surface_min(S, point, u, v, rational, atol)
        best = min(best, dist)          # any surface point bounds the minimum
        if is_min:
            _dedup_add(points_out, u, v, dist, pt, "min", ptol_u, ptol_v, atol)
        return dist

    counter = itertools.count()
    pq = [(d_lo0, next(counter), F, W2, Nu, Nv, 0.0, 1.0, 0.0, 1.0, 0)]
    pops = 0
    capped = False
    # Ledger L46: the interior best-first heap used to share one
    # ``max_cells`` with the four boundary searches, so a boundary phase
    # that burned the whole allowance starved the interior to ZERO pops
    # and the true interior global minimum silently dropped out of the
    # certified set (the band-semantics contract).  The interior keeps its
    # own pop allowance; boundary exhaustion still marks the RESULT capped
    # below, and the published work counters report the true total.
    interior_cap = max(1, int(max_cells))
    capped = capped or budget_exhausted
    while pq:
        if pops >= interior_cap:
            capped = True
            budget_exhausted = True
            break
        lb, _, Fc, W2c, Nuc, Nvc, u0, u1, v0, v1, depth = heapq.heappop(pq)
        pops += 1
        if lb > best + band:
            break        # best-first: every remaining cell is at least this far
        if _hull_excludes_zero(Nuc) or _hull_excludes_zero(Nvc):
            continue     # no interior stationary point in the cell
        d_lo, d_hi = _ratio_dist_bounds(Fc, W2c)
        if d_hi - d_lo < band:
            # Uniformly in-band flat cell: polish and classify by Hessian rank.
            um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            u, v, _, _ = newton_surface_closest_point(
                S, point, um, vm, rational=rational, bounds=(u0, u1, v0, v1))
            lam0, lam1, _, g = _hessian_eigs(S, point, u, v, rational)
            best = min(best, np.sqrt(max(g, 0.0)))
            if lam0 > _DEGEN_EIG_RATIO * max(abs(lam1), 1e-30):
                try_add_point(u, v)
            else:
                if _is_stationary_point(S, point, u, v, rational):
                    degen_seeds.append((u, v))
            continue
        small = (u1 - u0) <= ptol_u and (v1 - v0) <= ptol_v
        if small or depth > 60:
            um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
            u, v, _, _ = newton_surface_closest_point(
                S, point, um, vm, rational=rational, bounds=(u0, u1, v0, v1))
            try_add_point(u, v)
            continue
        # Newton probe from the cell centre: establishes/improves `best` early
        # (the probe is what makes the band prune bite before deep subdivision).
        um, vm = 0.5 * (u0 + u1), 0.5 * (v0 + v1)
        u, v, _, _ = newton_surface_closest_point(
            S, point, um, vm, rational=rational, bounds=(u0, u1, v0, v1))
        try_add_point(u, v)
        # Subdivide the wider axis; push children with their own bounds.
        ax = 0 if (u1 - u0) >= (v1 - v0) else 1
        FL, FR = _split_net(Fc, ax)
        WL, WR = _split_net(W2c, ax)
        AL, AR = _split_net(Nuc, ax)
        BL, BR = _split_net(Nvc, ax)
        if ax == 0:
            um2 = 0.5 * (u0 + u1)
            kids = [(FL, WL, AL, BL, u0, um2, v0, v1), (FR, WR, AR, BR, um2, u1, v0, v1)]
        else:
            vm2 = 0.5 * (v0 + v1)
            kids = [(FL, WL, AL, BL, u0, u1, v0, vm2), (FR, WR, AR, BR, u0, u1, vm2, v1)]
        for (Fk, Wk, Ak, Bk, a0, a1, b0, b1) in kids:
            lbk, _ = _ratio_dist_bounds(Fk, Wk)
            if lbk > best + band:
                continue          # sound: `best` only decreases
            heapq.heappush(pq, (lbk, next(counter), Fk, Wk, Ak, Bk,
                                a0, a1, b0, b1, depth + 1))

    if capped:
        warnings.warn(
            "bez_surface_closest_points: hit max_cells cap; "
            "result may be incomplete.")

    # Trace equidistant curves from unconsumed degenerate seeds. Continuation
    # iterations consume the same allowance as subdivision cells; a truncated
    # polyline is never promoted to a complete closest-set entity.
    trace_step = 0.01
    trace_steps = 0
    trace_capped = False
    if (degen_seeds and not budget_exhausted
            and boundary_cells + pops >= max_cells):
        budget_exhausted = True
        trace_capped = True
    trace_seeds = () if budget_exhausted else degen_seeds
    for (su_, sv_) in trace_seeds:
        if any(_uv_dist_to_polyline(su_, sv_, c["uv"]) < 2.0 * trace_step
               for c in curves_out):
            continue                       # already covered by a traced curve
        remaining = max_cells - boundary_cells - pops - trace_steps
        if remaining <= 0:
            budget_exhausted = True
            trace_capped = True
            break
        trace_stats = {}
        tr = trace_equidistant_curve(S, point, su_, sv_, rational=rational,
                                     step=trace_step, dist_band=band,
                                     max_steps=min(2000, remaining),
                                     max_total_steps=remaining,
                                     stats=trace_stats)
        if 'steps_processed' not in trace_stats:
            # Fail closed if a substituted/older tracer omits accounting.
            charged_steps = remaining
            trace_incomplete = True
        else:
            reported_steps = max(0, int(trace_stats['steps_processed']))
            charged_steps, overrun = reconcile_reported(
                reported_steps, remaining)
            trace_incomplete = (
                bool(trace_stats.get('budget_exhausted', False)) or overrun)
        trace_steps += charged_steps
        if trace_incomplete:
            budget_exhausted = True
            trace_capped = True
            break
        if tr is None:
            try_add_point(su_, sv_)        # fall back to isolated handling
            continue
        tr["u"] = float(tr["uv"][len(tr["uv"]) // 2, 0])
        tr["v"] = float(tr["uv"][len(tr["uv"]) // 2, 1])
        tr["point"] = np.asarray(tr["points"][len(tr["points"]) // 2])
        tr["kind"] = "degenerate_curve"
        cert = _certify_circle(tr["points"], point)
        if cert is not None:
            tr["circle"] = cert     # planar spherical curve == exact circle
        curves_out.append(tr)

    if trace_capped and not capped:
        warnings.warn(
            "bez_surface_closest_points: degenerate continuation hit the "
            "shared max_cells cap; result may be incomplete.")

    # A traced curve consumes point candidates lying on it.
    def consumed_by_curve(e):
        return any(abs(e["distance"] - c["distance"]) <= band
                   and _uv_dist_to_polyline(e["u"], e["v"], c["uv"]) < 2.0 * trace_step
                   for c in curves_out)

    merged_points = []
    for e in points_out + boundary_points:
        if consumed_by_curve(e):
            continue
        _dedup_add(merged_points, e["u"], e["v"], e["distance"], e["point"],
                   e["kind"], ptol_u, ptol_v, atol)

    entities = merged_points + curves_out + boundary_curves
    if not entities:      # paranoia fallback
        u, v, _, _ = newton_surface_closest_point(S, point, 0.5, 0.5, rational=rational)
        pt = eval_surface(S, u, v, rational=rational)
        entities = [{"u": float(u), "v": float(v), "point": np.asarray(pt),
                     "distance": float(np.linalg.norm(pt - point)), "kind": "min"}]

    # Final band filter and sort.
    gmin = min(e["distance"] for e in entities)
    entities = [e for e in entities if e["distance"] <= gmin + band]
    entities.sort(key=lambda e: e["distance"])
    if want_eval:
        for e in entities:
            if e["kind"] in ("min", "boundary_min"):
                pt, su, sv, _, _, _ = eval_surface_d2(S, e["u"], e["v"], rational=rational)
                e["eval"] = {"S": pt, "Su": su, "Sv": sv, "normal": np.cross(su, sv)}
    _publish_work_stats(
        stats, boundary_cells + pops + trace_steps, budget_exhausted,
        boundary_cells=boundary_cells, surface_cells=pops,
        trace_steps=trace_steps,
    )
    return entities


# ---------------------------------------------------------------------------
# Surface boundary handling
# ---------------------------------------------------------------------------

def _surface_boundary_entities(S, point, out_points, out_curves,
                               rational, atol, ptol_u, ptol_v,
                               max_cells=20000, stats=None):
    """Collect KKT-valid minima on the 4 edges + 4 corners of the patch.

    Point results go to ``out_points``; a whole-edge equidistant segment
    (e.g. a degenerate pole edge) becomes a degenerate-curve entity along
    the edge in ``out_curves``.
    """
    point = np.asarray(point, dtype=np.float64)
    edges = [
        (0, 0.0, S[0, :, :]),    # u = 0, runs along v
        (0, 1.0, S[-1, :, :]),   # u = 1
        (1, 0.0, S[:, 0, :]),    # v = 0, runs along u
        (1, 1.0, S[:, -1, :]),   # v = 1
    ]
    max_cells = max(int(max_cells), 0)
    cells = 0
    budget_exhausted = False
    for fixed_axis, side, iso in edges:
        remaining = max_cells - cells
        if remaining <= 0:
            budget_exhausted = True
            break
        curve_stats = {}
        iso_res = bez_curve_closest_points(
            iso, point, atol=atol, rational=rational,
            max_cells=remaining, stats=curve_stats)
        cells += int(curve_stats["cells_processed"])
        budget_exhausted = bool(curve_stats["budget_exhausted"])
        for e in iso_res:
            if e["kind"] == "degenerate_segment":
                # Whole edge equidistant. KKT at the representative decides
                # whether the surface can descend inward from this edge.
                t_mid = e["t"]
                u, v = (side, t_mid) if fixed_axis == 0 else (t_mid, side)
                if not _boundary_kkt_ok(S, point, u, v, rational, atol, ptol_u, ptol_v):
                    continue
                n_smp = 17
                ts = np.linspace(0.0, 1.0, n_smp)
                if fixed_axis == 0:
                    uv = np.column_stack([np.full(n_smp, side), ts])
                else:
                    uv = np.column_stack([ts, np.full(n_smp, side)])
                xyz = np.array([eval_surface(S, uu, vv, rational=rational)
                                for uu, vv in uv])
                dists = np.linalg.norm(xyz - point[None, :], axis=1)
                out_curves.append({
                    "kind": "degenerate_curve", "uv": uv, "points": xyz,
                    "closed": False, "u": float(uv[n_smp // 2, 0]),
                    "v": float(uv[n_smp // 2, 1]),
                    "point": np.asarray(xyz[n_smp // 2]),
                    "distance": float(np.mean(dists)),
                })
                continue
            s = e["t"]
            u, v = (side, s) if fixed_axis == 0 else (s, side)
            _try_add_boundary(S, point, out_points, u, v, rational, atol, ptol_u, ptol_v)
        if budget_exhausted:
            break
    for u, v in [(0.0, 0.0), (1.0, 0.0), (0.0, 1.0), (1.0, 1.0)]:
        _try_add_boundary(S, point, out_points, u, v, rational, atol, ptol_u, ptol_v)
    _publish_work_stats(stats, cells, budget_exhausted)


def _boundary_kkt_ok(S, point, u, v, rational, atol, ptol_u, ptol_v):
    """KKT: the surface must not descend into the interior from (u, v)."""
    pt, su, sv = eval_surface_d1(S, u, v, rational=rational)
    dvec = pt - point
    gu = float(np.dot(dvec, su))
    gv = float(np.dot(dvec, sv))
    if u <= ptol_u and gu < -atol:
        return False
    if u >= 1.0 - ptol_u and gu > atol:
        return False
    if v <= ptol_v and gv < -atol:
        return False
    if v >= 1.0 - ptol_v and gv > atol:
        return False
    return True


def _try_add_boundary(S, point, out, u, v, rational, atol, ptol_u, ptol_v):
    """KKT filter + dedup for a boundary point candidate at (u, v)."""
    if not _boundary_kkt_ok(S, point, u, v, rational, atol, ptol_u, ptol_v):
        return
    pt = eval_surface(S, u, v, rational=rational)
    dist = float(np.linalg.norm(pt - point))
    _dedup_add(out, u, v, dist, np.asarray(pt), "boundary_min", ptol_u, ptol_v, atol)


# ---------------------------------------------------------------------------
# NURBS-level wrappers
# ---------------------------------------------------------------------------
from mmcore.geom.nurbs import NURBSCurve, NURBSSurface
from mmcore.geom._nurbs_eval import (
    _nurbs_to_tuple, _curve_interval, _surface_interval,
    to_homogeneous_1d, to_homogeneous_2d,
)
from mmcore.geom._nurbs_knots import decompose_curve, decompose_surface


def _patch_curve_net(patch):
    """Homogeneous (q+1,4) net for a single-span Bézier curve patch tuple.

    INVARIANT (relied on by the param mapping in the wrappers): this strips the
    knot vector and returns a raw control-point net, which the Bézier cores
    interpret on Bernstein coordinates ``[0, 1]``. Because ``decompose_curve``
    reconstructs each patch's control points geometrically (knot insertion), that
    ``[0, 1]`` Bernstein parameterization corresponds *exactly* to the patch's
    global knot interval ``[p_lo, p_hi]``. Hence ``t_global = p_lo + t_local *
    (p_hi - p_lo)`` is the correct inverse. Do NOT change this to return a
    NURBSCurveTuple without also fixing the mapping.
    """
    return np.asarray(to_homogeneous_1d(patch.control_points, patch.weights), dtype=np.float64)


def _patch_surface_net(patch):
    """Homogeneous (m+1,n+1,4) net for a single-span Bézier surface patch tuple.

    Same ``[0, 1]`` Bernstein invariant as ``_patch_curve_net`` (per axis).
    """
    return np.asarray(to_homogeneous_2d(patch.control_points, patch.weights), dtype=np.float64)


def nurbs_curve_closest_points(curve, point, atol=1e-3, *,
                               max_cells=None, stats=None):
    """Globally closest points of a NURBS curve (band semantics), in GLOBAL
    parameters, sorted ascending by distance.

    Point entities plus ``degenerate_segment`` entities (adjacent equidistant
    Bézier segments are merged, e.g. a full circle about its center collapses
    to one entity spanning the domain). Internal seams are deduped; only the
    global domain ends are ``boundary_min``.

    Ledger L46: ``max_cells`` is ONE shared allowance across every Bézier
    span (default: the per-span 20k scaled by the span count, so ordinary
    input never truncates while a hog span may borrow from cheap ones);
    ``stats`` publishes the aggregate ``cells_processed`` /
    ``budget_exhausted`` — a capped sub-solve can return far-local-min
    entities, so a consumer of the band-semantics contract must check it.
    """
    if isinstance(curve, NURBSCurve):
        curve = _nurbs_to_tuple(curve)
    point = np.asarray(point, dtype=np.float64)
    g_lo, g_hi = _curve_interval(curve)
    patches = decompose_curve(curve)
    band = atol
    remaining = (20_000 * max(1, len(patches)) if max_cells is None
                 else max(0, int(max_cells)))
    agg_cells = 0
    agg_exhausted = False

    # Geometric closure (clamped NURBS: endpoints are the end control points).
    # On a closed curve the domain ends are a periodic seam, not a boundary.
    cp = np.asarray(curve.control_points, dtype=np.float64)
    geo_scale = float(np.linalg.norm(cp.max(0) - cp.min(0)))
    closed = float(np.linalg.norm(cp[0] - cp[-1])) < 1e-9 * max(geo_scale, 1e-12)

    best = np.inf
    pts = []
    segs = []
    for patch in patches:
        if remaining <= 0:
            agg_exhausted = True
            break
        p_lo, p_hi = _curve_interval(patch)
        net = _patch_curve_net(patch)
        patch_stats = {}
        local = bez_curve_closest_points(net, point, atol=atol, rational=True,
                                         upper_bound=best,
                                         max_cells=remaining,
                                         stats=patch_stats)
        used = max(0, int(patch_stats.get("cells_processed", 0)))
        remaining -= reconcile_reported(used, remaining)[0]
        agg_cells += used
        agg_exhausted |= bool(patch_stats.get("budget_exhausted", False))
        for e in local:
            best = min(best, e["distance"])
            if e["kind"] == "degenerate_segment":
                lo = p_lo + e["t_range"][0] * (p_hi - p_lo)
                hi = p_lo + e["t_range"][1] * (p_hi - p_lo)
                segs.append({"kind": "degenerate_segment", "t_range": (lo, hi),
                             "t": 0.5 * (lo + hi), "point": e["point"],
                             "distance": e["distance"]})
                continue
            t_glob = p_lo + e["t"] * (p_hi - p_lo)
            kind = e["kind"]
            if kind == "boundary_min":
                at_ends = (abs(t_glob - g_lo) < 1e-9 or abs(t_glob - g_hi) < 1e-9)
                if closed or not at_ends:
                    kind = "min"      # seam (periodic or internal), not a real end
            _merge_curve_point(pts, t_glob, e["point"], e["distance"], kind, atol)

    if closed:
        pts = _merge_periodic_seam_1d(pts, g_lo, g_hi, atol, geo_scale)

    # Merge adjacent degenerate segments (equidistant arc spanning seams).
    segs.sort(key=lambda s: s["t_range"][0])
    merged_segs = []
    for s in segs:
        if (merged_segs
                and abs(s["t_range"][0] - merged_segs[-1]["t_range"][1]) < 1e-9
                and abs(s["distance"] - merged_segs[-1]["distance"]) <= band):
            prev = merged_segs[-1]
            prev["t_range"] = (prev["t_range"][0], s["t_range"][1])
            prev["t"] = 0.5 * (prev["t_range"][0] + prev["t_range"][1])
        else:
            merged_segs.append(dict(s))

    # Points lying inside a degenerate segment are subsumed by it.
    def inside_seg(e):
        return any(s["t_range"][0] - 1e-9 <= e["t"] <= s["t_range"][1] + 1e-9
                   and abs(e["distance"] - s["distance"]) <= band
                   for s in merged_segs)

    _publish_work_stats(stats, agg_cells, agg_exhausted)
    entities = [e for e in pts if not inside_seg(e)] + merged_segs
    if not entities:
        return []
    gmin = min(e["distance"] for e in entities)
    entities = [e for e in entities if e["distance"] <= gmin + band]
    entities.sort(key=lambda e: e["distance"])
    return entities


def _merge_curve_point(merged, t, pt, dist, kind, atol):
    pt = np.asarray(pt)
    for e in merged:
        if abs(e["t"] - t) < 1e-7 or np.linalg.norm(e["point"] - pt) < max(atol, 1e-9):
            if dist < e["distance"]:
                e.update(t=float(t), point=pt, distance=float(dist), kind=kind)
            elif kind == "boundary_min" and e["kind"] != "boundary_min":
                e["kind"] = "boundary_min"
            return
    merged.append({"t": float(t), "point": pt, "distance": float(dist), "kind": kind})


def _merge_periodic_seam_1d(pts, g_lo, g_hi, atol, geo_scale):
    """On a closed curve, a candidate near t=g_lo and one near t=g_hi can be
    the SAME physical point polished from the two sides of the periodic seam
    (their Newton slop puts them just outside the plain geometric dedup).
    Merge such pairs, keeping the nearer one."""
    span = max(g_hi - g_lo, 1e-300)
    seam_t = 1e-3 * span
    seam_geo = max(atol, 1e-6 * max(geo_scale, 1.0))
    out = []
    for e in sorted(pts, key=lambda x: x["distance"]):
        dup = False
        for kept in out:
            near_opposite_ends = (
                (e["t"] - g_lo < seam_t and g_hi - kept["t"] < seam_t)
                or (kept["t"] - g_lo < seam_t and g_hi - e["t"] < seam_t))
            if near_opposite_ends and np.linalg.norm(e["point"] - kept["point"]) < seam_geo:
                dup = True
                break
        if not dup:
            out.append(e)
    return out


def nurbs_surface_closest_points(surface, point, atol=1e-3, want_eval=False,
                                 *, max_cells=None, stats=None):
    """Globally closest entities of a NURBS surface (band semantics), in
    GLOBAL parameters, sorted ascending by distance.

    Point entities, stitched ``degenerate_curve`` entities (traces meeting at
    patch seams are chained; a full ring is marked ``closed``), and
    ``degenerate_surface`` entities. Internal seams are deduped; only the
    global domain border is ``boundary_min``.

    Ledger L46: ``max_cells`` is ONE shared allowance across every Bézier
    patch (default: the per-patch 20k scaled by the patch count);
    ``stats`` publishes the aggregate ``cells_processed`` /
    ``budget_exhausted``.  A capped sub-solve may return far-local-min
    entities, so any consumer relying on the band-semantics guarantee
    ("never far local minima") must check the exhaustion flag.
    """
    if isinstance(surface, NURBSSurface):
        surface = _nurbs_to_tuple(surface)
    point = np.asarray(point, dtype=np.float64)
    (gu_lo, gu_hi), (gv_lo, gv_hi) = _surface_interval(surface)
    patches = decompose_surface(surface)
    band = atol
    remaining = (20_000 * max(1, len(patches)) if max_cells is None
                 else max(0, int(max_cells)))
    agg_cells = 0
    agg_exhausted = False

    # Geometric closure per direction (clamped net: opposite border rows
    # coincide). A closed direction's domain ends are a periodic seam.
    cps = np.asarray(surface.control_points, dtype=np.float64)
    geo_scale = float(np.linalg.norm(cps.reshape(-1, cps.shape[-1]).max(0)
                                     - cps.reshape(-1, cps.shape[-1]).min(0)))
    closed_u = float(np.max(np.linalg.norm(cps[0] - cps[-1], axis=-1))) < 1e-9 * max(geo_scale, 1e-12)
    closed_v = float(np.max(np.linalg.norm(cps[:, 0] - cps[:, -1], axis=-1))) < 1e-9 * max(geo_scale, 1e-12)

    best = np.inf
    pts = []
    curves = []
    surfs = []
    for patch in patches:
        if remaining <= 0:
            agg_exhausted = True
            break
        (pu_lo, pu_hi), (pv_lo, pv_hi) = _surface_interval(patch)
        net = _patch_surface_net(patch)
        patch_stats = {}
        local = bez_surface_closest_points(net, point, atol=atol, rational=True,
                                           want_eval=want_eval, upper_bound=best,
                                           max_cells=remaining,
                                           stats=patch_stats)
        used = max(0, int(patch_stats.get("cells_processed", 0)))
        remaining -= reconcile_reported(used, remaining)[0]
        agg_cells += used
        agg_exhausted |= bool(patch_stats.get("budget_exhausted", False))

        def to_gu(u):
            return pu_lo + u * (pu_hi - pu_lo)

        def to_gv(v):
            return pv_lo + v * (pv_hi - pv_lo)

        for e in local:
            best = min(best, e["distance"])
            if e["kind"] == "degenerate_surface":
                surfs.append({"kind": "degenerate_surface",
                              "u_range": (to_gu(0.0), to_gu(1.0)),
                              "v_range": (to_gv(0.0), to_gv(1.0)),
                              "u": to_gu(e["u"]), "v": to_gv(e["v"]),
                              "point": e["point"], "distance": e["distance"]})
            elif e["kind"] == "degenerate_curve":
                uv = e["uv"].copy()
                uv[:, 0] = pu_lo + uv[:, 0] * (pu_hi - pu_lo)
                uv[:, 1] = pv_lo + uv[:, 1] * (pv_hi - pv_lo)
                curves.append({"kind": "degenerate_curve", "uv": uv,
                               "points": e["points"], "closed": e["closed"],
                               "u": to_gu(e["u"]), "v": to_gv(e["v"]),
                               "point": e["point"], "distance": e["distance"]})
            else:
                u_g, v_g = to_gu(e["u"]), to_gv(e["v"])
                kind = e["kind"]
                if kind == "boundary_min":
                    # A domain border only counts as a real boundary when the
                    # surface is OPEN in that direction (a revolution's u seam
                    # is a periodic seam, not an edge).
                    on_global = ((not closed_u and (abs(u_g - gu_lo) < 1e-9
                                                    or abs(u_g - gu_hi) < 1e-9))
                                 or (not closed_v and (abs(v_g - gv_lo) < 1e-9
                                                       or abs(v_g - gv_hi) < 1e-9)))
                    if not on_global:
                        kind = "min"
                _merge_surface_point(pts, u_g, v_g, e, kind, atol)

    if closed_u:
        pts = _merge_periodic_seam_2d(pts, "u", gu_lo, gu_hi, atol, geo_scale)
    if closed_v:
        pts = _merge_periodic_seam_2d(pts, "v", gv_lo, gv_hi, atol, geo_scale)

    # Chain/subsume in NORMALIZED parameters: real domains can be extremely
    # anisotropic (e.g. u in [0, 2pi], v in [0, 1.8e4]); raw uv distances would
    # be dominated by the larger axis and meaningless for the smaller one.
    u_ext = max(gu_hi - gu_lo, 1e-300)
    v_ext = max(gv_hi - gv_lo, 1e-300)

    # Stitch tolerance must swallow the trace-endpoint slop at patch seams
    # (a trace ends within ~step of the seam in local params).
    curves = _stitch_degenerate_curves(curves, band, stitch_tol=0.02,
                                       u_ext=u_ext, v_ext=v_ext)
    # Re-certify circles on the stitched result (a full ring assembled from
    # per-patch arcs is certified as one exact circle).
    for c in curves:
        cert = _certify_circle(c["points"], point)
        if cert is not None:
            c["circle"] = cert
        else:
            c.pop("circle", None)

    # Points lying on a degenerate curve (or inside a degenerate patch) are
    # subsumed by the larger entity.
    def subsumed(e):
        for c in curves:
            uv_n = c["uv"] / np.array([u_ext, v_ext])
            if (abs(e["distance"] - c["distance"]) <= band
                    and _uv_dist_to_polyline(e["u"] / u_ext, e["v"] / v_ext, uv_n) < 0.05):
                return True
        for s in surfs:
            if (s["u_range"][0] - 1e-9 <= e["u"] <= s["u_range"][1] + 1e-9
                    and s["v_range"][0] - 1e-9 <= e["v"] <= s["v_range"][1] + 1e-9
                    and abs(e["distance"] - s["distance"]) <= band):
                return True
        return False

    _publish_work_stats(stats, agg_cells, agg_exhausted)
    entities = [e for e in pts if not subsumed(e)] + curves + surfs
    if not entities:
        return []
    gmin = min(e["distance"] for e in entities)
    entities = [e for e in entities if e["distance"] <= gmin + band]
    entities.sort(key=lambda e: e["distance"])
    return entities


def _merge_surface_point(merged, u, v, src, kind, atol):
    pt = np.asarray(src["point"])
    dist = src["distance"]
    for e in merged:
        if ((abs(e["u"] - u) < 1e-7 and abs(e["v"] - v) < 1e-7)
                or np.linalg.norm(e["point"] - pt) < max(atol, 1e-9)):
            if dist < e["distance"]:
                e.update(u=float(u), v=float(v), point=pt, distance=float(dist), kind=kind)
                if "eval" in src:
                    e["eval"] = src["eval"]
            elif kind == "boundary_min" and e["kind"] != "boundary_min":
                e["kind"] = "boundary_min"
            return
    entry = {"u": float(u), "v": float(v), "point": pt, "distance": float(dist), "kind": kind}
    if "eval" in src:
        entry["eval"] = src["eval"]
    merged.append(entry)


def _merge_periodic_seam_2d(pts, axis, g_lo, g_hi, atol, geo_scale):
    """Merge point entities duplicated across a periodic parameter seam
    (same physical point polished from both sides of ``axis``'s domain ends).
    Keeps the nearer duplicate."""
    span = max(g_hi - g_lo, 1e-300)
    seam_t = 1e-3 * span
    seam_geo = max(atol, 1e-6 * max(geo_scale, 1.0))
    key = axis  # "u" or "v"
    out = []
    for e in sorted(pts, key=lambda x: x["distance"]):
        dup = False
        for kept in out:
            near_opposite_ends = (
                (e[key] - g_lo < seam_t and g_hi - kept[key] < seam_t)
                or (kept[key] - g_lo < seam_t and g_hi - e[key] < seam_t))
            if near_opposite_ends and np.linalg.norm(e["point"] - kept["point"]) < seam_geo:
                dup = True
                break
        if not dup:
            out.append(e)
    return out


def _stitch_degenerate_curves(curves, band, stitch_tol, u_ext=1.0, v_ext=1.0):
    """Chain traced curve entities whose endpoints meet (patch seams).

    Endpoint proximity is measured in NORMALIZED parameters (uv divided by
    the per-axis domain extents) so anisotropic domains do not skew the
    tolerance. A chain whose final endpoints also meet is marked ``closed``
    (e.g. the equidistant ring of a full surface of revolution).
    """
    if len(curves) <= 1:
        return curves
    scale = np.array([u_ext, v_ext], dtype=np.float64)

    def gap(p, q):
        return float(np.hypot(*((p - q) / scale)))

    pool = [dict(c) for c in curves]
    out = []
    while pool:
        cur = pool.pop(0)
        if cur["closed"]:
            out.append(cur)
            continue
        changed = True
        while changed:
            changed = False
            for k, other in enumerate(pool):
                if other["closed"] or abs(other["distance"] - cur["distance"]) > band:
                    continue
                a0, a1 = cur["uv"][0], cur["uv"][-1]
                b0, b1 = other["uv"][0], other["uv"][-1]
                join = None
                if gap(a1, b0) < stitch_tol:
                    join = (cur["uv"], other["uv"], cur["points"], other["points"])
                elif gap(a1, b1) < stitch_tol:
                    join = (cur["uv"], other["uv"][::-1], cur["points"], other["points"][::-1])
                elif gap(a0, b1) < stitch_tol:
                    join = (other["uv"], cur["uv"], other["points"], cur["points"])
                elif gap(a0, b0) < stitch_tol:
                    join = (other["uv"][::-1], cur["uv"], other["points"][::-1], cur["points"])
                if join is not None:
                    uv = np.vstack([join[0], join[1][1:]])
                    xyz = np.vstack([join[2], join[3][1:]])
                    cur = {"kind": "degenerate_curve", "uv": uv, "points": xyz,
                           "closed": False, "u": float(uv[len(uv) // 2, 0]),
                           "v": float(uv[len(uv) // 2, 1]),
                           "point": np.asarray(xyz[len(xyz) // 2]),
                           "distance": float((cur["distance"] + other["distance"]) / 2.0)}
                    pool.pop(k)
                    changed = True
                    break
        # Closure of the assembled chain: either the UV ends meet (a loop
        # inside the domain) or the 3-D endpoints meet across a periodic
        # parameter seam (there uv=0 and uv=1 are the same physical point but
        # maximally distant in parameter space). Traces legitimately stop up
        # to ~a step short of patch edges, so "meet" in 3-D means within a
        # few trace sample steps, measured from the polyline itself.
        if len(cur["uv"]) > 3:
            uv_meet = gap(cur["uv"][0], cur["uv"][-1]) < stitch_tol
            seg = np.linalg.norm(np.diff(cur["points"], axis=0), axis=1)
            step_3d = float(np.mean(seg)) if len(seg) else 0.0
            xyz_meet = (float(np.linalg.norm(cur["points"][0] - cur["points"][-1]))
                        < max(2.0 * step_3d, 1e-12))
            if uv_meet or xyz_meet:
                cur["closed"] = True
        out.append(cur)
    return out


__all__ = [
    "point_curve_stationarity_net",
    "point_surface_stationarity_nets",
    "eval_curve_d2",
    "eval_surface_d2",
    "newton_curve_closest_point",
    "newton_surface_closest_point",
    "trace_equidistant_curve",
    "bez_curve_closest_points",
    "bez_surface_closest_points",
    "nurbs_curve_closest_points",
    "nurbs_surface_closest_points",
]

# mmcore/numeric/_bez_closest_point.py
"""Closest-point on rational Bézier/NURBS curves and surfaces via
squared-distance Bernstein nets.

Replaces the unreliable divide-and-conquer code in ``closest_point.py``
(kept untouched for A/B comparison). See
``docs/superpowers/specs/2026-06-25-closest-point-sq-dist-nets-design.md``.
"""
from __future__ import annotations

from math import comb

import numpy as np


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


# mmcore/numeric/_bez_closest_point.py  (append)
from mmcore.numeric import bern_sq_dist
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


# mmcore/numeric/_bez_closest_point.py  (append)
from mmcore.numeric._bern_homog import (
    eval_bezier_curve_homog_with_derivs,
    eval_bezier_surface_homog_with_derivs,
    project_curve_homog_to_cartesian,
    project_surface_homog_to_cartesian,
)
from mmcore.numeric.intersection._bezier_common import (
    _to_homog_curve, _to_homog_surface, eval_curve, eval_curve_d1,
    eval_surface, eval_surface_d1, extract_weights, _clamp01,
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


# mmcore/numeric/_bez_closest_point.py  (append)

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

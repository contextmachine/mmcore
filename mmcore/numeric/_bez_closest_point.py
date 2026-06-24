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

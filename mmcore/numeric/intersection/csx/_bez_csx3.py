from __future__ import annotations

import itertools
import time
from typing import TypedDict, List
import logging

import numpy as np
from numpy.typing import NDArray

from mmcore.numeric import compute_parametric_curvature_tolerance_surface
from mmcore.numeric._aabb import aabb, aabb_intersect
from mmcore.numeric.bern import (
    bernstein_product_conv,
    bernstein_partial_derivative_coeffs,
    de_casteljau_split_nd,
    bernstein_cutout_box_nd,
    bernstein_trim_nd,
    bernstein_boundaries_2d,
)
from mmcore.numeric._bern_homog import (
    eval_bezier_curve_homog_with_derivs,
    project_curve_homog_to_cartesian,
    eval_bezier_surface_homog_with_derivs,
    project_surface_homog_to_cartesian, eval_bezier_homogeneous_surface,eval_bezier_homogeneous_curve
)
from mmcore.numeric.interval import Interval
from mmcore.numeric.numeric import  compute_parametric_tolerance_surface, \
    compute_parametric_curvature_tolerance_curve


logger = logging.getLogger("mmcore")

def clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x
def clamp(x: float, a,b) -> float:
    if x <= a:
        return a
    if x >= b:
        return b
    return x
# ---------------------------------------------------------------------------
# Homogeneous helpers
# ---------------------------------------------------------------------------

def is_homogeneous_ctrl(ctrl, rational: bool | None = None, eps: float = 1e-12) -> bool:
    """Detect or force whether a control net is homogeneous (rational).

    Parameters
    ----------
    ctrl : ndarray
        Control net.
    rational : bool or None, optional
        If ``True`` forces homogeneous interpretation; if ``False`` forces
        Cartesian; if ``None`` a heuristic is used (dimension/weights).
    eps : float, optional
        Tolerance when inspecting weights in heuristic mode.
    """
    if rational is not None:
        return bool(rational)

    if ctrl is None:
        return False

    dim = ctrl.shape[-1]
    if dim >= 4:
        return True  # (x,y,z,w)
    if dim == 3:
        w = np.asarray(ctrl[..., -1], dtype=float)
        if np.all(w > eps) and np.max(np.abs(w - 1.0)) > eps:
            return True
    return False


def dehomogenize_ctrl(ctrl, rational: bool | None = None):
    if not rational:
        return ctrl
    w = np.asarray(ctrl[..., -1:], dtype=float)
    return ctrl[..., :-1] / w


# ---------------------------------------------------------------------------
# Bézier evaluation helpers (curve & surface)
# ---------------------------------------------------------------------------

def _to_homog_curve(ctrl: NDArray, rational: bool | None = None) -> NDArray:
    if is_homogeneous_ctrl(ctrl, rational=rational):
        return np.asarray(ctrl, dtype=float)
    ctrl = np.asarray(ctrl, dtype=float)
    w = np.ones((ctrl.shape[0], 1), dtype=ctrl.dtype)
    return np.concatenate([ctrl, w], axis=1)

def _to_homog_surface(ctrl: NDArray, rational: bool | None = None) -> NDArray:
    if is_homogeneous_ctrl(ctrl, rational=rational):
        return np.asarray(ctrl, dtype=float)
    ctrl = np.asarray(ctrl, dtype=float)
    w = np.ones(ctrl.shape[:-1] + (1,), dtype=ctrl.dtype)
    return np.concatenate([ctrl, w], axis=-1)


def eval_bezier_curve(P, t, rational=None):
    Ph = _to_homog_curve(P, rational=rational)
    Ch = eval_bezier_homogeneous_curve(Ph, t)            # no derivs
    return Ch[:-1] / Ch[-1]

def eval_bezier_surface(S, u, v, rational=None):
    Sh = _to_homog_surface(S, rational=rational)
    Sh0 = eval_bezier_homogeneous_surface(Sh, u, v)      # no derivs
    return Sh0[:-1] / Sh0[-1]
def eval_bezier_curve_and_deriv(P: NDArray, t: float, rational: bool | None = None):
    Ph = _to_homog_curve(P, rational=rational)
    Ch, Chd = eval_bezier_curve_homog_with_derivs(Ph, t, want_second=False)
    C, Ct = project_curve_homog_to_cartesian(Ch, Chd)
    return C, Ct


def eval_bezier_curve_derivs2(P: NDArray, t: float, rational: bool | None = None):
    Ph = _to_homog_curve(P, rational=rational)
    Ch, Chd, Ch2 = eval_bezier_curve_homog_with_derivs(Ph, t, want_second=True)
    C, Ct, Ctt = project_curve_homog_to_cartesian(Ch, Chd, Ch2)
    return C, Ct, Ctt


def eval_bezier_surface_and_derivs(S: NDArray, u: float, v: float, rational: bool | None = None):
    Sh = _to_homog_surface(S, rational=rational)
    Sh0, Shu, Shv = eval_bezier_surface_homog_with_derivs(Sh, u, v, want_second=False)
    S0, Su, Sv = project_surface_homog_to_cartesian(Sh0, Shu, Shv)
    return S0, Su, Sv


def eval_bezier_surface_derivs2(S: NDArray, u: float, v: float, rational: bool | None = None):
    Sh = _to_homog_surface(S, rational=rational)
    Sh0, Shu, Shv, Shuu, Shuv, Shvv = eval_bezier_surface_homog_with_derivs(Sh, u, v, want_second=True)
    S0, Su, Sv, Suu, Suv, Svv = project_surface_homog_to_cartesian(Sh0, Shu, Shv, Shuu, Shuv, Shvv)
    return S0, Su, Sv, Suu, Suv, Svv


# ---------------------------------------------------------------------------
# Bernstein distance net for curve-surface
# ---------------------------------------------------------------------------

def bernstein_distance_squared_net_trivariate(E: np.ndarray) -> np.ndarray:
    """Squared norm net for a trivariate Bernstein vector net.

    Parameters
    ----------
    E : (p+1, q+1, r+1, d) ndarray
        Trivariate control net of a vector field.

    Returns
    -------
    F : (2p+1, 2q+1, 2r+1) ndarray
        Trivariate Bernstein control net of ``||E||^2``.
    """
    E = np.asarray(E, dtype=float)
    if E.ndim != 4:
        raise ValueError("E must have shape (p+1, q+1, r+1, d).")

    p = E.shape[0] - 1
    q = E.shape[1] - 1
    r = E.shape[2] - 1

    ConvU = bernstein_product_conv(p)
    ConvV = bernstein_product_conv(q)
    ConvW = bernstein_product_conv(r)

    ConvU2 = ConvU.reshape(2 * p + 1, (p + 1) * (p + 1))
    ConvV2 = ConvV.reshape(2 * q + 1, (q + 1) * (q + 1))
    ConvW2 = ConvW.reshape(2 * r + 1, (r + 1) * (r + 1))

    # Convolution along u
    Tu = np.zeros((2 * p + 1, q + 1, r + 1, q + 1, r + 1), dtype=float)
    for j in range(q + 1):
        for k in range(r + 1):
            Ejk = E[:, j, k, :]
            for jp in range(q + 1):
                for kp in range(r + 1):
                    Ejpkp = E[:, jp, kp, :]
                    A = Ejk @ Ejpkp.T
                    Tu[:, j, k, jp, kp] = ConvU2 @ A.ravel()

    # Convolution along v
    Tv = np.zeros((2 * p + 1, 2 * q + 1, r + 1, r + 1), dtype=float)
    for a in range(2 * p + 1):
        for k in range(r + 1):
            for kp in range(r + 1):
                B = Tu[a, :, k, :, kp]
                Tv[a, :, k, kp] = ConvV2 @ B.ravel()

    # Convolution along w
    F = np.zeros((2 * p + 1, 2 * q + 1, 2 * r + 1), dtype=float)
    for a in range(2 * p + 1):
        for b in range(2 * q + 1):
            C = Tv[a, b, :, :]
            F[a, b, :] = ConvW2 @ C.ravel()

    return F


def distance_squared_net_curve_surface(C: np.ndarray, S: np.ndarray, rational: bool | None = None) -> np.ndarray:
    """Trivariate Bernstein net of the squared distance between a curve and a surface."""
    if rational is None:
        rational = is_homogeneous_ctrl(C) or is_homogeneous_ctrl(S)
    C = np.asarray(C, dtype=float)
    S = np.asarray(S, dtype=float)

    if rational:
        if C.shape[-1] < 3 or S.shape[-1] < 3:
            raise ValueError("Rational inputs must include weights as the last coordinate.")
        Cw = C
        Sw = S
        wc = Cw[:, -1]
        ws = Sw[:, :, -1]
        Cxyz = Cw[:, :-1]
        Sxyz = Sw[:, :, :-1]
        D = Cxyz[:, None, None, :] * ws[None, :, :, None] - Sxyz[None, :, :, :] * wc[:, None, None, None]
    else:
        D = C[:, None, None, :] - S[None, :, :, :]

    return bernstein_distance_squared_net_trivariate(D)


# ---------------------------------------------------------------------------
# G(t,u,v) = C(t) - S(u,v) system and Newton solver
# ---------------------------------------------------------------------------

def G_and_J_curve_surface(C, S, t, u, v, rational: bool | None = None):
    c, ct = eval_bezier_curve_and_deriv(C, t, rational=rational)
    s, su, sv = eval_bezier_surface_and_derivs(S, u, v, rational=rational)
    G = c - s
    J = np.stack([ct, -su, -sv], axis=1)
    return G, J


def G_only_curve_surface(C, S, t, u, v, rational: bool | None = None):
    return eval_bezier_curve(C, t, rational=rational) - eval_bezier_surface(S, u, v, rational=rational)


def newton_project_G0_curve_surface(
    C,
    S,
    t0,
    u0,
    v0,
    tol=1e-12,
    it=15,
    lm_damp=1e-12,
    step_tol=1e-9,
    delta_tol=1e-10,
    rational: bool | None = None,
):
    """Levenberg–Marquardt corrector to G(t,u,v)=0; clamps to [0,1]^3."""
    t, u, v = float(t0), float(u0), float(v0)
    delta_tol_sq = delta_tol * delta_tol
    tol_sq = tol * tol
    prev_g2 = np.inf
    stall_count = 0
    for _ in range(it):
        try:
            G, J = G_and_J_curve_surface(C, S, t, u, v, rational=rational)
        except Exception:
            break
        JT = J.T
        A = JT @ J + lm_damp * np.eye(3)
        b = -JT @ G
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            delta = np.zeros(3)

        step = 1.0
        g2 = float(np.dot(G, G))
        if g2 > 0.9 * prev_g2:
            stall_count += 1
            if stall_count > 2:
                break  # Not converging, don't waste iterations
        else:
            stall_count = 0

        prev_g2 = g2
        for _ls in range(8):
            tn = float(clamp01(t + step * delta[0]))
            un = float(clamp01(u + step * delta[1]))
            vn = float(clamp01(v + step * delta[2]))
            dgj = G_only_curve_surface(C, S, tn, un, vn, rational=rational)
            if float(np.dot(dgj, dgj)) <= g2:
                t, u, v = tn, un, vn
                break
            step *= 0.5
        if g2 < tol_sq:
            break

        if step < step_tol and np.dot(delta, delta) < delta_tol_sq:
            break

    try:
        G, J = G_and_J_curve_surface(C, S, t, u, v, rational=rational)
    except Exception:
        G = np.array([np.inf, np.inf, np.inf])
        J = np.zeros((3, 3))
    return t, u, v, G, J


# ---------------------------------------------------------------------------
# Fixed-t projector for overlap tracing
# ---------------------------------------------------------------------------

def project_G0_fixed_t(C, S, t_fixed, u0, v0, tol=1e-12, it=30, lm_damp=1e-12, rational: bool | None = None):
    """Solve min ||C(t_fixed) - S(u,v)|| with (u,v) in [0,1]^2."""
    sq_tol = tol * tol
    p1 = eval_bezier_curve(C, t_fixed, rational=rational)
    u, v = float(u0), float(v0)

    for _ in range(it):
        s, su, sv = eval_bezier_surface_and_derivs(S, u, v, rational=rational)
        G = s - p1
        J = np.stack([su, sv], axis=1)
        JTJ = J.T @ J + lm_damp * np.eye(2)
        JTG = J.T @ G
        try:
            delta = -np.linalg.solve(JTJ, JTG)
        except np.linalg.LinAlgError:
            delta = np.zeros(2)

        step = 1.0
        g0 = float(np.dot(G, G))
        while step > 1e-6:
            un = float(clamp01(u + step * delta[0]))
            vn = float(clamp01(v + step * delta[1]))
            dgj = eval_bezier_surface(S, un, vn, rational=rational) - p1
            if float(np.dot(dgj, dgj)) <= g0 + 1e-18:
                u, v = un, vn
                break
            step *= 0.5

        dgj = eval_bezier_surface(S, u, v, rational=rational) - p1
        if float(np.dot(dgj, dgj)) < sq_tol:
            return u, v, dgj, True

    dgj = eval_bezier_surface(S, u, v, rational=rational) - p1
    return u, v, dgj, (np.linalg.norm(dgj) < 5.0 * tol)
import numpy as np


def _eval_bezier_curve_and_d1(C, t, *, rational: bool | None = None, fd_eps: float = 1e-6):
    """
    Returns (c, ct) where:
      c  = C(t)
      ct = dC/dt at t

    Uses an analytic helper if your module provides one; otherwise falls back
    to a finite-difference derivative (works for both polynomial and rational
    as long as eval_bezier_curve works).
    """
    # Prefer analytic derivative helpers if they exist in the current module.
    fn = globals().get("eval_bezier_curve_and_derivs", None)
    if callable(fn):
        out = fn(C, t, rational=rational)
        # Common conventions: (c, ct) or (c, ct, ctt, ...)
        c = out[0]
        ct = out[1]
        return c, ct

    fn = globals().get("eval_bezier_curve_and_deriv", None)
    if callable(fn):
        c, ct = fn(C, t, rational=rational)
        return c, ct

    # Finite-difference fallback
    t = float(clamp01(t))
    h = float(fd_eps)

    c = eval_bezier_curve(C, t, rational=rational)

    # Forward/backward/central difference depending on proximity to boundary
    if t <= h:
        t1 = float(min(1.0, t + h))
        c1 = eval_bezier_curve(C, t1, rational=rational)
        dt = t1 - t
        ct = (c1 - c) / dt if dt > 0.0 else np.zeros_like(c)
        return c, ct

    if t >= 1.0 - h:
        t0 = float(max(0.0, t - h))
        c0 = eval_bezier_curve(C, t0, rational=rational)
        dt = t - t0
        ct = (c - c0) / dt if dt > 0.0 else np.zeros_like(c)
        return c, ct

    t0 = float(t - h)
    t1 = float(t + h)
    c0 = eval_bezier_curve(C, t0, rational=rational)
    c1 = eval_bezier_curve(C, t1, rational=rational)
    ct = (c1 - c0) / (t1 - t0)
    return c, ct


def project_G0_fixed_u(
    C, S, u_fixed, t0, v0,
    tol=1e-12, it=30, lm_damp=1e-12, rational: bool | None = None
):
    """Solve min ||C(t) - S(u_fixed, v)|| with (t,v) in [0,1]^2."""
    sq_tol = tol * tol
    u_fixed = float(clamp01(u_fixed))
    t, v = float(t0), float(v0)

    for _ in range(it):
        c, ct = _eval_bezier_curve_and_d1(C, t, rational=rational)
        s, su, sv = eval_bezier_surface_and_derivs(S, u_fixed, v, rational=rational)

        # Residual: surface - curve
        G = s - c

        # Jacobian columns: d/dt (s - c) = -ct, d/dv (s - c) = sv
        J = np.stack([-ct, sv], axis=1)

        JTJ = J.T @ J + lm_damp * np.eye(2)
        JTG = J.T @ G
        try:
            delta = -np.linalg.solve(JTJ, JTG)
        except np.linalg.LinAlgError:
            delta = np.zeros(2)

        step = 1.0
        g0 = float(np.dot(G, G))
        while step > 1e-6:
            tn = float(clamp01(t + step * delta[0]))
            vn = float(clamp01(v + step * delta[1]))

            cn = eval_bezier_curve(C, tn, rational=rational)
            sn = eval_bezier_surface(S, u_fixed, vn, rational=rational)
            dgj = sn - cn

            if float(np.dot(dgj, dgj)) <= g0 + 1e-18:
                t, v = tn, vn
                break
            step *= 0.5

        cn = eval_bezier_curve(C, t, rational=rational)
        sn = eval_bezier_surface(S, u_fixed, v, rational=rational)
        dgj = sn - cn
        if float(np.dot(dgj, dgj)) < sq_tol:
            return t, v, dgj, True

    cn = eval_bezier_curve(C, t, rational=rational)
    sn = eval_bezier_surface(S, u_fixed, v, rational=rational)
    dgj = sn - cn
    return t, v, dgj, (np.linalg.norm(dgj) < 5.0 * tol)


def project_G0_fixed_v(
    C, S, v_fixed, t0, u0,
    tol=1e-12, it=30, lm_damp=1e-12, rational: bool | None = None
):
    """Solve min ||C(t) - S(u, v_fixed)|| with (t,u) in [0,1]^2."""
    sq_tol = tol * tol
    v_fixed = float(clamp01(v_fixed))
    t, u = float(t0), float(u0)

    for _ in range(it):
        c, ct = _eval_bezier_curve_and_d1(C, t, rational=rational)
        s, su, sv = eval_bezier_surface_and_derivs(S, u, v_fixed, rational=rational)

        # Residual: surface - curve
        G = s - c

        # Jacobian columns: d/dt (s - c) = -ct, d/du (s - c) = su
        J = np.stack([-ct, su], axis=1)

        JTJ = J.T @ J + lm_damp * np.eye(2)
        JTG = J.T @ G
        try:
            delta = -np.linalg.solve(JTJ, JTG)
        except np.linalg.LinAlgError:
            delta = np.zeros(2)

        step = 1.0
        g0 = float(np.dot(G, G))
        while step > 1e-6:
            tn = float(clamp01(t + step * delta[0]))
            un = float(clamp01(u + step * delta[1]))

            cn = eval_bezier_curve(C, tn, rational=rational)
            sn = eval_bezier_surface(S, un, v_fixed, rational=rational)
            dgj = sn - cn

            if float(np.dot(dgj, dgj)) <= g0 + 1e-18:
                t, u = tn, un
                break
            step *= 0.5

        cn = eval_bezier_curve(C, t, rational=rational)
        sn = eval_bezier_surface(S, u, v_fixed, rational=rational)
        dgj = sn - cn
        if float(np.dot(dgj, dgj)) < sq_tol:
            return t, u, dgj, True

    cn = eval_bezier_curve(C, t, rational=rational)
    sn = eval_bezier_surface(S, u, v_fixed, rational=rational)
    dgj = sn - cn
    return t, u, dgj, (np.linalg.norm(dgj) < 5.0 * tol)

import numpy as np


def project_G0(
    C,
    S,
    t0,
    u0,
    v0,
    fixed: tuple[bool, bool, bool],
    tol=1e-12,
    it=30,
    lm_damp=1e-12,
    rational: bool | None = None,
    fd_eps: float = 1e-6,
):
    """
    Universal projection:
      Solve min || C(t) - S(u,v) || with t,u,v in [0,1],
      where any one or two variables can be fixed.

    Parameters
    ----------
    C, S : control data for curve/surface (whatever your eval_* expect)
    t0, u0, v0 : initial guesses; also used as fixed values when fixed flags are True
    fixed : (fix_t, fix_u, fix_v)
        - True  => that variable is held constant at its (clipped) initial value
        - False => that variable is optimized
        Requirement: at least one True and at least one False (i.e., fix 1 or 2 vars)
    tol, it, lm_damp : same meaning as your existing functions
    rational : forwarded to eval_* functions
    fd_eps : finite-difference step for curve derivative if no analytic helper exists

    Returns
    -------
    t, u, v : floats in [0,1]
    dgj : residual vector = S(u,v) - C(t)
    success : bool
    """
    #print(fixed)
    if not (isinstance(fixed, tuple) and len(fixed) == 3):
        raise TypeError("fixed must be a tuple[bool, bool, bool] of length 3: (fix_t, fix_u, fix_v).")

    fix_t, fix_u, fix_v = (bool(fixed[0]), bool(fixed[1]), bool(fixed[2]))
    def snap(v):
        return 0. if v < (1-v) else 1.
    # Must fix at least one and free at least one
    if (fix_t and fix_u and fix_v) or ((not fix_t) and (not fix_u) and (not fix_v)):
        return snap(t0),snap(u0),snap(v0),np.array([0.,0.,0]),0.
        #raise ValueError(f"fixed must contain at least one True and at least one False (fix 1 or 2 variables). ({t0}, {u0}, {v0}),({fix_t}, {fix_u}, {fix_v}).")

    def _curve_and_d1(t):
        """Return (c, ct) at t; ct via analytic helper if available, else finite-difference."""
        fn = globals().get("eval_bezier_curve_and_derivs", None)
        if callable(fn):
            out = fn(C, t, rational=rational)
            return out[0], out[1]

        fn = globals().get("eval_bezier_curve_and_deriv", None)
        if callable(fn):
            return fn(C, t, rational=rational)

        t = float(clamp01(t))
        h = float(fd_eps)

        c = eval_bezier_curve(C, t, rational=rational)

        if t <= h:
            t1 = float(min(1.0, t + h))
            c1 = eval_bezier_curve(C, t1, rational=rational)
            dt = t1 - t
            ct = (c1 - c) / dt if dt > 0.0 else np.zeros_like(c)
            return c, ct

        if t >= 1.0 - h:
            t0_ = float(max(0.0, t - h))
            c0_ = eval_bezier_curve(C, t0_, rational=rational)
            dt = t - t0_
            ct = (c - c0_) / dt if dt > 0.0 else np.zeros_like(c)
            return c, ct

        t0_ = float(t - h)
        t1_ = float(t + h)
        c0_ = eval_bezier_curve(C, t0_, rational=rational)
        c1_ = eval_bezier_curve(C, t1_, rational=rational)
        ct = (c1_ - c0_) / (t1_ - t0_)
        return c, ct

    sq_tol = tol * tol

    # Initialize and clamp
    t = float(clamp01(t0))
    u = float(clamp01(u0))
    v = float(clamp01(v0))

    # Cache fully-fixed side(s) for speed
    c_fixed = None
    if fix_t:
        c_fixed = eval_bezier_curve(C, t, rational=rational)

    s_fixed = None
    if fix_u and fix_v:
        s_fixed = eval_bezier_surface(S, u, v, rational=rational)

    for _ in range(it):
        # Evaluate curve (and tangent if needed)
        if fix_t:
            c = c_fixed
            ct = None
        else:
            c, ct = _curve_and_d1(t)

        # Evaluate surface (and partials if needed)
        if fix_u and fix_v:
            s = s_fixed
            su = None
            sv = None
        else:
            s, su, sv = eval_bezier_surface_and_derivs(S, u, v, rational=rational)

        # Residual: surface - curve
        G = s - c
        g0 = float(np.dot(G, G))
        if g0 < sq_tol:
            return t, u, v, G, True

        # Build Jacobian for free variables (order: t, u, v)
        cols = []
        if not fix_t:
            cols.append(-ct)   # d/dt (s - c) = -c_t
        if not fix_u:
            cols.append(su)    # d/du (s - c) = s_u
        if not fix_v:
            cols.append(sv)    # d/dv (s - c) = s_v

        if not cols:
            # Should be impossible due to validation above
            break

        J = np.stack(cols, axis=1)  # shape (dim, n_free)
        JTJ = J.T @ J + lm_damp * np.eye(J.shape[1])
        JTG = J.T @ G

        try:
            delta = -np.linalg.solve(JTJ, JTG)
        except np.linalg.LinAlgError:
            delta = np.zeros(J.shape[1], dtype=float)

        # Backtracking line search with clamping
        step = 1.0
        while step > 1e-6:
            tn, un, vn = t, u, v
            k = 0
            if not fix_t:
                tn = float(clamp01(t + step * float(delta[k])))
                k += 1
            if not fix_u:
                un = float(clamp01(u + step * float(delta[k])))
                k += 1
            if not fix_v:
                vn = float(clamp01(v + step * float(delta[k])))
                k += 1

            # Candidate residual (use caches if applicable)
            cn = c_fixed if fix_t else eval_bezier_curve(C, tn, rational=rational)
            sn = s_fixed if (fix_u and fix_v) else eval_bezier_surface(S, un, vn, rational=rational)
            dgj = sn - cn

            if float(np.dot(dgj, dgj)) <= g0 + 1e-18:
                t, u, v = tn, un, vn
                break

            step *= 0.5

        # Check convergence after accepted update (or tiny step)
        cn = c_fixed if fix_t else eval_bezier_curve(C, t, rational=rational)
        sn = s_fixed if (fix_u and fix_v) else eval_bezier_surface(S, u, v, rational=rational)
        dgj = sn - cn
        if float(np.dot(dgj, dgj)) < sq_tol:
            return t, u, v, dgj, True

    cn = c_fixed if fix_t else eval_bezier_curve(C, t, rational=rational)
    sn = s_fixed if (fix_u and fix_v) else eval_bezier_surface(S, u, v, rational=rational)
    dgj = sn - cn
    return t, u, v, dgj, (np.linalg.norm(dgj) < 5.0 * tol)

# ---------------------------------------------------------------------------
# Classification via Jacobian rank
# ---------------------------------------------------------------------------

def overlap_like_svd(J, sv_thresh=1e-8, rel_thresh=1e-6):
    try:
        s = np.linalg.svd(J, compute_uv=False)
    except np.linalg.LinAlgError:
        return False
    if s.shape[0] < 3:
        return False
    s_sorted = np.sort(s)[::-1]
    s_max, s_mid, s_min = s_sorted[0], s_sorted[1], s_sorted[-1]
    if s_max <= 0.0:
        return False
    # Guard against near-rank-1 degeneracy
    if s_mid <= max(1e-12 * s_max, sv_thresh * 1e-2):
        return False
    rel = s_min / s_max
    return (s_min < sv_thresh) or (rel < rel_thresh)


def classify_contact_curve_surface(J, sv_thresh=1e-8, rel_thresh: float | None = None):
    s = np.linalg.svd(J, compute_uv=False)
    s_sorted = np.sort(s)[::-1]
    if s_sorted.shape[0] < 3:
        return {"type": "ambiguous", "svals": s_sorted}
    s_max, s_mid, s_min = s_sorted[0], s_sorted[1], s_sorted[-1]
    mid_abs_ok = s_mid > 1e-10
    mid_rel_ok = s_mid > max(1e-12 * s_max, sv_thresh * 1e-2)
    overlap = (s_min < sv_thresh and (mid_abs_ok or mid_rel_ok))
    if (not overlap) and (rel_thresh is not None):
        if s_max > 0.0 and mid_rel_ok:
            if (s_min / s_max) < rel_thresh:
                overlap = True
    if overlap:
        return {"type": "overlap", "svals": (s_max, s_mid, s_min)}
    if s_min >= sv_thresh and (rel_thresh is None or (s_min / s_max) >= rel_thresh):
        return {"type": "isolated", "svals": (s_max, s_mid, s_min)}
    return {"type": "ambiguous", "svals": (s_max, s_mid, s_min)}


# ---------------------------------------------------------------------------
# Overlap span confirmation (analytic-segment criterion)
# ---------------------------------------------------------------------------

def confirm_overlap_span(
    C,
    S,
    tol_proj,
    tol_conf,
    angle_tol,
    sv_thresh,
    rational: bool | None = None,
    rel_thresh: float = 1e-6,
):
    """Confirm full-span overlap on a Bézier curve segment / surface patch.

    Returns dict with endpoints if confirmed, otherwise None.
    """
    # Use a few interior samples to confirm overlap across the span.
    t_samples = (0.2, 0.5, 0.8)
    u_seed, v_seed = 0.5, 0.5

    def check_tangent(tt, uu, vv):
        c_pt, c_tan = eval_bezier_curve_and_deriv(C, tt, rational=rational)
        s_pt, su, sv = eval_bezier_surface_and_derivs(S, uu, vv, rational=rational)
        n = np.cross(su, sv)
        n_norm = np.linalg.norm(n)
        t_norm = np.linalg.norm(c_tan)
        if n_norm < 1e-12 or t_norm < 1e-12:
            return False
        return abs(np.dot(c_tan / t_norm, n / n_norm)) <= angle_tol

    def _project_fixed_t(tt, uu, vv):
        t_fix, uu, vv, Gp, _ok = project_G0(
            C,
            S,
            tt,
            uu,
            vv,
            fixed=(True, False, False),
            tol=tol_proj,
            rational=rational,
        )
        return t_fix, uu, vv, Gp

    def _check_overlap_at_t(tt, uu, vv):
        t_fix, uu, vv, Gp = _project_fixed_t(tt, uu, vv)
        if np.linalg.norm(Gp) > tol_conf:
            return False, uu, vv, t_fix
        Gc, Jc = G_and_J_curve_surface(C, S, t_fix, uu, vv, rational=rational)
        # Guard against parametric degeneracy: if surface normal is near zero,
        # J becomes rank-deficient for reasons unrelated to overlap.
        s_pt, su, sv = eval_bezier_surface_and_derivs(S, uu, vv, rational=rational)
        n = np.cross(su, sv)
        if np.linalg.norm(n) < 1e-12:
            return False, uu, vv, t_fix
        if (not overlap_like_svd(Jc, sv_thresh=sv_thresh, rel_thresh=rel_thresh)) and (
            not check_tangent(t_fix, uu, vv)
        ):
            return False, uu, vv, t_fix
        return True, uu, vv, t_fix

    def _check_interval(t_a, t_b, uu, vv):
        if not np.isfinite(t_a) or not np.isfinite(t_b):
            return False, uu, vv
        if abs(t_b - t_a) < 1e-14:
            return False, uu, vv
        for alpha in t_samples:
            tt = float(clamp01(t_a + alpha * (t_b - t_a)))
            ok, uu, vv, _ = _check_overlap_at_t(tt, uu, vv)
            if not ok:
                return False, uu, vv
        return True, uu, vv

    # First, check the ends of the segment.
    t0, u0, v0, G0 = _project_fixed_t(0.0, u_seed, v_seed)
    on0 = np.linalg.norm(G0) <= tol_conf
    t1, u1, v1, G1 = _project_fixed_t(1.0, u0, v0)
    on1 = np.linalg.norm(G1) <= tol_conf

    # If both ends lie on the surface, confirm overlap across the full span.
    if on0 and on1:
        ok, u_seed, v_seed = _check_interval(0.0, 1.0, u0, v0)
        if not ok:
            return None
        x0 = eval_bezier_curve(C, 0.0, rational=rational)
        x1 = eval_bezier_curve(C, 1.0, rational=rational)
        return {
            "t_path": np.asarray([0.0, 1.0], dtype=float),
            "uv_path": np.asarray([(u0, v0), (u1, v1)], dtype=float),
            "xyz_path": np.asarray([x0, x1], dtype=float),
            "start": "boundary",
            "end": "boundary",
        }

    # If at least one end does not lie on the surface, use boundary intersection.
    iso_bnd, ovl_bnd = curve_surface_boundary_intersect(
        C, S, sv_thresh=sv_thresh, atol=tol_conf, rational=rational
    )
    # Deduplicate boundary points by t (boundary CCX can return identical endpoints).
    if iso_bnd:
        t_eps = max(1e-9, 1e-6 * tol_conf)
        iso_bnd = sorted(iso_bnd, key=lambda it: it["t"])
        iso_unique = [iso_bnd[0]]
        for it in iso_bnd[1:]:
            if abs(float(it["t"]) - float(iso_unique[-1]["t"])) > t_eps:
                iso_unique.append(it)
        iso_bnd = iso_unique
    if ovl_bnd:
        if len(ovl_bnd) > 1:
            logger.error(
                "Boundary overlap returned multiple segments (len=%d) for span confirm.", len(ovl_bnd)
            )
        ovl = dict(ovl_bnd[0])
        ovl["partial"] = True
        return ovl
    if not iso_bnd:
        # No boundary evidence of overlap for this span.
        return None

    if on0 ^ on1:
        if len(iso_bnd) == 2:
            t_e = 0.0 if on0 else 1.0
            # Drop a boundary point that coincides with the on-surface endpoint.
            t_eps = max(1e-9, 1e-6 * tol_conf)
            iso_bnd = [it for it in iso_bnd if abs(float(it["t"]) - t_e) > t_eps]
        if len(iso_bnd) == 1:
            bnd = iso_bnd[0]
            t_e = 0.0 if on0 else 1.0
            u_e, v_e = (u0, v0) if on0 else (u1, v1)
            t_b = float(bnd["t"])
            t_start, t_end = (t_e, t_b) if t_e <= t_b else (t_b, t_e)
            ok, u_seed, v_seed = _check_interval(t_start, t_end, u_e, v_e)
            if not ok:
                return None
            # Endpoints from surface or boundary intersection
            if t_start == t_e:
                uv_start = (u_e, v_e)
                uv_end = (float(bnd["u"]), float(bnd["v"]))
            else:
                uv_start = (float(bnd["u"]), float(bnd["v"]))
                uv_end = (u_e, v_e)
            x_start = eval_bezier_curve(C, t_start, rational=rational)
            x_end = eval_bezier_curve(C, t_end, rational=rational)
            return {
                "t_path": np.asarray([t_start, t_end], dtype=float),
                "uv_path": np.asarray([uv_start, uv_end], dtype=float),
                "xyz_path": np.asarray([x_start, x_end], dtype=float),
                "start": "boundary",
                "end": "boundary",
                "partial": True,
            }
        logger.error(
            "Span confirm: one endpoint on surface, boundary intersects=%d (expected 1).",
            len(iso_bnd),
        )
        raise RuntimeError("curve_surface_boundary_intersect returned unexpected count for span confirm")

    if not on0 and not on1:
        if len(iso_bnd) == 2:
            bnd_sorted = sorted(iso_bnd, key=lambda it: it["t"])
            t_start = float(bnd_sorted[0]["t"])
            t_end = float(bnd_sorted[1]["t"])
            ok, u_seed, v_seed = _check_interval(
                t_start, t_end, float(bnd_sorted[0]["u"]), float(bnd_sorted[0]["v"])
            )
            if not ok:
                return None
            uv_start = (float(bnd_sorted[0]["u"]), float(bnd_sorted[0]["v"]))
            uv_end = (float(bnd_sorted[1]["u"]), float(bnd_sorted[1]["v"]))
            x_start = eval_bezier_curve(C, t_start, rational=rational)
            x_end = eval_bezier_curve(C, t_end, rational=rational)
            return {
                "t_path": np.asarray([t_start, t_end], dtype=float),
                "uv_path": np.asarray([uv_start, uv_end], dtype=float),
                "xyz_path": np.asarray([x_start, x_end], dtype=float),
                "start": "boundary",
                "end": "boundary",
                "partial": True,
            }
        logger.error(
            "Span confirm: no endpoints on surface, boundary intersects=%d (expected 2).", len(iso_bnd)
        )
        raise RuntimeError("curve_surface_boundary_intersect returned unexpected count for span confirm")

    logger.error(
        "Span confirm: unexpected boundary state (on0=%s, on1=%s, iso=%d, ovl=%d).",
        on0,
        on1,
        len(iso_bnd),
        len(ovl_bnd),
    )
    raise RuntimeError("Unexpected span-confirm boundary state")

# ---------------------------------------------------------------------------
# Overlap tracing along curve parameter
# ---------------------------------------------------------------------------

def point_line_distance(p, a, b):
    ab = b - a
    denom = np.linalg.norm(ab)
    if denom < 1e-16:
        return np.linalg.norm(p - a)
    return np.linalg.norm(np.cross(ab, p - a)) / denom


def append_with_decimation(t_path, uv_path, xyz_path, t_new, uv_new, x_new, angle_tol, sag_tol):
    if len(xyz_path) < 2:
        t_path.append(t_new)
        uv_path.append(uv_new)
        xyz_path.append(x_new)
        return
    a, b = xyz_path[-2], xyz_path[-1]
    c = x_new
    ab = b - a
    bc = c - b
    lab = np.linalg.norm(ab)
    lbc = np.linalg.norm(bc)
    ang = 0.0
    if lab > 0 and lbc > 0:
        cosang = np.dot(ab, bc) / (lab * lbc)
        cosang = clamp(cosang, -1.0, 1.0)
        ang = np.arccos(cosang)
    sag = point_line_distance(b, a, c)
    if ang < angle_tol and sag < sag_tol:
        t_path[-1] = t_new
        uv_path[-1] = uv_new
        xyz_path[-1] = x_new
    else:
        t_path.append(t_new)
        uv_path.append(uv_new)
        xyz_path.append(x_new)
from mmcore.geom._nurbs_param_tol import _nurbs_curve_param_tol_conservative
from mmcore.geom._nurbs_knots import generate_knots

def trace_event(tuv):
    return np.argmin(np.minimum(np.abs(tuv),np.abs(1 - np.abs(tuv))))

from mmcore.geom.nurbs import CurveSurfaceEq
def trace_curve_surface_overlap(
    C,
    S,
    t_seed,
    u_seed,
    v_seed,
    sag_tol=None,
    dt_min=None,
    dt_max=None,
    angle_tol=1e-3,
    tol_proj=None,
    snap_eps=1e-6,
    step_growth=4,
    step_shrink=0.5,
    max_points=2000,
    rational: bool | None = None,
):
    """Trace a curve-on-surface overlap by stepping along the curve parameter t."""
    # scale & tolerances

    def bbox_diag_len(Cc, Ss):
        ec = dehomogenize_ctrl(Cc, rational=rational)
        es = dehomogenize_ctrl(Ss, rational=rational).reshape(-1, ec.shape[-1])
        mins = np.minimum(np.min(ec, axis=0), np.min(es, axis=0))
        maxs = np.maximum(np.max(ec, axis=0), np.max(es, axis=0))
        return float(np.linalg.norm(maxs - mins))

    scale = bbox_diag_len(C, S)
    if sag_tol is None:
        sag_tol = max(1e-7, 1e-5 * scale)
    if dt_min is None:
        dt_min = 1e-6
    if dt_max is None:
        dt_max = 0.25
    if tol_proj is None:
        tol_proj = max(1e-12, 1e-10 * scale)

    t0, u0, v0, G0, J0 = newton_project_G0_curve_surface(
        C, S, t_seed, u_seed, v_seed, tol=tol_proj, it=30, rational=rational
    )
    max_init = max(1e-6, 1e-5 * scale)
    if np.linalg.norm(G0) > max_init:
        return {"kind": "none"}

    # initial step size from curvature
    _, c1, c2 = eval_bezier_curve_derivs2(C, t0, rational=rational)
    dt = compute_parametric_curvature_tolerance_curve(c1, c2, spt=sag_tol)
    if rational:
        dt_n = _nurbs_curve_param_tol_conservative(
            C[..., :-1],
            C[..., -1],
            U=generate_knots(C.shape[0], C.shape[0] - 1),
            p=C.shape[0] - 1,
            tol=sag_tol,
        )
        if np.isfinite(dt_n):
            dt = dt_n if not np.isfinite(dt) else min(dt, dt_n)
    if not np.isfinite(dt):
        dt = dt_max
    dt = float(np.clip(dt, dt_min, dt_max))

    t_path = [t0]
    uv_path = [(u0, v0)]
    xyz_path = [eval_bezier_curve(C, t0, rational=rational)]

    def check_tangent(tt, uu, vv):
        c_pt, c_tan = eval_bezier_curve_and_deriv(C, tt, rational=rational)
        s_pt, su, sv = eval_bezier_surface_and_derivs(S, uu, vv, rational=rational)
        n = np.cross(su, sv)
        n_norm = np.linalg.norm(n)
        t_norm = np.linalg.norm(c_tan)
        if n_norm < 1e-12 or t_norm < 1e-12:
            return False
        return abs(np.dot(c_tan / t_norm, n / n_norm)) <= angle_tol

    def _nullspace_dir(tt, uu, vv):
        G, J = G_and_J_curve_surface(C, S, tt, uu, vv, rational=rational)
        try:
            _U, _S, Vt = np.linalg.svd(J)
        except np.linalg.LinAlgError:
            return None, G, J
        n = Vt[-1]
        n_norm = np.linalg.norm(n)
        if n_norm < 1e-14:
            return None, G, J
        return n / n_norm, G, J

    def _step_targets(tt, uu, vv):
        # Target param steps from curvature/tolerance
        _, C1, C2 = eval_bezier_curve_derivs2(C, tt, rational=rational)
        dt_target = compute_parametric_curvature_tolerance_curve(C1, C2, spt=sag_tol)
        if not np.isfinite(dt_target):
            dt_target = dt_max
        dt_target = float(np.clip(dt_target, dt_min, dt_max))

        try:
            _S0, Su, Sv, Suu, Suv, Svv = eval_bezier_surface_derivs2(S, uu, vv, rational=rational)
            du_target, dv_target = compute_parametric_tolerance_surface(Su, Sv, Suu, Suv, Svv, spt=sag_tol)
        except Exception:
            du_target, dv_target = dt_max, dt_max

        du_target = dt_max if not np.isfinite(du_target) else float(min(du_target, dt_max))
        dv_target = dt_max if not np.isfinite(dv_target) else float(min(dv_target, dt_max))
        return dt_target, du_target, dv_target

    def march_nullspace(direction):
        nonlocal dt
        t = t_path[0] if direction < 0 else t_path[-1]
        u = uv_path[0][0] if direction < 0 else uv_path[-1][0]
        v = uv_path[0][1] if direction < 0 else uv_path[-1][1]
        pts = 0
        ds = None
        last_ok = True
        n_prev = None

        while pts < max_points:
            n, Gc, Jc = _nullspace_dir(t, u, v)
            if n is None:
                break
            # Stay on the overlap branch even if local classification is noisy.

            # Keep direction consistent; make n[0] non-negative so direction controls sign.
            if n_prev is not None and np.dot(n, n_prev) < 0.0:
                n = -n
            if abs(n[0]) > 1e-12 and n[0] < 0.0:
                n = -n
            n_prev = n

            dt_target, du_target, dv_target = _step_targets(t, u, v)

            ds_candidates = []
            if abs(n[0]) > 1e-12:
                ds_candidates.append(dt_target / abs(n[0]))
            if abs(n[1]) > 1e-12:
                ds_candidates.append(du_target / abs(n[1]))
            if abs(n[2]) > 1e-12:
                ds_candidates.append(dv_target / abs(n[2]))
            ds_target = min(ds_candidates) if ds_candidates else dt_max
            if not np.isfinite(ds_target):
                ds_target = dt_max

            if ds is None:
                ds = ds_target
            elif last_ok:
                ds = min(ds_target, ds * step_growth)
            else:
                ds = min(ds_target, ds)

            # Minimum step derived from dt_min
            ds_min = np.inf
            if abs(n[0]) > 1e-12:
                ds_min = min(ds_min, dt_min / abs(n[0]))
            if abs(n[1]) > 1e-12:
                ds_min = min(ds_min, dt_min / abs(n[1]))
            if abs(n[2]) > 1e-12:
                ds_min = min(ds_min, dt_min / abs(n[2]))
            if not np.isfinite(ds_min):
                ds_min = dt_min
            ds = float(max(ds, ds_min))

            t_pred = t + direction * ds * n[0]
            u_pred = u + direction * ds * n[1]
            v_pred = v + direction * ds * n[2]

            t_cl = float(clamp01(t_pred))
            u_cl = float(clamp01(u_pred))
            v_cl = float(clamp01(v_pred))

            if t_cl == t and u_cl == u and v_cl == v:
                break

            t_corr, u_corr, v_corr, Gc, Jc = newton_project_G0_curve_surface(
                C, S, t_cl, u_cl, v_cl, tol=tol_proj, rational=rational
            )
            tol_accept = max(5.0 * tol_proj, 1e-8 * scale)
            ok = np.linalg.norm(Gc) <= tol_accept
            if ok:
                ok = overlap_like_svd(Jc, sv_thresh=1e-8, rel_thresh=1e-6) or check_tangent(t_corr, u_corr, v_corr)

            if not ok:
                ds = ds * step_shrink
                last_ok = False
                if ds <= ds_min * 1.01:
                    break
                continue

            prog = max(abs(t_corr - t), abs(u_corr - u), abs(v_corr - v))
            min_prog = max(1e-10, 0.1 * dt_min)
            if prog < min_prog:
                break

            x = eval_bezier_curve(C, t_corr, rational=rational)
            if direction > 0:
                t_path.append(t_corr)
                uv_path.append((u_corr, v_corr))
                xyz_path.append(x)
            else:
                t_path.insert(0, t_corr)
                uv_path.insert(0, (u_corr, v_corr))
                xyz_path.insert(0, x)

            t, u, v = t_corr, u_corr, v_corr
            last_ok = True
            pts += 1

        at_bnd = t <= 0.0 or t >= 1.0
        return "boundary" if at_bnd else "tangent_or_min_step"

    def march(direction):
        nonlocal dt
        t = t_path[0] if direction < 0 else t_path[-1]
        u = uv_path[0][0] if direction < 0 else uv_path[-1][0]
        v = uv_path[0][1] if direction < 0 else uv_path[-1][1]
        pts = 0
        tmax=0.25
        LAST_OK = True
        while pts < max_points:
            t_prev, u_prev, v_prev = t, u, v
            if LAST_OK:
                _, C1, C2 = eval_bezier_curve_derivs2(C, t_prev, rational=rational)
                dt_target = compute_parametric_curvature_tolerance_curve(C1, C2, spt=sag_tol)
                if not np.isfinite(dt_target):
                    dt_target = dt_max
                dt_target = float(np.clip(dt_target, dt_min, dt_max))
                dt = min(dt_target, dt * step_growth)
            dt = float(np.clip(dt, dt_min, dt_max))

            t_pred = float(clamp01(t_prev + direction * dt))
            if t_pred == t_prev:
                break

            u_pred, v_pred, Gp, ok = project_G0_fixed_t(
                C, S, t_pred, u_prev, v_prev, tol=tol_proj, rational=rational, it=35
            )
            xt = eval_bezier_curve(C, t_pred, rational=rational)
            xuv = eval_bezier_surface(S, u_pred, v_pred, rational=rational)
            dx = xt - xuv

            t_int,u_int,v_int=Interval(t_prev,t_pred),Interval(u_prev,u_pred),Interval(v_prev,v_pred)

            mask=np.array((t_int.contains(0,low_inclusive=True,up_inclusive=True) or t_int.contains(1,low_inclusive=True,up_inclusive=True),
                           ((u_int.contains(0,low_inclusive=True,up_inclusive=True) or u_int.contains(1,low_inclusive=True,up_inclusive=True)) and (u_int.width()!=0)),
                            ((v_int.contains(0,low_inclusive=False,up_inclusive=False) or v_int.contains(1,low_inclusive=False,up_inclusive=False)) and (v_int.width()!=0))),dtype=bool)

            tuv=np.array([t_pred,u_pred,v_pred])
            # mask = (np.isclose(tuv, 1) | np.isclose(tuv, 0)) & (~np.(tuv< (t, u, v)))
            if np.any(mask):

                if (not ok) or (np.linalg.norm(Gp) > 5.0 * tol_proj):
                    dt = max(dt_min, dt * step_shrink)
                    LAST_OK = False
                    if dt <= dt_min * 1.01:
                        break
                    continue

                if not np.all(mask):
                    t_proj, u_proj, v_proj, Gp, ok = project_G0(
                        C, S, *tuv, fixed=tuple(mask), tol=tol_proj, rational=rational
                    )
                    tuv = np.array([t_proj, u_proj, v_proj])

                if (not ok) or (np.linalg.norm(Gp) > 5.0 * tol_proj):
                    break
                xt = eval_bezier_curve(C, tuv[0], rational=rational)
                xuv = eval_bezier_surface(S, tuv[1], tuv[2], rational=rational)
                dx = xt - xuv
                _, Jb = G_and_J_curve_surface(C, S, tuv[0], tuv[1], tuv[2], rational=rational)
                cls = classify_contact_curve_surface(Jb, rel_thresh=1e-6)
                if (cls.get("type") != "overlap") or (not check_tangent(tuv[0], tuv[1], tuv[2])):
                    dt = max(dt_min, dt * step_shrink)
                    LAST_OK = False
                    if dt <= dt_min * 1.01:
                        break
                    continue
                if direction>0:

                    t_path.append(tuv[0])
                    uv_path.append(tuv[1:])
                    xyz_path.append(eval_bezier_curve(C, tuv[0], rational=rational))
                    t, u, v = tuv
                    break
                else:
                    t_path.insert(0, tuv[0])
                    uv_path.insert(0, tuv[1:])
                    xyz_path.insert(0, eval_bezier_curve(C, tuv[0], rational=rational))
                    t, u, v = tuv
                    break

            # if t_pred == t:
            #    break
            if (not ok) or np.linalg.norm(Gp) > 5.0 * tol_proj:
                dt = max(dt_min, dt * step_shrink)
                LAST_OK = False
                if dt <= dt_min * 1.01:
                    break
                continue

            if not check_tangent(t_pred, u_pred, v_pred):
                break

            # if (
            #    u_pred <= snap_eps
            #    or u_pred >= 1.0 - snap_eps
            #    or v_pred <= snap_eps
            #    or v_pred >= 1.0 - snap_eps
            # ):
            #    t_pred,u_pred,v_pred,_,_=project_G0(C,S,t_pred,u_pred,v_pred,(False,u_pred>=1.0 - snap_eps,v_pred>=1.0 - snap_eps), tol=tol_proj, rational=rational)
            #
            #    x = eval_bezier_curve(C, t_pred, rational=rational)
            #    if direction > 0:
            #        t_path.append(t_pred)
            #        uv_path.append( (u_pred, v_pred))
            #        xyz_path.append(x)
            #    else:
            #        t_path.insert(0, t_pred)
            #        uv_path.insert(0, (u_pred, v_pred))
            #        xyz_path.insert(0, x)
            #    print("e4",(t_pred, u_pred, v_pred))
            #    break

            x = eval_bezier_curve(C, t_pred, rational=rational)
            if direction > 0:

                t_path.append(t_pred)
                uv_path.append((u_pred, v_pred))
                xyz_path.append(x)
            else:
                t_path.insert(0, t_pred)
                uv_path.insert(0, (u_pred, v_pred))
                xyz_path.insert(0, x)

            t, u, v = t_pred, u_pred, v_pred
            LAST_OK = True
            pts += 1

        at_bnd = t <= 0.0 or t >= 1.0
        return "boundary" if at_bnd else "tangent_or_min_step"

    start_reason = march_nullspace(-1)
    # print(start_reason, uv_path)
    end_reason = march_nullspace(+1)
    # print(end_reason,uv_path)
    if len(t_path) <= 1:
        # Fallback to fixed-t marching if nullspace tracing fails to grow a path
        t_path[:] = [t0]
        uv_path[:] = [(u0, v0)]
        xyz_path[:] = [eval_bezier_curve(C, t0, rational=rational)]
        start_reason = march(-1)
        end_reason = march(+1)
    return {
        "kind": "overlap" if len(t_path) > 1 else "none",
        "t_path": t_path,
        "uv_path": uv_path,
        "xyz_path": xyz_path,
        "start": start_reason,
        "end": end_reason,
    }


def contact_detect_and_extract_curve_surface(
    Cseg,
    Sseg,
    seed_tuv=(0.5, 0.5, 0.5),
    sv_thresh=1e-8,
    tol_proj=1e-12,
    angle_tol=1e-3,
    sag_tol=None,
    rational: bool | None = None,
    extra_seeds=None,
):
    """Local classification of a single Bézier curve/surface pair on a cell."""
    seeds = [seed_tuv]
    if extra_seeds:
        seeds.extend(extra_seeds)

    for t0, u0, v0 in seeds:
        t, u, v, G, J = newton_project_G0_curve_surface(
            Cseg, Sseg, t0, u0, v0, tol=tol_proj, rational=rational
        )
        if np.linalg.norm(G) > 5.0 * tol_proj:
            continue

        cls = classify_contact_curve_surface(J, sv_thresh, rel_thresh=1e-6)
        if cls["type"] == "isolated":
            x = eval_bezier_curve(Cseg, t, rational=rational)
            return {"type": "isolated", "t": t, "u": u, "v": v, "point": x}
        if cls["type"] == "overlap":
            res = trace_curve_surface_overlap(
                Cseg,
                Sseg,
                t,
                u,
                v,
                sag_tol=sag_tol*100,
                angle_tol=angle_tol,
                rational=rational,
                tol_proj=tol_proj,
                max_points=2000,
            )
            if res["kind"] == "overlap":
                return {
                    "type": "overlap",
                    "t_path": res["t_path"],
                    "uv_path": res["uv_path"],
                    "xyz_path": res["xyz_path"],
                    "start": res["start"],
                    "end": res["end"],
                }

    return {"type": "none"}


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def bern_no_sign_change(coeffs, eps: float = 0.0):
    return (np.min(coeffs) > eps) or (np.max(coeffs) < -eps)


def bernstein_envelope_min(dnet):
    return dnet.min()


def map_local_to_global_3(t_loc, u_loc, v_loc, t0, t1, u0, u1, v0, v1):
    return (
        t0 + (t1 - t0) * t_loc,
        u0 + (u1 - u0) * u_loc,
        v0 + (v1 - v0) * v_loc,
    )


def map_local_range_to_global(rng, a0, a1):
    return (a0 + (a1 - a0) * rng[0], a0 + (a1 - a0) * rng[1])


def cell_contains_known_isolated(isolated, t0, t1, u0, u1, v0, v1, margin=1e-9):
    for it in isolated:
        if (
            (t0 - margin) <= it["t"] <= (t1 + margin)
            and (u0 - margin) <= it["u"] <= (u1 + margin)
            and (v0 - margin) <= it["v"] <= (v1 + margin)
        ):
            return True
    return False


def _aabb_euclidean(ctrl, rational: bool | None = None):
    ectrl = dehomogenize_ctrl(ctrl, rational=rational)
    return aabb(ectrl.reshape(-1, ectrl.shape[-1]))

from mmcore.geom._nurbs_eval import from_homogeneous_1d
def _bez_get_curve_tol_adapter(c, tol, rational=False, interval=None):
    if interval is None:
        interval = 0.,1.
    U=np.empty((c.shape[0]*2))
    U[:c.shape[0]]=interval[0]
    U[ c.shape[0]:] = interval[1]
    if rational:
        P,w=from_homogeneous_1d(c)
    else:
        P,w=c,np.ones(c.shape[0])

    return _nurbs_curve_param_tol_conservative(P,w, U,c.shape[0]-1,tol)

def _bez_get_surface_tol_adapter(s, tol, rational=False):
    try:
        _, Su, Sv, Suu, Suv, Svv = eval_bezier_surface_derivs2(s, 0.5, 0.5, rational=rational)
        du, dv = compute_parametric_tolerance_surface(Su, Sv, Suu, Suv, Svv, spt=tol)
        return float(du), float(dv)
    except Exception:
        return tol, tol


def curve_variation(ctrl, rational: bool | None = None):
    ectrl = dehomogenize_ctrl(ctrl, rational=rational)
    return float(np.sum(ectrl.max(axis=0) - ectrl.min(axis=0)))


def surface_axis_variation(ctrl, axis: int, rational: bool | None = None):
    ectrl = dehomogenize_ctrl(ctrl, rational=rational)
    if axis == 0:
        diff = ectrl[1:, :, :] - ectrl[:-1, :, :]
    else:
        diff = ectrl[:, 1:, :] - ectrl[:, :-1, :]
    return float(np.sum(np.linalg.norm(diff, axis=-1)))


def is_inside_cell(t, u, v, t0, t1, u0, u1, v0, v1, margin=0.0):
    return (
        (t0 - margin) <= t <= (t1 + margin)
        and (u0 - margin) <= u <= (u1 + margin)
        and (v0 - margin) <= v <= (v1 + margin)
    )


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------

class IsolatedIntersection(TypedDict):
    t: float
    u: float
    v: float
    point: NDArray


class OverlapIntersection(TypedDict):
    t_path: NDArray
    uv_path: NDArray
    xyz_path: NDArray
    start: str
    end: str


class IntersectionStats(TypedDict):
    cells: int
    pruned: int
    overlap_traces: int
    pruned_by: List[str]


class IntersectionResult(TypedDict):
    isolated: List[IsolatedIntersection]
    overlaps: List[OverlapIntersection]
    stats: IntersectionStats


# ---------------------------------------------------------------------------
# Main solver
# ---------------------------------------------------------------------------
def bernstein_eval_midpoint_trivariate(net: np.ndarray) -> np.ndarray:
    """
    Evaluate a trivariate Bernstein control net at (t,u,v) = (0.5, 0.5, 0.5)
    using De Casteljau midpoint averaging.

    Works for scalar-valued nets (..., 1) and vector-valued nets (..., k).
    Assumes the first 3 axes are (t,u,v) in that order.
    """
    Q = np.asarray(net, dtype=float)

    # Evaluate along t (axis 0) at 0.5
    while Q.shape[0] > 1:
        Q = 0.5 * (Q[:-1, ...] + Q[1:, ...])
    Q = Q[0, ...]

    # Evaluate along u (now axis 0) at 0.5
    while Q.shape[0] > 1:
        Q = 0.5 * (Q[:-1, ...] + Q[1:, ...])
    Q = Q[0, ...]

    # Evaluate along v (now axis 0) at 0.5
    while Q.shape[0] > 1:
        Q = 0.5 * (Q[:-1, ...] + Q[1:, ...])
    Q = Q[0, ...]

    return Q


def bernstein_lipschitz_lower_bound_midpoint_trivariate(
    dn: np.ndarray,
    dtn: np.ndarray,
    dun: np.ndarray,
    dvn: np.ndarray,
) -> tuple[float, float, tuple[float, float, float]]:
    """
    Certified Lipschitz-style lower bound on dn over [0,1]^3 using:
        dn(x) >= dn(c) - 0.5*(Mt + Mu + Mv)
    where c = (0.5,0.5,0.5) and
        Mt >= sup |∂dn/∂t|, etc.

    For Bernstein polynomials, the range of a derivative polynomial lies in the
    convex hull of its Bernstein coefficients, hence:
        sup |∂dn/∂t| <= max_i |(dtn)_i|  (and similarly for u,v).

    Returns
    -------
    lb : float
        Lower bound over the cube.
    dc : float
        dn evaluated at the midpoint.
    (Mt, Mu, Mv) : tuple[float,float,float]
        Sup-norm bounds for each partial derivative (via coefficient max abs).
    """
    dc = float(np.squeeze(bernstein_eval_midpoint_trivariate(dn)))

    Mt = float(np.max(np.abs(dtn))) if (dtn is not None and dtn.size) else 0.0
    Mu = float(np.max(np.abs(dun))) if (dun is not None and dun.size) else 0.0
    Mv = float(np.max(np.abs(dvn))) if (dvn is not None and dvn.size) else 0.0

    lb = dc - 0.5 * (Mt + Mu + Mv)
    return lb, dc, (Mt, Mu, Mv)


def choose_subdivision_axis_param_ratio(
    Pseg: np.ndarray,
    Sseg: np.ndarray,
    t0: float,
    t1: float,
    u0: float,
    u1: float,
    v0: float,
    v1: float,
    tol_t: float,
    tol_u: float,
    tol_v: float,
    eps: float = 1e-18,
) -> int:
    """
    Better axis selection: split along the parameter direction that is
    the least 'resolved' with respect to the current parameter tolerances.

    Score = (global width) / (param tolerance)

    Returns: axis index in {0,1,2} corresponding to (t,u,v).
    """
    scores = np.array([-np.inf, -np.inf, -np.inf], dtype=float)

    # Only allow splits where the corresponding geometry actually has degree >= 1
    # (curve: t axis; surface: u and v axes).
    if Pseg.shape[0] > 1:
        scores[0] = (t1 - t0) / max(float(tol_t), eps)
    if Sseg.shape[0] > 1:
        scores[1] = (u1 - u0) / max(float(tol_u), eps)
    if Sseg.shape[1] > 1:
        scores[2] = (v1 - v0) / max(float(tol_v), eps)

    return int(np.argmax(scores))

from mmcore.numeric.algorithms.cygjk import gjk
from mmcore.numeric.vectors import norm,dot
def classify_points_3d(P: np.ndarray, eps: float = 1e-3):
    """
    Classify Nx3 points as 'line', 'plane', or '3d' using covariance eigenvalues.
    Returns (kind, ratios, evals, centroid, normal_or_direction).
    """
    P = np.asarray(P, dtype=np.float64)
    assert P.ndim == 2 and P.shape[1] == 3 and P.shape[0] >= 2
    c = P.mean(axis=0)
    X = P - c
    C = (X.T @ X) / max(len(P), 1)
    evals, evecs = np.linalg.eigh(C)          # ascending
    evals = np.maximum(evals, 0.0)
    l1, l2, l3 = evals[::-1]                  # descending
    if l1 <= 0: return "point", (0.0, 0.0), evals, c, None
    r2, r3 = l2 / l1, l3 / l1
    if r2 < eps:   kind, vec = "line",  evecs[:, 2]   # largest-eigenvector
    elif r3 < eps: kind, vec = "plane", evecs[:, 0]   # smallest-eigenvector (normal)
    else:          kind, vec = "3d",    None
    return kind, (r2, r3), evals, c, vec
def ch_separability(pts1,pts2, atol,rational=False):
    pts1,pts2=pts1.reshape(-1, pts1.shape[-1]), pts2.reshape(-1,  pts2.shape[-1])
    if rational:
        pts1,pts2=pts1[...,:-1]/pts1[...,None,-1], pts2[...,:-1]/pts2[...,None,-1]
    # cent1=np.average(pts1,axis=0)
    # cent2 = np.average(pts2, axis=0)
    ### print(cent1,cent2)
    # v1=pts1-cent1
    # v2 = pts2- cent2
    # cent2+v1+v1*1e-6
    ## print(v1,v1)
    #
    # r1=np.max(dot(v1, v1))
    # rr1=np.sqrt(r1)
    # r2= np.max(dot(v2, v2))
    # rr2 = np.sqrt(r2)
    # v1*=(1+rr1*atol/2)
    # v2 *= (1 + rr2 * atol/2)

    # nv1=v1/ np.reshape( dot(v1,v1),(-1,1))
    # print(nv1)
    # nv2 = v2/  np.reshape( dot(v2,v2),(-1,1))

    # pts1=pts1+v1
    # pts2=pts2+ v2
    #if classify_points_3d(pts1, eps=1e-5)[0] != "3d":
    #        return 2
    #if classify_points_3d(pts2, eps=1e-5)[0] != "3d":
    #        return 3
    res=gjk(np.ascontiguousarray(pts1   ), np.ascontiguousarray(pts2  )   , tol=1e-5, max_iter=15  )
    # print(res,(pts1.tolist(), pts2.tolist()))
    return int(res)

from mmcore.numeric.intersection.ccx._bez_ccx3 import bezier_intersect_certified_full


# ---------------------------------------------------------------------------
# Curve-Surface Boundary Intersection
# ---------------------------------------------------------------------------

def curve_surface_boundary_intersect(
    C: NDArray,
    S: NDArray,
    sv_thresh: float = 1e-8,
    atol: float = 1e-3,
    rational: bool | None = None,
) -> tuple[list[IsolatedIntersection], list[OverlapIntersection]]:
    """
    Find intersections between a Bézier curve and the boundary curves of a Bézier surface.

    For a tensor-product surface S(u,v), the boundaries are:
    - u=0: isocurve at u=0, parameterized by v
    - u=1: isocurve at u=1, parameterized by v
    - v=0: isocurve at v=0, parameterized by u
    - v=1: isocurve at v=1, parameterized by u

    Parameters
    ----------
    C : NDArray
        Bézier curve control points, shape (p+1, dim) or (p+1, dim+1) if rational.
    S : NDArray
        Bézier surface control net, shape (nu+1, nv+1, dim) or (nu+1, nv+1, dim+1) if rational.
    sv_thresh : float, optional
        Singular-value threshold for distinguishing isolated vs. overlap contacts.
    atol : float, optional
        Geometric tolerance for intersection detection.
    rational : bool, optional
        When True, inputs are treated as homogeneous control nets with weights in last column.

    Returns
    -------
    isolated : list[IsolatedIntersection]
        List of isolated intersection points, each with:
        - t: curve parameter
        - u, v: surface parameters
        - point: 3D intersection point
    overlaps : list[OverlapIntersection]
        List of overlap segments (curve lying on boundary), each with:
        - t_path: curve parameter path
        - uv_path: surface (u,v) parameter path
        - xyz_path: 3D point path
        - start, end: boundary event descriptions

    Notes
    -----
    The function extracts the four boundary curves of the surface and intersects
    each with the input curve using certified curve-curve intersection. The resulting
    parameters are remapped to surface (u,v) coordinates:

    - u=0 boundary: CCX(C, S[0,:]) gives (t, s) → CSX(t, u=0, v=s)
    - u=1 boundary: CCX(C, S[-1,:]) gives (t, s) → CSX(t, u=1, v=s)
    - v=0 boundary: CCX(C, S[:,0]) gives (t, s) → CSX(t, u=s, v=0)
    - v=1 boundary: CCX(C, S[:,-1]) gives (t, s) → CSX(t, u=s, v=1)
    """
    if rational is None:
        rational = is_homogeneous_ctrl(C) or is_homogeneous_ctrl(S)

    # Extract boundary curves from surface
    u0_bnd, u1_bnd, v0_bnd, v1_bnd = bernstein_boundaries_2d(S)

    isolated: list[IsolatedIntersection] = []
    overlaps: list[OverlapIntersection] = []

    # Define boundary mapping: (boundary_curve, axis, fixed_value)
    # axis: 'u' means the fixed parameter is u, 'v' means fixed parameter is v
    # For u-fixed boundaries: CCX param 'v' maps to surface param 'v'
    # For v-fixed boundaries: CCX param 'v' maps to surface param 'u'
    boundaries = [
        (u0_bnd, 'u', 0.0),  # u=0 boundary, CCX.v -> surface.v
        (u1_bnd, 'u', 1.0),  # u=1 boundary, CCX.v -> surface.v
        (v0_bnd, 'v', 0.0),  # v=0 boundary, CCX.v -> surface.u
        (v1_bnd, 'v', 1.0),  # v=1 boundary, CCX.v -> surface.u
    ]

    for bnd_curve, fixed_axis, fixed_val in boundaries:
        # Intersect curve C with boundary curve
        ccx_result = bezier_intersect_certified_full(
            C, bnd_curve,
            sv_thresh=sv_thresh,
            atol=atol,
            rational=rational,
        )

        # Process isolated intersections
        for ccx_iso in ccx_result.get('isolated', []):
            t_curve = ccx_iso['u']  # parameter on input curve C
            t_bnd = ccx_iso['v']    # parameter on boundary curve
            point = ccx_iso['point']

            # Map to surface (u, v) coordinates
            if fixed_axis == 'u':
                # u is fixed, boundary is parameterized by v
                u_surf = fixed_val
                v_surf = t_bnd
            else:  # fixed_axis == 'v'
                # v is fixed, boundary is parameterized by u
                u_surf = t_bnd
                v_surf = fixed_val

            isolated.append(IsolatedIntersection(
                t=float(t_curve),
                u=float(u_surf),
                v=float(v_surf),
                point=np.asarray(point),
            ))

        # Process overlap intersections
        for ccx_ovl in ccx_result.get('overlaps', []):
            uv_path_ccx = ccx_ovl['uv_path']  # Each entry is (t_curve, t_bnd)
            xyz_path = ccx_ovl['xyz_path']
            start = ccx_ovl.get('start', 'unknown')
            end = ccx_ovl.get('end', 'unknown')

            # Build t_path and uv_path for surface coordinates
            t_path = []
            uv_path = []

            for uv_ccx in uv_path_ccx:
                t_curve = uv_ccx[0]  # parameter on input curve C
                t_bnd = uv_ccx[1]    # parameter on boundary curve

                # Map to surface (u, v) coordinates
                if fixed_axis == 'u':
                    u_surf = fixed_val
                    v_surf = t_bnd
                else:  # fixed_axis == 'v'
                    u_surf = t_bnd
                    v_surf = fixed_val

                t_path.append(float(t_curve))
                uv_path.append((float(u_surf), float(v_surf)))

            overlaps.append(OverlapIntersection(
                t_path=np.asarray(t_path),
                uv_path=np.asarray(uv_path),
                xyz_path=np.asarray(xyz_path),
                start=start,
                end=end,
            ))

    return isolated, overlaps


# Alias for backwards compatibility
check_boundaries_inter = curve_surface_boundary_intersect


def bezier_curve_surface_intersect_certified(
    C: NDArray,
    S: NDArray,
    sv_thresh: float = 1e-8,
    atol: float = 1e-3,
    rational: bool | None = None,
    angle_tol: float = 0.01,
    max_depth: int = 60,
    overlap_dist_tol: float | None = None,
) -> IntersectionResult:
    """Certified intersection for (possibly rational) Bézier curve & surface.

    Parameters
    ----------
    C : array_like
        Bézier curve control points, shape (p+1, dim) or (p+1, dim+1) if rational.
    S : array_like
        Bézier surface control net, shape (q+1, r+1, dim) or (q+1, r+1, dim+1) if rational.
    sv_thresh : float, optional
        Singular-value threshold distinguishing isolated vs. overlap contacts.
    atol : float, optional
        Geometric tolerance used for pruning and acceptance.
    rational : bool, optional
        When True, inputs are treated as homogeneous control nets with weights in the last column.
    angle_tol : float, optional
        Angular tolerance for overlap tracing (rad).
    max_depth : int, optional
        Safety cap on subdivision depth.
    overlap_dist_tol : float, optional
        Absolute distance tolerance for considering a span "overlapping".
        Only affects span-level overlap confirmation; tangency and rank
        criteria still apply.
    """

    if rational is None:
        rational = is_homogeneous_ctrl(C) or is_homogeneous_ctrl(S)
    # Fast span-level overlap confirmation using the analytic-segment criterion:
    # within a single Bézier span (C∞), if we confirm overlap at a few interior
    # points then the overlap is taken as the entire span. We intentionally
    # DO NOT extend across knot spans (continuity drops below C∞), and we also
    # reject degenerate surface normals to avoid false overlaps at singularities.
    # If overlap_dist_tol is provided, only the distance threshold is relaxed;
    # tangency/rank criteria are still required for confirmation.
    def _bbox_diag_len(Cc, Ss):
        ec = dehomogenize_ctrl(Cc, rational=rational)
        es = dehomogenize_ctrl(Ss, rational=rational).reshape(-1, ec.shape[-1])
        mins = np.minimum(np.min(ec, axis=0), np.min(es, axis=0))
        maxs = np.maximum(np.max(ec, axis=0), np.max(es, axis=0))
        return float(np.linalg.norm(maxs - mins))

    span_scale = _bbox_diag_len(C, S)
    span_tol_proj = max(1e-12, 1e-10 * span_scale)
    span_tol_conf = max(5.0 * span_tol_proj, 1e-9 * span_scale)
    if overlap_dist_tol is not None:
        overlap_dist_tol = float(overlap_dist_tol)
        if overlap_dist_tol > 0.0:
            span_tol_conf = max(span_tol_proj, overlap_dist_tol)
    span_overlap = confirm_overlap_span(
        C,
        S,
        span_tol_proj,
        span_tol_conf,
        angle_tol,
        sv_thresh,
        rational=rational,
        rel_thresh=1e-6,
    )
    pre_overlaps: list["OverlapIntersection"] = []
    stats = IntersectionStats(cells=0, pruned=0, overlap_traces=0, pruned_by=[])
    if span_overlap is not None:
        if span_overlap.get("partial"):
            span_overlap = {k: v for k, v in span_overlap.items() if k != "partial"}
            pre_overlaps.append(span_overlap)
            stats = IntersectionStats(
                cells=0, pruned=0, overlap_traces=1, pruned_by=["span_confirm_partial"]
            )
        else:
            stats = IntersectionStats(cells=1, pruned=0, overlap_traces=1, pruned_by=["span_confirm"])
            return IntersectionResult(isolated=[], overlaps=[span_overlap], stats=stats)
    #isolated,overlaps=curve_surface_boundary_intersect(C,S,rational=rational,atol=atol,sv_thresh=sv_thresh)
    #print(isolated)
    #print(overlaps)
    isolated: list["IsolatedIntersection"] = []
    overlaps: list["OverlapIntersection"] = pre_overlaps

    atol_sq = atol * atol

    dn0 = distance_squared_net_curve_surface(C, S, rational=rational)[..., np.newaxis]
    dt0 = bernstein_partial_derivative_coeffs(dn0, 0)
    du0 = bernstein_partial_derivative_coeffs(dn0, 1)
    dv0 = bernstein_partial_derivative_coeffs(dn0, 2)

    stack = [(
        C.copy(),
        S.copy(),
        dn0,
        dt0,
        du0,
        dv0,
        0.0,
        1.0,
        0.0,
        1.0,
        0.0,
        1.0,
        0,
    )]

    def near_existing_isolated(point: NDArray):
        for it in isolated:
            d = it["point"] - point
            if float(np.dot(d, d)) <= atol_sq:
                return True
        return False

    def _overlap_range(ov):
        t_vals = np.asarray(ov["t_path"], dtype=float)
        return float(np.min(t_vals)), float(np.max(t_vals))

    def _overlap_duplicate(new_ov, existing, tol=1e-6):
        if not existing:
            return False
        t0, t1 = _overlap_range(new_ov)
        x0 = new_ov["xyz_path"][0]
        x1 = new_ov["xyz_path"][-1]
        for ov in existing:
            a0, a1 = _overlap_range(ov)
            if (t0 >= a0 - tol) and (t1 <= a1 + tol):
                y0 = ov["xyz_path"][0]
                y1 = ov["xyz_path"][-1]
                if (np.linalg.norm(x0 - y0) <= tol) and (np.linalg.norm(x1 - y1) <= tol):
                    return True
        return False

    def _cell_inside_overlap(t0_cell, t1_cell, overlaps_list, tol=1e-6):
        if not overlaps_list:
            return False
        t_min_c = min(t0_cell, t1_cell)
        t_max_c = max(t0_cell, t1_cell)
        for ov in overlaps_list:
            t_min, t_max = _overlap_range(ov)
            if (t_min - tol) <= t_min_c and (t_max + tol) >= t_max_c:
                return True
        return False

    while stack:
        Pseg, Sseg, dn, dtn, dun, dvn, t0, t1, u0, u1, v0, v1, depth = stack.pop()
        stats["cells"] += 1

        # Skip cells fully covered by an already-confirmed overlap.
        if _cell_inside_overlap(t0, t1, overlaps, tol=atol):
            stats["pruned"] += 1
            stats["pruned_by"].append("overlap_covered")
            continue

        if depth > max_depth:
            stats["pruned"] += 1
            stats["pruned_by"].append("max_depth")
            continue

        box_c = _aabb_euclidean(Pseg, rational=rational)
        box_s = _aabb_euclidean(Sseg, rational=rational)
        if not aabb_intersect(box_c, box_s):
            stats["pruned"] += 1
            stats["pruned_by"].append("bbox")
            continue

        # Component sign test on G = C - S (or homogeneous cross-multiply)
        if not rational:
            Fst = Pseg[:, None, None, :] - Sseg[None, :, :, :]
            cert = all(Fst[..., d].min() <= 0.0 and Fst[..., d].max() >= 0.0 for d in range(Fst.shape[-1]))
        else:
            Ph = Pseg
            Sh = Sseg
            wc = Ph[:, -1]
            ws = Sh[:, :, -1]
            Ph_xyz = Ph[:, :-1][:, None, None, :]
            Sh_xyz = Sh[:, :, :-1][None, :, :, :]
            Fst = Ph_xyz * ws[None, :, :, None] - Sh_xyz * wc[:, None, None, None]
            eps = np.finfo(float).eps
            cert = all(Fst[..., d].min() <= eps and Fst[..., d].max() >= -eps for d in range(Fst.shape[-1]))

        if not cert:
            stats["pruned"] += 1
            stats["pruned_by"].append("component_sign")
            continue

        bmin = bernstein_envelope_min(dn)
        if bmin > 0.0:
            stats["pruned"] += 1
            stats["pruned_by"].append("distance_envelope")
            continue

        # (3.2) Certified Lipschitz-style positivity test:
        # If dn(mid) - 0.5*(Mt+Mu+Mv) > 0 => dn > 0 everywhere in the cell.
        lb, _dc, _Ms = bernstein_lipschitz_lower_bound_midpoint_trivariate(dn, dtn, dun, dvn)
        if lb > 0.0:
            stats["pruned"] += 1
            stats["pruned_by"].append("lipschitz_positive")
            continue

        eps_sign = 1e-14
        if (
            bern_no_sign_change(np.squeeze(dtn), eps=eps_sign)
            or bern_no_sign_change(np.squeeze(dun), eps=eps_sign)
            or bern_no_sign_change(np.squeeze(dvn), eps=eps_sign)
        ):
            stats["pruned"] += 1
            stats["pruned_by"].append("grad_sign")
            continue

        dbox_c = np.asarray(box_c[1]) - np.asarray(box_c[0])
        dbox_s = np.asarray(box_s[1]) - np.asarray(box_s[0])
        scale = max(float(np.linalg.norm(dbox_c)), float(np.linalg.norm(dbox_s)), 1.0)
        tol_proj = max(1e-12, 1e-10 * scale)

        ch_sep=ch_separability(Pseg,Sseg,atol,rational=rational)

        if ch_sep==0:
            stats["pruned"] += 1
            stats["pruned_by"].append("ch_sep")
            continue

        # Early contact detection (ccx-style)
        has_known_iso = cell_contains_known_isolated(isolated, t0, t1, u0, u1, v0, v1)
        res = contact_detect_and_extract_curve_surface(
            Pseg,
            Sseg,
            seed_tuv=(0.5, 0.5, 0.5),
            #extra_seeds=[(0.25, 0.5, 0.5), (0.75, 0.5, 0.5)],
            sv_thresh=sv_thresh,
            tol_proj=tol_proj,
            angle_tol=angle_tol,
            sag_tol=atol,
            rational=rational,
        )

        if res["type"] == "overlap":
            t_path_g = [t0 + (t1 - t0) * t for t in res["t_path"]]
            uv_path_g = [
                (u0 + (u1 - u0) * uv[0], v0 + (v1 - v0) * uv[1]) for uv in res["uv_path"]
            ]
            new_ov = {
                "t_path": np.asarray(t_path_g),
                "uv_path": np.asarray(uv_path_g),
                "xyz_path": np.asarray(res["xyz_path"]),
                "start": res["start"],
                "end": res["end"],
            }
            if not _overlap_duplicate(new_ov, overlaps, tol=atol):
                overlaps.append(new_ov)
                stats["overlap_traces"] += 1
            continue

        if res["type"] == "isolated" and not has_known_iso:
            t_loc, u_loc, v_loc = res["t"], res["u"], res["v"]
            t_hit, u_hit, v_hit = map_local_to_global_3(t_loc, u_loc, v_loc, t0, t1, u0, u1, v0, v1)
            x = res["point"]
            if not near_existing_isolated(x):
                isolated.append({"t": t_hit, "u": u_hit, "v": v_hit, "point": x})

            # Cut out neighborhood around the isolated root (early termination)
            C0,C1,C2= eval_bezier_curve_derivs2(Pseg,t_loc,rational=rational)
            S0, Su, Sv ,Suu,Suv , Svv =  eval_bezier_surface_derivs2(Sseg, u_loc,v_loc, rational=rational)
            tol_t=compute_parametric_curvature_tolerance_curve(C1,C2, spt=atol)
            tol_u, tol_v = compute_parametric_curvature_tolerance_surface(Su, Sv ,Suu, Svv, atol)
            res_cut = bernstein_cutout_box_nd(
                dn, np.array([t_loc, u_loc, v_loc]), half=np.array([tol_t, tol_u, tol_v]), return_ranges=True
            )
            for subpatch, ranges in res_cut:
                t_rng, u_rng, v_rng = ranges
                sub_dtn = bernstein_partial_derivative_coeffs(subpatch, 0)
                sub_dun = bernstein_partial_derivative_coeffs(subpatch, 1)
                sub_dvn = bernstein_partial_derivative_coeffs(subpatch, 2)
                Psub = bernstein_trim_nd(Pseg, ranges=[t_rng])
                Ssub = bernstein_trim_nd(Sseg, ranges=[u_rng, v_rng])
                t0g, t1g = map_local_range_to_global(t_rng, t0, t1)
                u0g, u1g = map_local_range_to_global(u_rng, u0, u1)
                v0g, v1g = map_local_range_to_global(v_rng, v0, v1)
                stack.append((Psub, Ssub, subpatch, sub_dtn, sub_dun, sub_dvn, t0g, t1g, u0g, u1g, v0g, v1g, depth + 1))

            stats["pruned"] += 1
            stats["pruned_by"].append("isolated_cutout")
            continue
        # Tolerances for parameter-space resolution
        tol_t = _bez_get_curve_tol_adapter(Pseg, atol, rational=rational)
        tol_u, tol_v = _bez_get_surface_tol_adapter(Sseg, atol, rational=rational)

        # Small cell check
        small_geom = (np.dot(dbox_c, dbox_c) < atol_sq) and (np.dot(dbox_s, dbox_s) < atol_sq)
        small_param = (t1 - t0) <= tol_t and (u1 - u0) <= tol_u and (v1 - v0) <= tol_v

        if small_geom or small_param:
            t_guess, u_guess, v_guess = 0.5, 0.5, 0.5
            t_loc, u_loc, v_loc, Gc, Jc = newton_project_G0_curve_surface(
                Pseg, Sseg, t_guess, u_guess, v_guess, tol=tol_proj, rational=rational
            )
            Dval = float(np.dot(Gc, Gc))
            if Dval <= atol_sq and is_inside_cell(t_loc, u_loc, v_loc, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, margin=max(tol_t, tol_u, tol_v)):
                t_hit, u_hit, v_hit = map_local_to_global_3(t_loc, u_loc, v_loc, t0, t1, u0, u1, v0, v1)
                x = eval_bezier_curve(Pseg, t_loc, rational=rational)
                cls = classify_contact_curve_surface(Jc, sv_thresh=sv_thresh, rel_thresh=1e-6)
                if cls["type"] == "overlap":
                    res = trace_curve_surface_overlap(
                        Pseg,
                        Sseg,
                        t_loc,
                        u_loc,
                        v_loc,
                        sag_tol=atol,
                        angle_tol=angle_tol,
                        rational=rational,
                        tol_proj=tol_proj,
                        max_points=2000,
                    )
                    if res["kind"] == "overlap":
                        t_path_g = [t0 + (t1 - t0) * t for t in res["t_path"]]
                        uv_path_g = [
                            (u0 + (u1 - u0) * uv[0], v0 + (v1 - v0) * uv[1]) for uv in res["uv_path"]
                        ]
                        overlaps.append(
                            {
                                "t_path": np.asarray(t_path_g),
                                "uv_path": np.asarray(uv_path_g),
                                "xyz_path": np.asarray(res["xyz_path"]),
                                "start": res["start"],
                                "end": res["end"],
                            }
                        )
                        stats["overlap_traces"] += 1
                        continue
                if not near_existing_isolated(x):
                    isolated.append({"t": t_hit, "u": u_hit, "v": v_hit, "point": x})
            stats["pruned"] += 1
            stats["pruned_by"].append("small_cell")

            continue

        # -------------------------------------------------------------------
        # (4) Better axis selection for subdivision
        #     Split direction chosen by param-width / param-tolerance ratio.
        # -------------------------------------------------------------------
        axis = choose_subdivision_axis_param_ratio(
            Pseg, Sseg, t0, t1, u0, u1, v0, v1, tol_t, tol_u, tol_v
        )

        # Always split at 0.5; maintain local-derivative nets by scaling the
        # derivative *in the split axis* by 0.5 (chain rule after reparam).
        if axis == 0:
            # Split curve in t
            PL, PR = de_casteljau_split_nd(Pseg, axis=0, t=0.5)

            dnL, dnR = de_casteljau_split_nd(dn, axis=0, t=0.5)
            dtnL, dtnR = de_casteljau_split_nd(dtn, axis=0, t=0.5)
            dunL, dunR = de_casteljau_split_nd(dun, axis=0, t=0.5)
            dvnL, dvnR = de_casteljau_split_nd(dvn, axis=0, t=0.5)

            # Chain rule scaling for local t on the child cell
            dtnL = 0.5 * dtnL
            dtnR = 0.5 * dtnR

            t_mid = 0.5 * (t0 + t1)
            stack.append((PR, Sseg, dnR, dtnR, dunR, dvnR, t_mid, t1, u0, u1, v0, v1, depth + 1))
            stack.append((PL, Sseg, dnL, dtnL, dunL, dvnL, t0, t_mid, u0, u1, v0, v1, depth + 1))

        elif axis == 1:
            # Split surface in u
            SL, SR = de_casteljau_split_nd(Sseg, axis=0, t=0.5)

            dnL, dnR = de_casteljau_split_nd(dn, axis=1, t=0.5)
            dtnL, dtnR = de_casteljau_split_nd(dtn, axis=1, t=0.5)
            dunL, dunR = de_casteljau_split_nd(dun, axis=1, t=0.5)
            dvnL, dvnR = de_casteljau_split_nd(dvn, axis=1, t=0.5)

            # Chain rule scaling for local u on the child cell
            dunL = 0.5 * dunL
            dunR = 0.5 * dunR

            u_mid = 0.5 * (u0 + u1)
            stack.append((Pseg, SR, dnR, dtnR, dunR, dvnR, t0, t1, u_mid, u1, v0, v1, depth + 1))
            stack.append((Pseg, SL, dnL, dtnL, dunL, dvnL, t0, t1, u0, u_mid, v0, v1, depth + 1))

        else:
            # Split surface in v
            SL, SR = de_casteljau_split_nd(Sseg, axis=1, t=0.5)

            dnL, dnR = de_casteljau_split_nd(dn, axis=2, t=0.5)
            dtnL, dtnR = de_casteljau_split_nd(dtn, axis=2, t=0.5)
            dunL, dunR = de_casteljau_split_nd(dun, axis=2, t=0.5)
            dvnL, dvnR = de_casteljau_split_nd(dvn, axis=2, t=0.5)

            # Chain rule scaling for local v on the child cell
            dvnL = 0.5 * dvnL
            dvnR = 0.5 * dvnR

            v_mid = 0.5 * (v0 + v1)
            stack.append((Pseg, SR, dnR, dtnR, dunR, dvnR, t0, t1, u0, u1, v_mid, v1, depth + 1))
            stack.append((Pseg, SL, dnL, dtnL, dunL, dvnL, t0, t1, u0, u1, v0, v_mid, depth + 1))

    def _t_in_overlap_local(t, overlaps_list, tol=1e-6):
        for ov in overlaps_list:
            t_min = float(np.min(ov["t_path"]))
            t_max = float(np.max(ov["t_path"]))
            if (t_min - tol) <= t <= (t_max + tol):
                return True
        return False

    # If we only found overlap-like isolated points, try tracing the full patch from the farthest t.
    if isolated:
        cand = None
        cand_t = None
        for it in isolated:
            t, u, v = it["t"], it["u"], it["v"]
            if _t_in_overlap_local(t, overlaps):
                continue
            Gc, Jc = G_and_J_curve_surface(C, S, t, u, v, rational=rational)
            if not overlap_like_svd(Jc, sv_thresh=sv_thresh, rel_thresh=1e-6):
                continue
            if cand_t is None or t > cand_t:
                cand = (t, u, v)
                cand_t = t
        if cand is not None:
            t, u, v = cand
            res = trace_curve_surface_overlap(
                C,
                S,
                t,
                u,
                v,
                sag_tol=atol,
                angle_tol=angle_tol,
                rational=rational,
                tol_proj=None,
                max_points=2000,
            )
            if res["kind"] == "overlap":
                overlaps.append(
                    {
                        "t_path": np.asarray(res["t_path"]),
                        "uv_path": np.asarray(res["uv_path"]),
                        "xyz_path": np.asarray(res["xyz_path"]),
                        "start": res["start"],
                        "end": res["end"],
                    }
                )

    if overlaps and isolated:
        isolated = [it for it in isolated if not _t_in_overlap_local(it["t"], overlaps)]

    return IntersectionResult(isolated=isolated, overlaps=overlaps, stats=stats)


if __name__ == "__main__":
    import numpy as np
    try:
        import rich
        from rich.console import Console
        from rich.style import Style, StyleType
        from rich.table import Table

        # Create a console instance
        console = Console()

        def make_rich_table(inter1, inter2, case_name=None, oracle_name="OCC", float_fmt=lambda x: "{:.6f}".format(float(x))):
            def uv_fmt(uv):
                return f"{float(uv[0]):4f}, {float(uv[1]):4f}"

            def overlap_fmt(overlap):
                return f"[{uv_fmt(overlap['uv_path'][0])}] to [{uv_fmt(overlap['uv_path'][-1])}]"

            mismatch_color = "#e34f66"
            match_color = "#0dd962"

            def match_str(v):
                return f"[{match_color}]{v}[/{match_color}]"

            def mismatch_str(v):
                return f"[{mismatch_color}]{v}[/{mismatch_color}]"

            # Create a table
            table = Table(title=case_name + " (isolated)")

            # Add columns
            table.add_column("mmcore", style="default", no_wrap=True)
            table.add_column(oracle_name, style="default")
            table.add_column("match", justify="left", style="bold")
            matches = []
            #print(inter1["isolated"])
            for first, second in itertools.zip_longest(
                sorted(inter1["isolated"], key=lambda x: x["u"]), sorted(inter2["isolated"], key=lambda x: x["u"])
            ):

                if first is None:
                    table.add_row(float_fmt(first), f"{float_fmt(second['u'])} ,{float_fmt(second['v'])}", mismatch_str("X"), style=mismatch_color)
                    matches.append(False)
                elif second is None:
                    table.add_row(f"{float_fmt(first['u'])} ,{float_fmt(first['v'])}", second, mismatch_str("X"), style=mismatch_color)
                    matches.append(False)
                else:
                    match_ = np.allclose(np.array((first["u"], first["v"])), np.array((second["u"], second["v"])))
                    matches.append(match_)
                    table.add_row(
                        f"{float_fmt(first['u'])} ,{float_fmt(first['v'])}",
                        f"{float_fmt(second['u'])} ,{float_fmt(second['v'])}",
                        match_str("OK") if match_ else mismatch_str("X"),
                    )
                    if not match_:

                        table.rows[-1].style = Style(color=mismatch_color, bold=True)
            if len(matches) == 0:

                table.add_row(None, None, match_str("OK"))
                matches.append(True)
            console.print(table, "\n")
            # Add rows
            table = Table(title=case_name + " (overlaps)")

            # Add columns
            table.add_column("mmcore", style="default", no_wrap=True)
            table.add_column(oracle_name, style="default")
            table.add_column("match", justify="left", style="bold")
            matches = []
            for first, second in itertools.zip_longest(
                sorted(inter1["overlaps"], key=lambda x: x["uv_path"][0][0]), sorted(inter2["overlaps"], key=lambda x: x["uv_path"][0][0])
            ):

                if first is None:
                    table.add_row(first, overlap_fmt(second), mismatch_str("X"), style=mismatch_color)
                    matches.append(False)
                elif second is None:
                    table.add_row(overlap_fmt(first), second, mismatch_str("X"), style=mismatch_color)
                    matches.append(False)

                else:

                    match_ = np.allclose((first["uv_path"][0], first["uv_path"][-1]), (second["uv_path"][0], second["uv_path"][-1]))
                    matches.append(match_)
                    table.add_row(overlap_fmt(first), overlap_fmt(second), match_str("OK") if match_ else mismatch_str("X"))

                    if not match_:
                        table.rows[-1].style = Style(color=mismatch_color, bold=True)

            if len(matches) == 0:
                table.add_row(None, None, match_str("OK"))
                matches.append(True)
            # Print the table
            console.print(table)

    except ImportError:

        def make_rich_table(inter1, inter2, *args, **kwargs):
            print('Pretty output requires "pip install rich"')
            print("verify with OCC (mmcore result, occ result):", inter1["isolated"], inter2["isolated"])

    np.set_printoptions(edgeitems=3)

    surf1 = np.array(
            [
                [[-29.28237848, 6.13222489, 0.0], [3.38719429, 30.19487236, 19.20895833]],
                [[-22.42788848, -22.25025954, 21.6074674], [19.17461186, -7.08406959, 0.0]],
            ]
        )

    crv1=np.array([[-21.19167858,  -1.06882577,  25.59702016],
           [-23.82588542, -26.53996332,   0.        ],
           [ -6.52097953,  26.04573929,   0.        ],
           [  5.81667121,  -9.0084582 ,  17.47802526]])
    s=time.perf_counter()
    res=bezier_curve_surface_intersect_certified(crv1, surf1,rational=False)

    gt={'isolated':[{
        "t":0.223246,
        "u":0.594769,'v':0.130421,
    }, {
        't':0.818801,"u":0.562506,'v':0.641888
    }],'overlaps':[]}
    print("case1",time.perf_counter()-s)
    make_rich_table(res,gt,'case1', "Rhino (8.26.25349.19002, 2025-12-15)")
    surf2 = np.array([[[-13.75962333043464, -9.35078823406258, 30.38984420876391, 1.0], [-15.661286427866942, 1.8280054186360069, 0.0, 1.0]], [[-9.729522963522962, -6.612005769745033, 21.488864919219694, 0.7071067811865476], [-3.1696010379412516, 2.6372738992898688, 0.0, 0.7071067811865476]], [[-13.75962333043464, -9.35078823406258, 30.38984420876391, 1.0], [-2.5808296777360518, -7.449125136630279, 0.0, 1.0]]]
                     )

    crv2 = np.array([[-8.089928672303788, -9.35078823406258, 15.194922104381954, 1.0], [-5.720443423501492, -2.6029262297235642, 10.744432459609845, 0.7071067811865476], [-13.75962333043464, -3.6810935759317296, 15.194922104381954, 1.0]])
    s=time.perf_counter()
    res=bezier_curve_surface_intersect_certified(crv2, surf2,rational=True)
    print("case2", time.perf_counter() - s)
    gt={
        "isolated": [],
        "overlaps": [
            {
                "t_path": [0.1153939700483303, 1.],
                "uv_path": [[1.0, 0.500000000000000], [0.11539397004833106, 0.500000000000000]],
            }
        ],
    }

    make_rich_table(res, gt, "case2", "Rhino (8.26.25349.19002, 2025-12-15)")

    surf3 = np.array([[[-15.057302653755286, 21.16410483032787, 5.328685886563102, 1.0], [-10.36538313655037, 13.550740864293562, 3.7679499252018203, 0.7071067811865476], [-14.658865410902903, 19.16364151048727, 3.288929594620489, 1.0]], [[-15.448870346346077, 14.008905674843858, 3.7679499252018203, 0.7071067811865476], [-9.724550702226589, 9.104779161190693, 2.664342943281552, 0.5000000000000001], [-13.752591491073645, 12.876102172167808, 2.3256244192012705, 0.7071067811865476]], [[-20.495481536187885, 13.020885085978726, 5.328685886563102, 1.0], [-13.077952798947893, 9.48889381764453, 3.7679499252018203, 0.7071067811865476], [-18.495018216347287, 13.41932232883111, 3.288929594620489, 1.0]]]
    )

    crv3 = np.array([[-13.704782222797013, 20.157097244338125, 4.980117410327855, 1.0], [-13.780432085162484, 14.253220150508163, 3.521474791948015, 0.7071067811865476], [-19.48847395019814, 14.373405516937003, 4.980117410327855, 1.0]]
                    )
    s=time.perf_counter()
    res=bezier_curve_surface_intersect_certified(crv3, surf3,rational=True)
    print("case3", time.perf_counter() - s)
    gt={
      "isolated": [],
      "overlaps": [
        {
          "t_path": [
            0.13399306057079113,
            1.0
          ],
          "uv_path": [
            [
              0.0,
              0.3836870056884687
            ],
            [
              0.8660069394292091,
              0.38368700568846853
            ]
          ]
        }
      ]
    }

    make_rich_table(res, gt, "case3", "Rhino (8.26.25349.19002, 2025-12-15)")

    crv4 = np.array(
        [

            [-18.563074584230037, 14.877023362701413, 2.4979810139681967e-16, 1.0],
            [-19.206202529475306, 16.631067902045306, 6.242512273153852, 1.0],
            [-17.39336863037036, 19.148652421144444, 6.242512273153852, 1.0],
            [-15.858027551293814, 18.75748696680054, 2.4979810139681967e-16, 1.0],
        ]
    )

    s=time.perf_counter()
    res = bezier_curve_surface_intersect_certified(crv4, surf3, rational=True)
    print("case4", time.perf_counter() - s)
    gt={
      "isolated": [
        {
          "t": 0.45271538416199897,
          "u": 0.49759019852545344,
          "v": 0.5370783336000106
        },
        {
          "t": 0.536075965524697,
          "u": 0.4428618201092868,
          "v": 0.5301262860564949
        }
      ],
      "overlaps": []
    }

    make_rich_table(res, gt, "case4", "Rhino (8.26.25349.19002, 2025-12-15)")

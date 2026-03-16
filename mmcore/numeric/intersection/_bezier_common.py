"""Shared Bezier evaluation and utility functions for intersection algorithms.

Provides a single, deduplicated set of:
  - homogeneous detection / weight extraction / dehomogenization
  - curve and surface evaluation (point + first derivative)
  - LM-damped Newton solvers for CCX (curve-curve) and CSX (curve-surface)

All routines accept both polynomial (Nx3) and rational/homogeneous (Nx4)
control nets.  The *rational* flag controls interpretation:
  True  -> last column is weight (homogeneous)
  False -> Cartesian; a w=1 column is added internally before calling
           the low-level Cython evaluators.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Cython back-end (already compiled in the build)
# ---------------------------------------------------------------------------
from mmcore.numeric._bern_homog import (
    eval_bezier_homogeneous_curve_inplace,
    eval_bezier_curve_homog_with_derivs,
    project_curve_homog_to_cartesian,
    eval_bezier_surface_homog_with_derivs,
    project_surface_homog_to_cartesian,
    eval_bezier_homogeneous_surface,
    eval_bezier_homogeneous_curve,
)


# ---------------------------------------------------------------------------
# Homogeneous helpers
# ---------------------------------------------------------------------------

def is_homogeneous(ctrl, rational=None, eps: float = 1e-12) -> bool:
    """Detect or force whether a control net is rational (homogeneous).

    Parameters
    ----------
    ctrl : ndarray
        Control net — shape (..., D) where D is 3 (xyz) or 4 (xyzw).
    rational : bool or None
        If *True* / *False*, the answer is forced.  If *None* a heuristic
        is used: 4-column data is always rational; 3-column data is rational
        only when the last column deviates from 1.
    eps : float
        Weight deviation tolerance for the heuristic.
    """
    if rational is not None:
        return bool(rational)
    if ctrl is None:
        return False
    dim = ctrl.shape[-1]
    if dim >= 4:
        return True
    if dim == 3:
        w = np.asarray(ctrl[..., -1], dtype=np.float64)
        if np.all(w > eps) and np.max(np.abs(w - 1.0)) > eps:
            return True
    return False


def extract_weights(ctrl, rational: bool = True):
    """Split a control net into coordinates and weights.

    Parameters
    ----------
    ctrl : ndarray
        For a curve: shape (N, D).  For a surface: shape (M, N, D).
    rational : bool
        If *True* the last column is treated as the weight.  If *False*
        weights of 1.0 are synthesized.

    Returns
    -------
    xyz : ndarray
        Coordinate columns (last axis shortened by 1 when rational).
    weights : ndarray
        1-D (curve) or 2-D (surface) weight array.
    """
    ctrl = np.asarray(ctrl, dtype=np.float64)
    if rational:
        xyz = ctrl[..., :-1]
        weights = ctrl[..., -1]
    else:
        xyz = ctrl
        weights = np.ones(ctrl.shape[:-1], dtype=np.float64)
    return xyz, weights


def dehomogenize_ctrl(ctrl):
    """Divide xyz columns by the weight column (last axis)."""
    ctrl = np.asarray(ctrl, dtype=np.float64)
    w = ctrl[..., -1:]
    return ctrl[..., :-1] / w


# ---------------------------------------------------------------------------
# Internal: ensure contiguous float64 arrays for Cython
# ---------------------------------------------------------------------------

def _c2d(arr):
    """Return a C-contiguous float64 2-D array."""
    return np.ascontiguousarray(arr, dtype=np.float64)


def _c3d(arr):
    """Return a C-contiguous float64 3-D array."""
    return np.ascontiguousarray(arr, dtype=np.float64)


def _to_homog_curve(ctrl, rational: bool = False):
    """Ensure *ctrl* is a homogeneous (Nx(D+1)) 2-D array.

    If *rational* is True the data already has a weight column and is
    returned as-is (after dtype coercion).  Otherwise a column of ones
    is appended.
    """
    ctrl = _c2d(ctrl)
    if rational:
        return ctrl
    ones = np.ones((ctrl.shape[0], 1), dtype=np.float64)
    return np.ascontiguousarray(np.concatenate([ctrl, ones], axis=1))


def _to_homog_surface(ctrl, rational: bool = False):
    """Ensure *ctrl* is a homogeneous (MxNx(D+1)) 3-D array."""
    ctrl = _c3d(ctrl)
    if rational:
        return ctrl
    ones = np.ones(ctrl.shape[:-1] + (1,), dtype=np.float64)
    return np.ascontiguousarray(np.concatenate([ctrl, ones], axis=-1))


# ---------------------------------------------------------------------------
# Curve evaluation
# ---------------------------------------------------------------------------

def eval_curve(C, t: float, rational: bool = True):
    """Evaluate a Bezier curve at parameter *t* and return the Euclidean point.

    Parameters
    ----------
    C : ndarray (N, D)
        Control polygon.  D=4 when *rational* is True, D=3 otherwise.
    t : float
        Parameter in [0, 1].
    rational : bool
        Whether the control net is homogeneous (has a weight column).
    """
    Ph = _to_homog_curve(C, rational=rational)
    Ch = eval_bezier_homogeneous_curve(Ph, float(t))
    # dehomogenize: xyz / w
    return np.asarray(Ch[:-1] / Ch[-1], dtype=np.float64)


def eval_curve_d1(C, t: float, rational: bool = True):
    """Evaluate curve point and first derivative at *t*.

    Returns
    -------
    point : ndarray (D,)
    derivative : ndarray (D,)
    """
    Ph = _to_homog_curve(C, rational=rational)
    Ch, Chd = eval_bezier_curve_homog_with_derivs(Ph, float(t), want_second=False)
    pt, d1 = project_curve_homog_to_cartesian(Ch, Chd)
    return np.asarray(pt, dtype=np.float64), np.asarray(d1, dtype=np.float64)


# ---------------------------------------------------------------------------
# Surface evaluation
# ---------------------------------------------------------------------------

def eval_surface(S, u: float, v: float, rational: bool = True):
    """Evaluate a Bezier surface at (u, v) and return the Euclidean point."""
    Sh = _to_homog_surface(S, rational=rational)
    Sh0 = eval_bezier_homogeneous_surface(Sh, float(u), float(v))
    return np.asarray(Sh0[:-1] / Sh0[-1], dtype=np.float64)


def eval_surface_d1(S, u: float, v: float, rational: bool = True):
    """Evaluate surface point and first partial derivatives at (u, v).

    Returns
    -------
    point : ndarray (D,)
    du : ndarray (D,)   -- partial wrt u
    dv : ndarray (D,)   -- partial wrt v
    """
    Sh = _to_homog_surface(S, rational=rational)
    Sh0, Shu, Shv = eval_bezier_surface_homog_with_derivs(Sh, float(u), float(v), want_second=False)
    pt, du, dv = project_surface_homog_to_cartesian(Sh0, Shu, Shv)
    return (np.asarray(pt, dtype=np.float64),
            np.asarray(du, dtype=np.float64),
            np.asarray(dv, dtype=np.float64))


# ---------------------------------------------------------------------------
# Newton solvers
# ---------------------------------------------------------------------------

def _clamp01(x: float) -> float:
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    return x


def newton_ccx(
    C1, C2, u0: float, v0: float, *,
    rational: bool = False,
    tol: float = 1e-7,
    max_it: int = 20,
    lm_damp: float = 1e-12,
):
    """LM-damped Newton for G(u,v) = C1(u) - C2(v) = 0.

    Parameters
    ----------
    C1, C2 : ndarray
        Control polygons of the two Bezier curves.
    u0, v0 : float
        Initial parameter guesses.
    rational : bool
        Whether the control nets are homogeneous.
    tol : float
        Convergence tolerance on ||G|| (residual norm).
    max_it : int
        Maximum Newton iterations.
    lm_damp : float
        Levenberg-Marquardt damping factor.

    Returns
    -------
    u, v : float
        Converged parameters (clamped to [0, 1]).
    G : ndarray
        Final residual vector G(u, v).
    converged : bool
        True if ||G|| < tol.
    """
    u, v = float(u0), float(v0)

    for _ in range(max_it):
        p1, d1 = eval_curve_d1(C1, u, rational=rational)
        p2, d2 = eval_curve_d1(C2, v, rational=rational)
        G = p1 - p2
        g2 = float(np.dot(G, G))
        if g2 < tol * tol:
            break

        # Jacobian: J = [dC1/du | -dC2/dv], shape (D, 2)
        J = np.column_stack([d1, -d2])
        JT = J.T
        A = JT @ J + lm_damp * np.eye(2)
        b = -JT @ G
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            delta = np.zeros(2)

        # Backtracking line search with clamping
        step = 1.0
        for _ls in range(8):
            un = _clamp01(u + step * delta[0])
            vn = _clamp01(v + step * delta[1])
            Gn = eval_curve(C1, un, rational=rational) - eval_curve(C2, vn, rational=rational)
            if float(np.dot(Gn, Gn)) <= g2:
                u, v = un, vn
                break
            step *= 0.5
        else:
            # Line search exhausted without improvement — accept clamped step
            u = _clamp01(u + step * delta[0])
            v = _clamp01(v + step * delta[1])

    # Final residual
    G = eval_curve(C1, u, rational=rational) - eval_curve(C2, v, rational=rational)
    converged = float(np.linalg.norm(G)) < tol
    return u, v, G, converged


def newton_csx(
    C, S, t0: float, u0: float, v0: float, *,
    rational: bool = False,
    tol: float = 1e-7,
    max_it: int = 20,
    lm_damp: float = 1e-12,
):
    """LM-damped Newton for G(t,u,v) = C(t) - S(u,v) = 0.

    Parameters
    ----------
    C : ndarray
        Control polygon of the Bezier curve.
    S : ndarray
        Control net of the Bezier surface.
    t0, u0, v0 : float
        Initial parameter guesses.
    rational : bool
        Whether the control nets are homogeneous.
    tol : float
        Convergence tolerance on ||G|| (residual norm).
    max_it : int
        Maximum Newton iterations.
    lm_damp : float
        Levenberg-Marquardt damping factor.

    Returns
    -------
    t, u, v : float
        Converged parameters (clamped to [0, 1]).
    G : ndarray
        Final residual vector G(t, u, v).
    converged : bool
        True if ||G|| < tol.
    """
    t, u, v = float(t0), float(u0), float(v0)

    for _ in range(max_it):
        c_pt, c_d = eval_curve_d1(C, t, rational=rational)
        s_pt, s_du, s_dv = eval_surface_d1(S, u, v, rational=rational)
        G = c_pt - s_pt
        g2 = float(np.dot(G, G))
        if g2 < tol * tol:
            break

        # Jacobian: dG/d(t,u,v) = [dC/dt | -dS/du | -dS/dv], shape (D, 3)
        J = np.column_stack([c_d, -s_du, -s_dv])
        JT = J.T
        A = JT @ J + lm_damp * np.eye(3)
        b = -JT @ G
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            delta = np.zeros(3)

        # Backtracking line search with clamping
        step = 1.0
        for _ls in range(8):
            tn = _clamp01(t + step * delta[0])
            un = _clamp01(u + step * delta[1])
            vn = _clamp01(v + step * delta[2])
            Gn = (eval_curve(C, tn, rational=rational)
                  - eval_surface(S, un, vn, rational=rational))
            if float(np.dot(Gn, Gn)) <= g2:
                t, u, v = tn, un, vn
                break
            step *= 0.5
        else:
            t = _clamp01(t + step * delta[0])
            u = _clamp01(u + step * delta[1])
            v = _clamp01(v + step * delta[2])

    # Final residual
    G = eval_curve(C, t, rational=rational) - eval_surface(S, u, v, rational=rational)
    converged = float(np.linalg.norm(G)) < tol
    return t, u, v, G, converged

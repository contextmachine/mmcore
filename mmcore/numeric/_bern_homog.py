# mmcore/numeric/_bern_homog.py

from __future__ import annotations

from functools import lru_cache

import numpy as np
from mmcore.geom._nurbs_knots import generate_knots
from mmcore.geom.nurbs import basis_functions

# ---------------------------------------------------------------------------
# Bernstein basis (optimized)
# ---------------------------------------------------------------------------

def _readonly(a: np.ndarray) -> np.ndarray:
    """Return array marked read-only (safe for caching)."""
    a.setflags(write=False)
    return a
from mmcore.numeric import cbern

@lru_cache(maxsize=8192, typed=False)
def bernstein_basis(n: int, t: float) -> np.ndarray:
    """
    Bernstein basis B_i^n(t), i=0..n.

    Notes
    -----
    Implemented with a stable recurrence (no per-k binomial calls, no outer).
    Uses a forward recurrence for t <= 0.5 and a backward recurrence otherwise
    to avoid large ratios near endpoints.

    Returns a read-only float64 array suitable for caching.
    """

    return np.asarray(cbern.bernstein_basis(n,t))
    n = int(n)
    t = float(t)

    if n < 0:
        raise ValueError("n must be >= 0")

    if n == 0:
        return _readonly(np.array([1.0], dtype=np.float64))

    if t <= 0.0:
        B = np.zeros(n + 1, dtype=np.float64)
        B[0] = 1.0
        return _readonly(B)

    if t >= 1.0:
        B = np.zeros(n + 1, dtype=np.float64)
        B[n] = 1.0
        return _readonly(B)

    omt = 1.0 - t
    B = np.empty(n + 1, dtype=np.float64)

    # Choose direction for numerical stability near endpoints.
    if t <= 0.5:
        # Forward: start at i=0
        B[0] = omt**n
        r = t / omt
        for i in range(0, n):
            B[i + 1] = B[i] * (n - i) / (i + 1) * r
    else:
        # Backward: start at i=n
        B[n] = t**n
        r = omt / t
        for i in range(n, 0, -1):
            # B_{i-1} = B_i * (i/(n-i+1)) * ((1-t)/t)
            B[i - 1] = B[i] * i / (n - i + 1) * r

    return _readonly(B)


@lru_cache(maxsize=8192, typed=False)
def bernstein_basis_deriv(n: int, t: float) -> np.ndarray:
    """
    First derivative of Bernstein basis of degree n:
        d/dt B_i^n = n * (B_{i-1}^{n-1} - B_i^{n-1})
    with out-of-range terms treated as 0.

    Returns read-only float64 array of shape (n+1,).
    """
    return cbern.bernstein_basis_deriv(n, t)
    n = int(n)
    t = float(t)

    if n <= 0:
        return _readonly(np.zeros(n + 1, dtype=np.float64))

    B = bernstein_basis(n - 1, t)  # length n
    Bd = np.empty(n + 1, dtype=np.float64)

    # i = 0: n * (0 - B0)
    Bd[0] = -n * B[0]

    # i = 1..n-1: n * (B[i-1] - B[i])
    if n > 1:
        Bd[1:n] = n * (B[:-1] - B[1:])

    # i = n: n * (B[n-1] - 0)
    Bd[n] = n * B[-1]

    return _readonly(Bd)


@lru_cache(maxsize=8192, typed=False)
def bernstein_basis_2nd(n: int, t: float) -> np.ndarray:
    """
    Second derivative of Bernstein basis of degree n:
        d2/dt2 B_i^n = n(n-1) * (B_{i-2}^{n-2} - 2 B_{i-1}^{n-2} + B_i^{n-2})
    with out-of-range terms treated as 0.

    Returns read-only float64 array of shape (n+1,).
    """
    return cbern.bernstein_basis_2nd(n, t)
    n = int(n)
    t = float(t)

    if n <= 1:
        return _readonly(np.zeros(n + 1, dtype=np.float64))

    B = bernstein_basis(n - 2, t)  # length n-1
    out = np.zeros(n + 1, dtype=np.float64)

    # out[i] += B[i]      term: B_i^{n-2}
    out[:-2] += B
    # out[i] += -2 B[i-1] term: -2 B_{i-1}^{n-2}
    out[1:-1] -= 2.0 * B
    # out[i] += B[i-2]    term: B_{i-2}^{n-2}
    out[2:] += B

    out *= (n * (n - 1))
    return _readonly(out)


# Backward-compatible alias used by some helpers in this file
def bernstein_row(n: int, t: float) -> np.ndarray:
    return bernstein_basis(n, float(t))


# ---------------------------------------------------------------------------
# Homogeneous derivative nets (curve)
# ---------------------------------------------------------------------------

def elevate_derivative_net_homog(P4: np.ndarray, order: int = 2) -> list[np.ndarray]:
    """Return list [D0, D1, ..., D_order] of control nets in homogeneous space."""
    P4 = np.asarray(P4, dtype=float)
    nets = [P4.copy()]
    for _k in range(1, order + 1):
        prev = nets[-1]
        n = prev.shape[0] - 1
        if n <= 0:
            nets.append(prev[:1])
            continue
        d = n * (prev[1:] - prev[:-1])  # derivative net in homogeneous space
        nets.append(d)
    return nets


def eval_homog_derivatives(P4: np.ndarray, t: float, order: int = 2) -> list[np.ndarray]:
    """Evaluate position and derivatives in homogeneous space up to 'order'."""
    t = float(t)
    nets = elevate_derivative_net_homog(P4, order=order)
    vals = []
    for net in nets:
        n = net.shape[0] - 1
        B = bernstein_basis(n, t)
        vals.append(B @ net)  # (dim_h,) vector
    return vals


def dehomogenize_chain(Hvals: list[np.ndarray]) -> list[np.ndarray]:
    """Apply quotient rules to get Euclidean derivatives from homogeneous chain."""
    # Hvals[k] = [..., W] for k=0..m
    N = [v[:-1] for v in Hvals]
    W = [v[-1] for v in Hvals]

    pos = N[0] / W[0]
    out = [pos]

    if len(Hvals) >= 2:
        c1 = (N[1] * W[0] - N[0] * W[1]) / (W[0] ** 2)
        out.append(c1)

    if len(Hvals) >= 3:
        # Matches your original formula for 3D; generalized for N dims.
        num = (
            (N[2] * (W[0] ** 2))
            - 2.0 * N[1] * W[0] * W[1]
            + 2.0 * N[0] * (W[1] ** 2)
            - N[0] * W[2] * W[0]
        )
        den = W[0] ** 3
        out.append(num / den)

    return out


# ---------------------------------------------------------------------------
# Homogeneous evaluation (curve & surface) — corrected (no outer+tensordot)
# ---------------------------------------------------------------------------

def eval_bezier_homogeneous_curve(Pw: np.ndarray, t: float) -> np.ndarray:
    """
    Evaluate homogeneous Bézier curve at t.

    Pw : (n+1, d_h) homogeneous control points
    Returns: (d_h,)
    """
    Pw = np.asarray(Pw, dtype=float, order="C")
    n = Pw.shape[0] - 1
    B = bernstein_basis(n, float(t))
    return B @ Pw


def eval_bezier_homogeneous_surface(Pw: np.ndarray, u: float, v: float) -> np.ndarray:
    """
    Evaluate homogeneous tensor-product Bézier surface at (u,v).

    Pw : (nu+1, nv+1, d_h) homogeneous control net
    Returns: (d_h,)
    """
    Pw = np.asarray(Pw, dtype=float, order="C")
    nu = Pw.shape[0] - 1
    nv = Pw.shape[1] - 1
    d_h = Pw.shape[2]

    Bu = bernstein_basis(nu, float(u))
    Bv = bernstein_basis(nv, float(v))

    # Two-stage contraction (avoids outer + general tensordot)
    Pw2 = Pw.reshape(nu + 1, (nv + 1) * d_h)   # (nu+1, (nv+1)*d_h)
    tmp = Bu @ Pw2                              # ((nv+1)*d_h,)
    tmp = tmp.reshape(nv + 1, d_h)              # (nv+1, d_h)
    return Bv @ tmp                             # (d_h,)


# ---------------------------------------------------------------------------
# Homogeneous eval with derivatives (curve & surface)
# ---------------------------------------------------------------------------

def eval_bezier_curve_homog_with_derivs(Pw: np.ndarray, t: float, want_second: bool = True):
    """
    Pw: (n+1, d_h) homogeneous control points [Hx, Hy, Hz, Hw] (or any d_h)
    Returns:
        Ch  : (d_h,) homogeneous point
        Chd : (d_h,) first homogeneous derivative
        Ch2 : (d_h,) second homogeneous derivative (if want_second)
    """
    Pw = np.asarray(Pw, dtype=float, order="C")
    n = Pw.shape[0] - 1
    t = float(t)

    B = bernstein_basis(n, t)
    Bd = bernstein_basis_deriv(n, t)
    Ch = B @ Pw
    Chd = Bd @ Pw

    if not want_second:
        return Ch, Chd

    B2 = bernstein_basis_2nd(n, t)
    Ch2 = B2 @ Pw
    return Ch, Chd, Ch2


def project_curve_homog_to_cartesian(Ch: np.ndarray, Chd: np.ndarray, Ch2: np.ndarray | None = None):
    """
    Convert homogeneous curve point and derivatives to Cartesian using quotient rules.

    Assumes last component is w and all preceding are weighted coordinates.
    """
    Ch = np.asarray(Ch, dtype=float)
    Chd = np.asarray(Chd, dtype=float)

    H = Ch[:-1]
    w = Ch[-1]
    Hd = Chd[:-1]
    wd = Chd[-1]

    C = H / w
    Cp = (w * Hd - H * wd) / (w * w)

    if Ch2 is None:
        return C, Cp

    Ch2 = np.asarray(Ch2, dtype=float)
    H2 = Ch2[:-1]
    w2 = Ch2[-1]

    num = (w * w) * H2 - 2.0 * w * wd * Hd - w * w2 * H + 2.0 * (wd * wd) * H
    Cpp = num / (w ** 3)

    return C, Cp, Cpp


def eval_bezier_surface_homog_with_derivs(Pw: np.ndarray, u: float, v: float, want_second: bool = True):
    """
    Pw: (nu+1, nv+1, d_h) homogeneous control net

    Returns:
        Sh   : (d_h,) homogeneous point
        Shu  : (d_h,) first partial wrt u (homogeneous)
        Shv  : (d_h,) first partial wrt v (homogeneous)
        Shuu : (d_h,) second partial wrt u (if want_second)
        Shuv : (d_h,) mixed partial
        Shvv : (d_h,) second partial wrt v
    """
    Pw = np.asarray(Pw, dtype=float, order="C")
    u = float(u)
    v = float(v)

    nu = Pw.shape[0] - 1
    nv = Pw.shape[1] - 1
    d_h = Pw.shape[2]

    Bu = bernstein_basis(nu, u)
    Bud = bernstein_basis_deriv(nu, u)

    Bv = bernstein_basis(nv, v)
    Bvd = bernstein_basis_deriv(nv, v)

    # Reshape once; then reuse u-contractions for all v-derivatives
    Pw2 = Pw.reshape(nu + 1, (nv + 1) * d_h)  # (nu+1, (nv+1)*d_h)

    # u contractions
    tmp0 = (Bu @ Pw2).reshape(nv + 1, d_h)    # Σ_i Bu[i] * Pw[i, j, :]
    tmpu = (Bud @ Pw2).reshape(nv + 1, d_h)   # Σ_i Bud[i]* Pw[i, j, :]

    # v contractions (reuse tmp0/tmpu)
    Sh = Bv @ tmp0
    Shu = Bv @ tmpu
    Shv = Bvd @ tmp0

    if not want_second:
        return Sh, Shu, Shv

    Bu2 = bernstein_basis_2nd(nu, u)
    Bv2 = bernstein_basis_2nd(nv, v)

    tmpuu = (Bu2 @ Pw2).reshape(nv + 1, d_h)  # Σ_i Bu2[i]* Pw[i, j, :]

    Shuu = Bv @ tmpuu
    Shuv = Bvd @ tmpu
    Shvv = Bv2 @ tmp0

    return Sh, Shu, Shv, Shuu, Shuv, Shvv


def project_surface_homog_to_cartesian(
    Sh: np.ndarray,
    Shu: np.ndarray,
    Shv: np.ndarray,
    Shuu: np.ndarray | None = None,
    Shuv: np.ndarray | None = None,
    Shvv: np.ndarray | None = None,
):
    """
    Convert homogeneous surface point and derivatives to Cartesian using quotient rules.

    Assumes last component is w and all preceding are weighted coordinates.
    """
    Sh = np.asarray(Sh, dtype=float)
    Shu = np.asarray(Shu, dtype=float)
    Shv = np.asarray(Shv, dtype=float)

    H = Sh[:-1]
    w = Sh[-1]

    Hu, Hv = Shu[:-1], Shv[:-1]
    wu, wv = Shu[-1], Shv[-1]

    S = H / w
    Su = (w * Hu - H * wu) / (w * w)
    Sv = (w * Hv - H * wv) / (w * w)

    if Shuu is None:
        return S, Su, Sv

    Shuu = np.asarray(Shuu, dtype=float)
    Shuv = np.asarray(Shuv, dtype=float)
    Shvv = np.asarray(Shvv, dtype=float)

    Huu, Huv, Hvv = Shuu[:-1], Shuv[:-1], Shvv[:-1]
    wuu, wuv, wvv = Shuu[-1], Shuv[-1], Shvv[-1]

    denom = w ** 3

    Suu = (w * w * Huu - 2.0 * w * wu * Hu - w * wuu * H + 2.0 * (wu * wu) * H) / denom
    Svv = (w * w * Hvv - 2.0 * w * wv * Hv - w * wvv * H + 2.0 * (wv * wv) * H) / denom
    Suv = (w * w * Huv - w * (wu * Hv + wv * Hu) - w * wuv * H + 2.0 * wu * wv * H) / denom

    return S, Su, Sv, Suu, Suv, Svv


def project_2nd(X, Xs, Xt, Xst):
    """
    Second derivative projection (s,t in {u,v} or t,t for curve).
    X, Xs, Xt, Xst are (4,)
    returns (3,)
    """
    X3, W = X[:3], X[3]
    x = X3 / W
    Xs3, Ws = Xs[:3], Xs[3]
    Xt3, Wt = Xt[:3], Xt[3]
    Xst3, Wst = Xst[:3], Xst[3]

    # Pi * Xst
    term1 = (Xst3 - x * Wst) / W
    # rank-1 correction: (1/W) * [[0.. -x_s],[..],[..]] * Xt  ->  -(x_s * Wt)/W
    xs = (Xs3 - x * Ws) / W
    return term1 - xs * (Wt / W)

def jac_curve_surface(Xc, Xct, Xs, Xsu, Xsv):
    """
    Build 3x3 Jacobian for F = c(t) - s(u,v).
    Inputs are homogeneous 4-vectors at (t,u,v):
      Xc: curve point, Xct: curve t-deriv
      Xs: surface point, Xsu: u-deriv, Xsv: v-deriv
    """
    ct = project_deriv(Xc, Xct)
    su = project_deriv(Xs, Xsu)
    sv = project_deriv(Xs, Xsv)
    # columns: [c_t, -s_u, -s_v]
    J = np.column_stack([ct, -su, -sv])
    # residual in Euclidean space (optional for Newton step)
    c = Xc[:3]/Xc[3]
    s = Xs[:3]/Xs[3]
    F = c - s
    return F, J
import numpy as np

def newton_curve_surface_intersection(
    Cw, Sw,
    t0=0.5, u0=0.5, v0=0.5,
    max_iter=30,
    tol_F=1e-10,
    tol_step=1e-10,
    step_damping=1.0,   # you can set <1.0 for damping (e.g. 0.5)
    clamp_params=True   # clamp t,u,v to [0,1] after each step
):
    """
    Solve c(t) = s(u,v) using Newton in (t,u,v).

    Cw: (nc+1, 4)   homogeneous Bezier control points of the curve
    Sw: (nu+1, nv+1, 4) homogeneous Bezier control net of the surface
    t0,u0,v0: initial guess in parameter space
    Returns:
        success (bool),
        (t,u,v),
        c_point (3,),
        s_point (3,),
        iterations (int)
    """
    t, u, v = float(t0), float(u0), float(v0)

    for it in range(max_iter):
        # 1) Evaluate homogeneous curve & surface + first derivatives
        Ch, Chd = eval_bezier_curve_homog_with_derivs(Cw, t, want_second=False)
        Sh, Shu, Shv = eval_bezier_surface_homog_with_derivs(Sw, u, v, want_second=False)

        # 2) Build residual F and Jacobian J (3x3) in Cartesian space
        F, J = jac_curve_surface(Ch, Chd, Sh, Shu, Shv)

        # 3) Check residual norm
        normF = np.linalg.norm(F, ord=2)
        if normF < tol_F:
            # Converged: decode final points
            c = Ch[:3] / Ch[3]
            s = Sh[:3] / Sh[3]
            return True, (t, u, v), c, s, it

        # 4) Solve J * step = -F
        try:
            step = np.linalg.solve(J, -F)
        except np.linalg.LinAlgError:
            # Singular or ill-conditioned J
            return False, (t, u, v), None, None, it

        # Optional: damping
        step *= step_damping

        # 5) Check step size
        step_norm = np.linalg.norm(step, ord=2)
        if step_norm < tol_step:
            c = Ch[:3] / Ch[3]
            s = Sh[:3] / Sh[3]
            return True, (t, u, v), c, s, it

        # 6) Update parameters
        dt, du, dv = step
        t += dt
        u += du
        v += dv

        # 7) Optionally clamp parameters to [0,1]
        if clamp_params:
            t = max(0.0, min(1.0, t))
            u = max(0.0, min(1.0, u))
            v = max(0.0, min(1.0, v))

    # If we exit the loop, we did not converge within max_iter
    Ch, _ = eval_bezier_curve_homog_with_derivs(Cw, t, want_second=False)
    Sh, _, _ = eval_bezier_surface_homog_with_derivs(Sw, u, v, want_second=False)
    c = Ch[:3] / Ch[3]
    s = Sh[:3] / Sh[3]
    return False, (t, u, v), c, s, max_iter

if __name__=="__main__":
    C = np.random.random((4, 4))
    S = np.random.random((5, 5, 4))
    F,J=jac_curve_surface(*eval_bezier_curve_homog_with_derivs(C, 0.5, want_second=False),
                      *eval_bezier_surface_homog_with_derivs(S, 0.5, 0.5, want_second=False))
    success, (t,u,v), c_pt, s_pt, iters = newton_curve_surface_intersection(
        C, S,
        t0=0.5, u0=0.5, v0=0.5,
        max_iter=30
    )

    print("Success:", success)
    print("Iterations:", iters)
    print("Params (t,u,v):", t, u, v)
    print("Curve point:", c_pt)
    print("Surface point:", s_pt)
    print("Residual norm:", np.linalg.norm(c_pt - s_pt))

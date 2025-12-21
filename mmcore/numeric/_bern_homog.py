from functools import lru_cache
from math import comb

import numpy as np

from mmcore.numeric.binom import binomial_coefficient_py
import numpy as np

def bernstein_row(n, t):
    i = np.arange(n+1)
    from math import comb
    B = np.array([comb(n, k) for k in i], dtype=float)
    return B * (t**i) * ((1-t)**(n-i))

def elevate_derivative_net_homog(P4, order=2):
    """Return list [D0, D1, ..., D_order] of control nets in 4D (homog)."""
    nets = [P4.copy()]
    for k in range(1, order+1):
        prev = nets[-1]
        n = prev.shape[0] - 1
        if n <= 0: nets.append(prev[:1]); continue
        d = n * (prev[1:] - prev[:-1])  # polynomial Bernstein derivative in 4D
        nets.append(d)
    return nets

def eval_homog_derivatives(P4, t, order=2):
    """Evaluate position and derivatives in homogeneous space up to 'order'."""
    nets = elevate_derivative_net_homog(P4, order=order)
    vals = []
    for k, net in enumerate(nets):
        n = net.shape[0] - 1
        B = bernstein_row(n, t)
        vals.append(B @ net)  # (4,) vector: [Nx, Ny, Nz, W]
    return vals  # [C_H, C_H', C_H'', ...]

def dehomogenize_chain(Hvals):
    """Apply exact quotient rules to get Euclidean derivatives from homogeneous chain."""
    # Hvals[k] = [Nx,Ny,Nz,W] for k=0..m
    N = [v[:3] for v in Hvals]
    W = [v[3]   for v in Hvals]
    pos = N[0] / W[0]
    out = [pos]
    if len(Hvals) >= 2:
        c1 = (N[1]*W[0] - N[0]*W[1]) / (W[0]**2)
        out.append(c1)
    if len(Hvals) >= 3:
        num = (N[2]*(W[0]**2)
               - 2*N[1]*W[0]*W[1]
               + 2*N[0]*(W[1]**2)
               - N[0]*W[2]*W[0])
        den = W[0]**3
        out.append(num/den)
    return out  # [c, c', c'']
from mmcore.geom.nurbs import basis_functions
# Example usage:
# P4: (n+1, 4) homogeneous control points, e.g., each row = [w*x, w*y, w*z, w]
# c, c1, c2 = dehomogenize_chain(eval_homog_derivatives(P4, t=0.37, order=2))


# ---------- Bernstein utilities ----------
@lru_cache(maxsize=None, typed=False)
def bernstein_basis(n, u):
    i = np.arange(n+1)
    # stable eval via log-space could be added; this is fine for moderate n

    B = np.array([binomial_coefficient_py(n, k)*(u**k)*((1-u)**(n-k)) for k in i], dtype=np.float64)
    return B  # shape (n+1,)
@lru_cache(maxsize=None, typed=False)
def bernstein_basis_deriv(n, t):
    # d/dt B_i^n = n( B_{i-1}^{n-1} - B_i^{n-1} )
    if n == 0:
        return np.zeros(1)
    B = bernstein_basis(n-1, t)
    # pad for shifted indexes
    left  = np.r_[0.0, B]
    right = np.r_[B, 0.0]
    return n * (left - right)

@lru_cache(maxsize=None, typed=False)
def bernstein_basis_2nd(n, t):
    # d2/dt2 B_i^n = n(n-1)( B_{i-2}^{n-2} - 2 B_{i-1}^{n-2} + B_i^{n-2} )
    if n <= 1:
        return np.zeros(n+1)
    B = bernstein_basis(n-2, t)
    a = np.r_[0.0, 0.0, B]
    b = np.r_[0.0, B, 0.0]
    c = np.r_[B, 0.0, 0.0]
    return n*(n-1)*(a - 2*b + c)
def eval_bezier_homogeneous_curve(Pw, t):
    B = bernstein_basis(len(Pw)-1, t)
    H = Pw.T @ B
    return H  # shape (4,)
def eval_bezier_homogeneous_surface(Pw, u, v):
    Bu = bernstein_basis(Pw.shape[0]-1, u)
    Bv = bernstein_basis(Pw.shape[1]-1, v)
    B  = np.outer(Bu, Bv)
    H  = np.tensordot(Pw, B, axes=([0,1],[0,1]))  # shape (4,)
    return H

def eval_bezier_curve_homog_with_derivs(Pw, t, want_second=True):
    """
    Pw: (n+1, 4) homogeneous control points [Hx, Hy, Hz, Hw]
    Returns:
        Ch  : (4,)  homogeneous point
        Chd : (4,)  first homogeneous derivative
        Ch2 : (4,)  second homogeneous derivative (if want_second)
    """
    Pw = np.asarray(Pw, dtype=float)
    n = Pw.shape[0] - 1

    B   = bernstein_basis(n, t)         # shape (n+1,)
    Bd  = bernstein_basis_deriv(n, t)   # shape (n+1,)
    B2  = bernstein_basis_2nd(n, t) if want_second else None

    Ch  = Pw.T @ B          # (4,)
    Chd = Pw.T @ Bd         # (4,)
    Ch2 = Pw.T @ B2 if want_second else None

    return (Ch, Chd, Ch2) if want_second else (Ch, Chd)
def project_curve_homog_to_cartesian(Ch, Chd, Ch2=None):
    H   = Ch[:3]
    w   = Ch[3]
    Hd  = Chd[:3]
    wd  = Chd[3]

    C  = H / w
    Cp = (w * Hd - H * wd) / (w * w)

    if Ch2 is None:
        return C, Cp

    H2  = Ch2[:3]
    w2  = Ch2[3]
    num = (w*w)*H2 - 2*w*wd*Hd - w*w2*H + 2*(wd*wd)*H
    Cpp = num / (w**3)

    return C, Cp, Cpp
def eval_bezier_surface_homog_with_derivs(Pw, u, v, want_second=True):
    """
    Pw: (nu+1, nv+1, 4) homogeneous control net [Hx, Hy, Hz, Hw]

    Returns:
        Sh   : (4,)   homogeneous point
        Shu  : (4,)   first partial wrt u (homogeneous)
        Shv  : (4,)   first partial wrt v (homogeneous)
        Shuu : (4,)   second partial wrt u (if want_second)
        Shuv : (4,)   mixed partial
        Shvv : (4,)   second partial wrt v
    """
    Pw = np.asarray(Pw, dtype=float)
    nu, nv = Pw.shape[0]-1, Pw.shape[1]-1

    Bu   = bernstein_basis(nu, u)
    Bud  = bernstein_basis_deriv(nu, u)
    Bu2  = bernstein_basis_2nd(nu, u) if want_second else None

    Bv   = bernstein_basis(nv, v)
    Bvd  = bernstein_basis_deriv(nv, v)
    Bv2  = bernstein_basis_2nd(nv, v) if want_second else None

    # tensor products for each derivative order
    B       = np.outer(Bu,  Bv)       # (0,0)
    BuBv    = np.outer(Bud, Bv)       # (1,0)
    BuBv_v  = np.outer(Bu,  Bvd)      # (0,1)

    Sh   = np.tensordot(Pw, B,      axes=([0,1],[0,1]))  # (4,)
    Shu  = np.tensordot(Pw, BuBv,   axes=([0,1],[0,1]))  # (4,)
    Shv  = np.tensordot(Pw, BuBv_v, axes=([0,1],[0,1]))  # (4,)

    if not want_second:
        return Sh, Shu, Shv

    Bu2Bv   = np.outer(Bu2, Bv)      # (2,0)
    BuBv2   = np.outer(Bu,  Bv2)     # (0,2)
    Bu1v1   = np.outer(Bud, Bvd)     # (1,1)

    Shuu = np.tensordot(Pw, Bu2Bv, axes=([0,1],[0,1]))
    Shvv = np.tensordot(Pw, BuBv2, axes=([0,1],[0,1]))
    Shuv = np.tensordot(Pw, Bu1v1,  axes=([0,1],[0,1]))

    return Sh, Shu, Shv, Shuu, Shuv, Shvv
def project_surface_homog_to_cartesian(Sh, Shu, Shv, Shuu=None, Shuv=None, Shvv=None):
    H   = Sh[:3]
    w   = Sh[3]

    Hu, Hv = Shu[:3], Shv[:3]
    wu, wv = Shu[3],  Shv[3]

    S  = H / w
    Su = (w * Hu - H * wu) / (w * w)
    Sv = (w * Hv - H * wv) / (w * w)

    if Shuu is None:
        return S, Su, Sv

    Huu, Huv, Hvv = Shuu[:3], Shuv[:3], Shvv[:3]
    wuu, wuv, wvv = Shuu[3],  Shuv[3],  Shvv[3]

    denom = w**3

    Suu = (w*w*Huu - 2*w*wu*Hu - w*wuu*H + 2*(wu*wu)*H) / denom
    Svv = (w*w*Hvv - 2*w*wv*Hv - w*wvv*H + 2*(wv*wv)*H) / denom
    Suv = (w*w*Huv - w*(wu*Hv + wv*Hu) - w*wuv*H + 2*wu*wv*H) / denom

    return S, Su, Sv, Suu, Suv, Svv

import numpy as np

def project_deriv(X, Xt):
    """
    Homogeneous projection via Carvchuk-style linearization.
    X: (4,) -> [X,Y,Z,W]
    Xt: (4,) directional derivative (u or v or t)
    returns (3,) Euclidean derivative
    """
    X3, W = X[:3], X[3]
    x = X3 / W
    Xt3, Wt = Xt[:3], Xt[3]
    # Pi * Xt  == (Xt3 - x * Wt) / W
    return (Xt3 - x * Wt) / W

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

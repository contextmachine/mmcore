import traceback

import numpy as np
from itertools import combinations

from mmcore.numeric.bern import bern_eval
from mmcore.numeric.ndinterval import interval, get_iarray,interval
# @Py

from math import comb
def vadd(a, b):
    return (a[0]+b[0], a[1]+b[1], a[2]+b[2])
def vsub(a, b):
    return (a[0]-b[0], a[1]-b[1], a[2]-b[2])
def vscale(s, a):
    return (s*a[0], s*a[1], s*a[2])
def cross3(a, b):
    ax, ay, az = a
    bx, by, bz = b
    return (
        ay * bz - az * by,
        az * bx - ax * bz,
        ax * by - ay * bx
    )
def bernstein_patch_derivative_s(P):
    """
    Derivative of Bernstein patch P(s,t) wrt s.
    P: (m+1) x (n+1) control net (3D vectors).
    Returns: (m) x (n+1) control net of degree (m-1, n).
    """
    m = len(P) - 1
    n = len(P[0]) - 1
    Du = [[None for _ in range(n+1)] for _ in range(m)]
    for i in range(m):
        for j in range(n+1):
            Du[i][j] = vscale(m, vsub(P[i+1][j], P[i][j]))
    return Du  # degree (m-1, n)
def bernstein_patch_derivative_t(P):
    """
    Derivative of Bernstein patch P(s,t) wrt t.
    P: (m+1) x (n+1) control net (3D vectors).
    Returns: (m+1) x (n) control net of degree (m, n-1).
    """
    m = len(P) - 1
    n = len(P[0]) - 1
    Dv = [[None for _ in range(n)] for _ in range(m+1)]
    for i in range(m+1):
        for j in range(n):
            Dv[i][j] = vscale(n, vsub(P[i][j+1], P[i][j]))
    return Dv  # degree (m, n-1)
def bernstein_patch_cross_same_params(P, Q):
    """
    Cross product of two Bernstein patches sharing the same (u,v).
    P: (m_u+1) x (m_v+1) control net, degree (m_u, m_v)
    Q: (n_u+1) x (n_v+1) control net, degree (n_u, n_v)
    Returns:
        R: control net of degree (m_u+n_u, m_v+n_v)
    """
    m_u = len(P)    - 1
    m_v = len(P[0]) - 1
    n_u = len(Q)    - 1
    n_v = len(Q[0]) - 1
    deg_u = m_u + n_u
    deg_v = m_v + n_v
    R = [[[0.0, 0.0, 0.0] for _ in range(deg_v + 1)]
         for _ in range(deg_u + 1)]
    for alpha in range(deg_u + 1):
        denom_u = comb(deg_u, alpha)
        for beta in range(deg_v + 1):
            denom_v = comb(deg_v, beta)
            i_min = max(0, alpha - n_u)
            i_max = min(m_u, alpha)
            for i in range(i_min, i_max + 1):
                k = alpha - i
                j_min = max(0, beta - n_v)
                j_max = min(m_v, beta)
                for j in range(j_min, j_max + 1):
                    l = beta - j
                    cu = comb(m_u, i) * comb(n_u, k) / denom_u
                    cv = comb(m_v, j) * comb(n_v, l) / denom_v
                    coeff = cu * cv
                    cx, cy, cz = cross3(P[i][j], Q[k][l])
                    R[alpha][beta][0] += coeff * cx
                    R[alpha][beta][1] += coeff * cy
                    R[alpha][beta][2] += coeff * cz
    return [ [tuple(v) for v in row] for row in R ]
def build_4d_cross_patch(N1, N2):
    """
    Given normal patches:
      N1(s,t) with degrees (r_s, r_t), control net size (r_s+1) x (r_t+1)
      N2(u,v) with degrees (s_u, s_v), control net size (s_u+1) x (s_v+1)
    build 4D control net F[a][b][c][d] for F(s,t,u,v) = N1 x N2.
    Returns:
        F: nested list [a][b][c][d] of 3D vectors.
        degrees: (r_s, r_t, s_u, s_v)
    """
    rs = len(N1)    - 1
    rt = len(N1[0]) - 1
    su = len(N2)    - 1
    sv = len(N2[0]) - 1
    F = [[[[None for _ in range(sv+1)]
           for _ in range(su+1)]
          for _ in range(rt+1)]
         for _ in range(rs+1)]
    for a in range(rs+1):
        for b in range(rt+1):
            for c in range(su+1):
                for d in range(sv+1):
                    F[a][b][c][d] = cross3(N1[a][b], N2[c][d])
    return F, (rs, rt, su, sv)
def Tangent_4d_patch(P1, P2):
    """
    P1: (m1+1) x (n1+1) control net of S1(s,t)
    P2: (p+1)  x (q+1)  control net of S2(u,v)
    Returns:
        F: 4D control net F[a][b][c][d] for
           F(s,t,u,v) = N1(s,t) x N2(u,v),

        degrees: (deg_s, deg_t, deg_u, deg_v)
    """
    # First surface
    P1_s = bernstein_patch_derivative_s(P1)
    P1_t = bernstein_patch_derivative_t(P1)
    N1   = bernstein_patch_cross_same_params(P1_s, P1_t)
    # Second surface
    P2_u = bernstein_patch_derivative_s(P2)
    P2_v = bernstein_patch_derivative_t(P2)
    N2   = bernstein_patch_cross_same_params(P2_u, P2_v)
    # 4D patch for cross of normals
    F, degrees = build_4d_cross_patch(N1, N2)
    return F, degrees
def dot3(a, b):
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2]

def outer_dot_4d(Pst, Quv):
    """
    Pst: (ms+1)x(nt+1) control net (3D vectors) in (s,t)
    Quv: (pu+1)x(qv+1) control net (3D vectors) in (u,v)
    Returns scalar 4D control net S[i][j][k][l] for Pst(s,t)·Quv(u,v).
    Degree: (ms, nt, pu, qv)
    """
    ms = len(Pst)    - 1
    nt = len(Pst[0]) - 1
    pu = len(Quv)    - 1
    qv = len(Quv[0]) - 1

    S = [[[[None for _ in range(qv+1)]
                for _ in range(pu+1)]
                for _ in range(nt+1)]
                for _ in range(ms+1)]
    for i in range(ms+1):
        for j in range(nt+1):
            for k in range(pu+1):
                for l in range(qv+1):
                    S[i][j][k][l] = dot3(Pst[i][j], Quv[k][l])
    return S

def negate_4d_scalar(S):
    ms = len(S) - 1
    nt = len(S[0]) - 1
    pu = len(S[0][0]) - 1
    qv = len(S[0][0][0]) - 1
    return [[[[ -S[i][j][k][l] for l in range(qv+1)]
                        for k in range(pu+1)]
                        for j in range(nt+1)]
                        for i in range(ms+1)]

def minors_Tpsi_from_control_nets(P1, P2):
    # first derivatives
    A = bernstein_patch_derivative_s(P1)  # R1_s
    B = bernstein_patch_derivative_t(P1)  # R1_t
    C = bernstein_patch_derivative_s(P2)  # R2_u
    D = bernstein_patch_derivative_t(P2)  # R2_v

    # normals
    N1 = bernstein_patch_cross_same_params(A, B)  # N1 = R1_s x R1_t
    N2 = bernstein_patch_cross_same_params(C, D)  # N2 = R2_u x R2_v

    # minor determinants (scalar 4D Bernstein nets)
    T1 = outer_dot_4d(B,  N2)                 #  R1_t · N2
    T2 = outer_dot_4d(A,  N2)                 #  R1_s · N2
    T3 = negate_4d_scalar(outer_dot_4d(N1, D))# -R2_v · N1
    T4 = negate_4d_scalar(outer_dot_4d(N1, C))# -R2_u · N1

    return T1, T2, T3, T4

import numpy as np
from dataclasses import dataclass
from itertools import combinations

# ------------------------------------------------------------
# Interval helpers (works with many "interval scalar" types)
# ------------------------------------------------------------

def _iv_bounds(x):
    """
    Convert an interval-like scalar to (lo,hi) floats.
    Supports:
      - objects with [0],[1]
      - objects with .lo/.hi
      - objects with .inf/.sup
      - plain floats/ints (degenerate interval)
    """
    if isinstance(x, (float, int, np.floating, np.integer)):
        v = float(x )
        return (v, v)
    # numpy scalar of custom dtype sometimes supports x[0], x[1]
    elif isinstance(x,interval):
        return x.l,x.u
    try:
        lo = x[0]; hi = x[1]
        return (float(lo), float(hi))
    except Exception:
        pass
    for a,b in (("lo","hi"), ("inf","sup"), ("lower","upper")):
        if hasattr(x, a) and hasattr(x, b):
            return (float(getattr(x,a)), float(getattr(x,b)))
    # last resort: try casting
    v = float(x)
    return (v, v)

def _iv_mid(I):
    lo, hi = I
    return 0.5*(lo+hi)

def _iv_wid(I):
    lo, hi = I
    return hi-lo

def _iv_contains0(I):
    lo, hi = I
    return (lo <= 0.0 <= hi)

def _iv_intersect(I, J):
    lo = max(I[0], J[0])
    hi = min(I[1], J[1])
    if lo > hi:
        return None
    return (lo, hi)

def _iv_add(I,J):  return (I[0]+J[0], I[1]+J[1])
def _iv_sub(I,J):  return (I[0]-J[1], I[1]-J[0])

def _iv_mul(I,J):
    a,b = I; c,d = J
    prods = [a*c, a*d, b*c, b*d]
    return (min(prods), max(prods))

def _iv_scale(alpha, I):
    lo, hi = I
    if alpha >= 0:
        return (alpha*lo, alpha*hi)
    else:
        return (alpha*hi, alpha*lo)

def _iv_is_subset(I, J, strict=False, eps=0.0):
    if strict:
        return (J[0] + eps < I[0]) and (I[1] < J[1] - eps)
    else:
        return (J[0] - eps <= I[0]) and (I[1] <= J[1] + eps)

# ------------------------------------------------------------
# Box helpers: internal boxes are float bounds ((lo,hi),...).
# ------------------------------------------------------------

def _box_from_any(B):
    # B can be tuples of intervals or tuples of (lo,hi)
    out = []
    for b in B:
        if isinstance(b, (tuple, list)) and len(b)==2 and all(isinstance(z,(int,float,np.floating,np.integer)) for z in b):
            out.append((float(b[0]), float(b[1])))
        else:
            out.append(_iv_bounds(b))
    return tuple(out)

def _box_mid(B):
    return np.array([_iv_mid(b) for b in B], dtype=float)

def _box_widest_axis(B):
    widths = [_iv_wid(b) for b in B]
    return int(np.argmax(widths))

def _box_split(B, axis=None):
    if axis is None:
        axis = _box_widest_axis(B)
    lo, hi = B[axis]
    mid = 0.5*(lo+hi)
    B1 = list(B); B2 = list(B)
    B1[axis] = (lo, mid)
    B2[axis] = (mid, hi)
    return tuple(B1), tuple(B2)

def _box_max_width(B):
    return max(_iv_wid(b) for b in B)

def _box_to_interval_params(B, interval_ctor):
    return tuple(interval_ctor(lo,hi) for (lo,hi) in B)

# ------------------------------------------------------------
# Bernstein eval wrappers
# ------------------------------------------------------------

def _beval_scalar(net, params, bern_eval):
    arr = bern_eval(net, params)
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    return arr.flat[0]

def _beval_vec3(net, params, bern_eval):
    arr = bern_eval(net, params)
    if not isinstance(arr, np.ndarray):
        arr = np.asarray(arr)
    if arr.shape[-1] != 3:
        raise ValueError("Expected a vec3 Bernstein net with last axis length 3.")
    if arr.ndim == 1:
        return arr
    return arr.reshape(-1,3)[0]

def _to_float_scalar(x):
    return _iv_mid(_iv_bounds(x))

def _to_float_vec3(v):
    return np.array([_to_float_scalar(v[0]), _to_float_scalar(v[1]), _to_float_scalar(v[2])], dtype=float)

def _net_midpoint_float(net):
    arr = np.asarray(net)
    if np.issubdtype(arr.dtype, np.floating):
        return arr.astype(float, copy=False)
    out = np.empty(arr.shape, dtype=float)
    it = np.nditer(arr, flags=["refs_ok", "multi_index"], op_flags=["readonly"])
    for x in it:
        out[it.multi_index] = _to_float_scalar(x.item())
    return out

# ------------------------------------------------------------
# Bézier derivatives for surfaces (tensor-product)
# P: (m+1,n+1,3)
# ------------------------------------------------------------

def bezier_du(P):
    P = np.asarray(P)
    m = P.shape[0]-1
    if m <= 0:
        # degree 0 => derivative 0 patch
        return np.zeros((1, P.shape[1], 3), dtype=P.dtype)
    return m*(P[1:,:,:] - P[:-1,:,:])

def bezier_dv(P):
    P = np.asarray(P)
    n = P.shape[1]-1
    if n <= 0:
        return np.zeros((P.shape[0], 1, 3), dtype=P.dtype)
    return n*(P[:,1:,:] - P[:,:-1,:])

# ------------------------------------------------------------
# ND Bernstein derivative for scalar (or vector) nets
# ------------------------------------------------------------

def bernstein_derivative_nd(net, axis):
    net = np.asarray(net)
    d = net.shape[axis]-1
    if d <= 0:
        return None  # identically 0
    return d*np.diff(net, axis=axis)

# ------------------------------------------------------------
# The deflated system Δ_B = Ψ ∪ {T1..T4} evaluators
# ------------------------------------------------------------

@dataclass
class DeflatedSystem:
    P1: np.ndarray
    P2: np.ndarray
    T:  tuple  # (T1,T2,T3,T4)
    bern_eval: callable
    interval_ctor: callable

    def __post_init__(self):
        # Surface Jacobian pieces
        self.P1_s = bezier_du(self.P1)
        self.P1_t = bezier_dv(self.P1)
        self.P2_u = bezier_du(self.P2)
        self.P2_v = bezier_dv(self.P2)
        # Midpoint float nets for point evaluations.
        self.P1_point = _net_midpoint_float(self.P1)
        self.P2_point = _net_midpoint_float(self.P2)
        self.P1_s_point = bezier_du(self.P1_point)
        self.P1_t_point = bezier_dv(self.P1_point)
        self.P2_u_point = bezier_du(self.P2_point)
        self.P2_v_point = bezier_dv(self.P2_point)
        self.T_point_nets = tuple(_net_midpoint_float(Ti) for Ti in self.T)

        # Gradients of Ti (4D scalar nets)
        self.dT = []  # list of (d/ds, d/dt, d/du, d/dv) nets or None
        self.dT_point = []
        for Ti in self.T:
            d0 = bernstein_derivative_nd(Ti, axis=0)
            d1 = bernstein_derivative_nd(Ti, axis=1)
            d2 = bernstein_derivative_nd(Ti, axis=2)
            d3 = bernstein_derivative_nd(Ti, axis=3)
            self.dT.append((d0,d1,d2,d3))
        for Ti in self.T_point_nets:
            d0 = bernstein_derivative_nd(Ti, axis=0)
            d1 = bernstein_derivative_nd(Ti, axis=1)
            d2 = bernstein_derivative_nd(Ti, axis=2)
            d3 = bernstein_derivative_nd(Ti, axis=3)
            self.dT_point.append((d0,d1,d2,d3))

    @staticmethod
    def _ordered_unique(rows):
        seen = set()
        out = []
        for r in rows:
            if r not in seen:
                seen.add(r)
                out.append(r)
        return out

    # ---- Ψ evaluation

    def psi_point(self, x):  # x float[4]
        s,t,u,v = x
        r1 = _beval_vec3(self.P1_point, (s,t), self.bern_eval)
        r2 = _beval_vec3(self.P2_point, (u,v), self.bern_eval)
        return np.asarray(r1 - r2, dtype=float)

    def psi_box(self, B):  # B float bounds
        sI,tI,uI,vI = _box_to_interval_params(B, self.interval_ctor)
        r1 = _beval_vec3(self.P1, (sI,tI), self.bern_eval)
        r2 = _beval_vec3(self.P2, (uI,vI), self.bern_eval)
        out = []
        for k in range(3):
            I1 = _iv_bounds(r1[k])
            I2 = _iv_bounds(r2[k])
            out.append(_iv_sub(I1,I2))
        return out  # 3 intervals

    # ---- T evaluation

    def T_point(self, x):  # returns 4 floats
        s,t,u,v = x
        vals = []
        for Ti in self.T_point_nets:
            val = _beval_scalar(Ti, (s,t,u,v), self.bern_eval)
            vals.append(float(val))
        return np.array(vals, dtype=float)

    def T_box(self, B):  # returns 4 intervals
        params = _box_to_interval_params(B, self.interval_ctor)
        out=[]
        for Ti in self.T:
            out.append(_iv_bounds(_beval_scalar(Ti, params, self.bern_eval)))
        return out

    # ---- Full Δ evaluation

    def delta_point(self, x):
        return self.delta_rows_point(x, (0,1,2,3,4,5,6))

    def delta_box(self, B):
        return self.delta_rows_box(B, (0,1,2,3,4,5,6))

    def delta_rows_point(self, x, rows):
        rows = tuple(rows)
        out = np.zeros((len(rows),), dtype=float)
        need_psi = any(idx < 3 for idx in rows)
        psi = self.psi_point(x) if need_psi else None
        s,t,u,v = x
        tvals = {}
        for idx in self._ordered_unique(rows):
            if idx >= 3:
                Ti_i = idx - 3
                val = _beval_scalar(self.T_point_nets[Ti_i], (s,t,u,v), self.bern_eval)
                tvals[Ti_i] = float(val)
        for i, idx in enumerate(rows):
            out[i] = psi[idx] if idx < 3 else tvals[idx - 3]
        return out

    def delta_rows_box(self, B, rows):
        rows = tuple(rows)
        out = [None] * len(rows)
        sI,tI,uI,vI = _box_to_interval_params(B, self.interval_ctor)
        params = (sI,tI,uI,vI)

        psi_map = {}
        if any(idx < 3 for idx in rows):
            r1 = _beval_vec3(self.P1, (sI,tI), self.bern_eval)
            r2 = _beval_vec3(self.P2, (uI,vI), self.bern_eval)
            for k in self._ordered_unique([idx for idx in rows if idx < 3]):
                psi_map[k] = _iv_sub(_iv_bounds(r1[k]), _iv_bounds(r2[k]))

        t_map = {}
        for idx in self._ordered_unique(rows):
            if idx >= 3:
                Ti_i = idx - 3
                t_map[Ti_i] = _iv_bounds(_beval_scalar(self.T[Ti_i], params, self.bern_eval))

        for i, idx in enumerate(rows):
            out[i] = psi_map[idx] if idx < 3 else t_map[idx - 3]
        return out

    # ---- Jacobian (numeric) at a point, shape (7,4)

    def jac_point(self, x):
        return self.jac_rows_point(x, (0,1,2,3,4,5,6))

    def jac_rows_point(self, x, rows):
        rows = tuple(rows)
        s,t,u,v = x
        J = np.zeros((len(rows),4), dtype=float)
        psi_needed = self._ordered_unique([idx for idx in rows if idx < 3])
        psi_rows = {}
        if psi_needed:
            a = _beval_vec3(self.P1_s_point, (s,t), self.bern_eval)  # dR1/ds
            b = _beval_vec3(self.P1_t_point, (s,t), self.bern_eval)  # dR1/dt
            c = _beval_vec3(self.P2_u_point, (u,v), self.bern_eval)  # dR2/du
            d = _beval_vec3(self.P2_v_point, (u,v), self.bern_eval)  # dR2/dv
            for k in psi_needed:
                psi_rows[k] = np.array([a[k], b[k], -c[k], -d[k]], dtype=float)

        t_rows = {}
        for idx in self._ordered_unique(rows):
            if idx >= 3:
                Ti_i = idx - 3
                d0,d1,d2,d3 = self.dT_point[Ti_i]
                row = np.zeros((4,), dtype=float)
                if d0 is not None: row[0] = float(_beval_scalar(d0, (s,t,u,v), self.bern_eval))
                if d1 is not None: row[1] = float(_beval_scalar(d1, (s,t,u,v), self.bern_eval))
                if d2 is not None: row[2] = float(_beval_scalar(d2, (s,t,u,v), self.bern_eval))
                if d3 is not None: row[3] = float(_beval_scalar(d3, (s,t,u,v), self.bern_eval))
                t_rows[Ti_i] = row

        for i, idx in enumerate(rows):
            J[i,:] = psi_rows[idx] if idx < 3 else t_rows[idx - 3]
        return J

    # ---- Interval Jacobian row for equation idx in {0..6}

    def jac_row_box(self, eq_idx, B):
        return self.jac_rows_box(B, (eq_idx,))[0]

    def jac_rows_box(self, B, rows):
        rows = tuple(rows)
        sI,tI,uI,vI = _box_to_interval_params(B, self.interval_ctor)
        params = (sI,tI,uI,vI)
        out = [None] * len(rows)

        psi_needed = self._ordered_unique([idx for idx in rows if idx < 3])
        psi_rows = {}
        if psi_needed:
            a = _beval_vec3(self.P1_s, (sI,tI), self.bern_eval)
            b = _beval_vec3(self.P1_t, (sI,tI), self.bern_eval)
            c = _beval_vec3(self.P2_u, (uI,vI), self.bern_eval)
            d = _beval_vec3(self.P2_v, (uI,vI), self.bern_eval)
            for k in psi_needed:
                psi_rows[k] = [
                    _iv_bounds(a[k]),
                    _iv_bounds(b[k]),
                    _iv_scale(-1.0, _iv_bounds(c[k])),
                    _iv_scale(-1.0, _iv_bounds(d[k])),
                ]

        t_rows = {}
        for idx in self._ordered_unique(rows):
            if idx >= 3:
                Ti_i = idx - 3
                d0,d1,d2,d3 = self.dT[Ti_i]
                row = []
                for dj in (d0,d1,d2,d3):
                    if dj is None:
                        row.append((0.0,0.0))
                    else:
                        row.append(_iv_bounds(_beval_scalar(dj, params, self.bern_eval)))
                t_rows[Ti_i] = row

        for i, idx in enumerate(rows):
            out[i] = psi_rows[idx] if idx < 3 else t_rows[idx - 3]
        return out

# ------------------------------------------------------------
# Overdetermined Newton (Gauss-Newton) witness finder
# ------------------------------------------------------------

def gauss_newton_witness(sys: DeflatedSystem, B, x0=None, max_iter=30, tol_f=1e-10, tol_step=1e-12):
    """
    Try to find x in B with Δ(x)=0 using Gauss-Newton on overdetermined system (7 eq, 4 unk).
    Returns (success, x, f_norm).
    """
    if x0 is None:
        x = _box_mid(B)
    else:
        x = np.array(x0, dtype=float)

    # clamp into B
    for i in range(4):
        x[i] = min(max(x[i], B[i][0]), B[i][1])

    fnorm_prev = None
    for _ in range(max_iter):
        f = sys.delta_point(x)
        fnorm = float(np.linalg.norm(f, ord=2))
        if fnorm < tol_f:
            return True, x, fnorm

        J = sys.jac_point(x)
        # Solve least squares: J*dx = -f
        dx, *_ = np.linalg.lstsq(J, -f, rcond=None)

        step = float(np.linalg.norm(dx, ord=2))
        if step < tol_step:
            return (fnorm < 1e-8), x, fnorm

        # simple backtracking to keep inside box and reduce residual
        alpha = 1.0
        for _ls in range(12):
            xn = x + alpha*dx
            # clamp into B
            for i in range(4):
                xn[i] = min(max(xn[i], B[i][0]), B[i][1])
            fn = sys.delta_point(xn)
            fnorm_n = float(np.linalg.norm(fn, ord=2))
            if fnorm_prev is None or fnorm_n < fnorm:
                x = xn
                fnorm_prev = fnorm_n
                break
            alpha *= 0.5

    # not converged
    f = sys.delta_point(x)
    return (float(np.linalg.norm(f)) < 1e-8), x, float(np.linalg.norm(f))

# ------------------------------------------------------------
# Krawczyk interval-Newton isolation for a square system
# ------------------------------------------------------------

@dataclass
class SquareSystem:
    # F: R^4 -> R^4
    F_point: callable          # F_point(x)->(4,)
    F_box:   callable          # F_box(B)->list of 4 intervals
    J_point: callable          # J_point(x)->(4,4)
    J_box:   callable          # J_box(B)->(4,4) intervals
    verify_full: callable      # verify_full(B)->bool (optional full Δ verification)

def _mat_float_times_iv(A, M):
    """
    A: float (4,4)
    M: interval (4,4) where M[i][j]=(lo,hi)
    Returns interval (4,4)
    """
    out = [[(0.0,0.0) for _ in range(4)] for _ in range(4)]
    for i in range(4):
        for j in range(4):
            s = (0.0,0.0)
            for k in range(4):
                s = _iv_add(s, _iv_scale(A[i,k], M[k][j]))
            out[i][j] = s
    return out

def _mat_iv_times_ivvec(M, v):
    """
    M: interval (4,4)
    v: interval (4,) vector
    returns interval (4,)
    """
    out = []
    for i in range(4):
        s = (0.0,0.0)
        for k in range(4):
            s = _iv_add(s, _iv_mul(M[i][k], v[k]))
        out.append(s)
    return out

def _krawczyk_operator(sys4: SquareSystem, B, eps_interior=0.0):
    """
    One Krawczyk test on box B:
      returns ("empty", None) or ("unique", Kcap) or ("unknown", Kcap)
    """
    x0 = _box_mid(B)
    try:
        J0 = sys4.J_point(x0)
        A  = np.linalg.inv(J0)
    except np.linalg.LinAlgError:
        return "unknown", None

    f0 = sys4.F_point(x0)
    base = x0 - A.dot(f0)   # float vector

    # Interval Jacobian on B
    JI = sys4.J_box(B)      # interval 4x4

    # Compute M = I - A*JI  (interval matrix)
    AJI = _mat_float_times_iv(A, JI)
    M = [[None]*4 for _ in range(4)]
    for i in range(4):
        for j in range(4):
            Iij = (1.0,1.0) if i==j else (0.0,0.0)
            M[i][j] = _iv_sub(Iij, AJI[i][j])

    # Compute (B - x0) interval vector
    Bmx0 = []
    for i in range(4):
        lo, hi = B[i]
        Bmx0.append((lo - x0[i], hi - x0[i]))

    delta = _mat_iv_times_ivvec(M, Bmx0)  # interval vec

    # K = base + delta
    K = []
    for i in range(4):
        K.append((base[i] + delta[i][0], base[i] + delta[i][1]))

    # intersect with B
    Kcap = []
    for i in range(4):
        inter = _iv_intersect(K[i], B[i])
        if inter is None:
            return "empty", None
        Kcap.append(inter)
    Kcap = tuple(Kcap)

    # uniqueness if Kcap subset of interior(B)
    if all(_iv_is_subset(Kcap[i], B[i], strict=True, eps=eps_interior) for i in range(4)):
        return "unique", Kcap

    return "unknown", Kcap

def isolate_roots_krawczyk(sys4: SquareSystem, B, max_depth=18, min_width=1e-6, max_nodes=20000):
    """
    Subdivide B, using Krawczyk to certify:
      - empty boxes (no root)
      - unique-root boxes
    """
    stack = [(B, 0)]
    unique_boxes = []
    visited = 0

    while stack:
        Bcur, depth = stack.pop()
        visited += 1
        if visited > max_nodes:
            break

        if _box_max_width(Bcur) < min_width or depth >= max_depth:
            # Can't decide further; keep as unresolved? Here we drop.
            continue

        # quick interval range test: if any component excludes 0 -> no root
        Fint = sys4.F_box(Bcur)
        if any(not _iv_contains0(I) for I in Fint):
            continue

        status, Kcap = _krawczyk_operator(sys4, Bcur, eps_interior=0.0)
        if status == "empty":
            continue
        if status == "unique":
            # optional full verification
            if sys4.verify_full is None or sys4.verify_full(Kcap):
                unique_boxes.append(Kcap)
            continue

        # unknown -> split
        B1, B2 = _box_split(Bcur)
        stack.append((B1, depth+1))
        stack.append((B2, depth+1))

    return unique_boxes, {"visited": visited, "certified": len(unique_boxes)}

# ------------------------------------------------------------
# Build square subsystems from Δ and (optional) one hyperplane L
# ------------------------------------------------------------

def build_square_from_subset(sys: DeflatedSystem, subset, hyperplane=None):
    """
    subset: 4 indices from {0..6} for equations in Δ = (Ψx,Ψy,Ψz,T1..T4)
    hyperplane: None or dict with keys {"a":np.array(4), "b":float}
                representing L(x) = a·x + b = 0  (adds an equation)
    If hyperplane is not None, subset must include index 7 to denote L.
    """
    subset = tuple(subset)
    base_rows = tuple(idx for idx in subset if idx != 7)

    def F_point(x):
        core_vals = sys.delta_rows_point(x, base_rows) if base_rows else np.empty((0,), dtype=float)
        vals = np.zeros((4,), dtype=float)
        it = iter(core_vals)
        for i, idx in enumerate(subset):
            if hyperplane is not None and idx == 7:
                a = hyperplane["a"]; b = hyperplane["b"]
                vals[i] = float(a.dot(x) + b)
            else:
                vals[i] = next(it)
        return vals

    def F_box(B):
        core_vals = sys.delta_rows_box(B, base_rows) if base_rows else []
        vals = [None] * 4
        it = iter(core_vals)
        for i, idx in enumerate(subset):
            if hyperplane is not None and idx == 7:
                a = hyperplane["a"]; b = hyperplane["b"]
                acc = (b, b)
                for j in range(4):
                    acc = _iv_add(acc, _iv_scale(a[j], B[j]))
                vals[i] = acc
            else:
                vals[i] = next(it)
        return vals

    def J_point(x):
        J = np.zeros((4,4), dtype=float)
        core_rows_J = sys.jac_rows_point(x, base_rows) if base_rows else np.zeros((0,4), dtype=float)
        it = iter(core_rows_J)
        for r, idx in enumerate(subset):
            if hyperplane is not None and idx == 7:
                J[r,:] = hyperplane["a"]
            else:
                J[r,:] = next(it)
        return J

    def J_box(B):
        out = [[None]*4 for _ in range(4)]
        core_rows_J = sys.jac_rows_box(B, base_rows) if base_rows else []
        it = iter(core_rows_J)
        for r, idx in enumerate(subset):
            if hyperplane is not None and idx == 7:
                a = hyperplane["a"]
                for j in range(4):
                    out[r][j] = (a[j], a[j])
            else:
                row = next(it)
                for j in range(4):
                    out[r][j] = row[j]
        return out

    def verify_full(Bcand):
        # Verify all 7 equations might be zero inside Bcand (necessary condition).
        fullI = sys.delta_box(Bcand)
        return all(_iv_contains0(I) for I in fullI)

    return SquareSystem(F_point=F_point, F_box=F_box, J_point=J_point, J_box=J_box, verify_full=verify_full)

def choose_best_subset(Jfull, require=None, eq_count=7, top_k=8):
    """
    Pick good 4 rows out of Jfull (eq_count x 4) using det magnitude.
    require: if not None, an integer index that must be included in the subset.
             (use 7 for hyperplane in augmented systems)
    """
    best = []
    eq_indices = list(range(eq_count))
    if require is None:
        combos = combinations(eq_indices, 4)
    else:
        combos = (c for c in combinations(eq_indices, 4) if require in c)

    for c in combos:
        M = Jfull[list(c), :]
        try:
            det = float(abs(np.linalg.det(M)))
            cond = float(np.linalg.cond(M))
        except np.linalg.LinAlgError:
            continue
        score = det / (1.0 + cond)
        best.append((score, c))
    best.sort(reverse=True, key=lambda z: z[0])
    return [c for _,c in best[:top_k]]

# ------------------------------------------------------------
# Main: analyse Δ_B and output singular features
# ------------------------------------------------------------

def analyse_deflated_system(
    P1, P2, T1, T2, T3, T4,
    B,
    bern_eval,
    interval_ctor,
    # controls
    witness_tol=1e-8,
    isolate_points=True,
    point_min_width=1e-7,
    point_max_depth=20,
    curve_slice_count=10,
    cover_min_width=5e-3,
    cover_max_depth=14,
    use_phi_fallback=False,
    rng_seed=0,
    curve_mode="trace",
    build_cover=False,
    curve_trace_h0=1e-2,
    curve_trace_h_min=1e-6,
    curve_trace_h_max=5e-2,
    curve_trace_max_steps=2000,
    curve_trace_tol=1e-10,
    curve_krawczyk_fallback=True
):
    """
    Analyse Δ_B = Ψ ∪ {T1..T4} inside box B and output singular features.

    This corresponds to the paper's C2 analysis (Eq. (8), Fig. 7):
      - no solution
      - finite solutions -> tangent point(s)
      - infinite solutions -> tangent curve or overlap
    """
    rng = np.random.default_rng(rng_seed)

    Bf = _box_from_any(B)
    P1_arr = np.asarray(P1)
    sys = DeflatedSystem(P1=np.asarray(P1), P2=np.asarray(P2), T=(T1,T2,T3,T4),
                         bern_eval=bern_eval, interval_ctor=interval_ctor)

    out = {
        "status": "unknown",
        "dimension": None,
        "singular_points": [],
        "curve_samples": [],
        "overlap_samples": [],
        "cover_boxes": [],
        "debug": {}
    }

    # Quick prune: if Ψ(B) excludes 0 in any component => no intersection => no Δ_B
    psiI = sys.psi_box(Bf)
    if any(not _iv_contains0(I) for I in psiI):
        out["status"] = "no_solution"
        out["debug"]["reason"] = "Psi excludes 0 on B"
        return out

    # Quick prune for Δ_B itself
    deltaI = sys.delta_box(Bf)
    if any(not _iv_contains0(I) for I in deltaI):
        out["status"] = "no_solution"
        out["debug"]["reason"] = "Some equation of Δ excludes 0 on B"
        return out

    # 1) Witness by overdetermined Newton (paper: Newton to certify existence) :contentReference[oaicite:5]{index=5}
    ok, xw, fn = gauss_newton_witness(sys, Bf, tol_f=witness_tol)
    out["debug"]["witness_ok"] = ok
    out["debug"]["witness_fnorm"] = fn
    out["debug"]["witness_x"] = tuple(map(float, xw))

    if not ok:
        # Optional: Φ fallback for isolated/tiny loop detection (paper Eq.(9), Lemma 2) :contentReference[oaicite:6]{index=6}
        out["status"] = "unknown"
        out["debug"]["reason"] = "No Δ witness found"
        if not use_phi_fallback:
            return out
        # Minimal Φ test: solve one square system (Ψx,Ψy,T1,L) to hunt points.
        a = rng.normal(size=4); a = a/np.linalg.norm(a)
        b = -float(a.dot(_box_mid(Bf)))  # pass through center
        hyp = {"a": a.astype(float), "b": float(b)}
        # Build augmented Jacobian at center for subset selection
        x0 = _box_mid(Bf)
        J7 = sys.jac_point(x0)              # 7x4
        J8 = np.vstack([J7, a.reshape(1,4)])  # 8x4
        candidates = choose_best_subset(J8, require=7, eq_count=8, top_k=4)
        for sub in candidates:
            sys4 = build_square_from_subset(sys, sub, hyperplane=hyp)
            boxes, info = isolate_roots_krawczyk(sys4, Bf, max_depth=point_max_depth, min_width=point_min_width)
            if boxes:
                for Bb in boxes:
                    xm = _box_mid(Bb)
                    xyz = _to_float_vec3(_beval_vec3(P1_arr, (xm[0],xm[1]), bern_eval))
                    out["singular_points"].append({"param": tuple(map(float,xm)), "xyz": tuple(map(float,xyz))})
                out["status"] = "finite"
                out["dimension"] = 0
                out["debug"]["phi_used"] = True
                out["debug"]["phi_subset"] = sub
                out["debug"]["phi_isolation"] = info
                return out
        return out

    # 2) Dimension estimate from Jacobian rank at witness:
    #    For a k-dim solution set in R^4, expect rank ≈ 4-k.
    #    This is consistent with the paper's "add 1 or 2 hyperplanes to determine dimension". :contentReference[oaicite:7]{index=7}
    Jw = sys.jac_point(xw)  # 7x4
    svals = np.linalg.svd(Jw, compute_uv=False)
    tol_rank = max(Jw.shape) * np.max(svals) * 1e-12
    rank = int(np.sum(svals > tol_rank))
    dim = max(0, 4 - rank)
    dim = min(dim, 2)  # we only care 0/1/2 here
    out["debug"]["jac_rank"] = rank
    out["debug"]["jac_svals"] = svals.tolist()
    out["dimension"] = dim

    # 3) Output features by dimension class
    if dim == 0:
        # Finite tangency point(s): isolate roots inside B by Krawczyk + subdivision
        out["status"] = "finite"

        if not isolate_points:
            xyz = _to_float_vec3(_beval_vec3(P1_arr, (xw[0],xw[1]), bern_eval))
            out["singular_points"].append({"param": tuple(map(float,xw)), "xyz": tuple(map(float,xyz))})
            return out

        # Choose a good square subset dynamically around witness, isolate all roots.
        best_subsets = choose_best_subset(Jw, require=None, eq_count=7, top_k=6)
        all_boxes = []
        diag = []
        for sub in best_subsets:
            sys4 = build_square_from_subset(sys, sub, hyperplane=None)
            boxes, info = isolate_roots_krawczyk(sys4, Bf, max_depth=point_max_depth, min_width=point_min_width)
            diag.append({"subset": sub, "info": info})
            all_boxes.extend(boxes)

        # Deduplicate boxes by midpoint clustering (cheap)
        pts = []
        for Bb in all_boxes:
            xm = _box_mid(Bb)
            pts.append(tuple(np.round(xm, 12)))
        uniq = list(dict.fromkeys(pts))

        for p in uniq:
            p = np.array(p, dtype=float)
            xyz = _to_float_vec3(_beval_vec3(P1_arr, (p[0],p[1]), bern_eval))
            out["singular_points"].append({"param": tuple(map(float,p)), "xyz": tuple(map(float,xyz))})

        out["debug"]["point_subsets_tried"] = diag
        return out

    if dim == 1:
        # Infinite solutions of Δ_B in B: singular/tangent curve case.
        out["status"] = "infinite"
        axis = _box_widest_axis(Bf)
        lo, hi = Bf[axis]
        if curve_mode != "krawczyk":
            gamma_rows = choose_gamma_3eq(sys, xw)
            curve_4d = trace_gamma(
                sys, gamma_rows, xw, Bf,
                h0=curve_trace_h0,
                h_min=curve_trace_h_min,
                h_max=curve_trace_h_max,
                max_steps=curve_trace_max_steps,
                tol=curve_trace_tol,
            )
            samples = []
            for xm in curve_4d:
                fn = float(np.linalg.norm(sys.delta_point(xm)))
                if fn < 1e-6:
                    xyz = _to_float_vec3(_beval_vec3(P1_arr, (xm[0],xm[1]), bern_eval))
                    samples.append({"param": tuple(map(float,xm)), "xyz": tuple(map(float,xyz))})
            if samples and curve_slice_count is not None and curve_slice_count > 0 and len(samples) > curve_slice_count:
                idxs = np.linspace(0, len(samples)-1, curve_slice_count, dtype=int)
                uniq_idxs = list(dict.fromkeys(int(i) for i in idxs))
                samples = [samples[i] for i in uniq_idxs]
            out["curve_samples"] = samples
            out["debug"]["curve_mode"] = "trace"
            out["debug"]["gamma_rows"] = tuple(map(int, gamma_rows))
            out["debug"]["trace_total_points"] = int(len(curve_4d))
            out["debug"]["trace_kept_points"] = int(len(samples))
            if build_cover and samples:
                rad = max(point_min_width, 0.5*cover_min_width)
                cover = []
                seen = set()
                for s in samples:
                    p = s["param"]
                    Bb = tuple((max(Bf[i][0], p[i]-rad), min(Bf[i][1], p[i]+rad)) for i in range(4))
                    key = tuple((round(b[0], 10), round(b[1], 10)) for b in Bb)
                    if key in seen:
                        continue
                    seen.add(key)
                    cover.append(Bb)
                out["cover_boxes"] = cover
            if samples or not curve_krawczyk_fallback:
                return out
            out["debug"]["curve_fallback"] = "krawczyk"

        # Krawczyk slicing fallback (slow but robust for pathological cases)
        if build_cover:
            cover = []
            stack = [(Bf, 0)]
            while stack:
                Bb, depth = stack.pop()
                if depth >= cover_max_depth or _box_max_width(Bb) <= cover_min_width:
                    fullI = sys.delta_box(Bb)
                    if all(_iv_contains0(I) for I in fullI):
                        cover.append(Bb)
                    continue
                fullI = sys.delta_box(Bb)
                if any(not _iv_contains0(I) for I in fullI):
                    continue
                B1,B2 = _box_split(Bb)
                stack.append((B1, depth+1))
                stack.append((B2, depth+1))
            out["cover_boxes"] = cover

        if hi > lo:
            samples = []
            for k in range(curve_slice_count):
                val = lo + (k+0.5)/curve_slice_count*(hi-lo)
                a = np.zeros(4); a[axis] = 1.0
                b = -float(val)
                hyp = {"a": a, "b": b}
                J7 = sys.jac_point(xw)
                J8 = np.vstack([J7, a.reshape(1,4)])
                candidates = choose_best_subset(J8, require=7, eq_count=8, top_k=3)
                Bb = list(Bf)
                half = 0.5*(hi-lo)/curve_slice_count
                Bb[axis] = (max(lo, val-half), min(hi, val+half))
                Bb = tuple(Bb)

                for sub in candidates:
                    sys4 = build_square_from_subset(sys, sub, hyperplane=hyp)
                    boxes, _ = isolate_roots_krawczyk(sys4, Bb, max_depth=14, min_width=max(point_min_width, half*0.2))
                    for rootB in boxes:
                        xm = _box_mid(rootB)
                        fn = np.linalg.norm(sys.delta_point(xm))
                        if fn < 1e-6:
                            xyz = _to_float_vec3(_beval_vec3(P1_arr, (xm[0],xm[1]), bern_eval))
                            samples.append({"param": tuple(map(float,xm)), "xyz": tuple(map(float,xyz))})
                    if samples:
                        break
            samples.sort(key=lambda d: d["param"][axis])
            out["curve_samples"] = samples

        return out

    # dim == 2
    out["status"] = "infinite"

    # For overlap-like 2D solutions, sample with two slices (two hyperplanes) to get witness points.
    axis1 = _box_widest_axis(Bf)
    # second-widest
    widths = np.array([_iv_wid(b) for b in Bf])
    axis2 = int(np.argsort(widths)[-2]) if axis1 != int(np.argsort(widths)[-2]) else int(np.argsort(widths)[-3])

    lo1,hi1 = Bf[axis1]; lo2,hi2 = Bf[axis2]
    if hi1>lo1 and hi2>lo2:
        grid = 3
        for i in range(grid):
            for j in range(grid):
                v1 = lo1 + (i+0.5)/grid*(hi1-lo1)
                v2 = lo2 + (j+0.5)/grid*(hi2-lo2)
                # Two hyperplanes: x_axis1=v1, x_axis2=v2
                # We'll fold them by creating one hyperplane with 2 constraints is not possible in 4eq,
                # so do sequential: narrow box in both axes and solve Δ with one hyperplane and strong narrowing.
                Bb = list(Bf)
                half1 = 0.5*(hi1-lo1)/grid
                half2 = 0.5*(hi2-lo2)/grid
                Bb[axis1] = (max(lo1, v1-half1), min(hi1, v1+half1))
                Bb[axis2] = (max(lo2, v2-half2), min(hi2, v2+half2))
                Bb = tuple(Bb)

                # Use one hyperplane for axis1, narrowing handles axis2
                a = np.zeros(4); a[axis1]=1.0
                hyp = {"a": a, "b": -float(v1)}
                J7 = sys.jac_point(_box_mid(Bb))
                J8 = np.vstack([J7, a.reshape(1,4)])
                candidates = choose_best_subset(J8, require=7, eq_count=8, top_k=3)
                for sub in candidates:
                    sys4 = build_square_from_subset(sys, sub, hyperplane=hyp)
                    boxes, _ = isolate_roots_krawczyk(sys4, Bb, max_depth=12, min_width=max(point_min_width, min(half1,half2)*0.2))
                    for rootB in boxes:
                        xm = _box_mid(rootB)
                        fn = np.linalg.norm(sys.delta_point(xm))
                        if fn < 1e-6:
                            xyz = _to_float_vec3(_beval_vec3(P1_arr, (xm[0],xm[1]), bern_eval))
                            out["overlap_samples"].append({"param": tuple(map(float,xm)), "xyz": tuple(map(float,xyz))})
                    if out["overlap_samples"]:
                        break

    return out



def choose_gamma_3eq(sys, x0, tol=1e-10):
    """
    Choose 3 equations from Δ (indices 0..6) such that their 3x4 Jacobian
    has rank 3 and is as well-conditioned as possible at x0.
    """
    J = sys.jac_point(x0)  # 7x4
    best = None
    best_rows = None

    for rows in combinations(range(7), 3):
        M = J[list(rows), :]  # 3x4
        s = np.linalg.svd(M, compute_uv=False)
        # rank 3 iff third singular value not tiny
        if s[-1] < tol:
            continue
        # score: large smallest singular value and decent conditioning
        score = s[-1] / (s[0] + 1e-30)
        if best is None or score > best:
            best = score
            best_rows = rows

    if best_rows is None:
        # fallback if something pathological happens
        best_rows = (0, 1, 3)
    return best_rows

def nullspace_direction(J3x4):
    # For 3x4 you must compute a 4x4 Vt to get the null vector
    _, _, Vt = np.linalg.svd(J3x4, full_matrices=True)
    t = Vt[-1, :]
    n = np.linalg.norm(t)
    return None if n == 0 else t/n

def newton_corrector(sys, gamma_rows, x_pred, t_pred,
                     x_init=None, max_iter=12, tol=1e-12):
    """
    Solve augmented 4x4 system:
      Γ(x)=0  (3 eqs)
      (x - x_pred)·t_pred = 0  (1 eq, pseudo-arclength)
    """
    x = x_pred.copy() if x_init is None else x_init.copy()
    gamma_rows = tuple(gamma_rows)

    for _ in range(max_iter):
        F = sys.delta_rows_point(x, gamma_rows)          # 3
        g = float(np.dot(x - x_pred, t_pred))            # 1
        G = np.concatenate([F, [g]])                     # 4

        if np.linalg.norm(G, ord=2) < tol:
            return True, x

        J = sys.jac_rows_point(x, gamma_rows)            # 3x4
        A = np.vstack([J, t_pred.reshape(1, 4)])         # 4x4

        try:
            dx = np.linalg.solve(A, -G)
        except np.linalg.LinAlgError:
            return False, x

        x = x + dx

    return False, x
import numpy as np

def first_boundary_hits(x, d, h, B, eps_dir=1e-14, eps_tie=1e-10):
    """
    Return a list of candidate (alpha, axis, bound_value) for the first face(s)
    hit by the segment x + tau * (h*d), tau in [0,1].
    If no boundary is hit within this step, return [].
    """
    hits = []
    for i, (lo, hi) in enumerate(B):
        di = d[i]
        if abs(di) < eps_dir:
            continue

        if di > 0:
            alpha = (hi - x[i]) / (h * di)
            bound = hi
        else:
            alpha = (lo - x[i]) / (h * di)  # di<0 so alpha is positive
            bound = lo

        # We consider "hit" if within the step
        if 0.0 <= alpha <= 1.0:
            hits.append((alpha, i, bound))

    if not hits:
        return []

    amin = min(a for a, _, _ in hits)
    # If several faces are hit simultaneously (edge/corner), return all near-amin
    return [(a,i,b) for (a,i,b) in hits if abs(a - amin) <= eps_tie]
def newton_corrector_boundary(sys, gamma_rows, x_init, axis, bound, B,
                             max_iter=20, tol=1e-12):
    """
    Solve 4x4 system:
      Γ(x)=0 (3 eqs)
      x[axis] - bound = 0
    with a damped Newton, staying inside B.
    """
    x = x_init.copy()
    gamma_rows = tuple(gamma_rows)
    e = np.zeros(4); e[axis] = 1.0

    def inside(z, eps=1e-12):
        return all((B[i][0]-eps) <= z[i] <= (B[i][1]+eps) for i in range(4))

    def residual(z):
        F = sys.delta_rows_point(z, gamma_rows)
        return np.concatenate([F, [z[axis] - bound]])

    for _ in range(max_iter):
        G = residual(x)
        if np.linalg.norm(G, ord=2) < tol:
            return True, x

        J = sys.jac_rows_point(x, gamma_rows)  # 3x4
        A = np.vstack([J, e.reshape(1,4)])     # 4x4

        try:
            dx = np.linalg.solve(A, -G)
        except np.linalg.LinAlgError:
            return False, x

        # Damping/backtracking to keep inside and reduce residual
        alpha = 1.0
        g0 = np.linalg.norm(G)
        accepted = False
        for _ls in range(12):
            xn = x + alpha * dx
            if not inside(xn):
                alpha *= 0.5
                continue
            gn = np.linalg.norm(residual(xn))
            if gn < g0:
                x = xn
                accepted = True
                break
            alpha *= 0.5

        if not accepted:
            return False, x

    return False, x
def trace_gamma(sys, gamma_rows, x0, B, h0=1e-2, h_min=1e-6, h_max=5e-2,
                max_steps=2000, tol=1e-10, eps_end=1e-10):
    gamma_rows = tuple(gamma_rows)

    def inside(x, eps=1e-12):
        return all(B[i][0]-eps <= x[i] <= B[i][1]+eps for i in range(4))

    def nullspace_direction(J3x4):
        # IMPORTANT: full_matrices=True for 3x4
        _, _, Vt = np.linalg.svd(J3x4, full_matrices=True)
        t = Vt[-1, :]
        n = np.linalg.norm(t)
        return None if n == 0 else t/n

    def newton_corrector_arclen(x_pred, t_pred, x_init=None, max_iter=12):
        x = x_pred.copy() if x_init is None else x_init.copy()
        for _ in range(max_iter):
            F = sys.delta_rows_point(x, gamma_rows)
            g = float(np.dot(x - x_pred, t_pred))
            G = np.concatenate([F, [g]])
            if np.linalg.norm(G) < tol:
                return True, x

            J = sys.jac_rows_point(x, gamma_rows)
            A = np.vstack([J, t_pred.reshape(1,4)])

            try:
                dx = np.linalg.solve(A, -G)
            except np.linalg.LinAlgError:
                return False, x

            x = x + dx
        return False, x

    def trace_one_direction(sign):
        pts = [x0.copy()]
        h = h0

        J0 = sys.jac_rows_point(x0, gamma_rows)
        t = nullspace_direction(J0)
        if t is None:
            return pts
        t *= sign

        for _ in range(max_steps):
            x_cur = pts[-1]

            # If we're already extremely close to a face, finish by solving on that face
            dists = []
            for i,(lo,hi) in enumerate(B):
                dists.append((abs(x_cur[i]-lo), i, lo))
                dists.append((abs(hi-x_cur[i]), i, hi))
            dmin, axis_near, bound_near = min(dists, key=lambda z: z[0])
            if dmin < eps_end:
                x_guess = x_cur.copy()
                x_guess[axis_near] = bound_near
                ok, x_end = newton_corrector_boundary(sys, gamma_rows, x_guess,
                                                     axis_near, bound_near, B, tol=tol)
                if ok and inside(x_end):
                    pts.append(x_end)
                break

            # Predict
            x_pred = x_cur + h * t

            # If this step would hit a face, solve for the endpoint on that face
            hits = first_boundary_hits(x_cur, t, h, B)
            if hits:
                # Try each tied face (edge/corner), pick the first that converges
                solved = False
                for alpha_hit, axis_hit, bound_hit in hits:
                    x_guess = x_cur + alpha_hit * h * t
                    x_guess[axis_hit] = bound_hit
                    ok, x_end = newton_corrector_boundary(sys, gamma_rows, x_guess,
                                                         axis_hit, bound_hit, B, tol=tol)
                    if ok and inside(x_end) and np.linalg.norm(sys.delta_point(x_end)) < 1e-7:
                        pts.append(x_end)
                        solved = True
                        break
                if solved:
                    break
                # if boundary solve failed, reduce step and retry
                h *= 0.5
                if h < h_min:
                    break
                continue

            # Normal arclength corrector
            if not inside(x_pred):
                # Should be rare now (numerical overshoot); just reduce step
                h *= 0.5
                if h < h_min:
                    break
                continue

            ok, x_new = newton_corrector_arclen(x_pred, t, x_init=x_pred)
            if not ok or not inside(x_new):
                h *= 0.5
                if h < h_min:
                    break
                continue

            # Filter: stay on the FULL deflated set Δ
            if np.linalg.norm(sys.delta_point(x_new), ord=2) > 1e-7:
                h *= 0.5
                if h < h_min:
                    break
                continue

            pts.append(x_new)

            # Update tangent
            Jn = sys.jac_rows_point(x_new, gamma_rows)
            t_new = nullspace_direction(Jn)
            if t_new is None:
                break
            if np.dot(t_new, t) < 0:
                t_new = -t_new
            t = t_new

            h = min(h_max, h * 1.2)

        return pts

    fwd = trace_one_direction(+1)
    bwd = trace_one_direction(-1)
    bwd = bwd[::-1]
    return bwd[:-1] + fwd


if __name__ == "__main__":
    from mmcore.numeric.sbern import bern_to_nurbs_bezier
    ts1=np.array([[[0.0, 0.0, 10.0], [5.0, 5.0, 10.0], [5.0, 10.0, 10.0], [0.0, 15.0, 10.0]], [[5.0, 0.0, 0.0], [10.0, 5.0, 0.0], [10.0, 10.0, 0.0], [5.0, 15.0, 0.0]], [[10.0, 0.0, 10.0], [15.0, 5.0, 10.0], [15.0, 10.0, 10.0], [10.0, 15.0, 10.0]]])
    ts2=np.array([[[0.0, 0.0, 0.0], [5.0, 5.0, 0.0], [5.0, 10.0, 0.0], [0.0, 15.0, 0.0]], [[5.0, 0.0, 10.0], [10.0, 5.0, 10.0], [10.0, 10.0, 10.0], [5.0, 15.0, 10.0]], [[10.0, 0.0, 0.0], [15.0, 5.0, 0.0], [15.0, 10.0, 0.0], [10.0, 15.0, 0.0]]])



    is1,is2=get_iarray(ts1,ts1),get_iarray(ts2,ts2)
    T1, T2, T3, T4=(np.asarray(t, dtype=interval) for t in minors_Tpsi_from_control_nets(ts1,ts2))
    B = (interval(0,1), interval(0,1), interval(0,1), interval(0,1))
    sys = DeflatedSystem(P1=is1, P2=is2, T=(T1,T2,T3,T4),
                         bern_eval=bern_eval, interval_ctor=interval)

    Bf = ((0.0,1.0),(0.0,1.0),(0.0,1.0),(0.0,1.0))

    res = analyse_deflated_system(
        is1, is2,
        T1, T2, T3, T4,
        B,
        bern_eval=bern_eval,
        interval_ctor=interval,
        isolate_points=False,
        curve_slice_count=8,
        cover_min_width=1e-9
    )


    x0 = np.array(res["debug"]["witness_x"], dtype=float)

    gamma_rows = choose_gamma_3eq(sys, x0)
    print(
        'gamma_rows',gamma_rows)
    print(
        'x0',x0)
    curve_4d = trace_gamma(sys, gamma_rows, x0, Bf, h0=1e-2)

    # Map to 3D on surface 1
    curve_3d = []
    for x in curve_4d:
        s,t,u,v = x
        p = np.asarray(bern_eval(ts1, (s,t))).reshape(-1,3)[0]  # use float ts1 here
        curve_3d.append(p)

    print("γ rows:", gamma_rows)
    print("points:", len(curve_3d))
    print("first/last:", curve_3d[0], curve_3d[-1])

    print(np.array(curve_3d).tolist())
    assert np.allclose([5.0, 0.0, 5.0],curve_3d[0]) and np.allclose([5.0, 15.0, 5.0],curve_3d[-1]) # check if the curve ends is correct

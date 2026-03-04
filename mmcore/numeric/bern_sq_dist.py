from math import comb

import numpy as np
from .bern import bernstein_product_conv

# ─── Point–Curve ─────────────────────────────────────────────────────
def bernstein_basis(n, t):
    """Bernstein basis of degree n evaluated at t."""
    return np.array([comb(n, i) * t**i * (1 - t)**(n - i) for i in range(n + 1)])

def point_curve_distance_squared_net_homog(
    point: np.ndarray, Q: np.ndarray, rational: bool = True
) -> np.ndarray:
    r"""Squared numerator net for ``||point - C(t)||^2``.

    ::

        Δ(t) = point · w_Q(t) − H_Q(t)

    Parameters
    ----------
    point : (d,) array
        Euclidean point (not homogeneous).
    Q : (q+1, d+1) or (q+1, d) array
        Bézier control points (homogeneous if *rational*).
    rational : bool

    Returns
    -------
    F : (2q+1,) ndarray
        Univariate Bernstein coefficients of ``||Δ(t)||^2``.

    Notes
    -----
    True squared distance: ``||point − C(t)||^2 = F_poly(t) / w_Q(t)^2``.
    """
    point = np.asarray(point, dtype=float)
    Q = np.asarray(Q, dtype=float)

    q = Q.shape[0] - 1
    Qw = Q[:, -1] if rational else np.ones(q + 1, dtype=float)
    Qxyz = Q[:, :-1] if rational else Q

    # Δ_j = point * Qw_j − Qxyz_j,  shape (q+1, dim)
    D = point[None, :] * Qw[:, None] - Qxyz

    Conv = bernstein_product_conv(q).reshape(2 * q + 1, (q + 1) ** 2)
    A = D @ D.T                       # (q+1, q+1) Gram matrix
    F = Conv @ A.ravel()              # (2q+1,)
    return F

# ─── Point–Surface ───────────────────────────────────────────────────

def point_surface_distance_squared_net_homog(
    point: np.ndarray, S: np.ndarray, rational: bool = True
) -> np.ndarray:
    r"""Squared numerator net for ``||point - S(u,v)||^2``.

    ::

        Δ(u,v) = point · w_S(u,v) − H_S(u,v)

    Parameters
    ----------
    point : (d,) array
        Euclidean point (not homogeneous).
    S : (m+1, n+1, d+1) or (m+1, n+1, d) array
        Bézier surface control net (homogeneous if *rational*).
    rational : bool

    Returns
    -------
    F : (2m+1, 2n+1) ndarray
        Bivariate Bernstein net of ``||Δ(u,v)||^2``.

    Notes
    -----
    True squared distance: ``||point − S(u,v)||^2 = F_poly(u,v) / w_S(u,v)^2``.
    """
    point = np.asarray(point, dtype=float)
    S = np.asarray(S, dtype=float)

    m = S.shape[0] - 1
    n = S.shape[1] - 1

    Sw = S[:, :, -1] if rational else np.ones((m + 1, n + 1), dtype=float)
    Sxyz = S[:, :, :-1] if rational else S

    # Δ_{ij} = point * Sw_{ij} − Sxyz_{ij},  shape (m+1, n+1, dim)
    D = point[None, None, :] * Sw[:, :, None] - Sxyz

    ConvU = bernstein_product_conv(m).reshape(2 * m + 1, (m + 1) ** 2)
    ConvV = bernstein_product_conv(n).reshape(2 * n + 1, (n + 1) ** 2)

    # Contract u-axis, then v-axis — same structure as curve–curve
    Dj = D.transpose(1, 0, 2)        # (n+1, m+1, dim)
    Tu = np.zeros((2 * m + 1, n + 1, n + 1), dtype=float)
    for j in range(n + 1):
        Dj_mat = Dj[j]               # (m+1, dim)
        for jp in range(n + 1):
            Djp_mat = Dj[jp]
            Tu[:, j, jp] = ConvU @ (Dj_mat @ Djp_mat.T).ravel()

    F = np.zeros((2 * m + 1, 2 * n + 1), dtype=float)
    for r in range(2 * m + 1):
        F[r, :] = ConvV @ Tu[r].ravel()
    return F

def curve_curve_squared_net_homog(P: np.ndarray, Q: np.ndarray, rational: bool | None = True) -> np.ndarray:
    r"""Squared numerator net for ``||C1-C2||^2`` with homogeneous inputs.

    We avoid dehomogenization by cross-multiplying weights::

        Δ(u,v) = H1_xyz(u) * w2(v) - H2_xyz(v) * w1(u)

    The zero set of ``||Δ||^2`` matches that of the Euclidean distance while
    keeping the computation polynomial (no divisions).

    Parameters
    ----------
    P : (p+1, d) array_like
        Bézier control net of the first curve. If the last column stores
        weights, the curve is treated as rational.
    Q : (q+1, d) array_like
        Bézier control net of the second curve. Same convention as ``P``.

    Returns
    -------
    F : (2p+1, 2q+1) ndarray
        Bivariate Bernstein control net of ``||Δ(u,v)||^2``.

    Notes
    -----
    This is algebraically equivalent to building the Euclidean distance
    between dehomogenized curves, but avoids divisions so that subsequent
    subdivision and pruning remain exact.
    """

    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)
    if P.ndim != 2 or Q.ndim != 2:
        raise ValueError("P and Q must be 2D arrays of shapes (p+1,d) and (q+1,d).")

    p = P.shape[0] - 1
    q = Q.shape[0] - 1

    Pw = P[:, -1] if rational else np.ones(P.shape[0], dtype=float)
    Qw = Q[:, -1] if rational else np.ones(Q.shape[0], dtype=float)
    Pxyz = P[:, :-1] if rational else P
    Qxyz = Q[:, :-1] if rational else Q

    # Tensor-product control net for Δ
    D = Pxyz[:, None, :] * Qw[None, :, None] - Qxyz[None, :, :] * Pw[:, None, None]

    # Reuse the exact convolution pattern from bernstein_distance_squared_net
    ConvU = bernstein_product_conv(p)  # (2p+1, p+1, p+1)
    ConvV = bernstein_product_conv(q)  # (2q+1, q+1, q+1)

    ConvU2 = ConvU.reshape(2 * p + 1, (p + 1) * (p + 1))
    ConvV2 = ConvV.reshape(2 * q + 1, (q + 1) * (q + 1))

    Tu = np.zeros((2 * p + 1, q + 1, q + 1), dtype=float)
    Dj = D.transpose(1, 0, 2)
    for j in range(q + 1):
        Dj_mat = Dj[j]
        for jp in range(q + 1):
            Djp_mat = Dj[jp]
            A = Dj_mat @ Djp_mat.T
            Tu[:, j, jp] = ConvU2 @ A.ravel()

    F = np.zeros((2 * p + 1, 2 * q + 1), dtype=float)
    for r in range(2 * p + 1):
        B = Tu[r, :, :]
        F[r, :] = ConvV2 @ B.ravel()

    return F

# ─── Curve–Surface ───────────────────────────────────────────────────

def curve_surface_distance_squared_net_homog(
    P: np.ndarray, S: np.ndarray, rational: bool = True
) -> np.ndarray:
    r"""Squared numerator net for ``||C(t) - S(u,v)||^2``.

    ::

        Δ(t,u,v) = H_C(t) · w_S(u,v) − H_S(u,v) · w_C(t)

    Parameters
    ----------
    P : (p+1, d+1) or (p+1, d) array
        Curve control points (homogeneous if *rational*).
    S : (m+1, n+1, d+1) or (m+1, n+1, d) array
        Surface control net (homogeneous if *rational*).
    rational : bool

    Returns
    -------
    F : (2p+1, 2m+1, 2n+1) ndarray
        Trivariate Bernstein net of ``||Δ(t,u,v)||^2``.

    Notes
    -----
    True squared distance:
    ``||C(t) − S(u,v)||^2 = F_poly(t,u,v) / (w_C(t)^2 · w_S(u,v)^2)``.
    """
    P = np.asarray(P, dtype=float)
    S = np.asarray(S, dtype=float)

    p = P.shape[0] - 1
    m = S.shape[0] - 1
    n = S.shape[1] - 1

    Pw = P[:, -1] if rational else np.ones(p + 1, dtype=float)
    Pxyz = P[:, :-1] if rational else P
    Sw = S[:, :, -1] if rational else np.ones((m + 1, n + 1), dtype=float)
    Sxyz = S[:, :, :-1] if rational else S

    # D[i,j,k,:] = Pxyz[i]*Sw[j,k] − Sxyz[j,k]*Pw[i]
    # shape (p+1, m+1, n+1, dim)
    D = (Pxyz[:, None, None, :] * Sw[None, :, :, None]
         - Sxyz[None, :, :, :] * Pw[:, None, None, None])

    ConvP = bernstein_product_conv(p).reshape(2 * p + 1, (p + 1) ** 2)
    ConvM = bernstein_product_conv(m).reshape(2 * m + 1, (m + 1) ** 2)
    ConvN = bernstein_product_conv(n).reshape(2 * n + 1, (n + 1) ** 2)

    # --- Step 1: contract curve axis (t) ---
    # T1[r, j, jp, k, kp] for all surface index pairs
    T1 = np.zeros((2 * p + 1, m + 1, m + 1, n + 1, n + 1), dtype=float)
    for j in range(m + 1):
        for jp in range(m + 1):
            for k in range(n + 1):
                for kp in range(n + 1):
                    A = D[:, j, k, :] @ D[:, jp, kp, :].T  # (p+1, p+1)
                    T1[:, j, jp, k, kp] = ConvP @ A.ravel()

    # --- Step 2: contract surface-u axis ---
    T2 = np.zeros((2 * p + 1, 2 * m + 1, n + 1, n + 1), dtype=float)
    for r in range(2 * p + 1):
        for k in range(n + 1):
            for kp in range(n + 1):
                T2[r, :, k, kp] = ConvM @ T1[r, :, :, k, kp].ravel()

    # --- Step 3: contract surface-v axis ---
    F = np.zeros((2 * p + 1, 2 * m + 1, 2 * n + 1), dtype=float)
    for r in range(2 * p + 1):
        for s in range(2 * m + 1):
            F[r, s, :] = ConvN @ T2[r, s, :, :].ravel()

    return F

# ─── Surface–Surface ─────────────────────────────────────────────────

def surface_surface_distance_squared_net_homog(
    P: np.ndarray, Q: np.ndarray, rational: bool = True
) -> np.ndarray:
    r"""Squared numerator net for ``||S1(u1,v1) - S2(u2,v2)||^2``.

    ::

        Δ(u1,v1,u2,v2) = H_{S1}(u1,v1) · w_{S2}(u2,v2)
                        − H_{S2}(u2,v2) · w_{S1}(u1,v1)

    Parameters
    ----------
    P : (p1+1, p2+1, d+1) or (p1+1, p2+1, d) array
        First surface control net (homogeneous if *rational*).
    Q : (q1+1, q2+1, d+1) or (q1+1, q2+1, d) array
        Second surface control net (homogeneous if *rational*).
    rational : bool

    Returns
    -------
    F : (2*p1+1, 2*p2+1, 2*q1+1, 2*q2+1) ndarray
        4-variate Bernstein net of ``||Δ||^2``.

    Notes
    -----
    True squared distance:
    ``||S1 − S2||^2 = F_poly / (w_{S1}(u1,v1)^2 · w_{S2}(u2,v2)^2)``.
    """
    P = np.asarray(P, dtype=float)
    Q = np.asarray(Q, dtype=float)

    p1, p2 = P.shape[0] - 1, P.shape[1] - 1
    q1, q2 = Q.shape[0] - 1, Q.shape[1] - 1

    Pw = P[:, :, -1] if rational else np.ones((p1 + 1, p2 + 1), dtype=float)
    Pxyz = P[:, :, :-1] if rational else P
    Qw = Q[:, :, -1] if rational else np.ones((q1 + 1, q2 + 1), dtype=float)
    Qxyz = Q[:, :, :-1] if rational else Q

    # D[i,j,k,l,:] = Pxyz[i,j]*Qw[k,l] − Qxyz[k,l]*Pw[i,j]
    # shape (p1+1, p2+1, q1+1, q2+1, dim)
    D = (Pxyz[:, :, None, None, :] * Qw[None, None, :, :, None]
         - Qxyz[None, None, :, :, :] * Pw[:, :, None, None, None])

    ConvP1 = bernstein_product_conv(p1).reshape(2 * p1 + 1, (p1 + 1) ** 2)
    ConvP2 = bernstein_product_conv(p2).reshape(2 * p2 + 1, (p2 + 1) ** 2)
    ConvQ1 = bernstein_product_conv(q1).reshape(2 * q1 + 1, (q1 + 1) ** 2)
    ConvQ2 = bernstein_product_conv(q2).reshape(2 * q2 + 1, (q2 + 1) ** 2)

    # --- Step 1: contract axis 0 (u1, degree p1) ---
    # Flatten remaining axes into one compound index
    N = (p2 + 1) * (q1 + 1) * (q2 + 1)
    D_flat = D.reshape(p1 + 1, N, -1)           # (p1+1, N, dim)
    # Gram: G[i,i'] is (N, N) — compute via batch outer product
    # G[i, i', c, c'] = sum_d D_flat[i,c,d] * D_flat[i',c',d]
    G = np.einsum('icd,jcd->ijc', D_flat, D_flat)  # WRONG shape
    # Actually need full pairwise — use matmul
    # G[i,i'] = D_flat[i] @ D_flat[i'].T   each (N, N)
    # Flatten to ((p1+1)^2, N^2) and contract
    G = np.zeros(((p1 + 1), (p1 + 1), N, N), dtype=float)
    for i in range(p1 + 1):
        for ip in range(p1 + 1):
            G[i, ip] = D_flat[i] @ D_flat[ip].T
    G_flat = G.reshape((p1 + 1) ** 2, N * N)
    T1 = (ConvP1 @ G_flat).reshape(2 * p1 + 1, p2 + 1, q1 + 1, q2 + 1,
                                    p2 + 1, q1 + 1, q2 + 1)

    # --- Step 2: contract axis 1 (v1, degree p2) ---
    T2 = np.zeros((2 * p1 + 1, 2 * p2 + 1,
                    q1 + 1, q2 + 1, q1 + 1, q2 + 1), dtype=float)
    for r in range(2 * p1 + 1):
        for k in range(q1 + 1):
            for kp in range(q1 + 1):
                for l in range(q2 + 1):
                    for lp in range(q2 + 1):
                        T2[r, :, k, l, kp, lp] = ConvP2 @ (
                            T1[r, :, k, l, :, kp, lp].ravel())

    # --- Step 3: contract axis 2 (u2, degree q1) ---
    T3 = np.zeros((2 * p1 + 1, 2 * p2 + 1,
                    2 * q1 + 1, q2 + 1, q2 + 1), dtype=float)
    for r in range(2 * p1 + 1):
        for s in range(2 * p2 + 1):
            for l in range(q2 + 1):
                for lp in range(q2 + 1):
                    T3[r, s, :, l, lp] = ConvQ1 @ (
                        T2[r, s, :, l, :, lp].ravel())

    # --- Step 4: contract axis 3 (v2, degree q2) ---
    F = np.zeros((2 * p1 + 1, 2 * p2 + 1,
                  2 * q1 + 1, 2 * q2 + 1), dtype=float)
    for r in range(2 * p1 + 1):
        for s in range(2 * p2 + 1):
            for t in range(2 * q1 + 1):
                F[r, s, t, :] = ConvQ2 @ T3[r, s, t, :, :].ravel()

    return F

# ─── Bounds ─────────────────────────────────────────────────
def bounds_point_curve(F, Qw):
    """
    Bounds on ||point - C(t)||^2 over t ∈ [0,1].

    F shape: (2q+1,)
    Qw shape: (q+1,)
    """
    N_min, N_max = F.min(), F.max()
    w_max = Qw.max()
    w_min = Qw.min()

    lb = N_min / w_max ** 2
    ub = N_max / w_min ** 2
    return lb, ub


def bounds_point_surface(F, Sw):
    """
    Bounds on ||point - S(u,v)||^2 over (u,v) ∈ [0,1]^2.

    F shape: (2m+1, 2n+1)
    Sw shape: (m+1, n+1)
    """
    N_min, N_max = F.min(), F.max()
    w_min = Sw.min()
    w_max = Sw.max()

    lb = N_min / w_max ** 2
    ub = N_max / w_min ** 2
    return lb, ub


def bounds_curve_curve(F, Pw, Qw):
    """
    Bounds on ||C1(u) - C2(v)||^2 over (u,v) ∈ [0,1]^2.

    F shape: (2p+1, 2q+1)
    Pw shape: (p+1,)
    Qw shape: (q+1,)
    """
    N_min, N_max = F.min(), F.max()
    denom_max = (Pw.max() * Qw.max()) ** 2
    denom_min = (Pw.min() * Qw.min()) ** 2

    lb = N_min / denom_max
    ub = N_max / denom_min
    return lb, ub


def bounds_curve_surface(F, Pw, Sw):
    """
    Bounds on ||C(t) - S(u,v)||^2 over (t,u,v) ∈ [0,1]^3.

    F shape: (2p+1, 2m+1, 2n+1)
    Pw shape: (p+1,)
    Sw shape: (m+1, n+1)
    """
    N_min, N_max = F.min(), F.max()
    w1_min, w1_max = Pw.min(), Pw.max()
    w2_min, w2_max = Sw.min(), Sw.max()

    denom_max = (w1_max * w2_max) ** 2
    denom_min = (w1_min * w2_min) ** 2

    lb = N_min / denom_max
    ub = N_max / denom_min
    return lb, ub


def bounds_surface_surface(F, Pw, Qw):
    """
    Bounds on ||S1(u1,v1) - S2(u2,v2)||^2 over (u1,v1,u2,v2) ∈ [0,1]^4.

    F shape: (2p1+1, 2p2+1, 2q1+1, 2q2+1)
    Pw shape: (p1+1, p2+1)
    Qw shape: (q1+1, q2+1)
    """
    N_min, N_max = F.min(), F.max()
    w1_min, w1_max = Pw.min(), Pw.max()
    w2_min, w2_max = Qw.min(), Qw.max()

    denom_max = (w1_max * w2_max) ** 2
    denom_min = (w1_min * w2_min) ** 2

    lb = N_min / denom_max
    ub = N_max / denom_min
    return lb, ub

def eval_curve_curve_distance_sq(F, Pw, Qw, s, t):
    p = (F.shape[0] - 1) // 2   # original curve degree
    q = (F.shape[1] - 1) // 2

    # Numerator: bivariate Bernstein of degree (2p, 2q)
    Bu = bernstein_basis(2 * p, s)
    Bv = bernstein_basis(2 * q, t)
    N = Bu @ F @ Bv

    # Denominator: w1(s)^2 * w2(t)^2
    w1 = bernstein_basis(p, s) @ Pw
    w2 = bernstein_basis(q, t) @ Qw

    return N / (w1 * w2) ** 2

def eval_point_curve_distance_sq(F, Qw, t):
    """F shape: (2q+1,)"""
    q = (F.shape[0] - 1) // 2

    N = bernstein_basis(2 * q, t) @ F
    w2 = bernstein_basis(q, t) @ Qw

    return N / w2 ** 2
def eval_point_surface_distance_sq(F, Sw, u, v):
    """F shape: (2m+1, 2n+1), Sw shape: (m+1, n+1)"""
    m = (F.shape[0] - 1) // 2
    n = (F.shape[1] - 1) // 2

    Bu = bernstein_basis(2 * m, u)
    Bv = bernstein_basis(2 * n, v)
    N = Bu @ F @ Bv

    wu = bernstein_basis(m, u)
    wv = bernstein_basis(n, v)
    w = wu @ Sw @ wv          # bivariate weight evaluation

    return N / w ** 2

def eval_curve_surface_distance_sq(F, Pw, Sw, t, u, v):
    """F shape: (2p+1, 2m+1, 2n+1)"""
    p = (F.shape[0] - 1) // 2
    m = (F.shape[1] - 1) // 2
    n = (F.shape[2] - 1) // 2

    Bt = bernstein_basis(2 * p, t)     # (2p+1,)
    Bu = bernstein_basis(2 * m, u)     # (2m+1,)
    Bv = bernstein_basis(2 * n, v)     # (2n+1,)

    # Step 1: contract t-axis
    # R1[j,k] = sum_i Bt[i] * F[i,j,k]
    R1 = np.zeros((2 * m + 1, 2 * n + 1))
    for j in range(2 * m + 1):
        for k in range(2 * n + 1):
            s = 0.0
            for i in range(2 * p + 1):
                s += Bt[i] * F[i, j, k]
            R1[j, k] = s

    # Step 2: contract u-axis
    # R2[k] = sum_j Bu[j] * R1[j,k]
    R2 = np.zeros(2 * n + 1)
    for k in range(2 * n + 1):
        s = 0.0
        for j in range(2 * m + 1):
            s += Bu[j] * R1[j, k]
        R2[k] = s

    # Step 3: contract v-axis
    # N = sum_k Bv[k] * R2[k]
    N = 0.0
    for k in range(2 * n + 1):
        N += Bv[k] * R2[k]

    # Denominator
    w1 = 0.0
    Bt_p = bernstein_basis(p, t)
    for i in range(p + 1):
        w1 += Bt_p[i] * Pw[i]

    wu = bernstein_basis(m, u)
    wv = bernstein_basis(n, v)
    w2 = 0.0
    for i in range(m + 1):
        for j in range(n + 1):
            w2 += wu[i] * Sw[i, j] * wv[j]

    return N / (w1 * w2) ** 2


def eval_surface_surface_distance_sq(F, Pw, Qw, u1, v1, u2, v2):
    """F shape: (2p1+1, 2p2+1, 2q1+1, 2q2+1)"""
    p1 = (F.shape[0] - 1) // 2
    p2 = (F.shape[1] - 1) // 2
    q1 = (F.shape[2] - 1) // 2
    q2 = (F.shape[3] - 1) // 2

    Bu1 = bernstein_basis(2 * p1, u1)
    Bv1 = bernstein_basis(2 * p2, v1)
    Bu2 = bernstein_basis(2 * q1, u2)
    Bv2 = bernstein_basis(2 * q2, v2)

    # Step 1: contract u1-axis
    # R1[j,k,l] = sum_i Bu1[i] * F[i,j,k,l]
    R1 = np.zeros((2 * p2 + 1, 2 * q1 + 1, 2 * q2 + 1))
    for j in range(2 * p2 + 1):
        for k in range(2 * q1 + 1):
            for l in range(2 * q2 + 1):
                s = 0.0
                for i in range(2 * p1 + 1):
                    s += Bu1[i] * F[i, j, k, l]
                R1[j, k, l] = s

    # Step 2: contract v1-axis
    # R2[k,l] = sum_j Bv1[j] * R1[j,k,l]
    R2 = np.zeros((2 * q1 + 1, 2 * q2 + 1))
    for k in range(2 * q1 + 1):
        for l in range(2 * q2 + 1):
            s = 0.0
            for j in range(2 * p2 + 1):
                s += Bv1[j] * R1[j, k, l]
            R2[k, l] = s

    # Step 3: contract u2-axis
    # R3[l] = sum_k Bu2[k] * R2[k,l]
    R3 = np.zeros(2 * q2 + 1)
    for l in range(2 * q2 + 1):
        s = 0.0
        for k in range(2 * q1 + 1):
            s += Bu2[k] * R2[k, l]
        R3[l] = s

    # Step 4: contract v2-axis
    # N = sum_l Bv2[l] * R3[l]
    N = 0.0
    for l in range(2 * q2 + 1):
        N += Bv2[l] * R3[l]

    # Denominator: w1(u1,v1)^2 * w2(u2,v2)^2
    wu1 = bernstein_basis(p1, u1)
    wv1 = bernstein_basis(p2, v1)
    w1 = 0.0
    for i in range(p1 + 1):
        for j in range(p2 + 1):
            w1 += wu1[i] * Pw[i, j] * wv1[j]

    wu2 = bernstein_basis(q1, u2)
    wv2 = bernstein_basis(q2, v2)
    w2 = 0.0
    for i in range(q1 + 1):
        for j in range(q2 + 1):
            w2 += wu2[i] * Qw[i, j] * wv2[j]

    return N / (w1 * w2) ** 2
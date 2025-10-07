import numpy as np
from typing import Tuple


from mmcore.geom._nurbs_eval import NURBSCurveTuple,NURBSSurfaceTuple

# ---------- Utilities: knots, spans, basis, derivatives ----------

def _find_span(n: int, p: int, u: float, U: np.ndarray) -> int:
    """
    Find the knot span index i in [p, n] such that U[i] <= u < U[i+1].
    Special case u == U[n+1]: return n.
    """
    if u >= U[n+1]:
        return n
    if u <= U[p]:
        return p
    low, high = p, n + 1
    while True:
        mid = (low + high) // 2
        if u < U[mid]:
            high = mid
        elif u >= U[mid + 1]:
            low = mid
        else:
            return mid


def _basis_funs(i: int, u: float, p: int, U: np.ndarray) -> np.ndarray:
    """
    Algorithm A2.2 from The NURBS Book: nonzero basis functions N_{i-p..i,p}(u).
    Returns shape (p+1,).
    """
    N = np.empty(p + 1, dtype=float)
    left = np.empty(p + 1, dtype=float)
    right = np.empty(p + 1, dtype=float)
    N[0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        for r in range(j):
            denom = right[r + 1] + left[j - r]
            term = 0.0 if denom == 0.0 else N[r] / denom
            temp = term * right[r + 1]
            N[r] = saved + temp
            saved = term * left[j - r]
        N[j] = saved
    return N


def _basis_matrix(U: np.ndarray, p: int, u_vec: np.ndarray) -> np.ndarray:
    """
    Build a (len(u_vec) x n+1) sparse-dense matrix of basis functions
    at the collocation points (each row has p+1 nonzeros).
    """
    U = np.asarray(U, dtype=float)
    u_vec = np.asarray(u_vec, dtype=float)
    n = len(U) - p - 2  # last control index
    B = np.zeros((len(u_vec), n + 1), dtype=float)
    for j, u in enumerate(u_vec):
        i = _find_span(n, p, u, U)
        N = _basis_funs(i, u, p, U)
        B[j, i - p:i + 1] = N
    return B


def _curve_derivative_ctrl(C: np.ndarray, U: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Control points and knot vector of the derivative of a degree p B-spline curve with control C.
    Returns (C', U', p-1).
    Formula: C'_i = p * (C_{i+1} - C_i) / (U_{i+p+1} - U_{i+1})
    """
    C = np.asarray(C, dtype=float)
    n = C.shape[0] - 1
    if p <= 0:
        raise ValueError("Degree p must be >= 1 to differentiate.")
    denom = (U[1 + p:1 + p + n] - U[1:1 + n])
    # Avoid division by zero for zero-length knot spans:
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = np.where(denom != 0.0, p / denom, 0.0)
    dC = (C[1:] - C[:-1]) * scale[:, None] if C.ndim > 1 else (C[1:] - C[:-1]) * scale
    Uprime = U[1:-1].copy()
    return dC, Uprime, p - 1


def _greville_abscissae(U: np.ndarray, p: int) -> np.ndarray:
    """
    Greville points for degree p with knot vector U.
    xi_i = (U[i+1] + ... + U[i+p]) / p, for i=0..n
    """
    n = len(U) - p - 2
    if p <= 0:
        # degree 0 Greville points are undefined; not used in this code path
        raise ValueError("Degree must be >=1 for Greville abscissae.")
    # Sum p interior knots per control index:
    xi = np.empty(n + 1, dtype=float)
    for i in range(n + 1):
        xi[i] = np.sum(U[i + 1:i + p + 1]) / p
    return xi


# ---------- Knot-vector construction for the derivative space ----------

def _final_derivative_knot_vector(U: np.ndarray, p: int) -> np.ndarray:
    """
    Build the shared knot vector for the exact derivative NURBS:
    - Degree q = 2p
    - End multiplicities = 2p+1
    - Each interior knot multiplicity mu_i becomes 2*mu_i + 1

    We use the *parametric domain* [a, b] = [U[p], U[-p-1]] and interior
    knots from U[p+1 : -p-1] with their multiplicities.
    """
    U = np.asarray(U, dtype=float)
    a = U[p]
    b = U[-p - 1]
    q = 2 * p

    interior = U[p + 1:-p - 1]
    vals = []
    mults = []
    if interior.size > 0:
        # Group with exact equality (knot vectors are exact sequences in NURBS)
        last = interior[0]
        count = 1
        for t in interior[1:]:
            if t == last:  # exact match (knots are typically exact floats)
                count += 1
            else:
                vals.append(last)
                mults.append(count)
                last, count = t, 1
        vals.append(last)
        mults.append(count)

    # Assemble final knots
    out = [a] * (q + 1)  # start with q+1 copies; we'll add one more to reach 2p+1 at the end append
    # Actually for ends we want exactly 2p+1, so we write directly:
    out = [a] * (q + 1)  # this is 2p+1 already
    for v, mu in zip(vals, mults):
        out.extend([v] * (2 * mu + 1))
    out.extend([b] * (q + 1))
    return np.array(out, dtype=float)


# ---------- Main algorithm ----------

def derivative_nurbs(curve: NURBSCurveTuple) -> NURBSCurveTuple:
    """
    Compute the derivative of a NURBS curve and return the resulting NURBS representation.

    This function computes the derivative of a given NURBS curve by transforming
    the original curve's control points, weights, and knot vector. The derivative
    is expressed as a new NURBS curve with updated order, control points, and weights.
    The procedure includes intermediate computations using homogeneous coordinates
    and derivative control points.

    :param curve: The input NURBS curve to be differentiated. It should be an instance
                  of NURBSCurveTuple containing attributes:
                  - `order`: Integer representing the order of the curve (degree + 1).
                  - `knot`: Knot vector of the curve.
                  - `control_points`: A 2D array (n+1, dim) representing the control points.
                  - `weights`: A 1D array (n+1,) representing the weights of the control points.
    :type curve: NURBSCurveTuple
    :return: A new NURBSCurveTuple representing the derivative curve. The returned tuple
             contains updated attributes:
             - `order`: Integer representing the new order (degree + 1) of the derivative curve.
             - `knot`: Updated knot vector for the derivative curve.
             - `control_points`: A 2D array representing the control points for the
                                 derivative curve in rational form.
             - `weights`: A 1D array containing the updated weights for the derivative curve.
    :rtype: NURBSCurveTuple
    """
    p = curve.order - 1
    U = np.asarray(curve.knot, dtype=float)
    P = np.asarray(curve.control_points, dtype=float)  # (n+1, dim)
    w = np.asarray(curve.weights, dtype=float)         # (n+1,)

    if P.ndim != 2:
        raise ValueError("control_points must be a 2D array (N, dim).")
    if P.shape[0] != w.shape[0]:
        raise ValueError("control_points and weights length mismatch.")
    if p < 1:
        raise ValueError("Curve degree must be >= 1.")

    n = P.shape[0] - 1
    if len(U) != (n + p + 2):
        raise ValueError("Inconsistent knot vector length.")

    dim = P.shape[1]

    # Homogeneous numerator coefficients:
    X_ctrl = P * w[:, None]     # (n+1, dim)
    w_ctrl = w                  # (n+1,)

    # Derivative control points (degree p-1, knot vector U'):
    Xd_ctrl, Ud, pd = _curve_derivative_ctrl(X_ctrl, U, p)  # (n, dim), knots U[1:-1], degree p-1
    wd_ctrl, _, _   = _curve_derivative_ctrl(w_ctrl, U, p)  # (n,), same Ud, pd

    # Final shared knot vector for derivative space (degree q = 2p)
    V = _final_derivative_knot_vector(U, p)
    q = 2 * p
    # Number of control points in the final space:
    Nfinal = len(V) - q - 1  # = (#basis functions)

    # Collocation points (Greville abscissae of the final space)
    xi = _greville_abscissae(V, q)  # (Nfinal,)

    # Basis matrices to evaluate X, w, X', w' at xi:
    Bp  = _basis_matrix(U,  p,  xi)         # (Nfinal, n+1)
    Bpd = _basis_matrix(Ud, pd, xi)         # (Nfinal, n)   (pd = p-1)

    # Evaluate values at xi:
    w_val  = Bp @ w_ctrl                    # (Nfinal,)
    X_val  = Bp @ X_ctrl                    # (Nfinal, dim)
    wd_val = Bpd @ wd_ctrl                  # (Nfinal,)
    Xd_val = Bpd @ Xd_ctrl                  # (Nfinal, dim)

    # Numerator/denominator samples at xi
    numer_samples = Xd_val * w_val[:, None] - X_val * wd_val[:, None]    # (Nfinal, dim)
    denom_samples = (w_val ** 2)                                         # (Nfinal,)

    # Basis matrix for final (degree q) space on V
    Bq = _basis_matrix(V, q, xi)               # (Nfinal, Nfinal)  square, well-conditioned

    # Solve for control coefficients in the final space:
    # Bq * C_den   = denom_samples
    # Bq * C_numer = numer_samples  (solve per coordinate)
    C_den = np.linalg.solve(Bq, denom_samples)
    C_num = np.linalg.solve(Bq, numer_samples)  # (Nfinal, dim)

    # Assemble homogeneous derivative control points and convert back to rational form
    # H_der[i] = [C_num[i,:], C_den[i]]
    eps = 0.0
    if np.any(C_den == 0.0):
        # In well-posed NURBS (positive weights) w(u) > 0, hence C_den should be positive.
        # If not, allow tiny epsilon to avoid division by zero in degenerate inputs.
        eps = np.finfo(float).eps
    weights_der = C_den
    ctrl_der = C_num / (weights_der[:, None] + eps)

    return NURBSCurveTuple(
        order=q + 1,
        knot=V,
        control_points=ctrl_der,
        weights=weights_der
    )




# ----------------------- Surface derivative control nets ----------------------

def _surface_derivative_ctrl_u(C: np.ndarray, U: np.ndarray, p: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Derivative w.r.t. u of a B-spline surface control net C (nu, nv, [D]).
    Returns (C_u, U', p-1) where C_u has shape (nu-1, nv, [D]).
    """
    C = np.asarray(C, dtype=float)
    nu = C.shape[0]
    denom = (U[1 + p:1 + p + nu - 1] - U[1:1 + nu - 1])  # length nu-1
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = np.where(denom != 0.0, p / denom, 0.0)
    diff = C[1:] - C[:-1]
    scale_b = scale.reshape((nu - 1,) + (1,) * (C.ndim - 1))
    Cu = diff * scale_b
    Ud = U[1:-1].copy()
    return Cu, Ud, p - 1


def _surface_derivative_ctrl_v(C: np.ndarray, V: np.ndarray, q: int) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Derivative w.r.t. v of a B-spline surface control net C (nu, nv, [D]).
    Returns (C_v, V', q-1) where C_v has shape (nu, nv-1, [D]).
    """
    C = np.asarray(C, dtype=float)
    nv = C.shape[1]
    denom = (V[1 + q:1 + q + nv - 1] - V[1:1 + nv - 1])  # length nv-1
    with np.errstate(divide='ignore', invalid='ignore'):
        scale = np.where(denom != 0.0, q / denom, 0.0)
    diff = C[:, 1:] - C[:, :-1]
    scale_b = (1,) + (nv - 1,) + (1,) * (C.ndim - 2)
    scale_b = np.ones(scale_b, dtype=float)  # broadcast helper
    scale_b = scale.reshape((1, nv - 1) + (1,) * (C.ndim - 2))
    Cv = diff * scale_b
    Vd = V[1:-1].copy()
    return Cv, Vd, q - 1


# ---------------------- Tensor evaluation and fitting ------------------------

def _eval_surface(C: np.ndarray, Bu: np.ndarray, Bv: np.ndarray) -> np.ndarray:
    """
    Evaluate tensor-product spline with control net C at the (tensor) grid given by
    Bu (Mu, Nu) and Bv (Mv, Nv). Works for scalar (nu,nv) and vector (nu,nv,D).
    Returns (Mu, Mv) or (Mu, Mv, D).
    """
    if C.ndim == 2:
        return np.einsum('iu,uv,jv->ij', Bu, C, Bv)
    else:
        return np.einsum('iu,uvd,jv->ijd', Bu, C, Bv)


def _two_sided_solve(Bu: np.ndarray, Bv: np.ndarray, samples: np.ndarray) -> np.ndarray:
    """
    Solve Bu * C * Bv^T = samples for C (control net) using two square solves.
    Bu and Bv must be square (collocation on Greville of the final space).
    Supports scalar (Mu,Mv) and vector (Mu,Mv,D) samples.
    """
    Mu, Nu = Bu.shape
    Mv, Nv = Bv.shape
    if Mu != Nu or Mv != Nv:
        raise ValueError("Bu and Bv must be square for two-sided solve.")

    if samples.ndim == 2:
        Y = np.linalg.solve(Bu, samples)           # (Nu, Mv)
        C = np.linalg.solve(Bv, Y.T).T             # (Nu, Nv)
        return C
    else:
        Mu_s, Mv_s, d = samples.shape
        if Mu_s != Mu or Mv_s != Mv:
            raise ValueError("Sample grid size mismatch.")
        # Solve along u for all right-hand sides at once
        Y = np.linalg.solve(Bu, samples.reshape(Mu, Mv * d))  # (Nu, Mv*d)
        Y = Y.reshape(Nu, Mv, d)
        # Solve along v (transpose to group columns)
        Yp = np.transpose(Y, (1, 0, 2)).reshape(Mv, Nu * d)   # (Mv, Nu*d)
        Zp = np.linalg.solve(Bv, Yp)                          # (Nv, Nu*d)
        Z = Zp.reshape(Nv, Nu, d).transpose(1, 0, 2)          # (Nu, Nv, d)
        return Z


# ----------------------------- Main entry point ------------------------------

def derivative_nurbs_surface(
    surf: NURBSSurfaceTuple
) -> Tuple[NURBSSurfaceTuple, NURBSSurfaceTuple]:
    """
    Exact partial derivatives of a general *rational* NURBS surface, returned as
    two rational NURBS surfaces S_u and S_v (no Bézier decomposition).

    Input:
        surf.order_u = p+1, surf.order_v = q+1
        surf.knot_u  = U  (len U = Nu + p + 1)
        surf.knot_v  = V  (len V = Nv + q + 1)
        surf.control_points = (Nu, Nv, dim)
        surf.weights        = (Nu, Nv)

    Output:
        (Su, Sv) where each is a NURBSSurfaceTuple with:
          - order_u = 2p + 1, order_v = 2q + 1
          - knot_u, knot_v = refined final knot vectors (shared)
          - control_points, weights representing S_u and S_v exactly.
    """
    U = np.asarray(surf.knot_u, dtype=float)
    V = np.asarray(surf.knot_v, dtype=float)
    P = np.asarray(surf.control_points, dtype=float)   # (Nu, Nv, dim)
    w = np.asarray(surf.weights, dtype=float)          # (Nu, Nv)
    p = surf.order_u - 1
    q = surf.order_v - 1

    if P.ndim != 3:
        raise ValueError("control_points must be (Nu, Nv, dim).")
    Nu_cp, Nv_cp, dim = P.shape

    # Basic consistency checks
    if w.shape != (Nu_cp, Nv_cp):
        raise ValueError("weights shape must match control_points' first two dims.")
    if len(U) != (Nu_cp + p + 1):
        raise ValueError("knot_u length must be Nu + p + 1.")
    if len(V) != (Nv_cp + q + 1):
        raise ValueError("knot_v length must be Nv + q + 1.")
    if p < 1 or q < 1:
        raise ValueError("Surface degrees must be >= 1.")

    # Homogeneous (numerator) nets
    X = P * w[..., None]  # (Nu, Nv, dim)

    # First derivatives of polynomial nets (control nets and 1D knot vectors)
    Xu, Ud_u, pd = _surface_derivative_ctrl_u(X, U, p)   # (Nu-1, Nv, dim), U' (U[1:-1])
    Xv, Vd_v, qd = _surface_derivative_ctrl_v(X, V, q)   # (Nu, Nv-1, dim), V' (V[1:-1])
    wu, Ud_w, _  = _surface_derivative_ctrl_u(w, U, p)   # (Nu-1, Nv)
    wv, Vd_w, _  = _surface_derivative_ctrl_v(w, V, q)   # (Nu, Nv-1)
    # (sanity)
    if not (np.allclose(Ud_u, Ud_w) and np.allclose(Vd_v, Vd_w)):
        raise RuntimeError("Internal: derivative knot vectors mismatch.")

    # Final shared knot vectors (degree 2p, 2q) for both partials
    U_final = _final_derivative_knot_vector(U, p)
    V_final = _final_derivative_knot_vector(V, q)
    pu_final = 2 * p
    pv_final = 2 * q

    # Collocation grid (Greville abscissae) in the final space
    xi_u = _greville_abscissae(U_final, pu_final)   # length Nu_final
    xi_v = _greville_abscissae(V_final, pv_final)   # length Nv_final

    # Basis matrices for evaluating original/derived nets at final Greville grid
    Bu_p   = _basis_matrix(U,     p,  xi_u)   # (Nu_final, Nu)
    Bv_q   = _basis_matrix(V,     q,  xi_v)   # (Nv_final, Nv)
    Bu_pd  = _basis_matrix(Ud_u,  pd, xi_u)   # (Nu_final, Nu-1)
    Bv_qd  = _basis_matrix(Vd_v,  qd, xi_v)   # (Nv_final, Nv-1)
    Bu_fin = _basis_matrix(U_final, pu_final, xi_u)  # square (Nu_final, Nu_final)
    Bv_fin = _basis_matrix(V_final, pv_final, xi_v)  # square (Nv_final, Nv_final)

    # Evaluate the components on the collocation grid
    w_val   = _eval_surface(w,  Bu_p,  Bv_q)           # (Nu_final, Nv_final)
    X_val   = _eval_surface(X,  Bu_p,  Bv_q)           # (Nu_final, Nv_final, dim)
    wu_val  = _eval_surface(wu, Bu_pd, Bv_q)           # (Nu_final, Nv_final)
    Xu_val  = _eval_surface(Xu, Bu_pd, Bv_q)           # (Nu_final, Nv_final, dim)
    wv_val  = _eval_surface(wv, Bu_p,  Bv_qd)          # (Nu_final, Nv_final)
    Xv_val  = _eval_surface(Xv, Bu_p,  Bv_qd)          # (Nu_final, Nv_final, dim)

    # Samples for numerator/denominator of the rational derivatives
    denom_samples = w_val ** 2                                         # (..)
    numer_u_samp  = Xu_val * w_val[..., None] - X_val * wu_val[..., None]
    numer_v_samp  = Xv_val * w_val[..., None] - X_val * wv_val[..., None]

    # Solve for control nets in the final space (two-sided square solves)
    C_den   = _two_sided_solve(Bu_fin, Bv_fin, denom_samples)          # (NuF, NvF)
    C_num_u = _two_sided_solve(Bu_fin, Bv_fin, numer_u_samp)           # (NuF, NvF, dim)
    C_num_v = _two_sided_solve(Bu_fin, Bv_fin, numer_v_samp)           # (NuF, NvF, dim)

    # Convert homogeneous derivative nets back to (P, w)
    eps = np.finfo(float).eps
    w_u = C_den
    P_u = C_num_u / (w_u[..., None] + eps)

    w_v = C_den
    P_v = C_num_v / (w_v[..., None] + eps)

    Su = NURBSSurfaceTuple(
        order_u=pu_final + 1,
        order_v=pv_final + 1,
        knot_u=U_final,
        knot_v=V_final,
        control_points=P_u,
        weights=w_u
    )
    Sv = NURBSSurfaceTuple(
        order_u=pu_final + 1,
        order_v=pv_final + 1,
        knot_u=U_final,
        knot_v=V_final,
        control_points=P_v,
        weights=w_v
    )
    return Su, Sv

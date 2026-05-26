import numpy as np


import math
from typing import List, Sequence, Tuple, Literal, Union, Optional


from mmcore.geom._nurbs_eval import NURBSSurfaceTuple, NURBSCurveTuple
from mmcore.geom._nurbs_knots import (
    split_curve_multiple,
    link_curves,
    make_curves_compatible_multiple,
)
from mmcore.numeric.approx import adaptive_curve_sampler
__all__=['sweep1']
# ------------------------------------------------------------
# NURBS utilities (The NURBS Book)
# ------------------------------------------------------------


def _validate_curve(curve: NURBSCurveTuple, name: str = "curve"):
    U = np.asarray(curve.knot, dtype=float)
    ncp = len(curve.control_points)
    order = int(curve.order)
    expected = ncp + order
    if len(U) != expected:
        raise ValueError(
            f"{name}: invalid knot length. len(knot)={len(U)} but expected n_cp + order = {expected} " f"(n_cp={ncp}, order={order})."
        )


def _find_span(n: int, p: int, u: float, U: np.ndarray) -> int:
    if u >= U[n + 1]:
        return n
    if u <= U[p]:
        return p
    low, high = p, n + 1
    mid = (low + high) // 2
    while (u < U[mid]) or (u >= U[mid + 1]):
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def _basis_funs(i: int, u: float, p: int, U: np.ndarray) -> np.ndarray:
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
            val = 0.0 if denom == 0.0 else N[r] / denom
            tmp = val * right[r + 1]
            N[r] = saved + tmp
            saved = val * left[j - r]
        N[j] = saved
    return N


def _ders_basis_funs(i: int, u: float, p: int, nder: int, U: np.ndarray) -> np.ndarray:
    ndu = np.zeros((p + 1, p + 1), dtype=float)
    left = np.zeros(p + 1, dtype=float)
    right = np.zeros(p + 1, dtype=float)
    ndu[0, 0] = 1.0
    for j in range(1, p + 1):
        left[j] = u - U[i + 1 - j]
        right[j] = U[i + j] - u
        saved = 0.0
        for r in range(j):
            ndu[j, r] = right[r + 1] + left[j - r]
            val = 0.0 if ndu[j, r] == 0.0 else ndu[r, j - 1] / ndu[j, r]
            ndu[r, j] = saved + right[r + 1] * val
            saved = left[j - r] * val
        ndu[j, j] = saved

    ders = np.zeros((nder + 1, p + 1), dtype=float)
    for j in range(p + 1):
        ders[0, j] = ndu[j, p]

    a = np.zeros((2, p + 1), dtype=float)
    for r in range(p + 1):
        s1, s2 = 0, 1
        a[0, 0] = 1.0
        for k in range(1, nder + 1):
            d = 0.0
            rk = r - k
            pk = p - k
            if r >= k:
                a[s2, 0] = a[s1, 0] / ndu[pk + 1, rk]
                d = a[s2, 0] * ndu[rk, pk]
            j1 = 1 if rk >= -1 else -rk
            j2 = k - 1 if (r - 1) <= pk else p - r
            for j in range(j1, j2 + 1):
                a[s2, j] = (a[s1, j] - a[s1, j - 1]) / ndu[pk + 1, rk + j]
                d += a[s2, j] * ndu[rk + j, pk]
            if r <= pk:
                a[s2, k] = -a[s1, k - 1] / ndu[pk + 1, r]
                d += a[s2, k] * ndu[r, pk]
            ders[k, r] = d
            s1, s2 = s2, s1

    rfact = float(p)
    for k in range(1, nder + 1):
        for j in range(p + 1):
            ders[k, j] *= rfact
        rfact *= p - k
    return ders


def _curve_eval_rational(curve: NURBSCurveTuple, u: float, der_order: int = 2):
    """Return [C, C', C''] (up to der_order) for a rational curve."""
    p = curve.order - 1
    U = np.asarray(curve.knot, dtype=float)
    Pw = np.concatenate([np.asarray(curve.control_points, dtype=float) * curve.weights[:, None], curve.weights[:, None]], axis=1)  # (n+1,4)
    n = Pw.shape[0] - 1
    uc = float(np.clip(u, U[p], U[n + 1] - 1e-14))
    span = _find_span(n, p, uc, U)
    dersN = _ders_basis_funs(span, uc, p, min(der_order, 2), U)

    CKw = [np.zeros(4, dtype=float) for _ in range(min(der_order, 2) + 1)]
    for k in range(len(CKw)):
        for j in range(p + 1):
            idx = span - p + j
            CKw[k] += dersN[k, j] * Pw[idx]

    C = [np.zeros(3, dtype=float) for _ in range(der_order + 1)]
    w0 = CKw[0][3]
    cw0 = CKw[0][:3]
    C[0] = cw0 / w0
    if der_order >= 1:
        w1 = CKw[1][3]
        cw1 = CKw[1][:3]
        C[1] = (cw1 - C[0] * w1) / w0
    if der_order >= 2:
        w1 = CKw[1][3]
        cw1 = CKw[1][:3]
        w2 = CKw[2][3]
        cw2 = CKw[2][:3]
        C[2] = (cw2 - 2.0 * C[1] * w1 - C[0] * w2) / w0
    return C


def _build_collocation_matrix(U: np.ndarray, p: int, params: np.ndarray) -> np.ndarray:
    m = len(U) - 1
    n = m - p - 1
    A = np.zeros((len(params), n + 1), dtype=float)
    for r, u in enumerate(params):
        uc = float(np.clip(u, U[p], U[n + 1] - 1e-14))
        span = _find_span(n, p, uc, U)
        N = _basis_funs(span, uc, p, U)
        i0 = span - p
        A[r, i0 : i0 + p + 1] = N
    return A


def _greville(U: np.ndarray, p: int) -> np.ndarray:
    m = len(U) - 1
    n = m - p - 1
    if p == 0:
        return 0.5 * (U[: n + 1] + U[1 : n + 2])
    xi = np.empty(n + 1, dtype=float)
    for i in range(n + 1):
        xi[i] = np.sum(U[i + 1 : i + p + 1]) / p
    return xi


# ------------------------------------------------------------
# Geometry helpers
# ------------------------------------------------------------


def _normalize(v: np.ndarray, eps: float = 1e-15) -> Tuple[np.ndarray, float]:
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy(), 0.0
    return v / n, n


def _closest_parameter_on_curve(
    curve: NURBSCurveTuple, P: np.ndarray, n_samples: int = 200, newton_iters: int = 20, tol: float = 1e-12
) -> float:
    p = curve.order - 1
    U = np.asarray(curve.knot, dtype=float)
    n = len(curve.control_points) - 1
    umin, umax = U[p], U[n + 1]
    us = np.linspace(umin, umax, n_samples)
    d2 = np.empty_like(us)
    for i, uu in enumerate(us):
        C0 = _curve_eval_rational(curve, float(uu), der_order=0)[0]
        d2[i] = np.dot(C0 - P, C0 - P)
    u = float(us[np.argmin(d2)])
    for _ in range(newton_iters):
        C0, C1, C2 = _curve_eval_rational(curve, u, der_order=2)[:3]
        r = C0 - P
        f = float(np.dot(r, C1))
        g = float(np.dot(C1, C1) + np.dot(r, C2))
        if abs(g) < 1e-18:
            break
        du = -f / g
        if abs(du) < tol * (umax - umin):
            break
        u = float(np.clip(u + du, umin, umax))
    return u


def _profile_anchor_point(profile: NURBSCurveTuple, spec: Union[str, Tuple[str, float]]) -> np.ndarray:
    Q = np.asarray(profile.control_points, dtype=float)
    if isinstance(spec, str):
        if spec == "centroid":
            return Q.mean(axis=0)
        if spec == "first_cp":
            return Q[0].copy()
        raise ValueError("anchors must be 'centroid', 'first_cp', or ('curve_param', v).")
    if isinstance(spec, tuple) and len(spec) == 2 and spec[0] == "curve_param":
        vparam = float(spec[1])
        return _curve_eval_rational(profile, vparam, der_order=0)[0]
    raise ValueError("anchors must be 'centroid', 'first_cp', or ('curve_param', v).")


# ------------------------------------------------------------
# Framing along rail: TNB (Frenet) with RMF fallback
# ------------------------------------------------------------


def _frames_along_params(
    rail: NURBSCurveTuple, params: np.ndarray, mode: Literal["TNB", "RMF"] = "TNB"
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """
    Return lists Ck, Nk, Bk, Tk along given parameters.
    - If mode="RMF": pure parallel transport (rotation-minimizing).
    - If mode="TNB": Frenet when κ>eps; otherwise fallback to parallel transport step.
    Ensures continuity by flipping (N,B) when necessary.
    """
    Ck: List[np.ndarray] = []
    Tk: List[np.ndarray] = []
    Nk: List[np.ndarray] = []
    Bk: List[np.ndarray] = []

    # first point
    C0, C1, C2 = _curve_eval_rational(rail, float(params[0]), der_order=2)[:3]
    T0, s0 = _normalize(C1)
    if s0 < 1e-14:
        # use small finite difference direction
        U = np.asarray(rail.knot, dtype=float)
        p = rail.order - 1
        umin, umax = U[p], U[-(p + 1)]
        du = max(1e-6 * (umax - umin), 1e-9)
        P1 = _curve_eval_rational(rail, min(umax, params[0] + du), 0)[0]
        T0, _ = _normalize(P1 - C0)
    if mode == "TNB":
        # N from second derivative, projected orthogonal to T
        N0_raw = C2 - np.dot(C2, T0) * T0
        N0, k0 = _normalize(N0_raw)
        if k0 < 1e-12:
            # fallback to any perpendicular for start; RMF will transport it
            cand = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(cand, T0)) > 0.9:
                cand = np.array([1.0, 0.0, 0.0])
            N0 = cand - np.dot(cand, T0) * T0
            N0, _ = _normalize(N0)
    else:
        # RMF start: choose arbitrary perpendicular
        cand = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(cand, T0)) > 0.9:
            cand = np.array([1.0, 0.0, 0.0])
        N0 = cand - np.dot(cand, T0) * T0
        N0, _ = _normalize(N0)

    B0 = np.cross(T0, N0)
    B0, _ = _normalize(B0)
    N0 = np.cross(B0, T0)  # re-orthonormalize

    Ck.append(C0)
    Tk.append(T0)
    Nk.append(N0)
    Bk.append(B0)

    for idx in range(1, len(params)):
        u = float(params[idx])
        C, C1, C2 = _curve_eval_rational(rail, u, der_order=2)[:3]
        T, s = _normalize(C1)
        if s < 1e-14:
            # finite difference
            U = np.asarray(rail.knot, dtype=float)
            p = rail.order - 1
            umin, umax = U[p], U[-(p + 1)]
            du = max(1e-6 * (umax - umin), 1e-9)
            P1 = _curve_eval_rational(rail, min(umax, u + du), 0)[0]
            T, _ = _normalize(P1 - Ck[-1])

        if mode == "RMF":
            # Parallel transport previous N along rotation from T_prev→T
            t_prev = Tk[-1]
            n_prev = Nk[-1]
            axis = np.cross(t_prev, T)
            sinang = np.linalg.norm(axis)
            cosang = float(np.clip(np.dot(t_prev, T), -1.0, 1.0))
            if sinang < 1e-14 and cosang > 0.0:
                n_par = n_prev
            else:
                axis, _ = _normalize(axis)
                angle = math.atan2(sinang, cosang)
                K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=float)
                I = np.eye(3)
                R = I + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
                n_par = R @ n_prev
            n_par = n_par - np.dot(n_par, T) * T
            N, _ = _normalize(n_par)
            B, _ = _normalize(np.cross(T, N))
            N = np.cross(B, T)
        else:
            # TNB when κ>eps, else RMF step
            N_raw = C2 - np.dot(C2, T) * T
            N_try, kappa = _normalize(N_raw)
            if kappa < 1e-12:
                # RMF fallback
                t_prev = Tk[-1]
                n_prev = Nk[-1]
                axis = np.cross(t_prev, T)
                sinang = np.linalg.norm(axis)
                cosang = float(np.clip(np.dot(t_prev, T), -1.0, 1.0))
                if sinang < 1e-14 and cosang > 0.0:
                    n_par = n_prev
                else:
                    axis, _ = _normalize(axis)
                    angle = math.atan2(sinang, cosang)
                    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=float)
                    I = np.eye(3)
                    R = I + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
                    n_par = R @ n_prev
                N, _ = _normalize(n_par - np.dot(n_par, T) * T)
            else:
                N = N_try
            B, _ = _normalize(np.cross(T, N))
            N = np.cross(B, T)
            # flip to keep continuity
            if np.dot(N, Nk[-1]) < 0.0:
                N = -N
                B = -B

        Ck.append(C)
        Tk.append(T)
        Nk.append(N)
        Bk.append(B)

    return Ck, Nk, Bk, Tk


# ------------------------------------------------------------
# Anchor assignment to Greville set (multi-profile)
# ------------------------------------------------------------


def _assign_anchors_to_params(params_u: np.ndarray, anchors_u: np.ndarray) -> Tuple[dict, np.ndarray]:
    """
    Map each anchor parameter to a unique index in params_u (closest available index).
    Returns (index_map, updated_params_u_with_replacements).
    """
    taken = np.zeros(len(params_u), dtype=bool)
    idx_map = {}
    params_new = params_u.copy()
    for a in anchors_u:
        diffs = np.abs(params_new - a)
        diffs[taken] = np.inf
        k = int(np.argmin(diffs))
        params_new[k] = a  # replace so collocation includes the anchor exactly
        taken[k] = True
        idx_map[k] = a
    return idx_map, params_new


from mmcore.geom._nurbs_eval import from_homogeneous_1d, to_homogeneous_1d


# ------------------------------------------------------------
# Main: simplified, multi-profile Sweep1
# ------------------------------------------------------------
def _twin_curves_compat(crv1: NURBSCurveTuple, crv2: NURBSCurveTuple, t: float):
    homo = (to_homogeneous_1d(crv1.control_points, crv1.weights) + to_homogeneous_1d(crv2.control_points, crv2.weights)) / 2
    pts, w = from_homogeneous_1d(homo)
    return crv1._replace(control_points=pts, weights=w)


from mmcore.numeric.closest_point import nurbs_curve_closest_point


def _build_arclength_table(rail: NURBSCurveTuple, tol: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build a monotone param->arclength lookup from adaptive sampling.
    Returns (params_samp, s_cum) with s_cum[0]=0 and len = len(params_samp).
    """
    params_samp, du_list, evals, s_list = adaptive_curve_sampler(rail, tol=tol, max_param_step_fraction=1, max_points=int(1e6))
    params_samp = np.asarray(params_samp, dtype=float)
    s_cum = np.zeros(len(params_samp), dtype=float)
    if len(params_samp) >= 2:
        s_incr = np.asarray(s_list, dtype=float)
        s_cum[1:] = np.cumsum(s_incr)
    return params_samp, s_cum


def _arc_length_at(u: float, params_samp: np.ndarray, s_cum: np.ndarray) -> float:
    """
    Approximate arclength from start to parameter u by linear interpolation
    on the (params_samp, s_cum) table.
    """
    # clamp
    if u <= params_samp[0]:
        return float(s_cum[0])
    if u >= params_samp[-1]:
        return float(s_cum[-1])
    i = int(np.searchsorted(params_samp, u, side="right") - 1)
    u0, u1 = params_samp[i], params_samp[i + 1]
    s0, s1 = s_cum[i], s_cum[i + 1]
    t = 0.0 if u1 == u0 else (u - u0) / (u1 - u0)
    return float((1.0 - t) * s0 + t * s1)


def _linear_blend_weights_by_arclength(anchors_u: np.ndarray, u: float, params_samp: np.ndarray, s_cum: np.ndarray) -> np.ndarray:
    """
    Piecewise-linear 'hat' weights w_i(u) based on rail arclength,
    with nonzero support only on the bracketing anchors.
    """
    M = len(anchors_u)
    w = np.zeros(M, dtype=float)
    if M == 1:
        w[0] = 1.0
        return w
    # precompute arclengths at anchors
    # (cache at a higher scope if you call this repeatedly)
    sA = np.array([_arc_length_at(a, params_samp, s_cum) for a in anchors_u], dtype=float)
    su = _arc_length_at(u, params_samp, s_cum)

    if su <= sA[0]:
        w[0] = 1.0
        return w
    if su >= sA[-1]:
        w[-1] = 1.0
        return w

    # find i such that sA[i] <= su <= sA[i+1]
    i = int(np.searchsorted(sA, su, side="right") - 1)
    s0, s1 = sA[i], sA[i + 1]
    denom = s1 - s0
    if denom <= 1e-18:
        # Anchors coincide in arclength; pick the closer one in parameter space
        j = i if abs(anchors_u[i] - u) <= abs(anchors_u[i + 1] - u) else i + 1
        w[j] = 1.0
        return w
    t = (su - s0) / denom
    w[i] = 1.0 - t
    w[i + 1] = t
    return w


# ---------- Refined Sweep: multi-profile with even arclength morphing ----------


def sweep1(
    rail: NURBSCurveTuple,
    profiles: Union[NURBSCurveTuple, Sequence[NURBSCurveTuple]],
    *,
    params: Optional[Sequence[float]] = None,
    anchors: Union[Literal["centroid", "first_cp"], Sequence[Union[Literal["centroid", "first_cp"], Tuple[str, float]]]] = "centroid",
    sampler_tol: float = 0.1,
    frame: Literal["TNB", "RMF"] = "TNB",
) -> NURBSSurfaceTuple:
    """
    Sweep one or more NURBS profile curves along a NURBS rail with continuous
    shape morphing between profiles, blended by rail arclength.

    - u-direction: inherited from (refined) rail; surface interpolates exact sections
      at the rail's Greville abscissae, with those at the given profile parameters
      equal to the profile geometry (pinned).
    - v-direction: unified across all profiles via make_curves_compatible_multiple.

    Parameters
    ----------
    rail : NURBSCurveTuple
    profiles : NURBSCurveTuple | list[NURBSCurveTuple]
        One or many cross-sections in world space.
    params : list[float] | None
        Rail parameters for each profile. If None, each profile is projected to the rail
        using `anchors`.
    anchors : "centroid" | "first_cp" | list[("curve_param", v)]
        How to choose the projection point per profile when `params` is None.
    sampler_tol : float
        Chord-height tolerance used to refine the rail (Rhino-like tightness).
    frame : "TNB" | "RMF"
        Framing along the rail: Frenet with RMF fallback ("TNB") or pure RMF.

    Returns
    -------
    NURBSSurfaceTuple
    """
    # ---- Validate rail ----
    _validate_curve(rail, "rail")

    # ---- Normalize profiles list and ensure v-compatibility ----
    if isinstance(profiles, NURBSCurveTuple):
        profiles_list = [profiles]
    else:
        profiles_list = list(profiles)
    if len(profiles_list) == 0:
        raise ValueError("profiles must contain at least one NURBSCurveTuple.")

    profiles_list = make_curves_compatible_multiple(profiles_list)
    order_v = profiles_list[0].order
    knot_v = np.asarray(profiles_list[0].knot, dtype=float)
    n_v = len(profiles_list[0].control_points) - 1
    for c in profiles_list[1:]:
        if c.order != order_v or not np.allclose(c.knot, knot_v):
            raise RuntimeError("make_curves_compatible_multiple failed to equalize v-structure.")

    # ---- Determine anchor parameters for profiles ----
    Uu0 = np.asarray(rail.knot, dtype=float)
    p_u0 = rail.order - 1
    umin0, umax0 = Uu0[p_u0], Uu0[-(p_u0 + 1)]

    if params is not None:
        if len(params) != len(profiles_list):
            raise ValueError("params length must match number of profiles.")
        anchors_u = np.clip(np.asarray(params, dtype=float), umin0, umax0 - 1e-14)
    else:
        # single spec for all, or per-profile list
        if isinstance(anchors, (str, tuple)):
            anchors_specs = [anchors] * len(profiles_list)
        else:
            if len(anchors) != len(profiles_list):
                raise ValueError("anchors must be a single spec or a list matching profiles.")
            anchors_specs = list(anchors)
        anchors_u = []
        for curve, spec in zip(profiles_list, anchors_specs):
            P = _profile_anchor_point(curve, spec)
            prm, *_ = nurbs_curve_closest_point(rail, P)
            anchors_u.append(prm)
        anchors_u = np.clip(np.asarray(anchors_u, dtype=float), umin0, umax0 - 1e-14)

    # Sort profiles by anchor parameter (enforce strictly increasing)
    sort_idx = np.argsort(anchors_u)
    anchors_u = anchors_u[sort_idx]
    profiles_list = [profiles_list[i] for i in sort_idx]

    # Check for duplicates (within tolerance) — cannot pin two different profiles at same u
    if np.any(np.diff(anchors_u) < 1e-12):
        raise ValueError("Anchor parameters must be strictly increasing (no duplicates).")

    # ---- Build arclength lookup for even morphing ----
    params_samp, s_cum = _build_arclength_table(rail, sampler_tol)

    # ---- Rail refinement: split by adaptive samples and all anchor params ----
    params_refine = np.unique(np.concatenate([params_samp, anchors_u]))
    tmin, tmax = rail.interval()
    internal = params_refine[(params_refine > tmin + 1e-14) & (params_refine < tmax - 1e-14)]
    rail_refined, _ = link_curves(split_curve_multiple(rail, internal.tolist()))
    rail = rail_refined
    _validate_curve(rail, "rail(refined)")

    # ---- u-structure / Greville and inject anchors ----
    Uu = np.asarray(rail.knot, dtype=float)
    p_u = rail.order - 1
    n_u = (len(Uu) - 1) - p_u - 1
    grev = _greville(Uu, p_u)

    # Replace nearest Greville(s) by anchor params so collocation contains them exactly
    taken = np.zeros_like(grev, dtype=bool)
    params_u = grev.copy()
    for a in anchors_u:
        k = int(np.argmin(np.where(taken, np.inf, np.abs(params_u - a))))
        params_u[k] = a
        taken[k] = True
    params_u = np.clip(params_u, Uu[p_u], Uu[-(p_u + 1)] - 1e-14)

    # ---- Frames at params_u ----
    Ck, Nk, Bk, Tk = _frames_along_params(rail, params_u, mode=frame)

    # ---- Precompute local control nets (and weights) for each anchor profile ----
    M = len(profiles_list)  # number of anchor profiles
    Qrel_list: List[np.ndarray] = []
    wv_list: List[np.ndarray] = []
    for i in range(M):
        # express profile i in its own anchor frame (at u = anchors_u[i])
        # but we only need local coordinates; for morphing we will blend them
        # and then place into the *target* frame at each u_k.
        # To get consistent local coords, we use the frame taken at the
        # collocation u which equals the anchor param (we forced it above).
        # Find index k where params_u == anchors_u[i] (within tol)
        k_match = int(np.argmin(np.abs(params_u - anchors_u[i])))
        R_ai = np.column_stack([Nk[k_match], Bk[k_match], Tk[k_match]])  # local->world
        C_ai = np.array(Ck[k_match], dtype=float)

        Qi = np.asarray(profiles_list[i].control_points, dtype=float)
        wi = np.asarray(profiles_list[i].weights, dtype=float)
        # world->local in anchor frame
        Qrel_i = (Qi - C_ai) @ R_ai
        Qrel_list.append(Qrel_i)  # (n_v+1, 3)
        wv_list.append(wi)  # (n_v+1,)

    # ---- Build sections at each params_u by blending profiles in *homogeneous* coords ----
    sections_ctrl = np.empty((n_u + 1, n_v + 1, 3), dtype=float)
    sections_wts = np.empty((n_u + 1, n_v + 1), dtype=float)

    for k, u in enumerate(params_u):
        # Weights over profiles by arclength (nonzero only on bracketing anchors)
        alpha = _linear_blend_weights_by_arclength(anchors_u, float(u), params_samp, s_cum)  # (M,)
        Rk = np.column_stack([Nk[k], Bk[k], Tk[k]])  # local->world
        C = Ck[k]

        # Blend per v-index in homogeneous coords
        # H_blend = sum_i alpha_i * [Qrel_i[j]*w_i[j], w_i[j]]
        # Then Cartesian local = H[:3]/H[3], and world = C + local @ Rk^T
        for j in range(n_v + 1):
            denom = 0.0
            numer = np.zeros(3, dtype=float)
            for i in range(M):
                wi = wv_list[i][j]
                if wi == 0.0 or alpha[i] == 0.0:
                    continue
                numer += alpha[i] * (Qrel_list[i][j] * wi)
                denom += alpha[i] * wi
            if denom <= 1e-18:
                # extremely unlikely with standard NURBS (weights>0), but guard anyway
                # fall back to simple average in Cartesian
                qavg = np.zeros(3, dtype=float)
                wsum = 0.0
                for i in range(M):
                    qavg += alpha[i] * Qrel_list[i][j]
                    wsum += alpha[i]
                qloc = qavg / max(wsum, 1e-18)
                wloc = 1.0
            else:
                qloc = numer / denom
                wloc = denom
            # map to world
            sections_ctrl[k, j, :] = C + qloc @ Rk.T
            sections_wts[k, j] = wloc

    # ---- Global interpolation in u (homogeneous), one v-column at a time ----
    Au = _build_collocation_matrix(Uu, p_u, params_u)  # (n_u+1, n_u+1)
    ctrl_homo_u = np.empty((n_u + 1, n_v + 1, 4), dtype=float)

    for j in range(n_v + 1):
        Y = np.empty((n_u + 1, 4), dtype=float)
        Y[:, 0:3] = sections_ctrl[:, j, :] * sections_wts[:, j][:, None]
        Y[:, 3] = sections_wts[:, j]
        try:
            H = np.linalg.solve(Au, Y)
        except np.linalg.LinAlgError:
            H = np.linalg.lstsq(Au, Y, rcond=None)[0]
        ctrl_homo_u[:, j, :] = H

    # ---- Cartesian control net + weights ----
    weights = ctrl_homo_u[:, :, 3].copy()
    epsw = 1e-14
    weights = np.where(np.abs(weights) < epsw, np.sign(weights) * epsw, weights)
    control_points = ctrl_homo_u[:, :, 0:3] / weights[:, :, None]

    # ---- Assemble surface ----
    return NURBSSurfaceTuple(order_u=rail.order, order_v=order_v, knot_u=Uu, knot_v=knot_v, control_points=control_points, weights=weights)


# ------------------------------------------------------------
# (Optional) Simple example usage
# ------------------------------------------------------------


# Example: sweep a rectangular profile along a 3D rail
# (adjust to your data structures as needed)

# Rail: a cubic open curve

# Build swept surface
'''
a = sweep1(c1, c2, anchors="first_cp", frame="RMF")
'''


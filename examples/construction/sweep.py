import numpy as np
import math

from typing import Literal

from mmcore.geom._nurbs_eval import NURBSSurfaceTuple, NURBSCurveTuple
from mmcore.geom._nurbs_knots import split_curve_multiple, link_curves


# ------------ NURBS core utilities (The NURBS Book) ------------


def _find_span(n, p, u, U):
    if u >= U[n + 1]:
        return n
    if u <= U[p]:
        return p
    low, high = p, n + 1
    mid = (low + high) // 2
    while u < U[mid] or u >= U[mid + 1]:
        if u < U[mid]:
            high = mid
        else:
            low = mid
        mid = (low + high) // 2
    return mid


def _basis_funs(i, u, p, U):
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
            temp = val * right[r + 1]
            N[r] = saved + temp
            saved = val * left[j - r]
        N[j] = saved
    return N


def _ders_basis_funs(i, u, p, nder, U):
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


def _curve_point_and_deriv_rational(curve: NURBSCurveTuple, u: float, der_order: int = 2):
    p = curve.order - 1
    U = np.asarray(curve.knot, dtype=float)
    Pw = np.concatenate([np.asarray(curve.control_points, dtype=float) * curve.weights[:, None], curve.weights[:, None]], axis=1)  # (n+1,4)
    n = Pw.shape[0] - 1
    # clamp u
    u = float(np.clip(u, U[p], U[n + 1] - 1e-14))
    span = _find_span(n, p, u, U)
    dersN = _ders_basis_funs(span, u, p, min(der_order, 2), U)  # up to 2 is enough here

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


def _greville_abscissae(U, p, n):
    if p == 0:
        # midpoints
        return 0.5 * (U[: n + 1] + U[1 : n + 2])
    xi = np.empty(n + 1, dtype=float)
    for i in range(n + 1):
        xi[i] = np.sum(U[i + 1 : i + p + 1]) / p
    return xi


def _validate_curve(curve, name="curve"):
    U = np.asarray(curve.knot, dtype=float)
    ncp = len(curve.control_points)
    order = int(curve.order)
    expected = ncp + order
    if len(U) != expected:
        raise ValueError(f"{name}: invalid knot length, got {len(U)} but need n_cp + order = {expected} " f"(n_cp={ncp}, order={order}).")


# ------------ Collocation matrix for arbitrary parameter set ------------


def _build_collocation_matrix(U, p, params):
    m = len(U) - 1
    n = m - p - 1  # n+1 control points
    A = np.zeros((len(params), n + 1), dtype=float)
    for r, u in enumerate(params):
        uc = float(np.clip(u, U[p], U[n + 1] - 1e-14))
        span = _find_span(n, p, uc, U)
        N = _basis_funs(span, uc, p, U)
        i0 = span - p
        A[r, i0 : i0 + p + 1] = N
    return A


# ------------ Closest parameter on rail to a point (robust) ------------


def _closest_parameter_on_curve(curve: NURBSCurveTuple, P, n_samples=200, newton_iters=20, tol=1e-12):
    p = curve.order - 1
    U = np.asarray(curve.knot, dtype=float)
    n = len(curve.control_points) - 1
    umin, umax = U[p], U[n + 1]
    us = np.linspace(umin, umax, n_samples)
    d2 = np.empty_like(us)
    for i, uu in enumerate(us):
        C0 = _curve_point_and_deriv_rational(curve, float(uu), der_order=0)[0]
        d2[i] = np.dot(C0 - P, C0 - P)
    u = float(us[np.argmin(d2)])
    for _ in range(newton_iters):
        C0, C1, C2 = _curve_point_and_deriv_rational(curve, u, der_order=2)[:3]
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


# ------------ Framing (parallel transport + user twist, relative to anchor) ------------


def _normalize(v, eps=1e-15):
    n = np.linalg.norm(v)
    if n < eps:
        return v.copy(), 0.0
    return v / n, n


def _parallel_transport_frames(rail: NURBSCurveTuple, params, up=None, twist_fn=None):
    p = rail.order - 1
    U = np.asarray(rail.knot, dtype=float)
    umin, umax = U[p], U[-(p + 1)]

    C_list, T_list = [], []
    for u in params:
        C0, C1 = _curve_point_and_deriv_rational(rail, u, der_order=1)[:2]
        t, n = _normalize(C1)
        if n < 1e-14:
            du = max(1e-6 * (umax - umin), 1e-9)
            u1 = min(umax, u + du)
            u0 = max(umin, u - du)
            P1 = _curve_point_and_deriv_rational(rail, u1, der_order=0)[0]
            P0 = _curve_point_and_deriv_rational(rail, u0, der_order=0)[0]
            t, _ = _normalize(P1 - P0)
        C_list.append(C0)
        T_list.append(t)

    if up is None:
        t0 = T_list[0]
        cand = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(cand, t0)) > 0.9:
            cand = np.array([1.0, 0.0, 0.0])
        up = cand
    up = np.asarray(up, dtype=float)
    t0 = T_list[0]
    n0 = up - np.dot(up, t0) * t0
    n0, ln = _normalize(n0)
    if ln < 1e-14:
        n0 = np.cross(t0, np.array([1.0, 0.0, 0.0])) if abs(t0[0]) < 0.9 else np.cross(t0, np.array([0.0, 1.0, 0.0]))
        n0, _ = _normalize(n0)

    N_list = [n0]
    B_list = [np.cross(t0, n0)]
    B_list[0], _ = _normalize(B_list[0])

    for k in range(1, len(params)):
        t_prev = T_list[k - 1]
        t_curr = T_list[k]
        n_prev = N_list[-1]
        axis = np.cross(t_prev, t_curr)
        sinang = np.linalg.norm(axis)
        cosang = float(np.clip(np.dot(t_prev, t_curr), -1.0, 1.0))
        if sinang < 1e-14 and cosang > 0.0:
            n_par = n_prev
        else:
            axis, _ = _normalize(axis)
            angle = math.atan2(sinang, cosang)
            K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=float)
            I = np.eye(3)
            R = I + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)
            n_par = R @ n_prev
        n_par = n_par - np.dot(n_par, t_curr) * t_curr
        n_par, _ = _normalize(n_par)
        b_curr = np.cross(t_curr, n_par)
        b_curr, _ = _normalize(b_curr)
        N_list.append(n_par)
        B_list.append(b_curr)

    # Apply user twist (about T), if provided
    if twist_fn is not None:
        for k, u in enumerate(params):
            ang = float(twist_fn(u))
            if abs(ang) > 1e-18:
                t = T_list[k]
                n = N_list[k]
                b = B_list[k]
                cosA, sinA = math.cos(ang), math.sin(ang)
                n_rot = n * cosA + b * sinA
                b_rot = np.cross(t, n_rot)
                b_rot, _ = _normalize(b_rot)
                n_rot = np.cross(b_rot, t)
                n_rot, _ = _normalize(n_rot)
                N_list[k] = n_rot
                B_list[k] = b_rot
    return C_list, N_list, B_list, T_list


# ------------ Helper: anchor point on profile ------------


def _profile_anchor_point(profile: NURBSCurveTuple, profile_anchor):
    Q = np.asarray(profile.control_points, dtype=float)
    if isinstance(profile_anchor, str):
        if profile_anchor == "centroid":
            return Q.mean(axis=0)
        elif profile_anchor == "first_cp":
            return Q[0].copy()
        else:
            raise ValueError("profile_anchor must be 'centroid', 'first_cp', or ('curve_param', v).")
    elif isinstance(profile_anchor, tuple) and len(profile_anchor) == 2 and profile_anchor[0] == "curve_param":
        vparam = float(profile_anchor[1])
        return _curve_point_and_deriv_rational(profile, vparam, der_order=0)[0]
    else:
        raise ValueError("profile_anchor must be 'centroid', 'first_cp', or ('curve_param', v).")


# ------------ New sweep1 (profile pinned as an isoline) ------------


def sweep1(
    rail: NURBSCurveTuple,
    profile: NURBSCurveTuple,
    *,
    mode: Literal["normal", "loose"] = "normal",
    tol=1e-3,
    up=None,
    twist=0.0,  # scalar total twist or callable(u)->angle
    scale=1.0,  # scalar or callable(u)->positive
    profile_anchor="centroid",  # where to 'aim' when auto-anchoring
    u_anchor=None,  # if None: project profile_anchor point to rail
    normalize_scale_at_anchor=True,  # force s(u_anchor)=1 so the isoline matches exactly
) -> NURBSSurfaceTuple:
    """
    Sweep a NURBS profile along a NURBS rail so that one u-isoline EXACTLY matches
    the profile as currently placed in world space.

    - The u-direction inherits rail order & knot vector.
    - The v-direction inherits profile order & knot vector.
    - At u = u_anchor (chosen or auto-projected), S(u, v) equals the rigid (unscaled) profile.

    Parameters
    ----------
    rail, profile : NURBSCurveTuple
    up            : optional 3-vector to initialize the frame
    twist         : float total twist over [umin,umax] (linear) or callable(u)->angle
    scale         : float or callable(u)->scale (>0). If callable and normalize_scale_at_anchor=True,
                    we use s_norm(u) = scale(u) / scale(u_anchor) to keep the anchor section identical.
    profile_anchor: "centroid" | "first_cp" | ("curve_param", v)
                    Used only to *find* u_anchor when u_anchor is None.
    u_anchor      : rail parameter where the surface must pass exactly through the given profile.
    normalize_scale_at_anchor: if True, s(u_anchor)=1 even if 'scale' is not 1 at the anchor.

    Returns
    -------
    NURBSSurfaceTuple
    """
    # ---- Validate curves ----
    _validate_curve(rail, "rail")
    _validate_curve(profile, "profile")
    from mmcore.numeric.approx import adaptive_curve_sampler
    from mmcore.geom._nurbs_knots import link_curves

    if mode == "normal":
        params, *_ = adaptive_curve_sampler(rail, tol, max_param_step_fraction=1)

        rail, _ = link_curves(split_curve_multiple(rail, params[1:][:-1]))
    # ---- Rail / Profile basics ----
    Uu = np.asarray(rail.knot, dtype=float)
    pu = rail.order - 1
    Vv = np.asarray(profile.knot, dtype=float)
    pv = profile.order - 1
    n_u = (len(Uu) - 1) - pu - 1
    n_v = (len(Vv) - 1) - pv - 1

    umin, umax = Uu[pu], Uu[-(pu + 1)]

    # ---- Choose anchor parameter ----
    if u_anchor is None:
        Pa = _profile_anchor_point(profile, profile_anchor)
        u_anchor = _closest_parameter_on_curve(rail, Pa)
    u_anchor = float(np.clip(u_anchor, umin, umax - 1e-14))

    # ---- Pick u-parameters for interpolation: Greville with one value replaced by u_anchor ----
    grev = _greville_abscissae(Uu, pu, n_u)
    # replace the closest Greville by u_anchor to keep matrix square and include the anchor
    k0 = int(np.argmin(np.abs(grev - u_anchor)))
    params_u = grev.copy()
    params_u[k0] = u_anchor
    # ensure strictly increasing (numerical safety)
    params_u = np.clip(params_u, umin, umax - 1e-14)
    params_u.sort()
    # find the (new) index of u_anchor
    anchor_mask = np.isclose(params_u, u_anchor, rtol=0, atol=1e-13)
    if not np.any(anchor_mask):
        # if u_anchor collided with a neighbor numerically, force it in mid position
        k0 = int(np.argmin(np.abs(params_u - u_anchor)))
        params_u[k0] = u_anchor
        anchor_mask = np.isclose(params_u, u_anchor, rtol=0, atol=1e-13)
    k_anchor = int(np.nonzero(anchor_mask)[0][0])

    # ---- Twist function relative to anchor (so twist(u_anchor)=0) ----
    if callable(twist):
        twist_raw = lambda u: float(twist(u))
    else:
        total = float(twist)
        span = (umax - umin) if (umax > umin) else 1.0
        twist_raw = lambda u: total * ((u - umin) / span)
    offset = twist_raw(u_anchor)
    twist_fn = lambda u: twist_raw(u) - offset

    # ---- Build frames at params_u (including the anchor) ----
    Ck, Nk, Bk, Tk = _parallel_transport_frames(rail, params_u, up=up, twist_fn=twist_fn)

    # ---- Anchor frame & scale normalization ----
    C_anchor = np.array(Ck[k_anchor], dtype=float)
    R_anchor = np.column_stack([Nk[k_anchor], Bk[k_anchor], Tk[k_anchor]])  # local->world

    # scale normalization
    def scale_at(u):
        return float(scale(u)) if callable(scale) else float(scale)

    s_anchor = max(scale_at(u_anchor), 1e-18)
    if normalize_scale_at_anchor:
        s_norm = lambda u: scale_at(u) / s_anchor
    else:
        s_norm = scale_at

    # ---- Express profile control net relative to anchor frame ----
    Q = np.asarray(profile.control_points, dtype=float)  # (n_v+1, 3)
    wv = np.asarray(profile.weights, dtype=float)  # (n_v+1,)
    # world->local (rows): (Q - C_anchor) @ R_anchor
    Qrel = (Q - C_anchor) @ R_anchor

    # ---- Build swept sections at each u_k ----
    sections_ctrl = np.empty((n_u + 1, n_v + 1, 3), dtype=float)
    for k, u in enumerate(params_u):
        Rk = np.column_stack([Nk[k], Bk[k], Tk[k]])  # local->world
        PW = (Qrel @ Rk.T) * s_norm(u)  # local->world, then scale
        PW += Ck[k]  # translate to rail center
        sections_ctrl[k, :, :] = PW

    # ---- Global interpolation in u (homogeneous), one v-index at a time ----
    Au = _build_collocation_matrix(Uu, pu, params_u)  # (n_u+1, n_u+1)
    ctrl_homo_u = np.empty((n_u + 1, n_v + 1, 4), dtype=float)
    for j in range(n_v + 1):
        wj = wv[j]
        Y = np.empty((n_u + 1, 4), dtype=float)
        Y[:, 0:3] = sections_ctrl[:, j, :] * wj
        Y[:, 3] = wj
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

    return NURBSSurfaceTuple(
        order_u=rail.order, order_v=profile.order, knot_u=Uu, knot_v=Vv, control_points=control_points, weights=weights
    )



'''
# Build swept surface
surf = sweep1(
    rail=c1,
    profile=c2,
    mode="normal",
    tol=1.0,
    twist=0.0,  # or e.g. lambda u: 0.5*np.sin(u*np.pi)
    scale=1.0,  # or e.g. lambda u: 1.0 + 0.2*np.sin(u*2*np.pi)
    profile_anchor="centroid",  # or "first_cp" or ("curve_param", v_value)
)
'''
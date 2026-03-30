"""Bezier curve-surface intersection using squared-distance Bernstein net classification.

This module implements a subdivision-based CSX algorithm that uses the
squared-distance net ``||C(t) - S(u,v)||^2`` in trivariate Bernstein form
to classify cells as NO_INTERSECTION, UNIQUE_ISOLATED, OVERLAP,
BOUNDARY_ZERO, or INDETERMINATE.

Structurally identical to the CCX algorithm in ``_bez_ccx4.py`` but one
dimension larger (trivariate sq-dist net).
"""
from __future__ import annotations

import numpy as np

from mmcore.numeric.bern import de_casteljau_split_nd, bernstein_boundary_nd
from mmcore.numeric.bern_sq_dist import curve_surface_distance_squared_net_homog
from mmcore.numeric.intersection._bezier_common import (
    extract_weights, eval_curve, eval_surface, newton_csx,
)
from mmcore.numeric.intersection.ccx._bez_ccx4 import bez_ccx as bez_ccx_v4
from mmcore.numeric.intersection._sq_dist_classify import (
    classify_sq_dist_net,
    NO_INTERSECTION,
    UNIQUE_ISOLATED,
    OVERLAP,
    INDETERMINATE,
    BOUNDARY_ZERO,
    BoundaryZero,
    _boundary_zero_to_param_point,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subdivide_curve(ctrl, t=0.5):
    n = ctrl.shape[0] - 1
    tmp = ctrl.copy()
    left = [tmp[0].copy()]
    right_rev = [tmp[n].copy()]
    for r in range(1, n + 1):
        tmp[: n + 1 - r] = (1.0 - t) * tmp[: n + 1 - r] + t * tmp[1 : n + 2 - r]
        left.append(tmp[0].copy())
        right_rev.append(tmp[n - r].copy())
    return np.array(left), np.array(right_rev[::-1])


def _subdivide_surface(ctrl, axis, t=0.5):
    n = ctrl.shape[axis] - 1
    tmp = np.moveaxis(ctrl.copy(), axis, 0)
    left = [tmp[0].copy()]
    right_rev = [tmp[n].copy()]
    for r in range(1, n + 1):
        tmp[:n + 1 - r] = (1.0 - t) * tmp[:n + 1 - r] + t * tmp[1:n + 2 - r]
        left.append(tmp[0].copy())
        right_rev.append(tmp[n - r].copy())
    left = np.moveaxis(np.array(left), 0, axis)
    right = np.moveaxis(np.array(right_rev[::-1]), 0, axis)
    return left, right


def _subdivide_sq_dist_net(F, axis, t=0.5):
    Fv = F[..., np.newaxis]
    left_v, right_v = de_casteljau_split_nd(Fv, axis=axis, t=t)
    return left_v[..., 0], right_v[..., 0]


def _subdivide_surface_weights(sw, axis, t=0.5):
    n = sw.shape[axis] - 1
    tmp = np.moveaxis(sw.copy(), axis, 0)
    left = [tmp[0].copy()]
    right_rev = [tmp[n].copy()]
    for r in range(1, n + 1):
        tmp[:n + 1 - r] = (1.0 - t) * tmp[:n + 1 - r] + t * tmp[1:n + 2 - r]
        left.append(tmp[0].copy())
        right_rev.append(tmp[n - r].copy())
    left = np.moveaxis(np.array(left), 0, axis)
    right = np.moveaxis(np.array(right_rev[::-1]), 0, axis)
    return left, right


def _compute_param_tols_csx(C, S, atol, rational):
    """Compute parametric tolerances for curve and surface.

    Returns (ptol_t, ptol_u, ptol_v).
    """
    from mmcore.geom._nurbs_param_tol import bez_curve_param_tolerance, bez_surface_param_tolerance
    ptol_t = float(bez_curve_param_tolerance(C, tol=atol, rational=rational))
    ptol_u, ptol_v = bez_surface_param_tolerance(S, tol=atol, rational=rational)
    return ptol_t, float(ptol_u), float(ptol_v)


def _boundary_zero_to_tuv(bz: BoundaryZero,
                           t0: float, t1: float,
                           u0: float, u1: float,
                           v0: float, v1: float) -> tuple[float, float, float]:
    """Convert a BoundaryZero to global (t, u, v) parameters.

    For a 3D CSX net with axes [0=t, 1=u, 2=v]:
    - axis=0 fixed: t = t0 or t1, (param, param2) map to (u, v)
    - axis=1 fixed: u = u0 or u1, (param, param2) map to (t, v)
    - axis=2 fixed: v = v0 or v1, (param, param2) map to (t, u)

    The free axes for a given fixed axis are [i for i in (0,1,2) if i != axis],
    and param maps to the first free axis, param2 to the second.
    """
    ranges = [(t0, t1), (u0, u1), (v0, v1)]
    pt = [0.0, 0.0, 0.0]

    # Fixed axis
    pt[bz.axis] = ranges[bz.axis][0] if bz.side == 0 else ranges[bz.axis][1]

    # Free axes
    free = [i for i in range(3) if i != bz.axis]
    pt[free[0]] = ranges[free[0]][0] + bz.param * (ranges[free[0]][1] - ranges[free[0]][0])
    if bz.param2 is not None:
        pt[free[1]] = ranges[free[1]][0] + bz.param2 * (ranges[free[1]][1] - ranges[free[1]][0])
    else:
        # Fallback: midpoint (shouldn't happen for properly constructed BoundaryZeros)
        pt[free[1]] = 0.5 * (ranges[free[1]][0] + ranges[free[1]][1])

    return pt[0], pt[1], pt[2]


# ---------------------------------------------------------------------------
# CSX boundary analysis: find zeros on the 6 faces of [0,1]^3
# ---------------------------------------------------------------------------

def _find_csx_boundary_zeros(F_3d, C, S, atol, rational):
    """Find precise intersection points on the boundary faces of the CSX domain.

    The 6 faces of [0,1]^3 decompose into two types:

    Type 1 — Curve endpoints (t=0, t=1):
      Extract the 2D slice D(u,v) = ||C(t_fixed) - S(u,v)||^2 from the
      trivariate net. This is a point-on-surface problem. We use the
      2D classifier (same as CCX) to find zeros.

    Type 2 — Surface boundaries (u=0, u=1, v=0, v=1):
      Extract the surface boundary isocurve (a Bezier curve in 3D) and
      call bez_ccx(C, iso_curve) to find intersections.

    Returns list of BoundaryZero objects with (axis, side, param, param2).
    """
    zeros = []
    F_3d_v = F_3d[..., np.newaxis]

    # --- Type 1: Curve endpoints (t=0, t=1) ---
    # Restrict the 3D net to t=0 and t=1 → 2D nets for point-on-surface
    for t_side in (0, 1):
        face_2d = bernstein_boundary_nd(F_3d_v, axis=0, side=t_side)[..., 0]

        # Use the 2D classifier (same as CCX) on this face
        _, Sw = extract_weights(S, rational=rational)
        sw_flat = Sw.ravel()
        # Weight for the point side is just C's weight at t=0 or t=1
        _, Cw = extract_weights(C, rational=rational)
        pw_point = np.array([Cw[0 if t_side == 0 else -1]])

        from mmcore.numeric.intersection._sq_dist_classify import (
            _check_min_of_net, _weight_max_product,
            _find_precise_boundary_zeros as _find_precise_bz_2d,
        )

        w_scale = _weight_max_product(pw_point, sw_flat)

        # Quick check: if min of 2D face > 0, no intersection at this endpoint
        if _check_min_of_net(face_2d, atol, w_scale):
            continue

        # Find precise zeros on the 2D face using the 1D solver
        # (finds zeros on edges of the face)
        face_zeros = _find_precise_bz_2d(face_2d, atol, w_scale)
        for bz_2d in face_zeros:
            # Map 2D face zero to 3D BoundaryZero
            # bz_2d has axis (0 or 1 in the 2D face) and param
            # In the 3D net, axis=0 is t, so the face's axes map to u (1) and v (2)
            zeros.append(BoundaryZero(
                axis=0, side=t_side,
                param=bz_2d.param,
                param2=bz_2d.param2 if bz_2d.param2 is not None else (
                    float(bz_2d.side) if bz_2d.axis == 1 else None
                ),
            ))

        # Also check if the 2D face has a minimum near zero in its interior
        # (not just on its edges). Use Newton on the point-on-surface problem.
        pt = eval_curve(C, float(t_side), rational=rational)
        # Simple Newton: project point onto surface
        from mmcore.numeric.intersection._bezier_common import eval_surface_d1
        u_s, v_s = 0.5, 0.5  # seed from center
        for _it in range(20):
            s_pt, s_du, s_dv = eval_surface_d1(S, u_s, v_s, rational=rational)
            G = s_pt - pt
            g2 = float(np.dot(G, G))
            if g2 < atol * atol:
                break
            J = np.column_stack([s_du, s_dv])
            JTJ = J.T @ J + 1e-12 * np.eye(2)
            JTG = J.T @ G
            try:
                delta = -np.linalg.solve(JTJ, JTG)
            except np.linalg.LinAlgError:
                break
            step = 1.0
            for _ls in range(8):
                un = max(0.0, min(1.0, u_s + step * delta[0]))
                vn = max(0.0, min(1.0, v_s + step * delta[1]))
                Gn = eval_surface(S, un, vn, rational=rational) - pt
                if float(np.dot(Gn, Gn)) <= g2:
                    u_s, v_s = un, vn
                    break
                step *= 0.5
            else:
                break
        G_final = eval_surface(S, u_s, v_s, rational=rational) - pt
        if float(np.linalg.norm(G_final)) < atol:
            # Found a point-on-surface intersection — add as BoundaryZero
            bz = BoundaryZero(axis=0, side=t_side, param=u_s, param2=v_s)
            # Check not duplicate
            is_dup = any(
                abs(z.param - u_s) < 0.01 and z.param2 is not None and abs(z.param2 - v_s) < 0.01
                for z in zeros if z.axis == 0 and z.side == t_side
            )
            if not is_dup:
                zeros.append(bz)

    # --- Type 2: Surface boundaries (u=0, u=1, v=0, v=1) ---
    # Extract boundary isocurves and run CCX
    for surf_axis, surf_side in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        # Extract isocurve: S at axis=surf_axis, side=surf_side
        # For S of shape (m+1, n+1, D): axis=0 → S[side, :, :], axis=1 → S[:, side, :]
        if surf_axis == 0:
            iso_curve = S[0 if surf_side == 0 else -1, :, :]  # shape (n+1, D)
        else:
            iso_curve = S[:, 0 if surf_side == 0 else -1, :]  # shape (m+1, D)

        # Run CCX between the original curve and this isocurve
        ccx_result = bez_ccx_v4(C, iso_curve, atol=atol, rational=rational)

        for iso in ccx_result['isolated']:
            t_val = iso['u']  # parameter on C
            s_val = iso['v']  # parameter along the isocurve

            # Map isocurve parameter to surface (u, v)
            # The net axis in the 3D CSX problem:
            # surf_axis=0 → u is fixed, s_val runs along v → csx_axis=1
            # surf_axis=1 → v is fixed, s_val runs along u → csx_axis=2
            csx_axis = surf_axis + 1  # 0→1 (u-axis in CSX), 1→2 (v-axis in CSX)

            if surf_axis == 0:
                # u is fixed, s_val is v-parameter
                bz = BoundaryZero(axis=csx_axis, side=surf_side,
                                  param=t_val, param2=s_val)
            else:
                # v is fixed, s_val is u-parameter
                bz = BoundaryZero(axis=csx_axis, side=surf_side,
                                  param=t_val, param2=s_val)
            zeros.append(bz)

        # Overlaps from CCX also produce boundary information
        for ovl in ccx_result['overlaps']:
            # Overlap endpoints are boundary zeros too
            ur = ovl.get('u_range', (0.0, 1.0))
            vr = ovl.get('v_range', (0.0, 1.0))
            csx_axis = surf_axis + 1
            for t_val, s_val in [(ur[0], vr[0]), (ur[1], vr[1])]:
                bz = BoundaryZero(axis=csx_axis, side=surf_side,
                                  param=t_val, param2=s_val)
                zeros.append(bz)

    return zeros


def _check_csx_overlap_valley(C, S, boundary_zeros, atol, rational):
    """Valley check for CSX: confirm overlap by Newton at the midpoint.

    In CSX, the overlap curve through 3D parameter space can be significantly
    curved (unlike CCX where the valley is approximately linear). Instead of
    stepping along a straight line, we use Newton CSX from the midpoint of
    two boundary zeros. If Newton converges to a point with small residual,
    the overlap is confirmed.
    """
    if len(boundary_zeros) < 2:
        return None

    for i in range(len(boundary_zeros)):
        bz_a = boundary_zeros[i]
        for j in range(i + 1, len(boundary_zeros)):
            bz_b = boundary_zeros[j]
            if (bz_a.axis, bz_a.side) == (bz_b.axis, bz_b.side):
                continue

            pt_a = _boundary_zero_to_param_point(bz_a, 3)
            pt_b = _boundary_zero_to_param_point(bz_b, 3)

            # Midpoint in parameter space
            mid = [0.5 * (pt_a[k] + pt_b[k]) for k in range(3)]

            # Must be in strict interior
            margin = 0.02
            if not all(margin < mid[k] < 1.0 - margin for k in range(3)):
                continue

            # Newton from midpoint
            t_sol, u_sol, v_sol, G, last_step = newton_csx(
                C, S, mid[0], mid[1], mid[2], rational=rational,
            )
            residual = float(np.linalg.norm(G))

            if residual < atol:
                # Newton found a point on the overlap curve interior → confirmed
                return [bz_a, bz_b]

    return None


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def bez_csx(
    C,
    S,
    atol=1e-3,
    rational=True,
    max_depth=50,
    max_cells=100_000,
) -> dict:
    """Bezier curve-surface intersection via sq-dist net classification.

    Parameters
    ----------
    C : ndarray
        Control polygon of the Bezier curve. Shape ``(p+1, D)``.
    S : ndarray
        Control net of the Bezier surface. Shape ``(m+1, n+1, D)``.
    atol : float
        Geometric tolerance for intersection detection.
    rational : bool
        Whether the control nets are homogeneous (last column = weight).
    max_depth : int
        Maximum subdivision depth.
    max_cells : int
        Maximum total cells processed (safety limit).

    Returns
    -------
    dict
        ``{'isolated': [...], 'overlaps': [...]}``
    """
    C = np.asarray(C, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)

    F = curve_surface_distance_squared_net_homog(C, S, rational=rational)

    _, Pw = extract_weights(C, rational=rational)
    _, Sw = extract_weights(S, rational=rational)

    C_orig = C
    S_orig = S

    # Parametric tolerances
    ptol_t, ptol_u, ptol_v = _compute_param_tols_csx(C, S, atol, rational)

    isolated = []
    overlaps = []

    # --- CSX boundary analysis on the initial patch ---
    # Find zeros on the 6 faces of [0,1]^3 using:
    #   - Point-on-surface for t=0, t=1 (2D restriction of the 3D net)
    #   - CCX for u=0, u=1, v=0, v=1 (curve vs surface boundary isocurves)
    w_scale = float(np.max(np.abs(Pw)) * np.max(np.abs(Sw.ravel())))
    csx_boundary_zeros = _find_csx_boundary_zeros(F, C, S, atol, rational)

    # Accept boundary zeros as isolated intersections
    for bz in csx_boundary_zeros:
        t_bz, u_bz, v_bz = _boundary_zero_to_tuv(bz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        pt_c = eval_curve(C, t_bz, rational=rational)
        pt_s = eval_surface(S, u_bz, v_bz, rational=rational)
        if float(np.linalg.norm(pt_c - pt_s)) < atol:
            if not _is_duplicate(isolated, pt_c, atol):
                isolated.append({
                    "t": float(t_bz), "u": float(u_bz), "v": float(v_bz),
                    "point": pt_c,
                })

    # Valley check for overlap
    if len(csx_boundary_zeros) >= 2:
        overlap_pair = _check_csx_overlap_valley(C, S, csx_boundary_zeros, atol, rational)
        if overlap_pair is not None:
            bz_a, bz_b = overlap_pair
            t_a, u_a, v_a = _boundary_zero_to_tuv(bz_a, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
            t_b, u_b, v_b = _boundary_zero_to_tuv(bz_b, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
            overlaps.append({
                "boundary_zeros": [(bz_a.axis, bz_a.side), (bz_b.axis, bz_b.side)],
                "overlap_endpoints": [bz_a, bz_b],
                "t_range": (min(t_a, t_b), max(t_a, t_b)),
                "u_range": (min(u_a, u_b), max(u_a, u_b)),
                "v_range": (min(v_a, v_b), max(v_a, v_b)),
            })
            # If overlap is confirmed, skip further subdivision — the boundary
            # zeros are the endpoints, no interior search needed.
            # Remove any isolated points that fall within the overlap range.
            t_lo, t_hi = min(t_a, t_b), max(t_a, t_b)
            isolated = [iso for iso in isolated
                        if not (t_lo - atol <= iso["t"] <= t_hi + atol)]
            return {"isolated": isolated, "overlaps": overlaps}

    stack = [(C.copy(), S.copy(), F, Pw.copy(), Sw.copy(),
              0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0)]
    cells_processed = 0

    while stack:
        if cells_processed >= max_cells:
            break
        cells_processed += 1

        seg_c, seg_s, F_cell, pw, sw, t0, t1, u0, u1, v0, v1, depth = stack.pop()

        sw_flat = sw.ravel()
        cls = classify_sq_dist_net(F_cell, atol, pw, sw_flat)

        if cls.kind == NO_INTERSECTION:
            continue

        elif cls.kind == UNIQUE_ISOLATED:
            t_mid = 0.5 * (t0 + t1)
            u_mid = 0.5 * (u0 + u1)
            v_mid = 0.5 * (v0 + v1)
            t_sol, u_sol, v_sol, G, last_step = newton_csx(
                C_orig, S_orig, t_mid, u_mid, v_mid, rational=rational,
            )
            step_norm = abs(last_step[0]) + abs(last_step[1]) + abs(last_step[2])
            residual_ok = float(np.linalg.norm(G)) < atol
            if ((step_norm > 0 or residual_ok)
                and abs(last_step[0]) <= ptol_t
                and abs(last_step[1]) <= ptol_u
                and abs(last_step[2]) <= ptol_v):
                pt = eval_curve(C_orig, t_sol, rational=rational)
                if not _is_duplicate(isolated, pt, atol):
                    isolated.append({
                        "t": float(t_sol), "u": float(u_sol), "v": float(v_sol),
                        "point": pt,
                    })
            continue

        elif cls.kind == OVERLAP:
            overlaps.append({
                "boundary_zeros": cls.boundary_zeros,
                "overlap_endpoints": cls.overlap_endpoints,
                "t_range": (t0, t1),
                "u_range": (u0, u1),
                "v_range": (v0, v1),
            })
            continue

        elif cls.kind == BOUNDARY_ZERO:
            for bz in cls.precise_zeros:
                if not isinstance(bz, BoundaryZero):
                    continue
                t_bz, u_bz, v_bz = _boundary_zero_to_tuv(bz, t0, t1, u0, u1, v0, v1)
                pt_c = eval_curve(C_orig, t_bz, rational=rational)
                pt_s = eval_surface(S_orig, u_bz, v_bz, rational=rational)
                if float(np.linalg.norm(pt_c - pt_s)) < atol:
                    if not _is_duplicate(isolated, pt_c, atol):
                        isolated.append({
                            "t": float(t_bz), "u": float(u_bz), "v": float(v_bz),
                            "point": pt_c,
                        })
            continue

        # INDETERMINATE → subdivide
        if depth >= max_depth:
            t_mid = 0.5 * (t0 + t1)
            u_mid = 0.5 * (u0 + u1)
            v_mid = 0.5 * (v0 + v1)
            t_sol, u_sol, v_sol, G, last_step = newton_csx(
                C_orig, S_orig, t_mid, u_mid, v_mid, rational=rational,
            )
            step_norm = abs(last_step[0]) + abs(last_step[1]) + abs(last_step[2])
            residual_ok = float(np.linalg.norm(G)) < atol
            if ((step_norm > 0 or residual_ok)
                and abs(last_step[0]) <= ptol_t
                and abs(last_step[1]) <= ptol_u
                and abs(last_step[2]) <= ptol_v):
                pt = eval_curve(C_orig, t_sol, rational=rational)
                if not _is_duplicate(isolated, pt, atol):
                    isolated.append({
                        "t": float(t_sol), "u": float(u_sol), "v": float(v_sol),
                        "point": pt,
                    })
            continue

        # Choose subdivision axis
        spans = [t1 - t0, u1 - u0, v1 - v0]
        axis = int(np.argmax(spans))

        if axis == 0:
            t_mid = 0.5 * (t0 + t1)
            seg_c_left, seg_c_right = _subdivide_curve(seg_c)
            F_left, F_right = _subdivide_sq_dist_net(F_cell, axis=0)
            if rational:
                pw_left = seg_c_left[:, -1].copy()
                pw_right = seg_c_right[:, -1].copy()
            else:
                pw_left = np.ones(seg_c_left.shape[0], dtype=np.float64)
                pw_right = np.ones(seg_c_right.shape[0], dtype=np.float64)
            stack.append((seg_c_left, seg_s.copy(), F_left, pw_left, sw.copy(),
                          t0, t_mid, u0, u1, v0, v1, depth + 1))
            stack.append((seg_c_right, seg_s.copy(), F_right, pw_right, sw.copy(),
                          t_mid, t1, u0, u1, v0, v1, depth + 1))

        elif axis == 1:
            u_mid = 0.5 * (u0 + u1)
            seg_s_left, seg_s_right = _subdivide_surface(seg_s, axis=0)
            F_left, F_right = _subdivide_sq_dist_net(F_cell, axis=1)
            if rational:
                sw_left, sw_right = _subdivide_surface_weights(sw, axis=0)
            else:
                sw_left = np.ones(seg_s_left.shape[:2], dtype=np.float64)
                sw_right = np.ones(seg_s_right.shape[:2], dtype=np.float64)
            stack.append((seg_c.copy(), seg_s_left, F_left, pw.copy(), sw_left,
                          t0, t1, u0, u_mid, v0, v1, depth + 1))
            stack.append((seg_c.copy(), seg_s_right, F_right, pw.copy(), sw_right,
                          t0, t1, u_mid, u1, v0, v1, depth + 1))

        else:
            v_mid = 0.5 * (v0 + v1)
            seg_s_left, seg_s_right = _subdivide_surface(seg_s, axis=1)
            F_left, F_right = _subdivide_sq_dist_net(F_cell, axis=2)
            if rational:
                sw_left, sw_right = _subdivide_surface_weights(sw, axis=1)
            else:
                sw_left = np.ones(seg_s_left.shape[:2], dtype=np.float64)
                sw_right = np.ones(seg_s_right.shape[:2], dtype=np.float64)
            stack.append((seg_c.copy(), seg_s_left, F_left, pw.copy(), sw_left,
                          t0, t1, u0, u1, v0, v_mid, depth + 1))
            stack.append((seg_c.copy(), seg_s_right, F_right, pw.copy(), sw_right,
                          t0, t1, u0, u1, v_mid, v1, depth + 1))

    isolated, overlaps = _postprocess(isolated, overlaps, C_orig, S_orig, atol, rational)
    return {"isolated": isolated, "overlaps": overlaps}


def _is_duplicate(isolated, pt, atol):
    for entry in isolated:
        if np.linalg.norm(np.asarray(entry["point"]) - pt) < atol:
            return True
    return False


# ---------------------------------------------------------------------------
# Post-processing
# ---------------------------------------------------------------------------

def _intervals_overlap_1d(a0, a1, b0, b1, gap=0.0):
    return a0 - gap <= b1 and b0 - gap <= a1


def _verify_overlap_geometric(C, S, t0, t1, u0, u1, v0, v1, atol, rational, n_samples=5):
    for k in range(n_samples + 1):
        alpha = k / n_samples
        t = t0 + alpha * (t1 - t0)
        u = u0 + alpha * (u1 - u0)
        v = v0 + alpha * (v1 - v0)
        p_c = eval_curve(C, t, rational=rational)
        p_s = eval_surface(S, u, v, rational=rational)
        if float(np.linalg.norm(p_c - p_s)) > atol:
            return False
    return True


def _postprocess(isolated, overlaps, C, S, atol, rational):
    if not overlaps:
        return isolated, overlaps

    n = len(overlaps)
    parent = list(range(n))

    def _find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a, b):
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    gap = atol * 0.1
    for i in range(n):
        ti, ui, vi = overlaps[i]["t_range"], overlaps[i]["u_range"], overlaps[i]["v_range"]
        for j in range(i + 1, n):
            tj, uj, vj = overlaps[j]["t_range"], overlaps[j]["u_range"], overlaps[j]["v_range"]
            if (_intervals_overlap_1d(ti[0], ti[1], tj[0], tj[1], gap)
                    and _intervals_overlap_1d(ui[0], ui[1], uj[0], uj[1], gap)
                    and _intervals_overlap_1d(vi[0], vi[1], vj[0], vj[1], gap)):
                _union(i, j)

    from collections import defaultdict
    comps = defaultdict(list)
    for i in range(n):
        comps[_find(i)].append(i)

    merged_overlaps = []
    collapse_isolated = []

    for indices in comps.values():
        t0 = min(overlaps[i]["t_range"][0] for i in indices)
        t1 = max(overlaps[i]["t_range"][1] for i in indices)
        u0 = min(overlaps[i]["u_range"][0] for i in indices)
        u1 = max(overlaps[i]["u_range"][1] for i in indices)
        v0 = min(overlaps[i]["v_range"][0] for i in indices)
        v1 = max(overlaps[i]["v_range"][1] for i in indices)

        if _verify_overlap_geometric(C, S, t0, t1, u0, u1, v0, v1, atol, rational):
            merged_overlaps.append({
                "boundary_zeros": [], "overlap_endpoints": [],
                "t_range": (t0, t1), "u_range": (u0, u1), "v_range": (v0, v1),
            })
        else:
            t_mid, u_mid, v_mid = 0.5*(t0+t1), 0.5*(u0+u1), 0.5*(v0+v1)
            t_sol, u_sol, v_sol, G, last_step = newton_csx(
                C, S, t_mid, u_mid, v_mid, rational=rational,
            )
            if float(np.linalg.norm(G)) < atol:
                pt = eval_curve(C, t_sol, rational=rational)
                collapse_isolated.append({
                    "t": float(t_sol), "u": float(u_sol), "v": float(v_sol), "point": pt,
                })

    remaining_isolated = []
    for iso in isolated:
        absorbed = False
        for indices in comps.values():
            t0c = min(overlaps[i]["t_range"][0] for i in indices)
            t1c = max(overlaps[i]["t_range"][1] for i in indices)
            u0c = min(overlaps[i]["u_range"][0] for i in indices)
            u1c = max(overlaps[i]["u_range"][1] for i in indices)
            v0c = min(overlaps[i]["v_range"][0] for i in indices)
            v1c = max(overlaps[i]["v_range"][1] for i in indices)
            margin = max(t1c-t0c, u1c-u0c, v1c-v0c, atol)
            if (t0c-margin <= iso["t"] <= t1c+margin
                    and u0c-margin <= iso["u"] <= u1c+margin
                    and v0c-margin <= iso["v"] <= v1c+margin):
                absorbed = True
                break
        if not absorbed:
            remaining_isolated.append(iso)

    all_isolated = list(remaining_isolated)
    for ci in collapse_isolated:
        if not _is_duplicate(all_isolated, ci["point"], atol):
            all_isolated.append(ci)

    if merged_overlaps:
        def _inside(iso):
            for ov in merged_overlaps:
                if (ov["t_range"][0]-atol <= iso["t"] <= ov["t_range"][1]+atol
                        and ov["u_range"][0]-atol <= iso["u"] <= ov["u_range"][1]+atol
                        and ov["v_range"][0]-atol <= iso["v"] <= ov["v_range"][1]+atol):
                    return True
            return False
        all_isolated = [iso for iso in all_isolated if not _inside(iso)]

    return all_isolated, merged_overlaps

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

from mmcore.numeric.bern import de_casteljau_split_nd
from mmcore.numeric.bern_sq_dist import curve_surface_distance_squared_net_homog
from mmcore.numeric.intersection._bezier_common import (
    extract_weights, eval_curve, eval_surface, newton_csx,
)
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

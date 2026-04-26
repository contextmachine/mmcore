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

from mmcore.numeric.bern import (
    de_casteljau_split_nd, bernstein_boundary_nd,
    bernstein_eval_nd, bernstein_partial_derivative_coeffs,
)
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

        # AABB pre-filter: skip isocurves whose bounding box doesn't overlap the curve's
        from mmcore.numeric._aabb import aabb, aabb_intersect
        if rational:
            c_pts = C[:, :-1] / C[:, -1:]
            iso_pts = iso_curve[:, :-1] / iso_curve[:, -1:]
        else:
            c_pts = C
            iso_pts = iso_curve
        c_bb = np.array(aabb(c_pts))
        c_bb[0] -= atol; c_bb[1] += atol
        iso_bb = np.array(aabb(iso_pts))
        iso_bb[0] -= atol; iso_bb[1] += atol
        if not aabb_intersect(c_bb, iso_bb):
            continue

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


def _project_point_on_surface(pt, S, u_seed, v_seed, atol, rational, max_it=20):
    """Project a 3D point onto a Bezier surface. Returns (u, v, dist)."""
    from mmcore.numeric.intersection._bezier_common import eval_surface_d1
    u, v = float(u_seed), float(v_seed)
    for _ in range(max_it):
        s_pt, s_du, s_dv = eval_surface_d1(S, u, v, rational=rational)
        G = s_pt - pt
        g2 = float(np.dot(G, G))
        if g2 < 1e-20:
            break
        J = np.column_stack([s_du, s_dv])
        A = J.T @ J + 1e-12 * np.eye(2)
        b = -J.T @ G
        try:
            delta = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            break
        step = 1.0
        for _ in range(8):
            un = max(0.0, min(1.0, u + step * delta[0]))
            vn = max(0.0, min(1.0, v + step * delta[1]))
            Gn = eval_surface(S, un, vn, rational=rational) - pt
            if float(np.dot(Gn, Gn)) <= g2:
                u, v = un, vn
                break
            step *= 0.5
        else:
            break
    dist = float(np.linalg.norm(eval_surface(S, u, v, rational=rational) - pt))
    return u, v, dist


def _check_csx_overlap_valley(C, S, boundary_zeros, atol, rational):
    """Valley check for CSX: step from each endpoint along the valley.

    For each pair of boundary zeros on different faces:
    1. From each boundary zero, step inward by 2*ptol_t along the curve
    2. Evaluate C(t_step) and project onto S to find (u, v)
    3. If dist < atol AND (u,v) moved away from the seed → valley continues
    4. If both sides confirm → overlap

    This does NOT assume the overlap is linear in parameter space.
    """
    if len(boundary_zeros) < 2:
        return None

    ptol_t, ptol_u, ptol_v = _compute_param_tols_csx(C, S, atol, rational)

    for i in range(len(boundary_zeros)):
        bz_a = boundary_zeros[i]
        for j in range(i + 1, len(boundary_zeros)):
            bz_b = boundary_zeros[j]
            if (bz_a.axis, bz_a.side) == (bz_b.axis, bz_b.side):
                continue

            pt_a = _boundary_zero_to_param_point(bz_a, 3)
            pt_b = _boundary_zero_to_param_point(bz_b, 3)
            t_a, u_a, v_a = pt_a[0], pt_a[1], pt_a[2]
            t_b, u_b, v_b = pt_b[0], pt_b[1], pt_b[2]

            # Check the t-separation is large enough for a meaningful overlap
            if abs(t_b - t_a) < ptol_t * 4:
                continue

            # Step inward from endpoint A
            step_t = 2 * ptol_t
            t_step_a = t_a + step_t if t_a < t_b else t_a - step_t
            t_step_a = max(0.0, min(1.0, t_step_a))

            pt_c_a = eval_curve(C, t_step_a, rational=rational)
            u_proj_a, v_proj_a, dist_a = _project_point_on_surface(
                pt_c_a, S, u_a, v_a, atol, rational,
            )
            moved_a = (abs(u_proj_a - u_a) > ptol_u or
                       abs(v_proj_a - v_a) > ptol_v)

            if dist_a >= atol or not moved_a:
                continue

            # Step inward from endpoint B
            t_step_b = t_b - step_t if t_b > t_a else t_b + step_t
            t_step_b = max(0.0, min(1.0, t_step_b))

            pt_c_b = eval_curve(C, t_step_b, rational=rational)
            u_proj_b, v_proj_b, dist_b = _project_point_on_surface(
                pt_c_b, S, u_b, v_b, atol, rational,
            )
            moved_b = (abs(u_proj_b - u_b) > ptol_u or
                       abs(v_proj_b - v_b) > ptol_v)

            if dist_b >= atol or not moved_b:
                continue

            # Both sides confirmed: valley continues inward from both endpoints
            return [bz_a, bz_b]

    return None


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------


def _restrict_net_t(F, t_lo, t_hi):
    """Restrict a trivariate sq-dist net to a sub-interval along axis 0 (t)."""
    Fv = F[..., np.newaxis]
    if t_lo > 1e-12:
        _, Fv = de_casteljau_split_nd(Fv, axis=0, t=t_lo)
    if t_hi < 1.0 - 1e-12:
        t_hi_rescaled = (t_hi - t_lo) / (1.0 - t_lo) if t_lo > 1e-12 else t_hi
        Fv, _ = de_casteljau_split_nd(Fv, axis=0, t=t_hi_rescaled)
    return Fv[..., 0]


def _restrict_curve(C, t_lo, t_hi):
    """Restrict a Bezier curve to a sub-interval [t_lo, t_hi]."""
    if t_lo > 1e-12:
        _, C = _subdivide_curve(C, t_lo)
    if t_hi < 1.0 - 1e-12:
        t_hi_rescaled = (t_hi - t_lo) / (1.0 - t_lo) if t_lo > 1e-12 else t_hi
        C, _ = _subdivide_curve(C, t_hi_rescaled)
    return C


def _split_intervals(cut, lo, hi, ptol):
    """Split [lo, hi] into up to 3 sub-intervals around cut ± ptol.

    Returns list of (sub_lo, sub_hi) intervals. The center interval
    [cut-ptol, cut+ptol] is the "excluded" zone (included in the list
    but marked by being the center).
    Returns all 3 intervals (left, center, right), filtering empties.
    """
    cut_lo = max(cut - ptol, lo)
    cut_hi = min(cut + ptol, hi)
    intervals = []
    if lo + 1e-15 < cut_lo:
        intervals.append((lo, cut_lo))
    intervals.append((cut_lo, cut_hi))  # center (to be excluded)
    if cut_hi < hi - 1e-15:
        intervals.append((cut_hi, hi))
    return intervals


def _restrict_net_axis(F_cell, axis, lo, hi, cell_lo, cell_hi):
    """Restrict a trivariate net along one axis to [lo, hi] within [cell_lo, cell_hi]."""
    span = cell_hi - cell_lo
    if span < 1e-30:
        return F_cell

    frac_lo = (lo - cell_lo) / span
    frac_hi = (hi - cell_lo) / span

    Fv = F_cell[..., np.newaxis]
    if frac_lo > 1e-12:
        _, Fv = de_casteljau_split_nd(Fv, axis=axis, t=frac_lo)
    if frac_hi < 1.0 - 1e-12:
        frac_hi_rescaled = (frac_hi - frac_lo) / (1.0 - frac_lo) if frac_lo > 1e-12 else frac_hi
        Fv, _ = de_casteljau_split_nd(Fv, axis=axis, t=frac_hi_rescaled)
    return Fv[..., 0]


def _cutout_3d(F_cell, seg_c, pw, sw, t0, t1, u0, u1, v0, v1, depth,
               t_cut, u_cut, v_cut, ptol_t, ptol_u, ptol_v, rational):
    """Cut out a ptol-neighborhood around (t_cut, u_cut, v_cut) from a cell.

    Splits along each axis at cut ± ptol, producing 3×3×3 = 27 boxes.
    The center box (ptol-neighborhood) is discarded. Remaining 26 are
    returned with restricted nets.
    """
    t_intervals = _split_intervals(t_cut, t0, t1, ptol_t)
    u_intervals = _split_intervals(u_cut, u0, u1, ptol_u)
    v_intervals = _split_intervals(v_cut, v0, v1, ptol_v)

    # Identify the center index for each axis
    t_center = len(t_intervals) // 2 if len(t_intervals) == 3 else (0 if len(t_intervals) == 1 else -1)
    u_center = len(u_intervals) // 2 if len(u_intervals) == 3 else (0 if len(u_intervals) == 1 else -1)
    v_center = len(v_intervals) // 2 if len(v_intervals) == 3 else (0 if len(v_intervals) == 1 else -1)

    sub_cells = []

    for ti, (t_lo, t_hi) in enumerate(t_intervals):
        for ui, (u_lo, u_hi) in enumerate(u_intervals):
            for vi, (v_lo, v_hi) in enumerate(v_intervals):
                # Skip the center box
                if ti == t_center and ui == u_center and vi == v_center:
                    continue

                # Skip empty intervals
                if t_hi - t_lo < 1e-15 or u_hi - u_lo < 1e-15 or v_hi - v_lo < 1e-15:
                    continue

                # Restrict the net to this sub-box
                F_sub = _restrict_net_axis(F_cell, 0, t_lo, t_hi, t0, t1)
                F_sub = _restrict_net_axis(F_sub, 1, u_lo, u_hi, u0, u1)
                F_sub = _restrict_net_axis(F_sub, 2, v_lo, v_hi, v0, v1)

                # Restrict curve to t sub-interval
                C_sub = _restrict_curve(seg_c,
                    (t_lo - t0) / max(t1 - t0, 1e-30),
                    (t_hi - t0) / max(t1 - t0, 1e-30))
                pw_sub = C_sub[:, -1].copy() if rational else np.ones(C_sub.shape[0])

                # Restrict surface weights to u,v sub-interval
                if rational:
                    sw_sub = sw  # TODO: proper weight restriction
                else:
                    sw_sub = sw.copy()

                sub_cells.append((F_sub, C_sub, pw_sub, sw_sub,
                                  t_lo, t_hi, u_lo, u_hi, v_lo, v_hi, depth + 1))

    return sub_cells


def _phase2_isolated_search(
    F_sub, C_sub, S, C_orig, S_orig,
    t_lo, t_hi, atol, rational, ptol_t, ptol_u, ptol_v,
    known_points=None,
    max_depth=50, max_cells=50_000,
):
    """Phase 2: find isolated intersections via subdivision + Newton + cutout.

    When Newton finds an intersection, the ptol-neighborhood is cut out
    from the cell, producing sub-cells that are guaranteed root-free at
    that location. This prevents re-convergence and enables efficient
    discovery of multiple intersections.
    """
    from mmcore.numeric.intersection._sq_dist_classify import (
        _check_min_of_net, _check_lipschitz, _weight_max_product,
    )

    _, Pw = extract_weights(C_sub, rational=rational)
    _, Sw = extract_weights(S, rational=rational)

    # Start with known points from Phase 1 so pre-Newton dedup can skip them
    isolated = list(known_points) if known_points else []
    n_known = len(isolated)
    cells = 0

    stack = [(F_sub, C_sub, Pw.copy(), Sw.copy(),
              t_lo, t_hi, 0.0, 1.0, 0.0, 1.0, 0)]

    while stack:
        if cells >= max_cells:
            break
        cells += 1

        F_cell, seg_c, pw, sw, t0, t1, u0, u1, v0, v1, depth = stack.pop()

        # Quick prune: min-of-net
        sw_flat = sw.ravel()
        w_sc = _weight_max_product(pw, sw_flat)
        if _check_min_of_net(F_cell, atol, w_sc):
            continue

        # Lipschitz prune
        if _check_lipschitz(F_cell, atol, w_sc):
            continue

        # Derivative sign check: if any partial derivative has all
        # coefficients of the same sign, no stationary point exists
        Fv = F_cell[..., np.newaxis]
        can_have_stationary = True
        for ax in range(3):
            dF = bernstein_partial_derivative_coeffs(Fv, axis=ax)
            coeffs = dF[..., 0]
            if np.min(coeffs) >= 0 or np.max(coeffs) <= 0:
                can_have_stationary = False
                break
        if not can_have_stationary:
            continue

        # ptol-based early termination: if the curve parameter span is
        # below ptol_t, further subdivision is meaningless — we can't
        # distinguish different t values within this cell.
        t_span = t1 - t0
        u_span = u1 - u0
        v_span = v1 - v0
        if t_span <= ptol_t:
            # Micro-fragment: verify cell center is near a root, then
            # Newton-refine before reporting. Refinement collapses
            # near-duplicates (parametrically close to a known root but
            # geometrically just over atol) onto the exact known root,
            # so the dedup catches them.
            t_mid = 0.5 * (t0 + t1)
            u_mid = 0.5 * (u0 + u1)
            v_mid = 0.5 * (v0 + v1)
            pt_c = eval_curve(C_orig, t_mid, rational=rational)
            pt_s = eval_surface(S_orig, u_mid, v_mid, rational=rational)
            if float(np.linalg.norm(pt_c - pt_s)) < atol:
                t_r, u_r, v_r, G_r, _ = newton_csx(
                    C_orig, S_orig, t_mid, u_mid, v_mid, rational=rational,
                )
                if float(np.linalg.norm(G_r)) < atol:
                    pt_r = eval_curve(C_orig, t_r, rational=rational)
                    if not _is_duplicate(isolated, pt_r, atol):
                        isolated.append({
                            "t": float(t_r), "u": float(u_r), "v": float(v_r),
                            "point": pt_r, "_micro": True,
                        })
            continue

        # Try Newton from cell center
        t_mid = 0.5 * (t0 + t1)
        u_mid = 0.5 * (u0 + u1)
        v_mid = 0.5 * (v0 + v1)
        t_sol, u_sol, v_sol, G, last_step = newton_csx(
            C_orig, S_orig, t_mid, u_mid, v_mid, rational=rational,
        )

        residual_ok = float(np.linalg.norm(G)) < atol
        newton_stalled = (
                abs(last_step[0]) <= ptol_t
                and abs(last_step[1]) <= ptol_u
                and abs(last_step[2]) <= ptol_v
        )
        if newton_stalled:

            if residual_ok and t0 - ptol_t <= t_sol <= t1 + ptol_t:
                pt = eval_curve(C_orig, t_sol, rational=rational)
                is_new = not _is_duplicate(isolated, pt, atol)
                if is_new:
                    isolated.append({
                        "t": float(t_sol), "u": float(u_sol), "v": float(v_sol),
                        "point": pt,
                    })
                    # 3D cutout: split cell into 27 boxes, discard center,
                    # push remaining 26 for further search
                    sub_cells = _cutout_3d(
                        F_cell, seg_c, pw, sw, t0, t1, u0, u1, v0, v1, depth,
                        float(t_sol), float(u_sol), float(v_sol),
                        ptol_t, ptol_u, ptol_v, rational,
                    )
                    stack.extend(sub_cells)
                    continue

                # Known root: its convergence basin covers
                # [min(t_sol, t_mid), max(t_sol, t_mid)] along t.
                # When t is the dominant span, split at t_mid and keep
                # only the far half. Otherwise fall through to normal
                # subdivision (which picks u or v).
                if t_span >= max(u_span, v_span):
                    seg_c_L, seg_c_R = _subdivide_curve(seg_c)
                    F_L, F_R = _subdivide_sq_dist_net(F_cell, axis=0)
                    if t_sol <= t_mid:
                        pw_R = seg_c_R[:, -1].copy() if rational else np.ones(seg_c_R.shape[0])
                        stack.append((F_R, seg_c_R, pw_R, sw.copy(),
                                      t_mid, t1, u0, u1, v0, v1, depth + 1))
                    else:
                        pw_L = seg_c_L[:, -1].copy() if rational else np.ones(seg_c_L.shape[0])
                        stack.append((F_L, seg_c_L, pw_L, sw.copy(),
                                      t0, t_mid, u0, u1, v0, v1, depth + 1))
                    continue
                # t is not the dominant span — fall through to
                # subdivide along u or v

            elif residual_ok:
                # Newton converged to a real root, but it lies outside
                # this cell's tolerance range. The root will be found
                # by a cell that contains it — prune this one.
                continue
            else:
                # Newton stalled with bad residual. Distinguish:
                #   • Clamped at the [0,1] domain boundary → the stall
                #     is a numerical artifact; the cell may still
                #     contain a root that Newton couldn't reach. Fall
                #     through to subdivision.
                #   • At an INTERIOR stationary point of ||C-S||² with
                #     residual > atol → the cell's min of ||C-S|| is at
                #     least this residual, which exceeds atol. No root
                #     can exist in this cell — prune.
                _bnd = 1e-12
                clamped = (
                    t_sol <= _bnd or t_sol >= 1.0 - _bnd or
                    u_sol <= _bnd or u_sol >= 1.0 - _bnd or
                    v_sol <= _bnd or v_sol >= 1.0 - _bnd
                )
                if not clamped:
                    continue
                # else: fall through to subdivision

        if depth >= max_depth:
            continue

        # Subdivide along the axis with the largest span
        spans = [t1 - t0, u1 - u0, v1 - v0]
        axis = int(np.argmax(spans))

        if axis == 0:
            t_split = 0.5 * (t0 + t1)
            seg_c_L, seg_c_R = _subdivide_curve(seg_c)
            F_L, F_R = _subdivide_sq_dist_net(F_cell, axis=0)
            pw_L = seg_c_L[:, -1].copy() if rational else np.ones(seg_c_L.shape[0])
            pw_R = seg_c_R[:, -1].copy() if rational else np.ones(seg_c_R.shape[0])
            stack.append((F_L, seg_c_L, pw_L, sw.copy(), t0, t_split, u0, u1, v0, v1, depth+1))
            stack.append((F_R, seg_c_R, pw_R, sw.copy(), t_split, t1, u0, u1, v0, v1, depth+1))
        elif axis == 1:
            u_split = 0.5 * (u0 + u1)
            F_L, F_R = _subdivide_sq_dist_net(F_cell, axis=1)
            if rational:
                sw_L, sw_R = _subdivide_surface_weights(sw, axis=0)
            else:
                sw_L = sw.copy()
                sw_R = sw.copy()
            stack.append((F_L, seg_c.copy(), pw.copy(), sw_L, t0, t1, u0, u_split, v0, v1, depth+1))
            stack.append((F_R, seg_c.copy(), pw.copy(), sw_R, t0, t1, u_split, u1, v0, v1, depth+1))
        else:
            v_split = 0.5 * (v0 + v1)
            F_L, F_R = _subdivide_sq_dist_net(F_cell, axis=2)
            if rational:
                sw_L, sw_R = _subdivide_surface_weights(sw, axis=1)
            else:
                sw_L = sw.copy()
                sw_R = sw.copy()
            stack.append((F_L, seg_c.copy(), pw.copy(), sw_L, t0, t1, u0, u1, v0, v_split, depth+1))
            stack.append((F_R, seg_c.copy(), pw.copy(), sw_R, t0, t1, u0, u1, v_split, v1, depth+1))

    # Return only NEW results (exclude the pre-loaded known points)
    return isolated[n_known:]


def _compute_remaining_intervals(excludes, lo, hi):
    """Compute [lo, hi] minus the union of exclude intervals."""
    if not excludes:
        return [(lo, hi)]

    excludes = sorted(excludes, key=lambda x: x[0])
    merged = [excludes[0]]
    for a, b in excludes[1:]:
        if a <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], b))
        else:
            merged.append((a, b))

    result = []
    cursor = lo
    for ex_lo, ex_hi in merged:
        ex_lo = max(ex_lo, lo)
        ex_hi = min(ex_hi, hi)
        if cursor < ex_lo:
            result.append((cursor, ex_lo))
        cursor = max(cursor, ex_hi)
    if cursor < hi:
        result.append((cursor, hi))

    return result


# ---------------------------------------------------------------------------
# Main algorithm: two-phase architecture
# ---------------------------------------------------------------------------

def bez_csx(
    C,
    S,
    atol=1e-3,
    rational=True,
    max_depth=50,
    max_cells=100_000,
) -> dict:
    """Bezier curve-surface intersection via two-phase architecture.

    Phase 1: Find boundary intersections and overlaps on the initial patch.
             These can ONLY exist at the boundaries of the original objects.
    Phase 2: Search for isolated intersections on the remaining curve intervals
             via subdivision + Newton. No boundary analysis or overlap checks.

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
        Maximum subdivision depth for Phase 2.
    max_cells : int
        Maximum total cells processed in Phase 2 (safety limit).

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

    ptol_t, ptol_u, ptol_v = _compute_param_tols_csx(C, S, atol, rational)

    isolated = []
    overlaps = []

    # ===================================================================
    # PHASE 1: Boundary analysis + overlap detection (initial patch only)
    # ===================================================================
    csx_boundary_zeros = _find_csx_boundary_zeros(F, C, S, atol, rational)

    t_exclude = []  # t-intervals to cut from the curve

    # Accept boundary zeros. Each candidate is refined via Newton:
    # boundary analysis can return near-miss candidates (e.g., CCX
    # finds where curves are closest, even if the closest distance
    # is above atol) that lie near a real root just inside the
    # boundary. Newton refinement converges to the real root if one
    # exists nearby; otherwise the candidate is rejected.
    for bz in csx_boundary_zeros:
        t_bz, u_bz, v_bz = _boundary_zero_to_tuv(bz, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        t_r, u_r, v_r, G_r, _ = newton_csx(C, S, t_bz, u_bz, v_bz, rational=rational)
        if float(np.linalg.norm(G_r)) < atol:
            pt_r = eval_curve(C, t_r, rational=rational)
            if not _is_duplicate(isolated, pt_r, atol):
                isolated.append({
                    "t": float(t_r), "u": float(u_r), "v": float(v_r),
                    "point": pt_r,
                })
                t_exclude.append((t_r - ptol_t, t_r + ptol_t))

    # Valley check for overlap
    if len(csx_boundary_zeros) >= 2:
        overlap_pair = _check_csx_overlap_valley(C, S, csx_boundary_zeros, atol, rational)
        if overlap_pair is not None:
            bz_a, bz_b = overlap_pair
            t_a, u_a, v_a = _boundary_zero_to_tuv(bz_a, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
            t_b, u_b, v_b = _boundary_zero_to_tuv(bz_b, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
            t_lo_ovl, t_hi_ovl = min(t_a, t_b), max(t_a, t_b)
            overlaps.append({
                "boundary_zeros": [(bz_a.axis, bz_a.side), (bz_b.axis, bz_b.side)],
                "overlap_endpoints": [bz_a, bz_b],
                "t_range": (t_lo_ovl, t_hi_ovl),
                "u_range": (min(u_a, u_b), max(u_a, u_b)),
                "v_range": (min(v_a, v_b), max(v_a, v_b)),
            })
            t_exclude.append((t_lo_ovl - ptol_t, t_hi_ovl + ptol_t))
            # Remove isolated points inside the overlap
            isolated = [iso for iso in isolated
                        if not (t_lo_ovl - atol <= iso["t"] <= t_hi_ovl + atol)]

    # ===================================================================
    # PHASE 2: Isolated intersection search on remaining curve intervals
    # ===================================================================
    t_intervals = _compute_remaining_intervals(t_exclude, 0.0, 1.0)

    for t_lo, t_hi in t_intervals:
        if t_hi - t_lo < ptol_t * 0.1:
            continue

        # Restrict net and curve to sub-interval
        F_sub = _restrict_net_t(F, t_lo, t_hi)
        C_sub = _restrict_curve(C, t_lo, t_hi)

        # Quick check: all positive → no intersection
        from mmcore.numeric.intersection._sq_dist_classify import (
            _check_min_of_net, _weight_max_product,
        )
        _, Pw_sub = extract_weights(C_sub, rational=rational)
        w_sc = _weight_max_product(Pw_sub, Sw.ravel())
        if _check_min_of_net(F_sub, atol, w_sc):
            continue

        # Search for isolated intersections
        phase2_iso = _phase2_isolated_search(
            F_sub, C_sub, S, C_orig, S_orig,
            t_lo, t_hi, atol, rational, ptol_t, ptol_u, ptol_v,
            known_points=isolated,
            max_depth=max_depth, max_cells=max_cells,
        )

        for iso in phase2_iso:
            if not _is_duplicate(isolated, iso["point"], atol):
                isolated.append(iso)

    return {"isolated": isolated, "overlaps": overlaps}


def _is_duplicate(isolated, pt, atol):
    for entry in isolated:
        if np.linalg.norm(np.asarray(entry["point"]) - pt) < atol:
            return True
    return False

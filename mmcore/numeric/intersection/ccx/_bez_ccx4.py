"""Bezier curve-curve intersection using squared-distance Bernstein net classification.

This module implements a subdivision-based CCX algorithm that uses the
squared-distance net ``||C1(u) - C2(v)||^2`` in Bernstein form to classify
cells as NO_INTERSECTION, UNIQUE_ISOLATED, OVERLAP, or INDETERMINATE,
avoiding explicit Jacobian-rank analysis.
"""
from __future__ import annotations

import numpy as np

from mmcore.numeric.bern import de_casteljau_split_nd
from mmcore.numeric.bern_sq_dist import curve_curve_squared_net_homog
from mmcore.numeric.intersection._bezier_common import extract_weights, eval_curve, newton_ccx
from mmcore.numeric.intersection._sq_dist_classify import (
    classify_sq_dist_net,
    NO_INTERSECTION,
    UNIQUE_ISOLATED,
    OVERLAP,
    INDETERMINATE,
    BOUNDARY_ZERO,
    BoundaryZero, _boundary_zero_to_param_point,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _subdivide_curve(ctrl, t=0.5):
    """Split a Bezier curve at parameter t using de Casteljau.

    Parameters
    ----------
    ctrl : ndarray, shape (n+1, D)
        Control polygon of a degree-n Bezier curve.
    t : float
        Split parameter in [0, 1].

    Returns
    -------
    left, right : ndarray
        Control polygons of the two halves.
    """
    n = ctrl.shape[0] - 1
    tmp = ctrl.copy()
    left = [tmp[0].copy()]
    right_rev = [tmp[n].copy()]
    for r in range(1, n + 1):
        tmp[: n + 1 - r] = (1.0 - t) * tmp[: n + 1 - r] + t * tmp[1 : n + 2 - r]
        left.append(tmp[0].copy())
        right_rev.append(tmp[n - r].copy())
    return np.array(left), np.array(right_rev[::-1])


def _subdivide_sq_dist_net(F, axis, t=0.5):
    """Subdivide the scalar sq-dist Bernstein net along *axis*.

    ``de_casteljau_split_nd`` requires a trailing value dimension, so we
    temporarily add one and squeeze it back off.
    """
    Fv = F[..., np.newaxis]
    left_v, right_v = de_casteljau_split_nd(Fv, axis=axis, t=t)
    return left_v[..., 0], right_v[..., 0]


def _boundary_zero_to_uv(bz: BoundaryZero, u0: float, u1: float, v0: float, v1: float) -> tuple[float, float]:
    """Convert a BoundaryZero from the classifier into global (u, v) parameters.

    For a 2D CCX net:
    - axis=0, side=0 → u=u0, v = v0 + bz.param * (v1 - v0)
    - axis=0, side=1 → u=u1, v = v0 + bz.param * (v1 - v0)
    - axis=1, side=0 → v=v0, u = u0 + bz.param * (u1 - u0)
    - axis=1, side=1 → v=v1, u = u0 + bz.param * (u1 - u0)
    """
    if bz.axis == 0:
        u = u0 if bz.side == 0 else u1
        v = v0 + bz.param * (v1 - v0)
    else:
        v = v0 if bz.side == 0 else v1
        u = u0 + bz.param * (u1 - u0)
    return u, v


def _compute_param_tols(C1, C2, atol, rational):
    """Compute parametric tolerances for both curves using curve geometry.

    Returns (tol_u, tol_v) — the maximum parameter perturbation in each
    curve that corresponds to geometric deviation <= atol.
    """
    from mmcore.geom._nurbs_param_tol import bez_curve_param_tolerance
    tol_u = bez_curve_param_tolerance(C1, tol=atol, rational=rational)
    tol_v = bez_curve_param_tolerance(C2, tol=atol, rational=rational)
    return float(tol_u), float(tol_v)



from mmcore.numeric.bern import bernstein_partial_derivative_coeffs

# ---------------------------------------------------------------------------
# Phase 2 helpers
# ---------------------------------------------------------------------------

def _restrict_net_axis(F, axis, lo, hi, cell_lo, cell_hi):
    """Restrict a bivariate net along one axis to [lo, hi] within [cell_lo, cell_hi]."""
    span = cell_hi - cell_lo
    if span < 1e-30:
        return F
    frac_lo = (lo - cell_lo) / span
    frac_hi = (hi - cell_lo) / span
    Fv = F[..., np.newaxis]
    if frac_lo > 1e-12:
        _, Fv = de_casteljau_split_nd(Fv, axis=axis, t=frac_lo)
    if frac_hi < 1.0 - 1e-12:
        frac_hi_rescaled = (frac_hi - frac_lo) / (1.0 - frac_lo) if frac_lo > 1e-12 else frac_hi
        Fv, _ = de_casteljau_split_nd(Fv, axis=axis, t=frac_hi_rescaled)
    return Fv[..., 0]


def _split_intervals(cut, lo, hi, ptol):
    """Split [lo, hi] into up to 3 sub-intervals around cut ± ptol."""
    cut_lo = max(cut - ptol, lo)
    cut_hi = min(cut + ptol, hi)
    intervals = []
    if lo + 1e-15 < cut_lo:
        intervals.append((lo, cut_lo))
    intervals.append((cut_lo, cut_hi))
    if cut_hi < hi - 1e-15:
        intervals.append((cut_hi, hi))
    return intervals


def _cutout_2d(F_cell, seg1, seg2, pw, qw, u0, u1, v0, v1, depth,
               u_cut, v_cut, ptol_u, ptol_v, rational):
    """Cut out a ptol-neighborhood around (u_cut, v_cut) from a 2D cell.

    Splits along both axes at cut ± ptol, producing 3×3 = 9 boxes.
    The center box is discarded. Remaining 8 are returned with restricted nets.
    """
    u_intervals = _split_intervals(u_cut, u0, u1, ptol_u)
    v_intervals = _split_intervals(v_cut, v0, v1, ptol_v)

    u_center = len(u_intervals) // 2 if len(u_intervals) == 3 else (0 if len(u_intervals) == 1 else -1)
    v_center = len(v_intervals) // 2 if len(v_intervals) == 3 else (0 if len(v_intervals) == 1 else -1)

    sub_cells = []
    for ui, (u_lo, u_hi) in enumerate(u_intervals):
        for vi, (v_lo, v_hi) in enumerate(v_intervals):
            if ui == u_center and vi == v_center:
                continue
            if u_hi - u_lo < 1e-15 or v_hi - v_lo < 1e-15:
                continue

            F_sub = _restrict_net_axis(F_cell, 0, u_lo, u_hi, u0, u1)
            F_sub = _restrict_net_axis(F_sub, 1, v_lo, v_hi, v0, v1)

            # Restrict curves
            u_frac_lo = (u_lo - u0) / max(u1 - u0, 1e-30)
            u_frac_hi = (u_hi - u0) / max(u1 - u0, 1e-30)
            C1_sub = seg1
            if u_frac_lo > 1e-12:
                _, C1_sub = _subdivide_curve(C1_sub, u_frac_lo)
            if u_frac_hi < 1.0 - 1e-12:
                uf_rescaled = (u_frac_hi - u_frac_lo) / (1.0 - u_frac_lo) if u_frac_lo > 1e-12 else u_frac_hi
                C1_sub, _ = _subdivide_curve(C1_sub, uf_rescaled)
            pw_sub = C1_sub[:, -1].copy() if rational else np.ones(C1_sub.shape[0])

            v_frac_lo = (v_lo - v0) / max(v1 - v0, 1e-30)
            v_frac_hi = (v_hi - v0) / max(v1 - v0, 1e-30)
            C2_sub = seg2
            if v_frac_lo > 1e-12:
                _, C2_sub = _subdivide_curve(C2_sub, v_frac_lo)
            if v_frac_hi < 1.0 - 1e-12:
                vf_rescaled = (v_frac_hi - v_frac_lo) / (1.0 - v_frac_lo) if v_frac_lo > 1e-12 else v_frac_hi
                C2_sub, _ = _subdivide_curve(C2_sub, vf_rescaled)
            qw_sub = C2_sub[:, -1].copy() if rational else np.ones(C2_sub.shape[0])

            sub_cells.append((C1_sub, C2_sub, F_sub, pw_sub, qw_sub,
                              u_lo, u_hi, v_lo, v_hi, depth + 1))
    return sub_cells


def _phase2_ccx(F, C1, C2, C1_orig, C2_orig,
                u_lo, u_hi, v_lo, v_hi,
                atol, rational, ptol_u, ptol_v,
                known_points=None,
                max_depth=50, max_cells=50_000):
    """Phase 2: find isolated intersections via subdivision + Newton + cutout.

    No boundary analysis, no overlap checks, no classifier.
    Just: min-of-net → derivative sign → Newton → cutout.
    """
    from mmcore.numeric.intersection._sq_dist_classify import (
        _check_min_of_net, _check_lipschitz, _weight_max_product,
    )

    _, Pw = extract_weights(C1, rational=rational)
    _, Qw = extract_weights(C2, rational=rational)

    isolated = list(known_points) if known_points else []
    n_known = len(isolated)
    cells = 0

    stack = [(C1, C2, F, Pw.copy(), Qw.copy(),
              u_lo, u_hi, v_lo, v_hi, 0)]

    while stack:
        if cells >= max_cells:
            break
        cells += 1

        seg1, seg2, F_cell, pw, qw, u0, u1, v0, v1, depth = stack.pop()

        w_sc = _weight_max_product(pw, qw)

        # min-of-net prune
        if _check_min_of_net(F_cell, atol, w_sc):
            continue

        # Lipschitz prune
        if _check_lipschitz(F_cell, atol, w_sc):
            continue

        # Derivative sign pruning
        Fv = F_cell[..., np.newaxis]
        can_have_stationary = True
        for ax in range(2):
            dF = bernstein_partial_derivative_coeffs(Fv, axis=ax)
            coeffs = dF[..., 0]
            if np.min(coeffs) >= 0 or np.max(coeffs) <= 0:
                can_have_stationary = False
                break
        if not can_have_stationary:
            continue

        # ptol-based early termination
        if (u1 - u0) <= ptol_u and (v1 - v0) <= ptol_v:
            u_mid = 0.5 * (u0 + u1)
            v_mid = 0.5 * (v0 + v1)
            pt = eval_curve(C1_orig, u_mid, rational=rational)
            if not _is_duplicate(isolated, pt, atol):
                isolated.append({"u": float(u_mid), "v": float(v_mid), "point": pt, "_micro": True})
            continue

        # Newton from cell center
        u_mid = 0.5 * (u0 + u1)
        v_mid = 0.5 * (v0 + v1)
        u_sol, v_sol, G, last_step = newton_ccx(
            C1_orig, C2_orig, u_mid, v_mid, rational=rational,
        )
        step_norm = abs(last_step[0]) + abs(last_step[1])
        residual_ok = float(np.linalg.norm(G)) < atol
        converged = ((step_norm > 0 or residual_ok)
                     and abs(last_step[0]) <= ptol_u
                     and abs(last_step[1]) <= ptol_v)

        if converged and u0 - ptol_u <= u_sol <= u1 + ptol_u and v0 - ptol_v <= v_sol <= v1 + ptol_v:
            pt = eval_curve(C1_orig, u_sol, rational=rational)
            is_new = not _is_duplicate(isolated, pt, atol)
            if is_new:
                isolated.append({"u": float(u_sol), "v": float(v_sol), "point": pt})
                sub_cells = _cutout_2d(
                    F_cell, seg1, seg2, pw, qw, u0, u1, v0, v1, depth,
                    float(u_sol), float(v_sol), ptol_u, ptol_v, rational,
                )
                stack.extend(sub_cells)
            continue

        if converged:
            # Converged outside cell → prune
            continue

        if depth >= max_depth:
            continue

        # Subdivide
        u_span = u1 - u0
        v_span = v1 - v0
        axis = 0 if u_span >= v_span else 1

        if axis == 0:
            u_mid_split = 0.5 * (u0 + u1)
            seg1_L, seg1_R = _subdivide_curve(seg1)
            F_L, F_R = _subdivide_sq_dist_net(F_cell, axis=0)
            pw_L = seg1_L[:, -1].copy() if rational else np.ones(seg1_L.shape[0])
            pw_R = seg1_R[:, -1].copy() if rational else np.ones(seg1_R.shape[0])
            stack.append((seg1_L, seg2.copy(), F_L, pw_L, qw.copy(), u0, u_mid_split, v0, v1, depth+1))
            stack.append((seg1_R, seg2.copy(), F_R, pw_R, qw.copy(), u_mid_split, u1, v0, v1, depth+1))
        else:
            v_mid_split = 0.5 * (v0 + v1)
            seg2_L, seg2_R = _subdivide_curve(seg2)
            F_L, F_R = _subdivide_sq_dist_net(F_cell, axis=1)
            qw_L = seg2_L[:, -1].copy() if rational else np.ones(seg2_L.shape[0])
            qw_R = seg2_R[:, -1].copy() if rational else np.ones(seg2_R.shape[0])
            stack.append((seg1.copy(), seg2_L, F_L, pw.copy(), qw_L, u0, u1, v0, v_mid_split, depth+1))
            stack.append((seg1.copy(), seg2_R, F_R, pw.copy(), qw_R, u0, u1, v_mid_split, v1, depth+1))

    return isolated[n_known:]


# ---------------------------------------------------------------------------
# Main algorithm: two-phase architecture
# ---------------------------------------------------------------------------

def bez_ccx(
    C1,
    C2,
    atol=1e-3,
    rational=False,
    max_depth=50,
    max_cells=100_000,
) -> dict:
    """Bezier curve-curve intersection via two-phase architecture.

    Phase 1: Find boundary intersections and overlaps on the initial patch.
             These can ONLY exist at the boundaries of the original objects.
    Phase 2: Search for isolated intersections on the remaining parameter
             intervals via subdivision + Newton + cutout.
    """
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)

    F = curve_curve_squared_net_homog(C1, C2, rational=rational)

    _, Pw = extract_weights(C1, rational=rational)
    _, Qw = extract_weights(C2, rational=rational)

    C1_orig = C1
    C2_orig = C2

    ptol_u, ptol_v = _compute_param_tols(C1, C2, atol, rational)

    isolated = []
    overlaps = []

    # ===================================================================
    # PHASE 1: Boundary analysis + overlap (initial patch only)
    # ===================================================================
    cls = classify_sq_dist_net(F, atol, Pw, Qw)

    if cls.kind == NO_INTERSECTION:
        return {"isolated": [], "overlaps": []}

    # Collect boundary zeros and overlaps
    u_exclude = []  # (lo, hi) intervals to cut from u-axis
    v_exclude = []

    # Accept boundary zeros as isolated intersections
    if cls.precise_zeros:
        for bz in cls.precise_zeros:
            if not isinstance(bz, BoundaryZero):
                continue
            u_bz, v_bz = _boundary_zero_to_uv(bz, 0.0, 1.0, 0.0, 1.0)
            pt1 = eval_curve(C1, u_bz, rational=rational)
            pt2 = eval_curve(C2, v_bz, rational=rational)
            if float(np.linalg.norm(pt1 - pt2)) < atol:
                if not _is_duplicate(isolated, pt1, atol):
                    isolated.append({"u": float(u_bz), "v": float(v_bz), "point": pt1})
                    u_exclude.append((u_bz - ptol_u, u_bz + ptol_u))
                    v_exclude.append((v_bz - ptol_v, v_bz + ptol_v))

    # Overlap detection
    if cls.kind == OVERLAP:
        if cls.overlap_endpoints and isinstance(cls.overlap_endpoints[0], BoundaryZero):
            ovl_pts = []
            for bz in cls.overlap_endpoints:
                u_bz, v_bz = _boundary_zero_to_uv(bz, 0.0, 1.0, 0.0, 1.0)
                u_sol, v_sol, G, last_step = newton_ccx(
                    C1_orig, C2_orig, u_bz, v_bz, rational=rational,
                )
                if abs(last_step[0]) <= ptol_u and abs(last_step[1]) <= ptol_v:
                    ovl_pts.append((float(u_sol), float(v_sol)))
                else:
                    ovl_pts.append((float(u_bz), float(v_bz)))
            if len(ovl_pts) >= 2:
                overlaps.append({
                    "boundary_zeros": cls.boundary_zeros,
                    "overlap_endpoints": cls.overlap_endpoints,
                    "u_range": (ovl_pts[0][0], ovl_pts[1][0]),
                    "v_range": (ovl_pts[0][1], ovl_pts[1][1]),
                })
                u_lo_ovl = min(ovl_pts[0][0], ovl_pts[1][0])
                u_hi_ovl = max(ovl_pts[0][0], ovl_pts[1][0])
                v_lo_ovl = min(ovl_pts[0][1], ovl_pts[1][1])
                v_hi_ovl = max(ovl_pts[0][1], ovl_pts[1][1])
                u_exclude.append((u_lo_ovl - ptol_u, u_hi_ovl + ptol_u))
                v_exclude.append((v_lo_ovl - ptol_v, v_hi_ovl + ptol_v))
                isolated = [iso for iso in isolated
                            if not (u_lo_ovl - atol <= iso["u"] <= u_hi_ovl + atol)]
        else:
            overlaps.append({
                "boundary_zeros": cls.boundary_zeros,
                "overlap_endpoints": cls.overlap_endpoints,
                "u_range": (0.0, 1.0),
                "v_range": (0.0, 1.0),
            })
            return {"isolated": isolated, "overlaps": overlaps}

    # If classifier found OVERLAP and it covers the full range, we're done
    if overlaps and cls.kind == OVERLAP:
        return {"isolated": isolated, "overlaps": overlaps}

    # ===================================================================
    # PHASE 2: Isolated intersection search
    # ===================================================================

    # Compute remaining intervals (full [0,1] minus excluded)
    # For CCX, the exclusion is 2D — we can't simply cut intervals from
    # each axis independently. Instead, run Phase 2 on the full [0,1]^2
    # but pass the known points so they're skipped.
    phase2_iso = _phase2_ccx(
        F, C1, C2, C1_orig, C2_orig,
        0.0, 1.0, 0.0, 1.0,
        atol, rational, ptol_u, ptol_v,
        known_points=isolated,
        max_depth=max_depth, max_cells=max_cells,
    )

    for iso in phase2_iso:
        iso.pop('_micro', None)
        if not _is_duplicate(isolated, iso["point"], atol):
            isolated.append(iso)

    return {"isolated": isolated, "overlaps": overlaps}


def _is_duplicate(isolated, pt, atol):
    """Check if *pt* is within *atol* of any existing isolated point."""
    for entry in isolated:
        existing = np.asarray(entry["point"])
        if np.linalg.norm(existing - pt) < atol:
            return True
    return False

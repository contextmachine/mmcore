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


# ---------------------------------------------------------------------------
# Main algorithm
# ---------------------------------------------------------------------------

def bez_ccx(
    C1,
    C2,
    atol=1e-3,
    rational=False,
    max_depth=50,
    max_cells=100_000,
) -> dict:
    """Bezier curve-curve intersection via sq-dist net classification.

    Parameters
    ----------
    C1, C2 : ndarray
        Control polygons of the two Bezier curves.  Shape ``(n+1, D)``
        where D=3 (polynomial) or D includes a weight column when
        *rational* is True.
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

        Each isolated entry is ``{'u': float, 'v': float, 'point': ndarray}``.
        Each overlap entry is ``{'boundary_zeros': [...], 'u_range': (u0, u1),
        'v_range': (v0, v1), ...}``.
    """
    C1 = np.asarray(C1, dtype=np.float64)
    C2 = np.asarray(C2, dtype=np.float64)

    # Build initial sq-dist net
    F = curve_curve_squared_net_homog(C1, C2, rational=rational)

    # Extract weights for the classifier
    _, Pw = extract_weights(C1, rational=rational)
    _, Qw = extract_weights(C2, rational=rational)

    # Keep references to the ORIGINAL curves for Newton refinement
    C1_orig = C1
    C2_orig = C2

    # Results
    isolated = []
    overlaps = []

    # Stack entries: (seg1, seg2, F, Pw, Qw, u0, u1, v0, v1, depth)
    stack = [(C1.copy(), C2.copy(), F, Pw.copy(), Qw.copy(), 0.0, 1.0, 0.0, 1.0, 0)]
    cells_processed = 0

    while stack:
        if cells_processed >= max_cells:
            break
        cells_processed += 1

        seg1, seg2, F_cell, pw, qw, u0, u1, v0, v1, depth = stack.pop()

        # Classify
        cls = classify_sq_dist_net(F_cell, atol, pw, qw)

        if cls.kind == NO_INTERSECTION:
            continue

        elif cls.kind == UNIQUE_ISOLATED:
            # Newton refine on ORIGINAL curves with global param guess
            u_mid = 0.5 * (u0 + u1)
            v_mid = 0.5 * (v0 + v1)
            u_sol, v_sol, G, converged = newton_ccx(
                C1_orig, C2_orig, u_mid, v_mid,
                rational=rational, tol=atol * 1e-2,
            )
            if converged:
                pt = eval_curve(C1_orig, u_sol, rational=rational)
                # Deduplication: skip if close to an existing point
                if not _is_duplicate(isolated, pt, atol):
                    isolated.append({"u": float(u_sol), "v": float(v_sol), "point": pt})
            continue

        elif cls.kind == OVERLAP:
            overlaps.append({
                "boundary_zeros": cls.boundary_zeros,
                "overlap_endpoints": cls.overlap_endpoints,
                "u_range": (u0, u1),
                "v_range": (v0, v1),
            })
            continue

        # INDETERMINATE -> subdivide
        if depth >= max_depth:
            # Fallback: try Newton from cell center
            u_mid = 0.5 * (u0 + u1)
            v_mid = 0.5 * (v0 + v1)
            u_sol, v_sol, G, converged = newton_ccx(
                C1_orig, C2_orig, u_mid, v_mid,
                rational=rational, tol=atol * 1e-2,
            )
            if converged and float(np.linalg.norm(G)) < atol:
                pt = eval_curve(C1_orig, u_sol, rational=rational)
                if not _is_duplicate(isolated, pt, atol):
                    isolated.append({"u": float(u_sol), "v": float(v_sol), "point": pt})
            continue

        # Choose subdivision axis: split along the axis with larger param span
        u_span = u1 - u0
        v_span = v1 - v0
        axis = 0 if u_span >= v_span else 1

        if axis == 0:
            # Subdivide C1 (axis 0 of the sq-dist net)
            u_mid = 0.5 * (u0 + u1)
            seg1_left, seg1_right = _subdivide_curve(seg1)
            F_left, F_right = _subdivide_sq_dist_net(F_cell, axis=0)

            # Extract weights from subdivided halves
            if rational:
                pw_left = seg1_left[:, -1].copy()
                pw_right = seg1_right[:, -1].copy()
            else:
                pw_left = np.ones(seg1_left.shape[0], dtype=np.float64)
                pw_right = np.ones(seg1_right.shape[0], dtype=np.float64)

            stack.append((seg1_left, seg2.copy(), F_left, pw_left, qw.copy(), u0, u_mid, v0, v1, depth + 1))
            stack.append((seg1_right, seg2.copy(), F_right, pw_right, qw.copy(), u_mid, u1, v0, v1, depth + 1))
        else:
            # Subdivide C2 (axis 1 of the sq-dist net)
            v_mid = 0.5 * (v0 + v1)
            seg2_left, seg2_right = _subdivide_curve(seg2)
            F_left, F_right = _subdivide_sq_dist_net(F_cell, axis=1)

            # Extract weights from subdivided halves
            if rational:
                qw_left = seg2_left[:, -1].copy()
                qw_right = seg2_right[:, -1].copy()
            else:
                qw_left = np.ones(seg2_left.shape[0], dtype=np.float64)
                qw_right = np.ones(seg2_right.shape[0], dtype=np.float64)

            stack.append((seg1.copy(), seg2_left, F_left, pw.copy(), qw_left, u0, u1, v0, v_mid, depth + 1))
            stack.append((seg1.copy(), seg2_right, F_right, pw.copy(), qw_right, u0, u1, v_mid, v1, depth + 1))

    isolated, overlaps = _postprocess(isolated, overlaps, C1_orig, C2_orig, atol, rational)
    return {"isolated": isolated, "overlaps": overlaps}


def _is_duplicate(isolated, pt, atol):
    """Check if *pt* is within *atol* of any existing isolated point."""
    for entry in isolated:
        existing = np.asarray(entry["point"])
        if np.linalg.norm(existing - pt) < atol:
            return True
    return False


# ---------------------------------------------------------------------------
# Post-processing: merge micro-overlaps / tangent deduplication
# ---------------------------------------------------------------------------

def _intervals_overlap_1d(a0, a1, b0, b1, gap=0.0):
    """Return True if [a0, a1] and [b0, b1] overlap or are within *gap*."""
    return a0 - gap <= b1 and b0 - gap <= a1


def _verify_overlap_geometric(C1, C2, u0, u1, v0, v1, atol, rational, n_samples=5):
    """Check whether C1 and C2 actually trace the same path over the interval.

    A genuine overlap means C1(u) == C2(v) for corresponding (u, v) along
    the overlap.  We sample a few points along a linear parameterisation
    from (u0, v0) to (u1, v1) and verify that the distance is below *atol*
    at every sample.
    """
    for k in range(n_samples + 1):
        alpha = k / n_samples
        u = u0 + alpha * (u1 - u0)
        v = v0 + alpha * (v1 - v0)
        p1 = eval_curve(C1, u, rational=rational)
        p2 = eval_curve(C2, v, rational=rational)
        dist = float(np.linalg.norm(p1 - p2))
        if dist > atol:
            return False
    return True


def _postprocess(isolated, overlaps, C1, C2, atol, rational):
    """Merge adjacent micro-overlaps and collapse spurious clusters.

    Tangent and near-tangent intersections cause the classifier to emit
    many tiny OVERLAP cells and nearby UNIQUE_ISOLATED points around the
    same geometric location.  This function:

    1. Merges overlap entries whose parameter boxes touch/overlap into
       connected components.
    2. For each component, verifies whether C1 and C2 actually coincide
       geometrically.  If not (spurious overlap), collapses to a single
       isolated point via Newton refinement.
    3. Removes isolated points that are geometrically duplicated by a
       verified overlap range.
    """
    if not overlaps:
        return isolated, overlaps

    # --- Step 1: merge overlaps into connected components ----------------
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

    gap = atol * 0.1  # allow tiny gaps between adjacent cells
    for i in range(n):
        ui = overlaps[i]["u_range"]
        vi = overlaps[i]["v_range"]
        for j in range(i + 1, n):
            uj = overlaps[j]["u_range"]
            vj = overlaps[j]["v_range"]
            if (_intervals_overlap_1d(ui[0], ui[1], uj[0], uj[1], gap)
                    and _intervals_overlap_1d(vi[0], vi[1], vj[0], vj[1], gap)):
                _union(i, j)

    # Build merged components
    from collections import defaultdict
    comps = defaultdict(list)
    for i in range(n):
        comps[_find(i)].append(i)

    merged_overlaps = []
    collapse_isolated = []

    for indices in comps.values():
        u0 = min(overlaps[i]["u_range"][0] for i in indices)
        u1 = max(overlaps[i]["u_range"][1] for i in indices)
        v0 = min(overlaps[i]["v_range"][0] for i in indices)
        v1 = max(overlaps[i]["v_range"][1] for i in indices)

        u_mid = 0.5 * (u0 + u1)
        v_mid = 0.5 * (v0 + v1)

        # Geometric verification: do C1 and C2 actually coincide?
        if _verify_overlap_geometric(C1, C2, u0, u1, v0, v1, atol, rational):
            merged_overlaps.append({
                "boundary_zeros": [],
                "overlap_endpoints": [],
                "u_range": (u0, u1),
                "v_range": (v0, v1),
            })
        else:
            # Spurious overlap cluster (tangent point, near-miss, etc.)
            # Collapse to a single isolated point via Newton.
            u_sol, v_sol, G, converged = newton_ccx(
                C1, C2, u_mid, v_mid,
                rational=rational, tol=atol * 1e-2,
            )
            if converged:
                pt = eval_curve(C1, u_sol, rational=rational)
                collapse_isolated.append({"u": float(u_sol), "v": float(v_sol), "point": pt})
            else:
                # Try Newton with a tighter check: if the residual is small
                # enough, accept it anyway.
                pt_mid = eval_curve(C1, u_mid, rational=rational)
                pt_mid2 = eval_curve(C2, v_mid, rational=rational)
                dist = float(np.linalg.norm(pt_mid - pt_mid2))
                if dist < atol:
                    collapse_isolated.append({"u": float(u_mid), "v": float(v_mid), "point": pt_mid})

    # --- Step 2: absorb isolated points near collapsed overlap clusters --
    # Each collapsed overlap had a parameter range; any isolated point whose
    # parameters fall inside (or near) that range is a duplicate.
    remaining_isolated = []
    for iso in isolated:
        absorbed = False
        u_iso = iso["u"]
        v_iso = iso["v"]
        for indices in comps.values():
            u0c = min(overlaps[i]["u_range"][0] for i in indices)
            u1c = max(overlaps[i]["u_range"][1] for i in indices)
            v0c = min(overlaps[i]["v_range"][0] for i in indices)
            v1c = max(overlaps[i]["v_range"][1] for i in indices)
            margin = max(u1c - u0c, v1c - v0c, atol)
            if (u0c - margin <= u_iso <= u1c + margin
                    and v0c - margin <= v_iso <= v1c + margin):
                absorbed = True
                break
        if not absorbed:
            remaining_isolated.append(iso)

    # Add collapse_isolated (deduplicated against remaining)
    all_isolated = list(remaining_isolated)
    for ci in collapse_isolated:
        if not _is_duplicate(all_isolated, ci["point"], atol):
            all_isolated.append(ci)

    # --- Step 3: remove isolated points inside verified overlap ranges ---
    if merged_overlaps:
        def _inside_overlap(iso):
            u = iso["u"]
            v = iso["v"]
            for ov in merged_overlaps:
                ur = ov["u_range"]
                vr = ov["v_range"]
                if ur[0] - atol <= u <= ur[1] + atol and vr[0] - atol <= v <= vr[1] + atol:
                    return True
            return False

        all_isolated = [iso for iso in all_isolated if not _inside_overlap(iso)]

    return all_isolated, merged_overlaps

"""Bezier curve-surface intersection using squared-distance Bernstein net classification.

This module implements a subdivision-based CSX algorithm that uses the
squared-distance net ``||C(t) - S(u,v)||^2`` in trivariate Bernstein form
to classify cells as NO_INTERSECTION, UNIQUE_ISOLATED, OVERLAP, or
INDETERMINATE, avoiding explicit Jacobian-rank analysis.

Structurally identical to the CCX algorithm in ``_bez_ccx4.py`` but one
dimension larger (trivariate sq-dist net).
"""
from __future__ import annotations

import numpy as np

from mmcore.numeric.bern import de_casteljau_split_nd
from mmcore.numeric.bern_sq_dist import curve_surface_distance_squared_net_homog
from mmcore.numeric.intersection._bezier_common import extract_weights, eval_curve, eval_surface, newton_csx
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


def _subdivide_surface(ctrl, axis, t=0.5):
    """Split surface net along axis (0=u, 1=v) at parameter t.

    Parameters
    ----------
    ctrl : ndarray, shape (m+1, n+1, D)
        Surface control net.
    axis : int
        0 for u-direction, 1 for v-direction.
    t : float
        Split parameter in [0, 1].

    Returns
    -------
    left, right : ndarray
        Control nets of the two halves.
    """
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
    """Subdivide the scalar sq-dist Bernstein net along *axis*.

    ``de_casteljau_split_nd`` requires a trailing value dimension, so we
    temporarily add one and squeeze it back off.
    """
    Fv = F[..., np.newaxis]
    left_v, right_v = de_casteljau_split_nd(Fv, axis=axis, t=t)
    return left_v[..., 0], right_v[..., 0]


def _subdivide_surface_weights(sw, axis, t=0.5):
    """Split surface weight array along axis using de Casteljau.

    Parameters
    ----------
    sw : ndarray, shape (m+1, n+1)
        Surface weight array.
    axis : int
        0 for u-direction, 1 for v-direction.
    t : float
        Split parameter.

    Returns
    -------
    left, right : ndarray
        Weight arrays of the two halves.
    """
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
        Control polygon of the Bezier curve.  Shape ``(p+1, D)``
        where D=3 (polynomial) or D includes a weight column when
        *rational* is True.
    S : ndarray
        Control net of the Bezier surface.  Shape ``(m+1, n+1, D)``.
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

        Each isolated entry is ``{'t': float, 'u': float, 'v': float, 'point': ndarray}``.
        Each overlap entry is ``{'boundary_zeros': [...], 't_range': ...,
        'u_range': ..., 'v_range': ...}``.
    """
    C = np.asarray(C, dtype=np.float64)
    S = np.asarray(S, dtype=np.float64)

    # Build initial sq-dist net: shape (2p+1, 2m+1, 2n+1)
    F = curve_surface_distance_squared_net_homog(C, S, rational=rational)

    # Extract weights for the classifier
    _, Pw = extract_weights(C, rational=rational)              # 1D: (p+1,)
    _, Sw = extract_weights(S, rational=rational)              # 2D: (m+1, n+1)

    # Keep references to the ORIGINAL curve/surface for Newton refinement
    C_orig = C
    S_orig = S

    # Results
    isolated = []
    overlaps = []

    # Stack entries: (seg_c, seg_s, F, pw, sw, t0, t1, u0, u1, v0, v1, depth)
    stack = [(C.copy(), S.copy(), F, Pw.copy(), Sw.copy(), 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0)]
    cells_processed = 0

    while stack:
        if cells_processed >= max_cells:
            break
        cells_processed += 1

        seg_c, seg_s, F_cell, pw, sw, t0, t1, u0, u1, v0, v1, depth = stack.pop()

        # Classify -- flatten surface weights to 1D for the classifier
        sw_flat = sw.ravel()
        cls = classify_sq_dist_net(F_cell, atol, pw, sw_flat)

        if cls.kind == NO_INTERSECTION:
            continue

        elif cls.kind == UNIQUE_ISOLATED:
            # Newton refine on ORIGINAL curve/surface with global param guess
            t_mid = 0.5 * (t0 + t1)
            u_mid = 0.5 * (u0 + u1)
            v_mid = 0.5 * (v0 + v1)
            t_sol, u_sol, v_sol, G, converged = newton_csx(
                C_orig, S_orig, t_mid, u_mid, v_mid,
                rational=rational, tol=atol * 1e-2,
            )
            if converged:
                pt = eval_curve(C_orig, t_sol, rational=rational)
                if not _is_duplicate(isolated, pt, atol):
                    isolated.append({"t": float(t_sol), "u": float(u_sol), "v": float(v_sol), "point": pt})
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

        # INDETERMINATE -> subdivide
        if depth >= max_depth:
            # Fallback: try Newton from cell center
            t_mid = 0.5 * (t0 + t1)
            u_mid = 0.5 * (u0 + u1)
            v_mid = 0.5 * (v0 + v1)
            t_sol, u_sol, v_sol, G, converged = newton_csx(
                C_orig, S_orig, t_mid, u_mid, v_mid,
                rational=rational, tol=atol * 1e-2,
            )
            if converged and float(np.linalg.norm(G)) < atol:
                pt = eval_curve(C_orig, t_sol, rational=rational)
                if not _is_duplicate(isolated, pt, atol):
                    isolated.append({"t": float(t_sol), "u": float(u_sol), "v": float(v_sol), "point": pt})
            continue

        # Choose subdivision axis: split along the axis with largest param span
        t_span = t1 - t0
        u_span = u1 - u0
        v_span = v1 - v0
        spans = [t_span, u_span, v_span]
        axis = int(np.argmax(spans))

        if axis == 0:
            # Subdivide curve (axis 0 of the sq-dist net)
            t_mid = 0.5 * (t0 + t1)
            seg_c_left, seg_c_right = _subdivide_curve(seg_c)
            F_left, F_right = _subdivide_sq_dist_net(F_cell, axis=0)

            # Extract weights from subdivided halves
            if rational:
                pw_left = seg_c_left[:, -1].copy()
                pw_right = seg_c_right[:, -1].copy()
            else:
                pw_left = np.ones(seg_c_left.shape[0], dtype=np.float64)
                pw_right = np.ones(seg_c_right.shape[0], dtype=np.float64)

            stack.append((seg_c_left, seg_s.copy(), F_left, pw_left, sw.copy(), t0, t_mid, u0, u1, v0, v1, depth + 1))
            stack.append((seg_c_right, seg_s.copy(), F_right, pw_right, sw.copy(), t_mid, t1, u0, u1, v0, v1, depth + 1))

        elif axis == 1:
            # Subdivide surface along u (axis 1 of the sq-dist net)
            u_mid = 0.5 * (u0 + u1)
            seg_s_left, seg_s_right = _subdivide_surface(seg_s, axis=0)
            F_left, F_right = _subdivide_sq_dist_net(F_cell, axis=1)

            # Subdivide surface weights along u (axis 0 of the weight array)
            if rational:
                sw_left, sw_right = _subdivide_surface_weights(sw, axis=0)
            else:
                sw_left = np.ones(seg_s_left.shape[:2], dtype=np.float64)
                sw_right = np.ones(seg_s_right.shape[:2], dtype=np.float64)

            stack.append((seg_c.copy(), seg_s_left, F_left, pw.copy(), sw_left, t0, t1, u0, u_mid, v0, v1, depth + 1))
            stack.append((seg_c.copy(), seg_s_right, F_right, pw.copy(), sw_right, t0, t1, u_mid, u1, v0, v1, depth + 1))

        else:
            # Subdivide surface along v (axis 2 of the sq-dist net)
            v_mid = 0.5 * (v0 + v1)
            seg_s_left, seg_s_right = _subdivide_surface(seg_s, axis=1)
            F_left, F_right = _subdivide_sq_dist_net(F_cell, axis=2)

            # Subdivide surface weights along v (axis 1 of the weight array)
            if rational:
                sw_left, sw_right = _subdivide_surface_weights(sw, axis=1)
            else:
                sw_left = np.ones(seg_s_left.shape[:2], dtype=np.float64)
                sw_right = np.ones(seg_s_right.shape[:2], dtype=np.float64)

            stack.append((seg_c.copy(), seg_s_left, F_left, pw.copy(), sw_left, t0, t1, u0, u1, v0, v_mid, depth + 1))
            stack.append((seg_c.copy(), seg_s_right, F_right, pw.copy(), sw_right, t0, t1, u0, u1, v_mid, v1, depth + 1))

    isolated, overlaps = _postprocess(isolated, overlaps, C_orig, S_orig, atol, rational)
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


def _verify_overlap_geometric(C, S, t0, t1, u0, u1, v0, v1, atol, rational, n_samples=5):
    """Check whether C and S actually trace the same path over the interval.

    A genuine overlap means C(t) == S(u,v) for corresponding (t, u, v) along
    the overlap.  We sample a few points along a linear parameterisation
    from (t0, u0, v0) to (t1, u1, v1) and verify that the distance is below
    *atol* at every sample.
    """
    for k in range(n_samples + 1):
        alpha = k / n_samples
        t = t0 + alpha * (t1 - t0)
        u = u0 + alpha * (u1 - u0)
        v = v0 + alpha * (v1 - v0)
        p_c = eval_curve(C, t, rational=rational)
        p_s = eval_surface(S, u, v, rational=rational)
        dist = float(np.linalg.norm(p_c - p_s))
        if dist > atol:
            return False
    return True


def _postprocess(isolated, overlaps, C, S, atol, rational):
    """Merge adjacent micro-overlaps and collapse spurious clusters.

    Tangent and near-tangent intersections cause the classifier to emit
    many tiny OVERLAP cells and nearby UNIQUE_ISOLATED points around the
    same geometric location.  This function:

    1. Merges overlap entries whose parameter boxes touch/overlap into
       connected components.
    2. For each component, verifies whether C and S actually coincide
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
        ti = overlaps[i]["t_range"]
        ui = overlaps[i]["u_range"]
        vi = overlaps[i]["v_range"]
        for j in range(i + 1, n):
            tj = overlaps[j]["t_range"]
            uj = overlaps[j]["u_range"]
            vj = overlaps[j]["v_range"]
            if (_intervals_overlap_1d(ti[0], ti[1], tj[0], tj[1], gap)
                    and _intervals_overlap_1d(ui[0], ui[1], uj[0], uj[1], gap)
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
        t0 = min(overlaps[i]["t_range"][0] for i in indices)
        t1 = max(overlaps[i]["t_range"][1] for i in indices)
        u0 = min(overlaps[i]["u_range"][0] for i in indices)
        u1 = max(overlaps[i]["u_range"][1] for i in indices)
        v0 = min(overlaps[i]["v_range"][0] for i in indices)
        v1 = max(overlaps[i]["v_range"][1] for i in indices)

        t_mid = 0.5 * (t0 + t1)
        u_mid = 0.5 * (u0 + u1)
        v_mid = 0.5 * (v0 + v1)

        # Geometric verification: do C and S actually coincide?
        if _verify_overlap_geometric(C, S, t0, t1, u0, u1, v0, v1, atol, rational):
            merged_overlaps.append({
                "boundary_zeros": [],
                "overlap_endpoints": [],
                "t_range": (t0, t1),
                "u_range": (u0, u1),
                "v_range": (v0, v1),
            })
        else:
            # Spurious overlap cluster (tangent point, near-miss, etc.)
            # Collapse to a single isolated point via Newton.
            t_sol, u_sol, v_sol, G, converged = newton_csx(
                C, S, t_mid, u_mid, v_mid,
                rational=rational, tol=atol * 1e-2,
            )
            if converged:
                pt = eval_curve(C, t_sol, rational=rational)
                collapse_isolated.append({"t": float(t_sol), "u": float(u_sol), "v": float(v_sol), "point": pt})
            else:
                # Try Newton with a tighter check: if the residual is small
                # enough, accept it anyway.
                pt_mid_c = eval_curve(C, t_mid, rational=rational)
                pt_mid_s = eval_surface(S, u_mid, v_mid, rational=rational)
                dist = float(np.linalg.norm(pt_mid_c - pt_mid_s))
                if dist < atol:
                    collapse_isolated.append({"t": float(t_mid), "u": float(u_mid), "v": float(v_mid), "point": pt_mid_c})

    # --- Step 2: absorb isolated points near collapsed overlap clusters --
    remaining_isolated = []
    for iso in isolated:
        absorbed = False
        t_iso = iso["t"]
        u_iso = iso["u"]
        v_iso = iso["v"]
        for indices in comps.values():
            t0c = min(overlaps[i]["t_range"][0] for i in indices)
            t1c = max(overlaps[i]["t_range"][1] for i in indices)
            u0c = min(overlaps[i]["u_range"][0] for i in indices)
            u1c = max(overlaps[i]["u_range"][1] for i in indices)
            v0c = min(overlaps[i]["v_range"][0] for i in indices)
            v1c = max(overlaps[i]["v_range"][1] for i in indices)
            margin = max(t1c - t0c, u1c - u0c, v1c - v0c, atol)
            if (t0c - margin <= t_iso <= t1c + margin
                    and u0c - margin <= u_iso <= u1c + margin
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
            t = iso["t"]
            u = iso["u"]
            v = iso["v"]
            for ov in merged_overlaps:
                tr = ov["t_range"]
                ur = ov["u_range"]
                vr = ov["v_range"]
                if (tr[0] - atol <= t <= tr[1] + atol
                        and ur[0] - atol <= u <= ur[1] + atol
                        and vr[0] - atol <= v <= vr[1] + atol):
                    return True
            return False

        all_isolated = [iso for iso in all_isolated if not _inside_overlap(iso)]

    return all_isolated, merged_overlaps

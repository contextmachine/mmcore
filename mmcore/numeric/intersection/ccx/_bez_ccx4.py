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

    return {"isolated": isolated, "overlaps": overlaps}


def _is_duplicate(isolated, pt, atol):
    """Check if *pt* is within *atol* of any existing isolated point."""
    for entry in isolated:
        existing = np.asarray(entry["point"])
        if np.linalg.norm(existing - pt) < atol:
            return True
    return False

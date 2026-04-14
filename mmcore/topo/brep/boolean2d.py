"""2D Boolean operations on NURBS curves, built on top of BRep + nurbs_ccx_multiple.

See docs/superpowers/specs/2026-04-14-2d-boolean-operations-design.md for design.
"""
from __future__ import annotations

import numpy as np

from mmcore.geom._nurbs_eval import NURBSCurveTuple, evaluate_nurbs_curve
from mmcore.numeric.intersection.ccx._nccx4 import nurbs_ccx_multiple


_PIP_ENDPOINT_EPS_MUL = 2.0  # u_seg must be > _PIP_ENDPOINT_EPS_MUL * tol from 0
_PIP_CROSSING_SAMPLE_DT = 1e-3  # fraction of curve parameter range for crossing-test samples


def point_in_region(
    point,
    region_curves,
    tol: float = 1e-6,
) -> bool:
    """Return True iff *point* lies strictly inside the region bounded by *region_curves*.

    Casts a single line segment (degree-1 NURBS) from *point* past the region's
    bounding box, intersects it with *region_curves* via nurbs_ccx_multiple, and
    counts transverse crossings using the even-odd rule.

    Tangent intersections are identified by parallelism between the region curve's
    tangent at the hit and the segment direction; they are ignored (parity unchanged).
    Overlaps (coincident runs) are also ignored for the same parity reason.

    Raises RuntimeError if the segment starts on a region curve (point lies on the
    region boundary — the PIP result is undefined there).
    """
    pt = np.asarray(point, dtype=float).reshape(-1)
    dim = pt.shape[0]

    # --- expanded bbox (region + query point) ---
    all_ctrl = np.concatenate([np.asarray(c.control_points, dtype=float)
                               for c in region_curves], axis=0)
    bbox_min = np.minimum(all_ctrl.min(axis=0), pt)
    bbox_max = np.maximum(all_ctrl.max(axis=0), pt)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    L = 2.0 * diag + max(1.0, diag) * 1e-3  # escape length from anywhere in the expanded bbox

    # --- direction (deterministic, single shot) ---
    theta = 0.31415
    d = np.zeros(dim, dtype=float)
    d[0] = float(np.cos(theta))
    d[1] = float(np.sin(theta))

    seg = NURBSCurveTuple(
        order=2,
        knot=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=np.array([pt, pt + L * d], dtype=float),
        weights=np.array([1.0, 1.0], dtype=float),
    )

    isolated, overlaps = nurbs_ccx_multiple([seg] + list(region_curves), tol=tol)

    endpoint_eps = _PIP_ENDPOINT_EPS_MUL * tol

    # Line equation for the segment: f(q) = (q.y - pt.y)*d.x - (q.x - pt.x)*d.y
    # f > 0, f < 0, f == 0 tells which side of the line the query point lies on.
    def _line_side(q) -> float:
        return (q[1] - pt[1]) * d[0] - (q[0] - pt[0]) * d[1]

    count = 0

    if isolated is not None:
        for rec in isolated:
            c1 = int(rec['curve1_i'])
            c2 = int(rec['curve2_i'])
            if c1 != 0 and c2 != 0:
                continue  # not involving segment

            u = float(rec['u'])
            v = float(rec['v'])
            if c1 == 0:
                u_seg = u
                t_curve = v
                curve_idx_in_region = c2 - 1  # region_curves is offset by +1
            else:
                u_seg = v
                t_curve = u
                curve_idx_in_region = c1 - 1

            # segment start lying on a region curve ⇒ point is on boundary
            if u_seg < endpoint_eps:
                raise RuntimeError(
                    f"point_in_region: point {pt.tolist()} lies on a region boundary "
                    f"(segment start intersects curve {curve_idx_in_region} at t={t_curve})"
                )

            # Crossing test by signed distance sampling:
            # Sample the curve slightly before and after t_curve. If both samples
            # lie on the same side of the segment line, the curve grazes the line
            # (tangent touch — no parity flip). If on opposite sides, it crosses
            # (transverse — flip parity).
            curve = region_curves[curve_idx_in_region]
            t_lo_curve, t_hi_curve = curve.interval()
            dt = _PIP_CROSSING_SAMPLE_DT * (t_hi_curve - t_lo_curve)
            # clamp samples to curve's valid parameter range
            t_before = max(t_curve - dt, t_lo_curve)
            t_after = min(t_curve + dt, t_hi_curve)
            # must have non-zero separation on both sides of t_curve
            if t_curve - t_before < dt * 0.1 or t_after - t_curve < dt * 0.1:
                # near curve endpoint — cannot reliably sample; fall back to
                # counting as a transverse crossing (conservative)
                count += 1
                continue

            pt_before = np.asarray(evaluate_nurbs_curve(curve, t_before, 0)['C'], dtype=float)
            pt_after = np.asarray(evaluate_nurbs_curve(curve, t_after, 0)['C'], dtype=float)

            s_before = _line_side(pt_before)
            s_after = _line_side(pt_after)

            if s_before * s_after > 0.0:
                # same side → grazing / tangent touch → no parity flip
                continue
            # opposite sides → transverse crossing
            count += 1

    # overlaps: segment lies along a region curve for a range. Ignored — they
    # contribute no parity change (you slide along the boundary, never crossing it).
    # The sub-curves on either side of the overlap either both return to the same
    # side (grazing) or cross somewhere outside the overlap (isolated hit handled
    # elsewhere). For point-in-region purposes this is safe.

    return (count % 2) == 1

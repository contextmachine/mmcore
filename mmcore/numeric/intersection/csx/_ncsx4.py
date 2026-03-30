"""NURBS curve-surface intersection using the v4 sq-dist Bezier CSX.

Drop-in replacement for nurbs_csx from _ncsx.py, using _bez_csx4.bez_csx
instead of the old recursive subdivision approach.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mmcore.geom.nurbs import NURBSCurve, NURBSSurface
from mmcore.geom._nurbs_eval import (
    NURBSCurveTuple, NURBSSurfaceTuple,
    to_homogeneous_1d, to_homogeneous_2d,
    _nurbs_to_tuple,
)
from mmcore.geom._nurbs_knots import decompose_curve, decompose_surface
from mmcore.geom._nurbs_param_tol import (
    nurbs_curve_param_tolerance, nurbs_surface_param_tolerance,
)
from mmcore.geom.bvh.lbvh import AABB, build_bvh, bvh_intersect

from mmcore.numeric.intersection.csx._bez_csx4 import bez_csx as bez_csx_v4
from mmcore.numeric.intersection._bezier_common import eval_curve, eval_surface


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_rational_curve(curve: NURBSCurveTuple) -> bool:
    return not np.allclose(curve.weights, 1.0)


def _is_rational_surface(surface: NURBSSurfaceTuple) -> bool:
    return not np.allclose(surface.weights, 1.0)


def _surface_patch_aabb(patch: NURBSSurfaceTuple, tol: float) -> AABB:
    """AABB from surface patch control points, inflated by tol."""
    pts = patch.control_points.reshape(-1, patch.control_points.shape[-1])
    bb = AABB.from_points(pts)
    bb.offset_inplace(tol)
    return bb


def _map_local_to_global_csx(t_loc, u_loc, v_loc, t0, t1, u0, u1, v0, v1):
    """Map local Bezier [0,1] params to global NURBS params."""
    t_glob = t0 + (t1 - t0) * t_loc
    u_glob = u0 + (u1 - u0) * u_loc
    v_glob = v0 + (v1 - v0) * v_loc
    return t_glob, u_glob, v_glob


# ---------------------------------------------------------------------------
# Parametric deduplication
# ---------------------------------------------------------------------------

def _dedup_csx_isolated(entries, curve, surface, tol):
    """Deduplicate CSX isolated intersections using parametric tolerances.

    Same principle as CCX dedup: span-boundary duplicates arise when
    decompose_curve/decompose_surface splits at knots, and the same
    geometric intersection appears from both adjacent segments/patches.

    Dedup by sorting on the curve parameter t and merging consecutive
    entries within ptol_t. Surface params u, v are also checked.
    """
    if len(entries) <= 1:
        return entries

    ptol_t = float(nurbs_curve_param_tolerance(curve, tol))
    ptol_u, ptol_v = nurbs_surface_param_tolerance(surface, tol)
    ptol_u, ptol_v = float(ptol_u), float(ptol_v)

    sorted_entries = sorted(entries, key=lambda e: e['t'])

    deduped = [sorted_entries[0]]
    for entry in sorted_entries[1:]:
        prev = deduped[-1]
        if (abs(entry['t'] - prev['t']) < ptol_t and
            abs(entry['u'] - prev['u']) < ptol_u and
            abs(entry['v'] - prev['v']) < ptol_v):
            # Duplicate — keep the one with smaller residual
            pt_new_c = eval_curve(
                to_homogeneous_1d(curve.control_points, curve.weights)
                if _is_rational_curve(curve) else curve.control_points,
                entry['t'],
                rational=_is_rational_curve(curve),
            )
            pt_new_s = eval_surface(
                to_homogeneous_2d(surface.control_points, surface.weights)
                if _is_rational_surface(surface) else surface.control_points,
                entry['u'], entry['v'],
                rational=_is_rational_surface(surface),
            )
            pt_old_c = eval_curve(
                to_homogeneous_1d(curve.control_points, curve.weights)
                if _is_rational_curve(curve) else curve.control_points,
                prev['t'],
                rational=_is_rational_curve(curve),
            )
            pt_old_s = eval_surface(
                to_homogeneous_2d(surface.control_points, surface.weights)
                if _is_rational_surface(surface) else surface.control_points,
                prev['u'], prev['v'],
                rational=_is_rational_surface(surface),
            )
            if np.linalg.norm(pt_new_c - pt_new_s) < np.linalg.norm(pt_old_c - pt_old_s):
                deduped[-1] = entry
            continue
        deduped.append(entry)

    return deduped


# ---------------------------------------------------------------------------
# nurbs_csx: NURBS curve × NURBS surface intersection
# ---------------------------------------------------------------------------

def nurbs_csx(
    curve,
    surface,
    tol: float = 1e-3,
    **kwargs,
):
    """Find all intersections between a NURBS curve and a NURBS surface.

    Parameters
    ----------
    curve : NURBSCurve or NURBSCurveTuple
    surface : NURBSSurface or NURBSSurfaceTuple
    tol : float
        Geometric tolerance.

    Returns
    -------
    isolated : list[dict] or None
        Each entry: {'t': float, 'u': float, 'v': float, 'point': ndarray}
    overlaps : list[dict] or None
        Each entry: {'t_range': (t0,t1), 'u_range': (u0,u1), 'v_range': (v0,v1)}
    """
    if isinstance(curve, NURBSCurve):
        curve = _nurbs_to_tuple(curve)
    if isinstance(surface, NURBSSurface):
        surface = _nurbs_to_tuple(surface)

    dim = max(curve.control_points.shape[-1], surface.control_points.shape[-1])
    rational = _is_rational_curve(curve) or _is_rational_surface(surface)

    # Decompose into Bezier segments/patches
    curve_segs = decompose_curve(curve)
    surf_patches = decompose_surface(surface)

    # Build BVHs
    bvh_curves = build_bvh([
        AABB.from_points(seg.control_points).offset(tol) for seg in curve_segs
    ])
    bvh_surfs = build_bvh([
        _surface_patch_aabb(patch, tol) for patch in surf_patches
    ])

    raw_isolated = []
    raw_overlaps = []

    for a, b in bvh_intersect(bvh_curves, bvh_surfs, exact=False):
        seg = curve_segs[a.object]
        patch = surf_patches[b.object]

        if rational:
            pts_c = to_homogeneous_1d(seg.control_points, seg.weights)
            pts_s = to_homogeneous_2d(patch.control_points, patch.weights)
        else:
            pts_c = seg.control_points
            pts_s = patch.control_points

        result = bez_csx_v4(pts_c, pts_s, atol=tol, rational=rational)

        seg_interval = seg.interval()
        patch_interval = patch.interval()  # ((u0, u1), (v0, v1))

        for iso in result['isolated']:
            t_glob, u_glob, v_glob = _map_local_to_global_csx(
                iso['t'], iso['u'], iso['v'],
                seg_interval[0], seg_interval[1],
                patch_interval[0][0], patch_interval[0][1],
                patch_interval[1][0], patch_interval[1][1],
            )
            raw_isolated.append({
                't': t_glob, 'u': u_glob, 'v': v_glob,
                'point': iso['point'],
            })

        for ovl in result['overlaps']:
            tr = ovl.get('t_range', (0.0, 1.0))
            ur = ovl.get('u_range', (0.0, 1.0))
            vr = ovl.get('v_range', (0.0, 1.0))
            t0g = seg_interval[0] + tr[0] * (seg_interval[1] - seg_interval[0])
            t1g = seg_interval[0] + tr[1] * (seg_interval[1] - seg_interval[0])
            u0g = patch_interval[0][0] + ur[0] * (patch_interval[0][1] - patch_interval[0][0])
            u1g = patch_interval[0][0] + ur[1] * (patch_interval[0][1] - patch_interval[0][0])
            v0g = patch_interval[1][0] + vr[0] * (patch_interval[1][1] - patch_interval[1][0])
            v1g = patch_interval[1][0] + vr[1] * (patch_interval[1][1] - patch_interval[1][0])
            raw_overlaps.append({
                't_range': (t0g, t1g),
                'u_range': (u0g, u1g),
                'v_range': (v0g, v1g),
            })

    # Parametric deduplication
    deduped = _dedup_csx_isolated(raw_isolated, curve, surface, tol)

    isolated = deduped if deduped else None
    overlaps = raw_overlaps if raw_overlaps else None

    return isolated, overlaps

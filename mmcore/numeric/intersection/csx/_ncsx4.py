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
from mmcore.numeric._bezier_common import eval_curve, eval_surface


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


_BEZIER_LIMIT_KWARGS = ('max_depth',)
_DEFAULT_MAX_CELLS = 100_000
_DEFAULT_MAX_RESULTS = 4_096


def _bezier_limit_kwargs(kwargs):
    """Forward only the bounded-solver controls understood by bez_csx v4."""
    return {name: kwargs[name] for name in _BEZIER_LIMIT_KWARGS
            if name in kwargs}


# Shared aggregate-status ledger (ledger L52): the implementation lives in
# `_adapter_status`; these wrappers keep the adapter's historical private
# names, its extra `parameter_fibers` ledger field, and its message texts.
from mmcore.numeric.intersection._adapter_status import (
    consume_bezier_status as _shared_consume_bezier_status,
    mark_incomplete as _mark_incomplete,
    new_status as _shared_new_status,
    reject_unknown_kwargs as _reject_unknown_kwargs,
    remaining_allowances as _remaining_allowances,
)


def _new_status(max_cells, max_results):
    return _shared_new_status(
        max_cells, max_results, extra_list_fields=('parameter_fibers',))


def _map_parameter_fiber(fiber, seg_interval, patch_interval):
    """Map a Bezier CSX parameter fiber into global NURBS parameters."""
    mapped = dict(fiber)
    t0, t1 = seg_interval
    (u0, u1), (v0, v1) = patch_interval

    if 't_range' in mapped:
        lo, hi = mapped['t_range']
        mapped['t_range'] = (t0 + (t1 - t0) * lo,
                             t0 + (t1 - t0) * hi)
    if 'u_range' in mapped:
        lo, hi = mapped['u_range']
        mapped['u_range'] = (u0 + (u1 - u0) * lo,
                             u0 + (u1 - u0) * hi)
    if 'v_range' in mapped:
        lo, hi = mapped['v_range']
        mapped['v_range'] = (v0 + (v1 - v0) * lo,
                             v0 + (v1 - v0) * hi)
    if 't' in mapped:
        mapped['t'] = t0 + (t1 - t0) * mapped['t']
    if 'u' in mapped:
        mapped['u'] = u0 + (u1 - u0) * mapped['u']
    if 'v' in mapped:
        mapped['v'] = v0 + (v1 - v0) * mapped['v']
    return mapped


def _consume_bezier_status(
    result, status, seg_interval, patch_interval, return_status,
    cell_allowance, result_allowance,
):
    """Aggregate one span result; CSX additionally maps parameter fibers
    into global NURBS parameters and refuses the legacy two-value return
    when a positive-dimensional fiber is present."""
    result, incomplete = _shared_consume_bezier_status(
        result, status,
        incomplete_message=(
            f"nurbs_csx spans {seg_interval} x {patch_interval}: "
            "incomplete Bezier CSX result (budget exhausted or boundary "
            "topology incomplete); pass return_status=True to receive "
            "explicit partial status"),
        return_status=return_status,
        cell_allowance=cell_allowance,
        result_allowance=result_allowance,
        list_keys=('isolated', 'overlaps', 'parameter_fibers'),
    )

    fibers = result['parameter_fibers']
    if fibers:
        status['parameter_fibers'].extend(
            _map_parameter_fiber(fiber, seg_interval, patch_interval)
            for fiber in fibers
        )
        if not return_status:
            raise RuntimeError(
                f"nurbs_csx spans {seg_interval} x {patch_interval}: "
                "positive-dimensional parameter fiber cannot be represented "
                "by the legacy two-value return; pass return_status=True"
            )
    return result, incomplete


# ---------------------------------------------------------------------------
# Overlap merging
# ---------------------------------------------------------------------------

def _merge_overlaps_by_t(overlaps, ptol_t):
    """Merge adjacent overlaps whose t-ranges touch or overlap.

    Adjacent Bezier pairs independently detect portions of the same
    geometric overlap. Sort by t_range start and merge consecutive
    entries whose t-ranges are within ptol_t of each other.
    """
    if not overlaps or len(overlaps) <= 1:
        return list(overlaps) if overlaps else []

    sorted_ovls = sorted(overlaps, key=lambda o: o['t_range'][0])

    merged = [dict(sorted_ovls[0])]  # copy
    for ovl in sorted_ovls[1:]:
        prev = merged[-1]
        # Merge if t-ranges touch or overlap
        if ovl['t_range'][0] <= prev['t_range'][1] + ptol_t:
            # Extend the previous overlap
            prev['t_range'] = (
                min(prev['t_range'][0], ovl['t_range'][0]),
                max(prev['t_range'][1], ovl['t_range'][1]),
            )
            prev['u_range'] = (
                min(prev['u_range'][0], ovl['u_range'][0]),
                max(prev['u_range'][1], ovl['u_range'][1]),
            )
            prev['v_range'] = (
                min(prev['v_range'][0], ovl['v_range'][0]),
                max(prev['v_range'][1], ovl['v_range'][1]),
            )
        else:
            merged.append(dict(ovl))

    return merged


# ---------------------------------------------------------------------------
# Parametric deduplication
# ---------------------------------------------------------------------------

def _is_seam_duplicate(u1, u2, v1, v2, surface, ptol_u, ptol_v):
    """Check if (u1,v1) and (u2,v2) are the same point on a periodic seam.

    Returns True only if:
    - One of u or v differs by the full domain span (the other matches within ptol)
    - The surface is at least C0 across that seam (first/last CP rows identical)
    """
    (u_lo, u_hi), (v_lo, v_hi) = surface.interval()
    u_span = u_hi - u_lo
    v_span = v_hi - v_lo

    # Check u-seam: u values at opposite domain ends, v matches
    if abs(v1 - v2) < ptol_v and u_span > 0:
        u_wrap = u_span - abs(u1 - u2)
        if abs(u_wrap) < ptol_u:
            # Verify C0 continuity: first and last CP rows in u must match
            cp = surface.control_points
            w = surface.weights
            if (np.allclose(cp[0], cp[-1], atol=1e-10) and
                    np.allclose(w[0], w[-1], atol=1e-10)):
                return True

    # Check v-seam: v values at opposite domain ends, u matches
    if abs(u1 - u2) < ptol_u and v_span > 0:
        v_wrap = v_span - abs(v1 - v2)
        if abs(v_wrap) < ptol_v:
            cp = surface.control_points
            w = surface.weights
            if (np.allclose(cp[:, 0], cp[:, -1], atol=1e-10) and
                    np.allclose(w[:, 0], w[:, -1], atol=1e-10)):
                return True

    return False


def _dedup_csx_isolated(entries, curve, surface, tol):
    """Deduplicate CSX isolated intersections using parametric tolerances.

    Same principle as CCX dedup: span-boundary duplicates arise when
    decompose_curve/decompose_surface splits at knots, and the same
    geometric intersection appears from both adjacent segments/patches.

    Dedup by sorting on the curve parameter t and merging consecutive
    entries within ptol_t. Also detects periodic seam duplicates where
    u (or v) values sit at opposite ends of the domain.
    """
    if len(entries) <= 1:
        return entries

    ptol_t = float(nurbs_curve_param_tolerance(curve, tol))
    ptol_u, ptol_v = nurbs_surface_param_tolerance(surface, tol)
    ptol_u, ptol_v = float(ptol_u), float(ptol_v)

    rational_c = _is_rational_curve(curve)
    rational_s = _is_rational_surface(surface)
    C_h = (to_homogeneous_1d(curve.control_points, curve.weights)
           if rational_c else curve.control_points)
    S_h = (to_homogeneous_2d(surface.control_points, surface.weights)
           if rational_s else surface.control_points)

    sorted_entries = sorted(entries, key=lambda e: e['t'])

    deduped = [sorted_entries[0]]
    for entry in sorted_entries[1:]:
        prev = deduped[-1]

        # Standard parametric proximity check
        is_dup = (abs(entry['t'] - prev['t']) < ptol_t and
                  abs(entry['u'] - prev['u']) < ptol_u and
                  abs(entry['v'] - prev['v']) < ptol_v)

        # Periodic seam check: same t, same v, u at opposite domain ends
        if not is_dup and abs(entry['t'] - prev['t']) < ptol_t:
            is_dup = _is_seam_duplicate(
                entry['u'], prev['u'], entry['v'], prev['v'],
                surface, ptol_u, ptol_v,
            )

        if is_dup:
            # Duplicate — keep the one with smaller residual
            pt_new_c = eval_curve(C_h, entry['t'], rational=rational_c)
            pt_new_s = eval_surface(S_h, entry['u'], entry['v'], rational=rational_s)
            pt_old_c = eval_curve(C_h, prev['t'], rational=rational_c)
            pt_old_s = eval_surface(S_h, prev['u'], prev['v'], rational=rational_s)
            if np.linalg.norm(pt_new_c - pt_new_s) < np.linalg.norm(pt_old_c - pt_old_s):
                deduped[-1] = entry
            continue
        deduped.append(entry)

    return deduped


# ---------------------------------------------------------------------------
# nurbs_csx: NURBS curve × NURBS surface intersection
# ---------------------------------------------------------------------------

def nurbs_csx(
    curve: NURBSCurveTuple,
    surface: NURBSSurfaceTuple,
    tol: float = 1e-3,
    *,
    return_status: bool = True,
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
    status : dict
        Third value, returned by default: aggregate bounded-solver
        diagnostics and globally mapped ``parameter_fibers``; read
        ``status['complete']`` before trusting the output as the whole
        truth (ledger L41 — the former raise-on-incomplete default turned
        collapsed-edge geometry into a crash for legacy-shaped callers).
        Pass ``return_status=False`` for the legacy two-value shape, which
        raises ``RuntimeError`` on partial or positive-dimensional
        sub-results instead (fail-fast opt-in).
    """
    _reject_unknown_kwargs(
        "nurbs_csx", kwargs, ("max_cells", "max_results") + _BEZIER_LIMIT_KWARGS)
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
    candidates = list(bvh_intersect(bvh_curves, bvh_surfs, exact=False))
    # Candidate-scaled default allowance — a flat total that a handful of
    # ordinary span x patch pairs can exhaust is a mispriced exchange rate
    # (ledger L41 / review finding 2); explicit ``max_cells`` stays absolute.
    aggregate_max_cells = kwargs.get('max_cells')
    if aggregate_max_cells is None:
        aggregate_max_cells = _DEFAULT_MAX_CELLS * max(1, len(candidates))
    aggregate_max_cells = max(0, int(aggregate_max_cells))
    aggregate_max_results = max(
        0, int(kwargs.get('max_results', _DEFAULT_MAX_RESULTS)))
    status = _new_status(aggregate_max_cells, aggregate_max_results)
    bezier_kwargs = _bezier_limit_kwargs(kwargs)

    for a, b in candidates:
        seg = curve_segs[a.object]
        patch = surf_patches[b.object]

        if rational:
            pts_c = to_homogeneous_1d(seg.control_points, seg.weights)
            pts_s = to_homogeneous_2d(patch.control_points, patch.weights)
        else:
            pts_c = seg.control_points
            pts_s = patch.control_points

        seg_interval = seg.interval()
        patch_interval = patch.interval()  # ((u0, u1), (v0, v1))
        context = f"nurbs_csx spans {seg_interval} x {patch_interval}"
        remaining_cells, remaining_results = _remaining_allowances(status)
        if remaining_cells <= 0 or remaining_results <= 0:
            _mark_incomplete(
                status, context, return_status,
                "aggregate CSX cell/result budget exhausted")
            break
        call_kwargs = dict(bezier_kwargs)
        call_kwargs['max_cells'] = remaining_cells
        call_kwargs['max_results'] = remaining_results
        result = bez_csx_v4(
            pts_c, pts_s, atol=tol, rational=rational, **call_kwargs,
        )

        result, stop_after_span = _consume_bezier_status(
            result, status, seg_interval, patch_interval, return_status,
            remaining_cells, remaining_results,
        )

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
        if stop_after_span:
            break

    # ---------------------------------------------------------------
    # Post-processing: merge overlaps, classify micro-fragments
    # ---------------------------------------------------------------
    ptol_t = float(nurbs_curve_param_tolerance(curve, tol))

    # 1. Merge adjacent overlaps by t-range
    merged_overlaps = _merge_overlaps_by_t(raw_overlaps, ptol_t)

    # 2. Parametric dedup of isolated points
    deduped_isolated = _dedup_csx_isolated(raw_isolated, curve, surface, tol)

    # 3. Classify micro-fragments: isolated points adjacent to overlaps
    #    become part of the overlap; others remain isolated
    if merged_overlaps and deduped_isolated:
        final_isolated = []
        for iso in deduped_isolated:
            t_iso = iso['t']
            absorbed = False
            for ovl in merged_overlaps:
                t_lo, t_hi = ovl['t_range']
                # If isolated point is within ptol of an overlap endpoint,
                # extend the overlap to include it
                if abs(t_iso - t_lo) < ptol_t * 10:
                    ovl['t_range'] = (t_iso, t_hi)
                    absorbed = True
                    break
                elif abs(t_iso - t_hi) < ptol_t * 10:
                    ovl['t_range'] = (t_lo, t_iso)
                    absorbed = True
                    break
                elif t_lo - ptol_t <= t_iso <= t_hi + ptol_t:
                    absorbed = True
                    break
            if not absorbed:
                final_isolated.append(iso)
        deduped_isolated = final_isolated

    # Remove the _micro tag from results
    for iso in deduped_isolated:
        iso.pop('_micro', None)

    isolated = deduped_isolated if deduped_isolated else None
    overlaps = merged_overlaps if merged_overlaps else None

    if return_status:
        return isolated, overlaps, status
    return isolated, overlaps

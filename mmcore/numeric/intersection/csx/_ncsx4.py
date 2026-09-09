"""NURBS curve-surface intersection using the v4 sq-dist Bezier CSX.

Drop-in replacement for nurbs_csx from _ncsx.py, using _bez_csx4.bez_csx
instead of the old recursive subdivision approach.
"""
from __future__ import annotations

import numpy as np

from mmcore.nurbs._core import NURBSCurve, NURBSSurface
from mmcore.nurbs._nurbs_eval import (
    NURBSCurveTuple, NURBSSurfaceTuple,
    to_homogeneous_1d, to_homogeneous_2d,
    _nurbs_to_tuple,
)
from mmcore.nurbs._nurbs_knots import decompose_curve, decompose_surface
from mmcore.nurbs._nurbs_param_tol import nurbs_curve_param_tolerance
from mmcore.numeric.bvh.lbvh import AABB, build_bvh, bvh_intersect

from mmcore.numeric.intersection.csx._bez_csx4 import bez_csx as bez_csx_v4
from mmcore.numeric.intersection._parameter_mapping import (
    map_isolated, map_overlap, reject_parameter_aliases, strip_mapping_metadata,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _is_rational_curve(curve: NURBSCurveTuple) -> bool:
    return not np.all(curve.weights == 1.0)


def _is_rational_surface(surface: NURBSSurfaceTuple) -> bool:
    return not np.all(surface.weights == 1.0)


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


_BEZIER_LIMIT_KWARGS = ('max_depth', 'tolerance_tier')
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
    """Join only exact paired endpoints; preserve every other correspondence.

    A common curve-parameter interval does not identify a surface preimage.
    The range endpoints retain their orientation, including decreasing UV.
    ``ptol_t`` remains accepted for compatibility, but cannot prove identity.
    """
    result = []
    for overlap in sorted(overlaps or (), key=lambda o: o['t_range'][0]):
        current = dict(overlap)
        if not result:
            result.append(current)
            continue
        previous = result[-1]
        keys = ('t_range', 'u_range', 'v_range')
        # Some exact planar proofs expose UV bounding boxes, not paired
        # endpoint values. Neither identity nor adjacency follows from
        # matching bounding boxes.
        if (previous.get('uv_range_is_enclosure', False)
                or current.get('uv_range_is_enclosure', False)):
            result.append(current)
            continue
        prior_exact = previous.get('_exact_global_ranges')
        current_exact = current.get('_exact_global_ranges')
        same_provenance = (prior_exact == current_exact
                           if prior_exact is not None or current_exact is not None else True)
        if same_provenance and all(tuple(previous[k]) == tuple(current[k]) for k in keys):
            continue
        joined_provenance = (all(a[1] == b[0] for a, b in zip(prior_exact, current_exact))
                             if prior_exact is not None and current_exact is not None
                             else prior_exact is current_exact)
        if joined_provenance and all(previous[k][1] == current[k][0] for k in keys):
            for key in keys:
                previous[key] = (previous[key][0], current[key][1])
            if prior_exact is not None:
                previous['_exact_global_ranges'] = tuple(
                    (a[0], b[1]) for a, b in zip(prior_exact, current_exact))
                previous.setdefault('joined_local_certificates', []).append(
                    current['local_overlap_certificate'])
        else:
            result.append(current)
    return result


# ---------------------------------------------------------------------------
# Parametric deduplication
# ---------------------------------------------------------------------------

def _is_seam_duplicate(u1, u2, v1, v2, surface, ptol_u, ptol_v):
    """Exact paired endpoints of a proved clamped C0 surface seam."""
    bounds = surface.interval()
    parameters = ((u1, u2, v1, v2), (v1, v2, u1, u2))
    for axis, (a, b, other_a, other_b) in enumerate(parameters):
        low, high = bounds[axis]
        if other_a != other_b or {a, b} != {low, high} or low == high:
            continue
        order = surface.order_u if axis == 0 else surface.order_v
        knots = np.asarray(surface.knot_u if axis == 0 else surface.knot_v)
        if not (np.all(knots[:order] == low) and np.all(knots[-order:] == high)):
            continue
        cp, weights = surface.control_points, surface.weights
        first, last = (cp[0], cp[-1]) if axis == 0 else (cp[:, 0], cp[:, -1])
        w_first, w_last = ((weights[0], weights[-1]) if axis == 0
                           else (weights[:, 0], weights[:, -1]))
        if np.array_equal(first, last) and np.array_equal(w_first, w_last):
            return True
    return False


def _dedup_csx_isolated(entries, curve, surface, tol):
    """Retain distinct parameter roots, including within modeling tolerance.

    Without a common isolating certificate, only identical paired parameters
    or exact clamped seam endpoints authorize destructive deduplication.
    """
    if len(entries) <= 1:
        return entries

    sorted_entries = sorted(entries, key=lambda e: e['t'])

    deduped = [sorted_entries[0]]
    for entry in sorted_entries[1:]:
        prev = deduped[-1]

        # Standard parametric proximity check
        is_dup = all(entry[k] == prev[k] for k in ('t', 'u', 'v'))

        # Periodic seam check: same t, same v, u at opposite domain ends
        if not is_dup and entry['t'] == prev['t']:
            is_dup = _is_seam_duplicate(
                entry['u'], prev['u'], entry['v'], prev['v'],
                surface, 0., 0.,
            )

        if is_dup:
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
    tolerance_tier : bool, optional
        Forwarded to the Bezier solver; False requests exact zero-set
        search and explicit unresolved status when it cannot be certified.

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
            mapped = map_isolated(
                iso, ('t', 'u', 'v'), (seg_interval, *patch_interval),
                ((pts_c, (0,)), (pts_s, (1, 2))), rational, tol,
                status, context, return_status,
            )
            if mapped is not None:
                raw_isolated.append(mapped)

        for ovl in result['overlaps']:
            enclosure = bool(ovl.get('uv_range_is_enclosure', False))
            mapped = map_overlap(
                ovl, ('t', 'u', 'v'), (seg_interval, *patch_interval),
                ((pts_c, (0,)), (pts_s, (1, 2))), rational, tol,
                status, context, return_status,
                enclosure_keys=('u', 'v') if enclosure else ())
            if mapped is not None:
                mapped['uv_range_is_enclosure'] = enclosure
                raw_overlaps.append(mapped)
        if stop_after_span and min(_remaining_allowances(status)) <= 0:
            break

    # ---------------------------------------------------------------
    # Post-processing: preserve paired-parameter correspondence
    # ---------------------------------------------------------------
    ptol_t = float(nurbs_curve_param_tolerance(curve, tol))

    # 1. Join exactly matching paired overlap endpoints
    merged_overlaps = _merge_overlaps_by_t(raw_overlaps, ptol_t)

    # 2. Exact paired-preimage deduplication
    raw_isolated = reject_parameter_aliases(
        raw_isolated, ('t', 'u', 'v'), status, 'nurbs_csx assembly', return_status)
    deduped_isolated = _dedup_csx_isolated(raw_isolated, curve, surface, tol)

    # An isolated root may lie on another sheet over the same t interval.
    # Endpoint identity is sufficient; t proximity alone is not.
    if merged_overlaps and deduped_isolated:
        deduped_isolated = [iso for iso in deduped_isolated if not any(
            (iso['_exact_global_parameters'] == tuple(
                interval[endpoint] for interval in overlap['_exact_global_ranges']))
            for overlap in merged_overlaps
            if not overlap.get('uv_range_is_enclosure', False)
            for endpoint in (0, 1))]

    # Remove the _micro tag from results
    for iso in strip_mapping_metadata(deduped_isolated):
        iso.pop('_micro', None)
    for overlap in merged_overlaps:
        overlap.pop('_exact_global_ranges', None)

    isolated = deduped_isolated if deduped_isolated else None
    overlaps = merged_overlaps if merged_overlaps else None

    if return_status:
        return isolated, overlaps, status
    return isolated, overlaps

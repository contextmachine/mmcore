"""NURBS curve-curve intersection using the v4 sq-dist Bezier CCX.

Drop-in replacement for nurbs_ccx / nurbs_ccx_multiple from _nccx.py,
using _bez_ccx4.bez_ccx instead of _bez_ccx3.bez_ccx.
"""
from __future__ import annotations

import numpy as np
from numpy import ndarray, dtype

from mmcore.geom.nurbs import NURBSCurve
from mmcore.geom._nurbs_eval import (
    NURBSCurveTuple, to_homogeneous_1d, _nurbs_to_tuple,
)
from mmcore.geom._nurbs_knots import decompose_curve
from mmcore.geom._nurbs_param_tol import nurbs_curve_param_tolerance
from mmcore.geom.bvh.lbvh import build_bvh, AABB, bvh_intersect
from mmcore.numeric.intersection.ccx._bez_ccx4 import bez_ccx as bez_ccx_v4
from mmcore.numeric.intersection._bezier_common import eval_curve

# ---------------------------------------------------------------------------
# Dtypes (self-contained, no dependency on _nccx.py)
# ---------------------------------------------------------------------------

_ccx_isolated_dtype = lambda dim: [
    ('u', np.float64), ('v', np.float64), ('point', np.float64, (dim,)),
]
_ccx_overlap_dtype = lambda dim: [
    ('u', np.float64, (2,)), ('v', np.float64, (2,)), ('point', np.float64, (2, dim)),
]
_curves_ref_dtype = [('curve1_i', np.uint64), ('curve2_i', np.uint64)]
_multiple_ccx_isolated_dtype = lambda dim: _ccx_isolated_dtype(dim) + _curves_ref_dtype
_multiple_ccx_overlap_dtype = lambda dim: _ccx_overlap_dtype(dim) + _curves_ref_dtype


def _is_rational(curve: NURBSCurveTuple) -> bool:
    return not np.allclose(curve.weights, 1.0)


def _map_local_to_global(u_loc, v_loc, u0, u1, v0, v1):
    return (u0 + (u1 - u0) * u_loc, v0 + (v1 - v0) * v_loc)


_BEZIER_LIMIT_KWARGS = ('max_depth',)
_DEFAULT_MAX_CELLS = 100_000
_DEFAULT_MAX_RESULTS = 4_096


def _bezier_limit_kwargs(kwargs):
    """Forward only the bounded-solver controls understood by bez_ccx v4."""
    return {name: kwargs[name] for name in _BEZIER_LIMIT_KWARGS
            if name in kwargs}


def _new_status(max_cells, max_results):
    return {
        'complete': True,
        'budget_exhausted': False,
        'boundary_topology_complete': True,
        'cells_processed': 0,
        'max_cells': int(max_cells),
        'results_processed': 0,
        'max_results': int(max_results),
        'partial_results': 0,
    }


def _mark_incomplete(status, context, return_status, message):
    status['complete'] = False
    status['budget_exhausted'] = True
    status['partial_results'] += 1
    if not return_status:
        raise RuntimeError(f"{context}: {message}; pass return_status=True "
                           "to receive explicit partial status")


def _remaining_allowances(status):
    return (
        max(0, status['max_cells'] - status['cells_processed']),
        max(0, status['max_results'] - status['results_processed']),
    )


def _consume_bezier_status(
    result, status, context, return_status, cell_allowance, result_allowance,
):
    """Aggregate and sanitize one span result under call-wide allowances."""
    result = dict(result)
    reported_cells = max(0, int(result.get('cells_processed', 0)))
    # Even an AABB-fast-rejected span consumed one adapter dispatch.  The
    # one-unit floor prevents an arbitrarily large BVH candidate set from
    # bypassing the aggregate allowance through zero-cell sub-results.
    charged_cells = max(1, reported_cells)
    cells_overrun = charged_cells > cell_allowance
    status['cells_processed'] += min(charged_cells, cell_allowance)

    kept = 0
    result_overrun = False
    for key in ('isolated', 'overlaps'):
        values = list(result.get(key, ()) or ())
        remaining = max(0, result_allowance - kept)
        if len(values) > remaining:
            values = values[:remaining]
            result_overrun = True
        result[key] = values
        kept += len(values)
    status['results_processed'] += kept

    exhausted = (bool(result.get('budget_exhausted', False))
                 or cells_overrun or result_overrun)
    topology_complete = bool(result.get('boundary_topology_complete', True))
    incomplete = exhausted or not topology_complete

    status['budget_exhausted'] |= exhausted
    status['boundary_topology_complete'] &= topology_complete
    if incomplete:
        status['complete'] = False
        status['partial_results'] += 1
        if not return_status:
            raise RuntimeError(
                f"{context}: incomplete Bezier CCX result "
                "(budget exhausted or boundary topology incomplete); "
                "pass return_status=True to receive explicit partial status"
            )
    return result, incomplete


# ---------------------------------------------------------------------------
# Parametric deduplication
# ---------------------------------------------------------------------------

def _dedup_isolated(entries, curves, tol):
    """Deduplicate isolated intersections using parametric tolerances.

    Duplicates arise only at Bezier span boundaries: decompose_curve splits
    at knots, so the same knot-intersection appears from both adjacent
    segments.  We group by canonical curve pair, compute the parametric
    tolerance for each curve, and merge entries whose parameters on BOTH
    curves are within their respective tolerances.

    Parameters
    ----------
    entries : list[dict]
        Raw results, each with keys 'u', 'v', 'curve1_i', 'curve2_i', 'point'.
        The 'u' param is on curve1_i, 'v' is on curve2_i, in GLOBAL NURBS
        parameter space.
    curves : list[NURBSCurveTuple]
        The original NURBS curves (for parametric tolerance computation).
    tol : float
        Geometric tolerance (atol).

    Returns
    -------
    list[dict]
        Deduplicated entries.
    """
    if len(entries) <= 1:
        return entries

    # Pre-compute parametric tolerance for each curve
    ptols = {}
    for e in entries:
        for ci in (e['curve1_i'], e['curve2_i']):
            if ci not in ptols:
                ptols[ci] = float(nurbs_curve_param_tolerance(curves[ci], tol))

    # Canonicalize: ensure curve1_i < curve2_i, swapping u/v if needed
    canonical = []
    for e in entries:
        c1, c2 = int(e['curve1_i']), int(e['curve2_i'])
        u, v = float(e['u']), float(e['v'])
        pt = e['point']
        if c1 <= c2:
            canonical.append((c1, c2, u, v, pt))
        else:
            canonical.append((c2, c1, v, u, pt))

    # Sort by (curve pair, u parameter)
    canonical.sort(key=lambda x: (x[0], x[1], x[2]))

    # Walk and merge within each curve pair
    kept = [canonical[0]]
    for entry in canonical[1:]:
        c1, c2, u, v, pt = entry
        prev_c1, prev_c2, prev_u, prev_v, prev_pt = kept[-1]

        if c1 == prev_c1 and c2 == prev_c2:
            ptol_a = ptols[c1]
            ptol_b = ptols[c2]
            if abs(u - prev_u) < ptol_a and abs(v - prev_v) < ptol_b:
                # Duplicate — skip (keep the earlier one, which was sorted first)
                continue

        kept.append(entry)

    # Convert back to dict format (un-canonicalize is not needed —
    # the canonical order is fine for the output)
    result = []
    for c1, c2, u, v, pt in kept:
        result.append({
            'u': u, 'v': v, 'point': pt,
            'curve1_i': c1, 'curve2_i': c2,
        })
    return result


def _dedup_isolated_pair(entries, curve1, curve2, tol):
    """Deduplicate isolated intersections for a single curve pair (nurbs_ccx).

    Same logic as _dedup_isolated but for two specific curves,
    without curve indices.
    """
    if len(entries) <= 1:
        return entries

    ptol_u = float(nurbs_curve_param_tolerance(curve1, tol))
    ptol_v = float(nurbs_curve_param_tolerance(curve2, tol))

    # Sort by u
    sorted_entries = sorted(entries, key=lambda e: e['u'])

    kept = [sorted_entries[0]]
    for entry in sorted_entries[1:]:
        prev = kept[-1]
        if abs(entry['u'] - prev['u']) < ptol_u and abs(entry['v'] - prev['v']) < ptol_v:
            continue
        kept.append(entry)

    return kept


# ---------------------------------------------------------------------------
# nurbs_ccx: two-curve intersection
# ---------------------------------------------------------------------------
from mmcore.geom._nurbs_eval import evaluate_nurbs_curve
def nurbs_ccx(
    curve1, curve2, tol: float = 1e-3, *, return_status: bool = True,
    **kwargs,
):
    """Find all intersections between two NURBS curves.

    Parameters
    ----------
    curve1, curve2 : NURBSCurve or NURBSCurveTuple
    tol : float
        Geometric tolerance.

    Returns
    -------
    isolated : ndarray or None
        Structured array with fields 'u', 'v', 'point'.
    overlaps : ndarray or None
        Structured array with fields 'u', 'v', 'point' (endpoint pairs).
    status : dict
        Third value, returned by default: aggregate bounded-solver
        diagnostics; read ``status['complete']`` before trusting the
        output as the whole truth (ledger L41 — the former
        raise-on-incomplete default crashed production callers on
        near-coincident input). Pass ``return_status=False`` for the
        legacy two-value shape, which raises ``RuntimeError`` on any
        incomplete sub-solve instead (fail-fast opt-in).
    """
    dim = max(crv.control_points.shape[1] for crv in (curve1, curve2))
    if isinstance(curve1, NURBSCurve):
        curve1 = _nurbs_to_tuple(curve1)
    if isinstance(curve2, NURBSCurve):
        curve2 = _nurbs_to_tuple(curve2)

    curves1 = decompose_curve(curve1)
    curves2 = decompose_curve(curve2)

    bvh1 = build_bvh([AABB.from_points(c.control_points).offset(tol) for c in curves1])
    bvh2 = build_bvh([AABB.from_points(c.control_points).offset(tol) for c in curves2])

    # Per-pair rationality
    rational = _is_rational(curve1) or _is_rational(curve2)

    raw_isolated = []
    raw_overlaps_u, raw_overlaps_v, raw_overlaps_xyz = [], [], []
    candidates = list(bvh_intersect(bvh1, bvh2, exact=False))
    # The DEFAULT aggregate allowance scales with the candidate-pair count:
    # a flat per-call total that a handful of ordinary rational span pairs
    # can exhaust (~25k cells each, ledger L41 / review finding 2) is a
    # mispriced exchange rate, not a safety property. The scaled default
    # matches the pre-aggregate per-pair budgets in the worst case while
    # staying one SHARED ledger (a hog pair may borrow from cheap ones).
    # An explicit ``max_cells`` remains an absolute promise.
    aggregate_max_cells = kwargs.get('max_cells')
    if aggregate_max_cells is None:
        aggregate_max_cells = _DEFAULT_MAX_CELLS * max(1, len(candidates))
    aggregate_max_cells = max(0, int(aggregate_max_cells))
    aggregate_max_results = max(
        0, int(kwargs.get('max_results', _DEFAULT_MAX_RESULTS)))
    status = _new_status(aggregate_max_cells, aggregate_max_results)
    bezier_kwargs = _bezier_limit_kwargs(kwargs)

    for a, b in candidates:
        _c1 = curves1[a.object]
        _c2 = curves2[b.object]

        if rational:
            pts1 = to_homogeneous_1d(_c1.control_points, _c1.weights)
            pts2 = to_homogeneous_1d(_c2.control_points, _c2.weights)
        else:
            pts1 = _c1.control_points
            pts2 = _c2.control_points

        context = (f"nurbs_ccx spans {_c1.interval()} x "
                   f"{_c2.interval()}")
        remaining_cells, remaining_results = _remaining_allowances(status)
        if remaining_cells <= 0 or remaining_results <= 0:
            _mark_incomplete(
                status, context, return_status,
                "aggregate CCX cell/result budget exhausted")
            break
        call_kwargs = dict(bezier_kwargs)
        call_kwargs['max_cells'] = remaining_cells
        call_kwargs['max_results'] = remaining_results
        result = bez_ccx_v4(
            pts1, pts2, atol=tol, rational=rational, **call_kwargs,
        )
        result, stop_after_span = _consume_bezier_status(
            result, status, context, return_status,
            remaining_cells, remaining_results,
        )

        for inter in result['isolated']:
            u_glob, v_glob = _map_local_to_global(
                inter['u'], inter['v'], *_c1.interval(), *_c2.interval(),
            )
            # Verify NURBS-level distance (Bezier-level may be valid at knot
            # seams but NURBS-level can differ)

            pt1 = evaluate_nurbs_curve(curve1, u_glob, 0)['C']
            pt2 = evaluate_nurbs_curve(curve2, v_glob, 0)['C']
            if float(np.linalg.norm(pt1 - pt2)) >= tol:
                continue
            raw_isolated.append({'u': u_glob, 'v': v_glob, 'point': inter['point']})

        for overlap in result['overlaps']:
            ur = overlap.get('u_range', (0.0, 1.0))
            vr = overlap.get('v_range', (0.0, 1.0))
            u0g, v0g = _map_local_to_global(ur[0], vr[0], *_c1.interval(), *_c2.interval())
            u1g, v1g = _map_local_to_global(ur[1], vr[1], *_c1.interval(), *_c2.interval())
            raw_overlaps_u.append([u0g, u1g])
            raw_overlaps_v.append([v0g, v1g])
            pt0 = eval_curve(pts1, ur[0], rational=rational)
            pt1 = eval_curve(pts1, ur[1], rational=rational)
            raw_overlaps_xyz.append([pt0, pt1])
        if stop_after_span:
            break

    # Dedup isolated
    deduped = _dedup_isolated_pair(raw_isolated, curve1, curve2, tol)

    # Pack into structured arrays
    if not deduped:
        isolated = None
    else:
        isolated = np.zeros(len(deduped), dtype=_ccx_isolated_dtype(dim))
        isolated['u'] = [e['u'] for e in deduped]
        isolated['v'] = [e['v'] for e in deduped]
        isolated['point'] = [e['point'] for e in deduped]

    if not raw_overlaps_u:
        overlaps = None
    else:
        overlaps = np.zeros(len(raw_overlaps_u), dtype=_ccx_overlap_dtype(dim))
        overlaps['u'] = raw_overlaps_u
        overlaps['v'] = raw_overlaps_v
        overlaps['point'] = raw_overlaps_xyz

    if return_status:
        return isolated, overlaps, status
    return isolated, overlaps


# ---------------------------------------------------------------------------
# nurbs_ccx_multiple: multi-curve intersection
# ---------------------------------------------------------------------------

def nurbs_ccx_multiple(
    curves: list[NURBSCurveTuple],
    tol: float = 1e-3,
    self_intersections: bool = False,
    *,
    return_status: bool = True,
    **kwargs,
):
    """Find all pairwise intersections among multiple NURBS curves.

    Parameters
    ----------
    curves : list[NURBSCurveTuple]
    tol : float
        Geometric tolerance.
    self_intersections : bool
        Whether to check for self-intersections within each curve.

    Returns
    -------
    isolated : ndarray or None
        Structured array with 'u', 'v', 'point', 'curve1_i', 'curve2_i'.
    overlaps : ndarray or None
        Structured array with 'u', 'v', 'point', 'curve1_i', 'curve2_i'.
    status : dict
        Third value, returned by default; read ``status['complete']``
        before trusting the output as the whole truth. Pass
        ``return_status=False`` for the legacy two-value shape, which
        raises ``RuntimeError`` on any incomplete Bezier pair instead.
    """
    dim = max(crv.control_points.shape[1] for crv in curves)

    # Decompose all curves into Bezier segments, build segment map
    counter = 0
    segm_map = {}        # segment_index -> curve_index
    segments = []
    bbs = []
    rational = False

    for i in range(len(curves)):
        if not rational:
            rational = _is_rational(curves[i])
        for segm in decompose_curve(curves[i]):
            segm_map[counter] = i
            counter += 1
            bb = AABB.from_points(segm.control_points)
            bb.offset_inplace(tol)
            bbs.append(bb)
            segments.append(segm)

    bvh = build_bvh(bbs)
    int_candidates = bvh.build_intersection_leaves_pairs(exact=False)

    raw_isolated = []
    raw_overlaps = []
    # Candidate-scaled default allowance — see nurbs_ccx (ledger L41).
    aggregate_max_cells = kwargs.get('max_cells')
    if aggregate_max_cells is None:
        aggregate_max_cells = _DEFAULT_MAX_CELLS * max(1, len(int_candidates))
    aggregate_max_cells = max(0, int(aggregate_max_cells))
    aggregate_max_results = max(
        0, int(kwargs.get('max_results', _DEFAULT_MAX_RESULTS)))
    status = _new_status(aggregate_max_cells, aggregate_max_results)
    bezier_kwargs = _bezier_limit_kwargs(kwargs)

    for first, second in int_candidates:
        segm1_i = bvh.nodes[first].object
        segm2_i = bvh.nodes[second].object
        segm1 = segments[segm1_i]
        segm2 = segments[segm2_i]
        curve1_i = segm_map[segm1_i]
        curve2_i = segm_map[segm2_i]

        # Self-intersection filtering
        if curve1_i == curve2_i:
            if not self_intersections:
                continue
            # Skip segments that share a parameter interval (same segment)
            interval1 = segm1.interval()
            interval2 = segm2.interval()
            lo = max(interval1[0], interval2[0])
            hi = min(interval1[1], interval2[1])
            if lo < hi:
                continue

        # Per-pair rationality would be ideal, but the BVH loop checks many
        # pairs from different curves. For now, use the global flag. If only
        # a few curves are rational, the overhead is in to_homogeneous_1d
        # (cheap) not in bez_ccx itself.
        if rational:
            pts1 = to_homogeneous_1d(segm1.control_points, segm1.weights)
            pts2 = to_homogeneous_1d(segm2.control_points, segm2.weights)
        else:
            pts1 = segm1.control_points
            pts2 = segm2.control_points

        context = (
            f"nurbs_ccx_multiple curves {curve1_i} x {curve2_i}, "
            f"spans {segm1.interval()} x {segm2.interval()}")
        remaining_cells, remaining_results = _remaining_allowances(status)
        if remaining_cells <= 0 or remaining_results <= 0:
            _mark_incomplete(
                status, context, return_status,
                "aggregate CCX cell/result budget exhausted")
            break
        call_kwargs = dict(bezier_kwargs)
        call_kwargs['max_cells'] = remaining_cells
        call_kwargs['max_results'] = remaining_results
        result = bez_ccx_v4(
            pts1, pts2, atol=tol, rational=rational, **call_kwargs,
        )
        result, stop_after_span = _consume_bezier_status(
            result, status, context, return_status,
            remaining_cells, remaining_results,
        )

        for inter in result['isolated']:
            u_glob, v_glob = _map_local_to_global(
                inter['u'], inter['v'], *segm1.interval(), *segm2.interval(),
            )
            # Verify NURBS-level distance at knot seams
            from mmcore.geom._nurbs_eval import evaluate_nurbs_curve
            pt1 = evaluate_nurbs_curve(curves[curve1_i], u_glob, 0)['C']
            pt2 = evaluate_nurbs_curve(curves[curve2_i], v_glob, 0)['C']
            if float(np.linalg.norm(pt1 - pt2)) >= tol:
                continue
            raw_isolated.append({
                'u': u_glob, 'v': v_glob,
                'point': inter['point'],
                'curve1_i': curve1_i, 'curve2_i': curve2_i,
            })

        for overlap in result['overlaps']:
            ur = overlap.get('u_range', (0.0, 1.0))
            vr = overlap.get('v_range', (0.0, 1.0))
            u0g, v0g = _map_local_to_global(ur[0], vr[0], *segm1.interval(), *segm2.interval())
            u1g, v1g = _map_local_to_global(ur[1], vr[1], *segm1.interval(), *segm2.interval())
            pt0 = eval_curve(pts1, ur[0], rational=rational)
            pt1 = eval_curve(pts1, ur[1], rational=rational)
            raw_overlaps.append({
                'u': [u0g, u1g], 'v': [v0g, v1g],
                'point': [pt0, pt1],
                'curve1_i': curve1_i, 'curve2_i': curve2_i,
            })
        if stop_after_span:
            break

    # Dedup isolated using parametric tolerances
    deduped = _dedup_isolated(raw_isolated, curves, tol)

    # Pack into structured arrays
    if not deduped:
        isolated = None
    else:
        isolated = np.zeros(len(deduped), dtype=_multiple_ccx_isolated_dtype(dim))
        isolated['u'] = [e['u'] for e in deduped]
        isolated['v'] = [e['v'] for e in deduped]
        isolated['point'] = [e['point'] for e in deduped]
        isolated['curve1_i'] = [e['curve1_i'] for e in deduped]
        isolated['curve2_i'] = [e['curve2_i'] for e in deduped]

    if not raw_overlaps:
        overlaps = None
    else:
        overlaps = np.zeros(len(raw_overlaps), dtype=_multiple_ccx_overlap_dtype(dim))
        overlaps['u'] = [e['u'] for e in raw_overlaps]
        overlaps['v'] = [e['v'] for e in raw_overlaps]
        overlaps['point'] = [e['point'] for e in raw_overlaps]
        overlaps['curve1_i'] = [e['curve1_i'] for e in raw_overlaps]
        overlaps['curve2_i'] = [e['curve2_i'] for e in raw_overlaps]

    if return_status:
        return isolated, overlaps, status
    return isolated, overlaps

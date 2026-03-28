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
from mmcore.geom.bvh.lbvh import BVH, build_bvh, AABB, bvh_intersect

from mmcore.numeric.intersection.ccx._bez_ccx4 import bez_ccx as bez_ccx_v4
from mmcore.numeric.intersection.ccx._nccx import (
    _ccx_isolated_dtype, _ccx_overlap_dtype,
    _multiple_ccx_isolated_dtype, _multiple_ccx_overlap_dtype,
    _is_rational,
)


def _map_local_to_global(u_loc, v_loc, u0, u1, v0, v1):
    return (u0 + (u1 - u0) * u_loc, v0 + (v1 - v0) * v_loc)


def nurbs_ccx(curve1, curve2, tol: float = 1e-3, **kwargs):
    dim = max(crv.control_points.shape[1] for crv in (curve1, curve2))
    if isinstance(curve1, NURBSCurve):
        curve1 = _nurbs_to_tuple(curve1)
    if isinstance(curve2, NURBSCurve):
        curve2 = _nurbs_to_tuple(curve2)
    curves1 = decompose_curve(curve1)
    curves2 = decompose_curve(curve2)

    bvh1 = build_bvh([AABB.from_points(crv.control_points).offset(tol) for crv in curves1])
    bvh2 = build_bvh([AABB.from_points(crv.control_points).offset(tol) for crv in curves2])
    isolated_u, isolated_v, isolated_xyz = [], [], []
    overlaps_u, overlaps_v, overlaps_xyz = [], [], []

    rational = any((_is_rational(curve1), _is_rational(curve2)))

    for a, b in bvh_intersect(bvh1, bvh2, exact=False):
        _c1 = curves1[a.object]
        _c2 = curves2[b.object]
        if rational:
            pts1 = to_homogeneous_1d(_c1.control_points, _c1.weights)
            pts2 = to_homogeneous_1d(_c2.control_points, _c2.weights)
        else:
            pts1 = _c1.control_points
            pts2 = _c2.control_points

        result = bez_ccx_v4(pts1, pts2, atol=tol, rational=rational)

        for inter in result['isolated']:
            u, v = inter['u'], inter['v']
            u_glob, v_glob = _map_local_to_global(u, v, *_c1.interval(), *_c2.interval())
            isolated_u.append(u_glob)
            isolated_v.append(v_glob)
            isolated_xyz.append(inter['point'])

        for overlap in result['overlaps']:
            # v4 overlaps have u_range/v_range instead of uv_path/xyz_path.
            # Extract endpoints and map to global params.
            ur = overlap.get('u_range', (0.0, 1.0))
            vr = overlap.get('v_range', (0.0, 1.0))
            u0g, v0g = _map_local_to_global(ur[0], vr[0], *_c1.interval(), *_c2.interval())
            u1g, v1g = _map_local_to_global(ur[1], vr[1], *_c1.interval(), *_c2.interval())
            overlaps_u.append([u0g, u1g])
            overlaps_v.append([v0g, v1g])
            # Evaluate endpoints geometrically
            from mmcore.numeric.intersection._bezier_common import eval_curve
            pt0 = eval_curve(pts1, ur[0], rational=rational)
            pt1 = eval_curve(pts1, ur[1], rational=rational)
            overlaps_xyz.append([pt0, pt1])

    if len(isolated_u) == 0:
        isolated = None
    else:
        isolated = np.zeros(len(isolated_u), dtype=_ccx_isolated_dtype(dim))
        isolated['u'] = isolated_u
        isolated['v'] = isolated_v
        isolated['point'] = isolated_xyz

    if len(overlaps_u) == 0:
        overlaps = None
    else:
        overlaps = np.zeros(len(overlaps_u), dtype=_ccx_overlap_dtype(dim))
        overlaps['u'] = overlaps_u
        overlaps['v'] = overlaps_v
        overlaps['point'] = overlaps_xyz

    return isolated, overlaps


def nurbs_ccx_multiple(curves, tol: float = 1e-3, self_intersections: bool = False, **kwargs):
    counter = 0
    segm_map = {}
    segments = []
    bbs = []
    rational = False
    dim = max(crv.control_points.shape[1] for crv in curves)

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
    isolated_u, isolated_v, isolated_xyz, isolated_crv1, isolated_crv2 = [], [], [], [], []
    overlaps_u, overlaps_v, overlaps_xyz, overlaps_crv1, overlaps_crv2 = [], [], [], [], []

    for first, second in int_candidates:
        segm1_i = bvh.nodes[first].object
        segm2_i = bvh.nodes[second].object
        segm1 = segments[segm1_i]
        segm2 = segments[segm2_i]
        curve1_i = segm_map[segm1_i]
        curve2_i = segm_map[segm2_i]

        if curve1_i == curve2_i:
            if not self_intersections:
                continue
            interval1 = segm1.interval()
            interval2 = segm2.interval()
            lo = max(interval1[0], interval2[0])
            hi = min(interval1[1], interval2[1])
            if lo < hi:
                continue

        if rational:
            pts1 = to_homogeneous_1d(segm1.control_points, segm1.weights)
            pts2 = to_homogeneous_1d(segm2.control_points, segm2.weights)
        else:
            pts1 = segm1.control_points
            pts2 = segm2.control_points

        result = bez_ccx_v4(pts1, pts2, atol=tol, rational=rational)

        for inter in result['isolated']:
            u, v = inter['u'], inter['v']
            u_glob, v_glob = _map_local_to_global(u, v, *segm1.interval(), *segm2.interval())
            isolated_u.append(u_glob)
            isolated_v.append(v_glob)
            isolated_xyz.append(inter['point'])
            isolated_crv1.append(curve1_i)
            isolated_crv2.append(curve2_i)

        for overlap in result['overlaps']:
            ur = overlap.get('u_range', (0.0, 1.0))
            vr = overlap.get('v_range', (0.0, 1.0))
            u0g, v0g = _map_local_to_global(ur[0], vr[0], *segm1.interval(), *segm2.interval())
            u1g, v1g = _map_local_to_global(ur[1], vr[1], *segm1.interval(), *segm2.interval())
            overlaps_u.append([u0g, u1g])
            overlaps_v.append([v0g, v1g])
            from mmcore.numeric.intersection._bezier_common import eval_curve
            pt0 = eval_curve(pts1, ur[0], rational=rational)
            pt1 = eval_curve(pts1, ur[1], rational=rational)
            overlaps_xyz.append([pt0, pt1])
            overlaps_crv1.append(curve1_i)
            overlaps_crv2.append(curve2_i)

    if len(isolated_u) == 0:
        isolated = None
    else:
        isolated = np.zeros(len(isolated_u), dtype=_multiple_ccx_isolated_dtype(dim))
        isolated['u'] = isolated_u
        isolated['v'] = isolated_v
        isolated['point'] = isolated_xyz
        isolated['curve1_i'] = isolated_crv1
        isolated['curve2_i'] = isolated_crv2

    if len(overlaps_u) == 0:
        overlaps = None
    else:
        overlaps = np.zeros(len(overlaps_u), dtype=_multiple_ccx_overlap_dtype(dim))
        overlaps['u'] = overlaps_u
        overlaps['v'] = overlaps_v
        overlaps['point'] = overlaps_xyz
        overlaps['curve1_i'] = overlaps_crv1
        overlaps['curve2_i'] = overlaps_crv2

    return isolated, overlaps

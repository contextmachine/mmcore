from __future__ import annotations


import numpy as np
import dataclasses
import functools
import sys
import rich


from numpy import ndarray, dtype, void

from mmcore.geom._nurbs_eval import (
    NURBSCurveTuple,
    to_homogeneous_1d,
    evaluate_nurbs_curve,
    _nurbs_to_tuple,
    _tuple_to_nurbs,
    NURBSSurfaceTuple,
    to_homogeneous_2d,
)
from mmcore.geom._nurbs_param_tol import nurbs_curve_param_tolerance
from mmcore.numeric.intersection.ccx._utils import merge_intervals_nd
from mmcore.numeric.intersection.csx._bez_csx3 import bez_csx,map_local_to_global_3,OverlapIntersection,IsolatedIntersection
from mmcore.geom._nurbs_knots import _curve_interval, decompose_curve, split_curve_multiple, decompose_surface
from mmcore.geom.bvh.lbvh import BVH, build_bvh, AABB, bvh_intersect

from numpy.typing import NDArray, DTypeLike

_csx_isolated_dtype = [("t", np.float64),("u", np.float64), ("v", np.float64), ("point", np.float64, (3,))]
_csx_overlap_dtype = [("t", np.float64, (2,)),("u", np.float64, (2,)), ("v", np.float64, (2,)), ("point", np.float64, (2, 3))]


def _is_rational(curve: NURBSCurveTuple|NURBSSurfaceTuple):
    return not np.allclose(curve.weights, 1)

def _curve_closed_c1(curve: NURBSCurveTuple, tol: float) -> bool:
    t0, t1 = curve.interval()
    if not (np.isfinite(t0) and np.isfinite(t1)):
        return False
    if np.isclose(t0, t1):
        return False
    ev0 = evaluate_nurbs_curve(curve, t0, d_order=1)
    ev1 = evaluate_nurbs_curve(curve, t1, d_order=1)
    if not np.allclose(ev0["C"], ev1["C"], atol=tol, rtol=1e-6):
        return False
    if not np.allclose(ev0["C1"], ev1["C1"], atol=tol, rtol=1e-6):
        return False
    return True


def _make_overlap_record(t0, t1, u0, u1, v0, v1, p0, p1):
    if t1 < t0:
        t0, t1 = t1, t0
        u0, u1 = u1, u0
        v0, v1 = v1, v0
        p0, p1 = p1, p0
    return {
        "t0": float(t0),
        "t1": float(t1),
        "u0": float(u0),
        "u1": float(u1),
        "v0": float(v0),
        "v1": float(v1),
        "p0": np.asarray(p0, dtype=float),
        "p1": np.asarray(p1, dtype=float),
    }


def _merge_overlaps_linear(overlaps, t_tol: float):
    if not overlaps:
        return []
    overlaps_sorted = sorted(overlaps, key=lambda o: o["t0"])
    merged = [overlaps_sorted[0].copy()]
    for ov in overlaps_sorted[1:]:
        cur = merged[-1]
        if (ov["t0"] - t_tol) <= (cur["t1"] + t_tol):
            if ov["t0"] < cur["t0"]:
                cur["t0"] = ov["t0"]
                cur["u0"] = ov["u0"]
                cur["v0"] = ov["v0"]
                cur["p0"] = ov["p0"]
            if ov["t1"] > cur["t1"]:
                cur["t1"] = ov["t1"]
                cur["u1"] = ov["u1"]
                cur["v1"] = ov["v1"]
                cur["p1"] = ov["p1"]
        else:
            merged.append(ov.copy())
    return merged


def _merge_overlaps_closed(overlaps, t_tol: float, t_min: float, t_max: float):
    if len(overlaps) < 2:
        return overlaps
    period = t_max - t_min
    if not np.isfinite(period) or period <= 0:
        return overlaps
    first = overlaps[0]
    last = overlaps[-1]
    gap = (t_max - last["t1"]) + (first["t0"] - t_min)
    if gap <= (2.0 * t_tol):
        merged = {
            "t0": last["t0"],
            "t1": first["t1"],
            "u0": last["u0"],
            "u1": first["u1"],
            "v0": last["v0"],
            "v1": first["v1"],
            "p0": last["p0"],
            "p1": first["p1"],
        }
        return overlaps[1:-1] + [merged]
    return overlaps


def _overlap_length(ov, t_min: float, t_max: float):
    t0, t1 = ov["t0"], ov["t1"]
    if t0 <= t1:
        return t1 - t0
    return (t_max - t0) + (t1 - t_min)


def _overlap_midpoint(ov, t_min: float, t_max: float):
    t0, t1 = ov["t0"], ov["t1"]
    if t0 <= t1:
        return 0.5 * (t0 + t1)
    length = _overlap_length(ov, t_min, t_max)
    mid = t0 + 0.5 * length
    if mid > t_max:
        mid = t_min + (mid - t_max)
    return mid


def _t_in_overlap(t: float, ov, t_tol: float, t_min: float, t_max: float):
    t0, t1 = ov["t0"], ov["t1"]
    if t0 <= t1:
        return (t >= (t0 - t_tol)) and (t <= (t1 + t_tol))
    return (t >= (t0 - t_tol)) or (t <= (t1 + t_tol))


def _dedup_isolated_by_t(points, t_tol: float, closed: bool, t_min: float, t_max: float):
    if not points:
        return []
    pts = sorted(points, key=lambda p: p["t"])
    merged = [pts[0]]
    merge_tol = 2.0 * t_tol
    for p in pts[1:]:
        if abs(p["t"] - merged[-1]["t"]) <= merge_tol:
            continue
        merged.append(p)
    if closed and len(merged) > 1:
        gap = (t_max - merged[-1]["t"]) + (merged[0]["t"] - t_min)
        if gap <= merge_tol:
            merged.pop()
    return merged


def nurbs_csx_v2(curve: NURBSCurveTuple, surface: NURBSSurfaceTuple, tol: float = 1e-3, angle_tol=0.052,**kwargs):
    if "atol" in kwargs and kwargs["atol"] is not None:
        tol = kwargs["atol"]
    overlap_dist_tol = kwargs.pop("overlap_dist_tol", tol)

    curves = decompose_curve(curve)

    surfaces = decompose_surface(surface)

    bvh1 = build_bvh([AABB.from_points(crv.control_points).offset(tol) for crv in curves])
    bvh2 = build_bvh([AABB.from_points(surf.control_points.reshape((-1,3))).offset(tol) for surf in surfaces])
    isolated_t, isolated_u, isolated_v, isolated_xyz = [],[], [], []
    overlaps_t,overlaps_u, overlaps_v, overlaps_xyz = [],[], [], []

    rational = any((_is_rational(surface), _is_rational(curve)))


    for a, b in bvh_intersect(bvh1, bvh2, exact=False):
        _c1 = curves[a.object]
        _c2 = surfaces[b.object]
        if rational:

            pts1 = to_homogeneous_1d(_c1.control_points, _c1.weights)
            pts2 = to_homogeneous_2d(_c2.control_points, _c2.weights)
        else:
            pts1 = _c1.control_points
            pts2 = _c2.control_points
        # print(tol)
        (t0, t1) =_c1.interval()
        (u0, v0), (u1, v1) = _c2.interval()

        result = bez_csx(
            pts1,
            pts2,
            atol=tol,
            rational=rational,
            overlap_dist_tol=overlap_dist_tol,angle_tol=angle_tol
        )
        if len(result["isolated"]) == 0 and len(result["overlaps"]) == 0:
            # print(set(result['stats']['pruned_by']))
            # print(pts1.tolist(),pts2.tolist())
            ...
        for inter in result["isolated"]:
            t, u, v = inter["t"], inter["u"], inter["v"]
            (u0,v0),(u1,v1)=_c2.interval()
            t_glob,u_glob, v_glob = map_local_to_global_3(t, u, v, t0,t1, u0,u1,v0,v1)
            isolated_t.append(t_glob)
            isolated_u.append(u_glob)
            isolated_v.append(v_glob)
            isolated_xyz.append(inter["point"])
        overlap: OverlapIntersection

        for overlap in result["overlaps"]:
            t_start,t_end=overlap["t_path"][0],overlap["t_path"][-1]
            u_start, u_end = overlap["uv_path"][(0, -1),0]
            v_start, v_end = overlap["uv_path"][(0, -1), 1]
            t_start_loc, u_start_loc,v_start_loc=map_local_to_global_3(t_start, u_start,v_start,  t0,t1,  u0, u1, v0, v1)
            t_end_loc, u_end_loc, v_end_loc = map_local_to_global_3(t_end, u_end, v_end, t0, t1, u0, u1, v0, v1)

            overlaps_t.append((t_start_loc,t_end_loc))
            overlaps_u.append((u_start_loc,u_end_loc))
            overlaps_v.append((v_start_loc,v_end_loc))
            overlaps_xyz.append(overlap["xyz_path"][(0, -1), :])

    t_min, t_max = curve.interval()
    t_tol = nurbs_curve_param_tolerance(curve, tol)
    if not np.isfinite(t_tol) or t_tol <= 0:
        t_tol = tol

    closed = _curve_closed_c1(curve, tol)

    isolated_records = [
        {"t": float(t), "u": float(u), "v": float(v), "point": np.asarray(p, dtype=float)}
        for t, u, v, p in zip(isolated_t, isolated_u, isolated_v, isolated_xyz)
    ]

    overlap_records = [
        _make_overlap_record(t[0], t[1], u[0], u[1], v[0], v[1], p[0], p[1])
        for t, u, v, p in zip(overlaps_t, overlaps_u, overlaps_v, overlaps_xyz)
    ]

    overlap_records = _merge_overlaps_linear(overlap_records, t_tol)
    if closed:
        overlap_records = _merge_overlaps_closed(overlap_records, t_tol, t_min, t_max)

    new_isolated = []
    kept_overlaps = []
    for ov in overlap_records:
        if _overlap_length(ov, t_min, t_max) < t_tol:
            t_mid = _overlap_midpoint(ov, t_min, t_max)
            u_mid = 0.5 * (ov["u0"] + ov["u1"])
            v_mid = 0.5 * (ov["v0"] + ov["v1"])
            p_mid = evaluate_nurbs_curve(curve, t_mid, d_order=0)["C"]
            new_isolated.append({"t": float(t_mid), "u": float(u_mid), "v": float(v_mid), "point": p_mid})
        else:
            kept_overlaps.append(ov)
    overlap_records = kept_overlaps
    isolated_records.extend(new_isolated)

    if overlap_records:
        isolated_records = [
            p for p in isolated_records
            if not any(_t_in_overlap(p["t"], ov, t_tol, t_min, t_max) for ov in overlap_records)
        ]

    isolated_records = _dedup_isolated_by_t(isolated_records, t_tol, closed, t_min, t_max)

    if len(isolated_records) == 0:
        isolated = None
    else:
        isolated = np.zeros(len(isolated_records), dtype=_csx_isolated_dtype)
        isolated["t"] = [p["t"] for p in isolated_records]
        isolated["u"] = [p["u"] for p in isolated_records]
        isolated["v"] = [p["v"] for p in isolated_records]
        isolated["point"] = [p["point"] for p in isolated_records]

    if len(overlap_records) == 0:
        overlaps = None
    else:
        overlaps = np.zeros(len(overlap_records), dtype=_csx_overlap_dtype)
        overlaps["t"] = [(ov["t0"], ov["t1"]) for ov in overlap_records]
        overlaps["u"] = [(ov["u0"], ov["u1"]) for ov in overlap_records]
        overlaps["v"] = [(ov["v0"], ov["v1"]) for ov in overlap_records]
        overlaps["point"] = [[ov["p0"], ov["p1"]] for ov in overlap_records]

    return isolated, overlaps


if __name__ == "__main__":
    from mmcore.geom._nurbs_eval import _nurbs_to_tuple
    import numpy as np

    curve1 = NURBSCurveTuple(
        order=4,
        knot=np.array(
            [
                -2.67615298,
                -2.67615298,
                -2.67615298,
                -2.67615298,
                0.0,
                0.0,
                0.0,
                3.12101814,
                3.12101814,
                3.12101814,
                6.88039589,
                6.88039589,
                6.88039589,
                6.88039589,
            ]
        ),
        control_points=np.array(
            [
                [-48.0003111, 64.08408847, 0.0],
                [-48.89236209, 64.08408847, 0.0],
                [-49.78441309, 64.08408847, 0.0],
                [-50.67646408, 64.08408847, 0.0],
                [-51.1718386, 64.99891638, 0.0],
                [-51.66721312, 65.91374429, 0.0],
                [-52.16258764, 66.82857221, 0.0],
                [-52.58835156, 67.61484744, 0.0],
                [-53.36295339, 69.04533557, 0.0],
                [-58.19051474, 67.75179441, 0.0],
            ]
        ),
        weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )

    surf1 = NURBSSurfaceTuple(
        order_u=4,
        order_v=3,
        knot_u=np.array([0.0, 0.0, 0.0, 0.0, 11.15682108, 11.15682108, 11.15682108, 11.15682108]),
        knot_v=np.array(
            [0.0, 0.0, 0.0, 1.57079633, 1.57079633, 3.14159265, 3.14159265, 4.71238898, 4.71238898, 6.28318531, 6.28318531, 6.28318531]
        ),
        control_points=np.array(
            [
                [
                    [-49.98993653, 62.81625062, 0.0],
                    [-49.98993653, 62.81625062, 1.0],
                    [-50.8692918, 62.34008435, 1.0],
                    [-51.74864706, 61.86391808, 1.0],
                    [-51.74864706, 61.86391808, 0.0],
                    [-51.74864706, 61.86391808, -1.0],
                    [-50.8692918, 62.34008435, -1.0],
                    [-49.98993653, 62.81625062, -1.0],
                    [-49.98993653, 62.81625062, 0.0],
                ],
                [
                    [-51.76077048, 66.08652041, 0.0],
                    [-51.76077048, 66.08652041, 1.0],
                    [-52.64012575, 65.61035414, 1.0],
                    [-53.51948102, 65.13418787, 1.0],
                    [-53.51948102, 65.13418787, 0.0],
                    [-53.51948102, 65.13418787, -1.0],
                    [-52.64012575, 65.61035414, -1.0],
                    [-51.76077048, 66.08652041, -1.0],
                    [-51.76077048, 66.08652041, 0.0],
                ],
                [
                    [-53.53160444, 69.3567902, 0.0],
                    [-53.53160444, 69.3567902, 1.0],
                    [-54.4109597, 68.88062393, 1.0],
                    [-55.29031497, 68.40445766, 1.0],
                    [-55.29031497, 68.40445766, 0.0],
                    [-55.29031497, 68.40445766, -1.0],
                    [-54.4109597, 68.88062393, -1.0],
                    [-53.53160444, 69.3567902, -1.0],
                    [-53.53160444, 69.3567902, 0.0],
                ],
                [
                    [-55.30243839, 72.62705999, 0.0],
                    [-55.30243839, 72.62705999, 1.0],
                    [-56.18179366, 72.15089372, 1.0],
                    [-57.06114892, 71.67472745, 1.0],
                    [-57.06114892, 71.67472745, 0.0],
                    [-57.06114892, 71.67472745, -1.0],
                    [-56.18179366, 72.15089372, -1.0],
                    [-55.30243839, 72.62705999, -1.0],
                    [-55.30243839, 72.62705999, 0.0],
                ],
            ]
        ),
        weights=np.array(
            [
                [1.0, 0.70710678, 1.0, 0.70710678, 1.0, 0.70710678, 1.0, 0.70710678, 1.0],
                [1.0, 0.70710678, 1.0, 0.70710678, 1.0, 0.70710678, 1.0, 0.70710678, 1.0],
                [1.0, 0.70710678, 1.0, 0.70710678, 1.0, 0.70710678, 1.0, 0.70710678, 1.0],
                [1.0, 0.70710678, 1.0, 0.70710678, 1.0, 0.70710678, 1.0, 0.70710678, 1.0],
            ]
        ),
    )
    print(curve1)
    print(surf1)
    isolated,overlaps=nurbs_csx_v2(curve1,surf1,atol=1e-3)
    print('\n\nisolated')
    print(isolated[
        'point'
          ].tolist())
    print('\n\noverlaps (xyz)')
    print(overlaps[ 'point'].tolist())
    print('\noverlaps (t)')
    print(overlaps["t"].tolist())

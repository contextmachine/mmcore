from __future__ import annotations

import dataclasses
import functools
import sys
import time

import numpy as np
from mmcore.geom.nurbs import NURBSCurve

from mmcore.geom._nurbs_eval import NURBSCurveTuple, evaluate_nurbs_curve
from mmcore.geom._nurbs_knots import _curve_interval, decompose_curve, split_curve_multiple
from mmcore.geom.bvh.lbvh import BVH, build_bvh, AABB, bvh_intersect
from mmcore.numeric import compute_parametric_curvature_tolerance_curve
from mmcore.numeric.approx import adaptive_curve_sampler
from mmcore.numeric.fdm import bounded_newtons_method
from mmcore.numeric.intersection.ccx.segment import segment_intersection
from mmcore.numeric.interval import Interval
from ._bez_overlap import _bez_curve_overlap, CCXInt


class CCXIntBox:
    s: Interval
    t: Interval
    f: CCXInt
    
    def __eq__(self, other: CCXIntBox):
        return self.intersects(other)
    
    def __lt__(self, other: CCXIntBox):
        return self.s.__lt__(other.s) and self.t.__lt__(other.t)
    
    def __gt__(self, other: CCXIntBox):
        return self.s.__gt__(other.s) and self.t.__gt__(other.t)
    
    def intersects(self, other) -> bool:
        return self.s.intersects(other.s) and self.t.intersects(other.t)
    
    def expand(self, other: CCXIntBox):
        self.s = Interval(min(self.s.low, other.s.low), max(self.s.upp, other.s.upp))
        
        self.t = Interval(min(self.t.low, other.t.low), max(self.t.upp, other.t.upp))
        
        self.f = min(self.f, other.f)


if sys.version_info >= (3, 10):
    
    CCXIntBox: type[CCXIntBox] = dataclasses.dataclass(slots=True)(functools.total_ordering(CCXIntBox)
                                                                   
                                                                   )




else:
    CCXIntBox = dataclasses.dataclass(functools.total_ordering(CCXIntBox)
                                      
                                      )

_CCX_INTER_NEW = 1
_CCX_INTER_NOT_NEW = 0


def _is_new_one(a, b):
    pinter = a
    pinter2 = b
    
    if pinter.intersects(pinter2):
        return _CCX_INTER_NOT_NEW
    else:
        return _CCX_INTER_NEW


def _try_insert(ints, new_int: CCXIntBox):
    new_pinter: CCXIntBox
    new_pinter = new_int
    
    if len(ints) == 0:
        ints.append(new_pinter)
        return
    
    next_index = _ccx_inter_bisect(ints, new_pinter)
    
    if (next_index == 0):
        res = _is_new_one(new_int, ints[next_index])
        
        if res == _CCX_INTER_NEW:
            ints.insert(next_index, new_int)
        else:
            ints[next_index].expand(new_int)
        
        return
    
    elif (next_index == len(ints)):
        
        res = _is_new_one(new_int, ints[next_index - 1])
        
        if res == _CCX_INTER_NEW:
            ints.insert(next_index, new_int)
        
        else:
            ints[next_index - 1].expand(new_int)
        
        return
    
    res_next = _is_new_one(new_int, ints[next_index])
    res_prev = _is_new_one(new_int, ints[next_index - 1])
    
    if (res_next == _CCX_INTER_NOT_NEW) and (res_prev == _CCX_INTER_NEW):
        nxt = ints.pop(next_index)
        ints[next_index - 1].expand(new_int)
        ints[next_index - 1].expand(nxt)
    
    
    elif (res_next == _CCX_INTER_NOT_NEW):
        ints[next_index].expand(new_int)
    
    elif (res_prev == _CCX_INTER_NOT_NEW):
        ints[next_index - 1].expand(new_int)
    
    
    
    else:
        ints.insert(next_index, new_int)
    
    return True


def nurbs_curve_bvh(curve: NURBSCurveTuple, tol: float = 1e-3) -> tuple[BVH, list[NURBSCurveTuple]]:
    # pts1, param1 = adaptive_polyline(_tuple_to_nurbs(curve), spt=spt, max_depth=100)
    params, *_ = adaptive_curve_sampler(curve, tol)
    curves = split_curve_multiple(curve, params[1:][:-1])
    # curves1=split_curve_multiple(curve,param1[1:][:-1])
    # bbs1 = [AABB.from_points(crv.control_points)for crv in curves1]
    bbs1 = [AABB.from_points(crv.control_points) for crv in curves]
    for bb in bbs1:
        bb.offset_inplace(tol)
    return build_bvh(bbs1), curves


import bisect


def _ccx_inter_bisect(inters: list[CCXIntBox], inter: CCXIntBox):
    return bisect.bisect(inters, inter,
                         )


def _normalize_rects(rects):
    """rects: (N,4) as [s0,t0,s1,t1] (corners in any order).
       returns (N,4) as [s_lo, t_lo, s_hi, t_hi].
    """
    rects = np.asarray(rects, dtype=float).reshape(-1, 4)
    s0, t0, s1, t1 = rects.T
    s_lo = np.minimum(s0, s1)
    s_hi = np.maximum(s0, s1)
    t_lo = np.minimum(t0, t1)
    t_hi = np.maximum(t0, t1)
    return np.stack([s_lo, t_lo, s_hi, t_hi], axis=1)


def _pairwise_overlap(rects, closed=True):
    """Vectorized overlap (N,N) boolean matrix.
       closed=True: touching edges/corners count as overlap.
       closed=False: require strictly positive area intersection.
    """
    s_lo, t_lo, s_hi, t_hi = rects.T
    if closed:
        s_ov = (s_lo[:, None] <= s_hi[None, :]) & (s_lo[None, :] <= s_hi[:, None])
        t_ov = (t_lo[:, None] <= t_hi[None, :]) & (t_lo[None, :] <= t_hi[:, None])
    else:
        s_ov = (s_lo[:, None] < s_hi[None, :]) & (s_lo[None, :] < s_hi[:, None])
        t_ov = (t_lo[:, None] < t_hi[None, :]) & (t_lo[None, :] < t_hi[:, None])
    ov = s_ov & t_ov
    np.fill_diagonal(ov, False)
    return ov


def _connected_components_from_adj(adj):
    """Return component label per node using a simple BFS over a dense boolean adj matrix."""
    n = adj.shape[0]
    labels = -np.ones(n, dtype=int)
    label = 0
    for i in range(n):
        if labels[i] != -1:
            continue
        stack = [i]
        labels[i] = label
        while stack:
            v = stack.pop()
            neigh = np.flatnonzero(adj[v])  # O(n) from the dense row
            unvisited = neigh[labels[neigh] == -1]
            labels[unvisited] = label
            stack.extend(unvisited.tolist())
        label += 1
    return labels


def _merge_by_labels(rects, labels):
    """Merge all rectangles in each connected component to their envelope."""
    s_lo, t_lo, s_hi, t_hi = rects.T
    k = labels.max() + 1
    out = np.empty((k, 4), dtype=float)
    for c in range(k):
        mask = (labels == c)
        out[c, 0] = s_lo[mask].min()
        out[c, 1] = t_lo[mask].min()
        out[c, 2] = s_hi[mask].max()
        out[c, 3] = t_hi[mask].max()
    return out


def _merge_2d_intervals(rects, *, closed=True, max_iter=None):
    """
    Merge axis-aligned 2D intervals into a non-overlapping set.

    Parameters
    ----------
    rects : array_like of shape (N, 4)
        Each row is [s0, t0, s1, t1] (two opposite corners).
    closed : bool, default True
        If True, rectangles that touch at edges/corners are merged.
        If False, only strictly overlapping interiors are merged.
    max_iter : int or None
        Safety cap on iterations (default: N).

    Returns
    -------
    merged : ndarray of shape (M, 4)
        Non-overlapping rectangles as [s_lo, t_lo, s_hi, t_hi].
    """
    rects = _normalize_rects(rects)
    if max_iter is None:
        max_iter = rects.shape[0]
    
    for _ in range(max_iter):
        adj = _pairwise_overlap(rects, closed=closed)
        labels = _connected_components_from_adj(adj)
        # If every rectangle is isolated, we're done
        if (labels.max() + 1) == rects.shape[0]:
            break
        rects = _merge_by_labels(rects, labels)
    return rects


def _nurbs_bvh_ccx(bvh1: BVH, bvh2: BVH, segms1: list[NURBSCurveTuple], segms2: list[NURBSCurveTuple], crv1, crv2,
                   tol=1e-3):
    inters = bvh_intersect(bvh1, bvh2, exact=True)
    ints = []
    
    for op1, op2 in inters:
        a = segms1[op1.object]
        b = segms2[op2.object]
        s0, s1 = _curve_interval(a)
        t0, t1 = _curve_interval(b)
        smid = s0 + (s1 - s0) / 2
        tmid = t0 + (t1 - t0) / 2
        ps = segment_intersection(a.start(), a.end(), b.start(), b.end())
        if ps is None:
            continue
        if isinstance(ps[0], tuple):
            # At this point, we have found all the true overlaps and can be sure that close, parallel segments do not lie on top of each other and do not intersect.
            continue
        
        def _eq(x):
            x = np.asarray(x)
            
            d = evaluate_nurbs_curve(crv1, x[0], 0)["C"] - evaluate_nurbs_curve(crv2, x[1], 0)["C"]
            return np.dot(d, d)
        
        res = bounded_newtons_method(_eq, np.array([smid, tmid]), bounds=((s0, s1), (t0, t1)), max_iter=45)
        
        if res is None or not np.all(np.isfinite(res)):
            continue
        elif not ((s0 <= res[0] <= s1) and (t0 <= res[1] <= t1)):
            continue
        else:
            res = np.asarray(res)
            d2 = _eq(res)
            
            evla = evaluate_nurbs_curve(a, res[0], d_order=2)
            evlb = evaluate_nurbs_curve(b, res[1], d_order=2)
            ds = compute_parametric_curvature_tolerance_curve(evla['C1'], evla['C2'], tol)
            dt = compute_parametric_curvature_tolerance_curve(evlb['C1'], evlb['C2'], tol)
            #print(ds,dt)
            pinter = CCXIntBox(Interval(min(res[0] - ds / 2, s0), max(res[0] + ds / 2, s1)),
                               Interval(min(res[1] - dt / 2, t0), max(res[1] + dt / 2, t1)),
                               CCXInt(res[0], res[1],d2, evla, evlb, ds, dt))
            
            _try_insert(ints, pinter)
        
        #print('\n')
   
        
        #print('\n\n')
    
    return ints


def nurbs_ccx(curve1: NURBSCurve | NURBSCurveTuple, curve2: NURBSCurve | NURBSCurveTuple, tol: float = 1e-3,
              angle_tol=0.0013):
    results = []
    overs = []
    overlaps = []
    vals = []
    _cvs2=decompose_curve(curve2)
    for c1 in decompose_curve(curve1):
        for c2 in _cvs2:
            over, val = _bez_curve_overlap(c1, c2, tol, angle_tol=angle_tol)
            if over:
                overs.append(val)
            elif val is not None:
                vals.append(val)
            results.append(over)
    segms1 = set()
    segms2 = set()
    for (s0, t0, s1, t1) in _merge_2d_intervals(np.array([[o.start.s, o.start.t, o.end.s, o.end.t] for o in overs])):
        segms1.add(s0)
        segms1.add(s1)
        segms2.add(t0)
        segms2.add(t1)
        overlaps.append((Interval(s0, s1), Interval((t0, t1))))
    s0, s1 = curve1.interval()
    t0, t1 = curve2.interval()
    s0s = s0 in segms1
    t0s = t0 in segms2
    segms1.discard(s0)
    segms1.discard(s1)
    segms2.discard(t0)
    segms2.discard(t1)
    curves1 = []
    curves2 = []
    
    for i, v in enumerate(split_curve_multiple(curve1, list(segms1))):
        
        if ((i % 2) == 1 if s0s else (i % 2) == 0):
            bvh, seg = nurbs_curve_bvh(v, tol=tol)
            curves1.append((bvh, seg))
    
    for i, v in enumerate(split_curve_multiple(curve2, list(segms2))):
        if ((i % 2) == 1 if t0s else (i % 2) == 0):
            bvh, seg = nurbs_curve_bvh(v, tol=tol)
            curves2.append((bvh, seg))
    
    intrs = []
    for bvh1, c1 in curves1:
        for bvh2, c2 in curves2:
            for inter in _nurbs_bvh_ccx(bvh1, bvh2, c1, c2, curve1, curve2, tol=tol):
                #print(inter)
                oi = iter(overlaps)
                
                append = True
                while append:
                    try:
                        over = next(oi)
                    except StopIteration:
                        
                        break
                    
                    if (over[0].intersects(inter.s) and over[1].intersects(inter.t)):
                        append = False
                        break
                if append:
                    intrs.append(inter)
    
    return [(float(i.f.s),float(i.f.t)) for i in intrs], overlaps


def multiple_ccx(curves: list[NURBSCurveTuple], spt: float = 1e-3, tol: float = 1e-7, bvh: BVH = None):
    if bvh is None:
        bbs = [AABB.from_points(c.control_points) for c in curves]
        for b in bbs:
            b.offset_inplace(spt)
        bvh = build_bvh(bbs)
    int_candidates = bvh.find_intersecting_leaves2(True)


if __name__ == "__main__":
    from mmcore.geom._nurbs_eval import _nurbs_to_tuple
    import numpy as np
    
    default_pt1 = [
        [22.158641527416805, -41.265945906519704, 0.0],
        [38.290860468167494, -12.153299366618626, 0.0],
        [-4.337633425866585, -26.514161725191443, 0.0],
        [15.0519385376206, 19.088976634204428, 0.0],
        [-32.020429607822194, -4.4209634623771734, 0.0],
        [-14.840038397145449, 35.28715780512934, 0.0],
        [-44.548538735168648, 25.263858719823133, 0.0],
    ]
    
    default_pt2 = [
        [-6.1666326988875966, 37.89197602192263, 0.0],
        [14.215544129727249, 34.295146382508435, 0.0],
        [40.163941122493227, 14.352347571166774, 0.0],
        [43.540592939134157, -9.415717002272153, 0.0],
        [16.580183934742877, -30.210021970020129, 0.0],
        [-10.513217234303696, -21.362760866641814, 0.0],
        [-26.377549521918183, -1.2133457261141416, 0.0],
        [-9.3086771658378353, 19.974390832869219, 0.0],
        [6.3667708626935706, 27.313795735872205, 0.0],
        [22.990902897521764, 11.683487552065344, 0.0],
        [26.711915155435108, 0.6064494223866177, 0.0],
        [19.37450960261674, -11.611372227389872, 0.0],
        [8.2582629104155956, -16.234752290968999, 0.0],
        [-3.0903039985573031, -11.940646639020102, 0.0],
        [-10.739472285742522, -2.2469933680379199, 0.0],
        [2.2509778312197994, 7.9168038191384795, 0.0],
        [14.498391690318186, -0.17203316116128065, 0.0],
    ]
    from mmcore.geom.nurbs import NURBSCurve
    
    c1, c2 = NURBSCurve(np.array(default_pt1)), NURBSCurve(np.array(default_pt2))
    from mmcore.geom._nurbs_eval import _nurbs_to_tuple
    
    nc1, nc2 = _nurbs_to_tuple(c1), _nurbs_to_tuple(c2)
    from mmcore.numeric.intersection.ccx._nccx import nurbs_ccx
    import time
    
    s = time.time()
    res = nurbs_ccx(nc1, nc2)
    pts, prms = zip(*res)
    print(np.array(pts).tolist())

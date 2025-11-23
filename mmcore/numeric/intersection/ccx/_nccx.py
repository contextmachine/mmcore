from __future__ import annotations



import numpy as np
import dataclasses
import functools
import sys

from mmcore.geom.nurbs import NURBSCurve
from numpy import ndarray, dtype, void

from mmcore.geom._nurbs_eval import NURBSCurveTuple, to_homogeneous_1d,evaluate_nurbs_curve,_nurbs_to_tuple,_tuple_to_nurbs
from mmcore.geom._nurbs_knots import _curve_interval, decompose_curve, split_curve_multiple
from mmcore.geom.bvh.lbvh import BVH, build_bvh, AABB, bvh_intersect
from mmcore.numeric import compute_parametric_curvature_tolerance_curve
from mmcore.numeric.approx import adaptive_curve_sampler
from mmcore.numeric.fdm import bounded_newtons_method, newtons_method
from mmcore.numeric.intersection.ccx._utils import _merge_2d_intervals
from mmcore.numeric.intersection.ccx.segment import segment_intersection
from mmcore.numeric.interval import Interval
from ._bez_overlap import _bez_curve_overlap, CCXInt

from numpy.typing import NDArray,DTypeLike

_ccx_isolated_dtype=lambda dim:[('u',np.float64),('v',np.float64),('point', np.float64,(dim,))]
_ccx_overlap_dtype=lambda dim:[('u',np.float64,(2,)),('v',np.float64,(2,)),('point', np.float64,(2,dim))]

_curves_ref_dtype=[('curve1_i',np.uint64),('curve2_i',np.uint64)]
_multiple_ccx_isolated_dtype=lambda dim:_ccx_isolated_dtype(dim)+_curves_ref_dtype
_multiple_ccx_overlap_dtype=lambda dim:_ccx_overlap_dtype(dim)+_curves_ref_dtype

#multiple_ccx_isolated_dtype=np.dtype(_multiple_ccx_isolated_dtype)
#multiple_ccx_overlap_dtype=np.dtype(_multiple_ccx_overlap_dtype)

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
    bbs1 = [AABB.from_points(crv.control_points).offset(tol) for crv in curves]
    for bb in bbs1:
        bb.offset_inplace(tol)
    return build_bvh(bbs1), curves


import bisect


def _ccx_inter_bisect(inters: list[CCXIntBox], inter: CCXIntBox):
    return bisect.bisect(inters, inter,
                         )


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
        p1,p2=a.start(), a.end()
        q1,q2=b.start(), b.end()
        
        ps = segment_intersection(p1, p2, q1, q2, eps=1e-9)
        
        if ps is None:
            
            pass
            
        elif isinstance(ps[0], tuple):
           
            # At this point, we have found all the true overlaps and can be sure that close, parallel segments do not lie on top of each other and do not intersect.
            continue
        
        def _eq(x):
            x = np.asarray(x)
            
            d = evaluate_nurbs_curve(crv1, x[0], 0)["C"] - evaluate_nurbs_curve(crv2, x[1], 0)["C"]
            return np.linalg.norm( d)
        
        res = newtons_method(_eq, np.array([smid, tmid]),  max_iter=500)
        d2 = _eq(res)
        print("R",res,d2)
        if res is None or not np.all(np.isfinite(res)):
            continue
        elif not ((s0 <= res[0] <= s1) and (t0 <= res[1] <= t1)):
            continue
        elif _eq(res)>tol:
            continue
        else:
           
            res = np.asarray(res)
            d2 = _eq(res)
            print('OOO', res,d2)
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
def _is_rational(curve:NURBSCurveTuple):
    return not np.allclose(curve.weights,1)
from mmcore.numeric.intersection.ccx._bez_ccx3 import bezier_intersect_certified_full,map_local_to_global,IsolatedIntersection,OverlapIntersection


def nurbs_ccx(curve1: NURBSCurve | NURBSCurveTuple, curve2: NURBSCurve | NURBSCurveTuple, tol: float = 1e-3,
              **kwargs):
    if isinstance(curve1, NURBSCurve):
        curve1=_nurbs_to_tuple(curve1)
    if isinstance(curve2, NURBSCurve):
        curve2 = _nurbs_to_tuple(curve2)
    curves1 = decompose_curve(curve1)
    curves2=decompose_curve(curve2)


    
    bvh1= build_bvh([AABB.from_points(crv.control_points).offset(tol) for crv in curves1])
    bvh2 = build_bvh([AABB.from_points(crv.control_points).offset(tol) for crv in curves2])
    isolated=[]
    overlaps=[]
    rational=any((_is_rational(curve1), _is_rational(curve2)))

    for a,b in bvh_intersect(bvh1,bvh2,exact=False):
        _c1=curves1[a.object]
        _c2 = curves2[b.object]
        if rational:

            pts1=to_homogeneous_1d(_c1.control_points,_c1.weights)
            pts2 = to_homogeneous_1d(_c2.control_points, _c2.weights)
        else:
            pts1 = _c1.control_points
            pts2 = _c2.control_points
        print(pts1.tolist(),pts2.tolist())
        result=bezier_intersect_certified_full(pts1,  pts2, atol=tol, rational=rational)
        isolated.extend(result['isolated'])
        overlaps.extend(result['overlaps'])
        

   
    
    return isolated, overlaps



def nurbs_ccx_multiple(curves: list[NURBSCurveTuple], tol: float = 1e-3, self_intersections:bool=False,**kwargs)-> tuple[ndarray[tuple[int], dtype[np.float64]], ndarray[
    tuple[int], dtype[np.float64]]] | None:
    counter=0
    segm_map:dict[int,int]=dict()
    def get_segm_curve_index(segment_index):
        return segm_map[segment_index]
    segments=[]
    bbs=[]
    rational=False
    dim=max(crv.control_points.shape[1] for crv in curves)
    for i in range(len(curves)):
            if not rational:
                rational=_is_rational(curves[i])
            for segm in decompose_curve(curves[i]):
                segm_map[counter]=i
                counter+=1

                bb=AABB.from_points(segm.control_points)

                bb.offset_inplace(tol)
                bbs.append(bb)
                segments.append(segm)


    bvh = build_bvh(bbs)
    int_candidates = bvh.build_intersection_leaves_pairs(exact=False)
    isolated_u,isolated_v,isolated_xyz,isolated_crv1,isolated_crv2=[],[],[],[],[]
    overlaps_u, overlaps_v, overlaps_xyz, overlaps_crv1, overlaps_crv2 = [], [], [], [], []

    for first,second in int_candidates:
        segm1_i= bvh.nodes[first].object
        segm2_i= bvh.nodes[second].object
        segm1=segments[segm1_i]
        segm2=segments[segm2_i]
        curve1_i = get_segm_curve_index(segm1_i)
        curve2_i = get_segm_curve_index(segm2_i)
        #print((curve1_i,segm1.interval()),(curve2_i,segm2.interval()))
        print(segm1)
        print('$')
        print(segm2)
        print(';')
        if curve1_i==curve2_i:
            if not self_intersections:

                continue
            interval1=segm1.interval()
            interval2=segm2.interval()
            lo = max(interval1[0], interval2[0])
            hi = min( interval1[1],  interval2[1])

            if (False if lo >= hi else True):
                continue


        if rational:

            pts1 = to_homogeneous_1d(segm1.control_points, segm1.weights)
            pts2 = to_homogeneous_1d(segm2.control_points, segm2.weights)
        else:
            pts1 = segm1.control_points
            pts2 = segm2.control_points

        result = bezier_intersect_certified_full(pts1, pts2, atol=tol, rational=rational)
        if len(result['isolated'])==0 and len(result['overlaps'])==0:
            continue
        else:

            for inter in result['isolated']:
                u,v=inter['u'], inter['v']
                u_glob,v_glob=map_local_to_global(u,v,*segm1.interval(),*segm2.interval())
                isolated_u.append(u_glob)
                isolated_v.append(v_glob)
                isolated_xyz.append(inter['point'])
                isolated_crv1.append(curve1_i)
                isolated_crv2.append(curve2_i)
            overlap:OverlapIntersection
            for overlap in result['overlaps']:
                overlaps_u.append(overlap['uv_path'][(0,-1),0])
                overlaps_v.append(overlap['uv_path'][(0, -1), 1])
                overlaps_xyz.append(overlap['xyz_path'][(0, -1), :])
                overlaps_crv1.append(curve1_i)
                overlaps_crv2.append(curve2_i)
    if len(isolated_u)==0 :
            isolated=None
    else:
            isolated=np.zeros(len(isolated_u),dtype=_multiple_ccx_isolated_dtype(dim))
            isolated['u']=isolated_u
            isolated['v']=isolated_v
            isolated['point']=isolated_xyz
            isolated['curve1_i']=isolated_crv1
            isolated['curve2_i']=isolated_crv2
    if len(overlaps_u)==0:
            overlaps=None
    else:
            overlaps=np.zeros(len(overlaps_u),dtype=_multiple_ccx_overlap_dtype(dim))

            overlaps['u']=overlaps_u
            overlaps['v']=overlaps_v
            overlaps['point']=overlaps_xyz
            overlaps['curve1_i']=overlaps_crv1
            overlaps['curve2_i']=overlaps_crv2
    return isolated, overlaps

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
   
    import time
    
    s = time.time()
    res = nurbs_ccx(nc1, nc2)
    pts, prms = zip(*res)
    print(np.array(pts).tolist())

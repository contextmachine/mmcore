from __future__ import annotations
import numpy as np

from mmcore.geom._nurbs_eval import NURBSCurveTuple,evaluate_nurbs_curve,_tuple_to_nurbs,_nurbs_to_tuple
from mmcore.geom._nurbs_knots import _curve_interval,split_curve,split_curve_multiple


from mmcore.geom.bvh.lbvh import BVH,build_bvh,AABB,BVHNode,bvh_intersect
from mmcore.numeric.newton.cnewton import newtons_method

from mmcore.geom.nurbs import NURBSCurve
from mmcore.numeric.algorithms.adaptive_polyline import adaptive_polyline

def _is_new(ints,new_int, tol=1e-4):
    new_pt,(tn,sn)=new_int
    for pt,(t,s) in ints:
        d=pt-new_pt
        if np.isclose(np.dot(d,d)    ,0) and abs(t-tn)<tol and abs(t-tn)<tol:
            return False
    return True

def nurbs_curve_bvh(curve:NURBSCurveTuple, spt:float=1e-3)->tuple[BVH,list[NURBSCurveTuple]]:
    pts1, param1 = adaptive_polyline(_tuple_to_nurbs(curve), tol=spt, max_depth=100)

    curves1=split_curve_multiple(curve,param1[1:][:-1])
    bbs1 = [AABB.from_points(crv.control_points)for crv in curves1]
    for bb in bbs1:
        bb.offset_inplace(spt)
    return build_bvh(bbs1),curves1

from mmcore.geom._nurbs_knots import decompose_curve

from mmcore.numeric.aabb import aabb,aabb_intersect
def _bez_ccx(
    curve1: NURBSCurve | NURBSCurveTuple, curve2: NURBSCurve | NURBSCurveTuple, spt: float = 1e-3
):

    if isinstance(curve1, NURBSCurve):
        curve1=_nurbs_to_tuple(curve1)
    if isinstance(curve2, NURBSCurve):
        curve2 = _nurbs_to_tuple(curve2)
    ints = []
    stack=[]
    stack.append((curve1,curve2))
    while stack:
        a,b=stack.pop(
        0
        )
        bba,bbb=aabb(a.control_points),aabb(b.control_points)
        if not aabb_intersect(bba,bbb):
            break
        t0, t1 = _curve_interval(a)
        s0, s1 = _curve_interval(b)
        tmid = t0 + (t1 - t0) / 2
        smid = s0 + (s1 - s0) / 2
        
        def _eq(x):
            x=np.array(x)
            print(x)
            d = evaluate_nurbs_curve(a, x[0], 0)["C"] - evaluate_nurbs_curve(b, x[1], 0)["C"]
            return np.dot(d, d)
        print(tmid,smid)
        res = np.array(newtons_method(_eq, np.array([tmid, smid])))

        if res is None:
            continue
        else:

            d2 = _eq(res)

            if np.isclose(d2, 0) or (np.sqrt(d2) < spt):
                pt = evaluate_nurbs_curve(a, res[0], d_order=0)["C"]

                if _is_new(ints, (pt, res)):
                    ints.append((pt, res))

    ints.sort(key=lambda x: x[1][0])
    return ints


def nurbs_ccx(curve1:NURBSCurve|NURBSCurveTuple,curve2:NURBSCurve|NURBSCurveTuple, spt:float=1e-3, bvh1:BVH=None,bvh2:BVH=None):

    if bvh1 is None:
        bvh1,curves1=nurbs_curve_bvh(curve1,spt=spt)
    else:
        curves1=   decompose_curve(curve1)
    if bvh2 is None:
        bvh2,curves2=nurbs_curve_bvh(curve2,spt=spt)
    else:
        curves2 = decompose_curve(curve2)
        
    inters = bvh_intersect(bvh1, bvh2, exact=True)
    ints=[]

    for op1,op2 in inters:
        a=curves1[op1.object ]
        b = curves2[op2.object]
        t0,t1= _curve_interval(a)
        s0, s1 = _curve_interval(b)
        tmid=t0+(t1-t0)/2
        smid = s0 + (s1 - s0) / 2
        def _eq(x):
            d = evaluate_nurbs_curve(a, x[0], 0)["C"] - evaluate_nurbs_curve(b, x[1], 0)["C"]
            return np.dot(d, d)

        res = np.array(newtons_method(_eq, np.array([tmid, smid])))

        if res is None:
            continue
        else:

            d2 = _eq(res)

            if np.isclose(d2,0) or (np.sqrt(d2)<spt):
                pt=evaluate_nurbs_curve(a, res[0], d_order=0)["C"]

                if _is_new(ints,(pt,res)):
                    ints.append((pt,res))

    ints.sort(key=lambda x: x[1][0])
    return ints


def multiple_ccx(curves: list[NURBSCurveTuple], spt: float = 1e-3, tol: float = 1e-7, bvh: BVH = None):

    if bvh is None:
        bbs=[AABB.from_points(c.control_points) for c in curves]
        for b in bbs:
            b.offset_inplace(spt)
        bvh=build_bvh(bbs)
    int_candidates=bvh.find_intersecting_leaves2(True)

if __name__ =="__main__":

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
    s=time.time()
    res = nurbs_ccx(nc1, nc2)

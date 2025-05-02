import numpy as np

from mmcore.geom._nurbs_eval import NURBSCurveTuple,evaluate_nurbs_curve,_tuple_to_nurbs
from mmcore.geom._nurbs_knots import _curve_interval,split_curve

from mmcore.numeric.aabb import aabb,aabb_overlap,aabb_offset,aabb_intersection
from mmcore.geom.bvh.lbvh import BVH,build_bvh,AABB,BVHNode
from mmcore.numeric.newton.cnewton import newtons_method
from mmcore.numeric.algorithms.adaptive_polyline import adaptive_polyline
def nurbs_ccx(curve1:NURBSCurveTuple,curve2:NURBSCurveTuple, spt:float=1e-3,tol:float=1e-7):
    stack=[(curve1,curve2)]
    pts,param1=adaptive_polyline(_tuple_to_nurbs(curve1),tol=spt,max_depth=100)
    pts, param2 = adaptive_polyline(_tuple_to_nurbs(curve1), tol=spt, max_depth=100)

    ints=[]
    while stack:
        c1,c2=stack.pop()
        t_int=_curve_interval(c1)
        s_int=_curve_interval(c2)

        bb1=aabb_offset(aabb(c1.control_points),spt)
        bb2=aabb_offset(aabb(c2.control_points),spt)
        if not aabb_overlap(bb1,bb2):
            continue

        bbint =np.array(aabb_intersection(bb1,bb2))
        t_mid = t_int[0] + (t_int[1] - t_int[0]) / 2

        s_mid = s_int[0] + (s_int[1] - s_int[0]) / 2
        if (t_int[1]-t_int[0])<tol:
            continue
        if np.min(bbint[1]-bbint[0])<=(spt*4):

            def _eq(x):
                d=evaluate_nurbs_curve(c1, x[0], 0)["C"] - evaluate_nurbs_curve(c2, x[1], 0)["C"]
                return np.dot(d,d)

            res=newtons_method(_eq, np.array([t_mid,s_mid]),tol=1e-5,max_iter=5)

            if  res is not None:
                res = np.array(newtons_method(_eq, np.array([t_mid, s_mid])))
                t,s=res
                if t_int[0]<=t<=t_int[1] and s_int[0]<=s<=s_int[1]:

                    sqd=_eq(res)
                    if np.sqrt(sqd)<spt:
                        ints.append(res)

                continue

        if t_int[0] <t_mid < t_int[1] and s_int[0] < s_mid < s_int[1]:

            c11, c12 = split_curve(curve1, t_mid)
            c21, c22 = split_curve(curve2, s_mid)
            stack.append((c11, c21))
            stack.append((c12, c21))
            stack.append((c11, c22))
            stack.append((c12, c22))

    return ints

def multiple_ccx(curves: list[NURBSCurveTuple], spt: float = 1e-3, tol: float = 1e-7, bvh: BVH = None):
    if bvh is None:
        bbs=[AABB.from_points(c.control_points) for c in curves]
        for b in bbs:
            b.offset_inplace(spt)
        bvh=build_bvh(bbs)
    int_candidates=bvh.find_intersecting_leaves2(True)

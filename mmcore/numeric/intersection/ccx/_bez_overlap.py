from __future__ import annotations
import math
from typing import NamedTuple

from mmcore.geom._nurbs_eval import (
    NURBSCurveTuple,
    _curve_interval,
    to_homogeneous_1d, to_homogeneous_2d, evaluate_nurbs_surface
)
import numpy as np

from mmcore.geom.bvh.lbvh import BVHNode, AABB
from mmcore.numeric import evaluate_curvature, evaluate_sectional_curvature
from mmcore.numeric._aabb import aabb
from mmcore.numeric.closest_point import nurbs_curve_closest_point, nurbs_surface_closest_point


class CCXInt(NamedTuple):
    s:float
    t:float
    fun:float
    c1_eval:dict
    c2_eval:dict
    ds:float
    dt:float
    
    def __eq__(self, other: CCXInt):
        return self.fun == other.fun
    
    def __lt__(self, other: CCXInt):
        return self.fun.__lt__(other.fun)
    
    def __gt__(self, other: CCXInt):
        return self.fun.__gt__(other.fun)
    
class Overlap(NamedTuple):
    start:CCXInt
    end:CCXInt
    


def _reverse_param(t:float, interval:tuple[float,float]):
    """
    
    :param t:
    :param interval:
    :return:
    """
    return (interval[0] + interval[1]   )-t


_EPS=np.finfo(np.float64).eps

def _aabb(pts):
    return np.array([np.min(pts, axis=0), np.max(pts, axis=0)])

_min_aeps=math.sin(math.radians(1))

def _bez_curve_overlap(c1:NURBSCurveTuple, c2:NURBSCurveTuple, spt:float=1e-3, angle_tol:float=0.0013704652454261668, **kwargs):
    """Finds the overlap between two Bezier curves.
     """

    s0,s1=_curve_interval(c1)
    t0,t1=_curve_interval(c2)
    start1 = evaluate_nurbs_curve(c1,s0,d_order=2)
    start1['T'],  start1['K'],_=     evaluate_curvature(start1['C1'],start1['C2'])
    end1 =evaluate_nurbs_curve(c1,s1,d_order=2)
    end1['T'], end1['K'],_= evaluate_curvature(end1['C1'],end1['C2'])
    start2 = evaluate_nurbs_curve(c2,t0,d_order=2)
    start2['T'],start2['K'],_=evaluate_curvature(start2['C1'],start2['C2'])
    end2 = evaluate_nurbs_curve(c2,t1,d_order=2)
    end2['T'],end2['K'],_=evaluate_curvature(end2['C1'],end2['C2'])
    bb1,bb2=AABB.from_points(to_homogeneous_1d(c1.control_points,c1.weights)).offset(spt).__array__(), AABB.from_points(to_homogeneous_1d(c2.control_points,c2.weights)).offset(spt).__array__()
    
    c1_ends,c2_ends=[(start1,s0),(end1,s1)],  [(start2,t0),(end2,t1)]

    c1_ends=list(filter(lambda x:point_in_aabb(bb2,x[0]['C']),c1_ends))
    c2_ends=list(filter(lambda x: point_in_aabb(bb1, x[0]['C']), c2_ends))
    if (len(c1_ends)+len(c2_ends))<2:
        #print(1)
        return False, None
    ints=[]
    for pt,prm in c1_ends:
        curve1_eval =pt
        ds=compute_parametric_tolerance_curve(pt['C1'],pt["C2"],spt=spt,angle_tol=angle_tol)
        t,  (fx, curve2_eval, dt)=nurbs_curve_closest_point(c2, pt['C'], spt=spt, angle_tol=angle_tol)
        curve2_eval["T"], curve2_eval["K"], _ = evaluate_curvature(curve2_eval["C1"], curve2_eval["C2"])

        if fx<spt:
            if( 1-np.abs(np.dot(curve2_eval['T'],pt["T"])))<angle_tol:
                #print("K",curve2_eval["K"],pt["K"])
                if compare_curvature(curve1_eval["C1"], curve1_eval["C2"], curve1_eval["K"], curve2_eval["C1"], curve2_eval["C2"], curve2_eval["K"]):
                   
                  
                        ints.append(CCXInt(prm,t, fx,pt,curve2_eval,ds,dt))

    for pt, prm in c2_ends:
        dt = compute_parametric_tolerance_curve(pt['C1'],pt["C2"] , spt=spt, angle_tol=angle_tol)
        curve2_eval=pt
        s,(fx, curve1_eval, ds) = nurbs_curve_closest_point(c1, pt["C"], spt=spt, angle_tol=angle_tol)
        curve1_eval["T"], curve1_eval["K"], _ = evaluate_curvature(curve1_eval["C1"], curve1_eval["C2"])
     
        if fx < spt:
            if (1 - np.abs(np.dot(curve1_eval["T"], pt["T"]))) < angle_tol:
                if compare_curvature(curve1_eval["C1"], curve1_eval["C2"], curve1_eval["K"], curve2_eval["C1"], curve2_eval["C2"], curve2_eval["K"]):
                        
                        
                        ints.append(CCXInt(s, prm, fx, curve1_eval, pt,ds, dt))

    if len(ints)<2:
        #print(2)
        return False, ints
    min_s, max_s = min(ints, key=lambda x: x.s), max(
        ints,
        key=lambda x: x.s,
    )

    ss,se=min_s.s,max_s.s
    ts,te=min_s.t,max_s.t

    s_mid = ss + (se - ss) / 2
    curve1_eval=evaluate_nurbs_curve(c1,s_mid,d_order=2)
    
    t_mid, (fx, curve2_eval, dt) = nurbs_curve_closest_point(c2, curve1_eval["C"], spt=spt, angle_tol=angle_tol)
    if fx < spt:
        
       
        curve1_eval["T"], curve1_eval["K"], _ = evaluate_curvature(curve1_eval["C1"], curve1_eval["C2"])
    
        curve2_eval["T"], curve2_eval["K"], _ = evaluate_curvature(curve2_eval["C1"], curve2_eval["C2"])
    
    
        if (1 - np.abs(np.dot(curve2_eval["T"], curve1_eval["T"]))) < angle_tol:
            if compare_curvature(curve1_eval["C1"], curve1_eval["C2"], curve1_eval["K"], curve2_eval["C1"], curve2_eval["C2"], curve2_eval["K"]):
                
                    pass
            else:
                    #print(fx,3)
                    return False, ints
        else:
            #print(fx,4)
            return False, ints


    return True, Overlap(    min_s,   max_s)

from mmcore.geom._nurbs_eval import (
    NURBSCurveTuple, NURBSSurfaceTuple,
    evaluate_nurbs_curve
)
from mmcore.numeric import (
   
    compute_parametric_tolerance_curve,
    compare_curvature
)
from mmcore.numeric.aabb import point_in_aabb, aabb_overlap


class CSXInt(NamedTuple):
    t: float        # curve parameter
    uv: tuple[float,float]
    c_eval: dict
    s_eval: dict
    dt: float
    duv: tuple[float,float]

class CurveSurfaceOverlap(NamedTuple):
    start: CSXInt
    end:   CSXInt
class CurveBoundaryHit(NamedTuple):
    t: float
    s: float
    uv: tuple[float,float]

    pt: np.ndarray
    boundary_curve:   NURBSCurveTuple



from mmcore.geom.nurbs_iso import extract_surface_boundaries_tuple

class CurveTree:
    __slots__ = ("bvh", "curve",'prims')
    def __init__(self, curve: NURBSCurveTuple,bvh:BVHNode, prims:list[NURBSCurveTuple]):
        self.curve = curve
        self.bvh = bvh
        self.prims=prims
from mmcore.geom._nurbs_eval import _surface_interval
def _curve_boundary_hits(crv: NURBSCurveTuple, srf: NURBSSurfaceTuple, spt: float = 1e-3):
    from mmcore.numeric.intersection.ccx._nccx import nurbs_ccx
    u0_curve, u1_curve, v0_curve, v1_curve=boundaries=extract_surface_boundaries_tuple(srf)
    (u0,u1),(v0,v1)=_surface_interval(srf)

    

    ints=[]
    overs=[]
    for i,bcrv in enumerate(boundaries):
 
        inters,overs=nurbs_ccx(crv, bcrv, spt)
        
        
        for inter in res:
            pt,(t,s)=inter
            if i==0:
                hit=CurveBoundaryHit(t, s,(u0,s), pt,bcrv)
            elif i==1:
                hit = CurveBoundaryHit(t, s, (u1, s), pt,bcrv)
            elif i==2:
                hit = CurveBoundaryHit(t, s, (s, v0), pt, bcrv)
            elif i==3:
                hit = CurveBoundaryHit(t, s, (s, v1), pt, bcrv)
            else:
                raise ValueError(f'{i} boundaries')
            ints.append(hit)

       
        
    ints.sort(key=lambda x: x.t)
    return ints


def _bez_curve_surface_overlap(
        crv: NURBSCurveTuple,
        srf: NURBSSurfaceTuple,
        spt: float = 1e-3,
        angle_tol: float = 0.052):
    """
    Detects whether a rational Bézier/NURBS curve `crv` overlaps a
    rational Bézier/NURBS surface patch `srf` over a *finite* segment.
    Adapted from Hu–Maekawa–Patrikalakis (1997), §3.1.2.
    Returns (flag, data) where:
        flag == True  → confirmed overlap,
        flag == False → no overlap (but `data` may hold tangential pts).
    """
    # 1  Bounding‑box coarse cull
    bb_crv = np.array(aabb(to_homogeneous_1d(crv.control_points,crv.weights)))
    bb_srf =  np.array(aabb(to_homogeneous_2d(srf.control_points, srf.weights).reshape(-1, 4)))
    bb_srf[0,...]-=spt
    bb_srf[1,...]+=spt
    bb_crv[0,...]-=spt
    bb_crv[1,...]+=spt
    if not aabb_overlap(bb_crv, bb_srf):
        return False, []

    # 2  Candidate points: curve ends + intersections with patch boundary
    # cand_params = {0.0, 1.0}

    cand_params=_curve_boundary_hits(crv, srf, spt)

    # 3  Classify each candidate
    hits = []

    for inter in cand_params:
        inter:CurveBoundaryHit
        c_eval = evaluate_nurbs_curve(crv, inter.t, d_order=2)

        s_eval = evaluate_nurbs_surface(srf, *inter.uv, d_order=2)
        c_eval["T"], c_eval["K"], _ = evaluate_curvature(c_eval["C1"], c_eval["C2"])

        # c_eval["NC2"]=c_eval["C2"]/np.linalg.norm(c_eval["C2"])
        n_s = np.cross(s_eval["Su"], s_eval["Sv"])
        if abs(np.dot(c_eval["T"], n_s)) < angle_tol :
            if not np.allclose(c_eval['K'],0):

                NC=np.cross(  c_eval["T"],c_eval["NC2"])
                # NC1,NC2=c_eval["C1"]/,c_eval["C2"]
                NC/=np.linalg.norm(NC)
                success, sectional_curvature_vector = evaluate_sectional_curvature(
                    s_eval["Su"], s_eval["Sv"], s_eval["Suu"], s_eval["Suv"], s_eval["Svv"], NC
                )

                # first‑order check: tangency

                # second‑order curvature check
                if (1-np.dot(sectional_curvature_vector/np.linalg.norm(sectional_curvature_vector),c_eval['K']/np.linalg.norm(c_eval['K'])))<angle_tol:

                    hits.append(((inter.t, inter.uv), (c_eval, s_eval)))
            else:
                hits.append(((inter.t, inter.uv), (c_eval, s_eval)))

    if len(hits) < 2:
        return False, hits

    # 4  Determine parametric interval of overlap
    hits.sort(key=lambda h: h[0][0])       # sort by curve parameter
    (t0, uv0),  eval0 = hits[0]
    (t1, uv1),  eval1 = hits[-1]

    # 5  Mid‑segment sanity test (as in your curve/curve version)
    tm = (t0 + t1) / 2
    mid_c = evaluate_nurbs_curve(crv, tm, d_order=2)
    mid_c["T"], mid_c["K"], _ = evaluate_curvature(mid_c["C1"], mid_c["C2"])

    uv_m, (fx_m, s_eval_m, duv_m) = nurbs_surface_closest_point(
        srf, mid_c["C"], spt=spt, angle_tol=angle_tol)

    mid_c["NC2"] = mid_c["C2"] / np.linalg.norm(mid_c["C2"])
    NC = np.cross(mid_c["T"], mid_c["NC2"])
    # NC1,NC2=c_eval["C1"]/,c_eval["C2"]

    success, sectional_curvature_vector = evaluate_sectional_curvature(
        s_eval_m["Su"], s_eval_m["Sv"], s_eval_m["Suu"], s_eval_m["Suv"], s_eval_m["Svv"], NC
    )
    n_sm = np.cross(s_eval_m["Su"], s_eval_m["Sv"])
    n_sm/=np.linalg.norm(n_sm)
    if fx_m > spt or abs(np.dot(mid_c["T"], n_sm)) > angle_tol or (1-np.dot(sectional_curvature_vector/np.linalg.norm(sectional_curvature_vector),mid_c['K']/np.linalg.norm(mid_c['K'])))>angle_tol:

        return False, hits

    return True, CurveSurfaceOverlap(
        CSXInt(t0, uv0, *eval0, 0., (0.,0.)),
        CSXInt(t1, uv1, *eval1,  0., (0.,0.))
    )

if __name__=="__main__":
    import numpy as np
    from mmcore.geom._nurbs_eval import NURBSCurveTuple

    import numpy as np
    from mmcore.geom._nurbs_eval import NURBSCurveTuple

    crv1 = NURBSCurveTuple(
        order=4,
        knot=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        control_points=np.array(
            [
                [-19.99999996, -15.99999996, 0.0],
                [-14.11168586, -10.11168586, 0.0],
                [-8.22337172, -9.17654935, 0.0],
                [-5.31116635, -12.59936871, 0.0],
            ]
        ),
        weights=np.array([1.0, 1.0, 1.0, 1.0]),
    )

    from mmcore.geom._nurbs_eval import NURBSCurveTuple

    crv2 = NURBSCurveTuple(
    order=4,
    knot=np.array([0., 0., 0., 0., 1., 1., 1., 1.]),
    control_points=np.array([ [ -4.        , -15.        ,   0.        ],
          
           [ -5.56948184, -10.29155448,   0.        ], [-10.21805524,  -9.27801883,   0.        ],[-15.52943104, -12.44265086,   0.        ]]
          ),
    weights=np.array([1., 1., 1., 1.])
    )
    res=_bez_curve_overlap(crv1, crv2) # True
    print(res)
    assert res[0] == True
    crv3 = NURBSCurveTuple(
        order=4,
        knot=np.array([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        control_points=np.array([[-15.31739026, -12.31832453, 0.0], [-8.0, -8.0, 0.0], [-3.0, -13.0, 0.0], [-4.0, -17.0, 0.0]]),
        weights=np.array([1.0, 1.0, 1.0, 1.0]),
    )

    res2 = _bez_curve_overlap(crv2, crv3) # False
    print(res2)
    assert res2[0] == False
    import numpy as np
    from mmcore.geom._nurbs_eval import NURBSCurveTuple

    crv4 =  NURBSCurveTuple(
    order=4,
    knot=np.array([0., 0., 0., 0., 1., 1., 1., 1.]),
    control_points=np.array([[-15.31739026, -12.31832453,   0.        ],
           [-10.22795811,  -9.31481933,   0.        ],
           [ -8.16419053, -10.24855722,   0.        ],
           [ -6.17138081, -11.77864314,   0.        ]]),
    weights=np.array([1., 1., 1., 1.])
    )

    res3 = _bez_curve_overlap(crv2, crv4)  # False
    print(res3)
    assert res3[0] == False
    line1 = NURBSCurveTuple(
        order=2,
        knot=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=np.array([[-26.0, -24.0, 0.0], [-5.0, -23.0, 0.0]]),
        weights=np.array([1.0, 1.0]),
    )
    line2 = NURBSCurveTuple(
        order=2,
        knot=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=np.array([[-26.0, -24.0, 0.0], [-5.0, -23.0, 0.0]]),
        weights=np.array([1.0, 1.0]),
    )

    res4=_bez_curve_overlap(line1, line2) # True
    print(res4)
    assert res4[0] == True
    crv5 = NURBSCurveTuple(
        order=9,
        knot=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
        control_points=np.array(
            [
                [-14.0, -14.0, 0.0],
                [-10.02802769, 2.68228368, 0.0],
                [-7.61515077, 5.76636677, 0.0],
                [-5.7068885, 7.875, 0.0],
                [-1.98515124, 9.00431836, 0.0],
                [1.52232143, 7.5, 0.0],
                [5.44520052, 4.79218832, 0.0],
                [7.80820182, 1.89765913, 0.0],
                [15.0, -14.0, 0.0],
            ]
        ),
        weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    import numpy as np
    from mmcore.geom._nurbs_eval import NURBSCurveTuple

    import numpy as np
    from mmcore.geom._nurbs_eval import NURBSCurveTuple

    crv6 = NURBSCurveTuple(
        order=3,
        knot=np.array([0.0, 0.0, 0.0, 89.27178858, 89.27178858, 89.27178858]),
        control_points=np.array([[-14.0, -14.0, 0.0], [-4.0, 28.0, 0.0], [15.0, -14.0, 0.0]]),
        weights=np.array([1.0, 1.0, 1.0]),
    )
    res5=_bez_curve_overlap(crv5, crv6) # False
    print(res5)
    assert res5[0] == False
    crv7= NURBSCurveTuple(
    order=8,
    knot=np.array([ 0.        ,  0.        ,  0.        ,  0.        ,  0.        ,
            0.        ,  0.        ,  0.        , 89.27178858, 89.27178858,
           89.27178858, 89.27178858, 89.27178858, 89.27178858, 89.27178858,
           89.27178858]),
    control_points=np.array([[-14.        , -14.        ,   0.        ],
           [-11.14285714,  -2.        ,   0.        ],
           [ -7.85714286,   6.        ,   0.        ],
           [ -4.14285714,  10.        ,   0.        ],
           [  0.        ,  10.        ,   0.        ],
           [  4.57142857,   6.        ,   0.        ],
           [  9.57142857,  -2.        ,   0.        ],
           [ 15.        , -14.        ,   0.        ]]),
    weights=np.array([1., 1., 1., 1., 1., 1., 1., 1.])
    )
    res6 = _bez_curve_overlap(crv6, crv7)  # True
    print(res6)
    assert res6[0] == True
    crv8 = NURBSCurveTuple(
        order=8,
        knot=np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
            ]
        ),
        control_points=np.array(
            [
                [-14.0, -14.0, 0.0],
                [-11.14285714, -2.0, 0.0],
                [-8.85714286, 6.0, 0.0],
                [-4.14285714, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                [4.67142857, 6.0, 0.0],
                [9.57142857, -2.0, 0.0],
                [15.0, -14.0, 0.0],
            ]
        ),
        weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    res7 = _bez_curve_overlap(crv6, crv8)  # False
    print(res7)
    assert res7[0]==False

    crv9 = NURBSCurveTuple(
        order=8,
        knot=np.array(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
                89.27178858,
            ]
        ),
        control_points=np.array(
            [
                [-14.0, -14.0, 0.0],
                [-11.14285714, -2.0, 0.0],
                [-7.95714286, 6.0, 0.0],
                [-4.14285714, 10.0, 0.0],
                [0.0, 10.0, 0.0],
                [4.67142857, 6.0, 0.0],
                [9.57142857, -2.0, 0.0],
                [15.0, -14.0, 0.0],
            ]
        ),
        weights=np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]),
    )
    res8 = _bez_curve_overlap(crv6, crv9)  # False
    print(res8)
    assert res8[0]==False

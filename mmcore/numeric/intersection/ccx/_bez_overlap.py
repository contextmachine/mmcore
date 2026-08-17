from __future__ import annotations
import math
from typing import NamedTuple

from mmcore.geom._nurbs_eval import (
    NURBSCurveTuple,
    _curve_interval,
    to_homogeneous_1d, to_homogeneous_2d, evaluate_nurbs_surface,
    evaluate_nurbs_curve,
)
import numpy as np

from mmcore.numeric.bvh.lbvh import BVHNode, AABB
from mmcore.numeric import evaluate_curvature, evaluate_sectional_curvature
from mmcore.numeric._aabb import aabb
from mmcore.numeric.aabb import point_in_aabb
from mmcore.numeric.numeric import compare_curvature, compute_parametric_tolerance_curve
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


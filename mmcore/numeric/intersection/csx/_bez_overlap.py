from __future__ import annotations

from mmcore.geom._nurbs_eval import NURBSCurveTuple, NURBSSurfaceTuple, to_homogeneous_1d, to_homogeneous_2d, \
    evaluate_nurbs_curve, evaluate_nurbs_surface
from mmcore.geom.bvh.lbvh import AABB
from mmcore.geom.nurbs_iso import extract_surface_boundaries_tuple
from mmcore.numeric import evaluate_curvature, evaluate_sectional_curvature
from mmcore.numeric.closest_point import nurbs_surface_closest_point
from mmcore.numeric.interval import Interval
import numpy as np
from mmcore.numeric.intersection.ccx._nccx import nurbs_ccx, nurbs_curve_bvh


def _bez_curve_surface_overlap(
        crv: NURBSCurveTuple,
        srf: NURBSSurfaceTuple,
        spt: float = 1e-3,
        angle_tol: float = 0.0013704652454261668, ):
    """
    Detects whether a rational Bézier/NURBS curve `crv` overlaps a
    rational Bézier/NURBS surface patch `srf` over a *finite* segment.
    Adapted from Hu–Maekawa–Patrikalakis (1997), §3.1.2.
    Returns (flag, data) where:
        flag == True  → confirmed overlap,
        flag == False → no overlap (but `data` may hold tangential pts).
    """
    # 1  Bounding‑box coarse cull
    bb_crv = AABB.from_points(to_homogeneous_1d(crv.control_points,crv.weights)).offset(spt)
    bb_srf = AABB.from_points(to_homogeneous_2d(srf.control_points,srf.weights).reshape((-1,4))).offset(spt)
    
    if not bb_crv.intersects(bb_srf):
        
        return False, []
    u0_curve, u1_curve, v0_curve, v1_curve=boundaries=extract_surface_boundaries_tuple(srf)
    ints=[]
    overlaps=[]
    s0,s1=crv.interval()
    (u0,u1),(v0,v1)=srf.interval()
    intervs=[(Interval(u0, u0), Interval(v0, v1)),
     (Interval(u1,u1), Interval(v0, v1)),
    (Interval(u0, u1), Interval(v0, v0)),
    (Interval(u0, u1), Interval(v1, v1))]
    for (int_u,int_v),b in zip(intervs, boundaries):
        
        inters, overs = nurbs_ccx(crv, b, tol=spt, angle_tol=angle_tol)
        #print(inters, overs)
        for inter in inters:
            if int_u.width() == 0:
                
                ints.append((inter[0], int_u.low, inter[1]))
            elif int_v.width() == 0:
                ints.append((inter[0], inter[1], int_v.low))
            else:
                raise ValueError('surface boundary interval {},{} is zero'.format(int_u, int_v))
        
                
        for over in overs:
            
            int_curve, int_b = over
            if int_u.width() > 0:
                
                overlaps.append((int_curve, int_u.intersect(int_b), int_v))
            
            
            elif int_v.width() > 0:
                overlaps.append((int_curve, int_u, int_v.intersect(int_b)))
            
            else:
                raise ValueError('surface boundary interval {},{} is zero'.format(int_u, int_v))
    if len(overlaps) >0:
        
        if len(overlaps)>1:
            #print(overlaps)
            raise ValueError('overlaps>1')
        #nd_boxes=np.zeros((len(overlaps),2,3),dtype=float)
        #for i,(o_c,o_u,o_v) in enumerate(overlaps):
        #    nd_boxes[i,...]=np.array([[o_c.low,o_u.low,o_v.low],[ o_c.upp,o_u.upp,o_v.upp]])
       
        
        
        #merged_nd_boxes = merge_intervals_nd_blocked(nd_boxes, closed=True)
        
       
        
        #return True,[tuple(Interval(merged_nd_boxes[i, 0, j], merged_nd_boxes[i, 1, j]) for j in range(merged_nd_boxes.shape[2])) for i   in range(merged_nd_boxes.shape[0])]
        return True,overlaps[0]
    
    '''
    segms1 = set()
    segms2 = set()
    segms3 = set()
    for start, end in merged_nd_boxes:
        start_s, start_u, start_v = start
        end_s, end_u, end_v = start
        segms1.add(start_s)
        segms1.add(end_s)
        
        segms2.add(start_u)
        segms2.add(end_u)
        segms3.add(start_v)
        segms3.add(end_v)
    s0s = s0 in segms1
    u0s = u0 in segms2
    v0s = v0 in segms3
    
    segms1.discard(s0)
    segms1.discard(s1)
    segms2.discard(u0)
    segms2.discard(u1)
    segms3.discard(v0)
    segms3.discard(v1)
    curves=[]
    surfaces=[]
    for i, v in enumerate(split_curve_multiple(crv, list(segms1))):
        
        if ((i % 2) == 1 if s0s else (i % 2) == 0):
            bvh, seg = nurbs_curve_bvh(v, tol=spt)
            curves.append((bvh, seg))
    
    for i,sspl in enumerate(split_surface_u_multiple(srf,list(segms2))):
        if ((i % 2) == 1 if u0s else (i % 2) == 0):
            for j,svpl in enumerate(split_surface_v_multiple(sspl,list(segms3))):
                if ((j % 2) == 1 if v0s else (i % 2) == 0):
                        
                        surfaces.append(svpl)
    for i, curve in enumerate(curves):
        for j,surface in enumerate(surfaces):
            bb_curve = aabb(to_homogeneous_1d(crv.control_points, crv.weights))
            bb_surf = aabb(to_homogeneous_2d(crv.control_points, crv.weights).reshape((-1, 4)))
            if not aabb_overlap(bb_curve, bb_surf):
               continue
               
    '''
    # 2  Candidate points: curve ends + intersections with patch boundary
    # cand_params = {0.0, 1.0}

    #cand_params=_curve_boundary_hits(crv, srf, spt)

    # 3  Classify each candidate
    hits = []
    #print(ints)
    for inter in ints:
        inter_s,inter_u,inter_v = inter
        c_eval = evaluate_nurbs_curve(crv, inter_s, d_order=2)

        s_eval = evaluate_nurbs_surface(srf, inter_u,inter_v, d_order=2)
        c_eval["T"], c_eval["K"], _ = evaluate_curvature(c_eval["C1"], c_eval["C2"])

        # c_eval["NC2"]=c_eval["C2"]/np.linalg.norm(c_eval["C2"])
        n_s = np.cross(s_eval["Su"], s_eval["Sv"])
        if abs(np.dot(c_eval["T"], n_s)) < angle_tol :
            if not np.allclose(c_eval['K'],0):

                NC=np.cross(  c_eval["T"],c_eval["NC2"])
                # NC1,NC2=c_eval["C1"]/,c_eval["C2"]

                success, sectional_curvature_vector = evaluate_sectional_curvature(
                    s_eval["Su"], s_eval["Sv"], s_eval["Suu"], s_eval["Suv"], s_eval["Svv"], NC
                )

                # first‑order check: tangency

                # second‑order curvature check
                if (1-np.dot(sectional_curvature_vector/np.linalg.norm(sectional_curvature_vector),c_eval['K']/np.linalg.norm(c_eval['K'])))<angle_tol:

                    hits.append(((inter_s, (inter_u,inter_v)), (c_eval, s_eval)))
            else:
                hits.append(((inter_s,  (inter_u,inter_v)), (c_eval, s_eval)))

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

    return True, (Interval(t0,t1), Interval(uv0[0],uv1[0]), Interval(uv0[1],uv1[1]))

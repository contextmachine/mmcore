import dataclasses
import heapq
from collections import deque
from typing import NamedTuple

import numpy as np

from mmcore.geom.nurbs import  NURBSCurve, NURBSSurface, CurveSurfaceEq
from numpy.typing import NDArray

from mmcore.geom._nurbs_eval import (
    NURBSCurveTuple,
    NURBSSurfaceTuple,
    evaluate_nurbs_curve,
    evaluate_nurbs_surface,
    _nurbs_to_tuple,
    _curve_interval,
    _surface_interval,
    _tuple_to_nurbs,
)
__all__=["nurbs_csx_v2"]

from mmcore.numeric.closest_point import nurbs_surface_closest_point
import logging
logger = logging.getLogger("mmcore")


import numpy as np


def curve_surface_angle_gap(T_c, N_s):
    """
    Given:
      - T_c: 3‐vector, the (unnormalized) curve tangent C'(t)
      - N_s: 3‐vector, the (unnormalized) surface normal (i.e. S_u × S_v)

    Returns the absolute cosine‐gap:
      |cos(angle(T_c, N_s))| = |(n_s · T_c_unit)|,
    where n_s = N_s / ||N_s|| and T_c_unit = T_c / ||T_c||.
    (Smaller → closer to exact orthogonality.)
    """
    norm_n_s = np.linalg.norm(N_s)
    if norm_n_s < 1e-14:
        raise ValueError("Surface normal is degenerate; cannot normalize.")
    n_s = N_s / norm_n_s

    norm_Tc = np.linalg.norm(T_c)
    if norm_Tc < 1e-14:
        raise ValueError("Curve tangent is near zero; cannot normalize.")
    T_c_unit = T_c / norm_Tc

    cos_val = abs(np.dot(n_s, T_c_unit))
    return cos_val


def curve_surface_normal_distance_gap(N_s, C_pt, S_pt):
    """
    |N_s · (C_pt - S_pt)| / ||N_s||, but if N_s is already unit‐length,
    this is just |N_s · (C_pt - S_pt)|.
    Here we pass N_s un‐normalized, so we explicitly normalize inside:
    """
    norm_n_s = np.linalg.norm(N_s)
    if norm_n_s < 1e-14:
        raise ValueError("Surface normal is degenerate; cannot normalize.")
    n_s = N_s / norm_n_s
    return abs(np.dot(n_s, (C_pt - S_pt)))


def within_curve_surface_gaps(T_c, N_s, C_pt, S_pt, eps_theta=None, eps_n=None):
    """
    Return True if either
      1) the tangent vs. normal‐angle (cos‐gap) ≤ eps_theta, or
      2) the point‐to‐surface‐plane distance ≤ eps_n.
    """
    # 1) angle gap:
    #print("within_curve_surface_gaps: eps_theta,eps_n: ",T_c.tolist(),#N_s.tolist(),eps_theta,eps_n)
    ca=curve_surface_angle_gap(T_c, N_s)
    if ca <= eps_theta:
        return True

    cdg = curve_surface_normal_distance_gap(N_s, C_pt, S_pt)
    # 2) distance‐along‐n gap:

    if cdg <= eps_n:
        return True
    return False


def _calculate_eps_n(spt_val, ang_tol_val):
    """
    Same heuristic as before: eps_n = (spt²)/(ang_tol + ε_small).
    """
    return spt_val**2 / (ang_tol_val + 1e-12)


def refine_curve_surface(
    t_uv0: np.ndarray,           # initial guess [t, u, v]
    curve:  NURBSCurveTuple,     # NURBSCurve data
    surface: NURBSSurfaceTuple,  # NURBSSurface data
    spt: float = 1e-3,           # geometric tolerance on ||C – S||
    angle_tol: float = 0.052,    # tolerance on |cos(angle)| between T_c and n_s
    eps_n: float = None,         # optional “distance‐along‐n” tolerance
    max_iter: int = 10
) -> tuple[bool, np.ndarray, dict, dict, float]:
    """
    Refine (t, u, v) so that C(t) ≈ S(u, v) and the curve tangent T_c is nearly
    orthogonal to the surface normal n_s. Returns:
      ( success, [t,u,v]_refined, curve_eval, surf_eval, final_error ).
    """
    t0, t1 = _curve_interval(curve)
    (u0, u1), (v0, v1) = _surface_interval(surface)

    # If eps_n not provided, use the heuristic:
    if eps_n is None:
        eps_n = _calculate_eps_n(spt, angle_tol)

    # Initialize
    it = 0
    t_cur, u_cur, v_cur = t_uv0
    curve_eval = {}
    surf_eval = {}
    success = False
    error = np.inf

    while it < max_iter:
        # 1) Evaluate curve (point + tangent)
        #    (Your evaluate_nurbs_curve must return a dict with keys "C" and "C1" or "C_prime")
        curve_eval = evaluate_nurbs_curve(curve, t_cur, d_order=1)
        C_pt = np.array(curve_eval["C"])       # 3‐vector
        T_c  = np.array(curve_eval["C1"])      # 3‐vector (tangent)

        # 2) Evaluate surface (point + partials)
        surf_eval = evaluate_nurbs_surface(surface, u_cur, v_cur, d_order=1)
        S_pt = np.array(surf_eval["S"])        # 3‐vector
        Su   = np.array(surf_eval["Su"])       # 3‐vector
        Sv   = np.array(surf_eval["Sv"])       # 3‐vector

        # 3) Compute the *current* geometric error:
        R = S_pt - C_pt
        error = np.linalg.norm(R)

        # 4) Compute un‐normalized surface normal:
        N_s = np.cross(Su, Sv)
        # (We will normalize inside the gap routines.)

        # 5) Check convergence *after* updating `error`:
        wh=within_curve_surface_gaps(T_c, N_s, C_pt, S_pt, angle_tol, eps_n)
        
        if (error <= spt) and wh:
            success = True
            break

        # 6) Build the average point in ℝ³:
        P_avg = 0.5 * (C_pt + S_pt)

        # 7) Solve for Δt from J_c Δt = P_avg - C_pt
        #    J_c is just the 3×1 column [T_c], so
        numer   = np.dot(T_c, (P_avg - C_pt))   # scalar
        denom   = np.dot(T_c, T_c)              # ||T_c||²
        if abs(denom) < 1e-14:
            # Tangent is nearly zero ⇒ cannot invert.
            # Bail out or slightly perturb t_cur (your choice).
            raise RuntimeError("Curve tangent is degenerate; cannot refine further.")
        delta_t = numer / denom

        # 8) Solve for [Δu, Δv] from J_s [Δu, Δv] = P_avg - S_pt
        J_s = np.column_stack((Su, Sv))        # 3×2
        # Use pseudoinverse to solve the 3×2 least‐squares problem:
        delta_uv = np.linalg.pinv(J_s) @ (P_avg - S_pt)
        delta_u, delta_v = float(delta_uv[0]), float(delta_uv[1])

        # 9) Update parameters:
        t_cur = t_cur + delta_t
        u_cur = u_cur + delta_u
        v_cur = v_cur + delta_v

        if t0<=t_cur<=t1 and u0<=u_cur<=u1 and v0<=v_cur<=v1:

            it += 1
            continue
        else:
            success=False
            break

        # 10) Increment iteration counter:

    return success, np.array([t_cur, u_cur, v_cur]), curve_eval, surf_eval, error
from mmcore.numeric import compute_parametric_tolerance_curve

from enum import Enum,auto
class CSType(int,Enum):
    NO_INT=auto()
    ISOLATED=auto()
    OVERLAP=auto()


from mmcore.numeric.implicitize import nurbs_surf_to_mono,nurbs_curve_to_mono,curve_patch_intersection
from mmcore.geom._nurbs_knots import decompose_surface,decompose_curve
from mmcore.geom.bvh.lbvh import build_bvh,AABB,bvh_intersect,BVH

class BezCurveBothRepr(NamedTuple):
    nurbs:NURBSCurveTuple
    monomial:NDArray

class BezSurfBothRepr(NamedTuple):
    nurbs:NURBSSurfaceTuple
    monomial:NDArray


from mmcore.geom._nurbs_eval import _curve_interval,_surface_interval,EvaluateCurveData,EvaluateSurfaceData


@dataclasses.dataclass(slots=True)
class CSInt:
    tuv:NDArray[float]
    curve_eval:EvaluateCurveData
    surf_eval:EvaluateSurfaceData
    tuv_tol:NDArray[float]
    error:float
    def compare_with_tol(self, tuv:NDArray[float]):
        return np.all(np.abs(tuv-self.tuv)<self.tuv_tol)

def _int_cs_bez( initial_curve:BezCurveBothRepr, initial_surface:BezSurfBothRepr, original_curve:NURBSCurveTuple, original_surface:NURBSSurfaceTuple, inters:list[CSInt],spt=1e-3,angle_tol=0.052,eps_n=None, **kwargs):

    intersections=curve_patch_intersection( initial_surface.monomial[..., 0], initial_surface.monomial[..., 1], initial_surface.monomial[..., 2], initial_surface.monomial[..., 3], initial_curve.monomial[..., 0], initial_curve.monomial[..., 1], initial_curve.monomial[..., 2], initial_curve.monomial[..., 3])
    logger.debug("intersections: t={}, points={}".format([_t.item() for _t,_ in intersections],[np.array(pt).tolist() for _,pt in intersections]))
    # Get original curve domain for proper parameter mapping
    (orig_t0, orig_t1) = _curve_interval(original_curve)
    
    orig_dt = orig_t1 - orig_t0

    print('ORIG',orig_t0,orig_t1)
    for t,pt in intersections:
        # Map from monomial [0,1] space directly to original curve parameter space
        t_real = orig_dt * t + orig_t0
        print(t,t_real)

        pt=evaluate_nurbs_curve(original_curve, t_real, d_order=0)['C']

        best_uv, (error, surf_eval, (du, dv)) = nurbs_surface_closest_point(original_surface, pt, spt=spt, angle_tol=angle_tol)
       
        if best_uv is None:

            continue
        initial_guess = np.array([t_real, *best_uv])
        print('initial_guess:',initial_guess, surf_eval['S'].tolist())
        success, tuv, curve_eval, surf_eval, error = refine_curve_surface(
            initial_guess, original_curve, original_surface, spt=spt, angle_tol=angle_tol, eps_n=eps_n, max_iter=500
        )

        tuv = np.array(tuv)

        print("after_refinement:",tuv, surf_eval['S'].tolist(), curve_eval['C'].tolist(), error)
        ##np.array([t_real, *best_uv]),initial_curve.nurbs,#initial_surface.nurbs,spt=spt,angle_tol=angle_tol,eps_n=eps_n,#max_iter=50)

        if success:

            is_visited = False
            for index in range(len(inters)):

                if inters[index].compare_with_tol(tuv):
                    is_visited = True
                    if inters[index].error>error:
                        dt = compute_parametric_tolerance_curve(curve_eval["C1"], curve_eval["C2"], spt=spt, angle_tol=angle_tol)
                        inters[index]=CSInt(np.array(tuv), curve_eval, surf_eval, np.array([dt, du, dv]), error)
                        logger.debug("Replace: {}".format(inters[index]))

                    else:
                        logger.debug("Pass: {}".format(inters[index]))

            if not is_visited:
                curve_eval = evaluate_nurbs_curve(original_curve, tuv[0], d_order=2)
                dt = compute_parametric_tolerance_curve(curve_eval["C1"], curve_eval["C2"], spt=spt, angle_tol=angle_tol)
                inters.append(CSInt(np.array(tuv), curve_eval, surf_eval, np.array([dt, du, dv]), error))
        else:
            logger.debug("Fail: tuv={}, err={}, pt {}".format(tuv,error, surf_eval['S'].tolist()))

def nurbs_csx_v2( initial_curve, initial_surface, spt=1e-3, angle_tol=0.052,eps_n=None,curve_bvh:BVH=None,surface_bvh:BVH=None,**kwargs):
    if eps_n is None:
        eps_n=_calculate_eps_n(spt,angle_tol)
    if isinstance(initial_surface, NURBSSurfaceTuple):
        init_s=initial_surface

    else:
        init_s=_nurbs_to_tuple(initial_surface)
    if isinstance(initial_curve, NURBSCurveTuple):
        init_c = initial_curve
    else:
        init_c=_nurbs_to_tuple(initial_curve)

    curves=decompose_curve(init_c)
    patches=decompose_surface(init_s)
    crvs=[BezCurveBothRepr(curve,nurbs_curve_to_mono(curve)) for curve in curves]
    srfs=[BezSurfBothRepr(patch,nurbs_surf_to_mono(patch)) for patch in patches]

    if curve_bvh is None:
        curve_bvh = build_bvh([AABB.from_points(crv.control_points).offset(spt) for crv in curves])
    if surface_bvh is None:
        surface_bvh = build_bvh([AABB.from_points(patch.control_points.reshape((-1,patch.control_points.shape[-1]))).offset(spt/2) for patch in patches])

    int_nodes=bvh_intersect(curve_bvh,surface_bvh,exact=True)
    logger.debug("INT NODES: {}".format(len(int_nodes)))
    

    inters=[]
 
    for a,b in int_nodes:
        _int_cs_bez(crvs[a.object],srfs[b.object], original_curve=init_c, original_surface=init_s, inters=inters, spt=spt, angle_tol=angle_tol,eps_n=eps_n)

    return inters

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(module)s.%(funcName)s   %(message)s")
    cpts = np.array(
        [[-9.1796875, 13.229166666666666, -4.5186767578125], [-9.1796875, 14.739583333333332, -4.49395751953125],
         [-9.1796875, 16.432291666666664, -4.580108642578125], [-9.1796875, 18.372395833333332, -4.8531036376953125]]
    )
    spts = np.array([[[-5.849180481790346, 18.372395833333336, -1.5018374203712104],
                      [-5.858792686592141, 16.432291666666668, -2.719633841323509],
                      [-5.871782152540512, 14.739583333333334, -3.1229032219131403],
                      [-5.8852911971268185, 13.229166666666668, -3.116512598566417]],
                     [[-6.88688536134276, 18.372395833333336, -1.6832837287863824],
                      [-6.894094514944105, 16.432291666666668, -2.9325012796112815],
                      [-6.9038366144053835, 14.739583333333334, -3.3409109281989706],
                      [-6.913968397845114, 13.229166666666668, -3.3217200441495054]],
                     [[-7.97766402100707, 18.372395833333336, -2.2658223298287345],
                      [-7.983070886208079, 16.432291666666668, -3.617300740689732],
                      [-7.990377460804037, 14.739583333333334, -4.0455099012702895],
                      [-7.997976298383835, 13.229166666666668, -3.992029157974829]],
                     [[-9.1863730157553, 18.372395833333336, -2.7409113689805125],
                      [-9.190428164656058, 16.432291666666668, -4.174381706229336],
                      [-9.195908095603027, 14.739583333333334, -4.61538352236864],
                      [-9.201607223787876, 13.229166666666668, -4.527011611303911]]]
                    )

    u, v, t = 0.9939461136471586, 0.995759608283125, 0.004240391716877873
    surf = NURBSSurface(np.array(spts), (3, 3))

    curve = NURBSCurve(cpts)
    # ress = new_intersection_candidates(surf, curve, u, v, t, np.array(surf.evaluate_v2(u, v)))

    from mmcore.numeric.intersection.csx._ncsx import nurbs_csx
    import time

    s=time.time()
    r1=nurbs_csx_v2(_nurbs_to_tuple(curve), _nurbs_to_tuple(surf))
    nurbs_csx_v2_time=time.time()-s
    print(nurbs_csx_v2_time, [(item.tuv.tolist(), item.curve_eval["C"].tolist()) for item in r1])
    s=time.time()
    r2=nurbs_csx(curve, surf)
    nurbs_csx_time=time.time()-s
    print(nurbs_csx_time, r2)
    def check_correct( nurbs_csx_result,nurbs_csx_v2_result):
        t1,u1,v1=nurbs_csx_result[0][2]
        print("nurbs_csx: ", t1,u1,v1)
        nurbs_csx_v2_result_sorted=sorted(nurbs_csx_v2_result, key=lambda x: x.tuv[0])
        t2,u2,v2=nurbs_csx_v2_result_sorted[0].tuv
        print("nurbs_csx_v2: ", t2, u2, v2)
        curve_tuple, surf_tuple = _nurbs_to_tuple(curve), _nurbs_to_tuple(surf)

        v1_curve_pt=evaluate_nurbs_curve(curve_tuple, t1,d_order=0)['C']
        v2_curve_pt = evaluate_nurbs_curve(curve_tuple, t2, d_order=0)["C"]
        v1_surf_pt=evaluate_nurbs_surface(surf_tuple, u1, v1, d_order=0)["S"]
        v2_surf_pt=evaluate_nurbs_surface(surf_tuple, u2, v2, d_order=0)["S"]
        v1_dist=np.linalg.norm(v1_curve_pt-v1_surf_pt)
        v2_dist = np.linalg.norm(v2_curve_pt - v2_surf_pt)
        print("nurbs_csx error: ", v1_dist)
        print("nurbs_csx_v2 error: ", v2_dist)
        print("nurbs_csx error: ", v1_dist)
        print("nurbs_csx_v2 better or same: ",np.isclose(v1_dist,v2_dist) or v1_dist>=v2_dist)
    check_correct(r2,r1)
    if nurbs_csx_time>nurbs_csx_v2_time:
        print(f"nurbs_csx_v2 is {nurbs_csx_time/nurbs_csx_v2_time} x faster")
    else:
        print(f"nurbs_csx is {nurbs_csx_v2_time/nurbs_csx_time} faster")
    print()
    import numpy as np
    from mmcore.geom._nurbs_eval import NURBSSurfaceTuple


    TEST_OVERLAP_CASE=True
    if TEST_OVERLAP_CASE:
        val = NURBSCurveTuple(
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

        val2 = NURBSSurfaceTuple(
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

        s = time.time()
        print("start3.int_cs_v2")
        r3 = nurbs_csx_v2(val, val2)

        print(time.time() - s, [item.curve_eval["C"].tolist() for item in r3])

        # s = time.time()
        # print("start3.3")
        # r5=nurbs_csx(_tuple_to_nurbs(val), _tuple_to_nurbs(val2)) # Running this code will cause a long or infinity freeze (the previous algorithm cannot handle overlaps). Only run if you can interrupt the terminal.

        # print(time.time() - s, [item[1] for item in r5])

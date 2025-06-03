import heapq
from collections import deque

import numpy as np

from mmcore.geom.nurbs import  NURBSCurve, NURBSSurface, CurveSurfaceEq

from mmcore.geom._nurbs_eval import NURBSCurveTuple, NURBSSurfaceTuple, evaluate_nurbs_curve, evaluate_nurbs_surface, \
    _nurbs_to_tuple,_curve_interval,_surface_interval
from mmcore.numeric.aabb import aabb_intersect_fast_3d, aabb_intersection, aabb, aabb_offset
from mmcore.numeric.newton.cnewton import newtons_method

from mmcore.geom._nurbs_knots import subdivide_surface, split_curve
from mmcore.numeric.algorithms.adaptive_polyline import chord_length
from mmcore.numeric.intersection.csx._ncsx_new_intersections_test import new_intersection_candidates

import numpy as np

from mmcore.numeric.interval import Interval,Comparison


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


def within_curve_surface_gaps(T_c, N_s, C_pt, S_pt, eps_theta, eps_n):
    """
    Return True if either
      1) the tangent vs. normal‐angle (cos‐gap) ≤ eps_theta, or
      2) the point‐to‐surface‐plane distance ≤ eps_n.
    """
    # 1) angle gap:
    print("within_curve_surface_gaps: eps_theta,eps_n: ",T_c.tolist(),N_s.tolist(),eps_theta,eps_n)
    ca=curve_surface_angle_gap(T_c, N_s)
    print("within_curve_surface_gaps: ca: ",T_c.tolist(),N_s.tolist(),ca)
    if ca <= eps_theta:

        return True
    cdg = curve_surface_normal_distance_gap(N_s, C_pt, S_pt)
    print("within_curve_surface_gaps: cdg: ",T_c.tolist(),N_s.tolist(),cdg)
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

    print(success,[t_cur, u_cur, v_cur])
    return success, np.array([t_cur, u_cur, v_cur]), curve_eval, surf_eval, error
from mmcore.numeric import evaluate_curvature, evaluate_sectional_curvature
from enum import Enum,auto
class CSType(int,Enum):
    NO_INT=auto()
    ISOLATED=auto()
    OVERLAP=auto()

def int_cs_simple(crv,srf,spt=1e-3, angle_tol=0.052, eps_n=None, crv_interv=None,srf_interv_u=None,srf_interv_v=None,max_iter=100,**kwargs)->tuple[bool,np.ndarray,dict,dict,float,CSType]:
    if eps_n is None:
        eps_n=_calculate_eps_n(spt, angle_tol)
    if crv_interv is None:
        t0, t1 =_curve_interval(crv)
    else:
        t0, t1 = crv_interv
    if srf_interv_u is None or srf_interv_v is None:
        (u0, u1), (v0, v1) = _surface_interval(srf)
    else:
        (u0, u1), (v0, v1) = srf_interv_u,srf_interv_v
    t_mid = (t1 - t0) * 0.5 + t0

    u_mid = (u1 - u0) * 0.5 + u0
    v_mid = (v1 - v0) * 0.5 + v0
    success, tuv, curve_eval, surf_eval, error = refine_curve_surface(
        np.array([t_mid, u_mid, v_mid]), crv, srf, spt=spt, angle_tol=angle_tol, eps_n=eps_n, max_iter=max_iter
    )
    if not success:
        if (error <= spt):
            tp=CSType.OVERLAP
        else:
            tp=CSType.NO_INT
    else:
        tp = CSType.ISOLATED
        

    return success, tuv, curve_eval, surf_eval, error,tp


def int_cs( initial_curve, initial_surface,spt=1e-3, angle_tol=0.052,debug=False,**kwargs):
    if isinstance(initial_surface, NURBSSurfaceTuple):
        init_s=initial_surface

    else:
        init_s=_nurbs_to_tuple(initial_surface)
    if isinstance(initial_curve, NURBSCurveTuple):
        init_c = initial_curve
    else:
        init_c=_nurbs_to_tuple(initial_curve)
    stack = [(init_s, init_c, None)]
    results=[]
    eps_n=_calculate_eps_n(spt, angle_tol)
    while stack:

        _surface, _curve, _tuv = stack.pop()

        sbb,cbb=np.asarray(  aabb(_surface.control_points.reshape((-1,3)))),np.asarray(aabb( _curve.control_points))

        if not aabb_intersect_fast_3d(sbb, cbb):
            continue
        t0, t1 = _curve_interval(_curve)
        (u0, u1), (v0, v1) = _surface_interval(_surface)

        t_mid = (t1 - t0) * 0.5 + t0
        u_mid = (u1 - u0) * 0.5 + u0
        v_mid = (v1 - v0) * 0.5 + v0

        if np.all((cbb[1]-cbb[0])<spt) and np.all((np.asarray(sbb[1])-np.asarray(sbb[0]))<spt):

            success, tuv, curve_eval, surf_eval, error,int_type = int_cs_simple(_curve,_surface, crv_interv=( t0, t1 ), srf_interv_u=(u0, u1),srf_interv_v=(v0, v1),spt=spt, angle_tol=angle_tol, eps_n=eps_n,max_iter=10)
            print("So",success, tuv, curve_eval, surf_eval, error,int_type )

            if success:

                results.append(
                    ("transversal", curve_eval['C'], (t_mid, u_mid, v_mid))
                )

            continue

        success,tuv, curve_eval, surf_eval, error =refine_curve_surface(np.array([t_mid,u_mid,v_mid]),_curve,_surface,spt=spt,angle_tol=angle_tol,eps_n=eps_n, max_iter=10)
        print("Sp",  success,tuv, curve_eval, surf_eval, error )

        # surf_curve_eq=CurveSurfaceEq(_curve,_surface)
        # surf_curve_eq=CurveSurfaceEq(_curve,_surface)
        t, u, v = tuv
        if not success:

            if not (t0 <= t <= t1 and u0 <= u <= u1 and v0 <= v <= v1):
                for s in subdivide_surface(_surface, u_mid, v_mid):
                    for c in split_curve(_curve, t_mid):
                        stack.append((s, c, _tuv))
                continue

            # print('n', t_mid,u_mid,v_mid)
            for s in subdivide_surface(_surface,tuv[1] , tuv[2]):
                for c in split_curve(_curve, tuv[0]):
                    stack.append((s, c, _tuv))
            continue

        else:
            if not (t0 <= t <= t1 and u0 <= u <= u1 and v0 <= v <= v1):
                continue
            N = np.cross(surf_eval["Su"], surf_eval["Sv"])
            N / np.linalg.norm(N)

            tng = curve_eval["C1"] / np.linalg.norm(curve_eval["C1"])

            if np.abs(np.dot(N, tng)) <angle_tol:
                
                results.append(("degenerate", surf_eval["S"], (t, u, v)))
                continue
            else:
                results.append(("transversal", surf_eval['S'], (t, u, v)))

            # print('g', t, u, v)

            cand = new_intersection_candidates(_surface, _curve, u, v, t, surf_eval['S'])
            if debug:
                print(len(cand))
            for sc,cc in cand:

                stack.append((sc,cc, (t,u,v)))
            continue

    return sorted(results,key=lambda x:x[2][0])


if __name__ == "__main__":
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

    surf.normalize_knots()

    curve = NURBSCurve(cpts)
    #ress = new_intersection_candidates(surf, curve, u, v, t, np.array(surf.evaluate_v2(u, v)))

    from mmcore.numeric.intersection.csx._ncsx import nurbs_csx
    import time
    s=time.time()
    r1=int_cs(curve, surf)
    print(time.time()-s, r1)
    s=time.time()
    r2=nurbs_csx(curve, surf)
    print(time.time()-s, r2)

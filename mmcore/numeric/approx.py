import math

import numpy as np

from mmcore.geom._nurbs_eval import NURBSCurveTuple,evaluate_nurbs_curve
from mmcore.numeric.numeric import compute_parametric_curvature_tolerance_curve



def chord_length(R, h):
    return 2 * np.sqrt(2 * R * h - (h * h))




def chord_height(radius: float, chord_length: float) -> float:
    """
    Compute the sagitta (height) of a chord in a circle.

    Parameters
    ----------
    radius : float
        Circle radius (must be > 0).
    chord_length : float
        Length of the chord (must satisfy 0 <= chord_length <= 2*radius).

    Returns
    -------
    float
        The sagitta h = r - sqrt(r^2 - (c/2)^2).
    """
    if not (np.isfinite(radius) and np.isfinite(chord_length)):
        raise ValueError("radius and chord_length must be finite numbers.")
    if radius <= 0:
        raise ValueError("radius must be > 0.")
    if chord_length < 0:
        raise ValueError("chord_length must be >= 0.")
    if chord_length > 2 * radius:
        raise ValueError("chord_length cannot exceed the diameter (2 * radius).")

    half = chord_length / 2.0
    # clamp inside the sqrt to avoid tiny negative due to floating-point roundoff
    inside = max(0.0, radius * radius - half * half)
    return radius - np.sqrt(inside)


def adaptive_curve_sampler_unsafe(crv: NURBSCurveTuple, tol: float = 1e-3):
    tmin,tmax=crv.interval()
    t_current=tmin
    params=[t_current]
    evals=[]
    duu=[]
    ll=[]
    while t_current<tmax:

        c_eval=evaluate_nurbs_curve(crv,t_current, d_order=2)
        if len(evals)>0:
            l=np.linalg.norm(evals[-1]['C'] -   c_eval['C'])
            ll.append(l)
        evals.append(c_eval)
        du= compute_parametric_curvature_tolerance_curve(c_eval["C1"], c_eval["C2"], tol)
        t_current=np.clip(t_current+du,tmin,tmax)
        duu.append(du)

        params.append(t_current)

    c_eval = evaluate_nurbs_curve(crv, t_current, d_order=2)
    l = np.linalg.norm(evals[-1]["C"] - c_eval["C"])
    ll.append(l)
    evals.append(c_eval)

    return params,duu,evals,ll
from mmcore.geom._nurbs_param_tol import nurbs_curve_param_tolerance
from mmcore.geom._nurbs_knots import find_multiplicity,split_curve_multiple
from mmcore.numeric.bern import bern_greville_abscissae
from mmcore.numeric.sbern import bern_to_nurbs_bezier
def adaptive_bez_sampler(crv, tol):
    if crv.control_points.shape[0]==2:
        return crv.control_points
    ptol=nurbs_curve_param_tolerance(crv, tol=tol)
    tmin,tmax=crv.interval()
    dinterv=tmax-tmin
    t=tmin
    params=[]
    evals=[]
    du_cap=ptol
    du_list=[]
    s_list=[]
    grev = bern_greville_abscissae(crv.control_points.shape[0], interval=(tmin, tmax))
    next_grev=1
    while t < tmax - 10 * np.finfo(float).eps:
        ce = evaluate_nurbs_curve(crv, t, d_order=2)  # {"C","C1","C2"}
   
        
        
        C0, C1, C2 = ce["C"], ce["C1"], ce["C2"]
        evals.append(ce)
        
        du = compute_parametric_curvature_tolerance_curve(C1, C2, tol)
        
        if not np.isfinite(du) or du <= 0 or du >= dinterv:
            # Fallback: step by a small param cap using local speed
            print('fallback to du_cap:',du_cap)
            du = du_cap
        
        # Don't overshoot
        du = max(du, du_cap)
        du = min(du, tmax - t)
        
        
        # Arc-length estimate for this step (consistent with your derivation)
        speed = np.linalg.norm(C1)
        s_i = speed * du
        
        du_list.append(du)
        s_list.append(s_i)
        
        t += du
        params.append(t)
        if t>grev[next_grev]:
            t = grev[next_grev]
            next_grev+=1
           
           
        
    
    # Ensure last sample is at tmax
    ce_end = evaluate_nurbs_curve(crv, tmax, d_order=2)
    evals.append(ce_end)
    
    return params, du_list, evals, s_list
 
from mmcore.geom._nurbs_ders import _greville_abscissae as nurbs_ders_greville_abscissae
def adaptive_curve_sampler(crv, tol=1e-3, max_param_step_fraction=12, max_points=int(1e+6)):
    """
    March once so each chord deviates by ~tol (sagitta) using your curvature-based
    stepper. Includes a fallback when κ≈0 so we never return inf.
    Returns:
        params
         du_list
         evals,
          s_list
    """
    
    ptol=nurbs_curve_param_tolerance(crv, tol=tol)
    interv=crv.interval()
    dinterv = interv[1] - interv[0]
    ptol=np.clip(ptol,0,dinterv,)

    prms = np.unique(crv.knot)
    params=[]
    du_list = []
    s_list = []
    evals = []
    
    if crv.order==2:
        
    
        
      
        for p in prms:
            ce = evaluate_nurbs_curve(crv, p, d_order=2)  #
            C0, C1, C2 = ce["C"], ce["C1"], ce["C2"]
            du = compute_parametric_curvature_tolerance_curve(C1, C2, tol)
            evals.append(ce)
            params.append(p)
            speed = np.linalg.norm(C1)
            s_i = speed * du
            du_list.append(du)
            s_list.append(s_i)
        return params,du_list,evals,s_list
    if len(np.unique(crv.knot))>2 :
      internal_knots=prms[1:][:-1]
      knots_to_split=[]
      for k in internal_knots:
        m=find_multiplicity(k,crv.knot)
        if m>1:
            knots_to_split.append(k)
      if len(knots_to_split)>1:
        for curve_segment in split_curve_multiple(crv,knots_to_split) :
                res=adaptive_curve_sampler(curve_segment, tol=tol, max_param_step_fraction=max_param_step_fraction,max_points=max_points)
                params.extend(res[0])
                du_list.extend(res[1])
                evals.extend(res[2])
                s_list.extend(res[3])
           
        return    params,du_list,evals,s_list
        
    

    
    tmin, tmax = crv.interval()
    t = tmin
    params = [t]
    du_list = []
    s_list = []
    evals = []

    # Parametric cap to avoid huge jumps at inflections / κ≈0
    du_cap = ptol
    print(ptol)
    tiny = np.finfo(float).eps
    grev=nurbs_ders_greville_abscissae(crv.knot,crv.order-1)
    next_grev=1
    n_pts = 0
    while t < tmax - 10*np.finfo(float).eps:
        ce = evaluate_nurbs_curve(crv, t, d_order=2)  # {"C","C1","C2"}
        n_pts += 1
        
        if (max_points is not None ) and (max_points >0) and n_pts > max_points:
            raise RuntimeError("Too many points; possible stagnation. Increase tol or max_points.")

        C0, C1, C2 = ce["C"], ce["C1"], ce["C2"]
        evals.append(ce)

        du = compute_parametric_curvature_tolerance_curve(C1, C2, tol)
        
        if not np.isfinite(du) or du <= 0 or du >= dinterv:
            # Fallback: step by a small param cap using local speed
            
            du = du_cap

        # Don't overshoot
        du = min(du, tmax - t)
        du = max(du, tiny)

        # Arc-length estimate for this step (consistent with your derivation)
        speed = np.linalg.norm(C1)
        s_i = speed * du

        du_list.append(du)
        s_list.append(s_i)

        t += du
        params.append(t)
        if t > grev[next_grev]:
            t = grev[next_grev]
            next_grev += 1
    
    # Ensure last sample is at tmax
    ce_end = evaluate_nurbs_curve(crv, tmax, d_order=2)
    evals.append(ce_end)

    return params, du_list, evals, s_list



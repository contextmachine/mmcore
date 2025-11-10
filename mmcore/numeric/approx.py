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


import numpy as np

from mmcore.numeric.bern import *
from scipy.spatial import ConvexHull

from mmcore.geom._nurbs_param_tol import _nurbs_curve_param_tol_conservative
from mmcore.numeric.sbern import bern_to_nurbs_bezier


def minimum_3d_obb(points, tol=1e-8):
    # … same as before …
    P = np.asarray(points)
    hull = ConvexHull(P)
    best = {'vol': np.inf}
    for tri in hull.simplices:
        A, B, C = P[tri]
        n = np.cross(B - A, C - A)
        norm = np.linalg.norm(n)
        if norm < tol: continue
        z = n / norm
        if abs(z[0]) < abs(z[1]):
            u = np.cross(z, [1, 0, 0])
        else:
            u = np.cross(z, [0, 1, 0])
        u /= np.linalg.norm(u)
        v = np.cross(z, u)
        
        proj = P.dot(np.vstack((u, v)).T)
        ch2 = ConvexHull(proj)
        pts2 = proj[ch2.vertices]
        
        zs = P.dot(z)
        for dx, dy in np.diff(np.vstack((pts2, pts2[0])), axis=0):
            th = np.arctan2(dy, dx)
            c, s = np.cos(th), np.sin(th)
            R = np.array([[c, s],
                          [-s, c]])
            rot = pts2.dot(R.T)
            min_x, max_x = rot[:, 0].min(), rot[:, 0].max()
            min_y, max_y = rot[:, 1].min(), rot[:, 1].max()
            lx, ly = max_x - min_x, max_y - min_y
            lz = zs.max() - zs.min()
            vol = lx * ly * lz
            if vol < best['vol']:
                best.update(
                    vol=vol,
                    origin=(u * (c * min_x - s * min_y) +
                            v * (s * min_x + c * min_y) +
                            z * zs.min()),
                    xaxis=c * u + s * v,
                    yaxis=-s * u + c * v,
                    zaxis=z,
                    extents=(lx, ly, lz)
                )
    return (best['origin'],
            best['xaxis'], best['yaxis'], best['zaxis'],
            best['extents'])




def fit_plane_svd(points, weights=None, eps=1e-12):
    """
    Best-fit plane via SVD.
    Returns: centroid (3,), unit normal (3,), max_abs_deviation (float), distances (N,)
    """
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3 or P.shape[0] < 3:
        raise ValueError("points must be (N,3) with N>=3")
    
    if weights is None:
        centroid = P.mean(axis=0)
        X = P - centroid
        U, S, Vt = np.linalg.svd(X, full_matrices=False)
    else:
        w = np.asarray(weights, dtype=float).reshape(-1)
        if w.shape[0] != P.shape[0]:
            raise ValueError("weights length must match number of points")
        if np.any(w < 0):
            raise ValueError("weights must be nonnegative")
        Wsum = w.sum()
        if Wsum <= eps:
            raise ValueError("sum of weights must be positive")
        centroid = (w[:, None] * P).sum(axis=0) / Wsum
        X = P - centroid
        Xw = X * np.sqrt(w[:, None])  # weighted design matrix
        U, S, Vt = np.linalg.svd(Xw, full_matrices=False)
    
    normal = Vt[-1]
    # guard against degeneracy
    nrm = np.linalg.norm(normal)
    if nrm <= eps:
        raise RuntimeError("degenerate configuration: points are (near) collinear or identical")
    normal = normal / nrm
    
    # signed point-to-plane distances
    dists = (P - centroid) @ normal
    max_abs_dev = np.max(np.abs(dists))
    return centroid, normal, max_abs_dev, dists

from mmcore.numeric.bern import de_casteljau_subdivide_2d


def _gen_cpts_to_display(scalar_net):
    greville = bern_greville_abscissae_nd(scalar_net.shape)
    Pts = np.zeros((*scalar_net.shape, scalar_net.ndim + 1))
    for i in range(scalar_net.shape[0]):
        for j in range(scalar_net.shape[1]):
            Pts[i, j, 2] = scalar_net[i, j]
            Pts[i, j, 0] = greville[0][i]
            Pts[i, j, 1] = greville[1][j]
    
    return Pts



class BernsteinTree2D:
    
    links:list[tuple[int,int]]
    control_points:list[np.ndarray]
    bounding_boxes:list[tuple[float,float,float,float]]
    
    
    def __init__(self, control_points:np.ndarray):
        self._initial_control_points=control_points
        self.greville_abscissae=bern_greville_abscissae_nd(control_points.shape)
    @property
    def degree_u(self):
        return self._initial_control_points.shape[0]-1
    
    @property
    def degree_v(self):
        return self._initial_control_points.shape[1]-1
    
    @property
    def order_u(self):
        return self._initial_control_points.shape[0]
    
    @property
    def order_v(self):
        return self._initial_control_points.shape[1]
        
def adaptive_bern_sampler_2d(nu: NDArray[float], tol:float=1e-3):
    stack = [nu]
    quads = []
    
    while stack:
        
        subpatch = stack.pop(0)
        
        centroid, normal, max_abs_dev, dists = fit_plane_svd(subpatch.reshape((-1, subpatch.shape[-1])))
        if max_abs_dev < tol:
            
            quads.append((subpatch[0, 0], subpatch[0, -1], subpatch[-1, -1], subpatch[-1, 0]))
            continue
        elif np.array(aabb(subpatch.reshape(-1, 3))).max() < tol:
            quads.append(
                (subpatch[0, 0], subpatch[0, -1], subpatch[-1, -1],
                 subpatch[-1, 0]))
            continue
        else:
            stack.extend(de_casteljau_subdivide_2d(subpatch, 0.5, 0.5))
    return quads


from mmcore.numeric.aabb import aabb




from mmcore.geom._nurbs_param_tol import nurbs_curve_param_tolerance


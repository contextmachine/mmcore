

from scipy.integrate import solve_ivp
import numpy as np

from mmcore.nurbs._core import NURBSSurface

from mmcore.numeric import compute_parametric_sectional_curvature_tolerance_surface, \
    compute_parametric_curvature_tolerance_curve, compute_parametric_tolerance_curve, \
    compute_parametric_tolerance_surface
from mmcore.numeric.closest_point import nurbs_surface_closest_point
from mmcore.nurbs._nurbs_eval import NURBSCurveTuple, NURBSSurfaceTuple, evaluate_nurbs_curve,evaluate_nurbs_surface


from mmcore.numeric.newton.cnewton import newtons_method
def pull_curve(surf:NURBSSurfaceTuple, curve:NURBSCurveTuple):
    """
    from numpy._typing import NDArray


#FIXME Что то на поверхности, improve_uv или еще где то рядом имеет серьезный баг из-за которого марш просто прекращается в как либо точке. Что на спиральках что здесь
class Sphere(Surface):
    def evaluate(self, uv) -> NDArray[float]:
        u,v=uv
        x = np.sin(u) * np.cos(v)
        y = np.sin(u) * np.sin(v)
        z = np.cos(u)
        return np.array([x, y, z])
    def interval(self):
        return np.array([(0.,np.pi*2),(-np.pi,0.0)])

    :param surf:
    :param curve:
    :return:
    """
    tmin,tmax=curve.interval()
    
    start=curve.start()
    
    uv0,_=nurbs_surface_closest_point(surf,start, 1e-3)

    def curve_derivative(t, p):
      
        eval_curve = evaluate_nurbs_curve(curve,t, d_order=1)
        return eval_curve['C1']
    def orthogonal_projective_tensor(uv):
        eval_surf=evaluate_nurbs_surface(surf, uv[0],uv[1], d_order=1)
        dS_du, dS_dv = eval_surf['Su'],eval_surf['Sv']
        K = np.array([[np.dot(dS_du, dS_du), np.dot(dS_du, dS_dv)],
                      [np.dot(dS_dv, dS_du), np.dot(dS_dv, dS_dv)]])
        return K,eval_surf

    def differential_equation(t, uv):
        K,eval_surf = orthogonal_projective_tensor(uv)
        #eval_surf = evaluate_nurbs_surface(surf, uv[0], uv[1], d_order=1)
        dS_du, dS_dv = eval_surf['Su'], eval_surf['Sv']
        dp_dt = curve_derivative(t, uv)
        uvd=improve_uv(  dS_du,  dS_dv, eval_surf['Su'],eval_surf['Su']+ dp_dt )
        #du_dt = np.linalg.solve(K, dp_dt[:2])
        print(uvd)
        return uvd

    sol = solve_ivp(differential_equation, ( tmin,5.), uv0, method='RK45',max_step=0.01)
    return sol

def pull_back(surface:NURBSSurfaceTuple, curve:NURBSCurveTuple,t0:float,t1:float,u0:float,v0:float,u1:float,v1:float,tol=1e-3):
    (t, u, v) = (t0, u0, v0)
    surf_eval = evaluate_nurbs_surface(surface, u, v, d_order=2)
    curve_eval = evaluate_nurbs_curve(curve, t, d_order=2)
    ts=[]
    uvs=[]
    ppts=[]
    while t < t1:
        ts.append(t)
        uvs.append((u,v))
        
        Su, Sv,Suu,Suv,Svv =surf_eval['Su'],surf_eval['Sv'],surf_eval['Suu'],surf_eval['Suv'],surf_eval['Svv']
        C1=curve_eval['C1']
        C2=curve_eval['C2']
        ppts.append(curve_eval['C'])

        #E = np.dot(Su, Su)
        #F = np.dot(Su, Sv)
        #G = np.dot(Sv, Sv)
        #b=[np.dot(Su, C1),np.dot(Sv, C1)]
        #dudv=np.linalg.lstsq([[E,F],[F,G]], b,rcond=None)[0]
        #print("DUDV", dudv)
        
        du, dv=        compute_parametric_sectional_curvature_tolerance_surface(Su, Sv, Suu,
                                                             Suv, Svv, C1/np.linalg.norm(C1),tol)
        if not np.isfinite(du):
            du=min((u1-u0) /4,u1-u)
            
            #du,dv = compute_parametric_tolerance_surface(Su, Sv, Suu, Suv,Svv,tol)
        if not np.isfinite(dv):
            dv=min((v1-v0) /4,v1-v)
            
        _dt = compute_parametric_curvature_tolerance_curve(C1,C2,  tol)
        if not np.isfinite(_dt):
            _dt=min((t1 - t0) / 4, t1 - t)

        u += du
        v += dv
        
        surf_eval = evaluate_nurbs_surface(surface, u, v, d_order=2)
       
        def _eq(dt):
            nonlocal surf_eval    ,t
            evl=evaluate_nurbs_curve(curve,t+np.abs(dt[0]), d_order=0)
            dd=evl['C']- surf_eval['S']
            return np.dot(dd,dd)
        result=newtons_method(_eq,  np.asarray([_dt])  )
        
        if result is not None and np.all(np.isfinite(result)):
            t+=result[0]
        else:
            t+=_dt
      
    ts.append(t)
    uvs.append((u, v))
    ppts.append(curve.end())
    return ts, uvs,ppts
        
        
        
        
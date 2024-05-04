"""Grasshopper Script"""
import numpy as np
import scipy

from scipy.optimize import fsolve
from mmcore.geom.vec import cross, norm
from mmcore.numeric import evaluate_curvature
from mmcore.numeric.closest_point import closest_point_on_line
from mmcore.numeric.plane import plane_plane_intersect,plane_plane_plane_intersect
from mmcore.geom.curves.knot import interpolate_curve
from mmcore.geom.curves.bspline import NURBSpline
TOL=0.1
def freeform_step_debug(s1,s2,uv1,uv2):
    pl1,pl2=s1.plane_at(uv1),s2.plane_at(uv2)
    ln=np.array(plane_plane_intersect(pl1,pl2))
    np1= closest_point_on_line((ln[0], ln[0] + ln[1]), pl1[0])
    np2= closest_point_on_line((ln[0], ln[0] + ln[1]), pl2[0])

    return np1, np1+(np2-np1)/2,np2
def solve_marching(s1,s2, uv1,uv2, tol):
    pl1,pl2=s1.plane_at(uv1),s2.plane_at(uv2)


    marching_direction=cross(pl1[-1], pl2[-1])
    K=evaluate_curvature(marching_direction, pl1[-1])[1]
    r=1/np.linalg.norm(K)

    step=np.sqrt(r ** 2 - (r - tol) ** 2) * 2
    #print(r,step)
    new_pln=np.array([pl1[0]+marching_direction*step, pl1[-1], pl2[-1], marching_direction])

    return plane_plane_plane_intersect(pl1,pl2,new_pln),step


def improve_uv(s,uv_old,xyz_better):
    x_old,y_old,z_old=s.evaluate(uv_old)

    dxdu,dydu,dzdu=s.derivative_u(uv_old)
    dxdv,dydv,dzdv=s.derivative_v(uv_old)
    x_better,y_better,z_better=xyz_better

    xy= [[dxdu,dxdv], [dydu, dydv]],[x_better-x_old,y_better-y_old]
    xz= [[dxdu,dxdv], [dzdu,dzdv]],[x_better-x_old,z_better-z_old]
    yz= [[dydu,dydv], [dzdu,dzdv]],[y_better-y_old,z_better-z_old]
    ##print( xy,xz,yz,'\n\n')
    max_det=sorted([ xy,xz,yz],key= lambda Ab: scipy.linalg.det(Ab[0]), reverse=True)[0]
    return np.linalg.solve( *max_det)

def freeform_step(s1,s2,uvb1,uvb2,tol=TOL, cnt=0):

    xyz_better=freeform_step_debug(s1,s2,uvb1,uvb2)[1]

    uvb1_better=uvb1+improve_uv(s1,uvb1, xyz_better)
    uvb2_better=uvb2+improve_uv(s2,uvb2, xyz_better)
    xyz1_new=s1(uvb1_better)
    xyz2_new=s2(uvb2_better)

    if np.linalg.norm(xyz1_new-xyz2_new)<tol:
        return (  xyz1_new, uvb1_better), (  xyz2_new, uvb2_better)
    else:



        return freeform_step(s1,s2, uvb1_better, uvb2_better,tol,cnt+1)

def marching_step(s1, s2, uvb1,uvb2, tol=TOL, cnt=0):

    xyz_better, step=solve_marching(s1, s2, uvb1, uvb2, tol)
    uvb1_better=uvb1+improve_uv(s1,uvb1, xyz_better)
    uvb2_better=uvb2+improve_uv(s2,uvb2, xyz_better)

    xyz1_new=s1(uvb1_better)
    xyz2_new=s2(uvb2_better)
    if np.linalg.norm(xyz1_new-xyz2_new)<tol:
        return (  xyz1_new, uvb1_better), (  xyz2_new, uvb2_better), step

    else:


            return marching_step(s1,s2, uvb1_better, uvb2_better,tol,cnt+1)

def check_surface_edge(uv, surf_interval, tol):
    bounds_u, bounds_v = surf_interval
    u, v = uv



    return any(    [u<=0,u>=1, v<=0,v>=1])
def check_wrap_back( xyz, initial_xyz, step, tol):
    distance=np.linalg.norm(xyz- initial_xyz)

    return  (distance < step/2 )

def stop_check(s1, s2, xyz1,xyz2,uv1,uv2, initial_xyz,step, tol=TOL, iterations=0, max_iter=100):
    res=[
        norm( xyz1-xyz2)>=tol,
        check_surface_edge( uv1, s1.interval(),tol) or check_surface_edge(  uv2, s2.interval(),tol),
        check_wrap_back(xyz1, initial_xyz, step,tol) or check_wrap_back(xyz2, initial_xyz, step, tol),
        iterations>=max_iter

    ]
    r=any(res)


    return r



def marching_method(
    s1, s2, initial_uv1, initial_uv2, tol=TOL, max_iter=1000, no_ff=False
):
    iterations = 0


    (initial_xyz1, uv1),(initial_xyz2,uv2)=freeform_step(s1,s2,initial_uv1,initial_uv2)
    (xyz1,uv1_new),(xyz2,uv2_new),step = marching_step(s1,s2,uv1,uv2)
    uvs = [(uv1_new, uv2_new)]
    pts = [(xyz1, xyz2)]
    #print(uv1_new, uv2_new)

    steps = [step]
    while True:
        iterations += 1

        uv1, uv2 = uv1_new, uv2_new
        (xyz1, uv1_new), (xyz2, uv2_new), step = marching_step(s1, s2, uv1, uv2)
        if not stop_check(
            s1,
            s2,
            xyz1,
            xyz2,
            uv1_new,
            uv2_new,
            initial_xyz1,
            step,
            tol,
            iterations,
            max_iter,
        ):
            pts.append((xyz1, xyz2))
            uvs.append((uv1_new.tolist(), uv2_new.tolist()))
            steps.append(step)
        else:
            break

    return uvs, pts, steps



def surface_local_cpt(surf1,surf2):
    def fun(t):
        return  np.array(surf2.evaluate(np.array([t[1],t[2]])))-np.array(surf1.evaluate(np.array([0.00,t[0]])))
    return fsolve(fun, np.array([0.5,0.5,0.5]))

def surface_ppi(surf1,surf2):
    aa = surface_local_cpt(surf1,surf2)
    objs = marching_method(surf2, surf1, np.array((aa[1], aa[2])), np.array((0.0, aa[0])))

    pts,knots, deg=interpolate_curve(  np.array(objs[1])[:,0,:],3)
    return NURBSpline(pts,knots=knots,degree=deg)



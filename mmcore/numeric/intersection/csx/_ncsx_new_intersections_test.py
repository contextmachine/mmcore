import numpy as np

from mmcore.numeric.intersection.separability.spherical import spherical_separability,separating_circles_test
from mmcore.numeric.aabb import aabb_intersect_fast_3d,aabb
from mmcore.geom.nurbs import (
    NURBSCurve,
    NURBSSurface,
    split_surface_v,
    split_surface_u,
    split_curve,
    subdivide_surface,
    CurveSurfaceEq,
)
from mmcore.geom._nurbs_eval import NURBSCurveTuple,NURBSSurfaceTuple,_surface_interval,_curve_interval
from mmcore.geom import _nurbs_knots
from ._ch2d import convex_hulls_intersect,convex_hull
from ._steriographic_projection import stereographic_projection
def _has_new_int_patch_segm(surface,curve, xyz):


    def sorter(x):
        d=xyz - x
        return np.dot(d,d)
    
    curve_points=np.array(sorted(curve.control_points,key=sorter)[1:])
    surf_points = np.array(sorted(surface.control_points.reshape((-1,3)), key=sorter)[1:])
    
    if not aabb_intersect_fast_3d(aabb(curve_points),aabb(surf_points)):
        return False
    curve_d = curve_points - xyz.reshape((-1, 3))
    surf_d = surf_points - xyz.reshape((-1, 3))
    curve_d2d= stereographic_projection(curve_d).tolist()
    surf_d2d = stereographic_projection(surf_d).tolist()
    #print([curve_d2d,surf_d2d])
    ch_curve=convex_hull(curve_d2d, 1e-15)
    ch_surf=convex_hull(surf_d2d,1e-15)

    return convex_hulls_intersect(ch_curve,ch_surf,1e-15)

def new_intersection_candidates(surface,curve, u,v,t,xyz):

    candidates=[]
    for s in _nurbs_knots.subdivide_surface(surface, u, v):
        for c in _nurbs_knots.split_curve(curve, t):

            res=_has_new_int_patch_segm(s,c,xyz)
            if res:
                candidates.append((s,c))

    return candidates
def _find_interv(segms,patches,t,u,v):
        patch=None
        segm=None
    
        for p in patches:
            (u0,u1),(v0,v1)=_surface_interval(p)
            if u0<=u<=u1 and v0<=v<=v1:
                patch=p
                break
        for s in segms:
            
            (t0,t1)=_curve_interval(s)
            if t0<=t<=t1:
                segm=s
        
                break
        return segm,patch
def _subdivide_d(obj, u=0.5,v=0.5):
    if isinstance(obj,NURBSCurveTuple):
        (t0,t1)=_curve_interval(obj)
        t_mid=(t1-t0)*u+t0
        return t_mid
    if isinstance(obj,NURBSSurfaceTuple):
        (u0,u1),(v0,v1)=_surface_interval(obj)
        u_mid=(u1-u0)*u+u0
        v_mid=(v1-v0)*v+v0
        return u_mid,v_mid
def _subdivide_p(obj, u=0.5,v=0.5):
    if isinstance(obj,NURBSCurveTuple):
        (t0,t1)=_curve_interval(obj)
        t_mid=(t1-t0)*u+t0
        return _nurbs_knots.split_curve(obj,t_mid)
    if isinstance(obj,NURBSSurfaceTuple):
        (u0,u1),(v0,v1)=_surface_interval(obj)
        u_mid=(u1-u0)*u+u0
        v_mid=(v1-v0)*v+v0
        
        return _nurbs_knots.subdivide_surface(obj,u_mid,v_mid)


def intersection_candidate_cutoff(surf,curve,u,v,t,xyz):
    """
    
    :param surf:
    :param curve:
    :param u:
    :param v:
    :param t:
    :param xyz:
    :return:
    
    o---------+---------+
    |  u1v1   |   u1v2  |
    |         |         |
    +---------+---------+
    |  u1v2   |         |
    |         |         |
    +---------+---------+
    
    """

    stack=[(_subdivide_d(curve,0.5),_subdivide_d(surf,0.5,0.5))]
    while stack:
        t_mid,(u_mid,v_mid)=stack.pop(0)

        segms= list(_nurbs_knots.split_curve(curve,t_mid))
        patches=  list(_nurbs_knots.subdivide_surface(surf, u_mid, v_mid))

        segm,patch=_find_interv(segms,patches, t, u, v)
    
        if _has_new_int_patch_segm( patch, segm,xyz):

            stack.append(
                (_subdivide_d(segm,0.5),
                 _subdivide_d(patch,0.5,0.5)
                 )
                
            )

        else:
  
            
            return list(filter(lambda x: x is not segm, segms)), list(filter(lambda x: x is not patch, patches))


def new_intersection_candidates_cutoff(surface, curve, u, v, t, xyz):

    candidates = []
    for s in _nurbs_knots.subdivide_surface(surface, u, v):
        for c in _nurbs_knots.split_curve(curve, t):

            res = _has_new_int_patch_segm(s, c, xyz)
            
            if res:
                segms,patches=intersection_candidate_cutoff(s, c, u, v, t, xyz)
                for c in segms:
                    for s in patches:
                        
                        candidates.append((s, c))
    
    return candidates

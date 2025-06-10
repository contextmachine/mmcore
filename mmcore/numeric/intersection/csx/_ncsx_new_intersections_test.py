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
from mmcore.geom._nurbs_eval import NURBSCurveTuple,NURBSSurfaceTuple
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




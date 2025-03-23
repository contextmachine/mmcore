import numpy as np

from mmcore.numeric.intersection.separability.spherical import spherical_separability,separating_circles_test
from mmcore.numeric.aabb import aabb_intersect_fast_3d
from mmcore.geom.nurbs import (
    NURBSCurve,
    NURBSSurface,
    split_surface_v,
    split_surface_u,
    split_curve,
    subdivide_surface,
    CurveSurfaceEq,
)
from ._ch2d import convex_hulls_intersect,convex_hull
from ._steriographic_projection import stereographic_projection
def _has_new_int_patch_segm(surface,curve, xyz):
    if not aabb_intersect_fast_3d(curve.bbox(),surface.bbox()):
        return False
    curve_d=[]
    surface_d=[]
    for cpt in np.array(curve.control_points):
        d = cpt - xyz

        if not np.all(np.abs(d)<1e-12):
            curve_d.append(d / np.linalg.norm(d))


    for cpt in np.array(surface.control_points_flat):
        d = cpt - xyz
        if not  np.all(np.abs(d)<1e-12):
            surface_d.append(d / np.linalg.norm(d))



    curve_d2d= stereographic_projection(np.array(curve_d)).tolist()
    surf_d2d = stereographic_projection(np.array(surface_d)).tolist()
    #print([curve_d2d,surf_d2d])
    ch_curve=convex_hull(curve_d2d, 1e-15)
    ch_surf=convex_hull(surf_d2d,1e-15)

    return convex_hulls_intersect(ch_curve,ch_surf,1e-15)

def new_intersection_candidates(surface,curve, u,v,t,xyz):

    candidates=[]
    for s in subdivide_surface(surface, u, v, tol=1e-12,normalize_knots=False):
        for c in split_curve(curve, t, tol=1e-12,normalize_knots=False):

            res=_has_new_int_patch_segm(s,c,xyz)
            if res:
                candidates.append((s,c))

    return candidates




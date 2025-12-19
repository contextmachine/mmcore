import math

import numpy as np

from mmcore.geom._nurbs_knots import generate_knots
from mmcore.geom._nurbs_eval import to_homogeneous_1d, from_homogeneous_2d, NURBSCurveTuple,NURBSSurfaceTuple

def nurbs_surface(control_points, degree_u=None, degree_v=None, rational=False, interval_u=None, interval_v=None, periodic_u=False, periodic_v=False,**kwargs)->NURBSSurfaceTuple:
    if periodic_u or periodic_v:
        raise NotImplementedError
    control_points=np.array(control_points)
    if degree_u is None:
        degree_u=min(control_points.shape[0]-1,3)
    if degree_v is None:
        degree_v=min(control_points.shape[1]-1,3)

    if rational:
        control_points, w=from_homogeneous_2d(control_points)
    else:
        w=np.ones_like(control_points[...,-1])
    ku=generate_knots(control_points.shape[0],degree=degree_u,interval=interval_u)
    kv = generate_knots(control_points.shape[1], degree=degree_v, interval=interval_v)
    return NURBSSurfaceTuple(degree_u+1, degree_v+1,ku, kv, control_points, w)

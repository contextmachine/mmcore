
import math

import numpy as np

from mmcore.geom._nurbs_knots import generate_knots
from mmcore.geom._nurbs_eval import to_homogeneous_1d, from_homogeneous_2d, NURBSCurveTuple,NURBSSurfaceTuple

def nurbs_curve(control_points, degree=None, rational=False, interval=None, periodic=False,**kwargs)->NURBSCurveTuple:
    if periodic:
        raise NotImplementedError
    control_points=np.array(control_points)
    if degree is None:
        degree=min(control_points.shape[0]-1,3)
    if rational:
        control_points, w=from_homogeneous_2d(control_points)
    else:
        w=np.ones_like(control_points[...,-1])
    k=generate_knots(control_points.shape[0],degree=degree,interval=interval)
    return NURBSCurveTuple(degree+1,k,control_points,w )
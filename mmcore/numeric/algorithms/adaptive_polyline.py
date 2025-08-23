from __future__ import annotations

from mmcore.numeric.approx import  adaptive_curve_sampler
from mmcore.geom.nurbs import NURBSCurve
from mmcore.geom._nurbs_eval import _nurbs_to_tuple,NURBSCurveTuple
import numpy as np

def adaptive_polyline(curve: NURBSCurve|NURBSCurveTuple, tol:float=1e-3, **kwargs):
  
    if isinstance(curve,NURBSCurve):
        curve=_nurbs_to_tuple(curve)
    params,duu,evals, s_segms= adaptive_curve_sampler(curve, tol=tol, **kwargs)
    return np.array([item['C'] for item in evals]), np.array(params)

import sys

from mmcore.numeric.approx import adaptive_curve_sampler

sys.setrecursionlimit(100000)
from mmcore.geom.nurbs import NURBSCurve
from mmcore.geom._nurbs_eval import _nurbs_to_tuple


def adaptive_polyline(curve: NURBSCurve, tol:float, **kwargs):
    spt=tol
    if isinstance(curve,NURBSCurve):
        curve=_nurbs_to_tuple(curve)
    params,duu,evals, _= adaptive_curve_sampler(curve, spt=spt)
    return np.array([item['C'] for item in evals]),np.array(params)
# Example usage:
import numpy as np


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------

# ------------------------------------------------------------
# Safe sampler (wraps your curvature-based step with a fallback)
# ------------------------------------------------------------


# ---------- Plane fit & projection ----------




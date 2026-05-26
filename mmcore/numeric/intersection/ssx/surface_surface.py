from __future__ import annotations

import warnings

from typing import NamedTuple, Literal,Any, Protocol, runtime_checkable,Callable

from numpy._typing import NDArray

from mmcore.geom.nurbs import NURBSSurface, NURBSCurve
from mmcore.geom._nurbs_eval import NURBSCurveTuple,NURBSSurfaceTuple,_nurbs_to_tuple,_tuple_to_nurbs




from mmcore.numeric.vectors import solve2x2
from mmcore.numeric.intersection.ssx._ssx_utils import improve_uv as cimprove_uv
from mmcore.numeric.algorithms.point_inversion import point_inversion_surface


import numpy as np
from mmcore.numeric.vectors import norm, det


def improve_uv(du, dv, xyz_old, xyz_better, res):
    dxdu, dydu, dzdu = du
    dxdv, dydv, dzdv = dv

    delta = xyz_better - xyz_old

    xy = np.array([[dxdu, dxdv], [dydu, dydv]]), [delta[0], delta[1]]
    xz = np.array([[dxdu, dxdv], [dzdu, dzdv]]), [delta[0], delta[2]]
    yz = np.array([[dydu, dydv], [dzdu, dzdv]]), [delta[1], delta[2]]

    max_det = max([xy, xz, yz], key=lambda Ab: det(Ab[0]))

    return solve2x2(max_det[0], np.array(max_det[1]), res)


def improve_uv_robust(surf, uv_old, du, dv, xyz_old, xyz_better, uv_better=None, ptol=1e-6):
    if uv_better is None:
        uv_better = np.zeros(2)

    success_first = cimprove_uv(du, dv, xyz_old, xyz_better, uv_better)

    if success_first == 1:
        uv_better[:] = point_inversion_surface(surf, xyz_better, *uv_old, ptol, ptol)
    else:
        uv_better += uv_old

    return uv_better



@runtime_checkable
class Implicit(Protocol):
    def __call__(self, p:NDArray[np.float64])-> float|NDArray[np.float64]:
        pass
    def bounds(self)->NDArray[np.float64]|tuple[tuple[float, float,float], tuple[float, float,float]]:
        pass

from mmcore.numeric.intersection.ssx._ssx4 import nurbs_ssx,SSXBranch,SSXPoint

class CommonSSXBranch(NamedTuple):
    curve: NURBSCurveTuple
    ssx_type:Literal["PP", "IP", "II"]
    source:SSXBranch|Any=None

class CommonSSXPoint(NamedTuple):
    xyz: NDArray[np.float64]
    ssx_type:Literal["PP", "IP", "II"]
    source:SSXPoint|Any=None

import logging
_logger=logging.getLogger('mmcore')
def ssx(surf1: NURBSSurface|NURBSSurfaceTuple, surf2: NURBSSurface|NURBSSurfaceTuple,  atol=0.001, angle_tol=0.052, **kwargs) -> tuple[list[CommonSSXBranch],list[CommonSSXPoint]]:
    """
    Calculate the intersection of two parametric surfaces.

    SSX Types:

    | kind           | P (Parametric) | I (Implicit)  |
    |----------------|----------------|---------------|
    | P (Parametric) | PP             | IP (not impl) |
    | I (Implicit)   | IP  (not impl) | II (not impl) |


    """
    if 'spt' in kwargs:
        warnings.warn('spt keyword is deprecated. Use atol instead', DeprecationWarning)
        atol=kwargs.pop('spt')
    if 'tol' in kwargs:
        warnings.warn('tol keyword is deprecated and will be removed in the future. The ssx public interface no longer supports setting a user-defined parametric tolerance; the value will be ignored. \nInstead, use a combination of atol and angle_tol to control accuracy.', DeprecationWarning)
    if isinstance(surf1,NURBSSurface):
        surf1=_nurbs_to_tuple(surf1)
    if isinstance(surf2,NURBSSurface):
        surf2=_nurbs_to_tuple(surf2)
    if isinstance(surf1,NURBSSurfaceTuple) and isinstance(surf2,NURBSSurfaceTuple):
        ssx_type="PP"
    elif isinstance(surf1,Implicit) and isinstance(surf2,Implicit):
        ssx_type="II"
        raise NotImplementedError("At this time, the ssx public interface does not support type II intersections (Implicit x Implicit).")
    elif any((isinstance(surf1, Implicit) ,isinstance(surf2, Implicit))) and any((isinstance(surf1, (NURBSSurface,NURBSSurfaceTuple)) ,isinstance(surf2, (NURBSSurface,NURBSSurfaceTuple)))):
        ssx_type = "IP"
        raise NotImplementedError("At this time, the ssx public interface does not support type IP intersections (Implicit x Parametric).")
    else:

        raise ValueError(f"Unsupported type: {type(surf1)} or {type(surf2)}")

    res = nurbs_ssx(surf1, surf2, atol=atol,angle_tol=angle_tol)

    curves,pts=res

    branches = []
    for i, curve_dt in enumerate(curves):
        if isinstance(curve_dt,SSXBranch):
            branches.append(CommonSSXBranch(curve_dt.curve_xyz, ssx_type="PP", source=curve_dt))

    isolated:list[CommonSSXPoint]=[]
    for i in range(len(pts)):
        pt = pts[i]
        if isinstance(pt,SSXPoint):
            isolated.append(CommonSSXPoint(pt.xyz,"PP",pt))

    return branches, isolated

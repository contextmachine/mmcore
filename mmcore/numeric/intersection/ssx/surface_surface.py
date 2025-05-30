from __future__ import annotations
import copy
import itertools
import sys
import time
import warnings
from array import array
from dataclasses import dataclass

from mmcore.geom.nurbs import NURBSSurface, NURBSCurve
from scipy.integrate import solve_bvp, solve_ivp

from mmcore.geom.surfaces import CurveOnSurface, Surface
from scipy.spatial import KDTree

from mmcore.geom.bvh import BoundingBox, intersect_bvh_objects, BVHNode, Triangle
from mmcore.geom.surfaces import Surface, Coons
from mmcore.numeric.vectors import solve2x2, det, scalar_dot, scalar_norm
from mmcore.numeric.intersection.ssx._ssx_utils import improve_uv as cimprove_uv
from mmcore.numeric.algorithms.point_inversion import point_inversion_surface

from mmcore.numeric.closest_point import (
    closest_point_on_ray,
    closest_points_on_surface,
    closest_point_on_surface,
    closest_point_on_nurbs_surface,
)
from mmcore.numeric.divide_and_conquer import divide_and_conquer_min_nd
from mmcore.numeric.intersection.csx import nurbs_csx
from mmcore.numeric.intersection.ssx._detect_intersections import detect_intersections

from mmcore.numeric.plane import plane_plane_intersect

from mmcore.geom.curves.bspline import NURBSpline, interpolate_nurbs_curve

import numpy as np
from mmcore.numeric.vectors import norm, det
from collections import namedtuple
from typing import NamedTuple, Optional, Tuple

from mmcore.numeric import scalar_cross, scalar_norm


from mmcore.numeric.fdm import DEFAULT_H
from mmcore.numeric.intersection.ssx._terminator import TerminatorType

from mmcore.numeric.plane import plane_plane_plane_intersect_points_and_normals
from mmcore.numeric.intersection.ssx._terminator import (
    surface_surface_boundary_intersection,
)

from numpy.typing import NDArray



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






from mmcore.numeric.intersection.ssx.boundary_intersection import find_boundary_intersections, \
    IntersectionPoint
from mmcore.geom.nurbs_iso import extract_isocurve



from mmcore.numeric.intersection.ssx._ssx31 import _nurbs_trace_intersection_curves_v2 as _nurbs_trace_intersection_curves,SamplingMethod
def surface_ppi(surf1: Surface, surf2: Surface, spt=0.001,tol=1e-7, tan_tol=1e-3, **kwargs):

    # s=time.perf_counter_ns()[(0.12254503038194443, 0.607421875), (0.12037037478552923, 0.6044921875),
    #edge_terminator = surface_surface_boundary_intersection(surf1, surf2, tol=tol)
    # times.append(time.perf_counter_ns()-s)

    #freeform = FreeFormMethod(surf1, surf2, tol=tol, boundary_terminators=edge_terminator, max_iter=19)
    # s = time.perf_counter_ns()
    if isinstance(surf1, NURBSSurface) and isinstance(surf2, NURBSSurface):
        return _nurbs_trace_intersection_curves(surf1,surf2,tol=tol,spt=spt,tan_tol=tan_tol)

    else:
        raise NotImplemented
import logging
_logger=logging.getLogger('mmcore')
def ssx(surf1: Surface, surf2: Surface,  spt=0.001,tol: float = 1e-7, **kwargs) -> tuple[list[tuple[NURBSCurve, CurveOnSurface, CurveOnSurface]],list[IntersectionPoint]]:
    """
    Calculate the intersection of two parametric surfaces.

    :param surf1: The first surface.
    :type surf1: Surface
    :param surf2: The second surface.
    :type surf2: Surface
    :param tol: The tolerance value for the intersection algorithm (optional, default is 0.01).
    :type tol: float
    :param max_iter: The maximum number of iterations for the intersection algorithm (optional, default is 500). Now
    this parameter exists primarily to debug recursion
    :type max_iter: int
    :param curvature_step:  Use curvature dependent step (experimental, default is False). At the moment it does not give an increase in speed.
    :type curvature_step: bool
    :return: A list of tuples, where each tuple contains an interpolated spatial NURBS curve intersection and the corresponding objects
             CurveOnSurface objects for surf1 and surf2.
    :rtype: list[tuple[NURBSCurve, CurveOnSurface, CurveOnSurface]]

    Note
    -----
    If successful (intersection found), this function returns a list of intersection results because two surfaces can form as many separate intersection curves as desired.

     Since two surfaces can form as many separate intersection curves as desired, the list of intersection results.
     Each intersection result is a separate intersection curve in three views:
        1. A spatial NURBS curve (NURBSpline object).
        2. A curve in the parametric space of the first surface (CurveOnSurface object).
        3. A curve in the parametric space of the second surface (CurveOnSurface

    """
    res = surface_ppi(surf1, surf2, tol=tol,spt=spt)
    if res is None:
        return [],[]
    curves,pts=res
    #curves, curves1_uvs,curves2_uvs, _= zip(*res)
    results = []
    for i, curve_dt in enumerate(curves):

        curve_pts, curve_uvs1, curve_uvs2=curve_dt[0],curve_dt[1],curve_dt[2]
        curve = interpolate_nurbs_curve(curve_pts, 3)


        curve_on_surf1 = interpolate_nurbs_curve(curve_uvs1, 3)
        curve_on_surf2 = interpolate_nurbs_curve(curve_uvs2, 3)

        results.append(
            (
                curve,
                CurveOnSurface(surf1, curve_on_surf1, interval=curve_on_surf1.interval()),
                CurveOnSurface(surf2, curve_on_surf2, interval=curve_on_surf2.interval()),
            )
        )

    return results, pts



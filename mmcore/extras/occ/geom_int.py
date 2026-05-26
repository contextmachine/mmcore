from __future__ import annotations

from typing import TypedDict

import numpy as np
from OCC.Core.Geom2d import Geom2d_Curve
from OCC.Core.Geom2dAPI import Geom2dAPI_InterCurveCurve
from OCC.Core.IntRes2d import IntRes2d_Position, IntRes2d_TypeTrans

from copy import deepcopy
from math import sqrt
from typing import Dict
from typing import List
from typing import Optional
from typing import Union

# from OCC.Core.TopoDS import TopoDS_Edge
from OCC.Core.Geom import Geom_BSplineCurve, Geom_Curve
from OCC.Core.GeomAPI import GeomAPI_Interpolate
from OCC.Core.GeomConvert import GeomConvert_CompCurveToBSplineCurve
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColStd import TColStd_Array1OfInteger
from OCC.Core.TColStd import TColStd_Array1OfReal
from numpy.typing import ArrayLike,NDArray
from typing import List

from OCC.Core.gp import gp_Pnt
from OCC.Core.TColgp import TColgp_Array1OfPnt
from OCC.Core.TColgp import TColgp_Array2OfPnt
from OCC.Core.TColgp import TColgp_HArray1OfPnt
from OCC.Core.TColStd import TColStd_Array1OfInteger
from OCC.Core.TColStd import TColStd_Array1OfReal
from OCC.Core.TColStd import TColStd_Array2OfReal


def array1_from_numpy(points: NDArray[float]) -> TColgp_Array1OfPnt:
    """Construct a one-dimensional point array from a list of points.

    """
    array = TColgp_Array1OfPnt(1, len(points))
    for i in range(points.shape[0]):
        array.SetValue(i + 1, gp_Pnt(*points[i]))
    return array


def harray1_from_numpy(points: NDArray[float]|List[float]) -> TColgp_HArray1OfPnt:
    """Construct a horizontal one-dimensional point array from a list of points.



    """
    array = TColgp_HArray1OfPnt(1, len(points))
    for i in range(points.shape[0]):
        array.SetValue(i + 1, gp_Pnt(*points[i]))
    return array


def array1_from_integers1(numbers: NDArray[int]|List[int]) -> TColStd_Array1OfInteger:
    """Construct a one-dimensional integer array from a list of integers.


    """
    array = TColStd_Array1OfInteger(1, len(numbers))
    for index, number in enumerate(numbers):
        array.SetValue(index + 1, number)
    return array


def array1_from_floats1(numbers: NDArray[float]) -> TColStd_Array1OfReal:
    """Construct a one-dimensional float array from a list of floats.


    """
    array = TColStd_Array1OfReal(1, len(numbers))
    for index, number in enumerate(numbers):
        array.SetValue(index + 1, number)
    return array


def occ_curve(
        points: NDArray[float],
        weights: NDArray[float],
        knots: NDArray[float],
        multiplicities: List[int],
        degree: int,
        is_periodic: bool,
) -> Geom_BSplineCurve:
    return Geom_BSplineCurve(
        array1_from_numpy(points),
        array1_from_floats1(weights),
        array1_from_floats1(knots),
        array1_from_integers1(multiplicities),
        degree,
        is_periodic,
    )


from mmcore.geom._nurbs_knots import generate_knots, find_multiplicity,from_homogeneous_1d


def occ_curve_from_points(points, degree: int = 3, rational:bool=False) -> Geom_BSplineCurve:
    """Construct a B-spline curve from a control_points."""
    p = len(points)
    if not rational:
        weights = np.ones(p)
    else:
        points,weights=from_homogeneous_1d(points)

    degree = degree if p > degree else p - 1


    knots = generate_knots(p, degree)
    unique_knots = np.unique(knots)
    multiplicities = [find_multiplicity(knot, knots) for knot in unique_knots]
    
    is_periodic = False
    
    return occ_curve(
        points,
        weights,
        unique_knots,
        multiplicities,
        degree,
        is_periodic,
    )

from mmcore.geom._nurbs_knots import generate_knots, find_multiplicity
from mmcore.geom._nurbs_eval import NURBSCurveTuple
def occ_curve_from_nt(curve:NURBSCurveTuple) -> Geom_BSplineCurve:
    """Construct a NURBS OCC curve from a NURBSCurveTuple.
    :param curve: mmcore's NURBS curve representation.
    :type: NURBSCurveTuple
    :return: OCC Curve as Geom_BSplineCurve. Yes, in OCC, BSpline accept weights. Don't ask...
    :rtype: Geom_BSplineCurve
    """
    p = len(curve.control_points)
    degree = curve.order-1

    knots = generate_knots(p, degree)
    unique_knots = np.unique(knots)
    multiplicities = [find_multiplicity(knot, knots) for knot in unique_knots]
    
    is_periodic = False
    
    return occ_curve(
        curve.control_points,
        curve.weights,
        unique_knots,
        multiplicities,
        degree,
        is_periodic,
    )


from OCC.Core.GeomAPI import geomapi
from OCC.Core.gp import gp_Pln
from OCC.Core.gp import gp_Pln, gp_Ax3, gp_Ax2, gp_Pnt, gp_Dir, gp_Vec, gp_XYZ

class CCXTransition(TypedDict):
    is_tangent: bool
    position_on_curve: IntRes2d_Position
    transition_type: IntRes2d_TypeTrans

def occ_world_xy():
    plane=((0., 0., 0), (1., 0., 0), (0., 1., 0), (0., 0., 1))
    origin = gp_XYZ(*(plane[0]))
    xaxis = gp_XYZ(*(plane[1]))
    normal = gp_XYZ(*(plane[-1]))
    ax2 = gp_Ax2(gp_Pnt(origin), gp_Dir(normal), gp_Dir(xaxis))
    ax3 = gp_Ax3(ax2)
    pln = gp_Pln(ax3)
    return pln

def occ_plane(plane:np.ndarray=None)->gp_Pln:
    if plane is None:
        return occ_world_xy()
    origin = gp_XYZ(*(plane[0]))
    xaxis = gp_XYZ(*(plane[1]))
    normal = gp_XYZ(*(plane[-1]))
    ax2 = gp_Ax2(gp_Pnt(origin), gp_Dir(normal), gp_Dir(xaxis))
    ax3 = gp_Ax3(ax2)
    pln = gp_Pln(ax3)
    return pln


def convert_trans(trans) -> CCXTransition:
    return CCXTransition(is_tangent=trans.IsTangent(),
                         position_on_curve=trans.PositionOnCurve(),
                         transition_type=trans.TransitionType())


def convert_pnt_location(inter_pt, pln: np.ndarray = None) -> np.ndarray:
    if pln is None:
        pln = np.array(((0., 0., 0), (1., 0., 0), (0., 1., 0), (0., 0., 1)), float)
    pt2d = inter_pt.Value()
    
    return pln[0] + pt2d.X() * pln[1] + pt2d.Y() * pln[2]


def convert_isolated_inter(pt1):
    return {'u': pt1.ParamOnFirst(), 'v': pt1.ParamOnSecond(), 'point': convert_pnt_location(pt1),
            'transition_u': convert_trans(pt1.TransitionOfFirst()), 'transition_v': convert_trans(pt1.TransitionOfSecond())}


def convert_overlap_inter(segm):
    import numpy as np
    uv_path = np.empty((2, 2), dtype=float)
    xyz_path = np.empty((2, 3), dtype=float)
    uv_path[:] = np.nan
    xyz_path[:] = np.nan
    overlap = {'uv_path': uv_path, 'xyz_path': xyz_path, 'start': 'unknown', 'end': 'unknown'}
    if segm.HasFirstPoint():
        overlap['start'] = 'boundary'
        pt = convert_isolated_inter(segm.FirstPoint())
        overlap['uv_path'][0] = (pt['u'], pt['v'])
        overlap['xyz_path'][0] = pt['point']
    if segm.HasLastPoint():
        overlap['end'] = 'boundary'
        pt = convert_isolated_inter(segm.LastPoint())
        overlap['uv_path'][1] = (pt['u'], pt['v'])
        overlap['xyz_path'][1] = pt['point']
    
    return overlap
_WXY = np.array(((0., 0., 0), (1., 0., 0.), (0., 1., 0.), (0., 0., 1.)))
def occ_curve_to_2d(curve:Geom_BSplineCurve,plane:NDArray[float]=None)->Geom2d_Curve:
   
    if plane is None:
        plane=occ_world_xy()
    
    elif isinstance(plane,gp_Pln):
        pass
    else:
       
        
        plane = occ_plane(plane)
    return geomapi.To2d(curve, plane)

def occ_ccx_2d(curve1, curve2, tol=1e-3):
    
    intersector = Geom2dAPI_InterCurveCurve()
    intersector.Init(curve1, curve2, tol)
    inter = intersector.Intersector()
    if not inter.IsDone():
        raise RuntimeError(f"OCC Fail: {inter.IsDone()}")
    points = []
    overlaps = []
    if not inter.IsEmpty():
        for i in range(inter.NbPoints()):
            points.append(convert_isolated_inter(inter.Point(i + 1)))
        for j in range(inter.NbSegments()):
            overlaps.append(convert_overlap_inter(inter.Segment(j + 1)))
    
    return {'isolated': points, 'overlaps': overlaps, 'stats': {}}

#!/usr/bin/env python3
"""
Implementation of the MARCHING METHOD algorithm for tracing the intersection curves
of two NURBS surfaces. This implementation strictly follows the algorithm description
presented in the paper, including both transversal and tangential intersection cases,
the formulation of the ODE system for marching, and a validated ODE solver strategy
(using adaptive step‐size control and event detection). The code integrates with the
existing functions for NURBS surface evaluation and initial intersection point
computation.

Dependencies: Python 3 standard library, numpy, and scipy.

Author: [Your Name]
Date: [Today's Date]
"""
from __future__ import annotations

from typing import Any,Literal

import numpy as np


from numpy import ndarray, dtype

from scipy.spatial import KDTree
from mmcore.numeric.intersection.ssx._ssx_utils import points_equal
from mmcore.geom.curves.curve_bool import unique_with_tolerance
from mmcore.numeric.intersection.ssx._detect_intersections import detect_intersections
from mmcore.numeric.intersection.ssx.boundary_intersection import find_boundary_intersections, IntersectionPoint

from mmcore.geom._nurbs_eval import evaluate_nurbs_surface, NURBSSurfaceTuple, _nurbs_to_tuple, _tuple_to_nurbs
from mmcore.numeric.intersection.ssx.ode import intersection_ode, _DEFAULT_ANGLE_TOL, validated_ode_solver

from mmcore.numeric.intersection.ssx.refine import calculate_eps_n, refine_intersection_point


def find_initial_intersection_points(surf1, surf2, tol=1e-3) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[IntersectionPoint]] | None:
    """
    A robust method that returns at least one point on each of the intersection branches of two NURBS Surfaces.
    :param surf1: First NURBS surface
    :param surf2: Second NURBS surface
    :param tol: Tolerance
    :returns: None if no intersections are found or tuple of three numpy arrays of equal length:
      - cartesian points (x,y,z) with shape (N,3),
      - points in parametric coordinates of the first surface (u,v) with shape (N,2),
      - points in parametric coordinates of the second surface (u,v) with shape (N,2)
    """
    # Placeholder implementation.
    # In production this function must robustly compute initial intersection points.
    # For this implementation, we assume at least one intersection exists.
    # Here, we simply return a dummy intersection at the center of the parametric domain.
    xyz = []
    u1 = []
    u2 = []
    ns1=surf1
    ns2=surf2
    if isinstance(surf1,tuple):
        ns1= _tuple_to_nurbs(surf1)
    if isinstance(surf1,tuple):
        ns2= _tuple_to_nurbs(surf2)
    boundary_intersections= find_boundary_intersections(ns1, ns2, spt=tol)
    for s1, s2 in detect_intersections(ns1, ns2, tol=tol):
        for pt in find_boundary_intersections(s1, s2, spt=tol):

            xyz.append(tuple(pt.point))
            u1.append(pt.surface1_params)
            u2.append(pt.surface2_params)

    if len(xyz) > 0:

        xyz,u1,u2=np.array(xyz), np.array(u1), np.array(u2)
        xyz=unique_with_tolerance(xyz, 1e-12)

        return np.array(xyz),np.array(u1),np.array(u2),boundary_intersections
    else:
        return None


# -------------------------------------------------------------------
# Utility Functions for the Marching Method
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Intersection ODE Function (Formulation of the ODE System)
# -------------------------------------------------------------------


# -------------------------------------------------------------------
# Validated ODE Solver (Robust Marching with Adaptive Step Size)
# -------------------------------------------------------------------


def _intersection_curve_tangents(s1,s2,s,t,u,v):
    d1=evaluate_nurbs_surface(s1, s,t, 1)
    d2=evaluate_nurbs_surface(s2, u,v, 1)
    #d=np.linalg.norm(d1['S']-d2['S'])
    n1 = np.cross(d1['Su'], d1['Sv'])
    n2= np.cross(d2['Su'], d2['Sv'])
    n1/=np.linalg.norm(n1)
    n2 /=np.linalg.norm(n2)
    t1=np.cross(n1,n2)
    t1/=np.linalg.norm(t1)



    return t1,d1['S'],d2['S'],n1,n2


def _distance(p, q):
    """Compute the Euclidean _distance between two points."""

    return np.linalg.norm(p-q)


'''
def _param_dist_edge_wrap(stuv1, stuv2,
                          param_min: np.ndarray,
                          param_max: np.ndarray,
                          spt: float) -> float:
    """
    For each parameter i in 0..3:
      - if |stuv1[i] - stuv2[i]| < spt, dist_i = |…|
      - elif (|stuv1[i]-min_i|<spt and |stuv2[i]-max_i|<spt)
         or (|stuv2[i]-min_i|<spt and |stuv1[i]-max_i|<spt):
           dist_i = 0
      - else dist_i = |stuv1[i] - stuv2[i]|
    Return max_i dist_i.
    """
 
    mins  = param_min
    maxs  = param_max
    delta = np.abs(stuv1 - stuv2)

    # Is a wrap-around match?
    at_min_1 = np.abs(stuv1 - mins) < spt
    at_max_1 = np.abs(stuv1 - maxs) < spt
    at_min_2 = np.abs(stuv2 - mins) < spt
    at_max_2 = np.abs(stuv2 - maxs) < spt

    wrap_match = (at_min_1 & at_max_2) | (at_max_1 & at_min_2)

    # If direct close, keep delta; else if wrap_match then zero; else keep delta
    # So we can just zero out the ones that wrap.
    effective = delta * (~wrap_match)
    

    #_logger.debug(f"_param_dist_edge_wrap deltas={delta}, wrap={wrap_match}, eff={effective}")
    return float(np.max(effective))


def points_equal(p, q,
                 spt:  float,
                 param_tol: float,
                 tan_tol:   float,
                 param_min: np.ndarray,
                 param_max: np.ndarray) -> bool:
    """
    p, q = (xyz 3-vector,
            stuv 4-vector,
            tangent 3-vector)
    param_min, param_max = arrays of length 4 giving the natural [min,max]
      for s,t,u,v from each surface’s clamped knot-span.
    """
    xyz1, stuv1, tan1 = p
    xyz2, stuv2, tan2 = q

    # 1) Cartesian
    cart_d = np.linalg.norm(xyz1 - xyz2)

    # 2) Parametric w/ edge-wrap
    param_d = _param_dist_edge_wrap(stuv1, stuv2,
                                    param_min, param_max,
                                    spt=param_tol)

    # 3) Tangent misalignment
    dot   = float(np.dot(tan1, tan2))
    tan_d = 1.0 - abs(dot)

    #_logger.debug(f"cart_d={cart_d}, param_d={param_d}, tan_d={tan_d} (dot={dot})")

    return (cart_d < spt and
            param_d < param_tol and
            tan_d < tan_tol)

'''
def reverse_polyline(polyline):
    """
    Reverse a polyline structure.

    polyline is a tuple (points, seg_info) where:
      - points is a list of points,
      - seg_info is a list of tuples (segment_index, flip).

    When reversing, the list of points is reversed and each flip flag is inverted.
    """
    points, seg_info = polyline
    rev_points = list(reversed(points))
    rev_seg_info = [(seg_idx, not flip) for seg_idx, flip in reversed(seg_info)]
    return (rev_points, rev_seg_info)


def join_segments_with_info(segments, tol, spt, tan_tol, interval1,interval2):
    """
    Join connected segments into polylines and keep track of segment indices and orientation.

    Parameters:
        segments: list of segments, where each segment is defined as (p1, p2)
                  with p1 and p2 being coordinate tuples (e.g. (x, y)).
        tol: error parameter tolerance. Two points are considered identical if their distance is less than spt.
        spt: error spatial tolerance.
        tan_tol: error tolerance for unit tangent alignment.
    Returns:
        A list of tuples, one per polyline. Each tuple is:
          (polyline_points, segments_info)
        where polyline_points is a list of points defining the polyline, and segments_info is a list
        of (segment_index, flip) tuples in the order that segments appear along the polyline.
        The flip flag is True if the segment was added in reverse.
    """
    # Initialize each segment as its own polyline.
    # Each polyline is a tuple: (points, segments_info)
    # For a segment given as (p1, p2), we use points = [p1, p2] and segments_info = [(index, False)]
    (smin,smax),(tmin,tmax)=interval1
    (umin, umax), (vmin, vmax) = interval2
    #param_min= np.array((smin,tmin,umin,vmin),dtype=float)
    #param_max = np.array((smax, tmax, umax, vmax),dtype=float)
    polylines = [(list(seg), [(idx, False)]) for idx, seg in enumerate(segments)]
    
    changed=True
    while changed:
        changed = False
        i = 0

        while i < len(polylines):
            poly1 = polylines[i]
            points1, seg_info1 = poly1
            
            # If polyline is closed, do not try to merge further.
            if points_equal(tuple(points1[0]), tuple(points1[-1]),  param_tol=tol,spt=spt, tan_tol=tan_tol,s_min=smin,t_min=tmin,u_min=umin,v_min=vmin,s_max=smax,t_max=tmax,u_max=umax,v_max=vmax):
                i += 1
                continue

            j = i + 1
            while j < len(polylines):
                poly2 = polylines[j]
                points2, seg_info2 = poly2
                # Skip poly2 if it is closed.
                if points_equal(tuple(points2[0]), tuple(points2[-1]), param_tol=tol,spt=spt, tan_tol=tan_tol,s_min=smin,t_min=tmin,u_min=umin,v_min=vmin,s_max=smax,t_max=tmax,u_max=umax,v_max=vmax):
                    j += 1
                    continue

                merged = None
                new_points = None
                new_seg_info = None

                # Case 1: End of poly1 equals beginning of poly2.
                if points_equal(tuple(points1[-1]), tuple(points2[0]), param_tol=tol,spt=spt, tan_tol=tan_tol,s_min=smin,t_min=tmin,u_min=umin,v_min=vmin,s_max=smax,t_max=tmax,u_max=umax,v_max=vmax):
                    new_points = points1 + points2[1:]
                    new_seg_info = seg_info1 + seg_info2
                    merged = True
                # Case 2: End of poly1 equals end of poly2 -> reverse poly2.
                elif points_equal(tuple(points1[-1]), tuple(points2[-1]),  param_tol=tol,spt=spt, tan_tol=tan_tol,s_min=smin,t_min=tmin,u_min=umin,v_min=vmin,s_max=smax,t_max=tmax,u_max=umax,v_max=vmax):
                    rev_poly2 = reverse_polyline(poly2)
                    new_points = points1 + rev_poly2[0][1:]
                    new_seg_info = seg_info1 + rev_poly2[1]
                    merged = True
                # Case 3: Beginning of poly1 equals end of poly2.
                elif points_equal(tuple(points1[0]), tuple(points2[-1]), param_tol=tol,spt=spt, tan_tol=tan_tol,s_min=smin,t_min=tmin,u_min=umin,v_min=vmin,s_max=smax,t_max=tmax,u_max=umax,v_max=vmax):
                    new_points = points2 + points1[1:]
                    new_seg_info = seg_info2 + seg_info1
                    merged = True
                # Case 4: Beginning of poly1 equals beginning of poly2 -> reverse poly2.
                elif points_equal(tuple(points1[0]), tuple(points2[0]), param_tol=tol,spt=spt, tan_tol=tan_tol,s_min=smin,t_min=tmin,u_min=umin,v_min=vmin,s_max=smax,t_max=tmax,u_max=umax,v_max=vmax):
                    rev_poly2 = reverse_polyline(poly2)
                    new_points = rev_poly2[0] + points1[1:]
                    new_seg_info = rev_poly2[1] + seg_info1
                    merged = True

                if merged:
                    # Replace poly1 with the merged polyline.
                    polylines[i] = (new_points, new_seg_info)

                    # Remove poly2.
                    polylines.pop(j)
                    changed = True
                    # Break inner loop to restart scanning from the beginning.
                    break
                else:
                    j += 1
            if not changed:
                i += 1
            else:
                # Restart the outer loop if a merge occurred.
                break

    return polylines


def _subd(s1, s2, uv1_start, uv1_end, uv2_start, uv2_end, tol=1e-3,spt=1e-2, recursion_limit=12):


    pt1_start = evaluate_nurbs_surface(s1, *uv1_start, 0)['S']
    pt1_end = evaluate_nurbs_surface(s1, *uv1_end, 0)['S']

    if np.linalg.norm(pt1_start-pt1_end)<(spt):
        return [pt1_start, pt1_end], [uv1_start, uv1_end], [uv2_start, uv2_end]
    if recursion_limit==0:
       
        return [pt1_start, pt1_end], [uv1_start, uv1_end], [uv2_start, uv2_end]
    uv1_mid=(uv1_end+uv1_start)/2
    uv2_mid=(uv2_end + uv2_start)/2



    #du1=uv1_end-uv1_start
    #du2 = uv2_end - uv2_start
    stuv, pt1, pt2, error = refine_intersection_point(np.array([*uv1_mid, *uv2_mid]), s1, s2, spt=spt, max_iter=1000)
    #print(error)


    #x_min, f_min=    golden_section_search(mindist,(0.,1.),1e-3)


    #print([pt1_start.tolist(),pt_mid.tolist(),pt_mid_real.tolist(),pt1_end.tolist()])


    if abs(np.linalg.norm(pt1['S']-((pt1_start+pt1_end)*0.5)))<(spt):
        return [pt1_start ,pt1_end],[uv1_start, uv1_end],[uv2_start,uv2_end]

    else:
        #print(x_min,f_min)
        #uv1_mid = du1 * x_min + uv1_start
        #uv2_mid = du2 * x_min + uv2_start
        uv1_mid=stuv[:2]
        uv2_mid = stuv[2:]
        pts_l,uvs1_l,uvs2_l=_subd(s1, s2, uv1_start,uv1_mid, uv2_start, uv2_mid, spt, recursion_limit=recursion_limit-1)
        pts_r,uvs1_r,uvs2_r=_subd(s1, s2, uv1_mid, uv1_end, uv2_mid, uv2_end, spt, recursion_limit=recursion_limit-1)
        return pts_l+pts_r[1:], uvs1_l+uvs1_r[1:], uvs2_l+uvs2_r[1:]


import math
import numpy as np


def _project_point_to_segment(p, a, b):
    """
    Projects point p onto the line defined by segment endpoints a and b in 3D.

    Parameters:
        p (tuple or list): The point to project, as (x, y, z).
        a (tuple or list): The first endpoint of the segment, as (x, y, z).
        b (tuple or list): The second endpoint of the segment, as (x, y, z).

    Returns:
        distance (float): The Euclidean distance between p and its projection.
        is_on_segment (bool): True if the projection lies within the segment [a, b],
                              False if it lies outside.
    """
    # Compute vector AB
    abx = b[0] - a[0]
    aby = b[1] - a[1]
    abz = b[2] - a[2]

    # Compute vector AP
    apx = p[0] - a[0]
    apy = p[1] - a[1]
    apz = p[2] - a[2]

    # Compute squared length of AB
    ab_squared = abx * abx + aby * aby + abz * abz

    # Handle degenerate segment (a and b are the same point)
    if ab_squared == 0:
        distance = math.sqrt(apx * apx + apy * apy + apz * apz)
        return distance, False

    # Compute the projection parameter t of p onto AB
    t = (apx * abx + apy * aby + apz * abz) / ab_squared

    # Compute the projected point coordinates on the line
    proj_x = a[0] + t * abx
    proj_y = a[1] + t * aby
    proj_z = a[2] + t * abz

    # Calculate the distance from p to its projection
    dx = p[0] - proj_x
    dy = p[1] - proj_y
    dz = p[2] - proj_z
    distance = math.sqrt(dx * dx + dy * dy + dz * dz)

    # Check if the projection lies within the segment [a, b]
    is_on_segment = (0.0 <= t <= 1.0)

    return distance, is_on_segment
from mmcore.numeric.intersection.ssx.boundary_intersection import IntersectionPoint


def _expand_interval(interv,val):
    return interv[0]-val,interv[1]+val


# -------------------------------------------------------------------
# Marching Method: Tracing the Intersection Curve
# -------------------------------------------------------------------

def trace_intersection_curve(
        surf1: NURBSSurfaceTuple,
        surf2: NURBSSurfaceTuple,
        init_params: np.ndarray,
        boundary_intersections:list[IntersectionPoint],
        s_max: float = 1.0,
        h_initial: float = 0.1,
        tol: float = 1e-6,
        spt: float = 1e-3,

        context:dict|None=None,
        boundary_check_spt=True,         angle_tol:float= _DEFAULT_ANGLE_TOL,
        eps_n=None

) -> tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], ndarray[Any, dtype[Any]], list[ndarray], int]:
    """
    Trace the intersection curve of two NURBS surfaces using the marching method.
    This function integrates the ODE system defined by the intersection_ode function, employs
    a validated ODE solver with adaptive step sizing, and integrates an advanced point refinement
    loop at each accepted step to ensure the computed intersection points lie within the Same Point
    Tolerance (SPT) of the true intersection.

    Parameters:
        surf1      : First NURBS surface.
        surf2      : Second NURBS surface.
        init_params: Initial parameter vector [sigma, t, u, v] on the intersection curve.
        s_max      : Total integration parameter (arc length) to trace.
        h_initial  : Initial step size for the ODE solver.
        tol        : Tolerance for error control and ODE validation.
        spt        : Same Point Tolerance for point refinement.
     
    Returns:
        A tuple containing:
         - curve_points: An (N,3) numpy array of points in 3D model space along the intersection.
         - params_surf1: An (N,2) numpy array of parametric coordinates on surf1.
         - params_surf2: An (N,2) numpy array of parametric coordinates on surf2.
         - enclosures : A list of interval enclosures (each a 2x4 numpy array) in the parametric space.
         - termination_reason: int
        
    """


    states, enclosures,term_reason = validated_ode_solver(f=intersection_ode, x0=init_params, surf1=surf1, surf2=surf2,
                                                          s_max=s_max, h_initial=h_initial, tol=tol, spt=spt,
                                                          boundary_intersections=boundary_intersections,
                                                          context=context, boundary_check_spt=boundary_check_spt, eps_n=eps_n, angle_tol=angle_tol)

    curve_points = []
    params_surf1 = []
    params_surf2 = []
    for state in states:
        sigma, t, u, v = state
        pt = evaluate_nurbs_surface(surf1, sigma, t, d_order=0)
        curve_points.append(pt["S"])
        params_surf1.append([sigma, t])
        params_surf2.append([u, v])

    return (np.array(curve_points),
            np.array(params_surf1),
            np.array(params_surf2),
            enclosures,term_reason)


# -------------------------------------------------------------------
# Example Usage (for production, integrate with robust initial point finder)
# -------------------------------------------------------------------

def trace_intersection_curves(surf1:NURBSSurfaceTuple, surf2:NURBSSurfaceTuple, init_points, init_uv1, init_uv2, boundary_intersections, h_initial=0.1, tol=1e-3, spt=1e-3, backward=True, boundary_check_spt=True, angle_tol:float= _DEFAULT_ANGLE_TOL,
                              eps_n=None):


        branches=[]


        initial_xs = np.zeros((init_uv1.shape[0], 4))
        initial_xs[..., :2] = init_uv1
        initial_xs[..., 2:] = init_uv2
        initial_points_tree = KDTree(initial_xs)
        #print(len(res))
        context = dict(init_points_tree=initial_points_tree)
        #boundary_intersections: list[IntersectionPoint]=find_boundary_intersections(surf1,surf2, spt=spt)
        i=0
        while   context['init_points_tree'] is not None:

            i+=1

            if isinstance(context['init_points_tree'],KDTree):

                init_point=context['init_points_tree'].data[0]
                pts = np.delete(context['init_points_tree'].data, 0, axis=0).reshape((-1,4))

                if pts.size == 0:

                    context["init_points_tree"] = None
                elif len(pts) == 1:


                    context["init_points_tree"] = pts
                else:
                    context["init_points_tree"] = KDTree(pts)
            elif isinstance(context['init_points_tree'], np.ndarray):
                init_point=context['init_points_tree'][0]

                context["init_points_tree"] = None

            else:
                break
            # Trace the intersection curve.

            #init_point_xyz = evaluate_nurbs_surface(surf1, init_point[0], init_point[1], 0)['S']

            curve_pts, params1, params2, encl, termination_reason = trace_intersection_curve(surf1, surf2, init_point,
                                                                                             boundary_intersections,
                                                                                             s_max=-1,
                                                                                             h_initial=h_initial,
                                                                                             tol=tol, spt=spt,
                                                                                             context=context,angle_tol=angle_tol,eps_n=eps_n,
                                                                                             boundary_check_spt=boundary_check_spt)
            if (termination_reason == 1):
                #print("CC",len(curve_pts),np.array(curve_pts).tolist(), [b.point.tolist()for b in boundary_intersections])
                ...




            if (termination_reason == 1) and backward:
                #init_point_xyz = evaluate_nurbs_surface(surf1, init_point[0], init_point[1], 0)['S']
                #print(1, init_point_xyz.tolist())

                curve_pts2, params12, params22, encl2, termination_reason2 = trace_intersection_curve(surf1, surf2,
                                                                                                      init_point,
                                                                                                      boundary_intersections,
                                                                                                      s_max=-1,
                                                                                                      h_initial=-h_initial,
                                                                                                      tol=tol, spt=spt,
                                                                                                      context=context,angle_tol=angle_tol,eps_n=eps_n,
                                                                                                      boundary_check_spt=boundary_check_spt)


                curve_pts, params1, params2, encl=np.array([*reversed(curve_pts2),*curve_pts[1:]]),np.array([*reversed(params12), *params1[1:]]),np.array([*reversed(params22),*params2[1:]]),[*reversed(encl2),encl[1:]]


            #print("END", context,i)


            #print(context)

            branches.append((curve_pts, params1, params2, encl))

        return branches
SamplingMethod=Literal['marching','subdivide']


def _compare_robust(a, b, tol):
    min_val=a-tol
    max_val=a+tol
    return (min_val<=b) and (b<=max_val)
import logging
_logger=logging.getLogger('mmcore')
def _nurbs_trace_intersection_curves_v2(surf1, surf2, spt=1e-3, tol=1e-7, tan_tol=1e-3, angle_tol:float= _DEFAULT_ANGLE_TOL, **kwargs) :
    """
    A robust method that returns at least one point on each of the intersection branches of two NURBS Surfaces.
    :param surf1: First NURBS surface
    :param surf2: Second NURBS surface
    :param tol: Tolerance
    :returns: None if no intersections are found or tuple of three numpy arrays of equal length:
      - cartesian points (x,y,z) with shape (N,3),
      - points in parametric coordinates of the first surface (u,v) with shape (N,2),
      - points in parametric coordinates of the second surface (u,v) with shape (N,2)
    """

    ns1=surf1
    ns2=surf2
    if isinstance(surf1,tuple):
        ns1= _tuple_to_nurbs(surf1)
    if isinstance(surf1,tuple):
        ns2= _tuple_to_nurbs(surf2)
    eps_n= calculate_eps_n(spt, angle_tol)
    branches=[]
    items=detect_intersections(ns1, ns2, spt=spt, tol=tol)
    _logger.debug('detected intersections: %s', items)
    beams=[]
    (smin,smax),(tmin,tmax)=s1_interval=ns1.interval()
    (umin, umax), (vmin, vmax) = s2_interval=ns2.interval()
    isolated_points=[]
    for i,(s1, s2) in enumerate(items):

        stars_points,overlaps= find_boundary_intersections(s1, s2, spt=spt)
        #for o in overlaps:
        #    branches.append((np.array([o._segm.start(),o._segm.end()]), np.array(o.get_ends(s1)),     np.array(o.get_ends(s2))))
        _logger.debug(f"finded boundary_intersections {i}, len starts points: {len(stars_points)}")
        # print('p',[p.point.tolist() for p in stars_points])

        if len(stars_points)==0:
            _logger.debug(f"pass")
            continue
        elif len(stars_points)==1:

            # Здесь мы должны проверить что точка находится на границе одной из исходных поверхностей.
            # Так как предполагается что сингулярные точки мы проверили ранее,
            # любое пересечение дающее одну точку не на границе одной из исходных поверхностей, является вырожденной
            # и связана с тем что соседнии подпатчи случайно совпали, оставлять такю точку нет никакого смысла.
            s,t,u,v=stars_points[0].stuv
            if (_compare_robust(smin, s, tol=tol) or _compare_robust(smax, s, tol=tol)
             or _compare_robust(tmin, t, tol=tol) or _compare_robust(tmax, t, tol=tol)
             or _compare_robust(umin, u, tol=tol) or _compare_robust(umax, u, tol=tol)
             or _compare_robust(vmin, v, tol=tol) or _compare_robust(vmax, v, tol=tol)) :
                # Точка находится на границе исходных поверхностей и если ни -900
                # одна ветвь не придет в нее, то мы вернем ее как изолированную точку.
                isolated_points.append(stars_points[0])
                _logger.debug(f"isolated")
            else:
                _logger.debug(f"not isolated: {stars_points[0].point.tolist()}")
            continue
        # OLDVALUE: elif len(stars_points)>2: (Кажется, что в текущей реализации marching ведет себя всегда лучше подгонки. В дальнейшем, можно рассмотреть более умную диспетчеризацию. Например, использовать подгонку на участках с малой кривизной, или большим углом между нормалями.
        elif len(stars_points)>2:
            print(_nurbs_to_tuple(s1))
            print(_nurbs_to_tuple(s2))
            if (len(stars_points)%2)==1:

                _logger.critical(f"hard case: {[s.point.tolist() for s in stars_points]}. An odd number of boundary intersection points")
                print()
                continue
            _logger.debug(f"many branches: {[s.point.tolist() for s in stars_points]}")
    
            bnd_points = list(stars_points[1:])

            st1 = _nurbs_to_tuple(s1)
            st2 = _nurbs_to_tuple(s2)
            l=len(stars_points)
            init_points=np.zeros((l,3))
            init_uv1 = np.zeros((l, 2))
            init_uv2 = np.zeros((l, 2))

            for i in range(l):
                pt=stars_points[i]
                # print('ss',pt.point.tolist())
                init_points[i]=pt.point
                init_uv1[i]=pt.surface1_params
                init_uv2[i]=pt.surface2_params
            min_uvd=np.inf
            for uv1 in init_uv1:
                for uv2 in init_uv1:
                    uvd=np.min(uv1-uv2)
                    if uvd<min_uvd:
                        min_uvd=uvd
            for uv1 in init_uv2:
                for uv2 in init_uv2:
                    uvd=np.min(uv1-uv2)
                    if uvd<min_uvd:
                        min_uvd=uvd

            for (curve_pts, params1, params2, encl) in trace_intersection_curves( st1, st2,init_points,init_uv1,init_uv2,boundary_intersections=bnd_points, tol=tol,spt=spt,h_initial= tol,backward=False,boundary_check_spt=False, eps_n=eps_n,angle_tol=angle_tol):
                curve_pts, params1, params2=np.array(curve_pts), np.array(params1), np.array(params2)
                branches.append((curve_pts,
                                 params1,
                                 params2)
                                )

                tangent_start, *_ = _intersection_curve_tangents(st1, st2, params1[0][0],params1[0][1],params2[0][0],params2[0][1])
                tangent_end, *_ = _intersection_curve_tangents(st1, st2, params1[-1][0], params1[-1][1], params2[-1][0],
                                                                 params2[-1][1])

                beams.append((
                    [ curve_pts[0], np.array([params1[0][0],params1[0][1],params2[0][0],params2[0][1]]),tangent_start],
                    [curve_pts[-1], np.array([params1[-1][0], params1[-1][1], params2[-1][0], params2[-1][1]]),
                     tangent_end]))

        else:
            _logger.debug(f"base case: {[s.point.tolist() for s in stars_points]}")
            pt_start=stars_points[0]
            pt_end = stars_points[-1]
            st1 = _nurbs_to_tuple(s1)
            st2 = _nurbs_to_tuple(s2)

            tangent_start,*_=_intersection_curve_tangents(st1,st2,*pt_start.surface1_params,*pt_start.surface2_params)
            tangent_end,*_=_intersection_curve_tangents(st1,st2,*pt_end.surface1_params,*pt_end.surface2_params)
            beams.append(
                ([pt_start.point, np.array([pt_start.surface1_params[0],pt_start.surface1_params[1],pt_start.surface2_params[0],pt_start.surface2_params[1]]),tangent_start],[pt_end.point, np.array([pt_end.surface1_params[0],pt_end.surface1_params[1],  pt_end.surface2_params[0],  pt_end.surface2_params[1]]), tangent_end]))
            # print([pt_start.point.tolist(),pt_end.point.tolist()])

            curve_points,params_surf1,params_surf2=_subd(st1, st2, np.array(pt_start.surface1_params), np.array(pt_end.surface1_params), np.array(pt_start.surface2_params), np.array(pt_end.surface2_params),tol=tol, spt=spt)

            branches.append((np.array(curve_points),
                    np.array(params_surf1),
                    np.array(params_surf2))
                    )
    _logger.debug(f"end tracing")

    branches_joined=[]
    ppp=join_segments_with_info(beams, tol=tol, spt=spt, tan_tol=tan_tol, interval1=s1_interval,interval2=s2_interval)

    isolated_points_indexes_to_delete = set()
    for pts, seg_info in ppp:
        branch_pt=[]
        branch_uv1=[]
        branch_uv2=[]
        for segm,flip in seg_info:
            pts,uv1,uv2=branches[segm]
            if flip:
                pts,uv1,uv2=list(reversed(pts)),list(reversed(uv1)),list(reversed(uv2))

            if len(branch_pt)==0:
                branch_pt.extend(pts)
                branch_uv1.extend(uv1)
                branch_uv2.extend(uv2)
            else:
                branch_pt.extend(pts[1:])
                branch_uv1.extend(uv1[1:])
                branch_uv2.extend(uv2[1:])
        branches_joined.append((branch_pt, branch_uv1, branch_uv2))
        ds,dt=branch_uv1[0] - branch_uv1[-1]
        du,dv = branch_uv2[0] - branch_uv2[-1]

        if np.linalg.norm([ds,dt,du,dv])<tol :
            pass
        else:
            pt_start=np.array(branch_pt[0])
            pt_end=np.array(branch_pt[-1])
            stuv_start=np.array([branch_uv1[0][0], branch_uv1[0][1], branch_uv2[0][0], branch_uv2[0][1]])
            stuv_end=np.array([branch_uv1[-1][0], branch_uv1[-1][1], branch_uv2[-1][0], branch_uv2[-1][1]])
            for i,pt in enumerate(isolated_points):
            
                if (np.linalg.norm(stuv_start-pt.stuv)<tol or np.linalg.norm(pt_start-pt.point)<spt):
                    isolated_points_indexes_to_delete.add(i)

                elif (np.linalg.norm(stuv_end - pt.stuv) < tol)or np.linalg.norm(pt_end-pt.point)<spt:

                    isolated_points_indexes_to_delete.add(i)

    # isolated_points_arr=np.array(isolated_points,dtype=object)

    real_isolated_points:list[IntersectionPoint]=(np.array(isolated_points, dtype=IntersectionPoint)[
        list(set(range(len(isolated_points))) - isolated_points_indexes_to_delete)]).tolist()

    return branches_joined, real_isolated_points

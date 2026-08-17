from __future__ import annotations

from typing import List

import numpy as np

from mmcore.nurbs._core import NURBSSurface, NURBSCurve, find_span, basis_functions

from mmcore.nurbs._nurbs_eval import  NURBSCurveTuple,NURBSSurfaceTuple,_surface_interval,to_homogeneous_2d,from_homogeneous_2d,to_homogeneous_1d,from_homogeneous_1d


def extract_surface_boundaries(surface: NURBSSurface|NURBSSurfaceTuple) -> List[NURBSCurve]|List[NURBSCurveTuple]:

    """
    Extract the four boundary curves of a NURBS surface.

    Args:
        surface (NURBSSurface): The input NURBS surface

    Returns:
        List[NURBSCurve]: List of four boundary curves in order:
            [u=0 curve, u=1 curve, v=0 curve, v=1 curve]
    """
    if isinstance(surface, NURBSSurfaceTuple):
        return extract_surface_boundaries_tuple(surface)

    (u_min, u_max), (v_min, v_max) = surface.interval()

    # Extract iso-curves at the boundaries


    u0_curve = extract_isocurve(surface, u_min, 'u')  # v-direction curve at u=0
    u1_curve = extract_isocurve(surface, u_max, 'u')  # v-direction curve at u=1
    v0_curve = extract_isocurve(surface, v_min, 'v')  # u-direction curve at v=0
    v1_curve = extract_isocurve(surface, v_max, 'v')  # u-direction curve at v=1

    return [u0_curve, u1_curve, v0_curve, v1_curve]
from mmcore.nurbs._nurbs_eval import _nurbs_to_tuple,_tuple_to_nurbs

def extract_isocurve(
        surface: NURBSSurface|NURBSSurfaceTuple, param: float, direction: str = "u"
) -> NURBSCurve|NURBSCurveTuple:
    """
    Extract an isocurve from a NURBS surface at a given parameter in the u or v direction.
    Args:
    surface (NURBSSurface): The input NURBS surface.
    param (float): The parameter value at which to extract the isocurve.
    direction (str): The direction of the isocurve, either 'u' or 'v'. Default is 'u'.
    Returns:
    NURBSCurve: The extracted isocurve as a NURBS curve.
    Raises:
    ValueError: If the direction is not 'u' or 'v', or if the param is out of range.
    """
    if isinstance(surface, NURBSSurfaceTuple):
        return _extract_isocurve_tuple(surface, param, direction)
    if direction not in ["u", "v"]:
        raise ValueError("Direction must be either 'u' or 'v'.")
    st = _nurbs_to_tuple(surface)
    ct=_extract_isocurve_tuple(st,param,direction)
    return _tuple_to_nurbs(ct)
    interval = surface.interval()
    # print('ij', surface.knots_u, surface.knots_v, surface.interval())
    if direction == "u":
        # For u-direction: we fix u and vary v
        # First check if the u parameter is in range
        param_range = interval[0]  # u range
        if param < param_range[0] or param > param_range[1]:
            raise ValueError(f"Parameter {param} is out of range {param_range}")

        # Find the span and basis functions in u direction (the direction we're fixing)
        n_u = surface.shape[0] - 1  # number of control points in u direction - 1
        degree_u = surface.degree[0]
        span = find_span(n_u, degree_u, param, surface.knots_u, 0)
        basis = basis_functions(span, param, degree_u, surface.knots_u)

        # The resulting curve will have as many control points as the surface has in v direction
        m = surface.shape[1]
        control_points = np.zeros((m, 4))

        # Compute control points for the extracted curve
        for i in range(m):  # iterate over v direction
            for j in range(degree_u + 1):  # combine with basis functions
                control_points[i] += basis[j] * surface.control_points_w[span - degree_u + j, i]

            # Return curve with v-direction degree and knots since we're varying in v

        cc=NURBSCurve(control_points, degree=surface.degree[1],knots=surface.knots_v)
        # cc.knots=surface.knots_v

        # print('j', cc.knots,cc.interval())
        return cc

    else:  # direction == 'v'
        # For v-direction: we fix v and vary u
        # First check if the v parameter is in range
        param_range = interval[1]  # v range
        if param < param_range[0] or param > param_range[1]:
            raise ValueError(f"Parameter {param} is out of range {param_range}")

        # Find the span and basis functions in v direction (the direction we're fixing)
        n_v = surface.shape[1] - 1  # number of control points in v direction - 1
        degree_v = surface.degree[1]
        span = find_span(n_v, degree_v, param, surface.knots_v, 0)
        basis = basis_functions(span, param, degree_v, surface.knots_v)

        # The resulting curve will have as many control points as the surface has in u direction
        m = surface.shape[0]
        control_points = np.zeros((m, 4))

        # Compute control points for the extracted curve
        for i in range(m):  # iterate over u direction
            for j in range(degree_v + 1):  # combine with basis functions
                control_points[i] += basis[j] * surface.control_points_w[i, span - degree_v + j]
        cc = NURBSCurve(control_points, surface.degree[0],surface.knots_u)
        # print('i',cc.knots,cc.interval())
        # cc.knots = surface.knots_u
        # Return curve with u-direction degree and knots since we're varying in u
        return cc


def _extract_isocurve_tuple(
        surface: NURBSSurfaceTuple, param: float, direction: str = "u"
) -> NURBSCurveTuple:
    """
    Extract an isocurve from a NURBS surface at a given parameter in the u or v direction.
    Args:
    surface (NURBSSurfaceTuple): The input NURBS surface.
    param (float): The parameter value at which to extract the isocurve.
    direction (str): The direction of the isocurve, either 'u' or 'v'. Default is 'u'.
    Returns:
    NURBSCurveTuple: The extracted isocurve as a NURBS curve.
    Raises:
    ValueError: If the direction is not 'u' or 'v', or if the param is out of range.
    """
    if direction not in ["u", "v"]:
        raise ValueError("Direction must be either 'u' or 'v'.")
    interval = _surface_interval(surface)
    # print('ij', surface.knots_u, surface.knots_v, surface.interval())
    if direction == "u":
        # For u-direction: we fix u and vary v
        # First check if the u parameter is in range
        param_range = interval[0]  # u range
        if param < param_range[0] or param > param_range[1]:
            raise ValueError(f"Parameter {param} is out of range {param_range}")

        # Find the span and basis functions in u direction (the direction we're fixing)
        n_u = surface.control_points.shape[0] - 1  # number of control points in u direction - 1
        degree_u = surface.order_u-1
        span = find_span(n_u, degree_u, param, np.array(surface.knot_u,dtype=float), 0)
        basis = basis_functions(span, param, degree_u,  np.array(surface.knot_u,dtype=float))

        # The resulting curve will have as many control points as the surface has in v direction
        m = surface.control_points.shape[1]
        new_control_points_w = np.zeros((m, surface.control_points.shape[-1]+1))
        control_points_w=to_homogeneous_2d(surface.control_points,surface.weights)
        # Compute control points for the extracted curve
        for i in range(m):  # iterate over v direction
            for j in range(degree_u + 1):  # combine with basis functions

                new_control_points_w[i] += basis[j] * control_points_w[span - degree_u + j, i]

            # Return curve with v-direction degree and knots since we're varying in v

        cc=NURBSCurveTuple(surface.order_v, surface.knot_v,*from_homogeneous_1d(new_control_points_w))

        # cc.knots=surface.knots_v

        # print('j', cc.knots,cc.interval())
        return cc

    else:  # direction == 'v'
        # For v-direction: we fix v and vary u
        # First check if the v parameter is in range
        param_range = interval[1]  # v range
        if param < param_range[0] or param > param_range[1]:
            raise ValueError(f"Parameter {param} is out of range {param_range}")

        # Find the span and basis functions in v direction (the direction we're fixing)
        n_v = surface.control_points.shape[1] - 1  # number of control points in v direction - 1
        degree_v = surface.order_v-1
        span = find_span(n_v, degree_v, param, np.array(surface.knot_v), 0)
        basis = basis_functions(span, param, degree_v,np.array( surface.knot_v))

        # The resulting curve will have as many control points as the surface has in u direction
        m = surface.control_points.shape[0]

        new_control_points_w = np.zeros((m, surface.control_points.shape[-1] + 1))
        control_points_w = to_homogeneous_2d(surface.control_points, surface.weights)
        # Compute control points for the extracted curve
        for i in range(m):  # iterate over u direction
            for j in range(degree_v + 1):  # combine with basis functions
                new_control_points_w[i] += basis[j] * control_points_w[i, span - degree_v + j]
        cc = NURBSCurveTuple(surface.order_u,surface.knot_u, *from_homogeneous_1d(new_control_points_w))
        # print('i',cc.knots,cc.interval())
        # cc.knots = surface.knots_u
        # Return curve with u-direction degree and knots since we're varying in u
        return cc

def extract_surface_boundaries_tuple(surface: NURBSSurfaceTuple) -> List[NURBSCurveTuple]:
    """
    Extract the four boundary curves of a NURBS surface.

    Args:
        surface (NURBSSurface): The input NURBS surface

    Returns:
        List[NURBSCurve]: List of four boundary curves in order:
            [u=0 curve, u=1 curve, v=0 curve, v=1 curve]
    """
    (u_min, u_max), (v_min, v_max) = _surface_interval(surface)

    # Extract iso-curves at the boundaries


    u0_curve = _extract_isocurve_tuple(surface, u_min, 'u')  # v-direction curve at u=0
    u1_curve = _extract_isocurve_tuple(surface, u_max, 'u')  # v-direction curve at u=1
    v0_curve = _extract_isocurve_tuple(surface, v_min, 'v')  # u-direction curve at v=0
    v1_curve = _extract_isocurve_tuple(surface, v_max, 'v')  # u-direction curve at v=1

    return [u0_curve, u1_curve, v0_curve, v1_curve]

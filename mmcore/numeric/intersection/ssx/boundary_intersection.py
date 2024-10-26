"""
This module implements boundary intersection detection for NURBS surfaces.
It provides functionality to extract surface boundaries and find their intersections
with another surface.
"""
from __future__ import annotations

from typing import List, Tuple, Optional
import numpy as np

from mmcore.geom.nurbs import NURBSSurface, NURBSCurve, find_span, basis_functions

from mmcore.numeric.algorithms.surface_area import v_max
from mmcore.numeric.intersection.csx import nurbs_csx
from tests.test_nurbs_algo import degree


def extract_surface_boundaries(surface: NURBSSurface) -> List[NURBSCurve]:
    """
    Extract the four boundary curves of a NURBS surface.
    
    Args:
        surface (NURBSSurface): The input NURBS surface
        
    Returns:
        List[NURBSCurve]: List of four boundary curves in order:
            [u=0 curve, u=1 curve, v=0 curve, v=1 curve]
    """
    (u_min, u_max), (v_min, v_max) = surface.interval()
    
    # Extract iso-curves at the boundaries


    u0_curve = extract_isocurve(surface, u_min, 'u')  # v-direction curve at u=0
    u1_curve = extract_isocurve(surface, u_max, 'u')  # v-direction curve at u=1
    v0_curve = extract_isocurve(surface, v_min, 'v')  # u-direction curve at v=0
    v1_curve = extract_isocurve(surface, v_max, 'v')  # u-direction curve at v=1
    
    return [u0_curve, u1_curve, v0_curve, v1_curve]

class IntersectionPoint:
    """Class representing an intersection point between a boundary curve and a surface"""
    
    def __init__(self, 
                point: np.ndarray,
                curve_param: float,
                surface_params: Tuple[float, float],
                boundary_index: int,
                is_from_first_surface: bool, interval):
        """
        Initialize an intersection point.
        
        Args:
            point (np.ndarray): 3D intersection point
            curve_param (float): Parameter value on the boundary curve
            surface_params (Tuple[float, float]): (u,v) parameters on the intersected surface
            boundary_index (int): Index of the boundary curve (0-3)
            is_from_first_surface (bool): True if the boundary is from the first surface
        """
        self.point = point
        self.curve_param = curve_param
        self.boundary_index = boundary_index
        self.is_from_first_surface = is_from_first_surface
        self.umin, self.umax = interval[0]
        self.vmin, self.vmax = interval[1]
        # Store parameters for both surfaces
        if is_from_first_surface:
            self.surface1_params = self._boundary_index_to_params(boundary_index, curve_param,self.umin,self.umax,self.vmin,self.vmax)
            self.surface2_params = surface_params
        else:
            self.surface1_params = surface_params
            self.surface2_params = self._boundary_index_to_params(boundary_index, curve_param, self.umin,self.umax,self.vmin,self.vmax)
            
        # Keep this for backward compatibility
        self.surface_params = surface_params
        
    def get_start_params(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get the parameter values on both surfaces for starting curve tracing.
        
        Returns:
            Tuple[np.ndarray, np.ndarray]: 
                (array([u1,v1]), array([u2,v2])) parameters for both surfaces
        """
        if self.is_from_first_surface:
            # For first surface, convert boundary index to fixed parameter
            u1, v1 = self._boundary_index_to_params(self.boundary_index, self.curve_param,self.umin,self.umax,self.vmin,self.vmax)
            # For second surface, use the found parameters
            u2, v2 = self.surface_params
        else:
            # For second surface, convert boundary index to fixed parameter
            u2, v2 =  self._boundary_index_to_params(self.boundary_index, self.curve_param,self.umin,self.umax,self.vmin,self.vmax)
            # For first surface, use the found parameters
            u1, v1 = self.surface_params
            
        return (np.array([u1, v1]), np.array([u2, v2]))
    
    @staticmethod
    def _boundary_index_to_params(boundary_index: int, param: float, umin,umax,vmin,vmax) -> Tuple[float, float]:
        """Convert boundary index and curve parameter to surface parameters"""
        if boundary_index == 0:  # u=0 curve
            return ( umin, param)
        elif boundary_index == 1:  # u=1 curve
            return (umax, param)
        elif boundary_index == 2:  # v=0 curve
            return (param, vmin)
        else:  # v=1 curve
            return (param, vmax)

def find_boundary_intersections(surf1: NURBSSurface, 
                              surf2: NURBSSurface, 
                              tol: float = 1e-6) -> List[IntersectionPoint]:
    """
    Find all intersection points between the boundaries of two NURBS surfaces.
    
    Args:
        surf1 (NURBSSurface): First NURBS surface
        surf2 (NURBSSurface): Second NURBS surface
        tol (float): Tolerance for intersection detection
        
    Returns:
        List[IntersectionPoint]: List of found intersection points
    """
    intersection_points = []
    
    # Get boundaries of both surfaces
    boundaries1 = extract_surface_boundaries(surf1)
    boundaries2 = extract_surface_boundaries(surf2)
    
    # Find intersections of surf1's boundaries with surf2
    for i, boundary in enumerate(boundaries1):
        intersections = nurbs_csx(boundary, surf2, tol=tol)

        for intersection_type, point, params in intersections:
            # params[0] is curve parameter, params[1:] are surface parameters
            intersection_points.append(
                IntersectionPoint(
                    point=point,
                    curve_param=params[0],
                    surface_params=tuple(params[1:]),
                    boundary_index=i,
                    is_from_first_surface=True,
                    interval=surf1.interval()
                )
            )
    
    # Find intersections of surf2's boundaries with surf1
    for i, boundary in enumerate(boundaries2):
        intersections = nurbs_csx(boundary, surf1, tol=tol)
        for intersection_type, point, params in intersections:
            intersection_points.append(
                IntersectionPoint(
                    point=point,
                    curve_param=params[0],
                    surface_params=tuple(params[1:]),
                    boundary_index=i,
                    is_from_first_surface=False,
                    interval=surf2.interval(),
                )
            )
    
    # Remove duplicate points (within tolerance)
    unique_points = []
    for point in intersection_points:
        is_duplicate = False
        for existing_point in unique_points:
            if np.linalg.norm(point.point - existing_point.point) < tol:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_points.append(point)
    
    return unique_points

def sort_boundary_intersections(points: List[IntersectionPoint]) -> List[List[IntersectionPoint]]:
    """
    Sort boundary intersection points into connected sequences that form intersection curves.
    
    Args:
        points (List[IntersectionPoint]): List of boundary intersection points
        
    Returns:
        List[List[IntersectionPoint]]: List of point sequences, each representing 
            the endpoints of an intersection curve
    """
    if not points:
        return []
        
    # Start with all points in unassigned set
    unassigned = set(range(len(points)))
    sequences = []
    
    while unassigned:
        # Start a new sequence with the first unassigned point
        current_sequence = []
        start_idx = unassigned.pop()
        current_sequence.append(points[start_idx])
        
        # Try to find the next closest point
        while True:
            current_point = current_sequence[-1]
            nearest_idx = None
            min_distance = float('inf')
            
            # Find the closest unassigned point
            for idx in unassigned:
                candidate = points[idx]
                distance = np.linalg.norm(current_point.point - candidate.point)
                
                if distance < min_distance:
                    min_distance = distance
                    nearest_idx = idx
            
            # If no close point found or sequence has 2 points, end sequence
            if nearest_idx is None or len(current_sequence) == 2:
                break
                
            # Add the nearest point to sequence and remove from unassigned
            current_sequence.append(points[nearest_idx])
            unassigned.remove(nearest_idx)
        
        sequences.append(current_sequence)
    
    return sequences


def extract_isocurve(
        surface: NURBSSurface, param: float, direction: str = "u"
) -> NURBSCurve:
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
    if direction not in ["u", "v"]:
        raise ValueError("Direction must be either 'u' or 'v'.")
    interval = surface.interval()
    #print('ij', surface.knots_u, surface.knots_v, surface.interval())
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
        #cc.knots=surface.knots_v

        #print('j', cc.knots,cc.interval())
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
        #print('i',cc.knots,cc.interval())
        #cc.knots = surface.knots_u


        # Return curve with u-direction degree and knots since we're varying in u
        return cc

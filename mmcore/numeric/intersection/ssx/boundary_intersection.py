"""
This module implements boundary intersection detection for NURBS surfaces.
It provides functionality to extract surface boundaries and find their intersections
with another surface.
"""
from __future__ import annotations


from typing import List, Tuple
import numpy as np

from mmcore.geom.nurbs import NURBSSurface

from mmcore.geom.nurbs_iso import extract_surface_boundaries
from mmcore.numeric.intersection.csx import nurbs_csx


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
        self.stuv=np.array([self.surface1_params[0], self.surface1_params[1],self.surface2_params[0],self.surface2_params[1] ])
        
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

def find_boundary_intersections(surf1: NURBSSurface, surf2: NURBSSurface, spt: float = 1e-6, tol=1e-6) -> List[IntersectionPoint]:
    """
    Find all intersection points between the boundaries of two NURBS surfaces.
    
    Args:
        surf1 (NURBSSurface): First NURBS surface
        surf2 (NURBSSurface): Second NURBS surface
        spt (float): Spatial tolerance
        tol (float): Parameter tolerance
        
    Returns:
        List[IntersectionPoint]: List of found intersection points

    """
    intersection_points = []
    
    # Get boundaries of both surfaces
    boundaries1 = extract_surface_boundaries(surf1)
    boundaries2 = extract_surface_boundaries(surf2)
    #print(boundaries1)
    #print(boundaries2)
    #print([boundary.control_points.tolist() for boundary in boundaries1])
    #print([boundary.control_points.tolist() for boundary in boundaries2])
    # Find intersections of surf1's boundaries with surf2
    for i, boundary in enumerate(boundaries1):
        intersections = nurbs_csx(boundary, surf2, tol=spt, ptol=tol)
        #intersections =int_cs(boundary,surf2,tol=tol,spt=spt)
        #print(i,intersections,boundary.control_points.tolist())
        for intersection_type, point, params in intersections:
            # params[0] is curve parameter, params[1:] are surface parameters
            if intersection_type == 'degenerate':
                print(intersection_type)
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
        intersections = nurbs_csx(boundary, surf1, tol=spt, ptol=tol)
        #intersections = int_cs(boundary, surf1, tol=tol, spt=spt)
        #print(i, intersections, boundary.control_points.tolist())
        for intersection_type, point, params in intersections:
            if intersection_type =='degenerate':
                print(intersection_type)
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
            N=np.linalg.norm(point.stuv - existing_point.stuv)
            #print("N",N)


            if  N < tol:
                is_duplicate = True
                break
            ...
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



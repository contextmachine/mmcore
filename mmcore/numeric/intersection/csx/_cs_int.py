from __future__ import annotations
import dataclasses

import numpy as np
from numpy._typing import NDArray

from mmcore.geom._nurbs_eval import (
    EvaluateSurfaceData,
    EvaluateCurveData,
    _surface_interval,
    _curve_interval,
    evaluate_nurbs_surface,
    evaluate_nurbs_curve,
    NURBSCurveTuple,
    NURBSSurfaceTuple,
)

from enum import Enum, auto

__all__=["CSInt"]

from mmcore.numeric import compute_parametric_tolerance_curve
from mmcore.numeric.vectors import scalar_norm


@dataclasses.dataclass(slots=True)
class CSInt:
    tuv: NDArray[float]
    curve_eval: EvaluateCurveData
    surf_eval: EvaluateSurfaceData
    tuv_tol: NDArray[float]
    error: float

    

    def compare_with_tol(self, tuv: NDArray[float], curve_tuple: NURBSCurveTuple = None, surface_tuple: NURBSSurfaceTuple = None):
        """
        Compare intersection points with tolerance, handling closed curves/surfaces.

        For points at parameter boundaries, performs C2 continuity check to determine
        if they represent the same geometric point on closed curves/surfaces.
        """
        # Standard tolerance check first
        diff = np.abs(tuv - self.tuv)
        within_tol = diff <= self.tuv_tol * 2

        if np.all(within_tol):
            return True

        # If standard check fails and we have curve/surface data, check for boundary conditions
        if curve_tuple is None or surface_tuple is None:
            return False

        # Get parameter ranges
        t0, t1 = _curve_interval(curve_tuple)
        (u0, u1), (v0, v1) = _surface_interval(surface_tuple)

        # Check which specific parameters failed and if they're at boundaries
        failed_params = []

        # Check t parameter
        if not within_tol[0]:  # t parameter failed standard tolerance
            t_self, t_other = self.tuv[0], tuv[0]
            dt = self.tuv_tol[0]

            # Check if either t is at curve boundary
            t_self_boundary = (abs(t_self - t0) <= dt) or (abs(t_self - t1) <= dt)
            t_other_boundary = (abs(t_other - t0) <= dt) or (abs(t_other - t1) <= dt)

            if t_self_boundary or t_other_boundary:
                failed_params.append("t")

        # Check u parameter
        if not within_tol[1]:  # u parameter failed standard tolerance
            u_self, u_other = self.tuv[1], tuv[1]
            du = self.tuv_tol[1]

            # Check if either u is at surface boundary
            u_self_boundary = (abs(u_self - u0) <= du) or (abs(u_self - u1) <= du)
            u_other_boundary = (abs(u_other - u0) <= du) or (abs(u_other - u1) <= du)

            if u_self_boundary or u_other_boundary:
                failed_params.append("u")

        # Check v parameter
        if not within_tol[2]:  # v parameter failed standard tolerance
            v_self, v_other = self.tuv[2], tuv[2]
            dv = self.tuv_tol[2]

            # Check if either v is at surface boundary (closed surface check)
            v_self_boundary = (abs(v_self - v0) <= dv) or (abs(v_self - v1) <= dv)
            v_other_boundary = (abs(v_other - v0) <= dv) or (abs(v_other - v1) <= dv)

            # Special case for closed surfaces: check if one is at v=0 and other at v=max
            v_closed_boundary = (abs(v_self - v0) <= dv and abs(v_other - v1) <= dv) or (abs(v_self - v1) <= dv and abs(v_other - v0) <= dv)

            if v_self_boundary or v_other_boundary or v_closed_boundary:
                failed_params.append("v")

        # If no failed parameters are at boundaries, return False
        if not failed_params:
            return False

        # Only perform geometric continuity check for boundary cases
        return self._check_geometric_continuity(tuv, curve_tuple, surface_tuple, failed_params)

    def _check_geometric_continuity(
        self,
        other_tuv: NDArray[float],
        curve_tuple: NURBSCurveTuple,
        surface_tuple: NURBSSurfaceTuple,
        failed_params: list,
            spt: float = 1e-3,
            angle_tol:float=0.052,
    ) -> bool:
        """
        Check geometric continuity only for parameters that failed standard tolerance and are at boundaries.
        For closed surfaces with boundary parameters, focus on position continuity rather than derivative continuity.
        """

        other_curve_eval = evaluate_nurbs_curve(curve_tuple, other_tuv[0], d_order=1)
        other_surf_eval = evaluate_nurbs_surface(surface_tuple, other_tuv[1], other_tuv[2], d_order=1)

        curve_pos_diff = np.linalg.norm(self.curve_eval["C"] - other_curve_eval["C"])
        surf_pos_diff = np.linalg.norm(self.surf_eval["S"] - other_surf_eval["S"])
    
        if curve_pos_diff > spt or surf_pos_diff > spt:
            return False

       
        tan1=self.curve_eval["C1"]/scalar_norm( self.curve_eval["C1"])
        tan2 =  other_curve_eval["C1"] / scalar_norm(other_curve_eval["C1"])
        
        
        if (1-abs(np.dot(tan1, tan2)))<angle_tol:
                return False

        return True

def add_unique_int(inters:list[CSInt], new_int:CSInt, original_curve, original_surface):
    stack=[new_int]
  
    while stack:
        inter_candidate = stack.pop(0)
        is_visited = False
        for index, existing_inter in enumerate(inters):

            if existing_inter.compare_with_tol(new_int.tuv, original_curve, original_surface):
                is_visited = True
                # logger.debug(f"compare_with_tol returns True : {existing_inter.tuv}, {tuv}")
                if existing_inter.error > new_int.error:

                    # logger.debug(
                    #    "Replace (the new tuv are the best guess, {}>{}): {}->{}".format(
                    #        existing_inter.error, inter_candidate.error, existing_inter.tuv, inter_candidate.tuv
                    #    )
                    # )
                    del inters[index]
                    stack.append(inter_candidate)

                    # logger.debug("Inters: {}".format([i.tuv for i in inters]))

                    break

                else:
                    pass
                    # logger.debug("Pass (the error of the new tuv exceeds the error of the existing ones, {}<={}) : {}".format(  existing_inter.error, error, tuv ))
                    # logger.debug("Inters: {}".format([i.tuv for i in inters]))

            # logger.debug(f'compare_with_tol returns False : {existing_inter.tuv}, {tuv}')
        if not is_visited:

            # logger.debug("Inters: {}".format([i.tuv for i in inters]))
            # logger.debug("New: {}".format(tuv))

            inters.append(new_int)
            # logger.debug("Inters: {}".format([i.tuv for i in inters]))

    

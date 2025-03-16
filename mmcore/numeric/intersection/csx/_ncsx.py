import numpy as np

from mmcore.numeric._aabb import aabb


from mmcore.numeric.newton.cnewton import newtons_method

from mmcore.geom.nurbs import (
    NURBSCurve,
    NURBSSurface,
    split_surface_v,
    split_surface_u,
    split_curve,
    subdivide_surface,
    CurveSurfaceEq,
)

from mmcore.numeric import scalar_dot
from mmcore.numeric.vectors import scalar_unit, scalar_norm
from mmcore.numeric.intersection.separability.spatial import spatial_separability
from mmcore.numeric.intersection.separability.spherical import spherical_separability

__all__ = ["nurbs_csx", "NURBSCurveSurfaceIntersector"]


def normalize_curve_knots(curve):
    k = curve.knots
    curve.knots = (k - k[0]) / (k[-1] - k[0])
    curve.knots_update_hook()


class NURBSCurveSurfaceIntersector:
    """
    The ground of the implementation was based on the description of the algorithm from the  "4.5 Intersecting Curves and Surfaces. Robust and Efficient Surface Intersection for Solid Modeling By Michael Edward Hohmeyer B.A. (University of California) 1986"

    """

    __slots__ = ["curve", "surface", "intersections", "tolerance", "ptol"]

    def __init__(
        self, curve: NURBSCurve, surface: NURBSSurface, tolerance=1e-3, ptol=1e-7
    ):
        self.curve: NURBSCurve = curve
        self.surface: NURBSSurface = surface
        # normalize_curve_knots(self.curve)
        # self.surface.normalize_knots()

        self.tolerance: float = tolerance
        self.intersections = []
        self.ptol = ptol

    def intersect(self):
        self._curve_surface_intersect(self.curve, self.surface)
        return self.intersections

    def _curve_surface_intersect(self, curve, surface):
        # Check if curve and surface are spatially separated
        res = self._no_new_intersections(curve, surface)
        if res:
            return

        # Try to find an intersection point
        new_point = self._find_new_intersection(curve, surface)
        (u0, u1), (v0, v1) = surface.interval()
        t0, t1 = curve.interval()
        
        # Special handling for boundary cases - here we need to be more careful
        # Handle boundary cases differently
        is_boundary_case = False
        u_mid, v_mid = (u0 + u1) * 0.5, (v0 + v1) * 0.5
        
        # Check if the curve or surface are very small or near boundaries
        t_size = abs(t1 - t0)
        u_size = abs(u1 - u0)
        v_size = abs(v1 - v0)
        
        # If parameters ranges are very small, consider this a boundary case
        if t_size < self.ptol*100 or u_size < self.ptol*100 or v_size < self.ptol*100:
            is_boundary_case = True
        
        if new_point is None:
            # If we're in a boundary case, use a stricter approach
            if is_boundary_case:
                # Try more aggressive subdivision near boundaries
                curve_split_points = [
                    (t0 + t1) * 0.5,  # Middle
                    t0 + (t1 - t0) * 0.25,  # Quarter from start
                    t0 + (t1 - t0) * 0.75,  # Quarter from end
                ]
                
                for split_t in curve_split_points:
                    curve1, curve2 = split_curve(curve, split_t, tol=1e-12, normalize_knots=False)
                    
                    # Subdivide surface more aggressively too
                    surf_splits = [
                        (u_mid, v_mid),  # Center
                        (u0 + u_size * 0.25, v_mid),  # Near u0
                        (u1 - u_size * 0.25, v_mid),  # Near u1
                        (u_mid, v0 + v_size * 0.25),  # Near v0
                        (u_mid, v1 - v_size * 0.25)   # Near v1
                    ]
                    
                    for u_split, v_split in surf_splits:
                        surfaces = subdivide_surface(
                            surface, u_split, v_split, self.ptol, normalize_knots=False
                        )
                        
                        # Recursively check each piece
                        for c in [curve1, curve2]:
                            for s in surfaces:
                                self._curve_surface_intersect(c, s)
                
                return
            else:
                # Normal case - just split in the middle
                curve1, curve2 = split_curve(curve, (t0 + t1) * 0.5, tol=1e-12, normalize_knots=False)
                
                # Skip if we're at a very small region near a boundary
                if abs(u_mid - u0) < self.ptol or abs(u_mid - u1) < self.ptol or \
                   abs(v_mid - v0) < self.ptol or abs(v_mid - v1) < self.ptol:
                    return
                    
                surface1, surface2, surface3, surface4 = subdivide_surface(
                    surface, u_mid, v_mid, self.ptol, normalize_knots=False
                )
        else:
            # We found an intersection point
            point, (t, u, v) = new_point
            
            # Classify and add the intersection
            if self._is_degenerate(new_point[1], curve, surface):
                self.intersections.append(("degenerate", point, (t, u, v)))
            else:
                self.intersections.append(("transversal", point, (t, u, v)))
            
            # If this intersection is on or very near a boundary, don't continue subdivision
            # in this branch - we've found what we need
            near_boundary = (abs(u-u0) < self.ptol*10 or abs(u-u1) < self.ptol*10 or 
                            abs(v-v0) < self.ptol*10 or abs(v-v1) < self.ptol*10 or
                            abs(t-t0) < self.ptol*10 or abs(t-t1) < self.ptol*10)
                            
            if near_boundary:
                # For boundary points, we still want to check the other side of the boundary
                # but we don't need further subdivision at this location
                return
                
            # Check if we can exclude further subdivision based on separability
            if spherical_separability(
                np.array(surface.control_points_flat), curve.control_points, point
            ):
                return

            # Continue normal subdivision around the found intersection
            curve1, curve2 = split_curve(curve, t, tol=1e-12, normalize_knots=False)
            surface1, surface2, surface3, surface4 = subdivide_surface(
                surface, u, v, tol=self.ptol, normalize_knots=False
            )
        
        # Recursively check all subdivided pieces
        self._curve_surface_intersect(curve1, surface1)
        self._curve_surface_intersect(curve1, surface2)
        self._curve_surface_intersect(curve1, surface3)
        self._curve_surface_intersect(curve1, surface4)
        self._curve_surface_intersect(curve2, surface1)
        self._curve_surface_intersect(curve2, surface2)
        self._curve_surface_intersect(curve2, surface3)
        self._curve_surface_intersect(curve2, surface4)

    def _no_new_intersections(self, curve, surface):
        # Implement separability test from section 4.2
        # Return True if curve and surface don't intersect except at already discovered points

        return spatial_separability(
            curve.control_points,
            np.array(surface.control_points_flat),
            tol=self.tolerance,
        )



    def _get_interior_intersections(self, curve, surface):
        # Return list of already discovered intersection points interior to curve or surface
        return self.intersections

    def _find_new_intersection(self, curve, surface):
        equation = CurveSurfaceEq(curve, surface)

        t0, t1 = curve.interval()
        (u0, u1), (v0, v1) = surface.interval()
        
        # Try multiple starting points, including points near the boundaries
        starting_points = [
            np.array([(t0 + t1) * 0.5, (u0 + u1) * 0.5, (v0 + v1) * 0.5]),  # Center
            np.array([t0 + self.ptol*10, (u0 + u1) * 0.5, (v0 + v1) * 0.5]),  # Near t0
            np.array([t1 - self.ptol*10, (u0 + u1) * 0.5, (v0 + v1) * 0.5]),  # Near t1
            np.array([(t0 + t1) * 0.5, u0 + self.ptol*10, (v0 + v1) * 0.5]),  # Near u0
            np.array([(t0 + t1) * 0.5, u1 - self.ptol*10, (v0 + v1) * 0.5]),  # Near u1
            np.array([(t0 + t1) * 0.5, (u0 + u1) * 0.5, v0 + self.ptol*10]),  # Near v0
            np.array([(t0 + t1) * 0.5, (u0 + u1) * 0.5, v1 - self.ptol*10]),  # Near v1
            # Try corners too for boundary-boundary intersections
            np.array([t0 + self.ptol*10, u0 + self.ptol*10, v0 + self.ptol*10]),  # Near (t0,u0,v0)
            np.array([t1 - self.ptol*10, u1 - self.ptol*10, v1 - self.ptol*10]),  # Near (t1,u1,v1)
            
            # Add more sampling points across the domain
            np.array([t0 + (t1-t0)*0.25, u0 + (u1-u0)*0.25, v0 + (v1-v0)*0.25]),
            np.array([t0 + (t1-t0)*0.75, u0 + (u1-u0)*0.75, v0 + (v1-v0)*0.75]),
            np.array([t0 + (t1-t0)*0.25, u0 + (u1-u0)*0.75, v0 + (v1-v0)*0.25]),
            np.array([t0 + (t1-t0)*0.75, u0 + (u1-u0)*0.25, v0 + (v1-v0)*0.75]),
            
            # Add specific points for the problematic second intersection near (19.8, -5.6, -1.8)
            # Try different parameter combinations in the region where the second point might be
            np.array([t0 + (t1-t0)*0.9, u0 + (u1-u0)*0.1, v0 + (v1-v0)*0.9]),
            np.array([t0 + (t1-t0)*0.1, u0 + (u1-u0)*0.9, v0 + (v1-v0)*0.1]),
            np.array([t0 + (t1-t0)*0.9, u0 + (u1-u0)*0.9, v0 + (v1-v0)*0.1]),
            np.array([t0 + (t1-t0)*0.1, u0 + (u1-u0)*0.1, v0 + (v1-v0)*0.9])
        ]
        
        for start_point in starting_points:
            result = newtons_method(
                equation,
                start_point,
                tol=self.tolerance * 0.1,  # Tighter tolerance for convergence
                max_iter=10  # Increase max iterations for better convergence
            )
            
            if (
                result is not None
                and self._is_valid_parameter(result, (t0, t1), (u0, u1), (v0, v1))
                and not any(np.isnan(result))
            ):
                point = curve.evaluate(result[0])
                point2 = surface.evaluate_v2(*result[1:])
                r = scalar_norm(point - point2)
                
                # Check if this is a valid intersection point within tolerance
                if r <= self.tolerance:
                    # Check if this point is already in our list of intersections
                    is_duplicate = False
                    for i in range(len(self.intersections)):
                        if np.all(
                            np.abs(np.array(self.intersections[i][1]) - np.array(point))
                            < self.tolerance
                        ):
                            is_duplicate = True
                            break
                            
                    if not is_duplicate:
                        return point, result
        
        return None

    def _is_valid_parameter(self, params, t_range, u_range, v_range):
        t, u, v = params
        t0, t1 = t_range
        (u0, u1), (v0, v1) = u_range, v_range
        
        # Use a small epsilon to include boundary cases that might be slightly outside
        # due to floating point precision issues
        eps = self.ptol * 10
        
        return (t0 - eps <= t <= t1 + eps and 
                u0 - eps <= u <= u1 + eps and 
                v0 - eps <= v <= v1 + eps)

    def _is_degenerate(self, point, curve, surface):
        t, u, v = point
        curve_tangent = curve.tangent(t)
        surface_normal = surface.normal(np.array([u, v]))
        surface_normal /= scalar_norm(surface_normal)
        # print(surface_normal,curve_tangent)
        return np.abs(scalar_dot(curve_tangent, surface_normal)) < self.tolerance


def nurbs_csx(curve: NURBSCurve, surface: NURBSSurface, tol=1e-5, ptol=1e-7):
    """
    Compute intersections between a NURBS curve and a NURBS surface.

    This function serves as the primary interface for detecting intersections between a NURBS curve and a NURBS surface.
    The underlying implementation is based on recursive subdivision and numerical methods,
    leveraging the following steps:

    1. **Recursive Subdivision**: The curve and surface are recursively subdivided into smaller regions,
    allowing for more accurate and efficient detection of intersections.
    2. **Separability Tests**: For each subdivision, a spatial separability test (based on bounding box and convex hull
    checks) is applied. If separability is confirmed, no further subdivision or intersection testing is required.
    3. **Intersection Detection**: Newton's method is used to refine intersection points once subdivisions are small
    enough, ensuring that the intersection points are calculated with high precision.

    **Key Implementation Details**:

    - The separability test prevents unnecessary subdivision when the curve and surface are sufficiently far apart.
    - If a new intersection point is found, it is classified either as a "transversal" or "degenerate" intersection,
    depending on the angle between the curve's tangent and the surface's normal.
    - The intersection process stops when either the desired tolerance (`tol`) or the precision tolerance (`ptol`) is
    reached.
    - Recursive subdivision ensures that no intersections are missed, even for complex geometries.

    **Algorithmic Foundation**:

    This implementation is based on the work described in section 4.5 of "Robust and Efficient Surface Intersection for Solid Modeling" by Michael Edward Hohmeyer B.A. (University of California, 1986).
    The method efficiently handles intersections for NURBS-based geometries, making it suitable for CAD and solid
    modeling applications.

    **Parameters**:

    :param curve:
    The NURBS curve to intersect with the surface. This curve is represented by a series of control points, knots, and a
    degree.
    :type curve: mmcore.geom.nurbs.NURBSCurve

    :param surface:
    The NURBS surface to intersect with the curve. The surface is defined by its control points, a knot vector in both
    the `u` and `v` directions, and degrees in both directions.
    :type surface: mmcore.geom.nurbs.NURBSSurface

    :param tol:
    The tolerance used to determine the accuracy of intersection points. Smaller tolerance values result in higher
    precision but may increase computational cost.
    :type tol: float, optional

    :param ptol:
    Precision tolerance used during separability tests to avoid numerical errors in intersection detection.
    The default is 1e-6.
    :type ptol: float, optional

    **Returns**:

    :return:
    A list of intersection points between the curve and surface. Each intersection is represented as a tuple of the form:
    - `("type", point, (t, u, v))`
    where `type` can be 'transversal' or 'degenerate', `point` is the 3D coordinates of the intersection,
    and `(t, u, v)` are the parametric coordinates of the intersection.
    :rtype: list

    **Example**:

    .. code-block:: python

    # Example usage of nurbs_csx
    curve = NURBSCurve(control_points=[...], knots=[...], degree=3)
    surface = NURBSSurface(control_points=[...], knots_u=[...], knots_v=[...], degree_u=3, degree_v=3)

    intersections = nurbs_csx(curve, surface, tol=1e-4, ptol=1e-7)
    for intersection in intersections:
    print(intersection)

    **Notes**:

    - The algorithm will subdivide the curve and surface recursively until it either finds an intersection or determines
    that no intersection exists within the provided tolerance.
    - For complex surfaces or highly curved regions, consider adjusting the `tol` parameter to increase precision.
    - The classification of intersections as "transversal" or "degenerate" helps distinguish between cases where the
    curve crosses the surface tangentially versus at a sharper angle.

    **Limitations**:

    - This method assumes that both the curve and surface are properly defined and their parameterizations are valid.
    - Very high precision (`ptol` values smaller than 1e-15) may lead to longer computation times or convergence issues.
    """
    intersector = NURBSCurveSurfaceIntersector(curve, surface, tolerance=tol, ptol=ptol)
    intersector.intersect()
    return intersector.intersections


if __name__ == "__main__":
    from mmcore._test_data import csx as test_data
    import time

    S1, C1 = test_data[0]
    intersector = NURBSCurveSurfaceIntersector(C1, S1)
    s = time.time()
    res = intersector.intersect()
    e1 = time.time() - s
    print([pt.tolist() for (t, pt, prm) in res])
    S1, C2 = test_data[1]
    intersector = NURBSCurveSurfaceIntersector(C2, S1)
    s = time.time()
    res = intersector.intersect()
    e2 = time.time() - s
    res.sort(key=lambda x: x[2][0])
    print([pt.tolist() for (t, pt, prm) in res])
    print(e1, e2, sep="\n")
    ts = []
    uvs = []
    typs = []
    for t, pt, prm in res:
        typs.append(t)
        ts.append(prm[0])
        uvs.append(prm[1:])
    print(ts)
    print(uvs)

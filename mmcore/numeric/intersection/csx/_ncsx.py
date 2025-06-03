import numpy as np

from mmcore.numeric._aabb import aabb, aabb_intersection,aabb_intersect_fast_3d

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

from mmcore.numeric.log_scaling import to_log,from_log
def normalize_curve_knots(curve):
    k = curve.knots
    curve.knots = (k - k[0]) / (k[-1] - k[0])
    curve.knots_update_hook()

def _surf_to_log_space(surf:NURBSSurface)->NURBSSurface:

    cptsw = np.array(surf.control_points_w)
    cpts = to_log(cptsw[..., :-1])
    cptsw[..., :-1] = cpts

    return NURBSSurface(cpts, degree=tuple(surf.degree), knots_u=surf.knots_u,knots_v=surf.knots_v)


def _curve_to_log_space(curve:NURBSCurve):

    cptsw=np.array(curve.control_pointsw)
    cpts=to_log(cptsw[..., :-1])
    cptsw[..., :-1]=cpts

    return NURBSCurve(cpts,degree=curve.degree,knots=curve.knots)


class NURBSCurveSurfaceIntersector:
    """
    The ground of the implementation was based on the description of the algorithm from the  "4.5 Intersecting Curves and Surfaces. Robust and Efficient Surface Intersection for Solid Modeling By Michael Edward Hohmeyer B.A. (University of California) 1986"

    """

    __slots__ = ["curve", "surface", 'initial_curve', 'initial_surface',"intersections", "tolerance", "ptol",'_equation']

    def __init__(
        self, curve: NURBSCurve, surface: NURBSSurface, tolerance=1e-3, ptol=1e-7
    ):

        self.initial_curve: NURBSCurve=curve
        self.initial_surface : NURBSSurface= surface

        self.curve: NURBSCurve = self.initial_curve
        self.surface: NURBSSurface =self.initial_surface

        # normalize_curve_knots(self.curve)
        # self.surface.normalize_knots()

        self.tolerance: float = tolerance
        self.intersections = []
        self.ptol = ptol
        self._equation=CurveSurfaceEq(self.initial_curve, self.initial_surface)
    def intersect(self):
        self._curve_surface_intersect(self.curve, self.surface)
        return self.intersections

    def _curve_surface_intersect(self, curve, surface):
        # print(self.intersections)

        res = self._no_new_intersections(curve, surface)

        # print(np.array(curve.interval()),np.array(curve.knots))
        # print(res)

        if res:
            return

        # interior_intersections = self._get_interior_intersections(curve, surface)

        new_point = self._find_new_intersection(curve, surface)
        (u0, u1), (v0, v1) = surface.interval()
        if new_point is None:
            t0, t1 = curve.interval()

            dt=abs(t1-t0)/2
            if dt<self.ptol:
                return
            curve1, curve2 = split_curve(curve, (t0 + t1) * 0.5,tol=1e-12, normalize_knots=False)
            # normalize_curve_knots(curve1)
            # normalize_curve_knots(curve2)
            u,v=(u0 + u1) * 0.5, (v0 + v1) * 0.5

            if abs(u - u0) < self.ptol or abs(u - u1) < self.ptol or abs(v - v0) < self.ptol or abs(v - v1) < self.ptol:

                return
            surface1, surface2, surface3, surface4 = subdivide_surface(
                surface, (u0 + u1) * 0.5, (v0 + v1) * 0.5, self.ptol, normalize_knots=False
            )

        else:
            point, (t, u, v) = new_point

            if self._is_degenerate(new_point[1], curve, surface):

                self.intersections.append(("degenerate", point, (t, u, v)))

            else:
                self.intersections.append(("transversal", point, (t, u, v)))
            if abs(u-u0)<self.ptol or abs(u-u1)<self.ptol or  abs(v-v0)<self.ptol or  abs(v-v1)<self.ptol:

                return
            if spherical_separability(
                np.array(surface.control_points_flat), curve.control_points, point
            ):
                return

            curve1, curve2 = split_curve(curve, t,tol=1e-12,normalize_knots=False)
            # normalize_curve_knots(curve1)
            # normalize_curve_knots(curve2)

            surface1, surface2, surface3, surface4  = subdivide_surface(surface, u, v, tol=1e-12, normalize_knots=False)

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

        #
        bb1=np.array(curve.bbox())
        bb2=np.array(surface.bbox())
        bb1[0]-=self.tolerance
        bb1[1] += self.tolerance
        bb2[0] -= self.tolerance
        bb2[1] += self.tolerance
        if not aabb_intersect_fast_3d(bb1,bb2):
            return

        # equation = CurveSurfaceEq(curve, surface)
        t0, t1 = curve.interval()
        (u0, u1), (v0, v1) = surface.interval()

        result = newtons_method(
            self._equation,
            np.array([(t0 + t1) * 0.5, (u0 + u1) * 0.5, (v0 + v1) * 0.5]),
            max_iter=5
        )

        # print(result)
        if (
            result is not None
            and self._is_valid_parameter(result, (t0, t1), (u0, u1), (v0, v1))
            and not any(np.isnan(result))
        ):
            # point = curve.evaluate(result[0])
            # point2 = surface.evaluate_v2(*result[1:])

            result = np.asarray(result)
            # point = self.initial_curve.evaluate(result[0])
            # point2 = self.initial_surface.evaluate_v2(result[1],result[2])
            result = newtons_method(
                self._equation,
                result

            )
            if result is None or np.any(np.isnan(result)) :
                return
            r=self._equation(result)**0.5

            if r <= self.tolerance and not self._is_degenerate(result, curve, surface):

                for i in range(len(self.intersections)):
                    tuv=np.asarray(self.intersections[i][2])
                    if np.all(
                                (result-tuv)<self.ptol) :

                        return

                point = self.initial_curve.evaluate(result[0])
                return point, result

        return None

    def _is_valid_parameter(self, params, t_range, u_range, v_range):
        t, u, v = params
        t0, t1 = t_range
        (u0, u1), (v0, v1) = u_range, v_range

        return t0 <= t <= t1 and u0 <= u <= u1 and v0 <= v <= v1

    def _is_degenerate(self, point, curve, surface):
        t, u, v = point
        curve_tangent = curve.tangent(t)
        surface_normal = np.cross(surface.derivative_u(np.array([u, v])),
        surface.derivative_v(np.array([u, v])))
        
        surface_normal=surface_normal / (np.linalg.norm(surface_normal)+1e-6)
        
        # print(surface_normal,curve_tangent)
        return np.abs(np.dot(curve_tangent, surface_normal)) < 0.001

from ._ncsx2 import int_cs
def nurbs_csx(curve: NURBSCurve, surface: NURBSSurface, tol=1e-3, ptol=1e-6):
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

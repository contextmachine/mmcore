import numpy as np

from mmcore.numeric._aabb import aabb, aabb_intersection, aabb_intersect_fast_3d

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
from mmcore.numeric.numeric import (
    compute_parametric_curvature_tolerance_curve,
    compute_parametric_curvature_tolerance_surface,
)

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

    __slots__ = [
        "curve",
        "surface",
        "initial_curve",
        "initial_surface",
        "intersections",
        "tolerance",
        "angle_tol",
        "_equation",
    ]

    def __init__(
        self, curve: NURBSCurve, surface: NURBSSurface, tolerance=1e-3, *, angle_tol: float = 0.0013
    ):

        self.initial_curve: NURBSCurve=curve
        self.initial_surface : NURBSSurface= surface

        self.curve: NURBSCurve = self.initial_curve
        self.surface: NURBSSurface =self.initial_surface

        # normalize_curve_knots(self.curve)
        # self.surface.normalize_knots()

        self.tolerance: float = tolerance
        self.angle_tol: float = angle_tol
        self.intersections = []
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

        # Flatness-based early stopping: if both the curve segment and the
        # surface subpatch are locally flat enough w.r.t. `self.tolerance`,
        # classify once and avoid further subdivision (prevents degenerate
        # splitting on overlaps and near-flat regions).
        if self._is_flat_cell(curve, surface):
            self._classify_flat_cell(curve, surface)
            return

        new_point = self._find_new_intersection(curve, surface)
        (u0, u1), (v0, v1) = surface.interval()
        if new_point is None:
            t0, t1 = curve.interval()

            # No explicit param tolerance: rely on flatness stop above
            # to terminate recursion in nearly-flat/narrow cells.
            curve1, curve2 = split_curve(
                curve, (t0 + t1) * 0.5, tol=1e-12, normalize_knots=False
            )
            # normalize_curve_knots(curve1)
            # normalize_curve_knots(curve2)
            u, v = (u0 + u1) * 0.5, (v0 + v1) * 0.5

            # Guard: avoid splitting exactly at boundaries
            eps = 1e-12
            if (
                abs(u - u0) <= eps or abs(u - u1) <= eps or
                abs(v - v0) <= eps or abs(v - v1) <= eps
            ):

                return
            surface1, surface2, surface3, surface4 = subdivide_surface(
                surface,
                (u0 + u1) * 0.5,
                (v0 + v1) * 0.5,
                1e-12,
                normalize_knots=False,
            )

        else:
            point, (t, u, v) = new_point

            if self._is_degenerate(new_point[1], curve, surface):
                self._insert_with_param_dedup(curve, surface, ("degenerate", point, (t, u, v)))
            else:
                self._insert_with_param_dedup(curve, surface, ("transversal", point, (t, u, v)))
            eps = 1e-12
            if (
                abs(u - u0) <= eps or abs(u - u1) <= eps or
                abs(v - v0) <= eps or abs(v - v1) <= eps
            ):

                return
            if spherical_separability(
                np.array(surface.control_points_flat), curve.control_points, point
            ):
                return

            curve1, curve2 = split_curve(
                curve, t, tol=1e-12, normalize_knots=False
            )
            # normalize_curve_knots(curve1)
            # normalize_curve_knots(curve2)

            surface1, surface2, surface3, surface4 = subdivide_surface(
                surface, u, v, tol=1e-12, normalize_knots=False
            )

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
        bb1 = np.array(curve.bbox())
        bb2 = np.array(surface.bbox())
        bb1[0] -= self.tolerance
        bb1[1] += self.tolerance
        bb2[0] -= self.tolerance
        bb2[1] += self.tolerance
        if not aabb_intersect_fast_3d(bb1, bb2):
            return

        # equation = CurveSurfaceEq(curve, surface)
        t0, t1 = curve.interval()
        (u0, u1), (v0, v1) = surface.interval()

        result = newtons_method(
            self._equation,
            np.array([(t0 + t1) * 0.5, (u0 + u1) * 0.5, (v0 + v1) * 0.5]),
            max_iter=5,
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
            result = newtons_method(self._equation, result)
            if result is None or np.any(np.isnan(result)):
                return
            r = self._equation(result) ** 0.5

            if r <= self.tolerance and not self._is_degenerate(result, curve, surface):
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
        surface_normal = np.cross(
            surface.derivative_u(np.array([u, v])),
            surface.derivative_v(np.array([u, v])),
        )

        nrm = np.linalg.norm(surface_normal)
        surface_normal = surface_normal / (nrm + 1e-12)

        # print(surface_normal,curve_tangent)
        return np.abs(np.dot(curve_tangent, surface_normal)) < np.sin(self.angle_tol)

    def _parametric_steps(self, curve: NURBSCurve, surface: NURBSSurface, tuv):
        t, u, v = tuv
        # Curve
        C1 = curve.derivative(t)
        C2 = curve.second_derivative(t)
        dt = compute_parametric_curvature_tolerance_curve(C1, C2, self.tolerance)
        # Surface
        Su = surface.derivative_u(np.array([u, v]))
        Sv = surface.derivative_v(np.array([u, v]))
        Suu = surface.second_derivative_uu(np.array([u, v]))
        Svv = surface.second_derivative_vv(np.array([u, v]))
        du, dv = compute_parametric_curvature_tolerance_surface(Su, Sv, Suu, Svv, self.tolerance)
        # Guard non-finite
        if not np.isfinite(dt):
            dt = np.inf
        if not np.isfinite(du):
            du = np.inf
        if not np.isfinite(dv):
            dv = np.inf
        return float(dt), float(du), float(dv)

    def _mismatch_norm(self, tuv):
        t, u, v = tuv
        pc = np.asarray(self.initial_curve.evaluate(t))
        ps = np.asarray(self.initial_surface.evaluate_v2(u, v))
        return float(np.linalg.norm(pc - ps))

    def _insert_with_param_dedup(self, curve: NURBSCurve, surface: NURBSSurface, rec):
        """Insert `rec` into `self.intersections`, deduplicating by local
        curvature-based parametric steps. If a duplicate is found, keep the
        one with smaller ||C(t) - S(u,v)||.

        rec format: (type_str, point_xyz, (t,u,v))
        """
        r_type, r_point, r_tuv = rec
        dt_r, du_r, dv_r = self._parametric_steps(curve, surface, r_tuv)
        best_idx = None

        for i, ex in enumerate(self.intersections):
            _, _, e_tuv = ex
            dt_e, du_e, dv_e = self._parametric_steps(curve, surface, e_tuv)
            t0, u0, v0 = r_tuv
            t1, u1, v1 = e_tuv
            if (
                abs(t0 - t1) <= min(dt_r, dt_e)
                and abs(u0 - u1) <= min(du_r, du_e)
                and abs(v0 - v1) <= min(dv_r, dv_e)
            ):
                # Duplicate by local param windows
                d_new = self._mismatch_norm(r_tuv)
                d_old = self._mismatch_norm(e_tuv)
                if d_new < d_old:
                    best_idx = i
                else:
                    return  # Keep old, drop new

        if best_idx is not None:
            self.intersections[best_idx] = rec
        else:
            self.intersections.append(rec)

    def _is_flat_cell(self, curve: NURBSCurve, surface: NURBSSurface) -> bool:
        """Return True if the curve segment and the surface subpatch are
        locally flat enough so that further subdivision is unnecessary.

        Criterion: for the mid-parameters (t,u,v), compute curvature-based
        parametric steps (dt, du, dv) whose sagittas are approximately
        `self.tolerance`. If the current parameter extents are smaller than
        these steps in all directions, treat the cell as flat.
        """
        t0, t1 = curve.interval()
        (u0, u1), (v0, v1) = surface.interval()

        tm = 0.5 * (t0 + t1)
        um = 0.5 * (u0 + u1)
        vm = 0.5 * (v0 + v1)

        # Curve derivatives at midpoint
        C1 = curve.derivative(tm)
        C2 = curve.second_derivative(tm)
        # Surface partials at midpoint
        Su = surface.derivative_u(np.array([um, vm]))
        Sv = surface.derivative_v(np.array([um, vm]))
        Suu = surface.second_derivative_uu(np.array([um, vm]))
        Svv = surface.second_derivative_vv(np.array([um, vm]))

        # Curvature-based parametric steps for sagitta ~= tolerance
        dt = compute_parametric_curvature_tolerance_curve(C1, C2, self.tolerance)
        du, dv = compute_parametric_curvature_tolerance_surface(
            Su, Sv, Suu, Svv, self.tolerance
        )

        # Current param extents
        d_t = abs(t1 - t0)
        d_u = abs(u1 - u0)
        d_v = abs(v1 - v0)

        # If any step is non-finite, treat that direction as already flat.
        if not np.isfinite(dt):
            dt = np.inf
        if not np.isfinite(du):
            du = np.inf
        if not np.isfinite(dv):
            dv = np.inf

        # Heuristic safety margin: avoid borderline churn
        safety = 0.99
        return (d_t <= safety * dt) and (d_u <= safety * du) and (d_v <= safety * dv)

    def _classify_flat_cell(self, curve: NURBSCurve, surface: NURBSSurface) -> None:
        """Classify a flat cell without further subdivision.

        - Try Newton from the cell center: if converged, append as usual.
        - Else, use a plane test to detect overlaps; if likely overlap,
          record a representative overlap point.
        - Otherwise, treat as empty (no intersection to report).
        """
        t0, t1 = curve.interval()
        (u0, u1), (v0, v1) = surface.interval()
        tm = 0.5 * (t0 + t1)
        um = 0.5 * (u0 + u1)
        vm = 0.5 * (v0 + v1)
        degenerate=False,lambda :...
        # 1) Attempt Newton from center
        res = newtons_method(self._equation, np.array([tm, um, vm]), max_iter=8)
        if (
            res is not None
            and np.all(np.isfinite(res))
            and self._is_valid_parameter(res, (t0, t1), (u0, u1), (v0, v1))
        ):
            r = self._equation(res) ** 0.5
            if r <= self.tolerance:
                pt = self.initial_curve.evaluate(res[0])
                if self._is_degenerate(res, curve, surface):
                    
                    def degenerate_cb():
                        self._insert_with_param_dedup(curve, surface,
                                                      ("degenerate", pt, tuple(res)))
                    degenerate = True, degenerate_cb
                else:
                    self._insert_with_param_dedup(curve, surface, ("transversal", pt, tuple(res)))
                return

        # 2) Angle-based plane test for likely overlap
        S0 = surface.evaluate_v2(um, vm)
        Su = surface.derivative_u(np.array([um, vm]))
        Sv = surface.derivative_v(np.array([um, vm]))
        N = np.cross(Su, Sv)
        nrm = np.linalg.norm(N)
        if nrm > 0:
            N = N / nrm  # unit normal
            P0 = curve.evaluate(t0)
            P1 = curve.evaluate(t1)
            v0 = P0 - S0
            v1 = P1 - S0
            n0 = np.linalg.norm(v0)
            n1 = np.linalg.norm(v1)
            if n0 > 0 and n1 > 0:
                # Overlap: vectors to endpoints are nearly perpendicular to N
                # i.e., angle(N, v) ~ 90° -> |cos| ~ 0; accept if |cos| <= sin(angle_tol)
                cos0 = abs(float(np.dot(N, v0)) / n0)
                cos1 = abs(float(np.dot(N, v1)) / n1)
                thresh = np.sin(self.angle_tol)
                if cos0 <= thresh and cos1 <= thresh:
                    pt = self.initial_curve.evaluate(tm)
                    self._insert_with_param_dedup(curve, surface, ("overlap", pt, (tm, um, vm)))
                    return
        if degenerate[0]:
            degenerate[1]()
            
        # 3) No reliable evidence of intersection inside this flat cell
        return


def nurbs_csx(curve: NURBSCurve, surface: NURBSSurface, tol=1e-3, ptol=None, angle_tol: float = 0.0524):
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
    enough. A curvature-based flatness stop prevents unnecessary subdivision in near-flat cells and overlap regions.

    **Key Implementation Details**:

    - The separability test prevents unnecessary subdivision when the curve and surface are sufficiently far apart.
    - If a new intersection point is found, it is classified as "transversal" or "degenerate" via the curve-tangent
      and surface-normal angle; flat cells use an angle-based test with `angle_tol` to detect overlaps.
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
    Deprecated and ignored. Present for backward compatibility.
    :type ptol: float, optional

    :param angle_tol:
    Angular tolerance in radians used for classifying near-tangential overlap in flat cells (default 0.0013).
    :type angle_tol: float, optional

    **Returns**:

    :return:
    A list of intersection records. Each record is a tuple of the form:
    - `("type", point, (t, u, v))`
    where `type` can be 'transversal', 'degenerate', or 'overlap'. The `point` is the 3D coordinates of the event
    (for overlap, a representative point), and `(t, u, v)` are the corresponding parametric coordinates.
    :rtype: list

    **Example**:

    .. code-block:: python

    # Example usage of nurbs_csx
    curve = NURBSCurve(control_points=[...], knots=[...], degree=3)
    surface = NURBSSurface(control_points=[...], knots_u=[...], knots_v=[...], degree_u=3, degree_v=3)

    intersections = nurbs_csx(curve, surface, tol=1e-4, angle_tol=0.0013)
    for intersection in intersections:
    print(intersection)

    **Notes**:

    - The algorithm will subdivide the curve and surface recursively until it either finds an intersection or determines
    that no intersection exists within the provided tolerance.
    - For complex surfaces or highly curved regions, consider adjusting the `spt` parameter to increase precision.
    - The classification of intersections as "transversal" or "degenerate" helps distinguish between cases where the
    curve crosses the surface tangentially versus at a sharper angle.

    **Limitations**:

    - This method assumes that both the curve and surface are properly defined and their parameterizations are valid.
    - Extremely small angle tolerances can cause more overlap-classification.
    """
    # `ptol` kept for backward compatibility but is unused.
    intersector = NURBSCurveSurfaceIntersector(curve, surface, tolerance=tol, angle_tol=angle_tol)
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

import numpy as np
from scipy.optimize import minimize

from mmcore.geom.bspline import SubCurve, NURBSpline
from mmcore.geom.vec import norm_sq
from mmcore.numeric.aabb import nurbs_curve_aabb, aabb_overlap

import multiprocess as mp


def nurbs_curve_intersect(curve1: NURBSpline, curve2: NURBSpline, tol: float = 1., workers=4) -> list[
    tuple[float, float] | None]:
    """
    Finds the intersections between two NURBS curves using a divide-and-conquer approach.

    :param curve1: The first NURBS curve.
    :param curve2: The second NURBS curve.
    :param tol: The tolerance for considering two points as intersecting.
    :return: A list of tuples representing the parameter values of the intersections.

    :note:
    The `nurbs_curve_intersect` function uses a divide-and-conquer approach to find the intersections between two NURBS curves. It recursively splits the curves and checks for intersections between the subcurves. The `nurbs_curve_aabb` function is used to compute the axis-aligned bounding box (AABB) of each curve segment, and the `aabb_overlap` function is used to check if the bounding boxes overlap. The recursion stops when the curve segments are small enough to be considered as intersecting.

    """

    def intersect_recursive(c1: NURBSpline, c2: NURBSpline, t1_start: float, t1_end: float, t2_start: float,
                            t2_end: float):
        # Check if the bounding boxes of the curves overlap
        box1 = nurbs_curve_aabb(SubCurve(c1, t1_start, t1_end), tol)
        box2 = nurbs_curve_aabb(SubCurve(c2, t2_start, t2_end), tol)

        if not aabb_overlap(box1, box2):
            return []

        # Check if the curves are small enough to be considered as intersecting
        if (t1_end - t1_start) < tol and (t2_end - t2_start) < tol:
            t1 = (t1_start + t1_end) / 2
            t2 = (t2_start + t2_end) / 2

            return [(t1, t2)]

        # Split the curves and recursively intersect the subcurves
        t1_mid = (t1_start + t1_end) / 2
        t2_mid = (t2_start + t2_end) / 2

        intersections = []

        intersections.extend(intersect_recursive(c1, c2, t1_start, t1_mid, t2_start, t2_mid))
        intersections.extend(intersect_recursive(c1, c2, t1_start, t1_mid, t2_mid, t2_end))
        intersections.extend(intersect_recursive(c1, c2, t1_mid, t1_end, t2_start, t2_mid))
        intersections.extend(intersect_recursive(c1, c2, t1_mid, t1_end, t2_mid, t2_end))

        return intersections

    t1_start, t1_end = curve1.interval()
    t2_start, t2_end = curve2.interval()

    intersections = intersect_recursive(curve1, curve2, t1_start, t1_end, t2_start, t2_end)
    return intersections

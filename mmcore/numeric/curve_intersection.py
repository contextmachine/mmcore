import numpy as np

from mmcore.geom.bspline import SubCurve, CurveProtocol
from mmcore.numeric.aabb import curve_aabb, aabb_overlap
from mmcore.api.bbox import BoundingBox3D


def bbsize(self):
    return self[1] - self[2]


def intersect_bb(self, other):
    self
    return self[1] - self[2]


def curve_intersect(
    curve1: CurveProtocol, curve2: CurveProtocol, tol: float = 0.01
) -> list[tuple[float, float]]:
    """
    Finds the intersections between two NURBS curves using a divide-and-conquer approach.

    :param curve1: The first NURBS curve.
    :param curve2: The second NURBS curve.
    :param tol: The tolerance for considering two points as intersecting.
    :return: A list of tuples representing the parameter values of the intersections.

    :note:
    The `nurbs_curve_intersect` function uses a divide-and-conquer approach to find the intersections between two NURBS
    curves. It recursively splits the curves and checks for intersections between the sub-curves. The `nurbs_curve_aabb`
    function is used to compute the axis-aligned bounding box (AABB) of each curve segment, and the `aabb_overlap`
    function is used to check if the bounding boxes overlap.
    The recursion stops when the curve segments are small enough to be considered as intersecting.

    """

    def intersect_recursive(
        c1: CurveProtocol,
        c2: CurveProtocol,
        t1_start: float,
        t1_end: float,
        t2_start: float,
        t2_end: float,
    ):
        # Check if the bounding boxes of the curves overlap
        box1 = BoundingBox3D(curve_aabb(SubCurve(c1, t1_start, t1_end), tol))
        box2 = BoundingBox3D(curve_aabb(SubCurve(c2, t2_start, t2_end), tol))

        if not box1.intersects(box2):
            return []

        box1.expand(box2.min)
        box1.expand(box2.max)
        # Check if the curves are small enough to be considered as intersecting
        if np.all( box1.sizes() <= tol):
            t1 = (t1_start + t1_end) / 2
            t2 = (t2_start + t2_end) / 2

            return [(t1, t2,  box1)]

        # Split the curves and recursively intersect the sub-curves
        t1_mid = (t1_start + t1_end) / 2
        t2_mid = (t2_start + t2_end) / 2

        intersections = []

        intersections.extend(
            intersect_recursive(c1, c2, t1_start, t1_mid, t2_start, t2_mid)
        )
        intersections.extend(
            intersect_recursive(c1, c2, t1_start, t1_mid, t2_mid, t2_end)
        )
        intersections.extend(
            intersect_recursive(c1, c2, t1_mid, t1_end, t2_start, t2_mid)
        )
        intersections.extend(
            intersect_recursive(c1, c2, t1_mid, t1_end, t2_mid, t2_end)
        )

        return intersections

    first_curve_bounds: tuple[float, float] = curve1.interval()
    second_curve_bounds: tuple[float, float] = curve2.interval()

    all_intersections: list[tuple[float, float]] = intersect_recursive(
        curve1, curve2, *first_curve_bounds, *second_curve_bounds
    )

    return list(zip(*all_intersections))

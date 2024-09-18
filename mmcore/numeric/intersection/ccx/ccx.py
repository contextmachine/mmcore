
from __future__ import annotations

from typing import Callable

import numpy as np
from numpy.typing import ArrayLike

from mmcore.geom.implicit.tree import ImplicitTree2D, implicit_find_features
from mmcore.numeric import scalar_norm

from mmcore.numeric.aabb import curve_aabb, aabb_overlap, curve_aabb_eager
from mmcore.numeric.divide_and_conquer import test_all_roots
from mmcore.numeric.routines import divide_interval
from mmcore.geom.nurbs import NURBSCurve, split_curve_multiple, split_curve

__all__ = ["ccx", "curve_curve_intersect", "curve_x_axis", "curve_x_ray", "curve_pix","curve_ppx", "curve_iix"]

def _calculate_spline_tolerance(spline, default_tol=1e-3):
    """
    spooky heuristic ...
    :param spline:
    :param default_tol:
    :return:
    """
    if hasattr(spline, "degree"):
        return 10 ** (-spline.degree)
    else:
        return default_tol

def curve_x_axis(curve, axis=1, step=0.5):
    return curve_pix(curve, lambda xyz: xyz[axis], step=step)


def curve_x_ray(curve, orig, axis=1, step=0.5):
    orig=orig if isinstance(orig, np.ndarray) else np.array(orig,dtype=float)
    return curve_pix(curve, lambda xyz: (orig - xyz)[axis], step=step)

def ccx(curve1, curve2, tol: float = 0.01):
    """
    Compute the intersection points between two curves (Curve X Curve Intersection).

    This function finds the intersection points between two curves, which can be in parametric
    or implicit form. The method used depends on the forms of the curves provided:

    - If one curve is parametric and the other is implicit, the `curve_pix` function is used.
    - If both curves are parametric, the `curve_ppx` function is used.
    - If both curves are implicit, the `curve_iix` function is used.
    - If neither curve has a recognizable form (parametric or implicit), an error is raised.

    :param curve1:
        The first curve, which may be in parametric or implicit form.
    :param curve2:
        The second curve, which may be in parametric or implicit form.
    :param tol:
        The tolerance for considering two points as intersecting. This parameter influences the
        precision of the intersection points.
    :type tol: float
    :return:
        A list of intersection points, with each point represented as a tuple of parameter values
        on `curve1` and `curve2`.
    :rtype: list[tuple[float, float]]

    .. note::

        To find an approximate intersection (e.g., when the curves do not precisely intersect but
        are close), consider using the `ccd` or `proximity_points_curve_curve` functions,
        which can handle near-intersections.

    **Method Selection**:

    +-------------------+-------------------+---------------------------+
    | Curve 1           | Curve 2           | Method Used                |
    +===================+===================+===========================+
    | Parametric        | Implicit          | `curve_pix`                |
    +-------------------+-------------------+---------------------------+
    | Implicit          | Parametric        | `curve_pix`                |
    +-------------------+-------------------+---------------------------+
    | Parametric        | Parametric        | `curve_ppx`                |
    +-------------------+-------------------+---------------------------+
    | Implicit          | Implicit          | `curve_iix`                |
    +-------------------+-------------------+---------------------------+
    | None              | None              | Error raised               |
    +-------------------+-------------------+---------------------------+

    **Usage Example**::

        >>> pts1 = np.array([[65.468, 86.661, 0.0], [78.389, 35.249, 0.0], [82.615, 73.760, 0.0],
        ...                  [1.217, 18.625, 0.0]])
        >>> pts2 = np.array([[61.974, 73.943, 0.0], [119.797, 4.443, 0.0]])

        >>> from mmcore.geom.nurbs import NURBSCurve
        >>> nc1, nc2 = NURBSCurve(pts1, degree=1), NURBSCurve(pts2, degree=1)
        >>> ccx(nc1, nc2, tol=1e-3)
        [(0.4714, 0.1658), (0.4718, 0.1659), (1.4348, 0.3157), (1.4353, 0.3157),
         (2.1610, 0.1302), (2.1609, 0.1304)]

        # Example with non-intersecting coplanar curves
        >>> pts3 = np.copy(pts1)
        >>> pts3[..., -1] += 1e-11  # Slight adjustment to the z-coordinate
        >>> nc3 = NURBSCurve(pts3, degree=1)
        >>> ccx(nc3, nc2, tol=1e-3)  # No intersections found due to coplanarity
        []

    """
    if hasattr(curve1, "implicit") and hasattr(curve2, "evaluate"):
        return curve_pix(curve2, curve1)
    elif hasattr(curve2, "implicit") and hasattr(curve1, "evaluate"):
        return curve_pix(curve1, curve2)
    elif hasattr(curve2, "evaluate") and hasattr(curve1, "evaluate"):
        return curve_ppx(curve1, curve2, tol=tol)
    elif hasattr(curve2, "implicit") and hasattr(curve1, "implicit"):
        raise curve_iix(
            curve1, curve2)
    else:
        raise ValueError(
            "curves must have parametric or implicit form - evaluate(t) and implicit(xyz) methods. "
            "If you want intersect a simple callables, use curve_PI, or curve_ppi function."
        )


curve_curve_intersect = ccx

def curve_pix(curve, implicit: Callable[[ArrayLike], float], step: float = 0.5, tol=1e-3) -> list[float]:
    """
    Find the intersection parameters between a parametric curve and an implicit form.

    This function computes the parameters at which a parametric curve intersects an implicit
    surface, curve, or any geometric entity defined by an equation of the form `f(x) = 0`,
    where `x` is a vector representing a point in space.

    :param curve:
        The curve in parametric form, which must provide an `evaluate(t)` method to return points on the curve.
    :param implicit:
        The implicit function defining the geometric entity to intersect with, such as a curve or surface.
        This should be a callable that takes a point in space (as an array-like object) and returns a float,
        which is zero when the point lies on the implicit entity.
    :type implicit: callable
    :param step:
        The initial step size used to divide the parameter space of the curve for intersection estimation.
        Default is 0.5.
    :type step: float
    :param tol:
        The tolerance for refining the intersection points. This value controls the precision of the intersection
        parameters in the parametric space. If the curve is a spline, this value might be adjusted automatically.
        Default is 0.001.
    :type tol: float
    :return:
        A list of parameter values where the parametric curve intersects the implicit form.
    :rtype: list[float]

    This algorithm is optimal for problems involving intersections between a parametric curve and an implicit form.
    It is typically more efficient than more general intersection algorithms and should be preferred when applicable.

    **Usage Example**::

        >>> import numpy as np
        >>> from mmcore.numeric.intersection.ccx import curve_pix
        >>> from mmcore.geom.curves import NURBSpline

        # Define an implicit curve (Cassini oval)
        >>> def cassini(x, a=1.1, c=1.0):
        ...     return ((x[0] ** 2 + x[1] ** 2) ** 2 - 2 * c * c * (x[0] ** 2 - x[1] ** 2) - (a ** 4 - c ** 4))

        # Create a parametric curve (NURBS spline)
        >>> spline = NURBSpline(np.array([(-2.38, 2.23, 0.0), (-1.15, 1.39, 0.0), (-1.24, 0.37, 0.0),
        ...                               (-0.83, -0.25, 0.0), (-0.06, -0.79, 0.0), (1.43, -1.29, 0.0),
        ...                               (1.02, -0.59, 0.0), (-0.41, 0.43, 0.0), (-0.09, 2.07, 0.0)]), degree=3)

        # Calculate intersection parameters
        >>> t = curve_pix(spline, cassini)
        >>> t
        [0.9994, 2.5384, 4.8581, 5.5516]

        # Evaluate the points on the curve at these parameters
        >>> spline(t)
        array([[-1.15033465,  0.52553561,  0.        ],
               [-0.39017002, -0.53625519,  0.        ],
               [ 0.89756699, -0.59935844,  0.        ],
               [-0.01352083,  0.45838774,  0.        ]])
    """
    implicit_form= getattr(implicit, "implicit", implicit)
    evaluate_parametric_form = getattr(curve, "evaluate", curve)
    #tol = tol
    roots = []
    for start, end in divide_interval(*curve.interval(), step=step):
        roots.extend(
            test_all_roots(
                lambda t:   implicit_form(evaluate_parametric_form(t)), (start, end), tol=tol
            )
        )
    return roots




def curve_ppx(curve1, curve2, tol: float = 0.001, tol_bbox=0.1, bounds1=None, bounds2=None, eager=True) -> list[
    tuple[float, float]]:
    """
    Find intersections between two parametric curves (Parametric X Parametric).

    This function computes the intersection points between two parametric curves using
    a divide-and-conquer approach. It recursively checks the overlap of the curves'
    axis-aligned bounding boxes (AABB) and subdivides the parameter space until the curves
    are within the specified tolerance.

    :param curve1:
        The first parametric curve, expected to implement an `evaluate(t)` method.
    :param curve2:
        The second parametric curve, similar to `curve1`.
    :param tol:
        The tolerance for considering two points as intersecting.
    :type tol: float
    :param tol_bbox:
        The tolerance for AABB overlap during the recursive subdivision.
    :type tol_bbox: float
    :param bounds1:
        Optional custom bounds for `curve1`. Defaults to the curve's full interval.
    :type bounds1: Optional[tuple[float,float]]
    :param bounds2:
        Optional custom bounds for `curve2`. Defaults to the curve's full interval.
    :type bounds1: Optional[tuple[float,float]]
    :param eager:
        If True, uses a precomputed AABB for faster but less precise checks.
        If False, checks bounding boxes on-the-fly.
    :type eager: bool
    :return: 
        A list of tuples representing the parameter values of the intersections on each curve.
    :rtype: list[tuple[float, float]]

    **Usage Example**::

        >>> first = NURBSpline(np.array([(-13.654, -19.908, 0.0), (3.758, -39.949, 0.0),
        ...                              (16.324, -18.019, 0.0), (44.907, -38.224, 0.0),
        ...                              (49.260, -13.419, 0.0)]))
        >>> second = NURBSpline(np.array([(40.965, -3.892, 0.0), (-9.548, -28.039, 0.0),
        ...                               (4.168, -58.265, 0.0), (37.269, -58.101, 0.0)]))
        >>> intersections = curve_ppx(first, second, tol=0.001)
        >>> print(intersections)
        [(0.6007, 0.3717)]
    """
    def recursive_intersect(c1, c2, t1_range, t2_range, tol):
        # print(c1.interval(), t1_range, c2.interval(), t2_range)


        if eager:
            if not aabb_overlap(
                    curve_aabb_eager(c1, bounds=t1_range, cnt=8),
                    curve_aabb_eager(c2, bounds=t2_range, cnt=8),
            ):
                return []

        else:
            if not aabb_overlap(
                    curve_aabb(c1, bounds=t1_range, tol=tol_bbox),
                    curve_aabb(c2, bounds=t2_range, tol=tol_bbox)):
                return []

        t1_mid = t1_range[0] + (t1_range[1] - t1_range[0]) / 2
        t2_mid = t2_range[0] + (t2_range[1] - t2_range[0]) / 2

        if np.linalg.norm(np.array(c1(t1_mid)) - np.array(c2(t2_mid))) <= tol:
            return [(t1_mid, t2_mid)]
        c1_left = t1_range[0], t1_mid
        c1_right = t1_mid, t1_range[1]
        c2_left = t2_range[0], t2_mid
        c2_right = t2_mid, t2_range[1]
        # c1_left, c1_right = split(c1, t1_mid)
        # c2_left, c2_right = split(c2, t2_mid)

        intersections = []
        intersections.extend(recursive_intersect(c1, c2, c1_left, c2_left, tol))
        intersections.extend(recursive_intersect(c1, c2, c1_left, c2_right, tol))
        intersections.extend(recursive_intersect(c1, c2, c1_right, c2_left, tol))
        intersections.extend(recursive_intersect(c1, c2, c1_right, c2_right, tol))

        return intersections

    result = []

    curve1_bounds = curve1.interval() if bounds1 is None else bounds1
    curve2_bounds = curve2.interval() if bounds2 is None else bounds2

    curve1_range = curve1_bounds[1] - curve1_bounds[0]
    curve2_range = curve2_bounds[1] - curve2_bounds[0]

    if curve1_range > 1.0 and curve2_range > 1.0:
        for start1, end1 in divide_interval(*curve1_bounds, 1.0):
            for start2, end2 in divide_interval(*curve2_bounds, 1.0):
                result.extend(
                    recursive_intersect(
                        curve1, curve2, (start1, end1), (start2, end2), tol=tol
                    )
                )
    elif curve1_range > 1.0:
        start2, end2 = curve2_bounds
        for start1, end1 in divide_interval(*curve1_bounds, 1.0):
            result.extend(
                recursive_intersect(
                    curve1, curve2, (start1, end1), (start2, end2), tol=tol
                )
            )
    elif curve2_range > 1.0:
        start1, end1 = curve1_bounds
        for start2, end2 in divide_interval(*curve2_bounds, 1.0):
            result.extend(
                recursive_intersect(
                    curve1, curve2, (start1, end1), (start2, end2), tol=tol
                )
            )
    else:
        result.extend(
            recursive_intersect(curve1, curve2, curve1_bounds, curve2_bounds, tol)
        )

    return result


def curve_iix(curve1, curve2, tree: ImplicitTree2D = None, rtol=None, atol=None):
    """
    Find intersections between two implicit curves (Implicit X Implicit).

    This function computes the intersection points between two implicit curves. Both curves must
    provide an `implicit` method that returns the implicit function value for a given point.

    :param curve1:
        The first implicit curve, which must implement an `implicit(xy)` method.
    :param curve2:
        The second implicit curve, similar to `curve1`.
    :param tree:
        Optional `ImplicitTree2D` object representing discretizations of primitives.
        If None, a new tree is constructed from the union of the curves.
    :param rtol:
        Relative tolerance for considering two intersection points as identical.
    :type rtol: float
    :param atol:
        Absolute tolerance for considering two intersection points as identical.
    :type atol: float
    :return:
        A list of intersection points in the form of 2D coordinates.
    :rtype: list[list[float]]

    **Note**:
    The `atol` and `rtol` parameters control the precision of intersection points and affect whether
    close intersections are considered identical. If not specified, default values from `np.allclose`
    are used.

    **Usage Example**::

        >>> from mmcore.geom.implicit import Implicit2D

        # Define two implicit circles
        >>> class Circle2D(Implicit2D):
        ...     def __init__(self, origin=(0.0, 0.0), radius=1):
        ...         super().__init__(autodiff=True)
        ...         self.origin = np.array(origin, dtype=float)
        ...         self.radius = radius
        ...     def implicit(self, v):
        ...         return np.linalg.norm(v - self.origin) - self.radius

        >>> c1, c2 = Circle2D((0., 1), 2), Circle2D((3., 3), 3)
        >>> intersections = curve_iix(c1, c2)
        >>> intersections
        [[1.8462, 0.2308], [0.0, 3.0]]
    """
    if tree is None:
        tree = ImplicitTree2D(lambda v: min(curve1.implicit(v), curve2.implicit(v)), depth=4)
        tree.build()
    return list(implicit_find_features((curve1, curve2), tree.border, atol=atol, rtol=rtol))






def curve_intersect_old(curve1, curve2, tol: float = 0.01) -> list[tuple[float, float]]:
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
            c1,
            c2,
            t1_start: float,
            t1_end: float,
            t2_start: float,
            t2_end: float,
    ):
        # Check if the bounding boxes of the curves overlap
        box1 = curve_aabb(c1, (t1_start, t1_end), tol)
        box2 = curve_aabb(c2, (t2_start, t2_end), tol)
        if not box1.intersects(box2):
            return []

        box1.expand(box2.min)
        box1.expand(box2.max)
        # Check if the curves are small enough to be considered as intersecting

        if np.all(box1.sizes() <= tol):
            t1 = (t1_start + t1_end) / 2
            t2 = (t2_start + t2_end) / 2

            return [(t1, t2, box1)]

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




if __name__ == "__main__":
    # print(res[0].control_points, res[1].control_points)
    from mmcore.geom.curves import NURBSpline

    aa, bb = NURBSpline(
        np.array(
            [
                (-13.654958030023677, -19.907874497194975, 0.0),
                (3.7576433265207765, -39.948793039632903, 0.0),
                (16.324284871574083, -18.018771519834026, 0.0),
                (44.907234268165922, -38.223959886390297, 0.0),
                (49.260384607302036, -13.419216444520401, 0.0),
            ]
        )
    ), NURBSpline(
        np.array(
            [
                (40.964758489325661, -3.8915666456564679, 0.0),
                (-9.5482124270650726, -28.039230791052990, 0.0),
                (4.1683178868166371, -58.264878428828240, 0.0),
                (37.268687446662931, -58.100608604709883, 0.0),
            ]
        )
    )
    import time

    s = time.time()
    res = curve_ppx(aa, bb, 0.001, tol_bbox=0.1, eager=True)

    print(time.time() - s)

    print(res)
    # [(0.600738525390625, 0.371673583984375)]
    print(aa(res[0][0]))

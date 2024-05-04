from __future__ import annotations

import numpy as np

from mmcore.numeric.plane import inverse_evaluate_plane
from mmcore.numeric.divide_and_conquer import test_all_roots
from mmcore.numeric.routines import divide_interval
from mmcore.numeric.aabb import curve_aabb, aabb_overlap, curve_aabb_eager

from typing import Callable, Any
from numpy.typing import ArrayLike


def _calculate_spline_tolerance(spline, default_tol=1e-3):
    if hasattr(spline, "degree"):
        return 10 ** (-spline.degree)
    else:
        return default_tol


def curve_pii(
    curve, implicit: Callable[[ArrayLike], float], step: float = 0.5, default_tol=1e-3
) -> list[float]:
    """
        The function finds the parameters where the parametric curve intersects some implicit form. It can be a curve, surface, ..., anything that has the form f(x)=0 where x is a vector denoting a point in space.


        :param curve: the curve in parametric form

        :param implicit: the implicit function like `f(x) = 0` where x is a vector denoting a point in space.
        :type implicit: callable
        :param step: Step with which the range of the curve parameter is initially split for estimation. defaults to 0.5
        :param default_tol: The error for computing the intersection in the parametric space of a curve.
        If the curve is a spline, the error is calculated automatically and the default_tolerance value is ignored.
        :type default_tol: float
        :return: list of parameters that implicit function is intersects
        :rtype: list[float]

        This algorithm is the preferred algorithm for curve crossing problems.
        Its solution is simpler and faster than the general case for PP (parametric-parametric) or II (implict-implict)
        intersection. You should use this algorithm if you can make one of the forms parametric and the other implicit.


    Intersection with implicit curve.
    -----------------


        >>> import numpy as np
        >>> from mmcore.numeric.curve_intersection import curve_pii
        >>> from mmcore.geom.curves import NURBSpline


    1. Define the implict curve.

        >>> def cassini(x, a=1.1, c=1.0):
        ...     return ((x[0] ** 2 + x[1] ** 2) * (x[0] ** 2 + x[1] ** 2)
        ...              - 2 * c * c * (x[0] ** 2 - x[1] ** 2)
        ...              - (a ** 4 - c ** 4))


     2. Сreate the parametric curve:

        >>> spline = NURBSpline(np.array([(-2.3815177882733494, 2.2254910228438045, 0.0),
        ...                               (-1.1536662710614194, 1.3922103249454953, 0.0),
        ...                               (-1.2404122859674858, 0.37403957301406443, 0.0),
        ...                               (-0.82957856158065857, -0.24797333823516698, 0.0),
        ...                               (-0.059146886557566614, -0.78757517340047745, 0.0),
        ...                               (1.4312784414267623, -1.2933712167625511, 0.0),
        ...                               (1.023775628607696, -0.58571247602345811, 0.0),
        ...                               (-0.40751426943615976, 0.43200382009529514, 0.0),
        ...                               (-0.091810780095197053, 2.0713419737806906, 0.0)]
        ...                           ),
        ...                           degree=3)

    3. Calculate the parameters in which the parametric curve intersects the implicit curve.

        >>> t = curve_pii(spline,cassini)
        >>> t
        [0.9994211794774993, 2.5383824909241675, 4.858054223961756, 5.551602752306095]

        4. Now you can evaluate the points by passing a list of parameters to a given parametric curve.

        >>> spline(t)
        array([[-1.15033465,  0.52553561,  0.        ],
           [-0.39017002, -0.53625519,  0.        ],
           [ 0.89756699, -0.59935844,  0.        ],
           [-0.01352083,  0.45838774,  0.        ]])


    Intersection with implicit surface.
    -----
    1. Define the implicit surface.

        >>> def genus2(x):
        ...     return 2 * x[1] * (x[1] ** 2 - 3 * x[0] ** 2) * (1 - x[2] ** 2) + (x[0] ** 2 + x[1] ** 2) ** 2 - (
        ...         9 * x[2] ** 2 - 1) * (1 - x[2] ** 2)

    2. Repeat the items from the previous example

        >>> t = curve_pii(spline, genus2)
        >>> t
        [0.6522415161474464, 1.090339012572083]

        >>> spline(t)
        array([[-1.22360866,  0.96065424,  0.        ],
           [-1.13538864,  0.43142482,  0.        ]])

    """

    evaluate_parametric_form = getattr(curve, "evaluate", curve)
    tol = _calculate_spline_tolerance(curve, default_tol)
    roots = []
    for start, end in divide_interval(*curve.interval(), step=step):
        roots.extend(
            test_all_roots(
                lambda t: implicit(evaluate_parametric_form(t)), (start, end), tol=tol
            )
        )
    return roots


def curve_x_axis(curve, axis=1, step=0.5):
    return curve_pii(curve, lambda xyz: xyz[axis], step=step)


def curve_x_plane(curve, plane, axis=2, step=0.5):
    return curve_pii(
        curve, lambda xyz: inverse_evaluate_plane(plane, xyz)[axis], step=step
    )


def curve_intersect(curve1, curve2, tol: float = 0.01):
    if hasattr(curve1, "implicit") and hasattr(curve2, "evaluate"):
        return curve_pii(curve2, curve1)
    elif hasattr(curve2, "implicit") and hasattr(curve1, "evaluate"):
        return curve_pii(curve1, curve2)
    elif hasattr(curve2, "evaluate") and hasattr(curve1, "evaluate"):
        return curve_ppi(curve1, curve2, tol=tol)
    elif hasattr(curve2, "implicit") and hasattr(curve1, "implicit"):
        raise NotImplemented("III has not been implemented yet.")
    else:
        raise ValueError(
            "curves must have parametric or implicit form - evaluate(t) and implicit(xyz) methods. "
            "If you want intersect a simple callables, use curve_pii, or curve_ppi function."
        )


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


def curve_ppi(
    curve1,
    curve2,
    tol: float = 0.001,
    tol_bbox=0.1,
    bounds1=None,
    bounds2=None,
    eager=True,
) -> list[tuple[float, float]]:
    """
    PPI
    ------

    PPI (Parametric Parametric Intersection) for the curves.
    curve1 and curve2 can be any object with a parametric curve interface.
    However, in practice it is worth using only if both curves do not have implicit representation,
    most likely they are two B-splines or something similar.
    Otherwise it is much more efficient to use PII (Parametric Implict Intersection).

    The function uses a recursive divide-and-conquer approach to find intersections between two curves.
    It checks the AABB overlap of the curves and recursively splits them until the distance between the curves is within
    the specified tolerance or there is no overlap. The function returns a list of tuples
    representing the parameter values of the intersections on each curve.

    Обратите внимание! Этот метод продолжает "Разделяй и властвуй" пока расстояние не станет меньше погрешности.
    Вы можете значительно ускорить поиск, начиная метод ньютона с того момента где для вас это приемлимо.
    Однако имейте ввиду что для правильной сходимости вы уже должны быть в "низине" с одним единственым минимумом.

    :param curve1: first curve
    :param curve2: second curve
    :param bounds1: [Optional] custom bounds for first NURBS curve. By default, the first NURBS curve interval.
    :param bounds2: [Optional] custom bounds for first NURBS curve. By default, the second NURBS curve interval.
    :param tol: A pair of points on a pair of Euclidean curves whose Euclidean distance between them is less than tol will be considered an intersection point
    :return: List containing all intersections, or empty list if there are no intersections. Where intersection
    is the tuple of the parameter values of the intersections on each curve.
    :rtype: list[tuple[float, float]] | list

    Example
    --------
    >>> first = NURBSpline(
    ...    np.array(
    ...        [
    ...            (-13.654958030023677, -19.907874497194975, 0.0),
    ...            (3.7576433265207765, -39.948793039632903, 0.0),
    ...            (16.324284871574083, -18.018771519834026, 0.0),
    ...            (44.907234268165922, -38.223959886390297, 0.0),
    ...            (49.260384607302036, -13.419216444520401, 0.0),
    ...        ]
    ...    )
    ... )
    >>> second= NURBSpline(
    ...     np.array(
    ...         [
    ...             (40.964758489325661, -3.8915666456564679, 0.0),
    ...             (-9.5482124270650726, -28.039230791052990, 0.0),
    ...             (4.1683178868166371, -58.264878428828240, 0.0),
    ...             (37.268687446662931, -58.100608604709883, 0.0),
    ...         ]
    ...     )
    ... )



    >>> intersections = curve_ppi(first, second, 0.001)
    >>> print(intersections)
    [(0.600738525390625, 0.371673583984375)]


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
    res = curve_ppi(aa, bb, 0.01, tol_bbox=0.1)
    print(time.time() - s)
    print(res)

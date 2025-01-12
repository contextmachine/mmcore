
from mmcore.numeric.divide_and_conquer import find_all_minima
from mmcore.numeric.vectors import scalar_dot


def ccd(curve1, curve2, tol: float = 0.001):
    """
    Curve-Curve Distance (CCD)
    Compute the all distance minima between two curves.

    This function calculates all minima of the distance function between two parametric curves
    over their respective domains. It first evaluates the squared distance between points on
    `curve1` and `curve2`, then refines the found minima using a Newton's method, where the
    accuracy of these minima is controlled by `tol`.

    :param curve1:
        The first curve, which must implement an `evaluate(t)` method that returns a point
        on the curve for a parameter `t`. The curve should also provide an `interval()` method
        that returns the domain of `t` as a tuple `(t_min, t_max)`.
    :param curve2:
        The second curve, which must implement an `evaluate(s)` method similar to `curve1`,
        and an `interval()` method that returns the domain of `s` as a tuple `(s_min, s_max)`.
    :param tol:
        The tolerance for Newton's method, which is used to refine the minima found.
        Smaller values of `tol` will result in more accurate minima but may require more iterations.
        Default is 0.001.

    :return:
        A list of tuples where each tuple contains:
        - The parameter on `curve1` where the minimum occurs.
        - The parameter on `curve2` where the minimum occurs.
        - The corresponding minimum squared distance between the points on the curves.
    :rtype:
        List[Tuple[float, float, float]]

    Usage example::

        minima = ccd(curve1, curve2, tol=0.001)
        for t, s, dist in minima:
    """

    def fun(t, s):
        d = curve1.evaluate(t) - curve2.evaluate(s)
        return scalar_dot(d, d)

    sol = find_all_minima(fun, curve1.interval(), curve2.interval(), tol=tol)
    return sol


def proximity_points_curve_curve(curve1, curve2, tol: float = 0.0001):
    """
    Find the closest proximity points between two curves.

    This function uses the CCD algorithm to identify the point on each curve that is closest
    to the other. The points are refined using Newton's method with the accuracy controlled by
    `tol`. It returns the parameters for these points and the corresponding minimum distance.

    :param curve1:
        The first curve, expected to implement `evaluate(t)` and `interval()` methods.
    :param curve2:
        The second curve, expected to implement `evaluate(s)` and `interval()` methods.
    :param tol:
        The tolerance for Newton's method used in refining the proximity points.
        A smaller `tol` provides more precise proximity points but may require more computation.
        Default is 0.0001.

    :return:
        A tuple containing:
        - The parameter on `curve1` where the closest point is located.
        - The parameter on `curve2` where the closest point is located.
        - The minimum squared distance between these two points.
    :rtype:
        Tuple[float, float, float]

    Usage example::

        closest_points = proximity_points_curve_curve(curve1, curve2, tol=0.0001)
        t, s, min_dist = closest_points
        print(f"Closest points: curve1(t={t}), curve2(s={s}), distance = {min_dist}")
    """
    return min(ccd(curve1, curve2, tol), key=lambda x: x[-1])
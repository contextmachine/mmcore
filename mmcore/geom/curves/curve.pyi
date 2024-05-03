from __future__ import annotations
from typing import Protocol,Union,Optional

import numpy as np
from numpy._typing import NDArray



class ImplicitCurveProtocol(Protocol):
    def interval(self) -> tuple[float, float]: ...
    def implicit(self, pt: NDArray[float]) -> float: ...


class ParametricCurveProtocol(Protocol):
    def interval(self) -> tuple[float, float]: ...
    def evaluate(self, t:float) ->  NDArray[float]: ...

    def intersect_with_curve(self, other: Union[ImplicitCurveProtocol, ParametricCurveProtocol]) -> list[
        tuple[float, Optional[float]]]: ...



class Curve:
    def invalidate_cache(self)->None:
        """
        Invalidates the cache.

        :return: None
        """
        ...
    def interval(self) -> tuple[float, float]:
        """
         Parametric domain of the curve.
        :param t:float
        :return: start and end parameter values

        """
    def evaluate(self, t: float) -> NDArray[float]:
        """
        Evaluate point on curve at t value
        :param t:float
        :return: point on curve at t parameter value
            as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.


        """
    def evaluate_multi(self, t: NDArray[float]) -> NDArray[float]:
        """
        Evaluate points on the curve at t values array
        :param t: np.ndarray[float] with shape N
        :return: array of points np.ndarray[float] with shape (N, M) where M 2 or 3 in 2d and 3d cases
        """
        ...
    def derivative(self, t) -> np.ndarray[(3,), np.dtype[float]]:
        """

        :param t:float
        :return: vector of first derivative   as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.


        """
        ...
    def second_derivative(self, t) -> np.ndarray[(3,), np.dtype[float]]:
        """

        :param t:float
        :return: vector of second derivative  as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.
        :rtype: np.ndarray[(3,), np.dtype[float]]

        """
        ...
    def curvature(self, t: float) -> NDArray[float]:
        """
        Evaluate a curvature vector in a given parameter t
        :param t: float
        :return: vector on curve at t parameter value
            as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.


        """
    def tangent(self, t: float) -> NDArray[float]:
        """

        Calculate tangent vector `T(t)` from the first `D1(t)` and second `D2(t)` derivatives at t parameter
        .. math::
            T(D1, D2)=\\dfrac{D2}{||D1||} \\text{ where }||D2|| \\neq{0}\\text{ else } \\dfrac{D1}{||D1||}
        where:

        D1 - first derivative vector \
        D2 - second derivative vector \
        T - tangent vector \

        :param t: float
        :return: tangent vector as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.



        """
        ...
    def normal(self, t: float) -> NDArray[float]:
        """

        :param t:float
        :return: normal vector as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.


        """
        ...
    def plane_at(self, t):
        """
        The plane formed by the tangent vector and normal vector at a given point
        :param t:float
        :return: normal vector as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.


        """
    def evaluate_length(self, bounds: tuple[float, float]) -> NDArray[float]:
        """
        Estimate the length of the curve from bounds[0] to bounds[1] using integration
        :param t: float
        :return: point on curve at t parameter value
            as numpy array of floats with shape (2,) for 2d case and (3,) for 3d case.


        """
    def evaluate_parameter_at_length(self, length, t0=None,**kwargs)->float:
        """
        Evaluates the parameter at a specific length.

        :param length: The specific length at which the parameter is to be evaluated.
        :param t0: The start time for evaluating the parameter. If not provided, it defaults to the start time of the interval.
        :param kwargs: Additional keyword arguments to be passed.
        :return: The parameter value at the specified length.
        """
        ...
    def circle_of_curvature(self, t:float)->tuple[NDArray[float],float,NDArray[float]]:...

    def intersect_with_curve(self, curve: Curve) -> list[tuple[float, float]]:
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
        >>> import numpy as np
        >>> from mmcore.geom.curves.bspline import NURBSpline
        >>> from mmcore.numeric.curve_intersection import curve_ppi
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
        >>> second = NURBSpline(
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
        ...

    def split(self, t:float)->tuple[TrimmedCurve, TrimmedCurve]:
        """
        Split Method
        --------------
        Split method splits a curve into two trimmed curves at a specified parameter value.

        :param t: The parameter value at which to split the curve. Should be a float.
        :return: A tuple containing two TrimmedCurve objects representing the divided parts of the curve.

        Example Usage:
        --------------
        >>> curve = TrimmedCurve()
        >>> left_curve, right_curve = curve.split(0.5)
        """
        ...





class TrimmedCurve(Curve):
    trim:tuple[float,float]
    curve:Curve

    @property
    def end(self) -> float: ...

    @property
    def start(self) -> float: ...


SubCurve=TrimmedCurve
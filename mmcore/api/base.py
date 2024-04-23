from __future__ import annotations

import abc
import math
from abc import ABCMeta, abstractmethod
from ._typing import Union, Generic, TypeVar, List, Any

from numpy._typing import ArrayLike

T = TypeVar

import numpy as np

from mmcore.api._base import Base, ObjectCollection, TB
from mmcore.api.bbox import BoundingBox2D, BoundingBox3D
from mmcore.api.enums import SurfaceTypes, Curve2DTypes, Curve3DTypes
from mmcore.api.vectors import Matrix3D, Matrix2D, Vector3D, Point3D, Vector2D, Point2D
from mmcore.geom.vec import norm, norm_sq
from mmcore.numeric.fdm import FDM
from mmcore.numeric.closest_point import closest_point_on_curve

TOLERANCE = 1e-5
ROUND_FACTOR = int(abs(math.log10(TOLERANCE)))
from mmcore.numeric.numeric import (
    evaluate_curvature,
    evaluate_tangent,
    evaluate_length,
    evaluate_curvature_vec,
    evaluate_tangent_vec,
    normal_at,
    plane_on_curve,
    evaluate_parameter_from_length,
)
from mmcore.numeric.aabb import curve_aabb
from mmcore.numeric.curve_intersection import curve_intersect


class Surface(Base):
    """
    This class represents a surface. It inherits from the Base class.
    """

    def transform(self, matrix: Matrix3D) -> bool:
        """
        Updates this surface by transforming it with a given input matrix.
        matrix : A 3D matrix that defines the transform to apply to the surface.
        Returns true if the transform was successful.
        """
        return bool()

    @property
    def surface_type(self) -> SurfaceTypes:
        """
        Returns the surface type.
        """
        return SurfaceTypes()

    @property
    def evaluator(self) -> SurfaceEvaluator:
        """
        Returns the surface evaluator.
        """
        return SurfaceEvaluator()

    def intersect_with_line(self, line) -> Point3D:
        ...


class Intersection(Base):
    def owners(self) -> tuple[TB, TB]:
        ...

    def get_parametric(self) -> ArrayLike:
        ...

    def get_geometry(self) -> ObjectCollection:
        ...


class BaseCurve(Base):
    """
    BaseCurve

    This class represents a base curve in 2D space.

    Methods:
    - interval() -> tuple[float, float]: This method returns the interval over which the curve is defined.

    - transform(matrix): Transforms this curve in 2D space.
      - matrix: A 2D matrix that defines the transform to apply to the curve.
      - Returns: True if the transform was successful.

    Properties:
    - curve_type: Returns the type of geometry this curve represents.
    - evaluator: Returns an evaluator object that lets you perform additional evaluations on the curve.

    Attributes:
    - evaluate_multi: A vectorized version of the evaluate method.
    - _derivatives: A list that stores the derivatives of the curve.
    - derivative: A property that returns the first derivative of the curve.
    - second_derivative: A property that returns the second derivative of the curve.

    Methods for Curve Analysis:
    - normal(t): Calculates the normal vector at a given parameter t on the curve.
    - tangent(t): Calculates the tangent vector at a given parameter t on the curve.
    - curvature(t): Calculates the curvature at a given parameter t on the curve.
    - plane_at(t): Calculates the plane on the curve at a given parameter t.

    Methods for Managing Derivatives:
    - add_derivative(): Adds a new derivative to the curve.
    - evaluate(t: float) -> ArrayLike: Abstract method to evaluate the curve at a given parameter t.

    Calling the Object:
    - __call__(t: Union[np.ndarray[Any, float], float]) -> np.ndarray[Any, np.dtype[float]]: Calls the evaluate_multi method to evaluate the curve at multiple parameter values.

    Example usage:
    ```python
    curve = BaseCurve()

    interval = curve.interval()
    print(interval)

    curve.transform(matrix)

    curve_type = curve.curve_type
    print(curve_type)

    evaluator = curve.evaluator
    print(evaluator)

    curve.add_derivative()
    curve.add_derivative()

    derivative = curve.derivative
    print(derivative)

    second_derivative = curve.second_derivative
    print(second_derivative)

    result = curve(t)
    print(result)
    ```
    """

    @abstractmethod
    def interval(self) -> tuple[float, float]:
        ...

    @abstractmethod
    def evaluate(self, t: float) -> ArrayLike:
        ...

    @abstractmethod
    def transform(self, matrix):
        """
        Transforms this curve in 2D space.
        matrix : A 2D matrix that defines the transform to apply to the curve.
        Return true if the transform was successful.
        """
        ...

    @property
    def curve_type(self) -> Curve2DTypes:
        """
        Returns the type of geometry this curve represents.
        """
        return Curve2DTypes()

    @property
    def evaluator(self) -> CurveEvaluator2D:
        """
        Returns an evaluator object that lets you perform additional evaluations on the curve.
        """
        return CurveEvaluator2D(self)

    def __init__(self):
        super().__init__()
        self.evaluate_multi = np.vectorize(self.evaluate, signature="()->(i)")
        self._derivatives = [self]
        self.add_derivative()
        self.add_derivative()

    def normal(self, t):
        return normal_at(self.derivative(t), self.second_derivative(t))

    def tangent(self, t):
        return evaluate_tangent(self.derivative(t), self.second_derivative(t))[0]

    def curvature(self, t):
        return evaluate_curvature(self.derivative(t), self.second_derivative(t))[1]

    def plane_at(self, t):
        return plane_on_curve(self(t), self.tangent(t), self.second_derivative(t))

    @property
    def derivative(self):
        return self._derivatives[1]

    @property
    def second_derivative(self):
        return self._derivatives[2]

    def add_derivative(self):
        self._derivatives.append(FDM(self._derivatives[-1]))
        return len(self._derivatives)

    def __call__(
        self, t: Union[np.ndarray[Any, float], float]
    ) -> np.ndarray[Any, np.dtype[float]]:
        return self.evaluate_multi(t)

    def intersect_with_curve(self, curve: Curve3D) -> ObjectCollection:
        """
        :param curve: The curve with which the object will be intersected.
        :return: An ObjectCollection containing the intersecting points on the curve.

        This method intersects the current object with the given curve. It returns the intersecting points on the curve as an ObjectCollection. If there are no intersecting points, an empty
        * ObjectCollection is returned.
        """
        res, dists = curve_intersect(self, curve, tol=0.01)
        if len(res) == 0:
            return ObjectCollection()
        else:
            return [(self.evaluate(i[0]), i) for i in res]


class Curve2D(BaseCurve):
    """
      :class:`Curve2D`
    ===============
    A 2D curve class that extends the BaseCurve class.

    :ivar matrix: A 2D matrix that defines the transform to apply to the curve.

    Methods
    -------
    transform(matrix)
        Transforms this curve in 3D space.
    intersect_with_curve(curve)
        Intersects this curve with another curve in 3D space.
    interval()
        Returns the interval of the parameter values of the curve.

    .. method:: transform: Transforms this curve in 2D space.

        :param matrix: A 2D matrix that defines the transform to apply to the curve.
        :type matrix: Matrix2D

        :return: True if the transform was successful.
        :rtype: bool

    .. method:: interval: Returns the interval of the curve.

        :return: A tuple containing the start and end values of the interval.
        :rtype: tuple[float, float]

    .. method::  evaluator: Returns a CurveEvaluator2D object for the curve.

        :return: A CurveEvaluator2D object.
        :rtype: CurveEvaluator2D
    """

    def transform(self, matrix: Matrix2D) -> bool:
        """
        Transforms this curve in 2D space (inplace).
        :param matrix:  A 2D matrix that defines the transform to apply to the curve.
        :type matrix: Matrix2D
        :return : True if the transform was successful.
        :rtype : bool

        """
        return False

    def interval(self) -> tuple[float, float]:
        """
        Returns the interval as a tuple.

        :return: A tuple representing the curve interval as (start, end).
        :rtype: tuple[float, float]
        """
        return (0.0, 1.0)

    @property
    def evaluator(self) -> CurveEvaluator2D:
        return CurveEvaluator2D(self)


class Curve3D(BaseCurve):
    """
    :class:`Curve3D`
    ===============

    A class representing a curve in 3D space.

    Methods
    -------
    transform(matrix)
        Transforms this curve in 3D space.
    intersect_with_curve(curve)
        Intersects this curve with another curve in 3D space.
    interval()
        Returns the interval of the parameter values of the curve.

    Attributes
    ----------
    curve_type
        The type of geometry this curve represents.
    evaluator
        An evaluator object that lets you perform additional evaluations on the curve.

    Methods
    -------

    .. method:: transform(matrix)

        Transforms this curve in 3D space.

        :param matrix: A :class:`Matrix3D` object that defines the transform to apply to the curve.
        :type matrix: :class:`Matrix3D`
        :return: True if the transform was successful, False otherwise.
        :rtype: bool

    .. method:: intersect_with_curve(curve)

        Intersects this curve with another curve in 3D space.

        :param curve: A :class:`Curve3D` object representing the curve to intersect with.
        :type curve: :class:`Curve3D`
        :return: A :class:`ObjectCollection` containing the intersection points.
        :rtype: :class:`ObjectCollection`

    .. method:: interval()

        Returns the interval of the parameter values of the curve.

        :return: A tuple containing the start and end parameter values.
        :rtype: tuple[float, float]

    Attributes
    ----------

    .. attribute:: curve_type

        The type of geometry this curve represents.

        :type: :class:`Curve3DTypes`

    .. attribute:: evaluator

        An evaluator object that lets you perform additional evaluations on the curve.

        :type: :class:`CurveEvaluator3D`

    """

    def transform(self, matrix: Matrix3D) -> bool:
        """
        Transforms this curve in 3D space.
        matrix : A 3D matrix that defines the transform to apply to the curve.
        Return true if the transform was successful.
        """
        return bool()

    @property
    def curve_type(self) -> Curve3DTypes:
        """
        Returns the type of geometry this curve represents.
        """
        return Curve3DTypes()

    def intersect_with_curve(self, curve: Curve3D) -> ObjectCollection:
        """
        :param curve: The curve with which the object will be intersected.
        :return: An ObjectCollection containing the intersecting points on the curve.

        This method intersects the current object with the given curve. It returns the intersecting points on the curve as an ObjectCollection. If there are no intersecting points, an empty
        * ObjectCollection is returned.
        """
        res, dists = curve_intersect(self, curve, 0.01)
        if len(res) == 0:
            return ObjectCollection()
        else:
            return [(self.evaluate(i[0]), i) for i in res]

    def interval(self) -> tuple[float, float]:
        return (0.0, 1.0)

    @property
    def evaluator(self) -> CurveEvaluator3D:
        """
        Returns an evaluator object that lets you perform additional evaluations on the curve.
        """
        return CurveEvaluator3D(self)


class CurveEvaluator3D(Base):
    """
    A class for evaluating curves in 3D space.

    Args:
        curve (Curve3D): The curve to be evaluated.

    Attributes:
        _curve (Curve3D): The curve to be evaluated.
        _get_first_derivative (FDM): The function for getting the first derivative of the curve.
        _get_second_derivative (FDM): The function for getting the second derivative of the curve.
        _get_third_derivative (FDM): The function for getting the third derivative of the curve.

    Example:
        >>> from mmcore.api.curves import NurbsCurve3D
        >>> spl2=NurbsCurve3D(np.array([(-26030.187675027133, 5601.3871095975337, 31638.841094491760),
        ...                             (14918.717302595671, -25257.061306278192, 14455.443462719517),
        ...                             (19188.604482326708, 17583.891501540096, 6065.9078795798523),
        ...                             (-18663.729281923122, 5703.1869371495322, 0.0),
        ...                             (20028.126297559378, -20024.715164607202, 2591.0893519960955),
        ...                             (4735.5467668945130, 25720.651181520021, -6587.2644037490491),
        ...                             (-20484.795362315021, -11668.741154421798, -14201.431195298581),
        ...                             (18434.653814767291, -4810.2095985021788, -14052.951382291201),
        ...                             (612.94310080525793, 24446.695569574043, -24080.735343204549),
        ...                             (-7503.6320665111089, 2896.2190847052334, -31178.971042788111)]),degree=3)
        ...
        >>> evaluator=spl2.evaluator
        >>> success,point_at_param = evaluator.get_point_at_parameter(0.4)
        >>> print(f"Point at parameter {0.4}: {point_at_param}")
        Point at parameter 0.4: Point3D(x=6489.8139391188415, y=-10492.901024701423, z=16491.56350183732) at 0x3063358b0

        >>> success,(param_value, err) = evaluator.get_parameter_at_point(point_at_param)
        >>> print(f"Parameter at point {point_at_param}: ", param_value, err)
        Parameter at point Point3D(x=6489.8139391188415, y=-10492.901024701423, z=16491.56350183732) at 0x3063358b0:  0.4 1.8446678830370784e-06

        >>> 0.4==param_value
        True

    """

    def __init__(self, curve: Curve3D):
        super().__init__()
        self._curve = curve
        self._get_first_derivative = FDM(self._curve)
        self._get_second_derivative = FDM(self.get_first_derivative)
        self._get_third_derivative = FDM(self._get_second_derivative)

    def get_curvatures(
        self, parameters: list[float]
    ) -> tuple[bool, list[Vector3D], np.ndarray[float]]:
        """
        :param parameters: A list of float values representing the input parameters.
        :return: A tuple consisting of a boolean value indicating the success of the method, a list of Vector3D objects representing the curvatures, and a numpy ndarray of float values representing
        * the radius of the curvatures.
        """
        sucess1, D1 = self._get_first_derivative(parameters)
        sucess2, D2 = self._get_second_derivative(parameters)
        T, K, success = evaluate_curvature_vec(D1, D2)
        return np.all(success), [Vector3D(k) for k in K], norm(K)

    def get_curvature(self, parameter: float) -> tuple[bool, Vector3D, float]:
        """
        :param parameter: The parameter at which to evaluate the curvature
        :return: A tuple containing:
            - A boolean indicating whether the evaluation was successful
            - A Vector3D representing the curvature at the given parameter
            - A float representing the radius of curvature
        """
        sucess1, D1 = self._get_first_derivative(parameter)
        sucess2, D2 = self._get_second_derivative(parameter)
        T, K, sucess = evaluate_curvature(D1, D2)
        kv = Vector3D(K)

        return sucess, kv, kv.length

    def get_end_points(self) -> tuple[bool, Point3D, Point3D]:
        """
        Returns the start and end points of the parameter extents.

        :return: A tuple containing three elements:
                 - A boolean indicating the success of getting the parameter extents.
                 - The start point of the parameter extents as a Point3D object.
                 - The end point of the parameter extents as a Point3D object.
        """
        success, start, end = self.get_parameter_extents()
        return (
            True,
            self.get_point_at_parameter(start)[1],
            self.get_point_at_parameter(end)[1],
        )

    def get_length_at_parameter(
        self, fromParameter: float, toParameter: float
    ) -> tuple[bool, float]:
        """
        :param fromParameter: The starting parameter value.
        :param toParameter: The ending parameter value.
        :return: A tuple containing a boolean indicating the success status and the length calculated at the given parameter range.

        """
        res, err = evaluate_length(
            self._get_first_derivative, t0=fromParameter, t1=toParameter
        )
        return True, res

    def get_parameter_at_length(
        self, fromParameter: float, length: float
    ) -> tuple[bool, float]:
        """
        Get the parameter at a specified length along a given parameter axis.

        :param fromParameter: The starting parameter value.
        :type fromParameter: float
        :param length: The length along the parameter axis.
        :type length: float
        :return: A tuple containing a boolean indicating whether the operation was successful and the parameter value at the specified length.
        :rtype: tuple[bool, float]
        """

        start, end = self._curve.interval()
        return round(
            evaluate_parameter_from_length(
                self._get_first_derivative,
                length,
                t0=fromParameter,
                t1_limit=end,
                tol=TOLERANCE,
            ),
            ROUND_FACTOR,
        )

    def get_parameters_at_points(self, points: list[Point3D]) -> list[float]:
        """
        :param points: A list of Point3D objects representing the points at which to retrieve the parameters.
        :return: A list of float values representing the parameters corresponding to the given points.
        """
        result = np.zeros(len(points), dtype=float)
        for i, point in enumerate(points):
            t, dst = closest_point_on_curve(self._curve, point._array, tol=TOLERANCE)
            result[i] = t
        return np.round(result, ROUND_FACTOR)

    def get_parameter_at_point(
        self, point: Point3D
    ) -> tuple[bool, tuple[float, float]]:
        """
        Get the parameter position that correspond to a point on the curve.
        For reliable results, the point should lie on the curve within model tolerance.
        If the point does not lie on the curve, the parameter of the nearest point on the curve will generally be returned.
        point : The point to get the curve parameter value at.
        parameter : The output parameter position corresponding to the point.
        Returns parameter and distance
        """
        t, dst = closest_point_on_curve(self._curve, point._array, tol=TOLERANCE)
        t = round(t, ROUND_FACTOR)
        return True, (t, dst)

    def get_parameter_extents(self) -> tuple[bool, float, float]:
        """Get the extents of the parameter for the curve.

        :return: A tuple containing three values:
                 - A boolean indicating if the retrieval was successful (True) or not (False).
                 - The start value of the parameter.
                 - The end value of the parameter.
        :rtype: tuple[bool, float, float]
        """
        start, end = self._curve.interval()
        return True, start, end

    def get_point_at_parameters(
        self, parameters: list[float]
    ) -> tuple[bool, list[Point3D]]:
        """
        Get the points on the curve at the given parameters.

        :param parameters: A list of parameter values at which to evaluate the curve.
        :type parameters: list[float]
        :return: A tuple containing a boolean indicating success or failure,
                 and a list of Point3D objects representing the points on the curve at the given parameters.
        :rtype: tuple[bool, list[Point3D]]
        """
        return True, [Point3D(pt) for pt in self._curve(parameters)]

    def get_point_at_parameter(self, parameter: float) -> tuple[bool, Point3D]:
        """
        :param parameter: The parameter value for which to calculate the point on the curve
        :return: A tuple containing a boolean value indicating the success of the calculation and the resulting Point3D
        """
        return True, Point3D(self._curve(parameter))

    def get_first_derivatives(
        self, parameters: list[float]
    ) -> tuple[bool, list[Vector3D]]:
        """
        :param parameters: The list of float values representing the parameters for which first derivatives need to be computed.
        :return: A tuple containing a boolean value indicating if the computation was successful and a list of Vector3D objects representing the first derivatives for each parameter.
        """
        return True, [Vector3D(p) for p in self._get_first_derivative(parameters)]

    def get_first_derivative(self, parameter: float) -> tuple[bool, Vector3D]:
        """
        Calculate the first derivative.

        :param parameter: The value of the parameter.
        :return: A tuple containing a boolean value indicating the success of the calculation and a Vector3D object representing the first derivative.
        """
        return True, Vector3D(self._get_first_derivative(parameter))

    def get_second_derivatives(
        self, parameters: list[float]
    ) -> tuple[bool, list[Vector3D]]:
        """
        Calculate the second derivatives based on the given parameters.

        :param parameters: A list of floats representing the parameters.
        :return: A tuple containing a bool value indicating success and a list of Vector3D objects representing the second derivatives.
        """

        return True, [Vector3D(p) for p in self._get_second_derivative(parameters)]

    def get_second_derivative(self, parameter: float) -> tuple[bool, Vector3D]:
        """
        :param parameter: A float value representing the parameter.
        :return: A tuple consisting of a boolean value indicating whether the second derivative calculation was successful, and a Vector3D object representing the calculated second derivative
        *.
        """
        return True, Vector3D(self._get_second_derivative(parameter))

    def get_third_derivatives(
        self, parameters: list[float]
    ) -> tuple[bool, list[Vector3D]]:
        """
        :param parameters: The list of float values representing the parameters for computing the third derivatives.
        :return: A tuple containing a boolean value indicating the success of the operation and a list of Vector3D objects representing the computed third derivatives.
        """
        return True, [Vector3D(p) for p in self._get_third_derivative(parameters)]

    def get_third_derivative(self, parameter: float) -> tuple[bool, Vector3D]:
        """
        :param parameter: The input parameter for which the third derivative needs to be calculated.

        :return: A tuple containing a boolean value indicating the success of the calculation and the third derivative as a Vector3D object.
        """
        return True, Vector3D(self._get_third_derivative(parameter))

    def get_strokes(
        self, fromParameter: float, toParameter: float, count: int
    ) -> tuple[bool, list[Point3D]]:
        """
        :param fromParameter: The starting parameter value for generating the strokes.
        :type fromParameter: float

        :param toParameter: The ending parameter value for generating the strokes.
        :type toParameter: float

        :param count: The number of strokes to be generated between the given parameter values.
        :type count: int

        :return: A tuple containing a boolean value indicating the success status of the operation and a list of Point3D objects representing the generated strokes.
        :rtype: tuple[bool, list[Point3D]]
        """

        return True, [
            Point3D(pt)
            for pt in self._curve(np.linspace(fromParameter, toParameter, count))
        ]

    def get_tangents(self, parameters: list[float]) -> tuple[bool, list[Vector3D]]:
        """
        :param parameters: A list of floats representing the parameters used to evaluate the tangents.
        :return: A tuple containing a boolean value indicating the success of the evaluation and a list of Vector3D objects representing the computed tangents.
        """
        T, success = evaluate_tangent_vec(
            self._get_first_derivative(parameters),
            self._get_second_derivative(parameters),
        )

        return np.all(success), [Vector3D(v) for v in T]

    def get_tangent(self, parameter: float) -> tuple[bool, Vector3D]:
        """
        :param parameter: The input parameter for evaluating the tangent
        :return: A tuple containing a boolean value indicating the success of the evaluation and a Vector3D representing the tangent
        """
        T, success = evaluate_tangent(
            self._get_first_derivative(parameter),
            self._get_second_derivative(parameter),
        )
        return success, Vector3D(T)

    def get_aabb(self) -> BoundingBox3D:
        return BoundingBox3D.create_from_array(curve_aabb(self._curve, self._curve.interval(), TOLERANCE))


class CurveEvaluator2D(Base):
    """
    Returns true if the first derivative was successfully returned.
    """

    def __init__(self, curve: Curve2D):
        super().__init__()
        self._curve = curve
        self._get_first_derivative = FDM(self._curve)
        self._get_second_derivative = FDM(self._get_first_derivative)
        self._get_third_derivative = FDM(self._get_second_derivative)

    @classmethod
    def cast(cls, arg) -> CurveEvaluator2D:
        return CurveEvaluator2D()

    def get_curvatures(
        self, parameters: list[float]
    ) -> tuple[bool, list[Vector2D], list[float]]:
        """
        Get the curvature values at a number of parameter positions on the curve.
        parameters : The array of parameter positions to return curvature information at.
        Each parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        directions : The output array of the direction of the curvature at each position on the curve.
        The length of this array will be the same as the length of the parameters array provided.
        curvatures : The output array of the magnitude of the curvature at the position on the curve.
        The length of this array will be the same as the length of the parameters array provided.
        Returns true if the curvatures were successfully returned.
        """
        sucess1, D1 = self._get_first_derivative(parameters)
        sucess2, D2 = self._get_second_derivative(parameters)
        T, K, success = evaluate_curvature_vec(D1, D2)
        return np.all(success), [Vector2D(k) for k in K], norm(K)

    def get_curvature(self, parameter: float) -> tuple[bool, Vector2D, float]:
        """
        Get the curvature value at a parameter position on the curve.
        parameter : The parameter position to return the curvature information at.
        This value must be within the range of the parameter extents as provided by get_parameter_extents.
        direction : The output direction of the curvature at the position on the curve.
        curvature : The output magnitude of the curvature at the position on the curve.
        Returns true if the curvature was successfully returned.
        """
        sucess1, D1 = self._get_first_derivative(parameter)
        sucess2, D2 = self._get_second_derivative(parameter)
        T, K, sucess = evaluate_curvature(D1, D2)
        kv = Vector2D(K)

        return sucess, kv, kv.length

    def get_end_points(self) -> tuple[bool, Point2D, Point2D]:
        """
        Get the end points of the curve.
        startPoint : The output start point of the curve. If the curve is unbounded at the start, this value will be null.
        endPoint : The output end point of the curve. If the curve is unbounded at the end, this value will be null.
        Returns true if the end points were successfully returned.
        """
        success, start, end = self.get_parameter_extents()
        return (
            True,
            self.get_point_at_parameter(start)[1],
            self.get_point_at_parameter(end)[1],
        )

    def get_length_at_parameter(
        self, fromParameter: float, toParameter: float
    ) -> tuple[bool, float]:
        """
        Get the length of the curve between two parameter positions on the curve.
        fromParameter : The parameter position to measure the curve length from.
        This value must be within the range of the parameter extents as provided by get_parameter_extents.
        toParameter : The parameter position to measure the curve length to.
        This value must be within the range of the parameter extents as provided by get_parameter_extents.
        length : The output curve length between the from and to parameter positions on the curve.
        Returns true if the length was successfully returned.
        """
        res, err = evaluate_length(
            self._get_first_derivative, t0=fromParameter, t1=toParameter
        )
        return True, res

    def get_parameter_at_length(
        self, fromParameter: float, length: float
    ) -> tuple[bool, float]:
        """
        Get the parameter position on the curve that is the specified curve length from the specified starting parameter position.
        fromParameter : The parameter position to start measuring the curve length from.
        This value must be within the range of the parameter extents as provided by get_parameter_extents.
        length : The curve length to offset the from parameter by.
        A negative length value will offset in the negative parameter direction.
        parameter : The output parameter value that is the specified curve length from the starting parameter position.
        Returns true if the parameter was successfully returned.
        """
        raise NotImplemented

    def get_parameters_at_points(
        self, points: list[Point2D]
    ) -> tuple[bool, list[float]]:
        """
        Get the parameter positions that correspond to a set of points on the curve.
        For reliable results, the points should lie on the curve within model tolerance.
        If the points do not lie on the curve, the parameter of the nearest point on the curve will generally be returned.
        points : An array of points to get the curve parameter values at.
        parameters : The output array of parameter positions corresponding to the set of points.
        The length of this array will be equal to the length of the points array specified.
        Returns true if the parameters were successfully returned.
        """

        return [self.get_parameter_at_point(i)[1] for i in points]

    def get_parameter_at_point(
        self, point: Point2D
    ) -> tuple[bool, tuple[float, float]]:
        """
        Get the parameter position that correspond to a point on the curve.
        For reliable results, the point should lie on the curve within model tolerance.
        If the point does not lie on the curve, the parameter of the nearest point on the curve will generally be returned.
        point : The point to get the curve parameter value at.
        parameter : The output parameter position corresponding to the point.
        Returns parameter and distance
        """
        t, dst = closest_point_on_curve(self._curve, point._array, tol=TOLERANCE)
        t = round(t, int(abs(math.log10(TOLERANCE))))
        return True, (t, dst)

    def get_parameter_extents(self) -> tuple[bool, float, float]:
        """
        Get the parametric range of the curve.
        startParameter : The output lower bound of the parameter range.
        endParameter : The output upper bound of the parameter range.
        Returns true if the curve is bounded and the parameter extents were successfully returned.
        """
        start, end = self._curve.interval()
        return True, start, end

    def get_points_at_parameters(
        self, parameters: list[float]
    ) -> tuple[bool, list[Point2D]]:
        """
        Get the points on the curve that correspond to evaluating a set of parameter positions on the curve.
        parameters : The array of parameter positions to evaluate the curve position at.
        Each parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        points : The output array of curve positions corresponding to evaluating the curve at that parameter position.
        The length of this array will be equal to the length of the parameters array specified.
        Returns true if the points were successfully returned.
        """
        return True, [Point2D(pt) for pt in self._curve(parameters)]

    def get_point_at_parameter(self, parameter: float) -> tuple[bool, Point2D]:
        """
        Get the point on the curve that corresponds to evaluating a parameter position on the curve.
        parameter : The parameter position to evaluate the curve position at.
        The parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        point : The output curve position corresponding to evaluating the curve at that parameter position.
        Returns true if the point was successfully returned.
        """
        return True, Point2D(self._curve(parameter))

    def get_first_derivatives(
        self, parameters: list[float]
    ) -> tuple[bool, list[Vector2D]]:
        """
        Get the first derivatives of the curve at the specified parameter positions.
        parameters : The array of parameter positions to get the curve first derivative at.
        Each parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        firstDerivatives : The output array of first derivative vectors at each parameter position specified.
        The length of this array is equal to the length of the parameters array specified.
        Returns true if the first derivatives were successfully returned.
        """
        return True, [Vector2D(p) for p in self._get_first_derivative(parameters)]

    def getFirstDerivative(self, parameter: float) -> tuple[bool, Vector2D]:
        """
        Get the first derivative of the curve at the specified parameter position.
        parameter : The parameter position to get the curve first derivative at.
        The parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        firstDerivative : The output first derivative vector at the parameter position specified.
        Returns true if the first derivative was successfully returned.
        """
        return True, Vector2D(self._get_first_derivative(parameter))

    def get_second_derivatives(
        self, parameters: list[float]
    ) -> tuple[bool, list[Vector2D]]:
        """
        Get the second derivatives of the curve at the specified parameter positions.
        parameters : The array of parameter positions to get the curve second derivative at.
        Each parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        secondDerivatives : The output array of second derivative vectors at each parameter position specified.
        The length of this array is equal to the length of the parameters array specified.
        Returns true if the second derivatives were successfully returned.
        """

        return True, [Vector2D(p) for p in self._get_second_derivative(parameters)]

    def get_second_derivative(self, parameter: float) -> tuple[bool, Vector3D]:
        """
        Get the second derivative of the curve at the specified parameter position.
        parameter : The parameter position to get the curve second derivative at.
        The parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        secondDerivative : The output second derivative vector at the parameter position specified.
        Returns true if the second derivative was successfully returned.
        """
        return True, Vector2D(self._get_second_derivative(parameter))

    def get_third_derivatives(
        self, parameters: list[float]
    ) -> tuple[bool, list[Vector2D]]:
        """
        Get the third derivatives of the curve at the specified parameter positions.
        parameters : The array of parameter positions to get the curve third derivative at.
        Each parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        thirdDerivatives : The output array of third derivative vectors at each parameter position specified.
        The length of this array is equal to the length of the parameters array specified.
        Returns true if the third derivatives were successfully returned.
        """
        return True, [Vector2D(p) for p in self._get_third_derivative(parameters)]

    def get_third_derivative(self, parameter: float) -> tuple[bool, Vector2D]:
        """
        Get the third derivative of the curve at the specified parameter position.
        parameter : The parameter position to get the curve third derivative at.
        The parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        thirdDerivative : The output third derivative vector at the parameter position specified.
        Returns true if the third derivative was successfully returned.
        """
        return True, Vector2D(self._get_third_derivative(parameter))

    def get_strokes(
        self, fromParameter: float, toParameter: float, count: int
    ) -> tuple[bool, list[Point2D]]:
        """
        Get a sequence of points between two curve parameter positions.
        The points will be a linear interpolation along the curve between these two
        parameter positions where the maximum deviation between the curve and each line
        segment will not exceed the specified tolerance value.
        fromParameter : The starting parameter position to interpolate points from.
        The parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        toParameter : The ending parameter position to interpolate points to.
        The parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        tolerance : The maximum distance tolerance between the curve and the linear interpolation.
        vertexCoordinates : The output array of linear interpolation points.
        Returns true if the interpolation points were successfully returned.
        """

        return True, [
            Point2D(pt)
            for pt in self._curve(np.linspace(fromParameter, toParameter, count))
        ]

    def get_tangents(self, parameters: list[float]) -> tuple[bool, list[Vector3D]]:
        """
        Get the tangent to the curve at a number of parameter positions on the curve.
        parameters : The array of parameter positions to return the tangent at.
        Each parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        tangents : The output array of tangent vectors for each position on the curve.
        The length of this array will be the same as the length of the parameters array provided.
        Returns true if the tangents were successfully returned.
        """
        T, success = evaluate_tangent_vec(
            self._get_first_derivative(parameters),
            self._get_second_derivative(parameters),
        )

        return np.all(success), [Vector2D(v) for v in T]

    def get_tangent(self, parameter: float) -> tuple[bool, Vector3D]:
        """
        Get the tangent to the curve at a parameter position on the curve.
        parameter : The parameter position to return the tangent at.
        This value must be within the range of the parameter extents as provided by get_parameter_extents.
        tangent : The output tangent vector at the curve position.
        Returns true if the tangent was successfully returned.
        """
        T, success = evaluate_tangent(
            self._get_first_derivative(parameter),
            self._get_second_derivative(parameter),
        )
        return success, Vector2D(T)


class SurfaceEvaluator(Base):
    """
    Returns true if the point was successfully returned.
    """

    def __init__(self):
        pass

    @classmethod
    def cast(cls, arg) -> SurfaceEvaluator:
        return SurfaceEvaluator()

    def get_model_curve_from_parametric_curve(
        self, parametricCurve: Curve2D
    ) -> ObjectCollection:
        """
        Creates the 3D equivalent curve in model space, of a 2D curve defined in the
        parametric space of the surface.
        parametricCurve : The parameter space curve to map into this surface's parameter space.
        Returns an ObjectCollection containing one or more curves.
        When the SufaceEvaluatior is obtained from a face, and the curve cuts across internal
        boundaries of the face, multiple curves are returned. The returned curves are trimmed
        to the boundaries of the face.
        If the SurfaceEvaluator is obtained from a geometry object, a single curve returned
        because there are no boundaries with which to trim the curve.
        The type of curve(s) returned depends on the shape of the input curve and surface.
        """
        return ObjectCollection()

    def get_iso_curve(self, parameter: float, isUDirection: bool) -> ObjectCollection:
        """
        Gets (by extraction) a curve that follows a constant u or v parameter along the surface.
        The curve will have the same properties as the surface in the direction of the extraction.
        For example, when a curve is extracted from the periodic direction of a surface, the extracted
        curve will also be periodic.
        The type of curve returned is dependent on the shape the surface.
        parameter : The parameter at which to extract the curve
        isUDirection : A bool that indicates whether to extract the curve from the U or V direction
        Returns an ObjectCollection that contains one or more curves.
        Multiple curves are returned when the SurfaceEvaluator is obtained from a Face
        and the curve cuts across internal boundaries. The resulting curves are trimmed to the
        boundaries of the Face.
        When the SurfaceEvaluator is obtained from a geometry object, a single curve is returned
        because there are no boundaries to trim the curve.
        The type of curve(s) returned is dependent on the shape of the surface.
        """
        return ObjectCollection()

    def get_curvatures(
        self, parameters: list[Point2D]
    ) -> tuple[bool, list[Vector3D], list[float], list[float]]:
        """
        Get the curvature values at a number of parameter positions on the surface.
        parameters : The array of parameter positions to return curvature information at.
        Each parameter position must be with the range of the parameter extents as verified by isParameterOnFace.
        maxTangents : The output array of directions of maximum curvature at each position on the surface.
        The length of this array will be the same as the length of the parameters array provided.
        maxCurvatures : The output array of the magnitude of the maximum curvature at each position on the surface.
        The length of this array will be the same as the length of the parameters array provided.
        minCurvatures : The output array of the magnitude of the minimum curvature at each position on the surface.
        The minimum curvature direction is perpendicular to the maximum curvature tangent directions.
        The length of this array will be the same as the length of the parameters array provided.
        Returns true if the curvatures were successfully returned.
        """
        return (bool(), [Vector3D()], [float()], [float()])

    def get_curvature(self, parameter: Point2D) -> tuple[bool, Vector3D, float, float]:
        """
        Get the curvature values at a parameter positions on the surface.
        parameter : The parameter positions to return curvature information at.
        maxTangent : The output directions of maximum curvature at the position on the surface.
        maxCurvature : The output magnitude of the maximum curvature at the position on the surface.
        minCurvature : The output magnitude of the minimum curvature at the position on the surface.
        The minimum curvature direction is perpendicular to the maximum curvature tangent directions.
        Returns true if the curvature was successfully returned.
        """
        return (bool(), Vector3D(), float(), float())

    def get_normals_at_parameters(
        self, parameters: list[Point2D]
    ) -> tuple[bool, list[Vector3D]]:
        """
        Gets the surface normal at a number of parameter positions on the surface.
        parameters : The array of parameter positions to return the normal at.
        Each parameter position must be with the range of the parameter extents as verified by isParameterOnFace.
        normals : The output array of normals for each parameter position on the surface.
        The length of this array will be the same as the length of the parameters array provided.
        Returns true if the normals were successfully returned.
        """
        return (bool(), [Vector3D()])

    def get_normal_at_parameter(self, parameter: Point2D) -> tuple[bool, Vector3D]:
        """
        Gets the surface normal at a parameter position on the surface.
        parameter : The parameter position to return the normal at.
        The parameter position must be with the range of the parameter extents as verified by isParameterOnFace.
        normal : The output normal for the parameter position on the surface.
        Returns true if the normal was successfully returned.
        """
        return (bool(), Vector3D())

    def get_normals_at_points(
        self, points: list[Point3D]
    ) -> tuple[bool, list[Vector3D]]:
        """
        Gets the surface normal at a number of positions on the surface.
        points : The array of points to return the normal at.
        For reliable results each point should lie on the surface.
        normals : The output array of normals for each point on the surface.
        The length of this array will be the same as the length of the points array provided.
        Returns true if the normals were successfully returned.
        """
        return (bool(), [Vector3D()])

    def get_normal_at_point(self, point: Point3D) -> tuple[bool, Vector3D]:
        """
        Gets the surface normal at a point on the surface.
        point : The point to return the normal at.
        For reliable results the point should lie on the surface.
        normal : The output normal for the point on the surface.
        Returns true if the normal was successfully returned.
        """
        return (bool(), Vector3D())

    def get_parameters_at_points(
        self, points: list[Point3D]
    ) -> tuple[bool, list[Point2D]]:
        """
        Get the parameter positions that correspond to a set of points on the surface.
        For reliable results, the points should lie on the surface within model tolerance.
        If the points do not lie on the surface, the parameter of the nearest point on the surface will generally be returned.
        points : An array of points to get the surface parameter values at.
        parameters : The output array of parameter positions corresponding to the set of points.
        The length of this array will be equal to the length of the points array specified.
        Returns true if the parameters were successfully returned.
        """
        return (bool(), [Point2D()])

    def get_parameter_at_point(self, point: Point3D) -> tuple[bool, Point2D]:
        """
        Get the parameter position that correspond to a point on the surface.
        For reliable results, the point should lie on the surface within model tolerance.
        If the point does not lie on the surface, the parameter of the nearest point on the surface will generally be returned.
        point : The point to get the curve parameter value at.
        parameter : The output parameter position corresponding to the point.
        Returns true of the parameter was successfully returned.
        """
        return (bool(), Point2D())

    def get_points_at_parameters(
        self, parameters: list[Point2D]
    ) -> tuple[bool, list[Point3D]]:
        """
        Get the points on the surface that correspond to evaluating a set of parameter positions on the surface.
        parameters : The array of parameter positions to evaluate the surface position at.
        Each parameter position must be within the range of the parameter extents as verified by isParameterOnFace.
        points : The output array of points corresponding to evaluating the curve at that parameter position.
        The length of this array will be equal to the length of the parameters array specified.
        Returns true if the points were successfully returned.
        """
        return (bool(), [Point3D()])

    def get_point_at_parameter(self, parameter: Point2D) -> tuple[bool, Point3D]:
        """
        Get the point on the surface that correspond to evaluating a parameter position on the surface.
        parameter : The parameter positions to evaluate the surface position at.
        The parameter position must be within the range of the parameter extents as verified by isParameterOnFace.
        point : The output point corresponding to evaluating the curve at that parameter position.
        Returns true if the point was successfully returned.
        """
        return (bool(), Point3D())

    def get_param_anomaly(
        self,
    ) -> tuple[bool, list[float], list[float], list[float], list[float], list[bool]]:
        """
        Gets details about anomalies in parameter space of the surface.
        This includes information about periodic intervals, singularities, or unbounded parameter ranges.
        periodicityU : The output array with information about the period of the surface in U.
        periodicityU[0] will contain the period of the surface in U.
        If periodicityU[0] is 0, the surface is not periodic in U.
        If the surface is periodic in U, peridocityU[1] will contain the parameter value at the start of the principle period.
        periodicityV : The output array with information about the period of the surface in V.
        periodicityV[0] will contain the period of the surface in V.
        If periodicityV[0] is 0, the surface is not periodic in V.
        If the surface is periodic in V, peridocityV[1] will contain the parameter value at the start of the principle period.
        singularitiesU : The output array parameter values of singularities in U.
        If this array is empty, there are no singularities in U.
        singularitiesV : The output array parameter values of singularities in V.
        If this array is empty, there are no singularities in V.
        unboundedParameters : The output array that indicates if the parameter range is unbounded in U or V.
        unboundedParameters[0] will be true if U is unbounded.
        unboundedParameters[1] will be true if V is unbounded.
        Returns true if the parameter anomalies were successfully returned.
        """
        return (bool(), [float()], [float()], [float()], [float()], [bool()])

    def get_first_derivatives(
        self, parameters: list[Point2D]
    ) -> tuple[bool, list[Vector3D], list[Vector3D]]:
        """
        Get the first derivatives of the surface at the specified parameter positions.
        parameters : The array of parameter positions to get the surface first derivative at.
        Each parameter position must be within the range of the parameter extents as verified by isParameterOnFace.
        partialsU : The output array of first derivative U partial vectors at each parameter position specified.
        The length of this array is equal to the length of the parameters array specified.
        partialsV : The output array of first derivative V partial vectors at each parameter position specified.
        The length of this array is equal to the length of the parameters array specified.
        Returns true if the first derivatives were successfully returned.
        """
        return (bool(), [Vector3D()], [Vector3D()])

    def get_first_derivative(
        self, parameter: Point2D
    ) -> tuple[bool, Vector3D, Vector3D]:
        """
        Get the first derivative of the surface at the specified parameter position.
        parameter : The parameter positions to get the surface first derivative at.
        The parameter position must be within the range of the parameter extents as verified by isParameterOnFace.
        partialU : The output first derivative U partial vector at the parameter position specified.
        partialV : The output first derivative V partial vector at the parameter position specified.
        Returns true if the first derivative was successfully returned.
        """
        return (bool(), Vector3D(), Vector3D())

    def get_second_derivatives(
        self, parameters: list[Point2D]
    ) -> tuple[bool, list[Vector3D], list[Vector3D], list[Vector3D]]:
        """
        Get the second derivatives of the surface at the specified parameter positions.
        parameters : The array of parameter positions to get the surface second derivative at.
        Each parameter position must be within the range of the parameter extents as verified by isParameterOnFace.
        partialsUU : The output array of second derivative UU partial vectors at each parameter position specified.
        The length of this array is equal to the length of the parameters array specified.
        partialsUV : The output array of second derivative UV mixed partial vectors at each parameter position specified.
        The length of this array is equal to the length of the parameters array specified.
        partialsVV : The output array of second derivative VV partial vectors at each parameter position specified.
        The length of this array is equal to the length of the parameters array specified.
        Returns true if the second derivatives were successfully returned.
        """
        return (bool(), [Vector3D()], [Vector3D()], [Vector3D()])

    def get_second_derivative(
        self, parameter: Point2D
    ) -> tuple[bool, Vector3D, Vector3D, Vector3D]:
        """
        Get the second derivative of the surface at the specified parameter position.
        parameter : The parameter position to get the surface second derivative at.
        The parameter position must be within the range of the parameter extents as verified by isParameterOnFace.
        partialUU : The output second derivative UU partial vector at each parameter position specified.
        partialUV : The output second derivative UV mixed partial vector at each parameter position specified.
        partialVV : The output second derivative VV partial vector at each parameter position specified.
        Returns true if the second derivative was successfully returned.
        """
        return (bool(), Vector3D(), Vector3D(), Vector3D())

    def get_third_derivatives(
        self, parameters: list[Point2D]
    ) -> tuple[bool, list[Vector3D], list[Vector3D]]:
        """
        Get the third derivatives of the surface at the specified parameter positions.
        parameters : The array of parameter positions to get the surface third derivative at.
        Each parameter position must be within the range of the parameter extents as verified by isParameterOnFace.
        partialsUUU : The output array of third derivative UUU partial vectors at each parameter position specified.
        The length of this array is equal to the length of the parameters array specified.
        partialsVVV : The output array of third derivative VVV partial vectors at each parameter position specified.
        The length of this array is equal to the length of the parameters array specified.
        Returns true if the third derivatives were successfully returned.
        """
        return (bool(), [Vector3D()], [Vector3D()])

    def get_third_derivative(
        self, parameter: Point2D
    ) -> tuple[bool, Vector3D, Vector3D]:
        """
        Get the third derivative of the surface at the specified parameter position.
        parameter : The parameter position to get the surface third derivative at.
        The parameter position must be within the range of the parameter extents as verified by isParameterOnFace.
        partialUUU : The output third derivative UUU partial vector at each parameter position specified.
        partialVVV : The output third derivative VVV partial vector at each parameter position specified.
        Returns true if the third derivative was successfully returned.
        """
        return (bool(), Vector3D(), Vector3D())

    def is_parameter_on_face(self, parameter: Point2D) -> bool:
        """
        Determines if the specified parameter position lies with the parametric range of the surface.
        parameter : The parameter position to test.
        Returns true if the parameter position lies within the valid parametric range of the surface.
        """
        return bool()

    def parametric_range(self) -> BoundingBox2D:
        """
        Returns the parametric range of the surface.
        If the surface is periodic in a direction, the range is set to the principle period's range.
        If the surface is only upper bounded in a direction, the lower bound is set to -double-max.
        If the surface is only lower bounded in a direction, the upper bound is set to double-max.
        If the surface is unbounded in a direction, the lower bound and upper bound of the range will both be zero.
        Returns the bounding box with the parameter extents, with the X value being the U range, and the Y value being the V range.
        """
        return BoundingBox2D()

    @property
    def is_closed_in_u(self) -> bool:
        """
        Indicates if the surface is closed (forms a loop) in the U direction
        """
        return bool()

    @property
    def is_closed_in_v(self) -> bool:
        """
        Indicates if the surface is closed (forms a loop) in the V direction
        """
        return bool()

    @property
    def area(self) -> float:
        """
        Returns the area of the surface. This is typically used when the SurfaceEvaluator is associated
        with a BRepFace object where it is always valid. This can fail in the case where the SurfaceEvaluator is
        associated with one of the geometry classes, (Plane, Cylinder, Cone, EllipticalCone, or EllipticalCylinder
        object), because these surfaces are unbounded. A BRepFace, even one of these shapes, is bounded by its
        edges and has a well-defined area.
        """
        return float()

from __future__ import annotations

from typing import Union

import numpy as np

from mmcore.api._base import Base, ObjectCollection
from mmcore.api.bbox import BoundingBox2D
from mmcore.api.enums import SurfaceTypes, Curve2DTypes, Curve3DTypes
from mmcore.api.vectors import Matrix3D, Matrix2D, Vector3D, Point3D, Vector2D, Point2D


class Surface(Base):
    """
    Describes a two-dimensional topological, manifold in three-dimensional space.
    It is used as the underlying geometry for a BRepFace.
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


class Curve2D(Base):
    """
    The base class for all 2D geometry classes.
    """

    def transform(self, matrix: Matrix2D) -> bool:
        """
        Transforms this curve in 2D space.
        matrix : A 2D matrix that defines the transform to apply to the curve.
        Return true if the transform was successful.
        """
        return bool()

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
        return CurveEvaluator2D()


class Curve3D(Base):
    """
    The base class for all 3D geometry classes.
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

    @property
    def evaluator(self) -> CurveEvaluator3D:
        """
        Returns an evaluator object that lets you perform additional evaluations on the curve.
        """
        return CurveEvaluator3D()

    def evaluate_param(self, t: float) -> Point3D: ...

    def evaluate_params(self, t: Union[np.ndarray, list[float]]) -> Union[list[Point3D], np.ndarray[Point3D]]: ...


class CurveEvaluator3D(Base):
    """
    3D curve evaluator that is obtained from a curve and allows you to perform
    various evaluations on the curve.
    """

    def get_curvatures(self, parameters: list[float]) -> tuple[bool, list[Vector3D], list[float]]:
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
        return (bool(), [Vector3D()], [float()])

    def get_curvature(self, parameter: float) -> tuple[bool, Vector3D, float]:
        """
        Get the curvature value at a parameter position on the curve.
        parameter : The parameter position to return the curvature information at.
        This value must be within the range of the parameter extents as provided by get_parameter_extents.
        direction : The output direction of the curvature at the position on the curve.
        curvature : The output magnitude of the curvature at the position on the curve.
        Returns true if the curvature was successfully returned.
        """
        return (bool(), Vector3D(), float())

    def get_end_points(self) -> tuple[bool, Point3D, Point3D]:
        """
        Get the end points of the curve.
        startPoint : The output start point of the curve. If the curve is unbounded at the start, this value will be null.
        endPoint : The output end point of the curve. If the curve is unbounded at the end, this value will be null.
        Returns true if the end points were successfully returned.
        """
        return (bool(), Point3D(), Point3D())

    def get_length_at_parameter(self, fromParameter: float, toParameter: float) -> tuple[bool, float]:
        """
        Get the length of the curve between two parameter positions on the curve.
        fromParameter : The parameter position to measure the curve length from.
        This value must be within the range of the parameter extents as provided by get_parameter_extents.
        toParameter : The parameter position to measure the curve length to.
        This value must be within the range of the parameter extents as provided by get_parameter_extents.
        length : The output curve length between the from and to parameter positions on the curve.
        Returns true if the length was successfully returned.
        """
        return (bool(), float())

    def get_parameter_at_length(self, fromParameter: float, length: float) -> tuple[bool, float]:
        """
        Get the parameter position on the curve that is the specified curve length from the specified starting parameter position.
        fromParameter : The parameter position to start measuring the curve length from.
        This value must be within the range of the parameter extents as provided by get_parameter_extents.
        length : The curve length to offset the from parameter by.
        A negative length value will offset in the negative parameter direction.
        parameter : The output parameter value that is the specified curve length from the starting parameter position.
        Returns true if the parameter was successfully returned.
        """
        return (bool(), float())

    def get_parameters_at_points(self, points: list[Point3D]) -> tuple[bool, list[float]]:
        """
        Get the parameter positions that correspond to a set of points on the curve.
        For reliable results, the points should lie on the curve within model tolerance.
        If the points do not lie on the curve, the parameter of the nearest point on the curve will generally be returned.
        points : An array of points to get the curve parameter values at.
        parameters : The output array of parameter positions corresponding to the set of points.
        The length of this array will be equal to the length of the points array specified.
        Returns true if the parameters were successfully returned.
        """
        return (bool(), [float()])

    def get_parameter_at_points(self, point: Point3D) -> tuple[bool, float]:
        """
        Get the parameter position that correspond to a point on the curve.
        For reliable results, the point should lie on the curve within model tolerance.
        If the point does not lie on the curve, the parameter of the nearest point on the curve will generally be returned.
        point : The point to get the curve parameter value at.
        parameter : The output parameter position corresponding to the point.
        Returns true of the parameter was successfully returned.
        """
        return (bool(), float())

    def get_parameter_extents(self) -> tuple[bool, float, float]:
        """
        Get the parametric range of the curve.
        startParameter : The output lower bound of the parameter range.
        endParameter : The output upper bound of the parameter range.
        Returns true if the curve is bounded and the parameter extents were successfully returned.
        """
        return (bool(), float(), float())

    def get_points_at_parameters(self, parameters: list[float]) -> tuple[bool, list[Point3D]]:
        """
        Get the points on the curve that correspond to evaluating a set of parameter positions on the curve.
        parameters : The array of parameter positions to evaluate the curve position at.
        Each parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        points : The output array of curve positions corresponding to evaluating the curve at that parameter position.
        The length of this array will be equal to the length of the parameters array specified.
        Returns true if the points were successfully returned.
        """
        return (bool(), [Point3D()])

    def get_point_at_parameter(self, parameter: float) -> tuple[bool, Point3D]:
        """
        Get the point on the curve that corresponds to evaluating a parameter position on the curve.
        parameter : The parameter position to evaluate the curve position at.
        The parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        point : The output curve position corresponding to evaluating the curve at that parameter position.
        Returns true if the point was successfully returned.
        """
        return (bool(), Point3D())

    def get_first_derivatives(self, parameters: list[float]) -> tuple[bool, list[Vector3D]]:
        """
        Get the first derivatives of the curve at the specified parameter positions.
        parameters : The array of parameter positions to get the curve first derivative at.
        Each parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        firstDerivatives : The output array of first derivative vectors at each parameter position specified.
        The length of this array is equal to the length of the parameters array specified.
        Returns true if the first derivatives were successfully returned.
        """
        return (bool(), [Vector3D()])

    def get_first_derivative(self, parameter: float) -> tuple[bool, Vector3D]:
        """
        Get the first derivative of the curve at the specified parameter position.
        parameter : The parameter position to get the curve first derivative at.
        The parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        firstDerivative : The output first derivative vector at the parameter position specified.
        Returns true if the first derivative was successfully returned.
        """
        return (bool(), Vector3D())

    def get_second_derivatives(self, parameters: list[float]) -> tuple[bool, list[Vector3D]]:
        """
        Get the second derivatives of the curve at the specified parameter positions.
        parameters : The array of parameter positions to get the curve second derivative at.
        Each parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        secondDerivatives : The output array of second derivative vectors at each parameter position specified.
        The length of this array is equal to the length of the parameters array specified.
        Returns true if the second derivatives were successfully returned.
        """
        return (bool(), [Vector3D()])

    def get_second_derivative(self, parameter: float) -> tuple[bool, Vector3D]:
        """
        Get the second derivative of the curve at the specified parameter position.
        parameter : The parameter position to get the curve second derivative at.
        The parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        secondDerivative : The output second derivative vector at the parameter position specified.
        Returns true if the second derivative was successfully returned.
        """
        return (bool(), Vector3D())

    def get_third_derivatives(self, parameters: list[float]) -> tuple[bool, list[Vector3D]]:
        """
        Get the third derivatives of the curve at the specified parameter positions.
        parameters : The array of parameter positions to get the curve third derivative at.
        Each parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        thirdDerivatives : The output array of third derivative vectors at each parameter position specified.
        The length of this array is equal to the length of the parameters array specified.
        Returns true if the third derivatives were successfully returned.
        """
        return (bool(), [Vector3D()])

    def get_third_derivative(self, parameter: float) -> tuple[bool, Vector3D]:
        """
        Get the third derivative of the curve at the specified parameter position.
        parameter : The parameter position to get the curve third derivative at.
        The parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        thirdDerivative : The output third derivative vector at the parameter position specified.
        Returns true if the third derivative was successfully returned.
        """
        return (bool(), Vector3D())

    def get_strokes(self, fromParameter: float, toParameter: float, tolerance: float) -> tuple[bool, list[Point3D]]:
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
        return (bool(), [Point3D()])

    def get_tangents(self, parameters: list[float]) -> tuple[bool, list[Vector3D]]:
        """
        Get the tangent to the curve at a number of parameter positions on the curve.
        parameters : The array of parameter positions to return the tangent at.
        Each parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        tangents : The output array of tangent vectors for each position on the curve.
        The length of this array will be the same as the length of the parameters array provided.
        Returns true if the tangents were successfully returned.
        """
        return (bool(), [Vector3D()])

    def get_tangent(self, parameter: float) -> tuple[bool, Vector3D]:
        """
        Get the tangent to the curve at a parameter position on the curve.
        parameter : The parameter position to return the tangent at.
        This value must be within the range of the parameter extents as provided by get_parameter_extents.
        tangent : The output tangent vector at the curve position.
        Returns true if the tangent was successfully returned.
        """
        return (bool(), Vector3D())


class CurveEvaluator2D(Base):
    """
    2D curve evaluator that is obtained from a curve and allows you to perform
    various evaluations on the curve.
    """

    def __init__(self):
        pass

    @classmethod
    def cast(cls, arg) -> CurveEvaluator2D:
        return CurveEvaluator2D()

    def get_curvatures(self, parameters: list[float]) -> tuple[bool, list[Vector2D], list[float]]:
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
        return (bool(), [Vector2D()], [float()])

    def get_curvature(self, parameter: float) -> tuple[bool, Vector2D, float]:
        """
        Get the curvature value at a parameter position on the curve.
        parameter : The parameter position to return the curvature information at.
        This value must be within the range of the parameter extents as provided by get_parameter_extents.
        direction : The output direction of the curvature at the position on the curve.
        curvature : The output magnitude of the curvature at the position on the curve.
        Returns true if the curvature was successfully returned.
        """
        return (bool(), Vector2D(), float())

    def get_tangents(self, parameters: list[float]) -> tuple[bool, list[Vector2D]]:
        """
        Get the tangent to the curve at a number of parameter positions on the curve.
        parameters : The array of parameter positions to return the tangent at.
        Each parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        tangents : The output array of tangent vectors for each position on the curve.
        The length of this array will be the same as the length of the parameters array provided.
        Returns true if the tangents were successfully returned.
        """
        return (bool(), [Vector2D()])

    def get_tangent(self, parameter: float) -> tuple[bool, Vector2D]:
        """
        Get the tangent to the curve at a parameter position on the curve.
        parameter : The parameter position to return the tangent at.
        This value must be within the range of the parameter extents as provided by get_parameter_extents.
        tangent : The output tangent vector at the curve position.
        Returns true if the tangent was successfully returned.
        """
        return (bool(), Vector2D())

    def get_end_points(self) -> tuple[bool, Point2D, Point2D]:
        """
        Get the end points of the curve.
        startPoint : The output start point of the curve. If the curve is unbounded at the start, this value will be null.
        endPoint : The output end point of the curve. If the curve is unbounded at the end, this value will be null.
        Returns true if the end points were successfully returned.
        """
        return (bool(), Point2D(), Point2D())

    def get_length_at_parameter(self, fromParameter: float, toParameter: float) -> tuple[bool, float]:
        """
        Get the length of the curve between two parameter positions on the curve.
        fromParameter : The parameter position to measure the curve length from.
        This value must be within the range of the parameter extents as provided by get_parameter_extents.
        toParameter : The parameter position to measure the curve length to.
        This value must be within the range of the parameter extents as provided by get_parameter_extents.
        length : The output curve length between the from and to parameter positions on the curve.
        Returns true if the length was successfully returned.
        """
        return (bool(), float())

    def get_parameter_at_length(self, fromParameter: float, length: float) -> tuple[bool, float]:
        """
        Get the parameter position on the curve that is the specified curve length from the specified starting parameter position.
        fromParameter : The parameter position to start measuring the curve length from.
        This value must be within the range of the parameter extents as provided by get_parameter_extents.
        length : The curve length to offset the from parameter by.
        A negative length value will offset in the negative parameter direction.
        parameter : The output parameter value that is the specified curve length from the starting parameter position.
        Returns true if the parameter was successfully returned.
        """
        return (bool(), float())

    def get_parameters_at_points(self, points: list[Point2D]) -> tuple[bool, list[float]]:
        """
        Get the parameter positions that correspond to a set of points on the curve.
        For reliable results, the points should lie on the curve within model tolerance.
        If the points do not lie on the curve, the parameter of the nearest point on the curve will generally be returned.
        points : An array of points to get the curve parameter values at.
        parameters : The output array of parameter positions corresponding to the set of points.
        The length of this array will be equal to the length of the points array specified.
        Returns true if the parameters were successfully returned.
        """
        return (bool(), [float()])

    def get_parameter_at_points(self, point: Point2D) -> tuple[bool, float]:
        """
        Get the parameter position that correspond to a point on the curve.
        For reliable results, the point should lie on the curve within model tolerance.
        If the point does not lie on the curve, the parameter of the nearest point on the curve will generally be returned.
        point : The point to get the curve parameter value at.
        parameter : The output parameter position corresponding to the point.
        Returns true of the parameter was successfully returned.
        """
        return (bool(), float())

    def get_parameter_extents(self) -> tuple[bool, float, float]:
        """
        Get the parametric range of the curve.
        startParameter : The output lower bound of the parameter range.
        endParameter : The output upper bound of the parameter range.
        Returns true if the curve is bounded and the parameter extents were successfully returned.
        """
        return (bool(), float(), float())

    def get_points_at_parameters(self, parameters: list[float]) -> tuple[bool, list[Point2D]]:
        """
        Get the points on the curve that correspond to evaluating a set of parameter positions on the curve.
        parameters : The array of parameter positions to evaluate the curve position at.
        Each parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        points : The output array of curve positions corresponding to evaluating the curve at that parameter position.
        The length of this array will be equal to the length of the parameters array specified.
        Returns true if the points were successfully returned.
        """
        return (bool(), [Point2D()])

    def get_point_at_parameter(self, parameter: float) -> tuple[bool, Point2D]:
        """
        Get the point on the curve that corresponds to evaluating a parameter position on the curve.
        parameter : The parameter position to evaluate the curve position at.
        The parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        point : The output curve position corresponding to evaluating the curve at that parameter position.
        Returns true if the point was successfully returned.
        """
        return (bool(), Point2D())

    def get_first_derivatives(self, parameters: list[float]) -> tuple[bool, list[Vector2D]]:
        """
        Get the first derivatives of the curve at the specified parameter positions.
        parameters : The array of parameter positions to get the curve first derivative at.
        Each parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        firstDerivatives : The output array of first derivative vectors at each parameter position specified.
        The length of this array is equal to the length of the parameters array specified.
        Returns true if the first derivatives were successfully returned.
        """
        return (bool(), [Vector2D()])

    def getFirstDerivative(self, parameter: float) -> tuple[bool, Vector2D]:
        """
        Get the first derivative of the curve at the specified parameter position.
        parameter : The parameter position to get the curve first derivative at.
        The parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        firstDerivative : The output first derivative vector at the parameter position specified.
        Returns true if the first derivative was successfully returned.
        """
        return (bool(), Vector2D())

    def get_second_derivatives(self, parameters: list[float]) -> tuple[bool, list[Vector2D]]:
        """
        Get the second derivatives of the curve at the specified parameter positions.
        parameters : The array of parameter positions to get the curve second derivative at.
        Each parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        secondDerivatives : The output array of second derivative vectors at each parameter position specified.
        The length of this array is equal to the length of the parameters array specified.
        Returns true if the second derivatives were successfully returned.
        """
        return (bool(), [Vector2D()])

    def get_second_derivative(self, parameter: float) -> tuple[bool, Vector2D]:
        """
        Get the second derivative of the curve at the specified parameter position.
        parameter : The parameter position to get the curve second derivative at.
        The parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        secondDerivative : The output second derivative vector at the parameter position specified.
        Returns true if the second derivative was successfully returned.
        """
        return (bool(), Vector2D())

    def get_third_derivatives(self, parameters: list[float]) -> tuple[bool, list[Vector2D]]:
        """
        Get the third derivatives of the curve at the specified parameter positions.
        parameters : The array of parameter positions to get the curve third derivative at.
        Each parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        thirdDerivatives : The output array of third derivative vectors at each parameter position specified.
        The length of this array is equal to the length of the parameters array specified.
        Returns true if the third derivatives were successfully returned.
        """
        return (bool(), [Vector2D()])

    def get_third_derivative(self, parameter: float) -> tuple[bool, Vector2D]:
        """
        Get the third derivative of the curve at the specified parameter position.
        parameter : The parameter position to get the curve third derivative at.
        The parameter value must be within the range of the parameter extents as provided by get_parameter_extents.
        thirdDerivative : The output third derivative vector at the parameter position specified.
        Returns true if the third derivative was successfully returned.
        """
        return (bool(), Vector2D())

    def get_strokes(self, fromParameter: float, toParameter: float, tolerance: float) -> tuple[bool, list[Point2D]]:
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
        return (bool(), [Point2D()])


class SurfaceEvaluator(Base):
    """
    Surface evaluator that is obtained from a surface and allows you to perform
    various evaluations on the surface.
    """

    def __init__(self):
        pass

    @staticmethod
    def cast(arg) -> SurfaceEvaluator:
        return SurfaceEvaluator()

    def get_model_curve_from_parametric_curve(self, parametricCurve: Curve2D) -> ObjectCollection:
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

    def get_curvatures(self, parameters: list[Point2D]) -> tuple[bool, list[Vector3D], list[float], list[float]]:
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

    def get_normals_at_parameters(self, parameters: list[Point2D]) -> tuple[bool, list[Vector3D]]:
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

    def get_normals_at_points(self, points: list[Point3D]) -> tuple[bool, list[Vector3D]]:
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

    def get_parameters_at_points(self, points: list[Point3D]) -> tuple[bool, list[Point2D]]:
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

    def get_points_at_parameters(self, parameters: list[Point2D]) -> tuple[bool, list[Point3D]]:
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

    def get_param_anomaly(self) -> tuple[bool, list[float], list[float], list[float], list[float], list[bool]]:
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

    def get_first_derivatives(self, parameters: list[Point2D]) -> tuple[bool, list[Vector3D], list[Vector3D]]:
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

    def get_first_derivative(self, parameter: Point2D) -> tuple[bool, Vector3D, Vector3D]:
        """
        Get the first derivative of the surface at the specified parameter position.
        parameter : The parameter positions to get the surface first derivative at.
        The parameter position must be within the range of the parameter extents as verified by isParameterOnFace.
        partialU : The output first derivative U partial vector at the parameter position specified.
        partialV : The output first derivative V partial vector at the parameter position specified.
        Returns true if the first derivative was successfully returned.
        """
        return (bool(), Vector3D(), Vector3D())

    def get_second_derivatives(self, parameters: list[Point2D]) -> tuple[
        bool, list[Vector3D], list[Vector3D], list[Vector3D]]:
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

    def get_second_derivative(self, parameter: Point2D) -> tuple[bool, Vector3D, Vector3D, Vector3D]:
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

    def get_third_derivatives(self, parameters: list[Point2D]) -> tuple[bool, list[Vector3D], list[Vector3D]]:
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

    def get_third_derivative(self, parameter: Point2D) -> tuple[bool, Vector3D, Vector3D]:
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

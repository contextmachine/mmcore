from __future__ import annotations

from abc import ABCMeta, abstractmethod
from typing import TypeVar


from numpy import ndarray

from ._typing import Union

import numpy as np
from numpy._typing import ArrayLike

from mmcore.api._base import Base
from mmcore.api.base import Curve2D, Curve3D, Surface, BaseCurve
from mmcore.api._base import ObjectCollection

from mmcore.api.vectors import Point2D, Point3D, Vector3D, Vector2D

from mmcore.geom.vec import *

__all__ = [
    "Curve2D",
    "Curve3D",
    "Line2D",
    "Line3D",
    "InfiniteLine3D",
    "Circle2D",
    "Circle3D",
    "NurbsCurve2D",
    "NurbsCurve3D",
    "OffsetCurve",
    "SubCurve"

]


def line_line_tsz(la, lb):
    (x1, y1, z1), (x2, y2, z2) = la
    (x3, y3, z3), (x4, y4, z4) = lb
    c1 = x1 * y3 - x1 * y4 - x2 * y3 + x2 * y4 - x3 * y1 + x3 * y2 + x4 * y1 - x4 * y2
    s = (x1 * y3 - x1 * y4 - x3 * y1 + x3 * y4 + x4 * y1 - x4 * y3) / c1

    t = -(x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2) / c1
    z0 = -(
        x1 * y2 * z3
        - x1 * y2 * z4
        - x1 * y3 * z2
        + x1 * y3 * z4
        + x1 * y4 * z2
        - x1 * y4 * z3
        - x2 * y1 * z3
        + x2 * y1 * z4
        - x2 * y3 * z4
        + x2 * y4 * z3
        + x3 * y1 * z2
        - x3 * y1 * z4
        + x3 * y2 * z4
        - x3 * y4 * z2
        - x4 * y1 * z2
        + x4 * y1 * z3
        - x4 * y2 * z3
        + x4 * y3 * z2
    ) / (x2 * y3 - x2 * y4 - x3 * y2 + x3 * y4 + x4 * y2 - x4 * y3)
    return t, s, z0


class Line2D(Curve2D):
    """
    2D line. A line is not displayed or saved in a document.
    2D lines are used as a wrapper to work with raw 2D line information.
    They are created statically using the create method of the Line2D class.
    """

    def __init__(self):
        super().__init__()

    @classmethod
    def cast(cls, arg) -> Line2D:
        return Line2D()

    @classmethod
    def create(cls, start_point: Point2D, end_point: Point2D) -> Line2D:
        """
        Creates a line.
        start_point : The start point of the line
        end_point : The end point of the line
        Returns the new Line2D object or null if the creation failed.
        """
        return Line2D()

    def copy(self) -> Line2D:
        """
        Creates and returns a copy of this line object.
        Returns an independent copy of this line object.
        """
        return Line2D()

    def get_data(self) -> tuple[bool, Point2D, Point2D]:
        """
        Gets all of the data defining the line segment.
        start_point : The output start point of the line.
        end_point : The output end point of the line.
        Returns true if successful.
        """
        return (bool(), Point2D(), Point2D())

    def set(self, start_point: Point2D, end_point: Point2D) -> bool:
        """
        Sets all of the data defining the line segment.
        start_point : The start point of the line
        end_point : The end point of the line
        Returns true if redefining the line is successful
        """
        return bool()

    @property
    def start_point(self) -> Point2D:
        """
        Gets and sets the start point of the line.
        """
        return Point2D()

    @start_point.setter
    def start_point(self, value: Point2D):
        """
        Gets and sets the start point of the line.
        """
        pass

    @property
    def end_point(self) -> Point2D:
        """
        Gets and sets the end point of the line.
        """
        return Point2D()

    @end_point.setter
    def end_point(self, value: Point2D):
        """
        Gets and sets the end point of the line.
        """
        pass

    @property
    def as_nurbs_curve(self) -> NurbsCurve2D:
        """
        Returns a NURBS curve that is geometrically identical to the line.
        """
        return NurbsCurve2D()


INFINITY_LINE_BOUNDS = (-1e9, 1e9)


class BaseLine3D(Curve3D, metaclass=ABCMeta):
    def __init__(self, origin: Point3D):
        super().__init__()

        self._origin = origin

    def is_colinear(self, line: InfiniteLine3D) -> bool:
        """
        Compare this line with another to check for collinearity.
        line : The line to compare with for collinearity.
        Returns true if the two lines are collinear.
        """

        return bool()

    def intersect_with_line(
        self, line: Union[Line3D, InfiniteLine3D], rtol=1e-5, atol=1e-5
    ) -> tuple[bool, float, float]:
        t1, t2, z = line_line_tsz(
            (self.start_point._array, self.end_point._array),
            (line.start_point._array, line.end_point._array),
        )

        if isinstance(line, InfiniteLine3D):
            return (
                all([np.allclose(z, 0, rtol=rtol, atol=atol), (0 <= t1 <= 1)]),
                t1,
                t2,
            )
        else:
            return (
                all(
                    [
                        np.allclose(z, 0, rtol=rtol, atol=atol),
                        (0 <= t1 <= 1),
                        (0 <= t2 <= 1),
                    ]
                ),
                t1,
                t2,
            )

    def intersect_with_curve(self, curve: Curve3D):
        """
        Intersect this line with a curve to get the intersection point(s).
        curve : The intersecting curve.
        The curve can be a Line3D, InfininteLine3D, Circle3D, Arc3D, EllipticalArc3D, Ellipse3D,
        or NurbsCurve3D.
        Returns a collection of the intersection points
        """
        if isinstance(curve, Line3D):
            success, t = self.intersect_with_line(curve)
            if not success:
                return []
            else:
                return [(Point3D(self(tt[0])), tt) for tt in t]
        return super().intersect_with_curve(curve)

    @abstractmethod
    def intersect_with_surface(self, surface):
        """
        Intersect this line with a surface to get the intersection point(s).
        surface : The intersecting surface.
        The surface can be a Plane, Cone, Cylinder, EllipticalCone, EllipticalCylinder, Sphere,
        Torus, or a NurbsSurface.
        Returns a collection of the intersection points.
        """
        pass

    @vectorize(excluded=[0], signature="()->(i)")
    def __call__(self, t):
        return self.start_point._array + self.direction._array * t

    @property
    def origin(self) -> Point3D:
        """
        Gets and sets the origin point of the line.
        """
        return self._origin

    @origin.setter
    def origin(self, value: Point3D):
        """
        Gets and sets the origin point of the line.
        """
        self._origin = value

    @property
    def start_point(self) -> Point3D:
        """ """
        return self.origin

    @property
    def end_point(self) -> Point3D:
        return self.origin + self._direction


class InfiniteLine3D(BaseLine3D):
    """
    3D infinite line. An infinite line is defined by a position and direction in space
    and has no start or end points.
    They are created statically using the create method of the InfiniteLine3D class.
    """

    def interval(self) -> tuple[float, float]:
        return INFINITY_LINE_BOUNDS

    def __init__(
        self,
        origin: Point3D = Point3D((0, 0, 0)),
        direction: Vector3D = Vector3D((1, 0, 0)),
    ):
        super().__init__(origin)
        self._direction = direction

    @classmethod
    def create(cls, origin: Point3D, direction: Vector3D) -> InfiniteLine3D:
        """
        Creates a 3D infinite line.
        origin : The origin point of the line.
        direction : The direction of the line.
        Returns the new InfiniteLine3D object or null if the creation failed.
        """
        return InfiniteLine3D(origin, direction)

    def copy(self) -> InfiniteLine3D:
        """
        Creates and returns a copy of this line object.
        Returns an independent copy of this line object.
        """
        return InfiniteLine3D(self.origin.copy(), self.direction.copy())

    def intersect_with_surface(self, surface: Surface) -> ObjectCollection:
        """
        Intersect this line with a surface to get the intersection point(s).
        surface : The intersecting surface.
        The surface can be a Plane, Cone, Cylinder, EllipticalCone, EllipticalCylinder, Sphere,
        Torus, or a NurbsSurface.
        Returns a collection of the intersection points.
        """
        ...

    def get_data(self) -> tuple[bool, Point3D, Vector3D]:
        """
        Gets all of the data defining the infinite line.
        origin : The output origin point of the line.
        direction : The output direction of the line.
        Returns true if successful.
        """
        return True, self._origin, self._direction

    def set(self, origin: Point3D, direction: Vector3D) -> bool:
        """
        Sets all of the data defining the infinite line.
        origin : The origin point of the line.
        direction : The direction of the line.
        Returns true if successful.
        """
        self._origin = origin
        self._direction = direction
        return True

    @property
    def direction(self) -> Vector3D:
        """
        Gets and sets the direction of the line.
        """
        return self._direction

    @direction.setter
    def direction(self, value: Vector3D):
        """
        Gets and sets the direction of the line.
        """
        self._direction = value

    @property
    def end_point(self) -> Point3D:
        """ """
        return self.origin + self._direction


def line_line_intersection(
    line1: Union[Line2D, InfiniteLine3D, Line3D],
    line2: Union[Line2D, InfiniteLine3D, Line3D],
) -> tuple[bool, float, float]:
    # Extract points and directions from input lines
    start1, end1 = line1.start_point, line1.end_point
    start2, end2 = line2.start_point, line2.end_point
    vec1, vec2 = end1 - start1, end2 - start2
    # Calculate direction vector of the line connecting the two points
    # Calculate direction vectors of the two input lines
    cross_product = cross(vec1._array, vec2._array)
    if np.allclose(cross_product, 0):
        # Lines are parallel, return None
        return False, np.nan, np.nan
    else:
        w = start1 - start2
        # Calculate parameters for point of intersection
        s1 = (
            dot(cross(w._array, vec2._array), cross_product)
            / np.linalg.norm(cross_product) ** 2
        )
        s2 = (
            dot(cross(w._array, vec1._array), cross_product)
            / np.linalg.norm(cross_product) ** 2
        )
        return True, s1, s2


class Line3D(BaseLine3D):
    """
    3D line. A line is not displayed or saved in a document.
    3D lines are used as a wrapper to work with raw 3D line information.
    They are created statically using the create method of the Line3D class.
    """

    def interval(self) -> tuple[float, float]:
        return (0.0, 1.0)

    @classmethod
    def cast(cls, arg) -> Base:
        return Line3D(Point3D.cast(arg[0]), Point3D.cast(arg[1]))

    def __init__(self, start_point: Point3D, end_point: Point3D):
        super().__init__(start_point)
        self._start_point = start_point
        self._end_point = end_point

    @classmethod
    def create(cls, start_point: Point3D, end_point: Point3D) -> Line3D:
        """
        Creates a line.
        start_point : The start point of the line.
        end_point : The end point of the line.
        Returns the new Line3D object or null if the creation failed.
        """
        return Line3D(start_point, end_point)

    @property
    def direction(self):
        return self._end_point - self._start_point

    def copy(self) -> Line3D:
        """
        Creates and returns a copy of this line object.
        Returns an independent copy of this line object.
        """
        return Line3D(self._start_point.copy(), self._end_point.copy())

    def as_infinite_line(self) -> InfiniteLine3D:
        """
        Creates an equivalent InfiniteLine3D.
        Returns an equivalent InfiniteLine3D
        """
        return InfiniteLine3D(self._start_point, self.direction)

    def as_line_2d(self):
        return Line2D(self._start_point, self.direction)

    def is_colinear(self, line: Line3D) -> bool:
        """
        Compare this line with another to check for collinearity
        line : The line to compare with for collinearity
        Returns true if the two lines are collinear
        """
        return self.direction.is_parallel(line.direction)

    def intersect_with_line(
        self, line: Union[Line3D, InfiniteLine3D], rtol=1e-5, atol=1e-5
    ) -> tuple[bool, float, float]:
        t1, t2, z = line_line_tsz(
            (self.start_point._array, self.end_point._array),
            (line.start_point._array, line.end_point._array),
        )

        if isinstance(line, InfiniteLine3D):
            return (
                all([np.allclose(z, 0, rtol=rtol, atol=atol), (0 <= t1 <= 1)]),
                t1,
                t2,
            )
        else:
            return (
                all(
                    [
                        np.allclose(z, 0, rtol=rtol, atol=atol),
                        (0 <= t1 <= 1),
                        (0 <= t2 <= 1),
                    ]
                ),
                t1,
                t2,
            )

    def evaluate_param(self, t: float) -> Point3D:
        return self._start_point + self.direction * t

    evaluate_params = np.vectorize(evaluate_param, excluded=[0], signature="()->()")

    def intersect_with_surface(self, surface: Surface) -> list[Point3D]:
        """
        Intersect this line with a surface to get the intersection point(s).
        surface : The intersecting surface.
        The surface can be a Plane, Cone, Cylinder, EllipticalCone, EllipticalCylinder, Sphere,
        Torus or a NurbsSurface.
        Returns a collection of the intersection points.
        """
        if surface.__class__.__name__ == "Plane":
            return [surface.intersect_with_line(self)]
        else:
            raise NotImplementedError

    def get_data(self) -> tuple[bool, Point3D, Point3D]:
        """
        Gets all of the data defining the line segment.
        start_point : The output start point of the line.
        end_point : The output end point of the line.
        Returns true if successful.
        """
        return (True, self._start_point, self._end_point)

    def set(self, start_point: Point3D, end_point: Point3D) -> bool:
        """
        Sets all of the data defining the line segment.
        start_point : The start point of the line.
        end_point : The end point of the line.
        Returns true if successful.
        """
        self._start_point = start_point
        self._end_point = end_point

    @property
    def start_point(self) -> Point3D:
        """
        Gets and sets the start point of the line.
        """
        return self._start_point

    @start_point.setter
    def start_point(self, value: Point3D):
        """
        Gets and sets the start point of the line.
        """
        self._start_point = value

    @property
    def end_point(self) -> Point3D:
        """
        Gets and sets the end point of the line.
        """
        return self._end_point

    @end_point.setter
    def end_point(self, value: Point3D):
        """
        Gets and sets the end point of the line.
        """
        self._end_point = value

    @property
    def as_nurbs_curve(self) -> NurbsCurve3D:
        """
        Returns a NURBS curve that is geometrically identical to the line.
        """
        ...

    @vectorize(excluded=[0], signature="()->(i)")
    def __call__(self, t):
        return self.start_point._array + self.direction._array * t


ILine = Union[Line2D, Line3D, InfiniteLine3D]
from math import sqrt, fabs, pow


def point_from_local_to_world(self, pt: np.ndarray):
    """

    :param pts: Points in local coordinates
    :type pts: np.ndarray
    :return: Points in world coordinates
    :rtype: np.ndarray
    """
    return self[0] + self[1] * pt[0] + self[2] * pt[1] + self[3] * pt[2]


def vector_from_local_to_world(self, pt: np.ndarray):
    """

    :param pts: Points in local coordinates
    :type pts: np.ndarray
    :return: Points in world coordinates
    :rtype: np.ndarray
    """
    return self[1] * pt[0] + self[2] * pt[1] + self[3] * pt[2]


def circle2d_from_3pt(x1, y1, x2, y2, x3, y3):
    x = (
        0.5
        * (
            pow(x1, 2) * y2
            - pow(x1, 2) * y3
            - pow(x2, 2) * y1
            + pow(x2, 2) * y3
            + pow(x3, 2) * y1
            - pow(x3, 2) * y2
            + pow(y1, 2) * y2
            - pow(y1, 2) * y3
            - y1 * pow(y2, 2)
            + y1 * pow(y3, 2)
            + pow(y2, 2) * y3
            - y2 * pow(y3, 2)
        )
        / (x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2)
    )
    y = (
        -0.5
        * (
            pow(x1, 2) * x2
            - pow(x1, 2) * x3
            - x1 * pow(x2, 2)
            + x1 * pow(x3, 2)
            - x1 * pow(y2, 2)
            + x1 * pow(y3, 2)
            + pow(x2, 2) * x3
            - x2 * pow(x3, 2)
            + x2 * pow(y1, 2)
            - x2 * pow(y3, 2)
            - x3 * pow(y1, 2)
            + x3 * pow(y2, 2)
        )
        / (x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2)
    )
    radius = (
        0.5
        * sqrt(
            (
                pow(x1, 2)
                - 2 * x1 * x2
                + pow(x2, 2)
                + pow(y1, 2)
                - 2 * y1 * y2
                + pow(y2, 2)
            )
            * (
                pow(x1, 2)
                - 2 * x1 * x3
                + pow(x3, 2)
                + pow(y1, 2)
                - 2 * y1 * y3
                + pow(y3, 2)
            )
            * (
                pow(x2, 2)
                - 2 * x2 * x3
                + pow(x3, 2)
                + pow(y2, 2)
                - 2 * y2 * y3
                + pow(y3, 2)
            )
        )
        / (x1 * y2 - x1 * y3 - x2 * y1 + x2 * y3 + x3 * y1 - x3 * y2)
    )

    return x, y, fabs(radius)


from ..numeric.plane import plane_from_3pt


class Circle2D(Curve2D):
    """
    2D circle. A circle is not displayed or saved in a document.
    circles are used as a wrapper to work with raw 2D arc information. They
    are created statically using one of the create methods of the Circle2D class.
    """

    def __init__(self, center=None, radius: float = 1.0):
        super().__init__()
        self._center = center if center is not None else Point2D()
        self._radius = radius

    @classmethod
    def create(cls, center: Point2D, radius: float):
        return cls(center, radius)

    def evaluate(self, t):
        return np.array(
            [
                self.radius * np.cos(t),
                self.radius * np.sin(t),
                0.0,
            ],
            dtype=float,
        )

    @classmethod
    def cast(cls, arg) -> Circle2D:
        return Circle2D(Point2D(np.array(arg[:2])), arg[2])

    @classmethod
    def create_by_center(cls, center: Point2D, radius: float) -> Circle2D:
        """
        Creates a 2D circle object by specifying a center and radius.
        center : A Point2D object that defines the center of the circle.
        radius : The radius of the circle.
        Returns the new Circle2D object or null if the creation failed.
        """
        return Circle2D(center, radius)

    @classmethod
    def create_by_three_points(
        cls, pointOne: Point2D, pointTwo: Point2D, pointThree: Point2D
    ) -> Circle2D:
        """
        Creates a 2D circle through three points.
        pointOne : The first point on the circle.
        pointTwo : The second point on the circle.
        pointThree : The third point on the circle.
        Returns the new Circle2D object or null if the creation failed.
        """
        return Circle2D.cast(
            circle2d_from_3pt(
                pointOne.x,
                pointOne.y,
                pointTwo.x,
                pointTwo.y,
                pointThree.x,
                pointThree.y,
            )
        )

    def copy(self) -> Circle2D:
        """
        Creates and returns an independent copy of this Circle2D object.
        Returns an independent copy of this Circle2D object.
        """
        return Circle2D(self.center.copy(), self.radius)

    def get_data(self) -> tuple[bool, Point2D, float]:
        """
        Gets all of the data defining the circle.
        center : The output point defining the center position of the circle.
        radius : The output radius of the circle.
        Returns true if successful.
        """
        return True, self.center, self.radius

    def set(self, center: Point2D, radius: float) -> bool:
        """
        Sets all of the data defining the circle.
        center : A point that defines the center position of the circle.
        radius : The radius of the circle.
        Returns true if redefining the circle is successful
        """
        self.center = center
        self.radius = radius
        return True

    @property
    def center(self) -> Point2D:
        """
        Gets and sets the center position of the circle.
        """
        return self._center

    @center.setter
    def center(self, value: Point2D):
        """
        Gets and sets the center position of the circle.
        """
        self._center = value

    @property
    def radius(self) -> float:
        """
        Gets and sets the radius of the circle.
        """
        return self._radius

    @radius.setter
    def radius(self, value: float):
        """
        Gets and sets the radius of the circle.
        """
        self._radius = value

    @property
    def as_nurbs_curve(self) -> NurbsCurve2D:
        """
        Returns a NURBS curve that is geometrically identical to the circle.
        """
        raise NotImplementedError

    def on_plane(self, pln: ArrayLike = None) -> "Circle3D":
        c = Circle3D(Point3D.cast(self.center), self.radius)
        if pln is not None:
            c._plane[:] = pln
        return c


class Circle3D(Curve3D):
    """
    3D circle. A circle is not displayed or saved in a document.
    3D circles are used as a wrapper to work with raw 3D circle information.
    They are created statically using one of the create methods of the Circle3D class.
    """

    def __init__(
        self, center: Point3D = None, radius: float = 1.0, plane: np.ndarray = None
    ):
        super().__init__()
        self._center = center if center is not None else Point3D()
        self._radius = radius
        arr = self._center._array
        self._plane = (
            np.array([arr, [1.0, 0, 0], [0.0, 1, 0], [0.0, 0, 1]])
            if plane is None
            else np.array(plane)
        )

        self._center._array = self._plane[0]

    @property
    def plane(self):
        return self._plane

    @classmethod
    def cast(cls, arg) -> Circle3D:
        return Circle3D()

    @classmethod
    def create_by_center(
        cls, center: Point3D, normal: Vector3D, radius: float
    ) -> Circle3D:
        """
        Creates a 3D circle object by specifying a center and radius.
        center : The center point of the circle.
        normal : The normal vector of the circle.
        The plane through the center point and perpendicular to the normal vector defines the plane of the circle.
        radius : The radius of the circle.
        Returns the new Circle3D object or null if the creation failed.
        """

        return Circle3D()

    @classmethod
    def create_by_three_points(
        cls, pointOne: Point3D, pointTwo: Point3D, pointThree: Point3D
    ) -> Circle3D:
        """
        Creates a 3D circle through three points.
        pointOne : The first point on the circle.
        pointTwo : The second point on the circle.
        This point cannot be coincident with pointOne or pointThree.
        This point cannot lie on the line defined by pointOne and pointThree.
        pointThree : The third point on the circle.
        This point cannot be coincident with pointOne or pointThree.
        Returns the new Circle3D object or null if the creation failed.
        """

        pln = plane_from_3pt(*np.array([pointOne, pointTwo, pointThree]))

        return Circle2D.create_by_three_points(
            Point2D.cast(pointOne), Point2D.cast(pointTwo), Point2D.cast(pointThree)
        ).on_plane(pln)

    def copy(self) -> Circle3D:
        """
        Creates and returns an independent copy of this Circle3D object.
        Returns an independent copy of this Circle3D object.
        """
        c = Circle3D(center=self._center.copy(), radius=self._radius)
        c._plane = np.copy(self._plane)
        return c

    def get_data(self) -> tuple[bool, Point3D, Vector3D, float]:
        """
        Gets all of the data defining the circle.
        center : The output center point of the circle.
        normal : The output normal vector.
        radius : The output radius of the circle.
        Returns true if successful
        """
        return True, self.center, self.normal, self.radius

    def set(self, center: Point3D, normal: Vector3D, radius: float) -> bool:
        """
        Sets all of the data defining the circle.
        center : The center point of the circle.
        normal : The normal vector of the circle.
        The plane through the center point and perpendicular to the normal vector defines the plane of the circle.
        radius : The radius of the circle.
        Returns true if successful
        """
        self.center = center
        self.normal = normal
        self.radius = radius
        return bool()

    @property
    def center(self) -> Point3D:
        """
        Gets and sets the center position of the circle.
        """
        return self._center

    @center.setter
    def center(self, value: Point3D):
        """
        Gets and sets the center position of the circle.
        """

        self._plane[0] = value._array
        value._array = self._plane[0]
        self._center = value

    @property
    def normal(self) -> Vector3D:
        """
        Gets and sets the normal of the circle.
        """
        return Vector3D(self._plane[-1])

    @normal.setter
    def normal(self, value: Vector3D):
        """
        Gets and sets the normal of the circle.
        """
        self._plane[-1] = value._array

    @property
    def radius(self) -> float:
        """
        Gets and sets the radius of the circle.
        """
        return self._radius

    @radius.setter
    def radius(self, value: float):
        """
        Gets and sets the radius of the circle.
        """
        self._radius = value

    @property
    def as_nurbs_curve(self) -> NurbsCurve3D:
        """
        Returns a NURBS curve that is geometrically identical to the circle.
        """
        raise NotImplementedError

    def evaluate(self, t: float) -> ArrayLike:
        return point_from_local_to_world(
            self._plane,
            np.array(
                [
                    self.radius * np.cos(t),
                    self.radius * np.sin(t),
                    0.0,
                ],
                dtype=float,
            ),
        )

    def interval(self) -> tuple[float, float]:
        return 0.0, 2 * np.pi


class Arc2D(Circle2D):
    """
    2D arc. A arc is not displayed or saved in a document.
    arcs are used as a wrapper to work with raw 2D arc information. They
    are created statically using one of the create methods supported by the Arc2D class.
    """

    def __init__(self, center: tuple[float, float], radius: float):
        pass

    @classmethod
    def cast(cls, arg) -> Arc2D:
        return Arc2D()

    @classmethod
    def create_by_center(
        cls,
        center: Point2D,
        radius: float,
        start_angle: float,
        end_angle: float,
        is_clockwise: bool,
    ) -> Arc2D:
        """
        Creates a 2D arc object specifying the center, radius and start and end angles.
        A arc is not displayed or saved in a document. arcs arcs are used as
        a wrapper to work with raw 2D arc information.
        center : A Point2D object that defines the center position of the arc in 2D space.
        radius : The radius of the arc.
        start_angle : The start angle in radians, where 0 is along the X-axis.
        end_angle : The end angle in radians, where 0 is along the X-axis.
        is_clockwise : Specifies if the sweep of the arc is clockwise or counterclockwise from the start to end angle.
        Returns the newly created arc or null if the creation failed.
        """
        return Arc2D()

    @classmethod
    def create_by_three_points(
        cls, start_point: Point2D, point: Point2D, end_point: Point2D
    ) -> Arc2D:
        """
        Creates a 2D arc by specifying 3 points.
        A arc is not displayed or saved in a document. arcs arcs are used as
        a wrapper to work with raw 2D arc information.
        start_point : The start point of the arc.
        point : A point along the arc.
        end_point : The end point of the arc.
        Returns the newly created arc or null if the creation failed.
        """
        return Arc2D()

    def copy(self) -> Arc2D:
        """
        Creates and returns an independent copy of this Arc2D object.
        Returns a new Arc2D object that is a copy of this Arc2D object.
        """
        return Arc2D()

    def get_data(self) -> tuple[bool, Point2D, float, float, float, bool]:
        """
        Gets all of the data defining the arc.
        center : The output center point of the arc.
        radius : The output radius of the arc.
        start_angle : The output start angle of the arc in radians, where 0 is along the x axis.
        end_angle : The output end angle of the arc in radians, where 0 is along the x axis.
        is_clockwise : The output value that indicates if the sweep direction is clockwise or counterclockwise.
        Returns true if successful
        """
        return (bool(), Point2D(), float(), float(), float(), bool())

    def set(
        self,
        center: Point2D,
        radius: float,
        start_angle: float,
        end_angle: float,
        is_clockwise: bool,
    ) -> bool:
        """
        Sets all of the data defining the arc.
        center : A Point2D object defining the center position of the arc.
        radius : The radius of the arc.
        start_angle : The start angle of the arc in radians, where 0 is along the x axis.
        end_angle : The end angle of the arc in radians, where 0 is along the x axis.
        is_clockwise : Indicates if the sweep direction is clockwise or counterclockwise.
        Returns true if redefining the arc is successful
        """
        return bool()

    @property
    def center(self) -> Point2D:
        """
        Gets and sets the center position of the arc.
        """
        return Point2D()

    @center.setter
    def center(self, value: Point2D):
        """
        Gets and sets the center position of the arc.
        """
        pass

    @property
    def radius(self) -> float:
        """
        Gets and sets the radius of the arc.
        """
        return float()

    @radius.setter
    def radius(self, value: float):
        """
        Gets and sets the radius of the arc.
        """
        pass

    @property
    def start_angle(self) -> float:
        """
        Gets and sets the start angle of the arc in radians, where 0 is along the x axis.
        """
        return float()

    @start_angle.setter
    def start_angle(self, value: float):
        """
        Gets and sets the start angle of the arc in radians, where 0 is along the x axis.
        """
        pass

    @property
    def end_angle(self) -> float:
        """
        Gets and sets the end angle of the arc in radians, where 0 is along the x axis.
        """
        return float()

    @end_angle.setter
    def end_angle(self, value: float):
        """
        Gets and sets the end angle of the arc in radians, where 0 is along the x axis.
        """
        pass

    @property
    def is_clockwise(self) -> bool:
        """
        Specifies if the sweep direction of the arc is clockwise or counterclockwise.
        """
        return bool()

    @property
    def start_point(self) -> Point2D:
        """
        Gets the position of the start point of the arc.
        """
        return Point2D()

    @property
    def end_point(self) -> Point2D:
        """
        Gets the position of the end point of the arc.
        """
        return Point2D()

    @property
    def as_nurbs_curve(self) -> NurbsCurve2D:
        """
        Returns a NURBS curve that is geometrically identical to the arc.
        """
        return NurbsCurve2D()


class Arc3D(Circle3D):
    """
    3D arc. A arc is not displayed or saved in a document.
    3D arcs are used as a wrapper to work with raw 3D arc information.
    They are created statically using one of the create methods of the Arc3D class.
    """

    def __init__(
        self,
        center: Point3D = None,
        radius: float = 1.0,
        start_angle=0.0,
        end_angle=np.pi / 2,
        plane: np.ndarray = None,
    ) -> None:
        super().__init__(center, radius=radius, plane=plane)
        self._start_angle = start_angle

        self._end_angle = end_angle
        self._reference_vector = Vector3D()
        self._reference_vector._array = self._plane[1]
        self._normal = Vector3D()
        self._normal._array = self._plane[-1]

    @classmethod
    def cast(cls, arg) -> Arc3D:
        return Arc3D()

    @classmethod
    def create_by_center(
        cls,
        center: Point3D,
        normal: Vector3D,
        referenceVector: Vector3D,
        radius: float,
        start_angle: float,
        end_angle: float,
    ) -> Arc3D:
        """
        Creates a 3D arc object by specifying a center point and radius.
        center : The center point of the arc.
        normal : The normal vector of the arc.
        The plane perpendicular to this normal at the center point is the plane of the arc.
        referenceVector : A reference vector from which the start and end angles are measured from.
        This vector must be perpendicular to the normal vector.
        radius : The radius of the arc.
        start_angle : The start angle in radians.
        This angle is measured from the reference vector using the right hand rule around the normal vector.
        end_angle : The end angle in radians.
        This angle is measured from the reference vector using the right hand rule around the normal vector.
        Returns the newly created arc or null if the creation failed.
        """
        yaxis = normal.cross(referenceVector)

        return Arc3D(
            center,
            radius,
            start_angle,
            end_angle,
            plane=np.array(
                [center._array, referenceVector._array, yaxis._array, normal._array]
            ),
        )

    @classmethod
    def create_by_three_points(
        cls, pointOne: Point3D, pointTwo: Point3D, pointThree: Point3D
    ) -> Arc3D:
        """
        Creates a 3D arc by specifying 3 points.
        A arc is not displayed or saved in a document. arcs are used as
        a wrapper to work with raw 3D arc information.
        pointOne : The start point of the arc.
        pointTwo : A point along the arc.
        This point must not be coincident with the first or third points.
        This point must not lie on the line between the first and third points.
        pointThree : The end point of the arc.
        This point must not be coincident with the first or second points.
        Returns the newly created arc or null if the creation failed.
        """

        res = super().create_by_three_points(pointOne, pointTwo, pointThree)

        return Arc3D(
            res.center,
            res.radius,
            *sorted(
                (
                    pointTwo.angle_to(Vector3D(res._plane[1])),
                    pointThree.angle_to(Vector3D(res._plane[1])),
                )
            ),
            plane=res._plane,
        )

    def set_axes(self, normal: Vector3D, referenceVector: Vector3D) -> bool:
        """
        Sets the normal and reference vectors of the arc.
        normal : The new normal vector.
        referenceVector : The new reference vector from which the start and end angles are measured from.
        The reference vector must be perpendicular to the normal vector.
        Returns true if successful
        """
        self._normal._array[:] = normal.as_array()
        self._reference_vector[:] = referenceVector.as_array()
        return True

    def copy(self) -> Arc3D:
        """
        Creates and returns an independent copy of this Arc3D object.
        Returns a new Arc3D object that is a copy of this Arc3D object.
        """
        return Arc3D(
            self.center.copy(),
            self.radius,
            self.start_angle,
            self.end_angle,
            plane=np.copy(self._plane),
        )

    def get_data(self) -> tuple[bool, Point3D, Vector3D, Vector3D, float, float, float]:
        """
        Gets all of the data defining the arc.
        center : The output center point of the arc.
        normal : The output normal vector.
        referenceVector : The output reference vector.
        radius : The output radius of the arc.
        start_angle : The output start angle in radians.
        This angle is measured from the reference vector using the right hand rule around the normal vector.
        end_angle : The output end angle in radians.
        This angle is measured from the reference vector using the right hand rule around the normal vector.
        Returns true if successful
        """
        return (
            True,
            self.center,
            self.referenceVector,
            self.normal,
            self.radius,
            self.start_angle,
            self.end_angle,
        )

    def set(
        self,
        center: Point3D,
        normal: Vector3D,
        referenceVector: Vector3D,
        radius: float,
        start_angle: float,
        end_angle: float,
    ) -> bool:
        """
        Sets all of the data defining the arc.
        center : The center point of the arc.
        normal : The normal vector of the arc.
        The plane perpendicular to this normal at the center point is the plane of the arc.
        referenceVector : A reference vector from which the start and end angles are measured from.
        This vector must be perpendicular to the normal vector.
        radius : The radius of the arc.
        start_angle : The start angle in radians.
        This angle is measured from the reference vector using the right hand rule around the normal vector.
        end_angle : The end angle in radians.
        This angle is measured from the reference vector using the right hand rule around the normal vector.
        Returns true if successful
        """
        self._center._array[:] = center.as_array()
        self._normal._array[:] = normal.as_array()
        self._reference_vector._array[:] = referenceVector.as_array()
        self.radius = radius
        self.start_angle = start_angle
        self.end_angle = end_angle
        return bool()

    @property
    def normal(self) -> Vector3D:
        """
        Gets and sets the normal of the arc.
        """
        return self._normal

    @property
    def referenceVector(self) -> Vector3D:
        """
        Gets and sets the reference vector of the arc.
        """
        return self._reference_vector

    @property
    def start_angle(self) -> float:
        """
        Gets and sets the start angle of the arc in radians.
        This angle is measured from the reference vector using the right hand rule around the normal vector.
        """
        return self._start_angle

    @start_angle.setter
    def start_angle(self, value: float):
        """
        Gets and sets the start angle of the arc in radians.
        This angle is measured from the reference vector using the right hand rule around the normal vector.
        """
        self._start_angle = value

    @property
    def end_angle(self) -> float:
        """
        Gets and sets the end angle of the arc in radians.
        This angle is measured from the reference vector using the right hand rule around the normal vector.
        """
        return self._end_angle

    @end_angle.setter
    def end_angle(self, value: float):
        """
        Gets and sets the end angle of the arc in radians.
        This angle is measured from the reference vector using the right hand rule around the normal vector.
        """
        self._end_angle = value

    @property
    def start_point(self) -> Point3D:
        """
        Gets the start point of the arc.
        """
        return Point3D(self.evaluate(self.start_angle))

    @property
    def end_point(self) -> Point3D:
        """
        Gets the end point of the arc.
        """
        return Point3D(self.evaluate(self.end_angle))

    @property
    def as_nurbs_curve(self) -> NurbsCurve3D:
        """
        Returns a NURBS curve that is geometrically identical to the arc.
        """
        return NurbsCurve3D()

    def interval(self) -> tuple[float, float]:
        return self.start_angle, self.end_angle


class Ellipse2D(Curve2D):
    """
    2D ellipse. A ellipse is not displayed or saved in a document.
    2D ellipses are used as a wrapper to work with raw 2D ellipse information.
    They are created statically using the create method of the Ellipse2D class.
    """

    def __init__(self):
        pass

    @classmethod
    def cast(cls, arg) -> Ellipse2D:
        return Ellipse2D()

    @classmethod
    def create(
        cls,
        center: Point2D,
        major_axis: Vector2D,
        major_radius: float,
        minor_radius: float,
    ) -> Ellipse2D:
        """
        Creates a 2D ellipse by specifying a center position, major and minor axes,
        and major and minor radii.
        center : A Point2D object that defines the center of the ellipse.
        major_axis : The major axis of the ellipse
        major_radius : The major radius of the of the ellipse.
        minor_radius : The minor radius of the of the ellipse.
        Returns the new Ellipse 2D object or null if the creation failed.
        """
        return Ellipse2D()

    def copy(self) -> Ellipse2D:
        """
        Creates and returns a copy of this Ellipse2D object.
        Returns a new Ellipse2D object that is a copy of this Ellipse2D object.
        """
        return Ellipse2D()

    def get_data(self) -> tuple[bool, Point2D, Vector2D, float, float]:
        """
        Gets all of the data defining the ellipse.
        center : The output center point of the ellipse.
        major_axis : The output major axis of the ellipse.
        major_radius : The output major radius of the of the ellipse.
        minor_radius : The output minor radius of the of the ellipse.
        Returns true if successful.
        """
        return (bool(), Point2D(), Vector2D(), float(), float())

    def set(
        self,
        center: Point2D,
        major_axis: Vector2D,
        major_radius: float,
        minor_radius: float,
    ) -> bool:
        """
        Sets all of the data defining the ellipse.
        center : A Point2D object that defines the center of the ellipse.
        major_axis : The major axis of the ellipse.
        major_radius : The major radius of the of the ellipse.
        minor_radius : The minor radius of the of the ellipse.
        Returns true if redefining the ellipse is successful.
        """
        return bool()

    @property
    def center(self) -> Point2D:
        """
        Gets and sets the center position of the ellipse.
        """
        return Point2D()

    @center.setter
    def center(self, value: Point2D):
        """
        Gets and sets the center position of the ellipse.
        """
        pass

    @property
    def major_axis(self) -> Vector2D:
        """
        Gets and sets the major axis of the ellipse.
        """
        return Vector2D()

    @major_axis.setter
    def major_axis(self, value: Vector2D):
        """
        Gets and sets the major axis of the ellipse.
        """
        pass

    @property
    def major_radius(self) -> float:
        """
        Gets and sets the major radius of the ellipse.
        """
        return float()

    @major_radius.setter
    def major_radius(self, value: float):
        """
        Gets and sets the major radius of the ellipse.
        """
        pass

    @property
    def minor_radius(self) -> float:
        """
        Gets and sets the minor radius of the ellipse.
        """
        return float()

    @minor_radius.setter
    def minor_radius(self, value: float):
        """
        Gets and sets the minor radius of the ellipse.
        """
        pass

    @property
    def as_nurbs_curve(self) -> NurbsCurve2D:
        """
        Returns a NURBS curve that is geometrically identical to the ellipse.
        """
        return NurbsCurve2D()


class Ellipse3D(Curve3D):
    """
    3D ellipse. A ellipse is n0t displayed or saved in a document.
    3D ellipses are used as a wrapper to work with raw 3D ellipse information.
    They are created statically using the create method of the Ellipse3D class.
    """

    def __init__(self):
        pass

    @classmethod
    def cast(cls, arg) -> Ellipse3D:
        return Ellipse3D()

    @classmethod
    def create(
        cls,
        center: Point3D,
        normal: Vector3D,
        major_axis: Vector3D,
        major_radius: float,
        minor_radius: float,
    ) -> Ellipse3D:
        """
        Creates a 3D ellipse object.
        center : The center point of the ellipse.
        normal : The normal vector of the ellipse.
        The plane through the center point and perpendicular to the normal vector defines the plane of the ellipse.
        major_axis : The major axis of the ellipse
        major_radius : The major radius of the of the ellipse.
        minor_radius : The minor radius of the of the ellipse.
        Returns the new Ellipse 3D object or null if the creation failed.
        """
        return Ellipse3D()

    def copy(self) -> Ellipse3D:
        """
        Creates a copy of this Ellipse3D object.
        Returns the independent copy of the ellipse.
        """
        return Ellipse3D()

    def get_data(self) -> tuple[bool, Point3D, Vector3D, Vector3D, float, float]:
        """
        Gets all of the data defining the ellipse.
        center : The output center point of the ellipse.
        normal : The output normal vector of the ellipse.
        major_axis : The output major axis of the ellipse
        major_radius : The output major radius of the of the ellipse.
        minor_radius : The output minor radius of the of the ellipse.
        Returns true if successful.
        """
        return (bool(), Point3D(), Vector3D(), Vector3D(), float(), float())

    def set(
        self,
        center: Point3D,
        normal: Vector3D,
        major_axis: Vector3D,
        major_radius: float,
        minor_radius: float,
    ) -> bool:
        """
        Sets all of the data defining the ellipse.
        center : The center point of the ellipse.
        normal : The normal vector of the ellipse.
        The plane through the center point and perpendicular to the normal vector defines the plane of the ellipse.
        major_axis : The major axis of the ellipse.
        major_radius : The major radius of the of the ellipse.
        minor_radius : The minor radius of the of the ellipse.
        Returns true if successful.
        """
        return bool()

    @property
    def center(self) -> Point3D:
        """
        Gets and sets the center position of the ellipse.
        """
        return Point3D()

    @center.setter
    def center(self, value: Point3D):
        """
        Gets and sets the center position of the ellipse.
        """
        pass

    @property
    def normal(self) -> Vector3D:
        """
        Gets and sets the normal of the ellipse.
        """
        return Vector3D()

    @property
    def major_axis(self) -> Vector3D:
        """
        Gets and sets the major axis of the ellipse.
        """
        return Vector3D()

    @major_axis.setter
    def major_axis(self, value: Vector3D):
        """
        Gets and sets the major axis of the ellipse.
        """
        pass

    @property
    def major_radius(self) -> float:
        """
        Gets and sets the major radius of the ellipse.
        """
        return float()

    @major_radius.setter
    def major_radius(self, value: float):
        """
        Gets and sets the major radius of the ellipse.
        """
        pass

    @property
    def minor_radius(self) -> float:
        """
        Gets and sets the minor radius of the ellipse.
        """
        return float()

    @minor_radius.setter
    def minor_radius(self, value: float):
        """
        Gets and sets the minor radius of the ellipse.
        """
        pass

    @property
    def as_nurbs_curve(self) -> NurbsCurve3D:
        """
        Returns a NURBS curve that is geometrically identical to the ellipse.
        """
        return NurbsCurve3D()


class EllipticalArc2D(Curve2D):
    """
    2D elliptical arc. A elliptical arc is not displayed or saved in a document.
    2D elliptical arcs are used as a wrapper to work with raw 2D elliptical arc information.
    They are created statically using the create method of the EllipticalArc2D class.
    """

    def __init__(self):
        pass

    @classmethod
    def cast(cls, arg) -> EllipticalArc2D:
        return EllipticalArc2D()

    @classmethod
    def create(
        cls,
        center: Point2D,
        major_axis: Vector2D,
        major_radius: float,
        minor_radius: float,
        start_angle: float,
        end_angle: float,
    ) -> EllipticalArc2D:
        """
        Creates a 2D elliptical arc
        center : A Point2D object that defines the center of the elliptical arc.
        major_axis : The major axis of the elliptical arc
        major_radius : The major radius of the of the elliptical arc.
        minor_radius : The minor radius of the of the elliptical arc.
        start_angle : The start angle of the elliptical arc in radians, where 0 is along the major axis.
        end_angle : The end angle of the elliptical arc in radians, where 0 is along the major axis.
        Returns the newly created elliptical arc or null if the creation failed.
        """
        return EllipticalArc2D()

    def copy(self) -> EllipticalArc2D:
        """
        Creates and returns a copy of this EllipticalArc2D object.
        Returns a new EllipticalArc2D object that is a copy of this Arc2D object.
        """
        return EllipticalArc2D()

    def get_data(self) -> tuple[bool, Point2D, Vector2D, float, float, float, float]:
        """
        Gets all of the data defining the elliptical arc.
        center : The output center point of the elliptical arc.
        major_axis : The output major axis of the elliptical arc.
        major_radius : The output major radius of the of the elliptical arc.
        minor_radius : The output minor radius of the of the elliptical arc.
        start_angle : The output start angle of the elliptical arc in radians, where 0 is along the major axis.
        end_angle : The output end angle of the elliptical arc in radians, where 0 is along the major axis.
        Returns true if successful
        """
        return (bool(), Point2D(), Vector2D(), float(), float(), float(), float())

    def set(
        self,
        center: Point2D,
        major_axis: Vector2D,
        major_radius: float,
        minor_radius: float,
        start_angle: float,
        end_angle: float,
    ) -> bool:
        """
        center : A Point2D object that defines the center of the elliptical arc.
        major_axis : The major axis of the elliptical arc.
        major_radius : The major radius of the of the elliptical arc.
        minor_radius : The minor radius of the of the elliptical arc.
        start_angle : The start angle of the elliptical arc in radians, where 0 is along the major axis.
        end_angle : The end angle of the elliptical arc in radians, where 0 is along the major axis.
        Returns true if redefining the elliptical arc is successful
        """
        return bool()

    @property
    def center(self) -> Point2D:
        """
        Gets and sets the center position of the elliptical arc.
        """
        return Point2D()

    @center.setter
    def center(self, value: Point2D):
        """
        Gets and sets the center position of the elliptical arc.
        """
        pass

    @property
    def major_axis(self) -> Vector2D:
        """
        Gets and sets the major axis of the elliptical arc.
        """
        return Vector2D()

    @major_axis.setter
    def major_axis(self, value: Vector2D):
        """
        Gets and sets the major axis of the elliptical arc.
        """
        pass

    @property
    def major_radius(self) -> float:
        """
        Gets and sets the major radius of the elliptical arc.
        """
        return float()

    @major_radius.setter
    def major_radius(self, value: float):
        """
        Gets and sets the major radius of the elliptical arc.
        """
        pass

    @property
    def minor_radius(self) -> float:
        """
        Gets and sets the minor radius of the elliptical arc.
        """
        return float()

    @minor_radius.setter
    def minor_radius(self, value: float):
        """
        Gets and sets the minor radius of the elliptical arc.
        """
        pass

    @property
    def start_angle(self) -> float:
        """
        Gets and sets the start angle of the elliptical arc in radians, where 0 is along the major axis.
        """
        return float()

    @start_angle.setter
    def start_angle(self, value: float):
        """
        Gets and sets the start angle of the elliptical arc in radians, where 0 is along the major axis.
        """
        pass

    @property
    def end_angle(self) -> float:
        """
        Gets and sets the end angle of the elliptical arc in radians, where 0 is along the major axis.
        """
        return float()

    @end_angle.setter
    def end_angle(self, value: float):
        """
        Gets and sets the end angle of the elliptical arc in radians, where 0 is along the major axis.
        """
        pass

    @property
    def is_clockwise(self) -> bool:
        """
        Indicates if the sweep direction of the elliptical arc is clockwise or counterclockwise.
        """
        return bool()

    @property
    def isCircular(self) -> bool:
        """
        Indicates if the elliptical arc is the geometric equivalent of a circular arc
        """
        return bool()

    @property
    def start_point(self) -> Point2D:
        """
        Gets the position of the start point of the elliptical arc.
        """
        return Point2D()

    @property
    def end_point(self) -> Point2D:
        """
        Gets the position of the end point of the elliptical arc.
        """
        return Point2D()

    @property
    def as_nurbs_curve(self) -> NurbsCurve2D:
        """
        Returns a NURBS curve that is geometrically identical to the elliptical arc.
        """
        return NurbsCurve2D()


class EllipticalArc3D(Curve3D):
    """
    3D elliptical arc. A elliptical arc is not displayed or saved in a document.
    3D elliptical arcs are used as a wrapper to work with raw 3D elliptical arc information.
    They are created statically using the create method of the EllipticalArc3D class.
    """

    def __init__(self):
        pass

    @classmethod
    def cast(cls, arg) -> EllipticalArc3D:
        return EllipticalArc3D()

    @classmethod
    def create(
        cls,
        center: Point3D,
        normal: Vector3D,
        major_axis: Vector3D,
        major_radius: float,
        minor_radius: float,
        start_angle: float,
        end_angle: float,
    ) -> EllipticalArc3D:
        """
        Creates a 3D elliptical arc.
        center : The center point of the elliptical arc.
        normal : The normal vector of the elliptical arc.
        major_axis : The major axis of the elliptical arc.
        major_radius : The major radius of the of the elliptical arc.
        minor_radius : The minor radius of the of the elliptical arc.
        start_angle : The start angle of the elliptical arc in radians, where 0 is along the major axis.
        end_angle : The end angle of the elliptical arc in radians, where 0 is along the major axis.
        Returns the newly created elliptical arc or null if the creation failed.
        """
        return EllipticalArc3D()

    def copy(self) -> EllipticalArc3D:
        """
        Creates and returns a copy of this EllipticalArc3D object.
        Returns a new EllipticalArc3D object that is a copy of this Arc3D object.
        """
        return EllipticalArc3D()

    def get_data(
        self,
    ) -> tuple[bool, Point3D, Vector3D, Vector3D, float, float, float, float]:
        """
        Gets all of the data defining the elliptical arc.
        center : The output center point of the elliptical arc.
        normal : The output normal vector of the elliptical arc.
        major_axis : The output major axis of the elliptical arc.
        major_radius : The output major radius of the of the elliptical arc.
        minor_radius : The output minor radius of the of the elliptical arc.
        start_angle : The output start angle of the elliptical arc in radians, where 0 is along the major axis.
        end_angle : The output end angle of the elliptical arc in radians, where 0 is along the major axis.
        Returns true if successful.
        """
        return (
            bool(),
            Point3D(),
            Vector3D(),
            Vector3D(),
            float(),
            float(),
            float(),
            float(),
        )

    def set(
        self,
        center: Point3D,
        normal: Vector3D,
        major_axis: Vector3D,
        major_radius: float,
        minor_radius: float,
        start_angle: float,
        end_angle: float,
    ) -> bool:
        """
        Sets all of the data defining the elliptical arc.
        center : The center point of the elliptical arc.
        normal : The normal vector of the elliptical arc.
        major_axis : The major axis of the elliptical arc.
        major_radius : The major radius of the of the elliptical arc.
        minor_radius : The minor radius of the of the elliptical arc.
        start_angle : The start angle of the elliptical arc in radians, where 0 is along the major axis.
        end_angle : The end angle of the elliptical arc in radians, where 0 is along the major axis.
        Returns true if successful.
        """
        return bool()

    @property
    def center(self) -> Point3D:
        """
        Gets and sets the center point of the elliptical arc.
        """
        return Point3D()

    @center.setter
    def center(self, value: Point3D):
        """
        Gets and sets the center point of the elliptical arc.
        """
        pass

    @property
    def normal(self) -> Vector3D:
        """
        Gets and sets the normal of the elliptical arc.
        """
        return Vector3D()

    @property
    def major_axis(self) -> Vector3D:
        """
        Gets and sets the major axis of the elliptical arc.
        """
        return Vector3D()

    @major_axis.setter
    def major_axis(self, value: Vector3D):
        """
        Gets and sets the major axis of the elliptical arc.
        """
        pass

    @property
    def major_radius(self) -> float:
        """
        Gets and sets the major radius of the elliptical arc.
        """
        return float()

    @major_radius.setter
    def major_radius(self, value: float):
        """
        Gets and sets the major radius of the elliptical arc.
        """
        pass

    @property
    def minor_radius(self) -> float:
        """
        Gets and sets the minor radius of the elliptical arc.
        """
        return float()

    @minor_radius.setter
    def minor_radius(self, value: float):
        """
        Gets and sets the minor radius of the elliptical arc.
        """
        pass

    @property
    def start_angle(self) -> float:
        """
        Gets and sets the start angle of the elliptical arc.
        """
        return float()

    @start_angle.setter
    def start_angle(self, value: float):
        """
        Gets and sets the start angle of the elliptical arc.
        """
        pass

    @property
    def end_angle(self) -> float:
        """
        Gets and sets the end angle of the elliptical arc.
        """
        return float()

    @end_angle.setter
    def end_angle(self, value: float):
        """
        Gets and sets the end angle of the elliptical arc.
        """
        pass

    @property
    def as_nurbs_curve(self) -> NurbsCurve3D:
        """
        Returns a NURBS curve that is geometrically identical to the elliptical arc.
        """
        return NurbsCurve3D()


from mmcore.geom.curves import NURBSpline


class NurbsCurveApiProxy3D(NURBSpline):
    def evaluate(self, t: float):
        arr = np.zeros((3,), dtype=float)
        sum_of_weights = 0  # sum of weight * basis function
        for i in range(self._control_points_count):
            b = self.basis_function(t, i, self.degree)

            if b > 0:
                arr[0] += b * self.weights[i] * self.control_points[i].x
                arr[1] += b * self.weights[i] * self.control_points[i].y
                arr[2] += b * self.weights[i] * self.control_points[i].z
                sum_of_weights += b * self.weights[i]
        # normalizing with the sum of weights to get rational B-spline
        if sum_of_weights > 0:
            arr[0] /= sum_of_weights
            arr[1] /= sum_of_weights
            arr[2] /= sum_of_weights
        return arr


class NurbsCurveApiProxy2D(NURBSpline):
    def evaluate(self, t: float):
        arr = np.zeros((2,), dtype=float)
        sum_of_weights = 0  # sum of weight * basis function
        for i in range(self._control_points_count):
            b = self.basis_function(t, i, self.degree)

            if b > 0:
                arr[0] += b * self.weights[i] * self.control_points[i].x
                arr[1] += b * self.weights[i] * self.control_points[i].y

                sum_of_weights += b * self.weights[i]
        # normalizing with the sum of weights to get rational B-spline

        arr[0] /= sum_of_weights + 1e-12
        arr[1] /= sum_of_weights + 1e-12
        return arr


class NurbsCurve2D(Curve2D):
    """
    2D NURBS curve. A NURBS curve is not displayed or saved in a document.
    2D NURBS curves are used as a wrapper to work with raw 2D NURBS curve information.
    They are created statically using one of the create methods of the NurbsCurve2D class.
    """

    def __init__(self, control_points, weights=None, degree=3, knots=None):
        super().__init__()
        cpts = []
        for i in range(len(control_points)):
            item = control_points[i]
            if isinstance(item, Point3D):
                pass
            else:
                p = Point3D()
                p._array = item
                cpts.append(p)

        self._proxy = NurbsCurveApiProxy2D(cpts, weights, degree, knots)

    def evaluate(self, t: float) -> ArrayLike:
        return self._proxy.evaluate(t)

    def interval(self):
        return self._proxy.interval()

    @classmethod
    def cast(cls, arg) -> NurbsCurve2D:
        return NurbsCurve2D()

    @classmethod
    def create_non_rational(
        cls,
        control_points: list[Point2D],
        degree: int,
        knots: list[float],
        is_periodic: bool,
    ) -> NurbsCurve2D:
        """
        Creates a 2D NURBS non-rational b-spline object.
        control_points : An array of control point that define the path of the spline
        degree : The degree of curvature of the spline
        knots : An array of numbers that define the knot vector of the spline. The knots is an array of (>=degree + N + 1) numbers, where N is the number of control points.
        is_periodic : A bool specifying if the spline is to be Periodic. A periodic spline has a start point and
        end point that meet forming a closed loop.
        Returns the new NurbsCurve2D object or null if the creation failed.
        """
        return NurbsCurve2D()

    @classmethod
    def create_rational(
        cls,
        control_points: list[Point2D],
        degree: int,
        knots: list[float],
        weights: list[float],
        is_periodic: bool,
    ) -> NurbsCurve2D:
        """
        Creates a 2D NURBS rational b-spline object.
        control_points : An array of control point that define the path of the spline
        degree : The degree of curvature of the spline
        knots : An array of numbers that define the knot vector of the spline. The knots is an array of (>=degree + N + 1) numbers, where N is the number of control points.
        weights : An array of numbers that define the weights for the spline.
        is_periodic : A bool specifying if the spline is to be Periodic. A periodic curve has a start point and
        end point that meet (with curvature continuity) forming a closed loop.
        Returns the new NurbsCurve2D object or null if the creation failed.
        """
        return NurbsCurve2D(control_points, weights, degree, knots)

    def copy(self) -> NurbsCurve2D:
        """
        Creates and returns an independent copy of this NurbsCurve2D object.
        Returns an independent copy of this NurbsCurve2D.
        """
        return NurbsCurve2D()

    def get_data(
        self,
    ) -> tuple[bool, list[Point2D], int, list[float], bool, ndarray | ndarray, bool]:
        """
        Gets the data that defines a 2D NURBS rational b-spline object.
        control_points : The output array of control point that define the path of the spline.
        degree : The output degree of curvature of the spline.
        knots : The output array of numbers that define the knots of the spline.
        is_rational : The output value indicating if the spline is rational. A rational spline will have a weight value
        for each control point.
        weights : The output array of numbers that define the weights for the spline.
        is_periodic : The output value indicating if the spline is Periodic. A periodic curve has a start point and
        end point that meet (with curvature continuity) forming a closed loop.
        Returns true if successful.
        """
        return (
            True,
            self.control_points,
            self.degree,
            self.knots,
            self.is_rational,
            self.weights,
            self.is_periodic,
        )

    @property
    def weights(self):
        return self._proxy.weights

    @weights.setter
    def weights(self, v):
        self._proxy.weights[:] = v

    def set(
        self,
        control_points: list[Point3D],
        degree: int,
        knots: list[float],
        is_rational: bool,
        weights: list[float],
        is_periodic: bool,
    ) -> bool:
        """
        Sets the data that defines a 2D NURBS rational b-spline object.
        control_points : The array of control point that define the path of the spline
        degree : The degree of curvature of the spline
        knots : An array of numbers that define the knots of the spline.
        is_rational : A bool indicating if the spline is rational. A rational spline must have a weight value
        for each control point.
        weights : An array of numbers that define the weights for the spline.
        is_periodic : A bool specifying if the spline is to be Periodic. A periodic curve has a start point and
        end point that meet (with curvature continuity) forming a closed loop.
        Returns true if successful
        """

        self._proxy.set(
            control_points=control_points, degree=degree, knots=knots, weights=weights
        )

        return True

    def extract(self, startParam: float, endParam: float) -> NurbsCurve3D:
        """
        Defines a new NURBS curve that is the subset of this NURBS curve in the parameter
        range of [startParam, endParam]
        startParam : The parameter position of the start of the subset.
        endParam : The parameter position of the end of the subset.
        Returns a new NurbsCurve2D object.
        """
        raise NotImplementedError

    def merge(self, nurbsCurve: NurbsCurve2D) -> NurbsCurve2D:
        """
        Define a new NURBS curve that is the result of combining this NURBS curve with
        another NURBS curve. The curves are merged with the end point of the current
        curve merging with the start point of the other curve. The curves are forced
        to join even if they are not physically touching so you will typically want
        to make sure the end and start points of the curves are where you expect them to be.
        nurbsCurve : The NURBS curve to combine with
        Returns a new NurbsCurve2D object.
        """
        raise NotImplementedError

    @property
    def controlPointCount(self) -> int:
        """
        Gets the number of control points that define the curve
        """
        return len(self._proxy.control_points)

    @property
    def degree(self) -> int:
        """
        Returns the degree of the curve
        """
        return self._proxy.degree

    @property
    def knot_count(self) -> int:
        """
        Returns the knot count of the curve
        """
        return len(self._proxy.knots)

    @property
    def is_rational(self) -> bool:
        """
        Indicates if the curve is rational or non-rational type
        """
        return not np.allclose(self.weights, 1)

    @property
    def is_closed(self) -> bool:
        """
        Indicates if the curve is closed
        """
        return False

    @property
    def is_periodic(self) -> bool:
        """
        Indicates if the curve is periodic.
        """
        raise NotImplementedError

    @property
    def control_points(self) -> list[Point2D]:
        """
        Returns an array of Point2D objects that define the control points of the curve.
        """
        return self._proxy.control_points

    @property
    def knots(self) -> list[float]:
        """
        Returns an array of numbers that define the Knots of the curve.
        """
        return self._proxy.knots


class NurbsCurve3D(Curve3D):
    """
    3D NURBS curve. A NURBS curve is not displayed or saved in a document.
    3D NURBS curves are used as a wrapper to work with raw 3D NURBS curve information.
    They are created statically using one of the create methods of the NurbsCurve3D class.
    """

    def __init__(self, control_points, weights=None, degree=3, knots=None):
        super().__init__()

        self._proxy = NURBSpline(control_points, weights, degree, knots)
        self._control_points = []
        for i in range(len(self._proxy.control_points)):
            pt = Point3D()
            pt._array = control_points[i]
            self._control_points.append(pt)

    @property
    def derivative(self):
        return self._proxy.derivative

    @property
    def second_derivative(self):
        return self._proxy.second_derivative

    def evaluate(self, t: float) -> ArrayLike:
        return self._proxy.evaluate(t)

    def normal(self, t):
        return self._proxy.normal(t)

    def tangent(self, t):
        return self._proxy.tangent(t)

    def curvature(self, t):
        return self._proxy.curvature(t)

    def interval(self):
        return self._proxy.interval()

    @classmethod
    def cast(cls, arg) -> NurbsCurve3D:
        raise NotImplementedError

    @classmethod
    def create_non_rational(
        cls,
        control_points: list[Point3D],
        degree: int,
        knots: list[float],
        is_periodic: bool,
    ) -> NurbsCurve3D:
        """
        Creates a 3D NURBS non-rational b-spline object.
        control_points : An array of control point that define the path of the spline.
        degree : The degree of curvature of the spline.
        knots : An array of numbers that define the knot vector of the spline. The knots is an array of (>=degree + N + 1) numbers, where N is the number of control points.
        is_periodic : A bool specifying if the spline is to be Periodic. A periodic spline has a start point and
        end point that meet forming a closed loop.
        Returns the new NurbsCurve3D object or null if the creation failed.
        """
        return NurbsCurve3D(control_points, None, degree, knots)

    @classmethod
    def create_rational(
        cls,
        control_points: list[Point3D],
        degree: int,
        knots: list[float],
        weights: list[float],
        is_periodic: bool,
    ) -> NurbsCurve3D:
        """
        Creates a 3D NURBS rational b-spline object.
        control_points : An array of control point that define the path of the spline.
        degree : The degree of curvature of the spline.
        knots : An array of numbers that define the knot vector of the spline. The knots is an array of (>=degree + N + 1) numbers, where N is the number of control points.
        weights : An array of numbers that define the weight at each control point.
        is_periodic : A bool specifying if the spline is to be Periodic. A periodic curve has a start point and
        end point that meet (with curvature continuity) forming a closed loop.
        Returns the new NurbsCurve3D object or null if the creation failed.
        """
        return NurbsCurve3D(control_points, weights, degree, knots)

    def get_data(
        self,
    ) -> tuple[bool, list[Point3D], int, list[float], bool, ndarray | ndarray, bool]:
        """
        Gets the data that defines a 3D NURBS rational b-spline object.
        control_points : The output array of control point that define the path of the spline.
        degree : The output degree of curvature of the spline.
        knots : The output array of numbers that define the knot vector of the spline.
        is_rational : The output value indicating if the spline is rational. A rational spline will have a weight value
        for each control point.
        weights : The output array of numbers that define the weights for the spline.
        is_periodic : The output value indicating if the spline is Periodic. A periodic curve has a start point and
        end point that meet (with curvature continuity) forming a closed loop.
        Returns true if successful.
        """
        return (
            True,
            self.control_points,
            self.degree,
            self.knots,
            self.is_rational,
            self.weights,
            self.is_periodic,
        )

    @property
    def weights(self):
        return self._proxy.weights

    @weights.setter
    def weights(self, v):
        self._proxy.weights[:] = v

    def set(
        self,
        control_points: list[Point3D],
        degree: int,
        knots: list[float],
        is_rational: bool,
        weights: list[float],
        is_periodic: bool,
    ) -> bool:
        """
        Sets the data that defines a 3D NURBS rational b-spline object.
        control_points : The array of control point that define the path of the spline.
        degree : The degree of curvature of the spline.
        knots : An array of numbers that define the knot vector of the spline.
        is_rational : A bool value indicating if the spline is rational. A rational spline must have a weight value
        for each control point.
        weights : An array of numbers that define the weights for the spline.
        is_periodic : A bool indicating if the spline is Periodic. A periodic curve has a start point and
        end point that meet (with curvature continuity) forming a closed loop.
        Returns true if successful.
        """

        self._proxy.set(
            control_points=control_points, degree=degree, knots=knots, weights=weights
        )

        return True

    def extract(self, startParam: float, endParam: float) -> NurbsCurve3D:
        """
        Defines a new NURBS curve that is the subset of this NURBS curve in the parameter
        range of [startParam, endParam]
        startParam : The parameter position that defines the start of the subset.
        endParam : The parameter position that defines the end of the subset.
        Returns a new NurbsCurve3D object.
        """

        raise NotImplementedError

    def merge(self, nurbsCurve: NurbsCurve3D) -> NurbsCurve3D:
        """
        Define a new NURBS curve that is the result of combining this NURBS curve with
        another NURBS curve.
        nurbsCurve : The NURBS curve to combine with.
        Returns a new NurbsCurve3D object.
        """
        raise NotImplementedError

    def copy(self) -> NurbsCurve3D:
        """
        Creates and returns an independent copy of this NurbsCurve3D object.
        Returns an independent copy of this NurbsCurve3D.
        """
        raise NotImplementedError

    @property
    def controlPointCount(self) -> int:
        """
        Gets the number of control points that define the curve.
        """
        return len(self._proxy.control_points)

    @property
    def degree(self) -> int:
        """
        Returns the degree of the curve.
        """
        return self._proxy.degree

    @property
    def knot_count(self) -> int:
        """
        Returns the knot count of the curve.
        """
        return len(self._proxy.knots)

    @property
    def is_rational(self) -> bool:
        """
        Indicates if the curve is rational or non-rational type.
        """
        return not np.allclose(self.weights, 1)

    @property
    def is_closed(self) -> bool:
        """
        Indicates if the curve is closed.
        """
        return False

    @property
    def is_periodic(self) -> bool:
        """
        Indicates if the curve is periodic.
        """
        raise NotImplementedError

    @property
    def control_points(self) -> list[Point3D]:
        """
        Returns an array of Point3D objects that define the control points of the curve.
        """
        return self._control_points

    @property
    def knots(self) -> list[float]:
        """
        Returns an array of numbers that define the knot vector of the curve.
        """
        return self._proxy.knots


CT = TypeVar("CT", bound=BaseCurve)


class OffsetCurve(BaseCurve):
    parent: CT
    distance: float

    def __init__(self, curve: CT, distance: float = 0.0):
        super().__init__()
        if isinstance(curve, OffsetCurve):
            distance = curve.distance + distance
            curve = curve.parent
        self.distance = distance
        self.parent = curve

    def evaluate(self, t: float) -> ArrayLike:
        return self.parent.evaluate(t) + self.parent.normal(t) * self.distance

    def interval(self) -> tuple[float, float]:
        return self.parent.interval()

    def transform(self, matrix):
        return False


class SubCurve(BaseCurve):
    parent: CT
    start: float
    end = float

    def __init__(self, curve: CT, start: float = None, end: float = None):
        super().__init__()
        self.parent = curve
        self.start = start if start is not None else self.parent.interval()[0]
        self.end = end if end is not None else self.parent.interval()[1]

    def evaluate(self, t: float) -> ArrayLike:
        return self.parent.evaluate(t)

    def interval(self) -> tuple[float, float]:
        return self.start, self.end

    def transform(self, matrix):
        return False

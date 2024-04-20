from __future__ import annotations

from typing import Union, Optional

import numpy as np

from mmcore.api import Surface, Point3D, Vector3D, Line3D, InfiniteLine3D, Curve3D, ObjectCollection
from mmcore.geom.vec import unit, cross, orthonormalize
from mmcore.numeric.plane import evaluate_plane, inverse_evaluate_plane, plane_from_3pt, plane_from_normal2
from mmcore.numeric.curve_intersection import curve_x_plane


class Plane(Surface):
    """
    plane.
    planes are used as a wrapper to work with raw plane information.
    A plane has no boundaries or size, but is infinite and is represented
    by a position, a normal, and an orientation in space.
    They are created statically using the create method of the Plane class.
    """

    def __init__(self, arr=None):
        super().__init__()
        self._array = (
            np.r_[np.zeros((1, 3), dtype=float), np.eye(3, dtype=float)]
            if arr is None
            else np.array(arr)
            if not isinstance(arr, np.ndarray)
            else arr
        )
        self._origin = Point3D()
        self._xaxis = Vector3D()
        self._yaxis = Vector3D()
        self._normal = Vector3D()

        self._set_by_array()

    def _set_by_array(self):
        self._origin._array = self._array[0]
        self._xaxis._array = self._array[1]
        self._yaxis._array = self._array[2]
        self._normal._array = self._array[3]

    @classmethod
    def cast(cls, arg: tuple) -> Plane:

        if isinstance(arg, np.ndarray) and arg.shape[0] == 4:
            return cls.__dispatch_table__[np.ndarray](arg)
        return cls.__dispatch_table__[tuple(type(a) for a in arg)].__func__(cls, *arg)

    @classmethod
    def create_from_array(cls, arr):
        return Plane(arr=arr)

    @classmethod
    def create_from_normal(cls, origin: Point3D, normal: Vector3D):
        return Plane(plane_from_normal2(origin._array, normal._array))

    @classmethod
    def create_from_3pt(cls, origin: Point3D, a: Point3D, b: Point3D):
        pln, _equation = plane_from_3pt(origin._array, a._array, b._array)

        return Plane(pln)

    @classmethod
    def create(cls, origin: Point3D, xaxis: Vector3D, yaxis: Vector3D) -> Plane:
        """
        Creates a plane object by specifying an origin and a normal direction.
        origin : The origin point of the plane.
        normal : The normal direction of the plane.
        Returns the new plane object or null if the creation failed.
        """

        xaxis, yaxis = unit(xaxis._array), unit(yaxis._array)
        normal = cross(xaxis, yaxis)

        return Plane(np.array([origin._array, xaxis, yaxis, normal]))

    @classmethod
    def create_from_3pt_numeric(cls, pt1: np.ndarray, pt2: np.ndarray, pt3: np.ndarray):
        p, _ = plane_from_3pt(pt1, pt2, pt3)
        return Plane(p)

    @classmethod
    def create_using_directions(
            cls, origin: Point3D, xaxis: Vector3D, yaxis: Vector3D
    ) -> Plane:
        """
        Creates a plane object by specifying an origin along with U and V directions.
        origin : The origin point of the plane.
        xaxis : The U direction for the plane.
        yaxis : The V direction for the plane.
        Returns the new plane object or null if the creation failed.
        """
        xaxis, yaxis = unit((xaxis._array, yaxis._array))
        return Plane(np.array([origin.as_array(), xaxis, yaxis, cross(xaxis, yaxis)]))

    __dispatch_table__ = {
        (Point3D, Point3D, Point3D): create_from_3pt,
        (Point3D, Vector3D, Vector3D): create_using_directions,
        (Point3D, Vector3D): create_from_normal,
        np.ndarray: create_from_array,
        (np.ndarray, np.ndarray, np.ndarray, np.ndarray): create_from_array,
        (np.ndarray, np.ndarray, np.ndarray): create_from_3pt_numeric,
        (list, list, list, list): create_from_array,
        (tuple, tuple, tuple, tuple): create_from_array
    }

    def set_xy_directions(self, xaxis: Vector3D, yaxis: Vector3D) -> bool:
        """
        Sets the U and V directions of the plane.
        xaxis : The U direction for the plane.
        yaxis : The V direction for the plane.
        Returns true if successful.
        """

        self.xaxis._array[:] = xaxis._array
        self._yaxis._array[:] = yaxis._array
        self._normal._array[:] = xaxis.cross(yaxis)._array
        return True

    def is_parallel_plane(self, plane: Plane) -> bool:
        # planes are parallel if their normals are parallel
        return self._normal.is_parallel(plane._normal)

    def is_parallel_line(self, line: Line3D) -> bool:
        # a plane is parallel to a line if the plane's normal is perpendicular to the line's direction
        return self._normal.is_perpendicular(line.direction)

    def is_perpendicular_plane(self, plane: Plane) -> bool:
        # planes are perpendicular if their normals are perpendicular
        return self._normal.is_perpendicular(plane._normal)

    def is_perpendicular_line(self, line: Line3D) -> bool:
        # a plane is perpendicular to a line if the plane's normal is parallel to the line's direction
        return self._normal.is_parallel(line.direction)

    def is_coplanar(self, plane: Plane) -> bool:
        # planes are co-planar if the distance between them is 0
        # here we calculate the distance from the plane to the origin point of the other plane
        # this can be done by taking the dot product of the other plane's origin
        # with this plane's normal and subtracting this plane's distance to the origin
        return (
                abs(plane._origin.dot(self._normal) - self._origin.dot(self._normal))
                < 1e-10
        )

    def intersect_with_plane(self, plane: Plane) -> Optional[InfiniteLine3D]:
        # two planes intersect in a line if they are not parallel
        if not self.is_parallel_plane(plane):
            # to find the intersection line, we take the cross product of the planes' normals
            direction = self._normal.cross(plane._normal)
            # we also need a point on the line, which can be found by solving a system of linear equations
            # for simplicity, we will just use (0, 0, 0) as the point for this demo code
            origin = Point3D((0, 0, 0))

            return InfiniteLine3D(origin, direction)
        else:
            return None  # planes are parallel and do not intersect

    def evaluate(self, uvh):
        return evaluate_plane(self._array, uvh)

    def evaluate_inverse(self, pt):
        return inverse_evaluate_plane(self._array, pt)

    def intersect_with_line(self, line: Union[InfiniteLine3D, Line3D]) -> Point3D:
        # a plane intersects a line at a point if the line is not parallel to the plane

        if not self.is_parallel_line(line):
            # to find the intersection point, we project the line's origin onto the plane
            v = line.origin - self._origin
            t = -self._normal.dot(v) / self._normal.dot(line.direction)

            if isinstance(line, InfiniteLine3D) or (0 <= t <= 1):
                return Point3D(line.origin._array + t * line.direction._array)

    # The following methods intersect_with_curve and intersect_with_surface need complex geometrical calculations,
    # which may be out of this scope as it requires the implementation of visualization and intersection
    # functionality for each curve and surface.

    def intersect_with_curve(self, curve: Curve3D) -> ObjectCollection:
        """
        To implement this method, you need to define how a plane can intersect
        with different types of curves (Line3D, InfiniteLine3D, Circle3D, Arc3D,
        EllipticalArc3D, Ellipse3D, NurbsCurve3D). Each of those types would
        require a specific method to calculate intersections.
        """

        raise curve_x_plane(curve, self._array)

    def intersect_with_surface(self, surface: Surface) -> ObjectCollection:
        """
        Similar to intersect_with_curve, this method requires implementing intersection
        calculations for a Plane with different types of surfaces (Plane, Cone, Cylinder,
        EllipticalCone, EllipticalCylinder, Sphere, Torus, NurbsSurface).
        """
        raise NotImplementedError()

    def copy(self) -> Plane:
        """
        Creates and returns an independent copy of this Plane object.
        Returns a new Plane object that is a copy of this Plane object.
        """
        return Plane(np.copy(self._array)

                     )

    @property
    def origin(self) -> Point3D:
        """
        Gets and sets the origin point of the plane.
        """
        return self._origin

    @origin.setter
    def origin(self, value: Point3D):
        self._origin = value

    @property
    def normal(self) -> Vector3D:
        """
        Gets and sets the normal of the plane.
        """

        return self._normal

    @normal.setter
    def normal(self, value: Vector3D):
        self._normal[:] = value._array
        (
            self._normal._array,
            self._xaxis._array,
            self._yaxis._array,
        ) = orthonormalize(
            self._normal._array, self._xaxis._array, self._yaxis._array
        )

    @property
    def xaxis(self) -> Vector3D:
        """
        Gets the U Direction of the plane.
        """
        return self._xaxis

    @property
    def yaxis(self) -> Vector3D:
        """
        Gets the V Direction of the plane.
        """
        return self._yaxis

    def to_data(self):
        self.origin.to_data()

    def as_array(self, dtype=float, *args, **kwargs) -> np.ndarray[(4, 3), np.dtype[float]]:
        return np.asarray(self._array, dtype=dtype, *args, **kwargs)

    def __array__(self, dtype=float):
        return np.array(self._array, dtype=dtype)

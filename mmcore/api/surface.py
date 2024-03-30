from __future__ import annotations

from ._typing import Union

from mmcore.api.base import Surface, Curve3D
from mmcore.api._base import ObjectCollection
from mmcore.api.curves import Line3D, InfiniteLine3D
from mmcore.api.enums import NurbsSurfaceProperties
from mmcore.api.vectors import Vector3D, Point3D
from mmcore.geom.plane import plane_from_normal2
from mmcore.geom.vec import orthonormalize


class Cone(Surface):
    """
    cone. A cone is not displayed or saved in a document.
    A cone is used as a wrapper to work with raw cone information.
    A cone has no boundaries.
    The cone always goes to a point in its narrowing direction, and is infinite in its
    widening direction.
    They are created statically using the create method of the Cone class.
    """

    def __init__(self):
        pass

    @classmethod
    def cast(cls, arg) -> Cone:
        return Cone()

    @classmethod
    def create(
        cls, origin: Point3D, axis: Vector3D, radius: float, half_angle: float
    ) -> Cone:
        """
        Creates a cone object.
        origin : The origin point (center) of the base of the cone.
        axis : The center axis (along the length) of the cone that defines its normal direction.
        radius : The radius of the cone.
        half_angle : The taper half-angle of the cone.
        Returns the new Cone object or null if the creation failed.
        """
        return Cone()

    def to_data(self) -> tuple[bool, Point3D, Vector3D, float, float]:
        """
        Gets the data that defines the cone.
        origin : The output origin point (center) of the base of the cone.
        axis : The output center axis (along the length) of the cone that defines its normal direction.
        radius : The output radius of the cone.
        half_angle : The output taper half-angle of the cone.
        Returns true if successful.
        """
        return (bool(), Point3D(), Vector3D(), float(), float())

    def set(
        self, origin: Point3D, axis: Vector3D, radius: float, half_angle: float
    ) -> bool:
        """
        Sets the data that defines the cone.
        origin : The origin point (center) of the base of the cone.
        axis : The center axis (along the length) of the cone that defines its normal direction.
        radius : The radius of the cone.
        half_angle : The taper half-angle of the cone.
        Returns true if successful.
        """
        return bool()

    def copy(self) -> Cone:
        """
        Creates and returns an independent copy of this Cone object.
        Returns a new Cone object that is a copy of this Cone object.
        """
        return Cone()

    @property
    def origin(self) -> Point3D:
        """
        Gets and sets the origin point (center) of the base of the cone.
        """
        return Point3D()

    @origin.setter
    def origin(self, value: Point3D):
        pass

    @property
    def axis(self) -> Vector3D:
        """
        Gets and sets the center axis (along the length) of the cone that defines its
        normal direction.
        """
        return Vector3D()

    @axis.setter
    def axis(self, value: Vector3D):
        pass

    @property
    def radius(self) -> float:
        """
        Gets and sets the radius of the cone.
        """
        return float()

    @radius.setter
    def radius(self, value: float):
        pass

    @property
    def half_angle(self) -> float:
        """
        Gets and sets the taper half-angle of the cone in radians.
        A negative value indicates that the cone is narrowing in the direction of the
        axis vector, whereas a positive value indicates that it is expanding in the direction of
        the axis vector.
        """
        return float()

    @half_angle.setter
    def half_angle(self, value: float):
        pass


class Cylinder(Surface):
    """
    cylinder. A cylinder is not displayed or saved in a document.
    A cylinder is but is used as a wrapper to work with raw cylinder information.
    A cylinder has no boundaries and is infinite in length.
    They are created statically using the create method of the Cylinder class.
    """

    def __init__(self):
        pass

    @classmethod
    def cast(cls, arg) -> Cylinder:
        return Cylinder()

    @classmethod
    def create(cls, origin: Point3D, axis: Vector3D, radius: float) -> Cylinder:
        """
        Creates a cylinder object.
        origin : The origin point (center) of the base of the cylinder.
        axis : The center axis (along the length) of the cylinder that defines its normal direction.
        radius : The radius of the cylinder.
        Returns the new Cylinder object or null if the creation failed.
        """
        return Cylinder()

    def to_data(self) -> tuple[bool, Point3D, Vector3D, float]:
        """
        Gets the data that defines the cylinder.
        origin : The output origin point (center) of the base of the cylinder.
        axis : The output center axis (along the length) of the cylinder that defines its normal direction.
        radius : The output radius of the cylinder.
        Returns true if successful.
        """
        return (bool(), Point3D(), Vector3D(), float())

    def set(self, origin: Point3D, axis: Vector3D, radius: float) -> bool:
        """
        Sets the data that defines the cylinder.
        origin : The origin point (center) of the base of the cylinder.
        axis : The center axis (along the length) of the cylinder that defines its normal direction.
        radius : The radius of the cylinder.
        Returns true if successful.
        """
        return bool()

    def copy(self) -> Cylinder:
        """
        Creates and returns an independent copy of this Cylinder object.
        Returns a new Cylinder object that is a copy of this Cylinder object.
        """
        return Cylinder()

    @property
    def origin(self) -> Point3D:
        """
        The origin point (center) of the base of the cylinder.
        """
        return Point3D()

    @origin.setter
    def origin(self, value: Point3D):
        pass

    @property
    def axis(self) -> Vector3D:
        """
        The center axis (along the length) of the cylinder that defines its normal direction.
        """
        return Vector3D()

    @axis.setter
    def axis(self, value: Vector3D):
        pass

    @property
    def radius(self) -> float:
        """
        The radius of the cylinder.
        """
        return float()

    @radius.setter
    def radius(self, value: float):
        pass


class EllipticalCone(Surface):
    """
    elliptical cone. A elliptical cone is not displayed or saved in a document.
    A elliptical cone is used as a wrapper to work with raw elliptical cone information.
    A elliptical cone has no boundaries.
    The cone always goes to a point in its narrowing direction, and is infinite in its
    widening direction.
    They are created statically using the create method of the EllipticalCone class.
    """

    def __init__(self):
        pass

    @classmethod
    def cast(cls, arg) -> EllipticalCone:
        return EllipticalCone()

    @classmethod
    def create(
        cls,
        origin: Point3D,
        axis: Vector3D,
        major_axisDirection: Vector3D,
        major_radius: float,
        minor_radius: float,
        half_angle: float,
    ) -> EllipticalCone:
        """
        Creates a elliptical cone object.
        origin : The origin point (center) of the base of the cone.
        axis : The center axis (along the length) of the cone that defines its normal direction.
        major_axisDirection : The direction of the major axis of the ellipse that defines the cone.
        major_radius : The major radius of the ellipse that defines the cone.
        minor_radius : The minor radius of the ellipse that defines the cone.
        half_angle : The taper half-angle of the cone.
        Returns the new EllipticalCone object or null if the creation failed.
        """
        return EllipticalCone()

    def get_axes(self) -> tuple[Vector3D, Vector3D]:
        """
        Gets the center axis of the cone that defines its normal direction and the major axis
        direction of the ellipse that defines it.
        axis : The output center axis (along the length) of the cone that defines its normal direction.
        major_axisDirection : The output direction of the major axis of the ellipse that defines the cone.
        """
        return (Vector3D(), Vector3D())

    def set_axes(self, axis: Vector3D, major_axisDirection: Vector3D) -> bool:
        """
        Sets the center axis of the cone and the major axis direction of the ellipse that defines it.
        axis : The center axis (along the length) of the cone that defines its normal direction.
        major_axisDirection : The direction of the major axis of the ellipse that defines the cone.
        Returns true if successful.
        """
        return bool()

    def to_data(self) -> tuple[bool, Point3D, Vector3D, Vector3D, float, float, float]:
        """
        Gets the data that defines the Elliptical Cone.
        origin : The output origin point (center) of the base of the cone.
        axis : The output center axis (along the length) of the cone that defines its normal direction.
        major_axisDirection : The output direction of the major axis of the ellipse that defines the cone.
        major_radius : The output major radius of the ellipse that defines the cone.
        minor_radius : The output minor radius of the ellipse that defines the cone.
        half_angle : The output taper half-angle of the cone.
        Returns true if successful.
        """
        return (bool(), Point3D(), Vector3D(), Vector3D(), float(), float(), float())

    def set(
        self,
        origin: Point3D,
        axis: Vector3D,
        major_axisDirection: Vector3D,
        major_radius: float,
        minor_radius: float,
        half_angle: float,
    ) -> bool:
        """
        Sets the data that defines the Elliptical Cone.
        origin : The origin point (center) of the base of the cone.
        axis : The center axis (along the length) of the cone that defines its normal direction.
        major_axisDirection : The direction of the major axis of the ellipse that defines the cone.
        major_radius : The major radius of the ellipse that defines the cone.
        minor_radius : The minor radius of the ellipse that defines the cone.
        half_angle : The taper half-angle of the cone.
        Returns true if successful.
        """
        return bool()

    def copy(self) -> EllipticalCone:
        """
        Creates and returns an independent copy of this EllipticalCone object.
        Returns a new EllipticalCone object that is a copy of this EllipticalCone object.
        """
        return EllipticalCone()

    @property
    def origin(self) -> Point3D:
        """
        Gets and sets the origin point (center) of the base of the cone.
        """
        return Point3D()

    @origin.setter
    def origin(self, value: Point3D):
        pass

    @property
    def major_radius(self) -> float:
        """
        Gets and sets the major radius of the ellipse that defines the cone.
        """
        return float()

    @major_radius.setter
    def major_radius(self, value: float):
        pass

    @property
    def minor_radius(self) -> float:
        """
        Gets and sets the minor radius of the ellipse that defines the cone.
        """
        return float()

    @minor_radius.setter
    def minor_radius(self, value: float):
        pass

    @property
    def half_angle(self) -> float:
        """
        Gets and sets the taper half-angle of the elliptical cone.
        A negative value indicates that the cone is narrowing in the direction of the axis vector,
        whereas a positive values indicates that it is expanding in the direction of the axis vector.
        """
        return float()

    @half_angle.setter
    def half_angle(self, value: float):
        pass


class EllipticalCylinder(Surface):
    """
    elliptical cylinder. A elliptical cylinder is not displayed or saved
    in a document.
    A elliptical cylinder is used as a wrapper to work with raw elliptical cylinder
    information.
    A elliptical cylinder has no boundaries and is infinite in length.
    They are created statically using the create method of the EllipticalCylinder class.
    """

    def __init__(self):
        pass

    @classmethod
    def cast(cls, arg) -> EllipticalCylinder:
        return EllipticalCylinder()

    @classmethod
    def create(
        cls,
        origin: Point3D,
        axis: Vector3D,
        major_axis: Vector3D,
        major_radius: float,
        minor_radius: float,
    ) -> EllipticalCylinder:
        """
        Creates a 3D elliptical cylinder object.
        origin : The origin point (center) of the base of the cylinder.
        axis : The center axis (along the length) of the cylinder that defines its normal direction.
        major_axis : The direction of the major axis of the ellipse that defines the cylinder.
        major_radius : The major radius of the ellipse that defines the cylinder.
        minor_radius : The minor radius of the ellipse that defines the cylinder.
        Returns the new EllipticalCylinder object or null if the creation failed.
        """
        return EllipticalCylinder()

    def to_data(self) -> tuple[bool, Point3D, Vector3D, Vector3D, float, float]:
        """
        Gets the data defining the elliptical cylinder.
        origin : The output origin point (center) of the base of the cylinder.
        axis : The output center axis (along the length) of the cylinder that defines its normal direction.
        major_axis : The output direction of the major axis of the ellipse that defines the cylinder.
        major_radius : The output major radius of the ellipse that defines the cylinder.
        minor_radius : The output minor radius of the ellipse that defines the cylinder.
        Returns true if successful.
        """
        return (bool(), Point3D(), Vector3D(), Vector3D(), float(), float())

    def set(
        self,
        origin: Point3D,
        axis: Vector3D,
        major_axis: Vector3D,
        major_radius: float,
        minor_radius: float,
    ) -> bool:
        """
        Sets the data defining the elliptical cylinder.
        origin : The origin point (center) of the base of the cylinder.
        axis : The center axis (along the length) of the cylinder that defines its normal direction.
        major_axis : The direction of the major axis of the ellipse that defines the cylinder.
        major_radius : The major radius of the ellipse that defines the cylinder.
        minor_radius : The minor radius of the ellipse that defines the cylinder.
        Returns true if successful.
        """
        return bool()

    def copy(self) -> EllipticalCylinder:
        """
        Creates and returns an independent copy of this EllipticalCylinder object.
        Returns a new EllipticalCylinder object that is a copy of this EllipticalCylinder object.
        """
        return EllipticalCylinder()

    @property
    def origin(self) -> Point3D:
        """
        Gets and sets the origin point (center) of the base of the cylinder.
        """
        return Point3D()

    @origin.setter
    def origin(self, value: Point3D):
        pass

    @property
    def axis(self) -> Vector3D:
        """
        Gets and set the center axis (along the length) of the cylinder that defines
        its normal direction.
        """
        return Vector3D()

    @axis.setter
    def axis(self, value: Vector3D):
        pass

    @property
    def major_axis(self) -> Vector3D:
        """
        Gets and sets the direction of the major axis of the ellipse that defines the cylinder.
        """
        return Vector3D()

    @major_axis.setter
    def major_axis(self, value: Vector3D):
        pass

    @property
    def major_radius(self) -> float:
        """
        Gets and sets the major radius of the ellipse that defines the cylinder.
        """
        return float()

    @major_radius.setter
    def major_radius(self, value: float):
        pass

    @property
    def minor_radius(self) -> float:
        """
        Gets and sets the minor radius of the ellipse that defines the cylinder.
        """
        return float()

    @minor_radius.setter
    def minor_radius(self, value: float):
        pass


class NurbsSurface(Surface):
    """
    NURBS surface. A NURBS surface is not displayed or saved in a document.
    A NURBS surface is used as a wrapper to work with raw NURBS surface information.
    A NURBS surface is bounded by it's natural boundaries and does not support the
    definition of arbitrary boundaries.
    A NURBS surface is typically obtained from a BREPFace object, which does have boundary information.
    They are created statically using the create method of the NurbsSurface class.
    """

    def __init__(self):
        pass

    @classmethod
    def cast(cls, arg) -> NurbsSurface:
        return NurbsSurface()

    @classmethod
    def create(
        cls,
        degree_u: int,
        degree_v: int,
        control_point_count_u: int,
        control_point_count_v: int,
        control_points: list[Point3D],
        knots_u: list[float],
        knots_v: list[float],
        weights: list[float],
        properties_u: NurbsSurfaceProperties,
        properties_v: NurbsSurfaceProperties,
    ) -> NurbsSurface:
        """
        Creates a NURBS surface object.
        degree_u : The degree in the U direction.
        degree_v : The degree in the V direction.
        control_point_count_u : The number of control points in the U direction.
        control_point_count_v : The number of control points in the V direction.
        control_points : An array of surface control points.
        The length of this array must be control_point_count_u * control_point_count_v.
        knots_u : The knot vector for the U direction.
        knots_v : The knot vector for the V direction.
        weights : An array of weights that corresponds to the control points of the surface.
        properties_u : The properties (NurbsSurfaceProperties) of the surface in the U direction.
        properties_u : The properties (NurbsSurfaceProperties) of the surface in the V direction.
        Returns the new NurbsSurface object or null if the creation failed.
        """
        return NurbsSurface()

    def to_data(
        self,
    ) -> tuple[
        bool,
        int,
        int,
        int,
        int,
        list[Point3D],
        list[float],
        list[float],
        list[float],
        NurbsSurfaceProperties,
        NurbsSurfaceProperties,
    ]:
        """
        Gets the data that defines the NURBS surface.
        degree_u : The output degree in the U direction.
        degree_v : The output degree in the V direction.
        control_point_count_u : The output number of control points in the U direction.
        control_point_count_v : The output number of control points in the V direction.
        control_points : An output array of surface control points.
        knots_u : The output knot vector for the U direction.
        knots_v : The output knot vector for the V direction.
        weights : An output array of weights that corresponds to the control points of the surface.
        properties_u : The output properties (NurbsSurfaceProperties) of the surface in the U direction.
        properties_u : The output properties (NurbsSurfaceProperties) of the surface in the V direction.
        Returns true if successful.
        """
        return (
            bool(),
            int(),
            int(),
            int(),
            int(),
            [Point3D()],
            [float()],
            [float()],
            [float()],
            NurbsSurfaceProperties,
            NurbsSurfaceProperties,
        )

    def set(
        self,
        degree_u: int,
        degree_v: int,
        control_point_count_u: int,
        control_point_count_v: int,
        control_points: list[Point3D],
        knots_u: list[float],
        knots_v: list[float],
        weights: list[float],
        properties_u: NurbsSurfaceProperties,
        properties_v: NurbsSurfaceProperties,
    ) -> bool:
        """
        Sets the data that defines the NURBS surface.
        degree_u : The degree in the U direction.
        degree_v : The degree in the V direction.
        control_point_count_u : The number of control points in the U direction.
        control_point_count_v : The number of control points in the V direction.
        control_points : An array of surface control points.
        knots_u : The knot vector for the U direction.
        knots_v : The knot vector for the V direction.
        weights : An array of weights that corresponds to the control points of the surface.
        properties_u : The properties (NurbsSurfaceProperties) of the surface in the U direction.
        properties_u : The properties (NurbsSurfaceProperties) of the surface in the V direction.
        Returns true if successful
        """
        return bool()

    def copy(self) -> NurbsSurface:
        """
        Creates and returns an independent copy of this NurbsSurface object.
        Returns a new NurbsSurface object that is a copy of this NurbsSurface object.
        """
        return NurbsSurface()

    @property
    def control_point_count_u(self) -> int:
        """
        Gets the number of control points in the U direction.
        """
        return int()

    @property
    def control_point_count_v(self) -> int:
        """
        Gets the number of control points in the V direction.
        """
        return int()

    @property
    def degree_u(self) -> int:
        """
        Gets the degree in the U direction.
        """
        return int()

    @property
    def degree_v(self) -> int:
        """
        Gets the degree in the V direction.
        """
        return int()

    @property
    def knot_count_u(self) -> int:
        """
        Gets the knot count in the U direction.
        """
        return int()

    @property
    def knot_count_v(self) -> int:
        """
        Gets thekKnot count in the V direction.
        """
        return int()

    @property
    def properties_u(self) -> NurbsSurfaceProperties:
        """
        Gets the properties (NurbsSurfaceProperties) of the surface in the U direction.
        """
        return NurbsSurfaceProperties

    @property
    def properties_v(self) -> NurbsSurfaceProperties:
        """
        Gets the properties (NurbsSurfaceProperties) of the surface in the V direction.
        """
        return NurbsSurfaceProperties

    @property
    def control_points(self) -> list[Point3D]:
        """
        Gets an array of control points from the surface.
        """
        return [Point3D()]

    @property
    def knots_u(self) -> list[float]:
        """
        Get the knot vector from the U direction.
        """
        return [float()]

    @property
    def knots_v(self) -> list[float]:
        """
        Get the knot vector from the V direction
        """
        return [float()]


class Plane(Surface):
    """
    plane.
    planes are used as a wrapper to work with raw plane information.
    A plane has no boundaries or size, but is infinite and is represented
    by a position, a normal, and an orientation in space.
    They are created statically using the create method of the Plane class.
    """

    def __init__(
        self,
        origin: Point3D = Point3D((0.0, 0.0, 0.0)),
        u_direction: Vector3D = Vector3D((1, 0, 0)),
        v_direction: Vector3D = Vector3D((0, 1, 0)),
        normal: Vector3D = Vector3D((0, 0, 1)),
    ):
        super().__init__()
        self._origin = origin
        self._u_direction = u_direction
        self._v_direction = v_direction
        self._normal = normal

    @classmethod
    def cast(cls, arg: tuple) -> Plane:
        return Plane(*arg)

    @classmethod
    def create_from_array(cls, arr):
        p = Plane()

        p.origin,p._u_direction,p._v_direction,p._normal=Point3D(arr[0]),Vector3D(arr[1]),  Vector3D(arr[2]),  Vector3D(arr[3])
        return p
    @classmethod
    def create(cls, origin: Point3D, normal: Vector3D) -> Plane:
        """
        Creates a plane object by specifying an origin and a normal direction.
        origin : The origin point of the plane.
        normal : The normal direction of the plane.
        Returns the new plane object or null if the creation failed.
        """
        normal.normalize()
        origin, u, v, z = plane_from_normal2(origin._array, normal._array)

        return Plane(origin, Vector3D(u), Vector3D(v), normal)

    @classmethod
    def create_using_directions(
        cls, origin: Point3D, u_direction: Vector3D, v_direction: Vector3D
    ) -> Plane:
        """
        Creates a plane object by specifying an origin along with U and V directions.
        origin : The origin point of the plane.
        u_direction : The U direction for the plane.
        v_direction : The V direction for the plane.
        Returns the new plane object or null if the creation failed.
        """
        u_direction.normalize()
        v_direction.normalize()
        return Plane(origin, u_direction, v_direction, u_direction.cross(v_direction))

    def set_uv_directions(self, u_direction: Vector3D, v_direction: Vector3D) -> bool:
        """
        Sets the U and V directions of the plane.
        u_direction : The U direction for the plane.
        v_direction : The V direction for the plane.
        Returns true if successful.
        """

        self._u_direction = u_direction
        self._v_direction = v_direction
        self._normal = u_direction.cross(v_direction)._array
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

    def intersect_with_plane(self, plane: Plane) -> InfiniteLine3D:
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
        raise NotImplementedError()

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
        return Plane(
            self.origin.copy(),
            self.u_direction.copy(),
            self.v_direction.copy(),
            self.normal.copy(),
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
        self._normal = value
        (
            self._normal._array,
            self._u_direction._array,
            self._v_direction._array,
        ) = orthonormalize(
            self._normal._array, self._u_direction._array, self._v_direction._array
        )

    @property
    def u_direction(self) -> Vector3D:
        """
        Gets the U Direction of the plane.
        """
        return self._u_direction

    @property
    def v_direction(self) -> Vector3D:
        """
        Gets the V Direction of the plane.
        """
        return self._v_direction

    def to_data(self):
        self.origin.to_data()


class Sphere(Surface):
    """
    sphere. A sphere is not displayed or saved in a document.
    spheres are used as a wrapper to work with raw sphere information.
    A sphere is a full sphere defined by a point and a radius.
    They are created statically using the create method of the Sphere class.
    """

    def __init__(self):
        pass

    @classmethod
    def cast(cls, arg) -> Sphere:
        return Sphere()

    @classmethod
    def create(cls, origin: Point3D, radius: float) -> Sphere:
        """
        Creates a sphere object.
        origin : The origin point (center) of the sphere.
        radius : The radius of the sphere.
        Returns the new Sphere object or null if the creation failed.
        """
        return Sphere()

    def to_data(self) -> tuple[bool, Point3D, float]:
        """
        Gets all of the data defining the sphere.
        origin : The output origin point (center) of the sphere.
        radius : The output radius of the sphere.
        Returns true if successful.
        """
        return (bool(), Point3D(), float())

    def set(self, origin: Point3D, radius: float) -> bool:
        """
        Sets all of the data defining the sphere.
        origin : The origin point (center) of the sphere.
        radius : The radius of the sphere.
        Returns true if successful.
        """
        return bool()

    def copy(self) -> Sphere:
        """
        Creates and returns an independent copy of this Sphere object.
        Returns a new Sphere object that is a copy of this Sphere object.
        """
        return Sphere()

    @property
    def origin(self) -> Point3D:
        """
        Gets and sets the origin point (center) of the sphere.
        """
        return Point3D()

    @origin.setter
    def origin(self, value: Point3D):
        pass

    @property
    def radius(self) -> float:
        """
        Gets and sets the radius of the sphere.
        """
        return float()

    @radius.setter
    def radius(self, value: float):
        pass


class Torus(Surface):
    """
    torus. A torus is not displayed or saved in a document.
    A torus is used as a wrapper to work with raw torus information.
    A torus is a full torus with no boundaries.
    They are created statically using the create method of the Torus class.
    """

    def __init__(self):
        pass

    @classmethod
    def cast(cls, arg) -> Torus:
        return Torus()

    @classmethod
    def create(
        cls, origin: Point3D, axis: Vector3D, major_radius: float, minor_radius: float
    ) -> Torus:
        """
        Creates a torus object.
        origin : The origin point (center) of the torus.
        axis : The center axis of the torus.
        major_radius : The major radius of the torus.
        minor_radius : The minor radius of the torus.
        Returns the new Torus object or null if the creation failed.
        """
        return Torus()

    def to_data(self) -> tuple[bool, Point3D, Vector3D, float, float]:
        """
        Gets all of the data defining the torus.
        origin : The output origin point (center) of the torus.
        axis : The output center axis of the torus.
        major_radius : The output major radius of the torus.
        minor_radius : The output minor radius of the torus.
        Returns true if successful.
        """
        return (bool(), Point3D(), Vector3D(), float(), float())

    def set(
        self, origin: Point3D, axis: Vector3D, major_radius: float, minor_radius: float
    ) -> bool:
        """
        Sets all of the data defining the torus.
        origin : The origin point (center) of the torus.
        axis : The center axis of the torus.
        major_radius : The major radius of the torus.
        minor_radius : The minor radius of the torus.
        Returns true if successful.
        """
        return bool()

    def copy(self) -> Torus:
        """
        Creates and returns an independent copy of this Torus object.
        Returns a new Torus object that is a copy of this Torus object.
        """
        return Torus()

    @property
    def origin(self) -> Point3D:
        """
        Gets and sets the origin point (center) of the torus.
        """
        return Point3D()

    @origin.setter
    def origin(self, value: Point3D):
        pass

    @property
    def axis(self) -> Vector3D:
        """
        Gets and sets the center axis of the torus.
        """
        return Vector3D()

    @axis.setter
    def axis(self, value: Vector3D):
        pass

    @property
    def major_radius(self) -> float:
        """
        Gets and sets the major radius of the torus.
        """
        return float()

    @major_radius.setter
    def major_radius(self, value: float):
        pass

    @property
    def minor_radius(self) -> float:
        """
        Gets and sets the minor radius of the torus.
        """
        return float()

    @minor_radius.setter
    def minor_radius(self, value: float):
        pass

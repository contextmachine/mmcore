from __future__ import annotations

from mmcore.api.base import Surface
from mmcore.api.enums import NurbsSurfaceProperties
from mmcore.api.vectors import Vector3D, Point3D


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

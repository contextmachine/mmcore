from __future__ import annotations
from mmcore.api._base import Base

from mmcore.api.vectors import Point2D, Point3D


class BoundingBox2D(Base):
    """
    object that represents a 2D bounding box. A 2D bounding box is a rectangle box that is parallel
    to the x and y axes. The box is defined by a minimum point (smallest x-y values) and maximum point (largest x-y values).
    This object is a wrapper for these points and serves as a way to pass bounding box information
    in and out of functions. It also provides some convenience function when working with the bounding box data.
    They are created statically using the create method of the BoundingBox2D class.
    """

    def __init__(self):
        pass

    @classmethod
    def cast(cls, arg) -> BoundingBox2D:
        return BoundingBox2D()

    @classmethod
    def create(cls, min_point: Point2D, max_point: Point2D) -> BoundingBox2D:
        """
        Creates a bounding box object.
        min_point : The minimum point of the box.
        max_point : The maximum point of the box.
        Returns the new bounding box.
        """
        return BoundingBox2D()

    def contains(self, point: Point2D) -> bool:
        """
        Determines if the specified point lies within the bounding box.
        point : The point to test containment with.
        Returns true if the point lies within the bounding box.
        """
        return bool()

    def expand(self, point: Point2D) -> bool:
        """
        Expand this bounding box to contain the specified point.
        point : The point to expand the box to.
        Returns true if successful.
        """
        return bool()

    def intersects(self, boundingBox: BoundingBox2D) -> bool:
        """
        Test if this bounding box intersects with the specified bounding box.
        boundingBox : The bounding box to test intersection with.
        Returns true if the bounding boxes intersect.
        """
        return bool()

    def copy(self) -> BoundingBox2D:
        """
        Create a copy of this bounding box.
        Returns the new bounding box copy.
        """
        return BoundingBox2D()

    def combine(self, boundingBox: BoundingBox2D) -> bool:
        """
        Combines this bounding box with the input bounding box. If the input
        bounding box extends outside this bounding box then this bounding box will
        be extended to encompass both of the original bounding boxes.
        boundingBox : The other bounding box. It is not edited but is used to extend the boundaries
        of the bounding box the method is being called on.
        Returns true if the combine was successful.
        """
        return bool()

    @property
    def min_point(self) -> Point2D:
        """
        Gets and sets the minimum point of the box.
        """
        return Point2D()

    @min_point.setter
    def min_point(self, value: Point2D):
        """
        Gets and sets the minimum point of the box.
        """
        pass

    @property
    def max_point(self) -> Point2D:
        """
        Gets and sets the maximum point of the box.
        """
        return Point2D()

    @max_point.setter
    def max_point(self, value: Point2D):
        """
        Gets and sets the maximum point of the box.
        """
        pass


class BoundingBox3D(Base):
    """
    object that represents a 3D bounding box.
    It defines a rectangular box whose sides are parallel to the model space x, y, and z
    planes. Because of the fixed orientation of the box it can be fully defined
    by two points at opposing corners; the min and max points. This object is usually
    used to provide a rough approximation of the volume in space that an entity occupies.
    It also provides some convenience function when working with the bounding box data.
    They are created statically using the create method of the BoundingBox3D class.
    """

    def __init__(self):
        pass

    @classmethod
    def cast(cls, arg) -> BoundingBox3D:
        return BoundingBox3D()

    @classmethod
    def create(cls, min_point: Point3D, max_point: Point3D) -> BoundingBox3D:
        """
        Creates a bounding box object. This object is created statically using the BoundingBox3D.create method.
        min_point : The point that defines the minimum corner of the bounding box.
        max_point : The point that defines the maximum corner of the bounding box.
        Returns the newly created bounding box or null if the creation failed.
        """
        return BoundingBox3D()

    def contains(self, point: Point3D) -> bool:
        """
        Determines if the specified point is within the bound box.
        point : The point you want to check to see if it's in the bounding box.
        Returns true if the point is within the bounding box.
        """
        return bool()

    def expand(self, point: Point3D) -> bool:
        """
        Expands the size of bounding box to include the specified point.
        point : The point to include within the bounding box.
        Returns true if the expansion was successful.
        """
        return bool()

    def intersects(self, boundingBox: BoundingBox3D) -> bool:
        """
        Determines if the two bounding boxes intersect.
        boundingBox : The other bounding box to check for intersection with.
        Returns true if the two boxes intersect.
        """
        return bool()

    def copy(self) -> BoundingBox3D:
        """
        Creates an independent copy of this bounding box.
        Returns the new bounding box or null if the copy failed.
        """
        return BoundingBox3D()

    def combine(self, boundingBox: BoundingBox3D) -> bool:
        """
        Combines this bounding box with the input bounding box. If the input
        bounding box extends outside this bounding box then this bounding box will
        be extended to encompass both of the original bounding boxes.
        boundingBox : The other bounding box. It is not edited but is used to extend the boundaries
        of the bounding box the method is being called on.
        Returns true if the combine was successful.
        """
        return bool()

    @property
    def min_point(self) -> Point3D:
        """
        Gets and sets the minimum point corner of the box.
        """
        return Point3D()

    @min_point.setter
    def min_point(self, value: Point3D):
        """
        Gets and sets the minimum point corner of the box.
        """
        pass

    @property
    def max_point(self) -> Point3D:
        """
        Gets and sets the maximum point corner of the box.
        """
        return Point3D()

    @max_point.setter
    def max_point(self, value: Point3D):
        """
        Gets and sets the maximum point corner of the box.
        """
        pass

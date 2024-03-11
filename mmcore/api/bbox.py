from __future__ import annotations

from abc import ABCMeta, abstractmethod

import numpy as np

from mmcore.api._base import Base
from mmcore.api._base_vectors import BaseVector

from mmcore.api.vectors import Point2D, Point3D
from mmcore.func import vectorize
from mmcore.numeric import cartesian_product


@vectorize(signature='(i),(i)->(j,i)')
def box_from_intervals(start, end):
    return cartesian_product(*(np.dstack((start, end))[0]))


def aabb(points: np.ndarray):
    return np.array((np.min(points, axis=len(points.shape) - 2), np.max(points, axis=len(points.shape) - 2)))


class BaseBoundingBox(Base, metaclass=ABCMeta):
    __point_class__ = BaseVector

    def sizes(self):
        return self.max._array - self.min._array

    def __init_subclass__(cls, point_class=BaseVector, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.__point_class__ = point_class

    def __repr__(self):
        return f'{self.__class__.__name__}(min={self.min}, max={self.max})'

    def to_vertices(self):
        return box_from_intervals(self.min._array, self.max._array)

    def __init__(self, pts=None):
        super().__init__()
        self.max = self.__point_class__()
        self.min = self.__point_class__()
        if pts is not None:
            self.set_from_array(np.array(pts))

    def __array__(self, dtype=float):
        return np.array([self.max._array, self.min._array], dtype=dtype)

    def union(self, other: BaseBoundingBox):
        return self.__class__.create_from_array(np.concatenate([np.array(self), np.array(other)]))

    @abstractmethod
    def set_from_array(self, pts):
        pass

    @classmethod
    def cast(cls, arg):
        if isinstance(arg, cls):
            return arg
        elif isinstance(arg, BaseBoundingBox):
            self = cls()
            self.min = cls.__point_class__.cast(arg.min)
            self.max = cls.__point_class__.cast(arg.max)
            return self
        elif isinstance(arg, (np.ndarray, list, tuple)):
            return cls.create_from_array(np.array(arg))
        else:
            raise ValueError(f'Unsupported argument type {type(arg).__name__}, arg={arg}.')

    @classmethod
    def create_from_min_max_points(cls, min_point: __point_class__, max_point: __point_class__):
        """
        Creates a bounding box object.
        min_point : The minimum point of the box.
        max_point : The maximum point of the box.
        Returns the new bounding box.
        """
        self = cls()
        self.max = cls.__point_class__.cast(min_point)
        self.min = cls.__point_class__.cast(max_point)
        return self

    @classmethod
    def create_from_many_points(cls, points: list[__point_class__]):
        """
        Creates a bounding box object.
        min_point : The minimum point of the box.
        max_point : The maximum point of the box.
        Returns the new bounding box.
        """

        self = cls(np.array(points))
        return self

    @classmethod
    def create_from_array(cls, arr: np.ndarray):
        return cls(arr)

    @abstractmethod
    def contains(self, point: __point_class__):
        """
        Determines if the specified point lies within the bounding box.
        point : The point to test containment with.
        Returns true if the point lies within the bounding box.
        """
        pass

    @abstractmethod
    def overlap(self, point: __point_class__) -> bool:
        """
        Determines if the specified point lies within the bounding box.
        point : The point to test containment with.
        Returns true if the point lies within the bounding box.
        """
        pass

    @abstractmethod
    def expand(self, point: __point_class__):
        """
        Expand this bounding box to contain the specified point.
        point : The point to expand the box to.
        Returns true if successful.
        """
        pass

    @abstractmethod
    def intersects(self, boundingBox):
        """
        Test if this bounding box intersects with the specified bounding box.
        boundingBox : The bounding box to test intersection with.
        Returns true if the bounding boxes intersect.
        """
        pass

    def copy(self):
        """
        Create a copy of this bounding box.
        Returns the new bounding box copy.
        """
        return self.__class__.create_from_min_max_points(self.min.copy(), self.max.copy())

    @abstractmethod
    def combine(self, boundingBox):
        """
        Combines this bounding box with the input bounding box. If the input
        bounding box extends outside this bounding box then this bounding box will
        be extended to encompass both of the original bounding boxes.
        boundingBox : The other bounding box. It is not edited but is used to extend the boundaries
        of the bounding box the method is being called on.
        Returns true if the combine was successful.
        """
        pass

    @property
    def min_point(self) -> __point_class__:
        """
        Gets and sets the minimum point of the box.
        """
        return self.min

    @min_point.setter
    def min_point(self, value: __point_class__):
        """
        Gets and sets the minimum point of the box.
        """
        self.min = value

    @property
    def max_point(self) -> __point_class__:
        """
        Gets and sets the maximum point of the box.
        """
        return self.max

    @max_point.setter
    def max_point(self, value: __point_class__):
        """
        Gets and sets the maximum point of the box.
        """
        self.max = value


class BoundingBox2D(BaseBoundingBox, point_class=Point2D):
    """
    object that represents a 2D bounding box. A 2D bounding box is a rectangle box that is parallel
    to the x and y axes. The box is defined by a minimum point (smallest x-y values) and maximum point (largest x-y values).
    This object is a wrapper for these points and serves as a way to pass bounding box information
    in and out of functions. It also provides some convenience function when working with the bounding box data.
    They are created statically using the create method of the BoundingBox2D class.
    """

    def area(self):
        u, v = self.sizes()
        return u * v

    def set_from_array(self, pts: np.ndarray):
        (self.min.x, self.min.y), (self.max.x, self.max.y) = aabb(pts)

    def contains(self, point: Point2D) -> bool:
        """
        Determines if the specified point lies within the bounding box.
        point : The point to test containment with.
        Returns true if the point lies within the bounding box.
        """

        return self.max.x >= point.x >= self.min.x and self.max.y >= point.y >= self.min.y

    def expand(self, point: Point2D) -> bool:
        """
        Expand this bounding box to contain the specified point.
        point : The point to expand the box to.
        Returns true if successful.
        """

        self.max.x = max(self.max.x, point.x)
        self.min.x = min(self.min.x, point.x)
        self.max.y = max(self.max.y, point.y)
        self.min.y = min(self.min.y, point.y)
        return True

    def intersects(self, other: BoundingBox2D) -> bool:
        """
        Test if this bounding box intersects with the specified bounding box.
        boundingBox : The bounding box to test intersection with.
        Returns true if the bounding boxes intersect.
        """
        if ((self.max.x < other.min.x or other.max.x < self.min.x) or
                (self.max.y < other.min.y or other.max.y < self.min.y)):
            return False
        return True

    def combine(self, other: BoundingBox2D) -> bool:
        """
        Combines this bounding box with the input bounding box. If the input
        bounding box extends outside this bounding box then this bounding box will
        be extended to encompass both of the original bounding boxes.
        boundingBox : The other bounding box. It is not edited but is used to extend the boundaries
        of the bounding box the method is being called on.
        Returns true if the combine was successful.
        """
        self.min.x = min(self.min.x, other.min.x)
        self.min.y = min(self.min.y, other.min.y)

        self.max.x = max(self.max.x, other.max.x)
        self.max.y = max(self.max.y, other.max.y)

        return True

    def overlap(self, other: BoundingBox2D) -> bool:

        d1x = other.min.x - self.max.x
        d1y = other.min.y - self.max.y
        d2x = self.min.x - other.max.x
        d2y = self.min.y - other.max.y
        if d1x > 0.0 or d1y > 0.0:
            return False
        if d2x > 0.0 or d2y > 0.0:
            return False
        return True


class BoundingBox3D(BaseBoundingBox, point_class=Point3D):
    """
    object that represents a 3D bounding box.
    It defines a rectangular box whose sides are parallel to the model space x, y, and z
    planes. Because of the fixed orientation of the box it can be fully defined
    by two points at opposing corners; the min and max points. This object is usually
    used to provide a rough approximation of the volume in space that an entity occupies.
    It also provides some convenience function when working with the bounding box data.
    They are created statically using the create method of the BoundingBox3D class.
    """

    def set_from_array(self, pts: np.ndarray):
        print(pts.shape, aabb(pts))
        (self.min.x, self.min.y, self.min.z), (self.max.x, self.max.y, self.max.z) = aabb(pts)

    def volume(self):

        u, v, h = self.sizes()
        return u * v * h

    def contains(self, point: Point3D) -> bool:
        """
        Determines if the specified point is within the bound box.
        point : The point you want to check to see if it's in the bounding box.
        Returns true if the point is within the bounding box.
        """

        return self.max.x >= point.x >= self.min.x and self.max.y >= point.y >= self.min.y

    def expand(self, point: Point3D) -> bool:
        """
        Expands the size of bounding box to include the specified point.
        point : The point to include within the bounding box.
        Returns true if the expansion was successful.
        """
        mem = (self.max.x, self.min.x, self.max.y, self.min.y)
        self.max.x = max(self.max.x, point.x)
        self.min.x = min(self.min.x, point.x)
        self.max.y = max(self.max.y, point.y)
        self.min.y = min(self.min.y, point.y)

        return True

    def intersects(self, other: BoundingBox3D) -> bool:
        """
        Determines if the two bounding boxes intersect.
        boundingBox : The other bounding box to check for intersection with.
        Returns true if the two boxes intersect.
        """
        if (self.max.x < other.min.x or other.max.x < self.min.x) or \
                (self.max.y < other.min.y or other.max.y < self.min.y) or \
                (self.max.z < other.min.z or other.max.z < self.min.z):
            return False
        return True

    def combine(self, other: BoundingBox3D) -> bool:
        """
        Combines this bounding box with the input bounding box. If the input
        bounding box extends outside this bounding box then this bounding box will
        be extended to encompass both of the original bounding boxes.
        boundingBox : The other bounding box. It is not edited but is used to extend the boundaries
        of the bounding box the method is being called on.
        Returns true if the combine was successful.
        """
        self.min.x = min(self.min.x, other.min.x)
        self.min.y = min(self.min.y, other.min.y)
        self.min.z = min(self.min.z, other.min.z)
        self.max.x = max(self.max.x, other.max.x)
        self.max.y = max(self.max.y, other.max.y)
        self.max.z = max(self.max.z, other.max.z)

        return True

    def overlap(self, other: BoundingBox2D) -> bool:
        d1x = other.min.x - self.max.x
        d1y = other.min.y - self.max.y
        d1z = other.min.z - self.max.z

        d2x = self.min.x - other.max.x
        d2y = self.min.y - other.max.y
        d2z = self.min.z - other.max.z

        if d1x > 0.0 or d1y > 0.0 or d1z > 0.0:
            return False

        if d2x > 0.0 or d2y > 0.0 or d2z > 0.0:
            return False

        return True


class IntervalNode:
    def __init__(self):
        self.bbox = BoundingBox2D()
        self.left = None
        self.right = None
        self.node_edges = []


class IntervalTree:
    def __init__(self, pts, edges, bbox):
        self.pts = pts
        self.edges = edges
        self.bbox = bbox
        self.pipResult = False
        self.root = None

    def splitNode(self, node):
        if node.bbox.min.y >= node.bbox.max.y:
            return

        if len(node.node_edges) < 3:
            return

        split = 0.5 * (node.bbox.min.y + node.bbox.max.y)

        node.left = IntervalNode()
        node.right = IntervalNode()

        remaining_node_edges = []
        tmpPt = Point2D()

        for i in range(len(node.node_edges)):
            e = self.edges[node.node_edges[i]]

            p1y = self.pts[e[0]][1]
            p2y = self.pts[e[1]][1]

            if p1y > p2y:
                p1y, p2y = p2y, p1y

            boxPtr = None

            if p2y < split:
                node.left.node_edges.append(node.node_edges[i])
                boxPtr = node.left.bbox
            elif p1y > split:
                node.right.node_edges.append(node.node_edges[i])
                boxPtr = node.right.bbox
            else:
                remaining_node_edges.append(node.node_edges[i])

            if boxPtr is not None:
                boxPtr.expand_by_point(self.pts[e[0]])
                boxPtr.expand_by_point(self.pts[e[1]])

        node.node_edges = remaining_node_edges

        if len(node.left.node_edges):
            self.splitNode(node.left)
        if len(node.right.node_edges):
            self.splitNode(node.right)

    def build(self):
        self.root = IntervalNode()
        self.root.node_edges = list(range(len(self.edges)))
        self.root.bbox = self.bbox.copy()

        self.splitNode(self.root)

    def pointInPolygonRec(self, node, x, y):
        # node.bbox.min.y <= y & & node.bbox.max.y
        if node.bbox.min.y <= y and node.bbox.max.y >= y:
            for i in range(len(node.node_edges)):
                e = self.edges[node.node_edges[i]]

                p1 = self.pts[e[0]]
                yflag0 = (p1[1] >= y)

                p2 = self.pts[e[1]]
                yflag1 = (p2[1] >= y)

                if yflag0 != yflag1:
                    if (((p2[1] - y) * (p1[0] - p2[0]) >= (p2[0] - x) * (p1[1] - p2[1])) == yflag1):
                        self.pipResult = not self.pipResult
        # (nl && nl.bbox.min.y <= y && nl.bbox.max.y >= y
        if node.left and node.left.bbox.min.y <= y and node.left.bbox.max.y >= y:
            self.pointInPolygonRec(node.left, x, y)

        if node.right and node.right.bbox.min.y <= y and node.right.bbox.max.y >= y:
            self.pointInPolygonRec(node.right, x, y)

    @vectorize(excluded=[0], signature='(i)->()')
    def pointInPolygon(self, pt):
        self.pipResult = False
        self.pointInPolygonRec(self.root, pt[0], pt[1])
        return self.pipResult

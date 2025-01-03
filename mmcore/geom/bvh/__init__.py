import itertools
from functools import reduce

import numpy as np

from mmcore.numeric._aabb import aabb,aabb_intersect


class Object3D:


    def __init__(self, bounding_box):
        self.bounding_box = bounding_box



class BVHNode:
    def __init__(self, bounding_box, left=None, right=None, object=None):
        self.bounding_box = bounding_box
        self.left = left
        self.right = right
        self.object = object  # None for internal nodes, leaf node holds the object


_BBOX_CORNERS_BINARY_COMBS={
    1:np.array([[0], [1]],dtype=int),
    2:np.array([[0, 0], [0, 1], [1, 0], [1, 1]],dtype=int),
    3:np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],dtype=int),
    4:np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]],dtype=int)
}
import numpy as np
from functools import reduce

def get_aabb_corners_numpy(point1: np.ndarray, point2: np.ndarray) -> np.ndarray:
    """
    Returns all corners of an Axis-Aligned Bounding Box (AABB) defined by two points using NumPy.

    Parameters:
    - point1: A NumPy array representing the first corner of the AABB.
    - point2: A NumPy array representing the opposite corner of the AABB.

    Returns:
    - A NumPy array of shape (2^N, N), where N is the number of dimensions. Each row represents a corner of the AABB.

    Raises:
    - ValueError: If the input points do not have the same dimensionality.
    """
    if point1.shape != point2.shape:
        raise ValueError("Both points must have the same number of dimensions.")

    # Ensure input points are 1-D arrays
    point1 = point1.flatten()
    point2 = point2.flatten()

    # Determine the minimum and maximum for each dimension
    min_coords = np.minimum(point1, point2)
    max_coords = np.maximum(point1, point2)

    # Stack min and max coordinates for each dimension
    bounds = np.stack((min_coords, max_coords), axis=1)

    # Generate all combinations using binary representations
    num_dims = bounds.shape[0]
    if num_dims in _BBOX_CORNERS_BINARY_COMBS:
        binary_combinations=_BBOX_CORNERS_BINARY_COMBS[num_dims]
    else:
        num_corners = 2 ** num_dims
        # Generate binary representations from 0 to 2^N - 1
        binary_combinations = np.array([list(map(int, bin(i)[2:].zfill(num_dims))) for i in range(num_corners)])
        _BBOX_CORNERS_BINARY_COMBS[num_dims]=binary_combinations

        print(binary_combinations.tolist())
    # Use binary combinations to select min or max for each dimension
    corners = bounds[np.arange(num_dims), binary_combinations]

    return corners



class BoundingBox:
    def __init__(self, min_point, max_point):
        self.min_point = np.array(min_point, dtype=float)  # (x_min, y_min, z_min)
        self.max_point = np.array(max_point, dtype=float)  # (x_max, y_max, z_max)
        self.center = (self.min_point + self.max_point) * 0.5
        self.dims = self.max_point - self.min_point
        self._arr=np.array([self.min_point, self.max_point], dtype=np.float32)

    def split(self, axis=0, parameter=0.5):
        left = BoundingBox(np.copy(self.min_point), np.copy(self.max_point))
        right = BoundingBox(np.copy(self.min_point), np.copy(self.max_point))
        right.min_point[axis] = left.max_point[axis] = (self.min_point[axis] + self.max_point[axis])*parameter
        return left, right

    def get_corners(self):
        return get_aabb_corners_numpy(self.min_point, self.max_point)

    def is_finite(self)->bool:
        return np.all(np.isfinite(self.min_point)) and np.all(np.isfinite(self.max_point))

    def is_non_zero_volume(self)->bool:
        return np.all(self.max_point > self.min_point)

    def is_consistency(self)->bool:
        return np.all(self.max_point >= self.min_point)

    def is_valid(self)->bool:
        return self.is_finite() and self.is_non_zero_volume() and self.is_consistency()

    def split4(self, axis1, axis2, parameter1=0.5, parameter2=0.5):
        a,b = self.split(axis2, parameter1)
        return a.split(axis1,parameter2) + b.split(axis1,parameter2)

    def split8(self, axis1=0, axis2=1, axis3=2, parameter1=0.5, parameter2=0.5, parameter3=0.5):
        return list(reduce(lambda x,y: x+y, (item.split(axis1, parameter1) for item in self.split4(axis2, axis3, parameter2, parameter3))))

    def evaluate(self, uvh):
        return (self.min_point + self.max_point)*uvh

    def centroid(self):
        return (self.min_point + self.max_point)*0.5

    def volume(self):
        return self.dims[0]*self.dims[1]*self.dims[2]

    def intersect(self, other):
        return aabb_intersect(np.asarray(self._arr,dtype=float), np.asarray(other._arr,dtype=float))

    def intersect_disjoint(self, other):
        return (
            self.min_point[0] < other.max_point[0] and
            self.max_point[0] > other.min_point[0] and
            self.min_point[1] < other.max_point[1] and
            self.max_point[1] > other.min_point[1] and
            self.min_point[2] < other.max_point[2] and
            self.max_point[2] > other.min_point[2]
        )

    def intersection(self, other):
        max_min_x = max(self.min_point[0], other.min_point[0])
        max_min_y = max(self.min_point[1], other.min_point[1])
        max_min_z = max(self.min_point[2], other.min_point[2])
        min_max_x = min(self.max_point[0], other.max_point[0])
        min_max_y = min(self.max_point[1], other.max_point[1])
        min_max_z = min(self.max_point[2], other.max_point[2])

        if max_min_x > min_max_x or max_min_y > min_max_y or max_min_z > min_max_z:
            return None

        return BoundingBox((max_min_x, max_min_y, max_min_z),(min_max_x, min_max_y, min_max_z))

    def merge(self, other):
        new_min = (
            min(self.min_point[0], other.min_point[0]),
            min(self.min_point[1], other.min_point[1]),
            min(self.min_point[2], other.min_point[2]),
        )
        new_max = (
            max(self.max_point[0], other.max_point[0]),
            max(self.max_point[1], other.max_point[1]),
            max(self.max_point[2], other.max_point[2]),
        )
        return BoundingBox(new_min, new_max)

    def contains_point(self, point):
        return (self.min_point[0] <= point[0] <= self.max_point[0] and
                self.min_point[1] <= point[1] <= self.max_point[1] and
                self.min_point[2] <= point[2] <= self.max_point[2])
    def contains_points(self, points):
        return ((self.min_point[0] <= points[...,0]) & (points[...,0] <= self.max_point[0] )&
                ((self.min_point[1] <= points[...,1]) & (points[...,1] <= self.max_point[1])) &
                ((self.min_point[2] <= points[...,2])&( points[...,2]<= self.max_point[2])))
    def get_edges(self):
        """
        Return all edges of the box as a list of arrays,
        each edge is [start_corner, end_corner] with shape (2, 3).
        """
        c = self.get_corners()
        edges_indices = [
            (0,1), (2,3), (4,5), (6,7),  # x-direction edges
            (0,2), (1,3), (4,6), (5,7),  # y-direction edges
            (0,4), (1,5), (2,6), (3,7)   # z-direction edges
        ]
        edges = []
        for (i1, i2) in edges_indices:
            edges.append(np.array([c[i1], c[i2]]))
        return edges

    def get_faces(self):
        """
        Return all faces of the box as a list of arrays,
        each face is a quadrilateral represented by 4 corners (4, 3).
        """
        c = self.get_corners()
        faces_indices = [
            (0,1,3,2),  # bottom (z-min)
            (4,5,7,6),  # top (z-max)
            (0,1,5,4),  # front (y-min)
            (2,3,7,6),  # back (y-max)
            (0,2,6,4),  # left (x-min)
            (1,3,7,5)   # right (x-max)
        ]
        faces = []
        for (i1, i2, i3, i4) in faces_indices:
            faces.append(np.array([c[i1], c[i2], c[i3], c[i4]]))
        return faces

    def __repr__(self):
        return f"BoundingBox({self.min_point}, {self.max_point})"

    def __or__(self, other):
        return self.merge(other)

    def __and__(self, other):
        return self.intersection(other)

    def __array__(self,dtype=None,copy=None):
        if int(np.__version__[0])<2:
            return self._arr.__array__(dtype=dtype)

        return self._arr.__array__(dtype=dtype,copy=copy)


def split_objects(objects):
    """Splits list of objects into two halves"""
    # Calculate bounding box of each object
    bounding_boxes = [obj.bounding_box for obj in objects]

    # Compute the midpoint of each centroid
    centroids = [
        [(box.min_point[i] + box.max_point[i]) / 2 for i in range(3)]
        for box in bounding_boxes
    ]

    # Choose the axis to split along (longest axis)
    centroid_array = np.array(centroids)
    min_centroid = np.min(centroid_array, axis=0)
    max_centroid = np.max(centroid_array, axis=0)
    axis = np.argmax(max_centroid - min_centroid)

    # Sort objects along chosen axis by centroids
    objects.sort(
        key=lambda obj: (
            obj.bounding_box.min_point[axis] + obj.bounding_box.max_point[axis]
        )
        / 2
    )

    # Split objects into two halves
    mid_index = len(objects) // 2
    return objects[:mid_index], objects[mid_index:]


def is_leaf(node: BVHNode):
    return node.object is not None

def build_bvh(objects):
    """Recursively build the BVH tree given a list of objects with bounding boxes"""
    if len(objects) == 1:
        # Leaf node
        return BVHNode(objects[0].bounding_box, object=objects[0])
    # Recursively build internal nodes
    left_objs, right_objs = split_objects(objects)
    left_node = build_bvh(left_objs)
    right_node = build_bvh(right_objs)
    # Merge bounding boxes
    merged_bounding_box = left_node.bounding_box.merge(right_node.bounding_box)
    # Create and return internal node
    return BVHNode(merged_bounding_box, left=left_node, right=right_node)


def intersect_bvh_objects(node1, node2):
    """
    Find all intersecting bounding boxes between two BVH trees.
    :param node1: Root of the first BVH tree
    :param node2: Root of the second BVH tree
    :return: List of intersecting bounding boxes

    Note
    ----
    The `intersect_bvh` function efficiently finds all intersecting bounding boxes between two BVH trees
    by simultaneously traversing both trees and pruning out non-intersecting subtrees. It returns a list of merged
    bounding boxes representing the intersections.

    The time complexity of this algorithm is O(n * m) in the worst case, where n and m are the number of nodes in the
    two trees, respectively. However, in practice, the performance is much better due to the pruning of
    non-intersecting subtrees, leading to a significant reduction in the number of comparisons.
    """
    intersections = []
    stack = [(node1, node2)]

    while stack:
        n1, n2 = stack.pop()

        if not n1.bounding_box.intersect(n2.bounding_box):
            continue

        if is_leaf(n1) and is_leaf(n2):
            intersections.append((n1,n2))
        else:
            if not is_leaf(n1):
                if not is_leaf(n2):
                    stack.append((n1.left, n2.left))
                    stack.append((n1.left, n2.right))
                    stack.append((n1.right, n2.left))
                    stack.append((n1.right, n2.right))
                else:
                    stack.append((n1.left, n2))
                    stack.append((n1.right, n2))
            else:
                stack.append((n1, n2.left))
                stack.append((n1, n2.right))

    return intersections


def intersect_bvh(node1, node2):
    """
    Find all intersecting bounding boxes between two BVH trees.
    :param node1: Root of the first BVH tree
    :param node2: Root of the second BVH tree
    :return: List of intersecting bounding boxes

    Note
    ----
    The `intersect_bvh` function efficiently finds all intersecting bounding boxes between two BVH trees
    by simultaneously traversing both trees and pruning out non-intersecting subtrees. It returns a list of merged
    bounding boxes representing the intersections.

    The time complexity of this algorithm is O(n * m) in the worst case, where n and m are the number of nodes in the
    two trees, respectively. However, in practice, the performance is much better due to the pruning of
    non-intersecting subtrees, leading to a significant reduction in the number of comparisons.
    """



    return [n1.bounding_box.intersection(n2.bounding_box) for n1,n2 in intersect_bvh_objects(node1,node2)]








def traverse_bvh(node, target_bbox, results):
    """Find all objects in the BVH tree that intersect with the given bounding box"""

    if not node.bounding_box.intersect(target_bbox):
        return
    if node.object is not None:
        results.append(node.object)

    if node.left:
        traverse_bvh(node.left, target_bbox, results)
    if node.right:
        traverse_bvh(node.right, target_bbox, results)


def traverse_bvh2(node, target_bbox, results):
    """Find all objects in the BVH tree that intersect with the given bounding box"""

    if not node.bounding_box.intersect(target_bbox):
        return
    if node.object is not None:
        results.append(node.object)
    l = len(results)
    if node.left:
        traverse_bvh(node.left, target_bbox, results)
    if node.right:
        traverse_bvh(node.right, target_bbox, results)

    if l == len(results):
        results.append(node.object)
        return


def contains_point(bvh_root, pt):
    results=[]
    pt_box = BoundingBox(pt, pt)
    traverse_bvh(bvh_root, pt_box,   results)
    return results

def contains_point2(bvh_root, pt):
    results=[]
    pt_box = BoundingBox(pt, pt)
    traverse_bvh2(bvh_root, pt_box,   results)
    return results




def traverse_bvh_point(node, target_point):
    """Find all objects in the BVH tree that intersect with the given bounding box"""

    if not node.bounding_box.contains_point(target_point):
        return
    if is_leaf(node):

        return node

    left = right = None
    if node.left:

        left = traverse_bvh_point(node.left, target_point)
    if node.right:
        right = traverse_bvh_point(node.right, target_point)

    if left is None and right is None:

        return node
    elif left is None:

        return right
    elif right is None:
        return left
    else:

        return node








def traverse_leafs(bvh_root, result):
    if bvh_root is None:
        return
    if bvh_root.object is not None:
        result.append(bvh_root.object)
        return
    if bvh_root.left is not None:

        traverse_leafs(bvh_root.left, result)
    if bvh_root.right is not None:
        traverse_leafs(bvh_root.right, result)


def traverse_all_objects_in_node(bvh_root, result):

    if bvh_root.object is not None:
        result.append(bvh_root.object)
    else:
        traverse_all_objects_in_node(bvh_root.left, result)
        traverse_all_objects_in_node(bvh_root.right, result)



def sdBox(p: np.array, b: np.array):
    d = np.abs(p) - b
    print(d)
    return min(np.max([d[0], d[1], d[2]]), 0.0) + np.linalg.norm(np.maximum(d, 0.0))




def traverse_all_bbox(bvh_root, result):
    result.append(
        [tuple(bvh_root.bounding_box.min_point), tuple(bvh_root.bounding_box.max_point)]
    )
    if bvh_root.object is not None:
        return
    else:
        traverse_all_bbox(bvh_root.left, result)
        traverse_all_bbox(bvh_root.right, result)


class Triangle(Object3D):
    def __init__(self, geom):
        self.pts = geom
        self.a, self.b, self.c = self.pts[0], self.pts[1], self.pts[2]
        super(Triangle, self).__init__(BoundingBox(*aabb(geom)))

class Quad(Object3D):
    def __init__(self, pts):
        self.pts = pts
        super().__init__(BoundingBox(*aabb(pts)))

class PTriangle(Triangle):
    def __init__(self, pts, uvs):

        self.uvs = uvs
        super().__init__(pts)

class PQuad(Object3D):
    def __init__(self, pts, uvs):
        self.pts = pts
        self.uvs = uvs
        super().__init__(BoundingBox(*aabb(pts)))


class PSegment(Object3D):
    def __init__(self, pts, t):
        self.pts = pts
        self.t = t
        super().__init__(BoundingBox(*aabb(pts)))
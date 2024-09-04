import numpy as np

def aabb(points: np.ndarray):
    """
     AABB (Axis-Aligned Bounding Box) of a point collection.
    :param points: Points
    :rtype: np.ndarray[(N, K), np.dtype[float]] where:
        - N is a points count.
        - K is the number of dims. For example in 3d case (x,y,z) K=3.
    :return: AABB of a point collection.
    :rtype: np.ndarray[(2, K), np.dtype[float]] at [a1_min, a2_min, ... an_min],[a1_max, a2_max, ... an_max],
    """

    return np.array(
        (
            np.min(points, axis=len(points.shape) - 2),
            np.max(points, axis=len(points.shape) - 2),
        )
    )
class Object3D:
    def __init__(self, bounding_box):
        self.bounding_box = bounding_box



class BVHNode:
    def __init__(self, bounding_box, left=None, right=None, object=None):
        self.bounding_box = bounding_box
        self.left = left
        self.right = right
        self.object = object  # None for internal nodes, leaf node holds the object



class BoundingBox:
    def __init__(self, min_point, max_point):
        self.min_point = min_point  # Should be a tuple (x_min, y_min, z_min)
        self.max_point = max_point  # Should be a tuple (x_max, y_max, z_max)
        self.center = np.average([min_point, max_point], axis=0)
        self.dims = np.array(
            [
                self.max_point[0] - self.min_point[0],
                self.max_point[1] - self.min_point[1],
                self.max_point[2] - self.min_point[2],
            ]
        )
    def volume(self):
        return self.dims[0]*  self.dims[1]*  self.dims[2]
    def intersect(self, other):
        """Check if this bounding box intersects with another"""
        return (
            self.min_point[0] <= other.max_point[0]
            and self.max_point[0] >= other.min_point[0]
            and self.min_point[1] <= other.max_point[1]
            and self.max_point[1] >= other.min_point[1]
            and self.min_point[2] <= other.max_point[2]
            and self.max_point[2] >= other.min_point[2]
        )
    def intersect_disjoint(self, other):
        return (
                self.min_point[0] < other.max_point[0]
                and self.max_point[0] > other.min_point[0]
                and self.min_point[1] < other.max_point[1]
                and self.max_point[1] > other.min_point[1]
                and self.min_point[2] < other.max_point[2]
                and self.max_point[2] > other.min_point[2]
        )
    def intersection(self, other):
        """
        Calculate the intersection of this bounding box with another bounding box.

        :param other: The other bounding box to intersect with.
        :return: A new BoundingBox representing the intersection, or None if there is no intersection.
        """
        # Calculate the maximum of the minimum points for each dimension
        max_min_x = max(self.min_point[0], other.min_point[0])
        max_min_y = max(self.min_point[1], other.min_point[1])
        max_min_z = max(self.min_point[2], other.min_point[2])

        # Calculate the minimum of the maximum points for each dimension
        min_max_x = min(self.max_point[0], other.max_point[0])
        min_max_y = min(self.max_point[1], other.max_point[1])
        min_max_z = min(self.max_point[2], other.max_point[2])

        # Check if the bounding boxes intersect
        if max_min_x > min_max_x or max_min_y > min_max_y or max_min_z > min_max_z:
            return None

        # Create and return the intersection bounding box
        intersection_min = (max_min_x, max_min_y, max_min_z)
        intersection_max = (min_max_x, min_max_y, min_max_z)
        return BoundingBox(intersection_min, intersection_max)

    def merge(self, other):
        """Create a new bounding box that contains both this and another bounding box"""
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
        return all(
            (
                self.min_point[0] <= point[0],
                self.min_point[1] <= point[1],
                self.min_point[2] <= point[2],
                self.max_point[0] >= point[0],
                self.max_point[1] >= point[1],
                self.max_point[2] >= point[2],
            )
        )

    def __repr__(self):
        return f"BoundingBox({self.min_point}, {self.max_point})"
    def __or__(self, other):
        return self.merge(other)
    def __and__(self, other):
        return self.intersection(other)


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



def contains_point(bvh_root, pt):
    results=[]
    pt_box = BoundingBox(pt, pt)
    traverse_bvh(bvh_root, pt_box,   results)
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
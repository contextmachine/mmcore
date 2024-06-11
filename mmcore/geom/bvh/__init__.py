import numpy as np
from mmcore.numeric.aabb import aabb
class Object3D:
    def __init__(self, bounding_box):
        self.bounding_box = bounding_box

def sd_triangle(p: np.array, a: np.array, b: np.array, c: np.array):
    ba = b - a
    pa = p - a
    cb = c - b
    pb = p - b
    ac = a - c
    pc = p - c
    nor = np.cross(ba, ac)
    S = _check_normals(p, a, b, c, np.cross(ba, c - a))
    condition = (
        sign_func(scalar_dot(np.cross(ba, nor), pa))
        + sign_func(scalar_dot(np.cross(cb, nor), pb))
        + sign_func(scalar_dot(np.cross(ac, nor), pc))
        < 2.0
    )

    if condition:
        return (
            np.sqrt(
                min(
                    min(
                        dot2(ba * clamp(scalar_dot(ba, pa) / dot2(ba), 0.0, 1.0) - pa),
                        dot2(cb * clamp(scalar_dot(cb, pb) / dot2(cb), 0.0, 1.0) - pb),
                    ),
                    dot2(ac * clamp(scalar_dot(ac, pc) / dot2(ac), 0.0, 1.0) - pc),
                )
            )
            * S
        )
    else:
        return np.sqrt(scalar_dot(nor, pa) * scalar_dot(nor, pa) / dot2(nor)) * S

class Triangle(Object3D):
    def __init__(self, geom):
        self.pts = geom
        self.a, self.b, self.c = self.pts[0], self.pts[1], self.pts[2]
        super(Triangle, self).__init__(BoundingBox(*aabb(geom)))


class BVHNode:
    def __init__(self, bounding_box, left=None, right=None, object=None):
        self.bounding_box = bounding_box
        self.left = left
        self.right = right
        self.object = object  # None for internal nodes, leaf node holds the object


def sdBox(p: np.array, b: np.array):
    d = np.abs(p) - b
    return min(np.max([d[0], d[1], d[2]]), 0.0) + np.linalg.norm(np.maximum(d, 0.0))


def maxcomp(x: np.array):
    return np.max(x)


def msign(x: np.array):
    return np.sign(x)



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


def is_leaf(node: BVHNode):
    return node.object is not None


def sd_mesh(node, target_point, min_dist=np.inf):
    target_point = (
        target_point if isinstance(target_point, np.ndarray) else np.array(target_point)
    )
    minDist = min_dist
    _sign = 1
    stack = [node]
    object3d = None
    while True:
        node = stack.pop()
        sdb = sdBox(
            node.bounding_box.center - target_point,
            node.bounding_box.dims / 2,
        )

        if sdb <= abs(minDist):

            if node.object is not None:

                tri = node.object

                _minDist = sd_triangle(target_point, tri.a, tri.b, tri.c)
                if abs(_minDist) < abs(minDist):
                    minDist = _minDist
                    object3d = tri

            else:
                i = sorted([0, 1, 2], key=lambda x: node.bounding_box.dims[x])[-1]
                closest = (
                    [node.right, node.left]
                    if target_point[i] < node.bounding_box.center[i]
                    else [node.left, node.right]
                )
                for obj in closest:
                    if obj is not None:

                        stack.append(obj)

        if len(stack) == 0:
            break
    return minDist, object3d


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


def traverse_all_bbox(bvh_root, result):
    result.append(
        [tuple(bvh_root.bounding_box.min_point), tuple(bvh_root.bounding_box.max_point)]
    )
    if bvh_root.object is not None:
        return
    else:
        traverse_all_bbox(bvh_root.left, result)
        traverse_all_bbox(bvh_root.right, result)


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


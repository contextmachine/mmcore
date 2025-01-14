from __future__ import annotations
from functools import reduce

import numpy as np
from numpy.typing import NDArray

from mmcore.numeric.aabb import aabb,aabb_intersect,ray_aabb_intersect,segment_aabb_intersect,segment_aabb_clip
from mmcore.geom.nurbs import NURBSCurve,split_curve
from mmcore.numeric.vectors import scalar_unit,scalar_norm,scalar_dot
from mmcore.numeric.algorithms.moller import intersect_triangles_segment_one


MAX_FLOAT64=MAX_PYFLOAT=float(np.finfo(float).max)
_BBOX_CORNERS_BINARY_COMBS={
    1:np.array([[0], [1]],dtype=int),
    2:np.array([[0, 0], [0, 1], [1, 0], [1, 1]],dtype=int),
    3:np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 1, 1], [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 1, 1]],dtype=int),
    4:np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0], [0, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 1], [0, 1, 1, 0], [0, 1, 1, 1], [1, 0, 0, 0], [1, 0, 0, 1], [1, 0, 1, 0], [1, 0, 1, 1], [1, 1, 0, 0], [1, 1, 0, 1], [1, 1, 1, 0], [1, 1, 1, 1]],dtype=int)
}

# BBox
class BoundingBox:
    __slots__ = ('min_point', 'max_point', 'center', 'dims','_arr')
    def __init__(self, min_point, max_point):
        self.min_point = np.array(min_point, dtype=float)  # (x_min, y_min, z_min)
        self.max_point = np.array(max_point, dtype=float)  # (x_max, y_max, z_max)
        self.center = (self.min_point + self.max_point) * 0.5
        self.dims = self.max_point - self.min_point
        self._arr=np.array([self.min_point, self.max_point], dtype=float)

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
    def expand(self, other:BoundingBox):
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
        self.min_point[:]=new_min
        self.max_point[:]=new_max

    def contains_point(self, point):
        return bool(np.all(self.min_point<=point)and np.all(point<=self.max_point))
    def contains_points(self, points):
        return ((self.min_point[0] <= points[...,0]) & (points[...,0] <= self.max_point[0] )&
                ((self.min_point[1] <= points[...,1]) & (points[...,1] <= self.max_point[1])) &
                ((self.min_point[2] <= points[...,2])& ( points[...,2]<= self.max_point[2])))
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

    def size_metric(self):
        """

        Returns: the bbox size metric, even if bbox has insufficient dimensionality .
        Example for three dimensions: If bbox has no dimensions close to 0, it will return volume, if one of the 3d bbox dimensions is close to zero, it will return area, etc.
        -------

        """
        m=1.
        for dim in self.dims:
            if not np.isclose(dim,0):
                m*=dim
        return m

# Object3D subclasses
class Object3D:

    __slots__=('bounding_box','value')

    def __init__(self, bounding_box, *, value=None):
        self.bounding_box = bounding_box
        self.value=value

class Triangle(Object3D):
    __slots__=('bounding_box','value','pts','a','b','c')

    def __init__(self, geom, **kwargs):
        self.pts = geom
        self.a, self.b, self.c = self.pts[0], self.pts[1], self.pts[2]
        super(Triangle, self).__init__(BoundingBox(*aabb(geom)),**kwargs)

class Quad(Object3D):
    __slots__ = ('bounding_box', 'value', 'pts')
    def __init__(self, pts, **kwargs):
        self.pts = pts
        super().__init__(BoundingBox(*aabb(pts)), **kwargs)

class PTriangle(Triangle):
    __slots__=('bounding_box','value','pts','a','b','c','uvs')
    def __init__(self, pts, uvs, **kwargs):

        self.uvs = uvs
        super().__init__(pts,**kwargs)

class PQuad(Quad):
    __slots__ = ('bounding_box', 'value', 'pts','uvs')
    def __init__(self, pts, uvs,**kwargs):

        self.uvs = uvs
        super().__init__(pts,**kwargs)

class PSegment(Object3D):
    __slots__ = ('bounding_box', 'value', 'pts', 't')
    def __init__(self, pts, t,**kwargs):
        self.pts = pts
        self.t = t
        super().__init__(BoundingBox(*aabb(pts)),**kwargs)

class NURBSCurveObject3D(Object3D):
    __slots__ = ('bounding_box', 'value', 'curve')
    def __init__(self, curve:NURBSCurve,**kwargs):
        super().__init__(BoundingBox(*np.array(curve.bbox()),**kwargs))
        self.curve=curve

    def split(self, t:float=None):
        if t is None:
            t0,t1=self.curve.interval()
            t=(t0+t1)/2
        try:
            crv1,crv2=split_curve(self.curve,t, normalize_knots=False,tol=1e-12)
        except ValueError as err:
            print(
                t
            )
            raise err
        return NURBSCurveObject3D(crv1),NURBSCurveObject3D(crv2)

class Segment(Object3D):
    __slots__ = ('bounding_box', 'value', '_arr','start', 'end', 'direction')
    def __init__(self, start: NDArray[np.floating], end: NDArray[np.floating],**kwargs):
        self._arr=np.array([start,end])

        self.start = self._arr[0,:]
        self.end = self._arr[1,:]
        self.direction=end-start

        super().__init__(BoundingBox(*np.array(aabb(np.array([self.start,self.end])))),**kwargs)

    def evaluate(self, t:float):
        return self.start+self.direction*t
    def subdivide(self, t:float=0.5):
        pt = self.start+self.direction*t
        return Segment(self.start, pt), Segment(pt, self.end)

    def clip(self, bbox:BoundingBox)->Segment|None:
        t=segment_aabb_intersect(bbox._arr,self._arr)
        if t is not None:
            start,end=self.evaluate(t[0]), self.evaluate(t[1])
            return Segment(start, end)


    def __arr__(self, dtype=None,copy=None):
        return self._arr.__array__(dtype,copy=copy)

    def __iter__(self):
        return iter(self._arr.tolist())
    def length_sq(self):
        return scalar_dot(self.direction,self.direction)
    def length(self):
        return scalar_norm(self.direction)
    def is_zero(self):
        return scalar_dot(self.direction,self.direction)<1e-15

class Ray(Segment):
    __slots__ = ('bounding_box', 'value', '_arr', 'start', 'end', 'direction')
    def __init__(self, start: NDArray[np.floating], direction, **kwargs):
        direction=np.array(scalar_unit(direction))
        super().__init__(start, start+direction*MAX_FLOAT64, **kwargs)

# BVHNode
class BVHNode:
    __slots__ = ('bounding_box', 'left','right','object',"max_objects_in_leaf")
    def __init__(self, bounding_box, left=None, right=None, object=None,max_objects_in_leaf=1):
        self.bounding_box = bounding_box
        self.left = left
        self.right = right
        self.object = object  # None for internal nodes, leaf node holds the object
        self.max_objects_in_leaf=max_objects_in_leaf

# Functions
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
class CompoundObject3D(Object3D):
    def __init__(self, objects:list[Object3D],**kwargs):

        if len(objects)==0:

                raise ValueError("Empty objects list in CompoundObject3D")
        self._arr=np.zeros((len(objects)*2,3))
        self.objects =[]
        for i in range(len(objects)):
            self._arr[i*2:i*2+2,:]=objects[i].bounding_box._arr
            self.objects.append(objects[i])


        super().__init__(BoundingBox(*aabb(self._arr)),**kwargs)





def build_bvh(objects, objects_in_leaf=1):
    """Recursively build the BVH tree given a list of objects with bounding boxes"""


    if len(objects) == 1 and objects_in_leaf==1:
        # Leaf node

        return BVHNode(objects[0].bounding_box, object=objects[0],max_objects_in_leaf=objects_in_leaf)

    elif len(objects) <= objects_in_leaf:
        co=CompoundObject3D(objects)
        return BVHNode(co.bounding_box,object=co,max_objects_in_leaf=objects_in_leaf)



    # Recursively build internal nodes
    left_objs, right_objs = split_objects(objects)
    left_node = build_bvh(left_objs, objects_in_leaf)
    right_node = build_bvh(right_objs, objects_in_leaf)
    # Merge bounding boxes
    merged_bounding_box = left_node.bounding_box.merge(right_node.bounding_box)
    # Create and return internal node
    return BVHNode(merged_bounding_box, left=left_node, right=right_node,max_objects_in_leaf=objects_in_leaf)

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

def _find_closest_vicinity(bvh:BVHNode, point:NDArray[float])-> tuple[list[Object3D],float] | None:


    if bvh.object is not None:
        return [bvh.object],sd_aabb( bvh.bounding_box._arr,point )
    if not bvh.bounding_box.contains_point(point):
        return
    if (bvh.left is not None) and (bvh.right is not None):

        l,r=bvh.left.bounding_box.contains_point(point) ,bvh.right.bounding_box.contains_point(point)

        if l and (not r):

            return _find_closest_vicinity(bvh.left, point)
        elif (not l) and  r:

            return _find_closest_vicinity(bvh.right, point)
        elif l and r :





                r1=_find_closest_vicinity(bvh.left, point)
                r2=_find_closest_vicinity(bvh.right, point)
                if (r1 is not None) and (r2 is not None):
                    
                    left,left_sd= r1
                    right,right_sd=r2
                    if left_sd<right_sd:
                        return left,left_sd
                    elif left_sd>right_sd:
                        return right,right_sd
                    else:



                        # It is possible if the point lies in the intersection zone of two bboxes.
                        # In this case we can't make a decision, because in fact we need to check both objects
                        return left+right,left_sd
                elif r1 is None and r2 is not None:
                    return r2
                elif r2 is None and r1 is not None:

                    return r1
                else:
                   return
        #If we're here, then the point is inside the parent's box, but not inside any of the children.


    elif bvh.left is not None :
        return _find_closest_vicinity(bvh.left, point)
    elif bvh.right is not None:
        return _find_closest_vicinity(bvh.right, point)

    else:
        raise ValueError(f"Empty BVH node with bbox: {bvh.bounding_box._arr.tolist()}")

def _find_closest_breadth(bvh:BVHNode, point:NDArray[float])->tuple[list[Object3D],float]:
    if bvh.object is not None:
        return [bvh.object],sd_aabb( bvh.bounding_box._arr,point )

    if (bvh.left is not None) and (bvh.right is None):
        return _find_closest_breadth(bvh.left, point)
    elif (bvh.left is None) and (bvh.right is not None):
        return _find_closest_breadth(bvh.right, point)
    elif  (bvh.left is  None) and (bvh.right is  None):
        raise ValueError(f"Empty BVH node with bbox: {bvh.bounding_box._arr.tolist()}")
    elif (bvh.left is not None) and (bvh.right  is not None):
        left_sd=sd_aabb(bvh.left.bounding_box._arr, point)
        right_sd = sd_aabb(bvh.right.bounding_box._arr, point)
        if left_sd<right_sd:
            return _find_closest_breadth(
                bvh.left,point
            )
        elif left_sd>right_sd:
            return _find_closest_breadth(
                bvh.right,point
            )
        else:
            left_obj, left_sd=_find_closest_breadth(
                bvh.left, point
            )
            right_obj, right_sd=_find_closest_breadth(
                bvh.right, point
            )
            if left_sd<right_sd:
                return left_obj,left_sd
            elif left_sd>right_sd:
                return right_obj,right_sd
            else:
                return left_obj+right_obj,left_sd

    else:
        raise ValueError(f"Unknown condition: {bvh.bounding_box._arr.tolist()}, {bvh.left},{bvh.right}")

def find_closest(bvh:BVHNode, point:NDArray[float], breadth:bool=True)->tuple[list[Object3D],float]|None:
    if breadth:
        return _find_closest_breadth(bvh, point)
    else:
        return _find_closest_vicinity(bvh, point)

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

    return min(np.max([d[0], d[1], d[2]]), 0.0) + np.linalg.norm(np.maximum(d, 0.0))

def sd_aabb(bbox, pt):
    cnt=(bbox[0]+bbox[1])/2
    return sdBox( pt-cnt,bbox[1]-cnt)

def traverse_all_bbox(bvh_root, result):
    result.append(
        [tuple(bvh_root.bounding_box.min_point), tuple(bvh_root.bounding_box.max_point)]
    )
    if bvh_root.object is not None:
        return
    else:
        traverse_all_bbox(bvh_root.left, result)
        traverse_all_bbox(bvh_root.right, result)




def bvh_segment_intersection_recursive(bvh:BVHNode,segment:Segment):


    def inner(bvh,segment, ints):
        if not bvh.bounding_box.intersect(segment.bounding_box):
            return
        segment=segment.clip(bvh.bounding_box)
        if segment is None :
            return
        if bvh.object is not None:
            ints.append((bvh,segment))
            return
        else:
            inner(bvh.left, segment, ints)
            inner(bvh.right, segment, ints)

    intersections=[]
    inner(bvh,segment,intersections)
    return intersections


def bvh_segment_intersection(bvh: BVHNode, segment:NDArray[float]|Segment):
    if not isinstance(segment,np.ndarray):
        segment=np.array(segment)

    intersections = []
    stack = [(bvh, segment)]

    while stack:
        current_bvh, current_segment = stack.pop()
        bb=aabb(segment)

        if not aabb_intersect(current_bvh.bounding_box._arr,bb):
            continue

        current_segment = segment_aabb_clip(current_bvh.bounding_box._arr, segment )

        if current_segment is None:
            continue

        if current_bvh.object is not None:
            intersections.append((current_bvh.object, current_segment))
        else:

            stack.append((current_bvh.left, current_segment))
            stack.append((current_bvh.right, current_segment))


    return intersections


#  72208
def bvh_triangle_segment_intersection_one(bvh: BVHNode, segment:NDArray[float]|Segment):
    if not isinstance(segment,np.ndarray):
        segment=np.array(segment)


    stack = [(bvh, segment)]
    compound=bvh.max_objects_in_leaf>1

    if compound:
        a = np.zeros((bvh.max_objects_in_leaf,3))
        b = np.zeros((bvh.max_objects_in_leaf,3))
        c = np.zeros((bvh.max_objects_in_leaf,3))
    while stack:
        current_bvh, current_segment = stack.pop()
        bb=aabb(segment)

        if not aabb_intersect(current_bvh.bounding_box._arr,bb):
            continue

        current_segment = segment_aabb_clip(current_bvh.bounding_box._arr, segment )

        if current_segment is None:
            continue

        if current_bvh.object is not None:

            if compound:

                a[:] = 0
                b[:] = 0
                c[:] = 0
                current_size=len(current_bvh.object.objects)
                for ixs in range(current_size):
                    a[ixs]=current_bvh.object.objects[ixs].a
                    b[ixs]= current_bvh.object.objects[ixs].b
                    c[ixs] = current_bvh.object.objects[ixs].c

                point, flag = intersect_triangles_segment_one(a[:current_size], b[:current_size],
                                                         c[:current_size], current_segment[0], current_segment[1])
            else:
                point,flag=intersect_triangle_segment(current_bvh.object.a,current_bvh.object.b,current_bvh.object.c,current_segment[0],current_segment[1])
            if flag!=0:
                return True,point,current_bvh.object

        else:

            stack.append((current_bvh.left, current_segment))
            stack.append((current_bvh.right, current_segment))


    return False,None,None


# perf history
# 495417
# 458500
# 436042
# 388833
# 374375
# 344167
# 320625
# 147500
# 139292
# 138250
# 120625

from mmcore.numeric.vectors import dot_array_x_vec
def bvh_ray_intersection(bvh:BVHNode, ray:Ray):
    """
    Determines the intersection of a BVH (Bounding Volume Hierarchy) node with a given ray.
    Calculates whether the ray interacts with the BVH bounding volume. In case there is an
    intersection, it further evaluates segment intersections to return the results. The
    method utilizes AABB (Axis-Aligned Bounding Box) clipping for determining segments of
    intersection before recursively checking against child nodes.

    :param bvh: The root node of the bounding volume hierarchy that represents the scene or
                object's spatial subdivision.
    :type bvh: BVHNode
    :param ray: Represents the ray to test for intersection with the BVH volumes.
    :type ray: Ray
    :return: A list of intersected primitives or nodes derived from the BVH where the ray
             intersects, or an empty list if no intersections are found.
    :rtype: list
    """
    segm=segment_aabb_clip(bvh.bounding_box._arr, ray._arr)
    if segm is None:
        return []
    else:
        result=bvh_segment_intersection(bvh,segm)



        return result


def build_bvh_from_mesh(points:NDArray[float],indices:NDArray[int], triangles_in_leaf=1):

    return build_bvh([Triangle(tri) for tri in points[indices]],objects_in_leaf=triangles_in_leaf)
def build_bvh_from_triangles(triangles:list[Triangle], triangles_in_leaf=1):

    return build_bvh(triangles,objects_in_leaf=triangles_in_leaf)

from mmcore.numeric.algorithms.moller import intersect_triangle_segment

def mesh_bvh_ray_intersection(triangles_bvh:BVHNode,ray:Ray):
    """
    Determines the intersection points where a ray intersects with the triangles
    within a bounding volume hierarchy (BVH). The function processes a ray,
    determines intersections with a BVH structure, and evaluates each triangle
    intersection, returning a list of intersected points, triangles, and their
    respective distances sorted by proximity from the ray origin.

    :param triangles_bvh: A bounding volume hierarchy (BVH) node that contains the
                          spatial hierarchy and triangle geometry for intersection
                          testing.
    :type triangles_bvh: BVHNode
    :param ray: The ray to be tested for intersection with the BVH and contained
                triangles.
    :type ray: Ray
    :return: A list of tuples representing the intersection data. Each tuple
             contains the intersection point, the intersected triangle, and the
             scalar distance from the ray origin.
    :rtype: list[tuple[Point, Triangle, float]]
    """
    maybe = []
    segment = segment_aabb_clip(triangles_bvh.bounding_box._arr, ray._arr)
    if segment is None:
        return maybe
    direction=segment[1]-segment[0]
    for tri,segm in bvh_segment_intersection(triangles_bvh,segment):



        point,flag=intersect_triangle_segment(tri.pts[0],tri.pts[1],tri.pts[2],segm[0],segm[1])
        if flag==0:

            continue
        #if len(maybe)>0 and scalar_dot(maybe[-1]-point)<1e-8:
        #    maybe.append(point)

        maybe.append((point,tri,scalar_dot(point-ray.start,direction)))

    maybe.sort(key=lambda x:x[-1])
    return maybe




def mesh_bvh_segment_intersection_one(mesh_bvh:BVHNode, segment:NDArray[float])->tuple[NDArray[float],Triangle] | None:
    """

    :param mesh_bvh: BVH with Triangle instances in leafs
    :param segment: numpy array with shape (2,3) e.g. [start_point,end_point]

    :return: the first intersection found in the form (point,triangle) or None if no intersection is found.
    :rtype: tuple[ndarray[(3,),float], Triangle] | None
    """
    success, point, tri = bvh_triangle_segment_intersection_one(mesh_bvh,segment)
    if success:
        return point,tri


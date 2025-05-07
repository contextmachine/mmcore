from __future__ import annotations
import dataclasses
from dataclasses import InitVar,field
from typing import Optional
import numpy as np

from numpy.typing import NDArray
def left(i):
    return 2 * i + 1


def right(i):
    return 2 * i + 2


def parent(i):
    return (i - 1) / 2


@dataclasses.dataclass
class AABB:

    min: np.ndarray = np.inf
    max: np.ndarray = -np.inf

    @property
    def dim(self):
        return self.min.shape[-1]

    def merge(self, other: "AABB"):

        return AABB(np.min([self.min, other.min], axis=0), np.max([self.max, other.max], axis=0))

    @classmethod
    def from_points(cls, pts):
        ptsarr=np.asarray(pts)
        return AABB(np.min(ptsarr,axis=0),np.max(ptsarr,axis=0))

    @classmethod
    def from_dim(cls, dim:int=3):
        _bnds = np.zeros((2,dim))
        ptsarr = np.asarray(_bnds)
        return AABB(np.min(ptsarr, axis=0), np.max(ptsarr, axis=0))

    def intersects(self, other: "AABB") -> bool:
        """Return True if this box intersects with another."""
        # No intersection if one is completely to one side of the other
        return not (np.any(self.max < other.min) or np.any(self.min > other.max))

    def intersects_exact(self, other: "AABB") -> bool:
        """Return True if this box intersects with another."""
        # No intersection if one is completely to one side of the other
        return not (np.any(self.max <= other.min) or np.any(self.min >= other.max))
    def offset(self, d:float):
        return AABB(self.min-d,self.max+d)
    def offset_inplace(self, d:float):
        self.min-=d
        self.max+=d

@dataclasses.dataclass
class BVHNode:
    bbox: AABB = dataclasses.field(default_factory=AABB)
    left: int = -1
    right: int = -1

    object: int = -1

    def is_leaf(self):
        return self.object!= -1


def split_objects(objects: list[tuple[int, AABB]])-> tuple[list[tuple[int, AABB]], list[tuple[int, AABB]]]:
    """Splits list of objects into two halves"""
    # Calculate bounding box of each object
    assert len(objects)>0
    # Compute the midpoint of each centroid
    centroids = [[(box.min[i] + box.max[i]) / 2 for i in range(box.dim)] for ix, box in objects]

    # Choose the axis to split along (longest axis)
    centroid_array = np.array(centroids)
    min_centroid = np.min(centroid_array, axis=0)
    max_centroid = np.max(centroid_array, axis=0)
    axis = np.argmax(max_centroid - min_centroid)

    # Sort objects along chosen axis by centroids

    objects.sort(key=lambda obj: (obj[1].min[axis] + obj[1].max[axis]) / 2)

    # Split objects into two halves

    mid_index = len(objects) // 2
    # print(objects[:mid_index], objects[mid_index:])

    return objects[:mid_index], objects[mid_index:]


def split_objects_v2(objects: list[tuple[int, AABB]]) -> tuple[list[tuple[int, AABB]], list[tuple[int, AABB]]]:
    """Splits list of objects into two halves"""
    # Calculate bounding box of each object
    assert len(objects) > 0
    # Compute the midpoint of each centroid
    centroids = [[(box.min[i] + box.max[i]) / 2 for i in range(box.dim)] for ix, box in objects]

    # Choose the axis to split along (longest axis)
    centroid_array = np.array(centroids)
    min_centroid = np.min(centroid_array, axis=0)
    max_centroid = np.max(centroid_array, axis=0)
    axis = np.argmax(max_centroid - min_centroid)

    # Sort objects along chosen axis by centroids

    objects.sort(key=lambda obj: obj[1].min[axis]+(obj[1].max[axis]-obj[1].min[axis] ) *0.5)

    # Split objects into two halves

    mid_index = len(objects) // 2

    # print(objects[:mid_index], objects[mid_index:])

    return objects[:mid_index], objects[mid_index:]


@dataclasses.dataclass
class BVH:
    nodes: list[BVHNode] = dataclasses.field(default_factory=list)
    root_index: Optional[int] = None
    max_objects_in_leaf:int=1
    leafs:list[int]=None
    def get_root(self) -> BVHNode:
        return self.nodes[self.root_index]

    def resize(self, new_size):
        old_size=len(self.nodes)
        if new_size==old_size:
            return
        elif old_size>new_size:

            for i in range(new_size,old_size):
                self.nodes.pop(-1)
        else:
            for i in range(old_size,new_size):
                self.nodes.append(BVHNode())

    def _build_bvh_internal(self, objects: list[tuple[int, AABB]], current_index: int=0)->int:
        """Recursively build the BVH tree given a list of objects with bounding boxes"""

        if current_index >= len(self.nodes):
            self.resize(current_index+1)

        node = self.nodes[current_index]

        if len(objects) <= self.max_objects_in_leaf:
            # Leaf node

            node.bbox = objects[0][1]
            node.object = objects[0][0]
            self.leafs.append(current_index)
            return current_index

        # Recursively build internal nodes
        left_objs, right_objs = split_objects(objects)

        node.left = self._build_bvh_internal(left_objs,  current_index=left(current_index))

        node.right = self._build_bvh_internal(right_objs,  current_index=right(current_index))

        # Merge bounding boxes
        self.nodes[current_index].bbox = self.nodes[node.left].bbox.merge(self.nodes[node.right].bbox)

        # Create and return internal node

        return current_index
    def separable(self, i, exact:bool=True):
        node=self.nodes[i]
        if node.is_leaf():
            return False
        if exact:
            return not self.nodes[node.left].bbox.intersects_exact( self.nodes[node.right].bbox)
        else:
            return not self.nodes[node.left].bbox.intersects(self.nodes[node.right].bbox)

    def build(self, bboxes: list[AABB]):
        self.leafs=[]
        self.nodes = [BVHNode() for _ in range((2 * len(bboxes) - 1))]

        self.root_index = self._build_bvh_internal([(i, bbox) for i, bbox in enumerate(bboxes)], current_index=0)
        return self

    def find_intersecting_leaves(self, exact:bool=True) -> dict[int, list[int]]:
        """
        Returns a dictionary mapping each leaf node index to a list of other leaf node indices
        whose bounding boxes intersect with it (excluding itself).
        """
        overlaps: dict[int, list[int]] = {}

        def recurse(i: int, j: int):
            # avoid duplicate and self pairs
            if i > j:
                return
            node_i = self.nodes[i]
            node_j = self.nodes[j]
            # prune if bboxes do not intersect
            if exact:
                if not node_i.bbox.intersects_exact(node_j.bbox):
                    return
            else:
                if not node_i.bbox.intersects(node_j.bbox):
                    return
            # if both are leaves, record overlap
            if node_i.is_leaf() and node_j.is_leaf():
                if i != j:
                    overlaps.setdefault(i, []).append(j)
                    overlaps.setdefault(j, []).append(i)
                return
            # recurse on children combinations
            if not node_i.is_leaf() and not node_j.is_leaf():
                recurse(node_i.left, node_j.left)
                recurse(node_i.left, node_j.right)
                recurse(node_i.right, node_j.left)
                recurse(node_i.right, node_j.right)
            elif not node_i.is_leaf():
                recurse(node_i.left, j)
                recurse(node_i.right, j)
            else:
                recurse(i, node_j.left)
                recurse(i, node_j.right)

        recurse(self.root_index, self.root_index)
        return overlaps

    def find_intersecting_leaves2(self, exact:bool=True) -> dict[int, list[int]]:
        """
        Returns a dictionary mapping each leaf node index to a list of other leaf node indices
        whose bounding boxes intersect with it (excluding itself).
        """
        dct=dict()
        for leaf_node_ix in self.leafs:
            res=_inter_bvh_node(self, leaf_node_ix,exact=exact)
            if len(res)>0:
                dct[leaf_node_ix]=res

        return dct

def build_bvh(bboxes)->BVH:
    tree = BVH()
    tree.build(bboxes)
    return tree


import heapq

from typing import Callable

import numpy as np

def point_segment_distance2(P: np.ndarray, A: np.ndarray, B: np.ndarray) -> float:
    """
    Distance from point P to segment AB.
    """
    AB = B - A
    t = np.dot(P - A, AB) / np.dot(AB, AB)
    t = np.clip(t, 0.0, 1.0)
    proj = A + t * AB
    d=P - proj
    return np.dot(d,d)


def point_triangle_distance2(P: np.ndarray, A: np.ndarray, B: np.ndarray, C: np.ndarray) -> float:
    """
    Distance from point P to triangle ABC.
    """
    # Triangle plane
    AB = B - A
    AC = C - A
    normal = np.cross(AB, AC)
    norm_len = np.linalg.norm(normal)

    # Degenerate triangle -> fallback to segment distances
    if norm_len < 1e-8:
        return min(point_segment_distance2(P, A, B), point_segment_distance2(P, B, C), point_segment_distance2(P, C, A))

    n_unit = normal / norm_len
    # signed distance to plane
    dist_plane = np.dot(P - A, n_unit)
    # projection of P onto plane
    P_proj = P - dist_plane * n_unit

    # Barycentric test for P_proj in triangle
    v0, v1, v2 = C - A, B - A, P_proj - A
    dot00 = np.dot(v0, v0)
    dot01 = np.dot(v0, v1)
    dot02 = np.dot(v0, v2)
    dot11 = np.dot(v1, v1)
    dot12 = np.dot(v1, v2)
    denom = dot00 * dot11 - dot01 * dot01

    if abs(denom) > 1e-8:
        u = (dot11 * dot02 - dot01 * dot12) / denom
        v = (dot00 * dot12 - dot01 * dot02) / denom
        if (u >= 0) and (v >= 0) and (u + v <= 1):
            return dist_plane*dist_plane

    # Otherwise, closest to one of the edges
    return min(point_segment_distance2(P, A, B), point_segment_distance2(P, B, C), point_segment_distance2(P, C, A))

def aabb_point_dist2(p: np.ndarray, node: BVHNode) -> float:
    """As before: squared distance from p to node.bbox."""
    d2 = 0.0
    for i in range(3):
        if p[i] < node.bbox.min[i]:
            d = node.bbox.min[i] - p[i]
            d2 += d*d
        elif p[i] > node.bbox.max[i]:
            d = p[i] - node.bbox.max[i]
            d2 += d*d
    return d2
from typing import Any
def bvh_nearest_point(
        tree: BVH,
        primitives:list|NDArray,
    point: np.ndarray,
    node_to_point_dist2: Callable[[np.ndarray, BVHNode], float],
    primitive_to_point_dist2: Callable[[np.ndarray, Any], float],
) -> tuple[float, int]:
    """
    Best-first search for the leaf in `tree` whose primitive is closest to `point`.

    Arguments
    ---------
    tree : BVH
        Your built BVH, whose leaves have `.object` = primitive ID.
    point : np.ndarray
        The 3D query point.
    node_to_point_dist2 :
        Lower-bound squared distance from `point` to `node.bbox` (or whatever volume).
    primitive_to_point_dist2 :
        Exact squared distance from `point` to the primitive with ID `leaf.object`.

    Returns
    -------
    (distance, leaf_node_index)


    Example of usage:
    -------

    """
    best_d2 = float("inf")
    best_node = -1

    # Min-heap of (lower_bound_dist2, node_index)
    pq: list[tuple[float, int]] = []
    root = tree.root_index
    heapq.heappush(pq, (node_to_point_dist2(point, tree.nodes[root]), root))

    while pq:
        lb2, ni = heapq.heappop(pq)
        # If this bound can't beat the best exact, we're done
        if lb2 >= best_d2:
            break

        node = tree.nodes[ni]
        if node.is_leaf():
            # exact test
            exact2 = primitive_to_point_dist2(point, primitives[node.object])
            if exact2 < best_d2:
                best_d2, best_node = exact2, ni
            continue

        # otherwise push children whose bbox‐bound < current best
        for c in (node.left, node.right):
            child = tree.nodes[c]
            child_lb2 = node_to_point_dist2(point, child)
            if child_lb2 < best_d2:
                heapq.heappush(pq, (child_lb2, c))

    return np.sqrt(best_d2), best_node
def _triangle_to_point_dist2(p,t): return point_triangle_distance2(p, t[0],t[1],t[2])

def triangle_soup_point_distance(point:NDArray[float], tris:NDArray[float],  bvh:BVH=None)->tuple[tuple[float,int,BVHNode], BVH]:
    boxes=[None]*tris.shape[0]
    for i in range(tris.shape[0]):

        boxes[i]=AABB.from_points(tris[i])

    if bvh is None:
        bvh:BVH=build_bvh(boxes)

    dst,node_index=bvh_nearest_point(bvh, tris, point, node_to_point_dist2=aabb_point_dist2, primitive_to_point_dist2=_triangle_to_point_dist2)
    node=bvh.nodes[node_index]
    return (dst, node.object, node), bvh


def mesh_point_distance(point:NDArray[float], vertices:NDArray[float],   faces:NDArray[int], bvh:BVH=None)->tuple[tuple[float,int,BVHNode], BVH]:
    """

    :param point:
    :param vertices:
    :param faces:
    :param bvh:
    :return:
    """
    return triangle_soup_point_distance(point,vertices[faces], bvh=bvh)

def _inter_bvh_node(bvh: BVH, node_ix: int,exact:bool=True):

    stack = [bvh.root_index]
    ints = []
    self=bvh.nodes[ node_ix]
    fun = self.bbox.intersects_exact if exact else self.bbox.intersects
    while stack:
        current_ix = stack.pop()
        node = bvh.nodes[current_ix]
        if current_ix==node_ix:
            continue

        if fun(node.bbox):
            if node.is_leaf():
                ints.append(current_ix)
            else:
                if node.left != -1:

                    stack.append(node.left)
                if node.right != -1:

                    stack.append(node.right)
    return ints

def inter_bvh(bvh: BVH, bbox: AABB,exact:bool=True):
    stack = [bvh.root_index]
    ints = []
    fun = AABB.intersects_exact if exact else AABB.intersects
    while stack:
        current_ix = stack.pop()
        node = bvh.nodes[current_ix]
        if fun(bbox,node.bbox):
            if node.is_leaf():
                ints.append(current_ix)
            else:
                if node.left != -1:

                    stack.append(node.left)
                if node.right != -1:

                    stack.append(node.right)
    return ints
def bvh_intersect(bvh1:BVH,bvh2:BVH,exact:bool=True):
    root1:BVHNode=bvh1.get_root()
    root2:BVHNode=bvh2.get_root()
    stack=[(root1, root2)]
    res=[]
    while stack:
        a,b=stack.pop(0)
        if not exact:
            is_inter=a.bbox.intersects(b.bbox)
        else:
            is_inter = a.bbox.intersects_exact(b.bbox)
        if not is_inter:
            continue
        elif a.is_leaf() and b.is_leaf():
            res.append((a,b))
        elif a.is_leaf() :

            stack.append((a,bvh2.nodes[b.left]))
            stack.append((a,bvh2.nodes[b.right]))
        elif b.is_leaf():
            stack.append(( bvh1.nodes[a.left],b))
            stack.append(( bvh1.nodes[a.right],b))
        else:
            for first in [bvh1.nodes[a.left],bvh1.nodes[a.right]]:
                for second in [ bvh2.nodes[b.left], bvh2.nodes[b.right]] :
                    stack.append((first, second))
    return res


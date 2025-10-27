from __future__ import annotations

import bisect
import os
from dataclasses import dataclass
from pathlib import Path

from mmcore.geom.bvh.lbvh import AABB
from mmcore.geom.implicit import Implicit3D
import itertools
import time
from collections import defaultdict
from collections.abc import Callable

from typing import Literal, List

from typing import NamedTuple, Union, Sequence, Optional, Tuple

import math
import numpy as np

class NodeIndex( NamedTuple):
    leval: int
    x:int
    y:int
    z:int





import numpy as np

def points_to_octants(aabb: np.ndarray, pts: np.ndarray) -> np.ndarray:
    """
    Vectorized octant classification for a 3D AABB.

    Parameters
    ----------
    aabb : (2, 3) array_like
        Axis-aligned bounding box as [[xmin, ymin, zmin], [xmax, ymax, zmax]].
    pts : (M, 3) array_like
        Array of 3D points.

    Returns
    -------
    idx : (M,) np.ndarray of dtype int64
        Octant index in [0..7] for points inside the AABB, or -1 for points outside.
        Octant encoding (bit-coded):
            bit 0 (1): x >= xmid  -> high-x
            bit 1 (2): y >= ymid  -> high-y
            bit 2 (4): z >= zmid  -> high-z
        So:
            0: (low-x, low-y, low-z)
            1: (high-x, low-y, low-z)
            2: (low-x, high-y, low-z)
            3: (high-x, high-y, low-z)
            4: (low-x, low-y, high-z)
            5: (high-x, low-y, high-z)
            6: (low-x, high-y, high-z)
            7: (high-x, high-y, high-z)

    Notes
    -----
    - Points exactly on the global AABB boundary are considered *inside*.
    - Split planes are assigned to the *upper* child along that axis (i.e., p == mid goes to the “high” side).
    - Handles degenerate boxes (min == max) gracefully: only points equal to that coordinate are inside;
      they map to the “high” side (bit = 1) for that axis.
    """

    # Ensure arrays and proper ordering of bounds
    aabb = np.asarray(aabb, dtype=np.float64)
    pts  = np.asarray(pts,  dtype=np.float64)
    lo = np.minimum(aabb[0], aabb[1])
    hi = np.maximum(aabb[0], aabb[1])

    # Midpoint split planes
    mid = (lo + hi) * 0.5

    # Inside-parent mask (inclusive bounds)
    inside_mask = (pts >= lo).all(axis=1) & (pts <= hi).all(axis=1)

    # Octant bits: high side if point >= mid along that axis
    bits = (pts >= mid).astype(np.int64)

    # Encode to [0..7]: x is LSB, then y, then z
    idx_all = (bits[:, 0]      # 1*x_high
               | (bits[:, 1] << 1)  # 2*y_high
               | (bits[:, 2] << 2)) # 4*z_high

    # Fill result and mark outside as -1
    out = np.full(pts.shape[0], -1, dtype=np.int64)
    out[inside_mask] = idx_all[inside_mask]
    return out

class Octree:
    def __init__(self, aabb: np.ndarray, max_depth: int = 16):
        self.aabb = aabb
        self.d=self.aabb[1]-self.aabb[0]
        self.half=self.d/2
        self.center=self.half+self.aabb[0]
        self.half[:]=self.half.max()
        self.aabb[1,:]=self.center+self.half
        self.aabb[0, :] = self.center - self.half
        self.d[:]=self.aabb[1]-self.aabb[0]
        self.min_half = None
        self.max_depth = max_depth
        self._index = dict()
        self._centroid_cache = dict()
        self._aabb_cache = dict()
        self.shape = self.sparse4d_shape()
        
    def set_max_depth_by_min_half(self, min_half):
        self.max_depth = self.required_depth(min_half)
        self.min_half = min_half
        self.shape=self.sparse4d_shape()
        
        return self.max_depth
    
    def required_depth(self, min_half: Union[float, Sequence[float]]) -> int:
        """As before: smallest L with span/(2^(L+1)) <= min_half per axis."""
        aabb = np.asarray(self.aabb, dtype=float)
        if aabb.shape != (2, 3):
            raise ValueError("root_aabb must have shape (2, 3).")
        spans = aabb[1] - aabb[0]
        
        if np.any(spans <= 0):
            raise ValueError("root_aabb must have strictly positive extent on all axes.")
        mh = np.asarray(min_half, dtype=float)
        if mh.ndim == 0: mh = np.array([mh, mh, mh], float)
        if mh.shape != (3,): raise ValueError("min_half must be a scalar or length-3.")
        if np.any(mh <= 0): raise ValueError("min_half values must be > 0.")
        
        ratios = spans / (2.0 * mh)
        r_max = float(np.max(ratios))
        return 0 if r_max <= 1.0 else int(math.ceil(math.log2(r_max)))
    
    def ravel_3dindex(self, i: int|np.ndarray[int], j: int|np.ndarray[int], k: int|np.ndarray[int]) -> int|np.ndarray[int]:
        return np.ravel_multi_index((i,j,k),self.shape[1:])
    def sparse4d_shape(self,
                              nodes: Optional[np.ndarray] = None,
                  
                              ) -> Tuple[int, int, int, int]:
        """
        Compute the shape (Ldim, Xdim, Ydim, Zdim) for a sparse 4D tensor such that
        a node (L, x, y, z) maps directly to tensor[L, x, y, z].

        Provide exactly one of:
          - max_level: use it directly
          - nodes: int[N,4] -> uses nodes[:,0].max()
          - root_aabb and min_half: computes required depth via 'required_depth'
        """
        
        # Determine L_max
        if self.max_depth is not None:
            Lmax = int(self.max_depth)
        elif nodes is not None:
            nodes = np.asarray(nodes)
            if nodes.ndim != 2 or nodes.shape[1] != 4:
                raise ValueError("nodes must have shape (N, 4) with columns [level, x, y, z].")
            if nodes.size == 0:
                raise ValueError("nodes is empty; provide max_level or (root_aabb, min_half).")
            Lmax = int(np.max(nodes[:, 0]))
        elif self.aabb is not None and self.min_half is not None:
            Lmax = self.required_depth(self.aabb, self.min_half)
        else:
            raise ValueError("Provide one of: max_level, nodes, or (root_aabb and min_half).")
        
        if Lmax < 0:
            raise ValueError("max_level must be >= 0.")
        
        side = 1 << Lmax  # 2^Lmax
        return (Lmax + 1, side, side, side)
    
    def get_root(self):
        return (0, 0, 0, 0)
    
    def r_box(self, half):
       
      
        return np.sqrt(np.sum(half*half))
    
    def get_parent(self, node: NodeIndex):
        """
        2) Return the parent node index of `child`.
           If `child` is the root (0,0,0,0), return None.

        Rule: parent(L, x, y, z) = (L-1, x//2, y//2, z//2) for L > 0.
        """
        
        L, x, y, z = node
        if L == 0:
            # Root has no parent
            return None
        return (L - 1, x >> 1, y >> 1, z >> 1)
    
    def get_center(self, node: NodeIndex) -> np.ndarray:
        if node in self._centroid_cache.keys():
            return self._centroid_cache[node]
        c = self._centroid_cache[node] = np.mean(self.get_bbox(node), axis=0)
        return c
    
    def get_bbox(self, node: NodeIndex) -> np.ndarray:
        """
        3) Compute the axis-aligned bounding box (AABB) of `node`
           within the given `root_aabb`.

        Parameters
        ----------
        node : (level, x, y, z)
            Implicit octree index. At level L, x,y,z are in [0, 2^L - 1].
        root_aabb : np.ndarray, shape (2, 3)
            [[min_x, min_y, min_z],
             [max_x, max_y, max_z]]

        Returns
        -------
        np.ndarray, shape (2, 3)
            The node's AABB as [[min_x, min_y, min_z], [max_x, max_y, max_z]].
        """
        if node in self._aabb_cache.keys():
            return self._aabb_cache[node]
        L, x, y, z = node
        
        aabb =self.aabb
        if aabb.shape != (2, 3):
            raise ValueError("root_aabb must have shape (2, 3).")
        mins, maxs = aabb[0], aabb[1]
        spans = maxs - mins
        if np.any(spans <= 0):
            raise ValueError("root_aabb must have strictly positive extent in all axes.")
        
        # Each level doubles resolution per axis: grid size is 2^L per axis
        cells_per_axis = float(1 << L)
        cell_size = spans / cells_per_axis  # vector of size per cell along each axis
        
        base = np.array([x, y, z], dtype=float)
        node_mins = mins + base * cell_size
        node_maxs = node_mins + cell_size
        
        bb = self._aabb_cache[node] = np.stack((node_mins, node_maxs), axis=0)
        return bb
    
    def morton3d_encode(self, node: NodeIndex):
        # Interleave L low bits of x,y,z into a single integer
        L, x, y, z = node
        
        def part(v):
            v &= (1 << L) - 1
            # “Dilate” bits: insert two zeros between each bit of v
            v = (v | (v << 32)) & 0x1f00000000ffff
            v = (v | (v << 16)) & 0x1f0000ff0000ff
            v = (v | (v << 8)) & 0x100f00f00f00f00f
            v = (v | (v << 4)) & 0x10c30c30c30c30c3
            v = (v | (v << 2)) & 0x1249249249249249
            return v
        
        return (part(x) << 0) | (part(y) << 1) | (part(z) << 2)
    
    def morton3d_decode(self, code, depth):
        def compact(v):
            v &= 0x1249249249249249
            v = (v ^ (v >> 2)) & 0x10c30c30c30c30c3
            v = (v ^ (v >> 4)) & 0x100f00f00f00f00f
            v = (v ^ (v >> 8)) & 0x1f0000ff0000ff
            v = (v ^ (v >> 16)) & 0x1f00000000ffff
            v = (v ^ (v >> 32)) & ((1 << depth) - 1)
            return v
        
        return NodeIndex(depth, compact(code >> 0), compact(code >> 1), compact(code >> 2))
    
    def iter_children(self, node: NodeIndex) -> np.ndarray:
        """
        Return the 8 child node indexes of `node`.

        Ordering: child_id = x_bit + 2*y_bit + 4*z_bit, with bits in {0,1}.
        i.e., (x_bit, y_bit, z_bit) ∈ {(0/1),(0/1),(0/1)}.
        """
        
        L, x, y, z = node
        Lp = L + 1
        xb, yb, zb = x << 1, y << 1, z << 1
        
        # child_id 0..7 encodes (x_bit, y_bit, z_bit) as (i&1, (i>>1)&1, (i>>2)&1)
        for i in range(8):
                yield    (Lp,
             xb | (i & 1),
             yb | ((i >> 1) & 1),
             zb | ((i >> 2) & 1))
       
      
    def get_children(self, node: NodeIndex) -> np.ndarray:
        """
        Return the 8 child node indexes of `node`.

        Ordering: child_id = x_bit + 2*y_bit + 4*z_bit, with bits in {0,1}.
        i.e., (x_bit, y_bit, z_bit) ∈ {(0/1),(0/1),(0/1)}.
        """
        
        L, x, y, z = node
        Lp = L + 1
        xb, yb, zb = x << 1, y << 1, z << 1
        
        # child_id 0..7 encodes (x_bit, y_bit, z_bit) as (i&1, (i>>1)&1, (i>>2)&1)
        return np.array([
            (Lp,
             xb | (i & 1),
             yb | ((i >> 1) & 1),
             zb | ((i >> 2) & 1))
            for i in range(8)
        ],dtype=np.int64)
    
    def get_children_multiple(self, nodes: np.ndarray) -> np.ndarray:
        """
        Return the 8 child node indexes of `node`.

        Ordering: child_id = x_bit + 2*y_bit + 4*z_bit, with bits in {0,1}.
        i.e., (x_bit, y_bit, z_bit) ∈ {(0/1),(0/1),(0/1)}.
        """
    
        L, x, y, z = nodes[...,0], nodes[...,1], nodes[...,2], nodes[...,3]
        Lp = L + 1
        
        xb, yb, zb = x << 1, y << 1, z << 1
        chlds=np.zeros((8,*nodes.shape),dtype=np.int64)
        for i in range(8):
            chlds[i,:, 0]=Lp
            chlds[i,:, 1]=(xb | (i & 1))
            chlds[i,:, 2]=(yb | ((i >> 1) & 1))
            chlds[i,:, 3]=( zb | ((i >> 2) & 1))
           
        # child_id 0..7 encodes (x_bit, y_bit, z_bit) as (i&1, (i>>1)&1, (i>>2)&1)
        return chlds
    def get_bboxes(self, nodes: np.ndarray, dtype=float) -> np.ndarray:
        """
        Vectorized AABB computation for octree nodes.

        Parameters
        ----------
        nodes : np.ndarray, shape (N, 4), dtype int
            Each row is [level, x, y, z].
        root_aabb : np.ndarray, shape (2, 3), dtype float
            [[min_x, min_y, min_z],
             [max_x, max_y, max_z]]
        dtype : target floating dtype for output (default: np.float64)

        Returns
        -------
        np.ndarray, shape (N, 2, 3), dtype=dtype
            Per-node AABBs: [ [min_x, min_y, min_z], [max_x, max_y, max_z] ].
        """
  
        if nodes.ndim != 2 or nodes.shape[1] != 4:
            raise ValueError(f"nodes must have shape (N, 4) with columns [level, x, y, z]. {nodes.shape}")
        
        aabb = np.asarray(self.aabb, dtype=dtype)
        if aabb.shape != (2, 3):
            raise ValueError("root_aabb must have shape (2, 3).")
        
        N = nodes.shape[0]
        if N == 0:
            return np.empty((0, 2, 3), dtype=dtype)
        
        # Extract columns (vectorized)
        L = nodes[:, 0].astype(np.int64)  # (N,)
        xyz = nodes[:, 1:4].astype(dtype)  # (N,3)
        
        mins = aabb[0]  # (3,)
        spans = aabb[1] - aabb[0]  # (3,)
        if np.any(spans <= 0):
            raise ValueError("root_aabb must have strictly positive extents along all axes.")
        
        # Per-node cell sizes: spans / 2^L  (use ldexp for exact power-of-two scaling)
        # Broadcasts (3,) with (N,1) -> (N,3)
        cell = np.ldexp(spans, -L[:, None]).astype(dtype)  # (N,3)
        
        node_mins = mins + xyz * cell  # (N,3)
        node_maxs = node_mins + cell  # (N,3)
        
        
        
        # Stack into (N,2,3)
        return np.stack((node_mins, node_maxs), axis=1)
    
    def get_centroids(self, nodes: np.ndarray,
                      
                      dtype=float) -> np.ndarray:
        """
        Vectorized centroid computation for octree nodes.

        Parameters
        ----------
        nodes : np.ndarray, shape (N, 4), dtype int
            Each row is [level, x, y, z].
        root_aabb : np.ndarray, shape (2, 3)
            [[min_x, min_y, min_z],
             [max_x, max_y, max_z]]
        dtype : output floating dtype (default: np.float64)

        Returns
        -------
        np.ndarray, shape (N, 3), dtype=dtype
            The centroid of each node.
        """
        root_aabb: np.ndarray = self.aabb
    
        if nodes.ndim != 2 or nodes.shape[1] != 4:
            raise ValueError("nodes must have shape (N, 4) with columns [level, x, y, z].")
        
        aabb = np.asarray(root_aabb, dtype=dtype)
        if aabb.shape != (2, 3):
            raise ValueError("root_aabb must have shape (2, 3).")
        
        N = nodes.shape[0]
        if N == 0:
            return np.empty((0, 3), dtype=dtype)
        
        # Extract columns (vectorized)
        L = nodes[:, 0].astype(np.int64)  # levels (N,)
        xyz = nodes[:, 1:4].astype(dtype)  # indices (N,3)
        
        mins = aabb[0]  # (3,)
        spans = aabb[1] - aabb[0]  # (3,)
        if np.any(spans <= 0):
            raise ValueError("root_aabb must have strictly positive extents on all axes.")
        if np.any(L < 0):
            raise ValueError("levels must be >= 0.")
        
        # Cell size per node per axis: spans / 2^L (using ldexp for exact power-of-two scaling)
        cell = np.ldexp(spans, -L[:, None]).astype(dtype)  # (N,3)
        
        # Centroid = mins + (xyz + 0.5) * cell
        centroids = mins + (xyz + 0.5) * cell  # (N,3)
        self._aabb_cache[np.array(nodes, dtype=np.int64)] = centroids
        return self._aabb_cache[np.array(nodes, dtype=np.int64)]

    
    def find_nodes(self, points: np.ndarray, node=None,min_points_per_node: int=10000) -> np.ndarray:
        points_ixs=np.arange(points.shape[0], dtype=np.int64)
        stack=[(self.get_root() , points_ixs)if node is None else (node,points_ixs)]
      
        inside_points=np.zeros(points.shape[0], dtype=bool)
        leaf_to_points=dict()
        
        while stack:
            node,pti = stack.pop(0)
           
            
            if pti.size==0:
      
                continue
            node = int(node[0]),int(node[1]),int(node[2]),int(node[3])
            if pti.shape[0] < min_points_per_node:
                inside_points[pti] = True
                leaf_to_points[node] = pti
                continue
         
            
            if node[0]>=self.max_depth:
                inside_points[pti]=True
                leaf_to_points[node]=pti
                
                continue
          
            bb=self.get_bbox(node)
     
            pt_to_child=points_to_octants(bb, np.atleast_2d(points[pti]))
            #pti = np.atleast_1d(pti)
            unq=np.unique(pt_to_child)
            buckets=[pti[i] for i in pt_to_child[None, ...] == unq.reshape(-1, 1)]
            
            childs =self.get_children(node)
            
            for buck, chi in zip(buckets,unq):
                if chi<0:
                    pass
                else:
                
                    stack.append((childs[chi],buck))
            
         
            
         
        return points_ixs[inside_points],leaf_to_points


def _dilate3_u64(v: np.ndarray, D: int) -> np.ndarray:
    """
    'Dilate' lower D bits so that there are 2 zeros between original bits:
    0000000000000000 b_{D-1} ... b1 b0 -> 00b_{D-1}00 ... 00b1 00b0
    Vectorized uint64 implementation.
    """
    v = v.astype(np.uint64) & (np.uint64(1) << np.uint64(D)) - np.uint64(1)
    v = (v | (v << np.uint64(32))) & np.uint64(0x1f00000000ffff)
    v = (v | (v << np.uint64(16))) & np.uint64(0x1f0000ff0000ff)
    v = (v | (v << np.uint64(8))) & np.uint64(0x100f00f00f00f00f)
    v = (v | (v << np.uint64(4))) & np.uint64(0x10c30c30c30c30c3)
    v = (v | (v << np.uint64(2))) & np.uint64(0x1249249249249249)
    return v


def morton3d_codes_for_points(points: np.ndarray, octree: 'Octree', D: int) -> np.ndarray:
    """
    Quantize points to a D-bit grid per axis inside *octree.aabb* (already normalized to a cube),
    then compute 64-bit Morton codes. Vectorized.

    Parameters
    ----------
    points : (N,3) float64
    octree : Octree
    D : int (<= 21 for 64-bit safety; can go higher with Python bigints if needed)

    Returns
    -------
    codes : (N,) uint64
    """
    pts = np.asarray(points, dtype=np.float64)
    aabb = np.asarray(octree.aabb, dtype=np.float64)
    lo, hi = aabb[0], aabb[1]
    span = hi - lo
    # Guard against degenerate spans; Octree constructor already enforces cube > 0
    grid_max = (np.uint64(1) << np.uint64(D)) - np.uint64(1)
    
    # Normalize to [0,1], clip, and scale to [0, 2^D - 1]
    f = (pts - lo) / span
    f = np.clip(f, 0.0, 1.0)
    q = np.floor(f * float(int(grid_max))).astype(np.uint64)  # (N,3)
    
    qx, qy, qz = q[:, 0], q[:, 1], q[:, 2]
    dx = _dilate3_u64(qx, D)
    dy = _dilate3_u64(qy, D)
    dz = _dilate3_u64(qz, D)
    codes = (dx << np.uint64(0)) | (dy << np.uint64(1)) | (dz << np.uint64(2))
    return codes


# ---------- Canonical prefix interval for any node (L, prefix) inside depth D ----------

@np.vectorize
def node_interval(prefix: np.uint64, L: np.uint8, D: int) -> Tuple[np.uint64, np.uint64]:
    """
    Compute canonical code interval [lo,hi] (inclusive) at fixed depth D for a node at level L with prefix.
    """
    shift = np.uint64(3 * (D - int(L)))
    lo = prefix << shift
    hi = ((prefix + np.uint64(1)) << shift) - np.uint64(1)
    return lo, hi


# ---------- Leaf table structure (minimal, pointer-less) ----------

@dataclass
class LeafTable:
    # All arrays are sorted by code_lo and represent disjoint, covering intervals.
    L: np.ndarray  # (M,) uint8
    prefix: np.ndarray  # (M,) uint64 (3*L bits meaningful)
    code_lo: np.ndarray  # (M,) uint64
    code_hi: np.ndarray  # (M,) uint64
    idx_lo: np.ndarray  # (M,) uint64  [start,end) indices into points_sorted
    idx_hi: np.ndarray  # (M,) uint64
    
    def __len__(self) -> int:
        return int(self.L.shape[0])


# ---------- Main index wrapper ----------
import tempfile
import shutil

_PERM_SORTED_PATH="perm_sorted.npy"
_CODES_SORTED_PATH="codes_sorted.npy"
_POINTS_SORTED_PATH="points_sorted.f64"
_LEAVES_PATH="leaves.npz"
_META_PATH="meta.npz"
class LinearOctreeIndex:
    """
    Linear implicit octree built over a fixed-depth Morton code space.
    - Stores points reordered by code (memmap-friendly).
    - Stores only leaf intervals (no internal nodes).
    - Can traverse top-down using interval existence/coverage tests.
    """
 
    
    
    def __init__(self,
                 root_aabb: np.ndarray,
                 D: Optional[int] = None,
                 min_points_per_leaf: int = 10_000,
                 min_half: Optional[float] = None,
                 workdir: Optional[str] = None,
                 dtype_points=np.float64, no_workdir: bool = False, ):
        """
        If D is None and min_half is provided, D is chosen s.t. half-size at depth D <= min_half.
        Otherwise D must be provided.
        """
        self._oct = Octree(np.array(root_aabb, dtype=np.float64))
        self.dtype_points = dtype_points
        self.min_points_per_leaf = int(min_points_per_leaf)
        
        if D is None:
            if min_half is None:
                raise ValueError("Provide either D or min_half to set the index depth.")
            # Use your Octree rule to derive a reasonable depth ceiling, then cap by 21 for u64 Morton
            D_req = self._oct.required_depth(np.array([min_half, min_half, min_half], dtype=float))
            self.D = int(min(D_req, 21))  # 3*21=63 bits fits in u64
        else:
            self.D = int(D)
        
        # If min_half is given we also keep a traversal max L to avoid refining infinitely
        self.traversal_L_max = self._oct.required_depth(
            np.array([min_half, min_half, min_half], dtype=float)) if min_half is not None else self.D
        
        # Files / memory
        # self.workdir = workdir
        # if workdir is not None:
        #    os.makedirs(workdir, exist_ok=True)
        if workdir is None and not no_workdir:
            
            with tempfile.TemporaryDirectory(delete=False) as tmpdirname:
                print('created temporary directory', tmpdirname)
                
                self.workdir = tmpdirname
                self.delete_workdir = True
        elif no_workdir:
            self.workdir = None
            self.delete_workdir = False
        else:
            self.workdir = workdir
            Path(workdir).mkdir(parents=True, exist_ok=True)
            self.delete_workdir = False
        
        self.points_sorted = None  # memmap or ndarray, shape (N,3)
        self.perm_sorted = None  # ndarray or memmap, shape (N,), maps sorted->original
        self.codes_sorted = None  # ndarray or memmap, shape (N,), uint64; not needed after building
        self.leaves: Optional[LeafTable] = None
        self.N = 0
    
    # ---------- BUILD PIPELINE ----------
    
    def build(self, points: np.ndarray, external_sort: bool = False, chunk_size: int = 10_000_000) -> None:
        """
        Build the linear index from a point cloud. By default uses in-memory sort.
        Optionally, a skeleton for an out-of-core sorter is included for very large data.
        """
        t0 = time.perf_counter()
        
        pts = np.asarray(points, dtype=self.dtype_points)
        
        N = pts.shape[0]
        self.N = N
        
        # 1) Morton codes at depth D (vectorized)
        codes = morton3d_codes_for_points(pts, self._oct, self.D)
        
        # 2) Sort by code (in memory). For out-of-core, use the external sort skeleton below.
        if external_sort and self.workdir is not None and N > chunk_size:
            perm = self._external_sort_codes(codes, chunk_size=chunk_size)
            codes_sorted = codes[perm]  # This is only to validate; you can drop storing codes altogether later
        else:
            perm = np.argsort(codes, kind='stable')  # stable keeps a bit more spatial locality when codes tie
            codes_sorted = codes[perm]
        
        # 3) Reorder points according to 'perm' into a memmap file (chunked, to avoid huge temporaries)
        if self.workdir is None:
            pts_sorted = pts[perm]  # in memory
            self.points_sorted = pts_sorted
            self.perm_sorted = perm.astype(np.uint64, copy=False)
            self.codes_sorted = codes_sorted.astype(np.uint64, copy=False)
        else:
            # Write sorted permutation and codes to disk
            self.perm_sorted = np.asarray(perm, dtype=np.uint64)
            np.save(os.path.join(self.workdir, _PERM_SORTED_PATH), self.perm_sorted)
            self.codes_sorted = codes_sorted.astype(np.uint64, copy=False)
            np.save(os.path.join(self.workdir, _CODES_SORTED_PATH), self.codes_sorted)
            
            # Create a memmap file for points_sorted and fill it chunked
            pts_mm_path = os.path.join(self.workdir, _POINTS_SORTED_PATH)
            pts_mm = np.memmap(pts_mm_path, dtype=self.dtype_points, mode='w+', shape=(N, 3))
            stride = max(1, min(N, chunk_size))
            for s in range(0, N, stride):
                e = min(N, s + stride)
                pts_mm[s:e, :] = pts[self.perm_sorted[s:e], :]
            pts_mm.flush()
            self.points_sorted = np.memmap(pts_mm_path, dtype=self.dtype_points, mode='r+', shape=(N, 3))
        
        # 4) Build leaf interval table (vectorized BFS over intervals)
        leaves = self._build_leaves_bfs(codes_sorted)
        self.leaves = leaves
        
        # 5) Persist leaf table if in workdir
        if self.workdir is not None:
            self._save_leaves(leaves)
        
        t1 = time.perf_counter()
        print(f"[LIO] Build complete: N={N:,} points, depth D={self.D}, leaves={len(leaves):,} in {t1 - t0:.3f}s")
    
    def _save_leaves(self, leaves: LeafTable):
        np.savez_compressed(os.path.join(self.workdir, _LEAVES_PATH),
                            L=leaves.L, prefix=leaves.prefix,
                            code_lo=leaves.code_lo, code_hi=leaves.code_hi,
                            idx_lo=leaves.idx_lo, idx_hi=leaves.idx_hi)
    
    @staticmethod
    def load(workdir: str) -> 'LinearOctreeIndex':
        """
        Load an index previously built with .build(..., workdir=...).
        """
        # We need root_aabb, D, traversal_L_max; store them in a small header file
        meta = np.load(os.path.join(workdir, _META_PATH))
        aabb = meta["aabb"]
        D = int(meta["D"])
        min_points_per_leaf = int(meta["min_points"])
        traversal_L_max = int(meta["traversal_L_max"])
        
        # Recreate the wrapper
        idx = LinearOctreeIndex(aabb, D=D, min_points_per_leaf=min_points_per_leaf, workdir=workdir)
        idx.traversal_L_max = traversal_L_max
        
        # Map points, codes, perm and leaves
        perm_sorted = np.load(os.path.join(workdir, _PERM_SORTED_PATH), mmap_mode='r')
        codes_sorted = np.load(os.path.join(workdir, _CODES_SORTED_PATH), mmap_mode='r')
        pts_mm = np.memmap(os.path.join(workdir, _POINTS_SORTED_PATH), dtype=np.float64, mode='r',
                           shape=(perm_sorted.shape[0], 3))
        leaves_npz = np.load(os.path.join(workdir, _LEAVES_PATH))
        
        idx.perm_sorted = perm_sorted
        idx.codes_sorted = codes_sorted
        idx.points_sorted = pts_mm
        idx.N = perm_sorted.shape[0]
        idx.leaves = LeafTable(
            L=leaves_npz["L"],
            prefix=leaves_npz["prefix"],
            code_lo=leaves_npz["code_lo"],
            code_hi=leaves_npz["code_hi"],
            idx_lo=leaves_npz["idx_lo"],
            idx_hi=leaves_npz["idx_hi"],
        )
        return idx
    
    def save_meta(self):
        if self.workdir is None:
            return
        np.savez(os.path.join(self.workdir, _META_PATH),
                 aabb=self._oct.aabb, D=np.array(self.D, np.int32),
                 min_points=np.array(self.min_points_per_leaf, np.int64),
                 traversal_L_max=np.array(self.traversal_L_max, np.int32))
    
    # ---------- Out-of-core sort skeleton (optional) ----------
    
    def _external_sort_codes(self, codes: np.ndarray, chunk_size: int = 10_000_000) -> np.ndarray:
        """
        Very simple multi-chunk sort + k-way merge for gigantic inputs.
        Produces a global permutation array 'perm' such that codes[perm] is sorted.
        Note: this is a skeleton; for production, prefer an on-disk radix sort or use
        a big data framework. Included here for completeness.
        """
        assert self.workdir is not None, "external_sort requires a workdir"
        N = codes.shape[0]
        chunk_files = []
        t0 = time.perf_counter()
        # 1) Chunk and sort locally
        for c, s in enumerate(range(0, N, chunk_size)):
            e = min(N, s + chunk_size)
            perm_c = np.arange(s, e, dtype=np.uint64)
            # argsort by codes segment
            order = np.argsort(codes[s:e], kind='stable')
            perm_c = perm_c[order]
            p_path = os.path.join(self.workdir, f"chunk_perm_{c:05d}.npy")
            np.save(p_path, perm_c)
            chunk_files.append(p_path)
        # 2) K-way merge the per-chunk permutations into a global 'perm'
        #    We do a streaming merge using a small heap keyed by codes[idx]
        import heapq
        # Open memmaps for each chunk perm
        per_chunk_perm = [np.load(p, mmap_mode='r') for p in chunk_files]
        # Pointers into each chunk
        ptrs = [0] * len(per_chunk_perm)
        # Initialize heap
        heap = []
        for ci, perm_c in enumerate(per_chunk_perm):
            if perm_c.shape[0] > 0:
                gidx = int(perm_c[0])  # global index into 'codes'
                heapq.heappush(heap, (int(codes[gidx]), ci, 0, gidx))
        # Output global perm
        perm = np.empty(N, dtype=np.uint64)
        k = 0
        while heap:
            code_val, ci, pi, gidx = heapq.heappop(heap)
            perm[k] = gidx
            k += 1
            pi += 1
            if pi < per_chunk_perm[ci].shape[0]:
                gnew = int(per_chunk_perm[ci][pi])
                heapq.heappush(heap, (int(codes[gnew]), ci, pi, gnew))
        t1 = time.perf_counter()
        print(f"[LIO] External sort skeleton produced permutation in {t1 - t0:.3f}s")
        # Cleanup chunk files if desired
        for p in chunk_files:
            try:
                os.remove(p)
            except Exception:
                pass
        return perm
    
    # ---------- Vectorized leaf construction (BFS over code intervals) ----------
    
    def _build_leaves_bfs(self, codes_sorted: np.ndarray) -> LeafTable:
        """
        Build leaves by splitting code intervals until min_points or spatial min_half is met.
        Uses vectorized searchsorted on arrays of child intervals for many nodes per iteration.
        """
        N = codes_sorted.shape[0]
        D = self.D
        # Max splitting depth due to spatial criterion (<= D)
        Lmax = min(self.traversal_L_max, D)
        
        # Root node covers whole array
        L_cur = np.array([0], dtype=np.uint8)
        prefix = np.array([np.uint64(0)], dtype=np.uint64)
        i0 = np.array([0], dtype=np.int64)
        i1 = np.array([N], dtype=np.int64)
        
        leaves_L, leaves_prefix, leaves_lo, leaves_hi, leaves_i0, leaves_i1 = [], [], [], [], [], []
        
        child_ids = (np.arange(8, dtype=np.uint64)).reshape(1, 8)  # broadcast
        
        while L_cur.size > 0:
            counts = (i1 - i0)
            stop_mask = (counts <= self.min_points_per_leaf) | (L_cur >= Lmax)
            
            # Emit leaves for stop_mask nodes
            if np.any(stop_mask):
                L_s = L_cur[stop_mask]
                p_s = prefix[stop_mask]
                i0_s = i0[stop_mask]
                i1_s = i1[stop_mask]
                # canonical intervals for these nodes
                los, his = node_interval(p_s.astype(np.uint64), L_s.astype(np.uint8), D)
                leaves_L.append(L_s.copy())
                leaves_prefix.append(p_s.copy())
                leaves_lo.append(los.astype(np.uint64))
                leaves_hi.append(his.astype(np.uint64))
                leaves_i0.append(i0_s.astype(np.uint64))
                leaves_i1.append(i1_s.astype(np.uint64))
            
            # Nodes to refine
            refine_mask = ~stop_mask
            if not np.any(refine_mask):
                break
            
            L_r = L_cur[refine_mask]
            p_r = prefix[refine_mask]
            r_i0 = i0[refine_mask]
            r_i1 = i1[refine_mask]
            
            # Compute 8 children per node (vectorized)
            L_child = (L_r + 1).astype(np.uint8)
            pbase = (p_r << np.uint64(3))  # shape (R,)
            # child prefixes shape (R,8)
            cp = (pbase[:, None] | child_ids)  # (R, 8) uint64
            # Canonical code intervals for all children: (R,8)
            clo, chi = node_interval(cp, L_child[:, None], D)
            clo = clo.astype(np.uint64);
            chi = chi.astype(np.uint64)
            
            # Global searchsorted on the full codes array (then clip to [i0,i1])
            j0 = np.searchsorted(codes_sorted, clo, side='left')  # (R,8)
            j1 = np.searchsorted(codes_sorted, chi, side='right')  # (R,8)
            
            # Clip to each parent's [i0,i1]
            r_i0c = r_i0[:, None]
            r_i1c = r_i1[:, None]
            j0 = np.maximum(r_i0c, np.minimum(j0, r_i1c))
            j1 = np.maximum(r_i0c, np.minimum(j1, r_i1c))
            
            # Keep only non-empty children
            present = j0 < j1
            if not np.any(present):
                # all empty (rare: if parent had no points due to previous culling)
                break
            
            # Flatten selected children into next-level arrays
            L_next = np.repeat(L_child, 8).reshape(-1)[present.reshape(-1)]
            p_next = cp.reshape(-1)[present.reshape(-1)]
            i0_next = j0.reshape(-1)[present.reshape(-1)]
            i1_next = j1.reshape(-1)[present.reshape(-1)]
            
            # Advance
            L_cur, prefix, i0, i1 = L_next, p_next, i0_next, i1_next
        
        # Concatenate leaves and sort by code_lo
        L_all = np.concatenate(leaves_L) if leaves_L else np.empty((0,), np.uint8)
        prefix_all = np.concatenate(leaves_prefix) if leaves_prefix else np.empty((0,), np.uint64)
        lo_all = np.concatenate(leaves_lo) if leaves_lo else np.empty((0,), np.uint64)
        hi_all = np.concatenate(leaves_hi) if leaves_hi else np.empty((0,), np.uint64)
        i0_all = np.concatenate(leaves_i0) if leaves_i0 else np.empty((0,), np.uint64)
        i1_all = np.concatenate(leaves_i1) if leaves_i1 else np.empty((0,), np.uint64)
        
        order = np.argsort(lo_all, kind='mergesort')  # keep stability
        return LeafTable(
            L=L_all[order],
            prefix=prefix_all[order],
            code_lo=lo_all[order],
            code_hi=hi_all[order],
            idx_lo=i0_all[order],
            idx_hi=i1_all[order],
        )
    
    # ---------- Interval queries over leaves (exist / cover) ----------
    
    def _interval_exists(self, lo: np.uint64, hi: np.uint64) -> bool:
        # True if any leaf interval overlaps [lo,hi]
        Llo = self.leaves.code_lo
        Lhi = self.leaves.code_hi
        i = bisect.bisect_right(Llo, hi) - 1
        return (i >= 0) and (Lhi[i] >= lo)
    
    def _interval_covering_leaf_idx(self, lo: np.uint64, hi: np.uint64) -> int:
        # Return index j if some leaf j fully covers [lo,hi], else -1
        Llo = self.leaves.code_lo
        Lhi = self.leaves.code_hi
        i = bisect.bisect_right(Llo, lo) - 1
        if i >= 0 and Lhi[i] >= hi:
            return i
        return -1
    
    # ---------- Node AABB and decode helpers ----------
    
    def _decode_prefix_to_node(self, prefix: np.uint64, L: int) -> Tuple[int, int, int, int]:
        # Reuse your octree's morton decoder (it expects the packed code,depth)
        L = int(L)
        nd = self._oct.morton3d_decode(int(prefix), L)
        return (int(nd[0]), int(nd[1]), int(nd[2]), int(nd[3]))
    
    def _node_bbox(self, prefix: np.uint64, L: int) -> np.ndarray:
        node = self._decode_prefix_to_node(prefix, L)
        return self._oct.get_bbox(node)  # shape (2,3)
    
    # ---------- Geometry predicates (example: sphere) ----------
    
    @staticmethod
    def _aabb_point_sqdist(aabb: np.ndarray, p: np.ndarray) -> float:
        """
        Min squared distance from a point to an AABB (vector version uses broadcasting).
        """
        lo, hi = aabb[0], aabb[1]
        q = np.maximum(lo, np.minimum(p, hi))
        d = q - p
        return float(np.dot(d, d))
    
    # ---------- Query: points within radius of a sphere (example) ----------
    
    def query_points_in_ball(self,
                             center: np.ndarray,
                             radius: float,
                             coalesce_io: bool = True,
                             return_points: bool = True) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fetch points within a ball B(center, radius) using top-down traversal
        with interval existence and coverage tests. Reads only contiguous ranges.

        Returns
        -------
        original_indices : (K,) int64  -- indices into the original input array
        points           : (K,3) float64 or None
        """
        assert self.leaves is not None, "Index not built"
        center = np.asarray(center, dtype=np.float64)
        r2 = float(radius) * float(radius)
        
        D = self.D
        # DFS stack of (L, prefix)
        stack = [(0, np.uint64(0))]
        ranges: List[Tuple[int, int]] = []
        
        while stack:
            L, p = stack.pop()
            # Optional traversal depth cap
            if L > self.traversal_L_max:
                continue
            
            lo, hi = node_interval(np.uint64(p), np.uint8(L), D)
            lo = np.uint64(lo);
            hi = np.uint64(hi)
            
            # Skip empty nodes
            if not self._interval_exists(lo, hi):
                continue
            
            # Distance prune w.r.t. sphere
            aabb = self._node_bbox(np.uint64(p), L)
            if self._aabb_point_sqdist(aabb, center) > r2:
                # AABB is entirely outside the ball
                continue
            
            # If fully covered by a single leaf, add its contiguous range and stop descending
            j = self._interval_covering_leaf_idx(lo, hi)
            if j >= 0:
                s = int(self.leaves.idx_lo[j])
                e = int(self.leaves.idx_hi[j])
                ranges.append((s, e))
                continue
            
            # Otherwise, descend, but only into existing children (cheap)
            base = (np.uint64(p) << np.uint64(3))
            Lp = L + 1
            for cid in range(8):
                cp = base | np.uint64(cid)
                clo, chi = node_interval(cp, np.uint8(Lp), D)
                if self._interval_exists(np.uint64(clo), np.uint64(chi)):
                    stack.append((Lp, cp))
        
        if not ranges:
            return np.empty((0,), dtype=np.int64), (
                np.empty((0, 3), dtype=self.dtype_points) if return_points else None)
        
        # Coalesce adjacent ranges to minimize I/O
        if coalesce_io:
            ranges.sort()
            merged = [list(ranges[0])]
            for s, e in ranges[1:]:
                if s <= merged[-1][1]:  # overlapping or touching
                    merged[-1][1] = max(merged[-1][1], e)
                else:
                    merged.append([s, e])
            ranges = [(s, e) for s, e in merged]
        
        # Pull slices and filter by exact distance
        idx_parts = []
        pts_parts = [] if return_points else None
        for s, e in ranges:
            pts_chunk = self.points_sorted[s:e]
            d2 = np.sum((pts_chunk - center[None, :]) ** 2, axis=1)
            keep = d2 <= r2
            if np.any(keep):
                # map back to original indices
                orig_idx = self.perm_sorted[s:e][keep].astype(np.int64)
                idx_parts.append(orig_idx)
                if return_points:
                    pts_keep = pts_chunk[keep]
                    pts_parts.append(pts_keep)
        
        if len(idx_parts) == 0:
            return np.empty((0,), dtype=np.int64), (
                np.empty((0, 3), dtype=self.dtype_points) if return_points else None)
        
        idx_all = np.concatenate(idx_parts, axis=0)
        if return_points:
            pts_all = np.concatenate(pts_parts, axis=0)
            return idx_all, pts_all
        else:
            return idx_all, None
    
    def __del__(self):
        if self.workdir is not None and self.delete_workdir:
            shutil.rmtree(self.workdir, ignore_errors=True)


# ---------- Example usage & sanity check ----------

from mmcore.numeric.aabb import point_in_aabb,aabb_intersect_fast_3d

    
def _trilerp_from_corners(corners, tx, ty, tz):
        """
        corners[(a,b,c)] for a,b,c in {0,1}; trilinear at local coords t∈[0,1]^3.
        """
        wx0, wx1, wy0, wy1, wz0, wz1 = (1.0 - tx), tx, (1.0 - ty), ty, (1.0 - tz), tz
        v = 0.0
        for a, wx in ((0, wx0), (1, wx1)):
            for b, wy in ((0, wy0), (1, wy1)):
                for c, wz in ((0, wz0), (1, wz1)):
                    v += corners[(a, b, c)] * wx * wy * wz
        return v


class SDFApprox(Implicit3D):
    far_approx_style: Literal['truncate', 'extrapolate', 'voxel']
    sd_min: float=None
    sd_max: float=None
    
    
    
    def __init__(self, bounds, max_depth=6, min_half=None, **kwargs):
        
        self.min_half = min_half
      
        self._tree = Octree(np.array(bounds), max_depth=max_depth)
        if self.min_half is not None:
            self._tree.set_max_depth_by_min_half(min_half)
        

        self._sd = dict()
  
        self._leafs=None

    def bounds(self):
        return self._tree.aabb
    
    def implicit(self, v) -> float:
        ...
    def __call__(self, pt):
        ...
    def _eval_and_cache(self, fun,nodes, points ):
        sd = np.atleast_1d(fun(points))
        self._sd.update({tuple(node_index): sd_item for node_index, sd_item in zip(nodes.tolist(), sd)})
        return sd
    

    def compute_masks(self,implicit, nodes):
        bbs = self._tree.get_bboxes(nodes)
        centers = np.mean(bbs, axis=1)
        bb = bbs[0]
        d = bb[1, :] - bb[0, :]
        h = d / 2
        
        r = self._tree.r_box(h)
        sd = self._eval_and_cache(implicit,nodes,centers)
       
        
        # sd_arr[nodes[...,0],nodes[...,1],nodes[...,2],nodes[...,3]]=sd
        mask_outside = sd > r
        
        mask_inside = sd < -r
        
        mask = (~(mask_outside | mask_inside))
        return mask_inside,mask, mask_outside

    def _build(self, sdf):
        
      
        #shape=self.sparse4d_shape()
        
        self._sd=dict(
        
        )
        
        #sd_arr= DOK(shape, dtype=float)
        _leafs=[]
        stack=[np.array([self._tree.get_root()],dtype=np.int64)]
        while stack:
       
            nodes = stack.pop(0)
         
            #print("n",nodes.shape)
            if len(nodes) < 1:
               continue
            print("L:", nodes[0][0], 'count:', len(nodes))
            depth_mask=nodes[...,0]>self._tree.max_depth
            

                
            
            mask_inside,mask, mask_outside=self.compute_masks(sdf, nodes)
       
 
            #nodes = nodes[~depth_mask]
            
            if mask.any():
                _leafs.extend(nodes[mask&depth_mask])
              
            
                ch=self._tree.get_children_multiple(  nodes[mask & (~depth_mask)])

                stack.append(np.concatenate(ch,dtype=np.int64))
            #stack.append(np.concatenate(np.apply_along_axis(self.get_children,1,nodes[mask])))
        #SD = coo_array((np.array(list(self._sd.values()),float),np.array(list(self._sd.keys()),int)),  shape=shape )
        #self._sd_ixs_map=dict(zip(self._sd.keys(),range(len(self._sd))))
        #self._sd_arr=np.array(list(self._sd.values()))
        self._leafs=np.array(_leafs, int)

    


    @classmethod
    def from_sdf(cls, sdf, bounds=None,min_half=None,max_depth=6):
        cc=cls(np.asarray(sdf.bounds() if bounds is None else bounds,float),max_depth=max_depth,min_half=min_half)
       
        cc._build(sdf)
        
        return cc
from mmcore.numeric.algorithms.implicit_point import surface_point_local
class ImplicitApprox(SDFApprox):
    def compute_masks(self,implicit, nodes):
        count=len(nodes)
        bbs = self._tree.get_bboxes(nodes)
        centers = np.mean(bbs, axis=1)

        
       
        sd=np.zeros(count)
        mask=np.zeros(count,dtype=bool)
        mask_inside=np.zeros(count,dtype=bool)
        mask_outside=np.zeros(count,dtype=bool)
        for i in range(centers.shape[0]):

            
            res=surface_point_local(implicit,centers[i], bounds=bbs[i],strict=False,full_output=True)
            dst=np.linalg.norm(centers[i]-res.point)
 
            if not res.success:
                
                sd[i]=np.copysign(dst,res.fun)
                
                if res.fun<0:
                    mask_inside[i]=True
                    
                else:
                    mask_outside[i]=True
            else:
                mask[i]=True
                sd[i]=np.copysign(dst,res.fun)
            self._sd[(int(nodes[i,0]),int(nodes[i,1]),int(nodes[i,2]),int(nodes[i,3]))]=sd[i]
            
            
            
        
        
        return mask_inside, mask, mask_outside
    @classmethod
    def from_implicit(cls, implicit, bounds=None, min_half=None, max_depth=6):
        cc = cls(np.asarray(implicit.bounds() if bounds is None else bounds, float), max_depth=max_depth, min_half=min_half)
        
        cc._build(implicit)
        
        return cc
    
        
    

if __name__ == '__main__':
    from mmcore.geom.octree import SDFApprox, Octree
    
    import numpy as np
    
    from mmcore.numeric.intersection.implicit_implicit import ImplicitIntersectionCurve, iterate_curves
    
    from mmcore.geom.primitives import Tube
    
    x, y, v, u, z = [[[12.359112840551504, -7.5948049557495425, 0.0], [2.656625109045951, 1.2155741170561933, 0.0]],
                     [[7.14384241216015, -6.934735074711716, -0.1073366304415263],
                      [7.0788761013028365, 10.016931402130641, 0.8727530304189204]],
                     [[8.072688942425103, -2.3061831591019826, 0.2615779273274319],
                      [7.173685617288537, -3.4427234423361512, 0.4324928834164773],
                      [7.683972288682133, -2.74630545102506, 0.07413871667321925],
                      [7.088944240699163, -4.61458155002528, -0.22460509818398067],
                      [7.304629277158477, -3.9462033818505433, 0.8955725109783643],
                      [7.304629277158477, -3.3362864951018985, 0.8955725109783643],
                      [7.304629277158477, -2.477065729786164, 0.7989970582016114],
                      [7.304629277158477, -2.0988672326949933, 0.7989970582016114]], 0.72648, 1.0]
    
    aa = np.array(x)
    bb = np.array(y)
    
    t1 = Tube(aa[0], aa[1], z, thickness=0.2)
    t2 = Tube(bb[0], bb[1], u, thickness=0.2)
    vv = np.array(v)
    
    import time
    
    crv = ImplicitIntersectionCurve(t1, t2)
    crv.build_tree()
    s = time.time()
    res = []
    for item in iterate_curves(crv):
        res.append(item)
    
    print(time.time() - s)
    
    print(len(res))
    
    from mmcore.geom.implicit import Intersection3D
    
    t12int = Intersection3D(t1, t2)
    bnds = np.array(t12int.bounds())
    
    import time
    
    s = time.perf_counter()
    tree = SDFApprox.from_sdf(t12int, min_half=0.05)
    
    print('build by sdf:', time.perf_counter() - s)
    print(f"SDF Octree built. nodes:{len(tree._sd.keys())}, leafs: {len(tree._leafs)}")
    np.save('/Users/andrewastakhov/dev/mmcore-clean/mmcore/bbs-all.npy',
            tree._tree.get_bboxes(np.asarray(list(tree._sd.keys()), int)))
    np.save('/Users/andrewastakhov/dev/mmcore-clean/mmcore/bbs.npy',
            tree._tree.get_bboxes(np.asarray(tree._leafs, int)))

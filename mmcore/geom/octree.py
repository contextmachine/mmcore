from __future__ import annotations

from mmcore.geom.bvh.lbvh import AABB
from mmcore.geom.implicit import Implicit3D
import itertools
import time
from collections import defaultdict
from collections.abc import Callable

from typing import Literal


# ---------- 2. Octree node -------------------------------------------------
class OctreeNodeCube:
    __slots__ = ("center", "half", "depth", "children")

    def __init__(self, center, half, depth=0):
        self.center = np.asarray(center, dtype=float)  # cube centre
        self.half = float(half)  # half‑edge length
        self.depth = depth
        self.children = []  # empty → leaf

    def is_leaf(self):
        return not self.children

    def _traverse_pts(self, pts=None):
        if pts is None:
            pts = []
        pts.append([self.center - self.half, self.center + self.half])
        if self.children is not None and self.children.__len__() > 0:
            for ch in self.children:
                ch._traverse_pts(pts)
                
        return pts


# ---------- 3. Conservative subdivision using the 1‑Lipschitz bound -------
def build_sdf_octree_cube(node, sdf, max_depth, min_half=1e-3):
    """
    Recursively subdivides 'node' only where the surface may pass through.
    Returns (kept_leaves, visited_nodes)
    """
    # Half‑diagonal length of the cube
    r_box = np.sqrt(3.0) * node.half

    # Evaluate SDF at the cube centre
    d_c = float(sdf(node.center))

    # ---- Outside test ----------------------------------------------------
    # If the *minimum* possible distance (|d_c| - r_box) is positive,
    # every point in the cube is farther than zero from the surface.
    if d_c > r_box:
        return 0, 1  # empty space – prune

    # ---- Inside test -----------------------------------------------------
    # If the *maximum* possible distance (d_c + r_box) is negative,
    # the entire cube lies inside the solid part of the SDF.
    if d_c < -r_box or node.half < min_half or node.depth >= max_depth:
        return 1, 1  # solid leaf (or resolution limit)

    # ---- Otherwise: the surface may cross this cube – subdivide ----------
    kept, visited = 0, 1
    child_half = node.half * 0.5
    for dx in (-child_half, child_half):
        for dy in (-child_half, child_half):
            for dz in (-child_half, child_half):
                child_center = node.center + np.array([dx, dy, dz])
                child = OctreeNodeCube(child_center, child_half, node.depth + 1)
                node.children.append(child)
                k, v = build_sdf_octree_cube(child, sdf, max_depth, min_half)
                kept += k
                visited += v
    return kept, visited
import math

class OctreeNode:
    __slots__ = ("cx","cy","cz","hx","hy","hz","depth","children", "state","_corners")
    def __init__(self, center, half, depth=0):
        self.cx, self.cy, self.cz = center
        self.hx, self.hy, self.hz = half
        self.depth   = depth
        self.children = []
        self.state = 0
        self._corners = None
    def _traverse_pts(self, pts=None):
        if pts is None:
            pts = []

        pts.append(self.get_min_max())
        if self.children is not None and self.children.__len__() > 0:
            for ch in self.children:
                ch._traverse_pts( pts)
        return pts
    
    def get_min_max(self):
        return [(self.cx - self.hx, self.cy - self.hy, self.cz - self.hz),
                (self.cx+ self.hx,
                 self.cy+ self.hy,
                 self.cz+ self.hz)]
    def generate_corners(self):
        if not self._corners:
                self._corners=[None]*8
                i=0
                for dx in (- self.hx,  self.hx):
                    for dy in (- self.hy,  self.hy):
                        for dz in (-self.hz, self.hz):
                            self._corners[i]=(self.cx + dx, self.cy + dy, self.cz + dz)
                            i+=1
            
    @property
    def corners(self):
        if not self._corners:
            self.generate_corners()
        return self._corners


def r_box(node):
    return math.sqrt(node.hx*node.hx +
                     node.hy*node.hy +
                     node.hz*node.hz)


def build_sdf_octree(node, sdf, max_depth, min_half=1e-3, leafs=None):
    if leafs is None:
        leafs = []
    rb = r_box(node)
    dc = float(sdf((node.cx, node.cy, node.cz)))

    if dc >  rb: return 0,1,leafs
    if dc < -rb or node.depth>=max_depth or \
       max(node.hx,node.hy,node.hz) < min_half:
        leafs.append(node)
        return 1,1, leafs

    kept, vis = 0, 1
    hx2, hy2, hz2 = node.hx*0.5, node.hy*0.5, node.hz*0.5
    for sx in (-1,1):
        for sy in (-1,1):
            for sz in (-1,1):
                child = OctreeNode(
                    (node.cx + sx*hx2,
                     node.cy + sy*hy2,
                     node.cz + sz*hz2),
                    (hx2, hy2, hz2),
                    node.depth+1)
                node.children.append(child)
                k,v,_ = build_sdf_octree(child, sdf, max_depth, min_half, leafs)
                kept += k;  vis += v
    return kept, vis, leafs


def subdivide(node, sdf, max_depth, min_half, leaves=None):
    if leaves is None:
        leaves = []
    rb = r_box(node)
    d  = sdf((node.cx, node.cy, node.cz))

    if d >  rb:          # outside
        node.state = -1
        leaves.append(node)
        return
    if d < -rb:          # inside
        node.state = 1
        leaves.append(node)
        return
    if node.depth >= max_depth or max(node.hx,node.hy,node.hz) < min_half:
        leaves.append(node)
        return

    # ambiguous → split
    hx2, hy2, hz2 = node.hx*0.5, node.hy*0.5, node.hz*0.5
    for sx in (-1,1):
        for sy in (-1,1):
            for sz in (-1,1):
                child = OctreeNode((node.cx + sx*hx2,
                              node.cy + sy*hy2,
                              node.cz + sz*hz2),
                             (hx2, hy2, hz2),
                             node.depth+1)
                node.children.append(child)
                subdivide(child, sdf, max_depth, min_half, leaves)
    return leaves
def build_cell_map(leaves=None):

    cell_map = defaultdict(list)      # depth -> list of leaves on that depth
    for leaf in leaves:
        cell_map[leaf.depth].append(leaf)
    return cell_map
def refine_leaf(leaf, cell_map):
    # split once; mark children same state as parent
    hx2, hy2, hz2 = leaf.hx*0.5, leaf.hy*0.5, leaf.hz*0.5
    leaf.children = []
    for sx in (-1,1):
        for sy in (-1,1):
            for sz in (-1,1):
                child = OctreeNode((leaf.cx + sx*hx2,
                              leaf.cy + sy*hy2,
                              leaf.cz + sz*hz2),
                             (hx2, hy2, hz2),
                             leaf.depth+1)
                child.state = leaf.state
                leaf.children.append(child)
                cell_map[child.depth].append(child)
    return leaf.children
def build_balanced(cell_map):
    balanced = False
    while not balanced:
        balanced = True
        # check only neighbour depths differing by >1
        for depth, lst in list(cell_map.items()):
            finer = cell_map.get(depth+2)
            if not finer: continue
            fine_size = finer[0].hx*2.0   # full edge length of depth+2 cell
            coarse_size = fine_size*4.0   # size of depth cell
            for leaf in lst:
                # quick AABB overlap test
                for f in finer:
                    if abs(f.cx-leaf.cx) < (leaf.hx+f.hx) and \
                       abs(f.cy-leaf.cy) < (leaf.hy+f.hy) and \
                       abs(f.cz-leaf.cz) < (leaf.hz+f.hz):
                        refine_leaf(leaf, cell_map)
                        lst.remove(leaf)
                        balanced = False
                        break
                if not balanced:
                    break
            if not balanced:
                break
    return [leaf for depth in cell_map for leaf in cell_map[depth] if not leaf.children]


MAX_DEPTH=6
def world_to_grid(x,bbox,max_depth=MAX_DEPTH):
    return int( round( (x/bbox + 0.5)*(1<<max_depth) ) )


def build_vertex_table(sdf,leaves, bbox, max_depth=MAX_DEPTH):
    vertex_sd = {}
    bbx,bby,bbz=(bbox[1]-bbox[0])
    for leaf in leaves:
        gx0 = world_to_grid(leaf.cx - leaf.hx, bbx,max_depth)
        gy0 = world_to_grid(leaf.cy - leaf.hy, bby,max_depth)
        gz0 = world_to_grid(leaf.cz - leaf.hz, bbz,max_depth)
        gx1 = world_to_grid(leaf.cx + leaf.hx, bbx,max_depth)
        gy1 = world_to_grid(leaf.cy + leaf.hy, bby,max_depth)
        gz1 = world_to_grid(leaf.cz + leaf.hz, bbz,max_depth)
        for ix in (gx0, gx1):
            for iy in (gy0, gy1):
                for iz in (gz0, gz1):
                    key = (ix,iy,iz)
                    if key not in vertex_sd:
                        # convert grid ix back to world coordinate
                        wx = (ix/(1<<max_depth) - 0.5)*bbx
                        wy = (iy/(1<<max_depth) - 0.5)*bby
                        wz = (iz/(1<<max_depth) - 0.5)*bbz
                        vertex_sd[key] = sdf((wx,wy,wz))
    return vertex_sd


def sample_trilinear(node, p, vertex_sd, max_depth=MAX_DEPTH):
    """Evaluate the adaptively‑sampled field at world point p=(x,y,z)."""
    # locate the leaf that owns p  (simple depth‑first walk)
    bbox=max([node.hx,node.hy,node.hz])*2
    while node.children:
        sx = -1 if p[0] < node.cx else 1
        sy = -1 if p[1] < node.cy else 1
        sz = -1 if p[2] < node.cz else 1
        idx = (sx>0)*4 + (sy>0)*2 + (sz>0)
        node = node.children[idx]

    # fetch its eight vertices
    gx0 = world_to_grid(node.cx - node.hx,bbox)
    gy0 = world_to_grid(node.cy - node.hy,bbox)
    gz0 = world_to_grid(node.cz - node.hz,bbox)
    gx1, gy1, gz1 = gx0 + (1<<(max_depth-node.depth)), \
                    gy0 + (1<<(max_depth-node.depth)), \
                    gz0 + (1<<(max_depth-node.depth))

    # fraction inside cell
    fx = (p[0] - (node.cx-node.hx)) / (2*node.hx)
    fy = (p[1] - (node.cy-node.hy)) / (2*node.hy)
    fz = (p[2] - (node.cz-node.hz)) / (2*node.hz)

    d000 = vertex_sd[(gx0,gy0,gz0)]
    d100 = vertex_sd[(gx1,gy0,gz0)]
    d010 = vertex_sd[(gx0,gy1,gz0)]
    d110 = vertex_sd[(gx1,gy1,gz0)]
    d001 = vertex_sd[(gx0,gy0,gz1)]
    d101 = vertex_sd[(gx1,gy0,gz1)]
    d011 = vertex_sd[(gx0,gy1,gz1)]
    d111 = vertex_sd[(gx1,gy1,gz1)]

    c00 = d000*(1-fx) + d100*fx
    c10 = d010*(1-fx) + d110*fx
    c01 = d001*(1-fx) + d101*fx
    c11 = d011*(1-fx) + d111*fx
    c0  = c00*(1-fy) + c10*fy
    c1  = c01*(1-fy) + c11*fy
    return c0*(1-fz) + c1*fz



from typing import NamedTuple, Union, Sequence, Optional, Tuple

import math
import numpy as np
NodeIndex = tuple[int, int, int, int]




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

    
    def find_nodes(self, points: np.ndarray, min_points_per_node: int=10000) -> np.ndarray:
        points_ixs=np.arange(points.shape[0], dtype=np.int64)
        stack=[(self.get_root(), points_ixs)]
      
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
            
     
           
            #print(depth_mask)
            #print("N",nodes)
            if len(nodes) < 1:
                continue
                
            
            bbs=self._tree.get_bboxes(nodes)
            centers=np.mean(bbs,axis=1)
            bb=bbs[0]
            d=bb[1,:]-bb[0,:]
            h=d/2
           
            r=self._tree.r_box(h)
            
            sd=np.atleast_1d(sdf(centers))
            #sd_arr[nodes[...,0],nodes[...,1],nodes[...,2],nodes[...,3]]=sd
            self._sd.update({tuple(node_index):sd_item for node_index, sd_item in zip(nodes.tolist(),sd)})
            mask_outside=sd>r
           
            mask_inside= sd < -r
         
            mask=(~(mask_outside | mask_inside))
  
            
                
            
            print(mask.shape, depth_mask.shape)
 
            #nodes = nodes[~depth_mask]
            
            if mask.any():
                _leafs.extend(nodes[mask&depth_mask])
              
            
                ch=self._tree.get_children_multiple(  nodes[mask & (~depth_mask)])
                print('c',ch.shape,ch.dtype)
                stack.append(np.concatenate(ch,dtype=np.int64))
            #stack.append(np.concatenate(np.apply_along_axis(self.get_children,1,nodes[mask])))
        #SD = coo_array((np.array(list(self._sd.values()),float),np.array(list(self._sd.keys()),int)),  shape=shape )
        #self._sd_ixs_map=dict(zip(self._sd.keys(),range(len(self._sd))))
        #self._sd_arr=np.array(list(self._sd.values()))
        self._leafs=np.array(_leafs, int)

    


    @classmethod
    def from_sdf(cls, sdf, min_half=None,max_depth=6):
        cc=cls(np.asarray(sdf.bounds(),float),max_depth=max_depth,min_half=min_half)
       
        cc._build(sdf)
        
        return cc
    
    
    
        



# ---------- 4. Driver ------------------------------------------------------
if __name__ == "__main__":

    # ---------- 1. True SDF: torus (major radius R, tube radius r) ----------
    def sdf_torus_vec(p, R=1.0, r=0.30):
        """
        True signed‑distance function of a torus centred at the origin, lying
        in the XY‑plane (so the 'ring' is R units from the Z‑axis).
        p : (..., 3) array‑like of points
        returns : (...,) ndarray of signed distances
        """
        p = np.asarray(p, dtype=float)

        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        q = np.stack((np.sqrt(x * x + y * y) - R, z), axis=-1)
        return np.linalg.norm(q, axis=-1) - r  # ∥q∥ − r
    def sdf_torus(p,R=1.0,r=0.3):
        x,y,z=p
        q=math.sqrt(x*x+y*y)-R
        return math.sqrt(q*q+z*z)-r
    s=time.perf_counter()
    root = OctreeNode(center=[0, 0, 0], half=[2.0,2.0,2.0])  # world cube [-2,2]³
    max_depth = 6  # 128³ voxel resolution
    kept, visited ,leafs= build_sdf_octree(root, sdf_torus_vec, max_depth)
    print(time.perf_counter()-s)
    print(f"Visited {visited:,d} nodes")
    print(f"Kept    {kept:,d} leaf cubes that may intersect the torus")
    from mmcore.geom.primitives import Tube
    from mmcore.geom.implicit import Intersection3D, Implicit3D
    import numpy as np

    from mmcore.numeric.intersection.implicit_implicit import ImplicitIntersectionCurve, iterate_curves

    from mmcore.geom.primitives import Tube

    x, y, v, u, z = [
        [[12.359112840551504, -7.5948049557495425, 0.0], [2.656625109045951, 1.2155741170561933, 0.0]],
        [[7.14384241216015, -6.934735074711716, -0.1073366304415263], [7.0788761013028365, 10.016931402130641, 0.8727530304189204]],
        [
            [8.072688942425103, -2.3061831591019826, 0.2615779273274319],
            [7.173685617288537, -3.4427234423361512, 0.4324928834164773],
            [7.683972288682133, -2.74630545102506, 0.07413871667321925],
            [7.088944240699163, -4.61458155002528, -0.22460509818398067],
            [7.304629277158477, -3.9462033818505433, 0.8955725109783643],
            [7.304629277158477, -3.3362864951018985, 0.8955725109783643],
            [7.304629277158477, -2.477065729786164, 0.7989970582016114],
            [7.304629277158477, -2.0988672326949933, 0.7989970582016114],
        ],
        0.72648,
        1.0,
    ]

    aa = np.array(x)
    bb = np.array(y)

    t1 = Tube(aa[0], aa[1], z, thickness=0.2)
    t2 = Tube(bb[0], bb[1], u, thickness=0.2)
    vv = np.array(v)

    inter=Intersection3D(t1, t2)
    bbox = np.array(inter.bounds())
    center=np.average(bbox, axis=0)
    root = OctreeNode(center=center, half=(bbox[1] - center))
    max_depth = 6  # 128³ voxel resolution
    s=time.perf_counter()
    kept, visited,leafs = build_sdf_octree(root, inter, max_depth)

    print(time.perf_counter()-s)

    pts=np.array(root._traverse_pts()).tolist()
    root2 = OctreeNodeCube(center=center, half=max(bbox[1] - center))
    max_depth = 6  # 128³ voxel resolution
    kept2, visited2 = build_sdf_octree_cube(root2, inter, max_depth)
    pts2=np.array(root2._traverse_pts()).tolist()

    bbox=4.

    # world box [-2,2]³
    root = OctreeNode((0.0, 0.0, 0.0), (4 * 0.5,) * 3, 0)
    print([root.hx * 2, root.hy * 2, root.hz * 2])
    bbox = np.array([[-root.hx, -root.hy, -root.hz],[root.hx, root.hy, root.hz]])
    MAX_DEPTH = 6
    MIN_HALF = max(bbox[1]-bbox[0]) / (2**MAX_DEPTH)

    leaves = []
    t0 = time.perf_counter()
    subdivide(root, sdf_torus, MAX_DEPTH, MIN_HALF, leaves)
    # cell_map = build_cell_map(leaves)
    # final_leaves =    build_balanced(cell_map)

    # print(f"Phase B balance: {len(final_leaves):,} leaves")

    vertex_sd = build_vertex_table(sdf_torus, leaves, bbox, MAX_DEPTH)
    print(f"   stored vertices: {len(vertex_sd):,}")
    t_build = time.perf_counter()-t0
    print(f"Total build time: {1000*t_build:.1f} ms")
    MAX_DEPTH=10
    # quick check across a coarse‑fine boundary
    pa = (-0.25, 0.0, 0.0)          # falls in 2× finer leaf than ...
    pb = (-0.2499, 0.0, 0.0)        # ... this point

    print("φ(pa),gp =", sample_trilinear(root,pa, vertex_sd),sdf_torus(np.array(pa)),
          "φ(pb),gp =", sample_trilinear(root,pb,vertex_sd),sdf_torus(np.array(pb)),)
    eval_pts=[]
    for i in np.linspace(-2,2,10):
        for j in np.linspace(-2,2,10):
            for k in np.linspace(-2,2,10):
                eval_pts.append((i,j,k))
    calls_stats=[]

    for pt in eval_pts:
        s=time.perf_counter()
        res=sample_trilinear(root, pt, vertex_sd)
        end=time.perf_counter()-s
        calls_stats.append((pt,res,end))
    P = np.random.uniform(bbox[0], bbox[1], size=(50_000, 3))

    pts,results,times=zip(*calls_stats)
    print("sample_trilinear time:",np.mean(
    times, ),f"min: {min(times)}" ,f"max: {max(times)}" )
    ress=np.apply_along_axis(lambda pt: sample_trilinear(root, pt, vertex_sd),1,P )
    rest=sdf_torus_vec(P)
    print(np.mean(ress-rest))
    
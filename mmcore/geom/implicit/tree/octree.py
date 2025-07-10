import time
from collections import defaultdict

import numpy as np


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
    __slots__ = ("cx","cy","cz","hx","hy","hz","depth","children", "_corners")
    def __init__(self, center, half, depth=0):
        self.cx, self.cy, self.cz = map(float, center)
        self.hx, self.hy, self.hz = map(float, half)
        self.depth   = depth
        self.children = []
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
        return [(self.cx - self.hx, self.cy - self.hy, self.cz - self.hz), (self.cx + self.hx, self.cy + self.hy, self.cz + self.hz)]
     # Utility: Cartesian coordinates of the 8 corners
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
    

class OctreeNodeV2:                 # axis‑aligned box (AABB)
    __slots__ = ("cx","cy","cz","hx","hy","hz","depth",
                 "children","state")                 # state: -1 out, 0 unknown, 1 in
    def __init__(self, center, half, depth):
        self.cx, self.cy, self.cz = center           # floats
        self.hx, self.hy, self.hz = half             # floats
        self.depth  = depth
        self.children = []
        self.state  = 0
    def get_min_max(self):
        return [(self.cx - self.hx, self.cy - self.hy, self.cz - self.hz),
        (self.cx+ self.hx,
        self.cy+ self.hy,
        self.cz+ self.hz)]


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
                child = OctreeNodeV2((node.cx + sx*hx2,
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
                child = OctreeNodeV2((leaf.cx + sx*hx2,
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
    for leaf in leaves:
        gx0 = world_to_grid(leaf.cx - leaf.hx, bbox,max_depth)
        gy0 = world_to_grid(leaf.cy - leaf.hy, bbox,max_depth)
        gz0 = world_to_grid(leaf.cz - leaf.hz, bbox,max_depth)
        gx1 = world_to_grid(leaf.cx + leaf.hx, bbox,max_depth)
        gy1 = world_to_grid(leaf.cy + leaf.hy, bbox,max_depth)
        gz1 = world_to_grid(leaf.cz + leaf.hz, bbox,max_depth)
        for ix in (gx0, gx1):
            for iy in (gy0, gy1):
                for iz in (gz0, gz1):
                    key = (ix,iy,iz)
                    if key not in vertex_sd:
                        # convert grid ix back to world coordinate
                        wx = (ix/(1<<max_depth) - 0.5)*bbox
                        wy = (iy/(1<<max_depth) - 0.5)*bbox
                        wz = (iz/(1<<max_depth) - 0.5)*bbox
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
    root = OctreeNode(center=[0, 0, 0], half=[2.0,2.0,2.0])  # world cube [-2,2]³
    max_depth = 6  # 128³ voxel resolution
    kept, visited ,leafs= build_sdf_octree(root, sdf_torus_vec, max_depth)

    print(f"Visited {visited:,d} nodes")
    print(f"Kept    {kept:,d} leaf cubes that may intersect the torus")
    from mmcore.geom.primitives import Tube
    from mmcore.geom.implicit import Intersection3D
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
    root = OctreeNodeV2((0.0, 0.0, 0.0), (bbox * 0.5,) * 3, 0)
    print([root.hx * 2, root.hy * 2, root.hz * 2])
    bbox = np.max([root.hx*2, root.hy*2, root.hz*2])
    MAX_DEPTH = 6
    MIN_HALF = bbox / (2**MAX_DEPTH)

    leaves = []
    t0 = time.perf_counter()
    subdivide(root, sdf_torus, MAX_DEPTH, MIN_HALF, leaves)
    cell_map = build_cell_map(leaves)
    final_leaves =    build_balanced(cell_map)

    print(f"Phase B balance: {len(final_leaves):,} leaves")

    vertex_sd = build_vertex_table(sdf_torus, leaves, bbox, MAX_DEPTH)
    print(f"   stored vertices: {len(vertex_sd):,}")

    t_build = time.perf_counter()-t0
    print(f"Total build time: {1000*t_build:.1f} ms")
    MAX_DEPTH=10
    # quick check across a coarse‑fine boundary
    pa = (-0.25, 0.0, 0.0)          # falls in 2× finer leaf than ...
    pb = (-0.2499, 0.0, 0.0)        # ... this point
    print("φ(pa) =", sample_trilinear(root,pa, vertex_sd),
          "φ(pb) =", sample_trilinear(root,pb,vertex_sd))
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

    pts,results,times=zip(*calls_stats)
    print("sample_trilinear time:",np.mean(
    times, ),f"min: {min(times)}" ,f"max: {max(times)}" )

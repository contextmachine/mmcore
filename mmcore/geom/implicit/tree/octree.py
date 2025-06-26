import numpy as np


# ---------- 1. True SDF: torus (major radius R, tube radius r) ----------
def sdf_torus(p, R=1.0, r=0.30):
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
    __slots__ = ("cx","cy","cz","hx","hy","hz","depth","children")
    def __init__(self, center, half, depth=0):
        self.cx, self.cy, self.cz = map(float, center)
        self.hx, self.hy, self.hz = map(float, half)
        self.depth   = depth
        self.children = []

    def _traverse_pts(self, pts=None):
        if pts is None:
            pts = []
        
        pts.append([(self.cx - self.hx, self.cy - self.hy, self.cz - self.hz), (self.cx + self.hx, self.cy + self.hy, self.cz + self.hz)])
        if self.children is not None and self.children.__len__() > 0:
            for ch in self.children:
                ch._traverse_pts( pts)
        return pts


def r_box(node):
    return math.sqrt(node.hx*node.hx +
                     node.hy*node.hy +
                     node.hz*node.hz)

def build_sdf_octree(node, sdf, max_depth, min_half=1e-3):
    rb = r_box(node)
    dc = float(sdf((node.cx, node.cy, node.cz)))

    if dc >  rb: return 0,1
    if dc < -rb or node.depth>=max_depth or \
       max(node.hx,node.hy,node.hz) < min_half:
        return 1,1

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
                k,v = build_sdf_octree(child, sdf, max_depth, min_half)
                kept += k;  vis += v
    return kept, vis

# ---------- 4. Driver ------------------------------------------------------
if __name__ == "__main__":
    root = OctreeNode(center=[0, 0, 0], half=[2.0,2.0,2.0])  # world cube [-2,2]³
    max_depth = 6  # 128³ voxel resolution
    kept, visited = build_sdf_octree(root, sdf_torus, max_depth)

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
    kept, visited = build_sdf_octree(root, inter, max_depth)
    pts=np.array(root._traverse_pts()).tolist()
    root2 = OctreeNodeCube(center=center, half=max(bbox[1] - center))
    max_depth = 6  # 128³ voxel resolution
    kept2, visited2 = build_sdf_octree_cube(root2, inter, max_depth)
    pts2=np.array(root2._traverse_pts()).tolist()


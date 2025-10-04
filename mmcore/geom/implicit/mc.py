from __future__ import annotations
import  itertools
import warnings

import numpy as np
from mmcore.geom.octree import Octree
from typing import List, Tuple, Callable



# Vertices of the unit cube in (x,y,z) order
VERTS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

# Generate **all 28 unique edges** automatically (12 + 12 + 4)
EDGE2IDX, EDGES = {}, []
for v0, v1 in itertools.combinations(range(8), 2):
    EDGE2IDX[(v0, v1)] = EDGE2IDX[(v1, v0)] = len(EDGES)
    EDGES.append((v0, v1))  # index → (vertex0, vertex1)

# 6‑tet “long‑diagonal” decomposition (works fine for MC table generation)
TETS = [(0, 5, 1, 6), (0, 1, 2, 6), (0, 2, 3, 6), (0, 3, 7, 6), (0, 7, 4, 6), (0, 4, 5, 6)]

# Local edges inside a tetra (indices into the *tet's* four vertices)
TET_EDGES = [(0, 1), (1, 2), (2, 0), (0, 3), (1, 3), (2, 3)]



def triangulate_tet(mask: int) -> List[Tuple[int, int, int]]:
    """
    Given the 4-bit sign mask of a tetra, return a list of triangles
    with *correct* connectivity and winding.
    """
    inside = [i for i in range(4) if mask & (1 << i)]
    # nothing to do if all in or all out
    if len(inside) in (0, 4):
        return []

    # collect the tetra edges whose endpoints have opposite signs
    edge_ids = [e_id for e_id, (a, b) in enumerate(TET_EDGES) if bool(mask & (1 << a)) ^ bool(mask & (1 << b))]

    # helper: do two local edges share a tetra‐vertex?
    def connected(e1: int, e2: int) -> bool:
        a1, b1 = TET_EDGES[e1]
        a2, b2 = TET_EDGES[e2]
        return bool({a1, b1} & {a2, b2})

    if len(edge_ids) == 3:
        # --- single triangle: order the three edges into a cycle
        e0 = edge_ids[0]
        # find the two neighbors of e0
        nbrs = [e for e in edge_ids if e != e0 and connected(e0, e)]
        e1, e2 = nbrs
        # make sure e1 and e2 themselves connect
        if not connected(e1, e2):
            e1, e2 = e2, e1
        return [(e0, e1, e2)]

    if len(edge_ids) == 4:
        # --- quad: walk the edges into a 4‐cycle
        cycle = [edge_ids[0]]
        used = {cycle[0]}
        while len(cycle) < 4:
            prev = cycle[-1]
            for e in edge_ids:
                if e not in used and connected(prev, e):
                    cycle.append(e)
                    used.add(e)
                    break
        a, b, c, d = cycle
        # split along the (a→c) diagonal
        return [(a, b, c), (a, c, d)]

    raise RuntimeError("Invalid tetra configuration")


# Pre‑compute the 16 patterns once
TET_TRI = [triangulate_tet(m) for m in range(16)]


def tet_edge_to_cube_edge(tet: Tuple[int, int, int, int], e_local: int) -> int:
    """Return the **global cube edge index** for a tet‑local edge."""
    a_local, b_local = TET_EDGES[e_local]
    v0, v1 = tet[a_local], tet[b_local]
    return EDGE2IDX[(v0, v1)]  # always exists (now!)


def generate_mc_table() -> List[List[int]]:
    mc = []
    for cube_mask in range(256):
        tri_edges = []
        for tet in TETS:
            # Build 4‑bit mask for this tetrahedron
            tmask = sum(1 << i for i, v in enumerate(tet) if cube_mask & (1 << v))
            for tri in TET_TRI[tmask]:
                for e_local in tri:
                    tri_edges.append(tet_edge_to_cube_edge(tet, e_local))
        tri_edges.append(-1)  # terminator
        mc.append(tri_edges)
    return mc


MC_TABLE = generate_mc_table()

# Precompute helpers for vectorised polygonisation
EDGES_ARR = np.asarray(EDGES, dtype=np.int64)
CUBE_OFFS = np.asarray([(vx * 2 - 1, vy * 2 - 1, vz * 2 - 1) for vx, vy, vz in VERTS], dtype=float)

def _build_mc_tris():
    out = []
    for row in MC_TABLE:
        tri_e = row[:-1]  # drop terminator
        if len(tri_e) == 0:
            out.append(np.empty((0, 3), dtype=np.int16))
        else:
            out.append(np.asarray(tri_e, dtype=np.int16).reshape(-1, 3))
    return out

MC_TABLE_TRIS = _build_mc_tris()



def polygonise_nodes_bulk(
    octree: Octree,
    nodes: np.ndarray,
    sdf: Callable,
    iso: float = 0.0,
    preserve_orientation: bool = True,
    q: float = 1e-6,
):
    """
    Build a watertight mesh for an *arbitrary* list of octree leaves
    with only TWO vectorised SDF calls, no matter how many leaves.

    Returns
    -------
    vdict : dict   {rounded‑coord → index}
    faces : list   [(i, j, k), …]
    """
    if nodes is None or len(nodes) == 0:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)

    # ── 1.  Corner sampling in one shot ───────────────────────────────────────
    n = len(nodes)
    bbs = octree.get_bboxes(nodes)
    centres = bbs.mean(axis=1)
    halves = 0.5 * (bbs[:, 1] - bbs[:, 0])

    cpos = centres[:, None, :] + halves[:, None, :] * CUBE_OFFS     # (n, 8, 3)
    cval = sdf(cpos.reshape(-1, 3)).reshape(n, 8)                   # ← SDF #1

    # Bit‑mask of inside corners per cube
    masks = np.zeros(n, dtype=np.uint16)
    for i in range(8):
        masks |= (cval[:, i] < iso).astype(np.uint16) << i

    # ── 2.  Build triangle soup (no more SDF calls here) ──────────────────────
    tri_blocks, eps_blocks = [], []         # collect per‑leaf triangle blocks
    a = EDGES_ARR[:, 0]
    b = EDGES_ARR[:, 1]
    unique_masks = np.unique(masks)
    for m in unique_masks:
        if m in (0, 0xFF):
            continue
        idxs = np.where(masks == m)[0]
        if idxs.size == 0:
            continue
        cp = cpos[idxs]                                # (k,8,3)
        cv = cval[idxs]                                # (k,8)

        pa, pb = cp[:, a], cp[:, b]                    # (k,28,3)
        va, vb = cv[:, a], cv[:, b]                    # (k,28)
        denom = (vb - va)
        with np.errstate(divide='ignore', invalid='ignore'):
            t = (iso - va) / denom                     # (k,28)
            t = np.where(denom == 0.0, 0.0, t)
        ep = pa + t[..., None] * (pb - pa)             # (k,28,3)

        tris_idx = MC_TABLE_TRIS[int(m)]               # (T,3)
        if tris_idx.size == 0:
            continue
        tris_pts = ep[:, tris_idx, :]                  # (k,T,3,3)
        k, Tn = tris_pts.shape[0], tris_pts.shape[1]
        tri_blocks.append(tris_pts.reshape(-1, 3, 3))
        eps_per_node = 1e-4 * halves[idxs].max(axis=1) # (k,)
        eps_blocks.append(np.repeat(eps_per_node, Tn))

    if not tri_blocks:
        return np.empty((0, 3), dtype=float), np.empty((0, 3), dtype=int)

    # ── 3.  One‑shot orientation fix (optional) ───────────────────────────────
    tris = np.concatenate(tri_blocks, axis=0)           # (T,3,3)
    eps  = np.concatenate(eps_blocks).reshape(-1, 1)     # (T,1)
    if preserve_orientation:

        nrm  = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        ln   = np.linalg.norm(nrm, axis=1, keepdims=True)
        

        
        with warnings.catch_warnings(action='ignore'):
            nrm  = np.where(ln == 0, nrm, nrm / ln) * eps        # unit * eps

        ctr  = tris.mean(axis=1)
        test = np.concatenate([ctr + nrm, ctr - nrm], axis=0)
        d    = sdf(test).reshape(2, -1)                      # ← SDF #2

        flip = d[0] < d[1]                                   # inward?
        if np.any(flip):
            tris[flip, 1], tris[flip, 2] = tris[flip, 2].copy(), tris[flip, 1].copy()

    # ── 4.  Deduplicate & index vertices ──────────────────────────────────────
    pts = tris.reshape(-1, 3)                     # (T*3,3)
    K = np.rint(pts / q).astype(np.int64)         # integer grid keys
    uniq, inv = np.unique(K, axis=0, return_inverse=True)
    V = uniq.astype(float) * q
    F = inv.reshape(-1, 3).astype(np.int64)
    return V, F
import functools
@functools.lru_cache(maxsize=None)
def _v_id(p, vdict, q=1e-6):
    key = (round(p[0] / q), round(p[1] / q), round(p[2] / q))
    if key not in vdict:
        vdict[key] = len(vdict)
    return vdict[key]


def write_ply(path: str, verts: dict, faces: List[Tuple[int, int, int]]):
    
    with open(path, "w", encoding="utf8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\n")
        f.write("end_header\n")
        for x, y, z in verts:
            f.write(f"{x} {y} {z}\n")
        for a, b, c in faces:
            f.write(f"3 {a} {b} {c}\n")



def _r_box(half: np.ndarray) -> np.ndarray:
    return np.linalg.norm(half, axis=1)


def _build_surface_leaves(octree: Octree, sdf: Callable, iso: float, min_half: float, max_depth: int) -> np.ndarray:
    """Vectorised octree refinement using 1‑Lipschitz pruning.

    Returns an array of nodes with shape (N, 4) where each row is [L, x, y, z]
    that likely intersects the iso‑surface or reached termination criteria.
    """
    stack = [np.array([octree.get_root()], dtype=np.int64)]
    leaves = []

    while stack:
        nodes = stack.pop(0)
        if nodes.size == 0:
            continue

        bbs = octree.get_bboxes(nodes)
        centres = bbs.mean(axis=1)
        halves = 0.5 * (bbs[:, 1] - bbs[:, 0])
        radii = _r_box(halves)

        # Distance to iso level
        
        vals = np.asarray(sdf(centres), dtype=float) - float(iso)

        outside = vals > radii
        inside = vals < -radii
        uncertain = ~(outside | inside)
    
        if not np.any(uncertain):
            # Nothing to refine in this batch
            continue

        # Termination: reached max depth OR cell small enough
        L = nodes[:, 0]
        small = np.max(halves, axis=1) <= float(min_half)
        stop = (L >= max_depth) | small

        # Keep uncertain nodes that should stop as leaves
        to_keep = uncertain & stop
        if np.any(to_keep):
            leaves.append(nodes[to_keep])

        # Refine uncertain nodes that can still split
        to_split = uncertain & (~stop)
        if np.any(to_split):
            ch = octree.get_children_multiple(nodes[to_split])  # (8, K, 4)
            stack.append(np.concatenate(ch, axis=0).astype(np.int64))

    return np.concatenate(leaves, axis=0) if leaves else np.empty((0, 4), dtype=np.int64)


def marching_cubes(sdf, bounds, min_half=1e-3, max_depth=7, iso: float = 0.0, preserve_orientation: bool = True):
    """Extract an iso‑surface using the virtual Octree with vectorised refinement.

    Parameters
    - sdf: Callable accepting (N,3) points and returning (N,) distances.
    - bounds: (2,3) array_like AABB.
    - min_half: minimum half‑size to stop refining (isotropic threshold).
    - max_depth: maximum octree depth.
    - iso: iso‑value to extract (default 0.0).
    """
    oct = Octree(np.array(bounds, dtype=float), max_depth=int(max_depth))
    if min_half is not None:
        try:
            oct.set_max_depth_by_min_half(float(min_half))
        except Exception:
            # Fallback to provided max_depth if heuristic fails
            pass

    nodes = _build_surface_leaves(oct, sdf, iso=float(iso), min_half=float(min_half), max_depth=int(oct.max_depth))
    verts, faces = polygonise_nodes_bulk(oct, nodes, sdf, iso=float(iso), preserve_orientation=preserve_orientation, q=min(1e-6, float(min_half)))
    return verts, faces


if __name__ == "__main__":

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

    # Simple demo with torus bounds
    bbox = np.array([[-2.5, -2.5, -2.0], [2.5, 2.5, 2.0]], dtype=float)
    import time

    s = time.perf_counter()
    V, F = marching_cubes(sdf_torus_vec, bbox, min_half=0.02, max_depth=8, iso=0.0)
    print("MC time:", time.perf_counter() - s)

    print(f"Mesh: {len(V):,d} vertices, {len(F):,d} triangles")
    write_ply("torus.ply", V, F)
    print("Output written to  torus.ply  (open in MeshLab / Blender)")
   
    
    import numpy as np
    

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
    

   
    
    from mmcore.geom.implicit import Intersection3D
    
    t12int = Intersection3D(t1, t2)
    bnds = np.array(t12int.bounds())
    
    s = time.perf_counter()
    V, F = marching_cubes(t12int, bnds, min_half=0.004, max_depth=8, iso=0.0)
    print("MC time:", time.perf_counter() - s)
    
    print(f"Mesh: {len(V):,d} vertices, {len(F):,d} triangles")
    write_ply("tubes_inter.ply", V, F)
    print("Output written to  torus.ply  (open in MeshLab / Blender)")

#!/usr/bin/env python3
"""
Adaptive Marching‑Cubes driven by the 1‑Lipschitz property of a true SDF.
– Builds a compact AABB tree with conservative pruning.
– Generates the *full* 256‑case MC table at run‑time (no hand typing).
– Extracts a watertight triangle mesh and writes it to torus.ply.

Tested on Python 3.11, no third‑party modules required.
"""

from __future__ import annotations
import math, itertools
from collections import defaultdict

import numpy as np

from mmcore.geom.implicit.tree.octree import OctreeNode,build_sdf_octree
from typing import List, Tuple

# -------------------------------------------------------------------------
# 1.  A TRUE signed‑distance function: torus centred at the origin
# -------------------------------------------------------------------------


# -------------------------------------------------------------------------
# 3.  Cube topology (28 edges, six tetrahedra)
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
# 4.  Build the 256‑entry Marching‑Cubes table on the fly
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
# 5.  Polygonise one adaptive leaf
# -------------------------------------------------------------------------
def interpolate(p0, p1, v0, v1, iso=0.0):
    t = (iso - v0) / (v1 - v0)
    return (p0[0] + t * (p1[0] - p0[0]), p0[1] + t * (p1[1] - p0[1]), p0[2] + t * (p1[2] - p0[2]))


def polygonise_leaf(node: OctreeNode, sdf, vdict: dict, faces: List[Tuple[int, int, int]], iso: float = 0.0, preserve_orientation: bool = True, q:float = 1e-6):
    # Corner positions & SDF values
    cpos, cval = [], []

    for vx, vy, vz in VERTS:
        px = node.cx + (vx * 2 - 1) * node.hx
        py = node.cy + (vy * 2 - 1) * node.hy
        pz = node.cz + (vz * 2 - 1) * node.hz
        val = sdf((px, py, pz))
        cpos.append((px, py, pz))
        cval.append(val)

    mask = sum((1 << i) for i, v in enumerate(cval) if v < iso)
    if mask == 0 or mask == 0xFF:  # no surface in this box
        return

    # On‑demand vertex cache: edge‑index → 3‑tuple
    evert = {}

    def vert_on_edge(eid: int):
        if eid not in evert:
            a, b = EDGES[eid]
            evert[eid] = interpolate(cpos[a], cpos[b], cval[a], cval[b], iso)
        return evert[eid]

    def _ensure_outward(p0, p1, p2):
        # geometric normal
        ux, uy, uz = p1[0] - p0[0], p1[1] - p0[1], p1[2] - p0[2]
        vx, vy, vz = p2[0] - p0[0], p2[1] - p0[1], p2[2] - p0[2]
        nx, ny, nz = uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx
        ln = math.sqrt(nx * nx + ny * ny + nz * nz)
        if ln == 0.0:  # degenerate, keep as is
            return p0, p1, p2

        # centroid + tiny step along the normal
        eps = 1e-4 * max(node.hx, node.hy, node.hz)
        nx, ny, nz = nx * eps, ny / ln * eps, nz / ln * eps
        cx, cy, cz = (p0[0] + p1[0] + p2[0]) / 3, (p0[1] + p1[1] + p2[1]) / 3, (p0[2] + p1[2] + p2[2]) / 3

        if sdf((cx + nx, cy + ny, cz + nz)) < sdf((cx - nx, cy - ny, cz - nz)):

            # stepping along the normal went *inside* → flip winding
            return p0, p2, p1

        return p0, p1, p2

    tri_edges = MC_TABLE[mask]
    for i in range(0, len(tri_edges) - 1, 3):  # stop before trailing ‑1
        ev0, ev1, ev2 = tri_edges[i : i + 3]
        p0, p1, p2 = vert_on_edge(ev0), vert_on_edge(ev1), vert_on_edge(ev2)
        if preserve_orientation:
            p0, p1, p2 = _ensure_outward(p0, p1, p2)
        v0 = _v_id(p0, vdict)
        v1 = _v_id(p1, vdict)
        v2 = _v_id(p2, vdict)
        faces.append((v0, v1, v2))

class _frozendict(dict):
    def __init__(self, *args, **kwargs):
        super().__init__(*args,**kwargs)
        self.__hash = hash(frozenset(self))
    def __hash__(self):
        return self.__hash
    
# ──────────────────────────────────────────────────────────────────────────────
# 4‑bis.  Polygonise *all* leaves in one vectorised pass
# ──────────────────────────────────────────────────────────────────────────────
def polygonise_leaves_bulk(
    leaves: list[OctreeNode],
    sdf,
    iso: float = 0.0,
    preserve_orientation: bool = True,q=1e-6,
):
    """
    Build a watertight mesh for an *arbitrary* list of octree leaves
    with only TWO vectorised SDF calls, no matter how many leaves.

    Returns
    -------
    vdict : dict   {rounded‑coord → index}
    faces : list   [(i, j, k), …]
    """
    if not leaves:
        return {}, []

    # ── 1.  Corner sampling in one shot ───────────────────────────────────────
    n = len(leaves)
    centres = np.array([(n.cx, n.cy, n.cz) for n in leaves])
    halves  = np.array([(n.hx, n.hy, n.hz) for n in leaves])

    offs = np.array([(vx * 2 - 1, vy * 2 - 1, vz * 2 - 1) for vx, vy, vz in VERTS])
    cpos = centres[:, None, :] + halves[:, None, :] * offs          # (n, 8, 3)
    cval = sdf(cpos.reshape(-1, 3)).reshape(n, 8)                   # ← SDF #1

    # Bit‑mask of inside corners per cube
    masks = np.zeros(n, dtype=np.uint16)
    for i in range(8):
        masks |= (cval[:, i] < iso).astype(np.uint16) << i

    # ── 2.  Build triangle soup (no more SDF calls here) ──────────────────────
    tri_coords, tri_eps = [], []            # collect for optional orientation
    faces, vdict        = [], {}            # final mesh
    for idx, node in enumerate(leaves):
        m = masks[idx]
        if m in (0, 0xFF):
            continue

        # Local caches are tiny and stay per leaf
        cv, cp = cval[idx], cpos[idx]
        evert = {}

        def v_on_edge(eid: int):
            if eid not in evert:
                a, b = EDGES[eid]
                evert[eid] = interpolate(cp[a], cp[b], cv[a], cv[b], iso)
            return evert[eid]

        for i in range(0, len(MC_TABLE[m]) - 1, 3):
            p0, p1, p2 = (v_on_edge(e) for e in MC_TABLE[m][i : i + 3])
            tri_coords.append((p0, p1, p2))
            tri_eps.append(1e-4 * max(node.hx, node.hy, node.hz))

    if not tri_coords:
        return {}, []

    # ── 3.  One‑shot orientation fix (optional) ───────────────────────────────
    if preserve_orientation:
        tris = np.asarray(tri_coords)                        # (T, 3, 3)
        eps  = np.asarray(tri_eps)[:, None]                  # (T, 1)

        nrm  = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
        ln   = np.linalg.norm(nrm, axis=1, keepdims=True)
        nrm  = np.where(ln == 0, nrm, nrm / ln) * eps        # unit * eps

        ctr  = tris.mean(axis=1)
        test = np.concatenate([ctr + nrm, ctr - nrm], axis=0)
        d    = sdf(test).reshape(2, -1)                      # ← SDF #2

        flip = d[0] < d[1]                                   # inward?
        for k, f in enumerate(flip):
            if f:
                p0, p1, p2 = tri_coords[k]
                tri_coords[k] = (p0, p2, p1)                 # swap

    # ── 4.  Deduplicate & index vertices ──────────────────────────────────────
    vdict=_frozendict(vdict)
    for p0, p1, p2 in tri_coords:
        p0_id=_v_id(p0, vdict,q=q)
        p1_id=_v_id(p1, vdict,q=q)
        p2_id=_v_id(p2, vdict,q=q)
        
      
        faces.append((p0_id,p1_id,p2_id))
   
  
    V = np.array([(x*q,y*q,z*q )for (x,y,z),k in sorted(vdict.items(), key=lambda kv: kv[1])] ,dtype=float)
    return V,  np.asarray(faces,dtype=int)
import functools
@functools.lru_cache(maxsize=None)
def _v_id(p, vdict, q=1e-6):
    key = (round(p[0] / q), round(p[1] / q), round(p[2] / q))
    if key not in vdict:
        vdict[key] = len(vdict)
    return vdict[key]


# -------------------------------------------------------------------------
# 6.  Write ASCII PLY (widely supported)
# -------------------------------------------------------------------------
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


# -------------------------------------------------------------------------
# 7.  Demo run
# -------------------------------------------------------------------------
def marching_cubes(sdf, bounds,min_half=1e-3,max_depth=7):
    c=(bounds[0,:]+bounds[1,:])/2
    h=bounds[1, :]-c
    root = OctreeNode(c, h)  # anisotropic root volume
    
    kept, visited ,leafs= build_sdf_octree(root, sdf,min_half=min_half, max_depth=max_depth)
    verts, faces = polygonise_leaves_bulk(leafs, sdf, q=min(1e-6,min_half))
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

    root = OctreeNode((0, 0, 0), (2, 2, 1))  # anisotropic root volume

    kept, visited ,leafs= build_sdf_octree(root, sdf_torus_vec, max_depth=7)
    print(f"Tree: visited {visited:,d}, kept {kept:,d} leaves")
    import time

    s = time.perf_counter()
    verts, faces = {}, []
    stack = [root]

    # while stack:
    #    n = stack.pop()
    #    if n.children:
    #        stack.extend(n.children)
    #    else:
    #        polygonise_leaf(n, sdf_torus_vec, verts, faces, preserve_orientation=True)
    verts, faces=polygonise_leaves_bulk(leafs,sdf_torus_vec)
    print(time.perf_counter() - s)

    print(f"Mesh: {len(verts):,d} vertices, {len(faces):,d} triangles")
    write_ply("torus.ply", verts, faces)
    print("Output written to  torus.ply  (open in MeshLab / Blender)")

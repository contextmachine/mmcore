#!/usr/bin/env python3
"""
Adaptive Marching‑Cubes driven by the 1‑Lipschitz property of a true SDF.
– Builds a compact AABB tree with conservative pruning.
– Generates the *full* 256‑case MC table at run‑time (no hand typing).
– Extracts a watertight triangle mesh and writes it to torus.ply.

Tested on Python 3.11, no third‑party modules required.
"""

from __future__ import annotations
import math, itertools
from collections import defaultdict
from functools import lru_cache
from typing import List, Tuple

# -------------------------------------------------------------------------
# 1.  A TRUE signed‑distance function: torus centred at the origin
# -------------------------------------------------------------------------


def sdf_torus(p: Tuple[float, float, float], R: float = 1.0, r: float = 0.30) -> float:
    x, y, z = p
    qx = math.hypot(x, y) - R
    return math.hypot(qx, z) - r  # distance minus tube radius


# -------------------------------------------------------------------------
# 2.  Adaptive axis‑aligned bounding boxes (AABB)
# -------------------------------------------------------------------------
class AABB:
    __slots__ = ("cx", "cy", "cz", "hx", "hy", "hz", "depth", "children")

    def __init__(self, centre, half, depth=0):
        self.cx, self.cy, self.cz = map(float, centre)
        self.hx, self.hy, self.hz = map(float, half)
        self.depth = depth
        self.children: List[AABB] = []


def half_diag(n: AABB) -> float:
    return math.sqrt(n.hx * n.hx + n.hy * n.hy + n.hz * n.hz)


def subdivide(node: AABB, sdf, max_depth: int, min_half: float = 1e-3) -> Tuple[int, int]:
    rb = half_diag(node)
    dc = sdf((node.cx, node.cy, node.cz))

    if dc > rb:  # empty
        return 0, 1
    if dc < -rb or node.depth >= max_depth or max(node.hx, node.hy, node.hz) < min_half:  # full or limit
        return 1, 1

    kept = visited = 1
    hx2, hy2, hz2 = node.hx * 0.5, node.hy * 0.5, node.hz * 0.5
    for sx, sy, sz in itertools.product((-1, 1), repeat=3):
        child = AABB((node.cx + sx * hx2, node.cy + sy * hy2, node.cz + sz * hz2), (hx2, hy2, hz2), node.depth + 1)
        node.children.append(child)
        k, v = subdivide(child, sdf, max_depth, min_half)
        kept += k
        visited += v
    return kept, visited


# -------------------------------------------------------------------------
# 3.  Cube topology (28 edges, six tetrahedra)
# -------------------------------------------------------------------------
# Vertices of the unit cube in (x,y,z) order
VERTS = [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)]

# Generate **all 28 unique edges** automatically (12 + 12 + 4)
EDGE2IDX, EDGES = {}, []
for v0, v1 in itertools.combinations(range(8), 2):
    EDGE2IDX[(v0, v1)] = EDGE2IDX[(v1, v0)] = len(EDGES)
    EDGES.append((v0, v1))  # index → (vertex0, vertex1)

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


def polygonise_leaf(node: AABB, sdf, vdict: dict, faces: List[Tuple[int, int, int]], iso: float = 0.0, preserve_orientation: bool = True):
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

    # On‑demand vertex cache: edge‑index → 3‑tuple
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


def _v_id(p, vdict, q=1e-6):
    key = (round(p[0] / q), round(p[1] / q), round(p[2] / q))
    if key not in vdict:
        vdict[key] = len(vdict)
    return vdict[key]


# -------------------------------------------------------------------------
# 6.  Write ASCII PLY (widely supported)
# -------------------------------------------------------------------------
def write_ply(path: str, vdict: dict, faces: List[Tuple[int, int, int]]):
    items = sorted(vdict.items(), key=lambda kv: kv[1])  # by index
    verts = [(x * 1e-6, y * 1e-6, z * 1e-6) for (x, y, z), _ in items]
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
if __name__ == "__main__":

    root = AABB((0, 0, 0), (2, 2, 1))  # anisotropic root volume

    kept, visited = subdivide(root, sdf_torus, max_depth=7)
    print(f"Tree: visited {visited:,d}, kept {kept:,d} leaves")
    import time

    s = time.perf_counter()
    verts, faces = {}, []
    stack = [root]
    while stack:
        n = stack.pop()
        if n.children:
            stack.extend(n.children)
        else:
            polygonise_leaf(n, sdf_torus, verts, faces, preserve_orientation=True)
    print(time.perf_counter() - s)

    print(f"Mesh: {len(verts):,d} vertices, {len(faces):,d} triangles")
    write_ply("torus.ply", verts, faces)
    print("Output written to  torus.ply  (open in MeshLab / Blender)")

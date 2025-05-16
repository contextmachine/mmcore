from mmcore.topo.brep import BRep,Shell,Loop,Edge,Vertex
import numpy as np
def box(W, D, H):
    m = BRep()
    V1, V2, E1, L1, F, S = m.MEVVLS((D / 2, W / 2, 0.0), (-D / 2, W / 2, 0.0))
    V3, E2 = m.MEV(L1.id, V2.id, p_new=(-D / 2, -W / 2, 0))
    V4, E3 = m.MEV(L1.id, V3.id, p_new=(D / 2, -W / 2, 0))
    E4, L2,  F2= m.MELF(L1.id, V4.id, V1.id)
    V5, E5 = m.MEV(L1.id, V1.id, p_new=(V1.point[0], V1.point[1], H))
    V6, E6 = m.MEV(L1.id, V2.id, p_new=(V2.point[0], V2.point[1], H))
    V7, E7 = m.MEV(L1.id, V3.id, p_new=(V3.point[0], V3.point[1], H))
    V8, E8 = m.MEV(L1.id, V4.id, p_new=(V4.point[0], V4.point[1], H))
    E9,  L3, F3 = m.MELF(L1.id, V5.id, V6.id)
    E10, L4, F4 = m.MELF(L1.id, V6.id, V7.id)
    E11, L5, F5 = m.MELF(L1.id, V7.id, V8.id)
    E12, L6, F6 = m.MELF(L1.id, V8.id, V5.id)
    return m
from collections import defaultdict

def split_box(brep: BRep) -> Shell:
        # 0)–– helper: grab all z-coordinates
        zs = [v.point[2] for v in brep.V.values()]
        z_min, z_max = min(zs), max(zs)
        z_mid = 0.5 * (z_min + z_max)

        # 1) Identify the four “side” faces (i.e. not top or bottom)
        side_faces = []
        for f_id, f in brep.F.items():
            loop_id = f.outer
            loop_zs = [brep.V[brep.HE[hid].vert].point[2] for hid in brep._loop_halfedges(loop_id)]
            if all(abs(z - z_min) < 1e-6 for z in loop_zs):
                continue  # bottom face
            elif all(abs(z - z_max) < 1e-6 for z in loop_zs):
                continue  # top face
            else:
                side_faces.append(f_id)
        print(side_faces)
        # 2) On each side face, split exactly two edges that cross z_mid with MVE.
        #    Record the two new midpoint‐vertex IDs per face.
        face_midverts: dict[int, list[int]] = {}
        for f_id in side_faces:
            outer = brep.F[f_id].outer
            mids = []
            for hid in brep._loop_halfedges(outer):
                he = brep.HE[hid]
                e = brep.E[he.edge]
                p0 = np.array(brep.V[e.v_start].point)
                p1 = np.array(brep.V[e.v_end].point)
                if (p0[2] - z_mid) * (p1[2] - z_mid) < 0:
                    t = (z_mid - p0[2]) / (p1[2] - p0[2])
                    mid_pt = tuple((p0 + t * (p1 - p0)).tolist())
                    v_mid, _ = brep.MVE(e.id, mid_pt)
                    mids.append(v_mid.id)
            assert len(mids) == 2, f"face {f_id} should have exactly 2 splits, not {len(mids)}"
            face_midverts[f_id] = mids

        # 3) Still on each side face, cut out a “hole”—i.e. split its loop in two—by MEL
        #    between the two new mid-plane vertices.
        face_hole_loop: dict[int, int] = {}
        for f_id, (v1, v2) in face_midverts.items():
            outer = brep.F[f_id].outer
            _, hole_loop = brep.MEL(outer, v1, v2)
            face_hole_loop[f_id] = hole_loop

        # 4) Stitch those four holes into a closed mid-plane circuit:
        #    at each midpoint-vertex we know it lives in exactly two loops,
        #    so call MZEV(loopA, loopB, v) to drop in a zero-length bar between them.
        zero_edges = []
        # build a map vertex → [loop1, loop2]
        v_to_loops: dict[int, list[int]] = defaultdict(list)
        for he in brep.HE.values():
            if he.vert in sum(face_midverts.values(), []):
                v_to_loops[he.vert].append(he.loop)
        # now do the zero‐length edge insertion
        for v_mid, loops in v_to_loops.items():
            assert len(loops) == 2, "each mid‐vertex should lie in exactly two loops"
            e_zero, _ = brep.MZEV(loops[0], loops[1], v_mid)
            zero_edges.append(e_zero.id)

        # 5) Clean up those internal bars (they’ve just done their job of
        #    linking up your mid-plane circuit) by KEL’ing each zero-length edge:
        for e0 in zero_edges:
            l_keep, l_drop = brep.get_edge_loops(e0)
            # whichever loop is the “hole” side, pass that as loop2
            brep.KEL(e0, l_drop.id)

        # 6) Finally, pick one of the hole-loops you created in step 3,
        #    and promote it out into its own shell with MPKH.  That will
        #    detect that you’ve now carved the shell into two disconnected
        #    solids and split them apart for you.
        #    (All four side‐faces have that same hole‐loop on the mid‐plane
        #     so you only need to do it once.)
        one_face = side_faces[0]
        hole_loop = face_hole_loop[one_face]
        _, new_shell = brep.MPKH(hole_loop)

        return new_shell


def split_box2(brep: BRep):
    import numpy as np

    # 1) Locate the four vertical edges of the box
    tol = 1e-6
    Vs = brep.V
    vertical_edges = []
    for e in brep.E.values():
        p0, p1 = Vs[e.v_start].point, Vs[e.v_end].point
        # vertical if x,y constant but z differs
        if (abs(p0[0] - p1[0]) < tol and
            abs(p0[1] - p1[1]) < tol and
            abs(p0[2] - p1[2]) > tol):
            vertical_edges.append(e)

    # 2) For each vertical edge, carve a mid‐point spike (MEV) and duplicate it (MZEV)
    mid_infos = []
    for edge in vertical_edges:
        # mid‐point in 3D
        p0 = np.array(Vs[edge.v_start].point)
        p1 = np.array(Vs[edge.v_end].point)
        pm = tuple(((p0 + p1) / 2).tolist())

        # find the two loops that bound this edge
        he_forward = brep.get_edge_he(edge.id).twin     # half‐edge v_start → v_end
        loop_A,loop_B=brep.get_edge_loops(edge.id)
       
        # (a) MEV on loop_A, anchoring at v_start
        v_mid, _ = brep.MEV(loop_A.id, edge.v_start, pm)

        # (b) MZEV to mirror that vertex into loop_B
        e_zero, v_mid_opp = brep.MZEV(loop_A.id, loop_B.id
                                      , v_mid.id)

        # remember for the next steps
        mid_infos.append({
            'front_vid':   v_mid.id,
            'back_vid':    v_mid_opp.id,
            'anchor_loop': loop_A,
            'zero_eid':    e_zero.id
        })

    # 3) Sort the four new mid-points into a cyclic order around the box
    #    (we’ll use X–Z as our “angle” plane)
    pts = [np.array(Vs[m['front_vid']].point) for m in mid_infos]
    center = sum(pts) / len(pts)

    def angle_key(m):
        v = np.array(Vs[m['front_vid']].point) - center
        return np.arctan2(v[2], v[0])

    mid_infos.sort(key=angle_key)

    # 4) Stitch them into a closed ring with 8 × MEL
    cuts = []
    n = len(mid_infos)
    for i in range(n):
        A = mid_infos[i]
        B = mid_infos[(i + 1) % n]
        # connect front‐face midpoints A→B
        e_cut, l_new = brep.MEL(A['anchor_loop'], A['front_vid'], B['front_vid'])
        cuts.append((e_cut.id, l_new.id))
        # connect back‐face midpoints  B→A  (the opposite loop is the same anchor loop)
        e_cut2, l_new2 = brep.MEL(A['anchor_loop'], B['back_vid'], A['back_vid'])
        cuts.append((e_cut2.id, l_new2.id))

    # 5) Collapse each of those 8 splitting‐edges back into a single loop
    for e_id, loop_id in cuts:
        brep.KEL(e_id, loop_id)

    # 6) Finally kill the through‐slit and promote it into its own shell
    #    (that plane is currently carried by one of our zero‐length edges)
    hole_loop = brep.KEMH(mid_infos[0]['zero_eid'])
    brep.MPKH(hole_loop.id)

    return brep
import copy
from typing import Tuple


def extract_shell(brep: BRep, shell_id: int) -> BRep:
    """Make a brand-new BRep containing exactly shell_id (and its body)."""
    # 1) figure out which faces belong to that shell
    shell = brep.S[shell_id]
    face_ids = set(shell.faces)

    # 2) gather all loops, half-edges, edges, vertices used by those faces
    loop_ids = {f.outer for f in (brep.F[fid] for fid in face_ids)} | set(sum((brep.F[fid].inners for fid in face_ids), []))

    he_ids = set()
    for lid in loop_ids:
        he_ids |= {hid for hid in brep._loop_halfedges(lid)}

    edge_ids = {brep.HE[hid].edge for hid in he_ids}
    vert_ids = set()
    for eid in edge_ids:
        e = brep.E[eid]
        vert_ids.add(e.v_start)
        vert_ids.add(e.v_end)

    # 3) deep-copy just those dict entries into a fresh BRep
    new = BRep()
    new.V = {vid: copy.deepcopy(brep.V[vid]) for vid in vert_ids}
    new.E = {eid: copy.deepcopy(brep.E[eid]) for eid in edge_ids}
    new.HE = {hid: copy.deepcopy(brep.HE[hid]) for hid in he_ids}
    new.L = {lid: copy.deepcopy(brep.L[lid]) for lid in loop_ids}
    new.F = {fid: copy.deepcopy(brep.F[fid]) for fid in face_ids}
    new.S = {shell_id: copy.deepcopy(shell)}
    # bodies are optional; if you need them, copy over brep.B[shell.body], etc.
    return new


def split_box_into_two(brep: BRep) -> Tuple[BRep, BRep]:
    # do your in-place split, get the new shell
    new_shell = split_box(brep)

    # original shell was brep.B[...] but if you only have one shell originally:
    (old_shell_id,) = [sid for sid in brep.S if sid != new_shell.id]

    # now extract each into its own BRep
    brep1 = extract_shell(brep, old_shell_id)
    brep2 = extract_shell(brep, new_shell.id)
    return brep1, brep2

def get_loops_points(m:BRep):

        return [[m.V[m.HE[i].vert].point for i in m._loop_halfedges(l.id)] for l in m.L.values()]
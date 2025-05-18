"""brep_primitives.py – B‑rep skeleton + Euler operators
========================================================
Data structures compatible with Parasolid/ACIS/CGM/OpenCascade/BMesh


"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Dict, List, Optional, Tuple, Any

import numpy as np


# ---------------------------------------------------------------------------
#  Minimal geometry placeholders (replace with real NURBS later)
# ---------------------------------------------------------------------------
class Curve3D:  # straight‑line stub
    pass


class Curve2D: ...


class Surface: ...


_V_AUTOID=itertools.count()
_E_AUTOID=itertools.count()
_HE_AUTOID=itertools.count()
_L_AUTOID=itertools.count()
_F_AUTOID=itertools.count()
_S_AUTOID=itertools.count()
_B_AUTOID=itertools.count()
# ---------------------------------------------------------------------------
#  Topological dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Vertex:
    point: Tuple[float, float, float]
    tol: float = 1e-6
    id: int = field(default_factory=lambda :next(_V_AUTOID), init=False)


@dataclass
class Edge:
    v_start: int
    v_end: int
    geom: Curve3D
    param: Tuple[float, float]
    id: int = field(default_factory=lambda :next(_E_AUTOID), init=False)


@dataclass
class HalfEdge:
    edge: int
    face: Optional[int]
    loop: Optional[int]
    next: Optional[int] = None
    prev: Optional[int] = None
    twin: Optional[int] = None
    vert: Optional[int] = None  # head vertex
    orient: bool = True
    pcurve: Optional[Curve2D] = None
    id: int = field(default_factory=lambda :next(_HE_AUTOID), init=False)


@dataclass
class Loop:
    face: Optional[int]
    he: int  # entry half‑edge id
    is_outer: bool = True
    id: int = field(default_factory=lambda :next(_L_AUTOID), init=False)


@dataclass
class Face:

    outer: int
    inners: List[int] = field(default_factory=list)
    shell: Optional[int] = None
    same_sense: bool = True
    surf: Optional[Surface]=None
    id: int = field(default_factory=lambda :next(_F_AUTOID), init=False)


@dataclass
class Shell:
    faces: List[int]
    body: Optional[int] = None
    closed: bool = False
    id: int = field(default_factory=lambda :next(_S_AUTOID), init=False)


@dataclass
class Body:
    shells: List[int]
    lump_type: str = "solid"
    attributes: Dict[str, Any] = field(default_factory=dict)
    id: int = field(default_factory=lambda :next(_S_AUTOID),  init=False)



# ---------------------------------------------------------------------------
#  Model container + Euler operators
# ---------------------------------------------------------------------------
@dataclass
class BRep:
    V: Dict[int, Vertex] = field(default_factory=dict)
    E: Dict[int, Edge] = field(default_factory=dict)
    HE: Dict[int, HalfEdge] = field(default_factory=dict)
    L: Dict[int, Loop] = field(default_factory=dict)
    F: Dict[int, Face] = field(default_factory=dict)
    S: Dict[int, Shell] = field(default_factory=dict)
    B: Dict[int, Body] = field(default_factory=dict)

    # ************************************************************
    #  Helper utilities
    # ************************************************************
    def _loop_halfedges(self, loop_id: int):
        """Generator over half‑edge IDs in CCW order."""
        start = self.L[loop_id].he
        he_id = start
        while True:
            yield he_id

            he_id = self.HE[he_id].next
            if he_id == start:
                break
    def _cycle_halfedges(self, start_he:int):
        """Generator over half‑edge IDs in CCW order."""
        start = start_he
        he_id = start
        while True:

            yield he_id
            # print(self.HE[he_id])
            he_id = self.HE[he_id].next
            if he_id == start:
                break
    def _edge_single_use(self, edge_id: int, he_set: set[int]):
        """Return True if *edge* is referenced only by given halfedges."""
        return all((he_id in he_set) for he_id, he in self.HE.items() if he.edge == edge_id)

    def summary(self) -> str:
        return (
            f"|V|={len(self.V)} |E|={len(self.E)} |HE|={len(self.HE)} |L|={len(self.L)} "
            f"|F|={len(self.F)} |S|={len(self.S)} |B|={len(self.B)}"
        )

    # ============================================================
    #  MEVVLS / KEVVLS  (unchanged from previous revision)
    # ============================================================
    def MEVVLS(self, p1: Tuple[float, float, float], p2: Tuple[float, float, float]) -> tuple[Vertex, Vertex, Edge, Loop, Face, Shell]:
        v1 = Vertex(p1)
        v2 = Vertex(p2)
        self.V[v1.id] = v1
        self.V[v2.id] = v2
        e = Edge(v1.id, v2.id, Curve3D(), (0.0, 1.0))
        self.E[e.id] = e
        he_fwd = HalfEdge(e.id, None, None, vert=v2.id, orient=True)
        he_rev = HalfEdge(e.id, None, None, vert=v1.id, orient=True)

        he_fwd.twin, he_rev.twin = he_rev.id, he_fwd.id
        he_fwd.next = he_rev.id
        he_rev.prev = he_fwd.id
        he_rev.next = he_fwd.id
        he_fwd.prev = he_rev.id
        self.HE[he_fwd.id] = he_fwd
        self.HE[he_rev.id] = he_rev
        loop = Loop(None, he_fwd.id, is_outer=True)
        self.L[loop.id] = loop
        he_fwd.loop = he_rev.loop = loop.id
        face = Face( loop.id)
        self.F[face.id] = face
        loop.face = face.id
        he_fwd.face = he_rev.face = face.id
        shell = Shell([face.id])
        self.S[shell.id] = shell
        face.shell = shell.id
        return v1, v2, e, loop, face, shell
    def he_points(self, he:HalfEdge):
        return [self.V[he.vert].point   , self.V[self.HE[he.twin].vert].point]
    def edge_points(self, edge:Edge):
        return [self.V[edge.v_start].point  ,self.V[edge.v_end].point]
    def loop_points(self, loop:Loop):
        return [self.V[self.HE[i].vert].point for i in self._loop_halfedges(loop.id)]

    def _shift_loop_to_vertex(self, l: Loop, v: Vertex) -> tuple[bool, HalfEdge]:
        start: int = l.he
        current = l.he
        while True:

            he = self.HE[current]
            # print(he)
            if he.vert == v.id:
                return True, he
            current = he.next

            if current == start:
                return False, he

    # ============================================================
    #  MEV – Make Edge & Vertex, inside loop L, from existing vertex v_from
    # ============================================================
    def _walk_to_vertex(self, he: HalfEdge, v: Vertex) -> tuple[list[HalfEdge], HalfEdge]:

        start: int = he.id
        current = start
        lst = []
        while True:

            h = self.HE[current]
            # print(h)
            if h.vert == v.id:

                return lst, h
            lst.append(h)
            current = h.next

            if current == start:
                loop=self.L[he.loop]
                raise ValueError(f"v: {v} not in Loop: {loop}, {list( self.HE[he_id].vert for he_id in self._loop_halfedges(he.loop))}")

    # ============================================================
    #  KEV – inverse of above (remove dangling vertex + edge from loop)
    # ============================================================
    def new_halfedge(
        self,
        edge: int,
        face: Optional[int],
        loop: Optional[int],
        next: Optional[int] = None,
        prev: Optional[int] = None,
        twin: Optional[int] = None,
        vert: Optional[int] = None,
        orient: bool = True,
        pcurve: Optional[Curve2D] = None,
    ) -> HalfEdge:
        he = HalfEdge(edge=edge, loop=loop, face=face, next=next, prev=prev, twin=twin, vert=vert, orient=orient, pcurve=pcurve)
        self.HE[he.id] = he
        return he

    def new_loop(self, face: Optional[int], he: int, is_outer: bool = True) -> Loop:  # entry half‑edge id
        l = Loop(face=face, he=he, is_outer=is_outer)

        self.L[l.id] = l
        return l

    def new_face(self,  outer: int, inners: List[int] = None, shell: Optional[int] = None, same_sense: bool = True,surf: Optional[Surface]=None) -> Face:
        if inners is None:
            inners = []
        f = Face(surf=surf, outer=outer, inners=inners, shell=shell, same_sense=same_sense)
        self.F[f.id] = f
        return f

    def new_edge(self, v_start: int, v_end: int, geom: Curve3D, param: Tuple[float, float]) -> Edge:
        e = Edge(v_start=v_start, v_end=v_end, geom=geom, param=param)
        self.E[e.id] = e
        return e

    def new_body(self, shells: List[int], lump_type: str = "solid", attributes: Dict[str, Any] = None) -> Body:
        if attributes is None:
            attributes = {}
        b = Body(shells=shells, lump_type=lump_type, attributes=attributes)
        self.B[b.id] = b
        return b

    def new_shell(self, faces: List[int], body: Optional[int] = None, closed: bool = False) -> Shell:
        s = Shell(faces=faces, body=body, closed=closed)
        self.S[s.id] = s
        return s

    def new_vertex(self, point: Tuple[float, float, float], tol: float = 1e-6) -> Vertex:
        v = Vertex(point=point, tol=tol)
        self.V[v.id] = v
        return v
    def _edge_split(self, edge_id: int, v_new: int) -> tuple[Edge, HalfEdge, HalfEdge]:
        """
        Split an edge *edge_id* into two edges, one going from its start to
        *v_new* and the other from *v_new* to its end.
        """
        E=self.E[edge_id]
        self.new_edge(E.v_start, v_new, E.geom, E.param)
        E.v_start=v_new

    def get_loop_first_vertex(self, loop: Loop)->Vertex:
        return self.V[self.HE[self.HE[loop.he].twin].vert]

    def KEVVLS(self, shell_id: int):
        if shell_id not in self.S:
            raise KeyError("Shell not found")
        shell = self.S[shell_id]
        if len(shell.faces) != 1:
            raise ValueError("shell not produced by MEVVLS")
        face_id = shell.faces[0]
        face = self.F[face_id]
        loop_id = face.outer
        loop = self.L[loop_id]
        he_ids = list(self._loop_halfedges(loop_id))
        if len(he_ids) != 2:
            raise ValueError("Unexpected loop length for MEVVLS")
        edge_id = self.HE[he_ids[0]].edge
        v_ids = [self.E[edge_id].v_start, self.E[edge_id].v_end]
        # --- deletions ---
        del self.S[shell_id]
        del self.F[face_id]
        del self.L[loop_id]
        for hid in he_ids:
            del self.HE[hid]
        del self.E[edge_id]
        for vid in v_ids:
            del self.V[vid]

    # ---------------------------------------------------------------------------
    #  MEL – Make-Edge-Loop  (split one loop into two)
    # ---------------------------------------------------------------------------
    def MEL(self, loop_id: int, v1_id: int, v2_id: int) -> tuple[Edge, Loop]:
        """
        Insert an edge between *v1* and *v2*, both on the same loop *loop_id*,
        thereby splitting that loop into two:

            ΔE = +1     ΔL = +1     (no face is created here)

        Returns
        -------
        (Edge, Loop)
            The newly created *Edge* record and the newly created *Loop*
            (the one whose entry half-edge runs CCW from v2 → … → v1).
        """
        # ----------  sanity checks  ------------------------------------------------
        if loop_id not in self.L:
            raise KeyError("Loop not found")
        if v1_id == v2_id:
            raise ValueError("Vertices must be distinct")
        if v1_id not in self.V or v2_id not in self.V:
            raise KeyError("Vertex not found")

        loop1 = self.L[loop_id]
        face_id = loop1.face

        # ----------  locate reference half-edges  ---------------------------------
        he_v1 = None
        for hid in self._loop_halfedges(loop_id):
            if self.HE[hid].vert == v1_id:
                he_v1 = self.HE[hid]
                break
        if he_v1 is None:
            raise ValueError("v1 not on the given loop")

        # walk CCW to find the half-edge whose *head* is v2
        tmp, he_v2 = self._walk_to_vertex(he_v1, self.V[v2_id])

        # adjacent vertices would give a degenerate split
        ##    raise ValueError("Vertices are adjacent – cannot split loop")

        he_v1_next = he_v1.next  # cache before we touch it
        he_v2_next = he_v2.next

        # ----------  create new edge + twin half-edges  ---------------------------
        e_new = self.new_edge(v1_id, v2_id, Curve3D(), (0.0, 1.0))

        he_uv = self.new_halfedge(e_new.id, face=face_id, loop=loop_id, vert=v2_id, orient=True)  # v1 → v2  (stays in loop-1)

        he_vu = self.new_halfedge(e_new.id, face=face_id, loop=None, vert=v1_id, orient=False)  # v2 → v1  (will belong to loop-2)

        he_uv.twin = he_vu.id
        he_vu.twin = he_uv.id

        # ----------  splice half-edges for loop-1  --------------------------------
        # …v1_prev → he_v1 → he_uv → he_v2_next…
        self.HE[he_v1.id].next = he_uv.id
        he_uv.prev = he_v1.id
        he_uv.next = he_v2_next
        self.HE[he_v2_next].prev = he_uv.id

        # ----------  splice half-edges for prospective loop-2  --------------------
        # …v2 → he_vu → he_v1_next…
        self.HE[he_v2.id].next = he_vu.id
        he_vu.prev = he_v2.id
        he_vu.next = he_v1_next
        self.HE[he_v1_next].prev = he_vu.id

        # ----------  construct the second loop & retag its edges ------------------
        loop2 = self.new_loop(face=face_id, he=he_vu.id, is_outer=loop1.is_outer)
        he_vu.loop = loop2.id

        cur = he_vu.next
        while cur != he_vu.id:  # walk until we close
            self.HE[cur].loop = loop2.id
            cur = self.HE[cur].next
        if self.HE[loop1.he].loop != loop_id:
            # the old anchor has wandered into loop2,
            # so point it at he_uv (which we know is in loop1)
            loop1.he = he_uv.id
        return e_new, loop2

    # ---------------------------------------------------------------------------
    #  KEL – Kill-Edge-Loop  (merge two sister loops back into one)
    # ---------------------------------------------------------------------------
    def KEL(self, edge_id: int, loop2_id: int) -> int:
        """
        Remove *edge_id* (whose two sides bound loops *loop_keep* and *loop2_id*),
        delete *loop2_id*, and merge everything into *loop_keep*.

            ΔE = −1     ΔL = −1

        Returns
        -------
        int
            The ID of the surviving loop (useful when chaining operators).
        """
        # ----------  sanity checks  ------------------------------------------------
        if edge_id not in self.E:
            raise KeyError("Edge not found")
        if loop2_id not in self.L:
            raise KeyError("Loop not found")

        # grab the two half-edges that realise *edge_id*
        he_list = [he for he in self.HE.values() if he.edge == edge_id]
        if len(he_list) != 2:
            raise ValueError("Edge should be referenced by exactly two half-edges")

        he_a, he_b = he_list
        if he_a.loop == loop2_id and he_b.loop != loop2_id:
            he_loop2, he_keep = he_a, he_b
        elif he_b.loop == loop2_id and he_a.loop != loop2_id:
            he_loop2, he_keep = he_b, he_a
        else:
            raise ValueError("Edge does not separate the specified loops")

        loop_keep_id = he_keep.loop

        # neighbours next/prev around the edge ends
        p_keep = self.HE[he_keep.prev]
        n_keep = self.HE[he_keep.next]
        p_del = self.HE[he_loop2.prev]
        n_del = self.HE[he_loop2.next]

        # ----------  stitch the rings together  -----------------------------------
        # connect p_keep → n_del   and   p_del → n_keep
        p_keep.next = n_del.id
        n_del.prev = p_keep.id

        p_del.next = n_keep.id
        n_keep.prev = p_del.id

        # ----------  retag former loop-2 edges to loop-keep  ----------------------
        cur = n_del.id
        while True:
            self.HE[cur].loop = loop_keep_id
            cur = self.HE[cur].next
            if cur == n_del.id:
                break

        # ----------  update anchor if necessary  ----------------------------------
        loop_keep = self.L[loop_keep_id]
        if loop_keep.he in (he_keep.id, he_loop2.id):
            loop_keep.he = n_keep.id  # any HE on the merged ring is fine

        # ----------  destroy topological records  ---------------------------------
        del self.HE[he_keep.id]
        del self.HE[he_loop2.id]
        del self.E[edge_id]
        del self.L[loop2_id]

        return loop_keep_id

    # ============================================================
    #  MEV – Make-Edge-and-Vertex  (v_from ➜ v_new inside loop L)
    # ============================================================
    def MEV(
        self,
        loop_id: int,
        v_from: int,
        p_new: Tuple[float, float, float],
    ) -> tuple[Vertex, Edge]:
        """
        Insert a *dangling* edge (v_from → v_new) inside outer loop *loop_id*.

        Returns
        -------
        (Vertex, Edge)
            The brand-new vertex V_new and the edge E_new that connects it
            to the existing vertex v_from.
        """
        # ---------- look-ups & validations ----------
        if loop_id not in self.L:
            raise KeyError("Loop not found")
        if v_from not in self.V:
            raise KeyError("Start vertex not found")

        loop = self.L[loop_id]

        # half-edge whose *head* is v_from (⇒ last edge before the new one)
        he_prev_id = next(
            (hid for hid in self._loop_halfedges(loop_id) if self.HE[hid].vert == v_from),
            None,
        )
        if he_prev_id is None:
            raise ValueError("v_from is not on the supplied loop")

        he_prev = self.HE[he_prev_id]  # … → v_from
        he_next = self.HE[he_prev.next]  # v_from → …

        # ---------- create topological entities ----------
        v_new = self.new_vertex(p_new)
        e_new = self.new_edge(v_from, v_new.id, Curve3D(), (0.0, 1.0))

        # half-edge v_from → v_new  (heads at v_new)
        he_fwd = self.new_halfedge(
            edge=e_new.id,
            face=he_prev.face,
            loop=loop_id,
            prev=he_prev_id,
            next=None,  # wired below
            twin=None,  # wired below
            vert=v_new.id,
            orient=True,
        )
        # half-edge v_new → v_from  (heads back at v_from)
        he_rev = self.new_halfedge(
            edge=e_new.id,
            face=he_prev.face,
            loop=loop_id,
            prev=he_fwd.id,
            next=he_next.id,
            twin=he_fwd.id,  # temporary – completed after creation
            vert=v_from,
            orient=False,
        )
        he_fwd.twin = he_rev.id

        # ---------- splice into the loop ----------
        he_fwd.next = he_rev.id

        he_prev.next = he_fwd.id
        he_next.prev = he_rev.id

        return v_new, e_new

    # ============================================================
    #  KEV – Kill-Edge-and-Vertex  (inverse of MEV)
    # ============================================================
    def KEV(self, loop_id: int, v_id: int) -> None:
        """
        Delete a *dangling* vertex V and its incident edge.
        Preconditions
        ------------
        * V must have degree 1 (exactly one incident edge).
        * That edge must lie on `loop_id`.
        """
        if loop_id not in self.L:
            raise KeyError("Loop not found")
        if v_id not in self.V:
            raise KeyError("Vertex not found")

        # two half-edges that use the dangling edge
        he_in_id = next(
            (hid for hid in self._loop_halfedges(loop_id) if self.HE[hid].vert == v_id),
            None,
        )
        if he_in_id is None:
            raise ValueError("Vertex not on the supplied loop")

        he_in = self.HE[he_in_id]  # … → V
        he_out = self.HE[he_in.prev]  # V → …

        # sanity check: edge used only by these two half-edges
        edge_id = he_in.edge
        if not self._edge_single_use(edge_id, {he_in_id, he_out.id}):
            raise ValueError("Vertex is not of degree 1")

        # bypass the dangling pair
        self.HE[he_out.prev].next = he_in.next
        self.HE[he_in.next].prev = he_out.prev

        # scrub data-base
        del self.HE[he_in_id]
        del self.HE[he_out.id]
        del self.E[edge_id]
        del self.V[v_id]

    def MVE(
        self,
        edge_id: int,
        point_new: Tuple[float, float, float],
    ) -> tuple[Vertex, Edge]:
        """
        Split edge `edge_id` at `point_new`.
          - The original edge E1 is shortened to (v_start → V_new).
          - The new edge E2 spans (V_new → old_v_end).
        Both half-edges of E2 are spliced into the same loop, and
        all .next/.prev/.twin invariants are preserved.
        Returns (V_new, E2).
        """
        # 1) look up the original edge + its two half-edges
        if edge_id not in self.E:
            raise KeyError("Edge not found")
        E1 = self.E[edge_id]
        v_start, v_old_end = E1.v_start, E1.v_end

        hes = [he for he in self.HE.values() if he.edge == edge_id]
        if len(hes) != 2:
            raise ValueError("Edge should have exactly two half-edges")

        # identify which half-edge currently “points to” the old end
        he_fwd = next(he for he in hes if he.vert == v_old_end)
        he_rev = self.HE[he_fwd.twin]

        loop_id = he_fwd.loop
        if loop_id is None:
            raise ValueError("Edge is not part of any loop")

        face_id = he_fwd.face

        # 2) create the new vertex and the new edge
        V_new = self.new_vertex(point_new)
        E2 = self.new_edge(V_new.id, v_old_end, E1.geom, E1.param)

        # 3) shorten E1 so that it now ends at V_new
        E1.v_end = V_new.id
        he_fwd.vert = V_new.id

        # 4) create the two new half-edges for E2 (forward + reverse)
        he2_fwd = self.new_halfedge(
            edge=E2.id,
            face=face_id,
            loop=loop_id,
            vert=v_old_end,
            orient=True,
        )
        he2_rev = self.new_halfedge(
            edge=E2.id,
            face=face_id,
            loop=loop_id,
            vert=V_new.id,
            orient=False,
        )
        he2_fwd.twin = he2_rev.id
        he2_rev.twin = he2_fwd.id

        # 5) splice he2_fwd in immediately after he_fwd
        old_next = he_fwd.next
        he_fwd.next = he2_fwd.id
        he2_fwd.prev = he_fwd.id
        he2_fwd.next = old_next
        self.HE[old_next].prev = he2_fwd.id

        # 6) splice he2_rev in immediately before he_rev  ← fixed
        old_prev2 = he_rev.prev
        self.HE[old_prev2].next = he2_rev.id
        he2_rev.prev           = old_prev2
        he2_rev.next           = he_rev.id
        he_rev.prev            = he2_rev.id

        return V_new, E2

    def KVE(self, edge_id: int, v_id: int) -> None:
        """
        Undo a previous MVE: merge the two edges incident to v_id back into
        a single edge edge_id and delete v_id.  Always leaves loop.he pointing
        at a valid half-edge.
        """
        # --- sanity checks --------------------------------------------------------
        if edge_id not in self.E:
            raise KeyError("Edge not found")
        if v_id not in self.V:
            raise KeyError("Vertex not found")

        # --- find the two half-edges that meet at v_id --------------------------
        incident_hes = [he for he in self.HE.values() if he.vert == v_id]
        if len(incident_hes) != 2:
            raise ValueError("Vertex degree is not 2")

        he1, he2 = incident_hes
        # identify which one is on the "other" edge we inserted
        other_e_id = he2.edge if he1.edge == edge_id else he1.edge

        # --- grab both half-edges of that other edge ----------------------------
        other_hes = [he for he in self.HE.values() if he.edge == other_e_id]
        if len(other_hes) != 2:
            raise RuntimeError("Corrupted edge data")

        # they both live in the same loop
        loop_id = other_hes[0].loop
        loop = self.L[loop_id]

        # --- 1) splice out the two half-edges of other_e_id ---------------------
        for he in other_hes:
            self.HE[he.prev].next = he.next
            self.HE[he.next].prev = he.prev

        # --- 2) retarget the surviving edge to span the two original verts -------
        E1 = self.E[edge_id]
        # v_keep is the original start/end that isn't v_id
        v_keep = E1.v_start if E1.v_start != v_id else E1.v_end
        # v_other is the far end of the other edge
        v_other = self.E[other_e_id].v_start if self.E[other_e_id].v_start != v_id else self.E[other_e_id].v_end
        if E1.v_start == v_id:
            E1.v_start = v_other
        else:
            E1.v_end = v_other

        # fix the vert-heads on the two remaining half-edges of E1
        for he in (h for h in self.HE.values() if h.edge == edge_id):
            if he.vert == v_id:
                he.vert = v_other

        # --- 3) ensure loop.he isn’t pointing at a deleted half-edge ----------
        dead_ids = {he.id for he in other_hes}
        if loop.he in dead_ids:
            # pick the one half-edge at v_id that belongs to E1
            surviving_he = he1 if he1.edge == edge_id else he2
            loop.he = surviving_he.id

        # --- 4) finally delete the stub edge, its half-edges, and the vertex ---
        for he in other_hes:
            del self.HE[he.id]
        del self.E[other_e_id]
        del self.V[v_id]



    def euler_characteristic(self) -> tuple[int, Dict[int, int]]:
            """
            Compute the Euler characteristic χ = V - E + F for each shell,
            and return (total_χ, per_shell_χ_dict).

            Returns
            -------
            total_chi : int
                The sum of χ over all shells (i.e. χ of the disjoint union of
                boundary surfaces).
            per_shell_chi : Dict[int,int]
                Mapping shell_id → χ_shell.
            """
            per_shell = {}
            total = 0

            for s_id, shell in self.S.items():
                # 1) Faces on this shell
                face_ids = shell.faces
                F_count = len(face_ids)

                # 2) Traverse loops of each face to collect boundary edges & verts
                edge_set = set()
                vert_set = set()

                for f_id in face_ids:
                    face = self.F[f_id]
                    loops = [face.outer] + face.inners
                    for loop_id in loops:
                        for he_id in self._loop_halfedges(loop_id):
                            he = self.HE[he_id]
                            edge_set.add(he.edge)
                            vert_set.add(he.vert)

                E_count = len(edge_set)
                V_count = len(vert_set)

                chi = V_count - E_count + F_count
                per_shell[s_id] = chi
                total += chi

            return total, per_shell

    def topology_check(self) -> int:
        """
            Returns (V - E + F) - sum_s(2 - 2*g_s).
            Here each shell's genus g_s is inferred as
              (# of inner loops on its faces) // 2.
            If the model is topologically consistent, this will return 0.
            """
        # 1) Compute LHS = V - E + F
        V = len(self.V)
        E = len(self.E)
        F = len(self.F)
        lhs = V - E + F

        # 2) Compute RHS = sum over shells of (2 - 2*g_s)
        rhs_total = 0
        for s_id, shell in self.S.items():
            # count all inner loops on this shell’s faces
            hole_loops = sum(len(self.F[f_id].inners) for f_id in shell.faces)
            # each tunnel gives two inner loops → genus = tunnels = hole_loops//2
            g_s = hole_loops // 2
            rhs_total += 2 - 2 * g_s

        return lhs - rhs_total

    def MELF(self, loop_id:int, v1_id: int, v2_id: int )->tuple[Edge,Loop,Face]:

        loop=self.L[loop_id]
        if not loop.is_outer:
            raise ValueError('loop must be outer')
        edge,loop2=self.MEL(loop_id,v1_id,v2_id)

        face = self.new_face( loop2.id, [], self.F[loop.face].shell, same_sense=True, surf=self.F[loop.face].surf)
        loop2.face=face.id
        self.S[self.F[loop.face].shell].faces.append(face.id)

        return edge, loop2, face

    def KELF(self,  edge1_id:int, loop2_id:int):

        loop2=self.L[loop2_id]
        face=self.F[loop2.face]
        shell=self.S[face.shell]
        shell.faces.remove(face.id)

        del self.F[loop2.face]
        loop2.face=None
        self.KEL(edge1_id,loop2_id)
    def get_edge_he(self, edge_id)->HalfEdge:
        E=self.E[edge_id]
        for k,v in self.HE.items():

            if v.edge==edge_id and v.vert==E.v_end:
                return v
        raise ValueError('no he')

    def get_edge_loops(self, edge_id: int) -> tuple[Loop, Loop]:
        """
        Get the left and right loops that contain the given edge.

        Parameters
        ----------
        edge_id : int
            The ID of the edge to find loops for

        Returns
        -------
        tuple[Loop, Loop]
            A tuple containing the left and right loops (Loop objects)

        Raises
        ------
        KeyError
            If the edge is not found
        ValueError
            If the edge doesn't have exactly two half-edges
        """
        if edge_id not in self.E:
            raise KeyError("Edge not found")

        # Find the two half-edges associated with this edge
        he_list = [he for he in self.HE.values() if he.edge == edge_id]
        if len(he_list) != 2:
            raise ValueError("Edge should be referenced by exactly two half-edges")

        he1, he2 = he_list

        # Get the loops these half-edges belong to
        if he1.loop is None or he2.loop is None:
            raise ValueError("One or both half-edges don't belong to a loop")

        loop1 = self.L[he1.loop]
        loop2 = self.L[he2.loop]

        return loop1, loop2

    # ============================================================
    #  MEKH  – Make-Edge / Kill-Hole
    #         outer_loop ⟷ hole_loop  (v_out → v_hole)
    # ============================================================
    def MEKH(
        self,
        outer_loop_id: int,
        hole_loop_id: int,
        v_outer_id: int,
        v_hole_id: int,
    ) -> Edge:
        """Add an edge between *v_outer* (on an outer loop) and
        *v_hole* (on a hole loop) and delete the hole loop."""
        o_loop = self.L[outer_loop_id]
        h_loop = self.L[hole_loop_id]

        if not o_loop.is_outer or h_loop.is_outer:
            raise ValueError("Loop classes must be (outer, hole)")
        if o_loop.face != h_loop.face:
            raise ValueError("Both loops have to belong to the same face")

        # ---- locate anchoring half-edges -------------------------------------------------
        ok, he_o = self._shift_loop_to_vertex(o_loop, self.V[v_outer_id])
        if not ok:
            raise ValueError("v_outer_id not on outer loop")
        ok, he_h = self._shift_loop_to_vertex(h_loop, self.V[v_hole_id])
        if not ok:
            raise ValueError("v_hole_id not on hole loop")

        he_o_prev = self.HE[he_o.prev]
        he_h_prev = self.HE[he_h.prev]

        # ---- make edge + half-edges ------------------------------------------------------
        e_new = self.new_edge(v_outer_id, v_hole_id, Curve3D(), (0.0, 1.0))
        he_out2hole = self.new_halfedge(
            e_new.id, face=o_loop.face, loop=outer_loop_id, vert=v_hole_id, orient=True
        )
        he_hole2out = self.new_halfedge(
            e_new.id, face=o_loop.face, loop=outer_loop_id, vert=v_outer_id, orient=False
        )
        he_out2hole.twin, he_hole2out.twin = he_hole2out.id, he_out2hole.id

        # ---- splice into the two boundary cycles -----------------------------------------
        # (outer loop side)
        he_o_prev.next = he_out2hole.id
        he_out2hole.prev = he_o_prev.id
        he_out2hole.next = he_h.id
        he_h.prev = he_out2hole.id

        # (hole loop side)
        he_h_prev.next = he_hole2out.id
        he_hole2out.prev = he_h_prev.id
        he_hole2out.next = he_o.id
        he_o.prev = he_hole2out.id

        # ---- move every half-edge of the old hole loop to the outer loop -----------------
        for hid in list(self._loop_halfedges(hole_loop_id)):
            self.HE[hid].loop = outer_loop_id

        # ---- drop the hole loop from topology --------------------------------------------
        face = self.F[o_loop.face]
        face.inners.remove(hole_loop_id)
        del self.L[hole_loop_id]

        return e_new

    # ============================================================
    #  KEMH  – Kill-Edge / Make-Hole   (inverse of MEKH)
    # ============================================================
    def KEMH(self, edge_id: int) -> Loop:
        """Remove a bridge edge that currently joins an outer boundary
        with what will become a hole, creating a new hole loop."""
        if edge_id not in self.E:
            raise KeyError("Edge not found")

        # 1. grab the two half-edges and their surrounding pointers
        he_a = self.get_edge_he(edge_id)        # goes v_start → v_end
        he_b = self.HE[he_a.twin]               # opposite orientation

        loop_id = he_a.loop
        loop_outer = self.L[loop_id]
        face_id = loop_outer.face

        # 2. split the ring into two independent cycles
        a_prev, a_next = self.HE[he_a.prev], self.HE[he_a.next]
        b_prev, b_next = self.HE[he_b.prev], self.HE[he_b.next]

        a_prev.next, b_next.prev = b_next.id, a_prev.id
        b_prev.next, a_next.prev = a_next.id, b_prev.id

        # 3. create the new hole loop, starting from b_next
        l_hole = self.new_loop(face=face_id, he=b_next.id, is_outer=False)

        # relabel half-edges that now belong to the hole
        for hid in self._cycle_halfedges(b_next.id):
            self.HE[hid].loop = l_hole.id

        # 4. update face data: the hole moves into *inners*
        self.F[face_id].inners.append(l_hole.id)

        # 5. erase the edge and its half-edges
        del self.HE[he_a.id], self.HE[he_b.id]
        del self.E[edge_id]

        return l_hole

    # ============================================================
    #  MPKH  – Make-Peripheral / Kill-Hole
    #          (promote *hole_loop_id* to its own shell)
    # ============================================================
    def MPKH(self, hole_loop_id: int) -> tuple[Face, Shell]:
        hl = self.L[hole_loop_id]
        if hl.is_outer:
            raise ValueError("Given loop is not a hole")
        face_old = self.F[hl.face]
        shell_old = self.S[face_old.shell]

        # 1. detach from old face
        face_old.inners.remove(hole_loop_id)

        # 2. promote loop & build new face + shell
        hl.is_outer = True
        f_new = self.new_face(outer=hole_loop_id, shell=None,
                              inners=[], same_sense=face_old.same_sense,
                              surf=face_old.surf)
        s_new = self.new_shell(faces=[f_new.id], body=shell_old.body)
        f_new.shell = s_new.id
        hl.face = f_new.id
        self.B[shell_old.body].shells.append(s_new.id)
        self.S[s_new.id] = s_new

        # 3. patch all half-edges to point at the new face
        for hid in self._loop_halfedges(hole_loop_id):
            self.HE[hid].face = f_new.id

        return f_new, s_new

    # ============================================================
    #  KPMH  – Kill-Peripheral / Make-Hole   (inverse of MPKH)
    #          host_loop absorbs *periph_loop* as a hole
    # ============================================================
    def KPMH(self, host_loop_id: int, periph_loop_id: int) -> None:
        host_loop = self.L[host_loop_id]
        per_loop = self.L[periph_loop_id]
        if not (host_loop.is_outer and per_loop.is_outer):
            raise ValueError("Both loops must currently be peripheral")
        if host_loop.face == per_loop.face:
            raise ValueError("periph_loop already belongs to host face")

        face_host = self.F[host_loop.face]
        face_per = self.F[per_loop.face]
        shell_per = self.S[face_per.shell]

        # 1. transfer the loop
        per_loop.is_outer = False
        per_loop.face = face_host.id
        face_host.inners.append(periph_loop_id)

        for hid in self._loop_halfedges(periph_loop_id):
            self.HE[hid].face = face_host.id

        # 2. remove the now-empty peripheral face
        shell_per.faces.remove(face_per.id)
        del self.F[face_per.id]

        # 3. if its shell becomes empty, cull the shell too
        if not shell_per.faces:
            body = self.B[shell_per.body]
            body.shells.remove(shell_per.id)
            del self.S[shell_per.id]
    def MZEV(self, loop1_id: int, loop2_id: int, v_id: int) -> tuple[Edge, Vertex]:
        """
        Make Zero-length Edge and Vertex:
        Split vertex v_id into v_id and a new vertex v2, connecting them by a zero-length edge e1.
        Inserts one half-edge of e1 into loop1 and its twin into loop2.
        Returns (e1, v2).
        """
        # --- lookups & validations ------------------------------------------------
        if loop1_id not in self.L or loop2_id not in self.L:
            raise KeyError("Loop not found")
        if v_id not in self.V:
            raise KeyError("Vertex not found")

        # locate the half-edge in loop1 whose head is v_id
        he1_prev_id = next(
            (hid for hid in self._loop_halfedges(loop1_id)
             if self.HE[hid].vert == v_id),
            None
        )
        if he1_prev_id is None:
            raise ValueError("Vertex not on loop1")
        he1_prev = self.HE[he1_prev_id]
        he1_next_id = he1_prev.next

        # locate the half-edge in loop2 whose head is v_id
        he2_prev_id = next(
            (hid for hid in self._loop_halfedges(loop2_id)
             if self.HE[hid].vert == v_id),
            None
        )
        if he2_prev_id is None:
            raise ValueError("Vertex not on loop2")
        he2_prev = self.HE[he2_prev_id]
        he2_next_id = he2_prev.next

        # --- create new vertex & zero-length edge ---------------------------------
        v_orig = self.V[v_id]
        v2 = self.new_vertex(v_orig.point)
        e1 = self.new_edge(v_id, v2.id, Curve3D(), (0.0, 1.0))

        # --- create the two new half-edges & link them as twins ------------------
        # loop1 half-edge: oriented v1 -> v2
        he1 = self.new_halfedge(
            edge=e1.id,
            face=self.L[loop1_id].face,
            loop=loop1_id,
            prev=he1_prev_id,
            next=he1_next_id,
            twin=None,
            vert=v2.id,
            orient=True,
        )
        # loop2 half-edge: oriented v2 -> v1
        he2 = self.new_halfedge(
            edge=e1.id,
            face=self.L[loop2_id].face,
            loop=loop2_id,
            prev=he2_prev_id,
            next=he2_next_id,
            twin=he1.id,
            vert=v_id,
            orient=False,
        )
        he1.twin = he2.id

        # --- splice each new half-edge into its loop -----------------------------
        # loop1 splice
        he1_prev.next = he1.id
        self.HE[he1_next_id].prev = he1.id

        # loop2 splice
        he2_prev.next = he2.id
        self.HE[he2_next_id].prev = he2.id

        return e1, v2

    def KZEV(self, edge_id: int) -> None:
        """
        Kill Zero-length Edge and Vertex:
        Remove a zero-length edge edge_id and its degree-1 vertex.
        """
        # --- validations ----------------------------------------------------------
        if edge_id not in self.E:
            raise KeyError("Edge not found")
        # capture endpoints before removal
        e1 = self.E[edge_id]
        v1, v2 = e1.v_start, e1.v_end

        # find the two half-edges of this zero-length edge
        hes = [he for he in self.HE.values() if he.edge == edge_id]
        if len(hes) != 2:
            raise ValueError("Edge should have exactly two half-edges")
        he1, he2 = hes

        # --- bypass each half-edge in its loop -----------------------------------
        # loop1
        self.HE[he1.prev].next = he1.next
        self.HE[he1.next].prev = he1.prev
        loop1 = self.L[he1.loop]
        if loop1.he in (he1.id, he2.id):
            loop1.he = he1.next

        # loop2
        self.HE[he2.prev].next = he2.next
        self.HE[he2.next].prev = he2.prev
        loop2 = self.L[he2.loop]
        if loop2.he in (he1.id, he2.id):
            loop2.he = he2.next

        # --- remove topological records ------------------------------------------
        del self.HE[he1.id]
        del self.HE[he2.id]
        del self.E[edge_id]

        # delete the degree-1 vertex
        deg1 = sum(1 for h in self.HE.values() if h.vert == v1)
        deg2 = sum(1 for h in self.HE.values() if h.vert == v2)
        # v2 was created in MZEV and has degree 1, so it should be the dead one
        if deg2 == 0:
            del self.V[v2]
        elif deg1 == 0:
            del self.V[v1]
        else:
            raise ValueError("No degree-1 vertex to kill")


def box(W, D, H):
    m = BRep()
    V1, V2, E1, L1, F, S = m.MEVVLS((D / 2, W / 2, 0.0), (-D / 2, W / 2, 0.0))
    V3, E2 = m.MEV(L1.id, V2.id, p_new=(-D / 2, -W / 2, 0))
    V4, E3 = m.MEV(L1.id, V3.id, p_new=(D / 2, -W / 2, 0))
    E4, L2  ,F2= m.MELF(L1.id, V4.id, V1.id)
    V5, E5 = m.MEV(L1.id, V1.id, p_new=(V1.point[0], V1.point[1], H))
    V6, E6 = m.MEV(L1.id, V2.id, p_new=(V2.point[0], V2.point[1], H))
    V7, E7 = m.MEV(L1.id, V3.id, p_new=(V3.point[0], V3.point[1], H))
    V8, E8 = m.MEV(L1.id, V4.id, p_new=(V4.point[0], V4.point[1], H))
    E9, L3 ,F3= m.MELF(L1.id, V5.id, V6.id)
    E10, L4,F4 = m.MELF(L1.id, V6.id, V7.id)
    E11, L5,F5 = m.MELF(L1.id, V7.id, V8.id)
    E12, L6 ,F6= m.MELF(L1.id, V8.id, V5.id)
    return m


if __name__ == "__main__":
    from mmcore.numeric.plane import plane_line_intersection
    def mve_kve_test(brep:BRep):
        print(get_loops_points(brep))
        edges=[v for v in brep.E.values()]
        edge = brep.E[edges[0].id]
        v1, v2 = brep.V[edge.v_start], brep.V[edge.v_end]

        v1, v2 = np.array(v1.point), np.array(v2.point)
        mid_pt = tuple((v1 + (v2 - v1) * 0.5).tolist())
        V_new, E_new = brep.MVE(edge.id, mid_pt)

        print(get_loops_points(brep))
        brep.KVE(E_new.id,V_new.id)
        print(get_loops_points(brep))

    def split_box(brep:BRep, plane:tuple ):
   

    
        hedges: list[HalfEdge] = [v for v in brep.HE.values()]
        split_edges = []
        split_pts = []
        origin, normal = plane

  
        he_inters = dict()
        verts = []
        per_loop = dict()
        mid_zero_edges = dict()
        per_he = dict()
        per_he_twin = dict()
        while hedges:
            he = hedges.pop(0)

            start, end = np.array(brep.he_points(he))

            if (np.dot(start - origin, normal) * np.dot(end - origin, normal)) < 0:
                # he is inter

                split_edges.append(he)
                twin = brep.HE[he.twin]

                pt_new = (plane_line_intersection(plane, (start, end - start), full_return=False) + (end - start) * 0.1).tolist()

                new_v, new_edge = brep.MVE(he.edge, pt_new)
                new_edge_mid, new_v_next = brep.MZEV(he.loop, twin.loop, new_v.id)
                mid_zero_edges[new_edge_mid.id] = (he.loop, twin.loop)

                if he.loop not in per_loop:
                    per_loop[he.loop] = []
                if twin.loop not in per_loop:
                    per_loop[twin.loop] = []
                per_he[he.id] = new_v_next
                per_he_twin[he.twin] = new_v

                per_loop[he.loop].append(new_v_next)
                per_loop[twin.loop].append(new_v)
        per_he_twin = per_he_twin

        print("ddd")
        per_he_it = list(per_he.items())
        per_he_twin_it = list(per_he_twin.items())
        print([brep.HE[i[0]].loop for i in per_he_twin_it])
        print([brep.HE[i[0]].loop for i in per_he_it])

        for loop_id, verts in per_loop.items():
            loop = brep.L[loop_id]

            new_edge, new_loop, new_face = brep.MELF(loop_id, verts[1].id, verts[0].id)
            

    def get_loops_points(m:BRep):

        return [[m.V[m.HE[i].vert].point for i in m._loop_halfedges(l.id)] for l in m.L.values()]

    def test_get_edge_loops():
        # Create a simple box model
        m = box(1, 1, 1)

        # Get an edge from the model
        edge_id = next(iter(m.E.keys()))

        # Get the loops for this edge
        try:
            loop1, loop2 = m.get_edge_loops(edge_id)
            print(f"Edge {edge_id} is contained in loops {loop1.id} and {loop2.id}")

            # Print the vertices of each loop to verify
            print(f"Loop {loop1.id} vertices: {[m.V[m.HE[i].vert].point for i in m._loop_halfedges(loop1.id)]}")
            print(f"Loop {loop2.id} vertices: {[m.V[m.HE[i].vert].point for i in m._loop_halfedges(loop2.id)]}")

            return True
        except Exception as e:
            print(f"Error testing get_edge_loops: {e}")
            return False
    # Run the test for get_edge_loops
    print("\n=== Testing get_edge_loops ===")
    test_result = test_get_edge_loops()
    print(f"Test {'passed' if test_result else 'failed'}")

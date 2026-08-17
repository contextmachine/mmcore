"""brep_primitives.py – B‑rep skeleton + Euler operators
========================================================
Data structures compatible with Parasolid/ACIS/CGM/OpenCascade/BMesh


"""

from __future__ import annotations

import itertools
import sys
import uuid
from dataclasses import dataclass, field
try:
    from itertools import pairwise
except ImportError:
    from more_itertools import pairwise
from typing import Dict, List, Optional, Tuple, Any, Literal

import numpy as np


# ---------------------------------------------------------------------------
#  Geometry ID counters (geometry objects stored in BRep.G_CRV / G_PCRV / G_SRF)
# ---------------------------------------------------------------------------
_G_CRV_AUTOID = itertools.count()
_G_PCRV_AUTOID = itertools.count()
_G_SRF_AUTOID = itertools.count()

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
    id: int = field(default_factory=lambda :uuid.uuid4().int, init=False)



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
    pcurve: Optional[int] = None
    id: int = field(default_factory=lambda :uuid.uuid4().int, init=False)
@dataclass
class Edge:
    v_start: int
    v_end: int
    geom: Optional[int] = None
    param: Tuple[float, float] = (0.0, 1.0)
    id: int = field(default_factory=lambda :uuid.uuid4().int, init=False)
    he: Optional[int] = None

@dataclass
class Loop:
    face: Optional[int]
    he: int  # entry half‑edge id
    is_outer: bool = True
    id: int = field(default_factory=lambda :uuid.uuid4().int, init=False)


@dataclass
class Face:

    outer: Optional[int] = None  # None for exterior faces (no outer boundary)
    inners: List[int] = field(default_factory=list)
    shell: Optional[int] = None
    same_sense: bool = True
    surf: Optional[int] = None
    id: int = field(default_factory=lambda :uuid.uuid4().int, init=False)


@dataclass
class Shell:
    faces: List[int]
    body: Optional[int] = None
    closed: bool = False
    id: int = field(default_factory=lambda :uuid.uuid4().int, init=False)


@dataclass
class Body:
    shells: List[int]
    lump_type: str = "solid"
    attributes: Dict[str, Any] = field(default_factory=dict)
    id: int = field(default_factory=lambda :uuid.uuid4().int,  init=False)



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
    G_CRV: Dict[int, Any] = field(default_factory=dict)
    G_PCRV: Dict[int, Any] = field(default_factory=dict)
    G_SRF: Dict[int, Any] = field(default_factory=dict)
    def cast(self, entity:int|Any,entity_type:Literal["V","E","HE","L","F","S","B"]=None)->Vertex:
        if not isinstance(entity,int):
            return entity
        if entity_type is None:
            raise ValueError("entity_type must be specified")
        return getattr(self,entity_type)[entity]


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

    def _retag_loop(self, loop_id: int):
        """Walk a loop cycle and set all HEs' loop and face fields consistently."""
        lp = self.L[loop_id]
        face_id = lp.face
        for hid in self._loop_halfedges(loop_id):
            self.HE[hid].loop = loop_id
            self.HE[hid].face = face_id

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
        e = Edge(v1.id, v2.id)
        self.E[e.id] = e
        he_fwd = HalfEdge(e.id, None, None, vert=v2.id, orient=True)
        he_rev = HalfEdge(e.id, None, None, vert=v1.id, orient=False)

        he_fwd.twin, he_rev.twin = he_rev.id, he_fwd.id
        he_fwd.next = he_rev.id
        he_rev.prev = he_fwd.id
        he_rev.next = he_fwd.id
        he_fwd.prev = he_rev.id
        self.HE[he_fwd.id] = he_fwd
        self.HE[he_rev.id] = he_rev
        e.he = he_fwd.id
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
        pcurve: Optional[int] = None,
    ) -> HalfEdge:
        he = HalfEdge(edge=edge, loop=loop, face=face, next=next, prev=prev, twin=twin, vert=vert, orient=orient, pcurve=pcurve)
        self.HE[he.id] = he
        return he

    def new_loop(self, face: Optional[int], he: int, is_outer: bool = True) -> Loop:  # entry half‑edge id
        l = Loop(face=face, he=he, is_outer=is_outer)

        self.L[l.id] = l
        return l

    def new_face(self, outer: Optional[int] = None, inners: List[int] = None, shell: Optional[int] = None, same_sense: bool = True, surf: Optional[int] = None) -> Face:
        if inners is None:
            inners = []
        f = Face(surf=surf, outer=outer, inners=inners, shell=shell, same_sense=same_sense)
        self.F[f.id] = f
        return f

    def new_edge(self, v_start: int, v_end: int, geom: Optional[int] = None, param: Tuple[float, float] = (0.0, 1.0)) -> Edge:
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

    def new_curve(self, geom) -> int:
        cid = uuid.uuid4().int
        self.G_CRV[cid] = geom
        return cid

    def new_pcurve(self, geom) -> int:
        cid = uuid.uuid4().int
        self.G_PCRV[cid] = geom
        return cid

    def new_surface(self, geom) -> int:
        sid = uuid.uuid4().int
        self.G_SRF[sid] = geom
        return sid
    def _edge_split(self, edge_id: int, v_new: int) -> tuple[Edge, HalfEdge, HalfEdge]:
        """
        Split an edge *edge_id* into two edges, one going from its start to
        *v_new* and the other from *v_new* to its end.
        """
        E=self.E[edge_id]
        self.new_edge(E.v_start, v_new, geom=E.geom, param=E.param)
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
        e_new = self.new_edge(v1_id, v2_id)

        he_uv = self.new_halfedge(e_new.id, face=face_id, loop=loop_id, vert=v2_id, orient=True)  # v1 → v2  (stays in loop-1)

        he_vu = self.new_halfedge(e_new.id, face=face_id, loop=None, vert=v1_id, orient=False)  # v2 → v1  (will belong to loop-2)

        he_uv.twin = he_vu.id
        he_vu.twin = he_uv.id
        e_new.he = he_uv.id

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
        e_new = self.new_edge(v_from, v_new.id)

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
        e_new.he = he_fwd.id

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

        he_in = self.HE[he_in_id]  # … → V (arrives at dangling vertex)
        he_out = self.HE[he_in.next]  # V → … (return leg of dangling edge)

        # sanity check: edge used only by these two half-edges
        edge_id = he_in.edge
        if not self._edge_single_use(edge_id, {he_in_id, he_out.id}):
            raise ValueError("Vertex is not of degree 1")

        # bypass the dangling pair: connect he_in.prev → he_out.next
        self.HE[he_in.prev].next = he_out.next
        self.HE[he_out.next].prev = he_in.prev

        # fix loop anchor if it pointed at a deleted HE
        loop = self.L[loop_id]
        if loop.he in (he_in_id, he_out.id):
            loop.he = he_out.next

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

        # he_fwd and he_rev may be on different loops/faces (shared edge)
        loop_id_fwd = he_fwd.loop
        face_id_fwd = he_fwd.face
        loop_id_rev = he_rev.loop
        face_id_rev = he_rev.face

        if loop_id_fwd is None:
            raise ValueError("Edge is not part of any loop")

        # 2) create the new vertex and the new edge
        V_new = self.new_vertex(point_new)
        E2 = self.new_edge(V_new.id, v_old_end, geom=E1.geom, param=E1.param)

        # 3) shorten E1 so that it now ends at V_new
        E1.v_end = V_new.id
        he_fwd.vert = V_new.id

        # 4) create the two new half-edges for E2 (forward + reverse)
        #    he2_fwd inherits from he_fwd; he2_rev inherits from he_rev
        he2_fwd = self.new_halfedge(
            edge=E2.id,
            face=face_id_fwd,
            loop=loop_id_fwd,
            vert=v_old_end,
            orient=True,
        )
        he2_rev = self.new_halfedge(
            edge=E2.id,
            face=face_id_rev,
            loop=loop_id_rev,
            vert=V_new.id,
            orient=False,
        )
        he2_fwd.twin = he2_rev.id
        he2_rev.twin = he2_fwd.id
        E2.he = he2_fwd.id
        E1.he = he_fwd.id

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
                    loops = ([face.outer] if face.outer is not None else []) + face.inners
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

    def validate(self) -> list[str]:
        """Check internal consistency. Returns list of error strings (empty = valid)."""
        errors = []

        # --- Twin symmetry ---
        for he_id, he in self.HE.items():
            if he.twin is None:
                errors.append(f"HE {he_id}: twin is None")
            elif he.twin not in self.HE:
                errors.append(f"HE {he_id}: twin {he.twin} not in HE dict")
            elif self.HE[he.twin].twin != he_id:
                errors.append(f"HE {he_id}: twin.twin={self.HE[he.twin].twin} != {he_id}")

        # --- Next/prev symmetry ---
        for he_id, he in self.HE.items():
            if he.next is None:
                errors.append(f"HE {he_id}: next is None")
            elif he.next not in self.HE:
                errors.append(f"HE {he_id}: next {he.next} not in HE dict")
            elif self.HE[he.next].prev != he_id:
                errors.append(f"HE {he_id}: next.prev={self.HE[he.next].prev} != {he_id}")
            if he.prev is None:
                errors.append(f"HE {he_id}: prev is None")
            elif he.prev not in self.HE:
                errors.append(f"HE {he_id}: prev {he.prev} not in HE dict")
            elif self.HE[he.prev].next != he_id:
                errors.append(f"HE {he_id}: prev.next={self.HE[he.prev].next} != {he_id}")

        # --- Loop closure + loop tag consistency ---
        for l_id, lp in self.L.items():
            if lp.he is None:
                errors.append(f"Loop {l_id}: he is None")
                continue
            if lp.he not in self.HE:
                errors.append(f"Loop {l_id}: he {lp.he} not in HE dict")
                continue
            visited = set()
            cur = lp.he
            while cur not in visited:
                visited.add(cur)
                h = self.HE[cur]
                if h.loop != l_id:
                    errors.append(f"HE {cur}: loop={h.loop}, expected {l_id}")
                if h.next is None:
                    errors.append(f"Loop {l_id}: broken next chain at HE {cur}")
                    break
                cur = h.next
            if cur != lp.he:
                errors.append(f"Loop {l_id}: does not close back to start (landed at {cur})")

        # --- Face tag consistency ---
        for l_id, lp in self.L.items():
            if lp.face is None:
                continue
            if lp.face not in self.F:
                errors.append(f"Loop {l_id}: face {lp.face} not in F dict")
                continue
            for hid in self._loop_halfedges(l_id):
                h = self.HE[hid]
                if h.face != lp.face:
                    errors.append(f"HE {hid}: face={h.face}, expected {lp.face} (from Loop {l_id})")

        # --- Edge consistency: each edge has exactly 2 HEs ---
        from collections import Counter
        edge_he_count = Counter()
        for he_id, he in self.HE.items():
            edge_he_count[he.edge] += 1
        for e_id in self.E:
            count = edge_he_count.get(e_id, 0)
            if count != 2:
                errors.append(f"Edge {e_id}: has {count} half-edges, expected 2")

        # --- Edge.he field consistency ---
        for e_id, e in self.E.items():
            if e.he is not None:
                if e.he not in self.HE:
                    errors.append(f"Edge {e_id}: he {e.he} not in HE dict")
                elif self.HE[e.he].edge != e_id:
                    errors.append(f"Edge {e_id}: he {e.he} points to edge {self.HE[e.he].edge}")

        # --- Vert reference check ---
        for he_id, he in self.HE.items():
            if he.vert is not None and he.vert not in self.V:
                errors.append(f"HE {he_id}: vert {he.vert} not in V dict")

        # --- Face/loop back-reference consistency ---
        for f_id, f in self.F.items():
            if f.outer is not None:
                if f.outer not in self.L:
                    errors.append(f"Face {f_id}: outer loop {f.outer} not in L dict")
                elif self.L[f.outer].face != f_id:
                    errors.append(f"Face {f_id}: outer loop {f.outer} has face={self.L[f.outer].face}")
            for inner_id in f.inners:
                if inner_id not in self.L:
                    errors.append(f"Face {f_id}: inner loop {inner_id} not in L dict")
                elif self.L[inner_id].face != f_id:
                    errors.append(f"Face {f_id}: inner loop {inner_id} has face={self.L[inner_id].face}")

        # --- Shell/face consistency ---
        for s_id, s in self.S.items():
            for f_id in s.faces:
                if f_id not in self.F:
                    errors.append(f"Shell {s_id}: face {f_id} not in F dict")
                elif self.F[f_id].shell != s_id:
                    errors.append(f"Shell {s_id}: face {f_id} has shell={self.F[f_id].shell}")

        # --- Geometry reference checks ---
        for e_id, e in self.E.items():
            if e.geom is not None and e.geom not in self.G_CRV:
                errors.append(f"Edge {e_id}: geom {e.geom} not in G_CRV")

        for f_id, f in self.F.items():
            if f.surf is not None and f.surf not in self.G_SRF:
                errors.append(f"Face {f_id}: surf {f.surf} not in G_SRF")

        for he_id, he in self.HE.items():
            if he.pcurve is not None and he.pcurve not in self.G_PCRV:
                errors.append(f"HE {he_id}: pcurve {he.pcurve} not in G_PCRV")

        # --- Every HE should be reachable from some loop ---
        reachable_hes = set()
        for l_id in self.L:
            try:
                for hid in self._loop_halfedges(l_id):
                    reachable_hes.add(hid)
            except (KeyError, RecursionError):
                pass
        orphan_hes = set(self.HE.keys()) - reachable_hes
        for hid in orphan_hes:
            errors.append(f"HE {hid}: not reachable from any loop")

        return errors

    def MELF(self, loop_id:int, v1_id: int, v2_id: int )->tuple[Edge,Loop,Face]:

        loop=self.L[loop_id]
        if not loop.is_outer:
            raise ValueError('loop must be outer')
        edge,loop2=self.MEL(loop_id,v1_id,v2_id)

        face = self.new_face( loop2.id, [], self.F[loop.face].shell, same_sense=True, surf=self.F[loop.face].surf)
        loop2.face=face.id
        self.S[self.F[loop.face].shell].faces.append(face.id)

        # Retag all half-edges in loop2 to point at the new face
        self._retag_loop(loop2.id)

        return edge, loop2, face

    def KELF(self, edge1_id: int, loop2_id: int):
        loop2 = self.L[loop2_id]
        old_face_id = loop2.face
        face = self.F[old_face_id]
        shell = self.S[face.shell]

        # Determine the surviving loop/face before destroying anything
        he_list = [he for he in self.HE.values() if he.edge == edge1_id]
        if len(he_list) != 2:
            raise ValueError("Edge should be referenced by exactly two half-edges")
        he_a, he_b = he_list
        keep_loop_id = he_b.loop if he_a.loop == loop2_id else he_a.loop
        keep_face_id = self.L[keep_loop_id].face

        # Merge loops (deletes loop2, edge, and its HEs)
        self.KEL(edge1_id, loop2_id)

        # Retag any HEs that still reference the deleted face
        for he in self.HE.values():
            if he.face == old_face_id:
                he.face = keep_face_id

        # Remove the face from shell and delete it
        shell.faces.remove(old_face_id)
        del self.F[old_face_id]
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
        e_new = self.new_edge(v_outer_id, v_hole_id)
        he_out2hole = self.new_halfedge(
            e_new.id, face=o_loop.face, loop=outer_loop_id, vert=v_hole_id, orient=True
        )
        he_hole2out = self.new_halfedge(
            e_new.id, face=o_loop.face, loop=outer_loop_id, vert=v_outer_id, orient=False
        )
        he_out2hole.twin, he_hole2out.twin = he_hole2out.id, he_out2hole.id
        e_new.he = he_out2hole.id

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

        # 3. create the new hole loop from the inner cycle (a_next side)
        #    After splice: a_next cycle = inner/hole HEs, b_next cycle = outer HEs
        l_hole = self.new_loop(face=face_id, he=a_next.id, is_outer=False)

        # relabel half-edges that now belong to the hole
        for hid in self._cycle_halfedges(a_next.id):
            self.HE[hid].loop = l_hole.id

        # 4. ensure outer loop anchor is valid (not a deleted bridge HE)
        if loop_outer.he in (he_a.id, he_b.id):
            loop_outer.he = b_next.id

        # 5. update face data: the hole moves into *inners*
        self.F[face_id].inners.append(l_hole.id)

        # 6. erase the edge and its half-edges
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

        # Retag all half-edges in the promoted loop to point at the new face
        self._retag_loop(hole_loop_id)

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
        e1 = self.new_edge(v_id, v2.id)

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
        e1.he = he1.id

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

    def find_edge_at_point(self, point, atol: float = 1e-3):
        """Find the edge whose 3D curve passes through *point*.

        Only edges with assigned geometry (geom is not None) are searched.
        The parameter must fall within the edge's trimmed param range.

        Returns (edge_id, t_param) if found, else None.
        """
        from mmcore.numeric.closest_point import nurbs_curve_closest_point

        point = np.asarray(point, dtype=float)
        for e_id, e in self.E.items():
            if e.geom is None:
                continue
            crv = self.G_CRV[e.geom]
            t, (dst, *_) = nurbs_curve_closest_point(crv, point)
            if dst < atol:
                t0, t1 = e.param
                tmin, tmax = min(t0, t1), max(t0, t1)
                if tmin - atol <= t <= tmax + atol:
                    return e_id, t
        return None

    def split_edge_at_point(self, point, atol: float = 1e-3) -> 'Vertex':
        """Split the edge containing *point*, inserting a new vertex.

        Calls MVE for the topological split, then updates geometry:
        - Both the shortened and new edge reference the same G_CRV curve.
        - Edge.param ranges are recomputed so they partition the original range.

        Returns the new Vertex, or raises ValueError if no edge contains the point.
        """
        result = self.find_edge_at_point(point, atol)
        if result is None:
            raise ValueError("Point does not lie on any edge")

        edge_id, t_split = result
        e_orig = self.E[edge_id]
        t0, t1 = e_orig.param  # MVE does not modify E1.param

        # topological split
        v_new, e_new = self.MVE(edge_id, tuple(np.asarray(point, dtype=float).tolist()))

        # geometry propagation: partition the param range at the split point
        # MVE already set E2.geom = E1.geom (shared curve reference)
        e_orig.param = (t0, t_split)
        e_new.param = (t_split, t1)

        return v_new

    def split_face_by_curve(self, curve, face_id=None, atol=1e-3):
        """Split a face by inserting a curve between two boundary edges.

        The curve's start/end points must lie on edges of the face.
        Two new vertices are inserted (via split_edge_at_point), then MELF
        splits the loop and creates a new face.

        Parameters
        ----------
        curve : object with .start(), .end(), .interval() methods
            The splitting curve (e.g. NURBSCurveTuple).
        face_id : int, optional
            The face to split. Disambiguates when vertices are on shared edges.
        atol : float
            Geometric tolerance for point-on-edge matching.

        Returns
        -------
        (v_start, v_end, edge_new, loop_new, face_new)
        """
        start = np.asarray(curve.start(), dtype=float)
        end = np.asarray(curve.end(), dtype=float)

        # insert vertices at curve endpoints
        v_start = self.split_edge_at_point(start, atol)
        v_end = self.split_edge_at_point(end, atol)

        # find the common loop on the target face
        v_start_loops = set()
        v_end_loops = set()
        for e in self.E.values():
            if e.v_start == v_start.id or e.v_end == v_start.id:
                for lp in self.get_edge_loops(e.id):
                    v_start_loops.add(lp.id)
            if e.v_start == v_end.id or e.v_end == v_end.id:
                for lp in self.get_edge_loops(e.id):
                    v_end_loops.add(lp.id)

        common = v_start_loops & v_end_loops

        if face_id is not None:
            common = {lid for lid in common if self.L[lid].face == face_id}

        if len(common) == 0:
            raise ValueError("No common loop found for the curve endpoints")
        if len(common) > 1:
            raise ValueError(
                f"Ambiguous: {len(common)} common loops found. "
                f"Pass face_id to disambiguate."
            )

        loop_id = common.pop()

        # topological split — MELF creates edge + loop + face
        e_new, l_new, f_new = self.MELF(loop_id, v_start.id, v_end.id)

        # assign geometry to the new edge
        e_new.geom = self.new_curve(curve)
        e_new.param = curve.interval()

        # inherit surface geometry on the new face
        parent_face = self.F[self.L[loop_id].face]
        f_new.surf = parent_face.surf

        # compute pcurves for the new edge's half-edges
        he_fwd = self.HE[e_new.he]
        he_rev = self.HE[he_fwd.twin]
        self.compute_pcurve(he_fwd.id)
        self.compute_pcurve(he_rev.id)

        return v_start, v_end, e_new, l_new, f_new

    def weld_edges(self, edge1_id: int, edge2_id: int, atol: float = 1e-3):
        """Weld two geometrically coincident edges into one seam edge.

        The two edges must connect the same pair of 3D points (within atol).
        After welding:
        - The coincident vertex pairs are merged (one vertex killed per pair).
        - One edge is killed; the surviving edge becomes a seam edge whose
          twin half-edges both belong to the same face.
        - The killed edge's wire-face half-edges are removed; their loops
          become self-referencing single-HE loops registered as inners
          on the wire face (with face.outer = None).
        - Closed edges (start≡end after vertex merge) get self-looping HEs.

        Parameters
        ----------
        edge1_id, edge2_id : int
            The two edges to weld. Must be geometrically coincident.
        atol : float
            Tolerance for geometric coincidence check.

        Returns
        -------
        (surviving_edge_id, merged_vertices)
            The ID of the surviving seam edge, and a list of
            (kept_vertex_id, killed_vertex_id) pairs.
        """
        if edge1_id not in self.E or edge2_id not in self.E:
            raise KeyError("Edge not found")
        if edge1_id == edge2_id:
            raise ValueError("Cannot weld an edge with itself")

        e1 = self.E[edge1_id]
        e2 = self.E[edge2_id]

        p1s = np.asarray(self.V[e1.v_start].point, dtype=float)
        p1e = np.asarray(self.V[e1.v_end].point, dtype=float)
        p2s = np.asarray(self.V[e2.v_start].point, dtype=float)
        p2e = np.asarray(self.V[e2.v_end].point, dtype=float)

        # --- determine orientation: anti-parallel or parallel ---
        anti = (np.linalg.norm(p1s - p2e) < atol and np.linalg.norm(p1e - p2s) < atol)
        para = (np.linalg.norm(p1s - p2s) < atol and np.linalg.norm(p1e - p2e) < atol)

        if not anti and not para:
            raise ValueError(
                f"Edges E{edge1_id} and E{edge2_id} are not geometrically coincident "
                f"(endpoint distances: {np.linalg.norm(p1s - p2s):.6e}, {np.linalg.norm(p1e - p2e):.6e}, "
                f"{np.linalg.norm(p1s - p2e):.6e}, {np.linalg.norm(p1e - p2s):.6e})"
            )

        # Identify vertex pairs to merge: (keep, kill)
        if anti:
            # E1: A→B, E2: B'→A' where A≡A' and B≡B'
            # E1.v_start≡E2.v_end, E1.v_end≡E2.v_start
            merge_pairs = [
                (e1.v_start, e2.v_end),   # keep e1.v_start, kill e2.v_end
                (e1.v_end, e2.v_start),   # keep e1.v_end, kill e2.v_start
            ]
        else:
            # Parallel: E1.v_start≡E2.v_start, E1.v_end≡E2.v_end
            merge_pairs = [
                (e1.v_start, e2.v_start),
                (e1.v_end, e2.v_end),
            ]

        # De-duplicate and skip if already merged or same vertex
        merge_pairs = [(keep, kill) for keep, kill in merge_pairs
                       if keep != kill and kill in self.V]

        # --- find the half-edges of both edges ---
        e1_hes = [he for he in self.HE.values() if he.edge == edge1_id]
        e2_hes = [he for he in self.HE.values() if he.edge == edge2_id]
        if len(e1_hes) != 2 or len(e2_hes) != 2:
            raise ValueError("Each edge must have exactly 2 half-edges")

        # Identify which HEs are on the body face (has surf) vs wire face (no surf)
        def _classify_hes(hes):
            body_he = wire_he = None
            for he in hes:
                face = self.F[he.face]
                if face.surf is not None:
                    body_he = he
                else:
                    wire_he = he
            return body_he, wire_he

        e1_body, e1_wire = _classify_hes(e1_hes)
        e2_body, e2_wire = _classify_hes(e2_hes)

        if e1_body is None or e2_body is None:
            raise ValueError("Both edges must have one half-edge on a face with geometry")
        if e1_wire is None or e2_wire is None:
            raise ValueError("Both edges must have one half-edge on a wire face")

        # The surviving edge is edge1; edge2 is killed
        # The surviving body HEs are e1_body and e2_body → they become twins (seam)
        # The wire HEs (e1_wire, e2_wire) are removed

        wire_face_id = e1_wire.face
        wire_face = self.F[wire_face_id]
        body_face_id = e1_body.face
        wire_loop_id = e1_wire.loop

        # --- Step 1: Merge vertices ---
        merged = []
        for v_keep, v_kill in merge_pairs:
            # Rewire all HE verts
            for he in self.HE.values():
                if he.vert == v_kill:
                    he.vert = v_keep
            # Rewire all edge endpoints
            for e in self.E.values():
                if e.v_start == v_kill:
                    e.v_start = v_keep
                if e.v_end == v_kill:
                    e.v_end = v_keep
            if v_kill in self.V:
                del self.V[v_kill]
            merged.append((v_keep, v_kill))

        # --- Step 2: Splice wire HEs out of their loop ---
        # Each wire HE's neighbors need to be reconnected.
        # After vertex merge, the neighboring HEs (on closed edges like circles)
        # may become self-loops.
        for wire_he in [e1_wire, e2_wire]:
            prev_he = self.HE[wire_he.prev]
            next_he = self.HE[wire_he.next]

            if prev_he.id == wire_he.id and next_he.id == wire_he.id:
                # Already a self-loop — just remove it
                pass
            elif prev_he.id == next_he.id:
                # The neighbor is the same HE — it becomes a self-loop
                prev_he.next = prev_he.id
                prev_he.prev = prev_he.id
            else:
                prev_he.next = next_he.id
                next_he.prev = prev_he.id

            # Update loop anchor if needed
            lp = self.L[wire_he.loop]
            if lp.he == wire_he.id:
                if next_he.id != wire_he.id:
                    lp.he = next_he.id
                elif prev_he.id != wire_he.id:
                    lp.he = prev_he.id

        # --- Step 3: Make e1_body and e2_body twins (seam edge) ---
        # First, disconnect old twin relationships
        old_e1_body_twin = e1_body.twin  # was e1_wire.id
        old_e2_body_twin = e2_body.twin  # was e2_wire.id

        e1_body.twin = e2_body.id
        e2_body.twin = e1_body.id

        # e2_body now references edge1 (the surviving edge)
        e2_body.edge = edge1_id

        # Update surviving edge's he pointer
        self.E[edge1_id].he = e1_body.id

        # --- Step 4: Create separate loops for each remaining wire HE ---
        # Find all wire-face HEs that survived (the ones from closed edges, not the killed ones)
        # These are self-looping HEs on the wire face
        surviving_wire_hes = set()
        for he in self.HE.values():
            if he.face == wire_face_id and he.id not in (e1_wire.id, e2_wire.id):
                surviving_wire_hes.add(he.id)

        # Each surviving wire HE becomes its own loop (inner on wire face)
        wire_face.outer = None
        wire_face.inners = []
        # Remove old wire loop
        if wire_loop_id in self.L:
            del self.L[wire_loop_id]

        for he_id in surviving_wire_hes:
            he = self.HE[he_id]
            he.next = he_id
            he.prev = he_id
            new_loop = self.new_loop(face=wire_face_id, he=he_id, is_outer=False)
            he.loop = new_loop.id
            he.face = wire_face_id
            wire_face.inners.append(new_loop.id)

        # --- Step 5: Delete the killed edge and its wire HEs ---
        del self.HE[e1_wire.id]
        del self.HE[e2_wire.id]
        del self.E[edge2_id]

        return edge1_id, merged

    def cap_planar_openings(self, atol: float = 1e-3) -> list:
        """Find and cap all planar open boundaries with flat faces.

        Cap candidates are loops on exterior faces (faces without surface
        geometry). After weld_edges, these are the inner loops on the
        exterior face — each representing an open boundary (like the top
        or bottom circle of a cylinder).

        Also checks for the pre-weld case: outer loops on body faces
        where all twin half-edges belong to faces without geometry.

        Returns a list of newly created (Face, Shell) pairs.
        """
        from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple, NURBSCurveTuple
        from mmcore.nurbs._nurbs_knots import reverse_curve, trim_curve

        results = []

        # --- find open boundary loops ---
        open_loops = []

        # Case 1 (post-weld): inner loops on exterior faces (face.surf is None)
        for f_id, face in list(self.F.items()):
            if face.surf is not None:
                continue  # body face, not exterior
            for inner_loop_id in face.inners:
                open_loops.append(inner_loop_id)

        # Case 2 (pre-weld): outer loops on body faces where all twins are on exterior
        for l_id, lp in list(self.L.items()):
            if not lp.is_outer:
                continue
            face = self.F[lp.face]
            if face.surf is None:
                continue
            all_twins_open = True
            for he_id in self._loop_halfedges(l_id):
                he = self.HE[he_id]
                twin = self.HE[he.twin]
                twin_face = self.F[twin.face]
                if twin_face.surf is not None:
                    all_twins_open = False
                    break
            if all_twins_open:
                open_loops.append(l_id)

        for loop_id in open_loops:
            lp = self.L[loop_id]

            # --- collect boundary 3D points ---
            # For self-loops (single HE on a closed edge), sample the edge curve
            # since there's only 1 vertex in the loop.
            he_ids = list(self._loop_halfedges(loop_id))
            if len(he_ids) == 1:
                he = self.HE[he_ids[0]]
                edge = self.E[he.edge]
                if edge.geom is None:
                    continue
                crv = self.G_CRV[edge.geom]
                from mmcore.nurbs._nurbs_eval import evaluate_nurbs_curve
                t0, t1 = edge.param
                n_sample = 16
                pts = np.array([
                    evaluate_nurbs_curve(crv, t0 + (t1 - t0) * i / n_sample, d_order=0)['C']
                    for i in range(n_sample)
                ], dtype=float)
            else:
                pts = np.array([
                    np.asarray(self.V[self.HE[hid].vert].point, dtype=float)
                    for hid in he_ids
                ])

            if len(pts) < 3:
                continue

            # --- test planarity via SVD ---
            centroid = pts.mean(axis=0)
            centered = pts - centroid
            _, s, Vt = np.linalg.svd(centered, full_matrices=False)
            # s[2] is the smallest singular value; if ~0, points are coplanar
            if len(s) < 3 or s[2] > atol:
                continue  # not planar enough
            # s[0] and s[1] are the in-plane extents; both must be significant
            # (reject degenerate loops that collapse to a line or point)
            if s[0] < atol or s[1] < atol:
                continue

            normal = Vt[2]  # normal to the best-fit plane
            # Orient normal away from the body surface.
            # Find the body face via the twin of the first HE in this loop.
            he_first = self.HE[he_ids[0]]
            twin_first = self.HE[he_first.twin]
            body_face = self.F[twin_first.face]
            if body_face.surf is not None:
                from mmcore.nurbs._nurbs_eval import evaluate_nurbs_surface
                srf = self.G_SRF[body_face.surf]
                (u_min, u_max), (v_min, v_max) = srf.interval()
                mid_eval = evaluate_nurbs_surface(srf, (u_min + u_max) / 2, (v_min + v_max) / 2, d_order=0)
                surf_center = mid_eval["S"]
                if np.dot(normal, centroid - surf_center) < 0:
                    normal = -normal

            # --- build planar surface ---
            # Local axes on the plane
            ref = np.array([1.0, 0.0, 0.0]) if abs(normal[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
            xaxis = np.cross(normal, ref)
            xaxis = xaxis / np.linalg.norm(xaxis)
            yaxis = np.cross(normal, xaxis)

            # Extent: bounding box of projected points + margin
            proj_u = centered @ xaxis
            proj_v = centered @ yaxis
            u_range = proj_u.max() - proj_u.min()
            v_range = proj_v.max() - proj_v.min()
            margin = max(u_range, v_range) * 0.25
            r = max(u_range, v_range) / 2 + margin

            c00 = centroid - r * xaxis - r * yaxis
            c10 = centroid + r * xaxis - r * yaxis
            c01 = centroid - r * xaxis + r * yaxis
            c11 = centroid + r * xaxis + r * yaxis

            cap_srf = NURBSSurfaceTuple(
                order_u=2, order_v=2,
                knot_u=np.array([0.0, 0.0, 1.0, 1.0]),
                knot_v=np.array([0.0, 0.0, 1.0, 1.0]),
                control_points=np.array([[c00, c01], [c10, c11]]),
                weights=np.ones((2, 2)),
            )

            # --- build boundary curve for the cap ---
            cap_curves = []
            if len(he_ids) == 1:
                # Self-loop: single closed edge — use its curve directly (reversed
                # so the cap faces outward, opposite to the body face's winding)
                he = self.HE[he_ids[0]]
                edge = self.E[he.edge]
                if edge.geom is not None:
                    crv = self.G_CRV[edge.geom]
                    t0, t1 = edge.param
                    trimmed = trim_curve(crv, min(t0, t1), max(t0, t1))
                    # Reverse: the cap boundary winds opposite to the body face
                    trimmed = reverse_curve(trimmed)
                    cap_curves.append(trimmed)
            else:
                # Multi-edge loop: walk the twin loop to collect reversed boundary
                he_first = self.HE[he_ids[0]]
                twin_first = self.HE[he_first.twin]
                twin_loop_id = twin_first.loop

                for he_id in self._loop_halfedges(twin_loop_id):
                    he = self.HE[he_id]
                    edge = self.E[he.edge]
                    if edge.geom is None:
                        continue
                    crv = self.G_CRV[edge.geom]
                    t0, t1 = edge.param
                    trimmed = trim_curve(crv, min(t0, t1), max(t0, t1))
                    if not he.orient:
                        trimmed = reverse_curve(trimmed)
                    cap_curves.append(trimmed)

            if not cap_curves:
                continue

            # --- create the cap face ---
            cap_face, cap_shell, cap_loop, cap_verts, cap_edges = self.make_face_from_surface(
                cap_srf, boundary_curves=cap_curves, atol=atol
            )

            # --- replace marched pcurves with analytic projections ---
            origin_3d = c00
            u_dir = c10 - c00
            v_dir = c01 - c00
            u_ext = np.linalg.norm(u_dir)
            v_ext = np.linalg.norm(v_dir)
            u_ax = u_dir / u_ext
            v_ax = v_dir / v_ext

            for he_id in self._loop_halfedges(cap_loop.id):
                he = self.HE[he_id]
                edge = self.E[he.edge]
                if edge.geom is None:
                    continue
                crv_3d = self.G_CRV[edge.geom]
                t0, t1 = edge.param
                trimmed = trim_curve(crv_3d, min(t0, t1), max(t0, t1))
                if not he.orient:
                    trimmed = reverse_curve(trimmed)

                # Affine projection of control points to UV
                rel = trimmed.control_points - origin_3d
                pts_2d = np.column_stack([
                    (rel @ u_ax) / u_ext,
                    (rel @ v_ax) / v_ext,
                ])
                pcurve = NURBSCurveTuple(
                    order=trimmed.order,
                    knot=trimmed.knot.copy(),
                    control_points=pts_2d,
                    weights=trimmed.weights.copy(),
                )
                if he.pcurve is not None and he.pcurve in self.G_PCRV:
                    del self.G_PCRV[he.pcurve]
                he.pcurve = self.new_pcurve(pcurve)

            # Also fix pcurves on the wire-side loop
            wire_loop_id = self.L[loop_id].face  # no — we need the other loop
            # The cap face's "other" loop (from MELF) also needs pcurves
            # For N>=2: loop (the wire loop from make_face_from_surface) is separate
            # For N==1: the MEVVLS face IS the cap face, no separate wire loop

            results.append((cap_face, cap_shell))

        return results

    def make_face_from_surface(
        self,
        surface,
        boundary_curves=None,
        atol: float = 1e-3,
        auto_close: bool = False,
    ) -> tuple:
        """Create a face bounded by curves on a surface.

        This is a high-level construction method that creates the full
        topology (vertices, edges, loops, face, shell) and assigns all
        geometry (edge curves, surface, pcurves).

        Parameters
        ----------
        surface : NURBSSurfaceTuple
            The surface geometry for the face.
        boundary_curves : list of NURBSCurveTuple, optional
            Ordered, oriented boundary curves forming a closed loop.
            End of curve[i] must match start of curve[i+1] within *atol*.
            If None, the four natural boundary isocurves of the surface
            are used (untrimmed face).
        atol : float
            Tolerance for endpoint matching. Also stored as vertex.tol.
        auto_close : bool
            If True, automatically detect and weld coincident opposite
            edges (e.g. seam edges on a cylinder, both seam pairs on a
            torus). Detection is based on endpoint coincidence within atol.

        Returns
        -------
        (face, shell, loop, vertices, edges)
            face: the new Face
            shell: the new Shell
            loop: the outer Loop of the face (on the "main" side)
            vertices: list of Vertex objects (one per boundary curve start)
            edges: list of Edge objects (one per boundary curve)
        """
        from mmcore.nurbs.nurbs_iso import extract_isocurve
        from mmcore.numeric.closest_point import nurbs_curve_closest_point

        # --- default: extract the 4 natural boundary isocurves ---
        if boundary_curves is None:
            from mmcore.nurbs._nurbs_knots import reverse_curve
            (u_min, u_max), (v_min, v_max) = surface.interval()
            # Natural isocurve directions:
            #   bottom (v=v_min): u_min → u_max
            #   right  (u=u_max): v_min → v_max
            #   top    (v=v_max): u_min → u_max  (needs reversal for CCW)
            #   left   (u=u_min): v_min → v_max  (needs reversal for CCW)
            c_bottom = extract_isocurve(surface, v_min, direction='v')
            c_right = extract_isocurve(surface, u_max, direction='u')
            c_top = reverse_curve(extract_isocurve(surface, v_max, direction='v'))
            c_left = reverse_curve(extract_isocurve(surface, u_min, direction='u'))
            boundary_curves = [c_bottom, c_right, c_top, c_left]
        else:
            # --- validate user-supplied boundary curves ---
            if len(boundary_curves) < 1:
                raise ValueError("Need at least 1 boundary curve")
            for i in range(len(boundary_curves)):
                crv = boundary_curves[i]
                nxt = boundary_curves[(i + 1) % len(boundary_curves)]
                gap = np.linalg.norm(
                    np.asarray(crv.end(), dtype=float) - np.asarray(nxt.start(), dtype=float)
                )
                if gap > atol:
                    raise ValueError(
                        f"Boundary curve gap at joint {i}→{(i+1) % len(boundary_curves)}: "
                        f"{gap:.6e} > atol={atol}. "
                        f"End of curve[{i}] does not match start of curve[{(i+1) % len(boundary_curves)}]."
                    )

        n = len(boundary_curves)
        srf_id = self.new_surface(surface)

        if n == 1:
            # --- single closed curve (e.g. circle on a cylinder cap) ---
            # The curve starts and ends at the same 3D point.
            # Topology: 1 vertex, 1 edge, 2 half-edges forming a digon loop.
            # MEVVLS creates exactly this (a digon with 2 vertices at
            # the same location). The existing face from MEVVLS serves
            # directly — no MELF needed.
            c0 = boundary_curves[0]
            p_vertex = tuple(np.asarray(c0.start(), dtype=float).tolist())

            v1, v2, e0, loop, face, shell = self.MEVVLS(p_vertex, p_vertex)
            v1.tol = atol
            v2.tol = atol
            e0.geom = self.new_curve(c0)
            e0.param = c0.interval()
            face.surf = srf_id

            # MEVVLS created two vertices at the same point. Topologically
            # they are distinct (edge goes v1→v2), which is correct for
            # a closed curve whose start/end are identified.
            loop_face = loop
            vertices = [v1, v2]
            edges = [e0]
        else:
            # --- N >= 2: standard construction ---
            c0 = boundary_curves[0]
            p_start = tuple(np.asarray(c0.start(), dtype=float).tolist())
            p_end = tuple(np.asarray(c0.end(), dtype=float).tolist())

            v_first, v_prev, e0, loop, face_wire, shell = self.MEVVLS(p_start, p_end)
            v_first.tol = atol
            v_prev.tol = atol

            e0.geom = self.new_curve(c0)
            e0.param = c0.interval()

            vertices = [v_first, v_prev]
            edges = [e0]

            # --- middle edges via MEV ---
            for i in range(1, n - 1):
                ci = boundary_curves[i]
                p_next = tuple(np.asarray(ci.end(), dtype=float).tolist())
                v_new, ei = self.MEV(loop.id, v_prev.id, p_new=p_next)
                v_new.tol = atol
                ei.geom = self.new_curve(ci)
                ei.param = ci.interval()
                vertices.append(v_new)
                edges.append(ei)
                v_prev = v_new

            # --- close with MELF ---
            c_last = boundary_curves[-1]
            e_close, loop_face, face = self.MELF(loop.id, v_prev.id, v_first.id)
            e_close.geom = self.new_curve(c_last)
            e_close.param = c_last.interval()
            edges.append(e_close)

        # --- assign surface to the face ---
        face.surf = srf_id

        # --- compute edge params from vertex projections ---
        for e in edges:
            crv = self.G_CRV[e.geom]
            # Skip closed curves (same start/end vertex point) —
            # closest_point returns the same t for both, giving param=(t,t).
            # The interval set earlier from c.interval() is already correct.
            if np.linalg.norm(
                np.asarray(self.V[e.v_start].point) - np.asarray(self.V[e.v_end].point)
            ) < atol:
                continue
            t0, _ = nurbs_curve_closest_point(crv, self.V[e.v_start].point)
            t1, _ = nurbs_curve_closest_point(crv, self.V[e.v_end].point)
            e.param = (t0, t1)

        # --- compute pcurves for all half-edges on the face ---
        for loop_id in [loop_face.id, loop.id]:
            lp = self.L[loop_id]
            if lp.face is not None and self.F[lp.face].surf is not None:
                for he_id in self._loop_halfedges(loop_id):
                    he = self.HE[he_id]
                    edge = self.E[he.edge]
                    if edge.geom is not None and he.pcurve is None:
                        self.compute_pcurve(he_id, tol=atol)

        # --- auto-close: weld coincident opposite edges ---
        if auto_close and n >= 4:
            # For a 4-edge boundary, opposite pairs are (0,2) and (1,3)
            # Check each pair for geometric coincidence
            weld_pairs = []
            for i, j in [(0, 2), (1, 3)]:
                if i >= len(edges) or j >= len(edges):
                    continue
                ei, ej = edges[i], edges[j]
                # Check if already welded (edge may have been killed)
                if ei.id not in self.E or ej.id not in self.E:
                    continue
                pi_s = np.asarray(self.V[ei.v_start].point, dtype=float)
                pi_e = np.asarray(self.V[ei.v_end].point, dtype=float)
                pj_s = np.asarray(self.V[ej.v_start].point, dtype=float)
                pj_e = np.asarray(self.V[ej.v_end].point, dtype=float)
                anti = (np.linalg.norm(pi_s - pj_e) < atol and
                        np.linalg.norm(pi_e - pj_s) < atol)
                para = (np.linalg.norm(pi_s - pj_s) < atol and
                        np.linalg.norm(pi_e - pj_e) < atol)
                if anti or para:
                    weld_pairs.append((ei.id, ej.id))

            for e1_id, e2_id in weld_pairs:
                if e1_id in self.E and e2_id in self.E:
                    self.weld_edges(e1_id, e2_id, atol=atol)

        return face, shell, loop_face, vertices, edges

    def compute_pcurve(self, he_id: int, tol: float = 1e-4) -> Optional[int]:
        """Compute and assign a parametric curve (pcurve) for a half-edge.

        The pcurve maps the 3D edge curve into the (u,v) parameter space
        of the face that the half-edge belongs to. Uses a predictor-corrector
        marching algorithm: Jacobian-based prediction followed by Newton
        correction, with curvature-adaptive step control in UV space.

        Preconditions: the half-edge's face must have surf set (in G_SRF),
        and its edge must have geom set (in G_CRV). If either is missing,
        returns None without error.

        Returns the pcurve ID (in G_PCRV), or None if preconditions aren't met.
        """
        from mmcore.nurbs._nurbs_eval import (
            evaluate_nurbs_curve,
            evaluate_nurbs_surface,
            NURBSCurveTuple,
        )
        from mmcore.numeric.closest_point import nurbs_surface_closest_point
        from mmcore.nurbs._nurbs_interp import interpolate_curve

        he = self.HE[he_id]
        edge = self.E[he.edge]
        face_id = he.face
        if face_id is None:
            return None
        face = self.F[face_id]

        # --- check geometry preconditions ---
        if edge.geom is None or face.surf is None:
            return None
        crv_3d = self.G_CRV[edge.geom]
        srf = self.G_SRF[face.surf]

        # --- determine marching direction from edge orientation ---
        # he.orient=True means the half-edge follows edge direction (v_start→v_end)
        # he.orient=False means it goes v_end→v_start
        t_start, t_end = edge.param
        if not he.orient:
            t_start, t_end = t_end, t_start

        # --- project endpoints to get UV start/end ---
        pt_start = np.asarray(evaluate_nurbs_curve(crv_3d, t_start, d_order=0)["C"], dtype=float)
        pt_end = np.asarray(evaluate_nurbs_curve(crv_3d, t_end, d_order=0)["C"], dtype=float)

        uv_start, _ = nurbs_surface_closest_point(srf, pt_start)
        uv_end, _ = nurbs_surface_closest_point(srf, pt_end)
        uv_start = np.asarray(uv_start, dtype=float)
        uv_end = np.asarray(uv_end, dtype=float)

        # --- march from uv_start to uv_end ---
        uv_points, t_params = _march_curve_on_surface(
            crv_3d, srf, t_start, t_end, uv_start, uv_end, tol
        )

        # --- fit 2D NURBS through the UV points ---
        # Use the curve parameters as the interpolation params so that
        # pcurve(t) and C(t) share the same parameter domain.
        degree = min(3, len(uv_points) - 1)
        if degree < 1:
            return None

        # Normalize t_params to [0, 1] for knot vector computation,
        # then rescale the knot vector back to [t_start, t_end]
        t_arr = np.array(t_params, dtype=float)
        t_lo, t_hi = t_arr[0], t_arr[-1]
        t_span = t_hi - t_lo
        if abs(t_span) < 1e-30:
            return None
        params_01 = (t_arr - t_lo) / t_span

        ctrl_pts, knots_01 = interpolate_curve(
            np.array(uv_points, dtype=float), degree, params=params_01
        )
        # Rescale knots from [0,1] to [t_start, t_end]
        knots = np.array(knots_01, dtype=float) * t_span + t_lo

        weights = np.ones(len(ctrl_pts), dtype=float)
        pcurve = NURBSCurveTuple(
            order=degree + 1,
            knot=np.array(knots, dtype=float),
            control_points=ctrl_pts,
            weights=weights,
        )

        pcurve_id = self.new_pcurve(pcurve)
        he.pcurve = pcurve_id
        return pcurve_id

from mmcore.nurbs._nurbs_eval import evaluate_nurbs_curve, evaluate_nurbs_surface

def _march_curve_on_surface(crv_3d, srf, t_start, t_end, uv_start, uv_end, tol):
    """March along a 3D curve, tracking its image in surface UV space.

    Uses a predictor-corrector scheme:
      Predictor: Jacobian pseudo-inverse maps 3D curve tangent → UV tangent
      Corrector: Newton iteration minimizes ‖S(u,v) - C(t)‖

    Parameters
    ----------
    crv_3d : NURBSCurveTuple
        The 3D edge curve.
    srf : NURBSSurfaceTuple
        The surface whose parameter space we're mapping into.
    t_start, t_end : float
        Curve parameter range to march over.
    uv_start, uv_end : ndarray of shape (2,)
        Known UV positions at the curve endpoints.
    tol : float
        Tolerance for UV-space accuracy.

    Returns
    -------
    (uv_points, t_params)
        uv_points: list of ndarray — UV points along the curve (including endpoints).
        t_params: list of float — corresponding curve parameter values.
    """

    uv_points = [uv_start.copy()]
    t_params = [t_start]
    t_cur = t_start
    uv_cur = uv_start.copy()
    dt_total = t_end - t_start
    sign = 1.0 if dt_total > 0 else -1.0
    dt_remain = abs(dt_total)

    # initial step: divide the curve into reasonable segments
    n_init = max(8, int(dt_remain / (tol * 100)))
    dt = dt_remain / n_init * sign

    MAX_STEPS = 2000
    for _ in range(MAX_STEPS):
        if abs(t_cur - t_end) < abs(dt_total) * 1e-12:
            break

        # clamp the last step to land exactly on t_end
        if abs(t_cur + dt - t_end) < abs(dt) * 0.5 or abs(t_cur + dt - t_start) > abs(dt_total):
            dt = t_end - t_cur

        t_next = t_cur + dt

        # ── predictor: Jacobian pseudo-inverse ──
        crv_eval = evaluate_nurbs_curve(crv_3d, t_cur, d_order=1)
        srf_eval = evaluate_nurbs_surface(srf, float(uv_cur[0]), float(uv_cur[1]), d_order=1)

        Su = srf_eval["Su"]
        Sv = srf_eval["Sv"]
        C1 = crv_eval["C1"]

        # J = [Su | Sv], a 3×2 matrix; we want J⁺ · C'(t) · dt
        J = np.column_stack([Su, Sv])  # (3, 2)
        tangent_3d = C1 * dt  # 3D displacement along curve
        # pseudo-inverse: (JᵀJ)⁻¹ Jᵀ
        JtJ = J.T @ J
        det = JtJ[0, 0] * JtJ[1, 1] - JtJ[0, 1] * JtJ[1, 0]
        if abs(det) > 1e-30:
            JtJ_inv = np.array([[JtJ[1, 1], -JtJ[0, 1]],
                                [-JtJ[1, 0], JtJ[0, 0]]]) / det
            duv_pred = JtJ_inv @ (J.T @ tangent_3d)
        else:
            # degenerate Jacobian — fall back to straight line in UV
            duv_pred = (uv_end - uv_start) * (dt / dt_total)

        uv_pred = uv_cur + duv_pred

        # ── corrector: Newton iteration on ‖S(u,v) - C(t_next)‖ ──
        target_3d = np.asarray(
            evaluate_nurbs_curve(crv_3d, t_next, d_order=0)["C"], dtype=float
        )
        uv_corr = _newton_uv_correction(srf, uv_pred, target_3d, max_iter=5, tol=tol)

        # ── adaptive step control ──
        # Compare predictor and corrector — large correction means step was too big
        correction_size = np.linalg.norm(uv_corr - uv_pred)
        if correction_size > tol * 10 and abs(dt) > abs(dt_total) * 1e-6:
            # step too large — halve and retry without advancing
            dt *= 0.5
            continue

        # accept step
        uv_cur = uv_corr
        t_cur = t_next
        uv_points.append(uv_cur.copy())
        t_params.append(t_cur)

        # grow step if correction was small
        if correction_size < tol * 0.1:
            dt = min(dt * 1.5, (t_end - t_cur) if sign > 0 else (t_cur - t_end))
            dt *= sign / abs(sign) if dt != 0 else sign

    # ensure the last point is exactly uv_end
    if np.linalg.norm(uv_points[-1] - uv_end) > tol:
        uv_points.append(uv_end.copy())
        t_params.append(t_end)
    print('marching',len(uv_points))
    return uv_points, t_params


def _newton_uv_correction(srf, uv_init, target_3d, max_iter=5, tol=1e-6):
    """Newton iteration to find (u,v) such that S(u,v) ≈ target_3d.

    Starts from uv_init (a good prediction) and refines.
    """
    from mmcore.nurbs._nurbs_eval import evaluate_nurbs_surface

    uv = uv_init.copy()
    for _ in range(max_iter):
        ev = evaluate_nurbs_surface(srf, float(uv[0]), float(uv[1]), d_order=1)
        residual = ev["S"] - target_3d
        dist = np.linalg.norm(residual)
        if dist < tol:
            break

        Su = ev["Su"]
        Sv = ev["Sv"]

        # 2×2 system: [Su·Su  Su·Sv] [du]   [Su·residual]
        #             [Sv·Su  Sv·Sv] [dv] = -[Sv·residual]
        a11 = float(np.dot(Su, Su))
        a12 = float(np.dot(Su, Sv))
        a22 = float(np.dot(Sv, Sv))
        b1 = -float(np.dot(Su, residual))
        b2 = -float(np.dot(Sv, residual))

        det = a11 * a22 - a12 * a12
        if abs(det) < 1e-30:
            break  # singular — accept current uv
        du = (a22 * b1 - a12 * b2) / det
        dv = (a11 * b2 - a12 * b1) / det
        uv[0] += du
        uv[1] += dv

    return uv


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

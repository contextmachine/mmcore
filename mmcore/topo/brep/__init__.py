"""brep_primitives.py – B‑rep skeleton + Euler operators
========================================================
Data structures compatible with Parasolid/ACIS/CGM/OpenCascade/BMesh +
**four low‑level Euler operators**:

* **MEVVLS** / **KEVVLS** – make/kill Edge‑Vertex‑Vertex‑Loop‑Shell
  (wire shell of two vertices and one edge)
* **MEV**  / **KEV**  – make/kill Edge‑Vertex inside an existing Loop
  (classic operator to grow a wire/face by adding one vertex and a new
  edge emanating from a given vertex on the loop)

Only topology manipulation – no geometric calculations beyond a
placeholder straight‑line `Curve3D`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any


# ---------------------------------------------------------------------------
#  Minimal geometry placeholders (replace with real NURBS later)
# ---------------------------------------------------------------------------
class Curve3D:  # straight‑line stub
    pass


class Curve2D: ...


class Surface: ...


# ---------------------------------------------------------------------------
#  Auto‑increment ID helper
# ---------------------------------------------------------------------------
class _AutoID:
    _seq: int = 0

    @classmethod
    def next(cls) -> int:
        cls._seq += 1
        return cls._seq


# ---------------------------------------------------------------------------
#  Topological dataclasses
# ---------------------------------------------------------------------------
@dataclass
class Vertex:
    point: Tuple[float, float, float]
    tol: float = 1e-6
    id: int = field(default_factory=_AutoID.next, init=False)


@dataclass
class Edge:
    v_start: int
    v_end: int
    geom: Curve3D
    param: Tuple[float, float]
    id: int = field(default_factory=_AutoID.next, init=False)


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
    id: int = field(default_factory=_AutoID.next, init=False)


@dataclass
class Loop:
    face: Optional[int]
    he: int  # entry half‑edge id
    is_outer: bool = True
    id: int = field(default_factory=_AutoID.next, init=False)


@dataclass
class Face:

    outer: int
    inners: List[int] = field(default_factory=list)
    shell: Optional[int] = None
    same_sense: bool = True
    surf: Optional[Surface]=None
    id: int = field(default_factory=_AutoID.next, init=False)


@dataclass
class Shell:
    faces: List[int]
    body: Optional[int] = None
    closed: bool = False
    id: int = field(default_factory=_AutoID.next, init=False)


@dataclass
class Body:
    shells: List[int]
    lump_type: str = "solid"
    attributes: Dict[str, Any] = field(default_factory=dict)
    id: int = field(default_factory=_AutoID.next, init=False)


def check_euler(m: BRep):
    V = len(m.V)
    E = len(m.E)
    F = len(m.F)
    L = len(m.L)
    S = len(m.S)
    G = len(m.B)
    return V - E + F - (L - F) - 2 * (S - G)


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
            #print(self.HE[he_id])
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

    def _shift_loop_to_vertex(self, l: Loop, v: Vertex) -> tuple[bool, HalfEdge]:
        start: int = l.he
        current = l.he
        while True:

            he = self.HE[current]
            #print(he)
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
            #print(h)
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
        if he_v1.next == he_v2.id or he_v2.next == he_v1.id:
            raise ValueError("Vertices are adjacent – cannot split loop")

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

    # ============================================================
    #  MVE – Make-Vertex-on-Edge  (split edge at *point_new*)
    # ============================================================
    def MVE(
        self,
        edge_id: int,
        point_new: Tuple[float, float, float],
    ) -> tuple[Vertex, Edge]:
        """
        Split *edge_id* at `point_new`.

        After the call:
        * *edge_id* spans  (v_start → V_new)
        * new edge E2 spans (V_new → old_v_end)

        Returns
        -------
        (Vertex, Edge)
            The inserted vertex V_new and the new edge E2.
        """
        if edge_id not in self.E:
            raise KeyError("Edge not found")

        E1 = self.E[edge_id]
        v_start, v_end_old = E1.v_start, E1.v_end

        # two half-edges of E1
        he_fwd = next(h for h in self.HE.values() if h.edge == edge_id and h.vert == v_end_old)  # v_start → v_end_old
        he_rev = self.HE[he_fwd.twin]  # v_end_old → v_start

        # ---------- create new entities ----------
        V_new = self.new_vertex(point_new)
        E2 = self.new_edge(V_new.id, v_end_old, E1.geom, (0.0, 1.0))

        # ---------- update original edge & its HE pair ----------
        E1.v_end = V_new.id  # edge now stops at the new vertex
        he_fwd.vert = V_new.id  # head = V_new           (v_start → V_new)

        # he_rev becomes half-edge of E2 (v_end_old → V_new)
        he_rev.edge = E2.id
        he_rev.vert = V_new.id  # head = V_new

        # ---------- add the missing twin of E2 (V_new → v_end_old) ----------
        he_new = self.new_halfedge(
            edge=E2.id,
            face=he_fwd.face,
            loop=he_fwd.loop,
            prev=he_fwd.id,
            next=he_fwd.next,
            twin=he_rev.id,
            vert=v_end_old,
            orient=he_fwd.orient,
        )
        he_rev.twin = he_new.id

        # ---------- stitch into loop ----------
        self.HE[he_fwd.next].prev = he_new.id
        he_fwd.next = he_new.id

        return V_new, E2

    # ============================================================
    #  KVE – Kill-Vertex-on-Edge  (inverse of MVE)
    # ============================================================
    def KVE(self, edge_id: int, v_id: int) -> None:
        """
        Undo a previous *MVE*:
        merge the two edges incident to *v_id* back into a single edge *edge_id*
        and delete *v_id*.

        Preconditions
        ------------
        * vertex v_id has degree 2,
          incident to edges *edge_id* and exactly one other edge.
        """
        if edge_id not in self.E:
            raise KeyError("Edge not found")
        if v_id not in self.V:
            raise KeyError("Vertex not found")

        # identify the two edges sharing v_id
        incident_hes = [he for he in self.HE.values() if he.vert == v_id]
        if len(incident_hes) != 2:
            raise ValueError("Vertex degree is not 2")

        he1, he2 = incident_hes
        other_e_id = he2.edge if he1.edge == edge_id else he1.edge

        # half-edges of the *other* edge
        other_hes = [he for he in self.HE.values() if he.edge == other_e_id]
        if len(other_hes) != 2:
            raise RuntimeError("Corrupted edge data")

        # ------------------------------------------------------------------
        #  1) Re-wire half-edge ring: bypass the pair that references other_e
        # ------------------------------------------------------------------
        for he in other_hes:
            self.HE[he.prev].next = he.next
            self.HE[he.next].prev = he.prev

        # ------------------------------------------------------------------
        #  2) Update surviving edge (edge_id) to span the two remote vertices
        # ------------------------------------------------------------------
        E1 = self.E[edge_id]
        v_keep = E1.v_start if E1.v_start != v_id else E1.v_end
        v_other = self.E[other_e_id].v_start if self.E[other_e_id].v_start != v_id else self.E[other_e_id].v_end
        if E1.v_start == v_id:
            E1.v_start = v_other
        else:
            E1.v_end = v_other

        # fix the head-vertex fields on the remaining HE pair of edge_id
        for he in (h for h in self.HE.values() if h.edge == edge_id):
            if he.vert == v_id:
                he.vert = v_other

        # ------------------------------------------------------------------
        #  3) delete doomed records
        # ------------------------------------------------------------------
        for he in other_hes:
            del self.HE[he.id]
        del self.E[other_e_id]
        del self.V[v_id]

    def MELF(self, loop_id:int, v1_id: int, v2_id: int )->tuple[Edge,Loop,Face]:

        loop=self.L[loop_id]
        if not loop.is_outer:
            raise ValueError('loop must be outer')
        edge,loop2=self.MEL(loop_id,v1_id,v2_id)

        face = self.new_face( loop2.id, [], self.F[loop.face].shell, same_sense=True, surf=self.F[loop.face].surf)
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

            if v.edge==edge_id and v.vert==E.v_start:
                return v
        raise ValueError('no he')

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

# ---------------------------------------------------------------------------
#  Quick smoke test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    m = BRep()
    v1,v2,edge1,loop,face,shell = m.MEVVLS((0, 0, 0), (1, 0, 0))

    print("Before MEV →", m.summary())
    v3,edge2 = m.MEV(loop.id, v1.id, (1, 1, 0))
    print("After MEV  →", m.summary())
    # now delete the new vertex
    m.KEV(loop.id, v3.id)

    print("After KEV  →", m.summary())

    

    


    
    def block(W, D, H):
        m = BRep()
        V1, V2, E1, L1, F, S = m.MEVVLS((D / 2, W / 2, 0.0), (-D / 2, W / 2, 0.0))
        print("#", 1)
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L1.id)])
    
        print(list(m._loop_halfedges(L1.id)))
        V3, E2 = m.MEV(L1.id, V2.id, p_new=(-D / 2, -W / 2, 0))
    
        print("#", 2)
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L1.id)])
        print(list(m._loop_halfedges(L1.id)))
        V4, E3 = m.MEV(L1.id, V3.id, p_new=(D / 2, -W / 2, 0))
        print("#", 3)
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L1.id)])
        print(list(m._loop_halfedges(L1.id)))
        E4, L2 = m.MEL(L1.id, V4.id, V1.id)
        print("#", 4)
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L1.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L2.id)])

    
        V5, E5 = m.MEV(L1.id, V1.id, p_new=(D / 2, W / 2, H))
        print("#", 5)
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L1.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L2.id)])

        print(V2)
    
        V6, E6 = m.MEV(L1.id, V3.id, p_new=(-D / 2, -W / 2, H))
        print("#", 6)
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L1.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L2.id)])

        V7, E7 = m.MEV(L1.id, V4.id, p_new=(D / 2, -W / 2, H))
        print("#", 7)
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L1.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L2.id)])

    
        V8, E8 = m.MEV(L1.id, V2.id, p_new=(-D / 2, W / 2, H))
        print("#", 8)
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L1.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L2.id)])
    
        E9, L3 = m.MEL(L1.id, V6.id, V7.id)
        print("#", 9)
    
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L1.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L2.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L3.id)])
    
        print(V6, V7)
        print("\n\n\n")
        E10, L4 = m.MEL(L1.id, V8.id, V6.id)
        print("#", 10)
        print(list(m._loop_halfedges(L1.id)))
        print(list(m._loop_halfedges(L2.id)))
        print(list(m._loop_halfedges(L3.id)))
        print(list(m._loop_halfedges(L4.id)))
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L1.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L2.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L3.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L4.id)])
    
        E11, L5 = m.MEL(L1.id, V7.id, V5.id)
        print("#", 11)
        print(list(m._loop_halfedges(L1.id)))
        print(list(m._loop_halfedges(L2.id)))
        print(list(m._loop_halfedges(L3.id)))
        print(list(m._loop_halfedges(L4.id)))
        print(list(m._loop_halfedges(L5.id)))
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L1.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L2.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L3.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L4.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L5.id)])
    
        E12, L6 = m.MEL(L1.id, V8.id, V5.id)
        print()
        print(list(m._loop_halfedges(L1.id)))
        print(list(m._loop_halfedges(L2.id)))
        print(list(m._loop_halfedges(L3.id)))
        print(list(m._loop_halfedges(L4.id)))
        print(list(m._loop_halfedges(L5.id)))
        print(list(m._loop_halfedges(L6.id)))
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L1.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L2.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L3.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L4.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L5.id)])
        print([m.V[m.HE[i].vert].point for i in m._loop_halfedges(L6.id)])
    
        return m
    
    mm=block(1.,1.,1.)
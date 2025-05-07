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

    # ============================================================
    #  MEV – Make Edge & Vertex, inside loop L, from existing vertex v_from
    # ============================================================
    def MEV(self, loop_id: int, v_from: int, p_new: Tuple[float, float, float]) -> tuple[Vertex, Edge]:
        if loop_id not in self.L:
            raise KeyError("Loop not found")
        if v_from not in self.V:
            raise KeyError("Vertex not found")
        # 1. locate half‑edge whose head == v_from
        he_from = None
        for hid in self._loop_halfedges(loop_id):
            if self.HE[hid].vert == v_from:
                he_from = self.HE[hid]
                break
        if he_from is None:
            raise ValueError("Vertex not on given loop")
        he_prev = self.HE[he_from.prev]

        # 2. create vertex & edge
        v_new = Vertex(p_new)
        self.V[v_new.id] = v_new

        e = Edge(v_from, v_new.id, Curve3D(), (0.0, 1.0))
        self.E[e.id] = e
        # 3. create half‑edges

        he_fwd = HalfEdge(e.id, he_from.face, loop_id, prev=he_prev.id, vert=v_from, orient=he_prev.orient)

        he_rev = HalfEdge(e.id, he_from.face, loop_id, prev=he_fwd.id, next=he_from.id, vert=v_new.id, orient=he_fwd.orient)
        he_fwd.twin=he_rev.id
        he_rev.twin = he_fwd.id
        self.HE[he_fwd.id] = he_fwd
        self.HE[he_rev.id] = he_rev
        he_fwd.next = he_rev.id
        he_prev.next = he_fwd.id
        he_from.prev = he_prev.id

        return v_new, e

    # ============================================================
    #  KEV – inverse of above (remove dangling vertex + edge from loop)
    # ============================================================
    def KEV(self, loop_id: int, v_id: int):
        if loop_id not in self.L:
            raise KeyError("Loop not found")
        if v_id not in self.V:
            raise KeyError("Vertex not found")
        # find half‑edge in loop ending at v_id (head)
        he_del = None
        for hid in self._loop_halfedges(loop_id):
            if self.HE[hid].vert == v_id:
                he_del = self.HE[hid]
                break
        if he_del is None:
            raise ValueError("Vertex not on loop")
        # check degree of vertex =1 and edge used only by these two half‑edges
        edge_id = he_del.edge

        he_prev = self.HE[he_del.prev]
        if he_prev.edge != edge_id:
            raise ValueError(f"prev he diff e: {he_prev}, {he_del}")
        print(he_prev.twin, he_del.id)
        # ensure no other half‑edges use this edge

        # ensure vertex not referenced elsewhere

        self.HE[he_prev.prev].next = he_del.next
        self.HE[he_del.next].prev = he_prev.prev
        if self.L[loop_id].he == he_del.id:
            self.L[loop_id].he = he_del.next

            # adjust loop anchor if needed

        # --- delete records ---

        del self.HE[he_del.id]
        del self.HE[he_prev.id]
        del self.E[edge_id]
        del self.V[v_id]

    def _shift_loop_to_vertex(self, l: Loop, v: Vertex) -> tuple[bool, HalfEdge]:
        start: int = l.he
        current = l.he
        while True:

            he = self.HE[current]
            if he.vert == v.id:
                return True, he
            current = he.next

            if current == start:
                return False, he

    def _walk_to_vertex(self, he: HalfEdge, v: Vertex) -> tuple[list[HalfEdge], HalfEdge]:

        start: int = he.id
        current = start
        lst = []
        while True:
            h = self.HE[current]

            if h.vert == v.id:

                return lst, h
            lst.append(h)
            current = h.next

            if current == start:
                raise ValueError("v not in Loop")

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

    def MEL(self, loop_id: int, v1_id: int, v2_id: int) -> tuple[Edge, Loop]:
        """
         +E, +L
        :param loop_id:
        :param v1_id:
        :param v2_id:
        :return:
        """
        loop: Loop = self.L[loop_id]
        v1: Vertex = self.V[v1_id]
        v2: Vertex = self.V[v2_id]
        success, he_start = self._shift_loop_to_vertex(loop, v1)
        loop.he=he_start.id
        if not success:
            raise ValueError("v1 not in Loop")
        assert he_start.vert==v1.id

        # he_start.prev
        hedges_v1_to_v2, he_v2 = self._walk_to_vertex(he_start, v2)

        assert he_v2.vert == v2.id
        l2 =self.new_loop(face=loop.face, he=he_v2.id, is_outer=False)

        e_new =self.new_edge(v1_id,v2_id,Curve3D(),(0.0, 1.0))

        he_new_l1 = self.new_halfedge(e_new.id, face=loop.face, loop=loop_id, vert=v2_id, orient=True)
        he_new_l2 = self.new_halfedge(e_new.id, face=loop.face, loop=l2.id, vert=v1_id, orient=False)
        l2.he=he_new_l2.id

        he_new_l1.twin=he_new_l2.id
        he_new_l2.twin=he_new_l1.id
        he_v2.prev=self.HE[he_start.twin].next=he_new_l2.id
        he_new_l2.prev=self.HE[he_start.twin].id
        he_new_l2.next=he_v2.id

        he_start.prev=self.HE[he_v2.twin].next=he_new_l1.id
        he_new_l1.prev=self.HE[he_v2.twin].id
        he_new_l1.next=he_start.id
        for he_id in self._cycle_halfedges(he_start.id):
            self.HE[he_id].loop=loop_id
        for he_id in self._cycle_halfedges(he_v2.id):
            self.HE[he_id].loop=l2.id

        return e_new, l2

    def KEL(self, edge1_id:int, loop2_id:int) :
        """
         +E, +L
        :param loop_id:
        :param v1_id:
        :param v2_id:
        :return:
        """
        loop2: Loop = self.L[loop2_id]
        edge1:Edge=self.E[edge1_id]

        v1: Vertex = self.V[edge1.v_start]

        success, he_l2 = self._shift_loop_to_vertex(loop2, v1)
        print(self.L[he_l2.loop])

        he_l1=self.HE[he_l2.twin]
        print(self.L[he_l1.loop])

        loop = he_l1.loop
        self.HE[he_l2.next].prev= self.HE[he_l1.prev].id
        self.HE[he_l2.prev].next = self.HE[he_l1.next].id

        self.HE[he_l1.next].prev= he_l2.prev
        self.HE[he_l1.prev].next = he_l2.next

        del self.HE[he_l1.id]
        del self.HE[he_l2.id]
        del self.E[edge1_id]
        del self.L[loop2_id]


        for he_id in self._cycle_halfedges(  he_l1.next):
            self.HE[he_id].loop=loop


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


def check_euler(m: BRep):
    V = len(m.V)
    E = len(m.E)
    F = len(m.F)
    L = len(m.L)
    S = len(m.S)
    G = len(m.B)
    return V - E + F - (L - F) - 2 * (S - G)

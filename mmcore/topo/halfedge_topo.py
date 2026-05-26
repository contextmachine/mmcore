"""
Half-Edge Topological Data Structure for Shell-Level BREP
=========================================================

A half-edge based boundary representation focused at the Shell level.
Geometry slots (point, curve, surface, pcurve) are provided but user-managed.

Entities:
    Vertex   - topological point; carries optional 3D point geometry
    HalfEdge - directed half of an edge; carries optional pcurve in face param space
    Edge     - undirected edge (pair of HalfEdges); carries optional 3D curve geometry
    Loop     - closed ring of HalfEdges (outer boundary or hole)
    Face     - oriented surface patch bounded by Loops; carries optional surface geometry
    Shell    - connected collection of Faces

Euler-like operators (shell level):
    make_face_shell       +1F +1S        create empty face + shell
    make_edge_loop        +2V +1E +1L    first edge on empty face
    split_edge            +1V +1E        insert vertex into edge
    make_edge_vertex      +1V +1E        extend spike from existing vertex
    split_face            +1E +1F        split face by connecting two verts in same loop
    join_loops            +1E -1L        bridge outer and inner loop
    add_inner_loop        +2V +1E +1L    add hole seed to face

Inverse operators:
    kill_face_shell       -1F -1S
    kill_edge_loop        -2V -1E -1L
    join_edge             -1V -1E
    kill_edge_vertex      -1V -1E
    join_face             -1E -1F
    separate_loop         -1E +1L
    kill_inner_loop       -2V -1E -1L
"""

from __future__ import annotations
from typing import Optional, Iterator, List, Any


# ═══════════════════════════════════════════════════════════════════════════════
# ENTITIES
# ═══════════════════════════════════════════════════════════════════════════════

class Vertex:
    """Topological vertex. Knows one outgoing half-edge."""
    __slots__ = ('id', 'halfedge', 'point')

    def __init__(self, id: int):
        self.id: int = id
        self.halfedge: Optional[HalfEdge] = None   # one outgoing half-edge
        self.point: Any = None                      # user 3D geometry

    def __repr__(self):
        return f"V{self.id}"

    # ── navigation ──────────────────────────────────────────────────────────

    def outgoing(self) -> Iterator[HalfEdge]:
        """Yield every outgoing half-edge around this vertex (fan traversal)."""
        if self.halfedge is None:
            return
        start = self.halfedge
        he = start
        while True:
            yield he
            he = he.prev.twin
            if he is start:
                break

    def incoming(self) -> Iterator[HalfEdge]:
        """Yield every incoming half-edge (twins of outgoing)."""
        for he in self.outgoing():
            yield he.twin

    def neighbors(self) -> Iterator[Vertex]:
        """Yield adjacent vertices."""
        for he in self.outgoing():
            yield he.target

    def edges(self) -> Iterator[Edge]:
        """Yield edges incident to this vertex."""
        for he in self.outgoing():
            yield he.edge

    def faces(self) -> Iterator[Face]:
        """Yield faces around this vertex (may include duplicates for spikes)."""
        for he in self.outgoing():
            if he.loop and he.loop.face:
                yield he.loop.face

    @property
    def degree(self) -> int:
        return sum(1 for _ in self.outgoing())


class HalfEdge:
    """Directed half of an edge. Fundamental navigation element."""
    __slots__ = ('id', 'twin', 'next', 'prev', 'vertex', 'edge', 'loop', 'pcurve')

    def __init__(self, id: int):
        self.id: int = id
        self.twin: Optional[HalfEdge] = None     # opposite half-edge (same edge)
        self.next: Optional[HalfEdge] = None      # next in loop
        self.prev: Optional[HalfEdge] = None      # prev in loop
        self.vertex: Optional[Vertex] = None       # origin vertex
        self.edge: Optional[Edge] = None           # parent edge
        self.loop: Optional[Loop] = None           # parent loop
        self.pcurve: Any = None                    # user parametric curve in face space

    def __repr__(self):
        v0 = self.vertex.id if self.vertex else '?'
        v1 = self.target.id if self.target else '?'
        return f"HE{self.id}({v0}→{v1})"

    @property
    def target(self) -> Optional[Vertex]:
        """Destination vertex (= twin.vertex = next.vertex)."""
        return self.twin.vertex if self.twin else None

    @property
    def face(self) -> Optional[Face]:
        """The face this half-edge belongs to."""
        return self.loop.face if self.loop else None


class Edge:
    """Undirected edge — a pair of twin half-edges."""
    __slots__ = ('id', 'halfedge', 'curve')

    def __init__(self, id: int):
        self.id: int = id
        self.halfedge: Optional[HalfEdge] = None   # one of the two half-edges
        self.curve: Any = None                      # user 3D curve geometry

    def __repr__(self):
        if self.halfedge:
            he = self.halfedge
            return f"E{self.id}({he.vertex}–{he.target})"
        return f"E{self.id}"

    @property
    def he(self) -> Optional[HalfEdge]:
        return self.halfedge

    @property
    def het(self) -> Optional[HalfEdge]:
        return self.halfedge.twin if self.halfedge else None

    @property
    def vertices(self) -> tuple:
        if self.halfedge:
            return (self.halfedge.vertex, self.halfedge.target)
        return ()

    def other_face(self, face: Face) -> Optional[Face]:
        """Return the face on the other side of this edge (or None)."""
        f0 = self.he.face if self.he else None
        f1 = self.het.face if self.het else None
        if f0 is face:
            return f1
        if f1 is face:
            return f0
        return None

    @property
    def is_boundary(self) -> bool:
        """True if both half-edges belong to the same face."""
        return self.he.face is self.het.face if (self.he and self.het) else True


class Loop:
    """Closed ring of half-edges. Represents an outer boundary or a hole."""
    __slots__ = ('id', 'halfedge', 'face')

    def __init__(self, id: int):
        self.id: int = id
        self.halfedge: Optional[HalfEdge] = None   # one half-edge in the ring
        self.face: Optional[Face] = None

    def __repr__(self):
        return f"L{self.id}"

    # ── iteration ───────────────────────────────────────────────────────────

    def halfedges(self) -> Iterator[HalfEdge]:
        """Yield all half-edges in this loop."""
        if self.halfedge is None:
            return
        start = self.halfedge
        he = start
        while True:
            yield he
            he = he.next
            if he is start:
                break

    def vertices(self) -> Iterator[Vertex]:
        """Yield vertices in loop order."""
        for he in self.halfedges():
            yield he.vertex

    def edges(self) -> Iterator[Edge]:
        """Yield edges in loop order."""
        for he in self.halfedges():
            yield he.edge

    @property
    def length(self) -> int:
        """Number of half-edges in this loop."""
        return sum(1 for _ in self.halfedges())

    @property
    def is_outer(self) -> bool:
        """True if this is the outer loop of its face."""
        return self.face is not None and self.face.outer_loop is self

    @property
    def is_inner(self) -> bool:
        """True if this is an inner loop (hole) of its face."""
        return self.face is not None and self in self.face.inner_loops


class Face:
    """Surface patch bounded by loops."""
    __slots__ = ('id', 'outer_loop', 'inner_loops', 'shell', 'surface')

    def __init__(self, id: int):
        self.id: int = id
        self.outer_loop: Optional[Loop] = None
        self.inner_loops: List[Loop] = []
        self.shell: Optional[Shell] = None
        self.surface: Any = None                    # user surface geometry

    def __repr__(self):
        return f"F{self.id}"

    def all_loops(self) -> Iterator[Loop]:
        if self.outer_loop:
            yield self.outer_loop
        yield from self.inner_loops

    def all_halfedges(self) -> Iterator[HalfEdge]:
        for lp in self.all_loops():
            yield from lp.halfedges()

    def all_edges(self) -> Iterator[Edge]:
        seen = set()
        for he in self.all_halfedges():
            if he.edge.id not in seen:
                seen.add(he.edge.id)
                yield he.edge

    def all_vertices(self) -> Iterator[Vertex]:
        seen = set()
        for he in self.all_halfedges():
            if he.vertex.id not in seen:
                seen.add(he.vertex.id)
                yield he.vertex

    def neighbor_faces(self) -> Iterator[Face]:
        """Yield distinct adjacent faces (across shared edges)."""
        seen = set()
        for he in self.all_halfedges():
            other = he.twin.face
            if other is not None and other is not self and other.id not in seen:
                seen.add(other.id)
                yield other


class Shell:
    """Connected collection of faces."""
    __slots__ = ('id', 'faces')

    def __init__(self, id: int):
        self.id: int = id
        self.faces: List[Face] = []

    def __repr__(self):
        return f"S{self.id}"

    def all_edges(self) -> Iterator[Edge]:
        seen = set()
        for f in self.faces:
            for e in f.all_edges():
                if e.id not in seen:
                    seen.add(e.id)
                    yield e

    def all_vertices(self) -> Iterator[Vertex]:
        seen = set()
        for f in self.faces:
            for v in f.all_vertices():
                if v.id not in seen:
                    seen.add(v.id)
                    yield v

    def boundary_edges(self) -> Iterator[Edge]:
        """Yield edges where both half-edges belong to the same face."""
        for e in self.all_edges():
            if e.is_boundary:
                yield e


# ═══════════════════════════════════════════════════════════════════════════════
# TOPOLOGY MANAGER — entity factory + Euler operators + queries
# ═══════════════════════════════════════════════════════════════════════════════

class Topology:
    """
    Owns all topological entities and provides Euler-like operators.

    Usage:
        topo = Topology()
        face, shell = topo.make_face_shell()
        v1, v2, edge, loop = topo.make_edge_loop(face)
        ...
    """

    def __init__(self):
        self._next_id: int = 0
        self.vertices: dict[int, Vertex] = {}
        self.halfedges: dict[int, HalfEdge] = {}
        self.edges: dict[int, Edge] = {}
        self.loops: dict[int, Loop] = {}
        self.faces: dict[int, Face] = {}
        self.shells: dict[int, Shell] = {}

    # ── id generator ────────────────────────────────────────────────────────

    def _new_id(self) -> int:
        self._next_id += 1
        return self._next_id

    # ── internal factories ──────────────────────────────────────────────────

    def _make_vertex(self) -> Vertex:
        v = Vertex(self._new_id())
        self.vertices[v.id] = v
        return v

    def _make_halfedge(self) -> HalfEdge:
        he = HalfEdge(self._new_id())
        self.halfedges[he.id] = he
        return he

    def _make_edge(self, he: HalfEdge, het: HalfEdge) -> Edge:
        e = Edge(self._new_id())
        e.halfedge = he
        he.edge = e
        het.edge = e
        he.twin = het
        het.twin = he
        self.edges[e.id] = e
        return e

    def _make_loop(self, he: HalfEdge) -> Loop:
        lp = Loop(self._new_id())
        lp.halfedge = he
        # set loop ref on all half-edges in the ring
        cur = he
        while True:
            cur.loop = lp
            cur = cur.next
            if cur is he:
                break
        self.loops[lp.id] = lp
        return lp

    def _make_face(self) -> Face:
        f = Face(self._new_id())
        self.faces[f.id] = f
        return f

    def _make_shell(self) -> Shell:
        s = Shell(self._new_id())
        self.shells[s.id] = s
        return s

    # ── internal destructors ────────────────────────────────────────────────

    def _kill_vertex(self, v: Vertex):
        del self.vertices[v.id]

    def _kill_halfedge(self, he: HalfEdge):
        del self.halfedges[he.id]

    def _kill_edge(self, e: Edge):
        del self.edges[e.id]

    def _kill_loop(self, lp: Loop):
        del self.loops[lp.id]

    def _kill_face(self, f: Face):
        del self.faces[f.id]

    def _kill_shell(self, s: Shell):
        del self.shells[s.id]

    # ── helpers ─────────────────────────────────────────────────────────────

    @staticmethod
    def _reloop(he: HalfEdge, lp: Loop):
        """Set loop reference for all half-edges reachable via `next` from he."""
        lp.halfedge = he
        cur = he
        while True:
            cur.loop = lp
            cur = cur.next
            if cur is he:
                break

    @staticmethod
    def _link_pair(a: HalfEdge, b: HalfEdge):
        """Set a.next = b and b.prev = a."""
        a.next = b
        b.prev = a

    # ═══════════════════════════════════════════════════════════════════════
    # FORWARD OPERATORS
    # ═══════════════════════════════════════════════════════════════════════

    def make_face_shell(self) -> tuple[Face, Shell]:
        """
        Create an empty, unbounded face and its shell.

        The face has no loops — it represents an infinite, unrestricted
        surface patch. Loops are added via subsequent operators.

        Returns: (face, shell)
        Topology delta: +1F, +1S
        """
        s = self._make_shell()
        f = self._make_face()
        f.shell = s
        s.faces.append(f)
        return f, s

    def make_edge_loop(self, face: Face) -> tuple[Vertex, Vertex, Edge, Loop]:
        """
        Place the first edge on an empty face, creating its outer loop.

        The two half-edges traverse the single edge in opposite directions,
        forming a degenerate closed loop (digon). Build up the boundary
        by subsequently applying split_edge, make_edge_vertex, split_face.

        Precondition: face.outer_loop is None
        Returns: (v1, v2, edge, loop)
        Topology delta: +2V, +1E, +1L
        """
        if face.outer_loop is not None:
            raise ValueError(f"{face} already has an outer loop; use add_inner_loop")

        v1 = self._make_vertex()
        v2 = self._make_vertex()
        he1 = self._make_halfedge()  # v1 → v2
        he2 = self._make_halfedge()  # v2 → v1

        he1.vertex = v1
        he2.vertex = v2
        v1.halfedge = he1
        v2.halfedge = he2

        # loop: he1 → he2 → he1
        self._link_pair(he1, he2)
        self._link_pair(he2, he1)

        e = self._make_edge(he1, he2)
        lp = self._make_loop(he1)
        lp.face = face
        face.outer_loop = lp
        return v1, v2, e, lp

    def add_inner_loop(self, face: Face) -> tuple[Vertex, Vertex, Edge, Loop]:
        """
        Seed a hole in a face by creating an inner loop with one edge.

        Same structure as make_edge_loop but registers as an inner loop.
        Build up the hole boundary with split_edge / make_edge_vertex,
        then optionally bridge to the outer loop with join_loops.

        Precondition: face.outer_loop is not None
        Returns: (v1, v2, edge, inner_loop)
        Topology delta: +2V, +1E, +1L
        """
        if face.outer_loop is None:
            raise ValueError(f"{face} has no outer loop; use make_edge_loop first")

        v1 = self._make_vertex()
        v2 = self._make_vertex()
        he1 = self._make_halfedge()
        he2 = self._make_halfedge()

        he1.vertex = v1
        he2.vertex = v2
        v1.halfedge = he1
        v2.halfedge = he2

        self._link_pair(he1, he2)
        self._link_pair(he2, he1)

        e = self._make_edge(he1, he2)
        lp = self._make_loop(he1)
        lp.face = face
        face.inner_loops.append(lp)
        return v1, v2, e, lp

    def split_edge(self, he: HalfEdge) -> tuple[Vertex, Edge]:
        """
        Insert a new vertex into he's edge, splitting it in two.

        Before:  ... → he(v_a → v_b) → ...
                 ... → het(v_b → v_a) → ...
        After:   ... → he(v_a → v_new) → he_new(v_new → v_b) → ...
                 ... → het(v_b → v_new) → het_new(v_new → v_a) → ...

        Edge assignment after split:
            old edge  e1: he     ↔ het_new    (v_a — v_new)
            new edge  e2: he_new ↔ het        (v_new — v_b)

        Returns: (v_new, new_edge)
        Topology delta: +1V, +1E
        """
        het = he.twin
        old_edge = he.edge

        v_new = self._make_vertex()

        he_new = self._make_halfedge()   # v_new → v_b
        het_new = self._make_halfedge()  # v_new → v_a

        # --- vertex assignment ---
        he_new.vertex = v_new
        het_new.vertex = v_new
        v_new.halfedge = he_new
        # he.vertex (v_a) and het.vertex (v_b) unchanged

        # --- loop threading (he's loop) ---
        he_next_old = he.next
        self._link_pair(he, he_new)
        self._link_pair(he_new, he_next_old)
        he_new.loop = he.loop

        # --- loop threading (het's loop) ---
        het_next_old = het.next
        self._link_pair(het, het_new)
        self._link_pair(het_new, het_next_old)
        het_new.loop = het.loop

        # --- edge & twin rewiring ---
        # old edge e1: he (v_a→v_new) ↔ het_new (v_new→v_a)
        he.twin = het_new
        het_new.twin = he
        he.edge = old_edge
        het_new.edge = old_edge
        old_edge.halfedge = he

        # new edge e2: he_new (v_new→v_b) ↔ het (v_b→v_new)
        he_new.twin = het
        het.twin = he_new
        new_edge = Edge(self._new_id())
        self.edges[new_edge.id] = new_edge
        new_edge.halfedge = he_new
        he_new.edge = new_edge
        het.edge = new_edge

        return v_new, new_edge

    def make_edge_vertex(self, he: HalfEdge) -> tuple[Vertex, Edge]:
        """
        From he.vertex, extend a new spike edge to a new vertex.

        A spike (or antenna) is inserted into the loop *before* he:
            Before: ... → he.prev → he → ...
            After:  ... → he.prev → he_out(v→v_new) → he_in(v_new→v) → he → ...

        The loop visits v, detours to v_new and back, then continues.

        Returns: (v_new, new_edge)
        Topology delta: +1V, +1E
        """
        v = he.vertex
        v_new = self._make_vertex()

        he_out = self._make_halfedge()  # v → v_new
        he_in = self._make_halfedge()   # v_new → v

        he_out.vertex = v
        he_in.vertex = v_new
        v_new.halfedge = he_in  # outgoing from v_new

        # splice into loop before he
        old_prev = he.prev
        self._link_pair(old_prev, he_out)
        self._link_pair(he_out, he_in)
        self._link_pair(he_in, he)

        he_out.loop = he.loop
        he_in.loop = he.loop

        e = self._make_edge(he_out, he_in)
        return v_new, e

    def split_face(self, he1: HalfEdge, he2: HalfEdge) -> tuple[Edge, Face]:
        """
        Connect he1.vertex and he2.vertex with a new edge, splitting the face.

        Preconditions:
            - he1 and he2 are in the SAME loop of the same face
            - he1 is not he2
            - Walking he1.next… eventually reaches he2 before returning to he1

        The new edge divides the loop into two loops, each becoming
        the outer loop of its own face:
            Loop A (new face): new_he → he2 → … → he1.prev → new_he
            Loop B (orig face): new_het → he1 → … → he2.prev → new_het

        Inner loops of the original face remain with the original face.
        Use `move_inner_loop` to reassign them if needed.

        Returns: (new_edge, new_face)
        Topology delta: +1E, +1F
        """
        if he1 is he2:
            raise ValueError("he1 and he2 must be different half-edges")
        if he1.loop is not he2.loop:
            raise ValueError("he1 and he2 must be in the same loop")

        old_face = he1.face
        old_loop = he1.loop
        old_shell = old_face.shell

        v1 = he1.vertex
        v2 = he2.vertex

        # new half-edges for the splitting edge
        new_he = self._make_halfedge()    # v1 → v2
        new_het = self._make_halfedge()   # v2 → v1

        new_he.vertex = v1
        new_het.vertex = v2

        # save old neighbours
        he1_prev = he1.prev
        he2_prev = he2.prev

        # --- rewire Loop A: new_he → he2 → … → he1_prev → new_he ---
        self._link_pair(he1_prev, new_he)
        self._link_pair(new_he, he2)

        # --- rewire Loop B: new_het → he1 → … → he2_prev → new_het ---
        self._link_pair(he2_prev, new_het)
        self._link_pair(new_het, he1)

        # edge
        new_edge = self._make_edge(new_he, new_het)

        # --- loops ---
        # new face gets Loop A
        new_loop = Loop(self._new_id())
        self.loops[new_loop.id] = new_loop
        self._reloop(new_he, new_loop)

        new_face = self._make_face()
        new_face.shell = old_shell
        old_shell.faces.append(new_face)
        new_loop.face = new_face
        new_face.outer_loop = new_loop

        # old face keeps Loop B
        self._reloop(new_het, old_loop)
        old_loop.face = old_face
        old_face.outer_loop = old_loop

        return new_edge, new_face

    def join_loops(self, he_outer: HalfEdge, he_inner: HalfEdge) -> Edge:
        """
        Bridge an outer-loop vertex to an inner-loop vertex, killing the inner loop.

        Preconditions:
            - he_outer is in face.outer_loop
            - he_inner is in one of face.inner_loops
            - same face

        Before:
            Outer: … → P → he_outer → …
            Inner: … → R → he_inner → …
        After (single outer loop):
            … → P → bridge_a(v1→v2) → he_inner → … → R → bridge_b(v2→v1) → he_outer → …

        Returns: new_edge (the bridge)
        Topology delta: +1E, −1L
        """
        face = he_outer.face
        if face is None or he_inner.face is not face:
            raise ValueError("Both half-edges must belong to the same face")

        outer_loop = he_outer.loop
        inner_loop = he_inner.loop
        if outer_loop is inner_loop:
            raise ValueError("he_outer and he_inner must be in different loops")
        if not outer_loop.is_outer:
            raise ValueError("he_outer must be in the outer loop")
        if inner_loop not in face.inner_loops:
            raise ValueError("he_inner must be in an inner loop of the face")

        v1 = he_outer.vertex
        v2 = he_inner.vertex

        bridge_a = self._make_halfedge()   # v1 → v2
        bridge_b = self._make_halfedge()   # v2 → v1

        bridge_a.vertex = v1
        bridge_b.vertex = v2

        P = he_outer.prev
        R = he_inner.prev

        self._link_pair(P, bridge_a)
        self._link_pair(bridge_a, he_inner)
        self._link_pair(R, bridge_b)
        self._link_pair(bridge_b, he_outer)

        new_edge = self._make_edge(bridge_a, bridge_b)

        # absorb inner loop into outer loop
        face.inner_loops.remove(inner_loop)
        self._kill_loop(inner_loop)
        self._reloop(bridge_a, outer_loop)

        return new_edge

    # ═══════════════════════════════════════════════════════════════════════
    # INVERSE OPERATORS
    # ═══════════════════════════════════════════════════════════════════════

    def kill_face_shell(self, face: Face) -> None:
        """
        Remove a shell that contains exactly one empty face.

        Precondition: shell has one face, face has no loops.
        Topology delta: −1F, −1S
        """
        shell = face.shell
        if face.outer_loop is not None or face.inner_loops:
            raise ValueError(f"{face} still has loops; remove them first")
        if len(shell.faces) != 1:
            raise ValueError(f"{shell} has {len(shell.faces)} faces; expected 1")

        shell.faces.remove(face)
        face.shell = None
        self._kill_face(face)
        self._kill_shell(shell)

    def kill_edge_loop(self, face: Face) -> None:
        """
        Remove the outer loop of a face that contains exactly one edge (digon).

        Inverse of make_edge_loop.
        Precondition: face.outer_loop has exactly 2 half-edges.
        Topology delta: −2V, −1E, −1L
        """
        lp = face.outer_loop
        if lp is None:
            raise ValueError(f"{face} has no outer loop")
        if lp.length != 2:
            raise ValueError(f"Loop has {lp.length} half-edges; expected 2 (digon)")

        he1 = lp.halfedge
        he2 = he1.next

        v1 = he1.vertex
        v2 = he2.vertex
        e = he1.edge

        face.outer_loop = None
        self._kill_loop(lp)
        self._kill_halfedge(he1)
        self._kill_halfedge(he2)
        self._kill_edge(e)
        self._kill_vertex(v1)
        self._kill_vertex(v2)

    def kill_inner_loop(self, face: Face, loop: Loop) -> None:
        """
        Remove an inner loop that is a digon (one edge, two half-edges).

        Inverse of add_inner_loop.
        Topology delta: −2V, −1E, −1L
        """
        if loop not in face.inner_loops:
            raise ValueError(f"{loop} is not an inner loop of {face}")
        if loop.length != 2:
            raise ValueError(f"Inner loop has {loop.length} half-edges; expected 2 (digon)")

        he1 = loop.halfedge
        he2 = he1.next
        v1 = he1.vertex
        v2 = he2.vertex
        e = he1.edge

        face.inner_loops.remove(loop)
        self._kill_loop(loop)
        self._kill_halfedge(he1)
        self._kill_halfedge(he2)
        self._kill_edge(e)
        self._kill_vertex(v1)
        self._kill_vertex(v2)

    def join_edge(self, vertex: Vertex) -> Edge:
        """
        Merge two edges at a degree-2 vertex, removing the vertex.

        Inverse of split_edge.
        Precondition: vertex has exactly degree 2 (two outgoing half-edges),
        and the two edges on each side are distinct.

        Before (at vertex v_mid with degree 2):
            Sequences through v_mid in the loop(s):
              … → he_in2(→v_mid) → he_out1(v_mid→) → …
              … → he_in1(→v_mid) → he_out2(v_mid→) → …
        After:
            he_in1 and he_in2 are repurposed as twins of the merged edge.
            he_out1 and he_out2 are removed from the loop(s).

        Returns: the surviving (merged) edge.
        Topology delta: −1V, −1E
        """
        if vertex.degree != 2:
            raise ValueError(f"{vertex} has degree {vertex.degree}; expected 2")

        # Identify half-edges around vertex
        outgoing = list(vertex.outgoing())
        he_out1 = outgoing[0]          # v_mid → v_far1
        he_out2 = outgoing[1]          # v_mid → v_far2
        he_in1 = he_out1.twin          # v_far1 → v_mid
        he_in2 = he_out2.twin          # v_far2 → v_mid

        e1 = he_in1.edge
        e2 = he_in2.edge
        if e1 is e2:
            raise ValueError("Cannot join: both sides are the same edge")

        # Verify fan structure: at degree-2 the half-edges alternate around vertex.
        # he_out1.prev must be he_in2, and he_out2.prev must be he_in1.
        if he_out1.prev is not he_in2 or he_out2.prev is not he_in1:
            he_out1, he_out2 = he_out2, he_out1
            he_in1, he_in2 = he_in2, he_in1
            e1, e2 = e2, e1
            if he_out1.prev is not he_in2 or he_out2.prev is not he_in1:
                raise ValueError("Unexpected half-edge topology around vertex")

        # Current sequences through v_mid:
        #   … → he_in2 → he_out1 → (he_out1.next) → …
        #   … → he_in1 → he_out2 → (he_out2.next) → …
        # We keep he_in1 and he_in2, remove he_out1 and he_out2.
        # Rewire so that he_in skips over the outgoing half-edge at v_mid.

        he_out1_next = he_out1.next
        he_out2_next = he_out2.next

        # Skip he_out1: link he_in2 directly to he_out1's successor
        self._link_pair(he_in2, he_out1_next)
        # Skip he_out2: link he_in1 directly to he_out2's successor
        self._link_pair(he_in1, he_out2_next)

        # Make he_in1 and he_in2 twins of the merged edge
        he_in1.twin = he_in2
        he_in2.twin = he_in1

        # Assign both to surviving edge e1
        he_in1.edge = e1
        he_in2.edge = e1
        e1.halfedge = he_in1

        # Fix loop references
        loop1 = he_in1.loop
        loop2 = he_in2.loop
        self._reloop(he_in1, loop1)
        if loop2 is not loop1:
            self._reloop(he_in2, loop2)

        # Fix vertex.halfedge references if they pointed to removed half-edges
        v_far1 = he_in1.vertex
        v_far2 = he_in2.vertex
        if v_far1.halfedge is he_out1 or v_far1.halfedge is he_out2:
            v_far1.halfedge = he_in1
        if v_far2.halfedge is he_out1 or v_far2.halfedge is he_out2:
            v_far2.halfedge = he_in2

        # Fix loop.halfedge references if they pointed to removed half-edges
        for lp in self.loops.values():
            if lp.halfedge is he_out1 or lp.halfedge is he_out2:
                lp.halfedge = he_in1 if he_in1.loop is lp else he_in2

        # Cleanup
        self._kill_halfedge(he_out1)
        self._kill_halfedge(he_out2)
        self._kill_edge(e2)
        self._kill_vertex(vertex)

        return e1

    def kill_edge_vertex(self, he_spike: HalfEdge) -> None:
        """
        Remove a spike (degree-1 vertex and its edge).

        Inverse of make_edge_vertex.
        Precondition: he_spike is the half-edge going FROM the spike vertex
        (i.e., he_spike.vertex has degree 1).

        Topology delta: −1V, −1E
        """
        v_tip = he_spike.vertex
        if v_tip.degree != 1:
            raise ValueError(f"{v_tip} has degree {v_tip.degree}; expected 1 (spike tip)")

        he_out = he_spike.twin  # goes towards v_tip
        he_in = he_spike        # goes from v_tip towards base

        # In the loop: ... → he_out.prev → he_out → he_in → he_in.next → ...
        # Remove both: link he_out.prev → he_in.next
        self._link_pair(he_out.prev, he_in.next)

        # Fix loop reference
        lp = he_in.loop
        if lp.halfedge is he_in or lp.halfedge is he_out:
            lp.halfedge = he_in.next

        e = he_in.edge
        self._kill_halfedge(he_in)
        self._kill_halfedge(he_out)
        self._kill_edge(e)
        self._kill_vertex(v_tip)

    def join_face(self, he: HalfEdge) -> None:
        """
        Remove an edge shared by two faces, merging them.

        Inverse of split_face.
        Precondition: he and he.twin are in different faces.

        The face of he.twin is destroyed; its loop is merged into he's face.
        Inner loops from the killed face are transferred.

        Topology delta: −1E, −1F
        """
        het = he.twin
        if he.face is het.face:
            raise ValueError("Both half-edges are on the same face; cannot join_face")

        face_keep = he.face
        face_kill = het.face
        loop_keep = he.loop
        loop_kill = het.loop

        # Splice out he and het from their loops:
        # Loop_keep: … → he.prev → he → he.next → …   →  … → he.prev → het.next → …
        # Wait, no. We need to merge the two loops into one.
        #
        # Loop_keep: … → he.prev → he → he.next → …
        # Loop_kill: … → het.prev → het → het.next → …
        #
        # After removing he and het:
        # … → he.prev → het.next → … → het.prev → he.next → …
        # This creates one merged loop.

        he_prev = he.prev
        he_next = he.next
        het_prev = het.prev
        het_next = het.next

        self._link_pair(he_prev, het_next)
        self._link_pair(het_prev, he_next)

        # Reloop everything to loop_keep
        self._reloop(he_next, loop_keep)
        loop_keep.face = face_keep
        face_keep.outer_loop = loop_keep

        # Transfer inner loops
        for inner_lp in face_kill.inner_loops:
            inner_lp.face = face_keep
            face_keep.inner_loops.append(inner_lp)

        # Remove killed face from shell
        shell = face_kill.shell
        if shell and face_kill in shell.faces:
            shell.faces.remove(face_kill)

        # Cleanup
        e = he.edge
        self._kill_halfedge(he)
        self._kill_halfedge(het)
        self._kill_edge(e)
        self._kill_loop(loop_kill)
        self._kill_face(face_kill)

    def separate_loop(self, he_bridge: HalfEdge) -> Loop:
        """
        Remove a bridge edge to recreate an inner loop (hole).

        Inverse of join_loops.
        Precondition: he_bridge is one half-edge of an edge that, when removed,
        separates the loop into an outer loop and an inner loop.
        he_bridge goes from outer region to inner region.

        Returns: the new inner loop
        Topology delta: −1E, +1L
        """
        het = he_bridge.twin
        face = he_bridge.face
        if face is None:
            raise ValueError("Half-edge has no face")
        if face is not het.face:
            raise ValueError("Both half-edges must be on the same face (same loop)")

        # Before (merged loop):
        # … → he_bridge.prev → he_bridge → he_bridge.next → … → het.prev → het → het.next → …
        #
        # After:
        # Outer loop: … → he_bridge.prev → het.next → …
        # Inner loop: … → het.prev → he_bridge.next → …

        he_prev = he_bridge.prev
        he_next = he_bridge.next
        het_prev = het.prev
        het_next = het.next

        # Reconnect outer loop
        self._link_pair(he_prev, het_next)
        # Reconnect inner loop
        self._link_pair(het_prev, he_next)

        # The outer loop keeps the old loop object
        outer_loop = he_bridge.loop
        self._reloop(het_next, outer_loop)
        outer_loop.face = face
        face.outer_loop = outer_loop

        # Create inner loop
        inner_loop = Loop(self._new_id())
        self.loops[inner_loop.id] = inner_loop
        self._reloop(he_next, inner_loop)
        inner_loop.face = face
        face.inner_loops.append(inner_loop)

        # Kill bridge edge
        e = he_bridge.edge
        self._kill_halfedge(he_bridge)
        self._kill_halfedge(het)
        self._kill_edge(e)

        return inner_loop

    # ═══════════════════════════════════════════════════════════════════════
    # UTILITY OPERATORS
    # ═══════════════════════════════════════════════════════════════════════

    def move_inner_loop(self, loop: Loop, from_face: Face, to_face: Face) -> None:
        """
        Reassign an inner loop from one face to another.

        Useful after split_face when inner loops need redistribution.
        """
        if loop not in from_face.inner_loops:
            raise ValueError(f"{loop} is not an inner loop of {from_face}")
        from_face.inner_loops.remove(loop)
        to_face.inner_loops.append(loop)
        loop.face = to_face
        for he in loop.halfedges():
            he.loop = loop  # already set, but ensures consistency

    def find_halfedge(self, v_from: Vertex, v_to: Vertex) -> Optional[HalfEdge]:
        """Find the half-edge going from v_from to v_to, or None."""
        for he in v_from.outgoing():
            if he.target is v_to:
                return he
        return None

    def find_edge(self, v1: Vertex, v2: Vertex) -> Optional[Edge]:
        """Find the edge between v1 and v2, or None."""
        he = self.find_halfedge(v1, v2)
        return he.edge if he else None

    # ═══════════════════════════════════════════════════════════════════════
    # VALIDATION
    # ═══════════════════════════════════════════════════════════════════════

    def validate(self) -> list[str]:
        """
        Check internal consistency. Returns a list of error messages (empty = valid).
        """
        errors = []

        def err(msg):
            errors.append(msg)

        # --- twin symmetry ---
        for he in self.halfedges.values():
            if he.twin is None:
                err(f"{he}: twin is None")
            elif he.twin.twin is not he:
                err(f"{he}: twin.twin != self")

        # --- next/prev symmetry ---
        for he in self.halfedges.values():
            if he.next is None:
                err(f"{he}: next is None")
            elif he.next.prev is not he:
                err(f"{he}: next.prev != self")
            if he.prev is None:
                err(f"{he}: prev is None")
            elif he.prev.next is not he:
                err(f"{he}: prev.next != self")

        # --- loops are closed ---
        for lp in self.loops.values():
            if lp.halfedge is None:
                err(f"{lp}: halfedge is None")
                continue
            visited = set()
            he = lp.halfedge
            while he.id not in visited:
                visited.add(he.id)
                if he.loop is not lp:
                    err(f"{he}: loop ref is {he.loop}, expected {lp}")
                he = he.next
                if he is None:
                    err(f"{lp}: broken next chain")
                    break
            if he is not lp.halfedge:
                err(f"{lp}: loop does not close back to start")

        # --- edge consistency ---
        for e in self.edges.values():
            he = e.halfedge
            if he is None:
                err(f"{e}: halfedge is None")
                continue
            if he.edge is not e:
                err(f"{e}: halfedge.edge != self")
            if he.twin and he.twin.edge is not e:
                err(f"{e}: twin.edge != self")

        # --- vertex.halfedge is outgoing ---
        for v in self.vertices.values():
            if v.halfedge is None:
                err(f"{v}: halfedge is None")
            elif v.halfedge.vertex is not v:
                err(f"{v}: halfedge.vertex != self")

        # --- face/loop consistency ---
        for f in self.faces.values():
            if f.outer_loop and f.outer_loop.face is not f:
                err(f"{f}: outer_loop.face != self")
            for il in f.inner_loops:
                if il.face is not f:
                    err(f"{f}: inner loop {il}.face != self")

        # --- shell/face consistency ---
        for s in self.shells.values():
            for f in s.faces:
                if f.shell is not s:
                    err(f"{s}: face {f}.shell != self")

        return errors

    # ═══════════════════════════════════════════════════════════════════════
    # STATS / DEBUG
    # ═══════════════════════════════════════════════════════════════════════

    def stats(self) -> dict:
        return {
            'V': len(self.vertices),
            'E': len(self.edges),
            'HE': len(self.halfedges),
            'L': len(self.loops),
            'F': len(self.faces),
            'S': len(self.shells),
        }

    def dump(self) -> str:
        """Human-readable dump of the entire topology."""
        lines = []
        lines.append(f"=== Topology: {self.stats()} ===")

        for s in self.shells.values():
            lines.append(f"\n  {s}:")
            for f in s.faces:
                lines.append(f"    {f}  surface={f.surface}")
                if f.outer_loop:
                    verts = ' → '.join(str(v) for v in f.outer_loop.vertices())
                    lines.append(f"      outer {f.outer_loop}: {verts}")
                for il in f.inner_loops:
                    verts = ' → '.join(str(v) for v in il.vertices())
                    lines.append(f"      inner {il}: {verts}")

        lines.append("\n  Edges:")
        for e in self.edges.values():
            he, het = e.he, e.het
            lines.append(
                f"    {e}  he={he} (face={he.face}) | het={het} (face={het.face})"
            )

        return '\n'.join(lines)
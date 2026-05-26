"""
High-Level Shell Operations
============================

Composes Euler operators into geometry-aware workflows.

The user provides geometry objects (surfaces, curves, points) as opaque
values — this module only stores them on the right topological entities
and orchestrates the correct sequence of Euler operators.

Operations
----------
create_shell(topo, surface, corners, boundary_curves, ...)
    Build a quad face with an outer loop of 4 edges, assign geometry.

add_wire(topo, segments)
    Split faces along a curve that may cross multiple face boundaries.


Topology created by create_shell
---------------------------------

                v0 ──── e_top ──── v1
                │                   │
             e_left             e_right
                │                   │
                v3 ── e_bottom ──── v2

    face_main:  outer loop  v0 → v1 → v2 → v3   (CCW)
    face_ext:   outer loop  v3 → v2 → v1 → v0   (CW, exterior backing)

    Both faces share all 4 edges. face_ext is a topological artifact
    that closes the shell into a manifold; ignore it or mark it.


Topology created by add_wire
------------------------------

    Given a wire (curve) crossing through faces F_a, F_b, ..., the
    algorithm processes each face in sequence:

    1. Split the entry edge    →  creates vertex v_in   (+1V, +1E)
    2. Split the exit edge     →  creates vertex v_out  (+1V, +1E)
    3. split_face(v_in, v_out) →  new edge + new face   (+1E, +1F)

    At face boundaries, the exit vertex of face N is reused as the
    entry vertex of face N+1 (they share the split edge).

    Before                           After
    ┌──────────────────┐            ┌──────────v_in──────┐
    │                  │            │          ╱          │
    │       F_a        │            │  F_a    ╱  F_new   │
    │                  │            │        ╱            │
    └──────────────────┘            └──────v_out─────────┘
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Optional, List, Tuple, Union, Callable, Set
from halfedge_topology import Topology, Vertex, HalfEdge, Edge, Loop, Face, Shell


# ═══════════════════════════════════════════════════════════════════════════════
# DATA TYPES
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ShellResult:
    """Result of create_shell."""
    face: Face                  # the main quad face (surface assigned)
    face_ext: Face              # exterior backing face
    shell: Shell
    vertices: List[Vertex]      # [v0, v1, v2, v3] in CCW order
    edges: List[Edge]           # [e_top, e_right, e_bottom, e_left]
    halfedges_cw: List[HalfEdge]  # HEs of face_main's outer loop, in order


@dataclass
class FaceSegment:
    """
    Describes how a wire passes through one face.

    The caller is responsible for computing the geometric intersections
    and providing the correct half-edges. Both enter_he and exit_he
    MUST be half-edges in the face's outer loop (the face is inferred
    from enter_he.face, or from exit_he.face for the first segment).

    For the first segment: enter_he is required (it will be split).
    For subsequent segments: enter_he is ignored — the entry vertex
    is carried over from the previous segment's exit.

    Vertex-entry / vertex-exit
    --------------------------
    If the wire starts or ends at an EXISTING vertex (e.g., a corner),
    set enter_vertex / exit_vertex instead of enter_he / exit_he.
    When a vertex is given, no edge split is performed.

    Same-edge crossing
    ------------------
    If enter_he and exit_he refer to the same HalfEdge, the entry is
    split first, and the exit is automatically adjusted to the remainder
    of the edge. The entry point must be closer to he.vertex than the
    exit point (i.e., closer to the start of the half-edge direction).
    """
    exit_he: Optional[HalfEdge] = None
    enter_he: Optional[HalfEdge] = None

    exit_vertex: Optional[Vertex] = None
    enter_vertex: Optional[Vertex] = None

    face: Optional[Face] = None         # explicit face (required when using only vertices)

    enter_point: Any = None         # point geometry at entry
    exit_point: Any = None          # point geometry at exit

    enter_t_edge: Optional[float] = None  # parameter [0,1] along entry edge
    exit_t_edge: Optional[float] = None   # parameter [0,1] along exit edge

    curve: Any = None               # 3D curve for the new edge
    pcurve_new: Any = None          # pcurve in the NEW face's param space
    pcurve_old: Any = None          # pcurve in the OLD (kept) face's param space


@dataclass
class WireResult:
    """Result of add_wire."""
    vertices: List[Vertex] = field(default_factory=list)
    edges: List[Edge] = field(default_factory=list)
    new_faces: List[Face] = field(default_factory=list)


# ═══════════════════════════════════════════════════════════════════════════════
# create_shell
# ═══════════════════════════════════════════════════════════════════════════════

def create_shell(
    topo: Topology,
    surface: Any,
    corners: List[Any],
    boundary_curves: List[Any],
    boundary_pcurves: Optional[List[Tuple[Any, Any]]] = None,
) -> ShellResult:
    """
    Create a quad face on a new shell, with full geometry assignment.

    Parameters
    ----------
    topo : Topology
        The topology manager.
    surface : Any
        Surface geometry object, assigned to face.surface.
    corners : list of 4
        Point geometry for the 4 corner vertices: [pt0, pt1, pt2, pt3].
        Ordered CCW when viewed from the face's outward normal.
    boundary_curves : list of 4
        Curve geometry for the 4 boundary edges:
        [crv_01, crv_12, crv_23, crv_30], where crv_ij goes from
        corner i to corner j.
    boundary_pcurves : list of 4 tuples, optional
        Each tuple (pcurve_main, pcurve_ext) provides the parametric
        curve for the main face side and exterior face side of each edge.

    Returns
    -------
    ShellResult

    Construction sequence
    ---------------------
    The quad is built using Euler operators in this order:

        make_face_shell         →  face, shell
        make_edge_loop          →  v0, v1  (digon: v0 ↔ v1)
        make_edge_vertex(v1)    →  v2      (spike: v1 → v2)
        make_edge_vertex(v0)    →  v3      (spike: v0 → v3)
        split_face(v3, v2)      →  e_bottom, face_ext

    After split_face, the original face keeps the quad loop:
        v0 →(e_top)→ v1 →(e_right)→ v2 →(e_bottom)→ v3 →(e_left)→ v0

    The new face (face_ext) gets the reverse-oriented exterior loop.
    """
    if len(corners) != 4:
        raise ValueError(f"Expected 4 corners, got {len(corners)}")
    if len(boundary_curves) != 4:
        raise ValueError(f"Expected 4 boundary curves, got {len(boundary_curves)}")

    # ── Step 1: face + shell ────────────────────────────────────────────
    face, shell = topo.make_face_shell()
    face.surface = surface

    # ── Step 2: first edge (digon v0 ↔ v1) ──────────────────────────────
    v0, v1, e_top, loop = topo.make_edge_loop(face)

    # ── Step 3: spike from v1 → v2 ──────────────────────────────────────
    # make_edge_vertex inserts a spike BEFORE the given HE, from he.vertex.
    # We want the spike at v1. Find the HE starting at v1 in the loop.
    # In the digon: he_01(v0→v1) → he_10(v1→v0) → he_01
    he_10 = v1.halfedge  # outgoing from v1; in the digon this is he_10
    # Verify it's in face's loop
    if he_10.face is not face:
        he_10 = he_10.twin
    v2, e_right = topo.make_edge_vertex(he_10)

    # ── Step 4: spike from v0 → v3 ──────────────────────────────────────
    # After spike 1, loop = he_01 → spike_out(v1→v2) → spike_in(v2→v1) → he_10 → he_01
    # We need HE from v0 in face's loop. That's he_01.
    he_01 = v0.halfedge
    if he_01.face is not face:
        he_01 = he_01.twin
    v3, e_left = topo.make_edge_vertex(he_01)

    # ── Step 5: close the quad ──────────────────────────────────────────
    # Loop now visits: ... → spike_in_1(v2→v1) → he_10(v1→v0) →
    #                  spike_out_2(v0→v3) → spike_in_2(v3→v0) → he_01(v0→v1) →
    #                  spike_out_1(v1→v2) → ...
    #
    # We connect v3 to v2 via split_face.
    # Find HE from v3 and HE from v2 in the loop.
    # We want the old face to keep the quad (v0→v1→v2→v3), and the new
    # face to be the exterior (v3→v2→v1→v0).
    #
    # split_face(he1, he2):
    #   Loop A (new face): new_he → he2 → ... → he1.prev
    #   Loop B (old face): new_het → he1 → ... → he2.prev
    #
    # For the old face to get the quad loop:
    #   Loop B should contain he_01(v0→v1) and spike_out_1(v1→v2).
    #   This happens when he1 is just after v3 (spike_in_2: v3→v0)
    #   and he2 is just after v2 (spike_in_1: v2→v1).
    #
    # So: split_face(he_from_v3_towards_v0, he_from_v2_towards_v1)

    # Find spike_in_2 (v3→v0)
    he_from_v3 = None
    for he in v3.outgoing():
        if he.face is face and he.target is v0:
            he_from_v3 = he
            break
    if he_from_v3 is None:
        # Fallback: any outgoing HE from v3 in face's loop
        for he in v3.outgoing():
            if he.face is face:
                he_from_v3 = he
                break

    # Find spike_in_1 (v2→v1)
    he_from_v2 = None
    for he in v2.outgoing():
        if he.face is face and he.target is v1:
            he_from_v2 = he
            break
    if he_from_v2 is None:
        for he in v2.outgoing():
            if he.face is face:
                he_from_v2 = he
                break

    e_bottom, face_ext = topo.split_face(he_from_v3, he_from_v2)

    # ── Assign geometry ─────────────────────────────────────────────────
    v0.point = corners[0]
    v1.point = corners[1]
    v2.point = corners[2]
    v3.point = corners[3]

    edges = [e_top, e_right, e_bottom, e_left]
    for e, crv in zip(edges, boundary_curves):
        e.curve = crv

    # pcurves: each edge has two HEs; one in face's loop, one in face_ext's loop
    if boundary_pcurves:
        for e, (pc_main, pc_ext) in zip(edges, boundary_pcurves):
            he_main = e.halfedge if e.halfedge.face is face else e.halfedge.twin
            he_ext = he_main.twin
            he_main.pcurve = pc_main
            he_ext.pcurve = pc_ext

    # Collect the HEs of the main face's outer loop in order
    loop_hes = list(face.outer_loop.halfedges())

    return ShellResult(
        face=face,
        face_ext=face_ext,
        shell=shell,
        vertices=[v0, v1, v2, v3],
        edges=edges,
        halfedges_cw=loop_hes,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# add_wire — internal helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _split_boundary_edge(
    topo: Topology,
    he: HalfEdge,
    point: Any,
) -> Tuple[Vertex, Edge, Edge]:
    """
    Split the edge at `he`, creating a new vertex.

    Returns (v_new, edge_lo, edge_hi).

    After split_edge(he):
        he  becomes  v_a → v_new          (same loop as before)  — edge_lo (old edge)
        he.next  is  v_new → v_b  (new)   (same loop as before)  — edge_hi (new edge)

    edge_lo = he.edge (keeps original curve)
    edge_hi = new edge (curve = None until assigned)

    Parameterization:
        If original edge was parameterized [0, 1] with he.vertex at 0,
        then edge_lo covers [0, t] and edge_hi covers [t, 1].
    """
    old_edge = he.edge
    v_new, new_edge = topo.split_edge(he)
    v_new.point = point
    return v_new, old_edge, new_edge


def _find_he_from_vertex_in_face(vertex: Vertex, face: Face) -> HalfEdge:
    """
    Find an outgoing half-edge from `vertex` that is in one of `face`'s loops.

    Raises ValueError if no such half-edge exists.
    """
    for he in vertex.outgoing():
        if he.face is face:
            return he
    raise ValueError(
        f"No outgoing half-edge from {vertex} found in any loop of {face}"
    )


# ═══════════════════════════════════════════════════════════════════════════════
# add_wire
# ═══════════════════════════════════════════════════════════════════════════════

def add_wire(topo: Topology, segments: List[FaceSegment],
             split_curve_fn: Any = None) -> WireResult:
    """
    Split faces along a wire (curve) that crosses one or more faces.

    Parameters
    ----------
    topo : Topology
    segments : list of FaceSegment
        Ordered sequence describing how the wire passes through each face.
        See FaceSegment docstring for field descriptions.
    split_curve_fn : callable, optional
        ``split_curve_fn(original_curve, t_edge) -> (curve_lo, curve_hi)``

        Called after each boundary edge split to assign curve geometry
        to both edge fragments. Without this, the original edge keeps
        its curve but the new fragment gets ``curve=None``, making it
        invisible to future intersection queries.

        ``t_edge`` is from the FaceSegment's enter_t_edge / exit_t_edge.

        Convention:
            - ``curve_lo``  covers the  [0, t_edge]  portion (old edge)
            - ``curve_hi``  covers the  [t_edge, 1]  portion (new edge)

    Returns
    -------
    WireResult
        Contains all new vertices, edges, and faces created.

    Algorithm
    ---------
    For each segment in order:

    1. **Determine entry vertex**
       - First segment: split enter_he → v_enter (or use enter_vertex)
       - Later segments: reuse v_exit from previous segment

    2. **Determine exit vertex**
       - Split exit_he → v_exit (or use exit_vertex)

    3. **Split the face**
       - Find HE from v_enter in face's loop
       - Find HE from v_exit in face's loop
       - split_face(he_enter, he_exit) → new edge, new face

    4. **Assign geometry**
       - new_edge.curve = segment.curve
       - HE pcurves for new and old face sides
       - If split_curve_fn: assign fragment curves to split boundary edges

    Shared vertices at face boundaries
    -----------------------------------
    When the wire exits face F_i at an edge shared with face F_{i+1},
    split_edge creates a vertex on that shared edge. The same vertex
    appears in both F_i's and F_{i+1}'s boundary loops. For the next
    segment, we look up the half-edge from that vertex in F_{i+1}'s
    loop — no second split is needed.

    ::

        Face F_i          shared edge         Face F_{i+1}
        ─────────── v_exit ──────────────────────────────
                    ↑                         ↑
                he_exit (F_i side)     he_exit.twin (F_{i+1} side)

    After split_edge(he_exit), he_exit.twin originates at v_exit
    and sits in F_{i+1}'s loop, providing the entry for the next segment.

    Constraints
    -----------
    - enter_he and exit_he must be in the SAME face's loops.
    - For consecutive segments, the exit edge of segment i must be
      shared with the face of segment i+1.
    - The wire must not cross the same edge twice within one face
      (no self-intersection within a single face).
    """
    if not segments:
        return WireResult()

    result = WireResult()

    def _do_split(he, point, t_edge):
        """Split boundary edge, apply split_curve_fn if available."""
        v_new, edge_lo, edge_hi = _split_boundary_edge(topo, he, point)
        if split_curve_fn is not None and t_edge is not None and edge_lo.curve is not None:
            curve_lo, curve_hi = split_curve_fn(edge_lo.curve, t_edge)
            edge_lo.curve = curve_lo
            edge_hi.curve = curve_hi
        return v_new, he.twin  # twin now originates at v_new, in neighbor face

    # Vertex carried from previous segment's exit
    prev_exit_vertex: Optional[Vertex] = None
    prev_exit_he_twin: Optional[HalfEdge] = None  # HE from v_exit in next face

    for i, seg in enumerate(segments):

        # ── Determine the face ──────────────────────────────────────────
        if i == 0:
            if seg.face is not None:
                face = seg.face
            else:
                ref_he = seg.enter_he or seg.exit_he
                if ref_he is None:
                    raise ValueError(f"Segment {i}: cannot determine face — "
                                     "provide face, enter_he, or exit_he")
                face = ref_he.face
        else:
            # Face is determined by where the previous segment exited
            if seg.face is not None:
                face = seg.face
            elif prev_exit_he_twin is not None:
                face = prev_exit_he_twin.face
            elif seg.exit_he is not None:
                face = seg.exit_he.face
            else:
                raise ValueError(f"Segment {i}: cannot determine face")

        # ── Entry vertex ────────────────────────────────────────────────
        if i == 0:
            # First segment: create or use entry vertex
            if seg.enter_vertex is not None:
                v_enter = seg.enter_vertex
            elif seg.enter_he is not None:
                v_enter, _ = _do_split(seg.enter_he, seg.enter_point, seg.enter_t_edge)
                result.vertices.append(v_enter)
            else:
                raise ValueError(f"Segment 0: must provide enter_he or enter_vertex")
        else:
            # Subsequent: reuse exit vertex from previous segment
            v_enter = prev_exit_vertex

        # ── Exit vertex ─────────────────────────────────────────────────
        if seg.exit_vertex is not None:
            v_exit = seg.exit_vertex
            prev_exit_he_twin = None  # no twin info for existing vertex
        elif seg.exit_he is not None:
            # Handle same-edge crossing: if exit_he was the same object as
            # enter_he (before it was split), the exit is now on he.next
            actual_exit_he = seg.exit_he
            actual_exit_t = seg.exit_t_edge
            if (i == 0 and seg.enter_he is not None
                    and seg.exit_he is seg.enter_he
                    and seg.enter_vertex is None):
                # enter_he was already split; exit is on the remainder
                # After split_edge(enter_he): enter_he is v_a→v_enter,
                # enter_he.next is v_enter→v_b (the remainder).
                actual_exit_he = seg.enter_he.next
                # Reparameterize: the remainder covers [t_enter, 1] of the
                # original edge, so the exit parameter in the remainder is
                # (t_exit - t_enter) / (1 - t_enter)
                if (actual_exit_t is not None and seg.enter_t_edge is not None
                        and seg.enter_t_edge < 1.0):
                    actual_exit_t = (
                        (seg.exit_t_edge - seg.enter_t_edge)
                        / (1.0 - seg.enter_t_edge)
                    )

            v_exit, prev_exit_he_twin = _do_split(
                actual_exit_he, seg.exit_point, actual_exit_t
            )
            result.vertices.append(v_exit)
        else:
            raise ValueError(f"Segment {i}: must provide exit_he or exit_vertex")

        # ── Find half-edges from v_enter and v_exit in face's loop ──────
        he_enter = _find_he_from_vertex_in_face(v_enter, face)
        he_exit = _find_he_from_vertex_in_face(v_exit, face)

        # Verify they're in the same loop
        if he_enter.loop is not he_exit.loop:
            raise ValueError(
                f"Segment {i}: entry and exit vertices are in different loops "
                f"({he_enter.loop} vs {he_exit.loop}) of {face}"
            )

        # ── Split the face ──────────────────────────────────────────────
        new_edge, new_face = topo.split_face(he_enter, he_exit)

        # split_face(he1=he_enter, he2=he_exit) produces:
        #   new_he  (v_enter → v_exit)  in new_face's loop
        #   new_het (v_exit → v_enter)  in old face's loop
        new_he = new_edge.halfedge
        new_het = new_he.twin

        # Identify which HE is in which face
        if new_he.face is new_face:
            he_in_new_face = new_he
            he_in_old_face = new_het
        else:
            he_in_new_face = new_het
            he_in_old_face = new_he

        # ── Assign geometry ─────────────────────────────────────────────
        new_edge.curve = seg.curve
        he_in_new_face.pcurve = seg.pcurve_new
        he_in_old_face.pcurve = seg.pcurve_old

        result.edges.append(new_edge)
        result.new_faces.append(new_face)
        prev_exit_vertex = v_exit

    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: split_face_at_vertices
# ═══════════════════════════════════════════════════════════════════════════════

def split_face_between(
    topo: Topology,
    face: Face,
    v1: Vertex,
    v2: Vertex,
    curve: Any = None,
) -> Tuple[Edge, Face]:
    """
    Split a face by connecting two existing vertices on its boundary.

    Convenience wrapper around split_face when both vertices already
    exist in the face's loop.

    Returns (new_edge, new_face).
    """
    he1 = _find_he_from_vertex_in_face(v1, face)
    he2 = _find_he_from_vertex_in_face(v2, face)

    if he1.loop is not he2.loop:
        raise ValueError(f"{v1} and {v2} are in different loops of {face}")

    new_edge, new_face = topo.split_face(he1, he2)
    new_edge.curve = curve
    return new_edge, new_face


# ═══════════════════════════════════════════════════════════════════════════════
# Convenience: insert_vertex_on_edge
# ═══════════════════════════════════════════════════════════════════════════════

def insert_vertex_on_edge(
    topo: Topology,
    edge: Edge,
    face: Face,
    point: Any = None,
) -> Vertex:
    """
    Insert a vertex on an edge, picking the half-edge in the given face's loop.

    Returns the new vertex.
    """
    he = edge.halfedge
    if he.face is not face:
        he = he.twin
    if he.face is not face:
        raise ValueError(f"Neither half-edge of {edge} is in {face}'s loops")

    v_new, _edge_lo, _edge_hi = _split_boundary_edge(topo, he, point)
    return v_new


# ═══════════════════════════════════════════════════════════════════════════════
# trace_wire — automated wire tracing
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Intersection:
    """
    Result of a curve–curve intersection.

    The user's ``intersect_fn`` must return a list of these.

    Attributes
    ----------
    t_wire : float
        Parameter along the wire curve at the intersection.
    t_edge : float
        Parameter along the edge curve at the intersection.
        Convention: t=0 corresponds to **he.vertex** of the half-edge
        whose direction aligns with the curve parameterization.
        (see ``edge_curve_he`` below).
    point : Any
        The intersection point geometry.
    """
    t_wire: float
    t_edge: float
    point: Any


@dataclass
class _Waypoint:
    """
    Internal: one crossing point of a wire with the shell boundary.

    Either an edge crossing (edge is set) or a vertex waypoint (vertex is set).
    """
    t_wire: float               # parameter along the wire
    point: Any                  # point geometry

    edge: Optional[Edge] = None           # crossed edge (mutually exclusive with vertex)
    t_edge: Optional[float] = None        # parameter along edge curve
    vertex: Optional[Vertex] = None       # existing vertex (corner / prior split)

    def faces(self) -> set:
        """Return the faces touching this waypoint."""
        if self.vertex is not None:
            return set(self.vertex.faces())
        elif self.edge is not None:
            result = set()
            he = self.edge.halfedge
            if he and he.face:
                result.add(he.face)
            if he and he.twin and he.twin.face:
                result.add(he.twin.face)
            return result
        return set()


@dataclass
class TraceResult:
    """Result of trace_wire."""
    wire_result: WireResult             # the underlying add_wire result
    crossings: List[_Waypoint] = field(default_factory=list)
    face_sequence: List[Face] = field(default_factory=list)


def _collect_crossings(
    shell: Shell,
    wire_curve: Any,
    intersect_fn: Callable,
    exclude_faces: Set[Face],
    vertex_tol: float,
) -> List[_Waypoint]:
    """
    Intersect the wire curve with every edge of the shell.

    Returns a sorted list of _Waypoint objects.
    """
    waypoints: List[_Waypoint] = []
    seen_edges: set = set()

    for face in shell.faces:
        if face in exclude_faces:
            continue
        for loop in face.all_loops():
            for he in loop.halfedges():
                eid = he.edge.id
                if eid in seen_edges:
                    continue
                seen_edges.add(eid)

                edge = he.edge
                if edge.curve is None:
                    continue

                hits = intersect_fn(wire_curve, edge.curve)
                if not hits:
                    continue

                for hit in hits:
                    wp = _Waypoint(
                        t_wire=hit.t_wire,
                        t_edge=hit.t_edge,
                        point=hit.point,
                        edge=edge,
                    )

                    # Check if intersection is at an existing vertex.
                    # We use t_edge proximity: t≈0 means he.vertex end,
                    # t≈1 means he.target end (for the HE aligned with
                    # the curve parameterization).
                    #
                    # Since we don't know which HE matches the curve
                    # direction, we check both ends of the edge.
                    v_start, v_end = edge.vertices
                    if hit.t_edge <= vertex_tol:
                        wp.vertex = v_start
                        wp.edge = None
                        wp.t_edge = None
                    elif hit.t_edge >= 1.0 - vertex_tol:
                        wp.vertex = v_end
                        wp.edge = None
                        wp.t_edge = None

                    waypoints.append(wp)

    # Sort by wire parameter
    waypoints.sort(key=lambda w: w.t_wire)
    return waypoints


def _dedup_vertex_waypoints(
    waypoints: List[_Waypoint],
    wire_tol: float,
) -> List[_Waypoint]:
    """
    Merge waypoints that refer to the same vertex (happens when the wire
    hits a corner shared by multiple edges — each edge reports the same
    intersection).
    """
    if not waypoints:
        return waypoints

    merged: List[_Waypoint] = [waypoints[0]]
    for wp in waypoints[1:]:
        prev = merged[-1]

        # Same vertex, close in wire parameter → duplicate
        if (prev.vertex is not None
                and wp.vertex is not None
                and prev.vertex is wp.vertex
                and abs(wp.t_wire - prev.t_wire) < wire_tol):
            continue

        merged.append(wp)
    return merged


def _find_face_between(
    wp_enter: _Waypoint,
    wp_exit: _Waypoint,
    exclude: Set[Face],
) -> Face:
    """
    Find the face traversed by the wire between two consecutive waypoints.

    The target face must appear in the boundary of BOTH waypoints
    (each edge/vertex borders a small set of faces).
    """
    faces_enter = wp_enter.faces() - exclude
    faces_exit = wp_exit.faces() - exclude
    shared = faces_enter & faces_exit

    if len(shared) == 1:
        return shared.pop()
    elif len(shared) > 1:
        # Multiple shared faces. Pick the one that is not in common with
        # ANY earlier/later waypoint (heuristic for branching topologies).
        # For typical grids, just take any.
        return shared.pop()
    else:
        raise ValueError(
            f"No shared face between waypoints at t_wire="
            f"{wp_enter.t_wire:.6f} and t_wire={wp_exit.t_wire:.6f}.\n"
            f"  Enter faces: {faces_enter}\n"
            f"  Exit faces:  {faces_exit}\n"
            f"  Excluded:    {exclude}"
        )


def _pick_he_in_face(edge: Edge, face: Face) -> HalfEdge:
    """Return the half-edge of `edge` that lies in one of `face`'s loops."""
    he = edge.halfedge
    if he.face is face:
        return he
    if he.twin.face is face:
        return he.twin
    raise ValueError(f"Neither half-edge of {edge} is in {face}")


def trace_wire(
    topo: Topology,
    shell: Shell,
    wire_curve: Any,
    intersect_fn: Callable[[Any, Any], List[Intersection]],
    *,
    exclude_faces: Optional[Set[Face]] = None,
    start_vertex: Optional[Vertex] = None,
    end_vertex: Optional[Vertex] = None,
    start_point: Any = None,
    end_point: Any = None,
    start_t: float = 0.0,
    end_t: float = 1.0,
    split_curve_fn: Optional[Callable] = None,
    vertex_tol: float = 1e-10,
    wire_tol: float = 1e-10,
) -> TraceResult:
    """
    Automatically trace a wire curve across a shell, splitting all crossed
    faces.

    This is the high-level "give me a curve, handle the rest" function.
    You only need to provide the intersection callback.

    Parameters
    ----------
    topo : Topology
        The topology manager.
    shell : Shell
        The shell to cut.
    wire_curve : Any
        The wire curve geometry (opaque — passed to ``intersect_fn``).
    intersect_fn : callable(wire_curve, edge_curve) → list[Intersection]
        Your intersection function. For each pair of curves, return a list
        of Intersection objects with ``t_wire``, ``t_edge``, ``point``.

        ``t_wire``:  parameter ∈ [0, 1] along the wire curve.
        ``t_edge``:  parameter ∈ [0, 1] along the edge curve, where
                     t=0 corresponds to ``edge.vertices[0]`` and
                     t=1 corresponds to ``edge.vertices[1]``.
        ``point``:   the 3D intersection point (stored on new vertices).

    exclude_faces : set of Face, optional
        Faces to skip (e.g., the exterior backing face from create_shell).
        Their boundary edges are still intersected if shared with a
        non-excluded face.
    start_vertex / end_vertex : Vertex, optional
        If the wire begins/ends at an existing vertex rather than crossing
        an edge, pass it here. No edge split is performed for that end.
    start_point / end_point : Any, optional
        Point geometry for start/end vertex (only used if creating new
        meaning via the vertex; ignored if start/end_vertex is given).
    start_t / end_t : float
        Wire parameter for the start/end waypoint (default 0.0 / 1.0).
    split_curve_fn : callable(curve, t) → (curve_lo, curve_hi), optional
        Called when an edge is split to subdivide its curve geometry.
    vertex_tol : float
        Tolerance on t_edge for snapping to existing vertices.
        If t_edge < vertex_tol or t_edge > 1 - vertex_tol, the
        intersection is treated as coincident with the edge endpoint.
    wire_tol : float
        Tolerance on t_wire for deduplicating vertex hits.

    Returns
    -------
    TraceResult
        Contains the WireResult from add_wire, plus the ordered list of
        crossings and the face sequence.

    Algorithm
    ---------
    1. **Collect crossings**: Intersect wire_curve with every edge curve
       in the shell. Each hit produces a Waypoint (edge + t_wire + point).

    2. **Snap to vertices**: If t_edge ≈ 0 or ≈ 1, the crossing is at
       an existing vertex — no edge split needed.

    3. **Sort & dedup**: Sort waypoints by t_wire. Merge duplicate vertex
       hits (a corner vertex may be reported by multiple edges).

    4. **Prepend / append terminal vertices**: If start_vertex or
       end_vertex is given, insert them as the first/last waypoint.

    5. **Find face sequence**: For each consecutive pair of waypoints,
       find the shared face (intersection of faces touching each waypoint,
       excluding the exterior).

    6. **Build FaceSegments**: For each face crossing, pick the correct
       half-edge in that face's loop. Set enter_he/enter_vertex and
       exit_he/exit_vertex accordingly.

    7. **Call add_wire**: Hand the FaceSegment list to add_wire, which
       performs the actual splits and face divisions.

    What you provide vs. what is automated
    ----------------------------------------

    ┌────────────────────────────┬──────────────────────────────────┐
    │  YOU provide               │  trace_wire does automatically   │
    ├────────────────────────────┼──────────────────────────────────┤
    │  intersect_fn              │  walks all shell edges           │
    │  wire_curve                │  sorts crossings by t_wire       │
    │  exclude_faces (exterior)  │  snaps to existing vertices      │
    │  (opt) start/end_vertex    │  deduplicates corner hits        │
    │  (opt) split_curve_fn      │  finds face between crossings    │
    │                            │  picks correct half-edges         │
    │                            │  builds FaceSegment list          │
    │                            │  calls add_wire                   │
    └────────────────────────────┴──────────────────────────────────┘

    Example
    -------
    ::

        def my_intersect(wire_crv, edge_crv):
            hits = nurbs_intersect(wire_crv, edge_crv)
            return [Intersection(t_wire=h.t1, t_edge=h.t2, point=h.pt)
                    for h in hits]

        result = trace_wire(
            topo, shell, my_wire_curve,
            intersect_fn=my_intersect,
            exclude_faces={shell_result.face_ext},
        )
    """
    if exclude_faces is None:
        exclude_faces = set()

    # ── 1. Collect crossings ────────────────────────────────────────────
    waypoints = _collect_crossings(
        shell, wire_curve, intersect_fn, exclude_faces, vertex_tol
    )

    # ── 2. Dedup vertex hits ────────────────────────────────────────────
    waypoints = _dedup_vertex_waypoints(waypoints, wire_tol)

    # ── 3. Prepend / append terminal vertices ───────────────────────────
    if start_vertex is not None:
        wp_start = _Waypoint(
            t_wire=start_t, point=start_point or start_vertex.point,
            vertex=start_vertex,
        )
        # Only prepend if the first waypoint isn't already this vertex
        if not waypoints or waypoints[0].vertex is not start_vertex:
            waypoints.insert(0, wp_start)

    if end_vertex is not None:
        wp_end = _Waypoint(
            t_wire=end_t, point=end_point or end_vertex.point,
            vertex=end_vertex,
        )
        if not waypoints or waypoints[-1].vertex is not end_vertex:
            waypoints.append(wp_end)

    # ── Validate ────────────────────────────────────────────────────────
    if len(waypoints) < 2:
        raise ValueError(
            f"trace_wire needs at least 2 crossings (entry + exit), "
            f"found {len(waypoints)}. Check that the wire actually "
            f"crosses the shell boundary edges."
        )

    # ── 4. Find face sequence ───────────────────────────────────────────
    face_sequence: List[Face] = []
    for i in range(len(waypoints) - 1):
        face = _find_face_between(waypoints[i], waypoints[i + 1], exclude_faces)
        face_sequence.append(face)

    # ── 5. Build FaceSegments ───────────────────────────────────────────
    segments: List[FaceSegment] = []

    for i, face in enumerate(face_sequence):
        wp_in = waypoints[i]
        wp_out = waypoints[i + 1]
        seg = FaceSegment(curve=wire_curve, face=face)

        # ── Entry ───────────────────────────────────────────────────────
        if i == 0:
            # First segment: set enter explicitly
            if wp_in.vertex is not None:
                seg.enter_vertex = wp_in.vertex
            else:
                seg.enter_he = _pick_he_in_face(wp_in.edge, face)
                seg.enter_point = wp_in.point
                seg.enter_t_edge = wp_in.t_edge
        # For subsequent segments: enter is auto-carried from prev exit

        # ── Exit ────────────────────────────────────────────────────────
        if wp_out.vertex is not None:
            seg.exit_vertex = wp_out.vertex
        else:
            seg.exit_he = _pick_he_in_face(wp_out.edge, face)
            seg.exit_point = wp_out.point
            seg.exit_t_edge = wp_out.t_edge

        segments.append(seg)

    # ── 6. Call add_wire ────────────────────────────────────────────────
    wire_result = add_wire(topo, segments, split_curve_fn=split_curve_fn)

    return TraceResult(
        wire_result=wire_result,
        crossings=waypoints,
        face_sequence=face_sequence,
    )
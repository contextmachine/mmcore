"""2D Boolean operations on NURBS curves, built on top of BRep + nurbs_ccx_multiple.

All operations assume the input curves lie in the z=0 plane. Non-planar inputs
(z != 0) are silently treated as 2D projections and will produce geometrically
wrong results. This module is designed for planar sketches and 2D BRep regions
only; for curves on surfaces or 3D boolean operations, see the spec's
out-of-scope list.

See docs/superpowers/specs/2026-04-14-2d-boolean-operations-design.md for design.
"""
from __future__ import annotations

import numpy as np

from mmcore.geom._nurbs_eval import NURBSCurveTuple, evaluate_nurbs_curve
from mmcore.geom._nurbs_knots import reverse_curve, split_curve_multiple, trim_curve
from mmcore.numeric.intersection.ccx._nccx4 import nurbs_ccx_multiple
from mmcore.topo.brep import BRep


_PIP_ENDPOINT_EPS_MUL = 2.0  # u_seg must be > _PIP_ENDPOINT_EPS_MUL * tol from 0
_PIP_CROSSING_SAMPLE_DT = 1e-3  # fraction of curve parameter range for crossing-test samples


def _shoelace_signed_area(pts) -> float:
    """Signed polygon area from a sequence of 2D/3D points.
    Positive = CCW in xy plane, negative = CW. Uses the standard shoelace
    formula on the first two columns.
    """
    arr = np.asarray(pts, dtype=float)
    if arr.ndim != 2 or arr.shape[0] < 3:
        return 0.0
    xs = arr[:, 0]
    ys = arr[:, 1]
    return 0.5 * float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))


def point_in_region(
    point,
    region_curves,
    tol: float = 1e-6,
) -> bool:
    """Return True iff *point* lies strictly inside the region bounded by *region_curves*.

    Casts a single line segment (degree-1 NURBS) from *point* past the region's
    bounding box, intersects it with *region_curves* via nurbs_ccx_multiple, and
    counts transverse crossings using the even-odd rule.

    Tangent intersections are identified by parallelism between the region curve's
    tangent at the hit and the segment direction; they are ignored (parity unchanged).
    Overlaps (coincident runs) are also ignored for the same parity reason.

    Raises RuntimeError if the segment starts on a region curve (point lies on the
    region boundary — the PIP result is undefined there).
    """
    pt = np.asarray(point, dtype=float).reshape(-1)
    dim = pt.shape[0]

    # --- expanded bbox (region + query point) ---
    all_ctrl = np.concatenate([np.asarray(c.control_points, dtype=float)
                               for c in region_curves], axis=0)
    bbox_min = np.minimum(all_ctrl.min(axis=0), pt)
    bbox_max = np.maximum(all_ctrl.max(axis=0), pt)
    diag = float(np.linalg.norm(bbox_max - bbox_min))
    L = 2.0 * diag + max(1.0, diag) * 1e-3  # escape length from anywhere in the expanded bbox

    # --- direction (deterministic, single shot) ---
    theta = 0.31415
    d = np.zeros(dim, dtype=float)
    d[0] = float(np.cos(theta))
    d[1] = float(np.sin(theta))

    seg = NURBSCurveTuple(
        order=2,
        knot=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=np.array([pt, pt + L * d], dtype=float),
        weights=np.array([1.0, 1.0], dtype=float),
    )

    isolated, overlaps, ccx_status = nurbs_ccx_multiple(
        [seg] + list(region_curves), tol=tol)
    if not ccx_status['complete']:
        # Ledger L41: an incomplete ray-casting CCX must not crash the
        # boolean (the former adapter default raised RuntimeError here);
        # a truncated crossing count can misclassify containment, so
        # surface it loudly while returning the best available answer.
        import warnings
        warnings.warn(
            "boolean2d point-in-region: incomplete CCX ray cast "
            "(bounded solve truncated) — containment may be unreliable",
            RuntimeWarning, stacklevel=2)

    endpoint_eps = 1/np.linalg.norm(seg.control_points[-1]-seg.control_points[0]) * tol


    # Line equation for the segment: f(q) = (q.y - pt.y)*d.x - (q.x - pt.x)*d.y
    # f > 0, f < 0, f == 0 tells which side of the line the query point lies on.
    def _line_side(q) -> float:
        return (q[1] - pt[1]) * d[0] - (q[0] - pt[0]) * d[1]

    count = 0

    if isolated is not None:
        for rec in isolated:
            c1 = int(rec['curve1_i'])
            c2 = int(rec['curve2_i'])
            if c1 != 0 and c2 != 0:
                continue  # not involving segment

            u = float(rec['u'])
            v = float(rec['v'])
            if c1 == 0:
                u_seg = u
                t_curve = v
                curve_idx_in_region = c2 - 1  # region_curves is offset by +1
            else:
                u_seg = v
                t_curve = u
                curve_idx_in_region = c1 - 1

            # segment start lying on a region curve ⇒ point is on boundary
            if u_seg < endpoint_eps:
                raise RuntimeError(
                    f"point_in_region: point {pt.tolist()} lies on a region boundary "
                    f"(segment start intersects curve {curve_idx_in_region} at t={t_curve})"
                )

            # Crossing test by signed distance sampling:
            # Sample the curve slightly before and after t_curve. If both samples
            # lie on the same side of the segment line, the curve grazes the line
            # (tangent touch — no parity flip). If on opposite sides, it crosses
            # (transverse — flip parity).
            curve = region_curves[curve_idx_in_region]
            t_lo_curve, t_hi_curve = curve.interval()
            dt = _PIP_CROSSING_SAMPLE_DT * (t_hi_curve - t_lo_curve)
            # clamp samples to curve's valid parameter range
            t_before = max(t_curve - dt, t_lo_curve)
            t_after = min(t_curve + dt, t_hi_curve)
            # must have non-zero separation on both sides of t_curve
            if t_curve - t_before < dt * 0.1 or t_after - t_curve < dt * 0.1:
                # near curve endpoint — cannot reliably sample; fall back to
                # counting as a transverse crossing (conservative)
                count += 1
                continue

            pt_before = np.asarray(evaluate_nurbs_curve(curve, t_before, 0)['C'], dtype=float)
            pt_after = np.asarray(evaluate_nurbs_curve(curve, t_after, 0)['C'], dtype=float)

            s_before = _line_side(pt_before)
            s_after = _line_side(pt_after)

            if s_before * s_after > 0.0:
                # same side → grazing / tangent touch → no parity flip
                continue
            # opposite sides → transverse crossing
            count += 1

    # overlaps: segment lies along a region curve for a range. Ignored — they
    # contribute no parity change (you slide along the boundary, never crossing it).
    # The sub-curves on either side of the overlap either both return to the same
    # side (grazing) or cross somewhere outside the overlap (isolated hit handled
    # elsewhere). For point-in-region purposes this is safe.

    return (count % 2) == 1


# ---------------------------------------------------------------------------
#  make_region_2d — builder for 2D BRep inputs
# ---------------------------------------------------------------------------

def _signed_area_xy_samples(curves: list[NURBSCurveTuple], n_per_curve: int = 16) -> float:
    """Shoelace signed area from sampled points along the loop's curves.

    Positive ⇒ CCW in xy plane ⇒ bounds material.
    Negative ⇒ CW in xy plane ⇒ bounds a hole.
    """
    pts = []
    for crv in curves:
        t0, t1 = crv.interval()
        for i in range(n_per_curve):
            t = t0 + (t1 - t0) * (i / n_per_curve)
            ev = evaluate_nurbs_curve(crv, t, 0)
            pts.append(np.asarray(ev['C'], dtype=float))
    return _shoelace_signed_area(np.asarray(pts))


def make_region_2d(loops: list[list[NURBSCurveTuple]]) -> BRep:
    """Build a 2D BRep from a list of closed loops.

    Each inner list is one closed loop whose curves are oriented end-to-end
    (curve[i].end() ≈ curve[i+1].start() and last.end() ≈ first.start()).
    CCW loops (positive signed area in xy) become body-face outer loops —
    one body face per CCW loop. CW loops become holes attached to the
    containing body face (determined by point-in-region tests).

    Single-shell form: one Body, one Shell, one wire Face (Face 0) with
    outer=None, N body faces with outer + inners. Every half-edge has a
    valid face reference.
    """
    brep = BRep()
    body = brep.new_body(shells=[])
    shell = brep.new_shell(faces=[], body=body.id)
    body.shells.append(shell.id)
    wire_face = brep.new_face(outer=None, inners=[], shell=shell.id, surf=None)
    shell.faces.append(wire_face.id)

    # Classify each loop by signed area in xy plane.
    loops_by_type: list[tuple[str, list[NURBSCurveTuple]]] = []
    for i, loop_curves in enumerate(loops):
        area = _signed_area_xy_samples(loop_curves)
        if abs(area) < 1e-12:
            raise ValueError(
                f"make_region_2d: loop index {i} has near-zero signed area "
                f"({area:.3e}); cannot determine CCW/CW orientation "
                "(degenerate or collinear loop?)"
            )
        kind = 'outer' if area > 0 else 'hole'
        loops_by_type.append((kind, loop_curves))

    # First pass: build all outer loops (one body face per outer loop).
    outer_face_ids: list[int] = []
    for kind, loop_curves in loops_by_type:
        if kind != 'outer':
            continue
        face_id = _add_loop_to_brep(brep, shell.id, wire_face.id, loop_curves, is_body_outer=True)
        outer_face_ids.append(face_id)

    # Second pass: for each hole, find the containing body face and attach as inner.
    for kind, loop_curves in loops_by_type:
        if kind != 'hole':
            continue
        # find which body face's material contains the hole's centroid
        hole_sample = _interior_sample_of_loop(loop_curves)
        host_face_id = None
        for face_id in outer_face_ids:
            face = brep.F[face_id]
            outer_loop_curves = _loop_curves_from_loop_id(brep, face.outer)
            if point_in_region(hole_sample, outer_loop_curves, tol=1e-6):
                host_face_id = face_id
                break
        if host_face_id is None:
            raise ValueError("hole loop is not contained by any outer loop")
        _add_loop_to_brep(brep, shell.id, wire_face.id, loop_curves,
                          is_body_outer=False, host_face_id=host_face_id)

    return brep


def _interior_sample_of_loop(loop_curves: list[NURBSCurveTuple]) -> np.ndarray:
    """Return a point that's (approximately) inside the loop.

    Simple strategy: shoelace centroid of the first curve's start points and
    the midpoint of each curve. Not guaranteed interior for very non-convex
    shapes but works for the shapes we care about (squares, circles, simple
    polygons). For exotic shapes, callers should supply their own sample.
    """
    pts = []
    for crv in loop_curves:
        t0, t1 = crv.interval()
        ev = evaluate_nurbs_curve(crv, 0.5 * (t0 + t1), 0)
        pts.append(np.asarray(ev['C'], dtype=float))
    return np.mean(np.asarray(pts), axis=0)


def _loop_curves_from_loop_id(brep: BRep, loop_id: int) -> list[NURBSCurveTuple]:
    """Walk a loop's half-edges and return the list of curves it traverses."""
    curves = []
    first = brep.L[loop_id].he
    he_id = first
    while True:
        he = brep.HE[he_id]
        edge = brep.E[he.edge]
        crv = brep.G_CRV[edge.geom]
        curves.append(crv)
        he_id = he.next
        if he_id == first:
            break
    return curves


def _add_loop_to_brep(
    brep: BRep,
    shell_id: int,
    wire_face_id: int,
    loop_curves: list[NURBSCurveTuple],
    *,
    is_body_outer: bool,
    host_face_id: int | None = None,
) -> int:
    """Insert the vertices, edges, half-edges, loops (body + wire twins), and
    (if is_body_outer) a body face for one closed loop of oriented NURBS curves.

    If is_body_outer is True, creates a new body face with this loop as its
    outer loop; returns the new body face id.

    If is_body_outer is False, treats the loop as a hole to be attached to
    host_face_id as an inner loop; returns host_face_id.
    """
    n = len(loop_curves)
    if n < 1:
        raise ValueError("loop_curves must have at least one curve")

    # Create one vertex per curve start. curve[i].end() ≈ curve[(i+1)%n].start().
    vertices: list[int] = []
    for i, crv in enumerate(loop_curves):
        start = tuple(np.asarray(crv.start(), dtype=float).tolist())
        v = brep.new_vertex(point=start, tol=1e-6)
        vertices.append(v.id)

    # Determine body-side face id
    if is_body_outer:
        body_face = brep.new_face(outer=None, inners=[], shell=shell_id,
                                  same_sense=True, surf=None)
        brep.S[shell_id].faces.append(body_face.id)
        body_face_id = body_face.id
    else:
        body_face_id = host_face_id  # type: ignore[assignment]

    # Create edges + half-edges for each curve.
    body_hes: list[int] = []  # in walk order
    wire_hes: list[int] = []  # in walk order (twins, reversed-winding cycle)
    for i, crv in enumerate(loop_curves):
        v_start = vertices[i]
        v_end = vertices[(i + 1) % n]
        crv_id = brep.new_curve(crv)
        edge = brep.new_edge(v_start=v_start, v_end=v_end, geom=crv_id,
                             param=crv.interval())
        # Body-side HE walks v_start→v_end (the direction the user supplied).
        he_body = brep.new_halfedge(
            edge=edge.id, face=body_face_id, loop=None,
            vert=v_end, orient=True, pcurve=None,
        )
        # Wire-side twin walks v_end→v_start on the wire (Face 0) face.
        he_wire = brep.new_halfedge(
            edge=edge.id, face=wire_face_id, loop=None,
            vert=v_start, orient=False, pcurve=None,
        )
        he_body.twin = he_wire.id
        he_wire.twin = he_body.id
        edge.he = he_body.id
        body_hes.append(he_body.id)
        wire_hes.append(he_wire.id)

    # Link next/prev along the body loop (forward cycle).
    for i in range(n):
        brep.HE[body_hes[i]].next = body_hes[(i + 1) % n]
        brep.HE[body_hes[(i + 1) % n]].prev = body_hes[i]

    # Link next/prev along the wire loop (reverse cycle: the wire HE for
    # curve i starts at v_{i+1} and ends at v_i, so the walk order is
    # wire_hes[n-1], wire_hes[n-2], ..., wire_hes[0]).
    for i in range(n):
        nxt_i = (i - 1) % n
        brep.HE[wire_hes[i]].next = wire_hes[nxt_i]
        brep.HE[wire_hes[nxt_i]].prev = wire_hes[i]

    # Create the two loop records and tag HEs.
    body_loop = brep.new_loop(face=body_face_id, he=body_hes[0],
                              is_outer=is_body_outer)
    wire_loop = brep.new_loop(face=wire_face_id, he=wire_hes[0], is_outer=False)
    for hid in body_hes:
        brep.HE[hid].loop = body_loop.id
    for hid in wire_hes:
        brep.HE[hid].loop = wire_loop.id

    # Attach body loop to its face.
    if is_body_outer:
        brep.F[body_face_id].outer = body_loop.id
    else:
        brep.F[body_face_id].inners.append(body_loop.id)

    # The wire loop always lives in Face 0's inners list.
    brep.F[wire_face_id].inners.append(wire_loop.id)

    return body_face_id


# ---------------------------------------------------------------------------
#  Boolean op pipeline — private helpers
# ---------------------------------------------------------------------------

def _collect_curves_with_sources(
    brep_a: BRep, brep_b: BRep
) -> tuple[list[NURBSCurveTuple], list[str]]:
    """Walk both BReps' edges and return (curves, sources) lists.

    Iterates brep.E.values() directly — format agnostic to whether the BRep
    stores boundaries as body-face outer/inner loops or as Face 0 inners.
    Each edge contributes exactly one curve (the one in G_CRV trimmed to the
    edge's param range).
    """
    # validate inputs up front (fail fast on malformed BReps)
    for name, brep in (('a', brep_a), ('b', brep_b)):
        errs = brep.validate()
        if errs:
            raise ValueError(
                f"input BRep {name!r} failed validate(): {errs[0]}"
            )

    curves: list[NURBSCurveTuple] = []
    sources: list[str] = []
    for brep, tag in ((brep_a, 'A'), (brep_b, 'B')):
        for e in brep.E.values():
            if e.geom is None:
                raise ValueError(
                    f"input BRep has an edge without geometry (edge id {e.id})"
                )
            base = brep.G_CRV[e.geom]
            t0, t1 = e.param
            if (t0, t1) == base.interval():
                curves.append(base)
            else:
                curves.append(trim_curve(base, min(t0, t1), max(t0, t1)))
            sources.append(tag)
    return curves, sources


from mmcore.geom._nurbs_param_tol import nurbs_curve_param_tolerance


def _split_curves_at_intersections(
    curves: list[NURBSCurveTuple],
    sources: list[str],
    tol: float,
) -> tuple[list[NURBSCurveTuple], list[str]]:
    """Split each curve at all CCX-reported intersections and dedup overlaps.

    Returns (sub_segments, sub_sources) where each source tag is 'A', 'B', or
    'AB' (the last indicates a segment produced by merging an overlap pair).
    """
    isolated, overlaps, ccx_status = nurbs_ccx_multiple(curves, tol=tol)
    if not ccx_status['complete']:
        # Ledger L41: proceed with the certified subset instead of raising —
        # missed split points degrade the arrangement locally; the warning
        # is the honest signal until boolean2d grows its own status channel.
        import warnings
        warnings.warn(
            "boolean2d split: incomplete CCX result (bounded solve "
            "truncated) — some intersection splits may be missing",
            RuntimeWarning, stacklevel=2)

    # Per-curve list of split parameters (including the overlap range
    # endpoints — overlaps must cause splits at both ends).
    split_params: list[list[float]] = [[] for _ in curves]
    if isolated is not None:
        for rec in isolated:
            c1, c2 = int(rec['curve1_i']), int(rec['curve2_i'])
            u, v = float(rec['u']), float(rec['v'])
            split_params[c1].append(u)
            split_params[c2].append(v)
    if overlaps is not None:
        for rec in overlaps:
            c1, c2 = int(rec['curve1_i']), int(rec['curve2_i'])
            u0, u1 = float(rec['u'][0]), float(rec['u'][1])
            v0, v1 = float(rec['v'][0]), float(rec['v'][1])
            split_params[c1].extend([u0, u1])
            split_params[c2].extend([v0, v1])

    # Dedupe each curve's params using parametric tolerance; drop boundary
    # params (split_curve_multiple rejects params equal to the curve domain
    # endpoints, and a split at a boundary would produce a zero-length piece
    # anyway).
    dedup_params: list[list[float]] = []
    for i, params in enumerate(split_params):
        t_lo, t_hi = curves[i].interval()
        if not params:
            # Closed, non-intersecting curve case: if the curve's endpoints
            # coincide (to within ``tol``) and no split params were reported,
            # ``_build_arrangement`` would see ``v0 == v1`` and silently drop
            # the entire segment — losing a whole connected component (e.g.
            # a small circle nested inside a big square with no crossings).
            # Inject a midpoint split so the closed curve becomes two
            # half-edges linked end-to-end through a fresh vertex.
            start = np.asarray(
                evaluate_nurbs_curve(curves[i], t_lo, 0)['C'], dtype=float
            )
            end = np.asarray(
                evaluate_nurbs_curve(curves[i], t_hi, 0)['C'], dtype=float
            )
            if float(np.linalg.norm(end - start)) < tol:
                dedup_params.append([0.5 * (t_lo + t_hi)])
            else:
                dedup_params.append([])
            continue
        ptol = float(nurbs_curve_param_tolerance(curves[i], tol))
        params.sort()
        kept: list[float] = []
        for p in params:
            # drop params on or near the curve boundary
            if p - t_lo <= ptol or t_hi - p <= ptol:
                continue
            if not kept or p - kept[-1] > ptol:
                kept.append(p)
        dedup_params.append(kept)

    # Split each curve. split_curve_multiple returns [curve] if params is empty.
    all_sub_segs: list[list[NURBSCurveTuple]] = []
    for i, crv in enumerate(curves):
        params = dedup_params[i]
        if params:
            pieces = split_curve_multiple(crv, params)
        else:
            pieces = [crv]
        all_sub_segs.append(list(pieces))

    # Dedupe overlap sub-segments: for each overlap, the piece on curve c1
    # between u0 and u1 is geometrically the same as the piece on curve c2
    # between v0 and v1. Keep one, mark source as 'AB', discard the other.
    killed: set[tuple[int, int]] = set()
    upgraded: set[tuple[int, int]] = set()

    def _find_sub_index_spanning(
        crv_idx: int,
        params: list[float],
        base_interval: tuple[float, float],
        u0: float,
        u1: float,
    ) -> int | None:
        """Find the sub-segment index whose param range is approximately [u0,u1]."""
        t_lo, t_hi = base_interval
        boundaries = [t_lo] + list(params) + [t_hi]
        lo_target, hi_target = min(u0, u1), max(u0, u1)
        # use a parametric tolerance scaled by the curve's param tolerance so
        # that CCX's overlap ranges match even after dedup-induced snapping.
        match_tol = max(50.0 * tol,
                        10.0 * float(nurbs_curve_param_tolerance(curves[crv_idx], tol)))
        for k in range(len(boundaries) - 1):
            bk_lo, bk_hi = boundaries[k], boundaries[k + 1]
            if abs(bk_lo - lo_target) < match_tol and abs(bk_hi - hi_target) < match_tol:
                return k
        return None

    if overlaps is not None:
        for rec in overlaps:
            c1, c2 = int(rec['curve1_i']), int(rec['curve2_i'])
            u0, u1 = float(rec['u'][0]), float(rec['u'][1])
            v0, v1 = float(rec['v'][0]), float(rec['v'][1])
            k1 = _find_sub_index_spanning(c1, dedup_params[c1], curves[c1].interval(), u0, u1)
            k2 = _find_sub_index_spanning(c2, dedup_params[c2], curves[c2].interval(), v0, v1)
            if k1 is None or k2 is None:
                continue
            if (c1, k1) in killed or (c2, k2) in upgraded:
                # Already-swapped pair from a symmetric overlap entry; skip.
                continue
            upgraded.add((c1, k1))
            killed.add((c2, k2))

    # Flatten into output lists, applying killed/upgraded sets.
    out_segs: list[NURBSCurveTuple] = []
    out_sources: list[str] = []
    for i, pieces in enumerate(all_sub_segs):
        for k, piece in enumerate(pieces):
            if (i, k) in killed:
                continue
            if (i, k) in upgraded:
                out_sources.append('AB')
            else:
                out_sources.append(sources[i])
            out_segs.append(piece)

    return out_segs, out_sources


from dataclasses import dataclass, field


@dataclass
class _ArrHalfEdge:
    idx: int                     # position in the half-edges list
    seg_idx: int                 # sub-segment this HE corresponds to
    forward: bool                # True if walks in segment's natural direction
    origin_vid: int              # tail vertex id (where the HE starts)
    head_vid: int                # head vertex id (where the HE ends)
    angle: float                 # outgoing tangent angle at origin, in (-π, π]
    twin: int | None = None
    next: int | None = None
    prev: int | None = None
    face: int | None = None
    ccw_prev: int | None = None  # helper link: previous HE CCW around origin
    sources: set[str] = field(default_factory=set)


@dataclass
class _ArrFace:
    idx: int
    hes: list[int]  # half-edges forming this face's boundary cycle
    unbounded: bool = False


@dataclass
class _Arrangement:
    vertices: list[np.ndarray]          # vid → xy point
    sub_segments: list[NURBSCurveTuple] # seg_idx → NURBSCurveTuple
    sources: list[str]                  # seg_idx → 'A' | 'B' | 'AB'
    half_edges: list[_ArrHalfEdge]
    faces: list[_ArrFace]


def _build_arrangement(
    sub_segments: list[NURBSCurveTuple],
    sub_sources: list[str],
    tol: float,
) -> _Arrangement:
    """Lightweight in-memory DCEL for the noded + dedup'd sub-segments."""
    # 1) Vertex pool: grid-hash the endpoints of every sub-segment.
    vertices: list[np.ndarray] = []
    vid_of: dict[tuple[int, int], int] = {}

    def _vid(p: np.ndarray) -> int:
        key = (round(float(p[0]) / tol), round(float(p[1]) / tol))
        if key not in vid_of:
            vid_of[key] = len(vertices)
            vertices.append(np.asarray(p, dtype=float))
        return vid_of[key]

    # 2) Build HEs with angles.
    half_edges: list[_ArrHalfEdge] = []
    for seg_idx, seg in enumerate(sub_segments):
        t0, t1 = seg.interval()
        start_ev = evaluate_nurbs_curve(seg, t0, 1)
        end_ev = evaluate_nurbs_curve(seg, t1, 1)
        p0 = np.asarray(start_ev['C'], dtype=float)
        p1 = np.asarray(end_ev['C'], dtype=float)
        t0_vec = np.asarray(start_ev['C1'], dtype=float)
        t1_vec = np.asarray(end_ev['C1'], dtype=float)

        v0 = _vid(p0)
        v1 = _vid(p1)
        if v0 == v1:
            # Degenerate (start==end vertex). Skip this segment — it contributes
            # nothing to the arrangement. Shouldn't happen for well-formed input.
            continue

        ang_fwd = float(np.arctan2(t0_vec[1], t0_vec[0]))
        # Reverse HE's outgoing tangent at v1 is -t1_vec.
        ang_rev = float(np.arctan2(-t1_vec[1], -t1_vec[0]))

        fwd_idx = len(half_edges)
        rev_idx = fwd_idx + 1
        sources = {sub_sources[seg_idx]} if sub_sources[seg_idx] != 'AB' else {'A', 'B'}

        half_edges.append(_ArrHalfEdge(
            idx=fwd_idx, seg_idx=seg_idx, forward=True,
            origin_vid=v0, head_vid=v1, angle=ang_fwd,
            twin=rev_idx, sources=sources,
        ))
        half_edges.append(_ArrHalfEdge(
            idx=rev_idx, seg_idx=seg_idx, forward=False,
            origin_vid=v1, head_vid=v0, angle=ang_rev,
            twin=fwd_idx, sources=sources,
        ))

    # 3) For each vertex, sort outgoing HEs CCW by angle.
    outgoing: dict[int, list[int]] = {}
    for he in half_edges:
        outgoing.setdefault(he.origin_vid, []).append(he.idx)
    for vid, hids in outgoing.items():
        hids.sort(key=lambda i: half_edges[i].angle)
        m = len(hids)
        for j in range(m):
            half_edges[hids[(j + 1) % m]].ccw_prev = hids[j]

    # 4) Link next = twin.ccw_prev  (standard "face on left" rule).
    for he in half_edges:
        twin = half_edges[he.twin]
        he.next = twin.ccw_prev
        half_edges[he.next].prev = he.idx

    # 5) Walk loops to enumerate faces.
    faces: list[_ArrFace] = []
    for he in half_edges:
        if he.face is not None:
            continue
        fidx = len(faces)
        cycle: list[int] = []
        cur = he.idx
        while half_edges[cur].face is None:
            half_edges[cur].face = fidx
            cycle.append(cur)
            cur = half_edges[cur].next
            if cur == he.idx:
                break
        faces.append(_ArrFace(idx=fidx, hes=cycle))

    # 6) Identify the unbounded face using signed area of the actual curve
    #    samples along each face's boundary cycle. The unbounded face walks
    #    each island's outer boundary CLOCKWISE (from outside looking at
    #    the island), so its signed area is strongly negative (sum of the
    #    inverted shapes of all interior islands). Bounded faces always
    #    have positive signed area under the "face-on-left" convention.
    #    Sampling along the curve (not just the endpoints) handles curved
    #    boundaries whose tangent at the extreme vertex doesn't reflect
    #    the actual orientation of the face.
    if not vertices:
        return _Arrangement(vertices=vertices, sub_segments=list(sub_segments),
                            sources=list(sub_sources), half_edges=half_edges, faces=faces)

    def _face_sampled_points(face: _ArrFace, n_samples: int = 16) -> list[np.ndarray]:
        pts: list[np.ndarray] = []
        for hid in face.hes:
            he = half_edges[hid]
            seg = sub_segments[he.seg_idx]
            t0, t1 = seg.interval()
            if he.forward:
                ts = np.linspace(t0, t1, n_samples)
            else:
                ts = np.linspace(t1, t0, n_samples)
            for k, t in enumerate(ts):
                if k == n_samples - 1:
                    # skip last point to avoid duplicating vertex with next HE
                    continue
                ev = evaluate_nurbs_curve(seg, float(t), 0)
                pts.append(np.asarray(ev['C'], dtype=float))
        return pts

    def _face_sampled_signed_area(face: _ArrFace) -> float:
        return _shoelace_signed_area(_face_sampled_points(face))

    face_areas = [_face_sampled_signed_area(f) for f in faces]
    if faces:
        unb_idx = int(np.argmin(face_areas))
        if face_areas[unb_idx] >= 0.0:
            raise RuntimeError(
                f"_build_arrangement: no face has negative signed area — "
                f"arrangement has no identifiable unbounded face (min area = "
                f"{face_areas[unb_idx]:.3e}). This indicates a topology bug in "
                f"face enumeration or a degenerate input."
            )
        faces[unb_idx].unbounded = True

    # ----- Merge pseudo-unbounded faces into their enclosing bounded faces -----
    # A face with negative signed area is an "exterior" walk of a connected
    # component (HE cycle winds clockwise in xy). Exactly one such face is the
    # true unbounded face of the arrangement (the most-negative one, already
    # tagged above). Every OTHER negative-area face is a DCEL artifact of a
    # disconnected component: it represents the same geometric region as some
    # enclosing bounded face. Merge by remapping all its HEs to that enclosing
    # face, so downstream classification and island extraction treat the
    # enclosing face's material as one unified region whose boundary
    # includes the component's (now-hole) loop.
    def _polygon_pip(sample: np.ndarray, polygon_pts: list[np.ndarray]) -> bool:
        """Classic ray-casting point-in-polygon test (2D)."""
        n = len(polygon_pts)
        if n < 3:
            return False
        inside = False
        sx, sy = float(sample[0]), float(sample[1])
        for i in range(n):
            x1, y1 = float(polygon_pts[i][0]), float(polygon_pts[i][1])
            x2, y2 = float(polygon_pts[(i + 1) % n][0]), float(polygon_pts[(i + 1) % n][1])
            if ((y1 > sy) != (y2 > sy)):
                x_cross = x1 + (sy - y1) * (x2 - x1) / (y2 - y1)
                if sx < x_cross:
                    inside = not inside
        return inside

    def _face_interior_sample(face: _ArrFace) -> np.ndarray | None:
        """Pick a point on the face-on-left side of any HE of this face."""
        for hid in face.hes:
            he = half_edges[hid]
            seg = sub_segments[he.seg_idx]
            t0, t1 = seg.interval()
            t_mid = 0.5 * (t0 + t1)
            ev = evaluate_nurbs_curve(seg, t_mid, 1)
            mid = np.asarray(ev['C'], dtype=float)
            tan = np.asarray(ev['C1'], dtype=float)
            if not he.forward:
                tan = -tan
            # face-on-left normal in xy plane: rotate tan 90° CCW → (-ty, tx)
            n = np.array([-tan[1], tan[0], 0.0], dtype=float)
            nn = float(np.linalg.norm(n))
            if nn < 1e-30:
                continue
            n = n / nn
            # small offset relative to the face's bbox diagonal so the sample
            # lands strictly inside the face's geometric region
            face_pts = _face_sampled_points(face, n_samples=6)
            if not face_pts:
                continue
            fp = np.asarray(face_pts)
            bbox_diag = float(np.linalg.norm(fp.max(axis=0) - fp.min(axis=0)))
            eps = max(tol * 100.0, min(bbox_diag * 1e-3, 1e-3))
            return mid + eps * n
        return None

    face_polygons: list[list[np.ndarray]] = [
        _face_sampled_points(f) for f in faces
    ]

    # Build containment: for each pseudo-unbounded face, find the smallest
    # positive-area face that contains its sample point.
    positive_face_indices = [
        i for i, a in enumerate(face_areas) if a > 0.0
    ]
    pseudo_unbounded = [
        i for i, a in enumerate(face_areas)
        if a < 0.0 and i != unb_idx
    ]

    merge_map: dict[int, int] = {}  # pseudo_face_idx -> target_face_idx
    for pf_idx in pseudo_unbounded:
        sample = _face_interior_sample(faces[pf_idx])
        if sample is None:
            continue
        # pick the smallest-area positive face whose polygon contains sample
        containing: tuple[float, int] | None = None
        for cand_idx in positive_face_indices:
            if cand_idx == pf_idx:
                continue
            poly = face_polygons[cand_idx]
            if not poly:
                continue
            if _polygon_pip(sample, poly):
                area_c = face_areas[cand_idx]
                if containing is None or area_c < containing[0]:
                    containing = (area_c, cand_idx)
        if containing is not None:
            merge_map[pf_idx] = containing[1]

    if merge_map:
        # Resolve chains so every pseudo face maps to a terminal real face.
        def _resolve(x: int) -> int:
            seen: set[int] = set()
            while x in merge_map:
                if x in seen:
                    break
                seen.add(x)
                x = merge_map[x]
            return x

        resolved: dict[int, int] = {k: _resolve(k) for k in merge_map}

        # Remap HEs and extend target faces' hes lists; clear the merged face.
        for src_idx, tgt_idx in resolved.items():
            src_face = faces[src_idx]
            tgt_face = faces[tgt_idx]
            for hid in src_face.hes:
                half_edges[hid].face = tgt_idx
            tgt_face.hes.extend(src_face.hes)
            src_face.hes = []

    return _Arrangement(
        vertices=vertices,
        sub_segments=list(sub_segments),
        sources=list(sub_sources),
        half_edges=half_edges,
        faces=faces,
    )


def _classify_faces(
    arr: _Arrangement,
    curves_a: list[NURBSCurveTuple],
    curves_b: list[NURBSCurveTuple],
    tol: float,
) -> dict[int, tuple[bool, bool]]:
    """Assign (inA, inB) to each face in the arrangement.

    For the unbounded face, returns (False, False) by definition.
    For each bounded face, picks an interior sample from any half-edge and
    runs point_in_region against the original A and B curves. The sample is
    offset along the inward normal by a distance that is adaptive to the
    local segment's chord length — this keeps the sample well away from the
    boundary (so point_in_region's boundary-rejection check doesn't fire)
    while remaining inside the same face even for small/curved features.
    """
    labels: dict[int, tuple[bool, bool]] = {}
    for face in arr.faces:
        if face.unbounded:
            labels[face.idx] = (False, False)
            continue
        if not face.hes:
            labels[face.idx] = (False, False)
            continue

        # Try multiple half-edges of the face in case the first one's sample
        # lands on a neighbouring boundary — in tight features the normal
        # offset may overshoot. We loop through the face's HEs and also
        # try progressively smaller offsets.
        inA: bool | None = None
        inB: bool | None = None
        for hid in face.hes:
            he = arr.half_edges[hid]
            seg = arr.sub_segments[he.seg_idx]
            t0, t1 = seg.interval()
            t_mid = 0.5 * (t0 + t1)
            ev = evaluate_nurbs_curve(seg, t_mid, 1)
            mid = np.asarray(ev['C'], dtype=float)
            tan = np.asarray(ev['C1'], dtype=float)
            if not he.forward:
                tan = -tan
            # inward normal = tan rotated 90° CCW = (-ty, tx) in xy plane
            n = np.array([-tan[1], tan[0], 0.0], dtype=float)
            nn = float(np.linalg.norm(n))
            if nn < 1e-30:
                continue
            n = n / nn

            # Adaptive offset: use a fraction of the segment's chord length,
            # but keep it large enough for point_in_region's boundary test
            # (which rejects samples within ~2*tol of the segment start).
            start_ev = evaluate_nurbs_curve(seg, t0, 0)
            end_ev = evaluate_nurbs_curve(seg, t1, 0)
            p0 = np.asarray(start_ev['C'], dtype=float)
            p1 = np.asarray(end_ev['C'], dtype=float)
            chord = float(np.linalg.norm(p1 - p0))
            base_eps = max(tol * 1000.0, min(chord * 0.25, 1.0))

            for scale in (1.0, 0.1, 0.01):
                eps = base_eps * scale
                sample = mid + eps * n
                try:
                    cand_A = point_in_region(sample, curves_a, tol=tol) if curves_a else False
                except RuntimeError:
                    cand_A = None
                try:
                    cand_B = point_in_region(sample, curves_b, tol=tol) if curves_b else False
                except RuntimeError:
                    cand_B = None
                if cand_A is not None and cand_B is not None:
                    inA, inB = cand_A, cand_B
                    break
            if inA is not None and inB is not None:
                break

        if inA is None or inB is None:
            raise RuntimeError(
                f"_classify_faces: could not compute (inA, inB) for face {face.idx} "
                f"after exhausting interior-sample retries — all samples hit a "
                f"boundary. This indicates a degenerate face or insufficient sample "
                f"diversity; investigate before silently corrupting downstream results."
            )
        labels[face.idx] = (inA, inB)
    return labels


def _select_kept_faces(
    arr: _Arrangement,
    labels: dict[int, tuple[bool, bool]],
    op: str,
) -> set[int]:
    """Apply the op rule. Returns a set of bounded face ids that are kept."""
    rules = {
        'union':        lambda inA, inB: inA or inB,
        'intersection': lambda inA, inB: inA and inB,
        'difference':   lambda inA, inB: inA and not inB,
        'xor':          lambda inA, inB: inA != inB,
    }
    if op not in rules:
        raise ValueError(f"unknown op {op!r}")
    rule = rules[op]
    kept: set[int] = set()
    for face in arr.faces:
        if face.unbounded:
            continue
        inA, inB = labels[face.idx]
        if rule(inA, inB):
            kept.add(face.idx)
    return kept


def _extract_island_loops(
    arr: _Arrangement,
    kept: set[int],
) -> list[tuple[list[int], list[list[int]]]]:
    """Group kept faces into islands and extract their boundary loops.

    Returns a list of (outer_loop_hes, [hole_loop_hes, ...]) tuples. Each
    loop is a list of half-edge indices forming a closed cycle, with the
    body material on the LEFT of the walk direction (so outer loops are
    CCW in xy plane, hole loops are CW).
    """
    parent = {f.idx: f.idx for f in arr.faces if f.idx in kept}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for he in arr.half_edges:
        if he.face is None or he.face not in kept:
            continue
        twin = arr.half_edges[he.twin]
        if twin.face is not None and twin.face in kept:
            _union(he.face, twin.face)

    island_of: dict[int, list[int]] = {}
    for fid in kept:
        r = _find(fid)
        island_of.setdefault(r, []).append(fid)

    islands_out: list[tuple[list[int], list[list[int]]]] = []
    for root, face_ids in island_of.items():
        face_set = set(face_ids)
        boundary_hes: set[int] = set()
        for he in arr.half_edges:
            if he.face in face_set:
                twin = arr.half_edges[he.twin]
                if twin.face not in face_set:
                    boundary_hes.add(he.idx)

        visited: set[int] = set()
        loops_hes: list[list[int]] = []
        for start in list(boundary_hes):
            if start in visited:
                continue
            cycle: list[int] = []
            cur = start
            while cur not in visited:
                visited.add(cur)
                cycle.append(cur)
                twin_idx = arr.half_edges[cur].twin
                # Walk CW around cur.head_vid via ccw_prev until we find
                # the next boundary HE (first candidate: twin_idx.ccw_prev).
                nxt = arr.half_edges[twin_idx].ccw_prev
                safety = 0
                while nxt is not None and nxt not in boundary_hes:
                    if nxt == twin_idx:
                        nxt = None
                        break
                    nxt = arr.half_edges[nxt].ccw_prev
                    safety += 1
                    if safety > len(arr.half_edges):
                        nxt = None
                        break
                if nxt is None:
                    break
                cur = nxt
                if cur == start:
                    break
            # Verify the cycle actually closed back to start.
            if cycle and arr.half_edges[cycle[-1]].head_vid != arr.half_edges[cycle[0]].origin_vid:
                raise RuntimeError(
                    f"_extract_island_loops: boundary walk starting from HE {start} "
                    f"did not close (cycle length {len(cycle)}, head at "
                    f"v{arr.half_edges[cycle[-1]].head_vid}, expected origin at "
                    f"v{arr.half_edges[cycle[0]].origin_vid}). Likely cause: "
                    f"degree-1 boundary vertex or missing ccw_prev link."
                )
            loops_hes.append(cycle)

        def _loop_signed_area(loop: list[int]) -> float:
            pts = []
            for hid in loop:
                he = arr.half_edges[hid]
                pts.append(arr.vertices[he.origin_vid])
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            return 0.5 * float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))

        if not loops_hes:
            continue
        areas = [abs(_loop_signed_area(l)) for l in loops_hes]
        outer_idx = max(range(len(loops_hes)), key=lambda i: areas[i])
        outer_loop = loops_hes[outer_idx]
        hole_loops = [loops_hes[i] for i in range(len(loops_hes)) if i != outer_idx]
        islands_out.append((outer_loop, hole_loops))

    return islands_out


def _materialize_result(
    arr: _Arrangement,
    islands: list[tuple[list[int], list[list[int]]]],
) -> BRep:
    """Build a fresh BRep in standard 2D form from the arrangement islands."""
    result = BRep()
    body = result.new_body(shells=[])
    shell = result.new_shell(faces=[], body=body.id)
    body.shells.append(shell.id)
    wire_face = result.new_face(outer=None, inners=[], shell=shell.id, surf=None)
    shell.faces.append(wire_face.id)

    for outer_loop_hes, hole_loops_hes in islands:
        outer_curves = [
            _oriented_subcurve_from_arr(arr, hid) for hid in outer_loop_hes
        ]
        hole_curves_list = [
            [_oriented_subcurve_from_arr(arr, hid) for hid in hole]
            for hole in hole_loops_hes
        ]
        body_face_id = _add_loop_to_brep(
            result, shell.id, wire_face.id, outer_curves,
            is_body_outer=True,
        )
        for hole_curves in hole_curves_list:
            _add_loop_to_brep(
                result, shell.id, wire_face.id, hole_curves,
                is_body_outer=False, host_face_id=body_face_id,
            )
    errs = result.validate()
    if errs:
        raise RuntimeError(
            f"_materialize_result: output BRep failed validate(): {errs[0]}. "
            f"This is a bug in island extraction or loop assembly."
        )
    return result


def _oriented_subcurve_from_arr(arr: _Arrangement, he_idx: int) -> NURBSCurveTuple:
    """Return the sub-segment curve oriented along the HE's walk direction."""
    he = arr.half_edges[he_idx]
    seg = arr.sub_segments[he.seg_idx]
    return seg if he.forward else reverse_curve(seg)


# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def _boolean2d(a: BRep, b: BRep, op: str, tol: float) -> BRep:
    """Run the full pipeline for a single operation.

    Both inputs may be empty (Body + Shell + wire Face only). If both are
    empty the result is an empty BRep. Otherwise the pipeline runs normally
    — nurbs_ccx_multiple returns empty results for single-source inputs, the
    arrangement is still built, and classification/selection proceed as usual.
    """
    # Validate inputs
    for name, brep in (('a', a), ('b', b)):
        errs = brep.validate()
        if errs:
            raise ValueError(f"input BRep {name!r} failed validate(): {errs[0]}")

    # Curves + sources
    curves, sources = _collect_curves_with_sources(a, b)
    curves_a = [c for c, s in zip(curves, sources) if s == 'A']
    curves_b = [c for c, s in zip(curves, sources) if s == 'B']

    # Both empty ⇒ empty result (no curves to build any arrangement).
    if not curves_a and not curves_b:
        return _empty_result_brep()

    # Split + dedup (handles empty isolated/overlaps naturally)
    sub_segs, sub_sources = _split_curves_at_intersections(curves, sources, tol)

    # Build the scratch arrangement
    arr = _build_arrangement(sub_segs, sub_sources, tol)

    # Classify every bounded face (inA, inB)
    labels = _classify_faces(arr, curves_a, curves_b, tol)

    # Apply the op rule
    kept = _select_kept_faces(arr, labels, op)

    # Group into islands
    islands = _extract_island_loops(arr, kept)

    # Materialize the result
    return _materialize_result(arr, islands)


def _empty_result_brep() -> BRep:
    """Empty 2D BRep in standard form: body + shell + wire face, no body faces."""
    brep = BRep()
    body = brep.new_body(shells=[])
    shell = brep.new_shell(faces=[], body=body.id)
    body.shells.append(shell.id)
    wire = brep.new_face(outer=None, inners=[], shell=shell.id, surf=None)
    shell.faces.append(wire.id)
    return brep


def union(a: BRep, b: BRep, tol: float = 1e-6) -> BRep:
    """Union of two 2D regions."""
    return _boolean2d(a, b, 'union', tol)


def intersection(a: BRep, b: BRep, tol: float = 1e-6) -> BRep:
    """Intersection of two 2D regions."""
    return _boolean2d(a, b, 'intersection', tol)


def difference(a: BRep, b: BRep, tol: float = 1e-6) -> BRep:
    """A \\ B (region in A but not in B)."""
    return _boolean2d(a, b, 'difference', tol)


def xor(a: BRep, b: BRep, tol: float = 1e-6) -> BRep:
    """Symmetric difference (in A XOR in B)."""
    return _boolean2d(a, b, 'xor', tol)

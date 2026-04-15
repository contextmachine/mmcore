"""2D Boolean operations on NURBS curves, built on top of BRep + nurbs_ccx_multiple.

See docs/superpowers/specs/2026-04-14-2d-boolean-operations-design.md for design.
"""
from __future__ import annotations

import numpy as np

from mmcore.geom._nurbs_eval import NURBSCurveTuple, evaluate_nurbs_curve
from mmcore.numeric.intersection.ccx._nccx4 import nurbs_ccx_multiple
from mmcore.topo.brep import BRep


_PIP_ENDPOINT_EPS_MUL = 2.0  # u_seg must be > _PIP_ENDPOINT_EPS_MUL * tol from 0
_PIP_CROSSING_SAMPLE_DT = 1e-3  # fraction of curve parameter range for crossing-test samples


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

    isolated, overlaps = nurbs_ccx_multiple([seg] + list(region_curves), tol=tol)

    endpoint_eps = _PIP_ENDPOINT_EPS_MUL * tol

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
    pts = np.asarray(pts)
    xs = pts[:, 0]
    ys = pts[:, 1]
    return 0.5 * float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))


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
                from mmcore.geom._nurbs_knots import trim_curve
                curves.append(trim_curve(base, min(t0, t1), max(t0, t1)))
            sources.append(tag)
    return curves, sources


from mmcore.geom._nurbs_knots import split_curve_multiple
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
    isolated, overlaps = nurbs_ccx_multiple(curves, tol=tol)

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
        if not params:
            dedup_params.append([])
            continue
        ptol = float(nurbs_curve_param_tolerance(curves[i], tol))
        t_lo, t_hi = curves[i].interval()
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

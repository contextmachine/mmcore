"""Integration tests for 2D Boolean operations and make_region_2d."""
from __future__ import annotations

import numpy as np
import pytest  # noqa: F401  # used from Task 2 onward (pytest.raises)

from mmcore.construction import circle  # noqa: F401  # used from Task 2 onward
from mmcore.nurbs._nurbs_eval import NURBSCurveTuple
from mmcore.topo.brep import BRep
from mmcore.topo.brep.boolean2d import (
    difference,
    intersection,
    make_region_2d,
    union,
    xor,
)


def _line(p0, p1) -> NURBSCurveTuple:
    """Degree-1 NURBS segment from p0 to p1 (3D points, z=0)."""
    return NURBSCurveTuple(
        order=2,
        knot=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=np.array([p0, p1], dtype=float),
        weights=np.array([1.0, 1.0], dtype=float),
    )


def _square_ccw(x0, y0, side) -> list[NURBSCurveTuple]:
    """CCW square boundary, 4 line segments."""
    return [
        _line([x0,        y0,        0.0], [x0 + side, y0,        0.0]),
        _line([x0 + side, y0,        0.0], [x0 + side, y0 + side, 0.0]),
        _line([x0 + side, y0 + side, 0.0], [x0,        y0 + side, 0.0]),
        _line([x0,        y0 + side, 0.0], [x0,        y0,        0.0]),
    ]


def _count_body_faces(brep: BRep) -> int:
    return sum(1 for f in brep.F.values() if f.outer is not None)


def test_make_region_2d_unit_square_creates_one_body_face():
    region = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    assert _count_body_faces(region) == 1
    # body face has 1 outer loop, 0 inners
    body_face = next(f for f in region.F.values() if f.outer is not None)
    assert body_face.outer is not None
    assert body_face.inners == []
    assert body_face.surf is None
    # Face 0 exists with outer=None and inners holding the twin of the outer
    wire_face = next(f for f in region.F.values() if f.outer is None)
    assert len(wire_face.inners) == 1
    # Topology is internally consistent
    assert region.validate() == []


def test_make_region_2d_empty_loops_list_produces_valid_empty_brep():
    """A region built from zero loops should still be a valid, internally
    consistent BRep — Body + Shell + wire Face with no body faces."""
    region = make_region_2d([])
    assert _count_body_faces(region) == 0
    # exactly one face (the wire Face 0) with outer=None
    faces = list(region.F.values())
    assert len(faces) == 1
    assert faces[0].outer is None
    assert faces[0].inners == []
    assert region.validate() == []


def test_make_region_2d_two_disjoint_squares():
    region = make_region_2d([
        _square_ccw(0.0, 0.0, 1.0),
        _square_ccw(2.0, 0.0, 1.0),
    ])
    assert _count_body_faces(region) == 2
    wire_face = next(f for f in region.F.values() if f.outer is None)
    # 2 body-face outer loops ⇒ 2 wire-twin inner loops
    assert len(wire_face.inners) == 2
    assert region.validate() == []


def test_make_region_2d_square_with_hole():
    outer = _square_ccw(0.0, 0.0, 4.0)
    # CW hole in the middle (reverse order of a CCW small square)
    hole = list(reversed(_square_ccw(1.5, 1.5, 1.0)))
    # also reverse the endpoints of each segment so curves are correctly oriented
    hole = [
        _line(crv.control_points[-1], crv.control_points[0])
        for crv in hole
    ]
    region = make_region_2d([outer, hole])
    assert _count_body_faces(region) == 1
    body_face = next(f for f in region.F.values() if f.outer is not None)
    assert len(body_face.inners) == 1  # the hole was attached
    # Face 0 has 2 inners: twin of outer loop + twin of hole loop
    wire_face = next(f for f in region.F.values() if f.outer is None)
    assert len(wire_face.inners) == 2
    assert region.validate() == []


def test_make_region_2d_hole_orientation_detection():
    """Orientation is auto-detected from signed area, regardless of input order."""
    # A lone CW loop with no enclosing outer ⇒ ValueError
    cw_square = list(reversed(_square_ccw(10.0, 10.0, 1.0)))
    cw_square = [
        _line(crv.control_points[-1], crv.control_points[0])
        for crv in cw_square
    ]
    with pytest.raises(ValueError, match="not contained"):
        make_region_2d([cw_square])


from mmcore.topo.brep.boolean2d import _collect_curves_with_sources


def test_collect_curves_with_sources_from_two_regions():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    curves, sources = _collect_curves_with_sources(a, b)
    # each square has 4 edges ⇒ 8 curves total
    assert len(curves) == 8
    assert sources == ['A'] * 4 + ['B'] * 4
    # every curve is a NURBSCurveTuple
    for c in curves:
        assert isinstance(c, NURBSCurveTuple)


from mmcore.topo.brep.boolean2d import _split_curves_at_intersections


def test_split_two_overlapping_squares_produces_correct_segment_count():
    """Two unit squares, one at (0,0) and one at (0.5, 0.5). The boundaries
    intersect transversely; every curve that gets cut should produce more
    sub-segments than the original, and no overlap tags should appear.
    """
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    curves, sources = _collect_curves_with_sources(a, b)
    sub_segs, sub_sources = _split_curves_at_intersections(curves, sources, tol=1e-6)
    # More segments than curves (some got split)
    assert len(sub_segs) > len(curves)
    # Source tags are single letters (no overlap dedup happened)
    for s in sub_sources:
        assert s in ('A', 'B')


def test_split_two_squares_sharing_one_edge_merges_overlap():
    """Two unit squares that share the edge x=1 from y=0 to y=1.
    CCX returns that shared segment as an overlap. After dedup, there must
    be exactly one sub-segment tagged 'AB' for the shared portion.
    """
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(1.0, 0.0, 1.0)])
    curves, sources = _collect_curves_with_sources(a, b)
    sub_segs, sub_sources = _split_curves_at_intersections(curves, sources, tol=1e-6)
    both = [s for s in sub_sources if s == 'AB']
    assert len(both) == 1


from mmcore.topo.brep.boolean2d import _build_arrangement


def test_build_arrangement_two_overlapping_squares_has_expected_face_count():
    """Two unit squares at (0,0) and (0.5, 0.5). Precise arrangement face count
    depends on how many arrangement faces emerge; assert at least 3 bounded
    plus exactly 1 unbounded, and that every HE has a face assigned.
    """
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    curves, sources = _collect_curves_with_sources(a, b)
    sub_segs, sub_sources = _split_curves_at_intersections(curves, sources, tol=1e-6)
    arr = _build_arrangement(sub_segs, sub_sources, tol=1e-6)
    bounded_count = sum(1 for f in arr.faces if not f.unbounded)
    assert bounded_count >= 3
    for he in arr.half_edges:
        assert he.face is not None
    assert sum(1 for f in arr.faces if f.unbounded) == 1


from mmcore.topo.brep.boolean2d import _classify_faces


def test_classify_faces_two_overlapping_squares_gives_expected_labels():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    curves_a, _ = _collect_curves_with_sources(a, BRep())
    curves_b, _ = _collect_curves_with_sources(BRep(), b)
    curves, sources = _collect_curves_with_sources(a, b)
    sub_segs, sub_sources = _split_curves_at_intersections(curves, sources, tol=1e-6)
    arr = _build_arrangement(sub_segs, sub_sources, tol=1e-6)
    labels = _classify_faces(arr, curves_a, curves_b, tol=1e-6)
    # Exactly one (True, True) face (the intersection lens)
    inAB = [k for k, v in labels.items() if v == (True, True)]
    assert len(inAB) == 1
    # At least one (True, False) and one (False, True)
    assert any(v == (True, False) for v in labels.values())
    assert any(v == (False, True) for v in labels.values())
    # Unbounded face is (False, False)
    unb_idx = next(f.idx for f in arr.faces if f.unbounded)
    assert labels[unb_idx] == (False, False)


from mmcore.topo.brep.boolean2d import _select_kept_faces, _extract_island_loops


def test_select_kept_faces_union():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    curves_a, _ = _collect_curves_with_sources(a, BRep())
    curves_b, _ = _collect_curves_with_sources(BRep(), b)
    curves, sources = _collect_curves_with_sources(a, b)
    sub_segs, sub_sources = _split_curves_at_intersections(curves, sources, tol=1e-6)
    arr = _build_arrangement(sub_segs, sub_sources, tol=1e-6)
    labels = _classify_faces(arr, curves_a, curves_b, tol=1e-6)
    kept = _select_kept_faces(arr, labels, 'union')
    # For union: all bounded faces that are inA or inB are kept.
    for face in arr.faces:
        if face.unbounded:
            assert face.idx not in kept
        else:
            inA, inB = labels[face.idx]
            if inA or inB:
                assert face.idx in kept
            else:
                assert face.idx not in kept


def test_extract_island_loops_overlapping_squares_union():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    curves_a, _ = _collect_curves_with_sources(a, BRep())
    curves_b, _ = _collect_curves_with_sources(BRep(), b)
    curves, sources = _collect_curves_with_sources(a, b)
    sub_segs, sub_sources = _split_curves_at_intersections(curves, sources, tol=1e-6)
    arr = _build_arrangement(sub_segs, sub_sources, tol=1e-6)
    labels = _classify_faces(arr, curves_a, curves_b, tol=1e-6)
    kept = _select_kept_faces(arr, labels, 'union')
    islands = _extract_island_loops(arr, kept)
    # Union of two overlapping unit squares ⇒ 1 island, 1 outer loop, 0 holes.
    assert len(islands) == 1
    outer_loop_hes, hole_loops_hes = islands[0]
    assert len(outer_loop_hes) >= 4
    assert hole_loops_hes == []


from mmcore.topo.brep.boolean2d import _materialize_result


def test_materialize_result_overlapping_squares_union():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    curves_a, _ = _collect_curves_with_sources(a, BRep())
    curves_b, _ = _collect_curves_with_sources(BRep(), b)
    curves, sources = _collect_curves_with_sources(a, b)
    sub_segs, sub_sources = _split_curves_at_intersections(curves, sources, tol=1e-6)
    arr = _build_arrangement(sub_segs, sub_sources, tol=1e-6)
    labels = _classify_faces(arr, curves_a, curves_b, tol=1e-6)
    kept = _select_kept_faces(arr, labels, 'union')
    islands = _extract_island_loops(arr, kept)
    result = _materialize_result(arr, islands)
    assert _count_body_faces(result) == 1
    assert result.validate() == []


def test_union_two_overlapping_squares_end_to_end():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    result = union(a, b, tol=1e-6)
    assert _count_body_faces(result) == 1
    assert result.validate() == []


def test_intersection_two_overlapping_squares_end_to_end():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    result = intersection(a, b, tol=1e-6)
    assert _count_body_faces(result) == 1
    assert result.validate() == []


def test_union_empty_and_nonempty():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = BRep()
    # Build minimal empty BRep (Body + Shell + wire Face, no body faces)
    body = b.new_body(shells=[])
    shell = b.new_shell(faces=[], body=body.id)
    body.shells.append(shell.id)
    wire = b.new_face(outer=None, inners=[], shell=shell.id, surf=None)
    shell.faces.append(wire.id)
    result = union(a, b, tol=1e-6)
    assert _count_body_faces(result) == 1
    assert result.validate() == []


# ---- Spec test T1: disjoint rectangles ----

def test_T1_union_disjoint_rectangles():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(2.0, 0.0, 1.0)])
    r = union(a, b, tol=1e-6)
    assert _count_body_faces(r) == 2
    assert r.validate() == []


def test_T1_intersection_disjoint_rectangles_is_empty():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(2.0, 0.0, 1.0)])
    r = intersection(a, b, tol=1e-6)
    assert _count_body_faces(r) == 0
    assert r.validate() == []


def test_T1_difference_disjoint_rectangles_is_a():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(2.0, 0.0, 1.0)])
    r = difference(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1
    assert r.validate() == []


def test_T1_xor_disjoint_rectangles():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(2.0, 0.0, 1.0)])
    r = xor(a, b, tol=1e-6)
    assert _count_body_faces(r) == 2
    assert r.validate() == []


# ---- Spec test T2: overlapping circles ----

def _circle_region(cx: float, cy: float, r: float) -> BRep:
    return make_region_2d([[circle(center=(cx, cy, 0.0), radius=r)]])


def test_T2_union_overlapping_circles():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(1.0, 0.0, 1.0)
    r = union(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1
    assert r.validate() == []


def test_T2_intersection_overlapping_circles():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(1.0, 0.0, 1.0)
    r = intersection(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1
    assert r.validate() == []


def test_T2_difference_overlapping_circles():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(1.0, 0.0, 1.0)
    r = difference(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1
    assert r.validate() == []


def test_T2_xor_overlapping_circles():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(1.0, 0.0, 1.0)
    r = xor(a, b, tol=1e-6)
    assert _count_body_faces(r) == 2
    assert r.validate() == []


# ---- Spec test T3: square with hole vs disk ----

def _cw_square(x0: float, y0: float, side: float) -> list[NURBSCurveTuple]:
    """Build a CW square boundary (suitable for use as a hole loop)."""
    return [
        _line([x0,        y0,        0.0], [x0,        y0 + side, 0.0]),
        _line([x0,        y0 + side, 0.0], [x0 + side, y0 + side, 0.0]),
        _line([x0 + side, y0 + side, 0.0], [x0 + side, y0,        0.0]),
        _line([x0 + side, y0,        0.0], [x0,        y0,        0.0]),
    ]


def test_T3_union_square_with_hole_and_disk():
    # Outer 4x4 square with a unit-square hole at (1.5,1.5)
    a = make_region_2d([
        _square_ccw(0.0, 0.0, 4.0),
        _cw_square(1.5, 1.5, 1.0),
    ])
    # A disk that straddles the hole
    b = _circle_region(2.0, 2.0, 1.25)
    r = union(a, b, tol=1e-6)
    # Exactly 1 island; the hole may be fully or partially filled
    assert _count_body_faces(r) == 1
    assert r.validate() == []


def test_T4_union_two_squares_sharing_one_edge():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(1.0, 0.0, 1.0)])
    r = union(a, b, tol=1e-6)
    # Result is a 1×2 rectangle — 1 body face
    assert _count_body_faces(r) == 1
    assert r.validate() == []


def test_T4_intersection_two_squares_sharing_one_edge_is_empty():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(1.0, 0.0, 1.0)])
    r = intersection(a, b, tol=1e-6)
    assert _count_body_faces(r) == 0
    assert r.validate() == []


# ---- Spec test T5: nested (A ⊆ B) ----

def test_T5_nested_union_is_outer():
    a = _circle_region(0.0, 0.0, 0.3)         # small disk inside
    b = make_region_2d([_square_ccw(-2.0, -2.0, 4.0)])  # big square
    r = union(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1  # just the square


def test_T5_nested_intersection_is_inner():
    a = _circle_region(0.0, 0.0, 0.3)
    b = make_region_2d([_square_ccw(-2.0, -2.0, 4.0)])
    r = intersection(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1  # the small circle


def test_T5_nested_difference_is_square_with_hole():
    a = make_region_2d([_square_ccw(-2.0, -2.0, 4.0)])  # big square
    b = _circle_region(0.0, 0.0, 0.3)                   # small circle
    r = difference(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1
    body_face = next(f for f in r.F.values() if f.outer is not None)
    assert len(body_face.inners) == 1  # circle became a hole
    assert r.validate() == []


# ---- Spec test T6: tangent circles ----

def test_T6_tangent_circles_union():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(2.0, 0.0, 1.0)  # touch at (1, 0)
    r = union(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1  # one merged figure-eight
    assert r.validate() == []


def test_T6_tangent_circles_intersection_is_empty():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(2.0, 0.0, 1.0)
    r = intersection(a, b, tol=1e-6)
    assert _count_body_faces(r) == 0
    assert r.validate() == []


# ---- Spec test T7: identical inputs ----

def test_T7_identical_inputs_union_is_a():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(0.0, 0.0, 1.0)
    r = union(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1
    assert r.validate() == []


def test_T7_identical_inputs_intersection_is_a():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(0.0, 0.0, 1.0)
    r = intersection(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1
    assert r.validate() == []


def test_T7_identical_inputs_difference_is_empty():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(0.0, 0.0, 1.0)
    r = difference(a, b, tol=1e-6)
    assert _count_body_faces(r) == 0
    assert r.validate() == []


def test_T7_identical_inputs_xor_is_empty():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(0.0, 0.0, 1.0)
    r = xor(a, b, tol=1e-6)
    assert _count_body_faces(r) == 0
    assert r.validate() == []


# ---- Spec test T8: composition chain ----

def test_T8_composition_union_then_intersection():
    """(square ∪ triangle) ∩ circle should round-trip through the API."""
    square = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    triangle = make_region_2d([[
        _line([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
        _line([2.0, 0.0, 0.0], [1.5, 1.0, 0.0]),
        _line([1.5, 1.0, 0.0], [1.0, 0.0, 0.0]),
    ]])
    step1 = union(square, triangle, tol=1e-6)
    assert step1.validate() == []
    c = _circle_region(1.0, 0.5, 1.2)
    step2 = intersection(step1, c, tol=1e-6)
    assert step2.validate() == []
    # At least one island remains
    assert _count_body_faces(step2) >= 1


def test_T9_surface_derived_input_accepted_by_boolean():
    """A BRep built from a planar NURBS surface via make_face_from_surface
    should be a valid input to the boolean ops — proves that the pipeline
    is agnostic to how the input BRep was constructed.
    """
    from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple
    # trivial planar surface: the z=0 unit square
    surf = NURBSSurfaceTuple(
        order_u=2, order_v=2,
        knot_u=np.array([0.0, 0.0, 1.0, 1.0]),
        knot_v=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=np.array([
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ]),
        weights=np.ones((2, 2)),
    )
    a = BRep()
    a.make_face_from_surface(surf)
    # make_face_from_surface creates a body face with outer loop — valid input
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    r = union(a, b, tol=1e-6)
    assert r.validate() == []
    assert _count_body_faces(r) >= 1


def test_T10_every_prior_test_output_validates():
    """T10 is the cross-cutting invariant: every public-API call in every
    prior test must have produced a BRep that satisfies validate(). This is
    already asserted inline per-test; this test just re-runs the most
    commonly-broken cases in a single-function sanity check.
    """
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    for op_fn in (union, intersection, difference, xor):
        r = op_fn(a, b, tol=1e-6)
        errs = r.validate()
        assert errs == [], f"{op_fn.__name__} produced invalid BRep: {errs}"


# ============================================================================
# Point-in-region cases (merged from test_point_in_region.py)
# ============================================================================
"""Tests for point_in_region (2D point-in-region via nurbs_ccx segment cast).

Uses 3D curves with z=0 for the "2D plane" convention.
"""

import numpy as np
import pytest

from mmcore.construction import circle
from mmcore.nurbs._nurbs_eval import NURBSCurveTuple
from mmcore.topo.brep.boolean2d import point_in_region


def line(p0, p1):
    """Build a degree-1 NURBS curve between two 3D points."""
    return NURBSCurveTuple(
        order=2,
        knot=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=np.array([p0, p1], dtype=float),
        weights=np.array([1.0, 1.0], dtype=float),
    )


def unit_square_ccw():
    """Unit square boundary (CCW in xy plane) as 4 line segments."""
    return [
        line([0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
        line([1.0, 0.0, 0.0], [1.0, 1.0, 0.0]),
        line([1.0, 1.0, 0.0], [0.0, 1.0, 0.0]),
        line([0.0, 1.0, 0.0], [0.0, 0.0, 0.0]),
    ]


def test_point_in_center_of_unit_square_is_inside():
    region = unit_square_ccw()
    assert point_in_region(np.array([0.5, 0.5, 0.0]), region, tol=1e-6) is True


def test_point_far_from_unit_square_is_outside():
    region = unit_square_ccw()
    assert point_in_region(np.array([2.0, 0.5, 0.0]), region, tol=1e-6) is False


def test_point_below_unit_square_is_outside():
    region = unit_square_ccw()
    assert point_in_region(np.array([0.5, -1.0, 0.0]), region, tol=1e-6) is False


def test_point_at_circle_center_is_inside():
    region = [circle(center=(0.0, 0.0, 0.0), radius=1.0)]
    assert point_in_region(np.array([0.0, 0.0, 0.0]), region, tol=1e-6) is True


def test_point_just_inside_circle_is_inside():
    region = [circle(center=(0.0, 0.0, 0.0), radius=1.0)]
    assert point_in_region(np.array([0.8, 0.0, 0.0]), region, tol=1e-6) is True


def test_point_just_outside_circle_is_outside():
    region = [circle(center=(0.0, 0.0, 0.0), radius=1.0)]
    assert point_in_region(np.array([1.2, 0.0, 0.0]), region, tol=1e-6) is False


def test_point_far_from_circle_is_outside():
    region = [circle(center=(0.0, 0.0, 0.0), radius=1.0)]
    assert point_in_region(np.array([5.0, 5.0, 0.0]), region, tol=1e-6) is False


def test_point_in_annulus_ring_is_inside():
    outer = circle(center=(0.0, 0.0, 0.0), radius=1.0)
    inner = circle(center=(0.0, 0.0, 0.0), radius=0.3)
    region = [outer, inner]
    # (0.6, 0) is between inner radius 0.3 and outer radius 1.0
    assert point_in_region(np.array([0.6, 0.0, 0.0]), region, tol=1e-6) is True


def test_point_at_annulus_center_is_outside():
    outer = circle(center=(0.0, 0.0, 0.0), radius=1.0)
    inner = circle(center=(0.0, 0.0, 0.0), radius=0.3)
    region = [outer, inner]
    # (0, 0) is inside the hole — outside the material
    assert point_in_region(np.array([0.0, 0.0, 0.0]), region, tol=1e-6) is False


def test_point_outside_annulus_is_outside():
    outer = circle(center=(0.0, 0.0, 0.0), radius=1.0)
    inner = circle(center=(0.0, 0.0, 0.0), radius=0.3)
    region = [outer, inner]
    assert point_in_region(np.array([2.0, 0.0, 0.0]), region, tol=1e-6) is False


def test_two_disjoint_circles_point_in_only_one_is_inside():
    c1 = circle(center=(-5.0, 0.0, 0.0), radius=1.0)
    c2 = circle(center=(+5.0, 0.0, 0.0), radius=1.0)
    region = [c1, c2]
    assert point_in_region(np.array([-5.0, 0.0, 0.0]), region, tol=1e-6) is True
    assert point_in_region(np.array([+5.0, 0.0, 0.0]), region, tol=1e-6) is True
    assert point_in_region(np.array([0.0, 0.0, 0.0]), region, tol=1e-6) is False


def test_point_on_square_edge_raises():
    """A point lying exactly on a region boundary has an undefined PIP result.

    The helper must raise RuntimeError rather than silently return True or False.
    """
    region = unit_square_ccw()
    # (0, 0.5) is the midpoint of the left edge — exactly on the boundary
    with pytest.raises(RuntimeError, match="lies on a region boundary"):
        point_in_region(np.array([0.0, 0.5, 0.0]), region, tol=1e-6)


# Measured at 1745s — 65% of the entire suite's wall clock, and it passes while
# warning that "containment may be unreliable". Deselecting it (-m "not slow") takes
# the suite from ~45 min to ~16 min. It is skipped by default in no configuration:
# run the full suite, or `-m slow` to run only this, to keep exercising it.
@pytest.mark.slow
def test_point_whose_segment_is_tangent_to_circle_is_outside():
    """Construct P so the default segment direction is exactly tangent to the unit circle.

    Default direction d = (cos θ, sin θ), θ = 0.31415. The tangent point on the
    unit circle where the tangent line is parallel to d is
    Q = (sin θ, -cos θ). Pick P at Q - d (one unit behind Q along d) so the
    segment from P in direction d grazes the circle exactly at Q.
    """
    theta = 0.31415
    cos_t, sin_t = float(np.cos(theta)), float(np.sin(theta))
    # tangent point on unit circle
    qx, qy = sin_t, -cos_t
    # place P one unit behind Q along d
    px, py = qx - cos_t, qy - sin_t
    # P is outside the unit circle (|P| ≈ sqrt(2))
    assert np.hypot(px, py) > 1.0

    region = [circle(center=(0.0, 0.0, 0.0), radius=1.0)]
    result = point_in_region(np.array([px, py, 0.0]), region, tol=1e-6)
    assert result is False

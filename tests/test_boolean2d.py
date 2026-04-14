"""Integration tests for 2D Boolean operations and make_region_2d."""
from __future__ import annotations

import numpy as np
import pytest  # noqa: F401  # used from Task 2 onward (pytest.raises)

from mmcore.construction import circle  # noqa: F401  # used from Task 2 onward
from mmcore.geom._nurbs_eval import NURBSCurveTuple
from mmcore.topo.brep import BRep
# NOTE: difference, intersection, union, xor will be added in Task 9; Task 1 only
# exercises make_region_2d. See Task 1 instructions — Task 9 will restore these imports.
from mmcore.topo.brep.boolean2d import make_region_2d


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

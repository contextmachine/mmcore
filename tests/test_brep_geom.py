"""Tests for BRep geometry-aware methods using real NURBS curves."""
import numpy as np
from mmcore.geom._nurbs_eval import NURBSCurveTuple
from mmcore.topo.brep import BRep


def _line_curve(p0, p1):
    """Create a degree-1 NURBS line segment from p0 to p1."""
    p0, p1 = np.array(p0, dtype=float), np.array(p1, dtype=float)
    return NURBSCurveTuple(
        order=2,
        knot=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=np.array([p0, p1]),
        weights=np.array([1.0, 1.0]),
    )


def _make_quad_with_geom():
    """Build a quad face v1(0,0)--v2(4,0)--v3(4,3)--v4(0,3) with curve geometry."""
    m = BRep()
    c_top = _line_curve((0, 0, 0), (4, 0, 0))
    c_right = _line_curve((4, 0, 0), (4, 3, 0))
    c_bot = _line_curve((4, 3, 0), (0, 3, 0))
    c_left = _line_curve((0, 3, 0), (0, 0, 0))

    v1, v2, e1, l1, f1, s1 = m.MEVVLS((0, 0, 0), (4, 0, 0))
    e1.geom = m.new_curve(c_top)
    e1.param = c_top.interval()

    v3, e2 = m.MEV(l1.id, v2.id, (4, 3, 0))
    e2.geom = m.new_curve(c_right)
    e2.param = c_right.interval()

    v4, e3 = m.MEV(l1.id, v3.id, (0, 3, 0))
    e3.geom = m.new_curve(c_bot)
    e3.param = c_bot.interval()

    e4, l2, f2 = m.MELF(l1.id, v4.id, v1.id)
    e4.geom = m.new_curve(c_left)
    e4.param = c_left.interval()

    return m, (v1, v2, v3, v4), (e1, e2, e3, e4), (l1, l2), (f1, f2)


def assert_valid(brep, ctx=""):
    errors = brep.validate()
    if errors:
        raise AssertionError(f"BRep invalid ({ctx}):\n" + "\n".join(f"  - {e}" for e in errors))


def test_find_edge_at_point_basic():
    m, verts, edges, loops, faces = _make_quad_with_geom()
    eid, t = m.find_edge_at_point((2, 0, 0))
    assert eid == edges[0].id


def test_find_edge_at_point_not_found():
    m, *_ = _make_quad_with_geom()
    result = m.find_edge_at_point((2, 1.5, 0))
    assert result is None


def test_split_edge_at_point_topology():
    m, verts, edges, loops, faces = _make_quad_with_geom()
    v_new = m.split_edge_at_point((2, 0, 0))
    assert_valid(m, "after split_edge_at_point")
    assert v_new is not None
    assert len(m.V) == 5


def test_split_edge_at_point_params():
    m, verts, edges, loops, faces = _make_quad_with_geom()
    e1 = edges[0]
    old_geom_id = e1.geom

    v_new = m.split_edge_at_point((2, 0, 0))

    e1_after = m.E[e1.id]
    assert e1_after.geom == old_geom_id

    new_edges = [e for e in m.E.values()
                 if e.id != e1.id and e.geom == old_geom_id]
    assert len(new_edges) == 1
    e2 = new_edges[0]

    t0_1, t1_1 = e1_after.param
    t0_2, t1_2 = e2.param
    assert abs(t1_1 - t0_2) < 1e-6 or abs(t1_2 - t0_1) < 1e-6


def test_split_edge_at_point_not_found():
    m, *_ = _make_quad_with_geom()
    try:
        m.split_edge_at_point((2, 1.5, 0))
        assert False, "Should have raised"
    except ValueError:
        pass


def test_split_face_by_curve_basic():
    """Splitting a quad face by a vertical line creates a new face."""
    m, verts, edges, loops, faces = _make_quad_with_geom()
    v1, v2, v3, v4 = verts
    l1, l2 = loops

    split_crv = _line_curve((2, 0, 0), (2, 3, 0))

    result = m.split_face_by_curve(split_crv, face_id=faces[1].id)
    va, vb, e_new, l_new, f_new = result
    assert_valid(m, "after split_face_by_curve")

    assert len(m.V) == 6
    assert len(m.F) == 3
    assert e_new.geom is not None
    assert e_new.geom in m.G_CRV


def test_split_face_by_curve_repeated():
    """Two sequential splits produce three faces from the original one."""
    m, verts, edges, loops, faces = _make_quad_with_geom()
    f2 = faces[1]

    crv1 = _line_curve((1, 0, 0), (1, 3, 0))
    crv2 = _line_curve((3, 0, 0), (3, 3, 0))

    _, _, _, _, f_new1 = m.split_face_by_curve(crv1, face_id=f2.id)
    assert_valid(m, "after first split")

    # Second split on the same original face (f2 retained one half)
    m.split_face_by_curve(crv2, face_id=f2.id)
    assert_valid(m, "after second split")

    assert len(m.F) == 4

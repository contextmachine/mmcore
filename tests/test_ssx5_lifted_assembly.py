"""Assembly must preserve the paired preimages and continuous geometry."""
import numpy as np

from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple
from mmcore.numeric.intersection.ssx._nssx5 import (
    _Frag, _axis_closed, _containment_dedup, _is_rational, _make_aggregate,
    _assemble_points, _assemble_branches, _DomainCtx,
)
from mmcore.numeric.intersection.ssx._ssx_polyline import (
    point_matches_polyline, polyline_contained,
)
from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch, SSXPoint


def _surface(cp, weights):
    nu, nv = np.shape(cp)[:2]
    return NURBSSurfaceTuple(nu, nv, np.r_[np.zeros(nu), np.ones(nu)],
                            np.r_[np.zeros(nv), np.ones(nv)],
                            np.asarray(cp, dtype=float), np.asarray(weights, dtype=float))


def test_nonuniform_near_unit_weights_are_not_rounded_away():
    cp = np.array([[[x, y, z] for y in (0., 1.)]
                   for x, z in zip((-1., 0., 1.), (1., -1., 1.))])
    for delta in (8e-6, np.spacing(1.0)):
        w = np.ones((3, 2))
        w[1] += delta
        assert _is_rational(_surface(cp, w))
    assert not _is_rational(_surface(cp, np.ones((3, 2))))


def test_small_open_surface_is_not_made_periodic():
    cp = np.array([[[0., 0., 0.], [0., 1., 0.]],
                   [[1e-11, 0., 0.], [1e-11, 1., 0.]]])
    assert not _axis_closed(_surface(cp, np.ones((2, 2))), 0)
    cp[1] = cp[0]
    assert _axis_closed(_surface(cp, np.ones((2, 2))), 0)


def test_equal_end_controls_do_not_close_an_unclamped_surface():
    from mmcore.nurbs._nurbs_eval import evaluate_nurbs_surface
    cp = np.array([[[x, y, 0.] for y in (0., 1.)]
                   for x in (0., 1., 2., 0.)])
    surface = NURBSSurfaceTuple(3, 2, np.arange(7, dtype=float),
                                np.array([0., 0., 1., 1.]), cp,
                                np.ones((4, 2)))
    assert surface.interval()[0] == (2., 4.)
    first = evaluate_nurbs_surface(surface, 2., .5)['S']
    last = evaluate_nurbs_surface(surface, 4., .5)['S']
    np.testing.assert_array_equal(first, [.5, .5, 0.])
    np.testing.assert_array_equal(last, [1., .5, 0.])
    assert not _axis_closed(surface, 0)


def test_xyz_coincident_curves_with_distinct_preimages_survive():
    xyz = np.array([[0., 0., 0.], [1., 0., 0.]])
    first = np.array([[.25, 0., 0., 0.], [.25, 1., 1., 0.]])
    second = first.copy()
    second[:, 0] = .75
    frags = [_Frag(s, xyz.copy(), 'transversal', False) for s in (first, second)]
    assert len(_containment_dedup(frags, 1e-3, _make_aggregate({}, 1))) == 2


def test_vertex_containment_does_not_delete_a_chord_across_a_detour():
    s = np.array([[0., 0., 0., 0.], [1., 0., 1., 0.]])
    x = np.array([[0., 0., 0.], [1., 0., 0.]])
    ks = np.vstack((s[0], (s[0] + s[1]) / 2, s[1]))
    kx = np.array([[0., 0., 0.], [.5, 1., 0.], [1., 0., 0.]])
    assert not polyline_contained(s, x, ks, kx, np.full(4, .01), .01)
    frags = [_Frag(s, x, 'transversal', False), _Frag(ks, kx, 'transversal', False)]
    assert len(_containment_dedup(frags, .01, _make_aggregate({}, 1))) == 2


def test_continuous_containment_accepts_different_sampling_and_direction():
    a = np.linspace(0., 1., 5)
    s = np.column_stack((a, a * 0, a, a * 0))
    x = np.column_stack((a, a * 0, a * 0))
    for ks, kx in ((s[[0, -1]], x[[0, -1]]), (s[::-1], x[::-1])):
        assert polyline_contained(s, x, ks, kx, np.full(4, 1e-9), 1e-9)
        assert polyline_contained(ks, kx, s, x, np.full(4, 1e-9), 1e-9)


def test_matching_uses_one_correspondence_for_parameters_and_xyz():
    s = np.array([[0., 0., 0., 0.], [1., 0., 1., 0.]])
    x = np.array([[0., 0., 0.], [1., 0., 0.]])
    assert not point_matches_polyline(s[0], x[1], s, x, np.full(4, .01), .01)
    assert point_matches_polyline((s[0] + s[1]) / 2, (x[0] + x[1]) / 2,
                                  s, x, np.full(4, .01), .01)


def test_point_on_other_surface_sheet_is_not_absorbed():
    s = np.array([[.25, 0., 0., 0.], [.25, 1., 1., 0.]])
    x = np.array([[0., 0., 0.], [1., 0., 0.]])
    branch = SSXBranch(curve=(s, x), closed=False, overlap=False, kind='transversal')
    point = SSXPoint(stuv=np.array([.75, .5, .5, 0.]), xyz=np.array([.5, 0., 0.]))
    ctx = _DomainCtx(np.zeros(4), np.ones(4), np.ones(4), np.full(4, .001), (False,) * 4)
    result = _assemble_points([point], [branch], ctx, .001, _make_aggregate({}, 1))
    assert len(result) == 1 and result[0] is point


def test_containment_cap_returns_unknown():
    s = np.array([[0., 0., 0., 0.], [1., 0., 1., 0.]])
    x = np.array([[0., 0., 0.], [1., 0., 0.]])
    assert polyline_contained(s, x, s, x, np.full(4, .01), .01,
                              charge=lambda _: False) is None


def test_distinct_parallel_roots_inside_tolerance_are_not_coalesced():
    """Two exact roots remain two components even inside geometric ptol.

    These are the lifted lines of graph/plane intersections at s=u=.5
    and s=u=.5+2**-11. Continuous proximity of the output polylines
    proves approximation quality, but does not prove root identity.
    """
    separation = 2.**-11
    s = np.array([[.5, 0., .5, 0.], [.5, 1., .5, 1.]])
    x = np.array([[.5, 0., 0.], [.5, 1., 0.]])
    nearby_s, nearby_x = s.copy(), x.copy()
    nearby_s[:, [0, 2]] += separation
    nearby_x[:, 0] += separation
    frags = [_Frag(s, x, 'transversal', False),
             _Frag(nearby_s, nearby_x, 'transversal', False)]
    ctx = _DomainCtx(np.zeros(4), np.ones(4), np.ones(4),
                     np.full(4, .001), (False,) * 4)
    kept = _containment_dedup(frags, .001, _make_aggregate({}, 1), ctx=ctx)
    assert len(kept) == 2


def test_periodic_seam_jump_does_not_absorb_an_interior_preimage():
    """A seam vertex pair is two endpoints, never its parameter chord.

    _concat_chain deliberately preserves (u=1,u=0) at a wrapped joint.
    An intersection at a third preimage u=.5 may share their world
    position on a folded surface; the seam jump must not erase it.
    """
    s = np.array([[1., .2, .3, .4], [0., .2, .3, .4]])
    x = np.zeros((2, 3))
    branch = SSXBranch(curve=(s, x), closed=False, overlap=False,
                       kind='transversal')
    interior = SSXPoint(stuv=np.array([.5, .2, .3, .4]), xyz=np.zeros(3))
    endpoint = SSXPoint(stuv=s[0].copy(), xyz=x[0].copy())
    ctx = _DomainCtx(np.zeros(4), np.ones(4), np.ones(4),
                     np.full(4, .001), (True, False, False, False))
    kept = _assemble_points([interior, endpoint], [branch], ctx, .001,
                             _make_aggregate({}, 1))
    # Neither point carries source-root incidence with the mapped branch.
    # Equal endpoint floats alone cannot replace that missing ownership.
    assert len(kept) == 2
    assert kept[0] is interior and kept[1] is endpoint


def test_nearby_point_preimage_is_not_absorbed_by_a_curve():
    s = np.array([[.5, 0., .5, 0.], [.5, 1., .5, 1.]])
    x = np.array([[.5, 0., 0.], [.5, 1., 0.]])
    branch = SSXBranch(curve=(s, x), closed=False, overlap=False,
                       kind='transversal')
    separation = 2.**-11
    point = SSXPoint(stuv=np.array([.5+separation, .5, .5+separation, .5]),
                     xyz=np.array([.5+separation, .5, 0.]))
    ctx = _DomainCtx(np.zeros(4), np.ones(4), np.ones(4),
                     np.full(4, .001), (False,) * 4)
    kept = _assemble_points([point], [branch], ctx, .001,
                             _make_aggregate({}, 1))
    assert len(kept) == 1 and kept[0] is point


def test_nearby_distinct_isolated_points_are_not_coalesced():
    # Exact zero set of z=((x-a)*(x-b))**2+(y-.5)**2 against z=0.
    roots = (.5, .5 + 2.**-11)
    points = [SSXPoint(stuv=np.array([r, .5, r, .5]),
                       xyz=np.array([r, .5, 0.])) for r in roots]
    ctx = _DomainCtx(np.zeros(4), np.ones(4), np.ones(4),
                     np.full(4, .001), (False,) * 4)
    kept = _assemble_points(points, [], ctx, .001, _make_aggregate({}, 1))
    assert len(kept) == 2
    assert all(a is b for a, b in zip(kept, points))


def test_nearby_distinct_components_are_not_stitched_into_a_false_loop():
    """Preservation must hold through endpoint assembly, after dedup too."""
    roots = (.5, .5 + 2.**-11)
    fragments = []
    for root in roots:
        s = np.array([[root, 0., root, 0.], [root, 1., root, 1.]])
        x = np.array([[root, 0., 0.], [root, 1., 0.]])
        fragments.append(_Frag(s, x, 'transversal', False))
    ctx = _DomainCtx(np.zeros(4), np.ones(4), np.ones(4),
                     np.full(4, .001), (False,) * 4)
    branches = _assemble_branches(fragments, ctx, .001, _make_aggregate({}, 1))
    assert len(branches) == 2
    assert not any(branch.closed for branch in branches)
    for branch in branches:
        xyz = np.asarray(branch.curve[1])
        assert np.ptp(xyz[:, 0]) == 0.
        assert np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum() == 1.

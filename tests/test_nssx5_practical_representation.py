"""Practical NURBS assembly preserves weights and paired parameter paths."""
import numpy as np

from mmcore.numeric._bezier_common import eval_surface
from mmcore.numeric.intersection.ssx import _nssx5 as ssx
from mmcore.numeric.intersection.ssx._ssx_substrate import SSXBranch, SSXPoint
from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple, evaluate_nurbs_surface


def surface(points, weights=None):
    points = np.asarray(points, dtype=float)
    nu, nv = points.shape[:2]
    return NURBSSurfaceTuple(
        nu, nv, np.r_[np.zeros(nu), np.ones(nu)],
        np.r_[np.zeros(nv), np.ones(nv)], points,
        np.ones((nu, nv)) if weights is None else np.asarray(weights))


def context(closed=False):
    return ssx._DomainCtx(np.zeros(4), np.ones(4), np.ones(4),
                          np.full(4, .001), (closed, False, False, False))


def test_near_unit_weights_reach_pair_solver_with_their_geometry(monkeypatch):
    cp = np.array([[[s, t, z] for t in (0., 1.)]
                   for s, z in zip((0., .5, 1.), (1000., -1000., 1000.))])
    weights = np.ones((3, 2))
    weights[1] += 8e-6
    original = surface(cp, weights)
    plane = surface([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])
    calls = []

    def capture(first, second, **kwargs):
        calls.append(kwargs['rational'])
        expected = evaluate_nurbs_surface(original, .5, .5)['S']
        actual = eval_surface(first, .5, .5, rational=kwargs['rational'])
        assert abs(expected[2]) > 100*kwargs['atol']
        np.testing.assert_allclose(actual, expected, atol=1e-12, rtol=0.)
        return {'branches': [], 'points': [], 'singularities': []}

    monkeypatch.setattr(ssx, 'bez_ssx', capture)
    result = ssx.nurbs_ssx(original, plane, atol=1e-5)
    assert calls == [True]
    assert 'unresolved_regions' not in result
    np.testing.assert_array_equal(original.weights, weights)


def test_equal_control_rows_do_not_close_unclamped_surface():
    points = np.array([[[x, t, 0.] for t in (0., 1.)] for x in (0., 1., 2., 0.)])
    original = NURBSSurfaceTuple(3, 2, np.arange(7.), np.array([0., 0., 1., 1.]),
                                 points, np.ones((4, 2)))
    np.testing.assert_allclose(evaluate_nurbs_surface(original, 2., .5)['S'], [.5, .5, 0.])
    np.testing.assert_allclose(evaluate_nurbs_surface(original, 4., .5)['S'], [1., .5, 0.])
    assert not ssx._axis_closed(original, 0)


def test_small_open_patch_has_no_structural_periodic_seam():
    original = surface([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1e-11)])
    assert not ssx._axis_closed(original, 0)


def test_tolerance_stitching_retains_numerically_close_joint():
    ctx = context()
    a = np.array([[0., .5, 0., .5], [.5, .5, .5, .5]])
    b = np.array([[.5+1e-8, .5, .5+1e-8, .5], [1., .5, 1., .5]])
    frags = [ssx._Frag(p, np.c_[p[:, :2], np.zeros(2)], 'transversal', False) for p in (a, b)]
    result = ssx._assemble_branches(frags, ctx, .001, ssx._make_aggregate({}, 1))
    assert len(result) == 1 and not result[0].closed
    np.testing.assert_allclose(result[0].curve[1][[0, -1], 0], [0., 1.])


def test_same_xyz_on_distinct_parameter_sheets_is_not_deduplicated():
    paths = [np.array([[s, 0., 0., 0.], [s, 1., 1., 0.]]) for s in (.25, .75)]
    xyz = np.array([[0., 0., 0.], [1., 0., 0.]])
    frags = [ssx._Frag(p, xyz.copy(), 'transversal', False) for p in paths]
    result = ssx._assemble_branches(frags, context(), .001, ssx._make_aggregate({}, 1))
    assert len(result) == 2


def test_same_path_with_different_sampling_is_coalesced():
    frags = []
    for samples in (np.linspace(0., 1., 3), np.linspace(1., 0., 8)):
        p = np.c_[samples, np.full(len(samples), .5), samples, np.full(len(samples), .5)]
        frags.append(ssx._Frag(p, np.c_[samples, np.zeros((len(samples), 2))],
                              'transversal', False))
    result = ssx._assemble_branches(frags, context(), .001, ssx._make_aggregate({}, 1))
    assert len(result) == 1 and not result[0].closed


def test_matching_endpoints_do_not_hide_a_resolved_detour():
    paths = [np.array([[0., 0., 0., 0.], [1., 0., 1., 0.]]),
             np.array([[0., 0., 0., 0.], [.5, .5, .5, .5], [1., 0., 1., 0.]])]
    frags = [ssx._Frag(p, np.c_[p[:, :2], np.zeros(len(p))], 'transversal', False)
             for p in paths]
    assert len(ssx._containment_dedup(frags, .001, ssx._make_aggregate({}, 1),
                                      ctx=context())) == 2


def test_point_within_both_tolerances_is_absorbed():
    p = np.array([[.25, 0., 0., 0.], [.25, 1., 1., 0.]])
    branch = SSXBranch(curve=(p, np.array([[0., 0., 0.], [1., 0., 0.]])))
    point = SSXPoint(np.array([.25+1e-6, .5, .5, 0.]), np.array([.5, 1e-6, 0.]))
    assert ssx._assemble_points([point], [branch], context(), .001,
                                ssx._make_aggregate({}, 1)) == []


def test_point_on_another_parameter_sheet_is_retained():
    p = np.array([[.25, 0., 0., 0.], [.25, 1., 1., 0.]])
    branch = SSXBranch(curve=(p, np.array([[0., 0., 0.], [1., 0., 0.]])))
    point = SSXPoint(np.array([.75, .5, .5, 0.]), np.array([.5, 0., 0.]))
    assert ssx._assemble_points([point], [branch], context(), .001,
                                ssx._make_aggregate({}, 1)) == [point]


def test_periodic_vertex_pair_is_not_an_interior_parameter_chord():
    p = np.array([[1., .5, .5, .5], [0., .5, .5, .5]])
    branch = SSXBranch(curve=(p, np.zeros((2, 3))))
    point = SSXPoint(np.array([.5, .5, .5, .5]), np.zeros(3))
    assert ssx._assemble_points([point], [branch], context(True), .001,
                                ssx._make_aggregate({}, 1)) == [point]


def test_point_on_opposite_periodic_boundary_is_absorbed():
    p = np.array([[1., 0., 0., 0.], [1., 1., 1., 0.]])
    branch = SSXBranch(curve=(p, np.array([[0., 0., 0.], [1., 0., 0.]])))
    point = SSXPoint(np.array([0., .5, .5, 0.]), np.array([.5, 0., 0.]))
    assert ssx._assemble_points([point], [branch], context(True), .001,
                                ssx._make_aggregate({}, 1)) == []


def test_zero_postprocess_allowance_keeps_all_fragments():
    p = np.array([[0., .5, 0., .5], [1., .5, 1., .5]])
    frags = [ssx._Frag(p.copy(), np.c_[p[:, :2], np.zeros(2)], 'transversal', False)
             for _ in range(2)]
    agg = ssx._make_aggregate({'max_postprocess_work': 0}, 1)
    result = ssx._assemble_branches(frags, context(), .001, agg)
    assert len(result) == 2
    assert agg.post.postprocess_work == 0
    assert not agg.result_fields()['complete']

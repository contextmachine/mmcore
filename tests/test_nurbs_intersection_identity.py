"""NURBS adapter assembly cannot erase root distinctions from its solver."""
import numpy as np
import pytest

from mmcore.nurbs._nurbs_eval import NURBSCurveTuple, NURBSSurfaceTuple
from mmcore.numeric.intersection.csx._ncsx4 import (
    nurbs_csx, _dedup_csx_isolated, _merge_overlaps_by_t, _is_seam_duplicate,
)
from mmcore.numeric.intersection.ccx._nccx4 import _dedup_isolated_pair, _dedup_isolated


def _huge_domain_pair(scale=1.):
    a, b = .125, .375
    z = np.array([a*b, a*b-(a+b)/2., (1.-a)*(1.-b)])
    knots = np.array([1e16]*3 + [1e16+2]*3)
    curve = NURBSCurveTuple(3, knots,
        scale*np.column_stack(([0., .5, 1.], [.5, .5, .5], z)), np.ones(3))
    plane = _plane()
    k = np.array([1e16]*2 + [1e16+2]*2)
    return curve, NURBSSurfaceTuple(2, 2, k, k, scale*plane.control_points, plane.weights)


@pytest.mark.parametrize('scale', [1., 2.**-30])
def test_global_parameter_rounding_does_not_silently_erase_roots(scale):
    curve, plane = _huge_domain_pair(scale)
    roots, overlaps, status = nurbs_csx(curve, plane, tol=1e-6,
                                       tolerance_tier=False, max_cells=100)
    assert not status['complete'], status
    assert not status['budget_exhausted'], status
    assert status['unrepresentable_parameters']
    assert roots is None and overlaps is None
    entries = status['unrepresentable_parameters']
    local = [x for entry in entries for x in entry.get('local_solutions', [entry])]
    assert len(local) == 2
    assert sorted(x['local_parameters'][0] for x in local) == [.125, .375]


def test_unrepresentable_global_parameters_raise_for_legacy_return():
    curve, plane = _huge_domain_pair()
    with pytest.raises(RuntimeError, match='global parameter representation'):
        nurbs_csx(curve, plane, tol=1e-6, tolerance_tier=False, return_status=False)


@pytest.mark.parametrize('multiple', [False, True])
def test_ccx_global_parameter_rejection_has_typed_local_solution(multiple):
    from mmcore.numeric.intersection.ccx._nccx4 import nurbs_ccx, nurbs_ccx_multiple
    k = np.array([1e16]*2+[1e16+2]*2)
    a = NURBSCurveTuple(2, k, np.array([[0., 0., 0.], [1., 0., 0.]]), np.ones(2))
    b = NURBSCurveTuple(2, k, np.array([[.25, -.5, 0.], [.25, .5, 0.]]), np.ones(2))
    result = (nurbs_ccx_multiple([a, b], tol=1e-6, tolerance_tier=False) if multiple else
              nurbs_ccx(a, b, tol=1e-6, tolerance_tier=False))
    roots, overlaps, status = result
    assert roots is None and overlaps is None
    assert not status['complete'], status
    assert status['unrepresentable_parameters']


def test_global_mapping_cannot_collapse_overlap_to_a_point():
    from mmcore.numeric.intersection.ccx._nccx4 import nurbs_ccx
    k = np.array([1e16]*2+[1e16+2]*2)
    a = NURBSCurveTuple(2, k, np.array([[0., 0., 0.], [1., 0., 0.]]), np.ones(2))
    b = NURBSCurveTuple(2, k, np.array([[.125, 0., 0.], [.375, 0., 0.]]), np.ones(2))
    roots, overlaps, status = nurbs_ccx(a, b, tol=1e-6, tolerance_tier=False)
    assert roots is None and overlaps is None
    assert not status['complete'], status
    assert status['unrepresentable_parameters']


def test_roundtrip_checks_source_pair_distance_not_only_midpoint_distance():
    from mmcore.numeric.intersection._parameter_mapping import map_isolated
    a = np.array([[0., 0., 0.], [1., 0., 0.]])
    b = np.array([[0., -.1, 0.], [1., .1, 0.]])
    status = dict(complete=True, boundary_topology_complete=True, partial_results=0)
    mapped = map_isolated(
        dict(u=.5, v=.5, point=np.array([.5, 0., 0.])), ('u', 'v'),
        ((1e16, 1e16+2), (1e16+2, 1e16+4)),
        ((a, (0,)), (b, (1,))), False, .6, status, 'test', True)
    assert mapped is None
    assert not status['complete']


def test_tolerance_minimizers_do_not_claim_distinct_exact_root_identities():
    from fractions import Fraction
    from mmcore.numeric.intersection._parameter_mapping import reject_parameter_aliases
    status = dict(complete=True, boundary_topology_complete=True, partial_results=0)
    entries = [dict(u=.5, v=.25, certification='tolerance',
                    _exact_global_parameters=(Fraction(1, 2), value))
               for value in (Fraction(1, 4), Fraction(1, 4)+Fraction(1, 2**60))]
    assert reject_parameter_aliases(entries, ('u', 'v'), status, 'test', True) == entries
    assert status['complete']


def _plane():
    k = np.array([0., 0., 1., 1.])
    return NURBSSurfaceTuple(2, 2, k, k,
        np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)]),
        np.ones((2, 2)))


def _line():
    return NURBSCurveTuple(2, np.array([0., 0., 1., 1.]),
                           np.array([[0., 0., 0.], [1., 0., 0.]]), np.ones(2))


def test_nurbs_csx_preserves_two_exact_roots_inside_modeling_tolerance():
    a, b = .5, .5+2.**-11
    z = np.array([a*b, a*b-(a+b)/2., (1.-a)*(1.-b)])
    cp = np.column_stack(([0., .5, 1.], [.5, .5, .5], z))
    curve = NURBSCurveTuple(3, np.array([0., 0., 0., 1., 1., 1.]), cp, np.ones(3))
    roots, overlaps, status = nurbs_csx(curve, _plane(), tol=1e-3, tolerance_tier=False)
    assert status['complete'], status
    assert overlaps is None
    assert roots is not None and len(roots) == 2
    np.testing.assert_allclose(sorted(r['t'] for r in roots), [a, b], rtol=0., atol=1e-12)


def test_all_adapter_deduplicators_preserve_distinct_exact_preimages():
    values = (.5, .5+2.**-11)
    ccx = [dict(u=x, v=x, point=np.array([x, 0., 0.]), certification='exact') for x in values]
    assert len(_dedup_isolated_pair(ccx, _line(), _line(), 1e-3)) == 2
    multi = [dict(root, curve1_i=0, curve2_i=1) for root in ccx]
    assert len(_dedup_isolated(multi, [_line(), _line()], 1e-3)) == 2
    csx = [dict(t=x, u=x, v=0., point=np.array([x, 0., 0.])) for x in values]
    assert len(_dedup_csx_isolated(csx, _line(), _plane(), 1e-3)) == 2


def test_overlapping_t_ranges_do_not_merge_different_surface_sheets():
    overlaps = [dict(t_range=(0., 1.), u_range=(u, u), v_range=(0., 1.))
                for u in (.25, .75)]
    assert len(_merge_overlaps_by_t(overlaps, .001)) == 2


def test_overlap_joint_preserves_reversed_surface_parameter_orientation():
    overlaps = [dict(t_range=(0., .5), u_range=(1., .5), v_range=(0., .5)),
                dict(t_range=(.5, 1.), u_range=(.5, 0.), v_range=(.5, 1.))]
    assert _merge_overlaps_by_t(overlaps, .001) == [
        dict(t_range=(0., 1.), u_range=(1., 0.), v_range=(0., 1.))]


def test_overlap_intervals_separated_by_a_real_gap_stay_separate():
    start = .5+2.**-11
    overlaps = [dict(t_range=(0., .5), u_range=(0., .5), v_range=(0., 0.)),
                dict(t_range=(start, 1.), u_range=(start, 1.), v_range=(0., 0.))]
    assert len(_merge_overlaps_by_t(overlaps, .001)) == 2


def test_uv_enclosures_do_not_supply_endpoint_identity():
    overlaps = [dict(t_range=(0., .5), u_range=(0., .5), v_range=(0., .5),
                      uv_range_is_enclosure=True),
                dict(t_range=(.5, 1.), u_range=(.5, 1.), v_range=(.5, 1.),
                      uv_range_is_enclosure=True)]
    assert len(_merge_overlaps_by_t(overlaps, .001)) == 2


def test_planar_overlap_components_with_different_joint_preimages_survive():
    curve = NURBSCurveTuple(2, np.array([0., 0., 1., 1.]),
                            np.array([[-1., 0., 0.], [1., 0., 0.]]), np.ones(2))
    knots = np.array([0., 0., .5, 1., 1.])
    surface = NURBSSurfaceTuple(2, 2, knots, knots, np.array([
        [[-1., -1., 0.], [-1., 1., 0.], [-1., -1., 1.]],
        [[0., -1., 0.], [0., 1., 0.], [0., -1., 0.]],
        [[1., -1., 1.], [1., 1., 0.], [1., -1., 0.]],
    ]), np.ones((3, 3)))
    roots, overlaps, status = nurbs_csx(curve, surface, tol=1e-3,
                                       tolerance_tier=False, max_cells=1000)
    assert status['complete'], status
    assert overlaps is not None and len(overlaps) == 2
    assert all(o['uv_range_is_enclosure'] for o in overlaps)
    assert sorted(o['t_range'] for o in overlaps) == [(0., .5), (.5, 1.)]


def test_equal_unclamped_end_control_rows_are_not_a_periodic_seam():
    cp = np.array([[[x, y, 0.] for y in (0., 1.)] for x in (0., 1., 2., 0.)])
    surface = NURBSSurfaceTuple(3, 2, np.arange(7, dtype=float),
                                np.array([0., 0., 1., 1.]), cp, np.ones((4, 2)))
    assert not _is_seam_duplicate(2., 4., .5, .5, surface, .001, .001)


def test_local_csx_partial_does_not_skip_later_independent_spans(monkeypatch):
    import mmcore.numeric.intersection.csx._ncsx4 as module
    curve = NURBSCurveTuple(2, np.array([0., 0., .5, 1., 1.]),
                            np.array([[0., 0., 0.], [.5, 0., 0.], [1., 0., 0.]]),
                            np.ones(3))
    calls = []

    def bounded_result(control, _surface, **_kwargs):
        calls.append(control)
        first = len(calls) == 1
        point = (control[0]+control[-1])/2.
        return dict(isolated=[] if first else [dict(t=.5, u=point[0], v=0., point=point)],
                    overlaps=[], cells_processed=1, budget_exhausted=first,
                    boundary_topology_complete=not first, truncation_cause='resolution' if first else None)

    monkeypatch.setattr(module, 'bez_csx_v4', bounded_result)
    roots, _overlaps, status = module.nurbs_csx(curve, _plane(), tolerance_tier=False)
    assert len(calls) == 2
    assert roots is not None and len(roots) == 1
    assert not status['complete']

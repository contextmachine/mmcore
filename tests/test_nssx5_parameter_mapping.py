"""Global knot-domain rounding must not alter SSX topology or certificates."""
from fractions import Fraction

import numpy as np

from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple
from mmcore.numeric.intersection.ssx._nssx5 import (
    _RawResults, _collect_pair, _finish_collecting, _make_aggregate, nurbs_ssx,
)
from mmcore.numeric.intersection.ssx._ssx_substrate import SSXPoint, SSXBranch


def test_huge_global_domain_retains_local_curve_as_partial():
    low, high = 1e16, 1e16 + 2.
    knots = np.array([low, low, high, high])
    a = np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])
    b = np.array([[[u, .5, v-.5] for v in (0., 1.)] for u in (0., 1.)])
    surfaces = [NURBSSurfaceTuple(2, 2, knots, knots, p, np.ones((2, 2)))
                for p in (a, b)]
    result = nurbs_ssx(*surfaces, atol=1e-6, max_cells=2000)
    assert not result['complete']
    assert 'parameter_representation' in result['status']['reasons']
    assert not result['branches']
    issues = [r for r in result['unresolved_regions']
              if r['reason'] == 'parameter_representation']
    assert issues and issues[0]['local_result']['branches']


def test_cross_pair_global_aliases_withhold_both_local_solutions():
    raw, agg = _RawResults(), _make_aggregate({}, 2)
    net = np.zeros((2, 2, 3))
    rect = (1e16, 1e16+2.) * 4
    for pair, t in [((0, 0), .125), ((1, 0), .375)]:
        result = {'points': [SSXPoint(np.full(4, t), np.zeros(3))]}
        _collect_pair(raw, result, rect, pair, sources=(net, net),
                      rational=False, atol=1e-3, agg=agg)
    _finish_collecting(raw)
    assert raw.points == []
    assert len(raw.unresolved) == 2
    assert all(r['local_result']['points'] for r in raw.unresolved)
    assert 'parameter_representation' in agg.reasons


def test_one_unrepresentable_branch_withholds_other_pair_entities_atomically():
    raw, agg = _RawResults(), _make_aggregate({}, 1)
    net = np.zeros((2, 2, 3))
    rect = (1e16, 1e16+2.) * 4
    result = {
        'branches': [SSXBranch((np.array([[.125]*4, [.375]*4]), np.zeros((2, 3))))],
        'points': [SSXPoint(np.ones(4), np.zeros(3))],
    }
    _collect_pair(raw, result, rect, (0, 0), sources=(net, net),
                  rational=False, atol=1e-3, agg=agg)
    _finish_collecting(raw)
    assert raw.frags == [] and raw.points == []
    assert len(raw.unresolved) == 1
    assert raw.unresolved[0]['local_result'] is result


def test_unresolved_bounds_enclose_exact_affine_parameter_mapping():
    low, high, t = .20323707831873447, .7064770887111032, .569861747188787
    raw = _RawResults()
    _collect_pair(raw, {'unresolved_regions': [{
        'stuv_min': [t]*4, 'stuv_max': [t]*4, 'reason': 'test',
    }]}, (low, high)*4, (0, 0))
    exact = Fraction(low)+(Fraction(high)-Fraction(low))*Fraction(t)
    assert all(Fraction(lo) <= exact <= Fraction(hi)
               for lo, hi in zip(raw.unresolved[0]['stuv_min'],
                                 raw.unresolved[0]['stuv_max']))


def test_unproved_candidate_keeps_local_and_exact_global_provenance():
    raw, agg = _RawResults(), _make_aggregate({}, 1)
    net = np.zeros((2, 2, 3))
    rect = (1e16, 1e16 + 2.) * 4
    candidate = (.25, .5, .75, 1.)
    result = {'unresolved_regions': [{
        'stuv_min': [0.]*4, 'stuv_max': [1.]*4,
        'reason': 'unresolved_multiplicity', 'candidate': candidate,
        'candidate_kind': 'tangent_point', 'source_existence': False,
    }]}
    _collect_pair(raw, result, rect, (0, 0), sources=(net, net),
                  rational=False, atol=1e-3, agg=agg)
    _finish_collecting(raw)
    diagnostic, = raw.unresolved
    assert 'candidate' not in diagnostic
    assert diagnostic['local_candidate'] == candidate
    assert diagnostic['candidate_parameter_bounds'] == ((1e16, 1e16+2.),)*4
    assert tuple(map(Fraction, diagnostic['exact_global_candidate'])) == tuple(
        Fraction(1e16) + 2*Fraction(t) for t in candidate)
    assert diagnostic['source_existence'] is False
    # An unproved proposal is diagnostic data, not a published float root.
    assert agg.reasons == []
    assert raw.points == [] and raw.singularities == []
    assert agg.post.postprocess_work == 0


def test_mapping_budget_denial_retains_atomic_local_payload():
    raw, agg = _RawResults(), _make_aggregate({'max_postprocess_work': 0}, 1)
    net = np.zeros((2, 2, 3))
    result = {'points': [SSXPoint(np.full(4, .5), np.zeros(3))]}
    _collect_pair(raw, result, (0., 1.)*4, (0, 0), sources=(net, net),
                  rational=False, atol=1e-3, agg=agg)
    _finish_collecting(raw)
    assert raw.points == []
    assert raw.unresolved[0]['reason'] == 'postprocess_cap'
    assert raw.unresolved[0]['local_result'] is result
    assert agg.post.postprocess_work == 0


def test_exact_roundtrip_uses_only_cheap_affine_mapping_allowance():
    raw, agg = _RawResults(), _make_aggregate({'max_postprocess_work': 1}, 1)
    net = np.zeros((32, 32, 3))
    result = {'points': [SSXPoint(np.full(4, .5), np.zeros(3))]}
    _collect_pair(raw, result, (0., 1.)*4, (0, 0), sources=(net, net),
                  rational=False, atol=1e-3, agg=agg)
    _finish_collecting(raw)
    assert len(raw.points) == 1
    assert agg.reasons == []
    assert agg.post.postprocess_work == 1


def test_unrepresentable_region_anchor_evaluation_is_typed_partial():
    from types import SimpleNamespace
    raw, agg = _RawResults(), _make_aggregate({}, 1)
    net = np.full((2, 2, 4), 1e308)
    net[..., 3] = 1e-308
    result = {'overlap_regions': [SimpleNamespace(interior_stuv=np.full(4, .5))]}
    _collect_pair(raw, result, (1e16, 1e16+2.)*4, (0, 0), sources=(net, net),
                  rational=True, atol=1e-3, agg=agg)
    _finish_collecting(raw)
    assert len(raw.unresolved) == 1
    assert raw.unresolved[0]['reason'] == 'parameter_representation'
    assert raw.unresolved[0]['local_result'] is result

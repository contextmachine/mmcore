from fractions import Fraction

import numpy as np

from mmcore.numeric.intersection.ssx._ssx_boundary_fibers import source_boundary_fiber_seeds


def fiber_pair():
    w = np.array([[1., 1.], [np.sqrt(.5), np.sqrt(.5)], [1., 1.]])
    first = np.zeros((3, 2, 4)); second = first.copy()
    first[..., 3] = second[..., 3] = w
    first[:, 1, 0] = [1., 0., -1.]
    first[:, 1, 2] = w[:, 1]
    second[:, 0, 2] = w[:, 0]
    second[:, 1, 1] = [1., 0., -1.]
    return first, second


def test_exact_original_fiber_has_exact_target_boundary_preimage():
    first, second = fiber_pair()
    result = source_boundary_fiber_seeds(first, second, 1, 0., max_cells=2000)
    assert not result['boundary_topology_complete']
    assert not result['budget_exhausted']
    assert result['isolated'] == result['overlaps'] == []
    assert result['boundary_seed_proposals'] == []
    assert len(result['parameter_fibers']) == 1
    seed = result['parameter_fibers'][0]
    assert (seed['u'], seed['v']) == (.5, 1.)
    assert seed['certification'] == 'exact_source_parameter_fiber'
    assert seed['source_fiber_certificate']['target_interval'] == (Fraction(1, 2),)*2
    reverse = source_boundary_fiber_seeds(first, second, 3, 0., max_cells=2000)
    assert (reverse['parameter_fibers'][0]['u'], reverse['parameter_fibers'][0]['v']) == (.5, 1.)


def test_noncollapsed_owner_declines():
    first, second = fiber_pair()
    assert source_boundary_fiber_seeds(first, second, 0, 0.) is None


def test_exact_point_off_target_has_no_false_fiber():
    first, second = fiber_pair()
    second[..., 0] += second[..., 3] * (1./1024)
    result = source_boundary_fiber_seeds(first, second, 1, 0.)
    assert result['parameter_fibers'] == []
    assert result['boundary_seed_proposals'] == []
    assert not result['boundary_topology_complete']


def test_rounded_cone_apex_is_only_a_numerical_proposal():
    from examples.ssx.bez_ssx5_case14 import S1, S2
    for first, second in ((S1, S2), (S1*1e-6, S2*1e6)):
        for axis in (1, 3):
            result = source_boundary_fiber_seeds(first, second, axis, 0., atol=1e-3)
            assert result['parameter_fibers'] == []
            assert len(result['boundary_seed_proposals']) == 1
            assert result['boundary_seed_proposals'][0]['certification'] == 'numerical_seed_proposal'
            assert not result['boundary_topology_complete']


def test_denied_work_and_output_keep_explicit_obligation():
    first, second = fiber_pair()
    for limit in (0, 1, 5):
        result = source_boundary_fiber_seeds(first, second, 1, 0., max_cells=limit)
        assert result['cells_processed'] <= limit
        assert result['budget_exhausted'] and not result['boundary_topology_complete']
    result = source_boundary_fiber_seeds(first, second, 1, 0., max_results=0)
    assert not result['parameter_fibers'] and not result['boundary_topology_complete']


def test_nearcollapsed_nonzero_edge_is_never_exact_fiber():
    first, second = fiber_pair()
    first[1, 0, 0] = 2.**-60
    result = source_boundary_fiber_seeds(first, second, 1, 0.)
    assert not result['parameter_fibers']
    assert not result['boundary_topology_complete']


def test_exact_target_collapsed_edge_preserves_both_free_parameters():
    first, second = fiber_pair()
    second[:, 0, 2] = 0.
    result = source_boundary_fiber_seeds(first, second, 1, 0.)
    fibers = result['parameter_fibers']
    assert any(seed.get('u_range') == (0., 1.) and seed['v'] == 0. for seed in fibers)


def test_exact_fiber_point_has_unique_interior_planar_preimage():
    first, _ = fiber_pair()
    target = np.array([[[-1., -1., 0., 1.], [-1., 1., 0., 1.]],
                       [[1., -1., 0., 1.], [1., 1., 0., 1.]]])
    result = source_boundary_fiber_seeds(first, target, 1, 0.)
    assert len(result['parameter_fibers']) == 1
    seed = result['parameter_fibers'][0]
    assert (seed['u'], seed['v']) == (.5, .5)
    assert seed['source_fiber_certificate']['target_parameter_box'] == ((Fraction(1, 2),)*2,)*2
    assert not result['boundary_topology_complete']
    target[..., 2] = 2.**-60
    result = source_boundary_fiber_seeds(first, target, 1, 0.)
    assert not result['parameter_fibers']


def test_convex_nonaffine_quad_membership_is_exact_not_sampled():
    first, _ = fiber_pair()
    target = np.array([[[-1., -1., 0., 1.], [-1., 1., 0., 1.]],
                       [[1., -1., 0., 1.], [2., 1., 0., 1.]]])
    result = source_boundary_fiber_seeds(first, target, 1, 0.)
    assert len(result['parameter_fibers']) == 1
    assert result['parameter_fibers'][0]['source_fiber_certificate']['target_parameter_box'] == ((Fraction(0), Fraction(1)),)*2
    assert not result['boundary_topology_complete']


def test_distinct_close_target_preimages_keep_exact_interval_identity():
    first, _ = fiber_pair()
    a, b = .5, .5+2.**-20
    q = np.array([a*b, a*b-(a+b)/2, 1-a-b+a*b])
    target = np.zeros((3, 2, 4)); target[..., 3] = 1.
    target[..., 0] = q[:, None]; target[:, 1, 1] = 1.
    result = source_boundary_fiber_seeds(first, target, 1, 0.)
    assert len(result['parameter_fibers']) == 2
    intervals = [f['source_fiber_certificate']['target_interval'] for f in result['parameter_fibers']]
    assert intervals[0][1] < intervals[1][0]


def test_unrepresentable_target_root_keeps_resolution_partial():
    first, _ = fiber_pair()
    target = np.zeros((3, 2, 4)); target[..., 3] = 1.
    target[..., 0] = np.array([1e16, 1e16, -1e16])[:, None]
    target[:, 1, 1] = 1.
    result = source_boundary_fiber_seeds(first, target, 1, 0., atol=1e-10)
    assert not result['parameter_fibers']
    assert not result['boundary_topology_complete']
    assert result['truncation_cause'] == 'resolution'


def test_two_target_roots_with_same_float_uv_remain_two_fiber_seeds():
    first, _ = fiber_pair()
    target = np.zeros((3, 2, 4)); target[..., 3] = 1.
    # (1-u)*(1-(1+2**-60)*u): the second root and u=1
    # have the same nearest floating parameter, but distinct source roots.
    target[..., 0] = np.array([1., -2.**-61, 0.])[:, None]
    target[:, 1, 1] = 1.
    result = source_boundary_fiber_seeds(first, target, 1, 0.)
    assert len(result['parameter_fibers']) == 2
    assert [(seed['u'], seed['v']) for seed in result['parameter_fibers']] == [(1., 0.)]*2
    certificates = [seed['source_fiber_certificate'] for seed in result['parameter_fibers']]
    assert certificates[0]['target_interval'] != certificates[1]['target_interval']
    assert not result['boundary_topology_complete']


def test_finite_homogeneous_target_with_unrepresentable_cartesian_inverse_is_partial():
    first, _ = fiber_pair()
    target = np.array([[[-1., -1., 0., 1e-320], [-1., 2., 0., 1e-320]],
                       [[2., -1., 0., 1e-320], [1., 1., 0., 1e-320]]])
    with np.errstate(over='ignore', invalid='ignore'):
        result = source_boundary_fiber_seeds(first, target, 1, 0.)
    assert not result['boundary_topology_complete']
    assert not result['parameter_fibers']


def test_original_source_fiber_representation_outranks_rounded_proposal_residual():
    from mmcore.numeric.intersection.ssx._bez_ssx5 import _find_ssx_boundary_zeros
    first, second = fiber_pair()
    source = source_boundary_fiber_seeds(first, second, 1, 0.)
    # The face callback carries original-source evidence while these nets
    # deliberately model a lossy proposal transform. Its residual is not
    # allowed to invalidate the independently supplied representation test.
    proposal = first.copy(); proposal[..., 0] += proposal[..., 3]
    def face_callback(curve, target, axis, side, **kwargs):
        return source if (axis, side) == (1, 0.) else dict(isolated=[], overlaps=[])
    fibers, checks = [], []
    def source_representation(parameters, xyz):
        checks.append(tuple(parameters))
        return np.array_equal(xyz, np.zeros(3))
    _find_ssx_boundary_zeros(proposal, second, 1e-3, rational=True,
        face_csx_fn=face_callback, fiber_sink=fibers,
        source_point_representation=source_representation)
    assert len(fibers) == len(checks) == 1
    assert fibers[0].parameter_fiber


def test_empty_fiber_seed_search_defers_to_complete_source_csx(monkeypatch):
    from mmcore.numeric.intersection.ssx import _bez_ssx5 as ssx
    from mmcore.numeric.intersection.ssx import _ssx_planar_cut as planar
    from mmcore.numeric.intersection.ssx import _ssx_boundary_fibers as fibers
    # A rational ruled wedge has a collapsed t=0 edge at z=1, OFF
    # the target z=0 plane. The actual ordinary line lies at t=1/2.
    # Failure to find an apex seed must not poison its boundary census.
    w = np.array([1., np.sqrt(.5), 1.])
    first = np.zeros((3, 2, 4)); first[..., 3] = w[:, None]
    first[:, 1, 0] = [1., 0., -1.]
    first[:, 1, 1] = w; first[:, 0, 2] = w; first[:, 1, 2] = -w
    second = np.array([[[-2., 0., 0., 1.], [-2., 1., 0., 1.]],
                       [[2., 0., 0., 1.], [2., 1., 0., 1.]]])
    no_seed_faces, source_calls = [], []
    original_seed, original_csx = fibers.source_boundary_fiber_seeds, ssx.bez_csx
    def seeds(*args, **kwargs):
        result = original_seed(*args, **kwargs)
        if result is not None and not result['parameter_fibers'] and not result['boundary_seed_proposals']:
            no_seed_faces.append(args[2:4])
        return result
    def source_csx(*args, **kwargs):
        assert kwargs.get('source_residual') is not None
        source_calls.append(True)
        return original_csx(*args, **kwargs)
    monkeypatch.setattr(fibers, 'source_boundary_fiber_seeds', seeds)
    monkeypatch.setattr(ssx, 'bez_csx', source_csx)
    monkeypatch.setattr(planar, 'supported_source_pair', lambda *args: False)
    result = ssx.bez_ssx(first, second, atol=.001, rational=True, max_cells=10000)
    assert no_seed_faces == [(1, 0.)]
    assert source_calls
    assert result['complete'], result['status']
    assert len(result['branches']) == 1
    xyz = np.asarray(result['branches'][0].curve[1])
    assert np.max(np.abs(xyz[:, 1]-.5)) < 1e-12
    assert np.max(np.abs(xyz[:, 2])) < 1e-12

import numpy as np
import pytest

from mmcore.numeric.intersection.ccx._bez_ccx4 import bez_ccx


def test_nearly_collinear_lines_share_one_exact_endpoint():
    a = np.array([[-1., 0., 0.], [0., 0., 0.]])
    b = np.array([[0., 0., 0.], [2.**-10, 2.**-30, 0.]])
    result = bez_ccx(a, b, rational=False, tolerance_tier=False, max_cells=10)
    assert result['boundary_topology_complete'], result
    assert not result['budget_exhausted']
    assert [(p['u'], p['v']) for p in result['isolated']] == [(1., 0.)]


def test_exact_line_tier_does_not_promote_a_nonzero_parallel_gap():
    a = np.array([[0., 0., 0.], [1., 0., 0.]])
    b = a + [0., 2.**-50, 0.]
    result = bez_ccx(a, b, rational=False, tolerance_tier=False)
    assert result['boundary_topology_complete']
    assert result['overlaps'] == []
    assert result['isolated'] == []


def test_rational_lines_clip_exact_overlap_in_both_parameters():
    a = np.array([[0., 0., 0., 1.], [2., 0., 0., 2.]])
    b = np.array([[.25, 0., 0., 1.], [.75, 0., 0., 1.]])
    result = bez_ccx(a, b, rational=True, tolerance_tier=False)
    assert result['boundary_topology_complete']
    assert len(result['overlaps']) == 1
    overlap = result['overlaps'][0]
    assert overlap['u_range'] == pytest.approx((1/7, 3/5))
    assert overlap['v_range'] == (0., 1.)


@pytest.mark.parametrize('scale', [1e8, 1e16])
def test_exact_line_root_with_unrepresentable_parameters_is_explicitly_partial(scale):
    a = np.array([[.5, .5, -scale], [.5, .5, 2*scale]])
    b = np.array([[0., .5, 0.], [1., .5, 0.]])
    result = bez_ccx(a, b, rational=False, tolerance_tier=False,
                     atol=1e-10, max_cells=10)
    assert not result['boundary_topology_complete'], result
    assert result['isolated'] == []
    assert result['unresolved_parameter_boxes'][0]['exact_parameters'] == (('1', '3'), ('1', '2'))


def test_exact_line_overlap_checks_endpoint_parameter_representation():
    a = np.array([[0., 0., -1e16], [0., 0., 2e16]])
    b = np.array([[0., 0., 0.], [0., 0., 1.]])
    result = bez_ccx(a, b, rational=False, tolerance_tier=False,
                     atol=1e-10, max_cells=10)
    assert not result['boundary_topology_complete'], result
    assert result['overlaps'] == []
    assert result['truncation_cause'] == 'resolution'

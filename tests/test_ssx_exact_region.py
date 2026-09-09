"""Exact affine chart intersections own every dimension of their common set."""
import numpy as np
import pytest

from mmcore.numeric._work_budget import SoftWorkBudget
from mmcore.numeric.intersection.ssx._ssx_exact_region import exact_affine_plane_region_ssx


def _plane(x=(0., 1.), y=(0., 1.), z=0.):
    return np.array([[[s, t, z] for t in y] for s in x])


def _solve(a, b, *, atol=1e-10, rational=False, cells=10000, outputs=100):
    budget = SoftWorkBudget(cells, 100, max_output_items=outputs)
    result = exact_affine_plane_region_ssx(a, b, atol, rational, budget)
    assert result is not None
    result.update(budget.result_fields())
    return result


@pytest.mark.parametrize('flip,transpose', [(False, False), (True, False), (False, True), (True, True)])
def test_full_region_preserves_paired_chart_orientation(flip, transpose):
    a = _plane()
    b = a[::-1].copy() if flip else a.copy()
    if transpose:
        b = b.swapaxes(0, 1).copy()
    result = _solve(a, b)
    assert result['complete']
    assert len(result['overlap_regions']) == 1 and len(result['branches']) == 4
    region = result['overlap_regions'][0]
    assert region.normal_agreement == (-1 if flip != transpose else 1)
    assert len(region.boundary[0]) == 4
    np.testing.assert_array_equal(region.uv1_loops[0][0], region.uv1_loops[0][-1])
    for branch in result['branches']:
        stuv, xyz = branch.curve
        np.testing.assert_array_equal(stuv[:, :2], xyz[:, :2])
        expected = stuv[:, 2:].copy()
        if transpose:
            expected = expected[:, ::-1]
        if flip:
            expected[:, 0] = 1-expected[:, 0]
        np.testing.assert_array_equal(expected, xyz[:, :2])


def test_partial_rotated_overlap_preserves_all_eight_rims():
    a = _plane((-1., 1.), (-1., 1.))
    b = np.array([[[-1.5, 0., 0.], [0., 1.5, 0.]],
                  [[0., -1.5, 0.], [1.5, 0., 0.]]])
    result = _solve(a, b)
    assert result['complete'] and len(result['branches']) == 8
    assert len(result['overlap_regions']) == 1
    loop = result['overlap_regions'][0].uv1_loops[0]
    assert len(loop) == 9
    np.testing.assert_array_equal(loop[0], loop[-1])


@pytest.mark.parametrize('x,y,counts', [((1., 2.), (0., 1.), (1, 0)),
                                      ((1., 2.), (1., 2.), (0, 1)),
                                      ((2., 3.), (0., 1.), (0, 0))])
def test_edge_point_and_empty_intersections_keep_their_dimension(x, y, counts):
    result = _solve(_plane(), _plane(x, y))
    assert result['complete'] and not result['overlap_regions']
    assert (len(result['branches']), len(result['points'])) == counts
    if counts[0]:
        assert result['branches'][0].kind == 'overlap'


def test_uniform_homogeneous_scaling_and_degree_elevation():
    a = _plane((0., .5, 1.), (0., .5, 1.))
    b = _plane((.5, 1.5), (0., 1.))
    ah = np.concatenate((a, np.ones(a.shape[:-1]+(1,))), axis=-1)*2.**-800
    bh = np.concatenate((b, np.ones(b.shape[:-1]+(1,))), axis=-1)*2.**800
    result = _solve(ah, bh, rational=True)
    assert result['complete'] and len(result['overlap_regions']) == 1


def test_nonaffine_chart_and_nonzero_plane_gap_are_not_promoted():
    a = _plane()
    curved = a.copy()
    curved[1, 1, 0] += .25
    for other in (curved, _plane(z=2.**-60)):
        budget = SoftWorkBudget(10000, 100)
        assert exact_affine_plane_region_ssx(a, other, 1e-3, False, budget) is None


def test_bad_float_parameters_preserve_exact_component_as_typed_partial():
    result = _solve(_plane((0., 1e16)), _plane((1., 2.)), atol=1e-30)
    assert not result['complete']
    assert not result['branches'] and not result['overlap_regions']
    issue = result['unresolved_regions'][0]
    assert issue['reason'] == 'parameter_representation'
    assert len(issue['exact_stuv']) == 4


def test_output_cap_never_publishes_a_region_with_missing_rims():
    result = _solve(_plane(), _plane(), outputs=4)
    assert not result['complete'] and 'output_cap' in result['status']['reasons']
    assert not result['branches'] and not result['overlap_regions']
    assert result['unresolved_regions'][0]['exact_dimension'] == 2


def test_float_alias_in_either_injective_chart_keeps_the_segment_unresolved():
    result = _solve(_plane((-1e16, 1e16)), _plane((0., 1.), (1., 2.)), atol=2.)
    assert not result['complete'] and not result['branches']
    assert result['unresolved_regions'][0]['exact_dimension'] == 1


def test_zero_allowance_prevents_exact_source_construction(monkeypatch):
    from mmcore.numeric.intersection.ssx import _ssx_exact_region as module
    monkeypatch.setattr(module, '_exact_points', lambda *args: (_ for _ in ()).throw(AssertionError('unpaid')))
    budget = SoftWorkBudget(0, 100)
    assert exact_affine_plane_region_ssx(_plane(), _plane(), 1e-3, False, budget) is None
    assert budget.exhausted and budget.cells_processed == 0

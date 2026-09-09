import numpy as np
import pytest

from mmcore.numeric.intersection.ccx._bez_ccx4 import bez_ccx


def line():
    return np.array([[0., 0., 0.], [1., 0., 0.]])


@pytest.mark.parametrize('degree', [2, 4, 10, 12])
def test_curve_line_exact_tangent_root_includes_multiplicity(degree):
    curve = np.column_stack((np.linspace(0., 1., degree+1),
                             [(-1.)**i*2.**-degree for i in range(degree+1)],
                             np.zeros(degree+1)))
    result = bez_ccx(curve, line(), tolerance_tier=False, max_cells=200)
    assert result['boundary_topology_complete'], result
    assert result['overlaps'] == []
    root, = result['isolated']
    assert root['u'] == .5
    assert root['v'] == pytest.approx(.5)
    assert root['parameter_root_certification'] == 'exact_curve_line_sturm'


def test_curve_line_common_polynomial_has_irrational_root():
    curve = np.array([[0., -.5, -1.], [.5, -.5, -1.], [1., .5, 1.]])
    result = bez_ccx(curve, line(), tolerance_tier=False, max_cells=200)
    assert result['boundary_topology_complete'], result
    root, = result['isolated']
    assert root['u'] == pytest.approx(np.sqrt(.5))
    assert root['v'] == pytest.approx(np.sqrt(.5))
    assert root['parameter_root_box'][0][0] <= np.sqrt(.5) <= root['parameter_root_box'][0][1]


def test_curve_line_distinct_component_roots_are_provably_empty():
    curve = np.array([[0., -.5, -.25], [1/3, -1/6, -.25],
                      [2/3, 1/6, -.25], [1., .5, .75]])
    result = bez_ccx(curve, line(), tolerance_tier=False, max_cells=200)
    assert result['boundary_topology_complete'], result
    assert result['isolated'] == []


def test_curve_line_isolated_endpoint_pair_is_not_overlap():
    curve = np.array([[0., 0., 0.], [.5, -.5, 0.], [1., 0., 0.]])
    result = bez_ccx(line(), curve, tolerance_tier=False, max_cells=200)
    assert result['boundary_topology_complete'], result
    assert result['overlaps'] == []
    assert [(p['u'], p['v']) for p in result['isolated']] == [(0., 0.), (1., 1.)]


def test_curve_line_near_endpoint_gap_is_provably_empty():
    curve = np.array([[0., 2.**-60, 0.], [.5, .5, 0.], [1., 1., 0.]])
    result = bez_ccx(line(), curve, tolerance_tier=False, max_cells=200)
    assert result['boundary_topology_complete'], result
    assert result['isolated'] == []


def test_curve_line_exact_root_caps_remain_shared():
    curve = np.array([[0., 0., 0.], [.5, -.5, 0.], [1., 0., 0.]])
    result = bez_ccx(curve, line(), tolerance_tier=False, max_cells=1)
    assert result['cells_processed'] <= 1
    assert not result['boundary_topology_complete']
    result = bez_ccx(curve, line(), tolerance_tier=False, max_cells=200, max_results=1)
    assert len(result['isolated']) <= 1
    assert not result['boundary_topology_complete']


def test_curve_line_preflights_coefficient_work_before_fraction_conversion(monkeypatch):
    from mmcore.numeric.intersection.ccx import _curve_line_exact as engine
    curve = np.zeros((1025, 3))
    curve[:, 0] = np.linspace(0., 1., len(curve))
    curve[:, 1] = 1.
    monkeypatch.setattr(engine, '_cartesian', lambda *args: pytest.fail('unbudgeted Fraction construction'))
    result = engine.exact_curve_line_ccx(curve, line(), max_cells=100)
    assert not result['boundary_topology_complete']
    assert result['cells_processed'] <= 100
    assert result['truncation_cause'] == 'preflight'

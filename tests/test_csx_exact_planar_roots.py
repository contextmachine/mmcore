from fractions import Fraction
from math import comb

import numpy as np
import pytest

from mmcore.numeric.intersection.csx._bez_csx4 import bez_csx


def plane():
    return np.array([[[0., 0., 0.], [0., 1., 0.]],
                     [[1., 0., 0.], [1., 1., 0.]]])


@pytest.mark.parametrize('degree', [2, 4, 10, 12])
def test_multiple_plane_root_has_one_exact_parameter_witness(degree):
    curve = np.column_stack((np.full(degree+1, .25),
                             np.linspace(0., 1., degree+1),
                             [(-1.)**(degree-i)*2.**(-degree)
                              for i in range(degree+1)]))
    result = bez_csx(curve, plane(), rational=False,
                     tolerance_tier=False, max_cells=200)
    assert result['boundary_topology_complete'], result
    assert not result['budget_exhausted']
    assert not result['overlaps']
    assert len(result['isolated']) == 1
    root = result['isolated'][0]
    assert root['t'] == .5
    assert root['u'] == pytest.approx(.25)
    assert root['v'] == pytest.approx(.5)
    assert root['parameter_root_certification'] == 'exact_sturm_isolation'
    assert root['root_multiplicity'] == degree
    assert root['parameter_root_box'][0] == (.5, .5)
    assert root['parameter_root_box'][1][0] <= .25 <= root['parameter_root_box'][1][1]
    assert root['parameter_root_box'][2][0] <= .5 <= root['parameter_root_box'][2][1]


def test_plane_root_outside_convex_quad_is_excluded_exactly():
    curve = np.array([[2., .5, -.5], [2., .5, .5]])
    result = bez_csx(curve, plane(), rational=False,
                     tolerance_tier=False, max_cells=200)
    assert result['boundary_topology_complete']
    assert result['isolated'] == []


def test_plane_root_on_nonaffine_bilinear_boundary_is_included():
    surface = plane()
    surface[1, 1, 0] = 1.5
    curve = np.array([[0., .5, -.5], [0., .5, .5]])
    result = bez_csx(curve, surface, rational=False,
                     tolerance_tier=False, max_cells=200)
    assert result['boundary_topology_complete'], result
    assert len(result['isolated']) == 1
    assert result['isolated'][0]['u'] == 0.


def test_two_close_plane_roots_remain_distinct():
    a, b = Fraction(8191, 16384), Fraction(8193, 16384)
    powers = [a*b, -(a+b), Fraction(1)]
    z = [float(sum(powers[j]*Fraction(comb(i, j), comb(2, j))
                   for j in range(i+1))) for i in range(3)]
    curve = np.column_stack(([.25]*3, [0., .5, 1.], z))
    result = bez_csx(curve, plane(), rational=False,
                     tolerance_tier=False, atol=1e-3, max_cells=200)
    assert result['boundary_topology_complete'], result
    assert sorted(root['t'] for root in result['isolated']) == [float(a), float(b)]


def test_planar_root_isolation_honors_cell_and_output_caps():
    curve = np.array([[.25, .5, -.5], [.25, .5, .5]])
    result = bez_csx(curve, plane(), rational=False,
                     tolerance_tier=False, max_cells=1)
    assert result['cells_processed'] <= 1
    assert not result['boundary_topology_complete']


def test_algebraic_plane_root_on_exact_boundary_uses_common_polynomial():
    # x=z=t*t-1/2: the irrational plane root is exactly on the u=0 edge.
    z = np.array([-.5, -.5, .5])
    curve = np.column_stack((z, [.5]*3, z))
    result = bez_csx(curve, plane(), rational=False,
                     tolerance_tier=False, max_cells=200)
    assert result['boundary_topology_complete'], result
    assert len(result['isolated']) == 1
    root = result['isolated'][0]
    assert root['t'] == pytest.approx(np.sqrt(.5), abs=2e-16)
    assert root['u'] == pytest.approx(0., abs=1e-15)
    assert root['v'] == pytest.approx(.5)


def test_positive_weight_rational_plane_root_uses_homogeneous_polynomial():
    curve = np.array([[.25, .5, -.5, 1.], [.5, 1., 1., 2.]])
    surface = np.concatenate((plane(), np.ones((2, 2, 1))), axis=-1)
    result = bez_csx(curve, surface, rational=True,
                     tolerance_tier=False, max_cells=200)
    assert result['boundary_topology_complete'], result
    root, = result['isolated']
    assert root['t'] == pytest.approx(1/3)
    assert (root['u'], root['v']) == pytest.approx((.25, .5))


@pytest.mark.parametrize('scale', [1e8, 1e16])
def test_unrepresentable_parameter_does_not_publish_an_off_plane_root(scale):
    curve = np.array([[.25, 0., -scale], [.25, 1., 2*scale]])
    result = bez_csx(curve, plane(), rational=False, tolerance_tier=False,
                     atol=1e-10, max_cells=200)
    assert not result['boundary_topology_complete'], result
    assert result['truncation_cause'] == 'resolution'
    assert result['isolated'] == []
    assert result['unresolved_parameter_boxes'][0]['reason'] == 'parameter_representation'


def test_finite_large_chart_does_not_overflow_numeric_inverse():
    from mmcore.numeric.intersection.csx._planar_roots import exact_planar_bilinear_roots
    scale = 1e200
    curve = scale*np.array([[.25, .25, -.5], [.25, .75, .5]])
    result = exact_planar_bilinear_roots(curve, scale*plane(), max_cells=200)
    assert result is not None
    assert result['cells_processed'] <= 200
    if not result['boundary_topology_complete']:
        assert result['truncation_cause'] == 'resolution'


def test_planar_roots_preflight_coefficient_work_before_fraction_conversion(monkeypatch):
    from mmcore.numeric.intersection.csx import _planar_roots as engine
    curve = np.zeros((1025, 3))
    curve[:, 0] = .5
    curve[:, 1] = np.linspace(0., 1., len(curve))
    curve[:, 2] = 1.
    surface = np.array([[[0., 0., 0.], [0., 1., 0.]], [[1., 0., 0.], [1., 1., 0.]]])
    monkeypatch.setattr(engine, '_exact_points', lambda *args: pytest.fail('unbudgeted Fraction construction'))
    result = engine.exact_planar_bilinear_roots(curve, surface, max_cells=100)
    assert not result['boundary_topology_complete']
    assert result['cells_processed'] <= 100
    assert result['truncation_cause'] == 'preflight'

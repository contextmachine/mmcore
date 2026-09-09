"""Exhaustive plane/ruling intersections from their exact scalar polynomial."""
import numpy as np
import pytest

from mmcore.numeric._work_budget import SoftWorkBudget
from mmcore.numeric.intersection.ssx._ssx_extrusion import exact_extrusion_plane_ssx


def _solve(a, b, **kwargs):
    budget = SoftWorkBudget(10000, 1000)
    result = exact_extrusion_plane_ssx(a, b, kwargs.get('atol', 1e-6),
                                      kwargs.get('rational', False), budget)
    assert result is not None
    result.update(budget.result_fields())
    return result


def _plane():
    return np.array([[[s, t, 0.] for t in (0., 1.)] for s in (0., 1.)])


@pytest.mark.parametrize('degree', [2, 4, 8, 10, 12])
@pytest.mark.parametrize('swap', [False, True])
def test_all_of_a_high_order_tangent_line_is_accounted_for(degree, swap):
    a = np.array([[[s, j/degree, (-1.)**(degree-j)*2.**-degree]
                   for j in range(degree+1)] for s in (0., 1.)])
    b = _plane()
    result = _solve(*( (b, a) if swap else (a, b)))
    assert result['complete']
    assert len(result['branches']) == 1
    branch = result['branches'][0]
    assert branch.kind == 'tangential'
    assert result['singularities'] == []
    np.testing.assert_allclose(branch.curve[1], [[0., .5, 0.], [1., .5, 0.]], atol=1e-16, rtol=0)


def test_nonbinary_rational_root_and_trim_are_exactly_accounted_for():
    a = np.array([[[s, y, z] for y, z in zip((0., .5, 1.), (1., -2., 4.))]
                  for s in (-1., 2.)])  # normal equation (3t-1)^2
    result = _solve(a, _plane())
    assert result['complete']
    assert len(result['branches']) == 1
    branch = result['branches'][0]
    np.testing.assert_allclose(branch.curve[1], [[0., 1/3, 0.], [1., 1/3, 0.]], atol=1e-16, rtol=0)
    np.testing.assert_allclose(branch.curve[0][:, 0], [1/3, 2/3], atol=1e-16, rtol=0)


def test_two_ordinary_rulings_and_surface_order_preserve_both_preimages():
    a = np.array([[[s, y, z] for y, z in zip((0., .5, 1.), (3/16, -5/16, 3/16))]
                  for s in (0., 1.)])
    result = _solve(a, _plane())
    assert result['complete'] and len(result['branches']) == 2
    assert all(b.kind == 'transversal' for b in result['branches'])
    assert sorted(b.curve[0][0, 1] for b in result['branches']) == [.25, .75]


def test_tiny_positive_gap_proves_empty_without_tolerance_promotion():
    a = np.array([[[s, y, z+2.**-40] for y, z in zip((0., .5, 1.), (.25, -.25, .25))]
                  for s in (0., 1.)])
    result = _solve(a, _plane(), atol=1e-3)
    assert result['complete'] and result['branches'] == [] and result['points'] == []


def test_same_image_on_two_parameter_rulings_remains_two_branches():
    a = np.array([[[s, 0.5, z] for z in (3/16, -5/16, 3/16)] for s in (0., 1.)])
    result = _solve(a, _plane())
    assert result['complete'] and len(result['branches']) == 2
    np.testing.assert_array_equal(result['branches'][0].curve[1], result['branches'][1].curve[1])
    assert result['branches'][0].curve[0][0, 1] != result['branches'][1].curve[0][0, 1]


def test_parameter_fiber_is_left_to_its_own_topology_tier():
    a = np.array([[[0., y, z] for y, z in zip((0., .5, 1.), (.25, -.25, .25))]
                  for _ in (0., 1.)])
    assert exact_extrusion_plane_ssx(a, _plane(), 1e-6, False,
                                   SoftWorkBudget(10000, 1000)) is None


@pytest.mark.parametrize('swap', [False, True])
@pytest.mark.parametrize('rational', [False, True])
@pytest.mark.parametrize('transpose', [False, True])
def test_exact_rank_deficient_ruling_keeps_cusp_curve_metadata(swap, rational, transpose):
    # x=3(2s-1)^2 uses exactly representable degree-three coefficients;
    # x=[1,-float(1/3),-float(1/3),1] would instead have a positive gap.
    a = np.array([[[x, y, float(t)] for t in (0, 1)]
                  for x, y in zip((3., -1., -1., 3.), (-1., 1., -1., 1.))])
    b = np.array([[[0., y, z] for z in (-.5, 1.5)] for y in (-1.5, 1.5)])
    if transpose:
        a = a.transpose(1, 0, 2)
    pair = [a, b]
    if rational:
        pair = [np.concatenate((net*w, np.full(net.shape[:2]+(1,), w)), axis=2)
                for net, w in zip(pair, (2., .5))]
    if swap:
        pair.reverse()
    result = _solve(*pair, rational=rational)
    assert result['complete'] and len(result['branches']) == 1
    assert result['branches'][0].kind != 'tangential'
    curves = [g for g in result['singularities'] if g.kind == 'cusp_curve']
    assert len(curves) == 1
    singularity = curves[0]
    assert singularity.surface == (2 if swap else 1)
    np.testing.assert_array_equal(singularity.samples, result['branches'][0].curve[0])
    np.testing.assert_array_equal(result['branches'][0].curve[1], [[0., 0., 0.], [0., 0., 1.]])
    assert singularity.branch_links == [(0, 0)]


def test_small_nonzero_ruling_normal_is_not_a_cusp_curve():
    delta = 2.**-30
    a = np.array([[[x, y+delta*dy, float(t)] for t in (0, 1)]
                  for x, y, dy in zip((3., -1., -1., 3.),
                                      (-1., 1., -1., 1.), (-1., -1., 1., 1.))])
    b = np.array([[[0., y, z] for z in (-.5, 1.5)] for y in (-1.5, 1.5)])
    result = _solve(a, b)
    assert result['complete'] and len(result['branches']) == 1
    assert result['branches'][0].kind == 'tangential'
    assert result['singularities'] == []


def test_ruling_with_an_isolated_chart_degeneracy_keeps_general_owner():
    a = np.array([[[s, (s-.5)*t, z] for t, z in zip((-.5, 0., .5), (.25, -.25, .25))]
                  for s in (0., 1.)])
    b = _plane()
    b[..., 1] = b[..., 1]-.5
    assert exact_extrusion_plane_ssx(a, b, 1e-6, False,
                                   SoftWorkBudget(10000, 1000)) is None


@pytest.mark.parametrize('scale', [2.**-20, 1., 2.**20])
@pytest.mark.parametrize('nurbs', [False, True])
def test_source_tier_preserves_tilted_high_order_geometry_across_frames(scale, nurbs):
    from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple
    transform = np.array([[1., 0., 0.], [0., 1., 0.], [2., 3., 1.]])
    offset = np.array([1024., -2048., 512.])
    a = np.array([[[s, j/4, (-1.)**j/16] for j in range(5)] for s in (0., 1.)])
    a, b = [(surface @ transform.T+offset)*scale for surface in (a, _plane())]
    expected = (np.array([[0., .5, 0.], [1., .5, 0.]]) @ transform.T+offset)*scale
    if nurbs:
        surfaces = []
        for net in (a, b):
            m, n = net.shape[:2]
            surfaces.append(NURBSSurfaceTuple(
                m, n, np.array([2.]*m+[5.]*m), np.array([2.]*n+[5.]*n),
                net, np.ones((m, n))))
        result = nurbs_ssx(*surfaces, atol=1e-6*scale)
    else:
        result = bez_ssx(a, b, atol=1e-6*scale, rational=False)
    assert result['complete'], result['status']
    assert len(result['branches']) == 1
    np.testing.assert_allclose(result['branches'][0].curve[1], expected, rtol=0., atol=1e-10*scale)

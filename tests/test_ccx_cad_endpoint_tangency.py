"""CAD endpoint contacts do not require isolated algebraic certificates."""
import numpy as np
import pytest

from mmcore.numeric._bezier_common import eval_curve
from mmcore.numeric.intersection.ccx._bez_ccx4 import bez_ccx
from mmcore.numeric.intersection.csx._bez_csx4 import bez_csx


@pytest.mark.parametrize('swap', [False, True])
def test_quadratic_endpoint_tangency_is_complete_at_modeling_tolerance(swap):
    first = np.array([[0., 0., 0.], [.5, 0., 0.], [1., 0., 0.]])
    second = np.array([[0., 1., 0.], [.5, 0., 0.], [1., 0., 0.]])
    if swap:
        first, second = second, first
    result = bez_ccx(first, second, atol=1e-3, rational=False, max_cells=1000)
    assert result['boundary_topology_complete'], result
    assert not result['budget_exhausted']
    assert len(result['isolated']) == 1
    assert np.linalg.norm(result['isolated'][0]['point']-[1., 0., 0.]) < 1e-3
    assert not result.get('unresolved_parameter_boxes')


def test_surface_endpoint_tangency_uses_the_same_cad_contract():
    curve = np.array([[0., 0., 0.], [.5, 0., 0.], [1., 0., 0.]])
    surface = np.array([[[u, v, z] for v in (0., 1.)]
                        for u, z in ((0., 1.), (.5, 0.), (1., 0.))])
    result = bez_csx(curve, surface, atol=1e-3, rational=False, max_cells=2000)
    assert result['boundary_topology_complete'], result
    assert not result['budget_exhausted']
    assert any(np.linalg.norm(entry['point']-[1., 0., 0.]) < 1e-3
               for entry in result['isolated']) or any(
                   entry['t_range'][1] == 1. for entry in result['overlaps'])


@pytest.mark.parametrize('gap', [2.**-7, 2.**-9, 2.**-11])
def test_metric_separated_close_parameter_roots_remain_distinct(gap):
    # A CAD-sized domain with parameter-near but spatially separated roots.
    # Dyadic data keeps the stated analytic roots faithful to the actual
    # float control net. The old 1e10 fixture's rounded coefficients moved
    # its intended roots by up to .12 model units at an atol of .001.
    # Even the closest pair here has a valley deeper than atol, so these
    # are two contacts, not one connected tolerance-coincidence span.
    scale = 2.**14
    atol = 1e-3
    first = np.array([[0., 0., 0.], [.5*scale, 0., 0.], [scale, 0., 0.]])
    low, high = .5-gap, .5+gap
    second = first.copy()
    second[:, 1] = scale*np.array([low*high, low*high-(low+high)/2,
                                   (1-low)*(1-high)])
    assert scale*gap*gap > atol
    result = bez_ccx(first, second, atol=atol, rational=False)
    assert len(result['isolated']) == 2, result
    roots = sorted(entry['u'] for entry in result['isolated'])
    assert np.max(np.abs(np.asarray(roots)-[low, high])*scale) <= atol
    assert np.linalg.norm(result['isolated'][0]['point']-
                          result['isolated'][1]['point']) > 1e-3
    for root in result['isolated']:
        a = eval_curve(first, root['u'], rational=False)
        b = eval_curve(second, root['v'], rational=False)
        assert np.linalg.norm(a-b) <= 1e-3
        assert np.linalg.norm(root['point']-a) <= 1e-3
        assert np.linalg.norm(root['point']-b) <= 1e-3

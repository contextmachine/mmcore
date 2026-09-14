"""CAD-tolerance coverage of collapsed surface parameter lines."""

import numpy as np
import pytest

from mmcore.numeric._bezier_common import eval_curve, eval_surface
from mmcore.numeric.intersection.csx._bez_csx4 import bez_csx


@pytest.mark.parametrize("fixed_parameter", [0., .25, .5, 1.])
@pytest.mark.parametrize("swap_axes", [False, True])
@pytest.mark.parametrize("curve_endpoint", [False, True])
def test_surface_fiber_does_not_flood_isolated_result_budget(
        fixed_parameter, swap_axes, curve_endpoint):
    # S(u,v)=(u, .5+(u-a)^2*(v-.5), (u-a)*v*(1-v)).
    # Its u=a line maps to one point; C reaches that point at t=.35.
    a = fixed_parameter
    linear = np.array([-a, .5-a, 1.-a])
    square = np.array([a*a, a*a-a, (1.-a)**2])
    surface = np.empty((3, 3, 3))
    surface[..., 0] = np.arange(3)[:, None]/2.
    surface[..., 1] = .5 + square[:, None]*np.array([-.5, 0., .5])
    surface[..., 2] = linear[:, None]*np.array([0., .5, 0.])
    if swap_axes:
        surface = surface.transpose(1, 0, 2).copy()
    curve = np.array([[a, -.2, 0.], [a, .5 if curve_endpoint else 1.8, 0.]])

    result = bez_csx(curve, surface, atol=1e-3, rational=False,
                     max_cells=5000, max_results=16)

    assert not result["budget_exhausted"], result
    assert result["truncation_cause"] is None
    assert 1 <= len(result["isolated"]) <= 4
    assert result.get("uncertified_overlap_span") is None
    assert result["overlaps"] == []
    for root in result["isolated"]:
        p = eval_curve(curve, root["t"], rational=False)
        q = eval_surface(surface, root["u"], root["v"], rational=False)
        assert np.linalg.norm(p-q) <= 1e-3
        assert np.linalg.norm(p-[a, .5, 0.]) <= 1e-3


def test_other_surface_preimages_at_same_curve_parameter_are_preserved():
    # The target folds in v but has two separate, regular preimages.
    # Covering a collapsed isoline must not restore the old whole-t slab cut.
    surface = np.empty((2, 3, 3))
    surface[..., 0] = np.array([0., 1.])[:, None]
    surface[..., 1] = .5 + np.array([.1875, -.3125, .1875])
    surface[..., 2] = 0.
    curve = np.array([[.5, .5, -.5], [.5, .5, .5]])

    result = bez_csx(curve, surface, atol=1e-3, rational=False,
                     max_cells=5000, max_results=16)

    assert not result['budget_exhausted'], result
    assert len(result['isolated']) == 2
    assert np.allclose(sorted(root['v'] for root in result['isolated']), [.25, .75], atol=1e-6)
    for root in result['isolated']:
        p = eval_curve(curve, root['t'], rational=False)
        q = eval_surface(surface, root['u'], root['v'], rational=False)
        assert np.linalg.norm(p-q) <= 1e-3


def test_crossing_surface_fibers_do_not_cover_an_unrelated_interior_preimage():
    # Both coordinate edges collapse to the origin, but (.75,.75) is a
    # separate regular preimage. Two free isolines do not cover their box.
    linear = np.array([0., .5, 1.])
    quadratic = np.array([0., -.375, .25])
    surface = np.zeros((3, 3, 3))
    surface[..., 0] = quadratic[:, None]*linear[None, :]
    surface[..., 1] = linear[:, None]*quadratic[None, :]
    curve = np.array([[0., 0., -.5], [0., 0., .5]])

    result = bez_csx(curve, surface, atol=1e-3, rational=False,
                     max_cells=5000, max_results=32)

    assert not result['budget_exhausted'], result
    assert any(np.linalg.norm([root['u']-.75, root['v']-.75]) < 1e-6
               for root in result['isolated'])
    assert any(min(abs(root['u']), abs(root['v'])) < 1e-6
               for root in result['isolated'])
    for root in result['isolated']:
        p = eval_curve(curve, root['t'], rational=False)
        q = eval_surface(surface, root['u'], root['v'], rational=False)
        assert np.linalg.norm(p-q) <= 1e-3


def test_surface_fiber_coverage_accepts_positive_rational_weights():
    surface = np.empty((3, 3, 3))
    surface[..., 0] = np.arange(3)[:, None]/2.
    surface[..., 1] = .5 + np.array([.25, -.25, .25])[:, None]*np.array([-.5, 0., .5])
    surface[..., 2] = np.array([-.5, 0., .5])[:, None]*np.array([0., .5, 0.])
    weights = np.broadcast_to(np.array([1., .5, 2.]), (3, 3))
    surface_h = np.concatenate([surface*weights[..., None], weights[..., None]], axis=2)
    curve = np.array([[.5, -.2, 0.], [.5, 1.8, 0.]])
    curve_weights = np.array([1., 2.])
    curve_h = np.column_stack([curve*curve_weights[:, None], curve_weights])

    result = bez_csx(curve_h, surface_h, atol=1e-3, rational=True,
                     max_cells=5000, max_results=16)

    assert not result['budget_exhausted'], result
    assert 1 <= len(result['isolated']) <= 4
    for root in result['isolated']:
        p = eval_curve(curve_h, root['t'], rational=True)
        q = eval_surface(surface_h, root['u'], root['v'], rational=True)
        assert np.linalg.norm(p-q) <= 1e-3

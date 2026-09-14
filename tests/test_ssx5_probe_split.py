"""Singular probes must subdivide the equations, not free extrusion axes."""
import numpy as np
import pytest

from mmcore.nurbs._nurbs_eval import to_homogeneous_2d
from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx



def _plane():
    return np.array([[[0., 0., 0.], [0., 1., 0.]],
                     [[1., 0., 0.], [1., 1., 0.]]])








@pytest.mark.parametrize("swap", [False, True])
@pytest.mark.parametrize("adapter", [False, True])
def test_near_unit_weight_valley_is_excluded_without_probe_queue_explosion(swap, adapter):
    """A nonintersecting critical line cannot hide two ordinary roots.

    At epsilon=8e-6 the rational height numerator has two simple roots,
    but its derivative vanishes between them. Searching all four
    parameter axes for that impossible tangency formerly exhausted
    15,000 work units and emitted 1,022 unresolved-region diagnostics.
    """
    epsilon = 8e-6
    control = np.array([[[x, y, z] for y in (-1., 1.)]
                        for x, z in zip((-1., 0., 1.), (1., -1., 1.))])
    weights = np.ones((3, 2))
    weights[1] += epsilon
    plane = np.array([[[-1., -1., 0.], [-1., 1., 0.]],
                      [[1., -1., 0.], [1., 1., 0.]]])
    if adapter:
        from mmcore.nurbs._nurbs_eval import NURBSSurfaceTuple
        from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
        first = NURBSSurfaceTuple(3, 2, np.array([0., 0., 0., 1., 1., 1.]),
                                  np.array([0., 0., 1., 1.]), control, weights)
        second = NURBSSurfaceTuple(2, 2, np.array([0., 0., 1., 1.]),
                                   np.array([0., 0., 1., 1.]), plane, np.ones((2, 2)))
    else:
        first = to_homogeneous_2d(control, weights)
        second = to_homogeneous_2d(plane, np.ones((2, 2)))
    if swap:
        first, second = second, first
    result = (nurbs_ssx(first, second, atol=1e-5, max_cells=15000) if adapter
              else bez_ssx(first, second, atol=1e-5, rational=True, max_cells=15000))
    assert result["complete"], result["status"]
    assert len(result["branches"]) == 2
    assert not result["singularities"]
    assert "unresolved_regions" not in result
    expected = np.sqrt(epsilon*(2.+epsilon)) / (2.*(1.+epsilon))
    positions = []
    for branch in result["branches"]:
        xyz = np.asarray(branch.curve[1])
        assert np.ptp(xyz[:, 0]) < 1e-9
        assert np.ptp(xyz[:, 1]) == 2.
        positions.append(float(xyz[:, 0].mean()))
    np.testing.assert_allclose(sorted(positions), [-expected, expected], atol=1e-9, rtol=0.)

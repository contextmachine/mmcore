"""Singular probes must subdivide the equations, not free extrusion axes."""
import numpy as np
import pytest

from mmcore.nurbs._nurbs_eval import to_homogeneous_2d
from mmcore.numeric.intersection.ssx._bez_ssx5 import (
    _Cell, _probe_children, _probe_split_axis, bez_ssx,
)
from mmcore.numeric.intersection.ssx._ssx_substrate import GaussMapBern


def _plane():
    return np.array([[[0., 0., 0.], [0., 1., 0.]],
                     [[1., 0., 0.], [1., 1., 0.]]])


def test_singular_constraint_chooses_active_axis_despite_long_free_spans():
    g = GaussMapBern.from_surf(_plane())
    tensor = np.array([-1., 1.]).reshape(2, 1, 1, 1)
    budget = object()
    cell = _Cell(g, g, [], ((.49, .51), (0., 1.), (0., 1.), (0., 1.)),
                 T1=tensor, T2=np.zeros_like(tensor), T3=np.zeros_like(tensor),
                 T4=np.zeros_like(tensor), work_budget=budget)
    assert _probe_split_axis(cell) == 0
    children = _probe_children(cell)
    assert len(children) == 2
    assert [child.box[0] for child in children] == [(.49, .5), (.5, .51)]
    assert all(child.box[1:] == cell.box[1:] for child in children)
    assert all(child.g2 is g and child.work_budget is budget for child in children)
    np.testing.assert_array_equal(children[0].T1.ravel(), [-1., 0.])
    np.testing.assert_array_equal(children[1].T1.ravel(), [0., 1.])


def test_restriction_rebalances_independently_varying_constraints():
    g = GaussMapBern.from_surf(_plane())
    first = np.array([-2., 2.]).reshape(2, 1, 1, 1)
    second = np.array([-1.5, 1.5]).reshape(1, 2, 1, 1)
    cell = _Cell(g, g, [], ((0., 1.),) * 4, T1=first, T2=second)
    assert _probe_split_axis(cell) == 0
    child = _probe_children(cell)[0]
    # The first derivative bound halved under restriction; the other
    # necessary constraint now has the larger unresolved variation.
    assert _probe_split_axis(child) == 1


def test_constant_tangent_constraints_fall_back_to_psi_variation():
    g1 = GaussMapBern.from_surf(np.zeros((2, 2, 3)))
    surface = np.zeros((2, 2, 3))
    surface[:, 1, 0] = 2.
    g2 = GaussMapBern.from_surf(surface)
    zero = np.zeros((1, 1, 1, 1))
    cell = _Cell(g1, g2, [], ((0., 1.),) * 4, T1=zero, T2=zero,
                 T3=zero, T4=zero)
    assert _probe_split_axis(cell) == 3


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
    assert not result["unresolved_regions"] and not result["singularities"]
    expected = np.sqrt(epsilon*(2.+epsilon)) / (2.*(1.+epsilon))
    positions = []
    for branch in result["branches"]:
        xyz = np.asarray(branch.curve[1])
        assert np.ptp(xyz[:, 0]) < 1e-9
        assert np.ptp(xyz[:, 1]) == 2.
        positions.append(float(xyz[:, 0].mean()))
    np.testing.assert_allclose(sorted(positions), [-expected, expected], atol=1e-9, rtol=0.)

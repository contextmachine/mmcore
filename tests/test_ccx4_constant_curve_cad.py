"""Practical curve/point contacts retain the varying curve's preimages."""

import numpy as np
import pytest

from mmcore.numeric._bezier_common import eval_curve
from mmcore.numeric.intersection.ccx._bez_ccx4 import bez_ccx


def _homogeneous(points, weights):
    weights = np.asarray(weights, dtype=float)
    return np.column_stack((np.asarray(points) * weights[:, None], weights))


def _check_representatives(result, first, second, rational, atol):
    assert result["boundary_topology_complete"]
    assert not result["budget_exhausted"]
    assert result["isolated"]
    for root in result["isolated"]:
        assert 0.0 <= root["u"] <= 1.0
        assert 0.0 <= root["v"] <= 1.0
        a = eval_curve(first, root["u"], rational=rational)
        b = eval_curve(second, root["v"], rational=rational)
        assert np.linalg.norm(a - b) <= atol
        assert np.linalg.norm(root["point"] - a) <= atol
        assert np.linalg.norm(root["point"] - b) <= atol


@pytest.mark.parametrize("rational", [False, True])
@pytest.mark.parametrize("swapped", [False, True])
def test_constant_curve_endpoint_fiber_returns_usable_representatives(rational, swapped):
    # This is the endpoint fiber of the pinch boundary that previously
    # returned three correct points plus dozens of impossible 2D uniqueness
    # obligations.  The free parameter is not an isolated-root coordinate.
    curve = np.array([[-1.0, 0.5, 0.0], [0.5, 0.5, 0.0]])
    point_curve = np.tile([0.5, 0.5, 0.0], (3, 1))
    if rational:
        curve = _homogeneous(curve, [1.0, 1.0])
        point_curve = _homogeneous(point_curve, [0.5, 2.0, 4.0])
    first, second = (point_curve, curve) if swapped else (curve, point_curve)
    result = bez_ccx(first, second, rational=rational, atol=1e-3, max_cells=1_000)
    _check_representatives(result, first, second, rational, 1e-3)
    varying = "v" if swapped else "u"
    assert all(abs(root[varying] - 1.0) < 1e-8 for root in result["isolated"])


@pytest.mark.parametrize("swapped", [False, True])
def test_constant_curve_keeps_both_preimages_on_folded_partner(swapped):
    # x(u)=(u-1/4)(u-3/4): one point has two separated preimages.
    curve = np.array([[3 / 16, 0.5, 0.0], [-5 / 16, 0.5, 0.0],
                      [3 / 16, 0.5, 0.0]])
    point_curve = np.tile([0.0, 0.5, 0.0], (2, 1))
    first, second = (point_curve, curve) if swapped else (curve, point_curve)
    result = bez_ccx(first, second, atol=1e-6, max_cells=2_000)
    _check_representatives(result, first, second, False, 1e-6)
    varying = "v" if swapped else "u"
    parameters = np.array([root[varying] for root in result["isolated"]])
    assert np.min(abs(parameters - 0.25)) < 1e-7
    assert np.min(abs(parameters - 0.75)) < 1e-7
    assert np.all(np.minimum(abs(parameters - 0.25), abs(parameters - 0.75)) < 1e-7)


def test_constant_curve_near_contact_uses_cad_distance():
    curve = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    point_curve = np.tile([0.5, 0.0, 5e-4], (3, 1))
    result = bez_ccx(curve, point_curve, atol=1e-3, max_cells=1_000)
    _check_representatives(result, curve, point_curve, False, 1e-3)
    assert min(abs(root["u"] - 0.5) for root in result["isolated"]) < 1e-6


def test_constant_curve_still_reports_resource_limits():
    curve = np.array([[-1.0, 0.5, 0.0], [0.5, 0.5, 0.0]])
    point_curve = np.tile([0.5, 0.5, 0.0], (3, 1))
    result = bez_ccx(curve, point_curve, max_cells=1)
    assert result["budget_exhausted"]
    assert not result["boundary_topology_complete"]
    assert result["cells_processed"] <= 1


def test_nonconstant_fold_retains_two_parameter_pairs():
    line = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    folded = np.array([[0.5, 3 / 16, 0.0], [0.5, -5 / 16, 0.0],
                       [0.5, 3 / 16, 0.0]])
    result = bez_ccx(line, folded, atol=1e-6, max_cells=2_000)
    _check_representatives(result, line, folded, False, 1e-6)
    parameters = np.array([root["v"] for root in result["isolated"]])
    assert np.min(abs(parameters - 0.25)) < 1e-7
    assert np.min(abs(parameters - 0.75)) < 1e-7

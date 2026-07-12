"""Exact-topology contracts for curve/surface intersections."""

import numpy as np
import pytest

from mmcore.numeric.intersection.csx._bez_csx4 import bez_csx


def _homogeneous_curve(points, weights=None):
    points = np.asarray(points, dtype=np.float64)
    if weights is None:
        weights = np.ones(len(points))
    weights = np.asarray(weights, dtype=np.float64)
    return np.column_stack([points * weights[:, None], weights])


def _homogeneous_surface(points):
    points = np.asarray(points, dtype=np.float64)
    return np.concatenate(
        [points, np.ones(points.shape[:-1] + (1,))], axis=-1)


def _plane():
    return np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
    ])


@pytest.mark.parametrize("weights", [None, [1.0, 3.0]])
def test_offset_curve_is_not_an_overlap_or_root(weights):
    curve = _homogeneous_curve(
        [[0.0, 0.5, 5e-4], [1.0, 0.5, 5e-4]], weights)
    result = bez_csx(
        curve, _homogeneous_surface(_plane()),
        atol=1e-3, rational=True)

    assert result["isolated"] == []
    assert result["overlaps"] == []
    assert result["parameter_fibers"] == []
    assert result["budget_exhausted"] is False


def test_endpoint_only_hump_is_not_promoted_to_overlap():
    curve = _homogeneous_curve([
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 1e-3],
        [1.0, 0.5, 0.0],
    ])
    result = bez_csx(
        curve, _homogeneous_surface(_plane()),
        atol=1e-3, rational=True)

    assert result["overlaps"] == []
    assert result["budget_exhausted"] is False
    assert sorted(p["t"] for p in result["isolated"]) == pytest.approx(
        [0.0, 1.0], abs=1e-8)


def test_exact_straight_overlap_is_preserved():
    curve = _homogeneous_curve([
        [0.0, 0.5, 0.0], [1.0, 0.5, 0.0]])
    result = bez_csx(
        curve, _homogeneous_surface(_plane()),
        atol=1e-3, rational=True)

    assert len(result["overlaps"]) == 1
    assert result["budget_exhausted"] is False


def test_exact_curved_affine_parameter_overlap_is_preserved():
    curve = _homogeneous_curve([
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 0.0],
        [1.0, 0.5, 1.0],
    ])
    surface = np.array([
        [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        [[0.5, 0.0, 0.0], [0.5, 1.0, 0.0]],
        [[1.0, 0.0, 1.0], [1.0, 1.0, 1.0]],
    ])
    result = bez_csx(
        curve, _homogeneous_surface(surface),
        atol=1e-3, rational=True)

    assert len(result["overlaps"]) == 1
    assert result["budget_exhausted"] is False


def test_collapsed_rational_curve_requires_exact_surface_membership():
    point = np.array([0.5, 0.5, 5e-4])
    curve = _homogeneous_curve([point, point], weights=[1.0, 2.0])
    result = bez_csx(
        curve, _homogeneous_surface(_plane()),
        atol=1e-3, rational=True)

    assert result["parameter_fibers"] == []
    assert result["isolated"] == []
    assert result["budget_exhausted"] is False

    exact = point.copy()
    exact[2] = 0.0
    exact_curve = _homogeneous_curve([exact, exact], weights=[1.0, 2.0])
    exact_result = bez_csx(
        exact_curve, _homogeneous_surface(_plane()),
        atol=1e-3, rational=True)
    assert len(exact_result["parameter_fibers"]) == 1
    assert exact_result["budget_exhausted"] is False


def test_exterior_boundary_root_rejection_is_translation_invariant():
    """A large common origin must not turn an exterior root into a root."""
    origin = 1.0e6
    surface = np.array([
        [[origin, 0.0, 0.0], [origin, 1.0, 0.0]],
        [[origin + 1.0, 0.0, 0.0], [origin + 1.0, 1.0, 0.0]],
    ])
    # The polynomial continuation meets the plane at u=-1e-8, outside the
    # surface domain.  On u=0 the residual remains exactly nonzero.
    curve = np.array([
        [origin - 1.0e-8, 0.5, -1.0],
        [origin - 1.0e-8, 0.5, 1.0],
    ])

    result = bez_csx(curve, surface, atol=1e-3, rational=False)

    assert result["isolated"] == []
    assert result["overlaps"] == []
    assert result["budget_exhausted"] is False


def test_translated_sub_tolerance_line_is_not_a_surface_overlap():
    """Exact-set membership is invariant under a large common translation."""
    origin = 2.0e10
    gap = 5.0e-4
    surface = np.array([
        [[origin, 0.0, 0.0], [origin, 0.0, 1.0]],
        [[origin, 1.0, 0.0], [origin, 1.0, 1.0]],
    ])
    curve = np.array([
        [origin + gap, 0.0, 0.5],
        [origin + gap, 1.0, 0.5],
    ])

    result = bez_csx(curve, surface, atol=1e-3, rational=False)

    assert result["isolated"] == []
    assert result["overlaps"] == []
    assert result["parameter_fibers"] == []
    assert result["budget_exhausted"] is False

    exact_curve = curve.copy()
    exact_curve[:, 0] = origin
    exact_result = bez_csx(
        exact_curve, surface, atol=1e-3, rational=False)
    assert len(exact_result["overlaps"]) == 1
    assert exact_result["budget_exhausted"] is False


def test_collapsed_fiber_identity_is_translation_invariant():
    """A relative-to-world-origin envelope cannot type a tolerance gap."""
    origin = 1.0e8
    gap = 5.0e-4
    surface_xyz = np.array([
        [[origin, 0.0, 0.0], [origin, 1.0, 0.0]],
        [[origin, 0.0, 1.0], [origin, 1.0, 1.0]],
    ])
    point = np.array([origin + gap, 0.5, 0.5])
    curve = _homogeneous_curve([point, point], weights=[1.0, 2.0])

    result = bez_csx(
        curve, _homogeneous_surface(surface_xyz),
        atol=1e-3, rational=True)

    assert result["parameter_fibers"] == []
    assert result["isolated"] == []
    assert result["budget_exhausted"] is False


def test_in_axis_drift_beyond_float_built_floor_is_not_certified():
    """L52 slice 6b: the shared two-term envelope (ccx structure).

    The former folded ``4096*n1*n2*eps_f64`` envelope certified a 3.0e-11
    in-axis control drift (~135k ulps of O(1) data — real geometry, not
    roundoff) as an EXACT affine overlap on this cubic/bilinear pair; the
    reconciled envelope refuses it. The 1e-12 companion pins the floor
    from below: legitimate float-built restriction roundoff (the
    8192*(n1+n2)*eps_f64 source family, calibrated by ccx's
    float-built-subcurve fixture) must keep certifying.
    """
    from mmcore.numeric.intersection.csx._bez_csx4 import (
        _certify_affine_csx_overlap)

    S = np.array([[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                  [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]])

    def drifted(dx):
        return np.array([[0.0, 0.5, 0.0],
                         [1.0 / 3.0 + dx, 0.5, 0.0],
                         [2.0 / 3.0 + dx, 0.5, 0.0],
                         [1.0, 0.5, 0.0]])

    a, b = (0.0, 0.0, 0.5), (1.0, 1.0, 0.5)
    assert _certify_affine_csx_overlap(drifted(0.0), S, a, b, rational=False)
    assert _certify_affine_csx_overlap(drifted(1e-12), S, a, b,
                                       rational=False)
    assert not _certify_affine_csx_overlap(drifted(3.0e-11), S, a, b,
                                           rational=False)
    # single-axis offsets stay rejected at every magnitude (the
    # per-coordinate source scale keeps a dz-sized floor on the z axis)
    dz_off = np.array([[0.0, 0.5, 1e-14],
                       [1.0 / 3.0, 0.5, 1e-14],
                       [2.0 / 3.0, 0.5, 1e-14],
                       [1.0, 0.5, 1e-14]])
    assert not _certify_affine_csx_overlap(dz_off, S, a, b, rational=False)

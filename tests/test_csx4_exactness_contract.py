"""CAD contact tolerance and optional exact-mode regression contracts."""

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
def test_offset_curve_is_a_tolerance_overlap_never_exact(weights):
    # L59 (USER DECISION 2026-07-12, theorem-first tier — the CCX-L56
    # twin): a 5e-4-offset line with both ends domain-pinned and no
    # crossing flips IS one tolerance-coincident span. The exactness
    # property survives sharpened: it must never certify 'exact'.
    curve = _homogeneous_curve(
        [[0.0, 0.5, 5e-4], [1.0, 0.5, 5e-4]], weights)
    result = bez_csx(
        curve, _homogeneous_surface(_plane()),
        atol=1e-3, rational=True)

    assert result["isolated"] == []
    assert result["parameter_fibers"] == []
    assert result["budget_exhausted"] is False
    assert len(result["overlaps"]) == 1
    o = result["overlaps"][0]
    assert o["certification"] == "tolerance"
    assert o["residual_max"] == pytest.approx(5e-4, rel=1e-6)
    assert o["t_range"][0] == pytest.approx(0.0, abs=1e-9)
    assert o["t_range"][1] == pytest.approx(1.0, abs=1e-9)


def test_sub_tolerance_hump_is_one_tolerance_overlap():
    # L59 / the CCX-L56 hump twin: apex deviation 5e-4 (ctrl 1e-3 / 2)
    # with exact end touches and no transverse flip — one tolerance
    # overlap whose span carries the endpoint touches.
    curve = _homogeneous_curve([
        [0.0, 0.5, 0.0],
        [0.5, 0.5, 1e-3],
        [1.0, 0.5, 0.0],
    ])
    result = bez_csx(
        curve, _homogeneous_surface(_plane()),
        atol=1e-3, rational=True)

    assert result["budget_exhausted"] is False
    assert result["isolated"] == []
    assert len(result["overlaps"]) == 1
    o = result["overlaps"][0]
    assert o["certification"] == "tolerance"
    assert o["residual_max"] == pytest.approx(5e-4, rel=1e-6)


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


@pytest.mark.parametrize("gap, expected", [(0., 1), (5e-4, 1), (2e-3, 0)])
@pytest.mark.parametrize("origin", [0., 1e8])
def test_collapsed_rational_curve_uses_cad_surface_membership(gap, expected, origin):
    from mmcore.numeric._bezier_common import eval_curve, eval_surface

    point = np.array([0.5, 0.5, origin + gap])
    curve = _homogeneous_curve([point, point], weights=[1.0, 2.0])
    surface = _plane()
    surface[..., 2] += origin
    surface = _homogeneous_surface(surface)
    result = bez_csx(curve, surface, atol=1e-3, rational=True)

    assert len(result["parameter_fibers"]) == expected
    assert result["isolated"] == [] and result["overlaps"] == []
    assert result["budget_exhausted"] is False
    for fiber in result["parameter_fibers"]:
        assert fiber["t_range"] == (0., 1.)
        assert 0. <= fiber["u"] <= 1. and 0. <= fiber["v"] <= 1.
        for t in (0., .5, 1.):
            assert np.linalg.norm(
                eval_curve(curve, t, rational=True)
                - eval_surface(surface, fiber["u"], fiber["v"], rational=True)) <= 1e-3

    strict = bez_csx(curve, surface, atol=1e-3, rational=True,
                     tolerance_tier=False)
    assert len(strict["parameter_fibers"]) == int(gap == 0.)
    assert strict["isolated"] == [] and strict["overlaps"] == []
    assert strict["budget_exhausted"] is False


@pytest.mark.parametrize("gap, expected", [(1e-8, 1), (5e-4, 1), (2e-3, 0)])
@pytest.mark.parametrize("origin", [0., 1e6])
def test_exterior_boundary_contact_respects_cad_tolerance(gap, expected, origin):
    from mmcore.numeric._bezier_common import eval_curve, eval_surface

    surface = _plane()
    surface[..., 0] += origin
    # The exact continuation root is outside the patch. CAD keeps a
    # contact on its edge when the actual geometric gap is within atol.
    curve = np.array([[origin - gap, .5, -1.], [origin - gap, .5, 1.]])
    result = bez_csx(curve, surface, atol=1e-3, rational=False)

    assert len(result["isolated"]) == expected
    assert result["overlaps"] == [] and result["parameter_fibers"] == []
    assert result["budget_exhausted"] is False
    for root in result["isolated"]:
        assert root["certification"] == "tolerance"
        assert root["d_min"] <= 1e-3
        assert all(0. <= root[p] <= 1. for p in ("t", "u", "v"))
        assert np.linalg.norm(eval_curve(curve, root["t"], rational=False)
                              - eval_surface(surface, root["u"], root["v"],
                                             rational=False)) <= 1e-3
        assert np.linalg.norm(root["point"] - [origin, .5, 0.]) <= 1e-3

    strict = bez_csx(curve, surface, atol=1e-3, rational=False,
                     tolerance_tier=False)
    assert strict["isolated"] == [] and strict["overlaps"] == []
    assert strict["budget_exhausted"] is False


def test_translated_sub_tolerance_line_certification_is_translation_invariant():
    """Translation preserves CAD classification within input resolution.

    At 2e10 one float64 ULP is about 3.8e-6, already larger than the
    old 5e-7 residual-comparison requirement. A few input ULPs are an
    appropriate allowance for this measurement; the public residual
    must still remain below the unchanged 1e-3 CAD tolerance.
    """
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
    assert result["parameter_fibers"] == []
    assert result["budget_exhausted"] is False
    assert len(result["overlaps"]) == 1
    far = result["overlaps"][0]
    assert far["certification"] == "tolerance"

    curve0 = curve.copy(); curve0[:, 0] -= origin
    surface0 = surface.copy(); surface0[..., 0] -= origin
    near = bez_csx(curve0, surface0, atol=1e-3, rational=False)["overlaps"][0]
    assert near["certification"] == "tolerance"
    measurement_allowance = 4. * np.spacing(origin)
    assert measurement_allowance < 0.02 * 1e-3
    assert abs(far["residual_max"] - near["residual_max"]) <= measurement_allowance
    assert max(far["residual_max"], near["residual_max"]) <= 1e-3

    exact_curve = curve.copy()
    exact_curve[:, 0] = origin
    exact_result = bez_csx(
        exact_curve, surface, atol=1e-3, rational=False)
    assert len(exact_result["overlaps"]) == 1
    assert exact_result["budget_exhausted"] is False


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


def test_exclusion_prune_carries_the_L1_roundoff_margin():
    """L52 slice 6c: `_residual_excludes_zero` refuses sub-margin clearance.

    §4 invariant: sign/hull exclusion only beyond k*eps*max|coeff|. A
    wrongful exclusion was NOT reached in practice (240 exact-Fraction
    restriction-chain comparisons: 0 flips, 0 exclusions within
    1e-12*scale of the boundary) — the margin is invariant compliance
    with measured zero practical impact, insurance against restriction
    chains this probe family does not cover.
    """
    from mmcore.numeric.intersection.csx._bez_csx4 import (
        _residual_excludes_zero)

    eps = np.finfo(np.float64).eps
    # One component clears zero by well over the 128*eps margin: excludes.
    clear = np.zeros((2, 2, 2, 3))
    clear[..., 0] = 1.0
    clear[..., 1] = np.array([[[1.0, 2.0], [1.0, 2.0]],
                              [[1.0, 2.0], [1.0, 2.0]]])
    clear[..., 2] = -1.0
    assert _residual_excludes_zero(clear)

    # Every component's clearance sits INSIDE the margin (min = 10*eps of
    # a max-magnitude-1 net): roundoff of the restriction chain could hide
    # a true sign change, so the prune must refuse.
    hairline = np.zeros((2, 2, 2, 3))
    for c in range(3):
        hairline[..., c] = 1.0
        hairline[0, 0, 0, c] = 10.0 * eps
    assert not _residual_excludes_zero(hairline)

    # Straddling hulls never exclude, margin or not.
    straddle = np.zeros((2, 2, 2, 3))
    straddle[0, 0, 0, :] = -1.0
    straddle[1, 1, 1, :] = 1.0
    assert not _residual_excludes_zero(straddle)

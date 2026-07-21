"""P1 whole-call normalization: helpers + invariance property (2026-07-21 design).

Spec: docs/superpowers/specs/2026-07-21-ssx5-invariance-normalization-design.md
"""

import numpy as np
import pytest

from mmcore.numeric.intersection.ssx._bez_ssx5 import (
    _ssx_normalization_context,
    _normalize_surface_net,
)


def _homog(S):
    S = np.asarray(S, dtype=np.float64)
    return np.concatenate([S, np.ones(S.shape[:-1] + (1,))], axis=-1)


def test_context_power_of_two_scale_and_center():
    # Joint AABB [0,10]^3 -> diag = 10*sqrt(3) ~ 17.32, log2 ~ 4.11 -> k = 16.
    s1 = np.array([[[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]], [[10.0, 0.0, 0.0], [10.0, 10.0, 0.0]]])
    s2 = np.array([[[0.0, 0.0, 10.0], [0.0, 10.0, 10.0]], [[10.0, 0.0, 10.0], [10.0, 10.0, 10.0]]])
    c, k = _ssx_normalization_context(s1, s2, rational=False)
    assert k == 16.0
    assert np.allclose(c, [5.0, 5.0, 5.0])
    # k is a power of two: scaling is mantissa-exact and reversible bit-for-bit.
    rng = np.random.default_rng(3)
    pts = rng.uniform(-1e4, 1e4, (64, 3))
    assert np.array_equal((pts / k) * k, pts)


def test_context_rational_uses_dehomogenized_points():
    s = np.array([[[0.0, 0.0, 0.0], [0.0, 4.0, 0.0]], [[4.0, 0.0, 0.0], [4.0, 4.0, 0.0]]])
    h = _homog(s)
    h2 = h.copy()
    h2[..., :3] *= 2.0  # same Cartesian points, w-scaled numerators would differ
    h2[..., 3] *= 2.0
    c1, k1 = _ssx_normalization_context(h, h, rational=True)
    c2, k2 = _ssx_normalization_context(h2, h2, rational=True)
    assert np.allclose(c1, c2) and k1 == k2


def test_context_degenerate_inputs_yield_identity():
    good = _homog(np.zeros((2, 2, 3)))
    bad_w = good.copy()
    bad_w[0, 0, 3] = 0.0
    c, k = _ssx_normalization_context(bad_w, good, rational=True)
    assert k == 1.0 and np.all(c == 0.0)
    bad_nan = np.zeros((2, 2, 3))
    bad_nan_ = bad_nan.copy()
    bad_nan_[0, 0, 0] = np.nan
    c, k = _ssx_normalization_context(bad_nan_, bad_nan, rational=False)
    assert k == 1.0 and np.all(c == 0.0)
    # Zero extent (all points coincide) -> identity, per spec.
    pt = np.full((2, 2, 3), 7.0)
    c, k = _ssx_normalization_context(pt, pt, rational=False)
    assert k == 1.0 and np.all(c == 0.0)


def test_normalize_surface_net_round_trip():
    rng = np.random.default_rng(11)
    s = rng.uniform(2350.0, 3200.0, (3, 4, 3))
    c, k = _ssx_normalization_context(s, s, rational=False)
    n = _normalize_surface_net(s, c, k, rational=False)
    # Normalized coords are O(1) and the map inverts to roundoff at world scale.
    assert np.max(np.abs(n)) <= 2.0
    assert np.allclose(n * k + c, s, atol=1e-9)
    assert not np.shares_memory(n, s)


def test_normalize_surface_net_rational_preserves_cartesian_points():
    from mmcore.numeric.intersection._bezier_common import eval_surface

    rng = np.random.default_rng(5)
    s = rng.uniform(900.0, 1100.0, (3, 3, 3))
    h = _homog(s)
    h[..., 3] = rng.uniform(0.5, 2.0, (3, 3))  # non-unit weights,
    h[..., :3] = s * h[..., 3:]  # numerators kept consistent
    c, k = _ssx_normalization_context(h, h, rational=True)
    n = _normalize_surface_net(h, c, k, rational=True)
    for u, v in [(0.0, 0.0), (0.3, 0.7), (1.0, 1.0)]:
        pw = eval_surface(h, u, v, rational=True)
        pn = eval_surface(n, u, v, rational=True)
        assert np.allclose(pn * k + c, pw, atol=1e-9)
    # Weights are frame-invariant: untouched by the transform.
    assert np.array_equal(n[..., 3], h[..., 3])


# SSXBranch/SSXPoint are re-exported through _bez_ssx5's namespace (:39-40).
from mmcore.numeric.intersection.ssx._bez_ssx5 import (
    _denormalize_result,
    SSXSingularity,
    SSXBranch,
    SSXPoint,
)


def _fake_result(branches=(), points=(), singularities=()):
    return {
        "branches": list(branches),
        "points": list(points),
        "singularities": list(singularities),
        "overlap_regions": [],
        "unresolved_regions": [],
        "complete": True,
        "status": {"reasons": [], "work": {}},
    }


def test_denormalize_maps_all_xyz_payloads_once():
    c, k = np.array([100.0, -50.0, 7.0]), 8.0
    stuv = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.5, 0.5]])
    xyz_n = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b = SSXBranch(curve=(stuv, xyz_n.copy()))
    p = SSXPoint(stuv=stuv[0].copy(), xyz=xyz_n[0].copy())
    s = SSXSingularity(kind="tangent_point", stuv=stuv[1].copy(), xyz=xyz_n[1].copy())
    r = _denormalize_result(_fake_result([b], [p], [s]), c, k)
    assert np.allclose(r["branches"][0].curve[1], xyz_n * k + c)
    # stuv is parameter-space: bit-identical, same object.
    assert r["branches"][0].curve[0] is stuv
    assert np.allclose(r["points"][0].xyz, xyz_n[0] * k + c)
    assert np.allclose(r["singularities"][0].xyz, xyz_n[1] * k + c)
    assert np.allclose(r["singularities"][0].stuv, stuv[1])


def test_denormalize_identity_is_noop_same_objects():
    xyz = np.array([[1.0, 2.0, 3.0]])
    b = SSXBranch(curve=(np.zeros((1, 4)), xyz))
    r = _denormalize_result(_fake_result([b]), np.zeros(3), 1.0)
    assert r["branches"][0].curve[1] is xyz


def test_denormalize_aliased_object_mapped_once():
    c, k = np.array([10.0, 0.0, 0.0]), 2.0
    p = SSXPoint(stuv=np.zeros(4), xyz=np.array([1.0, 1.0, 1.0]))
    r = _denormalize_result(_fake_result(points=[p, p]), c, k)  # same object twice
    assert np.allclose(r["points"][0].xyz, [12.0, 2.0, 2.0])
    assert r["points"][1] is r["points"][0]

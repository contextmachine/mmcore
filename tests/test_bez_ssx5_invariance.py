"""P1 whole-call normalization: helpers + invariance property (2026-07-21 design).

Spec: docs/superpowers/specs/2026-07-21-ssx5-invariance-normalization-design.md
"""

import numpy as np
import pytest

from mmcore.numeric.intersection.ssx._bez_ssx5 import (
    _ssx_normalization_context,
    _normalize_surface_net,
    _NORM_IDENTITY_WINDOW,
)


def _homog(S):
    S = np.asarray(S, dtype=np.float64)
    return np.concatenate([S, np.ones(S.shape[:-1] + (1,))], axis=-1)


def test_context_power_of_two_scale_and_center():
    # Joint AABB [0,160]^3 (outside the identity window) -> diag = 160*sqrt(3)
    # ~ 277.1, log2 ~ 8.11 -> k = 256.
    s1 = np.array([[[0.0, 0.0, 0.0], [0.0, 160.0, 0.0]], [[160.0, 0.0, 0.0], [160.0, 160.0, 0.0]]])
    s2 = np.array([[[0.0, 0.0, 160.0], [0.0, 160.0, 160.0]], [[160.0, 0.0, 160.0], [160.0, 160.0, 160.0]]])
    c, k = _ssx_normalization_context(s1, s2, rational=False)
    assert k == 256.0
    assert np.allclose(c, [80.0, 80.0, 80.0])
    # k is a power of two: scaling is mantissa-exact and reversible bit-for-bit.
    rng = np.random.default_rng(3)
    pts = rng.uniform(-1e4, 1e4, (64, 3))
    assert np.array_equal((pts / k) * k, pts)


def test_context_identity_window():
    # 2026-07-21 amendment: models whose joint coordinate magnitude lies in
    # the proven band keep the identity frame — re-framing near-origin models
    # regressed 4 singular fixtures (exact-structure rounding + absolute
    # singular-tier thresholds); the trace-certificate defect only appears
    # at magnitudes >= ~71.  Outside the band (either side) we normalize.
    lo_w, hi_w = _NORM_IDENTITY_WINDOW
    assert lo_w == 2.0**-5 and hi_w == 2.0**5
    inside = np.array([[[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]], [[10.0, 0.0, 0.0], [10.0, 10.0, 10.0]]])
    c, k = _ssx_normalization_context(inside, inside, rational=False)
    assert k == 1.0 and np.all(c == 0.0)
    # Below the band: tiny model is scaled UP (k < 1), mantissa-exactly.
    tiny = inside * 1e-3
    c, k = _ssx_normalization_context(tiny, tiny, rational=False)
    assert 0.0 < k < 1.0
    # Degenerate far point: outside the band but zero extent -> identity.
    far_pt = np.full((2, 2, 3), 1000.0)
    c, k = _ssx_normalization_context(far_pt, far_pt, rational=False)
    assert k == 1.0 and np.all(c == 0.0)


def test_context_rational_uses_dehomogenized_points():
    s = np.array([[[0.0, 0.0, 0.0], [0.0, 400.0, 0.0]], [[400.0, 0.0, 0.0], [400.0, 400.0, 0.0]]])
    h = _homog(s)
    h2 = h.copy()
    h2[..., :3] *= 2.0  # same Cartesian points, w-scaled numerators would differ
    h2[..., 3] *= 2.0
    c1, k1 = _ssx_normalization_context(h, h, rational=True)
    c2, k2 = _ssx_normalization_context(h2, h2, rational=True)
    assert k1 != 1.0  # outside the identity window: the transform is real
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


from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx


def test_bez_ssx_world_in_world_out_offset_planes():
    # Plane pair from the singular suite, pushed to case-11-like offsets.
    # z=5 sheet vs 0->10 ramp: intersection line x=5, z=5, y in [0,10].
    off = np.array([1e4, -2e4, 3e3])
    s1 = np.array([[[0.0, 0.0, 5.0], [0.0, 10.0, 5.0]], [[10.0, 0.0, 5.0], [10.0, 10.0, 5.0]]]) + off
    s2 = np.array([[[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]], [[10.0, 0.0, 10.0], [10.0, 10.0, 10.0]]]) + off
    r = bez_ssx(s1, s2, 1e-3, rational=False)
    assert r["complete"], r["status"]["reasons"]
    assert len(r["branches"]) == 1
    xyz = np.asarray(r["branches"][0].curve[1], dtype=float)
    assert np.all(np.abs(xyz[:, 0] - (off[0] + 5.0)) <= 5e-3)
    assert np.all(np.abs(xyz[:, 2] - (off[2] + 5.0)) <= 5e-3)
    assert xyz[:, 1].min() <= off[1] + 0.5 and xyz[:, 1].max() >= off[1] + 9.5
    stuv = np.asarray(r["branches"][0].curve[0], dtype=float)
    assert stuv.min() >= -1e-9 and stuv.max() <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Gate 5: bez_ssx(S*k + c, atol*k) must be equivalent to bez_ssx(S, atol)
# for translations up to ~1e4 and scales k in [1e-2, 1e3] (kickoff list).
# Reference runs at the pair's native coords (INSIDE the identity window,
# so they take the bit-for-bit legacy path); transformed runs place the
# model at offset/scale (outside the window -> canonical frame), so each
# parametrization compares the identity path against the normalized path
# across the window cliff.  Topology must match exactly; geometry must
# match through the map within a few atol.  Transversal pairs only:
# singular structure at extreme scales is the documented P1b limit.
# ---------------------------------------------------------------------------


def _plane_pair():
    s1 = np.array([[[0.0, 0.0, 5.0], [0.0, 10.0, 5.0]], [[10.0, 0.0, 5.0], [10.0, 10.0, 5.0]]])
    s2 = np.array([[[0.0, 0.0, 0.0], [0.0, 10.0, 0.0]], [[10.0, 0.0, 10.0], [10.0, 10.0, 10.0]]])
    return s1, s2, False


def _loop_pair():
    # Biquadratic bowl z = (2u-1)^2 + (2v-1)^2 (Bernstein z-coeffs [1,-1,1]
    # per axis, summed) against the plane z = 0.5: one closed transversal
    # loop strictly inside the domain.
    g = [0.0, 0.5, 1.0]
    zc = [1.0, -1.0, 1.0]
    s1 = np.array([[[g[i], g[j], zc[i] + zc[j]] for j in range(3)] for i in range(3)])
    s2 = np.array([[[-0.5, -0.5, 0.5], [-0.5, 1.5, 0.5]], [[1.5, -0.5, 0.5], [1.5, 1.5, 0.5]]])
    return s1, s2, False


def _rational_pair():
    # 90-degree circular-arc strip (radius 1, weights [1, sqrt(2)/2, 1])
    # extruded along y, against the plane z = 0.5: one transversal line
    # x = sqrt(3)/2 crossing the strip.  Exercises the homogeneous branch
    # of the transform.
    w = np.sqrt(2.0) / 2.0
    arc = [((1.0, 0.0), 1.0), ((1.0, 1.0), w), ((0.0, 1.0), 1.0)]
    s1 = np.zeros((3, 2, 4))
    for i, ((x, z), wi) in enumerate(arc):
        for j, y in enumerate((0.0, 1.0)):
            s1[i, j] = [x * wi, y * wi, z * wi, wi]
    s2 = np.array([[[-0.5, -0.5, 0.5], [-0.5, 1.5, 0.5]], [[1.5, -0.5, 0.5], [1.5, 1.5, 0.5]]])
    s2 = np.concatenate([s2, np.ones((2, 2, 1))], axis=-1)
    return s1, s2, True


PAIRS = [("planes", _plane_pair), ("loop", _loop_pair), ("rational-arc", _rational_pair)]

TRANSFORMS = [
    (np.array([1e3, -2e3, 5e2]), 1.0),
    (np.zeros(3), 1e-2),
    (np.zeros(3), 1e3),
    (np.array([1e4, 1e4, -1e4]), 1e3),
    (np.array([-5e3, 3e3, 1e4]), 1e-2),
]


def _apply_world_transform(S, c, k, rational):
    S = np.asarray(S, dtype=np.float64).copy()
    if rational:
        S[..., :3] = S[..., :3] * k + np.asarray(c) * S[..., 3:]
    else:
        S = S * k + np.asarray(c)
    return S


def _pt_seg_d(p, a, b):
    ab = b - a
    den = float(np.dot(ab, ab))
    t = 0.0 if den <= 0.0 else float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _poly_hausdorff(A, B):
    def directed(P, Q):
        if len(Q) == 1:
            return max(float(np.linalg.norm(p - Q[0])) for p in P)
        return max(min(_pt_seg_d(p, Q[i], Q[i + 1]) for i in range(len(Q) - 1)) for p in P)

    return max(directed(A, B), directed(B, A))


def _topology_signature(r):
    return (
        r["complete"],
        tuple(sorted(r["status"]["reasons"])),
        tuple(sorted((b.kind, bool(b.closed)) for b in r["branches"])),
        len(r["points"]),
        tuple(sorted(s.kind for s in r["singularities"])),
    )


@pytest.mark.parametrize("pair_name,make_pair", PAIRS)
@pytest.mark.parametrize("c,k", TRANSFORMS, ids=[f"c{i}" for i in range(len(TRANSFORMS))])
def test_bez_ssx_similarity_invariance(pair_name, make_pair, c, k):
    atol = 1e-3
    s1, s2, rational = make_pair()
    ref = bez_ssx(s1, s2, atol, rational=rational)
    assert ref["complete"], (pair_name, ref["status"]["reasons"])

    t1 = _apply_world_transform(s1, c, k, rational)
    t2 = _apply_world_transform(s2, c, k, rational)
    res = bez_ssx(t1, t2, atol * k, rational=rational)

    assert _topology_signature(res) == _topology_signature(ref), (pair_name, c, k, res["status"])

    # Geometry: each reference branch, mapped into the transformed frame,
    # must coincide with exactly one result branch within 10 atol_world
    # (chord-sampling differences between two float-distinct runs included).
    tol = 10.0 * atol * k
    remaining = list(range(len(res["branches"])))
    for rb in ref["branches"]:
        mapped = np.asarray(rb.curve[1], dtype=float) * k + c
        dists = [(_poly_hausdorff(mapped, np.asarray(res["branches"][j].curve[1], dtype=float)), j) for j in remaining]
        d, j = min(dists)
        assert d <= tol, (pair_name, c, k, d, tol)
        remaining.remove(j)
    assert not remaining

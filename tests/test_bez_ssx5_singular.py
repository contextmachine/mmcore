"""tests/test_bez_ssx5_singular.py — singular-case handling per Cheng et al. 2023."""
import numpy as np
import pytest

from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx, SSXSingularity


def test_result_has_singularities_key_and_branch_kind():
    # plain transversal case (planes) — no singularities, but the key exists
    s1 = np.array([[[0., 0., 5.], [0., 10., 5.]], [[10., 0., 5.], [10., 10., 5.]]])
    s2 = np.array([[[0., 0., 0.], [0., 10., 0.]], [[10., 0., 10.], [10., 10., 10.]]])
    r = bez_ssx(s1, s2, 1e-3, rational=False)
    assert "singularities" in r
    assert r["singularities"] == []
    assert all(b.kind in ("transversal", "tangential", "overlap") for b in r["branches"])


# ---------------------------------------------------------------------------
# Task 2: _ssx5_singular.py — nets + zero-dimensional Bernstein solver
# ---------------------------------------------------------------------------

from mmcore.numeric.intersection.ssx._ssx5_singular import (
    BoxNet, psi_vector_net, linear_net_4d, sigma_normal_net, solve_zero_dim,
)
from mmcore.numeric.intersection._bezier_common import eval_surface, eval_surface_d1


def _homog(S):
    return np.concatenate([S, np.ones(S.shape[:-1] + (1,))], axis=-1)


def test_psi_vector_net_matches_direct_eval():
    rng = np.random.default_rng(7)
    S1 = rng.uniform(-2, 2, (3, 3, 3)); S2 = rng.uniform(-2, 2, (4, 2, 3))
    G = psi_vector_net(_homog(S1), _homog(S2))          # (3,3,4,2,3)
    from mmcore.numeric.bern import bernstein_eval_nd
    for pt in rng.uniform(0, 1, (10, 4)):
        s, t, u, v = pt
        direct = eval_surface(_homog(S1), s, t, rational=True) - eval_surface(_homog(S2), u, v, rational=True)
        via_net = bernstein_eval_nd(G, np.array([s, t, u, v]))
        assert np.allclose(via_net, direct, atol=1e-12)


def test_linear_net_4d_matches():
    from mmcore.numeric.bern import bernstein_eval_nd
    L = linear_net_4d(c0=-0.3, coeffs=(1.0, -2.0, 0.5, 3.0))   # (2,2,2,2,1)
    for pt in np.random.default_rng(3).uniform(0, 1, (8, 4)):
        want = -0.3 + pt @ np.array([1.0, -2.0, 0.5, 3.0])
        # NOTE: bernstein_eval_nd(L, pt) returns a shape-(1,) array (value dim
        # kept even when all param axes are scalar); numpy >= 2 no longer
        # allows float() on a non-0-d array (even size-1), so use .item().
        assert abs(bernstein_eval_nd(L, pt).item() - want) < 1e-13


def test_solve_zero_dim_finds_plane_slice_roots():
    # transversal bilinear pair; slice Psi with the mid-plane s = 0.5.
    # ground truth: CSX of the s=0.5 isoline of S1 against S2.
    s1 = np.array([[[0., 0., 0.], [0., 10., 0.]], [[10., 0., 0.], [10., 10., 10.]]])
    s2 = np.array([[[0., 0., 3.], [0., 10., 3.]], [[10., 0., 3.], [10., 10., 3.]]])
    S1h, S2h = _homog(s1), _homog(s2)
    G = psi_vector_net(S1h, S2h)
    nets = [BoxNet(G[..., k:k + 1], axes=(0, 1, 2, 3)) for k in range(3)]
    nets.append(BoxNet(linear_net_4d(-0.5, (1.0, 0.0, 0.0, 0.0)), axes=(0, 1, 2, 3)))

    def newton(x0):
        # square Newton on {Psi(3), s - 0.5}
        x = np.asarray(x0, float).copy()
        for _ in range(30):
            p1, du1, dv1 = eval_surface_d1(S1h, x[0], x[1], rational=True)
            p2, du2, dv2 = eval_surface_d1(S2h, x[2], x[3], rational=True)
            F = np.concatenate([p1 - p2, [x[0] - 0.5]])
            J = np.zeros((4, 4))
            J[:3, 0], J[:3, 1], J[:3, 2], J[:3, 3] = du1, dv1, -du2, -dv2
            J[3, 0] = 1.0
            if np.linalg.norm(F) < 1e-12:
                break
            try:
                x = np.clip(x - np.linalg.solve(J, F), 0.0, 1.0)
            except np.linalg.LinAlgError:
                return None
        return x if np.linalg.norm(F) < 1e-9 else None

    sols, exhausted = solve_zero_dim(nets, newton, ptol=np.full(4, 1e-5), max_cells=5000)
    assert not exhausted
    # ground truth via CSX on the isoline
    from mmcore.numeric.bern import de_casteljau_split_nd
    from mmcore.numeric.intersection.csx._bez_csx4 import bez_csx
    left, _ = de_casteljau_split_nd(S1h, axis=0, t=0.5)
    iso = left[-1, :, :]
    ref = bez_csx(iso, S2h, atol=1e-9, rational=True)["isolated"]
    assert len(sols) == len(ref) >= 1
    ref_t = sorted(p["t"] for p in ref)
    got_t = sorted(s[1] for s in sols)
    assert np.allclose(ref_t, got_t, atol=1e-6)


def test_solve_zero_dim_reports_budget_exhaustion():
    # two-root system: S1 has z = (2t-1)^2 (ruled in s), S2 is the plane
    # z = 0.5; sliced with s = 0.5 the roots are t = (1 ± sqrt(0.5))/2.
    # A starved budget must be distinguishable from a complete enumeration
    # (pre-flag, max_cells=3 returned a silently-partial solution list).
    xs = [0.0, 1.0]; ys = [0.0, 0.5, 1.0]; zc = [1.0, -1.0, 1.0]
    s1 = np.array([[[xs[i], ys[j], zc[j]] for j in range(3)] for i in range(2)])
    s2 = np.array([[[-0.5, -0.5, 0.5], [-0.5, 1.5, 0.5]],
                   [[1.5, -0.5, 0.5], [1.5, 1.5, 0.5]]])
    S1h, S2h = _homog(s1), _homog(s2)
    G = psi_vector_net(S1h, S2h)
    nets = [BoxNet(G[..., k:k + 1], axes=(0, 1, 2, 3)) for k in range(3)]
    nets.append(BoxNet(linear_net_4d(-0.5, (1.0, 0.0, 0.0, 0.0)), axes=(0, 1, 2, 3)))

    def newton(x0):
        # square Newton on {Psi(3), s - 0.5}
        x = np.asarray(x0, float).copy()
        for _ in range(30):
            p1, du1, dv1 = eval_surface_d1(S1h, x[0], x[1], rational=True)
            p2, du2, dv2 = eval_surface_d1(S2h, x[2], x[3], rational=True)
            F = np.concatenate([p1 - p2, [x[0] - 0.5]])
            J = np.zeros((4, 4))
            J[:3, 0], J[:3, 1], J[:3, 2], J[:3, 3] = du1, dv1, -du2, -dv2
            J[3, 0] = 1.0
            if np.linalg.norm(F) < 1e-12:
                break
            try:
                x = np.clip(x - np.linalg.solve(J, F), 0.0, 1.0)
            except np.linalg.LinAlgError:
                return None
        return x if np.linalg.norm(F) < 1e-9 else None

    # full budget: complete enumeration, both roots, exhausted=False
    sols, exhausted = solve_zero_dim(nets, newton, ptol=np.full(4, 1e-5), max_cells=5000)
    assert not exhausted
    assert len(sols) == 2
    want_t = sorted([(1.0 - np.sqrt(0.5)) / 2.0, (1.0 + np.sqrt(0.5)) / 2.0])
    assert np.allclose(sorted(s[1] for s in sols), want_t, atol=1e-6)

    # starved budget: pending boxes dropped -> exhausted=True, partial sols
    sols_starved, exhausted_starved = solve_zero_dim(
        nets, newton, ptol=np.full(4, 1e-5), max_cells=3)
    assert exhausted_starved
    assert len(sols_starved) < 2


def test_sigma_net_matches_fd():
    rng = np.random.default_rng(11)
    S = rng.uniform(-1, 1, (4, 3, 3))
    N = sigma_normal_net(_homog(S), rational=False)
    from mmcore.numeric.bern import bernstein_eval_nd
    for st in rng.uniform(0.05, 0.95, (6, 2)):
        _, du, dv = eval_surface_d1(_homog(S), st[0], st[1], rational=True)
        want = np.cross(du, dv)
        got = bernstein_eval_nd(N, st)
        assert np.allclose(got, want, rtol=1e-9, atol=1e-11)


def test_sigma_net_accepts_cartesian_and_rejects_bad_weights():
    # dual input-shape contract: polynomial branch must accept BOTH a bare
    # Cartesian (m,n,3) array (used by Task 6) AND a homogeneous (m,n,4)
    # array with weights == 1 (used above); a homogeneous array with
    # non-unit weights must be rejected rather than silently mistreated.
    rng = np.random.default_rng(17)
    S = rng.uniform(-1, 1, (4, 3, 3))
    N_cart = sigma_normal_net(S, rational=False)
    N_homog = sigma_normal_net(_homog(S), rational=False)
    assert np.allclose(N_cart, N_homog, atol=1e-12)

    S_bad = _homog(S)
    S_bad[0, 0, -1] = 1.5   # break the w == 1 invariant
    with pytest.raises(ValueError):
        sigma_normal_net(S_bad, rational=False)


def test_sigma_net_degree0_raises():
    # degree-0 in either parametric direction: the partial derivative is
    # identically zero, so the normal net is meaningless — must be a loud
    # ValueError at entry, not a raw IndexError deep in _deflate.py.
    # Covers BOTH branches (polynomial and rational).
    rng = np.random.default_rng(5)
    deg0_u = rng.uniform(-1, 1, (1, 3, 3))   # degree 0 along the first direction
    deg0_v = rng.uniform(-1, 1, (3, 1, 3))   # degree 0 along the second direction
    for bad in (deg0_u, deg0_v):
        with pytest.raises(ValueError, match="degree >= 1"):
            sigma_normal_net(_homog(bad), rational=False)
        with pytest.raises(ValueError, match="degree >= 1"):
            sigma_normal_net(_homog(bad), rational=True)


def test_solve_zero_dim_empty_nets_raises():
    # nets=[] would silently degrade to an exhaustive Newton multistart
    # burning the whole budget — must be rejected loudly.
    with pytest.raises(ValueError, match="non-empty"):
        solve_zero_dim([], lambda x0: None, ptol=np.full(4, 1e-3))


def test_boxnet_partial_axes_restriction():
    # Task-6 shape: a Sigma net depends only on (s,t) — a 2-dim BoxNet
    # embedded in the 4D solver frame via axes=(0, 1).
    from mmcore.numeric.bern import de_casteljau_split_nd, bernstein_eval_nd
    rng = np.random.default_rng(21)
    S = rng.uniform(-1, 1, (3, 3, 3))
    N = sigma_normal_net(_homog(S), rational=False)
    bn = BoxNet(N[..., 0:1], axes=(0, 1))            # x-component net
    # (a) restriction along a foreign global axis (u or v) is an identity
    # no-op returning the SAME (frozen, shared) instance for both children
    for foreign in (2, 3):
        l, r = bn.split(foreign, 0.5)
        assert l is bn and r is bn
    with pytest.raises(AttributeError):              # frozen: shared instances stay immutable
        bn.axes = (0, 2)
    # (b) restriction along an own global axis matches de_casteljau_split_nd
    # and evaluates consistently on the half-domains
    l, r = bn.split(1, 0.5)
    L, R = de_casteljau_split_nd(bn.coeffs, axis=1, t=0.5)
    assert l.axes == (0, 1) and r.axes == (0, 1)
    assert np.allclose(l.coeffs, L) and np.allclose(r.coeffs, R)
    for a, b in rng.uniform(0, 1, (5, 2)):
        assert np.allclose(bernstein_eval_nd(l.coeffs, [a, b]),
                           bernstein_eval_nd(bn.coeffs, [a, 0.5 * b]), atol=1e-12)
        assert np.allclose(bernstein_eval_nd(r.coeffs, [a, b]),
                           bernstein_eval_nd(bn.coeffs, [a, 0.5 + 0.5 * b]), atol=1e-12)
    # (c) excludes_zero on the restricted 2-axis net
    assert BoxNet(np.abs(bn.coeffs) + 0.5, axes=(0, 1)).excludes_zero()
    assert BoxNet(-np.abs(bn.coeffs) - 0.5, axes=(0, 1)).excludes_zero()
    mixed = bn.coeffs.copy()
    mixed.flat[0], mixed.flat[-1] = -1.0, 1.0
    assert not BoxNet(mixed, axes=(0, 1)).excludes_zero()


def test_sigma_net_rational_matches_direction():
    # Acceptance gate for the rational branch (genuinely rational surface:
    # random control points + random weights in [0.5, 2.0]). Scale-invariant
    # direction check only (the rational normal numerator is a positive
    # multiple of the true normal, not equal to it) — see module docstring.
    rng = np.random.default_rng(13)
    P = rng.uniform(-1, 1, (3, 4, 3))
    w = rng.uniform(0.5, 2.0, (3, 4))
    S_h = np.concatenate([P * w[..., None], w[..., None]], axis=-1)
    N = sigma_normal_net(S_h, rational=True)
    from mmcore.numeric.bern import bernstein_eval_nd
    for st in rng.uniform(0.05, 0.95, (8, 2)):
        _, du, dv = eval_surface_d1(S_h, st[0], st[1], rational=True)
        want = np.cross(du, dv)
        got = bernstein_eval_nd(N, st)
        gn, wn = np.linalg.norm(got), np.linalg.norm(want)
        assert gn > 1e-8 and wn > 1e-8
        # normalized cross product ~ 0 => parallel; positive dot => same orientation
        assert np.linalg.norm(np.cross(got, want)) < 1e-6 * gn * wn
        assert np.dot(got, want) > 0


# ---------------------------------------------------------------------------
# Task 3: C2 — isolated tangent points (crossing-less tangency gate)
# ---------------------------------------------------------------------------

def _paraboloid_touch():
    """S1: z = (2s-1)^2 + (2t-1)^2 (deg 2x2), touching S2: z=0 plane at (0.5,0.5)."""
    xs = [0.0, 0.5, 1.0]; zc = [1.0, -1.0, 1.0]     # Bernstein coeffs of (2x-1)^2
    S1 = np.array([[[xs[i], xs[j], zc[i] + zc[j]] for j in range(3)] for i in range(3)])
    S2 = np.array([[[-0.5, -0.5, 0.], [-0.5, 1.5, 0.]],
                   [[1.5, -0.5, 0.], [1.5, 1.5, 0.]]])
    return S1, S2


def test_isolated_tangent_point_found():
    S1, S2 = _paraboloid_touch()
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    sing = [g for g in r["singularities"] if g.kind == "tangent_point"]
    assert len(sing) == 1
    g = sing[0]
    assert np.allclose(g.stuv[:2], [0.5, 0.5], atol=1e-4)
    assert np.allclose(g.xyz, [0.5, 0.5, 0.0], atol=1e-3)
    assert r["branches"] == []          # nothing else to trace


def _double_touch_asym():
    """S1: z = 16*((s-0.45)(s-0.9))^2 + (2t-1)^2 (deg 4x2), touching the z=0
    plane at (s,t) = (0.45, 0.5) AND (0.9, 0.5).

    Both touches sit inside the crossing-less TOP cell, and the box-center
    Gauss-Newton start converges into the s=0.45 basin only — a single-start
    witness followed by `continue` silently drops the s=0.9 tangency.
    (The symmetric variant survives by luck: the center start stalls between
    the basins, _check_tangency returns None, and subdivision separates the
    roots before any emission.)
    """
    from math import comb

    def mono_to_bern(a):
        n = len(a) - 1
        return [sum(a[j] * comb(i, j) / comb(n, j) for j in range(i + 1))
                for i in range(n + 1)]

    # q(s) = (s-0.45)(s-0.9) = s^2 - 1.35 s + 0.405 ; z_s(s) = 16 q(s)^2
    qs = np.polynomial.polynomial.polymul(
        [0.405, -1.35, 1.0], [0.405, -1.35, 1.0]) * 16.0
    zb = mono_to_bern(qs)                            # deg 4 Bernstein
    xb = mono_to_bern([0.0, 1.0, 0.0, 0.0, 0.0])     # x(s) = s, deg 4
    zc = [1.0, -1.0, 1.0]                            # (2t-1)^2, deg 2
    yb = [0.0, 0.5, 1.0]                             # y(t) = t, deg 2
    S1 = np.array([[[xb[i], yb[j], zb[i] + zc[j]] for j in range(3)]
                   for i in range(5)])
    S2 = np.array([[[-0.5, -0.5, 0.], [-0.5, 1.5, 0.]],
                   [[1.5, -0.5, 0.], [1.5, 1.5, 0.]]])
    return S1, S2


def test_two_isolated_tangent_points_same_cell():
    S1, S2 = _double_touch_asym()
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    sing = sorted((g for g in r["singularities"] if g.kind == "tangent_point"),
                  key=lambda g: float(g.stuv[0]))
    assert len(sing) == 2
    assert np.allclose(sing[0].stuv[:2], [0.45, 0.5], atol=1e-4)
    assert np.allclose(sing[0].xyz, [0.45, 0.5, 0.0], atol=1e-3)
    assert np.allclose(sing[1].stuv[:2], [0.9, 0.5], atol=1e-4)
    assert np.allclose(sing[1].xyz, [0.9, 0.5, 0.0], atol=1e-3)
    assert r["branches"] == []          # nothing else to trace

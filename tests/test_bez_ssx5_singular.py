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


def test_case11_default_nested_csx_budget_preserves_closed_loop():
    """A sound global budget must not make the historical case 11 partial.

    One internal line/surface cut needs just over 20k CSX cells.  The former
    per-call default stopped at 20k even though the call-wide SSX allowance
    still had more than 200k cells available, discarded the two certified
    cut roots, and returned zero branches flagged incomplete.
    """
    from examples.ssx.bez_ssx5_case11 import S1, S2

    # This is deliberately tight enough that paying for a discarded 20k
    # attempt and then restarting cannot complete, while one topology-critical
    # CSX call with the established allowance finishes the whole SSX solve.
    result = bez_ssx(
        S1, S2, 1e-3, rational=False, max_cells=60_000)

    assert result["complete"], result["status"]
    assert len(result["branches"]) == 1, result
    xyz = np.asarray(result["branches"][0].curve[1], dtype=float)
    assert np.linalg.norm(xyz[0] - xyz[-1]) <= 2e-3
    assert len(xyz) >= 32
    assert np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum() > 1.0


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
    # S2 spans x,y in [-0.5, 1.5] bilinearly => u=(x+0.5)/2, v=(y+0.5)/2;
    # the touch at xyz=(0.5, 0.5, 0) has the S2 preimage (0.5, 0.5).
    assert np.allclose(g.stuv[2:], [0.5, 0.5], atol=1e-4)
    assert np.allclose(g.xyz, [0.5, 0.5, 0.0], atol=1e-3)
    assert r["branches"] == []          # nothing else to trace
    # near-touch grazing seeds are subsumed by the emitted singularity
    assert r["points"] == []


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
    # S2 preimage of the touch: u=(x+0.5)/2, v=(y+0.5)/2 (bilinear span)
    assert np.allclose(sing[0].stuv[2:], [0.475, 0.5], atol=1e-4)
    assert np.allclose(sing[0].xyz, [0.45, 0.5, 0.0], atol=1e-3)
    assert np.allclose(sing[1].stuv[:2], [0.9, 0.5], atol=1e-4)
    assert np.allclose(sing[1].stuv[2:], [0.7, 0.5], atol=1e-4)
    assert np.allclose(sing[1].xyz, [0.9, 0.5, 0.0], atol=1e-3)
    assert r["branches"] == []          # nothing else to trace
    # near-touch grazing seeds are subsumed by the emitted singularities
    assert r["points"] == []


def _mexican_hat():
    """S1: z = q(q-1/2), q=(2s-1)^2+(2t-1)^2 (deg 4x4): tangent point at the
    center PLUS a transversal ring at q=1/2 — both inside one crossing-less
    cell. The tangency emission must not delete the ring."""
    from math import comb

    # z(s,t) = A(s) + A(t) + 2*C(s)*C(t) in monomials, where
    # C(x) = (2x-1)^2 and A(x) = (2x-1)^4 - 0.5*(2x-1)^2.
    C = np.array([1.0, -4.0, 4.0])
    A = np.array([0.5, -6.0, 22.0, -32.0, 16.0])
    M = np.zeros((5, 5))
    M[:, 0] += A
    M[0, :] += A
    M[:3, :3] += 2.0 * np.outer(C, C)
    # per-axis monomial -> Bernstein at degree 4: b_i = sum_j K[i,j] a_j
    K = np.array([[comb(i, j) / comb(4, j) if j <= i else 0.0
                   for j in range(5)] for i in range(5)])
    Z = K @ M @ K.T
    xb = np.array([i / 4.0 for i in range(5)])   # x(s)=s, y(t)=t at deg 4
    S1 = np.array([[[xb[i], xb[j], Z[i, j]] for j in range(5)] for i in range(5)])
    S2 = np.array([[[-0.5, -0.5, 0.], [-0.5, 1.5, 0.]],
                   [[1.5, -0.5, 0.], [1.5, 1.5, 0.]]])
    # self-check the construction: net vs q(q-1/2) at a few samples
    for s, t in [(0.2, 0.7), (0.5, 0.5), (0.85, 0.15), (0.5, 0.146)]:
        q = (2 * s - 1) ** 2 + (2 * t - 1) ** 2
        p = eval_surface(_homog(S1), s, t, rational=True)
        assert np.allclose(p, [s, t, q * (q - 0.5)], atol=1e-12)
    return S1, S2


def test_tangent_point_with_coexisting_transversal_ring():
    # DEFECT-A regression: the crossing-less tangency arm used to `continue`
    # unconditionally after emitting the tangent point, deleting the whole
    # cell — including the genuine transversal ring that coexists with the
    # touch in the same crossing-less cell.
    S1, S2 = _mexican_hat()
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    sing = [g for g in r["singularities"] if g.kind == "tangent_point"]
    assert len(sing) == 1
    assert np.allclose(sing[0].xyz, [0.5, 0.5, 0.0], atol=2e-3)
    closed = [b for b in r["branches"]
              if np.linalg.norm(np.asarray(b.curve[1])[0] - np.asarray(b.curve[1])[-1]) < 5e-3]
    assert len(closed) == 1, f"ring lost or duplicated: {len(closed)} closed branches"
    xyz = np.asarray(closed[0].curve[1])
    # the true ring: q = 1/2 is the circle of radius sqrt(1/2) in
    # (2s-1, 2t-1)-coords -> radius sqrt(1/2)/2 in (s,t); x=s, y=t, z=0,
    # so the xyz ring has radius sqrt(1/2)/2 ~ 0.35355 about (0.5, 0.5, 0).
    rr = np.linalg.norm(xyz[:, :2] - 0.5, axis=1)
    assert np.allclose(rr, np.sqrt(0.5) / 2.0, atol=5e-3)
    # near-touch grazing seeds are subsumed by the emitted singularity
    assert r["points"] == []


# ---------------------------------------------------------------------------
# Task 4: C2 — tangent point WITH transversal branches (saddle X-crossing)
# ---------------------------------------------------------------------------

def _saddle_touch():
    """S1: z = (2s-1)^2 - (2t-1)^2 saddle; S2: z=0 plane.
    SSI = two straight lines s=t and s=1-t crossing at the tangent point (0.5,0.5)."""
    zc = [1.0, -1.0, 1.0]; xs = [0.0, 0.5, 1.0]
    S1 = np.array([[[xs[i], xs[j], zc[i] - zc[j]] for j in range(3)] for i in range(3)])
    S2 = np.array([[[-0.5, -0.5, 0.], [-0.5, 1.5, 0.]],
                   [[1.5, -0.5, 0.], [1.5, 1.5, 0.]]])
    return S1, S2


def _pt_poly(p, poly):
    a, b = poly[:-1], poly[1:]
    ab = b - a
    den = np.einsum("ij,ij->i", ab, ab); den[den < 1e-30] = 1e-30
    tt = np.clip(np.einsum("ij,ij->i", p[None] - a, ab) / den, 0, 1)
    return float(np.linalg.norm(a + tt[:, None] * ab - p[None], axis=1).min())


def test_saddle_tangent_point_with_branches():
    S1, S2 = _saddle_touch()
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    sing = [g for g in r["singularities"] if g.kind == "tangent_point"]
    assert len(sing) == 1
    assert np.allclose(sing[0].xyz, [0.5, 0.5, 0.0], atol=2e-3)
    # both diagonals fully covered (sample the two true lines, X in [0,1]):
    polys = [np.asarray(b.curve[1]) for b in r["branches"]]
    assert polys, "no branches traced"
    for diag in (lambda a: (a, a, 0.0), lambda a: (a, 1.0 - a, 0.0)):
        for a in np.linspace(0.01, 0.99, 33):
            p = np.array(diag(a))
            d = min(_pt_poly(p, poly) for poly in polys)
            assert d < 5e-3, f"diagonal point {p} missed by {d}"
    # the X-crossing CSX near-roots at the touch are marched into the arms,
    # and anything at the touch itself is subsumed by the singularity
    assert r["points"] == []


def test_no_contact_lifted_paraboloid_emits_nothing():
    # Regression guard for the size-gated tangency arm: lift the paraboloid
    # clear of the plane (min gap 0.1, no contact anywhere) — every cell is
    # crossing-less and the F_sq prune must kill the pair early, with no
    # spurious tangent_point emission and no pruning fallout.
    S1, S2 = _paraboloid_touch()
    S1 = S1.copy()
    S1[..., 2] += 0.1
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    assert r["branches"] == []
    assert r["points"] == []
    assert r["singularities"] == []


# ---------------------------------------------------------------------------
# Task 5: C2 — tiny loops near tangency via Φ ∩ L seeding
# ---------------------------------------------------------------------------

def _touch_plus_loop(eps=0.04):
    """S1: z = r^4 - eps*r^2 with r^2=(2s-1)^2+(2t-1)^2  (deg 4x4);
    S2: z=0. SSI: tangent point at r=0 PLUS transversal loop at r=sqrt(eps).
    Paper Fig. 24 (Example 11) analog."""
    from math import comb

    def mono_to_bern(a):
        n = len(a) - 1
        return np.array([sum(a[j] * comb(k, j) / comb(n, j) for j in range(k + 1))
                         for k in range(n + 1)])
    f = np.array([1.0, -4.0, 4.0])                       # (2x-1)^2 monomial
    f2 = np.convolve(f, f)                               # degree 4
    z_st = np.zeros((5, 5))
    z_st[:5, 0] += f2; z_st[0, :5] += f2
    z_st[:3, :3] += 2.0 * np.outer(f, f)
    z_st[:3, 0] -= eps * f; z_st[0, :3] -= eps * f
    M = np.array([mono_to_bern(np.eye(5)[j]) for j in range(5)])   # rows: x^j in deg-4 Bernstein
    Bz = M.T @ z_st @ M
    xs = mono_to_bern([0.0, 1.0, 0.0, 0.0, 0.0])         # x = s in deg-4 Bernstein
    S1 = np.array([[[xs[i], xs[j], Bz[i, j]] for j in range(5)] for i in range(5)])
    S2 = np.array([[[-0.5, -0.5, 0.], [-0.5, 1.5, 0.]],
                   [[1.5, -0.5, 0.], [1.5, 1.5, 0.]]])
    # self-check the construction: net vs r^4 - eps*r^2 at random samples
    rng = np.random.default_rng(42)
    for s, t in rng.uniform(0, 1, (25, 2)):
        r2 = (2 * s - 1) ** 2 + (2 * t - 1) ** 2
        p = eval_surface(_homog(S1), s, t, rational=True)
        assert np.allclose(p, [s, t, r2 * r2 - eps * r2], atol=1e-12)
    return S1, S2


def test_tangent_point_plus_tiny_loop():
    S1, S2 = _touch_plus_loop(eps=0.04)     # loop radius 0.1 in s-units
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    sing = [g for g in r["singularities"] if g.kind == "tangent_point"]
    assert len(sing) == 1 and np.allclose(sing[0].xyz, [0.5, 0.5, 0.0], atol=2e-3)
    loops = [b for b in r["branches"]
             if np.linalg.norm(np.asarray(b.curve[1])[0] - np.asarray(b.curve[1])[-1]) < 5e-3]
    assert len(loops) == 1, f"expected the r=sqrt(eps) loop, got {len(loops)} closed branches"
    xyz = np.asarray(loops[0].curve[1])
    rr = np.linalg.norm(xyz[:, :2] - 0.5, axis=1)
    assert np.allclose(rr, 0.1, atol=5e-3)   # circle of radius sqrt(0.04)/2 in s-units


def test_touch_plus_loop_small_eps_no_phantom_tangential():
    # Ledger L2 regression (~33 s: the phantom candidates are still marched
    # before being rejected — the cost is the deep subdivision plus four
    # Φ∩L seeding rounds around a sub-tolerance feature cluster, not the
    # validation itself). At eps=1e-3 the valley floor (eps²/4 = 2.5e-7)
    # is Ψ-valid at atol and passes the 0.01·atol seed-refinement
    # acceptance; the Φ closed-loop marcher then shipped TWO 375-pt phantom
    # 'tangential' branches (transversal-normal along the arc, sin_ang up
    # to 2.6e-2, geometry up to 58·atol off the true SSI) — and one passed
    # through the touch, so the post-assembly subsumption filter deleted
    # the genuine tangent_point. Ground truth: exactly one touch at
    # (0.5,0.5,0) plus one transversal ring of radius sqrt(eps)/2 = 0.0158.
    S1, S2 = _touch_plus_loop(eps=1e-3)
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    tps = [g for g in r["singularities"] if g.kind == "tangent_point"]
    assert len(tps) == 1, f"expected exactly 1 tangent_point, got {len(tps)}"
    assert np.allclose(tps[0].xyz, [0.5, 0.5, 0.0], atol=2e-3)
    assert [b for b in r["branches"] if b.kind == "tangential"] == [], \
        "phantom 'tangential' branch shipped from the sub-tolerance valley"
    want_r = np.sqrt(1e-3) / 2.0                        # 0.01581
    closed = [b for b in r["branches"]
              if np.linalg.norm(np.asarray(b.curve[1])[0]
                                - np.asarray(b.curve[1])[-1]) < 5e-3]
    ring_like = []
    for b in closed:
        rr = np.linalg.norm(np.asarray(b.curve[1])[:, :2] - 0.5, axis=1)
        if np.allclose(rr, want_r, atol=5e-3):
            ring_like.append(b)
    assert ring_like, (
        f"transversal ring r~{want_r:.4f} lost; closed branches: "
        f"{[np.linalg.norm(np.asarray(b.curve[1])[:, :2] - 0.5, axis=1).mean() for b in closed]}")


@pytest.mark.parametrize("eps", [5e-3, 8e-3])
def test_touch_plus_loop_controls_stay_clean(eps):
    # L2 controls: these eps values were clean BEFORE the fix and must stay
    # clean after it (the tangency validation must not reject genuine
    # geometry or resurrect phantoms at coarser feature scales).
    S1, S2 = _touch_plus_loop(eps=eps)
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    tps = [g for g in r["singularities"] if g.kind == "tangent_point"]
    assert len(tps) == 1 and np.allclose(tps[0].xyz, [0.5, 0.5, 0.0], atol=2e-3)
    assert [b for b in r["branches"] if b.kind == "tangential"] == []
    want_r = np.sqrt(eps) / 2.0
    closed = [b for b in r["branches"]
              if np.linalg.norm(np.asarray(b.curve[1])[0]
                                - np.asarray(b.curve[1])[-1]) < 5e-3]
    assert any(np.allclose(
        np.linalg.norm(np.asarray(b.curve[1])[:, :2] - 0.5, axis=1),
        want_r, atol=5e-3) for b in closed)


# ---------------------------------------------------------------------------
# Follow-up: coexisting features on the crossing-BEARING tangency arm
# ---------------------------------------------------------------------------

def _line_plus_touch(cy=0.2):
    """S1: z = (2t-1)^2 * ((s-0.7)^2 + (t-cy)^2) (deg 2x4), z >= 0:
    tangent LINE along t=0.5 PLUS a coexisting isolated touch at (0.7, cy).
    S2: z=0 plane. The line's cell is crossing-BEARING (the line pierces the
    domain boundary), so the crossing-less arm never sees it — the touch is
    off every traced Phi-fragment and must be found by the off-curve
    enumeration (_emit_offcurve_tangent_roots), not by subdivision.
    `cy` close to 0.5 puts the touch at xyz distance |0.5 - cy| from the
    tangent line (the blind-band regression family)."""
    from math import comb

    A = np.array([1.0, -4.0, 4.0])                      # (2t-1)^2 in t
    Mb = np.zeros((3, 3))                               # (s-0.7)^2 + (t-cy)^2
    Mb[2, 0] = 1.0; Mb[1, 0] = -1.4; Mb[0, 0] = 0.49 + cy * cy
    Mb[0, 1] = -2.0 * cy; Mb[0, 2] = 1.0
    Mz = np.zeros((3, 5))                               # s-deg 2, t-deg 4
    for si in range(3):
        for tj in range(3):
            for k in range(3):
                Mz[si, tj + k] += Mb[si, tj] * A[k]

    def mono_to_bern(a, n):
        return np.array([sum(a[j] * comb(i, j) / comb(n, j)
                             for j in range(min(i, len(a) - 1) + 1))
                         for i in range(n + 1)])

    Ks = np.array([mono_to_bern(np.eye(3)[j], 2) for j in range(3)])
    Kt = np.array([mono_to_bern(np.eye(5)[j], 4) for j in range(5)])
    Bz = Ks.T @ Mz @ Kt
    xs = np.array([0.0, 0.5, 1.0])
    yt = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    S1 = np.array([[[xs[i], yt[j], Bz[i, j]] for j in range(5)] for i in range(3)])
    S2 = np.array([[[-0.5, -0.5, 0.], [-0.5, 1.5, 0.]],
                   [[1.5, -0.5, 0.], [1.5, 1.5, 0.]]])
    for s, t in [(0.3, 0.8), (0.7, cy), (0.5, 0.5), (0.9, 0.35)]:
        want = (2 * t - 1) ** 2 * ((s - 0.7) ** 2 + (t - cy) ** 2)
        p = eval_surface(_homog(S1), s, t, rational=True)
        assert np.allclose(p, [s, t, want], atol=1e-12)
    return S1, S2


def test_tangent_line_with_coexisting_isolated_touch():
    # DEFECT-D regression: the crossing-bearing tangency arm `continue`d
    # after Phi-tracing the line, deleting the coexisting isolated touch at
    # (0.7, 0.2) — the center witness converges into the CURVE's basin, and
    # this cell is the only holder (no descendants ever see the touch).
    S1, S2 = _line_plus_touch()
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    tps = [g for g in r["singularities"] if g.kind == "tangent_point"]
    assert len(tps) == 1, f"expected only the isolated touch, got {len(tps)}"
    assert np.allclose(tps[0].xyz, [0.7, 0.2, 0.0], atol=2e-3)
    assert np.allclose(tps[0].stuv[:2], [0.7, 0.2], atol=1e-3)
    # the tangent line itself is traced as one tangential branch: (s, 0.5, 0)
    tang = [b for b in r["branches"] if b.kind == "tangential"]
    assert len(tang) == 1
    xyz = np.asarray(tang[0].curve[1])
    assert np.allclose(xyz[:, 1], 0.5, atol=2e-3) and np.allclose(xyz[:, 2], 0.0, atol=2e-3)
    assert xyz[:, 0].min() < 0.02 and xyz[:, 0].max() > 0.98   # full span
    assert r["points"] == []


@pytest.mark.parametrize("delta", [0.01, 0.005])
def test_offcurve_touch_near_tangent_line_blind_band(delta):
    # Budget-starvation blind-band regression (review of 2d030bb+7ed47c0):
    # with the original per-pop max_cells charging in solve_zero_dim, the
    # on-curve Delta-flood's hull-excluded siblings starved the budget and
    # an isolated touch at 5-15*atol xyz from a coexisting tangent LINE was
    # silently lost (found at 20*atol, lost at 15/12.5/10/5*atol). The
    # skip-aware budget (Newton attempts charge; flood traversal is free,
    # bounded by max_cells + 16*charged) closes the band down to 5*atol —
    # 4*atol and below is legitimately subsumed by the post-assembly
    # tangent-point filter.
    cy = 0.5 - delta                     # touch at (s, t) = (0.7, cy)
    S1, S2 = _line_plus_touch(cy)
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    tang = [b for b in r["branches"] if b.kind == "tangential"]
    assert len(tang) == 1, f"expected the tangent line branch, got {len(tang)}"
    tps = [g for g in r["singularities"] if g.kind == "tangent_point"]
    want = np.array([0.7, cy, 0.0])
    hits = [g for g in tps
            if np.linalg.norm(np.asarray(g.xyz) - want) <= 2e-3]
    assert hits, (f"off-curve touch at {delta / 1e-3:.0f}*atol from the "
                  f"tangent line lost (blind band): {[g.xyz for g in tps]}")


def _closed_tangent_loop():
    """S1: z = (q - 1/4)^2 with q = (2s-1)^2 + (2t-1)^2 (deg 4x4); S2: z=0.
    z >= 0 with equality exactly on the circle q = 1/4 — a CLOSED TANGENT
    LOOP (radius 0.5 in (2s-1)-units = 0.25 in s-units), rank-deficient
    Psi-Jacobian everywhere on it, and NO isolated touch (z = 1/16 at the
    center). No boundary crossings anywhere: the crossing-less arm's
    Phi ∩ L seeding must find it and the Phi closed-loop marcher
    (_march_phi_closed backend, sin_ang <= 1e-3) must trace it."""
    from math import comb

    def mono_to_bern(a):
        return np.array([sum(a[j] * comb(k, j) / comb(4, j)
                             for j in range(k + 1)) for k in range(5)])

    f = np.array([1.0, -4.0, 4.0])                       # (2x-1)^2 monomial
    f2 = np.convolve(f, f)                               # (2x-1)^4
    z_st = np.zeros((5, 5))                              # q^2 - q/2 + 1/16
    z_st[:5, 0] += f2; z_st[0, :5] += f2
    z_st[:3, :3] += 2.0 * np.outer(f, f)
    z_st[:3, 0] -= 0.5 * f; z_st[0, :3] -= 0.5 * f
    z_st[0, 0] += 0.0625
    M = np.array([mono_to_bern(np.eye(5)[j]) for j in range(5)])
    Bz = M.T @ z_st @ M
    xs = mono_to_bern([0.0, 1.0, 0.0, 0.0, 0.0])         # x = s in deg-4
    S1 = np.array([[[xs[i], xs[j], Bz[i, j]] for j in range(5)] for i in range(5)])
    S2 = np.array([[[-0.5, -0.5, 0.], [-0.5, 1.5, 0.]],
                   [[1.5, -0.5, 0.], [1.5, 1.5, 0.]]])
    rng = np.random.default_rng(7)
    for s, t in rng.uniform(0, 1, (20, 2)):
        q = (2 * s - 1) ** 2 + (2 * t - 1) ** 2
        p = eval_surface(_homog(S1), s, t, rational=True)
        assert np.allclose(p, [s, t, (q - 0.25) ** 2], atol=1e-12)
    return S1, S2


def test_closed_tangent_loop_via_phi_marcher():
    # Pipeline coverage for the Phi closed-loop backend (_march_phi_closed):
    # the touch-plus-loop case exercises only the Psi backend (its loop is
    # transversal); here the loop itself is a tangent curve, so the seeds
    # refine to sin_ang ~ 0 and the Phi marcher must close it.
    S1, S2 = _closed_tangent_loop()
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    tang = [b for b in r["branches"] if b.kind == "tangential"]
    assert len(tang) == 1, f"expected 1 tangential loop, got {len(tang)}"
    assert len(r["branches"]) == 1
    xyz = np.asarray(tang[0].curve[1])
    endgap = float(np.linalg.norm(xyz[0] - xyz[-1]))
    assert endgap < 1e-9, f"loop not closed: endgap {endgap}"
    rr = np.linalg.norm(xyz[:, :2] - 0.5, axis=1)
    assert np.allclose(rr, 0.25, atol=5e-3)   # circle q=1/4 -> radius 0.25 in s-units


def _skew_ruled_touch_near_overlap(c=0.003 / 0.8):
    """S1(u,v) deg (3,2): x=v, y=c*u, z=10*u*((u-0.8)^2+(v-0.5)^2); S2: z=0
    plane spanning [-0.5,1.5]^2. The u=0 isoline is an OVERLAP branch
    (x=v, y=0, z=0); a genuine isolated TOUCH sits at (u,v)=(0.8,0.5),
    xyz=(0.5, 0.8c, 0). For c=0.003/0.8 the touch is 3*atol from the
    overlap polyline in xyz but du=0.8 away in parameters, with a 640*atol
    z-wall between the sheets (z=0.64 at u=0.4, v=0.5)."""
    from math import comb

    # z monomial-in-u coefficients: 10*(u(u-0.8)^2 (x) 1 + u (x) (v-0.5)^2)
    M = np.zeros((4, 3))
    M[:, 0] += 10.0 * np.array([0.0, 0.64, -1.6, 1.0])   # 10*u(u-0.8)^2
    M[1, 0] += 2.5                                       # 10*u*0.25
    M[1, 1] += -10.0                                     # 10*u*(-v)
    M[1, 2] += 10.0                                      # 10*u*v^2
    K3 = np.array([[comb(i, j) / comb(3, j) if j <= i else 0.0
                    for j in range(4)] for i in range(4)])
    K2 = np.array([[comb(i, j) / comb(2, j) if j <= i else 0.0
                    for j in range(3)] for i in range(3)])
    Bz = K3 @ M @ K2.T
    xu = np.array([0.0, 0.5, 1.0])                       # x = v (deg 2)
    yu = c * np.array([0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0])  # y = c*u (deg 3)
    S1 = np.array([[[xu[j], yu[i], Bz[i, j]] for j in range(3)]
                   for i in range(4)])
    S2 = np.array([[[-0.5, -0.5, 0.], [-0.5, 1.5, 0.]],
                   [[1.5, -0.5, 0.], [1.5, 1.5, 0.]]])
    rng = np.random.default_rng(9)
    for u, v in rng.uniform(0, 1, (20, 2)):
        want = [v, c * u, 10.0 * u * ((u - 0.8) ** 2 + (v - 0.5) ** 2)]
        p = eval_surface(_homog(S1), u, v, rational=True)
        assert np.allclose(p, want, atol=1e-12)
    return S1, S2


def test_param_far_touch_near_overlap_not_subsumed():
    # Ledger L3 regression: the post-assembly tangent_point subsumption
    # filter was xyz-only (4*atol, no parametric guard) and deleted this
    # certified touch — 3*atol from the u=0 overlap polyline in xyz but on
    # a DIFFERENT sheet (du=0.8, 640*atol z-wall between). Both guards
    # (xyz <= 4*atol AND per-axis stuv <= 2*unify_tol at the same segment
    # location) keep it. Control c=0.05/0.8 (touch 50*atol away) reported
    # the touch even before the fix.
    S1, S2 = _skew_ruled_touch_near_overlap()
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    # the subsumption-tempting configuration is really present: an overlap
    # branch passing within 4*atol (xyz) of the touch
    overlaps = [b for b in r["branches"] if b.kind == "overlap"]
    assert overlaps, "u=0 overlap branch lost — fixture no longer tests L3"
    assert len(overlaps) == 1
    assert len(r["branches"]) == 1, (
        "a regular tracer duplicated a subset of the certified overlap: "
        f"{[b.kind for b in r['branches']]}")
    touch_xyz = np.array([0.5, 0.003, 0.0])
    assert min(_pt_poly(touch_xyz, np.asarray(b.curve[1]))
               for b in overlaps) <= 4e-3
    tps = [g for g in r["singularities"] if g.kind == "tangent_point"]
    assert len(tps) == 1, (
        f"param-far touch subsumed by the overlap branch: {len(tps)} "
        f"tangent_points, kinds={[g.kind for g in r['singularities']]}")
    assert np.allclose(tps[0].xyz, touch_xyz, atol=2e-3)
    assert np.allclose(tps[0].stuv[:2], [0.8, 0.5], atol=1e-3)


# ---------------------------------------------------------------------------
# Task 6: C1 — parameterization cusps (Sigma nets, global pass)
# ---------------------------------------------------------------------------

def _cusp_edge_case():
    """S1(s,t) = ((2s-1)^2, (2s-1)^3, t): cuspidal edge along s=0.5 (deg 3x1).
    S2: plane z=0.5 spanning x in [-0.5,1.5], y in [-1.5,1.5].
    SSI: the classic cusp curve (a^2, a^3, 0.5) — C1 cusp point at
    stuv=(0.5, 0.5, ., .), xyz=(0,0,0.5). Paper Fig. 18 (Example 5) analog."""
    x3 = [1.0, -1.0 / 3.0, -1.0 / 3.0, 1.0]      # (2s-1)^2 in deg-3 Bernstein
    y3 = [-1.0, 1.0, -1.0, 1.0]                  # (2s-1)^3 in deg-3 Bernstein
    S1 = np.array([[[x3[i], y3[i], float(j)] for j in range(2)] for i in range(4)])
    S2 = np.array([[[-0.5, -1.5, 0.5], [-0.5, 1.5, 0.5]],
                   [[1.5, -1.5, 0.5], [1.5, 1.5, 0.5]]])
    # self-check the construction
    for s, t in [(0.1, 0.3), (0.5, 0.5), (0.85, 0.9)]:
        a = 2 * s - 1
        p = eval_surface(_homog(S1), s, t, rational=True)
        assert np.allclose(p, [a * a, a ** 3, t], atol=1e-12)
    return S1, S2


def test_cusp_point_on_branch():
    S1, S2 = _cusp_edge_case()
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    cusps = [g for g in r["singularities"] if g.kind == "cusp"]
    assert len(cusps) == 1
    g = cusps[0]
    assert abs(g.stuv[0] - 0.5) < 1e-4 and abs(g.stuv[1] - 0.5) < 1e-3
    assert np.allclose(g.xyz, [0.0, 0.0, 0.5], atol=1e-3)
    assert g.branch_links, "cusp not linked to its branch"
    # the branch itself must cover the cusp curve including near the cusp
    polys = [np.asarray(b.curve[1]) for b in r["branches"]]
    for a in np.linspace(-0.95, 0.95, 41):
        p = np.array([a * a, a ** 3, 0.5])
        assert min(_pt_poly(p, poly) for poly in polys) < 5e-3


# ---------------------------------------------------------------------------
# Task 7: C3 — self-intersections (Theorem 3 + 6-var Newton post-pass)
# ---------------------------------------------------------------------------

def _umbrella_case():
    """S1: Whitney-umbrella style (a*b, a, b^2), a=2s-1, b=2t-1 (deg 1x2);
    S2: plane z=0.5. SSI: x = a*b with b=+-sqrt(0.5) — two straight lines
    through (0,0,0.5), crossing there with DIFFERENT (s,t) preimages
    (t=(1+-sqrt(0.5))/2): a C3 self-intersection of the SSI image.
    Paper Fig. 22 (Example 9) analog."""
    a = [-1.0, 1.0]; bb = [-1.0, 0.0, 1.0]; bsq = [1.0, -1.0, 1.0]
    S1 = np.array([[[a[i] * bb[j], a[i], bsq[j]] for j in range(3)] for i in range(2)])
    S2 = np.array([[[-1.5, -1.5, 0.5], [-1.5, 1.5, 0.5]],
                   [[1.5, -1.5, 0.5], [1.5, 1.5, 0.5]]])
    # self-check the construction
    for s, t in [(0.2, 0.7), (0.5, 0.5), (0.9, 0.1)]:
        av = 2 * s - 1; bv = 2 * t - 1
        p = eval_surface(_homog(S1), s, t, rational=True)
        assert np.allclose(p, [av * bv, av, bv * bv], atol=1e-12)
    return S1, S2


def _assert_links_nearest_vertex(g, branches):
    # Ledger L11 contract: every branch_link carries the VERTEX index
    # nearest g.xyz on that branch's polyline (4*atol is unachievable at
    # default chord density — umbrella chords ~0.22, so the nearest vertex
    # legitimately sits up to ~half a chord away; what the contract
    # guarantees is OPTIMALITY: no other vertex on that branch is closer).
    assert g.branch_links, "expected at least one branch link"
    for bi, vi in g.branch_links:
        poly = np.asarray(branches[bi].curve[1], dtype=np.float64)
        assert 0 <= vi < len(poly)
        d = np.linalg.norm(poly - np.asarray(g.xyz)[None, :], axis=1)
        assert float(d[vi]) <= float(d.min()) + 1e-12, (
            f"link (b{bi},v{vi}) is {d[vi]:.6f} from xyz but vertex "
            f"{int(d.argmin())} is closer ({d.min():.6f})")


def test_self_intersection_point():
    S1, S2 = _umbrella_case()
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    assert r["complete"] is True
    assert r["status"]["work"]["cell_counts"].get("c3", 0) > 0
    c3 = [g for g in r["singularities"] if g.kind == "self_intersection"]
    assert len(c3) == 1
    g = c3[0]
    assert np.allclose(g.xyz, [0.0, 0.0, 0.5], atol=2e-3)
    assert g.stuv_mate is not None
    # the two preimages differ in (s,t) but share xyz
    assert abs(g.stuv[1] - g.stuv_mate[1]) > 0.2      # t = (1±0.707)/2 differ by ~0.707
    assert len({l[0] for l in g.branch_links}) >= 1   # linked to branch(es)
    _assert_links_nearest_vertex(g, r["branches"])


def test_self_intersection_point_s2_side():
    # Ledger L7: the 6-var system {R1(s,t)=R2(u,v), R1(p,q)=R2(u,v)} with
    # the (s,t)!=(p,q) guard only certifies S1-side doubles — swapping the
    # umbrella to the S2 role made the SAME self-intersection structurally
    # invisible (the two preimages differ on S2, the plane preimage is
    # unique, so the guard rejected every solution). The symmetric system
    # {R2=R1, R2'=R1} with the guard on the (u,v) pair catches it.
    S2umb, S1plane = _umbrella_case()          # umbrella now plays S2
    r = bez_ssx(S1plane, S2umb, 1e-3, rational=False)
    c3 = [g for g in r["singularities"] if g.kind == "self_intersection"]
    assert len(c3) == 1
    g = c3[0]
    assert np.allclose(g.xyz, [0.0, 0.0, 0.5], atol=2e-3)
    # stuv convention for an S2-side double: primary (s,t,u,v); the mate
    # carries the SECOND S2 preimage — stuv_mate = (s,t,u',v').
    assert np.all(np.abs(g.stuv[:2] - g.stuv_mate[:2]) <= 1e-9)
    assert abs(g.stuv[3] - g.stuv_mate[3]) > 0.2       # v = (1±0.707)/2
    assert len({l[0] for l in g.branch_links}) >= 1
    _assert_links_nearest_vertex(g, r["branches"])


def _cusp_edge_on_split_plane():
    """S1(s,t) = ((2s-1)^2, (2s-1)^3, t) (deg 3x1) vs S2: plane x=0 spanning
    y in [-1.5,1.5], z in [-0.5,1.5]. Unlike `_cusp_edge_case` (plane z=0.5,
    ONE isolated C1 point), here the whole set {Psi=0} n {Sigma1=0} is the
    CURVE {s=0.5, t free}: x = (2s-1)^2 vanishes exactly where Sigma1 does,
    and S1(0.5, t) = (0, 0, t) lies in the plane for every t."""
    x3 = [1.0, -1.0 / 3.0, -1.0 / 3.0, 1.0]      # (2s-1)^2 in deg-3 Bernstein
    y3 = [-1.0, 1.0, -1.0, 1.0]                  # (2s-1)^3 in deg-3 Bernstein
    S1 = np.array([[[x3[i], y3[i], float(j)] for j in range(2)] for i in range(4)])
    S2 = np.array([[[0.0, -1.5, -0.5], [0.0, -1.5, 1.5]],
                   [[0.0, 1.5, -0.5], [0.0, 1.5, 1.5]]])
    for s, t in [(0.1, 0.3), (0.5, 0.5), (0.85, 0.9), (0.5, 0.0), (0.5, 1.0)]:
        a = 2 * s - 1
        p = eval_surface(_homog(S1), s, t, rational=True)
        assert np.allclose(p, [a * a, a ** 3, t], atol=1e-12)
    return S1, S2


def test_cusp_curve_on_split_plane_not_knifed_out():
    # Ledger L1 regression: the strict `min > 0` Bernstein hull test excluded
    # BOTH children after solve_zero_dim's first split through s=0.5 — the
    # mathematically-zero coefficients drift to ~eps/8 under de Casteljau,
    # collapsing the 1-dimensional cusp CURVE to ONE isolated `cusp` with a
    # false-complete enumeration. With the roundoff margin the enumeration
    # floods (>12 sols) and the curve_flag types it as `cusp_curve`.
    S1, S2 = _cusp_edge_on_split_plane()
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    curves = [g for g in r["singularities"] if g.kind == "cusp_curve"]
    assert curves, (
        f"cusp curve knifed out: kinds={[g.kind for g in r['singularities']]}")
    assert [g for g in r["singularities"] if g.kind == "cusp"] == [], \
        "cusp curve mistyped as isolated cusp(s)"
    samples = np.concatenate([np.asarray(g.samples) for g in curves])
    assert len(samples) >= 13
    # every sample on the true singular curve {s = 0.5}, covering most of t
    assert np.allclose(samples[:, 0], 0.5, atol=1e-6)
    assert samples[:, 1].max() - samples[:, 1].min() > 0.8


def test_theorem3_skips_regular_case():
    # transversal bilinear pair. Since ledger L8 there is no per-cell
    # Theorem-3 gate at all — c3_pass runs whenever a collision is even
    # possible (here: one branch with plenty of segments). The guarantee
    # this test pins is the one that matters: the vectorized AABB
    # broadphase makes that unconditional run nearly free and it finds
    # nothing — zero spurious self-intersections on regular geometry.
    s1 = np.array([[[0., 0., 0.], [0., 10., 0.]], [[10., 0., 0.], [10., 10., 10.]]])
    s2 = np.array([[[0., 0., 3.], [0., 10., 3.]], [[10., 0., 3.], [10., 10., 3.]]])
    r = bez_ssx(s1, s2, 1e-3, rational=False)
    assert [g for g in r["singularities"] if g.kind == "self_intersection"] == []


def _figure_eight_wall_case():
    """S1 deg (3,2): P[i][j] = (X[i], Y[j], Z[i]+Q[j]) — the family
    S1(s,t) = (p^3 - p, 1.2 t, 0.35 p^2 + 0.3 + 0.16 (t-0.5)^2), p = 1.2(2s-1):
    a figure-eight wall whose 3D surface self-intersects along the line
    {x=0, z = 0.65 + 0.16(t-0.5)^2} (p = +-1, i.e. s = 0.5 +- 1/2.4).
    S2: plane z = 0.66 cuts that line at t = 0.25 and t = 0.75, so the SSI
    (two branches at s ~ 0.08 and s ~ 0.92) crosses itself in 3D at
    (0, 0.3, 0.66) and (0, 0.9, 0.66) — with the two preimages of each
    crossing in DIFFERENT traced cells (first guided cuts split at
    s ~ 0.102/0.898; measured: every branch-carrying traced cell certifies
    per-cell Theorem 3, so no per-cell gate can see this C3 — ledger L8)."""
    X = [-0.528, 2.128, -2.128, 0.528]
    Z = [0.804, 0.132, 0.132, 0.804]
    Y = [0.0, 0.6, 1.2]
    Q = [0.04, -0.04, 0.04]
    S1 = np.array([[[X[i], Y[j], Z[i] + Q[j]] for j in range(3)] for i in range(4)])
    S2 = np.array([[[-1.5, -0.5, 0.66], [-1.5, 1.7, 0.66]],
                   [[1.5, -0.5, 0.66], [1.5, 1.7, 0.66]]])
    # self-check the family and the wall's 3D double points
    for s, t in [(0.1, 0.2), (0.5, 0.5), (0.85, 0.9)]:
        p = 1.2 * (2 * s - 1)
        ref = [p ** 3 - p, 1.2 * t, 0.35 * p * p + 0.3 + 0.16 * (t - 0.5) ** 2]
        assert np.allclose(eval_surface(_homog(S1), s, t, rational=True), ref, atol=1e-12)
    for t in (0.25, 0.75):
        p1 = eval_surface(_homog(S1), 0.5 - 1 / 2.4, t, rational=True)
        p2 = eval_surface(_homog(S1), 0.5 + 1 / 2.4, t, rational=True)
        assert np.linalg.norm(p1 - p2) < 1e-12 and abs(p1[2] - 0.66) < 1e-12
    return S1, S2


def test_cross_cell_self_intersections_figure_eight():
    # Ledger L8: both preimages S1-side, in separate traced cells that each
    # certify per-cell injectivity — the removed c3_possible gate stayed
    # silent on exactly this class (any firing was subdivision-path luck
    # via empty corner-crossing cells). c3_pass must now run structurally
    # and report BOTH crossings (they are 600*atol apart in xyz — the
    # ledger-L16 dedup must keep them distinct).
    S1, S2 = _figure_eight_wall_case()
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    c3 = [g for g in r["singularities"] if g.kind == "self_intersection"]
    assert len(c3) == 2, f"expected both crossings, got {len(c3)}"
    got = sorted([tuple(np.round(g.xyz, 3)) for g in c3])
    assert np.allclose(got, [(0.0, 0.3, 0.66), (0.0, 0.9, 0.66)], atol=2e-3)
    for g in c3:
        # both preimages on S1 (s = 0.5 -+ 1/2.4), shared plane preimage
        assert abs(g.stuv[0] - g.stuv_mate[0]) > 0.5
        assert np.all(np.abs(g.stuv[2:] - g.stuv_mate[2:]) <= 1e-9)
        # both SSI branches pass through each crossing
        assert len({l[0] for l in g.branch_links}) == 2
        _assert_links_nearest_vertex(g, r["branches"])


def test_c3_dedup_both_guards():
    # Ledger L16: the old dedup ball — norm(z1 - z2) <= 4*max(ptol4) with
    # NO xyz guard — merged distinct C3 points hundreds of atol apart in
    # xyz whenever one parametric axis is slow (large ptol). The binding
    # both-guards ladder: same hit only if xyz <= 2*atol AND the unordered
    # preimage pairs match per-axis within 4*ptol.
    from mmcore.numeric.intersection.ssx._ssx5_singular import _c3_same_hit

    atol = 1e-3
    ptol4 = np.array([0.3, 0.3, 1e-4, 1e-4])           # s,t slow; u,v fast
    a = (np.array([0.10, 0.5, 0.5, 0.5]), np.array([0.90, 0.5, 0.5, 0.5]),
         np.array([0.0, 0.0, 0.0]))
    # param-close per-axis (|ds| = 0.25 <= 4*0.3) — the old 6D ball
    # (norm ~0.35 <= 4*max(ptol4) = 1.2) MERGED this pair — but 500*atol
    # apart in xyz: must be DISTINCT.
    b = (np.array([0.35, 0.5, 0.5, 0.5]), np.array([0.65, 0.5, 0.5, 0.5]),
         np.array([0.5, 0.0, 0.0]))
    assert not _c3_same_hit(*a, *b, atol, ptol4)
    # xyz-coincident but preimage pairs differing beyond 4*ptol on a fast
    # axis: distinct (one 3D point, genuinely different preimage pairs).
    c = (a[0].copy(), np.array([0.90, 0.5, 0.9, 0.9]), a[2].copy())
    assert not _c3_same_hit(*a, *c, atol, ptol4)
    # same feature, tiny param/xyz noise: merged.
    d = (a[0] + 1e-5, a[1] - 1e-5, a[2] + np.array([5e-4, 0.0, 0.0]))
    assert _c3_same_hit(*a, *d, atol, ptol4)
    # same feature with primary/mate SWAPPED (the two role-assignment runs
    # can present it either way): merged.
    e = (a[1].copy(), a[0].copy(), a[2].copy())
    assert _c3_same_hit(*a, *e, atol, ptol4)


# ---------------------------------------------------------------------------
# Ledger L4 / L5 / L6: loop-free-arm probe descent, measured tangential
# tagging, 1-dim Δ-sets
# ---------------------------------------------------------------------------

def _touch_plus_loop_offlattice(cx=0.3, eps=0.04):
    """Off-lattice variant of `_touch_plus_loop`: S1: z = r^4 - eps*r^2 with
    r^2 = (2(s-cx))^2 + (2(t-cx))^2 (deg 4x4); S2: z=0. Touch at (cx, cx)
    plus a transversal ring of radius sqrt(eps)/2 in s-units. With cx off
    the dyadic lattice the guided cuts pin the touch to box CORNERS of the
    loop-free cells, where the box-center Gauss-Newton witness dies in the
    touch/valley trap."""
    from math import comb

    def mono_to_bern(a):
        return np.array([sum(a[j] * comb(k, j) / comb(4, j)
                             for j in range(min(k, len(a) - 1) + 1))
                         for k in range(5)])

    f = np.array([4.0 * cx * cx, -8.0 * cx, 4.0])        # (2(x-cx))^2 monomial
    f2 = np.convolve(f, f)
    z_st = np.zeros((5, 5))
    z_st[:5, 0] += f2
    z_st[0, :5] += f2
    z_st[:3, :3] += 2.0 * np.outer(f, f)
    z_st[:3, 0] -= eps * f
    z_st[0, :3] -= eps * f
    M = np.array([mono_to_bern(np.eye(5)[j]) for j in range(5)])
    Bz = M.T @ z_st @ M
    xs = mono_to_bern([0.0, 1.0])
    S1 = np.array([[[xs[i], xs[j], Bz[i, j]] for j in range(5)] for i in range(5)])
    S2 = np.array([[[-0.5, -0.5, 0.], [-0.5, 1.5, 0.]],
                   [[1.5, -0.5, 0.], [1.5, 1.5, 0.]]])
    rng = np.random.default_rng(42)
    for s, t in rng.uniform(0, 1, (20, 2)):
        r2 = (2 * (s - cx)) ** 2 + (2 * (t - cx)) ** 2
        p = eval_surface(_homog(S1), s, t, rational=True)
        assert np.allclose(p, [s, t, r2 * r2 - eps * r2], atol=1e-12)
    return S1, S2


def test_offlattice_touch_plus_loop_found():
    # Ledger L4 regression: the loop-free arm's hull gate fired on the
    # touch-holding cells but the center-only witness failed (the (0.3,0.3)
    # touch sits at a box corner — guided cuts pass through the ring
    # crossings at s/t = 0.3 — and the center GN dies in the touch/valley
    # trap), and the arm `continue`d without any fallback: ALL touches were
    # lost (singularities == []). The probe descent (lean net-split
    # probe_only cells) now hunts the missed root; also pins the
    # `_aabb_disjoint` roundoff margin (the same L1 drift pattern pruned
    # the depth-8 touch-holding probe cells: patch z-hull max drifted to
    # -3e-17 against the plane's exact z = [0, 0]).
    S1, S2 = _touch_plus_loop_offlattice()
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    tps = [g for g in r["singularities"] if g.kind == "tangent_point"]
    hits = [g for g in tps
            if np.linalg.norm(np.asarray(g.xyz) - [0.3, 0.3, 0.0]) <= 2e-3]
    assert hits, (f"off-lattice touch lost (L4): tangent_points at "
                  f"{[np.round(np.asarray(g.xyz), 4).tolist() for g in tps]}")
    # The transversal ring r=0.1 about (0.3, 0.3): every traced branch lies
    # ON the true circle (no phantom geometry) ...
    polys = [np.asarray(b.curve[1]) for b in r["branches"]
             if len(b.curve[1]) >= 2]
    assert polys, "ring completely lost"
    for poly in polys:
        rr = np.linalg.norm(poly[:, :2] - 0.3, axis=1)
        assert np.allclose(rr, 0.1, atol=5e-3), "branch off the true ring"
        assert np.allclose(poly[:, 2], 0.0, atol=2e-3)
    # ... and covers most of it. NOT asserted: full 24/24 coverage or a
    # single closed branch — the off-lattice ring assembles as ~6 fragments
    # with arc gaps up to ~19*atol (measured IDENTICAL before/after the L4
    # fix; the probe descent never traces, so traced geometry is
    # bit-identical to baseline). That loop-assembly cluster is ledger L10,
    # not L4 — tighten this to 24/24 + closedness when L10 lands.
    covered = 0
    for a in np.linspace(0, 2 * np.pi, 24, endpoint=False):
        p = np.array([0.3 + 0.1 * np.cos(a), 0.3 + 0.1 * np.sin(a), 0.0])
        if min(_pt_poly(p, poly) for poly in polys) <= 5e-3:
            covered += 1
    assert covered >= 20, f"ring coverage collapsed: {covered}/24 samples"


def _cylinder_on_plane():
    """S1 deg (1,2): z = (2t-1)^2 — a parabolic cylinder TANGENT to z=0
    along the whole line t=0.5; S2: z=0 plane spanning [-0.5, 1.5]^2.
    Non-strict monotone T-hulls certify the top cell loop-free, so the
    tangent line is traced by the plain Ψ tracer through the LOOP-FREE
    path (never reaching `_deflate_tangent_cell`)."""
    S1 = np.array([[[0., 0., 1.], [0., 0.5, -1.], [0., 1., 1.]],
                   [[1., 0., 1.], [1., 0.5, -1.], [1., 1., 1.]]])
    S2 = np.array([[[-0.5, -0.5, 0.], [-0.5, 1.5, 0.]],
                   [[1.5, -0.5, 0.], [1.5, 1.5, 0.]]])
    return S1, S2


def test_tangent_line_loop_free_path_tagged_tangential():
    # Ledger L5 regression: the loop-free-path-traced tangent line shipped
    # kind='transversal' — breaking the output contract AND blinding the
    # kind-keyed subsumption filter, so the on-line center witness survived
    # as a stray tangent_point. Tagging is by MEASUREMENT
    # (_fragment_on_tangent_locus: normal alignment escalating to the
    # Δ-snap — the Ψ-marched samples wander ~1.4e-3 off t=0.5 in the
    # sub-tolerance valley, sin_ang up to 1.1e-2, so alignment alone can't
    # certify), never by provenance.
    # NOTE the mixed geometry z = (t-0.5)^2*(s-0.5) from the ledger is NOT
    # testable here: its tangent line t=0.5 is lost in ASSEMBLY (only
    # ~2e-3-long stubs near the degenerate X at (0.5,0.5) ship — measured
    # identical before/after this fix), an L10-family loss, so there is no
    # branch to assert a kind on.
    S1, S2 = _cylinder_on_plane()
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    assert len(r["branches"]) == 1
    b = r["branches"][0]
    assert b.kind == "tangential", f"tangent line shipped kind={b.kind!r}"
    xyz = np.asarray(b.curve[1])
    assert np.allclose(xyz[:, 1], 0.5, atol=2e-3)
    assert np.allclose(xyz[:, 2], 0.0, atol=2e-3)
    assert xyz[:, 0].min() < 0.02 and xyz[:, 0].max() > 0.98   # full span
    # the on-line witness is subsumed by the (now correctly kinded) branch
    assert r["singularities"] == [], (
        f"stray on-curve witnesses survived: "
        f"{[(g.kind, np.round(np.asarray(g.xyz), 4).tolist()) for g in r['singularities']]}")


def _plane_patch(x0, x1):
    return np.array([[[x0, 0., 0.], [x0, 1., 0.]],
                     [[x1, 0., 0.], [x1, 1., 0.]]], dtype=np.float64)


def test_partial_overlap_no_phantom_tangent_point():
    # Ledger L6(ii) regression: coplanar PARTIAL overlap (strip x in [1,2])
    # emitted a phantom "isolated" tangent_point at the strip's dead center
    # (0.75, 0.5, 0.25, 0.5) through the loop-free arm — every interior
    # point of a 2-dim overlap region is a Δ-root (paper Fig. 8 classifies
    # the overlap interior as 2-dim C2), and the strip INTERIOR is far from
    # every overlap-boundary polyline, so the post-assembly subsumption
    # filter could never catch it. Witness emission is now suppressed
    # inside the detected overlap regions' parametric boxes
    # (_overlap_region_boxes from the BoundaryOverlap segments).
    r = bez_ssx(_plane_patch(0, 2), _plane_patch(1, 3), 1e-3, rational=False)
    overlaps = [b for b in r["branches"] if b.kind == "overlap"]
    assert overlaps, "overlap branches lost — fixture no longer tests L6(ii)"
    assert r["singularities"] == [], (
        f"phantom tangent_point inside the overlap strip: "
        f"{[(g.kind, np.round(np.asarray(g.stuv), 4).tolist()) for g in r['singularities']]}")


def _offlattice_tangent_ring(cx=0.3, rad2=0.04):
    """S1: z = (r^2 - rad2)^2 with r^2 = (2(s-cx))^2 + (2(t-cx))^2
    (deg 4x4); S2: z=0. z >= 0 with equality exactly on the circle
    r^2 = rad2 — a CLOSED, crossing-less TANGENT ring of radius
    sqrt(rad2)/2 about (cx, cx), off the dyadic lattice (unlike
    `_closed_tangent_loop`, whose (0.5, 0.5) center lets every witness
    start ON the feature)."""
    from math import comb

    def mono_to_bern(a):
        return np.array([sum(a[j] * comb(k, j) / comb(4, j)
                             for j in range(min(k, len(a) - 1) + 1))
                         for k in range(5)])

    f = np.array([4.0 * cx * cx, -8.0 * cx, 4.0])        # (2(x-cx))^2
    f2 = np.convolve(f, f)
    z_st = np.zeros((5, 5))
    z_st[:5, 0] += f2
    z_st[0, :5] += f2
    z_st[:3, :3] += 2.0 * np.outer(f, f)
    z_st[:3, 0] -= 2.0 * rad2 * f
    z_st[0, :3] -= 2.0 * rad2 * f
    z_st[0, 0] += rad2 * rad2
    M = np.array([mono_to_bern(np.eye(5)[j]) for j in range(5)])
    Bz = M.T @ z_st @ M
    xs = mono_to_bern([0.0, 1.0])
    S1 = np.array([[[xs[i], xs[j], Bz[i, j]] for j in range(5)] for i in range(5)])
    S2 = np.array([[[-0.5, -0.5, 0.], [-0.5, 1.5, 0.]],
                   [[1.5, -0.5, 0.], [1.5, 1.5, 0.]]])
    rng = np.random.default_rng(3)
    for s, t in rng.uniform(0, 1, (20, 2)):
        r2 = (2 * (s - cx)) ** 2 + (2 * (t - cx)) ** 2
        p = eval_surface(_homog(S1), s, t, rational=True)
        assert np.allclose(p, [s, t, (r2 - rad2) ** 2], atol=1e-12)
    return S1, S2


def test_tangent_curve_no_point_flood():
    # Ledger L6(i) regression (B C4 family): a crossing-less 1-dim tangent
    # curve flooded spurious tangent_points — the crossing-less arm's full
    # enumeration ignored solve_zero_dim's `exhausted` flag (top cell here:
    # 16 deduped roots, exhausted=True — ptol-ladder SAMPLES of the ring,
    # not isolated touches), and `_emit_offcurve_tangent_roots` in
    # crossing-bearing descendants whose Φ-tracing produced nothing (no
    # tube) degraded to plain full enumeration (measured 33- and 17-point
    # floods). Both consumers now suppress emission on the 1-dim signature
    # (`_delta_roots_curve_like`: >12 roots, or exhausted with several) and
    # leave the curve to the tracing machinery, which ships it `tangential`.
    S1, S2 = _offlattice_tangent_ring()
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    tang = [b for b in r["branches"] if b.kind == "tangential"]
    assert tang, f"tangent ring lost: kinds={[b.kind for b in r['branches']]}"
    ring = max(tang, key=lambda b: len(b.curve[1]))
    rr = np.linalg.norm(np.asarray(ring.curve[1])[:, :2] - 0.3, axis=1)
    assert np.allclose(rr, 0.1, atol=5e-3)
    tps = [g for g in r["singularities"] if g.kind == "tangent_point"]
    assert len(tps) == 0, (
        f"point flood from the 1-dim tangent ring: {len(tps)} tangent_points "
        f"(was 4 surviving of ~50 emitted pre-fix)")


# ---------------------------------------------------------------------------
# L9: rational-input hygiene — TPsi minor nets from the quotient-rule
# numerator columns (dehomogenized control-point minors are the WRONG
# surface pair for non-uniform weights).
# ---------------------------------------------------------------------------

from mmcore.numeric.intersection.ssx._ssx5_singular import (
    _same_param_product_vec_scalar, minors_Tpsi_rational,
)


def _rationalize_s(S_h, wnet):
    """Multiply numerator (xyz) AND weight of a homogeneous net by the scalar
    net wnet(s) via the EXACT same-parameter Bernstein product — leaves the
    geometric surface P/w unchanged while making the weights non-uniform."""
    P = S_h[..., :3]; w = S_h[..., 3]
    Pnew = _same_param_product_vec_scalar(P, wnet)
    wnew = _same_param_product_vec_scalar(w[..., None], wnet)[..., 0]
    return np.concatenate([Pnew, wnew[..., None]], axis=-1)


def test_rational_tpsi_minors_match_true_jacobian():
    # The rational TPsi nets must match the TRUE rational-Jacobian minors in
    # SIGN and zero-crossings — each numerator minor equals the true minor
    # times a strictly positive power-of-W factor:
    #   T1,T2 : W1^2 W2^4 ;  T3,T4 : W1^4 W2^2   (W1,W2 > 0).
    from mmcore.numeric.bern import bernstein_eval_nd

    def _triple(a, b, c):
        return float(np.dot(a, np.cross(b, c)))

    def _rand_rat(rng, mu, mv):
        P = rng.uniform(-2, 2, (mu + 1, mv + 1, 3))
        w = rng.uniform(0.3, 3.0, (mu + 1, mv + 1))          # strictly positive
        return np.concatenate([P * w[..., None], w[..., None]], axis=-1)

    rng = np.random.default_rng(20260707)
    sign_ok = sign_tot = 0
    worst_rel = 0.0
    for _ in range(40):
        S1h = _rand_rat(rng, rng.integers(1, 4), rng.integers(1, 4))
        S2h = _rand_rat(rng, rng.integers(1, 4), rng.integers(1, 4))
        Tn = [np.asarray(T, dtype=np.float64) for T in minors_Tpsi_rational(S1h, S2h)]
        for _ in range(8):
            s, t, u, v = rng.uniform(0, 1, 4)
            net = np.array([bernstein_eval_nd(T[..., None], np.array([s, t, u, v])).item()
                            for T in Tn])
            _, R1s, R1t = eval_surface_d1(S1h, s, t, rational=True)
            _, R2u, R2v = eval_surface_d1(S2h, u, v, rational=True)
            tru = np.array([_triple(R1t, R2u, R2v), _triple(R1s, R2u, R2v),
                            -_triple(R1s, R1t, R2v), -_triple(R1s, R1t, R2u)])
            W1 = float(eval_surface(S1h, s, t, rational=False)[-1])
            W2 = float(eval_surface(S2h, u, v, rational=False)[-1])
            factor = np.array([W1**2 * W2**4, W1**2 * W2**4,
                               W1**4 * W2**2, W1**4 * W2**2])
            expected = factor * tru
            scale = np.maximum(np.abs(net), np.abs(expected))
            for k in range(4):
                sign_tot += 1
                if np.sign(net[k]) == np.sign(tru[k]) or scale[k] < 1e-12:
                    sign_ok += 1
                if scale[k] > 1e-9:
                    worst_rel = max(worst_rel, abs(net[k] - expected[k]) / scale[k])
    assert sign_ok == sign_tot, f"sign disagreement {sign_ok}/{sign_tot}"
    assert worst_rel < 1e-9, f"ratio error {worst_rel:.2e} exceeds 1e-9"


def test_rational_weights_tangent_point_found():
    # The paraboloid touch (isolated C2), rationalized EXACTLY by multiplying
    # numerator + weight by the deg-(1,0) net [[1],[2]] (weights 1..2, same
    # geometry). The dehomogenized-control-point minors describe a different
    # surface pair and lose the touch; the quotient-rule minors keep it.
    S1, S2 = _paraboloid_touch()
    S1_h = _homog(S1); S2_h = _homog(S2)
    S1_rat = _rationalize_s(S1_h, np.array([[1.0], [2.0]]))
    # geometry unchanged
    rng = np.random.default_rng(3)
    for s, t in rng.uniform(0, 1, (10, 2)):
        assert np.allclose(eval_surface(S1_h, s, t, rational=True),
                           eval_surface(S1_rat, s, t, rational=True), atol=1e-11)
    assert S1_rat[..., -1].max() > 1.5 and S1_rat[..., -1].min() < 1.01  # non-uniform
    r = bez_ssx(S1_rat, S2_h, 1e-3, rational=True)
    tps = [g for g in r["singularities"] if g.kind == "tangent_point"]
    assert len(tps) == 1, f"rationalized paraboloid lost its touch: {len(tps)} tangent_points"
    g = tps[0]
    assert np.allclose(g.stuv[:2], [0.5, 0.5], atol=1e-3)
    assert np.allclose(g.xyz, [0.5, 0.5, 0.0], atol=1e-3)
    assert r["branches"] == [] and r["points"] == []      # same contract as poly test


def test_rational_transversal_sanity_matches_polynomial():
    # A plain transversal pair, rationalized the same exact way, must trace
    # the SAME branch (1 branch, coverage within 5e-3 of the polynomial twin).
    s1 = np.array([[[0., 0., 5.], [0., 10., 5.]], [[10., 0., 5.], [10., 10., 5.]]])
    s2 = np.array([[[0., 0., 0.], [0., 10., 0.]], [[10., 0., 10.], [10., 10., 10.]]])
    rp = bez_ssx(s1, s2, 1e-3, rational=False)
    assert len(rp["branches"]) == 1
    S1_rat = _rationalize_s(_homog(s1), np.array([[1.0], [2.0]]))
    for s, t in np.random.default_rng(0).uniform(0, 1, (6, 2)):
        assert np.allclose(eval_surface(_homog(s1), s, t, rational=True),
                           eval_surface(S1_rat, s, t, rational=True), atol=1e-10)
    rr = bez_ssx(S1_rat, _homog(s2), 1e-3, rational=True)
    assert len(rr["branches"]) == 1, f"transversal rational gave {len(rr['branches'])} branches"
    poly = np.asarray(rp["branches"][0].curve[1])
    ratp = np.asarray(rr["branches"][0].curve[1])
    idx = np.linspace(0, len(poly) - 1, 20, dtype=int)
    for i in idx:
        assert _pt_poly(poly[i], ratp) <= 5e-3


# ---------------------------------------------------------------------------
# L13: C1 cusp acceptance scale must be weight-invariant (a pure weight
# rescale must not flip cusp detection).
# ---------------------------------------------------------------------------

def test_cusp_detection_weight_invariant():
    # Shipped cusp geometry: cuspidal edge ((2s-1)^2,(2s-1)^3,t) vs plane z=0.5.
    S1, S2 = _cusp_edge_case()
    S1_h = _homog(S1); S2_h = _homog(S2)

    def _run(S1_test):
        r = bez_ssx(S1_test, S2_h, 1e-3, rational=True)
        return [g for g in r["singularities"] if g.kind == "cusp"]

    # (a) pure constant weight rescale (surface UNCHANGED). The C1 GN
    # acceptance normalized the weight-invariant Cartesian normal residual by
    # a homogeneous-numerator scale (~W^4), so multiplying every weight by a
    # constant scaled the acceptance threshold by c^4 — a weight-dependent
    # bound (B C9). The fix normalizes by a sampled Cartesian normal scale
    # (weight-invariant). NOTE on this cuspidal-edge geometry the cusp is
    # found at ANY scale in BOTH old and new code (Sigma = 0 EXACTLY at the
    # cusp, so `norm(Nv) < tol*scale` holds for any positive scale); this is
    # a regression guard locking the invariance, not a live-flip repro
    # (exact cusps are inherently scale-robust — measured c in 0.003..100).
    S1_w3 = S1_h.copy(); S1_w3[..., :] *= 3.0     # P*=3 AND w*=3 => R = P/w unchanged
    for s, t in np.random.default_rng(1).uniform(0, 1, (8, 2)):
        assert np.allclose(eval_surface(S1_h, s, t, rational=True),
                           eval_surface(S1_w3, s, t, rational=True), atol=1e-11)
    cusps = _run(S1_w3)
    assert len(cusps) == 1, f"constant weight rescale flipped cusp detection: {len(cusps)}"
    assert np.allclose(cusps[0].xyz, [0.0, 0.0, 0.5], atol=1e-3)

    # (b) non-uniform weight net [[1],[2]] along s (surface still unchanged).
    S1_nu = _rationalize_s(S1_h, np.array([[1.0], [2.0], [1.0], [1.0]]))
    for s, t in np.random.default_rng(2).uniform(0, 1, (8, 2)):
        assert np.allclose(eval_surface(S1_h, s, t, rational=True),
                           eval_surface(S1_nu, s, t, rational=True), atol=1e-10)
    cusps_nu = _run(S1_nu)
    assert len(cusps_nu) == 1, f"non-uniform weights flipped cusp detection: {len(cusps_nu)}"
    assert np.allclose(cusps_nu[0].xyz, [0.0, 0.0, 0.5], atol=1e-3)


# ---------------------------------------------------------------------------
# Adversarial-review round: C1 (off-lattice loop topology) + C5 (determinism)
# ---------------------------------------------------------------------------

def _offcenter_touch_plus_loop(eps=0.04, ds=0.05, dt=0.02):
    """z = q(q-eps), q = (2s-1-2ds)^2 + (2t-1-2dt)^2 (deg 4x4): touch at
    (0.5+ds, 0.5+dt) OFF the midpoint-cut lattice, transversal circle of
    radius sqrt(eps)/2 around it. The generic-placement variant of
    _touch_plus_loop: guided cuts land tangent to the circle (grazing
    corners) and partition it into arcs — C1-review regression."""
    from math import comb
    Cs = np.array([(1 + 2 * ds) ** 2, -4 * (1 + 2 * ds), 4.0])   # (2s-1-2ds)^2
    Ct = np.array([(1 + 2 * dt) ** 2, -4 * (1 + 2 * dt), 4.0])
    M = np.zeros((5, 5))
    M[:5, 0] += np.convolve(Cs, Cs)
    M[0, :5] += np.convolve(Ct, Ct)
    M[:3, :3] += 2.0 * np.outer(Cs, Ct)
    M[:3, 0] -= eps * Cs
    M[0, :3] -= eps * Ct
    K = np.array([[comb(i, j) / comb(4, j) if j <= i else 0.0
                   for j in range(5)] for i in range(5)])
    Z = K @ M @ K.T
    xb = np.array([i / 4.0 for i in range(5)])
    S1 = np.array([[[xb[i], xb[j], Z[i, j]] for j in range(5)] for i in range(5)])
    S2 = np.array([[[-0.5, -0.5, 0.], [-0.5, 1.5, 0.]],
                   [[1.5, -0.5, 0.], [1.5, 1.5, 0.]]])
    rng = np.random.default_rng(3)
    for s, t in rng.uniform(0, 1, (25, 2)):
        q = (2 * s - 1 - 2 * ds) ** 2 + (2 * t - 1 - 2 * dt) ** 2
        p = eval_surface(_homog(S1), s, t, rational=True)
        assert np.allclose(p, [s, t, q * (q - eps)], atol=1e-10)
    return S1, S2


def test_offlattice_tangent_point_plus_loop():
    # C1 regression (final adversarial review, CONFIRMED critical): at any
    # generic (off-lattice) touch placement the loop used to come back as
    # FOUR out-and-back "closed" doubled arcs with two ~59-deg sectors
    # silently missing (52*atol worst gap) — grazing-corner seeds bounced
    # both march directions, junction slivers were eaten by containment
    # dedup's terminal-vertex clamp, and the closing march retraced arcs
    # backward. Must be: ONE closed loop, full circle, one tangent point.
    S1, S2 = _offcenter_touch_plus_loop(0.04, 0.05, 0.02)
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    sing = [g for g in r["singularities"] if g.kind == "tangent_point"]
    assert len(sing) == 1
    assert np.allclose(sing[0].xyz, [0.55, 0.52, 0.0], atol=2e-3)
    loops = [b for b in r["branches"]
             if np.linalg.norm(np.asarray(b.curve[1])[0] - np.asarray(b.curve[1])[-1]) < 5e-3]
    assert len(loops) == 1, f"expected 1 closed loop, got {len(loops)} of {len(r['branches'])}"
    xyz = np.asarray(loops[0].curve[1])
    rr = np.linalg.norm(xyz[:, :2] - np.array([0.55, 0.52]), axis=1)
    assert np.allclose(rr, 0.1, atol=5e-3)
    # full angular coverage: every 5-deg sector of the true circle hit
    th = np.radians(np.arange(0.0, 360.0, 5.0))
    ring = np.stack([0.55 + 0.1 * np.cos(th), 0.52 + 0.1 * np.sin(th),
                     np.zeros_like(th)], axis=1)
    for p in ring:
        assert _pt_poly(p, xyz) < 2e-3, f"circle point {p} missed"


def test_repeated_calls_bit_identical():
    # C5 regression (final adversarial review, CONFIRMED major): the
    # hemisphere-witness shuffle consumed module-global PRNG state, so
    # bit-identical inputs returned different branch topologies by call
    # index ([3,2,3,2,2,2] measured on this exact fixture) — the witness
    # RNGs are now reset at every bez_ssx entry. Marginal near-tangent
    # geometry chosen deliberately: regular cases never flipped.
    S1, S2 = _touch_plus_loop(0.012)
    sigs = []
    for _ in range(2):
        r = bez_ssx(S1.copy(), S2.copy(), 5e-3, rational=False)
        sig = (len(r["branches"]), len(r["points"]), len(r["singularities"]),
               tuple(np.asarray(b.curve[1]).tobytes() for b in r["branches"]))
        sigs.append(sig)
    assert sigs[0] == sigs[1], "bez_ssx is call-history dependent"
    # C6 regression on the same runs (CONFIRMED major): a 2-point
    # 'transversal' bridge from the touch to the ring used to survive all
    # filters — its single chord slides along the sub-atol grazing valley
    # (samples Ψ-valid, chord fiction: est. true-curve distance
    # residual/sin_ang ≈ 4.6·atol). Expect exactly the ring + the touch.
    assert len(r["branches"]) == 1
    assert len(r["points"]) == 0
    xyz = np.asarray(r["branches"][0].curve[1])
    assert np.linalg.norm(xyz[0] - xyz[-1]) < 1e-9      # closed
    rr = np.linalg.norm(xyz[:, :2] - 0.5, axis=1)
    assert np.allclose(rr, np.sqrt(0.012) / 2.0, atol=2.5e-3)


def test_aniso_tangent_point_plus_thin_loop():
    # C6/C2-adjacent regression (final adversarial review): aspect-16
    # ellipse (cap curvature κ≈160) — two Φ∩L seeds each marched a full
    # copy of the loop and 2·atol-sagitta samplings of the caps sat
    # ~3.7·atol apart, defeating containment dedup: the result carried TWO
    # closed duplicate rings plus partial-arc debris. Seeded closed loops
    # now march at 0.5·atol sagitta. Expect ONE closed ellipse + the touch.
    from math import comb
    eps, k = 0.04, 16.0
    C = np.array([1.0, -4.0, 4.0])
    M = np.zeros((5, 5))
    C2 = np.convolve(C, C)
    M[:5, 0] += C2
    M[0, :5] += (k * k) * C2
    M[:3, :3] += 2.0 * k * np.outer(C, C)
    M[:3, 0] -= eps * C
    M[0, :3] -= eps * k * C
    K = np.array([[comb(i, j) / comb(4, j) if j <= i else 0.0
                   for j in range(5)] for i in range(5)])
    Z = K @ M @ K.T
    xb = np.array([i / 4.0 for i in range(5)])
    S1 = np.array([[[xb[i], xb[j], Z[i, j]] for j in range(5)] for i in range(5)])
    S2 = np.array([[[-0.5, -0.5, 0.], [-0.5, 1.5, 0.]],
                   [[1.5, -0.5, 0.], [1.5, 1.5, 0.]]])
    rng = np.random.default_rng(5)
    for s, t in rng.uniform(0, 1, (25, 2)):
        q = (2 * s - 1) ** 2 + k * (2 * t - 1) ** 2
        p = eval_surface(_homog(S1), s, t, rational=True)
        assert np.allclose(p, [s, t, q * (q - eps)], atol=1e-10)

    r = bez_ssx(S1, S2, 1e-3, rational=False)
    sing = [g for g in r["singularities"] if g.kind == "tangent_point"]
    assert len(sing) == 1 and np.allclose(sing[0].xyz, [0.5, 0.5, 0.0], atol=2e-3)
    assert len(r["branches"]) == 1, \
        f"expected the single ellipse, got {len(r['branches'])} branches"
    xyz = np.asarray(r["branches"][0].curve[1])
    assert np.linalg.norm(xyz[0] - xyz[-1]) < 1e-9      # closed
    # on the true ellipse: q = eps
    q = (2 * xyz[:, 0] - 1) ** 2 + k * (2 * xyz[:, 1] - 1) ** 2
    assert np.allclose(q, eps, atol=0.2 * eps)
    # both semi-axes reached
    rr = np.linalg.norm(xyz[:, :2] - 0.5, axis=1)
    assert abs(rr.max() - np.sqrt(eps) / 2.0) < 5e-3
    assert abs(rr.min() - np.sqrt(eps / k) / 2.0) < 5e-3


def test_cone_apex_no_phantom_tangent_point():
    # Ledger L15 regression: at a Sigma=0 parameterization degeneracy all
    # four T-Psi minors vanish regardless of geometry, so the Delta-witness
    # converges at the bilinear cone's apex although the 3D crossing there
    # is TRANSVERSAL — a phantom C2 'tangent_point' shipped alongside the
    # correct C1 'cusp_curve'. The C2 emitters now reject Sigma=0 roots
    # (normal-degeneracy test, reparameterization-invariant).
    S1 = np.array([[[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
                   [[0.0, 1.0, -1.0], [1.0, 1.0, 1.0]]])   # (s*t, s, s*(2t-1))
    S2 = np.array([[[-1.5, -1.5, 0.0], [-1.5, 1.5, 0.0]],
                   [[1.5, -1.5, 0.0], [1.5, 1.5, 0.0]]])
    for s, t in [(0.3, 0.8), (0.0, 0.5), (1.0, 0.25), (0.6, 0.5)]:
        p = eval_surface(_homog(S1), s, t, rational=True)
        assert np.allclose(p, [s * t, s, s * (2 * t - 1)], atol=1e-12)
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    assert [g for g in r["singularities"] if g.kind == "tangent_point"] == [], \
        "phantom C2 tangent_point at the Sigma=0 apex"
    # the C1 classification must remain
    c1 = [g for g in r["singularities"] if g.kind in ("cusp", "cusp_curve")]
    assert c1 and all(g.surface == 1 for g in c1)
    assert any(np.linalg.norm(g.xyz) < 5e-3 for g in c1)
    # and the transversal branch through the apex is traced
    assert len(r["branches"]) == 1
    assert r["branches"][0].kind == "transversal"


def test_small_nonzero_surface_normal_is_not_retyped_as_degenerate():
    """A poorly conditioned but regular parametrization is still regular."""
    from mmcore.numeric.monomial import monomial_to_bezier
    from mmcore.numeric.intersection.ssx._bez_ssx5 import (
        _normals_degenerate_at)

    eps = 5e-7
    mono = np.zeros((3, 4, 3), dtype=np.float64)
    mono[1, 0, 0] = 1.0
    mono[0, 1, 0] = 1.0                    # x = s + t
    mono[0, 0, 1] = -0.125
    mono[0, 1, 1] = 0.75 + eps
    mono[0, 2, 1] = -1.5
    mono[0, 3, 1] = 1.0                    # y = (t-.5)^3 + eps*t
    mono[2, 0, 2] = 1.0
    mono[1, 0, 2] = -1.0
    mono[0, 2, 2] = 1.0
    mono[0, 1, 2] = -1.0
    mono[0, 0, 2] = 0.5                    # z = (s-.5)^2 + (t-.5)^2
    surface = _homog(monomial_to_bezier(mono))
    plane = _homog(np.array([
        [[0., 0., 0.], [0., 1., 0.]],
        [[1., 0., 0.], [1., 1., 0.]],
    ]))

    _, ds, dt = eval_surface_d1(surface, 0.5, 0.5, rational=True)
    assert np.linalg.norm(np.cross(ds, dt)) == pytest.approx(eps)
    assert not _normals_degenerate_at(
        surface, plane, np.full(4, 0.5))


def test_short_cusp_curve_typed_as_curve_not_isolated():
    # Ledger L14 regression: a cusp curve clipped to t-extent 0.2 (200x
    # resolution) yielded 11 xyz-dedup-sparse solutions — under the raw
    # count>12/exhausted heuristic it shipped as 11 isolated 'cusp's.
    # The connectivity probe (midpoint Newton between consecutive
    # solutions along the cloud's principal axis) types it 'cusp_curve'.
    from mmcore.numeric.intersection.ssx._ssx5_singular import c1_pass
    from mmcore.geom._nurbs_param_tol import bez_surface_param_tolerance
    x3 = [1.0, -1.0 / 3.0, -1.0 / 3.0, 1.0]
    y3 = [-1.0, 1.0, -1.0, 1.0]
    S1 = np.array([[[x3[i], y3[i], float(j)] for j in range(2)] for i in range(4)])
    S2 = np.array([[[0.0, -1.5, 0.4], [0.0, 1.5, 0.4]],
                   [[0.0, -1.5, 0.6], [0.0, 1.5, 0.6]]])   # plane x=0, z in [0.4,0.6]
    S1h, S2h = _homog(S1), _homog(S2)
    atol = 1e-3
    ps, pt = bez_surface_param_tolerance(S1h, atol, rational=True)
    pu, pv = bez_surface_param_tolerance(S2h, atol, rational=True)
    ptol4 = np.maximum(np.array([float(ps), float(pt), float(pu), float(pv)]), 1e-9)
    hits, flag = c1_pass(S1h, S2h, atol, ptol4)
    assert flag
    assert len(hits) == 1 and "curve_samples" in hits[0]
    assert len(hits[0]["curve_samples"]) >= 2
    assert hits[0]["surface"] == 1


def test_two_isolated_cusps_stay_isolated():
    # Ledger L14 negative control: du=0 at BOTH s=0.3 and s=0.7 (two
    # genuinely isolated parameterization cusps on one surface) — the
    # connectivity probe must NOT glue them into a phantom cusp_curve
    # (their midpoint Newton falls back onto an endpoint or diverges).
    from math import comb
    from mmcore.numeric.intersection.ssx._ssx5_singular import c1_pass
    from mmcore.geom._nurbs_param_tol import bez_surface_param_tolerance

    def mono_to_bern(a, n):
        return np.array([sum(a[j] * comb(i, j) / comb(n, j)
                             for j in range(min(i, len(a) - 1) + 1))
                         for i in range(n + 1)])

    # x'(s) = 6(s-0.3)(s-0.7),  y'(s) = 24(s-0.3)(s-0.7)(s-0.5)
    xb = mono_to_bern(6.0 * np.array([0.0, 0.21, -0.5, 1.0 / 3.0, 0.0]), 4)
    yb = mono_to_bern(24.0 * np.array([0.0, -0.105, 0.355, -0.5, 0.25]), 4)
    S1 = np.array([[[xb[i], yb[i], float(j)] for j in range(2)] for i in range(5)])
    for s in (0.15, 0.3, 0.5, 0.7, 0.9):
        p = eval_surface(_homog(S1), s, 0.4, rational=True)
        want_x = 6.0 * (s ** 3 / 3 - 0.5 * s ** 2 + 0.21 * s)
        want_y = 24.0 * (0.25 * s ** 4 - 0.5 * s ** 3 + 0.355 * s ** 2 - 0.105 * s)
        assert np.allclose(p, [want_x, want_y, 0.4], atol=1e-12)
    S2 = np.array([[[-1.0, -1.0, 0.5], [-1.0, 1.0, 0.5]],
                   [[1.0, -1.0, 0.5], [1.0, 1.0, 0.5]]])
    S1h, S2h = _homog(S1), _homog(S2)
    atol = 1e-3
    ps, pt = bez_surface_param_tolerance(S1h, atol, rational=True)
    pu, pv = bez_surface_param_tolerance(S2h, atol, rational=True)
    ptol4 = np.maximum(np.array([float(ps), float(pt), float(pu), float(pv)]), 1e-9)
    hits, flag = c1_pass(S1h, S2h, atol, ptol4)
    assert not flag, "two isolated cusps wrongly glued into a cusp_curve"
    cusps = [h for h in hits if "stuv" in h]
    assert len(cusps) == 2
    assert sorted(round(float(h["stuv"][0]), 2) for h in cusps) == [0.3, 0.7]


@pytest.mark.parametrize("k", [10.0, 100.0])
def test_touch_detection_scale_covariant(k):
    # Ledger L18 regression (B C13): at 7d47f68 the double-touch fixture
    # lost its typed tangent_points at x10 coordinate scale — INSIDE the
    # documented O(1)-O(100) envelope. Fixed by the intervening gates
    # (L4 probe-descent witness fallback, L1 hull margins): measured
    # clean through x100 on paraboloid, double-touch and saddle. The
    # envelope boundary now sits at ~x300 (1 of 2 touches) / x1000 (0) —
    # OUTSIDE the documented envelope, the pre-existing accepted limit
    # (plan risk 6). Scaling coords AND atol by k is a pure unit change:
    # the result must be covariant.
    S1, S2 = _paraboloid_touch()
    r = bez_ssx(S1 * k, S2 * k, 1e-3 * k, rational=False)
    tps = [g for g in r["singularities"] if g.kind == "tangent_point"]
    assert len(tps) == 1
    assert np.allclose(np.asarray(tps[0].xyz) / k, [0.5, 0.5, 0.0], atol=1e-3)
    assert r["branches"] == [] and r["points"] == []

    S1d, S2d = _double_touch_asym()
    r = bez_ssx(S1d * k, S2d * k, 1e-3 * k, rational=False)
    tps = sorted((g for g in r["singularities"] if g.kind == "tangent_point"),
                 key=lambda g: float(g.stuv[0]))
    assert len(tps) == 2
    assert np.allclose(tps[0].stuv[:2], [0.45, 0.5], atol=1e-3)
    assert np.allclose(tps[1].stuv[:2], [0.9, 0.5], atol=1e-3)
    assert r["branches"] == [] and r["points"] == []


# ---------------------------------------------------------------------------
# Ledger L27: corner-sharing bilinear patches (user-reported)
# ---------------------------------------------------------------------------

def _shared_edge_pair(flatten_s2=False):
    """Two bilinear patches sharing THREE corners and TWO whole edges:
    S1(s=1 edge) == S2(v=0 edge) (segment B-A) and S1(t=0 edge) ==
    S2(u=1 edge) (segment C-A), meeting at A = (48.557, -45.524, 0) where
    the surfaces are exactly tangent (sin_ang = 0). True SSI: the two
    shared edges as overlap curves + the junction tangency at A."""
    S1 = np.array([[[78.47558485, -33.55696868, 0.], [90.44303132, -69.45930808, 5.98372323]],
                   [[48.55696868, -45.52441515, 0.], [60.52441515, -75.44303132, 0.]]])
    S2 = np.array([[[60.52441515, -75.44303132, 0.], [93.44303132, -69.45930808, -1.98372323]],
                   [[48.55696868, -45.52441515, 0.], [78.47558485, -33.55696868, 0.]]])
    if flatten_s2:
        S2 = S2.copy()
        S2[..., 2] = 0.0
    return S1, S2


@pytest.mark.parametrize("flatten_s2", [False, True])
def test_shared_edges_junction_tangent_point(flatten_s2):
    # L27 regression (user report): the CSX boundary-overlap claims for the
    # shared edges carried garbage index-paired (u,v) endpoints (a whole
    # edge "overlapping" one surface point, residual ~39 model units =
    # 39000*atol); the unverified claims shipped as 3 garbage branches AND
    # their endpoints filtered out the genuine crossings, starving one true
    # branch to a 0.142 stub. The flat-projection variant additionally ran
    # orders of magnitude slower. Now: overlap endpoints are resolved by
    # point inversion + the shipped chord is residual-verified, and the
    # junction of the two overlap curves is emitted structurally as the
    # tangent_point the geometry demands (sin_ang = 0 measured at A).
    A = np.array([48.55696868, -45.52441515, 0.0])
    B = np.array([60.52441515, -75.44303132, 0.0])
    C = np.array([78.47558485, -33.55696868, 0.0])
    S1, S2 = _shared_edge_pair(flatten_s2)
    import time
    t0 = time.time()
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    dt = time.time() - t0
    assert dt < 10.0, f"corner case took {dt:.1f}s (flat-variant slowdown regression)"

    tps = [g for g in r["singularities"] if g.kind == "tangent_point"]
    assert len(tps) == 1
    assert np.allclose(tps[0].xyz, A, atol=2e-3)

    assert len(r["branches"]) == 2
    ends = []
    for b in r["branches"]:
        assert b.kind == "overlap"
        xyz = np.asarray(b.curve[1])
        # every shipped sample must be ON both surfaces
        stuv = np.asarray(b.curve[0])
        for s4 in stuv:
            p1 = eval_surface(_homog(S1), s4[0], s4[1], rational=True)
            p2 = eval_surface(_homog(S2), s4[2], s4[3], rational=True)
            assert np.linalg.norm(p1 - p2) <= 2e-3
        ends.append({tuple(np.round(xyz[0], 3)), tuple(np.round(xyz[-1], 3))})
    want_AB = {tuple(np.round(A, 3)), tuple(np.round(B, 3))}
    want_AC = {tuple(np.round(A, 3)), tuple(np.round(C, 3))}
    assert (ends[0] == want_AB and ends[1] == want_AC) or \
           (ends[0] == want_AC and ends[1] == want_AB)
    assert r["points"] == []


# ---------------------------------------------------------------------------
# Ledger L30 / L26: rational cone tangent branch and no-hang budgets
# ---------------------------------------------------------------------------

def test_rational_delta_witness_evaluates_true_surfaces():
    """The Delta refiner must not use per-control-point dehomogenization.

    On case 14's exact rational cone generator, the false polynomial
    ``P_i / w_i`` surfaces miss by hundreds of ``atol``.  The rational
    witness must converge on the true homogeneous quotient surfaces.
    """
    from examples.ssx.bez_ssx5_case14 import S1, S2
    from mmcore.numeric.intersection.ssx._bez_ssx5 import _delta_float_gn
    from mmcore.numeric.intersection.ssx._ssx5_singular import minors_Tpsi_rational

    T = minors_Tpsi_rational(S1, S2)
    gn, _ = _delta_float_gn(*T, S1, S2, rational=True, atol=1e-3)
    root = gn(np.full(4, 0.5))
    assert root is not None
    p1 = eval_surface(S1, root[0], root[1], rational=True)
    p2 = eval_surface(S2, root[2], root[3], rational=True)
    assert np.linalg.norm(p1 - p2) < 1e-7
    assert abs(root[0] - 0.251168868) < 1e-5
    assert abs(root[2] - 0.109103476) < 1e-5

    P1_wrong = S1[..., :-1] / S1[..., -1:]
    P2_wrong = S2[..., :-1] / S2[..., -1:]
    wrong_res = np.linalg.norm(
        eval_surface(P1_wrong, root[0], root[1], rational=False)
        - eval_surface(P2_wrong, root[2], root[3], rational=False))
    assert wrong_res > 0.1


def test_rational_delta_scaling_does_not_accept_physical_gap():
    """Conditioning scale must not become the physical root tolerance."""
    span = 1.0e9
    p1 = np.array([
        [[0.0, 0.0, 0.0], [0.0, span, 0.0]],
        [[span, 0.0, 0.0], [span, span, 0.0]],
    ])
    p2 = p1.copy()
    p2[..., 2] = 1.0e-2
    S1 = np.concatenate([p1, np.ones((2, 2, 1))], axis=-1)
    S2 = np.concatenate([p2, np.ones((2, 2, 1))], axis=-1)
    zero_minor = np.zeros((1, 1, 1, 1), dtype=np.float64)

    from mmcore.numeric.intersection.ssx._bez_ssx5 import _delta_float_gn
    gn, _ = _delta_float_gn(
        zero_minor, zero_minor, zero_minor, zero_minor,
        S1, S2, rational=True, atol=1e-3)
    assert gn(np.full(4, 0.5)) is None


def test_rational_delta_scaling_does_not_retype_transversal_loop():
    """A large remote T coefficient cannot hide physical normal mismatch."""
    from scipy.signal import convolve2d
    from mmcore.numeric.monomial import monomial_to_bezier
    from mmcore.numeric.intersection._deflate import (
        minors_Tpsi_from_control_nets)
    from mmcore.numeric.intersection.ssx._bez_ssx5 import (
        _check_tangency, _delta_float_gn)

    radius, scale = 0.2, 1.0e7
    delta = 0.01 / scale
    q = np.zeros((3, 3))
    q[2, 0] = 1.0
    q[1, 0] = -2.0 * (0.5 + radius)
    q[0, 0] = (0.5 + radius) ** 2 + 0.25 - radius ** 2
    q[0, 2] = 1.0
    q[0, 1] = -1.0
    h = np.zeros((3, 3))
    h[2, 0] = 1.0
    h[1, 0] = -1.0
    h[0, 0] = 0.5 + delta
    h[0, 2] = 1.0
    h[0, 1] = -1.0
    f = scale * convolve2d(q, h)

    m1 = np.zeros((5, 5, 3))
    m1[1, 0, 0] = 1.0
    m1[0, 1, 1] = 1.0
    m1[..., 2] = f
    m2 = np.zeros((5, 5, 3))
    m2[1, 0, 0] = 1.0
    m2[0, 1, 1] = 1.0
    p1, p2 = monomial_to_bezier(m1), monomial_to_bezier(m2)
    s1 = np.concatenate([p1, np.ones(p1.shape[:2] + (1,))], axis=-1)
    s2 = np.concatenate([p2, np.ones(p2.shape[:2] + (1,))], axis=-1)
    T = minors_Tpsi_from_control_nets(p1.tolist(), p2.tolist())
    x = np.full(4, 0.5)

    _, du1, dv1 = eval_surface_d1(s1, x[0], x[1], rational=True)
    _, du2, dv2 = eval_surface_d1(s2, x[2], x[3], rational=True)
    n1, n2 = np.cross(du1, dv1), np.cross(du2, dv2)
    sin_angle = float(np.linalg.norm(np.cross(n1, n2))
                      / (np.linalg.norm(n1) * np.linalg.norm(n2)))
    assert sin_angle > 3e-3

    gn, _ = _delta_float_gn(
        *T, s1, s2, rational=True, atol=1e-3)
    assert gn(x) is None
    assert _check_tangency(
        *T, s1, s2, ((0.0, 1.0),) * 4,
        rational=True, atol=1e-3) is not True


def test_rational_minor_fast_path_requires_exact_uniform_weights():
    from mmcore.numeric.intersection.ssx._bez_ssx5 import (
        _weight_net_uniform)

    surface = np.zeros((2, 2, 4), dtype=np.float64)
    surface[..., -1] = 3.0
    assert _weight_net_uniform(surface)
    surface[1, 1, -1] += 5e-13
    assert not _weight_net_uniform(surface)


def test_transversal_branch_between_two_collapsed_rational_fibers_is_preserved():
    """Collapsed endpoint fibers must still seed a regular interior branch.

    Each surface below is a rational quadratic cone wedge with one collapsed
    boundary edge.  The two wedges meet transversally along the z axis, but
    both ends of that SSI are represented by positive-dimensional boundary
    parameter fibers.  Treating fibers only as C2/tangency metadata drops the
    entire regular branch.
    """
    a = np.sqrt(0.5)
    weights = np.array([[1.0, 1.0], [a, a], [1.0, 1.0]])
    A = np.array([0.0, 0.0, 0.0])
    B = np.array([0.0, 0.0, 1.0])

    s1_euclidean = np.empty((3, 2, 3), dtype=np.float64)
    s1_euclidean[:, 0, :] = A
    s1_euclidean[:, 1, :] = np.array([
        B + [1.0, 0.0, 0.0], B, B - [1.0, 0.0, 0.0],
    ])
    s2_euclidean = np.empty((3, 2, 3), dtype=np.float64)
    s2_euclidean[:, 0, :] = B
    s2_euclidean[:, 1, :] = np.array([
        A + [0.0, 1.0, 0.0], A, A - [0.0, 1.0, 0.0],
    ])

    S1 = np.concatenate(
        [s1_euclidean * weights[..., None], weights[..., None]], axis=-1)
    S2 = np.concatenate(
        [s2_euclidean * weights[..., None], weights[..., None]], axis=-1)

    r = bez_ssx(
        S1, S2, 1e-3, rational=True,
        max_cells=60_000, max_csx_calls=2_000,
    )
    assert r["complete"] is False
    assert r["status"]["reasons"]
    assert r["status"]["work"]["cells_processed"] <= 60_000
    assert len(r["branches"]) == 1
    assert r["points"] == []
    assert [g for g in r["singularities"]
            if g.kind == "tangent_point"] == []
    regular = [b for b in r["branches"] if b.kind == "transversal"]
    assert len(regular) == 1
    stuv = np.asarray(regular[0].curve[0])
    xyz = np.asarray(regular[0].curve[1])
    expected_ends = np.array([
        [0.5, 0.0, 0.5, 1.0],
        [0.5, 1.0, 0.5, 0.0],
    ])
    direct = max(np.linalg.norm(stuv[0] - expected_ends[0]),
                 np.linalg.norm(stuv[-1] - expected_ends[1]))
    reverse = max(np.linalg.norm(stuv[0] - expected_ends[1]),
                  np.linalg.norm(stuv[-1] - expected_ends[0]))
    assert min(direct, reverse) <= 2e-3
    assert np.linalg.norm(xyz[-1] - xyz[0]) > 0.99
    assert np.linalg.norm(np.diff(xyz, axis=0), axis=1).sum() > 0.99
    for q in np.linspace(0.0, 1.0, 21):
        assert _pt_poly(np.array([0.0, 0.0, q]), xyz) <= 5e-3

    interior_sines = []
    for i, x in enumerate(stuv):
        p1 = eval_surface(S1, x[0], x[1], rational=True)
        p2 = eval_surface(S2, x[2], x[3], rational=True)
        assert np.linalg.norm(p1 - p2) <= 2e-3
        if 0 < i < len(stuv) - 1:
            _, ds1, dt1 = eval_surface_d1(
                S1, x[0], x[1], rational=True)
            _, ds2, dt2 = eval_surface_d1(
                S2, x[2], x[3], rational=True)
            n1, n2 = np.cross(ds1, dt1), np.cross(ds2, dt2)
            interior_sines.append(
                np.linalg.norm(np.cross(n1, n2))
                / (np.linalg.norm(n1) * np.linalg.norm(n2)))
    assert interior_sines and min(interior_sines) >= 0.9

    swapped = bez_ssx(
        S2, S1, 1e-3, rational=True,
        max_cells=60_000, max_csx_calls=2_000,
    )
    swapped_regular = [
        b for b in swapped["branches"] if b.kind == "transversal"]
    assert len(swapped_regular) == 1
    swapped_xyz = np.asarray(swapped_regular[0].curve[1])
    for q in np.linspace(0.0, 1.0, 21):
        assert _pt_poly(np.array([0.0, 0.0, q]), swapped_xyz) <= 5e-3


def test_case14_rational_cones_return_certified_branch_and_explicit_partial_status():
    from examples.ssx.bez_ssx5_case14 import S1, S2

    r = bez_ssx(
        S1, S2, 1e-3, rational=True,
        max_cells=60_000, max_csx_calls=2_000,
    )
    # The tangent generator is certified, but the positive-dimensional
    # Delta complement search cannot prove that no additional isolated root
    # exists before its local frontier cap.  That is useful partial output,
    # never a complete topology claim.
    assert r["complete"] is False
    assert r["status"]["work"]["cells_processed"] <= 60_000
    assert len(r["branches"]) == 1
    assert [g for g in r["singularities"]
            if g.kind == "tangent_point"] == []
    branch = r["branches"][0]
    assert branch.kind == "tangential"
    xyz = np.asarray(branch.curve[1])

    # Ground truth is the shared cone generator between the two apices.
    a = eval_surface(S1, 0.251168868, 0.0, rational=True)
    b = eval_surface(S1, 0.251168868, 1.0, rational=True)
    for q in np.linspace(0.0, 1.0, 21):
        assert _pt_poly((1.0 - q) * a + q * b, xyz) <= 5e-3

    stuv = np.asarray(branch.curve[0])
    for x in stuv:
        p1 = eval_surface(S1, x[0], x[1], rational=True)
        p2 = eval_surface(S2, x[2], x[3], rational=True)
        assert np.linalg.norm(p1 - p2) <= 2e-3


def test_case14_tangent_generator_is_homogeneous_scale_invariant():
    """Independent common weight factors cannot change rational geometry."""
    from examples.ssx.bez_ssx5_case14 import S1, S2

    r = bez_ssx(
        S1 * 1e-6, S2 * 1e6, 1e-3, rational=True,
        max_cells=60_000, max_csx_calls=2_000,
    )
    tangential = [b for b in r["branches"] if b.kind == "tangential"]
    assert len(tangential) == 1
    assert [g for g in r["singularities"]
            if g.kind == "tangent_point"] == []
    for x in np.asarray(tangential[0].curve[0]):
        p1 = eval_surface(S1, x[0], x[1], rational=True)
        p2 = eval_surface(S2, x[2], x[3], rational=True)
        assert np.linalg.norm(p1 - p2) <= 2e-3
    xyz = np.asarray(tangential[0].curve[1])
    a = eval_surface(S1, 0.251168868, 0.0, rational=True)
    b = eval_surface(S1, 0.251168868, 1.0, rational=True)
    for q in np.linspace(0.0, 1.0, 21):
        assert _pt_poly((1.0 - q) * a + q * b, xyz) <= 5e-3


def test_case13_rational_tangency_terminates_with_explicit_partial_status():
    """A deduplicated tangency must not re-run Phi seeding per descendant."""
    from examples.ssx.bez_ssx5_case13 import S1, S2

    r = bez_ssx(
        S1, S2, 1e-3, rational=True,
        max_cells=30_000, max_csx_calls=2_000,
    )

    # A residual near-tangent cell reaches a depth/CSX uncertainty frontier.
    # The point below is certified, but absence of more topology is not.
    assert r["complete"] is False
    assert r["status"]["work"]["cells_processed"] <= 30_000
    assert r["branches"] == [] and r["points"] == []
    tangencies = [g for g in r["singularities"]
                  if g.kind == "tangent_point"]
    assert len(tangencies) == 1
    x = np.asarray(tangencies[0].stuv)
    assert np.allclose(
        x, [0.3066527232, 0.3043305321, 0.3786701630, 0.3198906377],
        atol=2e-3)
    p1 = eval_surface(S1, x[0], x[1], rational=True)
    p2 = eval_surface(S2, x[2], x[3], rational=True)
    assert np.linalg.norm(p1 - p2) <= 2e-3
    assert np.allclose(p1, [4.639676354, -3.932548333, 0.295239623],
                       atol=2e-3)
    _, du1, dv1 = eval_surface_d1(S1, x[0], x[1], rational=True)
    _, du2, dv2 = eval_surface_d1(S2, x[2], x[3], rational=True)
    n1, n2 = np.cross(du1, dv1), np.cross(du2, dv2)
    sin_angle = float(np.linalg.norm(np.cross(n1, n2))
                      / (np.linalg.norm(n1) * np.linalg.norm(n2)))
    assert sin_angle <= 1e-6


def test_bez_ssx_global_soft_budget_returns_partial_result(monkeypatch):
    import mmcore.numeric.intersection.ssx._bez_ssx5 as ssx5

    s1 = np.array([[[0., 0., 0.], [0., 1., 0.]],
                   [[1., 0., 0.], [1., 1., 0.]]])
    s2 = np.array([[[0., 0., -1.], [0., 1., 1.]],
                   [[1., 0., -1.], [1., 1., 1.]]])
    def forbidden(*_args, **_kwargs):
        pytest.fail("the 4-D distance net was built with zero allowance")

    monkeypatch.setattr(
        ssx5, "surface_surface_distance_squared_net_homog", forbidden)
    r = ssx5.bez_ssx(
        s1, s2, 1e-3, rational=False,
        max_cells=0, max_csx_calls=8,
    )
    assert r["complete"] is False
    assert r["status"]["reasons"] == ["work_budget"]
    assert set(("branches", "points", "singularities")) <= set(r)


def test_bez_ssx_preflights_distance_net_work(monkeypatch):
    """A tiny global allowance cannot start a superlinear 4-D net build."""
    import mmcore.numeric.intersection.ssx._bez_ssx5 as ssx5

    surface = np.zeros((8, 8, 3), dtype=np.float64)
    surface[..., 0] = np.linspace(0.0, 1.0, 8)[:, None]
    surface[..., 1] = np.linspace(0.0, 1.0, 8)[None, :]

    def forbidden(*_args, **_kwargs):
        pytest.fail("distance-net construction bypassed its preflight charge")

    monkeypatch.setattr(
        ssx5, "surface_surface_distance_squared_net_homog", forbidden)
    result = ssx5.bez_ssx(
        surface, surface, 1e-3, rational=False,
        max_cells=1, max_csx_calls=1,
    )
    assert result["complete"] is False
    assert result["status"]["reasons"] == ["work_budget"]
    assert result["status"]["work"]["cells_processed"] == 0


def test_bez_ssx_output_budget_bounds_postprocessing():
    """A finite cell queue must not feed unbounded quadratic assembly."""
    s1 = np.array([[[0., 0., 0.], [0., 1., 0.]],
                   [[1., 0., 0.], [1., 1., 0.]]])
    s2 = np.array([[[0., 0., -1.], [0., 1., 1.]],
                   [[1., 0., -1.], [1., 1., 1.]]])
    r = bez_ssx(
        s1, s2, 1e-3, rational=False,
        max_output_items=0,
    )
    assert r["complete"] is False
    assert "output_cap" in r["status"]["reasons"]
    assert r["status"]["work"]["output_items"] == 0
    assert r["branches"] == [] and r["points"] == []


def test_zero_postprocess_budget_skips_all_postassembly_scans():
    """Certified overlaps may survive, but junction/filter scans may not run."""
    s1, s2 = _shared_edge_pair(False)
    result = bez_ssx(
        s1, s2, 1e-3, rational=False,
        max_postprocess_work=0,
    )

    work = result["status"]["work"]
    assert result["complete"] is False
    assert "postprocess_cap" in result["status"]["reasons"]
    assert work["postprocess_work"] == 0
    assert result["singularities"] == []
    assert result["points"] == []
    assert result["branches"]
    assert all(branch.kind == "overlap" for branch in result["branches"])


def test_bez_ssx_depth_cap_reports_unresolved_partial_result():
    """A depth limit is a soft stop, not a proof that an empty cell is done."""
    from examples.ssx.bez_ssx5_coverage_check import load_case_surfaces

    s1, s2, rational = load_case_surfaces(7)
    r = bez_ssx(
        s1, s2, 1e-3, rational=rational,
        max_depth=0, max_cells=20_000, max_csx_calls=100,
    )
    assert r["complete"] is False
    assert "depth_limit" in r["status"]["reasons"]


def test_bez_ssx_surfaces_c1_local_truncation(monkeypatch):
    """A locally capped C1 enumeration must not masquerade as complete."""
    import mmcore.numeric.intersection.ssx._ssx5_singular as singular

    s1 = np.array([[[0., 0., 0.], [0., 1., 0.]],
                   [[1., 0., 0.], [1., 1., 0.]]])
    s2 = np.array([[[0., 0., -1.], [0., 1., 1.]],
                   [[1., 0., -1.], [1., 1., 1.]]])

    def fake_c1(*_args, stats=None, **_kwargs):
        stats.update(budget_exhausted=True,
                     external_budget_exhausted=False,
                     incomplete=False)
        return [], False

    monkeypatch.setattr(singular, "c1_pass", fake_c1)
    r = bez_ssx(s1, s2, 1e-3, rational=False)
    assert r["complete"] is False
    assert "work_budget" in r["status"]["reasons"]


def test_phi_seed_local_truncation_marks_shared_budget(monkeypatch):
    """Missing a capped Phi slice can omit a whole closed component."""
    from types import SimpleNamespace
    import mmcore.numeric.intersection.ssx._bez_ssx5 as ssx
    import mmcore.numeric.intersection.ssx._ssx5_singular as singular

    surface = _homog(np.array(
        [[[0., 0., 0.], [0., 1., 0.]],
         [[1., 0., 0.], [1., 1., 0.]]]))
    budget = ssx._SSXSoftBudget(max_cells=100, max_csx_calls=10)
    zero = np.zeros((1, 1, 1, 1), dtype=np.float64)
    cell = SimpleNamespace(
        g1=SimpleNamespace(surface=surface),
        g2=SimpleNamespace(surface=surface),
        T1=zero, T2=zero, T3=zero, T4=zero,
        work_budget=budget,
    )

    monkeypatch.setattr(
        ssx, "_choose_phi_equations",
        lambda *_a, **_k: [((0, 1), 0)])

    def fake_phi(*_args, stats=None, **_kwargs):
        if stats is not None:
            stats.update(budget_exhausted=True,
                         external_budget_exhausted=False)
        return []

    monkeypatch.setattr(singular, "phi_loop_seeds", fake_phi)
    assert ssx._phi_slice_loop_fragments(
        cell, [np.full(4, 0.5)], 1e-3, 0.1, []) == []
    assert budget.incomplete is True and budget.exhausted is False
    assert budget.result_fields()["complete"] is False
    assert "unresolved_tangential_zone" in budget.reasons


def test_deflate_tangent_cell_does_not_march_after_shared_budget_exhaustion(
        monkeypatch):
    from types import SimpleNamespace
    import mmcore.numeric.intersection.ssx._bez_ssx5 as ssx5

    surface = _homog(np.array([
        [[0., 0., 0.], [0., 1., 0.]],
        [[1., 0., 0.], [1., 1., 0.]],
    ]))
    crossings = [
        ssx5.BoundaryPoint(np.zeros(4), np.zeros(3), (0, 0)),
        ssx5.BoundaryPoint(
            np.ones(4), np.array([1., 1., 0.]), (0, 1)),
    ]
    budget = ssx5._SSXSoftBudget(max_cells=0, max_csx_calls=1)
    cell = SimpleNamespace(
        g1=SimpleNamespace(surface=surface),
        g2=SimpleNamespace(surface=surface),
        work_budget=budget,
    )
    monkeypatch.setattr(
        ssx5, "_pair_crossings_for_tracing",
        lambda *_a, **_k: ([(0, 1)], []))

    def forbidden(*_args, **_kwargs):
        pytest.fail("a tangent marcher ran after hard budget exhaustion")

    monkeypatch.setattr(ssx5, "_march_intersection_curve", forbidden)
    monkeypatch.setattr(ssx5, "_march_phi_curve", forbidden)
    fragments, points = ssx5._deflate_tangent_cell(
        surface, surface, 0., 0., 0., 0., ((0., 1.),) * 4,
        crossings, 1e-3, cell=cell, originals=crossings)

    assert fragments == [] and points == []
    assert budget.exhausted and budget.incomplete
    assert budget.cell_counts.get("singular_trace", 0) == 0


def test_registration_trace_stops_before_march_when_shared_budget_is_empty(
        monkeypatch):
    from types import SimpleNamespace
    import mmcore.numeric.intersection.ssx._bez_ssx5 as ssx5

    surface = _homog(np.array([
        [[0., 0., 0.], [0., 1., 0.]],
        [[1., 0., 0.], [1., 1., 0.]],
    ]))
    crossings = [
        ssx5.BoundaryPoint(
            np.array([0., 0., 0., 0.]), np.array([0., 0., 0.]), (0, 0)),
        ssx5.BoundaryPoint(
            np.array([1., 1., 1., 1.]), np.array([1., 1., 0.]), (0, 1)),
    ]
    budget = ssx5._SSXSoftBudget(max_cells=0, max_csx_calls=1)
    cell = SimpleNamespace(
        g1=SimpleNamespace(surface=surface),
        g2=SimpleNamespace(surface=surface),
        crossings=crossings,
        box=((0., 1.),) * 4,
        work_budget=budget,
    )

    def forbidden(*_args, **_kwargs):
        pytest.fail("registration tracing ran after hard budget exhaustion")

    monkeypatch.setattr(ssx5, "_march_to_boundary", forbidden)
    fragments, points = ssx5._trace_cell_by_registrations(
        cell, atol=1e-3)

    assert fragments == [] and points == []
    assert budget.exhausted and budget.incomplete
    assert budget.cell_counts.get("branch_trace", 0) == 0


def test_fragment_dedup_stops_when_postprocess_budget_is_empty():
    import mmcore.numeric.intersection.ssx._bez_ssx5 as ssx5

    xyz = np.column_stack([
        np.linspace(0., 1., 1000), np.zeros(1000), np.zeros(1000)])
    stuv = np.column_stack([
        np.linspace(0., 1., 1000), np.zeros((1000, 3))])
    fragments = [
        ssx5._Fragment(None, None, stuv.copy(), xyz.copy()),
        ssx5._Fragment(None, None, stuv.copy(), xyz.copy()),
    ]
    budget = ssx5._SSXSoftBudget(max_cells=0, max_csx_calls=1)

    kept = ssx5._drop_duplicate_fragments(
        fragments, 1e-3, work_budget=budget)

    # Unknown containment is never used to delete a certified fragment.
    assert kept == fragments
    assert budget.exhausted and budget.incomplete
    assert budget.cell_counts.get("assembly", 0) == 0


def test_assembly_does_not_close_or_scan_pairs_after_budget_denial(monkeypatch):
    import mmcore.numeric.intersection.ssx._bez_ssx5 as ssx5

    stuv = np.array([
        [0.20, 0.20, 0.20, 0.20],
        [0.80, 0.20, 0.80, 0.20],
        [0.80, 0.80, 0.80, 0.80],
        [0.21, 0.21, 0.21, 0.21],
    ])
    xyz = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.01, 0.01, 0.0],
    ])
    fragment = ssx5._Fragment(None, None, stuv, xyz)
    surface = _homog(np.array([
        [[0., 0., 0.], [0., 1., 0.]],
        [[1., 0., 0.], [1., 1., 0.]],
    ]))
    budget = ssx5._SSXSoftBudget(max_cells=0, max_csx_calls=1)

    def forbidden(*_args, **_kwargs):
        pytest.fail("closing march ran without shared-budget allowance")

    monkeypatch.setattr(ssx5, "_march_intersection_curve", forbidden)
    branches = ssx5._assemble_fragments(
        [fragment], S1_full=surface, S2_full=surface,
        atol_full=1e-3, rational_full=True, work_budget=budget)

    assert len(branches) <= 1
    assert budget.exhausted and budget.incomplete
    assert budget.cell_counts.get("assembly", 0) == 0


def test_assembly_pair_join_stops_at_postprocess_budget():
    import mmcore.numeric.intersection.ssx._bez_ssx5 as ssx5

    fragments = []
    for i in range(40):
        x0 = 0.1 + 0.01 * i
        xyz = np.array([[x0, 0., 0.], [x0 + 0.02, 0., 0.]])
        stuv = np.array([
            [0.2 + 1e-4 * i] * 4,
            [0.3 + 1e-4 * i] * 4,
        ])
        fragments.append(ssx5._Fragment(None, None, stuv, xyz))
    budget = ssx5._SSXSoftBudget(
        max_cells=100, max_csx_calls=1, max_postprocess_work=8)

    branches = ssx5._assemble_fragments(
        fragments, atol_full=1e-3, unify_tol=None,
        work_budget=budget)

    assert branches
    assert budget.postprocess_work <= 8
    assert budget.postprocess_exhausted
    assert budget.result_fields()["complete"] is False


def test_phi_slice_closed_march_stops_on_external_budget_exhaustion(
        monkeypatch):
    from types import SimpleNamespace
    import mmcore.numeric.intersection.ssx._bez_ssx5 as ssx5
    import mmcore.numeric.intersection.ssx._ssx5_singular as singular

    surface = _homog(np.array([
        [[0., 0., 0.], [0., 1., 0.]],
        [[1., 0., 0.], [1., 1., 0.]],
    ]))
    budget = ssx5._SSXSoftBudget(max_cells=0, max_csx_calls=1)
    cell = SimpleNamespace(
        g1=SimpleNamespace(surface=surface),
        g2=SimpleNamespace(surface=surface),
        T1=0., T2=0., T3=0., T4=0., work_budget=budget,
    )
    monkeypatch.setattr(
        ssx5, "_choose_phi_equations",
        lambda *_a, **_k: [((0, 1), 0)])

    def exhausted_seed_search(
            *_args, charge_box=None, stats=None, **_kwargs):
        assert charge_box(1) is False
        stats.update(
            budget_exhausted=True, external_budget_exhausted=True)
        return [np.full(4, 0.25)]

    monkeypatch.setattr(
        singular, "phi_loop_seeds", exhausted_seed_search)

    def forbidden(*_args, **_kwargs):
        pytest.fail("continuation work ran after external budget denial")

    monkeypatch.setattr(ssx5, "_ssx_correct", forbidden)
    monkeypatch.setattr(ssx5, "_march_psi_closed", forbidden)
    monkeypatch.setattr(ssx5, "_march_phi_closed", forbidden)
    fragments = ssx5._phi_slice_loop_fragments(
        cell, [np.full(4, 0.25)], 1e-3, 0.1, [])

    assert fragments == []
    assert budget.exhausted and budget.incomplete


def test_tangent_marchers_report_zero_iterations_on_initial_failure(
        monkeypatch):
    import mmcore.numeric.intersection.ssx._bez_ssx5 as ssx5

    surface = _homog(np.array([
        [[0., 0., 0.], [0., 1., 0.]],
        [[1., 0., 0.], [1., 1., 0.]],
    ]))
    monkeypatch.setattr(
        ssx5, "_ssx_tangent_4d",
        lambda *_a, **_k: (None, None, None))
    stats = {"iterations": 99}
    path, _ = ssx5._march_intersection_curve(
        surface, surface, np.zeros(4), np.ones(4),
        max_points=7, stats=stats)
    assert len(path) == 1 and stats["iterations"] == 0

    monkeypatch.setattr(ssx5, "_jac_phi", lambda *_a, **_k: np.eye(4))
    stats = {"iterations": 99}
    path, _ = ssx5._march_phi_curve(
        surface, surface, np.zeros((1, 1, 1, 1, 1)), (0, 1),
        np.zeros(4), np.ones(4), max_points=7, stats=stats)
    assert len(path) == 1 and stats["iterations"] == 0


def test_delta_local_dimension_requires_rank_and_two_sided_continuation():
    from mmcore.numeric.intersection.ssx._bez_ssx5 import (
        _delta_root_local_dimension)

    class FakeGN:
        roundoff_scale = 1.0

        def __init__(self, jacobian, mode):
            self.jacobian = lambda _x: np.asarray(jacobian, dtype=float)
            self.mode = mode

        def __call__(self, seed):
            if self.mode == "line":
                return np.asarray(seed, dtype=float)
            return np.full(4, 0.5)

        @staticmethod
        def physical_residual(_x):
            return 0.0

        @staticmethod
        def residual(_x):
            return np.zeros(7)

    root = np.full(4, 0.5)
    ptol = np.full(4, 1e-3)
    full_rank = FakeGN(np.eye(4), "isolated")
    assert _delta_root_local_dimension(full_rank, root, ptol) is False

    rank3 = np.diag([1.0, 1.0, 1.0, 0.0])
    regular_curve = FakeGN(rank3, "line")
    charged = []
    assert _delta_root_local_dimension(
        regular_curve, root, ptol,
        charge_work=lambda n: charged.append(n) or True) is True
    assert charged == [24, 24]

    singular_isolated = FakeGN(rank3, "isolated")
    assert _delta_root_local_dimension(
        singular_isolated, root, ptol) is None
    assert _delta_root_local_dimension(
        regular_curve, root, ptol,
        charge_work=lambda _n: False) is None


def test_offcurve_delta_local_truncation_marks_shared_budget(monkeypatch):
    """Zero or one roots from a capped Delta solve are still only partial."""
    from types import SimpleNamespace
    import mmcore.numeric.intersection.ssx._bez_ssx5 as ssx
    import mmcore.numeric.intersection.ssx._ssx5_singular as singular

    surface = _homog(np.array(
        [[[0., 0., 0.], [0., 1., 0.]],
         [[1., 0., 0.], [1., 1., 0.]]]))
    budget = ssx._SSXSoftBudget(max_cells=100, max_csx_calls=10)
    zero = np.zeros((1, 1, 1, 1), dtype=np.float64)
    cell = SimpleNamespace(
        g1=SimpleNamespace(surface=surface),
        g2=SimpleNamespace(surface=surface),
        T1=zero, T2=zero, T3=zero, T4=zero,
        F_sq=None, w_scale=1.0,
        box=((0.0, 1.0),) * 4,
        work_budget=budget,
    )

    def fake_delta(*_args, **_kwargs):
        return (lambda _x: None), np.zeros((1, 1, 1, 1, 4))

    def fake_solve(*_args, **_kwargs):
        assert _kwargs["max_results"] == 64
        return [], True

    monkeypatch.setattr(ssx, "_delta_float_gn", fake_delta)
    monkeypatch.setattr(singular, "solve_zero_dim", fake_solve)
    ssx._emit_offcurve_tangent_roots(
        cell, [], 1e-3, np.full(4, 4e-3), [], max_cells=2)
    assert budget.incomplete is True and budget.exhausted is False
    assert budget.result_fields()["complete"] is False


def test_primary_delta_witness_local_truncation_marks_shared_budget(monkeypatch):
    """A capped one-root tangency witness is valid partial evidence only."""
    from types import SimpleNamespace
    import mmcore.numeric.intersection.ssx._bez_ssx5 as ssx

    surface = _homog(np.array(
        [[[0., 0., 0.], [0., 1., 0.]],
         [[1., 0., 0.], [1., 1., 0.]]]))
    budget = ssx._SSXSoftBudget(max_cells=100, max_csx_calls=10)
    zero = np.zeros((1, 1, 1, 1), dtype=np.float64)
    cell = SimpleNamespace(
        g1=SimpleNamespace(surface=surface),
        g2=SimpleNamespace(surface=surface),
        T1=zero, T2=zero, T3=zero, T4=zero,
        box=((0.0, 1.0),) * 4,
        work_budget=budget,
    )

    monkeypatch.setattr(
        ssx, "_tangency_witness",
        lambda *_a, **_k: (True, [np.full(4, 0.5)], None, True))
    ssx._emit_tangent_roots(
        cell, 1e-3, np.full(4, 4e-3), [], enumerate_all=True)
    assert budget.incomplete is True and budget.exhausted is False
    assert budget.result_fields()["complete"] is False


def test_solve_zero_dim_result_cap_stops_positive_dimensional_flood():
    net = BoxNet(
        np.zeros((2, 2, 2, 2, 1), dtype=np.float64),
        axes=(0, 1, 2, 3))
    stats = {}
    sols, exhausted = solve_zero_dim(
        [net], lambda x: np.asarray(x), ptol=np.full(4, 1e-6),
        max_cells=10_000, max_results=5, stats=stats)
    assert len(sols) == 5 and exhausted
    assert stats["result_limit_reached"] is True
    assert stats["boxes_processed"] < 100


def test_c3_pair_search_has_a_local_soft_budget():
    """The vectorized broadphase must not materialize an unbounded M^2 set."""
    from types import SimpleNamespace
    from mmcore.numeric.intersection.ssx._ssx5_singular import c3_pass

    surface = _homog(np.array(
        [[[0., 0., 0.], [0., 1., 0.]],
         [[1., 0., 0.], [1., 1., 0.]]]))
    xyz = np.column_stack([
        np.linspace(0.0, 1.0, 5),
        np.zeros(5),
        np.zeros(5),
    ])
    stuv = np.column_stack([
        np.linspace(0.0, 1.0, 5),
        np.zeros(5),
        np.linspace(0.0, 1.0, 5),
        np.zeros(5),
    ])
    branches = [SimpleNamespace(curve=(stuv, xyz)),
                SimpleNamespace(curve=(stuv, xyz))]
    stats = {}
    hits = c3_pass(
        surface, surface, branches, atol=1e-3, ptol4=np.full(4, 1e-3),
        max_work=3, stats=stats)
    assert hits == []
    assert stats["budget_exhausted"] is True
    assert stats["pairs_processed"] <= 3


def test_ssx_point_dedup_uses_bounded_spatial_buckets():
    """Final duplicate cleanup must scale with points, not all point pairs."""
    from mmcore.numeric.intersection.ssx._ssx4 import SSXPoint
    from mmcore.numeric.intersection.ssx._bez_ssx5 import (
        _deduplicate_ssx_points)

    points = []
    for i in range(200):
        stuv = np.array([10.0 * i, 0.0, 0.0, 0.0])
        xyz = np.array([10.0 * i, 0.0, 0.0])
        points.append(SSXPoint(stuv=stuv, xyz=xyz))
        points.append(SSXPoint(
            stuv=stuv + np.array([5e-4, 0.0, 0.0, 0.0]),
            xyz=xyz + np.array([5e-4, 0.0, 0.0])))

    stats = {}
    unique = _deduplicate_ssx_points(
        points, unify_tol=np.full(4, 1e-3), atol=1e-3, stats=stats)
    assert len(unique) == 200
    assert stats["comparisons"] < 4 * len(points)
    assert stats["bucket_probes"] < 120 * len(points)


def test_solve_zero_dim_shared_budget_denial_stops_before_box():
    net = BoxNet(
        np.zeros((2, 2, 2, 2, 1), dtype=np.float64),
        axes=(0, 1, 2, 3))
    calls = {"charge": 0, "newton": 0}

    def charge_box(units=1):
        calls["charge"] += units
        return calls["charge"] <= 3

    def newton(_x0):
        calls["newton"] += 1
        return None

    stats = {}
    sols, exhausted = solve_zero_dim(
        [net], newton, ptol=np.full(4, 1e-3), max_cells=100,
        charge_box=charge_box, stats=stats)
    assert sols == [] and exhausted
    assert calls == {"charge": 4, "newton": 3}
    assert stats["boxes_processed"] == 3
    assert stats["external_budget_exhausted"]


def test_c1_pass_fair_shares_one_budget_across_surfaces(monkeypatch):
    import mmcore.numeric.intersection.ssx._ssx5_singular as singular

    surface = _homog(np.array(
        [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
         [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]))
    normal = np.empty((2, 2, 3), dtype=np.float64)
    normal[..., 0] = [[-1.0, 1.0], [-1.0, 1.0]]
    normal[..., 1] = [[1.0, -1.0], [1.0, -1.0]]
    normal[..., 2] = [[-1.0, -1.0], [1.0, 1.0]]
    monkeypatch.setattr(singular, "sigma_normal_net", lambda *_a, **_k: normal)

    limits = []
    sentinel_charge = lambda _units=1: True

    def fake_solve(*_args, max_cells, charge_box=None, stats=None, **_kwargs):
        assert charge_box is sentinel_charge
        assert _kwargs["max_results"] == 64
        limits.append(max_cells)
        used = min(3, max_cells)
        stats.update(cells_processed=used, boxes_processed=used,
                     budget_exhausted=False,
                     external_budget_exhausted=False)
        return [], False

    monkeypatch.setattr(singular, "solve_zero_dim", fake_solve)
    stats = {}
    hits, curve_flag = singular.c1_pass(
        surface, surface, atol=1e-3, ptol4=np.full(4, 1e-3),
        max_cells=5, charge_box=sentinel_charge, stats=stats)
    assert hits == [] and not curve_flag
    assert limits == [2, 3] and sum(limits) == 5
    assert stats["cells_processed"] == 5


def test_c1_connectivity_probe_respects_shared_budget(monkeypatch):
    import mmcore.numeric.intersection.ssx._ssx5_singular as singular

    surface = _homog(np.array(
        [[[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
         [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]]]))
    normal = np.empty((2, 2, 3), dtype=np.float64)
    normal[..., 0] = [[-1.0, 1.0], [-1.0, 1.0]]
    normal[..., 1] = [[1.0, -1.0], [1.0, -1.0]]
    normal[..., 2] = [[-1.0, -1.0], [1.0, 1.0]]
    monkeypatch.setattr(
        singular, "sigma_normal_net", lambda *_a, **_k: normal)

    sols = [np.full(4, 0.25), np.full(4, 0.75)]

    def fake_solve(*_args, stats=None, **_kwargs):
        stats.update(
            cells_processed=1, boxes_processed=1,
            budget_exhausted=False,
            external_budget_exhausted=False)
        return list(sols), False

    def fake_connected(_sols, newton, _ptol):
        assert newton(np.full(4, 0.5)) is None
        return False

    charges = []
    monkeypatch.setattr(singular, "solve_zero_dim", fake_solve)
    monkeypatch.setattr(singular, "_connected_one_dim", fake_connected)
    stats = {}
    hits, curve_flag = singular.c1_pass(
        surface, surface, atol=1e-3, ptol4=np.full(4, 1e-3),
        max_cells=100,
        charge_box=lambda n=1: charges.append(n) or False,
        stats=stats)

    assert hits == [] and curve_flag is False
    assert charges == [1]
    assert stats["external_budget_exhausted"] is True
    assert stats["incomplete"] is True
    assert stats["solve_calls"] == 1


def test_subdivided_gauss_cone_contains_true_surface_normal():
    from mmcore.numeric.intersection.ssx._ssx4 import (
        GaussMapBern, separate_gauss_maps)

    points = np.array([
        [[0.0, 0.0, 0.12590804260840438],
         [0.0, 0.5, 0.4045619445326789],
         [0.0, 1.0, 0.43568455289279767]],
        [[0.5, 0.0, 0.41487198131264286],
         [0.5, 0.5, 0.32849583092501017],
         [0.5, 1.0, 0.3231481813148787]],
        [[1.0, 0.0, 0.6622262078328403],
         [1.0, 0.5, -0.5856500620605981],
         [1.0, 1.0, 0.6529671930891621]],
    ])
    surface = _homog(points)
    left = GaussMapBern.from_surf(surface, rational=True).split_u(0.5)[0]
    _, du, dv = eval_surface_d1(surface, 0.485, 0.01, rational=True)
    normal = np.cross(du, dv)
    normal /= np.linalg.norm(normal)
    w_left, w_sample = separate_gauss_maps(
        left.map_dirs(), np.repeat(normal[None, :], 4, axis=0))
    assert w_left is None or w_sample is None


def test_coverage_harness_preserves_case_rational_flag():
    from examples.ssx.bez_ssx5_coverage_check import load_case_surfaces

    p1, p2, polynomial_rational = load_case_surfaces(5)
    r1, r2, rational = load_case_surfaces(13)
    assert polynomial_rational is False and p1.shape[-1] == p2.shape[-1] == 3
    assert rational is True and r1.shape[-1] == r2.shape[-1] == 4


def test_coverage_check_propagates_rational_flag(monkeypatch):
    import examples.ssx.bez_ssx5_coverage_check as coverage

    surface = _homog(np.array(
        [[[0., 0., 0.], [0., 1., 0.]],
         [[1., 0., 0.], [1., 1., 0.]]]))
    seen = {}
    monkeypatch.setattr(
        coverage, "load_case_surfaces",
        lambda _case: (surface, surface, True))

    def fake_ssx(*_args, rational, **_kwargs):
        seen["ssx"] = rational
        return {"branches": [], "points": [], "singularities": [],
                "budget_exhausted": False}

    def fake_reference(*_args, rational, **_kwargs):
        seen["reference"] = rational
        return np.empty((0, 3)), np.empty(0)

    monkeypatch.setattr(coverage, "bez_ssx", fake_ssx)
    monkeypatch.setattr(coverage, "reference_cloud", fake_reference)
    assert coverage.check_case(999, n_ref=1) is True
    assert seen == {"ssx": True, "reference": True}


def test_coverage_check_rejects_partial_ssx(monkeypatch):
    import examples.ssx.bez_ssx5_coverage_check as coverage

    surface = np.array(
        [[[0., 0., 0.], [0., 1., 0.]],
         [[1., 0., 0.], [1., 1., 0.]]])
    monkeypatch.setattr(
        coverage, "load_case_surfaces",
        lambda _case: (surface, surface, False))
    monkeypatch.setattr(
        coverage, "bez_ssx",
        lambda *_a, **_k: {
            "branches": [], "points": [], "singularities": [],
            "complete": False,
            "status": {"reasons": ["work_budget"], "work": {}},
        })
    monkeypatch.setattr(
        coverage, "reference_cloud",
        lambda *_a, **_k: pytest.fail("partial SSX reached reference oracle"))
    assert coverage.check_case(999, n_ref=1) is False


def test_reference_cloud_rejects_partial_csx(monkeypatch):
    import examples.ssx.bez_ssx5_coverage_check as coverage

    surface = np.array(
        [[[0., 0., 0.], [0., 1., 0.]],
         [[1., 0., 0.], [1., 1., 0.]]])
    monkeypatch.setattr(
        coverage, "bez_csx",
        lambda *_a, **_k: {
            "isolated": [], "overlaps": [],
            "budget_exhausted": True,
            "boundary_topology_complete": False,
        })
    with pytest.raises(RuntimeError, match="reference CSX incomplete"):
        coverage.reference_cloud(surface, surface, n=1, rational=False)


# ---------------------------------------------------------------------------
# L41 — schema v2: result carries complete / status.reasons / status.work
# (review doc 2026-07-12 §6; replaces budget_exhausted / budget_usage)
# ---------------------------------------------------------------------------

def _transversal_planes():
    s1 = np.array([[[0., 0., 0.], [0., 1., 0.]],
                   [[1., 0., 0.], [1., 1., 0.]]])
    s2 = np.array([[[0., 0., -1.], [0., 1., 1.]],
                   [[1., 0., -1.], [1., 1., 1.]]])
    return s1, s2


def _assert_schema_v2(r):
    assert isinstance(r["complete"], bool)
    status = r["status"]
    assert isinstance(status["reasons"], list)
    work = status["work"]
    for key in ("cells_processed", "csx_calls", "max_cells", "max_csx_calls",
                "output_items", "max_output_items", "postprocess_work",
                "max_postprocess_work", "cell_counts"):
        assert key in work, key
    # The one consumer contract: complete iff no reasons.
    assert r["complete"] == (not status["reasons"]), (
        r["complete"], status["reasons"])
    # Schema v2 REPLACES the old flags (rename, not duplication).
    assert "budget_exhausted" not in r
    assert "budget_usage" not in r
    # The unread publications are gone (ledger L41).
    assert "hard_exhausted" not in work
    assert "output_counts" not in work


def test_schema_v2_complete_run_has_empty_reasons():
    s1, s2 = _transversal_planes()
    r = bez_ssx(s1, s2, 1e-3, rational=False)
    _assert_schema_v2(r)
    assert r["complete"] is True
    assert r["status"]["reasons"] == []
    assert len(r["branches"]) == 1


def test_schema_v2_curve_only_overlap_stays_complete():
    # L27's shared-edge pair: the coincidence set is 1-D (two edges), fully
    # represented by overlap branches + the junction tangent_point — the
    # region schema is NOT needed and the result must claim completeness.
    s1, s2 = _shared_edge_pair(False)
    r = bez_ssx(s1, s2, 1e-3, rational=False)
    _assert_schema_v2(r)
    assert r["complete"] is True, r["status"]["reasons"]


def test_schema_v2_zero_allowance_reports_work_budget():
    s1, s2 = _transversal_planes()
    r = bez_ssx(s1, s2, 1e-3, rational=False, max_cells=0)
    _assert_schema_v2(r)
    assert r["complete"] is False
    assert r["status"]["reasons"] == ["work_budget"]
    assert r["status"]["work"]["cells_processed"] == 0


def test_schema_v2_depth_ceiling_reports_depth_limit():
    from examples.ssx.bez_ssx5_coverage_check import load_case_surfaces

    s1, s2, rational = load_case_surfaces(7)
    r = bez_ssx(s1, s2, 1e-3, rational=rational,
                max_depth=0, max_cells=20_000, max_csx_calls=100)
    _assert_schema_v2(r)
    assert r["complete"] is False
    assert "depth_limit" in r["status"]["reasons"]


def test_schema_v2_2d_overlap_reports_region_unsupported():
    # Case-12 class (L28 pending): coplanar bilinear patches overlapping in
    # a 2-D region. The rank-deficient Δ-root inside the detected overlap
    # region's parametric box is the region's structural signature until
    # SSXOverlapRegion lands; the old schema billed this to the budget at
    # 1.9% usage (review doc §5 probe 1, case 12).
    r = bez_ssx(_plane_patch(0, 2), _plane_patch(1, 3), 1e-3, rational=False)
    _assert_schema_v2(r)
    assert r["complete"] is False
    assert "overlap_region_unsupported" in r["status"]["reasons"]


def test_schema_v2_collapsed_edge_reports_parameter_fiber():
    # Collapsed v=0 edge (a point-curve) lying ON the plane: boundary CSX
    # types it as a parameter fiber (case-14 class at unit-test size); the
    # limiting branch multiplicity is unproven, so the result is partial
    # for the structural reason, not a budget one.
    s1 = np.array([[[0., 0., 0.], [0., 0., 0.]],
                   [[0., 1., 1.], [1., 1., 1.]]])
    s2 = np.array([[[-1., -1., 0.], [-1., 1., 0.]],
                   [[1., -1., 0.], [1., 1., 0.]]])
    r = bez_ssx(s1, s2, 1e-3, rational=False)
    _assert_schema_v2(r)
    assert r["complete"] is False
    assert "parameter_fiber" in r["status"]["reasons"]


def test_zero_csx_allowance_knobs_are_honored(monkeypatch):
    # Ledger L41 / review finding 11: csx_max_cells=0,
    # boundary_csx_max_cells=0 and csx_max_results=0 were silently promoted
    # to 1 via max(1, ...) while max_cells=0/max_csx_calls=0 were honored —
    # a zero allowance is a hard promise that bez_csx never runs.
    import mmcore.numeric.intersection.ssx._bez_ssx5 as ssx5

    s1, s2 = _transversal_planes()
    calls = []
    real_bez_csx = ssx5.bez_csx

    def recording(*args, **kwargs):
        calls.append(int(kwargs.get("max_cells", -1)))
        return real_bez_csx(*args, **kwargs)

    monkeypatch.setattr(ssx5, "bez_csx", recording)

    r = ssx5.bez_ssx(s1, s2, 1e-3, rational=False,
                     boundary_csx_max_cells=0, csx_max_cells=0)
    assert calls == [], "bez_csx ran despite a zero per-call cell allowance"
    assert r["complete"] is False
    assert "work_budget" in r["status"]["reasons"]

    calls.clear()
    r = ssx5.bez_ssx(s1, s2, 1e-3, rational=False, csx_max_results=0)
    assert calls == [], "bez_csx ran despite a zero result allowance"
    assert r["complete"] is False
    assert "work_budget" in r["status"]["reasons"]

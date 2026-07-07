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


def test_self_intersection_point():
    S1, S2 = _umbrella_case()
    r = bez_ssx(S1, S2, 1e-3, rational=False)
    c3 = [g for g in r["singularities"] if g.kind == "self_intersection"]
    assert len(c3) == 1
    g = c3[0]
    assert np.allclose(g.xyz, [0.0, 0.0, 0.5], atol=2e-3)
    assert g.stuv_mate is not None
    # the two preimages differ in (s,t) but share xyz
    assert abs(g.stuv[1] - g.stuv_mate[1]) > 0.2      # t = (1±0.707)/2 differ by ~0.707
    assert len({l[0] for l in g.branch_links}) >= 1   # linked to branch(es)


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
    assert len(samples) >= 13         # the >12-sols curve_flag path fired
    # every sample on the true singular curve {s = 0.5}, covering most of t
    assert np.allclose(samples[:, 0], 0.5, atol=1e-6)
    assert samples[:, 1].max() - samples[:, 1].min() > 0.8


def test_theorem3_skips_regular_case():
    # transversal bilinear pair. NOTE (measured): the Theorem-3 gate does
    # NOT certify here — the whole curve traces from the TOP cell, whose
    # T1/T2 hulls touch zero at the domain edge, so the (sound, strict)
    # hull test fails and c3_pass DOES run. The guarantee this test pins is
    # the one that matters: the vectorized AABB broadphase makes that run
    # nearly free and it finds nothing — zero spurious self-intersections.
    s1 = np.array([[[0., 0., 0.], [0., 10., 0.]], [[10., 0., 0.], [10., 10., 10.]]])
    s2 = np.array([[[0., 0., 3.], [0., 10., 3.]], [[10., 0., 3.], [10., 10., 3.]]])
    r = bez_ssx(s1, s2, 1e-3, rational=False)
    assert [g for g in r["singularities"] if g.kind == "self_intersection"] == []

"""Singularity algebra for bez_ssx v5 (Cheng et al. 2023, C1/C2/C3).

Bernstein nets for the SSI singularity systems and a budget-bounded
zero-dimensional subdivision solver (hull exclusion + Newton — the proven
CSX phase-2 pattern lifted to 4D). Nets may depend on a SUBSET of the 4 axes
(e.g. Sigma_1 depends only on (s,t)); `BoxNet.axes` records the mapping
from the net's own dims to global axes so restriction skips foreign axes.

One responsibility: singularity algebra (net construction + zero-dimensional
solving). No tracing code lives here — `_bez_ssx5.py` owns the
marching/assembly that consumes these primitives.
"""
from __future__ import annotations

from dataclasses import dataclass
from math import comb
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from mmcore.numeric.bern import de_casteljau_split_nd, bernstein_partial_derivative_coeffs
from mmcore.numeric.intersection._deflate import (
    bernstein_patch_derivative_s,
    bernstein_patch_derivative_t,
    bernstein_patch_cross_same_params,
)


def psi_vector_net(S1_h: np.ndarray, S2_h: np.ndarray) -> np.ndarray:
    """Bernstein net of Psi = C1(s,t)*W2(u,v) - C2(u,v)*W1(s,t), shape (m1,n1,m2,n2,3).

    S1_h, S2_h are homogeneous control nets in the mmcore convention
    (trailing column = weight, leading D columns = weight*point); P1/w1 and
    P2/w2 below are the COEFFICIENT arrays of the weighted-numerator
    polynomials C1(s,t)/C2(u,v) and of the weight polynomials
    W1(s,t)/W2(u,v). (s,t) and (u,v) are disjoint variables, so the
    products are outer products — exact, no degree elevation (same trick
    as the CSX residual/G-net).

    This net's zero set coincides with R1(s,t) == R2(u,v) (the true
    rational-surface intersection condition) because W1, W2 > 0 everywhere
    for valid weights — clearing the denominators only rescales by a
    positive factor. It is intended for hull-exclusion pruning only; Newton
    refinement must use the smooth rational evaluators directly (see
    `solve_zero_dim`), never this net's raw value as a metric residual.
    """
    P1, w1 = S1_h[..., :-1], S1_h[..., -1]
    P2, w2 = S2_h[..., :-1], S2_h[..., -1]
    return (P1[:, :, None, None, :] * w2[None, None, :, :, None]
            - P2[None, None, :, :, :] * w1[:, :, None, None, None])


def linear_net_4d(c0: float, coeffs: Sequence[float]) -> np.ndarray:
    """Degree-(1,1,1,1) Bernstein net of L(x) = c0 + coeffs.x on [0,1]^4."""
    L = np.empty((2, 2, 2, 2, 1), dtype=np.float64)
    for idx in np.ndindex(2, 2, 2, 2):
        L[idx] = c0 + sum(coeffs[i] * idx[i] for i in range(4))
    return L


def _same_param_product_vec_scalar(V: np.ndarray, s: np.ndarray) -> np.ndarray:
    """Exact same-parameter Bernstein product of a vector patch and a scalar patch.

    V : vector patch, shape (mu+1, mv+1, 3), degree (mu, mv).
    s : scalar patch, shape (nu+1, nv+1), degree (nu, nv).

    Returns a vector patch of shape (mu+nu+1, mv+nv+1, 3) — the same
    convolution identity `bernstein_patch_cross_same_params` uses for the
    cross product (§ same (s,t), possibly different degrees, no separate
    degree-elevation step needed), specialized to elementwise vector*scalar
    multiplication. This is the EXACT Bernstein product; pointwise
    coefficient multiplication would be wrong (Bernstein coefficients are
    not closed under pointwise multiply).
    """
    m_u, m_v = V.shape[0] - 1, V.shape[1] - 1
    n_u, n_v = s.shape[0] - 1, s.shape[1] - 1
    deg_u, deg_v = m_u + n_u, m_v + n_v
    out = np.zeros((deg_u + 1, deg_v + 1, 3), dtype=np.float64)
    for alpha in range(deg_u + 1):
        denom_u = comb(deg_u, alpha)
        i_lo, i_hi = max(0, alpha - n_u), min(m_u, alpha)
        for beta in range(deg_v + 1):
            denom_v = comb(deg_v, beta)
            j_lo, j_hi = max(0, beta - n_v), min(m_v, beta)
            acc = np.zeros(3, dtype=np.float64)
            for i in range(i_lo, i_hi + 1):
                k = alpha - i
                cu = comb(m_u, i) * comb(n_u, k) / denom_u
                for j in range(j_lo, j_hi + 1):
                    l = beta - j
                    cv = comb(m_v, j) * comb(n_v, l) / denom_v
                    acc += (cu * cv) * V[i, j] * s[k, l]
            out[alpha, beta] = acc
    return out


def sigma_normal_net(S: np.ndarray, rational: bool) -> np.ndarray:
    """Bernstein net (2D, over the surface's own (a,b) params) of the
    surface-normal numerator N(a,b), trailing value dim 3.

    Polynomial case (`rational=False`): N = dP/da x dP/db as an exact
    Bernstein net (via `_deflate.py`'s same-parameter derivative/cross
    primitives). Accepts EITHER a Cartesian (m,n,3) array OR a homogeneous
    (m,n,4) array whose weights are all == 1 (the weight column is then
    stripped after verifying it is 1). A homogeneous array with non-unit
    weights raises ValueError — a genuinely rational net must go through
    `rational=True`, never be silently treated as polynomial.

    Rational case (`rational=True`): requires a homogeneous (m,n,4) input.
    Let P = weighted numerator net, w = weight net (S[...,:-1], S[...,-1]
    in the mmcore convention, so the true surface is R = P/w). By the
    quotient rule,
        R_a = (P_a*w - P*w_a) / w^2,   R_b = (P_b*w - P*w_b) / w^2,
    so
        R_a x R_b = [(P_a*w - P*w_a) x (P_b*w - P*w_b)] / w^4.
    Since w > 0 everywhere for valid weights, the zero set of R_a x R_b
    coincides exactly with the zero set of the NUMERATOR
        N_hom = (P_a*w - P*w_a) x (P_b*w - P*w_b),
    which is a polynomial (Bernstein-representable) vector field — this is
    what this branch returns. Built from exact same-parameter Bernstein
    products (`_same_param_product_vec_scalar` for vector*scalar,
    `bernstein_patch_cross_same_params` for the final cross product); both
    terms of each factor come out to identical degree automatically (the
    same-parameter product's result degree is the sum of the operand
    degrees, which is order-independent), so no separate degree-matching
    step is required.
    """
    S = np.asarray(S, dtype=np.float64)
    if S.ndim != 3 or S.shape[0] < 2 or S.shape[1] < 2:
        raise ValueError(
            "sigma_normal_net requires a (m+1, n+1, 3|4) control net with degree >= 1 in "
            f"BOTH parametric directions (got shape {S.shape}); a degree-0 direction has an "
            "identically-zero partial derivative — the normal net would be identically zero — "
            "and the underlying Bernstein product primitives fail on empty derivative nets."
        )
    if not rational:
        if S.shape[-1] == 4:
            if not np.allclose(S[..., -1], 1.0, rtol=0.0, atol=1e-12):
                raise ValueError(
                    "sigma_normal_net(rational=False) received a homogeneous net with "
                    "non-unit weights; use rational=True for a genuinely rational surface."
                )
            P = S[..., :-1]
        elif S.shape[-1] == 3:
            P = S
        else:
            raise ValueError(f"expected a trailing dim of 3 or 4, got {S.shape[-1]}")
        Pa = bernstein_patch_derivative_s(P.tolist())
        Pb = bernstein_patch_derivative_t(P.tolist())
        return np.asarray(bernstein_patch_cross_same_params(Pa, Pb), dtype=np.float64)

    if S.shape[-1] != 4:
        raise ValueError("sigma_normal_net(rational=True) requires a homogeneous (m,n,4) net")
    P, w = S[..., :-1], S[..., -1]
    Pa = bernstein_partial_derivative_coeffs(P, axis=0)
    Pb = bernstein_partial_derivative_coeffs(P, axis=1)
    wa = bernstein_partial_derivative_coeffs(w[..., None], axis=0)[..., 0]
    wb = bernstein_partial_derivative_coeffs(w[..., None], axis=1)[..., 0]
    A = _same_param_product_vec_scalar(Pa, w) - _same_param_product_vec_scalar(P, wa)
    B = _same_param_product_vec_scalar(Pb, w) - _same_param_product_vec_scalar(P, wb)
    return np.asarray(bernstein_patch_cross_same_params(A.tolist(), B.tolist()), dtype=np.float64)


@dataclass(frozen=True, eq=False)
class BoxNet:
    """A scalar Bernstein net over a sub-box of [0,1]^4.

    `coeffs` has one tensor dim per entry of `axes` plus a trailing value
    dim of size 1. `axes[i]` is the global axis the i-th tensor dim varies
    along; restriction along a global axis not in `axes` is a no-op (the
    net is constant along axes it doesn't depend on).

    Frozen: the foreign-axis no-op split returns the SAME instance for both
    children, so instances are shared across sibling boxes and must never
    be mutated. (`eq=False` keeps identity equality — the dataclass-generated
    `__eq__` would raise on ndarray fields.)
    """
    coeffs: NDArray[np.float64]
    axes: tuple[int, ...]

    def excludes_zero(self) -> bool:
        """Bernstein hull test: True proves the net has no zero over its box.

        NaN coefficients make both comparisons False — fail-open (never
        excludes), the safe direction."""
        c = self.coeffs
        return float(c.min()) > 0.0 or float(c.max()) < 0.0

    def split(self, global_axis: int, t: float = 0.5):
        if global_axis not in self.axes:
            return self, self
        d = self.axes.index(global_axis)
        lo, hi = de_casteljau_split_nd(self.coeffs, axis=d, t=t)
        return BoxNet(lo, self.axes), BoxNet(hi, self.axes)


@dataclass(frozen=True, eq=False)
class ShiftedPositiveNet(BoxNet):
    """One-sided exclusion net: excludes iff min(coeffs) > 0.

    For SHIFTED squared-distance nets (coeffs = F_sq − thresh): `min > 0`
    proves the box is entirely OUTSIDE the tolerance shell (no Ψ-zero —
    excludable), but `max < 0` only proves it is entirely INSIDE the shell
    (i.e. ON the intersection at tolerance — the opposite of excludable),
    so the base class's two-sided hull test would wrongly prune exactly
    the boxes that matter. Splits preserve the subclass.
    """

    def excludes_zero(self) -> bool:
        c = self.coeffs
        return float(c.min()) > 0.0

    def split(self, global_axis: int, t: float = 0.5):
        if global_axis not in self.axes:
            return self, self
        d = self.axes.index(global_axis)
        lo, hi = de_casteljau_split_nd(self.coeffs, axis=d, t=t)
        return ShiftedPositiveNet(lo, self.axes), ShiftedPositiveNet(hi, self.axes)


def solve_zero_dim(
    nets: list,                       # non-empty list[BoxNet] — ALL must contain 0 for a box to survive
    newton: Callable,                 # (x0: (4,)) -> Optional[(4,) solution in GLOBAL coords]
    ptol,                             # (4,) per-axis parametric resolution
    box=((0.0, 1.0),) * 4,
    max_cells: int = 20000,
    dedup_xyz: Optional[Callable] = None,   # (sol) -> (3,) point, for xyz dedup
    atol: float = 1e-3,
    skip_newton: Optional[Callable] = None,  # (box) -> True to skip the Newton attempt
    priority: Optional[Callable] = None,     # (box) -> float; HIGHER pops first (heap)
):
    """All isolated solutions of {net_i = 0} in `box`.

    Returns
    -------
    (sols, exhausted) : tuple[list, bool]
        `sols` — list of (4,) solutions found. `exhausted` — True iff the
        `max_cells` budget ran out with boxes still pending, i.e. the
        enumeration may be INCOMPLETE and `sols` is only a lower bound.
        Callers must check it (a silently-truncated list is
        indistinguishable from a complete one otherwise). Never raise
        `max_cells` to chase `exhausted=False` on a hang — a blown budget
        usually means the solution set isn't 0-dimensional and callers
        must handle that case themselves (e.g. a curve_flag path).

    Hull-exclusion subdivision + center-seeded Newton. Newton runs in
    GLOBAL coordinates on smooth evaluators; the nets are only used for
    hull-exclusion pruning within each box. The same root can therefore be
    (re-)found from several surviving boxes — `_dup` handles it (the
    CSX-proven destructive-dedup pattern: 1·ptol per-axis box AND, if
    `dedup_xyz` is given, xyz <= atol).

    The split axis is the one with the largest span measured in units of
    its OWN ptol (span_i / ptol_i), and the resolution floor fires only
    when that max ratio is <= 1 (every axis at/below its ptol). With
    heterogeneous per-axis ptols a plain widest-axis rule could stop while
    a tighter-ptol axis is still under-refined by orders of magnitude; for
    uniform ptol this is behavior-identical to widest-axis.

    Raises ValueError on empty `nets`: with nothing to prune, every box
    survives and the search silently degrades to an exhaustive Newton
    multistart burning the whole `max_cells` budget.

    `skip_newton(box)` suppresses the Newton attempt on a box WITHOUT
    affecting its subdivision — for callers that can prove any root inside
    a region is already represented elsewhere (e.g. within the tube of an
    already-traced tangent curve, whose emissions the downstream
    subsumption filter would delete anyway). `priority(box)` switches the
    pending set from LIFO to a max-heap ordered by the callable — callers
    facing a 1-dimensional solution component use it to explore boxes FAR
    from the known component first, so a `max_cells` exhaustion eats only
    the (skippable) component flood, not undiscovered isolated roots.
    """
    if not nets:
        raise ValueError(
            "solve_zero_dim: `nets` must be non-empty — with nothing to prune, the search "
            "degrades to an exhaustive Newton multistart over the whole budget."
        )
    ptol = np.asarray(ptol, dtype=np.float64)
    sols: list = []

    def _dup(x):
        for s in sols:
            if np.all(np.abs(np.asarray(s) - x) <= ptol):
                if dedup_xyz is None:
                    return True
                if float(np.linalg.norm(dedup_xyz(s) - dedup_xyz(x))) <= atol:
                    return True
        return False

    if priority is None:
        pending = [(tuple(box), list(nets))]
        _push = pending.append
        _pop = pending.pop
    else:
        import heapq
        _tie = iter(range(1 << 62))          # heap tiebreak — boxes aren't comparable
        pending = [(-float(priority(tuple(box))), next(_tie), tuple(box), list(nets))]

        def _push(item):
            heapq.heappush(pending, (-float(priority(item[0])), next(_tie), item[0], item[1]))

        def _pop():
            e = heapq.heappop(pending)
            return e[2], e[3]

    cells = 0
    while pending and cells < max_cells:
        cells += 1
        bx, bnets = _pop()
        if any(n.excludes_zero() for n in bnets):
            continue
        if skip_newton is None or not skip_newton(bx):
            mid = np.array([0.5 * (lo + hi) for lo, hi in bx])
            sol = newton(mid)
            if sol is not None:
                sol = np.asarray(sol, dtype=np.float64)
                inside = all(bx[i][0] - 1e-12 <= sol[i] <= bx[i][1] + 1e-12 for i in range(4))
                if inside and not _dup(sol):
                    sols.append(sol)
        # split the axis with the largest span in units of its OWN ptol —
        # with heterogeneous per-axis ptols the absolutely-widest axis can
        # already be resolved while a tighter-ptol axis is still orders of
        # magnitude above its floor
        ratios = [(hi - lo) / float(ptol[i]) for i, (lo, hi) in enumerate(bx)]
        widest = int(np.argmax(ratios))
        if ratios[widest] <= 1.0:
            continue      # resolution floor: every axis at/below its ptol
        # nets and box split in lockstep so the net's local 0.5 is exactly
        # the box's global midpoint
        left_nets, right_nets = [], []
        for n in bnets:
            l, r = n.split(widest, 0.5)
            left_nets.append(l); right_nets.append(r)
        m = 0.5 * (bx[widest][0] + bx[widest][1])
        bl = list(bx); bl[widest] = (bx[widest][0], m)
        br = list(bx); br[widest] = (m, bx[widest][1])
        _push((tuple(bl), left_nets))
        _push((tuple(br), right_nets))
    return sols, bool(pending)


def phi_loop_seeds(S1_h, S2_h, T_nets, psi_rows, t_idx, atol, ptol,
                   max_cells=4000):
    """Seed points of the regulated curve Phi = {Psi_a, Psi_b, T_k} sliced by
    deterministic mid-planes (paper 5.3.2; axis-aligned L instead of random
    hyperplanes — random L can miss small features, admitted in their 7.1).
    Returns a list of (4,) local-coordinate seeds.

    Exclusion nets: ALL THREE Psi components + T_k + L (stronger exclusion is
    sound — a genuine loop point on Phi has full Psi = 0, and pruning the
    off-intersection part of Phi keeps the search 0-dimensional); Newton
    solves the square {Psi_a, Psi_b, T_k, L}.

    Newton uses a Levenberg-damped solve (J^T J + lambda I), NOT a plain
    `np.linalg.solve`: on symmetric geometries the loop's T_k-extreme points
    land exactly ON an axis mid-plane with grad(T_k) PARALLEL to the plane
    normal (measured on the r^4 - eps*r^2 touch-plus-loop test: at
    (0.6, 0.5, u, 0.5), grad T_k has only a t-component while L is t=0.5, so
    the square Jacobian is rank 3 and the solution set is locally the whole
    reparameterized t=0.5 line). A plain solve raises/diverges there and the
    plane contributes NO seeds; the damped step converges onto the solution
    manifold instead, and the caller's full-Psi refinement snaps such
    manifold samples onto the actual intersection set.

    Seeds are opportunistic: per-plane budget exhaustion
    (`exhausted=True` from `solve_zero_dim`) only degrades seeding
    redundancy (4 mid-planes, each meeting a loop >= 2x by Lemma 2) —
    seeds already found stay valid, so exhaustion is NOT an error and
    `max_cells` must not be raised to chase it.
    """
    from mmcore.numeric.intersection._bezier_common import eval_surface_d1

    S1_h = np.asarray(S1_h, dtype=np.float64)
    S2_h = np.asarray(S2_h, dtype=np.float64)
    G = psi_vector_net(S1_h, S2_h)
    Tk = np.asarray(T_nets[t_idx], dtype=np.float64)[..., None]
    # T_k partial-derivative nets, once (NOT per Newton iteration)
    dTk = [bernstein_partial_derivative_coeffs(Tk, axis=ax) for ax in range(4)]

    def _eval_nd(net, x):
        # bernstein_eval_nd returns a shape-(1,) array (value dim kept)
        from mmcore.numeric.bern import bernstein_eval_nd
        return float(bernstein_eval_nd(net, x).item())

    def newton_factory(axis, value):
        def newton(x0):
            x = np.asarray(x0, dtype=np.float64).copy()
            for _ in range(30):
                p1, du1, dv1 = eval_surface_d1(S1_h, x[0], x[1], rational=True)
                p2, du2, dv2 = eval_surface_d1(S2_h, x[2], x[3], rational=True)
                psi = p1 - p2
                tval = _eval_nd(Tk, x)
                F = np.array([psi[psi_rows[0]], psi[psi_rows[1]], tval,
                              x[axis] - value])
                if np.linalg.norm(F) < 1e-11:
                    return np.clip(x, 0.0, 1.0)
                Jpsi = np.column_stack([du1, dv1, -du2, -dv2])
                grad_t = np.array([_eval_nd(dTk[ax], x) for ax in range(4)])
                J = np.vstack([Jpsi[psi_rows[0]], Jpsi[psi_rows[1]], grad_t,
                               np.eye(4)[axis]])
                A = J.T @ J + 1e-12 * np.eye(4)
                try:
                    x = np.clip(x - np.linalg.solve(A, J.T @ F), 0.0, 1.0)
                except np.linalg.LinAlgError:
                    return None
            return None
        return newton

    ptol = np.asarray(ptol, dtype=np.float64)
    seeds: list = []
    for axis in range(4):
        nets = [BoxNet(G[..., k:k + 1], axes=(0, 1, 2, 3)) for k in range(3)]
        nets.append(BoxNet(Tk, axes=(0, 1, 2, 3)))
        nets.append(BoxNet(linear_net_4d(-0.5, tuple(np.eye(4)[axis])),
                           axes=(0, 1, 2, 3)))
        ax_sols, _ax_exhausted = solve_zero_dim(
            nets, newton_factory(axis, 0.5), ptol,
            max_cells=max_cells, atol=atol)
        for s in ax_sols:
            if not any(np.all(np.abs(s - t) <= ptol) for t in seeds):
                seeds.append(s)
    return seeds


def c1_pass(S1_h, S2_h, atol, ptol4, max_cells=20000):
    """Global C1 detection (paper Fig. 5): parameterization cusps ON the SSI.

    A C1 singularity is a point of the intersection curve where one
    surface's parameterization is degenerate: Sigma_i = dR_i/da x dR_i/db
    vanishes. Solving {Psi = 0} ∩ {Sigma_i = 0} per surface keeps exactly
    the cusp candidates that lie on the intersection.

    Returns (hits, curve_flag). `hits` entries are dicts:
      {"surface": 1|2, "stuv": (4,), "xyz": (3,)}  — isolated cusp, or
      {"surface": 1|2, "curve_samples": (N,4)}      — 1-dimensional set
    (cusp CURVE, paper's infinite case): flagged when the enumeration
    returns many solutions (>12) or exhausts its budget after finding
    several — a 1-dim solution set floods `solve_zero_dim` by design.
    `exhausted` with 0-1 solutions is surfaced as-is (the found root is
    emitted; absence is NOT proof in that case — the enumeration may have
    been truncated mid-search, e.g. Newton failing near a degenerate spot).

    Cheap global precheck (the common case): Sigma_i = 0 needs ALL THREE
    components zero, so ONE component whose Bernstein hull excludes zero
    over the whole [0,1]^2 proves the normal never vanishes on surface i —
    skip it for the cost of three min/max scans. Regular surfaces
    (all 7 coverage cases, all legacy minis) exit here.
    """
    from mmcore.numeric.intersection._bezier_common import (
        eval_surface, eval_surface_d1,
    )

    out: list = []
    curve_flag = False
    G = psi_vector_net(S1_h, S2_h)
    for which, (Sh, axes2) in enumerate(((S1_h, (0, 1)), (S2_h, (2, 3))), start=1):
        # bez_ssx always passes homogeneous nets (w == 1 for polynomial
        # input): unit weights take the exact polynomial branch on the
        # Cartesian part; genuinely rational input takes the
        # homogeneous-numerator branch (zeros coincide, w > 0).
        if np.allclose(Sh[..., -1], 1.0, rtol=0.0, atol=1e-14):
            N = sigma_normal_net(np.ascontiguousarray(Sh[..., :-1]), rational=False)
        else:
            N = sigma_normal_net(Sh, rational=True)
        if any(float(N[..., c].min()) > 0.0 or float(N[..., c].max()) < 0.0
               for c in range(3)):
            continue

        nets = [BoxNet(G[..., k:k + 1], axes=(0, 1, 2, 3)) for k in range(3)]
        nets += [BoxNet(np.ascontiguousarray(N[..., c:c + 1]), axes=axes2)
                 for c in range(3)]
        nscale = float(np.abs(N).max())
        if nscale <= 0.0:
            # Sigma identically zero — a globally degenerate parameterization
            # (e.g. a surface collapsed to a curve). Not a meaningful cusp
            # enumeration; report as a curve-style hit with no samples.
            curve_flag = True
            out.append({"surface": which, "curve_samples": np.empty((0, 4))})
            continue

        def newton(x0, _Sh=Sh, _axes=axes2, _ns=nscale):
            # Gauss-Newton (lstsq) on the overdetermined {Psi(3), Sigma(3)}.
            # Sigma rows' Jacobian by forward differences on the two owning
            # axes — exact d(cross) is verbose; 1e-7 FD is adequate for a
            # refiner whose acceptance is checked independently below.
            x = np.asarray(x0, dtype=np.float64).copy()
            for _ in range(40):
                p1, du1, dv1 = eval_surface_d1(S1_h, x[0], x[1], rational=True)
                p2, du2, dv2 = eval_surface_d1(S2_h, x[2], x[3], rational=True)
                psi = p1 - p2
                _, dua, dvb = eval_surface_d1(_Sh, x[_axes[0]], x[_axes[1]],
                                              rational=True)
                Nv = np.cross(dua, dvb)
                if (np.linalg.norm(psi) < 1e-10
                        and np.linalg.norm(Nv) < 1e-8 * _ns):
                    return np.clip(x, 0.0, 1.0)
                J = np.zeros((6, 4))
                J[:3, 0], J[:3, 1], J[:3, 2], J[:3, 3] = du1, dv1, -du2, -dv2
                for ax in _axes:
                    xp = x.copy()
                    xp[ax] += 1e-7
                    _, duap, dvbp = eval_surface_d1(
                        _Sh, xp[_axes[0]], xp[_axes[1]], rational=True)
                    J[3:, ax] = (np.cross(duap, dvbp) - Nv) / (1e-7 * _ns)
                F = np.concatenate([psi, Nv / _ns])
                try:
                    dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
                except np.linalg.LinAlgError:
                    return None
                x = np.clip(x + dx, 0.0, 1.0)
                if float(np.linalg.norm(dx)) < 1e-12:
                    break
            p1 = eval_surface_d1(S1_h, x[0], x[1], rational=True)[0]
            p2 = eval_surface_d1(S2_h, x[2], x[3], rational=True)[0]
            _, dua, dvb = eval_surface_d1(_Sh, x[_axes[0]], x[_axes[1]],
                                          rational=True)
            if (np.linalg.norm(p1 - p2) < atol
                    and np.linalg.norm(np.cross(dua, dvb)) < 1e-6 * _ns):
                return np.clip(x, 0.0, 1.0)
            return None

        def _xyz(sol):
            return eval_surface(S1_h, sol[0], sol[1], rational=True)

        sols, exhausted = solve_zero_dim(nets, newton, ptol4,
                                         max_cells=max_cells,
                                         dedup_xyz=_xyz, atol=atol)
        if len(sols) > 12 or (exhausted and len(sols) > 1):
            # Many hits, or a truncated enumeration that already found
            # several: a 1-dimensional solution set (cusp curve).
            curve_flag = True
            out.append({"surface": which, "curve_samples": np.asarray(sols)})
            continue
        # NOTE: exhausted with 0-1 solutions means the enumeration may be
        # incomplete; the found root (if any) is still emitted — absence is
        # not proof in that case (documented blind spot, plan risk 3).
        for s in sols:
            out.append({"surface": which, "stuv": np.asarray(s), "xyz": _xyz(s)})
    return out, curve_flag

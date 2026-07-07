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


# --- Roundoff margin for Bernstein hull sign tests (ledger L1) ---------------
#
# De Casteljau splits are convex-combination schemes: a split EXACTLY through
# a zero of the polynomial (dyadic feature coordinates are ubiquitous — the
# guided cuts deliberately pass through discovered crossings, and
# solve_zero_dim always halves) leaves the mathematically-zero corner
# coefficient at a small nonzero float. Measured on the cuspidal-edge M11
# nets ((2s-1)^2 deg-3, split at s=0.5): the zero coefficients come out at
# +2.776e-17 = eps/8 (absolute, with max|parent| = 1) and stay EXACTLY that
# value down the whole zero-adjacent descent (corner coefficients of the
# kept child are never recombined), while the child's max|c| shrinks 4x per
# level for a quadratic zero. A strict `min > 0` hull test then excludes
# BOTH children of the solution-carrying box — the zero set is knifed out.
#
# Fix: a hull may only claim "clears zero" when it clears by MORE than
# MARGIN = HULL_MARGIN_K * eps * max|coeffs| of the net under test.
# K = 128 gives 128*eps ~ 2.8e-14 relative — >= 256x the measured drift at
# the split (1.1e-16 relative to the child max) and orders of magnitude
# below any genuine signal; it keeps zero-adjacent boxes alive for ~4
# extra split levels per axis (the drift is constant absolute while the
# child max shrinks), which is what lets 1-dimensional zero sets flood the
# enumeration again (the c1_pass curve_flag path needs those hits).
#
# DIRECTION (critical): a LARGER margin makes exclusion / definiteness
# certification STRICTER — fewer boxes excluded, fewer sign-definite
# certificates, tangency probes firing MORE often. That is always the SOUND
# direction; never shrink the margin to buy pruning speed. Conversely a
# "contains zero" test built on this helper (`not hull_excludes_zero`)
# fires MORE often with a larger margin — also the safe direction.
HULL_MARGIN_K = 128.0
_HULL_MARGIN_K_EPS = HULL_MARGIN_K * float(np.finfo(np.float64).eps)


def hull_excludes_zero(coeffs) -> bool:
    """Bernstein hull sign test with the L1 roundoff margin: True proves the
    net has no zero over its box AND the clearance is not split-roundoff
    debris (`min > K*eps*max|c|`, symmetric for `max < 0`).

    NaN coefficients make both comparisons False — fail-open (never
    excludes), the safe direction. An identically-zero net has margin 0 and
    never excludes."""
    c = np.asarray(coeffs)
    mn = float(c.min())
    mx = float(c.max())
    m = _HULL_MARGIN_K_EPS * max(mx, -mn)      # = K * eps * max|c|
    return mn > m or mx < -m


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

        Margin-guarded (`hull_excludes_zero`, ledger L1): a strict `> 0`
        knifed out zero sets after de Casteljau splits exactly through a
        zero. NaN coefficients never exclude (fail-open)."""
        return hull_excludes_zero(self.coeffs)

    def split(self, global_axis: int, t: float = 0.5):
        if global_axis not in self.axes:
            return self, self
        d = self.axes.index(global_axis)
        lo, hi = de_casteljau_split_nd(self.coeffs, axis=d, t=t)
        return BoxNet(lo, self.axes), BoxNet(hi, self.axes)


@dataclass(frozen=True, eq=False)
class VectorBoxNet(BoxNet):
    """A BUNDLE of scalar nets sharing one tensor shape (trailing value dim
    of size k): semantically identical to k separate `BoxNet`s over the
    same axes — the box survives only if EVERY component's hull contains 0,
    so exclusion fires when ANY component excludes. Bundling exists purely
    to cut per-box split cost (one de Casteljau call instead of k; measured
    on the off-curve tangent enumeration's tube flood, net splitting was
    the dominant per-box cost). Splits preserve the subclass.
    """

    def excludes_zero(self) -> bool:
        # Per-component L1 roundoff margin — semantically identical to
        # `hull_excludes_zero` on each of the k bundled scalar nets.
        c = self.coeffs
        flat = c.reshape(-1, c.shape[-1])
        mn = flat.min(axis=0)
        mx = flat.max(axis=0)
        m = _HULL_MARGIN_K_EPS * np.maximum(mx, -mn)   # K * eps * max|c| per component
        return bool(np.any((mn > m) | (mx < -m)))

    def split(self, global_axis: int, t: float = 0.5):
        if global_axis not in self.axes:
            return self, self
        d = self.axes.index(global_axis)
        lo, hi = de_casteljau_split_nd(self.coeffs, axis=d, t=t)
        return VectorBoxNet(lo, self.axes), VectorBoxNet(hi, self.axes)


@dataclass(frozen=True, eq=False)
class ShiftedPositiveNet(BoxNet):
    """One-sided exclusion net: excludes iff min(coeffs) > 0.

    For SHIFTED squared-distance nets (coeffs = F_sq − thresh): `min > 0`
    proves the box is entirely OUTSIDE the tolerance shell (no Ψ-zero —
    excludable), but `max < 0` only proves it is entirely INSIDE the shell
    (i.e. ON the intersection at tolerance — the opposite of excludable),
    so the base class's two-sided hull test would wrongly prune exactly
    the boxes that matter. Splits preserve the subclass.

    L1 margin audit: deliberately NO roundoff margin here. The caller's
    shift (`thresh = (atol*w_scale)^2`, _emit_offcurve_tangent_roots)
    already dwarfs de Casteljau drift: a mathematically-zero F_sq
    coefficient drifting by ~eps*max|F_sq| ~ eps*D^2*w^2 (D = xyz scale)
    would need D > atol/sqrt(eps) ~ 7e4 model units to overcome the
    atol^2*w^2 shift — far outside the documented O(1)–O(100) envelope.
    Adding the relative margin anyway would change the net's tolerance
    semantics (the threshold is a calibrated shell radius, not a sign
    test), so it is left strict on purpose.
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
    max_boxes: Optional[int] = None,         # hard backstop on TOTAL processed boxes
):
    """All isolated solutions of {net_i = 0} in `box`.

    Returns
    -------
    (sols, exhausted) : tuple[list, bool]
        `sols` — list of (4,) solutions found. `exhausted` — True iff a
        budget (EITHER `max_cells` or `max_boxes`, see below) ran out with
        boxes still pending, i.e. the enumeration may be INCOMPLETE and
        `sols` is only a lower bound. Callers must check it (a
        silently-truncated list is indistinguishable from a complete one
        otherwise). Never raise `max_cells` to chase `exhausted=False` on
        a hang — a blown budget usually means the solution set isn't
        0-dimensional and callers must handle that case themselves
        (e.g. a curve_flag path).

    Budget contract
    ---------------
    `max_cells` bounds the CHARGED units, `max_boxes` (default
    `16 * max_cells`) bounds ALL processed boxes:

    - Without `skip_newton`, every processed box charges one unit — the
      historical semantics, unchanged (both new bounds below can then
      never bind before `max_cells` does, since every box charges).
    - With `skip_newton`, ONLY boxes whose Newton attempt actually runs
      charge (the expensive unit is the Newton/interval-GN attempt;
      hull-exclusion scans and splits are in the same cheap class):
      neither the skip-suppressed 1-dim component flood NOR its
      hull-excluded sibling boxes can starve the budget before an
      off-component isolated root's boxes subdivide out of the skip
      region (the blind band that lost touches at 5–15·atol from a
      coexisting tangent curve — measured: the excluded siblings alone
      out-charged the touch 1582:0 under per-pop charging).
    - Traversal is bounded by `boxes <= max_cells + 16*cells` (checked at
      pop): free (skip/exclusion) traversal beyond the first `max_cells`
      boxes must be PAID FOR by charged Newton work. A pure flood with no
      Newton-eligible frontier (a fully-traced tangent curve's tube,
      where every root is skip-subsumed) stops after exactly `max_cells`
      boxes — the same traversal the historical per-pop budget allowed —
      while a frontier still attempting Newtons extends the traversal
      (measured on the blind-band family: the off-curve touch is
      accepted at box 1.5-7.2k with the box count at 29-60% of the
      bound at that moment, i.e. >= 1.7x margin).
    - `max_boxes` is the hard termination backstop on top of that: a
      pathological flood whose frontier keeps attempting Newtons still
      stops there. Stopping at ANY bound with work pending returns
      `exhausted=True`.

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
    from the known component first, so budget exhaustion hits the
    (skippable) component flood last, not undiscovered isolated roots.
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

    if max_boxes is None:
        max_boxes = 16 * max_cells
    cells = 0      # charged units (see "Budget contract" above)
    boxes = 0      # every processed box — bounded by the backstops
    while (pending and cells < max_cells
           and boxes < min(max_boxes, max_cells + 16 * cells)):
        boxes += 1
        bx, bnets = _pop()
        if any(n.excludes_zero() for n in bnets):
            if skip_newton is None:
                cells += 1
            continue
        if skip_newton is None or not skip_newton(bx):
            cells += 1
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
        # L1 margin-guarded skip: one component clearing zero by MORE than
        # the roundoff margin proves Sigma_i never vanishes — a strict test
        # would let a drifted-but-mathematically-zero coefficient skip a
        # genuine cusp enumeration (larger margin = skip LESS = sound).
        if any(hull_excludes_zero(N[..., c]) for c in range(3)):
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


def theorem3_excludes_c3(T1, T2, T3, T4) -> bool:
    """Paper Theorem 3: injectivity of the 3D image over the box. One
    sign-definite minor from {T1,T2} AND one from {T3,T4} suffices —
    then the box's own image cannot self-intersect. NOTE the certificate
    is per-box: it says nothing about collisions between the images of
    two DIFFERENT boxes; those are handled by c3_pass's segment-pair
    proximity search over the traced branches.

    UNWIRED as of ledger L8: bez_ssx no longer consults this as a gate for
    c3_pass — "every traced cell certifies" does NOT imply "no C3" (the
    cross-cell blind spot: two preimage strips in different cells, each
    truthfully injective on its own). The certificate itself remains sound
    and exported for per-box use; it is just not a valid GLOBAL gate.

    Definiteness carries the L1 roundoff margin: a net is sign-definite
    only if its hull CLEARS zero by more than K*eps*max|c| — a
    split-drifted zero coefficient must not certify injectivity (a larger
    margin certifies LESS, the sound direction)."""
    def _definite(T):
        return hull_excludes_zero(T)
    return (_definite(T1) or _definite(T2)) and (_definite(T3) or _definite(T4))


def c3_pass(S1_h, S2_h, branches, atol, ptol4):
    """Post-trace C3 detection: crossing branch segments -> square 6-var
    Newton on BOTH role assignments -> certified pairs.

    The SSI image self-intersects when two DISTINCT 4D preimages share one
    3D point. The doubled preimage can live on either surface (ledger L7):

      S1-side: {R1(s,t) = R2(u,v), R1(p,q) = R2(u,v)}, guard (s,t) != (p,q)
      S2-side: {R2(u,v) = R1(s,t), R2(u',v') = R1(s,t)}, guard (u,v) != (u',v')

    A double preimage on S1 with a SINGLE S2 preimage (the Whitney-umbrella
    class with the umbrella as S1) only solves the first system — its
    mirror (umbrella as S2, same S1 preimage on the plane, two S2
    preimages) only the second. Both systems run on the same broadphase
    candidates and their hits are deduplicated together.

    Candidates: pairs of polyline segments (across branches, or within one
    branch at index distance > 2) whose segment-segment xyz distance is
    < 2*atol. Each candidate seeds the square Newton in the doubled side's
    variables; a solution is a self-intersection iff both residuals meet
    atol and the doubled-side preimages differ by > 4*ptol on some axis
    (same-preimage solutions are the ordinary curve point, not a C3).

    Returns a list of dicts {"stuv": (4,), "stuv_mate": (4,), "xyz": (3,),
    "links": [(branch_i, k), (branch_j, l)]}. `stuv` is the primary 4D
    preimage (s,t,u,v); `stuv_mate` differs from it in the doubled side
    only: (p,q,u,v) for an S1-side double, (s,t,u',v') for an S2-side
    double.
    """
    from mmcore.numeric.intersection._bezier_common import (
        eval_surface, eval_surface_d1,
    )
    ptol4 = np.asarray(ptol4, dtype=np.float64)

    def seg_dist(p1, p2, q1, q2):
        d1 = p2 - p1; d2 = q2 - q1; r = p1 - q1
        a = d1 @ d1; e = d2 @ d2; f = d2 @ r
        if a < 1e-30 and e < 1e-30:
            return float(np.linalg.norm(r)), 0.0, 0.0
        if a < 1e-30:
            s_ = 0.0; t_ = float(np.clip(f / e, 0.0, 1.0))
        else:
            c = d1 @ r
            if e < 1e-30:
                t_ = 0.0; s_ = float(np.clip(-c / a, 0.0, 1.0))
            else:
                b = d1 @ d2; den = a * e - b * b
                s_ = float(np.clip((b * f - c * e) / den, 0.0, 1.0)) if den > 1e-30 else 0.0
                t_ = float(np.clip((b * s_ + f) / e, 0.0, 1.0))
        cp1 = p1 + s_ * d1; cp2 = q1 + t_ * d2
        return float(np.linalg.norm(cp1 - cp2)), s_, t_

    def newton6(z0, Sa_h, Sb_h, guard_ptol):
        # z = (a1,a2, b1,b2, c1,c2): solve {Ra(a) = Rb(c), Ra(b) = Rb(c)}.
        # (a1,a2) and (b1,b2) are the DOUBLED-side preimages on Sa; (c1,c2)
        # is the single preimage on Sb. Called once per candidate with
        # (S1_h, S2_h) and once with the roles swapped (ledger L7).
        z = np.asarray(z0, dtype=np.float64).copy()
        for _ in range(40):
            ra, dua, dva = eval_surface_d1(Sa_h, z[0], z[1], rational=True)
            rb, dub, dvb = eval_surface_d1(Sa_h, z[2], z[3], rational=True)
            rc, duc, dvc = eval_surface_d1(Sb_h, z[4], z[5], rational=True)
            F = np.concatenate([ra - rc, rb - rc])
            if np.linalg.norm(F) < 1e-11:
                break
            J = np.zeros((6, 6))
            J[:3, 0], J[:3, 1], J[:3, 4], J[:3, 5] = dua, dva, -duc, -dvc
            J[3:, 2], J[3:, 3], J[3:, 4], J[3:, 5] = dub, dvb, -duc, -dvc
            try:
                z = np.clip(z - np.linalg.solve(J, F), 0.0, 1.0)
            except np.linalg.LinAlgError:
                return None
        ra = eval_surface(Sa_h, z[0], z[1], rational=True)
        rb = eval_surface(Sa_h, z[2], z[3], rational=True)
        rc = eval_surface(Sb_h, z[4], z[5], rational=True)
        if max(float(np.linalg.norm(ra - rc)),
               float(np.linalg.norm(rb - rc))) > atol:
            return None
        if np.all(np.abs(z[:2] - z[2:4]) <= 4.0 * np.asarray(guard_ptol)):
            return None            # same preimage — not a self-intersection
        return z

    def solve_candidate(a4, b4):
        """Run both role assignments from one candidate's interpolated 4D
        params; return normalized hits [(stuv, stuv_mate, xyz), ...]."""
        hits = []
        # S1-side double: z = (s,t, p,q, u,v)
        z = newton6(np.array([a4[0], a4[1], b4[0], b4[1],
                              0.5 * (a4[2] + b4[2]), 0.5 * (a4[3] + b4[3])]),
                    S1_h, S2_h, ptol4[:2])
        if z is not None:
            hits.append((np.array([z[0], z[1], z[4], z[5]]),
                         np.array([z[2], z[3], z[4], z[5]]),
                         eval_surface(S2_h, z[4], z[5], rational=True)))
        # S2-side double: z = (u,v, u',v', s,t)
        z = newton6(np.array([a4[2], a4[3], b4[2], b4[3],
                              0.5 * (a4[0] + b4[0]), 0.5 * (a4[1] + b4[1])]),
                    S2_h, S1_h, ptol4[2:])
        if z is not None:
            hits.append((np.array([z[4], z[5], z[0], z[1]]),
                         np.array([z[4], z[5], z[2], z[3]]),
                         eval_surface(S1_h, z[4], z[5], rational=True)))
        return hits

    # --- vectorized AABB broadphase over ALL branch segments -------------
    # The per-box Theorem-3 gate in bez_ssx fires liberally (a coarse traced
    # cell's T-hull touching zero at a domain edge already defeats it — the
    # plain bilinear/plane pair traces from its TOP cell and fails the
    # certificate), so this search must be cheap when there is nothing to
    # find. One numpy broadcast tests every segment pair's atol-inflated
    # AABB overlap; the exact seg_dist + Newton run only on survivors
    # (measured: a couple of ms at M ~ 500 segments vs ~0.5 s for the plain
    # O(M^2) Python loop).
    found: list = []
    segs_a, segs_b, seg_s4a, seg_s4b, seg_branch, seg_idx = [], [], [], [], [], []
    for bi, b in enumerate(branches):
        xyz = np.asarray(b.curve[1], dtype=np.float64)
        stuv = np.asarray(b.curve[0], dtype=np.float64)
        if len(xyz) < 2:
            continue
        segs_a.append(xyz[:-1]); segs_b.append(xyz[1:])
        seg_s4a.append(stuv[:-1]); seg_s4b.append(stuv[1:])
        seg_branch.append(np.full(len(xyz) - 1, bi))
        seg_idx.append(np.arange(len(xyz) - 1))
    if not segs_a:
        return found
    A = np.concatenate(segs_a); B = np.concatenate(segs_b)
    S4a = np.concatenate(seg_s4a); S4b = np.concatenate(seg_s4b)
    br = np.concatenate(seg_branch); ix = np.concatenate(seg_idx)
    lo = np.minimum(A, B) - atol
    hi = np.maximum(A, B) + atol
    M = len(A)
    pairs = []
    block = 1024                    # bound the broadcast to blocks of M x block
    for r0 in range(0, M, block):
        r1 = min(r0 + block, M)
        ov = np.all((lo[r0:r1, None, :] <= hi[None, :, :])
                    & (lo[None, :, :] <= hi[r0:r1, None, :]), axis=2)
        ki, li = np.nonzero(ov)
        ki = ki + r0
        keep = li > ki              # unordered pairs once
        ki, li = ki[keep], li[keep]
        same = br[ki] == br[li]     # same-branch adjacency: index gap >= 3
        keep = ~same | (np.abs(ix[ki] - ix[li]) >= 3)
        pairs.append(np.stack([ki[keep], li[keep]], axis=1))
    pairs = np.concatenate(pairs) if pairs else np.empty((0, 2), dtype=int)
    def _pair_dist(h, stuv, mate):
        # Distance between two hits as unordered preimage PAIRS: the two
        # runs (and re-finds from other candidate seeds) may present the
        # same feature with primary/mate swapped.
        d_direct = max(float(np.abs(h["stuv"] - stuv).max()),
                       float(np.abs(h["stuv_mate"] - mate).max()))
        d_swap = max(float(np.abs(h["stuv"] - mate).max()),
                     float(np.abs(h["stuv_mate"] - stuv).max()))
        return min(d_direct, d_swap)

    for k, l in pairs:
        d, s_, t_ = seg_dist(A[k], B[k], A[l], B[l])
        if d > 2.0 * atol:
            continue
        a4 = (1 - s_) * S4a[k] + s_ * S4b[k]
        b4 = (1 - t_) * S4a[l] + t_ * S4b[l]
        for stuv, mate, xyz in solve_candidate(a4, b4):
            if any(_pair_dist(h, stuv, mate) <= 4.0 * float(np.max(ptol4))
                   for h in found):
                continue
            found.append({"stuv": stuv, "stuv_mate": mate, "xyz": xyz,
                          "links": [(int(br[k]), int(ix[k])),
                                    (int(br[l]), int(ix[l]))]})
    return found

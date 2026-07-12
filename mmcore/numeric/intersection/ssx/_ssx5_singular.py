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

from mmcore.numeric._work_budget import LatchingSpend
from mmcore.numeric.bern import (
    bernstein_eval_nd,
    bernstein_partial_derivative_coeffs,
    de_casteljau_split_nd,
)
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


def _rational_jacobian_numerator_columns(S_h: np.ndarray):
    """The two quotient-rule NUMERATOR column nets of a rational surface's
    Jacobian, as exact Bernstein vec3 patches over the surface's own (a,b)
    params (ledger L9).

    For R = P/w (P = weighted-numerator net S_h[...,:-1], w = weight net
    S_h[...,-1], mmcore convention), the quotient rule gives
        R_a = (P_a*w - P*w_a) / w^2,   R_b = (P_b*w - P*w_b) / w^2.
    This returns the NUMERATORS
        col_a = P_a*w - P*w_a,   col_b = P_b*w - P*w_b
    (built from the same exact same-parameter Bernstein products as
    `sigma_normal_net`'s rational branch — `_same_param_product_vec_scalar`).
    Since w > 0 everywhere for valid weights, R_a = col_a / w^2 with a
    strictly positive denominator, so any minor formed from these columns
    equals the true rational-Jacobian minor times a positive power-of-w
    factor — identical zero set and sign structure.
    """
    S_h = np.asarray(S_h, dtype=np.float64)
    if S_h.ndim != 3 or S_h.shape[-1] != 4:
        raise ValueError(
            "_rational_jacobian_numerator_columns requires a homogeneous "
            f"(m+1, n+1, 4) control net (got shape {S_h.shape})")
    P, w = S_h[..., :-1], S_h[..., -1]
    Pa = bernstein_partial_derivative_coeffs(P, axis=0)
    Pb = bernstein_partial_derivative_coeffs(P, axis=1)
    wa = bernstein_partial_derivative_coeffs(w[..., None], axis=0)[..., 0]
    wb = bernstein_partial_derivative_coeffs(w[..., None], axis=1)[..., 0]
    col_a = (_same_param_product_vec_scalar(Pa, w)
             - _same_param_product_vec_scalar(P, wa))
    col_b = (_same_param_product_vec_scalar(Pb, w)
             - _same_param_product_vec_scalar(P, wb))
    return col_a, col_b


def minors_Tpsi_rational(S1_h: np.ndarray, S2_h: np.ndarray):
    """TPsi minor nets for genuinely RATIONAL surface input (ledger L9).

    The polynomial path (`minors_Tpsi_from_control_nets` on dehomogenized
    control points) is WRONG for non-uniform weights: the Bernstein net of
    per-control-point quotients P/w is NOT the rational surface R = P/w, so
    its derivatives — hence its minors — describe a different surface pair.

    Instead build the true rational Jacobian's NUMERATOR columns via the
    quotient rule (`_rational_jacobian_numerator_columns`) and form the four
    minors from them (`minors_from_column_nets`). Each numerator minor equals
    the corresponding true rational minor times a strictly positive
    power-of-W factor (W1, W2 > 0 for valid weights), so it shares the true
    minor's zero set AND sign structure — exactly what every consumer needs
    (hull gates, sign-definiteness certificates, monotonicity, deflation
    zero-finding). Returns nested-list nets, same container type as
    `minors_Tpsi_from_control_nets`.
    """
    from mmcore.numeric.intersection._deflate import minors_from_column_nets
    A_s, A_t = _rational_jacobian_numerator_columns(S1_h)
    B_u, B_v = _rational_jacobian_numerator_columns(S2_h)
    return minors_from_column_nets(
        A_s.tolist(), A_t.tolist(), B_u.tolist(), B_v.tolist())


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
    stats: Optional[dict] = None,            # out-param: solver-side counters (see below)
    charge_box: Optional[Callable[[int], bool]] = None,
    # shared outer budget: charge_box(1) must approve BEFORE a box is processed
    max_results: Optional[int] = None,
):
    """All isolated solutions of {net_i = 0} in `box`.

    Returns
    -------
    (sols, exhausted) : tuple[list, bool]
        `sols` — list of (4,) solutions found. `exhausted` — True iff a
        budget (`max_cells`, the `16 * max_cells` traversal backstop, or
        the optional shared
        `charge_box`, see below) ran out with boxes still pending, i.e. the
        enumeration may be INCOMPLETE and `sols` is only a lower bound.
        Callers must check it (a
        silently-truncated list is indistinguishable from a complete one
        otherwise). Never raise `max_cells` to chase `exhausted=False` on
        a hang — a blown budget usually means the solution set isn't
        0-dimensional and callers must handle that case themselves
        (e.g. a curve_flag path).

    Budget contract
    ---------------
    `max_cells` bounds the CHARGED units; a fixed `16 * max_cells`
    backstop bounds ALL processed boxes:

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
    - The `16 * max_cells` box count is the hard termination backstop on
      top of that: a pathological flood whose frontier keeps attempting
      Newtons still stops there. Stopping at ANY bound with work pending
      returns `exhausted=True`.
    - `charge_box`, when supplied, is a shared outer-budget callback. It is
      invoked as `charge_box(1)` immediately before each box is popped. A
      false result stops the solve without processing that box and returns
      `exhausted=True`; the local `max_cells` / backstop limits remain in
      force independently. This lets one SSX-level allowance cover several
      nested zero-dimensional solves instead of resetting at every call.

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
    # `stats` out-param (ledger L14): 'floor_boxes' counts boxes that hit
    # the resolution floor WITHOUT being hull-excluded — the solver-side
    # dimensionality signature. Isolated roots leave O(1) floor boxes per
    # root; a 1-dimensional zero set floods ~(arc length / ptol) of them
    # even when xyz-dedup collapses the returned `sols` to a sparse
    # handful (measured: cusp curve at 200x resolution -> 11 sols but
    # hundreds of floor boxes; a lone cusp -> a few dozen).
    floor_boxes = 0
    # A surviving box at the parametric resolution floor is not by itself
    # a proof that a zero exists or that none exists.  Record the subset for
    # which Newton produced no in-box root witness so callers making a
    # topological type claim can surface honest partial status.
    unresolved_floor_boxes = 0

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

    max_boxes = 16 * max_cells
    cells = 0      # charged units (see "Budget contract" above)
    boxes = 0      # every processed box — bounded by the backstops
    external_budget_exhausted = False
    while (pending and cells < max_cells
           and boxes < min(max_boxes, max_cells + 16 * cells)
           and (max_results is None or len(sols) < max_results)):
        # The shared allowance is charged before the pop: denial must leave
        # the next box unprocessed so `exhausted=True` faithfully means the
        # returned solution list is only partial.
        if charge_box is not None and not charge_box(1):
            external_budget_exhausted = True
            break
        boxes += 1
        bx, bnets = _pop()
        if any(n.excludes_zero() for n in bnets):
            if skip_newton is None:
                cells += 1
            continue
        box_has_root_witness = False
        if skip_newton is None or not skip_newton(bx):
            cells += 1
            mid = np.array([0.5 * (lo + hi) for lo, hi in bx])
            sol = newton(mid)
            if sol is not None:
                sol = np.asarray(sol, dtype=np.float64)
                inside = all(bx[i][0] - 1e-12 <= sol[i] <= bx[i][1] + 1e-12 for i in range(4))
                if inside:
                    box_has_root_witness = True
                    if not _dup(sol):
                        sols.append(sol)
        # split the axis with the largest span in units of its OWN ptol —
        # with heterogeneous per-axis ptols the absolutely-widest axis can
        # already be resolved while a tighter-ptol axis is still orders of
        # magnitude above its floor
        ratios = [(hi - lo) / float(ptol[i]) for i, (lo, hi) in enumerate(bx)]
        widest = int(np.argmax(ratios))
        if ratios[widest] <= 1.0:
            floor_boxes += 1
            if not box_has_root_witness:
                unresolved_floor_boxes += 1
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
    exhausted = bool(pending)
    result_limit_reached = bool(
        pending and max_results is not None and len(sols) >= max_results)
    if stats is not None:
        stats["floor_boxes"] = floor_boxes
        stats["unresolved_floor_boxes"] = unresolved_floor_boxes
        stats["cells_processed"] = cells
        stats["boxes_processed"] = boxes
        stats["budget_exhausted"] = exhausted
        stats["external_budget_exhausted"] = external_budget_exhausted
        stats["result_limit_reached"] = result_limit_reached
    return sols, exhausted


def phi_loop_seeds(S1_h, S2_h, T_nets, psi_rows, t_idx, atol, ptol,
                   max_cells=4000, charge_box=None, stats=None):
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
    `max_cells` must not be raised to chase it. `charge_box`, when supplied,
    is passed unchanged to every plane solve so an SSX-level allowance is
    shared across them. The return value stays backward-compatible; callers
    that need to surface truncation may pass `stats` and inspect
    `budget_exhausted` / `external_budget_exhausted`.
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

    from mmcore.numeric.intersection._bezier_common import eval_surface

    ptol = np.asarray(ptol, dtype=np.float64)
    seeds: list = []
    seed_xyz: list = []
    any_exhausted = False
    external_budget_exhausted = False
    cells_processed = 0
    boxes_processed = 0
    solve_calls = 0
    for axis in range(4):
        nets = [BoxNet(G[..., k:k + 1], axes=(0, 1, 2, 3)) for k in range(3)]
        nets.append(BoxNet(Tk, axes=(0, 1, 2, 3)))
        nets.append(BoxNet(linear_net_4d(-0.5, tuple(np.eye(4)[axis])),
                           axes=(0, 1, 2, 3)))
        ax_stats = {}
        ax_sols, ax_exhausted = solve_zero_dim(
            nets, newton_factory(axis, 0.5), ptol,
            max_cells=max_cells, max_results=256,
            atol=atol, charge_box=charge_box,
            stats=ax_stats)
        solve_calls += 1
        any_exhausted |= bool(ax_exhausted)
        external_budget_exhausted |= bool(
            ax_stats.get("external_budget_exhausted", False)
        )
        cells_processed += int(ax_stats.get("cells_processed", 0))
        boxes_processed += int(ax_stats.get("boxes_processed", 0))
        for s in ax_sols:
            # Destructive dedup ladder (ledger L21): a parametric box is
            # not a metric ball — merge cross-plane seeds only when BOTH
            # param-close (1·ptol/axis) AND xyz-close (≤ atol). A seed
            # pair within ptol on a steep axis can be many atol apart in
            # space and must survive as two seeds.
            s_xyz = eval_surface(S1_h, s[0], s[1], rational=True)
            dup = False
            for t, t_xyz in zip(seeds, seed_xyz):
                if (np.all(np.abs(s - t) <= ptol)
                        and float(np.linalg.norm(s_xyz - t_xyz)) <= atol):
                    dup = True
                    break
            if not dup:
                seeds.append(s)
                seed_xyz.append(s_xyz)
        # A denied shared charge cannot become available again during this
        # synchronous call. Preserve the partial seeds and avoid three more
        # futile solve invocations.
        if external_budget_exhausted:
            break
    if stats is not None:
        stats["solve_calls"] = solve_calls
        stats["cells_processed"] = cells_processed
        stats["boxes_processed"] = boxes_processed
        stats["budget_exhausted"] = any_exhausted
        stats["external_budget_exhausted"] = external_budget_exhausted
    return seeds


def _connected_one_dim(sols, newton, ptol) -> bool:
    """True when 2..12 solutions sample a CONNECTED 1-dimensional zero set.

    Ledger L14: sort the cloud along its principal axis, then for each
    consecutive pair Newton the parametric MIDPOINT with the same solver
    the enumeration used. On a genuine curve the midpoint converges to a
    NEW root strictly between the pair (inside their joint AABB +ptol and
    distinct from both at 1·ptol); between genuinely isolated cusps it
    diverges or falls back onto an endpoint. Curve-like iff at least half
    the tested pairs connect. Pairs already adjacent at resolution
    (≤ 2·ptol per axis) count as connected — they are indistinguishable
    from a curve at solver resolution by definition. Cost: ≤ n-1
    Gauss-Newton calls, only on the ambiguous 2..12 window."""
    if len(sols) < 2 or len(sols) > 12:
        return False
    P = np.asarray(sols, dtype=np.float64)
    ptol = np.asarray(ptol, dtype=np.float64)
    U = P - P.mean(axis=0)
    try:
        _, _, Vt = np.linalg.svd(U, full_matrices=False)
    except np.linalg.LinAlgError:
        return False
    P = P[np.argsort(U @ Vt[0])]
    tested = 0
    connected = 0
    for a, b in zip(P[:-1], P[1:]):
        tested += 1
        if np.all(np.abs(b - a) <= 2.0 * ptol):
            connected += 1
            continue
        m = newton(0.5 * (a + b))
        if m is None:
            continue
        m = np.asarray(m, dtype=np.float64)
        lo = np.minimum(a, b) - ptol
        hi = np.maximum(a, b) + ptol
        if (np.all(m >= lo) and np.all(m <= hi)
                and np.any(np.abs(m - a) > ptol)
                and np.any(np.abs(m - b) > ptol)):
            connected += 1
    return tested > 0 and 2 * connected >= tested


def c1_pass(S1_h, S2_h, atol, ptol4, max_cells=20000,
            charge_box=None, stats=None):
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

    `max_cells` is one allowance for the whole C1 pass, shared by the two
    possible surface solves (it is not reset per surface). `charge_box`, if
    supplied, additionally shares an outer SSX-level allowance with other
    singularity work. The historical `(hits, curve_flag)` return is retained;
    pass `stats` to inspect `budget_exhausted` and distinguish an incomplete
    C1 pass from a complete empty result.

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
    cells_remaining = max(0, int(max_cells))
    cells_processed = 0
    boxes_processed = 0
    solve_calls = 0
    any_exhausted = False
    external_budget_exhausted = False
    incomplete = False
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
            # (max|N| is the exact identically-zero test — a sampled scale
            # could miss it — so it stays the gate here.)
            curve_flag = True
            out.append({"surface": which, "curve_samples": np.empty((0, 4))})
            continue

        sigma_roundoff_tol = _HULL_MARGIN_K_EPS * nscale

        def _sigma_numerator_is_roundoff_zero(
                x, _N=N, _axes=axes2, _tol=sigma_roundoff_tol):
            """Certify the exact polynomial Sigma numerator as numerical zero.

            The Cartesian normal is suitable for conditioning the GN system,
            but not for a topological ``Sigma == 0`` claim: a fixed local-angle
            or sampled-normal ratio can accept a merely small regular normal,
            while derivative-zero rational cusps suffer cancellation in the
            quotient-rule Cartesian evaluation.  The Bernstein numerator net
            is the algebraic system being subdivided and has the same zero set
            for valid positive weights.  Evaluate that net at the candidate and
            allow only its coefficient-scale roundoff envelope.
            """
            value = np.asarray(
                bernstein_eval_nd(
                    _N,
                    (float(x[_axes[0]]), float(x[_axes[1]])),
                ),
                dtype=np.float64,
            )
            return bool(
                np.all(np.isfinite(value))
                and float(np.linalg.norm(value)) <= _tol
            )

        # L13: weight-INVARIANT acceptance scale for the GN below. The GN
        # normalizes the CARTESIAN normal Nv = cross(du,dv) (from rational
        # eval_surface_d1 — invariant under a uniform weight rescale) by a
        # scale; using the homogeneous NUMERATOR net max |N| (which scales
        # as W^4) made acceptance weight-DEPENDENT, so rescaling every
        # weight by a constant c sent nscale -> c^4 nscale while Nv stayed
        # fixed and cusp detection FLIPPED (B C9). Use the max Cartesian
        # normal magnitude over a small sample grid instead: a cusp is where
        # the normal vanishes, so the grid max is a representative
        # non-degenerate scale (floored so a near-degenerate patch cannot
        # divide by ~0). Sampled, not the net-coefficient bound, so it is
        # the actual |cross(du,dv)| magnitude the residual is measured in.
        _grid = np.linspace(0.0, 1.0, 5)
        cart_nscale = 0.0
        for _ga in _grid:
            for _gb in _grid:
                _, _gdu, _gdv = eval_surface_d1(Sh, float(_ga), float(_gb),
                                                rational=True)
                cart_nscale = max(cart_nscale,
                                  float(np.linalg.norm(np.cross(_gdu, _gdv))))
        cart_nscale = max(cart_nscale, 1e-12)

        def newton(x0, _Sh=Sh, _axes=axes2, _ns=cart_nscale,
                   _sigma_zero=_sigma_numerator_is_roundoff_zero):
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
                        and _sigma_zero(x)):
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
            if (np.linalg.norm(p1 - p2) < atol
                    and _sigma_zero(x)):
                return np.clip(x, 0.0, 1.0)
            return None

        def _xyz(sol):
            return eval_surface(S1_h, sol[0], sol[1], rational=True)

        if cells_remaining <= 0:
            # This surface still needs enumeration, so a consumed shared
            # local allowance makes the overall C1 result incomplete.
            any_exhausted = True
            incomplete = True
            break
        # Reserve a fair share for every not-yet-visited surface. A
        # positive-dimensional C1 set can consume any allowance; handing it
        # the whole remainder starved the second surface on two-cone apex
        # cases even though both sets were readily classifiable.
        solve_allowance = max(1, cells_remaining // (3 - which))
        solve_stats = {}
        sols, exhausted = solve_zero_dim(nets, newton, ptol4,
                                         max_cells=solve_allowance,
                                         max_results=64,
                                         dedup_xyz=_xyz, atol=atol,
                                         charge_box=charge_box,
                                         stats=solve_stats)
        solve_calls += 1
        used_cells = int(solve_stats.get("cells_processed", 0))
        cells_processed += used_cells
        boxes_processed += int(solve_stats.get("boxes_processed", 0))
        cells_remaining = max(0, cells_remaining - used_cells)
        any_exhausted |= bool(exhausted)
        external_budget_exhausted |= bool(
            solve_stats.get("external_budget_exhausted", False)
        )
        if int(solve_stats.get("unresolved_floor_boxes", 0)) > 0:
            # These boxes survived every Bernstein exclusion test but had
            # no in-box Newton witness at the requested resolution.  They
            # are ambiguous: neither an empty C1 set nor a cusp is proven.
            incomplete = True
        if external_budget_exhausted:
            incomplete = True
            # A shared-budget denial can truncate the solution cloud in a
            # way that mimics either isolated cusps or a curve.  Keep prior
            # certified hits, but make no schema claim for this surface.
            break
        # 1-dimensional-set detection (ledger L14): raw count and the
        # exhausted flag miss a REALISTIC cusp curve whose xyz-dedup'd
        # solutions land in the 2..12 window without budget exhaustion
        # (measured: cuspidal edge in plane x=0 clipped to t-extent 0.2 =
        # 200x resolution -> 11 isolated 'cusp's). Solver-side counters
        # were measured NON-discriminative for face-aligned curves (the
        # common case — guided cuts pass through crossings, and dyadic
        # feature coordinates sit exactly on split faces: the descent
        # terminates by exclusion with ZERO resolution-floor boxes on
        # curves, 16 on the isolated control — inverted and fragile).
        # Decisive test: CONNECTIVITY — Newton the midpoints of
        # consecutive solutions; a curve yields new on-segment roots,
        # isolated cusps yield dups or divergence.
        curve_like = (len(sols) > 12
                      or (exhausted and len(sols) > 1))
        connection_denied = False

        def _charged_connection_newton(x0):
            nonlocal cells_remaining, cells_processed
            nonlocal any_exhausted, external_budget_exhausted, incomplete
            nonlocal connection_denied
            if cells_remaining <= 0:
                any_exhausted = True
                incomplete = True
                connection_denied = True
                return None
            if charge_box is not None and not charge_box(1):
                any_exhausted = True
                external_budget_exhausted = True
                incomplete = True
                connection_denied = True
                return None
            cells_remaining -= 1
            cells_processed += 1
            return newton(x0)

        if not curve_like:
            curve_like = _connected_one_dim(
                sols, _charged_connection_newton, ptol4)
        if connection_denied:
            # The gray-zone dimension test was truncated.  Its roots are
            # certified C1 members, but we cannot safely choose between the
            # isolated-cusp and cusp-curve output schemas.
            break
        if curve_like:
            curve_flag = True
            out.append({"surface": which, "curve_samples": np.asarray(sols)})
        else:
            # NOTE: exhausted with 0-1 solutions means the enumeration may be
            # incomplete; the found root (if any) is still emitted — absence is
            # not proof in that case (documented blind spot, plan risk 3).
            for s in sols:
                out.append({"surface": which, "stuv": np.asarray(s), "xyz": _xyz(s)})
            if exhausted:
                incomplete = True
        if external_budget_exhausted:
            incomplete = True
        if external_budget_exhausted or (exhausted and cells_remaining <= 0):
            break
    if stats is not None:
        stats["solve_calls"] = solve_calls
        stats["cells_processed"] = cells_processed
        stats["boxes_processed"] = boxes_processed
        stats["cells_remaining"] = cells_remaining
        stats["budget_exhausted"] = any_exhausted
        stats["external_budget_exhausted"] = external_budget_exhausted
        stats["incomplete"] = incomplete
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


def _c3_same_hit(stuv_a, mate_a, xyz_a, stuv_b, mate_b, xyz_b, atol, ptol4):
    """Both-guards C3 dedup predicate (ledger L16): two hits are the SAME
    self-intersection only if their 3D points agree within 2*atol AND
    their unordered preimage pairs match per-axis within 4*ptol4. Failing
    EITHER guard means distinct: a 4*ptol parametric box is not a metric
    ball (the old norm(z) <= 4*max(ptol4) ball merged distinct C3 points
    hundreds of atol apart in xyz wherever some axis' ptol is large), and
    one 3D point can carry genuinely distinct preimage pairs (both
    surfaces 2-to-1 there). Primary/mate may swap between the two
    role-assignment runs and between seeds, so both orderings are tested.
    """
    if float(np.linalg.norm(np.asarray(xyz_a, dtype=np.float64)
                            - np.asarray(xyz_b, dtype=np.float64))) > 2.0 * atol:
        return False
    box4 = 4.0 * np.asarray(ptol4, dtype=np.float64)
    sa = np.asarray(stuv_a, dtype=np.float64)
    ma = np.asarray(mate_a, dtype=np.float64)
    for pb, qb in ((stuv_b, mate_b), (mate_b, stuv_b)):
        if (np.all(np.abs(sa - np.asarray(pb, dtype=np.float64)) <= box4)
                and np.all(np.abs(ma - np.asarray(qb, dtype=np.float64)) <= box4)):
            return True
    return False


def c3_pass(S1_h, S2_h, branches, atol, ptol4, *,
            max_work=250_000, charge_work=None, stats=None):
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
    <= 5*atol (ledger L23: each traced chord may deviate up to the 2*atol
    sagitta bar from its true curve, so two chords can pass ~4*atol apart
    while the true curves intersect — 2*atol missed those; 5*atol adds
    slack, and the broadphase AABB padding matches at 2.5*atol per box).
    Each candidate seeds the square Newton in the doubled side's
    variables; a solution is a self-intersection iff both residuals meet
    atol and the doubled-side preimages differ by > 4*ptol on some axis
    (same-preimage solutions are the ordinary curve point, not a C3 —
    this guard is also what keeps the wider window safe: near-miss
    candidates between branches that do NOT cross converge back onto a
    single preimage and are rejected).

    Dedup: `_c3_same_hit` (both-guards, ledger L16); a duplicate re-find
    contributes any NEW branch links to the kept hit.

    ``max_work`` bounds segment setup, every unordered AABB comparison, and
    all downstream exact/Newton/anchor/dedup work. Blocks are streamed and
    processed immediately, so dense input cannot materialize an unbounded
    O(M^2) pair array. ``charge_work`` optionally spends the same work from
    an enclosing SSX allowance; ``stats`` reports local/external exhaustion.

    Returns a list of dicts {"stuv": (4,), "stuv_mate": (4,), "xyz": (3,),
    "links": [(branch_i, vertex_k), (branch_j, vertex_l)]}. `stuv` is the
    primary 4D preimage (s,t,u,v); `stuv_mate` differs from it in the
    doubled side only: (p,q,u,v) for an S1-side double, (s,t,u',v') for an
    S2-side double. Links carry VERTEX indices of the linked branch's
    polyline — the vertex nearest the refined crossing (ledger L11), not
    the broadphase segment index.
    """
    from mmcore.numeric.intersection._bezier_common import (
        eval_surface, eval_surface_d1,
    )
    ptol4 = np.asarray(ptol4, dtype=np.float64)
    pairs_processed = 0
    candidate_pairs = 0
    # check-then-charge / all-or-nothing / latching ledger with the shared
    # budget as the external hook — the L52 shared implementation of the
    # former hand-rolled ``_spend`` closure.
    _ledger = LatchingSpend(max_work=max_work, charge_external=charge_work)
    _spend = _ledger.spend

    def _publish_stats():
        if stats is not None:
            stats.update(
                work_processed=int(_ledger.work_processed),
                pairs_processed=int(pairs_processed),
                candidate_pairs=int(candidate_pairs),
                budget_exhausted=bool(_ledger.exhausted),
                external_budget_exhausted=bool(_ledger.external_exhausted),
                incomplete=bool(_ledger.exhausted),
            )

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
            if not _spend(1):
                return None
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
        if not _spend(1):
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
    # bez_ssx runs this pass UNCONDITIONALLY whenever a collision is
    # possible (ledger L8 removed the unsound per-cell Theorem-3 gate), so
    # this search must be cheap when there is nothing to find. One numpy
    # broadcast tests every segment pair's 2.5*atol-inflated AABB overlap
    # (pairwise 5*atol — matching the seg_dist window, ledger L23); the
    # exact seg_dist + Newton run only on survivors (measured: a couple of
    # ms at M ~ 500 segments vs ~0.5 s for the plain O(M^2) Python loop;
    # 0 pairs on coverage case 10's 115 well-separated segments).
    found: list = []
    segs_a, segs_b, seg_s4a, seg_s4b, seg_branch, seg_idx = [], [], [], [], [], []
    for bi, b in enumerate(branches):
        xyz = np.asarray(b.curve[1], dtype=np.float64)
        stuv = np.asarray(b.curve[0], dtype=np.float64)
        if len(xyz) < 2:
            continue
        if not _spend(len(xyz) - 1):
            _publish_stats()
            return found
        segs_a.append(xyz[:-1]); segs_b.append(xyz[1:])
        seg_s4a.append(stuv[:-1]); seg_s4b.append(stuv[1:])
        seg_branch.append(np.full(len(xyz) - 1, bi))
        seg_idx.append(np.arange(len(xyz) - 1))
    if not segs_a:
        _publish_stats()
        return found
    A = np.concatenate(segs_a); B = np.concatenate(segs_b)
    S4a = np.concatenate(seg_s4a); S4b = np.concatenate(seg_s4b)
    br = np.concatenate(seg_branch); ix = np.concatenate(seg_idx)
    lo = np.minimum(A, B) - 2.5 * atol
    hi = np.maximum(A, B) + 2.5 * atol
    M = len(A)

    def _anchor_vertex(bi, seg_k, xyz):
        # Ledger L11: links carry VERTEX indices — the polyline vertex
        # nearest the refined crossing, found by walking downhill from the
        # seeding segment (the broadphase segment index pointed up to
        # ~half a chord away from the Newton-refined point). The walk, not
        # a global argmin, keeps the anchor on the LOCAL pass when one
        # branch crosses itself: the other pass's globally-nearest vertex
        # would collapse both links onto one location.
        poly = np.asarray(branches[bi].curve[1], dtype=np.float64)
        v = int(seg_k)
        if not _spend(2):
            return None
        d = float(np.linalg.norm(poly[v] - xyz))
        d2 = float(np.linalg.norm(poly[v + 1] - xyz))
        if d2 < d:
            v, d = v + 1, d2
        improved = True
        while improved:
            improved = False
            for w in (v - 1, v + 1):
                if 0 <= w < len(poly):
                    if not _spend(1):
                        return None
                    dw = float(np.linalg.norm(poly[w] - xyz))
                    if dw < d:
                        v, d = w, dw
                        improved = True
                        break
        return v

    def _process_pair(k, l):
        if not _spend(1):
            return
        d, s_, t_ = seg_dist(A[k], B[k], A[l], B[l])
        if d > 5.0 * atol:          # ledger L23 (was 2*atol, below the
            return                  # 4*atol worst-case chord-pair gap)
        a4 = (1 - s_) * S4a[k] + s_ * S4b[k]
        b4 = (1 - t_) * S4a[l] + t_ * S4b[l]
        candidate_hits = solve_candidate(a4, b4)
        if _ledger.exhausted:
            return
        for stuv, mate, xyz in candidate_hits:
            ak = _anchor_vertex(int(br[k]), int(ix[k]), xyz)
            al = _anchor_vertex(int(br[l]), int(ix[l]), xyz)
            if _ledger.exhausted or ak is None or al is None:
                return
            links = [(int(br[k]), ak), (int(br[l]), al)]
            dup = None
            for h in found:
                if not _spend(1):
                    return
                if _c3_same_hit(h["stuv"], h["stuv_mate"], h["xyz"],
                                stuv, mate, xyz, atol, ptol4):
                    dup = h
                    break
            if dup is not None:
                for ln in links:
                    if ln not in dup["links"]:
                        dup["links"].append(ln)
                continue
            found.append({"stuv": stuv, "stuv_mate": mate, "xyz": xyz,
                          "links": links})

    # Stream bounded square tiles of the upper triangle.  The old code
    # appended every surviving pair from every Mx1024 broadcast and only
    # then began exact work, so a dense polyline could allocate O(M^2)
    # memory before any soft limit had a chance to fire.  Each tile is
    # charged *before* its AABB broadcast; denial leaves it wholly
    # unprocessed and therefore makes the returned hit set explicitly
    # partial.
    block = 256
    stop = False
    for r0 in range(0, M, block):
        r1 = min(r0 + block, M)
        for c0 in range(r0, M, block):
            c1 = min(c0 + block, M)
            nr, nc = r1 - r0, c1 - c0
            pair_count = (nr * (nr - 1) // 2
                          if c0 == r0 else nr * nc)
            if pair_count <= 0:
                continue
            # Ledger L43: one vectorized AABB pair test costs ~ns; charging
            # it 1:1 against subdivision-cell work (~ms) let ~350k raw
            # pairs burn the whole default allowance on ~10 ms of numpy.
            # Price per-128 like the SSX `precompute` convention; the
            # downstream seg_dist/Newton/anchor/dedup work (the real cost)
            # keeps its 1:1 pricing in `_process_pair`.
            if not _spend(max(1, (pair_count + 127) // 128)):
                stop = True
                break
            pairs_processed += pair_count

            ov = np.all(
                (lo[r0:r1, None, :] <= hi[None, c0:c1, :])
                & (lo[None, c0:c1, :] <= hi[r0:r1, None, :]),
                axis=2,
            )
            ki, li = np.nonzero(ov)
            ki = ki + r0
            li = li + c0
            if c0 == r0:
                keep = li > ki      # unordered pairs once on diagonal tile
                ki, li = ki[keep], li[keep]
            same = br[ki] == br[li]
            keep = ~same | (np.abs(ix[ki] - ix[li]) >= 3)
            ki, li = ki[keep], li[keep]
            candidate_pairs += len(ki)
            for k, l in zip(ki, li):
                _process_pair(int(k), int(l))
                if _ledger.exhausted:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    _publish_stats()
    return found

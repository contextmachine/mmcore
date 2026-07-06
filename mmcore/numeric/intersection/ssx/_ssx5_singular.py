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


@dataclass
class BoxNet:
    """A scalar Bernstein net over a sub-box of [0,1]^4.

    `coeffs` has one tensor dim per entry of `axes` plus a trailing value
    dim of size 1. `axes[i]` is the global axis the i-th tensor dim varies
    along; restriction along a global axis not in `axes` is a no-op (the
    net is constant along axes it doesn't depend on).
    """
    coeffs: NDArray[np.float64]
    axes: tuple

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


def solve_zero_dim(
    nets: list,                       # list[BoxNet] — ALL must contain 0 for a box to survive
    newton: Callable,                 # (x0: (4,)) -> Optional[(4,) solution in GLOBAL coords]
    ptol,                             # (4,) per-axis parametric resolution
    box=((0.0, 1.0),) * 4,
    max_cells: int = 20000,
    dedup_xyz: Optional[Callable] = None,   # (sol) -> (3,) point, for xyz dedup
    atol: float = 1e-3,
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
    """
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

    stack = [(tuple(box), list(nets))]
    cells = 0
    while stack and cells < max_cells:
        cells += 1
        bx, bnets = stack.pop()
        if any(n.excludes_zero() for n in bnets):
            continue
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
        stack.append((tuple(bl), left_nets))
        stack.append((tuple(br), right_nets))
    return sols, bool(stack)

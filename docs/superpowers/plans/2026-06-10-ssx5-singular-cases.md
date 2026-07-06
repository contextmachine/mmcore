# SSX v5 Singular Cases (C₁/C₂/C₃) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Detect, certify, and report all three fundamental SSI singularity classes of Cheng et al. 2023 (IATA, `3592452-2.pdf`) — C₁ parameterization cusps, C₂ tangencies (isolated points, points-with-branches, tiny loops), C₃ 3D self-intersections — in `bez_ssx`, with typed output and no regressions on the 11 existing validated cases.

**Architecture:** Everything reduces to certifying/solving small algebraic systems over Bernstein nets we already carry or can build by outer products: Ψ (vector residual net), TΨ¹…⁴ (already per-cell), Σ₁/Σ₂ (surface normal nets), plus degree-1 hyperplane nets. One new module `_ssx5_singular.py` provides the nets and a zero-dimensional Bernstein subdivision solver (hull-exclusion + Newton — the proven CSX phase-2 pattern lifted to 4D). `_bez_ssx5.py` gains: a fixed tangency gate (isolated tangent points currently vanish), Φ∩L seeding for tangency-adjacent loops, a global C₁ pass, a post-trace C₃ pass, and a `singularities` output list.

**Tech Stack:** numpy; existing mmcore primitives: `bern.py` (`de_casteljau_split_nd`, `bernstein_eval_nd`, `bernstein_partial_derivative_coeffs`), `_deflate.py` (`bernstein_patch_derivative_s/t`, `bernstein_patch_cross_same_params`, `minors_Tpsi_from_control_nets`, `gauss_newton_witness`), `_bezier_common.py` (`eval_surface`, `eval_surface_d1`).

**Paper → code map (read `3592452-2.pdf` §3–5 first):**
- C₁ (§4.1, Fig 5/6): cusp of R₁'s parameterization on the curve; system Δ = Ψ ∩ Σ₁, Σ₁(s,t) = ∂R₁/∂s × ∂R₁/∂t = 0. At C₁ points T³_Ψ = T⁴_Ψ = 0 but T¹,T² ≠ 0: **the 4D curve is regular** (our marcher already walks through; we only need to locate/report).
- C₂ (§4.2, Fig 7/8): tangency, T_Ψ = 0; deflation Δ = Ψ ∩ T_Ψ (exists: `_check_tangency`), regulated Φ = {Ψ_a, Ψ_b, TΨ_k} (exists: `_march_phi_curve`). Missing: isolated tangent points and tiny loops via Φ ∩ L (§5.3.2, Fig 9) — L is a hyperplane; it can miss a 0-dim point but reliably cuts the 1-dim Φ curve, which passes through every isolated tangency and meets every loop ≥ 2× (Lemma 2).
- C₃ (§4.3, Fig 11/12): distinct 4D preimages, same 3D point. Exclusion: Theorem 3 — one sign-definite minor from {T¹,T²} AND one from {T³,T⁴} ⇒ injective in the box. Detection: square 6-var Newton {R₁(s,t)=R₂(u,v), R₁(p,q)=R₂(u,v)} seeded from crossing branch segments. Paper runs C₃ **after** tracing (§5.4) — we do the same as a post-pass.
- Known paper weaknesses we deliberately avoid (§7.1): random hyperplanes can miss small features (we sweep deterministic axis mid-planes instead); interval-Newton cost (we use hull-exclusion subdivision, our measured-fast pattern).

**File structure:**
- Create: `mmcore/numeric/intersection/ssx/_ssx5_singular.py` — nets (Ψ vector, Σ, linear), `BoxNet`, `solve_zero_dim`, C₁/C₃ pass entry points, Φ∩L seeding helper. One responsibility: singularity algebra. No tracing code here.
- Modify: `mmcore/numeric/intersection/ssx/_bez_ssx5.py` — tangency gate, Φ∩L call, passes wiring, output schema.
- Modify: `mmcore/numeric/intersection/ssx/_ssx4.py` — `SSXBranch.kind` field (default keeps old behavior).
- Test: `tests/test_bez_ssx5_singular.py` — all new cases; `examples/ssx/bez_ssx5_coverage_check.py` untouched (used in final sweep).

**Tolerance conventions (established this project, do not deviate):** parametric radius `ptol` = `bez_surface_param_tolerance(S, atol)` per axis; destructive dedup = 1·ptol AND xyz ≤ atol; matching/unification = 4·ptol box AND xyz ≤ 2·atol; every solver is budget-bounded (max_cells), never unbounded.

---

### Task 1: Output schema — `SSXSingularity`, `SSXBranch.kind`, `result['singularities']`

**Files:**
- Modify: `mmcore/numeric/intersection/ssx/_ssx4.py` (SSXBranch dataclass)
- Modify: `mmcore/numeric/intersection/ssx/_bez_ssx5.py` (dataclass + result dict)
- Test: `tests/test_bez_ssx5_singular.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bez_ssx5_singular.py::test_result_has_singularities_key_and_branch_kind -v`
Expected: FAIL (`ImportError: cannot import name 'SSXSingularity'`)

- [ ] **Step 3: Implement**

In `_ssx4.py`, add to the `SSXBranch` dataclass (keep every existing field; add with default so old constructors work):

```python
    kind: str = "transversal"   # 'transversal' | 'tangential' | 'overlap'
```

and in `SSXBranch.__post_init__`/constructor path (if `overlap=True` is passed anywhere) leave `overlap` untouched — `kind` is additive. In `_bez_ssx5.py` next to `BoundaryPoint`:

```python
@dataclass
class SSXSingularity:
    """A certified singular feature of the SSI (Cheng et al. 2023 C1/C2/C3).

    kind:
      'tangent_point'      — C2: T_Psi = 0, isolated or on branches
      'cusp'               — C1: surface-parameterization cusp on the curve
      'cusp_curve'         — C1 infinite case: samples of a singular curve
      'self_intersection'  — C3: two 4D preimages, one 3D point
    """
    kind: str
    stuv: NDArray[np.float64]                    # (4,) primary preimage
    xyz: NDArray[np.float64]                     # (3,)
    stuv_mate: Optional[NDArray[np.float64]] = None   # (4,) C3 second preimage
    branch_links: list = field(default_factory=list)  # [(branch_index, vertex_index)]
    samples: Optional[NDArray[np.float64]] = None     # (N,4) for 'cusp_curve'
```

In `bez_ssx`: create `all_singularities: list = []` next to `all_fragments`, set `kind="overlap"` on overlap branches in `_overlaps_to_branches` (it already sets `overlap=True` — add `kind="overlap"`), set `kind="tangential"` on branches whose fragments came from `_deflate_tangent_cell` (tag the `_Fragment` with `tangential: bool = False` field; propagate in `_assemble_fragments`: a chained branch is `tangential` if ANY of its fragments is), and return `{'branches': ..., 'points': ..., 'singularities': all_singularities}`.

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_bez_ssx5_singular.py -v`
Expected: PASS

- [ ] **Step 5: Regression check + commit**

Run: `.venv/bin/python examples/ssx/bez_ssx5_coverage_check.py` (expect 7×100% coverage, unchanged)
```bash
git add mmcore/numeric/intersection/ssx/_ssx4.py mmcore/numeric/intersection/ssx/_bez_ssx5.py tests/test_bez_ssx5_singular.py
git commit -m "feat(ssx5): typed singularity output schema (SSXSingularity, branch kinds)"
```

---

### Task 2: `_ssx5_singular.py` — nets + zero-dimensional Bernstein solver

> **AS-BUILT (committed `3fb0d7e` + `21b8ac8` + `33aa60b`) — supersedes the sketch below:** `solve_zero_dim` returns a 2-tuple `(sols, exhausted)` — `exhausted=True` means the max_cells budget ran out with boxes unexplored, so the enumeration may be INCOMPLETE and absence is not proof. The resolution floor is ratio-based (stop splitting when `max_i span_i/ptol_i <= 1`; split axis = arg-max ratio). `BoxNet` is `frozen=True, eq=False`. Degenerate inputs raise `ValueError` (empty `nets`; degree-0 patches into `sigma_normal_net`). Rational `sigma_normal_net` is fully implemented (identity `R_a x R_b = N_hom / w^4`). All later-task snippets below have been updated to the tuple API — copy them as shown.

**Files:**
- Create: `mmcore/numeric/intersection/ssx/_ssx5_singular.py`
- Test: `tests/test_bez_ssx5_singular.py`

- [ ] **Step 1: Write the failing tests**

```python
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
        assert abs(float(bernstein_eval_nd(L, pt)) - want) < 1e-13


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

    sols = solve_zero_dim(nets, newton, ptol=np.full(4, 1e-5), max_cells=5000)
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
```

- [ ] **Step 2: Run to verify failure** (`ModuleNotFoundError`)

- [ ] **Step 3: Implement `_ssx5_singular.py`**

```python
"""Singularity algebra for bez_ssx v5 (Cheng et al. 2023, C1/C2/C3).

Bernstein nets for the SSI systems and a budget-bounded zero-dimensional
subdivision solver (hull exclusion + Newton — the proven CSX phase-2
pattern lifted to 4D). Nets may depend on a SUBSET of the 4 axes
(e.g. Sigma_1 depends only on (s,t)); `BoxNet.axes` records the mapping
from the net's own dims to global axes so restriction skips foreign axes.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np
from numpy.typing import NDArray

from mmcore.numeric.bern import de_casteljau_split_nd
from mmcore.numeric.intersection._deflate import (
    bernstein_patch_derivative_s, bernstein_patch_derivative_t,
    bernstein_patch_cross_same_params,
)


def psi_vector_net(S1_h, S2_h):
    """Bernstein net of Psi = R1·w2 − R2·w1, shape (m1,n1,m2,n2,3).

    (s,t) and (u,v) are disjoint variables, so the products are outer
    products — exact, no degree elevation (same trick as the CSX G-net)."""
    P1, w1 = S1_h[..., :-1], S1_h[..., -1]
    P2, w2 = S2_h[..., :-1], S2_h[..., -1]
    return (P1[:, :, None, None, :] * w2[None, None, :, :, None]
            - P2[None, None, :, :, :] * w1[:, :, None, None, None])


def linear_net_4d(c0: float, coeffs: Sequence[float]):
    """Degree-(1,1,1,1) Bernstein net of L(x) = c0 + coeffs·x on [0,1]^4."""
    L = np.empty((2, 2, 2, 2, 1))
    for idx in np.ndindex(2, 2, 2, 2):
        L[idx] = c0 + sum(coeffs[i] * idx[i] for i in range(4))
    return L


def sigma_normal_net(S_h, rational: bool):
    """Bernstein net (2D, per-(s,t)) of the surface normal numerator.

    Polynomial case: N = dP/ds × dP/dt.
    Rational case:   N_hom = (P_s w − P w_s) × (P_t w − P w_t)
    (the numerator of the rational normal; zeros coincide with normal
    vanishing since w > 0). Uses _deflate.py's exact Bernstein product
    primitives, which operate on nested lists of 3-vectors."""
    if not rational:
        P = S_h.tolist()
        Ns = bernstein_patch_derivative_s(P)
        Nt = bernstein_patch_derivative_t(P)
        return np.asarray(bernstein_patch_cross_same_params(Ns, Nt), dtype=np.float64)
    # homogeneous: build A = P_s w − P w_s and B = P_t w − P w_t as
    # Bernstein products via the same-parameter product machinery:
    # cross(A, B) where A, B computed with degree-elevated products.
    P = S_h[..., :-1]
    w = S_h[..., -1]
    Pw = (P * w[..., None])          # net of P·w (POINTWISE coeff product is
    # NOT the Bernstein product — build products properly:
    from mmcore.numeric.intersection._deflate import cross3, vsub  # noqa: F401
    # A = d/ds (P/w) numerator = P_s w − P w_s ; both terms are products of
    # two Bernstein patches in the SAME variables -> use the generic
    # same-param product by promoting scalars to 3-vectors is wasteful;
    # implement the scalar×vector Bernstein product directly:
    def _same_param_product(A, b):
        """Bernstein product of vector patch A (ma,na,3) and scalar patch b (mb,nb).
        Exact convolution with binomial weights."""
        from math import comb
        ma, na = A.shape[0] - 1, A.shape[1] - 1
        mb, nb = b.shape[0] - 1, b.shape[1] - 1
        out = np.zeros((ma + mb + 1, na + nb + 1, 3))
        den = np.zeros((ma + mb + 1, na + nb + 1))
        for i in range(ma + 1):
            for j in range(na + 1):
                for k in range(mb + 1):
                    for l in range(nb + 1):
                        wgt = comb(ma, i) * comb(mb, k) / comb(ma + mb, i + k) \
                            * comb(na, j) * comb(nb, l) / comb(na + nb, j + l)
                        out[i + k, j + l] += wgt * A[i, j] * b[k, l]
        return out
    Ps = np.asarray(bernstein_patch_derivative_s(P.tolist()), dtype=np.float64)
    Pt = np.asarray(bernstein_patch_derivative_t(P.tolist()), dtype=np.float64)
    ws = np.asarray(bernstein_patch_derivative_s(w[..., None].tolist()), dtype=np.float64)[..., 0]
    wt = np.asarray(bernstein_patch_derivative_t(w[..., None].tolist()), dtype=np.float64)[..., 0]
    A = _same_param_product(Ps, _elev_to(w, Ps)) - _same_param_product(_elev_vec_to(P, ws), ws)
    # NOTE: implementer — the two operands of the subtraction must share a
    # degree; use the max degree of both products and elevate the lower one
    # (write _elev_to/_elev_vec_to with the standard degree-elevation
    # convolution). Validate against finite differences in the unit test —
    # test_sigma_net_matches_fd below is the acceptance gate.
    B = _same_param_product(Pt, _elev_to(w, Pt)) - _same_param_product(_elev_vec_to(P, wt), wt)
    A, B = _match_degrees(A, B)
    return np.asarray(bernstein_patch_cross_same_params(A.tolist(), B.tolist()), dtype=np.float64)


@dataclass
class BoxNet:
    """A scalar Bernstein net over a sub-box of [0,1]^4.

    `coeffs` has one tensor dim per entry of `axes` plus a trailing value
    dim of size 1. `axes[i]` is the global axis the i-th tensor dim varies
    along; restriction along a global axis not in `axes` is a no-op."""
    coeffs: NDArray[np.float64]
    axes: tuple

    def excludes_zero(self) -> bool:
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

    Hull-exclusion subdivision + center-seeded Newton, budget-bounded.
    Dedup: 1·ptol per-axis box AND (if dedup_xyz given) xyz <= atol —
    the established destructive-dedup convention."""
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
        spans = [hi - lo for lo, hi in bx]
        widest = int(np.argmax(spans))
        if spans[widest] <= float(ptol[widest]):
            continue      # resolution floor
        # split the widest axis; local split parameter is 0.5 of the SUB-box,
        # which is 0.5 in each net's own (already restricted) domain
        left_nets, right_nets = [], []
        for n in bnets:
            l, r = n.split(widest, 0.5)
            left_nets.append(l); right_nets.append(r)
        m = 0.5 * (bx[widest][0] + bx[widest][1])
        bl = list(bx); bl[widest] = (bx[widest][0], m)
        br = list(bx); br[widest] = (m, bx[widest][1])
        stack.append((tuple(bl), left_nets))
        stack.append((tuple(br), right_nets))
    return sols
```

**Implementation notes for the executor (read before coding):**
1. `solve_zero_dim`'s Newton runs in GLOBAL coordinates while the nets are restricted per box — the nets are only used for exclusion, Newton uses smooth evaluators. That's why the same solution may be found from several boxes: the `_dup` check handles it (CSX-proven pattern).
2. The rational branch of `sigma_normal_net` is the only genuinely fiddly code (degree matching). Write `_elev_to`, `_elev_vec_to`, `_match_degrees` as standard Bernstein degree-elevation (convolution with binomial weights); the acceptance gate is the finite-difference test below. If time-boxed: land the polynomial branch first (all planned tests are polynomial), raise `NotImplementedError` for rational with a TODO, and file the rational branch as a follow-up commit inside this task.
3. Add this test alongside the others:

```python
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
```

- [ ] **Step 4: Run all Task-2 tests** → PASS
- [ ] **Step 5: Commit** `feat(ssx5): singularity nets + zero-dim Bernstein solver`

---

### Task 3: C₂ — isolated tangent points (fix the gate that loses them)

> **AS-BUILT (committed `23e39bc` + `237c8c9` + `e1db506`) — supersedes the snippet below in three ways:** (1) `_tangency_witness(cell, atol)` returns `(ok, roots, best_fn)` — center GN witness first, then `solve_zero_dim` enumeration of ALL Δ-roots in the cell (a single-witness `continue` demonstrably lost the second of two touches sharing one crossing-less cell); every root goes through the emission dedup. (2) The `continue` is size-gated, NOT unconditional: fall through to subdivision unless `ok` AND all four GLOBAL cell spans ≤ 4·unify_tol — an unconditional `continue` deleted coexisting transversal features in the same cell (Mexican hat `z=q(q−1/2)`: touch + transversal ring; Φ∩L cannot recover the ring — it is transversal, not on Φ). A failed witness also falls through (never drop a cell with neither emission nor subdivision). (3) A post-assembly filter drops micro-branches at emitted tangent points (every vertex ≤ 4·atol xyz of a tangent point AND arc ≤ 16·atol) — subdividing past the touch re-exposes CSX grazing-valley micro-fragments. Open question deliberately left to Task 4: near-touch SSXPoints (~1·atol from the touch) now surface in `result['points']` and need the analogous filter.

**Context:** `_bez_ssx5.py` main loop sets `is_clearly_transversal = True` when `cell.crossings` is empty — so a cell containing ONLY an isolated tangency never runs `_check_tangency`, subdivides to `max_depth`, and emits nothing. Verified live: paraboloid-touching-plane returns `{}` today.

**Files:**
- Modify: `mmcore/numeric/intersection/ssx/_bez_ssx5.py` (main-loop tangency gate)
- Test: `tests/test_bez_ssx5_singular.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify failure** (0 singularities today)

- [ ] **Step 3: Implement.** In the main loop of `bez_ssx`, replace the gate

```python
        is_clearly_transversal = False
        if not cell.crossings:
            is_clearly_transversal = True
        else:
```
with
```python
        is_clearly_transversal = False
        if not cell.crossings:
            # An isolated tangency or interior tangent loop lives in exactly
            # this kind of cell (no boundary crossings). Whether tangency is
            # even possible is already known for free: _check_monotonicity
            # failed (we are past the loop-free gate), i.e. all four T-Psi
            # hulls straddle zero. Run the tangency check instead of
            # assuming transversality — assuming it silently deleted
            # isolated tangent points (paper Fig. 24/25 class).
            pass
        else:
```
(the `for c in cell.crossings:` normal-angle pre-check stays as-is for crossing-bearing cells). Then extend the `tangency is True` branch: currently it requires `cell.crossings`; add the crossing-less case BEFORE it:

```python
        if tangency is True and not cell.crossings:
            # Isolated tangent point (or tangent feature with no boundary
            # contact). The Gauss-Newton witness from _check_tangency is the
            # point — recompute it here to get coordinates (cheap, the cell
            # is tiny by now) and emit a typed singularity.
            ok, xw, fn = _tangency_witness(cell)
            if ok:
                stuv_g = _local_to_global(np.asarray(xw), cell.box)
                xyz_w = eval_surface(cell.g1.surface, xw[0], xw[1], rational=True)
                if not any(g.kind == "tangent_point"
                           and np.all(np.abs(g.stuv - stuv_g) <= unify_tol)
                           and float(np.linalg.norm(g.xyz - xyz_w)) <= 2.0 * atol
                           for g in all_singularities):
                    all_singularities.append(SSXSingularity(
                        kind="tangent_point", stuv=stuv_g, xyz=xyz_w))
            continue
```

with the helper (place next to `_check_tangency`):

```python
def _tangency_witness(cell):
    """Gauss-Newton witness point of the deflated system Delta = Psi ∩ T_Psi
    on the cell (local coords). Returns (ok, x_local(4,), residual)."""
    from mmcore.numeric.bern import bern_eval as _bern_eval
    from mmcore.numeric.ndinterval import interval as iv_interval, get_iarray
    from mmcore.numeric.intersection._deflate import (
        DeflatedSystem, gauss_newton_witness, _box_from_any,
    )
    P1c = cell.g1.surface[..., :-1] / cell.g1.surface[..., -1:]
    P2c = cell.g2.surface[..., :-1] / cell.g2.surface[..., -1:]
    try:
        sys_ = DeflatedSystem(
            P1=get_iarray(P1c, P1c), P2=get_iarray(P2c, P2c),
            T=tuple(np.asarray(T, dtype=iv_interval) for T in (cell.T1, cell.T2, cell.T3, cell.T4)),
            bern_eval=_bern_eval, interval_ctor=iv_interval,
        )
        Bf = _box_from_any(tuple(iv_interval(0.0, 1.0) for _ in range(4)))
        ok, xw, fn = gauss_newton_witness(sys_, Bf, tol_f=1e-10, max_iter=24)
        return ok, xw, fn
    except Exception:
        return False, None, np.inf
```

**Checkpoint (manual, before Step 4):** instrument once and confirm on the paraboloid pair that (a) the touch cell reaches the tangency branch (not pruned by F_sq — the touch means min distance 0, so it can't be pruned), (b) exactly ONE `tangent_point` is emitted after the per-cell dedup vs `all_singularities` (neighboring cells will also confirm tangency near the touch; the `unify_tol`+2·atol dedup collapses them).

- [ ] **Step 4: Run the test** → PASS. Also `pytest tests/test_bez_ssx5_singular.py -v` all green.
- [ ] **Step 5: Full regression** — coverage harness 7×100%, legacy 4 mini-cases OK, `test_bez_csx4`/`test_bez_ccx4` green.
- [ ] **Step 6: Commit** `feat(ssx5): C2 isolated tangent points (fix crossing-less tangency gate)`

---

### Task 4: C₂ — tangent point WITH transversal branches (saddle X-crossing)

> **AS-BUILT (committed `79fa26a` + `5750bdd`):** Step 2 took a THIRD path the plan didn't predict: an on-lattice touch (saddle center at 0.5) lands exactly on the midpoint-cut lattice, so its cells are certified loop-free via NON-STRICT monotone T-nets (hull touches 0 at the corner) and traced directly — `_check_tangency` never runs. Emission therefore happens at THREE sites through one factored `_emit_tangent_roots` helper (same dedup everywhere): crossing-less arm (full `solve_zero_dim` enumeration, multi-touch safe), LOOP-FREE arm (gated by all four T-hulls containing 0; the site that fires on the saddle), crossing-bearing arm. The two new sites run `enumerate_all=False` (center GN witness only, ~1ms) — full enumeration there cost 16× on the tangential legacy case (69 spurious points along the tangent curve) and ~2s on near-tangent coverage cells. Post-assembly, two subsumption filters: near-touch SSXPoints within 2·atol xyz of an emitted tangent_point are dropped (measured: all debris ≤1·atol), and tangent_points lying within 4·atol of a tangential/overlap branch polyline are dropped (a tangent CURVE subsumes its pointwise re-confirmations; also kills a spurious corner singularity on legacy overlaps). Known accepted limits: off-lattice saddle (touch at 0.55) truncates one arm ~0.05 short of the X (pre-existing marcher behavior, identical with emission disabled — arm-through-singularity is plan-capped); multi-touch cells on the loop-free/crossing-bearing paths find one root per cell (neighbor cells' centers must cover the rest; crossing-less multi-touch keeps full enumeration).

**Files:**
- Test: `tests/test_bez_ssx5_singular.py` (+ small fixes in `_bez_ssx5.py` only if the test exposes them)

- [ ] **Step 1: Write the test**

```python
def _saddle_touch():
    """S1: z = (2s-1)^2 - (2t-1)^2 saddle; S2: z=0 plane.
    SSI = two straight lines s=t and s=1-t crossing at the tangent point (0.5,0.5)."""
    zc = [1.0, -1.0, 1.0]; xs = [0.0, 0.5, 1.0]
    S1 = np.array([[[xs[i], xs[j], zc[i] - zc[j]] for j in range(3)] for i in range(3)])
    S2 = np.array([[[-0.5, -0.5, 0.], [-0.5, 1.5, 0.]],
                   [[1.5, -0.5, 0.], [1.5, 1.5, 0.]]])
    return S1, S2


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


def _pt_poly(p, poly):
    a, b = poly[:-1], poly[1:]
    ab = b - a
    den = np.einsum("ij,ij->i", ab, ab); den[den < 1e-30] = 1e-30
    tt = np.clip(np.einsum("ij,ij->i", p[None] - a, ab) / den, 0, 1)
    return float(np.linalg.norm(a + tt[:, None] * ab - p[None], axis=1).min())
```

- [ ] **Step 2: Run it.** Two possible outcomes:
  - PASS → proceed to Step 4 (the boundary crossings of the 4 arms exist, tracing marches through/near the X; the tangent point is emitted by a neighboring crossing-bearing tangent cell via the Task-3 path — the `tangency is True and cell.crossings` branch already runs `_deflate_tangent_cell`; ADD the same witness emission there: one `tangent_point` singularity appended, deduped).
  - FAIL → diagnose which half fails:
    * missing `tangent_point`: extend the crossing-bearing tangency branch (`tangency is True and cell.crossings`) to ALSO emit the witness singularity exactly as in Task 3 (same dedup). This is expected to be required — write it directly.
    * missing arm coverage: the marcher stalls AT the X (rank-2 Jacobian). The corrector's `direction_hint` projection already handles null_dim=2; if arms are truncated, the acceptable v1 resolution per the paper is that arms END at the singular point: check that each arm's fragment ends within `4·ptol` of the X and, if the end is interior-truncated NEAR the X, synthesize the endpoint AT the witness point (reuse the synthesized-BoundaryPoint mechanism with the witness stuv). Cap this fix at that — do NOT attempt smart arm-pairing through the singularity (topologically ambiguous; paper also terminates branches at singular points).

- [ ] **Step 3: Implement whichever fix Step 2 demanded** (code above).
- [ ] **Step 4: Full regression sweep** (coverage 7×100% + legacy 4 + tangential legacy case still 1 branch @ ≤1e-9).
- [ ] **Step 5: Commit** `feat(ssx5): C2 tangent points coexisting with transversal branches`

---

### Task 5: C₂ — tiny loops near tangency via Φ ∩ L seeding

> **AS-BUILT (committed `2d030bb` + `7ed47c0`):** (1) `phi_loop_seeds` uses a Levenberg-damped Newton instead of the sketch's plain solve — symmetric geometries put mid-plane box centers exactly on degenerate manifolds where plain solve raises. (2) The load-bearing filter is `_phi_slice_loop_fragments`' full-Ψ Gauss-Newton refinement of every seed (rejects the sub-tolerance valley-floor ring at |Ψ|=ε²/4 < atol that would march a phantom loop at the wrong radius); backend per refined seed: sin_ang > 1e-3 → ordinary Ψ marcher (transversal loop — Φ only MEETS such loops at T_k extremes, plan risk 2), else Φ marcher. (3) `_march_closed_from_seed` shared closed-loop engine: arrival check armed only outside 3× the displacement radius; displacement sign picks the branch (risk-2 flip retry); Ψ-marched closures passing within 2·atol of an emitted tangent point are through-the-singularity artifacts (flipped, then discarded). (4) DEFECT-D (found post-commit): the crossing-BEARING arm's `continue` deleted a coexisting isolated touch (repro `z=(2t−1)²((s−0.7)²+(t−0.2)²)`: witness converges into the CURVE's basin; the cell is the only holder). Subdivide-until-tolerance costs 1349× on tangent curves (legacy crossed-saddles 0.15s→3m23s), so instead `_emit_offcurve_tangent_roots` enumerates the cell's REMAINING Δ-roots in place: `solve_zero_dim` with Newton SKIPPED inside the traced fragments' tube (param 4·ptol + box radius AND xyz 4·atol) and far-from-tube boxes explored FIRST (max-heap priority; budget exhaustion starves only the curve flood). `solve_zero_dim` gained `skip_newton`/`priority` hooks. Measured: touch found exactly, legacy tangential 4.2s (known cost: budget-bounded flood on the tangent-curve top cell — profiling follow-up option: KD-tree cover queries).

**Context (paper §5.3.2, Fig 9):** in a tangent cell, a tiny Ψ-loop around the tangency has no boundary crossings and can be smaller than subdivision reaches by `max_depth`. The regulated Φ curve passes through the tangency AND crosses the loop's neighborhood; slicing Φ with a mid-plane L yields seed points from which the loop is marched directly.

**Files:**
- Modify: `mmcore/numeric/intersection/ssx/_ssx5_singular.py` (add `phi_loop_seeds`)
- Modify: `mmcore/numeric/intersection/ssx/_bez_ssx5.py` (call in the crossing-less tangency branch; closed-loop Φ marching)
- Test: `tests/test_bez_ssx5_singular.py`

- [ ] **Step 1: Write the failing test**

```python
def _touch_plus_loop(eps=0.04):
    """S1: z = r^4 - eps*r^2 with r^2=(2s-1)^2+(2t-1)^2  (deg 4x4);
    S2: z=0. SSI: tangent point at r=0 PLUS transversal loop at r=sqrt(eps).
    Paper Fig. 24 (Example 11) analog."""
    import numpy.polynomial.polynomial as npoly
    # 1D Bernstein coeffs of p(x) on [0,1] via values at Greville-free exact
    # conversion: monomial->Bernstein with the standard lower-triangular map.
    def mono_to_bern(a):                     # a: monomial coeffs, len n+1
        n = len(a) - 1
        from math import comb
        b = np.zeros(n + 1)
        for k in range(n + 1):
            b[k] = sum(a[j] * comb(k, j) / comb(n, j) for j in range(k + 1))
        return b
    # z(s,t) = (f(s)+f(t))^2 - eps*(f(s)+f(t)), f(x)=(2x-1)^2 — expand in
    # tensor monomials then convert per axis:
    f = np.array([1.0, -4.0, 4.0])                       # (2x-1)^2 monomial
    # (f(s)+f(t))^2 = f(s)^2 + 2 f(s) f(t) + f(t)^2 ; f^2 monomial:
    f2 = np.convolve(f, f)                               # degree 4
    z_st = np.zeros((5, 5))
    z_st[:5, 0] += f2; z_st[0, :5] += f2
    z_st[:3, :3] += 2.0 * np.outer(f, f)
    z_st[:3, 0] -= eps * f; z_st[0, :3] -= eps * f
    Bs = np.zeros((5, 5))
    M = np.array([mono_to_bern(np.eye(5)[j]) for j in range(5)])   # rows: x^j in Bernstein deg 4
    Bz = M.T @ z_st @ M                                   # tensor conversion
    xs = mono_to_bern([0.0, 1.0] + [0.0] * 3)             # x = s in deg-4 Bernstein
    S1 = np.array([[[xs[i], xs[j], Bz[i, j]] for j in range(5)] for i in range(5)])
    S2 = np.array([[[-0.5, -0.5, 0.], [-0.5, 1.5, 0.]],
                   [[1.5, -0.5, 0.], [1.5, 1.5, 0.]]])
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
```

- [ ] **Step 2: Run to verify failure** (loop missed or only found for large eps via subdivision).

- [ ] **Step 3: Implement `phi_loop_seeds` in `_ssx5_singular.py`:**

```python
def phi_loop_seeds(S1_h, S2_h, T_nets, psi_rows, t_idx, atol, ptol,
                   max_cells=4000):
    """Seed points of the regulated curve Phi = {Psi_a, Psi_b, T_k} sliced by
    deterministic mid-planes (paper 5.3.2, with axis-aligned L instead of
    random hyperplanes — random L can miss small features, admitted in
    their 7.1). Returns list of (4,) local-coordinates seeds.

    Exclusion nets: ALL THREE Psi components + T_k + L (stronger exclusion
    is sound); Newton solves the square {Psi_a, Psi_b, T_k, L}."""
    from mmcore.numeric.bern import bernstein_partial_derivative_coeffs, bernstein_eval_nd
    from mmcore.numeric.intersection._bezier_common import eval_surface_d1

    G = psi_vector_net(S1_h, S2_h)
    Tk = np.asarray(T_nets[t_idx], dtype=np.float64)[..., None]

    def newton_factory(axis, value):
        def newton(x0):
            x = np.asarray(x0, dtype=np.float64).copy()
            for _ in range(30):
                p1, du1, dv1 = eval_surface_d1(S1_h, x[0], x[1], rational=True)
                p2, du2, dv2 = eval_surface_d1(S2_h, x[2], x[3], rational=True)
                psi = p1 - p2
                tval = float(bernstein_eval_nd(Tk, x))
                F = np.array([psi[psi_rows[0]], psi[psi_rows[1]], tval, x[axis] - value])
                if np.linalg.norm(F) < 1e-11:
                    return np.clip(x, 0.0, 1.0)
                Jpsi = np.column_stack([du1, dv1, -du2, -dv2])
                grad_t = np.array([float(bernstein_eval_nd(
                    bernstein_partial_derivative_coeffs(Tk, axis=ax), x)) for ax in range(4)])
                J = np.vstack([Jpsi[psi_rows[0]], Jpsi[psi_rows[1]], grad_t,
                               np.eye(4)[axis]])
                try:
                    x = np.clip(x - np.linalg.solve(J, F), 0.0, 1.0)
                except np.linalg.LinAlgError:
                    return None
            return None
        return newton

    seeds = []
    for axis in range(4):
        nets = [BoxNet(G[..., k:k + 1], axes=(0, 1, 2, 3)) for k in range(3)]
        nets.append(BoxNet(Tk, axes=(0, 1, 2, 3)))
        nets.append(BoxNet(linear_net_4d(-0.5, tuple(np.eye(4)[axis])), axes=(0, 1, 2, 3)))
        ax_sols, ax_exhausted = solve_zero_dim(nets, newton_factory(axis, 0.5), ptol,
                                               max_cells=max_cells, atol=atol)
        # AS-BUILT: solve_zero_dim returns (sols, exhausted). Exhaustion here
        # only degrades seeding redundancy (4 mid-planes, each cutting a loop
        # >= 2x) — seeds already found stay valid; nothing is invalidated.
        for s in ax_sols:
            if not any(np.all(np.abs(s - t) <= np.asarray(ptol)) for t in seeds):
                seeds.append(s)
    return seeds
```

In `_bez_ssx5.py`, in the crossing-less tangency branch (Task 3's code), after emitting the tangent point:

```python
            # Paper 5.3.2: slice the regulated Phi curve with mid-planes to
            # find loops/features around the tangency that have no boundary
            # crossings, then march Phi from each seed and keep the
            # Psi-valid samples (same filtering as _deflate_tangent_cell).
            # NOTE (as-built after 237c8c9): this whole block nests under
            # `if roots:` from the witness ENUMERATION above — Task 3 shipped
            # `_tangency_witness(cell, atol) -> (ok, roots, best_fn)` where
            # `roots` lists ALL tangent points in the cell (center GN witness
            # first, then solve_zero_dim enumeration; ok == bool(roots)).
            # `roots[0]` seeds the Phi-equation choice; without any root,
            # fall back to the cell center np.full(4, 0.5).
            psi_rows, t_idx = _choose_phi_equations(
                P1_cart_local, P2_cart_local,
                [np.asarray(T, dtype=np.float64)[..., None]
                 for T in (cell.T1, cell.T2, cell.T3, cell.T4)],
                np.asarray(roots[0]), rational=False)
            seeds = phi_loop_seeds(
                cell.g1.surface, cell.g2.surface,
                (cell.T1, cell.T2, cell.T3, cell.T4),
                psi_rows, t_idx, atol,
                ptol=_cell_ptol4(cell, atol))
            for seed in seeds:
                frag = _march_phi_closed(cell, seed, psi_rows, t_idx, atol, h_max)
                if frag is not None:
                    all_fragments.append(frag)
            continue
```

with two helpers in `_bez_ssx5.py`:

```python
def _cell_ptol4(cell, atol):
    from mmcore.geom._nurbs_param_tol import bez_surface_param_tolerance
    ps, pt = bez_surface_param_tolerance(cell.g1.surface, atol, rational=True)
    pu, pv = bez_surface_param_tolerance(cell.g2.surface, atol, rational=True)
    return np.maximum(np.array([float(ps), float(pt), float(pu), float(pv)]), 1e-9)


def _march_phi_closed(cell, seed_local, psi_rows, t_idx, atol, h_max):
    """March Phi from a seed with no known endpoint until the path returns to
    its start (closed loop) or exits the cell. Keep only Psi-valid samples
    (|S1-S2| < atol); require >= 6 valid samples and closure within 4·ptol
    to emit a closed tangential fragment; otherwise return None.

    Implementation: call _march_phi_curve with stuv_end=seed (its
    arrival check fires when the loop comes back) after first stepping away:
    take one predictor-corrector step manually to displace the current point
    off the seed, then march toward the seed as the target — the marcher
    walks the long way around the loop. If the first manual step fails,
    return None."""
    T_arrs = [np.asarray(T, dtype=np.float64)[..., None]
              for T in (cell.T1, cell.T2, cell.T3, cell.T4)]
    P1c = cell.g1.surface[..., :-1] / cell.g1.surface[..., -1:]
    P2c = cell.g2.surface[..., :-1] / cell.g2.surface[..., -1:]
    # one displacing step along the Phi tangent
    J = _jac_phi(P1c, P2c, T_arrs[t_idx], psi_rows, *seed_local, rational=False)
    _, _, Vt = np.linalg.svd(J, full_matrices=True)
    x1 = np.clip(seed_local + 0.02 * Vt[-1], 0.0, 1.0)
    stuv_path, xyz_path = _march_phi_curve(
        P1c, P2c, T_arrs[t_idx], psi_rows, x1, np.asarray(seed_local),
        atol=atol, rational=False, h_max=h_max)
    if len(stuv_path) < 6:
        return None
    ptol4 = _cell_ptol4(cell, atol)
    if not np.all(np.abs(stuv_path[-1] - stuv_path[0]) <= 8.0 * ptol4):
        return None
    # Psi-validity filter (same as _deflate_tangent_cell)
    keep = []
    for k in range(len(stuv_path)):
        p1 = eval_surface(P1c, stuv_path[k, 0], stuv_path[k, 1], rational=False)
        p2 = eval_surface(P2c, stuv_path[k, 2], stuv_path[k, 3], rational=False)
        if float(np.linalg.norm(p1 - p2)) < atol:
            keep.append(k)
    if len(keep) < 6:
        return None
    stuv_g = np.array([_local_to_global(stuv_path[k], cell.box) for k in keep])
    xyz_g = xyz_path[np.asarray(keep)]
    return _Fragment(start_point=None, end_point=None,
                     stuv_path=stuv_g, xyz_path=xyz_g, tangential=True)
```

**Checkpoints (do these before declaring the task done):**
1. On the test geometry, print the seeds — expect ≥ 2 (mid-planes s=0.5 and t=0.5 each cut the loop twice; dedup may merge symmetric ones).
2. The loop from Φ-marching must be Ψ-valid along its full length (the r=√eps circle IS on Ψ — the Φ curve coincides with it there). If Ψ-validity filtering fragments the loop, the Φ equations chosen were poor — retry with the second-best `(psi_rows, t_idx)` from `_choose_phi_equations` before giving up.
3. Duplicate protection: neighboring tangent cells may re-find the same loop — the assembly containment dedup (`_drop_duplicate_fragments`) must collapse them; verify only ONE closed branch survives.
4. Closed fragments with `start_point=end_point=None` must survive assembly as standalone closed branches — check `_assemble_fragments` treats a None-None fragment as its own chain (it does: no neighbors to walk), and that the closing-march heuristic does not fire on it (endpoints interior + gap small: it MAY fire and re-march the closing gap — harmless but verify no distortion; if it misbehaves, skip closing-march for fragments tagged `tangential`).

- [ ] **Step 4: Run the new test + tangential legacy case + touch-only test (Task 3)** → all PASS.
- [ ] **Step 5: Full regression sweep.**
- [ ] **Step 6: Commit** `feat(ssx5): C2 tiny loops via deterministic Phi-slice seeding`

---

### Task 6: C₁ — cusp detection (Σ nets, global pass, output)

> **AS-BUILT (committed `5243787`):** as designed, with the synced tuple API (`sols, exhausted`; curve_flag on >12 or exhausted-with->1; exhausted-with-0/1 documented as the truncation blind spot). Wiring runs AFTER the post-assembly branch filters so `branch_links` index the FINAL branch list (the filters drop branches), before the point filters. The Σ-hull precheck confirmed zero-cost on all coverage cases and both legacy minis; the cuspidal-edge branch covers the cusp curve within 5e-3 THROUGH the cusp with no marcher changes (xyz-reparameterized stepping absorbs 3D-speed→0).

**Files:**
- Modify: `mmcore/numeric/intersection/ssx/_ssx5_singular.py` (add `c1_pass`)
- Modify: `mmcore/numeric/intersection/ssx/_bez_ssx5.py` (call after assembly; branch links)
- Test: `tests/test_bez_ssx5_singular.py`

- [ ] **Step 1: Write the failing test**

```python
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
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement `c1_pass` in `_ssx5_singular.py`:**

```python
def c1_pass(S1_h, S2_h, atol, ptol4, max_cells=20000):
    """Global C1 detection (paper Fig. 5): solutions of Psi ∩ Sigma_i.

    Returns (singular_points, curve_flag) where singular_points is a list of
    dicts {stuv, xyz, surface: 1|2} and curve_flag indicates a 1-dimensional
    solution set was detected (many collinear-ish solutions).

    Cheap global precheck: if Sigma_1's component hulls exclude zero over
    [0,1]^2 the surface has no vanishing normal anywhere — skip (this is
    the common case; the pass costs three float comparisons)."""
    from mmcore.numeric.intersection._bezier_common import eval_surface_d1
    out = []
    curve_flag = False
    G = psi_vector_net(S1_h, S2_h)
    for which, (Sh, axes2) in enumerate(((S1_h, (0, 1)), (S2_h, (2, 3))), start=1):
        # Sh is ALWAYS homogeneous here (bez_ssx builds S*_h_top with w=1 for
        # polynomial input). Detect the w≡1 case and use the exact polynomial
        # branch on the Cartesian part; genuinely rational input takes the
        # homogeneous-numerator branch (or raises NotImplementedError if that
        # branch was deferred in Task 2 — never silently skip).
        if np.allclose(Sh[..., -1], 1.0, rtol=0.0, atol=1e-14):
            N = sigma_normal_net(Sh[..., :-1], rational=False)   # (ms,ns,3)
        else:
            N = sigma_normal_net(Sh, rational=True)
        if any(float(N[..., c].min()) > 0.0 or float(N[..., c].max()) < 0.0
               for c in range(3)):
            # a single sign-definite component proves |N| > 0 nowhere-zero?
            # NO — N=0 needs ALL components zero; one nonzero component
            # anywhere-in-the-box excludes N=0 in the whole box. Correct.
            continue
        nets = [BoxNet(G[..., k:k + 1], axes=(0, 1, 2, 3)) for k in range(3)]
        nets += [BoxNet(N[..., c:c + 1], axes=axes2) for c in range(3)]
        nscale = float(np.abs(N).max())

        def newton(x0, _Sh=Sh, _axes=axes2, _ns=nscale):
            # Gauss-Newton on the overdetermined {Psi(3), N(2 params)(3)}
            x = np.asarray(x0, dtype=np.float64).copy()
            for _ in range(40):
                p1, du1, dv1 = eval_surface_d1(S1_h, x[0], x[1], rational=True)
                p2, du2, dv2 = eval_surface_d1(S2_h, x[2], x[3], rational=True)
                psi = p1 - p2
                a, b = (x[_axes[0]], x[_axes[1]])
                _, dua, dvb = eval_surface_d1(_Sh, a, b, rational=True)
                Nv = np.cross(dua, dvb)
                F = np.concatenate([psi, Nv / _ns])
                if np.linalg.norm(psi) < 1e-10 and np.linalg.norm(Nv) < 1e-8 * _ns:
                    return np.clip(x, 0.0, 1.0)
                # Jacobian: psi rows as usual; N rows by finite differences
                # on the two owning axes (exact d(cross) is verbose; FD at
                # 1e-7 is adequate for a Gauss-Newton refiner)
                J = np.zeros((6, 4))
                J[:3, 0], J[:3, 1], J[:3, 2], J[:3, 3] = du1, dv1, -du2, -dv2
                for col, ax in enumerate(_axes):
                    xp = x.copy(); xp[ax] += 1e-7
                    _, duap, dvbp = eval_surface_d1(_Sh, xp[_axes[0]], xp[_axes[1]], rational=True)
                    J[3:, ax] = (np.cross(duap, dvbp) - Nv) / (1e-7 * _ns)
                try:
                    dx, *_ = np.linalg.lstsq(J, -F, rcond=None)
                except np.linalg.LinAlgError:
                    return None
                x = np.clip(x + dx, 0.0, 1.0)
                if np.linalg.norm(dx) < 1e-12:
                    break
            p1 = eval_surface_d1(S1_h, x[0], x[1], rational=True)[0]
            p2 = eval_surface_d1(S2_h, x[2], x[3], rational=True)[0]
            _, dua, dvb = eval_surface_d1(_Sh, x[_axes[0]], x[_axes[1]], rational=True)
            if (np.linalg.norm(p1 - p2) < atol
                    and np.linalg.norm(np.cross(dua, dvb)) < 1e-6 * _ns):
                return np.clip(x, 0.0, 1.0)
            return None

        def _xyz(sol):
            from mmcore.numeric.intersection._bezier_common import eval_surface
            return eval_surface(S1_h, sol[0], sol[1], rational=True)

        sols, exhausted = solve_zero_dim(nets, newton, ptol4, max_cells=max_cells,
                                         dedup_xyz=_xyz, atol=atol)
        if len(sols) > 12 or (exhausted and len(sols) > 1):
            # Many hits, or a truncated enumeration that already found several:
            # treat as a 1-dimensional solution set (cusp curve) — report samples.
            curve_flag = True
            out.append({"surface": which, "curve_samples": np.asarray(sols)})
            continue
        # AS-BUILT NOTE: `exhausted` with 0-1 solutions means the enumeration may
        # be incomplete (deep subdivision with Newton failing near a singularity).
        # Do not treat absence as proof there — the implementer must surface this
        # case (at minimum a comment + report mention; do not silently drop it).
        for s in sols:
            out.append({"surface": which, "stuv": np.asarray(s), "xyz": _xyz(s)})
    return out, curve_flag
```

In `bez_ssx`, after `_assemble_fragments` (and before the points-on-branch filter):

```python
    # --- C1 pass (paper Fig. 5): parameterization cusps on the SSI ---
    ptol4_global = np.maximum(
        np.array([float(_gp_s), float(_gp_t), float(_gp_u), float(_gp_v)]), 1e-9)
    c1_hits, c1_curve = c1_pass(S1_h_top, S2_h_top, atol, ptol4_global)
    for hit in c1_hits:
        if "curve_samples" in hit:
            all_singularities.append(SSXSingularity(
                kind="cusp_curve", stuv=hit["curve_samples"][0],
                xyz=eval_surface(S1_h_top, *hit["curve_samples"][0][:2], rational=True),
                samples=hit["curve_samples"]))
            continue
        links = []
        for bi, b in enumerate(all_branches):
            xyz = np.asarray(b.curve[1])
            d = np.linalg.norm(xyz - hit["xyz"][None, :], axis=1)
            k = int(d.argmin())
            if d[k] <= 4.0 * atol:
                links.append((bi, k))
        all_singularities.append(SSXSingularity(
            kind="cusp", stuv=hit["stuv"], xyz=hit["xyz"], branch_links=links))
```

**Checkpoints:**
1. The global precheck must fire (skip) on ALL existing cases 5–11 and legacy 4 (no case has degenerate parameterizations) — assert zero time regression (harness times within noise).
2. On the cusp test: exactly 1 cusp, branch coverage green. Watch the marcher near the cusp: 3D speed → 0 there (T_C = 0), our `h/speed` clamps to `max_step` in stuv — the deviation checks keep the polyline within tolerance (the round-2 machinery); if the coverage assert fails within 5·atol of the cusp only, accept up to `1e-2` locally and file the cusp-neighborhood sampling refinement as follow-up (the 4D curve is regular; only the 3D image degenerates).

- [ ] **Step 4: Run the test** → PASS.
- [ ] **Step 5: Full regression sweep incl. timing comparison.**
- [ ] **Step 6: Commit** `feat(ssx5): C1 cusp detection via Sigma-net certified pass`

---

### Task 7: C₃ — self-intersections (Theorem-3 certificate + post-trace Newton)

> **AS-BUILT (committed `47e5d36`) — one MEASURED DEVIATION from checkpoint 1:** the Theorem-3 gate does NOT certify the plain bilinear/plane case — the curve traces from the TOP cell, whose T1/T2 hulls touch zero at the domain edge, so a sound strict hull test fails there; the flag fires on all 7 coverage cases for the same reason. Checkpoint 1's expectation ("must pass by the FLAG being False") is unachievable with a sound per-box certificate. Resolution: keep the flag (it is cheap and would certify genuinely definite cells), and make the fired path free — `c3_pass`'s candidate search is a vectorized AABB broadphase over all branch segments (atol-inflated, blocked numpy broadcast, same-branch index gap ≥ 3) so the exact seg-dist + 6-var Newton run only on genuinely near pairs. Instrumented: 1 `c3_pass` call per coverage case, 0 spurious self-intersections, timings bit-identical to the pre-C3 baseline. Umbrella: exactly 1 `self_intersection` at (0,0,0.5), preimages t=0.1464/0.8536 sharing (u,v). Closed-loop wrap seams are rejected by the same-preimage 4·ptol guard, not by the adjacency mask.

**Files:**
- Modify: `mmcore/numeric/intersection/ssx/_ssx5_singular.py` (add `c3_pass`, `theorem3_excludes_c3`)
- Modify: `mmcore/numeric/intersection/ssx/_bez_ssx5.py` (cell flag; post-pass)
- Test: `tests/test_bez_ssx5_singular.py`

- [ ] **Step 1: Write the failing test**

```python
def _umbrella_case():
    """S1: Whitney-umbrella style (a*b, a, b^2), a=2s-1, b=2t-1 (deg 1x2);
    S2: plane z=0.5. SSI: two lines (±0.707*a·? ...) — concretely
    x = a*b with b=±sqrt(0.5): two straight lines through (0,0,0.5),
    crossing there with DIFFERENT (s,t) preimages: a C3 self-intersection
    of the SSI image. Paper Fig. 22 (Example 9) analog."""
    a = [-1.0, 1.0]; bb = [-1.0, 0.0, 1.0]; bsq = [1.0, -1.0, 1.0]
    S1 = np.array([[[a[i] * bb[j], a[i], bsq[j]] for j in range(3)] for i in range(2)])
    S2 = np.array([[[-1.5, -1.5, 0.5], [-1.5, 1.5, 0.5]],
                   [[1.5, -1.5, 0.5], [1.5, 1.5, 0.5]]])
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


def test_theorem3_skips_regular_case():
    # transversal bilinear pair: every traced cell satisfies Theorem 3,
    # so the C3 pass must not even run its pair search.
    s1 = np.array([[[0., 0., 0.], [0., 10., 0.]], [[10., 0., 0.], [10., 10., 10.]]])
    s2 = np.array([[[0., 0., 3.], [0., 10., 3.]], [[10., 0., 3.], [10., 10., 3.]]])
    r = bez_ssx(s1, s2, 1e-3, rational=False)
    assert [g for g in r["singularities"] if g.kind == "self_intersection"] == []
```

- [ ] **Step 2: Run to verify failure.**

- [ ] **Step 3: Implement.** In `_ssx5_singular.py`:

```python
def theorem3_excludes_c3(T1, T2, T3, T4) -> bool:
    """Paper Theorem 3: injectivity of the 3D image over the box. One
    sign-definite minor from {T1,T2} AND one from {T3,T4} suffices."""
    def _definite(T):
        T = np.asarray(T)
        return float(T.min()) > 0.0 or float(T.max()) < 0.0
    return (_definite(T1) or _definite(T2)) and (_definite(T3) or _definite(T4))


def c3_pass(S1_h, S2_h, branches, atol, ptol4):
    """Post-trace C3 detection: crossing branch segments -> square 6-var
    Newton {R1(s,t)=R2(u,v), R1(p,q)=R2(u,v)} -> certified pairs.

    Candidates: pairs of polyline segments (across branches, or within one
    branch at index distance > 2) whose segment-segment distance < 2*atol.
    """
    from mmcore.numeric.intersection._bezier_common import eval_surface, eval_surface_d1

    def seg_dist(p1, p2, q1, q2):
        d1 = p2 - p1; d2 = q2 - q1; r = p1 - q1
        a = d1 @ d1; e = d2 @ d2; f = d2 @ r
        if a < 1e-30 and e < 1e-30:
            return float(np.linalg.norm(r)), 0.0, 0.0
        if a < 1e-30:
            s_ = 0.0; t_ = np.clip(f / e, 0.0, 1.0)
        else:
            c = d1 @ r
            if e < 1e-30:
                t_ = 0.0; s_ = np.clip(-c / a, 0.0, 1.0)
            else:
                b = d1 @ d2; den = a * e - b * b
                s_ = np.clip((b * f - c * e) / den, 0.0, 1.0) if den > 1e-30 else 0.0
                t_ = np.clip((b * s_ + f) / e, 0.0, 1.0)
        cp1 = p1 + s_ * d1; cp2 = q1 + t_ * d2
        return float(np.linalg.norm(cp1 - cp2)), float(s_), float(t_)

    def newton6(z0):
        z = np.asarray(z0, dtype=np.float64).copy()   # (s,t,p,q,u,v)
        for _ in range(40):
            r1a, du1a, dv1a = eval_surface_d1(S1_h, z[0], z[1], rational=True)
            r1b, du1b, dv1b = eval_surface_d1(S1_h, z[2], z[3], rational=True)
            r2, du2, dv2 = eval_surface_d1(S2_h, z[4], z[5], rational=True)
            F = np.concatenate([r1a - r2, r1b - r2])
            if np.linalg.norm(F) < 1e-11:
                break
            J = np.zeros((6, 6))
            J[:3, 0], J[:3, 1], J[:3, 4], J[:3, 5] = du1a, dv1a, -du2, -dv2
            J[3:, 2], J[3:, 3], J[3:, 4], J[3:, 5] = du1b, dv1b, -du2, -dv2
            try:
                z = np.clip(z - np.linalg.solve(J, F), 0.0, 1.0)
            except np.linalg.LinAlgError:
                return None
        r1a = eval_surface(S1_h, z[0], z[1], rational=True)
        r1b = eval_surface(S1_h, z[2], z[3], rational=True)
        r2 = eval_surface(S2_h, z[4], z[5], rational=True)
        if max(np.linalg.norm(r1a - r2), np.linalg.norm(r1b - r2)) > atol:
            return None
        if np.all(np.abs(z[:2] - z[2:4]) <= 4.0 * np.array([ptol4[0], ptol4[1]])):
            return None            # same preimage — not a self-intersection
        return z

    found = []
    polys = [(bi, np.asarray(b.curve[1]), np.asarray(b.curve[0])) for bi, b in enumerate(branches)]
    for i, (bi, xi, si) in enumerate(polys):
        for j in range(i, len(polys)):
            bj, xj, sj = polys[j]
            for k in range(len(xi) - 1):
                l0 = k + 3 if i == j else 0        # skip adjacent segments of same branch
                for l in range(l0, len(xj) - 1):
                    d, s_, t_ = seg_dist(xi[k], xi[k + 1], xj[l], xj[l + 1])
                    if d > 2.0 * atol:
                        continue
                    a4 = (1 - s_) * si[k] + s_ * si[k + 1]
                    b4 = (1 - t_) * sj[l] + t_ * sj[l + 1]
                    z0 = np.array([a4[0], a4[1], b4[0], b4[1],
                                   0.5 * (a4[2] + b4[2]), 0.5 * (a4[3] + b4[3])])
                    z = newton6(z0)
                    if z is None:
                        continue
                    if any(np.linalg.norm(z - w["z"]) <= 4.0 * float(max(ptol4)) for w in found):
                        continue
                    xyz = eval_surface(S2_h, z[4], z[5], rational=True)
                    found.append({"z": z, "xyz": xyz,
                                  "links": [(bi, k), (bj, l)]})
    return found
```

In `bez_ssx`: during the main loop, right where `_check_loop_free` succeeds and the cell is traced, record `c3_possible = c3_possible or not theorem3_excludes_c3(cell.T1, cell.T2, cell.T3, cell.T4)` (initialize `c3_possible = False` before the loop; also set it True for tangent/subdivision-terminal cells that emitted fragments). After the C₁ pass:

```python
    if c3_possible and all_branches:
        for hit in c3_pass(S1_h_top, S2_h_top, all_branches, atol, ptol4_global):
            z = hit["z"]
            all_singularities.append(SSXSingularity(
                kind="self_intersection",
                stuv=np.array([z[0], z[1], z[4], z[5]]),
                stuv_mate=np.array([z[2], z[3], z[4], z[5]]),
                xyz=hit["xyz"], branch_links=hit["links"]))
```

**Checkpoints:**
1. `test_theorem3_skips_regular_case` must pass by the FLAG being False (add a debug assert while developing), not by the pair search finding nothing.
2. The umbrella case's two lines cross at branch INTERIORS — the segment-pair search complexity is O(N²) over segments; with ~100-point branches this is 1e4 segment pairs · cheap — fine. If a future case has 1e3-point branches, add the spatial hash then (YAGNI now).
3. Watch for false positives at shared branch ENDPOINTS (two branches meeting at a corner crossing share xyz with DIFFERENT... no — same stuv there). The `(s,t)≠(p,q)` guard at 4·ptol handles it; verify on case 10 (branches share no points; expect zero C₃).

- [ ] **Step 4: Run both tests** → PASS.
- [ ] **Step 5: Full regression sweep** (all 7 coverage cases + legacy 4 + all pytest suites).
- [ ] **Step 6: Commit** `feat(ssx5): C3 self-intersection certification (Theorem 3 + 6-var Newton)`

---

### Task 8: Validation sweep, harness extension, docs

**Files:**
- Modify: `examples/ssx/bez_ssx5_coverage_check.py` (report singularities per case)
- Modify: `docs/superpowers/plans/2026-06-10-ssx5-singular-cases.md` (mark checkboxes)
- Test: full suite

- [ ] **Step 1:** In `check_case`, after the branch report add:

```python
    for g in res.get("singularities", []):
        print(f"    singularity {g.kind}: stuv={np.round(g.stuv, 5).tolist()} "
              f"xyz={np.round(g.xyz, 5).tolist()} links={g.branch_links}")
```

- [ ] **Step 2:** Run everything and record:
  - `pytest tests/test_bez_ssx5_singular.py tests/test_bez_csx4.py tests/test_bez_ccx4.py tests/test_bez_ccx3_cases.py tests/test_bezier_common.py tests/test_bezier_curves_overlap.py -q` → all pass
  - `examples/ssx/bez_ssx5_coverage_check.py` → 7×100% coverage, zero spurious singularities on regular cases, timings within 1.2× of the pre-singularity baseline (record numbers in the commit message)
  - legacy 4 mini-cases → planes/transversal 1 branch 0 singularities; tangential 1 branch `kind='tangential'` (+ tangent_curve behavior unchanged); overlaps unchanged
- [ ] **Step 3:** Update the memory file `project_ssx5_branch_loss_fixes.md` (or a new `project_ssx5_singular_cases.md`) with what shipped, test locations, and any accepted deviations.
- [ ] **Step 4: Commit** `test(ssx5): singularity reporting in coverage harness + validation sweep`

---

## Risks / expected friction (from rounds 1–2 experience)

1. **Tolerance interplay** — every new dedup/match uses the established ladders (1·ptol+atol destructive; 4·ptol+2·atol matching). Any deviation needs a measured justification.
2. **Φ-loop marching near the tangency** — the Φ curve passes THROUGH the tangent point; a loop seed marching "toward" the seed may take the short way through the singularity instead of around the loop. The Ψ-validity filter kills the through-branch samples (they're on Φ but not on Ψ)... unless the segment is Ψ-valid too. If the emitted "loop" is actually seed→tangent-point→seed, detect via winding (path bbox ≪ expected) and re-seed from the other Φ-branch (flip the displacing step sign in `_march_phi_closed`).
3. **`solve_zero_dim` cell blowup near Σ/TΨ zero CURVES** (1-dim solution sets): the resolution floor (ratio-based: stop splitting when `max_i span_i/ptol_i ≤ 1`) bounds cells at ~(1/ptol) along the curve — with max_cells=20000 the budget cuts off first and sets `exhausted=True`; the `curve_flag` path (>12 solutions, or exhausted with >1) is the intended handling. Never raise max_cells to "fix" a hang.
4. **C₂ multiplicity ≥ 3** (paper's own admitted limit, §7.1): out of scope; the deflation handles multiplicity 2. Document, don't attempt.
5. **Rational Σ nets** — polynomial branch is required; homogeneous-numerator branch is best-effort within Task 2 (all planned tests are polynomial). If deferred, `c1_pass` must raise a clear `NotImplementedError` for rational inputs rather than silently skipping.
6. **Scale-invariance of tangency detection (found by Task-3 adversarial review, deliberately NOT hotfixed):** `_check_tangency` and `_tangency_witness` use absolute residual thresholds (GN tol_f 1e-8/1e-10, stall bar 1e-8) while TΨ magnitudes grow ~scale³. At input coordinates ~×1000 the detection silently switches off and the pre-fix pathology returns (spurious ~2·atol micro-branch instead of a typed `tangent_point`; verified bit-identical to pre-Task-3 output, so not a regression — a silent capability boundary). Fix direction when scheduled: scale-normalized residual criterion analogous to F_sq's `w_scale` (a T-net magnitude scale). Until then: tangency emission is reliable at O(1)–O(100) model scale.
7. **Coexisting-feature loss in crossing-less tangent cells (Mexican-hat class) — FOUND and FIXED during Task 3** (`e1db506`): the plan's original unconditional `continue` after witness emission deleted transversal features sharing the cell (touch + surrounding ring: the ring is NOT on Φ, so Task 5's Φ∩L could never recover it). As-built semantics: emit, then keep subdividing until all four global spans ≤ 4·unify_tol; descendants' re-confirmations absorbed by the singularity dedup; post-assembly filters drop the resurrected near-touch debris (micro-branches: every vertex ≤4·atol of an emitted tangent_point AND arc ≤16·atol; SSXPoints: ≤2·atol of an emitted tangent_point). Known cost: `_tangency_witness`'s solve_zero_dim enumeration burns its 2000-cell budget (~10s) at the top cell when a transversal curve coexists (the curve's Ψ-zero set defeats hull exclusion); acceptable per "runtime not critical", optimization candidate: plain-float GN in the enumeration's Newton callback instead of interval-arithmetic evaluation.

## Execution order and independence

Tasks 1 → 2 are prerequisites for everything. 3 → 4 → 5 build on each other (C₂). Task 6 (C₁) and Task 7 (C₃) are independent of 3–5 and of each other — parallelizable after Task 2. Task 8 last.

# Closest-Point via Squared-Distance Bernstein Nets — Design Spec

**Date:** 2026-06-25
**Status:** Implemented; **result contract superseded** — the "return all local
minima" decision (§2) was replaced by band semantics (the set of globally
closest entities within `d_min + atol`, including structured degenerate sets).
See `2026-07-02-closest-point-band-bnb-design.md`. The module structure, net
construction, prunes and Newton kernels described here still apply.
**Branch:** `closest-point-sq-dist-nets`

## 1. Motivation

The current closest-point code in `mmcore/numeric/closest_point.py` is unreliable and
inefficient:

- `_nurbs_curve_closest_point_divide_and_conquer` / `_nurbs_surface_closest_point_divide_and_conquer`
  use a coarse 3–9 point stencil with adaptive window shrinking plus a damped Newton/Gauss-Newton
  polish. They have **no completeness guarantee** — they can miss minima in unsampled basins and
  converge into the wrong basin for high-curvature or multi-extremum patches.
- `bez_curve_closest_point` (already net-based, 1D only) is the one good piece, but there is **no
  net-based surface closest point at all**. Surfaces still rely entirely on the divide-and-conquer
  path above.

The intersection redesign (`_bez_ccx4.py`, `_bez_csx4.py`, `_bez_ssx5.py`) demonstrated that the
**squared-distance Bernstein net** approach is both more reliable (sound Bernstein-hull prune
certificates) and more efficient (aggressive pruning, exact subdivision). This spec applies the
same foundation to the closest-point problem for rational/non-rational Bézier and NURBS **curves**
and **surfaces**.

## 2. Decisions (locked)

| Decision | Choice | Rationale |
|---|---|---|
| Result cardinality | **Global + all local minima** | Net subdivision enumerates all stationary candidates anyway; needed for offset/medial-axis/multi-pick. Global = nearest of them. |
| Rational handling | **Exact rational stationarity** | Build the exact numerator net `F'·w − 2F·w'`; correct under strong weight variation (commercial-grade). |
| Code organization | **New module, keep old for comparison** | Mirrors the intersection-redesign convention (`_bez_ccx3` vs `_bez_ccx4`). `closest_point.py` left untouched for A/B. |
| Module layout (a) | **Single new module** `_bez_closest_point.py` | Matches the project's flat per-algorithm file style; split into a package only if it grows past comfort. |
| Result payload (b) | Core dict `{param, point, distance, kind}`; **optional** `eval` payload (normal/derivatives) for the surface case on request | Keeps the hot path lean; preserves parity with the legacy rich return when callers want it. |
| Boundary/seam (c) | As specified in §6 | True-domain boundary only; internal seams deduped, never reported as minima. |

## 3. Core mathematics

### 3.1 True distance from the numerator net

With a rational surface `S(u,v) = A(u,v)/w(u,v)` (`A` = homogeneous xyz numerator net, `w` = weight
net), the true squared distance to a query point `P` is

```
g(u,v) = ‖P − S‖² = F(u,v) / w(u,v)² ,   F = ‖P·w − A‖²
```

`F` is the **existing** numerator net from `bern_sq_dist.point_surface_distance_squared_net_homog`
(bidegree `(2m, 2n)`); the curve analogue is `point_curve_distance_squared_net_homog`. True distance
at a candidate is recovered with the existing `eval_point_surface_distance_sq` /
`eval_point_curve_distance_sq`.

### 3.2 Exact stationarity nets

Minimizing `g` means `∇g = 0`. Since `w > 0`, this is equivalent to the **polynomial** conditions

```
∂g/∂u = (F_u·w − 2F·w_u)/w³ = 0   ⟺   N_u := F_u·w − 2F·w_u = 0
∂g/∂v = (F_v·w − 2F·w_v)/w³ = 0   ⟺   N_v := F_v·w − 2F·w_v = 0
```

Curve case: `N(t) := F'·w − 2F·w' = 0`.

Equivalent geometric form (residual ⊥ tangent): `N_u = −2⟨P·w − A, A_u·w − A·w_u⟩`. Either form is an
exact Bernstein net.

**Non-rational simplification:** `w ≡ 1`, `w_u = w_v = 0`, so `N_u = F_u`, `N_v = F_v`, `N = F'` — the
plain partial-derivative nets (what the existing 1D curve solver uses today).

**Net degrees:** `N_u` has bidegree `(3m−1, 3n)`, `N_v` has `(3m, 3n−1)`, `N` has degree `3q−1`. (The
`w_u`/`w_v` cross terms carry full dependence on the *other* axis, so both v- and u-degrees rise to
`3n`/`3m` respectively — the code derives shapes dynamically and does not hard-code these.)

### 3.3 Construction primitives

- `F`, `F_u`, `F_v`: existing net + `bern.bernstein_partial_derivative_coeffs(F, axis)`.
- `w`, `w_u`, `w_v`: weight column of the homogeneous control net + partial-derivative coeffs.
- Products `F_u·w`, `F·w_u`, …: require a **general (unequal-degree) Bernstein product**. The existing
  `bern.bernstein_product_conv(deg)` only handles *equal* degrees. We add a small helper using the
  identity `B_i^m·B_j^n = [C(m,i)C(n,j)/C(m+n,i+j)] · B_{i+j}^{m+n}`, applied per axis. After forming
  the two products at their (now equal) common bidegree, subtract coefficient-wise.

## 4. Prune certificate (completeness guarantee)

By the Bernstein convex-hull property, the coefficients of a net bound the polynomial over the cell.

- **Stationarity prune (sound, always on):** a cell can contain a joint stationary point **only if
  `N_u`'s coefficient hull straddles 0 AND `N_v`'s straddles 0**. If *either* hull excludes 0 (all
  coeffs `> 0` or all `< 0`), no stationary point exists in the cell → prune. (Curve: prune when
  `0 ∉ [min N, max N]` — this is exactly the existing `is_monotone` test.)
- **Value branch-and-bound prune (optional, global-only fast path):** if `bounds_point_surface`'s
  lower bound on `g` over the cell exceeds the current best distance, the cell cannot improve the
  global closest → prune. **Disabled in the default all-minima mode** (it would discard legitimate
  far local minima).

The sound 2D rule is the **OR** of the two "excludes 0" conditions (a stationary point needs *both*
partials to vanish, so monotonicity in *either* direction rules it out). Do not weaken this to
"prune only when both partials are monotone" — that is still sound but prunes far less and defeats
the purpose. The 1D curve rule is the single-partial special case.

## 5. Components (`mmcore/numeric/_bez_closest_point.py`)

### 5.1 Bernstein algebra helpers
- `_bernstein_product_nd(a, b, axes)` — general unequal-degree product along given axis/axes.
- `point_curve_stationarity_net(curve, point, rational)` → `N(t)` net (+ returns `F`, weights for
  reuse).
- `point_surface_stationarity_nets(surf, point, rational)` → `(N_u, N_v)` nets (+ `F`, weights).

### 5.2 `newton_surface_closest_point` (the only new numeric kernel)
2×2 Levenberg–Marquardt-damped solver of the stationarity system

```
r_u(u,v) = ⟨S(u,v) − P, S_u⟩ = 0
r_v(u,v) = ⟨S(u,v) − P, S_v⟩ = 0
```

Jacobian = the 2×2 Hessian of `½g`. Cell-bounded (clamps `(u,v)` to the current cell), monotone
backtracking line search, returns `(u, v, residual_vec, last_step)` — mirrors `newton_csx`'s bounded
interface. The curve case reuses the existing `newton_closest_point` (`closest_point.py:781`).

### 5.3 `bez_curve_closest_points(curve, point, atol, rational) -> list`
Hardened generalization of `bez_curve_closest_point`:
1. Build `F`, `N`. Interval-tree subdivision on `N` (de Casteljau on curve/F/N together).
2. Prune monotone cells (`0 ∉ bounds(N)`).
3. At flat/small cells (width < `bez_curve_param_tolerance`), root-find `N=0`, then polish with
   `newton_closest_point`.
4. Collect **all** roots; add endpoints `t=0,1` as boundary candidates.
5. Classify each candidate as a minimum via the pointwise second derivative `g''(t) > 0` (authoritative);
   the sign change of `N` across the root is a cheap pre-filter. Dedup by `ptol`.
6. Return all minima as `{t, point, distance, kind}`, sorted ascending by distance.

### 5.4 `bez_surface_closest_points(surf, point, atol, rational, want_eval=False) -> list`
1. Build `F`, `N_u`, `N_v`. Quadtree-style subdivision: split the axis with the wider gradient hull
   (or larger parametric span), de Casteljau on `surf`/`F`/`N_u`/`N_v` together.
2. **Stationarity prune** (§4). Optional value prune only in a global-only fast path.
3. At small cells (`width_u < tol_u and width_v < tol_v` from `bez_surface_param_tolerance`), run
   `newton_surface_closest_point` bounded to the cell; on convergence, **cut out** the
   `±ptol`-neighborhood (reuse the `_cutout_2d` idea from `_bez_ccx4.py`) so siblings don't
   re-converge to the same stationary point.
4. **Boundary handling:** run `bez_curve_closest_points` on each of the 4 boundary isocurves
   (`S[0,:]`, `S[-1,:]`, `S[:,0]`, `S[:,-1]`) and evaluate the 4 corners. These supply
   constrained (KKT) minima on the patch edge.
5. Classify interior candidates via the pointwise 2×2 Hessian of `g` (built from `Su,Sv,Suu,Suv,Svv`
   via the existing evaluators) — keep positive-definite interior minima and KKT-valid boundary
   minima; drop saddles/maxima. Dedup by `(ptol_u, ptol_v)` and point proximity.
6. Return all minima as `{u, v, point, distance, kind[, eval]}`, sorted ascending by distance.

### 5.5 NURBS-level wrappers
`nurbs_curve_closest_points` / `nurbs_surface_closest_points`:
1. `decompose_curve` / `decompose_surface` → list of single-span Bézier patches (each carries its
   native knot interval).
2. Run the per-patch core; map patch-local `[0,1]` params back to global NURBS params via the
   patch interval.
3. **Merge/dedup across patch seams** by global-parameter + point proximity. A patch edge is treated
   as a *real* constraint boundary **only** where it coincides with the global domain boundary;
   internal seams are smooth and must not yield spurious boundary minima (a stationary point sitting
   exactly on a seam appears in both adjacent patches → deduped to one).
4. Return all minima sorted, `result[0]` = global closest.

The legacy single-best wrappers `nurbs_curve_closest_point` / `nurbs_surface_closest_point` and the
`_divide_and_conquer` cores are left **untouched** in `closest_point.py` as the A/B comparison
baseline.

## 6. Result shape

```python
# curve entry
{"t": float, "point": np.ndarray, "distance": float, "kind": "min" | "boundary_min"}
# surface entry
{"u": float, "v": float, "point": np.ndarray, "distance": float,
 "kind": "min" | "boundary_min", "eval": {...}?}   # eval present only if want_eval=True
```

List sorted ascending by `distance`; `result[0]` is the global closest. Always non-empty (endpoints/
corners are always candidates).

## 7. Edge cases

- **Point on the geometry** (`g = 0`): still a valid minimum, reported normally.
- **Equidistant family** (e.g. query at a circle/sphere center → an equidistant ring): detect a
  near-flat stationarity net over a region (gradient hull collapses toward 0 over a wide cell) and
  report a representative candidate flagged as degenerate rather than enumerating the continuum.
- **Coincident/degenerate weights, tangency, near-singular Hessian:** handled by LM damping in the
  Newton kernels.
- **Result never empty:** boundary endpoints/corners are always evaluated as candidates.

## 8. Testing strategy

**Analytic ground truth**
- Point ↔ line segment: closest = orthogonal projection clamped to endpoints.
- Point ↔ **circle** (rational — exercises exact stationarity): off-center query → exactly 1 min +
  1 max, validating both stationarity-net construction and min/max classification.
- Point ↔ sphere / cylinder / torus surfaces (rational): known closest along the surface normal.

**Multiplicity**
- Query near a U-shaped curve → 2 local minima; assert both found and ordered.

**Property tests (random Béziers/NURBS)**
- Every returned surface minimum satisfies `⟨P−S, S_u⟩ ≈ 0`, `⟨P−S, S_v⟩ ≈ 0`.
- Reported minima have positive-definite `g` Hessian.
- The global (`result[0]`) matches a dense-grid argmin within tolerance.

**Cross-checks**
- Agreement with the legacy `_divide_and_conquer` path on single-minimum cases.
- Agreement of the global with a brute-force fine sampler.

**Multi-patch NURBS**
- Multi-knot curves/surfaces: no spurious seam minima; correct cross-seam dedup.

**Performance**
- Cell counts / wall-clock vs the legacy D&C on representative cases (the efficiency goal).

## 9. Out of scope (YAGNI)

- Batched/many-query-point projection (the `F` net is point-specific; revisit later if profiling
  shows a need).
- Per-cell *classified* unique-minimum certificates via Hessian nets (the user chose all-local-minima,
  not classified-stationary-points — pointwise Hessian classification suffices).
- Replacing or deleting the legacy code (kept for comparison).

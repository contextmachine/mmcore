# Closest-Point Valley-Marching — Design Spec

**Date:** 2026-06-28
**Status:** Approved (architecture); pending implementation plan
**Branch:** `closest-point-valley-marching`
**Builds on:** `docs/superpowers/specs/2026-06-25-closest-point-sq-dist-nets-design.md`
(`mmcore/numeric/_bez_closest_point.py`)

## 1. Motivation

`bez_surface_closest_points` subdivides the squared-distance stationarity nets `(N_u, N_v)` and
prunes a cell when either partial's Bernstein hull excludes 0. This is sound and fast for
**transversal** (isolated) minima, but it degrades catastrophically on a **degenerate valley**: a
configuration where the squared-distance function `g(u,v) = ‖P − S(u,v)‖²` has a 1-D *continuum* of
near-stationary points rather than isolated minima.

Reproducer: `examples/closest_point/surfcp1.py` — a point near the axis of a non-circular funnel of
revolution (with a degenerate `v=0` pole edge). Observed: **17 s, all four Bézier patches hit the
`max_cells=60000` cap.**

### Root cause (measured)
- `P` lies near the axis of revolution ⇒ `∂g/∂u` is **shallow** (near-zero but non-crossing) over a
  broad u-range — a near-degenerate "ring valley."
- The Bernstein-hull stationarity prune cannot certify "no root" in a shallow region: the count of
  `N_u`-straddling cells **doubles every subdivision level** (6→10→18→34→66→130→…), i.e. a constant
  *fraction* (~½) of the u-domain straddles 0 at every scale, instead of localizing to the ~1 true
  root. A tighter midpoint+Lipschitz bound is *worse* here (everything straddles), confirming the
  valley is genuine, not a prune-looseness artifact.
- This is amplified to a hard cap by an absurdly small parametric tolerance: the surface spans ~10⁵
  in xyz, so default `atol=1e-3` ⇒ `ptol ≈ 2e-8` (~25 subdivision levels). With `atol ≥ 10` the cap
  is not hit (no algorithmic change), and the returned answer is **identical at every `atol`** — the
  minima come from boundary handling + Newton; `ptol` only controls *when subdivision stops*, and the
  final result is Newton-polished regardless.

Degenerate valleys are common: a point at the center of an ellipsoid, on the axis of an elliptical
cone, near the axis of any surface of revolution, etc. Subdivision is the wrong tool for a 1-D
manifold of near-stationary points; we should **trace** it.

## 2. Validated findings (prototype)

A throwaway prototype confirmed the approach on `surfcp1`:
- The valley floor `V = {(u,v) : ∂g/∂v = 0, ∂²g/∂v² > 0}` is a clean 1-D curve **where it exists**
  (~63/241 u-samples); over the rest of u there is no interior cross-valley min and the floor merges
  into the pole `v=0`.
- On `V`, `dh/du = ∂g/∂u` (since `∂g/∂v=0` there). Marching `V` and tracking the sign of `∂g/∂u`
  recovers **exactly** the two true minima (`u=0`, `u=π`, dist 8504.84) and nothing spurious.
- **Naive seeding fails:** a single 1-D Newton in `v` from the cell center gets *captured by the
  pole* (`v*→0`) over the far-side u-range, fabricating ~14 spurious minima. Robust seeding (all
  interior `v`-roots, filtered by `∂²g/∂v² > 0`) fixes it. Seeding is the crux.

## 3. Core idea — hybrid subdivision + certified valley-marching

Keep subdivision as the **completeness backbone**. Add a valley **marcher** that a cell *graduates*
to only when a per-cell certificate proves marching is complete for that cell. Every cell is
discharged by exactly one sound mechanism:

| Mechanism | When | Sound because |
|---|---|---|
| **Prune** | a partial-hull excludes 0 | no stationary point in the cell (existing) |
| **Valley-resolve** | cross-valley Hessian-diagonal hull is sign-definite + orthogonal partial is shallow/stalling | the floor is the complete interior-stationary locus (§4) |
| **Newton leaf** | cell ≤ ptol | geometrically indistinguishable; one Newton (existing) |
| **Subdivide** | none of the above | defers to finer cells (existing) |

## 4. The completeness certificate (the linchpin)

A cell is a **clean valley cell** when, over the whole cell, the cross-valley second-derivative net
is **sign-definite** — the Bernstein hull of the `g_vv` net (or `g_uu`) excludes 0 — AND the
orthogonal first-derivative direction is shallow/stalling (§6 trigger). Pick the *definite* axis as
the corrector axis (say `v`) and the *shallow* axis as the march axis (`u`).

Under `∂²g/∂v² > 0` throughout the cell:
1. `∂g/∂v` is strictly monotone in `v` ⇒ **at most one floor point per `u`** ⇒ the floor `v*(u)` is a
   single graph (no branches, no folds inside the cell).
2. **Every interior local minimum has `∂g/∂v = 0` ⇒ it lies on the floor.** Enumerating the floor's
   `∂g/∂u` sign-changes therefore finds *all* interior minima in the cell — none can hide off-floor.
3. Minima on the cell's edges are boundary/constrained minima, already covered by the always-on
   edge-isocurve handler.

⇒ Marching the floor across the cell's march-axis extent **+** the existing edge handler is a
**provable cover** of the cell. That cover licenses *dropping* the cell from subdivision. No min is
lost because none can exist off-floor-and-off-edge.

Direction is chosen by which Hessian-diagonal net is hull-definite; if neither is, the cell is **not**
a clean valley cell and stays with subdivision.

## 5. The marcher

Input: a clean valley cell `[u0,u1]×[v0,v1]`, corrector axis `v` (`∂²g/∂v²` definite), march axis `u`.

1. **Seed from the cell's march-axis edges** (`u=u0`, `u=u1`) — not just the middle. Solve
   `∂g/∂v=0` in `v` over `[v0,v1]`. Monotonic `∂g/∂v` ⇒ 0 or 1 root: if none, the cross-valley min is
   on a `v`-edge (handed to the edge handler) and the floor does not enter at that `u`; if one, it is
   the unique floor entry `v*`.
2. **March `v*(u)` across `[u0,u1]`** by predictor–corrector — predictor `v += (dv/du)·Δu` with
   `dv/du = −g_uv/g_vv` (implicit-function theorem); corrector = 1-D Newton on `∂g/∂v=0`. Arc-length,
   curvature-adaptive step (reuse the SSX tracer / `_march_curve_on_surface` machinery).
3. **Event detection:** monitor `φ(u) = ∂g/∂u(u, v*(u)) = dh/du`. Each `−→+` sign change is a local
   min; `+→−` is a saddle. Refine the crossing (1-D root of `φ` along the march), then **2-D
   Newton-polish** the full closest-point system to nail `(u*,v*)`, classify via the 2×2 Hessian,
   dedup, emit.
4. **Termination:** when `v*(u)` exits `[v0,v1]` (the floor leaves through a `v`-edge — e.g. merges
   into the pole) or reaches `u1`. A `v`-edge exit hands that edge segment to the boundary handler.

Evaluators: the surface point + 1st/2nd derivatives are already available via `eval_surface_d2`; the
stationarity/Hessian *nets* (`g_uv`, `g_vv`, `g_uu`) come from `bernstein_partial_derivative_coeffs`
applied to the existing `F`/`N_u`/`N_v` nets.

## 6. Control-flow integration

Inside the `bez_surface_closest_points` cell loop, between the existing flat guard and the small-cell
leaf:

```
if hull_excludes_zero(N_u) or hull_excludes_zero(N_v):   continue      # prune (existing)
if both partials flat:                                   ... ; continue  # existing degenerate guard
# NEW — valley detection:
vv_definite = hull_excludes_zero(g_vv_net)               # cross-valley min line is clean in v
uu_definite = hull_excludes_zero(g_uu_net)
stalling     = (depth >= D_valley) and one-partial-shallow   # the doubling signature
if stalling and (vv_definite or uu_definite):
    march_axis, corr_axis = (u, v) if vv_definite else (v, u)
    out += march_valley_cell(cell, march_axis, corr_axis)   # §5
    continue                                                 # DROP the cell (covered)
if small (≤ ptol):  newton leaf; continue                 # existing
subdivide wider axis                                       # existing
```

`march_valley_cell` returns the in-cell minima (and feeds `v`-edge exits to the boundary handler).
Mark cells as valley cells **individually**, each self-certified by its own `g_vv` hull — **no
cross-cell "which cells did a global march cover" bookkeeping** (that bookkeeping is exactly where
subdomains would be lost). Adjacent valley cells re-march the shared floor; dedup absorbs the
overlap. Subdivision still owns every non-certified region.

## 7. Degenerate-manifold (exact continuum) semantics

When `P` is *exactly* on the axis/center, the whole floor is equidistant ⇒ `φ(u) = ∂g/∂u ≡ 0` on the
floor (not merely shallow). The marcher detects `|φ| ≈ 0` over a span and reports the floor as a
**degenerate minimum manifold** — one representative point plus the floor's parametric extent and a
`kind = "degenerate_min"` flag — instead of enumerating a continuum. Finite, correct answer.

## 8. Completeness argument (summary)

Every cell is discharged by a sound prune, a leaf, deferral (subdivide), or a *certified* valley
cover. The valley cover only fires under the `g_vv`-definite certificate, which guarantees the floor
is the complete interior-stationary locus of the cell. Folds / branches / the floor-into-pole
transition have `g_vv` hull straddling 0 ⇒ they **fail the certificate ⇒ stay with subdivision**,
which splits until the fold is isolated in a tiny leaf and the clean flanks become markable valley
cells. Boundary minima are always covered by the edge handler. Therefore no minimum is lost.

## 9. Complementary safeguard — bounded subdivision

Independent of the marcher, add a **relative floor on `ptol`** so subdivision can never chase
parametric resolution far below what is meaningful (the band-aid validated in investigation: a floor
of ~1e-4 of the patch domain removes the cap on `surfcp1`; final accuracy is unaffected because
leaves are Newton-polished). The floor engages only at pathological scales; for normal O(1) surfaces
the computed `ptol` already exceeds it, so existing behavior and tests are unchanged. This is a
safety net that guarantees termination even if valley detection mis-fires; the marcher is the
primary fix.

## 10. Edge cases / open risks (to resolve during planning)

- **Trigger tuning** (`D_valley`, "one-partial-shallow" threshold): must not mis-fire on normal cells
  (wasted marches) or miss a valley (revert to thrash). Needs empirical tuning against a case suite.
- **Pole-exit handoff:** the floor leaving through a `v`-edge must cleanly hand that segment to the
  boundary handler without the capture seen in the naive prototype.
- **Cost:** per-cell `g_uu/g_uv/g_vv` nets (more `bernstein_partial_derivative_coeffs`) — cheap, but
  only compute them when a cell is otherwise about to subdivide deeply (lazy).
- **Curve case:** 1-D curves have no 2-D valley; the existing flat guard already handles a 1-D
  constant-distance arc. Valley-marching is **surface-only**.

## 11. Testing strategy

- **`surfcp1` regression:** terminates well under cap, < ~1 s, returns the two minima (dist 8504.84)
  + the pole candidate, matching a dense-grid oracle. Assert no `max_cells` warning.
- **Analytic valleys with known minima:**
  - Point on the axis of an elliptical cone ⇒ exactly 2 minima (minor-axis directions); assert both,
    with the saddle (major axis) rejected.
  - Point at the center of an ellipsoid patch ⇒ minima at the near semi-axes.
  - Point *exactly* on a circular cone's axis ⇒ `degenerate_min` manifold reported once, not a
    continuum.
- **Non-regression:** all 25 existing `tests/test_bez_closest_point.py` cases unchanged (transversal
  cases must not trigger marching); the multi-span seam test still passes.
- **Completeness cross-check:** for random near-degenerate configurations, every dense-grid local
  minimum is matched by a returned minimum (no misses), and no spurious minima are returned.
- **Performance:** cell-count / wall-clock on `surfcp1` and the analytic valleys vs the current
  subdivision-only path.

## 12. Out of scope (YAGNI)

- Curve valley-marching (no 2-D valley in 1-D).
- A full medial-axis / all-critical-manifold extraction; we report minima (+ degenerate manifolds),
  not a complete Morse decomposition.
- Replacing the subdivision backbone; marching is an *add-on* resolver, not a replacement.

## 13. Decisions for review

1. **Trigger:** detect via `depth ≥ D_valley` + shallow-partial + definite-orthogonal-Hessian (as
   above), vs a direct "straddle-not-localizing" measurement. Recommend the former (cheap, robust).
2. **Marching granularity:** per-cell certified march (recommended — no cross-cell bookkeeping) vs a
   single global march per detected valley (faster but bookkeeping risks completeness).
3. **Safeguard:** include the relative `ptol` floor (§9) as a belt-and-suspenders termination
   guarantee (recommended) or rely solely on the marcher.
4. **Degenerate manifold output:** new `kind = "degenerate_min"` with extent, vs collapsing to a
   single `"min"`. Recommend the explicit flag.

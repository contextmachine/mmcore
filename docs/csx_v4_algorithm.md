# Bezier Curve-Surface Intersection: v4 Algorithm

## Overview

`_bez_csx4.py` implements curve-surface intersection (CSX) using the same architectural principles as the CCX algorithm: a squared-distance Bernstein net, a two-phase architecture, and Newton with cutout-based termination. The extension from 2D (curve-curve) to 3D (curve-surface) introduces specific challenges — most notably, the overlap path through parameter space can be significantly curved, and the absence of a practical 3D Hessian uniqueness certificate means the algorithm relies more heavily on Newton probing and boundary analysis.

Despite being implemented in Python with NumPy, the algorithm handles real-world CAD geometry (rational NURBS on torus surfaces, pullback curves from commercial modelers) at interactive speeds — typically under 50ms per Bezier pair and under 250ms for full NURBS intersections with dozens of patch pairs.

### Module map

| Module | Role |
|--------|------|
| `csx/_bez_csx4.py` | Main algorithm: two-phase architecture, boundary analysis, Phase 2 loop |
| `csx/_ncsx4.py` | NURBS adapter: decomposition, BVH, overlap merging |
| `ccx/_bez_ccx4.py` | Used by CSX Phase 1 for surface-boundary isocurve intersections |
| `_sq_dist_classify.py` | Classifier (shared with CCX) |
| `_bern_zero_1d.py` | 1D solver (shared with CCX) |
| `_bezier_common.py` | Evaluation, Newton (shared with CCX) |
| `bern_sq_dist.py` | Trivariate net construction |

## The trivariate squared-distance net

For a Bezier curve `C(t)` of degree `p` and a Bezier surface `S(u,v)` of degrees `(m,n)`, the squared distance is a trivariate polynomial:

```
D(t,u,v) = ||C(t) - S(u,v)||^2
```

Its Bernstein control net has shape `(2p+1, 2m+1, 2n+1)`. For typical degree-3 curves and biquadratic surfaces, this is `(7, 5, 5)` — a compact structure that encodes everything about the distance landscape.

The same convex hull properties apply as in CCX: if all coefficients are positive, no intersection exists. The net can be subdivided exactly via De Casteljau, restricted to boundary faces (collapsing one axis), and analyzed for sign changes along each direction.

## Two-phase architecture

The same principle as CCX: **for Bezier objects, overlaps and boundary intersections can only exist at the boundaries of the original objects.** This motivates a clean separation:

### Phase 1: Boundary analysis (initial patch only)

The 6 faces of the parameter cube `[0,1]^3` decompose into two geometric problems:

**Curve-endpoint faces (t=0, t=1):** "Does the curve endpoint lie on the surface?" The 2D slice `D(u,v) = ||C(t_fixed) - S(u,v)||^2` is extracted directly from the trivariate net — it equals the point-surface squared-distance net. A Newton projection finds the closest point on the surface; if the distance is below `atol`, it's a boundary intersection.

**Surface-boundary faces (u=0, u=1, v=0, v=1):** "Does the curve intersect a boundary isocurve of the surface?" The surface isocurve (a Bezier curve in 3D) is extracted and `bez_ccx` is called — **reusing the entire CCX machinery** including boundary analysis, overlap detection, and Phase 2 search. This is a deliberate architectural choice: by delegating to CCX, improvements to CCX automatically benefit CSX boundary analysis.

An **AABB pre-filter** skips isocurve CCX calls where the curve and isocurve bounding boxes don't overlap. In practice this eliminates 90% of the CCX calls — the single most impactful optimization for NURBS-level CSX performance.

### Overlap detection: stepping along the valley

The CCX valley check assumes the overlap path is approximately linear in 2D parameter space. In CSX, the overlap curve through 3D parameter space `(t, u, v)` can be significantly curved — a straight-line midpoint check fails for curved surfaces.

The solution: **step from the endpoint, not to the midpoint.**

1. From each boundary zero, take a small step inward along the curve parameter `t` (by `2 * ptol_t`)
2. Evaluate `C(t_step)` and project onto the surface via Newton to find `(u, v)`
3. If the projection distance is below `atol` AND the `(u,v)` moved away from the seed → the valley continues inward
4. If confirmed from both endpoints → overlap

This approach:
- Does NOT assume the overlap is linear in parameter space
- Does NOT iterate through the entire overlap (unnecessary — the Bezier endpoint property guarantees the overlap spans the full interval if it exists at both boundaries)
- Correctly handles "practical overlaps" where the curve lies within `atol` of the surface but isn't algebraically on it (e.g., pullback curves from commercial CAD modelers with finite modeling precision)

### Cutout: removing Phase 1 results from Phase 2

After Phase 1 finds overlaps and boundary intersections, their neighborhoods are **cut from the curve parameter `t`** — not from the 3D parameter space. This produces a list of remaining t-intervals, each guaranteed free of overlaps and boundary intersections.

Why cut only from `t` (the curve), not from `(t,u,v)`? Because overlaps and intersections are 1D phenomena on the curve, but they may run diagonally through the 2D surface parameter space. A 3D box cutout would either be too small (missing the diagonal overlap) or too large (removing valid isolated intersections that share surface parameter ranges with the overlap). Cutting only along `t` is both simpler and correct.

### Phase 2: Isolated intersection search

For each remaining t-interval × full surface:

1. **Min-of-net prune**: Weight-corrected lower bound from Bernstein coefficients
2. **Lipschitz tightening**: Midpoint evaluation + derivative sup-norm bound
3. **Derivative sign pruning**: If any of the three partial derivative nets `∂D/∂t`, `∂D/∂u`, `∂D/∂v` has all same-sign coefficients, no stationary point exists. This is the 3D Poincaré-Miranda test — cheap (three min/max operations) and effective.
4. **ptol-based termination**: If the curve parameter span drops below `ptol_t`, the cell is a micro-fragment. Report its center and stop.
5. **Newton**: From the cell center, on the original (un-subdivided) curves with global parameters. Convergence via parametric tolerance comparison.
6. **3D cutout**: When Newton finds a new intersection, split the cell into `3×3×3 = 27` boxes along all three axes at `sol ± ptol`, discard the center, push the remaining 26. Most are immediately pruned by min-of-net on the next iteration.
7. **Converged-outside-cell prune**: If Newton converges to a point outside the cell, prune immediately. This is the key to performance near known intersections: Newton escapes to the nearest one, the range check rejects it, the cell is pruned without further subdivision.

### Why no 3D uniqueness certificate?

The CCX algorithm has a Hessian positive-definiteness check (Sylvester's criterion on 2×2 Bernstein-bounded Hessian) that proves a unique stationary point. For the 3D CSX case, the analogous 3×3 Sylvester criterion using Bernstein coefficient bounds is too conservative to fire in practice — the cross-terms overwhelm the diagonal bounds.

Instead, CSX relies on **Newton probing at every subdivision level**: if Newton converges inside the cell, the intersection is found and the cutout handles the rest. If Newton escapes, the cell is pruned. If Newton fails to converge, subdivision continues. The derivative sign check provides the "no stationary point" certificate that the Hessian PD check would have given in the uniqueness direction.

This is a pragmatic trade-off: the 2D uniqueness certificate is a theoretical luxury that provides early termination for cells with a single crossing. The 3D version doesn't fire, so we compensate with aggressive Newton probing and cutout. The result is similar performance in practice.

## NURBS integration

`_ncsx4.py` wraps `bez_csx` for NURBS curve × NURBS surface intersection:

1. **Decomposition**: `decompose_curve` at curve knots, `decompose_surface` at surface knots in both directions
2. **BVH filtering**: Segment × patch pairs with overlapping AABBs
3. **Parametric deduplication**: Groups by curve parameter `t`, merges entries within `ptol_t` using `nurbs_curve_param_tolerance`
4. **Overlap merging**: Adjacent overlaps from different Bezier pairs are merged by t-range proximity. Isolated points near overlap endpoints are absorbed.
5. **Micro-fragment classification**: Points from ptol-terminated cells adjacent to overlaps become part of the overlap; isolated micro-fragments elsewhere remain as isolated intersections.

## Design principles

Several design principles emerged from extensive debugging on real CAD geometry:

**Boundary analysis runs once.** The single most impactful architectural decision. For a Bezier curve and surface, overlaps and boundary intersections are properties of the original objects — they cannot be created by subdivision. Running boundary analysis on every subdivision cell wastes enormous effort on artificial boundary zeros.

**Cut from the curve, not the parameter space.** Overlaps are 1D on the curve but diagonal in parameter space. Box cutouts in parameter space miss the diagonal. Cutting along the curve parameter is both simpler and correct.

**Newton is a proximity probe.** Newton doesn't just find intersections — it tells us about the neighborhood. If Newton escapes the cell, there's no intersection here. If Newton finds a known point, the cell is in its attraction basin. Only genuinely new convergence requires further analysis.

**The pruning cascade matters more than any single optimization.** The sequence AABB → min-of-net → Lipschitz → derivative sign → Newton is ordered by cost. Each level catches cells that the previous level missed, but most cells are caught at the cheapest levels. Moving AABB to the front (which was initially overlooked) provided the single largest speedup.

**Tolerance is geometry-aware.** The parametric tolerance from `bez_curve_param_tolerance` / `bez_surface_param_tolerance` converts the user's geometric tolerance into parameter space using the actual curve/surface speed. This replaces arbitrary thresholds with mathematically grounded criteria.

**Practical overlaps are overlaps.** A curve from a commercial CAD modeler may lie within `atol` of a surface without being algebraically on it. The valley check uses `atol` as its threshold, not machine precision. Users who need only algebraically exact overlaps can set a tighter tolerance.

## Performance

The algorithm processes typical Bezier pairs (degree 3-4, rational) in 1-50ms, and full NURBS intersections with dozens of patch pairs in 20-250ms. This is achieved through architecture, not through low-level optimization — the implementation is pure Python/NumPy with Cython only for Bernstein basis evaluation.

The key metrics on representative test cases:

| Case | Time | Notes |
|------|------|-------|
| Polynomial 2 isolated | 5ms | Bezier pair, direct |
| Rational overlap | 2-5ms | Detected in Phase 1, no subdivision |
| Practical overlap (torus pullback) | 127ms | 27 segs × 16 patches, NURBS level |
| Rational 2 isolated (near-tangent) | 22ms | 3D cutout resolves false duplicates |
| 7 NURBS examples (mixed) | 7-127ms each | All correct, no false positives |

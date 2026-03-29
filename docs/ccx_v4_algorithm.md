# Bezier Curve-Curve Intersection: v4 Algorithm

## Overview

`_bez_ccx4.py` implements a subdivision-based curve-curve intersection (CCX) algorithm built on a single unifying concept: the **squared-distance Bernstein control net**. Given two Bezier curves `C1(u)` and `C2(v)`, the algorithm constructs the bivariate polynomial `D(u,v) = ||C1(u) - C2(v)||^2` in Bernstein form and analyzes its structure to classify intersections — without Jacobian-rank analysis, SVD, or overlap tracing.

### Module map

| Module | Role |
|--------|------|
| `ccx/_bez_ccx4.py` | Main algorithm: subdivision loop, Newton refinement, post-processing |
| `_sq_dist_classify.py` | Classifier: analyzes the sq-dist net to decide intersection type |
| `_bern_zero_1d.py` | 1D solver: finds zeros on boundary faces of the net |
| `_bezier_common.py` | Shared utilities: curve/surface evaluation, Newton solver |
| `bern_sq_dist.py` | Net construction: builds `D(u,v)` from control points |
| `ccx/_nccx4.py` | NURBS adapter: decomposes NURBS into Bezier segments, calls `bez_ccx` |

## The squared-distance net

The core data structure is the Bernstein control net of the squared distance between two curves. For polynomial curves `C1` of degree `p` and `C2` of degree `q`, this is a bivariate Bernstein polynomial of degree `(2p, 2q)` with shape `(2p+1, 2q+1)`.

```
D(u,v) = ||C1(u) - C2(v)||^2
```

For rational curves in homogeneous form, the net represents the *numerator* of the squared distance (to avoid division by weights). The actual distance is recovered by dividing by the weight product: `d^2 = D / (w1 * w2)^2`.

**Why this representation matters:** The Bernstein coefficients of `D` provide guaranteed bounds on the function's range (convex hull property). If all coefficients are positive, the distance is provably nonzero everywhere on `[0,1]^2`. This is the tightest enclosure available without subdivision.

The net is constructed once at the top level by `curve_curve_squared_net_homog()` from `bern_sq_dist.py`, and then subdivided along with the curves throughout the algorithm. Subdivision of the net is a simple De Casteljau split along one axis — no recomputation from scratch.

## Algorithm flow

```
bez_ccx(C1, C2, atol):
    F = build sq-dist net from C1, C2
    ptol_u, ptol_v = parametric tolerances from curve geometry
    stack = [(C1, C2, F, weights, param_ranges, depth=0)]

    while stack:
        classify F → NO_INTERSECTION | UNIQUE_ISOLATED | OVERLAP | BOUNDARY_ZERO | INDETERMINATE

        NO_INTERSECTION  → discard cell
        UNIQUE_ISOLATED  → Newton refine → accept if converged
        OVERLAP          → report overlap endpoints
        BOUNDARY_ZERO    → direct curve evaluation at boundary params → accept if dist < atol
        INDETERMINATE    → subdivide along longer axis, push children

    post-process: merge spurious micro-overlaps, deduplicate
```

## Classification hierarchy

The classifier (`classify_sq_dist_net`) applies a sequence of increasingly expensive checks, returning as soon as one is conclusive:

### Check 1: Min-of-net positive

```
lower_bound = min(F) / (max|w1| * max|w2|)^2
if lower_bound > atol^2 → NO_INTERSECTION
```

The cheapest possible check. Uses the Bernstein convex hull property: the polynomial's range is bounded by the range of its coefficients. Weight correction converts from the numerator polynomial to actual squared distance.

### Check 2: Lipschitz tightening

```
F_mid = F evaluated at (0.5, 0.5) via De Casteljau averaging
L = sum of sup-norms of partial derivative nets
lower = F_mid - 0.5 * L
if lower / (max|w1| * max|w2|)^2 > atol^2 → NO_INTERSECTION
```

Strictly tighter than Check 1 when the minimum coefficient is an outlier (far from the midpoint value). Uses the fact that for a Lipschitz function, `f(x) >= f(x0) - L * ||x - x0||`, where `L` is bounded by the derivative coefficient sup-norm (convex hull property applied to the derivative).

### Check 3: Boundary zero analysis (cheap)

For each of the 4 boundary faces (`u=0`, `u=1`, `v=0`, `v=1`), extract the boundary restriction of `F` (simply the first or last slice along that axis — a free operation). If `min(coefficients)` on a boundary face is below `(atol * w_scale)^2`, that face "touches zero" and is flagged.

This check is informational — it doesn't return a classification by itself, but feeds into subsequent checks.

### Check 3b: Precise boundary zeros (1D solver)

For each flagged boundary face from Check 3, run the 1D Bernstein zero-finder (`find_bernstein_zeros_1d`) on the boundary restriction. This finds the exact parameter values where `D` reaches zero on the boundary.

The 1D solver works as follows:

1. **Endpoint check:** Bernstein coefficients at `t=0` and `t=1` are exact function values. If below `atol^2`, report as zeros.

2. **Derivative sign-change count:** Compute the derivative coefficients and count sign changes (Descartes' rule for Bernstein form). This gives an upper bound on the number of interior extrema:
   - 0 sign changes → monotone, no interior minimum
   - 1-2 sign changes → at most one minimum. Use Newton on the derivative to locate it, then check if the function value there is below `atol^2`.
   - 3+ sign changes → subdivide and recurse

3. **Safe subdivision:** When subdividing, the split point is chosen at the center of the **longest run of strictly positive coefficients**. This ensures the split happens in a region where the polynomial is well above zero, keeping each root intact in exactly one child. Splitting near a root would fragment it across both children, producing duplicates.

The results are stored as `BoundaryZero(axis, side, param)` objects — the precise parameter location on each boundary face where the squared distance reaches zero.

### Check 4: Uniqueness certificate (2D only)

For bivariate nets (CCX case), checks whether the sq-dist function has a unique critical point:

1. **Existence:** Each partial derivative net (`dD/du` and `dD/dv`) must have coefficients of both signs. By the intermediate value theorem applied to Bernstein polynomials, this guarantees at least one zero of the gradient.

2. **Uniqueness:** The Hessian matrix must be globally positive-definite:
   ```
   min(D_uu) > 0
   min(D_vv) > 0
   min(D_uu) * min(D_vv) - max(|D_uv|)^2 > 0
   ```
   This uses Bernstein coefficient bounds on the second derivatives. If the Hessian is PD everywhere, the function is strictly convex and has at most one critical point.

When both conditions hold, there is exactly one minimum of `D` in the domain → `UNIQUE_ISOLATED`.

This check is only implemented for 2D nets (CCX). For 3D nets (CSX), the 3x3 Hessian PD check via Sylvester's criterion is too conservative to be useful in practice.

### Check 5: Overlap certificate (valley check)

If two boundary zeros exist on **different faces** of the domain, the algorithm checks whether they are connected by a "valley" — a path along which `D` stays near zero:

1. Convert each boundary zero to a parameter-space point
2. Compute the direction from one zero toward the other
3. Step inward along this direction from each zero
4. Evaluate `D` at the stepped points and at the midpoint of the connecting line
5. If `D` stays below threshold at all three points, the valley is confirmed → `OVERLAP`

**Design decisions:**

- The step direction is along the connecting line (from zero A toward zero B), not perpendicular to the boundary face. An overlap valley runs diagonally through the parameter domain; stepping perpendicular to the boundary would go uphill.

- The midpoint must be in the strict interior of `[0,1]^2`. This rejects false overlaps where two boundary zeros sit near the same corner — the "valley" between them runs along the boundary face itself, not through the interior.

- The threshold for the stepped points is `100 * threshold` (generous) because the exact valley trajectory may not be perfectly linear.

### Check 6: Boundary zeros without overlap

If precise boundary zeros were found (Check 3b) but no overlap was confirmed (Checks 5/5b), return `BOUNDARY_ZERO`. This tells the caller: "there are proven zeros on the domain boundary, but they are isolated, not part of an overlap."

### Fallback: INDETERMINATE

If no check is conclusive, return `INDETERMINATE`. The caller must subdivide and try again on smaller cells.

## Newton refinement

Newton's method is used only for the `UNIQUE_ISOLATED` case (and as a fallback at max depth). The implementation is an LM-damped Gauss-Newton solver with backtracking line search:

```
G(u,v) = C1(u) - C2(v) = 0
J = [C1'(u), -C2'(v)]
delta = -(J^T J + lambda I)^{-1} J^T G
```

### Convergence criterion: parametric tolerance

Newton's convergence is **not** determined by the residual `||G||`. Instead, convergence is decided by comparing the last step `(du, dv)` against the **parametric tolerances** `ptol_u` and `ptol_v`, computed from the curve geometry via `bez_curve_param_tolerance()`.

The parametric tolerance is the maximum parameter perturbation that corresponds to geometric deviation `<= atol`. It accounts for the curve's speed (derivative magnitude) — fast-moving curves have tighter parametric tolerances than slow-moving ones.

```
ptol = atol / max_derivative_speed
```

This means "Newton's step is smaller than the parameter change that would move the point by `atol` in geometry." Further iteration cannot improve the result beyond the geometric tolerance.

**Why not `||G|| < atol`?** Two reasons:

1. **Near-miss intersections:** When curves pass within `atol` of each other but don't actually meet (imprecise input data), Newton converges to the closest approach point. The residual `||G||` is the closest distance — which is below `atol` but above Newton's internal tolerance. Using `||G|| < atol` would accept these, but it would also stop Newton prematurely for genuine intersections at acute angles where further iteration could improve precision.

2. **Boundary clamping:** Newton clips parameters to `[0,1]`. When the true zero is at the boundary, Newton stalls with zero step. The parametric tolerance check correctly identifies this as "not converged" (zero step trivially passes `|step| < ptol`, but the residual is large, and a guard rejects zero-step cases with nonzero residual).

### Zero-step guard

A special case: when Newton returns `last_step = (0, 0)`, it means either:
- Newton converged so well that no step was needed (residual already tiny), or
- Newton was clamped to the boundary and couldn't move

The guard distinguishes these: `step=0` is accepted only if `||G|| < atol` (genuine convergence). Otherwise it's rejected (boundary stall).

## Handling boundary intersections (BOUNDARY_ZERO)

When the classifier reports `BOUNDARY_ZERO`, the intersection lies exactly at the `[0,1]` domain boundary of one or both Bezier segments. This happens when the NURBS curve was decomposed at a knot, and the intersection falls on that knot.

**Why Newton doesn't work here:** Newton is clamped to `[0,1]`. When seeded near `u=0` or `u=1`, it can't step past the boundary. It stalls with zero step and a nonzero residual that may or may not be below `atol`.

**The solution:** Skip Newton entirely. The 1D solver already found the precise parameter on the boundary face. Evaluate both curves directly at those parameters and check the geometric distance:

```python
pt1 = C1(u_boundary)
pt2 = C2(v_from_1d_solver)
if ||pt1 - pt2|| < atol → accept as intersection
```

This is both faster and more reliable than Newton at boundaries.

## Handling overlaps

When the classifier reports `OVERLAP` (valley check confirmed), the overlap endpoints come directly from the boundary zeros — no tracing needed.

**Key insight from Bezier theory:** Two Bezier curves can overlap only over a continuous interval that starts at an endpoint of one curve and ends at an endpoint of the other. This means overlap endpoints always lie on the boundary of the `[0,1]^2` parameter domain. The 1D boundary solver finds them exactly.

**Post-processing:** The classifier may produce false `OVERLAP` classifications near tangent intersections (where the squared distance is very flat — second-order contact makes the valley check ambiguous). Post-processing verifies each overlap geometrically by sampling points along the reported overlap path and checking that `C1(u) ≈ C2(v)` at every sample. False overlaps are collapsed to a single isolated point via Newton.

## Subdivision strategy

When the classifier returns `INDETERMINATE`, the algorithm subdivides:

1. **Axis selection:** Split along the axis with the larger parameter span (`u1-u0` vs `v1-v0`).

2. **Net subdivision:** The sq-dist net `F` is split along the chosen axis at `t=0.5` using De Casteljau. This is exact — no recomputation from the control points. Each child inherits a net that covers its half of the parent's parameter domain.

3. **Curve subdivision:** The corresponding curve is also split at `t=0.5` using De Casteljau, producing two child control polygons.

4. **Weight tracking:** For rational curves, the child weights are extracted from the subdivided control polygon's last column.

5. **Termination:** Subdivision stops at `max_depth=50` or `max_cells=100,000`. At max depth, Newton is tried from the cell center as a last resort.

## NURBS integration (_nccx4.py)

`nurbs_ccx_multiple` wraps `bez_ccx` for multi-curve NURBS intersection:

1. **Decomposition:** Each NURBS curve is decomposed into Bezier segments at its internal knots via `decompose_curve`.

2. **BVH filtering:** All segments are inserted into a bounding volume hierarchy (BVH). Only segment pairs whose AABBs overlap are tested.

3. **Per-pair intersection:** Each BVH-matched pair calls `bez_ccx` with the segment control points.

4. **Parameter mapping:** Local Bezier parameters `[0,1]` are mapped to global NURBS parameters via the segment's knot interval.

5. **Deduplication:** Because adjacent segments share knot endpoints, the same geometric intersection appears from both segments at their shared boundary. Deduplication groups results by canonical curve pair `(min(i,j), max(i,j))`, computes parametric tolerances from the full NURBS curves, and merges entries where both `|du| < ptol_a` and `|dv| < ptol_b`.

This ensures:
- Close-but-distinct intersections from different curve pairs are preserved
- Same-pair double crossings at distant parameters are preserved
- Only true knot-boundary duplicates are merged

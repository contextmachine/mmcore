# Bezier Curve-Curve Intersection: v4 Algorithm

## Overview

`_bez_ccx4.py` implements curve-curve intersection (CCX) built on a single unifying concept: the **squared-distance Bernstein control net**. Given two Bezier curves `C1(u)` and `C2(v)`, the algorithm constructs the bivariate polynomial `D(u,v) = ||C1(u) - C2(v)||^2` in Bernstein form and analyzes its structure to classify intersections — without Jacobian-rank analysis, SVD, or overlap tracing.

The algorithm achieves performance comparable to commercial CAD engines while being implemented entirely in Python (with NumPy and Cython-accelerated Bernstein evaluation). The key to this performance is not brute-force optimization but **architectural choices that minimize wasted work**: boundary analysis runs once, overlaps are detected from net structure, and every subdivision cell is filtered by a cascade of increasingly expensive checks before Newton is invoked.

### Module map

| Module | Role |
|--------|------|
| `ccx/_bez_ccx4.py` | Main algorithm: two-phase architecture |
| `ccx/_nccx4.py` | NURBS adapter: decomposition, BVH, parametric dedup |
| `_sq_dist_classify.py` | Classifier: analyzes the sq-dist net |
| `_bern_zero_1d.py` | 1D solver: finds zeros on boundary faces |
| `_bezier_common.py` | Shared utilities: evaluation, Newton solver |
| `bern_sq_dist.py` | Net construction: builds `D(u,v)` from control points |

## The squared-distance net

The core data structure is the Bernstein control net of `D(u,v) = ||C1(u) - C2(v)||^2`. For polynomial curves of degrees `p` and `q`, this is a bivariate polynomial of degree `(2p, 2q)` with shape `(2p+1, 2q+1)`.

For rational curves in homogeneous form, the net represents the *numerator* of the squared distance (avoiding division by weights). The actual distance is recovered via weight correction: `d^2 = D / (w1 * w2)^2`.

**Why this representation:** The Bernstein coefficients provide guaranteed bounds on the function's range (convex hull property). If all coefficients are positive, the distance is provably nonzero everywhere. The net is constructed once and subdivided alongside the curves — De Casteljau split is exact and incremental, requiring no recomputation from scratch.

## Two-phase architecture

The central architectural insight: **for Bezier curves, overlaps and boundary intersections can only exist at the boundaries of the original objects.** Subdivision cannot create new ones — only artificial boundary zeros at cell edges. Therefore:

- **Phase 1** runs once on the initial patch: boundary analysis + overlap detection + cutout
- **Phase 2** operates on the remaining parameter intervals: only isolated intersections, no boundary machinery

This eliminates the single largest source of wasted work in naive subdivision algorithms: re-running boundary analysis and overlap checks on every subdivision cell.

### Phase 1: Boundary analysis and overlap

1. **Classify** the initial sq-dist net via the classifier hierarchy (min-of-net, Lipschitz, boundary zeros, uniqueness, overlap)

2. **Boundary zeros**: The 1D Bernstein solver (`_bern_zero_1d.py`) finds precise parameter values where `D` reaches zero on each boundary face. The solver uses derivative sign-change counting (Descartes' rule) for fast pruning, and subdivides away from roots (center of the longest positive coefficient run) to avoid fragmenting zeros.

3. **Overlap detection**: The valley check steps inward from each boundary zero along the connecting direction and verifies `D` stays near zero. For CCX (2D nets), the valley is approximately linear. The overlap endpoints come directly from the boundary zeros — no tracing needed, because a Bezier overlap must start and end at curve endpoints.

4. **Cutout**: Overlap and boundary intersection regions are removed from the first curve's parameter axis. Phase 2 receives only the remaining u-intervals, each guaranteed free of overlaps and boundary intersections. The cutout is 1D (along u only) — overlaps run diagonally in (u,v) space, so axis-aligned 2D cutout would miss the diagonal path.

### Phase 2: Isolated intersection search

For each remaining u-interval × full v-range:

1. **AABB prune**: Compare control-point bounding boxes of the two curve segments. If disjoint (inflated by atol), skip. This is the cheapest possible check — comparing 6 numbers — and catches the majority of cells in the subdivision tree.

2. **Min-of-net prune**: If `min(F) / w_max^2 > atol^2`, no zero exists in the cell.

3. **Lipschitz tightening**: Evaluate `F` at the midpoint and bound the deviation using partial derivative sup-norms.

4. **Derivative sign pruning**: If any partial derivative net `∂D/∂u` or `∂D/∂v` has all same-sign coefficients, no stationary point exists — the gradient can't vanish. This is the 2D analogue of the Poincaré-Miranda existence test.

5. **ptol-based termination**: If the cell is smaller than the parametric tolerance in both directions, report the center as a micro-fragment and stop.

6. **Newton**: LM-damped Gauss-Newton from the cell center, run on the original (un-subdivided) curves with global parameters. Convergence is determined by comparing the last step against parametric tolerances computed from the curve geometry (`bez_curve_param_tolerance`), not by an arbitrary residual threshold.

7. **Cutout after Newton**: When Newton finds a new intersection, the cell is split into 3×3 = 9 boxes (along both axes at `sol ± ptol`), the center box (containing the found intersection) is discarded, and the remaining 8 are pushed onto the stack. This prevents re-convergence to the same root and enables discovery of multiple intersections in the same region.

8. **Converged-outside-cell prune**: If Newton converges to a point outside the cell's parameter range, the cell doesn't contain an intersection — prune immediately. This is the most impactful single optimization for cases where the curve approaches an already-found intersection: Newton quickly escapes to the known point, the range check rejects it, and the cell is pruned without further subdivision.

### Newton convergence design

Newton's convergence is **not** determined by the residual `||G||`. Instead:

- Newton runs with a very tight internal step tolerance (`1e-14`) — effectively until it stalls
- The step tolerance acts as a stall guard, not a convergence criterion
- **Convergence** is decided by comparing the last step `(du, dv)` against the **parametric tolerances** `ptol_u`, `ptol_v`, computed from the curve geometry
- The parametric tolerance represents the maximum parameter change that corresponds to geometric deviation ≤ atol, accounting for the curve's speed (derivative magnitude)

This design was motivated by a fundamental insight: the geometric tolerance `atol` and Newton's internal accuracy are different things. A curve-curve pair with imprecise input data may have Newton's residual floor above Newton's internal tolerance but well below `atol` — the parametric tolerance correctly identifies this as "converged in the parameter-space sense."

**Zero-step guard**: When Newton returns step = (0, 0), it either converged perfectly (residual ≈ 0) or was stuck at a clamped boundary. The guard accepts zero-step only if the residual is below `atol` — rejecting boundary-clamped stalls that would otherwise produce false positives.

### BOUNDARY_ZERO handling

When the classifier finds zeros on the domain boundary (via the 1D solver), these represent intersections at Bezier segment boundaries — typically from NURBS knot decomposition. Newton can't converge at clamped boundaries, so these are handled by **direct curve evaluation**: evaluate both curves at the boundary parameters and check `||C1(u) - C2(v)|| < atol`.

This is Phase 1 only. Phase 2 never encounters boundary zeros because the cutout has removed all boundary intersection neighborhoods.

## NURBS integration

`_nccx4.py` wraps `bez_ccx` for multi-curve NURBS intersection:

1. **Decomposition**: Each NURBS curve is split into Bezier segments at internal knots
2. **BVH filtering**: Only segment pairs with overlapping AABBs are tested
3. **NURBS-level distance verification**: After mapping local Bezier params to global NURBS params, the geometric distance is verified using the original NURBS curves. This catches false positives at knot seams where the Bezier-level distance differs from the NURBS-level distance.
4. **Parametric deduplication**: Groups results by canonical curve pair, computes parametric tolerances from the full NURBS curves, and merges entries where both `|Δu| < ptol_a` AND `|Δv| < ptol_b`. This only merges true span-boundary duplicates — close-but-distinct intersections from different curve pairs or same-pair double crossings at distant parameters are preserved.

## Performance characteristics

The algorithm's performance comes not from micro-optimization but from **eliminating unnecessary work at the architectural level**:

- Phase 1 boundary analysis runs once, not per subdivision cell
- The pruning cascade (AABB → min-of-net → Lipschitz → derivative sign) catches cells at the cheapest possible level
- The 2D cutout after Newton prevents O(n²) re-convergence in multi-intersection regions
- Converged-outside-cell pruning provides O(1) rejection for cells in the attraction basin of known intersections
- 1D parameter cutout (not 2D box cutout) correctly handles diagonal overlaps

On representative benchmarks, the algorithm is 2-50x faster than the previous implementation across all test cases, while producing fewer duplicates and no false positives.

# bez_ssx v5 — Comprehensive Context for Next Session

This document captures everything learned during the SSX v5 design and implementation work. It is meant to be pasted as the initial prompt for the next session so we do not lose context. It is also intended to form the basis of the eventual algorithm documentation, so it is verbose and narrative in places.

---

## 0. Immediate task for next session

**Refactor `mmcore/numeric/intersection/ssx/_bez_ssx5.py` to remove dedup / merge / filter code that was added to work around a bez_csx bug that has since been fixed.**

Context: while debugging SSX we noticed duplicate / spurious branches in simple cases (e.g. two planes that intersect in a single straight line were returning 3 branches). We assumed this came from genuine topological ambiguity and wrote a lot of heuristic post-processing (`_filter_corner_touches`, `_merge_adjacent_branches`, `_dedup_branches`, `_dedup_overlaps`, `_dedup_crossings`) to clean up results.

In the last commit (`6d25fcc`) we discovered the root cause was in `bez_csx` — it was misinterpreting a stalled Newton as a found intersection, without checking the actual residual. Fixing CSX made the SSX "planes" case go from 3 branches to 1 immediately, and case 5 time dropped from 1.04s to 0.51s.

Now we need to carefully remove the heuristic cleanup code that was covering up CSX's false positives, because some of it actively hurts accuracy (we've seen legitimate distinct branches being merged because their endpoints happened to be geometrically close). Every removed piece must be verified against all test cases.

---

## 1. Project overview

`mmcore` is a modern CAD engine in Python + Cython. The intersection stack has three layers:

- **CCX** (curve-curve): `mmcore/numeric/intersection/ccx/_bez_ccx4.py` + `ccx/_nccx4.py` — **working, mature**
- **CSX** (curve-surface): `mmcore/numeric/intersection/csx/_bez_csx4.py` + `csx/_ncsx4.py` — **working, mature, one bug just fixed**
- **SSX** (surface-surface): `mmcore/numeric/intersection/ssx/_bez_ssx5.py` — **under active development, this project**

All three layers share the **squared-distance Bernstein net** foundation from `mmcore/numeric/bern_sq_dist.py`.

Main branch: `tiny`. Current dev branch: `0.53.0`. Python 3.14 in `.venv/bin/python`.

---

## 2. Reference papers

Both PDFs are in the repo root.

### 2.1 Krishnan & Manocha 1997 — `237748.237751.pdf`
*"An Efficient Surface Intersection Algorithm Based on Lower-Dimensional Formulation"*

What we borrowed:
- **Domain decomposition**: subdivide the parameter domain at isoparametric lines passing **through boundary crossing points**, NOT at midpoints. This is guided subdivision vs. blind bisection. Each cut creates new crossings on the internal isoline via a CSX call, and those new crossings are shared between adjacent sub-cells.
- **Domain decomposition terminates early**: a sub-cell with exactly 2 boundary crossings and certified loop-free contains exactly 1 curve component → safe to trace. Does not require cells to be tiny.
- **Terminal condition for decomposition**: when the sub-cell can be handled by tracing, stop subdividing. The loop-free certificate is the enabling precondition.

What we did NOT borrow:
- Their complex tracing of eigenvalue paths through the imaginary plane for loop detection (specific to their matrix formulation, not compatible with our Bernstein framework).
- Their unevaluated-determinant matrix representation of the intersection curve.

### 2.2 Cheng, Zhang, Xiao, Li 2023 IATA — `3592452-2.pdf`
*"Topology-driven approximation to rational SSI via interval algebraic topology analysis"*

What we borrowed:
- **TΨᵢ monotonicity criterion for loop absence.** The tangent vector of the 4D implicit curve Ψ = R1(s,t) − R2(u,v) is formed from the 3×3 minors of the 3×4 Jacobian:
  ```
  TΨ₁ = det(J without column 1),  TΨ₂ = det(J without column 2), etc.
  ```
  If any one `TΨᵢ` has all Bernstein coefficients of one sign (non-negative OR non-positive) over a box B, the curve is monotonic in variable i within B → **no interior loops**.

- **Three-way classification** when monotonicity fails:
  - All TΨᵢ straddle zero, but TΨ = 0 has no simultaneous solution → *regular turning points* (subdivision resolves them)
  - TΨ = 0 has a simultaneous solution → *true tangency (C₂)* → deflation needed

- **Deflation for C₂ cases**. The intersection curve on a tangent surface has rank-2 Jacobian (the Ψ Jacobian is degenerate). Marching along Ψ fails. Augment with TΨ to form the deflated system:
  ```
  Δ = Ψ ∩ TΨ   (3 + 4 = 7 equations in 4 unknowns)
  ```
  At a tangent point, the singular zero of Ψ becomes a regular zero of Δ.

- **Regulated system Φ** for tracing tangent curves:
  ```
  Φ = {Φ₁, Φ₂ ∈ Ψ, Φ₃ ∈ TΨ}   (3 equations in 4 unknowns → 1D curve)
  ```
  At a tangent point, the Ψ Jacobian has rank 2 (one tangent direction undefined), but the Φ Jacobian has rank 3 because the TΨᵢ gradient adds an independent direction. Marching Φ works where marching Ψ fails.

What we did NOT borrow:
- Their interval Newton / Krawczyk-based certification machinery (too heavy for our needs, and our tests showed it takes ~0.5s per check even when it returns "undetermined").
- The `analyse_deflated_system` from `_deflate.py` — we tried using it and it did not trace tangent curves for our case 1, returning only the 2 endpoints. We replaced it with our own Φ-marcher.
- Their full interval algebra implementation.

---

## 3. The sq-dist Bernstein net foundation (our core innovation)

This is the glue. It's also non-obvious and worth explaining carefully.

### 3.1 What it is

For two Bezier surfaces S1(s,t), S2(u,v) in homogeneous form, we build a 4-variate Bernstein polynomial:

```
D(s,t,u,v) = || S1(s,t) - S2(u,v) ||²
```

expressed as a 4D tensor of Bernstein coefficients. The function is `surface_surface_distance_squared_net_homog(S1_h, S2_h, rational)` in `mmcore/numeric/bern_sq_dist.py`.

Shape: `(2p1+1, 2q1+1, 2p2+1, 2q2+1)` — degree doubles because the square of a degree-p polynomial has degree 2p.

### 3.2 Non-obvious aspects

**1. Coefficients can be negative for homogeneous form.**

Even though D ≥ 0 everywhere as a function, the Bernstein control points of the numerator (before dividing by `w_S1² · w_S2²`) can be negative. This is because what we store is the **numerator** of the rational squared distance; the actual squared distance is `F_poly / (w_S1(s,t)² · w_S2(u,v)²)`.

This does NOT break our pruning logic because we check `min(F) / w_scale² > atol²`, where `w_scale = max_weight_product`. If `min(F)` is negative we can't prune but we haven't made a wrong claim.

**2. Lipschitz tightening works here.**

The Bernstein convex hull bound `[min(F), max(F)]` is loose. We tighten it using coefficient differences (the Lipschitz constant of a Bernstein polynomial on [0,1] is bounded by `degree × max|Δ coef|`). See `_check_lipschitz` in `_sq_dist_classify.py`. This makes min-of-net pruning significantly more effective.

**3. The net is expensive to build for high degrees.**

For bicubic × bicubic surfaces (degrees 3×3 × 3×3), the sq-dist net has shape `(7, 7, 7, 7)` = 2401 coefficients. For bi-quintic it jumps to `(11, 11, 11, 11)` = 14641. This is a known cost but acceptable.

**4. Rationality is baked in.**

The net is built in homogeneous form. For non-rational input (all weights = 1), the net is still correct — it's just that `w_scale = 1`. No special-case code needed.

### 3.3 How we use it in bez_ssx

- Level 1 (pruning): `_prune_ssx_cell` — builds F once, checks min-of-net and Lipschitz. Early exit if provably no intersection.
- We do NOT currently use F in every sub-cell of the domain decomposition (we rely on Gauss map separability instead, which is cheaper but perhaps less discriminating).

### 3.4 Relationship to TΨᵢ

The sq-dist net gives us a fast no-intersection check. TΨᵢ gives us a no-loop check. They are independent tools:

- `min(F) > atol²` → no intersection anywhere in the box (terminate)
- `min(TΨᵢ) > 0 OR max(TΨᵢ) < 0` for some i → no loops in the box (safe to decompose)
- Both fail → must subdivide and/or deflate

### 3.5 Why the monotonicity check works — Cheng et al. Lemma 5

Paper 2's Lemma 5 (Section 4.4) gives the precise statement: given a 4D box B and `0 ∉ TΨᵢ(B)` for some `i ∈ {1,2,3,4}`, (a) if Ψ ∩ ∂B = ∅ then B contains no intersection point, (b) if Ψ ∩ ∂B = {p, q} (two interior-of-face boundary crossings) then B contains a single connected intersection segment from p to q.

The proof is straightforward: a nonzero `TΨᵢ` everywhere in B means `dsᵢ/d(arc length)` has definite sign on any intersection curve in B, so the curve is monotonic in parameter `sᵢ`. A monotonic 1D curve cannot form a loop, cannot self-intersect, and its topology in B is fully determined by the boundary crossings.

The check on Bernstein form is exact (not conservative): if any single `TΨᵢ` coefficient array has all non-negative or all non-positive values, the polynomial is ≥ 0 (or ≤ 0) by the convex hull property, and we have the certificate. Importantly, boundary-touching zeros (e.g. `min(TΨᵢ) == 0` achieved only at a corner of [0,1]⁴) still count as "definite sign, no interior zeros" — this is why we use `>= 0` and `<= 0` rather than strict inequalities.

### 3.6 Why the deflation/Φ-tracer works — Cheng et al. Lemma 2

Paper 2's Lemma 2 (Section 4.2.3) is the enabling observation for tangential intersections:

> If p is an isolated tangent point of Ψ (so Ψ(p) = 0 and TΨ(p) = 0), then p is a **regular zero** of the over-determined system {Φ₁, Φ₂ ∈ Ψ, Φ₃, Φ₄ ∈ TΨ}. Furthermore, any closed loop of the intersection must have **at least two points** on the regular curve defined by the reduced Φ = {Φ₁, Φ₂ ∈ Ψ, Φ₃ ∈ TΨ} (3 eqs, 4 unknowns → 1D curve).

The intuition: the Ψ Jacobian at a tangent point has rank ≤ 2 (the surfaces share a tangent plane, so two partial derivative vectors are parallel and the 3×4 Jacobian has at most rank 2). Marching Ψ fails because the 4D tangent direction — the null vector of Jψ — is 2-dimensional and ambiguous.

Augmenting with a TΨᵢ row restores rank 3. The TΨᵢ gradient is generically transverse to the nullspace of Jψ, so the Φ curve passes through the tangent point with a well-defined tangent. Marching Φ works, and we can filter marched points by whether they also satisfy the full Ψ=0 (the third Ψ component that wasn't included in Φ).

This is why our Φ-tracer can reuse the same predictor-corrector marcher as the main algorithm — only the system being solved changes.

---

## 4. Previous SSX implementations — what we keep, what we reject

The old SSX lives in `mmcore/numeric/intersection/ssx/_ssx4.py` (keep for reference — we import from it). Earlier versions: `_ssx31.py`, `_detect_intersections.py`. None of them fully worked.

### 4.1 What we keep from `_ssx4.py` (import, do not copy)

- **`GaussMapBern` class** (lines ~331-572). Builds the Gauss map of a homogeneous Bezier surface as a Bernstein tensor. Splits surface and Gauss map simultaneously via De Casteljau (`split_u`, `split_v`, `split_uv`). Caches bounding box, mean normal, gauss radius, gauss variation. Built once at top level of bez_ssx, split alongside surfaces during domain decomposition — NO recomputation from scratch.

- **`separate_gauss_maps(dirs1, dirs2)`** — hemisphere witness algorithm. Returns `(p1, p2)` where both non-None means the two sets of normal directions can be separated on the unit sphere → no loops possible. Uses the Cython `hemisphere_witness_incremental_fast` from `_cap_witness`.

- **Data structures**: `SSXBranch`, `SSXPoint` (we use these as our output types).

- **`split_tensor_bezier_axis`** helper for de Casteljau on any tensor shape.

### 4.2 What we reject from old SSX

- **Recursive control flow.** Recursion was opaque, hard to reason about, and prevented building the global topology structure we need for branch merging. We use stack-based iteration.

- **`trace_between` / `adaptive_refine_bruteforce`** — old tracer in `ssx/trace_inter_segm.py`. It was slow and fragile. Our curvature-adaptive predictor-corrector marcher (`_march_intersection_curve`) is simpler and more accurate.

- **`analyse_deflated_system` from `_deflate.py`** — over-engineered for our needs. It tries to do dimension estimation, Krawczyk isolation, cover building, etc. When we tested it on our case 1 (tangential saddles), it returned 2 boundary points and failed to trace the tangent curve. We replaced with our own Φ-marcher.

- **Magic point Newton for hard cases** — this was a bolt-on heuristic in `_ssx4.py`. Superseded by the Φ-tracer (which is principled and works on the same marching infrastructure).

- **Speculative deflation before knowing the cell is tangent** — we use Krawczyk-based `_check_tangency` to decide *if* deflation is needed, only invoking Φ-tracer for confirmed tangent cases.

### 4.3 What we use from `_deflate.py` (the building blocks, not the full pipeline)

- **`minors_Tpsi_from_control_nets(P1, P2)`** — computes TΨ¹, TΨ², TΨ³, TΨ⁴ as 4D Bernstein nets via cross products and dot products of the derivative nets. Exact algebra in Bernstein form, no numerical approximation. This is the heart of the Cheng et al. monotonicity / deflation framework.

- **Interval machinery** (`DeflatedSystem`, `SquareSystem`, `_krawczyk_operator`, `gauss_newton_witness`) — available as primitives. We currently use them only in `_check_tangency`, and even there we stop after Gauss-Newton (skip the expensive Krawczyk subdivision).

---

### 4.4 Design principles developed during this project

These are principles the user articulated (sometimes as direct quotes) that should govern all future work:

**"No heuristics — it's that simple."**
Applied to `_trace_all_branches`: when the marcher arrives at a boundary, that point IS an endpoint, match it by exact face + parameters. Don't use proximity-nearest-neighbor search, don't use distance thresholds. The marcher doesn't go outside the sub-cell by construction, so its stopping point is always a boundary crossing.

**"Non-monotonicity ≠ tangency."**
All four `TΨᵢ` straddling zero only means the curve has turning points somewhere in the box. Regular loops have turning points without being tangent. True tangency requires all `TΨᵢ = 0` at the *same* point (their zero sets intersect simultaneously). Distinguishing these cases requires a simultaneous-zero check (interval Newton / Krawczyk on the 4×4 system TΨ = 0).

**"The goal is a loop-absence certificate, not subdivision depth."**
We do not subdivide for its own sake, and we do not subdivide until cells are "small enough". We subdivide only to turn a cell that *cannot yet* certify loop-absence into sub-cells that *can*. A cell terminates (stops subdividing, proceeds to tracing or deflation) as soon as ANY of these certificates holds:

- **TΨᵢ monotonicity** (any one TΨᵢ has definite sign over the cell — Paper 2 Lemma 5). Cheap: O(n) scan of Bernstein coefficients.
- **Gauss map separability** (hemisphere witness exists — from old `_ssx4.py`). Independent of TΨᵢ and often works where monotonicity fails (e.g. when the surfaces are both curved but facing away from each other).
- **Confirmed tangency** (interval Newton certifies TΨ = 0 has a simultaneous solution in the cell). Terminates with Φ-tracer / deflation, NOT with tracing.
- **Other singular certificates from Paper 2** (the C₁ / C₃ / mixed cases described in sections 4.1/4.3 of the paper — NOT YET IMPLEMENTED, but they belong in the same category: algebraic proof that the topology inside the cell is of a specific, traceable type).

If none of these certificates hold on the current cell, we fall through to subdivision as a way of producing sub-cells on which some certificate will hold. That is, subdivision is a *fallback*, not a default. The certificate checks must be ordered from cheapest to most expensive (currently: TΨᵢ monotonicity → Gauss map separability → tangency → subdivide). See section 12.5 on the check order.

This matters because each subdivision is expensive (one isoline CSX + two sub-cell checks) and because unnecessary subdivision inflates the branch-merging problem in the final output assembly.

**"Cut through crossing parameter values, not at gaps."**
From Paper 1 but it was easy to miss. Our first attempt cut at the midpoint of the widest gap between crossings (naive bisection). Case 5 exploded because this produced sub-cells that still had complex topology. Cutting at an actual crossing's s-value (or t, u, v) means that crossing is ON the partition curve, producing a clean decomposition where each sub-cell has a manageable number of crossings.

**"Iterative with a global topology structure beats recursion every time for complex algorithms."**
See section 4.5.

**"Identical s, t, u, v → topologically identical, safe to dedupe. Identical xyz → maybe a genuine fold, do NOT dedupe."**
Applied to corner crossings: the same stuv produced by CSX on two adjacent boundary faces must be merged. But two distinct parameter points that project to the same xyz are a legitimate topological feature (e.g. a fold or a surface self-intersection) and must be preserved.

**"Trust the marcher's stopping point."**
The marcher is bounded by the cell's [0,1]⁴. It cannot exit. When it stops, either it reached a boundary (which IS where a crossing must be) or it genuinely failed (and we should know and handle that). There's no "marcher went a bit past the boundary" case. See section 4.6.

**"No redundancy when the common partition can be known to both sides."**
Each internal partition curve is shared between exactly two sub-cells. Intersection points on the partition curve belong to both. A proper topology structure makes branch merging a simple parameter-match on the partition, not a geometric heuristic. See section 4.7.

**"Newton converges at the minimum, not at the root. Don't confuse the two."**
The CSX bug fix. A stationary point of the distance function is NOT an intersection. Newton convergence gives us *some* point where derivative = 0; we must separately check whether that point has residual < atol before calling it an intersection.

### 4.5 Why iterative, not recursive

We started with recursive subdivision (it felt natural). It caused three specific problems that ultimately motivated the rewrite:

1. **The algorithm structure was opaque.** Each `_subdivide_and_recurse` call went a level deeper without knowing the total depth or the siblings. Debugging required mentally simulating a nested call stack.

2. **Case 5 explosion.** With 4 boundary crossings and non-monotonic TΨᵢ, the recursive version made 62+ Gauss map checks, each triggering potentially more CSX calls. Profiling showed thousands of redundant operations. The issue was that each recursive call had no knowledge of what siblings had already done.

3. **Impossible to build the topology layer.** The desired design (section 4.7) requires global state: a registry of PartitionCurve objects with their adjacent cells, and a mechanism to merge branches that end on the same partition. This is natural in an iterative framework where the main loop owns the state. In a recursive framework, either you thread state through every call (awkward) or you use global mutable state (fragile).

The iterative version solves all three:

- The stack + main loop make the algorithm transparent. You can log per-iteration and see exactly what's happening.
- Global state (the list of branches found, the map of partition curves, etc.) lives in the main function.
- You can inspect and intervene: "if the stack grows to 100 cells, dump its state and stop."

### 4.6 Boundary-crossing classification: entry, exit, or both

This supersedes a simpler "zero-length march as classifier" idea I had earlier. The simpler idea was wrong and led to incorrect handling of corner crossings, which is one of the root causes of the odd-crossing-count warnings in the current code.

#### 4.6.1 The setup

A crossing has a 4D local parameter `stuv ∈ [0,1]⁴`. "On-boundary" axes are those `i` where `stuv[i] ∈ {0, 1}` (within tolerance). By definition, any boundary crossing has at least one on-boundary axis. A **corner** crossing has two or more.

At every such crossing we compute the raw 4D tangent `T` of the intersection curve (the SVD null vector of the 3×4 Jacobian, unclamped — NOT the clamped step the marcher uses during integration). The sign of `T` is arbitrary (it's a null vector) but the **relative** signs across axes are invariant.

#### 4.6.2 The classification rule

For each on-boundary axis `i`, compute:

| `stuv[i]` | `sign(T[i])` | classification (for axis `i`) |
|-----------|--------------|-------------------------------|
| 0         | +1           | **entry** (moves into `(0, 1]`) |
| 0         | -1           | **exit**  (moves toward negative, out of cell) |
| 1         | -1           | **entry** (moves into `[0, 1)`) |
| 1         | +1           | **exit**  (moves toward 2+, out of cell) |

Compact form: the tangent component points **toward the cell interior** for entries, **toward the exterior** for exits. Using `dir_in_i = 1 - 2*stuv[i]` (+1 at 0, -1 at 1), the rule is `sign(T[i]) == dir_in_i` → entry, else exit.

#### 4.6.3 Corner crossings: a single point can be both entry AND exit

A corner crossing (≥2 on-boundary axes) is classified **independently on each on-boundary axis**. Outcomes:

- **All entries** → the curve enters the cell at this corner, traces through the interior, and eventually exits somewhere else.
- **All exits** → the curve exits the cell at this corner; it entered somewhere else.
- **Mixed (some entries, some exits)** → the curve just **touches the corner**: it enters the cell via one face, immediately exits via another. Zero-length path inside the cell.

The critical observation: **a mixed corner crossing must be counted twice** — once as an entry (on the entry faces) and once as an exit (on the exit faces). It represents a "through-touch" where the curve grazes the corner of this sub-cell on its way between neighbors.

This is precisely the source of the "odd number of crossings" warning we've been seeing. In the current code, a mixed corner is counted once as a cell-level crossing, but it contributes to TWO different isoline-level crossing registers. The parity mismatch is real and expected.

#### 4.6.4 The data structure we need

Replace the current cell-level crossing list with an **isoline-indexed registry**. Each boundary curve — either an outer isoline of the original surface OR an internal partition isoline — owns a list of `(point, isoline_param, direction)` tuples where:

- `point` is the 4D crossing
- `isoline_param` is the scalar parameter along this specific isoline (single 1D coordinate)
- `direction` is `"in"` or `"out"` for the cell this registration belongs to

A single 4D crossing can appear on multiple isolines (e.g., a corner at `s=0, t=0` lives on the isoline `s=0` AND on the isoline `t=0`). It appears as an "in" on some and an "out" on others according to the per-axis classification of §4.6.2.

Each isoline is also tagged with its global parameter interval — for an internal partition, this is the range on the split axis in the parent's global coords; for an outer isoline, it's the global parameter range from the NURBS-level decomposition.

#### 4.6.5 Tracing with this structure

Within a cell, the marcher picks an unvisited `"in"` entry, marches until it reaches a boundary (which must be another registered crossing on some isoline), and records the traced path as `(start_isoline, start_param, end_isoline, end_param, path_points)`. The `"in"` entry is consumed; the corresponding `"out"` entry at the end is consumed.

If the marcher starts at a **mixed corner**, it behaves naturally: it will register a real march on the entry axes and also needs the mirror "exit" mark consumed on the exit axes. Conceptually: a through-touch point is simultaneously "in" (on entry isolines) and "out" (on exit isolines), and a single marcher invocation consumes one in and one out.

#### 4.6.6 Merging across internal partitions

Each internal partition isoline is shared between exactly two sub-cells (left and right). It maintains **two** registers — one from each adjacent cell. After all cells in both adjacencies have been traced, merging is a 1D matching:

For each internal partition isoline:
- `left_cell.out_list[isoline]` must match 1:1 with `right_cell.in_list[isoline]` by `isoline_param`
- Each matched pair is one continuous branch crossing this partition; concatenate the paths

Because each register contains only `(param, direction)` pairs on a single 1D isoline, matching is trivial sort-and-pair on `isoline_param`. No 4D proximity search, no geometric heuristic, no distance threshold. If the registers don't match 1:1, that's a diagnosable bug (odd entries, missed crossings, or the classification was wrong) — not silently covered up.

#### 4.6.7 Implementation implications

This replaces:
- `_trace_all_branches` (proximity-based endpoint matching) → becomes classification + march + register
- `_merge_adjacent_branches` (proximity-based concatenation) → becomes per-isoline 1D match
- `_filter_corner_touches` (heuristic corner filter) → becomes natural in the classification (a through-touch has matched in/out on different isolines; no filtering needed)
- `_dedup_branches` / `_dedup_crossings(xyz proximity)` → obsoleted by stuv-based dedup on isoline registers

What stays:
- The marcher itself (`_ssx_tangent_4d`, `_ssx_correct`, `_march_to_boundary`)
- The classification machinery (TΨᵢ, Gauss map, tangency check)
- The Φ-tracer for C₂ cases (Φ has its own topology; applies the same classification principles on the Φ curve)

The user's phrasing (paraphrased and expanded by discussion):
*"The marching algorithm should be run inside each of those cells. It should not extend beyond the boundaries of the subpatches. If a point is classified as entry on some axis and exit on others, it's a through-touch at a corner — count it on each isoline separately, as entry on some and exit on others. A single cell-level crossing can contribute to two or more isoline-level entries."*

### 4.7 The partition-curve topology layer (PLANNED, not yet implemented)

This section gives the concrete data-structure design that implements the classification framework of §4.6. Section 4.6 says *what* must happen (classify each axis as entry/exit, register on every isoline the point lies on, match in/out across adjacent cells). This section says *how* we store that information so the main loop and merge step can manipulate it.

Currently the main loop distributes "new crossings from isoline CSX" to left and right sub-cells as raw `BoundaryCrossing` objects without direction tags and without recording which isoline they belong to. This is the root cause of the fragile branch merging and of the odd-crossing warnings.

#### 4.7.1 Design

```python
@dataclass
class IsolineRegistration:
    """One crossing's registration on one isoline, from one adjacent cell's view."""
    point_id: int               # global id for this 4D crossing (see BoundaryPoint below)
    param: float                # 1D parameter along the isoline (scalar)
    direction: Literal["in", "out"]  # from the perspective of owner_cell
    owner_cell: _Cell           # which sub-cell this registration belongs to


@dataclass
class PartitionCurve:
    """An isoline that bounds one or more sub-cells.

    Outer partitions (from the original surface boundary) belong to exactly
    ONE cell — the cell whose box includes them as an outer face.
    Internal partitions (from domain-decomposition splits) belong to TWO cells,
    one on each side, with opposite in/out directions for the same crossing.
    """
    axis: int                       # 0..3 — which stuv coordinate is fixed
    value: float                    # global parameter value on that axis
    global_interval: tuple          # 3D: the fixed 3 coords, 1 None for the free axis
                                    # e.g. (0.5, None, 0.7, 0.3) means axis=1 (t) is free
    isoline: NDArray                # control points of the isocurve (1D bezier)
    adjacent_cells: list[_Cell]     # 1 (outer) or 2 (internal)
    registrations: list[IsolineRegistration]  # all in/out marks on this partition


@dataclass
class BoundaryPoint:
    """One 4D crossing point, potentially registered on multiple isolines."""
    stuv: NDArray                    # (4,) global parameter
    xyz: NDArray                     # (3,) Euclidean
    tangent_raw: NDArray             # (4,) unclamped tangent, for classification
    registrations: list[IsolineRegistration]  # one per isoline this point lies on


@dataclass
class _Cell:
    g1: GaussMapBern
    g2: GaussMapBern
    box: tuple                       # 4D global box
    partitions: list[PartitionCurve] # all isolines bounding this cell
                                     # (4 outer + N internal after splits)
    branches_in: list[BoundaryPoint] # unvisited registrations marked "in" on some partition
    depth: int
```

#### 4.7.2 How registrations are built

When CSX finds a crossing on an isoline, we do the following once, before the point is put into any cell:

1. Allocate a `BoundaryPoint` with global `stuv`, `xyz`, and the raw 4D tangent computed by `_ssx_tangent_4d` at the crossing WITHOUT clamping (see §4.6.1 for why raw).

2. For each on-boundary axis `i` (i.e. every `i` where `stuv[i] == 0` or `stuv[i] == 1` within tolerance): apply §4.6.2's rule to decide if this axis classifies as entry or exit **for the cell on whose boundary this point lies**.

3. Locate the `PartitionCurve` for axis `i` at value `stuv[i]` in the relevant cell(s) (outer isolines are pre-registered when the cell is created; internal isolines are registered when the cell is split).

4. Append an `IsolineRegistration(point_id=bp.id, param=isoline_param(stuv), direction=..., owner_cell=...)` to both `PartitionCurve.registrations` and `BoundaryPoint.registrations`.

5. For **internal partitions**, do this TWICE — once for each adjacent cell, with opposite directions. A point that's "out" from the left cell's perspective is "in" from the right cell's perspective, at the same `param` on the partition.

#### 4.7.3 Tracing using this structure

For each cell, the main loop consumes `cell.branches_in` in arbitrary order:

1. Pop an unvisited registration tagged `"in"`.
2. March from its `stuv` through the cell interior.
3. The marcher stops at a cell-boundary crossing (which must be another `BoundaryPoint` with a registration in this cell tagged `"out"` on some partition).
4. Record the traced path as:
   ```
   (start_point_id, start_partition_id, start_param,
    end_point_id,   end_partition_id,   end_param,
    path_points_xyz)
   ```
5. Mark both registrations as consumed. If either point has MORE unvisited registrations (a corner touch with multiple in/out roles), those remain for future marches.

#### 4.7.4 Merging across internal partitions

After every cell has been traced, for each internal `PartitionCurve`:

1. Collect `lefts = [r for r in registrations if r.direction == "out" and r.owner_cell is left_cell]`
2. Collect `rights = [r for r in registrations if r.direction == "in" and r.owner_cell is right_cell]`
3. Sort both by `param`.
4. Pair them 1:1 (by param, which must match within tolerance because they're the same physical crossing recorded twice).
5. For each matched pair, concatenate the two partial branches (one ending at `lefts[k]`, one starting at `rights[k]`) into a single branch.

No proximity search, no geometric heuristic. The matching is exact because `param` is a single scalar and the two registrations come from the same `BoundaryPoint`.

If the lists don't match (unequal lengths, or params don't align) that is a diagnosable bug in classification or registration, not something to paper over. The system should raise, log, or gracefully surface it — never silently discard.

#### 4.7.5 Why this is better than geometric merging

- **Merge complexity**: O(N log N) per partition (sort + walk), where N is the number of crossings on that partition — typically 0 to 4.
- **No false merges**: two truly distinct crossings on the same partition will have distinct `param` values; they won't be collapsed.
- **No missed merges**: a crossing that exists on both sides of an internal partition is inherently registered twice; we can always find the pair.
- **Debuggability**: you can print the registrations per partition and see the full topology.
- **Extensibility**: overlaps, tangent curves that cross partitions, and Φ-traced branches all fit the same `IsolineRegistration` vocabulary — they just need `direction` semantics adapted where appropriate.

#### 4.7.6 The overlap case fits naturally

A `BoundaryOverlap` is a 1D segment on an isoline — two endpoints and a curve between them. In this design, the overlap's two endpoints become two `BoundaryPoint`s each with their own `IsolineRegistration` on the isoline in question. The middle of the overlap is not a "point" — it's a curve segment registered on the partition as an overlap fragment (possibly with a separate `OverlapRegistration` variant).

Merging across internal partitions still works: if an overlap enters a sub-cell, leaves via a partition, and continues into the adjacent sub-cell, the two fragments will pair up by matching the overlap's parameters at the partition.

#### 4.7.7 Relation to §4.6 (classification)

Section 4.6 is the **specification**: what each crossing is (entry/exit/both on which isolines) and how merging must behave (1D matching on partition params).

Section 4.7 is the **implementation**: the classes that store the classification and make the operations concrete. If you can describe an algorithm in §4.6 terms, you can write it in §4.7 terms mechanically.

## 5. Architecture of bez_ssx v5

### 5.1 Five levels

```
bez_ssx(S1, S2, atol, rational):

  Level 1: Pruning
    - AABB non-overlap check (Euclidean control points)
    - Build sq-dist Bernstein net F(s,t,u,v)
    - min-of-net + Lipschitz → early return if provably no intersection

  Level 2: Boundary analysis
    - 8 CSX problems, one per face of [0,1]⁴:
      * s=0 face: isocurve S1(0,t) × full S2 → bez_csx
      * s=1, t=0, t=1, u=0, u=1, v=0, v=1 similarly
    - Returns BoundaryCrossing list (isolated points) + BoundaryOverlap list

  Level 3: Loop-absence classification
    - Build TΨ¹..TΨ⁴ once (minors_Tpsi_from_control_nets)
    - _check_loop_free(g1, g2, T1..T4):
      * TΨᵢ monotonicity: any TΨᵢ has definite sign → loop-free
      * Gauss map separability: separate_gauss_maps returns both witnesses → loop-free

  Level 4a: If loop-free
    - _trace_all_branches: marcher-driven topology discovery
    - Pick any unvisited crossing, march until boundary, match end to another crossing
    - Consume pair, repeat

  Level 4b: If not loop-free → iterative domain decomposition
    - Stack of _Cell objects (each stores g1, g2, crossings, box, depth)
    - For each cell:
      * If loop-free (Gauss maps check): trace and emit branches
      * Else: _choose_cut picks a crossing+axis for the internal isoline
      * Extract isoline, run ONE CSX call on it vs opposite surface
      * Split GaussMapBern at the cut (surface + Gauss map simultaneously)
      * Distribute crossings (original + new isoline crossings) to left/right
      * Push sub-cells onto stack

  Level 4c: If confirmed tangent (Krawczyk on TΨ=0) → Φ-tracer
    - _deflate_tangent_cell builds the regulated curve Φ
    - Marches Φ between boundary crossings
    - Filters points satisfying full Ψ=0 → tangent intersection curve

  Level 5: Output assembly (CURRENT HEURISTIC — TO BE REFACTORED)
    - _merge_adjacent_branches: concatenate where endpoints match
    - _dedup_branches: remove branches with same endpoints
    - Return {'branches': [...], 'points': [...]}
```

### 5.2 Coordinate system

**All crossings and branch endpoints are in GLOBAL [0,1]⁴ coordinates.** Surfaces inside sub-cells are in LOCAL [0,1]² after De Casteljau reparameterization. Conversion:

```python
global[axis] = cell.box[axis][0] + local[axis] * (cell.box[axis][1] - cell.box[axis][0])
local[axis]  = (global[axis] - cell.box[axis][0]) / (cell.box[axis][1] - cell.box[axis][0])
```

Helpers: `_local_to_global(stuv_local, box)`, `_global_to_local(stuv_global, box)`.

This replaced an earlier broken design where we tried to remap crossings between parent-local and child-local coords.

### 5.3 The marcher

`_march_intersection_curve` and `_march_to_boundary` in `_bez_ssx5.py`:

- Predictor: step along the 4D tangent direction (null space of the 3×4 Jacobian)
- Corrector: damped pseudoinverse Newton on Ψ=0
- Adaptive step: angle between consecutive tangents drives step size
- `_ssx_tangent_4d` — handles rank-deficient Jacobians via SVD + optional direction hint
- `_march_to_boundary` — marches until any parameter hits [0,1] edge (for `_trace_all_branches`)
- `_march_intersection_curve` — marches between two known endpoints (for `_trace_segment`)

Key fix in `_ssx_tangent_4d`: for a 3×4 Jacobian, the null space is always at least 1D. Earlier code checked `count(sigma < tol) > 0` which was wrong for full-rank matrices (rank 3 → null dim = 4-3 = 1, always). Now we compute `null_dim = 4 - rank` directly.

### 5.4 The Φ-tracer

`_march_phi_curve`, `_choose_phi_equations`, `_eval_phi`, `_jac_phi`, `_deflate_tangent_cell`:

- Pick 2 equations from Ψ (out of 3) and 1 equation from TΨ (out of 4) → 3 equations in 4 unknowns
- Choose based on conditioning of the resulting 3×4 Jacobian at the seed
- Marches the Φ curve using the same predictor-corrector as the main marcher
- At each marched point, verify full Ψ=0 (all 3 components) — only points satisfying both Φ=0 and full Ψ=0 are on the actual intersection

Case 1 (tangential crossed saddles) result: 1 branch, 9 points, `max|S1-S2| = 4e-15`, 45 ms. Tangent curve correctly traced.

---

## 6. What currently works and what does not

### 6.1 Working test cases

| Case | Input | Expected | Actual | Time |
|------|-------|----------|--------|------|
| Planes | z=5 plane vs z=x tilted plane | 1 branch (line x=z=5) | 1 branch | 13 ms |
| Transversal | bilinear 1 corner raised vs flat plane z=3 | 1 branch | 1 branch | 11 ms |
| Tangential | crossed bicubic saddles | 1 branch (Φ-traced tangent curve) | 1 branch, err 4e-15 | 40 ms |
| Overlaps | bilinear patch with 2 edges on a plane | 2 overlap branches | 2 overlap branches | 18 ms |
| Case 5 | wavy bicubic surfaces, 2 open branches | 2 branches | 2 branches | 0.51 s |

### 6.2 Not working

- **Case 6**: loop + open branch. The open branch is found, the loop is completely missed. The subdivision doesn't enter cells where there are no external boundary crossings — so cells containing an entire interior loop are never processed. This is a hard structural bug, not a heuristic issue.

### 6.3 Performance

Case 5 at 0.51s is acceptable but there's a ~60× overshoot in subdivision depth — profiling showed 62 Gauss map separability checks where a well-designed algorithm should do 2-4. The excess subdivisions all yield loop-free sub-cells; the cost is the checks themselves plus the isoline CSX calls.

---

## 7. The recent critical bug fix (commit `6d25fcc`)

### 7.1 The bug

`bez_csx.py :: _phase2_isolated_search` — the Newton convergence criterion accepted any parametric convergence as "found an intersection", without checking the actual residual. For a curve that does NOT intersect the surface, Newton converges to the *minimum-distance point* (a stationary point of ||C-S||²), which is not a root but has `last_step ≈ 0`. The code recorded it as an intersection.

Minimal reproducer (now a test in `tests/test_bez_csx4.py`):

```python
c1 = np.array([[0., 0., 5.], [0., 10., 5.]])         # line x=0, z=5
s2 = np.array([[[0.,0.,0.],[0.,10.,0.]],
               [[10.,0.,10.],[10.,10.,10.]]])        # bilinear, z = x
# Line has x=0 z=5 everywhere; surface needs x=z, so at z=5 needs x=5.
# They never meet.
bez_csx(c1, s2, rational=False)
# BEFORE: returned 1 isolated point with |C-S| = 3.54
# AFTER:  returns 0 isolated points
```

### 7.2 The fix

In `_bez_csx4.py:590-622`, renamed and separated the logic:

```python
step_norm = abs(last_step[0]) + abs(last_step[1]) + abs(last_step[2])
residual_ok = float(np.linalg.norm(G)) < atol
newton_stalled = (
    abs(last_step[0]) <= ptol_t
    and abs(last_step[1]) <= ptol_u
    and abs(last_step[2]) <= ptol_v
)

if newton_stalled:
    if residual_ok and t0 - ptol_t <= t_sol <= t1 + ptol_t:
        # Newton stopped AND at an actual root AND within this cell
        isolated.append(...)
        sub_cells = _cutout_3d(...)
        stack.extend(sub_cells)
        continue
    else:
        # Newton stopped but residual too large (stationary point, not root)
        # OR root is outside this cell. Either way, prune.
        continue
```

### 7.3 Impact on SSX

This was propagating through `_find_ssx_boundary_zeros` — each of the 8 boundary CSX calls was returning spurious "crossings" at the minimum-distance point of the isocurve vs the opposite surface. For the planes test case this produced 4 false crossings (instead of the 2 real boundary-touch crossings), making SSX think it had a complex topology and invoking domain decomposition heuristics that produced 3 branches.

**After the CSX fix: planes → 1 branch immediately, case 5 → 2× faster.**

### 7.4 Regression tests added

`tests/test_bez_csx4.py` — 5 new tests covering:

- `test_line_parallel_to_plane_no_intersection` — the exact user reproducer
- `test_line_crossing_plane_one_intersection` — positive control
- `test_line_lying_on_plane_detected_as_overlap` — overlap detection
- `test_line_near_plane_not_intersecting_no_false_positive` — line close but not touching
- `test_degree_one_line_no_false_positive_variants` — multiple orientations

All 12 tests in `test_bez_csx4.py` pass.

---

## 8. Refactoring priorities for next session

The user's directive: **remove the heuristic cleanup code that was hiding the bez_csx bug**. Now that CSX is correct, some of the SSX dedup / merge / filter logic is redundant, and some of it actively hurts accuracy (we've observed legitimate distinct branches being merged because their endpoints were geometrically close).

### 8.1 Code to audit for removal or simplification

In `_bez_ssx5.py`:

- **`_dedup_crossings`** (line 228) — checks stuv OR xyz proximity. The xyz check is dangerous because topologically distinct points can be geometrically close. The user's rule: only points with identical (s,t,u,v) are guaranteed duplicates. REMOVE the xyz check, keep stuv dedup only.

- **`_dedup_overlaps`** (line 252) — checks stuv endpoints in both orders. Probably still needed for corner-dedup when two S1 boundary faces meet. EVALUATE.

- **`_filter_corner_touches`** (line 1369) — tries to detect "crossings at cell corners that don't enter the cell". This was added to handle sub-cells where a crossing was at a corner and the marcher immediately exited. After CSX fix and proper partition handling, some of this should be unnecessary. EVALUATE carefully.

- **`_merge_adjacent_branches`** (line 1774) — concatenates branches whose endpoints match. If the topology handling is correct (marcher traces within cells, partition crossings are consumed), this should be obsolete. However, it may still be needed for joining pieces across internal partitions. EVALUATE.

- **`_dedup_branches`** (line 1827) — removes branches with same endpoints. PURE heuristic, added to cover up double-tracing. Should be removable.

- **`_is_on_both_boundaries`** (line 215) — helper for corner detection. Check if still used.

- **`_is_cell_corner`** (line 1338), `_tangent_enters_cell` (line 1348) — corner-touch infrastructure.

- **Overlap sub-segment containment check** in `_overlaps_to_branches` — does "remove shorter overlap whose endpoints lie on a longer one". Heuristic geometry. EVALUATE.

### 8.2 Method

1. For each candidate function, run the 5 working test cases (planes, transversal, tangential, overlaps, case 5) with it disabled.
2. If all 5 still produce correct output → remove the function.
3. If some case regresses → investigate *why* the function was needed and whether the underlying cause is the same CSX bug or something else.

### 8.3 Known issues to address AFTER the refactor (from `memory/project_ssx_v5_issues.md`)

1. **Corner dedup missing in `_find_ssx_boundary_zeros`**. Crossings at corners (two boundary faces meeting) generate two separate crossings with identical stuv. User's rule: identical stuv → remove as duplicate. Currently `_dedup_crossings` does this but also does xyz-based dedup which should be removed.

2. **Check order wrong in `_check_loop_free`**. Gauss map separability is checked BEFORE TΨᵢ monotonicity, even though Gauss map is more expensive. Case 5 has 62 Gauss map calls — most sub-cells would have been filtered by the cheaper monotonicity check first. SWAP the order.

3. **Case 6 loop completely missing**. Subdivision skips cells with 0 external crossings. A cell containing an entire interior loop has no crossings on its external box boundary, so domain decomposition never processes it. Need to ensure such cells ARE visited when loop-absence is not proven, and internal CSX on potential cut isolines can discover loop-crossing points.

4. **`_trace_all_branches` uses proximity matching**. When the marcher stops at a boundary, we find the "nearest" unvisited crossing. Should be exact face+parameter matching: the marcher exits at face (axis, side) with specific (s,t,u,v), and the matching crossing is the one on that same face with matching parameters. No proximity needed.

5. **Topological matching layer missing.** The plan called for `PartitionCurve` objects that know their left/right adjacent cells and the crossings on each side. With this structure, merging is a simple topological operation (match parameters on shared partition curves). Without it, we have redundant CSX calls and fragile branch merging.

---

## 9. File & function reference

### 9.1 Files

| Path | Purpose | Status |
|------|---------|--------|
| `mmcore/numeric/intersection/ssx/_bez_ssx5.py` | Main implementation | ACTIVE |
| `mmcore/numeric/intersection/ssx/_ssx4.py` | Old SSX — import `GaussMapBern`, `separate_gauss_maps`, `SSXBranch`, `SSXPoint`, `split_tensor_bezier_axis` | LEGACY (reused) |
| `mmcore/numeric/intersection/_deflate.py` | Deflation primitives. Use `minors_Tpsi_from_control_nets` directly. AVOID `analyse_deflated_system`. | PARTIAL REUSE |
| `mmcore/numeric/intersection/csx/_bez_csx4.py` | Curve-surface intersection. RECENTLY FIXED. | STABLE |
| `mmcore/numeric/intersection/csx/_ncsx4.py` | NURBS adapter for CSX | STABLE |
| `mmcore/numeric/intersection/ccx/_bez_ccx4.py` | Curve-curve intersection | STABLE |
| `mmcore/numeric/intersection/_bezier_common.py` | Newton solvers, surface eval | STABLE |
| `mmcore/numeric/intersection/_sq_dist_classify.py` | Min-of-net, Lipschitz, classification | STABLE |
| `mmcore/numeric/bern_sq_dist.py` | Sq-dist Bernstein net construction | STABLE |
| `mmcore/numeric/bern.py` | Bernstein operations (de Casteljau split, partial derivatives, etc.) | STABLE |
| `tests/test_bez_csx4.py` | CSX test suite (expanded with 5 new tests) | CURRENT |
| `examples/ssx/bez_ssx5_case5.py` | Two open branches on wavy surfaces | WORKING |
| `examples/ssx/bez_ssx5_case6.py` | Loop + open branch (loop currently MISSING) | BUG |

### 9.2 Key functions in `_bez_ssx5.py`

Data structures:
- `BoundaryCrossing(stuv, xyz, face)` — crossing on a face of [0,1]⁴
- `BoundaryOverlap(stuv_start, stuv_end, face)` — overlap curve on a face
- `SubdomainCell` — unused, remove
- `_Cell(g1, g2, crossings, box, depth)` — subdivision stack element

Core algorithm:
- `bez_ssx(S1, S2, atol, rational)` — main entry (line 1595)
- `_prune_ssx_cell` — Level 1 pruning
- `_find_ssx_boundary_zeros` — Level 2, 8 CSX calls
- `_check_loop_free(g1, g2, T1..T4)` — Level 3 classification
- `_trace_all_branches` — Level 4a, traces in a loop-free cell
- `_choose_cut`, `_extract_isoline`, `_isoline_csx_to_global` — Level 4b, domain decomposition
- `_deflate_tangent_cell`, `_march_phi_curve` — Level 4c, Φ-tracer

Marcher:
- `_ssx_tangent_4d` — SVD-based tangent computation with direction hint
- `_ssx_correct` — pseudoinverse Newton corrector
- `_march_intersection_curve` — march with known end
- `_march_to_boundary` — march until cell boundary hit

Coordinate conversion:
- `_local_to_global`, `_global_to_local`

To be removed/refactored:
- `_dedup_crossings` (xyz check harmful)
- `_merge_adjacent_branches` (topology heuristic)
- `_dedup_branches` (pure heuristic)
- `_filter_corner_touches` (maybe still needed for edge cases)
- Overlap sub-segment containment in `_overlaps_to_branches`

### 9.3 Key functions in imported modules

From `_ssx4.py`:
- `GaussMapBern.from_surf(H, rational)` — build Gauss map
- `g.map_dirs()` — flat unit normal directions
- `g.split_u(t)`, `g.split_v(t)` — split surface + Gauss map simultaneously
- `g.surface` — the underlying Bezier surface (in local [0,1]²)
- `separate_gauss_maps(dirs1, dirs2)` → `(p1, p2)` — hemisphere witness
- `SSXBranch(curve=(stuv_path, xyz_path), overlap=False, closed=False)`
- `SSXPoint(stuv, xyz)`

From `_deflate.py`:
- `minors_Tpsi_from_control_nets(P1_cartesian, P2_cartesian)` → `(T1, T2, T3, T4)` as nested Python lists (convert to numpy arrays with `np.asarray(T, dtype=np.float64)`)

From `_bez_csx4.py`:
- `bez_csx(C, S, atol, rational)` → `{'isolated': [...], 'overlaps': [...]}`
- Recently fixed for the Newton-stall false positive.

From `_bezier_common.py`:
- `eval_surface(S, u, v, rational)`, `eval_surface_d1(S, u, v, rational)`
- `eval_curve(C, t, rational)`, `eval_curve_d1(C, t, rational)`
- `extract_weights(ctrl, rational)`
- `newton_csx(C, S, t0, u0, v0, rational)` — the 3D Newton for CSX

From `bern_sq_dist.py`:
- `surface_surface_distance_squared_net_homog(S1_h, S2_h, rational)` — the 4D sq-dist net

From `_sq_dist_classify.py`:
- `_check_min_of_net(F, atol, w_scale)`
- `_check_lipschitz(F, atol, w_scale)`
- `_weight_max_product(w1_flat, w2_flat)`

---

## 10. Test cases (can be pasted directly)

### 10.1 Planes (should return 1 branch)

```python
s1 = np.array([[[0., 0., 5.], [0., 10., 5.]],
               [[10., 0., 5.], [10., 10., 5.]]])
s2 = np.array([[[0., 0., 0.], [0., 10., 0.]],
               [[10., 0., 10.], [10., 10., 10.]]])
```

### 10.2 Transversal bilinear (should return 1 branch)

```python
s1 = np.array([[[0., 0., 0.], [0., 10., 0.]],
               [[10., 0., 0.], [10., 10., 10.]]])
s2 = np.array([[[0., 0., 3.], [0., 10., 3.]],
               [[10., 0., 3.], [10., 10., 3.]]])
```

### 10.3 Tangential (should return 1 branch via Φ-tracer)

```python
s1 = np.array([[[0.,0.,10.],[5.,5.,10.],[5.,10.,10.],[0.,15.,10.]],
               [[5.,0.,0.],[10.,5.,0.],[10.,10.,0.],[5.,15.,0.]],
               [[10.,0.,10.],[15.,5.,10.],[15.,10.,10.],[10.,15.,10.]]])
s2 = np.array([[[0.,0.,0.],[5.,5.,0.],[5.,10.,0.],[0.,15.,0.]],
               [[5.,0.,10.],[10.,5.,10.],[10.,10.,10.],[5.,15.,10.]],
               [[10.,0.,0.],[15.,5.,0.],[15.,10.,0.],[10.,15.,0.]]])
```

### 10.4 Overlaps (should return 2 overlap branches)

```python
s1 = np.array([[[-128.25, -129.86, 67.44], [-128.25, 129.86, 0.]],
               [[128.25, -46.98, 0.], [128.25, 129.86, 0.]]])
s2 = np.array([[[-128.25, -129.86, 0.], [-128.25, 129.86, 0.]],
               [[128.25, -129.86, 0.], [128.25, 129.86, 0.]]])
```

### 10.5 Case 5 — two open branches (should return 2 branches)

```python
s1 = np.array([[[-19., -51., 3.],[-19., -46., 8.],[-19., -41., 3.],[-19., -36., 8.]],
               [[-14., -51., 8.],[-14., -46., 3.],[-14., -41., 8.],[-14., -36., 3.]],
               [[-9., -51., 3.],[-9., -46., 8.],[-9., -41., 3.],[-9., -36., 8.]],
               [[-4., -51., 8.],[-4., -46., 3.],[-4., -41., 8.],[-4., -36., 3.]]])
s2 = np.array([[[-20.35213885, -55.05885716, 0.],[-19., -46., 0.],[-18.09739608, -42.8559574, 0.],[-19.39149131, -33.02632827, 0.]],
               [[-15.52250972, -56.35295239, 7.],[-14., -46., 7.],[-14., -41., 13.],[-14.56186218, -31.73223305, 7.]],
               [[-10.69288059, -57.64704761, 7.],[-9., -46., 13.],[-9., -41., 13.],[-9.73223305, -30.43813782, 7.]],
               [[-5.86325146, -58.94114284, 0.],[-4., -46., 6.],[-4., -41., 0.],[-4.90260392, -29.1440426, 0.]]])
```

### 10.6 Case 6 — loop + open branch (open found, loop MISSING)

```python
s1 = np.array([[[  7.4968198 , -34.44808135,   6.627417  ],
                [  3.96128589, -40.81204238,  -8.372583  ],
                [ -3.8168887 , -48.59021697,   6.627417  ]],
               [[ 14.73328128, -36.02768858,   6.627417  ],
                [  6.95510669, -43.80586318,  -9.372583  ],
                [  3.41957278, -47.34139708,   6.627417  ]],
               [[ 17.72710208, -39.02150938,   6.627417  ],
                [  9.94892749, -46.79968397,  -7.372583  ],
                [  6.41339358, -50.33521788,   6.627417  ]],
               [[ 17.18538897, -45.55086408,   6.627417  ],
                [ 15.94274828, -49.79350477,  -2.372583  ],
                [ 13.16457369, -49.57167936,  -7.372583  ]]])
s2 = np.array([[[  0., -51.,   6.29241333], [  0., -46.,   6.09945352],
                [ -0., -46.,  -4.68504969], [  0., -36.,  -4.70758667]],
               [[  5., -51.,   6.09945352], [  5., -46.,  -4.68504969],
                [  5., -41.,   6.09945352], [ 10., -36.,  -4.68504969]],
               [[  6., -51.,  -4.68504969], [ 10., -46.,   6.09945352],
                [ 10., -41.,  -4.68504969], [ 10., -36.,   6.09945352]],
               [[ 15., -51.,  -4.70758667], [ 15., -42.,  -4.68504969],
                [ 15., -41.,   6.09945352], [ 15., -36.,   6.29241333]]])
```

---

## 11. Running the tests

```bash
# CSX unit tests (all must pass)
.venv/bin/python -m pytest tests/test_bez_csx4.py -v

# Run the 5 SSX test cases directly
.venv/bin/python -c "
import numpy as np, time
from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx
import warnings; warnings.filterwarnings('ignore')

# paste test case data from section 10
# r = bez_ssx(s1, s2, atol=1e-3, rational=False)
# print(len(r['branches']), len(r['points']))
"

# SSX example scripts
.venv/bin/python examples/ssx/bez_ssx5_case5.py
.venv/bin/python examples/ssx/bez_ssx5_case6.py

# Full test suite (excluding known-failing unrelated tests)
.venv/bin/python -m pytest tests/ -q --timeout=60 \
  --ignore=tests/test_nurbs_compose.py \
  --ignore=tests/test_boundary_intersection.py \
  --ignore=tests/test_boundary_intersection_robust.py \
  --ignore=tests/test_nurbs_ssx.py \
  --ignore=tests/test_curve_bool.py
```

---

## 12. Numerical details

### 12.1 The marcher

All three variants (`_march_intersection_curve`, `_march_to_boundary`, `_march_phi_curve`) share the same predictor-corrector structure with these constants:

```python
initial_step    = 0.05     # in 4D parameter space
min_step        = 1e-6     # floor; marcher gives up if it can't make progress at this scale
max_step        = 0.25     # ceiling; prevents overshoot on nearly-linear curves
angle_threshold = 0.1      # radians (~5.7°) — target angle between consecutive tangents
max_points      = 2000     # safety cap
```

Step adaptation rule:

```python
if angle > 1e-10:
    step = step * min(2.0, max(0.25, angle_threshold / angle))
else:
    step = min(max_step, step * 1.5)  # nearly-linear, grow aggressively
step = max(min_step, min(max_step, step))
```

Interpretation: on a highly curved region (large `angle`), step shrinks; on nearly linear regions, it grows up to `max_step`. Factor is clamped to [0.25, 2.0] per iteration to prevent oscillation.

### 12.2 The corrector (pseudoinverse Newton)

`_ssx_correct` implements damped pseudoinverse Newton on the 3×4 Jacobian:

```python
J = [S1_s, S1_t, -S2_u, -S2_v]           # (3, 4)
A = J.T @ J + 1e-12 * eye(4)              # Levenberg-Marquardt damping
delta = solve(A, -J.T @ G)                 # pseudoinverse step
```

- `max_iter = 5` (the predictor is close; many iterations mean predictor was wrong)
- `tol = 1e-14` (residual² tolerance — effectively machine epsilon)
- `lm_damp = 1e-12` (regularization, keeps A invertible in rank-deficient cases)

Clamping to [0, 1] after each step prevents escape from the parameter domain.

### 12.3 The tangent computation

`_ssx_tangent_4d` uses SVD of the 3×4 Jacobian:

```python
U, sigma, Vt = np.linalg.svd(J, full_matrices=True)
tol_sv = max(J.shape) * sigma[0] * 1e-10 if sigma[0] > 0 else 1e-10
rank = int(np.sum(sigma > tol_sv))
null_dim = 4 - rank
```

For full-rank J (rank=3), `null_dim=1` and the tangent is `Vt[-1]` (last right singular vector).

For rank-deficient J (rank<3), the null space is >1D. We project the `direction_hint` onto the null space and normalize. If no hint is available, we use `Vt[-1]` as a fallback.

**Critical bug fixed earlier**: the original code tried to count `sigma < tol` to determine null_dim, which doesn't work for 3×4 matrices (they always have 3 singular values regardless of rank). The correct formula is `null_dim = n_cols - rank`.

### 12.4 Tolerances

| Symbol | Value | Meaning |
|--------|-------|---------|
| `atol` | user-supplied, typically `1e-3` | geometric tolerance: |C(t) − S(u,v)| below this means "same point" |
| `ptol_t`, `ptol_u`, `ptol_v` | computed from curve/surface geometry | parametric tolerance: step size below this means "stalled" |
| machine epsilon | `1e-15` to `1e-14` | used in corrector residual target |
| `1e-10` | | null-space rank detection floor |
| `1e-12` | | LM damping, null vector norm floor |
| `1e-8` | | boundary detection (`_on_boundary`) |

The rule: `atol` is the only user-facing tolerance. Everything else is an implementation detail and should be tighter.

### 12.5 Loop-free check

`_check_loop_free(g1, g2, T1, T2, T3, T4)` currently checks Gauss map separability BEFORE TΨᵢ monotonicity. This is wrong — TΨᵢ sign check is O(n) in coefficients (cheap), Gauss map witness is O(n²) in control points (expensive). Should be reversed. See known issue #2.

Both checks are *sufficient* (not necessary) for loop absence. Either one proving absence is enough. They complement:

- **TΨᵢ monotonicity** catches cases where the curve is monotonic in some parameter axis. Common for non-tangent intersections.
- **Gauss map separability** catches cases where the surfaces' normal cones are "facing away" from each other. Common for transversal intersections with arbitrary curve orientation.

Case studies:
- Bilinear vs bilinear transversal: both checks pass.
- Bilinear vs tangent surface: both fail (correctly — there's a tangent intersection curve).
- Wavy vs flat: TΨᵢ may fail (curve undulates), Gauss maps separate (normals clearly distinct).

### 12.6 Tangency check

`_check_tangency(T1, T2, T3, T4, P1_cart, P2_cart, box)`:

1. Build `DeflatedSystem` over the box with the 4 interval-valued TΨᵢ nets.
2. Quick interval range check: if any `TΨᵢ(B)` excludes 0, return False.
3. Gauss-Newton witness on the full 7-equation Δ system (max 8 iterations, tol 1e-8). If converges → True.
4. If residual > 1.0 → clearly not tangent → False.
5. Otherwise → None (undetermined).

We removed the expensive Krawczyk isolation that used to follow — it added 0.5s per check. The simplified version rarely returns False conclusively but quickly returns True for real tangent cases and None for regular cases.

---

## 13. Commit history

Relevant commits during this project (most recent first):

| Commit | Description |
|--------|-------------|
| `6d25fcc` | fix(bez_csx): distinguish Newton stall from intersection, reject stationary points |
| `e75e61d` | fix: remove partition touch-points after marching |
| `748e271` | fix: proper zero-length march handling at partition boundaries |
| `427bb21` | fix: corner-touch filter + branch merge/dedup for bez_ssx |
| `df60f64` | refactor: global-coords approach for bez_ssx domain decomposition |
| `36608c4` | wip: iterative domain decomposition for bez_ssx v5 |
| `d76ecea` | feat: bez_ssx v5 — surface-surface intersection with Φ-tracer for tangential cases |

To revisit a specific era:

```bash
git show 6d25fcc --stat           # the CSX fix
git show 6d25fcc mmcore/numeric/intersection/csx/_bez_csx4.py
git log --oneline d76ecea..HEAD   # the whole SSX v5 history
```

---

## 14. Important conventions and gotchas

- The user strongly prefers **iterative (stack-based) implementations over recursion** for algorithms with complex topology.
- The user strongly prefers **exact parametric matching over geometric proximity heuristics** for topological operations.
- The user prefers **NURBSCurveTuple / NURBSSurfaceTuple over Cython NURBS objects** for debugging and development.
- User reads Russian and English; please write user-facing text in English but technical terms can be explained.
- `mmcore` does NOT guarantee backwards compatibility; API changes are fine.
- Python venv: `/Users/sthv/PycharmProjects/mmcore/.venv/bin/python` (Python 3.14)
- Git user: `sth-v`, main branch is `tiny`, current dev is `0.53.0`.

---

## 15. First action for next session

1. Read this entire document.
2. Read the current `_bez_ssx5.py` (whole file, it's 1865 lines).
3. Read the latest commit `6d25fcc` (the CSX fix).
4. Run all 5 test cases to see the current baseline.
5. Then START the refactoring: for each function in section 8.1, test with it disabled and decide whether to remove.

Good luck. The hardest conceptual work is done. What remains is careful refactoring and then two more structural improvements (partition-curve topology for merging, and making the loop-absence-failing cell path visit interior-loop cells).

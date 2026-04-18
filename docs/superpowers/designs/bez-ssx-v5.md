# bez_ssx v5 — Living Algorithm Design

**Status:** current design of record. Implementation (`mmcore/numeric/intersection/ssx/_bez_ssx5.py`) must agree with this file at the end of every iteration cycle. The companion measurements log is [`bez-ssx-v5-measurements.md`](./bez-ssx-v5-measurements.md).

This document describes a Bezier surface–surface intersection (SSX) algorithm that produces a complete topological description of the intersection set of two (possibly rational) Bezier surfaces:

- a set of non-self-intersecting 1D **branches** (open or closed) traced with user-specified geometric tolerance,
- a set of **overlap** segments where the two surfaces coincide on an open region,
- a set of isolated **point** hits (tangential touches with 0D intersection).

The algorithm is iterative, stack-based, and driven by certificates — *algebraic proofs* of a specific topology inside each subdomain. Subdivision is a fallback used only to produce sub-cells on which a cheaper certificate holds.

## 1. Foundations

Three mathematical tools do almost all of the work. Each one gives a different kind of guarantee, and the algorithm's structure is the composition of those guarantees.

### 1.1 Sq-dist Bernstein net (pruning)

For homogeneous surfaces `S1(s,t)`, `S2(u,v)`, the function
`D(s,t,u,v) = ‖S1(s,t) − S2(u,v)‖²`
has a 4-variate Bernstein representation (numerator over `w_S1² · w_S2²`). Let `F` be that numerator tensor. Because the Bernstein convex hull bounds the value, the min/Lipschitz tests on `F` give a cheap certificate `F > atol² · w_scale²` ⇒ no intersection anywhere in the box. Used only for top-level pruning.

Provided by `bern_sq_dist.surface_surface_distance_squared_net_homog`; checks in `_sq_dist_classify`.

### 1.2 TΨᵢ minors (loop-absence via monotonicity) — Paper 2 Lemma 5

The 4D implicit intersection curve `Ψ(s,t,u,v) = S1(s,t) − S2(u,v) = 0` (3 equations in 4 unknowns) has a 4D tangent whose components are the 3×3 minors of the Jacobian:
`TΨᵢ = det(J_Ψ with column i removed)`.

Each `TΨᵢ` is itself a 4D Bernstein tensor, exactly computable from `S1`, `S2` by cross products and dot products of the partial derivative nets (Cheng et al. 2023; implementation: `_deflate.minors_Tpsi_from_control_nets`).

**Certificate (Lemma 5):** if the Bernstein coefficient array of some `TΨᵢ` is entirely `≥ 0` or entirely `≤ 0` over a box `B`, then on any intersection arc contained in `B` the parameter `sᵢ` is monotonic along arc length — so the arc contains no interior loop, no self-intersection, and no turning point in `sᵢ`. Its topology in `B` is fully determined by its boundary crossings.

**Propagation:** `TΨᵢ` is a Bernstein tensor of known degree; it can be de Casteljau-split along any of its four axes. When the SSX stack splits a cell along axis `a`, it must split the cell's `T1..T4` along axis `a` in the same way (and by the same fraction). **Recomputing TΨᵢ from split control nets is incorrect at depth > 0** because sub-cells re-parameterize the surfaces; the coefficients must be split, not recomputed.

### 1.3 Gauss map separability (loop-absence via normal cones)

Normals of the two surfaces are represented as a 4D Bernstein tensor (degree doubled). A hemisphere witness, if found, separates the two normal cones into opposite hemispheres of `S²`. If both cones are strictly on different sides of a hyperplane through the origin, the curve `Ψ = 0` cannot form a loop inside `B` (because the cross product `n1 × n2` — the tangent to the intersection curve — cannot vanish).

Provided by `_ssx4.GaussMapBern`, `_ssx4.separate_gauss_maps`.

Gauss map splitting is already implemented in `GaussMapBern.split_u/split_v`; it splits surface and Gauss map tensor simultaneously. At cell-split time, both `GaussMapBern`s and the `T1..T4` tensors must be split together.

### 1.4 The deflated / regulated system (tangential intersections) — Paper 2 Lemma 2

At a point where `Ψ = 0` and `TΨ = 0` simultaneously (a *tangent point*, `C₂`), the Ψ Jacobian has rank ≤ 2 — marching `Ψ = 0` fails because the 4D tangent is ambiguous.

The **regulated curve** `Φ = { Φ₁, Φ₂ ∈ {Ψ₁,Ψ₂,Ψ₃}, Φ₃ ∈ {TΨ₁,TΨ₂,TΨ₃,TΨ₄} }` — 3 equations in 4 unknowns — has rank-3 Jacobian at a tangent point (TΨᵢ contributes a direction transverse to the null-space of `J_Ψ`). Marching `Φ = 0` works there, and every marched point is filtered by the excluded full-Ψ component so only points that *actually* lie on the intersection remain.

Provided by `_march_phi_curve`, `_deflate_tangent_cell`, `_choose_phi_equations`. The equation subset is chosen once per tangent cell for best conditioning.

## 2. Pipeline

Four layers, executed once per top-level call:

```
L0 pruning ─── AABB + sq-dist net min/Lipschitz
L1 boundary ── 8 CSX problems (one per face of [0,1]⁴)
               → boundary BoundaryPoints + BoundaryOverlaps
L2 tracing ─── iterative stack of _Cells, each cell terminates by a certificate:
                (a) loop-free (TΨᵢ monotone OR Gauss separable) → trace Ψ
                (b) confirmed tangent (Krawczyk on TΨ=0)        → trace Φ, filter by Ψ
                (c) neither → choose a cut through a crossing's
                    param, split surface + Gauss + T tensors,
                    add CSX crossings on the cut isoline, push sub-cells
L3 assembly ── per-partition 1D matching joins branches across shared isolines
```

**No subdivision is performed for its own sake.** A cell stops subdividing the moment it acquires any certificate. Subdivision exists only to turn a cell that cannot yet certify into sub-cells that can.

## 3. Coordinates and reparameterization

- **Persistent storage is global.** Boundary points, branch endpoints, partition curves, and the `box` of every cell are in *global* `[0,1]⁴` coordinates. A `_Cell.box` is a 4-tuple `((s_lo,s_hi),(t_lo,t_hi),(u_lo,u_hi),(v_lo,v_hi))`. Registrations on a partition record the free-axis coordinate in global coords so cross-cell matching (§5) works in one frame.
- **Transient computation is local.** Surfaces, Gauss maps, and `TΨᵢ` tensors inside a cell are the De Casteljau sub-tensors living in *local* `[0,1]²` (surfaces, Gauss maps) or `[0,1]⁴` (TΨᵢ). Boundary-crossing classification (§4) is performed in local coordinates because the boundary test collapses to `local_stuv[i] ∈ {0, 1}`; the cell box is consulted once when the classification result is written back as a global `param` on the registration. The marcher similarly converts its seed and its step to local before invoking the evaluators.
- **Conversion:** `global[i] = box[i].lo + local[i] * (box[i].hi - box[i].lo)`; `local[i] = (global[i] - box[i].lo) / (box[i].hi - box[i].lo)`.

## 4. Boundary-crossing classification

Classification is always performed **in the owning cell's local `[0,1]⁴` parameter space**. Every sub-cell, whatever its global box, has local boundaries at exactly `0` and `1` on each axis — so "does this axis lie on a cell boundary?" is the trivial test `local_stuv[i] ∈ {0, 1}` with no box lookup or tolerance-against-arbitrary-value arithmetic. Global parameters are recorded afterwards, on the registration, so that cross-cell matching (§5) has a single coordinate frame.

**Raw 4D tangent `T`.** Evaluate the SVD null vector of the 3×4 local Jacobian `J_Ψ(P_local)` *without* any cell-interior clamp. Clamping is a marching-time device that prevents excursion outside `[0,1]⁴`; applied here it zeroes the very component we need for classification and the scheme degenerates (`0` has no sign). The absolute sign of `T` is arbitrary (null vectors are defined up to sign); the relative signs of its components are invariant.

**On-boundary axes.** Axis `i` is *on-boundary for a cell* if the cell-local coordinate `local_stuv[i]` equals `0` or `1` within numerical tolerance (default `1e-8`). A point can be on 1, 2, 3, or 4 boundaries simultaneously — the 4-boundary case corresponds to a corner of the 4D box where four sub-patches meet. Axes with a strictly interior local value (`0 < local_stuv[i] < 1`) do not participate in classification for this cell and contribute no registration, because no isoline of the cell is held fixed at that local value.

**Per-axis classification — the `(local_param, sign)` table.** For each on-boundary axis `i`:

| `(local_stuv[i], sign(T[i]))` | meaning                                                   | direction for this cell |
|-------------------------------|-----------------------------------------------------------|-------------------------|
| `(0, +1)`                     | at local `0`, tangent points inward (toward `1`)          | **entry**               |
| `(0, −1)`                     | at local `0`, tangent points outward (below the cell)     | **exit**                |
| `(1, −1)`                     | at local `1`, tangent points inward (toward `0`)          | **entry**               |
| `(1, +1)`                     | at local `1`, tangent points outward (above the cell)     | **exit**                |

A point may classify as entry on some on-boundary axes and exit on others — a **through-touch**: the curve enters the cell via one face and immediately leaves via another, zero-length interior path. This is not an error; it is the expected case at corners shared by sub-cells after decomposition, and it is the source of the "odd number of crossings on a face" diagnostic the heuristic implementation emits.

**Worked example** (top-level cell, so local ≡ global). Point `stuv = (0, 0, 1, 0.2)` with raw tangent `(−0.154, +0.154, +0.309, +0.926)`:

1. Reduce the tangent to signs: `[−1, +1, +1, +1]`.
2. On-boundary axes: `s`, `t`, `u` (`v = 0.2` is strictly interior, not on any boundary).
3. Classification:
   - `s`: `(0, −1)` → this cell's `s = 0` face is an **exit**.
   - `t`: `(0, +1)` → this cell's `t = 0` face is an **entry**.
   - `u`: `(1, +1)` → this cell's `u = 1` face is an **exit**.

The single point contributes one entry and two exit registrations on three *different* isolines (`t = 0`, `s = 0`, `u = 1`). Matching must therefore take place at the isoline level, not at the point level.

**Sub-cell invariance.** A crossing shared by two sub-cells across an internal partition sits at local coordinate `1` in one cell and local coordinate `0` in its neighbour (same physical point, opposite local views). The `(local_param, sign)` table's first entry flips between `0` and `1`, `sign(T[i])` is identical (the local tangent only rescales, it does not change signs under positive affine rescaling), so the table flips `entry ↔ exit`. The crossing is registered on the shared partition with `direction = "in"` from one side and `direction = "out"` from the other — which is exactly what Invariant A (§5) requires.

**Global coordinates come in only at registration time.** Given a classified on-boundary axis `i` with direction `d`, the `PartitionCurve` holding coordinate `i` fixed at the cell's global `box[i].lo` (if `local_stuv[i] == 0`) or `box[i].hi` (if `local_stuv[i] == 1`) receives a new `IsolineRegistration` whose `param` is the point's global coordinate on the partition's free axis. That is the single local-to-global mapping the classification stage needs.

## 5. Partition-curve topology

An **isoline** is a 1D curve embedded in `[0,1]⁴`. It is completely specified by one fixed coordinate (which of `s,t,u,v` is held constant, and at what value) and the global interval of the free coordinate. Every isoline either is one of the eight outer faces of the top-level `[0,1]⁴` (fixed axis = 0 or 1, global extent = `[0,1]`), or an internal partition produced by a subdivision (fixed axis = some global value, global extent = the free-coord range of the parent cell at split time).

An **isoline registration** is precisely the tuple the user's description calls for:

```
IsolineRegistration = (isoline, isoline_global_interval, param, direction)
```

with `param` the scalar coordinate of the registered point along the isoline's free axis (in global coords) and `direction ∈ {"in", "out"}` produced by §4's classification for the *specific cell* that owns this registration.

```python
@dataclass
class IsolineRegistration:
    partition:   PartitionCurve        # which isoline
    param:       float                 # global param along partition.free_axis
    direction:   Literal["in", "out"]  # from §4 classification, in the owner cell's frame
    owner:       _Cell                 # the cell this registration describes
    point:       BoundaryPoint         # back-reference to the 4D crossing

@dataclass
class PartitionCurve:
    axis:            int                        # 0..3, the fixed coordinate
    value:           float                      # global value of the fixed coordinate
    free_axis:       int                        # 0..3, the varying coordinate
    global_extent:   tuple[float, float]        # [lo, hi] of the free axis in global coords
    adjacents:       list[_Cell]                # 1 for an outer face, 2 for an internal partition
    registrations:   list[IsolineRegistration]

@dataclass
class BoundaryPoint:
    stuv:           NDArray[np.float64]   # (4,), global
    xyz:            NDArray[np.float64]   # (3,)
    tangent_raw:    NDArray[np.float64]   # (4,), unclamped null vector of J_Ψ
    registrations:  list[IsolineRegistration]   # one per on-boundary axis per owning cell
```

The §4 classification feeds §5 mechanically: each on-boundary axis `i` of each owning cell yields one `IsolineRegistration` on the `PartitionCurve` that holds coordinate `i` fixed at that cell's `B[i].lo` or `B[i].hi`. A point that is through-touch on two axes produces two registrations (on two different partitions) from the *same cell* — one `in`, one `out` — and by construction these live on *different* isolines, never on the same one. A point on an internal partition that is shared by two cells produces registrations in *both* cells' views of that partition, with flipped directions.

**Invariant A — registration completeness.** For every `BoundaryPoint P` owned by cell `C`: `P` has exactly one `IsolineRegistration` on every `PartitionCurve` of `C` whose fixed axis is on-boundary at `P.stuv`. For an internal partition shared by cells `L` and `R`, the same physical crossing appears in both `L.partitions[...].registrations` and `R.partitions[...].registrations` with the *same* `param` and *opposite* `direction`. Mismatches are a classification bug, raised, not silently reconciled.

**Invariant B — exact param merging.** Joining partial branches across an internal partition is a 1:1 sort-merge of `(param, "out")` entries from one side against `(param, "in")` entries from the other. The match key is `param` alone; the residual between the two `param` values is expected to be numerical noise from the CSX discoveries on each side, always below `atol` under correct classification. No xyz proximity, no stuv-distance threshold.

**Invariant C — identical-stuv dedupe only.** Two `BoundaryPoint`s with identical `stuv` within numerical tolerance represent the same crossing and must be unified into one `BoundaryPoint` whose registrations are the union of the two lists. Two `BoundaryPoint`s with identical `xyz` but distinct `stuv` are *legitimate* (a self-intersection, a fold, or two branches crossing in 3-space) and must NOT be deduped — doing so silently erases topology.

## 6. Cell lifecycle

```python
@dataclass
class _Cell:
    box: Box4
    S1, S2: NDArray   # local homogeneous control nets
    g1, g2: GaussMapBern
    T1..T4: NDArray   # local Bernstein tensors
    partitions: list[PartitionCurve]   # 4 outer faces + N internal from splits
    depth: int
```

Each iteration of the main loop pops a cell and asks three questions in order.

1. **Cheap certificate — TΨᵢ monotonicity.** If any `Tᵢ` has all Bernstein coefficients of one sign, the cell is loop-free. Trace branches between the cell's boundary crossings (§ 7) and terminate the cell.
2. **Second cheap certificate — Gauss separability.** If `separate_gauss_maps(g1.map_dirs(), g2.map_dirs())` returns a witness pair, the cell is loop-free. Trace and terminate.
3. **Confirmed-tangency certificate.** A Krawczyk/Gauss-Newton run on the 4-equation system `TΨ = 0` over the cell. True → trace via `Φ` (deflation / regulated system). `False` (TΨ(B) does not contain 0) → impossible here because the two cheap certificates already failed; treat as None. `None` (undetermined) → fall through to subdivision.
4. **Subdivision.** Pick an axis + value equal to some existing crossing's coordinate along that axis (interior to the cell, not on the cell's boundary). Extract the isoline of the chosen surface at that parameter, run one CSX call against the opposite surface to discover new boundary crossings on the new partition. Split the chosen surface, its Gauss map, and the corresponding `T` tensors along that axis + value. Distribute existing crossings + new crossings to the two sub-cells; register them on the new internal `PartitionCurve`. Push sub-cells to the stack.

The order 1 → 2 → 3 → 4 is mandatory: the cheapest certificate must be tried first so the expensive certificates (and subdivision) are avoided when possible.

Safety cap: `max_depth` (default 12). Reaching it without a certificate is treated as a non-fatal warning; the crossings in that cell are emitted as isolated `SSXPoint`s.

## 7. Tracing inside a loop-free cell

Given a cell terminated by certificate (1) or (2), with its `BoundaryPoint` set:

```
while there exists an unvisited "in" registration in this cell:
    pick it; seed the marcher at its global stuv
    march the predictor–corrector until the 4D point
        reaches the cell boundary or another registration's stuv
    the stopping registration MUST exist in this cell's partitions
    (Invariant D); match it by (partition_id, param) exactly
    record the branch (start_point, end_point, path)
    mark both registrations consumed
```

**Invariant D — marcher stays in cell.** The marcher clamps to `[0,1]⁴` in local coords, then converts; it cannot exit the cell. When it stops at the boundary, the stopping `stuv` lies on exactly one `PartitionCurve` at a specific `param` — and by Invariant A there is a registered `BoundaryPoint` with an unconsumed `direction = "out"` at that `(partition, param)`. No nearest-neighbor search, no xyz proximity.

A through-touch (classified both in and out at the same corner) produces a single-marcher-call branch whose length may be zero; it consumes one `in` and one `out` registration on different partitions.

## 8. Tracing inside a tangent cell (Φ branch)

Given a cell terminated by certificate (3), we have at least one boundary crossing and a confirmed tangent point somewhere inside. We switch the marcher's system from `Ψ` to `Φ`:

- `_choose_phi_equations` picks the 2 Ψ rows + 1 TΨ row yielding the best-conditioned 3×4 `J_Φ` at the seed.
- `_march_phi_curve` uses the same predictor–corrector, but each marched point is accepted into the branch only if the excluded Ψ component is also zero within `atol` (i.e. the point lies on the *full* Ψ = 0 set, not just on the regulated curve).

The Φ-traced branch respects the same partition-curve registration protocol as Ψ-traced branches.

## 9. Output assembly

After every cell is processed:

- For each `PartitionCurve` with `len(adjacents) == 2` (internal partition): run the 1D param-match (Invariant B) on the two owners' registrations. Matched pairs concatenate the two partial branches into one.
- Zero-length branches (single point) become isolated `SSXPoint`s in the output.
- Overlap branches from L1 are appended directly (they are already full-length on their isoline).

The output is `{'branches': [...SSXBranch...], 'points': [...SSXPoint...]}`. There is no post-hoc xyz-based merging, no xyz-based deduping, no corner-touch filter. Such operations are forbidden by Invariant C and by the design principles below.

## 10. Design principles

These are non-negotiable. They shape the boundaries between "implementation detail" and "design property".

1. **No heuristics.** Every decision is algebraic or geometric with a specific certificate. "Probably a duplicate because xyz is close", "probably a touch because the tangent happens to point out" — banned.
2. **Non-monotonicity ≠ tangency.** All four `TΨᵢ` straddling 0 is a weaker condition than `TΨ = 0` having a simultaneous zero. The former is resolved by subdivision; the latter requires deflation.
3. **Certificates are the goal, subdivision is the fallback.** Not "subdivide until tiny". Subdivide only when no certificate currently applies; stop subdividing the moment one does.
4. **Cut through a crossing's parameter value.** Splits go through an existing crossing, never at a midpoint. This keeps each sub-cell's crossing set manageable (Krishnan & Manocha 1997).
5. **Iterative with a global topology structure beats recursion.** The stack + partition-curve registry make tracing, merging, and diagnostics observable.
6. **Identical stuv → unify. Identical xyz → keep distinct.**
7. **Trust the marcher's stopping point** — it is bounded to the cell by construction.
8. **Propagate certificates, don't recompute.** TΨᵢ and Gauss maps are split alongside surfaces when cells split. Recomputing from scratch at depth > 0 is wasteful and, for TΨᵢ, incorrect.

## 11. Paper references

- Krishnan & Manocha 1997 (`237748.237751.pdf`) — guided domain decomposition at crossing parameters; loop-free sub-cell termination.
- Cheng, Zhang, Xiao, Li 2023 IATA (`3592452-2.pdf`) — TΨᵢ monotonicity (Lemma 5), Φ regulated system for tangent curves (Lemma 2), Krawczyk certification for tangency.

## 12. Change history

- **2026-04-18 — Initial draft.** Codifies the target design discussed in the 2026-04-11 context doc (§§4.4–4.7). Implementation has the skeleton (pipeline + marcher + Φ-tracer) but deviates in four places: (1) `TΨᵢ` is computed once at top level and not propagated into sub-cells — sub-cell loop-absence relies only on Gauss separability; (2) no `PartitionCurve` / `BoundaryPoint` / `IsolineRegistration` structures — tracing uses proximity matching and post-hoc merging; (3) boundary crossings at corners are filtered heuristically (`_filter_corner_touches`) rather than classified by raw 4D tangent direction; (4) overlaps and crossings are de-duped by xyz proximity in places — Invariant C is violated. Goal of subsequent cycles: bring implementation to agreement with this design, measuring impact on tests + perf at each step.
- **2026-04-18 — §4/§5 refinement.** Rewrote the boundary-crossing classification as an explicit `(local_param, sign(T[i]))` table (user's `(param, sign)` presentation) with a worked through-touch example. Promoted the registration data structure to the 4-tuple `(isoline, isoline_global_interval, param, direction)` described in the context doc and made explicit that each on-boundary axis produces its own registration on a different partition. No functional change — same algorithm — but the presentation now matches the target wording closely enough that the §4 → §5 translation during implementation is mechanical.
- **2026-04-18 — interior-value axes explicit.** Added an explicit paragraph in §4 spelling out that an axis with a non-boundary value (e.g. `t = 0.2` in a cell spanning `t ∈ [0,1]`) produces no registration, because no isoline of the cell is fixed at that value. The rule is "coordinate equals the owning cell's box `lo` or `hi`", applied uniformly to outer faces and to partitions produced by subdivision. Again, no functional change, but it removes an ambiguity that would likely have surfaced as a classification bug during implementation.
- **2026-04-18 — classification in local coords.** §4 now states that classification runs in the owning cell's **local `[0,1]⁴`**, so the on-boundary test is the trivial `local_stuv[i] ∈ {0, 1}` without any `box.lo` / `box.hi` arithmetic. The global `param` is emitted only at registration time. §3 clarified accordingly: persistent storage stays global (single frame for cross-cell matching), transient classification and tangent computation are local.

### Iteration 1 — 2026-04-18 — TΨᵢ propagation to sub-cells

Design-alignment step. §1.2 and §6 already specified that TΨᵢ is de Casteljau-split alongside surfaces and Gauss maps at every cell split (never recomputed), and tried first in `_check_loop_free`. The implementation did not: sub-cells only ran Gauss separability. This iteration brings the code into agreement by adding `T1..T4` to `_Cell`, propagating them via a scalar-tensor de Casteljau wrapper, and passing them into `_check_loop_free` for every sub-cell.

Tests green, no regression. Case 5: 4 sub-cells now certified by TΨᵢ alone, ~1 % wall-time improvement (525 ms vs 532 ms). Outcome recorded in measurements; the small impact reflects that case 5's intersection curve is non-monotone in every parameter for most sub-cells, so the cheap certificate rarely suffices on its own. Kept: correct, design-consistent, cannot regress on other inputs.

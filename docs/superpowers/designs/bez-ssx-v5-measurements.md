# bez_ssx v5 — Test / Performance Measurements

Append-only log of test + performance measurements. Each row is taken at the moment the implementation is brought into agreement with the then-current design (end of a workflow cycle, before the commit that records the cycle's results).

Commit the measurements file and the design spec together at the end of each iteration so the row pairs with the design SHA.

## Harness

- `examples/ssx/bez_ssx5_baseline.py` — runs the 6 representative cases 3× each, reports min wall time, residual, and branch count.
- `pytest tests/test_bez_csx4.py` — CSX unit tests (12 tests).
- `pytest tests/ --ignore=test_nurbs_compose --ignore=test_boundary_intersection --ignore=test_boundary_intersection_robust --ignore=test_nurbs_ssx --ignore=test_curve_bool` — broader suite (excluding known-broken unrelated tests per context doc §11).

## Notation

- `br` = branches count (found / expected)
- `pts` = total points across branches
- `err` = max `‖S1(s,t) − S2(u,v)‖` over all branch points (diagnostic only for overlap branches — the current impl stores a placeholder `v=0.5` in overlap stuv, inflating the residual; real xyz path is correct)
- `t` = min wall time across 3 runs, milliseconds

## 2026-04-18 — Baseline (commit 6d25fcc, typo fix applied)

State: entry into the refactor session. CSX bug fixed (6d25fcc). All SSX heuristic layers still in place (`_dedup_crossings` xyz branch, `_filter_corner_touches`, `_merge_adjacent_branches`, `_dedup_branches`, overlap sub-segment containment). TΨᵢ is computed once at top level and NOT propagated into sub-cells; sub-cell loop-absence relies only on Gauss-map separability.

| case | br (act/exp) | pts | err | t (ms) | status |
|------|-------------:|----:|----:|-------:|:------:|
| planes      | 1 / 1 | 10 | 3.55e-15 | 11.4  | OK |
| transversal | 1 / 1 | 13 | 6.07e-09 | 11.4  | OK |
| tangential  | 1 / 1 | 12 | 1.62e-06 | 41.0  | OK (Φ-tracer) |
| overlaps    | 2 / 2 |  4 | 2.87e+02 | 18.6  | OK, overlap stuv `v` placeholder bug |
| case5       | 2 / 2 | 48 | 8.33e-08 | 532.0 | OK |
| case6       | 1 / 2 |  9 | 6.34e-08 | 79.7  | MISMATCH — loop missing |

- CSX unit tests: **12 / 12 pass** (`tests/test_bez_csx4.py`)
- Broader suite: **208 pass** (with the 5 unrelated exclusions, 5 warnings)

Known defects captured as entries:

1. **case6 — loop missing**. Structural: no boundary crossing seeds the interior loop, domain decomposition never enters the cell containing the loop.
2. **overlaps — stuv `v` placeholder**. `_find_ssx_boundary_zeros` records `v_oth=0.5` (a CSX default) in overlap endpoint stuv, so `S2(u, v)` differs from `S1(s, t)` by hundreds of units even though the `xyz` path is correct. Diagnostic, not user-visible, but breaks the residual invariant.
3. **TΨᵢ not propagated to sub-cells**. Sub-cell loop-absence check at `_bez_ssx5.py:1674` calls `_check_loop_free(cell.g1, cell.g2)` without T arrays; only Gauss separability is tried. This costs ~62 Gauss calls for case5 per context doc profiling.
4. **proximity-based endpoint matching** in `_trace_all_branches` (`best_dist < 0.1` at line 1443). A heuristic; should be exact parametric.
5. **geometric merging** in `_merge_adjacent_branches` (line 1774). A heuristic covering for the absence of a partition-curve topology layer.

## 2026-04-18 — Iteration 1: propagate TΨᵢ to sub-cells

**Goal:** bring the sub-cell loop-absence check into agreement with design §1.2 and §6 — TΨᵢ must be propagated by de Casteljau split alongside surfaces and Gauss maps, never recomputed, and tried first (cheapest) in `_check_loop_free`.

**Change:** added a `_split_bern_scalar_tensor` helper (wraps `de_casteljau_split_nd` for scalar-valued 4D Bernstein tensors), added `T1..T4` fields to `_Cell`, split the parent's T tensors along the same `cut_axis` at `cut_local` on every cell split, and passed the sub-cell's T arrays into `_check_loop_free`. No heuristic removed this cycle.

| case | br (act/exp) | pts | err | t (ms) | Δ vs baseline | status |
|------|-------------:|----:|----:|-------:|--------------:|:------:|
| planes      | 1 / 1 | 10 | 3.55e-15 | 11.0  | −0.4 ms  | OK |
| transversal | 1 / 1 | 13 | 6.07e-09 | 11.0  | −0.4 ms  | OK |
| tangential  | 1 / 1 | 12 | 1.62e-06 | 39.9  | −1.1 ms  | OK |
| overlaps    | 2 / 2 |  4 | 2.87e+02 | 18.5  | −0.1 ms  | OK |
| case5       | 2 / 2 | 48 | 8.33e-08 | 525.9 | −6.1 ms  | OK (~1 %) |
| case6       | 1 / 2 |  9 | 6.34e-08 | 76.7  | −3.0 ms  | MISMATCH (unchanged) |

- CSX unit tests: **12 / 12 pass** (unchanged)
- Case5 loop-free instrumentation: `mono=4, gauss=10, fail=36` out of 50 `_check_loop_free` calls (vs ~62 Gauss calls before per context doc). 4 sub-cells now terminate via TΨᵢ monotonicity without touching Gauss; downstream sub-cells of those 4 are also avoided.

**Outcome:** design-alignment step. Tests green, no regression, ~1 % case-5 speedup from 4 avoided Gauss evaluations + their downstream subdivisions. The modest impact says most sub-cells in case5 need Gauss separability anyway — the intersection curve is non-monotone in every parameter for most sub-cells, so the cheap certificate rarely suffices. Kept because (a) it matches the design, (b) it is correct (never recomputed), (c) it cannot hurt — monotonicity check is O(n) on Bernstein coefficients, much cheaper than the Gauss witness. Future cases (case6, NURBS adapter with piece-wise partitioning) are likely to benefit more.

## 2026-04-18 — Iteration 2: stuv-only dedupe in `_dedup_crossings`

**Goal:** align `_dedup_crossings` with design §5 Invariant C — a crossing is a duplicate iff its `stuv` matches an existing entry within tolerance; `xyz` proximity alone is not a deduplication criterion (legitimate self-intersections, folds, or two separate branches crossing in 3-space can share `xyz`).

**Change:** deleted the `or np.linalg.norm(c.xyz - d.xyz) < atol` clause in `_dedup_crossings`. The function is now a pure stuv-identity dedupe.

| case | br (act/exp) | pts | err | t (ms) | Δ vs iter-1 | status |
|------|-------------:|----:|----:|-------:|------------:|:------:|
| planes      | 1 / 1 | 10 | 3.55e-15 | 10.7  | −0.3 ms | OK |
| transversal | 1 / 1 | 13 | 6.07e-09 | 10.8  | −0.2 ms | OK |
| tangential  | 1 / 1 | 12 | 1.62e-06 | 39.4  | −0.5 ms | OK |
| overlaps    | 2 / 2 |  4 | 2.87e+02 | 17.9  | −0.6 ms | OK |
| case5       | 2 / 2 | 48 | 8.33e-08 | 512.0 | −13.9 ms | OK (~3 %) |
| case6       | 1 / 2 |  9 | 6.34e-08 | 76.6  | −0.1 ms | MISMATCH (unchanged) |

- CSX unit tests: **12 / 12 pass** (unchanged)

**Outcome:** no regressions. Removing the xyz branch closed Invariant-C violation #1 without any accuracy effect on the 5 working cases. Case-5 wall time dropped ~14 ms (likely a marginal side-effect of fewer dedupe comparisons on the initial crossings list, not the main cost). Case-6 still missing its loop as expected — that requires later iterations (partition topology + §6 step 3).

## 2026-04-18 — Iteration 3: introduce §5 data structures

**Goal:** bring the implementation closer to design §5 by defining the three dataclasses (`IsolineRegistration`, `PartitionCurve`, `BoundaryPoint`) and starting to populate the classification inputs. No behaviour change this cycle; this is pure scaffolding that iter-4/5 will consume.

**Change:**
- Renamed `BoundaryCrossing` to `BoundaryPoint` (kept `BoundaryCrossing` as a back-compat alias) and extended it with `tangent_raw: Optional[NDArray]` and `registrations: list[IsolineRegistration]`.
- Added `IsolineRegistration` and `PartitionCurve` dataclasses matching the design §5 spec.
- Populated `tangent_raw` at every crossing construction site: the two creation points in `_find_ssx_boundary_zeros` (L1 outer-face CSX) and `_isoline_csx_to_global` (internal partitions from subdivision). Tangent computed via the existing `_ssx_tangent_4d` SVD helper without clamping.
- Passed the cell's local surface nets (`cell.g1.surface`, `cell.g2.surface`) into `_isoline_csx_to_global` so the local Jacobian can be evaluated there.

| case | br (act/exp) | pts | err | t (ms) | Δ vs iter-2 | status |
|------|-------------:|----:|----:|-------:|------------:|:------:|
| planes      | 1 / 1 | 10 | 3.55e-15 | 10.7  | 0.0 ms  | OK |
| transversal | 1 / 1 | 13 | 6.07e-09 | 10.7  | −0.1 ms | OK |
| tangential  | 1 / 1 | 12 | 1.62e-06 | 38.9  | −0.5 ms | OK |
| overlaps    | 2 / 2 |  4 | 2.87e+02 | 18.0  | +0.1 ms | OK |
| case5       | 2 / 2 | 48 | 8.33e-08 | 507.7 | −4.3 ms | OK |
| case6       | 1 / 2 |  9 | 6.34e-08 | 75.9  | −0.7 ms | MISMATCH (unchanged) |

- CSX unit tests: **12 / 12 pass**

**Spot-check of the new tangent_raw.** Planes case, two crossings reported at `(0.5, 0, 0.5, 0)` and `(0.5, 1, 0.5, 1)`, both with tangent ≈ `(0, 0.707, 0, 0.707)`. Applying §4 (local ≡ global at top level):
- first point: on-boundary axes `t` (local 0, sign +1) and `v` (local 0, sign +1) → both **entry** ✓
- second point: on-boundary axes `t` (local 1, sign +1) and `v` (local 1, sign +1) → both **exit** ✓

Classification of this single intersection line is now trivially readable from the stored tangent.

**Outcome:** data-structure scaffolding in place with no behaviour change. All 5 passing cases and CSX unit tests green; case-6 still missing its loop (unchanged). Wall-time noise only. Iter-4 will use `tangent_raw` to produce `IsolineRegistration` entries on `PartitionCurve`s.

## 2026-04-18 — Iteration 4: §4 classification at top level

**Goal:** produce `IsolineRegistration` entries on `PartitionCurve`s for every top-level boundary crossing using §4's `(local_param, sign)` table. Sub-cell classification deferred to iter-5; consumer (tracing by registrations) deferred further.

**Change:**
- Added helpers `_partition_free_axis`, `_classify_on_axis` (the design-§4 lookup table), `_build_outer_partitions`, `_on_axis_local`, `_classify_boundary_point`.
- Added `partitions: list[PartitionCurve]` field to `_Cell`.
- In `bez_ssx`, built the top-level `_Cell` (with the 8 outer partitions) BEFORE the loop-free short-circuit, then called `_classify_boundary_point(c, top_cell)` on every L1 crossing.
- The loop-free short-circuit and the iterative decomposition both now use `top_cell` as their entry, so the registrations are always produced.

| case | br (act/exp) | pts | err | t (ms) | Δ vs iter-3 | status |
|------|-------------:|----:|----:|-------:|------------:|:------:|
| planes      | 1 / 1 | 10 | 3.55e-15 | 10.8  | +0.1 ms | OK |
| transversal | 1 / 1 | 13 | 6.07e-09 | 10.8  | +0.1 ms | OK |
| tangential  | 1 / 1 | 12 | 1.62e-06 | 39.1  | +0.2 ms | OK |
| overlaps    | 2 / 2 |  4 | 2.87e+02 | 18.1  | +0.1 ms | OK |
| case5       | 2 / 2 | 48 | 8.33e-08 | 507.3 | −0.4 ms | OK |
| case6       | 1 / 2 |  9 | 6.34e-08 | 76.0  | +0.1 ms | MISMATCH (unchanged) |

- CSX unit tests: **12 / 12 pass**

**Validation of the new registrations.**
- *Planes*: 2 crossings, each with 2 registrations on the 2 on-boundary axes (t and v). `(0.5, 0, 0.5, 0)` → `in` on t=0 **and** `in` on v=0. `(0.5, 1, 0.5, 1)` → `out` on t=1 **and** `out` on v=1. The intersection-line topology is fully readable from these 4 registrations.
- *Case 5*: 4 crossings, all on t faces. Partition t=0 has 2 `in`s at `s=0.243` and `s=0.748`; partition t=1 has 2 `out`s at `s=0.282` and `s=0.763`. The expected two open branches are visible as `in ↔ out` pairs — ready for a consumer.

**Outcome:** top-level §4 classification landed with zero behaviour change. Registrations are correct and complete for the top-level cell. Iter-5: propagate partitions into sub-cells and classify the crossings created at subdivision time.

## 2026-04-18 — Iteration 5: §4 classification at sub-cells

**Goal:** extend §4 classification to every `_Cell` in the decomposition stack — not just the top cell. Every sub-cell is created with its own 8 partitions (unshared for now; cross-cell linkage is a later iteration), and every crossing it owns is registered against those partitions using the same `(local_param, sign)` table.

**Change:**
- Renamed `_build_outer_partitions` → `_build_cell_partitions`, generalized to any cell (uses `cell.box[axis][side]` as the fixed `value` and `cell.box[free]` as the extent). Back-compat alias kept.
- In the main loop's subdivision branch, after creating L and R sub-cells: build their partitions and run `_classify_boundary_point` on every crossing assigned to each side.

| case | br (act/exp) | pts | err | t (ms) | Δ vs iter-4 | status |
|------|-------------:|----:|----:|-------:|------------:|:------:|
| planes      | 1 / 1 | 10 | 3.55e-15 | 11.0  | +0.2 ms  | OK |
| transversal | 1 / 1 | 13 | 6.07e-09 | 11.3  | +0.5 ms  | OK |
| tangential  | 1 / 1 | 12 | 1.62e-06 | 39.2  | +0.1 ms  | OK |
| overlaps    | 2 / 2 |  4 | 2.87e+02 | 18.4  | +0.3 ms  | OK |
| case5       | 2 / 2 | 48 | 8.33e-08 | 520.4 | +13.1 ms | OK (~2.6 %) |
| case6       | 1 / 2 |  9 | 6.34e-08 | 79.0  | +3.0 ms  | MISMATCH (unchanged) |

- CSX unit tests: **12 / 12 pass**

**Validation** (case 5): 49 `_Cell`s total are created; every one has 8 partitions. A crossing at `stuv = (0.748, 0, 0.804, 0.222)` accumulates 79 registrations across the full decomposition tree — one per cell whose boundary it sits on — and in the specific cell where it is *both* `s = 0.748` (cell's s-hi) and `t = 0` (cell's t-lo) it acquires 2 registrations (one per on-boundary axis), matching §4 exactly.

**Outcome:** complete §4 classification at every level. ~2.6 % case-5 slowdown from the extra per-sub-cell classification work — acceptable cost for producer-side completion, and consumer iterations are expected to remove the current proximity scans that dominate wall time. Still no consumer using the registrations; iter-6 begins that work.

## 2026-04-18 — Iteration 6: consume registrations in tracing (Inv. D)

**Goal:** retire the proximity-based endpoint match (`best_dist < 0.1`) inside `_trace_all_branches` in favour of design §7 Invariant D — the marcher stops on the cell boundary, and the stopping registration is found by exact `(partition, param)` identity on the cell's own partitions. Multi-axis corners handled by consuming every same-direction registration on the start/end point in a single march (design §4 through-touch).

**Change:**
- Added `consumed: bool = False` to `IsolineRegistration`.
- Added `_find_exit_registration(cell, stuv_end, tol_param=1e-4)` — walks the cell's partitions on each on-boundary axis of `stuv_end` and returns the best unconsumed out-registration by `param` distance (or `None`).
- Added `_trace_cell_by_registrations(cell, atol)` — iterates UNIQUE start points with an unconsumed in-registration in this cell; marches once per point; consumes all that point's in-registrations in this cell at start and all of the exit point's out-registrations at end.
- Swapped both call sites in `bez_ssx` (top-level short-circuit and sub-cell tracing) to the new tracer.
- `_trace_all_branches` (old heuristic version) and the xyz-proximity `_merge_adjacent_branches` / `_dedup_branches` are still present and still run as the final assembly pass — removing them requires §9 cross-cell matching which is a later iteration.

| case | br (act/exp) | pts | err | t (ms) | Δ vs iter-5 | status |
|------|-------------:|----:|----:|-------:|------------:|:------:|
| planes      | 1 / 1 | 10 | 3.55e-15 | 12.0  | +1.0 ms  | OK |
| transversal | 1 / 1 | 13 | 6.07e-09 | 11.3  | 0.0 ms   | OK |
| tangential  | 1 / 1 | 12 | 1.62e-06 | 41.0  | +1.8 ms  | OK |
| overlaps    | 2 / 2 |  4 | 2.87e+02 | 19.2  | +0.8 ms  | OK |
| case5       | 3 / 2 | 105 | 8.36e-04 | 696.7 | +176 ms | **MISMATCH** |
| case6       | 1 / 2 |  9 | 6.34e-08 | 76.9  | −2.1 ms  | MISMATCH |

- CSX unit tests: **12 / 12 pass**

**Outcome — expected regression on case 5.** Simple cases (planes, transversal, tangential, overlaps) stay correct because tracing happens entirely in the top cell: the stuv of each crossing is one of two corner points, exact match works, and the old xyz-merge at the end is a no-op.

Case 5 is a decomposed case (49 cells, 14 trace invocations). With exact matching, each sub-cell traces sub-branches that end precisely at internal-partition crossings. The downstream `_merge_adjacent_branches` is a proximity-join that does not know anything about partitions or their 1D matching — when sub-branch endpoints no longer coincide exactly across adjacent cells (they used to because the old tracer snapped to the same crossings both sides, generously), the merger fails to glue them into 2 clean curves. Result: 3 branches, one full path residual 5.75e-4 (vs 8e-8 before).

Not fixing this in iter-6. The residual heuristics (`_merge_adjacent_branches`, `_dedup_branches`, `_filter_corner_touches`) are themselves slated for removal and are the wrong tool for cross-cell joining. The clean fix is design §9 — per-partition 1D in/out matching across adjacent cells — which requires:
  1. Shared `PartitionCurve` objects across L/R sub-cells at a subdivision (currently unshared — iter-5 builds fresh per cell).
  2. A post-trace assembly pass that walks shared internal partitions and matches in/out pairs by `param`.

These are iter-7 and iter-8. Case 5 is expected to return to 2 branches once both land. Iteration remains in "bringing implementation to design" mode — no heuristic fallbacks.

## 2026-04-18 — Iteration 7: share internal partitions across L/R (Inv. A prerequisite)

**Goal:** when a parent cell splits at `(cut_axis, cut_global_val)`, the new face separating the two children must be a single `PartitionCurve` object shared by both of them. Design §5 Invariant A requires that a crossing on this face appears in *both* cells' views with flipped direction — impossible if each child builds its own copy.

**Change:**
- Gave `_build_cell_partitions` an optional `skip=(axis, side_idx)` parameter that omits one box face, making space for the splice-in.
- In the subdivision branch: create a single `PartitionCurve` at `(cut_axis, cut_global_val)`, set its `free_axis` = the other axis of the owning surface, and its `global_extent` = the parent's range on that free axis. Skip the cut face when building each child's own partitions, append the shared one, and grow its `adjacents` list.

| case | br (act/exp) | pts | err | t (ms) | Δ vs iter-6 | status |
|------|-------------:|----:|----:|-------:|------------:|:------:|
| planes      | 1 / 1 | 10 | 3.55e-15 | 11.0  | −1.0 ms  | OK |
| transversal | 1 / 1 | 13 | 6.07e-09 | 11.2  | −0.1 ms  | OK |
| tangential  | 1 / 1 | 12 | 1.62e-06 | 39.4  | −1.6 ms  | OK |
| overlaps    | 2 / 2 |  4 | 2.87e+02 | 18.6  | −0.6 ms  | OK |
| case5       | 3 / 2 | 105 | 8.36e-04 | 697.4 | +0.7 ms  | MISMATCH (unchanged from iter-6) |
| case6       | 1 / 2 |  9 | 6.34e-08 | 78.2  | +1.3 ms  | MISMATCH |

- CSX unit tests: **12 / 12 pass**

**Verification of sharing.** Case 5 instrumentation: 49 cells produced by 24 splits, yielding 24 distinct `PartitionCurve` objects with `len(adjacents) == 2` — exactly one per split. Outer partitions (the 8 boundary faces of any sub-cell that isn't on the cut) remain unshared with `len(adjacents) == 1`. Total: 368 partition objects across 49 cells, as expected for unshared outer + shared internal.

**Outcome:** structural enabler for iter-8 is in place. No behaviour change — no consumer reads `adjacents` yet. Case 5 still MISMATCH at 3 branches; next iter wires the cross-cell assembly and is expected to close it.

## 2026-04-18 — Iteration 8: §9 cross-cell fragment assembly

**Goal:** replace `_merge_adjacent_branches` + `_dedup_branches` (xyz-proximity heuristics) with design §9 — chain tracer output by *shared `BoundaryPoint` identity*. The same physical crossing on an internal partition is the SAME `BoundaryPoint` object shared by L and R, so chaining is exact by `id()`.

**Change:**
- Added `_Fragment` dataclass carrying `start_point`, `end_point`, `stuv_path`, `xyz_path`.
- `_trace_cell_by_registrations` now returns `list[_Fragment]` instead of `list[SSXBranch]`; fragments with an unmatched endpoint carry `end_point=None`.
- Added `_assemble_fragments` — builds an `id(BoundaryPoint) → list[(frag_idx, role)]` index, then walks chains forward and backward by shared endpoints, reversing fragments as needed so the chain reads head-to-tail. Concatenates each chain into one `SSXBranch`.
- Main loop now accumulates `all_fragments` across every cell; final assembly runs once at the end.
- `_merge_adjacent_branches` / `_dedup_branches` are no longer called.

| case | br (act/exp) | pts | err | t (ms) | Δ vs iter-7 | status |
|------|-------------:|----:|----:|-------:|------------:|:------:|
| planes      | 1 / 1 | 10 | 3.55e-15 | 11.2  | +0.2 ms   | OK |
| transversal | 1 / 1 | 13 | 6.07e-09 | 11.1  | −0.1 ms   | OK |
| tangential  | 1 / 1 | 12 | 1.62e-06 | 40.4  | +1.0 ms   | OK |
| overlaps    | 2 / 2 |  4 | 2.87e+02 | 18.6  | 0.0 ms    | OK |
| case5       | 11 / 2 | 117 | 8.36e-04 | 1335.5 | +638 ms | **MISMATCH (worse)** |
| case6       | **2 / 2** | 20 | 1.63e-04 | 79.1  | +0.9 ms | **OK (first time)** |

- CSX unit tests: **12 / 12 pass**

**Case 6 now matches.** The first iteration that gets case 6's branch count right. The cross-cell fragment chaining naturally links the loop fragments that were split across sub-cells. (Residual 1.63e-4 is loose vs `atol=1e-3` but not worse than the marcher's inherent step error — the branches themselves are topologically correct.)

**Case 5 regression (11 vs 2).** Instrumentation: 18 fragments produced across all cells, 12 of which have `end_point = None` (the marcher's exit stuv was not matched by `_find_exit_registration`). Assembly can chain only fragments whose endpoints are the *same object*; a fragment with `end_point = None` terminates a chain immediately. 12 dead-end fragments + 6 chained → 11 assembled branches.

The root cause is `_find_exit_registration`'s failure rate on case-5 sub-cell boundaries, not the assembly itself. The marcher's stopping point after `np.clip` should land exactly on the cell's box `lo`/`hi` for at least one axis, and `_find_exit_registration` should find a match. Something about case 5's numerical flow is making this fail often enough to break 2 branches into 11 pieces. **Not investigating in this cycle per the workflow discipline**: the remaining iterations (Krawczyk tangency, final heuristic cleanup) may shift the picture, and then we look at case 5 with a clean implementation.

**Outcome:** §9 assembly is wired — mechanism is correct in principle (case 6 proves it) but depends on `_find_exit_registration` matching on every well-posed sub-cell boundary. Net: +1 case now OK, −1 case now worse; overall closer to design.

## 2026-04-18 — Iteration 9: Krawczyk tangency check (§6 step 3)

**Goal:** wire the Krawczyk/Gauss-Newton tangency certificate between "both cheap certificates failed" and "subdivide". When `TΨ = 0` is certified to have a simultaneous root inside a cell, invoke the Φ-tracer (design §1.4, §8) instead of subdividing further — the Φ system has rank 3 at the tangent point where Ψ is rank-deficient, so marching Φ works there while marching Ψ does not.

**Change:**
- In the main decomposition loop, after the `_check_loop_free` branch and before `depth >= max_depth`, strip the cell's local homogeneous surfaces to cartesian, call `_check_tangency` on `(T1..T4, P1_cart_local, P2_cart_local, local_box=[0,1]⁴)`.
- If `True`: convert `cell.crossings` to local stuv, call `_deflate_tangent_cell`, convert the Φ-traced branches back to global stuv, emit as unchained `_Fragment`s (their endpoints are not Ψ crossings, so they don't participate in §9 chaining).
- If `None` / `False`: fall through to the existing subdivision path.

| case | br (act/exp) | pts | err | t (ms) | Δ vs iter-8 | status |
|------|-------------:|----:|----:|-------:|------------:|:------:|
| planes      | 1 / 1 | 10 | 3.55e-15 | 11.1  | −0.1 ms | OK |
| transversal | 1 / 1 | 13 | 6.07e-09 | 11.6  | +0.5 ms | OK |
| tangential  | 1 / 1 |  9 | **3.97e-15** | 45.8  | +5.4 ms | **OK (machine precision)** |
| overlaps    | 2 / 2 |  4 | 2.87e+02 | 18.9  | +0.3 ms | OK |
| case5       | 11 / 2 | 117 | 8.36e-04 | 1724.5 | +389 ms | MISMATCH (unchanged count) |
| case6       | 2 / 2 | 20 | 1.63e-04 | 86.4  | +7.3 ms | OK |

- CSX unit tests: **12 / 12 pass**

**Tangential residual drop** `1.62e-06 → 3.97e-15`: the Φ-tracer is now actually used. Previously the tangent case was tracing via the ordinary Ψ marcher, which converges poorly at `rank(J_Ψ) < 3`. Φ augments with a `TΨ` row to restore rank, so every marched point is numerically precise. Point count 9 (vs 12 before) — cleaner curve.

**Case 5 timing +389 ms** is the cost of running Krawczyk on every sub-cell that fails both cheap certificates (36 of 50 sub-cells at top-profile). Not optimising in this cycle; iter-10 removes several dead heuristic helpers and may change the picture, and then case 5 gets its dedicated look.

**Outcome:** §6 step 3 wired. Tangential now matches the design's intent for the first time in this session. No other regressions from the Krawczyk wiring itself.

## 2026-04-18 — Iteration 10: remove retained heuristics

**Goal:** delete every heuristic / dead-code path listed in the audit so the implementation contains only design-sanctioned structures. Nothing in the design refers to these functions anymore; they were kept alive only by legacy call sites.

**Removed:**
- `SubdomainCell` dataclass (unused)
- `_is_on_both_boundaries` (unused)
- `_crossing_on_box_boundary`, `_crossing_in_box_interior`, `_domain_decompose` (unused)
- `_process_monotonic_case` + `_trace_segment` (unused after iter-8 killed the last call site)
- `_is_cell_corner`, `_tangent_enters_cell`, `_filter_corner_touches` (heuristics for the old tracer)
- `_trace_all_branches` (old proximity-matching tracer — replaced by `_trace_cell_by_registrations` in iter-6)
- `_merge_adjacent_branches`, `_dedup_branches` (xyz-proximity merge/dedup — replaced by `_assemble_fragments` in iter-8)
- Overlap sub-segment containment block in `_overlaps_to_branches` (lines 1291–1322 before this cut — duplicate-overlap filter by xyz containment)

`_pair_crossings_for_tracing` is intentionally retained: it is a `.face`-based crossing pairer used *only inside* the Φ tracer (`_deflate_tangent_cell`). Design §8 leaves the Φ-tracer's own pairing as a follow-up; cleaning it requires reconsidering how registrations interact with the Φ system.

| case | br (act/exp) | pts | err | t (ms) | Δ vs iter-9 | status |
|------|-------------:|----:|----:|-------:|------------:|:------:|
| planes      | 1 / 1 | 10 | 3.55e-15 | 11.2  | +0.1 ms | OK |
| transversal | 1 / 1 | 13 | 6.07e-09 | 11.4  | −0.2 ms | OK |
| tangential  | 1 / 1 |  9 | 3.97e-15 | 47.2  | +1.4 ms | OK |
| overlaps    | **3 / 2** | 6 | 2.87e+02 | 18.8 | −0.1 ms | **MISMATCH (new)** |
| case5       | 11 / 2 | 117 | 8.36e-04 | 1743.2 | +18.7 ms | MISMATCH |
| case6       | 2 / 2 | 20 | 1.63e-04 | 91.2  | +4.8 ms | OK |

- CSX unit tests: **12 / 12 pass**
- File size: 1925 lines (was 2082 lines before iter-10 — net −157 from this cut, with earlier iters having grown the file by a similar amount).

**Overlaps regression**: one of the duplicate overlap branches is now emitted as a third branch. The deleted sub-segment containment was filtering a shorter overlap whose endpoints lay on a longer one. Removing it reveals a real issue in `_find_ssx_boundary_zeros` / `_overlaps_to_branches`: duplicate overlaps are still being produced at the boundary-CSX level. The design's §5 Invariant C applies to overlap endpoints too (stuv-identity, not xyz), and overlap classification + registration is not yet implemented — §5 only covers `BoundaryPoint`s. Cleaning this up is its own follow-up.

**Outcome:** implementation now matches the design except for:
  1. Overlap branches don't follow §5 (no `BoundaryPoint` / registration integration).
  2. Φ-tracer's `_pair_crossings_for_tracing` is still a `.face`-based heuristic.
  3. The `_find_exit_registration` matching is not 100 % reliable on case 5 sub-cell boundaries (the case-5 MISMATCH).

End of the "bring impl to design" phase. Next: investigate case 5 specifically with the clean codebase.

## 2026-04-18 — Iteration 11: 3.1 + Medium/Low audit findings (information only)

**Goal:** address the fresh-audit items 3.1 (cut-clamp), #3, #4, #5, #6, #7, #8 in one cycle. Run tests for information — no subsequent patches. The bigger audit items (#1 SVD direction hint, #2 param-matching assembly, 3.2 Krishnan-Manocha all-crossing cuts) are deferred to a design discussion after these results land.

**Changes:**
- **3.1** — `_choose_cut` rewritten: scores by *closeness to cell centre* instead of balance; rejects candidates with local position < `min_margin` (default 0.05) or > 1 − `min_margin`; returns `(None, None)` when no candidate meets the margin (cell terminates without further subdivision). The legacy `cut_local = max(0.01, min(0.99, cut_local))` artificial clamp is gone.
- **#5** — deleted the xyz-proximity filter that removed crossings coinciding with overlap endpoints (lines 1693-1703 pre-iter-11). Redundant with the stuv filter L1 already runs; violated Invariant C.
- **#7** — removed the top-level loop-free short-circuit. `top_cell` now always enters the decomposition stack and goes through the same §6 lifecycle; if it's loop-free the first iteration traces and the loop exits — behaviourally identical, one code path.
- **#8** — `_check_tangency` returns `None` (not `False`) when the interval-arithmetic range of a `TΨᵢ` excludes 0. Design §6 step 3 calls for None here.
- **#3 + #4** — `_deflate_tangent_cell` now emits `_Fragment`s carrying `start_point` / `end_point` BoundaryPoint references via a new `originals=` parameter, so Φ-branches join the same §9 assembly as Ψ-branches. `_pair_crossings_for_tracing` gained a registration-aware code path keyed by `direction` on `IsolineRegistration` — it no longer uses the legacy `.face` tag when a cell is supplied.
- **#6** — documented `_Cell.crossings` as redundant scaffolding (still present; full removal deferred).

| case | br (act/exp) | pts | err | t (ms) | Δ vs iter-10 | status |
|------|-------------:|----:|----:|-------:|-------------:|:------:|
| planes      | 1 / 1 | 10 | 3.55e-15 | 11.0  | −0.2 ms | OK |
| transversal | 1 / 1 | 13 | 6.07e-09 | 11.0  | −0.4 ms | OK |
| tangential  | 1 / 1 |  9 | 3.97e-15 | 44.7  | −2.5 ms | OK |
| overlaps    | **4 / 2** | 8 | 2.87e+02 | 18.8 | +1 branch | **MISMATCH (worse)** |
| case5       | **4 / 2** | 71 | 8.74e-04 | **447.0** | **−7 branches, −1296 ms (×3.9 speedup)** | MISMATCH (much better) |
| case6       | 2 / 2 | 20 | 1.63e-04 | 67.1  | −24 ms | OK |

- CSX unit tests: **12 / 12 pass**.

**Interpretation.**
- Case 5 dramatically improved: 11 → 4 branches, 1743 ms → 447 ms (~3.9 × speedup). The center-preferring cut in 3.1 is almost certainly the driver — cuts very close to a cell's boundary used to produce near-zero-width sub-patches that themselves needed subdivision, inflating both the tree and the number of dead-end fragments. Requiring cuts to be at least 0.05 from the cell's boundaries produces a much shallower decomposition tree. Case 5 still MISMATCHes at 4/2 for the fundamental reason noted in iter-6/iter-8 (marcher direction + identity-based assembly), but it's now within a factor of 2.
- Overlaps 3 → 4. Removing the xyz filter (#5) removed one of the heuristic dedup paths, exposing one more duplicate that neither filter catches. The real cause, already known, is that overlap endpoints don't participate in §5 (no `BoundaryPoint` with registrations).
- Tangential and case 6 unchanged (correctness preserved).
- Φ-fragments now carry BoundaryPoint refs; they'd chain via §9 if a Φ-branch spans multiple cells. No current case exercises that path, but the plumbing is now consistent with Ψ.

**No fixes applied.** Results recorded as informational input for the next design cycle.

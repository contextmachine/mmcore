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

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

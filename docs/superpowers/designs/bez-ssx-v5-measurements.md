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

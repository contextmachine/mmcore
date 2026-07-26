# P2 — the case-11 knob-unreachable tier: measurements + first fixes (2026-07-25)

Instrumentation pass per the 2026-07-20 kickoff §P2, run against the
post-cluster-4 engine (branch `ssx5-derived-envelopes`).  Monkeypatch-only
ledger; no source edits during measurement.

## Ground truth first

Case 11's intersection is a **single closed loop of length 1261.25**
(`complete=True, reasons=[]` at atol=1e-1 and 1e-2, both recovering it).
Coverage then collapses as the tolerance tightens:

| atol | result | traced length |
|---|---|---|
| 1e-1 | complete, 1 closed branch | 1261.25 |
| 1e-2 | complete, 1 closed branch | 1261.44 |
| 1e-3 | 2 open fragments | (overlapping) |
| 1e-4 | 4 open fragments | 907.5 |
| 1e-5 | 2 open fragments | 45.3 (3.6%) |
| 2.5e-6 | 2 open fragments | 22.5 (**1.8%**) |

This is not a convergence failure.  The march advances monotonically and
never revisits geometry (0/64 sampled anchors revisited within half a
step); it is simply **truncated**.  Raising the per-march cap extends
coverage exactly linearly (×1 / ×4 / ×16 → 11.3 / 45.8 / 183.9), and the
step size is constant at ~0.0575 throughout.

## Why no knob reaches it

Three limiters, none of them the public work budget:

1. **`_bez_ssx5.py:4774` — hardcoded `trace_limit = 400`.**  The binding
   constraint.  Step size scales as √atol, so the points needed scale as
   1/√atol: covering a 1261-long loop at atol=2.5e-6 needs ~22,000 points
   per march against a fixed allowance of 400.  This is a bare constant
   standing in for an arc-length / step ratio — the audit's D! pattern in
   the INTEGER-cap family, which the epsilon-literal sweep did not cover.
2. **`_nssx5.py:1364` — per-pair clamp to `_BEZ_DEFAULT_MAX_CELLS`.**
   `_make_aggregate` documents explicit values as "absolute aggregate
   promises", but every per-pair `bez_ssx` call was clamped to the module
   default regardless.  With ONE candidate pair that made the public knob
   inert: `max_cells=2_000_000` still handed the engine 250k.
3. **Reason misbilling at three sites**, all reporting `work_budget` —
   which consumers read as "raise a knob" — while the shared ledger sat at
   1.2–4.2% utilization.

## Landed this session (user decision: re-type + forward the budget)

- **`REASON_TRACE_POINT_CAP = "trace_point_cap"`** added, in the spirit of
  the existing `REASON_SINGULAR_SET` ("the honest replacement for the
  former work_budget misbilling").  `_trace_cell_by_registrations` now
  tracks WHICH allowance bounded each march and bills the shared ledger
  only when the ledger was actually the binding one.
  Measured: case 11 at atol=1e-3 now reports `['trace_point_cap']` alone.
- **`_per_pair_allowance`** replaces the fixed per-pair clamp.  An unset
  budget is unchanged BY CONSTRUCTION (aggregate = default × n_candidates,
  so an even split is exactly the default); only an explicit aggregate
  redistributes, and it stays fair by dividing what REMAINS among the
  candidates that remain.  Pinned by two tests, including one that the
  default path must not move.

## Adversarial review response (2026-07-26)

Two majors, both confirmed and fixed:

**The fair-share split violated the contract it cited.** I invented an
even `ceil(remaining / n_remaining)` slice on the explicit path, reasoning
that an absolute promise should still be shared fairly. But work is not
spread evenly over BVH candidates, and the house reading of "absolute" is
the reference adapters': `_ncsx4` and `_nccx4` both hand each call the
ENTIRE remainder. Measured cost of my version on harness case 1 (43
candidates): `max_cells=250_000` went from `complete=True, reasons=[]` to
`reasons=['work_budget']` with **61% of the caller's explicit aggregate
unspent** — reintroducing, on the explicit path, the exact misbilled
knob-unreachability this change exists to remove. Now grants the full
remainder; all four regressed rows restored to `complete=True` at their
pre-commit cell counts (100,922 / 21,571).
Root cause of MISSING it: both my tests used a single-candidate fixture,
where the fair-share branch is an identity. Multi-candidate coverage added.

**The re-typing never reached the consumer that acts on it.**
`trace_point_cap` was absent from `STRUCTURAL_REASONS` in
`examples/ssx/nurbs_ssx5_coverage_check.py` — the very file the commit
edited (CASE_NOTES twenty lines below). So case 11 was still counted a
resource FAIL. Fixed: **case 11 is now `PARTIAL(typed)` and the gate
PASSES.** Naming a reason is only half the work; the closed sets that
classify it are the other half.

Also fixed: `test_reason_vocabulary_is_stable` asserted individual names
but never the closed SET, so it could not detect the drift it exists to
detect — two reasons had already slipped through (`unresolved_singular_set`
at L52, `trace_point_cap` here). It now pins the set and names every
registration site in its failure message.

## Open — filed, not guessed

- **The `trace_limit = 400` derivation** is its own tier (the user's
  sequencing).  It needs an arc-length bound plus a fixture, and the cost
  is real: at ×16 the C3 tier went 987 → 165,917 cells.  Note the adapter
  fix does NOT move case 11 on its own — cells were never binding there
  (6,190 of 250,000), so the harness row stays at 92.68%.
- **The `_run_csx` site (`:6312`) is ALSO knob-unreachable**, contrary to
  `REASON_WORK_BUDGET`'s documented scope ("…or a CSX per-call tier ran
  dry"): raising `csx_max_cells` 10× and `boundary_csx_max_cells` 20×
  changes neither the reason nor the cell count (6,190).  The truncated
  `bez_csx` call used **1,817 of 100,000 cells (1.8%)** with
  `boundary_topology_complete=True` and 2 isolated roots — the signature
  of the Phase-2 `max_depth` guard, for which `REASON_DEPTH_LIMIT` already
  exists.  Typing it correctly requires `bez_csx` to REPORT its truncation
  cause rather than a bare `budget_exhausted` flag; that is a schema
  change (v2 contract) and is deliberately left un-guessed.
- The third site (`:7842`, C3 `_c3_stats['incomplete']`) was not isolated
  this session.

## Method note

The ledger idiom is worth keeping: wrap `SoftWorkBudget.mark_exhausted` /
`mark_incomplete` / `charge_*` and capture `traceback.extract_stack()`
filtered to mmcore frames.  It named all three sites on the first run.
The follow-up question — "is this reason honest?" — is answered by raising
the knob it names and re-measuring; two of three failed that test.

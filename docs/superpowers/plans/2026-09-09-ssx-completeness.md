# SSX Completeness Implementation Plan

> Execution is autonomous as explicitly requested by the user. Independent audits run in parallel; production ownership is separated by module.

**Goal:** Repair systematic branch-loss and redundant-search mechanisms, validate with independently derived geometry, and provide an evidence-based reliability assessment.

**Architecture:** Retain subdivision and shared budgets. Correct unsafe termination and representation predicates; separate lifted-parameter identity from geometric closeness; remove provably redundant nested work.

**Tech stack:** Python, NumPy, existing Bernstein/Cython primitives, pytest, cProfile.

**Spec:** `docs/superpowers/specs/2026-09-09-ssx-completeness-design.md`

## Constraints

- Preserve the existing baseline and unrelated local edits.
- Keep Python 3.9+ source compatibility and current public schemas.
- Preserve the finite work, call, output, and postprocess allowances. Explicit depth limits remain hard; the arbitrary default depth 13 is removed so unused work can resolve a source proof. Do not claim a universal proof from tests.
- All slow probes use subprocess watchdogs; tests derive expected geometry independently.

## Tasks

- [x] Establish isolated checkout and baseline NURBS SSX verification: 57 tests passed in 45.38s.
- [x] Audit Bézier search termination, tracing consumption, and unsearched tangent complements; repair confirmed defects in `_bez_ssx5.py`.
- [x] Audit NURBS representation and assembly. Replace approximate rationality with exact metadata handling; preserve distinct lifted branches/points in `_nssx5.py` and shared containment helpers.
- [x] Profile CSX/CCX, audit endpoint and terminal root isolation, and remove demonstrated repeated work while retaining root completeness.
- [x] Add independent analytic/generative branch coverage and representation-invariance verification.
- [x] Integrate changes, run relevant legacy regression gates, review mathematical assumptions and full diff independently, and record limitations and reproducible profile evidence. Final legacy diagnostic remains 91 passed / 21 failed / 5 watchdogs; this task completion records the audit, not a universal completeness proof.

## Evidence ledger

- Baseline logs: `/tmp/mmcore-ssx-audit/baseline-nssx5.txt`.
- Near-unit rational weight counterexample: `/tmp/mmcore-ssx-audit/probe_nurbs_weights.py`. Two exact transverse lines become one tangential line with `complete=True` through approximate `_is_rational`.
- Tangent line plus disjoint regular circle: `/tmp/ssx_audit_tangent_ring.py`. Boundary tangent tracing skips the ordinary complement.

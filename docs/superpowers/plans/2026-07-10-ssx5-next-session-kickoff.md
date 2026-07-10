# SSX v5 — next-session kickoff (singular-cases hardening)

**Written:** 2026-07-10, from the branched interactive session.
**Purpose:** everything a fresh session must read and obey before touching SSX code.

## 0. Read these, in this order, before any edit

1. Memory `project_ssx5_singular_cases.md` + `project_ssx5_branch_loss_fixes.md` (auto-indexed in MEMORY.md) — what shipped, accepted limits, tolerance conventions.
2. `docs/superpowers/plans/2026-06-10-ssx5-singular-cases-handoff.md` — pipeline map, tolerance ladders, primitives, debugging playbook. **Mandatory before touching `_bez_ssx5.py` or `_ssx5_singular.py`.**
3. `docs/superpowers/plans/2026-06-10-ssx5-singular-cases.md` — the executed plan; **AS-BUILT blockquotes override the original task text wherever they disagree.**
4. `docs/superpowers/issues/2026-07-07-ssx5-singular-review-ledger.md` — THE work-item source of truth (27 items, 25 closed at `dcb5a76`). Repro one-liners live in the item text; full evidence was in session task dirs that may no longer exist — reproduce from the ledger text, don't hunt for old /tmp files.
5. Reference paper: `3592452-2.pdf` (Cheng et al. 2023), §3–5 for C1/C2/C3 definitions; `237748.237751.pdf`/`.md` (Krishnan–Manocha) for the decomposition framework.

## 1. Git discipline (user requirement — changed from before)

- **All development on a dedicated branch, NOT on `tiny`.** First action:
  `git status && git log --oneline -5` (another session may still be running — reconcile, never clobber), then `git checkout -b ssx5-singular-hardening` from current `tiny` HEAD.
- Commit per completed+validated item, ledger updated in the same commit (existing convention: `fix(ssx5): L<n> — ...` / `docs(ssx5): ledger — ...`).
- Coordination protocol from the ledger header: mark an item `CLAIMED(<who>)` in the ledger before starting it.
- No pushes, no merges to `tiny` without the user.

## 2. Open work items (priority order — updated 2026-07-10 with new USER cases)

**P0 — COMPLETE on `ssx5-singular-hardening` (2026-07-10; pending commit):**
1. [x] **Case 14** (`examples/ssx/bez_ssx5_case14.py`) — instrumentation found the freeze in `_find_ssx_boundary_zeros → bez_csx → _phase2_isolated_search`: an identically collapsed rational apex edge was represented as 16,385 isolated pseudo-roots, then fed quadratic dedup/splitting. CSX now emits a typed parameter fiber through a bounded point-on-surface path; SSX canonicalizes both apex fibers and traces the certified rational tangent generator. The unresolved positive-dimensional Δ complement is honestly partial.
2. [x] **Case 13** (`examples/ssx/bez_ssx5_case13.py`) — crossing-less descendants repeatedly repaid the same Φ seed search after deduplication. Ancestor-box attempt caching plus shared Φ enumeration/marching charges make the search finite. The original script terminates with one certified tangent point and explicit partial status for the unresolved complement.
3. [x] **Global soft budget** — one call-wide allowance covers SSX cells, nested CSX calls/cells, singular/C1/C3 work, output growth, and a separate finite postprocess counter. Top-level independent boundary probes have a 20k local cap; topology-critical internal cuts receive one established 100k allowance, always clamped by the shared remainder. Zero/tiny allowances preflight the superlinear distance-net build. Exhaustion returns certified partial output with `'budget_exhausted': True` and detailed usage counters.

**P1 — AWAITING USER OUTPUT-SCHEMA DECISION (no implementation started):**
4. [ ] **Case 12 / L28** (`examples/ssx/bez_ssx5_case12.py`) — true answer is a 2D surface-overlap region. The choices to present are: (A) tagged 3D boundary branch(es), (B) a structured trimmed-region entity, or (C, recommended) the structured region with paired parameter-space loops on both surfaces plus references to compatibility boundary branches. Do not claim or implement L28 until the user selects the contract.

**P2 — previously open ledger items:**
5. [ ] **L25 — edge-graze transversal arc loss** (pre-existing at `40a79a1`). Not started because priority stops at the pending P1 schema decision. Use the minimal edge-graze geometry, coverage harness, and crossing-lifecycle instrumentation (2026-06-09 playbook) before any fix.
6. [x] **L26 — rational Ψ in the tangency witness** — completed as part of P0: Δ/Φ/tangency decisions now evaluate homogeneous rational surfaces exactly rather than per-control-point dehomogenized polynomial surrogates.
7. **L27 residuals** (documented, not regressions) — only on demand.

**Also completed:** the coverage harness honors a module-level `RATIONAL` flag and rejects incomplete reference CSX slices. Cases 12–14 are registered as L28–L30. The proactive overlap/rational/no-hang review added and fixed L31–L40; see the ledger for tests and evidence.

## 3. Non-negotiable validation gates (run after EVERY task, before every commit)

```bash
.venv/bin/python examples/ssx/bez_ssx5_coverage_check.py          # exit 0: 7 cases, 100% coverage AND zero spurious singularities (enforced)
.venv/bin/python -m pytest tests/test_bez_ssx5_singular.py -q     # the singular gate (50 tests at last count)
.venv/bin/python -m pytest tests/test_bez_csx4.py tests/test_bez_ccx4.py tests/test_bez_ccx3_cases.py tests/test_bezier_common.py tests/test_bezier_curves_overlap.py -q
```
Plus: legacy 4 mini-cases (planes / transversal / tangential / overlaps — overlaps now correctly 2 branches + 1 junction `tangent_point` since L27); timings within ~1.2× of the numbers recorded in the ledger/memory. A "fix" that trades coverage or adds spurious singularities is a regression, full stop.

## 4. Invariants that were paid for in blood (do not re-learn them)

- **A parametric box is not a metric ball.** Every destructive op (dedup, subsumption, unification) carries BOTH a param radius AND an xyz guard. Ladders: destructive = 1·ptol ∧ xyz ≤ atol; matching/unification = 4·ptol box ∧ xyz ≤ 2·atol.
- **Sound prunes only.** Never prune a cell/box from one Newton trajectory's behavior; inconclusive ⇒ subdivide; termination comes from hull certificates + resolution floors + budgets (`max_cells`), never from optimism.
- **Sign/hull tests need roundoff margins** (L1): exclusion only beyond `k·eps·max|coeff|`; margins make exclusion stricter (sound direction).
- **Never merge or report tolerance-touches in CSX** (sub-atol valleys): it destroys sub-tolerance topology (near-tangent loops). The G-net component sign prune is what keeps CSX fast — keep it sound.
- **Marchers are xyz-reparameterized**: step target `h` in xyz via local speed; deviation control = sagitta + mid-chord verification (S-span blind spot!); exit chords verified; on rejection make geometric progress (halfway retargeting), don't just halve; stagnation escapes bounded at 25; bounce = max-over-path xyz ≤ atol. `_march_to_boundary` returns a 3-tuple `(stuv, xyz, exit_info)`.
- **Determinism**: no module-global PRNG (L17); deterministic mid-plane slices instead of random hyperplanes (user decision).
- **Units**: everything in model units; `atol` (default 1e-3) is the only user tolerance; report deviations as multiples of atol, never invent "mm".
- **All solvers budget-bounded** — the original no-hang guarantee still stands.
- User decisions on record: `result['singularities']` typed list + `SSXBranch.kind`; Bernstein Φ∩L subdivision solver (not Krawczyk/random-L); C₂ multiplicity ≥ 3 out of scope (paper's own limit).

## 5. Environment quirks

- Python: `.venv/bin/python` (3.14). `timeout` command does not exist on this Mac (use `run_in_background`). `/tmp` gets wiped (keep durable artifacts in-repo; the coverage harness IS the durable diagnostic). `rich` once vanished from the venv (`pip install rich` fixes the `_ncsx2` import chain).
- Useful instrumentation patterns (all monkeypatch, no source edits): LoggedCell subclass of `_Cell`, wrapped `_trace_cell_by_registrations`, wrapped `bez_csx` — see the handoff doc's debugging playbook.

## 6. Process

- `superpowers:systematic-debugging` for any defect (root cause before fixes — this is how L-series items were found); TDD for every fix (failing test first, into `tests/test_bez_ssx5_singular.py` or `tests/test_bez_csx4.py`).
- After any substantial milestone: run an adversarial multi-agent review workflow over the diff (two of these caught 3 confirmed majors each in freshly-written, test-passing code). Verify findings before acting on them; append confirmed ones to the ledger.
- End of session: update the ledger checkboxes, memory files, and this kickoff doc's §2.

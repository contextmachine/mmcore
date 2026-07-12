# SSX v5 — next-session kickoff (singular-cases hardening)

**Written:** 2026-07-10, from the branched interactive session. **Updated 2026-07-12.**
**Purpose:** everything a fresh session must read and obey before touching SSX code.

> **Session-start prompt (copy-paste; updated 2026-07-12 late evening — L25/L47/L48/L49/L51/L53 are DONE, L55 resolved; see §2):**
> Continue SSX v5 hardening on branch `ssx5-singular-hardening` (NOT `tiny`; no pushes).
> Start with `git status && git log --oneline -10` — expect the L25/L47/L48/L49/L53/L51
> commits above `28a9e4d`; reconcile, never clobber.
> Read first: this kickoff §0–§4 (§2 = what shipped in BOTH 2026-07-12 sessions and
> what remains); then the ledger entries L50 (investigated-open, fixture analysis +
> fix direction recorded), L52, L54 in
> `docs/superpowers/issues/2026-07-07-ssx5-singular-review-ledger.md`; then review doc
> §11 steps 5–6 in `docs/superpowers/plans/2026-07-12-ssx5-budget-review-and-overlap-contract.md`.
> Queue: 1. **L54 results** — two Opus line-scan agents (A1 SSX; A2 CCX/CSX/closest-point
> + the z=const≠0 lead) ran 2026-07-12 evening; their findings are recorded in the
> ledger under L54 — VERIFY each before acting (superpowers:receiving-code-review),
> fix confirmed ones per TDD. 2. **L52** consolidation batch (one budget/status module,
> exactness kits, margin policies, dead code incl. `_csx_on_cut_face`/`_midpoint_split`,
> the c1-tier misbilling L49 found) + §11 steps 5–6 (budget_contract gate, de-budget
> milestones). 3. **L50** only if a REACHING two-tangency fixture is found (see the
> ledger's recorded analysis; do not fix without it). Schema v2 is IN PLACE: new work
> reads/emits `complete`/`status.reasons`, never `budget_exhausted`, at the bez_ssx
> level. CCX overlaps now carry `certification: 'exact'|'tolerance'` (L47 user contract).
> Process: mark items CLAIMED(<who>) in the ledger before starting; TDD (failing test
> first); run §3 gates after every item and before every commit; commit per item with
> the ledger update in the same commit; adversarial review after substantial milestones.
> Invariants: §4. Subagents default to Opus (haiku for mechanical checks); Fable only
> where genuinely needed.

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

## 2. Open work items (priority order — updated 2026-07-12 EVENING: the queue below is DONE)

**COMPLETE on `ssx5-singular-hardening` (2026-07-12, commits `ae6a4c6`..`1257fdb`; per-item evidence in the ledger):**
1. [x] **L41** — schema v2 shipped (`complete`/`status.reasons`/`status.work`; `mark_incomplete(reason)` required; reasons measured per case: 12 → overlap+work, 13 → depth_limit, 14 → fiber+tangential+multiplicity+work); adapters flipped to always-return-status with candidate-scaled default allowances; live v4 callers migrated (`boolean2d`×2 + public `ccx()`; `boundary_intersection`/`implicitize` verified on the OLD `_ncsx` adapter — review's caller graph corrected in the ledger); zero-allowance CSX knobs honored; dead knobs removed.
2. [x] **L42** — CSX curved-UV overlap fallback: bounded 4k Phase-2 allowance + continuum chain signature ⇒ `budget_exhausted` + `boundary_topology_complete=False` (repro: 1,679 roots @33,685 cells COMPLETE → 214 @4,006 partial); zero reachability from gate geometry measured before landing.
3. [x] **L28** — `SSXOverlapRegion` per approved Option C in NEW `_ssx5_overlap.py` (self-sampled rims incl. curved-UV, degree-1 stub peeling, multi-region loops, §8 band-rule witness, reason retirement via recorded structural sites); case 12 → 1 region, `complete=True, reasons=[]`; L27 skew = clean negative control. AS-BUILT: §8's "region + separate transversal branch" sub-case is unrealizable for exact polynomial pairs (ledger note).
4. [x] **L43–L46** — c3 pair pricing /128 (case-6 charge 18,731→348), dedup precharge linear (runs at the 1024 cap), NaN chain closed (accept-if gates + NaN-safe binned dedup), closest-point aggregators carry `max_cells`/`stats` + interior heap owns its pop allowance.

**COMPLETE on `ssx5-singular-hardening` (2026-07-12 EVENING session 2; commits `34f5ba6`.. — per-item evidence in the ledger):**
1. [x] **L25** — edge-graze arc loss: `_march_to_boundary` exit commits only on a certified on-face root (strict bar = the tracer's path certificate); refused exits retarget interior and march PAST the graze. Both loss modes pinned (total strict-kill at d0≲atol·sin; silent truncation at d0≳atol·sin). Case 15 in harness ALL_CASES; 20-variant sweep clean.
2. [x] **L47** — USER DECISION: residual tier. CCX overlaps carry `certification: 'exact'|'tolerance'` (dense inversion pairing ≤ atol, monotone, transverse-FLIP guard — crossing structure never merged, brackets → strict roots); bounded fallback arms on band evidence (inward probes), typed `uncertified_overlap_span` replaces bare budget misbilling; boolean2d shared-edge merge restored. **L55 RESOLVED as a side effect** (rational case-12 twin now complete, reasons=[], 15.3k cells). Residual: woven noise-amplitude twins stay typed-partial (needs a user band-bar decision to merge).
3. [x] **L48** — param-tol pins (test-only). CORRECTION: the new conservative bound is translation-INVARIANT; the review's 7.6e-7 was the old optimistic dispatch. Both legacy consumers translation-stable, now guarded.
4. [x] **L49** — cut-face fibers surfaced (`REASON_PARAMETER_FIBER`, boundary-path parity) via the exact interior-pinch fixture; branch loss REFUTED (3 variants); found: permanent work_budget misbilling from the bounded c1 tier on positive-dim Σ lines (5,693 of 1M cells) → §11.6/L52.
5. [x] **L51** — CSX boundary-exhaustion keeps certified zeros (CCX parity); valley classification gated on complete sets.
6. [x] **L53** — USER DECISION: repair + document. ssx6 filter fixed; stale-pre-budget-fork charter in the module docstring; reviews skip it by charter.

**Next session:**
1. **L54** — the two line-scan audit agents (A1: SSX hunks; A2: CCX/CSX/closest-point/param-tol hunks + the z=const≠0 lead) were launched 2026-07-12 evening; VERIFY their findings before acting (receiving-code-review discipline), ledger the confirmed ones.
2. **L50** — investigated-open: the top-cell seed-cache over-claim is real by reading, but three two-tangency fixtures could not REACH the double-slice state (analysis + fix direction recorded in the ledger entry). Fix only with a reaching fixture.
3. **L52** consolidation batch + review doc §11 steps 5–6 (in-repo budget_contract gate, de-budget milestones — the L49-found c1-tier misbilling belongs here).
Review doc (source of truth): `docs/superpowers/plans/2026-07-12-ssx5-budget-review-and-overlap-contract.md`. Subagent policy: default to cheaper models (Opus) for scan/verify fan-outs — session limits killed two Fable finders on 2026-07-12.

**P0 — COMPLETE on `ssx5-singular-hardening` (2026-07-10; pending commit):**
1. [x] **Case 14** (`examples/ssx/bez_ssx5_case14.py`) — instrumentation found the freeze in `_find_ssx_boundary_zeros → bez_csx → _phase2_isolated_search`: an identically collapsed rational apex edge was represented as 16,385 isolated pseudo-roots, then fed quadratic dedup/splitting. CSX now emits a typed parameter fiber through a bounded point-on-surface path; SSX canonicalizes both apex fibers and traces the certified rational tangent generator. The unresolved positive-dimensional Δ complement is honestly partial.
2. [x] **Case 13** (`examples/ssx/bez_ssx5_case13.py`) — crossing-less descendants repeatedly repaid the same Φ seed search after deduplication. Ancestor-box attempt caching plus shared Φ enumeration/marching charges make the search finite. The original script terminates with one certified tangent point and explicit partial status for the unresolved complement.
3. [x] **Global soft budget** — one call-wide allowance covers SSX cells, nested CSX calls/cells, singular/C1/C3 work, output growth, and a separate finite postprocess counter. Top-level independent boundary probes have a 20k local cap; topology-critical internal cuts receive one established 100k allowance, always clamped by the shared remainder. Zero/tiny allowances preflight the superlinear distance-net build. Exhaustion returns certified partial output with `'budget_exhausted': True` and detailed usage counters.

**P1 — AWAITING USER OUTPUT-SCHEMA DECISION (no implementation started):**
4. [ ] **Case 12 / L28** (`examples/ssx/bez_ssx5_case12.py`) — true answer is a 2D surface-overlap region. The choices to present are: (A) tagged 3D boundary branch(es), (B) a structured trimmed-region entity, or (C, recommended) the structured region with paired parameter-space loops on both surfaces plus references to compatibility boundary branches. Do not claim or implement L28 until the user selects the contract.
   → 2026-07-12: full contract proposal (refined Option C with dataclass, tolerance ladder, test list), the budget-concept review with measured evidence, and 15 ranked verified findings on `5d05ddc` are in `docs/superpowers/plans/2026-07-12-ssx5-budget-review-and-overlap-contract.md`. Blocking user decisions listed in its §11 step 0.

**P2 — previously open ledger items:**
5. [ ] **L25 — edge-graze transversal arc loss** (pre-existing at `40a79a1`). Not started because priority stops at the pending P1 schema decision. Use the minimal edge-graze geometry, coverage harness, and crossing-lifecycle instrumentation (2026-06-09 playbook) before any fix.
6. [x] **L26 — rational Ψ in the tangency witness** — completed as part of P0: Δ/Φ/tangency decisions now evaluate homogeneous rational surfaces exactly rather than per-control-point dehomogenized polynomial surrogates.
7. **L27 residuals** (documented, not regressions) — only on demand.

**Also completed:** the coverage harness honors a module-level `RATIONAL` flag and rejects incomplete reference CSX slices. Cases 12–14 are registered as L28–L30. The proactive overlap/rational/no-hang review added and fixed L31–L40; see the ledger for tests and evidence.

## 3. Non-negotiable validation gates (run after EVERY task, before every commit)

```bash
.venv/bin/python examples/ssx/bez_ssx5_coverage_check.py          # exit 0: 8 cases (incl. 15 = L25 edge-graze), 100% coverage AND zero spurious singularities (enforced)
.venv/bin/python -m pytest tests/test_bez_ssx5_singular.py -q     # the singular gate (112 tests at last count)
.venv/bin/python -m pytest tests/test_bez_csx4.py tests/test_bez_ccx4.py tests/test_bez_ccx3_cases.py tests/test_bezier_common.py tests/test_bezier_curves_overlap.py tests/test_nurbs_param_tol_regression.py -q
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

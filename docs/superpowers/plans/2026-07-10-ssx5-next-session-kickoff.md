# SSX v5 — next-session kickoff (singular-cases hardening)

**Written:** 2026-07-10, from the branched interactive session. **Updated 2026-07-12.**
**Purpose:** everything a fresh session must read and obey before touching SSX code.

> **Session-start prompt (copy-paste; updated 2026-07-12 end of day — L25/L47/L48/L49/L51/L53/L54 DONE, L55 resolved, L52 slices 1-4 shipped; see §2):**
> Continue SSX v5 hardening on branch `ssx5-singular-hardening` (NOT `tiny`; no pushes).
> Start with `git status && git log --oneline -14` — expect this kickoff-docs commit
> at HEAD, sitting directly on the L52-slice commits `2322399`/`09ebc47`/`e9736d4`
> (last code commits; 13+ commits above `28a9e4d`); reconcile, never clobber.
> Read first: this kickoff §0–§4 (§2 = what shipped in BOTH 2026-07-12 sessions and
> what remains); then the ledger entries L50 (investigated-open, fixture analysis +
> fix direction recorded), L52, and L54's results (A2's two PLAUSIBLE leads feed L52) in
> `docs/superpowers/issues/2026-07-07-ssx5-singular-review-ledger.md`; then review doc
> §11 steps 5–6 in `docs/superpowers/plans/2026-07-12-ssx5-budget-review-and-overlap-contract.md`.
> Queue: 1. **L52 remainder** (4 slices SHIPPED 2026-07-12 late session: dead code,
> preflight pins, adapter status-twin unification into `_adapter_status.py`, and the
> in-repo `budget_contract` gate — see the ledger's L52 progress block): the core
> 8-way budget merge, exactness kits (reconcile envelopes EXPLICITLY), margin
> policies, predicates/restrictors, kwarg hygiene, Optional-budget threading, the
> c1 tier + §11.6 de-budget (incl. the L49-found c1-enumeration misbilling), A2's
> csx `>12` chain constant + fixed-65-grid aliasing leads.
> 2. **L50** only if a REACHING two-tangency fixture is found
> (see the ledger's recorded analysis; do not fix without it). Schema v2 is IN PLACE:
> new work reads/emits `complete`/`status.reasons`, never `budget_exhausted`, at the
> bez_ssx level. CCX overlaps now carry `certification: 'exact'|'tolerance'` (L47 user
> contract; crossing structure is never merged — on-node crossings bridged, offset
> twins with tangent-parallel points are NOT crossing-free).
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
8. [x] **L52 slices 1–4** (`2322399`/`09ebc47`/`e9736d4`): measured-dead code removed (boundary-overcut near-miss caught by the gates, recorded in the ledger); zero-allowance net-build preflights pinned (already present — stale review note); `_nccx4`/`_ncsx4` status twins unified into `_adapter_status.py`; the review probes institutionalized as the exit-coded `examples/ssx/bez_ssx5_budget_contract.py` gate (§3 list updated). Remainder scoped in the ledger's L52 progress block.

7. [x] **L54** — BOTH audit angles completed same session (Opus agents over `6f362b9..HEAD`, findings main-session-verified): A2 CONFIRMED the on-node crossing absorption in the L47 flip test → fixed (bridged flip + on-node tests) + the offset-twin-crosses-at-parallel-tangent corollary pinned; the z=const≠0 lead REFUTED with measurements; A1 NO FINDINGS (13 leads refuted) + the NaN exit-commit latent weakness hardened (accept-if). Two PLAUSIBLE leads recorded in the ledger for L52's cycle (csx `>12` chain constant on short exact spans; even-crossings-per-interval aliasing on the fixed 65-grid).

**COMPLETE on `ssx5-singular-hardening` (2026-07-12 continuation session; commits `5ca7507`..`05f0a7c`, 11 commits — per-slice evidence in the ledger's L52 progress block):**
1. [x] **L52 slices 5a–5e** — the core 8-way budget merge COMPLETE at mechanics level: NEW `mmcore/numeric/_work_budget.py` owns REASON_* vocabulary, `SoftWorkBudget` (verbatim `_SSXSoftBudget`), `BernsteinZeroBudget` (+ContextVar), `DownCounter` (ccx/csx paired locals), `LatchingSpend` (c3 `_spend`), `reconcile_reported` (adapter/closest-point clamp family), `charge_hook` (guarded-lambda pattern ×7 sites); module docstring = the explicit charge-semantics registry (4 deliberate families). Policy pieces deliberately unmerged: c1 fair-share double-ledger (slice 9), `solve_zero_dim` charge timing (documented). Milestone adversarial review over `b0a2094..485a57d`: semantics/coverage/imports lenses NO FINDINGS; 1 confirmed minor (hump-pin span assertions) → fixed.
2. [x] **L56 (new)** — ccx exactness-contract suite was stale-red since L47 (outside the gates); all 9 failures verified = the approved L47 tolerance tier by measurement; re-pinned (sharpened property: sub-tolerance offsets NEVER certify 'exact', even 5e-324); both exactness suites + `test_work_budget` joined the §3 unit batch.
3. [x] **L52 slice 11** — coverage harness enforces per-case `status.work` ≤ 2× `bez_ssx5_work_baseline.json` (exit-coded; `--update-baseline`); baselines recorded for all 8 cases.
4. [x] **L52 slice 6 (a/b/c)** — exactness kits: shared `bernstein_product_1d` in `_bezier_common` (csx broadcast shapes + ccx longdouble factors, bit-identity pinned; dead `_eval_curve_longdouble` removed); the EXPLICIT envelope reconciliation (csx affine certificate → ccx's two-term `64·n₁n₂·ε_LD` op + `8192·(n₁+n₂)·ε_f64` source; measured in-axis boundary (3.0,3.5]e-11 → (2.6,3.0]e-11; single-axis offsets unaffected — the survey's raw 64-vs-4096 framing overstated the divergence); csx exclusion prune got the §4 L1 margin (240 exact-Fraction chains: unreachable in that family; pipeline impact ×1.00 on all 8 work baselines).
5. [x] **L52 slice 12 (first item)** — `_nurbs_param_tol` `== 0.0` → `< _TINY` (measured unreachable-overflow; consistency + denormal-sweep pin).
PROCESS: the first slice-5 review workflow run corrupted the tree (agents ran `git checkout` on 11 files) — killed, restored from HEAD, relaunched with an explicit READ-ONLY-git rule (lesson in persistent memory `feedback_workflow_agents_tree_safety`).

**ALSO COMPLETE same continuation session (commits `d867ca1`..`3324a45`):**
6. [x] **L52 slice 7 (a+b)** — shared `subdivide_curve`/`subdivide_sq_dist_net`/`restrict_net_axis(_v)` + `geometry_collapsed` ×3 in `_bezier_common`; margin policies ×3 resolved by documentation (different certificate shapes); residuals noted in the ledger (centering/cartesian prep ×2, ordered-restriction backend pair, csx one-axis specials).
7. [x] **L52 slice 8** — shared `reject_unknown_kwargs` on all 3 v4 adapters (first contact caught the test suite's own silently-swallowed `rational=`); `nurbs_csx` tolerance renamed `atol=` → `tol=` (adapter-level convention; zero live by-name callers verified); threading disposition documented in the registry (remaining `is not None` guards are site policy, NOT null-object candidates); new gated `tests/test_adapter_kwargs.py`.
8. [x] **L52 slice 9a** — §11.6 de-budget: NEW structural reason `unresolved_singular_set`; c1 wiring maps evidence (shared-dry → work_budget; curve-detected truncation → SINGULAR_SET — pinch fixture AND case 14 measured; no-curve → work_budget); 20k tier KEPT with the §7.3 justifying measurement in the call-site comment (5× tier >2 min without finishing); the "no gate case reports work_budget" INVARIANT enforced in the budget_contract gate. Case 14's residual work_budget = L42-contract boundary-CSX continuum truncation (honest, knob-backed).

**ALSO COMPLETE (same continuation session, commits `e19fc51`/`37eb45d`):** the slices-6–9a milestone adversarial review (3 minors, all resolved: c1 contention guard `_c1_tier_clamped`, latent nccx-3D-fixture debris, untracked-example disposition) and **L52 slice 10** — A2's leads RESOLVED: (a) the csx `>12` chain constant CONFIRMED-and-fixed via the corner-clipped short-span fixture (lattice-cluster detection with STRICT gap-midpoint verification; never-merge invariant pinned by the valley-chain negative control); (b) 65-grid aliasing verified documented (no change without the user's L47 band-bar decision).

**ALSO COMPLETE (late continuation session, commits `1efe8c2`..`ff2a10d`):** slice 9b (`unresolved_regions` typed complement — depth dumps, abandoned queues, bail-before-queue early returns all name their 4-D boxes); **L59 SHIPPED** (USER DECISION: theorem-first curve-on-surface overlap certification — see the ledger's L59 entry for the full record: the 5d05ddc regression located by bisection, the user's rationale verbatim, the tier + SSX integration, fixtures A/B at C-quality, all six user overlap scripts tracked as real-data gates); L57 absorbed into L59; L58 (sphere segments) PARKED by user decision.

**Next session:**
1. **L60** — the near-band emptiness grind (script-3 call 2 ≈51 s; profile + direction in the ledger; pre-existing, NOT an L59 regression).
2. **L52 slice 12 remainder** — §10 efficiency items; flag zoo (`hard_exhausted`/`output_counts`); slice-7 residuals (centering/cartesian prep ×2, ordered-restriction backend pair).
3. **L50** — investigated-open: fix ONLY with a reaching two-tangency fixture (analysis + fix direction in the ledger). **L58** — parked until the user re-opens rational spheres.
4. DEFERRED BY USER mid-session: the PR-to-tiny push + the nurbs_ssx foundation doc — re-confirm with the user before pushing anything.
Review doc (source of truth): `docs/superpowers/plans/2026-07-12-ssx5-budget-review-and-overlap-contract.md`. Subagent policy: default to Opus for scan/verify fan-outs. REVIEW-AGENT RULE (paid for): every repo-Bash agent prompt carries the READ-ONLY-git instruction; commit WIP before launching agent batches; `git status` after each batch.

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
.venv/bin/python examples/ssx/bez_ssx5_coverage_check.py          # exit 0: 8 cases (incl. 15 = L25 edge-graze), 100% coverage AND zero spurious singularities AND per-case work ≤ 2× bez_ssx5_work_baseline.json (all enforced; refresh baseline deliberately via --update-baseline)
.venv/bin/python examples/ssx/bez_ssx5_budget_contract.py         # exit 0: schema-v2 contract + residual sanity + determinism, 2 runs/case (L52 / review §11.5)
.venv/bin/python -m pytest tests/test_bez_ssx5_singular.py -q     # the singular gate (113 tests at last count)
.venv/bin/python -m pytest tests/test_bez_csx4.py tests/test_bez_ccx4.py tests/test_bez_ccx3_cases.py tests/test_bezier_common.py tests/test_bezier_curves_overlap.py tests/test_nurbs_param_tol_regression.py tests/test_ccx4_exactness_contract.py tests/test_csx4_exactness_contract.py tests/test_work_budget.py tests/test_adapter_kwargs.py -q   # exactness/budget/kwarg contracts joined the batch 2026-07-12 (L56: contract suites outside the gates go stale-red silently)
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

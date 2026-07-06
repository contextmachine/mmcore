# SSX v5 — Session Handoff (2026-06-10): state, knowledge, and the singular-cases task

Written immediately before a context compaction. Everything a fresh session needs to execute the
singular-cases plan without re-deriving two days of debugging. Read this FIRST, then the plan.

---

## 0. The immediate task

Execute **`docs/superpowers/plans/2026-06-10-ssx5-singular-cases.md`** — C₁/C₂/C₃ singularity
handling for `bez_ssx` per Cheng et al. 2023 (`3592452-2.pdf`, all 17 pages read and mapped this
session; the plan's "Paper → code map" section is the distilled version).

- **The user chose subagent-driven execution** (superpowers:subagent-driven-development): fresh
  subagent per task, strict review between tasks. Both previous adversarial review rounds found
  confirmed MAJOR bugs in freshly written code in this codebase — review is not optional theater
  here. Run an adversarial verification pass (agents trying to REFUTE each finding with repro
  scripts) after the implementation tasks, like this session did twice.
- Task order: 1 → 2 → {3 → 4 → 5} with {6} and {7} independent after 2; task 8 (validation) last.
- User decisions already made (AskUserQuestion, do not re-ask): output schema = new
  `result['singularities']` list of typed `SSXSingularity` + `SSXBranch.kind` tags;
  Φ∩L solver = Bernstein hull-exclusion subdivision + Newton with DETERMINISTIC axis mid-planes
  (NOT the paper's random hyperplanes, NOT Krawczyk-first).

## 1. Repo/session state (as of handoff)

- Branch `tiny`. **All of rounds 1–2 work is UNCOMMITTED** (user never asked to commit; ask or
  leave). `git status` shows modified: `mmcore/numeric/intersection/csx/_bez_csx4.py`,
  `mmcore/numeric/intersection/ssx/_bez_ssx5.py`, `tests/test_bez_csx4.py`; new (untracked):
  `examples/ssx/bez_ssx5_coverage_check.py`, `docs/superpowers/plans/2026-06-10-ssx5-singular-cases.md`,
  this file.
- venv: `/Users/sthv/PycharmProjects/mmcore/.venv/bin/python` (Python 3.14). No `timeout` command on
  this macOS — don't use it in Bash. `/tmp` was wiped mid-session once (diagnostic scripts in
  `/tmp/ssx5_diag/` are expendable; the durable harness lives in `examples/ssx/`). `rich` had to be
  pip-reinstalled into the venv (import chain `csx/__init__ → _ncsx2 → rich`).
- `_bez_ssx6.py` untouched all session — comparison baseline only, has its own copies of everything.
- Memory files: `project_ssx5_branch_loss_fixes.md` (rounds 1–2 full record, includes review
  verdicts + accepted risks) — trust it, it was updated continuously.

## 2. Validation infrastructure (use it after every task)

- **`examples/ssx/bez_ssx5_coverage_check.py`** — the objective harness. Builds an independent
  reference cloud (S1 isoline slices × `bez_csx`), reports per-branch point counts/step stats and
  coverage. Run: `.venv/bin/python examples/ssx/bez_ssx5_coverage_check.py` (all cases) or with case
  numbers. **Current baseline: ALL 7 cases (5,6,7,8,9,10,11) at 100% coverage** within 5·atol;
  timings ≈ case5 3.5s, case6 7.6s, case7 3.1s, case8 1.3s, case9 1.8s, case10 7.2s, case11 7.7s.
  Zero isolated points on all 7. Branch endpoints land EXACTLY on domain boundaries (e.g. v=1.0).
- Legacy 4 mini-cases (planes / transversal / tangential / overlaps — control nets are in
  `tests`-style snippets inside the memory file and the plan's Task tests): 1/1/1 branches at
  ~1e-14 residual; overlaps = 4 branches (KNOWN pre-existing 4-vs-2 overlap gap, out of scope).
  The tangential legacy case exercises the Φ-tracer: 75 uniform points @ 8e-15.
- Unit suites that must stay green (59 tests):
  `pytest tests/test_bez_csx4.py tests/test_bez_ccx4.py tests/test_bez_ccx3_cases.py tests/test_bezier_common.py tests/test_bezier_curves_overlap.py -q`
  (`tests/test_csx.py` and `tests/test_nccx4.py` have PRE-EXISTING import failures — `Coons`,
  `mmcore.geom.curves` removed in c14fd3e — not ours.)

## 3. What changed this session and WHY (condensed; full detail in memory file)

### `_bez_csx4.py` (CSX — root-cause of SSX branch loss)
- REMOVED two unsound phase-2 prunes: (a) "Newton from cell center converged to a root outside the
  cell ⇒ prune cell" — basin geometry made losses look nondeterministic (de Casteljau re-centers
  domains into different basins); (b) basin-discard heuristic. Regression test:
  `test_case_19_interior_root_not_pruned_by_outside_basin`.
- Newton now **bounded to the cell**; `_fp_slack=1e-12` applied to the in-cell test.
- NEW **vector residual net** `G[i,j,k] = C[i]·wS[j,k] − S[j,k]·wC[i]` (outer product — exact, no
  degree elevation) threaded through phase 2; prune when any component's Bernstein hull excludes 0.
  Sign tests converge LINEARLY in distance vs the sign-blind |G|² net's quadratic race — this
  recovered a 10× slowdown; CSX suite now FASTER than before the fix.
- Strict positivity prune `min(F)>0`; resolution-floor prune (cell ⊂ known-root ±2·ptol box).
- **NEGATIVE RESULT (do not retry):** reporting/merging "tolerance touches" (stalled Newton with
  r<atol) or excising linearized sub-atol valleys DESTROYS sub-tolerance topology — case 11's
  near-tangent loop shattered. Tolerance semantics ≠ topology semantics. CSX must preserve distinct
  zeros connected by sub-atol valleys.

### `_bez_ssx5.py` (tracer/marchers/assembly — reliability + xyz accuracy)
- `bez_ssx(S1, S2, atol=1e-3, rational=True, max_depth=12, max_xyz_step=None)`; returns
  `{'branches','points'}` (task 1 of the plan adds `'singularities'`).
- **Tracer** `_trace_cell_by_registrations(cell, atol, h_max=None)`: marches from each crossing
  (2 direction attempts); match exit↔crossing at 4·ptol/axis (global frame = local ptol × box span)
  AND xyz ≤ 2·atol; **no match ⇒ SYNTHESIZE a BoundaryPoint from the Newton-verified exit** ("trust
  the marcher's stopping point") — never discard; interior-truncated marches kept as open fragments
  (end_point=None); failed seeds surfaced as SSXPoints (later filtered if they lie ON a branch,
  ≤4·atol to polyline). Bounce detection = MAX-over-path xyz displacement ≤ atol (endpoint-only
  deleted genuine hairpin arcs; stuv-based deleted micro-fragments like case10's final sliver to
  the v=1 corner: 4e-4 in stuv but 5.3e-3 units in xyz).
- **Marchers** (`_march_to_boundary` — NOTE returns 3-tuple `(stuv, xyz, exit_info)` now;
  `_march_intersection_curve`; `_march_phi_curve`): **xyz-reparameterized stepping** — maintain xyz
  chord target `h`, convert per-iteration via `step = clip(h / speed, min_step, max_step=0.25)`,
  `speed` from `_tangent_3d(S1, stuv, tang4)` → returns `(unit_dir, speed)`. h adapted ONLY by 3D
  turning angle (4D parameter angle REMOVED from sizing — user's requirement: param curvature is
  reparameterization-dependent noise); sagitta reject `chord·angle3/8 > sag_tol=2·atol`; NEW
  `_mid_chord_deviates(...)` — correct the stuv midpoint onto the curve (`_ssx_correct`), measure
  deviation from the chord: catches S-shaped/inflection spans invisible to endpoint tangents.
  Boundary-exit and target-arrival commits are mid-chord-verified too; on failure retarget the
  predictor at the INTERIOR HALFWAY point (h-halving alone can NEVER shrink an exit chord — the
  ray-face init cancels step length, `_ssx_correct_fixed` returns the identical exit; proven by
  review repro). Φ-marcher: deviation checks use the Φ corrector (Ψ corrector is the wrong system)
  and speed = max over BOTH surface images (S1 image stalls off the Ψ-curve). Stagnation escape:
  25 consecutive rejections → clean truncation. `h_max = max(0.05·joint-AABB-diag, 4·atol)` —
  NOT an accuracy criterion (sagitta is; measured: maxdev ≤ 2·atol at every cap level); it only
  keeps the finite-probe checks in their trust region. `max_points=400` boundary / 2000 targeted.
- **Assembly** `_assemble_fragments(..., unify_tol, h_max)`: endpoint union-find at 4·ptol/axis AND
  xyz ≤ 2·atol AND merged-cluster bbox ≤ 2·tol/axis (transitive-chain cap); duplicate-fragment
  removal by GEOMETRIC CONTAINMENT (every sample of shorter within 2·atol of longer's polyline,
  arc-length sorted — single-midpoint test deleted genuine thin-loop arcs); sliver filter
  tolerance-scaled (≤5 pts, all within 4·atol of another branch's polyline); closing-march for
  near-closed loops (gap < 10·median step, both ends interior).
- **Subdivision loop**: crossings dedup new-vs-inherited at 1·ptol/axis AND xyz ≤ atol
  (`dedup_tol = unify_tol/4`); `unify_tol = 4·ptol_global` per axis from
  `bez_surface_param_tolerance` on the FULL surfaces. Guided cuts pass exactly THROUGH crossing
  parameter values ⇒ seeds in adjacent cells are 2-pinned corner points whose marches legitimately
  bounce (through-touch corners) — expected, handled.
- **TOLERANCE LADDER (respect it everywhere, incl. new code):** destructive ops (dedup) =
  1·ptol/axis AND xyz ≤ atol; matching/unification = 4·ptol/axis box AND xyz ≤ 2·atol; a parametric
  box is NOT a metric ball — |Δstuv| ≤ 4·ptol admits xyz separations ~16·atol where derivatives are
  large. EVERY destructive tolerance test needs the xyz guard.
- Units: everything (coords, atol, deviations) is in the SAME dimensionless model unit. Do not
  write "mm" (caused user confusion once); express deviations as multiples of atol.

### Known accepted risks (reviewed, documented, NOT fixed — don't re-flag as new):
- h≤2·h_floor hatch can still commit a deviating exit chord when zero interior progress is possible.
- `_mid_chord_deviates` fails OPEN if its midpoint correction diverges.
- In-marcher `h_max=None` fallbacks (0.05·local S1 diag) differ from pipeline global (0.05·joint
  diag) — dead in production paths.
- `_pop_neighbour` picks an arbitrary fragment at >2-fragment junctions (X-junction pairing
  ambiguity) — relevant to plan Task 4; paper also terminates branches at singular points.
- CSX eff_atol-floor stall acceptance can over-report 3-where-2 in synthetic sub-1e-6 grazing
  valleys (pre-existing, resolution limit).
- Dead code in `_bez_ssx5.py`: `_csx_on_cut_face`/`_midpoint_split` (contains a broken
  `list((lambda...))` non-filter that would crash if revived), `_find_exit_registration`,
  `_choose_cut`/`_choose_multi_cut` uncalled. The PartitionCurve/IsolineRegistration machinery is
  vestigial scaffolding (the "simplified tracer" doesn't use in/out registrations).

## 4. Current `bez_ssx` pipeline (what a subagent must know before editing)

Top level: prune (AABB + sq-dist net) → GaussMapBern for both surfaces → 8 boundary CSX calls
(`_find_ssx_boundary_zeros`) → TΨ¹..⁴ nets via `minors_Tpsi_from_control_nets` (cartesian ctrl pts)
→ F_sq net + w_scale + unify/dedup tols + h_max → BFS deque of `_Cell`s. Per cell: AABB → GJK →
F_sq min-net/Lipschitz prunes → `_check_loop_free` (TΨ monotonicity first, Gauss separability
second) → if loop-free & crossings: trace, `continue` → transversality pre-check (normal angle at
crossings; **currently `not cell.crossings ⇒ is_clearly_transversal=True` — THIS is the gate plan
Task 3 fixes**, it silently loses isolated tangencies) → `_check_tangency` (Gauss-Newton witness on
Δ=Ψ∩TΨ via `_deflate.DeflatedSystem`) → True & crossings ⇒ `_deflate_tangent_cell` (Φ-tracer) →
max_depth dump → dual-surface guided subdivision (`_compute_split_plan` from 1-pinned crossings,
midpoint fallback per surface, cut-face CSX per cut × opposite piece, t-endpoint filter 1e-6 drops
re-found corner roots, crossings distributed by box containment, T/F nets de Casteljau-split,
children get `_build_cell_partitions` + `_classify_boundary_point`). After the loop:
`_assemble_fragments` → overlap branches appended → points-on-branch filter → return.
Cell state: `g1,g2 (GaussMapBern, local [0,1]²), crossings (GLOBAL stuv), box, depth, T1..T4,
F_sq, w_scale, new_crossings (drive cuts), partitions`. Coordinates: crossings/fragments GLOBAL;
marching LOCAL per cell (`_local_to_global`/`_global_to_local`); signs invariant (positive affine).

## 5. Paper essentials already extracted (don't re-read unless stuck)

- **C₁** (§4.1): Σᵢ = ∂Rᵢ/∂a × ∂Rᵢ/∂b = 0 on the curve. At C₁ points of R1: T³=T⁴=0 but T¹,T²≠0 ⇒
  4D curve REGULAR (our marcher walks through; 3D image has a cusp, 3D speed→0, our h/speed clamps
  handle it). Locate via Ψ∩Σᵢ; finite ⇒ cusp points; infinite ⇒ cusp curves.
- **C₂** (§4.2): T_Ψ=0. Deflation Δ=Ψ∩TΨ (have). Regulated Φ={Ψa,Ψb,TΨk} regular through the
  singularity; loops hit Φ ≥2× (Lemma 2; have as Φ-tracer). §5.3.2: isolated points/tiny loops —
  hyperplane L through box center reliably cuts the 1-dim Φ (not the 0-dim feature); we use
  deterministic axis mid-planes (paper admits random L misses small features, §7.1).
- **C₃** (§4.3): Theorem 3 — (T¹ or T² sign-definite) AND (T³ or T⁴ sign-definite) ⇒ 3D-injective
  box (nearly free per cell, nets already carried). Detection: square 6-var Newton
  {R1(s,t)=R2(u,v), R1(p,q)=R2(u,v)}, (s,t)≠(p,q) at >4·ptol. Paper runs C₃ AFTER tracing (§5.4);
  we post-process traced branch segment pairs (segment-segment distance < 2·atol) as seeds.
- **C₀**: Lemma 5/6 = our monotonicity certificate (already consistent, non-strict ≥ correct).
- Paper's admitted limits (§7.1): multiplicity ≥3 not handled by deflation (out of scope for us
  too); random-L misses; slow Φ tracing (ours is the fixed xyz-driven marcher).

## 6. Available primitives inventory (verified this session)

- `_deflate.py`: `bernstein_patch_derivative_s/t(P)` (nested-list 2D patches),
  `bernstein_patch_cross_same_params(P,Q)` (exact Bernstein cross product — Σ nets!),
  `build_4d_cross_patch`, `outer_dot_4d`, `minors_Tpsi_from_control_nets(P1,P2)→T1..T4` (nested
  lists; convert `np.asarray(..., float)`), `DeflatedSystem`, `gauss_newton_witness(sys, box,
  tol_f, max_iter)`, `_box_from_any`, interval helpers.
- `bern.py`: `de_casteljau_split_nd(grid, axis, t)` (trailing value dim), `bernstein_eval_nd`,
  `bernstein_partial_derivative_coeffs(grid, axis)`, `bernstein_product_conv`.
- `_bezier_common.py`: `eval_surface(S,u,v,rational)`, `eval_surface_d1` (→ pt,du,dv),
  `eval_curve`, `eval_curve_d1`, `extract_weights`, `newton_csx(..., bounds=)`.
- `mmcore.geom._nurbs_param_tol.bez_surface_param_tolerance(S, tol, rational)` → per-axis ptol.
  In deep cells local ptol is LARGE (≈0.03) — it's the resolution scale, fine.
- Outer-product net trick (no degree elevation, disjoint variables): CSX `_residual_vec_net`,
  and Ψ 4-var version specified in plan Task 2 (`psi_vector_net`).

## 7. Debugging playbook that worked here (reuse for the new tasks)

- Monkeypatch instrumentation, no source edits: subclass `m._Cell` to log creation; wrap
  `m._trace_cell_by_registrations` / `m.bez_csx` (module-global lookups make this work); replay
  single marches verbosely. Example script preserved at `/tmp/ssx5_diag/micro_frag_case10.py`
  pattern (may be wiped — the pattern is: LoggedCell + traced_logged + exact-float containment).
- Judge losses against the reference cloud, never against expectations. Distinguish "lost geometry"
  (far from any branch) from "chord deviation" (5–9·atol off a sparse polyline).
- When a check misbehaves, ask WHERE its assumptions break in xyz (not param) space first.
- After implementing: adversarial review workflow (2 reviewers by dimension + refuting verifiers
  with live repro rights) — both rounds it converted "looks correct" into 3 confirmed majors.

## 8. Plan-specific warnings (from the planning analysis)

- Task 3's gate fix: neighboring cells around a tangency will EACH confirm tangency — dedup the
  emitted `tangent_point` against `all_singularities` (unify_tol box + 2·atol xyz).
- Task 5 `_march_phi_closed`: the Φ curve passes THROUGH the tangent point — a "loop" march may cut
  through the singularity instead of around; Ψ-validity filtering usually kills the through-path,
  else flip the displacing step sign. Closed None-None fragments must survive assembly (verify the
  closing-march heuristic doesn't distort them).
- Task 6: umbrella-style surfaces have Σ zeros at patch pinch points — the cusp TEST surface
  `((2s-1)², (2s-1)³, t)` has its cusp LINE at s=0.5 (Σ₁ vanishes along a 1-dim set in (s,t)!) —
  `solve_zero_dim`'s resolution floor + max_cells budget handles it; the Ψ∩Σ intersection is still
  0-dim (one point). Expect many hull-surviving cells along s=0.5; budget, don't panic.
- Task 7: branches sharing endpoint objects are NOT self-intersections (same stuv) — the
  (s,t)≠(p,q) > 4·ptol guard covers it; verify zero C₃ on case 10.
- `is_clearly_transversal` normal-angle pre-check (sin_ang > 1e-3 at any crossing ⇒ transversal)
  stays for crossing-BEARING cells — only the crossing-less shortcut changes.

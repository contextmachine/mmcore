# SSX v5 numerical-invariance fix — session kickoff (cases 6 & 11)

Written 2026-07-20, immediately after shipping `nurbs_ssx_v5` (merged to
`tiny` at `312ce85`; fixtures committed at `778b15e`). All line numbers
below refer to `312ce85`. Diagnosis is fully probe-verified in that
session; do not re-derive it — reproduce it, then fix.

## The task

Make `bez_ssx`'s trace certification and search numerics
translation/scale-invariant, so ordinary CAD-scale inputs certify at
ordinary tolerances. Two sub-problems, one branch:

- **P1 (primary, well-scoped): the trace path certificate is
  translation/scale-variant.** It silently rejects genuinely-correct
  marched branches on off-origin / non-unit-scale models, surfacing as
  `trace_unverified` + lost curve arms. Case 6 (coords only ~±75–112!)
  loses HALF its intersection curve at atol=1e-3; case 11 fails its
  certificate even at atol=0.1 purely because of a ~3000-unit offset.
- **P2 (secondary, investigation-first): a knob-unreachable internal
  tier** pins case 11's search at exactly 24,136 cells (19,798 on
  normalized coords) under EVERY exposed knob combination, marking
  `work_budget` and fragmenting the curve at ~1.25e-6 relative
  tolerance. Identify the tier, then decide: scale it, expose it, or
  re-type its exhaustion as a distinct structural reason.

Non-goals: no changes to the NURBS adapter `_nssx5.py` (it is correct
and merely reports engine truth); no weakening of any certificate or
tolerance (see Invariants below); no Cython work.

## Read first (in order)

1. This file.
2. `docs/superpowers/plans/2026-06-10-ssx5-singular-cases-handoff.md`
   §3 (tolerance ladder), §4 (pipeline map), §7 (debugging playbook —
   the monkeypatch instrumentation pattern is how P2 gets found).
3. Memory: `project_nurbs_ssx5_adapter.md` (gate table + probe record),
   `project_ssx5_hardening_2026-07-12.md` (L48: translation-invariant
   bound lesson), `project_ssx5_branch_loss_fixes.md` (negative result:
   never merge tolerance-touches).

Branch discipline: dedicated branch off `tiny` (suggest
`ssx5-invariance`), NOT tiny. Env: `.venv/bin/python` (3.14); macOS has
NO `timeout` command. Engine file: `mmcore/numeric/intersection/ssx/_bez_ssx5.py`.

## Probe evidence (measured 2026-07-20; the authority)

Case 6 = rational arc surface (deg (2,1), 9×2 net, w∈[0.707,1], 4
u-spans, AABB [-30,113]²×[±18.7]) × plane z=1 (deg (1,1), z≡1). True
SSI: ONE curve in z=1, mirror-symmetric about x=y, from [4.37,75]
through [5.47,5.47] to [75,4.37].

| Case 6 configuration | Result |
|---|---|
| original, atol=1e-3 | half-curve (116 pts) + two 3-pt stubs; `trace_unverified`; 2,340 cells |
| original, atol=1e-2 | still half + stub |
| original, atol=0.1 | ONE clean branch, complete=True, reasons=[] |
| recentered (center≈[19,19,0]), atol=1e-3 | arm traced but 2 branches + `trace_unverified` |
| **normalized /100, atol=1e-5 (== world 1e-3)** | **complete=True, reasons=[], ONE branch, 1,434 cells** |

Case 11 = pair at coords x∈[2372,3195] (extent ~823), AABB center
≈[2784,1652,120].

| Case 11 configuration | Result |
|---|---|
| original, atol=1e-3 | 4 fragments at x∈[2752,3046]; whole arm x∈[2635,2688] lost; `trace_unverified`+`work_budget`; 24,136 cells |
| ALL knobs raised (max_cells=5M/10M, max_csx_calls=100k/200k, csx_max_cells=1M, boundary_csx_max_cells=200k/400k, csx_max_results=4096, max_depth=20) | **byte-identical**, 24,136 cells |
| recentered only, atol=1e-3 | arm still lost, same reasons |
| **recentered only, atol=0.1** | **complete=True, reasons=[], 1 branch** (original coords at 0.1 kept `trace_unverified` → certificate failure tracks the OFFSET) |
| normalized /400, atol=1e-3 (world 0.4) | complete=True, reasons=[], 1 branch |
| normalized /400, atol=2.5e-6 (== world 1e-3) | arm traced, certificate clean, but `work_budget` + 2 open fragments; **19,798 cells — identical with raised tiers** |

cell_counts fingerprint at the case-11 fixed point: `{'precompute': 64,
'csx': 7434, 'ssx': 98, 'branch_trace': 4897, 'branch_trace_verify':
2452, 'c1': 232, 'c3': 8959}`.

## Code anchors (verified at 312ce85)

- **The single `trace_unverified` mark site**: `_bez_ssx5.py:4682`,
  inside the tracer's strict-path verification loop `:4667-4683` — for
  every path vertex it recomputes `‖S1(s,t)−S2(u,v)‖` **in world
  coordinates** and requires `_vres <= strict_root_tol` (accept-if form,
  L45). Charged as `branch_trace_verify`.
- **The allowance**: `_strict_ssx_root_tol` `:1958` — docstring CLAIMS
  "translation-invariant", and the allowance does scale with the model
  **extent** (`diag`)... but the *residual arithmetic* it budgets for is
  performed in world coordinates, whose roundoff scales with coordinate
  **magnitude** (offset included). Extent-scaled budget vs
  magnitude-scaled noise = the translation-variance. (L48 déjà vu: a
  bound believed translation-invariant, misjudged at review.)
- **Fixed absolute corrector tolerances**: `_ssx_correct(...,
  tol=1e-14)` `:1894` and `_ssx_correct_fixed(..., tol=1e-14)` `:2508`
  — at coords ~100 the achievable ‖Ψ‖ floor is already ≥ ~1e-13, so the
  corrector can stall at its max_iter without "converging"; at unit
  scale it converges. Prime suspect for case 6's one-sided half-loss
  (the two mirror halves differ only in parameterization/arithmetic).
  Other fixed 1e-14s at `:401,408,1886,2264,2271,2549,3235` — audit
  each: step-size guards are fine, residual thresholds are not.
- **Fix-pattern precedent**: CCX's common-origin machinery in
  `ccx/_bez_ccx4.py` — `_ccx_exactness_context`,
  `_center_curve_homogeneous_for_exactness`,
  `_eval_curve_scaled_components` (center at a common origin, normalize
  per-axis scales, THEN apply strict roundoff envelopes). Also the
  historical `_bern_homog.pyx` `float t` → `double t` fix (3e-3 errors
  at large coordinates) — this codebase has walked this road.

## Design decision to make FIRST (brainstorm, don't default)

Two viable shapes for P1 — decide explicitly with the user:

a. **Per-certificate invariance**: center/scale inside each strict
   check (the CCX pattern). Localized, lower risk, but every scale-
   sensitive site must be found individually (the 1e-14 correctors, the
   path certificate, possibly CSX-side gates the tracer consumes).
b. **Whole-call normalization preamble**: at `bez_ssx` entry, jointly
   center+scale S1,S2 (and atol) to a canonical frame; un-map xyz on
   exit (params are invariant; weights unchanged; singularity/region
   payloads need the same un-map). One transform kills ALL absolute-
   epsilon variance at once — this is effectively what commercial
   kernels do — but it changes internal atol semantics and every xyz-
   carrying output path must be un-mapped exactly once (branches,
   points, singularities incl. samples, region certification residuals
   are atol-relative so they survive).

Evidence hint: recentering alone did NOT fully fix case 6 at 1e-3
(still fragmented+unverified) while recenter+rescale did — so scale
matters, not just offset; option (b) or (a)-with-scaling, not centering
alone.

## P2 instrumentation plan (only after P1 lands)

Repro: normalized case 11 at atol=2.5e-6 pins at 19,798 cells. Find the
denying tier by monkeypatching (handoff §7 pattern — no source edits):
wrap `SoftWorkBudget.charge_cells`/`charge_csx_call` and
`bernstein_zero_budget` to log source + denial site; candidates: nested
1-D boundary solves (`_bern_zero_1d.bernstein_zero_budget`), CCX
phase-1 `boundary_root_cap=min(max_results,128)`, marcher
`max_points=400/2000` caps (one fragment had exactly 200 pts —
suspicious), the `_run_csx` per-call allowance chain. Then decide
scale/expose/re-type with the user. Note the reason-taxonomy issue: a
tier that budgets cannot reach should not bill `work_budget` (consumers
read it as "raise a knob") — candidates for a distinct typed reason.

## Acceptance gates

1. **Case 6 (P1 core)**: `nurbs_ssx(s1_6, s2_6, atol=1e-3)` on ORIGINAL
   coordinates → `complete=True`, `reasons=[]`, full curve (match the
   normalized-run topology: one branch [4.37,75,1]↔[75,4.37,1]).
   Harness: `.venv/bin/python examples/ssx/nurbs_ssx5_coverage_check.py 6`
   → OK at 100% (non-vacuous).
2. **Case 11 certificate half (P1)**: original coords at atol=0.1 →
   `complete=True, reasons=[]` (today only the recentered run achieves
   this). At atol=1e-3: `trace_unverified` gone from reasons
   (`work_budget` may remain until P2).
3. **Full harness**: `nurbs_ssx5_coverage_check.py` (all 10) → no
   regressions on the 8 OK rows; target 9 OK + case 11 improved after
   P1; 10 OK after P2. Update the harness `CASE_NOTES` for 6/11 as they
   flip.
4. **Regression floor** (engine is being touched — this is the guard):
   `tests/test_bez_ssx5_singular.py` (115), `tests/test_nssx5.py` (41),
   `tests/test_bez_csx4.py tests/test_bez_ccx4.py
   tests/test_bez_ccx3_cases.py tests/test_bezier_common.py
   tests/test_bezier_curves_overlap.py` (95) — all green. The bez-level
   coverage harness `examples/ssx/bez_ssx5_coverage_check.py` must stay
   at 100% on its cases.
5. **Invariance property test** (add it): for a fixed seed set of
   surface pairs, `bez_ssx(S1,S2,atol)` and
   `bez_ssx((S1−c)/k,(S2−c)/k, atol/k)` must produce equivalent
   topology (branch count/kinds/closure, reasons set) for translations
   c up to ~1e4 and scales k in [1e-2, 1e3]. This is the durable
   regression guard for the whole defect class.

## Invariants / warnings

- **Never fix by loosening.** The certificates exist to kill phantom
  topology; the fix is invariance (centering/scaling the arithmetic),
  not bigger tolerances. Memory: tolerance-touches negative result
  (case-11-CSX loop shattering); L48 (a "translation-invariant" claim
  was already misjudged once — measure, don't assert).
- Adversarial review after implementation is house-mandatory (two prior
  rounds each converted "looks correct" into confirmed majors).
- Fixtures: `examples/ssx/nurbs_nurbs_intersection_{5,6,8,10,11}.pkl`
  are committed; 1/2/4/7/9 exist untracked in the working tree.
- `test_nssx5.py` pins CURRENT engine truth in two places that may
  legitimately flip when engine behavior improves (marked in-file):
  the case-10 fixture expectation and the seam-tangency typed-partial
  contract — update them WITH the engine change if affected, per their
  comments. Tangential semantics should NOT change in this task; if
  they do, stop and re-scope.
- Stray debug prints leak from bez/csx internals (`17 30`, `5 6`,
  `215 218` on harness cases 1-2) — cosmetic; fine to remove while in
  there, as its own commit.

## Repro snippets

```python
import pickle, numpy as np
from mmcore.geom._nurbs_eval import NURBSSurfaceTuple
from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx

def load(n):
    with open(f'examples/ssx/nurbs_nurbs_intersection_{n}.pkl','rb') as f:
        return pickle.load(f)[0]

def xform(s, shift, scale=1.0):
    return NURBSSurfaceTuple(order_u=s.order_u, order_v=s.order_v,
                             knot_u=s.knot_u, knot_v=s.knot_v,
                             control_points=(s.control_points - shift)/scale,
                             weights=s.weights)

s1, s2 = load(6)
r = nurbs_ssx(s1, s2, atol=1e-3)          # BUG: half curve, trace_unverified
# center = AABB midpoint; normalized /100 @ atol=1e-5 -> clean (see tables)
```

For engine-level work, reproduce per-pair with `bez_ssx` directly
(decompose via `decompose_surface`, homogenize via `to_homogeneous_2d`)
so the adapter is out of the loop.

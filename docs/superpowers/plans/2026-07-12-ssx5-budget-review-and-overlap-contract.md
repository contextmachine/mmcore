# SSX v5 — budget concept review, overlap output contract (L28/L25), next steps

**Written:** 2026-07-12, independent review session over branch `ssx5-singular-hardening`
(scope = the one commit vs `tiny`: `5d05ddc` "fix(ssx5): harden singular and rational
intersections", +7,857/−573 across 27 files).
**Inputs:** 11-angle adversarial review of the diff (findings in §10), two purpose-built
empirical probes (§5), the validation gates re-run at HEAD, a survey of how other kernels
bound computation (§4), and the reference papers.

---

## 1. Verdict (TL;DR)

1. **The budget concept is sound and should stay — but not in its current public shape.**
   Every serious kernel bounds work; what varies is honesty. mmcore's variant
   (deterministic work units, one shared call-wide ledger, certified-partial output,
   usage counters) is the *most* honest variant in the taxonomy (§4). It is not a
   CAD-specific invention — it is the Z3-`rlimit` / SciPy-`maxiter+nfev` contract applied
   to geometry, which CAD kernels have historically avoided by hiding their caps.
2. **Measured reliability is excellent** (§5): all gates green at HEAD; on every regular
   coverage case the default budget has ≥4.4× headroom; a 49-run tiny-budget fuzz produced
   **0 crashes, 0 contract violations**, only verified geometry (worst residual ≈ 0·atol
   even under hard truncation), correct flags, and bit-identical determinism under binding
   budgets. The soft-stop design does what it promises.
3. **The real defect is semantic, not mechanical: `budget_exhausted` conflates three
   unrelated causes** — hard budget exhaustion, structural partiality (no overlap schema,
   unresolved fibers/multiplicity), and output/postprocess caps. Measured proof: case 12
   converges at 4,702 cells (1.9% of allowance), identically at any budget, yet reports
   `budget_exhausted=True`. A consumer who reads that name will raise budgets and change
   nothing. Rename before anything public consumes it (§6).
4. **The user should never manage budgets — and today they don't have to.** All knobs have
   defaults; the fields are read-only diagnostics. Keep it that way: the roadmap in §7
   drives the *hard*-exhaustion rate to zero on well-posed input (classification first,
   budget as backstop), which is exactly the "improve early exits and case classification"
   direction — the two are complements, not alternatives. Removing the backstop would
   reintroduce the case-13/14 freezes; fixed-precision tangency/multiplicity decisions
   cannot be classified exhaustively (L31's own honest-partial limit).
5. **L28 (2-D overlap region): recommend contract Option C** — a structured
   `SSXOverlapRegion` that *references* rim branches and carries paired parameter-space
   loops + certification (§8). Additive to the result schema; lands best together with the
   §6 rename.
6. **L25 is sequencing-coupled to L28, not schema-coupled** (§9): an edge-graze arc is
   transversal (measure-zero), so it does not need the region type — but its fix lives in
   the same boundary-claim/rim machinery L28 will harden, and the overlap-vs-transversal
   tolerance ladder should be decided in the L28 spec so L25 has a rule to lean on.
7. Review outcome (§10): the SSX core's exhaustion semantics hold under fuzz, **but the
   review found real defects at the edges**: (a) two budget *mis-pricing* bugs that make
   exhaustion fire spuriously at ordinary scale (C3 broadphase pair-tests priced as
   cells; point-dedup precharged O(n²) for an O(n) algorithm — the dedup is then
   *skipped*); (b) a crash-regression class in the NURBS adapters, whose new default
   **raises `RuntimeError` on any incomplete sub-result** — and every production caller
   (`boolean2d`, `implicitize`, ssx `boundary_intersection`) uses the default; (c) CCX
   overlap semantics silently narrowed from tolerance-based to exact-affine certification
   (near-coincident pairs now lose all geometry); (d) a NaN acceptance→crash chain; plus
   the consolidation debt (8 hand-rolled budget accountings with diverging semantics,
   duplicated exactness kits, dead knobs). Ranked list with verdicts in §10.

---

## 2. What this branch actually adds

One commit, four thrusts (ledger items in parentheses):

| Thrust | Content |
|---|---|
| No-hang fixes | Case 13 (L29): ancestor-box attempt caching + shared Φ charges. Case 14 (L30): collapsed rational apex edge typed as a **parameter fiber** in CSX (16,385 pseudo-roots → one typed positive-dimensional entity) |
| Global soft budget | `_SSXSoftBudget` threaded through SSX + nested CSX + singular/C1/C3 passes + postprocess; result fields `budget_exhausted` / `budget_usage` |
| Rational hygiene | L26: tangency witness / Δ refiner / Φ selection / singular trace on true homogeneous quotients |
| Audit batch | L31–L40: multiplicity valleys, zero-allowance preflight, NURBS-adapter aggregate budgets, postprocess charging, closest-point continuation budgets, translation-invariant exactness certificates, overlap-subset dedup, 20k/100k cap policy, CSX depth 64, no retry double-spend |

Main files: `_bez_ssx5.py` (+2,781), `_bez_ccx4.py`/`_bez_csx4.py` (+659 each),
`_ssx5_singular.py` (+401), adapters `_nccx4.py`/`_ncsx4.py`, `_bern_zero_1d.py`,
`_bez_closest_point.py`, `_nurbs_param_tol.py`, ~3.3k test lines.

## 3. The budget mechanism as built (inventory)

`_SSXSoftBudget` (`_bez_ssx5.py:54`): `max_cells=250_000`, `max_csx_calls=10_000`,
`max_output_items=1_024`, `max_postprocess_work=None→max_cells`; counters + two flags
(`exhausted`, `incomplete`) + `postprocess_exhausted`.

**Charge classes** (from `budget_usage.cell_counts` at HEAD): `precompute` (distance-net
preflight, charged *before* the superlinear build — L32), `csx` (nested CSX actual usage,
charged after each child from its reported `cells_processed`), `ssx` (subdivision pops),
`branch_trace`/`branch_trace_verify`, `singular`/`phi`/`singular_trace`/`singular_dimension`,
`c1`, `c3`, `fiber_promotion`. Postprocess work is a **separate** counter so a hard-stopped
search can still assemble certified fragments (`_assembly_spend`, `_bez_ssx5.py:4839`).

**Tier policy:** independent top-boundary CSX probes get `min(20k, remaining)` and their
truncation is *soft* (face discarded, run marked incomplete, other faces continue);
topology-critical internal cuts get one established `min(100k, remaining)` and truncation
is a *hard* stop (a truncated root set never drives topology decisions —
`_run_csx`, `_bez_ssx5.py:6309`). The C1 pass has its own fixed
`min(20_000, remaining)` tier (`_bez_ssx5.py:7694`) — see §7.3.

**Exhaustion semantics by stage** (all verified by reading + fuzz):
search = stop scheduling, keep certified output; CSX child = discard-or-stop per tier;
assembly = stop optional scans conservatively; **verification filter = omit any branch it
lacks allowance to verify** (`_bez_ssx5.py:5474` — the sound direction: nothing unverified
ships); `result_fields()` publishes `budget_exhausted = exhausted OR incomplete` plus the
counters. No budget path raises; the only `raise` sites in `_bez_ssx5.py` are non-budget
internal errors. The v6 draft (`_bez_ssx6.py:_require_complete_csx_result`) deliberately
raises on unsafe CSX input — an integration-boundary assertion, pinned by
`tests/test_bez_ssx6_contract.py`, not a user-facing behavior.

**Flag zoo (defect):** three internal booleans, four published views
(`budget_exhausted`, `hard_exhausted`, `incomplete`, `postprocess_exhausted`);
`hard_exhausted` and `output_counts` have **zero readers** in the entire repo (verified by
grep); `charge_postprocess` denial sets all three flags in lockstep by convention only.
`solve_zero_dim`'s new `stop_when`/`stop_requested` machinery has zero callers.

## 4. Is a budget concept CAD-specific? Survey of other engines

No — bounding work is universal; **what differs is whether the bound is admitted.**
Four regimes:

| Regime | Representatives | User-visible? | On hitting the bound |
|---|---|---|---|
| **Hidden internal caps** | SISL (recursive subdivision + iteration, caps and recursion limits internal), OCCT's internal walkers/iteration ceilings, essentially every marching/subdivision intersector of the 1990s lineage | No | Silent partial result or silent success — the classic "boolean returned garbage with no warning" failure class |
| **Cooperative cancellation + transactional state** | Parasolid: registered interrupt → `PK_SESSION_abort`, model recovered via (partitioned) rollback; ACIS: `outcome` objects + `CheckOutcome` progress/interrupt interface; OCCT: `Message_ProgressRange` / `UserBreak()` + BOPAlgo error/**warning alert reports** on completion | Partly (cancel + status) | Clean failure; model state restored; alerts enumerate what went wrong |
| **Declared deterministic work metering** | Z3 `rlimit` (deterministic counter → result `unknown` + reason, reproducible across machines/load); every numerical library's `maxiter` + `nfev` + `status` (SciPy, IPOPT, …); gas/fuel metering in VMs as the general pattern | **Yes** | Honest status: "here is what I proved, here is what I spent, the answer is incomplete" |
| **Exact computation** | CGAL | N/A | No budget — unbounded time instead; a different contract, not a free lunch |

mmcore's `_SSXSoftBudget` is squarely regime 3 — with two properties even Z3 doesn't
promise: **certified partial output** (everything returned is residual-verified; the fuzz
in §5 confirms it) and **cross-subsystem sharing** (SSX + nested CSX + singular passes
spend one ledger; the pre-L33 per-span reset bug is exactly what regime-1 engines never
notice). The genuinely uncommon part relative to commercial kernels is *exposing usage
counters in the result* — which numerical computing has done for fifty years (`nfev`),
and which is what makes §7.4's CI regression tracking possible at all.

Two implications for us:
- **We are not out on a limb.** OCCT already ships "the operation finished with warnings,
  here is the alert list" — our `complete/status` proposal (§6) is that, made precise.
- **We should adopt the one thing regimes 2 have that we lack:** a crisp separation
  between *the answer is partial* (status) and *why* (reasons). That is §6.

Sources: [Parasolid error handling / rollback](http://www.q-solid.com/Parasolid_Docs/chapters/fd_chap.60.html),
[Parasolid session support](http://www.q-solid.com/Parasolid_Docs/chapters/fd_chap.59.html),
[OCCT BOPAlgo_Options (progress, alerts)](https://dev.opencascade.org/doc/refman/html/class_b_o_p_algo___options.html),
[OCCT BOPAlgo_Algo](https://dev.opencascade.org/doc/occt-7.6.0/refman/html/class_b_o_p_algo___algo.html),
[aborting long OCCT computations](https://incoherency.co.uk/blog/stories/freecad-abort-long-running-operations.html),
[ACIS outcome/progress/interrupt](https://blog.spatial.com/3d-acis/outcome-checking-logging-and-progress-reporting-3d-acis),
[Z3 rlimit issue #56](https://github.com/Z3Prover/z3/issues/56),
[Z3 rlimit reproducibility discussion #611](https://github.com/Z3Prover/z3/issues/611),
[F* on Z3 rlimit semantics](https://fstar-lang.org/tutorial/book/under_the_hood/uth_smt.html),
[SISL / Dokken intersection lineage](https://www.sintef.no/projectweb/computational-geometry/intersections/).

## 5. Measured reliability at HEAD

**Gates:** `bez_ssx5_coverage_check.py` exit 0 (7 cases, 100% coverage, zero spurious
singularities); `test_bez_ssx5_singular.py` + `test_bez_ssx6_contract.py` — 91 passed.

**Probe 1 — default budgets, cases 5–14** (usage of `max_cells=250_000`):

| case | t, s | cells | % | csx_calls | flags | note |
|---|---|---|---|---|---|---|
| 5 | 4.7 | 21,076 | 8.4% | 155 | complete | |
| 6 | 9.9 | 56,552 | 22.6% | 292 | complete | worst regular case → headroom 4.4× |
| 7 | 4.5 | 21,804 | 8.7% | 216 | complete | |
| 8 | 2.3 | 15,755 | 6.3% | 50 | complete | |
| 9 | 2.2 | 7,824 | 3.1% | 51 | complete | |
| 10 | 9.1 | 42,585 | 17.0% | 261 | complete | |
| 11 | 11.7 | 49,913 | 20.0% | 263 | complete | the L38/L40 regression case |
| 12 | 0.9 | 4,702 | **1.9%** | 8 | `budget_exhausted=True`, hard=False | **structural**: no overlap schema (L28) — identical at 5k/20k/60k budget |
| 13 | 9.8 | 31,480 | 12.6% | 212 | `budget_exhausted=True`, hard=False | finds its tangent_point; honest partial for the near-tangent complement |
| 14 | 64.5 | 76,286 | 30.5% | 8 | `budget_exhausted=True`, hard=False | finds branch + cusp_curve + cusp; `c1` charge = 20,000 exactly — its tier saturated (§7.3) |

Branch-sample residuals (points evaluated on both surfaces): ≈0·atol in every case.

**Probe 2 — exhaustion fuzz** (cases {5,7,11,12,13,14} × `max_cells` {0,1,100,1k,5k,20k,60k}
+ tiny `max_csx_calls`/`max_output_items`/`max_postprocess_work` axes + repeat-call digests):
**49 runs, 0 crashes, 0 contract violations.** Specifically: schema always present; no
NaN/inf; stuv always in [0,1]⁴; worst branch residual 0.00·atol in *every* run including
hard-truncated ones; `hard_exhausted ⇒ budget_exhausted` held; tiny budgets stop in
≤0.2 s (the no-hang promise); results under a binding budget are **bit-identical across
repeated calls** (cases 7/13 @5k, 12 @60k).

Two behaviors worth naming:
- **Flags are conservative, in the sound direction:** at `max_cells=20k` cases 5/7 ship
  the full correct geometry yet still flag `budget_exhausted=True` (a sub-pass was denied;
  completeness is no longer *proven*). A `True` flag with complete output is possible;
  a `False` flag with incomplete output was never observed.
- **Crash exposure in `bez_ssx` v5 itself ≈ 0:** v5 never raises on budget paths; the
  fail-fast `RuntimeError` lives in the un-integrated v6 draft as a deliberate contract.
  **However**, the NURBS *adapters* (`_nccx4`/`_ncsx4`) now raise by default on any
  incomplete sub-result, and all production callers use the default — that is the one
  real crash-regression class this commit ships (§10, findings 1–2).

## 6. The naming defect and the schema fix (do this before public integration)

Today `budget_exhausted=True` means any of: (a) the work ledger ran dry
(`hard_exhausted`), (b) the result is structurally partial though the ledger has 97%+
remaining (case 12 — missing L28 schema; case 13 — unresolved positive-dimensional
complement; boundary fibers), (c) an output/postprocess cap tripped. Three different
consumer reactions are required (raise budget / wait for schema / raise cap), and the
name steers all three to the wrong one. `budget_usage` partially disambiguates
(`hard_exhausted` vs `incomplete`) but nothing reads it, and the name still frames a
completeness question as a resource question — which is precisely the "shifting budget
responsibility onto the user" smell.

**Proposal (schema v2):**

```python
result = {
  'branches': [...], 'points': [...], 'singularities': [...],
  'overlap_regions': [...],          # new, L28 (§8)
  'complete': bool,                  # the one bit consumers act on
  'status': {
     'reasons': [                    # empty iff complete
        'work_budget',               # hard ledger exhaustion — raising budget can help
        'output_cap', 'postprocess_cap',
        'parameter_fiber',           # structural — raising budget cannot help
        'overlap_region_unsupported',# retired by L28
        'unresolved_tangential_zone','unresolved_multiplicity',
     ],
     'work': { ...current budget_usage counters, incl. cell_counts... },
  },
}
```

- Nothing outside tests/examples consumes `budget_exhausted`/`budget_usage` yet (the
  public `ssx()` does not route through `_bez_ssx5` — verified by grep), so the rename is
  cheap **now** and expensive after BRep integration.
- Keep the kwargs (`max_cells`, …) as expert knobs with the current defaults. The user
  contract becomes: *read `complete` (and `reasons` if you care why); never tune knobs to
  get correctness.*
- Fold the flag zoo into it: one internal status object; delete the unread
  `hard_exhausted`/`output_counts` published fields (or move under `work`); make
  `reasons` the single source of truth the internal `mark_incomplete()` sites feed with a
  reason string instead of a bare boolean.
- Update `_bez_ssx6._require_complete_csx_result` + its contract test and the coverage
  harness reader in the same change.

## 7. Keep budgets at all? (the user's alternative: better exits + classification)

The measured record says the two are complements with a clear division of labor:

1. **Classification does the correctness work.** Cases 13/14 terminate *because* the
   commit classified their degeneracies (fiber typing, attempt caching) — not because a
   budget truncates them. Regular cases never get near the ledger (§5). This is the
   direction to keep investing (L28 next — §8).
2. **The budget does the honesty work.** Fixed-precision classification is provably
   incompletable: L31 keeps degrees beyond its Decimal fallback "explicitly partial";
   the scale envelope degrades at ×300; sub-2e-4-eps trap sheets sit below the float GN
   stall bar (documented in `_delta_float_gn`). For inputs outside every classifier, the
   choice is: hang (pre-branch behavior), lie (regime-1 engines, §4), or bound-and-report.
   Bound-and-report is correct.
3. **Drive the firing rate to zero — make hard exhaustion a monitored never-event:**
   - **L28 region schema** retires the biggest *structural* flag class (case 12).
   - **Type the case-13 complement**: emit the unresolved Δ-complement as a typed
     diagnostic entity (its 4-D AABB + reason) instead of a bare flag, so "partial"
     always names *what* is unresolved. Same pattern as `parameter_fibers`.
   - **Audit fixed tiers against the shared remainder** — the C1 pass's
     `min(20_000, remaining)` (`_bez_ssx5.py:7694`) saturates on case 14 (charge exactly
     20,000 with ~174k shared cells unspent). This is the same disease L38/L40 just fixed
     for internal CSX cuts; either give C1 the established-allowance treatment or justify
     the tier in a comment with a measurement.
   - **Unit-pricing audit.** A budget is only as honest as its exchange rates: §10 found
     C3 AABB pair-tests (~ns of numpy) charged 1:1 like subdivision cells (~ms), and a
     postprocess precharge quoting the pre-rewrite O(n²) cost for an O(n) algorithm.
     Rule to adopt: *charge units must be proportional to real cost* (the `precompute`
     pair-coefficients/128 convention is the template); any `charge(N²)` on a linear
     algorithm is a bug.
   - **CI observability**: the gates should record `status.work.cells_processed` per case
     and fail on >2× drift — headroom regressions become visible the day they happen, not
     the day a user's model freezes. (This is what the counters are *for*; it is also the
     argument for keeping them in the result.)
   - **One budget implementation.** The review found the same accounting hand-rolled
     8 times with already-divergent charge semantics (§10). Semantic drift across copies
     is how a sound design decays; consolidate into one shared class/module.

## 8. L28 — the overlap-region output contract (decision needed)

**Problem.** Case 12 (`examples/ssx/bez_ssx5_case12.py`): two coplanar bilinear patches
sharing an edge + a corner, interiors overlapping in a 2-D region — Cheng et al. Fig. 8,
the C2 sub-case #(Δ_B)=∞, 2-dimensional. The branch/point schema cannot express it; at
HEAD the case deterministically returns its 2 rim curves as `overlap` branches +
`budget_exhausted=True` (structural). Neither reference paper prescribes an output
structure (Krishnan–Manocha assume generic position; Cheng et al. classify but do not
represent), so this is genuinely our API decision. Industry practice for coincident
regions (STEP/Parasolid "same-domain" faces) is: region = boundary loops with **pcurves
in both parameter spaces + shared 3-D edges** — i.e., exactly option C below.

**In-repo precedents to stay consistent with** (the dimension ladder):

| Solver | 0-dim image | positive-dim preimage of a point | positive-dim image |
|---|---|---|---|
| CCX | `isolated` t-pairs | — | `overlaps`: paired t-intervals |
| CSX | `isolated` (t,u,v) | `parameter_fibers` (collapsed curve → fiber over one surface point) | `overlaps`: certified {t_range, u_range, v_range} claims |
| SSX | `points` | (fibers propagated from CSX) | **today:** 2-pt `overlap` chords; **L28:** `overlap_regions` |
| closest-point | `min` points | `degenerate_curve` rings | `degenerate_surface` patches |

**Options** (from the kickoff, refined):

- **A — tagged 3-D boundary branches only.** Cheapest; no region identity; every consumer
  (booleans, trimming) must re-derive loop topology and pairing itself. Rejected: it
  reproduces the L27 corrupt-pairing class downstream in every consumer.
- **B — standalone trimmed-region entity** with materialized uv loops on both surfaces.
  Self-contained, but *duplicates* rim geometry that also ships as branches — two sources
  of truth for the same curves, the exact drift the both-guards conventions exist to kill.
- **C — structured region referencing rim branches (RECOMMENDED, refined):**

```python
@dataclass
class SSXOverlapRegion:
    """C2 positive-dimensional component: S1 ≡ S2 (within atol) over a 2-D region.
    Cheng et al. Fig. 8, #(Δ_B)=∞ / 2-dimensional (full or partial overlap)."""
    boundary: list[list[tuple[int, bool]]]
    #   one inner list per closed rim loop; entries (branch_index, reversed)
    #   reference result['branches'] (kind='overlap' rim curves), ordered head-to-tail;
    #   loop 0 = outer, others = holes (islands where the surfaces depart).
    uv1_loops: list[NDArray]   # (N_i, 2) closed polylines on S1 — materialized from the
    uv2_loops: list[NDArray]   # referenced branches' stuv columns; sample-synchronized:
    #   uv1_loops[i][k] and uv2_loops[i][k] are preimages of the same 3-D point.
    normal_agreement: int      # +1 aligned normals over the region, −1 opposed
    #   (booleans/trimming need it; constant over a connected coincidence region)
    interior_stuv: NDArray     # (4,) certified interior witness (point-in-region seed)
    certification: dict        # {'boundary_resid_max', 'interior_resid', 'n_samples'} in atol units
```

  Semantics and rules:
  - `result['overlap_regions']` is additive; `branches`/`points`/`singularities`
    unchanged → all existing consumers keep working (and option A is a projection of C).
  - Rim branches stay `kind='overlap'`, but get **properly sampled** (the current 2-point
    chords are an L27 documented residual; curved genuine overlaps need real sampling —
    fold that upgrade into L28 since the region certification needs the samples anyway).
  - Rim loop elements are overlap branches (an edge-of-S1 lying on S2 *is* an intersection
    curve — CSX already certifies these claims post-L27); loops may also traverse
    junction `tangent_point` singularities (L27's structural junctions) — junction
    vertices need no new type.
  - Orientation: loops ordered so the region is on the left in S1's (u,v); `uv2` loop
    orientation then encodes `normal_agreement` redundantly (assert-consistency).
  - **Tolerance ladder (also the L25 hinge, §9):** a coincidence band of width < atol in
    the transverse direction stays a curve (`overlap` branch, no region); a 2-D region
    requires an interior witness at ≥ 4·ptol from every rim loop with residual ≤ atol.
    Destructive dedup near rims keeps the established both-guards ladder.
  - Sub-cases the spec must pin with tests: partial overlap (case 12: expect **1 region,
    rim = shared-edge branch + interior rim branches, plus the existing junction
    semantics**); full containment (one loop entirely from one surface's domain-edge
    images); identical patches (region = whole domain, 4 domain-edge rim pieces);
    L27's skew fixture as a **negative control** (curve-only overlap ⇒ 0 regions);
    rational twin of case 12; region + separate transversal branch elsewhere.
  - CSX/CCX stay as they are (their `overlaps` are the 1-D analogue and already carry
    certified endpoints); SSX's region assembler consumes CSX overlap claims + traced
    rim fragments.

**Where it lands:** assembly stage of `_bez_ssx5.py` (a new `_ssx5_overlap.py` module for
loop assembly + certification), after the L27 claim-verification machinery. Schema v2
(§6) first, so the region ships with `reasons=[]` on case 12 rather than a budget flag.

## 9. L25 (edge-graze transversal arc loss) — relationship to the overlap decision

Known facts: pre-existing at `40a79a1`, found by review lens [A m6]; the detailed
evidence expired with the session task dir; the kickoff prescribes re-pinning with the
minimal edge-graze geometry + coverage harness + crossing-lifecycle instrumentation
(2026-06-09 playbook) before any fix.

Assessment: an edge-graze is a **transversal** arc tangent to a patch *domain* edge — a
measure-zero feature, so it does not need `SSXOverlapRegion` as its output type. The
user's instinct that L25 and L28 are linked is still right, twice over:

1. **Shared code locus.** A grazing arc lives exactly where CSX boundary claims, rim
   verification (L27), and the tracer's graze-signature exits meet — the same machinery
   L28's rim-loop assembly will harden and regression-pin.
2. **Shared tolerance ladder.** Near the graze, "short overlap claim on the edge" vs
   "transversal arc that kisses the edge" is decided by the band-width rule — which §8
   defines. Fixing L25 before that rule exists invites another round of L27-style
   corrupt-claim ambiguity; fixing it after inherits a spec.

Hence the sequencing below (contract → L28 → L25), which also matches the kickoff's
priority order. The L25 work item itself: build the minimal graze fixture (plane vs
patch whose SSI is tangent to u=0 at one point), instrument the crossing lifecycle,
verify where the arc is dropped (claim filter vs tracer exit vs assembly), then fix at
that altitude with the harness asserting arc coverage.

## 10. Review findings (11-angle adversarial review + verification)

Method: 11 independent finder angles over the diff (line-scan ×2, removed-behavior,
cross-file tracer, language pitfalls, wrapper correctness, reuse, simplification,
efficiency, altitude, conventions), then verification. Verification was partly
re-planned mid-session (the account's session limit killed the A1 line-scan agent and
prevented a separate verifier fleet): the verdicts below rest on (a) finders' own
executed repros (marked *measured*), (b) main-session re-runs and direct line reads
(B2 re-reproduced, H3/H5/D2/E3 read at the exact lines, B1/B4 caller graph grepped).
Coverage note: both dedicated line-scan angles (A1: SSX files; A2: CCX/CSX/closest-point
files) were killed mid-scan by the session limit — their territory was substantially
but not exhaustively re-covered by the removed-behavior/tracer/wrapper/efficiency
angles, so a follow-up line-scan of `_bez_ssx5.py`'s new hunks is cheap insurance
(A2's dying lead worth checking: strict residual gates on curves lying in a plane
z = const ≠ 0). The conventions angle returned clean (no CLAUDE.md violations; no
>3.9 syntax; Black 140 respected).

**Ranked findings** (severity-first; CONFIRMED = executed repro or direct line evidence,
PLAUSIBLE = mechanism verified, trigger fixture not yet built):

1. **[CONFIRMED, measured] CSX overlaps narrowed to affine-certifiable only — curved-UV
   exact overlaps silently flood as isolated roots and are reported COMPLETE.**
   `csx/_bez_csx4.py:1487`. Parabola lying exactly on a bilinear patch: `tiny` →
   `{isolated:0, overlaps:1}`; HEAD → `{isolated:1679, overlaps:0,
   budget_exhausted:False, boundary_topology_complete:True}` after 33,685 cells. No flag
   anywhere; every downstream consumer accepts it. CCX received a non-affine fallback in
   this same commit; CSX did not. *This is the worst finding: silent wrong topology,
   flagged complete.* Fix: port the CCX fallback (or emit an honest partial).
2. **[CONFIRMED] `nurbs_ccx`/`nurbs_ccx_multiple` default-raise on any incomplete span
   crashes production callers.** `ccx/_nccx4.py:120` (raise), callers
   `topo/brep/boolean2d.py:80/420`, public `ccx()`. Near-coincident or non-affine
   same-support region curves → `RuntimeError` inside 2-D booleans that previously
   returned. One such span pair measured 25,476 cells, so a handful exhausts the new
   shared 100k allowance even without degeneracy.
3. **[CONFIRMED, live] `nurbs_csx` default-raise on `parameter_fibers` crashes the legacy
   path on exactly the geometry this commit fixed at the Bezier level.**
   `csx/_ncsx4.py:173`; callers `ssx/boundary_intersection.py:226/253`,
   `implicitize.py:460` pass no `return_status`. Collapsed edge (cone apex / sphere
   pole) on-surface → `RuntimeError`; pre-commit returned (slow) results.
4. **[CONFIRMED, re-run] CCX overlap loss for near-coincident / non-affinely-parameterized
   coincident pairs.** `ccx/_bez_ccx4.py:898`. Cubic vs itself +1e-9 (atol=1e-3): old →
   overlap (0,1); HEAD → 0 isolated + 0 overlaps, 2,015 cells, `budget_exhausted=True`
   (misattributed to budget). For an offset pair "empty" is arguably the exact-semantics
   answer, but a same-locus non-affine reparameterization (a genuine overlap) is also
   uncertifiable and lost, and the failure is billed to the budget instead of to the
   certificate. Needs the same non-affine fallback as #1 + reason-correct status.
5. **[CONFIRMED] C3 broadphase mis-pricing can burn the whole budget on numpy noise.**
   `_ssx5_singular.py:1360`: every AABB pair test (~ns, vectorized) is charged one shared
   cell (~ms of real work elsewhere). ~707 polyline segments → 250k pairs = the entire
   default budget spent on ~10 ms of numpy; C3 truncates and the run flags
   `budget_exhausted` on ordinary-sized output. (Probe corroboration: `c3` is already
   the #2 cell consumer on regular cases — 18.7k of case 6's 56.5k.) Price as
   pairs/128 like the `precompute` convention.
6. **[CONFIRMED] Final point-dedup precharges the pre-rewrite O(n²) cost, then SKIPS
   dedup when denied.** `_bez_ssx5.py:7832-7835` charges `n²` postprocess units for the
   new O(n·108) bucketed `_deduplicate_ssx_points`; at n>500 (always at the 1024 output
   cap: 1,048,576 > 250k) the charge is denied → exactly the dense pseudo-root outputs
   the rewrite targeted ship **un-deduplicated** with a spurious partial flag.
7. **[PLAUSIBLE ×2 independent finders] Cut-face CSX consumers drop `parameter_fibers`
   unflagged → silent branch loss through interior pinches.** `_bez_ssx5.py:5986` +
   multi-cut loops (7207/7256) read only `result['isolated']`; the fiber-producing CSX
   path returns `budget_exhausted=False`, so nothing is even marked partial. Same loss
   class case 14 fixed, one consumer over. Needs a pinched-interior-isoline fixture,
   then one shared CSX-result adapter for all three consumer sites.
8. **[CONFIRMED] Closest-point NURBS aggregators drop the budget signal entirely.**
   `_bez_closest_point.py:1272` (+1156): no `stats=`, no shared allowance, no kwarg to
   raise the 20k default; a patch hitting its cap warns and returns far-local-min
   entities that the aggregator merges as the certified globally-closest set — silently
   wrong answer at the public level. Related [PLAUSIBLE] `:843`: under the new single
   shared budget the 4 boundary searches can starve the interior heap to zero pops
   (old code gave the interior its own allowance).
9. **[CONFIRMED crash repro; trigger PLAUSIBLE] NaN → `math.floor` crash in the new
   binned dedup discards a whole completed run.** `_bez_ssx5.py:305`: `ValueError` on
   NaN / `OverflowError` on inf (legacy comparison dedup was NaN-safe). Feeder:
   **[CONFIRMED mechanism]** `:606` — strict gate written reject-if-greater
   (`if pres > tol: continue`), so a NaN residual is ACCEPTED (unsound direction); same
   inverted pattern at `:4696`. One NaN-producing rational eval (w→0) turns into either
   a garbage certified crossing or a crash after the budget was spent. Fix: accept-if
   (`pres <= tol`) + `np.isfinite` guards at both sites.
10. **[CONFIRMED, measured] `_nurbs_param_tol` semantic change reaches untouched legacy
    consumers.** `_nurbs_param_tol.py:541`: all non-uniform-weight rational inputs now
    take the conservative bound (was: negative-weight only) + rewritten optimistic
    formula. Rational quarter-circle: ptol shrinks 1.75×; same arc translated to
    (1000,2000): ptol grows ~350× (2.7e-4 vs 7.6e-7). Consumers `_bez_csx3.py:1661/1665`
    (public `ssx()` path) and `closest_point.py:745` shift acceptance/dedup radii with
    zero test coverage in this commit. Needs regression tests on the legacy pipeline or
    a consumer-scoped tolerance policy.
11. **[CONFIRMED] Zero-allowance contract violated for three knobs.**
    `_bez_ssx5.py:6322/6332`: `csx_max_results=0`, `boundary_csx_max_cells=0`,
    `csx_max_cells=0` are silently promoted to 1 via `max(1, …)`, while `max_cells=0` /
    `max_csx_calls=0` are honored as hard promises — inconsistent with the commit's own
    documented zero-allowance semantics (and its preflight tests).
12. **[PLAUSIBLE] CSX boundary-phase exhaustion discards already-verified boundary
    zeros.** `csx/_bez_csx4.py:1440`: certified roots in hand are dropped
    (`csx_boundary_zeros = []`) and Phase 2 has ~0 cells left to re-find them — the
    certified-partial contract loses its certified part (CCX keeps its validated hits in
    the same situation).
13. **[PLAUSIBLE] Case-13 seed cache degenerates to a global "seeded once" bit.**
    `_bez_ssx5.py:6921` + `6549`: for any depth>0 cell the cache key is rewritten to the
    top cell, so the first productive Φ∩L pass marks all of [0,1]⁴ seeded; a second,
    geometrically distinct tangency system's crossing-free loops would never be seeded —
    the exact branch-loss class the Φ∩L machinery exists to prevent. Needs a
    two-tangency fixture before/with any fix.
14. **[CONFIRMED] ssx6 draft: the new guard sits on top of a corrupted filter that
    crashes the whole interior path.** `_bez_ssx6.py:1902`:
    `list((lambda x: …, iterable))` builds `[<lambda>, list]` → unconditional
    `TypeError` in `_isoline_csx_to_global`; the contract test only exercises the
    boundary path. ssx6 is also confirmed a stale pre-budget fork (no budget kwargs, no
    singularities, fresh 100k per nested CSX). Decide: fix the filter + document ssx6's
    status, or quarantine it explicitly.
15. **[CONFIRMED] The same budget accounting is hand-rolled 8× with already-divergent
    semantics.** Inventory: `_SSXSoftBudget`, `BernsteinZeroBudget` (ContextVar-based),
    closest-point `_publish_work_stats`+inline loops, `bez_ccx`/`bez_csx` inline
    counters, `_nccx4`/`_ncsx4` `_new_status` twins (~110 lines copy-pasted, already
    diverged), `_ssx5_singular` closures. Charge semantics differ per copy
    (refuse-and-mark vs charge-then-compare vs check-then-increment vs charge-min).
    Plus the flag zoo: `hard_exhausted` and `output_counts` are published but have zero
    readers; `stop_when`/`stop_requested` and `max_boxes=max(256,16·max_cells)` are dead
    knobs. Consolidate into one shared module as part of schema v2 (§6).

**Additional verified debt** (doc-only; fold into the §11.4 consolidation batch):
duplicated exactness kits ccx/csx with *diverged roundoff envelopes* (64·n·ε_longdouble
vs 4096·n·ε_float64 for the same certificate class) and duplicate `_bernstein_product_1d`
names; three coexisting margin policies for the same hull-exclusion certificate (csx:
none, ssx: K=128 L1, ccx: depth-dependent); collapsed-geometry predicate in 3 places;
three Bezier interval restrictors with different edge behavior; `tol=` vs `atol=` kwarg
naming across sibling adapters with silent `**kwargs` swallowing of the wrong spelling;
`_curve_component_scale` dead; Optional-budget threading (41 `is not None` guards, 5
duplicated `charge_box` lambdas — null-object or ContextVar, which `_bern_zero_1d`
already demonstrates); zero-allowance preflight exists only at the SSX top entry while
`bez_csx`/`bez_ccx` still build superlinear nets before their first charge; C1-pass
fixed `min(20k, remaining)` tier saturates on case 14 (charge = 20,000 exactly, ~174k
shared cells unspent) — same disease L38/L40 fixed for internal cuts; efficiency items
(hull certificate ordered before the 18× cheaper AABB prune in ccx phase 2; strict
polish ×5 seeds ~9 ms/cell; `_choose_phi_equations` rebuilds 16 derivative tensors per
seed, 1.48 ms/call; O(B²) identity-mapping attempts ~1.1 s worst; loop-invariant
`_strict_ssx_root_tol` recomputed per root; overlap-duplicate postprocess keeps paying
per-branch midpoint evals after budget denial; `dt == 0.0` exact-equality guard in
`_nurbs_param_tol` where `< _TINY` was intended).

## 11. Next steps (proposed order)

0. **User decisions:**
   (a) overlap contract — Option C as specified in §8 — **APPROVED 2026-07-12**;
   (b) schema v2 rename (§6) + adapter default flip to always-return-status —
   **APPROVED 2026-07-12**;
   (c) ledger registration — **APPROVED + REGISTERED 2026-07-12 as L41–L54** in
   `docs/superpowers/issues/2026-07-07-ssx5-singular-review-ledger.md` (curated mapping,
   not 1:1 with the 15 findings; folds and splits by class):
   - L41 `[tracking]` schema v2 + adapter status policy — the approved (b); absorbs
     findings 2, 3, 11 and the naming conflation.
   - L42 `[P0 fix]` CSX curved-UV overlap fallback (finding 1) — **prerequisite for
     L28**: region rims come from CSX overlap claims, and curved rims are exactly what
     the affine-only certificate drops.
   - L43 `[P0 fix]` C3 broadphase pair pricing (finding 5).
   - L44 `[P0 fix]` point-dedup precharge + skipped dedup (finding 6).
   - L45 `[P0 fix]` NaN chain: inverted gates :606/:4696 + floor crash :305 (finding 9).
   - L46 `[P0 fix]` closest-point budget signal dropped + interior starvation (finding 8).
   - L47 `[P1 decide-first]` CCX near-coincident / non-affine coincident overlap
     semantics (finding 4) — exact-vs-tolerance contract is a policy call, then code.
   - L48 `[P1 fix]` param-tol semantic shift: regression tests for legacy consumers
     (finding 10).
   - L49 `[P2 fixture-first]` cut-face parameter_fiber drop (finding 7).
   - L50 `[P2 fixture-first]` Φ∩L seed-cache top-cell degeneration (finding 13).
   - L51 `[P2 fixture-first]` CSX boundary-exhaustion discards verified zeros (finding 12).
   - L52 `[P2 batch]` consolidation: one budget/status module, flag zoo, dead knobs,
     exactness kits, margin policies (finding 15 + §10 extended debt).
   - L53 `[P2 decide-first]` ssx6 disposition: fix corrupted filter + document staleness,
     or quarantine (finding 14).
   - L54 `[P2 audit, optional]` follow-up line-scan of `_bez_ssx5.py` new hunks
     (A1/A2 territory), incl. the z=const≠0 strict-gate lead.
1. **Schema v2 + adapter-policy fix** (small, 1 session): status object, reason strings
   at every `mark_incomplete` site, delete unread fields, update ssx6 contract test +
   harness; **flip the `_nccx4`/`_ncsx4` default from raise-on-incomplete to
   always-return-status** (soft-partial, same philosophy as the ssx5 core) and migrate
   the six production call sites — this closes §10 findings 1–2 in the same stroke.
2. **L28** (TDD off the §8 test list; rim-branch sampling upgrade included; ledger claim
   `CLAIMED(...)` per protocol; gates + case-12 objective harness).
3. **L25** (playbook re-pin, then fix; regression into the singular gate).
4. **Consolidation batch** from §10's confirmed cleanup findings — one shared budget
   module (kills the 8-way drift), shared exactness kit in `_bezier_common.py`, margin
   policy unification, dead-code removal (`stop_when`, `_curve_component_scale`, …).
5. **CI observability**: record per-case `status.work` in the coverage harness, fail on
   2× drift; institutionalize the two probes from this session (they run in ~3 min) as a
   `budget_contract` gate — 0 crashes / 0 contract violations / determinism is now a
   *tested* property, keep it that way.
6. **De-budget milestones** (§7.3): typed case-13 complement, C1 tier audit, then declare
   the invariant "no gate case reports `work_budget` in reasons" and enforce it.

## Appendix — probe method

Probe scripts: scratchpad `budget_probe.py` / `budget_fuzz.py` (session
`ca19a011…/scratchpad`; step 5 above moves them in-repo). Both reuse
`bez_ssx5_coverage_check.load_case_surfaces`, evaluate branch samples on both surfaces
via an independent de Casteljau, and validate: schema presence, finiteness, stuv ⊂
[0,1]⁴, residual ≤ 6·atol, `hard_exhausted ⇒ budget_exhausted`, md5-digest determinism.
Full JSON logs in the session task outputs.

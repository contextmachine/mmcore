# SSX v5 P1 — whole-call normalization preamble (design)

Approved 2026-07-21 (user decision via brainstorm: whole-call
normalization over per-certificate invariance; recorded rationale below).
Companion to the kickoff
`docs/superpowers/plans/2026-07-20-ssx5-invariance-kickoff.md`, which
holds the probe evidence, code anchors, and gate list — this spec does
not restate them. Scope: P1 only. P2 (the knob-unreachable tier) starts
after P1 lands and is decided separately with the user.

## Decision and rationale

`bez_ssx` runs its entire search in a canonical frame: both surfaces are
jointly centered and uniformly scaled at entry, `atol` is scaled with
them, and xyz outputs are un-mapped exactly once at exit. The
alternative — CCX-style centered/scaled contexts inside each strict
check — was rejected because its site list is open-ended (the kickoff
itself lists "possibly CSX-side gates"; 7,434 of case 11's 24,136 cells
are CSX sub-searches that would stay at world coordinates), this defect
class has recurred three times (`float t`, L48, this), and the probes
show normalization also *reduces* search work (case 6: 2,340→1,434
cells; case 11: 24,136→19,798). A uniform similarity preserves angles
and all atol-relative ratios exactly, so in exact arithmetic the result
is identical — the only change is floating-point conditioning.

## 1. Normalization context

New helper in `_bez_ssx5.py`:

- `_ssx_normalization_context(S1, S2, rational) -> (c, k)`
  - Joint Cartesian AABB over both control nets (dehomogenize when
    `rational`; `w` entries that are zero/non-finite make the context
    degenerate → identity).
  - `c` = AABB center (a float64 triple; any representable value is
    fine).
  - `k` = AABB diagonal snapped to the nearest power of two
    (`2**round(log2(diag))`), so the scale divide is mantissa-exact and
    only the one-time centering multiply-subtract rounds.
    **[Superseded by Amendment 2: the shipped formula is
    `2**round(log2(diag/16))`, targeting the native band.]**
  - Degenerate guard: `diag` zero or non-finite → `c = 0, k = 1`
    (identity transform; the pipeline behaves exactly as today).
- ~~Normalization is **unconditional**~~ **AMENDED 2026-07-21 (measured;
  user-approved): normalization is windowed.** Models whose joint
  coordinate magnitude (max |dehomogenized coord| over both nets) lies in
  `_NORM_IDENTITY_WINDOW = [2⁻⁵, 2⁵]` keep the identity frame; only
  models outside it are normalized. See "Amendment" below for the
  evidence that refuted the unconditional clause.

## 2. Entry transform

In `bez_ssx`, immediately after the existing `np.asarray` conversions:

- Rational: `H[..., :3] -= c * H[..., 3:]` then `H[..., :3] /= k` (the
  `_center_curve_homogeneous_for_exactness` pattern from
  `ccx/_bez_ccx4.py`, adapted to (n,m,4) surface nets). Non-rational:
  `(S - c) / k`.
- `atol_n = atol / k` (exact for power-of-2 `k`).
- `max_xyz_step`, when the caller provides it, is an xyz length →
  `max_xyz_step / k`. All other kwargs are counts/depths — invariant.
- Everything downstream (budgets, ladders, certificates, CSX calls,
  marchers, assembly) runs unchanged on the normalized data.

## 3. Exit un-map — exactly once

`_denormalize_result(result, c, k)` applied at the single `_result`
closure choke point that every return path already goes through.
Identity `(c=0, k=1)` short-circuits.

Un-map inventory (`xyz_world = xyz_n * k + c`):

- `SSXBranch.curve` control points' xyz components — **before** the
  derived `curve_xyz` / `curve_st` / `curve_uv` caches are materialized
  (they are `init=False` lazies; un-mapping the source curve first means
  the caches are built from world data).
- `SSXPoint.xyz`.
- `SSXSingularity.xyz` (`stuv`, `stuv_mate`, `samples` (N,4
  parameter-space), `branch_links` are invariant).
- `overlap_regions`: rim curves are branch references (already covered);
  uv loops, `interior_stuv` are parameter-space; `certification`
  residuals are recorded in atol units — invariant.
- `unresolved_regions`: payload audited during planning; any xyz field
  found joins this inventory (plan-level detail, not a new decision).

Rounding argument: the un-map adds one multiply-add per coordinate,
error ~`eps·|c|` absolute (≈7e-13 at case 11's offset) — far below any
supported `atol`.

## 4. Contract documentation (no arithmetic changes)

- `_strict_ssx_root_tol`: docstring gains its true precondition — it is
  evaluated in the normalized frame, where extent ≈ magnitude, so the
  extent-scaled budget matches the residual roundoff by construction.
  The 2026-07-20 diagnosis (extent-scaled budget vs magnitude-scaled
  noise) becomes structurally impossible rather than accidentally
  avoided.
- `bez_ssx` docstring documents the internal canonical frame and the
  world-in/world-out contract.
- The fixed 1e-14 corrector tolerances (`_ssx_correct`,
  `_ssx_correct_fixed`), the tolerance ladder, and all certificate
  arithmetic stay **byte-identical**. Never fix by loosening — and under
  normalization, never fix by touching at all.

## 5. Testing

- **Invariance property test** (kickoff gate 5, the durable class
  guard): fixed seed set of surface pairs; `bez_ssx(S1, S2, atol)` vs
  `bez_ssx((S1-c)/k, (S2-c)/k, atol/k)` must agree in topology (branch
  count/kinds/closure, reasons set) AND in un-mapped geometry, for
  translations `c` up to ~1e4 and scales `k ∈ [1e-2, 1e3]`. This test
  is simultaneously the guard against a missed/doubled un-map payload:
  an xyz field off by `k`/`c` fails the geometry comparison loudly.
- **Acceptance**: the kickoff's gates verbatim — case 6 original coords
  at atol=1e-3 → `complete=True, reasons=[]`, one branch matching the
  normalized-run topology; case 11 original at atol=0.1 → complete, at
  1e-3 → `trace_unverified` gone (`work_budget` may remain until P2).
- **Regression floor**: `tests/test_bez_ssx5_singular.py` (115),
  `tests/test_nssx5.py` (41), the 95-test bez/ccx/csx set, both
  coverage harnesses (`bez_ssx5_coverage_check.py` at 100%,
  `nurbs_ssx5_coverage_check.py` 8 OK rows unchanged, target 9 OK).
- Cell-count-sensitive pins and harness `CASE_NOTES` shift with the
  change (normalization alters every run's numerical trajectory —
  expected, kickoff anticipates it); update them WITH the engine change
  per their in-file comments.

## Amendment 2026-07-21: the identity window (measured, user-approved)

Wiring the unconditional frame regressed 4 of the 115 singular-suite
tests (all near-origin fixtures, native magnitudes 3–12.6), confirmed
against the pre-preamble engine. A variant experiment (identity /
scale-only / center-only / full, all four fixtures) split the cause:

- `cusp_curve_on_split_plane` fails under **center-only**: the centering
  subtract's ~1-ulp rounding (center contains 1/3) destroys the exact
  derivative-zero line the fixture encodes.
- `tangent_curve_no_point_flood` (k=16) and `positive_dim_sigma` (k=4)
  fail under **scale-only**, which is mantissa-exact — proving the
  singular tier's absolute thresholds (`gauss_newton_witness
  tol_f=1e-8/1e-10`, `|F| < 1e-11` accepts) are magnitude-sensitive.
  This is a pre-existing latent property the frame exposes, not one it
  creates.
- `closed_tangent_loop` fails only under the combination.

A full-suite survey (identity frame: 115 passed; native magnitudes
1–362; all fixtures above magnitude 32 also pass under the full frame)
plus the gate probes (case 6 @1e-3: complete, one branch
[4.37,75,1]↔[75,4.37,1]; case 11 @0.1 complete; case 11 @1e-3
`trace_unverified` gone) bound the fix: identity inside
`[2⁻⁵, 2⁵]` joint magnitude, normalize outside. Every measured
constraint is satisfied: the 4 fixtures (≤12.6) sit inside, the
trace-certificate defect is only measured at magnitudes ≥ ~71 (case 6
recentered), and everything the suite exercises above 32 is proven on
the normalized path.

Follow-up **P1b** (not this work package): make the singular tier's
absolute thresholds scale-aware so the window can eventually widen to
"always" — `docs/superpowers/issues/2026-07-21-ssx5-p1b-singular-tier-scale-invariance.md`
holds the experiment table and threshold-site inventory.

## Amendment 2, 2026-07-21: target-band scale (measured)

Task 6's verification exposed a second regression class: bez-harness
case 10 (regular transversal, magnitude 79.7 → out-of-window, k=32
under Amendment 1) dropped from 218/218 coverage to 211/218 — one arm
fragmented at two INTERIOR endpoints around s≈0.392 with
`complete=True, reasons=[]` (silent incompleteness — a latent
accounting bug, filed with a native reaching fixture as **P1c**:
`docs/superpowers/issues/2026-07-21-ssx5-p1c-silent-fragment-completeness.md`).

Cartography (forced pure scales, mantissa-exact, c=0): case 10 keeps
218/218 for k ≤ 8 (post-frame magnitude ≥ ~10) and silently fragments
for k ≥ 16 (magnitude ≤ ~5) — identical missed segment at every broken
scale and under every centering variant. So normalizing to O(1)
magnitude, as Amendment 1 did, walks the TRANSVERSAL marching path out
of its calibrated regime, the same latent class as P1b's singular tier.
A scale-only frame (no centering, bit-exact both ways) fixed case 10
and all gates but failed the offset-dominated property extreme
(offset/extent ~4e5: normalized extent collapsed to ~3e-5, the CSX
ladder's absolute floors exceeded the scaled atol, and the search
bailed at 342/250,000 cells claiming `work_budget`).

**The synthesis (shipped):** center at the joint AABB midpoint AND
scale by `k = 2^round(log2(diag / (2·_NORM_TARGET_MAG)))` with
`_NORM_TARGET_MAG = 8`, landing the post-center AABB diagonal in
`[11.3, 22.6]` and hence max|coords| in `[8/√6, 8√2] ≈ [3.27, 11.31]`
depending on extent anisotropy — inside the fixture-proven native band
`[1, 22.6]`. (Adversarial-review correction: first stated as
`[5.66, 11.31]`, true only for single-axis extents; sub-5 magnitudes
are safe because the P1c crossing-guard removed the one measured sub-5
failure mechanism.)
Both halves are load-bearing: centering preserves extent for
offset-dominated inputs; the band target preserves the marching regime
for extent-dominated ones. Measured green under this map: case 10
218/218 (2 branches), case-6/11 gates exact, the 4 singular fixtures,
all 25 invariance tests including every property extreme.

## 6. Out of scope

`_nssx5.py` (correct; reports engine truth), tangential semantics (stop
and re-scope if they move), Cython, P2. If a fixture ever demands
per-axis anisotropic handling inside a specific predicate, the
CCX-style context remains available there later — normalization does
not foreclose it.

# Cluster 4 RESOLVED — common-origin centering cancellation (2026-07-25)

Burn-down tier 1 of the derived-envelopes program
(`docs/superpowers/plans/2026-07-25-ssx5-derived-envelopes-kickoff.md`,
audit `2026-07-25-ssx5-threshold-audit.md` §Cluster 4).  Branch
`ssx5-derived-envelopes` off `ssx5-invariance`.

The audit deliberately left this cluster's mechanism unlocalized ("the
collapse mechanism is NOT yet localized to a constant").  It localized to
something the audit's taxonomy could not have found by enumerating
literals: **not a bare threshold at all, but a correctly-shaped relative
envelope measured against the wrong quantity.**

## Localization (frame binary search, as directed)

Direct `bez_csx(s1 v=0 edge, s2)` under the four frame components:

| frame | u=1 edge | v=0 edge |
|---|---|---|
| identity | `exact` overlap | `exact` overlap |
| scale-only (k=2, c=0) | `exact` | **`exact`** |
| center-only (k=1, c) | `exact` | **0 overlaps, 4046 pseudo-roots, exhausted** |
| both (engine) | `exact` | 0 overlaps, 3989 pseudo-roots |

**The scale half of the P1 frame is fully covariant; the translation half
is the whole defect.**  The k=2 framing in the kickoff's title is
incidental — centering alone reproduces it.

Chain, top down: the losing call is not the CSX overlap tier itself but a
CCX nested inside its boundary sweep — `C × s2`'s v=0 boundary isocurve,
a transversal crossing of two coplanar segments at t = 122/175.  Identity
frame finds it (405 cells); centered frame returns 0 roots in 5 cells.
Losing that boundary zero drops the count 2 → 1, which both skips
`_check_csx_overlap_valley` (needs ≥ 2) and fails the L59 arming gate
(`_valley_pair_seen or not csx_boundary_zeros or any(bz.axis == 0)` — the
survivor is axis 1).  Phase 2 then grinds the coincident span into
thousands of lattice roots and the span evaporates into
`reasons=['overlap_region_unsupported']`.

The squared-distance net `F` is **bit-identical** in all four frames, as
are the classifier verdict and every net-derived prune.  Phase 1 is
innocent; only the predicates that touch raw coordinates diverge.

## Mechanism

`_center_curve_homogeneous_for_exactness` (ccx) and
`_center_homogeneous_for_exactness` (csx) normalize BEFORE translating:

```python
H /= scale                            # scale = max|H|, whole net
H[:, :-1] -= origin * H[:, -1:]       # <- cancellation lives here
```

A coordinate that is identically zero after translation is therefore
computed as `fl(x/scale) - fl(origin*fl(w/scale))`.  Those two roundings
are independent, so the result is **cancellation noise, not an exact
zero**.  In world coordinates the fixture's z column is literally 0 and
every downstream predicate sees an exact zero; centered, it sees ~1e-19 of
garbage.  Whether that garbage is same-sign — and therefore whether the
run survives — is luck: measured over random translations, 38% of
positions produce a nonzero z "scale", 30% produce a false rejection.

Three predicates consumed the cancelled value as if it were data:

1. `_vector_residual_hull_excludes_zero` (ccx Phase-2 prune) — margin
   `op_factor*(|lhs|+|rhs|)` built from the already-cancelled products:
   **5.7e-19 of noise against a 2.6e-31 margin**, so the noise reads as
   "provably nonzero" and the whole root cell is deleted.
2. `_ccx_exactness_context` → `_eval_curve_scaled_components` — reports
   the noise as `component_scale` and then DIVIDES by it, amplifying
   1e-19 to O(1) and making `_strict_residual_ok` refuse every root.
3. `_strict_polish_ccx._reported_residual_ok` — requires `value == 0.0`
   exactly on a zero-scale axis, reachable only when the sources are
   literally zero.

Plus the csx twin `_strict_csx_residual_ok`, which bounds the residual by
`max(component_scale, |pc|, |ps|)` — all three post-cancellation.

## Fix (derivation, no new constants)

`_center_*_for_exactness` optionally returns the **magnitudes of the two
operands the subtraction consumed**, carried through the same final
normalization.  That is the quantity the error is actually relative to.
Consumers then:

- **prune**: add the house SOURCE term
  `8192·(n₁+n₂)·eps_f64 · (src₁⊗|w₂| + |w₁|⊗src₂)` — the identical
  two-term structure `_overlap_mapping_is_identity` and
  `_certify_affine_csx_overlap` already carry.  New margin ≈ 2.1e-13
  against 5.7e-19 of noise, and ≈ 11 orders below a genuine 1-unit offset:
  the prune keeps full strength.
- **context**: an axis whose post-centering content does not exceed its
  own centering envelope reports scale 0 — i.e. "absent" — restoring
  exactly the world-frame behaviour.  Bar is the `32·degree_factor`
  family already local to `_strict_residual_ok`; the fixture's noise sits
  ~4 orders below it.
- **`_reported_residual_ok` / `_strict_csx_residual_ok`**: on an absent
  axis the bar is that envelope instead of exact zero.  Sources genuinely
  zero on an axis give envelope 0, so the pre-existing exact-zero
  behaviour is preserved bit-for-bit.

Adversarial measurement of the prune's strength (20,000 random generic
curve pairs, degrees 2–4, depths 0–7, seeded): the landed two-term form and
the old operator-only form agree on **20,000/20,000** — the prune is
bit-for-bit as strong on ordinary geometry, and the ×1.00 work counts on
the bez harness say the same at the pipeline level.  On 5,000 random
translations of the coplanar INTERSECTING pair, where the correct answer is
always "do not prune", the old form wrongly pruned **1,748** and the landed
form **0**.  The change is surgical, not a relaxation.

Direction note (why the same weakness was benign in the siblings): the
certificates ask "is this residual explainable as zero?", where an
underestimate merely refuses to certify.  The prune asserts a residual is
provably NON-zero and deletes the cell — an underestimate silently loses
solutions with no downstream recourse.  Only the prune's copy was ever
load-bearing in the unsound direction.

**One measured near-miss worth keeping:** the first csx attempt added the
envelope to EVERY axis rather than only degenerate ones.  Near a tangency
the parameter error grows as √(residual envelope), so that blanket second
term moved `test_bounded_newton_stall_near_tangent_is_not_a_distinct_root`
off t=0.5 by 1.7e-7 against a 1e-7 bar — a ×√2 widening, exactly as
predicted.  Widening a residual envelope is never free even when it looks
dimensionally harmless; the landed form touches degenerate axes only.

## Acceptance

- Reaching fixtures (new, all previously failing): 12 ccx cells
  (`test_coplanar_crossing_survives_translation` ×7,
  `test_vector_residual_hull_prune_is_sound_under_translation` ×4, plus
  an anti-loosening guard), 2 csx property tests (300 random world
  positions: **99 → 0** false rejections; genuine-offset guard retained),
  2 nssx5 tests carrying the user's pair with analytically-derived truth
  (branch lengths 14.2465057 / 5.4848060, corner outside s2 ⇒ two
  branches).
- Similarity sweep extended: `boundary-coincidence` joins planes / loop /
  rational-arc → 4 classes × 6 transforms = 24 cells green.  This is the
  tier's acceptance gate per kickoff §3.
- Floors: 116 singular / 34 invariance / 46 nssx5 / 134 ccx-csx, bez
  harness 8×100% at baseline work (×1.00).
- NURBS harness A/B on its heaviest case (2): HEAD and branch agree
  bit-for-bit — 1,054,598 cells, 19,864 csx calls, identical per-tier
  counts, 11 branches, complete, 332.0s vs 330.6s.  The engine is
  unchanged on non-degenerate work; case 2's runtime is intrinsic.
- Pre-existing at HEAD, NOT caused by this tier (verified by stash):
  `tests/test_nccx4.py`'s 3 collection errors (missing module), and
  `tests/test_ssx5_c1_regular_normal.py::test_high_order_tangent_never_publishes_off_locus_branches[10,12]`
  (published branch at y=0 instead of y=0.5).  The latter is a real
  off-locus publication on a suite outside the documented floor list —
  worth its own triage, unrelated to centering.

## Investigated and deferred — the centering ORDER (follow-up, fixture-first)

The landed envelope is correct but inherits one property worth naming
before someone rediscovers it: it is built from the operand magnitudes of
the centering subtraction, and those operands carry the model's WORLD
POSITION (because `H /= scale` runs first, with `scale` sized by that
position).  So the prune's separation floor on a degenerate axis degrades
with distance from the origin — measured ~1e-10 world units when centered
on the model, ~1e-7 at a 1e3 translation.

That direction is SAFE — the prune merely declines to fire and the cell
falls through to the sound subdivision/Newton path — and the floor sits
orders below any tolerance the engine is called with.  It is nevertheless
the program's own target class (a bound whose strength depends on where
the model sits), so it is filed, not forgotten.

The real fix is to reorder: subtract the origin BEFORE normalizing, which
for unit weights makes the subtraction exact and removes the noise instead
of budgeting for it.  **Attempted this session and reverted**: it regresses
`test_ccx4_exactness_contract.py::test_float_built_quadratic_subcurve_remains_an_overlap`,
a fixture calibrated against the current arithmetic, and it does not by
itself tighten the envelope (which is computed from operand magnitudes
either way — exploiting the new exactness needs a second change).  Proper
scope: its own tier, with that contract fixture re-derived rather than
re-calibrated.  The overflow guard the current order exists for must
survive as a fallback.

## Consequences for the program

- **The user's fixture passes and `ssx5-invariance` is unblocked** — the
  P1 frame regression it exposed is repaired at the predicate level, not
  by weakening the frame.
- The audit's method has a documented blind spot: enumerating
  epsilon-family *literals* cannot see an envelope whose constant is fine
  and whose *operand* is wrong.  Cluster 1's per-row normalization work
  should be read with this in mind — ask what each bound is relative TO,
  not only how big it is.
- Frame-retirement (kickoff §4) gains evidence: two of the frame's halves
  are now known to differ in kind.  Power-of-two scaling is exactly
  covariant and cheap; centering is the half that perturbs predicates and
  it bought nothing here.  Not a decision to take yet, but the scale half
  looks retirable independently of the centering half.

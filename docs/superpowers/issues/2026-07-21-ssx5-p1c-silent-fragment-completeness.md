# P1c: silent incompleteness — interior-truncated fragments certify complete (filed 2026-07-21)

> **PRODUCER FIXED same day (this branch):** the measured mechanism was
> NOT a marcher truncation — probe cartography (pop-sequence event log)
> showed the **GJK prune declaring a 2-crossing cell "separated"**
> (`_bez_ssx5.py` main loop, pop #161: `_aabb_disjoint` False → `gjk`
> False → continue), deleting the arc through it. Fixed by the soundness
> guard *certified crossings outrank approximate prunes*: AABB/GJK/F_sq
> prunes are skipped for crossing-bearing cells. Post-fix cartography:
> case 10 holds 218/218 with bit-identical branches at EVERY forced scale
> k=1..32 (the knife edge is removed, not avoided). Regression pinned by
> `test_small_scale_case10_keeps_certified_crossing_cells` (native
> 1/16-scale input, no monkeypatch). REMAINING OPEN (below): the general
> accounting audit — an interior-ending open fragment without a typed
> reason should never certify complete. No reaching fixture exists after
> the guard; fixture-first (L50) says wait for one (P2's instrumentation
> is the likely source) rather than ship an unverifiable check.

Discovered during P1 canonical-frame cartography (see the P1 spec's
amendments and `2026-07-21-ssx5-p1b-singular-tier-scale-invariance.md`).
This is a **pre-existing latent soundness bug**, reachable today without
any P1 machinery; P1's measurements produced the first reaching fixture.

## The defect

`bez_ssx` can return `complete=True, reasons=[]` while a traced branch is
missing an interior sub-arc: the arc's two surviving fragments BOTH
terminate at interior stuv points (not domain boundaries, not matched
endpoints), yet nothing marks the result partial. `complete` is the one
bit consumers act on (schema v2) — this violates its contract.

## Native reaching fixture (measured 2026-07-21)

Scale the bez-harness case-10 input nets by 1/16 and atol by 1/16 —
i.e., author the same geometry at ~5-unit size, which the identity
window passes straight to native arithmetic (this is bit-identical to
the forced-k=16 probe run):

- Result: `complete=True, reasons=[]`, **3** branches (native full-size:
  2 branches, 218/218 reference coverage).
- The s:[0, 0.599] arm fragments into s:[0, 0.3749] + s:[0.4121, 0.599];
  fragment ends are INTERIOR: `[0.3749, 0.4013, 0.5838, 0.519]` and
  `[0.4121, 0.3856, 0.571, 0.5805]`. The missing arc ≈ 0.286 xyz units
  (~57 × 5·atol at that scale); 7 of 218 reference points uncovered.
- Deterministic; identical under forced k=16 and k=32 (and under the
  scale-only / quantized / full centering variants — the trigger is the
  magnitude regime, not the transform kind). k ≤ 8 (mag ≥ ~10): intact.

Probe recipe: `case10_scale_threshold.py` pattern (scratchpad) — force
`_ssx_normalization_context` to `(zeros, k)` and run
`examples/ssx/bez_ssx5_coverage_check.py` case 10 in-process.

## Two distinct work items

1. **Accounting (the soundness half, smaller):** a branch fragment that
   terminates interior without a structural reason must not leave
   `complete=True`. Locate the keep-path for these two fragments (the
   tracer keeps interior-truncated marches as open fragments by design —
   handoff §3; some keep-paths mark reasons, this one doesn't) and make
   it surface a typed reason. Never certify what wasn't verified.
2. **The march break itself (P1b-class):** why does the continuation die
   near s≈0.392 only below magnitude ~5? Same absolute-threshold
   inventory question as P1b, but on the TRANSVERSAL path (marchers /
   CSX registrations / exit matching), not the singular tier. The k=8
   vs k=16 bracket localizes the responsible constant's scale.

## Constraints

- Never fix by loosening (house invariant).
- The 115-suite, the bez harness at native scale, and the P1 invariance
  property test must stay green bit-for-bit.
- Fixture-first: commit the 1/16-scaled case-10 fixture with the fix
  (L50 lesson: no fix without a reaching fixture).

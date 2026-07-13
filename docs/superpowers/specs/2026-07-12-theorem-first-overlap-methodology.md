# Theorem-first overlap certification — methodology (USER DECISION 2026-07-12)

**Status:** adopted; implemented for CSX in `_bez_csx4._tolerance_csx_overlap_certificate`
(ledger L59/L60); CCX's L47 tier is the ancestor. This document is the methodology of
record for all coincidence handling in mmcore's intersection engines, and the template
for the future quadric/rational tier (L58).

## 1. The principle (the user's rationale, on record)

Two polynomial/rational Bézier arcs that coincide on **any open sub-arc** lie on the
same irreducible algebraic curve — coincidence is all-or-nothing, never local.
Therefore a maximal overlap can only **terminate at a domain boundary** of one of the
operands: nothing interior to both domains can end it.

The methodological consequence: **do not fight floating point for the interior.**
Numerics is used only to establish *which structure* we are in:

1. verify the span's ends are **domain-pinned** (see §3),
2. verify **interior witnesses** are within tolerance,
3. verify **no interior crossing structure** (see §5),

and then the *theorem*, not the arithmetic, carries the interior of the span. A
deliberately incomplete numerical proof of a mathematically complete property. This
replaces two failed regimes:

- **tiny's valley rule** (pre-`5d05ddc`): permissive pairing with no certificate —
  fast and often right, but provably unsound (merges sub-tolerance-distinct sets;
  fails every exactness contract).
- **the 5d05ddc hardening**: exact-affine-only certification with no tier for genuine
  non-affine overlaps — sound but incomplete; real coincidences (curve on a
  non-parallelogram planar quad, curve on an extrusion) fell into a subdivision grind
  and shipped truncated or empty (the "what got overlooked" of this branch).

## 2. Tolerance semantics (resolves the L47 band-bar residual)

**Tolerance-coincidence IS coincidence.** A span whose ends are domain-pinned, whose
witnesses sit within `atol`, and which carries no interior crossing structure is ONE
overlap. The `certification` field says which grade:

- `'exact'` — witnesses at roundoff level (the `tiny = 4096·ε·max(1, diag)` floor);
  algebraic identity in the source-envelope sense.
- `'tolerance'` — witnesses within `atol`; sets that are distinct-but-within-tolerance
  are *reported as* coincident-within-tolerance. The exactness property survives in
  sharpened form: **sub-tolerance-separated sets must never certify `'exact'`**
  (measured: even a 5e-324 offset stays `'tolerance'`).

Consequences already pinned in tests: parallel offsets at 0.5·atol, sub-atol humps,
and translated variants all promote as `'tolerance'` with translation-invariant
residuals (`test_csx4_exactness_contract.py`, `test_ccx4_exactness_contract.py`).

## 3. Domain pinning (with the real-data amendments)

A span end is pinned when any of:
- it is the **curve's t-domain end** (t=0/1),
- the projection onto the surface sits on a **uv-domain edge**,
- **(amendment, measured)** the projection *one grid step outward* clamps to a uv
  edge — because on real data the tolerance boundary and the domain exit COINCIDE
  (measured: d crosses atol at almost the same t where the path leaves through u=1,
  so the projection at the refined boundary is still interior), and
- **(amendment, measured)** a span end may have **no exact 3-D root at all**: the
  curve can leave the patch through an edge *region* at sub-atol clearance
  (measured 6.7e-4 from the edge line). Pins are verified by inversion + residual,
  never by requiring a boundary zero to exist.

A span with **both ends interior-fading** (distance rising through atol with the
projection interior) is the **offset-twin signature and is never promoted** — the
CCX L47 rule, kept verbatim.

## 4. Structural amendments (from the algebraic geometry)

- **Multiplicity of spans:** improper/folded reparameterizations (a degree-6 double
  traversal) and domain clipping legitimately produce **multiple** pinned spans —
  the certificate returns a set, never assumes one.
- **Nodes ride along:** arcs of the same nodal curve can also meet at isolated
  self-intersection points outside the spans; those stay isolated roots.
- **Corner contact is not band evidence:** a single-grid-sample in-band run (a graze
  at one node whose bisection fringe can exceed 4·ptol) is refused; a genuine span
  must be in-band across ≥ 2 consecutive grid samples (caught live: a 4·atol stub
  branch at a shared corner).

## 5. The crossing guard (what protects the blood-bought invariants)

The **never-merge-tolerance-touches invariant** (near-tangent loop topology) is
protected by the *flip guard*, not by refusing overlaps: a transverse (normal-side)
**sign flip between consecutive interior gap samples** is crossing structure and
refuses promotion of that span.

- Root-like samples (residual ≤ `tiny`) are bridged (the root is the coincidence).
- **(amendment, measured)** END-adjacent flips are exempt: a genuine touch AT a
  pinned span end on real-world-inexact data sits above the roundoff floor (1.6e-9
  in the fixture) so bridging cannot cover it — it is the span's own endpoint root;
  the theorem terminates the overlap there anyway. Interior flips still refuse.
- Sub-atol **valley chains** between strict-distinct zeros never merge: the strict
  gap-midpoint certificate fails on valley floors (slice-10 lattice-cluster rule,
  pinned by the 3-root valley-chain negative control).

**Open (user decision pending):** CCX's woven near-coincident twins (crossings at
fitting-noise amplitude) still refuse per the pinned L47 contract test. If the
tolerance semantics of §2 should bridge sub-band weaving in CCX too, the candidate
rule is a *relative* bar (flip flanks ≤ K× the span's median in-band residual =
noise), which discriminates the pinned ±1.2e-4-crossings-in-band fixture (120,000×
median → refuse) from fit-noise weaving (~1× median → bridge). Not implemented.

## 6. Cost discipline (the §11.5 lesson, twice)

- **Arming is evidence-gated and cheap:** a curve-end zero on-surface, a valley
  pair, or a zero-boundary-zero pair (coincident stretches entering AND exiting
  through patch edges produce NO exact boundary roots — measured; transversal
  nested calls always carry zeros and never pay the scan).
- **Split pricing:** the 17-projection arming scan bills 17 cells; the dense pass
  (65 witnesses + refines) bills 145 **only on a hit**. A flat combined price
  tripped the work-drift gate on nested cut-face calls (case 15 ×3.61) and a
  constrained-budget test — the gate caught it live, twice, on the day it was built.
- Certified spans are **t-excluded from Phase 2**: no subdivision grind inside a
  certified span. Baselines in `examples/ssx/bez_ssx5_work_baseline.json` refresh
  only deliberately (`--update-baseline`).

## 7. Measured record (why this methodology is trusted)

| Case | tiny (valley rule) | 5d05ddc..pre-L59 | theorem-first |
|---|---|---|---|
| script-3 call 2 (extrusion) | 2.7 s, 1 overlap, unsound rule | 57 s, 0 overlaps, lost geometry | 4.2 s, 1 overlap **(5.9230, 20.1120)** vs tiny's (5.9238, 20.1131) — sub-ptol agreement |
| L42 parabola-on-bilinear | 1,679 lattice roots "complete" | bounded typed-partial | 1 `'exact'` overlap, 204 cells |
| bilinear L-junction (fixtures A/B) | — | branch truncated to 37%/11% | both branches full, linked to the tangent point |
| exactness contracts | FAIL (merges distinct sets) | pass | pass (re-pinned, sharpened never-'exact') |

**The remaining fundamental cost:** inputs whose answer is genuinely delicate
(clearance ≈ atol, fast-rotating normals, real near-tangencies) must be resolved at
tolerance scale by any *honest* method. tiny was faster there only by answering wrong.

## 8. Known incompleteness (merge-blocking, ledger L61)

The user reports the bilinear non-affine family **still loses part of the second
branch** in their environment. At HEAD `f6d0015` every constructible variant passes
(A/B/C plus both order-swapped forms D/E: full 9.763-length branches, linked tangent
point, complete). The failing configuration is therefore not yet captured: it may be
the NURBS-level driver (non-unit knots), a geometry variant, or a stale checkout.
**Repro-first is mandatory**: no fix without the user's exact failing script and
geometry. The branch does not merge to `tiny` until this is closed.

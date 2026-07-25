# SSX derived-envelopes kickoff — eliminate calibrated thresholds (written 2026-07-25)

Written at the end of the P1-invariance session, immediately after a
user-authored test case exposed the class limit of that approach. All
measurements below were made this session at branch `ssx5-invariance`
HEAD `497aa84`; do not re-derive them — reproduce, then execute.

## The mandate (user decision, 2026-07-25 — adopt as a repo invariant)

> Calibration is acceptable only if it provides real benefits and can be
> reliably determined automatically.

Operationalized: **no bare numeric threshold in any residual or
classification predicate; every envelope must be derived from its
operands** (dimensionless, atol-relative, or roundoff-derived
`K·eps·⟨operand magnitude/condition factors⟩`). This sits next to
"never fix by loosening" in priority. The user's driving question —
"how many more cases must I document before the next surprise?" — has
the answer *unbounded* under calibrated constants and *one fixture per
structural class* once envelopes are derived. That conversion is this
work package.

## The two threshold families (the diagnosis in one table)

| Family | Examples | Scale bugs ever |
|---|---|---|
| **Derived** | L1 hull margin `K·eps·max\|c\|` (`_ssx5_singular.py:35-79`), CCX common-origin contexts (`ccx/_bez_ccx4.py:100-206`), `_strict_ssx_root_tol` (extent-scaled), all angle/ratio/param-space tests, atol-relative ladders | zero |
| **Bare (calibrated)** | `tol_f=1e-8` (`_bez_ssx5.py:846`), `tol_f=1e-10` (`:880`, `:1239`), `\|F\|<1e-11` (`_ssx5_singular.py:641`), corrector `tol=1e-14` (`_ssx_correct`/`_ssx_correct_fixed`), GJK internals (`cygjk`), whatever broke in the CSX overlap tier at k=2 (below) | every incident of 2026-07-21..25 |

Measured members of the failure class (each has session evidence):
1. **Singular tier** — 2 fixtures flip under *mantissa-exact* rescale
   (P1b issue, experiment table inside).
2. **Transversal marching path** — case 10 silently fragmented below
   post-frame magnitude ~5; root cause was the GJK prune deleting a
   2-crossing cell (fixed by the crossing-guard, `1fe0a1a`); the
   underlying march sensitivity is bracketed k=8 vs k=16 (P1c).
3. **CSX overlap tier** — the new user case (below): overlap
   certification is perfect at world frame, collapses at k=2.
4. **GJK primitive** — contact → "separated" (20000/20000 measured),
   gap<atol → "separated" (atol arg gives no proximity margin;
   dimensionally not a length), iteration exhaustion → "separated".
   Unsound at ANY scale; Cython (P1c inventory item 3).

The P1 canonical frame (windowed target-band, spec + 2 amendments) fixed
the tiers whose envelopes were already derived (trace certificate:
cases 6/11 gates green) and RELOCATES the calibrated tiers to a
different point of their miscalibration curve — which is exactly how
the user case broke. The frame is a stopgap, not the fix; the fix is
per-predicate derivation (the CCX way — which was the design fork's
option (a), passed over on 2026-07-21; the evidence has now reversed
that trade-off for the long term).

## The triggering case (commit this as the priority fixture)

User-authored bilinear pair, found on their FIRST new test. Single-span
(pure Bezier) surfaces; s1 height `z = 5(1−u)v` ⇒ its z=0 locus is the
u=1 and v=0 DOMAIN EDGES; s2 is a planar (z=0) non-parallelogram quad.
True intersection: two straight segments ON s1's domain boundary,
clipped to s2's quad; shared corner (−36,2,0) is outside s2 ⇒ two
separate branches (boundary-coincidence class).

```python
import numpy as np
from mmcore.geom._nurbs_eval import NURBSSurfaceTuple

s1 = NURBSSurfaceTuple(
    order_u=2, order_v=2,
    knot_u=np.array([0.0, 0.0, 29.20616373, 29.20616373]),
    knot_v=np.array([0.0, 0.0, 18.68154169, 18.68154169]),
    control_points=np.array([[[-16., -27., 0.], [-8., -25., 5.]],
                             [[-36., 2., 0.], [-20., -3., 0.]]]),
    weights=np.ones((2, 2)))
s2 = NURBSSurfaceTuple(
    order_u=2, order_v=2,
    knot_u=np.array([0.0, 0.0, 19.84943324, 19.84943324]),
    knot_v=np.array([0.0, 0.0, 12.04159458, 12.04159458]),
    control_points=np.array([[[-34., -7., 0.], [-26., 2., 0.]],
                             [[-19., -20., 0.], [-17., -10., 0.]]]),
    weights=np.ones((2, 2)))
# nurbs_ssx(s1, s2, atol=1e-3)  ->  complete=False,
# reasons=['overlap_region_unsupported'], v=0-edge branch 29% covered.
```

Measured failure chain (2026-07-25, every step probe-verified):
- Direct `bez_csx(edge, S2)` at WORLD frame: both edge spans certified
  `'exact'`, residual ~1e-14 — u=1 span t∈[0.489, 0.816], v=0 span
  t∈[0.377, 0.782]. The CCX-style machinery handles this flawlessly.
- The pair's joint magnitude is 36 — FOUR units above the identity
  window bound 2⁵ — so bez_ssx reframes at k=2, c=[−22,−12.5,2.5] (a
  bit-exact transform for these dyadic inputs: NOT a rounding issue).
- Same edge, canonical frame: **0 overlaps, 3,989 isolated
  pseudo-roots** — the CSX overlap tier collapses at a 2× reframe.
- The v=0 span therefore travels as an "uncertified span"; the L28
  region assembler can use those only as 2-D rim evidence; open spans
  form no region (`regions=0`) ⇒ the span EVAPORATES into the typed
  reason. The u=1 span happened to certify (its BoundaryOverlap took
  the L59 33-sample path and shipped as the surviving overlap branch).
- Identity frame forced (monkeypatch): `complete=True, reasons=[]`,
  BOTH branches exact-certified overlap curves, lengths 5.49 / 14.25
  matching analytic truth. **The P1 frame regressed this case.**

Why the P1 gates missed it: no coincidence-class fixture exists in the
out-of-window magnitude band, and the invariance property test is
transversal-only by documented design.

Probe idioms (scratchpad is wiped between sessions — recreate from
these): force frames by monkeypatching
`m._ssx_normalization_context = lambda a,b,rational=True: (np.zeros(3), K)`;
spy on in-engine CSX via `m.bez_csx`; the boundary sweep is
`_find_ssx_boundary_zeros`/`_process_face` (`_bez_ssx5.py:~600-660`,
the L59 curved-path block and the `chord_ok` `continue`); span→branch
conversion `_overlaps_to_branches` (`:4483`); region tier + reason
retirement `:7901-7935`; ground truth by sampling the z=0 edge segments
against s2's quad (matplotlib.path.Path containment).

## The program (sequence for the next session(s))

1. **Audit (do this FIRST — it answers the user's question with a
   number).** Mechanically enumerate every floating-point literal used
   in a comparison/threshold across `_bez_ssx5.py`, `_bez_csx4.py`,
   `_ssx5_singular.py`, `_bezier_common.py`, `bern_sq_dist.py`,
   `_work_budget.py`, `cygjk.pyx`, and the boundary/overlap modules.
   Classify each: (a) dimensionless/param-space, (b) atol-relative,
   (c) roundoff-derived, (d) **bare residual/classification threshold**.
   Deliverable: an audit report doc with per-tier counts and the
   burn-down list, committed. Grep is not enough — classification needs
   reading each site's dimensional context (the 2026-07-20 kickoff's
   note "step-size guards are fine, residual thresholds are not" is the
   classification rule in miniature).
2. **Burn-down, fixture-first, one tier per commit.** Order by
   evidence: (i) CSX overlap tier (has the fixture above; breaking at
   k=2 with bit-exact inputs suggests one shallow constant — find WHICH
   threshold kills the overlap detection before designing); (ii) the
   singular-tier `tol_f`/`1e-11` family (P1b experiment table + its
   fixtures-by-rescale recipe); (iii) corrector 1e-14s; (iv) GJK
   (soundness at any scale; Cython — may need its own package; the
   kickoff invariant "no Cython work" from P1 does NOT carry over
   here, but scope it explicitly with the user). Derived-envelope
   pattern to copy: L1 margin and `_ccx_exactness_context`.
3. **Automated invariance guarantee.** Grow
   `tests/test_bez_ssx5_invariance.py` into a structural-class ×
   random-similarity sweep (fixed seeds): classes = transversal open,
   closed loop, tangential curve, isolated tangency,
   boundary-coincident (THIS case), 2-D overlap region, cusp curve —
   one representative each; transforms sweep magnitudes ~1e-6..1e6 and
   atol proportionally; assert result equivalence through the map.
   Singular classes enter the sweep AS their tiers get derived
   envelopes (today they'd fail — that's the point: the sweep is the
   burn-down's acceptance gate).
4. **Frame retirement decision (with the user).** Once the predicates
   self-normalize, the windowed frame stops being load-bearing:
   keep as a conditioning optimization or retire. Do not decide early;
   cases 6/11 gates must stay green throughout (they currently DEPEND
   on the frame).

Interaction with open items: **P2** (case-11 knob-unreachable tier,
19,798-cell fixed point — instrumentation plan in the 2026-07-20
kickoff §P2) is still owed and may itself be a calibrated-cap member —
the audit will likely find its tier; sequence P2 vs burn-down with the
user at session start. **P1b** is subsumed by this program (its issue
file becomes the singular-tier section of the audit). **P1c**'s
accounting audit and GJK items fold in likewise.

## State and discipline

- Branch `ssx5-invariance` @ `497aa84`, off `tiny` (81dd297), NOT
  merged, user chose keep-as-is. All P1 gates green at HEAD (116
  singular / 28 invariance / 44 nssx5 / 95 ccx-csx / bez harness 8×100%
  / NURBS harness 9 OK / adversarial review 0 majors) — but the user
  case above fails at HEAD, so do NOT merge before deciding how it's
  covered. Suggested: new branch `ssx5-derived-envelopes` off
  `ssx5-invariance` (P1 stays a coherent reviewable unit).
- Env: `.venv/bin/python` (3.14); macOS, NO `timeout` command; engine
  file huge — read only anchored regions; line anchors above verified
  at `497aa84`.
- Invariants: never fix by loosening; the derived-envelope mandate
  (top of this file); fixture-first (no fix without a reaching
  fixture — L50); every destructive tolerance test keeps its xyz guard
  (handoff §3 ladder); adversarial review after implementation is
  house-mandatory; commit the user fixture as a test WITH its tier fix
  (a failing test cannot land on the suite — xfail-pin it if the fix
  is deferred).
- Read-first for a fresh session: this file; memory
  `project_ssx5_p1_invariance_shipped.md` (what the frame is and why);
  `docs/superpowers/issues/2026-07-21-ssx5-p1b-…` and `…-p1c-…` (the
  threshold inventories); spec
  `docs/superpowers/specs/2026-07-21-ssx5-invariance-normalization-design.md`
  Amendments 1-2 (why the frame is windowed and band-targeted);
  handoff `docs/superpowers/plans/2026-06-10-ssx5-singular-cases-handoff.md`
  §3/§7 (tolerance ladder, monkeypatch playbook).

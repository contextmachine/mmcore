# CCX 3D near-miss contact: `tol` is not an acceptance distance — typed `exact|tolerance` tier for isolated intersections

**Status:** IMPLEMENTED 2026-08-19 on branch `l62-ccx-tolerance-tier`
(engine commit `85a8a06`); the two strict xfails fired and are unpinned
(grid 25/25, dedup == 25); all §5 gates green at the branch head.
**Ledger ID:** L62 (confirmed free before first use).

**Owner decisions made during the implementation session (2026-08-19),
superseding the corresponding parts of §4 below:**

1. **No band outcome exists** *(supersedes §4.6(c))*: only an overlap is a
   long touch, and an overlap must begin and end at a curve-domain
   endpoint (L47 gate, unchanged).  Every other case is either ONE
   isolated tangent contact — the minimum-distance pair of the compact
   sub-`tol` region — or, when the curves cross in and immediately back
   out, the k exact crossings, distinguished at high precision.  The
   pending L47 band-bar / K×median question therefore does not apply to
   the isolated tier at all.
2. **The tier applies in 2D**, same predicate (no dimension gate; in 2D
   the transversal near-miss configuration does not exist, so the
   canonical 2D near-miss is the tangent graze — pinned in
   `tests/test_ccx4_tolerance_tier.py`).
3. **The parallel-planes accept-path pins re-scoped** *(owner approved the
   analysis)*: their membership form contradicted §1 on 11/15 and 1/3
   parameter combos.  Re-pinned in `tests/test_bez_ccx4.py` to (a)
   membership tracking the REALIZED gap against `atol` in both directions
   at every world position, and (b) a resolvable nonzero gap never
   carrying `certification='exact'` — the phantom-root guard now lives on
   the tag, where `test_absent_axis_is_checked_not_skipped` pins it at
   unit level.
4. **CSX opt-out** *(resolves the §4.5 open question)*: `bez_ccx` gained
   `tolerance_tier=True`; the nested CSX boundary-zero call passes
   `False` (exact-only, byte-identical legacy) — whether CSX wants its own
   isolated-contact tier stays a separate ledger item.

**Implementation notes beyond the §4 design (measured during the session):**

- The closed inequality is enforced at measurement resolution: `gap == tol`
  measures `atol ± roundoff` off the net, so membership accepts within the
  certified envelope `eps_d` of the boundary, and the min-of-net/Lipschitz
  prune bars carry the same envelope slack (`gap == tol` was otherwise
  lost to a 1-ulp coefficient rounding).
- The measurement envelope's SOURCE term is rational-only: polynomial
  `D_ij = P_i − Q_j` is one correctly rounded subtraction of exact inputs
  (Sterbenz), so a world translation cannot inflate it; rational
  cross-products round at world scale, which is where the typed
  cannot-decide tail is genuinely reachable (pinned end-to-end with a
  |T|=1e12 unequal-weights fixture).
- Descent cost needed three structural rules, all measured: coarse
  terminal stops for zero-free cells (wholly-in-band hull bound;
  Hessian-PD unique-minimizer via `_check_uniqueness_2d` one level up) and
  anisotropic refinement for curve pieces collapsed under half the dedup
  radius.  A shallow rational ellipse–spline crossing went 545k → 587
  cells; the 11-curve 3D grid went 46 s → 2.4 s while keeping 25/25.

**Adversarial review (2026-08-19, 35-agent workflow, 5 lenses × 2-skeptic
verification): 15/15 findings sustained, all reproduced, all fixed** in the
follow-up commits on the branch.  The load-bearing corrections:

- The net measurement envelope is GLOBAL (extent²-scaled) and reached an
  accept path: at |ctrl| ≈ 3e3 with atol=1e-3 it certified gaps up to
  1.68·atol as members and armed the cannot-decide tail ~6 orders early.
  Membership now uses the sharper of the net and a direct-evaluation
  measurement (`_measure_contact`), whose envelopes fail in complementary
  regimes (extent² vs world position).
- The tolerance minimizer is basin-clamped to its cell (unbounded GN
  jumped super-tol ridges and lost the abandoned basin's contact) —
  deliberately unlike the exact tier's unbounded Newton doctrine.
- The tier never stands down call-wide on overlap-class/band evidence
  (that deleted members with topology claimed complete); jurisdiction is
  enforced in the drain: certified overlap spans, plus band-anchor
  connectivity armed only under crossing evidence (= the never-merge
  boundary, no wider).
- Component identity is decided by CONNECTIVITY only, and the walk follows
  the inversion pairing with an arrival check (straight (u,v) chords
  split curved components; a 3D-radius shortcut merged disconnected
  ones).  Grid-verified: exactly one contact per component at the argmin.
- The endpoint pre-filter bar gained the same envelope slack as every
  other level-atol bar (a rotated gap==atol terminus contact was lost
  134/300 times to 1-ulp coefficient rounding).
- Adapter: a per-candidate typed cannot-decide is recorded on
  `status['uncertified_contacts']` (global params + curve indices) and the
  scan CONTINUES — it no longer aborts unrelated span pairs; the
  NURBS-level seam re-verification carries an operand slack so it cannot
  reverse closed-boundary decisions.
**Pinned at:** `tests/test_nccx4.py` — two `xfail(strict=True)` on
`TestNurbsCCXMultiple3D::{test_ground_truth,test_no_span_boundary_duplicates}`.
Because they are strict, implementing the fix makes them FIRE — remove the pins as
part of the fix commit; they are the acceptance tests.
**Origin commit:** `5d05ddc` (2026-07-10). **Discovery:** 2026-08-16/18, during the
restructure (`docs/superpowers/plans/2026-08-16-restructure-phase2-execution.md` §10,
follow-up #1).

---

## 0. Session-start prompt (copy-paste)

```
Implement L62: the CCX tolerance tier for isolated intersections.

READ FIRST, in order — then say what you're going to do before touching code:
1. docs/superpowers/issues/2026-08-18-ccx-3d-near-miss-tolerance-tier.md — the
   whole spec. §1 (membership contract), §4.2 (strict-envelope jurisdiction) and
   §4.6 (sub-level-set component discriminator) are OWNER-DECIDED: do not
   relitigate them, implement them.
2. git show 5d05ddc -- mmcore/numeric/intersection/ccx/_bez_ccx4.py — the
   doctrine being re-scoped (not reverted).
3. mmcore/numeric/intersection/ccx/_bez_ccx4.py — bez_ccx docstring (two-phase
   architecture), _strict_residual_ok (~:253), the L47 tolerance-overlap
   certificate (~:553), the typed-outcome plumbing (~:1266-1330).
4. mmcore/numeric/bern_sq_dist.py + mmcore/numeric/_bez_closest_point.py — the
   minimum-bounding machinery the tier should reuse, not reinvent.
5. docs/superpowers/specs/2026-07-12-theorem-first-overlap-methodology.md §2.

DECIDED — inherit, don't re-derive:
- Membership is d_min <= tol (closed inequality) at every tol; topology correct
  at every tol. The strict roundoff envelope has NO membership role: it grades
  the metadata tag, powers the sub-atol topology guards, and covers the
  |coords| >~ atol/eps straddle tail (typed cannot-decide, never a guess).
- Classify by connected components of {D^2 <= tol^2}: boundary-anchored both
  ends -> L47 overlap path; compact interior -> isolated tier (certified zeros
  inside -> exact roots only; zero-free -> exactly ONE contact at the certified
  argmin; elongated vs the param-tol mapping -> typed grazing band); single
  boundary touch -> endpoint contact (Phase 1 lifted from level 0 to tol^2).
- certification: 'exact'|'tolerance' is metadata only.

ASK THE OWNER when you reach them (do not decide alone):
- the band-bar: the relative rule (flip flanks <= K x median in-band residual
  => coincidence noise) — acceptance + the value of K;
- the elongation bar separating point contact from typed band;
- whether the tier applies in 2D (expected: yes, same predicate).

GATES — all must hold before calling it done:
- tests/test_nccx4.py: the two strict xfails FIRE when the fix works — remove
  the pins; test_ground_truth 25/25, dedup count == 25.
- A parameterized tol-scaling test over (gap, tol): exactly one intersection
  iff tol >= gap, count never exceeds one (values are instances, not constants).
- A translated large-offset tolerance-contact case (acceptance must be
  translation-invariant; do not reopen the hole 5d05ddc closed).
- tests/test_bez_ccx4.py stays green unchanged — the overlap/point boundary and
  the never-merge invariants must not move.
- Full suite: python -m pytest tests -q -m "not slow" from the REPO ROOT,
  non-increasing; tools/check_imports.py and tools/check_layering.py exit 0.

DISCIPLINE:
- Dedicated branch off tiny; L62 in commit subjects; confirm L62 is the next
  free ledger ID first (last known used: L61).
- OUT OF SCOPE, do not touch: _bez_ssx5.py / _deflate.py (Q13/Q14); the
  linux-scoped FP-sensitivity xfails in test_csx4_exactness_contract.py /
  test_csx_overlap_tier.py; centering removal (separate empirical A/B item —
  the tier must work WITH the current certificate).
- tests/test_nccx4.py builds fixtures by exec'ing the head of
  examples/ccx/multiple_int_3d.py, split at the line
  "from mmcore.numeric.intersection.ccx import" — keep that file's data section
  and marker stable.
- Env: .venv/bin/python (3.14); pytest from the repo root; example viewers run
  from the poetry venv. CI on push: 15-leg build + gates (~50 min).
```

---

## 1. The contract (normative — owner formulation, 2026-08-18)

Intersection existence is tolerance-determined, the standard CAD semantics:

> For a curve pair whose true minimum distance is `d_min`: at any `tol >= d_min`
> the pair has exactly one isolated intersection there; at any `tol < d_min` it
> has none. Topology — the count and structure of the result — must be correct
> at every `tol`. Concretely: curves 5e-4 apart → `tol` of 5e-4 / 1e-3 / 1e-2 →
> a single isolated intersection; `tol` of 1e-4 / 1e-5 → not an intersection.
> It cannot be any other way.

The exact-zero behavior shipped by `5d05ddc` is therefore a **defect of the
public contract**, not an alternative contract. What `5d05ddc` got right — and
what must survive the fix — is orthogonal to membership: raw Newton residuals
and `atol`-sized numerical artifacts at large coordinates must never masquerade
as geometry. Its strict certificate is **re-scoped**, not kept as a membership
rule: it certifies the *measurement* of `d_min` (scale-robust,
translation-invariant per `e0ab4a0`); `tol` alone then decides membership via
`d_min <= tol` (closed inequality). False-root protection and tolerance
acceptance are both mandatory and neither trades against the other.

One consequence the formulation forces: when the certified error envelope of the
`d_min` measurement straddles the `tol` boundary — the engine genuinely cannot
decide membership — the outcome must be **typed** (the `uncertified_overlap_span`
pattern from L47), never a silent accept or silent reject.

## 2. Measurements (all reproducible; commands in §6)

**Minimal repro** — two straight lines crossing in XY, separated in Z by `gap`,
`nurbs_ccx(tol=1e-3)`:

| gap at crossing | isolated found | status |
|---|---|---|
| 0.0 | 1 | complete |
| **1e-9** (tol/1e6) | **0** | complete |
| 1e-6 / 5e-4 / 9e-4 | 0 | complete |

A gap one-millionth of the tolerance erases the intersection, and the engine
reports its answer as complete. Raising `tol` to `1e-2` (20× a real 4.8e-4 gap)
still finds nothing — `tol` does not participate in isolated-point acceptance at all.

**Real data** — the 11-curve 3D grid (`examples/ccx/multiple_int_3d.py`, ground
truth `tests/expected_nurbs_ccx_01.json`, 25 curve-0-filtered entries): all 25 are
geometrically real; the 5 found have curve–curve distance exactly 0.0; the 20
missed have distance **4.4e-6 … 5.0e-4**. `nurbs_ccx_multiple` is consistent with
pairwise `nurbs_ccx` (both find the same 5 — not an aggregation bug).

**History** (overlay-bisect of all 20 pre-restructure `_bez_ccx4.py` versions in a
built worktree; probe = the line pair above + the grid):

| engine era | near-miss @5e-4, tol 1e-3 | grid total |
|---|---|---|
| `5469f06` (03-28) … `1d9a511` (05-26) | accepted | **38** |
| `5d05ddc` (07-10) … today | rejected | **5** |

The two commits are consecutive in the file's history; `5d05ddc` is the exact origin.

## 3. Why it changed — and why the change must be kept

`5d05ddc` ("fix(ssx5): harden singular and rational intersections") introduced
`_strict_residual_ok` and deleted the old acceptances
(`norm(G) < atol`, `dist(pt1, pt2) < atol`). Its docstring states the doctrine:

> `atol` is a search/resolution tolerance, not membership in the exact near-root.
> Neither Newton's step size nor `atol` can accept the result; the component-wise
> residual certificate above is the sole membership.

Motivation: `atol`-acceptance produced **false roots at large coordinate scales**
(an `atol`-sized gap at x≈1e4 is roundoff, not geometry). `e0ab4a0` (07-26,
"exactness certificates must not decay with world position") reinforced the same
principle. The false-positive protection is wanted; the collateral was silently
losing every true within-tol 3D contact.

**Why nobody saw it:** the only test pinning the 25-hit contract had been
error-masked since `c14fd3e` (2026-06-09) — a dead import inside the example file
the test `exec`s — one month *before* the behavior changed. Repaired 2026-08-18
(`b41a0e5`); the tests ran for the first time since June and exposed the gap.

## 4. Fix design — typed tolerance tier for isolated contacts

Mirror L47's overlap contract at the isolated-point level:

1. **Engine (`_bez_ccx4.py`)**: today a cell whose squared-distance Bernstein net
   has a certified positive minimum is pruned as "no zero". Add a branch: if the
   certified minimum satisfies `min(D²) <= atol²`, descend/polish to the
   **minimizer** (not a root — Newton on the gradient of D², or reuse the
   closest-point machinery: `bern_sq_dist` bounds + the band logic of
   `_bez_closest_point`) and emit the contact
   `{u, v, point, certification: 'tolerance', d_min}`. Certified zeros carry
   `certification: 'exact'`. **The tag is metadata only** (useful diagnostics,
   L47 symmetry) — membership is `d_min <= tol` and nothing else; dropping the
   tag entirely is a one-line owner call, the contract does not depend on it.
   A cell whose certified envelope straddles `tol` yields the typed
   cannot-decide outcome of §1, not a guess.
2. **Envelope discipline (the design law):** the acceptance predicate is
   `d_min <= atol` with `atol` the caller's geometric tolerance — an operand
   envelope, not a bare threshold. The *certification* of `d_min` itself must be
   translation-invariant (the `e0ab4a0` requirement): compare against the
   sq-dist net's certified bounds, never against a raw Newton residual, so a
   tolerance contact at x≈1e4 does not reappear as a false positive. This is the
   heart of the fix — the tier must not reopen the hole `5d05ddc` closed.

   **Post-L62 jurisdiction of the strict roundoff envelope (owner, 2026-08-18):**
   `5d05ddc` made `_strict_residual_ok` "the sole membership" gate (its own
   docstring) — that conflation is the defect. After L62 the strict envelope has
   exactly three jobs and NO role in membership: (a) grading the metadata tag
   (`'exact'` = agreement inside the roundoff envelope); (b) sub-`atol`
   TOPOLOGY — the never-merge invariants (distinct crossings inside a band,
   valley chains) need resolution finer than `atol` and this is where strict
   precision is legitimately load-bearing; (c) the straddle guard — note the
   scale: `d_min` measurement noise ~ `eps*|coords|` threatens an
   `atol`-membership decision only for `|coords| >~ atol/eps` (≈1e12 for
   atol=1e-3), so the typed cannot-decide outcome is an exotic-tail guard, not
   a common path. Membership is `atol` + the topological criteria, period.
3. **Reporting midpoint vs pair:** a tolerance contact has two witness points
   (one per curve); report the parameter pair of the minimizer and `point` as the
   chord midpoint (consistent with how the old engine populated the ground
   truth — verify against `expected_nurbs_ccx_01.json` values, which store one
   3D point per contact).
4. **Dedup:** a shallow minimum valley can yield several sub-`atol` minima per
   cell neighborhood; dedup in (u,v) with the existing `_is_duplicate` machinery
   sized by the param-tol mapping (`_nurbs_param_tol`), not by a bare constant.
5. **Adapter (`_nccx4.py`)**: surface `certification` in the structured dtype (or
   a parallel field) for `nurbs_ccx` and `nurbs_ccx_multiple`; default output
   includes both tiers (that is the contract the ground truth pins). Decide
   whether an opt-out (`exact_only=True`) is wanted.
6. **Structural discriminator vs the overlap tier (owner, 2026-08-18):** classify
   by the connected components of the sub-tolerance region `{D² <= tol²}` of the
   sq-dist patch — the boundary rule lifted from the zero level-set to `tol²`:
   - *Exact overlaps can only terminate at a curve-domain end* (theorem at level
     zero: analytic continuation), so an overlap valley enters and exits through
     the patch **boundary**. At tolerance level this stops being a theorem
     (interior-ended near-coincident bands exist) and survives as L47's
     admissibility **gate**: only boundary-anchored components are promoted to
     overlaps; interior-ended bands stay typed-partial.
   - A component **compactly contained in the interior** is the isolated-contact
     signature — the dual of the overlap rule. Rules per component:
     (a) contains certified zeros → resolved by the exact machinery (k crossings
     = k exact roots), no tolerance contact emitted — the tiers cannot
     double-count by construction; (b) zero-free and compact → **exactly one**
     tolerance contact at the certified argmin ("one component = one
     intersection" is the general form of the count guarantee; dedup becomes
     structural, with only decomposition-seam stitching left as in
     `test_no_span_boundary_duplicates`); (c) zero-free but elongated against
     the param-tol mapping → a tangential grazing band, a typed outcome carrying
     its (u,v) extent — lands on the L47 band-bar decision still pending for the
     woven twins.
   - A component touching the boundary at a single point/edge is an **endpoint
     contact** (curve terminus within `tol` of the other curve): Phase 1's
     boundary analysis extended from level 0 to level `tol²`.
   The architecture keeps its shape: Phase-1 boundary doctrine, cell
   classification and component logic run unchanged one level up; the classifier
   answers at two levels (0 and `tol²`) instead of one. (The grid data is all
   transversal; the L-junction/twin fixtures in `test_bez_ccx4.py` guard the
   overlap side.)

**Open sub-questions for the session** (decide, don't inherit): does the tier
apply in 2D as well (a 2D near-miss inside `atol` is currently also invisible —
probably yes, same predicate, cheap)? And does CSX want the same isolated-contact
tier afterwards (same classifier family; separate ledger item if yes)?

**Related empirical item (owner-flagged 2026-08-18, own ledger entry when picked
up): is CENTERING needed at all?** The cluster-4 record says scale-only reframing
is covariant and centering is the operation that breaks predicates; the
2026-07-26 review's shipped defects clustered in the centering-compensation
machinery; and agent traces reportedly showed byte-identical results with and
without centering in at least some cases. The physical argument FOR it —
evaluation noise drops from `eps*|world|` to `eps*|local extent|`, enabling
sub-envelope resolution far from the origin (the parallel-planes-at-X0=1e6
merge bug) — applies only to the strict envelope's post-L62 jobs (grading,
sub-atol topology), never to membership. Settle by measurement: A/B the
certificate with centering removed across the exactness/invariance suites and
the far-origin fixtures (the |T|=1e6 float-built-subcurve case at
`_strict_residual_ok`'s comment). If byte-identity holds, delete the centering
apparatus as defect-prone complexity.

## 5. Gates

- The two strict xfails in `tests/test_nccx4.py` flip to acceptance tests
  (remove pins; `test_ground_truth` 25/25, dedup count == 25).
- `tests/test_bez_ccx4.py` (incl. L47 overlap-tier cases) stays green — proves
  the overlap/point boundary didn't move.
- Translation-invariance: every new acceptance must hold under the
  `test_csx4_exactness_contract.py`-style translated fixtures — add one for the
  tolerance tier at large offsets (e.g. the 5e-4-gap line pair at x += 1e4).
- The minimal line-pair probe (§6) as a unit test: gaps {0, 1e-9, 5e-4, 9e-4}
  all report exactly one contact at `tol=1e-3` (typed exact for gap 0,
  tolerance otherwise); gap 2e-3 reports none.
- Tol-scaling: for the same pair, membership must track `tol` exactly —
  one intersection whenever `tol >= gap`, none whenever `tol < gap`, count
  never exceeding one. (The values in the owner's example — gap 5e-4 with
  `tol` in {5e-4, 1e-3, 1e-2} vs {1e-4, 1e-5} — are arbitrary instances of
  the law, not special constants: parameterize the test over (gap, tol).)
- Full suite `-m "not slow"` non-increasing.

## 6. Repro commands

```bash
# minimal line-pair probe (adapt gaps as needed)
.venv/bin/python - <<'EOF'
import numpy as np
from mmcore.nurbs._nurbs_eval import NURBSCurveTuple
from mmcore.numeric.intersection.ccx import nurbs_ccx
def line(p0, p1):
    return NURBSCurveTuple(order=2, knot=np.array([0.,0.,1.,1.]),
                           control_points=np.array([p0, p1], float),
                           weights=np.array([1., 1.]))
for gap in (0.0, 1e-9, 5e-4, 9e-4, 2e-3):
    iso, ovl, st = nurbs_ccx(line([-1,0,0],[1,0,0]), line([0,-1,gap],[0,1,gap]), tol=1e-3)
    print(f"gap={gap}: isolated={0 if iso is None else len(iso)} complete={st.get('complete')}")
EOF

# the pinned grid tests (currently 12 passed / 2 xfailed)
.venv/bin/python -m pytest tests/test_nccx4.py -q

# the visual: 11-curve grid, only the 5 exact crossings get markers today
python examples/ccx/multiple_int_3d.py            # (poetry venv for the viewer)
```

## 7. Read first

1. `git show 5d05ddc -- mmcore/numeric/intersection/ccx/_bez_ccx4.py` — the
   doctrine change; understand `_strict_residual_ok` and
   `_vector_residual_hull_excludes_zero` before touching anything.
2. `mmcore/numeric/intersection/ccx/_bez_ccx4.py` — two-phase architecture
   (docstring of `bez_ccx`), the sq-dist classification path, `DownCounter`
   budgets, the L47 overlap certification (`a867707`) as the pattern to mirror.
3. `mmcore/numeric/bern_sq_dist.py` + `mmcore/numeric/_bez_closest_point.py` —
   the minimum-bounding machinery the tolerance tier should reuse.
4. `e0ab4a0` — translation-invariance of certificates; the tier must satisfy it.
5. `tests/expected_nurbs_ccx_01.json` + `tests/test_nccx4.py` fixtures (note:
   `_load_3d_curves` execs the head of `examples/ccx/multiple_int_3d.py`, split
   at the `from mmcore.numeric.intersection.ccx import` line — keep that file's
   data section and marker stable).

## 8. Branch discipline

Dedicated branch off `tiny` (not `tiny` directly). CI gates run on PR/push:
import-health, layering, full suite `-m "not slow"` (~40 min on the ubuntu/3.12
leg). Note `_bez_ssx5.py`/`_deflate.py` remain out of scope (Q13/Q14), and the
two linux-scoped FP-sensitivity xfails
(`test_csx4_exactness_contract.py` / `test_csx_overlap_tier.py`) are a separate
open item — don't fold them into this change.

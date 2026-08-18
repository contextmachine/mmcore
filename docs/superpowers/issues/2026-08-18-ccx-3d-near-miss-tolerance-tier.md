# CCX 3D near-miss contact: `tol` is not an acceptance distance — typed `exact|tolerance` tier for isolated intersections

**Status:** OPEN — measured, bisected, pinned; fix designed below, not started.
**Proposed ledger ID:** L62 *(confirm against the ledger before first use — last known used is L61).*
**Pinned at:** `tests/test_nccx4.py` — two `xfail(strict=True)` on
`TestNurbsCCXMultiple3D::{test_ground_truth,test_no_span_boundary_duplicates}`.
Because they are strict, implementing the fix makes them FIRE — remove the pins as
part of the fix commit; they are the acceptance tests.
**Origin commit:** `5d05ddc` (2026-07-10). **Discovery:** 2026-08-16/18, during the
restructure (`docs/superpowers/plans/2026-08-16-restructure-phase2-execution.md` §10,
follow-up #1).

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
6. **Interaction with the overlap tier:** an extended near-parallel approach
   inside `atol` is L47's residual-overlap territory, not a point contact — the
   classifier must hand extended sub-`atol` valleys to the overlap path and only
   emit point contacts for locally-isolated minima. (The grid data is all
   transversal; the L-junction/twin fixtures in `test_bez_ccx4.py` guard the
   other side.)

**Open sub-questions for the session** (decide, don't inherit): does the tier
apply in 2D as well (a 2D near-miss inside `atol` is currently also invisible —
probably yes, same predicate, cheap)? And does CSX want the same isolated-contact
tier afterwards (same classifier family; separate ledger item if yes)?

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

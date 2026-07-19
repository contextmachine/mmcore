# NURBS SSX over bez_ssx v5 (`_nssx5.py`)

Design approved 2026-07-19 (brainstorming session, branch `nurbs-ssx-over-bez_ssx5`).

## Problem

`bez_ssx` (`mmcore/numeric/intersection/ssx/_bez_ssx5.py`) is a hardened
surface-surface intersection solver for single rational Bézier patches:
typed branches, points, singularities, overlap regions, schema-v2 status.
There is no NURBS-level entry point over it. The existing public
`nurbs_ssx` (`_ssx4.py`) drives the old v4 recursive tracer, merges
fragments with param-only tolerances (no xyz guard), drops `kind` on
merge, and knows nothing of singularities, regions, or status. This
closes the "SSX redesign (deferred)" line of
`2026-03-17-intersection-algorithm-redesign.md` at the NURBS layer.

## Goal

A NURBS×NURBS wrapper in the proven `_nccx4.py` / `_ncsx4.py` adapter
style: decompose to Bézier patches, run `bez_ssx` per BVH-candidate patch
pair under a shared work ledger, remap to global knot domains, and
assemble one certified NURBS-level result — stitched branches, deduped
points and singularities, unified overlap regions, aggregated status.

## Decisions (user, 2026-07-19 — do not re-ask)

1. **Result shape**: the `bez_ssx` dict schema verbatim, parameters in
   global knot domains.
2. **Native-layer contract**: polylines only. No curve fitting, no kwargs
   that change the return schema. Fitting belongs to a future casual
   (Fusion/Rhino-style) API layer.
3. **Overlap regions**: full unification across patch pairs by seam-rim
   dissolution + rim chaining. No re-certification — tiles are already
   residual-verified; certification dicts merge conservatively.
4. **Periodic seams**: stitch across C0-closed axes; the polyline keeps
   both seam preimages as consecutive vertices (same xyz, parameter
   jump); `closed=True` when the chain closes.
5. **Naming**: module `mmcore/numeric/intersection/ssx/_nssx5.py`,
   function `nurbs_ssx`, exported from `ssx/__init__.py` as
   `nurbs_ssx_v5`. Legacy `nurbs_ssx` untouched.
6. **Architecture**: Approach A — flat pair sweep + one global assembly
   pass. `_bez_ssx5.py` is not modified.

## Module structure

```
mmcore/numeric/intersection/ssx/_nssx5.py      # the wrapper (new)
mmcore/numeric/intersection/ssx/__init__.py    # + nurbs_ssx_v5 export (1 line)
tests/test_nssx5.py                            # fast unit suite (new)
examples/ssx/nurbs_ssx5_coverage_check.py      # objective harness (new)
```

Imports (nothing else in the repo changes):

- `bez_ssx` from `._bez_ssx5`; dataclasses `SSXBranch`, `SSXPoint` from
  `._ssx4`; `SSXSingularity` from `._bez_ssx5`; `SSXOverlapRegion` from
  `._ssx5_overlap`.
- `decompose_surface` (`mmcore.geom._nurbs_knots`); `_nurbs_to_tuple`,
  `to_homogeneous_2d`, `NURBSSurfaceTuple` (`mmcore.geom._nurbs_eval`);
  `nurbs_surface_param_tolerance` (`mmcore.geom._nurbs_param_tol`).
- `AABB`, `build_bvh`, `bvh_intersect` (`mmcore.geom.bvh.lbvh`).
- `REASON_*` vocabulary + `SoftWorkBudget`
  (`mmcore.numeric._work_budget`); `reject_unknown_kwargs`
  (`mmcore.numeric.intersection._adapter_status`).

## Public contract

```python
def nurbs_ssx(surf1, surf2, atol=1e-3, **expert_knobs) -> dict
```

- `surf1`, `surf2`: `NURBSSurfaceTuple` or Cython `NURBSSurface`
  (converted via `_nurbs_to_tuple`).
- Returns `{'branches', 'points', 'singularities', 'overlap_regions',
  'unresolved_regions', 'complete', 'status': {'reasons', 'work'}}` —
  the `bez_ssx` schema. `stuv = (s, t, u, v)`: `(s, t)` in
  `surf1.interval()`, `(u, v)` in `surf2.interval()`.
- Branches carry `curve = (stuv (N,4), xyz (N,3))` polylines;
  `curve_xyz/curve_st/curve_uv` stay `None`.
- Incompleteness is always soft: partial certified output +
  `complete=False` + typed `reasons`; exceptions only for caller errors
  (unknown kwargs via `reject_unknown_kwargs`, invalid inputs).
- Rationality: if either surface has a non-unit weight, all patches go
  through `to_homogeneous_2d` and `bez_ssx(..., rational=True)`;
  otherwise Cartesian nets with `rational=False` (the `_ncsx4` rule).

### Expert knobs

Same names as `bez_ssx`; two forwarding classes:

| Knob | Semantics |
|---|---|
| `max_cells`, `max_csx_calls`, `max_output_items` | **Aggregate ledgers**, shared across all pairs. Default = bez default (250 000 / 10 000 / 1 024) × `max(1, n_candidates)`. An explicit value is an absolute aggregate promise. Per pair: `min(bez_default, remaining)`. |
| `max_depth`, `max_xyz_step`, `csx_max_cells`, `boundary_csx_max_cells`, `csx_max_results` | Forwarded verbatim to every call. |
| `max_postprocess_work` | Aggregate cap for the **wrapper's own** assembly ledger (default `None` → the aggregate cells allowance, `SoftWorkBudget`'s rule). Not forwarded per-call — each `bez_ssx` keeps its own internal default. |

## Pipeline

1. **Decompose & index.** `decompose_surface(surf, "uv")` on both →
   Bézier patches (`NURBSSurfaceTuple`), `.interval()` =
   `((a0,a1),(b0,b1))` in the parent knot domain. One LBVH per surface
   over control-point AABBs offset by `atol`;
   `bvh_intersect(tree1, tree2, exact=False)` → candidate pairs, sorted
   by `(i, j)` for determinism.
2. **Per-pair call.** For each candidate while ledgers are positive:
   `bez_ssx(P1, P2, atol=atol, rational=..., **caps)`. Afterwards read
   `result['status']['work']` and decrement the shared ledgers by actual
   `cells_processed`, `csx_calls`, `output_items`. Empty ledger ⇒ every
   remaining pair is skipped and contributes one `unresolved_regions`
   entry (its global stuv box, `reason=REASON_WORK_BUDGET`) and
   `complete=False`.
3. **Remap local→global** per pair (affine per axis,
   `g = a0 + (a1−a0)·x_loc`): branch stuv paths, point stuv,
   singularity `stuv`/`stuv_mate`/`samples`, region
   `uv1_loops`/`uv2_loops`/`interior_stuv`, unresolved-region boxes.
   Results accumulate in flat lists tagged `(pair_id,
   local_branch_index)`; `branch_links` and region `boundary` refs keep
   that identity until final assembly rebuilds them.
4. **Assembly** (below), all scans charged to the wrapper's postprocess
   ledger; exhaustion ⇒ `REASON_POSTPROCESS_CAP`, `complete=False`,
   partial output returned.
5. **Status aggregation.** `complete` = AND of executed pairs'
   `complete` AND no skipped pairs AND no wrapper truncation.
   `reasons` = sorted union (pairs + wrapper). `work` = counters summed
   (incl. wrapper postprocess charges) against the aggregate caps;
   `cell_counts` merged key-wise.

## Tolerances (NURBS-level ladder — handoff §3 rule)

`(ptol_s, ptol_t) = nurbs_surface_param_tolerance(surf1, atol)`,
`(ptol_u, ptol_v)` likewise for `surf2`.

- **Matching / unification**: per-axis `|Δ| ≤ 4·ptol_axis` AND
  `‖Δxyz‖ ≤ 2·atol`.
- **Destructive dedup**: per-axis `|Δ| ≤ 1·ptol_axis` AND
  `‖Δxyz‖ ≤ atol`.
- Every destructive test carries the xyz guard. On C0-periodic axes both
  tests compare `Δ` modulo the domain span (wrap-aware).
- Periodicity per axis is detected once per surface at NURBS level:
  first/last control-point row (and weight row) equality, the
  `_ncsx4._is_seam_duplicate` predicate.

## Branch assembly

Rim branches referenced by any tile's region `boundary` are routed to
the region assembler (below) and excluded from steps 2–4 here.
Overlap-curve branches not referenced by a region participate normally.

1. **Duplicate-fragment removal** (before stitching, so containment
   duplicates cannot inflate junction clusters): sort fragments by arc
   length descending; drop a fragment when every sample lies within
   `2·atol` of a longer kept fragment's polyline (the Bézier-level
   geometric-containment rule, applied cross-pair). This handles curves
   traced twice because they run along a knot line.
2. **Endpoint stitching**: union-find over fragment endpoints with the
   matching predicate (wrap-aware). Concatenate head-to-tail with
   orientation flips; at an ordinary joint the duplicated vertex
   collapses to one (destructive rule); across a periodic wrap both
   seam preimages stay as consecutive vertices (same xyz, parameter
   jump).
3. **Junctions**: a match cluster with more than two fragment ends
   (X-junction, e.g. at a C3 self-intersection) is never chained
   through — fragments terminate there; the singularity entry marks the
   point.
4. **Kind**: fragments of one physical branch share `kind`; a genuine
   kind conflict inside a cluster leaves those fragments unmerged rather
   than guessing. `overlap` flags OR together on merge.
5. **Closing**: a chain whose own ends match (wrap-aware) becomes
   `closed=True`, with the periodic vertex-pair contract when the
   closure crosses a seam.

## Points

Concatenate → destructive dedup (wrap-aware) → global points-on-branch
filter: drop any point whose xyz lies within `4·atol` of a final branch
polyline (the Bézier-level constant). Catches cross-pair cases where one
pair's isolated point lies on the neighbour pair's branch.

## Singularities

- Point-like kinds (`tangent_point`, `cusp`, `self_intersection`):
  dedup across pairs by same `kind` + matching predicate on `stuv`
  (wrap-aware) + `xyz ≤ 2·atol`; for `self_intersection`, `stuv_mate`
  must match under the same rule too.
- `cusp_curve`: samples remapped; entries kept per-pair (diagnostic
  clouds, not chained topology); collapse only near-identical clouds
  (pairwise sample containment within `2·atol`).
- `branch_links`: recomputed from scratch against the final branch list
  by the L11 rule — nearest polyline vertex of each branch passing
  within the Bézier-level linking distance. Reuse `_bez_ssx5`'s linking
  helper/constant (import it or mirror it exactly; pin the symbol during
  planning), never translate indices through stitching arithmetic.

## Overlap regions (full unification)

Per-tile regions arrive with globally remapped loops and
`(pair_id, local_branch_index)` rim refs.

a. **Seam-rim detection**: a rim segment is a seam candidate when all
   its samples lie within `4·ptol_axis` of one interior-knot coordinate
   of its patch edge (an edge shared with a neighbouring tile). It
   dissolves when the adjacent tile contributes a partner segment
   matching endpoint-and-midpoint under the matching predicate with
   opposite orientation. Both partners' branches are dropped.
b. **Chaining**: surviving rim segments chain head-to-tail (orientation
   via `reversed` flags) into the union's outer/hole loops;
   `uv1_loops`/`uv2_loops` are rebuilt by concatenating the surviving
   segments' sample runs in loop order (per-segment uv1/uv2
   synchronization is preserved by construction).
c. **Certification merge** (conservative, no re-verification): max of
   residual fields, summed `n_samples`, AND of `orientation_consistent`;
   `interior_stuv` from any constituent tile.
d. **`normal_agreement`**: constant on a connected region; tiles that
   disagree are never merged — both remain as separate, honest regions.
e. Surviving rim branches enter the final branch list
   (`kind='overlap'`); region `boundary` refs are rebuilt against final
   indices.

## Error handling & edge cases

- No candidates / disjoint AABBs → clean empty result, `complete=True`.
- Identical-surface input: **no special-case bail** (the legacy
  `nurbs_ssx` returned bare `None` — dropped). Diagonal pairs report
  coincidence; unification yields one domain-covering region. Documented
  as quadratic-cost-but-correct.
- Collapsed-edge patches: flow through `bez_ssx` (fibers →
  `REASON_PARAMETER_FIBER` in the aggregated reasons).
- Single-span surfaces (already Bézier): one candidate pair; result is
  `bez_ssx` parity with identity remap.
- Determinism: sorted candidates, deterministic fragment ordering and
  union-find traversal.

## Testing strategy

`tests/test_nssx5.py` (fast pytest):

1. Single-patch parity: Bézier-only inputs → results match a direct
   `bez_ssx` call (params, counts, `complete=True`).
2. Multi-span transversal crossing → one stitched branch; vertex-gap
   continuity; endpoints on domain boundary.
3. Intersection along a knot line → exactly one branch (containment
   dedup).
4. Cylinder × plane → one branch, `closed=True`, seam vertex-pair
   contract verified.
5. Seam-straddling tangency → one deduped `tangent_point`.
6. Twin multi-span surfaces → one unified region, seam rims dissolved,
   `complete=True`, `reasons=[]`.
7. Starvation (tiny `max_cells`) → `complete=False`, typed reasons,
   `unresolved_regions` non-empty, no exception.
8. Status aggregation: counters sum; explicit aggregate cap respected.
9. 2–3 small `nurbs_nurbs_intersection_*.pkl` fixtures inline.

`examples/ssx/nurbs_ssx5_coverage_check.py` (objective gate, manual/CI-
optional): adapt the existing harness — reference cloud = isoline slices
of surf1 × `nurbs_csx_v4` — and require full coverage (within `5·atol`)
on all 11 `nurbs_nurbs_intersection_*.pkl` cases. Stored reference
curves are geometry ground truth, not branch-count gospel (v5 topology
may legitimately differ from the old tracer's).

Existing suites stay untouched and green.

## Non-goals

- Curve fitting / casual API layer (future layer above the engine).
- Modifying `_bez_ssx5.py` (incl. Approach C's shared seam boundary
  analysis — a later optimization).
- Cross-pair `cusp_curve` merging beyond near-identical collapse.
- Self-intersection (surf1 ≡ surf2 object) as a dedicated mode; no
  multi-surface API.
- Cython optimization; backwards compatibility (mmcore does not
  guarantee it).

## Risks / open items for planning

- Pin the exact `_bez_ssx5` symbol/constant for singularity→branch
  linking (L11) before implementation.
- Verify `bez_ssx` branch endpoints land exactly on patch-domain
  boundaries in all singular kinds (handoff claims it for traced
  branches; check overlap/tangential branches during implementation).
- Postprocess charge granularity for the containment scans (per
  sample-segment test) — follow `_point_dedup_charge`'s pattern.

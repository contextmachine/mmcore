# 2D Boolean Operations for NURBS Curves — Design

**Status:** Approved (brainstorming), ready for plan
**Date:** 2026-04-14
**Author:** Claude (Opus 4.6) with @sth-v
**Module:** `mmcore/topo/brep/boolean2d.py`

## Goal

Add 2D Boolean operators — Union, Intersection, Difference, XOR — over
regions bounded by NURBS curves. The operators work entirely inside the
existing `mmcore.topo.brep.BRep` data structure and rely on the production
`nurbs_ccx_multiple` implementation in `mmcore/numeric/intersection/ccx/_nccx4.py`
for all intersection computation. This is a topology layer on top of an
already-robust geometry primitive, not a new geometry engine.

## Non-goals

- No new intersection algorithm. `nurbs_ccx_multiple` already handles
  transverse, tangent, and overlap cases.
- No robustness framework (symbolic perturbation, exact arithmetic). Rely on
  `nurbs_ccx_multiple`'s tolerance-based robustness.
- No `Region2D` wrapper type. Work directly in terms of `BRep`.
- No performance tuning in the first version. Correctness first.

## Conventions

### BRep "Face 0" convention (existing, see `project_brep_face0_convention.md`)

- Face 0 in any BRep is a persistent "infinite exterior" face with
  `outer=None`. It carries any number of **inner** loops — one per wire
  boundary that separates the exterior from body material.
- Body faces carry at most one outer loop plus any number of inner (hole)
  loops.
- `weld_edges` already produces this shape automatically when closing
  surfaces.

### 2D orientation convention on Face 0

- A 2D BRep has `outer=None` on its wire face (Face 0) and all region
  boundaries live in `inners`.
- **CCW** loop in the plane (induced normal `+ẑ`) ⇒ bounds body **material**.
- **CW** loop in the plane (induced normal `−ẑ`) ⇒ bounds a **hole** in body
  material.
- Matches Shapely/SVG fill-even-odd and the sign of `nurbs_curve_area`.
- Empirically confirmed via `make_face_from_surface(cylinder_surface(...))`
  — the two opening circles end up as inner loops on Face 0 with opposite
  apparent xy orientations, because each loop's induced normal is anchored
  to a different side of the lateral body face. In 2D the plane normal is
  global, so the convention is unambiguous per loop.

### Standard form for a 2D BRep (input and output)

| Entity | Role |
|---|---|
| 1 `Body` | the whole 2D scene |
| 1 `Shell` | contains all body faces (2D is plane-connected, so one shell suffices) |
| 1 wire `Face` (Face 0) | `outer=None`, `surf=None`, `inners=[CW twins of every island-outer loop + CCW twins of every hole loop]` — exists so every half-edge has a valid `face` field |
| `N` body `Face` records (one per island) | `outer=CCW_loop_in_plane` bounding that island, `inners=[CW hole loops]`, `surf=None` |

**Rationale.** Body-face-per-island is preferred over everything-on-Face-0
because downstream consumers want `face.outer` to directly give an island
outline and `face.inners` to directly give its holes, with no post-hoc
ray-cast pass needed to reconstruct which loop belongs to which island.

**Input tolerance.** The algorithm walks `brep.E.values()` directly (not
loops) so older 2D BReps that store everything on Face 0 still work as
inputs — we just read the curves from `edge.geom`.

## Public API

```python
# mmcore/topo/brep/boolean2d.py

from mmcore.topo.brep import BRep
from mmcore.geom._nurbs_eval import NURBSCurveTuple

def union       (a: BRep, b: BRep, tol: float = 1e-6) -> BRep: ...
def intersection(a: BRep, b: BRep, tol: float = 1e-6) -> BRep: ...
def difference  (a: BRep, b: BRep, tol: float = 1e-6) -> BRep: ...
def xor         (a: BRep, b: BRep, tol: float = 1e-6) -> BRep: ...

def make_region_2d(loops: list[list[NURBSCurveTuple]]) -> BRep:
    """Build a 2D BRep in standard form from a list of closed loops.

    Each inner list is one closed loop whose curves are oriented end-to-end
    and whose last curve's end matches the first curve's start within tol.
    CCW-in-plane loops become material boundaries (body-face outer loops);
    CW loops become hole boundaries (body-face inner loops) attached to
    the enclosing island's body face.
    """
```

All four boolean functions route through one internal `_boolean2d(a, b, op_tag, tol)`.
The four public names are thin wrappers that set `op_tag` to one of the four
membership rules.

## Algorithm

One-pass arrangement overlay. Steps 1–3 extract and split; 4–6 build the
scratch topology; 7–9 classify and select; 10 materializes the result BRep.

### 1. Collect boundary curves from inputs

For each input BRep, walk `brep.E.values()` and for each edge `e` fetch the
trimmed NURBS curve `brep.G_CRV[e.geom]` restricted to `e.param`. Result: two
lists `curves_A`, `curves_B`, each `list[NURBSCurveTuple]`. A source-tag
array `src = [A, A, …, B, B, …]` records which input each curve came from.

We do *not* walk loops — iterating the edge dict directly makes the input
format agnostic. Any 2D BRep form works (body-face-per-island or everything
on Face 0).

### 2. Compute all intersections in one shot

```python
from mmcore.numeric.intersection.ccx._nccx4 import nurbs_ccx_multiple
isolated, overlaps = nurbs_ccx_multiple(curves_A + curves_B, tol=tol)
```

Single call gives every transverse crossing and every coincident overlap
range, already paired as `(curve1_i, curve2_i, u, v, point)` or as
`(curve1_i, curve2_i, u_range, v_range, points)` respectively.

### 3. Split each curve at its intersection parameters

For each curve index `i`, accumulate every split parameter:
- `u` from `isolated` where `curve1_i == i`;
- `v` from `isolated` where `curve2_i == i`;
- `u_range[0]` and `u_range[1]` from `overlaps` where `curve1_i == i`;
- `v_range[0]` and `v_range[1]` from `overlaps` where `curve2_i == i`.

Dedupe the param list using `nurbs_curve_param_tolerance(curves[i], tol)`
from `mmcore/geom/_nurbs_param_tol.py` to merge adjacent duplicates. Then:

```python
from mmcore.geom._nurbs_knots import split_curve_multiple
sub_segments_i = split_curve_multiple(curves[i], params)
```

Each sub-segment inherits the source tag of its parent curve.

### 4. Merge overlap duplicates

For each entry in `overlaps`, the CCX tells us "curve `curve1_i` from
`u_range` is geometrically the same as curve `curve2_i` from `v_range`."
Locate the corresponding sub-segments in step 3's output (by matching
parameter endpoints within tolerance), keep the first, discard the second,
and upgrade the kept segment's source tag from `{A}` or `{B}` to `{A, B}`.

This is `O(overlaps)` and needs no geometric test — the CCX already told us
which segments coincide.

### 5. Build a vertex pool

Collect every endpoint of every surviving sub-segment. Merge endpoints that
are within `tol` of each other (grid-hashed bucket keyed on
`(round(x/tol), round(y/tol))`). Each unique point becomes one working-graph
vertex ID. For each sub-segment, record its two vertex IDs.

### 6. Build the half-edge arrangement (scratch DCEL, not BRep)

A lightweight in-memory half-edge graph used only for classification — the
BRep is constructed fresh in step 10.

- For each sub-segment create two `HalfEdge`s (forward, reverse) with twin
  pointers wired immediately.
- At each vertex, sort all outgoing half-edges by tangent angle (CCW in the
  plane). Tangent comes from `evaluate_nurbs_curve(sub, t, d_order=1)` at
  the relevant endpoint — use `C1`, not the straight-line chord.
- Link `he.next = he.twin.ccw_prev` (standard DCEL "face on left" rule).
- Walk every half-edge once; each walk yields one closed cycle = one face.
- Identify the unbounded face: find the vertex with minimum `(y, x)`; the
  HE whose outgoing tangent has the smallest angle at that vertex has its
  twin incident to the unbounded face.

### 7. Classify every bounded face for `(inA, inB)`

For each bounded face `f`:
1. Pick any half-edge `h` on `f`'s boundary.
2. Compute `h`'s midpoint `m` and its tangent `t` (via derivative eval at
   the midpoint parameter).
3. Compute the inward normal `n = rotate90_ccw(t)` (points into `f` because
   `f` is on the left of `h`).
4. `sample = m + ε * n` where `ε = tol` (a small step strictly into `f`).
5. Run two point-in-region tests:

```python
inA = point_in_region(sample, curves_A_original, tol=tol)
inB = point_in_region(sample, curves_B_original, tol=tol)
```

Complexity: `O(F × (|A| + |B|))`. Can be swapped for flood-fill propagation
later without changing the rest of the pipeline.

#### `point_in_region` — segment-based PIP using `nurbs_ccx`

We do **not** use `mmcore.numeric.algorithms.point_in_curve.curve_x_ray` —
it is an older implementation, known to have correctness issues on tangent
and endpoint hits, and not tuned for NURBS. Instead we implement a small
helper that leverages the same production `nurbs_ccx_multiple` the main
pipeline already uses. This helper is **validated** against 13 focused tests
in `tests/test_point_in_region.py` (unit square, rational circle, annulus,
disjoint circles, tangent grazing, boundary-point guard) — see commit
`38f26cf`.

```python
def point_in_region(
    point: np.ndarray,
    region_curves: list[NURBSCurveTuple],
    tol: float = 1e-6,
) -> bool:
    """Return True iff point lies strictly inside the region bounded by
    region_curves (even-odd winding rule).

    Casts a single line segment (a degree-1 NURBS curve) from point past
    the region's bounding box, intersects it with region_curves via
    nurbs_ccx_multiple, and counts transverse crossings using signed-
    distance sampling to distinguish transverse from tangent hits.
    Raises RuntimeError if the point lies on a region boundary.
    """
```

**Segment construction — one shot, no retries.**
1. Compute the axis-aligned bounding box of `region_curves + {point}` by
   taking `min`/`max` over every curve's `control_points` and including
   `point` itself (the control polygon is a conservative bound for a NURBS
   curve).
2. Let `D = ||bbox_max - bbox_min||` (diagonal length). Use
   `L = 2 * D + max(1, D) * 1e-3` as the segment length — guaranteed to
   escape the bbox from any starting point, with a safety margin.
3. Pick a direction `d = (cos θ, sin θ)` with `θ = 0.31415` — a fixed
   irrational-ish seed, not axis-aligned and not parallel to typical CAD
   features. **No retries** — one direction suffices.
4. Build a degree-1 NURBS:
   ```python
   seg = NURBSCurveTuple(
       order=2,
       knot=np.array([0., 0., 1., 1.]),
       control_points=np.array([point, point + L * d]),
       weights=np.array([1., 1.]),
   )
   ```

**Intersecting and counting.**
1. `isolated, overlaps = nurbs_ccx_multiple([seg] + region_curves, tol=tol)`.
2. Define the signed line-equation scalar for the segment's supporting line:
   ```python
   def line_side(q):
       return (q[1] - pt[1]) * d[0] - (q[0] - pt[0]) * d[1]
   ```
   `line_side(q) > 0` and `line_side(q) < 0` identify the two sides of the
   line; `== 0` is on the line. This is the 2D cross product of `(q - pt)`
   with `d`.
3. For each `isolated` entry involving the segment (`curve1_i == 0` or
   `curve2_i == 0`):
   - Let `u_seg` be the segment parameter and `t_curve` the region-curve
     parameter. If `u_seg < 2 * tol`, `point` lies on a region curve —
     raise `RuntimeError("point_in_region: point lies on a region boundary")`.
   - Sample the hit curve at `t_curve ± dt` where
     `dt = 1e-3 * (t_end - t_start)` (fraction of the curve's parameter
     range). Clamp samples to the curve's valid interval.
   - If either clamp ran up against an endpoint (so we couldn't sample both
     sides), count this hit as a transverse crossing (conservative fall-back
     for curve-endpoint corner cases).
   - Otherwise, evaluate the curve at `t_before` and `t_after`, compute
     `s_before = line_side(p_before)`, `s_after = line_side(p_after)`:
     - `s_before * s_after > 0` ⇒ both samples on the same side ⇒ the curve
       grazes the line (tangent touch) — do **not** count.
     - `s_before * s_after < 0` ⇒ samples on opposite sides ⇒ transverse
       crossing — count it.
4. **Ignore overlaps entirely.** A segment lying along a curve for a range
   doesn't flip parity: you're sliding along the boundary, not crossing it.
   The sampling check above handles re-entry at isolated hits on either
   side of any overlap.
5. Return `count % 2 == 1`.

**Why sampling, not just "parallel tangent" detection.** A naïve "the
intersection is tangent iff the curve's tangent at the hit is parallel to
`d`" check **fails on transverse crossings with parallel tangents** — e.g.
a cubic S-curve crossing the segment line at its inflection point has
parallel tangents at a real crossing. Sampling the curve before and after
the hit and comparing signs of `line_side(...)` is a topologically correct
test for "does the curve cross the line at this hit point" and handles
grazing and inflection-transverse uniformly. Measured result: the parallel-
tangent heuristic fires at `|cross| ≈ 2e-7` for a geometrically exact
tangent even on a well-formed rational circle, making threshold tuning
impossible. The sampling check reduces to two scalar arithmetic ops per hit
and does not depend on a threshold at all.

**Why a segment and not a ray.** `nurbs_ccx_multiple` wants bounded NURBS
inputs. A ray (unbounded) isn't a NURBS curve; a long-enough segment is
equivalent for counting purposes and lets us reuse the same tuned,
well-tested intersection code that produced the arrangement in steps 2–4.
No new intersection primitive needed.

### 8. Apply the operation rule

| Operator | Keep rule |
|---|---|
| `union` | `inA or inB` |
| `intersection` | `inA and inB` |
| `difference` | `inA and not inB` |
| `xor` | `inA != inB` |

Tag each bounded face `kept: bool`.

### 9. Group kept faces into islands and identify holes

- **Island** = connected component of kept faces under the adjacency
  "share an arrangement edge whose other side is also kept." Union-find
  over edges.
- **Island boundary** = union of all half-edges belonging to kept faces in
  the component whose twin half-edge belongs to a *not kept* face. Extract
  closed loops by walking `he.next` restricted to boundary half-edges.
- Per island, the loop enclosing the largest signed area (shoelace on the
  sampled endpoints) is the **outer loop** (CCW in the plane); all others
  are **holes** (CW in the plane).

An interior edge — one where both sides are kept — does not appear on any
output loop; it is dropped silently.

### 10. Materialize the result BRep

Start from an empty `BRep`:

```python
result = BRep()
body = result.new_body(shells=[])
shell = result.new_shell(faces=[], body=body.id)
body.shells.append(shell.id)
face0 = result.new_face(outer=None, inners=[], shell=shell.id, surf=None)
shell.faces.append(face0.id)
```

For each island, in this order — outer loop first, then each hole:

1. Walk the loop's sub-segments. For each unique endpoint encountered,
   create a `Vertex` (deduping against existing vertices of this result BRep
   by ID, not position — each arrangement vertex maps to exactly one result
   Vertex).
2. For each sub-segment, create:
   - a fresh entry in `result.G_CRV` via `result.new_curve(subcurve)`;
   - an `Edge` with that `geom` and `param = subcurve.interval()`;
   - two `HalfEdge`s with twin pointers wired.
3. Splice the HEs into two loops: the "body-face side" and the "Face 0 side."
   The body-face side is the loop whose walk direction keeps the island's
   material on the left (CCW for outer, CW for hole). The Face 0 side is
   the twin ring.
4. On the body-face side, if this is the island's outer loop, record the new
   `Loop` as `body_face.outer`. Otherwise append to `body_face.inners`.
5. On the Face 0 side, append the new `Loop` to `face0.inners`.

After all islands are placed:
- Create the body `Face` for each island: `outer=outer_loop.id,
  inners=[hole_loop.ids…], shell=shell.id, surf=None, same_sense=True`.
- Append each body face id to `shell.faces`.

Finally call `result.validate()` — any non-empty return is a real bug
(caught in tests, not silently swallowed).

## Edge cases

| Situation | Handling |
|---|---|
| **Empty input** (BRep has no edges) | Skip CCX. `union`/`xor` return a structural clone of the non-empty side. `intersection`/`difference(A,∅)` return empty BRep. `difference(∅,B)` returns empty BRep. |
| **Disjoint inputs** | `nurbs_ccx_multiple` returns `(None, None)`. Step 3 is a no-op. Arrangement has separate components per input. Classification and selection proceed normally. |
| **Identical inputs** | Every edge matches itself in `overlaps`. Step 4 collapses each pair. `union = intersection = A`, `difference = xor = empty`. |
| **A fully inside B** (no intersections, nested) | No splits. Classification handles it via point-in-region tests. `union = B`, `intersection = A`, `difference(B,A) = B with A-shaped hole`, `xor = same as difference(B,A)`. |
| **Tangent intersections** | CCX returns tangent touches as isolated crossings with valid `(u,v)`. The arrangement vertex has degree 2 on each side and classifies correctly. |
| **Overlap with opposite parameter directions** | CCX's `u_range` and `v_range` capture both directions. Step 4 keeps one kept segment regardless — the arrangement doesn't care about the parent curves' original parameterization, only about the sub-segment geometry. |
| **Interior sample hits a boundary** | The `sample = midpoint + ε * inward_normal` construction guarantees strict interiority in any non-degenerate face. The signed-distance sampling inside `point_in_region` handles tangent grazing and inflection-transverse hits topologically (same side ⇒ grazing, opposite side ⇒ transverse). If the sample somehow ends up exactly on a region curve, `point_in_region` raises `RuntimeError` rather than returning an arbitrary parity. |
| **Non-closed loop in input** | Call `input_brep.validate()` at the top of step 1. If it returns a non-empty error list, raise `ValueError("input BRep failed validate(): ...")` with the first error attached. This catches dangling edges, mis-wired half-edges, and orphaned loops before they corrupt the arrangement. |
| **Output loop fails to close** (numerical classification flip-flop) | Step 9 raises `RuntimeError` with the offending face ids. No silent malformation. |
| **Empty result** | Return a valid BRep with Body + Shell + Face 0 (`outer=None`, `inners=[]`) and **zero** body faces. The empty-region signal is `len([f for f in brep.F.values() if f.outer is not None]) == 0`. |

No tolerance escalation. No retry loops. No "smart" fallbacks. Every failure
path is either a clean result or a raised exception with a concrete message.

## Complexity

- CCX: dominated by `nurbs_ccx_multiple` (BVH-accelerated).
- Splitting + dedup: `O(N + I)` where `N` = total curves, `I` = total
  intersections.
- Arrangement build: `O((N + I) log(N + I))` for the vertex hash + per-vertex
  angular sort.
- Classification: `O(F × (|A| + |B|))` where `F` = arrangement face count.
  This is the dominant non-CCX term; acceptable for a first version.
- Result BRep construction: `O(output edges)`.

## Test plan

New file `tests/test_boolean2d.py`. Each test builds two input BReps via
`make_region_2d` (or by hand), runs one of the four ops, and asserts on the
result.

| # | Name | Description | Assertions |
|---|---|---|---|
| T1 | Disjoint rectangles | Two non-touching squares | 4 ops exercised; union ⇒ 2 islands, intersection ⇒ empty, difference ⇒ 1 island, xor ⇒ 2 islands |
| T2 | Overlapping circles | Two unit circles, centres distance 1 apart | 4 ops exercised; union ⇒ 1 peanut, intersection ⇒ 1 lens, difference ⇒ 1 crescent, xor ⇒ 2 crescents |
| T3 | Square-with-hole vs disk | Annular square + disk that crosses the hole | `union` ⇒ island with residual hole |
| T4 | Shared edge | Two adjacent squares sharing one edge exactly | Exercises overlap dedup; `union` ⇒ 1 rectangle (4 vertices, not 5); `intersection` ⇒ empty |
| T5 | Nested (A ⊆ B) | Small circle inside big square, no intersections | `union = B`; `intersection = A`; `difference(B,A)` ⇒ square with circular hole; `xor = same` |
| T6 | Tangent circles | Two unit circles touching at one point | `union` ⇒ figure-eight outline; `intersection` ⇒ empty |
| T7 | Identical inputs | A = B = unit circle | `union = intersection = A`; `difference = xor = empty` |
| T8 | Composition | `(square ∪ triangle) ∩ circle` with first result fed to second | Validates round-trip: output form is a valid input |
| T9 | Surface-derived input | `make_face_from_surface` on a planar NURBS surface, then boolean against a hand-built one | Validates input-form agnosticism |
| T10 | validate() cross-cutting | Every result in T1–T9 must satisfy `result.validate() == []` | Catches topology bugs |

Coverage targets:
- All four operators exercised on at least T1 and T2.
- At least one operator exercised on each of T3–T9.
- `validate()` called on every returned BRep.

No performance benchmarks in the first cut.

## Dependencies

- `mmcore.numeric.intersection.ccx._nccx4.nurbs_ccx_multiple` — existing.
- `mmcore.geom._nurbs_knots.split_curve_multiple` — existing.
- `mmcore.geom._nurbs_param_tol.nurbs_curve_param_tolerance` — existing.
- `mmcore.geom._nurbs_eval.evaluate_nurbs_curve` — existing.
- `point_in_region(point, region_curves, tol)` — **implemented** in
  `mmcore/topo/brep/boolean2d.py` as a private helper. Builds a segment-as-
  NURBS, calls `nurbs_ccx_multiple`, and classifies each hit as
  transverse/tangent via signed-distance sampling of the region curve on
  both sides of the hit parameter. Validated by
  `tests/test_point_in_region.py` (13 cases including tangent grazing on a
  rational circle, annulus ring/hole, disjoint regions, and a boundary-
  point guard). Replaces the older, less reliable
  `mmcore.numeric.algorithms.point_in_curve.curve_x_ray`.
- `mmcore.topo.brep.BRep` — existing; uses factory helpers `new_vertex`,
  `new_edge`, `new_halfedge`, `new_loop`, `new_face`, `new_shell`, `new_body`,
  `new_curve`, and the `validate()` method.

No new Cython, no new low-level primitives.

## Out of scope

- 3D Boolean operations (deferred — different problem).
- Performance tuning (flood-fill classification, edge-label propagation).
- Intermediate `Region2D` wrapper type (may be added later).
- Fuzzing / property-based tests for robustness (deferred; targeted tests
  only in the first cut).
- Visualization helpers for 2D BReps (separate concern).

## Future extensions

- Flood-fill classification: replace step 7's per-face PIP with a single
  PIP on the unbounded face plus propagation across edges, flipping
  `inA`/`inB` flags based on the crossed edge's source tag. Drops
  classification cost from `O(F × (|A| + |B|))` to `O(F + E)`.
- Multi-input boolean: `union(a, b, c, ...)` via iterated pairwise union, or
  (better) a single-pass N-way arrangement. Same core pipeline, just N
  source tags instead of two.
- Buffer / offset operations: feed `_nurbs_offset.py` output into the same
  pipeline.
- Winding-rule parameter: accept `even_odd` (default, as specified here) or
  `non_zero` for membership determination.

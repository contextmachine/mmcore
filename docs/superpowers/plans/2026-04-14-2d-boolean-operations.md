# 2D Boolean Operations Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 2D Boolean operators (Union, Intersection, Difference, XOR) over NURBS-curve regions on top of the existing `BRep` data structure, using `nurbs_ccx_multiple` for all intersection work.

**Architecture:** Single module `mmcore/topo/brep/boolean2d.py` containing one public helper (`make_region_2d`), four public ops (`union`/`intersection`/`difference`/`xor`), and a pipeline of private helpers that together implement a one-pass planar-arrangement overlay (collect → split at intersections → dedup overlaps → build lightweight scratch DCEL → classify faces via `point_in_region` → select kept faces → group into islands with holes → materialize fresh `BRep`). `point_in_region` is already implemented and tested (commit `38f26cf`).

**Tech Stack:** Python 3.9+; NumPy; `NURBSCurveTuple` as the sole curve type (the deprecated Cython `NURBSCurve` class is never used); `nurbs_ccx_multiple` from `mmcore/numeric/intersection/ccx/_nccx4.py`; `split_curve_multiple`/`reverse_curve` from `mmcore/geom/_nurbs_knots.py`; `BRep` factory helpers (`new_vertex`, `new_edge`, `new_halfedge`, `new_loop`, `new_face`, `new_shell`, `new_body`, `new_curve`) from `mmcore/topo/brep/__init__.py`; pytest.

---

## File Structure

- **Modify:** `mmcore/topo/brep/boolean2d.py` — already exists with `point_in_region`. Add `make_region_2d`, pipeline helpers, `_boolean2d`, and four public ops.
- **Create:** `tests/test_boolean2d.py` — integration tests for `make_region_2d` and the four ops (T1–T10 from the spec).
- **Unchanged:** `tests/test_point_in_region.py` — already green (13 tests).

All internals are private helpers (`_collect_curves`, `_split_and_dedup`, `_build_arrangement`, `_classify_faces`, `_select_kept_faces`, `_extract_island_loops`, `_materialize_result`). The public surface is `make_region_2d` + `union`/`intersection`/`difference`/`xor`.

Reference spec: `docs/superpowers/specs/2026-04-14-2d-boolean-operations-design.md` (the "Algorithm" section is the source of truth for the pipeline).

---

## Task 1: `make_region_2d` — single closed CCW loop

Builds a `BRep` for the simplest region (one closed boundary, no holes, one island). The engineer will extend this in Task 2 for holes and multi-island inputs.

**Files:**
- Create: `tests/test_boolean2d.py`
- Modify: `mmcore/topo/brep/boolean2d.py` (append after `point_in_region`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_boolean2d.py` with the following content (all tests in this plan will accumulate here):

```python
"""Integration tests for 2D Boolean operations and make_region_2d."""
from __future__ import annotations

import numpy as np
import pytest

from mmcore.construction import circle
from mmcore.geom._nurbs_eval import NURBSCurveTuple
from mmcore.topo.brep import BRep
from mmcore.topo.brep.boolean2d import (
    difference,
    intersection,
    make_region_2d,
    union,
    xor,
)


def _line(p0, p1) -> NURBSCurveTuple:
    """Degree-1 NURBS segment from p0 to p1 (3D points, z=0)."""
    return NURBSCurveTuple(
        order=2,
        knot=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=np.array([p0, p1], dtype=float),
        weights=np.array([1.0, 1.0], dtype=float),
    )


def _square_ccw(x0, y0, side) -> list[NURBSCurveTuple]:
    """CCW square boundary, 4 line segments."""
    return [
        _line([x0,        y0,        0.0], [x0 + side, y0,        0.0]),
        _line([x0 + side, y0,        0.0], [x0 + side, y0 + side, 0.0]),
        _line([x0 + side, y0 + side, 0.0], [x0,        y0 + side, 0.0]),
        _line([x0,        y0 + side, 0.0], [x0,        y0,        0.0]),
    ]


def _count_body_faces(brep: BRep) -> int:
    return sum(1 for f in brep.F.values() if f.outer is not None)


def test_make_region_2d_unit_square_creates_one_body_face():
    region = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    assert _count_body_faces(region) == 1
    # body face has 1 outer loop, 0 inners
    body_face = next(f for f in region.F.values() if f.outer is not None)
    assert body_face.outer is not None
    assert body_face.inners == []
    assert body_face.surf is None
    # Face 0 exists with outer=None and inners holding the twin of the outer
    wire_face = next(f for f in region.F.values() if f.outer is None)
    assert len(wire_face.inners) == 1
    # Topology is internally consistent
    assert region.validate() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py::test_make_region_2d_unit_square_creates_one_body_face -v`

Expected: `ImportError: cannot import name 'make_region_2d' from 'mmcore.topo.brep.boolean2d'`.

- [ ] **Step 3: Add `make_region_2d` to `boolean2d.py` — single-loop case**

Append the following to `mmcore/topo/brep/boolean2d.py` (after `point_in_region`):

```python
# ---------------------------------------------------------------------------
#  make_region_2d — builder for 2D BRep inputs
# ---------------------------------------------------------------------------

def _signed_area_xy_samples(curves: list[NURBSCurveTuple], n_per_curve: int = 16) -> float:
    """Shoelace signed area from sampled points along the loop's curves.

    Positive ⇒ CCW in xy plane ⇒ bounds material.
    Negative ⇒ CW in xy plane ⇒ bounds a hole.
    """
    pts = []
    for crv in curves:
        t0, t1 = crv.interval()
        for i in range(n_per_curve):
            t = t0 + (t1 - t0) * (i / n_per_curve)
            ev = evaluate_nurbs_curve(crv, t, 0)
            pts.append(np.asarray(ev['C'], dtype=float))
    pts = np.asarray(pts)
    xs = pts[:, 0]
    ys = pts[:, 1]
    return 0.5 * float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))


def make_region_2d(loops: list[list[NURBSCurveTuple]]) -> BRep:
    """Build a 2D BRep from a list of closed loops.

    Each inner list is one closed loop whose curves are oriented end-to-end
    (curve[i].end() ≈ curve[i+1].start() and last.end() ≈ first.start()).
    CCW loops (positive signed area in xy) become body-face outer loops —
    one body face per CCW loop. CW loops become holes attached to the
    containing body face (determined by point-in-region tests).

    Single-shell form: one Body, one Shell, one wire Face (Face 0) with
    outer=None, N body faces with outer + inners. Every half-edge has a
    valid face reference.
    """
    brep = BRep()
    body = brep.new_body(shells=[])
    shell = brep.new_shell(faces=[], body=body.id)
    body.shells.append(shell.id)
    wire_face = brep.new_face(outer=None, inners=[], shell=shell.id, surf=None)
    shell.faces.append(wire_face.id)

    # Classify each loop by signed area in xy plane.
    loops_by_type: list[tuple[str, list[NURBSCurveTuple]]] = []
    for loop_curves in loops:
        area = _signed_area_xy_samples(loop_curves)
        kind = 'outer' if area > 0 else 'hole'
        loops_by_type.append((kind, loop_curves))

    # First pass: build all outer loops (one body face per outer loop).
    outer_face_ids: list[int] = []
    outer_sample_points: list[np.ndarray] = []
    for kind, loop_curves in loops_by_type:
        if kind != 'outer':
            continue
        face_id = _add_loop_to_brep(brep, shell.id, wire_face.id, loop_curves, is_body_outer=True)
        outer_face_ids.append(face_id)
        # sample an interior point for later hole containment tests
        sample = _interior_sample_of_loop(loop_curves)
        outer_sample_points.append(sample)

    # Second pass: for each hole, find the containing body face and attach as inner.
    for kind, loop_curves in loops_by_type:
        if kind != 'hole':
            continue
        # find which body face's material contains the hole's centroid
        hole_sample = _interior_sample_of_loop(loop_curves)
        host_face_id = None
        for face_id in outer_face_ids:
            face = brep.F[face_id]
            outer_loop_curves = _loop_curves_from_loop_id(brep, face.outer)
            if point_in_region(hole_sample, outer_loop_curves, tol=1e-6):
                host_face_id = face_id
                break
        if host_face_id is None:
            raise ValueError("hole loop is not contained by any outer loop")
        _add_loop_to_brep(brep, shell.id, wire_face.id, loop_curves,
                          is_body_outer=False, host_face_id=host_face_id)

    return brep


def _interior_sample_of_loop(loop_curves: list[NURBSCurveTuple]) -> np.ndarray:
    """Return a point that's (approximately) inside the loop.

    Simple strategy: shoelace centroid of the first curve's start points and
    the midpoint of each curve. Not guaranteed interior for very non-convex
    shapes but works for the shapes we care about (squares, circles, simple
    polygons). For exotic shapes, callers should supply their own sample.
    """
    pts = []
    for crv in loop_curves:
        t0, t1 = crv.interval()
        ev = evaluate_nurbs_curve(crv, 0.5 * (t0 + t1), 0)
        pts.append(np.asarray(ev['C'], dtype=float))
    return np.mean(np.asarray(pts), axis=0)


def _loop_curves_from_loop_id(brep: BRep, loop_id: int) -> list[NURBSCurveTuple]:
    """Walk a loop's half-edges and return the list of curves it traverses."""
    curves = []
    first = brep.L[loop_id].he
    he_id = first
    while True:
        he = brep.HE[he_id]
        edge = brep.E[he.edge]
        crv = brep.G_CRV[edge.geom]
        curves.append(crv)
        he_id = he.next
        if he_id == first:
            break
    return curves


def _add_loop_to_brep(
    brep: BRep,
    shell_id: int,
    wire_face_id: int,
    loop_curves: list[NURBSCurveTuple],
    *,
    is_body_outer: bool,
    host_face_id: int | None = None,
) -> int:
    """Insert the vertices, edges, half-edges, loops (body + wire twins), and
    (if is_body_outer) a body face for one closed loop of oriented NURBS curves.

    If is_body_outer is True, creates a new body face with this loop as its
    outer loop; returns the new body face id.

    If is_body_outer is False, treats the loop as a hole to be attached to
    host_face_id as an inner loop; returns host_face_id.
    """
    n = len(loop_curves)
    if n < 1:
        raise ValueError("loop_curves must have at least one curve")

    # Create one vertex per curve start. curve[i].end() ≈ curve[(i+1)%n].start().
    vertices: list[int] = []
    for i, crv in enumerate(loop_curves):
        start = tuple(np.asarray(crv.start(), dtype=float).tolist())
        v = brep.new_vertex(point=start, tol=1e-6)
        vertices.append(v.id)

    # Determine body-side face id
    if is_body_outer:
        body_face = brep.new_face(outer=None, inners=[], shell=shell_id,
                                  same_sense=True, surf=None)
        brep.S[shell_id].faces.append(body_face.id)
        body_face_id = body_face.id
    else:
        body_face_id = host_face_id  # type: ignore[assignment]

    # Create edges + half-edges for each curve.
    body_hes: list[int] = []  # in walk order
    wire_hes: list[int] = []  # in walk order (twins, reversed-winding cycle)
    for i, crv in enumerate(loop_curves):
        v_start = vertices[i]
        v_end = vertices[(i + 1) % n]
        crv_id = brep.new_curve(crv)
        edge = brep.new_edge(v_start=v_start, v_end=v_end, geom=crv_id,
                             param=crv.interval())
        # Body-side HE walks v_start→v_end (the direction the user supplied).
        he_body = brep.new_halfedge(
            edge=edge.id, face=body_face_id, loop=None,
            vert=v_end, orient=True, pcurve=None,
        )
        # Wire-side twin walks v_end→v_start on the wire (Face 0) face.
        he_wire = brep.new_halfedge(
            edge=edge.id, face=wire_face_id, loop=None,
            vert=v_start, orient=False, pcurve=None,
        )
        he_body.twin = he_wire.id
        he_wire.twin = he_body.id
        edge.he = he_body.id
        body_hes.append(he_body.id)
        wire_hes.append(he_wire.id)

    # Link next/prev along the body loop (forward cycle).
    for i in range(n):
        brep.HE[body_hes[i]].next = body_hes[(i + 1) % n]
        brep.HE[body_hes[(i + 1) % n]].prev = body_hes[i]

    # Link next/prev along the wire loop (reverse cycle: the wire HE for
    # curve i starts at v_{i+1} and ends at v_i, so the walk order is
    # wire_hes[n-1], wire_hes[n-2], ..., wire_hes[0]).
    for i in range(n):
        nxt_i = (i - 1) % n
        brep.HE[wire_hes[i]].next = wire_hes[nxt_i]
        brep.HE[wire_hes[nxt_i]].prev = wire_hes[i]

    # Create the two loop records and tag HEs.
    body_loop = brep.new_loop(face=body_face_id, he=body_hes[0],
                              is_outer=is_body_outer)
    wire_loop = brep.new_loop(face=wire_face_id, he=wire_hes[0], is_outer=False)
    for hid in body_hes:
        brep.HE[hid].loop = body_loop.id
    for hid in wire_hes:
        brep.HE[hid].loop = wire_loop.id

    # Attach body loop to its face.
    if is_body_outer:
        brep.F[body_face_id].outer = body_loop.id
    else:
        brep.F[body_face_id].inners.append(body_loop.id)

    # The wire loop always lives in Face 0's inners list.
    brep.F[wire_face_id].inners.append(wire_loop.id)

    return body_face_id
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py::test_make_region_2d_unit_square_creates_one_body_face -v`

Expected: `PASSED`.

- [ ] **Step 5: Run the existing PIP tests and full validate to catch regressions**

Run: `.venv/bin/python -m pytest tests/test_point_in_region.py tests/test_boolean2d.py -v`

Expected: all 14 tests pass (13 existing + 1 new).

- [ ] **Step 6: Commit**

```bash
git add mmcore/topo/brep/boolean2d.py tests/test_boolean2d.py
git commit -m "feat(boolean2d): make_region_2d for single-loop inputs"
```

---

## Task 2: `make_region_2d` — holes and multiple islands

Extends the helper to handle multi-loop regions (CCW outer + nested CW holes) and multiple disjoint islands.

**Files:**
- Modify: `tests/test_boolean2d.py`

- [ ] **Step 1: Write three failing tests**

Append to `tests/test_boolean2d.py`:

```python
def test_make_region_2d_two_disjoint_squares():
    region = make_region_2d([
        _square_ccw(0.0, 0.0, 1.0),
        _square_ccw(2.0, 0.0, 1.0),
    ])
    assert _count_body_faces(region) == 2
    wire_face = next(f for f in region.F.values() if f.outer is None)
    # 2 body-face outer loops ⇒ 2 wire-twin inner loops
    assert len(wire_face.inners) == 2
    assert region.validate() == []


def test_make_region_2d_square_with_hole():
    outer = _square_ccw(0.0, 0.0, 4.0)
    # CW hole in the middle (reverse order of a CCW small square)
    hole = list(reversed(_square_ccw(1.5, 1.5, 1.0)))
    # also reverse the endpoints of each segment so curves are correctly oriented
    hole = [
        _line(crv.control_points[-1], crv.control_points[0])
        for crv in hole
    ]
    region = make_region_2d([outer, hole])
    assert _count_body_faces(region) == 1
    body_face = next(f for f in region.F.values() if f.outer is not None)
    assert len(body_face.inners) == 1  # the hole was attached
    # Face 0 has 2 inners: twin of outer loop + twin of hole loop
    wire_face = next(f for f in region.F.values() if f.outer is None)
    assert len(wire_face.inners) == 2
    assert region.validate() == []


def test_make_region_2d_hole_orientation_detection():
    """Orientation is auto-detected from signed area, regardless of input order."""
    outer = _square_ccw(0.0, 0.0, 4.0)
    # Provide the hole in CCW order; the auto-detection should recognise
    # that it's nested inside `outer` but... wait, a CCW hole is still material
    # in the 2D convention. Instead test: a CW outer that's actually a
    # reverse-wound island (no enclosing outer).
    cw_square = list(reversed(_square_ccw(10.0, 10.0, 1.0)))
    cw_square = [
        _line(crv.control_points[-1], crv.control_points[0])
        for crv in cw_square
    ]
    # A lone CW loop with no enclosing outer ⇒ ValueError
    with pytest.raises(ValueError, match="not contained"):
        make_region_2d([cw_square])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py -v`

Expected: the new three tests fail (first two with assertion errors or topology validation errors, third with whatever error the unhandled-hole case produces).

- [ ] **Step 3: Implementation already handles these cases**

The `make_region_2d` implementation from Task 1 already handles holes via the second-pass containment test, and disjoint outer loops via the loop. Running the tests should either pass or reveal a bug.

If a test fails, debug the specific failure before moving on. Common bugs:
- `_signed_area_xy_samples` may compute area wrong for 1-curve loops — reverify shoelace formula.
- `_add_loop_to_brep`'s wire-loop `next/prev` linking may be off-by-one — walk with a small example on paper and verify.
- The wire-twin HE at index i should have its `vert` set to the START of curve `i`, not the end. Re-read the `_add_loop_to_brep` code and confirm.

- [ ] **Step 4: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_point_in_region.py tests/test_boolean2d.py -v`

Expected: all 17 tests pass.

- [ ] **Step 5: Commit**

```bash
git add tests/test_boolean2d.py mmcore/topo/brep/boolean2d.py
git commit -m "test(boolean2d): make_region_2d holes and multi-island"
```

---

## Task 3: `_collect_curves_with_sources`

Helper that walks two input BReps and returns flat lists of curves plus source tags (A/B).

**Files:**
- Modify: `mmcore/topo/brep/boolean2d.py` (append)
- Modify: `tests/test_boolean2d.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_boolean2d.py`:

```python
from mmcore.topo.brep.boolean2d import _collect_curves_with_sources


def test_collect_curves_with_sources_from_two_regions():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    curves, sources = _collect_curves_with_sources(a, b)
    # each square has 4 edges ⇒ 8 curves total
    assert len(curves) == 8
    assert sources == ['A'] * 4 + ['B'] * 4
    # every curve is a NURBSCurveTuple
    for c in curves:
        assert isinstance(c, NURBSCurveTuple)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py::test_collect_curves_with_sources_from_two_regions -v`

Expected: `ImportError: cannot import name '_collect_curves_with_sources'`.

- [ ] **Step 3: Implement `_collect_curves_with_sources`**

Append to `mmcore/topo/brep/boolean2d.py`:

```python
# ---------------------------------------------------------------------------
#  Boolean op pipeline — private helpers
# ---------------------------------------------------------------------------

def _collect_curves_with_sources(
    brep_a: BRep, brep_b: BRep
) -> tuple[list[NURBSCurveTuple], list[str]]:
    """Walk both BReps' edges and return (curves, sources) lists.

    Iterates brep.E.values() directly — format agnostic to whether the BRep
    stores boundaries as body-face outer/inner loops or as Face 0 inners.
    Each edge contributes exactly one curve (the one in G_CRV trimmed to the
    edge's param range).
    """
    # validate inputs up front (fail fast on malformed BReps)
    for name, brep in (('a', brep_a), ('b', brep_b)):
        errs = brep.validate()
        if errs:
            raise ValueError(
                f"input BRep {name!r} failed validate(): {errs[0]}"
            )

    curves: list[NURBSCurveTuple] = []
    sources: list[str] = []
    for brep, tag in ((brep_a, 'A'), (brep_b, 'B')):
        for e in brep.E.values():
            if e.geom is None:
                raise ValueError(
                    f"input BRep has an edge without geometry (edge id {e.id})"
                )
            base = brep.G_CRV[e.geom]
            t0, t1 = e.param
            if (t0, t1) == base.interval():
                curves.append(base)
            else:
                from mmcore.geom._nurbs_knots import trim_curve
                curves.append(trim_curve(base, min(t0, t1), max(t0, t1)))
            sources.append(tag)
    return curves, sources
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py::test_collect_curves_with_sources_from_two_regions -v`

Expected: `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add mmcore/topo/brep/boolean2d.py tests/test_boolean2d.py
git commit -m "feat(boolean2d): _collect_curves_with_sources"
```

---

## Task 4: `_split_curves_at_intersections` — isolated + overlap dedup

Runs `nurbs_ccx_multiple` over all input curves, splits each curve at its intersection parameters, and deduplicates coincident segments from overlap entries. Returns a list of sub-segments tagged with their source set.

**Files:**
- Modify: `mmcore/topo/brep/boolean2d.py` (append)
- Modify: `tests/test_boolean2d.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_boolean2d.py`:

```python
from mmcore.topo.brep.boolean2d import _split_curves_at_intersections


def test_split_two_overlapping_squares_produces_correct_segment_count():
    """Two unit squares, one at (0,0) and one at (0.5, 0.5). Each pair of
    opposite edges intersects once; each square's boundary is split into 6
    sub-segments (2 edges are cut once, 2 are cut twice → wait: actually each
    pair contributes 2 crossings, giving 4 total crossings). Both squares get
    crossed on 2 of their 4 edges, each crossed edge splits into 2.
    Resulting segments per square = 2 + 2 + 2 + 2 = 8-ish. We don't assert a
    tight count — just that splits happened and no overlaps were claimed.
    """
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    curves, sources = _collect_curves_with_sources(a, b)
    sub_segs, sub_sources = _split_curves_at_intersections(curves, sources, tol=1e-6)
    # More segments than curves (some got split)
    assert len(sub_segs) > len(curves)
    # Source tags are single letters (no dedup happened)
    for s in sub_sources:
        assert s in ('A', 'B')


def test_split_two_squares_sharing_one_edge_merges_overlap():
    """Two unit squares that share the edge x=1 from y=0 to y=1.
    That shared edge is an overlap: CCX returns it as a single overlap range.
    After splitting and dedup, we should have exactly ONE sub-segment tagged {A,B}
    for the shared portion, not two.
    """
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(1.0, 0.0, 1.0)])
    curves, sources = _collect_curves_with_sources(a, b)
    sub_segs, sub_sources = _split_curves_at_intersections(curves, sources, tol=1e-6)
    # Exactly one sub-segment has both sources.
    both = [s for s in sub_sources if s == 'AB']
    assert len(both) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py::test_split_two_overlapping_squares_produces_correct_segment_count tests/test_boolean2d.py::test_split_two_squares_sharing_one_edge_merges_overlap -v`

Expected: `ImportError: cannot import name '_split_curves_at_intersections'`.

- [ ] **Step 3: Implement `_split_curves_at_intersections`**

Append to `mmcore/topo/brep/boolean2d.py`:

```python
from mmcore.geom._nurbs_knots import split_curve_multiple
from mmcore.geom._nurbs_param_tol import nurbs_curve_param_tolerance


def _split_curves_at_intersections(
    curves: list[NURBSCurveTuple],
    sources: list[str],
    tol: float,
) -> tuple[list[NURBSCurveTuple], list[str]]:
    """Split each curve at all CCX-reported intersections and dedup overlaps.

    Returns (sub_segments, sub_sources) where each source tag is 'A', 'B', or
    'AB' (the last indicates a segment produced by merging an overlap pair).
    """
    isolated, overlaps = nurbs_ccx_multiple(curves, tol=tol)

    # Per-curve list of split parameters (including the overlap range
    # endpoints — overlaps must cause splits at both ends).
    split_params: list[list[float]] = [[] for _ in curves]
    if isolated is not None:
        for rec in isolated:
            c1, c2 = int(rec['curve1_i']), int(rec['curve2_i'])
            u, v = float(rec['u']), float(rec['v'])
            split_params[c1].append(u)
            split_params[c2].append(v)
    if overlaps is not None:
        for rec in overlaps:
            c1, c2 = int(rec['curve1_i']), int(rec['curve2_i'])
            u0, u1 = float(rec['u'][0]), float(rec['u'][1])
            v0, v1 = float(rec['v'][0]), float(rec['v'][1])
            split_params[c1].extend([u0, u1])
            split_params[c2].extend([v0, v1])

    # Dedupe each curve's params using parametric tolerance.
    dedup_params: list[list[float]] = []
    for i, params in enumerate(split_params):
        if not params:
            dedup_params.append([])
            continue
        ptol = float(nurbs_curve_param_tolerance(curves[i], tol))
        params.sort()
        kept = [params[0]]
        for p in params[1:]:
            if p - kept[-1] > ptol:
                kept.append(p)
        dedup_params.append(kept)

    # Split each curve. split_curve_multiple returns [curve] if params is empty.
    all_sub_segs: list[list[NURBSCurveTuple]] = []
    for i, crv in enumerate(curves):
        params = dedup_params[i]
        if params:
            # split_curve_multiple expects interior params only (not the endpoints)
            # but tolerates endpoint params via the ptol dedup above. Use all params.
            pieces = split_curve_multiple(crv, params)
        else:
            pieces = [crv]
        all_sub_segs.append(list(pieces))

    # Dedupe overlap sub-segments: for each overlap, the piece on curve c1
    # between u0 and u1 is geometrically the same as the piece on curve c2
    # between v0 and v1. Keep one, mark source as 'AB', discard the other.
    # We track which (curve_index, sub_index) pairs are "killed" by an overlap
    # and which are upgraded to 'AB'.
    killed: set[tuple[int, int]] = set()
    upgraded: set[tuple[int, int]] = set()

    def _find_sub_index_spanning(params: list[float], base_interval, u0: float, u1: float) -> int | None:
        """Find the sub-segment index whose param range is approximately [u0,u1]."""
        # The piece list after split_curve_multiple has piece boundaries at
        # [t_base_start, p0, p1, ..., pN, t_base_end]. Piece k covers
        # [boundaries[k], boundaries[k+1]].
        t_lo, t_hi = base_interval
        boundaries = [t_lo] + list(params) + [t_hi]
        lo_target, hi_target = min(u0, u1), max(u0, u1)
        for k in range(len(boundaries) - 1):
            bk_lo, bk_hi = boundaries[k], boundaries[k + 1]
            if abs(bk_lo - lo_target) < 10 * tol and abs(bk_hi - hi_target) < 10 * tol:
                return k
        return None

    if overlaps is not None:
        for rec in overlaps:
            c1, c2 = int(rec['curve1_i']), int(rec['curve2_i'])
            u0, u1 = float(rec['u'][0]), float(rec['u'][1])
            v0, v1 = float(rec['v'][0]), float(rec['v'][1])
            k1 = _find_sub_index_spanning(dedup_params[c1], curves[c1].interval(), u0, u1)
            k2 = _find_sub_index_spanning(dedup_params[c2], curves[c2].interval(), v0, v1)
            if k1 is None or k2 is None:
                continue
            # Keep (c1, k1) as 'AB', kill (c2, k2).
            upgraded.add((c1, k1))
            killed.add((c2, k2))

    # Flatten into output lists, applying killed/upgraded sets.
    out_segs: list[NURBSCurveTuple] = []
    out_sources: list[str] = []
    for i, pieces in enumerate(all_sub_segs):
        for k, piece in enumerate(pieces):
            if (i, k) in killed:
                continue
            if (i, k) in upgraded:
                out_sources.append('AB')
            else:
                out_sources.append(sources[i])
            out_segs.append(piece)

    return out_segs, out_sources
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py::test_split_two_overlapping_squares_produces_correct_segment_count tests/test_boolean2d.py::test_split_two_squares_sharing_one_edge_merges_overlap -v`

Expected: both PASS. If the overlap test fails (likely — the index-matching is delicate), inspect the CCX overlap output and adjust `_find_sub_index_spanning` to handle the actual parameter ordering (parameters on curve 2 may run in reverse).

- [ ] **Step 5: Run all tests**

Run: `.venv/bin/python -m pytest tests/test_point_in_region.py tests/test_boolean2d.py -v`

Expected: all pass.

- [ ] **Step 6: Commit**

```bash
git add mmcore/topo/brep/boolean2d.py tests/test_boolean2d.py
git commit -m "feat(boolean2d): _split_curves_at_intersections with overlap dedup"
```

---

## Task 5: `_build_arrangement` — scratch DCEL

Builds a lightweight half-edge graph from the sub-segments, sorts outgoing HEs by tangent angle at each vertex, links `next = twin.ccw_prev`, walks loops to enumerate faces, and identifies the unbounded face.

**Files:**
- Modify: `mmcore/topo/brep/boolean2d.py` (append)
- Modify: `tests/test_boolean2d.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_boolean2d.py`:

```python
from mmcore.topo.brep.boolean2d import _build_arrangement


def test_build_arrangement_two_overlapping_squares_has_expected_face_count():
    """Two unit squares at (0,0) and (0.5, 0.5). The arrangement has 4 bounded
    regions (A only, B only, A∩B, and parts) — actually 3 bounded regions plus
    1 unbounded. Precise count depends on the arrangement; assert ≥3 bounded.
    """
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    curves, sources = _collect_curves_with_sources(a, b)
    sub_segs, sub_sources = _split_curves_at_intersections(curves, sources, tol=1e-6)
    arr = _build_arrangement(sub_segs, sub_sources, tol=1e-6)
    bounded_count = sum(1 for f in arr.faces if not f.unbounded)
    assert bounded_count >= 3
    # Every half-edge has a face assigned
    for he in arr.half_edges:
        assert he.face is not None
    # Exactly one unbounded face
    assert sum(1 for f in arr.faces if f.unbounded) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py::test_build_arrangement_two_overlapping_squares_has_expected_face_count -v`

Expected: `ImportError: cannot import name '_build_arrangement'`.

- [ ] **Step 3: Implement `_build_arrangement`**

Append to `mmcore/topo/brep/boolean2d.py`:

```python
from dataclasses import dataclass, field


@dataclass
class _ArrHalfEdge:
    idx: int                     # position in the half-edges list
    seg_idx: int                 # sub-segment this HE corresponds to
    forward: bool                # True if walks in segment's natural direction
    origin_vid: int              # tail vertex id (where the HE starts)
    head_vid: int                # head vertex id (where the HE ends)
    angle: float                 # outgoing tangent angle at origin, in (-π, π]
    twin: int | None = None
    next: int | None = None
    prev: int | None = None
    face: int | None = None
    ccw_prev: int | None = None  # helper link: previous HE CCW around origin
    sources: set[str] = field(default_factory=set)


@dataclass
class _ArrFace:
    idx: int
    hes: list[int]  # half-edges forming this face's boundary cycle
    unbounded: bool = False


@dataclass
class _Arrangement:
    vertices: list[np.ndarray]          # vid → xy point
    sub_segments: list[NURBSCurveTuple] # seg_idx → NURBSCurveTuple
    sources: list[str]                  # seg_idx → 'A' | 'B' | 'AB'
    half_edges: list[_ArrHalfEdge]
    faces: list[_ArrFace]


def _build_arrangement(
    sub_segments: list[NURBSCurveTuple],
    sub_sources: list[str],
    tol: float,
) -> _Arrangement:
    """Lightweight in-memory DCEL for the noded + dedup'd sub-segments."""
    # 1) Vertex pool: grid-hash the endpoints of every sub-segment.
    vertices: list[np.ndarray] = []
    vid_of: dict[tuple[int, int], int] = {}

    def _vid(p: np.ndarray) -> int:
        key = (round(float(p[0]) / tol), round(float(p[1]) / tol))
        if key not in vid_of:
            vid_of[key] = len(vertices)
            vertices.append(np.asarray(p, dtype=float))
        return vid_of[key]

    # 2) Build HEs with angles.
    half_edges: list[_ArrHalfEdge] = []
    for seg_idx, seg in enumerate(sub_segments):
        t0, t1 = seg.interval()
        start_ev = evaluate_nurbs_curve(seg, t0, 1)
        end_ev = evaluate_nurbs_curve(seg, t1, 1)
        p0 = np.asarray(start_ev['C'], dtype=float)
        p1 = np.asarray(end_ev['C'], dtype=float)
        t0_vec = np.asarray(start_ev['C1'], dtype=float)
        t1_vec = np.asarray(end_ev['C1'], dtype=float)

        v0 = _vid(p0)
        v1 = _vid(p1)
        if v0 == v1:
            # Degenerate (start==end vertex). Skip this segment — it contributes
            # nothing to the arrangement. Shouldn't happen for well-formed input.
            continue

        ang_fwd = float(np.arctan2(t0_vec[1], t0_vec[0]))
        # Reverse HE's outgoing tangent at v1 is -t1_vec.
        ang_rev = float(np.arctan2(-t1_vec[1], -t1_vec[0]))

        fwd_idx = len(half_edges)
        rev_idx = fwd_idx + 1
        sources = {sub_sources[seg_idx]} if sub_sources[seg_idx] != 'AB' else {'A', 'B'}

        half_edges.append(_ArrHalfEdge(
            idx=fwd_idx, seg_idx=seg_idx, forward=True,
            origin_vid=v0, head_vid=v1, angle=ang_fwd,
            twin=rev_idx, sources=sources,
        ))
        half_edges.append(_ArrHalfEdge(
            idx=rev_idx, seg_idx=seg_idx, forward=False,
            origin_vid=v1, head_vid=v0, angle=ang_rev,
            twin=fwd_idx, sources=sources,
        ))

    # 3) For each vertex, sort outgoing HEs CCW by angle.
    outgoing: dict[int, list[int]] = {}
    for he in half_edges:
        outgoing.setdefault(he.origin_vid, []).append(he.idx)
    for vid, hids in outgoing.items():
        hids.sort(key=lambda i: half_edges[i].angle)
        m = len(hids)
        for j in range(m):
            half_edges[hids[(j + 1) % m]].ccw_prev = hids[j]

    # 4) Link next = twin.ccw_prev  (standard "face on left" rule).
    for he in half_edges:
        twin = half_edges[he.twin]
        he.next = twin.ccw_prev
        half_edges[he.next].prev = he.idx

    # 5) Walk loops to enumerate faces.
    faces: list[_ArrFace] = []
    for he in half_edges:
        if he.face is not None:
            continue
        fidx = len(faces)
        cycle: list[int] = []
        cur = he.idx
        while half_edges[cur].face is None:
            half_edges[cur].face = fidx
            cycle.append(cur)
            cur = half_edges[cur].next
            if cur == he.idx:
                break
        faces.append(_ArrFace(idx=fidx, hes=cycle))

    # 6) Identify the unbounded face: find the vertex with min (y, x);
    #    the outgoing HE with the smallest angle there has its twin's face
    #    as the unbounded one.
    if not vertices:
        return _Arrangement(vertices=vertices, sub_segments=list(sub_segments),
                            sources=list(sub_sources), half_edges=half_edges, faces=faces)
    extreme_vid = min(range(len(vertices)),
                      key=lambda i: (vertices[i][1], vertices[i][0]))
    extreme_outs = outgoing.get(extreme_vid, [])
    if extreme_outs:
        ext_he_idx = min(extreme_outs, key=lambda i: half_edges[i].angle)
        twin_face_idx = half_edges[half_edges[ext_he_idx].twin].face
        if twin_face_idx is not None:
            faces[twin_face_idx].unbounded = True

    return _Arrangement(
        vertices=vertices,
        sub_segments=list(sub_segments),
        sources=list(sub_sources),
        half_edges=half_edges,
        faces=faces,
    )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py::test_build_arrangement_two_overlapping_squares_has_expected_face_count -v`

Expected: `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add mmcore/topo/brep/boolean2d.py tests/test_boolean2d.py
git commit -m "feat(boolean2d): _build_arrangement (scratch DCEL)"
```

---

## Task 6: `_classify_faces` — per-face PIP

For each bounded face, pick an interior sample and run `point_in_region` against the original curves of A and of B.

**Files:**
- Modify: `mmcore/topo/brep/boolean2d.py` (append)
- Modify: `tests/test_boolean2d.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_boolean2d.py`:

```python
from mmcore.topo.brep.boolean2d import _classify_faces


def test_classify_faces_two_overlapping_squares_gives_expected_labels():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    curves_a, _ = _collect_curves_with_sources(a, BRep())
    curves_b, _ = _collect_curves_with_sources(BRep(), b)
    curves, sources = _collect_curves_with_sources(a, b)
    sub_segs, sub_sources = _split_curves_at_intersections(curves, sources, tol=1e-6)
    arr = _build_arrangement(sub_segs, sub_sources, tol=1e-6)
    labels = _classify_faces(arr, curves_a, curves_b, tol=1e-6)
    # Exactly one (True, True) face (the intersection lens)
    inAB = [k for k, v in labels.items() if v == (True, True)]
    assert len(inAB) == 1
    # At least one (True, False) and one (False, True)
    assert any(v == (True, False) for v in labels.values())
    assert any(v == (False, True) for v in labels.values())
    # Unbounded face is (False, False)
    unb_idx = next(f.idx for f in arr.faces if f.unbounded)
    assert labels[unb_idx] == (False, False)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py::test_classify_faces_two_overlapping_squares_gives_expected_labels -v`

Expected: `ImportError: cannot import name '_classify_faces'`.

- [ ] **Step 3: Implement `_classify_faces`**

Append to `mmcore/topo/brep/boolean2d.py`:

```python
def _classify_faces(
    arr: _Arrangement,
    curves_a: list[NURBSCurveTuple],
    curves_b: list[NURBSCurveTuple],
    tol: float,
) -> dict[int, tuple[bool, bool]]:
    """Assign (inA, inB) to each face in the arrangement.

    For the unbounded face, returns (False, False) by definition.
    For each bounded face, picks an interior sample from any half-edge and
    runs point_in_region against the original A and B curves.
    """
    labels: dict[int, tuple[bool, bool]] = {}
    for face in arr.faces:
        if face.unbounded:
            labels[face.idx] = (False, False)
            continue
        if not face.hes:
            labels[face.idx] = (False, False)
            continue
        he = arr.half_edges[face.hes[0]]
        seg = arr.sub_segments[he.seg_idx]
        t0, t1 = seg.interval()
        t_mid = 0.5 * (t0 + t1)
        ev = evaluate_nurbs_curve(seg, t_mid, 1)
        mid = np.asarray(ev['C'], dtype=float)
        tan = np.asarray(ev['C1'], dtype=float)
        # forward/backward orientation
        if not he.forward:
            tan = -tan
        # inward normal = tan rotated 90° CCW = (-ty, tx)
        n = np.array([-tan[1], tan[0], 0.0], dtype=float)
        nn = float(np.linalg.norm(n))
        if nn < 1e-30:
            labels[face.idx] = (False, False)
            continue
        n = n / nn
        eps = tol * 10.0
        sample = mid + eps * n
        try:
            inA = point_in_region(sample, curves_a, tol=tol) if curves_a else False
        except RuntimeError:
            inA = False
        try:
            inB = point_in_region(sample, curves_b, tol=tol) if curves_b else False
        except RuntimeError:
            inB = False
        labels[face.idx] = (inA, inB)
    return labels
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py::test_classify_faces_two_overlapping_squares_gives_expected_labels -v`

Expected: `PASSED`.

- [ ] **Step 5: Commit**

```bash
git add mmcore/topo/brep/boolean2d.py tests/test_boolean2d.py
git commit -m "feat(boolean2d): _classify_faces via point_in_region"
```

---

## Task 7: `_select_kept_faces`, `_extract_island_loops`

Apply the op rule to choose which arrangement faces become output material, then group connected kept faces into islands and extract each island's outer + hole loops.

**Files:**
- Modify: `mmcore/topo/brep/boolean2d.py` (append)
- Modify: `tests/test_boolean2d.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_boolean2d.py`:

```python
from mmcore.topo.brep.boolean2d import _select_kept_faces, _extract_island_loops


def test_select_kept_faces_union():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    curves_a, _ = _collect_curves_with_sources(a, BRep())
    curves_b, _ = _collect_curves_with_sources(BRep(), b)
    curves, sources = _collect_curves_with_sources(a, b)
    sub_segs, sub_sources = _split_curves_at_intersections(curves, sources, tol=1e-6)
    arr = _build_arrangement(sub_segs, sub_sources, tol=1e-6)
    labels = _classify_faces(arr, curves_a, curves_b, tol=1e-6)
    kept = _select_kept_faces(arr, labels, 'union')
    # For union: all bounded faces that are inA or inB are kept.
    for face in arr.faces:
        if face.unbounded:
            assert face.idx not in kept
        else:
            inA, inB = labels[face.idx]
            if inA or inB:
                assert face.idx in kept
            else:
                assert face.idx not in kept


def test_extract_island_loops_overlapping_squares_union():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    curves_a, _ = _collect_curves_with_sources(a, BRep())
    curves_b, _ = _collect_curves_with_sources(BRep(), b)
    curves, sources = _collect_curves_with_sources(a, b)
    sub_segs, sub_sources = _split_curves_at_intersections(curves, sources, tol=1e-6)
    arr = _build_arrangement(sub_segs, sub_sources, tol=1e-6)
    labels = _classify_faces(arr, curves_a, curves_b, tol=1e-6)
    kept = _select_kept_faces(arr, labels, 'union')
    islands = _extract_island_loops(arr, kept)
    # Union of two overlapping unit squares ⇒ 1 island, 1 outer loop, 0 holes.
    assert len(islands) == 1
    outer_loop_hes, hole_loops_hes = islands[0]
    assert len(outer_loop_hes) >= 4  # at least 4 HEs in the outer boundary
    assert hole_loops_hes == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py::test_select_kept_faces_union tests/test_boolean2d.py::test_extract_island_loops_overlapping_squares_union -v`

Expected: both fail with `ImportError`.

- [ ] **Step 3: Implement `_select_kept_faces` and `_extract_island_loops`**

Append to `mmcore/topo/brep/boolean2d.py`:

```python
def _select_kept_faces(
    arr: _Arrangement,
    labels: dict[int, tuple[bool, bool]],
    op: str,
) -> set[int]:
    """Apply the op rule. Returns a set of bounded face ids that are kept."""
    rules = {
        'union':        lambda inA, inB: inA or inB,
        'intersection': lambda inA, inB: inA and inB,
        'difference':   lambda inA, inB: inA and not inB,
        'xor':          lambda inA, inB: inA != inB,
    }
    if op not in rules:
        raise ValueError(f"unknown op {op!r}")
    rule = rules[op]
    kept: set[int] = set()
    for face in arr.faces:
        if face.unbounded:
            continue
        inA, inB = labels[face.idx]
        if rule(inA, inB):
            kept.add(face.idx)
    return kept


def _extract_island_loops(
    arr: _Arrangement,
    kept: set[int],
) -> list[tuple[list[int], list[list[int]]]]:
    """Group kept faces into islands and extract their boundary loops.

    Returns a list of (outer_loop_hes, [hole_loop_hes, ...]) tuples. Each
    loop is a list of half-edge indices forming a closed cycle, with the
    body material on the LEFT of the walk direction (so outer loops are
    CCW in xy plane, hole loops are CW).
    """
    # Island = connected component over "kept-to-kept" edges. Two kept faces
    # on opposite sides of a sub-segment belong to the same island; that
    # sub-segment is interior and its HEs are *not* boundary HEs of the
    # island.
    parent = {f.idx: f.idx for f in arr.faces if f.idx in kept}

    def _find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def _union(a: int, b: int) -> None:
        ra, rb = _find(a), _find(b)
        if ra != rb:
            parent[ra] = rb

    for he in arr.half_edges:
        if he.face is None or he.face not in kept:
            continue
        twin = arr.half_edges[he.twin]
        if twin.face is not None and twin.face in kept:
            _union(he.face, twin.face)

    # Group face ids by root
    island_of: dict[int, list[int]] = {}
    for fid in kept:
        r = _find(fid)
        island_of.setdefault(r, []).append(fid)

    # Build boundary-HE set per island
    islands_out: list[tuple[list[int], list[list[int]]]] = []
    for root, face_ids in island_of.items():
        face_set = set(face_ids)
        # boundary HE: belongs to a face in this island AND its twin's face
        # is *not* in this island
        boundary_hes: set[int] = set()
        for he in arr.half_edges:
            if he.face in face_set:
                twin = arr.half_edges[he.twin]
                if twin.face not in face_set:
                    boundary_hes.add(he.idx)

        # Walk boundary HEs into closed loops. The DCEL "next" pointer walks
        # around a single arrangement face — but an island may span multiple
        # arrangement faces, so at each step we must skip interior edges
        # (edges whose both sides are in the same island).
        #
        # At the head vertex of the current boundary HE h (call the vertex v):
        # the next island-boundary HE is the next outgoing HE at v starting
        # from h.twin and walking CW (via `ccw_prev`, which advances CCW-
        # previous = CW in our convention) until we hit another boundary HE.
        # All these candidates are outgoing HEs at v (same vertex) — do NOT
        # follow .twin (that jumps to a neighbouring vertex).
        visited: set[int] = set()
        loops_hes: list[list[int]] = []
        for start in list(boundary_hes):
            if start in visited:
                continue
            cycle: list[int] = []
            cur = start
            while cur not in visited:
                visited.add(cur)
                cycle.append(cur)
                twin_idx = arr.half_edges[cur].twin
                # Walk CW around cur.head_vid via ccw_prev until we find
                # the next boundary HE (first candidate: twin_idx.ccw_prev).
                nxt = arr.half_edges[twin_idx].ccw_prev
                safety = 0
                while nxt is not None and nxt not in boundary_hes:
                    if nxt == twin_idx:
                        nxt = None
                        break
                    nxt = arr.half_edges[nxt].ccw_prev
                    safety += 1
                    if safety > len(arr.half_edges):
                        nxt = None
                        break
                if nxt is None:
                    break
                cur = nxt
                if cur == start:
                    break
            loops_hes.append(cycle)

        # Classify loops: the one with the largest (absolute) signed area in
        # xy is the outer loop; the rest are holes.
        def _loop_signed_area(loop: list[int]) -> float:
            pts = []
            for hid in loop:
                he = arr.half_edges[hid]
                pts.append(arr.vertices[he.origin_vid])
            xs = np.array([p[0] for p in pts])
            ys = np.array([p[1] for p in pts])
            return 0.5 * float(np.sum(xs * np.roll(ys, -1) - np.roll(xs, -1) * ys))

        if not loops_hes:
            continue
        areas = [abs(_loop_signed_area(l)) for l in loops_hes]
        outer_idx = max(range(len(loops_hes)), key=lambda i: areas[i])
        outer_loop = loops_hes[outer_idx]
        hole_loops = [loops_hes[i] for i in range(len(loops_hes)) if i != outer_idx]
        islands_out.append((outer_loop, hole_loops))

    return islands_out
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py::test_select_kept_faces_union tests/test_boolean2d.py::test_extract_island_loops_overlapping_squares_union -v`

Expected: both PASS. If `test_extract_island_loops_overlapping_squares_union` fails, the most likely bug is in the boundary-walk loop (the "advance to next boundary HE" logic). Print the `cycle` list to inspect; a valid cycle should close (last HE's head vertex equals first HE's origin vertex).

- [ ] **Step 5: Commit**

```bash
git add mmcore/topo/brep/boolean2d.py tests/test_boolean2d.py
git commit -m "feat(boolean2d): _select_kept_faces + _extract_island_loops"
```

---

## Task 8: `_materialize_result` — build the output BRep

Given an arrangement and a list of (outer_loop, hole_loops) tuples of HE indices, construct a fresh `BRep` in the standard 2D form.

**Files:**
- Modify: `mmcore/topo/brep/boolean2d.py` (append)
- Modify: `tests/test_boolean2d.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_boolean2d.py`:

```python
from mmcore.topo.brep.boolean2d import _materialize_result


def test_materialize_result_overlapping_squares_union():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    curves_a, _ = _collect_curves_with_sources(a, BRep())
    curves_b, _ = _collect_curves_with_sources(BRep(), b)
    curves, sources = _collect_curves_with_sources(a, b)
    sub_segs, sub_sources = _split_curves_at_intersections(curves, sources, tol=1e-6)
    arr = _build_arrangement(sub_segs, sub_sources, tol=1e-6)
    labels = _classify_faces(arr, curves_a, curves_b, tol=1e-6)
    kept = _select_kept_faces(arr, labels, 'union')
    islands = _extract_island_loops(arr, kept)
    result = _materialize_result(arr, islands)
    assert _count_body_faces(result) == 1
    assert result.validate() == []
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py::test_materialize_result_overlapping_squares_union -v`

Expected: `ImportError: cannot import name '_materialize_result'`.

- [ ] **Step 3: Implement `_materialize_result`**

Append to `mmcore/topo/brep/boolean2d.py`:

```python
def _materialize_result(
    arr: _Arrangement,
    islands: list[tuple[list[int], list[list[int]]]],
) -> BRep:
    """Build a fresh BRep in standard 2D form from the arrangement islands."""
    result = BRep()
    body = result.new_body(shells=[])
    shell = result.new_shell(faces=[], body=body.id)
    body.shells.append(shell.id)
    wire_face = result.new_face(outer=None, inners=[], shell=shell.id, surf=None)
    shell.faces.append(wire_face.id)

    for outer_loop_hes, hole_loops_hes in islands:
        # Collect all curves walked in this island, in order, from the HE cycles.
        outer_curves = [
            _oriented_subcurve_from_arr(arr, hid) for hid in outer_loop_hes
        ]
        hole_curves_list = [
            [_oriented_subcurve_from_arr(arr, hid) for hid in hole]
            for hole in hole_loops_hes
        ]
        # Use _add_loop_to_brep from Task 1 to build the body face and its holes.
        body_face_id = _add_loop_to_brep(
            result, shell.id, wire_face.id, outer_curves,
            is_body_outer=True,
        )
        for hole_curves in hole_curves_list:
            _add_loop_to_brep(
                result, shell.id, wire_face.id, hole_curves,
                is_body_outer=False, host_face_id=body_face_id,
            )
    return result


def _oriented_subcurve_from_arr(arr: _Arrangement, he_idx: int) -> NURBSCurveTuple:
    """Return the sub-segment curve oriented along the HE's walk direction."""
    from mmcore.geom._nurbs_knots import reverse_curve
    he = arr.half_edges[he_idx]
    seg = arr.sub_segments[he.seg_idx]
    return seg if he.forward else reverse_curve(seg)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py::test_materialize_result_overlapping_squares_union -v`

Expected: `PASSED`. If `result.validate()` returns errors, they will tell you exactly which invariant broke — fix the `_add_loop_to_brep` path accordingly. Common cases: twin symmetry, next/prev wiring, missing `edge.he` field.

- [ ] **Step 5: Commit**

```bash
git add mmcore/topo/brep/boolean2d.py tests/test_boolean2d.py
git commit -m "feat(boolean2d): _materialize_result"
```

---

## Task 9: Public API — `_boolean2d` orchestrator and four ops

Wire up the pipeline. Also handle the empty-input short-circuit cases.

**Files:**
- Modify: `mmcore/topo/brep/boolean2d.py` (append)
- Modify: `tests/test_boolean2d.py` (append)

- [ ] **Step 1: Write the failing test**

Append to `tests/test_boolean2d.py`:

```python
def test_union_two_overlapping_squares_end_to_end():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    result = union(a, b, tol=1e-6)
    assert _count_body_faces(result) == 1
    assert result.validate() == []


def test_intersection_two_overlapping_squares_end_to_end():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    result = intersection(a, b, tol=1e-6)
    assert _count_body_faces(result) == 1
    assert result.validate() == []


def test_union_empty_and_nonempty():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = BRep()
    # Build minimal empty BRep (Body + Shell + wire Face, no body faces)
    body = b.new_body(shells=[])
    shell = b.new_shell(faces=[], body=body.id)
    body.shells.append(shell.id)
    wire = b.new_face(outer=None, inners=[], shell=shell.id, surf=None)
    shell.faces.append(wire.id)
    result = union(a, b, tol=1e-6)
    assert _count_body_faces(result) == 1
    assert result.validate() == []
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py -k "end_to_end or empty_and_nonempty" -v`

Expected: all three fail with `ImportError: cannot import name 'union'` (or similar).

- [ ] **Step 3: Implement the orchestrator and public ops**

Append to `mmcore/topo/brep/boolean2d.py`:

```python
# ---------------------------------------------------------------------------
#  Public API
# ---------------------------------------------------------------------------

def _boolean2d(a: BRep, b: BRep, op: str, tol: float) -> BRep:
    """Run the full pipeline for a single operation.

    Both inputs may be empty (Body + Shell + wire Face only). If both are
    empty the result is an empty BRep. Otherwise the pipeline runs normally
    — nurbs_ccx_multiple returns empty results for single-source inputs, the
    arrangement is still built, and classification/selection proceed as usual.
    """
    # Validate inputs
    for name, brep in (('a', a), ('b', b)):
        errs = brep.validate()
        if errs:
            raise ValueError(f"input BRep {name!r} failed validate(): {errs[0]}")

    # Curves + sources
    curves, sources = _collect_curves_with_sources(a, b)
    curves_a = [c for c, s in zip(curves, sources) if s == 'A']
    curves_b = [c for c, s in zip(curves, sources) if s == 'B']

    # Both empty ⇒ empty result (no curves to build any arrangement).
    if not curves_a and not curves_b:
        return _empty_result_brep()

    # Split + dedup (handles empty isolated/overlaps naturally)
    sub_segs, sub_sources = _split_curves_at_intersections(curves, sources, tol)

    # Build the scratch arrangement
    arr = _build_arrangement(sub_segs, sub_sources, tol)

    # Classify every bounded face (inA, inB)
    labels = _classify_faces(arr, curves_a, curves_b, tol)

    # Apply the op rule
    kept = _select_kept_faces(arr, labels, op)

    # Group into islands
    islands = _extract_island_loops(arr, kept)

    # Materialize the result
    return _materialize_result(arr, islands)


def _empty_result_brep() -> BRep:
    """Empty 2D BRep in standard form: body + shell + wire face, no body faces."""
    brep = BRep()
    body = brep.new_body(shells=[])
    shell = brep.new_shell(faces=[], body=body.id)
    body.shells.append(shell.id)
    wire = brep.new_face(outer=None, inners=[], shell=shell.id, surf=None)
    shell.faces.append(wire.id)
    return brep


def union(a: BRep, b: BRep, tol: float = 1e-6) -> BRep:
    """Union of two 2D regions."""
    return _boolean2d(a, b, 'union', tol)


def intersection(a: BRep, b: BRep, tol: float = 1e-6) -> BRep:
    """Intersection of two 2D regions."""
    return _boolean2d(a, b, 'intersection', tol)


def difference(a: BRep, b: BRep, tol: float = 1e-6) -> BRep:
    """A \\ B (region in A but not in B)."""
    return _boolean2d(a, b, 'difference', tol)


def xor(a: BRep, b: BRep, tol: float = 1e-6) -> BRep:
    """Symmetric difference (in A XOR in B)."""
    return _boolean2d(a, b, 'xor', tol)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py -v`

Expected: all tests pass. If the empty-input test fails, check that `_collect_curves_with_sources` returns empty lists cleanly for an empty BRep (body + shell + wire face but no body faces), and that `_build_arrangement` handles a list with curves from only one source.

- [ ] **Step 5: Commit**

```bash
git add mmcore/topo/brep/boolean2d.py tests/test_boolean2d.py
git commit -m "feat(boolean2d): public union/intersection/difference/xor API"
```

---

## Task 10: T1 + T2 — disjoint rectangles and overlapping circles

The first two spec tests, exercising all four operators on simple inputs.

**Files:**
- Modify: `tests/test_boolean2d.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_boolean2d.py`:

```python
# ---- Spec test T1: disjoint rectangles ----

def test_T1_union_disjoint_rectangles():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(2.0, 0.0, 1.0)])
    r = union(a, b, tol=1e-6)
    assert _count_body_faces(r) == 2
    assert r.validate() == []


def test_T1_intersection_disjoint_rectangles_is_empty():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(2.0, 0.0, 1.0)])
    r = intersection(a, b, tol=1e-6)
    assert _count_body_faces(r) == 0
    assert r.validate() == []


def test_T1_difference_disjoint_rectangles_is_a():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(2.0, 0.0, 1.0)])
    r = difference(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1
    assert r.validate() == []


def test_T1_xor_disjoint_rectangles():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(2.0, 0.0, 1.0)])
    r = xor(a, b, tol=1e-6)
    assert _count_body_faces(r) == 2
    assert r.validate() == []


# ---- Spec test T2: overlapping circles ----

def _circle_region(cx: float, cy: float, r: float) -> BRep:
    return make_region_2d([[circle(center=(cx, cy, 0.0), radius=r)]])


def test_T2_union_overlapping_circles():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(1.0, 0.0, 1.0)
    r = union(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1
    assert r.validate() == []


def test_T2_intersection_overlapping_circles():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(1.0, 0.0, 1.0)
    r = intersection(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1
    assert r.validate() == []


def test_T2_difference_overlapping_circles():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(1.0, 0.0, 1.0)
    r = difference(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1
    assert r.validate() == []


def test_T2_xor_overlapping_circles():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(1.0, 0.0, 1.0)
    r = xor(a, b, tol=1e-6)
    assert _count_body_faces(r) == 2
    assert r.validate() == []
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py -k "T1 or T2" -v`

Expected: all 8 tests pass. **Likely failures and debugging hints:**
- **Disjoint-rectangles union** may show 1 body face if the arrangement merges disjoint components incorrectly, or if `_extract_island_loops`'s union-find groups unrelated faces. Debug by printing `len(islands)` after `_extract_island_loops` — for two disjoint squares it must be 2.
- **Circle tests** may fail with `ImportError` for some helper — re-check that all imports are in place.
- **Difference A\\B** may return 2 body faces instead of 1 if the arrangement's hole-from-a-hole logic mis-groups — check the signed-area-based outer detection in `_extract_island_loops`.

- [ ] **Step 3: Run full test suite to check for regressions**

Run: `.venv/bin/python -m pytest tests/test_point_in_region.py tests/test_boolean2d.py -v`

Expected: all tests pass.

- [ ] **Step 4: Commit**

```bash
git add tests/test_boolean2d.py
git commit -m "test(boolean2d): T1 disjoint rects + T2 overlapping circles"
```

---

## Task 11: T3 + T4 — square-with-hole and shared edge

**Files:**
- Modify: `tests/test_boolean2d.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_boolean2d.py`:

```python
def _cw_square(x0: float, y0: float, side: float) -> list[NURBSCurveTuple]:
    """Build a CW square boundary (suitable for use as a hole loop)."""
    return [
        _line([x0,        y0,        0.0], [x0,        y0 + side, 0.0]),
        _line([x0,        y0 + side, 0.0], [x0 + side, y0 + side, 0.0]),
        _line([x0 + side, y0 + side, 0.0], [x0 + side, y0,        0.0]),
        _line([x0 + side, y0,        0.0], [x0,        y0,        0.0]),
    ]


def test_T3_union_square_with_hole_and_disk():
    # Outer 4x4 square with a unit-square hole at (1.5,1.5)
    a = make_region_2d([
        _square_ccw(0.0, 0.0, 4.0),
        _cw_square(1.5, 1.5, 1.0),
    ])
    # A disk that straddles the hole
    b = _circle_region(2.0, 2.0, 1.25)
    r = union(a, b, tol=1e-6)
    # Exactly 1 island; the hole may be fully or partially filled
    assert _count_body_faces(r) == 1
    assert r.validate() == []


def test_T4_union_two_squares_sharing_one_edge():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(1.0, 0.0, 1.0)])
    r = union(a, b, tol=1e-6)
    # Result is a 1×2 rectangle — 1 body face
    assert _count_body_faces(r) == 1
    assert r.validate() == []


def test_T4_intersection_two_squares_sharing_one_edge_is_empty():
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(1.0, 0.0, 1.0)])
    r = intersection(a, b, tol=1e-6)
    assert _count_body_faces(r) == 0
    assert r.validate() == []
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py -k "T3 or T4" -v`

Expected: all 3 tests pass. Shared-edge tests exercise the overlap dedup path — if they fail, the bug is in `_split_curves_at_intersections`'s `_find_sub_index_spanning` (overlap param direction) or in how `_build_arrangement` treats an edge with `sources={'A','B'}` when both faces on its twin side are kept.

- [ ] **Step 3: Commit**

```bash
git add tests/test_boolean2d.py
git commit -m "test(boolean2d): T3 square+hole+disk, T4 shared edge"
```

---

## Task 12: T5 + T6 — nested and tangent circles

**Files:**
- Modify: `tests/test_boolean2d.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_boolean2d.py`:

```python
def test_T5_nested_union_is_outer():
    a = _circle_region(0.0, 0.0, 0.3)         # small disk inside
    b = make_region_2d([_square_ccw(-2.0, -2.0, 4.0)])  # big square
    r = union(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1  # just the square


def test_T5_nested_intersection_is_inner():
    a = _circle_region(0.0, 0.0, 0.3)
    b = make_region_2d([_square_ccw(-2.0, -2.0, 4.0)])
    r = intersection(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1  # the small circle


def test_T5_nested_difference_is_square_with_hole():
    a = make_region_2d([_square_ccw(-2.0, -2.0, 4.0)])  # big square
    b = _circle_region(0.0, 0.0, 0.3)                   # small circle
    r = difference(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1
    body_face = next(f for f in r.F.values() if f.outer is not None)
    assert len(body_face.inners) == 1  # circle became a hole
    assert r.validate() == []


def test_T6_tangent_circles_union():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(2.0, 0.0, 1.0)  # touch at (1, 0)
    r = union(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1  # one merged figure-eight
    assert r.validate() == []


def test_T6_tangent_circles_intersection_is_empty():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(2.0, 0.0, 1.0)
    r = intersection(a, b, tol=1e-6)
    assert _count_body_faces(r) == 0
    assert r.validate() == []
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py -k "T5 or T6" -v`

Expected: all 5 tests pass. T5 nested tests stress the classification (no intersections ⇒ empty `sub_segs` besides original ones, so the arrangement is essentially the two disjoint loops; classification separates the faces). T6 tangent tests stress CCX's tangent handling.

- [ ] **Step 3: Commit**

```bash
git add tests/test_boolean2d.py
git commit -m "test(boolean2d): T5 nested, T6 tangent circles"
```

---

## Task 13: T7 + T8 — identical inputs and composition

**Files:**
- Modify: `tests/test_boolean2d.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_boolean2d.py`:

```python
def test_T7_identical_inputs_union_is_a():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(0.0, 0.0, 1.0)
    r = union(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1
    assert r.validate() == []


def test_T7_identical_inputs_intersection_is_a():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(0.0, 0.0, 1.0)
    r = intersection(a, b, tol=1e-6)
    assert _count_body_faces(r) == 1
    assert r.validate() == []


def test_T7_identical_inputs_difference_is_empty():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(0.0, 0.0, 1.0)
    r = difference(a, b, tol=1e-6)
    assert _count_body_faces(r) == 0
    assert r.validate() == []


def test_T7_identical_inputs_xor_is_empty():
    a = _circle_region(0.0, 0.0, 1.0)
    b = _circle_region(0.0, 0.0, 1.0)
    r = xor(a, b, tol=1e-6)
    assert _count_body_faces(r) == 0
    assert r.validate() == []


def test_T8_composition_union_then_intersection():
    """(square ∪ triangle) ∩ circle should round-trip through the API."""
    square = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    triangle = make_region_2d([[
        _line([1.0, 0.0, 0.0], [2.0, 0.0, 0.0]),
        _line([2.0, 0.0, 0.0], [1.5, 1.0, 0.0]),
        _line([1.5, 1.0, 0.0], [1.0, 0.0, 0.0]),
    ]])
    step1 = union(square, triangle, tol=1e-6)
    assert step1.validate() == []
    c = _circle_region(1.0, 0.5, 1.2)
    step2 = intersection(step1, c, tol=1e-6)
    assert step2.validate() == []
    # At least one island remains
    assert _count_body_faces(step2) >= 1
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py -k "T7 or T8" -v`

Expected: all 5 tests pass. T7 exercises the every-edge-is-an-overlap path in `_split_curves_at_intersections`. T8 verifies that the output BRep format is accepted as input — if this fails, check that `_collect_curves_with_sources` correctly walks all edges of the post-op BRep without double-counting.

- [ ] **Step 3: Commit**

```bash
git add tests/test_boolean2d.py
git commit -m "test(boolean2d): T7 identical inputs, T8 composition"
```

---

## Task 14: T9 + T10 — surface-derived input and cross-cutting validate

**Files:**
- Modify: `tests/test_boolean2d.py` (append)

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_boolean2d.py`:

```python
def test_T9_surface_derived_input_accepted_by_boolean():
    """A BRep built from a planar NURBS surface via make_face_from_surface
    should be a valid input to the boolean ops — proves that the pipeline
    is agnostic to how the input BRep was constructed.
    """
    from mmcore.geom._nurbs_eval import NURBSSurfaceTuple
    # trivial planar surface: the z=0 unit square
    surf = NURBSSurfaceTuple(
        order_u=2, order_v=2,
        knot_u=np.array([0.0, 0.0, 1.0, 1.0]),
        knot_v=np.array([0.0, 0.0, 1.0, 1.0]),
        control_points=np.array([
            [[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            [[1.0, 0.0, 0.0], [1.0, 1.0, 0.0]],
        ]),
        weights=np.ones((2, 2)),
    )
    a = BRep()
    a.make_face_from_surface(surf)
    # make_face_from_surface creates a body face with outer loop — valid input
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    r = union(a, b, tol=1e-6)
    assert r.validate() == []
    assert _count_body_faces(r) >= 1


def test_T10_every_prior_test_output_validates():
    """T10 is the cross-cutting invariant: every public-API call in every
    prior test must have produced a BRep that satisfies validate(). This is
    already asserted inline per-test; this test just re-runs the most
    commonly-broken cases in a single-function sanity check.
    """
    a = make_region_2d([_square_ccw(0.0, 0.0, 1.0)])
    b = make_region_2d([_square_ccw(0.5, 0.5, 1.0)])
    for op_fn in (union, intersection, difference, xor):
        r = op_fn(a, b, tol=1e-6)
        errs = r.validate()
        assert errs == [], f"{op_fn.__name__} produced invalid BRep: {errs}"
```

- [ ] **Step 2: Run tests**

Run: `.venv/bin/python -m pytest tests/test_boolean2d.py -k "T9 or T10" -v`

Expected: both tests pass.

- [ ] **Step 3: Run the FULL test suite**

Run: `.venv/bin/python -m pytest tests/test_point_in_region.py tests/test_boolean2d.py -v`

Expected: every test green. Report the final count.

- [ ] **Step 4: Commit**

```bash
git add tests/test_boolean2d.py
git commit -m "test(boolean2d): T9 surface-derived + T10 validate sanity"
```

---

## Summary of deliverables

After Task 14, the repo contains:
- `mmcore/topo/brep/boolean2d.py` — `point_in_region` (pre-existing), `make_region_2d`, `_collect_curves_with_sources`, `_split_curves_at_intersections`, `_build_arrangement`, `_classify_faces`, `_select_kept_faces`, `_extract_island_loops`, `_materialize_result`, `_boolean2d`, and the four public ops (`union`, `intersection`, `difference`, `xor`).
- `tests/test_boolean2d.py` — `make_region_2d` tests (3 cases), pipeline unit tests (6 cases), and spec tests T1–T10 (~18 cases). Total: ~27 tests, plus the 13 existing `test_point_in_region.py` tests.

No new Cython. No new core primitives. Everything is pure Python on top of existing infrastructure.

## Out of scope (per spec)

- 3D Boolean operations.
- Performance tuning (flood-fill classification, propagation).
- `Region2D` wrapper type.
- Fuzzing / property-based tests.
- Visualization helpers.
- Multi-input `union(a, b, c, ...)`.

These are explicit follow-ups if and when needed.

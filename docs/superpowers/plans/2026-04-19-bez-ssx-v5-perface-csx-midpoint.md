# Plan — bez_ssx v5: per-cut-face boundary CSX + midpoint fallback

**Status:** pending implementation
**Companion design doc:** [`bez-ssx-v5.md`](../designs/bez-ssx-v5.md) — will be updated §6 / §6.5 at iteration close
**Measurements log:** [`bez-ssx-v5-measurements.md`](../designs/bez-ssx-v5-measurements.md)

## Goal

Close the case-6 interior-loop gap. The current implementation silently drops
cells with no boundary crossings when no certificate fires, so an intersection
loop strictly interior to the top-level cell is never discovered. The fix is to
bring the dispatch into line with Krishnan-Manocha §5.1 (`237748.237751.pdf`
pg. 90–93): guided cuts at existing boundary crossings, plus a **midpoint
fallback** when no productive cut exists. Boundary CSX runs at every newly
introduced cut face so freshly exposed crossings propagate down the tree.

## What's actually broken today

- `bez_ssx` main loop at `_bez_ssx5.py:1925-1992` terminates a cell that
  reaches the subdivision branch with no productive crossings: `_choose_multi_cut`
  returns `(None, None)` (its `len(crossings) ≤ 2` guard fires) and the `if
  cut_axis is None: ... continue` branch exits without further subdivision.
- `_find_ssx_boundary_zeros` runs only at top level (`_bez_ssx5.py:1884`).
  Subdivision CSXs the single cut isoline (`_bez_ssx5.py:2005-2009`) but never
  treats new cut faces as a fresh source of boundary crossings in a structured
  way — the results feed back only into the parent's cut decision.
- No midpoint fallback exists: there's no path for the algorithm to make
  progress when a cell has no productive crossings but cannot certify
  loop-freeness either.

## Algorithm (paper §5.1 translated to 4D)

```
while stack:
    cell = stack.pop()

    # --- terminate by certificate (our additions over the paper) ---
    if _check_loop_free(cell):          trace_by_registrations(cell); continue
    if _check_tangency(cell) is True:   deflate_and_trace(cell);      continue
    if cell.depth >= max_depth or _box_below_atol(cell.box, atol):
        emit_as_points(cell);                                         continue

    # --- guided subdivision (paper case 3.else) ---
    productive = [c for c in cell.crossings if _pinned_count(c, cell.box) == 1]
    axis, cuts = _choose_multi_cut_axis(productive, cell.box)   # None,None if stuck

    if axis is not None:
        strips, new_faces = _multi_cut_into_strips(cell, axis, cuts)
        for face in new_faces:
            new_crossings = _csx_on_cut_face(face, cell.g1.surface, cell.g2.surface)
            _register_crossings_on_face(new_crossings, face)  # both adjacent strips
        for s in strips:
            _classify_all(s)  # §4 classification per sub-cell
            stack.append(s)
        continue

    # --- midpoint fallback (paper case 3) ---
    axis = _pick_midpoint_axis(cell)
    left, right, new_face = _midpoint_cut(cell, axis)
    new_crossings = _csx_on_cut_face(new_face, cell.g1.surface, cell.g2.surface)
    _register_crossings_on_face(new_crossings, new_face)
    _classify_all(left); _classify_all(right)
    stack.extend([left, right])
```

### Pinned-count classification

A crossing `c` returned by the CSX at a new cut face sits somewhere in the
cell's local `[0,1]⁴`. Count how many coordinates are exactly `0` or `1` within
tolerance.

- **1-pinned:** lives in the interior of a face. Legitimate new boundary
  crossing — driver for the next subdivision, plus a BoundaryPoint used for
  tracing.
- **≥2-pinned:** lives on an edge/corner of the 4D cube. Either coincides with
  a parent-cut point (already registered and propagated down by parent's
  distribution) or is a genuine multi-axis corner on the original `[0,1]⁴`
  boundary. Kept as a BoundaryPoint for tracing/registration but **never feeds
  the next cut decision**: any cut at its pinned coordinate is a zero-width
  strip.

Invariant-C dedup (current `_dedup_crossings` policy) handles any stuv
collision between new and inherited.

### Multi-cut — one axis, every 1-pinned crossing's value

Matches paper §5.1 and current `bez-ssx-v5.md` §6.5. For the chosen axis `a`,
cuts are the distinct 1-pinned crossing values along `a` that lie in
`(min_margin, 1 − min_margin)` locally. Axis choice: prefer the axis with the
most such cuts; tiebreak by widest spread.

### Midpoint fallback — one axis, local 0.5

Paper §5.1 case 3. **Two** sub-cells, not 16. User-confirmed choice.

```python
def _pick_midpoint_axis(cell) -> int:
    """TODO: pick the axis to cut at 0.5. Three plausible rules:
       (a) round-robin — `return cell.depth % 4`
       (b) largest local span — pick `argmax(box[i].hi - box[i].lo)`
       (c) longest-uncut streak — track which axis parent cut; avoid it
       Default to (a) until case experience suggests otherwise.
    """
    return cell.depth % 4
```

### Boundary CSX per cut face (not per sub-cell)

Paper §5.1: "compute new xsection points on each dividing line". We do this
per new cut face, **not** per sub-cell. The `k` cuts on axis `a` produce `k`
new cut faces; each cut face is shared by exactly 2 adjacent strips; its CSX
results register on both.

The 6 inherited faces (in a multi-cut, more generally `8 - k_new` inherited)
are not re-CSX'd. Parent-to-child inheritance via the strip distribution
(`_bez_ssx5.py:2108-2109`) covers them. This is paper-faithful and cheaper
than rediscovering every sub-cell boundary from scratch.

## Implementation steps (iterative, measure between each)

**Step 1 — per-sub-cell boundary CSX as the foundation.**
The only thing that matters algorithmically in this plan. Until this is in
place, every other change is sand on broken concrete.

1. Extract `_csx_on_cut_face(cut_face, cell)` helper. Takes the face
   specification (which axis is pinned, which surface the isoline comes from,
   the global param value), runs `bez_csx` on the isoline against the opposite
   surface, converts results to global 4D stuv BoundaryPoints, returns them.
   This replaces the inline CSX block at `_bez_ssx5.py:2005-2013` and the
   conversion logic currently sitting in `_isoline_csx_to_global`.
2. Rewire the existing multi-cut path to call `_csx_on_cut_face` once per new
   cut face. Results register on BOTH adjacent strips (shared `PartitionCurve`).
3. Add the midpoint-fallback branch — triggered when productive crossings are
   absent OR `_choose_multi_cut_axis` returns None. Uses `_csx_on_cut_face` on
   the single new midcut face. Paper §5.1 case 3: 2 sub-cells, 1 axis.
4. Result: *every* sub-cell produced by subdivision is exposed to boundary
   CSX on its newly-created face. No code path creates a sub-cell without
   running CSX on its new boundary.

*Expected:* case 6 closes to 2/2. Cases 1-5 untouched (they never needed the
fallback; multi-cut path behaviour is preserved because `_csx_on_cut_face` is
just the refactored body of the existing inline CSX).

*Verification before step 2:*
- Instrument the main loop to log every sub-cell's `depth`, `box`, `crossings
  count`, and the new-cut-face CSX result count. Run case 6 and confirm:
  (a) at least one midpoint fallback is taken, (b) the new cut face for that
  fallback yields crossings, (c) those crossings have `pinned_count == 1`,
  (d) they get classified and registered, (e) the next iteration certifies
  loop-freeness in the strip containing the loop fragment.
- Cases 1-5: branch counts and residuals within the baseline ranges from
  `bez-ssx-v5-measurements.md`.

**Step 2 — drop `len ≤ 2` guard in `_choose_multi_cut`, add 1-pinned filter.**
Pre-filter `cell.crossings` to 1-pinned before handing to `_choose_multi_cut`.
Remove the early-return guard — it's redundant now that the midpoint fallback
handles the empty-productive case.
*Expected:* no behaviour change if step 1 is correct. Confirmation that the
filter is the right way to classify "productive" crossings.

**Step 3 — slow-convergence trigger (optional).**
If multi-cut produces a strip whose size on the cut axis is > 0.9 × parent,
flag as slow; re-queue it for midpoint fallback instead of another multi-cut.
Paper §5.1 case 3 includes this. Skip unless a case needs it.

**Step 4 — geometric floor.**
If all `box[i].hi - box[i].lo < atol`, emit remaining crossings as
`SSXPoint` and skip subdivision. Prevents pathological recursion when the
certificate never fires.

## Acceptance criteria

Per the measurements log protocol:

| case        | expected after step 2 | note |
|-------------|-----------------------|------|
| planes      | 1/1, machine-precision residual | unchanged |
| transversal | 1/1, ≤ 1e-8 residual | unchanged |
| tangential  | 1/1, machine-precision | unchanged |
| overlaps    | 4/2 MISMATCH | out of scope (§5 overlap gap) |
| case5       | 2/2, ≤ 1e-7 residual | unchanged |
| **case6**   | **2/2, ≤ 1e-7 residual** | **loop recovered** |

CSX unit tests: `tests/test_bez_csx4.py` 12/12 pass throughout.

## Risks

- **Midpoint cascade.** A midpoint split can produce sub-cells that also hit
  the fallback — recursion continues until either a certificate fires or
  `max_depth` is hit. `max_depth=12` allows 4096 midpoint-only sub-cells worst
  case; in practice certificates fire well before that.
- **Parent inheritance completeness.** Paper's inheritance assumes top-level
  CSX found every root on every face. If an L0 CSX call missed a root,
  children won't rediscover the missing boundary point. Accepting this matches
  paper guarantees.
- **Axis-pick heuristic.** If `_pick_midpoint_axis` consistently picks an axis
  orthogonal to the hidden loop's principal direction, the loop may need deep
  midpoint cascades before being exposed. Round-robin is the safest default.

## Out of scope

- Overlap `BoundaryPoint`/registration integration (design §5 gap that keeps
  overlaps 4/2). Separate iteration.
- §4.2 Φ-side classifier for tangent cells (currently handled by legacy
  `.face`-based `_pair_crossings_for_tracing`). Separate iteration.
- `_assemble_fragments` → §9 adjacency walk revision. Already attempted
  (iter-13, rolled back). Revisit after case 6 lands cleanly.

## References

- Krishnan & Manocha 1997 (`237748.237751.pdf`) — §5.1 DomainDecomposition
  pseudocode pg. 92–93, convergence discussion pg. 94.
- Cheng et al. 2023 IATA (`3592452-2.pdf`) — Lemma 5 (TΨᵢ monotonicity
  certificate), Lemma 2 (regulated Φ system), Krawczyk tangency test.
- This repo's `bez-ssx-v5.md` §6 (cell lifecycle) and §6.5 (multi-crossing cut
  — currently the only subdivision branch; will be updated to "guided OR
  midpoint-fallback").

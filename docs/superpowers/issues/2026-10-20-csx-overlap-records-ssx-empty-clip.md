# CSX overlap records (L63) and the SSX empty-clip rule (L64)

Decided with the owner on 2026-10-20, after comparing mmcore with NVIDIA SMLib.

## L63 — CSX output records

**Contract.**

- A point is `(t, u, v)`.
- An overlap is a pair of `(t, u, v)`: start and end in curve order. `t_range`,
  `u_range`, `v_range` hold the paired ends; they are never per-axis min/max.
- Overlaps of any shape are reported. Affine-only certification was never a requirement.
- Periodic seam: a curve lying on the seam gives two records, one per side, with the
  same `t_range`. A curve crossing the seam is split exactly at the crossing `t`: one
  record ends on one side, the next starts on the other.
- A point on a seam already comes back as two points (one per side) in both mmcore and
  SMLib.

**What happens now.**

- `bez_csx` writes overlap `u_range`/`v_range` as per-axis min/max (the affine valley
  record and the tolerance certificate). The NURBS adapter reads them as paired ends,
  so decreasing-u overlaps fail the endpoint check (L-junctions: no overlap,
  `complete=False`) or pass silently when the knots are unit.
- The adapter joins tiles only on bit-identical ends and drops a root only when it
  equals an overlap end bit for bit. Ulp noise leaves an overlap in tiles (case18.v1).
- Seams and knot lines on the unit cylinder (`csx.arc_on_cylinder` surface, seam at u=0≡1):

| Curve | SMLib | mmcore before L63 |
|---|---|---|
| Line through the seam point | 2 points, u=0 and u=1 | 2 points, u=0 and u=1 |
| Line on the seam, z −1..1 | 2 records, u=0 and u=1 | 1 record (u=0), `complete=False` |
| Line on the seam, z 2..5 | 2 records | 2 records |
| Arc crossing the seam | 2 records meeting at t=0.5 | 2 records, t 0→0.5006 and 0.4994→1 |
| Full circle starting on the seam | 1 record | 4 records split at the knots, plus 8 points |
| Full circle starting at 90° | 3 records (one duplicated) | 4 records, plus 7 points |
| Generator on the knot line u=0.25 | 1 record | the same record twice |

**What to change.**

- Write paired ends in `bez_csx`.
- In the adapter, join records whose ends meet at a decomposition boundary (curve knot or
  surface patch edge) with matching paired parameters; drop exact duplicates; absorb
  roots that lie on a reported overlap.
- Seam: emit both records along a seam; split a crossing record at the seam `t`.

## L64 — SSX loop-free test on the clipped box

**What happens now.** `_check_loop_free` runs TΨ monotonicity, then Gauss-map separation,
on the whole cell. In `ssx.nearby_circles` the surfaces cross at about 1° next to a ring
of near-contact at 0.244·atol; both tests fail near the ring, cells pile up to the
250k work budget, and the FIFO queue drops arcs (swap: 3 pieces; transpose: 14 pieces).

**Contract.**

- Shrink each cell with `clip_residual_box` first. If the clipped box is empty, skip the
  cell: no tracing, no further checks. Boundary points found there continue in
  neighbouring cells.
- The clip margin is roundoff (`_HULL_MARGIN_K_EPS` times the operand scale). The
  check below shows the skip loses nothing SSX found before.

**Check: does the skip lose anything when the surfaces come within `atol`?** No. The
results are identical before and after L64 on these cases (`bez_ssx`, atol=1e-3) and on
the 31 comparison cases. Each surface sits on a horizontal square, either touching it
(gap 0) or lifted by 0.5·atol:

| Case | Gap 0 | Gap 0.5·atol |
|---|---|---|
| A second horizontal square, half overlapping | 4 overlap branches + 1 region | nothing |
| A vertical square standing on its bottom edge | 1 overlap branch | nothing |
| A parabolic trough, lowest along a line | 1 tangential branch | nothing |
| A paraboloid bowl, lowest at one point | 1 singularity, no point, `complete=False` | nothing |
- Otherwise run TΨ and the Gauss-map test on the whole cell, then on the clipped box.
  A cell passing neither is split.

**Measured before the change** (scratch diagnostics, all 31 SSX comparison cases):

- Same branches, closed flags and points in 29 cases; nearby_circles swap and transpose
  become 2 closed circles.
- Fewer cells in 19 cases (nearby_circles ~249k → 26k–89k, case19 409k → 264k,
  sing.touch_plus_loop 145k → 71k); the other 12 unchanged.
- Cells skipped while holding boundary points: only sing.closed_tangent_loop (48), with
  unchanged output.
- The Gauss-map test certified 20 cells TΨ could not across 24 cases (bez11, bez13,
  pkl7); on the clipped box it certified none.

## As built (2026-10-20)

**L63.**

- `_bez_csx4`: the valley record and the tolerance certificate write paired ends in curve
  order.
  - An end that leaves through a patch edge is polished onto that edge, so tiles meet
    exactly at patch edges.
  - An end at a pole takes the free coordinate of the neighbouring sample on the overlap.
- `_planar_overlap`: paired ends from the exact chart. The UV-enclosure output and its
  adapter handling are deleted.
- `_ncsx4`:
  - Records whose ends match at a curve knot or a surface patch edge are joined.
  - Duplicates are dropped.
  - Records meeting at a seam or a pole share one `t`.
  - Roots lying on record ends are removed before alias rejection.
- `map_overlap`: only `t` may not collapse for CSX; a constant `u` or `v` is legitimate.
- After the change, on the CSX comparison cases:
  - L-junctions 1–3: one overlap each, complete, ends equal to SMLib's.
  - case18.u0: two seam records with constant `u`, apex end included.
  - case18.v1: one record, complete.
  - The unit-cylinder table above matches the contract.
- Unchanged: exact-tier curved overlaps (L42 still reports incomplete), poles in the exact
  tier, tolerance contacts (CSX-3), the constant curve on a collapsed edge (case18.v0).

**L64.**

- `bez_ssx` clips each cell before the loop-free test, skips an empty clip, and reruns
  TΨ and the Gauss-map test on the clipped box. The clip is charged to the work budget
  at the existing clipping rate, under `loop_free_clip`.
- 31 SSX comparison cases: the same branches, closed flags and points everywhere, except
  nearby_circles swap and transpose, which now return 2 closed circles.
- nearby_circles: 41k–130k cells, against 249k before.
- At the clipping rate the charged units rise in some cases while wall time does not.
  sing.closed_tangent_loop now also reports `work_budget` (249,999 cells), with the same
  output.

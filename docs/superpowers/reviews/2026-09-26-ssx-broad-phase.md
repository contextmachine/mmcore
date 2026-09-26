# NURBS SSX broad-phase performance

The broad phase now rejects separated patch pairs before calling `bez_ssx`.
The original patches, parameter domains, pair order, and requested `atol`
are retained for every surviving solve.

## Measurements

Diagnostic runs on the same local Python 3.14 environment at `atol=1e-3`:

| Fixture | Bézier patches | Original calls | Filtered calls | Before | After |
|---|---:|---:|---:|---:|---:|
| `nurbs_nurbs_intersection_2.py` | 108 × 108 | 2,485 | 446 | 117.5 s | 47.3 s |
| `case_19.py` | 492 × 664 | 7,426 | 290 | 651.2 s | 111.4 s |

These are individual diagnostic timings, not general speed guarantees.
The returned branch arrays, points, singularities and overlap regions were
byte-identical to each baseline. Existing completion/reason status was also
unchanged. In particular, this performance change does not resolve case19's
existing `depth_limit` and `unresolved_multiplicity` reports.

For the reported example, 2,039 complete-empty patch solves consumed 69.3 s.
Boundary CSX alone took 109.9 s across the original candidate list. Tree
construction and traversal took only about 11 ms; decomposition took 43 ms.

## What changed

The existing `lbvh.py` tree is already built top-down, by median splitting
along the largest centroid span. Its 2,485 reported pairs exactly match the
overlapping leaf AABBs. A different split strategy with the same leaf boxes
would change traversal cost, but not which pairs reach the narrow phase.

The new sequence keeps that tree and adds:

1. Cached PCA directions for participating patches. Six frame axes and nine
   cross axes supply the usual OBB test directions, but intervals are measured
   over the actual control points rather than the looser fitted boxes.
2. A native GJK direction proposal for remaining hull pairs. The old Boolean
   API is unchanged. Every proposed axis is checked again against the full
   control nets; a Boolean negative alone never deletes a candidate.
3. Up to two levels of lazy homogeneous subdivision when the original hulls
   remain too loose. An original pair is rejected only if all tested child
   combinations are separated. Children are not sent to `bez_ssx`.

The split axis uses control-net curvature, with an edge-length fallback.
That choice controls efficiency only: either partition covers the parent.
For case19, fixed `u` subdivision barely helped; subdividing the more curved
direction reduced the hull-filter survivors from 1,634 to 483, then 290.

| Case19 stage | Remaining original pairs |
|---|---:|
| Padded AABB traversal | 7,426 |
| Full-control support intervals on OBB directions | 2,222 |
| Verified GJK directions | 1,634 |
| Two child-hull levels | 290 |

The entire added filter took about 0.12 s in the measured case19 run.
All 216 pairs producing geometry, and both incomplete pairs, survived.

## Geometric and resource handling

Both hulls retain an `atol` cushion: a projection gap must exceed
`2*atol` plus arithmetic allowance. This is a conservative Euclidean contact
margin; it does not require retaining every false positive admitted by the
old world-axis expanded boxes. The backend still receives the original
`atol`, without adjustment.

Positive rational weights are already required by the NURBS API. Bounds use
Cartesian controls. Child hulls are computed in homogeneous coordinates with
normalized positive weights, and their exclusion margin includes subdivision
roundoff. Failed fitting, unusable directions, or unsafe child arithmetic
retain candidates. A failed split keeps the original node as its enclosure.

The GJK diagnostic exposed seven case19 hull pairs less than `2*atol` apart
despite a Boolean "separated" result. The verified direction filter retains
all seven. Gauss-map separation stays in the narrow phase: it classifies
possible loops, rather than proving the surfaces spatially disjoint.

Projection temporaries are chunked across pairs and controls. Large native
control products skip GJK's quadratic cache; large split work skips refinement.
Refinement has comparison limits per level. Any parent with an untested
descendant survives. Initial control caches remain proportional to the
participating input data.

Aggregate allowances are still computed from the original AABB candidates.
Pruning does not reduce the resources available to surviving geometry.
An older native extension can still use the projection/refinement filters;
rebuilding adds the optional GJK direction stage.

## Reproduction and checks

The benchmark bypasses only the new filters for its baseline, keeping the
same narrow-phase code and dependency versions:

```sh
.venv/bin/python tools/benchmark_ssx_broad_phase.py examples/ssx/nurbs_nurbs_intersection_2.py --baseline --output /tmp/ssx-before.json
.venv/bin/python tools/benchmark_ssx_broad_phase.py examples/ssx/nurbs_nurbs_intersection_2.py --output /tmp/ssx-after.json
```

Use `case_19.py` for the larger fixture. `--no-gjk`, `--no-refinement`, and
`--repeat` isolate stage costs. JSON includes candidate counts, wall/CPU time,
returned geometry counts, status reasons, and a complete geometry digest.

New tests use analytic separation/contact constructions at model-space
tolerance, including nongauge rational weights, transformed parameter charts,
collapsed patches, invalid direction proposals, and interrupted refinement.
They also verify that a curved empty pair is excluded before the full solver.
Existing geometry tests were not loosened. Full validation is recorded in
the pull request.

Algorithm references: [OBBTree](https://ai.stanford.edu/~latombe/cs326/2002/pqp.pdf)
describes the 15-axis box test; [van den Bergen's GJK paper](https://solid.sourceforge.net/jgt98convex.pdf)
distinguishes a separating support-plane lower bound from an unconverged
simplex distance. [PBRT's BVH chapter](https://www.pbr-book.org/4ed/Primitives_and_Intersection_Acceleration/Bounding_Volume_Hierarchies)
provides the hierarchy-construction comparison. Here, measurements favor
tighter leaf enclosures over rebuilding the tree.

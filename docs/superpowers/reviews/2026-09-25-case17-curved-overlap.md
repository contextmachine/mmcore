# Case 17: complete curved overlap boundaries

`examples/ssx/case_17.py` contains two partially overlapping spherical
NURBS patches. At `atol=1e-3`, their common boundary is a closed spherical
quadrilateral: two arcs from each surface. The failing result contained
only the two arcs from the second surface, stitched into an open L, and
no overlap region.

## Causes and corrections

1. CSX rejected the first surface's two coincident edges because their
   signed normal gaps changed sign. The deviations were about `1e-7`, far
   below the requested `1e-3`. Default CAD overlap classification now
   accepts such spans and retains paired contact proposals for SSX to
   refine. Explicit exact-mode behavior remains separate. Normal-gap
   roundoff uses Cartesian geometry, so homogeneous weight scaling does
   not change which contacts survive.
2. Curved-rim discovery found all four edges independently, but stopped
   each endpoint at its own tolerance fringe. Two descriptions of one
   corner were `0.00365` apart; the loop builder discarded the open chain.
   Incident domain edges now resolve one shared paired corner. Parallel
   edges, different parameter faces, and remote contact spans are not
   joined by enlarging a spatial radius.
3. Seventeen fixed rim samples were insufficient for these curved arcs.
   Adaptive subdivision now bounds both lifted parameter chords against
   the reported spatial chord over the whole interval, using positive
   homogeneous Bernstein weights. Shared poles retain their distinct
   parameter representatives. Initial samples, corrected corners, and
   adaptive samples consistently use paired surface midpoints; mixing
   those with owner-only points had rejected valid gaps close to `atol`.
4. A narrow internal CSX overlap fell entirely between the global sample
   locations even though its two boundary contacts were already known.
   Those contacts now seed a candidate interval that undergoes the same
   membership and domain-end checks. The captured cut changed from
   22.54 seconds / 67,104 work units to 0.233 seconds / 1,126 units.

The first full-suite run caught two integration regressions before merge.
The candidate-interval path could mistake following a surface edge for
leaving its domain, joining two lines eight tolerances apart. Domain
pinning must establish an actual exit; touching an edge and returning to
the interior does not qualify either. Both independent negative controls
retain the separate roots.

Changed subdivision also exposed an existing coordinate-dependent apex
classification: a point at an original collapsed cone edge could sit far
from that edge in a very thin child's rescaled parameters, acquiring a
false isolated tangent-point label. Cells now retain their original
surface context. The type check uses the fixed parameter axis and distance
to the collapsed edge's image, while retaining local checks for an interior
collapsed isoline exposed by a split. The cone generator and genuine
isolated touches remain covered by their existing geometry assertions.

Correcting the boundary still left the ordinary one-dimensional search
subdividing a two-dimensional coincident area. A separate geometric
reduction now handles supported rational quadratic spherical octants.
The fitted template is only a proposal: a whole-patch rational difference
bound must place each actual chart within the allocated CAD tolerance of
its ideal chart. A complete analytic hemisphere census is then matched
against the actual assembled boundary, including all corners, cyclic
ordering, monotone parameter intervals, and whole-chord accuracy.
The distance bound also accounts for error amplification at nearly
opposed domain boundaries; an uncertain footprint uses general SSX.

This reduction uses the original surfaces for returned parameters and
geometry. It makes no early empty-result claim. Other charts continue
through general SSX and receive the curved-rim and CSX fixes above. A
surface with identical boundary curves and one matching interior point,
but an interior bulge larger than tolerance, cannot select the spherical
reduction. Small within-tolerance control and weight perturbations can.

Shared spherical poles need geometric treatment as well: different
longitude parameters at a pole are not separate cusp curves or crossing
branches. A Bernstein normal-hull check excludes rank defects outside a
small pole cap. The entire cap must fit within a reported CAD corner, or
be separated from the other actual patch. Only then is the ordinary C1
enumeration unnecessary. The validated simple region boundary also
resolves C3; other reductions retain their existing singularity checks.

The first integrated NURBS run returned one region, four referenced rim
branches, no isolated points or singularities, and `complete=True` in
about 0.31 seconds. The failing version took about 81 seconds on the same
machine. These are single-run diagnostic timings, not a general speed claim.

## Tests

The spherical reference comes from independently intersecting six inward
hemispheres. Case 17 has perimeter approximately `26.16026806` and area
`44.90430351`; both collapsed poles lie outside its common region. Tests
check every boundary component, area and perimeter, both source images,
intermediate chord points, and the absence of leftover branches or points.

Generated cases cover six-sided intersections, swapping and reversing
charts, scaling, a narrow overlap with a shared pole, homogeneous gauge,
separated patches, and lower-dimensional contacts. Additional controls
cover hidden interior bulges, weight-only distortions, ordinary nonspherical
patches with modest work allowances, and interrupted output publication.

The former test requiring a line at signed distance `+/-1e-7` to be rejected
as a CAD overlap at `1e-3` was inappropriate. It now requires the overlap
and retained contact proposals. Above-tolerance crossings and explicit
exact-mode roots remain checked. The new narrow-strip test has an
independent nonlinear parameter correspondence; its interval is only
`0.0005` in curve parameters but spans fifty model tolerances.

Contact proposals stored inside overlap records share the result allowance
with isolated roots. A denied CAD-classification step reports the work
stop and retains already found geometry. Completed paired rims also survive
work denial during spherical-region assembly, without invalid region indices.

## Historical comparison

Using the exact supplied rounded fixture and the same native extensions,
the open L also reproduced at `76735f9`; the pre-PR-48 `552d972` version
returned no geometry. The working version of this particular example has
therefore not been identified. The corner and normal-side assumptions
predate the latest merge; this investigation does not attribute the whole
failure to PR #48.

Final validation is recorded in the repair pull request. Focused runs and
the timing above do not replace the full-suite and GitHub checks.

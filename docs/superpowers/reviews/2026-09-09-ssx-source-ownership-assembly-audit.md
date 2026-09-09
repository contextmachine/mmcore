# SSX source ownership and assembly audit

The destructive assembly predicates now distinguish actual source ownership
from membership in an approximation polyline. A geometric comparison can
search for a candidate duplicate; it cannot identify a source component.

## Concrete defects and changes

- **A separate isolated root on a valid parabola chord.** For
  `z=(t-(s-.5)^2)*((s-.5)^2+(t-a^2)^2)`, with `a=1/32`, the plane
  intersection contains a parabola and the separate point `(.5,a^2)`.
  That point lies exactly on the parabola's endpoint chord, whose true
  approximation error is `a^2 < .001`. Generic point/chord cleanup deleted
  a real source component. Production cleanup now requires shared source
  root registration; unknown incidence is retained.
- **Two lens arcs with the same endpoints and approximation chord.**
  `F=(t-x^2)*(t+x^2-2a^2)`, clipped to `-a <= x <= a`, has two distinct
  source arcs. Both can be represented within `.001` by the same chord.
  Endpoint identity and even exact polyline equality cannot delete either
  arc. Independent fragments remain in the graph and form the lens cycle.
- **A midpoint residual estimate deleting a valid arc.** For
  `h=512*g*(2a^2-g)`, `g=v-(u-.5)^2`, the lower parabola is within `a^2`
  of its chord, and both source evaluations along the public parameter
  chord are within `1/2048` of the reported XYZ chord. At its midpoint,
  the graph gradient vanishes. Dividing residual by a clamped normal angle
  wrongly rejects the whole real arc. The destructive valley filter is
  removed.
- **A declined source identity overridden during later stitching.**
  A secondary join pass merged equal displayed endpoint tuples even after
  the source root matcher declined their identity. That pass is removed;
  shared or proved root objects govern graph stitching.
- **A positive seam gap rounded to zero.** On a closed `[0,1]` axis,
  `abs(1-2^-60)` rounds to `1`, and subtracting the domain span gives zero.
  Destructive equality now compares coordinates directly or verifies the
  exact opposite domain endpoints. Distance arithmetic is only a broadphase.
- **A boundary root assigned to the wrong child.** A displayed parameter
  can lie on the other side of a cut from its source root. Both cut-grid
  and inherited-event distribution now use source enclosure overlap with
  every closed child. Unknown or unpaid source evidence retains the event
  in every possible child. Exact source intervals preserve sub-ULP gaps.
- **Distinct singular strata coalesced by proximity.** Nearby tangent
  points and self-intersection mates now require exact paired preimages;
  cusp sample clouds do not identify whole source strata. Cusp records
  on different source surfaces retain both ownership tags.
- **Rounded source transport used to exclude a genuine zero.** Main
  residual exclusions now restrict the original world residual and carry
  its absolute coefficient error. A fixed aligned projection is merely a
  proposal; outward interval dot products must prove separation. Generic
  face CSX also searches the original residual enclosure, bypassing
  proposal-geometry AABBs, boundary shortcuts and overlap classification.
- **Unsupported rational control hulls.** Bézier and NURBS SSX now reject
  nonfinite coordinates and zero, negative or nonfinite weights before
  normalization, decomposition and hull rejection. The supported rational
  input contract requires finite, strictly positive weights.
- **Incomplete faces poisoning unrelated descendants.** Missing boundary
  events now carry closed source-face boxes. Children retain every
  intersecting obligation and recover a complete census only when all
  localized obligations are disjoint. Missing localization remains unknown
  on every descendant; a denied partition scan also retains unknown status.
- **Fallback cuts bypassing progress checks.** Budget fallback midpoints
  and the adaptive second cut are checked again for strict interiority.
  An unsplittable cell retains registered points and an explicit parameter
  representation obligation. The default depth no longer stops productive
  refinement at 13; explicit caller depth ceilings still return partial.
- **Exact source roots sharing every displayed coordinate.** The scaled
  circle source-cut capture contains different algebraic roots at the
  identical displayed tuple `(.25,.75,.375,.625)`. Registered-point dedup
  now requires source identity. A three-vertex open source arc cannot be
  marked closed by equal displayed endpoints after its source endpoints
  were proved distinct. NURBS collection retains these owners so within a
  pair its later floating comparisons cannot reverse the source decision.
- **The active owner disappearing from partial results.** Work denial
  after a queue pop formerly omitted that cell from the unresolved-domain
  dump. Every main-loop stop before a full child partition is queued now
  returns the active owner to the frontier. Denied probe-child allocation
  does the same. This also prevents staged ancestor traces from being
  classified as replaced merely because the missing active owner had no
  diagnostic box.
- **Fresh root solves replacing existing ownership.** Equivalent source
  faces now transport already established source root boxes to the exact
  common pin. They do not run independent new solves around the displayed
  coordinates and silently adopt whichever root those proposals find.
  Original-face membership and common-pin containment are required; the
  union still must prove injectivity. New scalar-certificate enrichment
  of an owned root requires a refinement contained in its old unique box.

## Retained positive ownership proofs

Exact affine source tiers attach Fraction-valued parameter paths proved to
lie entirely in the source zero set. NURBS assembly maps those paths through
exact affine span maps. `_ssx_arc_ownership.source_path_covered` proves full
segment coverage by exact clipping and interval union, with no gap epsilon.
It permits legitimate exact tile-rim dissolution and represented-rim
absorption. Missing certificates and budget denial retain the geometry.

The source-root child distribution and source-path predicates received an
independent arithmetic/provenance read from the CSX/CCX reviewer, with no
remaining concrete blocker reported.

## Verification snapshot

- Root identity, owned-root transport, exact affine constraints, source
  point validation and active-owner denial: **50 passed in 0.67 s**.
  The transport adversary has two different exact polynomial face roots,
  overlapping valid ownership boxes, and fresh proposals converging to
  the same root; the former code incorrectly merged them. The revised
  code retains both and preserves positive same-root joins.
- Full curved-circle world-transform coverage and declined known-port
  identity: **7 passed in 5.42 s** after the owned-box transport change.
  A separate trace-owner fix prevents a source-proved unregistered exit
  from replacing either of a cell's two already known source ports; an
  unidentified endpoint leaves that owner partial instead of publishing
  two reverse traversals with a complete claim.
- Source-point display validation, active-owner budget denial, localized
  obligations and child ownership: **20 passed in 0.57 s**. The two
  denial-after-pop regressions failed before the owner-retention fix.
  The source-point controls distinguish exact root ownership from display
  accuracy, reject incorrect/NaN XYZ, preserve the face on denied work,
  and prevent uncertified candidates from using accuracy as existence.
- Exact affine coordinate constraints, including the new common-world
  projection nullspaces: **24 passed in 0.53 s** after an independent
  arithmetic review. These are focused checks while other source changes
  continue; the earlier hash-verified full adapter snapshot is separate.
- Latest full `test_nssx5.py`, source input, localized obligations and
  captured source-alias assembly gate: **89 passed in 23.55 s**. All source
  and selected test hashes were unchanged during the run. Log and manifest:
  `/tmp/mmcore-ssx-audit/nssx-source-identity-final.txt` and
  `/tmp/mmcore-ssx-audit/nssx-source-identity-final-hashes.json`.
- Captured source aliases, exact root separation, point and branch source
  incidence: **19 passed in 0.69 s**. The formerly failing c5 scaled-loop
  invariance control also passed after the stricter closure decision.
- Latest full `test_nssx5.py` plus source input and localized boundary
  obligations: **86 passed in 23.57 s**. Log:
  `/tmp/mmcore-ssx-audit/nssx-source-obligations-final.txt`.
- Latest main search, default-depth, child ownership, localized obligations
  and exact seam singleton gate: **38 passed in 5.88 s**.
- Full `test_nssx5.py` plus new source singularity, child ownership,
  exact source path, branch incidence and point incidence regressions:
  **88 passed in 24.57 s**. Log:
  `/tmp/mmcore-ssx-audit/nssx-source-owned-graph-only.txt`.
- Focused assembly, registered-point, search and certified seam gate:
  **62 passed in 3.42 s**.
- Original-source exclusions, face transport, raw source CSX, child census
  and regular empty separators: **35 passed in 0.65 s** after the source
  transport changes. This is a focused gate, not a replacement for the
  earlier full adapter integration snapshot.
- Source input validation: **16 passed in 0.57 s**, including both input
  sides, zero-work calls and pre-decomposition rejection.
- `git diff --check` passed.

The explicit depth regression now uses actual CSX16 and SSX2 ceilings. It
retains its greater-than-half-loop coverage assertion, verifies continued
census work beyond the eight outer faces, and reports only `depth_limit`.
The former CSX20-only fixture became complete after source ownership proofs
improved and therefore no longer exercised a depth refusal.

A separate actual case11 control now compares explicit depth13 against the
default under the same 60,000-unit allowance: the explicit cap stops with
`depth_limit` at 21,340 units; the default uses additional available work
and returns one complete closed loop of independently checked length.
The seam-straddling paraboloid now requires complete exact singleton output
because PSD/RREF elimination proves its entire source zero set. The generic
witness-floor test explicitly requests the generic path and checks that its
mocked witness ran; it does not override the new exact singleton proof.

## Certification limits distinguished from fixed branch loss

The regular zero/one/two/multi-port arguments require an exhaustive source
boundary census. Strict source cofactors supply regularity and a monotone
coordinate. The new child supercover preserves existing census coverage.
Exact Sturm refinement against owner faces is root isolation, not assumed
clipping of an existence enclosure.

The final completion-path read found no unconditional cell retirement
after unsupported singular tracing: deflation retains the ordinary
complement, and complete regular cells require original-source strict
cofactors plus exhaustive boundary ownership. The new unsupported convex
rational-bilinear region owner returns a whole-domain 2D obligation; it
does not infer a paired region representation from a sampled rim.
Projected coordinate constraints use exact uniform-weight Bernstein
coefficient nullspaces and only restrict necessary zero-set domains.
Equivalent exact face keys select a common source slice; root identity
still requires source existence and union uniqueness.

Generic numerical marching still has an approximation-error limitation:
after unique source-arc ownership is proved, its long-march fallback checks
rounded-cell samples and endpoint identity rather than proving every output
chord's global source error. C1/C3 branch links also remain geometric
associations to approximation segments.

The generic face census now transports the original residual net and
coefficient-error envelope through exclusion, uniqueness, ownership and
Krawczyk existence tests. A supplied exact original-source identity can
prove a closed-face root at an exactly mapped rational parameter; a generic
boundary algebraic root without such an identity or strict interval
inclusion remains unresolved. Returned proposal geometry still requires
separate source representation validation. No claim of complete formal
certification for every algebraic surface pair follows from this audit.

## Final frozen review

The final bounded read found no new production correctness blocker in the
reviewed source ownership, regular-cell completion, assembly, and seedless
fiber-fallback paths. The opposed supporting-halfspace tier was also read:
one-sign exact heights reduce all intersections to the enumerated boundary
strata, and exact paired segment intersection preserves their correspondence.
This is a bounded review verdict, not a universal certification theorem.

Current production files match all 56 hashes in the final integrated SSX
manifest and all 84 hashes in the final NURBS/toroidal manifest. No additional
runtime suite was launched during the quiet final benchmark. Reported final
gates are 563 SSX, 431 raw curve-search, 136 NURBS/fiber, 25 toroidal, and
63 analytic cases passing; these overlapping suites are not summed.

The final reliability report and singular disposition accurately retain
**91 passed, 21 failed, and 5 timed out** in the legacy diagnostic. Exact
source-contact counterexamples invalidate only the asserted stationary
contacts; they do not excuse ordinary branch-count failures or the five
unresolved runtimes. Unsupported paired-UV regions remain actual failures.
The report explicitly distinguishes formal source ownership from generic
marcher chord accuracy, geometric C1/C3 links, floating NURBS decomposition,
and cross-patch algebraic identity limitations.

The consumer contract now explains that empty partial output is not a
no-intersection proof, unresolved boxes are conservative, candidates are
unproved, and output caps may truncate diagnostics themselves. Retained
NURBS local solutions keep local parameters and an explicit affine map.
Together with typed reasons and capped work counters, this is a useful
partial-result API without overstating what `complete=True` establishes.

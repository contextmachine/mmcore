# SSX branch preservation and intersection search audit

> Historical report of the withdrawn exact-topology implementation. Its
> mandatory algebraic acceptance rules regressed working CAD intersections.
> SSX has returned to numerical acceptance at the caller's geometric tolerance;
> rational source certificates and `unresolved_regions` are not its public
> result contract. The passing subsets recorded below did not establish a
> passing integration: the separate singular suite had known failures.

Work branch: `codex/ssx-completeness`, based on `76735f9`.

The main checkout and its unrelated `mmcore/numeric/bern.py` edit are preserved.
This report records structural defects and their remedies. Regression counts
and final benchmark evidence are recorded after integration below.

## Scope and meaning of the result

The investigation covers Bézier SSX, its NURBS adapter, the CSX searches used
on patch boundaries and subdivision faces, and CCX root handling. In this
checkout the nested call chain is **SSX → CSX → CCX**; the CCX implementation does
not call CSX. Both curve solvers share several numerical mechanisms, so the
CCX audit remains relevant.

An intersection is the zero set of the supplied surface/curve representation.
`atol` controls numerical approximation; it does not turn two distinct
parameter roots into one root, turn near-unit weights into unit weights, or
turn a small nonzero residual into an exact overlap. Public CSX retains its
existing optional tolerance tier. SSX explicitly requests exact topology
search from CSX.

Finite budgets remain enforced. An incomplete search returns its useful
geometry and typed unresolved regions/reasons. Finding a component does not
authorize discarding its unsearched complement.

## Confirmed defects and structural changes

| Mechanism | Counterexample or violated invariant | Change |
|---|---|---|
| Approximate rationality | Weights `1, 1+8e-6, 1` split a polynomial double root into two transverse roots; `allclose(weights,1)` erased the split. | Exact representation predicates in the adapters and knot operations. |
| CSX shared-face exclusion | A folded patch has two transverse roots on a subdivision face; non-strict derivative signs excluded both closed children. | Closed-domain root accounting; exact mode uses unsquared vector separation. |
| Squared-distance cancellation | A restricted squared Bernstein net has positive coefficients at an exact zero. | Remove squared-net exclusion from exact CSX/SSX; carry source-scale error through unsquared residual bounds. |
| Unresolved surface parameters | Small curve-parameter span was treated as a fully isolated 3-variable root box. | Resolve the entire product parameter domain; certify uniqueness before removing a root neighborhood. |
| One preimage consumes a slab | A curve on `S(u,v)=((u-.5)^2,v,0)` has two overlap correspondences. | An affine overlap witness requires UV injectivity before discharging other preimages; otherwise retain uncertainty. |
| A boundary root consumes another root's strip | `A(u)=(u,u²,0)` and `B(v)=(.5,.25+v(v-.75),0)` intersect at `(.5,0)` and `(.5,.75)`. Removing the first root's entire `u` strip lost the second exact root and returned an incorrect tolerance minimum. | Remove only a proved product-domain root neighborhood, in both exact and tolerance modes. |
| Repeated plane roots | General 3D search creates large clouds around high-order tangencies. | Exact rational polynomial reduction, square-free decomposition, and real-root isolation for the supported planar chart class. |
| Uniqueness mistaken for existence | A box can contain at most one root while containing none; tiny positive gaps were accepted as exact intersections. | Require exact source identity or an inclusion certificate before reporting an exact generic root. |
| Rounded subdivision boundaries | Intersecting a cut with already split opposite patches changes the boundary polynomials through rounding and repeats work. | Compute each cut census against the unsplit opposite patch, then assign events to the covering children. |
| Rounded geometry becomes the source census | Checking original-source identity only for returned CSX roots cannot prove that a rounded child polynomial hid no other roots. | Transport the original homogeneous residual and its absolute coefficient-error envelope to each face. Generic CSX uses a full product search whose exclusions, existence, uniqueness, and ownership tests all use that source envelope. |
| Child ownership uses a displayed coordinate | A proved source root enclosure straddles a split while its displayed coordinate lies on only one side. | Copy the root to every closed child intersecting its source enclosure; unknown or unpaid ownership stays with every possible child. |
| Global parameter aliasing | In the knot domain `[1e16,1e16+2]`, different local roots round to one global tuple; valid local XYZ no longer agrees with the public parameters. | Validate global representation before assembly, preserve exact affine provenance, and retain unrepresentable local solutions in a typed partial payload. |
| Budget exit erases completed work | A denied first postprocess unit discarded ordinary branches that had already been assembled. | Preserve completed geometry and retain the unfinished ownership obligation. |
| Incorrect cutout | A two-interval split used index `-1` for the removed interval and retained it. | Remove the identified interval, eliminating repeated work. |
| Approximate restriction identity | Very short nonzero trims were silently treated as full intervals. | Honor every representable nonempty restriction. |
| Tangency consumes a regular complement | An analytic tangent line coexists with a disjoint regular circle. | Continue ordinary discovery after processing the tangential component. |
| Failed tracing claims completion | A cell with a seed but failed continuation became an isolated point and was discharged. | Subdivide or preserve an unresolved owner cell. |
| Displaced tracing crosses a cell boundary | Toroidal tracing jumped over an interval outside its owning cell and prepended a false chord. | Check the strict regular corner cone before displaced recovery. |
| Geometric proximity becomes topology | Close distinct sheets, points, or branches were deleted by XYZ or parameter tolerance. Even exactly identical lifted approximation chords can represent distinct source arcs. | Require actual source-root, source-arc, or exact rim provenance for destructive assembly; preserve unproved alternatives. |
| Periodic seam chord | Joining preimages at domain endpoints created an artificial parameter segment through the middle of the domain. | Preserve seam endpoint pairs and exclude wrap-only segments from containment. |
| False NURBS closure | Equal end control rows on an unclamped surface need not be its actual boundary curves. | The control-row seam shortcut requires clamped end knots and exact equality. |
| Region absorption across a gap | A region ending at `.5` absorbed a separate line or tile beginning at `.5+2**-11`. | Exact seam identity and continuous represented-rim evidence replace tolerance bands. |
| Image identity becomes exhaustive correspondence | An identical self-intersecting chart has a diagonal region and additional off-diagonal intersection curves. | Exact image identity and global chart injectivity are separate obligations. Projected UV footprints cannot retire pair-level uncertainty. |
| Probe queue explosion | Uniform 4D splitting spent work on free extrusion axes of a 1D singular constraint. | Split according to Bernstein derivative variation of the unresolved equations. |
| Corrector output escapes its owner | A toroidal cut produced 128 copies of a root outside the active CSX cell, exhausted its result allowance, and lost an arc. | Exact CSX publishes only source-proved roots owned by the current closed cell; other cells remain searchable. |
| Rounded minor sign becomes loop absence | An exactly invertible, ill-conditioned world transformation makes the computed determinant sign disagree with the original-source determinant range. | Bound source cofactors with outward arithmetic and operand errors; polynomial coefficient products retain dependencies. |
| Linked parameters searched independently | Graph/plane pairs generated large Cartesian grids of impossible parameter combinations. | Exact affine source-coordinate relations contract necessary domains, preserve the old exhaustive boundary census, and identify equivalent face queries. |
| Modeling tolerance hides an exact root gap | Two distinct certified roots near a polynomial extremum were merged into one padded subdivision band. | Certified coordinate bands retain their actual enclosures; representable dyadic gaps remain available for subdivision. |
| Loop absence mistaken for complete ownership | A monotone cell can contain several disjoint arcs, or have an unfinished boundary census. | Require source-proved boundary ownership before discharging the cell; unresolved pairings stay in subdivision. |
| Rounded root becomes a false cell corner | A root interval touches a face at an endpoint that is not a zero of its original exact polynomial. | Use exact closed-owner membership and open Sturm endpoint information, including exact linked parameter coordinates. |
| Different face records become different roots | One exact source corner has several scalar cut certificates with different varying axes. | Recognize identical exact singleton source boxes and request source-face evidence before declining cross-face identity. |
| Guided subdivision leaves the same interior unresolved | Cuts between nearby boundary ports approach a circle extremum while retaining an unrelated stationary point between two circles. | Before source regularity is proved, bisect longest chart axes to advance discovery over the whole owner. |
| Second cut family undoes balanced subdivision | A later rewrite replaced the second chart midpoint by a narrow root gap even in a nonregular cell. | Root-guided recutting requires proved regularity; both unresolved chart directions keep balanced progress. |
| Rejected proposals become false unresolved regions | Unproved Newton proposals were converted into large reservations even after their entire domains had been searched. | Keep proposals as seeds. Only source-certified roots become output; the actual subdivision obligations describe the unresolved complement. |
| Assembly bypasses failed identity | A secondary endpoint scan joined equal rounded tuples after the source matcher had declined their identity. | Assemble from the source registration graph and preserve unknown free endpoints. |
| Rounded normalized residual erases proved roots | A translated, scaled rational strip had two exact source boundary roots, but both failed a roundoff-sized test on normalized proposal controls. | Preserve source existence and identity; bound the displayed point against both original surfaces at the requested approximation accuracy. |
| Distinct source events share a floating tuple | Two true circle ports have disjoint exact isolating intervals but identical displayed parameters. | Order ports by strict original-source intervals; refine overlapping intervals with exact Sturm counts. Registered identities also control point deduplication and closure. |
| A partial face poisons unrelated descendants | A positive-dimensional cut left every child incomplete, including children disjoint from that face. | Carry localized closed source-face obligations through the covering partition; only proved disjoint children recover a complete census. |
| Fixed depth stops ordinary discovery | The default depth 13 stopped a regular closed component with most of its work allowance unused. | Make the default depth uncapped while preserving explicit depth caps and the shared work allowance. Reject subdivisions that cannot make representable progress. |
| Unresolved singular frontier starves owned arcs | A tangent line and a disjoint circle compete for the same work budget; the regular circle remained fragmented. | Finish children of a source-proved regular owner before expanding the unresolved singular frontier, retaining every queued child. |
| World-axis dependence hides affine relations | Two curved graphs intersect in a closed saddle-embedded circle; an exact invertible world mixing consumed 250,000 work units with no branch. | Exact nullspaces of original Bernstein coefficient identities recover common projected affine coordinates. Equivalent source-face keys support subsequent root existence and uniqueness checks. |
| An unidentified exit duplicates an owned arc | A two-port regular cell traced both directions after an independently proved exit failed to match the known partner. | Require source identity with the owned partner before discharging the cell; a numerical exit remains unresolved. |
| A popped owner disappears on budget denial | Denial during cut census or child allocation left the current owner outside the final queue dump. | Retain the active owner in the unresolved frontier before every such stop. |
| Sampled rim coverage becomes an exhaustion proof | Sampled common-plane rims and projected box centers cannot exclude another trimmed component. | Sampled region reconstruction preserves existing branches and remains partial; exact source region certificates own complete region claims. |
| Unsupported two-dimensional sets consume the search | Strictly convex nonaffine rational bilinear overlaps spent the entire budget subdividing a known continuum. | Prove the complete common image is a positive-area convex polygon and return its exact vertices with a typed unresolved paired-region representation. |
| A seed helper blocks an empty-face proof | A collapsed or nearly collapsed edge has no target witness, but its helper short-circuits to an unresolved face and prevents ordinary source CSX from excluding it. | A helper without a seed falls through to the full original-source census; witness absence is not a conclusion about that face. |
| Rounded singular candidate becomes confirmed contact | Several intended multiple-root fixtures have no exact stationary zero for their supplied binary coefficients. | Validate new candidates against original-source equations and minors; preserve unproved locations as typed diagnostics. |
| Singular witness mixes local and global parameters | A singleton from a restricted patch was published directly in the original domain. | Preserve the registered global source point and verify exact candidate provenance. |
| Two-vertex seam uses a sample-index midpoint | Oppositely oriented representations select opposite endpoints as their midpoint, blocking a valid region join. | Validate the entire paired seam polyline and its exact endpoint correspondence. |

## What is established by the new certificates

The polynomial chart injectivity check in `_ssx5_overlap.py` chooses a constant
2×3 projection numerically and verifies it with exact `Fraction` arithmetic on
the input binary coefficients. The condition

```
max_row_sum(abs(I - R DS)) < 1
```

holds over the entire Bernstein parameter square. Integrating along the
segment between two parameter points proves that their projected images
cannot coincide. This supports arbitrary graph heights, not only planes.
Unsupported nonuniform-weight charts conservatively fail this sufficient
test; failure does not mean the chart is noninjective.

The shared interval-Jacobian root-box check establishes **at most one root**.
A small polished residual alone does not establish existence. The NURBS seam
join additionally uses an inclusion check and exact shared-boundary
provenance. These distinctions matter: uniqueness, existence, approximation
quality, and exhaustive coverage are different claims.

The lifted-polyline helper checks continuous coverage of the represented
segments rather than testing only vertices. It compares the actual returned
polylines. Even exact correspondence of their parameter chords does not prove
that they represent the same source component. An exact polynomial lens has
two distinct arcs with common endpoints, each validly approximated by the same
chord. Another exact example has an isolated point on the approximation chord
of a separate parabola. These counterexamples require source incidence before
destructive deduplication; ordinary approximate polyline containment is no
longer sufficient.

For a regular cell, a strict original-source cofactor gives a global monotone
parameter and excludes closed components. An exhaustive two-event boundary
census, source existence and ownership, and at least one proved inward germ
identify one actual arc: its other endpoint must be the other event. Cofactor
ratios enclose that arc before approximation is accepted. More than two events
are not paired by proximity. When all event germs are proved, a zero prefix
count between ordered event bands certifies an empty separating face.

Both the exact scalar root enclosure and any independently proved source
enclosure describe the same root; their intersection can tighten ownership.
Clipping an existence box to a cell alone is not such a proof. A closed Sturm
interval can include a non-root endpoint, so exact polynomial evaluation also
distinguishes an outside borrowed event or an inactive face from an actual
source root on a shared closed boundary.

The exact ruling/plane tier handles a uniform-weight surface that is exactly
affine in one parameter and whose plane-distance polynomial is independent
of that parameter. The target is a convex affine bilinear plane. Every
component then lies on a ruling indexed by a root of one scalar polynomial;
exact Sturm counts and halfspace clipping account for all such components.
The present tier requires isolation to expose singleton root intervals:
dyadic roots, or an arbitrary rational root when the square-free polynomial
is linear. Other algebraic roots conservatively use the general solver.
An unsupported tier does not infer that the intersection is empty.

The quadratic-constraint tier handles polynomial residuals of total degree
at most two. Exact row reduction eliminates linear equalities. A positive or
negative semidefinite quadratic at its zero extremum is replaced by its
equivalent linear gradient equations. A rationally factored indefinite
quadratic is handled as the union of its exact affine factors; the factor
product is checked against the original polynomial. Each factor triggers
fresh linear elimination and is checked against every source equation.
Supported resulting affine parameter lines are clipped against all four
domains, and their actual curved images are bounded. Exact shared endpoints
split the components into their incident arms. A constant-image parameter
line is retained as a fiber, with source-normal and endpoint-incidence
evidence. This handles both a crossing saddle and a cone generator with its
collapsed apex fiber. Irrational factors, unsupported dimensions, and
remaining nonlinear equalities stay in the general search.
When source-equivalent elimination leaves a zero-dimensional singleton,
exact domain membership and original surface normals distinguish an isolated
regular contact from a degenerate surface point; the displayed parameters
and world point still require a separate approximation bound.

Original homogeneous boundary controls also distinguish an exact collapsed
fiber from a nearly collapsed edge. Exact component-polynomial GCD and Sturm
isolation find supported target preimages, preserving distinct algebraic
roots even when their floating UV tuples coincide. Such fibers remain
positive-dimensional boundary obligations. Near-collapse proposals can seed
useful partial branches, whose displayed segments must satisfy the original
source approximation bound; they do not acquire exact fiber certificates.

The opposed-boundary tier proves that a plane weakly separates both original
control hulls. Common points must lie on the zero-height strata of both
charts. For the supported affine edges and corners, exact paired-stratum
intersection then accounts for the full correspondence, including shared
endpoints. The separate affine-plane-region tier accounts for complete
two-dimensional coincidence regions and their exact clipped rims.

Necessary affine parameter contraction is a zero-set operation, not a
residual bound away from the intersection. Cofactor consumers explicitly
use conditional bounds on that zero set. Geometric cell contraction is used
only for positive-width boxes; an exact newly introduced extreme is attained
on an old boundary in the same affine component. Outward rounding only adds
root-free boundary faces. Existing source root objects and their enclosures
are preserved, and unrepresentable or degenerate contractions are declined.

For uniform-weight polynomial charts, these affine constraints also use
exact common world projections. A three-column nullspace annihilates every
Bernstein coefficient residual after subtracting the proposed affine
coordinate on each chart. Exact echelon elimination establishes the identity
without a floating rank threshold. Shared projected coordinates supply
necessary parameter relations under changes of world basis. Equal exact
face keys select a common source slice; existence and union injectivity are
still required to identify two root events. Already owned root enclosures
are transported to that slice, rather than replaced by fresh solves around
their displayed coordinates.

When affine contraction alone cannot establish regularity, a necessary
interval Gauss-Seidel contraction uses the original residual and its
derivative envelopes. A numerical row preconditioner is only a proposal:
projected Bernstein coefficient intervals are formed before their hulls, and
every interval operation is outward rounded. A strict nonzero pivot gives a
necessary coordinate interval for every zero. Its intersection can contract
the domain or prove it empty; it never establishes existence or uniqueness.
One full sweep is cached per owner, with subsequent subdivisions providing
further refinement.

The generic face-census route bypasses AABBs, overlap/fiber promotions,
boundary CCX deletion, and Phase 1 range subtraction derived from rounded
proposal operands. Its source coefficient envelope supports strict
Krawczyk existence in an interior box. For closed-boundary candidates, exact
homogeneous evaluation uses the exact affine image of local floating
parameters, rather than a rounded global tuple. Unsupported boundary or
positive-dimensional root sets remain explicit obligations. Callback work
is prepaid inside the same raw CSX allowance.

For straight tangent paths and short regular arcs, a small physical image
on one surface is insufficient to validate the lifted parameter chord.
The exact path helper restricts both original surface nets to the proposed
affine parameter segment, converts the tensor diagonal to one-dimensional
Bernstein coefficients, and bounds the homogeneous residual of the whole
path against the reported XYZ chord. Source existence, component ownership,
and representation accuracy remain separate obligations.

For a regular arc with a strict monotone parameter, source cofactor ratios
bound its parameter velocity. Exact source derivative intervals then bound
world velocity. The derivative-range secant inequality bounds coordinate
error by one quarter of parameter span times velocity-range width, plus the
actual rounded endpoint error. This supports useful long chords without
relying on one midpoint sample or requiring the entire surrounding surface
patch to fit inside the geometric tolerance.

## Independent coverage evidence

`examples/ssx/ssx5_analytic_audit.py` constructs graph/plane intersections whose
intended entire zero sets are prescribed products of line and circle factors.
The coverage code is independent of SSX tracing. It checks all components,
whole output segments, total length, circle angular travel, branch ownership,
and closure. Deliberately missing arcs, duplicate traversals, and chords
through the interior of a circle are rejected by tests of the audit itself.

The audit includes nine geometric families and seven representations:
surface order swap, parameter reversals/transposition, and exact knot
insertion. It uses a common tolerance and the normal solver budgets, with an
external per-run watchdog. Rational tests separately use the analytically
derived roots of the near-unit-weight family, including both adapters and
surface orderings.

The toroidal test oracles were also audited. Counting vertex indices between
nearby samples falsely classified dense forward sampling as retracing; the
replacement uses accumulated travel and direction. Averaging the two surface
images at an averaged parameter tuple is not an intersection point and
overestimated sagitta. Its replacement independently solves the original
NURBS intersection together with a chord-normal plane, with residual,
parameter-domain, and local-branch checks. Synthetic positive and negative
controls validate both oracles. No production sampling was removed to make
those tests pass.

Baseline evidence is retained in
`../issues/2026-09-09-ssx5-analytic-coverage-baseline.json`.

## Limit of the reliability claim

These changes remove reproducible invalid exclusion and ownership decisions;
they do not constitute a universal proof of arbitrary NURBS intersection
topology. Such a proof additionally requires validated coefficient
construction throughout the pipeline, existence and continuation enclosures
for every traced arc, and exhaustive singular-set and overlap-trim topology.
The generic numerical tracer and sampled overlap-rim reconstruction are not
replaced by a complete exact algebraic-geometry engine here.

Source ownership of a regular arc does not by itself certify every chord
produced by the fallback numerical marcher. The new exact-path and
source-velocity secant routes do provide continuous bounds; the fallback
still uses numerical continuation and local chord checks. Some C1/C3
`branch_links` remain geometric associations rather than proved source
incidence. The NURBS adapter also relies on floating knot decomposition;
exact preservation of every nonrepresentable decomposed coefficient is not
a theorem established by this change. Cross-patch assembly retains exact
displayed global preimage checks and existing shared-seam certificates;
these are not a general algebraic root-identity proof across arbitrary
decomposed NURBS patches. Near-collapse numerical seeds and some singular
classification helpers are proposals, not exact fiber-incidence proofs.

Accordingly, passing regression and analytic coverage tests must not be
described as proving that every possible branch can never be missed. The
useful stronger contract is explicit accounting: implemented proof refusals
and budget stops retain visible obligations. Declared work units and callback
costs are capped; this is not a cap on every scalar operation or wall time.
Known remaining limitations and final verification results belong in
the final integration record below.

## Final integration record

The final regular gates used unchanged production sources throughout each
run. Counts below are suite counts, with some shared tests; they are not
summed into a purported unique-test total.

| Gate | Result | Reproducible evidence |
|---|---|---|
| CCX/CSX and shared curve search | 431 passed in 44.90 s | [Command, hashes, and execution history](2026-09-09-ccx-csx-verification.json) |
| Integrated SSX, invariance, source ownership, and budgets | 563 passed in 121.69 s | [Command and hashes](2026-09-09-ssx-integrated-final.json), [pytest log](2026-09-09-ssx-integrated-final.txt) |
| NURBS adapter, source fibers, identity, and limits | 136 passed in 23.72 s | [Commands and hashes](2026-09-09-nssx-torus-final.json) |
| Toroidal closure, length, traversal, sagitta, and order | 25 passed in 116.17 s | [Commands and hashes](2026-09-09-nssx-torus-final.json) |
| Independent analytic whole-component matrix | 63/63 passed; no timeout or silent failure | [All cases and one source manifest](2026-09-09-analytic-matrix-release.json) |
| Python compatibility check | 38 changed production files parse with Python 3.9 grammar | [File hashes and qualification](2026-09-09-python39-final-grammar.json) |

The runtime was Python 3.14.6 with the existing local native extensions.
Python 3.9 runtime compatibility was not exercised. Earlier failing audit
snapshots remain available; a later targeted pass does not rewrite their
outcomes. In particular, the boundary-fiber integration briefly regressed
the torus cases, and both the failed run and the corrected complete run are
retained.

The rational-weight counterexample is available as [PNG](assets/ssx-2026-09-09/rational-branch-preservation.png),
[SVG](assets/ssx-2026-09-09/rational-branch-preservation.svg), and
[numerical output](assets/ssx-2026-09-09/rational-branch-preservation.json).

The full legacy singular diagnostic is **91 passed, 21 failed, and 5 timed
out**, with a 30-second subprocess watchdog per test and one unchanged
source manifest. [Every result](2026-09-09-singular-final-frozen.json) and the
[assertion-level disposition](2026-09-09-singular-release-disposition.json)
are retained. These are actual nonpasses, not expected-failure or skip marks.
The remaining issues are:

- Fifteen failures involve intended exact contacts that do not exist as
  stationary zeros of the supplied binary coefficients. Exact polynomial
  source oracles establish that narrower statement. Two currently fail on
  branch counts; absence of the intended contact does **not** settle their
  nearby ordinary zero-set topology or excuse those geometry failures.
- Case 13 still lacks an original-source singular existence or absence
  certificate. It returns explicit partial status without promoting its
  numerical witness into a certified tangent point.
- Five nonaffine rational bilinear overlap cases prove one two-dimensional
  owner and its exact physical image polygon, but do not yet return the
  required paired-UV region representation. They now terminate promptly
  instead of exhausting the watchdog, while their geometry assertions fail.
- Three nearly singular mixed-contact searches and two nonlinear interior
  pinch/fiber searches still hit the watchdog. Their geometry and runtime
  remain unresolved; finite work accounting does not establish a 30-second
  execution bound.

Between the preceding full singular run and the final frozen run, only
three obsolete budget/depth fault injections were changed to exercise the
retained mechanisms. Earlier justified oracle repairs are described above.
The failing geometry assertions above were not weakened. Exact source
oracle inputs were rechecked against the final
fixtures in [the release manifest](2026-09-09-singular-source-oracles-release.json).

## Performance record

The final quiet world-transformation benchmark completed all six curved
graph variants in **0.212–1.269 seconds**, using **5,461–12,103** work units
under the same 60,000-unit allowance. Each returned one closed component;
whole-curve coverage, length, and accumulated angular travel checks reject
missing segments and duplicate traversal. The largest sampled reference-to-
polyline distance was 0.001985 under the audit's explicit `2 * atol` coverage
threshold, with `atol=0.001`. This is a sampled verification metric, not an
exact Hausdorff theorem. [Source, counters, and all rows](../../ssx_world_mix_benchmark.json)
record the unchanged source hashes.

The [final CSX/CCX benchmark](../../ssx_completeness_benchmark_final.json)
contains 33 CSX rows, 12 CCX rows, and one SSX run, with no watchdog timeout.
The source-module baseline is `76735f9`; it uses current shared/native support,
so these timings are not a comparison of two independently built releases.
All rows called current in this artifact use the final source hashes.

| Probe | Baseline | Final |
|---|---|---|
| Case 11 boundary CSX, A-B axis 1 | 2,044 units; 0.553 s | 289 units; 0.037 s |
| Generic quadratic CCX, exact mode | One root, partial; 10,000 units; 2.570 s | Two roots, complete; 1,084 units; 0.287 s |
| Folded-patch CSX | Incorrect empty result marked complete | Two source roots, complete |
| Same first-curve parameter, two partner preimages | One exact CCX root | Both exact CCX roots |

The final complete SSX case 11 run returned one closed branch in **1.870 s**,
using **23,148 of 60,000** work units. Its 190 raw CSX calls used 3,981 units
and 0.505 s; there were no nested CCX calls or direct source-planar cuts.
Thus the remaining time in this example is largely outside raw CSX, in
source ownership, bounds, subdivision, and assembly. Singular searches still
have the runtime limitations recorded above. The benchmark scripts retain
the exact inputs, solver parameters, counters, and external watchdogs.

## Consumer contract and delivery

The public entry points and result dictionary remain in place. Consumers
must inspect `result["complete"]`; when it is false, retain the useful
geometry together with `result["unresolved_regions"]` and
`result["status"]["reasons"]`. A true value means that no unresolved
obligation was recorded by the implemented predicates, subject to the
limitations above. It is not a universal algebraic completeness certificate.

An empty partial result is not a proof of no intersection. Unresolved boxes
are conservative search obligations; candidate locations are unproved.
The `output_cap` reason can also mean that the diagnostic list itself was
truncated. A NURBS `parameter_representation` diagnostic can retain a
`local_result`: its parameters remain local to the Bezier pair, with the
global affine mapping supplied separately in `parameter_bounds`.

Finite source control data and strictly positive rational weights are now
validated at entry. Unsupported zero or negative weights raise `ValueError`.
`max_depth=None` removes the arbitrary default depth cutoff while retaining
shared work, call, output, and postprocess limits; an explicit depth remains
a hard cap. No result should be made trustworthy merely by raising these
limits.

The branch remains isolated from the original checkout. Production changes,
independent analytic generators, watchdog runners, regression tests, and
verification artifacts are delivered on that branch for review.

Production and executable verification are committed through `ee5f73a`
(on top of `a6d8770`, `9684760`, and `65b0665`). The final bounded
[independent review](2026-09-09-ssx-source-ownership-assembly-audit.md) found
no new production blocker and confirmed the explicit limitations. Before
commit, 631 final artifact file-hash records matched the worktree, and every
staged Python file matched the verified source bytes. Native binaries used
for local execution were not added to Git.

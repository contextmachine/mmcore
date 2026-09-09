# Bézier SSX search audit

Baseline: `76735f9`. Baseline line references identify original discard decisions; current implementation symbols are more stable than working-tree line numbers. This records bounded repairs and remaining obligations, not a general exact SSX algorithm.

## Confirmed root causes

1. **Geometric tolerance became parameter-space identity.** Baseline `_dedup_crossings` lines 687–714 merges by `norm(stuv_a-stuv_b)<atol`. For `S1=(16s,t,32(s-.49)(s-.4905))`, `S2=(16u,v,0)`, `atol=.001`, this merges ordinary lines eight geometric tolerances apart. Per-axis and xyz proximity fixes the unit error but is still not root identity. Current `BoundaryRootIdentity` requires an actual common exact source slice, independent source existence enclosures, and injectivity on their union. Exact affine coordinate equalities reduce supported closed-face problems. Affine interval maps use exact binary-rational arithmetic and outward conversion. Close distinct roots inside both tolerances remain separate.

2. **A tangent component consumed its entire cell.** Baseline lines 7239–7363 trace a deflated curve, enumerate Delta roots, and unconditionally continue. Delta is `Psi=0 AND all T_i=0`; it cannot enumerate ordinary rank-three components elsewhere. Current `regular_complement` descendants preserve ordinary search after singular tracing. The durable exact-binary mixed oracle is `z=3*(t-3/4)^2*((s-3/8)^2+(t-1/4)^2-(1/8)^2)`: both the repeated line and separate circle survive. Earlier decimal/interpolated probes are not exact multiplicity oracles.

3. **Internal endpoint bands erased roots without ownership transfer.** Baseline lines 7477–7478 and 7535–7536 drop every cut-CSX root within `1e-6` of an endpoint. Closed children now retain them; regression uses local parameter `2e-7` with `atol=1e-9`. Separately, `span>1e-15` mapped both faces of a representable narrow cell to `.5`; `_global_to_local` now preserves every positive representable width.

4. **Overlap endpoints lost incident ordinary arcs.** Baseline lines 670–684 remove these registrations. For `z=t*(t-s*(1-s))` versus zero, boundary overlap `t=0` shares its endpoints with an ordinary arch. Registrations are now retained. SSX calls `bez_csx(..., tolerance_tier=False)` so near-coincidence does not become topology. Exact common-image existence also needs a separate injectivity/exhaustiveness gate before consuming the entire surface pair.

5. **Marcher failure was accepted as isolation.** Baseline lines 5119–5135 plus the loop-free `continue` at 7084 discharge a cell after failed continuation. Failed or interior-truncated traces now leave subdivision obligations, with localized diagnostics at caps. A strict clipped corner can avoid launch when neither tangent orientation enters the child. A synthesized exit needs original-source existence inside that child, or exact source equality. This rejects residual-only microbranches in `z=(s-.5)^2+4h*t*(1-t)`, whose only domain roots are two endpoint touches.

6. **Displaced launch and optional closure invented connectors.** The toroidal audit found C→E while adjacent cells already correctly traced C→D and D→E, separated by an outside-child interval. The strict corner test now runs before displacement; the first valid boundary exit terminates a launch. Assembly no longer invents a closing march between nearby free ends.

7. **Failed source isolation caused duplicate traversal upstream.** Four case11 cells traced their same two registered endpoints twice, in reverse directions. Their first local source-inclusion proposals were too wide; a differently sized assembly proposal later correctly unified endpoints, creating graph valence three/four. Source proposal radii now refine geometrically within the ledger. Union uniqueness shrinks only optional padding, always retaining both complete existence enclosures. Partner consumption now happens during tracing. An independent two-root regression proves the refinement merges two representatives of one root while retaining its neighbor.

8. **XYZ containment erased different preimages.** Baseline fragment, branch, and sliver filters use spatial proximity. Even zero-slack lifted-polyline containment cannot prove source-component identity: two lens arcs can have the same endpoint chord, and an unrelated isolated root can lie on a valid approximation chord. Production deletion now needs actual source-arc ownership, shared fragment provenance, or a source-boundary retrace proof with a unique supported target inverse. Registered source-root endpoint identity controls final point cleanup. Ambiguous numerical retraces and unrelated chord-interior points remain visible.

9. **Squared-distance roundoff excluded an exact line.** For `z=a*(2s-1)`, exactly stored `a=.8435019602765045`, the line `s=u=.5,t=v` is exact. Restricting the Gram squared net to `[.5-1e-9,.5+1e-9]^4` gives minimum `1.1877787903652713e-17 > atol²=1e-22`. All SSX squared-F exclusions, allocation, propagation, and dead cell fields were removed. Unsquared component/fixed-projection hulls carry source-operand, restriction, and additive subnormal error. Preflight prices the tensors actually constructed.

10. **Rounded cofactor signs were treated as exact.** Zero minors do not exclude closed lifted fibers. More strongly, a circle graph under the exact determinant-one binary world map `[[N,N-1,0],[N+1,N,0],[0,0,1]]`, `N=2^24`, acquires a one-sign rounded minor although its exact minors straddle zero. Production loop absence and probe singular exclusion now require `SourceCofactorBounds`: original unsquared residual derivatives, source-error-aware restriction, outward determinant intervals. Rounded-minor or Gauss-map proposals alone are not the production certification fallback.

11. **Small physical diameter did not validate the lifted chord.** Let `e=2^-14`, `A=(e*(.25+.5s),e*(1/16+.5t),0)`, `B=(e*(.25+.5u),e*(1/16+.5v),1/16+.5v-(.25+.5u)^2)`. The tiny exact arc has endpoint parameters all zero/all one, but its linear lifted midpoint on B has z=`1/16`. `_small_regular_cell_arc` now separately requires source-enclosed ports belonging to the child, a strict source minor, opposite strict germs, outward cofactor-ratio contraction, exact original-source image-hull diameter, and exact bounds on both lifted source chords. Rounded child derivative estimates were removed.

12. **Postprocess denial erased discovered ordinary geometry.** The first denied assembly unit after tracing returned only overlaps. Denial now preserves assembled ordinary branches and other found output while recording the cap. The zero-postprocess regression explicitly exercises ordinary search. Closure metadata follows lifted endpoint identity instead of defaulting to false or using xyz alone.

13. **Boundary-only zero sets entered unconstrained numerical tracing.** Case8 has exact source height `c*(1-s)*(1-t)`, `c>0`, against a convex planar bilinear target. Its entire zero set consists of two source edges. Numerical tracing introduced slightly unequal representatives of the exact relation `s=u`; global parameter mapping then correctly rejected their floating aliases and emitted no branches. `_ssx_boundary_strata` now proves the nonzero one-sign Bernstein height, enumerates exactly the zero edges and remaining zero corners, and clips affine edges against the exact target quadrilateral. A nonaffine target inverse is sampled adaptively, with exact original-source bounds on every published lifted chord. Unsupported curved edges fall through; parameter representation failures stay partial. Distinct parameter edges sharing the same spatial line remain distinct.

14. **A singleton tangency used local parameters as global output.** `_deflate_tangent_cell`'s fewer-than-two-crossings return ignored its `originals` argument. The exact paraboloid touch emitted `stuv=(1,.5,.5,.5)` with `xyz=(.5,.5,0)` in a descendant cell. The singleton now retains the original global registration and point provenance. Both its helper regression and the actual isolated-paraboloid geometry test pass.

15. **Singular proposals were published before original-source existence.** Numerical witnesses one to three ulps away from the source root produced repeated or unsupported tangent points. Two point-emission sites now coalesce only identical lifted parameters; unproved candidates belong to localized unresolved records, not confirmed singularities. Independent projected overlap boxes no longer suppress points or retire multiplicity. `SourceSingularCandidates` generates a new candidate from exact affine source constraints and independently proves original homogeneous residual and all four Jacobian minors zero. The old numerical candidate is not identified with its replacement. Exact saddle and paraboloid roots survive; inexact intended-contact fixtures remain explicit candidates when source existence is unproved.

16. **Numerical tangent cones and singular size floors discharged unknown neighborhoods.** Production corner exclusion now uses source existence enclosures and source cofactor intervals; a representative rounded onto a face does not prove active-face membership. Two-port arc construction requires a definitely active source face and strict inward germs on every possibly active face touched by each enclosure. Singular witness or probe size stops emit localized multiplicity obligations. Registered graph identity or identical lifted endpoints, never a floating parameter-error threshold, supplies closure metadata.

## Search/discard ledger

| Decision | Required obligation | Current action or remaining limitation |
|---|---|---|
| Source AABB exclusion | Positive rational weights and outward Cartesian hull | Uses original supplied nets before expensive setup. |
| Local AABB/GJK exclusion | Ancestor-aware reliable separating bound | Standalone rounded-child AABB/GJK and child-scaled probe residual negatives were removed; original-operand residual separation now owns this discharge. |
| Residual hull exclusion | Operand and restriction error in lower bound | Unsquared residual or fixed projected hull; direction choice is only a proposal. |
| Cofactor loop absence | Strict source minor sign on the whole zero fiber in the cell | Source Jacobian interval determinant with exact necessary affine parameter contraction; rounded net sign and zero minor are insufficient. |
| Empty loop-free cell | Complete authoritative-source boundary census | Local CSX partials remain explicit. General rounded-child-to-original absence transport still needs care. |
| One-root regular cell | Exhaustive census and source root membership | Source enclosure wholly in child, or exact source equality, required before relative-cell isolation. |
| Two-port arc | Rank three, no loop, two distinct inward ports, faithful output | Topology, exact source velocity-range secant bound or image diameter, and both lifted chord bounds are separate checks. |
| Singular witness | Source existence, dimension, complement coverage | Exact source equations validate published candidates; unproved numerical samples record candidate locations in unresolved multiplicity boxes. Ordinary complement survives. |
| Boundary-strata early return | Nonzero one-sign exact height, exhaustive edge/corner census, injective target | Exact source clipping plus independently validated lifted representation; never promoted from residual tolerance. |
| Probe witness / tangent size stop | All singular components represented | Witnesses do not discharge the box; unsupported size stops retain typed unresolved obligations. |
| Registration consumption | Reached event is the same source root | Common exact face, independent existence, union uniqueness; charged refinement cannot erase the neighbor enclosure. |
| Local CSX depth/resolution stop | Preserve independent domains | Found roots remain; owner boxes carry diagnostics. Source recovery discharges root-only reservations only when the entire pending region is covered by source existence and uniqueness. |
| Geometry deletion | Actual source-path identity or structural source ownership | Even exact approximation-chord incidence is insufficient. An incomplete parent trace batch is provisional when a full child partition takes ownership; it remains a fallback for unresolved descendants. |
| Shared caps | No completion/emptiness claim | Found output and typed diagnostics survive; defaults were not increased. |

## Verification and remaining work

- Focused search, root-identity, C1, and source-cofactor set: **45 passed in 12.83 s**, including the final source image and lifted-chord guards.
- Boundary-strata analytic helper tests: **19 passed in 0.64 s**. Both orders, degree elevation, clipping, isolated corners, curved target inverses, distinct preimages, representation floors, and interrupted-prefix retention are covered.
- Ordinary case11 after standalone rounded-child AABB/GJK removal: **complete, one closed 230-vertex curve, length 1.5325808731, 10.42 s, 49,311 shared cells**, no failed local traces. This snapshot predates the subsequent original-source planar-cut integration. The earlier false-complete two-open-branch output was diagnosed and repaired.
- Parent-owned exact extrusion/plane and straight-path tiers handle supported high-order loci without confusing tiny residual with distance to the locus.
- Rounded critical faces and retained ancestor prefixes caused nested-loop fragmentation. Cut relocation experiments did not fix the cause and were removed. Original-source planar-cut census, exact source root identity, and provisional trace ownership subsequently restored the nested pair to two complete closed loops in the parent audit. The wider nearby-loop and transformed matrix still needs a fresh final run.
- Toroidal review replaced invalid sampling-dependent retrace and surface-midpoint sagitta proxies with independent corrected-locus checks. The reviewer isolated the reverse-order missing arc to a nested CSX owned-cell budget denial. After that substrate fix, its full **25-test suite passed**, including reverse-order `atol=1e-4` and `1e-5` sagitta. Later source-proof changes still require their final integration sweep.
- The full legacy singular file reached its **300 s watchdog** before completion. Subsequent focused diagnosis restored exact paraboloid and saddle output through source candidate and coordinate fixes. Decimal/interpolated contact fixtures do not prove the intended multiplicities of their binary input nets; unresolved candidate output is retained rather than asserted to be a confirmed tangent point.

## Complete per-test singular diagnostic

The later diagnostic collected all **117** singular tests and ran each in a
separate process with an external **60 s** watchdog. Tests affected by the
temporary scalar-tensor restriction error were rerun after that error was
fixed. The resulting snapshot was **77 passed, 38 failed, 2 timed out**;
the JSON companion records every test, elapsed time, first failure, and
the production source hashes loaded by its process. This is a diagnostic
snapshot, not a final all-suite success claim.

Subsequent bounded fixes passed **33 tests in 5.96 s**: all 25 boundary-strata
tests, the two actual shared-edge variants and their schema test, and five
fault-injection/progress/diagnostic tests. The latter now exercise the
actual numerical boundary gate or general search path; exact source tiers
can correctly bypass those paths. Unresolved regions retain their own
causes instead of all being required to say `depth_limit`.

The new opposed-halfspace certificate exhausts a general exact class:
one-sign nonzero Bernstein heights on opposite sides of a common plane
force every common image onto zero boundary edges/corners on **both**
charts. All affine stratum pairs are intersected in exact rational
arithmetic and their published lifted paths are independently bounded on
both source surfaces. This restores the shared-edge example to two
complete overlap branches plus the exact tangent junction. Exact endpoint
provenance survives publication; distinct exact endpoints that round to
the same floating tuple produce `parameter_representation`, never an
invented junction.

An independent exact symbolic audit of 13 graph fixtures/variants found
the unit Groebner basis for `(P, P_s, P_t)` in **every** case. Consequently
their supplied binary control nets have no tangent point anywhere, even
over the complex numbers. This includes the old double-touch, Mexican
hat, touch-plus-loop, line-plus-touch, and closed-tangent-loop families.
The result invalidates their claimed exact multiplicities, not their
ordinary nearby curve-coverage obligations. The two 60 s timeouts were
the old line-plus-touch and closed-tangent-loop examples. Source-polynomial
coefficients and exact CAS results are in
`2026-09-09-singular-source-oracles.json`; the optional-SymPy reproducer is
`examples/ssx/ssx5_singular_source_oracle.py`.

Two additional construction errors are exact and simpler:

- The split-plane cusp fixture has x coefficients
  `[1, -float(1/3), -float(1/3), 1]`; its quadratic minimum is
  `1/2**56 > 0`, so its source never reaches the target plane x=0.
- The supposedly exact homogeneous multiplication in the rationalized
  paraboloid fixture gives source homogeneous height `3/2**57 > 0` at
  the claimed center, so that sample is not an exact tangency.

Real remaining issues in that snapshot include repeated fragments around
collapsed rational endpoint fibers, missing exact saddle singular
classification after affine cell contraction, retained but uncoalesced
regular/overlap retraces, non-affine planar region completeness, and
source-critical proposal discovery. These are separate from the invalid
old contact oracles and must
not be hidden by reinstating tolerance-based root confirmation or
proximity-based deletion.

The exact cusp counterpart has subsequently been repaired: with x
coefficients `[3,-1,-1,3]`, the source normal is an exact affine vector
polynomial along each ruling. An identically zero normal produces the
existing `cusp_curve` entity, source-side attribution, represented samples,
and branch links; zero normals are not classified as ordinary tangency.
Eight exact controls cover operand order, parameter transpose, and positive
uniform homogeneous weight scales. An isolated normal zero along a ruling
currently retains the general solver as its owner. The original rounded
cusp fixture now proves its exact positive gap and expects empty output.
The extrusion plus C1 gate passed **45 tests in 9.61 s**, followed by the
expanded source-candidate/boundary/extrusion/C1 gate of **80 tests in 9.85 s**.

Boundary registration now independently checks possible exact singular
events. Exact source residual/minor equality, nonzero chart normals, and
rank four of `D(Psi,T)` prove an isolated regular-chart singular root.
This restores the exact saddle tangent-point metadata without sampling an
entire tangential curve into point entities. Failure is inconclusive and
does not discharge any search obligation. Independent arithmetic review
confirmed the homogeneous derivative argument; degree-zero charts are
explicitly refused before evaluating empty derivative nets.

Point cleanup retains live boundary-root owners on points and branches,
then coalesces independently sampled endpoints only after source root
identity succeeds. The proposed segment-AABB tree reduced geometric
point/polyline query work, but review established that geometric incidence
does not prove source-arc incidence. Its generic production point-deletion
call was therefore removed. The helper remains a geometric query utility;
it is not a topology certificate.
- General continuation, legacy singular enumeration, and rounded-boundary absence transport remain outside a fully validated interval topology engine. `complete=True` means no recorded unresolved obligation under implemented predicates, not an arbitrary-input algebraic proof.

## Refreshed singular disposition

The final per-test diagnostic completed all 117 tests with individual 30 s
watchdogs: **86 passed, 28 failed, 3 timed out**. The durable
`2026-09-09-singular-final-diagnostic.json` records every outcome and
before/after source hashes. This improves the prior 77/38/2 snapshot, but
the older watchdog was 60 s, so timing counts are not directly comparable.

Two failed tests were subsequently repaired: a real lost rational
transversal line needed exact constant world-coordinate equations in the
necessary affine parameter constraints, and the zero-postprocess fault
injection needed to exercise ordinary assembly rather than a now-exact
shared-edge tier. The combined focused gate passed **88 tests in 1.49 s**,
including exact saddle full-diagonal coverage, L-junction metadata, root
identity, boundary strata, and trace ownership. The rational case14
scaling watchdog passes unchanged in 29.81 s on an individual follow-up.

Six additional exact source-polynomial audits cover the rationalized
paraboloid, skew near-overlap contact, repeated-call loop fixture,
anisotropic ellipse, and double-touch scales 10/100. All have a unit
Groebner basis for the necessary regular plane-contact equations (the
skew chart uses its exact nonzero-s factor). These account for 16 failed
tests with invalid asserted tangent-contact premises. They do not prove
that the solver covers all nearby ordinary components. The two old
line/tangent-loop tests remain unfinished after 90 s follow-ups.

Real remaining snapshot failures are the ordinary case11 depth partial,
nonimmersed cone/pinch fibers, and five nonaffine planar-region tests
(the rational twin has no region output). Case13's numerical tangent
witness has no source-existence proof; its pending source region remains
partial, and neither exact existence nor absence is claimed. The complete
per-failure disposition is `2026-09-09-singular-final-disposition.json`.
None of these unresolved geometry assertions were weakened.

Two additional repaired search causes are now covered by source tests.
Boundary source roots at an exact singleton4D corner must be identical
even when different face polynomials use different varying axes. Lazy
source recertification on each event's own face supplies that proof when
the numerical registrations initially have no common face. For exact
quadratic saddles, rational affine factorization followed by exact
residual validation exhausts both source-line components; all four arms
and their exact shared junction survive. Published float aliases between
distinct exact component endpoints remain representation partials.

The cone snapshot was then repaired by the same exact factor framework.
Each factor now receives its own exact linear residual elimination;
remaining nonlinear equations still cause fallback. A constant image
with an identically zero source normal is represented as a full parameter
fiber (`cusp_curve`), while its incident generator keeps a separate
ordinary branch and an exact source endpoint link. A nonzero one-sign
Bernstein normal component proves immersion on each open line interval;
endpoint normal zeros are explicitly typed. The unchanged cone test and
independent operand/parameter/homogeneous-scale controls pass. The source
pair now completes in **0.00395 s, 1,428 work units, zero CSX calls**,
compared with the prior full 250,000-unit partial with no generator.
The interior pinch retains residual nonlinear equations and remains
outside this bounded affine-factor class.


## Final source-root and geometry verification

The exact quadratic reducer now handles a zero-dimensional result of its
PSD/RREF equivalences. A full-source singleton is clipped to the unit
parameter box, checked against the original exact polynomials, and typed
using exact source normals. Publication separately checks world-coordinate
rounding and the images of both rounded parameter representatives. This
proves the seam-straddling paraboloid contact without relying on a small
residual or a singular-search witness. Domain-boundary roots, nonimmersed
charts, homogeneous scale changes, output denial, and unrepresentable world
coordinates have independent controls.

Raw source root boxes can be valid yet too broad for a child cell.
`BoundaryRootIdentity.refine_enclosure` now builds a temporary independent
source enclosure and accepts it only when wholly contained in the original
source-root-bearing uniqueness box. Failed refinement preserves the old
box and certificate. This restores ordinary four-line cell ownership while
preserving distinct close roots. Parent integration also removed the
arbitrary derived default depth of 13; explicit depth limits remain hard
limits, and the existing work allowance still bounds default search.

The final focused gate passed **138 tests in 7.25 s**, with no source or
test hash changes during that run. It includes quadratic/factored/cone/0D
controls, source constraints and root identity, boundary strata, provisional
trace ownership, full saddle and rational-line geometry, the L-junction,
actual seam tangency, and the generic witness fault injection. The two
positive-gap endpoint controls retain their complete geometry assertions;
the exact supplied Bernstein coefficients prove two isolated endpoints,
and the newly completed search now reports that result as complete.

The final analytic matrix passed **63 of 63** whole-geometry checks, with
zero watchdogs or silent failures: nine independently defined families
under seven operand/parameter/knot variants. The artifact
`2026-09-09-analytic-matrix-final.json` records both source manifests and the
change during case 32; this result is intentionally not described as a
single frozen-source run. The earlier 117-test singular diagnostic remains
a separately versioned snapshot. Its unsupported nonaffine region/pinch
classes and near-root performance obligations have not been relabeled as
passing tests.


## Fresh entire singular suite after final source changes

A new full run, preserved separately as
`2026-09-09-singular-final-current.json`, completed all 117 tests with
individual 30 s watchdogs and **one unchanged source manifest**:
**84 passed, 23 failed, 10 timed out**. This supersedes the old 86/28/3
snapshot for current suite disposition, without replacing its history.
Case11, the exact rational line, the cone, and the corrected zero-postprocess
test now pass.

The run exposed real rational-boundary regressions: the regular branch
between two collapsed rational endpoint fibers is absent, and case14's
generator tests also return no branch. The source-census path no longer
exports the original collapsed-edge seeds; that repair is assigned to the
substrate agent. A separate collapsed-edge schema test loses its
`parameter_fiber` reason. Three newly failing fault-injection expectations
are obsolete: two require work denial from assembly scans that were removed
as unsafe, and one expects a default depth ceiling after that implicit cap
was removed. Those tests were not changed in this diagnostic.

The ten watchdogs remain failures of bounded verification: the two previous
near-contact runtime cases, one skew near-overlap case, five nonaffine planar
region cases, and two pinch/fiber cases. The artifact includes every failure
and timeout, all assertion outputs, source hashes, and an explicit comparison
with the previous full run. The 138-test focused pass and 63-case analytic
matrix remain valid narrower gates, not a claim that this full singular
suite passes.


## Bounded unsupported strata and projected source coordinates

The five nonaffine planar-region watchdogs now terminate through an exact
source owner. For positive-weight bilinear charts, exact strict convexity
of the ordered corner quads plus the nonzero signed homogeneous planar
Jacobian proves each map is a homeomorphism onto its quad. The Jacobian
determinant's degree and corner signs are checked in exact arithmetic.
Exact convex clipping with positive area then proves the entire paired
zero set is one two-dimensional component. No isolated or ordinary
complement can hide behind this classification. The solver retains one
explicit whole-domain `overlap_region_unsupported` record containing the
exact image polygon; it does not manufacture paired UV boundary samples
or claim completeness. All five source cases return in 0.00055–0.00276 s
with 922–1,539 charged units and no CSX calls. Their original unsupported
region geometry assertions remain unchanged.

Three obsolete fault injections now exercise real mechanisms: registered
endpoint graph lookups deny postprocess work while preserving all supplied
arcs, and the depth-diagnostic test explicitly requests a depth ceiling.
The expanded focused gate passed 151 tests in 10.13 s on unchanged hashes.
The substrate agent's separate source-fiber follow-up repairs the four
rational branch/fiber regressions detected in the full run. These targeted
passes do not rewrite the earlier full-run counts.

A further ordinary counterexample used two curved graphs with an exact
circular parameter intersection. An integer invertible world transform
hid all individually affine world coordinates, so necessary parameter
constraints disappeared and the transformed search exhausted its budget.
Exact three-column nullspaces of the original source coefficient rows now
find common affine projections q.P. Each row identity proves the projected
coordinate is affine in one parameter on each entire source chart; only
uniform positive weights are supported. The projections enter a copy of
the constraints maps under distinct coordinate IDs, leaving inverse and
immersion maps unchanged.

That repair exposed a separate continuation defect. With projected face
identity deliberately disabled, the transformed result covered the entire
physical loop to 0.001914 at atol=0.001, but accumulated angular travel
8.63938 instead of2pi. The earlier inference of missing arcs from far open
endpoints was incorrect: three known-two-port cells emitted opposite
copies of the same arc. An unregistered exit with a source-existence box
left the known partner unused, so it launched a reverse trace while the
cell was marked complete. Equivalent source-face constraints now select
the appropriate identity proof. Independently, the tracer must identify
the already proved partner of a two-port arc before discharging it; failed
identity retains the owner as unresolved. No closure flag or proximity
merge substitutes for this proof. The six transformed whole-component
controls, 24 exact constraint tests, and refusal regression passed together
(31 tests).


## Final frozen analytic matrix

`2026-09-09-analytic-matrix-frozen.json` records **63/63 passed**, zero
watchdogs, zero silent failures, and **one unchanged source manifest**.
This run includes the projected-coordinate and known-port continuation
guard, the exact source-root transport, and the positive-dimensional owner
termination. It supersedes the earlier mixed-manifest matrix for final
ordinary analytic coverage evidence. The new full singular rerun is queued
after the NURBS/toroidal correctness window to avoid watchdog contention.


The full singular watchdog driver is delivered as
`examples/ssx/ssx5_singular_regression_audit.py`. Run it with the configured
repository Python environment:

```sh
python examples/ssx/ssx5_singular_regression_audit.py --output /tmp/ssx-singular-audit.json --timeout 30 --workers 2
```

The output/log destinations must not already exist. The driver checkpoints
every node atomically, records hashes of all `mmcore` Python sources and SSI
fixture scripts before/after each test, and preserves failing assertion
output and external-watchdog outcomes. `--test-file` supports a bounded
runner smoke test; the default collects the entire legacy singular file.
A one-node real pytest run verified the delivered driver.

## Release verification on frozen sources

The latest ordinary matrix, after the source-fiber no-seed fallthrough
repair, is `2026-09-09-analytic-matrix-release.json`: **63/63 passed**, no
watchdogs or silent failures, and one unchanged source manifest. The
earlier frozen matrix is retained as history.

The delivered driver then completed the entire singular file with two
workers and independent 30 s watchdogs. Its authoritative artifact is
`2026-09-09-singular-final-frozen.json`: **91 passed, 21 failed, 5 timed
out**, with one unchanged manifest covering all `mmcore` Python sources,
SSI example scripts, and the test file. Seven failures from the preceding
84/23/10 snapshot now pass: the four rational branch/fiber regressions and
the three corrected budget/depth fault injections. The five unsupported
region cases now fail their unchanged geometry assertions promptly rather
than reaching a watchdog. There are no newly failing tests in this run.

The 26 nonpassing nodes are individually recorded in
`2026-09-09-singular-release-disposition.json`:

- Fifteen fixtures assert regular tangent contacts inconsistent with their
  exact supplied binary coefficients. Reconstructed source arrays are
  byte-identical to all 19 independently audited oracle inputs in
  `2026-09-09-singular-source-oracles-release.json`. This does **not** prove
  the remaining ordinary zero sets are represented correctly. In
  particular, nodes 23 and 42 currently fail branch-count assertions;
  their nearby ordinary geometry and branch classification remain
  independently unresolved.
- Case13 still has no established original-source singular existence or
  absence proof. Its unproved numerical contact is not published as a
  confirmed tangent point; the pending source owner remains partial.
- Five convex nonaffine planar region cases have an exact two-dimensional
  owner and exact physical image polygon, but no supported public paired
  UV region representation. Their required region geometry is still
  missing, and their tests remain failed.
- Three near-contact cases and two nonlinear interior pinch/fiber cases
  still exceed the 30 s per-node watchdog. Their runtime and required
  remaining geometry are outstanding. Invalid contact premises do not
  excuse those timeouts.

Both repaired case14 tests finish within the original watchdog
(26.71 s and 26.77 s in this correctness run), so no longer-timeout rerun
was substituted for their results. The remaining five watchdogs are the
same outstanding nodes as in the preceding 30 s run. These are correctness
audit durations under two concurrent test workers, not isolated benchmark
claims. No failing geometry oracle was relaxed, skipped, or counted as a
pass in the final disposition.

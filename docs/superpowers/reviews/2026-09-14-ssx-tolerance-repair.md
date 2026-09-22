# SSX tolerance repair

Status: implementation and validation in progress; not yet integrated into `tiny`.

## September 23: case 16 and test-contract correction

The user's new `case16.py` cases were compared using the same Python environment,
native extensions, and committed `_case16_data.py`. Local `tiny` returned the
whole curve for all five pairs. The repair initially lost 725–925 model units
from one end while reporting a complete result.

The loss happened inside Bézier assembly, before NURBS stitching. A cleanup
filter compared the two surface evaluations at interpolated parameter midpoints
and treated their residual as distance from the intersection. This is not a
geometric chord-error measure: different surface charts need not interpolate
corresponding physical points. It deleted correctly traced fragments, including
ordinary transverse intersections. The filter and its empirical sagitta-credit
constant are removed. Geometry must be refined during tracing; deleting the
whole recovered component cannot repair an inaccurate chord.

All five real cases now pass endpoint and original-source residual checks at
`1e-3`. Twelve independent straight-intersection cases with nonlinear parameter
correspondence reproduce the old deletion and pass after removal. The original
touch-plus-ring controls also pass with the corrected CAD contact admission.

The accompanying [test audit](2026-09-23-ssx-test-contract.md) replaces excessive
output precision with physical-distance checks, while retaining coverage,
paired-source geometry, and actual work-limit checks. These focused results
do not replace the pending complete integration run.

## Regression and responsibility

The working reference is `76735f9fc11a5e8149fb60d171f46cacd0ff4ec9`.
The subsequent changes imposed algebraic root-existence and ownership conditions
on a numerical CAD intersection API. This was an implementation decision, not a
requirement agreed with the user. It rejected useful tangent points and overlap
geometry at the requested modeling tolerance. Returning rational diagnostic
regions did not repair the missing geometry.

The relevant changes, in order, were:

| Commit | Change and consequence |
| --- | --- |
| `a6d8770` | Added exact parameter-root handling to CCX/CSX; introduced the first pinched-surface regression. |
| `9684760`, `65b0665` | Extended exact ownership and original-residual requirements through the nested curve solvers. |
| `ee5f73a` | Applied the source-topology redesign to SSX and assembly; tangent publication and overlap output regressed. |
| `65dff9d` | Recorded the audit without resolving the failing integration tests. The validation used for integration was insufficient. |
| `ec5c958`, `87bab06`, `4662fab` | Later example, Bernstein sorting, and version changes; preserved by this repair. |

The reported GitHub failures were not expected or acceptable. The earlier
focused passing runs did not establish that the full suite passed.

## Repair

The Bézier and NURBS SSX implementation starts again from the working numerical
baseline. Tangent points, cusp geometry, overlaps, and ordinary branches use the
requested CAD tolerance. Public `unresolved_regions` and the unused exact SSX
machinery are removed. Actual caller-imposed resource limits still retain the
geometry already found and report the existing status fields.

The repair also addresses numerical defects independently of the withdrawn
algebraic policy:

- Paired surface parameters and spatial distance both participate in joining and
  deduplication, preserving separate sheets that happen to occupy the same place.
- Supplied NURBS weights are retained, including values close to one. Periodic
  joins use the actual surface seam and preserve paired parameter paths.
- A failed continuation keeps its boundary seeds available for recovery. Face
  tangency uses curvature as well as the first derivative. A collapsed surface
  parameter direction is distinguished from a missing spatial branch.
- Recovery prefixes are sampled adaptively on the original surfaces instead of
  inserting a long straight displacement chord.
- Planar overlap rims are clipped together and their inverse parameter curves
  are refined against both source surfaces.
- Numerical isoline paths avoid redundant four-parameter searches when the
  surface structure supports them; unsupported inputs use the general solver.
- Default CAD CCX/CSX no longer wait for optional exact-root certificates.
  Constant curves and collapsed target isolines no longer fill the root list
  with duplicate representatives of one geometric contact.
- General continuation runs before the independent parameter-singularity
  census, so an extended singular set cannot consume all remaining work before
  an already seeded intersection branch is traced.
- CSX overlap sidedness tests the signed normal gap itself. Tangential
  projection error no longer turns normal rounding noise into a false crossing.
- Validated point contacts absorbed into a CSX overlap remain available to
  internal SSX cut faces, with their paired parameters intact.

## Test contract

The original geometric assertions for the 25 failing singular/overlap tests are
retained. Restored cusp assertions also replace the later assertion that a
roundoff-sized gap should erase the cusp.

Forty test files for the withdrawn internal exact SSX implementation are
retired; none existed at the working baseline commit. Their meaningful public
geometry cases are carried into the CAD strata,
analytic coverage, lifted assembly, and practical NURBS tests. These include
nested circles, clipped boundary lines, high-order rulings, curved tangential
paths, multiple parameter preimages, saddle arms, overlap rims, and seam joins.
The four CI assertions about exact rational enclosures or bit-identical rounded
Sturm roots are not requirements of the restored numerical API.

The suite still checks complete reference-curve coverage, missing sectors,
retracing, geometric chord error, surface-order reversal, coordinate transforms,
nonuniform weights, and preservation of previously found geometry at a work
limit. Tests of intentionally interrupted internal paths explicitly select that
path; they do not require a faster supported path to perform unnecessary work.

## Performance evidence

The call chain is SSX to CSX to boundary CCX. CCX itself does not call CSX.
The repairs address repeated work at both levels: redundant regular-arc seeds
in SSX and duplicate free-parameter representatives in CSX.

Measured on the same local machine against `4662fab`, using three raw-solver
runs per fixture:

| Case | Before | Repair |
| --- | --- | --- |
| Endpoint tangency, CCX | 83 cells, about 70 ms, stopped incomplete | 12 cells, about 4.5 ms, complete |
| Pinched isoline, CSX | 301 cells, 127 representatives, about 40 ms | 43 cells, 2 representatives, about 4 ms, complete |
| Regular plane, CSX | 191 cells | 191 cells |
| Fold with two preimages, CSX | 94 cells, both preimages | 94 cells, both preimages |
| User partial-overlap example, CSX | 188 cells, about 32 ms | 198 cells, about 42 ms |

These are targeted measurements, not a claim that every case became faster.
The endpoint-CSX example also does more work after repair because it now
continues past the former early stop. Source hashes and call/cell measurements
are recorded by the benchmark scripts; elapsed times depend on the machine.

A separate CCX cost was repeated polishing of identical subdivision-corner
starts against unchanged original curves. A bounded cache local to each search
reuses those calculations without changing tolerances, cell traversal, or
results. Four captured cut-face queries produced byte-identical serialized
results and identical cell counts with caching enabled and disabled. Their
polish counts changed from 353/853/1391/1390 to 179/422/683/682; elapsed time
decreased by approximately 31–38% in that comparison.

## Validation

Final full-suite results, clean Linux build results, and integration commit are
pending. Focused results are diagnostic evidence only until those gates finish.

The first broad repair run at `9cea23c` was not green: macOS reported 6 failures
and 1241 passes; Linux aarch64 reported 7 failures, 1239 passes, and one xpass.
The original reported singular/overlap regressions passed. These broader runs
exposed the cone-census scheduling defect, Linux overlap-sidedness defect,
sheared-circle search cost, and two stale test-path/status expectations. The
case-10 expectation now includes an independent analytic check of its entire
curved tangential ruling; the boundary fault-injection test explicitly selects
general continuation and verifies that its injected calls occur. The remaining
production corrections require another complete run before integration.

The local clean Linux environment is Python 3.12 on Linux aarch64, not the
GitHub-hosted x86_64 runner. A local pass must not be described as a successful
GitHub Actions run.

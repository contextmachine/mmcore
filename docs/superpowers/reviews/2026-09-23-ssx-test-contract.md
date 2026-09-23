# SSX test contract audit

The CAD intersection contract uses the caller's model-space tolerance, normally
`atol=1e-3`. Tests must not impose extra parameter digits or an algebraic proof
as conditions for accepting otherwise valid geometry. The checks below were
reviewed against that contract before modification; they were not adjusted to
make an observed geometric failure pass.

## Corrections

| Test | Change and retained protection |
| --- | --- |
| `test_ssx5_isolines.py`: multiple-root and nonbinary clipped rulings | Replace `1e-10`/`1e-11` output comparisons with Euclidean endpoint and line-position error at requested `atol`; also check both source evaluations and their paired residual. Branch count and type remain unchanged. |
| Same file: public collapsed isoline | Replace `1e-10` point/endpoint position checks with model-space `atol`; verify source evaluations. Preserve point versus real curve, fiber ownership, and full parameter coverage. |
| `test_ssx5_matched_isolines.py`: curved ruling and world-axis mixing | Replace exact endpoint coordinates and `1e-12`/`1e-11` geometry checks with model-space errors. Transformed geometry is checked in the world frame where `atol` applies. Preserve source/chord checks and the single tangential branch. |
| `test_ssx5_probe_split.py`: near-unit weights | Use the requested `1e-5` for line position and extent instead of `1e-9` and exact extent equality; add both-source residual checks. Both spatially separated branches are still required. The explicitly supplied tolerance is unchanged. |
| `test_ssx_nonplanar_projection_invariance.py` | Keep strict signed winding, one closed branch, complete circle coverage, and length checks. Replace `1e-6` radians of permitted reverse travel with at most one `atol` of reverse world-space travel. |
| `test_nssx5.py`: multispan crossing | Measure complete endpoint extent in world coordinates at `1e-3`, replacing a fixed `1e-6` parameter bound. Preserve one continuous branch and both-plane geometry checks. |
| Same file: seam tangent point | Allow successful completion instead of requiring historical partial reasons. Still require one tangent point at the independent expected position and consistency between completion and reasons. |
| Same file: case 10 | Preserve the newly added complete analytic contact and reference-curve coverage. Replace `1e-10`/`1e-12` parameter comparisons with per-vertex distance to `(5+15*t*(1-t),15*t,5)`, using the physical output y coordinate to identify t. Source residual checks remain. |
| Same file: boundary coincidence | Use `1e-3` for output distance from the plane instead of `1e-6`; add paired source evaluations. Preserve both branches, endpoint matching, and length checks. |
| `test_bez_ssx5_singular.py`: cusp ruling | Evaluate sample distance from the known spatial cusp curve at `1e-3`, replacing fixed `1e-6` parameter accuracy. Preserve cusp classification and extent. |
| Same file: collapsed-fiber branch, cases 13 and 14 | Allow successful completion instead of requiring old incomplete outcomes. Preserve resource caps, branch/touch counts, expected geometry, and source residuals. Cases 13/14 test names now describe the recovered geometry. |
| Same file: case 13 normal angle | Remove the additional `sin(angle)<=1e-6` requirement at approximate output parameters. The typed touch, independent expected location, and paired source residual remain checked; curvature and chart scale make a universal angular error inappropriate for a model-space tolerance. |
| Same file: identical overlap patches | Compare source evaluations of the interior witness within `atol`, rather than requiring `1e-6` equality of parameters. Require both parameter pairs to remain interior; preserve the whole region and closed rim checks. |

## Checks intentionally retained

- Component counts, distinct parameter sheets, full curve coverage, missing
  sectors, closed-loop winding, and geometric errors larger than requested
  tolerance are not relaxed.
- Exact copied data, joined endpoint identity, preservation of supplied weights,
  knot/periodicity metadata, and fixture-construction arithmetic remain strict.
- Forced budget/depth tests still require the actual stop to be reported and
  previously recovered geometry to remain available.
- Tests explicitly selecting optional exact raw backends remain separate from
  default CAD behavior; this audit does not change their arithmetic assertions.

## Raw CSX/CCX follow-up

The initial audit found default-mode tests excluding contacts within `atol` and
requiring empty partial output for near-coincident curves. Those expectations
are being corrected together with the source policy, rather than hiding the
inconsistency by moving the default-mode examples into exact mode.

- `test_csx4_exactness_contract.py` now checks collapsed-curve and exterior-edge
  contact membership on both sides of `atol`, including translated inputs and
  paired source residuals. Separate explicit `tolerance_tier=False` checks keep
  the optional exact-set behavior scoped. The large-translation overlap test
  uses the input ULP scale for its residual measurement comparison, preserves
  the `1e-3` geometric bound, and no longer has a Linux xfail.
- `test_bez_csx4.py::test_case_13` now requires both the rounded endpoint contact
  (about `2.3e-8` from the surface) and the separate interior crossing, with
  source residual checks before and after translation. This is distinct from
  the SSX case 13 fixture.
- The CCX close-root test now uses faithful dyadic controls at scale `2^14`,
  with valley depth above `atol`, physical root-location checks, and the normal
  solver allowance. The former `1e10` coefficient multiplication shifted its
  intended roots by about `0.0078` and `0.1216` world units; its `1e-9` parameter
  assertion allowed 10 world units. Neither the distorted reference nor the
  arbitrary 2,000-cell cap was retained as a production requirement.
- The two near-coincident cubic tests in `test_bez_ccx4.py` now require real
  overlap geometry. The offset twin has a full `[0,1]` pairing with a residual
  near `1e-9`. The fitted pair has correspondence
  `u=[0,0.827597762202295]`, `v=[0.1906907548416867,1]`, whose entire cubic
  difference is bounded by `3.78e-9` from its Bernstein control hull. Its
  unused tails end about `4.89` and `5.03` model units away. Returned overlap
  spans are checked with the same independent continuous difference bound,
  and omitted range is bounded by physical control-polygon arc length.
  Complete output is allowed; an explicit partial result is allowed only if
  its actual spans retain this geometry. An empty diagnostic cannot pass.

## Verification

The new tangent-line/regular-circle regression initially imposed `atol` on
the circle's polyline coverage. General continuation already allowed
`2*atol` chord sagitta at the working baseline `76735f9`; this was an
inappropriate new assertion. Linux returned both complete, correctly typed
components with `1.568*atol` maximum circle-to-polyline distance. The circle
coverage check now uses the existing `2*atol` chord allowance. Straight-line
coverage and analytic vertex accuracy remain at `atol`, and both-source
vertex residual checks at `atol` are added. Component counts and kinds are
unchanged; the solver and its tolerances are unchanged by this correction.

The subsequent case11 check replaces its fixed 32-vertex minimum with
bidirectional coverage of the supplied Rhino reference and paired-source
residuals. The positive-gap endpoint-touch controls retain their two analytic
contact locations and allow completion when corner classification resolves
their local arcs. The source fix and these checks passed 17 tests.

- Isoline, matched-isoline, and probe tests: **41 passed** (3.93 seconds).
- Initial focused public NURBS, singular, and six transformed-circle cases:
  **12 passed, 2 failed** (89.44 seconds).
- That run exposed source defects: case 13 returned 15 untyped points, and the
  identity circle reported `depth_limit`. Their geometry/status assertions were
  retained; neither is converted into an expected failure.
- Fixture cases 5, 8, and 10 after the final assertion correction:
  **3 passed** (1.14 seconds).
- The revised near-coincident cubic tests initially both failed on empty
  overlap lists (8.24 seconds), confirming that they expose missing geometry.
  Their coverage helper accepts independently constructed valid complete and
  partial spans and rejects a truncated span. Source-fix verification is
  recorded in the main repair report.
- `git diff --check` passed. This note is not a full-suite green claim.

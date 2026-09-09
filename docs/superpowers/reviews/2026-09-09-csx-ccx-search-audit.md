# CSX and CCX substrate audit

The original nested cost was `SSX -> CSX -> CCX`. A public CSX boundary analysis
projects the two curve endpoints onto the patch and intersects the curve
with up to four patch edges using CCX. CCX does not call CSX. SSX uses the
raw Bézier CSX solver; NURBS CSX adapter changes alone cannot alter this
cost or repair its boundary topology. SSX now uses an exact original-source
cut tier where supported, otherwise a full source-residual CSX search that
bypasses the derived-control CCX boundary phase.

## Established failures and changes

* Weak squared-distance derivative signs discarded a subdivision face
  carrying exact roots. For `S(u,v)=((u-.5)^2,v,0)` and
  `C(t)=(1/16,.5,t-.5)`, baseline 76735f9 returned no roots and complete
  status. Both regular roots, at `(.5,.25,.5)` and `(.5,.75,.5)`, are now
  preserved. Exact CSX topology no longer uses unvalidated scalar squared
  distance hull, Lipschitz, or derivative pruning.
* A boundary isolated root excluded its entire curve-parameter slab,
  deleting other surface preimages. Exact CSX now searches the remaining
  parameter product and requires an interval-Jacobian at-most-one-root
  certificate before consuming a root neighborhood. Termination must
  resolve all three parameters. Ambiguous terminal boxes are partial.
* Pure geometric CCX deduplication merged distinct parameter pairs at
  the same spatial point. Exact adapters preserve root certificates and
  both parameter preimages. Generic CCX searches the complete parameter
  product, including two partner roots with the same first parameter.
  Both public and exact modes mark neighborhoods without a uniqueness
  certificate partial; a modeling tolerance does not prove root identity.
* Tolerance-overlap certificates were consumed as exact SSX boundary
  topology. CSX now provides `tolerance_tier=False`, used by SSX, and keeps
  the public modeling-tolerance mode available. Exact convex planar
  overlap and scalar plane-root reductions avoid dense projection trials.
* Outside-cell Newton proposals bypassed CSX's owned root-box identity
  logic and were repeatedly published with ulp-shifting parameters. A
  rational torus cut filled its 128-root cap with copies of one root.
  Exact search now publishes through closed-cell ownership and keeps the
  outside-attractor cell in search; that captured cut returns one root
  with complete status in 511 cells.
* A witnessed affine curve-on-surface relation does not exhaust a folded
  patch's parameter preimages. Generic exact affine CSX overlap requires
  a uniform UV injectivity certificate before discharging its slab;
  otherwise it returns the witness and an explicit unresolved complement.
* Restriction helpers ignored trims below `1e-12`, split helpers dropped
  positive intervals below `1e-15`, and boundary-root cutouts retained
  their nominal center. These now use representable interval boundaries
  and explicit center-box identity.
* Residual hull errors based only on shrinking, cancelled coefficients
  could falsely exclude true roots. CSX propagates source/depth-derived
  error bounds; projection exclusions price their dot-product operands.
  The shared error bound also accumulates an additive subnormal rounding
  allowance per operation, rather than relying on a relative error model
  after underflow. Repeated restrictions of subnormal controls give a
  concrete exact-arithmetic regression for this distinction.
  The low-variance coefficient direction supplies a stronger separating
  projection, with the same bounded Bernstein sign acceptance test.
* Near-unit rational weights were treated as unit weights by adapter
  `allclose` checks. Exact metadata predicates now retain those weights;
  the parent change also repairs curve knot insertion/decomposition.

## Exact supported reductions

`csx/_planar_roots.py` uses exact rational arithmetic over the supplied
binary coefficients. For positive-weight curves and regular convex planar
bilinear patches with uniform surface weights, it isolates every root of
the homogeneous plane-distance polynomial using square-free factorization
and Sturm counts. Exact halfspace signs at each algebraic root establish
patch membership, including common polynomial roots on patch edges.
The numerical curve parameter and unique bilinear inverse are output
representatives; an exact rational isolating interval is retained.
Identically planar curves use the separate exact overlap certificate.
Polynomial remainder and interval-refinement work share the caller cap.

`ccx/_linear_exact.py` uses exact linear elimination or collinear clipping
for nonconstant degree-one positive-weight curves. This removes the
nearly-collinear endpoint case that previously spent 2,000 fallback cells
and reported an unsupported overlap span. `ccx/_curve_line_exact.py`
handles arbitrary-degree positive-weight curves against a degree-one
line: exact cross-product component GCD followed by Sturm isolation and
exact line-domain inequalities. Common multiple roots and tiny nonzero
endpoint gaps therefore have an algebraic outcome. The shared univariate
engine prepays coefficient construction and charges remainder/refinement
work against the same cap.

The shared `_root_box_certificate.py` proves **at most one** root by an
interval-Jacobian contraction. It does not infer existence from a small
floating residual. Exact source evaluation or Krawczyk inclusion now gates
general isolated-root publication in exact topology mode. For CCX's overdetermined residual,
Krawczyk existence also requires an exact common-plane projection.
Unproved search reservations are exported as unresolved boxes. Exact
affine overlap promotion uses degree-bounded exact polynomial identity
evaluation, and cannot accept a merely roundoff-sized residual.
Unsupported/ill-conditioned regions can remain explicitly partial.
These changes do not constitute a universal
exact algebraic solver for arbitrary rational patch products.

The exact reductions separately check whether their floating parameters
and XYZ can represent the certified root at the requested tolerance.
Those checks use exact source arithmetic; a root at `t=1/3` on a curve
with speed `1e16` can be mathematically certain but unrepresentable by the
floating result schema. It receives resolution partial status and exact
rational parameter data, rather than an off-locus XYZ.

## Original-source SSX certificates

Uniform-weight sources first use outward Bernstein coefficient products
for each surface normal and the four minor dot products. Exact binomial
product weights are enclosed before floating multiplication and every
accumulation rounds outward, including subnormal operations. This keeps
within-surface polynomial dependency that entrywise Jacobian intervals
lose. A strict polynomial minor can resolve the cell immediately;
otherwise the more general interval-Jacobian certificate remains available.

The interval cofactor helper differentiates the original homogeneous
residual before restriction, propagates source arithmetic error, then
bounds each three-column determinant with outward interval operations.
Bounds are in global parameter units. A fixed-sign cofactor and its
ratios can support regularity and short-arc arguments only when this
source-derived interval is strict. Restricting an already rounded
cofactor net and inflating by its own magnitude is insufficient: an
exact dyadic determinant-one shear can turn a computed minor strictly
negative over a cell containing an entire true circle. The new helper
rejects that false no-loop certificate. Per-box work is prepaid and
cached.

Exact pure-affine world-coordinate identities also supply necessary
relations between source parameters. A small Fraction graph solves each
connected relation component, detects inconsistent or fixed-value cycles,
and intersects all parameter intervals before mapping them outward.
Cofactor bounds on this contracted domain are bounds **on the zero set**,
not on the entire original product box. They may prove regularity along
all possible intersection branches without paying for unrelated parameter
combinations. An empty constraint domain is an exact exclusion; a work
shortfall preserves the original box and reports exhaustion.

The planar cut census restricts original homogeneous controls using
Fraction arithmetic, then calls the exact scalar-root engine. Its exact
polynomial remains expressed in the original varying parameter; root
intervals and paired affine inverse enclosures are mapped outward.
Reverse plane-owner cuts retain exact rational fixed-coordinate
relations. Complete face censuses are shared only when the exact affine
component parameter is identical; floating aliases of different Fraction
pins remain separate. Reuse rebinds source metadata without changing the
published floating tuple, and partial global censuses cannot leak roots
outside a smaller queried owner. This avoids treating roots of a rounded restricted isoline as
roots of the original surface near an extremum. Unsupported cuts retain
the general solver and its explicit unresolved outcomes.

The general route receives `source_residual=(net, coefficient_error)` over
the complete local three-parameter product. Rounded C/S controls provide
Newton proposals only. AABB, boundary CCX, exact planar shortcuts and
overlap-driven range deletion cannot certify this different source
polynomial. Source coefficient errors enter every hull, uniqueness,
ownership and Krawczyk test. Exact source evaluation can establish closed
boundary roots; each callback is prepaid from the raw call's allowance.
Krawczyk images supply tight existence enclosures separately from larger
at-most-one-root boxes. Published XYZ is still validated against the
original surfaces by SSX.

Top boundary curves with a constant image need a separate source fiber
path: a three-variable isolated-root search cannot resolve their free
curve parameter. The new helper proves `P=cW` with exact rational source
arithmetic and finds target boundary preimages by polynomial GCD/Sturm,
or an interior preimage by exact convex planar-chart membership. Distinct
algebraic preimages remain separate even when their displayed UV tuples
coincide. Every seed-bearing face stays explicitly unresolved; these seeds do
not prove incident branch counts or a complete target-preimage census.
If the helper finds no seed, the full original-source boundary census
still runs. A failed seed search cannot poison an otherwise empty face;
doing so caused a toroidal closure regression that the final gate caught.
The rounded legacy cone apices fail the exact constant-image identity.
They now provide explicitly numerical continuation proposals, with no
exact fiber certificate. Output chords from either fallback are bounded
against both original source surfaces before publication, and the result
remains partial. This recovers useful generator coverage without treating
independently rounded CAD data as an exact source identity.

The public `tolerance_tier=True` CSX mode remains a modeling layer: it can
merge sub-resolution isolated events. Its completion flag does not promise
an exhaustive census of every exact root. SSX uses the separate source
proof route rather than interpreting modeling contacts as exact topology.

## Reproduction and measurements

Run `examples/ssx/csx_ccx_completeness_benchmark.py`; the checked-in
`docs/ssx_completeness_benchmark_final.json` records uniform caps, parameters,
watchdogs, call counts, topology, and timing. The baseline reloads both
CSX and nested CCX source from 76735f9 while using the checkout's shared
and native support, so this is a solver-module comparison.

| Case | Baseline CSX | Current exact CSX |
| --- | --- | --- |
| Case11 A-to-B middle section, axis1 | 2,044 cells, 0.553 s | 289 cells, 0.037 s |
| Folded patch, two regular preimages | Zero roots, falsely complete | Two roots, complete, 77 cells |
| Degree10 multiple plane root | One modeling overlap | One exact parameter root, 69 cells |
| Constant nonzero plane gap | One modeling overlap, 174 cells | Empty exact intersection, 4 cells |

The recorded source-residual end-to-end case11 run completes one closed
branch in 1.870 s and 23,148 total work units. Its 190 raw CSX calls consume
3,981 units and 0.505 s, with zero nested CCX calls and zero direct
planar-cut calls. The 60,000-unit work cap is unchanged. Source regularity
and boundary-arc certificates require continuing beyond the former
tolerance-derived default depth stop; an explicitly supplied `max_depth`
remains enforced. Source hashes did not change during this measurement.
The full final run includes 33 CSX rows and 12 CCX rows without timeouts.
The generic quadratic CCX cancellation case improves from one root and a
partial result at the 10,000-unit cap to two roots and a complete result
in 1,084 units (0.287 s). Prior measurements, including partial SSX rows,
remain in `docs/ssx_completeness_benchmark.json`. Reproduce the full final
run with `--baseline-ref 76735f9 --timeout 30 --output
docs/ssx_completeness_benchmark_final.json`.
The final substrate gate records 431 passing tests in 44.90 s, covering
raw CCX/CSX, their NURBS adapters, source residual transport and shared
numeric predicates; SSX integration suites are recorded separately.

The separate `examples/ssx/source_world_mix_benchmark.py` probe records
six curved-graph circle runs in `docs/ssx_world_mix_benchmark.json`:
identity, mixed, and sheared world coordinates, each in both operand
orders. All six return one complete closed branch with no retracing,
using the same `atol=0.001` and 60,000-unit cap. Solve times range from
0.212 to 1.269 s and total work from 5,461 to 12,103 units. The worst
reference-circle coverage distance is 0.001985, below the probe's
0.004 coverage threshold. Each run records raw CSX calls separately from
nested CCX and direct source-planar cuts; the latter two are zero here.
The production source snapshot is identical to the final case11 run and
unchanged across all six measurements.

The final expanded NURBS/source-fiber gate passes 136 tests in 23.72 s,
and the full toroidal gate passes 25 tests in 116.17 s, including closure,
length, retracing, operand order, and sagitta checks through `1e-5`.
Their commands, logs, and unchanged before/after source hashes are in
`docs/superpowers/reviews/2026-09-09-nssx-torus-final.json`.
Timings are local samples, not a claim of a universal speedup or a
substitute for topology checks.

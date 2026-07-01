# Closest-Point Band Branch-and-Bound + Degenerate-Set Tracing — Design Spec

**Date:** 2026-07-02
**Status:** Implemented (`mmcore/numeric/_bez_closest_point.py`)
**Supersedes:** the valley-marching design (2026-06-28, removed) and the
all-local-minima contract of the 2026-06-25 spec (§2 there; module structure in
that spec still applies).

## 1. Contract (band semantics)

`bez_curve_closest_points`, `bez_surface_closest_points`,
`nurbs_curve_closest_points`, `nurbs_surface_closest_points` return the
**complete set of globally closest entities**: everything whose distance lies
within `[d_min, d_min + atol]`. Local minima farther than that are
intentionally not reported.

Rationale (the two applications that motivated multi-result output):

1. **Reproducibility for autonomous pipelines.** When the query point is on
   the medial axis, the closest point is a *set*. Commercial engines return an
   implementation-chance member; mmcore returns the whole set,
   deterministically.
2. **Degeneracy marking** (offsets, fitting): callers need to know whether the
   closest point is unique, and where equidistant continua lie, to split
   curves/surfaces there.

The band is used strictly for **clipping** (pruning provably-farther cells);
it never widens the result: isolated minima are exact Newton-polished
stationary points, not `atol`-basins.

## 2. Structure theorem (what a degenerate answer can be)

With `g = F/W` (`F` = squared-distance numerator net,
`W = w⊗w` the weight-square net of the SAME (bi)degree, `W > 0`), the set
`{g = d²_min}` is the zero set of the polynomial `P = F − d²_min·W ≥ 0`.
A polynomial vanishing on a set with interior vanishes identically, so per
Bézier patch the closest-point set is **exactly one of**:

* **finitely many isolated points** (generic);
* **a 1-D real algebraic curve** in the parameter domain — and since `P ≥ 0`
  touches zero there, `∇g = 0` along the whole curve: it is a stationary
  manifold with rank-1 Hessian (`λ_min = 0` tangentially, `λ_max > 0`
  transversally);
* **the whole patch** (`F ≡ d²·W`, e.g. a sphere about its center).

1-D corollary: per Bézier curve segment the equidistant set is
**all-or-nothing** (isolated points or the entire segment); partial arcs only
arise at NURBS level as unions of whole segments.

Entity taxonomy follows directly: point entities, `degenerate_curve` (traced
parametric curve, `closed` flag), `degenerate_surface` / `degenerate_segment`.

## 3. Algorithm

### 3.1 Bounds
Tight rational Bernstein bounds per cell: coefficient-wise ratios `F_i/W_i`
bound `g` (exact for constant `g`, so a sphere-about-center resolves at the
root). This subsumes control-point AABB distance bounds and needs no extra
nets — `F` and `W` subdivide alongside the stationarity nets.

### 3.2 Surface core (best-first B&B)
1. **Whole-patch certificate**: root distance-range `< atol` → return the
   `degenerate_surface` entity (boundary entities are subsumed).
2. **Boundary first**: edge isocurves (curve core) + corners, KKT-filtered —
   cheap, seeds the incumbent `best`. A whole-edge `degenerate_segment`
   (e.g. a collapsed pole edge) becomes an edge-aligned `degenerate_curve`
   candidate.
3. **Best-first loop** on a priority queue ordered by the cell's lower
   distance bound:
   * pop with `lb > best + atol` → **terminate** (everything remaining is
     farther: single break, the band prune);
   * gradient prune: either stationarity-net hull excluding 0 → skip;
   * distance-range `< atol` (uniformly in-band flat cell) → Newton polish →
     Hessian classification: `λ_min/λ_max > 1e-6` → isolated candidate,
     else → **degenerate seed**;
   * small cell (`ptol`) → Newton leaf;
   * otherwise Newton probe (improves `best` early — this makes the band
     prune bite), split the wider axis, push children (skip a child whose
     `lb` already exceeds `best + atol`).
4. **Trace degenerate seeds** (§3.3); a traced curve consumes point
   candidates and further seeds lying on it.
5. Merge/dedup, final band filter (`≤ d_min + atol`), sort ascending.

Isolated-minimum candidates must pass a **stationarity test** (residual ⊥
both tangents, `|cos| < 1e-4` — the numerically robust form of the
"projection direction ∥ normal" condition) plus Hessian positive
semi-definiteness.

### 3.3 Equidistant-curve tracer (`trace_equidistant_curve`)
Pull-Curve-style continuation on the stationary manifold `{∇g = 0}`:

* **tangent** = Hessian null eigenvector (differentiate `∇g(γ(s)) = 0` →
  `H·γ' = 0`);
* **predictor** step along the tangent; **corrector** = LM Newton on the
   2×2 stationarity system (its minimum-norm step is transverse to the null
  space — no tangential slide-back);
* terminates at the domain boundary, on closure (loop), or aborts if the
  distance drifts off the equidistant level (then the seed falls back to
  isolated handling — graceful degradation for "almost circular" geometry);
* declines seeds whose Hessian is not rank-1 degenerate.

The same machinery with a moving source `C(t)` instead of a fixed point is
the Pull Curve algorithm (future mmcore feature; this is its seed — keep the
tracer general, do not specialize it away).

**Sphere structure and exact-circle certification.** By definition the whole
closest set lies ON the sphere `Σ` of radius `d_min` about the query point,
tangentially (the surface cannot enter `Σ`). Two consequences are built in:
the band prune is exactly the `Σ`-shell intersection test, and the tracer's
drift-abort is exactly sphere membership. A third is post-processing: a
PLANAR curve on a sphere is exactly a circle (plane ∩ sphere), so traced
curves are tested for planarity (SVD) and, when planar, certified with exact
``{"circle": {center, normal, radius, arc_angle}}`` on the entity — the
common case of any surface of revolution queried on its axis (cones,
cylinders, tori), including the angular extent on partial (non-periodic)
surfaces. Non-planar spherical equidistant curves exist for rational
surfaces (e.g. tangent surfaces along stereographic images of rational plane
curves), so the general tracer remains the backbone; the certificate is an
annotation, not a replacement.

### 3.4 Curve core
Same band clipping on the 1-D subdivision tree; whole-segment certificate at
the root (all-or-nothing per §2); endpoint KKT; final band filter.

### 3.5 NURBS wrappers
Decompose → per-patch cores sharing the incumbent `best` (`upper_bound`
parameter) → map to global parameters → seam dedup for points →
**stitch** per-patch `degenerate_curve` entities whose endpoints meet
(chain; a ring across a periodic seam is detected as `closed` via 3-D
endpoint coincidence, since `u=0`/`u=1` are parametrically distant) →
merge adjacent `degenerate_segment`s (full circle about center → one
segment spanning the domain) → global band filter.

## 4. Safeguards

* `_PTOL_FLOOR = 1e-4` relative parametric-tolerance floor (leaf accuracy
  comes from Newton polish, not subdivision depth);
* `max_cells` pop cap with a warning (never silently truncates without one).

## 5. Measured results (validation)

| Case | Before (all-local-minima) | After |
|---|---|---|
| surfcp1 funnel (scale 1e5, P near axis) | 17 s, 240k cells capped | ~30 ms, ~80 pops |
| Sphere octant, P at center | pathological | 1 pop → `degenerate_surface` |
| Circular cone, P on axis | pathological | ring traced, dist exact vs analytic |
| Full-revolution cone (NURBS) | — | one stitched `closed` ring |
| Elliptical cone, P on axis | — | deterministic isolated minima |
| Plane / ruled-U / seam / rational octant | fine | identical answers |

## 6. Out of scope

* Pull Curve proper (moving source) — future feature seeded by the tracer.
* Fitting a NURBS curve through traced `uv` polylines (callers can use
  `interpolate_curve`); entities expose raw polylines.
* The legacy `closest_point.py` remains untouched for A/B comparison.

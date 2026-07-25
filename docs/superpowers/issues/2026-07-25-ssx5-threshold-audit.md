# Threshold audit — every calibrated constant in the intersection engine (2026-07-25)

Executed per the derived-envelopes kickoff
(`docs/superpowers/plans/2026-07-25-ssx5-derived-envelopes-kickoff.md`)
at branch `ssx5-invariance` HEAD `ab50537`. Audit only — no engine
changes. Method: mechanical extraction of every epsilon-family literal
in a threshold context (comparison, tol-default, margin, floor,
regularizer) across the 13 engine-tier files, then per-site dimensional
classification by reading each context. 251 candidate lines; ~15 are
docstring text (excluded); every code site below was classified.

## Taxonomy

- **A — dimensionless / parameter-space.** Angles (`sin_ang > 1e-3`),
  ratios, [0,1]-parameter comparisons and floors, unit-vector norm
  guards. Scale-invariant by construction. **~120 sites. No action.**
- **B — atol-relative.** User-tolerance multiples (`4·atol`,
  `0.01·atol`, public `atol=1e-3` defaults). Scale along with the
  input. **~25 sites. No action.**
- **C — roundoff-derived envelopes.** `K·eps·⟨operand magnitude⟩`
  (the L1 margin, the CCX contexts, `_strict_ssx_root_tol`, the
  `tiny = max(4096·eps, 1e-12)·max(1, diag)` pattern, relative rank
  bars `1e-10·σ₀`, `1e-7·scale`). Scale-invariant. **~10 sites**, of
  which **4 carry the `max(1.0, ·)` sub-unit floor caveat** (below).
- **D° — bare but benign.** Division/normalization guards (`1e-30`,
  `1e-300`, `1e-15` on unit-ish vectors), Levenberg regularizers
  (`1e-12/1e-14 · eye`), step floors relative to `h_max`. Roles that
  cannot flip a classification; only pathological at astronomically
  extreme scales. **~55 sites. Watch, don't burn.**
- **D! — bare load-bearing.** Residual or classification bars in
  geometry (xyz or mixed) units. THE defect class; every measured
  incident of 2026-07-21..25 lives here. **38 sites in 5 clusters.**

## The answer to the driving question

**38 bare load-bearing sites remain, and 24 of them are one stack.**
After the burn-down, new fixtures should only ever be needed for new
*structural* classes — the magnitude axis becomes the property sweep's
job.

## The burn-down list (D!), by cluster

### Cluster 1 — the deflation / singular-witness stack (24 sites; the bulk)

Mixed-unit residual bars on Δ = Ψ∩TΨ systems, where Ψ rows carry xyz
units and TΨ rows carry high-degree product units — the worst
dimensional inconsistency in the engine, and the measured cause of the
P1 singular-fixture flips (exact rescale flips them; P1b experiment
table).

| Site | Constant | Role |
|---|---|---|
| `_deflate.py:606` | `tol_f=1e-10`, `tol_step=1e-12` | gauss_newton_witness defaults |
| `_deflate.py:633,652` | `fnorm < 1e-8` | witness ACCEPT bars |
| `_deflate.py:900,915` | `witness_tol=1e-8`, `curve_trace_tol=1e-10` | solver-stack defaults |
| `_deflate.py:1062,1131,1179` | `fn < 1e-6` | Φ-solver accepts (largest bare bars in the engine) |
| `_deflate.py:1189,1223,1283,1335` | `1e-10/1e-12/1e-10` | hyperplane/trace solver tols |
| `_deflate.py:1411,1440` | `‖Δ(x)‖ ≶ 1e-7` | Φ-trace validity bars |
| `_bez_ssx5.py:776,1228` | `atol=1e-8` (internal defaults) | Δ-probe/enumeration bars |
| `_bez_ssx5.py:846` | `tol_f=1e-8` | `_check_tangency` witness |
| `_bez_ssx5.py:1500` | `‖gn.residual‖ > 1e-8` | continuation neighbour test |
| `_ssx5_singular.py:641,1181` | `‖F‖ < 1e-11` | Φ∩L Newton accepts |
| `_ssx5_singular.py:895` | `‖ψ‖ < 1e-10` | Ψ-residual accept |
| `_bez_ssx5.py:1372-1400` | `fnorm<1e-10/1e-8` + **bare `1e-3` physical-tangency bar** | C3 self-intersection acceptor (partially `_Tscale`-normalized — the pattern to finish; the `1e-3` literal coincides with default atol but is NOT atol) |

Derivation pattern to apply: per-row normalization by operand net
scales (the C3 acceptor's `_Tscale` division is the half-done in-house
example; the L1 margin is the finished one). Fixture recipe: rescale
the four P1b fixtures (they already flip under `×1/4..1/16`).

### Cluster 2 — Newton convergence stops in xyz units (9 sites)

`tol=1e-14` (`_ssx_correct` `:1894`, `_ssx_correct_fixed` `:2663`),
`dot(f,f) < 1e-20` Φ-corrector stops (`:3345,:3760`), `1e-24` polish
stop (`:4072` — acceptance is `0.01·atol`, so stop-only), `newton_ssx
tol=1e-12` (`:4401`), `newton_csx tol=1e-14` (`_bezier_common.py:217,
287`), `g2 < 1e-20` (`_bez_csx4.py:955`). Reachable only in a
magnitude band (~≤ 32 for 1e-14); outside it they silently degrade to
max_iter stalls — the original case-6 half-loss mechanism. The P1
frame currently shields out-of-window inputs; in-window inputs near
magnitude 32 sit close to the 1e-14 floor. Derivation: stop at
`K·eps·⟨local point magnitude⟩` (one shared helper).

### Cluster 3 — the GJK primitive (1 site, soundness at any scale)

`cygjk.pyx:18` `tol=1e-6` default; `_gjk.cpp`: `tol` compared against
**dot products** (`dot(ab,ao) > tol` — length², `len > tol` — length:
dimensionally inconsistent), and **iteration exhaustion returns
"separated"** (`return false; // ran out of iterations`). Measured:
20000/20000 exact-contact hull pairs and 5000/5000 sub-atol-gap pairs
report "separated". The engine-side crossing-guard (`1fe0a1a`)
neutralizes it only for crossing-bearing cells; crossing-less
probe/tangency cells still consume the verdict (P1c inventory §3).
Cython/C++ work — own package. Note: callers pass `atol` as `tol`, so
the call-site scaling is fine; the INTERNAL usage is the defect.

### Cluster 4 — the CSX overlap-detection chain (the k=2 fixture, mechanism unlocalized)

> **RESOLVED 2026-07-25** — `2026-07-25-ssx5-cluster4-centering-cancellation.md`.
> None of the candidates below was the cause.  The defect is the
> common-origin CENTERING (scale-only reframing is exactly covariant):
> normalizing before translating turns an identically-zero coordinate into
> cancellation noise, which four predicates then measured relative to the
> already-cancelled value.  Note for the remaining clusters: this class is
> invisible to a literal-enumeration audit — the constants were all fine,
> the *operand* was wrong.  Ask what each bound is relative TO.

The user's boundary-coincidence fixture: world frame → 1 span
`'exact'` @1e-14; k=2 canonical frame (bit-exact transform) → 0
overlaps + 3,989 isolated pseudo-roots. The collapse mechanism is NOT
yet localized to a constant — first burn-down task, fixture in the
kickoff. Audited candidates in the chain: `identity_bound =
min(atol, 2e-10·max(1.0, component_scale))` (`_bez_csx4.py:1014` —
derived via the strict common-origin context but floor-caveated, and
its effective ratio to atol shifts under CENTERED reframes because
component scales don't transform like atol); the valley-confirmation
stepping (`2·ptol_t` coupling); the June-noted eff_atol-floor stall
acceptance in phase 2. Localize by binary-searching the frame (k=2
c=0 vs c only vs both) against per-stage CSX internals before touching
anything.

### Cluster 5 — assembly closure bar (1 site)

`_bez_ssx5.py:5457`: `‖xyz₀ − xyz₋₁‖ ≤ 1e-9` "already exactly closed"
short-circuit — bare xyz. Mis-skips only at extreme scales (closed
loops at magnitude ≫1e3 have roundoff gaps >1e-9 → harmlessly
reprocessed; sub-1e-6-scale models could wrongly treat open micro-gaps
as closed). Lowest severity; fix with the Cluster-2 helper.

## The `max(1.0, ·)` floor caveat (4 C-sites — the k<1 direction)

`_strict_ssx_root_tol` (`scale = max(1.0, diag)`),
`tiny = …·max(1.0, diag)` (`_bez_csx4.py:530`, `_bez_ccx4.py:451`),
`identity_bound`'s `max(1.0, component_scale)` (`_bez_csx4.py:1014`).
Each is a correct derived envelope for operands ≥ 1 that silently goes
ABSOLUTE for sub-unit operands: for a model of extent 1e-3, the
"derived" envelope is 1000× too loose relative to the geometry. Today
the identity window's lower bound (2⁻⁵) plus the frame's scale-UP of
tiny models shields the engine paths; the floors are still wrong as
written and the property sweep's small-scale cells will reach them
when the frame is retired. Burn down WITH cluster 2 (same helper).

## Notable good citizens (patterns to copy)

- L1 hull margin `K·eps·max|c|` + its documented safe-direction note
  (`_ssx5_singular.py:35-79`).
- CCX common-origin exactness stack (`_bez_ccx4.py:100-206`) — CCX
  audits nearly clean: its only D! members are the shared `newton_csx`
  stops (cluster 2).
- The C3 acceptor's `_Tscale` row normalization — half of exactly the
  right idea, in the middle of cluster 1.
- Relative rank bars with explicit empirical-factor comments and
  "never alone deletes output" role guards (`_bez_ssx5.py:1455-1465`).
- `_bern_zero_1d.py` — entirely parameter-space; clean.

## Proposed burn-down order (fixture-first, one tier per commit)

1. **Cluster 4** — has the reaching fixture; localize the k=2
   collapse, derive the responsible envelope. Unblocks the user case
   and the ssx5-invariance merge.
2. **Cluster 2 (+ the floor caveats)** — one shared derived-stop
   helper, ~13 mechanical call-site conversions; enables shrinking or
   removing the frame's protective role for correctors.
3. **Cluster 1** — the big one; per-row normalization of the Δ stack;
   its acceptance gate is adding the singular classes to the
   similarity sweep.
4. **Cluster 3 (GJK)** — Cython package: dimensionally consistent
   internal margins + "unknown ⇒ not separated" exhaustion semantics.
5. **Cluster 5** — folds into 2.

Only after 1–4: the frame-retirement decision (kickoff §4).

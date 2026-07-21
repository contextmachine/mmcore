# P1b: the singular tier is not scale-invariant (filed 2026-07-21)

Discovered while wiring the P1 canonical-frame preamble
(spec: `docs/superpowers/specs/2026-07-21-ssx5-invariance-normalization-design.md`,
Amendment section). The P1 fix ships with an identity window
`[2⁻⁵, 2⁵]` on joint coordinate magnitude precisely because of this
issue; P1b is the work that would let the window widen to "always".

## The defect

The singular/tangency classification tier of `bez_ssx` changes its
answers under exact uniform rescaling of the input. This is latent today
(reachable by authoring the same model at a different size), and the P1
frame made it reproducible on 4 fixtures of `tests/test_bez_ssx5_singular.py`.

## Evidence (measured 2026-07-21, working tree at 408d9b3 + P1 wiring)

Variant experiment — each transform half applied alone via monkeypatched
`_ssx_normalization_context` (script pattern:
`norm_variant_experiment.py`, scratchpad; identity = control):

| fixture (native magnitude) | identity | scale-only (exact) | center-only | full |
|---|---|---|---|---|
| closed_tangent_loop (~6, k=8, c=[.5,.5,.5625]) | PASS | PASS | PASS | FAIL (endgap 0.5) |
| cusp_curve_on_split_plane (~3, k=4, c=[1/3,0,.5]) | PASS | PASS | **FAIL** | FAIL (half t-span) |
| tangent_curve_no_point_flood (~12.6, k=16, c=[.5,.5,6.15]) | PASS | **FAIL** | PASS | FAIL (ring→'transversal') |
| positive_dim_sigma (~2, k=4, c=[.5,.5,0]) | PASS | **FAIL** | PASS | FAIL (reason→'parameter_fiber') |

Scale-only is mantissa-exact (power-of-2) — zero rounding — so its two
failures are pure threshold-magnitude effects. Center-only introduces
~1-ulp per-coefficient rounding — its failure is exact-degenerate-
structure destruction (the s=0.5 cusp line requires an exact
derivative-zero relation; c=1/3 breaks it).

Full-suite survey (identity frame): 115 passed; 77 bez_ssx calls;
native magnitudes min=1 p50=3 p90=85 p99=306 max=362; the 17 calls
above magnitude 32 ALSO pass under the full frame (implementer's
4-failed/111-passed run) — the sensitivity is fixture-specific
knife-edges, not a hard window.

## Threshold-site inventory (candidates, verified at 408d9b3)

- `_bez_ssx5.py:846` — `gauss_newton_witness(sys, Bf, tol_f=1e-8,
  max_iter=8)` in `_check_tangency` (absolute residual on the deflated
  system; Ψ scales ~length, TΨ components scale as higher powers).
- `_bez_ssx5.py:880`, `:1239` — `tol_f=1e-10` witness accepts
  (docstring-pinned contracts).
- `_ssx5_singular.py:641` — `np.linalg.norm(F) < 1e-11` accept in the
  Φ∩L Newton.
- `_ssx5_singular.py:647` — `1e-12` Levenberg regularizer (mild but
  magnitude-relative in effect).
- NOT the L1 hull margin (`K·eps·max|c|`, `_ssx5_singular.py:35-79`) —
  correctly relative, scale-invariant.
- Centering half: any future non-identity centering of near-origin exact
  structure needs an exactness story (quantized `c_q = k·round(c/k)` was
  designed but deliberately NOT shipped in P1 — YAGNI while the window
  keeps such models on the identity frame).

## Direction (per the house invariant: never fix by loosening)

Scale-aware envelopes derived from net magnitudes (the L1-margin
pattern: `K·eps·max|coeffs|`, or per-system magnitude products), NOT
bigger constants. Each site needs its own derivation of how its residual
scales with coordinate magnitude (Ψ ~ L, Σ ~ L², TΨ minors ~ L^p).

## Requirements before fixing

- Fixtures FIRST: reproduce each flip by rescaling the native fixture
  (no P1 frame involved) — e.g. tangent_ring × 1/16, positive_dim × 1/4;
  a fix without a reaching fixture is unverifiable (L50 lesson).
- The 115-suite must stay green at native scale bit-for-bit.
- On completion: widen `_NORM_IDENTITY_WINDOW` (or remove it) and add
  singular pairs to the P1 invariance property test — that is the
  acceptance test for P1b.

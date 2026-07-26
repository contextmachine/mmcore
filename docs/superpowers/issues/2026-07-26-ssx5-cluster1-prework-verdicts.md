# Cluster 1 pre-work — "relative to WHAT?" per subfamily (2026-07-26)

Pre-work ONLY, per the session mandate: no engine edits. Applies the
two-axis invariant (audit Amendment 2026-07-26) to the 24-site
deflation/singular stack, anchored on the four P1b fixtures.

Method: (1) reachability census — wrap every bearing entry point and run
each anchor, recording call counts; (2) operand identification — read what
each bar's residual is actually composed of; (3) rescale response — drive
the anchors through mantissa-exact `k` and watch each row family move.

## The operand, stated once

`DeflatedSystem.delta_point(x)` returns a **7-vector**:

| rows | content | units |
|---|---|---|
| 0–2 | Ψ = S₁(s,t) − S₂(u,v) | **length** (k¹) |
| 3–6 | TΨ — evaluations of the minor/tangency product nets | **product** (measured k³) |

Every cluster-1 bar compares `‖f‖₂` **over all seven rows** against one
absolute number. The norm is dominated by whichever family is larger, and
the two families do not transform alike.

### Measured (this is the whole cluster in one table)

`closed_tangent_loop`, driven through mantissa-exact scale (native
magnitude ~6, so k=4 stays INSIDE the identity window and the P1 frame
does not intervene):

| k | \|Ψ\|max | ×  | \|TΨ\|max | × | T-net scale | × | `1e-8` bar, relative to TΨ |
|---|---|---|---|---|---|---|---|
| 1 | 7.50e-01 | 1.0 | 3.99e+00 | 1.0 | 5.60e+01 | 1.0 | 2.51e-09 |
| 4 | 3.00e+00 | **4.0** | 2.56e+02 | **64.2** | 3.58e+03 | **64.0** | **3.91e-11** |

**Ψ moves as k, TΨ moves as k³.** A bit-exact ×4 therefore tightens the
effective acceptance on the TΨ rows by 64× while loosening nothing — which
is the P1b flip mechanism, measured, and attributed to the OPERAND rather
than to the size of any constant.

Above the window (k=16, 64 in the same run) the numbers stop moving: the
canonical frame pulls the model back into the band. **So cluster 1's
exposure is INSIDE the identity window — precisely where the frame offers
no protection.** This is the mirror image of cluster 2, which is shielded
inside and exposed outside. The two clusters must not be scheduled with
the same reasoning, and cluster 1 is NOT a frame-retirement prerequisite:
it is live today.

## Reachability census — which anchors reach which subfamily

| subfamily (sites) | closed_tangent_loop | cusp_on_split_plane | positive_dim_sigma | tangent_ring |
|---|---|---|---|---|
| `_bez_ssx5._delta_float_gn` (:1228, :1239 `tol_f=1e-10`) | 4 | **4070** | 278 | 36 |
| `_bez_ssx5._check_tangency` (:776, :846 `tol_f=1e-8`) | 107 | — | — | 215 |
| `_deflate.gauss_newton_witness` (:606, :633, :652) | 107 | — | — | 215 |
| `_ssx5_singular.c1_pass` (:895 `‖ψ‖<1e-10`) | 1 | 1 | 1 | 1 |
| `_deflate` Φ-solver accepts (:1062, :1131, :1179 `fn<1e-6`) | — | — | — | — |
| `_deflate` hyperplane/trace tols (:1189, :1223, :1283, :1335) | — | — | — | — |
| `_deflate` Φ-trace validity (:1411, :1440 `‖Δ‖≶1e-7`) | — | — | — | — |

## Verdict table

| # | Subfamily | Relative to WHAT today? | Should be relative to | Reaching fixture | Verdict |
|---|---|---|---|---|---|
| A | `gauss_newton_witness` accept bars — `_deflate:606/633/652`, defaults `:900/915`; `_bez_ssx5:846/880/1239/1500` | **nothing** — absolute, on a norm mixing k¹ and k³ rows | per-ROW: Ψ rows to ⟨xyz magnitude⟩, TΨ rows to each T-net's own coefficient scale | **YES** — `closed_tangent_loop` ×4 (in-window, 64× effective shift measured) | **CONFIRMED operand defect. Convert first.** Highest call volume and the only subfamily with a measured, in-window reaching fixture. |
| B | Δ-probe / enumeration `atol=1e-8` — `_bez_ssx5:776`, `:1228` | absolute, same mixed norm | same per-row normalization | **YES** — all four anchors reach `_delta_float_gn`; `cusp_on_split_plane` drives it 4070× | **CONFIRMED reachable; operand defect by inspection.** Convert with A — they share `delta_point`. |
| C | Ψ-residual accept `‖ψ‖<1e-10` — `_ssx5_singular:895` (`c1_pass`) | absolute, but on **Ψ ONLY** (rows 0–2) | ⟨xyz magnitude⟩ — a single clean length scale | YES (all anchors, 1 call each) | **Genuine but EASY** — pure length units, no mixing. This is the one site in cluster 1 that the cluster-2 helper shape actually fits. Low risk. |
| D | Φ∩L Newton accepts `‖F‖<1e-11` — `_ssx5_singular:641`, `:1181` | absolute; F is the Φ∩L system | Φ row scales (hyperplane rows are dimensionless, Ψ rows are length — mixed again) | not isolated by the anchors (nested inside `newton_factory`/`newton6`) | **SUSPECT — needs its own fixture.** Do not convert blind. |
| E | Φ-solver accepts `fn<1e-6` — `_deflate:1062/1131/1179` | absolute, mixed norm; **the largest bare bars in the engine** | per-row as A | **NONE of the four anchors reach it** | **UNPROVEN. Must not be converted this campaign** without a fixture that executes it. Being the biggest constant is not evidence. |
| F | hyperplane/trace solver tols — `_deflate:1189/1223/1283/1335` | absolute | mixed: γ-row scales vs Ψ | **NONE** | **UNPROVEN — and possibly dead code on the SSX path.** Establish reachability before anything else; if unreachable from bez_ssx, they leave cluster 1 entirely. |
| G | Φ-trace validity `‖Δ‖≶1e-7` — `_deflate:1411/1440` | absolute, mixed norm | per-row as A | **NONE** | **UNPROVEN.** Same treatment as F. |
| H | C3 self-intersection acceptor — `_bez_ssx5:1372-1400` (`fnorm<1e-10/1e-8` + bare `1e-3`) | **partly** — already divides by `_Tscale` | finish the `_Tscale` pattern; the `1e-3` is a physical-tangency bar coinciding with default atol but NOT atol | not measured this pass | **Half-done in-house example.** Finishing it is the cheapest demonstration of the target pattern. |

## Consequences for the campaign

1. **The 24 is really ~10 with evidence and ~14 without.** Subfamilies E,
   F, G — 9 of the 24 sites — are reached by NO anchor. Under the amended
   audit rule (a D! classification is a hypothesis) they cannot be
   converted this campaign. First task is a reachability sweep of the whole
   singular suite, not a rewrite.
2. **Start with A+B.** They share `delta_point`, carry essentially all the
   call volume, and have a measured in-window reaching fixture. One helper
   (per-row scales from the T-nets, which are available on the system
   object as `T_point_nets`) serves both.
3. **C is a freebie** — pure length units; it is the only cluster-1 site
   the cluster-2 style helper fits unchanged.
4. **Before any of it, answer the conditioning question** handed over from
   cluster 2 (audit, cluster-1 blockquote): on an isolated tangency the
   corrector cannot reach ANY of these bars — it stalls at 4.5e-11 (mag 1)
   to 2.7e-7 (mag 32). If the iteration cannot reach a bar, renormalizing
   the bar changes nothing. For each of A, B, C, measure the residual the
   solver actually delivers on the anchors BEFORE choosing the new
   envelope, or the campaign repeats cluster 2's mistake at 24× the size.
5. **Acceptance gate stands** (kickoff §3): singular classes enter the
   similarity sweep as their tiers get derived envelopes. Note the sweep
   must include an IN-WINDOW rescale cell (e.g. ×4 on a magnitude-6
   fixture) — the existing TRANSFORMS all leave the window, and this
   cluster's defect lives inside it.

# Cluster 3 (GJK) — measured: a real latent defect, NOT live (2026-07-26)

Investigation of the GJK primitive, prioritised first on the reasoning that
it is a leaf module and an unsound "separated" verdict could be deleting
features unnoticed across many callers. The reasoning was right; the
audit's premise it rested on was not. **No fix applied** — fixture-first
(L50) and the amended audit rule (a D! classification is a hypothesis until
a fixture reaches it).

Supersedes the scope in `2026-07-26-gjk-soundness-scope.md`, which was
written from the audit's claims rather than from measurement.

## The audit's claims, re-measured from scratch

| audit claim | measured |
|---|---|
| exact-contact hulls report "separated" — 20000/20000 | **0/2000** face contact, 1/2000 vertex contact |
| overlapping hulls lost to iteration exhaustion | **0/300** at every max_iter; 0/200 down to overlap depth 1e-9 |
| verdict not scale/translation invariant | **invariant** — correct at \|T\| ≤ 1e9 × scales 1e-3…1e3 |
| `tol` gives no proximity margin | **confirmed 200/200** — but see "why it does not matter" |

The first three do not reproduce. GJK is not the broken-everywhere
primitive the audit described.

## The real defect (unit level, reproducible)

Two conditions produce a genuinely WRONG "separated" on hulls that
provably overlap:

1. **Starved iterations.** `max_iter` 1 or 2 → **200/200** false separated
   on boxes overlapping by half their extent. This is `_gjk.cpp:272`
   (`return false;  // ran out of iterations`) reporting *unknown* as a
   definite negative.
2. **Geometry comparable to `tol`.** With `tol=atol=1e-3`, deeply
   overlapping boxes of extent `s`:

   | s | s/tol | verdict |
   |---|---|---|
   | 1e-1 | 100 | True (correct) |
   | 1e-2 | 10 | **False — wrong** |
   | 1e-3 | 1 | **False — wrong** |
   | 1e-6 | 0.001 | **False — wrong** |

   The threshold is roughly `extent ≲ 10·tol`. This is the dimensional
   defect the audit pointed at (`dot(ab,ao) > tol` on length² vs
   `len > tol` on length, `_gjk.cpp:79/105/121/129` vs `:255`) — real, and
   scale-dependent exactly as predicted.

## Why it does not currently matter

**The convex-hull property does the heavy lifting.** A surface lies inside
its control hull, so hull-disjoint ⇒ surface-disjoint. A *correct*
"separated" therefore can never delete an exact intersection; only a
*wrong* one can. So the only dangerous direction is the false-negative
above.

Audited every live prune the engine actually performs: instrumented
`bez_ssx` across 26 singular-suite tests, captured all 79 verdicts where
`gjk` returned "separated", and checked each against an exact LP
feasibility test (`0 ∈ conv(A) − conv(B)`, scipy/HiGHS).

> **UNSOUND prunes: 0 / 79.**

Every prune the engine performs is correct. Three independent engine-level
reaching attempts also failed to make GJK change a result:

- parallel planes at sub-atol gaps — bypassing GJK changes nothing
  (the engine reports nothing there either way);
- bowl-tangent-to-plane, gap swept 0…2e-3 — no divergence;
- the same tangency with atol swept 1e-1…1e-4 to force the
  `extent ≲ 10·tol` regime — **GJK is consulted 0 times**.

Why the primitive's defect stays off the engine's path:

- GJK is only consulted for **crossing-less** cells (the `1fe0a1a`
  crossing-guard) — 145 consultations across 26 tests;
- `_trust_gjk` rejects ~50% of candidates (5,611 of 11,317);
- the engine passes `max_iter=15`, well clear of the 1–2 starvation zone;
- and the surrounding prunes (`_aabb_disjoint(...,atol)` before,
  `_check_min_of_net(...,atol,...)` after) are atol-aware, so GJK is never
  the only thing standing between a cell and its features.

Riskiest caller is `GaussMap.intersects` (`gauss_map.py:378`), which passes
NO tolerance at all and inherits the bare `1e-6` default — but its only
live call site is `ssx/dqr4.py:119`, off the shipped path.
`_detect_intersections` and `_ssx4` are likewise not imported by the ssx5
engine.

## Verdict and recommendation

Cluster 3 is a **latent** defect: real in the primitive, unreachable in the
engine today. This is the same shape as cluster 2 — the second audit
cluster whose premise a fixture disproved. Under the amended audit rule it
must not be "fixed" on unit-level evidence alone.

Proportionate action when it is scheduled (small, ~half a session):

1. `_gjk.cpp:272` exhaustion → return "not separated" (conservative). Costs
   work, never features.
2. Make the two internal margins scale-relative and dimensionally distinct
   — the `extent ≲ 10·tol` cliff disappears.
3. Give `GaussMap.intersects` a derived tolerance instead of the default.
4. **Regression pin: the 79 captured live verdicts must all stay
   "separated"** — a hardened primitive that turns them into "not
   separated" trades a latent bug for a real performance collapse.

## Prerequisite discovered the hard way

`python build.py` rebuilds all 24 Cython extensions, after which **every
one of them SIGKILLs the interpreter on import**. macOS crash report:
`CODESIGNING` + `EXC_BAD_ACCESS` — the kernel refuses freshly-linked
bundles. `codesign -v` passes, so it is invisible to the obvious check.

    codesign --force --sign - <each rebuilt .so>

This is mandatory in the loop for ANY Cython work in this repo, and the
`.so` files are untracked, so a bad rebuild is not recoverable with git.

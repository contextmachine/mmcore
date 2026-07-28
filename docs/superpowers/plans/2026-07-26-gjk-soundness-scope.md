# Cluster 3 — GJK primitive soundness: scope for a dedicated session

Scoped 2026-07-26 at user request ("scope the Cython fix properly").
No code written. Sources: `mmcore/numeric/algorithms/cygjk.pyx`,
`mmcore/numeric/algorithms/_gjk.cpp`. Audit entry: cluster 3 of
`docs/superpowers/issues/2026-07-25-ssx5-threshold-audit.md`.

Read the audit's amended two-axis rule first: for each margin below, the
question is *relative to WHAT*, and a D! classification is a hypothesis
until a fixture reaches it.

## Defect inventory (verified at source)

| # | Site | Defect |
|---|---|---|
| 1 | `_gjk.cpp:79, 105, 121, 129` | `dot(ab, ao) > tol` — a **length²** quantity compared against `tol` |
| 2 | `_gjk.cpp:255` | `len > tol` — a **length** compared against the SAME `tol`. One symbol, two dimensions: no single value can be right for both, at any scale. |
| 3 | `_gjk.cpp:272` | `return false;  // ran out of iterations` — *unknown* reported as the definite negative "separated" |
| 4 | `cygjk.pyx:18` | `double tol=1e-6` default — a bare absolute length |

Measured consequence (audit, not re-measured this session): exact-contact
hull pairs report "separated" **20000/20000**; sub-atol-gap pairs
**5000/5000**. The primitive is unsound at ANY scale, not just extreme
ones — #1/#2 guarantee that, because no `tol` satisfies both comparisons.

Note `_gjk.cpp:265` (`support point failed to pass origin`) is a
LEGITIMATE separating-axis conclusion and must stay `false`. Only #3 is
the unsound return.

## Two corrections to the audit's framing (found while scoping)

**(a) "Callers pass `atol` as `tol`, so the call-site scaling is fine" is
not true in general.** Census:

| caller | tol passed | max_iter |
|---|---|---|
| `_bez_ssx5.py:6599` (SHIPPED path) | `atol` | **15** |
| `_bez_ssx6.py:3453` | `atol` | 15 |
| `_detect_intersections.py:176, 384` | **hardcoded 1e-8** | 150 / 50 |
| `_ssx4.py:798, 1684` | `gjk_tol` (default 1e-8) | param |
| `gauss_map.py:378` `GaussMap.intersects` | **none — the 1e-6 default** | 25 |

So three of five consumers feed an absolute constant, and
`GaussMap.intersects` inherits the bare default with no scaling at all.
Fixing only the internals leaves those call sites bare; the tier has to
cover both ends.

**(b) The shipped path uses `max_iter=15`**, which makes defect #3
materially reachable rather than theoretical — GJK on near-degenerate
hulls routinely needs more. Any fix that keeps a low cap MUST return
"unknown", not "separated".

## The work

1. **Separate the two margins by dimension.** `handleSimplex`'s tests are
   sign tests on dot products; the correct comparison is against a
   *relative* bound derived from the operand magnitudes
   (`|ab|·|ao|·K·eps`), not against a caller's length tolerance. The
   `len > tol` progress test is a genuine length and belongs against a
   length-scaled bound derived from the hull extent. Two different
   quantities, two different derivations — do not unify them.
2. **Exhaustion returns UNKNOWN.** Change the C++ signature to a tri-state
   (or an out-param `bool& conclusive`). `false` must mean "proved
   separated by a witness axis"; exhaustion and non-progress mean
   "unknown". Then every caller's `if not gjk(...)` — currently "prune
   this cell" — must treat unknown as **do not prune**.
3. **Scale the call sites.** Give `gjk` a required tolerance derived from
   the operand hulls, and fix `GaussMap.intersects` to pass one.
4. **Retire the engine-side workaround if it becomes redundant.** The
   crossing-guard (`1fe0a1a`) exists to neutralize this for
   crossing-bearing cells; P1c inventory §3 notes crossing-less
   probe/tangency cells still consume the verdict. Decide with the user
   whether the guard stays as defence in depth.

## Fixture strategy (fixture-first, L50)

Unit-level is not enough — the audit's 20000/20000 is already a unit
measurement and it did not drive a fix. Needed:

- **Reaching fixture through the engine**: a crossing-less tangency or
  probe cell where the wrong "separated" verdict deletes a feature from
  `bez_ssx` output. Without it, #3 stays a hypothesis (amended audit rule).
- **Unit property tests** in the same commit: exact contact ⇒ not
  separated; gap > derived bound ⇒ separated; verdict invariant under
  similarity (translate 1e-9..1e9, scale 2^±k) — the sweep that caught
  cluster 4.
- **Anti-loosening guard**: genuinely separated hulls at every world
  position must still prune, or the tier trades a soundness bug for a
  performance collapse.

## Build loop and risk

- `build.py:143-144` builds `cygjk` from `cygjk.pyx` + `_gjk.cpp`; the
  session needs `python build.py` in the loop, and the repo's tracked
  generated sources are inconsistent (`_nurbs.cpp` tracked, `nurbs.cpp`
  not) — settle what gets committed BEFORE the first rebuild, or a stray
  `git add -A` will sweep ~1M lines of artifacts (it did on 2026-07-26).
- **Direction of risk is favourable but not free**: "unknown ⇒ do not
  prune" can only ADD work, never delete solutions. Expect the bez and
  NURBS harness work counts to rise; the ×1.00 baseline gate will trip by
  design, so re-baseline deliberately and record the new numbers rather
  than relaxing the gate.
- Blast radius includes `_ssx4` and `gauss_map`, which are outside the
  ssx5 floors. Add coverage for them or scope them out explicitly.

## Acceptance

Reaching fixture green; the three unit property classes above; full floors
(220 SSX-tier / 187 ccx-csx-budget / both harnesses) with work counts
re-baselined and the deltas explained; similarity sweep extended with the
GJK-dependent class. Adversarial review after implementation, with an
operand-correctness lens on both new margins.

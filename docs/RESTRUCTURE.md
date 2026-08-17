# mmcore — Target Package Structure (decided plan)

**Status:** EXECUTED — all phases complete on branch `restructure-phase2`
(2026-08-16): Phase 1 at `aa818c2` on tiny; Step 0 + batches 2a–2e + Steps 13–14 as
the `restructure(*)` commit series. Import surface clean (6 benign extras leaves
only), layering violations 15 → 0, `geom/` → `nurbs/` with `_core.pyx`, pickles
migrated losslessly. This document is now the historical record of the migration;
the living conventions moved to `docs/ARCHITECTURE.md`.
**Baseline:** branch `tiny` @ `f6b3655`, measured 2026-08-13; Phase 1 delta re-measured
2026-08-14 (§7.0); Phase 2 preflight re-pinned at `aa818c2` with zero drift, 2026-08-16.
**Execution companion:** `docs/superpowers/plans/2026-08-16-restructure-phase2-execution.md`
— NEW-Q decisions with evidence, branch/CI audit, batch order. Corrections from that
preflight are folded in below, flagged **PREFLIGHT 2026-08-16**.

This document exists because the baseline is reached — all fundamental algorithms are
present, the named-tuple NURBS core is readable and debuggable, construction functions
and BRep topology support rendering and STEP export — and the next gain is no longer a
new algorithm. It is removing what is no longer carrying weight, and making the shape of
what remains obvious from the directory listing.

> **Three measurements in this pass overturned a claim the design itself made.** They are
> flagged **CORRECTION** where they appear, because each would have caused wrong work:
> `_detect_intersections.py` is **not** broken (§7.5); `cbern.pyx` is **slower** than the
> pure-Python code on the real hot path despite being 12x faster in isolation (§6.3); and
> `test_nurbs_compose.py` **cannot** be re-pointed at `sbern.py` (§7.1).
>
> **PREFLIGHT 2026-08-16: the third correction was itself wrong.** The re-point IS
> possible — 3 of the suite's tests already target sbern and all 12 run green against it
> (§7.1, rewritten). The other two corrections stand.

---

## 0. Ground rules for reading this

**Licence.** mmcore has never claimed backward compatibility and nothing outside it
imports it. Anything here may be renamed, moved, merged or deleted. **No compatibility
shim, deprecation alias, or `compat` re-export layer is proposed anywhere in this
document**, by explicit owner instruction. A "keep it for back-compat" argument is
inadmissible.

**Verdict vocabulary.** Every file gets exactly one:

| Verdict | Meaning |
|---|---|
| **KEEP** | Live: has a consumer, or is a designed entry point |
| **MERGE-INTO-`x`** | Content belongs elsewhere; target named |
| **DELETE** | Proven dead against all six evidence channels below |
| **QUARANTINE** | Evidence cannot decide. An owner question is stated. Never a guessed delete |

**The DELETE predicate.** A file is DELETE only when **all six** hold:

1. zero Python importers in `mmcore/`
2. zero importers in `tests/` and `examples/`
3. zero Cython `cimport` consumers
4. not built by `build.py`
5. no dynamic reference (monkey-patch, `setattr`, string import)
6. not a public entry point

Channel 3 and channel 5 are not decoration. During this analysis a static Python import
graph marked `geom/parametric`, `geom/knots.pxd`, `numeric/algorithms/quicksort` and
`numeric/binom.pxd` as zero-importer — all four are the C base layer of the NURBS
hierarchy, reached only by `cimport`. Deleting on the graph alone would have removed
the foundation of `nurbs.pyx`. Likewise `examples/ssx/bez_ssx5_diag.py:23,57` and
`examples/ssx/bez_ssx5_cert_trace.py:117,123,182` rebind `_bez_ssx5._deflate_tangent_cell`
at runtime; a rename breaks them silently and invisibly to any import graph.

**Zero importers is a CANDIDATE signal, never a verdict.**

**Evidence precedence.** Three sources disagree, and they rank:

1. **The code and git history at HEAD** — authoritative on what exists and when it changed.
2. **`docs/superpowers/{issues,plans,specs,designs}`** — 34 tracked documents, the
   maintainer's own issue record. Authoritative on intent. **Newer and more decisive
   than the external notes.**
3. **`/Users/sthv/mmcore-claude-memory/`** (23 files) — external session notes, untracked.
   Authoritative on what a session decided *at the time*; demonstrably stale on merge
   state and on at least two "still owed" items that have since shipped.

Where 2 and 3 conflict, 2 wins. Where 1 and 2 conflict, 1 wins — this is not theoretical:
`issues/2026-07-26-gjk-measured-latent-not-live.md` says "No fix applied", but
`_gjk.cpp:60-90` now contains `dotSignMargin()` with `tol` explicitly retired. The doc is
stale; the code is the truth.

---

## 1. Measured baseline

Everything in this section was measured on 2026-08-13 against `f6b3655`, not estimated.

> **This section is the FROZEN BASELINE and is deliberately not updated.** Every step in §7
> is verified as a delta against exactly these numbers, so rewriting them would destroy the
> yardstick. For the current state after Phase 1, see **§7.0**. Where a fact below has since
> changed, the change is noted inline and the number is left alone.

### Import health — `.venv/bin/python` 3.14.6

**170 OK / 17 FAIL of 187 importable modules.** The 17 split into two very different groups.

*Benign (6) — optional-dependency leaves. This is correct behaviour, not debt:*

| Module | Missing dependency |
|---|---|
| `extras/occ/geom_int.py` | `OCC` |
| `extras/renderer/renderer2d.py` | `plotly` |
| `extras/renderer/renderer3d.py` | `glfw` |
| `extras/rhino/__init__.py` | `rhino3dm` |
| `extras/torch/algorithms/implicit_point.py` | `torch` |
| `extras/torch/numeric.py` | `torch` |

*Real (11) — the actual debt:*

| Module | Failure |
|---|---|
| `numeric/algorithms/point_in_curve.py` | `No module named 'mmcore.geom.curves'` |
| `numeric/intersection/implicit_implicit.py` | `No module named 'mmcore.geom.curves'` |
| `numeric/intersection/ssx/_ssx31.py` | `No module named 'mmcore.geom.curves'` |
| `numeric/intersection/ssx/boundary_intersection.py` | `No module named 'mmcore.geom.curves'` |
| `numeric/intersection/ssx/ode.py` | `No module named 'mmcore.geom.curves'` |
| `numeric/intersection/ssx/dqr.py` | `No module named 'mmcore.geom.surfaces'` |
| `numeric/area.py` | `No module named 'notes.offset'` |
| `topo/shell_ops.py` | `No module named 'halfedge_topology'` |
| `numeric/intersection/csx/_overlaps.py` | `cannot import name 'refine_curve_surface' from '_ncsx2'` |
| `numeric/newton/constrained.py` | `IndexError: list index out of range` (at import) |
| `numeric/projection/certified_proj.py` | `No module named 'sympy'` |

Five share one cause: `c14fd3e` (2026-06-09) deleted `mmcore/geom/curves` and
`mmcore/geom/surfaces`. Three are their own species and are worth naming:

- **`numeric/area.py:6`** does `from notes.offset import evaluate_nurbs_curve`. `notes/`
  contains only markdown and images — there is no `notes/offset.py` anywhere in the repo.
  Library code importing from the notes directory is a category error even if the file
  existed.
- **`topo/shell_ops.py:60`** imports bare `halfedge_topology`, not `mmcore.topo.halfedge_topo`.
- **`numeric/newton/constrained.py`** raises `IndexError` *at import time* — module-level
  code executing on import.

### Test health — measured, full run 44:58

| Metric | Value |
|---|---|
| Collected | **771 tests** |
| Collection errors | **6** |
| `pytest tests` as-is | **aborts, runs nothing** (`Interrupted: 6 errors during collection`) |
| Full run, 6 uncollectables ignored | **11 failed · 756 passed · 1 xfailed · 3 errors** in 2698.78s |

The suite does not run. That is the single most important health fact in this document:
`pytest tests` from a clean checkout executes zero tests.

> ✅ **FIXED in Phase 1** — collection errors 6 → 0; the suite now runs end to end. See §7.0
> and §7.1. The baseline numbers above are retained as the yardstick.

The 6 collection errors implicate a **fifth missing library file** nobody had listed:
`tests/test_nurbs_compose.py:9` imports `mmcore/geom/_nurbs_compose.py`, which does not
exist on disk and does not appear in `c14fd3e`'s deletions. Three further **runtime**
errors in `test_nccx4.py` (setup-time, same `geom.curves` cause) are invisible to a
collect-only check — so the `geom.curves` deletion damaged **7 test files, not 4**.

> Follow-up: `_nurbs_compose.py` was **never tracked in git at all** (`git log
> --diff-filter=D` finds no deletion commit), which is why it appears in no commit's
> deletions. Its content is therefore unrecoverable from history. See §7.1.

Two triaged reds:

- `test_ssx5_c1_regular_normal.py::test_high_order_tangent_never_publishes_off_locus_branches[10,12]`
  — **the notes are wrong; it PASSES.** All three params green in 3.01s, host-attested.
  Budget no repair for it.
- The ~10 `surface_surface.py` failures — **confirmed, exactly 10 failed / 2 passed**, and
  they are *two* causes, not one: six count assertions where the contract moved
  (`spt` deprecated, `tol` now ignored), and **four hard `AttributeError` at
  `mmcore/geom/_nurbs_knots.py:2043`: `'NURBSCurve' object has no attribute 'knot'`**.
  That last one is a live typo bug (`.knot` → `.knots`) in a current 2671-line L0 module.
  It will outlive any decision about `surface_surface.py` and should be fixed on its own.
  ✅ **FIXED in Phase 1** — and it was not a rename: both curve representations reach that
  module and spell the field differently. See §7.2.

A third red nobody predicted: `test_newton.py::test_newthon` — `IndexError: list index
out of range`, the only coverage of `cnewton.pyx` + `fdm`.

**Cost outlier:** `test_point_in_region.py::test_point_whose_segment_is_tangent_to_circle_is_outside`
alone takes **1745s — 65% of the entire suite** — and passes while warning "containment
may be unreliable". Marking it slow takes the suite from ~45 min to ~16 min.

### Import graph

AST-derived over all 317 tracked `.py` files → 218 `mmcore` modules.
**51 have zero importers and zero cimport consumers** (the true DELETE candidate pool).
7 more have test/example importers but no library consumer.

### Native surface

`build.py` declares **28 `Extension(` entries: 24 active — 23 Cython (`cython_extensions`,
line 62) plus 1 pure-C (`native_extensions`, line 273, `ndinterval`) — and 4 commented out**
(lines 127, 135, 187, 223). Compiled artifacts: 24 module families × 3 ABIs
(cp39, cp313, cp314). **No orphan `.so`** — nothing was lost by a past deletion.

---

## 2. The two facts that mislead every newcomer

These belong at the top of any onboarding document, because both are cases where the
obvious reading is the wrong one.

**1. In `ssx`, a higher filename number means OLDER lineage.**
`_bez_ssx6.py` (3790 L) is an 80 %-similarity *copy* (git `C080`) of `_bez_ssx5.py`,
taken 2026-05-16 and frozen by ledger L53 on 2026-07-12. `_bez_ssx5.py` (8225 L, added
2026-04-10, last worked 2026-07-28) is the maintained engine. The file says so on line 1
— but only if you open it.

**2. The package default resolves to the DEAD engine, in two families.**
`csx/__init__.py:2` binds plain `nurbs_csx` → `_ncsx.py` (last real work 2025-09-29),
while the current adapter `_ncsx4.py` is reachable only as `nurbs_csx_v4`.
`ssx/__init__.py:2` binds `nurbs_ssx` → `_ssx4.py`, while the maintained path is
`_nssx5.py`, exposed as `nurbs_ssx_v5`. In both families **the good name points at the
old code and the current code hides behind a version suffix.**

The proposal in §4 fixes this by construction: after it, there is no version-suffixed
public name left to be wrong about.

---

## 3. Target tree

The shape below is the proposal. It is not a rename-everything exercise: most paths stay
where they are. Three things change structurally — substrate moves out of `intersection/`,
each solver family gets one obvious entry point, and dead weight leaves.

```
mmcore/
├── __init__.py              version + the small public façade (see §4)
│
├── geom/                    ── L0 · NURBS core ──────────────────────────
│   ├── nurbs.pyx/.pxd       C++ NURBSCurve / NURBSSurface
│   ├── parametric.pyx/.pxd  C base classes (cimport-only — NOT dead)
│   ├── knots.pxd            header-only span/insertion inlines (cimport-only)
│   ├── primitives.pyx       analytic surfaces
│   ├── _nurbs_eval.py       NURBSCurveTuple / NURBSSurfaceTuple — the tuple ABI
│   ├── _nurbs_knots.py      decompose_curve / decompose_surface  ⚠ live bug :2043
│   ├── _nurbs_ders.py       derivatives
│   ├── _nurbs_param_tol.py  parametric tolerance
│   ├── _nurbs_interp.py     interpolation
│   ├── _nurbs_join.py       joining
│   ├── _nurbs_construct.py  construction helpers
│   ├── _nurbs_transform.py  transforms
│   ├── nurbs_iso.py         iso-curve extraction
│   ├── octree.py            spatial subdivision
│   ├── bvh/                 ⚠ THREE overlapping implementations — §7 Q6
│   └── implicit/            implicit surfaces + dual contouring
│
├── numeric/                 ── L0 · numeric substrate ───────────────────
│   ├── bern.py  sbern.py  bern_sq_dist.py     Bernstein bases + squared-distance nets
│   ├── _bern_homog.pyx  _cdecasteljau.pyx  binom.pyx  cbern? (§7 Q7)
│   ├── _work_budget.py      THE shared work budget (L52 8-way merge)
│   ├── _bezier_common.py    ← MOVED here from intersection/ (see §5)
│   ├── _bern_zero_1d.py     ← MOVED here from intersection/
│   ├── _bez_closest_point.py  closest point via squared-distance nets
│   ├── closest_point.py     legacy baseline
│   ├── newton/  interval/  integrate/  plane/  matrix/  projection/
│   ├── algorithms/          cygjk, moller, quicksort, implicit_point, adaptive_polyline
│   └── intersection/        ── L1 · solvers ─────────────────────────────
│       ├── _adapter_status.py    shared adapter status objects
│       ├── _deflate.py           deflation stack (cluster 1 — LIVE)
│       ├── _sq_dist_classify.py  _interval_cutout.py  separability/  tracing/
│       ├── ccx/            curve × curve
│       │   ├── __init__.py → ccx.py → _nccx4.py → _bez_ccx4.py
│       │   ├── _bezier_eval.pyx  ⚠ imported but NEVER BUILT (§6)
│       │   └── _bez_overlap.py  _utils.py  segment.py
│       ├── csx/            curve × surface
│       │   └── __init__.py → _ncsx4.py → _bez_csx4.py  (+ _cbez_csx.pyx)
│       └── ssx/            surface × surface
│           ├── __init__.py → _nssx5.py → _bez_ssx5.py
│           ├── _ssx5_singular.py  _ssx5_overlap.py   typed singularities C1/C2/C3
│           ├── refine.py  trace_inter_segm.py
│           └── _ssx_utils.pyx
│
├── construction/            ── L2 · modelling ───────────────────────────
│   └── _ruled _revolved _sweep _cone _cylinder _torus _curve _surface
├── topo/                    brep/ · mesh/ · halfedge_topo · curve_boolean
├── compat/step/             STEP I/O
└── extras/                  ── L3 · leaf integrations ───────────────────
    └── occ/ · rhino/ · renderer/ · torch/     optional deps, ZERO inbound edges
```

`ds/` is absent from the tree above deliberately — see §3.4, where it dissolves.

*(§3.4 comes first because it decides the tree's **shape**; §3.1–§3.3 then classify the
individual files inside it.)*

### 3.4 The owner's two structural questions

Both were asked against the tree above. Both are answered here with measured counts.

#### A — "Rename `geom/` → `nurbs/`?" — **not as a rename. Move the non-NURBS residents out first.**

The honest objection is that **`geom/` is not all NURBS**, so the rename would create false
paths rather than remove them. Measured contents of `mmcore/geom/` (43 filesystem children,
24 NURBS-core / 19 not):

| Non-NURBS resident | What it actually is | Import lines (lib / test / example) |
|---|---|---|
| `bvh/` | spatial acceleration, consumed mostly by *solvers* | 16 / 0 / 8 = **24** |
| `implicit/` | implicit surfaces + dual contouring | 13 / 1 / 5 = **19** |
| `octree.py` | spatial subdivision | 2 / 0 / 0 = **2** |
| `parametric.pyx/.pxd`, `primitives.pyx/.pxd` | the C base-class hierarchy `nurbs.pxd` cimports | 6 / 3 / 2 = **11** (`primitives`); `parametric` has 0 Python importers, cimport-only |

A literal directory rename would produce `mmcore.nurbs.bvh.lbvh` and
`mmcore.nurbs.implicit.sdf` — each a *worse* name than today's, because it asserts something
false. It would also produce the `mmcore.nurbs.nurbs` stutter.

**Blast radius, tracked files only** (`git ls-files` piped through grep; the working tree
shows more because `examples/` has untracked scratch):

| Scope | Files touching `mmcore.geom` | Lines |
|---|---:|---:|
| `mmcore/` | 91 | 277 (incl. 1 relative import, `topo/mesh/tess.py:13`) |
| `tests/` | 26 | 47 |
| `examples/` | 56 | 143 |
| **Total** | **173** | **467** |

Beyond Python source:

- **9 `cimport` lines** across 7 files (`geom/nurbs.pxd:7,8`, `geom/nurbs.pyx:4,17`,
  `geom/knots.pyx:11`, `geom/primitives.pyx:8`, `numeric/calgorithms.pyx:9`,
  `numeric/calgorithms.pxd:6`, `ssx/cydqr.pyx:10`);
- **8 active `build.py` lines** (+4 in already-dead commented blocks);
- **0** `include` directives, **0** dynamic/`importlib`/`sys.modules` references;
- ⚠ **5 tracked pickle fixtures embed the literal string `mmcore.geom._nurbs_eval`** —
  `examples/ssx/nurbs_nurbs_intersection_{5,6,8,10,11}.pkl`, loaded by
  `tests/test_nssx5.py:908,981,1268`. **A rename silently breaks unpickling** of the largest
  ssx suite's fixtures. No import graph shows this edge; it was found by grepping the binaries.

Renaming `nurbs.pyx` too (to kill the stutter) adds 63 Python path-bearing lines, 2 cimports,
2 `build.py` lines, 2 source files, 3 `.so` files and the generated C identity.

**Decision — sequenced, and only the first step is committed to:**

1. **After Phase 2's deletions** (fewer files to move), relocate the non-NURBS residents:
   `bvh/` and `octree.py` → `numeric/` (they are spatial *substrate*, and their consumers are
   already mostly `numeric/`), `implicit/` → its own package at **`mmcore/implicit/`**
   (PREFLIGHT pin: it must stay INSIDE the `mmcore` package — pyproject declares no
   packaging directives, poetry-core auto-discovers only `mmcore/`, so a repo-root
   `implicit/` would be silently dropped from wheel and sdist). Cost ≈ **45 import
   lines**, and it removes a real misfiling regardless of any rename.
2. **Only then** ask whether the remainder has earned the name. If `geom/` is by that point
   NURBS core plus the C base classes, `nurbs/` is honest — and the rename costs the ~422
   remaining lines plus the 9 cimports and 8 build lines. **PREFLIGHT 2026-08-16: the
   pickle-fixture cost is neither regeneration nor a shim** — that was a false dichotomy.
   A `find_class`-remapping Unpickler (`tools/pickle_module_migrate.py`) rewrites the
   embedded module path losslessly: proven bit-exact on all 5 fixtures, with the consuming
   `test_nssx5.py` selections passing 7/7 against the migrated files and the negative
   control failing. No engine re-run, no shipped shim. Constraint: run it AFTER the rename
   lands (re-dump stamps the class's live `__module__`). Also migrate the untracked
   working pickles (`examples/ssx/nurbs_nurbs_intersection_{1,2,4,7,9}.pkl`,
   `tests/norm*.pkl`, `examples/csx/result1.pkl`, `brep_result.pkl`,
   `examples/topo/cylinder.pkl`), and note a sixth tracked consumer this document missed:
   `examples/ssx/nurbs_ssx5_coverage_check.py:98`.
3. The `mmcore.nurbs.nurbs` stutter is then removed by renaming `nurbs.pyx` → `_core.pyx` in
   the same commit, not left for later.

**Do not do step 2 before step 1.** Renaming first means moving `bvh/`, `octree.py` and
`implicit/` *twice*, and the second move would be out of a directory whose name already lies
about them.

#### B — "Delete `ds/` if nobody imports it; move `union_find` to its caller." — **confirmed; `ds/` dissolves.**

Measured: `mmcore/ds/` holds 9 tracked files and has **exactly one** import edge from outside
itself, repo-wide:

```
mmcore/geom/bvh/lbvh.py:212:from mmcore.ds.union_find import group_tuples
```

`bkdtree.py`, `tree/avl.py`, `tree/rtree.py` and `cdll/*` have zero external importers
(`cdll` imports only itself) and are already on the §3.1 DELETE list. So the whole package
dies with them except `union_find.py`.

Two corrections to the brief, both verified:

- The import is **module-level**, not function-local — unindented at `lbvh.py:212`,
  physically mid-file between two definitions. It is unconditional.
- Only `group_tuples` is imported. The `UnionFind` class it is built on is never imported
  anywhere, including by tests.

**Decision:** move `group_tuples` — and the `UnionFind` class it needs — into
`mmcore/geom/bvh/lbvh.py` beside its single caller (`lbvh.py:354`), then delete `mmcore/ds/`
entirely. Both are small and have no other consumer; a package existing to hold one function
for one caller is the thing this restructure is for.

**On the second union-find** (`_nurbs_knots.py:1260`, `_union_find_clusters`): **keep them
separate, do not merge.** They share only the abstract disjoint-set idea:

| | `ds/union_find.py::group_tuples` | `_nurbs_knots.py::_union_find_clusters` |
|---|---|---|
| Input | list of integer-like tuples | `(n, d)` float point array + `tol` |
| Edges | shared tuple *values* (exact identity) | metric proximity, `‖pᵢ−pⱼ‖² ≤ tol²` |
| DSU | recursive full path compression + **union by rank** | iterative path halving, **no rank heuristic** |
| Output | `list[list[tuple]]` | `dict[root, list[point_index]]` |
| Cost | `O(T·α(n))`, dominated by building the value→tuple index | **`O(n²d)`**, dominated by the all-pairs distance scan |

`group_tuples` cannot express metric proximity — it consumes edges that already exist. The
knot routine's real cost is the `O(n²)` scan, which a shared DSU would not touch, and its
caller `join_curves` (`:1342`) needs the point-index→root mapping to build deterministic join
nodes (`:1347–1367`). Merging them would add a `geom → ds` dependency to buy a cosmetic
de-duplication of ~15 lines while making the `O(n²)` bottleneck *less* visible. A shared
substrate becomes worth it only if a third genuinely generic DSU consumer appears.

### 3.1 What leaves — DELETE (51 candidates, all six channels clear)

Grouped by why they died, oldest last-touch first. Each was checked against Python
importers, test/example importers, `cimport` consumers, `build.py`, dynamic references,
and public-entry-point status.

**Broken since the `geom.curves`/`geom.surfaces` deletion (2026-06-09, `c14fd3e`)** — these
cannot even be imported today, so nothing can be depending on them:

| Path | Last work | Note |
|---|---|---|
| `numeric/algorithms/point_in_curve.py` | 2025-11-04 | |
| `numeric/intersection/implicit_implicit.py` | 2025-05-18 | also imported by `geom/octree.py` — cut that edge first |
| `numeric/intersection/ssx/_ssx31.py` | 2025-12-30 | **Q2: DELETE.** Takes `_ssx_utils.pyx` with it — see §6.4 |
| `numeric/intersection/ssx/boundary_intersection.py` | 2025-12-30 | also a legacy-CSX consumer |
| `numeric/intersection/ssx/ode.py` | 2025-12-30 | **Q3: DELETE.** `_detect_intersections.py` is KEPT instead — see §7.5 |
| `numeric/intersection/ssx/dqr.py` | 2025-08-11 | **Q1: DELETE**, with `dqr4.py`, `cydqr.pyx` and `_dqr.cpp` — see §6.5 |

> **CORRECTION — `_detect_intersections.py` was wrongly listed as broken.** Earlier drafts
> put it in this group. It is **not** in it: `grep` finds no `mmcore.geom.curves` import in the
> file, and `import mmcore.numeric.intersection.ssx._detect_intersections` **succeeds at HEAD**.
> Its only legacy coupling is inside `if __name__ == "__main__":` (`:432`). Q3 keeps it, and
> the repair is small — §7.5.

**Structurally broken on their own terms:**

| Path | Why |
|---|---|
| `numeric/area.py` | imports `notes.offset`, which does not exist anywhere |
| `topo/shell_ops.py` | imports bare `halfedge_topology` |
| `numeric/newton/constrained.py` | `IndexError` at import time |
| `numeric/intersection/csx/_overlaps.py` | imports a name absent from `_ncsx2` |

**Superseded solver generations** — each has a live successor in the same directory:

`ccx/_bex_ccx2.py`, `ccx/_bez_ccx3.py`, `ccx/_nccx.py` (⚠ see below),
`csx/_ncsx.py`, `csx/_ncsx2.py`, `csx/_bez_csx3.py`, `csx/_bez_overlap.py`,
`csx/_cs_int.py`, `csx/_ch2d.py`, `csx/_steriographic_projection.py`,
`csx/_ncsx_new_intersections_test.py`, `ssx/_ssx4.py`, `ssx/_bez_ssx6.py`, `ssx/dqr4.py`.

> **⚠ `ccx/_nccx.py` is the one trap in this group — PREFLIGHT 2026-08-16: softer than
> stated.** 453 lines, zero direct test, 3 library importers — but all three edges are
> dead-path: `csx/_bez_overlap.py` has zero importers repo-wide and is itself DELETE;
> `geom/_nurbs_construct.py:382` feeds only a `__main__` demo and the broken zero-consumer
> `network_surface` (calls undefined `nurbs_curve_bvh`); `ccx/_bez_overlap.py:190` sits in
> an unimported branch (`_curve_boundary_hits`/`_bez_curve_surface_overlap`, nothing
> imports them). `_nccx.py` can go in the same batch once `csx/_bez_overlap.py` is deleted,
> `_nurbs_construct.py`'s dead tail is cut (its live exports `circle`/`ruled` don't touch
> `_nccx`), and the dead branch in `ccx/_bez_overlap.py` is removed.

**Never-referenced utilities** (zero of everything, some for two years):

`numeric/prime.py`, `numeric/bisection.py`, `numeric/log_scaling.py`,
`numeric/sdf_mininum.py`, `numeric/implicitize.py`, `numeric/algorithms/surface_area.py`,
`numeric/algorithms/bounding_planes.py`, `numeric/algorithms/tnb.py`,
`numeric/algorithms/point_inversion.py`, `numeric/algorithms/sweep_line/*`,
`numeric/intersection/pp.py`, `numeric/intersection/classify_contact.py`,
`numeric/intersection/_stencil.py` (untracked, zero importers),
`numeric/newton/bounded.py`, `geom/_nurbs_extension.py`, `geom/_nurbs_reparametrize.py`,
`geom/_nurbs_offset.py`, `geom/implicit/{cassini,genus2,mc,sdf}.py`,
`geom/implicit/tree/utils.py`, `ds/bkdtree.py`, `ds/tree/{avl,rtree}.py`,
`topo/brep/_funcs.py`, `topo/mesh/adaptive.py`, `extras/renderer/snaps.py`.

**Now DELETE by owner decision** (each was QUARANTINE until Q1–Q12 were answered):

| Path | Answer |
|---|---|
| `numeric/interval/solver.py` | **Q4: dead.** The only `interval/` sibling excluded from `__init__.py` — that exclusion was the intent, not an oversight |
| `numeric/projection/certified_proj.py` (1033 L) | **Q5: delete.** Not revived; the `sympy`/`blosc2`/`mpmath` dependency stays out. Clears the last real import failure |
| `numeric/geodesic/{geodesic,offset}.py` + empty `__init__` | **Q9: delete** |
| `numeric/newton/constrained.py` | **Q12: delete.** The `IndexError` at import is not worth repairing |
| `ssx/dqr4.py` | **Q1: delete**, with `dqr.py` |
| `numeric/cbern.pyx` | **Q7: delete — measured, see §6.3.** ✅ *already executed* |
| `ds/cdll/*`, `ds/__init__.py`, `ds/tree/__init__.py` | **§3.4-B:** the rest of `ds/` after `union_find` moves to `lbvh.py` |
| `geom/bvh/_lbvh.pyx`, `geom/bvh/__init__.py` implementation | **Q6: `bvh/lbvh.py` survives**; the other two go |
| `ssx/surface_surface.py` | **Q8: retire** — see §4.2. Clears 6 of the remaining baseline failures |
| `ssx/_ssx_utils.pyx` (+ `.c`, 3 `.so`, `build.py` entry) | **NEW-Q1 resolved: delete** — see §6.4 |
| `ssx/cydqr.pyx` + `ssx/_dqr.cpp` + commented `build.py:139-146` | **Q1 consequence: delete** — see §6.5 |

**Native** (see §6): `geom/cimplicit.pyx`, `geom/knots.pyx`, `numeric/routines/sliding_window.pyx`,
`numeric/integrate/rk45.pyx`, `numeric/intersection/_parametric_parametric.pyx`,
`numeric/matrix/{__init__.pyx,__init__.pxd,cmatrix.h}`, `rect_subtract_cy.pyx`,
and (PREFLIGHT 2026-08-16) `geom/bvh/__init__.pxd` — 0 bytes, zero cimporters, same
class as `cimplicit.pyx`.

### 3.2 MERGE

| From | Into | Why |
|---|---|---|
| `numeric/intersection/_bezier_common.py` | `numeric/_bezier_common.py` | L0 substrate misfiled under L1 — see §5 |
| `numeric/intersection/_bern_zero_1d.py` | `numeric/_bern_zero_1d.py` | same |
| `ccx/_bez_overlap.py` + `csx/_bez_overlap.py` | one shared module | duplicated file in two packages |
| `tests/test_brep_geom.py` | `tests/test_brep.py` | same subsystem |
| `tests/test_point_in_region.py` | `tests/test_boolean2d.py` | same subsystem; mark the 1745s case slow |

### 3.3 Formerly QUARANTINE — now resolved

Every row below carried an owner question. All are now decided; the two that remain open
are marked and restated in §8 as NEW-Qs.

| Path | Resolution |
|---|---|
| `_bez_ssx5.py` alpha-sizing (`:4867-4896`) | **KEEP, untouched. Q14: "нужно эксперементировать, пока не трогать".** LIVE and load-bearing — the displaced-seed prepend is a deliberate topological short-circuit ("without it the arc is lost"), pinned `xfail(strict=True)` at `test_nssx5_toroidal_loop.py:245`. **Out of scope for this restructure.** |
| `_deflate.py` cluster-1 stack (24 sites) | **KEEP, untouched. Q13: "не знаю, нужно будет проверять, пока не трогать".** Subfamilies E/F/G stay in place; no reachability fixture is built now. **Out of scope.** |
| `algorithms/cygjk.pyx` + `_gjk.cpp` | **KEEP.** LIVE and already fixed — `dotSignMargin()` exists, `tol` retired; the issue doc claiming "no fix applied" is stale. 8 library importers. |
| `ssx/surface_surface.py` | **Q8: RETIRE.** See §4.2 — now a decision, not a recommendation. |
| `ssx/_detect_intersections.py` | **Q3: KEEP.** ⚠ CORRECTION: it is not broken and never was transitively orphaned — the dependency runs the *other* way (`_ssx31.py:28` imports *it*). Repair recipe in §7.5. |
| `numeric/interval/solver.py` | **Q4: DELETE** (dead). |
| `numeric/projection/certified_proj.py` | **Q5: DELETE.** Not revived. |
| `numeric/geodesic/{geodesic,offset}.py` | **Q9: DELETE.** |
| `numeric/cbern.pyx` | **Q7: DELETE — measured.** Identical to 1.7e-16, but only 1.24x end-to-end and **0.81x (slower) on the ssx hot path**. §6.3. ✅ *executed* |
| `ssx/cydqr.pyx` | **DELETE** with `dqr.py`/`dqr4.py` (Q1). Nothing cimports it; §6.5. |
| `ssx/_ssx_utils.pyx` | **DELETE** — orphaned by Q2; §6.4. |
| `geom/bvh/_lbvh.pyx` | **Q6: `bvh/lbvh.py` survives**, so this one goes. |
| `tests/test_nurbs_compose.py` | ⚠ **Q11 amendment does not hold — still open (NEW-Q3).** Quarantined with an in-file note; §7.1. |
| `tests/test_newton.py` | **Still open (NEW-Q4).** Sole coverage of `cnewton.pyx` + `fdm`, fails with `IndexError`. Not resolved by any Q1–Q15 answer. |

---

## 4. Public API

### 4.1 The rule

**One obvious entry point per solver family. No version-suffixed public names — at all.**

The `_v2`/`_v4`/`_v5` suffixes are the disease, not the cure. Once the package binds the
maintained engine, the suffix has no job left; keeping it would recreate exactly the
ambiguity this section removes. Private modules keep their `_`-prefixed filenames — those
are implementation, not API.

| Family | Current public name → target | Proposed binding | Callers to rewrite |
|---|---|---|---|
| **ccx** | `ccx`, `nurbs_ccx`, `curve_iix`, `curve_pix`, `nurbs_ccx_multiple` | unchanged → `ccx.py` → `_nccx4.py` | none — already correct |
| **csx** | `nurbs_csx` → **`_ncsx4.py`** (was `_ncsx.py`); `bez_csx` kept | drop `nurbs_csx_v2`, `nurbs_csx_v4` | **13 files / 15 lines** (below; PREFLIGHT 2026-08-16 correction — was 9/10) |
| **ssx** | `nurbs_ssx` → **`_nssx5.py`** (was `_ssx4.py`) | drop `nurbs_ssx_v5` | 1 library, 1 test, 20 example lines |

**CSX legacy-binding call sites, grep-verified at HEAD** — 3 library, 1 test, 5 example files:

| Site | Class |
|---|---|
| `mmcore/numeric/implicitize.py:435` | library (function-local import) |
| `mmcore/numeric/intersection/ssx/_detect_intersections.py:434` | library (function-local) |
| `mmcore/numeric/intersection/ssx/boundary_intersection.py:20` | library — already broken at HEAD |
| `tests/test_csx.py:5` | test |
| `examples/csx/overlap_nurbs_intersection.py:12`, `_3.py:9`, `_4.py:12`, `_6.py:12` **and `:23`**, `_6_new.py:12` | example |

Note `_6.py` imports *both* bindings, and `_6_new.py` — a `_new` file — still pulls the
legacy one at line 12 while using `_ncsx4` at 23. Flipping the binding fixes both by
construction.

> **PREFLIGHT 2026-08-16 — the table above undercounts.** Dropping `nurbs_csx_v2` (this
> step also does that, and `_ncsx2.py` is on the §3.1 DELETE list) adds **4 example
> files** absent from the table: `examples/csx/nurbs_nurbs_intersection_1.py:5`,
> `overlap_nurbs_intersection_2.py:12`, `overlap_nurbs_intersection_4_new.py:12`,
> `overlap_nurbs_intersection_5.py:229`; plus one further library edge into `_ncsx2`,
> `csx/_overlaps.py:7` (already broken, dies in Step 6). True blast radius:
> **13 files / 15 lines**. Also: `tests/test_csx.py` is currently **GREEN** (1 passed) —
> deleting it with `_ncsx.py` drops the passed count by one; expected, not a regression.

**SSX legacy `ssx` name:** exactly **one** library consumer,
`mmcore/geom/_nurbs_transform.py:5`. **That edge must be cut before `surface_surface.py`
can go** — and it is also an L0→L1 layering violation, so cutting it pays twice.

**`_bez_ssx6` consumers: exactly two, neither library** — `tests/test_bez_ssx6_contract.py:4`
and `examples/ssx/bez_ssx6_baseline.py:21`. Deleting the 3790-line frozen fork means
deleting exactly those two files. The test's only purpose is keeping the fork importable.

### 4.2 `surface_surface.py` — the module a static graph gets wrong

It is imported by `ssx/__init__.py:1` and exported as the public `ssx`, so every import
graph calls it healthy and LIVE. It is not:

- `NotImplementedError` at `:87` and `:90` — **2 of its 4 advertised type combinations
  do not work**;
- it carries 10 of the suite's 11 failures;
- its own code warns that `spt` is deprecated and `tol` is now ignored — the contract
  moved out from under it;
- 4 of those 10 failures are not its fault at all, but the `_nurbs_knots.py:2043` typo.

**DECIDED — Q8: retire it, and bind `ssx` to `nurbs_ssx_v5`'s entry point.**

One number in the original argument needs correcting now that the `.knot` typo is fixed
(§7.2). Retiring `surface_surface.py` clears **6** baseline failures, not 10: the other 4
were never its fault — they were the `_nurbs_knots.py:2043` bug, and they are **already
green**. The 6 that remain are its own contract-drift count assertions (`spt` deprecated,
`tol` ignored), and they die with the module.

`tests/test_nurbs_ssx.py` tests the retired public surface, so it goes with it — its
successor coverage is `tests/test_nssx5.py` and `tests/test_nssx5_toroidal_loop.py`.

### 4.3 `import mmcore`

Today `mmcore/__init__.py` is four lines that call `importlib.metadata.version("mmcore")`.
From a source checkout without an installed distribution this raises
`PackageNotFoundError` — bare `import mmcore` fails. That is not hypothetical; it happened
twice during this analysis.

**Proposed:** make the root import dependency-free and never-raising —

```python
try:
    __version__ = version("mmcore")
except PackageNotFoundError:      # source checkout, not installed
    __version__ = "0.0.0+source"
```

**DECIDED — Q10: stay thin.** The root does *not* re-export the three solver entry points.
Callers import from the family (`from mmcore.numeric.intersection.ssx import nurbs_ssx`).
A façade would cost import time on a package whose heavy modules are Cython, and would
re-create in `__init__.py` exactly the "one name, which engine?" ambiguity §4.1 removes.

The `PackageNotFoundError` fix above still applies — it is independent of the façade
question, and `import mmcore` must stop raising either way.

---

## 5. Layering

### The rule, as a checkable predicate

Assign each module a layer:

| Layer | Contents |
|---|---|
| **L0** | `geom/` NURBS core + `numeric/` substrate (Bernstein/Bézier cells, work budget, Newton, intervals, AABB/BVH) |
| **L1** | `numeric/intersection/` — the ccx/csx/ssx solver families |
| **L2** | `construction/`, `topo/`, `compat/` |
| **L3** | `extras/` — occ, rhino, renderer, torch |

> **`layer(importer) >= layer(imported)` must hold for every import edge in `mmcore/`.**
> Equivalently: no module may import from a higher layer. `extras/` must have **zero**
> inbound edges from L0–L2.

This is mechanically checkable — the script that produced the table below is ~40 lines of
`ast` and belongs in CI once the debt is cleared.

### Measured violations: 15 edges

**L0 → L1 (8)**

| Importer | Imports |
|---|---|
| `geom/_nurbs_construct.py` | `intersection/ccx/_nccx.py` |
| `geom/_nurbs_transform.py` | `intersection/ssx/__init__.py` |
| `geom/octree.py` | `intersection/implicit_implicit.py` |
| `numeric/_bez_closest_point.py` | `intersection/_bezier_common.py` |
| `numeric/closest_point.py` | `intersection/_bern_zero_1d.py` |
| `numeric/closest_point.py` | `intersection/_bezier_common.py` |
| `numeric/algorithms/point_in_curve.py` | `intersection/ccx/ccx.py` |
| `numeric/implicitize.py` | `intersection/csx/__init__.py` |

**L0 → L2 (3):** `geom/_nurbs_param_tol.py`, `geom/_nurbs_transform.py`,
`numeric/implicitize.py` → `construction/`.
**L1 → L2 (2):** `ssx/_ssx4.py`, `ssx/trace_inter_segm.py` → `construction/`.
**L0/L1 → L3 (2 — the ones that matter most):** `geom/_nurbs_transform.py` →
`extras/renderer/`, and `ccx/_bez_ccx3.py` → `extras/occ/geom_int.py`.

### Three of these are not real violations — they are misfiled files

`_bezier_common.py` and `_bern_zero_1d.py` sit under `intersection/` but are pure
Bernstein/Bézier substrate with no solver-family knowledge. `numeric/closest_point.py`
and `_bez_closest_point.py` import them because they genuinely need substrate — the code
is right and the *directory* is wrong. Moving both files to `numeric/` (§3.2) erases 3
violations without touching a single import statement's intent.

Of the remaining 12: 6 vanish with modules already marked DELETE
(`point_in_curve`, `implicitize`, `_ssx4`, `_bez_ccx3`, `octree`→`implicit_implicit`,
`_nurbs_construct`→`_nccx`), 2 are the `extras/` edges that must be cut on principle,
and 4 are genuine `construction/` couplings needing a real decision.

**Both `extras/` inbound edges deserve emphasis:** a core module importing a renderer, and
a solver importing OpenCASCADE, are exactly the couplings that make a kernel un-embeddable.
Both sit in modules already slated for DELETE or rework, so this debt is nearly free to clear.

---

## 6. Native / Cython surface

### Reconciled inventory

`build.py` contains 28 `Extension(` occurrences: **24 active** (23 Cython in
`cython_extensions` at line 62, plus 1 pure-C in `native_extensions` at line 273) and
**4 commented out** at lines 127, 135, 187, 223. On disk: 24 module families × 3 ABIs
(cp39/cp313 universal2, cp314 arm64-only — consistent with `-mcpu=native -flto` at line 48).
**No orphan `.so`.** The defect is the inverse — an orphan *source*, below.

### Two defects worth fixing before anything is moved

**6.1 — `profile=True` and `linetrace=True` ship in the default build.**

```python
# build.py:305-313
    embedsignature=False,
    language_level="3str",
    freethreading_compatible=True,
    #subinterpreters_compatible=True,
    profile=True,          # ← line 309
    linetrace =True        # ← line 310
)
```

These reach **all 23** Cython extensions through `cythonize(compiler_directives=...)` at
lines 317-323. There is **no way to turn them off**: `os.environ` is never *read* in
`build.py` (only written at line 17, in the MSVC block), and there is no `sys.argv` or
setup argument. Every user pays function-call profiling overhead on the hottest numerical
paths in the kernel.

One nuance, verified in the checked-in generated C rather than assumed: the
`CYTHON_TRACE`/`CYTHON_TRACE_NOGIL` macros (lines 44-45) live in `define_macros`, and
`define_macros=` appears at **exactly one line — 268, `triangle.core` only**. So
`binom.c:1428-1430` resolves `CYTHON_TRACE` → 0 while `:1416-1418` resolves
`CYTHON_PROFILE` → 1. **Profiling is on everywhere; line tracing is on nowhere.** The cost
is real but smaller than "both directives everywhere" would imply.

*Proposed fix* — one opt-in flag, default off:

```python
_DEBUG_TRACE = bool(os.environ.get("MMCORE_DEBUG_TRACE"))
...
    profile=_DEBUG_TRACE,
    linetrace=_DEBUG_TRACE,
# and gate the CYTHON_TRACE macro tuples on the same flag
```

✅ **EXECUTED** (§7 step 3). `build.py` now derives `MMCORE_DEBUG_TRACE` once near the top
and uses it in both places — the `profile`/`linetrace` compiler directives and the
`CYTHON_TRACE`/`CYTHON_TRACE_NOGIL` macro tuples. Default OFF; anything other than
unset/empty/`0`/`false`/`no` turns it on. Verified in both directions, and the full rebuild
completed cleanly with extensions importing (§7.0).

**6.2 — `ccx/_bezier_eval.pyx` is imported but has never been built.**

`_bez_ccx3.py:15-24` does `try: from ._bezier_eval import eval_bezier_raw ... except
Exception: _eval_bezier_raw_fast = None`. The `.pyx` exists. There is **no `Extension`
entry for it** in `build.py`, and **no `.so` in any ABI** — only a stale `_bezier_eval.c`,
cythonized once and never compiled. **The `except` branch has fired on every install
since the file was written; that accelerator has never executed.**

Two independent decisions follow:

- **source verdict: KEEP** — it is imported, so channel 1 of the DELETE predicate fails.
- **build decision:** add the missing `Extension` entry (a one-line fix restoring a dead
  fast path) **or** delete both the `.pyx` and its dead import. Note `_bez_ccx3.py` is
  itself marked DELETE as a superseded generation — if it goes, this question dissolves
  with it. That ordering is why the migration in §7 sequences the deletions first.

### 6.3 — `cbern.pyx`: measured, and the answer inverts the micro-benchmark ⚠ CORRECTION

Q7 asked for equivalence first, then a >2x speedup on a fundamental algorithm as the bar for
replacing `bern_sq_dist.py`. Both were measured. **The verdict is DELETE**, and the reason is
worth keeping because the obvious measurement gives the opposite answer.

**Equivalence: PASS.** 700 cases (degrees 0–24 × {0, 1, 0.5, 1e-15, 1−1e-15, 0.25, 0.75,
1e-8} plus 20 random `t` per degree). Max absolute deviation **1.665e-16** — machine epsilon.
Zero mismatches at `rtol=1e-11`. Partition-of-unity error is a wash (cbern better in 3 cases,
worse in 2, tie in 3): no divergence in either direction that matters.

**Speed, in isolation — looks like REPLACE:**

| degree | `bern_sq_dist` | `cbern` | speedup |
|---:|---:|---:|---:|
| 3 | 0.1110 s | 0.0626 s | **1.77x** |
| 5 | 0.1586 s | 0.0618 s | **2.57x** |
| 10 | 0.1282 s | 0.0321 s | **4.00x** |
| 20 | 0.1170 s | 0.0190 s | **6.16x** |
| 50 | 0.1236 s | 0.0100 s | **12.36x** |

**Speed, on the real consumers — REPLACE evaporates.** `bernstein_basis` is not called from
outside; its 24 call sites are all inside `bern_sq_dist.py`'s own `eval_*` functions, which
*are* the fundamental algorithm (squared-distance evaluation on the closest-point and
intersection paths). Swapping the implementation under them:

| consumer | stock | cbern | speedup |
|---|---:|---:|---:|
| `eval_point_curve_distance_sq` (deg 3) | 0.0561 s | 0.0453 s | 1.24x |
| `eval_curve_curve_distance_sq` (deg 3) | 0.1037 s | 0.0892 s | 1.16x |
| `eval_point_surface_distance_sq` (3×3) | 0.0539 s | 0.0446 s | 1.21x |
| `eval_surface_surface_distance_sq` (3×3) | 2.1868 s | 2.6921 s | **0.81x — SLOWER** |

All four verified to return identical results. **Best end-to-end gain: 1.24x < 2x → DELETE.**

**Why the inversion.** `cbern.bernstein_basis` returns a `_memoryviewslice`, not an
`ndarray`. Every consumer immediately does `B @ Pw`, so each call pays a
memoryview→ndarray conversion. At surface–surface sizes, where 8+ bases are combined per
call, that conversion costs more than the C kernel saves — which is exactly the hot path
that mattered.

Two further facts make "replace `bern_sq_dist.py` with `cbern.pyx`" impossible as stated:
`cbern` exposes **3** public functions against `bern_sq_dist`'s **18**, sharing exactly one
name (`bernstein_basis`) — it implements ~6 % of the surface. And the solvers that genuinely
need a fast Bernstein basis **already use a different Cython module**, `_bern_homog.pyx`
(built, 5+ library importers), whose `*_inplace` `nogil` fills avoid precisely the allocation
and conversion that sink `cbern`. `cbern.pyx` is a strictly worse duplicate of a fast path
that already exists and is already wired in.

✅ **EXECUTED**: `mmcore/numeric/cbern.pyx` removed, its `build.py` `Extension` entry removed,
and the generated `cbern.c` plus 3 `.so` artifacts deleted.

### 6.4 — `_ssx_utils.pyx` becomes zero-consumer once `_ssx31` goes → DELETE

The document warned this would happen; Q2 triggered it. Verified exhaustively:

- the **only** importer is `_ssx31.py:26` (`from ..._ssx_utils import points_equal`);
- **no** `.pyx`/`.pxd` cimports it, and it has **no `.pxd` of its own** — so deleting it
  forces no sibling signature edits (the concern that made it QUARANTINE);
- it **is** actively built (`build.py:228-229`) with `_ssx_utils.c` + 3 `.so` present;
- no test or example touches it.

There is a sharper detail: `_ssx31.py:152-157` **redefines `points_equal` in Python**, shadowing
the imported Cython one, and the calls at `:239-273` use the Cython keyword names rather than
the local signature. So the single import that kept this extension alive was already dead code
inside a module that does not import.

**DELETE** `_ssx_utils.pyx` together with `_ssx31.py`, plus its `build.py` entry and generated
artifacts. This is the same rule the owner applied to `cbern`: a built extension with no
consumer must prove its value or go — and here there is not even a consumer to measure.

### 6.5 — `cydqr.pyx` goes cleanly with `dqr.py`/`dqr4.py`

Confirmed as Q1 requires, before saying so:

- **nothing** imports or cimports `cydqr` — repo-wide search finds only documentation and its
  own commented-out build block (`build.py:139-146`; PREFLIGHT correction — was cited 140-147);
- **no compiled `.so`** exists for it;
- no test or example references it;
- it provides **no `.pxd`**; it only *consumes* sibling declarations (`vectors.pxd`,
  `cygjk.pxd`, `nurbs.pxd` at `cydqr.pyx:4,7,10`). Removing it therefore **removes** sibling
  constraints rather than creating them — the inverse of the original worry.
- `cydqr.pyx:16` is the only reference to `ssx/_dqr.cpp`, which becomes an orphaned companion.

**DELETE** `cydqr.pyx`, `_dqr.cpp`, `dqr.py`, `dqr4.py`, and the commented `build.py` block.

### The `.pyx` / `.py` boundary

**`cimport` consumers make a module KEEP even at zero Python importers.** The confirmed
edges:

| Module | Zero Python importers, but cimported by |
|---|---|
| `geom/parametric.pyx/.pxd` | `primitives.pyx:8`, `nurbs.pxd:8` — C base class of the whole hierarchy |
| `geom/knots.pxd` | `nurbs.pxd:7`, `nurbs.pyx` — header-only `cdef inline` |
| `numeric/binom.pxd` | `nurbs.pxd:9` |
| `numeric/algorithms/quicksort.pyx` | `nurbs.pyx:19` |

> **PREFLIGHT 2026-08-16 — the "zero Python importers" column is wrong for the last two
> rows, in the direction of MORE life:** `quicksort.pyx` has a Python importer
> (`gauss_map.py:8` imports `unique`), and the `binom` module has three
> (`monomial.py:8`, `bern.py:6`, `_nurbs_knots.py:1117` import `binomial_coefficient_py`).
> Both KEEPs stand, on stronger grounds than tabled.

`geom/knots.pxd` being header-only is also *why* it needs no build entry — and why the
separate `geom/knots.pyx` is redundant and marked DELETE.

### Native verdicts

**KEEP:** the 23 built Cython extensions, the pure-C `ndinterval`, the 9 companion `.pxd`
files, `knots.pxd`, and `_bezier_eval.pyx` (pending 6.2).

**DELETE:** `geom/cimplicit.pyx` and `numeric/routines/sliding_window.pyx` (both 0 bytes);
`geom/knots.pyx` (redundant with the header-only `.pxd`);
`numeric/integrate/rk45.pyx` (executes at module scope);
`numeric/intersection/_parametric_parametric.pyx` (uses names it never imports);
`numeric/matrix/{__init__.pyx,__init__.pxd,cmatrix.h}` — **provably unbuildable**:
`cmatrix.h:35` and `:44` both define `set_matrix_from_array2d`;
**`rect_subtract_cy.pyx`** (532 L at the repo root, outside `mmcore/`, not in `build.py`,
zero references repo-wide — and its semantics already ship live in
`_interval_cutout.py:237`).

**build.py hygiene:** three commented-out blocks reference sources that no longer exist —
`matrix` (147-153), `_cubic` (199-206) and `ellipsoid` (235-242) (PREFLIGHT line-number
corrections). Remove them.

**Formerly QUARANTINE — all three now decided:** `numeric/cbern.pyx` **DELETE** (§6.3,
measured: 1.24x end-to-end, 0.81x on the ssx path); `ssx/cydqr.pyx` **DELETE** (§6.5, nothing
cimports it and it constrains nothing); `geom/bvh/_lbvh.pyx` **DELETE** (Q6 — `bvh/lbvh.py`
survives). Add `ssx/_ssx_utils.pyx` **DELETE** (§6.4, orphaned by Q2).

**That dependency is now resolved:** `_ssx_utils.pyx` was KEEP only through `_ssx31.py:26`,
and Q2 deletes `_ssx31`. It is therefore DELETE too — see §6.4 for the full verification,
including the detail that its one import was already shadowed by a local redefinition.

---

## 7. Migration plan

### The contract

The repository does **not** import cleanly today and the suite does **not** run. Any plan
promising "green after every step" would be unfalsifiable. So every step is verified as a
**delta against this frozen baseline**:

| Baseline @ `f6b3655` | Value |
|---|---|
| Module import failures | **17** (6 benign optional-dep + 11 real) |
| Test collection errors | **6** |
| Tests collected | **771** |
| Full-run result (6 ignored) | **11 failed · 756 passed · 1 xfailed · 3 errors** |

> **Step criterion: a step must not increase any baseline number.** Steps that *reduce* one
> state the expected new value.

### 7.0 Phase 1 result — measured delta (2026-08-14)

Steps 1–4 are **executed**. Every number below was re-measured, not predicted:

| Metric | Baseline | After Phase 1 | Verdict |
|---|---:|---:|---|
| Module import failures | 17 | **17** | unchanged ✅ (6 benign + 11 real; nothing regressed) |
| Test collection errors | **6** | **0** | **fixed — the suite runs for the first time** ✅ |
| Tests collected | 771 | **771** | unchanged ✅ *(see note)* |
| `tests/test_nurbs_ssx.py` failures | 10 | **6** | as predicted ✅ |
| Default build profiling | always on | **off unless `MMCORE_DEBUG_TRACE`** | ✅ builds clean, extensions import |
| Slow-test deselection | none | `-m "not slow"` deselects exactly **1** | ✅ ~45 min → ~16 min |

Full-suite result, run **with** the slow test so it is like-for-like with the baseline:

| Full run | Baseline | After Phase 1 |
|---|---:|---:|
| failed | 11 | **7** |
| passed | 756 | **760** |
| skipped | 0 | **1** (the quarantined `test_nurbs_compose.py`, §7.1) |
| xfailed | 1 | **1** |
| errors | 3 | **3** |
| wall clock | 2698.78 s | 2659.92 s |

Measured end to end: `7 failed · 760 passed · 1 skipped · 1 xfailed · 3 errors in 2659.92s`.
The baseline's 756 passed becomes 760 — the same +4 seen in the failure column, i.e. the four
`.knot` cases moved from failing to passing rather than disappearing.

**No baseline number increased; one dropped by 4.** Those 4 are exactly the `.knot`
`AttributeError` cases. The 7 that remain are the 6 `test_nurbs_ssx.py` contract-drift
assertions — which die with `surface_surface.py` under Q8 — plus `test_newton.py`
(**NEW-Q4**). The 3 errors are the known `test_nccx4.py` setup-time `geom.curves` errors,
cleared by Step 5.

> **Note on "771 → 771".** The three deleted test files contributed **zero** collectable
> tests — they aborted collection rather than adding to it. So removing them changed the
> error count without changing the collected count. That is the correct outcome and worth
> stating, because the original step text predicted "771 minus the removed files".

Baseline command, used by every step (run from the repo root):

```bash
.venv/bin/python - <<'EOF'
import importlib, subprocess, sys, io, warnings
warnings.filterwarnings("ignore")
mods = []
for f in subprocess.run(["git","ls-files","mmcore"],capture_output=True,text=True).stdout.split():
    if f.endswith(".py"):
        m = f[:-3].replace("/",".")
        mods.append(m[:-9] if m.endswith(".__init__") else m)
bad = {}
for m in sorted(mods):
    try:
        o,e = sys.stdout, sys.stderr
        sys.stdout = sys.stderr = io.StringIO()
        try: importlib.import_module(m)
        finally: sys.stdout, sys.stderr = o,e
    except BaseException as ex:
        bad[m] = f"{type(ex).__name__}: {ex}"
print(f"IMPORT FAILURES: {len(bad)}")
for k,v in sorted(bad.items()): print(" ", k, "->", v)
EOF
```

### Sequence — cheapest and highest-value first

Steps 1-3 pay for themselves immediately and touch no algorithm.

---

**Step 1 — Make the test suite runnable.** ✅ **DONE** *(highest value in the document)*

`pytest tests` executed **zero tests**; it now runs end to end. Deleted the 5 test files whose
library targets are gone: `test_curve_bool.py`, `test_nurbs_periodic.py`,
`test_perf_intersection.py`, and `test_boundary_intersection{,_robust}.py` (the latter two with
their module, which Q2/Q3 delete). All were tracked, so all are recoverable from git.

#### 7.1 ⚠⚠ DOUBLE CORRECTION (PREFLIGHT 2026-08-16) — the re-point IS possible; §7.1's original claims were wrong

The original §7.1 correction claimed the suite could not be re-pointed at `sbern.py`.
**Three of its load-bearing claims are contradicted by the file's own body:**

- "the 3 sbern imports are unused" — **false**: the body calls `nurbs_bezier_to_bern` at
  `:232,233,274,275,316,317` and `compose_curve_curve` + `bern_to_nurbs_bezier` at
  `:236,278,320`. The whole `TestNURBSComposition` class (3 tests) targets sbern's real
  API and **passes 3/3** once the import line is restored. The quarantining commit
  (`aa818c2`) renamed the call sites to sbern's current names (`compose_curve_curve_sb` →
  `compose_curve_curve`) and then deleted the import block it needed — the suite is
  broken by a deleted import line, not by a missing capability.
- "all 15 tests exercise `_nurbs_compose` symbols only" — it has **12** tests, not 15,
  and all 12 were run **green** via a 34-line shim over existing sbern primitives.
- Per-symbol: `compose_nurbs_curves` and `compose_bezier_segments` were **dead imports**
  with zero call sites (equivalents at `sbern.py:709` and `:410`); 4 more are 3-to-6-line
  derivations over `compose_curve_curve`, `decompose_curve`, and sbern's private helpers
  (`_segment_interval:364`, `_roots_against_constant:369`, `_collect_split_parameters:392`);
  `BezierSegment` is a 6-line attribute bag. **No capability is absent.**

The suite is *not* redundant with `tests/test_nurbs_curve_compose.py` (14 tests, green):
it uniquely pins the scalar composition algebra against closed form (t²∘t² = t⁴), the
power-basis root finder, breakpoint finding, Bézier span extraction, and the
`nurbs_bezier_to_bern`/`bern_to_nurbs_bezier` round trip with non-unit weights.

**NEW-Q3 RESOLVED (owner, 2026-08-16): rewrite the suite in place against sbern's real
API** — test-file only, zero production change. Conventions decided: derivations stay
test-local helpers (sbern's `_`-helpers are not promoted until a second consumer appears),
and sbern's strictly-interior breakpoint convention is kept (endpoints recovered via
`_segment_interval` where a test needs them). Executed in Phase-2 batch 2a.

```bash
.venv/bin/python -m pytest tests -q --co 2>&1 | tail -3
```
*Expect:* collection errors **6 → 0**; collected ≈ 771 minus the removed files; the suite
runs end to end for the first time.

---

**Step 2 — Fix `_nurbs_knots.py:2043`.** ✅ **DONE — but it was not a one-character fix.**

#### 7.2 What the typo actually was

The diagnosis "`.knot` → `.knots`" is right about the symptom and wrong about the cure: a
blind rename breaks the other caller. **Two curve representations reach this module:**

| | spelling | replace | 
|---|---|---|
| `NURBSCurveTuple` (`_nurbs_eval.py:41`, the tuple ABI) | `.knot` | `._replace(...)` |
| `NURBSCurve` (`geom/nurbs.pyx:1121`, Cython) | `.knots` | settable properties |

`refine_curve` was written for the tuple and receives the Cython curve — `mmcore/construction/_ruled.py:80-82`
converts tuples *to* `NURBSCurve` before calling in, and then reads the Cython-only
`._control_points` off the result (`:95`). So the fix has to be **type-preserving in both
directions**, and renaming the attribute would simply move the `AttributeError` to the tuple path.

Line 2044 hid a second failure behind the first: `curve._replace(...)` exists **only** on the
named tuple, so the Cython path would have died one line later anyway.

**Fix applied:** two small helpers in the same module, used by `refine_curve` and by the
existing `_get_knots`/`_set_knots` (which carried the identical latent bug at `:2106`/`:2113`):

- `_get_knots(curve)` — reads `.knot` or `.knots`, whichever the object has;
- `_replace_curve(curve, *, knots=…, control_points=…, weights=…)` — returns **the same
  representation** it was given. For the tuple that is `_replace`. For the Cython curve it
  **rebuilds through the constructor**, because its `control_points` setter explicitly refuses a
  change in point count ("You cannot change the number of control points…") — and knot
  refinement adds control points by definition, so in-place assignment can never be right here.
  A knot-only edit still uses the setter, which preserves periodicity and cache state via
  `knots_update_hook()`.

Verified: `tests/test_nurbs_ssx.py` **10 failed → 6 failed** exactly as predicted, and
`tests/test_knot_removal.py` stays 11/11 green.

```bash
.venv/bin/python -m pytest tests/test_nurbs_ssx.py -q 2>&1 | tail -3
```
*Expect:* failures **10 → 6** (the 4 `AttributeError` cases clear; the 6 contract-drift
count assertions remain, and are Q8's business).

---

**Step 3 — Turn off default profiling** (§6.1). ✅ **DONE.** No behaviour change; a straight
speedup. Gated behind `MMCORE_DEBUG_TRACE` at both sites (compiler directives + the
`CYTHON_TRACE` macro tuples). Verified in both directions: with the flag unset,
`profile=False, linetrace=False` and no `CYTHON_TRACE` macros; with it set, all three return.
Full rebuild exits 0 and `import mmcore.geom.nurbs, mmcore.numeric.bern` succeeds.

```bash
.venv/bin/python build.py && \
.venv/bin/python -c "import mmcore.geom.nurbs, mmcore.numeric.bern; print('ext ok')"
```
*Expect:* builds clean, extensions import, baseline unchanged.

---

**Step 4 — Mark the 1745s test slow.** ✅ **DONE.**
`test_point_in_region.py::test_point_whose_segment_is_tangent_to_circle_is_outside` is 65% of
the suite's wall clock. Marked `@pytest.mark.slow`, and the marker is **registered** in a new
`[tool.pytest.ini_options]` block in `pyproject.toml` — the project had no pytest config at
all, so an unregistered marker would have emitted `PytestUnknownMarkWarning` on every run.
**Nothing is deselected by default**: `pytest tests` still runs all 771.
Verified: `-m "not slow"` collects 770/771 (1 deselected), `-m slow` collects exactly 1, and
zero unknown-mark warnings.

```bash
.venv/bin/python -m pytest tests -q -m "not slow" --durations=10 2>&1 | tail -5
```
*Expect:* ~45 min → ~16 min; pass/fail counts otherwise unchanged.

---

**Step 4b — Close Q7 with data.** ✅ **DONE.** Equivalence verified (700 cases, max deviation
**1.665e-16**), then benchmarked. The micro-benchmark says 1.77x–12.36x → REPLACE; the **real
consumers** say **1.24x best, and 0.81x — slower — on the ssx hot path** → **DELETE** under the
owner's own >2x rule. Cause: `cbern` returns a memoryview, and every consumer immediately
does `B @ Pw`. Full measurement in **§6.3**. `cbern.pyx`, its `build.py` entry, and its
generated artifacts are removed.

---

**Step 5 — Delete the 5 `geom.curves`-broken modules, and REPAIR `_detect_intersections.py`.**

The broken set is `point_in_curve.py`, `implicit_implicit.py`, `_ssx31.py`,
`boundary_intersection.py`, `ode.py`, `dqr.py` — cut the two live edges first
(`geom/octree.py` → `implicit_implicit.py`, and any consumer of `point_in_curve.py`).
Delete `_ssx_utils.pyx` in the same step (§6.4) and `cydqr.pyx`/`_dqr.cpp`/`dqr4.py` (§6.5).

#### 7.5 ⚠ CORRECTION — `_detect_intersections.py` is NOT broken, and Q3 keeps it

Earlier drafts listed it as "transitively orphaned via `_ssx31`" and implicitly broken.
**Both claims are false, and this was verified two ways:**

- `grep` finds **no `mmcore.geom.curves` import** anywhere in the file (its imports are
  lines 4–17: current NURBS, AABB, GJK, BVH, Gauss-map, SciPy, vectors);
- `import mmcore.numeric.intersection.ssx._detect_intersections` **succeeds at HEAD**, and it
  does not appear in the 17-failure import baseline.

The dependency also runs the **opposite** way from what was assumed: `_ssx31.py:28` imports
`detect_intersections`, not the reverse. Deleting `_ssx31` orphans nothing here.

**The repair it needs is therefore small and confined to one place.** The only legacy coupling
is inside `if __name__ == "__main__":` (`:432`) — a demo block, not library code:

- `:434` — `from mmcore.numeric.intersection.csx import nurbs_csx`, which `csx/__init__.py:2`
  binds to the dead `_ncsx.py`. This is one of the 9 legacy-CSX call sites in §4.1.
- `:490, :495, :510, :511` — the demo consumes the **legacy return contract** (a list of
  tuples, indexed `oo[2][0]`). The modern `_ncsx4.py:285-317` returns
  `(isolated, overlaps, status)` and `:319-320` **rejects the old `ptol` keyword**. So flipping
  the binding without touching the demo would fail at runtime, not just import.

**Recommended repair: delete the `__main__` block (`:430-518`).** No test or example exercises
it, its only importer is the module being deleted, and it is the module's sole legacy coupling
— removing it repairs the file *and* removes one of the 9 CSX call sites Step 7 must rewrite.

If the demo is worth keeping, rewrite it structurally instead:

```python
from mmcore.numeric.intersection.csx import nurbs_csx        # after Step 7 this IS _ncsx4
isolated, _overlaps, _status = nurbs_csx(c, j, tol=TOL)      # no ptol
for item in isolated:
    ptss.append(np.asarray(item["point"]).tolist())
```

*Expect:* real import failures **11 → 5** (PREFLIGHT correction — this step removes **6**
of them, not 5: `dqr.py` is a sixth deletion, failing on `geom.surfaces` rather than
`geom.curves`; `_detect_intersections` was never one). Collection errors stay 0; test
counts unchanged.

---

**Step 6 — Delete the 4 structurally-broken modules** (`area.py`, `shell_ops.py`,
`newton/constrained.py`, `csx/_overlaps.py`). **Q12 answered: delete `constrained.py`** — the
import-time `IndexError` is not worth repairing.

*Expect:* real import failures **5 → 1** (PREFLIGHT correction — was "6 → 2"), then
**→ 0** once Q5 deletes `certified_proj.py` in Step 10. **`numeric/interval/solver.py`
imports cleanly at HEAD** — it is not among the 17 failures, so Q4's delete is dead-code
removal that changes the import count by zero. **A clean import surface is reachable in
this phase.**

---

**Step 7 — Flip the CSX binding** to `_ncsx4` and drop `nurbs_csx_v2`/`nurbs_csx_v4`.
Rewrite the 3 library call sites; the **9** example files follow (PREFLIGHT: 5 legacy
`nurbs_csx` + 4 `nurbs_csx_v2` — see the corrected §4.1 blast radius, 13 files / 15 lines).

```bash
.venv/bin/python -c "from mmcore.numeric.intersection.csx import nurbs_csx, bez_csx; print(nurbs_csx.__module__)"
.venv/bin/python -m pytest tests/test_ncsx4.py tests/test_csx4_exactness_contract.py -q 2>&1 | tail -3
```
*Expect:* prints `..._ncsx4`; those tests stay green. `tests/test_csx.py` dies with `_ncsx.py`.

---

**Step 8 — Flip the SSX binding** to `_nssx5`, drop `nurbs_ssx_v5`, and **retire
`surface_surface.py` (Q8)** in the same step — they are one change, since the legacy `ssx`
name is what `surface_surface.py` is bound to. **Cut `geom/_nurbs_transform.py:5` first** — it
is the only library consumer of the legacy name and an L0→L1 violation, so cutting it pays
twice. `tests/test_nurbs_ssx.py` goes with the retired surface.

> **PREFLIGHT 2026-08-16 — sweep hazard:** `from mmcore._test_data import ssx` is a
> DIFFERENT `ssx` (the test-data list) at 8 sites (`compat/step/step_writer.py:748`,
> `numeric/gauss_map.py:495`, `ssx/_detect_intersections.py:433`, `ssx/_ssx4.py:2014`,
> `ssx/dqr4.py:232`, `newton/constrained.py:35,198`, `tests/test_newton.py:4`). The
> binding-flip grep must not rewrite those.

*Expect:* the **6 remaining `test_nurbs_ssx.py` failures disappear with the module**, taking
the suite's failure count down accordingly. (The other 4 of the original 10 are already fixed
— §7.2.)

```bash
.venv/bin/python -c "from mmcore.numeric.intersection.ssx import nurbs_ssx; print(nurbs_ssx.__module__)"
.venv/bin/python -m pytest tests/test_nssx5.py tests/test_nssx5_toroidal_loop.py -q 2>&1 | tail -3
```
*Expect:* prints `..._nssx5`; the two ssx5 suites stay green.

---

**Step 9 — Delete `_bez_ssx6.py`** with exactly its two consumers
(`tests/test_bez_ssx6_contract.py`, `examples/ssx/bez_ssx6_baseline.py`). **-3790 lines,
and the ssx naming trap disappears permanently.**

---

**Step 10 — Delete superseded generations and never-referenced utilities** (§3.1), in small
reviewable batches, now including the Q4/Q5/Q9/Q12 modules and the `bvh` losers (Q6).
**`ccx/_nccx.py` is excluded** — it is untested and still imported by 3 library modules (cut
those consumers or add a test first).

Run the baseline command after each batch. *Expect:* every number non-increasing.

---

**Step 10b — Dissolve `ds/` (§3.4-B).** Move `group_tuples` + the `UnionFind` class it needs
into `mmcore/geom/bvh/lbvh.py` beside its single caller, then delete `mmcore/ds/` entirely.
One import line disappears (`lbvh.py:212`) and a whole package goes with it.
*Do not* merge it with `_nurbs_knots.py::_union_find_clusters` — different algorithm, see §3.4-B.

---

**Step 11 — Move the misfiled substrate** (§3.2): `_bezier_common.py` and `_bern_zero_1d.py`
→ `numeric/`. Update the importers.

*Expect:* layering violations **15 → 12**, baseline unchanged.

---

**Step 12 — Cut the two `extras/` edges** and the remaining `construction/` couplings, then
add the layering check to CI:

```bash
.venv/bin/python tools/check_layering.py   # ~40 lines of ast; exit 1 on any upward edge
```
*Expect:* exit 0 — and the rule stays true from then on without anyone remembering it.

---

**Step 13 — Move the non-NURBS residents out of `geom/` (§3.4-A).** `bvh/` and `octree.py` →
`numeric/`; `implicit/` → **`mmcore/implicit/`** (inside the package — see the §3.4-A
PREFLIGHT pin). ≈45 import lines. Rewrite the one build entry that moves with it —
`mmcore.geom.implicit.tree.cbuild_tree3d` (`build.py:206-212`, dotted name AND source
path) — and delete its stale `.c`/3×`.so` at the old path (poetry-core packages untracked
unignored files, so stale binaries would ship at a dead import path). `bvh/` and `octree.py`
have no build entries. This is worth doing on its own merits — those subsystems are
misfiled today regardless of any rename.

**Step 14 — the `geom/` → `nurbs/` rename (§3.4-A). NEW-Q2 RESOLVED (owner, 2026-08-16):
run it**, as the final commit of Phase 3. The cost is the ~422 remaining import lines,
9 cimports, 8 `build.py` lines, and **migrating (not regenerating) the pickle fixtures**
via `tools/pickle_module_migrate.py` — proven lossless, see the §3.4-A PREFLIGHT note;
run it AFTER the rename lands, over the 5 tracked fixtures AND the untracked working
pickles. Rename `nurbs.pyx` → `_core.pyx` in the same commit to kill the
`mmcore.nurbs.nurbs` stutter. **Never before Step 13**, or the non-NURBS subsystems get
moved twice.

---

## 8. Decisions (Q1–Q15) and what is still open

### 8.1 Answered — these are now the plan

| # | Question | Answer | Where it lands |
|---|---|---|---|
| **Q1** | `ssx/dqr.py` + `dqr4.py` — delete both? | **Yes** | §3.1, Step 5. Takes `cydqr.pyx` + `_dqr.cpp` with them (§6.5) |
| **Q2** | `ssx/_ssx31.py` — abandoned? | **Yes** | §3.1, Step 5. Orphans `_ssx_utils.pyx` → also DELETE (§6.4) |
| **Q3** | Do `ode.py` and `_detect_intersections.py` go with it? | **Keep `_detect_intersections.py`**; `ode.py` goes | §7.5 — and it is *not broken*, correcting this document |
| **Q4** | `numeric/interval/solver.py` — unfinished or dead? | **Dead** | §3.1, Step 10 |
| **Q5** | `projection/certified_proj.py` — revive with `sympy` or delete? | **Delete** | §3.1, Step 10. Clears the last real import failure |
| **Q6** | Which of the three BVH implementations survives? | **`bvh/lbvh.py`** | §3.1; `_lbvh.pyx` + the `__init__.py` implementation go |
| **Q7** | `cbern.pyx` — unwired fast path or superseded? | **Conditional → measured → DELETE** | §6.3. Owner's 2x rule applied to real consumers, not the micro-benchmark ✅ *executed* |
| **Q8** | `surface_surface.py` — keep as public `ssx`, or retire? | **Retire it** | §4.2, Step 8. Clears **6** failures (not 10 — 4 were the §7.2 typo) |
| **Q9** | `numeric/geodesic/*` — keep as reference or delete? | **Delete** | §3.1, Step 10 |
| **Q10** | Should `import mmcore` re-export the solvers? | **Stay thin** | §4.3. The `PackageNotFoundError` fix still applies |
| **Q11** | `geom/_nurbs_compose.py` — deliberately deleted or lost? | **Deleted; moved to `sbern.py`** | ⚠ §7.1 — the move *did not land*; the test cannot be re-pointed. Quarantined, see **NEW-Q3** |
| **Q12** | `newton/constrained.py` — delete or repair? | **Delete** | §3.1, Step 6 |
| **Q13** | Cluster-1 subfamilies E/F/G — fixture or treat as dead? | **"пока не трогать"** | **Untouched.** Out of scope for this restructure |
| **Q14** | Alpha-sizing successors — still intended? | **"пока не трогать"** | **Untouched.** Out of scope |
| **Q15** | `trace_limit=512` undived at `_bez_ssx5.py:4413,4464` | **"судя по всему да"** — in scope | Same defect class as the shipped 400 derivation. **Not a restructure item** — it is an algorithm change in a live engine, so it belongs in its own ledger-tracked commit with the derivation, not in a deletion batch. Recorded here so it is not lost. |

### 8.2 NEW-Q1…NEW-Q4 — RESOLVED (owner approval 2026-08-16, evidence in the execution companion doc)

| # | Question | Decision |
|---|---|---|
| **NEW-Q1** | `ssx/_ssx_utils.pyx` — DELETE? | **DELETE, confirmed** — verified stronger than §6.4: sole importer shadowed on both sides, the extension shadows its own `points_equal` internally (cpdef `:43` vs def `:81`), and the `improve_uv` escape route is closed (`pull_curve.py:58` calls it without importing it — a latent NameError in `pull_curve`, not an edge). Executed in batch 2a. |
| **NEW-Q2** | `geom/` → `nurbs/`: Step 13 only, or 13 **and** 14? | **Both.** The pickle cost collapsed to a proven lossless migration (`tools/pickle_module_migrate.py`); `implicit/` pinned to `mmcore/implicit/`. Executed in Phase 3. |
| **NEW-Q3** | `tests/test_nurbs_compose.py` — re-implement, rewrite, or drop? | **Rewrite against sbern's real API** — §7.1's premise was refuted (see the rewritten §7.1); all 12 tests proven green against sbern. Executed in batch 2a. |
| **NEW-Q4** | `tests/test_newton.py` — repair or delete? | **REPAIR** — a 2-line stale-fixture fix (`ssx_cases[2]`→`[1]`, index drift from `1c49fd6`; plus `res1`→`res2` at `:34`, a vacuous assert). `cnewton` (9 importers, built) and `fdm` (14 importers, the `Implicit2D.dxdy` path) both fail the DELETE predicate. Follow-up ledger item: after Phase 2, cnewton's surviving call sites all lose their consumers — schedule its own measured keep-or-delete decision, with this repaired test as the instrument. Executed in batch 2a. |

### Appendix — tracked root files

13 tracked files live at the repo root. None is a stray module.

| File | Verdict |
|---|---|
| `pyproject.toml`, `LICENSE`, `README.md`, `.gitignore`, `.dockerignore` | **KEEP** |
| `build.py` | **KEEP** — §6.1 fix ✅ applied; dead-block cleanup still pending (Step 10) |
| `Dockerfile`, `docker-build-step1.sh`, `docker-build-step2.sh` | **KEEP** — packaging |
| `clean`, `clean.py` | **QUARANTINE** — `clean` is a 16-byte shim; keep one, not both |
| `upd-version.sh`, `updversion.py` | **QUARANTINE** — same duplication; `pyproject.toml` is the version SSOT |

> The two QUARANTINE rows above are duplicate-tooling housekeeping, not algorithm decisions.
> They were not part of Q1–Q15 and are left as they are; pick one of each pair whenever the
> root is next touched.

Untracked root scratch (session logs, `.pkl`, `.stp`, `.png`, `.pstat`) is not part of the
tree. It should be covered by `.gitignore` rather than enumerated.

**Note:** `rect_subtract_cy.pyx` is at the root but is a native source — its DELETE verdict
is in §6.

### Appendix — orphaned test fixtures

`tests/norm{1,2,3,4,7}.pkl` exist but are **untracked write-only outputs** of the legacy
`_ssx4.py` `__main__` block (lines 2040/2066/2118/2288) via a hardcoded absolute developer
path. No test reads them. **`tests/ssx-test{1,2}.json` are tracked with zero references
repo-wide.** All are orphaned and should go with `_ssx4.py`.

**Restructure hazard:** `tests/test_nssx5.py:908,911,981,1268` loads fixtures from
`examples/ssx/`, and `tests/test_nccx4.py:71` opens an `examples/ccx/` *source file*.
**Moving `examples/` breaks the largest ssx test.** Both paths are also CWD-relative, so
the suite only passes when run from the repo root.

---

## 9. Documentation plan

**`CLAUDE.md` is the problem, not the solution.** It is untracked — `git ls-files` does not
know it — so it has never been reviewed, and it is stale twice over: it claims version
`0.53.0` (`pyproject.toml` says `0.54.0` since `486228c`, 2026-07-02) and it points CSX at
`_ncsx.py`, the dead engine. An untracked file that instructs a newcomer to use the wrong
module is worse than no file.

**`docs/` is already the right home** — `docs/superpowers/{specs,plans,issues,designs}` is
tracked, actively used, and holds 34 documents. Nothing new needs inventing.

Proposed layout:

| Document | Owns |
|---|---|
| **`README.md`** | What mmcore is, install, a 10-line "intersect two NURBS surfaces" example |
| **`docs/ARCHITECTURE.md`** *(new)* | The layering rule and its checkable predicate; the three-tier solver shape (`_bez_*` engine → `_n*` adapter → package entry); **the two traps from §2, stated at the top**; the ledger-`L##` governance convention |
| **`docs/RESTRUCTURE.md`** *(this file)* | The migration, until it is done. Then it is deleted or archived — it is a plan, not a permanent authority |
| **`docs/superpowers/**`** | Unchanged: per-campaign specs, plans, issues |
| **Module docstrings** | Why *this* module exists and its ledger IDs — the existing convention, which works |
| **`CLAUDE.md`** | **Delete.** Its useful content moves to `README.md` + `docs/ARCHITECTURE.md`, both tracked |

Two conventions worth writing down explicitly, because both cost real time here:

1. **Governance is ledger IDs, not TODOs.** Only 8 TODO/FIXME exist in all of
   `mmcore/**/*.py`. Defects live as `L##` in commit subjects and module docstrings. To
   learn why something exists, grep the ledger ID — searching for TODOs finds nothing and
   suggests, wrongly, that nothing is tracked.
2. **`docs/superpowers/` outranks any external notes**, and code outranks both. Say it
   once, in `ARCHITECTURE.md`, so the next reader does not rediscover it by contradiction.

A third, added by this pass:

3. **A dual-type ABI must be stated where the types meet.** The `.knot`/`.knots` bug (§7.2)
   survived in a live L0 module because nothing wrote down that `NURBSCurveTuple` and the
   Cython `NURBSCurve` are *both* first-class inputs to `_nurbs_knots.py`, spelled
   differently. `ARCHITECTURE.md` should say so once, next to the tuple-ABI description —
   and the `_get_knots`/`_replace_curve` helpers are now the single place that knows it.

### 9.1 What Phase 1 changed on disk

**Committed at `aa818c2` on `tiny`, 2026-08-16** (this note previously said "nothing is
committed"; stale).

| File | Change |
|---|---|
| `mmcore/geom/_nurbs_knots.py` | `refine_curve` + `_get_knots`/`_set_knots` made type-aware; new `_replace_curve` helper (§7.2) |
| `build.py` | `MMCORE_DEBUG_TRACE` gate on `profile`/`linetrace` + `CYTHON_TRACE` macros; `cbern` `Extension` entry removed |
| `pyproject.toml` | new `[tool.pytest.ini_options]` registering the `slow` marker |
| `tests/test_point_in_region.py` | `@pytest.mark.slow` on the 1745s case |
| `tests/test_nurbs_compose.py` | module-level quarantine skip + the §7.1 evidence in-file |
| `mmcore/numeric/cbern.pyx` | **deleted** (Q7, §6.3) — with generated `.c` and 3 `.so` |
| `tests/test_curve_bool.py`, `test_nurbs_periodic.py`, `test_perf_intersection.py`, `test_boundary_intersection.py`, `test_boundary_intersection_robust.py` | **deleted** (Step 1) |

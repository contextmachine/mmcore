# Restructure — Phase 2 execution proposal (next steps)

**Status:** PROPOSAL — for owner approval. Nothing below has been executed.
**Measured at:** branch `tiny` @ `aa818c2` (2026-08-16), verified this session by a 7-agent
read-only preflight (import health, collection, call-site inventories, native claims,
branch audit, CI/packaging audit, pickle-migration proof). Repo untouched: tracked-file
`git status` clean before and after; all proofs ran on scratchpad copies.
**Authority:** `docs/RESTRUCTURE.md` (the decided plan) remains the spec. This document
adds: answers for NEW-Q1…NEW-Q4 (all four now have data), corrections the preflight found
in the spec's own gate numbers, branch/CI/packaging facts the spec does not know, and a
revised execution order.

---

## 0. Where you actually are

- **Phase 1 is executed AND committed** — `aa818c2` contains all of §9.1's changes.
  (`RESTRUCTURE.md:1306` still says "Nothing is committed"; stale, amend.)
- **Both Phase-2 gate numbers re-pin with ZERO drift** at `aa818c2`: 17 import failures
  (the same 6 benign optional-dep + 11 real modules, byte-identical list) and
  771 collected / 0 collection errors. Phase 2 can proceed against the spec's numbers,
  after the §3 corrections below (four of its *per-step expected values* are wrong).
- **The governance is untracked.** `docs/RESTRUCTURE.md`, `docs/RESTRUCTURE-ANSWERS.md`
  and `CLAUDE.md` are all invisible to `git ls-files` — the exact failure mode
  `RESTRUCTURE.md` §9 criticizes in CLAUDE.md. Fix first (Step 0).

---

## 1. Step 0 (new) — pin governance before deleting anything

1. **Commit the spec.** Fold §3's corrections into `docs/RESTRUCTURE.md`, commit it,
   and delete `docs/RESTRUCTURE-ANSWERS.md` (superseded interim copy — the decided doc
   contains every answer).
2. **Branch discipline.** Execute Phase 2 on a dedicated branch off `tiny`
   (e.g. `restructure-phase2`), one commit per step/batch, merged back in reviewed
   chunks. Deletions first, moves last — deletions are conflict-cheap, moves are
   conflict-expensive against anything in flight.
3. **`.gitignore` the root scratch** (`*.pkl` at root, `*.stp`, `*.pstat`, `.DS_Store`,
   session logs, `__pycache__/`) instead of enumerating it. Decide `*.so`/`*.c` policy
   consciously: today they are untracked-but-unignored, and **poetry-core packages
   untracked unignored files** — that is both how the wheel gets its binaries and how
   stale binaries of deleted modules would ship (see §5, hygiene rule).

---

## 2. NEW-Q1…NEW-Q4 — recommendations, now with data

### NEW-Q1 — `_ssx_utils.pyx`: **DELETE, confirmed — and stronger than §6.4 states**

Every §6.4 claim verified at HEAD. New facts, all in the same direction: the extension
*internally* shadows its own `points_equal` (cpdef `:43` vs plain def `:81`); the sole
importer `_ssx31.py:26` is shadowed at `:152-157` *and* the call sites `:239-273` use
keyword names that don't match the shadowing def (dead in both directions); and the one
adversarial escape route — its other export `improve_uv` — is closed:
`pull_curve.py:58` calls the name without importing it from anywhere (a latent
NameError, `pull_curve.py`'s own bug, not an `_ssx_utils` edge).
Delete `.pyx` + `.c` + 3 `.so` + `build.py:227-233`.

### NEW-Q2 — the `geom/` → `nurbs/` rename: **run Step 13 AND Step 14**

The rename's dominant cost is gone. §3.4-A offered "regenerate the 5 pickle fixtures
(or a compat shim, which the licence forbids)" — **that is a false dichotomy, proven
this session.** A third option exists and was demonstrated end-to-end: a one-shot
offline path rewrite via a `find_class`-remapping Unpickler.

- All 5 tracked fixtures embed exactly ONE mmcore module (`mmcore.geom._nurbs_eval`,
  two NamedTuple qualnames); everything else is rename-immune numpy internals.
- Round trip proven **bit-exact** (strict deep-compare: dtype/shape/`tobytes()`,
  float-bit equality; 269/3493/151/678/437 nodes per file, all IDENTICAL).
- The real fixture tests (`test_nssx5.py:908/981/1268` selections) pass **7/7 against
  the migrated files** and fail 7/7 in the negative control (alias removed) — the pass
  genuinely resolves through the new path.
- The engine is never re-run; no shim ships; the migrated file is a re-serialisation
  with identical values.

Tool preserved at **`tools/pickle_module_migrate.py`** (untracked; review, then commit
in Step 0 or with Step 14). One sequencing constraint, stated in its docstring: run it
**after** the rename lands (pickle re-dump stamps the class's live `__module__`).
Also point it at the **untracked** working pickles that would otherwise silently break
(`examples/ssx/nurbs_nurbs_intersection_{1,2,4,7,9}.pkl`, `tests/norm*.pkl`,
`examples/csx/result1.pkl`, `brep_result.pkl`, `examples/topo/cylinder.pkl` — the last
two also embed `mmcore.topo.brep`, untouched by this rename), and note a 6th tracked
consumer the spec missed: `examples/ssx/nurbs_ssx5_coverage_check.py:98`.

With that cost reduced to a mechanical sweep (~422 import lines, 9 cimports, 8 build
lines — all sed-able) plus one tool run, my recommendation is **yes to the rename**:
you proposed the name yourself, and after Step 13 the directory is honestly NURBS.
Do it as the **final commit of Phase 3**, with `nurbs.pyx` → `_core.pyx` in the same
commit per the spec. Two hard pins from the packaging audit:

- **`implicit/` must land at `mmcore/implicit/`, NOT the repo root.** `RESTRUCTURE.md`
  says "top-level `implicit/`" ambiguously (`:329`, `:1183`); pyproject has **no
  packaging directives at all** (poetry-core auto-discovers the single `mmcore/`
  package), so a repo-root `implicit/` would be **silently dropped from wheel and
  sdist** with no error.
- Step 13 must rewrite one build entry: `mmcore.geom.implicit.tree.cbuild_tree3d`
  (`build.py:206-212`, both dotted name and source path) — and delete its stale
  `.c`/3×`.so` at the old path. `bvh/` and `octree.py` have no build entries; that half
  is a pure-Python move.

### NEW-Q3 — `tests/test_nurbs_compose.py`: **rewrite against sbern's real API (~1 hour) — §7.1's premise is REFUTED**

Three load-bearing claims in §7.1 are each contradicted by the file's own body:

- "the 3 sbern imports are unused" — false: `nurbs_bezier_to_bern` is called at
  `:232,233,274,275,316,317`, `compose_curve_curve` + `bern_to_nurbs_bezier` at
  `:236,278,320`. The whole `TestNURBSComposition` class (3 tests) already targets
  sbern and **passes 3/3** with only the import line restored — the quarantining
  commit (`aa818c2`) renamed the call sites to sbern's current names and then deleted
  the import it needed. The suite is broken by a deleted import line, not by a missing
  capability.
- "all 15 tests exercise `_nurbs_compose` symbols only" — it has **12** tests, not 15,
  and all 12 were run **green** via a 34-line shim over existing sbern primitives
  (proof: scratchpad `test_full_rewrite.py`, `12 passed in 0.45s`).
- Per-symbol: 2 of the 7 missing names were **dead imports** never called
  (`compose_nurbs_curves`, `compose_bezier_segments` — equivalents exist at
  `sbern.py:709` and `:410`); 4 are 3-to-6-line derivations over `compose_curve_curve`,
  `decompose_curve`, and sbern's private helpers (`_segment_interval:364`,
  `_roots_against_constant:369`, `_collect_split_parameters:392`); `BezierSegment` is a
  6-line attribute bag.

**Action:** rewrite the suite in place against sbern's real API (~40 lines added /
~25 removed, test-file only, zero production change), un-skip, and correct §7.1.
The suite is *not* redundant with `test_nurbs_curve_compose.py` (14 tests, green) —
it uniquely pins the scalar composition algebra against closed form (t²∘t² = t⁴), the
power-basis root finder, breakpoint finding, Bézier span extraction, and the
`nurbs_bezier_to_bern`/`bern_to_nurbs_bezier` round trip with non-unit weights.
Two small judgement calls for you, defaults proposed: keep the derivations as test-local
helpers (don't promote sbern's `_`-helpers to public until a second consumer appears),
and keep sbern's strictly-interior breakpoint convention (endpoints are one
`_segment_interval` line away; the open convention is correct for `split_curve_multiple`).

### NEW-Q4 — `tests/test_newton.py`: **REPAIR — a 2-line test fix, verified green**

The `IndexError` is stale-fixture index drift, not API drift: commit `1c49fd6` deleted
the first (already-dead, Coons-based) entry of `_test_data.ssx`, shifting indices;
old `ssx[2]` is exactly today's `ssx[1]` (same variables, same surfaces). Fix:

- `tests/test_newton.py:7` — `ssx_cases[2][0]` → `ssx_cases[1][0]` (verified 3/3 green;
  index 0 is the *wrong* restore — non-convergence → TypeError);
- `:34` — `res1` → `res2` (the assertion at `:36` is currently vacuously true).

Both `cnewton.pyx` (9 importer files, built, reached from every public solver entry)
and `fdm.py` (14 importer files, the documented `Implicit2D.dxdy` path) **fail the
six-channel DELETE predicate** — deleting the test would strand a built extension with
zero coverage, the exact cbern failure mode. **Follow-up ledger item, not a restructure
step:** after Phase 2's deletions, cnewton's surviving *call sites* all lose their
consumers (`bern_roots_1d`, `find_all_minima`, `pull_curve`, and a never-called import
in `closest_point.py`; none of the six live engines touch cnewton/fdm) — schedule its
own measured keep-or-delete decision later, with this repaired test as the instrument.

---

## 3. Corrections to fold into `RESTRUCTURE.md` (Step 0.1)

These matter because the §7 step expectations are the pass/fail gates; wrong expected
values will fail good steps or pass bad ones.

| Where | Correction |
|---|---|
| Step 5 (`:1085`) | expected import failures **11 → 5**, not 11 → 6 — `dqr.py` is a *sixth* deletion in the step (fails on `geom.surfaces`, not `geom.curves`) |
| Step 6 (`:1094`) | expected **5 → 1**, not 6 → 2; sole remainder `certified_proj.py`. **`numeric/interval/solver.py` imports cleanly at HEAD** — it is not among the 17, so Q4's delete changes the import count by zero (delete it in Step 10 as dead code, not as an import repair) |
| Step 7 (`:514`, `:1114`) | blast radius **13 files / 15 lines**, not 9/10 — dropping `nurbs_csx_v2` adds 4 example files: `nurbs_nurbs_intersection_1.py:5`, `overlap_nurbs_intersection_2.py:12`, `_4_new.py:12`, `_5.py:229`; plus the (already-broken) `csx/_overlaps.py:7` edge into `_ncsx2` |
| Step 7 note | `tests/test_csx.py` is currently **GREEN** (1 passed) — deleting it drops the passed count 760 → 759; state it so the reviewer isn't surprised |
| §3.1 `_nccx` trap (`:428`) | **softer than stated** — all 3 library edges are dead-path: `csx/_bez_overlap.py` has zero importers and is itself DELETE; `_nurbs_construct.py:382` feeds only a `__main__` demo and a broken zero-consumer `network_surface` (calls undefined `nurbs_curve_bvh`); `ccx/_bez_overlap.py:190` sits in an unimported branch. `_nccx.py` can go in the same batch once those two files are deleted and `_nurbs_construct.py`'s dead tail (`construct_gordon_surface`/`network_surface` + the `:382` import) is cut — its live exports `circle`/`ruled` don't touch `_nccx` |
| Step 8 sweep hazard | `from mmcore._test_data import ssx` is a **different `ssx`** at 8 sites (`step_writer.py:748`, `gauss_map.py:495`, `_detect_intersections.py:433`, `_ssx4.py:2014`, `dqr4.py:232`, `constrained.py:35,198`, `test_newton.py:4`) — the binding-flip grep must not rewrite them |
| §6.5 / §6 line numbers | cydqr commented block is `build.py:139-146` (not 140-147); `_cubic` is `:199-206` (not 187-193); `ellipsoid` is `:235-242` (not 223-229) |
| cimport-KEEP table | `quicksort.pyx` "zero Python importers" is refuted — `gauss_map.py:8` imports `unique`; `binom` has 3 Python importers of `binomial_coefficient_py`. Both KEEPs stand, on stronger grounds |
| §7.1 | rewrite per NEW-Q3 above (12 tests not 15; the re-point IS possible; 3 tests already re-pointed) |
| §3.4-A (`:336`) / Step 14 | replace "regenerate the fixtures" with the proven lossless migration (NEW-Q2); add the 6th consumer `nurbs_ssx5_coverage_check.py:98` |
| §9.1 (`:1306`) | Phase 1 is committed at `aa818c2`, not working-tree-only |
| §6 native additions | `geom/bvh/__init__.pxd` is 0 bytes with zero cimporters — same class as `cimplicit.pyx`; add to the native DELETE list |

---

## 4. Branch decisions — BEFORE Phase 2 starts

Audited: 5 branches unmerged into `tiny`.

- **Delete now, nothing to save:** `master` (tree byte-identical to its merge-base —
  zero unique content; only hazard is a human mistaking it for the default branch),
  `dev-sfd` (2023-era scene-graph/server code; all 10 paths absent from tiny),
  `dependabot/...download-artifact-4.1.7` (superseded — tiny already uses `@v4`).
- **Delete `tiny-nurbs-periodic`:** three hard blockers (its `nurbs.pxd` imports the
  deleted `mmcore.geom.curves.knot`; its `build.py` registers a nonexistent
  `nurbsv2.pyx`; it modifies two files tiny deleted, incl. un-doing Phase-1 work), and
  today's `nurbs.pyx` already carries broader periodicity than the branch adds — its
  `.pxd` change would actually *regress* the periodic-aware span lookup. Optionally
  cherry-pick the one-line `cimport cython` in `quicksort.pxd`. It is also the only
  branch touching the Step-14 rename's cimport surface, so removing it de-risks Phase 3.
- **`claude/elegant-bardeen` — the only real decision.** Zero path collision with any
  plan step (it touches only `_deflate.py`), and its worktree has **zero uncommitted
  tracked work** — the branch tip is the whole story. But it rewrites
  `DeflatedSystem.__init__/psi_point/T_point/jac_point`, which the live engine consumes
  (`_bez_ssx5.py:811-814`) — merging it is exactly the "touch `_deflate.py`" Q13
  forbids, and a 3-way merge against tiny's own drift produces 5 conflict regions (one
  157 lines). Its `analyse_deflated_system` hunk is a *behaviour deletion* whose only
  library caller (`_ssx4.py`) dies at Step 10 anyway.
  **Recommendation:** do not let it ride with the restructure. Either (a) later, rebase
  only the four `DeflatedSystem` fast-evaluator hunks as a standalone perf change gated
  on the ssx5 suites (and prove the interval→float-midpoint path changes no result — a
  Q13-class decision in its own right), or (b) delete branch + worktree now. Park it;
  don't merge it mid-restructure.

No unmerged branch touches the Step 7/8 binding-flip files, any §3.2 merge source, or
any Step 13 mover — **Steps 7, 8, 11, 13 are branch-clean.**

---

## 5. Revised execution order

Keep the spec's Steps 5–14 with the corrected expectations; three amendments:

1. **Insert Step 0** (§1 above) before everything.
2. **One hygiene rule for every step that deletes or moves a native source:** delete the
   generated `.c` and all 3 `.so` in the same commit (`_ssx_utils.{c,so×3}` now;
   `cbuild_tree3d` at Step 13). Poetry-core packages untracked unignored files — stale
   binaries would ship in the wheel at dead import paths.
3. **Repair batch alongside Step 5** (both are "make the numbers move" fixes):
   the NEW-Q4 two-line `test_newton.py` fix and the NEW-Q3 suite rewrite. Both are
   test-only, both verified green this session, and both *raise* the passed count while
   Step 5 lowers the failure count — a clean, reviewable first commit pair on the branch.

Batch shape (one commit each, baseline command + `pytest --co` after each):

| Batch | Contents | Expected after |
|---|---|---|
| 2a | Step 5 (6 broken modules + `_ssx_utils` family + `cydqr`/`_dqr.cpp`/`dqr4`) + test repairs | real import failures 11 → **5**; passed +13 (test_newton +1, compose +12) |
| 2b | Step 6 (4 structurally broken) | 5 → **1** |
| 2c | Step 7 CSX flip (13 files / 15 lines) · Step 8 SSX flip + retire `surface_surface` + cut `_nurbs_transform.py:5` first · Step 9 `_bez_ssx6` | remaining `certified_proj` dies in 2d; `test_nurbs_ssx.py`'s 6 reds and `test_csx.py` leave with their modules |
| 2d | Step 10 batches (incl. Q4/Q5/Q9/Q12, bvh losers, `_nccx` per the softened trap) + Step 10b `ds/` dissolution | import failures = **exactly the 6 benign leaves**; DELETE pool empty |
| 2e | Step 11 substrate moves · Step 12 layering + CI gates (§7 below) | layering violations 15 → 0; CI green |
| 3 | Step 13 moves (`implicit` → `mmcore/implicit/`; `bvh`+`octree` → `numeric/`; rewrite `cbuild_tree3d` entry) → Step 14 rename + `tools/pickle_module_migrate.py` run (tracked + untracked pickles) | tree = target tree; no `mmcore.geom` left |

---

## 6. Documentation phase — one amendment to §9

Everything in §9 stands (README, `docs/ARCHITECTURE.md` with the two traps + layering
predicate + ledger convention), with one change from the file's primary consumer:

**Do not delete `CLAUDE.md` — replace it with a thin tracked one.** Claude Code loads
`CLAUDE.md` automatically at session start; `ARCHITECTURE.md` is not auto-loaded.
Deleting it means every future session starts blind or re-discovers the two traps the
hard way. A ~20-line tracked `CLAUDE.md` — build/test commands, "tuple ABI is the
public representation; `NURBSCurveTuple`/`NURBSSurfaceTuple` from `_nurbs_eval`", the
dual-type `.knot`/`.knots` note, `tiny` is main, "read `docs/ARCHITECTURE.md` first" —
is strictly better than none and stays reviewable in git. The current stale untracked
CLAUDE.md should indeed die; the *slot* should not.

Also in this phase (cheap, found by the packaging audit): fix the dangling entry point
`rhino = "mmcore.rhino"` → `"mmcore.extras.rhino"` (`pyproject.toml:67` — the target
does not exist today and the bad path is baked into installed metadata), and either fix
or drop the cosmetic `[tool.poetry.extras]` names that poetry-core silently discards.

---

## 7. CI — a gap the spec doesn't cover

Measured: **CI never runs pytest** (three workflows: poetry-build / upload-to-pypi /
Docker; push-only triggers, no `pull_request:`). Worse, a live release defect:
**`.github/workflows/upload_pypi.yml` publishes to PyPI even when the build FAILED** —
its `workflow_run` trigger has `types: [completed]` with no
`conclusion == 'success'` guard (and it duplicates poetry-build's own publish job, with
a broken artifact-name reference at `:31`). Any gate added is meaningless until this
workflow is deleted or guarded — **do that first, in Step 0 if you like.**

Then, at the end of Phase 2 (Step 12), add three steps to the existing `build` job of
`poetry-build.yml`, after `poetry build` (so the fresh `.so` are in-tree), on the
`ubuntu-latest` / one-Python leg only (~16 min once, not 15×):

1. `poetry run python -m pytest tests -m "not slow"` — must be `python -m pytest` from
   the repo root: there is no `conftest.py`, and fixtures are CWD-relative
   (`test_nccx4.py:71`, `test_nssx5.py` → `examples/…`);
2. `tools/check_imports.py` — the §7 baseline script as a tracked file, asserting the
   failure set is a **subset of** the 6 benign optional-dep leaves (never
   `len == 6`: a run with extras installed legitimately fixes 5 of them);
3. `tools/check_layering.py` — the ~40-line ast check; enable only at the end of
   Phase 2 (it fails today by design), with a layer assignment for the new
   `mmcore/implicit/` after Step 13.

Add `pull_request: branches: [tiny]` to the trigger, and create `tools/` (doesn't
exist yet — `pickle_module_migrate.py` is its first resident).

---

## 8. Out of scope — unchanged, restated so nothing rides along

Q13 (deflation subfamilies E/F/G), Q14 (alpha-sizing), Q15 (`trace_limit=512` — a
derived-envelope ledger item for the engine, not a restructure step), and the
`elegant-bardeen` perf work (parked behind Q13). The restructure must not touch
`_bez_ssx5.py` or `_deflate.py` beyond the mechanical import-path edits of Phase 3.

## 9. Session work products — preserved in-repo (untracked, review before committing)

- `tools/pickle_module_migrate.py` — the proven migration tool (NEW-Q2).
- `tools/test_full_rewrite.py` — the 12/12-green compose rewrite; starting point for
  NEW-Q3 (becomes the new `tests/test_nurbs_compose.py` body after review).
- `tools/test_newton_fixed.py` — the repaired newton test (NEW-Q4); a 2-line diff
  against `tests/test_newton.py`.

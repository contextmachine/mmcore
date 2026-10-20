# CLAUDE.md

Guidance for Claude Code in this repository. **Read `docs/ARCHITECTURE.md` first** —
it states the layer rule, the solver shape, and the dual-type NURBS ABI.

## Commands

```bash
poetry install                      # dependencies
python build.py                     # build Cython extensions (MMCORE_DEBUG_TRACE=1 for profiling)
python -m pytest tests -m "not slow"   # suite, from the REPO ROOT (fixtures are CWD-relative)
python tools/check_imports.py       # import-health gate (subset of 6 benign extras leaves)
python tools/check_layering.py      # no upward import edges
python clean.py                     # clean build artifacts
```

Local venv: `.venv/bin/python` (Python 3.14). Main branch: `tiny`.

## Key facts

- **Primary NURBS representation:** `NURBSCurveTuple` / `NURBSSurfaceTuple` from
  `mmcore.nurbs._nurbs_eval` (named tuples — readable, debuggable). The Cython classes
  live in `mmcore.nurbs._core`; both are first-class inputs and spell their knots field
  differently (`.knot` vs `.knots`) — see the ABI table in ARCHITECTURE.md.
- **Solver entry points are unversioned:** `nurbs_ccx`, `nurbs_csx`, `nurbs_ssx` from
  `mmcore.numeric.intersection.{ccx,csx,ssx}` always bind the maintained engine.
  Never add `_vN` public names or numbered sibling modules.
- **Layers:** nurbs/implicit/numeric (L0) → numeric/intersection (L1) →
  construction/topo/compat (L2) → extras (L3). No upward imports; CI enforces it.
- **Governance:** decisions and defects are ledger IDs (`L##`) in commit subjects and
  `docs/superpowers/` issue/plan docs — not in code comments, docstrings or TODOs. Code
  behavior at HEAD outranks docs, docs outrank external notes.
- **Numerics design law:** no bare thresholds in residual/classification predicates —
  derive envelopes from operands.
- mmcore does not guarantee backwards compatibility; no compat shims or deprecation
  aliases — flip bindings and delete superseded code instead.

## Working rules

- **Comments are not evidence.** Comments and docstrings may be stale, unapproved or
  wrong. Establish facts from code behavior and runs, not from comments.
- **Do not add comments to code.** When a change makes an existing comment or docstring
  false, delete it.
- **Count intersections as CCX does (L62).** The topology and the number of
  intersection points are determined precisely: no unpredictable extra points within
  `atol` of an intersection, and a curve that crosses another twice gives two points
  even when they are closer than `atol`.
- **One concrete question per decision.** Ask design questions one at a time, as a
  question about observable output on an example ("two records at a seam?"), with a
  recommendation and measured evidence. Never re-offer alternatives to a settled contract.

## Settled contracts

Recorded in `docs/superpowers/issues/2026-10-20-csx-overlap-records-ssx-empty-clip.md`.

- **CSX output (L63).** A point is `(t, u, v)`. An overlap is a pair of `(t, u, v)`: its
  start and its end in curve order — `t_range`, `u_range`, `v_range` hold paired ends,
  never per-axis min/max. Overlaps of any shape are reported; there is no affine-only
  restriction. At a periodic seam: a curve lying on the seam gives two records (one per
  side, same `t_range`); a curve crossing the seam is split exactly at the crossing `t`.
- **SSX loop-free test (L64).** Each cell is first shrunk to the box where the two
  surfaces can meet; a cell whose box is empty is skipped. Otherwise TΨ monotonicity and
  the Gauss-map test run on the whole cell, then on the shrunk box; a cell passing
  neither is split. Checked on 40 cases: the skip loses nothing SSX found before.
- **Judging results.** Every reported point must lie within the caller's `atol` of both
  operands; that tests accuracy, not what to report. An intersection curve returned in
  pieces (split at a seam or a knot line) is an error, and so is the same overlap or
  root returned twice. CSX seam records (L63) describe parts of the input curve, so they
  are not a split curve.

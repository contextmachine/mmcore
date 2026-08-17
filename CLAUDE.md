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
  module docstrings, not TODOs. `docs/superpowers/` holds specs/plans/issues; code at
  HEAD outranks docs, docs outrank external notes.
- **Numerics design law:** no bare thresholds in residual/classification predicates —
  derive envelopes from operands.
- mmcore does not guarantee backwards compatibility; no compat shims or deprecation
  aliases — flip bindings and delete superseded code instead.

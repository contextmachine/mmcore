# mmcore — Architecture

This document states the load-bearing conventions of the codebase: the layer rule,
the solver shape, the dual-type NURBS ABI, and how decisions are tracked. It exists so
that the shape of the package is explicit rather than re-discovered by every reader.
(The 2026-08 restructure that produced this tree is documented in
`docs/RESTRUCTURE.md` and executed in `docs/superpowers/plans/2026-08-16-restructure-phase2-execution.md`.)

## The tree

```
mmcore/
├── nurbs/          L0 · the NURBS core
│   ├── _nurbs_eval.py      NURBSCurveTuple / NURBSSurfaceTuple — the tuple ABI (primary)
│   ├── _nurbs_knots.py     knot algebra: insertion, refinement, decompose, join
│   ├── _nurbs_ders.py  _nurbs_interp.py  _nurbs_join.py  _nurbs_construct.py
│   ├── _nurbs_transform.py  _nurbs_param_tol.py  nurbs_iso.py
│   ├── _core.pyx/.pxd      C++ NURBSCurve / NURBSSurface (Cython accelerator)
│   ├── parametric.pyx/.pxd knots.pxd  primitives.pyx     C base layer (cimport web)
│   └── _nurbs.cpp  portability.h                          handwritten C++ core
├── implicit/       L0 · implicit geometry + dual contouring
├── numeric/        L0 · numeric substrate
│   ├── bern.py  sbern.py  bern_sq_dist.py  _bern_homog.pyx   Bernstein bases + sq-dist nets
│   ├── _bezier_common.py  _bern_zero_1d.py  _work_budget.py  shared solver substrate
│   ├── closest_point.py  _bez_closest_point.py
│   ├── bvh/  octree.py                                       spatial acceleration
│   ├── newton/  interval/  integrate/  plane/  algorithms/
│   └── intersection/   L1 · the solver families
│       ├── ccx/   __init__ → ccx.py → _nccx4.py → _bez_ccx4.py
│       ├── csx/   __init__ → _ncsx4.py → _bez_csx4.py  (+ _cbez_csx.pyx)
│       └── ssx/   __init__ → _nssx5.py → _bez_ssx5.py  (+ _ssx_substrate.py)
├── construction/   L2 · high-level builders (ruled, revolved, sweep, torus, …)
├── topo/           L2 · BRep topology (Euler operators), meshing
├── compat/step/    L2 · STEP I/O
└── extras/         L3 · optional leaf integrations (renderer, rhino, occ, torch)
```

## The layer rule

> `layer(importer) >= layer(imported)` must hold for every import edge in `mmcore/`.
> Equivalently: no module imports from a higher layer, and `extras/` has zero inbound edges.

This is mechanically enforced: `python tools/check_layering.py` exits nonzero on any
upward edge (imports under `if __name__ == "__main__":` are script demos and exempt).
It runs in CI together with `tools/check_imports.py`, which asserts that the only
modules allowed to fail import are the six optional-dependency leaves under `extras/`.

## The solver shape — one entry point per family, no version suffixes

Each intersection family has exactly three tiers:

1. **`_bez_*` engine** — operates on Bernstein/Bézier cells (homogeneous control nets),
   owns the numerics.
2. **`_n*` adapter** — decomposes NURBS into cells, calls the engine, assembles global
   results, owns work budgets and typed status.
3. **package entry** — `from mmcore.numeric.intersection.ssx import nurbs_ssx` (same for
   `ccx`/`csx`). The package `__init__` always binds the maintained engine.

Two conventions are law here, because their violation cost real time historically:

- **No version-suffixed public names.** `nurbs_ssx_v5`-style names are the disease:
  they let the good name point at dead code. If an engine is superseded, the package
  binding flips and the old generation is deleted — not kept beside its successor.
- **Filename numbers do not encode recency.** When a numbered sibling of a live module
  appears (`_bez_ssx6` was an older frozen fork of `_bez_ssx5`), the module docstring
  is the authority. Prefer not to create numbered siblings at all.

## The dual-type NURBS ABI

Two curve/surface representations are first-class inputs across the codebase, and they
spell their fields differently:

| | representation | knots field | update style |
|---|---|---|---|
| `NURBSCurveTuple` (`nurbs/_nurbs_eval.py`) | named tuple — **primary** | `.knot` | `._replace(...)` |
| `NURBSCurve` (`nurbs/_core.pyx`) | Cython class — accelerator | `.knots` | settable properties |

Code that accepts "a curve" must accept both and return the same representation it was
given. `nurbs/_nurbs_knots.py` has the canonical helpers (`_get_knots`,
`_replace_curve`) — reuse them rather than re-deriving the dispatch. A `.knot`/`.knots`
mix-up survived undetected in a live L0 module for months; this table is why it won't again.

## Governance

- **Defects and decisions live as ledger IDs (`L##`)**, in commit subjects and module
  docstrings — not as TODO comments. To learn why something exists, grep the ledger ID.
- **Evidence precedence when sources disagree:** the code and git history at HEAD win
  over `docs/superpowers/{specs,plans,issues}`, which win over any external session
  notes. Say it once here so nobody rediscovers it by contradiction.
- **Design law for numerics:** no bare thresholds in residual/classification
  predicates — envelopes are derived from the operands they judge
  (see the derived-envelope program in `docs/superpowers/`).

## Tests and fixtures

- `pytest tests` from the **repo root** (fixture paths are CWD-relative; there is no
  conftest.py). One test is marked `slow` (~29 min); CI runs `-m "not slow"`.
- The tracked ssx fixtures under `examples/ssx/*.pkl` embed class paths by module name.
  If a module path they reference is ever renamed again, migrate them with
  `tools/pickle_module_migrate.py` (lossless load-with-remap + re-dump; never re-run
  the engine to regenerate a fixture).

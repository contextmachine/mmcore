# SSX v5 P1 — Whole-Call Normalization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `bez_ssx` translation/scale-invariant by running its entire search in a canonical frame (joint center, power-of-2 scale) and un-mapping xyz outputs exactly once at exit, so case 6 certifies its complete curve at original coordinates and atol=1e-3.

**Architecture:** Three new module-level helpers in `_bez_ssx5.py` (`_ssx_normalization_context`, `_normalize_surface_net`, `_denormalize_result`) plus a ~8-line entry preamble and a one-line hook in the existing `_result` closure — the single choke point every return path already goes through. No certificate, tolerance, or ladder arithmetic changes anywhere.

**Tech Stack:** Python 3.14 (`.venv/bin/python`), numpy, pytest. Engine: `mmcore/numeric/intersection/ssx/_bez_ssx5.py` (7,754 lines). macOS: NO `timeout` command.

> **AS-BUILT DEVIATION (2026-07-21, user-approved):** wiring the
> unconditional frame regressed 4 near-origin singular fixtures (Task 3
> implementer correctly BLOCKED). Root-caused by variant experiment
> (scale-half: absolute singular-tier thresholds; center-half:
> exact-structure rounding) and fixed by an **identity window**
> `_NORM_IDENTITY_WINDOW = [2⁻⁵, 2⁵]` on joint coordinate magnitude in
> `_ssx_normalization_context` — identity inside, normalize outside.
> Task 1's helper + tests and the spec were amended accordingly; the
> singular-tier follow-up is filed as P1b
> (`docs/superpowers/issues/2026-07-21-ssx5-p1b-singular-tier-scale-invariance.md`).
>
> **AS-BUILT DEVIATION 2 (2026-07-21, Task-6 BLOCKED finding):** the
> O(1)-target frame silently fragmented bez-harness case 10 (transversal,
> out-of-window) — cartography bracketed the safe regime at post-frame
> magnitude ≳ 5, and a scale-only attempt failed offset-dominated
> extremes. Shipped synthesis: centered + **target-band scale**
> `k = 2^round(log2(diag/16))` landing post-center magnitude in
> [5.66, 11.31] (spec Amendment 2). The silent-completeness accounting
> gap is filed as P1c
> (`docs/superpowers/issues/2026-07-21-ssx5-p1c-silent-fragment-completeness.md`).

**Authority documents (read before starting):**
- Spec: `docs/superpowers/specs/2026-07-21-ssx5-invariance-normalization-design.md`
- Kickoff (probe evidence, anchors, gates): `docs/superpowers/plans/2026-07-20-ssx5-invariance-kickoff.md`
- Invariant: **never fix by loosening.** If a gate fails, debug with the kickoff's probes; do not touch tolerances, certificates, or the ladder.

**Verified facts this plan relies on (checked 2026-07-21 at `2bd5787`):**
- `bez_ssx` starts at `_bez_ssx5.py:5924`; `S1 = np.asarray(S1, dtype=np.float64)` and the `S2` twin sit immediately after the docstring, before `budget = _SSXSoftBudget(...)`.
- The `_result` closure (`:5982`) builds the dict for EVERY return path (`return {` hits inside `_run_csx` are a nested CSX adapter, not `bez_ssx` returns).
- `SSXBranch.curve` in the v5 engine is a **tuple `(stuv_path, xyz_path)`** of polyline arrays (constructed at `:4367,:5230,:5379,:5395`); the `curve_xyz/curve_st/curve_uv` dataclass fields stay `None` in the v5 path (the adapter reads `b.curve[0]`/`b.curve[1]` directly — `_nssx5.py:393-403,718,775,1249`).
- `unresolved_regions` entries are `{'stuv_min', 'stuv_max', 'reason'}` dicts — parameter-space only, **nothing to un-map** (audit closed).
- `SSXPoint.xyz` and `SSXSingularity.xyz` are the only other xyz payloads; `stuv`, `stuv_mate`, `samples` (N,4), `branch_links`, region uv-loops, `interior_stuv`, and certification residuals (atol units) are invariant.
- `max_xyz_step` is consumed once (`:6265`) and is an xyz length → must scale. All other kwargs are counts/depths.
- Reason constants live in `mmcore/numeric/_work_budget.py` (`REASON_TRACE_UNVERIFIED = "trace_unverified"`).
- Fixtures: `examples/ssx/nurbs_nurbs_intersection_{6,11}.pkl` committed; loader idiom `pickle.load(f)[0]` → `(s1, s2)`.

---

### Task 1: Normalization context + net transform helpers

**Files:**
- Modify: `mmcore/numeric/intersection/ssx/_bez_ssx5.py` (insert after `_strict_ssx_root_tol`, which ends at `:1970`)
- Create: `tests/test_bez_ssx5_invariance.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_bez_ssx5_invariance.py`:

```python
"""P1 whole-call normalization: helpers + invariance property (2026-07-21 design).

Spec: docs/superpowers/specs/2026-07-21-ssx5-invariance-normalization-design.md
"""
import numpy as np
import pytest

from mmcore.numeric.intersection.ssx._bez_ssx5 import (
    _ssx_normalization_context,
    _normalize_surface_net,
)


def _homog(S):
    S = np.asarray(S, dtype=np.float64)
    return np.concatenate([S, np.ones(S.shape[:-1] + (1,))], axis=-1)


def test_context_power_of_two_scale_and_center():
    # Joint AABB [0,10]^3 -> diag = 10*sqrt(3) ~ 17.32, log2 ~ 4.11 -> k = 16.
    s1 = np.array([[[0., 0., 0.], [0., 10., 0.]], [[10., 0., 0.], [10., 10., 0.]]])
    s2 = np.array([[[0., 0., 10.], [0., 10., 10.]], [[10., 0., 10.], [10., 10., 10.]]])
    c, k = _ssx_normalization_context(s1, s2, rational=False)
    assert k == 16.0
    assert np.allclose(c, [5.0, 5.0, 5.0])
    # k is a power of two: scaling is mantissa-exact and reversible bit-for-bit.
    rng = np.random.default_rng(3)
    pts = rng.uniform(-1e4, 1e4, (64, 3))
    assert np.array_equal((pts / k) * k, pts)


def test_context_rational_uses_dehomogenized_points():
    s = np.array([[[0., 0., 0.], [0., 4., 0.]], [[4., 0., 0.], [4., 4., 0.]]])
    h = _homog(s)
    h2 = h.copy()
    h2[..., :3] *= 2.0   # same Cartesian points, w-scaled numerators would differ
    h2[..., 3] *= 2.0
    c1, k1 = _ssx_normalization_context(h, h, rational=True)
    c2, k2 = _ssx_normalization_context(h2, h2, rational=True)
    assert np.allclose(c1, c2) and k1 == k2


def test_context_degenerate_inputs_yield_identity():
    good = _homog(np.zeros((2, 2, 3)))
    bad_w = good.copy(); bad_w[0, 0, 3] = 0.0
    c, k = _ssx_normalization_context(bad_w, good, rational=True)
    assert k == 1.0 and np.all(c == 0.0)
    bad_nan = np.zeros((2, 2, 3)); bad_nan_ = bad_nan.copy(); bad_nan_[0, 0, 0] = np.nan
    c, k = _ssx_normalization_context(bad_nan_, bad_nan, rational=False)
    assert k == 1.0 and np.all(c == 0.0)
    # Zero extent (all points coincide) -> identity, per spec.
    pt = np.full((2, 2, 3), 7.0)
    c, k = _ssx_normalization_context(pt, pt, rational=False)
    assert k == 1.0 and np.all(c == 0.0)


def test_normalize_surface_net_round_trip():
    rng = np.random.default_rng(11)
    s = rng.uniform(2350.0, 3200.0, (3, 4, 3))
    c, k = _ssx_normalization_context(s, s, rational=False)
    n = _normalize_surface_net(s, c, k, rational=False)
    # Normalized coords are O(1) and the map inverts to roundoff at world scale.
    assert np.max(np.abs(n)) <= 2.0
    assert np.allclose(n * k + c, s, atol=1e-9)
    assert not np.shares_memory(n, s)


def test_normalize_surface_net_rational_preserves_cartesian_points():
    from mmcore.numeric.intersection._bezier_common import eval_surface
    rng = np.random.default_rng(5)
    s = rng.uniform(900.0, 1100.0, (3, 3, 3))
    h = _homog(s)
    h[..., 3] = rng.uniform(0.5, 2.0, (3, 3))   # non-unit weights,
    h[..., :3] = s * h[..., 3:]                 # numerators kept consistent
    c, k = _ssx_normalization_context(h, h, rational=True)
    n = _normalize_surface_net(h, c, k, rational=True)
    for (u, v) in [(0.0, 0.0), (0.3, 0.7), (1.0, 1.0)]:
        pw = eval_surface(h, u, v, rational=True)
        pn = eval_surface(n, u, v, rational=True)
        assert np.allclose(pn * k + c, pw, atol=1e-9)
    # Weights are frame-invariant: untouched by the transform.
    assert np.array_equal(n[..., 3], h[..., 3])
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_bez_ssx5_invariance.py -x -q`
Expected: FAIL at import — `cannot import name '_ssx_normalization_context'`.

- [ ] **Step 3: Implement the helpers**

In `_bez_ssx5.py`, directly after `_strict_ssx_root_tol` (after `:1970`), add:

```python
def _ssx_normalization_context(S1, S2, rational=True):
    """Canonical-frame context (c, k) for the whole-call preamble.

    P1 invariance (2026-07-21 design): every absolute roundoff envelope in
    this module (the strict Psi-zero certificates, the 1e-14 corrector
    stops) is correct only for O(1) coordinates.  The whole search
    therefore runs in a frame jointly centered at the two nets' Cartesian
    AABB midpoint and scaled by the AABB diagonal snapped to a power of
    two — the snap makes the scale divide mantissa-exact, so only the
    one-time centering multiply-subtract rounds at all.  Degenerate input
    (zero/non-finite weight, non-finite point, zero extent) falls back to
    the identity frame: the pipeline then behaves exactly as before P1.
    """
    identity = (np.zeros(3, dtype=np.float64), 1.0)
    corners = []
    for S in (S1, S2):
        S = np.asarray(S, dtype=np.float64)
        if rational:
            w = S[..., -1:]
            if not np.all(np.isfinite(w)) or np.any(w == 0.0):
                return identity
            pts = S[..., :-1] / w
        else:
            pts = S
        if not np.all(np.isfinite(pts)):
            return identity
        corners.append(pts.reshape(-1, 3))
    pts = np.vstack(corners)
    lo = pts.min(axis=0)
    hi = pts.max(axis=0)
    diag = float(np.linalg.norm(hi - lo))
    if not np.isfinite(diag) or diag <= 0.0:
        return identity
    c = 0.5 * (lo + hi)
    k = float(2.0 ** round(math.log2(diag)))
    return c, k


def _normalize_surface_net(S, c, k, rational=True):
    """Map a control net into the canonical frame: x' = (x - c) / k.

    Rational nets transform homogeneously (numerator -= c*w, then /k), so
    Cartesian points map exactly as above while weights stay untouched.
    Always returns a copy; the caller's world-frame net is never mutated.
    """
    S = np.asarray(S, dtype=np.float64).copy()
    c = np.asarray(c, dtype=np.float64)
    if rational:
        S[..., :-1] -= c * S[..., -1:]
        S[..., :-1] /= k
    else:
        S -= c
        S /= k
    return S
```

`import math` already exists at `_bez_ssx5.py:21` — nothing to add.

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_bez_ssx5_invariance.py -x -q`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/intersection/ssx/_bez_ssx5.py tests/test_bez_ssx5_invariance.py
git commit -m "feat(ssx5): canonical-frame context + net transform helpers (P1 normalization)"
```

---

### Task 2: `_denormalize_result` — the exactly-once un-map

**Files:**
- Modify: `mmcore/numeric/intersection/ssx/_bez_ssx5.py` (directly after `_normalize_surface_net` from Task 1)
- Test: `tests/test_bez_ssx5_invariance.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_bez_ssx5_invariance.py`:

```python
# SSXBranch/SSXPoint are re-exported through _bez_ssx5's namespace (:39-40).
from mmcore.numeric.intersection.ssx._bez_ssx5 import (
    _denormalize_result,
    SSXSingularity,
    SSXBranch,
    SSXPoint,
)


def _fake_result(branches=(), points=(), singularities=()):
    return {
        'branches': list(branches), 'points': list(points),
        'singularities': list(singularities), 'overlap_regions': [],
        'unresolved_regions': [], 'complete': True,
        'status': {'reasons': [], 'work': {}},
    }


def test_denormalize_maps_all_xyz_payloads_once():
    c, k = np.array([100.0, -50.0, 7.0]), 8.0
    stuv = np.array([[0.1, 0.2, 0.3, 0.4], [0.5, 0.5, 0.5, 0.5]])
    xyz_n = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    b = SSXBranch(curve=(stuv, xyz_n.copy()))
    p = SSXPoint(stuv=stuv[0].copy(), xyz=xyz_n[0].copy())
    s = SSXSingularity(kind='tangent_point', stuv=stuv[1].copy(),
                       xyz=xyz_n[1].copy())
    r = _denormalize_result(_fake_result([b], [p], [s]), c, k)
    assert np.allclose(r['branches'][0].curve[1], xyz_n * k + c)
    # stuv is parameter-space: bit-identical, same object.
    assert r['branches'][0].curve[0] is stuv
    assert np.allclose(r['points'][0].xyz, xyz_n[0] * k + c)
    assert np.allclose(r['singularities'][0].xyz, xyz_n[1] * k + c)
    assert np.allclose(r['singularities'][0].stuv, stuv[1])


def test_denormalize_identity_is_noop_same_objects():
    xyz = np.array([[1.0, 2.0, 3.0]])
    b = SSXBranch(curve=(np.zeros((1, 4)), xyz))
    r = _denormalize_result(_fake_result([b]), np.zeros(3), 1.0)
    assert r['branches'][0].curve[1] is xyz


def test_denormalize_aliased_object_mapped_once():
    c, k = np.array([10.0, 0.0, 0.0]), 2.0
    p = SSXPoint(stuv=np.zeros(4), xyz=np.array([1.0, 1.0, 1.0]))
    r = _denormalize_result(_fake_result(points=[p, p]), c, k)  # same object twice
    assert np.allclose(r['points'][0].xyz, [12.0, 2.0, 2.0])
    assert r['points'][1] is r['points'][0]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `.venv/bin/python -m pytest tests/test_bez_ssx5_invariance.py -x -q`
Expected: FAIL at import — `cannot import name '_denormalize_result'`.

- [ ] **Step 3: Implement**

Add after `_normalize_surface_net`:

```python
def _denormalize_result(result, c, k):
    """Map every xyz payload of a bez_ssx result back to world frame, once.

    Un-map inventory (2026-07-21 design §3, audited): branch polylines
    (``SSXBranch.curve = (stuv, xyz)`` — xyz only), ``SSXPoint.xyz``,
    ``SSXSingularity.xyz``.  Everything else is parameter-space (stuv,
    cusp-curve samples, region uv loops, unresolved-region stuv boxes) or
    atol-relative (region certification residuals) and must NOT be touched.
    Out-of-place arrays plus an id() guard make the map exactly-once even
    if one object is referenced twice.  Called only at the `_result` choke
    point, immediately before returning to the caller.
    """
    if k == 1.0 and not np.any(c):
        return result
    c = np.asarray(c, dtype=np.float64)
    seen = set()

    def _once(obj):
        if id(obj) in seen:
            return False
        seen.add(id(obj))
        return True

    for b in result['branches']:
        if _once(b):
            stuv, xyz = b.curve
            b.curve = (stuv, np.asarray(xyz, dtype=np.float64) * k + c)
    for p in result['points']:
        if _once(p):
            p.xyz = np.asarray(p.xyz, dtype=np.float64) * k + c
    for s in result['singularities']:
        if _once(s):
            s.xyz = np.asarray(s.xyz, dtype=np.float64) * k + c
    return result
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `.venv/bin/python -m pytest tests/test_bez_ssx5_invariance.py -x -q`
Expected: 8 passed.

- [ ] **Step 5: Commit**

```bash
git add mmcore/numeric/intersection/ssx/_bez_ssx5.py tests/test_bez_ssx5_invariance.py
git commit -m "feat(ssx5): exactly-once xyz un-map for bez_ssx results (P1 normalization)"
```

---

### Task 3: Wire the preamble into `bez_ssx`

**Files:**
- Modify: `mmcore/numeric/intersection/ssx/_bez_ssx5.py:5972-5997` (entry + `_result` closure)
- Test: `tests/test_bez_ssx5_invariance.py`

- [ ] **Step 1: Write the failing test (world-in/world-out at offset coordinates)**

Append to `tests/test_bez_ssx5_invariance.py`:

```python
from mmcore.numeric.intersection.ssx._bez_ssx5 import bez_ssx


def test_bez_ssx_world_in_world_out_offset_planes():
    # Plane pair from the singular suite, pushed to case-11-like offsets.
    # z=5 sheet vs 0->10 ramp: intersection line x=5, z=5, y in [0,10].
    off = np.array([1e4, -2e4, 3e3])
    s1 = np.array([[[0., 0., 5.], [0., 10., 5.]],
                   [[10., 0., 5.], [10., 10., 5.]]]) + off
    s2 = np.array([[[0., 0., 0.], [0., 10., 0.]],
                   [[10., 0., 10.], [10., 10., 10.]]]) + off
    r = bez_ssx(s1, s2, 1e-3, rational=False)
    assert r['complete'], r['status']['reasons']
    assert len(r['branches']) == 1
    xyz = np.asarray(r['branches'][0].curve[1], dtype=float)
    assert np.all(np.abs(xyz[:, 0] - (off[0] + 5.0)) <= 5e-3)
    assert np.all(np.abs(xyz[:, 2] - (off[2] + 5.0)) <= 5e-3)
    assert xyz[:, 1].min() <= off[1] + 0.5 and xyz[:, 1].max() >= off[1] + 9.5
    stuv = np.asarray(r['branches'][0].curve[0], dtype=float)
    assert stuv.min() >= -1e-9 and stuv.max() <= 1.0 + 1e-9
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m pytest tests/test_bez_ssx5_invariance.py::test_bez_ssx_world_in_world_out_offset_planes -x -q`
Expected: FAIL — before the fix this offset regime loses the strict certificate (`complete` False / `trace_unverified`) or mis-certifies. **Record the actual pre-fix failure mode in the commit message.** (If it unexpectedly PASSES, the plane pair is too benign — do not weaken the assert; the property test in Task 4 and the case-6 gate in Task 5 are the load-bearing checks. Continue.)

- [ ] **Step 3: Insert the preamble and the un-map hook**

In `bez_ssx`, the current entry (`:5972`) is:

```python
    S1 = np.asarray(S1, dtype=np.float64)
    S2 = np.asarray(S2, dtype=np.float64)
    budget = _SSXSoftBudget(
```

Replace the first two lines with:

```python
    S1 = np.asarray(S1, dtype=np.float64)
    S2 = np.asarray(S2, dtype=np.float64)
    # P1 invariance (2026-07-21 design): the whole search runs in a
    # canonical frame — jointly centered, power-of-two scaled — so every
    # absolute roundoff envelope below sees O(1) coordinates regardless of
    # where the model sits in world space.  World-in/world-out: `_result`
    # un-maps xyz exactly once on the way out; parameters and weights are
    # frame-invariant.  atol and max_xyz_step are lengths and scale along.
    _norm_c, _norm_k = _ssx_normalization_context(S1, S2, rational=rational)
    S1 = _normalize_surface_net(S1, _norm_c, _norm_k, rational=rational)
    S2 = _normalize_surface_net(S2, _norm_c, _norm_k, rational=rational)
    atol = float(atol) / _norm_k
    if max_xyz_step is not None:
        max_xyz_step = float(max_xyz_step) / _norm_k
```

In the `_result` closure (`:5982`), the current tail is:

```python
        result.update(budget.result_fields())
        return result
```

Change to:

```python
        result.update(budget.result_fields())
        return _denormalize_result(result, _norm_c, _norm_k)
```

Nothing else changes. Do NOT touch `_ssx_control_aabbs_disjoint`, the precompute charge, `_run_csx`, any certificate, or any tolerance — they now simply operate on normalized data.

- [ ] **Step 4: Run the new test and the fast invariance file**

Run: `.venv/bin/python -m pytest tests/test_bez_ssx5_invariance.py -x -q`
Expected: 9 passed.

- [ ] **Step 5: Run the engine regression floor**

Run: `.venv/bin/python -m pytest tests/test_bez_ssx5_singular.py -q`
Expected: 115 passed. These tests run at O(1)–O(10) coordinates where the canonical frame is nearly the identity; failures here mean the preamble broke a contract (most likely a missed un-map or a double-scaled length input) — STOP and debug with superpowers:systematic-debugging before proceeding. Never adjust a tolerance to make this suite pass.

- [ ] **Step 6: Commit**

```bash
git add mmcore/numeric/intersection/ssx/_bez_ssx5.py tests/test_bez_ssx5_invariance.py
git commit -m "feat(ssx5): whole-call canonical-frame preamble in bez_ssx (P1)

Entry: joint center + power-of-2 scale, atol/max_xyz_step scaled along.
Exit: xyz un-mapped exactly once at the _result choke point.
Pre-fix failure mode of the offset-planes test: <record what Step 2 showed>"
```

---

### Task 4: Invariance property test (kickoff gate 5 — the durable class guard)

**Files:**
- Test: `tests/test_bez_ssx5_invariance.py`

- [ ] **Step 1: Write the property test**

Append to `tests/test_bez_ssx5_invariance.py`:

```python
# ---------------------------------------------------------------------------
# Gate 5: bez_ssx(S*k + c, atol*k) must be equivalent to bez_ssx(S, atol)
# for translations up to ~1e4 and scales k in [1e-2, 1e3] (kickoff list).
# Reference runs at the pair's native coords; transformed runs place the
# model AT the offset (the defect regime).  Topology must match exactly;
# geometry must match through the map within a few atol.
# ---------------------------------------------------------------------------

def _plane_pair():
    s1 = np.array([[[0., 0., 5.], [0., 10., 5.]],
                   [[10., 0., 5.], [10., 10., 5.]]])
    s2 = np.array([[[0., 0., 0.], [0., 10., 0.]],
                   [[10., 0., 10.], [10., 10., 10.]]])
    return s1, s2, False


def _loop_pair():
    # Biquadratic bowl z = (2u-1)^2 + (2v-1)^2 (Bernstein z-coeffs [1,-1,1]
    # per axis, summed) against the plane z = 0.5: one closed transversal
    # loop strictly inside the domain.
    g = [0.0, 0.5, 1.0]
    zc = [1.0, -1.0, 1.0]
    s1 = np.array([[[g[i], g[j], zc[i] + zc[j]] for j in range(3)]
                   for i in range(3)])
    s2 = np.array([[[-0.5, -0.5, 0.5], [-0.5, 1.5, 0.5]],
                   [[1.5, -0.5, 0.5], [1.5, 1.5, 0.5]]])
    return s1, s2, False


def _rational_pair():
    # 90-degree circular-arc strip (radius 1, weights [1, sqrt(2)/2, 1])
    # extruded along y, against the plane z = 0.5: one transversal line
    # x = sqrt(3)/2 crossing the strip.  Exercises the homogeneous branch
    # of the transform.
    w = np.sqrt(2.0) / 2.0
    arc = [((1.0, 0.0), 1.0), ((1.0, 1.0), w), ((0.0, 1.0), 1.0)]
    s1 = np.zeros((3, 2, 4))
    for i, ((x, z), wi) in enumerate(arc):
        for j, y in enumerate((0.0, 1.0)):
            s1[i, j] = [x * wi, y * wi, z * wi, wi]
    s2 = np.array([[[-0.5, -0.5, 0.5], [-0.5, 1.5, 0.5]],
                   [[1.5, -0.5, 0.5], [1.5, 1.5, 0.5]]])
    s2 = np.concatenate([s2, np.ones((2, 2, 1))], axis=-1)
    return s1, s2, True


PAIRS = [("planes", _plane_pair), ("loop", _loop_pair),
         ("rational-arc", _rational_pair)]

TRANSFORMS = [
    (np.array([1e3, -2e3, 5e2]), 1.0),
    (np.zeros(3), 1e-2),
    (np.zeros(3), 1e3),
    (np.array([1e4, 1e4, -1e4]), 1e3),
    (np.array([-5e3, 3e3, 1e4]), 1e-2),
]


def _apply_world_transform(S, c, k, rational):
    S = np.asarray(S, dtype=np.float64).copy()
    if rational:
        S[..., :3] = S[..., :3] * k + np.asarray(c) * S[..., 3:]
    else:
        S = S * k + np.asarray(c)
    return S


def _pt_seg_d(p, a, b):
    ab = b - a
    den = float(np.dot(ab, ab))
    t = 0.0 if den <= 0.0 else float(np.clip(np.dot(p - a, ab) / den, 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def _poly_hausdorff(A, B):
    def directed(P, Q):
        if len(Q) == 1:
            return max(float(np.linalg.norm(p - Q[0])) for p in P)
        return max(min(_pt_seg_d(p, Q[i], Q[i + 1])
                       for i in range(len(Q) - 1)) for p in P)
    return max(directed(A, B), directed(B, A))


def _topology_signature(r):
    return (
        r['complete'],
        tuple(sorted(r['status']['reasons'])),
        tuple(sorted((b.kind, bool(b.closed)) for b in r['branches'])),
        len(r['points']),
        tuple(sorted(s.kind for s in r['singularities'])),
    )


@pytest.mark.parametrize("pair_name,make_pair", PAIRS)
@pytest.mark.parametrize("c,k", TRANSFORMS,
                         ids=[f"c{i}" for i in range(len(TRANSFORMS))])
def test_bez_ssx_similarity_invariance(pair_name, make_pair, c, k):
    atol = 1e-3
    s1, s2, rational = make_pair()
    ref = bez_ssx(s1, s2, atol, rational=rational)
    assert ref['complete'], (pair_name, ref['status']['reasons'])

    t1 = _apply_world_transform(s1, c, k, rational)
    t2 = _apply_world_transform(s2, c, k, rational)
    res = bez_ssx(t1, t2, atol * k, rational=rational)

    assert _topology_signature(res) == _topology_signature(ref), (
        pair_name, c, k, res['status'])

    # Geometry: each reference branch, mapped into the transformed frame,
    # must coincide with exactly one result branch within 10 atol_world
    # (chord-sampling differences between two float-distinct runs included).
    tol = 10.0 * atol * k
    remaining = list(range(len(res['branches'])))
    for rb in ref['branches']:
        mapped = np.asarray(rb.curve[1], dtype=float) * k + c
        dists = [(_poly_hausdorff(
            mapped, np.asarray(res['branches'][j].curve[1], dtype=float)), j)
            for j in remaining]
        d, j = min(dists)
        assert d <= tol, (pair_name, c, k, d, tol)
        remaining.remove(j)
    assert not remaining
```

- [ ] **Step 2: Run the property test**

Run: `.venv/bin/python -m pytest tests/test_bez_ssx5_invariance.py -q`
Expected: all pass (9 + 15 parametrized). Runtime target: under ~2 minutes total (the pairs are 2×2 to 3×3 nets). If any parametrization fails: this is the defect class the task exists to kill — debug with superpowers:systematic-debugging (likely a missed un-map payload or a length input not scaled); NEVER widen `tol` beyond 10·atol_world or drop a transform to pass.

- [ ] **Step 3: Sanity-check the test catches un-map misses (temporary mutation)**

Temporarily comment out the `p.xyz = ...` line for points in `_denormalize_result` — no, points may be empty in these pairs. Instead temporarily replace the branch un-map line `b.curve = (stuv, np.asarray(xyz, ...) * k + c)` with `b.curve = (stuv, np.asarray(xyz, dtype=np.float64))` and run one parametrization:

Run: `.venv/bin/python -m pytest "tests/test_bez_ssx5_invariance.py::test_bez_ssx_similarity_invariance[c0-planes]" -x -q`
Expected: FAIL (geometry off by the transform) — proving the guard is non-vacuous. **Revert the mutation immediately** and re-run to green before committing.

- [ ] **Step 4: Commit**

```bash
git add tests/test_bez_ssx5_invariance.py
git commit -m "test(ssx5): similarity-invariance property test locks the P1 defect class (gate 5)"
```

---

### Task 5: Acceptance-gate pins for cases 6 and 11 (kickoff gates 1–2)

**Files:**
- Test: `tests/test_nssx5.py` (append at end; reuse its `FIXTURE_DIR`)

- [ ] **Step 1: Write the gate tests**

Append to `tests/test_nssx5.py`:

```python
# ---------------------------------------------------------------------------
# P1 invariance acceptance gates (kickoff 2026-07-20 gates 1-2; design
# 2026-07-21).  Case 6: ~100-unit coords; case 11: ~800-unit part at
# ~3000-unit offset — both must certify at ORIGINAL world coordinates.
# ---------------------------------------------------------------------------

def _load_fixture_pair(num):
    with open(FIXTURE_DIR / f"nurbs_nurbs_intersection_{num}.pkl", "rb") as f:
        return pickle.load(f)[0]


def test_case6_original_coords_complete_at_atol_1e3():
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    s1, s2 = _load_fixture_pair(6)
    r = nurbs_ssx(s1, s2, atol=1e-3)
    assert r['complete'], r['status']['reasons']
    assert r['status']['reasons'] == []
    assert len(r['branches']) == 1
    xyz = np.asarray(r['branches'][0].curve[1], dtype=float)
    # Kickoff engine truth: one x=y-mirror-symmetric arm in the plane z=1
    # from ~[4.37, 75] to ~[75, 4.37] passing through ~[5.47, 5.47].
    assert np.all(np.abs(xyz[:, 2] - 1.0) <= 5e-3)
    lo, hi = (xyz[0], xyz[-1]) if xyz[0][0] < xyz[-1][0] else (xyz[-1], xyz[0])
    assert np.allclose(lo[:2], [4.37, 75.0], atol=1.0)
    assert np.allclose(hi[:2], [75.0, 4.37], atol=1.0)


def test_case11_original_coords_complete_at_atol_0_1():
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    s1, s2 = _load_fixture_pair(11)
    r = nurbs_ssx(s1, s2, atol=0.1)
    assert r['complete'], r['status']['reasons']
    assert r['status']['reasons'] == []
    assert len(r['branches']) == 1


def test_case11_original_coords_certificate_clean_at_atol_1e3():
    # P1 fixes the certificate half; the knob-unreachable tier (P2) may
    # still mark work_budget — trace_unverified specifically must be gone.
    from mmcore.numeric._work_budget import REASON_TRACE_UNVERIFIED
    from mmcore.numeric.intersection.ssx._nssx5 import nurbs_ssx
    s1, s2 = _load_fixture_pair(11)
    r = nurbs_ssx(s1, s2, atol=1e-3)
    assert REASON_TRACE_UNVERIFIED not in r['status']['reasons'], r['status']
```

- [ ] **Step 2: Run the gate tests**

Run: `.venv/bin/python -m pytest tests/test_nssx5.py -q -k "case6_original or case11_original"`
Expected: 3 passed (case 11 at 1e-3 runs a ~20k-cell search — allow ~1–2 minutes). These are the headline acceptance criteria. If case 6 fails: STOP — do not tune anything; reproduce with the kickoff's repro snippet at bez level (decompose + per-pair `bez_ssx`) and debug with superpowers:systematic-debugging using the kickoff's probe tables as the expected-behavior baseline.

- [ ] **Step 3: Run the full adapter suite**

Run: `.venv/bin/python -m pytest tests/test_nssx5.py -q`
Expected: 44 passed (41 existing + 3 new). Two existing pins are marked in-file as legitimate-to-flip WITH an engine improvement (the case-10 fixture expectation and the seam-tangency typed-partial contract) — if either fails, read its in-file comment and update it per that comment in THIS commit, recording the flip in the commit message. If TANGENTIAL semantics changed: STOP and re-scope (kickoff invariant).

- [ ] **Step 4: Commit**

```bash
git add tests/test_nssx5.py
git commit -m "test(nssx5): P1 acceptance gates — cases 6 and 11 certify at original coordinates"
```

---

### Task 6: Full regression floor + coverage harnesses + CASE_NOTES

**Files:**
- Modify: `examples/ssx/nurbs_ssx5_coverage_check.py:71-80` (CASE_NOTES)

- [ ] **Step 1: Run the three regression suites (kickoff gate 4)**

```bash
.venv/bin/python -m pytest tests/test_bez_ssx5_singular.py tests/test_nssx5.py -q
.venv/bin/python -m pytest tests/test_bez_csx4.py tests/test_bez_ccx4.py tests/test_bez_ccx3_cases.py tests/test_bezier_common.py tests/test_bezier_curves_overlap.py -q
```

Expected: 115 + 44 pass; 95 pass (the ccx/csx suites are untouched by this change — any failure there is a real regression, STOP and debug).

- [ ] **Step 2: Run the bez-level coverage harness**

Run: `.venv/bin/python examples/ssx/bez_ssx5_coverage_check.py`
Expected: 100% coverage on all its cases (kickoff gate 4). Timings may shift (the canonical frame changes every run's float trajectory — case-6-class cases got FASTER in the probes); coverage must not.

- [ ] **Step 3: Run the NURBS-level harness (kickoff gates 1 and 3)**

```bash
.venv/bin/python examples/ssx/nurbs_ssx5_coverage_check.py 6
.venv/bin/python examples/ssx/nurbs_ssx5_coverage_check.py
```

Expected: case 6 → OK at 100% (non-vacuous); full run → the 8 previously-OK rows stay OK, case 6 flips to OK (target: 9 OK), case 11 improved (certificate reasons gone; `work_budget` may remain until P2).

- [ ] **Step 4: Update CASE_NOTES**

In `examples/ssx/nurbs_ssx5_coverage_check.py`, `CASE_NOTES` currently reads (entries 6 and 11):

```python
    6: ("typed partial: bez-level trace continuation loses an SSI arm; "
        "honestly reported complete=False"),
    11: ("far-coordinate geometry; work_budget at candidate-scaled "
         "defaults — see budget probe in the Task-7 report"),
```

Replace with the measured post-fix truth, e.g.:

```python
    6: ("OK since the P1 canonical-frame fix (2026-07-21): certifies the "
        "complete mirror-symmetric curve at original coords, atol=1e-3"),
    11: ("P1 fixed the trace certificate at original coords; work_budget "
         "remains from the knob-unreachable internal tier (P2, open)"),
```

Adjust the wording to what the harness ACTUALLY printed in Step 3 — notes must record measurements, not intentions.

- [ ] **Step 5: Commit**

```bash
git add examples/ssx/nurbs_ssx5_coverage_check.py
git commit -m "chore(ssx5): update coverage-harness CASE_NOTES for post-P1 cases 6/11"
```

---

### Task 7: Contract docstrings (design §4)

**Files:**
- Modify: `mmcore/numeric/intersection/ssx/_bez_ssx5.py:1958-1970` (`_strict_ssx_root_tol` docstring), `:5939+` (`bez_ssx` docstring)

- [ ] **Step 1: Update `_strict_ssx_root_tol`'s docstring**

Current (one line): `"""Translation-invariant roundoff scale for an exact Psi zero."""`

Replace with (arithmetic untouched):

```python
    """Roundoff scale for an exact Psi zero, valid in the canonical frame.

    PRECONDITION (P1, 2026-07-21): callers pass control nets that the
    bez_ssx preamble has centered and power-of-two scaled, so coordinate
    MAGNITUDE ~ model EXTENT ~ O(1).  The envelope scales with the extent
    (`diag`); the residual arithmetic it budgets rounds off with the
    magnitude.  In the canonical frame the two coincide by construction —
    the 2026-07-20 case-6/11 diagnosis (extent-scaled budget vs
    magnitude-scaled noise on off-origin models) is structurally
    impossible here rather than accidentally avoided.  Do not call this
    on raw world-frame nets.
    """
```

- [ ] **Step 2: Document the frame in `bez_ssx`'s docstring**

Append to the end of the `bez_ssx` docstring (before the closing `"""`):

```
    Numerical frame (P1, 2026-07-21 design): the search runs in a canonical
    frame — both nets jointly centered and scaled by a power of two — so
    the strict Psi-zero certificates and fixed corrector tolerances see
    O(1) coordinates for any world placement.  The contract stays
    world-in/world-out: xyz outputs are un-mapped exactly once at exit;
    parameters and weights are frame-invariant; `atol`/`max_xyz_step`
    scale with the frame internally.  Results are similarity-invariant:
    bez_ssx(S*k + c, atol*k) is equivalent to bez_ssx(S, atol) (see
    tests/test_bez_ssx5_invariance.py).
```

- [ ] **Step 3: Verify nothing broke and commit**

Run: `.venv/bin/python -m pytest tests/test_bez_ssx5_invariance.py -q`
Expected: all pass (docstring-only change).

```bash
git add mmcore/numeric/intersection/ssx/_bez_ssx5.py
git commit -m "docs(ssx5): canonical-frame contract notes on bez_ssx and _strict_ssx_root_tol"
```

---

### Task 8 (optional, own commit): remove stray debug prints

The kickoff notes stray debug prints leaking from bez/csx internals (`17 30`, `5 6`, `215 218` on harness cases 1–2) — cosmetic, approved for removal "as its own commit".

- [ ] **Step 1: Locate them**

Run: `.venv/bin/python examples/ssx/nurbs_ssx5_coverage_check.py 1 2>&1 | head -20` and trace each stray line to its `print(` site (grep the printed shape, e.g. `grep -rn "print(" mmcore/numeric/intersection/ssx/ mmcore/numeric/intersection/csx/ | grep -v "#"`).

- [ ] **Step 2: Delete only bare debug `print(` calls** — nothing that writes through a logger or is part of a harness/CLI output path.

- [ ] **Step 3: Verify + commit**

Run: `.venv/bin/python -m pytest tests/test_bez_ssx5_singular.py -q` (115 passed) and re-run the harness case 1 (stray lines gone, coverage unchanged).

```bash
git add -u
git commit -m "chore(ssx): drop stray debug prints from bez/csx internals"
```

---

## After the plan: mandatory follow-ups (not tasks, process)

1. **Adversarial review** (house-mandatory per kickoff: two prior rounds each converted "looks correct" into confirmed majors). Reviewer subagents get the READ-ONLY-git rule (`git show` only; old versions into scratchpad — never checkout/restore). Commit all WIP before launching any agent batch.
2. **P2 begins only after P1 lands and reviews clean**: monkeypatch instrumentation per handoff §7 on normalized case 11 at atol=2.5e-6 (pins at 19,798 cells), then scale/expose/re-type is decided WITH the user.
3. Memory + kickoff updates: record P1 outcome, update `MEMORY.md` pointers.
